"""replay: a log (core.logreplay) re-simulated in CARLA, recorded like a drive (runs in CARLA's Python).

    python -m falcon_autonomy.workers replay CONFIG.json

CONFIG.json is a drive config (falcon.drive_config/1: the world, weather, rig, seed, step) with a "replay" block
holding the replay plan (core.logreplay.plan, falcon.replay_plan/1) — so the run folder's drive.json is a complete
input and `falcon reproduce` re-runs it through the drive worker, which hands such a config to LogDirector. The run is
the drive's (workers.drive): synchronous mode at the plan's step, the rig's cameras with every label, lidar, radar,
GNSS, IMU, frames.jsonl, actors.jsonl, lanes.json, dataset.json, run.json — so every downstream tool works on
replays — with the log's actors in place of traffic.

LogDirector (the drive's director interface, as workers.director's):

1. Placement. 'anchored' plans are moved rigidly onto their anchor (a spawn point of the map, by index, or a pose);
   'matched' and 'geo' plans are used as they are. The match of the ego's path with the server's driving lanes
   (Map.generate_waypoints every 2 m) is measured (core.logreplay.mapmatch) and recorded; a 'matched' or 'geo' plan
   whose verdict is 'none' is refused with the numbers unless the plan allows it.
2. Actors. Each log actor is spawned once (its own blueprint when the log names one the server has, else one of its
   class, chosen with the seed), parked 3 km away while it is not observed, and otherwise placed at its planned pose
   with CARLA's physics off (kinematic: repeatable bit for bit) — every actor and the ego in one synchronous batch
   before each tick (workers.common.place_now: set_transform is asynchronous and could land a tick late). A plan's
   actor position is its box's centre: the actor's origin is put at centre − R·(its blueprint's box centre), so the
   replayed box is centred where the logged one was; its size is the blueprint's (CARLA cannot scale a mesh) —
   replay.json lists the size of each against the log's. Heights: 'matched' plans keep the log's; others put each box on the map's ground under it
   (the nearest driving lane's height). Parked actors are left out of actors.jsonl.
3. The ego. 'log': placed at its logged pose every step (its control is unknown: frames.jsonl control is null).
   'follower' (the built-in lane follower, plugins.lane_follower) or 'plugin' (a policy plug-in): driven along the
   logged path, asked for the logged ego's speed at the same moment of the log (a logged stop is a stop; it goes
   again when the log did) but never more than lets it stop at the path's end at 3 m/s², braking for what is in its
   corridor (any replayed actor within 1.8 m of the path ahead, or that will be within 2 s at its present velocity:
   a pedestrian about to cross — heuristic), integrated with the director's kinematic bicycle model (3 m/s² full
   throttle, 8 m/s² full brake, 70° wheel angle).
4. Records. trajectories.json (every step's box centre, heading, velocity and footprint of the ego and each observed
   actor: core.criticality's input) and run.json "scenario" (criticality and the no-collision check), and
   replay.json: the placement, the measured map match, the changes, the actors' blueprints and sizes, and for a
   re-driven ego its deviation from the logged path.

Times: step i shows the log at t_start + i / fps; trajectories.json's t is i / fps, the same as frames.jsonl's sim_t.
"""
from __future__ import annotations

import argparse
import json
import math
import os

from ..core import protocol as P
from ..core.provenance import read_json, write_json
from .common import ROLE_PREFIX, Failed, place_now

PARK_OFFSET_M = 3000.0
CORRIDOR_M = 1.8
REDRIVE_DECEL = 3.0           # m/s²: the deceleration a re-driven ego plans to stop at its path's end with
LEAD_HORIZON_S = 2.0          # s: an actor that will be in the ego's corridor within this, at its velocity, is braked for
LEAD_TAUS = (0.0, 0.5, 1.0, 1.5, 2.0)


class _Spot:
    def __init__(self, transform):
        self.transform = transform


class LogDirector:
    """Executes a replay plan inside the drive worker (the director interface of workers.director.Director)."""

    kinematic = True

    def __init__(self, w, world, cfg, lib, rng):
        from ..core.logreplay.plan import PLAN_SCHEMA
        self.w, self.world, self.lib, self.rng = w, world, lib, rng
        self.carla = w.carla
        self.cfg = cfg
        self.moves: list = []                             # (actor, transform) to place before the next tick
        self.plan = cfg['replay']['plan']
        if self.plan.get('schema') != PLAN_SCHEMA:
            raise Failed(f'not a replay plan (schema {self.plan.get("schema")!r})')
        self.n = int(self.plan['steps'])
        self.dt = 1.0 / float(self.plan['fps'])
        if abs(self.dt - 1.0 / float(cfg.get('fps', 20.0))) > 1e-9:
            raise Failed('the replay plan\'s step is not the drive\'s')
        self.scn = {'id': 'replay', 'name': f'replay of {self.plan["log"].get("name")}',
                    'duration_s': self.n * self.dt, 'pass': {'no_collision': True}, 'values': {}}
        self.map = world.get_map()
        self.traj = {'ego': []}
        self.events, self.fired = [], []
        self.ego = None
        self.ego_control = None
        self.control_unknown = self.plan['ego']['driver'] == 'log'
        self.actors = {}                 # plan id -> {'actor', 'item', 'loc_off', 'active', 'tf', 'vel'}
        self.by_carla = {}
        self.i = 0
        self.t = 0.0
        self.report_doc = {}
        self._ground_cache = {}
        self.snap = None

    # ---------------------------------------------------------------- placement
    def place_ego(self, spawns, order):
        from ..core.logreplay.plan import rigid_anchor
        pl = self.plan['placement']
        carla = self.carla
        if pl['mode'] == 'anchored':
            a = pl.get('anchor') or {'spawn_index': 0}
            if 'spawn_index' in a:
                k = int(a['spawn_index'])
                if not spawns or not 0 <= k < len(spawns):
                    raise Failed(f'the anchor spawn point {k} does not exist (the map has {len(spawns)})')
                tf = spawns[k]
                ax, ay, ayaw = tf.location.x, tf.location.y, tf.rotation.yaw
            else:
                ax, ay, ayaw = float(a['x']), float(a['y']), float(a.get('yaw', 0.0))
            self.report_doc['anchor'] = rigid_anchor(self.plan, ax, ay, ayaw)
        e = self.plan['ego']
        self.park0 = (e['x'][0] + PARK_OFFSET_M, e['y'][0] + PARK_OFFSET_M)
        loc = carla.Location(x=self.park0[0] - 20.0, y=self.park0[1], z=(e['z'][0] or 0.0))
        return None, _Spot(carla.Transform(loc, carla.Rotation(yaw=0.0)))

    def _match(self):
        import numpy as np
        from ..core.logreplay import mapmatch as MM
        lanes = []
        Driving = self.carla.LaneType.Driving
        for wp in self.map.generate_waypoints(2.0):
            if wp.lane_type != Driving:
                continue
            t = wp.transform
            lanes.append((t.location.x, t.location.y, t.rotation.yaw, wp.lane_width))
        self.w.alive()
        e = self.plan['ego']
        res = MM.match(e['x'], e['y'], e['yaw'], np.asarray(lanes, float).reshape(-1, 4),
                       'the server\'s map (Map.generate_waypoints every 2 m)')
        pl = self.plan['placement']
        res['road_matches'] = pl['mode'] in ('matched', 'geo') and res['verdict'] != 'none'
        self.report_doc['map_match'] = res
        if pl['mode'] in ('matched', 'geo') and res['verdict'] == 'none' and not pl.get('allow_mismatch'):
            raise Failed(f'the log does not match this world\'s roads ({res["why"]}): refused — replay it without a '
                         'matching road (an anchored placement), or allow the mismatch')
        if pl['mode'] == 'anchored':
            self.w.note(f'{pl["why"]} — the log is replayed on an anchor of this map; measured there: '
                        f'{res["why"]}')
        elif res['verdict'] == 'partial':
            self.w.note(f'the log matches this world\'s roads only partly: {res["why"]}')

    def ground(self, x, y):
        k = (round(x / 2.0), round(y / 2.0))
        z = self._ground_cache.get(k)
        if z is None:
            wp = self.map.get_waypoint(self.carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True,
                                       lane_type=self.carla.LaneType.Driving)
            z = self._ground_cache[k] = float(wp.transform.location.z) if wp is not None else 0.0
        return z

    def blueprint(self, item):
        hint = item.get('blueprint')
        if not hasattr(self, '_bp_ids'):
            self._bp_ids = {b.id for b in self.lib.filter('*')}
        ids = self._bp_ids
        if hint and hint in ids and hint.startswith(('vehicle.', 'walker.')):
            return self.lib.find(hint), 'the log\'s'
        cls = item['class']
        if cls == 'pedestrian':
            bps = sorted(self.lib.filter('walker.pedestrian.*'), key=lambda b: b.id)
            bp = bps[int(self.rng.integers(len(bps)))]
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            return bp, 'of its class (seeded)'
        from .director import KIND_BLUEPRINTS
        cands = [c for c in KIND_BLUEPRINTS.get(cls, KIND_BLUEPRINTS['car']) if c in ids]
        if not cands:
            from .scene import blueprints_by_type
            pool = sorted(blueprints_by_type(self.lib).get(cls, []) or blueprints_by_type(self.lib)['car'],
                          key=lambda b: b.id)
            return pool[int(self.rng.integers(len(pool)))], 'of its class (seeded)'
        return self.lib.find(cands[int(self.rng.integers(len(cands)))]), 'of its class (seeded)'

    def setup(self, ego, ego_wp):
        carla = self.carla
        self._match()
        self.ego = ego
        ego.set_simulate_physics(False)
        self.ego_bb = ego.bounding_box
        self.by_carla[ego.id] = 'ego'
        e = self.plan['ego']
        self.z_mode = self.plan['placement'].get('z', 'log')
        self.ego_state = None
        sizes = []
        for k, item in enumerate(self.plan['actors']):
            bp, how = self.blueprint(item)
            bp.set_attribute('role_name', ROLE_PREFIX + 'replay')
            park = carla.Transform(carla.Location(x=self.park0[0] + 8.0 * k, y=self.park0[1], z=(e['z'][0] or 0.0)
                                                  + 1.0), carla.Rotation())
            actor = self.world.try_spawn_actor(bp, park)
            if actor is None:
                raise Failed(f'actor {item["id"]} ({bp.id}) could not be spawned')
            self.w.own(actor)
            actor.set_simulate_physics(False)
            bb = actor.bounding_box
            self.actors[item['id']] = {'actor': actor, 'item': item, 'bb': bb, 'park': park, 'active': False,
                                       'tf': None, 'vel': None, 'walker': item['class'] == 'pedestrian'}
            self.by_carla[actor.id] = item['id']
            sizes.append({'id': item['id'], 'class': item['class'], 'carla_id': actor.id, 'blueprint': bp.id,
                          'blueprint_choice': how, 'log_size': item['size'], 'log_size_source': item['size_source'],
                          'replay_size': [round(2 * bb.extent.x, 3), round(2 * bb.extent.y, 3),
                                          round(2 * bb.extent.z, 3)]})
        self.w.alive()
        self.report_doc['actors'] = sizes
        bp = self.lib.find('sensor.other.collision')
        bp.set_attribute('role_name', ROLE_PREFIX + 'sensor')
        self.col = self.w.own(self.world.spawn_actor(bp, carla.Transform(), attach_to=ego))
        # the sensor names CARLA ids; trajectories.json names the log's ids: the events are given the log's
        self.col.listen(lambda ev: self.events.append((
            round(self.i * self.dt, 4), self.by_carla.get(int(ev.other_actor.id), int(ev.other_actor.id))
            if ev.other_actor else 0, float(math.sqrt(ev.normal_impulse.x ** 2 + ev.normal_impulse.y ** 2 +
                                                      ev.normal_impulse.z ** 2)))))
        drv = e['driver']
        if drv in ('follower', 'plugin'):
            from ..plugins import BUILTIN_POLICIES, load
            spec = BUILTIN_POLICIES['lane_follower'] if drv == 'follower' else e['policy']
            vmax = max([v for v in e['path']['v'] if v is not None] + [1.0])
            params = {'cruise_kmh': vmax * 3.6 + 1.0} if drv == 'follower' else {}
            self.policy = load(spec, **params)
            self.policy.reset(int(self.plan.get('seed', 0)))
            self.report_doc['policy'] = {'driver': drv, 'spec': spec, 'name': getattr(self.policy, 'name', spec)}
        self.place(0)

    # ---------------------------------------------------------------- per step
    def _ego_pose(self, i):
        e = self.plan['ego']
        z = e['z'][i] if self.z_mode == 'log' else self.ground(e['x'][i], e['y'][i])
        return e['x'][i], e['y'][i], z, e['yaw'][i], e['pitch'][i] or 0.0, e['roll'][i] or 0.0

    def place(self, i):
        """Every actor at plan step i (and the ego, when it replays its log)."""
        carla = self.carla
        self.i = i
        for aid, st in self.actors.items():
            item = st['item']
            k = i - item['i0']
            x = item['x'][k] if 0 <= k < len(item['x']) else None
            if x is None:
                if st['active']:
                    self.moves.append((st['actor'], st['park']))
                    st['active'], st['tf'], st['vel'] = False, None, None
                continue
            y, yaw = item['y'][k], item['yaw'][k]
            pitch, roll = item['pitch'][k] or 0.0, item['roll'][k] or 0.0
            bb = st['bb']
            if self.z_mode == 'log':
                cz = item['z'][k]
            else:
                cz = self.ground(x, y) + bb.extent.z
            # the actor's origin so that its box is centred on the planned centre: centre − R·(box centre)
            from ..core.rig import carla_rotation
            import numpy as np
            off = carla_rotation(roll, pitch, yaw) @ np.array([bb.location.x, bb.location.y, bb.location.z])
            tf = carla.Transform(carla.Location(x=x - off[0], y=y - off[1], z=cz - off[2]),
                                 carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
            self.moves.append((st['actor'], tf))
            st['active'], st['tf'] = True, tf
            st['vel'] = carla.Vector3D(item['vx'][k] or 0.0, item['vy'][k] or 0.0, 0.0)
            st['centre'] = (x, y, yaw)
        if self.plan['ego']['driver'] == 'log' or i == 0:
            x, y, z, yaw, pitch, roll = self._ego_pose(i)
            self.ego_tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw,
                                                                                        roll=roll))
            e = self.plan['ego']
            self.ego_v = (e['vx'][i] or 0.0, e['vy'][i] or 0.0)
            self.moves.append((self.ego, self.ego_tf))
            if i == 0:
                self.ego_state = [x, y, z, yaw, math.hypot(*self.ego_v)]
                self.path_k = 0
        moves, self.moves = self.moves, []
        place_now(self.w.client, carla, moves)            # applied before the tick (set_transform is asynchronous)

    def sync(self, frame, timeout=5.0):
        import time
        t0 = time.monotonic()
        while True:
            snap = self.world.get_snapshot()
            if snap.frame >= frame:
                self.snap = snap
                return
            if time.monotonic() - t0 > timeout:
                raise Failed(f'the snapshot of frame {frame} did not arrive within {timeout:g} s')
            time.sleep(0.0005)

    def step(self, t):
        i = int(round(t / self.dt))
        self.t = t
        if i == 0:
            return                                   # step 0 was placed at setup
        prev_actors = {aid: (st['active'], st.get('centre'), st['vel']) for aid, st in self.actors.items()}
        if self.plan['ego']['driver'] != 'log':
            self.drive_ego(prev_actors)
        self.place(i)

    def drive_ego(self, prev):
        """One step of the re-driven ego from its state (the previous step's), seeing the actors as they were."""
        carla = self.carla
        x, y, z, yaw, v = self.ego_state
        path = self.plan['ego']['path']
        px, py, pv = path['x'], path['y'], path['v']
        n = len(px)
        k0 = self.path_k
        best, bk = None, k0
        for k in range(max(k0 - 5, 0), min(k0 + 60, n)):
            d = (px[k] - x) ** 2 + (py[k] - y) ** 2
            if best is None or d < best:
                best, bk = d, k
        self.path_k = bk
        # the speed asked for: the logged ego's speed at this moment of the log (so a logged stop is a stop, and the
        # ego goes again when the log did), and never more than lets it stop at the path's end (REDRIVE_DECEL)
        e = self.plan['ego']
        k_now = min(self.i, len(e['vx']) - 1)          # self.i: the step just shown (this one integrates to i + 1)
        v_log = math.hypot(e['vx'][k_now] or 0.0, e['vy'][k_now] or 0.0)
        left_m = max(n - 1 - bk, 0) * float(path.get('step_m', 1.0))
        limit = max(min(v_log, math.sqrt(2.0 * REDRIVE_DECEL * left_m)), 1e-3)   # > 0: the follower reads 0 as none
        route = [[px[k], py[k], limit] for k in range(bk + 2, min(bk + 52, n), 2)]
        if not route and n:
            route = [[px[-1], py[-1], 1e-3]]
        lead = self.lead(route, x, y, prev)
        obs = {'t': self.t, 'dt': self.dt, 'ego': {'x': x, 'y': y, 'yaw': yaw, 'speed_mps': v}, 'route': route,
               'lead': lead}
        u = self.policy.act(obs)
        self.ego_control = dict(u)
        if u.get('hand_brake'):
            v = 0.0
        else:
            v = max(v + (3.0 * float(u.get('throttle', 0.0)) - 8.0 * float(u.get('brake', 0.0))) * self.dt, 0.0)
        wheelbase = max(2.0 * self.ego_bb.extent.x * 0.6, 0.8)
        yaw2 = yaw + math.degrees(v * math.tan(float(u.get('steer', 0.0)) * math.radians(70.0)) / wheelbase * self.dt)
        mid = math.radians((yaw + yaw2) / 2.0)
        x, y, yaw = x + math.cos(mid) * v * self.dt, y + math.sin(mid) * v * self.dt, yaw2
        z = self.ground(x, y)
        self.ego_state = [x, y, z, yaw, v]
        self.ego_tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
        self.ego_v = (v * math.cos(math.radians(yaw)), v * math.sin(math.radians(yaw)))
        self.moves.append((self.ego, self.ego_tf))       # placed with the actors in place()

    def lead(self, route, ex, ey, prev):
        """What the re-driven ego brakes for: the nearest replayed actor that is within CORRIDOR_M of the route ahead,
        or will be within LEAD_HORIZON_S at its present velocity (a pedestrian about to cross) — the gap bumper to
        bumper along the route to its (predicted) place, and its speed along the route (0 for one not yet in it, or
        crossing it)."""
        if not route:
            return None
        best = None
        ext = self.ego_bb.extent.x
        cum, acc, last = [], 0.0, (ex, ey)
        for p in route:
            acc += math.hypot(p[0] - last[0], p[1] - last[1])
            cum.append(acc)
            last = p
        for aid, (active, c, vel) in prev.items():
            if not active or c is None:
                continue
            vx, vy = (vel.x, vel.y) if vel is not None else (0.0, 0.0)
            for tau in LEAD_TAUS:
                qx, qy = c[0] + vx * tau, c[1] + vy * tau
                k = next((j for j, p in enumerate(route) if math.hypot(qx - p[0], qy - p[1]) < CORRIDOR_M), None)
                if k is None:
                    continue
                p = route[k]
                o = route[k - 1] if k else (ex, ey)
                hx, hy = p[0] - o[0], p[1] - o[1]
                hn = math.hypot(hx, hy) or 1.0
                hx, hy = hx / hn, hy / hn
                along = cum[k] + (qx - p[0]) * hx + (qy - p[1]) * hy        # its projection on the route
                d = max(along - ext - self.actors[aid]['bb'].extent.x, 0.0)
                sp = max(vx * hx + vy * hy, 0.0) if tau == 0.0 else 0.0
                if best is None or d < best[0]:
                    best = (d, sp)
                break
        return None if best is None else {'distance_m': best[0], 'speed_mps': best[1]}

    # ---------------------------------------------------------------- what the drive asks
    def k_vel(self, actor):
        if self.ego is not None and actor.id == self.ego.id:
            return self.carla.Vector3D(self.ego_v[0], self.ego_v[1], 0.0)
        aid = self.by_carla.get(actor.id)
        st = self.actors.get(aid) if aid is not None else None
        return st['vel'] if st is not None and st['active'] else None

    def exact_tf(self, actor):
        """The pose this replay set (CARLA's snapshot of a set_transform varies by a hair between runs)."""
        if self.ego is not None and actor.id == self.ego.id:
            return self.ego_tf
        aid = self.by_carla.get(actor.id)
        st = self.actors.get(aid) if aid is not None else None
        return st['tf'] if st is not None and st['active'] else None

    def hidden_ids(self) -> set:
        return {st['actor'].id for st in self.actors.values() if not st['active']}

    def record(self, _t):
        import numpy as np
        from ..core.rig import carla_rotation
        t = round(self.i * self.dt, 6)
        tf = self.ego_tf
        bb = self.ego_bb
        R = carla_rotation(tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw)
        c = R @ np.array([bb.location.x, bb.location.y, bb.location.z])
        self.traj['ego'].append((t, round(tf.location.x + c[0], 6), round(tf.location.y + c[1], 6),
                                 round(tf.rotation.yaw, 6), round(self.ego_v[0], 6), round(self.ego_v[1], 6),
                                 round(2 * bb.extent.x, 4), round(2 * bb.extent.y, 4)))
        for aid, st in self.actors.items():
            if not st['active']:
                continue
            x, y, yaw = st['centre']
            b = st['bb']
            self.traj.setdefault(aid, []).append((t, x, y, yaw, round(st['vel'].x, 6), round(st['vel'].y, 6),
                                                  round(2 * b.extent.x, 4), round(2 * b.extent.y, 4)))

    def report(self) -> dict:
        import numpy as np
        e = self.plan['ego']
        doc = dict(self.report_doc)
        if e['driver'] != 'log':
            ex, ey = np.array(e['x'], float), np.array(e['y'], float)
            got = np.array([(r[1], r[2]) for r in self.traj['ego']], float)
            px, py = np.array(e['path']['x'], float), np.array(e['path']['y'], float)
            dev = [float(np.min(np.hypot(px - gx, py - gy))) for gx, gy in got] if len(px) else []
            doc['ego_vs_log'] = {
                'path_deviation_m': {'mean': round(float(np.mean(dev)), 4), 'max': round(float(np.max(dev)), 4)}
                if dev else None,
                'final_gap_to_logged_position_m': round(float(np.hypot(got[-1, 0] - ex[len(got) - 1],
                                                                       got[-1, 1] - ey[len(got) - 1])), 4)
                if len(got) else None}
        doc.update({'placement': self.plan['placement'], 'changes': self.plan['changes'],
                    'weather_change': self.plan.get('weather'), 'log': self.plan['log'],
                    'ego': {'driver': e['driver'], 'policy': e.get('policy'), 'offset_m': e.get('offset_m', 0.0),
                            'origin': self.plan['log'].get('ego_origin'),
                            'size_log': e.get('size'), 'size_replay': [round(2 * self.ego_bb.extent.x, 3),
                                                                       round(2 * self.ego_bb.extent.y, 3),
                                                                       round(2 * self.ego_bb.extent.z, 3)]},
                    'steps': self.n, 'fps': self.plan['fps'], 't_start': self.plan['t_start']})
        if self.w.out_dir:
            write_json(os.path.join(self.w.out_dir, 'replay.json'), dict({'schema': 'falcon.replay/1'}, **doc))
        return {'id': 'replay', 'name': self.scn['name'], 'kind': 'replay', 'values': {}, 'motion': 'kinematic',
                'seed': self.plan.get('seed'), 'fired': [], 'collisions': self.events,
                'handedness': self.cfg.get('handedness', 'RHT'),
                'replay': {'placement': self.plan['placement']['mode'],
                           'road_matches': (doc.get('map_match') or {}).get('road_matches'),
                           'map_match': {k: (doc.get('map_match') or {}).get(k) for k in
                                         ('verdict', 'why', 'share_on_lane_in_direction', 'p95_lateral_m')},
                           'driver': e['driver'], 'changes': len(self.plan['changes'])}}


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker replay')
    ap.add_argument('config', help='a drive config with a "replay" block (core.logreplay)')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    if not cfg.get('replay'):
        print(f'failed: {a.config} has no "replay" block', flush=True)
        return P.FAILED
    from .drive import Drive

    class Replay(Drive):
        name = 'replay'
    return Replay(cfg, argv, a.config).execute()


if __name__ == '__main__':                      # pragma: no cover
    import sys
    sys.exit(main(sys.argv[1:]))
