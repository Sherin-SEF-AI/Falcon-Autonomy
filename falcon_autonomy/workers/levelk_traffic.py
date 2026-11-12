"""Level-k background traffic (runs in CARLA's Python): the vehicles of a traffic profile whose behaviour model is
'levelk' (core.project.traffic_model) are not put on CARLA's traffic manager; this drives them with core.levelk's
per-tick controller instead. docs/levelk.md is the guide.

Made by workers.scene.spawn_traffic (w.levelk_traffic) and stepped by workers.scene.speeds_step after every tick
(placing every vehicle for the next tick in one synchronous batch, workers.common.place_now), so every worker that
spawns traffic — drives, the learning environment, Live, stacks, HIL — drives it without code of its own.

Each vehicle has CARLA's physics off and moves kinematically (the director's kinematic bicycle model: the same in
either motion, so it repeats bit for bit), along its lane path (workers.director.lane_path from its waypoint, the
straight-most continuation through junctions, renewed 40 m before its end) with a lateral offset inside the
carriageway's same-direction lanes (workers.director.lane_bounds). Its parameters are the profile's levelk block by
kind (core.levelk.profile_params: {kind: {k, lambda, alpha, v_des, t_gap, compliance}}; the default level 1). Every
other road user in range — the ego, scenario actors, walkers, CARLA's own traffic — is a level-0 road user for its
predictions (read from the world snapshot of the frame just ticked). Red lights: each traffic light's stop lines
(TrafficLight.get_stop_waypoints) on a vehicle's path are its stops, with the light's state now; compliance below 0.5
ignores them (core.levelk). Brake lights follow the executed deceleration (light_on, which the token observation
reads).

Limits: no junction right of way (crossing paths meet only through the corridors and conflict gaps of core.levelk);
kinematic vehicles cannot be pushed by physics (contacts are counted, not resolved); a vehicle at a dead end stops.
"""
from __future__ import annotations

import math

from ..core import levelk as LK
from .common import ROLE_TRAFFIC, place_now

PATH_M = 150.0
RENEW_M = 40.0


class LevelKTraffic:
    """The level-k background vehicles of a worker (module doc)."""

    def __init__(self, w, world, prof: dict):
        self.w, self.world = w, world
        self.carla = w.carla
        self.map = world.get_map()
        self.table = LK.profile_params(prof)
        cfg = getattr(w, 'cfg', None) or {}
        self.dt = 1.0 / float(cfg.get('fps', 20.0) or 20.0)
        lht = str(cfg.get('handedness', 'RHT')).upper() == 'LHT'
        self.ctl = LK.Controller(seed=int(cfg.get('seed', 0)), dt=self.dt, lht=lht)
        self.rows: dict = {}             # carla id -> {'actor', 'kind', 'type', 'path', 'lk', 's', 'd', 'v', 'z', …}
        self.light_on: dict = {}
        self.stops = None
        self.ego = None                  # the worker's ego actor when it sets it (else found by role hero)
        self.contacts = 0
        self.steps = 0
        self.renewed = 0
        self.stopped_dead_end = set()

    def add(self, a, kind: str, cls=None, label=None, transform=None):
        """Take a just-spawned vehicle: physics off, its lane path from its waypoint. `transform` is where it was
        spawned: a real server reports a just-spawned actor at the origin until the next tick (measured on 0.9.16: every
        level-k vehicle's path was built at the waypoint nearest (0, 0) and they all ended stacked there), so the caller
        passes the spawn point it used; get_transform() is only the fallback."""
        from .director import lane_path
        if cls is not None:
            from ..core import assets as AS
            if not hasattr(self.w, 'subclass'):
                self.w.subclass = {}
            self.w.subclass[a.id] = label or (AS.CLASSES.get(cls) or {}).get('label') or cls
        tr = transform if transform is not None else a.get_transform()
        wp = self.map.get_waypoint(tr.location)
        a.set_simulate_physics(False)
        typ = LK.agent_type(cls or kind)
        bb = a.bounding_box.extent
        params = LK.params_for(self.table, typ)
        if 'v_des' not in (self.table.get(typ) or {}) and 'v_des' not in (self.table.get('*') or {}):
            try:                                              # within CARLA's speed limit for it (km/h)
                lim = float(a.get_speed_limit())
                if math.isfinite(lim) and lim > 0:
                    params['v_des'] = min(params['v_des'], lim / 3.6)
            except Exception:                                 # noqa: BLE001 (an API without it: the type's)
                pass
        row = {'actor': a, 'kind': kind, 'type': typ, 'params': params, 'v': 0.0, 'a': 0.0,
               'length': 2.0 * bb.x, 'width': 2.0 * bb.y,
               'd': 0.0, 'yaw': tr.rotation.yaw, 'x': tr.location.x, 'y': tr.location.y,
               'z': wp.transform.location.z if wp is not None else tr.location.z, 'path': None, 'lk': None}
        self.rows[a.id] = row
        if wp is not None:
            self.new_path(row, wp, lane_path)
        return row

    def new_path(self, row, wp, lane_path):
        p = lane_path(wp, PATH_M)
        row['path'] = p
        from .director import lane_bounds
        row['lk'] = LK.RefPath(p.pts, lane_bounds(self.map, self.carla, p))
        row['s'] = row['lk'].project(row['x'], row['y'])[0]
        row['z'] = p.z
        row['stops'] = self.path_stops(row['lk'])

    def path_stops(self, rp) -> list:
        """[(light, s)] of the traffic lights' stop lines on a path (within its half lane, heading within 30°)."""
        if self.stops is None:
            self.stops = []
            for tl in self.world.get_actors().filter('traffic.traffic_light'):
                try:
                    for wp in tl.get_stop_waypoints():
                        t = wp.transform
                        self.stops.append((tl, t.location.x, t.location.y, t.rotation.yaw, wp.lane_width / 2.0))
                except (RuntimeError, AttributeError):
                    continue
        out = []
        for tl, x, y, yaw, half in self.stops:
            s, d = rp.project(x, y)
            if 0.0 < s < rp.length and abs(d) < half + 0.5 and \
                    abs((float(rp.heading(s)) - yaw + 180.0) % 360.0 - 180.0) < 30.0:
                out.append((tl, s))
        return out

    def ego_actor(self):
        if self.ego is None:
            from .common import ROLE_HERO, role
            for a in self.world.get_actors().filter('vehicle.*'):
                if role(a) == ROLE_HERO:
                    self.ego = a
                    break
        return self.ego

    def step(self, sim_t: float):
        """Decide and place every level-k vehicle for the next tick (module doc)."""
        from .director import lane_path
        if not self.rows:
            return
        snap = self.world.get_snapshot()
        agents, others = [], []
        mine = set(self.rows)
        ego = self.ego_actor()
        for a in self.world.get_actors():
            tid = a.type_id
            if a.id in mine or not tid.startswith(('vehicle.', 'walker.')):
                continue
            sa = snap.find(a.id)
            if sa is None:
                continue
            tr, v = sa.get_transform(), sa.get_velocity()
            try:
                bb = a.bounding_box.extent
                box = {'length': 2.0 * bb.x, 'width': 2.0 * bb.y}
            except RuntimeError:
                box = {}
            others.append({**box, 'id': f'o{a.id}', 'type': 'pedestrian' if tid.startswith('walker.') else 'car',
                           'path': LK.RefPath.line(tr.location.x, tr.location.y, tr.rotation.yaw, 120.0, 10.0),
                           's': 10.0, 'd': 0.0, 'v': math.hypot(v.x, v.y), 'controlled': False,
                           'priority': ego is not None and a.id == ego.id})
        for aid, r in self.rows.items():
            if r['lk'] is None:
                continue
            if r['s'] > r['lk'].length - RENEW_M:
                wp = self.map.get_waypoint(self.carla.Location(x=r['x'], y=r['y'], z=r['z']))
                if wp is not None:
                    d_keep = r['d']
                    self.new_path(r, wp, lane_path)
                    r['d'] = d_keep
                    self.renewed += 1
            stops = [(s, str(tl.get_state())) for tl, s in r.get('stops') or ()]
            agents.append({'id': aid, 'type': r['type'], 'path': r['lk'], 's': r['s'], 'd': r['d'], 'v': r['v'],
                           'params': r['params'], 'controlled': True, 'stops': stops, 'length': r['length'],
                           'width': r['width']})
        acts = self.ctl.step(sim_t, agents + others)
        moves = []
        carla = self.carla
        for aid, r in self.rows.items():
            act = acts.get(aid)
            if act is None or r['lk'] is None:
                continue
            ag = {'type': r['type'], 's': r['s'], 'd': r['d'], 'v': r['v'], 'a': r['a']}
            LK.execute(ag, act['plan_accel'], act['lateral'], self.dt)
            if ag['s'] >= r['lk'].length - 0.5 and self.map.get_waypoint(
                    carla.Location(x=r['x'], y=r['y'], z=r['z'])) is None:
                ag['v'], ag['a'] = 0.0, 0.0
                self.stopped_dead_end.add(aid)
            r.update(s=ag['s'], d=ag['d'], v=ag['v'], a=ag['a'])
            x, y, yaw = (float(q) for q in r['lk'].at(r['s'], r['d']))
            r['x'], r['y'], r['yaw'] = x, y, yaw
            moves.append((r['actor'], carla.Transform(carla.Location(x=x, y=y, z=r['z']), carla.Rotation(yaw=yaw))))
            on = bool(act['brake_light'])
            if on != self.light_on.get(aid, False):
                self.set_light(r['actor'], on)
            self.light_on[aid] = on
        place_now(self.w.client, carla, moves)
        self.steps += 1

    def set_light(self, actor, on):
        from ..core.tokenfeatures import with_brake
        try:
            cur = actor.get_light_state()
            bits = with_brake(cur, on)
            try:
                state = self.carla.VehicleLightState(bits)
            except TypeError:
                state = bits
            actor.set_light_state(state)
        except (AttributeError, RuntimeError):
            pass

    def k_vel(self, actor):
        """The velocity of a level-k vehicle (CARLA reports a physics-off actor as still), or None."""
        r = self.rows.get(actor.id)
        if r is None:
            return None
        y = math.radians(r['yaw'])
        return self.carla.Vector3D(r['v'] * math.cos(y), r['v'] * math.sin(y), 0.0)

    def report(self) -> dict:
        by = {}
        for r in self.rows.values():
            by[r['type']] = by.get(r['type'], 0) + 1
        return {'model': 'levelk', 'vehicles': len(self.rows), 'by_type': by, 'decisions': self.ctl.decisions,
                'steps': self.steps, 'paths_renewed': self.renewed, 'stopped_at_dead_end': len(self.stopped_dead_end),
                'parameters': {k: v for k, v in self.table.items()}, 'role': ROLE_TRAFFIC}
