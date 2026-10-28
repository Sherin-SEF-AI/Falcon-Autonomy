"""The scenario director: runs a concrete scenario (core.scenarios) inside a drive — places the ego and the actors,
drives them, fires the timeline, records trajectories for the criticality measures (runs in CARLA's Python).

Placement. The ego starts at a spawn point (seeded order) whose lane runs straight for the scenario's length (yaw
within 10° over 150 m) and has what the scenario needs: "adjacent_same" — a driving lane beside it in the same
direction on the overtaking side; "opposite" — a lane of the other direction; "junction_turn" — a junction within 60 m
with a branch turning across the oncoming lanes. Lanes are relative to the traffic's handedness (cfg 'handedness'):
in right-hand traffic the overtaking side and the oncoming lanes are on the left and the kerb on the right; in
left-hand traffic the other way round. A scenario no start of the map can host is refused, saying what was missing.

Control. Every scripted actor is driven through CARLA's physics, deterministically in synchronous mode: vehicles by
pure-pursuit steering along their path and a speed controller (its set speed ramps at the action's acceleration or
deceleration), walkers by WalkerControl. A lane change moves the path's lateral offset linearly over its duration.
Moving actors and the ego start at their speed (set_target_velocity on the first step). The ego is driven by the
built-in lane follower along its route (driver "follower": it brakes for what is ahead in its corridor), at constant
set speed without braking ("none"), or by CARLA's traffic manager ("autopilot").

Records: per step the ego's and every actor's (t, x, y, yaw, vx, vy, length, width) and the ego's collision events
(sensor.other.collision), for core.criticality; the timeline's firings with their times.
"""
from __future__ import annotations

import math

import numpy as np

from .common import ROLE_PREFIX, Failed

STRAIGHT_M, STRAIGHT_DEG = 150.0, 10.0
STEP_M = 1.0
KIND_BLUEPRINTS = {
    'car': ('vehicle.tesla.model3', 'vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.nissan.micra'),
    'truck': ('vehicle.carlamotors.carlacola', 'vehicle.carlamotors.european_hgv'),
    'van': ('vehicle.mercedes.sprinter', 'vehicle.ford.ambulance', 'vehicle.volkswagen.t2'),
    'bus': ('vehicle.mitsubishi.fusorosa',),
    'motorcycle': ('vehicle.vespa.zx125', 'vehicle.yamaha.yzf', 'vehicle.kawasaki.ninja'),
    'bicycle': ('vehicle.diamondback.century', 'vehicle.bh.crossbike', 'vehicle.gazelle.omafiets'),
}


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class Path:
    """A polyline in CARLA's world (x, y) with its headings, the lane width along it, and a lateral offset function
    (positive: to the lane's right in CARLA's frame)."""

    def __init__(self, pts, yaws, widths, z):
        self.pts = np.asarray(pts, float)
        self.yaw = np.asarray(yaws, float)
        self.w = np.asarray(widths, float)
        self.z = float(z)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1) if len(self.pts) > 1 else np.zeros(0)
        self.s = np.r_[0.0, np.cumsum(seg)]

    def at(self, s, lateral=0.0):
        """(x, y, yaw) at arc length s, shifted `lateral` m to the right."""
        s = float(np.clip(s, 0.0, self.s[-1]))
        k = int(np.clip(np.searchsorted(self.s, s) - 1, 0, max(len(self.s) - 2, 0)))
        L = max(self.s[k + 1] - self.s[k], 1e-9) if len(self.s) > 1 else 1.0
        f = (s - self.s[k]) / L if len(self.s) > 1 else 0.0
        p = self.pts[k] * (1 - f) + self.pts[min(k + 1, len(self.pts) - 1)] * f
        yaw = self.yaw[k]
        a = math.radians(yaw)
        right = np.array([-math.sin(a), math.cos(a)])
        q = p + right * lateral
        return float(q[0]), float(q[1]), float(yaw)

    def project(self, x, y) -> float:
        """Arc length of the nearest point on the path."""
        d = np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y)
        return float(self.s[int(np.argmin(d))])

    def width(self, s):
        return float(self.w[int(np.clip(np.searchsorted(self.s, s), 0, len(self.w) - 1))])


def lane_path(wp, length, backwards=False):
    """The lane from wp, `length` m along its direction (or against it), straight-most through junctions."""
    pts, yaws, ws = [], [], []
    cur = wp
    for _ in range(int(length / STEP_M) + 1):
        t = cur.transform
        yaw = t.rotation.yaw + (180.0 if backwards else 0.0)
        pts.append((t.location.x, t.location.y))
        yaws.append(yaw)
        ws.append(cur.lane_width)
        nxt = cur.previous(STEP_M) if backwards else cur.next(STEP_M)
        if not nxt:
            break
        cur = min(nxt, key=lambda n: abs(_wrap(n.transform.rotation.yaw - cur.transform.rotation.yaw)))
    return Path(pts, yaws, ws, wp.transform.location.z)


class Director:
    def __init__(self, w, world, cfg, lib, rng):
        self.w, self.world, self.lib, self.rng = w, world, lib, rng
        self.carla = w.carla
        self.scn = cfg['scenario']
        self.map = world.get_map()
        self.lht = str(cfg.get('handedness', 'RHT')).upper() == 'LHT'
        self.dt = 1.0 / float(cfg.get('fps', 20.0))
        self.actors = {}                 # id -> {'actor', 'kind', 'path', 's', 'lateral', 'speed', 'target', …}
        self.fired = []
        self.traj = {'ego': []}
        self.events = []
        self.t = 0.0
        self.started = False

    # ------------------------------------------------------------------ lanes relative to the handedness
    def overtaking(self, wp):
        """The lane beside wp in its direction on the overtaking side, or None."""
        n = wp.get_right_lane() if self.lht else wp.get_left_lane()
        if n is not None and n.lane_type == self.carla.LaneType.Driving and (n.lane_id > 0) == (wp.lane_id > 0):
            return n
        return None

    def oncoming(self, wp):
        """The nearest driving lane of the other direction (crossing towards the oncoming side), or None."""
        cur = wp
        for _ in range(6):
            cur = cur.get_right_lane() if self.lht else cur.get_left_lane()
            if cur is None:
                return None
            if cur.lane_type == self.carla.LaneType.Driving and (cur.lane_id > 0) != (wp.lane_id > 0):
                return cur
        return None

    def kerb_sign(self):
        """+1: the kerb is to the lane's right (right-hand traffic), −1: to its left."""
        return -1.0 if self.lht else 1.0

    # ------------------------------------------------------------------ placement
    def place_ego(self, spawns, order):
        needs = set(self.scn.get('needs') or [])
        route = self.scn.get('ego', {}).get('route', 'straight')
        reasons = {}
        for k in list(order):
            wp = self.map.get_waypoint(spawns[k].location)
            if wp is None or wp.is_junction:
                continue
            ok, why = self.suits(wp, needs, route)
            if ok:
                order.remove(k)
                return spawns[k], wp
            reasons[why] = reasons.get(why, 0) + 1
        raise Failed(f'no start on this map can host the scenario "{self.scn["name"]}" '
                     f'({", ".join(f"{n} starts: {r}" for r, n in reasons.items()) or "no spawn points"})')

    def suits(self, wp, needs, route):
        path = lane_path(wp, STRAIGHT_M)
        if route != 'turn_across':
            if path.s[-1] < STRAIGHT_M * 0.8:
                return False, 'the lane ends too soon'
            if np.max(np.abs([_wrap(y - path.yaw[0]) for y in path.yaw])) > STRAIGHT_DEG:
                return False, 'not straight'
        if 'adjacent_same' in needs and self.overtaking(wp) is None:
            return False, 'no lane beside it in the same direction'
        if ('opposite' in needs or route == 'turn_across') and self.oncoming(wp) is None:
            return False, 'no oncoming lane'
        if route == 'turn_across' and self.turn_route(wp) is None:
            return False, 'no junction turn across the traffic within 60 m'
        return True, ''

    def turn_route(self, wp):
        """A route that enters a junction within 60 m and leaves it turning across the oncoming lanes (left in
        right-hand traffic, right in left-hand), or None."""
        cur = wp
        for _ in range(60):
            nxt = cur.next(STEP_M)
            if not nxt:
                return None
            if nxt[0].is_junction:
                want = -1.0 if not self.lht else 1.0      # CARLA yaw: + is to the right
                best = None
                for n in nxt:
                    p = lane_path(n, 40.0)
                    turn = _wrap(p.yaw[-1] - p.yaw[0])
                    if want * turn > 45.0 and (best is None or want * turn > want * best[0]):
                        best = (turn, n)
                if best is None:
                    return None
                approach = lane_path(wp, wp.transform.location.distance(cur.transform.location) + 1.0)
                through = lane_path(best[1], 60.0)
                pts = np.r_[approach.pts, through.pts]
                return Path(pts, np.r_[approach.yaw, through.yaw], np.r_[approach.w, through.w], approach.z)
            cur = nxt[0]
        return None

    def ego_route(self, wp):
        route = self.scn.get('ego', {}).get('route', 'straight')
        return self.turn_route(wp) if route == 'turn_across' else lane_path(wp, 400.0)

    def actor_path(self, a, ego_wp):
        place = a.get('place') or {}
        lane, ahead = place.get('lane', 'ego'), float(place.get('ahead_m', 0.0))
        lateral = float(place.get('lateral_m', 0.0))
        heading = a.get('heading', 'lane')
        base = ego_wp
        if lane == 'adjacent':
            base = self.overtaking(ego_wp)
        elif lane == 'opposite':
            base = self.oncoming(ego_wp)
        if base is None:
            raise Failed(f'actor {a["id"]}: the map has no {lane} lane here')
        at = base.next(ahead)[0] if ahead > 0 else (base.previous(-ahead)[0] if ahead < 0 else base)
        if lane == 'opposite':
            # the oncoming lane: ahead_m is measured along the ego's direction, the actor drives its own lane's way
            at = base.previous(ahead)[0] if ahead > 0 else base
        ks = self.kerb_sign()
        half_w = at.lane_width / 2.0
        off = 0.0
        if lane == 'line':
            off = -ks * half_w + lateral                   # the lane line on the overtaking side
        elif lane == 'kerb':
            size = self.size_of(a['kind'])
            off = ks * (half_w + size[1] / 2.0 + 0.2 - lateral)
        elif lane == 'sidewalk':
            off = ks * (half_w + 2.0) + lateral
        elif lane == 'far_sidewalk':
            off = -ks * (half_w + at.lane_width + 2.0) + lateral
        else:
            off = ks * lateral if heading != 'against' or lane != 'ego' else ks * lateral
        if heading == 'across':
            x, y, yaw = lane_path(at, 1.0).at(0.0, off)
            d = -ks                                        # walks from its side across the road
            a_yaw = yaw + (90.0 if d > 0 else -90.0)
            r = math.radians(a_yaw)
            pts = [(x + math.cos(r) * k, y + math.sin(r) * k) for k in np.arange(0, 30.0, STEP_M)]
            return Path(pts, [a_yaw] * len(pts), [at.lane_width] * len(pts), at.transform.location.z), 0.0
        if heading == 'against' and lane == 'ego':
            return lane_path(at, 200.0, backwards=True), -off          # its right is the ego lane's left
        return lane_path(at, 300.0), off

    def size_of(self, kind):
        return {'bus': (10.3, 3.9), 'truck': (6.0, 2.6), 'van': (5.5, 2.0), 'motorcycle': (2.2, 0.9),
                'bicycle': (1.7, 0.6), 'pedestrian': (0.5, 0.5), 'child': (0.4, 0.4)}.get(kind, (4.6, 2.0))

    def blueprint(self, kind):
        if kind in ('pedestrian', 'child'):
            bps = sorted(self.lib.filter('walker.pedestrian.*'), key=lambda b: b.id)
            if kind == 'child':
                kids = [b for b in bps if b.has_attribute('age') and b.get_attribute('age').as_str() == 'child']
                bps = kids or bps
            bp = bps[int(self.rng.integers(len(bps)))]
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            return bp
        ids = {b.id for b in self.lib.filter('vehicle.*')}
        for c in KIND_BLUEPRINTS.get(kind, KIND_BLUEPRINTS['car']):
            if c in ids:
                return self.lib.find(c)
        from .scene import blueprints_by_type
        pool = sorted(blueprints_by_type(self.lib).get(kind, []) or blueprints_by_type(self.lib)['car'],
                      key=lambda b: b.id)
        return pool[int(self.rng.integers(len(pool)))]

    def setup(self, ego, ego_wp):
        carla = self.carla
        self.ego = ego
        self.ego_path = self.ego_route(ego_wp)
        self.ego_speed = float(self.scn.get('ego', {}).get('speed_kmh', 30.0)) / 3.6
        self.driver = self.scn.get('ego', {}).get('driver', 'follower')
        from ..plugins.lane_follower import LaneFollower
        self.policy = LaneFollower(cruise_kmh=self.ego_speed * 3.6 + 1e-6)
        for a in self.scn.get('actors', []):
            path, lateral = self.actor_path(a, ego_wp)
            x, y, yaw = path.at(0.0, lateral)
            bp = self.blueprint(a['kind'])
            bp.set_attribute('role_name', ROLE_PREFIX + 'scenario')
            tf = carla.Transform(carla.Location(x=x, y=y, z=path.z + (1.0 if a['kind'] in ('pedestrian', 'child')
                                                                       else 0.5)), carla.Rotation(yaw=yaw))
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is None:
                raise Failed(f'actor {a["id"]} ({a["kind"]}) could not be placed: something is in the way')
            self.w.own(actor)
            moving = a.get('start', 'moving') == 'moving'
            sp = float(a.get('speed_kmh', 0.0)) / 3.6 if moving else 0.0
            self.actors[a['id']] = {'actor': actor, 'kind': a['kind'], 'path': path, 'lateral': lateral,
                                    'lat_from': lateral, 'lat_to': lateral, 'lat_t0': 0.0, 'lat_dur': 0.0,
                                    'speed': sp, 'target': sp, 'accel': 3.0, 'walking': False,
                                    'walk_speed': 0.0, 'walker': a['kind'] in ('pedestrian', 'child'),
                                    'heading': a.get('heading', 'lane')}
            self.traj[a['id']] = []
        bp = self.lib.find('sensor.other.collision')
        bp.set_attribute('role_name', ROLE_PREFIX + 'sensor')
        self.col = self.w.own(self.world.spawn_actor(bp, carla.Transform(), attach_to=ego))
        self.col.listen(lambda e: self.events.append((self.t, int(e.other_actor.id) if e.other_actor else 0,
                                                       float(math.sqrt(e.normal_impulse.x ** 2 + e.normal_impulse.y ** 2
                                                                       + e.normal_impulse.z ** 2)))))
        if self.driver == 'autopilot':
            ego.set_autopilot(True, self.w.tm.get_port())

    # ------------------------------------------------------------------ stepping
    def start(self):
        """At the first step: everyone at their speed."""
        carla = self.carla
        self.started = True
        a = math.radians(self.ego.get_transform().rotation.yaw)
        self.ego.set_target_velocity(carla.Vector3D(self.ego_speed * math.cos(a), self.ego_speed * math.sin(a), 0))
        for st in self.actors.values():
            if st['speed'] > 0 and not st['walker']:
                y = math.radians(st['actor'].get_transform().rotation.yaw)
                st['actor'].set_target_velocity(carla.Vector3D(st['speed'] * math.cos(y), st['speed'] * math.sin(y),
                                                               0))

    def step(self, t):
        """Before each tick: fire the timeline, steer everyone. t: seconds since the scenario started."""
        self.t = t
        if not self.started:
            self.start()
        self.timeline(t)
        for aid, st in self.actors.items():
            self.drive_actor(st, t)
        if self.driver in ('follower', 'none'):
            self.drive_ego()

    def record(self, t):
        for aid, a in [('ego', self.ego)] + [(k, v['actor']) for k, v in self.actors.items()]:
            try:
                tr, v, bb = a.get_transform(), a.get_velocity(), a.bounding_box
            except RuntimeError:
                continue
            self.traj[aid].append((round(t, 4), tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y,
                                   2 * bb.extent.x, 2 * bb.extent.y))

    def timeline(self, t):
        for k, ev in enumerate(self.scn.get('timeline', [])):
            if k in {f[0] for f in self.fired}:
                continue
            trig = ev.get('trigger', {})
            go = False
            if trig.get('type') == 'time':
                go = t >= float(trig['t_s'])
            elif trig.get('type') in ('distance', 'ahead'):
                st = self.actors[trig['actor']]
                gap = self.gap_ahead(st['actor'])
                if trig['type'] == 'distance':
                    go = self.dist(st['actor']) < float(trig['lt_m'])
                else:
                    go = gap is not None and gap > float(trig['gt_m'])
            if go:
                self.fired.append((k, round(t, 3)))
                for act in ev.get('actions', []):
                    self.act(act, t)

    def act(self, a, t):
        st = self.actors[a['actor']]
        do = a['do']
        if do == 'speed':
            st['target'] = float(a['kmh']) / 3.6
            st['accel'] = float(a.get('accel', 3.0))
        elif do == 'brake':
            st['target'] = 0.0
            st['accel'] = float(a.get('decel', 6.0))
        elif do == 'stop':
            st['target'], st['accel'] = 0.0, 6.0
        elif do == 'go':
            st['target'] = max(st['target'], 5.0)
        elif do == 'walk':
            st['walking'], st['walk_speed'] = True, float(a.get('speed_mps', 1.4))
        elif do == 'lane_change':
            ks = self.kerb_sign()
            p = st['path']
            s_now = p.project(*_xy(st['actor']))
            to = a.get('to', 'ego')
            here = self.map.get_waypoint(self.ego.get_location())
            # the lateral offset (from the actor's own path) that puts it on the target lane's centre
            ego_s = self.ego_path.project(*_xy(st['actor']))
            ex, ey, _ = self.ego_path.at(ego_s, 0.0)
            px, py, pyaw = p.at(s_now, 0.0)
            r = math.radians(pyaw)
            lat_to_ego = (ex - px) * -math.sin(r) + (ey - py) * math.cos(r)
            target = {'ego': lat_to_ego, 'kerb': lat_to_ego + ks * here.lane_width,
                      'adjacent': lat_to_ego - ks * here.lane_width}.get(to, lat_to_ego)
            st.update(lat_from=st['lateral'], lat_to=target, lat_t0=t, lat_dur=float(a.get('duration_s', 2.0)))
        elif do == 'open_door':
            door = a.get('door', 'offside')
            VD = getattr(self.carla, 'VehicleDoor', None)
            if VD is not None and hasattr(st['actor'], 'open_door'):
                # the offside door opens into the traffic: the driver's door, on the side away from the kerb
                name = {'offside': 'FR' if self.lht else 'FL'}.get(door, door)
                st['actor'].open_door(getattr(VD, name))
        self.w.emit.log(f'scenario: {a["actor"]} {do} at {t:.2f} s')

    def drive_actor(self, st, t):
        carla = self.carla
        actor = st['actor']
        if st['walker']:
            if st['walking']:
                yaw = math.radians(st['path'].yaw[0])
                actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0),
                                                        speed=st['walk_speed']))
            return
        # the set speed ramps towards the target at the action's rate
        if st['speed'] < st['target']:
            st['speed'] = min(st['target'], st['speed'] + st['accel'] * self.dt)
        elif st['speed'] > st['target']:
            st['speed'] = max(st['target'], st['speed'] - st['accel'] * self.dt)
        if st['lat_dur'] > 0:
            f = min(max((t - st['lat_t0']) / st['lat_dur'], 0.0), 1.0)
            st['lateral'] = st['lat_from'] + (st['lat_to'] - st['lat_from']) * f
        u = follow(actor, st['path'], st['lateral'], st['speed'], self.dt, st)
        actor.apply_control(carla.VehicleControl(**u))

    def drive_ego(self):
        carla = self.carla
        tr = self.ego.get_transform()
        v = self.ego.get_velocity()
        s = self.ego_path.project(tr.location.x, tr.location.y)
        route = [list(self.ego_path.at(s + 2.0 * k, 0.0)[:2]) + [None] for k in range(1, 26)]
        lead = self.lead(route) if self.driver == 'follower' else None
        obs = {'t': self.t, 'dt': self.dt,
               'ego': {'x': tr.location.x, 'y': tr.location.y, 'yaw': tr.rotation.yaw,
                       'speed_mps': math.hypot(v.x, v.y)}, 'route': route, 'lead': lead}
        u = self.policy.act(obs)
        self.ego.apply_control(carla.VehicleControl(throttle=float(u['throttle']), steer=float(u['steer']),
                                                    brake=float(u['brake'])))

    def lead(self, route):
        """The nearest scenario actor or vehicle within 1.8 m of the ego's route ahead (its corridor)."""
        best = None
        cands = [st['actor'] for st in self.actors.values()]
        for a in cands:
            try:
                l = a.get_location()
            except RuntimeError:
                continue
            for k, p in enumerate(route):
                if math.hypot(l.x - p[0], l.y - p[1]) < 1.8:
                    d = 2.0 * (k + 1)
                    if best is None or d < best[0]:
                        av = a.get_velocity()
                        best = (d, math.hypot(av.x, av.y))
                    break
        return None if best is None else {'distance_m': best[0], 'speed_mps': best[1]}

    def dist(self, actor):
        a, b = actor.get_location(), self.ego.get_location()
        return math.hypot(a.x - b.x, a.y - b.y)

    def gap_ahead(self, actor):
        """The actor's rear ahead of the ego's front along the ego's heading (m), or None when behind."""
        et = self.ego.get_transform()
        r = math.radians(et.rotation.yaw)
        f = np.array([math.cos(r), math.sin(r)])
        a = actor.get_location()
        along = (a.x - et.location.x) * f[0] + (a.y - et.location.y) * f[1]
        return along - actor.bounding_box.extent.x - self.ego.bounding_box.extent.x

    def report(self):
        return {'id': self.scn.get('id'), 'name': self.scn.get('name'), 'values': self.scn.get('values'),
                'seed': self.scn.get('seed'), 'fired': self.fired, 'collisions': self.events,
                'handedness': 'LHT' if self.lht else 'RHT'}


def _xy(actor):
    l = actor.get_location()
    return l.x, l.y


def follow(actor, path, lateral, speed, dt, st):
    """Pure-pursuit steering along the path (shifted by lateral) and a PI speed controller: VehicleControl kwargs."""
    tr = actor.get_transform()
    v = actor.get_velocity()
    vel = math.hypot(v.x, v.y)
    s = path.project(tr.location.x, tr.location.y)
    la = max(4.0, 0.8 * vel)
    tx, ty, _ = path.at(s + la, lateral)
    yaw = math.radians(tr.rotation.yaw)
    dx, dy = tx - tr.location.x, ty - tr.location.y
    fx = math.cos(yaw) * dx + math.sin(yaw) * dy
    fy = -math.sin(yaw) * dx + math.cos(yaw) * dy
    L = 2.0 * max(actor.bounding_box.extent.x, 0.6) * 0.6
    steer = math.atan(L * 2.0 * fy / max(fx * fx + fy * fy, 1e-6)) / math.radians(70.0)
    err = speed - vel
    st['i_err'] = max(min(st.get('i_err', 0.0) + err * dt, 5.0), -5.0)
    u = 0.5 * err + 0.08 * st['i_err']
    if speed <= 0.01 and vel < 0.3:
        return {'throttle': 0.0, 'steer': 0.0, 'brake': 1.0, 'hand_brake': True}
    if u >= 0:
        return {'throttle': min(u, 1.0), 'steer': max(min(steer, 1.0), -1.0), 'brake': 0.0}
    return {'throttle': 0.0, 'steer': max(min(steer, 1.0), -1.0), 'brake': min(-u, 1.0)}
