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

Motion: kinematic (the default) or physics. Measured on CARLA 0.9.16 (Town10HD_Opt, the reference machine): with
CARLA's vehicle physics, two runs of the same scenario with the same seed — synchronous mode, a fixed step, a freshly
reloaded world, every state read from the snapshot of the frame just ticked — drift apart: the ego by 3 cm with no
feedback controller and by up to 0.4 m with its lane follower within 12 s, whatever its blueprint, role or sensors.
In "kinematic" motion the director integrates every scenario vehicle (the ego included) with a kinematic bicycle
model driven by the same controllers, and walkers along their direction, and places them with CARLA's physics off
in one synchronous batch before each tick (workers.common.place_now: set_transform is asynchronous and could land a
tick late): the motion is computed here and repeats bit for bit; collisions are measured on the boxes
(core.criticality). Its limits: the wheels do not turn, and there are no tyre, suspension or collision dynamics.
"physics" motion drives them through CARLA's physics (realistic dynamics, not bit-for-bit repeatable).

Every pose and velocity is read from the world snapshot of exactly the frame just ticked (sync()).

Records: per step the ego's and every actor's (t, x, y, yaw, vx, vy, length, width) and the ego's collision events
(sensor.other.collision), for core.criticality; the timeline's firings with their times; each scenario actor's CARLA
id and blueprint (report(): 'actor_ids', 'blueprints'), which the labels' track ids refer to.

Appearance. An actor's optional "variant": k (core.rootcause's appearance swap) spawns the k-th next blueprint of its
kind's pool (adults among adults, children among children) with the generator drawn from exactly as without it.

Indian classes (core.assets). "autorickshaw" and "cycle_rickshaw" are stand-in vehicles (the inventory's nearest in
measured size) driven with their class's limits: the set speed never above its top speed and ramping up no faster
than its acceleration. "cattle", "handcart" and "vendor" are proxy props (a static prop, or a cooked mesh through
static.prop.mesh at a uniform scale) placed on the ground at their true size; a moving proxy is moved with
set_transform (props have no physics) along its path — measured on 0.9.16: a static.prop.* moves so, while a scaled
static.prop.mesh does not move and loses its scale, so moving proxies are unscaled props. Each gets its Falcon
subclass (w.subclass) and its true box (w.proxies: CARLA reports a scaled mesh's box centre unscaled), which the labels
and actors.jsonl use.
"""
from __future__ import annotations

import math

import numpy as np

from .common import ROLE_PREFIX, Failed, place_now

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


STAND_INS = ('autorickshaw', 'cycle_rickshaw')
PROXIES = {'cattle': 'cattle', 'handcart': 'handcart', 'vendor': 'street_vendor'}


def _class_of(kind):
    """The core.assets class of a scenario kind, or None for CARLA's own kinds."""
    return kind if kind in STAND_INS else PROXIES.get(kind)


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
        self.moves = None                # inside step(): kinematic moves collected for one batch before the tick
        self.fired = []
        self.traj = {'ego': []}
        self.events = []
        self.t = 0.0
        self.started = False
        self.snap = None
        self.kinematic = str(cfg.get('scenario', {}).get('motion', cfg.get('motion', 'kinematic'))) == 'kinematic'
        self.kin = {}                     # actor id -> [x, y, z, yaw_deg, v] when kinematic
        self.ego_control = None           # the control drive_ego applied for the coming tick (None: not driving it)
        self.external = set()             # actor ids driven from outside (workers/adversary.py's learned traffic)
        self.ego_filter = None            # f(control, observation) -> control: a stack's degraded mode (workers.stack)
        self.attached_only = False        # attach(): a run without a scenario; only actors added during it
        self.added = []                   # actors added during the run (trigger rules): {id, kind, t, carla_id}
        self.extra_timeline = []          # timeline entries added during the run
        self.fired_added = []             # (index in extra_timeline, t) as fired

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
        cls = _class_of(kind)
        if cls is not None:
            c = self.asset(kind)
            if c.get('size_m'):
                return (float(c['size_m'][0]), float(c['size_m'][1]))
        return {'bus': (10.3, 3.9), 'truck': (6.0, 2.6), 'van': (5.5, 2.0), 'motorcycle': (2.2, 0.9),
                'bicycle': (1.7, 0.6), 'pedestrian': (0.5, 0.5), 'child': (0.4, 0.4)}.get(kind, (4.6, 2.0))

    def asset(self, kind):
        """The stand-in or proxy of an Indian kind (core.assets.choose on the repo's measured inventory), cached."""
        cache = self.__dict__.setdefault('_assets', {})
        if kind not in cache:
            from ..core import assets as AS
            cache[kind] = AS.choose(_class_of(kind))
        return cache[kind]

    def spawn_proxy(self, a, path, lateral):
        """A proxy prop on the ground at the path's start: (actor, its true box {'center', 'extent'} in its frame)."""
        carla = self.carla
        from ..core import assets as AS
        c = self.asset(a['kind'])
        bp = self.lib.find(c['blueprint'])
        scale = float(c.get('scale') or 1.0)
        if c.get('mesh_path'):
            bp.set_attribute('mesh_path', c['mesh_path'])
            bp.set_attribute('scale', str(scale))
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', ROLE_PREFIX + 'prop')
        inv = AS.load_inventory()
        row = next((r for r in inv['props'] if r['id'] == c['id']), None)
        x, y, yaw = path.at(0.0, lateral)
        z = path.z + AS.placement_z(inv, c['id'], scale) + 0.02
        actor = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=z),
                                                               carla.Rotation(yaw=yaw + float(c.get('yaw') or 0.0))))
        if actor is None:
            raise Failed(f'actor {a["id"]} ({a["kind"]}) could not be placed: something is in the way')
        box = row['box'] if row else {'center': [0.0, 0.0, 0.5], 'extent': [0.5, 0.5, 0.5]}
        true = {'center': [v * scale for v in box['center']], 'extent': [v * scale for v in box['extent']],
                'z': z, 'yaw_offset': float(c.get('yaw') or 0.0)}
        return actor, true

    def blueprint(self, kind, variant: int = 0):
        """The blueprint of an actor of this kind. variant k > 0 takes the k-th next blueprint of the same pool (the
        root-cause attribution's appearance swap, core.rootcause): the generator is drawn from exactly as for the
        variant 0, so nothing else in the run changes. Adults stay adults and children children."""
        variant = int(variant or 0)
        if kind in ('pedestrian', 'child'):
            def age(b):
                return b.get_attribute('age').as_str() if b.has_attribute('age') else ''
            bps = sorted(self.lib.filter('walker.pedestrian.*'), key=lambda b: b.id)
            if kind == 'child':
                bps = [b for b in bps if age(b) == 'child'] or bps
            bp = bps[int(self.rng.integers(len(bps)))]          # the same draw whatever the variant
            if variant:
                pool = [b for b in bps if age(b) == age(bp)]
                ids = [b.id for b in pool]
                bp = pool[(ids.index(bp.id) + variant) % len(pool)]
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            return bp
        ids = {b.id for b in self.lib.filter('vehicle.*')}
        if kind in STAND_INS:
            bid = self.asset(kind)['blueprint']
            try:
                return self.lib.find(bid)
            except (IndexError, RuntimeError):
                raise Failed(f'the server has no {bid} (the {kind} stand-in)') from None
        avail = [c for c in KIND_BLUEPRINTS.get(kind, KIND_BLUEPRINTS['car']) if c in ids]
        if avail and (not variant or len(avail) > 1):
            return self.lib.find(avail[variant % len(avail)])
        from .scene import blueprints_by_type
        pool = sorted(blueprints_by_type(self.lib).get(kind, []) or blueprints_by_type(self.lib)['car'],
                      key=lambda b: b.id)
        if avail and variant:                            # one listed blueprint: the variant from the type's pool
            pool = [b for b in pool if b.id != avail[0]] or pool
            return pool[(variant - 1) % len(pool)]
        return pool[int(self.rng.integers(len(pool)))]

    def setup(self, ego, ego_wp):
        carla = self.carla
        self.ego = ego
        self.ego_path = self.ego_route(ego_wp)
        self.ego_speed = float(self.scn.get('ego', {}).get('speed_kmh', 30.0)) / 3.6
        self.driver = self.scn.get('ego', {}).get('driver', 'follower')
        from ..plugins.lane_follower import LaneFollower
        self.policy = LaneFollower(cruise_kmh=self.ego_speed * 3.6 + 1e-6)
        if not hasattr(self.w, 'subclass'):
            self.w.subclass = {}
        if not hasattr(self.w, 'proxies'):
            self.w.proxies = {}
        for a in self.scn.get('actors', []):
            self.spawn_actor(a, ego_wp)
        bp = self.lib.find('sensor.other.collision')
        bp.set_attribute('role_name', ROLE_PREFIX + 'sensor')
        self.col = self.w.own(self.world.spawn_actor(bp, carla.Transform(), attach_to=ego))
        self.col.listen(lambda e: self.events.append((self.t, int(e.other_actor.id) if e.other_actor else 0,
                                                       float(math.sqrt(e.normal_impulse.x ** 2 + e.normal_impulse.y ** 2
                                                                       + e.normal_impulse.z ** 2)))))
        if self.driver == 'autopilot':
            ego.set_autopilot(True, self.w.tm.get_port())
            if getattr(self.w, 'speed_control', None) is not None:     # an imported world's road limits
                self.w.speed_control.add(ego, role='ego')

    def spawn_actor(self, a, ego_wp):
        """Place one scenario actor (an entry of the scenario's "actors") relative to the ego's waypoint and register
        it; its trajectory is recorded from the next record(). Returns its state."""
        carla = self.carla
        from ..core import assets as AS
        path, lateral = self.actor_path(a, ego_wp)
        x, y, yaw = path.at(0.0, lateral)
        cls = _class_of(a['kind'])
        prop = a['kind'] in PROXIES
        if prop:
            actor, true = self.spawn_proxy(a, path, lateral)
            self.w.proxies[actor.id] = true
        else:
            bp = self.blueprint(a['kind'], a.get('variant', 0))
            bp.set_attribute('role_name', ROLE_PREFIX + 'scenario')
            tf = carla.Transform(carla.Location(x=x, y=y, z=path.z + (1.0 if a['kind'] in ('pedestrian', 'child')
                                                                       else 0.5)), carla.Rotation(yaw=yaw))
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is None:
                raise Failed(f'actor {a["id"]} ({a["kind"]}) could not be placed: something is in the way')
        self.w.own(actor)
        spec = AS.CLASSES.get(cls) or {}
        if cls is not None:
            self.w.subclass[actor.id] = spec.get('label', cls)
        top = spec.get('top_kmh')
        moving = a.get('start', 'moving') == 'moving'
        sp = float(a.get('speed_kmh', 0.0)) / 3.6 if moving else 0.0
        if top is not None:
            sp = min(sp, float(top) / 3.6)
        self.actors[a['id']] = {'actor': actor, 'kind': a['kind'], 'path': path, 'lateral': lateral,
                                'lat_from': lateral, 'lat_to': lateral, 'lat_t0': 0.0, 'lat_dur': 0.0,
                                'speed': sp, 'target': sp, 'accel': 3.0, 'walking': False,
                                'walk_speed': 0.0, 'walker': a['kind'] in ('pedestrian', 'child'),
                                'heading': a.get('heading', 'lane'), 'prop': prop, 's': 0.0,
                                'top': (float(top) / 3.6) if top is not None else None,
                                'max_accel': spec.get('accel_mps2'), 'class': cls}
        self.traj[a['id']] = []
        return self.actors[a['id']]

    def add_actor(self, a, t, rng=None):
        """An actor added during the run (a trigger rule's spawn, core.triggers): placed relative to where the ego
        is now, driven from the next step like the scenario's own (kinematic when the run is), its random choices
        from rng (the rule's own generator: the run's draws stay as they were). Returns the actor's CARLA id."""
        ego_wp = self.map.get_waypoint(self.loc(self.ego))
        if ego_wp is None:
            raise Failed('the ego is off the road network')
        if self.attached_only:                     # the ego's own route from here (lane changes aim at its lane)
            self.ego_path = lane_path(ego_wp, 400.0)
        if a['id'] in self.actors:
            raise Failed(f'an actor {a["id"]!r} is already in the run')
        keep = self.rng
        if rng is not None:
            self.rng = rng
        try:
            st = self.spawn_actor(a, ego_wp)
        finally:
            self.rng = keep
        actor = st['actor']
        if self.started:
            if st['prop']:
                tr = actor.get_transform()
                px = self.w.proxies[actor.id]
                self.kin[actor.id] = [tr.location.x, tr.location.y, px['z'], tr.rotation.yaw - px['yaw_offset'],
                                      st['speed']]
            elif self.kinematic:
                self.kin_init(actor)
                self.kin[actor.id][4] = st['speed']
            elif st['speed'] > 0 and not st['walker']:
                y = math.radians(actor.get_transform().rotation.yaw)
                actor.set_target_velocity(self.carla.Vector3D(st['speed'] * math.cos(y), st['speed'] * math.sin(y),
                                                              0))
        self.added.append({'id': a['id'], 'kind': a['kind'], 't': round(float(t), 4), 'carla_id': int(actor.id)})
        return int(actor.id)

    def add_timeline(self, entry):
        """A timeline entry added during the run (a rule's cut-in: its lane change when the gap opens); fired as the
        scenario's own, listed in report() "fired_added"."""
        self.extra_timeline.append(entry)

    @classmethod
    def attach(cls, w, world, cfg, lib, rng, ego):
        """A director for a run without a scenario (a plain drive, a Live session) that drives only the actors added
        to it (add_actor): the ego stays as the run drives it (the traffic manager, the keyboard, a policy) and is
        never placed or controlled here; a collision sensor on it records contacts with the added actors."""
        d = cls(w, world, dict(cfg, scenario={'id': None, 'name': 'trigger rules', 'actors': [], 'timeline': [],
                                              'ego': {'driver': 'external'}, 'motion': cfg.get('motion',
                                                                                              'kinematic')}),
                lib, rng)
        d.attached_only = True
        d.ego = ego
        d.driver = 'external'
        d.ego_speed = 0.0
        d.ego_path = lane_path(d.map.get_waypoint(ego.get_location()), 400.0)
        if not hasattr(w, 'subclass'):
            w.subclass = {}
        if not hasattr(w, 'proxies'):
            w.proxies = {}
        bp = lib.find('sensor.other.collision')
        bp.set_attribute('role_name', ROLE_PREFIX + 'sensor')
        d.col = w.own(world.spawn_actor(bp, w.carla.Transform(), attach_to=ego))
        d.col.listen(lambda e: d.events.append((d.t, int(e.other_actor.id) if e.other_actor else 0,
                                                float(math.sqrt(e.normal_impulse.x ** 2 + e.normal_impulse.y ** 2
                                                                + e.normal_impulse.z ** 2)))))
        d.started = True                            # nothing to start: actors added later are started as they come
        return d

    # ------------------------------------------------------------------ frame-consistent state
    def sync(self, frame, timeout=5.0):
        """Wait until the client holds the world snapshot of `frame` (World.tick()'s), then read from it."""
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

    def tf(self, actor):
        k = self.k_tf(actor)
        if k is not None:
            return k
        a = self.snap.find(actor.id) if self.snap is not None else None
        return a.get_transform() if a is not None else actor.get_transform()

    def vel(self, actor):
        k = self.k_vel(actor)
        if k is not None:
            return k
        a = self.snap.find(actor.id) if self.snap is not None else None
        return a.get_velocity() if a is not None else actor.get_velocity()

    def loc(self, actor):
        return self.tf(actor).location

    # ------------------------------------------------------------------ kinematic motion
    ACCEL, BRAKE, STEER_DEG = 3.0, 8.0, 70.0

    def kin_init(self, actor):
        tr = actor.get_transform()
        wp = self.map.get_waypoint(tr.location)
        z = wp.transform.location.z if wp is not None else tr.location.z
        actor.set_simulate_physics(False)
        self.kin[actor.id] = [tr.location.x, tr.location.y, z, tr.rotation.yaw, 0.0]

    def kin_step(self, actor, u, walk=None):
        """Integrate one step: a vehicle by the kinematic bicycle model from its controls (u), a walker along
        `walk` = (yaw_deg, speed). Places the actor; its state stays here."""
        carla = self.carla
        x, y, z, yaw, v = self.kin[actor.id]
        dt = self.dt
        if walk is not None:
            yaw, v = walk
            r = math.radians(yaw)
            x, y = x + math.cos(r) * v * dt, y + math.sin(r) * v * dt
        else:
            if u.get('hand_brake'):
                v = 0.0
            else:
                v = max(v + (self.ACCEL * u.get('throttle', 0.0) - self.BRAKE * u.get('brake', 0.0)) * dt, 0.0)
            wheelbase = max(2.0 * actor.bounding_box.extent.x * 0.6, 0.8)
            yaw2 = yaw + math.degrees(v * math.tan(u.get('steer', 0.0) * math.radians(self.STEER_DEG)) /
                                      wheelbase * dt)
            mid = math.radians((yaw + yaw2) / 2.0)
            x, y, yaw = x + math.cos(mid) * v * dt, y + math.sin(mid) * v * dt, yaw2
        self.kin[actor.id] = [x, y, z, yaw, v]
        self.move(actor, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)))

    def move(self, actor, tf):
        """Place an actor before the next tick: collected during step() and applied in one synchronous batch at its
        end; outside step() (a caller's own kin_step) at once. Never set_transform: it is asynchronous and can land a
        tick late (workers.common.place_now)."""
        if self.moves is not None:
            self.moves.append((actor, tf))
        else:
            place_now(self.w.client, self.carla, [(actor, tf)])

    def prop_init(self):
        """Proxies (props: no physics) are always moved here, along their path, in either motion."""
        for st in self.actors.values():
            if st['prop']:
                tr = st['actor'].get_transform()
                self.kin[st['actor'].id] = [tr.location.x, tr.location.y, self.w.proxies[st['actor'].id]['z'],
                                            tr.rotation.yaw - self.w.proxies[st['actor'].id]['yaw_offset'], st['speed']]

    def prop_step(self, st):
        """Advance a proxy along its path at its set speed (shifted by its lateral offset); placed only when it
        moves (a scaled static.prop.mesh cannot be moved: measured)."""
        carla = self.carla
        a = st['actor']
        k = self.kin[a.id]
        k[4] = st['speed']
        if st['speed'] <= 0.0 and st['lat_dur'] <= 0.0:
            return
        st['s'] += st['speed'] * self.dt
        x, y, yaw = st['path'].at(st['s'], st['lateral'])
        k[0], k[1], k[3] = x, y, yaw
        self.move(a, carla.Transform(carla.Location(x=x, y=y, z=k[2]),
                                     carla.Rotation(yaw=yaw + self.w.proxies[a.id]['yaw_offset'])))

    def k_tf(self, actor):
        k = self.kin.get(actor.id)
        if k is None:
            return None
        carla = self.carla
        return carla.Transform(carla.Location(x=k[0], y=k[1], z=k[2]), carla.Rotation(yaw=k[3]))

    def k_vel(self, actor):
        k = self.kin.get(actor.id)
        if k is None:
            return None
        r = math.radians(k[3])
        return self.carla.Vector3D(k[4] * math.cos(r), k[4] * math.sin(r), 0.0)

    # ------------------------------------------------------------------ stepping
    def start(self):
        """At the first step: everyone at their speed."""
        carla = self.carla
        self.started = True
        if self.kinematic:
            # an ego on the traffic manager's autopilot keeps CARLA's physics (the traffic manager drives through it)
            ego_kin = self.driver != 'autopilot'
            for a in ([self.ego] if ego_kin else []) + [st['actor'] for st in self.actors.values() if not st['prop']]:
                self.kin_init(a)
            self.prop_init()
            if ego_kin:
                self.kin[self.ego.id][4] = self.ego_speed
            for st in self.actors.values():
                self.kin[st['actor'].id][4] = st['speed']
            return
        self.prop_init()
        a = math.radians(self.tf(self.ego).rotation.yaw)
        self.ego.set_target_velocity(carla.Vector3D(self.ego_speed * math.cos(a), self.ego_speed * math.sin(a), 0))
        for st in self.actors.values():
            if st['speed'] > 0 and not st['walker'] and not st['prop']:
                y = math.radians(self.tf(st['actor']).rotation.yaw)
                st['actor'].set_target_velocity(carla.Vector3D(st['speed'] * math.cos(y), st['speed'] * math.sin(y),
                                                               0))

    def step(self, t):
        """Before each tick: fire the timeline, steer everyone. t: seconds since the scenario started."""
        self.t = t
        if not self.started:
            self.start()
        self.timeline(t)
        self.moves = []
        try:
            for aid, st in self.actors.items():
                if aid not in self.external:
                    self.drive_actor(st, t)
            if self.driver in ('follower', 'none'):
                self.drive_ego()
        finally:
            moves, self.moves = self.moves, None
        place_now(self.w.client, self.carla, moves)

    def record(self, t):
        for aid, a in [('ego', self.ego)] + [(k, v['actor']) for k, v in self.actors.items()]:
            try:
                tr, v, bb = self.tf(a), self.vel(a), a.bounding_box
            except RuntimeError:
                continue
            ex, ey = bb.extent.x, bb.extent.y
            px = (getattr(self.w, 'proxies', None) or {}).get(a.id)
            if px is not None:                           # its true box, along the path when it stands turned
                ex, ey = px['extent'][0], px['extent'][1]
                if round(abs(px['yaw_offset'])) % 180 == 90:
                    ex, ey = ey, ex
            self.traj[aid].append((round(t, 4), tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y,
                                   2 * ex, 2 * ey))

    def timeline(self, t):
        self._fire(t, self.scn.get('timeline', []), self.fired)
        self._fire(t, self.extra_timeline, self.fired_added)

    def _fire(self, t, entries, fired):
        for k, ev in enumerate(entries):
            if k in {f[0] for f in fired}:
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
                fired.append((k, round(t, 3)))
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
            ax, ay = self.loc(st['actor']).x, self.loc(st['actor']).y
            s_now = p.project(ax, ay)
            to = a.get('to', 'ego')
            here = self.map.get_waypoint(self.loc(self.ego))
            # the lateral offset (from the actor's own path) that puts it on the target lane's centre
            ego_s = self.ego_path.project(ax, ay)
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
            if self.kinematic:
                self.kin_step(actor, {}, walk=(st['path'].yaw[0], st['walk_speed'] if st['walking'] else 0.0))
            elif st['walking']:
                yaw = math.radians(st['path'].yaw[0])
                actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0),
                                                        speed=st['walk_speed']))
            return
        # the set speed ramps towards the target at the action's rate (a stand-in or proxy: within its class's
        # top speed and acceleration)
        if st.get('top') is not None:
            st['target'] = min(st['target'], st['top'])
        if st['speed'] < st['target']:
            up = st['accel'] if not st.get('max_accel') else min(st['accel'], float(st['max_accel']))
            st['speed'] = min(st['target'], st['speed'] + up * self.dt)
        elif st['speed'] > st['target']:
            st['speed'] = max(st['target'], st['speed'] - st['accel'] * self.dt)
        if st['lat_dur'] > 0:
            f = min(max((t - st['lat_t0']) / st['lat_dur'], 0.0), 1.0)
            st['lateral'] = st['lat_from'] + (st['lat_to'] - st['lat_from']) * f
        if st['prop']:
            self.prop_step(st)
            return
        u = follow(actor, st['path'], st['lateral'], st['speed'], self.dt, st, self.tf(actor), self.vel(actor))
        if self.kinematic:
            self.kin_step(actor, u)
        else:
            actor.apply_control(carla.VehicleControl(**u))

    def drive_ego(self):
        carla = self.carla
        tr = self.tf(self.ego)
        v = self.vel(self.ego)
        s = self.ego_path.project(tr.location.x, tr.location.y)
        route = [list(self.ego_path.at(s + 2.0 * k, 0.0)[:2]) + [None] for k in range(1, 26)]
        lead = self.lead(route) if self.driver == 'follower' else None
        obs = {'t': self.t, 'dt': self.dt,
               'ego': {'x': tr.location.x, 'y': tr.location.y, 'yaw': tr.rotation.yaw,
                       'speed_mps': math.hypot(v.x, v.y)}, 'route': route, 'lead': lead}
        u = self.policy.act(obs)
        if self.ego_filter is not None:
            u = self.ego_filter(dict(u), obs)
        self.ego_control = dict(u)
        if self.kinematic:
            self.kin_step(self.ego, u)
            return
        self.ego.apply_control(carla.VehicleControl(throttle=float(u['throttle']), steer=float(u['steer']),
                                                    brake=float(u['brake'])))

    def lead(self, route):
        """The nearest scenario actor or vehicle within 1.8 m of the ego's route ahead (its corridor)."""
        best = None
        cands = [st['actor'] for st in self.actors.values()]
        for a in cands:
            try:
                l = self.loc(a)
            except RuntimeError:
                continue
            for k, p in enumerate(route):
                if math.hypot(l.x - p[0], l.y - p[1]) < 1.8:
                    # bumper to bumper: the route distance less both half-lengths
                    d = max(2.0 * (k + 1) - self.ego.bounding_box.extent.x - a.bounding_box.extent.x, 0.0)
                    if best is None or d < best[0]:
                        av = self.vel(a)
                        best = (d, math.hypot(av.x, av.y))
                    break
        return None if best is None else {'distance_m': best[0], 'speed_mps': best[1]}

    def dist(self, actor):
        a, b = self.loc(actor), self.loc(self.ego)
        return math.hypot(a.x - b.x, a.y - b.y)

    def gap_ahead(self, actor):
        """The actor's rear ahead of the ego's front along the ego's heading (m), or None when behind."""
        et = self.tf(self.ego)
        r = math.radians(et.rotation.yaw)
        f = np.array([math.cos(r), math.sin(r)])
        a = self.loc(actor)
        along = (a.x - et.location.x) * f[0] + (a.y - et.location.y) * f[1]
        return along - actor.bounding_box.extent.x - self.ego.bounding_box.extent.x

    def report(self):
        return {'id': self.scn.get('id'), 'name': self.scn.get('name'), 'values': self.scn.get('values'),
                'motion': 'kinematic' if self.kinematic else 'physics',
                'seed': self.scn.get('seed'), 'fired': self.fired, 'collisions': self.events,
                'handedness': 'LHT' if self.lht else 'RHT',
                'actor_ids': {k: int(v['actor'].id) for k, v in self.actors.items()},
                'blueprints': {k: v['actor'].type_id for k, v in self.actors.items()},
                'subclasses': {k: self.w.subclass[v['actor'].id] for k, v in self.actors.items()
                               if v['actor'].id in (getattr(self.w, 'subclass', None) or {})},
                **({'added': self.added, 'fired_added': self.fired_added} if self.added else {})}


def follow(actor, path, lateral, speed, dt, st, tr, v):
    """Pure-pursuit steering along the path (shifted by lateral) and a PI speed controller: VehicleControl kwargs.
    tr, v: the actor's transform and velocity at this frame."""
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
