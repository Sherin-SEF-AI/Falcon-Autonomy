"""Scene setup shared by the drive and live workers: the map (a town, or an OpenDRIVE world generated from a file),
the weather, the ego, the traffic, the rig's cameras with their label cameras, and writing their frames.

Everything spawned is registered with the worker (Worker.own) so that its tidy-up removes it in a safe order.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import math
import os
import queue
import time

from ..core.budget import label_cameras
from .common import ROLE_HERO, ROLE_SENSOR, ROLE_TRAFFIC, Failed, place_now

FRAME_TIMEOUT_S = 10.0
OPENDRIVE_DEFAULTS = {'vertex_distance': 2.0, 'max_road_length': 500.0, 'wall_height': 0.0, 'additional_width': 0.6,
                      'smooth_junctions': True, 'enable_mesh_visibility': True, 'enable_pedestrian_navigation': True}


def load_map(w, cfg: dict):
    """Make the server's world the config's: a town by name (loaded when another is open), or an OpenDRIVE world
    (cfg['xodr'], a file) generated with cfg['opendrive'] parameters — skipped when the server already has exactly
    that road network (its Map.to_opendrive() hashes the same). Returns the world."""
    client, world = w.client, w.world
    load_s = float(cfg.get('load_timeout_s', 180.0))
    w.speed_control = None
    w.traffic_keeper = None
    w.road_look = None
    if cfg.get('xodr'):
        with open(cfg['xodr'], encoding='utf-8') as fh:
            text = fh.read()
        want = hashlib.sha256(text.encode()).hexdigest()
        try:
            have = hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest()
        except RuntimeError:
            have = ''
        if have != want:
            w.emit.progress(0.01, 'generating the OpenDRIVE world', force=True)
            p = dict(OPENDRIVE_DEFAULTS, **(cfg.get('opendrive') or {}))
            params = w.carla.OpendriveGenerationParameters(
                vertex_distance=float(p['vertex_distance']), max_road_length=float(p['max_road_length']),
                wall_height=float(p['wall_height']), additional_width=float(p['additional_width']),
                smooth_junctions=bool(p['smooth_junctions']), enable_mesh_visibility=bool(p['enable_mesh_visibility']),
                enable_pedestrian_navigation=bool(p['enable_pedestrian_navigation']))
            w.expect(load_s)
            client.set_timeout(load_s)
            t0 = time.monotonic()
            world = client.generate_opendrive_world(text, params)
            client.set_timeout(w.timeout)
            w.run.setdefault('timings', {})['generate_world_s'] = round(time.monotonic() - t0, 2)
        w.world = world
        w.alive()
        speed_control_for(w, cfg, text)
        from .trafficguard import TrafficKeeper
        if cfg.get('keep_traffic', True):       # traffic kept in the world at roads that end (workers.trafficguard)
            w.traffic_keeper = TrafficKeeper(w, world, int(cfg.get('seed', 0)))
        # the road ahead of a vehicle (dead ends on its planned path): RoadEndGuard looks with it for the ego
        w.road_look = w.traffic_keeper or TrafficKeeper(w, world, int(cfg.get('seed', 0)))
        if cfg.get('city'):                     # buildings, trees, lamps and ground around the roads (core.cityprops)
            from ..core.cityprops import populate
            populate(w, world, cfg)
        return world
    town = cfg.get('town')
    if town and cfg.get('reload_world') and world.get_map().name.rpartition('/')[2] == town:
        # a fresh world: CARLA's physics repeat bit for bit only from the same initial state
        w.emit.progress(0.01, f'reloading {town}', force=True)
        w.expect(load_s)
        client.set_timeout(load_s)
        world = client.reload_world(False)
        client.set_timeout(w.timeout)
        w.world = world
        w.alive()
        return world
    if town and world.get_map().name.rpartition('/')[2] != town:
        w.emit.progress(0.01, f'loading {town}', force=True)
        w.expect(load_s)
        client.set_timeout(load_s)
        t0 = time.monotonic()
        world = client.load_world(town)
        client.set_timeout(w.timeout)
        w.run.setdefault('timings', {})['load_world_s'] = round(time.monotonic() - t0, 2)
        w.world = world
        w.alive()
    return world


def synchronous(w, world, fps: float, seed: int, tm_port: int):
    """World and traffic manager in synchronous mode at 1 / fps, seeded."""
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = 1.0 / float(fps)
    world.apply_settings(st)
    w.tm = w.client.get_trafficmanager(int(tm_port))
    w.tm.set_synchronous_mode(True)
    w.tm.set_random_device_seed(int(seed))
    if hasattr(world, 'set_pedestrians_seed'):
        world.set_pedestrians_seed(int(seed))


def weather_params(carla, fields: dict):
    wp = carla.WeatherParameters()
    for k, v in (fields or {}).items():
        if hasattr(wp, k):
            setattr(wp, k, float(v))
    return wp


def spawn_ego(w, world, lib, spawns, order, blueprint: str, autopilot=True):
    bp = lib.find(blueprint)
    bp.set_attribute('role_name', ROLE_HERO)
    for k in list(order):
        ego = world.try_spawn_actor(bp, spawns[k])
        if ego is not None:
            order.remove(k)
            w.own(ego)
            if autopilot:
                ego.set_autopilot(True, w.tm.get_port())
                if getattr(w, 'speed_control', None) is not None:
                    w.speed_control.add(ego, role='ego')
            w.alive()
            return ego
    raise Failed('no free spawn point for the ego')


class RoadEndGuard:
    """Keeps the ego in the world. An imported (OpenDRIVE) world ends where its roads end: at the edge of an imported
    area some roads simply stop, and CARLA's autopilot drives straight on, off the road mesh into empty space, until
    Unreal destroys the vehicle below the world's kill depth (measured: a 1 km import of central Bengaluru, the ego
    on the autopilot fell from a dead end 542 m from the centre after 31 s; the live view showed a grey void and a
    falling speed of 300 km/h and more). CARLA's own towns are closed and never trigger it.

    step() after each tick: when the ego is on no driving lane (Map.get_waypoint, project_to_road off) and sinking —
    more than FALL_M below the last height it had on a lane, or falling faster than FALL_MPS — it is put back on the
    road at the spawn point nearest the spawn points' centroid that lies more than AWAY_M from where it left (so the
    autopilot does not drive straight back to the same dead end), stopped, and the event is returned and kept
    (events). Kinematic, director-driven egos are not guarded (the director places them itself).

    In an imported world the traffic manager's OSM mode also destroys a vehicle on the autopilot, the ego included,
    as soon as its planned path meets a dead end of its road graph (workers.trafficguard). So on a lane, every
    LOOK_EVERY_S, the guard looks along the ego's planned path with the world's road look-ahead (w.road_look,
    TrafficKeeper.dead_end_ahead): a dead end within it puts the ego back the same way before it gets there (at a
    spawn point with no dead end within SPAWN_CLEAR_M ahead), on the autopilot again so the traffic manager plans
    afresh. The traffic manager's graph has dead ends the map does not (trafficguard), which this cannot see: lost()
    then says what happened."""
    FALL_M = 1.0
    FALL_MPS = 3.0
    AWAY_M = 60.0
    LOOK_EVERY_S = 0.25

    def __init__(self, w, world, ego, spawns):
        self.carla = w.carla
        self.w, self.ego = w, ego
        self.cmap = world.get_map()
        locs = [sp.location for sp in spawns]
        if locs:
            cx = sum(l.x for l in locs) / len(locs)
            cy = sum(l.y for l in locs) / len(locs)
            self.spawns = sorted(spawns, key=lambda sp: math.hypot(sp.location.x - cx, sp.location.y - cy))
        else:
            self.spawns = []
        self.last_road_z = None
        self.last_seen = None            # (x, y, on a driving lane) at the last step
        self.look = getattr(w, 'road_look', None)        # an imported world's dead ends ahead (None: a town)
        self.looked = -1e9
        self.events: list = []

    def lost(self, sim_t: float):
        """None while the ego exists; else why it is gone, for the run's error (the server destroyed it: it cannot be
        driven or recorded any further)."""
        try:
            if self.ego.is_alive:
                return None
        except RuntimeError:
            pass
        where = ''
        if self.last_seen is not None:
            x, y, on_lane = self.last_seen
            where = (f' last seen on a driving lane at ({x:.0f}, {y:.0f}): the traffic manager removes a vehicle on '
                     'the autopilot whose planned path meets a dead end of its road graph (an imported world\'s '
                     'edge, docs/worlds.md)') if on_lane else \
                f' last seen off the road network at ({x:.0f}, {y:.0f}): it left the world below its kill depth'
        return f'the ego was destroyed by the simulator at {sim_t:.1f} s;{where}'.rstrip(';')

    def step(self, sim_t: float):
        carla = self.carla
        t = self.ego.get_transform()
        loc = t.location
        wp = self.cmap.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
        self.last_seen = (loc.x, loc.y, wp is not None)
        if wp is not None:
            self.last_road_z = loc.z
            if self.look is None or sim_t - self.looked < self.LOOK_EVERY_S:
                return None
            self.looked = sim_t
            v = self.ego.get_velocity()
            d = self.look.dead_end_ahead(self.ego, wp, loc, math.hypot(v.x, v.y))
            if d is None:
                return None
            return self.put_back(loc, sim_t, 'dead_end_ahead',
                                 f'a road ends {d:.0f} m ahead on the ego\'s planned path (e.g. at the edge of an '
                                 'imported area), where the traffic manager would remove it: put back on the road')
        vz = self.ego.get_velocity().z
        sinking = (self.last_road_z is not None and loc.z < self.last_road_z - self.FALL_M) or vz < -self.FALL_MPS
        if not sinking:
            return None
        return self.put_back(loc, sim_t, 'fell', 'the ego left the road network (a road ends there, e.g. at the edge '
                                                 'of an imported area) and was falling: put back on the road')

    def clear_ahead(self, sp) -> bool:
        if self.look is None:
            return True
        wp = self.cmap.get_waypoint(sp.location, project_to_road=True, lane_type=self.carla.LaneType.Driving)
        return wp is not None and self.look._walk(wp, 0.0, self.look.SPAWN_CLEAR_M) is None

    def put_back(self, loc, sim_t: float, cause: str, detail: str):
        far = [sp for sp in self.spawns if math.hypot(sp.location.x - loc.x, sp.location.y - loc.y) > self.AWAY_M]
        target = next((sp for sp in far if self.clear_ahead(sp)), None) or (far[0] if far else None)
        if target is None:
            return None
        carla = self.carla
        self.ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.ego.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        place_now(self.w.client, carla, [(self.ego, target)])     # applied before the next tick
        if cause == 'dead_end_ahead':                              # the autopilot plans afresh from there
            tm = getattr(self.w, 'tm', None)
            if tm is not None:
                self.ego.set_autopilot(True, tm.get_port())
            sc = getattr(self.w, 'speed_control', None)
            if sc is not None and self.ego.id in sc.rows:
                sc.rows[self.ego.id][3] = None                     # the desired speed is sent again
                sc.update(self.ego.id)
        self.last_road_z = target.location.z
        ev = {'sim_t': round(float(sim_t), 3), 'kind': 'road_end', 'cause': cause,
              'from': {'x': round(loc.x, 2), 'y': round(loc.y, 2), 'z': round(loc.z, 2)},
              'to': {'x': round(target.location.x, 2), 'y': round(target.location.y, 2),
                     'z': round(target.location.z, 2)},
              'detail': detail}
        self.events.append(ev)
        return ev


def blueprints_by_type(lib) -> dict:
    by_type = {}
    for bp in lib.filter('vehicle.*'):
        base = bp.get_attribute('base_type').as_str() if bp.has_attribute('base_type') else 'car'
        by_type.setdefault(base.lower(), []).append(bp)
    return by_type


def spawn_traffic(w, world, lib, spawns, order, rng, prof: dict) -> list:
    """prof['vehicles'] vehicles drawn by prof['fleet'] weights per CARLA base type (seeded), on the autopilot with
    prof['behaviour']. A fleet entry may name blueprints directly ('blueprint:vehicle.micro.microlino')."""
    import numpy as np
    n = int(prof.get('vehicles', 0))
    if n <= 0:
        return []
    fleet = {k: float(v) for k, v in (prof.get('fleet') or {'car': 1.0}).items() if float(v) > 0}
    by_type = blueprints_by_type(lib)
    pools = {}
    for k in sorted(fleet):
        if k.startswith('blueprint:'):
            ids = {b.id for b in lib.filter(k.split(':', 1)[1])}
            pools[k] = sorted((b for b in lib.filter('vehicle.*') if b.id in ids), key=lambda b: b.id)
        else:
            pools[k] = sorted(by_type.get(k, []), key=lambda b: b.id)
    missing = sorted(k for k, v in pools.items() if not v)
    if missing:
        w.note(f'the server has no blueprints for {", ".join(missing)}: left out of the traffic')
    kinds = [k for k in sorted(pools) if pools[k]]
    if not kinds:
        return []
    p = np.array([fleet[k] for k in kinds])
    p = p / p.sum()
    beh = prof.get('behaviour') or {}
    placed = []
    from ..core import assets as AS
    classes = AS.fleet_classes(prof)
    labels = prof.get('fleet_labels') or {}
    for k in list(order):
        if len(placed) >= n:
            break
        kind = kinds[int(rng.choice(len(kinds), p=p))]
        cands = pools[kind]
        bp = cands[int(rng.integers(len(cands)))]
        bp.set_attribute('role_name', ROLE_TRAFFIC)
        a = world.try_spawn_actor(bp, spawns[k])
        if a is None:
            continue
        order.remove(k)
        w.own(a)
        a.set_autopilot(True, w.tm.get_port())
        apply_behaviour(w.tm, a, beh)
        diff = float(beh.get('speed_difference_pct', 0.0))
        if beh.get('speed_spread_pct') and hasattr(w.tm, 'vehicle_percentage_speed_difference'):
            spread = float(beh['speed_spread_pct'])
            diff += float(rng.uniform(-spread, spread))
            w.tm.vehicle_percentage_speed_difference(a, diff)
        if kind in classes:                               # a stand-in: its class's label and limits
            class_limits(w).add(a, classes[kind], diff, labels.get(kind))
        elif getattr(w, 'speed_control', None) is not None:   # an imported world: its roads' limits
            w.speed_control.add(a, diff)
        if getattr(w, 'traffic_keeper', None) is not None:    # an imported world: kept in it at roads that end
            w.traffic_keeper.add(a, beh, diff if diff or beh.get('speed_spread_pct') else None, kind,
                                 classes.get(kind), labels.get(kind))
        placed.append((a, kind))
    w.alive()
    if len(placed) < n:
        w.note(f'{len(placed)} of {n} traffic vehicles placed (the map has {len(spawns)} spawn points)')
    counts = {}
    for _, kind in placed:
        counts[kind] = counts.get(kind, 0) + 1
    w.run.setdefault('traffic', {})['placed'] = counts
    return placed


def spawn_walkers(w, world, lib, rng, prof: dict) -> list:
    """prof['walkers'] pedestrians at seeded points of the navigation mesh, each walking to another under CARLA's
    walker AI; prof['behaviour']['walker_crossing_factor'] of them cross roads anywhere."""
    from .common import ROLE_WALKER
    n = int(prof.get('walkers', 0))
    if n <= 0:
        return []
    carla = w.carla
    beh = prof.get('behaviour') or {}
    if hasattr(world, 'set_pedestrians_cross_factor'):
        world.set_pedestrians_cross_factor(float(beh.get('walker_crossing_factor', 0.0)))
    bps = sorted(lib.filter('walker.pedestrian.*'), key=lambda b: b.id)
    ctrl_bp = lib.find('controller.ai.walker')
    out = []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = bps[int(rng.integers(len(bps)))]
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        bp.set_attribute('role_name', ROLE_WALKER)
        walker = world.try_spawn_actor(bp, carla.Transform(loc))
        if walker is None:
            continue
        w.own(walker)
        ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), walker)
        w.own(ctrl)
        out.append((walker, ctrl))
    world.tick()                                         # controllers start after a step
    for walker, ctrl in out:
        ctrl.start()
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            ctrl.go_to_location(dest)
        ctrl.set_max_speed(float(rng.uniform(1.0, 1.8)))
    w.alive()
    if len(out) < n:
        w.note(f'{len(out)} of {n} pedestrians placed (the map\'s navigation mesh had no room for more)')
    w.run.setdefault('traffic', {})['walkers'] = len(out)
    return out


def apply_behaviour(tm, a, beh: dict):
    if 'distance_to_leading_m' in beh:
        tm.distance_to_leading_vehicle(a, float(beh['distance_to_leading_m']))
    if 'speed_difference_pct' in beh and hasattr(tm, 'vehicle_percentage_speed_difference'):
        tm.vehicle_percentage_speed_difference(a, float(beh['speed_difference_pct']))
    if 'ignore_lights_pct' in beh:
        tm.ignore_lights_percentage(a, float(beh['ignore_lights_pct']))
    if 'ignore_signs_pct' in beh:
        tm.ignore_signs_percentage(a, float(beh['ignore_signs_pct']))
    if 'ignore_walkers_pct' in beh:
        tm.ignore_walkers_percentage(a, float(beh['ignore_walkers_pct']))
    if 'auto_lane_change' in beh:
        tm.auto_lane_change(a, bool(beh['auto_lane_change']))
    if 'random_left_lanechange_pct' in beh and hasattr(tm, 'random_left_lanechange_percentage'):
        tm.random_left_lanechange_percentage(a, float(beh['random_left_lanechange_pct']))
    if 'random_right_lanechange_pct' in beh and hasattr(tm, 'random_right_lanechange_percentage'):
        tm.random_right_lanechange_percentage(a, float(beh['random_right_lanechange_pct']))


class ClassLimits:
    """The stand-ins of a run (core.assets): each vehicle's Falcon subclass (w.subclass: actor id → label, read by the
    labels and actors.jsonl) and its class's kinematic limits on the traffic manager's autopilot — the engine torque
    scaled once (VehiclePhysicsControl; measured on 0.9.16 to give the class's acceleration) and, every step(), the
    desired speed: the road's limit (Vehicle.get_speed_limit) less the vehicle's percentage difference, capped at the
    class's top speed (set_desired_speed replaces the traffic manager's limit-relative target, so the limit is applied
    here; measured: a cap of 20 km/h held a Microlino at p95 19.3 km/h on Town10HD's 30 km/h roads)."""

    EVERY_S = 1.0

    def __init__(self, w):
        self.w = w
        self.rows = {}                    # actor id → (actor, class id, diff %, top km/h)
        self.last = -1e9
        self.applied = {}

    def add(self, a, cls, diff_pct=0.0, label=None):
        from ..core import assets as AS
        spec = AS.CLASSES.get(cls) or {}
        self.w.subclass[a.id] = label or spec.get('label') or cls
        self.rows[a.id] = (a, cls, float(diff_pct), spec.get('top_kmh'))
        ts = spec.get('torque_scale')
        if ts and hasattr(a, 'get_physics_control'):
            try:
                pc = a.get_physics_control()
                tc = list(getattr(pc, 'torque_curve', None) or [])
                if tc:
                    pc.torque_curve = [self.w.carla.Vector2D(p.x, p.y * float(ts)) for p in tc]
                    a.apply_physics_control(pc)
                    self.applied.setdefault(cls, {})['torque_scale'] = float(ts)
            except RuntimeError as e:
                self.w.note(f'{cls}: the torque scale could not be applied ({e})')
        sc = getattr(self.w, 'speed_control', None)
        if sc is not None:                # an imported world: the road's limit from the file, capped at the class's top
            sc.add(a, float(diff_pct), spec.get('top_kmh'), role=cls)
            if spec.get('top_kmh'):
                self.applied.setdefault(cls, {})['top_kmh'] = spec['top_kmh']
            return
        self.update(a.id, force=True)

    def update(self, aid, force=False):
        from ..core import assets as AS
        a, cls, diff, top = self.rows[aid]
        tm = self.w.tm
        if tm is None or not hasattr(tm, 'set_desired_speed') or not top:
            return
        try:
            lim = float(a.get_speed_limit()) if hasattr(a, 'get_speed_limit') else 0.0
        except RuntimeError:
            return                        # removed by the server (a dead end)
        if lim <= 0:
            lim = top
        tm.set_desired_speed(a, AS.desired_speed_kmh(lim, diff, top))
        self.applied.setdefault(cls, {})['top_kmh'] = top

    def step(self, sim_t: float):
        if not self.rows or sim_t - self.last < self.EVERY_S or getattr(self.w, 'speed_control', None) is not None:
            return
        self.last = sim_t
        for aid in list(self.rows):
            self.update(aid)

    def report(self) -> dict:
        counts = {}
        for _, cls, _, _ in self.rows.values():
            counts[cls] = counts.get(cls, 0) + 1
        return {'stand_ins': counts, 'limits': self.applied}


def speed_control_for(w, cfg: dict, xodr_text: str):
    """An OpenDRIVE world's SpeedControl (workers.speedcontrol: each vehicle at its road's limit from the file, slowed
    for breakers) as w.speed_control — unless cfg['speed_limits'] is 'carla' (the traffic manager's own: 30 km/h on a
    world without speed signs). run.json traffic.speed_source says which."""
    mode = cfg.get('speed_limits', 'map')
    if mode == 'carla':
        w.speed_control = None
        w.run.setdefault('traffic', {})['speed_source'] = {
            'mode': 'carla', 'method': 'the traffic manager\'s own: Vehicle.get_speed_limit() (no speed signs on an '
                                      'OpenDRIVE world: 30 km/h)'}
        return None
    from ..core import speedlimits as SL
    from .speedcontrol import SpeedControl
    w.speed_control = SpeedControl(w, xodr_text, SL.features_path(cfg['xodr']))
    return w.speed_control


def speeds_step(w, sim_t: float) -> list:
    """Re-read the vehicles' desired speeds (the stand-ins' class limits, an imported world's road limits), and keep
    an imported world's traffic in it (workers.trafficguard). Returns the traffic moved this step, as events."""
    for c in (getattr(w, 'class_limits', None), getattr(w, 'speed_control', None)):
        if c is not None:
            c.step(sim_t)
    k = getattr(w, 'traffic_keeper', None)
    return k.step(sim_t) if k is not None else []


def speeds_report(w):
    """run.json traffic: the stand-ins' limits, the speed source and the traffic kept in an imported world."""
    t = w.run.setdefault('traffic', {})
    if getattr(w, 'class_limits', None) is not None:
        t.update(w.class_limits.report())
    if getattr(w, 'speed_control', None) is not None:
        t['speed_source'] = w.speed_control.report()
    if getattr(w, 'traffic_keeper', None) is not None:
        t['keeper'] = w.traffic_keeper.report()


def light_vehicles(w, cfg: dict, weather: dict | None = None, actors: list | None = None) -> dict:
    """Vehicle lights by the config's vehicle_lights (workers.lights: 'auto' on when the sun is below the horizon)
    for the worker's own vehicles (or actors), with weather in place of the config's when the weather changed (Live):
    lights switched on by an earlier call go off again when the mode resolves to off. Returns what was applied."""
    from . import lights
    c = cfg if weather is None else dict(cfg, weather=weather)
    acts = actors if actors is not None else [a for a in w.spawned if str(getattr(a, 'type_id', '')).startswith(
        'vehicle.')]
    alive = []
    for a in acts:
        try:
            if a is not None and a.is_alive:
                alive.append(a)
        except RuntimeError:
            pass
    applied = lights.apply(w, c, alive)
    prev = getattr(w, 'lights_applied', None)
    if actors is None and applied['resolved'] == 'off' and prev and prev.get('resolved') == 'on':
        for a in alive:
            try:
                a.set_light_state(w.carla.VehicleLightState.NONE)
            except RuntimeError:
                pass
    if actors is None:
        w.lights_applied = applied
    return applied


def class_limits(w) -> ClassLimits:
    """The worker's ClassLimits (made on first use), with w.subclass."""
    if not hasattr(w, 'subclass'):
        w.subclass = {}
    if getattr(w, 'class_limits', None) is None:
        w.class_limits = ClassLimits(w)
    return w.class_limits


def label_kinds(s) -> list:
    """(output kind, CARLA camera) per label camera a sensor needs, in a fixed order."""
    names = {'semantic_segmentation': 'semantic', 'instance_segmentation': 'instance', 'depth': 'depth'}
    return [(names[c], c) for c in label_cameras(s.get('labels'))]


def camera_transform(carla, p: dict):
    return carla.Transform(carla.Location(x=p['x'], y=p['y'], z=p['z']),
                           carla.Rotation(pitch=p['pitch'], yaw=p['yaw'], roll=p['roll']))


class Cameras:
    """Spawned cameras (each with its label cameras) and the frames they deliver, by frame number."""

    def __init__(self, w):
        self.w = w
        self.queues: dict = {}
        self.cams: list = []              # [(sensor dict, {kind: carla sensor})]
        self.pool = cf.ThreadPoolExecutor(max_workers=4)
        self.pending: list = []

    def spawn_rig(self, world, lib, ego, sensors, with_labels=True):
        for s in sensors:
            if s['type'] != 'camera.pinhole':
                continue                                   # fisheye, lidar, radar, GNSS, IMU: milestone 3
            kinds = [('rgb', 'sensor.camera.rgb')] + ([(k, 'sensor.camera.' + c) for k, c in label_kinds(s)]
                                                       if with_labels else [])
            from ..core import faults as F
            pose = F.actual_pose(s['pose'], s.get('faults'))           # mount faults: CARLA spawns it moved
            blur = next((f for f in s.get('faults') or [] if f.get('kind') == 'motion_blur'), None)
            attrs = {} if blur is None else {'motion_blur_intensity': blur.get('intensity', 0.45),
                                             'motion_blur_max_distortion': blur.get('max_distortion', 0.35)}
            self.cams.append((s, {k: self.spawn(world, lib, bp, s['width'], s['height'], s['fov'],
                                                camera_transform(self.w.carla, pose), ego,
                                                attrs=attrs if k == 'rgb' else None) for k, bp in kinds}))

    def spawn(self, world, lib, bp_id, width, height, fov, tf, parent=None, attachment=None, attrs=None):
        bp = lib.find(bp_id)
        for k, v in (attrs or {}).items():
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))
        bp.set_attribute('image_size_x', str(int(width)))
        bp.set_attribute('image_size_y', str(int(height)))
        bp.set_attribute('fov', str(float(fov)))
        bp.set_attribute('sensor_tick', '0.0')
        bp.set_attribute('role_name', ROLE_SENSOR)
        if attachment is not None:
            cam = world.spawn_actor(bp, tf, attach_to=parent, attachment_type=attachment)
        else:
            cam = world.spawn_actor(bp, tf, attach_to=parent) if parent is not None else world.spawn_actor(bp, tf)
        self.w.own(cam)
        q = queue.Queue()
        self.queues[cam.id] = q
        cam.listen(q.put)
        self.w.alive()
        return cam

    def get(self, sid, frame):
        """The sensor's data of exactly this frame (older ones dropped); Failed when it does not come."""
        q = self.queues[sid]
        t0 = time.monotonic()
        while True:
            left = FRAME_TIMEOUT_S - (time.monotonic() - t0)
            if left <= 0:
                raise Failed(f'sensor {sid} sent no data for frame {frame} within {FRAME_TIMEOUT_S:g} s')
            try:
                d = q.get(timeout=min(left, 1.0))
            except queue.Empty:
                self.w.check()
                continue
            if d.frame == frame:
                return d
            if d.frame > frame:
                raise Failed(f'sensor {sid} skipped frame {frame} (sent {d.frame}): the server is not in '
                             'synchronous mode')

    def drain(self, frame):
        for q in self.queues.values():
            while True:
                try:
                    d = q.get_nowait()
                except queue.Empty:
                    break
                if d.frame >= frame:
                    break

    def remove(self, sensor):
        self.queues.pop(sensor.id, None)

    def submit_write(self, kind, data, path, image_faults=None):
        """Write a frame in the background; image_faults: (faults, rng, K, omega) applied to an RGB image first."""
        import numpy as np
        a = np.frombuffer(data.raw_data, np.uint8).reshape(data.height, data.width, 4).copy()
        if image_faults is not None and kind == 'rgb':
            from ..core import faults as F
            fl, rng, K, omega = image_faults
            bgr, _ = F.apply_image(np.ascontiguousarray(a[:, :, :3]), fl, rng, K, omega)
            a = np.dstack([bgr, a[:, :, 3]])
        self.pending.append(self.pool.submit(write_frame, kind, a, path))

    def flush(self, keep):
        while len(self.pending) > keep:
            self.pending.pop(0).result()

    def stop(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()


def write_frame(kind, bgra, path):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if kind == 'rgb':
        ok = cv2.imwrite(path, bgra[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 92])
    elif kind == 'semantic':
        ok = cv2.imwrite(path, bgra[:, :, 2])            # R = the tag
    else:                                                # instance (R tag, G + 256·B id), depth (24-bit): lossless
        ok = cv2.imwrite(path, bgra[:, :, :3])
    if not ok:
        raise OSError(f'could not write {path}')


def intrinsics(width, height, fov):
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]]


def yaw_rate_optical(s: dict, yaw_rate: float):
    """The vehicle's yaw rate (rad/s, turning right positive) as an angular velocity in a camera's optical frame
    (x right, y down, z forward): about the world's down axis seen from the camera (pitch and roll rates are not
    used — the rolling-shutter model is yaw-driven)."""
    import numpy as np
    from ..core.rig import carla_rotation
    p = s['pose']
    Rb = carla_rotation(p['roll'], p['pitch'], p['yaw'])
    down = Rb.T @ np.array([0.0, 0.0, -1.0])                 # world down in the camera's body axes
    return [yaw_rate * down[1], yaw_rate * -down[2], yaw_rate * down[0]]


def rgb_of(image):
    """A CARLA camera image (BGRA raw data) as an RGB uint8 array."""
    import numpy as np
    a = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    return np.ascontiguousarray(a[:, :, 2::-1])


def ego_state(ego) -> dict:
    t = ego.get_transform()
    v = ego.get_velocity()
    return {'ego': {'x': round(t.location.x, 4), 'y': round(t.location.y, 4), 'z': round(t.location.z, 4),
                    'roll': round(t.rotation.roll, 4), 'pitch': round(t.rotation.pitch, 4),
                    'yaw': round(t.rotation.yaw, 4)},
            'speed_mps': round(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z), 4)}


def dataset_doc(name, cams, n, fps, run, extra=None) -> dict:
    """dataset.json of a recording. cams: camera recorders (camgt.CameraRecorder) or (sensor, {kind: …}) pairs."""
    from ..core.groundtruth import classes_doc
    sensors = []
    for c in cams:
        if isinstance(c, tuple):
            s, kinds = c
            K, lens_doc, views = intrinsics(s['width'], s['height'], s['fov']), None, 1
        else:
            s, kinds = c.s, c.kinds()
            K, lens_doc, views = [list(map(float, r)) for r in c.plan['K']], c.plan['summary'].get('lens'), \
                len(c.plan['views'])
        sensors.append({'id': s['id'], 'name': s['name'], 'type': s['type'], 'width': s['width'],
                        'height': s['height'], 'fov': s.get('fov'), 'K': K, 'lens': lens_doc, 'views': views,
                        'pose': s['pose'], 'streams': sorted(kinds)})
    doc = {'schema': 'falcon.dataset/1', 'layout': 'falcon-raw', 'name': name, 'frames': n, 'fps': fps,
           'sensors': sensors,
           'labels': {'semantic': {'encoding': 'uint8 PNG, a class per pixel (CARLA 0.9.16 semantic tags; 29 = the '
                                               'ego vehicle)', 'classes': classes_doc()},
                      'instance': {'encoding': 'uint16 PNG, 0 = none, else the id of an object in objects/'},
                      'range': {'encoding': 'uint16 PNG, round(range_m × 256) along each pixel\'s ray, 0 = nothing '
                                            'within 256 m'},
                      'objects': {'encoding': 'JSON per frame: id, track_id, class, category, bbox, bbox_full, '
                                              'truncated, occluded, range_m, pixels, box3d (vehicle frame), rle; '
                                              'subclass for a stand-in or proxy (docs/indian-assets.md): its class '
                                              'and category stay those of the CARLA mesh it is'},
                      'lidar': {'encoding': 'float32 x y z intensity (sensor frame); .labels.bin: uint8 tag then '
                                            'uint32 actor id per point; tag 30 = weather_noise (a spurious return '
                                            'of Falcon\'s sensor physics, actor id 0)'}},
           'frames_index': 'frames.jsonl',
           'coordinates': 'CARLA frames (x forward, y right, z up; left-handed), metres and degrees',
           'provenance': run['provenance']}
    doc.update(extra or {})
    return doc

GROUND_AS_SKY_SHARE = 0.5


def ground_label_check(data) -> str | None:
    """A warning when a semantic image labels the ground as sky: measured on CARLA 0.9.16, the road and sidewalk
    meshes of a world built by generate_opendrive_world come out as tag 11 (sky) — 99 % of pixels of a central
    Bengaluru drive — while CARLA's towns label them road (1) and sidewalk (2). Checked on the image's lower third,
    which a forward camera sees as ground."""
    import numpy as np
    tags = np.frombuffer(data.raw_data, np.uint8).reshape(data.height, data.width, 4)[:, :, 2]
    low = tags[2 * data.height // 3:]
    share = float(np.mean(low == 11))
    if share < GROUND_AS_SKY_SHARE:
        return None
    return (f'semantic labels: {100 * share:.0f} % of the lower third of the image is tagged sky — CARLA 0.9.16 '
            'labels the road and sidewalk meshes of a generated OpenDRIVE world as sky (11). Road, sidewalk and '
            'ground pixels of this run are mislabelled; other classes (vehicles, walkers, poles, lights) are not '
            'affected. Map-based ground relabelling comes with the data engine (milestone 3).')


class Others:
    """Radar, GNSS and IMU of the rig at their rates (sensor_tick = 1 / rate): radar detections per step as .npy
    (float32 velocity, azimuth, altitude, depth — CARLA's order), GNSS and IMU readings as lines of jsonl."""

    def __init__(self, w, world, lib, ego, sensors, cameras):
        self.w, self.cams, self.items = w, cameras, []
        for s in sensors:
            t = s['type']
            bp_id = {'radar': 'sensor.other.radar', 'gnss': 'sensor.other.gnss', 'imu': 'sensor.other.imu'}.get(t)
            if bp_id is None:
                continue
            bp = lib.find(bp_id)
            for k, v in s.items():
                if k in ('horizontal_fov', 'vertical_fov', 'range', 'points_per_second') and bp.has_attribute(k):
                    bp.set_attribute(k, str(v))
            if t == 'gnss':
                for k in ('noise_lat_stddev', 'noise_lon_stddev', 'noise_alt_stddev'):
                    if bp.has_attribute(k):
                        bp.set_attribute(k, str(s.get(k, 0.0)))
            if t == 'imu':
                for ax in 'xyz':
                    for kind, key in (('accel', 'noise_accel_stddev'), ('gyro', 'noise_gyro_stddev')):
                        a = f'noise_{kind}_stddev_{ax}'
                        if bp.has_attribute(a):
                            bp.set_attribute(a, str(s.get(key, 0.0)))
            bp.set_attribute('sensor_tick', str(1.0 / float(s.get('rate_hz', 20.0))))
            bp.set_attribute('role_name', ROLE_SENSOR)
            a = w.own(world.spawn_actor(bp, camera_transform(w.carla, s['pose']), attach_to=ego))
            q = queue.Queue()
            cameras.queues[a.id] = q
            a.listen(q.put)
            self.items.append((s, a, q))
        self.files = {}
        self.ego = ego
        self.physics = None                 # core.sensorphysics.Session: rain on the radars

    def step(self, frame, sim_t, out_dir, rec):
        import json
        import numpy as np
        for s, a, q in self.items:
            got = []
            while True:
                try:
                    d = q.get_nowait()
                except queue.Empty:
                    break
                if d.frame == frame:
                    got.append(d)
            for d in got:
                t = s['type']
                if t == 'radar':
                    rel = os.path.join('radar', s['id'], f'{rec["step"]:06d}.npy')
                    path = os.path.join(out_dir, rel)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    dets = np.frombuffer(d.raw_data, np.float32).reshape(-1, 4)
                    if self.physics is not None:
                        v = self.ego.get_velocity()
                        dets, flags, st = self.physics.radar(s, dets, math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))
                        if st is not None and (st['dropped'] or st['clutter']):
                            np.save(path[:-4] + '.flags.npy', flags)
                            rec['files'][f'{s["id"]}/radar_flags'] = rel[:-4] + '.flags.npy'
                            rec.setdefault('physics', {})[s['id']] = [k for k, n in (
                                ('radar_dropout', st['dropped']), ('radar_clutter', st['clutter'])) if n]
                    np.save(path, dets)
                    rec['files'][f'{s["id"]}/radar'] = rel
                    continue
                if t == 'gnss':
                    row = {'frame': int(frame), 'sim_t': sim_t, 'lat': d.latitude, 'lon': d.longitude,
                           'alt': d.altitude}
                else:
                    row = {'frame': int(frame), 'sim_t': sim_t,
                           'accel': [d.accelerometer.x, d.accelerometer.y, d.accelerometer.z],
                           'gyro': [d.gyroscope.x, d.gyroscope.y, d.gyroscope.z], 'compass': d.compass}
                fh = self.files.get(s['id'])
                if fh is None:
                    os.makedirs(os.path.join(out_dir, t), exist_ok=True)
                    fh = self.files[s['id']] = open(os.path.join(out_dir, t, f'{s["id"]}.jsonl'), 'w',
                                                    encoding='utf-8')
                fh.write(json.dumps(row) + '\n')

    def close(self):
        for fh in self.files.values():
            fh.close()
        self.files.clear()



class FaultState:
    """Per-sensor fault bookkeeping in a run: a seeded generator per sensor, the last delivered files (freeze), what
    was applied (run.json "faults_applied"). Faults started mid-run (start(): a trigger rule's fault action, a stack
    test's timed fault) join the planned ones with their [onset_s, until_s) window (core.faults.active)."""

    def __init__(self, sensors, seed: int):
        self.seed = int(seed)
        self.f = {s['id']: list(s.get('faults') or []) for s in sensors if s.get('faults')}
        self.rng = {sid: self._rng(sid) for sid in self.f}
        self.last = {}
        self.applied = {sid: {} for sid in self.f}
        self.started = []                 # [{sensor, fault, onset_s, until_s}] faults started during the run

    def _rng(self, sid):
        import zlib
        import numpy as np
        return np.random.default_rng([self.seed, zlib.crc32(sid.encode())])

    def start(self, sid, fault: dict, onset_s: float, for_s: float | None = None) -> dict:
        """Start a fault on a sensor from onset_s (for for_s seconds, or to the end of the run)."""
        f = dict(fault, onset_s=round(float(onset_s), 6))
        if for_s:
            f['until_s'] = round(float(onset_s) + float(for_s), 6)
        if sid not in self.f:
            self.f[sid], self.rng[sid], self.applied[sid] = [], self._rng(sid), {}
        self.f[sid].append(f)
        self.started.append({'sensor': sid, 'fault': f})
        return f

    def count(self, sid, kind):
        self.applied[sid][kind] = self.applied[sid].get(kind, 0) + 1

    def delivery(self, sid, step, sim_t):
        from ..core import faults as F
        if sid not in self.f:
            return 'ok'
        d = F.delivery(self.f[sid], step, sim_t, self.rng[sid])
        if d != 'ok':
            self.count(sid, d)
        return d

    def image(self, sid, K, omega, sim_t=None):
        from ..core import faults as F
        fl = [f for f in self.f.get(sid, []) if f.get('kind') in F.IMAGE_FAULTS and
              (sim_t is None or F.active(f, sim_t))]
        if not fl:
            return None
        for f in fl:
            self.count(sid, f['kind'])
        return fl, self.rng[sid], K, omega

    def stamp(self, sid, sim_t):
        from ..core import faults as F
        return sim_t + F.clock_offset_s(self.f.get(sid))

    def report(self) -> dict:
        out = {sid: {'planned': [f for f in self.f[sid] if not any(f is x['fault'] for x in self.started)],
                     'frames': self.applied[sid]} for sid in self.f}
        for x in self.started:
            out[x['sensor']].setdefault('started', []).append(x['fault'])
        return out


def write_lanes(w, cmap, out_dir, step: float = 2.0):
    """lanes.json: the map's driving lanes for lane-line and lane-graph labels, in CARLA's world frame.

    'points': one row per waypoint every `step` m of every driving lane — x, y, z, yaw (°), width, road id, section
    id, lane id, s, and the left and right lane markings (CARLA's LaneMarkingType and LaneMarkingColor names); a lane
    boundary is its centre ± width / 2 along the lane's right (x, y) = (−sin yaw, cos yaw). 'graph': the map's
    topology — lane segments [road, section, lane, s] → [road, section, lane, s]."""
    import json

    def mk(m):
        return [str(getattr(m.type, 'name', m.type)), str(getattr(m.color, 'name', m.color))] if m is not None \
            else ['NONE', 'Standard']
    rows = []
    for wp in cmap.generate_waypoints(step):
        if wp.lane_type != w.carla.LaneType.Driving:
            continue
        t = wp.transform
        rows.append([round(t.location.x, 3), round(t.location.y, 3), round(t.location.z, 3),
                     round(t.rotation.yaw, 3), round(wp.lane_width, 3), int(wp.road_id), int(wp.section_id),
                     int(wp.lane_id), round(wp.s, 3)] + mk(getattr(wp, 'left_lane_marking', None)) +
                    mk(getattr(wp, 'right_lane_marking', None)))
    graph = []
    for a, b in cmap.get_topology():
        graph.append([[int(a.road_id), int(a.section_id), int(a.lane_id), round(a.s, 3)],
                      [int(b.road_id), int(b.section_id), int(b.lane_id), round(b.s, 3)]])
    doc = {'schema': 'falcon.lanes/1', 'frame': 'CARLA world (x, y, z metres; yaw degrees; left-handed)',
           'step_m': step, 'columns': ['x', 'y', 'z', 'yaw', 'width', 'road', 'section', 'lane', 's',
                                       'left_type', 'left_color', 'right_type', 'right_color'],
           'points': rows, 'graph': graph}
    with open(os.path.join(out_dir, 'lanes.json'), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, separators=(',', ':'))
    return len(rows)



class ActorLog:
    """actors.jsonl: every vehicle and walker (and every proxy prop a scenario placed, core.assets) at every recorded
    step, in CARLA's world frame, from the world snapshot of that frame — {step, frame, sim_t, actors: [{id, type_id,
    category, role, x, y, z, roll, pitch, yaw, vx, vy, vz, extent [ex, ey, ez], bbox_center [cx, cy, cz] (in the
    actor's frame), subclass (a stand-in's or proxy's Falcon subclass, when it has one)}]}. Boxes and categories are
    read once per actor (they do not change). Actors a kinematic director moves (physics off: CARLA reports them still)
    take the director's own velocity. path None: no file (rows() only; the gym worker's observations use the same rows)."""

    def __init__(self, w, path):
        self.w = w
        self.fh = open(path, 'w', encoding='utf-8') if path else None
        self.static = {}

    def step(self, world, i, frame, sim_t, director=None):
        import json
        rows = self.rows(world, director)
        self.fh.write(json.dumps({'step': i, 'frame': int(frame), 'sim_t': sim_t, 'actors': rows},
                                 separators=(',', ':')) + '\n')

    def rows(self, world, director=None) -> list:
        """Every vehicle and walker now (the snapshot of the frame just ticked), as actors.jsonl rows."""
        from ..core.groundtruth import category_of
        snap = world.get_snapshot()
        rows = []
        # a log replay (workers.replay) parks the actors it is not showing: left out; and knows its exact poses
        hidden = director.hidden_ids() if director is not None and hasattr(director, 'hidden_ids') else ()
        exact = getattr(director, 'exact_tf', None) if director is not None else None
        subclass = getattr(self.w, 'subclass', None) or {}
        proxies = getattr(self.w, 'proxies', None) or {}
        for a in world.get_actors():
            tid = a.type_id
            if (not tid.startswith(('vehicle.', 'walker.')) and a.id not in proxies) or a.id in hidden:
                continue
            st = self.static.get(a.id)
            if st is None:
                try:
                    bb = a.bounding_box
                    attrs = dict(getattr(a, 'attributes', {}) or {})
                except RuntimeError:
                    continue
                ext = [bb.extent.x, bb.extent.y, bb.extent.z]
                ctr = [bb.location.x, bb.location.y, bb.location.z]
                if a.id in proxies:              # a proxy's true box (a scaled mesh's reported centre is unscaled)
                    ctr, ext = proxies[a.id]['center'], proxies[a.id]['extent']
                st = self.static[a.id] = {'type_id': tid, 'category': category_of(tid, attrs),
                                          'role': attrs.get('role_name', ''),
                                          'extent': [round(float(x), 4) for x in ext],
                                          'bbox_center': [round(float(x), 4) for x in ctr]}
                if subclass.get(a.id):
                    st['subclass'] = subclass[a.id]
            sa = snap.find(a.id)
            if sa is None:
                continue
            t, v = (exact(a) if exact is not None else None) or sa.get_transform(), None
            if director is not None and director.kinematic:
                v = director.k_vel(a)
            v = v if v is not None else sa.get_velocity()
            rows.append(dict({'id': a.id, 'x': round(t.location.x, 4), 'y': round(t.location.y, 4),
                              'z': round(t.location.z, 4), 'roll': round(t.rotation.roll, 4),
                              'pitch': round(t.rotation.pitch, 4), 'yaw': round(t.rotation.yaw, 4),
                              'vx': round(v.x, 4), 'vy': round(v.y, 4), 'vz': round(v.z, 4)}, **st))
        return rows

    def close(self):
        if self.fh is not None:
            self.fh.close()
