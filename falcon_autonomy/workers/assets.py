"""assets: what CARLA's packaged build can spawn, measured on the server; and the checks behind Falcon's Indian road
assets (runs in CARLA's Python).

    python -m falcon_autonomy.workers assets CONFIG.json

CONFIG.json (schema falcon.assets/1): host, port, tm_port, out, mode:

* "inventory" — every blueprint of the server's library with its tags and attributes; every vehicle, walker and
  static prop spawned one at a time on the open map (synchronous mode, two steps settled) and its bounding box, its
  semantic tags (Actor.semantic_tags) and, for vehicles, its physics control (mass, wheels, maximum steering, the
  engine's peak torque and maximum rpm); static.prop.mesh spawned with mesh paths (listed props, meshes the package
  lists as no prop, a path that does not exist) to see what mesh-path spawning does in the packaged build. Writes
  inventory.json (core.assets.INVENTORY_SCHEMA). Nothing is left behind (each actor is destroyed by id before the
  next is spawned).
* "verify" — the checks docs/indian-assets.md reports, written to verify.json stage by stage (a server crash keeps
  what was measured before it):
  - "tm_speed" (on the open town): a stand-in vehicle on the traffic manager's autopilot with set_desired_speed vs one
    without: its speeds over the run (does a per-vehicle cap hold?);
  - "puddles_town" / "puddles_generated": a 640 × 360 camera's image with precipitation_deposits 0 and 100
    (wetness and rain 0): the mean absolute difference, overall and in the lower half (the road);
  - "mesh" (a straight test road generated from OpenDRIVE with the features of core.roadfeatures, at each mesh
    vertex distance asked): the road surface's height every 2 cm across each feature from world.ground_projection
    (a ray against the generated mesh's collision) against the profile written, and the map's waypoint z;
  - "drives": a vehicle driven by Falcon's speed and lane controller through CARLA's physics over the features at
    the speeds asked: its z, pitch and the IMU's vertical acceleration around each feature against the flat road;
  - "accel": a stand-in from rest at full throttle with its engine torque curve scaled (VehiclePhysicsControl), the
    peak acceleration over 0.5 s and the time to 30 km/h per scale;
  - "props": candidate proxies spawned on the road in front of an instance-segmentation camera: their pixels (by
    actor id), semantic tag and box; then moved 1.5 m with physics off (set_transform) and seen again;
  - "shape": a road with a lateralProfile <shape> raising its right lanes: the surface height across the road.
"""
from __future__ import annotations

import argparse
import math
import os
import time

from ..core import protocol as P
from ..core.provenance import read_json, write_json
from .common import ROLE_PROP, ROLE_SENSOR, ROLE_TRAFFIC, ROLE_WALKER, Failed, Worker

SCHEMA = 'falcon.assets/1'
MESH_TRIALS = (
    ('/Game/Carla/Static/Dynamic/Trash/SM_Box01.SM_Box01', 'a listed prop\'s mesh (Box01)'),
    ('/Game/Carla/Static/Dynamic/Street/Materials/FoodCart/SM_FoodCart__cart.SM_FoodCart__cart',
     'a listed prop\'s mesh (FoodCart)'),
    ('/Game/Carla/Static/Dynamic/Street/Materials/FoodCart/SM_FoodCart__umA.SM_FoodCart__umA',
     'a cooked mesh the package lists as no prop (a food-cart umbrella)'),
    ('/Game/Carla/Static/Other/SM_Plane.SM_Plane', 'a cooked mesh the package lists as no prop (a plane)'),
)
MISSING_MESH = '/Game/Falcon/DoesNotExist/SM_Nothing.SM_Nothing'


def _value(a):
    """An attribute's value: as_str() refuses some (0.9.16: a Bool attribute such as ros_publish_tf raises 'bad
    attribute cast: cannot convert to String'), so each typed reader is tried."""
    for f in ('as_str', 'as_bool', 'as_int', 'as_float'):
        try:
            return getattr(a, f)()
        except (RuntimeError, AttributeError, ValueError, TypeError):
            continue
    return None


def _attr_rows(bp) -> dict:
    out = {}
    for a in bp:
        out[a.id] = {'type': str(getattr(a, 'type', '')), 'value': _value(a),
                     'recommended': [str(v) for v in (getattr(a, 'recommended_values', None) or [])],
                     'modifiable': bool(getattr(a, 'is_modifiable', True))}
    return out


def _box(a) -> dict:
    bb = a.bounding_box
    return {'extent': [round(bb.extent.x, 4), round(bb.extent.y, 4), round(bb.extent.z, 4)],
            'center': [round(bb.location.x, 4), round(bb.location.y, 4), round(bb.location.z, 4)],
            'size_m': [round(2 * bb.extent.x, 3), round(2 * bb.extent.y, 3), round(2 * bb.extent.z, 3)]}


def _tags(a) -> list:
    try:
        return sorted(int(t) for t in (getattr(a, 'semantic_tags', None) or []))
    except (TypeError, ValueError, RuntimeError):
        return []


def _physics(v) -> dict:
    try:
        pc = v.get_physics_control()
    except (RuntimeError, AttributeError):
        return {}
    tc = [(float(p.x), float(p.y)) for p in (getattr(pc, 'torque_curve', None) or [])]
    wheels = list(getattr(pc, 'wheels', None) or [])
    return {'mass_kg': round(float(getattr(pc, 'mass', 0.0)), 2),
            'max_rpm': round(float(getattr(pc, 'max_rpm', 0.0)), 1),
            'peak_torque': round(max((y for _, y in tc), default=0.0), 2), 'torque_curve': tc,
            'wheels': len(wheels), 'wheel_radius_cm': [round(float(getattr(w, 'radius', 0.0)), 2) for w in wheels],
            'max_steer_deg': round(max((float(getattr(w, 'max_steer_angle', 0.0)) for w in wheels), default=0.0), 2)}


class Assets(Worker):
    name = 'assets'

    def __init__(self, cfg, argv, cfg_path=''):
        super().__init__(argv, cfg['out'], cfg.get('timeout', 60.0), 0, cfg,
                         {'config': cfg_path} if cfg_path else None)
        self.cfg = cfg
        self.doc = {}

    # ------------------------------------------------------------------ helpers
    def save(self, name):
        write_json(os.path.join(self.out_dir, name), self.doc)

    def sync(self, world, dt=0.05):
        st = world.get_settings()
        st.synchronous_mode = True
        st.fixed_delta_seconds = dt
        world.apply_settings(st)
        world.tick()
        self.alive()

    def gone(self, *actors):
        """Destroy these actors by id in a batch (never on the autopilot) and forget them."""
        ids = [a.id for a in actors if a is not None]
        if not ids:
            return
        self.client.apply_batch_sync([self.carla.command.DestroyActor(i) for i in ids], False)
        self.spawned = [a for a in self.spawned if a.id not in ids]
        self.alive()

    def spawn(self, world, bp, tf, role=ROLE_PROP, attach=None):
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', role)
        a = world.try_spawn_actor(bp, tf, attach_to=attach) if attach is not None else world.try_spawn_actor(bp, tf)
        self.alive()
        return self.own(a) if a is not None else None

    def ticks(self, world, n):
        for _ in range(n):
            self.check()
            world.tick()
        self.alive()

    # ------------------------------------------------------------------ body
    def body(self):
        cfg = self.cfg
        if cfg.get('schema') != SCHEMA:
            raise Failed(f'not an assets config (schema {cfg.get("schema")!r})')
        self.import_carla()
        self.connect(cfg.get('host', '127.0.0.1'), cfg['port'], cfg.get('connect_wait_s', 120.0))
        os.makedirs(self.out_dir, exist_ok=True)
        if cfg.get('mode') == 'inventory':
            return self.inventory()
        if cfg.get('mode') == 'verify':
            return self.verify()
        raise Failed(f'unknown mode {cfg.get("mode")!r} (inventory or verify)')

    # ================================================================== inventory
    def inventory(self):
        carla, world = self.carla, self.world
        self.phase('run')
        self.sync(world)
        lib = world.get_blueprint_library()
        cmap = world.get_map()
        spawns = list(cmap.get_spawn_points())
        if not spawns:
            raise Failed('the open map has no spawn points to measure on')
        tf0 = spawns[0]
        ground = carla.Transform(carla.Location(tf0.location.x, tf0.location.y, tf0.location.z - 0.5), tf0.rotation)
        self.doc = {'schema': 'falcon.carla_inventory/1', 'server': self.client.get_server_version(),
                    'client': self.client.get_client_version(), 'map': cmap.name.rpartition('/')[2],
                    'measured_at': [round(tf0.location.x, 2), round(tf0.location.y, 2), round(tf0.location.z, 2)],
                    'blueprints': [], 'vehicles': [], 'walkers': [], 'props': [], 'mesh_spawning': {}}
        bps = sorted(lib, key=lambda b: b.id)
        for b in bps:
            self.doc['blueprints'].append({'id': b.id, 'tags': sorted(str(t) for t in b.tags),
                                           'attributes': _attr_rows(b)})
        self.save('inventory.json')
        todo = [b for b in bps if b.id.startswith(('vehicle.', 'walker.', 'static.prop.'))
                and b.id != 'static.prop.mesh']
        for k, b in enumerate(todo):
            self.check()
            self.emit.progress(0.05 + 0.8 * k / len(todo), f'measuring {b.id}')
            row = {'id': b.id}
            if b.id.startswith('vehicle.'):
                for key in ('base_type', 'special_type', 'number_of_wheels', 'generation', 'has_dynamic_doors',
                            'has_lights', 'object_type'):
                    if b.has_attribute(key):
                        row[key] = _value(b.get_attribute(key))
                a = self.spawn(world, b, carla.Transform(carla.Location(tf0.location.x, tf0.location.y,
                                                                        tf0.location.z + 0.3), tf0.rotation),
                               ROLE_TRAFFIC)
            elif b.id.startswith('walker.'):
                for key in ('age', 'gender', 'can_use_wheelchair', 'speed', 'generation'):
                    if b.has_attribute(key):
                        row[key] = _value(b.get_attribute(key))
                if b.has_attribute('is_invincible'):
                    b.set_attribute('is_invincible', 'false')
                a = self.spawn(world, b, carla.Transform(carla.Location(tf0.location.x, tf0.location.y,
                                                                        tf0.location.z + 1.0), tf0.rotation),
                               ROLE_WALKER)
            else:
                if b.has_attribute('size'):
                    row['size_class'] = _value(b.get_attribute('size'))
                a = self.spawn(world, b, carla.Transform(ground.location, carla.Rotation()), ROLE_PROP)
            if a is None:
                row['spawned'] = False
            else:
                self.ticks(world, 2)
                row.update(spawned=True, box=_box(a), semantic_tags=_tags(a))
                if b.id.startswith('vehicle.'):
                    row['physics'] = _physics(a)
                self.gone(a)
                world.tick()
            self.doc['vehicles' if b.id.startswith('vehicle.') else 'walkers' if b.id.startswith('walker.')
                     else 'props'].append(row)
        self.save('inventory.json')
        self.emit.progress(0.88, 'mesh-path spawning', force=True)
        self.doc['mesh_spawning'] = self.mesh_trials(world, lib, ground)
        self.save('inventory.json')
        counts = {k: len(self.doc[k]) for k in ('blueprints', 'vehicles', 'walkers', 'props')}
        return {'summary': dict(counts, mesh=self.doc['mesh_spawning'].get('verdict'))}

    def mesh_trials(self, world, lib, ground) -> dict:
        carla = self.carla
        out = {'blueprint': None, 'trials': []}
        try:
            bp0 = lib.find('static.prop.mesh')
        except (IndexError, RuntimeError):
            out['verdict'] = 'no static.prop.mesh blueprint'
            return out
        out['blueprint'] = _attr_rows(bp0)
        trials = [(p, why, None) for p, why in MESH_TRIALS]
        if bp0.has_attribute('scale'):
            trials.append((MESH_TRIALS[0][0], 'the Box01 mesh with scale 2.0', '2.0'))
        trials.append((MISSING_MESH, 'a path that does not exist', None))
        for path, why, scale in trials:
            self.check()
            bp = lib.find('static.prop.mesh')
            row = {'mesh_path': path, 'what': why}
            try:
                bp.set_attribute('mesh_path', path)
                if scale is not None:
                    bp.set_attribute('scale', scale)
                    row['scale'] = float(scale)
            except (IndexError, RuntimeError) as e:
                row['error'] = f'set_attribute: {e}'
                out['trials'].append(row)
                continue
            try:
                a = world.try_spawn_actor(bp, carla.Transform(ground.location, carla.Rotation()))
            except RuntimeError as e:
                a, row['error'] = None, str(e)
            self.alive()
            if a is None:
                row['spawned'] = False
            else:
                self.own(a)
                self.ticks(world, 2)
                row.update(spawned=True, type_id=a.type_id, box=_box(a), semantic_tags=_tags(a))
                self.gone(a)
                world.tick()
            out['trials'].append(row)
            self.save('inventory.json')
        ok = [r for r in out['trials'] if r.get('spawned') and max(r['box']['size_m']) > 0.01]
        out['verdict'] = (f'{len(ok)} of {len(out["trials"])} mesh paths spawned with a non-empty box: '
                          + ', '.join(r['what'] for r in ok)) if ok else 'no mesh path spawned a visible mesh'
        return out

    # ================================================================== verify
    def verify(self):
        cfg = self.cfg
        self.phase('run')
        self.doc = {'schema': 'falcon.assets_verify/1', 'server': self.client.get_server_version(), 'stages': {}}
        stages = cfg.get('stages') or ['tm_speed', 'puddles_town', 'mesh', 'drives', 'accel', 'props',
                                       'puddles_generated', 'shape']
        for st in stages:
            self.check()
            self.emit.progress(0.02 + 0.9 * stages.index(st) / len(stages), f'stage {st}', force=True)
            t0 = time.monotonic()
            res = getattr(self, 'v_' + st)()
            res['wall_s'] = round(time.monotonic() - t0, 2)
            self.doc['stages'][st] = res
            self.save('verify.json')
        return {'summary': {'stages': list(self.doc['stages'])}}

    # ---- the town
    def v_tm_speed(self):
        carla, world = self.carla, self.world
        self.sync(world, 0.05)
        tm = self.tm = self.client.get_trafficmanager(int(self.cfg.get('tm_port', 8310)))
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(0)
        lib = world.get_blueprint_library()
        spawns = list(world.get_map().get_spawn_points())
        bpid = self.cfg.get('stand_in', 'vehicle.micro.microlino')
        cap = float(self.cfg.get('cap_kmh', 20.0))
        cars = []
        for k in range(len(spawns)):
            if len(cars) == 2:
                break
            a = self.spawn(world, lib.find(bpid), spawns[(k * 7) % len(spawns)], ROLE_TRAFFIC)
            if a is not None:
                cars.append(a)
        if len(cars) < 2:
            raise Failed('could not place two stand-ins on the town')
        for a in cars:
            a.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(cars[0], cap)
        sp = {0: [], 1: []}
        lim = {0: [], 1: []}
        for i in range(int(self.cfg.get('tm_steps', 800))):
            self.check()
            world.tick()
            for j, a in enumerate(cars):
                v = a.get_velocity()
                sp[j].append(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) * 3.6)
                if i % 20 == 0:
                    lim[j].append(float(a.get_speed_limit()))
        self.alive()
        for a in cars:
            a.set_autopilot(False, tm.get_port())
        world.tick()
        self.gone(*cars)

        def stats(v):
            import numpy as np
            v = np.asarray(v[100:])                       # after the first 5 s
            return {'max_kmh': round(float(v.max()), 2), 'p95_kmh': round(float(np.percentile(v, 95)), 2),
                    'mean_moving_kmh': round(float(v[v > 2].mean()), 2) if (v > 2).any() else 0.0}
        return {'blueprint': bpid, 'cap_kmh': cap, 'steps': len(sp[0]), 'dt': 0.05,
                'capped': stats(sp[0]), 'free': stats(sp[1]),
                'speed_limits_seen_kmh': sorted({round(x, 1) for x in lim[0] + lim[1]})}

    def camera(self, world, tf, kind='rgb', w=640, h=360, fov=90.0):
        import queue
        lib = world.get_blueprint_library()
        bp = lib.find('sensor.camera.' + kind)
        bp.set_attribute('image_size_x', str(w))
        bp.set_attribute('image_size_y', str(h))
        bp.set_attribute('fov', str(fov))
        bp.set_attribute('role_name', ROLE_SENSOR)
        cam = self.own(world.spawn_actor(bp, tf))
        q = queue.Queue()
        cam.listen(q.put)
        return cam, q

    def grab(self, world, q, n=1):
        import numpy as np
        d = None
        for _ in range(n):
            frame = world.tick()
            while True:
                d = q.get(timeout=20.0)
                if d.frame >= frame:
                    break
        self.alive()
        return np.frombuffer(d.raw_data, np.uint8).reshape(d.height, d.width, 4).copy()

    def puddles(self, tf, tag):
        import cv2
        import numpy as np
        carla, world = self.carla, self.world
        self.sync(world, 0.05)
        cam, q = self.camera(world, tf)
        imgs = {}
        for dep in (0.0, 100.0):
            wp = carla.WeatherParameters(cloudiness=5.0, precipitation=0.0, precipitation_deposits=dep,
                                         wind_intensity=0.0, sun_azimuth_angle=-1.0, sun_altitude_angle=45.0,
                                         wetness=0.0, fog_density=0.0)
            world.set_weather(wp)
            imgs[dep] = self.grab(world, q, 30)[:, :, :3].astype(np.int16)
            cv2.imwrite(os.path.join(self.out_dir, f'puddles_{tag}_{int(dep)}.jpg'), imgs[dep].astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        cam.stop()
        cam.destroy()
        self.spawned = [a for a in self.spawned if a.id != cam.id]
        d = np.abs(imgs[100.0] - imgs[0.0]).mean(axis=2)
        h = d.shape[0]
        return {'camera': [tf.location.x, tf.location.y, tf.location.z, tf.rotation.pitch, tf.rotation.yaw],
                'mean_abs_diff': round(float(d.mean()), 3),
                'lower_half_mean_abs_diff': round(float(d[h // 2:].mean()), 3),
                'changed_share_gt10': round(float((d[h // 2:] > 10).mean()), 4),
                'images': [f'puddles_{tag}_0.jpg', f'puddles_{tag}_100.jpg']}

    def v_puddles_town(self):
        carla = self.carla
        sp = list(self.world.get_map().get_spawn_points())[0]
        tf = carla.Transform(carla.Location(sp.location.x, sp.location.y, sp.location.z + 2.0),
                             carla.Rotation(pitch=-15.0, yaw=sp.rotation.yaw))
        r = self.puddles(tf, 'town')
        r['map'] = self.world.get_map().name.rpartition('/')[2]
        return r

    # ---- the generated test road
    def generate(self, text, vertex_distance, nav=False):
        carla = self.carla
        params = carla.OpendriveGenerationParameters(
            vertex_distance=float(vertex_distance), max_road_length=500.0, wall_height=0.0, additional_width=0.6,
            smooth_junctions=True, enable_mesh_visibility=True, enable_pedestrian_navigation=bool(nav))
        self.expect(240.0)
        self.client.set_timeout(240.0)
        t0 = time.monotonic()
        self.world = self.client.generate_opendrive_world(text, params)
        self.client.set_timeout(self.timeout)
        self.alive()
        self.sync(self.world, float(self.cfg.get('dt', 0.02)))
        return round(time.monotonic() - t0, 2)

    def road_text(self):
        from ..core import roadfeatures as RF
        feats = self.cfg.get('features') or [{'kind': 'irc_hump', 's': 100.0}, {'kind': 'osm_bump', 's': 160.0},
                                             {'kind': 'table', 's': 220.0}, {'kind': 'dip', 's': 280.0}]
        text, placed, skipped = RF.apply(RF.test_road(float(self.cfg.get('road_m', 360.0))),
                                         [dict(f, road='1') for f in feats])
        if skipped:
            raise Failed(f'test features not placed: {skipped}')
        return text, placed

    def surface(self, x0, x1, y, step=0.02):
        """(x, z) of the generated mesh along a line at CARLA y, from rays straight down (ground_projection)."""
        carla, world = self.carla, self.world
        out = []
        x = x0
        while x <= x1 + 1e-9:
            p = world.ground_projection(carla.Location(x=x, y=y, z=2.0), 5.0)
            out.append((round(x, 3), round(p.location.z, 4) if p is not None else None,
                        int(p.label) if p is not None else None))
            x += step
        self.alive()
        return out

    def v_mesh(self):
        from ..core import roadfeatures as RF
        text, placed = self.road_text()
        with open(os.path.join(self.out_dir, 'test_road.xodr'), 'w', encoding='utf-8') as fh:
            fh.write(text)
        res = {'features': placed, 'by_vertex_distance': {}}
        for vd in self.cfg.get('vertex_distances') or [2.0, 0.5, 0.25, 0.1]:
            gen = self.generate(text, vd)
            cmap = self.world.get_map()
            rows = []
            for f in placed:
                L = f['length_m']
                xs = self.surface(f['s'] - L / 2 - 1.0, f['s'] + L / 2 + 1.0, 1.75)
                import numpy as np
                good = [(x, z) for x, z, _ in xs if z is not None]
                X = np.array([g[0] for g in good])
                Z = np.array([g[1] for g in good])
                flat = Z[(X < f['s'] - L / 2 - 0.3) | (X > f['s'] + L / 2 + 0.3)]
                z0 = float(np.median(flat)) if len(flat) else 0.0
                want = RF.height_at(RF.profile_entries([{'kind': f['kind'], 's': f['s']}]), X)
                peak = float((Z - z0).max() if f['height_m'] > 0 else (Z - z0).min())
                wps = [cmap.get_waypoint(self.carla.Location(x=float(x), y=1.75, z=0.0)) for x in
                       (f['s'] - L / 4, f['s'], f['s'] + L / 4)]
                rows.append({'kind': f['kind'], 's': f['s'], 'written_height_m': f['height_m'], 'length_m': L,
                             'surface_peak_m': round(peak, 4), 'surface_baseline_z': round(z0, 4),
                             'rms_error_m': round(float(np.sqrt(np.mean((Z - z0 - want) ** 2))), 4),
                             'hit_share': round(len(good) / max(len(xs), 1), 3),
                             'labels': sorted({lab for _, _, lab in xs if lab is not None}),
                             'waypoint_z': [round(w.transform.location.z, 4) if w is not None else None for w in wps],
                             'profile': [[x, z] for x, z, _ in xs][::5]})
            res['by_vertex_distance'][str(vd)] = {'generate_s': gen, 'features': rows}
            self.doc['stages']['mesh'] = res
            self.save('verify.json')
        return res

    def drive_over(self, bpid, kmh, x0=20.0, x1=330.0, y=1.75):
        """Falcon's speed (PI) and lane (pure pursuit on the line y) control through CARLA's physics along +x: rows
        (t, x, y, z, pitch, roll, speed m/s, imu accel z)."""
        import queue
        carla, world = self.carla, self.world
        lib = world.get_blueprint_library()
        dt = world.get_settings().fixed_delta_seconds
        v = self.spawn(world, lib.find(bpid), carla.Transform(carla.Location(x=x0, y=y, z=0.6), carla.Rotation()),
                       ROLE_TRAFFIC)
        if v is None:
            raise Failed(f'{bpid} could not be placed on the test road')
        ib = lib.find('sensor.other.imu')
        ib.set_attribute('role_name', ROLE_SENSOR)
        imu = self.own(world.spawn_actor(ib, carla.Transform(), attach_to=v))
        q = queue.Queue()
        imu.listen(q.put)
        self.ticks(world, 30)                              # settle on the road
        target = kmh / 3.6
        rows, ierr = [], 0.0
        t = 0.0
        for _ in range(int((x1 - x0) / max(target, 1.0) / dt * 1.6) + 200):
            self.check()
            tr = v.get_transform()
            vel = v.get_velocity()
            spd = math.hypot(vel.x, vel.y)
            if tr.location.x >= x1:
                break
            yaw = math.radians(tr.rotation.yaw)
            la = max(4.0, 0.8 * spd)
            dx, dy = la, y - tr.location.y
            fy = -math.sin(yaw) * dx + math.cos(yaw) * dy
            fx = math.cos(yaw) * dx + math.sin(yaw) * dy
            Lw = 2.0 * max(v.bounding_box.extent.x, 0.6) * 0.6
            steer = math.atan(Lw * 2.0 * fy / max(fx * fx + fy * fy, 1e-6)) / math.radians(70.0)
            err = target - spd
            ierr = max(min(ierr + err * dt, 5.0), -5.0)
            u = 0.6 * err + 0.15 * ierr
            ctrl = carla.VehicleControl(throttle=min(max(u, 0.0), 1.0), brake=min(max(-u, 0.0), 1.0),
                                        steer=max(min(steer, 1.0), -1.0))
            v.apply_control(ctrl)
            frame = world.tick()
            t += dt
            az = None
            while True:
                try:
                    d = q.get(timeout=5.0)
                except queue.Empty:
                    break
                if d.frame >= frame:
                    az = d.accelerometer.z
                    break
            tr = v.get_transform()
            rows.append((round(t, 3), round(tr.location.x, 4), round(tr.location.y, 4), round(tr.location.z, 5),
                         round(tr.rotation.pitch, 4), round(tr.rotation.roll, 4), round(spd, 4),
                         round(az, 4) if az is not None else None))
        self.alive()
        imu.stop()
        imu.destroy()
        self.spawned = [a for a in self.spawned if a.id != imu.id]
        self.gone(v)
        world.tick()
        return rows

    def v_drives(self):
        import numpy as np
        text, placed = self.road_text()
        out = {'runs': []}
        for vd in self.cfg.get('drive_vertex_distances') or [0.25, 2.0]:
            self.generate(text, vd)
            for bpid in self.cfg.get('drive_vehicles') or ['vehicle.micro.microlino', 'vehicle.lincoln.mkz_2020']:
                for kmh in self.cfg.get('drive_speeds_kmh') or [15.0, 30.0]:
                    rows = self.drive_over(bpid, kmh)
                    A = np.array([[np.nan if c is None else c for c in r] for r in rows], float)
                    run = {'vertex_distance': vd, 'blueprint': bpid, 'set_kmh': kmh, 'steps': len(rows), 'features': []}
                    for f in placed:
                        L = f['length_m']
                        near = (A[:, 1] > f['s'] - L / 2 - 6.0) & (A[:, 1] < f['s'] + L / 2 + 8.0)
                        flat = (A[:, 1] > f['s'] - 30.0) & (A[:, 1] < f['s'] - L / 2 - 8.0)
                        if not near.any() or not flat.any():
                            continue
                        z0, p0 = np.median(A[flat, 3]), np.median(A[flat, 4])
                        az0 = np.nanmedian(A[flat, 7])
                        run['features'].append({
                            'kind': f['kind'], 's': f['s'], 'height_m': f['height_m'],
                            'speed_kmh': round(float(np.median(A[near, 6])) * 3.6, 2),
                            'z_rise_m': round(float(A[near, 3].max() - z0), 4),
                            'z_drop_m': round(float(A[near, 3].min() - z0), 4),
                            'pitch_range_deg': round(float(A[near, 4].max() - A[near, 4].min()), 3),
                            'pitch_max_abs_deg': round(float(np.abs(A[near, 4] - p0).max()), 3),
                            'flat_pitch_sd_deg': round(float(A[flat, 4].std()), 4),
                            'flat_z_sd_m': round(float(A[flat, 3].std()), 5),
                            'imu_az_peak_dev_mps2': round(float(np.nanmax(np.abs(A[near, 7] - az0))), 3),
                            'flat_imu_az_sd_mps2': round(float(np.nanstd(A[flat, 7])), 3)})
                    out['runs'].append(run)
                    np.save(os.path.join(self.out_dir, f'drive_{vd}_{bpid.split(".")[-1]}_{int(kmh)}.npy'), A)
                    self.doc['stages']['drives'] = out
                    self.save('verify.json')
        return out

    def v_accel(self):
        import numpy as np
        from ..core import roadfeatures as RF
        carla, world = self.carla, self.world
        self.generate(RF.test_road(float(self.cfg.get('road_m', 360.0))), 2.0)
        world = self.world
        lib = world.get_blueprint_library()
        dt = world.get_settings().fixed_delta_seconds
        out = {'runs': []}
        for bpid in self.cfg.get('accel_vehicles') or ['vehicle.micro.microlino']:
            for scale in self.cfg.get('torque_scales') or [1.0, 0.6, 0.4, 0.3, 0.2]:
                v = self.spawn(world, lib.find(bpid), carla.Transform(carla.Location(x=10.0, y=1.75, z=0.6),
                                                                      carla.Rotation()), ROLE_TRAFFIC)
                if v is None:
                    raise Failed(f'{bpid} could not be placed')
                self.ticks(world, 30)
                pc = v.get_physics_control()
                base = [(p.x, p.y) for p in pc.torque_curve]
                pc.torque_curve = [carla.Vector2D(x, y * float(scale)) for x, y in base]
                v.apply_physics_control(pc)
                self.ticks(world, 5)
                sp = []
                for _ in range(int(25.0 / dt)):
                    self.check()
                    v.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
                    world.tick()
                    vel = v.get_velocity()
                    sp.append(math.hypot(vel.x, vel.y))
                    if v.get_transform().location.x > 340.0:
                        break
                self.alive()
                self.gone(v)
                world.tick()
                sp = np.array(sp)
                k = max(int(round(0.5 / dt)), 1)
                acc = (sp[k:] - sp[:-k]) / (k * dt) if len(sp) > k else np.zeros(1)
                t30 = next((i * dt for i, s in enumerate(sp) if s >= 30 / 3.6), None)
                out['runs'].append({'blueprint': bpid, 'torque_scale': scale,
                                    'peak_accel_mps2': round(float(acc.max()), 3),
                                    't_0_30_kmh_s': round(t30, 2) if t30 is not None else None,
                                    'speed_after_s_kmh': {
                                        str(s): round(float(sp[min(int(s / dt), len(sp) - 1)]) * 3.6, 2)
                                        for s in (2, 5, 10, 20)},
                                    'max_kmh': round(float(sp.max()) * 3.6, 2)})
                self.doc['stages']['accel'] = out
                self.save('verify.json')
        return out

    def v_props(self):
        """Each candidate (a blueprint id, or {'id', 'mesh_path', 'scale', 'yaw', 'z'}) 10 m in front of an
        instance camera: its pixels by actor id, their tag (R), its box, the top of its mesh from a ray cast straight
        down (world.cast_ray: the real height, whatever the box says); then moved 1.5 m on and 1 m across with
        physics off, and seen again."""
        import numpy as np
        from ..core import roadfeatures as RF
        carla = self.carla
        self.generate(RF.test_road(120.0), 2.0)
        world = self.world
        lib = world.get_blueprint_library()
        cam_tf = carla.Transform(carla.Location(x=40.0, y=1.75, z=1.6), carla.Rotation(pitch=-5.0))
        cam, q = self.camera(world, cam_tf, 'instance_segmentation')
        out = {'camera': 'instance segmentation 640 × 360, fov 90, 10 m behind the object, 1.6 m up', 'props': []}
        for item in self.cfg.get('props') or ['static.prop.foodcart', 'static.prop.shoppingcart']:
            self.check()
            spec = item if isinstance(item, dict) else {'id': item}
            bpid = 'static.prop.mesh' if spec.get('mesh_path') else spec['id']
            try:
                bp = lib.find(bpid)
            except (IndexError, RuntimeError):
                out['props'].append({'spec': spec, 'error': 'not in the library'})
                continue
            if spec.get('mesh_path'):
                bp.set_attribute('mesh_path', spec['mesh_path'])
                bp.set_attribute('scale', str(float(spec.get('scale', 1.0))))
            if bpid.startswith('walker.') and bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            z = 1.0 if bpid.startswith('walker.') else float(spec.get('z', 0.0))
            parent = None
            if spec.get('move') == 'parent':            # attached to an empty actor, which is what moves
                parent = self.spawn(world, lib.find('util.actor.empty'),
                                    carla.Transform(carla.Location(x=50.0, y=1.75, z=0.0)), ROLE_PROP)
                a = self.spawn(world, bp, carla.Transform(carla.Location(z=z), carla.Rotation(
                    yaw=float(spec.get('yaw', 0.0)))), ROLE_PROP, attach=parent)
            else:
                a = self.spawn(world, bp, carla.Transform(carla.Location(x=50.0, y=1.75, z=z),
                                                          carla.Rotation(yaw=float(spec.get('yaw', 0.0)))),
                               ROLE_WALKER if bpid.startswith('walker.') else ROLE_PROP)
            if a is None:
                out['props'].append({'spec': spec, 'spawned': False})
                continue
            img = self.grab(world, q, 10)
            ids = img[:, :, 1].astype(np.int32) + 256 * img[:, :, 0].astype(np.int32)
            mine = ids == (a.id & 0xFFFF)
            row = {'spec': spec, 'blueprint': bpid, 'actor_id': a.id, 'spawned': True, 'box': _box(a),
                   'semantic_tags': _tags(a), 'pixels': int(mine.sum()),
                   'pixel_tags': sorted({int(t) for t in np.unique(img[:, :, 2][mine])}) if mine.any() else []}
            if mine.any():
                ys, xs = np.nonzero(mine)
                row['bbox_px'] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            tops = []
            for dx in np.arange(-1.5, 1.51, 0.1):
                for dy in np.arange(-1.0, 1.01, 0.1):
                    hits = world.cast_ray(carla.Location(x=50.0 + dx, y=1.75 + dy, z=6.0),
                                          carla.Location(x=50.0 + dx, y=1.75 + dy, z=-1.0))
                    if hits:
                        tops.append((float(hits[0].location.z), int(hits[0].label), float(dx), float(dy)))
            self.alive()
            above = [t for t in tops if t[0] > 0.02]
            if above:
                row['ray_top_m'] = round(max(t[0] for t in above), 3)
                row['ray_labels'] = sorted({t[1] for t in above})
                row['ray_footprint_m'] = [round(max(t[2] for t in above) - min(t[2] for t in above) + 0.1, 2),
                                          round(max(t[3] for t in above) - min(t[3] for t in above) + 0.1, 2)]
            mover = parent if parent is not None else a
            if parent is None:
                mover.set_simulate_physics(False)       # (an empty actor refuses it: measured)
            t0 = mover.get_transform()
            to = carla.Location(t0.location.x + 1.5, t0.location.y - 1.0, t0.location.z)
            if spec.get('move') == 'set_location':
                mover.set_location(to)
            else:
                mover.set_transform(carla.Transform(to, t0.rotation))
            row['move'] = spec.get('move', 'set_transform')
            img = self.grab(world, q, 3)
            ids = img[:, :, 1].astype(np.int32) + 256 * img[:, :, 0].astype(np.int32)
            mine2 = ids == (a.id & 0xFFFF)
            t1 = a.get_transform()
            row['moved'] = {'to': [round(t1.location.x, 3), round(t1.location.y, 3)], 'pixels': int(mine2.sum())}
            if mine2.any():
                ys, xs = np.nonzero(mine2)
                row['moved']['bbox_px'] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            out['props'].append(row)
            self.gone(a, parent)
            world.tick()
            self.doc['stages']['props'] = out
            self.save('verify.json')
        cam.stop()
        cam.destroy()
        self.spawned = [a for a in self.spawned if a.id != cam.id]
        return out

    def v_puddles_generated(self):
        from ..core import roadfeatures as RF
        carla = self.carla
        self.generate(RF.test_road(120.0), 2.0)
        tf = carla.Transform(carla.Location(x=20.0, y=1.75, z=2.0), carla.Rotation(pitch=-15.0))
        return self.puddles(tf, 'generated')

    def v_shape(self):
        from ..core import roadfeatures as RF
        # the right side raised: a shape from t = -1 (into the right lane) rising linearly 0.10 m per metre
        text = RF.test_road(120.0, [], shape={'s': 0.0, 't': -1.0, 'a': 0.0, 'b': -0.10, 'c': 0.0, 'd': 0.0})
        gen = self.generate(text, 0.5)
        carla, world = self.carla, self.world
        across = []
        y = -5.0
        while y <= 5.0 + 1e-9:
            p = world.ground_projection(carla.Location(x=60.0, y=y, z=2.0), 5.0)
            across.append([round(y, 2), round(p.location.z, 4) if p is not None else None])
            y += 0.25
        self.alive()
        return {'generate_s': gen, 'shape': 'lateralProfile <shape s=0 t=-1 a=0 b=-0.10> (OpenDRIVE y; CARLA y = −t)',
                'surface_across_at_x60': across}


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker assets')
    ap.add_argument('config')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return Assets(cfg, argv, a.config).execute()
