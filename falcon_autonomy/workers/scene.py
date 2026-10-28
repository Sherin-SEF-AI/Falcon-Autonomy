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
from .common import ROLE_HERO, ROLE_SENSOR, ROLE_TRAFFIC, Failed

FRAME_TIMEOUT_S = 10.0
OPENDRIVE_DEFAULTS = {'vertex_distance': 2.0, 'max_road_length': 500.0, 'wall_height': 0.0, 'additional_width': 0.6,
                      'smooth_junctions': True, 'enable_mesh_visibility': True, 'enable_pedestrian_navigation': True}


def load_map(w, cfg: dict):
    """Make the server's world the config's: a town by name (loaded when another is open), or an OpenDRIVE world
    (cfg['xodr'], a file) generated with cfg['opendrive'] parameters — skipped when the server already has exactly
    that road network (its Map.to_opendrive() hashes the same). Returns the world."""
    client, world = w.client, w.world
    load_s = float(cfg.get('load_timeout_s', 180.0))
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
            w.alive()
            return ego
    raise Failed('no free spawn point for the ego')


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
        if beh.get('speed_spread_pct') and hasattr(w.tm, 'vehicle_percentage_speed_difference'):
            spread = float(beh['speed_spread_pct'])
            w.tm.vehicle_percentage_speed_difference(
                a, float(beh.get('speed_difference_pct', 0.0)) + float(rng.uniform(-spread, spread)))
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
                                              'truncated, occluded, range_m, pixels, box3d (vehicle frame), rle'},
                      'lidar': {'encoding': 'float32 x y z intensity (sensor frame); .labels.bin: uint8 tag then '
                                            'uint32 actor id per point'}},
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
                    np.save(path, np.frombuffer(d.raw_data, np.float32).reshape(-1, 4))
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
    was applied (run.json "faults_applied")."""

    def __init__(self, sensors, seed: int):
        import zlib
        import numpy as np
        self.f = {s['id']: list(s.get('faults') or []) for s in sensors if s.get('faults')}
        self.rng = {sid: np.random.default_rng([int(seed), zlib.crc32(sid.encode())]) for sid in self.f}
        self.last = {}
        self.applied = {sid: {} for sid in self.f}

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

    def image(self, sid, K, omega):
        from ..core import faults as F
        fl = [f for f in self.f.get(sid, []) if f.get('kind') in F.IMAGE_FAULTS]
        if not fl:
            return None
        for f in fl:
            self.count(sid, f['kind'])
        return fl, self.rng[sid], K, omega

    def stamp(self, sid, sim_t):
        from ..core import faults as F
        return sim_t + F.clock_offset_s(self.f.get(sid))

    def report(self) -> dict:
        return {sid: {'planned': self.f[sid], 'frames': self.applied[sid]} for sid in self.f}


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
    """actors.jsonl: every vehicle and walker at every recorded step, in CARLA's world frame, from the world snapshot
    of that frame — {step, frame, sim_t, actors: [{id, type_id, category, role, x, y, z, roll, pitch, yaw, vx, vy, vz,
    extent [ex, ey, ez], bbox_center [cx, cy, cz] (in the actor's frame)}]}. Boxes and categories are read once per
    actor (they do not change). Actors a kinematic director moves (physics off: CARLA reports them still) take the
    director's own velocity."""

    def __init__(self, w, path):
        self.w = w
        self.fh = open(path, 'w', encoding='utf-8')
        self.static = {}

    def step(self, world, i, frame, sim_t, director=None):
        import json
        from ..core.groundtruth import category_of
        snap = world.get_snapshot()
        rows = []
        for a in world.get_actors():
            tid = a.type_id
            if not tid.startswith(('vehicle.', 'walker.')):
                continue
            st = self.static.get(a.id)
            if st is None:
                try:
                    bb = a.bounding_box
                    attrs = dict(getattr(a, 'attributes', {}) or {})
                except RuntimeError:
                    continue
                st = self.static[a.id] = {'type_id': tid, 'category': category_of(tid, attrs),
                                          'role': attrs.get('role_name', ''),
                                          'extent': [round(bb.extent.x, 4), round(bb.extent.y, 4), round(bb.extent.z, 4)],
                                          'bbox_center': [round(bb.location.x, 4), round(bb.location.y, 4),
                                                          round(bb.location.z, 4)]}
            sa = snap.find(a.id)
            if sa is None:
                continue
            t, v = sa.get_transform(), None
            if director is not None and director.kinematic:
                v = director.k_vel(a)
            v = v if v is not None else sa.get_velocity()
            rows.append(dict({'id': a.id, 'x': round(t.location.x, 4), 'y': round(t.location.y, 4),
                              'z': round(t.location.z, 4), 'roll': round(t.rotation.roll, 4),
                              'pitch': round(t.rotation.pitch, 4), 'yaw': round(t.rotation.yaw, 4),
                              'vx': round(v.x, 4), 'vy': round(v.y, 4), 'vz': round(v.z, 4)}, **st))
        self.fh.write(json.dumps({'step': i, 'frame': int(frame), 'sim_t': sim_t, 'actors': rows},
                                 separators=(',', ':')) + '\n')

    def close(self):
        self.fh.close()
