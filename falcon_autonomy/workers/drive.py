"""drive: one recorded drive in CARLA — synchronous mode at a fixed step, the ego and traffic on the traffic manager,
the rig's cameras with their label cameras at the same pose and tick.

    python -m falcon_autonomy.workers drive CONFIG.json

CONFIG.json (falcon_autonomy.core.drive.drive_config writes it; docs/protocol.md lists every field) says the server,
the map, the weather, the rig, the traffic, the seed, the step and how long. The run:

1. connects; loads the map when another is open; puts the world and the traffic manager in synchronous mode at
   fixed_delta_seconds = 1 / fps; seeds the traffic manager, the pedestrians and the spawn choices; sets the weather;
2. spawns the ego (role 'hero') at a seeded spawn point and the traffic (role falcon_traffic) at others, all on the
   traffic manager's autopilot, then each camera (role falcon_sensor) and, beside it, one CARLA camera per label
   kind it records (semantic segmentation for 'semantic'), same size, field of view, pose and tick;
3. warms up warmup_steps steps unrecorded (the first frames after a spawn are not settled), then records `steps`
   steps (or until max_wall_s of wall clock, for the smoke drive): per step every camera's frame of exactly that
   tick (by frame number) — RGB as JPEG, semantic tags as a single-channel PNG — and a line in frames.jsonl with the
   step, frame, simulation time and the ego's pose and speed;
4. writes dataset.json (the sensors' intrinsics and extrinsics, the label kinds, the palette) and run.json (what
   ran, the measured throughput), and tidies up in all cases (common.Worker.tidy).

A cancelled or failed drive leaves only its run.json (never half a dataset). Throughput is measured over the recorded
steps only: wall seconds per step, rendered Mpx per step, the real-time factor.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import queue
import time

from ..core import protocol as P
from ..core.budget import check_budget, label_cameras, rig_mpx
from ..core.provenance import read_json, sha256_json
from .common import ROLE_HERO, ROLE_SENSOR, ROLE_TRAFFIC, Failed, Worker

DRIVE_SCHEMA = 'falcon.drive_config/1'
DATASET_SCHEMA = 'falcon.dataset/1'
MAX_IN_FLIGHT = 8                       # frames being encoded; beyond, the drive waits for them
FRAME_TIMEOUT_S = 10.0


class Drive(Worker):
    name = 'drive'

    def __init__(self, cfg: dict, argv: list, cfg_path: str = ''):
        super().__init__(argv, cfg['out'], cfg.get('timeout', 20.0), cfg.get('seed', 0), cfg,
                         {'config': cfg_path} if cfg_path else None)
        self.cfg = cfg
        self.remove_outputs_on_cancel = True
        self.queues: dict = {}
        self.cams: list = []            # [(sensor dict, {kind: carla sensor})]
        self.pool = cf.ThreadPoolExecutor(max_workers=4)
        self.pending: list = []

    # ------------------------------------------------------------------ setup
    def body(self) -> dict:
        cfg = self.cfg
        if cfg.get('schema') != DRIVE_SCHEMA:
            raise Failed(f'not a drive config (schema {cfg.get("schema")!r}, expected {DRIVE_SCHEMA})')
        ok, why = check_budget(cfg['rig'], cfg.get('render_budget_mpx', 200.0))
        if not ok:
            raise Failed(why, P.REFUSED)
        carla = self.import_carla()
        import numpy as np
        self.np = np
        self.emit.progress(0.0, 'connecting', force=True)
        client, world = self.connect(cfg.get('host', '127.0.0.1'), cfg['port'], cfg.get('connect_wait_s', 60.0))
        town = cfg.get('town')
        if town and world.get_map().name.rpartition('/')[2] != town:
            self.emit.progress(0.01, f'loading {town}', force=True)
            self.expect(cfg.get('load_timeout_s', 120.0))
            client.set_timeout(cfg.get('load_timeout_s', 120.0))
            world = client.load_world(town)
            client.set_timeout(self.timeout)
            self.world = world
            self.alive()
        self.check()
        seed = int(cfg.get('seed', 0))
        rng = np.random.default_rng(seed)
        fps = float(cfg.get('fps', 20.0))
        st = world.get_settings()
        st.synchronous_mode = True
        st.fixed_delta_seconds = 1.0 / fps
        world.apply_settings(st)
        self.tm = client.get_trafficmanager(int(cfg.get('tm_port', 8310)))
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(seed)
        if hasattr(world, 'set_pedestrians_seed'):
            world.set_pedestrians_seed(seed)
        self.apply_weather(world, cfg.get('weather') or {})
        world.tick()
        self.alive()
        lib = world.get_blueprint_library()
        spawns = list(world.get_map().get_spawn_points())
        if not spawns:
            raise Failed(f'the map {town} has no spawn points')
        order = list(rng.permutation(len(spawns)))
        ego = self.spawn_ego(world, lib, spawns, order)
        self.spawn_traffic(world, lib, spawns, order, rng)
        self.spawn_cameras(world, lib, ego)
        self.check()
        os.makedirs(self.out_dir, exist_ok=True)

        # ------------------------------------------------------------ warm-up and recording
        self.phase('run')
        warm = int(cfg.get('warmup_steps', 10))
        for k in range(warm):
            self.check()
            frame = world.tick()
            self.alive()
            self.drain(frame)
            self.emit.progress(0.02 + 0.03 * (k + 1) / max(warm, 1), 'warming up')
        steps = int(cfg.get('steps', 200))
        max_wall = float(cfg.get('max_wall_s', 0) or 0)
        mpx = rig_mpx(cfg['rig'])
        index = open(os.path.join(self.out_dir, 'frames.jsonl'), 'w', encoding='utf-8')
        t_rec = time.monotonic()
        n = 0
        sim_t0 = None
        late = 0
        try:
            for i in range(steps):
                self.check()
                frame = world.tick()
                self.alive()
                snap_t = world.get_snapshot().timestamp.elapsed_seconds
                sim_t0 = snap_t if sim_t0 is None else sim_t0
                rec = {'step': i, 'frame': int(frame), 'sim_t': round(snap_t - sim_t0, 6), 'files': {}}
                rec.update(self.ego_state(ego))
                for s, sensors in self.cams:
                    for kind, sensor in sensors.items():
                        data = self.get(sensor.id, frame)
                        rel = os.path.join(kind, s['id'], f'{i:06d}.{"jpg" if kind == "rgb" else "png"}')
                        rec['files'][f'{s["id"]}/{kind}'] = rel
                        self.submit_write(kind, data, os.path.join(self.out_dir, rel))
                index.write(json.dumps(rec, separators=(',', ':')) + '\n')
                n += 1
                wall = time.monotonic() - t_rec
                if max_wall and wall >= max_wall:
                    break
                f = (wall / max_wall) if max_wall else (i + 1) / steps
                self.emit.progress(0.05 + 0.9 * f, f'step {i + 1}{"" if max_wall else f" of {steps}"} '
                                                   f'({(i + 1) / max(wall, 1e-6):.1f} steps/s)')
                if len(self.pending) > MAX_IN_FLIGHT * max(len(self.cams), 1):
                    late += 1
                    self.flush(MAX_IN_FLIGHT)
            self.flush(0)
        finally:
            index.close()
        wall = time.monotonic() - t_rec
        sim_s = n / fps
        perf = {'steps': n, 'wall_s': round(wall, 3), 'steps_per_s': round(n / max(wall, 1e-6), 3),
                'sim_s': round(sim_s, 3), 'realtime_factor': round(sim_s / max(wall, 1e-6), 3),
                'mpx_per_step': round(mpx, 3), 'mpx_per_s': round(mpx * n / max(wall, 1e-6), 2),
                'camera_frames': n * len(self.cams), 'encoder_waits': late, 'source': 'measured'}
        for k in ('steps_per_s', 'realtime_factor', 'mpx_per_s'):
            self.emit.metric(k, perf[k], {'steps_per_s': '1/s', 'realtime_factor': '×', 'mpx_per_s': 'Mpx/s'}[k])
        self.write_dataset(n, fps)
        self.emit.progress(1.0, f'recorded {n} steps in {wall:.1f} s', force=True)
        return {'performance': perf, 'summary': {'steps': n, 'cameras': len(self.cams), 'town': town,
                                                  'steps_per_s': perf['steps_per_s'],
                                                  'realtime_factor': perf['realtime_factor']},
                'name': cfg.get('name', 'drive'), 'config_sha256': sha256_json(cfg)}

    def apply_weather(self, world, w: dict):
        carla = self.carla
        wp = carla.WeatherParameters()
        for k, v in w.items():
            if hasattr(wp, k):
                setattr(wp, k, float(v))
        world.set_weather(wp)

    def spawn_ego(self, world, lib, spawns, order):
        bp = lib.find(self.cfg['rig']['vehicle'])
        bp.set_attribute('role_name', ROLE_HERO)
        for k in list(order):
            ego = world.try_spawn_actor(bp, spawns[k])
            if ego is not None:
                order.remove(k)
                self.own(ego)
                ego.set_autopilot(True, self.tm.get_port())
                self.alive()
                return ego
        raise Failed('no free spawn point for the ego')

    def spawn_traffic(self, world, lib, spawns, order, rng):
        prof = self.cfg.get('traffic') or {}
        n = int(prof.get('vehicles', 0))
        if n <= 0:
            return
        fleet = {k: float(v) for k, v in (prof.get('fleet') or {'car': 1.0}).items() if float(v) > 0}
        by_type = {}
        for bp in lib.filter('vehicle.*'):
            base = bp.get_attribute('base_type').as_str() if bp.has_attribute('base_type') else 'car'
            by_type.setdefault(base.lower(), []).append(bp)
        kinds = [k for k in sorted(fleet) if by_type.get(k)]
        missing = sorted(set(fleet) - set(kinds))
        if missing:
            self.note(f'the server has no blueprints of base type {", ".join(missing)}: left out of the traffic')
        if not kinds:
            return
        w = self.np.array([fleet[k] for k in kinds])
        w = w / w.sum()
        beh = prof.get('behaviour') or {}
        placed = 0
        for k in list(order):
            if placed >= n:
                break
            kind = kinds[int(rng.choice(len(kinds), p=w))]
            cands = sorted(by_type[kind], key=lambda b: b.id)
            bp = cands[int(rng.integers(len(cands)))]
            bp.set_attribute('role_name', ROLE_TRAFFIC)
            a = world.try_spawn_actor(bp, spawns[k])
            if a is None:
                continue
            order.remove(k)
            self.own(a)
            a.set_autopilot(True, self.tm.get_port())
            self.apply_behaviour(a, beh)
            placed += 1
        self.alive()
        if placed < n:
            self.note(f'{placed} of {n} traffic vehicles placed (the map has {len(spawns)} spawn points)')

    def apply_behaviour(self, a, beh):
        tm = self.tm
        if 'distance_to_leading_m' in beh:
            tm.distance_to_leading_vehicle(a, float(beh['distance_to_leading_m']))
        if 'speed_difference_pct' in beh and hasattr(tm, 'vehicle_percentage_speed_difference'):
            tm.vehicle_percentage_speed_difference(a, float(beh['speed_difference_pct']))
        if 'ignore_lights_pct' in beh:
            tm.ignore_lights_percentage(a, float(beh['ignore_lights_pct']))
        if 'ignore_signs_pct' in beh:
            tm.ignore_signs_percentage(a, float(beh['ignore_signs_pct']))
        if 'auto_lane_change' in beh:
            tm.auto_lane_change(a, bool(beh['auto_lane_change']))

    def spawn_cameras(self, world, lib, ego):
        carla = self.carla
        for s in self.cfg['rig']['sensors']:
            if s['type'] != 'camera.pinhole':
                continue                                   # fisheye, lidar, radar, GNSS, IMU: falcon M3 (rig editor)
            p = s['pose']
            tf = carla.Transform(carla.Location(x=p['x'], y=p['y'], z=p['z']),
                                 carla.Rotation(pitch=p['pitch'], yaw=p['yaw'], roll=p['roll']))
            sensors = {}
            for kind, bp_id in [('rgb', 'sensor.camera.rgb')] + [
                    (k, 'sensor.camera.' + c) for k, c in self.label_kinds(s)]:
                bp = lib.find(bp_id)
                bp.set_attribute('image_size_x', str(s['width']))
                bp.set_attribute('image_size_y', str(s['height']))
                bp.set_attribute('fov', str(s['fov']))
                bp.set_attribute('sensor_tick', '0.0')
                bp.set_attribute('role_name', ROLE_SENSOR)
                cam = self.own(world.spawn_actor(bp, tf, attach_to=ego))
                q = queue.Queue()
                self.queues[cam.id] = q
                cam.listen(q.put)
                sensors[kind] = cam
            self.cams.append((s, sensors))
            self.alive()

    @staticmethod
    def label_kinds(s) -> list:
        """(output kind, CARLA camera) per label camera this sensor needs, in a fixed order."""
        cams = label_cameras(s.get('labels'))
        names = {'semantic_segmentation': 'semantic', 'instance_segmentation': 'instance', 'depth': 'depth'}
        return [(names[c], c) for c in cams]

    def ego_state(self, ego) -> dict:
        t = ego.get_transform()
        v = ego.get_velocity()
        return {'ego': {'x': round(t.location.x, 4), 'y': round(t.location.y, 4), 'z': round(t.location.z, 4),
                        'roll': round(t.rotation.roll, 4), 'pitch': round(t.rotation.pitch, 4),
                        'yaw': round(t.rotation.yaw, 4)},
                'speed_mps': round(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z), 4)}

    # ------------------------------------------------------------------ sensor data
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
                self.check()
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

    def submit_write(self, kind, data, path):
        np = self.np
        a = np.frombuffer(data.raw_data, np.uint8).reshape(data.height, data.width, 4).copy()
        self.pending.append(self.pool.submit(_write, kind, a, path))

    def flush(self, keep):
        while len(self.pending) > keep:
            self.pending.pop(0).result()

    def write_dataset(self, n, fps):
        from ..core.labels import TAGS
        from ..core.provenance import write_json
        sensors = []
        for s, cams in self.cams:
            f = s['width'] / (2.0 * math.tan(math.radians(s['fov']) / 2.0))
            sensors.append({'id': s['id'], 'name': s['name'], 'type': s['type'], 'width': s['width'],
                            'height': s['height'], 'fov': s['fov'],
                            'K': [[f, 0.0, s['width'] / 2.0], [0.0, f, s['height'] / 2.0], [0.0, 0.0, 1.0]],
                            'pose': s['pose'], 'streams': sorted(cams)})
        write_json(os.path.join(self.out_dir, 'dataset.json'), {
            'schema': DATASET_SCHEMA, 'layout': 'falcon-raw', 'name': self.cfg.get('name', 'drive'),
            'frames': n, 'fps': fps, 'town': self.cfg.get('town'), 'weather': self.cfg.get('weather'),
            'sensors': sensors,
            'labels': {'semantic': {'encoding': 'uint8 PNG, the CARLA 0.9.16 semantic tag per pixel',
                                    'classes': {str(k): {'name': v[0], 'color': v[1]} for k, v in TAGS.items()}}},
            'frames_index': 'frames.jsonl',
            'coordinates': 'CARLA world frame (x forward, y right, z up; left-handed), metres and degrees',
            'provenance': self.run['provenance']})


def _write(kind, bgra, path):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if kind == 'rgb':
        ok = cv2.imwrite(path, bgra[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 92])
    elif kind in ('semantic',):
        ok = cv2.imwrite(path, bgra[:, :, 2])            # R = the tag
    elif kind == 'instance':
        ok = cv2.imwrite(path, bgra[:, :, :3])           # R tag, G + 256·B actor id: kept losslessly
    else:                                                # depth: CARLA's 24-bit encoding, kept losslessly
        ok = cv2.imwrite(path, bgra[:, :, :3])
    if not ok:
        raise OSError(f'could not write {path}')


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker drive')
    ap.add_argument('config', help='drive config JSON (falcon_autonomy.core.drive)')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return Drive(cfg, argv, a.config).execute()
