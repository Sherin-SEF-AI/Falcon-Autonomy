"""drive: one recorded drive in CARLA — synchronous mode at a fixed step, the ego and traffic on the traffic manager,
the rig's cameras with their label cameras at the same pose and tick.

    python -m falcon_autonomy.workers drive CONFIG.json

CONFIG.json (falcon_autonomy.core.drive.drive_config writes it; docs/protocol.md lists every field) says the server,
the map (a town, or an OpenDRIVE file), the weather, the rig, the traffic, the seed, the step and how long. The run:

1. connects; makes the server's world the config's (scene.load_map); puts the world and the traffic manager in
   synchronous mode at fixed_delta_seconds = 1 / fps; seeds the traffic manager, the pedestrians and the spawn
   choices; sets the weather;
2. spawns the ego (role 'hero') at a seeded spawn point and the traffic (role falcon_traffic) at others, all on the
   traffic manager's autopilot, then each camera (role falcon_sensor) and, beside it, one CARLA camera per label
   kind it records (semantic segmentation for 'semantic'), same size, field of view, pose and tick;
3. warms up warmup_steps steps unrecorded (the first frames after a spawn are not settled), then records `steps`
   steps (or until max_wall_s of wall clock, for the smoke drive): per step every camera's frame of exactly that
   tick (by frame number) — RGB as JPEG, semantic tags as a single-channel PNG — and a line in frames.jsonl with the
   step, frame, simulation time and the ego's pose and speed;
4. writes dataset.json (the sensors' intrinsics and extrinsics, the label kinds, the palette) and run.json (what
   ran, the measured throughput), and tidies up in all cases (common.Worker.tidy).

A cancelled or failed drive leaves only its run.json and its input (never half a dataset). Throughput is measured
over the recorded steps only: wall seconds per step, rendered Mpx per step, the real-time factor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from ..core import protocol as P
from ..core.budget import check_budget, rig_mpx
from ..core.provenance import read_json, sha256_json, write_json
from . import camgt, scene
from .common import ROLE_SENSOR, Failed, Worker
from .lidar import LidarRecorder, mount_matrix

DRIVE_SCHEMA = 'falcon.drive_config/1'
MAX_IN_FLIGHT = 8                       # frames being encoded per camera; beyond, the drive waits for them


class Drive(Worker):
    name = 'drive'

    def __init__(self, cfg: dict, argv: list, cfg_path: str = ''):
        super().__init__(argv, cfg['out'], cfg.get('timeout', 20.0), cfg.get('seed', 0), cfg,
                         {'config': cfg_path, 'xodr': cfg.get('xodr')} if cfg_path else None)
        self.cfg = cfg
        self.remove_outputs_on_cancel = True
        self.keep_files = {'run.json', os.path.basename(cfg_path) if cfg_path else 'drive.json'}
        self.cameras = scene.Cameras(self)

    def stop_writers(self):
        self.cameras.stop()

    def body(self) -> dict:
        cfg = self.cfg
        if cfg.get('schema') != DRIVE_SCHEMA:
            raise Failed(f'not a drive config (schema {cfg.get("schema")!r}, expected {DRIVE_SCHEMA})')
        ok, why = check_budget(cfg['rig'], cfg.get('render_budget_mpx', 200.0))
        if not ok:
            raise Failed(why, P.REFUSED)
        carla = self.import_carla()
        import numpy as np
        self.emit.progress(0.0, 'connecting', force=True)
        self.connect(cfg.get('host', '127.0.0.1'), cfg['port'], cfg.get('connect_wait_s', 60.0))
        world = scene.load_map(self, cfg)
        self.check()
        seed = int(cfg.get('seed', 0))
        rng = np.random.default_rng(seed)
        fps = float(cfg.get('fps', 20.0))
        scene.synchronous(self, world, fps, seed, cfg.get('tm_port', 8310))
        world.set_weather(scene.weather_params(carla, cfg.get('weather')))
        world.tick()
        self.alive()
        lib = world.get_blueprint_library()
        spawns = list(world.get_map().get_spawn_points())
        if not spawns:
            raise Failed(f'the map {cfg.get("town") or cfg.get("xodr")} has no spawn points')
        order = list(rng.permutation(len(spawns)))
        rig_cams = [x for x in cfg['rig']['sensors'] if x['type'] in ('camera.pinhole', 'camera.fisheye')]
        need_gt = any(camgt.wants_gt(x) for x in rig_cams)
        level = camgt.level_boxes(self, world) if need_gt else []      # before any vehicle is spawned
        raster = camgt.LaneRaster(carla, world.get_map()) if cfg.get('xodr') and (
            need_gt or any(x['type'] == 'lidar' for x in cfg['rig']['sensors'])) else None
        ego = scene.spawn_ego(self, world, lib, spawns, order, cfg['rig']['vehicle'])
        scene.spawn_traffic(self, world, lib, spawns, order, rng, cfg.get('traffic') or {})
        scene.spawn_walkers(self, world, lib, rng, cfg.get('traffic') or {})
        cams = self.cameras
        from ..core import faults as F
        self.recorders = [camgt.CameraRecorder(self, world, lib, ego, x, cams, F.actual_pose(x['pose'], x.get('faults')))
                          for x in rig_cams]
        fps0 = float(cfg.get('fps', 20.0))
        self.lidars = [LidarRecorder(self, world, lib, ego, s, cams, ROLE_SENSOR, raster)
                       for s in cfg['rig']['sensors'] if s['type'] == 'lidar']
        for L in self.lidars:
            L.set_rate(fps0)
        self.others = scene.Others(self, world, lib, ego, cfg['rig']['sensors'], cams)
        # the range check: the first pinhole camera with ground truth (its depth view), against the first lidar
        depth_rec = next((r for r in self.recorders if r.gt and r.s['type'] == 'camera.pinhole'), None)
        depth_cam = depth_rec.s if depth_rec is not None else None
        M_cam_from_lidar = None
        if depth_cam is not None and self.lidars:
            M_cam_from_lidar = np.linalg.inv(mount_matrix(depth_cam['pose'])) @ mount_matrix(self.lidars[0].s['pose'])
        self.check()
        os.makedirs(self.out_dir, exist_ok=True)
        if need_gt:
            scene.write_lanes(self, world.get_map(), self.out_dir)

        self.phase('run')
        warm = int(cfg.get('warmup_steps', 10))
        for k in range(warm):
            self.check()
            frame = world.tick()
            self.alive()
            cams.drain(frame)
            self.emit.progress(0.02 + 0.03 * (k + 1) / max(warm, 1), 'warming up')
        steps = int(cfg.get('steps', 200))
        max_wall = float(cfg.get('max_wall_s', 0) or 0)
        mpx = rig_mpx(cfg['rig'])
        index = open(os.path.join(self.out_dir, 'frames.jsonl'), 'w', encoding='utf-8')
        t_rec = time.monotonic()
        n = late = 0
        sim_t0 = None
        label_warnings = []
        if raster is not None:
            label_warnings.append('ground labelled from the map: CARLA\'s label cameras do not see a generated '
                                  'OpenDRIVE world\'s road mesh (depth: nothing; tags: sky), so its pixels were '
                                  'given their range and class by their ray\'s hit on the map\'s (flat) ground and '
                                  'a 0.25 m raster of its lanes; lidar points of the mesh (tag 0) likewise '
                                  '(objects.json "ground_relabelled", run.json lidar_checks)')
        faults = scene.FaultState(cfg['rig']['sensors'], int(cfg.get('seed', 0)))
        prev_yaw = None
        try:
            for i in range(steps):
                self.check()
                frame = world.tick()
                self.alive()
                snap_t = world.get_snapshot().timestamp.elapsed_seconds
                sim_t0 = snap_t if sim_t0 is None else sim_t0
                rec = {'step': i, 'frame': int(frame), 'sim_t': round(snap_t - sim_t0, 6), 'files': {}}
                rec.update(scene.ego_state(ego))
                depth_now = None
                yaw = rec['ego']['yaw']
                yaw_rate = 0.0 if prev_yaw is None else ((yaw - prev_yaw + 180.0) % 360.0 - 180.0) * fps0 * \
                    math.pi / 180.0
                prev_yaw = yaw
                boxes = camgt.scene_boxes(self, world, ego, level) if need_gt else None
                for r in self.recorders:
                    s = r.s
                    deliver = faults.delivery(s['id'], i, rec['sim_t'])
                    if s['id'] in faults.f:
                        rec.setdefault('stamps', {})[s['id']] = round(faults.stamp(s['id'], rec['sim_t']), 6)
                        if deliver != 'ok':
                            rec.setdefault('faults', {})[s['id']] = deliver
                    imgs = r.collect(frame)
                    if r is depth_rec:
                        depth_now = imgs[0]['depth']
                    if deliver == 'drop':
                        continue
                    keys = [f'{s["id"]}/{k}' for k in r.kinds()]
                    if deliver == 'freeze' and all(k in faults.last for k in keys):
                        for k in keys:
                            rec['files'][k] = faults.last[k]
                        continue
                    img_faults = faults.image(s['id'], r.plan['K'], scene.yaw_rate_optical(s, yaw_rate))
                    for kind in r.kinds():
                        ext = {'rgb': 'jpg', 'objects': 'json'}.get(kind, 'png')
                        rel = os.path.join(kind, s['id'], f'{i:06d}.{ext}')
                        rec['files'][f'{s["id"]}/{kind}'] = rel
                        faults.last[f'{s["id"]}/{kind}'] = rel
                    cams.pending.append(cams.pool.submit(r.write, imgs, i, self.out_dir, boxes, raster, img_faults))
                for k, L in enumerate(self.lidars):
                    m_raw = L.step(frame, rec['sim_t'], self.out_dir, rec)
                    if k == 0 and depth_now is not None and i % 5 == 0:
                        L.range_check(np.frombuffer(m_raw.raw_data, np.float32).reshape(-1, 4)[:, :3]
                                      .astype(np.float64), depth_cam, depth_now, M_cam_from_lidar)
                self.others.step(frame, rec['sim_t'], self.out_dir, rec)
                index.write(json.dumps(rec, separators=(',', ':')) + '\n')
                n += 1
                wall = time.monotonic() - t_rec
                if max_wall and wall >= max_wall:
                    break
                f = (wall / max_wall) if max_wall else (i + 1) / steps
                self.emit.progress(0.05 + 0.9 * f, f'step {i + 1}{"" if max_wall else f" of {steps}"} '
                                                   f'({(i + 1) / max(wall, 1e-6):.1f} steps/s)')
                if len(cams.pending) > MAX_IN_FLIGHT * max(len(self.recorders), 1):
                    late += 1
                    cams.flush(MAX_IN_FLIGHT)
            cams.flush(0)
        finally:
            index.close()
            self.others.close()
        wall = time.monotonic() - t_rec
        sim_s = n / fps
        perf = {'steps': n, 'wall_s': round(wall, 3), 'steps_per_s': round(n / max(wall, 1e-6), 3),
                'sim_s': round(sim_s, 3), 'realtime_factor': round(sim_s / max(wall, 1e-6), 3),
                'mpx_per_step': round(mpx, 3), 'mpx_per_s': round(mpx * n / max(wall, 1e-6), 2),
                'camera_frames': n * len(self.recorders), 'encoder_waits': late, 'source': 'measured'}
        for k, unit in (('steps_per_s', '1/s'), ('realtime_factor', '×'), ('mpx_per_s', 'Mpx/s')):
            self.emit.metric(k, perf[k], unit)
        write_json(os.path.join(self.out_dir, 'dataset.json'), scene.dataset_doc(
            cfg.get('name', 'drive'), self.recorders, n, fps, self.run,
            {'town': cfg.get('town'), 'xodr': cfg.get('xodr'), 'weather': cfg.get('weather'),
             'label_warnings': label_warnings}))
        self.emit.progress(1.0, f'recorded {n} steps in {wall:.1f} s', force=True)
        lidar_checks = [L.report() for L in self.lidars]
        for c in lidar_checks:
            m = c['match']
            self.emit.log(f'lidar {c["sensor"]}: {m["exact_slices"]} of {m["slices"]} slices matched by order '
                          f'(worst direction {m["max_dir_err_deg"]:.2g}°), points in vehicle boxes labelled vehicle: '
                          f'{c["box_label_share"]}')
        return {'performance': perf, 'lidar_checks': lidar_checks, 'faults_applied': faults.report(), 'summary': {'steps': n, 'cameras': len(cams.cams), 'town': cfg.get('town'),
                                                  'steps_per_s': perf['steps_per_s'],
                                                  'realtime_factor': perf['realtime_factor']},
                'name': cfg.get('name', 'drive'), 'config_sha256': sha256_json(cfg)}


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
