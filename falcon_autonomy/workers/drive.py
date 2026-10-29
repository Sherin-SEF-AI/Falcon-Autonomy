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
   step, frame, simulation time, the ego's pose and speed, and the ego's control in effect during that tick;
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
        for n in cfg.get('notes') or []:               # what the config already knows is not represented
            self.note(n)
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
        director = None
        if cfg.get('scenario') or cfg.get('replay'):
            if cfg.get('replay'):                       # a log replay (workers.replay): the same interface
                from .replay import LogDirector as Director
            else:
                from .director import Director
            director = Director(self, world, cfg, lib, rng)
            start, ego_wp = director.place_ego(spawns, order)
            ego_bp = lib.find(cfg['rig']['vehicle'])
            ego_bp.set_attribute('role_name', cfg.get('ego_role', 'hero'))
            st0 = ego_wp.transform
            st0.location.z += 0.3
            ego = world.try_spawn_actor(ego_bp, st0)
            if ego is None:
                raise Failed('the ego could not be placed at the scenario\'s start')
            self.own(ego)
            world.tick()
            director.setup(ego, ego_wp)
        else:
            ego = scene.spawn_ego(self, world, lib, spawns, order, cfg['rig']['vehicle'])
        # an autopilot ego can drive off a road that ends (an imported area's edge): put back, marked in the frame
        guard = scene.RoadEndGuard(self, world, ego, spawns) if director is None else None
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
            if director is not None:
                director.sync(frame)
            self.emit.progress(0.02 + 0.03 * (k + 1) / max(warm, 1), 'warming up')
        steps = int(cfg.get('steps', 200))
        if director is not None:
            steps = int(round(float(director.scn.get('duration_s', 12.0)) * fps))
        max_wall = float(cfg.get('max_wall_s', 0) or 0)
        mpx = rig_mpx(cfg['rig'])
        index = open(os.path.join(self.out_dir, 'frames.jsonl'), 'w', encoding='utf-8')
        actor_log = scene.ActorLog(self, os.path.join(self.out_dir, 'actors.jsonl'))
        t_rec = time.monotonic()
        n = late = 0
        sim_t0 = None
        label_warnings = []
        every = max(int(cfg.get('record_every', 1)), 1)
        if raster is not None:
            label_warnings.append('ground labelled from the map: CARLA\'s label cameras do not see a generated '
                                  'OpenDRIVE world\'s road mesh (depth: nothing; tags: sky), so its pixels were '
                                  'given their range and class by their ray\'s hit on the map\'s (flat) ground and '
                                  'a 0.25 m raster of its lanes; lidar points of the mesh (tag 0) likewise '
                                  '(objects.json "ground_relabelled", run.json lidar_checks)')
        faults = scene.FaultState(cfg['rig']['sensors'], int(cfg.get('seed', 0)))
        physics, watch = physics_and_watch(self, world, cfg, self.lidars, self.others)     # sensor physics, ODD
        prev_yaw = None
        try:
            for i in range(steps):
                self.check()
                if director is not None:
                    director.step(i / fps)
                frame = world.tick()
                if director is not None:
                    director.sync(frame)
                    director.record((i + 1) / fps)
                self.alive()
                snap_t = world.get_snapshot().timestamp.elapsed_seconds
                sim_t0 = snap_t if sim_t0 is None else sim_t0
                rec = {'step': i, 'frame': int(frame), 'sim_t': round(snap_t - sim_t0, 6), 'files': {}}
                if getattr(self, 'class_limits', None) is not None:     # stand-ins' speeds (core.assets)
                    self.class_limits.step(rec['sim_t'])
                if guard is not None:
                    ev = guard.step(rec['sim_t'])
                    if ev is not None:               # the frame's pose jumps: say so where every tool reads it
                        rec['teleport'] = ev
                        if len(guard.events) == 1:
                            self.note('the ego left the road network where a road ends and was put back on the road '
                                      '(frames.jsonl "teleport"); see docs/worlds.md')
                rec.update(scene.ego_state(ego))
                if director is not None and director.kinematic and director.k_vel(ego) is not None:
                    v = director.k_vel(ego)              # physics off: CARLA reports the ego still; the director knows
                    rec['speed_mps'] = round(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z), 4)
                rec['control'] = ego_control(ego, director)
                depth_now = None
                yaw = rec['ego']['yaw']
                yaw_rate = 0.0 if prev_yaw is None else ((yaw - prev_yaw + 180.0) % 360.0 - 180.0) * fps0 * \
                    math.pi / 180.0
                prev_yaw = yaw
                record = i % every == 0                  # record_every: simulate every step, write every n-th
                rec['recorded'] = record
                boxes = camgt.scene_boxes(self, world, ego, level) if need_gt and record else None
                pboxes = boxes if boxes is not None or physics is None or not record or not physics.needs_actors() \
                    else camgt.scene_boxes(self, world, ego, [])          # night headlights: the actors' boxes
                watch_rgb = None
                for r in self.recorders:
                    s = r.s
                    deliver = faults.delivery(s['id'], i, rec['sim_t'])
                    if s['id'] in faults.f:
                        rec.setdefault('stamps', {})[s['id']] = round(faults.stamp(s['id'], rec['sim_t']), 6)
                        if deliver != 'ok':
                            rec.setdefault('faults', {})[s['id']] = deliver
                    imgs = r.collect(frame)
                    if watch is not None and watch_rgb is None and watch.wants_frame():
                        watch_rgb = scene.rgb_of(imgs[0]['rgb'])
                    if r is depth_rec:
                        depth_now = imgs[0]['depth']
                    if deliver == 'drop' or not record:
                        continue
                    keys = [f'{s["id"]}/{k}' for k in r.kinds()]
                    if deliver == 'freeze' and all(k in faults.last for k in keys):
                        for k in keys:
                            rec['files'][k] = faults.last[k]
                        continue
                    img_faults = faults.image(s['id'], r.plan['K'], scene.yaw_rate_optical(s, yaw_rate))
                    for kind in r.kinds():
                        ext = {'rgb': 'jpg', 'objects': 'json', 'flow': 'npy'}.get(kind, 'png')
                        rel = os.path.join(kind, s['id'], f'{i:06d}.{ext}')
                        rec['files'][f'{s["id"]}/{kind}'] = rel
                        faults.last[f'{s["id"]}/{kind}'] = rel
                    phys = None if physics is None else physics.camera_frame(s['id'], every / fps, pboxes)
                    if phys is not None and phys[1]['effects']:
                        rec.setdefault('physics', {})[s['id']] = phys[1]['effects']
                    cams.pending.append(cams.pool.submit(r.write, imgs, i, self.out_dir, boxes, raster, img_faults,
                                                         phys))
                for k, L in enumerate(self.lidars):
                    m_raw = L.step(frame, rec['sim_t'], self.out_dir, rec)
                    if k == 0 and depth_now is not None and i % 5 == 0:
                        L.range_check(np.frombuffer(m_raw.raw_data, np.float32).reshape(-1, 4)[:, :3]
                                      .astype(np.float64), depth_cam, depth_now, M_cam_from_lidar)
                self.others.step(frame, rec['sim_t'], self.out_dir, rec)
                if watch is not None:
                    rec['odd'] = watch.step(rec['sim_t'], ego, rec['speed_mps'], watch_rgb)
                actor_log.step(world, i, frame, rec['sim_t'], director)
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
            actor_log.close()
            self.others.close()
        wall = time.monotonic() - t_rec
        sim_s = n / fps
        perf = {'steps': n, 'wall_s': round(wall, 3), 'steps_per_s': round(n / max(wall, 1e-6), 3),
                'sim_s': round(sim_s, 3), 'realtime_factor': round(sim_s / max(wall, 1e-6), 3),
                'mpx_per_step': round(mpx, 3), 'mpx_per_s': round(mpx * n / max(wall, 1e-6), 2),
                'camera_frames': -(-n // every) * len(self.recorders), 'record_every': every, 'encoder_waits': late, 'source': 'measured'}
        for k, unit in (('steps_per_s', '1/s'), ('realtime_factor', '×'), ('mpx_per_s', 'Mpx/s')):
            self.emit.metric(k, perf[k], unit)
        write_json(os.path.join(self.out_dir, 'dataset.json'), scene.dataset_doc(
            cfg.get('name', 'drive'), self.recorders, n, fps, self.run,
            {'town': cfg.get('town'), 'xodr': cfg.get('xodr'), 'weather': cfg.get('weather'),
             'label_warnings': label_warnings}))
        self.emit.progress(1.0, f'recorded {n} steps in {wall:.1f} s', force=True)
        scenario = None
        if director is not None:
            scenario = director.report()
            scenario.update(scenario_result(director, fps))
            with open(os.path.join(self.out_dir, 'trajectories.json'), 'w', encoding='utf-8') as fh:
                json.dump({'dt': 1.0 / fps, 'columns': ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'length', 'width'],
                           'frame': 'CARLA world', 'actors': director.traj}, fh, separators=(',', ':'))
        lidar_checks = [L.report() for L in self.lidars]
        for c in lidar_checks:
            m = c['match']
            self.emit.log(f'lidar {c["sensor"]}: {m["exact_slices"]} of {m["slices"]} slices matched by order '
                          f'(worst direction {m["max_dir_err_deg"]:.2g}°), points in vehicle boxes labelled vehicle: '
                          f'{c["box_label_share"]}')
        from .oddwatch import write as write_watch
        write_watch(self, self.out_dir, watch)
        if getattr(self, 'class_limits', None) is not None:
            self.run.setdefault('traffic', {}).update(self.class_limits.report())
        if getattr(self, 'subclass', None):
            self.run['subclasses'] = {str(k): v for k, v in sorted(self.subclass.items())}
        return {'performance': perf, 'lidar_checks': lidar_checks, 'scenario': scenario, 'faults_applied': faults.report(),
                'sensor_physics': physics.report() if physics is not None else {'enabled': False},
                'odd_monitor': watch.report() if watch is not None else None,
                'summary': {'steps': n, 'cameras': len(cams.cams), 'town': cfg.get('town'),
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


def physics_and_watch(w, world, cfg, lidars=(), others=None, with_physics=True):
    """(core.sensorphysics.Session or None, oddwatch.OddWatch or None) for a run's config; the lidars and the radars
    (scene.Others) get the session. Neither exists unless the config asks (sensor_physics.enabled, odd). A worker that
    does not apply the physics (with_physics False) says so when the config asks for it."""
    from ..core import sensorphysics as SPH
    from .oddwatch import OddWatch
    if not with_physics and (cfg.get(SPH.KEY) or {}).get('enabled'):
        w.note(f'sensor physics is not applied by the {w.name} worker: its sensors give CARLA\'s data as it is '
               '(docs/sensor-physics.md)')
    try:
        physics = SPH.Session.from_config(cfg) if with_physics else None
        watch = OddWatch.from_config(w, world, cfg)
    except (ValueError, KeyError) as e:
        raise Failed(f'sensor physics or ODD: {e}') from None
    for L in lidars:
        L.physics = physics
    if others is not None:
        others.physics = physics
    if physics is not None:
        w.emit.log(f'sensor physics on for {", ".join(sorted(physics.sensors)) or "no sensor"}: '
                   f'visibility {physics.cond["visibility_m"] or "∞"} m, rain {physics.cond["rain_mm_h"]} mm/h '
                   f'({physics.cond["rain_source"]})')
    return physics, watch


def ego_control(ego, director=None) -> dict | None:
    """The ego's control in effect during the tick that produced this step (docs/dataset-format.md, "Frames"): the
    director's when it drives the ego (in kinematic motion no VehicleControl reaches CARLA), else the vehicle's
    get_control() — what the traffic manager (or the physics-driven director) applied for that tick."""
    if director is not None and getattr(director, 'control_unknown', False):
        return None                                     # a log replay's ego: placed at its logged poses, no control
    u = director.ego_control if director is not None and director.ego_control is not None else None
    if u is None:
        c = ego.get_control()
        u = {'throttle': c.throttle, 'steer': c.steer, 'brake': c.brake, 'hand_brake': c.hand_brake,
             'reverse': c.reverse}
    return {'throttle': round(float(u.get('throttle', 0.0)), 4), 'steer': round(float(u.get('steer', 0.0)), 4),
            'brake': round(float(u.get('brake', 0.0)), 4), 'hand_brake': bool(u.get('hand_brake', False)),
            'reverse': bool(u.get('reverse', False))}


def scenario_result(director, fps) -> dict:
    """Criticality measures of the run (core.criticality) and the scenario's pass criteria against them."""
    import numpy as np
    from ..core import criticality as C
    ego = np.asarray(director.traj['ego'], float)
    actors = {k: np.asarray(v, float) for k, v in director.traj.items() if k != 'ego' and v}
    summ = C.summarise(ego, actors, 1.0 / fps, director.events)
    crit = director.scn.get('pass') or {}
    checks = []
    overall = summ.get('overall') or {}
    if crit.get('no_collision'):
        hit = sorted(k for k, a in (summ.get('actors') or {}).items() if a.get('collision'))
        n = len(hit) + len(summ.get('other_collisions') or [])
        checks.append({'name': 'no collision', 'ok': n == 0,
                       'detail': f'collided with {", ".join(hit + ["another object"] * len(summ.get("other_collisions") or []))}'
                       if n else 'none'})
    if crit.get('min_ttc_s') is not None:
        ttc = overall.get('min_ttc_s')
        checks.append({'name': f'time to collision ≥ {crit["min_ttc_s"]:g} s',
                       'ok': ttc is None or ttc >= float(crit['min_ttc_s']),
                       'detail': f'minimum {ttc:.2f} s' if ttc is not None else 'never on a collision course'})
    return {'criticality': summ, 'checks': checks, 'passed': all(c['ok'] for c in checks)}
