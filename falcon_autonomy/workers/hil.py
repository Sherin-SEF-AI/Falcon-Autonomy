"""hil: a hardware-in-the-loop session — the world stepped in synchronous mode at a fixed rate paced to the wall clock,
one camera and one lidar of the rig streamed to a device (a Jetson, or a simulated one on this machine) through the
gateway, the device's control applied at the next step, and every frame's latency recorded stage by stage.

    python -m falcon_autonomy.workers hil CONFIG.json

CONFIG.json is core.hil.hil_config's (falcon.hil_config/1). A remote device's address comes in FALCON_HIL_ADDRESS
(the user's settings; never written to a file). Commands on stdin, besides cancel: {"cmd": "stop"} ends the session
normally (the report is written).

The run: start the gateway (falcon_autonomy.hil.gateway, in the app's Python: CARLA's venv has no GStreamer bindings)
on a socket pair and wait until the device is ready; connect to CARLA, load the world, synchronous mode at 1 / fps,
spawn the ego off the autopilot (the device drives it) with the camera and the lidar, and the traffic; warm up; then
per step:

1. apply the newest control that came back (stamped "applied": the step's start); with none yet the ego is held
   braked, and with none for control_timeout_s it brakes (a lost device never drives on);
2. tick (stamps tick_start, tick_end); read the camera and the lidar of exactly that frame (ready); stamp them with
   the emulated sensor clocks (core.hil.EmulatedClock); hand them to the gateway (handoff);
3. sleep until the step's wall-clock time; a step that ran late is counted and pacing restarts from now (never a burst
   to catch up), so the real-time factor falls below 1 rather than steps being skipped.

At the end: the gateway's and the device's files (clock.jsonl, device.json, gateway.json), latency.jsonl (one record
per frame with every stamp, core.hil's field names), the summary in run.json ("hil") and the report (report.html,
waterfall.png). A cancelled session keeps what it measured: a latency record is never half a dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time

from ..core import protocol as P
from ..core.provenance import read_json, write_json
from . import scene
from .common import ROLE_SENSOR, Failed, Worker

HIL_SCHEMA = 'falcon.hil_config/1'
GATEWAY_READY_S = 60.0
LIDAR_ATTRS = ('channels', 'range', 'points_per_second', 'rotation_frequency', 'upper_fov', 'lower_fov',
               'horizontal_fov')


def now() -> int:
    return time.monotonic_ns()


class Hil(Worker):
    name = 'hil'

    def __init__(self, cfg: dict, argv: list, cfg_path: str = ''):
        super().__init__(argv, cfg['out'], cfg.get('timeout', 20.0), cfg.get('seed', 0), cfg,
                         {'config': cfg_path, 'xodr': cfg.get('xodr')} if cfg_path else None)
        self.cfg = cfg
        self.cameras = scene.Cameras(self)
        self.stop_requested = False
        self.gw = None                     # the gateway process
        self.gs = None                     # our end of the socket pair
        self.glock = threading.Lock()
        self.ready = None
        self.gw_error = None
        self.done = threading.Event()
        self.done_msg = {}
        self.rlock = threading.Lock()
        self.records: dict = {}            # frame → record
        self.inbox: list = []              # controls not yet applied
        self.est: dict = {}
        self.counts = {'control_timeouts': 0, 'steps_late': 0, 'held_before_first_control': 0}
        self.saved = False
        self.load0 = None
        self.watch = None                  # oddwatch.OddWatch when the config has an ODD

    def on_command(self, cmd: dict):
        if cmd.get('cmd') == 'stop':
            self.stop_requested = True

    # ------------------------------------------------------------------ the gateway
    def start_gateway(self):
        from ..hil.device_agent import pack
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
        env.update(PYTHONPATH=src, PYTHONUNBUFFERED='1')
        py = self.cfg.get('gateway_python') or 'python3'
        cmd = [py, '-m', 'falcon_autonomy.hil.gateway', '--fd', str(b.fileno()), '--config',
               os.path.join(self.out_dir, 'hil.json')]
        try:
            self.gw = subprocess.Popen(cmd, pass_fds=(b.fileno(),), stdin=subprocess.DEVNULL, env=env)
        except OSError as e:
            raise Failed(f'cannot start the gateway in {py}: {e}') from e
        b.close()
        self.gs = a
        self.pack = pack
        threading.Thread(target=self.from_gateway, daemon=True, name='falcon-hil-gateway').start()
        self.expect(GATEWAY_READY_S + 10.0)
        t0 = time.monotonic()
        while self.ready is None:
            self.check()
            if self.gw_error:
                raise Failed(f'the device is not ready: {self.gw_error}')
            if self.gw.poll() is not None:
                raise Failed(f'the gateway ended before the device was ready (exit {self.gw.returncode}): '
                             f'{self.gw_error or "see the log"}')
            if time.monotonic() - t0 > GATEWAY_READY_S:
                raise Failed(f'the device was not ready within {GATEWAY_READY_S:.0f} s')
            time.sleep(0.05)
            self.alive()
        self.phase(self.life['phase'])

    def to_gateway(self, header: dict, *payloads):
        n = sum(memoryview(p).nbytes for p in payloads)
        h = json.dumps(header, separators=(',', ':')).encode('utf-8')
        from ..hil.device_agent import HEAD
        with self.glock:
            self.gs.sendall(HEAD.pack(len(h), n) + h)
            for p in payloads:
                self.gs.sendall(p)

    def from_gateway(self):
        from ..hil.device_agent import read_msg
        while True:
            try:
                h, _ = read_msg(self.gs)
            except (ConnectionError, OSError, ValueError):
                break
            t = h.get('t')
            if t == 'control':
                h['ctrl_worker'] = now()
                with self.rlock:
                    self.inbox.append(h)
            elif t == 'gw':
                with self.rlock:
                    r = self.records.get(int(h['frame']))
                    if r is not None:
                        r.update({k: v for k, v in h.items() if k not in ('t', 'frame')})
            elif t == 'clock':
                self.est = h.get('estimate') or {}
            elif t == 'ready':
                self.ready = h
                self.est = h.get('clock') or {}
            elif t == 'log':
                self.emit.log(f'gateway: {h.get("text")}', h.get('level', 'info'))
            elif t == 'error':
                self.gw_error = h.get('text')
                self.emit.log(f'gateway: {h.get("text")}', 'error')
            elif t == 'done':
                self.done_msg = h
                self.done.set()
        self.done.set()

    def stop_gateway(self):
        if self.gs is None:
            return
        try:
            self.to_gateway({'t': 'bye'})
        except OSError:
            pass
        self.done.wait(12.0)
        if self.gw is not None:
            try:
                self.gw.wait(10.0)
            except subprocess.TimeoutExpired:
                self.gw.kill()
                self.gw.wait(5.0)
        try:
            self.gs.close()
        except OSError:
            pass
        self.gs = None

    # ------------------------------------------------------------------ the session
    def body(self) -> dict:
        cfg = self.cfg
        if cfg.get('schema') != HIL_SCHEMA:
            raise Failed(f'not a HIL config (schema {cfg.get("schema")!r}, expected {HIL_SCHEMA})')
        from ..core.budget import check_budget
        cam_s, lid_s = cfg['camera']['sensor'], cfg['lidar']['sensor']
        ok, why = check_budget({'sensors': [cam_s]}, cfg.get('render_budget_mpx', 200.0))
        if not ok:
            raise Failed(why, P.REFUSED)
        for n in cfg.get('notes') or []:
            self.note(n)
        self.emit.progress(0.0, 'starting the gateway and the device', force=True)
        self.start_gateway()
        dev = self.ready.get('device') or {}
        self.emit.log(f'device ready: {dev.get("model") or dev.get("platform")}, decoder {self.ready.get("decoder")}, '
                      f'encoder {self.ready.get("encoder")}, stack {self.ready.get("stack")}')
        carla = self.import_carla()
        import numpy as np
        from ..core.hil import EmulatedClock, live_stats
        self.emit.progress(0.02, 'connecting to CARLA', force=True)
        self.connect(cfg.get('host', '127.0.0.1'), cfg['port'], cfg.get('connect_wait_s', 60.0))
        world = scene.load_map(self, cfg)
        seed = int(cfg.get('seed', 0))
        rng = np.random.default_rng(seed)
        fps = float(cfg['fps'])
        self.dt = 1.0 / fps
        period = int(round(1e9 / fps))
        scene.synchronous(self, world, fps, seed, cfg.get('tm_port', 8310))
        world.set_weather(scene.weather_params(carla, cfg.get('weather')))
        world.tick()
        self.alive()
        lib = world.get_blueprint_library()
        spawns = list(world.get_map().get_spawn_points())
        if not spawns:
            raise Failed('the map has no spawn points')
        order = list(rng.permutation(len(spawns)))
        ego = scene.spawn_ego(self, world, lib, spawns, order, cfg['vehicle'], autopilot=False)
        scene.spawn_traffic(self, world, lib, spawns, order, rng, cfg.get('traffic') or {})
        scene.light_vehicles(self, cfg)                 # headlights at night (workers.lights vehicle_lights)
        cam = self.cameras.spawn(world, lib, 'sensor.camera.rgb', cam_s['width'], cam_s['height'], cam_s['fov'],
                                 scene.camera_transform(carla, cam_s['pose']), ego)
        lidar = self.spawn_lidar(world, lib, lid_s, ego)
        from .drive import physics_and_watch
        _, self.watch = physics_and_watch(self, world, cfg, with_physics=False)       # the ODD monitor, when asked
        clocks = cfg.get('clocks') or {}
        cclk = EmulatedClock(**(clocks.get('camera') or {}), seed=seed * 2 + 1)
        lclk = EmulatedClock(**(clocks.get('lidar') or {}), seed=seed * 2 + 2)
        hold = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False)
        ego.apply_control(hold)
        self.phase('run')
        for _ in range(int(cfg.get('warmup_steps', 10))):
            self.check()
            self.cameras.drain(world.tick())
            self.alive()
        from . import lights
        self.run['lights'] = lights.report(world, [a for a in self.spawned if a.type_id.startswith('vehicle.')],
                                           self.lights_applied)
        steps = int(round(float(cfg.get('seconds', 0.0)) * fps))
        self.load0 = os.getloadavg()                           # the machine's load: it shows in every number
        t_start = time.monotonic()
        next_t = t_start
        last_ctrl_t = None
        last_applied = -1
        n = 0
        last_live = time.monotonic()
        self.emit.progress(0.05, 'streaming', force=True)
        while not self.stop_requested and (steps <= 0 or n < steps):
            self.check()
            if self.done.is_set() or self.gw.poll() is not None:
                raise Failed(f'the gateway stopped: {self.gw_error or "it ended"}')
            # 1. the newest control that came back
            with self.rlock:
                box, self.inbox = self.inbox, []
            t_apply = now()
            fresh = [c for c in box if int(c['frame']) > last_applied]
            newest = max(fresh, key=lambda c: int(c['frame'])) if fresh else None
            with self.rlock:
                for c in box:
                    r = self.records.get(int(c['frame']))
                    if r is None:
                        continue
                    self.merge_control(r, c)
                    if c is not newest:
                        r['status'] = 'superseded'
            if newest is not None:
                ego.apply_control(carla.VehicleControl(throttle=min(max(newest['throttle'], 0.0), 1.0),
                                                       steer=min(max(newest['steer'], -1.0), 1.0),
                                                       brake=min(max(newest['brake'], 0.0), 1.0)))
                last_applied = int(newest['frame'])
                last_ctrl_t = time.monotonic()
                applied_rec = self.records.get(last_applied)
            else:
                applied_rec = None
                if last_ctrl_t is None:
                    self.counts['held_before_first_control'] += 1
                    ego.apply_control(hold)
                elif time.monotonic() - last_ctrl_t > float(cfg.get('control_timeout_s', 0.5)):
                    self.counts['control_timeouts'] += 1
                    ego.apply_control(hold)
            # 2. the step
            t0 = now()
            frame = world.tick()
            t1 = now()
            self.alive()
            if applied_rec is not None:
                with self.rlock:
                    applied_rec.update(applied=t_apply, applied_frame=int(frame),
                                       status='on_time' if int(frame) == applied_rec['frame'] + 1 else 'late')
            snap = world.get_snapshot()
            sim_t = snap.timestamp.elapsed_seconds
            scene.speeds_step(self, sim_t)
            img = self.cameras.get(cam.id, frame)
            lm = self.cameras.get(lidar.id, frame)
            t_ready = now()
            v = ego.get_velocity()
            speed = float((v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5)
            cam_hw, lid_hw = cclk.stamp(sim_t), lclk.stamp(sim_t)
            npts = memoryview(lm.raw_data).nbytes // 16
            rec = {'frame': int(frame), 'sim_t': round(sim_t, 6), 'speed_mps': round(speed, 4), 'cam_hw_t': cam_hw,
                   'lidar_hw_t': lid_hw, 'tick_start': t0, 'tick_end': t1, 'ready': t_ready, 'lidar_points': npts,
                   'status': 'missing'}
            if self.watch is not None:
                rec['odd'] = self.watch.step(sim_t, ego, speed)
            with self.rlock:
                self.records[int(frame)] = rec
            cb, lb = img.raw_data, lm.raw_data
            hdr = {'t': 'frame', 'frame': int(frame), 'sim_t': round(sim_t, 6), 'speed_mps': round(speed, 4),
                   'cam': {'w': img.width, 'h': img.height, 'hw_t': cam_hw, 'bytes': memoryview(cb).nbytes},
                   'lidar': {'n': npts, 'hw_t': lid_hw, 'bytes': memoryview(lb).nbytes}}
            try:
                self.to_gateway(hdr, cb, lb)
            except OSError as e:
                raise Failed(f'the gateway stopped taking frames: {self.gw_error or e}') from e
            rec['handoff'] = now()
            n += 1
            if time.monotonic() - last_live >= 1.0:
                last_live = time.monotonic()
                with self.rlock:
                    recent = [dict(r) for f, r in self.records.items() if f >= int(frame) - 2 * int(fps)
                              and f <= int(frame) - 3]
                st = live_stats(recent, self.est, period)
                st['steps'] = n
                st['realtime_factor'] = round(n * self.dt / max(time.monotonic() - t_start, 1e-6), 3)
                self.emit.send(dict(st, t='hil'))
                self.emit.progress(0.05 + 0.9 * (n / steps if steps > 0 else 0.0),
                                   f'step {n}{f" of {steps}" if steps > 0 else ""}; e2e median '
                                   f'{st["e2e"].get("median")} ms', force=steps <= 0)
            # 3. pacing
            next_t += self.dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -self.dt:
                self.counts['steps_late'] += 1
                next_t = time.monotonic()
        wall = time.monotonic() - t_start
        # the last controls: give the device a moment, then hold the ego
        time.sleep(min(0.5, 10 * self.dt))
        with self.rlock:
            for c in self.inbox:
                r = self.records.get(int(c['frame']))
                if r is not None:
                    self.merge_control(r, c)
                    r['status'] = 'after_end'                 # it came back after the session's last step
            self.inbox = []
        ego.apply_control(hold)
        self.emit.progress(0.96, 'collecting the device\'s numbers', force=True)
        self.stop_gateway()
        return self.finish_session(n, wall, period)

    def merge_control(self, r: dict, c: dict):
        r.update(ctrl_rx=c.get('ctrl_rx'), ctrl_worker=c.get('ctrl_worker'), missing_inputs=c.get('missing') or [],
                 control={k: round(float(c.get(k, 0.0)), 4) for k in ('throttle', 'steer', 'brake')},
                 stack_info=c.get('info') or {})
        for k, v in (c.get('dev') or {}).items():
            r['dev_' + k] = v

    def spawn_lidar(self, world, lib, s: dict, ego):
        import queue
        bp = lib.find('sensor.lidar.ray_cast')
        for k in LIDAR_ATTRS:
            if k in s and bp.has_attribute(k):
                bp.set_attribute(k, str(s[k]))
        for k, v in (('noise_stddev', s.get('noise_stddev', 0.0)), ('dropoff_general_rate', 0.0),
                     ('dropoff_intensity_limit', 0.0), ('dropoff_zero_intensity', 0.0), ('sensor_tick', 0.0),
                     ('role_name', ROLE_SENSOR)):
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))
        a = self.own(world.spawn_actor(bp, scene.camera_transform(self.carla, s['pose']), attach_to=ego))
        q = queue.Queue()
        self.cameras.queues[a.id] = q
        a.listen(q.put)
        self.alive()
        return a

    def write_records(self):
        """latency.jsonl: every frame's record, with what the device received even where no control came back."""
        device = {}
        try:
            device = read_json(os.path.join(self.out_dir, 'device.json'))
        except (OSError, ValueError):
            self.note('the gateway wrote no device.json: the device\'s own counts are missing')
        stats = device.get('stats') or {}
        fields = stats.get('arrival_fields') or []
        with self.rlock:
            for row in stats.get('arrivals') or []:
                a = dict(zip(fields, row))
                r = self.records.get(int(a.get('frame', -1)))
                if r is None:
                    continue
                for k in ('cam_rx', 'dec_end', 'lidar_rx'):
                    if a.get(k) is not None and r.get('dev_' + k) is None:
                        r['dev_' + k] = a[k]
            recs = [self.records[f] for f in sorted(self.records)]
        with open(os.path.join(self.out_dir, 'latency.jsonl'), 'w', encoding='utf-8') as fh:
            for r in recs:
                fh.write(json.dumps(r, separators=(',', ':')) + '\n')
        return recs

    def session_info(self) -> dict:
        gateway = {}
        try:
            gateway = read_json(os.path.join(self.out_dir, 'gateway.json'))
        except (OSError, ValueError):
            pass
        ready = self.ready or {}
        cs, ls = self.cfg['camera']['sensor'], self.cfg['lidar']['sensor']
        remote = self.cfg['device']['mode'] == 'remote'
        return {
            'device': {'mode': self.cfg['device']['mode'],
                       'address': 'from the user\'s settings (not recorded)' if remote else '127.0.0.1 (simulated)',
                       'env': ready.get('device'), 'decoder': ready.get('decoder'), 'stack': ready.get('stack'),
                       'clock': ready.get('device_clock'),
                       'emulated_clock': None if remote else {k: (self.cfg.get('sim_device') or {}).get(k)
                                                              for k in ('clock_offset_ms', 'clock_drift_ppm')}},
            'gateway': gateway,
            'camera': dict({k: v for k, v in self.cfg['camera'].items() if k != 'sensor'}, width=cs['width'],
                           height=cs['height'], fov=cs['fov'], encoder_used=ready.get('encoder'),
                           encoder_props=ready.get('encoder_props')),
            'lidar': dict({k: v for k, v in self.cfg['lidar'].items() if k != 'sensor'}, channels=ls['channels'],
                          points_per_second=ls['points_per_second'], rotation_frequency=ls['rotation_frequency']),
            'fps': self.cfg['fps'], 'seed': self.cfg.get('seed', 0),
            'host_load': {'start': [round(x, 2) for x in self.load0] if self.load0 else None,
                          'end': [round(x, 2) for x in os.getloadavg()], 'cpus': os.cpu_count(),
                          'note': '1, 5 and 15 minute load averages of this machine'}}

    def finish_session(self, n: int, wall: float, period: int) -> dict:
        from ..core.hil import summarise_run
        self.write_records()
        scene.speeds_report(self)
        perf = {'steps': n, 'wall_s': round(wall, 3), 'sim_s': round(n * self.dt, 3),
                'realtime_factor': round(n * self.dt / max(wall, 1e-6), 3), 'source': 'measured'}
        write_json(os.path.join(self.out_dir, 'session.json'),
                   {'counts': dict(self.counts), 'performance': perf, 'session': self.session_info()})
        self.saved = True
        summary = summarise_run(self.out_dir)
        self.run.update(performance=perf, hil=summary)
        write_json(os.path.join(self.out_dir, 'run.json'), self.run)    # the report reads it; finish_run rewrites it
        try:
            from ..hil.report import write_report
            files = write_report(self.out_dir)
        except Exception as e:                                # noqa: BLE001 (the numbers are kept regardless)
            files = {}
            self.note(f'the report could not be written: {type(e).__name__}: {e}')
        e2e = summary['measures']['e2e']
        return {'performance': perf, 'hil': summary, 'report': files,
                'summary': {'steps': n, 'realtime_factor': perf['realtime_factor'], 'e2e_median_ms': e2e.get('median'),
                            'e2e_p95_ms': e2e.get('p95'), 'on_time': summary['deadline']['on_time'],
                            'deadline_misses': summary['deadline']['deadline_misses'],
                            'report': files.get('html')},
                'odd_monitor': self.watch.report() if self.watch is not None else None,
                'name': self.cfg.get('name', 'HIL session')}

    def tidy(self):
        try:
            if self.gs is not None:
                self.stop_gateway()
            if self.records and not self.saved:              # cancelled or failed mid-session: keep what was measured
                self.write_records()
                write_json(os.path.join(self.out_dir, 'session.json'),
                           {'counts': dict(self.counts), 'performance': {}, 'session': self.session_info()})
        finally:
            self.cameras.stop()
            super().tidy()


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker hil')
    ap.add_argument('config', help='HIL session config JSON (falcon_autonomy.core.hil)')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return Hil(cfg, argv, a.config).execute()
