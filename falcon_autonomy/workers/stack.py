"""stack: an external driving stack drives Falcon's ego — through ROS 2 (Falcon's bridge), or a baseline (Falcon's lane
follower, CARLA's autopilot), or a replay of a previous run's controls — on a scenario or free driving, in synchronous
mode, closed loop (runs in CARLA's Python).

    python -m falcon_autonomy.workers stack CONFIG.json

CONFIG.json (schema falcon.stack_config/1, core.stack.stack_config writes it): a drive config (server, map, weather,
rig, traffic, seed, fps, steps, an optional concrete scenario with its pass criteria) and

    driver   "ros2"      the stack behind Falcon's bridge (core.stackbridge; bridge: its section)
             "follower"  Falcon's built-in lane follower (the scenario director's)        } baselines: no bridge,
             "autopilot" CARLA's traffic manager                                          } no sensors spawned
             "replay"    the controls of a previous run (replay: {"run": folder}) applied step by step
    motion   "kinematic" (default; repeatable) or "physics" — every scenario vehicle, the ego included
    bridge   host, port, control (autoware | ackermann | carla), control_topic, deadline_s, start_wait_s,
             connect_wait_s, on_miss (hold | brake), image_encoding, namespace, autoware_status, ego_tf, odometry,
             topics (renames), qos, max_missed (a pass criterion: at most this many missed deadlines; null: not
             checked); the port must not be 0 here (sim.stacksuite.run_one picks a free one for 0)

A step (driver ros2). The director fires the scenario's timeline and moves its actors; the control the stack sent for
the previous step drives the ego (kinematic: the director's bicycle model; physics: VehicleControl); one tick; then
every sensor reading of exactly that frame is encoded as a ROS 2 message (core.ros2msg) and sent to the bridge node in
one "step"; the worker waits for the stack's control of this step (stackbridge.Deadlines: deadline_s, or start_wait_s
for the first) and maps it to throttle / steer / brake (stackbridge.ControlMapper). The first tick runs with no control
(coasting). The simulation never advances while a deadline is open; a missed deadline applies on_miss and is recorded.

Topics (stackbridge.topic_plan; messages stamped with the simulation time of the frame, World snapshot
elapsed_seconds, and /clock the same): /clock, /tf_static (once, latched), each camera's image (bgr8 by default) and
camera_info, each lidar's points (a full sweep: the steps of one rotation gathered, published at the step completing it;
points are in the lidar's frame at each slice's step, not motion-compensated), each IMU and GNSS (every step:
sensor_tick 0, the rig's rate is not applied), each roadside unit's CPM when it sends one (/falcon/v2x/<rsu>/cpm as
udp_msgs/UdpPacket, BTP-B + UPER, and cpm_json; workers.rsu, docs/v2x.md), then /tf (map -> base_link), the odometry and Autoware's
/vehicle/status/* (each can be turned off: ego_tf, odometry, autoware_status). In kinematic motion the ego's pose and velocity are the director's (CARLA reports a physics-off
actor as still).

Outputs: frames.jsonl (per step: step, frame, sim_t, ego, speed_mps, control in effect during the tick, and for ros2
"bridge": wait_ms, missed, stale), trajectories.json and the scenario's criticality and checks (a scenario), run.json
("bridge": the deadlines, the per-step costs — tick, sensors, encode, send, the node's publish, the wait — measured;
"performance"). A replay of a kinematic run repeats it bit for bit (measured on the fake; docs/scenarios.md for the
real server's kinematic repetition).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import socket
import threading
import time

from ..core import protocol as P
from ..core.provenance import read_json, sha256_json
from . import scene
from .common import ROLE_SENSOR, Failed, Worker

STACK_SCHEMA = 'falcon.stack_config/1'
DRIVERS = ('ros2', 'follower', 'autopilot', 'replay', 'none')


class Link:
    """The worker's end of the bridge link: listens, accepts the node, a reader thread queues what the node sends."""

    def __init__(self, host: str, port: int):
        from ..core.livelink import Reader
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.srv.bind((host, int(port)))
        except OSError as e:
            self.srv.close()
            raise Failed(f'cannot listen for the bridge node on {host}:{port}: {e}', P.REFUSED) from e
        self.srv.listen(1)
        self.sock = None
        self.q: queue.Queue = queue.Queue()
        self.reader = Reader()
        self.closed = ''

    def accept(self, w, wait_s: float):
        self.srv.settimeout(0.5)
        t0 = time.monotonic()
        while self.sock is None:
            w.check()
            try:
                s, _ = self.srv.accept()
            except socket.timeout:
                if time.monotonic() - t0 > wait_s:
                    raise Failed(f'no bridge node connected within {wait_s:g} s (start it: falcon stack bridge, or '
                                 'falcon_autonomy/ros2/falcon_bridge_node.py --connect HOST:PORT)') from None
                continue
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s
        threading.Thread(target=self._read, daemon=True, name='falcon-bridge-link').start()

    def _read(self):
        try:
            while True:
                data = self.sock.recv(1 << 20)
                if not data:
                    break
                for m in self.reader.feed(data):
                    self.q.put(m)
        except (OSError, ValueError) as e:
            self.closed = str(e)
        self.closed = self.closed or 'the bridge node closed the link'
        self.q.put(({'t': 'closed', 'why': self.closed}, b''))

    def send(self, header: dict, payload: bytes = b''):
        from ..core.livelink import pack
        try:
            self.sock.sendall(pack(header, payload))
        except OSError as e:
            raise Failed(f'the bridge link broke: {e}') from e

    def close(self):
        for s in (self.sock, self.srv):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass


class Stack(Worker):
    name = 'stack'

    def __init__(self, cfg: dict, argv: list, cfg_path: str = ''):
        super().__init__(argv, cfg['out'], cfg.get('timeout', 20.0), cfg.get('seed', 0), cfg,
                         {'config': cfg_path, 'xodr': cfg.get('xodr'),
                          'replay_of': os.path.join(cfg['replay']['run'], 'frames.jsonl')
                          if cfg.get('driver') == 'replay' and (cfg.get('replay') or {}).get('run') else None}
                         if cfg_path else None)
        self.cfg = cfg
        self.keep_files = {'run.json', os.path.basename(cfg_path) if cfg_path else 'stack.json'}
        self.cameras = scene.Cameras(self)
        self.link = None

    def stop_writers(self):
        self.cameras.stop()

    # ------------------------------------------------------------------ setup
    def body(self) -> dict:
        cfg = self.cfg
        if cfg.get('schema') != STACK_SCHEMA:
            raise Failed(f'not a stack config (schema {cfg.get("schema")!r}, expected {STACK_SCHEMA})')
        driver = cfg.get('driver', 'ros2')
        if driver not in DRIVERS:
            raise Failed(f'driver {driver!r}: one of {", ".join(DRIVERS)}')
        from ..core import stackbridge as SB
        bridge = dict(SB.bridge_defaults(), **(cfg.get('bridge') or {}))
        if driver == 'ros2':
            bad = SB.check_bridge(bridge)
            if bad:
                raise Failed('; '.join(bad))
            from ..core.budget import check_budget
            ok, why = check_budget(cfg['rig'], cfg.get('render_budget_mpx', 200.0))
            if not ok:
                raise Failed(why, P.REFUSED)
        replay = None
        if driver == 'replay':
            replay = self.read_replay((cfg.get('replay') or {}).get('run', ''))
        carla = self.import_carla()
        import numpy as np
        self.np = np
        if driver == 'ros2':                           # listen first: the node may start before the server is up
            if not int(bridge['port']):
                raise Failed('bridge port 0: the runner picks a free port (sim.stacksuite.run_one); give one to run '
                             'the worker by hand')
            self.link = Link(bridge['host'], int(bridge['port']))
        self.emit.progress(0.0, 'connecting', force=True)
        self.connect(cfg.get('host', '127.0.0.1'), cfg['port'], cfg.get('connect_wait_s', 60.0))
        world = scene.load_map(self, cfg)
        self.check()
        seed = int(cfg.get('seed', 0))
        rng = np.random.default_rng(seed)
        fps = float(cfg.get('fps', 20.0))
        self.fps = fps
        scene.synchronous(self, world, fps, seed, cfg.get('tm_port', 8310))
        world.set_weather(scene.weather_params(carla, cfg.get('weather')))
        world.tick()
        self.alive()
        lib = world.get_blueprint_library()
        spawns = list(world.get_map().get_spawn_points())
        if not spawns:
            raise Failed('the map has no spawn points')
        order = list(rng.permutation(len(spawns)))
        director = None
        if cfg.get('scenario'):
            from .director import Director
            conc = dict(cfg['scenario'])
            conc['ego'] = dict(conc.get('ego') or {}, driver='stack' if driver in ('ros2', 'replay') else driver)
            conc['motion'] = cfg.get('motion', conc.get('motion', 'kinematic'))
            director = Director(self, world, dict(cfg, scenario=conc), lib, rng)
            _start, ego_wp = director.place_ego(spawns, order)
            ego_bp = lib.find(cfg['rig']['vehicle'])
            ego_bp.set_attribute('role_name', 'hero')
            st0 = ego_wp.transform
            st0.location.z += 0.3
            ego = world.try_spawn_actor(ego_bp, st0)
            if ego is None:
                raise Failed('the ego could not be placed at the scenario\'s start')
            self.own(ego)
            world.tick()
            director.setup(ego, ego_wp)
        else:
            if driver in ('follower', 'none'):
                raise Failed(f'driver {driver!r} needs a scenario (its route and set speed come from it)')
            ego = scene.spawn_ego(self, world, lib, spawns, order, cfg['rig']['vehicle'],
                                  autopilot=driver == 'autopilot')
        scene.spawn_traffic(self, world, lib, spawns, order, rng, cfg.get('traffic') or {})
        scene.spawn_walkers(self, world, lib, rng, cfg.get('traffic') or {})
        self.run['lights'] = scene.light_vehicles(self, cfg)      # headlights at night (workers.lights)
        self.ego, self.director, self.world_ = ego, director, world
        sensors = self.spawn_sensors(world, lib, ego, cfg['rig']) if driver == 'ros2' else []
        kinematic = director is not None and director.kinematic and driver != 'autopilot'
        max_steer = SB.KINEMATIC_MAX_STEER_DEG if kinematic else self.max_steer_deg(ego)
        mapper = SB.ControlMapper(max_steer)
        plan, control = [], None
        self.stations = []
        if driver == 'ros2' and bridge.get('v2x', True) and (cfg.get('infrastructure') or {}).get('rsus'):
            from . import rsu as RSU                   # roadside units: their CPMs published (docs/v2x.md)
            self.stations = RSU.stations(self, world, lib, cfg, self.cameras, record=False)
        if driver == 'ros2':
            plan = SB.topic_plan(cfg['rig'], bridge, [st.id for st in self.stations])
            control = SB.control_spec(bridge)
            self.phase('bridge')
            self.emit.progress(0.02, f'waiting for the bridge node on {bridge["host"]}:{bridge["port"]}', force=True)
            self.link.accept(self, float(bridge['connect_wait_s']))
            self.link.send({'t': 'hello', 'protocol': SB.PROTOCOL, 'run': cfg.get('name', 'stack'),
                            'topics': [{k: e[k] for k in ('topic', 'type', 'qos')} for e in plan],
                            'control': {'topic': control['topic'], 'type': control['type']}})
            ready = self.wait_message('ready', 30.0)
            if ready is None:
                raise Failed('the bridge node did not answer the hello within 30 s')
            ctl = ready.get('control') or {}
            if not ctl.get('ok'):
                raise Failed(f'the bridge node cannot subscribe {control["topic"]} ({control["type"]}): '
                             f'{ctl.get("why") or "no reason given"}')
            self.run['bridge_node'] = {k: ready.get(k) for k in ('node', 'ros_distro', 'rmw')}
            gone = {m['topic'] for m in ready.get('missing') or []}
            if gone:                                   # e.g. an image built before udp_msgs was added
                self.note('the bridge node cannot publish ' + ', '.join(f'{m["topic"]} ({m["type"]})' for m in
                                                                      ready['missing'])
                          + ': its ROS lacks the message package (rebuild the image: falcon stack image)')
                plan = [e for e in plan if e['topic'] not in gone]
            self.emit.log(f'bridge node ready ({ready.get("ros_distro") or "?"}, {ready.get("rmw") or "?"}); control '
                          f'from {control["topic"]} ({control["type"]})')
            self.phase('setup')
        os.makedirs(self.out_dir, exist_ok=True)
        from .drive import physics_and_watch
        _, watch = physics_and_watch(self, world, cfg, with_physics=False)       # the ODD monitor, when asked
        watch_cam = next((x['id'] for x in sensors if x['kind'] == 'camera'), None)

        # ------------------------------------------------------------------ the run
        self.phase('run')
        warm = int(cfg.get('warmup_steps', 5))
        for _ in range(warm):
            self.check()
            frame = world.tick()
            self.alive()
            self.cameras.drain(frame)
            if director is not None:
                director.sync(frame)
        steps = int(cfg.get('steps', 200))
        if director is not None:
            steps = int(round(float(director.scn.get('duration_s', 12.0)) * fps))
        if replay is not None:
            steps = min(steps, len(replay))
        coast = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0, 'hand_brake': False, 'reverse': False}
        u = dict(coast)
        dl = SB.Deadlines()
        costs = {k: [] for k in ('tick_ms', 'sensors_ms', 'encode_ms', 'send_ms', 'publish_ms', 'wait_ms', 'step_ms')}
        sweeps = {s['id']: [] for s in sensors if s['kind'] == 'lidar'}
        index = open(os.path.join(self.out_dir, 'frames.jsonl'), 'w', encoding='utf-8')
        cpm_log = RSU.CpmLog(os.path.join(self.out_dir, 'cpm.jsonl')) if self.stations else None
        sim_t0 = None
        prev_yaw = None
        t_run = time.monotonic()
        n = 0
        try:
            for i in range(steps):
                self.check()
                t_step = time.monotonic()
                if director is not None:
                    director.step(i / fps)
                if replay is not None:
                    u = replay[i]
                if driver in ('ros2', 'replay'):
                    self.apply(carla, u, kinematic)
                t0 = time.monotonic()
                frame = world.tick()
                self.alive()
                if director is not None:
                    director.sync(frame)
                    director.record((i + 1) / fps)
                costs['tick_ms'].append((time.monotonic() - t0) * 1e3)
                snap = world.get_snapshot()
                sim_abs = snap.timestamp.elapsed_seconds
                sim_t0 = sim_abs if sim_t0 is None else sim_t0
                scene.speeds_step(self, sim_abs - sim_t0)
                ego_doc, vel, ang = self.ego_now(snap, director)
                speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                applied = u if driver in ('ros2', 'replay') else self.director_control(director)
                rec = {'step': i, 'frame': int(frame), 'sim_t': round(sim_abs - sim_t0, 6), 'ego': ego_doc,
                       'speed_mps': round(speed, 4), 'control': _round_control(applied)}
                if driver == 'ros2':
                    if kinematic:
                        dyaw = 0.0 if prev_yaw is None else ((ego_doc['yaw'] - prev_yaw + 180.0) % 360.0 - 180.0)
                        ang = (0.0, 0.0, dyaw * fps)
                    prev_yaw = ego_doc['yaw']
                    t_ns = int(round(sim_abs * 1e9))
                    t0 = time.monotonic()
                    data = self.collect(sensors, frame)
                    cpms = {}
                    for st in self.stations:
                        line = st.step(i, frame, rec['sim_t'], {'files': {}}, record_frame=False, director=director)
                        if line is not None:
                            cpms[st.id] = line
                            cpm_log.write(line)
                    costs['sensors_ms'].append((time.monotonic() - t0) * 1e3)
                    t0 = time.monotonic()
                    msgs = self.encode(plan, sensors, data, sweeps, i, t_ns, ego_doc, vel, ang, applied, mapper,
                                       bridge, cfg['rig'], first=(i == 0), cpms=cpms)
                    costs['encode_ms'].append((time.monotonic() - t0) * 1e3)
                    t0 = time.monotonic()
                    self.link.send({'t': 'step', 'step': i, 'stamp_ns': t_ns, 'control': True,
                                    'msgs': [{'topic': tp, 'nbytes': len(b)} for tp, b in msgs]},
                                   b''.join(b for _, b in msgs))
                    costs['send_ms'].append((time.monotonic() - t0) * 1e3)
                    t0 = time.monotonic()
                    cmd, pub_us = self.wait_control(i, t_ns, control, dl,
                                                    float(bridge['start_wait_s'] if i == 0 else bridge['deadline_s']))
                    wait = (time.monotonic() - t0) * 1e3
                    costs['wait_ms'].append(wait)
                    dl.waits_ms.append(wait)
                    if pub_us is not None:
                        costs['publish_ms'].append(pub_us / 1e3)
                    if cmd is None:
                        dl.miss(i)
                        u = dict(coast, brake=1.0) if bridge['on_miss'] == 'brake' else u
                    else:
                        u = mapper(cmd, speed)
                    rec['bridge'] = {'wait_ms': round(wait, 3), 'missed': cmd is None}
                if watch is not None:
                    rgb = scene.rgb_of(data[watch_cam]) if driver == 'ros2' and watch_cam and watch.wants_frame() \
                        else None
                    rec['odd'] = watch.step(rec['sim_t'], ego, speed, rgb)
                index.write(json.dumps(rec, separators=(',', ':')) + '\n')
                n += 1
                costs['step_ms'].append((time.monotonic() - t_step) * 1e3)
                wall = time.monotonic() - t_run
                self.emit.progress(0.05 + 0.9 * (i + 1) / steps, f'step {i + 1} of {steps} '
                                                                 f'({(i + 1) / max(wall, 1e-6):.1f} steps/s)')
        finally:
            index.close()
            if cpm_log is not None:
                cpm_log.close()
            if self.link is not None and self.link.sock is not None:
                try:
                    self.link.send({'t': 'bye', 'reason': 'the run ended'})
                except Failed:
                    pass
        wall = time.monotonic() - t_run
        perf = {'steps': n, 'wall_s': round(wall, 3), 'steps_per_s': round(n / max(wall, 1e-6), 3),
                'sim_s': round(n / fps, 3), 'realtime_factor': round(n / fps / max(wall, 1e-6), 3),
                'source': 'measured'}
        self.emit.metric('steps_per_s', perf['steps_per_s'], '1/s')
        scene.speeds_report(self)
        out = {'performance': perf, 'driver': driver, 'motion': 'kinematic' if kinematic else 'physics',
               'name': cfg.get('name', 'stack run'), 'config_sha256': sha256_json(cfg)}
        if driver == 'ros2':
            out['bridge'] = dict(dl.report(), control=control, topics=[e['topic'] for e in plan],
                                 deadline_s=bridge['deadline_s'], on_miss=bridge['on_miss'],
                                 costs_ms={k: _pct(v) for k, v in costs.items()},
                                 max_steer_deg=round(max_steer, 3))
        scenario = None
        if director is not None:
            from .drive import scenario_result
            scenario = director.report()
            scenario.update(scenario_result(director, fps))
            if driver == 'ros2' and bridge.get('max_missed') is not None:
                ok = dl.missed <= int(bridge['max_missed'])
                scenario['checks'].append({'name': f'missed deadlines ≤ {int(bridge["max_missed"])}', 'ok': ok,
                                           'detail': f'{dl.missed} missed of {n}'})
                scenario['passed'] = scenario['passed'] and ok
            with open(os.path.join(self.out_dir, 'trajectories.json'), 'w', encoding='utf-8') as fh:
                json.dump({'dt': 1.0 / fps, 'columns': ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'length', 'width'],
                           'frame': 'CARLA world', 'actors': director.traj}, fh, separators=(',', ':'))
        out['scenario'] = scenario
        if watch is not None:
            from .oddwatch import write as write_watch
            write_watch(self, self.out_dir, watch)
            out['odd_monitor'] = watch.report()
        if self.stations:
            out['infrastructure'] = {'stations': [st.report() for st in self.stations]}
        out['summary'] = {'steps': n, 'driver': driver, 'steps_per_s': perf['steps_per_s'],
                          'passed': None if scenario is None else scenario['passed'],
                          'missed_deadlines': dl.missed if driver == 'ros2' else None}
        self.emit.progress(1.0, f'{n} steps in {wall:.1f} s', force=True)
        return out

    def tidy(self):
        super().tidy()
        if self.link is not None:
            self.link.close()

    # ------------------------------------------------------------------ pieces
    def read_replay(self, run: str) -> list:
        path = os.path.join(run, 'frames.jsonl')
        try:
            with open(path, encoding='utf-8') as fh:
                rows = [json.loads(x) for x in fh if x.strip()]
        except (OSError, ValueError) as e:
            raise Failed(f'cannot read the run to replay ({path}): {e}') from e
        if not rows or any('control' not in r for r in rows):
            raise Failed(f'{path} has no controls to replay')
        return [dict(r['control']) for r in rows]

    def max_steer_deg(self, ego) -> float:
        """The front wheels' max steer angle (CARLA's physics control), else the kinematic model's 70°."""
        try:
            wheels = ego.get_physics_control().wheels
            return float(max(w.max_steer_angle for w in wheels)) or 70.0
        except (RuntimeError, AttributeError, ValueError):
            return 70.0

    def apply(self, carla, u: dict, kinematic: bool):
        if kinematic:
            self.director.kin_step(self.ego, u)
        else:
            self.ego.apply_control(carla.VehicleControl(throttle=float(u['throttle']), steer=float(u['steer']),
                                                        brake=float(u['brake']),
                                                        hand_brake=bool(u.get('hand_brake', False)),
                                                        reverse=bool(u.get('reverse', False))))

    def director_control(self, director):
        """The director's own command when it drives the ego (unrounded), else the vehicle's control."""
        if director is not None and director.ego_control is not None:
            return dict(director.ego_control)
        c = self.ego.get_control()
        return {'throttle': c.throttle, 'steer': c.steer, 'brake': c.brake, 'hand_brake': c.hand_brake,
                'reverse': c.reverse}

    def ego_now(self, snap, director):
        """(ego pose doc, velocity, angular velocity deg/s) at the frame just ticked."""
        if director is not None:
            tr, v = director.tf(self.ego), director.vel(self.ego)
        else:
            a = snap.find(self.ego.id)
            tr = a.get_transform() if a is not None else self.ego.get_transform()
            v = a.get_velocity() if a is not None else self.ego.get_velocity()
        a = snap.find(self.ego.id)
        w = a.get_angular_velocity() if a is not None else self.ego.get_angular_velocity()
        doc = {'x': round(tr.location.x, 4), 'y': round(tr.location.y, 4), 'z': round(tr.location.z, 4),
               'roll': round(tr.rotation.roll, 4), 'pitch': round(tr.rotation.pitch, 4),
               'yaw': round(tr.rotation.yaw, 4)}
        return doc, (v.x, v.y, v.z), (w.x, w.y, w.z)

    def spawn_sensors(self, world, lib, ego, rig) -> list:
        """The rig's pinhole cameras (RGB only), lidars (ray-cast), IMUs and GNSS receivers, every step
        (sensor_tick 0). Fisheye cameras and radars are not bridged (noted)."""
        from .lidar import lidar_blueprints
        out = []
        for s in rig['sensors']:
            t = s['type']
            tf = scene.camera_transform(self.carla, s['pose'])
            if t == 'camera.pinhole':
                cam = self.cameras.spawn(world, lib, 'sensor.camera.rgb', s['width'], s['height'], s['fov'], tf, ego)
                out.append({'id': s['id'], 'kind': 'camera', 's': s, 'actor': cam})
                continue
            if t == 'lidar':
                bp = lidar_blueprints(lib, s, ROLE_SENSOR)[0]
            elif t in ('imu', 'gnss'):
                bp = lib.find('sensor.other.' + t)
                bp.set_attribute('sensor_tick', '0.0')
                bp.set_attribute('role_name', ROLE_SENSOR)
            else:
                self.note(f'sensor {s["id"]} ({t}) is not published by the bridge (pinhole cameras, lidars, IMUs '
                          'and GNSS are)')
                continue
            a = self.own(world.spawn_actor(bp, tf, attach_to=ego))
            q = queue.Queue()
            self.cameras.queues[a.id] = q
            a.listen(q.put)
            self.alive()
            out.append({'id': s['id'], 'kind': t, 's': s, 'actor': a})
        return out

    def collect(self, sensors, frame) -> dict:
        return {x['id']: self.cameras.get(x['actor'].id, frame) for x in sensors}

    def encode(self, plan, sensors, data, sweeps, i, t_ns, ego, vel, ang, applied, mapper, bridge, rig, first,
               cpms=None):
        """[(topic, CDR bytes)] of a step, in the plan's kinds: sensors first, then /tf, the odometry and the
        vehicle status (the follower and similar stacks act on the odometry, which comes last)."""
        np = self.np
        from ..core import ros2msg as R
        by = {x['id']: x for x in sensors}
        out = []
        tail = []
        for e in plan:
            k = e['kind']
            m = None
            if k == 'clock':
                m = R.clock(t_ns)
            elif k == 'tf_static':
                if first:
                    m = R.static_tf(rig, t_ns)
            elif k in ('image', 'camera_info'):
                x = by[e['sensor']]
                fid = f'{R.link_name(x["s"])}_optical'
                if k == 'image':
                    d = data[x['id']]
                    m = R.image_bgr(d.raw_data, d.width, d.height, t_ns, fid, bridge.get('image_encoding', 'bgr8'))
                else:
                    m = R.camera_info(x['s'], t_ns, fid)
            elif k == 'points':
                x = by[e['sensor']]
                d = data[x['id']]
                sweeps[x['id']].append(np.frombuffer(d.raw_data, np.float32).reshape(-1, 4))
                per = max(int(round(self.fps / max(float(x['s'].get('rotation_frequency', self.fps)), 1e-6))), 1)
                if (i + 1) % per == 0:
                    pts = np.concatenate(sweeps[x['id']]) if sweeps[x['id']] else np.zeros((0, 4), np.float32)
                    sweeps[x['id']].clear()
                    m = R.pointcloud(pts, t_ns, f'{R.link_name(x["s"])}_link')
            elif k == 'imu':
                x = by[e['sensor']]
                d = data[x['id']]
                a, g = d.accelerometer, d.gyroscope
                m = R.imu((a.x, a.y, a.z), (g.x, g.y, g.z), t_ns, f'{R.link_name(x["s"])}_link',
                          float(x['s'].get('noise_accel_stddev', 0.0)) ** 2,
                          float(x['s'].get('noise_gyro_stddev', 0.0)) ** 2)
            elif k == 'fix':
                x = by[e['sensor']]
                d = data[x['id']]
                m = R.navsatfix(d.latitude, d.longitude, d.altitude, t_ns, f'{R.link_name(x["s"])}_link', x['s'])
            elif k in ('cpm', 'cpm_json'):            # a roadside unit's CPM of this step, when it sent one
                line = (cpms or {}).get(e['station'])
                if line is not None:
                    from ..core import cpm as CPM
                    if k == 'cpm':
                        import base64
                        m = R.udp_packet(CPM.btp_payload(base64.b64decode(line['uper'])), t_ns)
                    else:
                        m = {'data': json.dumps({x: line[x] for x in ('step', 'sim_t', 'station', 'station_id', 'cpm',
                                                                       'objects')}, separators=(',', ':'))}
            elif k == 'tf':
                tail.append((e, R.ego_tf(ego, t_ns)))
                continue
            elif k == 'odometry':
                tail.append((e, R.odometry(ego, vel, ang, t_ns)))
                continue
            elif k in ('velocity_status', 'steering_status', 'gear_status', 'control_mode'):
                if k == 'velocity_status':
                    yaw = math.radians(ego['yaw'])
                    lon = vel[0] * math.cos(yaw) + vel[1] * math.sin(yaw)
                    lat = -(-vel[0] * math.sin(yaw) + vel[1] * math.cos(yaw))     # ROS: positive left
                    sign = -1.0 if applied.get('reverse') else 1.0
                    tail.append((e, R.velocity_report(sign * abs(lon), lat, -math.radians(ang[2]), t_ns)))
                elif k == 'steering_status':
                    tail.append((e, R.steering_report(mapper.tyre_angle(applied.get('steer', 0.0)), t_ns)))
                elif k == 'gear_status':
                    tail.append((e, R.gear_report(bool(applied.get('reverse')), t_ns)))
                else:
                    tail.append((e, R.control_mode_report(t_ns)))
                continue
            if m is not None:
                out.append((e['topic'], R.encode(e['type'], m)))
        for e, m in tail:
            out.append((e['topic'], R.encode(e['type'], m)))
        return out

    def wait_message(self, kind: str, timeout: float):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.check()
            try:
                h, _ = self.link.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if h.get('t') == 'closed':
                raise Failed(f'the bridge link closed: {h.get("why")}')
            if h.get('t') == kind:
                return h
        return None

    def wait_control(self, step, t_ns, control, dl, limit):
        """(command or None when the deadline passed, the node's publish time in µs or None)."""
        from ..core import ros2msg as R
        self.phase('stack')                    # waiting on the stack, not the server: the watchdog does not count it
        t0 = time.monotonic()
        pub_us = None
        try:
            while True:
                self.check()
                left = limit - (time.monotonic() - t0)
                if left <= 0:
                    return None, pub_us
                try:
                    h, payload = self.link.q.get(timeout=min(left, 0.25))
                except queue.Empty:
                    continue
                t = h.get('t')
                if t == 'closed':
                    raise Failed(f'the bridge link closed at step {step}: {h.get("why")}')
                if t == 'published' and int(h.get('step', -1)) == step:
                    pub_us = int(h.get('publish_us', 0))
                elif t == 'control':
                    try:
                        cmd = R.command_of(control['type'], R.decode(control['type'], payload))
                    except (ValueError, KeyError, IndexError) as e:
                        self.note(f'step {step}: an undecodable control message ({e})')
                        continue
                    if dl.accept(step, int(h.get('step', -1)), t_ns, cmd.get('stamp_ns', 0)):
                        return cmd, pub_us
        finally:
            self.phase('run')


def _round_control(u: dict) -> dict:
    """The control as recorded: full precision (JSON keeps a float exactly), so a replay applies the very same."""
    return {'throttle': float(u.get('throttle', 0.0)), 'steer': float(u.get('steer', 0.0)),
            'brake': float(u.get('brake', 0.0)), 'hand_brake': bool(u.get('hand_brake', False)),
            'reverse': bool(u.get('reverse', False))}


def _pct(v) -> dict | None:
    if not v:
        return None
    import numpy as np
    a = np.asarray(v, float)
    return {'mean': round(float(a.mean()), 3), 'p50': round(float(np.percentile(a, 50)), 3),
            'p95': round(float(np.percentile(a, 95)), 3), 'max': round(float(a.max()), 3), 'n': int(len(a))}


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker stack')
    ap.add_argument('config', help='stack config JSON (falcon_autonomy.core.stack.stack_config)')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return Stack(cfg, argv, a.config).execute()
