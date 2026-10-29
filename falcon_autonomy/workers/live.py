"""live: a live session — the world stepped in synchronous mode at a fixed step and paced to the wall clock, a view
streamed to the GUI over the live link (core.livelink), driven by hand, by CARLA's autopilot or by a policy, with
live events and recording.

    python -m falcon_autonomy.workers live CONFIG.json

CONFIG.json (core.live.live_config) is a drive config plus: socket (the GUI's live-link path), view ({kind: chase |
free | sensor, sensor, width, height, fov}), mode (manual | autopilot | policy), policy ("module:Class"),
recordings_dir, and no step count — the session runs until stopped (stdin "stop" or "cancel").

Commands on stdin (core.protocol), besides cancel:
    {"cmd": "control", "throttle", "steer", "brake", "hand_brake", "reverse"}   manual driving (latest wins; with no
                                                    control for 0.5 s the ego brakes: a lost input never drives on)
    {"cmd": "mode", "mode": "manual" | "autopilot" | "policy"}
    {"cmd": "view", "kind": "chase" | "free" | "sensor", "sensor": id}
    {"cmd": "free_camera", "move": [forward, right, up], "turn": [yaw, pitch]}   metres and degrees, the free view
    {"cmd": "overlay", "boxes": true}                ground-truth boxes projected into the view
    {"cmd": "event", "kind": "pedestrian" | "cut_in" | "weather" | "sensor_fail", ...}
    {"cmd": "record", "on": true | false}           record the rig's cameras with their labels (a falcon-raw run)
    {"cmd": "stop"}                                 end the session normally (recordings are kept)

Pacing: each step is simulated, its view sent, then the worker sleeps until the step's wall-clock time; when a step
takes longer than the step the session runs slower than real time and says so ("rtf" < 1) — it never skips steps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

from ..core import protocol as P
from ..core.livelink import Sender
from ..core.provenance import new_run, read_json, utc_now, write_json
from . import scene
from .common import ROLE_PREFIX, Failed, Worker

LIVE_SCHEMA = 'falcon.live_config/1'
CHASE = {'x': -6.5, 'y': 0.0, 'z': 2.8, 'roll': 0.0, 'pitch': -12.0, 'yaw': 0.0}
CONTROL_TIMEOUT_S = 0.5
ACTOR_RANGE_M = 120.0
BOX_RANGE_M = 80.0
JPEG_QUALITY = 80


def transform_matrix(loc, rot):
    """CARLA's Transform::GetMatrix (actor to world), as nested lists (no numpy needed by callers)."""
    import numpy as np
    cy, sy = math.cos(math.radians(rot.yaw)), math.sin(math.radians(rot.yaw))
    cr, sr = math.cos(math.radians(rot.roll)), math.sin(math.radians(rot.roll))
    cp, sp = math.cos(math.radians(rot.pitch)), math.sin(math.radians(rot.pitch))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, loc.x],
                     [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, loc.y],
                     [sp, -cp * sr, cp * cr, loc.z],
                     [0.0, 0.0, 0.0, 1.0]])


class Live(Worker):
    name = 'live'

    def __init__(self, cfg: dict, argv: list, cfg_path: str = ''):
        super().__init__(argv, cfg['out'], cfg.get('timeout', 20.0), cfg.get('seed', 0), cfg,
                         {'config': cfg_path, 'xodr': cfg.get('xodr')} if cfg_path else None)
        self.cfg = cfg
        self.cameras = scene.Cameras(self)
        self.lock = threading.Lock()
        self.cmds: list = []
        self.stop_requested = False
        self.mode = cfg.get('mode', 'autopilot')
        self.control = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0, 'hand_brake': False, 'reverse': False}
        self.control_t = 0.0
        self.view = dict(cfg.get('view') or {'kind': 'chase'})
        self.view_cam = None
        self.overlay = bool(cfg.get('overlay', True))
        self.free_tf = None
        self.failures: dict = {}          # sensor id or 'view' -> (mode, until_sim_t)
        self.last_frames: dict = {}       # for freeze failures
        self.events: list = []            # (sim_t, kind, detail)
        self.rec = None                   # the recording in progress: dict
        self.recordings: list = []
        self.policy = None
        self.link = None
        self.physics = None               # core.sensorphysics.Session (the config's sensor_physics)
        self.watch = None                 # oddwatch.OddWatch (the config's odd)
        self.odd_state = None

    # ------------------------------------------------------------------ commands (stdin thread)
    def on_command(self, cmd: dict):
        with self.lock:
            if cmd.get('cmd') == 'control':
                for k in ('throttle', 'steer', 'brake'):
                    if k in cmd:
                        self.control[k] = float(cmd[k])
                for k in ('hand_brake', 'reverse'):
                    if k in cmd:
                        self.control[k] = bool(cmd[k])
                self.control_t = time.monotonic()
            elif cmd.get('cmd') == 'stop':
                self.stop_requested = True
            else:
                self.cmds.append(cmd)

    # ------------------------------------------------------------------ the session
    def body(self) -> dict:
        cfg = self.cfg
        if cfg.get('schema') != LIVE_SCHEMA:
            raise Failed(f'not a live config (schema {cfg.get("schema")!r}, expected {LIVE_SCHEMA})')
        carla = self.import_carla()
        import numpy as np
        self.np = np
        self.link = Sender(cfg['socket'])
        self.connect(cfg.get('host', '127.0.0.1'), cfg['port'], cfg.get('connect_wait_s', 60.0))
        world = scene.load_map(self, cfg)
        seed = int(cfg.get('seed', 0))
        self.rng = np.random.default_rng(seed)
        fps = float(cfg.get('fps', 20.0))
        self.dt = 1.0 / fps
        scene.synchronous(self, world, fps, seed, cfg.get('tm_port', 8310))
        world.set_weather(scene.weather_params(carla, cfg.get('weather')))
        world.tick()
        self.alive()
        self.lib = world.get_blueprint_library()
        self.map = world.get_map()
        spawns = list(self.map.get_spawn_points())
        if not spawns:
            raise Failed('the map has no spawn points')
        order = list(self.rng.permutation(len(spawns)))
        self.ego = scene.spawn_ego(self, world, self.lib, spawns, order, cfg['rig']['vehicle'],
                                   autopilot=self.mode == 'autopilot')
        self.guard = scene.RoadEndGuard(self, world, self.ego, spawns)      # roads that end: never fall out
        scene.spawn_traffic(self, world, self.lib, spawns, order, self.rng, cfg.get('traffic') or {})
        scene.spawn_walkers(self, world, self.lib, self.rng, cfg.get('traffic') or {})
        if cfg.get('policy'):
            from ..plugins import load
            self.policy = load(cfg['policy'], **(cfg.get('policy_params') or {}))
            self.policy.reset(seed)
        self.send_map()
        self.set_view(world, self.view)
        from .drive import physics_and_watch
        self.physics, self.watch = physics_and_watch(self, world, cfg)
        self.phase('run')
        for _ in range(int(cfg.get('warmup_steps', 5))):
            self.check()
            self.cameras.drain(world.tick())
            self.alive()
        t_start = time.monotonic()
        next_t = t_start
        hist = []                         # (wall, sim_t) for the real-time factor
        steps = 0
        sim_t0 = None
        while not self.stop_requested:
            self.check()
            self.apply_commands(world)
            self.drive_step()
            frame = world.tick()
            self.alive()
            snap = world.get_snapshot()
            sim_t = snap.timestamp.elapsed_seconds
            sim_t0 = sim_t if sim_t0 is None else sim_t0
            st = sim_t - sim_t0
            if not self.ego.is_alive:
                raise Failed('the ego was destroyed by the simulator (it left the world below its kill depth)')
            ev = self.guard.step(st)
            if ev is not None:
                self.events.append((st, 'road_end', ev))
                self.emit.log(f'the road ended at ({ev["from"]["x"]:.0f}, {ev["from"]["y"]:.0f}) — the ego was '
                              f'falling and was put back on the road at ({ev["to"]["x"]:.0f}, {ev["to"]["y"]:.0f})',
                              'warning')
            steps += 1
            now = time.monotonic()
            hist.append((now, st))
            while hist and now - hist[0][0] > 2.0:
                hist.pop(0)
            rtf = (hist[-1][1] - hist[0][1]) / max(hist[-1][0] - hist[0][0], 1e-6) if len(hist) > 1 else 1.0
            tps = (len(hist) - 1) / max(hist[-1][0] - hist[0][0], 1e-6) if len(hist) > 1 else 0.0
            img = self.cameras.get(self.view_cam.id, frame)
            if self.watch is not None:
                self.odd_state = self.watch.step(st, self.ego, None, scene.rgb_of(img) if self.watch.wants_frame()
                                                 else None)
            if self.rec is not None:
                self.record_step(frame, st)
            self.send_frame(img, frame, st, now - t_start, tps, rtf)
            self.expire_failures(st)
            next_t += self.dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -self.dt:
                next_t = time.monotonic()        # behind: pace from now on, never catch up by bursting
        if self.rec is not None:
            self.stop_recording()
        wall = time.monotonic() - t_start
        perf = {'steps': steps, 'wall_s': round(wall, 3), 'sim_s': round(steps * self.dt, 3),
                'realtime_factor': round(steps * self.dt / max(wall, 1e-6), 3),
                'view_frames_sent': self.link.sent, 'view_frames_dropped': self.link.dropped,
                'view': self.view, 'source': 'measured'}
        return {'performance': perf, 'events': [{'sim_t': round(t, 3), 'kind': k, 'detail': d}
                                                for t, k, d in self.events],
                'recordings': self.recordings,
                'sensor_physics': self.physics.report() if self.physics is not None else {'enabled': False},
                'odd_monitor': self.watch.report() if self.watch is not None else None,
                'summary': {'steps': steps, 'realtime_factor': perf['realtime_factor'],
                            'recordings': len(self.recordings)}, 'name': cfg.get('name', 'live session')}

    def tidy(self):
        try:
            if self.link is not None:
                self.link.close('session ended')
        finally:
            self.cameras.stop()
            super().tidy()

    # ------------------------------------------------------------------ the map and the view
    def send_map(self):
        wps = self.map.generate_waypoints(3.0)
        xs = [round(w.transform.location.x, 2) for w in wps]
        ys = [round(w.transform.location.y, 2) for w in wps]
        yaws = [round(w.transform.rotation.yaw, 1) for w in wps]
        rule = self.cfg.get('handedness', 'RHT')
        self.link.put_urgent({'t': 'map', 'x': xs, 'y': ys, 'yaw': yaws, 'step_m': 3.0,
                              'name': self.map.name.rpartition('/')[2], 'rule': rule})

    def set_view(self, world, view: dict):
        carla = self.carla
        if self.view_cam is not None:
            self.view_cam.stop()
            self.cameras.remove(self.view_cam)
            self.view_cam.destroy()
            self.spawned.remove(self.view_cam)
            self.view_cam = None
        kind = view.get('kind', 'chase')
        w, h = int(view.get('width', 1280)), int(view.get('height', 720))
        fov = float(view.get('fov', 90.0))
        if kind == 'sensor':
            s = next((x for x in self.cfg['rig']['sensors'] if x['id'] == view.get('sensor')), None)
            if s is None or s['type'] != 'camera.pinhole':
                self.emit.log(f'no pinhole camera {view.get("sensor")!r} in the rig: showing the chase view', 'warning')
                kind = 'chase'
            else:
                fov = float(s['fov'])
                h = int(round(w * s['height'] / s['width']))
                self.view_cam = self.cameras.spawn(world, self.lib, 'sensor.camera.rgb', w, h, fov,
                                                   scene.camera_transform(carla, s['pose']), self.ego)
        if kind == 'free':
            if self.free_tf is None:
                t = self.ego.get_transform()
                yaw = math.radians(t.rotation.yaw)
                self.free_tf = carla.Transform(carla.Location(t.location.x - 12 * math.cos(yaw),
                                                              t.location.y - 12 * math.sin(yaw),
                                                              t.location.z + 8.0),
                                               carla.Rotation(pitch=-25.0, yaw=t.rotation.yaw))
            self.view_cam = self.cameras.spawn(world, self.lib, 'sensor.camera.rgb', w, h, fov, self.free_tf)
        if kind == 'chase':
            self.view_cam = self.cameras.spawn(world, self.lib, 'sensor.camera.rgb', w, h, fov,
                                               scene.camera_transform(carla, CHASE), self.ego)
        self.view = dict(view, kind=kind, width=w, height=h, fov=fov)
        self.view_K = scene.intrinsics(w, h, fov)

    # ------------------------------------------------------------------ commands, applied between steps
    def apply_commands(self, world):
        with self.lock:
            cmds, self.cmds = self.cmds, []
        for c in cmds:
            k = c.get('cmd')
            if k == 'mode':
                self.set_mode(c.get('mode', 'autopilot'))
            elif k == 'view':
                self.set_view(world, dict(self.view, kind=c.get('kind', 'chase'), sensor=c.get('sensor')))
            elif k == 'free_camera' and self.view.get('kind') == 'free':
                self.move_free(c)
            elif k == 'overlay':
                self.overlay = bool(c.get('boxes', True))
            elif k == 'event':
                self.event(world, c)
            elif k == 'record':
                if c.get('on') and self.rec is None:
                    self.start_recording(world)
                elif not c.get('on') and self.rec is not None:
                    self.stop_recording()

    def set_mode(self, mode):
        if mode == 'policy' and self.policy is None:
            from ..plugins import BUILTIN_POLICIES, load
            self.policy = load(BUILTIN_POLICIES['lane_follower'])
            self.policy.reset(int(self.cfg.get('seed', 0)))
        if mode == self.mode:
            return
        self.ego.set_autopilot(mode == 'autopilot', self.tm.get_port())
        self.mode = mode
        self.log_event('mode', mode)

    def move_free(self, c):
        carla = self.carla
        t = self.free_tf
        fwd, right, up = (list(c.get('move') or [0, 0, 0]) + [0, 0, 0])[:3]
        dyaw, dpitch = (list(c.get('turn') or [0, 0]) + [0, 0])[:2]
        yaw = math.radians(t.rotation.yaw)
        loc = carla.Location(t.location.x + fwd * math.cos(yaw) - right * math.sin(yaw),
                             t.location.y + fwd * math.sin(yaw) + right * math.cos(yaw), t.location.z + up)
        rot = carla.Rotation(pitch=max(min(t.rotation.pitch + dpitch, 89.0), -89.0), yaw=t.rotation.yaw + dyaw)
        self.free_tf = carla.Transform(loc, rot)
        self.view_cam.set_transform(self.free_tf)

    def drive_step(self):
        carla = self.carla
        if self.mode == 'manual':
            with self.lock:
                c = dict(self.control)
                stale = time.monotonic() - self.control_t > CONTROL_TIMEOUT_S
            if stale:
                c.update(throttle=0.0, brake=1.0)
            self.ego.apply_control(carla.VehicleControl(throttle=c['throttle'], steer=c['steer'], brake=c['brake'],
                                                        hand_brake=c['hand_brake'], reverse=c['reverse']))
            self.control_applied = c
        elif self.mode == 'policy':
            obs = self.observation()
            u = self.policy.act(obs)
            self.ego.apply_control(carla.VehicleControl(throttle=float(u.get('throttle', 0)),
                                                        steer=float(u.get('steer', 0)),
                                                        brake=float(u.get('brake', 0))))
            self.control_applied = u
        else:
            self.control_applied = None

    def observation(self) -> dict:
        t = self.ego.get_transform()
        v = self.ego.get_velocity()
        speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        wp = self.map.get_waypoint(t.location)
        route = []
        cur = wp
        for _ in range(25):
            nxt = cur.next(2.0)
            if not nxt:
                break
            cur = min(nxt, key=lambda n: abs(((n.transform.rotation.yaw - cur.transform.rotation.yaw) + 180) % 360
                                              - 180))       # straight-most through junctions
            limit = None
            route.append([cur.transform.location.x, cur.transform.location.y, limit])
        bb = self.ego.bounding_box.extent
        return {'t': 0.0, 'dt': self.dt,
                'ego': {'x': t.location.x, 'y': t.location.y, 'z': t.location.z, 'yaw': t.rotation.yaw,
                        'speed_mps': speed, 'length': 2 * bb.x, 'width': 2 * bb.y},
                'route': route, 'lead': self.lead(t, route)}

    def lead(self, t, route):
        """The nearest vehicle within 1.8 m of the route ahead: distance along the route and speed."""
        if not route:
            return None
        best = None
        for a in self.world.get_actors().filter('vehicle.*'):
            if a.id == self.ego.id:
                continue
            l = a.get_location()
            for k, p in enumerate(route):
                if abs(l.x - p[0]) < 2.5 and abs(l.y - p[1]) < 2.5 and math.hypot(l.x - p[0], l.y - p[1]) < 1.8:
                    # bumper to bumper: the route distance less both half-lengths
                    d = max(2.0 * (k + 1) - self.ego.bounding_box.extent.x - a.bounding_box.extent.x, 0.0)
                    if best is None or d < best[0]:
                        av = a.get_velocity()
                        best = (d, math.sqrt(av.x ** 2 + av.y ** 2))
                    break
        return None if best is None else {'distance_m': best[0], 'speed_mps': best[1]}

    # ------------------------------------------------------------------ events
    def log_event(self, kind, detail):
        t = self.world.get_snapshot().timestamp.elapsed_seconds
        self.events.append((t, kind, detail))
        self.emit.log(f'event: {kind} {detail}')
        if self.rec is not None:
            self.rec['events'].write(json.dumps({'sim_t': round(t - self.rec['t0'], 3), 'kind': kind,
                                                 'detail': detail}) + '\n')

    def event(self, world, c):
        kind = c.get('kind')
        try:
            if kind == 'pedestrian':
                self.ev_pedestrian(world, float(c.get('distance_m', 25.0)), c.get('side', 'right'))
            elif kind == 'cut_in':
                self.ev_cut_in(world, c.get('side', 'left'))
            elif kind == 'weather':
                fields = c.get('fields') or {}
                world.set_weather(scene.weather_params(self.carla, fields))
                if self.physics is not None:
                    from ..core.weather import resolve
                    self.physics.set_weather(resolve(fields))
                self.log_event('weather', {k: fields[k] for k in sorted(fields)[:6]})
            elif kind == 'sensor_fail':
                sid = c.get('sensor', 'view')
                until = world.get_snapshot().timestamp.elapsed_seconds + float(c.get('seconds', 3.0))
                self.failures[sid] = (c.get('mode', 'black'), until)
                self.log_event('sensor_fail', {'sensor': sid, 'mode': c.get('mode', 'black'),
                                               'seconds': float(c.get('seconds', 3.0))})
            else:
                self.emit.log(f'unknown event {kind!r}', 'warning')
        except Failed as e:
            self.emit.log(f'event {kind} not possible here: {e}', 'warning')

    def ev_pedestrian(self, world, distance, side):
        carla = self.carla
        wp = self.map.get_waypoint(self.ego.get_location())
        ahead = wp.next(distance)
        if not ahead:
            raise Failed('the lane ends before that distance')
        tgt = ahead[0].transform
        yaw = math.radians(tgt.rotation.yaw)
        sgn = 1.0 if side == 'right' else -1.0
        off = ahead[0].lane_width / 2.0 + 2.0
        rx, ry = -math.sin(yaw) * sgn, math.cos(yaw) * sgn            # towards the chosen side (CARLA: y right)
        loc = carla.Location(tgt.location.x + rx * off, tgt.location.y + ry * off, tgt.location.z + 1.0)
        bps = sorted(self.lib.filter('walker.pedestrian.*'), key=lambda b: b.id)
        bp = bps[int(self.rng.integers(len(bps)))]
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        bp.set_attribute('role_name', ROLE_PREFIX + 'walker')
        walker = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=math.degrees(math.atan2(-ry, -rx)))))
        if walker is None:
            raise Failed('no room to place the pedestrian')
        self.own(walker)
        walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(-rx, -ry, 0.0), speed=1.4))
        self.log_event('pedestrian', {'distance_m': distance, 'side': side, 'actor': walker.id})

    def ev_cut_in(self, world, side):
        wp = self.map.get_waypoint(self.ego.get_location())
        adj = wp.get_left_lane() if side == 'left' else wp.get_right_lane()
        if adj is None or adj.lane_type != self.carla.LaneType.Driving or (adj.lane_id > 0) != (wp.lane_id > 0):
            other = wp.get_right_lane() if side == 'left' else wp.get_left_lane()
            if other is not None and other.lane_type == self.carla.LaneType.Driving and \
                    (other.lane_id > 0) == (wp.lane_id > 0):
                adj, side = other, 'right' if side == 'left' else 'left'
            else:
                raise Failed('no adjacent lane in the same direction')
        back = adj.previous(8.0)
        start = (back[0] if back else adj).transform
        start.location.z += 0.5
        bps = sorted(scene.blueprints_by_type(self.lib).get('car', []), key=lambda b: b.id)
        bp = bps[int(self.rng.integers(len(bps)))]
        bp.set_attribute('role_name', ROLE_PREFIX + 'cutin')
        car = world.try_spawn_actor(bp, start)
        if car is None:
            raise Failed('no room for the cutting-in car')
        self.own(car)
        car.set_autopilot(True, self.tm.get_port())
        v = self.ego.get_velocity()
        self.tm.set_desired_speed(car, 3.6 * math.sqrt(v.x ** 2 + v.y ** 2) + 15.0)
        self.tm.auto_lane_change(car, False)
        # towards the ego's lane: from its left lane that is a change to the right (CARLA: True = right)
        self.pending_cut = (car, side == 'left', world.get_snapshot().timestamp.elapsed_seconds + 1.5)
        self.tm.force_lane_change(car, side == 'left')
        self.log_event('cut_in', {'side': side, 'actor': car.id})

    def expire_failures(self, st):
        now = self.world.get_snapshot().timestamp.elapsed_seconds
        for sid, (_, until) in list(self.failures.items()):
            if now >= until:
                del self.failures[sid]
                self.log_event('sensor_restored', {'sensor': sid})

    def degrade(self, sid, rgb):
        """A failed sensor's image: black, frozen (the last good one) or none (dropped)."""
        f = self.failures.get(sid)
        if f is None:
            self.last_frames[sid] = rgb
            return rgb
        mode = f[0]
        if mode == 'black':
            return rgb * 0
        if mode == 'freeze':
            return self.last_frames.get(sid, rgb)
        return None

    # ------------------------------------------------------------------ the view frame
    def send_frame(self, img, frame, st, wall, tps, rtf):
        import cv2
        np = self.np
        bgr = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
        bgr = self.degrade('view', bgr)
        effects = None
        if bgr is not None and self.physics is not None and self.view.get('kind') == 'sensor':
            fr = self.physics.camera_frame(self.view.get('sensor'), self.dt, None)
            if fr is not None:                           # a rig camera's view: its weather physics (no headlights)
                bgr, effects = fr[0].apply(np.ascontiguousarray(bgr), fr[1])
        actors, boxes = self.actors_and_boxes(img)
        hdr = {'t': 'frame', 'frame': int(frame), 'sim_t': round(st, 3), 'wall_t': round(wall, 3),
               'view': self.view.get('kind'), 'w': img.width, 'h': img.height,
               'ticks_per_s': round(tps, 2), 'rtf': round(rtf, 3), 'sent': self.link.sent,
               'dropped': self.link.dropped, 'mode': self.mode,
               'speed_kmh': round(3.6 * scene.ego_state(self.ego)['speed_mps'], 1),
               'control': self.control_applied, 'actors': actors, 'boxes': boxes,
               'recording': self.rec['dir'] if self.rec else None,
               'failures': {k: v[0] for k, v in self.failures.items()}, 'odd': self.odd_state,
               'physics': effects}
        if bgr is None:
            self.link.put(dict(hdr, failed=True), b'')
            return
        ok, jpg = cv2.imencode('.jpg', np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        self.link.put(hdr, jpg.tobytes() if ok else b'')

    def actors_and_boxes(self, img):
        """Vehicles and walkers near the ego for the top-down map, and their full 3D boxes projected into the
        view (the 8 corners' extremes, clipped to the image; occlusion is not considered — a full box)."""
        np = self.np
        ego_l = self.ego.get_location()
        cam_t = self.view_cam.get_transform()
        W2C = np.linalg.inv(transform_matrix(cam_t.location, cam_t.rotation))
        K = self.view_K
        actors, boxes = [], []
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        for a in self.world.get_actors():
            tid = a.type_id
            if not tid.startswith(('vehicle.', 'walker.')):
                continue
            t = a.get_transform()
            d = math.hypot(t.location.x - ego_l.x, t.location.y - ego_l.y)
            if d > ACTOR_RANGE_M:
                continue
            kind = 'e' if a.id == self.ego.id else ('w' if tid.startswith('walker.') else 'v')
            bb = a.bounding_box
            actors.append([a.id, kind, round(t.location.x, 2), round(t.location.y, 2), round(t.rotation.yaw, 1),
                           round(bb.extent.x, 2), round(bb.extent.y, 2)])
            if not self.overlay or kind == 'e' or d > BOX_RANGE_M:
                continue
            M = transform_matrix(t.location, t.rotation)
            P_local = corners * np.array([bb.extent.x, bb.extent.y, bb.extent.z]) + \
                np.array([bb.location.x, bb.location.y, bb.location.z])
            Pw = (M @ np.c_[P_local, np.ones(8)].T).T
            Pc = (W2C @ Pw.T).T                         # camera: x forward, y right, z up
            front = Pc[:, 0] > 0.1
            if not front.any():
                continue
            Pc = Pc[front]
            u = K[0][0] * Pc[:, 1] / Pc[:, 0] + K[0][2]
            v = -K[1][1] * Pc[:, 2] / Pc[:, 0] + K[1][2]
            x0, x1 = max(u.min(), 0.0), min(u.max(), img.width - 1.0)
            y0, y1 = max(v.min(), 0.0), min(v.max(), img.height - 1.0)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            boxes.append([a.id, kind, round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])
        return actors, boxes

    # ------------------------------------------------------------------ recording
    def start_recording(self, world):
        n = len(self.recordings) + 1
        base = self.cfg.get('recordings_dir') or os.path.join(self.out_dir, 'recordings')
        d = os.path.join(base, f'{os.path.basename(self.out_dir)}-rec-{n:03d}')
        os.makedirs(d, exist_ok=True)
        from . import camgt
        cams = scene.Cameras(self)
        rig_cams = [x for x in self.cfg['rig']['sensors'] if x['type'] in ('camera.pinhole', 'camera.fisheye')]
        if not hasattr(self, 'level'):
            self.level = []                     # (level boxes are read before vehicles exist: not mid-session)
            self.raster = camgt.LaneRaster(self.carla, self.map) if self.cfg.get('xodr') else None
        recs = [camgt.CameraRecorder(self, world, self.lib, self.ego, x, cams, x['pose']) for x in rig_cams]
        t0 = world.get_snapshot().timestamp.elapsed_seconds
        run = new_run('live-recording', self.argv, self.cfg.get('seed', 0), self.cfg)
        run['provenance']['host'].update(self.run['provenance']['host'])
        self.rec = {'dir': d, 'cams': cams, 'recs': recs, 'n': 0, 't0': t0, 'run': run, 'started': time.monotonic(),
                    'warnings': ([] if self.raster is None else ['ground relabelled from the map (see docs/worlds.md)']),
                    'index': open(os.path.join(d, 'frames.jsonl'), 'w', encoding='utf-8'),
                    'events': open(os.path.join(d, 'events.jsonl'), 'w', encoding='utf-8')}
        self.log_event('record', {'on': True, 'dir': d})
        self.rec['cams'].drain(world.tick())         # the new cameras' first frame
        self.alive()

    def record_step(self, frame, st):
        r = self.rec
        cams = r['cams']
        i = r['n']
        rec = {'step': i, 'frame': int(frame), 'sim_t': round(
            self.world.get_snapshot().timestamp.elapsed_seconds - r['t0'], 6), 'files': {}, 'mode': self.mode}
        rec.update(scene.ego_state(self.ego))
        rec['control'] = self.control_applied
        from . import camgt
        boxes = camgt.scene_boxes(self, self.world, self.ego, self.level) if any(c.gt for c in r['recs']) else None
        for c in r['recs']:
            s = c.s
            failed = self.failures.get(s['id'])
            imgs = c.collect(frame)
            if failed is not None and failed[0] == 'drop':
                continue
            if failed is not None:
                rec.setdefault('faults', {})[s['id']] = failed[0]
            for kind in c.kinds():
                ext = {'rgb': 'jpg', 'objects': 'json', 'flow': 'npy'}.get(kind, 'png')
                rec['files'][f'{s["id"]}/{kind}'] = os.path.join(kind, s['id'], f'{i:06d}.{ext}')
            phys = None if self.physics is None else self.physics.camera_frame(s['id'], self.dt, boxes)
            if phys is not None and phys[1]['effects']:
                rec.setdefault('physics', {})[s['id']] = phys[1]['effects']
            cams.pending.append(cams.pool.submit(c.write, imgs, i, r['dir'], boxes, self.raster, None, phys))
        r['index'].write(json.dumps(rec, separators=(',', ':')) + '\n')
        r['n'] += 1
        if len(cams.pending) > 8 * max(len(r['recs']), 1):
            cams.flush(8)

    def stop_recording(self):
        r, self.rec = self.rec, None
        r['cams'].flush(0)
        r['cams'].stop()
        for c in r['recs']:
            for d in c.views:
                for sensor in d.values():
                    sensor.stop()
                    sensor.destroy()
                    if sensor in self.spawned:
                        self.spawned.remove(sensor)
        r['index'].close()
        r['events'].close()
        wall = time.monotonic() - r['started']
        write_json(os.path.join(r['dir'], 'dataset.json'), scene.dataset_doc(
            f'{self.cfg.get("name", "live")} recording', r['recs'], r['n'], 1.0 / self.dt, r['run'],
            {'town': self.cfg.get('town'), 'xodr': self.cfg.get('xodr'), 'live': True, 'events': 'events.jsonl',
             'label_warnings': r['warnings']}))
        run = r['run']
        run.update(status='ok', exit_code=0, finished=utc_now(), name=f'{self.cfg.get("name", "live")} recording',
                   notes=list(r['warnings']),
                   performance={'steps': r['n'], 'wall_s': round(wall, 3), 'sim_s': round(r['n'] * self.dt, 3),
                                'realtime_factor': round(r['n'] * self.dt / max(wall, 1e-6), 3),
                                'source': 'measured'},
                   summary={'steps': r['n'], 'cameras': len(r['recs']), 'live': True})
        write_json(os.path.join(r['dir'], 'run.json'), run)
        self.recordings.append(r['dir'])
        self.emit.send({'t': 'recording', 'path': r['dir'], 'steps': r['n']})
        self.events.append((self.world.get_snapshot().timestamp.elapsed_seconds, 'record', {'on': False}))


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker live')
    ap.add_argument('config', help='live config JSON (falcon_autonomy.core.live)')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return Live(cfg, argv, a.config).execute()
