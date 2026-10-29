"""gym: a driving environment for learning — CARLA stepped in synchronous mode at a fixed step, one step per agent
action, observations, reward terms, termination and infractions per step (the definitions are core.drivefeatures';
the client is core.gymenv.FalconDriveEnv, in the app's Python).

    python -m falcon_autonomy.workers gym CONFIG.json

CONFIG.json (schema falcon.gym_config/1, core.gymenv.gym_config writes it): the server and the map (a town or an
OpenDRIVE file, as a drive), weather, fps, seed, the ego's blueprint, background traffic (default none), socket (the
client's Unix socket path), the default episode (scenario: a built-in key, a scenario document or null for free
driving; values; motion kinematic | physics; route_m; max_episode_s; target_speed_kmh), observation extras (bev: a
BEV spec or null; camera: {width, height, fov, pose} or null), action (continuous | discrete), reward weights,
thresholds, terminate_on, log_actions, idle_timeout_s, no_rendering.

Commands on stdin (core.protocol), besides cancel:
    {"cmd": "reset", "id": n, "seed": s, "options": {...}}    a new episode (options override the config's episode)
    {"cmd": "step", "id": n, "action": [steer, accel] | index}
    {"cmd": "close"}                                          end normally
Replies go to the client's socket in the live link's framing (core.livelink.pack: a JSON header and a payload):
    {"t": "ready", "observation": {...}, "action": {...}, "features": [...], ...}     once, after setup
    {"t": "obs", "id": n, "reward": r, "terminated": b, "truncated": b, "info": {...}, "arrays": [{"key", "dtype",
     "shape", "nbytes"}, ...]}     payload: the arrays' bytes in that order ("state" float32; "bev" packed bits (H, W)
                                   uint8; "camera" (H, W, 3) uint8 RGB)
    {"t": "error", "id": n, "error": "..."}                   a command that could not be done (the worker goes on)
    {"t": "bye", "reason": "..."}

An episode. reset destroys the previous episode's actors (sensors by destroy(), vehicles and walkers by id in a batch —
none of them was ever on the traffic manager), draws everything from a generator seeded by the reset's seed, places
the ego (a scenario: the director's placement; free driving: the first spawn point in seeded order that is not in a
junction and whose lane runs at least min(route_m, 30) m) and the scenario's actors, resets the traffic lights' cycle
(World.reset_all_traffic_lights), and in kinematic motion places every scenario vehicle and the ego exactly at its
planned start pose with physics off (so CARLA's physics settling never enters the episode). The first observation is
taken after one tick. A step: the director fires the timeline and drives its actors, the agent's action drives the
ego (kinematic: the director's bicycle model; physics: VehicleControl), one tick, then the observation and the step's
facts — on which driving lane the ego's centre is (CARLA's map, project_to_road=False), box contacts with every vehicle
and walker (core.criticality), the collision sensor's new events, red-light crossings of the traffic lights' stop
lines (TrafficLight.get_stop_waypoints, the state after the tick), stop-sign trigger volumes — go to the reward and the
infraction monitor (core.drivefeatures).

Determinism: with kinematic motion, no background traffic and the same seed, an episode is a function of the seed,
the options and the actions (measured on the fake; tests/test_gym.py); CARLA's physics (motion "physics", or
traffic-manager traffic) is not bit-for-bit repeatable (docs/scenarios.md). Traffic lights cycle on simulation time
from the reset, so they repeat too where reset_all_traffic_lights is honoured.

Outputs in the run folder: episodes.jsonl (one line per finished or interrupted episode: seed, options, the concrete
scenario, start, route, return, reward terms, termination, infractions, the actions when log_actions, and a per-step
trace for traffic-rule checking when log_trace — core.rules.TRACE_COLUMNS: pose, speed, lane facts, lead and
vulnerable-road-user gaps, the light ahead, CARLA's speed limit), lanes.json
when the BEV is on, run.json. The watchdog guards every call to the server; between commands the worker is idle (not
guarded) until idle_timeout_s, and it ends when the client's socket closes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import select
import socket
import time

from ..core import drivefeatures as DF
from ..core.rules import TRACE_COLUMNS
from ..core import protocol as P
from ..core.livelink import pack
from ..core.provenance import read_json
from . import scene
from .common import Failed, Worker, place_now

GYM_SCHEMA = 'falcon.gym_config/1'
WARMUP_TICKS = 1
CONTACT_RANGE_M = 15.0
RESET_ATTEMPTS = 5


class BadCommand(Exception):
    pass


class Gym(Worker):
    name = 'gym'

    def __init__(self, cfg: dict, argv: list, cfg_path: str = ''):
        super().__init__(argv, cfg['out'], cfg.get('timeout', 20.0), cfg.get('seed', 0), cfg,
                         {'config': cfg_path, 'xodr': cfg.get('xodr')} if cfg_path else None)
        self.cfg = cfg
        self.cmds: queue.Queue = queue.Queue()
        self.cameras = scene.Cameras(self)
        self.sock = None
        self.ep = None
        self.n_episodes = 0
        self.n_steps = 0
        self.step_wall = 0.0
        self.rendering_changed = None
        self.lane_map = None
        self.ended_by = ''

    # ------------------------------------------------------------------ commands (stdin thread)
    def on_command(self, cmd: dict):
        self.cmds.put(cmd)

    def next_command(self):
        """The next command; None when the client went away or idle_timeout_s passed. Not guarded by the watchdog."""
        self.phase('idle')
        idle = float(self.cfg.get('idle_timeout_s', 900.0))
        t0 = time.monotonic()
        while True:
            self.check()
            try:
                return self.cmds.get(timeout=0.25)
            except queue.Empty:
                pass
            r, _, _ = select.select([self.sock], [], [], 0)
            if r:
                try:
                    gone = self.sock.recv(1, socket.MSG_PEEK) == b''
                except OSError:
                    gone = True
                if gone:
                    self.ended_by = 'the client closed its socket without closing the environment'
                    return None
                self.sock.recv(4096)                       # the client never sends here: discard
            if time.monotonic() - t0 > idle:
                self.ended_by = f'no command for {idle:.0f} s (idle_timeout_s)'
                return None

    def send(self, header: dict, arrays=()):
        """A reply on the socket: header plus arrays [(key, ndarray)] as its payload."""
        heads, blobs = [], []
        for key, a in arrays:
            b = self.np.ascontiguousarray(a)
            heads.append({'key': key, 'dtype': str(b.dtype), 'shape': list(b.shape), 'nbytes': int(b.nbytes)})
            blobs.append(b.tobytes())
        if heads:
            header = dict(header, arrays=heads)
        self.sock.sendall(pack(header, b''.join(blobs)))

    def reply_error(self, cmd, text):
        self.emit.log(f'{cmd.get("cmd")}: {text}', 'warning')
        self.send({'t': 'error', 'id': cmd.get('id'), 'error': str(text)})

    # ------------------------------------------------------------------ the session
    def body(self) -> dict:
        cfg = self.cfg
        if cfg.get('schema') != GYM_SCHEMA:
            raise Failed(f'not a gym config (schema {cfg.get("schema")!r}, expected {GYM_SCHEMA})')
        carla = self.import_carla()
        import numpy as np
        self.np = np
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        try:
            self.sock.connect(cfg['socket'])
        except OSError as e:
            raise Failed(f'cannot connect to the client\'s socket {cfg.get("socket")}: {e}') from e
        self.sock.settimeout(120.0)
        self.connect(cfg.get('host', '127.0.0.1'), cfg['port'], cfg.get('connect_wait_s', 60.0))
        world = scene.load_map(self, cfg)
        self.check()
        seed = int(cfg.get('seed', 0))
        self.fps = float(cfg.get('fps', 20.0))
        self.dt = 1.0 / self.fps
        scene.synchronous(self, world, self.fps, seed, cfg.get('tm_port', 8310))
        if cfg.get('no_rendering'):
            st = world.get_settings()
            self.rendering_changed = bool(st.no_rendering_mode)
            st.no_rendering_mode = True
            world.apply_settings(st)
        world.set_weather(scene.weather_params(carla, cfg.get('weather')))
        world.tick()
        self.alive()
        self.lib = world.get_blueprint_library()
        self.map = world.get_map()
        self.spawns = list(self.map.get_spawn_points())
        if not self.spawns:
            raise Failed('the map has no spawn points')
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(len(self.spawns)))
        traffic = cfg.get('traffic') or {}
        scene.spawn_traffic(self, world, self.lib, self.spawns, order, rng, traffic)
        scene.spawn_walkers(self, world, self.lib, rng, traffic)
        if traffic.get('vehicles') or traffic.get('walkers'):
            self.note('background traffic is driven by CARLA\'s traffic manager and physics: episodes are not '
                      'bit-for-bit repeatable')
        self.background = len(self.spawned)
        self.background_actors = sum(1 for a in self.spawned if a.type_id.startswith(('vehicle.', 'walker.')))
        self.actor_log = scene.ActorLog(self, None)
        self.lights = self.read_lights(world)
        self.stops = self.read_stops(world)
        self.bev = DF.bev_spec(cfg['bev']) if cfg.get('bev') is not None else None
        if self.bev is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            self.expect(180.0)                             # the map's waypoints and the lane raster: slow on big maps
            scene.write_lanes(self, self.map, self.out_dir)
            lanes = read_json(os.path.join(self.out_dir, 'lanes.json'))
            from ..core.worldmodel import LaneMap
            pts = np.asarray([[r[0], r[1]] for r in lanes['points']], float)
            pad = math.hypot(max(self.bev.forward, self.bev.back), max(self.bev.left, self.bev.right)) + 2.0
            self.lane_map = LaneMap(lanes, (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad,
                                            pts[:, 1].max() + pad), self.bev.res / 2.0) if len(pts) else None
            self.phase('setup')
        self.action_kind = cfg.get('action', 'continuous')
        if self.action_kind not in ('continuous', 'discrete'):
            raise Failed(f'action must be continuous or discrete, not {self.action_kind!r}')
        os.makedirs(self.out_dir, exist_ok=True)
        self.ep_log = open(os.path.join(self.out_dir, 'episodes.jsonl'), 'a', encoding='utf-8')
        from .drive import physics_and_watch
        _, self.watch = physics_and_watch(self, world, cfg, with_physics=False)       # the ODD monitor, when asked
        self.send(self.ready_doc())
        t_start = time.monotonic()
        try:
            while True:
                cmd = self.next_command()
                if cmd is None:
                    break
                self.phase('run')
                kind = cmd.get('cmd')
                if kind == 'close':
                    self.ended_by = 'closed by the client'
                    break
                try:
                    if kind == 'reset':
                        self.reset(cmd)
                    elif kind == 'step':
                        self.step(cmd)
                    else:
                        raise BadCommand(f'unknown command {kind!r}')
                except BadCommand as e:
                    self.reply_error(cmd, e)
                except Failed as e:
                    if e.code != P.FAILED:
                        raise
                    self.reply_error(cmd, e)
            if self.ep is not None and not self.ep['done']:
                self.end_episode('interrupted')
        finally:
            self.ep_log.close()
        wall = time.monotonic() - t_start
        perf = {'episodes': self.n_episodes, 'steps': self.n_steps, 'wall_s': round(wall, 3),
                'step_wall_s': round(self.step_wall, 3),
                'steps_per_s': round(self.n_steps / max(self.step_wall, 1e-9), 2) if self.n_steps else None,
                'sim_s': round(self.n_steps * self.dt, 3), 'source': 'measured'}
        if self.n_steps:
            self.emit.metric('steps_per_s', perf['steps_per_s'], '1/s')
        if self.ended_by and self.ended_by != 'closed by the client':
            raise Failed(f'the environment ended: {self.ended_by}')
        return {'performance': perf, 'summary': {'episodes': self.n_episodes, 'steps': self.n_steps,
                                                 'steps_per_s': perf['steps_per_s']},
                'odd_monitor': self.watch.report() if self.watch is not None else None,
                'name': cfg.get('name', 'gym environment')}

    def ready_doc(self) -> dict:
        obs = {'state': {'shape': [DF.N_FEATURES], 'dtype': 'float32'}}
        if self.bev is not None:
            obs['bev'] = {'shape': [8, self.bev.h, self.bev.w], 'dtype': 'uint8', 'spec': self.bev.doc(),
                          'transport': 'packed bits (H, W) uint8; bit k = channel k'}
        cam = self.cfg.get('camera')
        if cam:
            obs['camera'] = {'shape': [int(cam['height']), int(cam['width']), 3], 'dtype': 'uint8'}
        act = {'kind': self.action_kind, 'keys': list(DF.ACTION_KEYS)}
        if self.action_kind == 'discrete':
            act.update(n=len(DF.DISCRETE_ACTIONS), table=[list(a) for a in DF.DISCRETE_ACTIONS])
        else:
            act.update(low=[-1.0, -1.0], high=[1.0, 1.0])
        return {'t': 'ready', 'observation': obs, 'action': act, 'features': [list(f) for f in DF.FEATURES],
                'drivefeatures': DF.SCHEMA, 'fps': self.fps, 'map': self.map.name.rpartition('/')[2],
                'reward_weights': dict(DF.DEFAULT_REWARD, **(self.cfg.get('reward') or {})),
                'thresholds': dict(DF.DEFAULT_THRESHOLDS, **(self.cfg.get('thresholds') or {})),
                'terminate_on': list(self.cfg.get('terminate_on') or DF.DEFAULT_TERMINATE_ON),
                'traffic_lights': len(self.lights), 'stop_signs': len(self.stops), 'pid': os.getpid()}

    # ------------------------------------------------------------------ the map's lights and stop signs
    def read_lights(self, world) -> list:
        """[(traffic light actor, [(x, y, yaw, half width) of each stop line])]: CARLA's stop waypoints."""
        out = []
        for tl in world.get_actors().filter('traffic.traffic_light'):
            lines = []
            try:
                for wp in tl.get_stop_waypoints():
                    t = wp.transform
                    lines.append((t.location.x, t.location.y, t.rotation.yaw, wp.lane_width / 2.0))
            except (RuntimeError, AttributeError):
                continue
            if lines:
                out.append((tl, lines))
        self.alive()
        return out

    def read_stops(self, world) -> list:
        """[(stop sign id, centre x, y, yaw, half extents x, y)] of every stop sign's trigger volume, in the world."""
        out = []
        for a in world.get_actors().filter('traffic.stop'):
            try:
                tv = a.trigger_volume
                tf = a.get_transform()
                c = tf.transform(tv.location)
                out.append((a.id, c.x, c.y, tf.rotation.yaw + tv.rotation.yaw, tv.extent.x, tv.extent.y))
            except (RuntimeError, AttributeError):
                continue
        self.alive()
        return out

    # ------------------------------------------------------------------ episodes
    def episode_options(self, opts: dict) -> dict:
        cfg = self.cfg
        keys = ('scenario', 'values', 'motion', 'route_m', 'max_episode_s', 'target_speed_kmh', 'start_speed_kmh')
        base = {k: cfg.get(k) for k in keys if cfg.get(k) is not None}
        unknown = sorted(set(opts) - set(keys))
        if unknown:
            raise BadCommand(f'unknown reset options {unknown} (allowed: {list(keys)})')
        out = dict(base, **{k: v for k, v in opts.items()})
        motion = out.get('motion', 'kinematic')
        if motion not in ('kinematic', 'physics'):
            raise BadCommand(f'motion must be kinematic or physics, not {motion!r}')
        return out

    def resolve_scenario(self, o: dict, seed: int):
        from ..core import scenarios as SC
        scn = o.get('scenario')
        if scn in (None, '', 'free'):
            return None
        if isinstance(scn, str):
            if scn not in SC.LIBRARY:
                raise BadCommand(f'no built-in scenario {scn!r} (built-ins: {sorted(SC.LIBRARY)})')
            scn = SC.builtin(scn)
        if not isinstance(scn, dict):
            raise BadCommand('scenario must be a built-in key, a scenario document or null')
        problems = SC.validate(scn)
        if problems:
            raise BadCommand(f'scenario {scn.get("id")!r}: {"; ".join(problems)}')
        conc = SC.sample(scn, seed, o.get('values') or {})
        conc['ego'] = dict(conc.get('ego') or {}, driver='agent')
        conc['motion'] = o.get('motion', 'kinematic')
        return conc

    def destroy_episode(self):
        """Remove the previous episode's actors (its sensors first; nothing of it was on the traffic manager)."""
        ep, self.ep = self.ep, None
        if ep is None:
            return
        mine = self.spawned[self.background:]
        self.spawned = self.spawned[:self.background]
        sensors = [a for a in mine if a.type_id.startswith('sensor.')]
        rest = [a for a in mine if a not in sensors]
        for s in sensors:
            try:
                s.stop()
                self.cameras.remove(s)
                s.destroy()
            except RuntimeError:
                pass
        self.client.apply_batch_sync([self.carla.command.DestroyActor(a.id) for a in reversed(rest)])
        self.alive()

    def our_actor_ids(self) -> set:
        """Ids of the vehicles, walkers and sensors in the world with a Falcon role (falcon_* or hero) — for the tidy
        checks: after the first tick of an episode, those this worker does not own are stale (a previous episode's
        actors not removed); after tidy-up, none should be left."""
        from .common import ROLE_HERO, ROLE_PREFIX, role
        return {a.id for a in self.world.get_actors() if a.type_id.startswith(('vehicle.', 'walker.', 'sensor.'))
                and (role(a).startswith(ROLE_PREFIX) or role(a) == ROLE_HERO)}

    def reset(self, cmd):
        np, carla, world = self.np, self.carla, self.world
        try:
            seed = int(cmd.get('seed', 0))
        except (TypeError, ValueError):
            raise BadCommand(f'seed must be an integer, not {cmd.get("seed")!r}') from None
        opts = cmd.get('options') or {}
        if not isinstance(opts, dict):
            raise BadCommand('options must be an object')
        o = self.episode_options(opts)
        conc = self.resolve_scenario(o, seed)
        if self.ep is not None and not self.ep['done']:
            self.end_episode('interrupted')
        self.destroy_episode()
        if hasattr(world, 'reset_all_traffic_lights'):
            world.reset_all_traffic_lights()
        from .director import Director, lane_path
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(len(self.spawns)))
        motion = o.get('motion', 'kinematic')
        free = conc is None
        if free:
            target = float(o.get('target_speed_kmh', 30.0)) / 3.6
            conc = {'id': 'free', 'name': 'free driving', 'needs': [], 'actors': [], 'timeline': [],
                    'ego': {'speed_kmh': float(o.get('start_speed_kmh', 0.0)), 'driver': 'agent',
                            'route': 'straight'}, 'motion': motion,
                    'duration_s': float(o.get('max_episode_s', 60.0))}
        else:
            target = float(conc['ego'].get('speed_kmh', 30.0)) / 3.6
        dcfg = dict(self.cfg, scenario=conc, fps=self.fps, motion=motion)
        route_m = float(o.get('route_m') or (200.0 if free else 400.0))
        last_err = None
        for _attempt in range(RESET_ATTEMPTS):
            director = Director(self, world, dcfg, self.lib, rng)
            ego = None
            try:
                if free:
                    ego_wp, path = self.free_start(order, route_m, lane_path)
                else:
                    _sp, ego_wp = director.place_ego(self.spawns, order)
                    path = None
                bp = self.lib.find(self.cfg.get('vehicle', 'vehicle.tesla.model3'))
                bp.set_attribute('role_name', 'hero')
                tf = ego_wp.transform
                tf.location.z += 0.3
                ego = world.try_spawn_actor(bp, tf)
                if ego is None:
                    raise Failed('the ego could not be placed at the start (something is in the way)')
                self.own(ego)
                world.tick()
                self.alive()
                stale = len(self.our_actor_ids() - {a.id for a in self.spawned})
                director.setup(ego, ego_wp)
                break
            except Failed as e:
                last_err = e
                self.ep = {'done': True}                    # destroy what this attempt placed
                self.destroy_episode()
                if 'no start on this map' in str(e):
                    raise
        else:
            raise Failed(f'no episode could be placed in {RESET_ATTEMPTS} attempts: {last_err}')
        try:
            self.start_episode(cmd, seed, opts, o, conc, free, target, director, ego, ego_wp, path, route_m, stale)
        except Failed:
            self.ep = {'done': True}                        # nothing half-made is left behind
            self.destroy_episode()
            raise

    def start_episode(self, cmd, seed, opts, o, conc, free, target, director, ego, ego_wp, path, route_m, stale):
        """The rest of a reset once the ego and the scenario's actors are placed: the route, the camera, the first
        tick, the episode's state and the first observation."""
        np, carla, world = self.np, self.carla, self.world
        from .director import Path
        motion = o.get('motion', 'kinematic')
        if path is not None:
            director.ego_path = path
        elif route_m < director.ego_path.s[-1]:
            k = int(np.searchsorted(director.ego_path.s, route_m)) + 1
            p = director.ego_path
            director.ego_path = Path(p.pts[:k], p.yaw[:k], p.w[:k], p.z)
        director.ego_speed = float(conc['ego'].get('speed_kmh', 0.0)) / 3.6
        cam = None
        if self.cfg.get('camera'):
            c = self.cfg['camera']
            pose = dict({'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}, **(c.get('pose') or {}))
            cam = self.cameras.spawn(world, self.lib, 'sensor.camera.rgb', int(c['width']), int(c['height']),
                                     float(c.get('fov', 90.0)), scene.camera_transform(carla, pose), ego)
        for _ in range(WARMUP_TICKS):
            frame = world.tick()
            self.alive()
            self.cameras.drain(frame)
        director.start()
        if director.kinematic:
            self.exact_start(director, ego, ego_wp)
        frame = world.tick()
        self.alive()
        director.sync(frame)
        dp = director.ego_path
        lane_keys = set()
        for x, y in dp.pts[::2]:
            wp = self.map.get_waypoint(carla.Location(x=float(x), y=float(y), z=dp.z))
            if wp is not None:
                lane_keys.add((int(wp.road_id), int(wp.lane_id)))
        route = DF.Route(dp.pts, dp.w, lane_keys)
        o_s = float(o.get('max_episode_s') or (60.0 if free else float(conc.get('duration_s', 12.0))))
        bb = ego.bounding_box.extent
        self.ep = {
            'n': self.n_episodes, 'seed': seed, 'options': opts, 'resolved': o, 'scenario': conc, 'free': free,
            'director': director, 'ego': ego, 'cam': cam, 'route': route, 'k': 0, 'done': False,
            'feat': DF.Featuriser(route, target, self.dt),
            'monitor': DF.InfractionMonitor(self.dt, self.cfg.get('thresholds'),
                                            self.cfg.get('terminate_on') or DF.DEFAULT_TERMINATE_ON),
            'max_steps': max(int(round(o_s * self.fps)), 1), 'max_episode_s': o_s, 'target': target,
            'len': 2.0 * bb.x, 'wid': 2.0 * bb.y, 'prev_action': [0.0, 0.0], 'ret': 0.0,
            'terms': {k: 0.0 for k in DF.REWARD_TERMS}, 'actions': [], 's_max': 0.0, 'travel': 0.0,
            'route_lights': self.route_lights(route), 'n_events': 0, 'wall0': time.monotonic(),
            'start': {k: round(v, 4) for k, v in zip(('x', 'y', 'yaw'), (ego_wp.transform.location.x,
                                                                       ego_wp.transform.location.y,
                                                                       ego_wp.transform.rotation.yaw))},
            'stale_actors': stale, 'trace': [],
        }
        self.n_episodes += 1
        state, facts, arrays, rows = self.observe(frame)
        self.ep['s_prev'] = facts['s']
        self.ep['front_prev'] = self.front()
        info = {'episode': self.ep['n'], 'seed': seed, 'step': 0, 'sim_t': 0.0,
                'scenario': None if free else {'id': conc.get('id'), 'name': conc.get('name'),
                                               'values': conc.get('values')},
                'motion': motion, 'route_length_m': round(route.length, 3), 'start': self.ep['start'],
                'target_speed_mps': round(target, 4), 'max_steps': self.ep['max_steps'],
                'stale_actors': stale, 'background_actors': self.background_actors,
                'ego': self.ego_pose(), 'traffic_lights_on_route': len(self.ep['route_lights'])}
        info, arrays = self.reset_extras(info, arrays)
        self.send({'t': 'obs', 'id': cmd.get('id'), 'kind': 'reset', 'reward': 0.0, 'terminated': False,
                   'truncated': False, 'info': info}, [('state', state)] + arrays)
        self.emit.progress(0.0, f'episode {self.ep["n"]} (seed {seed})')

    def reset_extras(self, info: dict, arrays: list):
        """A subclass's additions to the first observation of an episode (workers/adversary.py); none here."""
        return info, arrays

    def summary_extras(self, summ: dict):
        """A subclass's additions to an episode's summary line before it is written; none here."""
        return summ

    def free_start(self, order, route_m, lane_path):
        """The first spawn point in seeded order outside junctions whose lane runs ≥ min(route_m, 30) m."""
        need = min(route_m, 30.0)
        for k in list(order):
            wp = self.map.get_waypoint(self.spawns[k].location)
            order.remove(k)
            if wp is None or wp.is_junction:
                continue
            path = lane_path(wp, route_m)
            if path.s[-1] >= need:
                return wp, path
        raise Failed(f'no spawn point of this map starts a lane of {need:.0f} m')

    def exact_start(self, director, ego, ego_wp):
        """Kinematic motion: every vehicle exactly at its planned start pose (not where physics let it settle)."""
        carla = self.carla
        t = ego_wp.transform
        k = director.kin.get(ego.id)          # an ego on the traffic manager keeps CARLA's physics (not in kin)
        moves = []
        if k is not None:
            k[0], k[1], k[3] = t.location.x, t.location.y, t.rotation.yaw
            moves.append((ego, carla.Transform(carla.Location(x=k[0], y=k[1], z=k[2]), carla.Rotation(yaw=k[3]))))
        for st in director.actors.values():
            a = st['actor']
            if a.id not in director.kin:
                continue
            x, y, yaw = st['path'].at(0.0, st['lateral'])
            k = director.kin[a.id]
            k[0], k[1], k[3] = x, y, yaw
            moves.append((a, carla.Transform(carla.Location(x=x, y=y, z=k[2]), carla.Rotation(yaw=yaw))))
        place_now(self.client, carla, moves)  # before the tick (set_transform is asynchronous)

    def route_lights(self, route) -> list:
        """[(light index, stop line arc length on the route)]: the stop lines on the route's lane, its direction."""
        out = []
        for i, (_tl, lines) in enumerate(self.lights):
            for x, y, yaw, half in lines:
                s, lat, h = route.project(x, y)
                if abs(lat) < route.width_at(s) / 2.0 + 0.5 and abs(float(DF.wrap_deg(yaw - h))) < 30.0 and \
                        0.0 < s < route.length:
                    out.append((i, s))
        return out

    # ------------------------------------------------------------------ the state
    def ego_pose(self) -> dict:
        ep = self.ep
        d = ep['director']
        tr = d.tf(ep['ego'])
        return {'x': tr.location.x, 'y': tr.location.y, 'z': tr.location.z, 'yaw': tr.rotation.yaw,
                'speed_mps': self.ego_speed()}

    def ego_speed(self) -> float:
        d, ego = self.ep['director'], self.ep['ego']
        k = d.kin.get(ego.id)
        if k is not None:
            return float(k[4])
        v = d.vel(ego)
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def front(self):
        p = self.ego_pose()
        r = math.radians(p['yaw'])
        h = self.ep['len'] / 2.0
        return (p['x'] + math.cos(r) * h, p['y'] + math.sin(r) * h)

    def observe(self, frame):
        """(state vector, facts, extra arrays, actor rows) after the tick of `frame`."""
        np, ep = self.np, self.ep
        d, ego = ep['director'], ep['ego']
        pose = self.ego_pose()
        rows = self.actor_log.rows(self.world, d)
        for r in rows:                     # kinematic actors: the director's exact pose, not CARLA's snapshot of it
            k = d.kin.get(r['id'])         # (measured on 0.9.16: the snapshot of a set_transform pose can differ
            if k is not None:              # between runs by enough to move a BEV pixel edge)
                r.update(x=k[0], y=k[1], z=k[2], yaw=k[3], roll=0.0, pitch=0.0)
        lights = [(s, str(self.lights[i][0].get_state())) for i, s in ep['route_lights']]
        e = dict(pose, length=ep['len'], width=ep['wid'])
        state, facts = ep['feat'](e, rows, lights, ep['prev_action'], ego.id)
        arrays = []
        if self.bev is not None:
            tr = d.tf(ego)
            ego_pose = {'x': tr.location.x, 'y': tr.location.y, 'z': tr.location.z, 'roll': tr.rotation.roll,
                        'pitch': tr.rotation.pitch, 'yaw': tr.rotation.yaw}
            arrays.append(('bev', DF.bev_bits(ego_pose, rows, self.bev, self.lane_map, ego.id)))
        if ep['cam'] is not None:
            img = self.cameras.get(ep['cam'].id, frame)
            bgra = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)
            arrays.append(('camera', np.ascontiguousarray(bgra[:, :, 2::-1])))
        return state, facts, arrays, rows

    def lane_facts(self, pose) -> dict:
        """Where the ego's centre is on the map: off road (and how far), wrong way, outside the route's lanes."""
        carla = self.carla
        loc = carla.Location(x=pose['x'], y=pose['y'], z=pose['z'])
        wp = self.map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
        if wp is None:
            near = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            off = 0.0
            if near is not None:
                t = near.transform.location
                off = max(math.hypot(t.x - pose['x'], t.y - pose['y']) - near.lane_width / 2.0, 0.0)
            return {'off_road': True, 'offroad_m': off, 'wrong_way': False, 'outside_lanes': False}
        junction = bool(wp.is_junction)
        wrong = not junction and abs(float(DF.wrap_deg(wp.transform.rotation.yaw - pose['yaw']))) > DF.WRONG_WAY_DEG
        keys = self.ep['route'].lane_keys
        outside = not junction and bool(keys) and (int(wp.road_id), int(wp.lane_id)) not in keys
        return {'off_road': False, 'offroad_m': 0.0, 'wrong_way': wrong, 'outside_lanes': outside}

    def trace_row(self, sim_t, pose, state, lf) -> list:
        """One row of the episode's trace (core.rules.TRACE_COLUMNS): the ego, its lane on CARLA's map (heading,
        lateral offset + right, width, junction, same-direction lanes beside it by Waypoint.get_left_lane /
        get_right_lane, its markings), the lead and vulnerable-road-user gaps and the light ahead from the observation,
        and CARLA's speed limit for the ego (Vehicle.get_speed_limit; None where the API has none)."""
        FI = DF.FI
        r = lambda v, nd=3: None if v is None or not math.isfinite(float(v)) else round(float(v), nd)   # noqa: E731
        lane = self.lane_row(pose)
        lead = r(state[FI['lead_gap_m']]) if state[FI['lead_present']] > 0.5 else None
        vru = r(state[FI['vru_ahead_m']]) if state[FI['vru_present']] > 0.5 else None
        known = state[FI['tl_known']] > 0.5
        tl_red = (1 if state[FI['tl_red']] > 0.5 else 0) if known else None
        tl_d = r(state[FI['tl_distance_m']]) if known and state[FI['tl_distance_m']] < DF.TL_RANGE_M else None
        if known and tl_d is None:
            tl_red = 0
        return [r(sim_t, 4), r(pose['x']), r(pose['y']), r(pose['yaw']), r(pose['speed_mps'])] + lane + \
            [lead, vru, tl_red, tl_d, self.speed_limit(self.ep['ego'])]

    def speed_limit(self, actor):
        """CARLA's speed limit for a vehicle (Vehicle.get_speed_limit, km/h), None where the API has none."""
        get = getattr(actor, 'get_speed_limit', None)
        if get is None:
            return None
        try:
            v = float(get())
        except Exception:            # noqa: BLE001 (an API without it: unknown)
            return None
        return round(v, 2) if math.isfinite(v) else None

    def lane_row(self, pose) -> list:
        """The lane part of a trace row (core.rules.TRACE_COLUMNS on_lane … right_marking) for a pose {'x', 'y',
        'z', 'yaw'}: on a driving lane, junction, the lane's heading, the lateral offset (+ right), its width, the
        same-direction lanes beside it, its markings."""
        carla = self.carla
        r = lambda v, nd=3: None if v is None or not math.isfinite(float(v)) else round(float(v), nd)   # noqa: E731
        wp = self.map.get_waypoint(carla.Location(x=pose['x'], y=pose['y'], z=pose.get('z', 0.0)),
                                   project_to_road=False, lane_type=carla.LaneType.Driving)
        lane = [0, None, None, None, None, None, None, None, None]
        if wp is not None:
            t = wp.transform
            ry = math.radians(t.rotation.yaw)
            lat = (pose['x'] - t.location.x) * -math.sin(ry) + (pose['y'] - t.location.y) * math.cos(ry)

            def count(step):
                n, w, seen = 0, wp, 0
                while seen < 8:
                    w = step(w)
                    seen += 1
                    if w is None or str(getattr(w, 'lane_type', 'Driving')) != 'Driving' or \
                            (w.lane_id > 0) != (wp.lane_id > 0):
                        break
                    n += 1
                return n
            mk = lambda m: None if m is None else str(getattr(m.type, 'name', m.type))     # noqa: E731
            lane = [1, int(bool(wp.is_junction)), r(t.rotation.yaw), r(lat), r(wp.lane_width),
                    count(lambda w: w.get_left_lane()), count(lambda w: w.get_right_lane()),
                    mk(getattr(wp, 'left_lane_marking', None)), mk(getattr(wp, 'right_lane_marking', None))]
        return lane

    def contacts(self, pose, rows) -> list:
        """[(other id, kind, source)]: box overlaps with vehicles and walkers, and the collision sensor's new
        events (whatever it hit)."""
        from ..core import criticality as C
        ep = self.ep
        ego_id = ep['ego'].id
        A = C.box_corners(pose['x'], pose['y'], pose['yaw'], ep['len'], ep['wid'])
        out, seen = [], set()
        kinds = {}
        for r in rows:
            kinds[r['id']] = 'pedestrian' if r['type_id'].startswith('walker.') else 'vehicle'
            if r['id'] == ego_id or math.hypot(r['x'] - pose['x'], r['y'] - pose['y']) > CONTACT_RANGE_M:
                continue
            B = C.box_corners(r['x'], r['y'], r['yaw'], 2.0 * r['extent'][0], 2.0 * r['extent'][1])
            if C.box_distance(A, B) <= 0.0:
                out.append((r['id'], kinds[r['id']], 'box'))
                seen.add(r['id'])
        evs = ep['director'].events
        for _t, oid, _imp in evs[ep['n_events']:]:
            if oid not in seen and oid != ego_id:
                out.append((oid, kinds.get(oid, 'static'), 'sensor'))
                seen.add(oid)
        ep['n_events'] = len(evs)
        return out

    def red_lights(self, p0, p1) -> list:
        out = []
        for tl, lines in self.lights:
            for x, y, yaw, half in lines:
                if math.hypot(x - p1[0], y - p1[1]) > 20.0:
                    continue
                if DF.crossed_line(p0, p1, (x, y), yaw, half) and str(tl.get_state()) == 'Red':
                    out.append(tl.id)
                    break
        return out

    # ------------------------------------------------------------------ a step
    def step(self, cmd):
        ep = self.ep
        if ep is None:
            raise BadCommand('no episode: reset first')
        if ep['done']:
            raise BadCommand('the episode has ended: reset first')
        try:
            u, applied = DF.to_control(cmd.get('action'), self.action_kind)
        except ValueError as e:
            raise BadCommand(str(e)) from None
        t0 = time.monotonic()
        world, carla = self.world, self.carla
        d, ego = ep['director'], ep['ego']
        d.step(ep['k'] * self.dt)
        if d.kinematic:
            d.kin_step(ego, u)
        else:
            ego.apply_control(carla.VehicleControl(throttle=u['throttle'], steer=u['steer'], brake=u['brake']))
        frame = world.tick()
        self.alive()
        d.sync(frame)
        ep['k'] += 1
        k = ep['k']
        sim_t = k * self.dt
        d.record(sim_t)
        steer_rate = (applied[0] - ep['prev_action'][0]) / self.dt
        ep['prev_action'] = applied
        state, facts, arrays, rows = self.observe(frame)
        pose = self.ego_pose()
        lf = self.lane_facts(pose)
        front = self.front()
        travel = facts['speed'] * self.dt
        ep['travel'] += travel
        ds = max(min(facts['s'] - ep['s_prev'], 5.0), -5.0)
        ep['s_prev'] = facts['s']
        ep['s_max'] = max(ep['s_max'], facts['s'])
        inside = {sid: True for sid, cx, cy, yaw, ex, ey in self.stops if DF.in_box(pose['x'], pose['y'], cx, cy,
                                                                                    yaw, ex, ey)}
        mf = dict(lf, contacts=self.contacts(pose, rows), red_light=self.red_lights(ep['front_prev'], front),
                  stop_inside=inside, lateral=facts['lateral'], speed=facts['speed'], s=facts['s'],
                  route_length=facts['route_length'], travel_m=travel)
        ep['front_prev'] = front
        new, reason = ep['monitor'].update(k, sim_t, pose, mf)
        q = dict(lf, ds=ds, speed=facts['speed'], target=ep['target'], lateral=facts['lateral'],
                 lane_width=facts['lane_width'], heading_err=facts['heading_err'], jerk=facts['jerk'],
                 steer_rate=steer_rate)
        terms = DF.reward_terms(q, new, self.cfg.get('reward'), self.dt)
        reward = float(sum(terms[t] for t in DF.REWARD_TERMS))
        ep['ret'] += reward
        for t_ in DF.REWARD_TERMS:
            ep['terms'][t_] += terms[t_]
        if self.cfg.get('log_actions', True):
            ep['actions'].append(applied if self.action_kind == 'continuous' else
                                 int(self.np.asarray(cmd.get('action')).reshape(-1)[0]))
        if self.cfg.get('log_trace', True):
            ep['trace'].append(self.trace_row(sim_t, pose, state, lf))
        terminated = reason is not None
        truncated = not terminated and k >= ep['max_steps']
        info = {'episode': ep['n'], 'step': k, 'sim_t': round(sim_t, 4), 'reward_terms': terms,
                'infractions': new, 'progress_m': round(facts['s'], 4),
                'route_completion': round(min(max(facts['s'] / max(facts['route_length'], 1e-9), 0.0), 1.0), 4),
                'ego': {kk: round(float(v), 4) for kk, v in pose.items()}, 'action_applied': applied,
                'off_road': lf['off_road'], 'wrong_way': lf['wrong_way'], 'outside_route_lanes': lf['outside_lanes'],
                'termination': reason if terminated else ('time_limit' if truncated else None)}
        if getattr(self, 'watch', None) is not None:
            info['odd'] = self.watch.step(sim_t, ego, facts['speed'])
        self.n_steps += 1
        self.step_wall += time.monotonic() - t0
        if terminated or truncated:
            info['episode_summary'] = self.end_episode(info['termination'], terminated, truncated)
        self.send({'t': 'obs', 'id': cmd.get('id'), 'kind': 'step', 'reward': reward, 'terminated': terminated,
                   'truncated': truncated, 'info': info}, [('state', state)] + arrays)

    def end_episode(self, reason, terminated=False, truncated=False) -> dict:
        ep = self.ep
        ep['done'] = True
        k = ep['k']
        ep['monitor'].finish(k, k * self.dt)
        conc = ep['scenario']
        route = ep['route']
        summ = {'episode': ep['n'], 'seed': ep['seed'], 'options': ep['options'],
                'scenario': None if ep['free'] else {'id': conc.get('id'), 'name': conc.get('name'),
                                                     'values': conc.get('values'), 'seed': conc.get('seed')},
                'motion': ep['resolved'].get('motion', 'kinematic'), 'start': ep['start'],
                'route': route.doc(), 'route_length_m': round(route.length, 3),
                'progress_m': round(ep['s_max'], 3),
                'route_completion': round(min(ep['s_max'] / max(route.length, 1e-9), 1.0), 4),
                'distance_m': round(ep['travel'], 3), 'steps': k, 'sim_s': round(k * self.dt, 4),
                'return': round(ep['ret'], 6), 'reward_terms': {t: round(v, 6) for t, v in ep['terms'].items()},
                'termination': reason, 'terminated': terminated, 'truncated': truncated,
                'target_speed_mps': round(ep['target'], 4), 'infractions': ep['monitor'].events,
                'infraction_counts': ep['monitor'].counts(), 'wall_s': round(time.monotonic() - ep['wall0'], 3),
                'stale_actors': ep['stale_actors']}
        summ['success'] = DF.episode_success(summ)
        if getattr(self, 'watch', None) is not None:
            summ['odd_monitor'] = self.watch.end_episode()
        if self.cfg.get('log_actions', True):
            summ['actions'] = ep['actions']
        if self.cfg.get('log_trace', True):
            summ['trace'] = {'columns': list(TRACE_COLUMNS), 'rows': ep['trace'],
                             'ego': {'length': round(ep['len'], 3), 'width': round(ep['wid'], 3)},
                             'handedness': self.cfg.get('handedness'), 'town': self.cfg.get('town') or None}
        summ = self.summary_extras(summ)
        self.ep_log.write(json.dumps(summ, separators=(',', ':')) + '\n')
        self.ep_log.flush()
        return {k2: v for k2, v in summ.items() if k2 not in ('actions', 'trace')}

    # ------------------------------------------------------------------ tidy-up
    def tidy(self):
        try:
            if self.sock is not None:
                try:
                    self.sock.sendall(pack({'t': 'bye', 'reason': self.ended_by or 'ended'}))
                except OSError:
                    pass
        finally:
            self.cameras.stop()
            super().tidy()
            if self.world is not None and self.rendering_changed is not None:
                try:
                    st = self.world.get_settings()
                    st.no_rendering_mode = self.rendering_changed
                    self.world.apply_settings(st)
                except RuntimeError as e:
                    self.note(f'could not restore rendering: {e}')
            if self.world is not None:
                try:
                    if hasattr(self.world, 'wait_for_tick'):
                        self.world.wait_for_tick(5.0)
                    left = len(self.our_actor_ids())
                    self.run['tidy_check'] = {'falcon_actors_left': left}
                    if left:
                        self.note(f'{left} of this worker\'s actors are still in the world after tidying')
                except RuntimeError:
                    pass
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker gym')
    ap.add_argument('config', help='gym config JSON (falcon_autonomy.core.gymenv.gym_config)')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return Gym(cfg, argv, a.config).execute()
