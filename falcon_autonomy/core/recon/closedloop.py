"""A closed-loop neural simulator: a policy drives an ego through a reconstruction of a recorded drive, seeing the
reconstruction rendered from wherever its own decisions took it (torch, numpy).

    sim = NeuralSim('…/recon/drive', out_dir='…/runs/neural-drive', deviate_m=3.0)
    rollout(load_policy('…/policies/bc'), sim, episodes=3)       # core.policy.rollout, unchanged

NeuralSim has FalconDriveEnv's surface (core.gymenv): reset(seed=, options=) -> (obs, info), step(action) -> (obs,
reward, terminated, truncated, info), feature_names, observation_space and action_space, run_dir, ready, close() —
so core.policy.rollout and every policy that drives the CARLA environment drive here unchanged.

* The ego: the kinematic bicycle model of the CARLA environment's kinematic motion (a numpy port of
  workers/director.py kin_step: v += (3·throttle − 8·brake)·dt, yaw rate v·tan(steer·70°) / wheelbase, the wheelbase
  1.2 × the ego's half length, the position advanced along the mean heading), in CARLA's convention of the
  reconstruction's world (x, y, yaw positive to the right), starting at the logged ego pose and speed of the start
  time (option start_s). Height, roll and pitch follow the logged path at the nearest point (the reconstruction has
  no road surface model). Actions are the environment's: [steer, accel] in [−1, 1]² or a Discrete(25) index
  (core.drivefeatures.to_control).
* The other road users: an ActorController — LoggedActors (the default) replays every track's logged motion; a
  reactive controller (level-k traffic, wave 2) implements the same three methods and sees the ego's state. A
  controller says where each track's box is at t; the reconstruction draws that track's Gaussians there (its
  appearance is the logged one; an actor driven off its logged path keeps its own look but meets lighting and
  backgrounds it was never seen against).
* Every step renders the reconstruction's cameras (the trained cameras' mounts, at obs_size) at the ego's pose and
  the step's time; the observation is FalconDriveEnv's: 'state' (core.drivefeatures.Featuriser on the logged ego
  path as the route — no map: lane width 3.5 m assumed, no traffic lights: tl_known 0) and, with camera (the
  default), 'camera' (H, W, 3) uint8 from the first camera.
* Infractions and termination: core.drivefeatures.InfractionMonitor with box contacts against every actor (core.
  criticality's oriented boxes), route deviation, blocking and route completion; no off-road or lane facts (no map).
  Reward: core.drivefeatures.reward_terms.
* **The trust envelope**: how far from the logged path the reconstruction still renders faithfully, from the
  reconstruction's own measurement (evaluate.lateral_shift: the held-out frames shifted sideways against
  depth-warped CARLA references — `falcon recon eval --warp-shift`): trusted_m is the largest measured shift at
  which the PSNR is within TRUST_DROP_DB of the unshifted control on both sides. Every step records the ego's lateral
  deviation from the logged path, the PSNR expected there (interpolated from the measurement) and whether it is
  within the envelope; beyond deviate_m (3 m by default) a step is flagged and the episode ends, truncated, with
  termination 'outside_reconstruction': the images there are the reconstruction's guesses. Without the measurement
  the envelope is 'not measured' and only deviate_m applies.

Each episode is written as a normal Falcon run folder (ep-<n>/: frames.jsonl, actors.jsonl, rgb/<camera>/,
drive.json, dataset.json, run.json, closedloop.json), so the traffic rules (core.rules.check_run), STL, criticality
and the exporters read it like a CARLA run; the run folder holds episodes.jsonl (FalconDriveEnv's), closedloop.json
(the envelope and every episode's deviations) and rollout.json when core.policy.rollout writes one.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .. import drivefeatures as DF

SCHEMA = 'falcon.neural_sim/1'
TRUST_DROP_DB = 3.0
DEVIATE_M = 3.0
LANE_WIDTH_M = 3.5
ACCEL, BRAKE, STEER_DEG = 3.0, 8.0, 70.0          # workers/director.py's kinematic model
EGO_SIZE = (4.9, 2.1)                              # (length, width) when the drive does not say (CARLA's mkz_2020)


# =========================================================================== the ego
def kin_step(state: list, u: dict, dt: float, wheelbase: float) -> list:
    """One step of the kinematic bicycle model (workers/director.py kin_step without CARLA): state [x, y, yaw (deg,
    CARLA's sense), v]; u {'throttle', 'brake', 'steer', 'hand_brake'}."""
    x, y, yaw, v = state
    if u.get('hand_brake'):
        v = 0.0
    else:
        v = max(v + (ACCEL * u.get('throttle', 0.0) - BRAKE * u.get('brake', 0.0)) * dt, 0.0)
    yaw2 = yaw + math.degrees(v * math.tan(u.get('steer', 0.0) * math.radians(STEER_DEG)) / wheelbase * dt)
    mid = math.radians((yaw + yaw2) / 2.0)
    return [x + math.cos(mid) * v * dt, y + math.sin(mid) * v * dt, yaw2, v]


# =========================================================================== actors
class ActorController:
    """Where the other road users are (module doc). A controller is reset at each episode and stepped after the ego;
    poses(t) gives {track id: (R world ← box (3, 3), centre (3,), velocity (3,))} in the reconstruction's right-handed
    world for every actor present at t. sim: the NeuralSim (its ego state, tracks and time)."""

    def reset(self, sim, seed: int):
        pass

    def step(self, sim, t: float, dt: float):
        pass

    def poses(self, t: float) -> dict:
        raise NotImplementedError


class LoggedActors(ActorController):
    """Every track replays its logged motion (core.recon.actors.Track.pose)."""

    def __init__(self, tracks: dict):
        self.tracks = tracks

    def poses(self, t: float) -> dict:
        out = {}
        for k, tr in self.tracks.items():
            po = tr.pose(t)
            if po is not None:
                out[k] = (po[0], po[1], tr.velocity(t))
        return out


# =========================================================================== the trust envelope
def trust_envelope(evaluation: dict | None, drop_db: float = TRUST_DROP_DB) -> dict:
    """The envelope (module doc) from a reconstruction's evaluation.json: {'trusted_m' (None when not measured),
    'curve': [[|shift| m, PSNR dB (the worse side)], …], 'control_psnr', 'source'}."""
    shifts = (((evaluation or {}).get('novel_view') or {}).get('lateral_warped') or {}).get('shifts') or []
    by = {}
    for s in shifts:
        if s.get('psnr') is None:
            continue
        k = round(abs(float(s['lateral_m'])), 4)
        by[k] = min(by.get(k, math.inf), float(s['psnr']))
    if 0.0 not in by:
        return {'trusted_m': None, 'curve': [], 'control_psnr': None, 'drop_db': drop_db,
                'source': 'not measured: run `falcon recon eval RECON --warp-shift 0.5 --warp-shift 1 …` on a '
                          'reconstruction of a Falcon run'}
    curve = sorted(by.items())
    p0 = by[0.0]
    trusted = 0.0
    for d, p in curve:
        if p >= p0 - drop_db:
            trusted = d
        else:
            break
    return {'trusted_m': trusted, 'curve': [[d, round(p, 3)] for d, p in curve], 'control_psnr': round(p0, 3),
            'drop_db': drop_db, 'source': 'evaluate.lateral_shift: held-out frames shifted sideways against depth-'
                                          'warped CARLA references'}


def expected_psnr(env: dict, dev_m: float):
    """The PSNR the envelope's curve gives at a deviation (linear between measured shifts; None beyond them)."""
    c = env.get('curve') or []
    if not c:
        return None
    d = abs(float(dev_m))
    xs = [x for x, _ in c]
    if d > xs[-1]:
        return None
    return round(float(np.interp(d, xs, [p for _, p in c])), 3)


# =========================================================================== policies
class ConstantSpeedPolicy:
    """A built-in policy for checks: holds a speed (m/s; default the observation's target) and follows the route by
    steering against the lateral offset and heading error (gains heuristic). act(obs, names) like core.policy's."""

    def __init__(self, speed_mps: float | None = None, k_lat: float = 0.3, k_head: float = 1.2):
        self.speed, self.k_lat, self.k_head = speed_mps, k_lat, k_head
        self.path = None

    def act(self, obs, names=DF.FEATURE_NAMES):
        f = np.asarray(obs['state'], float)
        ix = {n: i for i, n in enumerate(names)}
        v = f[ix['speed_mps']]
        target = self.speed if self.speed is not None else f[ix['target_speed_mps']]
        steer = -self.k_lat * f[ix['lateral_offset_m']] - self.k_head * f[ix['heading_error_rad']]
        accel = 0.5 * (target - v)
        return np.clip([steer, accel], -1.0, 1.0).astype(np.float32)


def policy_of(spec: str, device: str = 'cpu'):
    """'constant' or 'constant:KMH' (ConstantSpeedPolicy), else a policy folder (core.policy.load_policy)."""
    if spec.startswith('constant'):
        _, _, kmh = spec.partition(':')
        return ConstantSpeedPolicy(float(kmh) / 3.6 if kmh else None)
    from ..policy import load_policy
    return load_policy(os.path.abspath(os.path.expanduser(spec)), device)


# =========================================================================== the simulator
class NeuralSim:
    """The closed-loop simulator (module doc). recon: a core.recon.scene.Recon or its folder; out_dir: the run folder;
    cameras: the reconstruction's camera ids to render (default every trained camera); obs_size: (w, h) of each
    rendered camera; camera: put the first camera's image in the observation; controller: an ActorController (default
    LoggedActors); fps: the step rate (default the source's frame rate); deviate_m: the deviation that ends an
    episode; max_episode_s: default the logged drive's length."""

    metadata = {'render_modes': []}
    render_mode = None

    def __init__(self, recon, out_dir: str, cameras=None, obs_size=(160, 90), camera: bool = True, controller=None,
                 fps: float | None = None, deviate_m: float = DEVIATE_M, target_speed_mps: float | None = None,
                 action: str = 'continuous', max_episode_s: float | None = None, device: str = 'auto',
                 write_images: bool = True, seed: int = 0):
        from ..gymenv import make_box, make_dict, make_discrete
        from .scene import Recon
        self.rec = recon if isinstance(recon, Recon) else Recon.load(recon, device)
        self.out = os.path.abspath(out_dir)
        os.makedirs(self.out, exist_ok=True)
        self.obs_size = (int(obs_size[0]), int(obs_size[1]))
        self.want_camera, self.write_images = bool(camera), bool(write_images)
        self.deviate_m = float(deviate_m)
        self.action_kind = action
        self._closed = False
        frames = sorted(self.rec.frames, key=lambda f: f['t'])
        cams = []
        for f in frames:
            c = f.get('camera') or self.rec.views['source'].get('camera') or 'camera'
            if c not in cams:
                cams.append(c)
        self.cameras = [c for c in (cameras or cams) if c in cams] or cams[:1]
        self.mounts = {}
        for f in frames:
            c = f.get('camera') or self.cameras[0]
            if c in self.cameras and c not in self.mounts:
                self.mounts[c] = (np.asarray(f['mount'], float).reshape(4, 4), f)
        # the logged ego path: one sample per frame time (the first camera's)
        first = [f for f in frames if (f.get('camera') or self.cameras[0]) == self.cameras[0]]
        self.log_t = np.array([f['t'] for f in first], float)
        self.log_ego = [np.asarray(f['ego'], float).reshape(4, 4) for f in first]
        from .scene import carla_pose_of
        self.log_pose = [carla_pose_of(E) for E in self.log_ego]
        xy = np.array([[p['x'], p['y']] for p in self.log_pose])
        dt_log = np.diff(self.log_t)
        sp = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T) / np.maximum(dt_log, 1e-6)] if len(xy) > 1 else np.zeros(1)
        if len(sp) > 1:
            sp[0] = sp[1]
        self.log_speed = sp
        self.route = DF.Route(xy, np.full(len(xy), LANE_WIDTH_M)) if len(xy) > 1 else None
        if self.route is None:
            raise ValueError('the reconstruction\'s drive has fewer than two distinct ego poses: no route to follow')
        self.fps = float(fps) if fps else (float((len(self.log_t) - 1) / max(self.log_t[-1] - self.log_t[0], 1e-6))
                                           if len(self.log_t) > 1 else 10.0)
        self.dt = 1.0 / self.fps
        self.target = float(target_speed_mps) if target_speed_mps is not None else \
            float(np.median(sp[sp > 0.5])) if np.any(sp > 0.5) else 5.0
        self.max_episode_s = float(max_episode_s) if max_episode_s else float(self.log_t[-1] - self.log_t[0])
        self.tracks = {}
        for a in self.rec.actors:
            self.tracks[a.track.id] = a.track
        self.controller = controller or LoggedActors(self.tracks)
        ev_p = os.path.join(self.rec.path, 'evaluation.json')
        ev = None
        if os.path.isfile(ev_p):
            with open(ev_p, encoding='utf-8') as fh:
                ev = json.load(fh)
        self.envelope = trust_envelope(ev)
        drive = {}
        src = self.rec.views.get('source') or {}
        if src.get('kind') == 'falcon' and os.path.isfile(os.path.join(src.get('path', ''), 'drive.json')):
            with open(os.path.join(src['path'], 'drive.json'), encoding='utf-8') as fh:
                drive = json.load(fh)
        self.src_drive = drive
        veh = (drive.get('rig') or {}).get('vehicle')
        from ..rig import VEHICLES
        L, W = (VEHICLES[veh][0], VEHICLES[veh][1]) if veh in VEHICLES else EGO_SIZE
        self.ego_len, self.ego_wid = float(L), float(W)
        self.wheelbase = max(2.0 * (L / 2.0) * 0.6, 0.8)
        self._rng = np.random.default_rng(int(seed))
        self._n_ep = 0
        self.ep = None
        self.episodes_path = os.path.join(self.out, 'episodes.jsonl')
        self.summary = {'schema': SCHEMA, 'reconstruction': self.rec.path, 'envelope': self.envelope,
                        'deviate_m': self.deviate_m, 'cameras': self.cameras, 'obs_size': list(self.obs_size),
                        'fps': self.fps, 'episodes': []}
        obs = {'state': {'shape': [DF.N_FEATURES], 'dtype': 'float32'}}
        sp_ = {'state': make_box(-np.inf, np.inf, (DF.N_FEATURES,), np.float32)}
        if self.want_camera:
            obs['camera'] = {'shape': [self.obs_size[1], self.obs_size[0], 3], 'dtype': 'uint8'}
            sp_['camera'] = make_box(0, 255, (self.obs_size[1], self.obs_size[0], 3), np.uint8)
        self.observation_space = make_dict(sp_)
        if action == 'discrete':
            self.action_space = make_discrete(len(DF.DISCRETE_ACTIONS))
            act = {'kind': 'discrete', 'n': len(DF.DISCRETE_ACTIONS), 'keys': list(DF.ACTION_KEYS)}
        else:
            self.action_space = make_box(np.array([-1, -1], np.float32), np.array([1, 1], np.float32), (2,),
                                         np.float32)
            act = {'kind': 'continuous', 'keys': list(DF.ACTION_KEYS), 'low': [-1.0, -1.0], 'high': [1.0, 1.0]}
        self.ready = {'t': 'ready', 'observation': obs, 'action': act, 'features': [list(f) for f in DF.FEATURES],
                      'drivefeatures': DF.SCHEMA, 'fps': self.fps,
                      'map': f'reconstruction:{os.path.basename(self.rec.path)}', 'simulator': 'neural',
                      'reward_weights': dict(DF.DEFAULT_REWARD), 'thresholds': dict(DF.DEFAULT_THRESHOLDS),
                      'terminate_on': list(DF.DEFAULT_TERMINATE_ON), 'envelope': self.envelope}

    # ------------------------------------------------------------------ gym surface
    @property
    def run_dir(self) -> str:
        return self.out

    @property
    def feature_names(self) -> tuple:
        return DF.FEATURE_NAMES

    @property
    def unwrapped(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def render(self):
        return None

    # ------------------------------------------------------------------ logged path
    def logged_at(self, t: float) -> dict:
        """The logged ego (CARLA convention) nearest in time to t, and its speed."""
        k = int(np.argmin(np.abs(self.log_t - t)))
        return dict(self.log_pose[k], speed_mps=float(self.log_speed[k]), index=k)

    def _ego_matrix(self, st, s_route: float) -> np.ndarray:
        """world ← vehicle (right-handed) of the ego state, z / roll / pitch from the logged pose nearest along the
        route."""
        from ..exporters import pose_matrix, rh
        xy = np.array([[p['x'], p['y']] for p in self.log_pose])
        k = int(np.argmin(np.hypot(xy[:, 0] - st[0], xy[:, 1] - st[1])))
        ref = self.log_pose[k]
        return rh(pose_matrix({'x': st[0], 'y': st[1], 'z': ref['z'], 'roll': ref['roll'], 'pitch': ref['pitch'],
                               'yaw': st[2]}))

    # ------------------------------------------------------------------ rendering
    def _render(self, ego: np.ndarray, t: float, poses: dict) -> dict:
        """{camera: (H, W, 3) uint8} at the ego pose and time; actors at the controller's poses."""
        import torch
        from . import data as D
        model = self.rec.scene_model()
        boxes = {k: (R, c) for k, (R, c, _v) in poses.items()}
        out = {}
        for c in self.cameras:
            mount, f = self.mounts[c]
            w, h = self.obs_size
            K = D.scaled_K(f['K'], f['size'], (w, h))
            vm = torch.as_tensor(D.inv(ego @ mount), dtype=torch.float32, device=self.rec.device)
            with torch.no_grad():
                r = model.render(vm, K, w, h, t, boxes=boxes)
            out[c] = (r['image'].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
        return out

    # ------------------------------------------------------------------ episodes
    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError('the simulator is closed')
        if self.ep is not None and not self.ep['done']:
            self._end('interrupted', False, True)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            ep_seed = int(seed)
        else:
            ep_seed = int(self._rng.integers(0, 2 ** 31 - 1))
        opts = dict(options or {})
        t0 = float(self.log_t[0]) + float(opts.get('start_s', 0.0))
        start = self.logged_at(t0)
        lat0 = float(opts.get('lateral_m', 0.0))
        yaw = math.radians(start['yaw'])
        x0 = start['x'] - math.sin(yaw) * lat0              # lateral_m to the right (CARLA: y right, yaw clockwise)
        y0 = start['y'] + math.cos(yaw) * lat0
        v0 = float(opts['start_speed_mps']) if 'start_speed_mps' in opts else start['speed_mps']
        n = self._n_ep
        self._n_ep += 1
        ep_dir = os.path.join(self.out, f'ep-{n:04d}')
        os.makedirs(ep_dir, exist_ok=True)
        feat = DF.Featuriser(self.route, float(opts.get('target_speed_mps', self.target)), self.dt)
        mon = DF.InfractionMonitor(self.dt, opts.get('thresholds'), opts.get('terminate_on',
                                                                             DF.DEFAULT_TERMINATE_ON))
        max_s = float(opts.get('max_episode_s', self.max_episode_s - float(opts.get('start_s', 0.0))))
        self.ep = {'n': n, 'seed': ep_seed, 'options': opts, 'dir': ep_dir, 't0': t0, 'k': 0,
                   'state': [x0, y0, start['yaw'], v0],
                   'feat': feat, 'monitor': mon, 'max_steps': max(1, int(round(max_s * self.fps))), 'ret': 0.0,
                   'terms': {t: 0.0 for t in DF.REWARD_TERMS}, 'prev_action': (0.0, 0.0), 's_prev': None,
                   's_max': 0.0, 'travel': 0.0, 'done': False, 'rows': [], 'actor_rows': [], 'dev': [],
                   'wall0': time.monotonic(), 'flagged': 0}
        for d in ('rgb',):
            for c in self.cameras:
                os.makedirs(os.path.join(ep_dir, d, c), exist_ok=True)
        self.controller.reset(self, ep_seed)
        obs, facts, _ = self._observe()
        self.ep['s_prev'] = facts['s']
        self.ep['s_max'] = facts['s']
        info = {'episode': n, 'seed': ep_seed, 'start': {k: start[k] for k in ('x', 'y', 'yaw')},
                'envelope': self.envelope}
        return obs, info

    def _observe(self):
        ep = self.ep
        k = ep['k']
        t = ep['t0'] + k * self.dt
        st = ep['state']
        poses = self.controller.poses(t)
        s_route = self.route.project(st[0], st[1], ep['s_prev'])[0]
        E = self._ego_matrix(st, s_route)
        imgs = self._render(E, t, poses)
        actors = self._actor_rows(poses)
        ego = {'x': st[0], 'y': st[1], 'yaw': st[2], 'speed_mps': st[3], 'length': self.ego_len,
               'width': self.ego_wid}
        vec, facts = ep['feat'](ego, actors, None, ep['prev_action'])
        obs = {'state': vec}
        if self.want_camera:
            obs['camera'] = imgs[self.cameras[0]]
        self._record(t, E, imgs, actors, facts)
        return obs, facts, actors

    def _actor_rows(self, poses: dict) -> list:
        """actors.jsonl-style rows (CARLA's convention) of the actors at their poses."""
        from .scene import carla_pose_of
        rows = []
        for k, (R, c, v) in poses.items():
            tr = self.tracks.get(k)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, c
            p = carla_pose_of(T)
            L, W, H = tr.size if tr is not None else (4.5, 2.0, 1.5)
            walker = tr is not None and tr.kind == 'pedestrian'
            cat = (tr.category if tr is not None and tr.category and '.' in tr.category and
                   not tr.category.startswith(('vehicle.', 'walker.')) else
                   ('human.pedestrian.adult' if walker else 'vehicle.car'))
            rows.append({'id': int(k), 'x': p['x'], 'y': p['y'], 'z': p['z'], 'roll': p['roll'], 'pitch': p['pitch'],
                         'yaw': p['yaw'], 'vx': round(float(v[0]), 4), 'vy': round(float(-v[1]), 4),
                         'vz': round(float(v[2]), 4), 'type_id': ('walker.pedestrian.recon' if walker else
                                                                  'vehicle.recon.track'),
                         'category': cat, 'role': '', 'extent': [round(L / 2, 4), round(W / 2, 4), round(H / 2, 4)],
                         'bbox_center': [0.0, 0.0, 0.0]})
        return rows

    def _record(self, t, E, imgs, actors, facts):
        import cv2
        from .scene import carla_pose_of
        ep = self.ep
        k = ep['k']
        files = {}
        if self.write_images:
            for c, img in imgs.items():
                rel = f'rgb/{c}/{k:06d}.jpg'
                cv2.imwrite(os.path.join(ep['dir'], rel), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                files[f'{c}/rgb'] = rel
        pose = carla_pose_of(E)
        st = ep['state']
        dev = float(facts['lateral'])
        within = None if self.envelope['trusted_m'] is None else abs(dev) <= self.envelope['trusted_m'] + 1e-9
        d = {'step': k, 'sim_t': round(t - ep['t0'], 6), 'lateral_m': round(dev, 4),
             'expected_psnr': expected_psnr(self.envelope, dev), 'within_envelope': within,
             'beyond_limit': abs(dev) > self.deviate_m}
        ep['dev'].append(d)
        if d['beyond_limit']:
            ep['flagged'] += 1
        ep['rows'].append({'step': k, 'frame': k, 'sim_t': round(t - ep['t0'], 6), 'files': files, 'ego': pose,
                           'speed_mps': round(float(st[3]), 4), 'recorded': True,
                           'control': None if k == 0 else {'steer': round(float(ep['prev_action'][0]), 4),
                                                            'accel': round(float(ep['prev_action'][1]), 4)},
                           'deviation': d, 'recon_t': round(t, 6)})
        hero = {'id': 0, 'x': pose['x'], 'y': pose['y'], 'z': pose['z'], 'roll': pose['roll'], 'pitch': pose['pitch'],
                'yaw': pose['yaw'], 'vx': round(st[3] * math.cos(math.radians(st[2])), 4),
                'vy': round(st[3] * math.sin(math.radians(st[2])), 4), 'vz': 0.0,
                'type_id': (self.src_drive.get('rig') or {}).get('vehicle') or 'vehicle.lincoln.mkz_2020',
                'category': 'vehicle.car', 'role': 'hero',
                'extent': [self.ego_len / 2, self.ego_wid / 2, 0.75], 'bbox_center': [0.0, 0.0, 0.75]}
        ep['actor_rows'].append({'step': k, 'sim_t': round(t - ep['t0'], 6), 'actors': [hero] + actors})

    def step(self, action):
        from .. import criticality as C
        ep = self.ep
        if ep is None or ep['done']:
            raise RuntimeError('step() needs reset() first: the episode ended or never started')
        u, applied = DF.to_control(action, self.action_kind)
        st0 = list(ep['state'])
        ep['state'] = kin_step(ep['state'], u, self.dt, self.wheelbase)
        steer_rate = (applied[0] - ep['prev_action'][0]) / self.dt
        ep['prev_action'] = applied
        ep['k'] += 1
        k = ep['k']
        t = ep['t0'] + k * self.dt
        self.controller.step(self, t, self.dt)
        obs, facts, actors = self._observe()
        st = ep['state']
        travel = math.hypot(st[0] - st0[0], st[1] - st0[1])
        ep['travel'] += travel
        ds = max(min(facts['s'] - ep['s_prev'], 5.0), -5.0)
        ep['s_prev'] = facts['s']
        ep['s_max'] = max(ep['s_max'], facts['s'])
        A = C.box_corners(st[0], st[1], st[2], self.ego_len, self.ego_wid)
        contacts = []
        for a in actors:
            if math.hypot(a['x'] - st[0], a['y'] - st[1]) > 15.0:
                continue
            B = C.box_corners(a['x'], a['y'], a['yaw'], 2.0 * a['extent'][0], 2.0 * a['extent'][1])
            if C.box_distance(A, B) <= 0.0:
                contacts.append((a['id'], 'pedestrian' if a['type_id'].startswith('walker.') else 'vehicle', 'box'))
        pose = {'x': st[0], 'y': st[1], 'z': 0.0, 'yaw': st[2], 'speed_mps': st[3]}
        mf = {'contacts': contacts, 'lateral': facts['lateral'], 'speed': facts['speed'], 's': facts['s'],
              'route_length': facts['route_length'], 'travel_m': travel, 'off_road': False, 'wrong_way': False,
              'outside_lanes': False}
        new, reason = ep['monitor'].update(k, t - ep['t0'], pose, mf)
        q = {'ds': ds, 'speed': facts['speed'], 'target': ep['feat'].target, 'lateral': facts['lateral'],
             'lane_width': facts['lane_width'], 'heading_err': facts['heading_err'], 'jerk': facts['jerk'],
             'steer_rate': steer_rate, 'off_road': False, 'wrong_way': False, 'outside_lanes': False}
        terms = DF.reward_terms(q, new, None, self.dt)
        reward = float(sum(terms[x] for x in DF.REWARD_TERMS))
        ep['ret'] += reward
        for x in DF.REWARD_TERMS:
            ep['terms'][x] += terms[x]
        terminated = reason is not None
        beyond = abs(facts['lateral']) > self.deviate_m
        truncated = not terminated and (k >= ep['max_steps'] or beyond)
        termination = reason if terminated else ('outside_reconstruction' if beyond else
                                                 ('time_limit' if truncated else None))
        info = {'episode': ep['n'], 'step': k, 'sim_t': round(t - ep['t0'], 4), 'reward_terms': terms,
                'infractions': new, 'progress_m': round(facts['s'], 4),
                'route_completion': round(min(max(facts['s'] / max(facts['route_length'], 1e-9), 0.0), 1.0), 4),
                'ego': {kk: round(float(v), 4) for kk, v in pose.items()}, 'action_applied': applied,
                'deviation': ep['dev'][-1], 'termination': termination}
        if terminated or truncated:
            info['episode_summary'] = self._end(termination, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _end(self, reason, terminated, truncated) -> dict:
        ep = self.ep
        ep['done'] = True
        k = ep['k']
        ep['monitor'].finish(k, k * self.dt)
        dev = [abs(d['lateral_m']) for d in ep['dev']]
        summ = {'episode': ep['n'], 'seed': ep['seed'], 'options': ep['options'], 'scenario': None,
                'motion': 'kinematic', 'simulator': 'neural', 'start': {'t': ep['t0']},
                'route': self.route.doc(), 'route_length_m': round(self.route.length, 3),
                'progress_m': round(ep['s_max'], 3),
                'route_completion': round(min(ep['s_max'] / max(self.route.length, 1e-9), 1.0), 4),
                'distance_m': round(ep['travel'], 3), 'steps': k, 'sim_s': round(k * self.dt, 4),
                'return': round(ep['ret'], 6), 'reward_terms': {t: round(v, 6) for t, v in ep['terms'].items()},
                'termination': reason, 'terminated': terminated, 'truncated': truncated,
                'target_speed_mps': round(ep['feat'].target, 4), 'infractions': ep['monitor'].events,
                'infraction_counts': ep['monitor'].counts(), 'wall_s': round(time.monotonic() - ep['wall0'], 3),
                'run_dir': ep['dir'],
                'deviation': {'max_m': round(max(dev), 4) if dev else 0.0,
                              'mean_m': round(float(np.mean(dev)), 4) if dev else 0.0,
                              'steps_beyond_limit': ep['flagged'],
                              'steps_outside_envelope': None if self.envelope['trusted_m'] is None else
                              sum(1 for d in ep['dev'] if d['within_envelope'] is False),
                              'limit_m': self.deviate_m, 'trusted_m': self.envelope['trusted_m']}}
        summ['success'] = DF.episode_success(summ)
        self._write_run(summ)
        with open(self.episodes_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(summ, separators=(',', ':'), default=float) + '\n')
        self.summary['episodes'].append({k2: summ[k2] for k2 in ('episode', 'seed', 'termination', 'steps',
                                                                   'progress_m', 'deviation', 'run_dir')})
        from ..provenance import write_json
        write_json(os.path.join(self.out, 'closedloop.json'), self.summary)
        return summ

    def _write_run(self, summ: dict):
        """The episode as a falcon-raw run folder (module doc)."""
        from .. import provenance as PV
        ep = self.ep
        d = ep['dir']
        with open(os.path.join(d, 'frames.jsonl'), 'w', encoding='utf-8') as fh:
            for r in ep['rows']:
                fh.write(json.dumps(r, separators=(',', ':')) + '\n')
        with open(os.path.join(d, 'actors.jsonl'), 'w', encoding='utf-8') as fh:
            for r in ep['actor_rows']:
                fh.write(json.dumps(r, separators=(',', ':')) + '\n')
        from .scene import carla_pose_of
        sensors = []
        for c in self.cameras:
            mount, f = self.mounts[c]
            from . import data as D
            w, h = self.obs_size
            K = D.scaled_K(f['K'], f['size'], (w, h))
            sensors.append({'id': c, 'name': c, 'type': 'camera.pinhole', 'width': w, 'height': h,
                            'fov': round(math.degrees(2 * math.atan(w / (2 * K[0]))), 6),
                            'pose': carla_pose_of(mount, optical=True), 'labels': [],
                            'K': [[K[0], 0.0, K[2]], [0.0, K[1], K[3]], [0.0, 0.0, 1.0]],
                            'streams': ['rgb'] if self.write_images else []})
        what = {'from': 'a closed-loop drive through a reconstruction (core.recon.closedloop)',
                'reconstruction': self.rec.path, 'envelope': self.envelope, 'deviate_m': self.deviate_m,
                'labels': 'none: rgb rendered by the reconstruction; actors.jsonl the controller\'s boxes',
                'not_available': ['semantic', 'instance', 'depth', 'objects', 'lidar', 'radar']}
        src = self.src_drive
        drive = dict({k: v for k, v in src.items() if k in ('town', 'xodr', 'opendrive', 'handedness', 'weather')},
                     schema='falcon.drive/1', name=f'neural drive {os.path.basename(self.rec.path)} ep {ep["n"]}',
                     fps=self.fps, seed=ep['seed'],
                     rig={'id': 'recon', 'name': 'reconstruction cameras',
                          'vehicle': (src.get('rig') or {}).get('vehicle') or 'vehicle.lincoln.mkz_2020',
                          'sensors': [{k: v for k, v in s.items() if k not in ('K', 'streams')} for s in sensors]},
                     neural=what, out=d)
        if not drive.get('town'):
            drive['town'] = self.rec.views.get('town') or ''
        PV.write_json(os.path.join(d, 'drive.json'), drive)
        PV.write_json(os.path.join(d, 'dataset.json'), {
            'schema': 'falcon.dataset/1', 'layout': 'falcon-raw', 'name': drive['name'], 'frames': len(ep['rows']),
            'fps': self.fps, 'sensors': sensors, 'labels': {},
            'coordinates': 'CARLA\'s frames: x forward, y right, z up (left-handed); metres, degrees',
            'reconstruction': what})
        PV.write_json(os.path.join(d, 'closedloop.json'), {'schema': SCHEMA, 'episode': ep['n'], 'seed': ep['seed'],
                                                           'envelope': self.envelope, 'deviate_m': self.deviate_m,
                                                           'steps': ep['dev'], 'summary': {
                                                               k: summ[k] for k in ('termination', 'progress_m',
                                                                                    'distance_m', 'deviation',
                                                                                    'infraction_counts')}})
        run = PV.new_run('recon.closedloop', [], ep['seed'], {'options': ep['options'], 'fps': self.fps,
                                                              'deviate_m': self.deviate_m},
                         {'recon/run.json': os.path.join(self.rec.path, 'run.json'),
                          'recon/point_cloud.ply': os.path.join(self.rec.path, 'point_cloud.ply')})
        run.update(status='ok', exit_code=0, finished=PV.utc_now(), name=drive['name'], neural=what,
                   notes=['the camera images are the reconstruction\'s renders, not a simulator\'s ground truth',
                          f'{ep["flagged"]} step(s) beyond the {self.deviate_m} m limit'])
        run['outputs'] = PV.hash_tree(d, skip=('run.json',))
        PV.write_json(os.path.join(d, 'run.json'), run)

    def close(self):
        if self._closed:
            return
        if self.ep is not None and not self.ep['done']:
            self._end('interrupted', False, True)
        self._closed = True


def run_closedloop(recon: str, out_dir: str, policy: str = 'constant', episodes: int = 1, seed: int = 0,
                   deviate_m: float = DEVIATE_M, obs_size=(160, 90), device: str = 'auto', options=None,
                   emitter=None) -> dict:
    """`falcon recon closedloop`: episodes of a policy in the NeuralSim through core.policy.rollout (rollout.json in
    out_dir); returns the summary."""
    from ..policy import rollout
    sim = NeuralSim(recon, out_dir, obs_size=obs_size, deviate_m=deviate_m, device=device, seed=seed)
    pol = policy_of(policy, 'cpu')
    try:
        def prog(k, summ):
            if emitter is not None:
                emitter.progress(k / max(episodes, 1), f'episode {k} of {episodes}: {summ.get("termination")}')
        doc = rollout(pol, sim, episodes, seed, options, out_dir, progress=prog)
    finally:
        sim.close()
    return {'run': out_dir, 'episodes': [e['run_dir'] for e in doc['episodes']], 'aggregate': doc['aggregate'],
            'envelope': sim.envelope, 'deviation': [e['deviation'] for e in doc['episodes']]}
