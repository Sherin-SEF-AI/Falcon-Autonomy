"""The learning environment's client, in the app's Python (numpy; no carla, no torch): FalconDriveEnv starts the
environment worker (workers/gym.py) in CARLA's Python through CarlaService — the same path as every CARLA job, so it
takes the machine-wide CARLA lock, waits for memory and starts Falcon's own server when needed — and talks to it:
commands on the worker's stdin, replies on a Unix socket this process listens on (core.livelink framing; the
observation arrays travel as raw bytes). The observation, action, reward, termination and infraction definitions are
core.drivefeatures'; docs/policies.md is the guide.

    cfg = gym_config(project, 'world-town10', out_dir, svc.port, svc.tm_port, scenario='lead_brake')
    with FalconDriveEnv(cfg, svc) as env:
        obs, info = env.reset(seed=3)
        obs, reward, terminated, truncated, info = env.step([0.0, 0.3])

The Gymnasium API (gymnasium.Env, 1.x), and what of it is honoured:
* FalconDriveEnv subclasses gymnasium.Env and uses gymnasium.spaces when gymnasium is importable; otherwise a minimal
  shim below (Box, Discrete, Dict: shape, dtype, low, high, n, sample() from a seeded generator, contains(), seed()).
  Falcon itself never needs gymnasium.
* reset(*, seed=None, options=None) -> (obs, info). A seed makes that episode's seed exactly `seed` and reseeds the
  env's generator, from which later resets without a seed draw their episode seeds — so a sequence of resets is
  reproducible from the first seed. Unlike Gymnasium, an env never seeded draws from the config's seed, not from
  entropy (Falcon's runs are reproducible by default). options: scenario (a built-in key, a scenario document or
  None for free driving), values (scenario parameters; the rest are drawn with the episode seed), motion
  ('kinematic' | 'physics'), route_m, max_episode_s, target_speed_kmh, start_speed_kmh.
* step(action) -> (obs, reward: float, terminated: bool, truncated: bool, info: dict). terminated: a terminal state
  of the task (collision, off road, route deviation, blocked, route complete — per terminate_on); truncated: the time
  limit (max_episode_s, a scenario's duration_s by default). Calling step() after either without reset() raises
  (Gymnasium leaves it undefined). An action outside the space is clipped (continuous) or refused (an index out of
  range, a non-finite value: ValueError).
* observation_space: Dict({'state': Box(−inf, inf, (N,), float32)} + optional 'bev' Box(0, 1, (8, H, W), uint8),
  'camera' Box(0, 255, (H, W, 3), uint8)); action_space: Box(−1, 1, (2,), float32) or Discrete(25).
* close() ends the worker (it tidies the server) and is idempotent; the env is also a context manager.
* render() returns None (render_mode None: there is no human view; ask for the 'camera' observation instead).
  metadata = {'render_modes': []}; spec is None; no vector API: one CARLA server can host one synchronous master,
  and Falcon runs one CARLA job per GPU at a time, so a second environment on the same server is not possible and
  more servers do not fit this machine's memory — PPO (core.rl) collects from one environment sequentially.
Not verified: gymnasium's env_checker (gymnasium is not installed on the reference machine).

Timeouts: start_timeout_s for the worker to come up (it may wait for the CARLA lock, memory or the server to start),
step_timeout_s per reset or step; a worker that ends or stops answering raises EnvError with run.json's reason.
"""
from __future__ import annotations

import os
import socket
import tempfile
import time
import uuid

import numpy as np

from . import drivefeatures as DF

try:                                       # optional: Falcon never needs it
    import gymnasium as _gymnasium
    from gymnasium import spaces as _gspaces
    HAVE_GYMNASIUM = True
except ImportError:                        # the reference machine: not installed
    _gymnasium = None
    _gspaces = None
    HAVE_GYMNASIUM = False

GYM_SCHEMA = 'falcon.gym_config/1'


# ============================================================================================ spaces (shim)
class Box:
    """A box in R^shape (gymnasium.spaces.Box's attributes and behaviour for sample/contains)."""

    def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape) if shape is not None else np.shape(low)
        self.low = np.broadcast_to(np.asarray(low, np.float64), self.shape).astype(self.dtype)
        self.high = np.broadcast_to(np.asarray(high, np.float64), self.shape).astype(self.dtype)
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def sample(self):
        """Uniform where bounded on both sides, normal where unbounded, shifted exponential where half-bounded
        (as gymnasium's Box); integers uniform over [low, high]."""
        if self.dtype.kind in 'iu':
            return self.np_random.integers(self.low, self.high.astype(np.int64) + 1, size=self.shape).astype(self.dtype)
        lo, hi = self.low.astype(np.float64), self.high.astype(np.float64)
        out = np.empty(self.shape)
        both, none = np.isfinite(lo) & np.isfinite(hi), ~np.isfinite(lo) & ~np.isfinite(hi)
        low_only, high_only = np.isfinite(lo) & ~np.isfinite(hi), ~np.isfinite(lo) & np.isfinite(hi)
        out[both] = self.np_random.uniform(lo[both], hi[both])
        out[none] = self.np_random.normal(size=int(none.sum()))
        out[low_only] = lo[low_only] + self.np_random.exponential(size=int(low_only.sum()))
        out[high_only] = hi[high_only] - self.np_random.exponential(size=int(high_only.sum()))
        return out.astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return bool(x.shape == self.shape and np.can_cast(x.dtype, self.dtype, 'same_kind') and
                    np.all(x >= self.low) and np.all(x <= self.high))

    __contains__ = contains

    def __repr__(self):
        return f'Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})'


class Discrete:
    def __init__(self, n, seed=None, start=0):
        self.n, self.start = int(n), int(start)
        self.shape, self.dtype = (), np.dtype(np.int64)
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def sample(self):
        return np.int64(self.start + self.np_random.integers(self.n))

    def contains(self, x) -> bool:
        try:
            v = np.asarray(x)
            return v.shape == () and v.dtype.kind in 'iu' and self.start <= int(v) < self.start + self.n
        except (TypeError, ValueError):
            return False

    __contains__ = contains

    def __repr__(self):
        return f'Discrete({self.n})'


class Dict:
    def __init__(self, spaces: dict, seed=None):
        self.spaces = dict(spaces)
        self.seed(seed)

    def seed(self, seed=None):
        ss = np.random.SeedSequence(seed).spawn(len(self.spaces))
        for sp, s in zip(self.spaces.values(), ss):
            sp.seed(int(s.generate_state(1)[0]))
        return [seed]

    def sample(self):
        return {k: sp.sample() for k, sp in self.spaces.items()}

    def contains(self, x) -> bool:
        return isinstance(x, dict) and set(x) == set(self.spaces) and all(self.spaces[k].contains(x[k]) for k in x)

    __contains__ = contains

    def __getitem__(self, k):
        return self.spaces[k]

    def keys(self):
        return self.spaces.keys()

    def items(self):
        return self.spaces.items()

    def __repr__(self):
        return f'Dict({self.spaces})'


def make_box(low, high, shape, dtype):
    return _gspaces.Box(low, high, shape, dtype) if HAVE_GYMNASIUM else Box(low, high, shape, dtype)


def make_discrete(n):
    return _gspaces.Discrete(n) if HAVE_GYMNASIUM else Discrete(n)


def make_dict(d):
    return _gspaces.Dict(d) if HAVE_GYMNASIUM else Dict(d)


class _ShimEnv:
    """gymnasium.Env's surface when gymnasium is absent."""
    metadata = {'render_modes': []}
    render_mode = None
    spec = None

    @property
    def unwrapped(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


_EnvBase = _gymnasium.Env if HAVE_GYMNASIUM else _ShimEnv


class EnvError(RuntimeError):
    """The environment worker failed, refused a command, or stopped answering."""


# ============================================================================================ the config
def gym_config(project, world_id: str, out_dir: str, port: int, tm_port: int, *, rig_id: str | None = None,
               seed: int = 0, scenario=None, values: dict | None = None, motion: str = 'kinematic',
               action: str = 'continuous', bev=None, camera=None, fps: float = 20.0, traffic_vehicles: int = 0,
               traffic_walkers: int = 0, route_m: float | None = None, max_episode_s: float | None = None,
               target_speed_kmh: float = 30.0, start_speed_kmh: float = 0.0, reward: dict | None = None,
               thresholds: dict | None = None, terminate_on=None, log_actions: bool = True, no_rendering: bool = False,
               idle_timeout_s: float = 900.0, timeout: float = 20.0, workspace: str = '', host: str = '127.0.0.1',
               name: str = '') -> dict:
    """The environment worker's config (falcon.gym_config/1): the world as a drive's (core.drive.drive_config: its
    town or OpenDRIVE file, weather, handedness), the rig's vehicle as the ego, background traffic off by default
    (it makes episodes non-repeatable), and the default episode. scenario: a built-in key, a project scenario id or
    a scenario document; bev: True (drivefeatures.BEV_DEFAULT) or a spec dict; camera: True (128 × 72, 90°, on the
    roof) or {width, height, fov, pose}; timeout: CARLA's client time-out per call (the watchdog allows twice it)."""
    from .drive import drive_config
    rig_id = rig_id or project.doc['rigs'][0]['id']
    d = drive_config(project, world_id, rig_id, out_dir, port, tm_port, seconds=1.0, fps=fps, seed=seed, host=host,
                     name=name or 'learning environment', workspace=workspace)
    if isinstance(scenario, str) and scenario not in ('', 'free'):
        from . import scenarios as SC
        doc = project.item('scenarios', scenario)
        if doc is None and scenario not in SC.LIBRARY:
            raise ValueError(f'no scenario {scenario!r} in the project or the built-ins')
        scenario = doc if doc is not None else scenario
    if motion not in ('kinematic', 'physics'):
        raise ValueError(f'motion must be kinematic or physics, not {motion!r}')
    if action not in ('continuous', 'discrete'):
        raise ValueError(f'action must be continuous or discrete, not {action!r}')
    if bev is True:
        bev = dict(DF.BEV_DEFAULT)
    if camera is True:
        camera = {'width': 128, 'height': 72, 'fov': 90.0}
    cfg = {
        'schema': GYM_SCHEMA, 'name': d['name'], 'host': host, 'port': int(port), 'tm_port': int(tm_port),
        'town': d['town'], 'xodr': d['xodr'], 'opendrive': d['opendrive'], 'handedness': d['handedness'],
        'weather': d['weather'], 'weather_source': d['weather_source'], 'vehicle': d['rig']['vehicle'],
        'traffic': dict(d['traffic'], vehicles=int(traffic_vehicles), walkers=int(traffic_walkers)),
        'fps': float(fps), 'seed': int(seed), 'out': os.path.abspath(out_dir), 'socket': '',
        'scenario': scenario if scenario not in ('', 'free') else None, 'values': dict(values or {}),
        'motion': motion, 'route_m': route_m, 'max_episode_s': max_episode_s,
        'target_speed_kmh': float(target_speed_kmh), 'start_speed_kmh': float(start_speed_kmh), 'bev': bev or None,
        'camera': camera or None,
        'action': action, 'reward': dict(reward or {}), 'thresholds': dict(thresholds or {}),
        'terminate_on': list(terminate_on) if terminate_on is not None else list(DF.DEFAULT_TERMINATE_ON),
        'log_actions': bool(log_actions), 'no_rendering': bool(no_rendering) and not camera,
        'idle_timeout_s': float(idle_timeout_s), 'timeout': float(timeout), 'project': dict(d['project'], rig=rig_id),
    }
    from .oddmonitor import CFG_KEYS
    cfg.update({k: d[k] for k in CFG_KEYS if k in d})     # the workspace's ODD, monitored per step (info['odd'])
    return cfg


def make_service(settings=None, fake_carla: str = '', workspace: str = '', lock_dir: str | None = None):
    """A CarlaService with a subprocess launcher (the CLI's and the tests'); settings default to the user's."""
    from ..sim.service import CarlaService
    from .jobs import JobQueue, SubprocessLauncher
    from .settings import Settings
    return CarlaService(settings or Settings(), JobQueue(SubprocessLauncher(), lock_dir), workspace,
                        fake_carla=fake_carla)


# ============================================================================================ the environment
class FalconDriveEnv(_EnvBase):
    """A driving environment in CARLA (module doc). config: gym_config(); service: a CarlaService (None: one is made
    from settings / fake_carla and shut down at close())."""

    metadata = {'render_modes': []}
    render_mode = None

    def __init__(self, config: dict, service=None, *, settings=None, fake_carla: str = '',
                 start_timeout_s: float = 900.0, step_timeout_s: float = 120.0, close_timeout_s: float = 60.0,
                 title: str = 'Learning environment'):
        if config.get('schema') != GYM_SCHEMA:
            raise ValueError(f'not a gym config (schema {config.get("schema")!r})')
        self.cfg = dict(config)
        self._own_service = service is None
        self.svc = service if service is not None else make_service(settings, fake_carla)
        self.step_timeout_s, self.close_timeout_s = float(step_timeout_s), float(close_timeout_s)
        self.job = None
        self.conn = None
        self._reader = None
        self._msgs: list = []
        self._id = 0
        self._closed = False
        self._needs_reset = True
        self._rng = np.random.default_rng(int(self.cfg.get('seed', 0)))
        self.ready: dict = {}
        try:
            self._start(float(start_timeout_s), title)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ the worker
    @property
    def run_dir(self) -> str:
        return self.cfg['out']

    @property
    def episodes_path(self) -> str:
        return os.path.join(self.cfg['out'], 'episodes.jsonl')

    @property
    def feature_names(self) -> tuple:
        return tuple(f[0] for f in self.ready.get('features', DF.FEATURES))

    def _start(self, timeout, title):
        from ..core.jobs import FINAL
        from ..core.livelink import Reader
        from ..core.paths import ensure_dir
        from ..core.provenance import write_json
        path = os.path.join(tempfile.gettempdir(), f'falcon-gym-{os.getuid()}-{uuid.uuid4().hex[:10]}.sock')
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(path)
            srv.listen(1)
            srv.settimeout(0.25)
            self.cfg['socket'] = path
            run_dir = ensure_dir(self.cfg['out'])
            cfg_path = os.path.join(run_dir, 'gym.json')
            write_json(cfg_path, self.cfg)
            spec = self.svc.worker_spec('gym', [cfg_path], title, 'gym', run_dir, meta={'config': cfg_path})
            self.job = self.svc.queue.submit(spec)
            t0 = time.monotonic()
            while True:
                self.svc.queue.poll()
                try:
                    self.conn, _ = srv.accept()
                    break
                except socket.timeout:
                    pass
                if self.job.state in FINAL:
                    raise EnvError(f'the environment worker ended before it was ready: {self._why()}')
                if time.monotonic() - t0 > timeout:
                    raise EnvError(f'the environment worker was not ready within {timeout:.0f} s '
                                   f'({self.job.wait_reason or self.job.text or self.job.state})')
        finally:
            srv.close()
            try:
                os.unlink(path)                    # the connection lives on; the name is no longer needed
            except OSError:
                pass
        self.conn.settimeout(0.25)
        self._reader = Reader()
        self.ready = self._recv(('ready',), timeout)[0]
        self._make_spaces()

    def _make_spaces(self):
        o = self.ready['observation']
        sp = {'state': make_box(-np.inf, np.inf, tuple(o['state']['shape']), np.float32)}
        if 'bev' in o:
            sp['bev'] = make_box(0, 1, tuple(o['bev']['shape']), np.uint8)
        if 'camera' in o:
            sp['camera'] = make_box(0, 255, tuple(o['camera']['shape']), np.uint8)
        self.observation_space = make_dict(sp)
        a = self.ready['action']
        self.action_kind = a['kind']
        self.action_space = make_discrete(a['n']) if a['kind'] == 'discrete' else \
            make_box(np.asarray(a['low'], np.float32), np.asarray(a['high'], np.float32), (len(a['low']),),
                     np.float32)

    def _why(self) -> str:
        from ..core.provenance import read_json
        msg = ''
        try:
            run = read_json(os.path.join(self.cfg['out'], 'run.json'))
            msg = run.get('error') or run.get('status') or ''
        except (OSError, ValueError):
            pass
        if not msg and self.job is not None:
            tail = [t for lvl, t in self.job.log if lvl in ('error', 'stderr', 'warning') and t][-3:]
            msg = ' | '.join(tail) or f'exit {self.job.exit_code}'
        return msg

    def _recv(self, kinds, timeout, want_id=None):
        """The next message of one of these kinds (and id); messages of other kinds are dropped."""
        from ..core.jobs import FINAL
        t0 = time.monotonic()
        while True:
            while self._msgs:
                h, payload = self._msgs.pop(0)
                if h.get('t') == 'bye':
                    raise EnvError(f'the environment worker ended: {h.get("reason")} ({self._why()})')
                if h.get('t') in kinds and (want_id is None or h.get('id') == want_id):
                    return h, payload
            try:
                b = self.conn.recv(1 << 20)
                if not b:
                    raise EnvError(f'the environment worker closed its connection ({self._why()})')
                self._msgs += self._reader.feed(b)
                continue
            except socket.timeout:
                pass
            if self.job is not None and self.job.state in FINAL:
                raise EnvError(f'the environment worker ended: {self._why()}')
            if time.monotonic() - t0 > timeout:
                raise EnvError(f'no reply from the environment worker within {timeout:.0f} s')

    def _request(self, cmd: dict):
        if self._closed:
            raise EnvError('the environment is closed')
        self._id += 1
        cmd = dict(cmd, id=self._id)
        if not self.svc.queue.send(self.job, cmd):
            raise EnvError(f'the environment worker is not running ({self._why()})')
        h, payload = self._recv(('obs', 'error'), self.step_timeout_s, self._id)
        if h['t'] == 'error':
            raise EnvError(h.get('error') or 'the environment refused the command')
        return h, payload

    def _obs(self, h, payload) -> dict:
        out, off = {}, 0
        for a in h.get('arrays') or ():
            n = int(a['nbytes'])
            arr = np.frombuffer(payload[off:off + n], dtype=np.dtype(a['dtype'])).reshape(a['shape']).copy()
            off += n
            out[a['key']] = DF.unpack_bev(arr) if a['key'] == 'bev' else arr
        return out

    # ------------------------------------------------------------------ the Gymnasium API
    def reset(self, *, seed=None, options=None):
        if HAVE_GYMNASIUM:
            super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            ep_seed = int(seed)
        else:
            ep_seed = int(self._rng.integers(0, 2 ** 31 - 1))
        h, payload = self._request({'cmd': 'reset', 'seed': ep_seed, 'options': dict(options or {})})
        self._needs_reset = False
        return self._obs(h, payload), h.get('info') or {}

    def step(self, action):
        if self._needs_reset:
            raise EnvError('step() needs reset() first: the episode ended or never started')
        if self.action_kind == 'discrete':
            a = int(np.asarray(action).reshape(-1)[0])
        else:
            a = [float(x) for x in np.asarray(action, np.float64).reshape(-1)]
        h, payload = self._request({'cmd': 'step', 'action': a})
        term, trunc = bool(h['terminated']), bool(h['truncated'])
        self._needs_reset = term or trunc
        return self._obs(h, payload), float(h['reward']), term, trunc, h.get('info') or {}

    def render(self):
        return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        from ..core.jobs import FINAL
        try:
            if self.job is not None and self.job.state not in FINAL:
                self.svc.queue.send(self.job, {'cmd': 'close'})
                if not self.svc.queue.wait(self.job, self.close_timeout_s, interval=0.05):
                    self.svc.queue.cancel(self.job)
                    if not self.svc.queue.wait(self.job, 20.0) and self.job.handle is not None:
                        self.job.handle.kill()
        finally:
            if self.conn is not None:
                try:
                    self.conn.close()
                except OSError:
                    pass
            if self._own_service:
                self.svc.shutdown()

    @property
    def exit_status(self) -> str:
        """The worker's job state after close(): done, failed or cancelled."""
        return self.job.state if self.job is not None else 'not started'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
