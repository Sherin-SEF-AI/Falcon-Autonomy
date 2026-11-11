"""Level-k road users in the neural closed loop (numpy; core.levelk): an ActorController for core.recon.closedloop.
NeuralSim whose tracked road users decide their own accelerations and lateral offsets against the ego the policy
drives and against each other, instead of replaying their logged motion.

    sim = NeuralSim(recon, out_dir, controller=LevelKActors({'two_wheeler': {'k': 1, 'alpha': 0.6}}))
    falcon recon closedloop RECON --actors levelk [--levelk-params JSON|FILE]

Frames. The reconstruction's world is right-handed (x, y left, z up); core.levelk, like the ego's state in NeuralSim
(sim.ep['state'] = [x, y, yaw°, v]), works in CARLA's convention of that world: y mirrored, yaw positive to the right.
Every logged pose is taken into CARLA's convention (x, −y, −yaw) once, level-k runs there with the ego's state as it
is, and each pose is mirrored back for rendering: the box centre (x, −y, z) and the rotation Rz(yaw) · the logged
tilt, where z and the tilt (roll and pitch: Rz(−ψ_logged) · R_logged) come from the logged annotation nearest along the
track's path — the reconstruction has no road surface.

Who is driven. Every dynamic track of a road-user kind (car — a bus or truck by its class —, autorickshaw,
two-wheeler, pedestrian) is a controlled level-k agent; static tracks and rigid objects replay their logged poses and
are obstacles the agents see (predicted at level 0 with a desired speed of their logged one, at least 0.3 m/s). The ego
is an uncontrolled agent predicted at level 0 by everyone (core.levelk: "vs a level-0 ego"), on the logged ego path,
with the episode's target speed as its desired speed.

* An agent joins at the first instant its logged track is shown at or after the episode's start, from its logged
  pose and speed there (the speed along its heading), and is driven to the episode's end: it does not vanish when its
  logged track ends.
* Its reference path is its own logged path (the annotated centres in time order, points closer than 0.3 m to the
  last kept dropped; a track that barely moved gets a straight line along its logged heading), extrapolated straight
  past either end (core.levelk.RefPath). Lateral targets (a lane change, a two-wheeler on the lane line, an
  autorickshaw's squeeze) are offsets from it, unbounded: the reconstruction has no map of the carriageway.
* Parameters per kind from a traffic profile's behaviour.levelk block ({kind: {k, lambda, alpha, v_des, t_gap,
  compliance}}, '*' for every kind; core.levelk.profile_params) or a calibration (falcon levelk calibrate
  --from-nearmiss: its behaviour.levelk). The desired speed, when the block does not set it, is the track's logged
  median speed over the samples where it moved (≥ 2 m/s), else the type's.
* Right of way (only level-0 gap acceptance reads it): the ego has it; priority 'road' (the default) also gives it to
  tracks travelling along the ego's road — their logged heading within 30° of the logged ego path's (either way) and
  their centre within 8 m of it where they join; 'ego' gives it to the ego only; a list names the track ids that have
  it.
* Decisions are core.levelk.Controller's (re-planned every 0.25 s of simulation time, executed every tick through the
  jerk limit; mode 'sample' draws the quantal response from a generator seeded by the episode's seed: the same seed,
  the same motion). Decisions at a step use every agent's state at the previous tick, the ego's included
  (simultaneous moves).

Appearance. An actor keeps its logged appearance: its Gaussians are drawn at the level-k box, and its wheels spin and
a pedestrian's limbs move as they did at that time in the log (they follow time, not the new motion). An actor driven
off its logged path meets lighting and backgrounds it was never seen against, and its far side may never have been
seen at all.

Each episode's folder gains levelk.json: the agents (type, parameters, right of way, join time), every agent's
trajectory in CARLA's convention (t, x, y, yaw, v, a, s, d), core.levelk.realism over the agents present for the whole
episode (levelk_realistic: the level-k agents' dynamics within the limits and no box overlap with anyone — the ego's
own dynamics are the policy's), and core.levelk.inching for each two-wheeler and autorickshaw against the vehicles
with the right of way.
Parameters that came from a calibration on dashcam footage mark the episode's run.json (source dashcam): the episode
is then private (assistant.privacy) and never leaves the machine.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .. import levelk as LK
from .closedloop import ActorController

SCHEMA = 'falcon.recon_levelk/1'
ROAD_KINDS = ('car', 'autorickshaw', 'two_wheeler', 'pedestrian')
PATH_STEP_M = 0.3
MOVING_MPS = 2.0
ROAD_DEG = 30.0
ROAD_LATERAL_M = 8.0
EGO = 'ego'


# =========================================================================== frames
def carla_xy(c) -> tuple:
    """CARLA's (x, y) of a right-handed point."""
    return float(c[0]), -float(c[1])


def yaw_rh(R) -> float:
    """Heading (rad, counter-clockwise from x) of a right-handed rotation's x axis."""
    R = np.asarray(R, float)
    return math.atan2(R[1, 0], R[0, 0])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def lk_type(track) -> str:
    """core.levelk's type of a reconstruction track: its kind, a car whose class says bus or truck a bus."""
    if track.kind == 'car':
        text = f'{track.cls} {track.category}'.lower()
        return 'bus' if ('bus' in text or 'truck' in text) else 'car'
    return LK.agent_type(track.kind)


def params_table(block) -> dict:
    """A behaviour.levelk block ({kind: {k, lambda, alpha, …}}) checked and keyed by core.levelk type (core.levelk.
    profile_params); {} for None."""
    if not block:
        return {}
    if not isinstance(block, dict):
        raise ValueError('level-k parameters are a JSON object {kind: {k, lambda, alpha, v_des, t_gap}}')
    return LK.profile_params({'behaviour': {'levelk': block}})


def load_params(text: str) -> tuple:
    """(block, source) of --levelk-params: inline JSON or a JSON file — a params block, a traffic profile, or a
    calibration (falcon levelk calibrate: its behaviour.levelk). source: {'file', 'private': reason or None}."""
    src = {'file': None, 'private': None}
    t = (text or '').strip()
    if not t:
        return {}, src
    if os.path.isfile(os.path.expanduser(t)):
        p = os.path.abspath(os.path.expanduser(t))
        with open(p, encoding='utf-8') as fh:
            doc = json.load(fh)
        from ...assistant.privacy import _says_dashcam, private_reason
        src = {'file': p, 'private': private_reason(p) or _says_dashcam(doc) or None}
    else:
        try:
            doc = json.loads(t)
        except ValueError as e:
            raise ValueError(f'--levelk-params is neither a file nor JSON: {e}') from None
    if not isinstance(doc, dict):
        raise ValueError('level-k parameters are a JSON object')
    if isinstance(doc.get('behaviour'), dict) and 'levelk' in doc['behaviour']:
        doc = doc['behaviour']['levelk']
    elif isinstance(doc.get('levelk'), dict):
        doc = doc['levelk']
    params_table(doc)
    return doc, src


# =========================================================================== the controller
class LevelKActors(ActorController):
    """Level-k road users (module doc). params: a behaviour.levelk block; mode 'sample' | 'best'; lht: left-hand
    traffic (filtering on the right; None: the source drive's handedness); priority 'road' | 'ego' | track ids;
    source: where the parameters came from (load_params)."""

    name = 'levelk'

    def __init__(self, params: dict | None = None, mode: str = 'sample', lht: bool | None = None,
                 priority='road', weights: dict | None = None, source: dict | None = None):
        if mode not in ('sample', 'best'):
            raise ValueError(f'mode {mode!r}: sample or best')
        self.block = dict(params or {})
        self.table = params_table(self.block)
        self.mode, self.lht_opt, self.priority, self.weights = mode, lht, priority, weights
        self.source = dict(source or {'file': None, 'private': None})
        self.sim = None
        self.agents: dict = {}
        self.static: dict = {}
        self.trace: dict = {}

    # ------------------------------------------------------------------ setup
    def _ego_path(self, sim):
        xy = np.array([[p['x'], p['y']] for p in sim.log_pose], float)
        keep = [0]
        for k in range(1, len(xy)):
            if np.hypot(*(xy[k] - xy[keep[-1]])) > PATH_STEP_M:
                keep.append(k)
        pts = xy[keep]
        if len(pts) < 2:
            r = math.radians(sim.log_pose[0]['yaw'])
            pts = np.array([xy[0], xy[0] + [math.cos(r), math.sin(r)]])
        return LK.RefPath(pts)

    def _track_path(self, tr):
        """(RefPath in CARLA's convention, s of each annotation, z of each, tilt of each (3, 3))."""
        from .actors import quat_to_rot_np
        R = quat_to_rot_np(tr.quat)
        xy = np.array([carla_xy(c) for c in tr.centre], float)
        keep = [0]
        for k in range(1, len(xy)):
            if np.hypot(*(xy[k] - xy[keep[-1]])) > PATH_STEP_M:
                keep.append(k)
        pts = xy[keep]
        if len(pts) < 2 or float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) < 1.0:
            yaw_c = -math.degrees(yaw_rh(R[0]))
            path = LK.RefPath.line(xy[0][0], xy[0][1], yaw_c, length=400.0, back=0.0)
        else:
            path = LK.RefPath(pts)
        s = np.array([path.project(x, y)[0] for x, y in xy])
        tilt = np.array([rz(-yaw_rh(Rk)) @ Rk for Rk in R])
        return path, s, np.asarray(tr.centre, float)[:, 2], tilt

    def _has_priority(self, sim, tid, x, y, yaw_c) -> bool:
        if self.priority == 'ego':
            return False
        if self.priority != 'road':
            return tid in set(self.priority or ())
        s, d = self.ego_path.project(x, y)
        h = float(self.ego_path.heading(s))
        rel = abs((yaw_c - h + 180.0) % 360.0 - 180.0)
        return abs(d) <= ROAD_LATERAL_M and min(rel, 180.0 - rel) <= ROAD_DEG

    def reset(self, sim, seed: int):
        self.sim = sim
        drive_lht = str((getattr(sim, 'src_drive', None) or {}).get('handedness', '')).lower() == 'left'
        self.lht = drive_lht if self.lht_opt is None else bool(self.lht_opt)
        self.ctl = LK.Controller(seed=int(seed), dt=sim.dt, mode=self.mode, weights=self.weights, lht=self.lht)
        self.seed = int(seed)
        self.ego_path = self._ego_path(sim)
        self.t0 = float(sim.ep['t0'])
        self.agents, self.static, self.trace = {}, {}, {}
        self.pending = {}
        for tid, tr in sim.tracks.items():
            if not len(tr.t):
                continue
            if tr.dynamic and tr.kind in ROAD_KINDS:
                path, s_ann, z_ann, tilt = self._track_path(tr)
                order = np.argsort(s_ann)
                sp = np.linalg.norm(np.diff(tr.centre[:, :2], axis=0), axis=1) / np.maximum(np.diff(tr.t), 1e-6) \
                    if len(tr.t) > 1 else np.zeros(0)
                moving = sp[sp >= MOVING_MPS]
                self.pending[tid] = {'track': tr, 'type': lk_type(tr), 'path': path, 's_ann': s_ann[order],
                                     'z_ann': z_ann[order], 'tilt': tilt[order],
                                     'v_logged': float(np.median(moving)) if len(moving) else None}
            else:
                self.static[tid] = tr
        self._ego_prev = list(sim.ep['state'])
        self._join(self.t0)
        self._record(self.t0)

    def _join(self, t: float):
        for tid in list(self.pending):
            p = self.pending[tid]
            tr = p['track']
            po = tr.pose(t)
            if po is None:
                continue
            R, c = po
            x, y = carla_xy(c)
            yaw_c = -math.degrees(yaw_rh(R))
            s, d = p['path'].project(x, y)
            vel = tr.velocity(t)
            hd = math.radians(float(p['path'].heading(s)))
            v = max(float(vel[0]) * math.cos(hd) + float(-vel[1]) * math.sin(hd), 0.0)
            typ = p['type']
            extra = {}
            if 'v_des' not in (self.table.get(typ) or {}) and 'v_des' not in (self.table.get('*') or {}) and \
                    p['v_logged'] is not None:
                extra['v_des'] = float(np.clip(p['v_logged'], *LK.PARAM_BOUNDS['v_des']))
            params = LK.params_for(self.table, typ, extra)
            L, W, _H = tr.size
            self.agents[tid] = dict(p, id=tid, s=float(s), d=float(d), v=v, a=0.0, d_ref=float(d), params=params,
                                    length=float(L), width=float(W), joined_t=round(t - self.t0, 6),
                                    priority=self._has_priority(self.sim, tid, x, y, yaw_c), controlled=True)
            del self.pending[tid]

    # ------------------------------------------------------------------ per tick
    def _ego_agent(self, state):
        sim = self.sim
        x, y, _yaw, v = state
        s, d = self.ego_path.project(x, y)
        target = float(getattr(sim.ep.get('feat'), 'target', None) or sim.target)
        return {'id': EGO, 'type': 'car', 'path': self.ego_path, 's': s, 'd': d, 'v': float(v), 'a': 0.0,
                'controlled': False, 'priority': True, 'd_ref': d,
                'params': {'level': 0, 'v_des': float(np.clip(target, *LK.PARAM_BOUNDS['v_des']))},
                'length': sim.ego_len, 'width': sim.ego_wid}

    def _static_agents(self, t: float) -> list:
        out = []
        for tid, tr in self.static.items():
            po = tr.pose(t)
            if po is None:
                continue
            R, c = po
            x, y = carla_xy(c)
            yaw_c = -math.degrees(yaw_rh(R))
            vel = tr.velocity(t)
            v = float(np.hypot(vel[0], vel[1]))
            out.append({'id': tid, 'type': lk_type(tr) if tr.kind in ROAD_KINDS else 'car',
                        'path': LK.RefPath.line(x, y, yaw_c, length=200.0, back=0.0), 's': 0.0, 'd': 0.0, 'v': v,
                        'a': 0.0, 'controlled': False, 'priority': False, 'd_ref': 0.0,
                        'params': {'level': 0, 'v_des': max(v, LK.PARAM_BOUNDS['v_des'][0])},
                        'length': float(tr.size[0]), 'width': float(tr.size[1])})
        return out

    def _agent_dict(self, a: dict) -> dict:
        return {k: a[k] for k in ('id', 'type', 'path', 's', 'd', 'v', 'a', 'controlled', 'priority', 'd_ref',
                                  'params', 'length', 'width')}

    def step(self, sim, t: float, dt: float):
        t_prev = t - dt
        agents = [self._ego_agent(self._ego_prev)] + self._static_agents(t_prev) + \
            [self._agent_dict(a) for a in self.agents.values()]
        acts = self.ctl.step(t_prev, agents)
        for tid, a in self.agents.items():
            b = acts.get(tid)
            if b is None:
                continue
            LK.execute(a, b['plan_accel'], b['lateral'], dt)
            a['level'], a['brake_light'] = b['level'], b['brake_light']
        self._ego_prev = list(sim.ep['state'])
        self._join(t)
        self._record(t)

    # ------------------------------------------------------------------ poses
    def _pose(self, a: dict) -> tuple:
        x, y, yaw_c = (float(q) for q in a['path'].at(a['s'], a['d']))
        k = int(np.clip(np.searchsorted(a['s_ann'], a['s']), 0, len(a['s_ann']) - 1))
        if k > 0 and abs(a['s_ann'][k - 1] - a['s']) < abs(a['s_ann'][k] - a['s']):
            k -= 1
        z = float(np.interp(a['s'], a['s_ann'], a['z_ann']))
        psi = -math.radians(yaw_c)
        R = rz(psi) @ a['tilt'][k]
        c = np.array([x, -y, z])
        v = np.array([a['v'] * math.cos(psi), a['v'] * math.sin(psi), 0.0])
        return R, c, v

    def poses(self, t: float) -> dict:
        out = {}
        for tid, tr in self.static.items():
            po = tr.pose(t)
            if po is not None:
                out[tid] = (po[0], po[1], tr.velocity(t))
        for tid, a in self.agents.items():
            out[tid] = self._pose(a)
        return out

    # ------------------------------------------------------------------ the episode's record
    def _record(self, t: float):
        st = self.sim.ep['state']
        e = self.trace.setdefault(EGO, {k: [] for k in ('t', 'x', 'y', 'yaw', 'v', 'a', 's', 'd')})
        s, d = self.ego_path.project(st[0], st[1])
        prev_v = e['v'][-1] if e['v'] else st[3]
        for key, val in (('t', t - self.t0), ('x', st[0]), ('y', st[1]), ('yaw', st[2]), ('v', st[3]),
                         ('a', (st[3] - prev_v) / self.sim.dt if e['v'] else 0.0), ('s', s), ('d', d)):
            e[key].append(round(float(val), 5))
        for tid, a in self.agents.items():
            x, y, yaw = (float(q) for q in a['path'].at(a['s'], a['d']))
            tr = self.trace.setdefault(tid, {k: [] for k in ('t', 'x', 'y', 'yaw', 'v', 'a', 's', 'd')})
            for key, val in (('t', t - self.t0), ('x', x), ('y', y), ('yaw', yaw), ('v', a['v']), ('a', a['a']),
                             ('s', a['s']), ('d', a['d'])):
                tr[key].append(round(float(val), 5))

    def episode_doc(self) -> dict:
        """levelk.json of the episode (module doc)."""
        sim = self.sim
        info = [{'id': EGO, 'type': 'car', 'controlled': False, 'priority': True,
                 'length': sim.ego_len, 'width': sim.ego_wid, 'joined_t': 0.0,
                 'params': {'level': 0, 'v_des': round(float(sim.ep['feat'].target), 4)}}]
        for tid, a in self.agents.items():
            info.append({'id': tid, 'type': a['type'], 'controlled': True, 'priority': a['priority'],
                         'length': a['length'], 'width': a['width'], 'joined_t': a['joined_t'],
                         'params': {k: (round(v, 6) if isinstance(v, float) else v) for k, v in a['params'].items()},
                         'path': a['path'].to_doc()})
        n = len(self.trace[EGO]['t'])
        full = [x for x in info if len(self.trace.get(x['id'], {}).get('t', ())) == n]
        sub = {'dt': sim.dt, 'trajectories': {x['id']: self.trace[x['id']] for x in full}, 'agents': full}
        realism = LK.realism(sub) if len(full) > 1 and n > 2 else None
        if realism is not None:          # the level-k agents' own (the ego is the policy under test, judged apart)
            realism['levelk_realistic'] = all(v['realistic_dynamics'] for k, v in realism['agents'].items()
                                              if k != EGO) and not realism['collisions']
        inching = {}
        for x in full:
            if x['controlled'] and x['type'] in ('two_wheeler', 'autorickshaw'):
                r = LK.inching(sub, rider=x['id'], t_gap=float(x['params']['t_gap']))
                inching[str(x['id'])] = r
        return {'schema': SCHEMA, 'seed': self.seed, 'mode': self.mode, 'lht': self.lht, 'dt': sim.dt,
                'frames': 'CARLA\'s convention of the reconstruction\'s world (x, y right, yaw clockwise), metres, '
                          'seconds from the episode\'s start',
                'parameters': self.block, 'parameters_source': self.source.get('file'),
                'decisions': self.ctl.decisions, 'agents': info, 'not_joined': sorted(str(k) for k in self.pending),
                'trajectories': {str(k): v for k, v in self.trace.items()},
                'realism': realism, 'inching': inching,
                'appearance': 'each actor keeps its logged appearance: drawn at the level-k box, its wheels and a '
                              'pedestrian\'s limbs move as logged at that time'}

    def run_record(self) -> dict:
        """What the episode's run.json gains: the actor model, and the dashcam marker for parameters calibrated on
        dashcam footage (the run is then private)."""
        out = {'actors': {'model': 'levelk', 'mode': self.mode, 'parameters_file': self.source.get('file')}}
        if self.source.get('private'):
            out['source'] = {'format': 'dashcam', 'kind': 'levelk-parameters',
                             'why': 'level-k parameters calibrated on the dashcam footage'}
        return out
