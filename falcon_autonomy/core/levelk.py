"""Level-k game-theoretic traffic (numpy only, standard library and numpy at import: it runs per tick inside CARLA's
Python, where there is no torch): road users of several types that choose discrete accelerations and lateral targets
by best-responding to what they predict the others will do, a level below their own. core.trafficgraph (torch, the
app's Python) calibrates their parameters by maximum likelihood and searches adversarial scenarios through a
differentiable rollout; docs/levelk.md is the guide.

    falcon levelk simulate [--scene crossing|merge|filtering|pedestrian|platoon] [--level K]... [--seconds S] --out DIR
    falcon levelk check [--scene NAME]... [--seeds N] [--out DIR]     realism of simulated agents (core.advfeatures)
    falcon levelk calibrate --log FILE... --out DIR                   maximum likelihood (core.trafficgraph; torch)

Frames and units as core.drivefeatures: CARLA's (x forward, y right, z up, left-handed), metres, seconds; a heading
yaw has its right at (−sin yaw, cos yaw).

Agents (TYPES). car, two_wheeler, autorickshaw, bus, pedestrian, each with a box, a top speed, an acceleration limit
and its lateral targets: every type may keep its lane (0) or change a lane (± LANE_W, pedestrians excepted); a
two-wheeler may also share or filter (± FILTER_M: on the lane line beside a car) and an autorickshaw squeeze (±
SQUEEZE_M). An agent moves along a reference path (a lane centre polyline, RefPath) at arc length s with a lateral
offset d from it (+ right); its parameters: level k, assertiveness α ∈ [0, 1], the quantal-response temperature λ
(its rationality: λ → ∞ is a best response), desired speed v_des, gap acceptance t_gap (s) and light compliance.

Actions. A plan is a constant acceleration from ACCELS (−4 … 2 m/s²) and a lateral target, held over the horizon
(HORIZON_S = 3 s in steps of DT_PLAN = 0.25 s); speeds stay within [0, the type's top speed], the lateral offset moves
towards its target at the type's lateral speed. Agents re-plan every DT_PLAN; between decisions (and in CARLA, per
tick) the executed acceleration follows the plan's at no more than JERK_MAX m/s³ (core.advfeatures.JERK_ACT), so a
level-k agent's jerk and accelerations stay within core.advfeatures.REALISM_LIMITS by construction.

Utility of a plan (utility(); WEIGHTS; the same function evaluates numpy and torch arrays — core.trafficgraph
differentiates it in α, λ, v_des and t_gap):
    U = W_prog (1 + α) · progress + W_speed · speed − W_comf · comfort − W_safe (1 − α) · barrier
        − W_coll · collisions − W_light · compliance · red − W_lane · lane
  progress    arc length gained over the horizon, at most v_des · horizon, / (v_des · horizon)
  speed       − mean over the horizon of ((v − v_des) / v_des)²
  comfort     (a / A_REF)² + W_latmove · |lateral target − lateral now| / LANE_W
  barrier     mean over the horizon of Σ over the others of exp(−gap / (S0 + v · t_gap / 2)): the TTC / gap barrier
              (gap: the box-to-box clearance in the agent's frame, ≥ 0; its scale grows with speed and t_gap)
  collisions  how many others' boxes the plan comes within COLLISION_MARGIN_M + MARGIN_GROWTH_MPS · t of at some
              step t (the margin absorbs prediction error, which grows with time), + 1 when it would hit a leader
              braking at BRAKE_WORST (worst_case: room for a hard stop ahead); the same for every α: assertive agents
              still avoid collisions
  red         1 when the plan crosses a red stop line on the agent's path
  lane        the plan's final lateral offset from a lane centre or (two-wheelers) a lane line, squared, / LANE_W²
  target      (an adversary with a target road user, weight TARGET_WEIGHT; else 0) + the target's barrier term:
              closeness to it is a gain, so it presses on the target while collisions stay as costly
The choice is quantal (McKelvey & Palfrey 1995): P(plan) ∝ exp(λ U); λ → ∞ best-responds. Simulation samples
from it with a seeded generator (or takes the most likely plan: mode 'best').

Levels (Stahl & Wilson 1995; Costa-Gomes, Crawford & Broseta 2001; in driving, Oyler et al. 2016, Li et al. 2017 —
"Game theoretic modeling of driver and vehicle interactions for verification and validation of autonomous vehicle
control systems", IEEE TCST 26(5)).
  Level 0: non-strategic — IDM car following (Treiber, Hennecke & Helbing 2000: a = a_max [1 − (v/v_des)⁴ −
           (s*/gap)²], s* = S0 + v T + v Δv / (2 √(a_max b))) on the nearest road user in its corridor (its own frame:
           ahead, laterally overlapping its box + CORRIDOR_MARGIN_M), and gap acceptance at a conflict with a road
           user that has the right of way: it goes only when the other arrives later than t_gap, else it stops
           before the conflict; a two-wheeler blocked by a slower leader filters to the lane line when the line is
           clear for t_gap.
  Level k ≥ 1: best-responds (quantally) to the others predicted at level k − 1. A level-0 other is predicted
           REACTIVELY: within the horizon its IDM responds, step by step, to the candidate plan (and to the others'
           predicted motion) — which is what lets a level-1 agent expect a level-0 driver to yield to it. A
           higher-level other commits to its own (most likely) plan. Uncontrolled road users (the ego under test,
           CARLA's traffic) are predicted as level 0 by everyone ("vs a level-0 ego").
"Inching" (a two-wheeler creeping into a gap a level-0 rider would reject, until the driver yields) is not scripted:
it emerges at level 1 against a level-0 ego (measured: docs/levelk.md, tests/test_levelk.py).

The per-tick controller (Controller.step): the interface CARLA's workers (workers/adversary.py, workers/director.py,
workers/levelk_traffic.py) and a neural simulator call — states in, accelerations and lateral targets out:

    ctl = Controller(params_by_id, seed=0, dt=0.05)
    actions = ctl.step(t, agents)    agents: [{'id', 'type', 'path': RefPath, 's', 'd', 'v', 'controlled': bool,
                                               'priority': bool, 'd_ref': its lane centre's offset, 'params': {…},
                                               'stops': [(s_stop, 'Red'|'Green'|…)], 'target': an id | None,
                                               'weights': {…} | None}, ...]
    -> {id: {'accel': m/s² (jerk-limited, executed now), 'lateral': target offset m, 'plan_accel', 'level',
             'brake_light': bool}}     for every controlled agent

Limits: agents follow fixed reference paths (lane changes are lateral offsets from them, not route choices); corridors
use each agent's local heading (approximate on sharp bends); conflicts are where two paths cross (≥ 30°) and the
right of way is the 'priority' flag (no junction rules); plans are constant over the horizon; the reactive prediction
covers level-0 others only; pedestrians walk their path and are given way by nobody unless they are in a corridor.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

SCHEMA = 'falcon.levelk/1'
DT_PLAN = 0.25
HORIZON_S = 3.0
N_STEPS = int(round(HORIZON_S / DT_PLAN))
ACCELS = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)
LANE_W = 3.5
FILTER_M = 0.5 * LANE_W + 0.15           # the lane line beside a car (a filtering two-wheeler's place)
SQUEEZE_M = 0.9
A_REF = 2.0
S0 = 2.0                                  # m: the standstill gap (IDM s0 and the barrier's floor)
IDM_T = 1.2                               # s: IDM's time headway
IDM_B = 2.0                               # m/s²: IDM's comfortable deceleration
IDM_MIN = -4.0                            # m/s²: the hardest braking level 0 is given (core.advfeatures.REALISM_LIMITS)
JERK_MAX = 8.0                            # m/s³ (core.advfeatures.JERK_ACT)
CORRIDOR_MARGIN_M = 0.3
COLLISION_MARGIN_M = 0.3                  # a plan coming this close to another box counts as a collision …
BRAKE_WORST = -6.0                        # m/s²: the hard stop a follower allows for (core.advfeatures.BRAKE_MAX)
MARGIN_GROWTH_MPS = 0.15                  # … plus this per second ahead (the predictions' error grows with time)
CONFLICT_DEG = 30.0
RANGE_M = 80.0                            # others further than this are not considered
BRAKE_LIGHT_DECEL = 0.5                   # m/s²: an executed deceleration beyond this lights the brake lights

TYPES = {
    'car': {'length': 4.5, 'width': 1.9, 'v_des': 11.0, 'v_max': 22.0, 'a_max': 2.0, 'lat_mps': 1.0,
            'lateral': (0.0, -LANE_W, LANE_W), 'line': False},
    'two_wheeler': {'length': 2.0, 'width': 0.8, 'v_des': 11.0, 'v_max': 20.0, 'a_max': 2.0, 'lat_mps': 1.5,
                    'lateral': (0.0, -FILTER_M, FILTER_M, -LANE_W, LANE_W), 'line': True},
    'autorickshaw': {'length': 2.7, 'width': 1.3, 'v_des': 9.0, 'v_max': 18.0, 'a_max': 2.0, 'lat_mps': 1.2,
                     'lateral': (0.0, -SQUEEZE_M, SQUEEZE_M, -LANE_W, LANE_W), 'line': False},
    'bus': {'length': 11.0, 'width': 2.5, 'v_des': 9.0, 'v_max': 16.0, 'a_max': 1.2, 'lat_mps': 0.6,
            'lateral': (0.0, -LANE_W, LANE_W), 'line': False},
    'pedestrian': {'length': 0.5, 'width': 0.5, 'v_des': 1.4, 'v_max': 2.5, 'a_max': 1.0, 'lat_mps': 0.0,
                   'lateral': (0.0,), 'line': False},
}
KIND_TYPE = {'car': 'car', 'truck': 'bus', 'van': 'car', 'bus': 'bus', 'motorcycle': 'two_wheeler',
             'bicycle': 'two_wheeler', 'autorickshaw': 'autorickshaw', 'cycle_rickshaw': 'autorickshaw',
             'pedestrian': 'pedestrian', 'child': 'pedestrian', 'two_wheeler': 'two_wheeler'}
PARAM_DEFAULTS = {'level': 1, 'alpha': 0.3, 'lam': 8.0, 'v_des': None, 't_gap': 3.0, 'compliance': 1.0}
PARAM_BOUNDS = {'level': (0, 3), 'alpha': (0.0, 1.0), 'lam': (0.01, 200.0), 'v_des': (0.3, 40.0),
                't_gap': (0.2, 10.0), 'compliance': (0.0, 1.0)}
WEIGHTS = {'progress': 1.0, 'speed': 0.5, 'comfort': 0.1, 'latmove': 0.5, 'safety': 2.0, 'collision': 50.0,
           'light': 20.0, 'lane': 0.2, 'target': 0.0}
TARGET_WEIGHT = 1.0                       # an agent with a target: the pull towards it (heuristic)
ADVERSARY = {'level': 1, 'alpha': 0.8, 'lam': 20.0, 't_gap': 1.0}      # heuristic: an assertive, ego-targeting agent


def spec(ag: dict) -> dict:
    """An agent's type entry (TYPES) with its own box when it carries one ('length', 'width': a CARLA actor's
    measured box, so predicted clearances are the real ones)."""
    t = TYPES[ag['type']]
    if ag.get('length') is None and ag.get('width') is None:
        return t
    return dict(t, length=float(ag.get('length') or t['length']), width=float(ag.get('width') or t['width']))


def agent_type(kind: str) -> str:
    """The TYPES key of a scenario kind or a type name ('car' for anything unknown)."""
    return KIND_TYPE.get(str(kind), 'car')


ALIASES = {'k': 'level', 'lambda': 'lam'}      # a traffic profile's names (core.project.levelk_params)


def params_of(p: dict | None, typ: str) -> dict:
    """An agent's parameters: PARAM_DEFAULTS over p (the profile's names k and lambda accepted for level and lam),
    v_des the type's when unset; ValueError out of PARAM_BOUNDS."""
    out = dict(PARAM_DEFAULTS, **{ALIASES.get(k, k): v for k, v in (p or {}).items() if v is not None})
    unknown = sorted(set(out) - set(PARAM_DEFAULTS))
    if unknown:
        raise ValueError(f'unknown level-k parameters {unknown} (parameters: {sorted(PARAM_DEFAULTS)})')
    if out['v_des'] is None:
        out['v_des'] = TYPES[typ]['v_des']
    out['level'] = int(out['level'])
    for k, (lo, hi) in PARAM_BOUNDS.items():
        if not lo <= float(out[k]) <= hi:
            raise ValueError(f'level-k parameter {k} = {out[k]} is outside [{lo}, {hi}]')
    return out


def profile_params(tp: dict) -> dict:
    """A traffic profile's level-k parameters by agent type (core.project.levelk_params: {kind: {k, lambda, alpha,
    …}}, kinds as scenario kinds or TYPES names; '*' for every type), checked; {} when it has none."""
    from .project import levelk_params
    out = {}
    for kind, p in levelk_params(tp).items():
        if kind != '*' and kind not in KIND_TYPE and kind not in TYPES:
            raise ValueError(f'level-k parameters for an unknown kind {kind!r} (kinds: {sorted(KIND_TYPE)})')
        typ = '*' if kind == '*' else agent_type(kind)
        params_of(p, 'car' if typ == '*' else typ)
        out[typ] = {ALIASES.get(k, k): v for k, v in (p or {}).items()}
    return out


def params_for(table: dict, typ: str, extra: dict | None = None) -> dict:
    """An agent's parameters from a by-type table (profile_params) and extra overrides."""
    return params_of(dict(table.get('*') or {}, **(table.get(typ) or {}), **(extra or {})), typ)


# ============================================================================================ reference paths
class RefPath:
    """A reference polyline (x, y) with its arc length: at(s, d) → (x, y, yaw°) vectorised, extrapolated straight past
    either end; project(x, y) → (s, d). d is + to the right (CARLA's frame). bounds: (lo, hi), the drivable road's
    lateral extent around it (lateral targets keep the agent's box inside), or None (unbounded)."""

    def __init__(self, pts, bounds=None):
        self.bounds = None if bounds is None else (float(bounds[0]), float(bounds[1]))
        p = np.asarray(pts, float).reshape(-1, 2)
        if len(p) < 2:
            raise ValueError('a reference path needs at least two points')
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
        self.pts = p[keep]
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.r_[0.0, np.cumsum(self.seg_len)]
        self.dir = seg / self.seg_len[:, None]
        self.yaw = np.degrees(np.arctan2(self.dir[:, 1], self.dir[:, 0]))
        self.length = float(self.s[-1])

    @classmethod
    def line(cls, x0, y0, yaw_deg, length=400.0, back=50.0, bounds=None):
        r = math.radians(yaw_deg)
        c, s = math.cos(r), math.sin(r)
        return cls([(x0 - back * c, y0 - back * s), (x0 + length * c, y0 + length * s)], bounds)

    @classmethod
    def from_doc(cls, d: dict):
        return cls(d['pts'], d.get('bounds'))

    def to_doc(self) -> dict:
        return {'pts': np.round(self.pts, 4).tolist(), 'bounds': list(self.bounds) if self.bounds else None}

    def _seg(self, s):
        return np.clip(np.searchsorted(self.s, s, side='right') - 1, 0, len(self.seg_len) - 1)

    def at(self, s, d=0.0):
        s = np.asarray(s, float)
        d = np.broadcast_to(np.asarray(d, float), s.shape)
        k = self._seg(s)
        u = s - self.s[k]
        base = self.pts[k] + self.dir[k] * u[..., None]
        right = np.stack([-self.dir[k][..., 1], self.dir[k][..., 0]], axis=-1)
        q = base + right * d[..., None]
        return q[..., 0], q[..., 1], self.yaw[k]

    def heading(self, s):
        return self.yaw[self._seg(np.asarray(s, float))]

    def project(self, x, y, hint=None, window=60.0):
        """(s, d) of the point nearest (x, y) on the path (within ±window of hint when given)."""
        lo, hi = 0, len(self.seg_len)
        if hint is not None:
            lo = int(max(self._seg(hint - window), 0))
            hi = int(min(self._seg(hint + window) + 1, len(self.seg_len)))
        a = self.pts[lo:hi]
        dv = self.dir[lo:hi]
        rel = np.asarray([x, y], float) - a
        u = np.clip(np.sum(rel * dv, axis=1), 0.0, self.seg_len[lo:hi])
        near = a + dv * u[:, None]
        dist = np.hypot(*(np.asarray([x, y]) - near).T)
        j = int(np.argmin(dist))
        s = float(self.s[lo + j] + u[j])
        if j == 0 and lo == 0 and float(np.sum(rel[0] * dv[0])) < 0:      # before the start: extrapolated
            s = float(np.sum(rel[0] * dv[0]))
        last = hi - 1 - lo
        if j == last and hi == len(self.seg_len):
            along = float(np.sum(rel[last] * dv[last]))
            if along > self.seg_len[-1]:
                s = float(self.s[-2] + along)
        k = lo + j
        r = np.asarray([x, y]) - (self.pts[k] + self.dir[k] * (s - self.s[k]))
        d = float(-r[0] * self.dir[k][1] + r[1] * self.dir[k][0])
        return s, d


def conflict(pa: RefPath, pb: RefPath):
    """(s on a, s on b) where the two paths cross at ≥ CONFLICT_DEG, or None (the first crossing along a)."""
    A0, A1 = pa.pts[:-1], pa.pts[1:]
    B0, B1 = pb.pts[:-1], pb.pts[1:]
    best = None
    for i in range(len(A0)):
        da = A1[i] - A0[i]
        db = B1 - B0
        den = da[0] * db[:, 1] - da[1] * db[:, 0]
        ok = np.abs(den) > 1e-9
        w = B0 - A0[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(ok, (w[:, 0] * db[:, 1] - w[:, 1] * db[:, 0]) / den, -1)
            u = np.where(ok, (w[:, 0] * da[1] - w[:, 1] * da[0]) / den, -1)
        hit = np.nonzero(ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))[0]
        for j in hit:
            ang = abs((pa.yaw[i] - pb.yaw[j] + 180.0) % 360.0 - 180.0)
            if min(ang, 180.0 - ang) < CONFLICT_DEG:
                continue
            sa = pa.s[i] + t[j] * pa.seg_len[i]
            if best is None or sa < best[0]:
                best = (float(sa), float(pb.s[j] + u[j] * pb.seg_len[j]))
        if best is not None:
            return best
    return best


# ============================================================================================ the array shim
class _NP:
    """The few array operations utility() needs, for numpy (core.trafficgraph has the torch twin)."""
    exp = staticmethod(np.exp)
    abs = staticmethod(np.abs)

    @staticmethod
    def minimum(a, b):
        return np.minimum(a, b)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def mean(a, axis):
        return np.mean(a, axis=axis)

    @staticmethod
    def sum(a, axis):
        return np.sum(a, axis=axis)


NP = _NP()


def utility(xp, theta: dict, F: dict, weights: dict | None = None):
    """U per plan (module doc). theta: {'alpha', 'v_des', 't_gap', 'compliance'} (scalars or 0-d tensors); F: plan
    features (plan_features) — ds (P,), v (P, T), a (P,), latmove (P,), gaps (P, T, J), coll (P,), red (P,),
    lane (P,), H (the horizon, s)."""
    w = dict(WEIGHTS, **(weights or {}))
    alpha, vdes, tgap = theta['alpha'], theta['v_des'], theta['t_gap']
    comp = theta.get('compliance', 1.0)
    vd = xp.maximum(vdes * 1.0 + 0.0 * F['ds'], 0.5 + 0.0 * F['ds'])                      # (…, P)
    progress = xp.minimum(F['ds'], vd * F['H']) / (vd * F['H'])
    speed = -xp.mean(((F['v'] - vd[..., None]) / vd[..., None]) ** 2, -1)
    comfort = (F['a'] / A_REF) ** 2 + w['latmove'] * F['latmove'] / LANE_W
    if F['gaps'].shape[-1]:
        scale = S0 + F['v'][..., None] * tgap * 0.5                                        # (…, P, T, 1)
        barrier = xp.mean(xp.sum(xp.exp(-F['gaps'] / scale), -1), -1)
    else:
        barrier = 0.0 * F['ds']
    return (w['progress'] * (1.0 + alpha) * progress + w['speed'] * speed - w['comfort'] * comfort
            - w['safety'] * (1.0 - alpha) * barrier - w['collision'] * F['coll'] - w['light'] * comp * F['red']
            - w['lane'] * F['lane'] + (w['target'] * F['target'] if 'target' in F else 0.0))


def quantal(U, lam) -> np.ndarray:
    """P(plan) ∝ exp(λ U) (numerically stable)."""
    z = float(lam) * (np.asarray(U, float) - np.max(U))
    e = np.exp(z)
    return e / e.sum()


# ============================================================================================ agents and plans
def make_agent(aid, typ, path: RefPath, s=0.0, d=0.0, v=0.0, params=None, controlled=True, priority=False,
               stops=()) -> dict:
    typ = agent_type(typ)
    return {'id': aid, 'type': typ, 'path': path, 's': float(s), 'd': float(d), 'v': float(v), 'a': 0.0,
            'd_ref': 0.0, 'params': params_of(params, typ), 'controlled': bool(controlled),
            'priority': bool(priority), 'stops': list(stops)}


def plans(ag: dict) -> tuple:
    """(accelerations (P,), lateral targets (P,)) of an agent's plan options."""
    t = spec(ag)
    lat = [ag['d_ref'] + o for o in t['lateral']]
    b = ag['path'].bounds
    if b is not None:
        lo, hi = b[0] + t['width'] / 2.0 - 0.05, b[1] - t['width'] / 2.0 + 0.05
        lat = [L for k, L in enumerate(lat) if k == 0 or lo <= L <= hi]
    A, D = np.meshgrid(np.asarray(ACCELS, float), np.asarray(lat, float), indexing='ij')
    return A.ravel(), D.ravel()


def roll(ag: dict, A, D, n=N_STEPS, dt=DT_PLAN) -> dict:
    """Rollouts of constant plans (accel A, lateral target D; (P,)): s, d, v (P, n) and x, y, yaw (P, n) after each
    step. The acceleration reaches the plan's from the agent's current one at JERK_MAX, as it is executed."""
    t = spec(ag)
    A, D = np.asarray(A, float), np.asarray(D, float)
    P = len(A)
    s = np.empty((P, n))
    d = np.empty((P, n))
    v = np.empty((P, n))
    s0, d0, v0 = np.full(P, ag['s']), np.full(P, ag['d']), np.full(P, ag['v'])
    acc = np.full(P, float(ag.get('a', 0.0)))
    step_lat = t['lat_mps'] * dt
    for k in range(n):
        acc = acc + np.clip(A - acc, -JERK_MAX * dt, JERK_MAX * dt)
        v1 = np.clip(v0 + acc * dt, 0.0, t['v_max'])
        s0 = s0 + 0.5 * (v0 + v1) * dt
        d0 = d0 + np.clip(D - d0, -step_lat, step_lat)
        v0 = v1
        s[:, k], d[:, k], v[:, k] = s0, d0, v1
    x, y, yaw = ag['path'].at(s, d)
    return {'s': s, 'd': d, 'v': v, 'x': x, 'y': y, 'yaw': yaw}


def box_gap(xa, ya, yawa, La, Wa, xb, yb, Lb, Wb):
    """Box-to-box clearance in a's frame (b's box taken axis-aligned in a's frame — approximate when they cross at an
    angle): max of the longitudinal and lateral clearances; ≤ 0 overlapping. Broadcasts."""
    r = np.radians(yawa)
    c, sn = np.cos(r), np.sin(r)
    dx, dy = xb - xa, yb - ya
    lon = np.abs(dx * c + dy * sn) - (La + Lb) / 2.0
    lat = np.abs(-dx * sn + dy * c) - (Wa + Wb) / 2.0
    return np.maximum(lon, lat)


def worst_case(sc, ag, R, near, A, D):
    """1 for the plans that would hit a leader (a road user ahead going the same way, within 45°; not a pedestrian)
    braking at BRAKE_WORST from now on its path: the following distance must allow for a hard stop ahead
    (Shalev-Shwartz, Shammah & Shashua 2017, "On a formal model of safe and scalable self-driving cars",
    arXiv:1708.06374 — the longitudinal rule's idea; its formula is not evaluated)."""
    worst = []
    x0, y0, yaw0 = (float(q) for q in ag['path'].at(ag['s'], ag['d']))
    for j in near:
        o = sc.agents[j]
        ox, oy, oyaw = sc.pose(j)
        fx, _fy = _local(x0, y0, yaw0, ox, oy)
        same = math.cos(math.radians(oyaw - yaw0)) > math.cos(math.radians(45.0))
        if o['type'] == 'pedestrian' or not same or fx <= 0:       # leaders only: ahead, the same way
            continue
        Rw = roll(o, [BRAKE_WORST], [o['d']])
        t = spec(o)
        worst.append((Rw['x'], Rw['y'], t['length'], t['width']))
    if not worst:
        return np.zeros(len(A))
    return (plan_features(ag, R, worst, A, D)['coll'] > 0).astype(float)


def plan_features(ag: dict, R: dict, others: list, A, D) -> dict:
    """Plan features for utility(): R the agent's rollouts (roll), others [(x (P|1, T), y, L, W)] their predicted
    positions per plan (P) or shared (1)."""
    t = spec(ag)
    P = len(A)
    gaps = np.zeros((P, R['s'].shape[1], len(others)))
    for j, (ox, oy, L, W) in enumerate(others):
        gaps[:, :, j] = box_gap(R['x'], R['y'], R['yaw'], t['length'], t['width'], ox, oy, L, W)
    margin = COLLISION_MARGIN_M + MARGIN_GROWTH_MPS * DT_PLAN * np.arange(1, R['s'].shape[1] + 1)
    coll = (gaps <= margin[None, :, None]).any(axis=1).sum(axis=1).astype(float) if len(others) else np.zeros(P)
    red = np.zeros(P)
    s_front0 = ag['s'] + t['length'] / 2.0
    for s_stop, state in ag.get('stops') or ():
        if str(state) == 'Red' and s_front0 <= s_stop:
            red = np.maximum(red, (R['s'][:, -1] + t['length'] / 2.0 > s_stop).astype(float))
    dfin = R['d'][:, -1] - ag['d_ref']
    if t['line']:
        dev = np.minimum(np.abs(dfin), np.abs(np.abs(dfin) - FILTER_M))
    else:
        dev = np.abs(dfin - LANE_W * np.round(dfin / LANE_W))
    return {'ds': R['s'][:, -1] - ag['s'], 'v': R['v'], 'a': np.asarray(A, float),
            'latmove': np.abs(np.asarray(D, float) - ag['d']), 'gaps': np.maximum(gaps, 0.0), 'coll': coll,
            'red': red, 'lane': (dev / LANE_W) ** 2, 'H': R['s'].shape[1] * DT_PLAN}


# ============================================================================================ level 0
def idm(v, v_des, gap, dv, a_max):
    """IDM acceleration (module doc); gap ≥ 0.1 m; broadcasts."""
    s_star = S0 + np.maximum(v * IDM_T + v * dv / (2.0 * math.sqrt(a_max * IDM_B)), 0.0)
    free = 1.0 - (np.maximum(v, 0.0) / np.maximum(v_des, 0.1)) ** 4
    return a_max * (free - (s_star / np.maximum(gap, 0.1)) ** 2)


def _local(px, py, yaw, qx, qy):
    r = np.radians(yaw)
    c, s = np.cos(r), np.sin(r)
    dx, dy = qx - px, qy - py
    return dx * c + dy * s, -dx * s + dy * c


def leader(ag: dict, x, y, yaw, others, v_self, same_dir: bool = False):
    """(gap, closing speed v_self − the leader's speed along the agent's heading) to the nearest road user in the
    agent's corridor, (inf, 0) when none. others: [(x, y, L, W, v, yaw)] with array or scalar entries (broadcast over
    plans); same_dir: only those heading within 45° of it."""
    t = spec(ag)
    best_gap, best_dv = None, None
    for ox, oy, L, W, ov, oyaw in others:
        fx, fy = _local(x, y, yaw, ox, oy)
        rel = np.radians(np.asarray(oyaw) - np.asarray(yaw))
        inside = (fx > 0) & (np.abs(fy) < (t['width'] + W) / 2.0 + CORRIDOR_MARGIN_M)
        if same_dir:
            inside = inside & (np.cos(rel) > math.cos(math.radians(45.0)))
        g = np.where(inside, fx - (t['length'] + L) / 2.0, np.inf)
        along = np.asarray(ov) * np.cos(rel)
        dv = np.asarray(v_self) - along
        if best_gap is None:
            best_gap, best_dv = g, np.where(inside, dv, 0.0)
        else:
            closer = g < best_gap
            best_gap = np.where(closer, g, best_gap)
            best_dv = np.where(closer, dv, best_dv)
    if best_gap is None:
        return np.inf, 0.0
    return best_gap, best_dv


def level0(scene, i: int) -> tuple:
    """A level-0 agent's (acceleration, lateral target) now (module doc)."""
    ag = scene.agents[i]
    t = spec(ag)
    p = ag['params']
    x, y, yaw = (float(q) for q in ag['path'].at(ag['s'], ag['d']))
    others = []
    for j, o in enumerate(scene.agents):
        if j == i:
            continue
        ox, oy, oyaw = scene.pose(j)
        if math.hypot(ox - x, oy - y) > RANGE_M:
            continue
        others.append((ox, oy, spec(o)['length'], spec(o)['width'], o['v'], oyaw))
    gap, dv = leader(ag, x, y, yaw, others, ag['v'])
    gap, dv = float(gap), float(dv)
    a = float(idm(ag['v'], p['v_des'], gap if math.isfinite(gap) else 1e6, dv if math.isfinite(gap) else 0.0,
                  t['a_max']))
    # gap acceptance at conflicts with road users that have the right of way
    s_front = ag['s'] + t['length'] / 2.0
    for j, o in enumerate(scene.agents):
        if j == i or not (o['priority'] and not ag['priority']):
            continue
        c = scene.conflict(i, j)
        if c is None:
            continue
        si, sj = c
        to = spec(o)
        if ag['s'] - t['length'] / 2.0 > si + to['width'] / 2.0:     # already past the conflict
            continue
        if o['s'] - to['length'] / 2.0 > sj + t['width'] / 2.0:      # the other has passed it
            continue
        t_other = (sj - t['width'] / 2.0 - (o['s'] + to['length'] / 2.0)) / max(o['v'], 0.1)
        if s_front > si - to['width'] / 2.0:                          # already in the conflict zone: clear it
            continue
        if t_other <= p['t_gap']:
            stop_at = si - to['width'] / 2.0 - CORRIDOR_MARGIN_M - s_front
            a = min(a, float(idm(ag['v'], p['v_des'], max(stop_at, 0.0), ag['v'], t['a_max'])))
    for s_stop, state in ag.get('stops') or ():
        if str(state) in ('Red', 'Yellow') and s_front <= s_stop and p['compliance'] >= 0.5:
            a = min(a, float(idm(ag['v'], p['v_des'], max(s_stop - s_front, 0.0), ag['v'], t['a_max'])))
    lat = ag['d']
    fgap, _ = leader(ag, x, y, yaw, others, ag['v'], same_dir=True)
    if t['line'] and math.isfinite(float(fgap)) and float(fgap) < 15.0 and ag['v'] < p['v_des'] - 1.0:
        side = FILTER_M if scene.lht else -FILTER_M                   # filter on the overtaking side
        if scene.lateral_clear(i, ag['d_ref'] + side, p['t_gap']):
            lat = ag['d_ref'] + side
    return float(np.clip(a, IDM_MIN, t['a_max'])), float(lat)


# ============================================================================================ the scene
class Scene:
    """Agents on reference paths (module doc); lht: left-hand traffic (overtaking on the right)."""

    def __init__(self, agents: list, lht: bool = False):
        self.agents = list(agents)
        self.lht = bool(lht)
        self._conf: dict = {}

    def index(self, aid) -> int:
        return next(k for k, a in enumerate(self.agents) if a['id'] == aid)

    def conflict(self, i, j):
        key = (id(self.agents[i]['path']), id(self.agents[j]['path']))
        if key not in self._conf:
            self._conf[key] = None if self.agents[i]['path'] is self.agents[j]['path'] else \
                conflict(self.agents[i]['path'], self.agents[j]['path'])
        return self._conf[key]

    def pose(self, i):
        a = self.agents[i]
        x, y, yaw = a['path'].at(a['s'], a['d'])
        return float(x), float(y), float(yaw)

    def lateral_clear(self, i, d_target, t_gap) -> bool:
        a = self.agents[i]
        t = spec(a)
        x, y, yaw = (float(q) for q in a['path'].at(a['s'], d_target))
        for j, o in enumerate(self.agents):
            if j == i:
                continue
            ox, oy, _ = self.pose(j)
            fx, fy = _local(x, y, yaw, ox, oy)
            to = spec(o)
            if abs(fy) < (t['width'] + to['width']) / 2.0 + CORRIDOR_MARGIN_M and \
                    -(t['length'] + to['length']) / 2.0 - a['v'] * 0.5 < fx < (t['length'] + to['length']) / 2.0 + \
                    max(a['v'] - o['v'], 0.0) * t_gap:
                return False
        return True

    def near(self, i) -> list:
        x, y, _ = self.pose(i)
        return [j for j in range(len(self.agents)) if j != i and math.hypot(self.pose(j)[0] - x,
                                                                             self.pose(j)[1] - y) <= RANGE_M]


def predict_level0(scene: Scene, j_list: list, cand: dict | None, n=N_STEPS, dt=DT_PLAN, P=1) -> dict:
    """The level-0 agents j_list simulated over the horizon, reacting by IDM to the candidate's rollouts (cand: {'x',
    'y', 'v', 'yaw' (P, n), 'L', 'W'}) and to each other: {j: (x (P, n), y, v, yaw)}. Lateral targets are held."""
    st = {}
    for j in j_list:
        a = scene.agents[j]
        acc, lat = level0(scene, j)
        st[j] = {'s': np.full(P, a['s']), 'd': np.full(P, a['d']), 'v': np.full(P, a['v']), 'lat': lat,
                 'x': np.empty((P, n)), 'y': np.empty((P, n)), 'vv': np.empty((P, n)), 'yaw': np.empty((P, n)),
                 'acc0': acc}
    if not j_list:
        return {}
    for k in range(n):
        cur = {}
        for j in j_list:
            a = scene.agents[j]
            x, y, yaw = a['path'].at(st[j]['s'], st[j]['d'])
            cur[j] = (x, y, yaw)
        for j in j_list:
            a = scene.agents[j]
            t = spec(a)
            p = a['params']
            x, y, yaw = cur[j]
            others = []
            if cand is not None:
                cx, cy = (cand['x'][:, k - 1], cand['y'][:, k - 1]) if k > 0 else (cand['x0'], cand['y0'])
                cv = cand['v'][:, k - 1] if k > 0 else cand['v0']
                cyaw = cand['yaw'][:, k - 1] if k > 0 else cand['yaw0']
                others.append((cx, cy, cand['L'], cand['W'], cv, cyaw))
            for m in j_list:
                if m != j:
                    mx, my, myaw = cur[m]
                    others.append((mx, my, spec(scene.agents[m])['length'],
                                   spec(scene.agents[m])['width'], st[m]['v'], myaw))
            gap, dv = leader(a, x, y, yaw, others, st[j]['v'])
            acc = np.where(np.isfinite(gap), idm(st[j]['v'], p['v_des'], np.where(np.isfinite(gap), gap, 1e6),
                                                 dv, t['a_max']),
                           idm(st[j]['v'], p['v_des'], 1e6, 0.0, t['a_max']))
            acc = np.minimum(acc, st[j]['acc0']) if k == 0 else acc          # its own stop decisions hold
            acc = np.clip(acc, IDM_MIN, t['a_max'])
            v1 = np.clip(st[j]['v'] + acc * dt, 0.0, t['v_max'])
            st[j]['s'] = st[j]['s'] + 0.5 * (st[j]['v'] + v1) * dt
            st[j]['d'] = st[j]['d'] + np.clip(st[j]['lat'] - st[j]['d'], -t['lat_mps'] * dt, t['lat_mps'] * dt)
            st[j]['v'] = v1
        for j in j_list:
            a = scene.agents[j]
            x, y, yaw = a['path'].at(st[j]['s'], st[j]['d'])
            st[j]['x'][:, k], st[j]['y'][:, k], st[j]['vv'][:, k] = x, y, st[j]['v']
            st[j]['yaw'][:, k] = yaw
    return {j: (st[j]['x'], st[j]['y'], st[j]['vv'], st[j]['yaw']) for j in j_list}


class Solver:
    """Levels for every agent of a scene at one decision (module doc): action(i, k) → (accel, lateral target, P,
    U, plans) with the lower levels cached."""

    def __init__(self, scene: Scene, rng=None, mode: str = 'sample', weights: dict | None = None):
        self.scene = scene
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.mode = mode
        self.weights = weights
        self.cache: dict = {}

    def level_of(self, j, k_view):
        """The level j is predicted at by an agent of level k_view."""
        a = self.scene.agents[j]
        if not a['controlled']:
            return 0
        return max(k_view - 1, 0)

    def action(self, i: int, k: int | None = None, chosen: bool = False):
        sc = self.scene
        ag = sc.agents[i]
        k = ag['params']['level'] if k is None else int(k)
        key = (i, k, chosen)
        if key in self.cache:
            return self.cache[key]
        if k <= 0:
            a, lat = level0(sc, i)
            out = {'accel': a, 'lateral': lat, 'level': 0}
            self.cache[key] = out
            return out
        A, D = plans(ag)
        R = roll(ag, A, D)
        t = spec(ag)
        x0, y0, yaw0 = sc.pose(i)
        near = sc.near(i)
        lvl0 = [j for j in near if self.level_of(j, k) == 0]
        cand = {'x': R['x'], 'y': R['y'], 'v': R['v'], 'yaw': R['yaw'], 'L': t['length'], 'W': t['width'],
                'x0': np.full(len(A), x0), 'y0': np.full(len(A), y0), 'v0': np.full(len(A), ag['v']),
                'yaw0': np.full(len(A), yaw0)}
        pred = predict_level0(sc, lvl0, cand, P=len(A))
        others = []
        for j in near:
            o = sc.agents[j]
            to = spec(o)
            if j in pred:
                ox, oy, _ov, _oyaw = pred[j]
            else:
                b = self.action(j, self.level_of(j, k))
                Rj = roll(o, [b['accel']], [b['lateral']])
                ox, oy = Rj['x'], Rj['y']
            others.append((ox, oy, to['length'], to['width']))
        F = plan_features(ag, R, others, A, D)
        F['coll'] = F['coll'] + worst_case(sc, ag, R, near, A, D)
        p = ag['params']
        w = dict(self.weights or {}, **(ag.get('weights') or {}))
        tgt = ag.get('target')
        if tgt is not None:
            ids = [sc.agents[j]['id'] for j in near]
            if tgt in ids:
                j = ids.index(tgt)
                scale = S0 + R['v'] * p['t_gap'] * 0.5
                F['target'] = np.mean(np.exp(-F['gaps'][:, :, j] / scale), 1)
                w.setdefault('target', TARGET_WEIGHT)
        U = utility(NP, p, F, w)
        Pr = quantal(U, p['lam'])
        if self.mode == 'best' or not chosen:
            m = int(np.argmax(Pr))
        else:
            m = int(self.rng.choice(len(Pr), p=Pr))
        out = {'accel': float(A[m]), 'lateral': float(D[m]), 'level': k, 'P': Pr, 'U': U, 'A': A, 'D': D,
               'index': m, 'features': F}
        self.cache[key] = out
        return out


# ============================================================================================ execution
def execute(ag: dict, plan_a: float, plan_lat: float, dt: float):
    """Advance an agent by dt: the executed acceleration towards the plan's at ≤ JERK_MAX, speed within [0, top],
    the lateral offset towards its target at the type's lateral speed. Returns the executed acceleration."""
    t = spec(ag)
    da = JERK_MAX * dt
    a = ag['a'] + float(np.clip(plan_a - ag['a'], -da, da))
    v1 = min(max(ag['v'] + a * dt, 0.0), t['v_max'])
    if v1 <= 0.0 and a < 0:
        a = max(a, -ag['v'] / dt) if ag['v'] > 0 else 0.0
    ag['s'] += 0.5 * (ag['v'] + v1) * dt
    ag['d'] += float(np.clip(plan_lat - ag['d'], -t['lat_mps'] * dt, t['lat_mps'] * dt))
    ag['v'], ag['a'] = v1, a
    return a


class Controller:
    """The per-tick controller (module doc): re-plans every DT_PLAN of simulation time, executes every call."""

    def __init__(self, params: dict | None = None, seed: int = 0, dt: float = 0.05, mode: str = 'sample',
                 weights: dict | None = None, lht: bool = False):
        self.params = dict(params or {})
        self.rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 31337])
        self.dt, self.mode, self.weights, self.lht = float(dt), mode, weights, bool(lht)
        self.plan: dict = {}
        self.t_plan = -math.inf
        self.exec_a: dict = {}
        self.decisions = 0

    def step(self, t: float, agents: list) -> dict:
        sc_agents = []
        for a in agents:
            typ = agent_type(a.get('type', 'car'))
            pa = params_of(a.get('params') or self.params.get(a['id']) or self.params.get('*'), typ)
            sc_agents.append({'id': a['id'], 'type': typ, 'length': a.get('length'), 'width': a.get('width'),
                              'path': a['path'], 's': float(a['s']), 'd': float(a['d']),
                              'v': float(a['v']), 'a': float(self.exec_a.get(a['id'], a.get('a', 0.0))),
                              'd_ref': float(a.get('d_ref', 0.0)), 'params': pa,
                              'controlled': bool(a.get('controlled', True)), 'priority': bool(a.get('priority')),
                              'stops': list(a.get('stops') or ()), 'target': a.get('target'),
                              'weights': a.get('weights')})
        scene = Scene(sc_agents, self.lht)
        if t - self.t_plan >= DT_PLAN - 1e-9:
            self.t_plan = t
            solver = Solver(scene, self.rng, self.mode, self.weights)
            for i, a in enumerate(sc_agents):
                if a['controlled']:
                    b = solver.action(i, chosen=True)
                    self.plan[a['id']] = (b['accel'], b['lateral'], b['level'])
            self.decisions += 1
        out = {}
        for a in sc_agents:
            if not a['controlled'] or a['id'] not in self.plan:
                continue
            pa, pl, lvl = self.plan[a['id']]
            da = JERK_MAX * self.dt
            ea = a['a'] + float(np.clip(pa - a['a'], -da, da))
            if a['v'] <= 0.0 and ea < 0:
                ea = 0.0
            self.exec_a[a['id']] = ea
            out[a['id']] = {'accel': ea, 'lateral': pl, 'plan_accel': pa, 'level': lvl,
                            'brake_light': bool(ea < -BRAKE_LIGHT_DECEL or (a['v'] < 0.1 and pa < 0))}
        return out


# ============================================================================================ simulation
def simulate(scene: Scene, seconds: float, dt: float = 0.05, seed: int = 0, mode: str = 'sample',
             weights: dict | None = None, uncontrolled=None) -> dict:
    """Run a scene: controlled agents by the Controller, uncontrolled ones (None: by level 0 every tick, i.e. IDM
    with gap acceptance; or a function (t, scene, i) → (accel, lateral)). Returns the trajectories {id: {t, x, y, yaw,
    v, a, s, d}} and per-decision levels."""
    ctl = Controller(seed=seed, dt=dt, mode=mode, weights=weights, lht=scene.lht)
    for a in scene.agents:
        ctl.params[a['id']] = a['params']
    traj = {a['id']: {k: [] for k in ('t', 'x', 'y', 'yaw', 'v', 'a', 's', 'd')} for a in scene.agents}
    n = int(round(seconds / dt))
    for k in range(n + 1):
        t = k * dt
        for i, a in enumerate(scene.agents):
            x, y, yaw = scene.pose(i)
            tr = traj[a['id']]
            for key, val in (('t', t), ('x', x), ('y', y), ('yaw', yaw), ('v', a['v']), ('a', a['a']), ('s', a['s']),
                             ('d', a['d'])):
                tr[key].append(round(float(val), 5))
        if k == n:
            break
        acts = ctl.step(t, [dict(a) for a in scene.agents])
        unc = {}
        for i, a in enumerate(scene.agents):
            if not a['controlled']:
                unc[i] = uncontrolled(t, scene, i) if uncontrolled is not None else level0(scene, i)
        for i, a in enumerate(scene.agents):
            if a['controlled']:
                b = acts[a['id']]
                execute(a, b['plan_accel'], b['lateral'], dt)
            else:
                acc, lat = unc[i]
                execute(a, acc, lat, dt)
    return {'schema': SCHEMA, 'dt': dt, 'seconds': seconds, 'trajectories': traj, 'decisions': ctl.decisions,
            'agents': [{'id': a['id'], 'type': a['type'], 'params': a['params'], 'controlled': a['controlled'],
                        'priority': a['priority'], 'path': a['path'].to_doc(), 'length': spec(a)['length'],
                        'width': spec(a)['width']} for a in scene.agents]}


# ============================================================================================ scenes
SCENES = ('crossing', 'merge', 'filtering', 'pedestrian', 'platoon')


def scene(name: str, levels: dict | None = None, params: dict | None = None, lht: bool = False) -> Scene:
    """A built-in scene (module doc's CLI). levels: {agent id: level}; params: {agent id: {param: value}}."""
    levels, params = dict(levels or {}), dict(params or {})

    def P(aid, base=None):
        d = dict(base or {}, **params.get(aid, {}))
        if aid in levels:
            d['level'] = levels[aid]
        return d
    road = (-1.5 * LANE_W, 0.5 * LANE_W + 1.4)               # two lanes (0 and the overtaking one) and a shoulder
    main = RefPath.line(0.0, 0.0, 0.0, 400.0, 0.0, road)          # arc length = x
    if name == 'crossing':
        # a stream of cars with the right of way (the ego first, 2.4 s from the side road; the others 22 m apart,
        # under 2.5 s: no gap a level-0 rider accepts) and a two-wheeler waiting on a side road that crosses it 30 m
        # ahead of the ego
        side = RefPath([(30.0, 12.0), (30.0, -60.0)], (-0.5 * LANE_W, 0.5 * LANE_W))     # heading −y: crosses it
        ags = [make_agent('ego', 'car', main, 8.0, 0.0, 9.0, P('ego', {'level': 0}), controlled=False, priority=True)]
        for k in range(1, 4):
            ags.append(make_agent(f'car{k}', 'car', main, 8.0 - 22.0 * k, 0.0, 9.0, P(f'car{k}', {'level': 0}),
                                  controlled=False, priority=True))
        ags.append(make_agent('rider', 'two_wheeler', side, 6.5, 0.0, 0.0, P('rider', {'level': 1, 'alpha': 0.6})))
        return Scene(ags, lht)
    if name == 'merge':
        # a two-wheeler at the kerb (lateral +2.5 m) wanting into the ego's lane ahead
        return Scene([make_agent('ego', 'car', main, 0.0, 0.0, 9.0, P('ego', {'level': 0}), controlled=False,
                                 priority=True),
                      make_agent('rider', 'two_wheeler', main, 28.0, 2.6, 1.0, P('rider', {'level': 1}))], lht)
    if name == 'filtering':
        # a queue of cars behind a bus, a two-wheeler and an autorickshaw behind them
        ags = [make_agent('bus', 'bus', main, 60.0, 0.0, 3.0, P('bus', {'level': 0, 'v_des': 3.0}))]
        for k in range(3):
            ags.append(make_agent(f'car{k}', 'car', main, 50.0 - 9.0 * k, 0.0, 3.0, P(f'car{k}', {'level': 0})))
        ags.append(make_agent('rider', 'two_wheeler', main, 10.0, 0.0, 6.0, P('rider', {'level': 1})))
        ags.append(make_agent('auto', 'autorickshaw', main, 2.0, 0.0, 6.0, P('auto', {'level': 1})))
        lane2 = RefPath.line(0.0, -LANE_W, 0.0, 400.0, 0.0, (road[0] + LANE_W, road[1] + LANE_W))
        ags.append(make_agent('ego', 'car', lane2, 5.0, 0.0, 8.0,
                              P('ego', {'level': 0}), controlled=False))
        return Scene(ags, lht)
    if name == 'pedestrian':
        cross = RefPath([(25.0, 6.0), (25.0, -6.0)], (0.0, 0.0))
        return Scene([make_agent('auto', 'autorickshaw', main, 0.0, 0.0, 8.0, P('auto', {'level': 1}),
                                 priority=False),
                      make_agent('walker', 'pedestrian', cross, 0.0, 0.0, 1.3, P('walker', {'level': 0}),
                                 priority=True)], lht)
    if name == 'platoon':
        ags = [make_agent('lead', 'car', main, 40.0, 0.0, 10.0, P('lead', {'level': 0, 'v_des': 6.0}))]
        for k, typ in enumerate(('car', 'autorickshaw', 'two_wheeler', 'bus')):
            ags.append(make_agent(f'{typ}{k}', typ, main, 30.0 - 10.0 * k - (6.0 if typ == 'bus' else 0.0), 0.0,
                                  10.0, P(f'{typ}{k}')))
        return Scene(ags, lht)
    raise ValueError(f'unknown scene {name!r} (scenes: {", ".join(SCENES)})')


# ============================================================================================ measures
def inching(sim: dict, rider: str = 'rider', t_gap: float = 3.0, inch_mps: float = 2.5) -> dict:
    """Did the rider enter the corridor of a vehicle with the right of way while that vehicle's time to reach it was
    below t_gap (a gap a level-0 rider rejects), at low speed, and did that vehicle slow for it? From a simulate()
    result of the crossing or merge scene; 'yielded_by' names the vehicle it entered in front of."""
    T = sim['trajectories']
    info = {a['id']: a for a in sim['agents']}
    r = T[rider]
    rt = spec(info[rider])
    t = np.asarray(r['t'])
    rv = np.asarray(r['v'])
    best = None
    for eid, e in T.items():
        if eid == rider or not info[eid].get('priority'):
            continue
        et = spec(info[eid])
        ex, ey, eyaw = np.asarray(e['x']), np.asarray(e['y']), np.asarray(e['yaw'])
        fx, fy = _local(ex, ey, eyaw, np.asarray(r['x']), np.asarray(r['y']))
        ahead = fx - (et['length'] + rt['length']) / 2.0
        in_front = (np.abs(fy) < (et['width'] + rt['width']) / 2.0) & (ahead > 0)
        if not in_front.any():
            continue
        k = int(np.argmax(in_front))
        if best is None or k < best[0]:
            best = (k, eid, e, ahead)
    out = {'entered': best is not None, 'entry_t': float(t[best[0]]) if best else None}
    if best is None:
        return dict(out, inched=False)
    k, eid, e, ahead = best
    ev = np.asarray(e['v'])
    tta = float(ahead[k] / ev[k]) if ev[k] > 0.1 else None
    first = int(np.argmax(rv > 0.05)) if (rv[:k + 1] > 0.05).any() else k
    approach = rv[first:k + 1]
    out.update(yielded_by=eid, vehicle_time_to_reach_s=tta, entry_speed_mps=float(rv[k]),
               mean_approach_speed_mps=float(approach.mean()) if len(approach) else 0.0,
               vehicle_max_decel_after_mps2=float(-np.min(np.asarray(e['a'])[k:])),
               vehicle_min_speed_mps=float(ev[k:].min()))
    out['inched'] = bool(tta is not None and tta < t_gap and out['mean_approach_speed_mps'] < inch_mps and
                         out['vehicle_max_decel_after_mps2'] > 0.5)
    return out


def realism(sim: dict, limits: dict | None = None) -> dict:
    """Per agent: core.advfeatures.Dynamics over its trajectory (|longitudinal acceleration|, |lateral acceleration|,
    |jerk|; seconds over core.advfeatures.REALISM_LIMITS) and the pairwise box overlaps (collisions)."""
    from .advfeatures import DYN_BUDGET_S, Dynamics, REALISM_LIMITS
    lim = dict(REALISM_LIMITS, **(limits or {}))
    T = sim['trajectories']
    dt = float(sim['dt'])
    out = {}
    info = {a['id']: a for a in sim['agents']}
    types = {a['id']: a['type'] for a in sim['agents']}
    for aid, tr in T.items():
        dyn = Dynamics(dt, lim)
        for v, yaw in zip(tr['v'], tr['yaw']):
            dyn.update(v, yaw)
        peaks = {k: float(np.max(np.abs(x))) if x else 0.0 for k, x in dyn.samples.items()}
        over = sum(dyn.over_s.values())
        out[aid] = {'type': types[aid], 'peaks': peaks, 'over_s': {k: round(v, 4) for k, v in dyn.over_s.items()},
                    'realistic_dynamics': over <= DYN_BUDGET_S + 1e-9}
    ids = list(T)
    coll = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = T[ids[i]], T[ids[j]]
            ta, tb = spec(info[ids[i]]), spec(info[ids[j]])
            g = box_gap(np.asarray(a['x']), np.asarray(a['y']), np.asarray(a['yaw']), ta['length'], ta['width'],
                        np.asarray(b['x']), np.asarray(b['y']), tb['length'], tb['width'])
            if (g <= 0).any():
                coll.append({'a': ids[i], 'b': ids[j], 't': float(a['t'][int(np.argmax(g <= 0))])})
    return {'agents': out, 'collisions': coll, 'limits': lim,
            'realistic': all(v['realistic_dynamics'] for v in out.values()) and not coll}


# ============================================================================================ logs (calibration)
def decisions_from_sim(sim: dict, agent_ids=None, every_s: float = DT_PLAN) -> list:
    """Decision samples for calibration from a simulate() result (or a log of the same shape): for each listed agent
    every `every_s`, its state, the others' observed positions over the next horizon, and the plan it was seen to
    follow (the nearest of its plan options to its observed motion over the next DT_PLAN)."""
    T = sim['trajectories']
    dt = float(sim['dt'])
    agents = {a['id']: a for a in sim['agents']}
    paths = {aid: RefPath.from_doc(a['path']) for aid, a in agents.items()}
    stride = max(int(round(every_s / dt)), 1)
    hstep = max(int(round(DT_PLAN / dt)), 1)
    out = []
    ids = agent_ids or [aid for aid, a in agents.items() if a.get('controlled', True)]
    n = len(next(iter(T.values()))['t'])
    for aid in ids:
        a = agents[aid]
        tr = T[aid]
        for k in range(0, n - hstep * N_STEPS - 1, stride):
            ag = make_agent(aid, a['type'], paths[aid], tr['s'][k], tr['d'][k], tr['v'][k], a['params'])
            ag['a'] = tr['a'][k]
            ag['length'], ag['width'] = a.get('length'), a.get('width')
            A, D = plans(ag)
            # the observed plan: acceleration over the next DT_PLAN, lateral target by the motion's direction
            a_obs = (tr['v'][k + hstep] - tr['v'][k]) / (hstep * dt)
            d_obs = tr['d'][k + hstep] - tr['d'][k]
            lat_opts = sorted(set(np.round(D, 6)))
            want = tr['d'][k] + np.sign(d_obs) * (abs(d_obs) > 0.05) * 99.0
            lat_t = min(lat_opts, key=lambda L: (abs(L - want) if abs(d_obs) > 0.05 else abs(L - tr['d'][k])))
            m = int(np.argmin(np.abs(A - a_obs) + 10.0 * np.abs(D - lat_t)))
            others = []
            for bid, b in agents.items():
                if bid == aid:
                    continue
                trb = T[bid]
                idx = [min(k + hstep * (q + 1), n - 1) for q in range(N_STEPS)]
                tb = spec(b)
                others.append((np.asarray([trb['x'][q] for q in idx])[None],
                               np.asarray([trb['y'][q] for q in idx])[None], tb['length'], tb['width']))
            R = roll(ag, A, D)
            F = plan_features(ag, R, others, A, D)
            out.append({'agent': aid, 'type': a['type'], 't': tr['t'][k], 'choice': m, 'features': F})
    return out


# ============================================================================================ the CLI
def _write(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, default=float)


def main_levelk(argv) -> int:
    """`falcon levelk simulate|check|calibrate` (module doc). 0 ok, 2 failed or a usage error."""
    import argparse
    from . import protocol as P
    usage = ('usage: falcon levelk simulate [--scene NAME] [--level ID=K]... [--param ID.KEY=V]... [--seconds S] '
             '[--seed N] --out DIR\n'
             '       falcon levelk check [--scene NAME]... [--seeds N] [--seconds S] [--out DIR]\n'
             '       falcon levelk calibrate --log FILE... [--agent ID]... [--steps N] --out DIR')
    if not argv or argv[0] not in ('simulate', 'check', 'calibrate'):
        print(usage, flush=True)
        return P.FAILED if argv and argv[0] not in ('-h', '--help') else 0
    what = argv[0]
    ap = argparse.ArgumentParser(prog=f'falcon levelk {what}')
    if what in ('simulate', 'check'):
        ap.add_argument('--scene', action='append', default=None, choices=SCENES,
                        help='a built-in scene (check: repeat for several; default all)')
        ap.add_argument('--level', action='append', default=[], metavar='ID=K', help='an agent\'s level')
        ap.add_argument('--param', action='append', default=[], metavar='ID.KEY=V',
                        help='an agent\'s parameter (alpha, lam, v_des, t_gap, compliance)')
        ap.add_argument('--seconds', type=float, default=8.0)
        ap.add_argument('--seed', type=int, default=0)
        ap.add_argument('--mode', default='sample', choices=('sample', 'best'),
                        help='sample the quantal response or take the most likely plan')
        ap.add_argument('--lht', action='store_true', help='left-hand traffic (filtering on the right)')
        if what == 'check':
            ap.add_argument('--seeds', type=int, default=5)
        ap.add_argument('--out', '-o', default='' if what == 'check' else None, required=what == 'simulate')
    else:
        ap.add_argument('--log', action='append', required=True,
                        help='a levelk simulate result (trajectories.json) or an episodes.jsonl of a CARLA run')
        ap.add_argument('--agent', action='append', default=None, help='agents to calibrate (default: all '
                                                                       'controlled)')
        ap.add_argument('--steps', type=int, default=300)
        ap.add_argument('--out', '-o', required=True)
    try:
        a = ap.parse_args(argv[1:])
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    try:
        if what == 'calibrate':
            from .trafficgraph import calibrate_logs
            doc = calibrate_logs(a.log, a.out, agents=a.agent, steps=a.steps, log=lambda m: print(m, flush=True))
            print(json.dumps({k: doc[k] for k in ('estimates', 'n_decisions', 'loglik')}, indent=2, default=float),
                  flush=True)
            return P.OK
        levels = {}
        for kv in a.level:
            k_, _, v = kv.partition('=')
            levels[k_] = int(v)
        params: dict = {}
        for kv in a.param:
            k_, _, v = kv.partition('=')
            aid, _, key = k_.partition('.')
            params.setdefault(aid, {})[key] = float(v)
        names = a.scene or (['crossing'] if what == 'simulate' else list(SCENES))
        if what == 'simulate':
            sim = simulate(scene(names[0], levels, params, a.lht), a.seconds, seed=a.seed, mode=a.mode)
            sim['scene'] = names[0]
            sim['realism'] = realism(sim)
            if names[0] in ('crossing', 'merge'):
                sim['inching'] = inching(sim)
            _write(os.path.join(a.out, 'trajectories.json'), sim)
            print(json.dumps({'scene': names[0], 'realistic': sim['realism']['realistic'],
                              'collisions': sim['realism']['collisions'], 'inching': sim.get('inching')},
                             indent=2, default=float), flush=True)
            return P.OK
        rows = []
        for nm in names:
            for sd in range(a.seeds):
                sim = simulate(scene(nm, levels, params, a.lht), a.seconds, seed=sd, mode=a.mode)
                r = realism(sim)
                rows.append({'scene': nm, 'seed': sd, 'realistic': r['realistic'],
                             'collisions': len(r['collisions']),
                             'agents': {k: {'peaks': v['peaks'], 'over_s': v['over_s']}
                                        for k, v in r['agents'].items()}})
        doc = {'schema': SCHEMA + '#check', 'runs': rows,
               'realistic_share': float(np.mean([r['realistic'] for r in rows])),
               'limits': realism(simulate(scene(names[0]), 0.1))['limits']}
        if a.out:
            _write(os.path.join(a.out, 'check.json'), doc)
        print(json.dumps({'runs': len(rows), 'realistic_share': doc['realistic_share'],
                          'unrealistic': [(r['scene'], r['seed']) for r in rows if not r['realistic']]}, indent=2),
              flush=True)
        return P.OK
    except (OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
