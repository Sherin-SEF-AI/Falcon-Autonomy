"""Two-speed failure search: a fast batched kinematic simulator screens a scenario's parameter space, and only the
candidates it rates most critical are confirmed in CARLA (numpy; torch optional for the rollout). `falcon prescreen`;
docs/two-speed-search.md is the manual.

THE FAST SIMULATOR (compile_scenario, rollout, criticality, LowFidelity). A built-in or project scenario
(core.scenarios) is run for a whole batch of parameter sets at once on ONE STRAIGHT ROAD, in road coordinates (s along
the ego's lane, y to its right as CARLA's frame has it):

- the ego is driven by the SAME controller the director's kinematic motion uses (workers.director, driver
  "follower"): plugins.lane_follower.LaneFollower's time-gap rule and speed controller on the nearest actor within
  1.8 m of its route (points every 2 m, 50 m ahead, from the route point nearest the ego — the 1 m and 2 m steps are
  kept), integrated by director.kin_step's speed law (Director.ACCEL, Director.BRAKE). Its constants are read from
  those classes, not copied. On a straight road the ego does not steer. `driver="idm"` replaces the follower by the
  intelligent driver model (core.levelk.idm's formula and constants, clipped to [−BRAKE, ACCEL]) on the same lead — a
  stated driver model for a stack that cannot run outside CARLA; it is a different driver, and the fidelity report
  says how well it ranks.
- every scenario vehicle follows its path as the director drives it: the set speed ramping at the action's rate (a
  stand-in within its class's top speed and acceleration, core.assets), director.follow's speed controller and pure
  pursuit (look-ahead max(4 m, 0.8 s · v)) through the kinematic bicycle of kin_step, a lane change moving the path's
  lateral offset linearly; walkers walk along their direction; proxies (cattle, handcart, vendor) move along their
  path at the set speed.
- the timeline fires as the director fires it (time, distance, ahead triggers; speed, brake, stop, go, walk,
  lane_change; open_door is accepted and has no kinematic effect — in kinematic motion the director measures
  collisions on the boxes, which a door does not change).
- the objective is core.criticality's scalar, computed by core.criticality's own functions on the simulated
  trajectories (ttc_rows, drac_rows, the box distance and overlap, the pointwise PET): the SAME formula as a CARLA
  run's. Three exact prefilters keep it fast (a pair further apart than its closing speed covers in the TTC horizon
  has no TTC; an actor outside the ego's swept corridor has no DRAC; only the samples that can hold the minimum
  distance are measured exactly). One approximation: a PET above PET_MAX_S (6 s) is taken as absent, which changes
  the criticality by less than 0.9 · exp(−6) ≈ 0.002.

What is reused and what is mirrored: core.criticality's functions and core.search's optimisers and Gaussian process
are called as they are; the controllers' constants are read from Director and LaneFollower. director.kin_step and
director.follow act on one CARLA actor each, and core.trafficgraph.bicycle_rollout takes its controls for the whole
horizon (open loop) — neither can step a batch in closed loop — so their arithmetic is written out here for arrays
and checked against the real LaneFollower.act and director.follow, step for step, in tests/test_fastsim.py.

What it does not have (LIMITS): the map (one straight lane of LANE_WIDTH_M with one lane each side; a scenario
that needs a junction, the opposite lanes or a speed breaker is refused by name — unsupported()); the vehicles'
real boxes (the director's nominal sizes, core.assets' for the Indian kinds; the ego NOMINAL_EGO_M); CARLA's physics
(it mirrors kinematic motion only); the weather; background traffic; a level-k or replayed actor; the RSS label
(the search's default objective counts only avoidable failures, which the fast simulator cannot label — the
confirmed pairs show what that costs). It is a screen, never a result: every reported failure is a CARLA run.

THE TWO-SPEED LOOP (TwoSpeed, run_two_speed). With a budget counted in CARLA runs:

1. Screen: `screen` Latin-hypercube points (McKay, Beckman & Conover 1979) plus `refine` points of an existing
   optimiser (core.search: the cross-entropy method by default) on the cheap objective; every point's cheap value is
   kept (the pool).
2. Each round asks for a batch: its CONTROL share (default 20 %, at least one run of a batch of two or more) are
   points drawn uniformly at random, independent of the screen; the rest are the pool's most promising points,
   greedily, no two closer than 0.04·√d in unit coordinates (and none on a point already run).
3. After each batch the low → high relationship is refitted from every confirmed pair: Spearman's rank correlation
   (average ranks), an isotonic regression of the CARLA value on the cheap one (pool adjacent violators: Ayer et al.
   1955; Barlow et al. 1972), and a Gaussian process (core.search's Matérn 5/2) on the residual over the parameters —
   a monotone map alone cannot change the order, the residual model can. The pool is re-ranked by the probability
   that the corrected value reaches the threshold, Φ((iso(low) + μ_GP − t) / σ_GP).
4. Trust. With at least `min_pairs` pairs, a rank correlation below `min_spearman` (default 0.5) means the screen is
   NOT TRUSTED for this scenario: the rest of the budget goes to a plain searcher (core.search, default bayes-pof)
   told every confirmed run. The verdict is final for the study (resume with --retrust to judge again).

THE FIDELITY GAP (fidelity()). Reported from the confirmed pairs: Spearman ρ over all pairs (the selected points
stretch the range, so it reads higher than on a random sample) and over the random controls alone (unbiased, few
points); the residual of the isotonic fit (in sample); and the FALSE-NEGATIVE RISK — of the random controls the
screen predicted BELOW the threshold at the time they were drawn (the prediction is recorded before the run, so the
estimate is not fitted to its own answer), the share that failed in CARLA, with the one-sided Clopper–Pearson 95 %
bound (core.failrate.binom_upper). It estimates P(fails in CARLA | the screen would have dropped it) for points
uniform over the search space; with n dropped controls and none failing the bound is 1 − 0.05^(1/n) ≈ 3/n, so 8
controls bound it at 31 %, 30 at 9.5 %: a small control sample cannot show a small risk, and the output says n.

Output: an ordinary study (sim.studio.Study: results.jsonl rows as any search's, each with a "screen" record — the
cheap value, its role and round; before 4.2 that record was called "fidelity" and is still read so) plus
search-prescreen.json in the search result's format with the fidelity gap as "gap", and prescreen.json
(PRESCREEN_SCHEMA: the pairs, the fit, the trust word, the throughput measured in this run). Everything that reads a
search reads it.

FIDELITY (core.fidelity, 4.2). The cheap value is "kinematic" (`screen_fidelity`); the result — every failure listed,
every pair's high side — is a CARLA run whose run.json says what it was, and prescreen.json's top-level "fidelity" is
core.fidelity.combine over those confirmed runs ("fake" on the fake CARLA, "mixed" when runs of different fidelity
were pooled). Each pair and each failure carries its run's level.

WHY CORRELATION IS LOW (diagnose()). When ρ is below the trust threshold, or trust is not "trusted", diagnose() reads
the pairs for the usual causes, each a measured fact: the pairs whose ranks disagree most (cheap rank against CARLA
rank), saturation and ties in the cheap value (a screen that gives many candidates the same value cannot order them),
fewer pairs than the trust rule needs, a scenario parameter along which the disagreement concentrates (Spearman of
|rank difference| against each numeric parameter), and a camera too small for the stack to see what the kinematic
model assumes it sees. They are leads, not proof: a Spearman over a dozen pairs is itself noisy.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import criticality as CR
from . import fidelity as FD
from . import scenarios as SC
from . import search as S

PRESCREEN_SCHEMA = 'falcon.prescreen/1'
LANE_WIDTH_M = 3.5                 # assumed: the lane width of the straight road (CARLA's town lanes are 3.5 m)
NOMINAL_EGO_M = (4.6, 2.0)         # assumed: the director's nominal car, not the ego blueprint's box
PATH_STEP_M = 1.0                  # workers.director.STEP_M: a path's points, to which a projection snaps
ROUTE_STEP_M, ROUTE_POINTS, CORRIDOR_M = 2.0, 25, 1.8      # director.drive_ego / director.lead
PET_MAX_S = 6.0                    # a longer post-encroachment time is taken as absent (module doc)
DRIVERS = ('follower', 'idm', 'none')
TRUST = ('trusted', 'not trusted', 'not judged yet')
DIVERSITY = 0.04                   # × √d: the smallest unit-cube distance between two confirmed points
ROW_CHUNK = 120000                 # rows (scenario × step) per criticality call: bounds the temporaries' memory
SCREEN_FIDELITY = 'kinematic'      # core.fidelity: what the cheap value is


class Unsupported(ValueError):
    """A scenario the fast simulator cannot run; str(e) is the reason."""


# ================================================================================================ array back ends
class _NP:
    """numpy as the rollout's array library (the default)."""
    name = 'cpu'

    def __init__(self):
        self.xp = np

    def arr(self, v, n):
        a = np.asarray(v, dtype=float)
        return np.broadcast_to(a, (n,)).copy() if a.ndim == 0 else a.astype(float).copy()

    def full(self, n, v, dtype=float):
        return np.full(n, v, dtype=dtype)

    def where(self, c, a, b):
        return np.where(c, a, b)

    def mn(self, a, b):
        return np.minimum(a, b)

    def mx(self, a, b):
        return np.maximum(a, b)

    def stack(self, seq):
        return np.stack(seq, axis=1)

    def numpy(self, a):
        return np.asarray(a)

    def __getattr__(self, k):                       # sqrt, cos, sin, tan, arctan, floor, abs
        return getattr(np, k)


class _Torch:
    """torch tensors (float64) on a device: the same rollout, for `device="cuda"`."""

    def __init__(self, device: str):
        import torch
        self.t, self.dev, self.name = torch, torch.device(device), str(device)

    def arr(self, v, n):
        a = self.t.as_tensor(np.asarray(v, dtype=float), dtype=self.t.float64, device=self.dev)
        return a.expand(n).clone() if a.ndim == 0 else a.clone()

    def full(self, n, v, dtype=float):
        return self.t.full((n,), v, dtype=self.t.bool if dtype is bool else self.t.float64, device=self.dev)

    def _t(self, a, like):
        return a if self.t.is_tensor(a) else self.t.as_tensor(a, dtype=like.dtype, device=self.dev)

    def where(self, c, a, b):
        ref = a if self.t.is_tensor(a) else b
        return self.t.where(c, self._t(a, ref), self._t(b, ref))

    def mn(self, a, b):
        ref = a if self.t.is_tensor(a) else b
        return self.t.minimum(self._t(a, ref), self._t(b, ref))

    def mx(self, a, b):
        ref = a if self.t.is_tensor(a) else b
        return self.t.maximum(self._t(a, ref), self._t(b, ref))

    def stack(self, seq):
        return self.t.stack(seq, dim=1)

    def numpy(self, a):
        return a.detach().cpu().numpy()

    def __getattr__(self, k):
        return getattr(self.t, {'arctan': 'atan'}.get(k, k))


def backend(device: str = 'cpu'):
    """The rollout's array library: 'cpu' (numpy) or a torch device ('cuda', 'cuda:0', 'torch-cpu'). ValueError when
    torch or the device is not there."""
    if device in ('', 'cpu', 'numpy'):
        return _NP()
    try:
        import torch
    except ImportError:
        raise ValueError(f'device {device!r} needs torch (the perception extra); use cpu') from None
    dev = 'cpu' if device == 'torch-cpu' else device
    if dev.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is not available here; use cpu')
    try:
        return _Torch(dev)
    except RuntimeError as e:
        raise ValueError(f'device {device!r}: {str(e).splitlines()[0][:120]}') from None


# ================================================================================================ the scenario's plan
class _Sizes:
    """workers.director's own nominal sizes and stand-ins (Director.size_of, Director.asset), without a server."""

    def __init__(self):
        from ..workers.director import Director
        self._d = Director
        self._assets = {}

    def asset(self, kind):
        return self._d.asset(self, kind)

    def size_of(self, kind) -> tuple:
        return self._d.size_of(self, kind)


def _kind_info(kind: str, sizes: _Sizes) -> dict:
    from ..workers.director import PROXIES, _class_of
    cls = _class_of(kind)
    spec = {}
    if cls is not None:
        from . import assets as AS
        spec = AS.CLASSES.get(cls) or {}
    ln, wd = sizes.size_of(kind)
    return {'length': float(ln), 'width': float(wd), 'walker': kind in ('pedestrian', 'child'),
            'prop': kind in PROXIES, 'top': (float(spec['top_kmh']) / 3.6) if spec.get('top_kmh') is not None else None,
            'max_accel': spec.get('accel_mps2')}


def unsupported(scn: dict) -> str | None:
    """Why the fast simulator cannot run a scenario (None: it can). Structure only: no parameter values needed."""
    ego = scn.get('ego') or {}
    if ego.get('route', 'straight') != 'straight':
        return f'the ego\'s route is {ego.get("route")!r}: the fast simulator has one straight road, no junction'
    if ego.get('driver', 'follower') not in ('follower', 'none'):
        return (f'the ego\'s driver is {ego.get("driver")!r}: only the director\'s lane follower (or "none") is '
                'modelled')
    bad = sorted(set(scn.get('needs') or ()) - {'adjacent_same'})
    if bad:
        return f'it needs {", ".join(bad)}: a road feature the straight road does not have'
    if scn.get('route_xy') or scn.get('start'):
        return 'it is a recorded event placed on a map (start / route_xy): the fast simulator has no map'
    ids = set()
    for a in scn.get('actors') or ():
        ids.add(a.get('id'))
        if a.get('track'):
            return f'actor {a.get("id")} replays a recorded track'
        if a.get('driver') == 'levelk':
            return f'actor {a.get("id")} is a level-k driver (it plans; not scripted)'
        lane = (a.get('place') or {}).get('lane', 'ego')
        if lane == 'opposite':
            return f'actor {a.get("id")} is in the opposite lanes: where they are depends on the map'
        if lane not in ('ego', 'adjacent', 'line', 'kerb', 'sidewalk', 'far_sidewalk'):
            return f'actor {a.get("id")}: unknown lane {lane!r}'
        if a.get('heading', 'lane') not in ('lane', 'against', 'across'):
            return f'actor {a.get("id")}: unknown heading {a.get("heading")!r}'
        if a.get('heading') == 'against' and lane != 'ego':
            return f'actor {a.get("id")} drives against the traffic outside the ego\'s lane'
    for k, ev in enumerate(scn.get('timeline') or ()):
        t = (ev.get('trigger') or {}).get('type')
        if t not in SC.TRIGGERS:
            return f'timeline {k}: unknown trigger {t!r}'
        for act in ev.get('actions') or ():
            if act.get('do') not in SC.ACTIONS:
                return f'timeline {k}: unknown action {act.get("do")!r}'
            if act.get('actor') not in ids:
                return f'timeline {k}: unknown actor {act.get("actor")!r}'
            if act.get('do') == 'lane_change':
                a = next(x for x in scn['actors'] if x['id'] == act['actor'])
                if a.get('heading') == 'across':
                    return f'timeline {k}: a lane change of {act["actor"]}, which moves across the road'
    return None


def coverage() -> dict:
    """{built-in scenario: None (covered) | the reason it is refused}."""
    return {k: unsupported(SC.builtin(k)) for k in SC.LIBRARY}


# ================================================================================================ the rollout
def _controller_constants() -> dict:
    """The constants of the controllers the kinematic scenario mode runs, read from their classes."""
    from ..plugins.lane_follower import LaneFollower
    from ..workers.director import Director
    lf = LaneFollower()
    return {'accel': float(Director.ACCEL), 'brake': float(Director.BRAKE), 'steer': math.radians(Director.STEER_DEG),
            'gap_s': float(lf.gap_s), 'min_gap': float(lf.min_gap)}


def idm_accel(B, v, v_des, gap, dv, a_max: float):
    """core.levelk.idm for the rollout's array library (the same formula and constants); gap inf: no lead."""
    from . import levelk as LK
    s_star = LK.S0 + B.mx(v * LK.IDM_T + v * dv / (2.0 * math.sqrt(a_max * LK.IDM_B)), 0.0)
    free = 1.0 - (B.mx(v, 0.0) / B.mx(v_des, 0.1)) ** 4
    return a_max * (free - B.where(gap < math.inf, (s_star / B.mx(gap, 0.1)) ** 2, 0.0))


def rollout(scn: dict, values: dict, n: int, fps: float = 20.0, driver: str = 'follower', device: str = 'cpu',
            lane_width: float = LANE_WIDTH_M, ego_size=NOMINAL_EGO_M, rht: bool = True) -> dict:
    """Run n parameter sets of a scenario at once. values: {parameter: an (n,) array, or one value for all (a choice
    parameter must be one value: LowFidelity groups a batch by its choices)}. Returns numpy arrays:
    {'t': (T,), 'ego': (n, T, 8), 'actors': {id: (n, T, 8)}} — rows of core.criticality's columns (t, x = s, y,
    yaw°, vx, vy, length, width) AFTER each step, as the director records them — and 'fired': {timeline index: (n,)
    time or nan}. Unsupported when the scenario cannot be run."""
    why = unsupported(scn)
    if why:
        raise Unsupported(why)
    if driver not in DRIVERS:
        raise ValueError(f'driver {driver!r}: one of {", ".join(DRIVERS)}')
    B = backend(device)
    K = _controller_constants()
    sizes = _Sizes()
    conc = SC.concrete(scn, values)
    dt = 1.0 / float(fps)
    dur = conc.get('duration_s', 12.0)
    steps = int(round(float(np.max(np.asarray(dur, dtype=float))) * fps))
    ks = 1.0 if rht else -1.0
    W = float(lane_width)
    e_half = float(ego_size[0]) / 2.0
    ego_cfg = conc.get('ego') or {}
    ego_driver = ego_cfg.get('driver', 'follower')
    if ego_driver == 'none':
        driver = 'none'
    v0 = B.arr(ego_cfg.get('speed_kmh', 30.0), n) / 3.6
    cruise = v0 + 1e-6 / 3.6
    es, ev, ei = B.full(n, 0.0), v0.clone() if hasattr(v0, 'clone') else v0.copy(), B.full(n, 0.0)

    acts = []
    for a in conc.get('actors') or ():
        info = _kind_info(str(a['kind']), sizes)
        place = a.get('place') or {}
        lane, heading = place.get('lane', 'ego'), a.get('heading', 'lane')
        ahead, lateral = B.arr(place.get('ahead_m', 0.0), n), B.arr(place.get('lateral_m', 0.0), n)
        y_base = {'adjacent': -ks * W}.get(lane, 0.0)
        if lane == 'line':
            off = -ks * W / 2.0 + lateral
        elif lane == 'kerb':
            off = ks * (W / 2.0 + info['width'] / 2.0 + 0.2 - lateral)
        elif lane == 'sidewalk':
            off = ks * (W / 2.0 + 2.0) + lateral
        elif lane == 'far_sidewalk':
            off = -ks * (W / 2.0 + W + 2.0) + lateral
        else:
            off = ks * lateral
        ox, oy = ahead, y_base + off                 # where it stands at the start (road coordinates)
        if heading == 'across':
            u, lat, plen = (0.0, -ks), B.full(n, 0.0), 29.0              # walks from its side across the road
        elif heading == 'against':
            u, lat, plen = (-1.0, 0.0), -off, 200.0                      # its right is the ego lane's left
            oy = B.full(n, y_base)
        else:
            u, lat, plen = (1.0, 0.0), off, 300.0
            oy = B.full(n, y_base)
        nx, ny = -u[1], u[0]                         # the path's right (CARLA: (−sin yaw, cos yaw))
        moving = a.get('start', 'moving') == 'moving'
        sp = B.arr(a.get('speed_kmh', 0.0), n) / 3.6 if moving else B.full(n, 0.0)
        if info['top'] is not None:
            sp = B.mn(sp, info['top'])
        st = dict(info, id=a['id'], u=u, nrm=(nx, ny), plen=plen, ox=ox, oy=oy, lat=lat, lat_from=lat, lat_to=lat,
                  lat_t0=B.full(n, 0.0), lat_dur=B.full(n, 0.0), px=ox + nx * lat, py=oy + ny * lat,
                  yaw=B.full(n, math.atan2(u[1], u[0])), v=sp, set=sp, target=sp, accel=B.full(n, 3.0),
                  i_err=B.full(n, 0.0), walking=B.full(n, False, bool), walk_speed=B.full(n, 0.0),
                  prog=B.full(n, 0.0), half=info['length'] / 2.0, y_base=y_base)
        acts.append(st)
    by_id = {a['id']: a for a in acts}
    timeline = conc.get('timeline') or []
    fired = [B.full(n, False, bool) for _ in timeline]
    fired_t = [B.full(n, float('nan')) for _ in timeline]
    rec_e, rec_a = [], {a['id']: [] for a in acts}

    def apply(act, m, t):
        st = by_id[act['actor']]
        do = act['do']
        if do == 'speed':
            st['target'] = B.where(m, B.arr(act.get('kmh', 30.0), n) / 3.6, st['target'])
            st['accel'] = B.where(m, B.arr(act.get('accel', 3.0), n), st['accel'])
        elif do in ('brake', 'stop'):
            st['target'] = B.where(m, 0.0, st['target'])
            st['accel'] = B.where(m, B.arr(act.get('decel', 6.0), n) if do == 'brake' else 6.0, st['accel'])
        elif do == 'go':
            st['target'] = B.where(m, B.mx(st['target'], 5.0), st['target'])
        elif do == 'walk':
            st['walking'] = st['walking'] | m
            st['walk_speed'] = B.where(m, B.arr(act.get('speed_mps', 1.4), n), st['walk_speed'])
        elif do == 'lane_change':
            to_ego = (0.0 - st['y_base']) * st['nrm'][1]
            tgt = {'ego': to_ego, 'kerb': to_ego + ks * W, 'adjacent': to_ego - ks * W}.get(act.get('to', 'ego'),
                                                                                             to_ego)
            st['lat_from'] = B.where(m, st['lat'], st['lat_from'])
            st['lat_to'] = B.where(m, tgt, st['lat_to'])
            st['lat_t0'] = B.where(m, t, st['lat_t0'])
            st['lat_dur'] = B.where(m, B.arr(act.get('duration_s', 2.0), n), st['lat_dur'])
        # open_door: no kinematic effect (module doc)

    for k in range(steps):
        t = k * dt
        # ---- the timeline, on the state the director reads (the frame just ticked)
        for j, entry in enumerate(timeline):
            trig = entry.get('trigger') or {}
            if trig.get('type') == 'time':
                go = B.full(n, True, bool) & (t >= B.arr(trig['t_s'], n))
            else:
                st = by_id[trig['actor']]
                if trig['type'] == 'distance':
                    go = B.sqrt((st['px'] - es) ** 2 + st['py'] ** 2) < B.arr(trig['lt_m'], n)
                else:
                    go = (st['px'] - es) - st['half'] - e_half > B.arr(trig['gt_m'], n)
            new = go & ~fired[j]
            fired[j] = fired[j] | go
            fired_t[j] = B.where(new, round(t, 3), fired_t[j])
            for act in entry.get('actions') or ():
                apply(act, new, t)
        # ---- the ego's lead (director.lead): the first route point within the corridor of an actor, on the
        #      positions of the frame just ticked
        lead_d, lead_v = B.full(n, math.inf), B.full(n, 0.0)
        if driver != 'none':
            e_snap = B.floor(es / PATH_STEP_M + 0.5) * PATH_STEP_M
            for st in acts:
                r2 = CORRIDOR_M ** 2 - st['py'] ** 2
                r = B.sqrt(B.mx(r2, 0.0))
                ds = st['px'] - e_snap
                m_ = B.mx(B.floor((ds - r) / ROUTE_STEP_M) + 1.0, 1.0)
                hit = (r2 > 0) & (m_ <= ROUTE_POINTS) & (B.abs(ds - ROUTE_STEP_M * m_) < r)
                d = B.mx(ROUTE_STEP_M * m_ - e_half - st['half'], 0.0)
                better = hit & (d < lead_d)
                lead_d = B.where(better, d, lead_d)
                lead_v = B.where(better, st['v'], lead_v)
        # ---- the actors (director.drive_actor), from the same frame
        for st in acts:
            if st['walker']:
                sp = B.where(st['walking'], st['walk_speed'], 0.0)
                st['px'] = st['px'] + st['u'][0] * sp * dt
                st['py'] = st['py'] + st['u'][1] * sp * dt
                st['v'] = sp
                continue
            if st['top'] is not None:
                st['target'] = B.mn(st['target'], st['top'])
            up = st['accel'] if not st['max_accel'] else B.mn(st['accel'], float(st['max_accel']))
            st['set'] = B.where(st['set'] < st['target'], B.mn(st['target'], st['set'] + up * dt),
                                B.where(st['set'] > st['target'], B.mx(st['target'], st['set'] - st['accel'] * dt),
                                        st['set']))
            f = B.mn(B.mx((t - st['lat_t0']) / B.mx(st['lat_dur'], 1e-9), 0.0), 1.0)
            st['lat'] = B.where(st['lat_dur'] > 0, st['lat_from'] + (st['lat_to'] - st['lat_from']) * f, st['lat'])
            ux, uy = st['u']
            nx, ny = st['nrm']
            if st['prop']:                           # director.prop_step: along its path at the set speed
                st['prog'] = st['prog'] + st['set'] * dt
                a_ = B.mn(st['prog'], st['plen'])
                st['px'], st['py'] = st['ox'] + ux * a_ + nx * st['lat'], st['oy'] + uy * a_ + ny * st['lat']
                st['v'] = st['set']
                continue
            # director.follow: pure pursuit to the path shifted by the lateral offset, and the speed controller
            vel = st['v']
            a_ = (st['px'] - st['ox']) * ux + (st['py'] - st['oy']) * uy
            a_ = B.mn(B.mx(B.floor(a_ / PATH_STEP_M + 0.5) * PATH_STEP_M, 0.0), st['plen'])
            la = B.mx(0.8 * vel, 4.0)
            s_t = B.mn(a_ + la, st['plen'])
            tx, ty = st['ox'] + ux * s_t + nx * st['lat'], st['oy'] + uy * s_t + ny * st['lat']
            dx, dy = tx - st['px'], ty - st['py']
            c, s_ = B.cos(st['yaw']), B.sin(st['yaw'])
            fx, fy = c * dx + s_ * dy, -s_ * dx + c * dy
            Lw = 2.0 * max(st['half'], 0.6) * 0.6
            steer = B.arctan(Lw * 2.0 * fy / B.mx(fx * fx + fy * fy, 1e-6)) / K['steer']
            steer = B.mn(B.mx(steer, -1.0), 1.0)
            err = st['set'] - vel
            st['i_err'] = B.mn(B.mx(st['i_err'] + err * dt, -5.0), 5.0)
            uu = 0.5 * err + 0.08 * st['i_err']
            hold = (st['set'] <= 0.01) & (vel < 0.3)
            thr = B.where(uu >= 0, B.mn(uu, 1.0), 0.0)
            brk = B.where(uu >= 0, 0.0, B.mn(-uu, 1.0))
            v1 = B.where(hold, 0.0, B.mx(vel + (K['accel'] * thr - K['brake'] * brk) * dt, 0.0))
            steer = B.where(hold, 0.0, steer)
            wb = max(2.0 * st['half'] * 0.6, 0.8)
            yaw2 = st['yaw'] + v1 * B.tan(steer * K['steer']) / wb * dt
            mid = (st['yaw'] + yaw2) / 2.0
            st['px'], st['py'] = st['px'] + B.cos(mid) * v1 * dt, st['py'] + B.sin(mid) * v1 * dt
            st['yaw'], st['v'] = yaw2, v1
        # ---- the ego (director.drive_ego + LaneFollower.act + kin_step; straight road: no steering)
        if driver == 'idm':
            acc = idm_accel(B, ev, cruise, lead_d, ev - lead_v, K['accel'])
            ev = B.mx(ev + B.mn(B.mx(acc, -K['brake']), K['accel']) * dt, 0.0)
        else:
            safe = K['min_gap'] + K['gap_s'] * ev
            close = lead_d < safe
            limit = B.where(close, B.mn(cruise, B.mx(lead_v * B.where(close, lead_d, 0.0) / safe, 0.0)), cruise)
            stop = close & (limit < 0.3) & (lead_d < K['min_gap'] + 2.0)
            err = limit - ev
            ei = B.where(stop, 0.0, B.mn(B.mx(ei + err * dt, -5.0), 5.0))
            uu = 0.35 * err + 0.05 * ei
            thr = B.where(stop, 0.0, B.where(uu >= 0, B.mn(uu, 0.75), 0.0))
            brk = B.where(stop, 1.0, B.where(uu >= 0, 0.0, B.mn(-uu, 1.0)))
            ev = B.mx(ev + (K['accel'] * thr - K['brake'] * brk) * dt, 0.0)
        es = es + ev * dt
        # ---- record (after the step, as the director does after the tick)
        zero = B.full(n, 0.0)
        rec_e.append(B.stack([es, zero, zero, ev, zero]))
        for st in acts:
            c, s_ = B.cos(st['yaw']), B.sin(st['yaw'])
            rec_a[st['id']].append(B.stack([st['px'], st['py'], st['yaw'] * (180.0 / math.pi), st['v'] * c,
                                            st['v'] * s_]))
    T = np.arange(1, steps + 1) * dt

    def rows(seq, ln, wd):
        out = np.empty((n, steps, 8))
        out[:, :, 0] = T[None, :]
        if seq:
            out[:, :, 1:6] = B.numpy(B.stack(seq))                     # (n, T, 5)
        out[:, :, 6], out[:, :, 7] = ln, wd
        return out
    return {'t': T, 'ego': rows(rec_e, float(ego_size[0]), float(ego_size[1])),
            'actors': {st['id']: rows(rec_a[st['id']], st['length'], st['width']) for st in acts},
            'fired': {j: B.numpy(ft) for j, ft in enumerate(fired_t)}, 'device': B.name, 'driver': driver}


# ================================================================================================ the criticality
def _aabb(Tr: np.ndarray) -> tuple:
    """Half extents (x, y) of the axis-aligned boxes around (…, 8) oriented boxes."""
    r = np.radians(Tr[..., CR.YAW])
    c, s = np.abs(np.cos(r)), np.abs(np.sin(r))
    return (c * Tr[..., CR.LEN] + s * Tr[..., CR.WID]) / 2.0, (s * Tr[..., CR.LEN] + c * Tr[..., CR.WID]) / 2.0


def _pet_batch(E: np.ndarray, A: np.ndarray, dt: float, max_s: float = PET_MAX_S) -> np.ndarray:
    """The post-encroachment time per scenario (inf: none within max_s) of (n, T, 8) trajectories: the smallest
    |t_i − t_j| over the footprint pairs that overlap — core.criticality.pet_rows' definition — searched by
    increasing lag. A pair is tested with core.criticality's overlap test only when the boxes' axis-aligned bounds
    meet (necessary for an overlap; for boxes along the road it is the overlap)."""
    n, T, _ = E.shape
    out = np.full(n, np.inf)
    ex, ey, ax, ay = E[:, :, CR.X], E[:, :, CR.Y], A[:, :, CR.X], A[:, :, CR.Y]
    ehx, ehy = _aabb(E)
    ahx, ahy = _aabb(A)
    # the swept envelopes must meet at all
    act = np.nonzero(((ex - ehx).min(1) <= (ax + ahx).max(1)) & ((ax - ahx).min(1) <= (ex + ehx).max(1)) &
                     ((ey - ehy).min(1) <= (ay + ahy).max(1)) & ((ay - ahy).min(1) <= (ey + ehy).max(1)))[0]
    for lag in range(0, min(T, int(max_s / dt + 1e-9) + 1)):
        if not len(act):
            break
        done = np.zeros(len(act), bool)
        for flip in ((False,) if lag == 0 else (False, True)):
            ie = slice(lag, T) if flip else slice(0, T - lag)
            ia = slice(0, T - lag) if flip else slice(lag, T)
            near = (np.abs(ex[act, ie] - ax[act, ia]) <= ehx[act, ie] + ahx[act, ia] + CR.EPS) & \
                   (np.abs(ey[act, ie] - ay[act, ia]) <= ehy[act, ie] + ahy[act, ia] + CR.EPS)
            b, i = np.nonzero(near)
            if not len(b):
                continue
            rows_e = E[act[b], i + (lag if flip else 0)]
            rows_a = A[act[b], i + (0 if flip else lag)]
            hit = CR._overlap(CR._corners_of(rows_e), CR._corners_of(rows_a))
            done[np.unique(b[hit])] = True
        out[act[done]] = lag * dt
        act = act[~done]
    return out


def _measures(E: np.ndarray, A: np.ndarray, horizon: float) -> tuple:
    """(min TTC, max finite DRAC, min distance, collided) per scenario of (n, T, 8) trajectories, by
    core.criticality's row functions behind three exact prefilters (module doc)."""
    n, T, _ = E.shape
    e, a = E.reshape(-1, 8), A.reshape(-1, 8)
    re = 0.5 * np.hypot(e[:, CR.LEN], e[:, CR.WID])
    ra = 0.5 * np.hypot(a[:, CR.LEN], a[:, CR.WID])
    dx, dy = a[:, CR.X] - e[:, CR.X], a[:, CR.Y] - e[:, CR.Y]
    dc = np.hypot(dx, dy)
    # TTC: within the horizon the centres cannot close more than |v_rel| · horizon
    vrel = np.hypot(a[:, CR.VX] - e[:, CR.VX], a[:, CR.VY] - e[:, CR.VY])
    ttc = np.full(len(e), np.inf)
    m = dc <= vrel * horizon + re + ra + 1e-6
    if m.any():
        ttc[m] = CR.ttc_rows(e[m], a[m], horizon)
    # DRAC: an actor whose circle is outside the ego's forward corridor is not in its path
    r = np.radians(e[:, CR.YAW])
    lat = -np.sin(r) * dx + np.cos(r) * dy
    lon = np.cos(r) * dx + np.sin(r) * dy
    drac = np.zeros(len(e))
    m = (np.abs(lat) <= e[:, CR.WID] / 2.0 + ra + 1e-6) & (lon >= -(e[:, CR.LEN] / 2.0 + ra + 1e-6))
    if m.any():
        drac[m] = CR.drac_rows(e[m], a[m])
    # distance: only samples whose lower bound (centres − radii) is under the scenario's smallest upper bound
    ub = dc.reshape(n, T).min(1)
    m = (dc - re - ra) <= np.repeat(ub, T) + 1e-9
    dist = np.full(len(e), np.inf)
    dist[m] = CR._distance(CR._corners_of(e[m]), CR._corners_of(a[m]))
    dist = dist.reshape(n, T)
    fin = np.where(np.isfinite(drac), drac, -1.0).reshape(n, T).max(1)
    return ttc.reshape(n, T).min(1), np.maximum(fin, 0.0), dist.min(1), (dist <= 0.0).any(1)


def criticality(traj: dict, horizon: float = 10.0) -> dict:
    """core.criticality's scalar for every scenario of a rollout: {'criticality': (n,), 'collision': (n,) bool,
    'actors': {id: {'min_ttc_s', 'pet_s', 'max_drac_mps2', 'min_distance_m', 'collision', 'criticality'}}} — an
    absent measure is inf (TTC, PET). The terms and the score are core.criticality.criticality_terms and
    criticality_score, per actor and overall as summarise() combines them."""
    E = traj['ego']
    n, T, _ = E.shape
    dt = float(traj['t'][1] - traj['t'][0]) if T > 1 else 0.05
    per = {}
    crit = np.zeros(n)
    coll = np.zeros(n, bool)
    chunk = max(1, ROW_CHUNK // max(T, 1))
    for aid, A in traj['actors'].items():
        ttc, drac, dist, hit, pet = (np.empty(n) for _ in range(5))
        hit = np.zeros(n, bool)
        for s0 in range(0, n, chunk):
            sl = slice(s0, s0 + chunk)
            ttc[sl], drac[sl], dist[sl], hit[sl] = _measures(E[sl], A[sl], horizon)
            pet[sl] = _pet_batch(E[sl], A[sl], dt)
        c = np.empty(n)
        for i in range(n):
            terms = CR.criticality_terms(ttc[i], pet[i], drac[i], dist[i])
            c[i] = CR.criticality_score(terms, bool(hit[i]))
        per[aid] = {'min_ttc_s': ttc, 'pet_s': pet, 'max_drac_mps2': drac, 'min_distance_m': dist, 'collision': hit,
                    'criticality': c}
        crit = np.maximum(crit, c)
        coll |= hit
    crit = np.where(coll, 1.0, crit)
    return {'criticality': crit, 'collision': coll, 'actors': per}


class LowFidelity:
    """The cheap objective of a scenario over a search space: __call__(list of parameter dicts) -> (n,) criticality.
    Dimensions the fast simulator does not have ("weather.<field>") are ignored — said in .ignored. Counts its
    evaluations and their time (throughput())."""

    def __init__(self, scn: dict, space: list | None = None, fps: float = 20.0, driver: str = 'follower',
                 device: str = 'cpu', batch: int = 4096):
        why = unsupported(scn)
        if why:
            raise Unsupported(f'{scn.get("id") or scn.get("name")}: {why}')
        self.scn, self.fps, self.driver, self.device, self.batch = scn, float(fps), driver, device, int(batch)
        backend(device)                                    # refuse an unusable device now
        self.space = S.normalise_space(space if space is not None else SC.search_space(scn))
        names = set(scn.get('params') or {})
        self.ignored = [p['name'] for p in self.space if p['name'] not in names]
        self.fixed = {k: p['value'] for k, p in (scn.get('params') or {}).items() if p.get('dist') == 'fixed'}
        self.n, self.seconds = 0, 0.0

    def __call__(self, params_list) -> np.ndarray:
        params_list = list(params_list)
        out = np.empty(len(params_list))
        names = list(self.scn.get('params') or {})
        missing = [k for k in names if k not in self.fixed and params_list and k not in params_list[0]]
        if missing:
            raise ValueError(f'the fast simulator needs every parameter of the scenario; missing: '
                             f'{", ".join(missing)}')
        groups = {}
        for i, p in enumerate(params_list):
            key = tuple((k, p[k]) for k in names if k in p and isinstance(p[k], str))
            groups.setdefault(key, []).append(i)
        t0 = time.perf_counter()
        for key, idx in groups.items():
            for s0 in range(0, len(idx), self.batch):
                ii = idx[s0:s0 + self.batch]
                vals = dict(self.fixed)
                vals.update(dict(key))
                for k in names:
                    if k not in vals:
                        vals[k] = np.array([float(params_list[i][k]) for i in ii])
                tr = rollout(self.scn, vals, len(ii), self.fps, self.driver, self.device)
                out[ii] = criticality(tr)['criticality']
        self.n += len(params_list)
        self.seconds += time.perf_counter() - t0
        return out

    def throughput(self) -> dict:
        """What this object measured: evaluations, seconds, evaluations per second, where."""
        return {'evaluations': self.n, 'seconds': round(self.seconds, 3), 'device': self.device,
                'per_s': round(self.n / self.seconds, 1) if self.seconds > 0 else None}


# ================================================================================================ the statistics
def lhs(n: int, d: int, rng) -> np.ndarray:
    """A Latin hypercube sample of the unit cube (McKay, Beckman & Conover 1979): each dimension's n strata hold
    one point each, at a uniform place in its stratum, the strata paired at random."""
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def ranks(v) -> np.ndarray:
    """Average ranks (ties share the mean of their places), 1-based."""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind='stable')
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(x, y) -> float | None:
    """Spearman's rank correlation (Pearson's on average ranks); None with fewer than 3 pairs or a constant side."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx <= 0 or sy <= 0:
        return None
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


class Isotonic:
    """The least-squares non-decreasing fit of y on x by pool adjacent violators; predict() is the fitted value on
    a block, interpolates between blocks and is constant outside them."""

    def __init__(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        order = np.argsort(x, kind='stable')
        xs, ys = x[order], y[order]
        vals, wts, lo, hi = [], [], [], []
        for xi, yi in zip(xs, ys):
            vals.append(float(yi))
            wts.append(1.0)
            lo.append(float(xi))
            hi.append(float(xi))
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]
                vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
                wts[-2], hi[-2] = w, hi[-1]
                for seq in (vals, wts, lo, hi):
                    seq.pop()
        kx, ky = [], []
        for a, b, v in zip(lo, hi, vals):                  # a block is flat from its first x to its last
            kx.append(a)
            ky.append(v)
            if b > a:
                kx.append(b)
                ky.append(v)
        self.x, self.y = np.array(kx), np.array(ky)

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not len(self.x):
            return x.copy()
        return np.interp(x, self.x, self.y)

    def knots(self) -> list:
        return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(self.x, self.y)]


# ================================================================================================ the loop
class TwoSpeed:
    """The multi-fidelity proposer (module doc). low: a callable (list of parameter dicts) -> values. ask(n) gives n
    candidates [{'params', 'low', 'role': 'top' | 'control', 'predicted', 'p_fail', 'round'}]; tell(cands, highs)
    reports their CARLA values (None: the run failed). Deterministic in (space, seed, the pairs told)."""

    def __init__(self, space, low, threshold: float = 0.5, seed: int = 0, screen: int = 10000, refine: int = 2000,
                 control_fraction: float = 0.2, min_spearman: float = 0.5, min_pairs: int = 12,
                 optimiser: str = 'cem'):
        self.space = S.normalise_space(space)
        self.dim = len(self.space)
        self.low, self.threshold, self.seed = low, float(threshold), int(seed)
        self.n_screen, self.n_refine = int(screen), int(refine)
        if not 0.0 <= control_fraction < 1.0:
            raise ValueError('the control share must be in [0, 1)')
        self.control_fraction, self.min_spearman, self.min_pairs = float(control_fraction), float(min_spearman), \
            int(min_pairs)
        self.optimiser = optimiser
        self.pairs = []                   # confirmed: {'x', 'params', 'low', 'high', 'role', 'predicted', 'round'}
        self.round = 0
        self.pool_x = self.pool_low = None
        self._model = None
        self.final_trust = None           # 'not trusted' once judged so: it stays (module doc)

    # ---------------------------------------------------------------- the screen
    def screen(self):
        """Screen the space on the cheap objective (step 1); idempotent."""
        if self.pool_x is not None:
            return
        rng = np.random.default_rng([self.seed, 11])
        X = lhs(self.n_screen, self.dim, rng)
        P = [S.to_params(self.space, x) for x in X]
        L = np.asarray(self.low(P), dtype=float)
        if self.n_refine > 0:
            opt = S.make(self.optimiser, self.space, seed=self.seed, threshold=self.threshold)
            pop = max(int(opt.default_batch), 20)
            done = 0
            while done < self.n_refine:
                ask = opt.ask(min(pop, self.n_refine - done))
                vals = np.asarray(self.low(ask), dtype=float)
                opt.tell(ask, [float(v) for v in vals])
                P += ask
                L = np.r_[L, vals]
                done += len(ask)
        self.pool_x = np.array([S.to_unit(self.space, p) for p in P])
        self.pool_params, self.pool_low = P, L

    # ---------------------------------------------------------------- the low → high relationship
    def _fit(self):
        """(isotonic, GP on the residual or None) from the pairs; None when there are fewer than 4."""
        ok = [p for p in self.pairs if p['high'] is not None]
        if len(ok) < 4:
            return None
        lo = np.array([p['low'] for p in ok])
        hi = np.array([p['high'] for p in ok])
        iso = Isotonic(lo, hi)
        res = hi - iso.predict(lo)
        gp = None
        if float(np.ptp(res)) > 1e-9:
            try:
                gp = S.fit_gp(np.array([p['x'] for p in ok]), res)
            except np.linalg.LinAlgError:
                gp = None
        return iso, gp, float(np.sqrt(np.mean(res ** 2)))

    def predict(self, X, low) -> tuple:
        """(the predicted CARLA value, its standard deviation) at unit points X with cheap values low: the cheap
        value itself (sd: none, 0) before there are 4 pairs, then iso(low) + the residual GP's mean."""
        low = np.asarray(low, dtype=float)
        m = self._model
        if m is None:
            return low.copy(), np.zeros(len(low))
        iso, gp, rmse = m
        base = iso.predict(low)
        if gp is None:
            return base, np.full(len(low), max(rmse, 1e-3))
        mu, sd = gp.predict(np.atleast_2d(X))
        return base + mu * gp.ys + gp.ym, np.maximum(sd * gp.ys, 1e-3)

    def trust(self) -> str:
        if self.final_trust:
            return self.final_trust
        ok = [p for p in self.pairs if p['high'] is not None]
        if len(ok) < self.min_pairs:
            return TRUST[2]
        rho = spearman([p['low'] for p in ok], [p['high'] for p in ok])
        if rho is None:
            return TRUST[2]
        return TRUST[0] if rho >= self.min_spearman else TRUST[1]

    # ---------------------------------------------------------------- ask / tell
    def ask(self, n: int) -> list:
        self.screen()
        n = int(n)
        n_ctrl = 0 if n < 2 or self.control_fraction <= 0 else min(max(1, int(round(self.control_fraction * n))),
                                                                   n - 1)
        out = []
        if n_ctrl:                                         # uniform, independent of the screen and of the fit
            rng = np.random.default_rng([self.seed, 13, self.round])
            Xc = rng.random((n_ctrl, self.dim))
            Pc = [S.to_params(self.space, x) for x in Xc]
            Xc = np.array([S.to_unit(self.space, p) for p in Pc])
            Lc = np.asarray(self.low(Pc), dtype=float)
            pred, _sd = self.predict(Xc, Lc)
            for p, x, lo, pr in zip(Pc, Xc, Lc, pred):
                out.append({'params': p, 'x': x.tolist(), 'low': float(lo), 'role': 'control',
                            'predicted': float(pr), 'p_fail': None, 'round': self.round})
        pred, sd = self.predict(self.pool_x, self.pool_low)
        if self._model is None:
            score = pred
            pf = np.full(len(pred), np.nan)
        else:
            pf = S.probability_of_failure(pred, sd, self.threshold)
            score = pf + 1e-6 * pred                       # ties (pf 0 or 1 everywhere): the predicted value
        taken = [np.asarray(p['x']) for p in self.pairs] + [np.asarray(c['x']) for c in out]
        gap = DIVERSITY * math.sqrt(self.dim)
        for i in np.argsort(-score, kind='stable'):
            if len(out) >= n:
                break
            x = self.pool_x[i]
            if any(np.linalg.norm(x - q) < gap for q in taken):
                continue
            taken.append(x)
            out.append({'params': self.pool_params[i], 'x': x.tolist(), 'low': float(self.pool_low[i]),
                        'role': 'top', 'predicted': float(pred[i]),
                        'p_fail': None if math.isnan(pf[i]) else float(pf[i]), 'round': self.round})
        return out

    def tell(self, cands: list, highs) -> None:
        for c, h in zip(cands, highs, strict=True):
            self.pairs.append(dict(c, high=None if h is None else float(h)))
        self.round = max([self.round] + [int(c.get('round', 0)) + 1 for c in cands])
        self._model = self._fit()
        if self.trust() == TRUST[1]:
            self.final_trust = TRUST[1]

    def add_prior(self, params: dict, high: float) -> None:
        """A CARLA evaluation made outside this loop (an earlier search of the study): a pair, never a control."""
        x = S.to_unit(self.space, params)
        lo = float(np.asarray(self.low([params]))[0])
        self.pairs.append({'params': params, 'x': x.tolist(), 'low': lo, 'role': 'prior', 'predicted': None,
                           'p_fail': None, 'round': -1, 'high': float(high)})
        self._model = self._fit()

    # ---------------------------------------------------------------- the report
    def fidelity(self) -> dict:
        """The fidelity gap from the confirmed pairs (module doc)."""
        from .failrate import binom_upper
        ok = [p for p in self.pairs if p['high'] is not None]
        ctrl = [p for p in ok if p['role'] == 'control']
        lo, hi = [p['low'] for p in ok], [p['high'] for p in ok]
        thr = self.threshold
        dropped = [p for p in ctrl if p['predicted'] is not None and p['predicted'] < thr]
        missed = [p for p in dropped if p['high'] >= thr]
        ctrl_fail = [p for p in ctrl if p['high'] >= thr]
        tops = [p for p in ok if p['role'] == 'top']
        m = self._model
        out = {'pairs': len(ok), 'spearman': spearman(lo, hi),
               'spearman_controls': spearman([p['low'] for p in ctrl], [p['high'] for p in ctrl]),
               'controls': len(ctrl), 'min_spearman': self.min_spearman, 'min_pairs': self.min_pairs,
               'trust': self.trust(),
               'isotonic': m[0].knots() if m else None, 'residual_rmse': round(m[2], 6) if m else None,
               'residual_model': 'gp' if m and m[1] is not None else None,
               'selected': {'runs': len(tops), 'failures': len([p for p in tops if p['high'] >= thr])},
               'control_failures': len(ctrl_fail),
               'false_negative': {
                   'dropped_controls': len(dropped), 'failed': len(missed),
                   'rate': (len(missed) / len(dropped)) if dropped else None,
                   'upper95': binom_upper(len(missed), len(dropped), 0.95) if dropped else None,
                   'control_failures': len(ctrl_fail),
                   'control_failures_kept': len([p for p in ctrl_fail if p not in missed]),
                   'about': 'of the random controls the screen predicted below the threshold when they were drawn, '
                            'the share that failed in CARLA; upper95: one-sided Clopper–Pearson'}}
        return out


def run_two_speed(ts: TwoSpeed, high, budget: int, batch: int = 8, fallback: str = 'bayes-pof',
                  on_batch=None) -> dict:
    """Spend `budget` CARLA runs (counting the pairs ts already holds): two-speed rounds while the screen is trusted
    (or not judged yet), then — if it is judged not trusted — a plain searcher told every confirmed run. high(list of
    parameter dicts) -> list of values (None: a run that failed). on_batch(ts, candidates, values) after each batch.
    Returns {'history', 'failures', 'gap' (the fidelity gap: TwoSpeed.fidelity()), 'fallback'}."""
    from ..sim.studio import method_class

    def spent():
        return len(ts.pairs)
    stalled = 0
    while spent() < budget and ts.trust() != TRUST[1]:
        cands = ts.ask(min(batch, budget - spent()))
        if not cands:
            break                                          # the pool has no point left apart from those run
        ys = list(high([c['params'] for c in cands]))
        ts.tell(cands, ys)
        if on_batch:
            on_batch(ts, cands, ys)
        stalled = stalled + 1 if all(y is None for y in ys) else 0
        if stalled >= 2:
            raise RuntimeError('every evaluation of two batches failed')
    fb = None
    if spent() < budget and ts.trust() == TRUST[1]:
        cls = method_class(fallback)
        opts = {'threshold': ts.threshold} if cls in (S.AdaptiveSampling, S.BayesFailure) else {}
        algo = cls(ts.space, seed=ts.seed, **opts)
        past = [p for p in ts.pairs if p['high'] is not None]
        algo.tell([p['params'] for p in past], [p['high'] for p in past])
        rho = ts.fidelity()['spearman']
        fb = {'method': fallback, 'from_run': spent(),
              'why': ('the screen was judged not trusted before' if rho is None or rho >= ts.min_spearman else
                      f'Spearman {rho:.2f} below {ts.min_spearman:g}') + f' ({len(past)} pairs)'}
        failed = 0
        while spent() < budget:
            ask = algo.ask(min(batch, budget - spent()))
            ys = list(high(ask))
            algo.tell(ask, ys)
            lows = np.asarray(ts.low(ask), dtype=float)
            cands = [{'params': p, 'x': S.to_unit(ts.space, p).tolist(), 'low': float(lo), 'role': 'fallback',
                      'predicted': None, 'p_fail': None, 'round': ts.round} for p, lo in zip(ask, lows)]
            for c, h in zip(cands, ys):
                ts.pairs.append(dict(c, high=None if h is None else float(h)))
            ts.round += 1
            if on_batch:
                on_batch(ts, cands, ys)
            failed = failed + 1 if all(y is None for y in ys) else 0
            if failed >= 2:
                raise RuntimeError('every evaluation of two batches failed')
        fb['runs'] = spent() - fb['from_run']
    hist = [{'params': p['params'], 'value': p['high'], 'i': i, 'low': p['low'], 'role': p['role']}
            for i, p in enumerate(q for q in ts.pairs if q['high'] is not None)]
    return {'history': hist, 'failures': S.failures(hist, ts.threshold), 'gap': ts.fidelity(), 'fallback': fb}


# ================================================================================================ the study
def default_name(scenario_id: str) -> str:
    """The study a two-speed search writes by default: the Rare cases page's criticality search of that scenario."""
    from ..sim.studio import search_name
    return search_name(scenario_id)


def read_prescreen(study_dir: str) -> dict | None:
    """prescreen.json of a study folder (None when there is none). A file written before 4.2 holds the fidelity gap
    under "fidelity": it is moved to "gap", and the fidelity level is read from its pairs' runs (unknown when they do
    not say)."""
    try:
        with open(os.path.join(study_dir, 'prescreen.json'), encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get('schema') != PRESCREEN_SCHEMA:
        return None
    if isinstance(d.get('fidelity'), dict):                     # before 4.2: the gap was called "fidelity"
        d['gap'] = d.pop('fidelity')
    if d.get('fidelity') not in FD.LEVELS:
        d['fidelity'] = FD.combine(p.get('fidelity') or (FD.of_run(p['run']) if p.get('run') else 'unknown')
                                   for p in d.get('pairs') or ())
    d.setdefault('screen_fidelity', SCREEN_FIDELITY)
    return d


def screen_record(r: dict) -> dict:
    """A results.jsonl row's two-speed record (cheap value, role, round …): "screen", or before 4.2 "fidelity"."""
    rec = r.get('screen')
    if isinstance(rec, dict):
        return rec
    rec = r.get('fidelity')
    return rec if isinstance(rec, dict) else {}


def run_study(study, budget: int, batch: int = 8, threshold: float = 0.5, screen: int = 10000, refine: int = 2000,
              control_fraction: float = 0.2, min_spearman: float = 0.5, min_pairs: int = 12,
              fallback: str = 'bayes-pof', driver: str = 'follower', device: str = 'cpu', retrust: bool = False,
              progress=None, log=None) -> dict:
    """The two-speed search of a sim.studio.Study (its objective: the criticality): resumes from the study's finished
    evaluations — its own (their "fidelity" record says what they were) and any other search's (pairs, never
    controls). Writes each row's fidelity record, prescreen.json after every batch and search-prescreen.json at the
    end. Unsupported when the fast simulator cannot run the scenario."""
    from ..sim.studio import row_fidelity
    from .provenance import utc_now, write_json
    log = log or (lambda text: None)
    if study.spec['objective'] != 'criticality':
        raise Unsupported(f'the fast simulator scores driving criticality, not the {study.spec["objective"]} '
                          'objective')
    space = study.space()
    low = LowFidelity(study.scn, space, fps=study.fps or 20.0, driver=driver, device=device)
    if low.ignored:
        log(f'the fast simulator ignores {", ".join(low.ignored)} (it has no weather)')
    ts = TwoSpeed(space, low, threshold, study.search_seed, screen, refine, control_fraction, min_spearman, min_pairs)
    t0 = time.perf_counter()
    ts.screen()
    screen_s = time.perf_counter() - t0
    tp = dict(low.throughput(), screen_seconds=round(screen_s, 3))
    log(f'screened {low.n} candidates in {screen_s:.1f} s ({tp["per_s"]} evaluations/s, {device})')
    past = study.current_results()
    prev = read_prescreen(study.dir) or {}
    by_round = {}
    for r in past:
        f = screen_record(r)
        if f.get('role') in ('top', 'control', 'fallback'):
            by_round.setdefault(int(f.get('round', 0)), []).append(r)
        else:
            ts.add_prior(r['values'], r['objective'])
    for k in sorted(by_round):
        rows = by_round[k]
        ts.tell([{'params': r['values'], 'x': S.to_unit(space, r['values']).tolist(), 'low': screen_record(r)['low'],
                  'role': screen_record(r)['role'], 'predicted': screen_record(r).get('predicted'),
                  'p_fail': screen_record(r).get('p_fail'), 'round': k} for r in rows], [r['objective'] for r in rows])
    if (prev.get('trust') == TRUST[1] or ts.final_trust) and not retrust:
        ts.final_trust = TRUST[1]
    elif retrust:
        ts.final_trust = None
    keys, levels = {}, {}                                      # parameters -> run; run -> its fidelity level

    def doc(extra=None):
        ok = [p for p in ts.pairs if p['high'] is not None]
        runs = [keys.get(json.dumps(p['params'], sort_keys=True)) for p in ok]
        lv = [levels.get(r) or (row_fidelity({'run': r}) if r else 'unknown') for r in runs]
        d = {'schema': PRESCREEN_SCHEMA, 'study': study.name, 'scenario': study.scn.get('id'),
             'updated': utc_now(), 'threshold': threshold, 'budget': budget, 'spent': len(ts.pairs),
             'screened': low.n, 'throughput': tp, 'driver': driver, 'objective': study.objective_doc(),
             'options': {'batch': batch, 'screen': screen, 'refine': refine, 'control_fraction': control_fraction,
                         'min_spearman': min_spearman, 'min_pairs': min_pairs, 'fallback': fallback},
             'trust': ts.trust(), 'gap': ts.fidelity(), 'fidelity': FD.combine(lv),
             'screen_fidelity': SCREEN_FIDELITY,
             'pairs': [{'low': round(p['low'], 6), 'high': round(p['high'], 6), 'role': p['role'],
                        'round': p['round'], 'predicted': p['predicted'], 'values': p['params'],
                        'run': r, 'fidelity': f} for p, r, f in zip(ok, runs, lv)],
             'limits': 'the cheap value is a kinematic model on a straight road (core.fastsim): a screen, never a '
                       'result; every failure listed is a CARLA run'}
        d.update(extra or {})
        return d

    for r in past:
        keys[json.dumps(r['values'], sort_keys=True)] = r['run']
        levels[r['run']] = row_fidelity(r)

    def high(batch_params):
        rows = study.evaluate(batch_params, progress)
        return [(study.told(r) if r and r.get('status') == 'ok' else None) for r in rows]

    def on_batch(ts_, cands, ys):
        rows = study.results()
        by = {json.dumps(r['values'], sort_keys=True): r for r in rows}
        for c, y in zip(cands, ys):
            r = by.get(json.dumps(c['params'], sort_keys=True))
            if r is None or y is None:
                continue
            r['screen'] = {'low': round(c['low'], 6), 'role': c['role'], 'round': c['round'],
                           'predicted': c['predicted'], 'p_fail': c['p_fail'], 'driver': driver}
            if isinstance(r.get('fidelity'), dict):              # a row written before 4.2: its level, not the record
                r['fidelity'] = row_fidelity({'run': r['run']})
            keys[json.dumps(c['params'], sort_keys=True)] = r['run']
            levels[r['run']] = row_fidelity(r)
        study._rewrite(rows)
        write_json(os.path.join(study.dir, 'prescreen.json'), doc({'state': 'running'}))
        f = ts_.fidelity()
        rho = f['spearman']
        log(f'{len(ts_.pairs)} of {budget} CARLA runs; Spearman ' + ('—' if rho is None else f'{rho:.2f}') +
            f' over {f["pairs"]} pairs: {f["trust"]}')

    res = run_two_speed(ts, high, budget, batch, fallback, on_batch)
    fails = []
    for h in res['failures']:
        run = keys.get(json.dumps(h['params'], sort_keys=True))
        fails.append({'value': h['value'], 'low': h['low'], 'role': h['role'], 'values': h['params'], 'run': run,
                      'fidelity': levels.get(run) or (row_fidelity({'run': run}) if run else 'unknown')})
    d = doc({'state': 'finished', 'fallback': res['fallback'], 'failures': fails})
    out = {'method': 'prescreen', 'budget': budget, 'history': res['history'], 'failures': res['failures'],
           'threshold': threshold, 'objective': study.objective_doc(), 'space': space,
           'gap': res['gap'], 'fallback': res['fallback'], 'fidelity': d['fidelity'],
           'screen_fidelity': SCREEN_FIDELITY}
    write_json(os.path.join(study.dir, 'search-prescreen.json'), out)
    write_json(os.path.join(study.dir, 'prescreen.json'), d)
    return d


def headline(d: dict) -> str:
    """One paragraph of a prescreen.json for the terminal and the assistant."""
    f = d.get('gap') or {}
    fn = f.get('false_negative') or {}
    rho = f.get('spearman')
    tp = d.get('throughput') or {}
    lines = [f'{d.get("study")}: {d.get("spent")} of {d.get("budget")} CARLA runs, {len(d.get("failures") or ())} '
             f'failures (objective ≥ {d.get("threshold"):g}); screened {d.get("screened")} candidates at '
             f'{tp.get("per_s")} evaluations/s ({tp.get("device")})',
             'fidelity gap: Spearman ' + ('not defined' if rho is None else f'{rho:.2f}') + f' over {f.get("pairs")} '
             f'pairs (needs ≥ {f.get("min_spearman"):g} over ≥ {f.get("min_pairs")}): the screen is {d.get("trust")}',
             f'fidelity: the confirmed runs {FD.label(d.get("fidelity"))}, the screen '
             f'{FD.label(d.get("screen_fidelity") or SCREEN_FIDELITY)}']
    if fn.get('dropped_controls'):
        lines.append(f'false-negative risk: {fn["failed"]} of {fn["dropped_controls"]} random controls the screen '
                     f'would have dropped failed in CARLA (≤ {fn["upper95"]:.0%} at 95 %)')
    else:
        lines.append('false-negative risk: no random control predicted below the threshold yet (no estimate)')
    if d.get('fallback'):
        fb = d['fallback']
        lines.append(f'fell back to {fb["method"]} after {fb["from_run"]} runs: {fb["why"]}')
    return '\n'.join(lines)


# ================================================================================================ why ρ is low
TINY_CAMERA_PX = 640 * 360         # heuristic: a camera below this many pixels resolves a car at 30 m in a few pixels
CONCENTRATED_RHO = 0.5             # heuristic: |Spearman(|rank diff|, parameter)| from which a parameter is named
SATURATED_SHARE = 0.25             # heuristic: this share of the pairs sharing one cheap value is saturation


def _camera_pixels(rig: dict | None) -> list:
    """(name, width, height) of a rig's cameras (core.project's sensor records)."""
    out = []
    for s in (rig or {}).get('sensors') or ():
        if str(s.get('type', '')).startswith('camera') and s.get('width') and s.get('height'):
            out.append((str(s.get('name') or s.get('id') or 'camera'), int(s['width']), int(s['height'])))
    return out


def diagnose(pairs, min_spearman: float = 0.5, min_pairs: int = 12, rig: dict | None = None,
             top: int = 3) -> dict:
    """Why the cheap and the CARLA criticality agree so little (module doc), from the confirmed pairs alone — a pure
    function. pairs: prescreen.json's [{'low', 'high', 'values', 'run', …}]; rig: the study's rig (its cameras'
    sizes). Returns {'spearman', 'pairs', 'low' (ρ below min_spearman, or not computable), 'findings' [{'key', 'text',
    'tone'}] most telling first, 'disagreements' [the `top` pairs with the largest |cheap rank − CARLA rank|: values,
    low, high, rank_low, rank_high, run], 'ties' {'distinct', 'largest_share', 'at', 'at_zero', 'at_one'},
    'parameter' ({'name', 'rho', 'n'} or None), 'cameras' [(name, w, h) under TINY_CAMERA_PX]}. Every finding is a
    measured fact about these pairs; none is proof of a cause (a Spearman over a dozen pairs is itself noisy)."""
    ok = [p for p in pairs or () if p.get('high') is not None and p.get('low') is not None]
    n = len(ok)
    lo = np.array([float(p['low']) for p in ok]) if n else np.zeros(0)
    hi = np.array([float(p['high']) for p in ok]) if n else np.zeros(0)
    rho = spearman(lo, hi) if n >= 3 else None
    out = {'spearman': rho, 'pairs': n, 'min_spearman': float(min_spearman), 'min_pairs': int(min_pairs),
           'low': rho is None or rho < min_spearman, 'findings': [], 'disagreements': [], 'ties': None,
           'parameter': None, 'cameras': []}
    F = out['findings']
    if n < min_pairs:
        F.append({'key': 'few_pairs', 'tone': 'warning',
                  'text': f'{n} of the {min_pairs} pairs the trust rule needs: ρ over so few pairs moves a lot with '
                          'each new run'})
    if n:
        vals, counts = np.unique(np.round(lo, 6), return_counts=True)
        k = int(np.argmax(counts))
        share = float(counts[k]) / n
        out['ties'] = {'distinct': int(len(vals)), 'largest_share': round(share, 3), 'count': int(counts[k]),
                       'at': float(vals[k]),
                       'at_zero': int(np.sum(lo <= 1e-6)), 'at_one': int(np.sum(lo >= 1.0 - 1e-6))}
        if share >= SATURATED_SHARE and counts[k] >= 3:
            where = 'at 0 (the screen sees no conflict)' if vals[k] <= 1e-6 else \
                'at 1 (the screen sees a collision)' if vals[k] >= 1.0 - 1e-6 else f'at {vals[k]:.3f}'
            F.append({'key': 'saturation', 'tone': 'warning',
                      'text': f'{int(counts[k])} of {n} pairs share one cheap value {where}: tied values cannot be '
                              'ordered, so ρ cannot reach the CARLA order among them'})
    if n >= 2:
        rl, rh = ranks(lo), ranks(hi)
        diff = np.abs(rl - rh)
        order = np.argsort(-diff, kind='stable')[:max(int(top), 0)]
        for i in order:
            if diff[i] <= 0:
                break
            p = ok[int(i)]
            out['disagreements'].append({'values': p.get('values') or {}, 'low': float(lo[i]), 'high': float(hi[i]),
                                         'rank_low': float(rl[i]), 'rank_high': float(rh[i]),
                                         'rank_diff': float(diff[i]), 'run': p.get('run') or '',
                                         'role': p.get('role'), 'fidelity': p.get('fidelity') or 'unknown'})
        if out['disagreements']:
            d0 = out['disagreements'][0]
            kind = ('the screen ranks it low, CARLA high' if d0['rank_high'] > d0['rank_low'] else
                    'the screen ranks it high, CARLA low')
            F.append({'key': 'disagreement', 'tone': '',
                      'text': f'largest rank disagreement: cheap {d0["low"]:.3f} against CARLA {d0["high"]:.3f} '
                              f'({kind}; {d0["rank_diff"]:.0f} places apart of {n})'})
        if n >= 5:
            best = None
            names = sorted({k for p in ok for k, v in (p.get('values') or {}).items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
            for name in names:
                xs = [(p.get('values') or {}).get(name) for p in ok]
                if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in xs):
                    continue
                r = spearman(np.array(xs, dtype=float), diff)
                if r is not None and (best is None or abs(r) > abs(best['rho'])):
                    best = {'name': name, 'rho': round(float(r), 3), 'n': n}
            if best is not None and abs(best['rho']) >= CONCENTRATED_RHO:
                out['parameter'] = best
                side = 'high' if best['rho'] > 0 else 'low'
                F.append({'key': 'parameter', 'tone': 'warning',
                          'text': f'the disagreement grows with {best["name"]} (Spearman {best["rho"]:+.2f} of the '
                                  f'rank gap against it): the screen misreads {side} {best["name"]}'})
    tiny = [c for c in _camera_pixels(rig) if c[1] * c[2] < TINY_CAMERA_PX]
    if tiny:
        out['cameras'] = tiny
        c = tiny[0]
        F.append({'key': 'camera', 'tone': 'warning',
                  'text': f'the rig\'s camera {c[0]} is {c[1]} × {c[2]}: a stack that sees through it can miss what '
                          'the kinematic screen (perfect knowledge) assumes it sees'})
    if rho is not None and rho < 0:
        F.insert(0, {'key': 'inverted', 'tone': 'critical',
                     'text': f'ρ {rho:+.2f}: the screen orders the runs against CARLA — it is not measuring the same '
                             'thing for this scenario'})
    return out


def bench(scenario: str = 'lead_brake', n: int = 4096, device: str = 'cpu', seed: int = 0, driver: str = 'follower',
          repeats: int = 2) -> dict:
    """Measure the fast simulator's throughput here: n uniform parameter sets of a built-in scenario, the rollout
    and the criticality timed separately (the best of `repeats`; the first torch call's start-up is not counted)."""
    scn = SC.builtin(scenario)
    space = SC.search_space(scn)
    rng = np.random.default_rng(seed)
    P = [S.to_params(space, x) for x in rng.random((n, len(space)))]
    low = LowFidelity(scn, space, driver=driver, device=device, batch=n)
    names = [k for k in scn['params'] if not isinstance(P[0].get(k), str)]
    first = {k: P[0][k] for k in scn['params'] if isinstance(P[0].get(k), str)}
    same = [p for p in P if all(p[k] == v for k, v in first.items())]
    vals = dict(first, **{k: np.array([p[k] for p in same]) for k in names})
    best_r = best_c = math.inf
    rollout(scn, {k: (v[:8] if isinstance(v, np.ndarray) else v) for k, v in vals.items()}, min(8, len(same)),
            device=device, driver=driver)                                     # warm-up
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        tr = rollout(scn, vals, len(same), device=device, driver=driver)
        t1 = time.perf_counter()
        c = criticality(tr)
        t2 = time.perf_counter()
        best_r, best_c = min(best_r, t1 - t0), min(best_c, t2 - t1)
    m = len(same)
    return {'scenario': scenario, 'n': m, 'steps': int(tr['ego'].shape[1]), 'device': low.device, 'driver': driver,
            'rollout_s': round(best_r, 4), 'criticality_s': round(best_c, 4),
            'rollout_per_s': round(m / best_r, 1), 'per_s': round(m / (best_r + best_c), 1),
            'failures_share': float(np.mean(c['criticality'] >= 0.5))}


# ================================================================================================ the command
def prescreen_argv(project_path: str, scenario: str, name: str, budget: int, batch: int, threshold: float,
                   seed: int, world: str, rig: str, screen: int = 10000, controls: float = 0.2,
                   min_spearman: float = 0.5, fallback: str = 'bayes-pof', count: str = 'avoidable',
                   fake_carla: str = '', on: str = 'local') -> list:
    """`falcon prescreen run …` as Rare cases ▸ Two-speed search queues it (the arguments after
    `python -m falcon_autonomy`)."""
    argv = ['prescreen', 'run', project_path, '--scenario', scenario, '--name', name, '--budget', str(int(budget)),
            '--batch', str(int(batch)), '--threshold', str(threshold), '--seed', str(int(seed)), '--world', world,
            '--rig', rig, '--screen', str(int(screen)), '--controls', str(controls), '--min-spearman',
            str(min_spearman), '--fallback', fallback, '--count', count]
    if on and on != 'local':
        argv += ['--on', on]
    return argv + (['--fake-carla', fake_carla] if fake_carla else [])


def _scenario_of(proj, key: str):
    scn = (proj.item('scenarios', key) if proj is not None else None) or \
        (SC.builtin(key) if key in SC.LIBRARY else None)
    if scn is None and key.endswith('.json') and os.path.isfile(key):
        with open(key, encoding='utf-8') as fh:
            scn = json.load(fh)
    return scn


def main_prescreen(argv) -> int:
    """falcon prescreen run PROJECT --scenario KEY […] | show PROJECT [--name N | --scenario KEY] [--json] |
    screen --scenario KEY [--n N] [--top K] | scenarios | bench [--scenario KEY] [--n N] [--device D]
    (exit codes as the CLI's)."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon prescreen', description='Two-speed failure search: a fast kinematic '
                                 'simulator screens a scenario\'s parameters, CARLA confirms the most promising '
                                 '(docs/two-speed-search.md).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='the two-speed search of a scenario, as a resumable study')
    r.add_argument('project')
    r.add_argument('--scenario', required=True, help='a built-in, a project scenario id or a scenario file (.json)')
    r.add_argument('--name', default='', help='the study (default: the Rare cases page\'s search of the scenario)')
    r.add_argument('--world', default='')
    r.add_argument('--rig', default='')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--search-seed', type=int, default=None, help='the screen\'s and the controls\' seed (default: '
                                                                 '--seed)')
    r.add_argument('--budget', type=int, default=40, help='CARLA runs in all (the study\'s finished runs count)')
    r.add_argument('--batch', type=int, default=8, help='CARLA runs per round')
    r.add_argument('--threshold', type=float, default=0.5)
    r.add_argument('--screen', type=int, default=10000, help='cheap candidates screened (Latin hypercube)')
    r.add_argument('--refine', type=int, default=2000, help='more cheap candidates from the cross-entropy method')
    r.add_argument('--controls', type=float, default=0.2, help='the share of each round spent on uniform random '
                                                               'controls (default 0.2: 20 %%)')
    r.add_argument('--min-spearman', type=float, default=0.5, help='below this rank correlation the screen is not '
                                                                   'trusted and a plain search takes over')
    r.add_argument('--min-pairs', type=int, default=12, help='confirmed pairs before trust is judged')
    r.add_argument('--fallback', default='bayes-pof', choices=('bayes', 'bayes-pof', 'cmaes', 'cem', 'adaptive',
                                                               'random'))
    r.add_argument('--retrust', action='store_true', help='judge trust again on a study that fell back')
    r.add_argument('--driver', default='follower', choices=DRIVERS[:2], help='the fast simulator\'s ego: the '
                   'director\'s lane follower (default) or the intelligent driver model')
    r.add_argument('--device', default='cpu', help='cpu (numpy) or a torch device for the rollout (cuda)')
    r.add_argument('--count', default='avoidable', choices=('avoidable', 'all'),
                   help='CARLA\'s objective counts avoidable failures only (RSS; default) or all')
    r.add_argument('--no-feasibility', action='store_true')
    r.add_argument('--fps', type=float, default=0.0, help='the simulation step rate (default 20)')
    r.add_argument('--fake-carla', default='')
    from ..sim.service import placement_arg
    placement_arg(r)
    s = sub.add_parser('show', help='a two-speed study\'s result: the fidelity gap, the trust word, the failures')
    s.add_argument('project')
    s.add_argument('--name', default='')
    s.add_argument('--scenario', default='')
    s.add_argument('--json', action='store_true')
    c = sub.add_parser('screen', help='the cheap screen alone (no CARLA, no project): the most critical candidates')
    c.add_argument('--scenario', required=True)
    c.add_argument('--n', type=int, default=10000)
    c.add_argument('--top', type=int, default=10)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--driver', default='follower', choices=DRIVERS[:2])
    c.add_argument('--device', default='cpu')
    c.add_argument('--json', action='store_true')
    sub.add_parser('scenarios', help='which built-in scenarios the fast simulator runs, and why not the others')
    b = sub.add_parser('bench', help='the fast simulator\'s throughput on this machine')
    b.add_argument('--scenario', default='lead_brake')
    b.add_argument('--n', type=int, default=4096)
    b.add_argument('--device', default='cpu')
    b.add_argument('--driver', default='follower', choices=DRIVERS[:2])
    a = ap.parse_args(argv)

    if a.cmd == 'scenarios':
        for k, why in coverage().items():
            print(f'{k:<24} ' + ('covered' if why is None else f'refused: {why}'))
        return P.OK
    if a.cmd == 'bench':
        try:
            print(json.dumps(bench(a.scenario, a.n, a.device, driver=a.driver), indent=2))
        except (Unsupported, ValueError, KeyError) as e:
            print(f'failed: {e}', flush=True)
            return P.REFUSED
        return P.OK
    if a.cmd == 'screen':
        scn = _scenario_of(None, a.scenario)
        if scn is None:
            print(f'failed: no scenario {a.scenario!r}', flush=True)
            return P.FAILED
        try:
            low = LowFidelity(scn, driver=a.driver, device=a.device)
        except (Unsupported, ValueError) as e:
            print(f'failed: {e}', flush=True)
            return P.REFUSED
        X = lhs(a.n, len(low.space), np.random.default_rng([a.seed, 11]))
        Pp = [S.to_params(low.space, x) for x in X]
        L = low(Pp)
        order = np.argsort(-L)[:a.top]
        out = {'scenario': scn.get('id'), 'screened': a.n, 'throughput': low.throughput(), 'fidelity': SCREEN_FIDELITY,
               'share_at_or_over_0.5': float(np.mean(L >= 0.5)),
               'top': [{'low': round(float(L[i]), 4), 'values': Pp[i]} for i in order]}
        if a.json:
            print(json.dumps(out, indent=2))
        else:
            tp = out['throughput']
            print(f'{a.n} candidates in {tp["seconds"]} s ({tp["per_s"]} evaluations/s, {tp["device"]}); '
                  f'{out["share_at_or_over_0.5"]:.1%} at or over 0.5 by the cheap model (a screen, not a result)')
            for t in out['top']:
                print(f'  {t["low"]:.3f}  ' + ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                                         for k, v in sorted(t['values'].items())))
        return P.OK

    from .project import Project, ProjectError
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    if a.cmd == 'show':
        name = a.name or (default_name(a.scenario) if a.scenario else '')
        if not name:
            print('failed: give --name or --scenario', flush=True)
            return P.REFUSED
        d = read_prescreen(os.path.join(proj.workspace, 'studies', name))
        if d is None:
            print(f'failed: no two-speed result in studies/{name} (falcon prescreen run …)', flush=True)
            return P.FAILED
        print(json.dumps(d, indent=2) if a.json else headline(d))
        return P.OK

    scn = _scenario_of(proj, a.scenario)
    if scn is None:
        print(f'failed: no scenario {a.scenario!r}', flush=True)
        return P.FAILED
    why = unsupported(scn)
    if why:
        print(f'failed: the fast simulator cannot run {scn.get("id") or a.scenario}: {why}. Use falcon search '
              '(every evaluation in CARLA).', flush=True)
        return P.REFUSED
    try:
        backend(a.device)
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    from ..cluster.bundle import placement_refusal
    from ..sim.service import study_service
    from ..sim.studio import Study
    world = a.world or proj.doc['worlds'][0]['id']
    why = placement_refusal(a.on, scn, proj.item('worlds', world))
    if why:
        print(f'failed: {why}', flush=True)
        return P.REFUSED
    try:
        svc = study_service(None, proj.workspace, fake_carla=a.fake_carla, placement=a.on)
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    em = P.Emitter()
    em.hello('prescreen')
    try:
        st = Study(svc, proj, scn, a.name or default_name(scn['id']), world, a.rig or proj.doc['rigs'][0]['id'],
                   a.seed, objective={'objective': 'criticality', 'count': a.count,
                                      'feasibility': not a.no_feasibility},
                   fps=a.fps or None, search_seed=a.search_seed)
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.REFUSED
    st.emit = em
    try:
        d = run_study(st, a.budget, a.batch, a.threshold, a.screen, a.refine, a.controls, a.min_spearman,
                      a.min_pairs, a.fallback, a.driver, a.device, a.retrust,
                      lambda n: em.progress(min(n / max(a.budget, 1), 1.0), f'{n} of {a.budget}', force=True),
                      em.log)
        for line in headline(d).splitlines():
            em.log(line)
        f = d['gap']
        em.result('ok', summary={'evaluated': d['spent'], 'failures': len(d['failures']), 'trust': d['trust'],
                                 'spearman': f['spearman'], 'fidelity': d['fidelity'], 'screened': d['screened'],
                                 'fallback': (d.get('fallback') or {}).get('method')})
    except (RuntimeError, ValueError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.FAILED
    svc.shutdown()
    return P.OK
