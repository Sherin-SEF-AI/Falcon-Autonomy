"""Level-k parameters calibrated on the road users of your own dashcam near-misses (numpy; torch through
core.trafficgraph): `falcon levelk calibrate --from-nearmiss INGEST_DIR [--kind two-wheeler]…`.

PRIVATE. Everything here is derived from the user's dashcam footage: it is read from, and by default written into,
the ingest's own falcon-ingest/ folder (assistant.privacy keeps all of it from the assistant's model), the result
carries the dashcam markers (run.json worker dashcam-levelk-calibrate, source dashcam; its behaviour block
levelk_source dashcam) wherever it is written, and a traffic profile it is loaded into carries levelk_source too — so
that profile, and every drive, run and study bundle with it, stays on this machine (assistant.privacy.levelk_reason;
cluster and cloud bundles refuse it). Nothing here prints a position, a time of day or a clip's name.

Input: core.dashcam_ingest's falcon-ingest/clips/<key>/nearmiss.json (core.nearmiss.mine_clip: every candidate
window's events — one per tracked road user — with the smoothed (N, 8) trajectories t, x, y, yaw°, vx, vy, length,
width of the road user and of the ego from the GPS, in local east-north metres) and, when it is there, the clip's
perception.json (the tracked boxes' fused ranges).

Decision samples (decisions). In each window every road user of the asked kinds (core.levelk types: car, bus — a
truck too —, two_wheeler — motorcycles and bicycles —, autorickshaw — D's heuristic class included —, pedestrian;
cattle and anything unknown are left out) is a level-k agent whose reference path is a quadratic fitted to its
smoothed path over the window (its lane's direction and curvature, not its swerves), every DT_PLAN of its own samples.
The likelihood is that of its ACCELERATION given the lateral course it took: every acceleration option is rolled out
along the reference path with the lateral offset it actually had at
each step's time — the joint quantal response conditioned on its lateral choice, which a track cannot identify (a
lane's centre and lines are not known without a map):
* its state: arc length on the path, speed (the smoothed velocity's norm), executed acceleration (the speed's change
  over the previous re-plan interval), its box (D's typical size for the kind);
* the plan it was seen to follow: the acceleration options whose execution through core.levelk's jerk limit from that
  acceleration reproduces the observed change of speed over the next DT_PLAN (the largest multiple of the sampling
  interval within it: 0.2 s at 10 frames/s) to within DV_TOL (0.1 m/s) of the best — the likelihood is that of the
  set of plans the observation cannot tell apart: the jerk limit binds for both −2 and −4 m/s² from rest, at a
  standstill every braking plan stays still, and a smoothed track blurs its accelerations (measured on synthetic
  tracks through core.nearmiss's smoother: the speed within 0.02 m/s (median), 0.24 m/s at a track's ends — the first
  and last EDGE_S (0.5 s) of a track are left out);
* the others — the ego and every other road user seen at that instant — over the horizon (3 s) as OBSERVED, beyond
  the end of their track extrapolated at constant velocity; their boxes; a leader's hard stop (core.levelk.worst_case,
  as the level-k solver adds it).
The likelihood is core.trafficgraph's (the quantal response of core.levelk.utility, differentiable in α, λ, v_des and
t_gap), one calibration per kind, with a lapse of LAPSE (2 %): each decision is the quantal response with probability
0.98 and any plan alike with 0.02. Without it one observed plan inside the model's collision margin — monocular boxes
put safe gaps there (measured: every range × 0.9 on synthetic tracks drove λ to its floor) — decides the estimate.

Intervals (every result states them):
* Wald 95 % intervals from the inverse Hessian — optimistic: consecutive decisions of one road user are treated as
  independent, which they are not, and the others' motion is taken as observed (a quantal best response to what they
  actually did — the inverse-planning assumption — not to the model's level-(k − 1) prediction of them);
* the depth scale's uncertainty, propagated: core.nearmiss's bootstrap factor (every range × U[0.9, 1.1], one factor
  per window) redrawn depth_draws times, the road users' trajectories rebuilt — exactly, by core.nearmiss.actor_traj
  on the clip's perception rows, or, without them, approximately (their offsets from the ego scaled) — and the
  calibration repeated from the estimate (warm start); the spread of those estimates is reported and added in
  quadrature to the Wald variance (the combined interval, within the parameter's bounds). depth_draws 0: not
  propagated, stated.
* A parameter whose estimate lands on its bound (core.levelk.PARAM_BOUNDS) is not identified by the decisions: it has
  no interval (said so in at_bound), and the others' intervals come from a refit that holds it there.
* Level k is not estimated (the likelihood is that of a level-1 best response to the observed others): the block says
  k 1; v_des, t_gap, α and λ are the estimates.

Output (out_dir, default falcon-ingest/levelk/): calibration.json — per kind the estimates, standard errors, the Wald,
depth-scale and combined intervals, decisions, road users and windows used, convergence, the assumptions — and its
behaviour block {"model": "levelk", "levelk": {kind: {k, lambda, alpha, v_des, t_gap}}, "levelk_source": {format
dashcam}}, which Worlds ▸ Traffic profile ▸ *Load calibrated parameters…* applies (one undoable edit) and
`falcon recon closedloop --levelk-params` reads; run.json the provenance and the dashcam marker.

Limits: monocular tracks (ranges rest on the horizon and a flat road; sizes typical, not measured), a window is ±3 s
around a candidate (near-misses over-represent hard interactions: the estimates describe road users near your car in
the moments that were mined, not traffic at large); the ego's motion is the GPS's (1 Hz, smoothed); lateral choices
are not identified; level k is not estimated.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import levelk as LK

SCHEMA = 'falcon.levelk_nearmiss/1'
MIN_DECISIONS = 20
DEPTH_DRAWS = 20
STILL_MPS = 0.3
DV_TOL = 0.1                      # m/s: how well a smoothed track gives a speed change over 0.25 s (module doc)
EDGE_S = 0.5                      # s: the smoother's ends left out of a track
LAPSE = 0.02                      # the trembling hand of the likelihood (core.trafficgraph.nll; module doc)
DEPTH_SCALE = 0.10                # core.nearmiss.DEPTH_SCALE (not imported: kept light; checked by the tests)
ASSUMPTIONS = (
    'the others\' motion over the horizon is taken as observed: a quantal best response to what they actually did '
    '(inverse planning), not to the model\'s level-(k − 1) prediction of them',
    'Wald intervals treat consecutive decisions of one road user as independent, which they are not: they are '
    'optimistic',
    'the likelihood is that of each road user\'s acceleration given the lateral course it took (lateral choices are '
    'not identified from a track without a map)',
    'level k is not estimated (k 1: a best response to the observed others)',
    'a 2 % lapse (any plan alike) keeps one inexplicable decision from deciding the estimate',
    'monocular ranges: the depth scale\'s uncertainty is propagated by redrawing core.nearmiss\'s bootstrap factor '
    '(U[0.9, 1.1] per window) when depth_draws > 0',
    'near-miss windows over-represent hard interactions: the estimates describe road users near your car in the mined '
    'moments, not traffic at large')


def kind_type(kind: str) -> str | None:
    """core.levelk's type of a near-miss kind or a type name ('two-wheeler' as 'two_wheeler'); None when level-k has
    no model of it (cattle, anything unknown)."""
    k = str(kind).strip().lower().replace('-', '_').replace(' ', '_')
    if k in LK.TYPES:
        return k
    return LK.KIND_TYPE.get(k)


# ================================================================================================ reading the ingest
def ingest_root(folder: str) -> str:
    """The falcon-ingest/ folder of an ingest (the folder of clips, or falcon-ingest/ itself)."""
    from .dashcam_ingest import INGEST_DIR
    f = os.path.abspath(os.path.expanduser(folder))
    if os.path.basename(f) == INGEST_DIR and os.path.isfile(os.path.join(f, 'ingest.json')):
        return f
    if os.path.isfile(os.path.join(f, INGEST_DIR, 'ingest.json')):
        return os.path.join(f, INGEST_DIR)
    raise ValueError(f'{folder}: not a dashcam ingest (no {INGEST_DIR}/ingest.json — falcon dashcam ingest DIR first)')


def windows(root: str) -> list:
    """Every mined window of every clip: [{'clip', 'window', 'ego' (N, 8), 'actors': [{'id', 'track', 'kind', 'traj'
    (N_i, 8), 'rows' (perception rows or None)}], 'ego_track' (dict of arrays, or None)}] — trajectories right-handed
    east-north as core.nearmiss writes them."""
    from . import nearmiss as NM
    out = []
    cdir = os.path.join(root, 'clips')
    for ck in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else ():
        mp = os.path.join(cdir, ck, 'nearmiss.json')
        if not os.path.isfile(mp):
            continue
        with open(mp, encoding='utf-8') as fh:
            m = json.load(fh)
        perc = None
        pp = os.path.join(cdir, ck, 'perception.json')
        if os.path.isfile(pp):
            with open(pp, encoding='utf-8') as fh:
                perc = json.load(fh)
        tracks = NM._tracks(perc) if perc else {}
        k_of = {int(f['n']): k for k, f in enumerate(perc['frames'])} if perc else {}
        eg = None
        if m.get('ego_track'):
            eg = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in m['ego_track'].items()}
        by_win: dict = {}
        for e in m.get('events') or ():
            tr = e.get('trajectory') or {}
            if not tr.get('actor') or not tr.get('ego'):
                continue
            key = tuple(round(float(x), 3) for x in e.get('window_s') or (0.0, 0.0))
            w = by_win.setdefault(key, {'clip': ck, 'window': list(key), 'ego': np.asarray(tr['ego'], float),
                                        'actors': [], 'ego_track': eg})
            rows = None
            if perc and e.get('track') is not None and eg is not None:
                ks = {k_of[n] for n in e.get('frames') or () if n in k_of}
                rows = [r for r in tracks.get(int(e['track']), ()) if r[0] in ks] or None
            w['actors'].append({'id': str(e.get('id') or f'{ck}-{e.get("track")}'), 'track': e.get('track'),
                                'kind': e.get('kind', ''), 'class_source': e.get('class_source', 'model'),
                                'traj': np.asarray(tr['actor'], float), 'rows': rows})
        out += [by_win[k] for k in sorted(by_win)]
    return out


def scaled(win: dict, actor: dict, scale: float) -> tuple:
    """(the road user's (N, 8) trajectory at a depth scale, 'exact' | 'approximate'): core.nearmiss.actor_traj re-run
    on its perception rows with every range × scale, or (no rows) its offset from the ego at each instant scaled."""
    from . import nearmiss as NM
    if abs(scale - 1.0) < 1e-12:
        return actor['traj'], 'exact'
    if actor['rows'] and win['ego_track'] is not None:
        return NM.actor_traj(actor['rows'], win['ego_track'], actor['kind'], float(scale)), 'exact'
    a = actor['traj'].copy()
    e = win['ego']
    for c in (1, 2, 4, 5):
        ec = np.interp(a[:, 0], e[:, 0], e[:, c])
        a[:, c] = ec + scale * (a[:, c] - ec)
    return a, 'approximate'


# ================================================================================================ decisions
def _carla(T: np.ndarray) -> np.ndarray:
    """(N, 8) right-handed east-north → core.levelk's frame (CARLA's: y mirrored, yaw clockwise)."""
    C = np.array(T, float, copy=True)
    C[:, 2] *= -1.0
    C[:, 3] *= -1.0
    C[:, 5] *= -1.0
    return C


def _ref_path(T: np.ndarray) -> LK.RefPath:
    """A road user's reference path over a window: a quadratic in the distance it travelled fitted to its smoothed
    positions (its lane's direction and curvature, not its swerves), sampled every metre and extended straight 50 m
    past either end; a straight line along its heading when it moved less than 2 m."""
    xy = T[:, 1:3]
    u = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    if u[-1] < 2.0:
        return LK.RefPath.line(T[0, 1], T[0, 2], T[0, 3], length=200.0, back=50.0)
    deg = 2 if u[-1] >= 10.0 and len(T) >= 6 else 1
    cx, cy = np.polyfit(u, xy[:, 0], deg), np.polyfit(u, xy[:, 1], deg)
    q = np.linspace(0.0, u[-1], max(int(u[-1]), 2) + 1)
    pts = np.c_[np.polyval(cx, q), np.polyval(cy, q)]
    d0 = pts[1] - pts[0]
    d1 = pts[-1] - pts[-2]
    d0, d1 = d0 / np.linalg.norm(d0), d1 / np.linalg.norm(d1)
    return LK.RefPath(np.r_[[pts[0] - 50.0 * d0], pts, [pts[-1] + 50.0 * d1]])


def _rollouts(ag: dict, A, d_t, vmax: float) -> dict:
    """core.levelk.roll's rollouts of acceleration plans A (P,) through the jerk limit, the lateral offset following
    the road user's observed offset at each step's time (d_t (N_STEPS,)): the same for every option — the lateral
    course it took."""
    A = np.asarray(A, float)
    P, n, dt = len(A), LK.N_STEPS, LK.DT_PLAN
    s, v = np.empty((P, n)), np.empty((P, n))
    s0, v0 = np.full(P, ag['s']), np.full(P, ag['v'])
    acc = np.full(P, float(ag['a']))
    for k in range(n):
        acc = acc + np.clip(A - acc, -LK.JERK_MAX * dt, LK.JERK_MAX * dt)
        v1 = np.clip(v0 + acc * dt, 0.0, vmax)
        s0 = s0 + 0.5 * (v0 + v1) * dt
        v0 = v1
        s[:, k], v[:, k] = s0, v1
    d = np.broadcast_to(np.asarray(d_t, float)[None, :], (P, n))
    x, y, yaw = ag['path'].at(s, d)
    return {'s': s, 'd': np.array(d), 'v': v, 'x': x, 'y': y, 'yaw': yaw}


def _future(T: np.ndarray, t0: float) -> tuple:
    """(x, y) of a trajectory at t0 + DT_PLAN · (1 … N_STEPS): interpolated inside it, constant velocity past its
    end."""
    ts = t0 + LK.DT_PLAN * np.arange(1, LK.N_STEPS + 1)
    x = np.interp(ts, T[:, 0], T[:, 1])
    y = np.interp(ts, T[:, 0], T[:, 2])
    past = ts > T[-1, 0]
    x[past] = T[-1, 1] + T[-1, 4] * (ts[past] - T[-1, 0])
    y[past] = T[-1, 2] + T[-1, 5] * (ts[past] - T[-1, 0])
    return x, y


def _dv(a0: float, v0: float, A: float, h: int, dt: float, vmax: float) -> float:
    """The speed change over h samples of a plan's acceleration executed through the jerk limit from a0."""
    acc, v = a0, v0
    for _ in range(h):
        acc += float(np.clip(A - acc, -LK.JERK_MAX * dt, LK.JERK_MAX * dt))
        v = min(max(v + acc * dt, 0.0), vmax)
    return v - v0


def decisions(win: dict, types, scale: float = 1.0, every_s: float = LK.DT_PLAN) -> tuple:
    """(samples for core.trafficgraph.calibrate, {'still': decisions at a standstill, 'how': exact|approximate}) of one
    window: every road user of the given core.levelk types (module doc)."""
    others_all = {'ego': _carla(win['ego'])}
    trajs, how = {}, 'exact'
    for a in win['actors']:
        T, h_ = scaled(win, a, scale)
        how = 'approximate' if h_ == 'approximate' else how
        trajs[a['id']] = (a, _carla(T))
    others_all.update({aid: T for aid, (_a, T) in trajs.items()})
    out, still = [], 0
    for aid, (a, T) in trajs.items():
        typ = kind_type(a['kind'])
        if typ is None or typ not in types or len(T) < 3:
            continue
        dt = float(np.median(np.diff(T[:, 0])))
        if dt <= 0:
            continue
        h = max(int(math.floor(LK.DT_PLAN / dt + 1e-6)), 1)
        path = _ref_path(T)
        spec = LK.TYPES[typ]
        L, W = float(T[0, 6]), float(T[0, 7])
        sp = np.hypot(T[:, 4], T[:, 5])
        dd = np.array([path.project(x, y)[1] for x, y in T[:, 1:3]])
        last_t = -math.inf
        for k in range(len(T) - h):
            t = float(T[k, 0])
            if t < last_t + every_s - 1e-6 or t < T[0, 0] + EDGE_S - 1e-6 or T[k + h, 0] > T[-1, 0] - EDGE_S + 1e-6:
                continue
            last_t = t
            v0, v1 = float(sp[k]), float(sp[k + h])
            kb = max(k - h, 0)                               # what it executed over the last re-plan interval
            a0 = float((sp[k] - sp[kb]) / max(T[k, 0] - T[kb, 0], 1e-6))
            a0 = float(np.clip(a0, min(LK.ACCELS), max(LK.ACCELS)))
            s, d = path.project(T[k, 1], T[k, 2])
            ag = LK.make_agent(aid, typ, path, s, d, v0)
            ag.update(a=a0, d_ref=d, length=L, width=W)
            A = np.asarray(LK.ACCELS, float)
            D = np.full(len(A), d)
            obs = v1 - v0
            dvs = {Ai: _dv(a0, v0, Ai, h, dt, spec['v_max']) for Ai in A.tolist()}
            best = min(dvs, key=lambda Ai: (abs(dvs[Ai] - obs), abs(Ai)))
            m = int(np.argmin(np.abs(A - best)))
            err = abs(dvs[best] - obs) + DV_TOL
            same = [i for i, Ai in enumerate(A.tolist()) if abs(dvs[Ai] - obs) <= err + 1e-9]
            still += int(v0 < STILL_MPS and v1 < STILL_MPS)
            d_t = np.interp(t + LK.DT_PLAN * np.arange(1, LK.N_STEPS + 1), T[:, 0], dd)
            others, sc_agents = [], [ag]
            for oid, O in others_all.items():
                if oid == aid or not (O[0, 0] - 1e-6 <= t <= O[-1, 0] + 1e-6):
                    continue
                ox, oy = np.interp(t, O[:, 0], O[:, 1]), np.interp(t, O[:, 0], O[:, 2])
                if math.hypot(ox - T[k, 1], oy - T[k, 2]) > LK.RANGE_M:
                    continue
                fx, fy = _future(O, t)
                others.append((fx[None], fy[None], float(O[0, 6]), float(O[0, 7])))
                oyaw = float(np.interp(t, O[:, 0], O[:, 3]))
                ov = float(np.hypot(np.interp(t, O[:, 0], O[:, 4]), np.interp(t, O[:, 0], O[:, 5])))
                okind = 'car' if oid == 'ego' else (kind_type(trajs[oid][0]['kind']) or 'car')
                o = LK.make_agent(oid, okind, LK.RefPath.line(ox, oy, oyaw, length=200.0, back=0.0), 0.0, 0.0, ov,
                                  controlled=False)
                o.update(length=float(O[0, 6]), width=float(O[0, 7]))
                sc_agents.append(o)
            R = _rollouts(ag, A, d_t, spec['v_max'])
            F = LK.plan_features(ag, R, others, A, D)
            sc = LK.Scene(sc_agents)
            F['coll'] = F['coll'] + LK.worst_case(sc, ag, R, list(range(1, len(sc_agents))), A, D)
            smp = {'agent': aid, 'type': typ, 't': t, 'choice': m, 'features': F}
            if len(same) > 1:
                smp['choices'] = same
            out.append(smp)
    return out, {'still': still, 'how': how}


# ================================================================================================ calibration
def _at_bound(k: str, v: float) -> bool:
    """An estimate on (within 0.1 % of the range of) its parameter's bound (core.levelk.PARAM_BOUNDS)."""
    lo, hi = LK.PARAM_BOUNDS[k]
    tol = 1e-3 * (min(hi, 50.0) - lo)
    return v <= lo + tol or v >= hi - tol


def _all(wins, types, scales=None) -> tuple:
    samples, still, how = [], 0, 'exact'
    for i, w in enumerate(wins):
        s, info = decisions(w, types, 1.0 if scales is None else float(scales[i]))
        samples += s
        still += info['still']
        how = 'approximate' if info['how'] == 'approximate' else how
    return samples, still, how


def calibrate_nearmiss(folder: str, out_dir: str = '', kinds=None, steps: int = 300, depth_draws: int = DEPTH_DRAWS,
                       seed: int = 0, min_decisions: int = MIN_DECISIONS, log=None) -> dict:
    """`falcon levelk calibrate --from-nearmiss` (module doc): writes calibration.json and run.json in out_dir
    (default falcon-ingest/levelk/) and returns the document."""
    from . import trafficgraph as TG
    from .provenance import block, utc_now, write_json
    log = log or (lambda m: None)
    t_start = time.monotonic()
    root = ingest_root(folder)
    out_dir = os.path.abspath(out_dir or os.path.join(root, 'levelk'))
    wins = windows(root)
    if not wins:
        raise ValueError('the ingest has no mined windows with trajectories (falcon dashcam ingest DIR first)')
    want = []
    for k in kinds or ('car', 'bus', 'two_wheeler', 'autorickshaw', 'pedestrian'):
        t = kind_type(k)
        if t is None:
            raise ValueError(f'level-k has no model of {k!r} (kinds: car, bus, two-wheeler, autorickshaw, pedestrian, '
                             'or a near-miss kind such as motorcycle or truck)')
        if t not in want:
            want.append(t)
    seen = {}
    for w in wins:
        for a in w['actors']:
            t = kind_type(a['kind'])
            seen.setdefault(t or 'none', set()).add(a['id'])
    log(f'{len(wins)} windows; road users by type: '
        + ', '.join(f'{k} {len(v)}' for k, v in sorted(seen.items())))
    rng = np.random.default_rng(int(seed))
    draws = [rng.uniform(1.0 - DEPTH_SCALE, 1.0 + DEPTH_SCALE, len(wins)) for _ in range(int(depth_draws))]
    results, block_ = {}, {}
    for typ in want:
        samples, still, how = _all(wins, (typ,))
        users = sorted({s['agent'] for s in samples})
        res = {'decisions': len(samples), 'road_users': len(users), 'standing_still': still,
               'windows': len({w['clip'] + str(w['window']) for w in wins
                               for a in w['actors'] if a['id'] in set(users)})}
        if len(samples) < int(min_decisions):
            res['note'] = f'too few decisions to calibrate (at least {int(min_decisions)})'
            results[typ] = res
            log(f'{typ}: {len(samples)} decisions — too few')
            continue
        log(f'{typ}: {len(samples)} decisions of {len(users)} road users')
        cal = TG.calibrate(samples, steps=steps, lapse=LAPSE)
        est = cal['estimates']
        at_bound = [k for k in TG.FREE if _at_bound(k, est[k])]
        if at_bound:
            # a parameter on its bound has no meaningful curvature: the others' intervals with it held there (the
            # softplus bounds λ, v_des and t_gap only from below, so an estimate can run past its upper bound: held AT it)
            est = {k: (min(max(v, LK.PARAM_BOUNDS[k][0]), LK.PARAM_BOUNDS[k][1]) if k in at_bound else v)
                   for k, v in est.items()}
            cal = TG.calibrate(samples, init=est, steps=max(int(steps) // 3, 40), lapse=LAPSE,
                               fixed={k: est[k] for k in at_bound})
            est = cal['estimates']
            for k in at_bound:
                cal['se'][k], cal['ci95'][k] = None, None
            res['at_bound'] = {k: f'{est[k]:g} is on its bound: not identified by these decisions (no interval; '
                                  'the others\' intervals hold it there)' for k in at_bound}
        res.update(estimates=est, se=cal['se'], wald95=cal['ci95'], loglik=cal['loglik'],
                   converged=cal['converged'], gradient_norm=cal['gradient_norm'])
        boot = []
        for b, sc in enumerate(draws):
            sb, _, how = _all(wins, (typ,), sc)
            if len(sb) < int(min_decisions):
                continue
            cb = TG.calibrate(sb, init=est, steps=max(int(steps) // 3, 40), lapse=LAPSE)
            boot.append(cb['estimates'])
            log(f'{typ}: depth draw {b + 1} of {len(draws)}')
        if boot:
            sd = {k: float(np.std([x[k] for x in boot], ddof=1)) if len(boot) > 1 else 0.0 for k in TG.FREE}
            res['depth_scale'] = {'draws': len(boot), 'how': how, 'sd': sd,
                                  'p5_p95': {k: [float(np.percentile([x[k] for x in boot], 5)),
                                                 float(np.percentile([x[k] for x in boot], 95))] for k in TG.FREE}}
            comb = {}
            for k in TG.FREE:
                se = (cal['se'] or {}).get(k)
                lo, hi = LK.PARAM_BOUNDS[k]                  # within the parameter's bounds
                comb[k] = None if se is None else [max(est[k] - 1.96 * math.hypot(se, sd[k]), lo),
                                                   min(est[k] + 1.96 * math.hypot(se, sd[k]), hi)]
            res['combined95'] = comb
        else:
            res['depth_scale'] = {'draws': 0, 'how': 'not propagated (depth_draws 0): the intervals leave out the '
                                                     'monocular depth scale\'s uncertainty (±10 %)'}
            res['combined95'] = res['wald95']
        results[typ] = res
        p = {'k': 1}
        for key, name in (('lam', 'lambda'), ('alpha', 'alpha'), ('v_des', 'v_des'), ('t_gap', 't_gap')):
            lo, hi = LK.PARAM_BOUNDS[key]
            p[name] = round(float(np.clip(est[key], lo, hi)), 4)
        block_[typ] = p
    behaviour = {'model': 'levelk', 'levelk': block_,
                 'levelk_source': {'format': 'dashcam', 'kind': 'nearmiss',
                                   'made_by': 'falcon levelk calibrate --from-nearmiss'}}
    if block_:
        LK.profile_params({'behaviour': behaviour})
    doc = {'schema': SCHEMA, 'created': utc_now(), 'worker': 'dashcam-levelk-calibrate',
           'source': {'format': 'dashcam', 'kind': 'nearmiss'},
           'private': 'derived from your dashcam footage: it never leaves this machine (assistant, cluster and cloud '
                      'refuse it)',
           'kinds': results, 'behaviour': behaviour, 'windows': len(wins), 'assumptions': list(ASSUMPTIONS),
           'settings': {'steps': int(steps), 'depth_draws': int(depth_draws), 'seed': int(seed),
                        'min_decisions': int(min_decisions), 'every_s': LK.DT_PLAN, 'lapse': LAPSE,
                        'dv_tol_mps': DV_TOL, 'edge_s': EDGE_S},
           'wall_s': round(time.monotonic() - t_start, 2)}
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, 'calibration.json'), doc)
    write_json(os.path.join(out_dir, 'run.json'), {
        'worker': 'dashcam-levelk-calibrate', 'status': 'ok', 'created': utc_now(), 'source': {'format': 'dashcam'},
        'summary': {k: {'decisions': v['decisions'], 'estimated': 'estimates' in v} for k, v in results.items()},
        'provenance': block({'ingest': os.path.join(root, 'ingest.json')}, seed=int(seed),
                            settings=doc['settings'])})
    doc['out'] = out_dir
    return doc
