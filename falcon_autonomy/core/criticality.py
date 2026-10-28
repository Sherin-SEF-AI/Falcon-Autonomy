"""Criticality of a scenario run (standard library + numpy): how close the ego came to a collision with each actor.

Input. A trajectory is an (N, 8) array of rows (t, x, y, yaw_deg, vx, vy, length, width) sampled at a fixed dt in
CARLA's world frame (metres, seconds, degrees). CARLA is left-handed (x forward/east, y right, z up): yaw turns from +x
towards +y, so a box's heading is (cos yaw, sin yaw) and its right side is (-sin yaw, cos yaw). Boxes are the actors'
footprints: rectangles of length × width centred on (x, y). The ego and each actor are matched step by step by time
(an actor that appears late or is destroyed early is compared only where both exist).

Measures.
- Distance between two oriented boxes: 0 when they overlap (separating axis test), otherwise the smallest
  vertex-to-edge distance either way, which is exact for disjoint convex polygons (Ericson 2005, Real-Time Collision
  Detection, §5.1.2 and §5.2.1).
- TTC, time to collision (Hayward 1972, Near-miss determination through use of a scale of danger, HRR 384): both
  boxes extrapolated at constant velocity and constant yaw; the first time within the horizon at which they overlap,
  inf if never. Box-based and exact, not sampled: in the ego's frame the actor's box translates along a line, so on
  each separating axis the overlap is one time interval and the boxes overlap on the intersection of those intervals
  (the moving separating-axis test, Ericson 2005 §5.5.8).
- PET, post-encroachment time (Allen, Shin & Cooper 1978, Analysis of traffic conflicts and collisions, TRR 667):
  the conflict area is where the two swept footprints intersect; PET is the time between the first agent leaving it
  and the second entering it. Computed in its pointwise form: the smallest |t_i - t_j| over the footprint pairs (ego
  at t_i, actor at t_j) that overlap, i.e. over every point of the conflict area the gap between one agent leaving it
  and the other reaching it (for two rectangles crossing at a right angle this equals the area form). For a follower
  on the leader's path it is the time gap. None when the swept paths never overlap. Sampled occupancy can only
  overestimate it, by less than two sample spacings (the last occupied sample is up to one step before leaving, the
  first up to one after entering); a finer dt interpolates the trajectories linearly to it.
- Minimum box-to-box distance over a run, and when.
- DRAC, deceleration rate to avoid a crash (Almqvist, Hydén & Risser 1991; as used by Archer 2005, Indicators for
  traffic safety assessment and prediction, KTH): for an actor in the ego's path, max(0, v_rel² / (2 · gap)) with
  v_rel the closing speed along the ego's heading and gap the distance the ego's box can travel along its heading
  before touching the actor's (exact, by the same moving separating-axis test at unit speed). 0 when nothing is in
  the path or it is not closing; inf when the boxes already overlap (a collision, not a braking demand).
- Collisions: the steps where the boxes overlap, and the collision sensor's events (t, other_id, impulse) if given.

Criticality scalar (heuristic; what the scenario search maximises), per actor and overall:
    s_ttc = exp(-TTC_min / 1.5 s)       1.5 s: the TTC below which encounters were judged critical (van der Horst 1990)
    s_pet = exp(-PET_min / 1.0 s)       1.0 s: heuristic scale
    s_drac = 1 - exp(-DRAC_max / 3.35)  3.35 m/s²: the DRAC threshold Archer (2005) cites
    s_dist = exp(-d_min / 2.0 m)        2.0 m: heuristic scale
    criticality = 1 if the actor collided, else 0.9 · (1 - (1 - s_ttc)(1 - s_pet)(1 - s_drac)(1 - s_dist))
an absent measure (TTC inf, PET None) contributes 0; overall = 1 on any collision (boxes or sensor), else the max over
actors. The noisy-or makes any one strong indicator count; the 0.9 cap keeps every collision above every near miss.

Limits. Footprints are planar (no z: a bridge over a road counts as a conflict); yaw rate is not extrapolated (TTC
along curves is the straight-line TTC); trajectories are taken as sampled, so a contact shorter than one step can be
missed by the overlap test of collisions() (the sensor's events cover it).
"""
from __future__ import annotations

import math

import numpy as np

COLUMNS = ('t', 'x', 'y', 'yaw_deg', 'vx', 'vy', 'length', 'width')
T, X, Y, YAW, VX, VY, LEN, WID = range(8)
EPS = 1e-9

TTC_SCALE_S = 1.5
PET_SCALE_S = 1.0
DRAC_SCALE_MPS2 = 3.35
DIST_SCALE_M = 2.0
NEAR_MISS_CAP = 0.9
CRITICALITY_FORMULA = ('heuristic: 1 if collision else 0.9·(1-(1-s_ttc)(1-s_pet)(1-s_drac)(1-s_dist)); '
                       's_ttc=exp(-TTC/1.5 s), s_pet=exp(-PET/1.0 s), s_drac=1-exp(-DRAC/3.35 m/s²), '
                       's_dist=exp(-d_min/2.0 m); absent measures count 0')


def as_traj(a) -> np.ndarray:
    """An (N, 8) float array of (t, x, y, yaw_deg, vx, vy, length, width); one row may be given as an 8-vector."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f'a trajectory is rows of {COLUMNS}; got shape {arr.shape}')
    return arr


def box_corners(x, y, yaw_deg, length, width) -> np.ndarray:
    """Corners of oriented rectangles in order front-right, front-left, rear-left, rear-right (CARLA's left-handed
    frame: right is (-sin yaw, cos yaw)). Scalars give (4, 2); arrays give (..., 4, 2)."""
    x, y, yaw, ln, wd = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y, yaw_deg, length, width)))
    r = np.radians(yaw)
    c, s = np.cos(r), np.sin(r)
    hx, hy = c * ln / 2, s * ln / 2            # half length along the heading
    rx, ry = -s * wd / 2, c * wd / 2           # half width towards the right
    cx = np.stack([x + hx + rx, x + hx - rx, x - hx - rx, x - hx + rx], axis=-1)
    cy = np.stack([y + hy + ry, y + hy - ry, y - hy - ry, y - hy + ry], axis=-1)
    return np.stack([cx, cy], axis=-1)


def _corners_of(traj: np.ndarray) -> np.ndarray:
    return box_corners(traj[:, X], traj[:, Y], traj[:, YAW], traj[:, LEN], traj[:, WID])


def _axes(P: np.ndarray) -> np.ndarray:
    """Unit edge normals of convex polygons (..., k, 2) -> (..., k, 2)."""
    e = np.roll(P, -1, axis=-2) - P
    n = np.stack([-e[..., 1], e[..., 0]], axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def _projections(A: np.ndarray, B: np.ndarray):
    """Separating-axis projections of both polygon batches onto every edge normal of either: the axes (..., a, 2)
    and (min, max) of A and of B on each (..., a)."""
    axes = np.concatenate([_axes(A), _axes(B)], axis=-2)
    pa = np.einsum('...kd,...ad->...ak', A, axes)
    pb = np.einsum('...kd,...ad->...ak', B, axes)
    return axes, pa.min(-1), pa.max(-1), pb.min(-1), pb.max(-1)


def _overlap(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Whether convex polygons overlap (touching counts), batched: (..., k, 2), (..., m, 2) -> (...)."""
    _, a0, a1, b0, b1 = _projections(A, B)
    return ~np.any((a1 < b0 - EPS) | (b1 < a0 - EPS), axis=-1)


def _point_segment(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Smallest distance from the vertices of P (..., k, 2) to the edges of polygon Q (..., m, 2) -> (...)."""
    q0 = Q[..., None, :, :]
    e = (np.roll(Q, -1, axis=-2) - Q)[..., None, :, :]
    d = P[..., :, None, :] - q0
    t = np.clip(np.sum(d * e, -1) / np.maximum(np.sum(e * e, -1), 1e-18), 0.0, 1.0)
    return np.linalg.norm(d - t[..., None] * e, axis=-1).min(axis=(-1, -2))


def _distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = np.minimum(_point_segment(A, B), _point_segment(B, A))
    return np.where(_overlap(A, B), 0.0, d)


def box_distance(a, b) -> float:
    """Minimum distance between two convex polygons given by their corners (e.g. box_corners(...)); 0 when they
    overlap or touch."""
    A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(_distance(A[None], B[None])[0])


def _sweep(A: np.ndarray, B: np.ndarray, w: np.ndarray):
    """The time interval [lo, hi] over which polygon B, translating at velocity w (..., 2) relative to A, overlaps A
    (moving separating-axis test). lo > hi: never. Batched over the leading axes."""
    axes, a0, a1, b0, b1 = _projections(A, B)
    s = np.einsum('...ad,...d->...a', axes, w)
    still = np.abs(s) < 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (a0 - b1) / s
        t2 = (a1 - b0) / s
    lo_k = np.minimum(t1, t2)
    hi_k = np.maximum(t1, t2)
    # an axis the motion is parallel to: overlapping on it always or never
    ok = (a0 - b1 <= EPS) & (a1 - b0 >= -EPS)
    lo_k = np.where(still, np.where(ok, -np.inf, np.inf), lo_k)
    hi_k = np.where(still, np.where(ok, np.inf, -np.inf), hi_k)
    return lo_k.max(-1), hi_k.min(-1)


def _ttc_rows(ego: np.ndarray, actor: np.ndarray, horizon: float) -> np.ndarray:
    A, B = _corners_of(ego), _corners_of(actor)
    w = actor[:, [VX, VY]] - ego[:, [VX, VY]]
    lo, hi = _sweep(A, B, w)
    hit = (lo <= hi) & (hi >= 0)
    t = np.maximum(lo, 0.0)
    return np.where(hit & (t <= horizon), t, np.inf)


def ttc(ego_state, actor_state, horizon: float = 10.0) -> float:
    """Time to collision (s) of two states (8-vectors): the first time the boxes overlap under constant velocity,
    0 if they overlap now, inf if not within the horizon. Exact (no time sampling)."""
    return float(_ttc_rows(as_traj(ego_state), as_traj(actor_state), horizon)[0])


def _align(ego: np.ndarray, actor: np.ndarray):
    """Index pairs (ego, actor) of the steps both trajectories have (times equal within a quarter step)."""
    te, ta = ego[:, T], actor[:, T]
    if not len(te) or not len(ta):
        return np.zeros(0, int), np.zeros(0, int)
    step = float(np.median(np.diff(te))) if len(te) > 1 else 0.0
    tol = 1e-6 + 0.25 * step
    if len(ta) == 1:
        cand = np.zeros(len(te), int)
    else:
        j = np.clip(np.searchsorted(ta, te), 1, len(ta) - 1)
        cand = np.where(np.abs(ta[j - 1] - te) <= np.abs(ta[j] - te), j - 1, j)
    keep = np.abs(ta[cand] - te) <= tol
    return np.nonzero(keep)[0], cand[keep]


def ttc_series(ego_traj, actor_traj, horizon: float = 10.0):
    """(t, ttc) over the steps both exist: the TTC from each step's states."""
    e, a = as_traj(ego_traj), as_traj(actor_traj)
    ie, ia = _align(e, a)
    return e[ie, T], _ttc_rows(e[ie], a[ia], horizon)


def min_distance(ego_traj, actor_traj):
    """(distance m, t) — the smallest box-to-box distance over the run and when; (inf, None) if never together."""
    e, a = as_traj(ego_traj), as_traj(actor_traj)
    ie, ia = _align(e, a)
    if not len(ie):
        return math.inf, None
    d = _distance(_corners_of(e[ie]), _corners_of(a[ia]))
    k = int(np.argmin(d))
    return float(d[k]), float(e[ie[k], T])


def _drac_rows(ego: np.ndarray, actor: np.ndarray) -> np.ndarray:
    r = np.radians(ego[:, YAW])
    h = np.stack([np.cos(r), np.sin(r)], axis=-1)
    # the actor moving at -heading, unit speed, relative to the ego: the interval is in metres of ego travel
    lo, hi = _sweep(_corners_of(ego), _corners_of(actor), -h)
    in_path = (lo <= hi) & (hi >= 0)
    overlapping = in_path & (lo <= 0)
    v_rel = np.sum((ego[:, [VX, VY]] - actor[:, [VX, VY]]) * h, axis=-1)
    with np.errstate(divide='ignore'):
        d = np.where(in_path & (v_rel > 0), v_rel ** 2 / (2.0 * np.maximum(lo, 0.0)), 0.0)
    return np.where(overlapping, np.inf, d)


def required_deceleration(ego_state, actor_state) -> float:
    """DRAC (m/s²) the ego needs to stop closing on an actor in its path: v_rel² / (2 · gap) along its heading; 0 if
    nothing in the path or not closing, inf if the boxes overlap."""
    return float(_drac_rows(as_traj(ego_state), as_traj(actor_state))[0])


def required_deceleration_series(ego_traj, actor_traj):
    """(t, drac) over the steps both exist."""
    e, a = as_traj(ego_traj), as_traj(actor_traj)
    ie, ia = _align(e, a)
    return e[ie, T], _drac_rows(e[ie], a[ia])


def _impulse(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return abs(float(v))
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def collisions(ego_traj, actor_traj=None, events=None, actor_id=None) -> dict:
    """{'times': [t where the boxes overlap], 'events': [{'t', 'other_id', 'impulse'}], 'first_t'}. events are the
    collision sensor's (t, other_id, impulse) — impulse a scalar or a 3-vector (its norm, N·s); with actor_id only
    that actor's events are kept."""
    times = []
    if actor_traj is not None:
        e, a = as_traj(ego_traj), as_traj(actor_traj)
        ie, ia = _align(e, a)
        if len(ie):
            hit = _overlap(_corners_of(e[ie]), _corners_of(a[ia]))
            times = [float(t) for t in e[ie[hit], T]]
    evs = []
    for ev in events or ():
        t, other, imp = (list(ev) + [None, None])[:3]
        if actor_id is not None and str(other) != str(actor_id):
            continue
        evs.append({'t': float(t), 'other_id': other, 'impulse': _impulse(imp)})
    firsts = times[:1] + [min(ev['t'] for ev in evs)] if evs else times[:1]
    return {'times': times, 'events': evs, 'first_t': min(firsts) if firsts else None}


def _resample(traj: np.ndarray, dt: float) -> np.ndarray:
    """Linear interpolation of a trajectory onto a dt grid over its own span (yaw unwrapped)."""
    t = traj[:, T]
    if len(t) < 2 or dt <= 0:
        return traj
    n = int(math.floor((t[-1] - t[0]) / dt + 1e-9)) + 1
    tg = t[0] + dt * np.arange(n)
    out = np.empty((n, 8))
    out[:, T] = tg
    yaw = np.degrees(np.unwrap(np.radians(traj[:, YAW])))
    for c in range(1, 8):
        out[:, c] = np.interp(tg, t, yaw if c == YAW else traj[:, c])
    return out


def _footprint_runs(traj: np.ndarray, tol: float = 0.01):
    """Footprints merged while the box stays put (within tol m at every corner): corners (K, 4, 2), occupancy
    interval starts and ends (K,) — so a long stop is one footprint over a time interval, not thousands."""
    C = _corners_of(traj)
    t = traj[:, T]
    starts = [0]
    for i in range(1, len(C)):
        if np.max(np.abs(C[i] - C[starts[-1]])) > tol:
            starts.append(i)
    idx = np.asarray(starts)
    ends = np.append(idx[1:] - 1, len(C) - 1)
    return C[idx], t[idx], t[ends]


def pet_detail(ego_traj, actor_traj, dt: float | None = None) -> dict | None:
    """PET with where and who: {'pet', 'first' ('ego', 'actor', or None when both occupy the same place at once),
    't_first_leaves', 't_second_enters', 'x', 'y' (a point of the conflict area)}; None if the paths never overlap.
    dt: resolution (the trajectories are interpolated to it when finer than their sampling)."""
    e, a = as_traj(ego_traj), as_traj(actor_traj)
    if dt:
        e, a = _resample(e, dt), _resample(a, dt)
    CE, e0, e1 = _footprint_runs(e)
    CA, a0, a1 = _footprint_runs(a)
    if not len(CE) or not len(CA):
        return None
    ce, ca = CE.mean(axis=1), CA.mean(axis=1)
    re = np.linalg.norm(CE - ce[:, None, :], axis=-1).max(-1)
    ra = np.linalg.norm(CA - ca[:, None, :], axis=-1).max(-1)
    gi, gj, gg = [], [], []
    for s in range(0, len(CE), 256):          # bounding-circle prefilter, in blocks of ego footprints
        d = np.linalg.norm(ce[s:s + 256, None, :] - ca[None, :, :], axis=-1)
        i, j = np.nonzero(d <= re[s:s + 256, None] + ra[None, :] + EPS)
        i = i + s
        gi.append(i)
        gj.append(j)
        gg.append(np.maximum.reduce([np.zeros(len(i)), a0[j] - e1[i], e0[i] - a1[j]]))
    i, j, gap = np.concatenate(gi), np.concatenate(gj), np.concatenate(gg)
    order = np.argsort(gap, kind='stable')
    for s in range(0, len(order), 4096):      # in order of time gap: the first overlapping pair is the PET
        k = order[s:s + 4096]
        hit = _overlap(CE[i[k]], CA[j[k]])
        if not hit.any():
            continue
        m = k[int(np.argmax(hit))]
        ii, jj = i[m], j[m]
        p = np.vstack([CE[ii], CA[jj]]).mean(axis=0)
        if gap[m] <= 0:
            first, t_out, t_in = None, float(max(e0[ii], a0[jj])), float(max(e0[ii], a0[jj]))
        elif e1[ii] < a0[jj]:
            first, t_out, t_in = 'ego', float(e1[ii]), float(a0[jj])
        else:
            first, t_out, t_in = 'actor', float(a1[jj]), float(e0[ii])
        return {'pet': float(gap[m]), 'first': first, 't_first_leaves': t_out, 't_second_enters': t_in,
                'x': float(p[0]), 'y': float(p[1])}
    return None


def pet(ego_traj, actor_traj, dt: float | None = None) -> float | None:
    """Post-encroachment time (s); None if the swept paths never overlap. See pet_detail."""
    d = pet_detail(ego_traj, actor_traj, dt)
    return None if d is None else d['pet']


def _finite(v):
    return None if v is None or not math.isfinite(v) else float(v)


def criticality_terms(min_ttc=None, pet_s=None, max_drac=None, min_dist=None) -> dict:
    """The four terms of the criticality scalar (heuristic; see the module docstring); None or inf counts 0."""
    def ok(v):
        return v is not None and math.isfinite(v)
    return {'s_ttc': math.exp(-max(min_ttc, 0.0) / TTC_SCALE_S) if ok(min_ttc) else 0.0,
            's_pet': math.exp(-max(pet_s, 0.0) / PET_SCALE_S) if ok(pet_s) else 0.0,
            's_drac': 1.0 - math.exp(-max(max_drac, 0.0) / DRAC_SCALE_MPS2) if ok(max_drac) else 0.0,
            's_dist': math.exp(-max(min_dist, 0.0) / DIST_SCALE_M) if ok(min_dist) else 0.0}


def criticality_score(terms: dict, collided: bool) -> float:
    """1 on a collision, else 0.9 · noisy-or of the terms (heuristic)."""
    if collided:
        return 1.0
    q = 1.0
    for v in terms.values():
        q *= 1.0 - min(max(v, 0.0), 1.0)
    return NEAR_MISS_CAP * (1.0 - q)


def summarise(ego_traj, actors: dict, dt: float | None = None, events=None, horizon: float = 10.0) -> dict:
    """Per actor and overall: min TTC and when, PET, min distance and when, max DRAC and when, collisions, and the
    criticality scalar. JSON-ready: an absent measure (TTC never within the horizon, paths never crossing) is None.
    actors: {id: trajectory}; dt: PET resolution; events: collision sensor events (t, other_id, impulse)."""
    e = as_traj(ego_traj)
    per = {}
    for aid, traj in (actors or {}).items():
        a = as_traj(traj)
        t, tt = ttc_series(e, a, horizon)
        k = int(np.argmin(tt)) if len(tt) else None
        min_ttc = float(tt[k]) if k is not None and math.isfinite(tt[k]) else None
        p = pet_detail(e, a, dt)
        dmin, dmin_t = min_distance(e, a)
        td, dr = required_deceleration_series(e, a)
        fin = np.where(np.isfinite(dr), dr, -1.0)
        kd = int(np.argmax(fin)) if len(fin) and fin.max() > 0 else None
        col = collisions(e, a, events, actor_id=aid)
        collided = bool(col['times'] or col['events'])
        terms = criticality_terms(min_ttc, p['pet'] if p else None, float(dr[kd]) if kd is not None else 0.0,
                                  dmin)
        imps = [ev['impulse'] for ev in col['events'] if ev['impulse'] is not None]
        per[aid] = {
            'min_ttc_s': min_ttc, 'min_ttc_t': float(t[k]) if min_ttc is not None else None,
            'pet_s': p['pet'] if p else None, 'pet_first': p['first'] if p else None,
            'pet_t': p['t_second_enters'] if p else None,
            'min_distance_m': _finite(dmin), 'min_distance_t': dmin_t,
            'max_drac_mps2': float(dr[kd]) if kd is not None else 0.0,
            'max_drac_t': float(td[kd]) if kd is not None else None,
            'collision': collided, 'collision_t': col['first_t'], 'collision_steps': len(col['times']),
            'collision_impulse': max(imps) if imps else None,
            'terms': terms, 'criticality': criticality_score(terms, collided)}
    known = {str(k) for k in per}
    other = [{'t': float(ev[0]), 'other_id': ev[1], 'impulse': _impulse(ev[2] if len(ev) > 2 else None)}
             for ev in events or () if str(ev[1]) not in known]

    def pick(key, fn):
        vals = [(v[key], aid) for aid, v in per.items() if v[key] is not None]
        return fn(vals, key=lambda v: v[0]) if vals else (None, None)

    min_ttc, ttc_aid = pick('min_ttc_s', min)
    min_pet, pet_aid = pick('pet_s', min)
    min_d, d_aid = pick('min_distance_m', min)
    max_dr, dr_aid = pick('max_drac_mps2', max)
    dr_aid = dr_aid if max_dr else None
    collided = any(v['collision'] for v in per.values()) or bool(other)
    crit = 1.0 if collided else max([v['criticality'] for v in per.values()] or [0.0])
    critical = max(per, key=lambda k: per[k]['criticality']) if per else None
    col_ts = [v['collision_t'] for v in per.values() if v['collision_t'] is not None] + [o['t'] for o in other]
    return {'actors': per,
            'overall': {'min_ttc_s': min_ttc, 'min_ttc_actor': ttc_aid, 'min_pet_s': min_pet, 'min_pet_actor': pet_aid,
                        'min_distance_m': min_d, 'min_distance_actor': d_aid, 'max_drac_mps2': max_dr,
                        'max_drac_actor': dr_aid, 'collision': collided,
                        'collision_t': min(col_ts) if col_ts else None,
                        'collided_with': sorted({str(k) for k, v in per.items() if v['collision']} |
                                                {str(o['other_id']) for o in other}),
                        'critical_actor': critical},
            'other_collisions': other,
            'criticality': crit, 'criticality_formula': CRITICALITY_FORMULA}
