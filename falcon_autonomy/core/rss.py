"""Responsibility-Sensitive Safety (numpy): safe distances, dangerous situations, proper responses and blame, after
Shalev-Shwartz, Shammah & Shashua 2017, "On a Formal Model of Safe and Scalable Self-driving Cars", arXiv:1708.06374
(v6, 27 Oct 2018; the definition and lemma numbers below are that version's) — and a scenario run's failure labelled
by who broke it: the ego (an avoidable failure, a planner blind spot), the other road user, or nobody it could answer
for (kinematically unavoidable).

Safe distances (longitudinal: from the rear car's front-most point to the front car's rear-most; lateral: between
the facing sides; [x]+ = max(x, 0)):

* same direction, Lemma 2 (Definition 1) — rear car c_r at v_r, front car c_f at v_f:
      d_min = [ v_r ρ + ½ a_max,accel ρ² + (v_r + ρ a_max,accel)² / (2 a_min,brake) − v_f² / (2 a_max,brake) ]+
* opposite directions, Lemma 3 (Definition 2) — c1 in the lane's direction (v1 ≥ 0), c2 against it (v2 < 0),
  v1,ρ = v1 + ρ a_max,accel, v2,ρ = |v2| + ρ a_max,accel:
      d_min = (v1 + v1,ρ)/2 ρ + v1,ρ² / (2 a_min,brake,correct) + (|v2| + v2,ρ)/2 ρ + v2,ρ² / (2 a_min,brake)
* lateral, Lemma 4 (Definition 6) — c1 left of c2, lateral velocities positive to the right, v1,ρ = v1 + ρ a_lat,
  v2,ρ = v2 − ρ a_lat:
      d_min = μ + [ (v1 + v1,ρ)/2 ρ + v1,ρ|v1,ρ| / (2 a_lat,brake) − ((v2 + v2,ρ)/2 ρ + v2,ρ|v2,ρ| / (2 a_lat,brake)) ]+
  The paper writes v1,ρ² and −v2,ρ², the case it states (both moving towards each other after ρ); the signed
  v|v| keeps a car still moving away after ρ from being counted as approaching — equal to the paper's in its case
  (unit-tested). Each car's own ρ and lateral limits are used (the paper's are common to both).

Dangerous situations (Definitions 3, 7, 9): a time is longitudinally (laterally) dangerous when that distance is not
safe; dangerous when both are. Its Danger Threshold t_b = max(t_b^long, t_b^lat), each the start of the unbroken
unsafe run up to it. Proper response (Definition 10, from Definitions 4 and 8): if t_b = t_b^long, longitudinal —
the rear car accelerates at most a_max,accel in [t_b, t_b + ρ) and brakes at least a_min,brake from t_b + ρ until
safe (or stopped); the front car brakes at most a_max,brake; against the direction both brake from t_b + ρ until
they stop (a_min,brake,correct for the car in the lane's direction). If t_b = t_b^lat, lateral — each car's lateral
acceleration stays within a_lat,accel in [t_b, t_b + ρ); after that each brakes laterally (at least a_lat,brake)
until its μ-lateral velocity towards the other is 0 (Definition 5; here: within LAT_VEL_TOL). Both apply when both
thresholds coincide. Responsibility (Section 3.10): whoever did not comply.

Labels of a failure (label_run): the run's critical event — the first contact with the critical actor
(core.criticality), else its minimum TTC, else its minimum distance — and its Danger Threshold:

    avoidable    the ego did not comply with its proper response in [t_b, event] (it could have, and did not)
    other        the ego complied; the other road user did not (it cut in, braked harder than a_max,brake, …)
    unavoidable  the danger began with the other already inside the ego's stopping envelope (the longitudinal gap
                 at t_b below the ego's braking distance at a_max,brake — it appeared there), or the other exceeded
                 its kinematic feasibility limits (FEASIBILITY) around it, or both complied by these checks
    safe         the event was not dangerous in RSS's sense (a near miss by the criticality scalar only)

Frame and limits. A scenario run starts on a straight road (the director: yaw within 10° over 150 m), so the road
axis is the ego's mean heading over the run when its heading spreads ≤ AXIS_SPREAD_DEG, else the ego's heading
averaged over the 2 s before each step; lateral velocities are relative to it. The ego is taken to drive in the lane's
direction (it does in every built-in), so an oncoming actor is the one against it. Junction geometry (Definitions
14–17: routes of different geometry, priority), occlusion (Section 3.9) and evasive manoeuvres (Definitions 11–13)
are not evaluated: a crossing at a junction is judged on the straight axis. A walker is a road user with the
pedestrian parameters (the paper's own example, Section 3.8: ρ = 0.5 s, acceleration and braking 2 m/s²) in both
directions; footprints are boxes; accelerations are differences of the recorded velocities (ACCEL_TOL absorbs their
noise). Every default is cited or labelled heuristic (PARAMS, FEASIBILITY); none is a regulation.
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA = 'falcon.rss/1'
# Parameters. Vehicles: the conservative starting values of Intel's ad-rss-lib parameter discussion
# (intel.github.io/ad-rss-lib, "Appendix: Parameter Discussion", drawn from the RSS paper; the 4 and 8 m/s² brakes a
# German driving-school rule of thumb) — heuristic, open for discussion there too. Walkers: the paper's example
# (Section 3.8). μ: ad-rss-lib's 10 cm lateral fluctuation margin.
PARAMS = {
    'rho_ego_s': 1.0, 'rho_other_s': 2.0,
    'a_max_accel': 3.5, 'a_min_brake': 4.0, 'a_max_brake': 8.0, 'a_min_brake_correct': 3.0,
    'a_lat_max_accel': 0.2, 'a_lat_min_brake': 0.8, 'mu_m': 0.1,
    'walker_rho_s': 0.5, 'walker_a_accel': 2.0, 'walker_a_brake': 2.0,
}
PARAM_SOURCES = {
    'vehicles': 'ad-rss-lib parameter discussion (conservative starting values, from Shalev-Shwartz et al. 2017; '
                'heuristic)',
    'walkers': 'Shalev-Shwartz et al. 2017, Section 3.8 example: ρ 0.5 s, acceleration and braking 2 m/s²',
    'mu_m': 'ad-rss-lib lateral fluctuation margin, 10 cm (heuristic)'}
# Kinematic feasibility of a scenario's adversary (heuristic, labelled): accelerating ≤ 4 m/s² (a brisk car; a
# stand-in's own class limit is lower and the director already keeps it), braking ≤ 9 m/s² (≈ 0.9 g, a dry road's
# tyre limit), lateral acceleration ≤ 4 m/s² and jerk ≤ 8 m/s³ (Falcon's adversarial-traffic realism bounds,
# core.advfeatures.REALISM_LIMITS), each tolerated for budget_s per run (core.advfeatures.DYN_BUDGET_S).
FEASIBILITY = {'a_accel_mps2': 4.0, 'a_brake_mps2': 9.0, 'a_lat_mps2': 4.0, 'jerk_mps3': 8.0, 'budget_s': 0.5,
               'lane_width_m': 3.5}
ACCEL_TOL = 0.5           # m/s²: compliance checks allow this much (differences of recorded velocities; heuristic)
LAT_VEL_TOL = 0.1         # m/s: a lateral velocity this small counts as μ-lateral velocity 0 (heuristic)
STOPPED_MPS = 0.2         # m/s: a car slower than this counts as stopped
AXIS_SPREAD_DEG = 15.0
LABELS = ('avoidable', 'other', 'unavoidable', 'safe')
COUNTED = ('avoidable',)
NON_AVOIDABLE_CAP = 0.45   # heuristic: the RSS objective caps a run whose event is not avoidable here — below the
                           # default failure threshold 0.5, so it is not a failure and the search moves on; a lower
                           # threshold than the cap counts such runs again (count "all" instead)
WALKER_KINDS = ('pedestrian', 'child')


# ================================================================================================ the formulas
def safe_longitudinal_same(v_r: float, v_f: float, rho: float, a_max_accel: float, a_min_brake: float,
                           a_max_brake: float) -> float:
    """Lemma 2: the minimal safe distance from the rear car's front to the front car's rear (m)."""
    v_r, v_f = max(float(v_r), 0.0), max(float(v_f), 0.0)
    v_rho = v_r + rho * a_max_accel
    d = v_r * rho + 0.5 * a_max_accel * rho ** 2 + v_rho ** 2 / (2.0 * a_min_brake) - v_f ** 2 / (2.0 * a_max_brake)
    return max(d, 0.0)


def safe_longitudinal_opposite(v1: float, v2: float, rho1: float, rho2: float, a_max_accel: float,
                               a_min_brake: float, a_min_brake_correct: float, a_max_accel2: float | None = None) \
        -> float:
    """Lemma 3: c1 drives in the lane's direction at v1 ≥ 0, c2 against it at v2 (its sign ignored: |v2|); each
    with its own response time (the paper's ρ is common)."""
    a2 = a_max_accel if a_max_accel2 is None else a_max_accel2
    v1 = max(float(v1), 0.0)
    v2 = abs(float(v2))
    v1r, v2r = v1 + rho1 * a_max_accel, v2 + rho2 * a2
    return (v1 + v1r) / 2.0 * rho1 + v1r ** 2 / (2.0 * a_min_brake_correct) + \
        (v2 + v2r) / 2.0 * rho2 + v2r ** 2 / (2.0 * a_min_brake)


def safe_lateral(v1: float, v2: float, mu: float, rho1: float, a_lat1: float, brake1: float,
                 rho2: float | None = None, a_lat2: float | None = None, brake2: float | None = None) -> float:
    """Lemma 4 (signed: see the module doc): c1 is left of c2, lateral velocities positive to the right."""
    rho2 = rho1 if rho2 is None else rho2
    a_lat2 = a_lat1 if a_lat2 is None else a_lat2
    brake2 = brake1 if brake2 is None else brake2
    v1r = v1 + rho1 * a_lat1
    v2r = v2 - rho2 * a_lat2
    c1 = (v1 + v1r) / 2.0 * rho1 + v1r * abs(v1r) / (2.0 * brake1)
    c2 = (v2 + v2r) / 2.0 * rho2 + v2r * abs(v2r) / (2.0 * brake2)
    return mu + max(c1 - c2, 0.0)


def braking_distance(v: float, a: float) -> float:
    return max(float(v), 0.0) ** 2 / (2.0 * a)


# ================================================================================================ trajectories
def _wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def road_axis(ego: np.ndarray, dt: float) -> tuple:
    """(heading per step in degrees, how it was chosen): the ego's mean heading when it spreads ≤ AXIS_SPREAD_DEG
    over the run (a straight road), else its heading averaged over the 2 s before each step."""
    yaw = np.radians(ego[:, 3])
    mean = math.degrees(math.atan2(np.sin(yaw).mean(), np.cos(yaw).mean()))
    spread = float(np.max(np.abs(_wrap(ego[:, 3] - mean)))) if len(ego) else 0.0
    if spread <= AXIS_SPREAD_DEG:
        return np.full(len(ego), mean), f'the ego\'s mean heading ({mean:.1f}°; spread {spread:.1f}°)'
    k = max(int(round(2.0 / dt)), 1)
    out = np.empty(len(ego))
    for i in range(len(ego)):
        s = yaw[max(i - k, 0):i + 1]
        out[i] = math.degrees(math.atan2(np.sin(s).mean(), np.cos(s).mean()))
    return out, f'the ego\'s heading over the previous 2 s (its heading spreads {spread:.1f}° over the run)'


def _align(ego: np.ndarray, a: np.ndarray):
    """The rows of ego and actor at common times (an actor that appears late is compared where both exist)."""
    te = np.round(ego[:, 0], 4)
    ta = np.round(a[:, 0], 4)
    common, ie, ia = np.intersect1d(te, ta, return_indices=True)
    return ie, ia


def _half_extents(len_, wid, dyaw_rad):
    """A box's half extents along a frame turned dyaw from it."""
    c, s = np.abs(np.cos(dyaw_rad)), np.abs(np.sin(dyaw_rad))
    return c * len_ / 2.0 + s * wid / 2.0, s * len_ / 2.0 + c * wid / 2.0


def _accel(v: np.ndarray, dt: float) -> np.ndarray:
    """Acceleration from a velocity series: central differences (one-sided at the ends)."""
    if len(v) < 2:
        return np.zeros(len(v))
    return np.gradient(v, dt)


def geometry(ego: np.ndarray, actor: np.ndarray, dt: float, walker: bool = False, params: dict | None = None,
             axis: np.ndarray | None = None) -> dict:
    """Per common step: gaps, velocities in the road frame, safe distances and the danger flags of the ego and one
    actor. Arrays (all the same length) plus the step indices into the ego's trajectory."""
    P = dict(PARAMS, **(params or {}))
    ie, ia = _align(ego, actor)
    E, A = ego[ie], actor[ia]
    ax = (axis if axis is not None else road_axis(ego, dt)[0])[ie]
    r = np.radians(ax)
    f = np.stack([np.cos(r), np.sin(r)], 1)                         # along the road
    n = np.stack([-np.sin(r), np.cos(r)], 1)                        # to its right (CARLA: y right of x)
    d = A[:, 1:3] - E[:, 1:3]
    lon = (d * f).sum(1)                                            # actor ahead: positive
    lat = (d * n).sum(1)                                            # actor to the right: positive
    ve, va = E[:, 4:6], A[:, 4:6]
    ve_lon, ve_lat = (ve * f).sum(1), (ve * n).sum(1)
    va_lon, va_lat = (va * f).sum(1), (va * n).sum(1)
    ehx, ehy = _half_extents(E[:, 6], E[:, 7], np.radians(_wrap(E[:, 3] - ax)))
    ahx, ahy = _half_extents(A[:, 6], A[:, 7], np.radians(_wrap(A[:, 3] - ax)))
    gap_lon = np.abs(lon) - ehx - ahx
    gap_lat = np.abs(lat) - ehy - ahy
    rho_o = P['walker_rho_s'] if walker else P['rho_other_s']
    acc_o = P['walker_a_accel'] if walker else P['a_max_accel']
    brk_o = P['walker_a_brake'] if walker else P['a_min_brake']
    lat_acc_o = P['walker_a_accel'] if walker else P['a_lat_max_accel']
    lat_brk_o = P['walker_a_brake'] if walker else P['a_lat_min_brake']
    N = len(ie)
    dmin_lon = np.zeros(N)
    dmin_lat = np.zeros(N)
    relation = np.empty(N, dtype=object)
    for k in range(N):
        ahead = lon[k] > 0
        if ahead and va_lon[k] < -STOPPED_MPS and not walker:       # oncoming: Lemma 3 (the ego in the lane's way)
            relation[k] = 'oncoming'
            dmin_lon[k] = safe_longitudinal_opposite(ve_lon[k], va_lon[k], P['rho_ego_s'], rho_o, P['a_max_accel'],
                                                     brk_o, P['a_min_brake_correct'], acc_o)
        elif ahead:                                                  # the ego is the rear car
            relation[k] = 'ego_behind'
            dmin_lon[k] = safe_longitudinal_same(ve_lon[k], max(va_lon[k], 0.0), P['rho_ego_s'], P['a_max_accel'],
                                                 P['a_min_brake'], P['a_max_brake'] if not walker
                                                 else P['walker_a_brake'])
        elif va_lon[k] > ve_lon[k] - 1e-9 or va_lon[k] > STOPPED_MPS:  # the actor behind, same way: it is the rear
            relation[k] = 'ego_ahead'
            dmin_lon[k] = safe_longitudinal_same(max(va_lon[k], 0.0), ve_lon[k], rho_o, acc_o, brk_o,
                                                 P['a_max_brake'])
        else:                                                        # behind and going away (or backwards)
            relation[k] = 'apart'
            dmin_lon[k] = 0.0
        # lateral: c1 the left one
        if lat[k] >= 0:                                              # the actor is to the ego's right
            dmin_lat[k] = safe_lateral(ve_lat[k], va_lat[k], P['mu_m'], P['rho_ego_s'], P['a_lat_max_accel'],
                                       P['a_lat_min_brake'], rho_o, lat_acc_o, lat_brk_o)
        else:
            dmin_lat[k] = safe_lateral(va_lat[k], ve_lat[k], P['mu_m'], rho_o, lat_acc_o, lat_brk_o,
                                       P['rho_ego_s'], P['a_lat_max_accel'], P['a_lat_min_brake'])
    lon_unsafe = gap_lon < dmin_lon
    lat_unsafe = gap_lat < dmin_lat
    return {'t': E[:, 0], 'i_ego': ie, 'i_actor': ia, 'lon': lon, 'lat': lat, 'gap_lon': gap_lon, 'gap_lat': gap_lat,
            've_lon': ve_lon, 've_lat': ve_lat, 'va_lon': va_lon, 'va_lat': va_lat, 'dmin_lon': dmin_lon,
            'dmin_lat': dmin_lat, 'lon_unsafe': lon_unsafe, 'lat_unsafe': lat_unsafe,
            'dangerous': lon_unsafe & lat_unsafe, 'relation': relation, 'rho_other': rho_o, 'walker': walker}


def _run_start(flags: np.ndarray, k: int) -> int:
    """The first index of the unbroken True run of flags ending at k."""
    j = k
    while j > 0 and flags[j - 1]:
        j -= 1
    return j


def danger_threshold(g: dict, k: int) -> dict | None:
    """Definition 9 at step k (an index into g): t_b, t_b^long, t_b^lat (as indices) and which response applies."""
    if not g['dangerous'][k]:
        return None
    jl = _run_start(g['lon_unsafe'], k)
    jt = _run_start(g['lat_unsafe'], k)
    jb = max(jl, jt)
    return {'k_b': jb, 'k_long': jl, 'k_lat': jt, 'longitudinal': jb == jl, 'lateral': jb == jt,
            't_b': float(g['t'][jb])}


def _over(x: np.ndarray, limit: float, dt: float, budget: float) -> float:
    """Seconds by which x exceeds limit beyond budget (0: within)."""
    return max(float(np.count_nonzero(x > limit)) * dt - budget, 0.0)


def feasibility(actor: np.ndarray, dt: float, limits: dict | None = None, t0: float | None = None,
                t1: float | None = None) -> dict:
    """The actor's measured dynamics against FEASIBILITY over [t0, t1]: peaks and whether it stayed within them
    (each limit tolerated for budget_s)."""
    L = dict(FEASIBILITY, **(limits or {}))
    A = actor
    if t0 is not None:
        A = A[A[:, 0] >= t0 - 1e-9]
    if t1 is not None:
        A = A[A[:, 0] <= t1 + 1e-9]
    if len(A) < 3:
        return {'ok': True, 'checked_s': 0.0}
    v = np.hypot(A[:, 4], A[:, 5])
    yaw = np.unwrap(np.radians(A[:, 3]))
    a_lon = _accel(v, dt)
    a_lat = v * np.gradient(yaw, dt)
    jerk = _accel(a_lon, dt)
    over = {'accel': _over(a_lon, L['a_accel_mps2'] + ACCEL_TOL, dt, L['budget_s']),
            'brake': _over(-a_lon, L['a_brake_mps2'] + ACCEL_TOL, dt, L['budget_s']),
            'lateral': _over(np.abs(a_lat), L['a_lat_mps2'] + ACCEL_TOL, dt, L['budget_s']),
            'jerk': _over(np.abs(jerk), L['jerk_mps3'], dt, L['budget_s'])}
    return {'ok': not any(v > 0 for v in over.values()), 'over_s': {k: round(v, 3) for k, v in over.items() if v},
            'peak': {'accel': round(float(a_lon.max()), 3), 'brake': round(float(-a_lon.min()), 3),
                     'lateral': round(float(np.abs(a_lat).max()), 3), 'jerk': round(float(np.abs(jerk).max()), 3)},
            'checked_s': round(float(A[-1, 0] - A[0, 0]), 3)}


def compliance(g: dict, ego: np.ndarray, actor: np.ndarray, dt: float, thr: dict, k_end: int,
               params: dict | None = None) -> dict:
    """Did the ego and the actor comply with the proper response from the Danger Threshold to step k_end?
    {'ego': [violations], 'actor': [violations]}, each {'what', 't', 'value', 'limit'}."""
    P = dict(PARAMS, **(params or {}))
    out = {'ego': [], 'actor': []}
    kb = thr['k_b']
    t = g['t']
    ae = _accel(g['ve_lon'], dt)
    aa = _accel(g['va_lon'], dt)
    ale = _accel(g['ve_lat'], dt)
    ala = _accel(g['va_lat'], dt)
    rho_e, rho_o = P['rho_ego_s'], g['rho_other']
    rel = g['relation'][kb]
    walker = g['walker']

    def add(who, what, k, value, limit):
        if not any(v['what'] == what for v in out[who]):
            out[who].append({'what': what, 't': round(float(t[k]), 3), 'value': round(float(value), 3),
                             'limit': round(float(limit), 3)})
    for k in range(kb, k_end + 1):
        since = t[k] - t[kb]
        if thr['longitudinal']:
            if rel == 'ego_behind':
                if since < rho_e:
                    if ae[k] > P['a_max_accel'] + ACCEL_TOL:
                        add('ego', 'accelerated beyond a_max,accel in the response time', k, ae[k], P['a_max_accel'])
                elif g['ve_lon'][k] > STOPPED_MPS and g['lon_unsafe'][k] and ae[k] > -P['a_min_brake'] + ACCEL_TOL:
                    add('ego', 'braked less than a_min,brake after the response time', k, -ae[k], P['a_min_brake'])
                lim = P['walker_a_brake'] if walker else P['a_max_brake']
                if aa[k] < -(lim + ACCEL_TOL) and g['va_lon'][k] > STOPPED_MPS:
                    add('actor', 'the front road user braked harder than a_max,brake', k, -aa[k], lim)
            elif rel == 'ego_ahead':
                if ae[k] < -(P['a_max_brake'] + ACCEL_TOL) and g['ve_lon'][k] > STOPPED_MPS:
                    add('ego', 'braked harder than a_max,brake in front of a follower', k, -ae[k], P['a_max_brake'])
                if since >= rho_o and g['va_lon'][k] > STOPPED_MPS and g['lon_unsafe'][k] and \
                        aa[k] > -P['a_min_brake'] + ACCEL_TOL:
                    add('actor', 'the follower braked less than a_min,brake after its response time', k, -aa[k],
                        P['a_min_brake'])
            elif rel == 'oncoming':
                if since >= rho_e and g['ve_lon'][k] > STOPPED_MPS and ae[k] > -P['a_min_brake_correct'] + ACCEL_TOL:
                    add('ego', 'did not brake (a_min,brake,correct) for an oncoming road user', k, -ae[k],
                        P['a_min_brake_correct'])
                if since >= rho_o and abs(g['va_lon'][k]) > STOPPED_MPS and aa[k] < P['a_min_brake'] - ACCEL_TOL:
                    add('actor', 'the wrong-way road user did not brake (a_min,brake)', k, aa[k], P['a_min_brake'])
        if thr['lateral']:
            side = 1.0 if g['lat'][kb] >= 0 else -1.0                # the actor's side of the ego (+: right)
            ve_to = g['ve_lat'][k] * side                            # the ego's lateral velocity towards it
            va_to = -g['va_lat'][k] * side                           # the actor's towards the ego
            lim_o = P['walker_a_accel'] if walker else P['a_lat_max_accel']
            if since < rho_e:
                if abs(ale[k]) > P['a_lat_max_accel'] + ACCEL_TOL:
                    add('ego', 'lateral acceleration beyond a_lat,accel in the response time', k, abs(ale[k]),
                        P['a_lat_max_accel'])
            else:
                v0 = max(g['ve_lat'][min(kb + int(round(rho_e / dt)), k)] * side, 0.0)
                allowed = max(v0 - P['a_lat_min_brake'] * (since - rho_e), 0.0) + LAT_VEL_TOL
                if ve_to > allowed:
                    add('ego', 'kept moving towards it laterally (no lateral braking to μ-lateral velocity 0)', k,
                        ve_to, allowed)
            if since < rho_o:
                if abs(ala[k]) > lim_o + ACCEL_TOL:
                    add('actor', 'lateral acceleration beyond its a_lat,accel in the response time', k, abs(ala[k]),
                        lim_o)
            else:
                ko = min(kb + int(round(rho_o / dt)), k)
                v0 = max(-g['va_lat'][ko] * side, 0.0)
                brk = P['walker_a_brake'] if walker else P['a_lat_min_brake']
                allowed = max(v0 - brk * (since - rho_o), 0.0) + LAT_VEL_TOL
                if va_to > allowed:
                    add('actor', 'kept moving towards the ego laterally (no lateral braking)', k, va_to, allowed)
    return out


def event_of(crit: dict, actor_id: str) -> tuple:
    """(time, kind) of an actor's critical event in a criticality summary: first contact, else min TTC, else min
    distance."""
    a = (crit.get('actors') or {}).get(actor_id) or {}
    if a.get('collision_t') is not None:
        return float(a['collision_t']), 'collision'
    if a.get('min_ttc_t') is not None:
        return float(a['min_ttc_t']), 'min_ttc'
    if a.get('min_distance_t') is not None:
        return float(a['min_distance_t']), 'min_distance'
    return None, None


def label_failure(ego, actor, dt: float, t_event: float, walker: bool = False, params: dict | None = None,
                  limits: dict | None = None) -> dict:
    """The RSS label of the ego's critical event with one actor at t_event (module doc)."""
    P = dict(PARAMS, **(params or {}))
    ego, actor = np.asarray(ego, float), np.asarray(actor, float)
    g = geometry(ego, actor, dt, walker, P)
    if not len(g['t']):
        return {'label': 'safe', 'reason': 'the actor and the ego never existed at the same time'}
    k = int(np.argmin(np.abs(g['t'] - t_event)))
    out = {'t_event': round(float(g['t'][k]), 3)}
    thr = danger_threshold(g, k)
    if thr is None:
        out.update(label='safe', reason='not dangerous in RSS\'s sense at the event (the longitudinal or the lateral '
                                        'distance was safe)',
                   gap_lon=round(float(g['gap_lon'][k]), 3), dmin_lon=round(float(g['dmin_lon'][k]), 3),
                   gap_lat=round(float(g['gap_lat'][k]), 3), dmin_lat=round(float(g['dmin_lat'][k]), 3))
        return out
    kb = thr['k_b']
    rel = g['relation'][kb]
    resp = ('longitudinal and lateral' if thr['longitudinal'] and thr['lateral'] else
            'longitudinal' if thr['longitudinal'] else 'lateral')
    out.update(t_b=round(thr['t_b'], 3), response=resp, relation=str(rel),
               at_t_b={'gap_lon': round(float(g['gap_lon'][kb]), 3), 'dmin_lon': round(float(g['dmin_lon'][kb]), 3),
                       'gap_lat': round(float(g['gap_lat'][kb]), 3), 'dmin_lat': round(float(g['dmin_lat'][kb]), 3),
                       'ego_speed': round(float(g['ve_lon'][kb]), 3)})
    comp = compliance(g, ego, actor, dt, thr, k, P)
    feas = feasibility(actor, dt, limits, thr['t_b'] - 2.0, float(g['t'][k])) if not walker else {'ok': True}
    envelope = braking_distance(g['ve_lon'][kb], P['a_max_brake'])
    # it appeared inside the envelope: the danger began with a gap the ego could not close to zero speed in even at
    # a_max,brake at once (an actor ahead of it, the danger not of the ego's own making: a lateral entry or both at once)
    inside = rel == 'ego_behind' and thr['lateral'] and g['gap_lon'][kb] < envelope
    out.update(ego_violations=comp['ego'], actor_violations=comp['actor'], actor_feasible=feas,
               stopping_envelope_m=round(envelope, 3))
    if inside:
        out.update(label='unavoidable', reason=f'it entered the ego\'s path {g["gap_lon"][kb]:.1f} m ahead, inside '
                                               f'its {envelope:.1f} m braking distance at a_max,brake')
    elif comp['ego']:
        out.update(label='avoidable', reason='the ego ' + comp['ego'][0]['what'])
    elif not feas['ok']:
        out.update(label='unavoidable', reason='the other exceeded its feasibility limits ('
                                               + ', '.join(sorted(feas['over_s'])) + ')')
    elif comp['actor']:
        out.update(label='other', reason=comp['actor'][0]['what'])
    else:
        out.update(label='unavoidable', reason='both complied with their proper responses by these checks')
    return out


def label_run(traj: dict, crit: dict, kinds: dict | None = None, params: dict | None = None,
              limits: dict | None = None) -> dict:
    """The RSS label of a scenario run (its trajectories.json and its criticality summary): the critical actor's
    event; kinds: {actor id: scenario kind} (walkers get the pedestrian parameters)."""
    dt = float(traj.get('dt') or 0.05)
    actors = traj.get('actors') or {}
    ego = np.asarray(actors.get('ego') or [], float)
    ov = (crit or {}).get('overall') or {}
    aid = None
    hit = ov.get('collided_with') or []
    for h in hit:
        if h in actors and h != 'ego':
            aid = h
            break
    aid = aid or ov.get('critical_actor')
    base = {'schema': SCHEMA, 'params': dict(PARAMS, **(params or {})), 'axis': None}
    if not len(ego) or aid is None or aid not in actors or not actors[aid]:
        return dict(base, label='safe', reason='no critical actor', actor=aid)
    t_ev, kind = event_of(crit, aid)
    if t_ev is None:
        return dict(base, label='safe', reason='no critical event', actor=aid)
    walker = (kinds or {}).get(aid) in WALKER_KINDS
    _, axis_how = road_axis(ego, dt)
    out = label_failure(ego, np.asarray(actors[aid], float), dt, t_ev, walker, params, limits)
    out.update(base, actor=aid, event=kind, walker=walker, axis=axis_how)
    out['counted'] = out['label'] in COUNTED
    return out


def kinds_of(scenario: dict | None) -> dict:
    """{actor id: kind} of a run's scenario (its own actors and those trigger rules added)."""
    sc = scenario or {}
    out = {a['id']: a.get('kind') for a in sc.get('actors') or [] if isinstance(a, dict) and 'id' in a}
    for a in sc.get('added') or []:
        out[a['id']] = a.get('kind')
    return out


def label_run_dir(run_dir: str, params: dict | None = None, limits: dict | None = None) -> dict | None:
    """The RSS label of a finished scenario run folder (None without trajectories or a scenario result)."""
    import json
    import os
    try:
        with open(os.path.join(run_dir, 'run.json'), encoding='utf-8') as fh:
            run = json.load(fh)
        with open(os.path.join(run_dir, 'trajectories.json'), encoding='utf-8') as fh:
            traj = json.load(fh)
    except (OSError, ValueError):
        return None
    sc = run.get('scenario') or {}
    crit = sc.get('criticality')
    if not crit:
        return None
    settings = (run.get('provenance') or {}).get('settings') or {}
    kinds = kinds_of(settings.get('scenario'))
    kinds.update(kinds_of(sc))
    return label_run(traj, crit, kinds, params, limits)


def rss_objective(criticality: float, label: dict | None) -> float:
    """The search's value counting avoidable failures only: the criticality when the run's event is avoidable (or it
    has no RSS label), else min(criticality, NON_AVOIDABLE_CAP) — continuous, so the search keeps its gradient below
    the cap, and a non-avoidable event (other, unavoidable, safe) never reaches the default threshold."""
    if not label or label.get('label') in COUNTED:
        return float(criticality)
    return min(float(criticality), NON_AVOIDABLE_CAP)


# ================================================================================================ feasibility of a search
def feasible_space(scn: dict, space: list, limits: dict | None = None) -> tuple:
    """The search space with each parameter that sets an adversary's dynamics directly narrowed to what is
    kinematically feasible (FEASIBILITY, heuristic): a brake's "decel" ≤ a_brake, a speed action's "accel" ≤ a_accel,
    a lane change's "duration_s" ≥ 2·√(lane width / a_lat) (the shortest lateral move of one lane at a_lat, accelerating
    then braking). Returns (space, [notes]). A parameter that leaves nothing feasible is left as it is, and said."""
    L = dict(FEASIBILITY, **(limits or {}))
    bounds = {}

    def ref(v):
        return v[1:] if isinstance(v, str) and v.startswith('$') and v[1:].isidentifier() else None
    t_min = 2.0 * math.sqrt(L['lane_width_m'] / L['a_lat_mps2'])
    for ev in scn.get('timeline') or []:
        for a in ev.get('actions') or []:
            do = a.get('do')
            if do == 'brake' and ref(a.get('decel')):
                bounds.setdefault(ref(a['decel']), {})['hi'] = L['a_brake_mps2']
            elif do == 'speed' and ref(a.get('accel')):
                bounds.setdefault(ref(a['accel']), {})['hi'] = L['a_accel_mps2']
            elif do == 'lane_change' and ref(a.get('duration_s')):
                bounds.setdefault(ref(a['duration_s']), {})['lo'] = t_min
    out, notes = [], []
    for p in space:
        b = bounds.get(p['name'])
        if not b or 'choices' in p:
            out.append(p)
            continue
        q = dict(p)
        lo, hi = max(p['lo'], b.get('lo', -math.inf)), min(p['hi'], b.get('hi', math.inf))
        if lo >= hi:
            notes.append(f'{p["name"]}: no value in {p["lo"]:g}…{p["hi"]:g} is within the feasibility limit '
                         f'({"≥ " + format(b["lo"], ".3g") if "lo" in b else "≤ " + format(b["hi"], ".3g")}); left '
                         'as it is')
            out.append(p)
            continue
        if (lo, hi) != (p['lo'], p['hi']):
            notes.append(f'{p["name"]}: {p["lo"]:g}…{p["hi"]:g} narrowed to {lo:.3g}…{hi:.3g} (feasibility: '
                         + ('a lane change ≥ ' + format(t_min, '.2f') + ' s' if 'lo' in b else
                            'at most ' + format(b['hi'], 'g') + ' m/s²') + ')')
        q['lo'], q['hi'] = lo, hi
        out.append(q)
    return out, notes
