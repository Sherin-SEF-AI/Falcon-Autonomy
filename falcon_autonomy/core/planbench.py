"""A closed-loop planner benchmark in the style of nuPlan's closed-loop score and NAVSIM's PDM score: fixed routes
(a world, a scenario, a pinned seed), any planner Falcon can drive, and the scoring as a pure function of what a run
recorded (numpy at import; the runner is falcon_autonomy.sim.benchrun).

Definitions (docs/planner-benchmark.md cites them and says where Falcon differs). A run's record is the ego's and
every scenario actor's (t, x, y, yaw°, vx, vy, length, width) in CARLA's world frame (trajectories.json, or a gym
episode's `trajectories`), the collision sensor's events, and the map's driving lanes (lanes.json or the OpenDRIVE
file, through core.rules.RoadMap). Per run:

    NC   no at-fault collisions        1; 0.5 when the only at-fault collisions are with static objects; 0 otherwise
    DAC  drivable-area compliance      0 when a corner of the ego's box is more than 0.3 m outside every driving
                                       lane at any sample, else 1
    DDC  driving-direction compliance  the distance driven on lanes of the other direction only, summed over any
                                       1 s window: ≤ 2 m → 1, ≤ 6 m → 0.5, more → 0
    TTC  time to collision within bound 0 when, at a sample with the ego moving (> 0.005 m/s), the box-exact
                                       constant-velocity TTC (core.criticality) to an actor not behind the ego and not
                                       already in contact is below 0.95 s; else 1
    EP   ego progress                  the ego's progress along the route (the reference run's path) over the
                                       reference's progress, clipped to [0, 1]; 1 when the reference progressed < 5 m
    C    comfort                       1 when longitudinal acceleration ∈ [−4.05, 2.40] m/s², |lateral acceleration|
                                       ≤ 4.89 m/s², |yaw rate| ≤ 0.95 rad/s, |yaw acceleration| ≤ 1.93 rad/s²,
                                       |longitudinal jerk| ≤ 4.13 m/s³ and |jerk| ≤ 8.37 m/s³ at every sample; else 0

    PDMS = NC × DAC × DDC × (5·TTC + 5·EP + 2·C) / 12          (Falcon's aggregate; DDC as nuPlan's multiplier)
    PDMS_navsim_v1 = NC × DAC × (5·TTC + 5·EP + 2·C) / 12      (NAVSIM v1's PDM score, for comparison)

A term that cannot be evaluated on a run (no map: DAC and DDC) is left out of the product or the weighted mean, and
the run's score says so ('partial'). At fault (nuPlan's collision classes, Falcon's approximation): the ego stopped
(< 0.05 m/s) — not at fault; the other stopped — at fault; otherwise the contact point (the corners of either box
inside the other, their mean) in the ego's frame: front (x > ½ half-length) at fault, rear (x < −½ half-length) not,
lateral at fault only when the ego's body crosses a lane line or leaves the lanes at that sample (nuPlan: "in
multiple lanes or a non-drivable area"); without a map a lateral contact counts as at fault. Each actor's collision
counts once (its first contact). Static objects: the proxies that do not drive (handcart, vendor) and sensor contacts
with something that is not a scenario actor; vehicles, two-wheelers, pedestrians and cattle are not.

Kinematics for comfort: velocity as recorded (vx, vy; the director's own for kinematic actors), then a local quadratic
fit over ±0.35 s (a Savitzky–Golay filter of 15 samples at 20 Hz) for each derivative.

Sets. A planner's score on a set is the mean over its routes of each term and of PDMS, with 95 % percentile bootstrap
intervals over routes (core.metrics.bootstrap); two planners are compared on the routes both drove by a paired
bootstrap (core.metrics.paired_bootstrap). Routes are independent units; with few routes the intervals are too narrow
(core.metrics.uncertainty's limits).
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

SCHEMA = 'falcon.planbench/1'
RESULTS_SCHEMA = 'falcon.planbench_results/1'
# nuPlan's metric thresholds (nuplan-devkit's closed-loop metric configuration, as its documentation states them)
COMFORT = {'lon_accel_min': -4.05, 'lon_accel_max': 2.40, 'lat_accel_abs': 4.89, 'yaw_rate_abs': 0.95,
           'yaw_accel_abs': 1.93, 'lon_jerk_abs': 4.13, 'jerk_abs': 8.37}
DAC_TOL_M = 0.3
DDC_WINDOW_S, DDC_OK_M, DDC_BAD_M = 1.0, 2.0, 6.0
TTC_BOUND_S, TTC_HORIZON_S, MOVING_MPS = 0.95, 1.0, 0.005
STOPPED_MPS = 0.05
EP_MIN_REF_M = 5.0              # Falcon's choice (module doc); NAVSIM and nuPlan use a small threshold of their own
ROUTE_LAT_M = 4.0               # a sample further than this from the route does not count as progress along it
WEIGHTS = {'ttc': 5.0, 'ep': 5.0, 'comfort': 2.0}
MULTIPLIERS = ('nc', 'dac', 'ddc')
TERMS = ('nc', 'dac', 'ddc', 'ttc', 'ep', 'comfort')
TERM_NAMES = {'nc': 'no at-fault collisions', 'dac': 'drivable-area compliance', 'ddc': 'driving-direction compliance',
              'ttc': 'time to collision within bound', 'ep': 'ego progress', 'comfort': 'comfort',
              'pdms': 'PDM score (Falcon: with DDC)', 'pdms_navsim_v1': 'PDM score (NAVSIM v1: without DDC)'}
SMOOTH_S = 0.35
STATIC_KINDS = ('handcart', 'vendor')
PLANNERS_DRIVE = ('follower', 'none', 'autopilot')
REFERENCE = 'follower'

# --------------------------------------------------------------------------------------------- the fixed routes
# A route is a world, a scenario and a pinned seed: the seed draws the scenario's parameters and the start (the
# director's seeded spawn order), so the same route is the same situation for every planner. Scenario actors move
# kinematically and there is no background traffic: CARLA 0.9.16's vehicle physics is not bit-for-bit repeatable
# (docs/scenarios.md), a kinematic scenario is.
_BLR = ('lead_brake', 'autorickshaw_cut_in', 'two_wheeler_filtering', 'jaywalker_behind_bus', 'wrong_way_scooter',
        'bus_pulling_out', 'stray_cattle', 'handcart_kerb_lane', 'stalled_vehicle', 'pedestrian_crossing')
_T10 = ('lead_brake', 'cut_in', 'pedestrian_crossing', 'cyclist_crossing', 'door_opening', 'stalled_vehicle',
        'unprotected_turn')
_T04 = ('lead_brake', 'cut_in', 'stalled_vehicle')


def _route(world, short, scn, seed):
    return {'id': f'{short}-{scn}-s{seed}', 'world': world, 'scenario': scn, 'seed': seed}


ROUTE_SETS = {
    'falcon-pdm-v1': ([_route('osm:bengaluru', 'blr', s, 11) for s in _BLR]
                      + [_route('osm:bengaluru', 'blr', 'autorickshaw_cut_in', 12)]
                      + [_route('town:Town10HD_Opt', 't10', s, 21) for s in _T10]
                      + [_route('town:Town10HD_Opt', 't10', s, 22) for s in ('lead_brake', 'cut_in')]
                      + [_route('town:Town04', 't04', s, 31) for s in _T04]),
    'falcon-pdm-mini': [_route('town:Town10HD_Opt', 't10', s, 1)
                        for s in ('lead_brake', 'stalled_vehicle', 'pedestrian_crossing', 'cut_in')],
}
ROUTE_SET_ABOUT = {
    'falcon-pdm-v1': 'central Bengaluru (an imported OpenStreetMap world: 11 routes of Indian and generic scenarios), '
                     'Town10HD_Opt (9) and Town04 (3); kinematic scenario actors, no background traffic',
    'falcon-pdm-mini': 'four Town10HD_Opt routes for a quick check',
}


class BenchError(Exception):
    pass


def routes(name: str = 'falcon-pdm-v1') -> list:
    if name not in ROUTE_SETS:
        raise BenchError(f'no route set {name!r} (sets: {", ".join(ROUTE_SETS)})')
    return [dict(r, motion='kinematic', traffic=0) for r in ROUTE_SETS[name]]


def resolve_world(project, spec: str) -> tuple:
    """(project world id, None) for a route's world ('town:NAME' or 'osm:bengaluru'), or (None, what is missing)."""
    kind, _, name = spec.partition(':')
    worlds = project.doc.get('worlds') or []
    if kind == 'town':
        w = next((w for w in worlds if w.get('kind') == 'town' and w.get('town') == name), None)
        return (w['id'], None) if w else (None, f'the project has no world for the CARLA town {name} '
                                                f'(Worlds ▸ Add a town)')
    if kind == 'osm':
        pat = {'bengaluru': r'bengal|bangal|blr'}.get(name, re.escape(name))
        w = next((w for w in worlds if w.get('kind') == 'osm'
                  and re.search(pat, (w.get('name', '') + ' ' + w['id']).lower())), None)
        return (w['id'], None) if w else (None, f'the project has no imported {name} world (Worlds ▸ Import a place: '
                                                f'central Bengaluru)')
    raise BenchError(f'a route world is town:NAME or osm:NAME, not {spec!r}')


# --------------------------------------------------------------------------------------------- records
def record(ego, actors=None, lanes=None, events=None, source='given', notes=()) -> dict:
    """A run's record from arrays: ego (N, 8) rows (t, x, y, yaw°, vx, vy, length, width); actors {id: {'kind',
    'traj': (M, 8)}}; lanes a lanes.json document or a core.rules.RoadMap (None: no map); events [(t, actor id or
    None)] from the collision sensor."""
    from .rules import RoadMap
    m = lanes
    if isinstance(lanes, dict):
        m = RoadMap()
        m.add_lanes_json(lanes)
    ego = np.asarray(ego, float).reshape(-1, 8)
    if m is not None and getattr(m, 'has_lanes', False):
        m.crop(ego[:, 1:3], margin=60.0)
        if not m.has_lanes:
            m = None
    elif m is not None:
        m = None
    acts = {str(k): {'kind': v.get('kind', 'car'), 'traj': np.asarray(v['traj'], float).reshape(-1, 8)}
            for k, v in (actors or {}).items() if v.get('traj') is not None and len(v['traj'])}
    return {'ego': ego, 'actors': acts, 'map': m, 'events': list(events or ()), 'source': source,
            'notes': list(notes)}


def _kinds_of(cfg: dict) -> dict:
    return {a.get('id'): a.get('kind', 'car') for a in ((cfg.get('scenario') or {}).get('actors') or ())}


def record_from_run(run_dir: str, xodr: str | None = None, carla_root: str = '', use_map: bool = True) -> dict:
    """A recorded run's record: a scenario drive or a stack run (trajectories.json), or any drive (frames.jsonl and
    actors.jsonl, through core.rules.trace_from_run)."""
    from . import rules as RU
    cfg = RU._run_config(run_dir)
    run = {}
    rp = os.path.join(run_dir, 'run.json')
    if os.path.isfile(rp):
        with open(rp, encoding='utf-8') as fh:
            run = json.load(fh)
    notes, roadmap = [], None
    tr = None
    try:
        if xodr is None and os.path.isfile(os.path.join(run_dir, 'lanes.json')):
            xodr = ''                              # the run's own lanes suffice: no OpenDRIVE file to parse
        tr = RU.trace_from_run(run_dir, xodr=xodr, carla_root=carla_root, use_map=use_map)
        roadmap = tr.map if tr.map.has_lanes else None
        if roadmap is None:
            notes.append('no map (lanes.json or the OpenDRIVE file): drivable area and driving direction not '
                         'evaluable')
    except RU.RulesError as e:
        notes.append(f'map and frames not read: {e}')
    tj = os.path.join(run_dir, 'trajectories.json')
    kinds = _kinds_of(cfg)
    scn = run.get('scenario') or {}
    if os.path.isfile(tj):
        with open(tj, encoding='utf-8') as fh:
            doc = json.load(fh)
        rows = doc.get('actors') or {}
        if not rows.get('ego'):
            raise BenchError(f'{run_dir}: trajectories.json has no ego')
        actors = {k: {'kind': kinds.get(k, 'car'), 'traj': v} for k, v in rows.items() if k != 'ego'}
        ids = {int(v): k for k, v in (scn.get('actor_ids') or {}).items()}
        events = [(float(e[0]), ids.get(int(e[1]) if e[1] is not None else -1)) for e in scn.get('collisions') or ()]
        return record(rows['ego'], actors, roadmap, events, 'trajectories.json', notes)
    if tr is None:
        raise BenchError(f'{run_dir}: neither trajectories.json nor frames.jsonl')
    yaw = np.radians(tr.yaw)
    sp = np.nan_to_num(tr.speed)
    L, W = tr.ego_size
    ego = np.column_stack([tr.t, tr.x, tr.y, tr.yaw, sp * np.cos(yaw), sp * np.sin(yaw),
                           np.full(len(tr.t), L), np.full(len(tr.t), W)])
    actors = {}
    for i, acts in enumerate(tr.actors or ()):
        for a in acts:
            d = actors.setdefault(str(a['id']), {'kind': 'pedestrian' if a.get('kind') == 'pedestrian' else 'car',
                                                 'traj': []})
            d['traj'].append([tr.t[i], a['x'], a['y'], a['yaw'], a['vx'], a['vy'], a['l'], a['w']])
    notes.append('no trajectories.json: the ego from frames.jsonl, the others from actors.jsonl')
    return record(ego, actors, roadmap, [], 'frames.jsonl', notes)


def record_from_episode(e: dict, lanes=None) -> dict:
    """A gym or leaderboard episode's record (its `trajectories`, logged by the gym worker since Falcon 2.5)."""
    tj = e.get('trajectories')
    if not tj or not (tj.get('actors') or {}).get('ego'):
        raise BenchError('the episode carries no trajectories (recorded before Falcon 2.5, or log_trace off)')
    kinds = tj.get('kinds') or {}
    acts = {k: {'kind': kinds.get(k, 'car'), 'traj': v} for k, v in tj['actors'].items() if k != 'ego'}
    notes = [] if lanes is not None else ['no lanes.json beside the episode: drivable area and driving direction '
                                          'not evaluable']
    events = [(float(x[0]), x[1]) for x in tj.get('events') or ()]
    return record(tj['actors']['ego'], acts, lanes, events, 'episode trajectories', notes)


# --------------------------------------------------------------------------------------------- geometry
def _corners(rows) -> np.ndarray:
    from .criticality import _corners_of
    return _corners_of(np.asarray(rows, float))


def _wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def _lane_frame(m, Q):
    """For points Q (K, 2): lon, lat (K, M) in every lane point's frame."""
    r = np.radians(m.yaw)
    f = np.stack([np.cos(r), np.sin(r)], 1)
    g = np.stack([-np.sin(r), np.cos(r)], 1)
    d = Q[:, None, :] - m.P[None, :, :]
    return np.einsum('kmd,md->km', d, f), np.einsum('kmd,md->km', d, g), d


def outside_m(m, Q, chunk: int = 256) -> np.ndarray:
    """How far each point lies outside every driving lane of the map (0 inside one), metres."""
    out = np.zeros(len(Q))
    half = m.width / 2.0
    reach = 0.75 * m.step
    for k in range(0, len(Q), chunk):
        lon, lat, d = _lane_frame(m, Q[k:k + chunk])
        alon = np.abs(lon) <= reach
        ex = np.where(alon, np.abs(lat) - half[None, :], np.inf)
        best = ex.min(1)
        none = ~np.isfinite(best)
        if none.any():
            dist = np.hypot(d[none, :, 0], d[none, :, 1]) - half[None, :]
            best[none] = dist.min(1)
        out[k:k + chunk] = np.maximum(best, 0.0)
    return out


def lane_facts(m, ego) -> dict:
    """Per sample: on a lane (centre), on a lane of the ego's direction, the body's margin inside its lane."""
    n = len(ego)
    on, with_dir, margin = np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan)
    half = m.width / 2.0
    hw = ego[:, 7] / 2.0
    for k in range(0, n, 256):
        Q = ego[k:k + 256, 1:3]
        lon, lat, _ = _lane_frame(m, Q)
        inside = (np.abs(lon) <= 0.75 * m.step) & (np.abs(lat) <= half[None, :])
        dyaw = np.abs(_wrap(m.yaw[None, :] - ego[k:k + 256, 3][:, None]))
        on[k:k + 256] = inside.any(1)
        same = inside & (dyaw <= 90.0)
        with_dir[k:k + 256] = same.any(1)
        cost = np.where(same, np.hypot(lon, lat) + dyaw / 90.0, np.inf)
        j = cost.argmin(1)
        ok = np.isfinite(cost[np.arange(len(Q)), j])
        mg = half[j] - np.abs(lat[np.arange(len(Q)), j]) - hw[k:k + 256]
        margin[k:k + 256] = np.where(ok, mg, np.nan)
    return {'on_lane': on, 'with_direction': with_dir, 'lane_margin_m': margin}


# --------------------------------------------------------------------------------------------- the terms
def _deriv(t, y, half: float = SMOOTH_S) -> np.ndarray:
    """dy/dt by a local quadratic fit over ±half seconds (a Savitzky–Golay derivative on any spacing)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    n = len(t)
    out = np.zeros(n)
    if n < 3:
        return out
    for i in range(n):
        k = np.flatnonzero(np.abs(t - t[i]) <= half + 1e-9)
        if len(k) < 3:
            lo, hi = max(0, i - 1), min(n, i + 2)
            k = np.arange(lo, hi)
        x = t[k] - t[i]
        deg = min(2, len(k) - 1)
        c = np.polyfit(x, y[k], deg)
        out[i] = c[-2]
    return out


def kinematics(ego) -> dict:
    """Longitudinal and lateral acceleration, yaw rate and acceleration, longitudinal jerk and jerk magnitude."""
    ego = np.asarray(ego, float)
    t = ego[:, 0]
    yaw = np.unwrap(np.radians(ego[:, 3]))
    ax, ay = _deriv(t, ego[:, 4]), _deriv(t, ego[:, 5])
    c, s = np.cos(yaw), np.sin(yaw)
    a_lon = ax * c + ay * s
    a_lat = -ax * s + ay * c
    yr = _deriv(t, yaw)
    jx, jy = _deriv(t, ax), _deriv(t, ay)
    return {'lon_accel': a_lon, 'lat_accel': a_lat, 'yaw_rate': yr, 'yaw_accel': _deriv(t, yr),
            'lon_jerk': _deriv(t, a_lon), 'jerk': np.hypot(jx, jy), 'speed': np.hypot(ego[:, 4], ego[:, 5])}


def comfort(rec) -> dict:
    k = kinematics(rec['ego'])
    if len(rec['ego']) < 3:
        return {'score': None, 'why': 'fewer than three samples'}
    worst = {'lon_accel_min': float(k['lon_accel'].min()), 'lon_accel_max': float(k['lon_accel'].max()),
             'lat_accel_abs': float(np.abs(k['lat_accel']).max()), 'yaw_rate_abs': float(np.abs(k['yaw_rate']).max()),
             'yaw_accel_abs': float(np.abs(k['yaw_accel']).max()), 'lon_jerk_abs': float(np.abs(k['lon_jerk']).max()),
             'jerk_abs': float(np.abs(k['jerk']).max())}
    failed = [n for n, v in worst.items() if (v < COMFORT[n] if n == 'lon_accel_min' else v > COMFORT[n])]
    return {'score': 0.0 if failed else 1.0, 'worst': {n: round(v, 4) for n, v in worst.items()}, 'failed': failed,
            'bounds': COMFORT}


def _in_box(P, box):
    """Which points P (K, 2) lie in the oriented box row (8,)."""
    r = math.radians(box[3])
    d = P - box[1:3]
    lon = d[:, 0] * math.cos(r) + d[:, 1] * math.sin(r)
    lat = -d[:, 0] * math.sin(r) + d[:, 1] * math.cos(r)
    return (np.abs(lon) <= box[6] / 2 + 1e-6) & (np.abs(lat) <= box[7] / 2 + 1e-6)


def contact_kind(e, a) -> str:
    """front, rear or lateral: where the contact point (the corners of either box inside the other) lies in the
    ego's frame."""
    ce, ca = _corners(e[None])[0], _corners(a[None])[0]
    pts = np.r_[ce[_in_box(ce, a)], ca[_in_box(ca, e)]]
    p = pts.mean(0) if len(pts) else (e[1:3] + a[1:3]) / 2.0
    r = math.radians(e[3])
    d = p - e[1:3]
    x = d[0] * math.cos(r) + d[1] * math.sin(r)
    hl = e[6] / 2.0
    if x > 0.5 * hl:
        return 'front'
    if x < -0.5 * hl:
        return 'rear'
    return 'lateral'


def collisions(rec, facts=None, dac_out=None) -> dict:
    """NC (module doc) and every collision with its class."""
    from .criticality import _align, _corners_of, _overlap
    ego = rec['ego']
    found = {}
    for aid, a in rec['actors'].items():
        ie, ia = _align(ego, a['traj'])
        if not len(ie):
            continue
        hit = _overlap(_corners_of(ego[ie]), _corners_of(a['traj'][ia]))
        if hit.any():
            k = int(np.argmax(hit))
            found[aid] = (int(ie[k]), int(ia[k]))
    others = []
    for t, aid in rec.get('events') or ():
        if aid is not None and str(aid) in rec['actors']:
            if str(aid) in found:
                continue
            a = rec['actors'][str(aid)]['traj']
            i = int(np.argmin(np.abs(ego[:, 0] - t)))
            j = int(np.argmin(np.abs(a[:, 0] - t)))
            found[str(aid)] = (i, j)
        else:
            others.append(float(t))
    rows = []
    for aid, (i, j) in sorted(found.items(), key=lambda kv: kv[1][0]):
        e, a = ego[i], rec['actors'][aid]['traj'][j]
        kind = rec['actors'][aid]['kind']
        ve, va = math.hypot(e[4], e[5]), math.hypot(a[4], a[5])
        static = kind in STATIC_KINDS and va < STOPPED_MPS
        if ve < STOPPED_MPS:
            cls, fault = 'stopped_ego', False
        elif va < STOPPED_MPS:
            cls, fault = 'stopped_track', True
        else:
            side = contact_kind(e, a)
            cls = f'active_{side}'
            if side == 'front':
                fault = True
            elif side == 'rear':
                fault = False
            else:
                crossing = facts is not None and (not facts['on_lane'][i] or
                                                  (np.isfinite(facts['lane_margin_m'][i])
                                                   and facts['lane_margin_m'][i] < 0))
                off = dac_out is not None and dac_out[i] > DAC_TOL_M
                fault = bool(crossing or off) if facts is not None else True
        rows.append({'actor': aid, 'kind': kind, 't': round(float(e[0]), 3), 'class': cls, 'at_fault': bool(fault),
                     'static': bool(static), 'ego_speed_mps': round(ve, 3)})
    for t in others:
        i = int(np.argmin(np.abs(ego[:, 0] - t)))
        ve = math.hypot(ego[i, 4], ego[i, 5])
        rows.append({'actor': None, 'kind': 'object', 't': round(t, 3), 'class': 'stopped_ego' if ve < STOPPED_MPS
                     else 'object', 'at_fault': ve >= STOPPED_MPS, 'static': True, 'ego_speed_mps': round(ve, 3)})
    fault = [r for r in rows if r['at_fault']]
    score = 1.0 if not fault else (0.5 if all(r['static'] for r in fault) else 0.0)
    return {'score': score, 'collisions': rows, 'at_fault': len(fault)}


def drivable_area(rec) -> tuple:
    """(DAC result, per-sample worst corner distance outside the lanes or None)."""
    m = rec['map']
    if m is None:
        return {'score': None, 'why': 'no map'}, None
    C = _corners(rec['ego']).reshape(-1, 2)
    out = outside_m(m, C).reshape(-1, 4).max(1)
    worst = float(out.max()) if len(out) else 0.0
    bad = np.flatnonzero(out > DAC_TOL_M)
    return ({'score': 0.0 if len(bad) else 1.0, 'max_outside_m': round(worst, 3), 'tolerance_m': DAC_TOL_M,
             'first_t': round(float(rec['ego'][bad[0], 0]), 3) if len(bad) else None}, out)


def driving_direction(rec, facts) -> dict:
    if facts is None:
        return {'score': None, 'why': 'no map'}
    ego = rec['ego']
    ds = np.r_[0.0, np.hypot(np.diff(ego[:, 1]), np.diff(ego[:, 2]))]
    against = facts['on_lane'] & ~facts['with_direction']
    d = np.where(against, ds, 0.0)
    t = ego[:, 0]
    worst = 0.0
    j = 0
    acc = 0.0
    for i in range(len(t)):                        # the sum over every 1 s window (two pointers)
        acc += d[i]
        while t[i] - t[j] > DDC_WINDOW_S:
            acc -= d[j]
            j += 1
        worst = max(worst, acc)
    score = 1.0 if worst <= DDC_OK_M else (0.5 if worst <= DDC_BAD_M else 0.0)
    return {'score': score, 'max_against_m': round(worst, 3), 'window_s': DDC_WINDOW_S}


def ttc_bound(rec) -> dict:
    from .criticality import _align, _corners_of, _overlap, _ttc_rows
    ego = rec['ego']
    best, who, when = math.inf, None, None
    for aid, a in rec['actors'].items():
        ie, ia = _align(ego, a['traj'])
        if not len(ie):
            continue
        E, A = ego[ie], a['traj'][ia]
        r = np.radians(E[:, 3])
        xr = (A[:, 1] - E[:, 1]) * np.cos(r) + (A[:, 2] - E[:, 2]) * np.sin(r)
        moving = np.hypot(E[:, 4], E[:, 5]) > MOVING_MPS
        touching = _overlap(_corners_of(E), _corners_of(A))
        keep = moving & ~touching & (xr >= 0.0)
        if not keep.any():
            continue
        ttc = _ttc_rows(E[keep], A[keep], TTC_HORIZON_S)
        k = int(np.argmin(ttc))
        if ttc[k] < best:
            best, who, when = float(ttc[k]), aid, float(E[keep][k, 0])
    return {'score': 0.0 if best < TTC_BOUND_S else 1.0, 'min_ttc_s': None if not math.isfinite(best) else
            round(best, 3), 'actor': who, 't': None if when is None else round(when, 3), 'bound_s': TTC_BOUND_S,
            'horizon_s': TTC_HORIZON_S}


def route_path(rec) -> np.ndarray:
    """The route a record drove: its ego positions with repeats (< 0.1 m apart) left out."""
    P = rec['ego'][:, 1:3]
    keep = [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - P[keep[-1]])) >= 0.1:
            keep.append(i)
    return P[keep]


def progress_along(path: np.ndarray, P: np.ndarray) -> float:
    """The farthest arc length along the path reached by the points (within ROUTE_LAT_M of it), less the first
    point's."""
    if len(path) < 2 or not len(P):
        return 0.0
    a, b = path[:-1], path[1:]
    seg = b - a
    L = np.hypot(seg[:, 0], seg[:, 1])
    s0 = np.r_[0.0, np.cumsum(L)]
    out = []
    for p in P:
        u = np.clip(np.einsum('kd,kd->k', p - a, seg) / np.maximum(L ** 2, 1e-12), 0.0, 1.0)
        q = a + seg * u[:, None]
        d = np.hypot(*(p - q).T)
        k = int(np.argmin(d))
        out.append((s0[k] + u[k] * L[k], d[k]))
    first = out[0][0]
    reach = [s for s, d in out if d <= ROUTE_LAT_M]
    return float(max(max(reach, default=first) - first, 0.0))


def score_record(rec, reference=None) -> dict:
    """Every term of one run (module doc), the PDMS when the reference's progress is known (reference: the reference
    run's record, or its progress in metres along this run's own path)."""
    facts = lane_facts(rec['map'], rec['ego']) if rec['map'] is not None else None
    dac, out = drivable_area(rec)
    nc = collisions(rec, facts, out)
    out_terms = {'nc': nc, 'dac': dac, 'ddc': driving_direction(rec, facts), 'ttc': ttc_bound(rec),
                 'comfort': comfort(rec)}
    if isinstance(reference, dict):
        path = route_path(reference)
        own = progress_along(path, rec['ego'][:, 1:3])
        ref = progress_along(path, reference['ego'][:, 1:3])
    else:
        own = progress_along(route_path(rec), rec['ego'][:, 1:3])
        ref = reference
    row = {'terms': out_terms, 'progress_m': round(own, 3), 'source': rec.get('source'), 'notes': rec.get('notes')}
    return finalise(row, ref)


def ep_score(progress_m, ref_m) -> float | None:
    if ref_m is None:
        return None
    if ref_m < EP_MIN_REF_M:
        return 1.0
    return float(min(max(progress_m / ref_m, 0.0), 1.0))


def finalise(row: dict, ref_m) -> dict:
    """Set EP from the reference's progress and compute PDMS (module doc); a term that is None is left out."""
    terms = row['terms']
    terms['ep'] = {'score': ep_score(row['progress_m'], ref_m), 'progress_m': row['progress_m'],
                   'reference_m': None if ref_m is None else round(float(ref_m), 3), 'min_reference_m': EP_MIN_REF_M}
    s = {k: terms[k]['score'] for k in TERMS}
    missing = [k for k in TERMS if s[k] is None]
    mult = float(np.prod([s[k] for k in MULTIPLIERS if s[k] is not None])) if any(
        s[k] is not None for k in MULTIPLIERS) else 1.0
    w = {k: v for k, v in WEIGHTS.items() if s[k] is not None}
    avg = sum(v * s[k] for k, v in w.items()) / sum(w.values()) if w else None
    row['scores'] = dict(s)
    row['pdms'] = None if avg is None else mult * avg
    nav = float(np.prod([s[k] for k in ('nc', 'dac') if s[k] is not None])) if avg is not None else None
    row['pdms_navsim_v1'] = None if avg is None else nav * avg
    row['partial'] = missing
    return row


# --------------------------------------------------------------------------------------------- sets of runs
def _mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def flat(row: dict) -> dict:
    """A scored run as one flat row of numbers (for tables and the bootstrap)."""
    out = {k: row['scores'].get(k) for k in TERMS}
    out.update(pdms=row.get('pdms'), pdms_navsim_v1=row.get('pdms_navsim_v1'))
    return out


def aggregate(scored: dict, reference: str | None = REFERENCE, n_boot: int = 1000, seed: int = 0) -> dict:
    """scored: {planner: {route id: a score_record row without EP's reference (or with)}}. EP's reference per route is
    the reference planner's progress when it drove the route, else the best progress any planner made on it (stated).
    Returns per planner the per-route rows, the means with bootstrap intervals, and the paired comparisons."""
    from .metrics import bootstrap, paired_bootstrap
    if not scored:
        raise BenchError('no scored runs')
    route_ids = sorted({r for p in scored.values() for r in p})
    ref_m, ref_from = {}, {}
    for rid in route_ids:
        if reference and rid in scored.get(reference, {}):
            ref_m[rid], ref_from[rid] = scored[reference][rid]['progress_m'], reference
        else:
            best = max((p[rid]['progress_m'] for p in scored.values() if rid in p), default=None)
            ref_m[rid], ref_from[rid] = best, 'the best planner on the route'
    planners = {}
    for name, runs in scored.items():
        rows = []
        for rid in sorted(runs):
            r = finalise(dict(runs[rid]), ref_m[rid])
            rows.append(dict(flat(r), route=rid, partial=r['partial']))
        keys = list(TERMS) + ['pdms', 'pdms_navsim_v1']
        boot = bootstrap(lambda rs: {k: _mean(rs, k) for k in keys}, rows, n=n_boot, seed=seed, unit='image')
        planners[name] = {'routes': rows, 'n_routes': len(rows),
                          'metrics': {k: {x: boot['metrics'][k][x] for x in ('value', 'ci_low', 'ci_high')}
                                      for k in keys if k in boot['metrics']},
                          'partial_routes': sum(1 for r in rows if r['partial'])}
    comparisons = []
    names = sorted(planners)
    base = reference if reference in planners else (names[0] if names else None)
    for name in names:
        if name == base:
            continue
        shared = sorted(set(scored[base]) & set(scored[name]))
        if not shared:
            continue
        table = {p: {r['route']: r for r in planners[p]['routes']} for p in (base, name)}

        def fn(items, p, table=table):
            return {k: _mean([table[p][i] for i in items], k) for k in ('pdms', 'nc', 'ttc', 'ep', 'comfort')}
        pb = paired_bootstrap(fn, shared, base, name, n=n_boot, seed=seed, unit='image')
        comparisons.append({'a': base, 'b': name, 'routes': len(shared),
                            'metrics': {k: {x: v.get(x) for x in ('a', 'b', 'diff', 'ci_low', 'ci_high', 'p_b_gt_a')}
                                        for k, v in pb['metrics'].items()}})
    return {'schema': RESULTS_SCHEMA, 'planners': planners, 'comparisons': comparisons,
            'reference': {'planner': reference, 'progress_m': ref_m, 'from': ref_from},
            'method': 'nuPlan / NAVSIM style PDM score (core.planbench); 95 % percentile bootstrap over routes',
            'n_boot': n_boot, 'seed': seed, 'terms': TERM_NAMES}


# --------------------------------------------------------------------------------------------- a benchmark's folder
def bench_dir(workspace: str, name: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name or ''):
        raise BenchError(f'a benchmark name is letters, digits, _ . - (not {name!r})')
    return os.path.join(workspace, 'benchmarks', name)


def planner_slug(planner: str) -> str:
    kind, _, arg = planner.partition(':')
    tail = os.path.basename(os.path.normpath(arg)) if arg else ''
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', f'{kind}-{tail}' if tail else kind).strip('-')[:48]


def read_index(bdir: str) -> list:
    """The benchmark's finished runs: index.jsonl rows {planner, route, run (folder) or episodes + episode}."""
    p = os.path.join(bdir, 'index.jsonl')
    if not os.path.isfile(p):
        return []
    with open(p, encoding='utf-8') as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    last = {}
    for r in rows:                              # a re-run route replaces its earlier line
        last[(r['planner'], r['route'])] = r
    return list(last.values())


def score_entry(entry: dict, bdir: str = '') -> dict:
    """Score one index row (a run folder, or a line of a planner's episodes.jsonl)."""
    if entry.get('episodes'):
        path = entry['episodes'] if os.path.isabs(entry['episodes']) else os.path.join(bdir, entry['episodes'])
        with open(path, encoding='utf-8') as fh:
            eps = [json.loads(x) for x in fh if x.strip()]
        e = eps[int(entry.get('episode', 0))]
        lp = os.path.join(entry.get('env_dir') or os.path.dirname(path), 'lanes.json')
        lanes = None
        if os.path.isfile(lp):
            with open(lp, encoding='utf-8') as fh:
                lanes = json.load(fh)
        return score_record(record_from_episode(e, lanes))
    run = entry['run'] if os.path.isabs(entry['run']) else os.path.join(bdir, entry['run'])
    return score_record(record_from_run(run))


def results(bdir: str, n_boot: int = 1000, seed: int = 0, reference: str | None = None) -> dict:
    """Score every finished run of a benchmark folder and aggregate (writes scores.json and results.json)."""
    from .provenance import write_json
    doc = {}
    bp = os.path.join(bdir, 'benchmark.json')
    if os.path.isfile(bp):
        with open(bp, encoding='utf-8') as fh:
            doc = json.load(fh)
    scored, failed = {}, []
    for e in read_index(bdir):
        if e.get('status', 'ok') != 'ok':
            failed.append(e)
            continue
        try:
            scored.setdefault(e['planner'], {})[e['route']] = score_entry(e, bdir)
        except (BenchError, OSError, ValueError, KeyError, IndexError) as ex:
            failed.append(dict(e, error=f'scoring: {ex}'))
    if not scored:
        raise BenchError(f'{bdir}: no finished runs to score')
    ref = reference or doc.get('reference') or REFERENCE
    out = aggregate(scored, ref if any(ref == p for p in scored) else None, n_boot, seed)
    out['failed'] = failed
    out['route_set'] = doc.get('route_set')
    out['name'] = doc.get('name') or os.path.basename(bdir)
    write_json(os.path.join(bdir, 'scores.json'), {'schema': SCHEMA, 'runs': {
        p: {r: {k: v for k, v in row.items()} for r, row in runs.items()} for p, runs in scored.items()}})
    write_json(os.path.join(bdir, 'results.json'), out)
    return out


def list_benchmarks(workspace: str) -> list:
    root = os.path.join(workspace, 'benchmarks')
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        res = None
        rp = os.path.join(d, 'results.json')
        if os.path.isfile(rp):
            try:
                with open(rp, encoding='utf-8') as fh:
                    res = json.load(fh)
            except (OSError, ValueError):
                res = None
        out.append({'name': name, 'dir': d, 'runs': len(read_index(d)), 'results': res})
    return out


def table(res: dict) -> str:
    """The planners' means with their intervals, one line each (the CLI's text)."""
    lines = [f'{"planner":24s} {"routes":>6s}  ' + '  '.join(f'{k:>16s}' for k in ('pdms', 'nc', 'dac', 'ddc', 'ttc',
                                                                                 'ep', 'comfort'))]
    for name, p in sorted(res['planners'].items(), key=lambda kv: -(kv[1]['metrics'].get('pdms', {}).get('value')
                                                                     or 0)):
        cells = []
        for k in ('pdms', 'nc', 'dac', 'ddc', 'ttc', 'ep', 'comfort'):
            m = p['metrics'].get(k) or {}
            v = m.get('value')
            cells.append(f'{"—":>16s}' if v is None else
                         f'{v:5.3f} [{m["ci_low"]:.2f},{m["ci_high"]:.2f}]'.rjust(16))
        lines.append(f'{name[:24]:24s} {p["n_routes"]:6d}  ' + '  '.join(cells))
    for c in res.get('comparisons') or ():
        m = c['metrics'].get('pdms') or {}
        if m.get('diff') is not None:
            lines.append(f'{c["b"]} − {c["a"]}: PDMS {m["diff"]:+.3f} [{m["ci_low"]:+.3f}, {m["ci_high"]:+.3f}] over '
                         f'{c["routes"]} shared routes')
    return '\n'.join(lines)


# --------------------------------------------------------------------------------------------- the command
def main_bench(argv) -> int:
    """`falcon bench routes|score|results|run …` (docs/planner-benchmark.md). 0 ok, 2 failed."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon bench')
    sub = ap.add_subparsers(dest='what', required=True)
    r = sub.add_parser('routes', help='the fixed route sets')
    r.add_argument('--set', default='falcon-pdm-v1', choices=sorted(ROUTE_SETS))
    r.add_argument('--json', action='store_true')
    s = sub.add_parser('score', help='score recorded runs (scenario drives, stack runs)')
    s.add_argument('runs', nargs='+')
    s.add_argument('--reference', default='', help='a reference run of the same route (EP); default: none')
    s.add_argument('--json', action='store_true')
    res = sub.add_parser('results', help='score a benchmark\'s runs and compare its planners')
    res.add_argument('project', help='a project file (its workspace) or a benchmark folder')
    res.add_argument('--name', default='')
    res.add_argument('--boot', type=int, default=1000)
    res.add_argument('--reference', default='', help='the reference planner for EP (default: follower)')
    res.add_argument('--json', action='store_true')
    ru = sub.add_parser('run', help='drive a planner over a route set (sim.benchrun)')
    ru.add_argument('project')
    ru.add_argument('--name', required=True)
    ru.add_argument('--planner', action='append', required=True,
                    help='follower | none | autopilot | policy:DIR | stack:SUITE.json (repeatable)')
    ru.add_argument('--set', default='falcon-pdm-v1', choices=sorted(ROUTE_SETS))
    ru.add_argument('--route', action='append', default=None, help='only these route ids')
    ru.add_argument('--rig', default='')
    ru.add_argument('--no-reference', action='store_true', help='do not also drive the reference planner (follower)')
    ru.add_argument('--boot', type=int, default=1000)
    ru.add_argument('--fake-carla', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    try:
        if a.what == 'routes':
            rs = routes(a.set)
            if a.json:
                print(json.dumps({'set': a.set, 'about': ROUTE_SET_ABOUT[a.set], 'routes': rs}, indent=1))
            else:
                print(f'{a.set}: {ROUTE_SET_ABOUT[a.set]}')
                for x in rs:
                    print(f'  {x["id"]:34s} {x["world"]:20s} {x["scenario"]:22s} seed {x["seed"]}')
            return P.OK
        if a.what == 'score':
            ref = record_from_run(a.reference) if a.reference else None
            rows = {}
            for run in a.runs:
                rows[run] = score_record(record_from_run(run), ref)
            if a.json:
                print(json.dumps(rows, indent=1, default=float))
            else:
                for run, row in rows.items():
                    sc = row['scores']
                    pd = '—' if row['pdms'] is None else f'{row["pdms"]:.3f}'
                    print(f'{os.path.basename(os.path.normpath(run))}: PDMS {pd}  ' + '  '.join(
                        f'{k} {"—" if sc[k] is None else f"{sc[k]:g}"}' for k in TERMS)
                        + (f'  (not evaluable: {", ".join(row["partial"])})' if row['partial'] else ''))
            return P.OK
        if a.what == 'results':
            if os.path.isdir(a.project):
                bdir = a.project
            else:
                from .project import Project
                if not a.name:
                    raise BenchError('--name NAME is needed with a project')
                bdir = bench_dir(Project.load(a.project).workspace, a.name)
            out = results(bdir, a.boot, reference=a.reference or None)
            print(json.dumps(out, indent=1) if a.json else table(out))
            return P.OK
        from ..sim.benchrun import main_run
        return main_run(a)
    except (BenchError, OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
