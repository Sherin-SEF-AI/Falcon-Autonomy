"""Two runs on one clock, and where they part (numpy): Replay's synchronized side-by-side comparison.

A run's TRACKS come from what it recorded: trajectories.json when it has one (a replay, a scenario run: every step's
box centre, heading, velocity and footprint — core.criticality's input), else actors.jsonl (every vehicle and walker
of every step with its velocity and box; the ego is the `hero`) or, without it, frames.jsonl's ego pose alone (its
velocity from the positions, a typical 4.6 × 2.0 m box — heuristic, said in `notes`). Times are seconds from each
run's first sample.

The CLOCK is one grid for both runs: a step of the finer of their two steps, from 0 to the longer run's end. At grid
time t each run shows its sample nearest t, or nothing when it has none within half its own step (it has not started
or has ended), so two runs at different rates are still compared at the same moment, never by step index.

A grid step DIVERGES when, with both runs present there, any of:
  * the egos are further apart than `gap_m` (default 1.0 m; positions of the ego's box centre);
  * one run's ego box overlaps an actor's box and the other's does not (a collision in one only);
  * the smallest TTC to any actor (core.criticality.ttc_rows, constant velocity, horizon 10 s) differs by more than
    `ttc_s` (default 1.0 s) while at least one of the two is within the horizon;
  * an actor is present in one run and not in the other. Actors are matched by id; when the two runs share fewer
    than half of their ids (two servers, different spawns) ids mean nothing across them, and the number of actors
    present is compared instead (said in `notes`).
A run that recorded no actors (frames.jsonl alone) is compared by its ego's position only (said in `notes`): its
missing actors are not recorded, not absent.
Thresholds are the caller's choice, not a standard. A divergence EVENT is a maximal stretch of diverging steps; its
start is what the slider marks. Everything is computed once, up front; seeking is an index lookup.
"""
from __future__ import annotations

import json
import os

import numpy as np

GAP_M = 1.0
TTC_S = 1.0
HORIZON_S = 10.0
TYPICAL_EGO = (4.6, 2.0)
REASONS = (('gap', 'ego position gap'), ('collision', 'collision in one run only'),
           ('ttc', 'min TTC change'), ('actors', 'actor in one run only'))


class SyncError(ValueError):
    pass


def _rows_from_trajectories(path: str):
    with open(path, encoding='utf-8') as fh:
        doc = json.load(fh)
    acts = {}
    for k, v in (doc.get('actors') or {}).items():
        a = np.asarray(v, float)
        if a.ndim == 2 and a.shape[0] and a.shape[1] >= 8:
            acts[str(k)] = a[:, :8]
    if 'ego' not in acts:
        raise SyncError(f'{path}: no ego in trajectories.json')
    return acts.pop('ego'), acts, float(doc.get('dt') or 0.0)


def _rows_from_actors(path: str):
    """(ego rows or None, {id: rows}) from actors.jsonl (box centres: the transform applied to the box's centre)."""
    import math
    per: dict = {}
    ego = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            t = float(row.get('sim_t') or 0.0)
            for a in row.get('actors') or ():
                yaw = float(a.get('yaw') or 0.0)
                c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                bc = a.get('bbox_center') or (0.0, 0.0, 0.0)
                x = float(a['x']) + c * bc[0] - s * bc[1]
                y = float(a['y']) + s * bc[0] + c * bc[1]
                ex = a.get('extent') or (2.3, 1.0, 0.75)
                r = (t, x, y, yaw, float(a.get('vx') or 0.0), float(a.get('vy') or 0.0), 2 * ex[0], 2 * ex[1])
                if a.get('role') == 'hero':
                    ego.append(r)
                else:
                    per.setdefault(str(a['id']), []).append(r)
    return (np.asarray(ego, float) if ego else None), {k: np.asarray(v, float) for k, v in per.items()}


def _ego_from_frames(path: str) -> np.ndarray:
    rows = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                f = json.loads(line)
            except ValueError:
                continue
            e = f.get('ego') or {}
            if 'x' in e and 'y' in e:
                rows.append((float(f.get('sim_t') or 0.0), float(e['x']), float(e['y']), float(e.get('yaw') or 0.0)))
    if not rows:
        raise SyncError(f'{path}: no ego poses in frames.jsonl')
    r = np.asarray(rows, float)
    if len(r) > 1:
        vx, vy = np.gradient(r[:, 1], r[:, 0]), np.gradient(r[:, 2], r[:, 0])
    else:
        vx = vy = np.zeros(1)
    n = len(r)
    return np.column_stack([r, vx, vy, np.full(n, TYPICAL_EGO[0]), np.full(n, TYPICAL_EGO[1])])


def load_run(run_dir: str) -> dict:
    """One run's tracks: {'dir', 'name', 'ego' (N, 8), 'actors' {id: (M, 8)}, 'source', 'notes', 'fidelity'}; times
    rebased to the run's first sample."""
    from .. import fidelity as F
    d = os.path.abspath(run_dir)
    if not os.path.isdir(d):
        raise SyncError(f'{run_dir}: not a folder')
    notes = []
    tp, ap, fp = (os.path.join(d, n) for n in ('trajectories.json', 'actors.jsonl', 'frames.jsonl'))
    if os.path.isfile(tp):
        ego, actors, _ = _rows_from_trajectories(tp)
        source = 'trajectories.json'
    else:
        ego, actors = (None, {})
        if os.path.isfile(ap):
            ego, actors = _rows_from_actors(ap)
        if ego is None:
            if not os.path.isfile(fp):
                raise SyncError(f'{run_dir}: no trajectories.json, actors.jsonl or frames.jsonl (no poses to compare)')
            ego = _ego_from_frames(fp)
            notes.append('the ego from frames.jsonl: velocity from its positions, a typical 4.6 × 2.0 m box '
                         '(heuristic)')
            source = 'actors.jsonl + frames.jsonl' if actors else 'frames.jsonl'
        else:
            source = 'actors.jsonl'
    if not len(ego):
        raise SyncError(f'{run_dir}: the ego has no samples')
    t0 = float(ego[0, 0])
    for a in actors.values():
        if len(a):
            t0 = min(t0, float(a[0, 0]))
    ego = ego.copy()
    ego[:, 0] -= t0
    actors = {k: np.column_stack([v[:, :1] - t0, v[:, 1:]]) for k, v in actors.items() if len(v)}
    name = os.path.basename(d)
    try:
        with open(os.path.join(d, 'run.json'), encoding='utf-8') as fh:
            name = json.load(fh).get('name') or name
    except (OSError, ValueError):
        pass
    return {'dir': d, 'name': name, 'ego': ego, 'actors': actors, 'source': source, 'notes': notes,
            'actors_recorded': source != 'frames.jsonl',
            'fidelity': F.of_run(d)}


def _step(ts: np.ndarray) -> float:
    return float(np.median(np.diff(ts))) if len(ts) > 1 else 0.05


def _index_on(ts: np.ndarray, grid: np.ndarray, half: float) -> np.ndarray:
    """Per grid time, the index of the sample nearest it, or -1 when none is within `half` seconds."""
    if not len(ts):
        return np.full(len(grid), -1, int)
    j = np.clip(np.searchsorted(ts, grid), 1, max(len(ts) - 1, 1)) if len(ts) > 1 else np.zeros(len(grid), int)
    if len(ts) > 1:
        left = j - 1
        j = np.where(np.abs(ts[left] - grid) <= np.abs(ts[j] - grid), left, j)
    ok = np.abs(ts[j] - grid) <= half + 1e-9
    return np.where(ok, j, -1)


def _on_grid(run: dict, grid: np.ndarray) -> dict:
    """The run sampled on the grid: ego (K, 8) NaN where absent, actors {id: (K, 8)} NaN where absent."""
    from .. import criticality as C
    ego = run['ego']
    k_e = _index_on(ego[:, 0], grid, _step(ego[:, 0]) / 2)
    E = np.full((len(grid), 8), np.nan)
    E[k_e >= 0] = ego[k_e[k_e >= 0]]
    A = {}
    for aid, rows in run['actors'].items():
        k = _index_on(rows[:, 0], grid, _step(rows[:, 0]) / 2 if len(rows) > 1 else _step(ego[:, 0]) / 2)
        if not (k >= 0).any():
            continue
        M = np.full((len(grid), 8), np.nan)
        M[k >= 0] = rows[k[k >= 0]]
        A[aid] = M
    # the smallest TTC and an overlap (a collision) per grid step, over every actor present with the ego
    ttc = np.full(len(grid), np.inf)
    hit = np.zeros(len(grid), bool)
    have_e = ~np.isnan(E[:, 1])
    for M in A.values():
        both = have_e & ~np.isnan(M[:, 1])
        if not both.any():
            continue
        t = C.ttc_rows(E[both], M[both], HORIZON_S)
        ttc[both] = np.minimum(ttc[both], t)
        hit[both] |= t <= 0.0
    return {'ego': E, 'actors': A, 'ttc': ttc, 'collision': hit, 'present': have_e}


def _events(mask: np.ndarray) -> list:
    """(start, end) index pairs of the maximal True stretches."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def synchronize(run_a: str, run_b: str, gap_m: float = GAP_M, ttc_s: float = TTC_S) -> dict:
    """Both runs on one clock and their divergences (module doc). Returned arrays are numpy (not JSON):
    'grid' (K,) s; 'a'/'b': the run (load_run) plus 'on' (_on_grid); 'gap' (K,) m (NaN where either ego is absent);
    'reasons' (K,) bit mask over REASONS; 'events' [{'start', 'end', 't', 't_end', 'reasons' [names]}]; 'bounds'
    (x0, x1, y0, y1) of both runs; 'notes'; 'summary' (JSON-ready numbers per run and of the pair)."""
    A, B = load_run(run_a), load_run(run_b)
    sa, sb = _step(A['ego'][:, 0]), _step(B['ego'][:, 0])
    dt = max(min(sa, sb), 1e-3)
    T = max(float(A['ego'][-1, 0]), float(B['ego'][-1, 0]))
    grid = np.arange(int(np.floor(T / dt + 1e-6)) + 1) * dt
    ga, gb = _on_grid(A, grid), _on_grid(B, grid)
    both = ga['present'] & gb['present']
    gap = np.full(len(grid), np.nan)
    gap[both] = np.hypot(ga['ego'][both, 1] - gb['ego'][both, 1], ga['ego'][both, 2] - gb['ego'][both, 2])
    reasons = np.zeros(len(grid), np.int64)
    reasons[both & (gap > gap_m)] |= 1
    notes = list(A['notes']) + [f'B: {n}' for n in B['notes'] if n not in A['notes']]
    with_actors = A['actors_recorded'] and B['actors_recorded']
    if not with_actors:
        notes.append(f'{"A" if not A["actors_recorded"] else "B"} recorded no actors (frames.jsonl only): only the ego '
                     'positions are compared')
    else:
        reasons[both & (ga['collision'] != gb['collision'])] |= 2
        ta, tb = np.minimum(ga['ttc'], HORIZON_S), np.minimum(gb['ttc'], HORIZON_S)
        reasons[both & (np.abs(ta - tb) > ttc_s) & (np.minimum(ta, tb) < HORIZON_S)] |= 4
    ids_a, ids_b = set(ga['actors']), set(gb['actors'])
    shared = ids_a & ids_b
    if with_actors and (ids_a or ids_b):
        by_id = len(shared) >= 0.5 * max(len(ids_a), len(ids_b))
        if by_id:
            diff = np.zeros(len(grid), bool)
            for aid in ids_a | ids_b:
                pa = ~np.isnan(ga['actors'][aid][:, 1]) if aid in ga['actors'] else np.zeros(len(grid), bool)
                pb = ~np.isnan(gb['actors'][aid][:, 1]) if aid in gb['actors'] else np.zeros(len(grid), bool)
                diff |= pa != pb
        else:
            na = sum((~np.isnan(M[:, 1])).astype(int) for M in ga['actors'].values()) if ids_a else 0
            nb = sum((~np.isnan(M[:, 1])).astype(int) for M in gb['actors'].values()) if ids_b else 0
            diff = np.asarray(na != nb) if np.ndim(na != nb) else np.full(len(grid), bool(na != nb))
            notes.append(f'the runs share {len(shared)} of {max(len(ids_a), len(ids_b))} actor ids: actors are '
                         'compared by how many are present, not by id')
        reasons[both & diff] |= 8
    events = []
    for s, e in _events(reasons > 0):
        bits = int(np.bitwise_or.reduce(reasons[s:e + 1]))
        events.append({'start': s, 'end': e, 't': round(float(grid[s]), 4), 't_end': round(float(grid[e]), 4),
                       'reasons': [REASONS[i][0] for i in range(len(REASONS)) if bits >> i & 1]})
    xs = np.concatenate([A['ego'][:, 1], B['ego'][:, 1]] + [v[:, 1] for v in A['actors'].values()]
                        + [v[:, 1] for v in B['actors'].values()])
    ys = np.concatenate([A['ego'][:, 2], B['ego'][:, 2]] + [v[:, 2] for v in A['actors'].values()]
                        + [v[:, 2] for v in B['actors'].values()])

    def side(r, g):
        fin = g['ttc'][np.isfinite(g['ttc'])]
        e = r['ego']
        return {'name': r['name'], 'dir': r['dir'], 'source': r['source'], 'fidelity': r['fidelity'],
                'duration_s': round(float(e[-1, 0]), 4), 'actors': len(r['actors']),
                'min_ttc_s': round(float(fin.min()), 4) if len(fin) else None,
                'collision': bool(g['collision'].any()),
                'ego_path_m': round(float(np.sum(np.hypot(np.diff(e[:, 1]), np.diff(e[:, 2])))), 4)}
    k_max = int(np.nanargmax(gap)) if np.isfinite(gap).any() else None
    summary = {'a': side(A, ga), 'b': side(B, gb), 'dt': dt, 'steps': len(grid),
               'thresholds': {'gap_m': gap_m, 'ttc_s': ttc_s, 'ttc_horizon_s': HORIZON_S},
               'max_gap_m': round(float(gap[k_max]), 4) if k_max is not None else None,
               'max_gap_t': round(float(grid[k_max]), 4) if k_max is not None else None,
               'divergences': len(events), 'first_divergence_t': events[0]['t'] if events else None,
               'common_s': round(float(grid[both][-1]), 4) if both.any() else 0.0}
    return {'grid': grid, 'a': dict(A, on=ga), 'b': dict(B, on=gb), 'gap': gap, 'reasons': reasons,
            'events': events, 'bounds': (float(np.nanmin(xs)), float(np.nanmax(xs)), float(np.nanmin(ys)),
                                         float(np.nanmax(ys))), 'notes': notes, 'summary': summary}


def reason_text(names) -> str:
    d = dict(REASONS)
    return ', '.join(d.get(n, n) for n in names)
