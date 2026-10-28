"""Comparing an original replay with changed ones (numpy; a perception model through core.harness).

Pairs. The original run A and each changed run B must be replays of the same log over the same window at the same
step (their replay.json say so; otherwise refused): step i of A and step i of B show the same log time, so every
comparison is paired by step.

Driving behaviour, per run, from trajectories.json (the replay's own exact poses), with core.criticality: the minimum
time to collision and with whom, the minimum box-to-box distance, the maximum required deceleration, the
post-encroachment time, collisions (boxes overlapping, and the collision sensor's events), the criticality scalar
(heuristic, core.criticality's); and the ego's motion: path driven, mean and minimum speed, time stopped (below
0.1 m/s) and, for a re-driven ego, its deviation from the logged path. Per pair: each number's difference B − A, and
the ego's position and speed difference step by step (largest, and when).

Intervals. A kinematic replay is deterministic: one pair is one sample, and no interval can be drawn from it — the
result says so. With several changed runs (the same changes drawn with different seeds — a pedestrian's speed from a
range, say — or different changes), each number's mean difference gets a 95 % bootstrap interval over the pairs
(seeded); with fewer than five pairs it is flagged as not reliable (a bootstrap over so few has few distinct
resamples), as core.harness flags drive-level intervals.

A model. core.harness.collect runs the model on every n-th frame of one camera of A and of each B against Falcon's
ground truth of each run; frames are paired by step and core.harness.compare gives every number's difference B − A
with a paired bootstrap — over pairs of runs when there are several, else over frames (which are not independent: the
interval is then too narrow, and the result says so).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

COMPARISON_SCHEMA = 'falcon.replay_comparison/1'
MIN_PAIRS = 5
STOPPED_MPS = 0.1
METRICS = ('min_ttc_s', 'min_distance_m', 'max_drac_mps2', 'min_pet_s', 'criticality', 'collision',
           'ego_path_m', 'ego_mean_speed_mps', 'ego_min_speed_mps', 'ego_stopped_s', 'ego_path_deviation_max_m')


class CompareError(ValueError):
    pass


def _read(d, name):
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        raise CompareError(f'{d}: no {name} (not a finished replay)')
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def behaviour(run_dir: str) -> dict:
    """The driving numbers of one replay run (see the module docstring)."""
    from .. import criticality as C
    tr = _read(run_dir, 'trajectories.json')
    rp = _read(run_dir, 'replay.json')
    run = _read(run_dir, 'run.json')
    if run.get('status') != 'ok':
        raise CompareError(f'{run_dir}: the replay did not finish ok ({run.get("status")})')
    dt = float(tr['dt'])
    ego = np.asarray(tr['actors']['ego'], float)
    actors = {k: np.asarray(v, float) for k, v in tr['actors'].items() if k != 'ego' and v}
    events = (run.get('scenario') or {}).get('collisions') or []
    summ = C.summarise(ego, actors, dt, events)
    ov = summ['overall']
    sp = np.hypot(ego[:, 4], ego[:, 5])
    path = float(np.sum(np.hypot(np.diff(ego[:, 1]), np.diff(ego[:, 2])))) if len(ego) > 1 else 0.0
    dev = ((rp.get('ego_vs_log') or {}).get('path_deviation_m') or {}).get('max')
    m = {'min_ttc_s': ov.get('min_ttc_s'), 'min_ttc_actor': ov.get('min_ttc_actor'),
         'min_distance_m': ov.get('min_distance_m'), 'min_distance_actor': ov.get('min_distance_actor'),
         'max_drac_mps2': ov.get('max_drac_mps2'), 'min_pet_s': ov.get('min_pet_s'),
         'criticality': summ.get('criticality'), 'collision': 1.0 if ov.get('collision') else 0.0,
         'collided_with': ov.get('collided_with'), 'ego_path_m': round(path, 4),
         'ego_mean_speed_mps': round(float(sp.mean()), 4) if len(sp) else None,
         'ego_min_speed_mps': round(float(sp.min()), 4) if len(sp) else None,
         'ego_stopped_s': round(float(np.sum(sp < STOPPED_MPS) * dt), 4), 'ego_path_deviation_max_m': dev}
    return {'dir': os.path.abspath(run_dir), 'metrics': m, 'per_actor': summ['actors'],
            'replay': {k: rp.get(k) for k in ('changes', 'placement', 'steps', 'fps', 't_start')},
            'ego_driver': (rp.get('ego') or {}).get('driver'), 'log': rp.get('log'), 'dt': dt,
            'ego': ego, 'criticality_formula': summ.get('criticality_formula')}


def _same_timeline(a: dict, b: dict):
    ra, rb = a['replay'], b['replay']
    for k in ('steps', 'fps', 't_start'):
        if ra.get(k) != rb.get(k):
            raise CompareError(f'{b["dir"]} is not on {a["dir"]}\'s timeline ({k}: {rb.get(k)} vs {ra.get(k)})')
    la, lb = a.get('log') or {}, b.get('log') or {}
    if (la.get('sha256') and lb.get('sha256') and la['sha256'] != lb['sha256']) or la.get('name') != lb.get('name'):
        raise CompareError(f'{b["dir"]} replays another log ({lb.get("name")}) than {a["dir"]} ({la.get("name")})')


def _diff(a, b):
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 6)


def _boot(vals: list, n_boot: int, seed: int):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)]


def compare(original: str, changed: list, model: dict | None = None, workspace: str = '', camera: str | None = None,
            every: int = 1, n_boot: int = 1000, seed: int = 0, out_dir: str | None = None, progress=None) -> dict:
    """The comparison document (COMPARISON_SCHEMA); written to out_dir/comparison.json when given."""
    from ..provenance import block, write_json
    if not changed:
        raise CompareError('at least one changed replay is needed')
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    A = behaviour(original)
    Bs = [behaviour(d) for d in changed]
    for B in Bs:
        _same_timeline(A, B)
    pairs = []
    for j, B in enumerate(Bs):
        d = {k: _diff(A['metrics'].get(k), B['metrics'].get(k)) for k in METRICS}
        ea, eb = A['ego'], B['ego']
        n = min(len(ea), len(eb))
        pos = np.hypot(eb[:n, 1] - ea[:n, 1], eb[:n, 2] - ea[:n, 2])
        spd = np.hypot(eb[:n, 4], eb[:n, 5]) - np.hypot(ea[:n, 4], ea[:n, 5])
        k_pos = int(np.argmax(pos)) if n else 0
        k_spd = int(np.argmax(np.abs(spd))) if n else 0
        pairs.append({'changed': B['dir'], 'changes': B['replay'].get('changes'), 'ego_driver': B['ego_driver'],
                      'metrics': B['metrics'], 'difference': d,
                      'ego_step_by_step': {'max_position_difference_m': round(float(pos[k_pos]), 4) if n else None,
                                           'at_t': round(float(ea[k_pos, 0]), 4) if n else None,
                                           'max_speed_difference_mps': round(float(spd[k_spd]), 4) if n else None,
                                           'speed_at_t': round(float(ea[k_spd, 0]), 4) if n else None,
                                           'steps': n}})
        if progress:
            progress(0.3 * (j + 1) / len(Bs), f'behaviour {j + 1} of {len(Bs)}')
    intervals = None
    if len(pairs) >= 2:
        intervals = {'unit': 'pair (original, changed)', 'n_pairs': len(pairs), 'n_boot': n_boot, 'seed': seed,
                     'note': None if len(pairs) >= MIN_PAIRS else
                     f'only {len(pairs)} pairs: a bootstrap over so few has few distinct resamples, so its interval '
                     f'is not reliable; compare {MIN_PAIRS} or more changed runs',
                     'mean_difference': {}}
        for k in METRICS:
            vals = [p['difference'][k] for p in pairs]
            ok = [v for v in vals if v is not None]
            intervals['mean_difference'][k] = {
                'mean': round(float(np.mean(ok)), 6) if ok else None, 'n': len(ok),
                'ci95': _boot(vals, n_boot, seed)}
    doc = {'schema': COMPARISON_SCHEMA, 'original': A['dir'], 'changed': [B['dir'] for B in Bs],
           'log': A.get('log'), 'original_metrics': A['metrics'], 'pairs': pairs, 'intervals': intervals,
           'interval_note': None if intervals else
           'one changed run: a kinematic replay is deterministic, so one pair is one sample and no interval can be '
           'drawn; compare several changed runs (seeds of a change with a range, or several changes) for intervals',
           'method': {'pairing': 'by step: step i of each run shows the same log time',
                      'criticality': A.get('criticality_formula'),
                      'stopped_below_mps': STOPPED_MPS, 'min_pairs_for_reliable_intervals': MIN_PAIRS},
           'model': None}
    if model is not None:
        doc['model'] = compare_model(A['dir'], [B['dir'] for B in Bs], model, workspace, camera, every, n_boot, seed,
                                     out_dir, progress)
    ins = {'original/run.json': os.path.join(A['dir'], 'run.json')}
    for j, B in enumerate(Bs):
        ins[f'changed{j}/run.json'] = os.path.join(B['dir'], 'run.json')
    doc['provenance'] = block(ins, seed, {'n_boot': n_boot, 'camera': camera, 'every': every,
                                          'model': (model or {}).get('id')})
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_json(os.path.join(out_dir, 'comparison.json'), _clean(doc))
    if progress:
        progress(1.0, 'compared')
    return _clean(doc)


def compare_model(original: str, changed: list, model: dict, workspace: str = '', camera: str | None = None,
                  every: int = 1, n_boot: int = 1000, seed: int = 0, out_dir: str | None = None, progress=None):
    """A model on the original and each changed replay, paired by step (core.harness)."""
    from .. import harness as H

    def prog(lo, hi):
        return (lambda f, t: progress(lo + (hi - lo) * f, t)) if progress else None
    pa = os.path.join(out_dir, 'predictions-original.jsonl') if out_dir else None
    col_a = H.collect([original], model, workspace, camera, every, prog(0.3, 0.5), pa)
    fa_all, fb_all, per = [], [], []
    for j, d in enumerate(changed):
        pb = os.path.join(out_dir, f'predictions-changed{j}.jsonl') if out_dir else None
        col_b = H.collect([d], model, workspace, camera, every, prog(0.5 + 0.4 * j / len(changed),
                                                                    0.5 + 0.4 * (j + 1) / len(changed)), pb)
        by_step = {f['step']: f for f in col_a['frames']}
        fa, fb = [], []
        for f in col_b['frames']:
            a = by_step.get(f['step'])
            if a is None:
                continue
            key = f'pair{j}'
            fa.append(dict(a, image_id=f'{key}/{f["step"]}', drive=key))
            fb.append(dict(f, image_id=f'{key}/{f["step"]}', drive=key))
        if not fa:
            raise CompareError(f'{d}: no frame shares a step with the original')
        per.append({'changed': os.path.abspath(d), 'frames': len(fa),
                    'gt_objects': {'original': sum(len(f['gt']) for f in fa) if 'gt' in fa[0] else None,
                                   'changed': sum(len(f['gt']) for f in fb) if 'gt' in fb[0] else None}})
        fa_all += fa
        fb_all += fb
    res = H.compare(fa_all, fb_all, model['kind'], n_boot, seed)
    note = None
    if res.get('unit') == 'image':
        note = ('one pair of runs: the interval is over frames, which are not independent — it is too narrow; '
                'compare several changed runs for an interval over pairs')
    elif (res.get('n_units') or 0) < MIN_PAIRS:
        note = (f'only {res.get("n_units")} pairs of runs: the interval over pairs is not reliable '
                f'(fewer than {MIN_PAIRS})')
    return {'model': {k: model.get(k) for k in ('id', 'name', 'kind', 'runtime', 'entry', 'path')},
            'synthetic_model': col_a.get('synthetic'), 'cameras': col_a.get('cameras'), 'every': every,
            'pairs': per, 'comparison': res, 'note': note}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not isinstance(v, np.ndarray) or k != 'ego'}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o
