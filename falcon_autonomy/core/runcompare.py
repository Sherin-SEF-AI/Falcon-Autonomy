"""Side-by-side comparison of recorded runs (Data ▸ Compare; docs/runs.md, "Comparing runs").

Runs are aligned by **simulation time from each run's first frame**, never by step index: two runs at different
rates or of different lengths show, at time t, each run's frame nearest t (and nothing past a run's end). Every
number comes from what a run recorded; one it did not record is None with the reason, so the table can say "not
recorded" rather than show a zero.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import runindex as RI


def load_frames(run_dir: str) -> list:
    out = []
    try:
        with open(os.path.join(run_dir, 'frames.jsonl'), encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def times(frames: list) -> np.ndarray:
    """Seconds from the run's first frame (sim_t, else the step at an unknown rate: None)."""
    if not frames:
        return np.zeros(0)
    t0 = float(frames[0].get('sim_t') or 0.0)
    return np.array([float(f.get('sim_t') or 0.0) - t0 for f in frames])


def frame_at(ts: np.ndarray, t: float, tol: float | None = None) -> int | None:
    """The index of the frame nearest t, or None past the run's end (beyond its last frame by more than a step) or
    before its start."""
    if len(ts) == 0:
        return None
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
    tol = step if tol is None else tol
    if t > ts[-1] + max(tol, 1e-9) or t < -max(tol, 1e-9):
        return None
    return int(np.argmin(np.abs(ts - t)))


def cameras(frames: list) -> list:
    """The cameras with an image stream, in the run's order."""
    for f in frames:
        cams = sorted({k.split('/')[0] for k in (f.get('files') or {}) if k.endswith('/rgb')})
        if cams:
            return cams
    return []


def _read(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def events(run_dir: str, entry: dict | None = None) -> list:
    """The run's events on its own time axis: [{'t', 't1' (an interval's end, or None), 'kind', 'text'}]."""
    run = _read(os.path.join(run_dir, 'run.json'))
    e = entry or RI.summarise(run_dir) or {}
    out = []
    sres = run.get('scenario') or {}
    crit = sres.get('criticality') or {}
    for aid, a in (crit.get('actors') or {}).items():
        if a.get('collision') and a.get('collision_t') is not None:
            out.append({'t': float(a['collision_t']), 't1': None, 'kind': 'collision', 'text': f'collision with {aid}'})
        if a.get('min_ttc_t') is not None and a.get('min_ttc_s') is not None:
            out.append({'t': float(a['min_ttc_t']), 't1': None, 'kind': 'min_ttc',
                        'text': f'min TTC {a["min_ttc_s"]:.2f} s ({aid})'})
    for o in crit.get('other_collisions') or ():
        out.append({'t': float(o.get('t') or 0.0), 't1': None, 'kind': 'collision', 'text': 'collision'})
    for fired in sres.get('fired') or ():
        try:
            k, t = fired
            out.append({'t': float(t), 't1': None, 'kind': 'trigger', 'text': f'trigger {k} fired'})
        except (TypeError, ValueError):
            pass
    for t in e.get('teleports') or ():
        out.append({'t': float(t), 't1': None, 'kind': 'teleport', 'text': 'put back on the road'})
    odd = _read(os.path.join(run_dir, 'odd-monitor.json'))
    for s in odd.get('segments') or ():
        if not s.get('inside', True):
            out.append({'t': float(s['t0']), 't1': float(s['t1']), 'kind': 'odd_exit',
                        'text': 'outside the ODD: ' + '; '.join(s.get('reasons') or s.get('outside') or [])})
    return sorted(out, key=lambda x: x['t'])


def rule_metrics(run_dir: str) -> dict:
    """The traffic-rule check (core.rules, the rule set by the run's traffic side) and the ego's lane margin from the
    same trace. Slow on long runs: call it off the GUI thread."""
    from . import rules as RU
    try:
        tr = RU.trace_from_run(run_dir)
    except Exception as e:                           # noqa: BLE001 — a run the checker cannot read: say why
        return {'error': str(e)}
    rs = RU.load_ruleset(None, tr.handedness)
    res = RU.check(tr, rs)
    sig = RU.signals(tr)
    lm = np.asarray(sig.get('lane_margin_m'), float)
    known = lm[~np.isnan(lm)]
    viol = [{'t': float(v['start']), 't1': float(v['end']), 'kind': 'rule', 'text': r['title']}
            for r in res['rules'] for v in r['violations']]
    return {'ruleset': res['ruleset'].get('name'), 'violations': res['totals']['violations'],
            'per_km': res['totals']['per_km'], 'not_evaluable': res['totals']['not_evaluable'],
            'lane_margin_min': float(known.min()) if known.size else None,
            'lane_margin_median': float(np.median(known)) if known.size else None,
            'lane_share': float(known.size / max(lm.size, 1)), 'events': viol}


ROWS = (
    ('duration_s', 'Duration', 's', 1), ('step_s', 'Step', 's', 3), ('frames', 'Frames', '', 0),
    ('distance_m', 'Distance', 'm', 1), ('speed_mean_kmh', 'Mean speed', 'km/h', 1),
    ('speed_max_kmh', 'Max speed', 'km/h', 1), ('collisions', 'Collisions', '', 0),
    ('min_ttc_s', 'Minimum TTC', 's', 2), ('teleports', 'Put back on the road', '', 0),
    ('odd_outside_s', 'Outside the ODD', 's', 1), ('rule_violations', 'Rule violations', '', 0),
    ('lane_margin_min', 'Lane margin, minimum', 'm', 2), ('lane_margin_median', 'Lane margin, median', 'm', 2),
)


def table(entries: list, rules: list | None = None) -> list:
    """Rows [{'key', 'label', 'unit', 'cells': [{'value', 'text', 'delta'}]}] for these runs (index summaries) and
    their rule metrics (None while not computed). The delta is against the first run."""
    rules = rules or [None] * len(entries)
    rows = []
    for key, lab, unit, nd in ROWS:
        cells = []
        for e, rm in zip(entries, rules):
            v, why = _value(key, e, rm)
            cells.append({'value': v, 'text': _fmt(v, unit, nd) if v is not None else why, 'delta': None})
        a = cells[0]['value'] if cells else None
        for c in cells[1:]:
            if a is not None and c['value'] is not None:
                d = c['value'] - a
                c['delta'] = f'{d:+.{nd}f}' if nd else f'{int(round(d)):+d}'
        rows.append({'key': key, 'label': lab, 'unit': unit, 'cells': cells})
    return rows


def _value(key, e, rm):
    if key == 'step_s':
        return (1.0 / float(e['fps'])) if e.get('fps') else None, 'not recorded'
    if key == 'teleports':
        return float(len(e.get('teleports') or ())), ''
    if key in ('collisions', 'min_ttc_s'):
        if e.get('collisions') is None:
            return None, 'not recorded (no scenario)'
        if key == 'min_ttc_s' and e.get('min_ttc_s') is None:
            return None, 'never closing'
        return (float(e[key]) if e.get(key) is not None else None), ''
    if key == 'odd_outside_s':
        return e.get('odd_outside_s'), 'not monitored'
    if key in ('rule_violations', 'lane_margin_min', 'lane_margin_median'):
        if rm is None:
            return None, 'checking…'
        if rm.get('error'):
            return None, f'not evaluable: {rm["error"][:60]}'
        v = rm.get('violations' if key == 'rule_violations' else key)
        return (float(v) if v is not None else None), 'not recorded (no lane map)'
    v = e.get(key)
    return (float(v) if v is not None else None), 'not recorded'


def _fmt(v, unit, nd):
    s = f'{v:.{nd}f}' if nd else f'{int(round(v))}'
    return f'{s} {unit}'.strip()
