"""The safety-case report of a suite: scenario coverage against an operational design domain (ODD), results with
intervals, triggering conditions found, residual risks with their evidence, and the report's limits — in the spirit of
ISO 21448 (SOTIF). It is a report, not a certification claim (standard library + numpy).

The ODD (schema falcon.odd/1; example_odd() writes one to edit):

    {"schema": "falcon.odd/1", "name": "...", "about": "...",
     "dimensions": {
        "<name>": {"kind": "categorical", "values": [...]}            place, road_type, handedness, scenario
                | {"kind": "set", "values": [...]}                    actors: a run covers each kind it has
                | {"kind": "range", "lo": x, "hi": y, "bins": [edges], "unit": "..."}
                                                                      ego_speed_kmh, traffic_vehicles, weather and
                                                                      light: precipitation, fog_density, wetness,
                                                                      cloudiness, sun_altitude_deg
        ...},
     "pairs": [["actors", "ego_speed_kmh"], ...]}                     two-way coverage tables

A run's place in the ODD (run_coordinates) is read from its own config (stack.json): place — the CARLA map and the
project world's id (either may be listed); road_type — what the scenario director required at the start (heuristic:
"straight" always — the director starts on 150 m within 10°; "multi_lane" when it needs an adjacent lane in the same
direction; "two_way" when it needs, or an actor uses, an oncoming lane; "junction" for a junction turn); handedness;
the weather fields CARLA ran with (sun_altitude_deg is CARLA's sun_altitude_angle); traffic_vehicles; ego_speed_kmh
(the scenario's ego set speed); actors (the scenario's actor kinds); scenario (its id).

Coverage: per dimension, how many runs fell in each value or bin; values and bins no run reached are gaps; a run
outside a dimension's ODD values is counted apart (outside the ODD) and does not cover anything there. "share" is the
covered values and bins over all of them.

Residual risks: every scenario with a failure (its failure rate and Wilson interval, the failing runs); every scenario
with none whose failure rate may still be above max_failure_rate (5 % by default: a reporting threshold, not a validated
acceptance criterion) by the interval's upper bound; every ODD gap (no evidence); missed
control deadlines; failing parameter sets a rare-case search found; and the simulation's limits that bound all of the
evidence (docs/carla-lessons.md); and, first, every traffic rule a run violated.

Traffic rules (core.rules through core.stack.rule_summary, the suite's "rules" set or the one for the runs'
handedness): per rule the runs evaluated, the violations, per km with a bootstrap interval over runs, the runs
violating with Wilson's interval and the lowest robustness — beside the pass/fail results, not part of them; each
run's check is written to its folder (rules.json). Outputs: safety-case.json ("traffic_rules") and safety-case.html
(inline CSS and SVG, no network).

Time outside the ODD (core.oddmonitor.suite_outside, "odd_monitor"): the runtime ODD monitor's condition monitors over
each run's recorded frames — the weather, the ego's actual speed, place, road type and the rest — summed per scenario:
the time spent outside the ODD and why. Evidence gathered outside the ODD is not evidence about the ODD; any such time
is also listed among the residual risks.
"""
from __future__ import annotations

import html
import math
import os

from .provenance import block, read_json, write_json

ODD_SCHEMA = 'falcon.odd/1'
REPORT_SCHEMA = 'falcon.safety_case/1'
NOT_A_CLAIM = ('This is a test report generated from simulation runs. It is not a certification claim, not a safety '
               'case in the sense of a complete argument that the system is acceptably safe, and not evidence of '
               'conformance with ISO 21448, ISO 26262 or UL 4600.')
SIM_LIMITS = [
    ('Lidar and radar ignore weather', 'CARLA 0.9.16: no loss of range in rain or fog (docs/carla-lessons.md). '
                                       'Weather-related sensor limitations of a real lidar are not exercised.'),
    ('Camera rendering is not the real world', 'The sim-to-real gap of camera images was measured large on the '
                                               'reference machine (docs/sim-to-real.md); perception results do not '
                                               'transfer as they are.'),
    ('Kinematic motion', 'Scenario vehicles (the ego included) move by a kinematic bicycle model in kinematic '
                         'motion: no tyre, suspension or collision dynamics; braking and steering limits are the '
                         'model\'s (docs/scenarios.md).'),
    ('Collisions are measured on boxes', 'Contact is judged on 2D footprints (core.criticality); shapes, heights '
                                         'and contacts shorter than a step can be missed.'),
    ('Scenario actors are scripted', 'Other road users follow the scenario\'s timeline and do not react to the '
                                     'ego, except where the scenario says so.'),
]
SOTIF_NOT_COVERED = [
    'Hazard analysis and risk evaluation, and the acceptance criteria and validation targets a SOTIF argument needs '
    '(the pass criteria here are test criteria chosen for the suite, not validated acceptance criteria).',
    'The specification and its functional insufficiencies beyond what these scenarios exercise; reasonably '
    'foreseeable misuse by people; the human–machine interface.',
    'The unknown unsafe area (area 3) is only probed inside the parameter ranges of the scenarios searched; nothing '
    'outside them is explored.',
    'Real-world validation: sensors, vehicle dynamics, weather effects and other road users\' behaviour as they '
    'really are; field monitoring after release.',
    'Hardware and software faults (ISO 26262\'s scope) and cybersecurity (ISO/SAE 21434).',
    'Statistical arguments for residual risk (exposure, distance or time driven): the run counts here are far too '
    'small for one.',
]

ROAD_TYPES = ('straight', 'multi_lane', 'two_way', 'junction')


def example_odd() -> dict:
    """An ODD to start from: urban roads in CARLA's Town10 and an imported left-hand city, day, dry to moderate rain,
    light fog, 0–30 background vehicles, ego up to 60 km/h, the usual road users. Every number is an example to
    edit, not a recommendation."""
    return {'schema': ODD_SCHEMA, 'name': 'Urban day driving (example)',
            'about': 'An example ODD to edit: every value is illustrative.',
            'dimensions': {
                'place': {'kind': 'categorical', 'values': ['Town10HD_Opt', 'w-blr'],
                          'about': 'CARLA map names or project world ids (an imported Bengaluru world: w-blr)'},
                'road_type': {'kind': 'categorical', 'values': list(ROAD_TYPES)},
                'handedness': {'kind': 'categorical', 'values': ['RHT', 'LHT']},
                'scenario': {'kind': 'categorical', 'values': ['lead_brake', 'cut_in', 'pedestrian_crossing',
                                                               'stalled_vehicle', 'two_wheeler_filtering',
                                                               'jaywalker_behind_bus']},
                'actors': {'kind': 'set', 'values': ['car', 'truck', 'bus', 'van', 'motorcycle', 'bicycle',
                                                     'pedestrian']},
                'ego_speed_kmh': {'kind': 'range', 'lo': 0, 'hi': 60, 'bins': [0, 20, 40, 60], 'unit': 'km/h'},
                'traffic_vehicles': {'kind': 'range', 'lo': 0, 'hi': 30, 'bins': [0, 1, 10, 30]},
                'precipitation': {'kind': 'range', 'lo': 0, 'hi': 60, 'bins': [0, 10, 30, 60], 'unit': '%'},
                'fog_density': {'kind': 'range', 'lo': 0, 'hi': 20, 'bins': [0, 5, 20], 'unit': '%'},
                'sun_altitude_deg': {'kind': 'range', 'lo': 10, 'hi': 90, 'bins': [10, 30, 60, 90], 'unit': '°'}},
            'pairs': [['actors', 'ego_speed_kmh'], ['scenario', 'precipitation']]}


def validate_odd(odd: dict) -> list:
    out = []
    if odd.get('schema') != ODD_SCHEMA:
        out.append(f'schema {odd.get("schema")!r}, expected {ODD_SCHEMA}')
    dims = odd.get('dimensions') or {}
    if not dims:
        out.append('no dimensions')
    for k, d in dims.items():
        kind = d.get('kind')
        if kind in ('categorical', 'set'):
            if not d.get('values'):
                out.append(f'{k}: no values')
        elif kind == 'range':
            try:
                lo, hi = float(d['lo']), float(d['hi'])
                edges = [float(x) for x in d.get('bins') or [lo, hi]]
            except (KeyError, TypeError, ValueError):
                out.append(f'{k}: a range needs numbers lo, hi (and bins)')
                continue
            if lo > hi or edges != sorted(edges) or len(edges) < 2 or edges[0] > lo or edges[-1] < hi:
                out.append(f'{k}: bins must be ascending edges covering lo..hi')
        else:
            out.append(f'{k}: kind {kind!r} (categorical, set or range)')
    for p in odd.get('pairs') or []:
        if len(p) != 2 or any(x not in dims for x in p):
            out.append(f'pair {p}: two dimensions of the ODD')
    return out


def run_coordinates(cfg: dict) -> dict:
    """A run's coordinates in ODD terms, from its config (module docstring says how each is read)."""
    scn = cfg.get('scenario') or {}
    w = cfg.get('weather') or {}
    needs = set(scn.get('needs') or [])
    lanes = {(a.get('place') or {}).get('lane') for a in scn.get('actors') or []}
    road = ['straight']
    if 'adjacent_same' in needs or lanes & {'adjacent', 'line'}:
        road.append('multi_lane')
    if 'opposite' in needs or 'opposite' in lanes:
        road.append('two_way')
    if 'junction_turn' in needs or (scn.get('ego') or {}).get('route') == 'turn_across':
        road.append('junction')
    place = [p for p in (cfg.get('town'), os.path.basename(cfg.get('xodr') or '') or None,
                         (cfg.get('project') or {}).get('world')) if p]
    ego_speed = (scn.get('ego') or {}).get('speed_kmh')
    return {'place': place, 'road_type': road, 'handedness': [cfg.get('handedness', 'RHT')],
            'scenario': [scn.get('id')] if scn.get('id') else [],
            'actors': sorted({a.get('kind') for a in scn.get('actors') or [] if a.get('kind')}),
            'ego_speed_kmh': float(ego_speed) if isinstance(ego_speed, (int, float)) else None,
            'traffic_vehicles': float((cfg.get('traffic') or {}).get('vehicles', 0)),
            'precipitation': _num(w.get('precipitation')), 'fog_density': _num(w.get('fog_density')),
            'wetness': _num(w.get('wetness')), 'cloudiness': _num(w.get('cloudiness')),
            'sun_altitude_deg': _num(w.get('sun_altitude_angle'))}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _labels(d: dict) -> list:
    if d['kind'] == 'range':
        e = [float(x) for x in d.get('bins') or [d['lo'], d['hi']]]
        u = f' {d["unit"]}' if d.get('unit') else ''
        return [f'{e[i]:g}–{e[i + 1]:g}{u}' for i in range(len(e) - 1)]
    return [str(v) for v in d['values']]


def _cells(d: dict, v) -> tuple:
    """(indices of the dimension's values/bins this coordinate covers, outside: bool)."""
    if d['kind'] == 'range':
        if v is None:
            return [], True
        lo, hi = float(d['lo']), float(d['hi'])
        if not lo <= v <= hi:
            return [], True
        e = [float(x) for x in d.get('bins') or [lo, hi]]
        for i in range(len(e) - 1):
            if e[i] <= v < e[i + 1] or (i == len(e) - 2 and v == e[-1]):
                return [i], False
        return [], True
    vals = [str(x) for x in d['values']]
    got = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v]) if x is not None]
    idx = sorted({vals.index(x) for x in got if x in vals})
    if d['kind'] == 'categorical':                    # one value; several only when the run has aliases (place)
        return idx, not idx
    return idx, any(x not in vals for x in got)


def coverage(odd: dict, coords: list) -> dict:
    """Per dimension the runs per value or bin, the gaps and the runs outside; two-way tables for the ODD's pairs."""
    dims = {}
    total = covered = 0
    for name, d in odd['dimensions'].items():
        labels = _labels(d)
        counts = [0] * len(labels)
        outside = 0
        for c in coords:
            idx, out = _cells(d, c.get(name))
            for i in idx:
                counts[i] += 1
            outside += bool(out)
        total += len(labels)
        covered += sum(1 for n in counts if n)
        dims[name] = {'kind': d['kind'], 'labels': labels, 'counts': counts, 'outside': outside,
                      'gaps': [labels[i] for i, n in enumerate(counts) if n == 0]}
    pairs = []
    for a, b in odd.get('pairs') or []:
        da, db = odd['dimensions'][a], odd['dimensions'][b]
        la, lb = _labels(da), _labels(db)
        M = [[0] * len(lb) for _ in la]
        for c in coords:
            ia, _ = _cells(da, c.get(a))
            ib, _ = _cells(db, c.get(b))
            for i in ia:
                for j in ib:
                    M[i][j] += 1
        cells = len(la) * len(lb)
        pairs.append({'dims': [a, b], 'rows': la, 'cols': lb, 'counts': M,
                      'covered': sum(1 for r in M for n in r if n), 'cells': cells})
    return {'dimensions': dims, 'pairs': pairs, 'values': total, 'covered': covered,
            'share': covered / total if total else 0.0, 'runs': len(coords)}


def triggering_conditions(results: dict, studies: list) -> list:
    """Failing parameter sets: the suite's failing runs, and the failures of rare-case studies (studies/<name>/:
    results.jsonl and search-*.json) — with, per parameter, the range the failures span against the range tried."""
    out = []
    by = {}
    for f in results.get('failures') or []:
        by.setdefault(f['scenario'], []).append(f)
    for scn, fs in sorted(by.items()):
        out.append({'source': 'suite', 'scenario': scn, 'failures': len(fs),
                    'cases': [{'seed': f['seed'], 'values': f.get('values'), 'run': f.get('run'),
                               'criticality': f.get('criticality')} for f in fs[:20]]})
    for d in studies:
        try:
            st = read_json(os.path.join(d, 'study.json'))
        except (OSError, ValueError):
            continue
        rows = []
        p = os.path.join(d, 'results.jsonl')
        if os.path.isfile(p):
            import json
            with open(p, encoding='utf-8') as fh:
                rows = [json.loads(x) for x in fh if x.strip()]
        thr = 0.5
        for f in sorted(os.listdir(d)):
            if f.startswith('search-') and f.endswith('.json'):
                thr = float(read_json(os.path.join(d, f)).get('threshold', thr))
        fails = [r for r in rows if r.get('status') == 'ok' and float(r.get('objective', 0.0)) >= thr]
        span = {}
        for k in sorted({k for r in rows for k in (r.get('values') or {})}):
            tried = [r['values'][k] for r in rows if isinstance((r.get('values') or {}).get(k), (int, float))]
            failed = [r['values'][k] for r in fails if isinstance((r.get('values') or {}).get(k), (int, float))]
            if tried:
                span[k] = {'tried': [min(tried), max(tried)], 'failing': [min(failed), max(failed)] if failed
                           else None}
        out.append({'source': f'study {os.path.basename(os.path.normpath(d))}',
                    'scenario': (st.get('scenario') or {}).get('id'), 'evaluations': len(rows),
                    'threshold': thr, 'failures': len(fails), 'parameters': span,
                    'cases': [{'values': r.get('values'), 'run': r.get('run'), 'criticality': r.get('objective')}
                              for r in sorted(fails, key=lambda r: -r.get('objective', 0.0))[:20]]})
    return out


def residual_risks(results: dict, cov: dict, triggers: list, max_failure_rate: float = 0.05) -> list:
    out = []
    for scn, g in sorted((results.get('scenarios') or {}).items()):
        n = g.get('runs') or 0
        if not n:
            continue
        lo, hi = g['pass_rate_ci95']
        if g['failed']:
            out.append({'kind': 'failure', 'scenario': scn, 'severity': 'high',
                        'text': f'{g["name"]}: the stack failed {g["failed"]} of {n} runs (failure rate '
                                f'{g["failed"] / n:.0%}, 95 % interval {1 - hi:.0%}–{1 - lo:.0%}).',
                        'evidence': [f['run'] for f in results.get('failures') or [] if f['scenario'] == scn]})
        elif 1 - lo > max_failure_rate:
            out.append({'kind': 'insufficient_evidence', 'scenario': scn, 'severity': 'medium',
                        'text': f'{g["name"]}: no failure in {n} runs, but with so few runs the failure rate may '
                                f'still be as high as {1 - lo:.0%} (Wilson 95 % upper bound) — above '
                                f'{max_failure_rate:.0%}.', 'evidence': []})
        if g.get('missed_deadlines'):
            out.append({'kind': 'timing', 'scenario': scn, 'severity': 'medium',
                        'text': f'{g["name"]}: {g["missed_deadlines"]} missed control deadlines: the stack did not '
                                'answer in time and the last command was held (or the ego braked).', 'evidence': []})
    for name, d in cov['dimensions'].items():
        if d['gaps']:
            out.append({'kind': 'odd_gap', 'dimension': name, 'severity': 'medium',
                        'text': f'ODD {name}: not exercised — {", ".join(d["gaps"])}. There is no evidence about the '
                                'stack there.', 'evidence': []})
    for t in triggers:
        if t['source'].startswith('study') and t['failures']:
            out.append({'kind': 'triggering_condition', 'scenario': t['scenario'], 'severity': 'high',
                        'text': f'{t["source"]}: {t["failures"]} of {t["evaluations"]} searched parameter sets of '
                                f'{t["scenario"]} were critical (criticality ≥ {t["threshold"]:g}).',
                        'evidence': [c['run'] for c in t['cases']]})
    for title, text in SIM_LIMITS:
        out.append({'kind': 'simulation_limit', 'severity': 'context', 'text': f'{title}. {text}', 'evidence': []})
    return out


def build_report(suite_dir: str, odd: dict, out_dir: str, studies=(), odd_path: str = '',
                 max_failure_rate: float = 0.05) -> dict:
    """safety-case.json and safety-case.html of a suite (its results.json, results.jsonl and each run's stack.json)."""
    from .stack import read_results, summarise
    bad = validate_odd(odd)
    if bad:
        raise ValueError('the ODD: ' + '; '.join(bad))
    rows = read_results(suite_dir)
    if not rows:
        raise ValueError(f'{suite_dir} has no finished runs (results.jsonl)')
    suite = read_json(os.path.join(suite_dir, 'suite.json')) if os.path.isfile(os.path.join(suite_dir, 'suite.json')) \
        else {}
    results = summarise(rows)
    coords = []
    for r in rows:
        if r.get('status') != 'ok':
            continue
        p = os.path.join(r['run'], 'stack.json')
        if os.path.isfile(p):
            c = run_coordinates(read_json(p))
            c['_run'] = r['run']
            coords.append(c)
    cov = coverage(odd, coords)
    trig = triggering_conditions(results, list(studies))
    risks = residual_risks(results, cov, trig, max_failure_rate)
    from .oddmonitor import suite_outside
    outside = suite_outside(rows, odd)
    for scn, g in sorted(outside['scenarios'].items()):
        if g['time_outside_s'] > 0:
            risks.insert(0, {'kind': 'outside_odd', 'scenario': scn, 'severity': 'medium',
                             'text': f'{g["name"]}: {g["time_outside_s"]:.1f} s of {g["time_s"]:.1f} s '
                                     f'({g["share"]:.0%}) in {g["runs_outside"]} of {g["runs"]} runs were outside the '
                                     f'ODD ({", ".join(sorted(g["by_dimension_s"]))}): that evidence is not about '
                                     'the ODD.', 'evidence': []})
    from .stack import rule_summary
    rules = rule_summary(rows, suite.get('rules') or None)
    for rr in (rules or {}).get('rules') or ():
        if rr['violations']:
            risks.insert(0, {'kind': 'traffic_rule', 'severity': 'medium',
                             'text': f'{rr["title"]} ({rr["id"]}): {rr["violations"]} violation(s) in '
                                     f'{rr["runs_violating"]} of {rr["runs_evaluated"]} runs'
                                     + (f', {rr["per_km"]:.2f} per km' if rr['per_km'] is not None else ''),
                             'evidence': []})
    inputs = {'suite_results': os.path.join(suite_dir, 'results.jsonl')}
    if odd_path:
        inputs['odd'] = odd_path
    for k, d in enumerate(studies):
        inputs[f'study_{k}'] = os.path.join(d, 'results.jsonl')
    rep = {'schema': REPORT_SCHEMA, 'title': f'Safety-case report: {suite.get("name", os.path.basename(suite_dir))}',
           'not_a_claim': NOT_A_CLAIM, 'suite': {'name': suite.get('name'), 'dir': suite_dir,
                                                 'driver': (suite.get('stack') or {}).get('driver'),
                                                 'criteria': suite.get('criteria')},
           'odd': odd, 'coverage': cov, 'results': results, 'triggering_conditions': trig, 'residual_risks': risks,
           'traffic_rules': rules, 'odd_monitor': outside,
           'limits': {'sotif_not_covered': SOTIF_NOT_COVERED, 'simulation': [{'title': t, 'text': x}
                                                                            for t, x in SIM_LIMITS],
                      'road_type': 'heuristic: the road features the scenario director required at the start, not a '
                                   'classification of the whole route',
                      'max_failure_rate': {'value': max_failure_rate,
                                           'about': 'the failure rate above which a scenario\'s evidence is called '
                                                    'insufficient: a reporting threshold chosen for this report, not '
                                                    'a validated acceptance criterion'}},
           'provenance': block(inputs, seed=0)}
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, 'safety-case.json'), rep)
    with open(os.path.join(out_dir, 'safety-case.html'), 'w', encoding='utf-8') as fh:
        fh.write(render_html(rep))
    return rep


# --------------------------------------------------------------------------------------------------------------- HTML
CSS = """
:root{--bg:#fbfaf7;--fg:#1f2328;--muted:#5f6670;--line:#d8d4cc;--panel:#ffffff;--pass:#2f7d4f;--fail:#b3261e;
--warn:#a35f00;--acc:#2b5d9b;--cell0:#f1efe9}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e8e6e1;--muted:#a3a7ad;--line:#33373e;--panel:#1d2025;
--pass:#5cb87e;--fail:#ef6b60;--warn:#e0a24a;--acc:#7aa7e0;--cell0:#23262c}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 8px;border-bottom:1px solid var(--line);
padding-bottom:4px}h3{font-size:16px;margin:18px 0 6px}
.claim{border:1px solid var(--warn);border-left:5px solid var(--warn);padding:10px 14px;background:var(--panel);
margin:14px 0}
.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
.pass{color:var(--pass);font-weight:600}.fail{color:var(--fail);font-weight:600}
.tag{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 6px;font-size:12px;
color:var(--muted)}.sev-high{color:var(--fail)}.sev-medium{color:var(--warn)}
.cells td{text-align:center}.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px 14px;min-width:140px}
.kpi b{display:block;font-size:22px}code{font-size:13px}svg text{fill:var(--fg);font-size:12px}
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _pct(v) -> str:
    return '—' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:.0f} %'


def _bars_svg(per: dict) -> str:
    """Pass rate per scenario: a bar and its Wilson 95 % interval as a whisker."""
    items = [(k, g) for k, g in per.items() if g.get('runs')]
    if not items:
        return ''
    row, left, width = 26, 230, 420
    h = row * len(items) + 30
    out = [f'<svg viewBox="0 0 {left + width + 70} {h}" width="100%" role="img" aria-label="pass rate per scenario">']
    for x in (0, 0.5, 1.0):
        X = left + width * x
        out.append(f'<line x1="{X}" y1="4" x2="{X}" y2="{h - 22}" stroke="var(--line)"/>'
                   f'<text x="{X}" y="{h - 6}" text-anchor="middle">{int(x * 100)} %</text>')
    for i, (k, g) in enumerate(items):
        y = 6 + i * row
        p = g['pass_rate']
        lo, hi = g['pass_rate_ci95']
        col = 'var(--pass)' if g['failed'] == 0 else 'var(--fail)'
        out.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end">{_e(g.get("name", k))[:34]}</text>'
                   f'<rect x="{left}" y="{y + 3}" width="{width * p:.1f}" height="14" fill="{col}" opacity="0.8"/>'
                   f'<line x1="{left + width * lo:.1f}" y1="{y + 10}" x2="{left + width * hi:.1f}" y2="{y + 10}" '
                   f'stroke="var(--fg)" stroke-width="1.5"/>'
                   f'<line x1="{left + width * lo:.1f}" y1="{y + 5}" x2="{left + width * lo:.1f}" y2="{y + 15}" '
                   f'stroke="var(--fg)"/><line x1="{left + width * hi:.1f}" y1="{y + 5}" x2="{left + width * hi:.1f}" '
                   f'y2="{y + 15}" stroke="var(--fg)"/>'
                   f'<text x="{left + width + 8}" y="{y + 14}">{g["passed"]}/{g["runs"]}</text>')
    out.append('</svg>')
    return ''.join(out)


def _values(v) -> str:
    return ', '.join(f'{k}={x:.3g}' if isinstance(x, float) else f'{k}={x}' for k, x in sorted((v or {}).items()))


def _shade(n: int, top: int) -> str:
    if n == 0:
        return 'background:var(--cell0)'
    a = 0.25 + 0.6 * n / max(top, 1)
    return f'background:rgba(43,93,155,{a:.2f});color:#fff'


def _rules_html(rules: dict | None) -> str:
    """The traffic-rule section (core.rules through core.stack.rule_summary): per rule the runs evaluated, the
    violations, per km with the bootstrap interval over runs, the runs violating with Wilson's interval."""
    if not rules:
        return ('<h2>Traffic rules</h2><p class="muted">Not checked: no run could be read by the rule checker '
                '(docs/traffic-rules.md).</p>')
    rs = rules.get('ruleset') or {}
    out = [f'<h2>Traffic rules</h2><p class="muted">Rule set {_e(rs.get("name") or rs.get("id"))} '
           f'(<code>{_e((rs.get("sha256") or "")[:12])}</code>), checked as signal-temporal-logic formulas on each '
           f'run\'s recorded state; beside the pass/fail criteria, not part of them. {rules["runs"]} runs, '
           f'{rules["distance_km"]:.3f} km. Per km: {_e(rules.get("interval"))}. Rules marked heuristic use Falcon\'s '
           f'thresholds; facts not recorded (turn indicators, light states in drives) are not checked.</p>',
           '<table><tr><th>rule</th><th class="num">runs evaluated</th><th class="num">violations</th>'
           '<th class="num">per km (95 %)</th><th class="num">runs violating (95 %)</th>'
           '<th class="num">lowest robustness</th></tr>']
    for r in rules['rules']:
        pk = '—' if r['per_km'] is None else f'{r["per_km"]:.2f}'
        if r.get('per_km_ci95') and r['per_km_ci95'][0] is not None:
            pk += f' ({r["per_km_ci95"][0]:.2f}–{r["per_km_ci95"][1]:.2f})'
        sh = r.get('share_ci95') or [None, None]
        rob = '—' if r['min_robustness'] is None else f'{r["min_robustness"]:.3f}'
        out.append(f'<tr><td>{_e(r["title"])} <span class="tag">{_e(r["id"])}</span></td><td class="num">'
                   f'{r["runs_evaluated"]}</td><td class="num{" fail" if r["violations"] else ""}">{r["violations"]}'
                   f'</td><td class="num">{pk}</td><td class="num">{r["runs_violating"]} ({_pct(sh[0])}–'
                   f'{_pct(sh[1])})</td><td class="num">{rob}</td></tr>')
    out.append('</table>')
    return ''.join(out)


def _odd_monitor_html(om: dict | None) -> str:
    """The time each scenario's runs spent outside the ODD (core.oddmonitor.suite_outside)."""
    if not om:
        return ''
    out = [f'<h2>Time outside the ODD</h2><p class="muted">The runtime ODD monitor\'s condition monitors over each '
           f'run\'s recorded frames ({om["runs"]} runs, {om["time_s"]:.1f} s simulated): weather and light from the '
           f'run, the ego\'s actual speed, place, road type, actors. Outside the ODD in total: '
           f'{om["time_outside_s"]:.1f} s ({_pct(om["share"])}).'
           + (f' {om["unread"]} runs could not be read.' if om.get('unread') else '') + '</p>',
           '<table><tr><th>scenario</th><th class="num">runs outside</th><th class="num">time outside</th>'
           '<th class="num">share</th><th>why (seconds)</th></tr>']
    for k, g in sorted(om['scenarios'].items()):
        why = ', '.join(f'{d} {s:.1f}' for d, s in sorted(g['by_dimension_s'].items())) or '—'
        out.append(f'<tr><td>{_e(g.get("name") or k)}</td><td class="num">{g["runs_outside"]} of {g["runs"]}</td>'
                   f'<td class="num{" fail" if g["time_outside_s"] else ""}">{g["time_outside_s"]:.1f} s</td>'
                   f'<td class="num">{_pct(g["share"])}</td><td>{_e(why)}</td></tr>')
    out.append('</table>')
    return ''.join(out)


def render_html(rep: dict) -> str:
    res, cov = rep['results'], rep['coverage']
    o = res['overall']
    p = []
    p.append(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
             f'content="width=device-width,initial-scale=1"><title>{_e(rep["title"])}</title><style>{CSS}</style>'
             f'</head><body><main><h1>{_e(rep["title"])}</h1><p class="muted">ODD: {_e(rep["odd"].get("name"))} · '
             f'stack: {_e(rep["suite"].get("driver"))} · generated {_e(rep["provenance"]["created"])} by Falcon '
             f'Autonomy {_e(rep["provenance"]["app_version"])}</p><div class="claim"><b>Not a certification claim.</b>'
             f' {_e(rep["not_a_claim"])}</div>')
    ci = o.get('pass_rate_ci95') or [None, None]
    p.append('<div class="kpis">'
             f'<div class="kpi"><span class="muted">runs</span><b>{o["runs"]}</b></div>'
             f'<div class="kpi"><span class="muted">passed</span><b>{o["passed"]}</b></div>'
             f'<div class="kpi"><span class="muted">pass rate (95 % interval)</span><b>{_pct(o["pass_rate"])}</b>'
             f'<span class="muted">{_pct(ci[0])}–{_pct(ci[1])}</span></div>'
             f'<div class="kpi"><span class="muted">ODD values covered</span><b>{_pct(cov["share"])}</b>'
             f'<span class="muted">{cov["covered"]} of {cov["values"]}</span></div>'
             f'<div class="kpi"><span class="muted">collisions</span><b>{o["collisions"]}</b></div></div>')
    p.append('<h2>Results per scenario</h2><p class="muted">Bars: the pass rate; whiskers: its Wilson 95 % interval. '
             f'Criteria: <code>{_e(rep["suite"].get("criteria"))}</code>.</p>')
    p.append(_bars_svg(res['scenarios']))
    p.append('<table><tr><th>scenario</th><th class="num">runs</th><th class="num">passed</th><th class="num">pass '
             'rate (95 %)</th><th class="num">mean criticality (95 %)</th><th class="num">min TTC s</th>'
             '<th class="num">collisions</th><th class="num">missed deadlines</th></tr>')
    for k, g in res['scenarios'].items():
        c = g.get('criticality') or {}
        cci = c.get('ci95')
        pr = g['pass_rate_ci95'] or [None, None]
        crit = '—' if not c else f'{c["mean"]:.3f}' + (f' ({cci[0]:.3f}–{cci[1]:.3f})' if cci else '')
        ttc = '—' if g['min_ttc_s'] is None else f'{g["min_ttc_s"]:.2f}'
        p.append(f'<tr><td>{_e(g.get("name", k))}</td><td class="num">{g["runs"]}</td><td class="num">{g["passed"]}'
                 f'</td><td class="num">{_pct(g["pass_rate"])} ({_pct(pr[0])}–{_pct(pr[1])})</td>'
                 f'<td class="num">{crit}</td><td class="num">{ttc}</td><td class="num">{g["collisions"]}</td>'
                 f'<td class="num">{g["missed_deadlines"]}</td></tr>')
    p.append('</table>')
    if res.get('failures'):
        p.append('<h3>Failing runs</h3><table><tr><th>scenario</th><th class="num">seed</th><th>failed checks</th>'
                 '<th>run</th></tr>')
        for f in res['failures'][:50]:
            bad = '; '.join(f'{c["name"]}: {c["detail"]}' for c in f.get('checks') or [] if not c.get('ok'))
            p.append(f'<tr><td>{_e(f["scenario"])}</td><td class="num">{f["seed"]}</td><td class="fail">{_e(bad)}'
                     f'</td><td><code>{_e(f.get("run"))}</code></td></tr>')
        p.append('</table>')
    p.append(_rules_html(rep.get('traffic_rules')))
    p.append(_odd_monitor_html(rep.get('odd_monitor')))
    p.append('<h2>ODD coverage</h2><p class="muted">Runs per value or bin of each dimension of the ODD; an empty cell '
             'is a gap. Runs outside a dimension\'s values are counted apart.</p>')
    for name, d in cov['dimensions'].items():
        top = max(d['counts'] or [0])
        outside = f' <span class="tag">{d["outside"]} outside</span>' if d['outside'] else ''
        p.append(f'<h3>{_e(name)} <span class="tag">{_e(d["kind"])}</span>{outside}</h3>'
                 '<table class="cells"><tr>' + ''.join(f'<th>{_e(lb)}</th>' for lb in d['labels']) + '</tr><tr>'
                 + ''.join(f'<td style="{_shade(n, top)}">{n}</td>' for n in d['counts']) + '</tr></table>')
    for pr in cov['pairs']:
        top = max([n for r in pr['counts'] for n in r] or [0])
        p.append(f'<h3>{_e(pr["dims"][0])} × {_e(pr["dims"][1])} <span class="tag">{pr["covered"]} of {pr["cells"]} '
                 'cells</span></h3><table class="cells"><tr><th></th>' + ''.join(f'<th>{_e(c)}</th>'
                                                                                for c in pr['cols']) + '</tr>')
        for lb, r in zip(pr['rows'], pr['counts']):
            p.append(f'<tr><th>{_e(lb)}</th>' + ''.join(f'<td style="{_shade(n, top)}">{n}</td>' for n in r)
                     + '</tr>')
        p.append('</table>')
    p.append('<h2>Triggering conditions found</h2>')
    if not rep['triggering_conditions']:
        p.append('<p class="muted">None: no run failed and no rare-case study was given.</p>')
    for t in rep['triggering_conditions']:
        p.append(f'<h3>{_e(t["source"])}: {_e(t["scenario"])} <span class="tag">{t["failures"]} failing</span></h3>')
        if t.get('parameters'):
            p.append('<table><tr><th>parameter</th><th>range searched</th><th>range of the failures</th></tr>')
            for k, s in t['parameters'].items():
                f = s['failing']
                p.append(f'<tr><td>{_e(k)}</td><td>{s["tried"][0]:.3g} – {s["tried"][1]:.3g}</td><td>'
                         f'{"—" if f is None else f"{f[0]:.3g} – {f[1]:.3g}"}</td></tr>')
            p.append('</table>')
        rows = ''.join(f'<tr><td><code>{_e(_values(c.get("values")))}</code></td><td class="num">'
                       f'{"" if c.get("criticality") is None else format(c["criticality"], ".3f")}</td></tr>'
                       for c in t['cases'][:10])
        if rows:
            p.append(f'<table><tr><th>parameters</th><th class="num">criticality</th></tr>{rows}</table>')
    p.append('<h2>Residual risks</h2><table><tr><th>kind</th><th>risk</th><th>evidence</th></tr>')
    for r in rep['residual_risks']:
        ev = '<br>'.join(f'<code>{_e(x)}</code>' for x in r['evidence'][:5]) + (
            f'<br><span class="muted">… {len(r["evidence"]) - 5} more</span>' if len(r['evidence']) > 5 else '')
        p.append(f'<tr><td><span class="tag sev-{_e(r["severity"])}">{_e(r["kind"])}</span></td><td>{_e(r["text"])}'
                 f'</td><td>{ev or "<span class=muted>—</span>"}</td></tr>')
    p.append('</table><h2>Limits of this report</h2><div class="claim">' + _e(rep['not_a_claim']) + '</div>'
             '<h3>Not covered (ISO 21448 activities and areas)</h3><ul>'
             + ''.join(f'<li>{_e(x)}</li>' for x in rep['limits']['sotif_not_covered']) + '</ul>'
             '<h3>Limits of the simulation evidence</h3><ul>'
             + ''.join(f'<li><b>{_e(x["title"])}.</b> {_e(x["text"])}</li>' for x in rep['limits']['simulation'])
             + f'</ul><p class="muted">Road type is {_e(rep["limits"]["road_type"])}.</p>')
    pv = rep['provenance']
    p.append('<h2>Provenance</h2><table>' + ''.join(
        f'<tr><td>{_e(k)}</td><td><code>{_e(v.get("path"))}</code></td><td><code>{_e(v.get("sha256"))}</code></td></tr>'
        for k, v in pv['inputs'].items()) + f'</table><p class="muted">{_e(pv["app"])} {_e(pv["app_version"])}, '
        f'Python {_e(pv["host"].get("python"))}, {_e(pv["host"].get("platform"))}, {_e(pv["created"])}.</p>')
    p.append('</main></body></html>\n')
    return ''.join(p)
