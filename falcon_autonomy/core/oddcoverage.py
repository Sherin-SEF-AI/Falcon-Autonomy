"""Coverage of the operating domain: a grid of the ODD — weather × road type × actor kind × speed band × light — with
the tested scenario runs in each cell, the failures among them, which cells are untested or have too few runs to bound
their failure rate, the failure-rate estimates of core.failrate per cell, and a plan of scenarios and rare-case
searches for the cells that need runs (standard library + numpy). docs/odd-coverage.md is the manual.

The cells. Each scenario run is placed by its own config (drive.json, a suite's stack.json), read with the safety
case's run_coordinates (core.safetycase):
* weather — from the CARLA weather the run used, in this order (heuristic thresholds, stated): rain when precipitation
  ≥ 10; fog when fog_density ≥ 20 and the fog starts within 50 m (fog_distance; CARLA's ClearNight preset has density
  60 starting at 75 m and is not called fog); dust when dust_storm ≥ 50; wet when wetness or precipitation_deposits
  ≥ 40; cloudy when cloudiness ≥ 50; else clear;
* light — the sun's altitude as core.weather.describe reads it: night below −6°, twilight to 6°, low sun to 20°, day;
* road — the most specific road feature the scenario required at its start (junction > two-way > multi-lane >
  straight; the safety case's heuristic: what the director required, not the whole route);
* actor — each kind of the scenario's actors (a run with a bus and a pedestrian is a run in both kinds' cells, as the
  safety case counts its "set" dimension);
* speed — the ego's set speed in bands (0–20, 20–40, 40–60, 60–80, 80–130 km/h).
Runs without a scenario (free drives) are not tests of an encounter and are not counted. The domain is every
combination of Falcon's labels — or, when the workspace has an ODD (odd.json, or the Loop's; core.oddmonitor), its
actor kinds, road types and speed bins, and the weather and light classes whose representative conditions (CELL_WEATHER,
CELL_SUN) lie inside its weather and sun ranges.

What counts. Every finished scenario run in the workspace: rare-case studies (studies/*: sweeps, searches and
failure-rate estimates; a failure is the study's objective ≥ its threshold, as the safety case's triggering conditions
read it), stack suites (suites/*: a failure is a failed pass criterion, as core.stack judges it) and single scenario
runs (runs/*: the scenario's own pass criteria).

The rule (stated, and the user's to change): a cell has ENOUGH runs when the one-sided Clopper–Pearson upper bound on
its failure rate at the confidence (default 95 %) is at most the target (default 5 % per encounter, the safety case's
reporting threshold) — with no failure that is n ≥ ln(1 − c) / ln(1 − target) runs (59 for 5 %; the rule of three:
≈ 3 / target); TOO FEW below that (the runs it still needs, if none of them fail, are given); ABOVE TARGET when even
the lower bound is above the target; UNTESTED with none. The bound treats the runs as independent draws of the cell's
encounters: a search's runs are not (they are concentrated where failures are likely), so its failures over-state the
rate (pessimistic) and its passes are not a random sample — the sources are shown per cell.

Filling the gaps (fill_plan): for each untested or too-few cell (untested first, then the largest shortfall), a
scenario of the library or the project whose actors include the cell's kind, whose road requirement is the cell's road
type and whose ego speed can lie in the cell's band — copied as a project scenario with the cell's weather and light
(CELL_WEATHER, CELL_SUN) and its ego-speed parameter narrowed to the band — and a rare-case search of it (sim.studio,
`falcon search`; or a sweep, whose random runs are the independent draws the rule assumes). A cell no scenario can
produce is listed as such.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

from . import failrate as FR

SCHEMA = 'falcon.odd_coverage/1'
DIMENSIONS = ('weather', 'road', 'actor', 'speed', 'light')
WEATHER = ('clear', 'cloudy', 'wet', 'rain', 'fog', 'dust')
LIGHT = ('day', 'low_sun', 'twilight', 'night')
ROAD = ('straight', 'multi_lane', 'two_way', 'junction')
SPEED_BINS = (0.0, 20.0, 40.0, 60.0, 80.0, 130.0)
TARGET = 0.05
CONFIDENCE = 0.95
STATUSES = ('untested', 'too_few', 'enough', 'above_target')
LABELS = {'weather': {'clear': 'clear', 'cloudy': 'cloudy', 'wet': 'wet road', 'rain': 'rain', 'fog': 'fog',
                      'dust': 'dust storm'},
          'light': {'day': 'day', 'low_sun': 'low sun', 'twilight': 'twilight', 'night': 'night'},
          'road': {'straight': 'straight', 'multi_lane': 'multi-lane', 'two_way': 'two-way', 'junction': 'junction'}}
# the representative CARLA weather of a cell (a generated scenario's weather): each lands in its own class
CELL_WEATHER = {'clear': {'cloudiness': 5.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0, 'wetness': 0.0,
                          'fog_density': 0.0, 'dust_storm': 0.0},
                'cloudy': {'cloudiness': 70.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0, 'wetness': 0.0,
                           'fog_density': 0.0, 'dust_storm': 0.0},
                'wet': {'cloudiness': 40.0, 'precipitation': 0.0, 'precipitation_deposits': 50.0, 'wetness': 60.0,
                        'fog_density': 0.0, 'dust_storm': 0.0},
                'rain': {'cloudiness': 80.0, 'precipitation': 60.0, 'precipitation_deposits': 60.0, 'wetness': 70.0,
                         'fog_density': 0.0, 'dust_storm': 0.0},
                'fog': {'cloudiness': 60.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0, 'wetness': 0.0,
                        'fog_density': 50.0, 'fog_distance': 0.0, 'dust_storm': 0.0},
                'dust': {'cloudiness': 100.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0, 'wetness': 0.0,
                         'fog_density': 2.0, 'dust_storm': 100.0}}
CELL_SUN = {'day': 45.0, 'low_sun': 12.0, 'twilight': 0.0, 'night': -30.0}


# =============================================================================================== classification
def weather_class(w: dict | None) -> str:
    """A run's weather class from CARLA's fields (module doc: the thresholds)."""
    from . import weather as W
    d = W.resolve({k: v for k, v in (w or {}).items() if k in W.FIELDS or k == 'preset'})
    if d['precipitation'] >= 10:
        return 'rain'
    if d['fog_density'] >= 20 and d['fog_distance'] <= 50:
        return 'fog'
    if d.get('dust_storm', 0.0) >= 50:
        return 'dust'
    if d['wetness'] >= 40 or d['precipitation_deposits'] >= 40:
        return 'wet'
    if d['cloudiness'] >= 50:
        return 'cloudy'
    return 'clear'


def light_class(sun_altitude: float | None) -> str | None:
    if sun_altitude is None:
        return None
    s = float(sun_altitude)
    return 'night' if s < -6 else 'twilight' if s < 6 else 'low_sun' if s < 20 else 'day'


def road_class(road_types) -> str:
    rt = set(road_types or ())
    return next((r for r in reversed(ROAD) if r in rt), 'straight')


def speed_label(lo: float, hi: float) -> str:
    return f'{lo:g}-{hi:g}'


def speed_class(v: float | None, bins=SPEED_BINS) -> str | None:
    if v is None:
        return None
    for a, b in zip(bins, bins[1:]):
        if a <= v < b or (b == bins[-1] and v == b):
            return speed_label(a, b)
    return None


def cell_id(labels: dict) -> str:
    return '/'.join(str(labels[d]) for d in DIMENSIONS)


def parse_cell(cid: str) -> dict:
    parts = cid.split('/')
    if len(parts) != len(DIMENSIONS):
        raise ValueError(f'a cell is {"/".join(DIMENSIONS)}, not {cid!r}')
    return dict(zip(DIMENSIONS, parts))


def describe_cell(cid: str) -> str:
    c = parse_cell(cid)
    return (f'{LABELS["weather"].get(c["weather"], c["weather"])}, {LABELS["light"].get(c["light"], c["light"])}, '
            f'{LABELS["road"].get(c["road"], c["road"])} road, {c["actor"]}, ego {c["speed"].replace("-", "–")} km/h')


def run_cells(cfg: dict, bins=SPEED_BINS) -> tuple:
    """(the cell ids of a scenario run's config, why not when it has none)."""
    from .safetycase import run_coordinates
    if not (cfg or {}).get('scenario'):
        return [], 'no scenario'
    co = run_coordinates(cfg)
    lab = {'weather': weather_class(cfg.get('weather')), 'road': road_class(co.get('road_type')),
           'light': light_class(co.get('sun_altitude_deg')), 'speed': speed_class(co.get('ego_speed_kmh'), bins)}
    if lab['light'] is None:
        lab['light'] = light_class((cfg.get('weather') or {}).get('sun_altitude_angle', 45.0))
    if lab['speed'] is None:
        return [], 'its ego speed is unknown or outside the bands'
    actors = co.get('actors') or []
    if not actors:
        return [], 'the scenario has no actors'
    return [cell_id(dict(lab, actor=a)) for a in actors], ''


# =============================================================================================== the domain
def domain(odd: dict | None = None) -> dict:
    """The labels of each dimension: Falcon's, or restricted by a (safety-case) ODD (module doc)."""
    from . import scenarios as SC
    dims = {'weather': list(WEATHER), 'road': list(ROAD), 'actor': list(SC.KINDS), 'light': list(LIGHT)}
    bins = list(SPEED_BINS)
    src = 'every label Falcon records'
    if odd:
        d = odd.get('dimensions') or {}
        src = f'the ODD {odd.get("name") or ""}'.strip()
        if d.get('actors', {}).get('values'):
            dims['actor'] = [a for a in d['actors']['values'] if a in SC.KINDS] or dims['actor']
        if d.get('road_type', {}).get('values'):
            dims['road'] = [r for r in ROAD if r in d['road_type']['values']] or dims['road']
        sp = d.get('ego_speed_kmh')
        if sp and sp.get('kind') == 'range':
            e = [float(x) for x in sp.get('bins') or [sp['lo'], sp['hi']]]
            if len(e) >= 2:
                bins = e

        def inside(name, v):
            r = d.get(name)
            return r is None or r.get('kind') != 'range' or float(r['lo']) <= v <= float(r['hi'])
        dims['weather'] = [w for w in WEATHER if all(inside(k, v) for k, v in CELL_WEATHER[w].items()
                                                     if k in ('precipitation', 'fog_density', 'wetness',
                                                              'cloudiness', 'precipitation_deposits'))] \
            or dims['weather']
        dims['light'] = [x for x in LIGHT if inside('sun_altitude_deg', CELL_SUN[x])] or dims['light']
    dims['speed'] = [speed_label(a, b) for a, b in zip(bins, bins[1:])]
    total = 1
    for k in DIMENSIONS:
        total *= len(dims[k])
    return {'dims': {k: dims[k] for k in DIMENSIONS}, 'speed_bins': bins, 'cells': total, 'source': src}


def workspace_odd(workspace: str, project=None) -> dict | None:
    """The workspace's ODD as the safety case's format (odd.json, else the project's Loop ODD), or None."""
    from .oddmonitor import PROJECT_ODD, monitor_odd
    p = os.path.join(workspace, PROJECT_ODD) if workspace else ''
    raw = None
    if p and os.path.isfile(p):
        try:
            with open(p, encoding='utf-8') as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            raw = None
    elif project is not None and (project.doc.get('loop') or {}).get('odd'):
        raw = project.doc['loop']['odd']
    if not raw:
        return None
    try:
        return monitor_odd(raw, project)
    except Exception:                     # noqa: BLE001 (an invalid ODD: the full domain, said by the source)
        return None


# =============================================================================================== the runs
def _read(path: str):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _jsonl(path: str) -> list:
    out = []
    try:
        with open(path, encoding='utf-8') as fh:
            for x in fh:
                if x.strip():
                    try:
                        out.append(json.loads(x))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _cfg(run_dir: str) -> dict | None:
    for name in ('drive.json', 'stack.json'):
        d = _read(os.path.join(run_dir, name))
        if isinstance(d, dict) and d.get('scenario'):
            return d
    return None


def study_threshold(d: str) -> float:
    """A study's failure threshold: its failure-rate estimate's, else its searches' (as the safety case reads it),
    else 0.5."""
    fr = _read(os.path.join(d, 'failure-rate.json'))
    if isinstance(fr, dict) and (fr.get('options') or {}).get('threshold') is not None:
        return float(fr['options']['threshold'])
    thr = 0.5
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else ():
        if f.startswith('search-') and f.endswith('.json'):
            v = (_read(os.path.join(d, f)) or {}).get('threshold')
            thr = float(v) if v is not None else thr
    return thr


def study_kind(d: str) -> str:
    names = os.listdir(d) if os.path.isdir(d) else []
    if 'failure-rate.json' in names or 'failrate-state.json' in names:
        return 'estimate'
    if any(n.startswith(('search-', 'searcher-')) for n in names):
        return 'search'
    return 'sweep'


def collect(workspace: str, bins=SPEED_BINS) -> dict:
    """Every finished scenario run of the workspace with its cells and verdict (module doc: what counts)."""
    from .provenance import read_json
    samples, skipped, seen = [], {}, set()

    def add(run_dir, fail, source, origin, cfg=None):
        key = os.path.realpath(run_dir)
        if key in seen:
            return
        seen.add(key)
        cfg = cfg or _cfg(run_dir)
        if cfg is None:
            skipped['no scenario config'] = skipped.get('no scenario config', 0) + 1
            return
        cells, why = run_cells(cfg, bins)
        if not cells:
            skipped[why] = skipped.get(why, 0) + 1
            return
        samples.append({'run': run_dir, 'fail': bool(fail), 'source': source, 'origin': origin, 'cells': cells,
                        'scenario': (cfg.get('scenario') or {}).get('id')})
    sd = os.path.join(workspace, 'studies')
    for name in sorted(os.listdir(sd)) if os.path.isdir(sd) else ():
        d = os.path.join(sd, name)
        if not os.path.isfile(os.path.join(d, 'study.json')):
            continue
        thr, kind = study_threshold(d), study_kind(d)
        for r in _jsonl(os.path.join(d, 'results.jsonl')):
            if r.get('status') == 'ok' and r.get('run'):
                add(r['run'], float(r.get('objective', r.get('criticality', 0.0)) or 0.0) >= thr, kind,
                    f'study {name}')
    su = os.path.join(workspace, 'suites')
    for name in sorted(os.listdir(su)) if os.path.isdir(su) else ():
        rows = {}
        for r in _jsonl(os.path.join(su, name, 'results.jsonl')):
            if r.get('key'):
                rows[r['key']] = r
        for r in rows.values():
            if r.get('status') == 'ok' and r.get('passed') is not None and r.get('run'):
                add(r['run'], r['passed'] is False, 'suite', f'suite {name}')
    rd = os.path.join(workspace, 'runs')
    for name in sorted(os.listdir(rd)) if os.path.isdir(rd) else ():
        d = os.path.join(rd, name)
        run = _read(os.path.join(d, 'run.json'))
        if not isinstance(run, dict) or run.get('status') != 'ok' or not isinstance(run.get('scenario'), dict):
            continue
        if run['scenario'].get('passed') is None:
            continue
        try:
            cfg = read_json(os.path.join(d, 'drive.json'))
        except (OSError, ValueError):
            cfg = None
        add(d, run['scenario']['passed'] is False, 'run', 'runs')
    return {'samples': samples, 'skipped': skipped}


# =============================================================================================== the grid
def judge(n: int, k: int, target: float = TARGET, conf: float = CONFIDENCE) -> dict:
    """The rule (module doc) for n runs with k failures."""
    if n == 0:
        return {'status': 'untested', 'upper': None, 'lower': None, 'needed': FR.runs_needed(target, conf, 0)}
    up, lo = FR.binom_upper(k, n, conf), FR.binom_lower(k, n, conf)
    if lo > target:
        st = 'above_target'
        need = 0
    elif up <= target:
        st, need = 'enough', 0
    else:
        st = 'too_few'
        need = max(0, FR.runs_needed(target, conf, k) - n)
    return {'status': st, 'upper': up, 'lower': lo, 'needed': need}


def estimates(workspace: str) -> list:
    """The failure-rate estimates of the workspace's studies with their per-cell breakdown (core.failrate.by_cell),
    as [{'study', 'scenario', 'estimate', 'by_cell', 'rate'}]."""
    out = []
    sd = os.path.join(workspace, 'studies')
    for name in sorted(os.listdir(sd)) if os.path.isdir(sd) else ():
        fr = _read(os.path.join(sd, name, 'failure-rate.json'))
        if isinstance(fr, dict) and fr.get('schema') == FR.SCHEMA:
            out.append({'study': name, 'scenario': fr.get('scenario'), 'estimate': fr.get('estimate'),
                        'by_cell': fr.get('by_cell') or {}, 'rate': fr.get('rate'), 'judge': fr.get('judge'),
                        'done': fr.get('done')})
    return out


def coverage(workspace: str, project=None, target: float = TARGET, conf: float = CONFIDENCE,
             odd: dict | None = None) -> dict:
    """The grid (module doc): per tested cell its runs, failures, sources, status, bounds, runs still needed and the
    estimates there; the counts per status over the whole domain; the ODD it was restricted to."""
    if not 0 < target < 1 or not 0 < conf < 1:
        raise ValueError('target and confidence lie in (0, 1)')
    odd = odd if odd is not None else workspace_odd(workspace, project)
    dom = domain(odd)
    col = collect(workspace, dom['speed_bins'])
    allowed = {k: set(v) for k, v in dom['dims'].items()}
    cells, outside = {}, 0
    for i, s in enumerate(col['samples']):
        inside = False
        for c in s['cells']:
            lab = parse_cell(c)
            if all(lab[k] in allowed[k] for k in DIMENSIONS):
                inside = True
                e = cells.setdefault(c, {'runs': 0, 'failures': 0, 'sources': {}, 'samples': []})
                e['runs'] += 1
                e['failures'] += s['fail']
                e['sources'][s['source']] = e['sources'].get(s['source'], 0) + 1
                e['samples'].append(i)
        outside += not inside
    ests = estimates(workspace)
    for e in ests:
        for c, row in (e['by_cell'] or {}).items():
            if c in cells:
                cells[c].setdefault('estimates', []).append(dict(row, study=e['study']))
            elif all(parse_cell(c)[k] in allowed[k] for k in DIMENSIONS):
                cells[c] = {'runs': 0, 'failures': 0, 'sources': {}, 'samples': [],
                            'estimates': [dict(row, study=e['study'])]}
    counts = {k: 0 for k in STATUSES}
    for c, e in cells.items():
        e.update(judge(e['runs'], e['failures'], target, conf))
        e['labels'] = parse_cell(c)
        counts[e['status']] += 1
    counts['untested'] += dom['cells'] - len(cells)          # the cells no run reached
    return {'schema': SCHEMA, 'workspace': workspace, 'domain': dom, 'odd': (odd or {}).get('name'),
            'rule': {'target': target, 'confidence': conf, 'runs_if_no_failure': FR.runs_needed(target, conf, 0),
                     'about': (f'a cell has enough runs when its one-sided Clopper–Pearson {conf:.0%} upper bound on '
                               f'the failure rate is ≤ {target:g} per encounter ({FR.runs_needed(target, conf, 0)} '
                               'runs without a failure); the runs are treated as independent draws — a search\'s '
                               'are not (docs/odd-coverage.md)')},
            'cells': cells, 'counts': counts, 'runs': len(col['samples']), 'outside_domain': outside,
            'skipped': col['skipped'], 'samples': col['samples'], 'estimates': [
                {k: e[k] for k in ('study', 'scenario', 'estimate', 'rate', 'judge', 'done')} for e in ests]}


def pivot(cov: dict, rows: str = 'weather', cols: str = 'light', where: dict | None = None) -> dict:
    """Two dimensions of the grid, the others summed over (or fixed by where={dim: label}): per (row, column) the
    distinct runs and failures, the rule on them, and how many of its full cells are untested, too few, enough or above
    the target."""
    if rows not in DIMENSIONS or cols not in DIMENSIONS or rows == cols:
        raise ValueError(f'rows and cols: two different dimensions of {", ".join(DIMENSIONS)}')
    dom = cov['domain']['dims']
    where = {k: v for k, v in (where or {}).items() if v not in (None, '', 'all')}
    for k, v in where.items():
        if k not in DIMENSIONS or v not in dom[k]:
            raise ValueError(f'{k}={v!r} is not a label of the domain')
    free = [d for d in DIMENSIONS if d not in (rows, cols) and d not in where]
    per = 1
    for d in free:
        per *= len(dom[d])
    rule = cov['rule']
    out = []
    for r in dom[rows]:
        line = []
        for c in dom[cols]:
            runs, fails, st = set(), set(), {k: 0 for k in STATUSES}
            ests = []
            for cid, e in cov['cells'].items():
                lab = e['labels']
                if lab[rows] != r or lab[cols] != c or any(lab[k] != v for k, v in where.items()):
                    continue
                st[e['status']] += 1
                runs.update(e['samples'])
                fails.update(i for i in e['samples'] if cov['samples'][i]['fail'])
                ests += [dict(x, cell=cid) for x in e.get('estimates') or ()]
            st['untested'] += per - sum(st.values())
            j = judge(len(runs), len(fails), rule['target'], rule['confidence'])
            line.append(dict(j, runs=len(runs), failures=len(fails), cells=per, cell_status=st, estimates=ests))
        out.append(line)
    return {'rows': rows, 'cols': cols, 'row_labels': dom[rows], 'col_labels': dom[cols], 'where': where,
            'grid': out, 'cells_per_entry': per}


# =============================================================================================== filling the gaps
def _ego_speed(scn: dict):
    """(the parameter name that sets the ego's speed, its range) or (None, fixed value)."""
    v = (scn.get('ego') or {}).get('speed_kmh')
    if isinstance(v, str) and v.startswith('$') and v[1:].isidentifier():
        p = (scn.get('params') or {}).get(v[1:]) or {}
        if p.get('dist') == 'uniform':
            return v[1:], (float(p['lo']), float(p['hi']))
        if p.get('dist') == 'normal':
            return v[1:], (float(p.get('lo', p['mean'] - 3 * p['sd'])), float(p.get('hi', p['mean'] + 3 * p['sd'])))
        if p.get('dist') == 'fixed':
            return None, float(p['value'])
        return None, None
    if isinstance(v, (int, float)):
        return None, float(v)
    return None, None


def scenario_road(scn: dict) -> str:
    from .safetycase import run_coordinates
    return road_class(run_coordinates({'scenario': scn}).get('road_type'))


def _producers(scenarios: list) -> list:
    """Per scenario what it can produce: (scenario, actor kinds, road class, ego-speed parameter, its range or
    value)."""
    out = []
    for s in scenarios:
        if s.get('coverage_cell'):
            continue                                       # a generated one: start from the original
        name, rng = _ego_speed(s)
        if rng is None:
            continue
        out.append((s, {a.get('kind') for a in s.get('actors') or []}, scenario_road(s), name, rng))
    return out


def scenario_for_cell(cid: str, scenarios: list, producers: list | None = None) -> dict | None:
    """The first scenario (library order, then the project's) that can produce the cell: its actors include the
    cell's kind, its road requirement is the cell's road type, its ego speed can lie in the band."""
    lab = parse_cell(cid)
    lo, hi = (float(x) for x in lab['speed'].split('-'))
    for s, kinds, road, name, rng in producers if producers is not None else _producers(scenarios):
        if lab['actor'] not in kinds or road != lab['road']:
            continue
        if name is None and not lo <= rng < hi:
            continue
        if name is not None and (min(rng[1], hi) - max(rng[0], lo)) <= 0:
            continue
        return s
    return None


def cell_scenario(base: dict, cid: str) -> dict:
    """A copy of base made for the cell: the cell's weather and light, the ego-speed parameter narrowed to the band,
    a stable id from the base and the cell."""
    lab = parse_cell(cid)
    lo, hi = (float(x) for x in lab['speed'].split('-'))
    d = copy.deepcopy(base)
    h = hashlib.sha256(f'{base.get("id")}|{cid}'.encode()).hexdigest()[:8]
    d.update(id=f'cov-{base.get("id")}-{h}', name=f'{base.get("name")} · {describe_cell(cid)}', builtin=False,
             coverage_cell=cid, coverage_base=base.get('id'))
    d.setdefault('schema', 'falcon.scenario/1')
    d['about'] = (base.get('about') or '') + f' Generated for the ODD cell {cid} (core.oddcoverage).'
    d['weather'] = dict(CELL_WEATHER[lab['weather']], sun_altitude_angle=CELL_SUN[lab['light']])
    name, rng = _ego_speed(base)
    if name is not None:
        p = dict(d['params'][name])
        a, b = max(rng[0], lo), min(rng[1], hi)
        d['params'][name] = {'dist': 'uniform', 'lo': a, 'hi': b, 'unit': p.get('unit', 'km/h'),
                             'about': (p.get('about') or '') + f' (narrowed to the cell\'s band {lo:g}–{hi:g} km/h)'}
    return d


def fill_plan(cov: dict, scenarios: list, max_cells: int = 10, kind: str = 'search', budget: int = 0) -> dict:
    """Scenarios and studies for the cells that need runs (module doc). Order: too-few cells, the largest shortfall
    first; then untested cells, those one dimension away from a tested cell first (coverage grows outward from what
    has been run), then two, and so on. Each entry: the cell, its status and the runs it still needs, the base
    scenario, the generated scenario, the study's name and kind, and its runs (budget, or the runs still needed, at
    least 8 and at most 200). The domain's cells no scenario of the library or the project can produce are counted
    apart, by actor, road and speed."""
    import itertools
    if kind not in ('search', 'sweep'):
        raise ValueError('kind: search or sweep')
    dom = cov['domain']['dims']
    prod = _producers(scenarios)
    tested = [tuple(parse_cell(c)[d] for d in DIMENSIONS) for c, e in cov['cells'].items() if e['runs'] > 0]
    want = []
    for c, e in cov['cells'].items():
        if e['status'] == 'too_few':
            want.append((0, -e['needed'], c, e['status'], e['needed']))
    need0 = cov['rule']['runs_if_no_failure']
    impossible: dict = {}
    for combo in itertools.product(*(dom[d] for d in DIMENSIONS)):
        c = '/'.join(combo)
        if scenario_for_cell(c, scenarios, prod) is None:
            lab = dict(zip(DIMENSIONS, combo))
            key = f'{lab["actor"]} on a {LABELS["road"][lab["road"]]} road at {lab["speed"].replace("-", "–")} km/h'
            impossible[key] = impossible.get(key, 0) + 1
            continue
        if c not in cov['cells'] or cov['cells'][c]['runs'] == 0:
            near = min((sum(a != b for a, b in zip(combo, t)) for t in tested), default=len(DIMENSIONS))
            want.append((near, -need0, c, 'untested', need0))
    want.sort(key=lambda t: (t[0], t[1], t[2]))
    plan = []
    for _, _, c, status, needed in want[:int(max_cells)]:
        base = scenario_for_cell(c, scenarios, prod)
        scn = cell_scenario(base, c)
        runs = int(budget) if budget else max(8, min(200, int(needed)))
        plan.append({'cell': c, 'about': describe_cell(c), 'status': status, 'needed': int(needed),
                     'base': base.get('id'), 'scenario': scn, 'kind': kind, 'runs': runs,
                     'study': f'{"search" if kind == "search" else "sweep"}-{scn["id"]}'})
    return {'plan': plan, 'candidates': len(want),
            'no_scenario': [{'what': k, 'cells': n} for k, n in sorted(impossible.items())],
            'no_scenario_cells': sum(impossible.values()),
            'about': ('a search (bayes-pof: maps where it fails) per cell; its runs count in the cell but are not '
                      'random draws' if kind == 'search' else
                      'a sweep of random runs per cell: the independent draws the rule assumes')}


def apply_plan(project, plan: dict) -> list:
    """Add each plan entry's generated scenario to the project (an undoable insert; one already there is kept) and
    return the edits made."""
    from .project import Edit
    edits = []
    for e in plan['plan']:
        if project.item('scenarios', e['scenario']['id']) is None:
            edits.append(project.apply(Edit('insert', ['scenarios'], e['scenario'],
                                            text=f'Add the coverage scenario {e["scenario"]["name"]}')))
    return edits


def plan_argv(project_path: str, entry: dict, world: str, rig: str, seed: int = 0, threshold: float = 0.5,
              fake_carla: str = '') -> list:
    """`falcon search|sweep …` for a plan entry (the arguments after `python -m falcon_autonomy`)."""
    from ..sim.studio import search_argv, sweep_argv
    if entry['kind'] == 'sweep':
        return sweep_argv(project_path, entry['scenario']['id'], entry['study'], entry['runs'], seed, world, rig,
                          fake_carla)
    return search_argv(project_path, entry['scenario']['id'], entry['study'], 'bayes-pof', entry['runs'], 4,
                       threshold, seed, world, rig, fake_carla=fake_carla)
