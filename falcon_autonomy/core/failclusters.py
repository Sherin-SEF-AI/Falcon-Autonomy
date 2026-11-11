"""Failure clustering (numpy): group a workspace's failures into failure modes, name each from what sets it apart, and
say which failures of a new CI run fall in no known mode. `falcon clusters …`; docs/failure-clusters.md is the manual.

The failures (collect) — every one read from the files the other features wrote, each a plain dict (FAILURE_KEYS):
    search  a study's run (studies/<id>/results.jsonl: a sweep or a rare-case search) whose objective reached the
            study's failure threshold (its search-*.json, else 0.5)
    suite   a stack suite's finished run that failed its pass criteria (suites/<id>/results.jsonl)
    ci      a CI run's finished run that failed (ci/<name>/ci-runs.jsonl, core.ci)
with its scenario, parameter values, the conditions it ran in (from its run folder's run.json: world, fog density,
sun altitude, precipitation, time of day), its outcome (collision, minimum TTC, criticality, the failed checks, the RSS
label when the study recorded one) and, when a "why did it fail?" answer exists for its run (rootcause/*/
rootcause.json, core.rootcause), the factors whose removal fixed it ("causes") and those present.

The distance (distance_matrix) — Gower's general similarity coefficient (Gower 1971, A general coefficient of
similarity and some of its properties, Biometrics 27) over five feature groups, each group the mean of its features'
distances and the whole the weighted mean of the groups both failures have (GROUP_WEIGHTS, all 1):
    scenario     the scenario's id (0 the same, 1 not)
    parameters   each parameter value: a number |a − b| / its range over all failures; a choice 0 or 1
    conditions   world (0/1), time of day (0/1), fog, sun altitude, precipitation (range-normalised)
    outcome      collision (0/1), RSS label (0/1), failed checks (Jaccard distance of the sets), minimum TTC and
                 criticality (range-normalised)
    cause        the root-cause factors that fixed it (Jaccard), only when both have an attribution
A feature one of the two lacks is left out (Gower's rule for missing values); two failures with no group in common are
at distance 1. Every distance is in [0, 1].

The clusters (cluster) — agglomerative clustering with average linkage (UPGMA) on that matrix, computed by the
nearest-neighbour chain (O(n²) time and memory; exact for average linkage, which never inverts), cut where the
average distance between two groups exceeds CUT (0.30: failures in one mode differ, on average, in less than about a
third of what they could). Deterministic: ties go to the lower index; failures in time order. Each cluster gets an id
from its medoid (the member nearest on average to the others: 'C-' + 8 hex of its key), so a mode keeps its id while its
medoid stays, its size, its sources and scenarios, its cohesion (mean distance within) and its distinguishing features.

The names (describe) — from the features that set a cluster apart from the other failures, deterministically: a value
shared by ≥ 75 % of its members and ≥ 25 points more often than outside (a scenario, a world, a time of day, a
collision, an RSS label, a failed check, a cause), a number whose median inside is a quarter of its range or more away
from the median outside (shown as the cluster's 10th–90th percentile range, or ≥ / ≤ when it sits at one end). The name
is three of them: the scenario, the best of where and how it ran (a parameter, a condition, a cause), the best of how it
failed (the outcome), else the next best by difference; on equal standing a value goes before a number. The assistant's
fast model may reword a name when it is on; the wording is kept only when every number in it is one of the cluster's own
(assistant.suggest.phrase_clusters).

New failure modes (check_ci) — a CI run's failures against the clusters of every other failure of the workspace: one
whose average distance to the nearest cluster's members is at most CUT falls in that mode; the others are new, and
are clustered among themselves (each new cluster a new failure mode). Run at the end of every `falcon ci run` whose
project (or output folder WORKSPACE/ci/NAME) names a workspace; written to OUT/failure-modes.json and the summary.

The API for other code (the research autopilot): collect(ws, sources, exclude, keep) → failures; distance_matrix(
failures) → (n, n) array; cluster(failures, cut, D) → clusters (dicts, CLUSTER_KEYS) largest first; build(ws, …) → the
document (also saved to WORKSPACE/clusters/clusters.json under core.team.FileLock); load(ws); assign(new, reference,
cut) → per new failure its cluster or None; check_ci(ws, ci_dir).

Limits: a distance over hand-chosen features with equal group weights — two failures can be near for reasons that do
not matter and far for ones that do; parameter ranges are per name across scenarios (two scenarios' "speed" are
compared as one); the cut is one global number; time of day and weather come from the run's config, not from what the
sensors saw; at most MAX_FAILURES (the newest) are clustered.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

SCHEMA = 'falcon.failure_clusters/1'
MODES_SCHEMA = 'falcon.failure_modes/1'
SOURCES = ('search', 'suite', 'ci')
CUT = 0.30
MAX_FAILURES = 2000
GROUP_WEIGHTS = {'scenario': 1.0, 'parameters': 1.0, 'conditions': 1.0, 'outcome': 1.0, 'cause': 1.0}
SHARE_MIN, LIFT_MIN, NUM_SHIFT_MIN = 0.75, 0.25, 0.25
FAILURE_KEYS = ('key', 'source', 'origin', 'index', 'run', 'scenario', 'seed', 'values', 'conditions', 'outcome',
                'causes', 'present', 'time', 'commit')
CLUSTER_KEYS = ('id', 'name', 'size', 'members', 'medoid', 'features', 'sources', 'scenarios', 'cohesion')
TOD_BOUNDS = ((-6.0, 'night'), (6.0, 'twilight'), (20.0, 'low sun'))


# ================================================================================================ reading failures
def _json(p, default=None):
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _jsonl(p) -> list:
    out = []
    try:
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def tod_of(sun):
    if sun is None:
        return None
    for b, name in TOD_BOUNDS:
        if sun < b:
            return name
    return 'day'


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def run_conditions(run_dir: str, cache: dict | None = None) -> dict:
    """World, fog, sun altitude, precipitation and time of day from a run's run.json (its config in provenance)."""
    if not run_dir:
        return {}
    if cache is not None and run_dir in cache:
        return cache[run_dir]
    r = _json(os.path.join(run_dir, 'run.json'), {}) or {}
    st = (r.get('provenance') or {}).get('settings') or {}
    if not st:
        st = _json(os.path.join(run_dir, 'drive.json'), {}) or {}
    w = st.get('weather') or {}
    proj = st.get('project') or {}
    sun = _num(w.get('sun_altitude_angle'))
    out = {'world': proj.get('world') or st.get('town') or None, 'fog': _num(w.get('fog_density')),
           'sun': sun, 'rain': _num(w.get('precipitation')), 'tod': tod_of(sun)}
    out = {k: v for k, v in out.items() if v is not None}
    if cache is not None:
        cache[run_dir] = out
    return out


def _checks(checks) -> list:
    out = []
    for c in checks or ():
        if isinstance(c, dict):
            if not c.get('ok'):
                out.append(str(c.get('name') or '?'))
        elif isinstance(c, str):
            out.append(c.split(':', 1)[0].strip())
    return sorted(set(out))


def rootcause_index(ws: str) -> dict:
    """realpath of a case's run -> {'causes': factors whose removal fixed it, 'present': factors present, 'id'} from
    every counterfactual answer in rootcause/ (the newest per run)."""
    root = os.path.join(ws, 'rootcause')
    out = {}
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for n in names:
        d = _json(os.path.join(root, n, 'rootcause.json'))
        if not isinstance(d, dict) or d.get('kind') != 'counterfactual' or d.get('status') not in ('attributed',
                                                                                                  'no_factor'):
            continue
        run = (d.get('case') or {}).get('run')
        if not run:
            continue
        fs = d.get('factors') or []
        out[os.path.realpath(run)] = {'causes': sorted(f['factor'] for f in fs if f.get('removing_fixes')),
                                      'present': sorted(f['factor'] for f in fs if f.get('present')), 'id': n}
    return out


def _attach_cause(f: dict, rc: dict):
    if f.get('run'):
        a = rc.get(os.path.realpath(f['run']))
        if a:
            f['causes'], f['present'], f['rootcause'] = a['causes'], a['present'], a['id']


def _study_failures(ws: str, d: str, rc: dict, cc: dict) -> list:
    sid = os.path.basename(d)
    st = _json(os.path.join(d, 'study.json'), {}) or {}
    thr = 0.5
    for f in sorted(os.listdir(d)):
        if f.startswith('search-') and f.endswith('.json'):
            thr = float((_json(os.path.join(d, f), {}) or {}).get('threshold') or thr)
    scn = st.get('scenario')
    scn = scn.get('id') if isinstance(scn, dict) else scn
    out = []
    for i, r in enumerate(_jsonl(os.path.join(d, 'results.jsonl'))):
        obj = _num(r.get('objective', r.get('criticality')))
        if r.get('status') not in (None, 'ok') or obj is None or obj < thr:
            continue
        sc = r.get('scenario') or {}
        crit = sc.get('criticality') or {}
        ov = crit.get('overall') or crit
        f = {'key': f'search:{sid}#{i}', 'source': 'search', 'origin': sid, 'index': i, 'run': r.get('run'),
             'scenario': sc.get('id') or scn, 'seed': sc.get('seed'), 'values': dict(r.get('values') or {}),
             'conditions': run_conditions(r.get('run'), cc),
             'outcome': {'collision': bool(ov.get('collision')) if ov else None,
                         'min_ttc_s': _num(ov.get('min_ttc_s')), 'criticality': _num(r.get('criticality', obj)),
                         'checks': _checks(sc.get('checks')), 'rss': (r.get('rss') or {}).get('label')},
             'time': _mtime(r.get('run')), 'commit': None}
        _attach_cause(f, rc)
        out.append(f)
    return out


def _mtime(run_dir) -> str:
    r = _json(os.path.join(run_dir, 'run.json'), {}) if run_dir else {}
    return (r or {}).get('started') or ''


def _suite_failures(ws: str, d: str, rc: dict, cc: dict) -> list:
    from . import stack as ST
    sid = os.path.basename(d)
    out = []
    try:
        rows = ST.read_results(d)
    except (OSError, ValueError, KeyError):
        return out
    for r in rows:
        if r.get('status') != 'ok' or r.get('passed') is not False:
            continue
        f = {'key': f'suite:{sid}:{r.get("key")}', 'source': 'suite', 'origin': sid, 'index': None, 'run': r.get('run'),
             'scenario': r.get('scenario'), 'seed': r.get('seed'), 'values': dict(r.get('values') or {}),
             'conditions': run_conditions(r.get('run'), cc),
             'outcome': {'collision': bool(r.get('collision')), 'min_ttc_s': _num(r.get('min_ttc_s')),
                         'criticality': _num(r.get('criticality')), 'checks': _checks(r.get('checks')), 'rss': None},
             'time': _mtime(r.get('run')), 'commit': None}
        _attach_cause(f, rc)
        out.append(f)
    return out


def ci_failures(ci_dir: str, records: list | None = None, rc: dict | None = None, cc: dict | None = None) -> list:
    """The failed finished runs of a CI output folder (or of its records as run_ci holds them)."""
    name = os.path.basename(os.path.normpath(ci_dir))
    res = _json(os.path.join(ci_dir, 'ci-results.json'), {}) or {}
    commit = (res.get('commit') or {}).get('sha')
    recs = records if records is not None else _jsonl(os.path.join(ci_dir, 'ci-runs.jsonl'))
    out = []
    for r in recs:
        if r.get('status') != 'ok' or r.get('passed') is not False:
            continue
        m = r.get('metrics') or {}
        f = {'key': f'ci:{name}:{r.get("id") or r.get("pair")}', 'source': 'ci', 'origin': name, 'index': None,
             'run': r.get('dir'), 'scenario': r.get('group'), 'seed': r.get('seed'),
             'values': dict(r.get('values') or {}), 'conditions': run_conditions(r.get('dir'), cc),
             'outcome': {'collision': m.get('collision'), 'min_ttc_s': _num(m.get('min_ttc_s')),
                         'criticality': _num(m.get('criticality')), 'checks': _checks(r.get('failed_checks')),
                         'rss': None},
             'time': _mtime(r.get('dir')) or res.get('started') or '', 'commit': commit}
        if rc:
            _attach_cause(f, rc)
        out.append(f)
    return out


def _folders(root: str, must: str) -> list:
    try:
        return [os.path.join(root, n) for n in sorted(os.listdir(root)) if not n.startswith('.')
                and os.path.isfile(os.path.join(root, n, must))]
    except OSError:
        return []


def collect(ws: str, sources=SOURCES, exclude=(), keep=None) -> list:
    """Every failure of the workspace from the sources (module doc), oldest first, at most MAX_FAILURES (the newest).
    exclude: origin folders (study, suite or CI names, or their paths) left out; keep(failure) -> bool filters (the
    assistant leaves out private runs)."""
    bad = {os.path.basename(os.path.normpath(e)) for e in exclude}
    rc, cc = rootcause_index(ws), {}
    out = []
    if 'search' in sources:
        for d in _folders(os.path.join(ws, 'studies'), 'results.jsonl'):
            if os.path.basename(d) not in bad:
                out += _study_failures(ws, d, rc, cc)
    if 'suite' in sources:
        for d in _folders(os.path.join(ws, 'suites'), 'results.jsonl'):
            if os.path.basename(d) not in bad:
                out += _suite_failures(ws, d, rc, cc)
    if 'ci' in sources:
        for d in _folders(os.path.join(ws, 'ci'), 'ci-runs.jsonl'):
            if os.path.basename(d) not in bad:
                out += ci_failures(d, None, rc, cc)
    if keep is not None:
        out = [f for f in out if keep(f)]
    out.sort(key=lambda f: (f.get('time') or '', f['key']))
    return out[-MAX_FAILURES:]


# ================================================================================================ features, distance
def feature_rows(f: dict) -> dict:
    """A failure's features: name -> (group, kind, value) with kind num, cat or set."""
    out = {'scenario': ('scenario', 'cat', f.get('scenario'))}
    for k, v in (f.get('values') or {}).items():
        x = _num(v)
        out[f'p.{k}'] = ('parameters', 'num', x) if x is not None else ('parameters', 'cat', str(v))
    c = f.get('conditions') or {}
    for k in ('world', 'tod'):
        if c.get(k) is not None:
            out[k] = ('conditions', 'cat', c[k])
    for k in ('fog', 'sun', 'rain'):
        if c.get(k) is not None:
            out[k] = ('conditions', 'num', c[k])
    o = f.get('outcome') or {}
    if o.get('collision') is not None:
        out['collision'] = ('outcome', 'cat', bool(o['collision']))
    if o.get('rss'):
        out['rss'] = ('outcome', 'cat', o['rss'])
    out['checks'] = ('outcome', 'set', tuple(o.get('checks') or ()))
    for k in ('min_ttc_s', 'criticality'):
        if o.get(k) is not None:
            out[k] = ('outcome', 'num', o[k])
    if f.get('causes') is not None:
        out['causes'] = ('cause', 'set', tuple(f['causes']))
    return {k: v for k, v in out.items() if v[2] is not None}


def distance_matrix(failures: list, weights: dict | None = None) -> np.ndarray:
    """Gower distances between failures (module doc): an (n, n) float array, zeros on the diagonal."""
    w = dict(GROUP_WEIGHTS, **(weights or {}))
    n = len(failures)
    rows = [feature_rows(f) for f in failures]
    names = sorted({k for r in rows for k in r})
    gsum = {g: np.zeros((n, n)) for g in w}
    gcnt = {g: np.zeros((n, n)) for g in w}
    for name in names:
        group = next(r[name][0] for r in rows if name in r)
        kind = next(r[name][1] for r in rows if name in r)
        have = np.array([name in r and r[name][1] == kind for r in rows])
        if have.sum() < 1:
            continue
        mask = np.outer(have, have)
        if kind == 'num':
            x = np.array([float(r[name][2]) if have[i] else 0.0 for i, r in enumerate(rows)])
            rng = float(x[have].max() - x[have].min()) if have.any() else 0.0
            d = np.abs(x[:, None] - x[None, :]) / rng if rng > 0 else np.zeros((n, n))
        elif kind == 'cat':
            vals = [r[name][2] if have[i] else None for i, r in enumerate(rows)]
            codes = {v: j for j, v in enumerate(dict.fromkeys(json.dumps(v, default=str) for v in vals))}
            x = np.array([codes[json.dumps(v, default=str)] for v in vals])
            d = (x[:, None] != x[None, :]).astype(float)
        else:
            sets = [frozenset(r[name][2]) if have[i] else frozenset() for i, r in enumerate(rows)]
            universe = sorted(set().union(*sets)) if sets else []
            if universe:
                M = np.array([[e in s for e in universe] for s in sets], dtype=float)
                inter = M @ M.T
                union = M.sum(1)[:, None] + M.sum(1)[None, :] - inter
                d = np.where(union > 0, 1.0 - inter / np.where(union > 0, union, 1.0), 0.0)
            else:
                d = np.zeros((n, n))
        gsum[group] += np.where(mask, d, 0.0)
        gcnt[group] += mask
    num = np.zeros((n, n))
    den = np.zeros((n, n))
    for g, wt in w.items():
        have = gcnt[g] > 0
        num += np.where(have, wt * gsum[g] / np.where(have, gcnt[g], 1.0), 0.0)
        den += np.where(have, wt, 0.0)
    D = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 1.0)


# ================================================================================================ clustering
def linkage(D: np.ndarray) -> list:
    """Average-linkage merges [(a, b, height, size)] by the nearest-neighbour chain (module doc); a and b are the
    lowest member indices of the two groups merged (the merged group keeps a)."""
    n = D.shape[0]
    if n < 2:
        return []
    M = D.astype(float).copy()
    np.fill_diagonal(M, np.inf)
    active = np.ones(n, bool)
    size = np.ones(n)
    merges, chain = [], []
    while active.sum() > 1:
        if not chain:
            chain.append(int(np.flatnonzero(active)[0]))
        a = chain[-1]
        row = np.where(active, M[a], np.inf)
        row[a] = np.inf
        b = int(np.argmin(row))
        if len(chain) > 1 and row[chain[-2]] <= row[b]:
            b = chain[-2]
        if len(chain) > 1 and b == chain[-2]:
            chain.pop()
            chain.pop()
            h = float(M[a, b])
            lo, hi = min(a, b), max(a, b)
            new = (size[lo] * M[lo] + size[hi] * M[hi]) / (size[lo] + size[hi])
            M[lo], M[:, lo] = new, new
            M[lo, lo] = np.inf
            M[hi, :], M[:, hi] = np.inf, np.inf
            active[hi] = False
            size[lo] += size[hi]
            merges.append((lo, hi, h, int(size[lo])))
            chain = [c for c in chain if active[c]]
        else:
            chain.append(b)
    return merges


def labels_at(n: int, merges: list, cut: float) -> np.ndarray:
    """A label per item: the groups formed by the merges at height ≤ cut (label = the group's lowest index)."""
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, h, _s in merges:
        if h <= cut + 1e-12:
            ra, rb = root(a), root(b)
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([root(i) for i in range(n)])


def cluster(failures: list, cut: float = CUT, D: np.ndarray | None = None) -> list:
    """The clusters of failures (module doc), largest first (ties: the earlier medoid), each with CLUSTER_KEYS."""
    n = len(failures)
    if n == 0:
        return []
    D = distance_matrix(failures) if D is None else D
    lab = labels_at(n, linkage(D), cut)
    groups: dict = {}
    for i, g in enumerate(lab):
        groups.setdefault(int(g), []).append(i)
    out = []
    for idx in groups.values():
        sub = D[np.ix_(idx, idx)]
        med = idx[int(np.argmin(sub.mean(1)))] if len(idx) > 1 else idx[0]
        out.append(_cluster_doc(failures, idx, med, float(sub[np.triu_indices(len(idx), 1)].mean())
                                if len(idx) > 1 else 0.0))
    out.sort(key=lambda c: (-c['size'], c['_order']))
    rest_all = list(range(n))
    for c in out:
        inside = set(c.pop('_idx'))
        c.pop('_order')
        c['features'] = describe(failures, sorted(inside), [i for i in rest_all if i not in inside])
        c['name'] = name_of(c['features'], c['size'])
    return out


def _cluster_doc(failures, idx, med, cohesion) -> dict:
    keys = [failures[i]['key'] for i in idx]
    src, scn = {}, {}
    for i in idx:
        src[failures[i]['source']] = src.get(failures[i]['source'], 0) + 1
        s = failures[i].get('scenario') or '?'
        scn[s] = scn.get(s, 0) + 1
    return {'id': 'C-' + hashlib.sha1(failures[med]['key'].encode()).hexdigest()[:8], 'name': '', 'size': len(idx),
            'members': keys, 'medoid': failures[med]['key'], 'features': [], 'sources': src, 'scenarios': scn,
            'cohesion': round(cohesion, 4), '_idx': idx, '_order': med}


# ================================================================================================ names
def _fmt(x: float) -> str:
    if abs(x) >= 10:
        return str(int(round(x)))
    return f'{x:.2f}'.rstrip('0').rstrip('.') or '0'


UNITS = {'fog': '', 'sun': '°', 'rain': '', 'min_ttc_s': ' s', 'criticality': ''}
TITLES = {'fog': 'fog', 'sun': 'sun', 'rain': 'rain', 'min_ttc_s': 'min TTC', 'criticality': 'criticality'}


def _phrase(name: str, kind: str, v) -> str:
    if name == 'scenario':
        return str(v)
    if name == 'world':
        return f'in {v}'
    if name == 'tod':
        return str(v)
    if name == 'collision':
        return 'collision' if v else 'no collision'
    if name == 'rss':
        return f'RSS {v}'
    if name.startswith('check:'):
        return f'fails "{name[6:]}"'
    if name.startswith('cause:'):
        return f'fixed by removing {name[6:]}'
    if name.startswith('p.'):
        return f'{name[2:]} = {v}'
    return f'{name} = {v}'


def describe(failures: list, inside: list, outside: list) -> list:
    """The distinguishing features of a group of failures against the rest (module doc), best first: each {'feature',
    'kind', 'text', 'score', …the numbers it rests on}."""
    rows = [feature_rows(f) for f in failures]
    feats = []
    names = sorted({k for i in inside for k in rows[i]})
    for name in names:
        group, kind = next(rows[i][name][:2] for i in inside if name in rows[i])
        if kind == 'num':
            xin = np.array([float(rows[i][name][2]) for i in inside if name in rows[i] and rows[i][name][1] == 'num'])
            xout = np.array([float(rows[i][name][2]) for i in outside if name in rows[i] and
                             rows[i][name][1] == 'num'])
            allx = np.concatenate([xin, xout])
            if len(xin) < max(1, int(np.ceil(SHARE_MIN * len(inside)))) or allx.size < 2:
                continue
            rng = float(allx.max() - allx.min())
            if rng <= 0:
                continue
            med_in = float(np.median(xin))
            med_out = float(np.median(xout)) if xout.size else float(np.median(allx))
            score = abs(med_in - med_out) / rng
            if xout.size and score < NUM_SHIFT_MIN:
                continue
            lo, hi = (float(np.percentile(xin, 10)), float(np.percentile(xin, 90))) if xin.size >= 5 else \
                (float(xin.min()), float(xin.max()))
            title = TITLES.get(name, name[2:] if name.startswith('p.') else name)
            unit = UNITS.get(name, '')
            if xout.size and lo > med_out:
                text = f'{title} ≥ {_fmt(lo)}{unit}'
            elif xout.size and hi < med_out:
                text = f'{title} ≤ {_fmt(hi)}{unit}'
            elif _fmt(lo) == _fmt(hi):
                text = f'{title} {_fmt(lo)}{unit}'
            else:
                text = f'{title} {_fmt(lo)}–{_fmt(hi)}{unit}'
            feats.append({'feature': name, 'group': group, 'kind': 'num', 'text': text,
                          'score': round(score if xout.size else 0.5, 4),
                          'low': round(lo, 4), 'high': round(hi, 4), 'median_in': round(med_in, 4),
                          'median_out': round(med_out, 4) if xout.size else None})
            continue
        values = []
        if kind == 'set':
            elems = sorted({e for i in inside if name in rows[i] for e in rows[i][name][2]})
            pref = 'check:' if name == 'checks' else 'cause:'
            for e in elems:
                values.append((pref + e, lambda i, e=e: name in rows[i] and e in rows[i][name][2], True))
        else:
            vals = sorted({json.dumps(rows[i][name][2], default=str) for i in inside if name in rows[i]})
            for vj in vals:
                v = json.loads(vj)
                values.append((name, lambda i, vj=vj: name in rows[i] and json.dumps(rows[i][name][2],
                                                                                     default=str) == vj, v))
        for fname, test, v in values:
            s_in = sum(1 for i in inside if test(i)) / len(inside)
            s_out = (sum(1 for i in outside if test(i)) / len(outside)) if outside else 0.0
            if s_in < SHARE_MIN or (outside and s_in - s_out < LIFT_MIN):
                continue
            feats.append({'feature': fname, 'group': group, 'kind': kind, 'value': v, 'text': _phrase(fname, kind, v),
                          'share_in': round(s_in, 3), 'share_out': round(s_out, 3) if outside else None,
                          'score': round(s_in - s_out, 4)})
    feats.sort(key=lambda d: (d['feature'] != 'scenario', -d['score'], d['kind'] == 'num', d['feature'], d['text']))
    return feats


def name_of(features: list, size: int) -> str:
    """The deterministic name: three distinguishing features — the scenario, the best of where and how it ran
    (parameters, conditions, cause), the best of how it failed (outcome), then the next best — else 'mixed
    failures'."""
    pick, rest = [], list(features)
    for want in (('scenario',), ('parameters', 'conditions', 'cause'), ('outcome',)):
        f = next((x for x in rest if x.get('group') in want), None)
        if f is not None:
            pick.append(f)
            rest.remove(f)
    pick += rest[:max(0, 3 - len(pick))]
    parts = [f['text'] for f in pick[:3]]
    return ', '.join(parts) if parts else f'mixed failures ({size})'


# ================================================================================================ the workspace
def clusters_path(ws: str) -> str:
    return os.path.join(ws, 'clusters', 'clusters.json')


def build(ws: str, sources=SOURCES, cut: float = CUT, keep=None, save: bool = True, ident: dict | None = None) -> dict:
    """Collect, cluster and (save) the workspace's failure modes: the document of WORKSPACE/clusters/clusters.json."""
    from .provenance import utc_now
    fails = collect(ws, sources, keep=keep)
    D = distance_matrix(fails) if fails else np.zeros((0, 0))
    cl = cluster(fails, cut, D)
    doc = {'schema': SCHEMA, 'built': utc_now(), 'workspace': ws, 'sources': list(sources), 'cut': cut,
           'method': {'distance': 'Gower (1971), group-weighted', 'groups': GROUP_WEIGHTS,
                      'linkage': 'average (UPGMA), nearest-neighbour chain', 'cut': cut,
                      'names': f'features shared by ≥ {SHARE_MIN:g} of a cluster and ≥ {LIFT_MIN:g} more than outside; '
                               f'numbers whose median moved ≥ {NUM_SHIFT_MIN:g} of their range'},
           'failures': len(fails), 'by_source': {s: sum(1 for f in fails if f['source'] == s) for s in sources},
           'clusters': cl, 'singletons': sum(1 for c in cl if c['size'] == 1),
           'index': {f['key']: _slim(f) for f in fails}}
    if save:
        from .paths import atomic_write
        from .team import FileLock, identity
        p = clusters_path(ws)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with FileLock(p + '.lock', ident or identity()):
            atomic_write(p, json.dumps(doc, indent=1, default=str))
    return doc


def _slim(f: dict) -> dict:
    return {k: f.get(k) for k in ('source', 'origin', 'index', 'run', 'scenario', 'seed', 'values', 'outcome',
                                  'conditions', 'causes', 'commit')}


def load(ws: str) -> dict | None:
    d = _json(clusters_path(ws))
    return d if isinstance(d, dict) and d.get('schema') == SCHEMA else None


def assign(new: list, reference: list, cut: float = CUT, clusters: list | None = None) -> list:
    """For each new failure: the reference cluster it falls in (average distance to its members ≤ cut) or None, and
    that distance. clusters: the reference's clusters (computed when not given)."""
    if not new:
        return []
    allf = list(reference) + list(new)
    D = distance_matrix(allf)
    nr = len(reference)
    cl = clusters if clusters is not None else cluster(reference, cut, D[:nr, :nr])
    pos = {f['key']: i for i, f in enumerate(reference)}
    out = []
    for j in range(len(new)):
        best = (None, 1.0)
        for c in cl:
            idx = [pos[k] for k in c['members'] if k in pos]
            if not idx:
                continue
            d = float(D[nr + j, idx].mean())
            if d < best[1] - 1e-12 or best[0] is None and d <= best[1]:
                best = (c, d)
        c, d = best
        out.append({'key': new[j]['key'], 'cluster': c['id'] if c is not None and d <= cut else None,
                    'nearest': c['id'] if c is not None else None, 'nearest_name': c['name'] if c is not None else None,
                    'distance': round(d, 4)})
    return out


def check_ci(ws: str, ci_dir: str, records: list | None = None, cut: float = CUT, write: bool = True,
             keep=None) -> dict:
    """A CI run's failures against the workspace's other failures (module doc); written to CI_DIR/failure-modes.json."""
    from .provenance import utc_now, write_json
    rc = rootcause_index(ws) if ws else {}
    new = ci_failures(ci_dir, records, rc, {})
    if keep is not None:
        new = [f for f in new if keep(f)]
    ref = collect(ws, SOURCES, exclude=(ci_dir,), keep=keep) if ws else []
    doc = {'schema': MODES_SCHEMA, 'checked': utc_now(), 'ci': os.path.basename(os.path.normpath(ci_dir)),
           'workspace': ws, 'cut': cut, 'failures': len(new), 'reference': {'failures': len(ref)}, 'known': [],
           'new': [], 'new_modes': []}
    if new:
        D = distance_matrix(ref + new)
        nr = len(ref)
        cl = cluster(ref, cut, D[:nr, :nr]) if ref else []
        doc['reference']['clusters'] = len(cl)
        res = assign(new, ref, cut, cl)
        for f, a in zip(new, res):
            (doc['known'] if a['cluster'] else doc['new']).append(dict(a, scenario=f.get('scenario'),
                                                                        seed=f.get('seed')))
        fresh = [j for j, a in enumerate(res) if not a['cluster']]
        if fresh:
            idx = [nr + j for j in fresh]
            modes = cluster([new[j] for j in fresh], cut, D[np.ix_(idx, idx)])
            # a new mode is named against the reference too: what sets it apart from everything known
            allf = ref + new
            at = {f['key']: nr + j for j, f in enumerate(new)}
            for m in modes:
                inside = sorted(at[k] for k in m['members'])
                m['features'] = describe(allf, inside, [i for i in range(len(allf)) if i not in set(inside)])
                m['name'] = name_of(m['features'], m['size'])
            doc['new_modes'] = modes
    if write:
        write_json(os.path.join(ci_dir, 'failure-modes.json'), doc)
    return doc


def ci_workspace(cfg: dict, out: str) -> str:
    """The workspace a CI run belongs to: its config's project's, else WORKSPACE when OUT is WORKSPACE/ci/NAME."""
    from .paths import workspace_for
    if cfg.get('project') and os.path.isfile(cfg['project']):
        return workspace_for(cfg['project'])
    parent = os.path.dirname(os.path.normpath(out))
    if os.path.basename(parent) == 'ci':
        return os.path.dirname(parent)
    return ''


# ================================================================================================ the CLI
def main_clusters(argv) -> int:
    """`falcon clusters build|show|check …` (docs/failure-clusters.md). 0 ok, 2 failed."""
    import argparse
    import sys
    from .paths import PROJECT_SUFFIX, workspace_for
    ap = argparse.ArgumentParser(prog='falcon clusters', description='failure modes: failures clustered and named '
                                                                     '(docs/failure-clusters.md)')
    sub = ap.add_subparsers(dest='what', required=True)
    b = sub.add_parser('build', help='cluster the workspace\'s failures and save them')
    b.add_argument('project', help='the project file or its workspace folder')
    b.add_argument('--sources', default=','.join(SOURCES), help='search,suite,ci (default all)')
    b.add_argument('--cut', type=float, default=CUT)
    b.add_argument('--json', action='store_true')
    s = sub.add_parser('show', help='the saved clusters (or one, with its members)')
    s.add_argument('project')
    s.add_argument('cluster', nargs='?', default='')
    s.add_argument('--json', action='store_true')
    c = sub.add_parser('check', help='a CI run\'s failures: in a known failure mode, or new')
    c.add_argument('project')
    c.add_argument('ci', help='the CI output folder')
    c.add_argument('--cut', type=float, default=CUT)
    c.add_argument('--json', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    p = os.path.abspath(a.project)
    ws = workspace_for(p) if p.endswith(PROJECT_SUFFIX) or os.path.isfile(p) else p
    try:
        if a.what == 'build':
            srcs = tuple(x for x in a.sources.split(',') if x)
            if any(x not in SOURCES for x in srcs) or not 0 < a.cut <= 1:
                print('failed: --sources from search,suite,ci; --cut in (0, 1]', file=sys.stderr)
                return 2
            doc = build(ws, srcs, a.cut)
            if a.json:
                print(json.dumps({k: v for k, v in doc.items() if k != 'index'}, indent=2, default=str))
                return 0
            print(f'{doc["failures"]} failures in {len(doc["clusters"])} clusters ({doc["singletons"]} single) at '
                  f'cut {a.cut:g}; saved {clusters_path(ws)}')
            for cl in doc['clusters']:
                print(f'  {cl["id"]}  {cl["size"]:4d}  {cl["name"]}')
            return 0
        if a.what == 'show':
            doc = load(ws)
            if doc is None:
                print('no clusters yet: falcon clusters build PROJECT', file=sys.stderr)
                return 2
            if a.cluster:
                cl = next((x for x in doc['clusters'] if x['id'] == a.cluster), None)
                if cl is None:
                    print(f'failed: no cluster {a.cluster}', file=sys.stderr)
                    return 2
                if a.json:
                    print(json.dumps(cl, indent=2, default=str))
                    return 0
                print(f'{cl["id"]}: {cl["name"]} — {cl["size"]} failures, cohesion {cl["cohesion"]}')
                for f in cl['features']:
                    print(f'  {f["text"]}  (score {f["score"]})')
                for k in cl['members']:
                    print(f'  - {k}')
                return 0
            if a.json:
                print(json.dumps({k: v for k, v in doc.items() if k != 'index'}, indent=2, default=str))
                return 0
            print(f'built {doc["built"]}: {doc["failures"]} failures, {len(doc["clusters"])} clusters')
            for cl in doc['clusters']:
                print(f'  {cl["id"]}  {cl["size"]:4d}  {cl["name"]}')
            return 0
        doc = check_ci(ws, os.path.abspath(a.ci), cut=a.cut)
        if a.json:
            print(json.dumps(doc, indent=2, default=str))
            return 0
        print(f'{doc["failures"]} failures in {doc["ci"]}: {len(doc["known"])} in known failure modes, '
              f'{len(doc["new"])} new ({len(doc["new_modes"])} new modes)')
        for m in doc['new_modes']:
            print(f'  NEW  {m["size"]:3d}  {m["name"]}')
        return 0
    except (OSError, ValueError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
