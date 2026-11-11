"""Falsification of signal-temporal-logic requirements: the robustness of a run as a rare-case search's objective, and
the commands `falcon falsify` and `falcon stl` (numpy at import; docs/falsification.md is the manual).

A requirement is a formula of core.stl over the signals core.rules computes from a recorded run (core.rules.SIGNALS,
the surrogate-safety ones included: ttc_min_s, drac_mps2, pet_s, …), or one or more rules of a rule set ("SET:ID" or
"SET:ID,ID": india, generic or a rule file). A FALSIFIER searches a scenario's parameters for a run that violates it:

    ρ          the run's robustness (core.rules.robustness): a formula's at the run's first sample (write G … for
               "at every sample"), the rules' lowest over the known samples, as `falcon rules check` reports them;
               ρ < 0 violated (falsified), ρ ≥ 0 satisfied, |ρ| in the predicate's units; NaN unknown
    value      the search's objective, σ(−ρ/scale) = 1 / (1 + exp(ρ/scale)), clamped to the exact verdict's side of
               0.5: ρ < 0 ⇔ value > 0.5 and ρ ≥ 0 ⇔ value < 0.5 (ρ = 0, satisfied in STL, maps just below 0.5). So the
               failure threshold 0.5 that core.safetycase, core.oddcoverage, core.failrate and the Rare cases page use
               reads "falsified", and higher = worse as for every objective. `scale` (default 1, the predicate's units)
               only shapes the landscape a model-based searcher (bayes) fits; rank-based ones (cmaes, cem) see the
               same order whatever it is
    smoothing  optional β > 0: the value is computed from core.stl.evaluate_smooth (soft minimum and maximum, −1/β·log
               Σ exp(−β·x)) — an APPROXIMATION that removes the kinks of min/max for the searcher; still clamped to
               the exact ρ's side of 0.5, and the exact ρ is what every row, verdict and report states

Every scored run keeps ρ (exact), ρ smoothed when asked, the value, the verdict, the time that decides ρ (core.stl.
witness: the sample of the predicate that sets it) and the share of the run where ρ was known; its per-sample series
are written beside the study (stl/<run>-<key>.json) for the traces the Falsify tab draws. ρ is written as a number,
'inf' or '-inf' (JSON has no infinity: a requirement nothing can violate, or a collision under a DRAC bound), or None
when unknown. A run whose ρ is unknown (the formula's signals were not recorded) is not a failure and teaches the
searcher nothing (it is told None).

A study for a requirement is an ordinary scenario study (sim.studio.Study, objective "robustness": core.milsearch); a
study's key hashes the formula's normal form (or the rule set's SHA-256 and the rule ids), the scale, β and
core.rules.SIGNALS_VERSION, so another formula never reuses another's scores (its runs are re-scored, not re-driven).

    falcon falsify run PROJECT --scenario S (--stl TEXT | --rule SET:ID) [--method cmaes|cem|bayes|…] [--budget N] …
                                           [--on local|cluster|cluster:POOL]  the drives on the cluster's nodes (the
                                           study here); a dashcam-derived scenario or world is refused off this machine
    falcon falsify show STUDY_DIR          falsify.json: best ρ, falsified runs (ρ < 0), near misses (0 ≤ ρ < margin)
    falcon falsify check                   the planted-landscape comparison of robustness against a binary objective
    falcon stl compile TEXT|--rule SET:ID [--save NAME --project P]    parse, normal form, horizon (formulas.json)
    falcon stl robustness RUN… (--stl TEXT | --rule SET:ID | --name NAME --project P) [--series] [--json]

Saved requirements: <workspace>/safety/formulas.json (FORMULAS_SCHEMA), {name: {formula | rule, scale, smooth,
about, saved}} — what `stl compile --save` and the assistant's formal_stl_compile write.

Limits: robustness mixes units across predicates of one formula (the STL caveat); a run is falsified only at the
recorded rate (events between samples are seen through the samples around them); ρ of a formula with unbounded future
operators near the end of a run reads only what remains (core.stl's finite traces); the planted landscape is a
synthetic check of the searchers, not a measurement of any stack.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

SCHEMA = 'falcon.falsify/1'
SCORE_SCHEMA = 'falcon.stl_score/1'
FORMULAS_SCHEMA = 'falcon.formulas/1'
DEFAULT_SCALE = 1.0
DEFAULT_MARGIN = 1.0
SPEC_KEYS = ('stl', 'rule', 'scale', 'smooth')
_LO, _HI = math.nextafter(0.5, 0.0), math.nextafter(0.5, 1.0)


class STLObjectiveError(ValueError):
    pass


# ================================================================================================ numbers
def jnum(v):
    """ρ for JSON: a float, 'inf' / '-inf', or None when unknown (NaN or None)."""
    if v is None:
        return None
    v = float(v)
    if math.isnan(v):
        return None
    if math.isinf(v):
        return 'inf' if v > 0 else '-inf'
    return v


def fnum(v) -> float:
    """jnum's inverse: a float (NaN for None)."""
    if v is None:
        return float('nan')
    if isinstance(v, str):
        return float(v)                          # 'inf', '-inf'
    return float(v)


def value(rho, scale: float = DEFAULT_SCALE, rho_smooth=None) -> float | None:
    """The search's value of a robustness (module doc): σ(−ρ/scale) — of the smoothed ρ when given — clamped to the
    exact ρ's side of 0.5; None when ρ is unknown."""
    r = fnum(rho)
    if math.isnan(r):
        return None
    s = fnum(rho_smooth) if rho_smooth is not None else float('nan')
    x = r if math.isnan(s) else s
    z = -x / float(scale)
    if math.isinf(z):
        v = 1.0 if z > 0 else 0.0
    elif z >= 0:
        v = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        v = e / (1.0 + e)
    return max(v, _HI) if r < 0 else min(v, _LO)


def falsified(rho) -> bool:
    r = fnum(rho)
    return not math.isnan(r) and r < 0


# ================================================================================================ the specification
def parse_rule_ref(ref: str) -> tuple:
    """'SET:ID[,ID…]' → (set, [ids]); SET is india, generic or a rule file's path (split at its last colon)."""
    rs, sep, ids = str(ref).rpartition(':')
    names = [i.strip() for i in ids.split(',') if i.strip()]
    if not sep or not rs or not names:
        raise STLObjectiveError(f'a rule is SET:ID (e.g. india:following_distance), not {ref!r}')
    return rs, names


def normalise(spec: dict) -> dict:
    """A validated requirement {'stl' | 'rule', 'scale', 'smooth'} plus what describes it: 'formula' (the normal
    form, or the rules' formulas), 'signals', and for rules 'ruleset' {id, sha256} — STLObjectiveError when it does
    not parse, names an unknown signal or rule, or has no formula."""
    from . import rules as R
    from . import stl as S
    stl, rule = (spec or {}).get('stl') or '', (spec or {}).get('rule') or ''
    if bool(stl) == bool(rule):
        raise STLObjectiveError('give a formula (stl) or a rule (SET:ID), one of them')
    try:
        scale = float(spec.get('scale') if spec.get('scale') is not None else DEFAULT_SCALE)
        smooth = float(spec.get('smooth') or 0.0)
    except (TypeError, ValueError):
        raise STLObjectiveError('scale and smooth are numbers') from None
    if not scale > 0 or not math.isfinite(scale):
        raise STLObjectiveError('scale must be positive (the predicate\'s units)')
    if smooth < 0 or not math.isfinite(smooth):
        raise STLObjectiveError('smooth (β) must be 0 (exact) or positive')
    out = {'stl': None, 'rule': None, 'scale': scale, 'smooth': smooth}
    try:
        if stl:
            node = R.parse_formula(str(stl))
            out.update(stl=node.text(), formula=node.text(), signals=sorted(S.names(node)))
        else:
            rs_spec, ids = parse_rule_ref(rule)
            rs = R.load_ruleset(rs_spec)
            by = {r['id']: r for r in rs['rules']}
            bad = [i for i in ids if i not in by]
            if bad:
                raise STLObjectiveError(f'no rule {", ".join(bad)} in the set {rs["id"]} ({", ".join(by)})')
            out.update(rule=f'{rs_spec}:{",".join(ids)}', rule_ids=ids,
                       formula=' and '.join(f'G ({by[i]["formula"]})' for i in ids),
                       signals=sorted({s for i in ids for s in by[i]['signals']}),
                       ruleset={'id': rs['id'], 'sha256': rs['sha256'], 'source': rs['source']})
    except (S.STLError, R.RulesError) as e:
        raise STLObjectiveError(str(e)) from None
    return out


def key_doc(spec: dict) -> dict:
    """What a score depends on (hashed into the study's objective key)."""
    from .rules import SIGNALS_VERSION
    s = normalise(spec)
    return {'stl': s['stl'], 'rule_ids': s.get('rule_ids'), 'ruleset_sha256': (s.get('ruleset') or {}).get('sha256'),
            'scale': s['scale'], 'smooth': s['smooth'], 'signals_version': SIGNALS_VERSION}


def spec_key(spec: dict) -> str:
    from .provenance import sha256_json
    return 'stl-' + sha256_json(key_doc(spec))[:16]


def describe(spec: dict) -> str:
    s = normalise(spec)
    what = f'formula {s["stl"]}' if s['stl'] else f'rule {s["rule"]} ({s["formula"]})'
    out = (f'falsification of the {what}: value σ(−ρ/{s["scale"]:g}) clamped to the verdict\'s side of 0.5 (ρ < 0 '
           f'falsified ⇔ value > 0.5; core.stlobjective)')
    if s['smooth'] > 0:
        out += (f'; the searcher sees the soft-min/max robustness with β = {s["smooth"]:g} (an approximation; the '
                'exact ρ is reported)')
    return out


def _target(s: dict):
    from . import rules as R
    if s['stl']:
        return R.parse_formula(s['stl']), None
    rs_spec, ids = parse_rule_ref(s['rule'])
    return ids, R.load_ruleset(rs_spec)


# ================================================================================================ scoring
def score_trace(tr, spec: dict, series: bool = False) -> dict:
    """A trace's score under a requirement (module doc): {'schema', 'rho', 'rho_smooth', 'value', 'state',
    'falsified', 'worst_t', 'evaluated_share', 'missing_signals', 'formula'}; series adds 't', 'rho_series' (and
    'smooth_series'), JSON-ready."""
    from . import rules as R
    s = normalise(spec)
    target, rs = _target(s)
    r = R.robustness(tr, target, rs, smooth=s['smooth'])
    rho = r['rho_t0']
    out = {'schema': SCORE_SCHEMA, 'rho': jnum(rho), 'rho_smooth': jnum(r.get('rho_smooth')) if s['smooth'] else None,
           'value': value(rho, s['scale'], r.get('rho_smooth') if s['smooth'] else None), 'state': r['state'],
           'falsified': falsified(rho), 'worst_t': None if r['worst_t'] is None else round(float(r['worst_t']), 4),
           'evaluated_share': round(float(r['evaluated_share']), 4), 'missing_signals': r['missing_signals'],
           'formula': r['formula'], 'per_rule': {k: jnum(v) for k, v in (r['per_rule'] or {}).items()} or None}
    if series:
        out['t'] = [round(float(x), 4) for x in r['t']]
        out['rho_series'] = [jnum(x) for x in r['series']]
        if s['smooth']:
            out['smooth_series'] = [jnum(x) for x in r['series_smooth']]
    return out


def score_run(run_dir: str, spec: dict, series: bool = False, xodr: str | None = None) -> dict:
    """score_trace of a recorded run (core.rules.trace_from_run); a run that cannot be read is 'not evaluable' with
    the reason in 'error'."""
    from . import rules as R
    try:
        tr = R.trace_from_run(run_dir, xodr)
    except (R.RulesError, OSError, ValueError, KeyError) as e:
        return {'schema': SCORE_SCHEMA, 'rho': None, 'rho_smooth': None, 'value': None, 'state': 'not evaluable',
                'falsified': False, 'worst_t': None, 'evaluated_share': 0.0, 'missing_signals': [], 'formula': None,
                'per_rule': None, 'error': str(e)}
    out = score_trace(tr, spec, series)
    if tr.notes:
        out['notes'] = list(tr.notes)
    return out


def row_summary(sc: dict) -> dict:
    """The compact record a study keeps per run in results.jsonl (the series stay in stl/<run>-<key>.json)."""
    keep = ('rho', 'rho_smooth', 'value', 'state', 'falsified', 'worst_t', 'evaluated_share', 'missing_signals',
            'error')
    return {k: sc.get(k) for k in keep if k in sc}


def summarise(rows: list, margin: float = DEFAULT_MARGIN) -> dict:
    """Over a study's rows (results.jsonl lines with 'stl'): the best (lowest) ρ, the falsified runs (ρ < 0), the near
    misses (0 ≤ ρ < margin) and the runs whose ρ was unknown — each most critical first."""
    scored = []
    for r in rows:
        st = r.get('stl') or {}
        rho = fnum(st.get('rho'))
        scored.append((rho, r))
    known = [(x, r) for x, r in scored if not math.isnan(x)]
    known.sort(key=lambda p: p[0])

    def item(x, r):
        st = r.get('stl') or {}
        return {'rho': jnum(x), 'value': st.get('value'), 'worst_t': st.get('worst_t'), 'values': r.get('values'),
                'run': r.get('run'), 'key': r.get('key')}
    fals = [item(x, r) for x, r in known if x < 0]
    near = [item(x, r) for x, r in known if 0 <= x < margin]
    return {'evaluated': len(rows), 'known': len(known), 'unknown': len(rows) - len(known),
            'best_rho': jnum(known[0][0]) if known else None, 'best': item(*known[0]) if known else None,
            'falsified': fals, 'near_misses': near, 'margin': float(margin),
            'missing_signals': sorted({m for _, r in scored for m in ((r.get('stl') or {}).get('missing_signals')
                                                                      or ())})}


# ================================================================================================ saved formulas
def formulas_path(workspace: str) -> str:
    return os.path.join(workspace, 'safety', 'formulas.json')


def load_formulas(workspace: str) -> dict:
    p = formulas_path(workspace)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        raise STLObjectiveError(f'{p}: {e}') from None
    if d.get('schema') != FORMULAS_SCHEMA:
        raise STLObjectiveError(f'{p} is not {FORMULAS_SCHEMA}')
    return d.get('formulas') or {}


def save_formula(workspace: str, name: str, spec: dict, about: str = '') -> dict:
    """Add (or replace) a named requirement in formulas.json; returns its entry (normalised)."""
    import re
    from .provenance import utc_now, write_json
    if not re.fullmatch(r'[A-Za-z][\w.-]{0,63}', name or ''):
        raise STLObjectiveError('a name is a letter then letters, digits, _ . - (at most 64)')
    s = normalise(spec)
    fs = load_formulas(workspace)
    fs[name] = {'stl': s['stl'], 'rule': s['rule'], 'scale': s['scale'], 'smooth': s['smooth'],
                'formula': s['formula'], 'signals': s['signals'], 'about': about, 'saved': utc_now(),
                'key': spec_key(s)}
    os.makedirs(os.path.dirname(formulas_path(workspace)), exist_ok=True)
    write_json(formulas_path(workspace), {'schema': FORMULAS_SCHEMA, 'formulas': dict(sorted(fs.items()))})
    return fs[name]


def named(workspace: str, name: str) -> dict:
    fs = load_formulas(workspace)
    if name not in fs:
        raise STLObjectiveError(f'no saved formula {name!r} ({", ".join(fs) or "none saved"})')
    return {k: fs[name].get(k) for k in SPEC_KEYS}


# ================================================================================================ studies
def study_name(scenario: str, spec: dict, weather=(), fps: float = 20) -> str:
    """A falsification study's name: one per scenario, requirement (its key), searched weather and rate."""
    from ..sim.studio import WEATHER_TAGS
    parts = ['falsify', scenario, spec_key(spec)[4:12]] + [WEATHER_TAGS.get(k, k) for k in weather]
    if fps != 20:
        parts.append(f'{fps:g}hz')
    return '-'.join(parts)


def study_rows(study_dir: str) -> list:
    p = os.path.join(study_dir, 'results.jsonl')
    if not os.path.isfile(p):
        return []
    with open(p, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def series_path(study_dir: str, run: str, key: str) -> str:
    return os.path.join(study_dir, 'stl', f'{os.path.basename(os.path.normpath(run))}-{key}.json')


def write_result(study, method: str, budget: int, margin: float = DEFAULT_MARGIN) -> dict:
    """falsify.json of a robustness study (its current results): the requirement, the summary, the searcher."""
    from .provenance import block, write_json
    rows = [r for r in study.current_results() if r.get('stl')]
    doc = {'schema': SCHEMA, 'study': os.path.basename(study.dir), 'scenario': study.scn.get('id'),
           'world': study.world, 'rig': study.rig, 'seed': study.seed, 'method': method, 'budget': int(budget),
           'objective': study.objective_doc(), 'requirement': {k: study.spec.get(k) for k in SPEC_KEYS},
           'formula': (study.objective_doc().get('stl') or {}).get('formula'),
           'summary': summarise(rows, margin),
           'provenance': block({'results': os.path.join(study.dir, 'results.jsonl')}, seed=study.search_seed)}
    write_json(os.path.join(study.dir, 'falsify.json'), doc)
    return doc


def headline(doc: dict) -> str:
    s = doc.get('summary') or {}
    if not s.get('known'):
        miss = ', '.join(s.get('missing_signals') or ()) or 'the run could not be read'
        return f'{s.get("evaluated", 0)} runs, robustness unknown in every one ({miss})'
    best = s['best_rho']
    return (f'{s["evaluated"]} runs: best ρ = {best}{" — FALSIFIED" if s["falsified"] else ""}; '
            f'{len(s["falsified"])} falsified (ρ < 0), {len(s["near_misses"])} near misses (0 ≤ ρ < '
            f'{s["margin"]:g})' + (f', {s["unknown"]} unknown' if s['unknown'] else ''))


# ================================================================================================ the planted check
PLANTED_SPACE = [{'name': 'cut_gap_m', 'lo': 2.0, 'hi': 15.0}, {'name': 'cutter_speed_kmh', 'lo': 35.0, 'hi': 80.0},
                 {'name': 'cut_time_s', 'lo': 1.0, 'hi': 3.0}]
# the falsifying region in unit coordinates: a short gap, a fast cutter, a quick lane change (synthetic)
PLANTED_REGION = {'center': [0.18, 0.82, 0.24], 'radii': [0.065, 0.065, 0.065]}


class PlantedRobustness:
    """A synthetic robustness landscape (not a model of any stack): ρ(x) = gain·(d(x) − 1) + noise, d the ellipsoidal
    distance from the planted centre in unit coordinates (ρ < 0 exactly inside the region: about 0.1 % of the cube),
    noise N(0, noise²) a deterministic function of the parameters. robustness(params) is ρ; value(params) the
    falsifier's objective (σ(−ρ/scale), module doc); binary(params) 1 if falsified else 0 — what a search without
    robustness sees."""

    def __init__(self, region=None, space=None, gain: float = 1.0, noise: float = 0.02, scale: float = 1.0,
                 seed: int = 0):
        from .search import normalise_space
        self.space = normalise_space(space or PLANTED_SPACE)
        r = region or PLANTED_REGION
        self.center, self.radii = np.asarray(r['center'], float), np.asarray(r['radii'], float)
        self.gain, self.noise, self.scale, self.seed = float(gain), float(noise), float(scale), int(seed)
        n = len(self.space)
        self.volume = math.pi ** (n / 2) / math.gamma(n / 2 + 1) * float(np.prod(self.radii))

    def robustness(self, params: dict) -> float:
        import hashlib
        from .search import to_unit
        x = to_unit(self.space, params)
        d = float(np.sqrt(np.sum(((x - self.center) / self.radii) ** 2)))
        h = hashlib.sha256(json.dumps([self.seed, np.round(x, 12).tolist()]).encode()).digest()
        eps = np.random.default_rng(int.from_bytes(h[:8], 'little')).standard_normal()
        return self.gain * (d - 1.0) + self.noise * float(eps)

    def value(self, params: dict) -> float:
        return value(self.robustness(params), self.scale)

    def binary(self, params: dict) -> float:
        return 1.0 if self.robustness(params) < 0 else 0.0


def planted_check(methods=('cmaes', 'cem', 'bayesopt'), seeds=range(10), budget: int = 100,
                  objectives=('robustness', 'binary'), landscape: PlantedRobustness | None = None) -> dict:
    """Each method on each seed with the same budget, on the planted landscape's robustness value and on its binary
    verdict: per method and objective the seeds that reached ρ < 0 within the budget, their share, and the
    evaluations to the first falsifying run (None: not within the budget)."""
    from . import search as S
    f = landscape or PlantedRobustness()
    out = {'budget': int(budget), 'seeds': list(seeds), 'volume': f.volume, 'methods': {}}
    for m in methods:
        out['methods'][m] = {}
        for obj in objectives:
            fn = f.value if obj == 'robustness' else f.binary
            first = []
            for sd in seeds:
                s = S.make(m, f.space, sd, 0.5)
                k, found = 0, None
                while k < budget and found is None:
                    ps = s.ask(min(s.default_batch, budget - k))
                    vals = [fn(p) for p in ps]
                    s.tell(ps, vals)
                    for j, p in enumerate(ps):
                        if found is None and f.robustness(p) < 0:
                            found = k + j + 1
                    k += len(ps)
                first.append(found)
            hit = [x for x in first if x is not None]
            out['methods'][m][obj] = {'found': len(hit), 'share': len(hit) / len(first), 'first': first,
                                      'median_first': float(np.median(hit)) if hit else None}
    return out


# ================================================================================================ commands
def _spec_args(ap, required: bool = False):
    g = ap.add_mutually_exclusive_group(required=required)
    g.add_argument('--stl', default='', help='the requirement as an STL formula over the rule signals (falcon rules '
                                             'signals), e.g. "G (ttc_min_s >= 1.5)"')
    g.add_argument('--rule', default='', help='a rule set\'s rule(s) as the requirement: SET:ID or SET:ID,ID (india, '
                                              'generic or a rule file)')
    g.add_argument('--name', dest='formula_name', default='', help='a requirement saved in the workspace\'s '
                                                                  'safety/formulas.json (needs --project)')
    ap.add_argument('--scale', type=float, default=DEFAULT_SCALE, help='ρ\'s scale in σ(−ρ/scale) (the predicate\'s '
                                                                       'units)')
    ap.add_argument('--smooth', type=float, default=0.0, help='β of the soft-min robustness the searcher sees (0: '
                                                             'exact; an approximation, exact ρ always reported)')


def _spec_from(a, workspace: str = '') -> dict:
    if getattr(a, 'formula_name', ''):
        if not workspace:
            raise STLObjectiveError('--name needs --project (the workspace whose safety/formulas.json holds it)')
        return named(workspace, a.formula_name)
    return {'stl': a.stl or None, 'rule': a.rule or None, 'scale': a.scale, 'smooth': a.smooth}


def main_falsify(argv) -> int:
    """`falcon falsify run|show|check …` (module doc): 0 ok (a falsified requirement is a result), 2 failed, 6
    refused (a requirement that does not parse, an unknown method: core.protocol's codes)."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon falsify', description='search a scenario for a run that violates a '
                                 'signal-temporal-logic requirement (docs/falsification.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run', help='a resumable falsification study of a scenario')
    r.add_argument('project')
    r.add_argument('--scenario', required=True)
    _spec_args(r)
    r.add_argument('--study', default='', help='the study\'s name (default falsify-<scenario>-<requirement key>…)')
    r.add_argument('--method', default='cmaes', help='cmaes (default), cem, bayes, bayes-pof, adaptive or random')
    r.add_argument('--budget', type=int, default=60)
    r.add_argument('--batch', type=int, default=4)
    r.add_argument('--margin', type=float, default=DEFAULT_MARGIN, help='a near miss: 0 ≤ ρ < margin')
    r.add_argument('--world', default='')
    r.add_argument('--rig', default='')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--search-seed', type=int, default=None)
    r.add_argument('--weather', action='append', default=[], metavar='FIELD[:LO:HI]')
    r.add_argument('--no-feasibility', action='store_true')
    r.add_argument('--fps', type=float, default=0.0)
    r.add_argument('--fake-carla', default='')
    from ..sim.service import placement_arg
    placement_arg(r)
    s = sub.add_parser('show', help='a falsification study\'s result: falcon falsify show STUDY_DIR')
    s.add_argument('study')
    s.add_argument('--json', action='store_true')
    c = sub.add_parser('check', help='the searchers on a planted robustness landscape, against a binary objective '
                                     '(no CARLA)')
    c.add_argument('--methods', default='cmaes,cem,bayesopt')
    c.add_argument('--seeds', type=int, default=10)
    c.add_argument('--budget', type=int, default=100)
    c.add_argument('--json', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if a.action == 'check':
        rep = planted_check([m for m in a.methods.split(',') if m], range(a.seeds), a.budget)
        if a.json:
            print(json.dumps(rep, indent=2))
        else:
            print(f'planted region: {100 * rep["volume"]:.2f} % of the cube; budget {rep["budget"]}, '
                  f'{len(rep["seeds"])} seeds')
            for m, d in rep['methods'].items():
                for obj, x in d.items():
                    print(f'{m:9s} {obj:10s} falsified in {x["found"]:2d} of {len(rep["seeds"])} seeds '
                          f'({100 * x["share"]:.0f} %), median evaluations to the first: {x["median_first"]}')
        return 0
    if a.action == 'show':
        p = a.study if a.study.endswith('.json') else os.path.join(a.study, 'falsify.json')
        try:
            with open(p, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            print(f'failed: {e}', flush=True)
            return P.FAILED
        print(json.dumps(doc, indent=2) if a.json else headline(doc), flush=True)
        return P.OK
    return _run(a)


def _run(a) -> int:
    from . import milsearch as MS
    from . import protocol as P
    from . import scenarios as SC
    from .project import Project, ProjectError
    from ..sim.service import study_service
    from ..sim.studio import METHOD_NAMES, Study
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    scn = proj.item('scenarios', a.scenario) or (SC.builtin(a.scenario) if a.scenario in SC.LIBRARY else None)
    if scn is None:
        print(f'failed: no scenario {a.scenario!r}', flush=True)
        return P.FAILED
    if a.method not in METHOD_NAMES:
        print(f'failed: method {a.method!r}: one of {", ".join(METHOD_NAMES)}', flush=True)
        return P.REFUSED
    try:
        req = _spec_from(a, proj.workspace)
        spec = MS.normalise_spec(dict(req, objective='robustness', feasibility=not a.no_feasibility))
        extra = MS.weather_space(a.weather)
    except (STLObjectiveError, MS.MilError) as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    name = a.study or study_name(scn['id'], spec, [w.split(':')[0] for w in a.weather], a.fps or 20)
    from ..cluster.bundle import placement_refusal
    world = a.world or proj.doc['worlds'][0]['id']
    why = placement_refusal(a.on, scn, proj.item('worlds', world))
    if why:
        print(f'failed: {why}', flush=True)
        return P.REFUSED
    try:
        svc = study_service(workspace=proj.workspace, fake_carla=a.fake_carla, placement=a.on)
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    em = P.Emitter()
    em.hello('falsify')
    try:
        st = Study(svc, proj, scn, name, a.world or proj.doc['worlds'][0]['id'], a.rig or proj.doc['rigs'][0]['id'],
                   a.seed, objective=spec, extra_space=extra, fps=a.fps or None, search_seed=a.search_seed)
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.REFUSED
    st.emit = em
    em.log(f'requirement: {MS.describe(spec)}')
    try:
        st.search(a.method, a.budget, a.batch, 0.5,
                  lambda n: em.progress(min(n / max(a.budget, 1), 1.0), f'{n} of {a.budget}', force=True))
        doc = write_result(st, a.method, a.budget, a.margin)
    except (RuntimeError, ValueError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.FAILED
    svc.shutdown()
    em.log(headline(doc))
    s = doc['summary']
    em.result('ok', summary={'study': name, 'evaluated': s['evaluated'], 'best_rho': s['best_rho'],
                             'falsified': len(s['falsified']), 'near_misses': len(s['near_misses']),
                             'unknown': s['unknown']})
    return P.OK


def main_stl(argv) -> int:
    """`falcon stl compile|robustness …` (module doc): 0 ok, 2 failed (a formula that does not parse, a run that
    cannot be read)."""
    import argparse
    from .project import ProjectError
    ap = argparse.ArgumentParser(prog='falcon stl', description='signal temporal logic requirements over the rule '
                                 'signals (docs/falsification.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    c = sub.add_parser('compile', help='parse a requirement: its normal form, signals and horizon; --save names it')
    c.add_argument('formula', nargs='?', default='', help='the formula (or give --rule)')
    c.add_argument('--rule', default='', help='SET:ID[,ID]')
    c.add_argument('--scale', type=float, default=DEFAULT_SCALE)
    c.add_argument('--smooth', type=float, default=0.0)
    c.add_argument('--save', default='', help='save it under this name in the project\'s safety/formulas.json')
    c.add_argument('--about', default='')
    c.add_argument('--project', default='')
    c.add_argument('--json', action='store_true')
    r = sub.add_parser('robustness', help='the robustness of recorded runs')
    r.add_argument('runs', nargs='+')
    _spec_args(r, required=True)
    r.add_argument('--project', default='', help='for --name')
    r.add_argument('--xodr', default=None, help='the map\'s OpenDRIVE file ("none": no file)')
    r.add_argument('--series', action='store_true', help='include the per-sample ρ')
    r.add_argument('--json', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        ws = ''
        if getattr(a, 'project', ''):
            from .project import Project
            ws = Project.load(a.project).workspace
        if a.action == 'compile':
            spec = {'stl': a.formula or None, 'rule': a.rule or None, 'scale': a.scale, 'smooth': a.smooth}
            s = normalise(spec)
            from . import rules as R
            from . import stl as S
            target, rs = _target(s)
            nodes = [target] if rs is None else [r['node'] for r in rs['rules'] if r['id'] in target]
            past = max(S.horizon(n)[0] for n in nodes)
            future = max(S.horizon(n)[1] for n in nodes)
            out = {'formula': s['formula'], 'signals': {k: R.SIGNALS[k][0] for k in s['signals']},
                   'horizon_s': {'past': horizon_text(past), 'future': horizon_text(future)}, 'key': spec_key(s)}
            if a.save:
                if not ws:
                    raise STLObjectiveError('--save needs --project')
                out['saved'] = save_formula(ws, a.save, spec, a.about)
                out['path'] = formulas_path(ws)
            if a.json:
                print(json.dumps(out, indent=2))
            else:
                print(out['formula'])
                print('signals: ' + ', '.join(f'{k} ({u})' for k, u in out['signals'].items()))
                print(f'looks back {out["horizon_s"]["past"]}, ahead {out["horizon_s"]["future"]}; key {out["key"]}')
                if a.save:
                    print(f'saved as {a.save} in {out["path"]}')
            return 0
        spec = _spec_from(a, ws)
        normalise(spec)
        res = {}
        for run in a.runs:
            res[run] = score_run(run, spec, a.series, '' if a.xodr == 'none' else a.xodr)
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            for run, sc in res.items():
                why = sc.get('error') or (f'unknown signals {", ".join(sc["missing_signals"])}'
                                          if sc['missing_signals'] else '')
                print(f'{run}: ρ = {sc["rho"]} ({sc["state"]}), decided at t = {sc["worst_t"]} s, evaluated '
                      f'{100 * sc["evaluated_share"]:.0f} %' + (f'; {why}' if why else ''))
        return 0
    except (STLObjectiveError, ProjectError, OSError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        return 2


def horizon_text(v: float) -> str:
    return 'unbounded' if math.isinf(v) else f'{v:g} s'
