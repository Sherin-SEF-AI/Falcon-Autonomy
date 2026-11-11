"""Hypothesis-driven experiments (ablations) and the experiment registry (standard library + numpy at import; the
drive service, the harness and the models inside the functions). docs/experiments.md is the manual.

An ablation starts from a stated hypothesis and its factors — "the fine-tuned detector raises night recall", factors
model (base, fine-tuned) × weather (clear, rain) × light (day, night) — and Falcon:

1. plans it (plan): the grid of cells (every combination of the factors' levels), the replicates of each cell with
   their seeds, and how many replicates the planned comparison needs — a power analysis of the paired comparison at
   the stated effect, alpha and power (below);
2. runs it (run) through the code paths the pages use: for the "detection" executor every drive cell (the factors
   other than the model) is a recorded drive (core.drive.drive_config through sim.service.CarlaService's job queue,
   the weather of the cell's levels), and every model level is evaluated on the same drive (core.harness.evaluate),
   so the model comparison is paired by construction; each run is a tracked run of an core.experiments.Experiment
   (params, tags, the metric, its pins); runs that finished are skipped, so a stopped experiment resumes;
3. analyses it (analyse): per contrast a paired bootstrap of the difference (core.metrics.paired_bootstrap, the
   expanded percentile interval), the standardised effect with its interval, a two-sided p-value, Holm's correction
   over the contrasts, and the verdict on the hypothesis;
4. reports it (table, core.paper for LaTeX, CSV and figures) and re-runs it (rerun): the same definition and seeds
   into a new experiment, checked against the original (core.reproduce.compare_experiments), with every pin that
   changed listed (core.provenance.pins: Falcon's version, the CARLA build, the world, rig, traffic and weights
   hashes, the seeds).

The spec (SPEC_SCHEMA; normalise() fills the defaults and checks it, against the project when one is given):

    {"id": "night-recall", "name": "…",
     "hypothesis": {"text": "the fine-tuned detector raises night recall", "direction": "increase" | "decrease" |
                    "change", "rule": "all" | "any"},
     "metric": "AR100"                    what one run measures (the detection executor: a number of the harness's
                                          table — AP, AP50, AP75, AR100, AP.<class>, …; higher is better unless
                                          "higher_is_better": false, which only labels the tables)
     "factors": [{"name": "model", "kind": "model", "levels": ["base", "ft"]},
                 {"name": "weather", "kind": "weather", "levels": ["ClearNoon", "HardRainNoon"]},
                 {"name": "light", "kind": "weather",
                  "levels": {"day": {"sun_altitude_angle": 45}, "night": {"sun_altitude_angle": -90}}}],
     "contrasts": [{"factor": "model", "a": "base", "b": "ft", "where": {"light": "night"}, "by": ["weather"],
                    "interaction": {"factor": "light", "a": "day", "b": "night"} (optional)}],
     "design": {"effect": 0.05, "sd": 0.06 | "dz": 0.8, "alpha": 0.05, "power": 0.8, "replicates": null,
                "min_replicates": 2, "max_replicates": 50, "seed": 0},
     "analysis": {"n_boot": 2000, "boot_seed": 0, "ci": 0.95, "correction": "holm" | "bonferroni" | "none",
                  "interval": "expanded" | "percentile", "cluster": "unit" | "replicate"},
     "run": {"executor": "detection" | "simulated", "world", "rig", "traffic", "seconds": 10, "fps": 10, "every": 1,
             "camera", "model" (when no factor is a model), "simulate": {…}}}

Factor kinds: model (registered model ids, evaluated on the same drive), weather (a CARLA preset name, or a dict of
weather fields over the world's weather — several weather factors apply in order), world, rig, traffic (project ids),
and value (any levels; for the simulated executor and executors passed from Python).

Contrasts. A contrast compares level b of a factor with level a, at the levels "where" fixes; "by" splits it into one
contrast per level of those factors; an "interaction" contrast is the difference of that difference between two
levels of another factor ((b − a at d) − (b − a at c)). Its UNIT is everything else: the levels of the factors it
does not name and the replicate; for each unit the runs at a and b (and c, d) make one paired value. Units missing a
run are dropped and counted. With cluster "replicate" the bootstrap resamples whole replicates (the units sharing a
seed) instead of units — use it when cells that share a seed are not independent.

Power analysis (plan). The comparison is planned as a two-sided paired t-test of the units' differences: the stated
effect is dz = effect / sd (sd of the paired differences, e.g. from a pilot) or dz itself; with m contrasts each is
planned at alpha / m (Bonferroni, which Holm's first step equals and its later steps relax); the number of pairs is
the smallest n whose exact power (core.metrics.paired_power: the noncentral t) reaches the target, and the
replicates per cell are that n over the units one replicate gives the contrast, rounded up, the largest over the
contrasts, within [min_replicates, max_replicates] (capped: said so, with the power reached). The analysis is a
paired bootstrap, not a t-test: the expanded percentile interval matches the t interval for moderate n, and the
tests simulate both (tests/test_ablation.py). Each contrast is resampled often enough (boot_size) that the most
extreme quantile it takes — the simultaneous interval's — rests on about 20 resamples, up to MAX_BOOT; a p-value can
never fall below the one half a resample gives, which the result says when it matters.

Verdicts. A contrast SUPPORTS the hypothesis when its Holm-adjusted p-value is below alpha and its difference goes the
stated way (any way for "change"); it CONTRADICTS it when significant the other way; otherwise "no detectable
effect" — not "no effect" — with the power the experiment had for the planned effect at the observed spread. The
hypothesis is supported when every contrast supports it (rule "all") or at least one does and none contradicts it
(rule "any"); contradicted when any contrast contradicts it; else not supported. Intervals are shown at the nominal
level and at the simultaneous (Bonferroni) level 1 − (1 − ci) / m.

The executors. "detection" needs the project and the drive service (CARLA, or the fake one in tests). "simulated"
draws each run's value from a stated model — base + the effects whose "when" matches the cell + a replicate effect
N(0, unit_sd) shared by every cell of the replicate + noise N(0, noise_sd) per run — seeded: a simulation of the
design, never a measurement (its results say so). A Python caller (a tool, the tests) can pass any executor: a
callable run(run, ctx) -> {'metrics': {metric: value}, 'pins': {…}} (optionally with prepare(runs, ctx)).

The registry is the workspace's experiments/ folder (every experiment, of every kind: the files are the truth) and
the catalogue's experiment rows (catalog.add_experiment, re-indexed by Catalog.rebuild); registry() lists it.

The API a tool calls: example_spec(), normalise(spec, project), plan(spec), create(ws, spec, project), run(ws, id,
project, service, executor, emitter, cancel), analyse(ws, id), table(results), rerun(ws, id, project, service,
executor, emitter, cancel), registry(ws), summary(exp_dir), load(ws, id).
"""
from __future__ import annotations

import copy
import itertools
import math
import os

import numpy as np

SPEC_SCHEMA = 'falcon.ablation/1'
PLAN_SCHEMA = 'falcon.ablation_plan/1'
RESULTS_SCHEMA = 'falcon.ablation_results/1'
RERUN_SCHEMA = 'falcon.ablation_rerun/1'
KIND = 'ablation'
FACTOR_KINDS = ('model', 'weather', 'world', 'rig', 'traffic', 'value')
DETECTION_KINDS = ('model', 'weather', 'world', 'rig', 'traffic')
EXECUTORS = ('detection', 'simulated')
DIRECTIONS = ('increase', 'decrease', 'change')
CORRECTIONS = ('holm', 'bonferroni', 'none')
MAX_RUNS = 5000
MAX_BOOT = 50000
DEFAULTS = {
    'id': None, 'name': None, 'hypothesis': {'text': '', 'direction': 'increase', 'rule': 'all'}, 'metric': None,
    'higher_is_better': True, 'factors': [], 'contrasts': [],
    'design': {'effect': None, 'sd': None, 'dz': None, 'alpha': 0.05, 'power': 0.8, 'replicates': None,
               'min_replicates': 2, 'max_replicates': 50, 'seed': 0},
    'analysis': {'n_boot': 2000, 'boot_seed': 0, 'ci': 0.95, 'correction': 'holm', 'interval': 'expanded',
                 'cluster': 'unit'},
    'run': {'executor': 'detection', 'world': None, 'rig': None, 'traffic': None, 'seconds': 10.0, 'fps': 10.0,
            'every': 1, 'camera': None, 'model': None, 'simulate': None},
    'rerun_of': None,
}
SIMULATE_DEFAULTS = {'base': 0.5, 'effects': [], 'unit_sd': 0.0, 'noise_sd': 0.05, 'seed': 0}


class AblationError(Exception):
    pass


# the loop's: a cancel raised while drives are queued is the one its _wait_jobs catches (it cancels them first)
from .loop import Cancelled  # noqa: E402


class _Quiet:
    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass

    def metric(self, *a, **k):
        pass


def example_spec() -> dict:
    """The manual's example: the fine-tuned detector against the base one, at night, over two weathers."""
    return {'schema': SPEC_SCHEMA, 'id': 'night-recall', 'name': 'Fine-tuning and night recall',
            'hypothesis': {'text': 'the fine-tuned detector raises night recall', 'direction': 'increase'},
            'metric': 'AR100',
            'factors': [{'name': 'model', 'kind': 'model', 'levels': ['base', 'finetuned']},
                        {'name': 'weather', 'kind': 'weather', 'levels': ['ClearNoon', 'HardRainNoon']},
                        {'name': 'light', 'kind': 'weather', 'levels': {'day': {'sun_altitude_angle': 45.0},
                                                                         'night': {'sun_altitude_angle': -90.0}}}],
            'contrasts': [{'factor': 'model', 'a': 'base', 'b': 'finetuned', 'where': {'light': 'night'},
                           'by': ['weather']}],
            'design': {'effect': 0.05, 'sd': 0.06, 'alpha': 0.05, 'power': 0.8},
            'run': {'executor': 'detection', 'seconds': 10.0, 'fps': 10.0}}


# ================================================================================================ the spec
def _merge(defaults: dict, given) -> dict:
    out = copy.deepcopy(defaults)
    for k, v in (given or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = {**out[k], **{kk: vv for kk, vv in v.items()}}
        else:
            out[k] = copy.deepcopy(v)
    return out


def _safe_id(s) -> str:
    import re
    t = re.sub(r'[^A-Za-z0-9_.-]', '-', str(s or '')).strip('.-')[:64]
    if not t:
        raise AblationError(f'{s!r} is not a usable experiment id')
    return t


def _levels(f: dict) -> tuple:
    lv = f.get('levels')
    if isinstance(lv, (list, tuple)) and isinstance(f.get('values'), dict) and \
            all(str(x) in f['values'] for x in lv):                    # already normalised: {names, values}
        names = [str(x) for x in lv]
        values = {n: f['values'][n] for n in names}
    elif isinstance(lv, dict):
        names, values = [str(k) for k in lv], {str(k): v for k, v in lv.items()}
    elif isinstance(lv, (list, tuple)):
        names = [str(x) for x in lv]
        values = {str(x): x for x in lv}
    else:
        raise AblationError(f'factor {f.get("name")!r}: levels are a list or a {{name: value}} object')
    if not names:
        raise AblationError(f'factor {f.get("name")!r} has no levels')
    if len(set(names)) != len(names):
        raise AblationError(f'factor {f.get("name")!r} names a level twice')
    return names, values


def normalise(spec: dict, project=None) -> dict:
    """The spec with its defaults, checked (raises AblationError with what is wrong). With a project, the model,
    world, rig and traffic ids and weather presets are checked against it."""
    if not isinstance(spec, dict):
        raise AblationError('the spec is a JSON object')
    s = _merge(DEFAULTS, {k: v for k, v in spec.items() if k != 'schema'})
    s['schema'] = SPEC_SCHEMA
    if isinstance(s['hypothesis'], str):
        s['hypothesis'] = dict(DEFAULTS['hypothesis'], text=s['hypothesis'])
    h = s['hypothesis'] = _merge(DEFAULTS['hypothesis'], s['hypothesis'])
    if not str(h.get('text') or '').strip():
        raise AblationError('state the hypothesis (hypothesis.text)')
    if h['direction'] not in DIRECTIONS:
        raise AblationError(f'hypothesis.direction is one of {", ".join(DIRECTIONS)}')
    if h['rule'] not in ('all', 'any'):
        raise AblationError('hypothesis.rule is all or any')
    s['id'] = _safe_id(s['id'] or h['text'])
    s['name'] = str(s['name'] or h['text'])
    if not s['metric'] or not isinstance(s['metric'], str):
        raise AblationError('name the metric each run measures (metric)')
    facs, seen = [], set()
    for f in s['factors'] or ():
        if not isinstance(f, dict) or not f.get('name'):
            raise AblationError('every factor has a name, a kind and levels')
        name = str(f['name'])
        if name in seen or name in ('replicate', 'seed'):
            raise AblationError(f'factor name {name!r} is used twice or reserved')
        seen.add(name)
        kind = f.get('kind', 'value')
        if kind not in FACTOR_KINDS:
            raise AblationError(f'factor {name!r}: kind is one of {", ".join(FACTOR_KINDS)}')
        names, values = _levels(f)
        facs.append({'name': name, 'kind': kind, 'levels': names, 'values': values})
    if not facs:
        raise AblationError('give at least one factor')
    s['factors'] = facs
    by_name = {f['name']: f for f in facs}
    if not s['contrasts']:
        two = next((f for f in facs if len(f['levels']) >= 2), None)
        if two is None:
            raise AblationError('no factor has two levels to compare')
        s['contrasts'] = [{'factor': two['name'], 'a': two['levels'][0], 'b': two['levels'][1]}]
    cons = []
    for c in s['contrasts']:
        c = {'factor': c.get('factor'), 'a': str(c.get('a')), 'b': str(c.get('b')),
             'where': {str(k): str(v) for k, v in (c.get('where') or {}).items()}, 'by': list(c.get('by') or []),
             'interaction': dict(c['interaction']) if c.get('interaction') else None}
        f = by_name.get(c['factor'])
        if f is None:
            raise AblationError(f'contrast: no factor {c["factor"]!r}')
        if c['a'] not in f['levels'] or c['b'] not in f['levels'] or c['a'] == c['b']:
            raise AblationError(f'contrast on {f["name"]}: a and b are two different levels of '
                                f'{", ".join(f["levels"])}')
        used = {f['name']}
        if c['interaction']:
            g = by_name.get(c['interaction'].get('factor'))
            ia, ib = str(c['interaction'].get('a')), str(c['interaction'].get('b'))
            if g is None or g['name'] in used or ia not in g['levels'] or ib not in g['levels'] or ia == ib:
                raise AblationError('contrast interaction: another factor and two different levels of it')
            c['interaction'] = {'factor': g['name'], 'a': ia, 'b': ib}
            used.add(g['name'])
        for k, v in c['where'].items():
            if k not in by_name or k in used or v not in by_name[k]['levels']:
                raise AblationError(f'contrast where: {k}={v} is not a level of another factor')
        for k in c['by']:
            if k not in by_name or k in used or k in c['where']:
                raise AblationError(f'contrast by: {k!r} is not another factor')
        cons.append(c)
    s['contrasts'] = cons
    d = s['design']
    for k in ('alpha', 'power'):
        if not 0 < float(d[k]) < 1:
            raise AblationError(f'design.{k} lies in (0, 1)')
    if d.get('dz') is None and d.get('effect') is not None and d.get('sd') is not None:
        if float(d['sd']) <= 0:
            raise AblationError('design.sd (the spread of the paired differences) must be positive')
    if d.get('replicates') is None and d.get('dz') is None and (d.get('effect') is None or d.get('sd') is None):
        raise AblationError('state the effect to detect (design.effect and design.sd, or design.dz) or the '
                            'replicates per cell (design.replicates)')
    if d.get('replicates') is not None and int(d['replicates']) < 2:
        raise AblationError('design.replicates is at least 2 (a paired spread needs two units)')
    if int(d['min_replicates']) < 2 or int(d['max_replicates']) < int(d['min_replicates']):
        raise AblationError('design: 2 ≤ min_replicates ≤ max_replicates')
    a = s['analysis']
    if a['correction'] not in CORRECTIONS:
        raise AblationError(f'analysis.correction is one of {", ".join(CORRECTIONS)}')
    if a['interval'] not in ('expanded', 'percentile') or a['cluster'] not in ('unit', 'replicate'):
        raise AblationError('analysis.interval is expanded or percentile; analysis.cluster unit or replicate')
    if not 0 < float(a['ci']) < 1 or int(a['n_boot']) < 100:
        raise AblationError('analysis: ci in (0, 1) and n_boot ≥ 100')
    r = s['run']
    if r['executor'] not in EXECUTORS:
        raise AblationError(f'run.executor is one of {", ".join(EXECUTORS)}')
    if r['executor'] == 'simulated':
        r['simulate'] = _merge(SIMULATE_DEFAULTS, r.get('simulate') or {})
    if r['executor'] == 'detection':
        bad = [f['name'] for f in facs if f['kind'] not in DETECTION_KINDS]
        if bad:
            raise AblationError(f'the detection executor runs model, weather, world, rig and traffic factors; not '
                                f'{", ".join(bad)}')
        if not any(f['kind'] == 'model' for f in facs) and not r.get('model'):
            raise AblationError('name the model to evaluate (a model factor, or run.model)')
        for k in ('world', 'rig', 'traffic'):
            if sum(f['kind'] == k for f in facs) > 1:
                raise AblationError(f'at most one {k} factor')
        if float(r['seconds']) <= 0 or float(r['fps']) <= 0 or int(r['every']) < 1:
            raise AblationError('run: seconds and fps positive, every at least 1')
    if project is not None:
        _check_project(s, project)
    return s


def _check_project(s: dict, project):
    from . import weather as W
    doc = project.doc
    ids = {sec: {x['id'] for x in doc.get(sec) or ()} for sec in ('models', 'worlds', 'rigs', 'traffic_profiles')}
    sec_of = {'model': 'models', 'world': 'worlds', 'rig': 'rigs', 'traffic': 'traffic_profiles'}
    for f in s['factors']:
        if f['kind'] in sec_of:
            for lv in f['levels']:
                v = f['values'][lv]
                if str(v) not in ids[sec_of[f['kind']]]:
                    raise AblationError(f'factor {f["name"]}: no {f["kind"]} {v!r} in the project')
        if f['kind'] == 'weather':
            for lv in f['levels']:
                v = f['values'][lv]
                preset = v if isinstance(v, str) else (v.get('preset') if isinstance(v, dict) else None)
                if not isinstance(v, (str, dict)):
                    raise AblationError(f'factor {f["name"]}: a weather level is a preset name or a dict of fields')
                if preset is not None and preset not in W.PRESETS:
                    raise AblationError(f'factor {f["name"]}: {preset!r} is not a CARLA weather preset')
                if isinstance(v, dict):
                    bad = [k for k in v if k != 'preset' and k not in W.FIELDS]
                    if bad:
                        raise AblationError(f'factor {f["name"]}: not weather fields: {", ".join(bad)}')
    r = s['run']
    for k, sec in (('world', 'worlds'), ('rig', 'rigs'), ('traffic', 'traffic_profiles'), ('model', 'models')):
        if r.get(k) and r[k] not in ids[sec]:
            raise AblationError(f'run.{k}: no {k} {r[k]!r} in the project')
    if r['executor'] == 'detection' and (not ids['worlds'] or not ids['rigs']):
        raise AblationError('the project needs a world and a rig')


# ================================================================================================ the plan
def cells(s: dict) -> list:
    """Every combination of the factors' levels, in the factors' order: [{factor: level}]."""
    names = [f['name'] for f in s['factors']]
    return [dict(zip(names, combo)) for combo in itertools.product(*[f['levels'] for f in s['factors']])]


def expand_contrasts(s: dict) -> list:
    """The contrasts with "by" split into one per level combination: [{'id', 'label', 'factor', 'a', 'b', 'where',
    'interaction'}]."""
    by_name = {f['name']: f for f in s['factors']}
    out = []
    for c in s['contrasts']:
        for combo in itertools.product(*[by_name[k]['levels'] for k in c['by']]) if c['by'] else [()]:
            where = dict(c['where'], **dict(zip(c['by'], combo)))
            lab = f'{c["factor"]}: {c["b"]} − {c["a"]}'
            if c['interaction']:
                g = c['interaction']
                lab = f'({lab}) at {g["factor"]}={g["b"]} − at {g["factor"]}={g["a"]}'
            if where:
                lab += ' | ' + ', '.join(f'{k}={v}' for k, v in where.items())
            out.append({'id': f'c{len(out) + 1}', 'label': lab, 'factor': c['factor'], 'a': c['a'], 'b': c['b'],
                        'where': where, 'interaction': c['interaction']})
    return out


def _free(s: dict, k: dict) -> list:
    """The factors a contrast's unit spans: those it does not name."""
    named = {k['factor']} | set(k['where']) | ({k['interaction']['factor']} if k['interaction'] else set())
    return [f['name'] for f in s['factors'] if f['name'] not in named]


def _units_per_replicate(s: dict, k: dict) -> int:
    by_name = {f['name']: f for f in s['factors']}
    return int(np.prod([len(by_name[n]['levels']) for n in _free(s, k)])) if _free(s, k) else 1


def power_plan(s: dict) -> dict:
    """The sample size of the planned comparison (the module docstring): {'method', 'dz', 'alpha',
    'alpha_per_contrast', 'power', 'replicates', 'capped', 'contrasts': [{'id', 'units_per_replicate',
    'pairs_needed', 'pairs', 'power'}], 'notes'}."""
    from .metrics import paired_power, paired_sample_size
    d = s['design']
    ks = expand_contrasts(s)
    m = len(ks)
    alpha = float(d['alpha'])
    corr = s['analysis']['correction']
    a_c = alpha / m if corr != 'none' and m > 1 else alpha
    dz = d.get('dz')
    if dz is None and d.get('effect') is not None and d.get('sd') is not None:
        dz = float(d['effect']) / float(d['sd'])
    notes = []
    rows = []
    fixed = d.get('replicates')
    need = {}
    if dz is not None:
        for k in ks:
            n = paired_sample_size(float(dz), a_c, float(d['power']))
            need[k['id']] = n['n']
    if fixed is not None:
        reps = int(fixed)
        capped = False
    elif dz is not None:
        want = max(math.ceil(need[k['id']] / _units_per_replicate(s, k)) for k in ks)
        reps = min(max(want, int(d['min_replicates'])), int(d['max_replicates']))
        capped = want > int(d['max_replicates'])
        if capped:
            notes.append(f'{want} replicates per cell would reach the planned power; capped at max_replicates '
                         f'{d["max_replicates"]}: the power reached is below the target (per contrast below)')
    else:
        raise AblationError('nothing to plan: no effect and no replicates')
    for k in ks:
        u = _units_per_replicate(s, k)
        pairs = reps * u
        rows.append({'id': k['id'], 'units_per_replicate': u, 'pairs_needed': need.get(k['id']), 'pairs': pairs,
                     'power': paired_power(pairs, float(dz), a_c) if dz is not None else None})
    if dz is None:
        notes.append('no effect stated: the replicates were given, and the power is not computed')
    if s['analysis']['cluster'] == 'replicate':
        notes.append('cluster "replicate": the analysis resamples whole replicates; the plan counts units as '
                     'pairs, so it is optimistic when the units of a replicate are correlated')
    return {'method': ('two-sided paired t-test of the units\' differences, power from the noncentral t (exact, '
                       'core.metrics.paired_power); each of the m contrasts at alpha/m (Bonferroni, which Holm\'s '
                       'first step equals); pairs = replicates × the units one replicate gives the contrast'),
            'dz': None if dz is None else float(dz), 'effect': d.get('effect'), 'sd': d.get('sd'),
            'alpha': alpha, 'alpha_per_contrast': a_c, 'contrasts_m': m, 'target_power': float(d['power']),
            'replicates': reps, 'capped': capped, 'contrasts': rows, 'notes': notes}


def plan(s: dict) -> dict:
    """The grid, the replicates and seeds, the runs and the power analysis of a normalised spec (PLAN_SCHEMA)."""
    pw = power_plan(s)
    reps = pw['replicates']
    cs = cells(s)
    drive_facs = [f['name'] for f in s['factors'] if f['kind'] != 'model']
    dkeys = {}
    for c in cs:
        dkeys.setdefault(tuple(c[n] for n in drive_facs), len(dkeys))
    seeds = [int(s['design']['seed']) + r for r in range(reps)]
    runs = []
    for r, seed in enumerate(seeds):
        for ci, c in enumerate(cs):
            dk = dkeys[tuple(c[n] for n in drive_facs)]
            runs.append({'name': f'c{ci:03d}-r{r:03d}', 'cell': c, 'cell_index': ci, 'replicate': r, 'seed': seed,
                         'drive': f'd{dk:03d}-r{r:03d}'})
    if len(runs) > MAX_RUNS:
        raise AblationError(f'{len(runs)} runs ({len(cs)} cells × {reps} replicates) is more than {MAX_RUNS}: '
                            'fewer levels, or a larger effect')
    ex = s['run']['executor']
    counts = {'cells': len(cs), 'replicates': reps, 'runs': len(runs),
              'drives': len(dkeys) * reps if ex == 'detection' else 0,
              'evaluations': len(runs) if ex == 'detection' else 0}
    if ex == 'detection':
        counts['simulated_seconds'] = counts['drives'] * float(s['run']['seconds'])
    return {'schema': PLAN_SCHEMA, 'cells': cs, 'contrasts': expand_contrasts(s), 'power': pw, 'seeds': seeds,
            'runs': runs, 'counts': counts}


# ================================================================================================ the registry
def root(workspace: str) -> str:
    return os.path.join(workspace, 'experiments')


def path_of(workspace: str, exp_id: str) -> str:
    """An experiment's folder: experiments/<id>, or exp_id itself when it is a folder with experiment.json."""
    if os.path.isfile(os.path.join(exp_id, 'experiment.json')):
        return os.path.abspath(exp_id)
    return os.path.join(root(workspace), _safe_id(exp_id))


def load(workspace: str, exp_id: str) -> tuple:
    """(Experiment, its normalised spec, its plan) of an ablation; raises AblationError."""
    from . import provenance as PV
    from .experiments import Experiment, ExperimentError
    try:
        exp = Experiment(path_of(workspace, exp_id))
    except ExperimentError as e:
        raise AblationError(str(e)) from None
    if exp.doc.get('kind') != KIND:
        raise AblationError(f'{exp.id} is a {exp.doc.get("kind")} experiment, not an ablation')
    pl = PV.read_json(os.path.join(exp.path, 'plan.json'))
    return exp, exp.doc['definition']['spec'], pl


def _plan_pins(s: dict, pl: dict, project, workspace: str) -> dict:
    from . import provenance as PV
    r = s['run']
    of = {k: [str(f['values'][lv]) for f in s['factors'] if f['kind'] == k for lv in f['levels']]
          for k in ('model', 'world', 'rig', 'traffic')}
    doc = getattr(project, 'doc', None) or {}
    if r['executor'] == 'detection':
        of['world'] = of['world'] or [r.get('world') or (doc.get('worlds') or [{}])[0].get('id')]
        of['rig'] = of['rig'] or [r.get('rig') or (doc.get('rigs') or [{}])[0].get('id')]
        of['model'] = of['model'] or [r.get('model')]
        if r.get('traffic'):
            of['traffic'] = of['traffic'] or [r['traffic']]
    clean = lambda xs: sorted({x for x in xs if x})                              # noqa: E731
    return PV.pins(project, workspace, clean(of['world']), clean(of['rig']), clean(of['model']),
                   clean(of['traffic']), pl['seeds'], None,
                   {'executor': r['executor'], 'spec_sha256': PV.sha256_json(s)})


def create(workspace: str, spec: dict, project=None) -> dict:
    """Register an ablation: normalise the spec, plan it and write experiments/<id>/ (experiment.json — the definition
    and its hash, refused when the id holds another definition —, plan.json with the pins), indexed in the
    catalogue. Returns {'id', 'path', 'plan'}."""
    from . import provenance as PV
    from .experiments import Experiment, ExperimentError
    s = normalise(spec, project)
    pl = plan(s)
    try:
        exp = Experiment.create(root(workspace), s['id'], {'kind': KIND, 'spec': s, 'seed': s['design']['seed']},
                                s['name'])
    except ExperimentError as e:
        raise AblationError(str(e)) from None
    p = os.path.join(exp.path, 'plan.json')
    if not os.path.isfile(p):
        pl['pins'] = _plan_pins(s, pl, project, workspace)
        pl['provenance'] = PV.block(settings={'spec_sha256': PV.sha256_json(s)}, seed=s['design']['seed'])
        PV.write_json(p, pl)
    index(workspace, exp.path)
    return {'id': exp.id, 'path': exp.path, 'plan': PV.read_json(p)}


def status(exp_dir: str) -> dict:
    """{'status': planned | running | failed | cancelled | complete | analysed, 'runs': {state: count}} from the
    files."""
    from . import provenance as PV
    from .experiments import Experiment
    exp = Experiment(exp_dir)
    states = {}
    for n in exp.run_names():
        st = exp.run(n).status or 'running'
        states[st] = states.get(st, 0) + 1
    planned = None
    if os.path.isfile(os.path.join(exp_dir, 'plan.json')):
        try:
            planned = len(PV.read_json(os.path.join(exp_dir, 'plan.json'))['runs'])
        except (OSError, ValueError, KeyError):
            planned = None
    ok = states.get('ok', 0)
    if os.path.isfile(os.path.join(exp_dir, 'results.json')):
        st = 'analysed'
    elif states.get('failed'):
        st = 'failed'
    elif states.get('cancelled'):
        st = 'cancelled'
    elif not states:
        st = 'planned'
    elif planned is not None and ok >= planned:
        st = 'complete'
    else:
        st = 'running'
    return {'status': st, 'runs': states, 'planned': planned}


def summary(exp_dir: str) -> dict:
    """One registry row of any experiment folder: {'id', 'name', 'kind', 'created', 'path', 'status', 'runs',
    'planned', 'hypothesis', 'factors', 'metric', 'verdict', 'rerun_of'} (None where the kind has none)."""
    from . import provenance as PV
    doc = PV.read_json(os.path.join(exp_dir, 'experiment.json'))
    st = status(exp_dir)
    d = doc.get('definition') or {}
    row = {'id': doc.get('id'), 'name': doc.get('name'), 'kind': doc.get('kind'), 'created': doc.get('created'),
           'path': exp_dir, 'status': st['status'], 'runs': st['runs'], 'planned': st['planned'],
           'hypothesis': None, 'factors': None, 'metric': None, 'verdict': None, 'rerun_of': None}
    res = {}
    if os.path.isfile(os.path.join(exp_dir, 'results.json')):
        try:
            res = PV.read_json(os.path.join(exp_dir, 'results.json'))
        except (OSError, ValueError):
            res = {}
    if doc.get('kind') == KIND:
        s = d.get('spec') or {}
        row.update(hypothesis=(s.get('hypothesis') or {}).get('text'), metric=s.get('metric'),
                   factors={f['name']: f['levels'] for f in s.get('factors') or ()}, rerun_of=s.get('rerun_of'),
                   verdict=res.get('verdict'))
    elif doc.get('kind') == 'mixed':
        row['factors'] = {'synthetic share': [f'{round(float(x) * 100):g}%' for x in d.get('shares') or ()]}
        row['metric'] = 'AP'
        arms = [a for a in res.get('arms') or () if a.get('vs_real_only')]
        if arms:
            row['verdict'] = '; '.join(f'{a["arm"]}: {a["vs_real_only"]["AP"]["verdict"]}' for a in arms)
    elif doc.get('kind') == 'generation':                 # generation rounds (core.genround)
        from .genround import HYPOTHESIS
        s = d.get('spec') or {}
        row.update(hypothesis=HYPOTHESIS, metric='AP per cell (class × condition)', verdict=res.get('verdict'),
                   factors={'conditions': list(s.get('conditions') or ())})
    return row


def registry(workspace: str) -> list:
    """Every experiment in the workspace (newest first), as summary() rows; ablations with their hypothesis, factors,
    status and verdict. The files are read (the truth); the catalogue rows are refreshed on the way."""
    out = []
    r = root(workspace)
    if os.path.isdir(r):
        for n in sorted(os.listdir(r)):
            p = os.path.join(r, n)
            if os.path.isfile(os.path.join(p, 'experiment.json')):
                try:
                    out.append(summary(p))
                except (OSError, ValueError, KeyError):
                    continue
    out.sort(key=lambda x: (x.get('created') or '', x.get('id') or ''), reverse=True)
    return out


def index(workspace: str, exp_dir: str):
    """Put the experiment's row in the workspace's catalogue (kind experiment); a catalogue that cannot be opened is
    skipped (the files stay the truth)."""
    import sqlite3
    from .catalog import Catalog
    try:
        cat = Catalog(os.path.join(workspace, 'catalog.sqlite'))
    except (OSError, sqlite3.Error):
        return
    try:
        cat.add_experiment(exp_dir)
    except (OSError, ValueError, KeyError, sqlite3.Error):
        pass
    finally:
        cat.close()


# ================================================================================================ running
class RunContext:
    """What an executor gets: the workspace, the experiment, the spec and plan, the project, the drive service, the
    emitter and cancel()."""

    def __init__(self, workspace, exp, spec, plan_, project=None, service=None, emitter=None, cancel=None):
        self.workspace, self.exp, self.spec, self.plan = workspace, exp, spec, plan_
        self.project, self.service = project, service
        self.em = emitter or _Quiet()
        self.cancel = cancel or (lambda: False)

    def check(self):
        if self.cancel():
            raise Cancelled()


def _seed_of(*parts) -> int:
    from .provenance import sha256_json
    return int(sha256_json(list(parts))[:15], 16)


class SimulatedExecutor:
    """Each run's value drawn from the spec's stated model (the module docstring): a simulation of the design."""

    def __call__(self, r: dict, ctx: RunContext) -> dict:
        sim = ctx.spec['run']['simulate']
        v = float(sim['base'])
        for e in sim.get('effects') or ():
            if all(r['cell'].get(k) == str(lv) for k, lv in (e.get('when') or {}).items()):
                v += float(e.get('add', 0.0))
        u = np.random.default_rng(_seed_of('unit', sim['seed'], r['replicate'])).normal(0.0, float(sim['unit_sd']))
        e = np.random.default_rng(_seed_of('noise', sim['seed'], r['name'])).normal(0.0, float(sim['noise_sd']))
        return {'metrics': {ctx.spec['metric']: float(v + u + e)}, 'pins': {'carla_server': None},
                'simulated': True}


class DetectionExecutor:
    """A recorded drive per drive cell (the job queue, under the CARLA lock), each model level evaluated on it."""

    def _drive_cfg(self, r: dict, ctx: RunContext, out: str) -> dict:
        from . import weather as W
        from .drive import drive_config
        s, proj, svc = ctx.spec, ctx.project, ctx.service
        rs = s['run']
        world, rig, traffic = rs.get('world'), rs.get('rig'), rs.get('traffic')
        weather = None
        for f in s['factors']:
            v = f['values'][r['cell'][f['name']]]
            if f['kind'] == 'world':
                world = str(v)
            elif f['kind'] == 'rig':
                rig = str(v)
            elif f['kind'] == 'traffic':
                traffic = str(v)
        world = world or proj.doc['worlds'][0]['id']
        rig = rig or proj.doc['rigs'][0]['id']
        w = dict((proj.item('worlds', world) or {}).get('weather') or {'preset': W.DEFAULT_PRESET})
        for f in s['factors']:
            if f['kind'] != 'weather':
                continue
            v = f['values'][r['cell'][f['name']]]
            if isinstance(v, str):
                w = {'preset': v}
            elif 'preset' in v:
                w = dict(v)
            else:
                w.update(v)
            weather = w
        from .project import ProjectError
        try:
            cfg = drive_config(proj, world, rig, out, svc.port, svc.tm_port, float(rs['seconds']), float(rs['fps']),
                               int(r['seed']), traffic, workspace=ctx.workspace,
                               render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                               name=f'ablation drive {r["drive"]}')
        except ProjectError as e:
            raise AblationError(str(e)) from None
        if weather is not None:
            cfg['weather'] = W.resolve(weather)
            cfg['weather_source'] = dict(weather)
        return cfg

    def drive_dir(self, r: dict, ctx: RunContext) -> str:
        return os.path.join(ctx.exp.path, 'drives', r['drive'])

    @staticmethod
    def _ok(d: str) -> bool:
        from . import provenance as PV
        try:
            return PV.read_json(os.path.join(d, 'run.json')).get('status') == 'ok'
        except (OSError, ValueError):
            return False

    def prepare(self, runs: list, ctx: RunContext):
        """Record every drive the runs still need (one job each, queued together; the queue runs them one at a time)."""
        import shutil
        if ctx.service is None or ctx.project is None:
            raise AblationError('the detection executor needs the project and the drive service')
        need = {}
        for r in runs:
            d = self.drive_dir(r, ctx)
            if r['drive'] not in need and not self._ok(d):
                need[r['drive']] = (r, d)
        if not need:
            return
        jobs = []
        for k, (r, d) in enumerate(sorted(need.values(), key=lambda x: x[0]['drive'])):
            if os.path.isdir(d):
                shutil.rmtree(d)                              # an unfinished drive is recorded again from the start
            cfg = self._drive_cfg(r, ctx, d)
            jobs.append(ctx.service.submit_drive(cfg, title=f'Ablation {ctx.exp.id}: drive {k + 1} of {len(need)}'))
        ctx.em.log(f'{len(jobs)} drive(s) queued')
        from .loop import LoopError, _wait_jobs
        try:
            _wait_jobs(ctx.service, jobs, ctx.cancel, lambda f, t: ctx.em.progress(0.5 * f, t))
        except LoopError as e:
            raise AblationError(str(e)) from None

    def __call__(self, r: dict, ctx: RunContext) -> dict:
        from . import harness as H
        from . import provenance as PV
        s, proj = ctx.spec, ctx.project
        mid = next((str(f['values'][r['cell'][f['name']]]) for f in s['factors'] if f['kind'] == 'model'),
                   s['run'].get('model'))
        m = proj.item('models', mid)
        if m is None:
            raise AblationError(f'no model {mid!r} in the project')
        d = self.drive_dir(r, ctx)
        if not self._ok(d):
            raise AblationError(f'drive {r["drive"]} did not finish')
        out = os.path.join(ctx.exp.path, 'runs', r['name'], 'evaluation')
        locks = []
        try:
            if m.get('runtime') != 'python':
                from .training import gpu_locks
                locks = gpu_locks(ctx.em, ctx.cancel)
            doc = H.evaluate([d], m, out, ctx.workspace, s['run'].get('camera'), int(s['run']['every']), n_boot=0,
                             seed=int(r['seed']), cancel=ctx.cancel)
        except H.HarnessError as e:
            if str(e) == 'cancelled':
                raise Cancelled() from None
            raise AblationError(f'{r["name"]}: evaluation: {e}') from None
        finally:
            for lk in locks:
                lk.release()
        vals = {row['metric']: row['value'] for row in doc['scores']['table']}
        if s['metric'] not in vals:
            raise AblationError(f'the evaluation has no number {s["metric"]!r}: one of {", ".join(sorted(vals))}')
        from .datasets import drive_key
        rj = PV.read_json(os.path.join(d, 'run.json'))
        host = (rj.get('provenance') or {}).get('host') or {}
        return {'metrics': {s['metric']: vals[s['metric']], 'frames': doc['n_frames']},
                'pins': {'carla_server': host.get('carla_server'),
                         'drive_key': drive_key(PV.read_json(os.path.join(d, 'drive.json'))),
                         'model_weights': PV.weights_sha256(m, ctx.workspace), 'model': mid},
                'inputs': {'drive/run.json': os.path.join(d, 'run.json')},
                'artifacts': [os.path.join(out, 'evaluation.json')],
                'synthetic_model': doc.get('synthetic_model')}


def executor_for(s: dict):
    return {'detection': DetectionExecutor, 'simulated': SimulatedExecutor}[s['run']['executor']]()


def run(workspace: str, exp_id: str, project=None, service=None, executor=None, emitter=None, cancel=None) -> dict:
    """Run (or resume) an ablation's planned runs and analyse them: returns results.json's document. A finished run is
    skipped; the first failure stops it (raises AblationError; running it again resumes). Raises Cancelled."""
    exp, s, pl = load(workspace, exp_id)
    ex = executor if executor is not None else executor_for(s)
    ctx = RunContext(workspace, exp, s, pl, project, service, emitter, cancel)
    todo = [r for r in pl['runs'] if exp.run(r['name']).status != 'ok']
    ctx.em.log(f'{exp.id}: {len(pl["runs"])} runs planned, {len(pl["runs"]) - len(todo)} finished already')
    if todo and hasattr(ex, 'prepare'):
        ctx.check()
        ex.prepare(todo, ctx)
    index(workspace, exp.path)
    lo = 0.5 if isinstance(ex, DetectionExecutor) else 0.0
    for k, r in enumerate(todo):
        ctx.check()
        tr = exp.run(r['name'])
        tr.start({**r['cell'], 'replicate': r['replicate'], 'seed': r['seed']},
                 {'kind': KIND, **r['cell'], 'replicate': r['replicate']}, r['seed'])
        try:
            out = ex(r, ctx)
        except Cancelled:
            tr.finish('cancelled')
            raise
        except AblationError as e:
            tr.finish('failed', str(e))
            index(workspace, exp.path)
            raise
        except Exception as e:                       # noqa: BLE001 — an executor's own failure is the run's
            tr.finish('failed', f'{type(e).__name__}: {e}')
            index(workspace, exp.path)
            raise AblationError(f'{r["name"]}: {type(e).__name__}: {e}') from e
        tr.log_metrics(out.get('metrics') or {})
        for a in out.get('artifacts') or ():
            if os.path.isfile(a):
                tr.log_artifact(a)
        pins = dict(pl.get('pins') or {}, **(out.get('pins') or {}))
        tr.finish('ok', pins=pins, simulated=bool(out.get('simulated')),
                  synthetic_model=out.get('synthetic_model'))
        ctx.em.progress(lo + (1 - lo) * (k + 1) / max(len(todo), 1), f'run {k + 1} of {len(todo)}')
    res = analyse(workspace, exp.id)
    index(workspace, exp.path)
    return res


# ================================================================================================ the analysis
def _mean(its, key):
    return float(np.mean([it[key] for it in its]))


def _dz(its):
    d = np.array([it['b'] - it['a'] for it in its], float)
    if d.size < 2:
        return None
    sd = float(np.std(d, ddof=1))
    return float(d.mean() / sd) if sd > 0 else None


def _items(s: dict, pl: dict, vals: dict, k: dict) -> tuple:
    """The contrast's units: [{'a', 'b', 'replicate', 'unit'}] and how many were dropped (a run missing or without
    the metric)."""
    names = [f['name'] for f in s['factors']]
    by_name = {f['name']: f for f in s['factors']}
    at = {(tuple(r['cell'][n] for n in names), r['replicate']): vals.get(r['name']) for r in pl['runs']}
    free = _free(s, k)
    g = k['interaction']
    items, dropped = [], 0
    for combo in itertools.product(*[by_name[n]['levels'] for n in free]) if free else [()]:
        base = dict(zip(free, combo), **k['where'])
        for rep in range(pl['power']['replicates']):
            def v(fl, gl=None):
                c = dict(base, **{k['factor']: fl})
                if g:
                    c[g['factor']] = gl
                return at.get((tuple(c[n] for n in names), rep))
            if g:
                xs = [v(k['a'], g['a']), v(k['b'], g['a']), v(k['a'], g['b']), v(k['b'], g['b'])]
                if any(x is None for x in xs):
                    dropped += 1
                    continue
                a, b = xs[1] - xs[0], xs[3] - xs[2]
            else:
                a, b = v(k['a']), v(k['b'])
                if a is None or b is None:
                    dropped += 1
                    continue
            items.append({'a': a, 'b': b, 'replicate': rep, 'unit': '/'.join(map(str, combo)) + f'#{rep}'})
    return items, dropped


def boot_size(n_units: int, ci: float, method: str, base: int) -> int:
    """Resamples enough for the most extreme quantile an interval takes (the simultaneous one's) to rest on about 20
    resamples, and for the p-value's floor (half a resample) to sit well below the level: at least base, at most
    MAX_BOOT. With fewer than 5 units the bootstrap has too few distinct resamples for more to help: base."""
    from statistics import NormalDist
    from .metrics import t_ppf
    if n_units < 5:
        return base
    if method == 'expanded':
        q = NormalDist().cdf(-math.sqrt(n_units / (n_units - 1)) * t_ppf((1 + ci) / 2, n_units - 1))
    else:
        q = (1 - ci) / 2
    return int(min(max(base, math.ceil(20.0 / q)), MAX_BOOT))


def _verdict(p_adj, diff, alpha, direction) -> str:
    if p_adj is None or diff is None:
        return 'not measurable'
    if p_adj < alpha:
        want = {'increase': 1, 'decrease': -1, 'change': 0}[direction]
        if want == 0 or (diff > 0) == (want > 0):
            return 'supported'
        return 'contradicted'
    return 'no detectable effect'


def analyse(workspace: str, exp_id: str) -> dict:
    """The contrasts with paired bootstrap intervals, effect sizes, p-values corrected for the m contrasts and the
    verdict; the cells' means with intervals; results.json (recomputed only when a run's value changed)."""
    from . import metrics as M
    from . import provenance as PV
    exp, s, pl = load(workspace, exp_id)
    a = s['analysis']
    vals, states = {}, {}
    sim = False
    for r in pl['runs']:
        tr = exp.run(r['name'])
        info = tr.info()
        states[r['name']] = info.get('status') or 'missing'
        sim = sim or bool(info.get('simulated'))
        if info.get('status') == 'ok':
            v = tr.latest().get(s['metric'])
            vals[r['name']] = float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
    key = PV.sha256_json({'values': vals, 'analysis': a, 'spec': PV.sha256_json(s)})
    out_p = os.path.join(exp.path, 'results.json')
    if os.path.isfile(out_p):
        try:
            old = PV.read_json(out_p)
            if old.get('key') == key:
                return old
        except (OSError, ValueError):
            pass
    ks = pl['contrasts']
    m = len(ks)
    ci, n_boot, bseed = float(a['ci']), int(a['n_boot']), int(a['boot_seed'])
    ci_sim = 1.0 - (1.0 - ci) / m if m > 1 and a['correction'] != 'none' else ci
    unit = 'drive' if a['cluster'] == 'replicate' else 'image'
    alpha = float(s['design']['alpha'])
    rows = []
    for k in ks:
        items, dropped = _items(s, pl, vals, k)
        row = {'id': k['id'], 'label': k['label'], 'factor': k['factor'], 'a': k['a'], 'b': k['b'],
               'where': k['where'], 'interaction': k['interaction'], 'n_pairs': len(items), 'dropped': dropped,
               'mean_a': None, 'mean_b': None, 'diff': None, 'ci_low': None, 'ci_high': None, 'sim_ci_low': None,
               'sim_ci_high': None, 'sd_diff': None, 'dz': None, 'dz_ci_low': None, 'dz_ci_high': None,
               'p_value': None, 'p_adjusted': None, 'n_units': None, 'power_at_planned': None, 'n_boot': None,
               'p_at_floor': None}
        if len(items) >= 2:
            fn = (lambda its, w: _mean(its, w))
            n_u = len({it['replicate'] for it in items}) if unit == 'drive' else len(items)
            nb = boot_size(n_u, ci_sim, a['interval'], n_boot)
            pb = M.paired_bootstrap(fn, items, 'a', 'b', n=nb, seed=bseed, ci=ci, unit=unit,
                                    unit_key='replicate', method=a['interval'])
            ps = M.paired_bootstrap(fn, items, 'a', 'b', n=nb, seed=bseed, ci=ci_sim, unit=unit,
                                    unit_key='replicate', method=a['interval']) if ci_sim != ci else pb
            d = np.array([it['b'] - it['a'] for it in items])
            row.update(mean_a=pb['a'], mean_b=pb['b'], diff=pb['diff'], ci_low=pb['ci_low'], ci_high=pb['ci_high'],
                       sim_ci_low=ps['ci_low'], sim_ci_high=ps['ci_high'], p_value=pb['p_value'],
                       sd_diff=float(np.std(d, ddof=1)), n_units=pb['n_units'], n_boot=nb,
                       p_at_floor=pb['p_b_gt_a'] in (0.0, 1.0))
            dz = _dz(items)
            if dz is not None:
                bz = M.bootstrap(_dz, items, n=n_boot, seed=bseed, ci=ci, unit=unit, unit_key='replicate')
                row.update(dz=dz, dz_ci_low=bz['ci_low'], dz_ci_high=bz['ci_high'])
            planned = pl['power']
            eff = s['design'].get('effect')
            if planned.get('dz') is not None:
                dz_obs_sd = (float(eff) / row['sd_diff']) if eff is not None and row['sd_diff'] else planned['dz']
                row['power_at_planned'] = M.paired_power(row['n_units'], dz_obs_sd, planned['alpha_per_contrast'])
        rows.append(row)
    ps_ = [r['p_value'] for r in rows]
    if a['correction'] == 'holm':
        adj = M.holm(ps_)
    elif a['correction'] == 'bonferroni':
        adj = [None if p is None else min(1.0, p * m) for p in ps_]
    else:
        adj = ps_
    direction = s['hypothesis']['direction']
    target = float(s['design']['power'])
    for r, pa in zip(rows, adj):
        r['p_adjusted'] = pa
        r['verdict'] = _verdict(pa, r['diff'], alpha, direction)
        if r['verdict'] == 'no detectable effect' and r['p_at_floor']:
            r['note'] = (f'every resample went the same way: with {r["n_units"]} units the bootstrap p-value cannot '
                         f'fall below {r["p_value"]:.3g} ({r["n_boot"]} resamples); more replicates can show it')
        elif r['verdict'] == 'no detectable effect' and r['power_at_planned'] is not None and \
                r['power_at_planned'] < target:
            r['note'] = (f'the experiment had {100 * r["power_at_planned"]:.0f} % power for the planned effect at '
                         'the observed spread: a real effect of that size could have been missed')
    verdicts = [r['verdict'] for r in rows]
    if 'contradicted' in verdicts:
        verdict = 'contradicted'
    elif verdicts and (all(v == 'supported' for v in verdicts) if s['hypothesis']['rule'] == 'all'
                       else 'supported' in verdicts):
        verdict = 'supported'
    elif all(v == 'not measurable' for v in verdicts):
        verdict = 'not measurable'
    else:
        verdict = 'not supported'
    n_ok = sum(1 for st in states.values() if st == 'ok')
    warn = []
    if n_ok < len(pl['runs']):
        warn.append(f'{len(pl["runs"]) - n_ok} of {len(pl["runs"])} runs have not finished: the analysis uses the '
                    'complete units only')
    if sim:
        warn.append('SIMULATED: the values come from the spec\'s stated model (run.simulate), not from runs in the '
                    'simulator; they test the design and the analysis, not the hypothesis')
    if any(r['n_units'] is not None and r['n_units'] < 10 for r in rows):
        warn.append('fewer than 10 units in a contrast: its interval rests on few units')
    cells_ = _cell_rows(s, pl, vals, n_boot, bseed, ci)
    doc = {'schema': RESULTS_SCHEMA, 'experiment': exp.id, 'key': key, 'hypothesis': s['hypothesis'],
           'metric': s['metric'], 'higher_is_better': s['higher_is_better'], 'verdict': verdict,
           'factors': [{'name': f['name'], 'kind': f['kind'], 'levels': f['levels']} for f in s['factors']],
           'contrasts': rows, 'cells': cells_, 'design': pl['power'], 'replicates': pl['power']['replicates'],
           'seeds': pl['seeds'], 'runs': {'planned': len(pl['runs']), 'ok': n_ok,
                                          'states': {st: sum(1 for x in states.values() if x == st)
                                                     for st in sorted(set(states.values()))}},
           'simulated': sim, 'warnings': warn,
           'method': {
               'unit': 'one paired value per unit: the levels of the factors a contrast does not name, and the '
                       'replicate (seed); interaction contrasts pair four runs',
               'interval': (f'{100 * ci:g} % paired bootstrap over the units (at least {n_boot} resamples, more '
                            f'when the simultaneous interval\'s tail needs them — each contrast\'s n_boot; seed '
                            f'{bseed}; core.metrics.paired_bootstrap, {a["interval"]} percentile'
                            + (', whole replicates resampled' if unit == 'drive' else '') + ')'),
               'simultaneous': (f'{100 * ci_sim:.4g} % (Bonferroni over {m} contrasts)' if ci_sim != ci else
                                'one contrast: the nominal interval'),
               'p_value': 'two-sided, inverting the bootstrap interval', 'correction': a['correction'],
               'effect_size': 'dz = mean difference / sd of the differences, with a percentile bootstrap interval',
               'verdict': f'supported when the {a["correction"]}-adjusted p < {alpha:g} in the stated direction '
                          f'({direction}); rule {s["hypothesis"]["rule"]} over the contrasts'},
           'provenance': PV.block(settings={'analysis': a}, seed=bseed)}
    PV.write_json(out_p, doc)
    return doc


def _cell_rows(s: dict, pl: dict, vals: dict, n_boot: int, seed: int, ci: float) -> list:
    """Each cell's mean over its replicates with a bootstrap interval (expanded percentile, over replicates)."""
    from . import metrics as M
    out = []
    for i, c in enumerate(pl['cells']):
        xs = [vals.get(r['name']) for r in pl['runs'] if r['cell_index'] == i]
        xs = [x for x in xs if x is not None]
        row = {'cell': c, 'n': len(xs), 'mean': float(np.mean(xs)) if xs else None,
               'sd': float(np.std(xs, ddof=1)) if len(xs) > 1 else None, 'ci_low': None, 'ci_high': None}
        if len(xs) >= 2:
            b = M.bootstrap(lambda its: float(np.mean([it['v'] for it in its])), [{'v': x} for x in xs], n=n_boot,
                            seed=seed, ci=ci, unit='image', method='expanded')
            row.update(ci_low=b['ci_low'], ci_high=b['ci_high'])
        out.append(row)
    return out


def _f(v, nd=4, signed=False):
    if v is None:
        return '—'
    return f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}'


def table(res: dict) -> str:
    """results.json as text: the hypothesis, the verdict, each contrast with its interval, effect size and p."""
    h = res['hypothesis']
    lines = [f'hypothesis: {h["text"]} ({h["direction"]}; rule {h["rule"]})',
             f'verdict: {res["verdict"].upper()}', f'metric: {res["metric"]}; {res["replicates"]} replicates per '
             f'cell; {res["runs"]["ok"]} of {res["runs"]["planned"]} runs', '',
             f'{"contrast":46s} {"n":>4s}  {"Δ [CI]":30s} {"dz":>6s} {"p adj":>7s}  verdict', '-' * 118]
    for r in res['contrasts']:
        ci = (f'{_f(r["diff"], signed=True)} [{_f(r["ci_low"], signed=True)}, {_f(r["ci_high"], signed=True)}]'
              if r['diff'] is not None else '—')
        lines.append(f'{r["label"][:46]:46s} {r["n_pairs"]:4d}  {ci:30s} {_f(r["dz"], 2):>6s} '
                     f'{_f(r["p_adjusted"], 4):>7s}  {r["verdict"]}')
        if r.get('note'):
            lines.append(f'{"":52s}{r["note"]}')
    d = res['design']
    if d.get('dz') is not None:
        lines.append('')
        lines.append(f'planned: dz {d["dz"]:.3g} at alpha {d["alpha_per_contrast"]:.3g} per contrast, power '
                     f'{d["target_power"]:g} → {d["replicates"]} replicates' + (' (capped)' if d['capped'] else ''))
    lines += [f'note: {w}' for w in res.get('warnings') or ()]
    return '\n'.join(lines)


# ================================================================================================ re-running
def rerun(workspace: str, exp_id: str, project=None, service=None, executor=None, emitter=None, cancel=None,
          tol: dict | None = None) -> dict:
    """Re-run an ablation from its own definition and seeds into experiments/<id>-rerun<k>, then check it against the
    original (core.reproduce.compare_experiments): returns rerun.json's document {'original', 'rerun', 'check'}."""
    from . import provenance as PV
    from .reproduce import compare_experiments
    exp, s, pl = load(workspace, exp_id)
    base = s.get('rerun_of') or exp.id
    k = 1
    while os.path.isdir(path_of(workspace, f'{base}-rerun{k}')):
        k += 1
    s2 = copy.deepcopy(s)
    s2['id'], s2['rerun_of'] = f'{base}-rerun{k}', base
    s2['design']['replicates'] = pl['power']['replicates']            # the same runs, not a new plan
    created = create(workspace, s2, project)
    res = run(workspace, created['id'], project, service, executor, emitter, cancel)
    chk = compare_experiments(exp.path, created['path'], tol)
    doc = {'schema': RERUN_SCHEMA, 'original': exp.id, 'rerun': created['id'], 'verdicts': [
        res.get('verdict'), PV.read_json(os.path.join(exp.path, 'results.json')).get('verdict')
        if os.path.isfile(os.path.join(exp.path, 'results.json')) else None], 'check': chk,
        'created': PV.utc_now()}
    PV.write_json(os.path.join(created['path'], 'rerun.json'), doc)
    index(workspace, created['path'])
    return doc


# ================================================================================================ the command
def main_ablation(argv) -> int:
    """`falcon experiment ablation example|plan|new|run|analyse|show|rerun|list …`: protocol lines on stdout for run
    and rerun; 0 ok, 2 failed, 3 cancelled."""
    import argparse
    import json
    import signal
    from . import protocol as P
    from . import provenance as PV
    ap = argparse.ArgumentParser(prog='falcon experiment ablation',
                                 description='hypothesis-driven experiments: plan with a power analysis, run, analyse '
                                             'with paired bootstrap intervals, re-run and check (docs/experiments.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('example', help='print an example spec (JSON)')
    p = sub.add_parser('plan', help='the grid and the sample size of a spec, without creating anything')
    p.add_argument('spec')
    p.add_argument('--project', default='')
    p.add_argument('--json', action='store_true')
    for name, hlp in (('new', 'register an ablation from a spec (experiments/<id>/)'),
                      ('run', 'run (or resume) its runs, then analyse'), ('analyse', 'analyse the finished runs'),
                      ('show', 'the results table'), ('rerun', 're-run it with the same seeds and check it against '
                                                               'the original'), ('list', 'the registry')):
        q = sub.add_parser(name, help=hlp)
        if name == 'new':
            q.add_argument('spec')
        elif name != 'list':
            q.add_argument('id', help='the experiment id (or its folder)')
        q.add_argument('--project', default='', help='the project file (its workspace holds the experiments)')
        q.add_argument('--workspace', default='', help='the workspace, when no project is given')
        q.add_argument('--json', action='store_true')
        if name in ('run', 'rerun'):
            q.add_argument('--fake-carla', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    if a.action == 'example':
        print(json.dumps(example_spec(), indent=2, ensure_ascii=False))
        return P.OK
    proj = None
    try:
        if a.project:
            from .project import Project, ProjectError
            try:
                proj = Project.load(a.project)
            except (OSError, ProjectError) as e:
                raise AblationError(str(e)) from None
        ws = getattr(a, 'workspace', '') or (proj.workspace if proj is not None else '')
        if a.action == 'plan':
            pl = plan(normalise(PV.read_json(a.spec), proj))
            print(json.dumps(pl, indent=2, ensure_ascii=False) if a.json else describe_plan(pl), flush=True)
            return P.OK
        if not ws:
            raise AblationError('--project PROJECT or --workspace DIR is needed')
        if a.action == 'list':
            rows = registry(ws)
            if a.json:
                print(json.dumps(rows, indent=2, ensure_ascii=False, default=str), flush=True)
            for r in rows:
                fac = ' × '.join(f'{k} ({len(v)})' for k, v in (r['factors'] or {}).items())
                print(f'{r["id"]:28s} {r["kind"] or "?":9s} {r["status"]:9s} {r["verdict"] or "—":14s} '
                      f'{(r["hypothesis"] or "")[:50]:50s} {fac}', flush=True)
            return P.OK
        if a.action == 'new':
            c = create(ws, PV.read_json(a.spec), proj)
            print(f'registered {c["id"]} in {c["path"]}', flush=True)
            print(describe_plan(c['plan']), flush=True)
            return P.OK
        if a.action in ('analyse', 'show'):
            res = analyse(ws, a.id) if a.action == 'analyse' else PV.read_json(
                os.path.join(path_of(ws, a.id), 'results.json'))
            print(json.dumps(res, indent=2, ensure_ascii=False) if a.json else table(res), flush=True)
            return P.OK
    except (AblationError, OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    # run, rerun: the drive service when the executor needs it, cancellable
    em = P.Emitter()
    em.hello('ablation')
    state = {'cancel': False}

    def on_sig(*_):
        if not state['cancel']:
            state['cancel'] = True
            raise Cancelled()

    def on_cmd(d):
        if d.get('cmd') == 'cancel':
            import _thread
            _thread.interrupt_main(signal.SIGTERM)
    old = (signal.signal(signal.SIGTERM, on_sig), signal.signal(signal.SIGINT, on_sig))
    P.CommandReader(on_cmd).start()
    svc = None
    try:
        _e, s, _p = load(ws, a.id)
        if s['run']['executor'] == 'detection':
            if proj is None:
                raise AblationError('the detection executor needs --project')
            from ..sim.service import study_service
            svc = study_service(None, ws, fake_carla=a.fake_carla)
        if a.action == 'run':
            res = run(ws, a.id, proj, svc, None, em, lambda: state['cancel'])
            print(table(res), flush=True)
            em.result('ok', results=os.path.join(path_of(ws, a.id), 'results.json'), verdict=res['verdict'])
        else:
            doc = rerun(ws, a.id, proj, svc, None, em, lambda: state['cancel'])
            for c in doc['check']['checks']:
                print(f'  {"ok " if c["ok"] else "-- "} {c["name"]:22s} {c["detail"]}', flush=True)
            for d in doc['check']['pins_changed']:
                print(f'  pin changed: {d["key"]}: {d["a"]} → {d["b"]}', flush=True)
            print('re-run agrees with the original' if doc['check']['ok'] else 'the re-run DIFFERS from the original',
                  flush=True)
            em.result('ok' if doc['check']['ok'] else 'failed', rerun=doc['rerun'], agrees=doc['check']['ok'])
            return P.OK if doc['check']['ok'] else P.FAILED
        return P.OK
    except Cancelled:
        em.result('cancelled')
        return P.CANCELLED
    except (AblationError, OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        return P.FAILED
    finally:
        state['cancel'] = True
        if svc is not None:
            svc.shutdown()
        signal.signal(signal.SIGTERM, old[0])
        signal.signal(signal.SIGINT, old[1])


def describe_plan(pl: dict) -> str:
    """A plan as text: the grid, the runs, the power analysis and its method."""
    pw, c = pl['power'], pl['counts']
    lines = [f'{c["cells"]} cells × {c["replicates"]} replicates = {c["runs"]} runs (seeds {pl["seeds"][0]}…'
             f'{pl["seeds"][-1]})' + (f'; {c["drives"]} drives and {c["evaluations"]} evaluations, '
                                      f'{c["simulated_seconds"]:g} simulated seconds' if c.get('drives') else '')]
    if pw.get('dz') is not None:
        lines.append(f'power analysis: dz {pw["dz"]:.3g}, alpha {pw["alpha"]:g} ({pw["alpha_per_contrast"]:.3g} per '
                     f'contrast over {pw["contrasts_m"]}), target power {pw["target_power"]:g}')
    for r, k in zip(pw['contrasts'], pl['contrasts']):
        lines.append(f'  {k["label"]}: {r["pairs"]} pairs ({r["units_per_replicate"]} per replicate)'
                     + (f', {r["pairs_needed"]} needed, power {100 * r["power"]:.1f} %' if r['power'] is not None
                        else ''))
    lines.append(f'method: {pw["method"]}')
    lines += [f'note: {n}' for n in pw['notes']]
    return '\n'.join(lines)
