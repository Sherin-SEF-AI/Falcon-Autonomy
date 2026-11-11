"""Generation rounds: active learning that generates its data instead of only selecting it (standard library + numpy;
the tools it runs are imported inside functions). Training ▸ Generation, `falcon generate`; docs/generation.md is the
manual.

core.activelearn chooses frames from drives already recorded. A generation round asks the simulator for new data
where the model is weak: it ranks the cells of the operating domain by the expected value of more data there,
records drives and scenario runs in those cells, labels them from the ground truth (automatically: every recorded
frame carries its objects), adds them to a dataset version, retrains the detector and evaluates it again on a held-out
set that no generated drive ever touches — the before and after per cell with bootstrap intervals. The machinery is
the loop's (core.loop: the held-out set, its leak check, cell_table, compare_cells, the retrain from a Falcon-trained
detector's own settings) and the data engine's (sim.engine drive configs, cache keys, dataset versions); the scenarios
come from the coverage grid's fill (core.oddcoverage.cell_scenario); the experiment is in the registry
(core.experiments, core.ablation.registry).

The definition (normalise; fixed once registered — the same id with another definition is refused):

    {"id", "name", "model": a project model id (a detector Falcon trained: core.loop.training_run_of),
     "world": world id, "rig": rig id (its first pinhole camera records boxes2d), "traffic_profile": id or "",
     "conditions": ["clear/day", "rain/night", ...]   weather/light classes of core.oddcoverage (WEATHER × LIGHT),
     "exposure": {condition: weight}                  the condition's share of the operating domain (default equal),
     "classes": [Falcon classes] (default: the model's),
     "held_out": {"drives_per_condition", "drive_s", "fps", "record_every", "seed", "n_boot"},
     "budget": {"drives", "frames"}                    per round,
     "generate": {"drive_s", "fps", "record_every", "scenario_share", "traffic_vehicles", "seed"},
     "acquisition": {"max_cells", "min_objects", "n0", "reference": "overall" | "perfect"},
     "rounds": maximum rounds, "epochs": null (the model's own) or n}

A CELL is a class × condition (a condition is a weather class and a light class, each drive's weather set to the
condition's representative CARLA weather: core.oddcoverage.CELL_WEATHER, CELL_SUN). Its AP on the held-out set
comes with a percentile bootstrap interval over drives (core.loop.cell_table, one range bin from 0 m).

The acquisition rule (ACQUISITION; heuristic, stated): for a cell c of class k in condition w,

    EV(c) = π(w) · gap(c) · s(c) · f(c)
    π(w)   = exposure(w) / Σ exposure                      the condition's share of the domain (default equal)
    gap(c) = max(0, AP_ref − AP(c))                       AP_ref the held-out AP over everything (reference
                                                           "overall"), or 1 ("perfect": for a model weak everywhere)
    s(c)   = sqrt(n0 / (n0 + n(c)))                        n(c) objects of k already in training in condition w (the
                                                           model's sources and the rounds' data): a learning curve's
                                                           diminishing return, n0 = 100 by default
    f(c)   = 1 + (failures + 1) / (runs + 2)               the coverage grid's scenario runs (core.oddcoverage) with an
                                                           actor of class k in w: high-failure and untested cells
                                                           (0.5 with none) count more

Cells with at least min_objects held-out objects and EV > 0 are ranked by EV (ties: more held-out objects); the first
max_cells are chosen and the round's budget (drives, and frames: drives × frames per drive) is shared among them in
proportion to EV (largest remainders; one drive each first). A cell's drives are scenario runs of a library or project
scenario that puts an actor of its class in the condition (core.oddcoverage.cell_scenario — scenario_share of them,
background traffic traffic_vehicles) and free drives with traffic in the condition (sim.engine, cache-keyed). It is
a ranking, not a prediction of the gain: the record says so.

A ROUND is one resumable chain of steps (STEPS): rank → generate → version → retrain → evaluate → compare. Before the
first round the held-out set is FIXED (fix(): drives_per_condition drives per condition at fixed seeds, recorded once
through core.loop.ensure_evalset and hashed into state.json) and the starting model evaluated on it; every round
checks the manifest against that hash, refuses a generated drive whose cache key or folder is a held-out drive's, and
refuses any training source that is, holds or lists one (core.loop.assert_no_leak). Each round's version holds every
drive generated so far (sim.engine.new_version, all to train); the retrain is the starting model's own training
settings (architecture, initialisation, seed, epochs) on its own sources plus that version — no warm start, so rounds
differ by their data. Before and after are the previous round's (or the baseline's) evaluation and this round's,
on the same held-out frames: per cell its AP with its interval, and the difference with a paired bootstrap over drives
(core.loop.compare_cells); improvement only when the interval of the difference lies wholly above zero.

STOP (stop_reason): when no chosen cell improved in a round; when no cell has a positive expected value; after
"rounds" rounds. A stopped or complete chain is not run again (a new id starts another). The registry row
(core.ablation.summary) has the rule as its hypothesis and the outcome as its verdict.

Files: experiments/<id>/experiment.json (the definition), plan.json (the rounds), state.json (the held-out set's
hash, the baseline, each round's status, why it stopped), baseline/ (the starting model's evaluation), rounds/rNNN/
(round.json with each step's state, outputs and hashes; rank.json, generate.json, version.json, retrain-config.json,
after/, cells.json), runs/round-NNN/ (a tracked run: params, metrics), results.json (every round's rows). The
generated drives: datasets/gen-<id>/drives/<cache key[:16]>, versions in datasets/gen-<id>/versions/.
"""
from __future__ import annotations

import copy
import math
import os
import re
import time

from . import loop as L
from . import provenance as PV

KIND = 'generation'
STATE_SCHEMA = 'falcon.generation_state/1'
ROUND_SCHEMA = 'falcon.generation_round/1'
RESULTS_SCHEMA = 'falcon.generation_results/1'
STEPS = ('rank', 'generate', 'version', 'retrain', 'evaluate', 'compare')
STEP_TITLES = {'rank': 'Rank the cells', 'generate': 'Generate drives and scenario runs', 'version': 'Dataset version',
               'retrain': 'Retrain', 'evaluate': 'Evaluate on the held-out set', 'compare': 'Before and after'}
GEN_SEED_BASE = 700_000_000              # generated drives' seeds, far from the held-out set's (loop.EVAL_SEED_BASE)
HYPOTHESIS = ('data generated for the cells of highest expected value raises their AP on a held-out set the '
              'generated data never touches')
ACQUISITION = ('EV(c) = π(w) · gap(c) · s(c) · f(c): π the condition\'s exposure share; gap = max(0, AP_ref − AP(c)) '
               'on the held-out set; s = sqrt(n0 / (n0 + n(c))), n(c) the class\'s training objects in the condition; '
               'f = 1 + (failures + 1) / (runs + 2) of the coverage grid\'s scenario runs with the class in the '
               'condition. Heuristic: a ranking, not a predicted gain')
DEFAULTS = {
    'conditions': ['clear/day', 'rain/day', 'fog/day', 'clear/night'], 'exposure': {}, 'classes': [],
    'traffic_profile': '',
    'held_out': {'drives_per_condition': 3, 'drive_s': 20.0, 'fps': 20.0, 'record_every': 5, 'seed': 1,
                 'n_boot': 1000},
    'budget': {'drives': 8, 'frames': 400},
    'generate': {'drive_s': 20.0, 'fps': 20.0, 'record_every': 5, 'scenario_share': 0.5, 'traffic_vehicles': 10,
                 'seed': 2},
    'acquisition': {'max_cells': 3, 'min_objects': 30, 'n0': 100.0, 'reference': 'overall'},
    'rounds': 3, 'epochs': None,
}
# a detector class -> the scenario actor kinds that put one in the frame (Indian stand-ins left out: their labelled
# class is the stand-in blueprint's)
CLASS_KINDS = {'car': ('car', 'van'), 'truck': ('truck',), 'bus': ('bus',), 'motorcycle': ('motorcycle',),
               'bicycle': ('bicycle',), 'pedestrian': ('pedestrian', 'child'), 'rider': ('motorcycle', 'bicycle')}


class GenError(Exception):
    pass


class Stopped(Exception):
    """The chain stops here, for a reason the record keeps (the stop rule, a model Falcon cannot retrain)."""


Cancelled = L.Cancelled


# ================================================================================================ the definition
def _merge(d, g):
    return L._merge(d, g)


def condition_labels(cond: str) -> tuple:
    from . import oddcoverage as OC
    w, _, l = str(cond).partition('/')
    if w not in OC.WEATHER or l not in OC.LIGHT:
        raise GenError(f'condition {cond!r}: weather/light with weather one of {", ".join(OC.WEATHER)} and light one '
                       f'of {", ".join(OC.LIGHT)}')
    return w, l


def condition_weather(cond: str) -> dict:
    """The condition's representative CARLA weather (core.oddcoverage: each lands in its own class)."""
    from . import oddcoverage as OC
    w, l = condition_labels(cond)
    return dict(OC.CELL_WEATHER[w], sun_altitude_angle=OC.CELL_SUN[l])


def frames_per_drive(g: dict) -> int:
    return max(int(float(g['drive_s']) * float(g['fps']) / max(int(g['record_every']), 1)), 1)


def normalise(spec: dict, project) -> dict:
    """The definition with defaults filled in, or GenError naming every problem."""
    from . import harness as H
    errs = []
    s = _merge({'id': '', 'name': '', 'model': '', 'world': '', 'rig': '', **copy.deepcopy(DEFAULTS)}, spec or {})
    if not re.match(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$', str(s['id'] or '')):
        errs.append('id: letters, digits, "_", "." or "-", at most 41 characters')
    s['name'] = s['name'] or s['id']
    m = project.item('models', s['model']) if s['model'] else None
    if m is None:
        errs.append(f'model: no model {s["model"]!r} in the project' if s['model'] else 'model: a detector id')
    else:
        try:
            L.training_run_of(m)
        except L.Stopped as e:
            errs.append(f'model: {e}')
    if project.item('worlds', s['world']) is None:
        errs.append(f'world: no world {s["world"]!r} in the project')
    rig = project.item('rigs', s['rig'])
    if rig is None:
        errs.append(f'rig: no rig {s["rig"]!r} in the project')
    else:
        cam = next((x for x in rig.get('sensors', []) if x.get('type') == 'camera.pinhole'), None)
        if cam is None or 'boxes2d' not in (cam.get('labels') or []):
            errs.append(f'rig {rig["name"]!r}: its first pinhole camera must record the boxes2d label (the labels '
                        'come from it)')
    if s['traffic_profile'] and project.item('traffic_profiles', s['traffic_profile']) is None:
        errs.append(f'traffic_profile: no profile {s["traffic_profile"]!r}')
    if not s['conditions'] or len(set(s['conditions'])) != len(s['conditions']):
        errs.append('conditions: at least one, each once')
    for c in s['conditions']:
        try:
            condition_labels(c)
        except GenError as e:
            errs.append(str(e))
    for c, v in (s['exposure'] or {}).items():
        if c not in s['conditions'] or float(v) < 0:
            errs.append(f'exposure: {c!r} is not a condition of the definition, or its weight is negative')
    if s['exposure'] and sum(float(s['exposure'].get(c, 0.0)) for c in s['conditions']) <= 0:
        errs.append('exposure: the weights sum to zero')
    if m is not None:
        known = H.model_classes(m)
        s['classes'] = [c for c in (s['classes'] or known) if c in known]
        if not s['classes']:
            errs.append('classes: none the model can name')
    ho, b, g, a = s['held_out'], s['budget'], s['generate'], s['acquisition']
    for sect, keys in ((ho, ('drives_per_condition', 'record_every', 'n_boot')), (b, ('drives', 'frames')),
                       (g, ('record_every',)), (a, ('max_cells', 'min_objects'))):
        for k in keys:
            if int(sect[k]) < 1:
                errs.append(f'{k}: at least 1')
    if not 0.0 <= float(g['scenario_share']) <= 1.0:
        errs.append('generate.scenario_share: between 0 and 1')
    if a['reference'] not in ('overall', 'perfect'):
        errs.append('acquisition.reference: overall or perfect')
    if int(s['rounds']) < 1:
        errs.append('rounds: at least 1')
    if int(b['frames']) < frames_per_drive(g):
        errs.append(f'budget.frames: at least one drive\'s frames ({frames_per_drive(g)})')
    if errs:
        raise GenError('; '.join(errs))
    return s


def example_spec(project) -> dict:
    """A definition for the project: its first Falcon-trained detector, first world and a labelled rig."""
    det = next((m['id'] for m in project.doc.get('models') or () if m.get('kind') == 'detector'
                and (m.get('source') or {}).get('training_run')), '')
    rig = next((r['id'] for r in project.doc.get('rigs') or () for x in r.get('sensors', [])
                if x.get('type') == 'camera.pinhole' and 'boxes2d' in (x.get('labels') or [])),
               (project.doc.get('rigs') or [{}])[0].get('id', ''))
    world = (project.doc.get('worlds') or [{}])[0].get('id', '')
    return {'id': 'gen-1', 'model': det, 'world': world, 'rig': rig, **copy.deepcopy(DEFAULTS)}


def held_out_odd(s: dict) -> dict:
    """The held-out set as a loop ODD (core.loop.ensure_evalset records and hashes it): one weather entry per
    condition, drives_per_condition drives each at fixed seeds."""
    ho = s['held_out']
    return L._merge(L.ODD_DEFAULTS, {
        'schema': L.ODD_SCHEMA, 'id': f'gen-{s["id"]}'[:41], 'name': f'Generation {s["id"]}: held-out set',
        'rig': s['rig'], 'worlds': [s['world']], 'traffic_profile': s['traffic_profile'],
        'weathers': [dict(condition_weather(c), name=c, preset='ClearNoon') for c in s['conditions']],
        'classes': list(s['classes']), 'range_bins': [0.0],
        'evaluation': {'drives_per_cell': int(ho['drives_per_condition']), 'drive_s': float(ho['drive_s']),
                       'fps': float(ho['fps']), 'record_every': int(ho['record_every']), 'seed': int(ho['seed']),
                       'every': 1, 'n_boot': int(ho['n_boot'])}})


def gen_spec(s: dict) -> dict:
    """The dataset spec the generated drives belong to (sim.engine: their folders and versions)."""
    g = s['generate']
    return {'id': f'gen-{s["id"]}', 'name': f'Generation {s["id"]}', 'rig': s['rig'], 'fps': float(g['fps']),
            'record_every': int(g['record_every']), 'drive_s': float(g['drive_s']), 'worlds': [s['world']],
            'weathers': [], 'scenarios': [], 'targets': {}, 'bins': {'distance_m': [0.0, L.SPEC_FAR_M]},
            'bin_targets': {}, 'traffic_profile': s['traffic_profile'] or None, 'seed': int(g['seed']),
            'max_drives': 100000, 'made_by': f'generation {s["id"]}'}


# ================================================================================================ the rule
def exposure_share(s: dict) -> dict:
    e = s.get('exposure') or {}
    w = {c: float(e.get(c, 1.0 if not e else 0.0)) for c in s['conditions']}
    tot = sum(w.values()) or 1.0
    return {c: v / tot for c, v in w.items()}


def rank_cells(rows: list, overall, s: dict, train_counts: dict, failures: dict) -> list:
    """The acquisition rule (module doc) over cell rows ({'class', 'condition', 'AP', 'ci_low', 'ci_high',
    'n_objects'}; condition = a condition id). train_counts: {(class, condition): objects in training};
    failures: {(class, condition): (failures, runs)}. Returns every row with 'ev' and its terms, ranked, the chosen
    ones with 'rank' (1-based)."""
    a = s['acquisition']
    ref = 1.0 if a['reference'] == 'perfect' else overall
    pi = exposure_share(s)
    n0 = float(a['n0'])
    out = []
    for r in rows:
        r = dict(r)
        key = (r['class'], r['condition'])
        n = int(train_counts.get(key, 0))
        k, m = failures.get(key, (0, 0))
        gap = None if ref is None or r.get('AP') is None else max(0.0, float(ref) - float(r['AP']))
        terms = {'pi': pi.get(r['condition'], 0.0), 'gap': gap, 's': math.sqrt(n0 / (n0 + n)),
                 'f': 1.0 + (k + 1.0) / (m + 2.0), 'train_objects': n, 'failures': int(k), 'runs': int(m)}
        r['ev'] = None if gap is None else terms['pi'] * gap * terms['s'] * terms['f']
        r['terms'] = terms
        r['eligible'] = bool(r['n_objects'] >= int(a['min_objects']) and r['ev'] is not None and r['ev'] > 0)
        r.pop('rank', None)
        out.append(r)
    out.sort(key=lambda r: (not r['eligible'], -(r['ev'] or 0.0), -r['n_objects'], r['class'], r['condition']))
    for i, r in enumerate([r for r in out if r['eligible']][:int(a['max_cells'])]):
        r['rank'] = i + 1
    return out


def allocate(evs: list, drives: int) -> list:
    """Drives per chosen cell in proportion to EV (largest remainders), one each first while drives last."""
    n = len(evs)
    if n == 0 or drives <= 0:
        return [0] * n
    if drives <= n:
        order = sorted(range(n), key=lambda i: -evs[i])
        return [1 if i in order[:drives] else 0 for i in range(n)]
    rest = drives - n
    tot = sum(evs) or 1.0
    raw = [rest * e / tot for e in evs]
    base = [int(math.floor(x)) for x in raw]
    left = rest - sum(base)
    for i in sorted(range(n), key=lambda i: -(raw[i] - base[i]))[:left]:
        base[i] += 1
    return [1 + b for b in base]


def round_drives(s: dict) -> int:
    """The drives one round may record: the drive budget, and the frame budget over a free drive's frames."""
    return max(0, min(int(s['budget']['drives']), int(s['budget']['frames']) // frames_per_drive(s['generate'])))


def stop_reason(cells: list) -> str | None:
    """Why the chain stops after a round's comparison of its chosen cells (None: it goes on)."""
    if not cells:
        return 'no cell was chosen'
    if not any(L.verdict(c['diff']) == 'improvement' for c in cells):
        return ('no chosen cell improved: none of their paired 95 % intervals of the AP difference lies wholly above '
                'zero')
    return None


# ================================================================================================ inputs of the rule
def run_condition(run_dir: str) -> str | None:
    """A recorded drive's condition id (weather/light classes of the weather it used), or None."""
    from . import oddcoverage as OC
    cfg = None
    for name in ('drive.json', 'run.json'):           # the config, or the one run.json keeps in its provenance
        try:
            d = PV.read_json(os.path.join(run_dir, name))
        except (OSError, ValueError):
            continue
        cfg = d if name == 'drive.json' else (d.get('provenance') or {}).get('settings')
        if cfg and cfg.get('weather'):
            break
    if not cfg or not cfg.get('weather'):
        return None
    w = cfg['weather']
    return f'{OC.weather_class(w)}/{OC.light_class(w.get("sun_altitude_angle", 45.0))}'


def train_counts(sources: list, classes, conditions) -> tuple:
    """({(class, condition): objects in the training sources' drives}, notes): real sets have no condition here."""
    from . import datasets as D
    out, notes, seen = {}, [], set()
    for src in sources:
        runs = L.source_runs(src)
        if not runs and (src.get('kind') if isinstance(src, dict) else str(src).split(':')[0]) not in ('falcon',
                                                                                                        'version'):
            notes.append(f'{src.get("kind") if isinstance(src, dict) else src}: a real set, its conditions not '
                         'counted')
        for r in runs:
            rp = os.path.realpath(r)
            if rp in seen:
                continue
            seen.add(rp)
            cond = run_condition(r)
            if cond not in conditions:
                continue
            try:
                cov = D.coverage_of_run(r)
            except (OSError, ValueError, KeyError):
                continue
            for c in classes:
                n = int((cov.get('objects') or {}).get(c, 0))
                if n:
                    out[(c, cond)] = out.get((c, cond), 0) + n
    return out, notes


def coverage_failures(ws: str, classes, conditions) -> dict:
    """{(class, condition): (failures, runs)} of the coverage grid's scenario runs (core.oddcoverage.collect) with an
    actor of the class's kinds in the condition."""
    from . import oddcoverage as OC
    out = {}
    try:
        samples = OC.collect(ws)['samples']
    except (OSError, ValueError, KeyError):
        return out
    for smp in samples:
        hit = set()
        for cid in smp['cells']:
            lab = OC.parse_cell(cid)
            cond = f'{lab["weather"]}/{lab["light"]}'
            if cond not in conditions:
                continue
            for c in classes:
                if lab['actor'] in CLASS_KINDS.get(c, ()):
                    hit.add((c, cond))
        for key in hit:
            k, m = out.get(key, (0, 0))
            out[key] = (k + int(smp['fail']), m + 1)
    return out


def scenario_for(cls: str, cond: str, scenarios: list) -> dict | None:
    """A scenario that puts an actor of the class in the condition: the coverage grid's fill (core.oddcoverage) —
    the first library or project scenario with such an actor, copied for its cell (the condition's weather and
    light, its own road and the first speed band its ego speed reaches)."""
    from . import oddcoverage as OC
    w, l = condition_labels(cond)
    for kind in CLASS_KINDS.get(cls, ()):
        for base, kinds, road, _name, rng in OC._producers(scenarios):
            if kind not in kinds:
                continue
            lo_v, hi_v = (rng, rng) if not isinstance(rng, tuple) else rng
            band = next((OC.speed_label(a, b) for a, b in zip(OC.SPEED_BINS, OC.SPEED_BINS[1:])
                         if min(hi_v, b) - max(lo_v, a) > 0 or a <= lo_v < b), None)
            if band is None:
                continue
            cid = OC.cell_id({'weather': w, 'road': road, 'actor': kind, 'speed': band, 'light': l})
            return OC.cell_scenario(base, cid)
    return None


def all_scenarios(project) -> list:
    from . import scenarios as SC
    return [SC.builtin(k) for k in SC.LIBRARY] + [x for x in project.doc.get('scenarios') or ()
                                                  if not x.get('builtin')]


# ================================================================================================ files
def exp_dir(ws: str, exp_id: str) -> str:
    from .ablation import root
    return os.path.join(root(ws), exp_id)


def round_dir(ws: str, exp_id: str, n: int) -> str:
    return os.path.join(exp_dir(ws, exp_id), 'rounds', f'r{int(n):03d}')


def read_state(ws: str, exp_id: str) -> dict:
    p = os.path.join(exp_dir(ws, exp_id), 'state.json')
    if os.path.isfile(p):
        return PV.read_json(p)
    return {'schema': STATE_SCHEMA, 'id': exp_id, 'status': 'planned', 'held_out': None, 'baseline': None,
            'rounds': [], 'stopped': None}


def write_state(ws: str, exp_id: str, st: dict):
    st['updated'] = PV.utc_now()
    PV.write_json(os.path.join(exp_dir(ws, exp_id), 'state.json'), st)


def create(ws: str, spec: dict, project) -> dict:
    """Register the definition (experiments/<id>/experiment.json; the same id with another definition is refused) and
    its plan (the rounds). Returns {'id', 'path', 'spec'}."""
    from .ablation import index, root
    from .experiments import Experiment, ExperimentError
    s = normalise(spec, project)
    try:
        exp = Experiment.create(root(ws), s['id'], {'kind': KIND, 'spec': s, 'seed': s['held_out']['seed']},
                                s['name'])
    except ExperimentError as e:
        raise GenError(str(e)) from None
    p = os.path.join(exp.path, 'plan.json')
    if not os.path.isfile(p):
        PV.write_json(p, {'schema': 'falcon.generation_plan/1', 'runs': [f'round-{n:03d}' for n in
                                                                          range(1, int(s['rounds']) + 1)],
                          'drives_per_round': round_drives(s), 'frames_per_drive': frames_per_drive(s['generate']),
                          'held_out_drives': len(s['conditions']) * int(s['held_out']['drives_per_condition']),
                          'rule': ACQUISITION})
    if not os.path.isfile(os.path.join(exp.path, 'state.json')):
        write_state(ws, exp.id, read_state(ws, exp.id))
    index(ws, exp.path)
    return {'id': exp.id, 'path': exp.path, 'spec': s}


def load(ws: str, exp_id: str) -> tuple:
    """(experiment folder, definition, state); GenError when it is not a generation experiment."""
    d = exp_dir(ws, exp_id)
    p = os.path.join(d, 'experiment.json')
    if not os.path.isfile(p):
        raise GenError(f'no experiment {exp_id!r} in {os.path.dirname(d)}')
    doc = PV.read_json(p)
    if doc.get('kind') != KIND:
        raise GenError(f'{exp_id} is a {doc.get("kind")} experiment, not a generation experiment')
    return d, doc['definition']['spec'], read_state(ws, exp_id)


def describe_plan(s: dict) -> str:
    fpd = frames_per_drive(s['generate'])
    return (f'{s["id"]}: model {s["model"]}, world {s["world"]}, rig {s["rig"]}; held-out set '
            f'{len(s["conditions"])} condition(s) × {s["held_out"]["drives_per_condition"]} drives of '
            f'{s["held_out"]["drive_s"]:g} s (fixed before round 1, never trained on); per round at most '
            f'{round_drives(s)} drives ({s["budget"]["drives"]} drives, {s["budget"]["frames"]} frames, {fpd} frames '
            f'per free drive) for the {s["acquisition"]["max_cells"]} cells of highest expected value; up to '
            f'{s["rounds"]} rounds, stopping when no chosen cell improves.\nRule: {ACQUISITION}')


# ================================================================================================ records
def _hash(ws, paths) -> dict:
    return L._hash_outputs(ws, paths)


def _up(rd: dict, step: str, before: dict) -> str:
    i = STEPS.index(step)
    return PV.sha256_json([before] + [rd['steps'][x].get('outputs') or {} for x in STEPS[:i]])


def step_done(ws: str, rd: dict, step: str, before: dict) -> bool:
    rec = rd['steps'][step]
    return rec.get('state') == 'done' and L._outputs_ok(ws, rec.get('outputs')) and rec.get('upstream') == \
        _up(rd, step, before)


def results(ws: str, exp_id: str) -> dict:
    """Every round's rows (results.json's content): the chosen cells with their EV, the drives and frames, AP over
    everything and per cell before and after with intervals and the paired difference, the verdicts; why it
    stopped."""
    d, s, st = load(ws, exp_id)
    rows = []
    for r in st['rounds']:
        rdir = os.path.join(ws, r['dir'])
        try:
            cells = PV.read_json(os.path.join(rdir, 'cells.json'))
            rank = PV.read_json(os.path.join(rdir, 'rank.json'))
            gen = PV.read_json(os.path.join(rdir, 'generate.json'))
        except (OSError, ValueError):
            continue
        rows.append({'round': r['n'], 'status': r['status'], 'chosen': [
            {k: c.get(k) for k in ('class', 'condition', 'AP', 'ci_low', 'ci_high', 'n_objects', 'ev', 'terms',
                                   'drives')} for c in rank['chosen']],
            'drives': len(gen['drives']), 'frames': gen['frames'], 'objects': gen.get('objects'),
            'AP': cells['headline'], 'chosen_cells': cells['chosen'], 'other_cells': cells['others'],
            'improved': cells['improved'], 'model_after': r.get('model')})
    warn = []
    if int(s['held_out']['drives_per_condition']) < 5:
        warn.append(f'{s["held_out"]["drives_per_condition"]} held-out drives per condition: the intervals over '
                    'drives are not reliable below five (core.harness.MIN_DRIVES)')
    verdict = (f'stopped after round {len(rows)}: {st["stopped"]}' if st.get('stopped') else
               f'complete: {len(rows)} round(s)' if st['status'] == 'complete' else
               f'running: {len(rows)} of {s["rounds"]} round(s) done')
    if rows:
        imp = sorted({c for r in rows for c in r['improved']})
        verdict += '; improved: ' + (', '.join(imp) if imp else 'no cell')
    return {'schema': RESULTS_SCHEMA, 'id': exp_id, 'status': st['status'], 'verdict': verdict,
            'hypothesis': HYPOTHESIS, 'rule': ACQUISITION, 'held_out': st.get('held_out'),
            'baseline': st.get('baseline'), 'rounds': rows, 'stopped': st.get('stopped'), 'warnings': warn,
            'method': 'per cell (class × condition) AP on the held-out set with a 95 % percentile bootstrap interval '
                      'over drives; before and after with a paired bootstrap over drives (core.loop.compare_cells); '
                      'improvement only when the interval of the difference lies wholly above zero'}


def _cell_name(c: dict) -> str:
    return f'{c["class"]} · {c["condition"]}'


def table(res: dict) -> str:
    """The results as text: each round's chosen cells (EV), AP before → after [interval] and the verdicts."""
    out = [f'{res["id"]}: {res["verdict"]}']
    for r in res['rounds']:
        h = r['AP']
        out.append(f'round {r["round"]}: {r["drives"]} drives, {r["frames"]} frames; AP {L._f(h["before"])} → '
                   f'{L._f(h["after"])} (Δ {L._f(h["diff"], True)} [{L._f(h["ci_low"], True)}, '
                   f'{L._f(h["ci_high"], True)}])')
        for c in r['chosen_cells']:
            out.append(f'  {c["cell"]}: EV {c["ev"]:.4f}; AP {L._f(c["before"]["value"])} '
                       f'[{L._f(c["before"]["ci_low"])}, {L._f(c["before"]["ci_high"])}] → '
                       f'{L._f(c["after"]["value"])} [{L._f(c["after"]["ci_low"])}, {L._f(c["after"]["ci_high"])}]'
                       f'; Δ {L._f(c["diff"]["value"], True)} [{L._f(c["diff"]["ci_low"], True)}, '
                       f'{L._f(c["diff"]["ci_high"], True)}] — {c["verdict"]}')
    for w in res.get('warnings') or ():
        out.append(f'warning: {w}')
    return '\n'.join(out)


# ================================================================================================ running
class Runner:
    """Fixes the held-out set and runs rounds with a CarlaService (its queue runs every drive and the training)."""

    def __init__(self, project, svc, exp_id: str, emit=None, cancel=None):
        self.project, self.svc = project, svc
        self.ws = project.workspace
        self.id = exp_id
        self.dir, self.s, self.st = load(self.ws, exp_id)
        self.em, self.cancel = emit, cancel

    def log(self, text, level='info'):
        if self.em is not None:
            self.em.log(text, level)

    def progress(self, f, text=''):
        if self.em is not None:
            self.em.progress(min(max(f, 0.0), 1.0), text)

    def _save(self):
        write_state(self.ws, self.id, self.st)

    # -- the held-out set
    def origin(self) -> tuple:
        m = self.project.item('models', self.s['model'])
        if m is None:
            raise Stopped(f'the starting model {self.s["model"]!r} is no longer in the project')
        try:
            tdir, run = L.training_run_of(m)
        except L.Stopped as e:
            raise Stopped(str(e)) from None
        return m, tdir, run

    def evalset(self) -> dict:
        """The fixed held-out manifest, checked against the hash state.json recorded when it was fixed."""
        ho = self.st.get('held_out')
        if not ho:
            raise GenError('the held-out set is not fixed yet')
        doc = L.read_evalset(self.ws, ho['odd_key'])
        if doc is None or PV.sha256_file(L.absp(self.ws, ho['manifest'])) != ho['sha256']:
            raise GenError('refused: the held-out set changed since it was fixed (a drive re-recorded or its manifest '
                           'edited); the rounds compare only on the set fixed before round 1')
        return doc

    def fix(self):
        """Record the held-out set once (cached drives reused) and evaluate the starting model on it."""
        m, _tdir, run = self.origin()
        if not self.st.get('held_out'):
            odd = held_out_odd(self.s)
            norm, errs = L.normalise_odd(odd, self.project)
            if errs:
                raise GenError('the held-out set: ' + '; '.join(errs))
            self.log(f'fixing the held-out set: {len(self.s["conditions"])} condition(s) × '
                     f'{self.s["held_out"]["drives_per_condition"]} drives')
            doc = L.ensure_evalset(self.project, norm, self.svc, self.cancel,
                                   lambda f, t='': self.progress(0.5 * f, t), self.log)
            key = L.odd_key(norm, self.project)
            p = L.evalset_path(self.ws, key)
            self.st['held_out'] = {'odd_key': key, 'manifest': L.rel(self.ws, p), 'sha256': PV.sha256_file(p),
                                   'drives': len(doc['drives']), 'fixed': PV.utc_now(),
                                   'conditions': {d['cell']: d['cell'].split(' · ', 1)[-1] for d in doc['drives']}}
            self.st['status'] = 'running'
            self._save()
        ev = self.evalset()
        prev = run['provenance']['settings']
        L.assert_no_leak(list(prev.get('train') or []) + list(prev.get('val') or []) + list(prev.get('test') or []),
                         ev, self.ws)
        bl = os.path.join(self.dir, 'baseline')
        if not (self.st.get('baseline') and L._outputs_ok(self.ws, self.st['baseline'].get('outputs'))):
            outs = self._evaluate(m, bl)
            self.st['baseline'] = {'dir': L.rel(self.ws, bl), 'model': m.get('id'), 'outputs': _hash(self.ws, outs)}
            self._save()
        return ev

    def _evaluate(self, m, out_dir) -> list:
        from . import harness as H
        ev = self.evalset()
        rig = self.project.item('rigs', self.s['rig']) or {}
        cam = next((x['id'] for x in rig.get('sensors', []) if x.get('type') == 'camera.pinhole'), None)
        locks = []
        try:
            if m.get('runtime') != 'python':
                from .training import gpu_locks
                locks = gpu_locks(self.em, self.cancel)
            H.evaluate(L.evalset_runs(self.ws, ev), m, out_dir, self.ws, cam, 1, int(self.s['held_out']['n_boot']),
                       int(self.s['held_out']['seed']), cancel=self.cancel)
        except H.HarnessError as e:
            if str(e) == 'cancelled':
                raise Cancelled() from None
            raise GenError(f'evaluation: {e}') from None
        finally:
            for lk in locks:
                lk.release()
        return [os.path.join(out_dir, 'evaluation.json'), os.path.join(out_dir, 'predictions.jsonl')]

    # -- rounds
    def before(self, n: int) -> tuple:
        """(the evaluation folder the round starts from, its outputs' hashes, the model evaluated there)."""
        if n == 1:
            b = self.st['baseline']
            return L.absp(self.ws, b['dir']), b['outputs'], self.project.item('models', self.s['model'])
        prev = next(r for r in self.st['rounds'] if r['n'] == n - 1)
        rd = PV.read_json(os.path.join(L.absp(self.ws, prev['dir']), 'round.json'))
        rec = rd['steps']['evaluate']
        return L.absp(self.ws, rec['result']['dir']), rec['outputs'], rd['steps']['retrain']['result']['registration']

    def run(self, rounds: int | None = None) -> dict:
        """Fix the held-out set if needed, then run (or resume) rounds — at most `rounds` more (None: until the stop
        rule or the definition's rounds). Returns the state."""
        if self.st.get('stopped'):
            raise Stopped(f'the chain stopped: {self.st["stopped"]}')
        if self.st['status'] == 'complete':
            raise Stopped(f'the chain is complete: its {self.s["rounds"]} round(s) ran')
        self.fix()
        done = 0
        while rounds is None or done < int(rounds):
            L._check(self.cancel)
            cur = next((r for r in self.st['rounds'] if r['status'] != 'done'), None)
            n = cur['n'] if cur else len(self.st['rounds']) + 1
            if n > int(self.s['rounds']):
                self.st['status'] = 'complete'
                break
            try:
                self.run_round(n)
            except Stopped as e:
                self.st['stopped'] = str(e)
                self.st['status'] = 'stopped'
                self._finish()
                raise
            done += 1
            if self.st.get('stopped'):
                break
            if n >= int(self.s['rounds']):
                self.st['status'] = 'complete'
                break
        self._finish()
        return self.st

    def _finish(self):
        self._save()
        PV.write_json(os.path.join(self.dir, 'results.json'), results(self.ws, self.id))
        from .ablation import index
        index(self.ws, self.dir)

    def run_round(self, n: int):
        rdir = round_dir(self.ws, self.id, n)
        os.makedirs(rdir, exist_ok=True)
        p = os.path.join(rdir, 'round.json')
        rd = PV.read_json(p) if os.path.isfile(p) else {
            'schema': ROUND_SCHEMA, 'experiment': self.id, 'n': n, 'created': PV.utc_now(),
            'steps': {x: {'state': 'pending'} for x in STEPS}}
        entry = next((r for r in self.st['rounds'] if r['n'] == n), None)
        if entry is None:
            entry = {'n': n, 'dir': L.rel(self.ws, rdir), 'status': 'running'}
            self.st['rounds'].append(entry)
        entry['status'] = 'running'
        self._save()
        _bdir, bout, _bm = self.before(n)
        for k, step in enumerate(STEPS):
            if step_done(self.ws, rd, step, bout):
                continue
            rec = rd['steps'][step]
            rec.update(state='running', started=PV.utc_now(), upstream=_up(rd, step, bout), error=None,
                       runs=int(rec.get('runs') or 0) + 1)
            PV.write_json(p, rd)
            self.log(f'round {n}, step {k + 1} ({STEP_TITLES[step]})')
            self.progress(0.5 + 0.5 * k / len(STEPS), f'round {n}: {STEP_TITLES[step]}')
            t0 = time.monotonic()
            try:
                outs = getattr(self, f'step_{step}')(n, rd, rec, rdir)
            except Stopped as e:
                rec.update(state='stopped', error=str(e), finished=PV.utc_now())
                entry['status'] = 'stopped'
                PV.write_json(p, rd)
                raise
            except Cancelled:
                rec.update(state='cancelled', finished=PV.utc_now())
                entry['status'] = 'cancelled'
                PV.write_json(p, rd)
                self._save()
                raise
            except Exception as e:      # noqa: BLE001 (recorded, then re-raised)
                rec.update(state='failed', error=str(e) if isinstance(e, (GenError, L.LoopError)) else
                           f'{type(e).__name__}: {e}', finished=PV.utc_now())
                entry['status'] = 'failed'
                PV.write_json(p, rd)
                self._save()
                raise
            rec.update(state='done', finished=PV.utc_now(), outputs=_hash(self.ws, outs),
                       wall_s=round(time.monotonic() - t0, 3))
            PV.write_json(p, rd)
        entry['status'] = 'done'
        entry['model'] = (rd['steps']['retrain'].get('result') or {}).get('registration', {}).get('id')
        cells = PV.read_json(os.path.join(rdir, 'cells.json'))
        why = stop_reason(cells['chosen'])
        if why:
            self.st['stopped'] = f'round {n}: {why}'
            self.st['status'] = 'stopped'
        self._track(n, rd, cells)
        self._save()

    def _track(self, n, rd, cells):
        """The round as a tracked run of the experiment (the registry counts them)."""
        from .experiments import Experiment
        tr = Experiment(self.dir).run(f'round-{n:03d}')
        gen = rd['steps']['generate'].get('result') or {}
        tr.start({'round': n, 'cells': [c['cell'] for c in cells['chosen']], 'drives': gen.get('drives'),
                  'frames': gen.get('frames')}, {'kind': KIND}, int(self.s['held_out']['seed']))
        h = cells['headline']
        tr.log_metrics({'AP_before': h['before'], 'AP_after': h['after'], 'AP_diff': h['diff'],
                        'cells_improved': len(cells['improved']), 'drives': gen.get('drives'),
                        'frames': gen.get('frames')}, step=n)
        tr.finish('ok')

    # -- 1
    def step_rank(self, n, rd, rec, rdir):
        from . import harness as H
        bdir, _bout, bm = self.before(n)
        ev = self.evalset()
        frames = H.read_predictions(os.path.join(bdir, 'predictions.jsonl'))
        ed = PV.read_json(os.path.join(bdir, 'evaluation.json'))
        cell_of = {os.path.basename(L.absp(self.ws, d['run'])): d['cell'] for d in ev['drives']}
        cond_of = self.st['held_out']['conditions']
        rows = [dict(r, condition=cond_of.get(r['condition'], r['condition']), label=r['condition'])
                for r in L.cell_table(frames, cell_of, self.s['classes'], [0.0], int(self.s['held_out']['n_boot']),
                                      int(self.s['held_out']['seed'])) if r['condition'] != L.ALL]
        _m, _tdir, run = self.origin()
        srcs = list(run['provenance']['settings'].get('train') or [])
        if n > 1:
            prev = PV.read_json(os.path.join(round_dir(self.ws, self.id, n - 1), 'version.json'))
            srcs.append({'kind': 'version', 'path': L.absp(self.ws, prev['version']), 'split': 'train'})
        counts, notes = train_counts(srcs, self.s['classes'], self.s['conditions'])
        fails = coverage_failures(self.ws, self.s['classes'], self.s['conditions'])
        overall = L.headline(ed)['value']
        ranked = rank_cells(rows, overall, self.s, counts, fails)
        chosen = [r for r in ranked if r.get('rank')]
        nd = round_drives(self.s)
        for c, k in zip(chosen, allocate([c['ev'] for c in chosen], nd)):
            c['drives'] = k
        chosen = [c for c in chosen if c['drives'] > 0]
        doc = {'schema': 'falcon.generation_rank/1', 'n': n, 'rule': ACQUISITION, 'overall_AP': L.headline(ed),
               'reference': self.s['acquisition']['reference'], 'model': (bm or {}).get('id'),
               'rows': ranked, 'chosen': chosen, 'drives': nd, 'notes': notes,
               'provenance': PV.block({'evaluation': os.path.join(bdir, 'evaluation.json')},
                                      int(self.s['held_out']['seed']), {'acquisition': self.s['acquisition']})}
        out = os.path.join(rdir, 'rank.json')
        PV.write_json(out, doc)
        rec['decisions'] = [f'{_cell_name(c)}: EV {c["ev"]:.4f} (gap {c["terms"]["gap"]:.3f}, s '
                            f'{c["terms"]["s"]:.3f}, f {c["terms"]["f"]:.3f}), {c["drives"]} drive(s)' for c in chosen]
        if not chosen:
            raise Stopped('no cell has a positive expected value with enough held-out objects (min_objects '
                          f'{self.s["acquisition"]["min_objects"]}; reference {self.s["acquisition"]["reference"]})'
                          + (': a model uniformly weak has no gap to the overall AP — reference "perfect" ranks '
                             'by the shortfall from 1' if self.s['acquisition']['reference'] == 'overall' else ''))
        return [out]

    # -- 2
    def step_generate(self, n, rd, rec, rdir):
        from . import datasets as D
        from ..core.drive import scenario_config
        from ..sim import engine as E
        g = self.s['generate']
        rank = PV.read_json(os.path.join(rdir, 'rank.json'))
        ev = self.evalset()
        held_keys = {d['key'] for d in ev['drives']}
        held_dirs = {os.path.realpath(p) for p in L.evalset_runs(self.ws, ev)}
        spec = gen_spec(self.s)
        base = os.path.join(E.spec_dir(self.ws, spec['id']), 'drives')
        scns = all_scenarios(self.project)
        budget_frames = int(self.s['budget']['frames'])
        cfgs, skipped, frames, i = [], [], 0, 0
        for ci, c in enumerate(rank['chosen']):
            scn = scenario_for(c['class'], c['condition'], scns) if float(g['scenario_share']) > 0 else None
            n_scn = int(round(c['drives'] * float(g['scenario_share']))) if scn else 0
            plan = []
            for j in range(c['drives']):
                seed = GEN_SEED_BASE + int(g['seed']) * 1_000_000 + n * 1000 + ci * 100 + j
                plan.append(('scenario' if j < n_scn else 'drive', seed))
            for kind, seed in plan:
                cell = {'world': self.s['world'], 'weather': condition_weather(c['condition']), 'scenario': None,
                        'label': c['condition'], 'class': c['class']}
                if kind == 'drive':
                    (cfg, _), = E.drive_configs(self.project, spec, {'drives': [{'cell': cell, 'seed': seed,
                                                                                'index': i}]},
                                                self.svc.port, self.svc.tm_port,
                                                float(self.svc.settings.get('render_budget_mpx')))
                else:
                    cfg = scenario_config(self.project, scn, self.s['world'], self.s['rig'], '/pending',
                                          self.svc.port, self.svc.tm_port, seed=seed, fps=float(g['fps']),
                                          traffic_vehicles=int(g['traffic_vehicles']),
                                          render_budget_mpx=float(self.svc.settings.get('render_budget_mpx')),
                                          workspace=self.ws)
                    cfg['record_every'] = int(g['record_every'])
                    cfg['dataset'] = {'spec': spec['id'], 'index': i, 'cell': dict(cell, scenario=scn['id'])}
                    key = D.drive_key(cfg)
                    cfg['out'], cfg['cache_key'] = os.path.join(base, key[:16]), key
                nf = int(cfg['steps']) // max(int(cfg.get('record_every', 1)), 1)
                if frames + nf > budget_frames:
                    skipped.append(f'{_cell_name(c)}: a {kind} of {nf} frames would pass the frame budget '
                                   f'({budget_frames})')
                    continue
                if cfg['cache_key'] in held_keys or os.path.realpath(cfg['out']) in held_dirs:
                    raise GenError('refused: a generated drive is a held-out drive (same cache key or folder)')
                frames += nf
                cfgs.append((cfg, kind, c, nf, scn['id'] if kind == 'scenario' else None))
                i += 1
        jobs = []
        for cfg, kind, c, _nf, _sid in cfgs:
            try:
                ok = PV.read_json(os.path.join(cfg['out'], 'run.json')).get('status') == 'ok'
            except (OSError, ValueError):
                ok = False
            if not ok:
                jobs.append(self.svc.submit_drive(cfg, title=f'Generation {self.id} r{n}: {kind} for '
                                                             f'{_cell_name(c)}'))
        self.log(f'round {n}: {len(cfgs)} drive(s), {len(cfgs) - len(jobs)} already recorded')
        L._wait_jobs(self.svc, jobs, self.cancel, lambda f, t='': self.progress(f, t), 'generated drives')
        drives, objects = [], {}
        for cfg, kind, c, nf, sid in cfgs:
            cov = D.coverage_of_run(cfg['out'])
            got = int((cov.get('objects') or {}).get(c['class'], 0))
            objects[_cell_name(c)] = objects.get(_cell_name(c), 0) + got
            drives.append({'run': L.rel(self.ws, cfg['out']), 'key': cfg['cache_key'], 'kind': kind,
                           'scenario': sid, 'cell': _cell_name(c), 'seed': cfg['seed'], 'frames': cov['frames'],
                           'objects_of_class': got})
        doc = {'schema': 'falcon.generation_generate/1', 'n': n, 'spec': spec['id'], 'drives': drives,
               'frames': sum(d['frames'] for d in drives), 'planned_frames': frames, 'objects': objects,
               'skipped': skipped, 'budget': self.s['budget'],
               'labels': 'from the ground truth of each recorded frame (the rig\'s boxes2d), no annotation',
               'provenance': PV.block({'rank': os.path.join(rdir, 'rank.json')}, int(g['seed']), {'generate': g})}
        out = os.path.join(rdir, 'generate.json')
        PV.write_json(out, doc)
        rec['result'] = {'drives': len(drives), 'frames': doc['frames'], 'scenario_runs':
                         sum(d['kind'] == 'scenario' for d in drives)}
        rec['decisions'] = [f'{k}: {v} object(s) of its class labelled' for k, v in objects.items()] + skipped
        if not drives:
            raise Stopped('the budget allowed no drive this round (' + '; '.join(skipped[:2]) + ')')
        return [out] + [os.path.join(L.absp(self.ws, d['run']), 'run.json') for d in drives]

    # -- 3
    def step_version(self, n, rd, rec, rdir):
        from ..sim import engine as E
        doc = E.new_version(self.ws, gen_spec(self.s), 'drive', {'train': 1.0}, int(self.s['generate']['seed']))
        vpath = os.path.join(E.spec_dir(self.ws, doc['spec']), 'versions', f'v{doc["version"]:03d}.json')
        L.assert_no_leak([{'kind': 'version', 'path': vpath, 'split': 'train'}], self.evalset(), self.ws)
        out = os.path.join(rdir, 'version.json')
        PV.write_json(out, {'version': L.rel(self.ws, vpath), 'number': doc['version'], 'runs': len(doc['runs']),
                            'sha256': doc['sha256']})
        rec['decisions'] = [f'dataset version gen-{self.id} v{doc["version"]}: {len(doc["runs"])} generated drive(s) '
                            'so far, all to train; none is a held-out drive']
        return [out, vpath]

    # -- 4
    def step_retrain(self, n, rd, rec, rdir):
        from . import training as T
        from .jobs import FINAL
        m, tdir, run = self.origin()
        prev = copy.deepcopy(run['provenance']['settings'])
        ver = PV.read_json(os.path.join(rdir, 'version.json'))
        new = {'kind': 'version', 'path': L.absp(self.ws, ver['version']), 'split': 'train'}
        L.assert_no_leak(list(prev.get('train') or []) + list(prev.get('val') or []) + list(prev.get('test') or [])
                         + [new], self.evalset(), self.ws)
        cfg = {k: v for k, v in prev.items() if k != 'id'}
        cfg['train'] = list(prev.get('train') or []) + [new]
        cfg['export'] = True
        if self.s.get('epochs'):
            cfg['epochs'] = int(self.s['epochs'])
        cfg_path = os.path.join(rdir, 'retrain-config.json')
        PV.write_json(cfg_path, cfg)
        name = re.sub(r'[^A-Za-z0-9_.-]', '-', f'gen-{self.id}-r{n:03d}')[:60]
        out = os.path.join(self.ws, 'training', name)
        job = self.svc.queue.submit(T.job_spec(['--config', cfg_path, '--out', out, '--workspace', self.ws],
                                               f'Generation {self.id} r{n}: retrain {m.get("name")}'))
        try:
            while job.state not in FINAL:
                self.svc.queue.poll()
                L._check(self.cancel)
                self.progress(job.progress, job.text)
                time.sleep(0.2)
        except (Cancelled, KeyboardInterrupt):
            self.svc.queue.cancel(job)
            t0 = time.monotonic()
            while job.state not in FINAL and time.monotonic() - t0 < 60:
                self.svc.queue.poll()
                time.sleep(0.1)
            raise Cancelled() from None
        if job.state != 'done':
            try:
                err = PV.read_json(os.path.join(out, 'run.json')).get('error') or ''
            except (OSError, ValueError):
                err = '; '.join(t for _lvl, t in job.log[-5:] if t)
            raise GenError(f'the training {job.state}: {err}')
        reg = PV.read_json(os.path.join(out, 'model.json'))
        reg['name'] = f'{m.get("name")} + generation {self.id} r{n}'
        rec['result'] = {'registration': reg, 'training': L.rel(self.ws, out)}
        rec['decisions'] = [f'the starting model\'s settings ({prev.get("arch")}, init {prev.get("init")}, '
                            f'{cfg.get("epochs")} epochs, seed {prev.get("seed")}) on its '
                            f'{len(prev.get("train") or [])} source(s) + dataset version v{ver["number"]} ({ver["runs"]} generated drives); '
                            'checked: no source is or holds a held-out drive']
        return [cfg_path, os.path.join(out, 'run.json'), os.path.join(out, 'model.json')]

    # -- 5
    def step_evaluate(self, n, rd, rec, rdir):
        reg = rd['steps']['retrain']['result']['registration']
        out = os.path.join(rdir, 'after')
        outs = self._evaluate(reg, out)
        rec['result'] = {'dir': L.rel(self.ws, out), 'AP': L.headline(PV.read_json(outs[0]))}
        return outs

    # -- 6
    def step_compare(self, n, rd, rec, rdir):
        from . import harness as H
        bdir, _bout, _bm = self.before(n)
        adir = os.path.join(rdir, 'after')
        fa = H.read_predictions(os.path.join(bdir, 'predictions.jsonl'))
        fb = H.read_predictions(os.path.join(adir, 'predictions.jsonl'))
        ev = self.evalset()
        cell_of = {os.path.basename(L.absp(self.ws, d['run'])): d['cell'] for d in ev['drives']}
        nb, seed = int(self.s['held_out']['n_boot']), int(self.s['held_out']['seed'])
        rank = PV.read_json(os.path.join(rdir, 'rank.json'))
        cmp = H.compare(fa, fb, 'detector', nb, seed)
        hd = next((r for r in cmp['rows'] if r['metric'] == 'AP'), {})
        ea = L.headline(PV.read_json(os.path.join(bdir, 'evaluation.json')))
        eb = L.headline(PV.read_json(os.path.join(adir, 'evaluation.json')))
        after_rows = L.cell_table(fb, cell_of, self.s['classes'], [0.0], nb, seed)

        def back(r):                   # rank rows carry the condition id; compare_cells wants the held-out label
            return dict(r, condition=r['label'])
        chosen = L.compare_cells(fa, fb, cell_of, [back(r) for r in rank['chosen']], after_rows, [0.0], nb, seed)
        others_in = [back(r) for r in rank['rows'] if not r.get('rank') and
                     r['n_objects'] >= int(self.s['acquisition']['min_objects'])]
        others = L.compare_cells(fa, fb, cell_of, others_in, after_rows, [0.0], nb, seed) if others_in else []
        cond_of = self.st['held_out']['conditions']
        for c, r in zip(chosen, rank['chosen']):
            c.update(cell=_cell_name(r), condition=r['condition'], ev=r['ev'], verdict=L.verdict(c['diff']))
        for c in others:
            c['condition'] = cond_of.get(c['condition'], c['condition'])
            c.update(cell=f'{c["class"]} · {c["condition"]}', verdict=L.verdict(c['diff']))
        headline = {'before': ea['value'], 'after': eb['value'], 'diff': hd.get('diff'), 'ci_low': hd.get('ci_low'),
                    'ci_high': hd.get('ci_high'), 'verdict': L.verdict(hd), 'unit': cmp.get('unit'),
                    'n_units': cmp.get('n_units')}
        improved = [c['cell'] for c in chosen if c['verdict'] == 'improvement']
        doc = {'schema': 'falcon.generation_cells/1', 'n': n, 'headline': headline, 'chosen': chosen,
               'others': others, 'improved': improved,
               'regressed': [c['cell'] for c in chosen + others if c['verdict'] == 'regression'],
               'stop': stop_reason(chosen),
               'method': 'the same held-out frames before and after; per cell a 95 % percentile bootstrap interval '
                         'over drives, the difference by a paired bootstrap over drives',
               'provenance': PV.block({'before': os.path.join(bdir, 'evaluation.json'),
                                       'after': os.path.join(adir, 'evaluation.json')}, seed, {'n_boot': nb})}
        out = os.path.join(rdir, 'cells.json')
        PV.write_json(out, doc)
        rec['decisions'] = [f'{c["cell"]}: AP {L._f(c["before"]["value"])} → {L._f(c["after"]["value"])}, Δ '
                            f'{L._f(c["diff"]["value"], True)} [{L._f(c["diff"]["ci_low"], True)}, '
                            f'{L._f(c["diff"]["ci_high"], True)}] — {c["verdict"]}' for c in chosen]
        return [out]


# ================================================================================================ the command
def _spec_from_args(a, project) -> dict:
    s = PV.read_json(a.spec) if a.spec else example_spec(project)
    s['id'] = a.id or s.get('id')
    for k in ('model', 'world', 'rig'):
        if getattr(a, k):
            s[k] = getattr(a, k)
    if a.conditions:
        s['conditions'] = [c.strip() for c in a.conditions.split(',') if c.strip()]
    if a.rounds:
        s['rounds'] = a.rounds
    b = dict(s.get('budget') or DEFAULTS['budget'])
    if a.budget_drives:
        b['drives'] = a.budget_drives
    if a.budget_frames:
        b['frames'] = a.budget_frames
    s['budget'] = b
    if a.max_cells:
        s['acquisition'] = dict(s.get('acquisition') or DEFAULTS['acquisition'], max_cells=a.max_cells)
    return s


def main_generate(argv) -> int:
    """`falcon generate plan|new|run|show|list PROJECT …` (docs/generation.md): plan prints the definition and the
    rule; new registers it; run fixes the held-out set and runs rounds (protocol lines; 0 ok — also when a round's
    comparison stopped the chain, said in the result —, 2 failed, 3 cancelled, 6 refused: a chain already stopped or
    complete, no cell to rank, a model Falcon cannot retrain); show prints the results; list the generation
    experiments."""
    import argparse
    import json
    import signal
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon generate', description='generation rounds (docs/generation.md)')
    ap.add_argument('action', choices=('plan', 'new', 'run', 'show', 'list'))
    ap.add_argument('project')
    ap.add_argument('--id', default='')
    ap.add_argument('--spec', default='', help='plan/new: a definition JSON file (the rest from the flags)')
    ap.add_argument('--model', default='')
    ap.add_argument('--world', default='')
    ap.add_argument('--rig', default='')
    ap.add_argument('--conditions', default='', help='weather/light,… e.g. clear/day,rain/night')
    ap.add_argument('--rounds', type=int, default=0, help='plan/new: the most rounds; run: at most this many more')
    ap.add_argument('--budget-drives', type=int, default=0)
    ap.add_argument('--budget-frames', type=int, default=0)
    ap.add_argument('--max-cells', type=int, default=0)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--fake-carla', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    from .project import Project, ProjectError
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    ws = proj.workspace
    try:
        if a.action == 'plan':
            s = normalise(_spec_from_args(a, proj), proj)
            print(json.dumps(s, indent=2) if a.json else describe_plan(s))
            return P.OK
        if a.action == 'new':
            c = create(ws, _spec_from_args(a, proj), proj)
            print(f'registered {c["id"]} in {c["path"]}\n' + describe_plan(c['spec']))
            return P.OK
        if a.action == 'list':
            from .ablation import registry
            rows = [r for r in registry(ws) if r['kind'] == KIND]
            for r in rows:
                print(f'{r["id"]}: {r["status"]} — {r.get("verdict") or "no round yet"}')
            if not rows:
                print('no generation experiment')
            return P.OK
        if a.action == 'show':
            res = results(ws, a.id)
            print(json.dumps(res, indent=2, ensure_ascii=False) if a.json else table(res))
            return P.OK
    except GenError as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    # run
    from .jobs import JobQueue, SubprocessLauncher
    from .settings import Settings
    from ..sim.service import CarlaService
    em = P.Emitter()
    em.hello('generate')
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
    svc = CarlaService(Settings(), JobQueue(SubprocessLauncher()), ws, fake_carla=a.fake_carla)
    code = P.OK
    try:
        r = Runner(proj, svc, a.id, em, lambda: state['cancel'])
        st = r.run(a.rounds or None)
        em.result('ok', experiment=a.id, state=st['status'], rounds=len(st['rounds']), stopped=st.get('stopped'))
        if st.get('stopped'):
            print(f'stopped: {st["stopped"]}', flush=True)
    except Stopped as e:
        print(f'stopped: {e}', flush=True)
        em.result('stopped', error=str(e))
        code = P.REFUSED
    except Cancelled:
        em.result('cancelled')
        code = P.CANCELLED
    except (GenError, L.LoopError, OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        code = P.FAILED
    finally:
        state['cancel'] = True
        svc.shutdown()
        signal.signal(signal.SIGTERM, old[0])
        signal.signal(signal.SIGINT, old[1])
    return code
