"""The closed loop (spec section O; the Loop page and `falcon loop`): pick a model and a target operational design
domain (ODD), evaluate it, find its weak cells, search for rare cases there, generate targeted data, retrain, evaluate
again and show the before and after with intervals — each iteration a versioned, reproducible record, each step's cost
shown before it runs (standard library + numpy at import; the tools it orchestrates are imported inside functions).

The loop does not reimplement any tool. Its steps run the data engine (core.datasets plans, sim.engine drive configs,
drives through sim.service.CarlaService's job queue), the test harness (core.harness.evaluate, compare and core.metrics'
bootstraps), the rare-case search (sim.studio.Study with core.search's methods) and the training lab (`falcon train`
as a job, exactly the Training page's).

The ODD (a JSON object, the project's loop.odd; ODD_SCHEMA, docs/loop.md):

    {"schema": "falcon.odd/1", "id", "name",
     "rig": rig id (its first pinhole camera is evaluated; it must record the boxes2d label),
     "worlds": [world ids], "traffic_profile": profile id,
     "weathers": [{"name": "monsoon dusk", "preset": "MidRainSunset", <CARLA weather fields>…,
                   "ranges": {field: [lo, hi]}}],      ranges: each drive draws the field uniformly (seeded)
     "classes": [Falcon classes of interest], "range_bins": [0, 20, 40, 60, 80]   (metres; the last bin is open),
     "evaluation": {"drives_per_cell", "drive_s", "fps", "record_every", "seed", "every", "n_boot"},
     "weak_cells": {"min_objects", "max_cells", "fallback": "none" | "lowest"},
     "search": {"method", "budget", "batch", "threshold", "fps", "traffic_vehicles", "seed"},
     "generate": {"objects_per_cell", "drive_s", "fps", "record_every", "max_drives", "seed", "reruns_per_failure",
                  "max_failures"},
     "retrain": {"epochs": null (the model's own) | n}}

A cell (a condition) is one world × one weather entry of the ODD; the evaluation set has drives_per_cell drives in
each, at fixed seeds (EVAL_SEED_BASE + seed × 10000 + i), recorded once per ODD (its key hashes the ODD's worlds,
weathers, traffic profile, rig and drive settings as the project defines them) under
datasets/loop-eval-<key[:12]>/drives/ and listed in loop/evalsets/<id>.json. It is **never trained on**: the retrain
step refuses any training, validation or test source that is, contains or lists a drive of it (assert_no_leak), and
the generate step refuses a drive whose cache key is an evaluation drive's.

The steps (STEPS) and their rules:
1. pick — freeze the model (its registration and file hash) and the ODD (normalised, hashed) into pick.json.
2. evaluate — record the evaluation set if needed, then core.harness.evaluate on it (drive-level bootstrap). An
   iteration that follows a finished one whose step 7 evaluated this very model (same file hash) on the same set
   reuses that evaluation instead of repeating it.
3. weak cells — from the evaluation's predictions, COCO AP per class × range bin × condition (each condition, and
   all conditions together, "*"), with a percentile bootstrap interval over drives (core.metrics.bootstrap). A cell
   is WEAK when it holds at least min_objects scored objects and the upper end of its interval is below the overall
   AP of the evaluation; weak cells are ranked by that upper end, lowest first (the surest shortfall first), and the
   first max_cells of distinct class × range bin are chosen. With no weak cell (a model uniformly weak or strong)
   nothing is targeted — unless the ODD's weak_cells.fallback is "lowest": then the cells with the lowest upper ends
   are, each marked as a fallback. HEURISTIC: per-class AP is compared with the AP over all classes, and the interval
   rests on few drives when the set is small (harness.MIN_DRIVES).
4. search — per chosen cell, a rare-case search (sim.studio.Study, core.search) on a built-in scenario matched to the
   cell's class (SCENARIOS_FOR_CLASS; the first whose runs succeed) in the cell's world and weather (for "*" the
   condition with the lowest AP for that class and bin). The search's rig is the ODD's without label cameras (the
   criticality needs none, and it renders a third of the pixels). THE OBJECTIVE IS THE SCENARIO'S CRITICALITY
   (core.criticality; 1 for a collision), NOT THE MODEL'S DETECTIONS: it finds where driving is hard in the weak
   cell's conditions, not where the model fails. The range bin is not mapped onto scenario parameters.
5. generate — (a) a dataset spec targeting the chosen cells (their classes, range bins and weathers): the drives
   needed come from the yield of the cell's class in its range bin measured on the evaluation set (objects per
   recorded frame), planned with core.datasets.plan and run through sim.engine (cache-keyed, resumable); (b) the
   search's failing cases (criticality ≥ threshold; with none, the most critical cases, said so) re-run with the
   labelled rig: the same seed (the same case, now labelled) and reruns_per_failure − 1 further seeds (the director
   places the scenario at another start).
6. retrain — `falcon train` with the model's own training settings (its training run.json) on its original sources
   plus the new drives; a model Falcon did not train (a plug-in, an imported file) cannot be retrained: the step
   stops, saying why. The retrained model is exported and registered (its model.json).
7. evaluate again — the retrained model on the same evaluation set.
8. compare — core.harness.compare (a paired bootstrap over drives) of the two evaluations, and the same paired
   bootstrap for each chosen weak cell; the before and after with their intervals.

Costs. Before a step runs its estimate is shown and recorded (estimate()), and its wall time is measured after:
drives from core.budget.Throughput (this machine's measured throughput, or the reference machine's labelled
typical); training from the images/s and peak GPU memory measured by the model's own or another training of the same
architecture on the same device (run.json timings), else the reference machine's numbers in docs/training.md labelled
typical (CUDA only); evaluation from a measured earlier evaluation of the same model file; a step with nothing measured
to go on says "not estimated" and why — never a made-up figure.

Records. loop/<NNN>/iteration.json (ITERATION_SCHEMA): the model, the ODD and its hash, the evaluation set, per step
its state, estimate, measured wall time, outputs (paths relative to the workspace) with their SHA-256, the hash of
the previous step's outputs it ran on, its decisions and notes; the iteration's summary and provenance. A step whose
outputs exist, hash-match and were made from the current upstream is not run again (resume); the tools it runs keep
their own caches (drives by cache key, studies by results.jsonl, trainings by checkpoint), so a cancelled step resumes
inside. The project's loop.iterations holds one summary entry per iteration (sync_project).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import time

from . import provenance as PV

ODD_SCHEMA = 'falcon.odd/1'
ITERATION_SCHEMA = 'falcon.loop_iteration/1'
EVALSET_SCHEMA = 'falcon.loop_evalset/1'
STEPS = ('pick', 'evaluate', 'weak_cells', 'search', 'generate', 'retrain', 'reevaluate', 'compare')
STEP_TITLES = {'pick': 'Pick the model and the ODD', 'evaluate': 'Evaluate', 'weak_cells': 'Find the weak cells',
               'search': 'Search for rare cases', 'generate': 'Generate targeted data', 'retrain': 'Retrain',
               'reevaluate': 'Evaluate again', 'compare': 'Before and after'}
EVAL_SEED_BASE = 900_000_000
ALL = '*'                                  # the condition "all cells together"
OBJECTIVE_NOTE = ('the search maximises the scenario\'s criticality (core.criticality; 1 for a collision), not the '
                  'model\'s detections: it finds where driving is hard in the weak cell\'s conditions, not where the '
                  'model fails')
WEAK_RULE = ('heuristic: a cell (class × range bin × condition) with at least min_objects scored objects whose AP '
             'interval (95 %, bootstrap over drives) lies wholly below the evaluation\'s overall AP; ranked by the '
             'interval\'s upper end, lowest first; one cell per class × range bin')

ODD_DEFAULTS = {
    'evaluation': {'drives_per_cell': 3, 'drive_s': 20.0, 'fps': 20.0, 'record_every': 5, 'seed': 1, 'every': 1,
                   'n_boot': 1000},
    'weak_cells': {'min_objects': 30, 'max_cells': 3, 'fallback': 'none'},
    'search': {'method': 'bayes', 'budget': 16, 'batch': 4, 'threshold': 0.5, 'fps': 20.0, 'traffic_vehicles': 0,
               'seed': 0},
    'generate': {'objects_per_cell': 300, 'drive_s': 30.0, 'fps': 20.0, 'record_every': 5, 'max_drives': 20,
                 'seed': 2, 'reruns_per_failure': 1, 'max_failures': 4},
    'retrain': {'epochs': None},
}
DEFAULT_CLASSES = ['pedestrian', 'rider', 'motorcycle', 'bicycle', 'car', 'bus', 'truck']
DEFAULT_RANGE_BINS = [0.0, 20.0, 40.0, 60.0, 80.0]
SPEC_FAR_M = 1000.0                        # the open last range bin's far edge in a dataset spec (finite for JSON)

# Weak cell's class -> built-in scenarios in preference order, with parameters held fixed ({} for none). The
# scenario puts an actor of that class in a critical place; its search then finds the critical parameters.
SCENARIOS_FOR_CLASS = {
    'pedestrian': [('jaywalker_behind_bus', {}), ('pedestrian_crossing', {})],
    'rider': [('two_wheeler_filtering', {}), ('cyclist_crossing', {})],
    'motorcycle': [('two_wheeler_filtering', {}), ('wrong_way_scooter', {})],
    'bicycle': [('cyclist_crossing', {})],
    'car': [('cut_in', {}), ('lead_brake', {}), ('door_opening', {})],
    'truck': [('stalled_vehicle', {'kind': 'truck'})],
    'bus': [('bus_pulling_out', {}), ('jaywalker_behind_bus', {})],
}

# The reference machine's measured training throughput and peak GPU memory (docs/training.md: RTX 5080, COCO
# fine-tuning at 640 × 384, batch 8, AMP, epoch 2) — used only as "typical", and only for CUDA.
TYPICAL_TRAINING = {
    'fasterrcnn_mobilenet_v3_large_fpn': {'images_per_s': 126.0, 'peak_gpu_mib': 1123.0},
    'fcos_resnet50_fpn': {'images_per_s': 103.0, 'peak_gpu_mib': 2757.0},
    'retinanet_resnet50_fpn_v2': {'images_per_s': 97.0, 'peak_gpu_mib': 2388.0},
}
TYPICAL_TRAINING_SOURCE = ('typical: the reference machine (RTX 5080, 16 GB), COCO fine-tuning at 640 × 384, batch 8, '
                           'AMP, epoch 2 (docs/training.md); not measured on this machine')


class LoopError(Exception):
    """A step failed (bad input or a tool failed); the record says why. Running again retries it."""


class Stopped(Exception):
    """A step cannot be done as asked (for example retraining a model Falcon did not train); the iteration stops."""


class Cancelled(Exception):
    pass


# ================================================================================================ the ODD
def _merge(defaults: dict, given: dict | None) -> dict:
    out = copy.deepcopy(defaults)
    for k, v in (given or {}).items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def normalise_odd(odd: dict | None, project) -> tuple:
    """(the ODD with defaults filled in, [problems]) — problems say what is missing or wrong, in words."""
    from . import weather as W
    from .groundtruth import CLASS_ID
    errs = []
    if not isinstance(odd, dict):
        return None, ['no ODD is set: choose or write one (an example: the Bengaluru monsoon-dusk ODD)']
    o = _merge({'schema': ODD_SCHEMA, 'id': '', 'name': '', 'rig': '', 'worlds': [], 'traffic_profile': '',
                'weathers': [], 'classes': list(DEFAULT_CLASSES), 'range_bins': list(DEFAULT_RANGE_BINS),
                **copy.deepcopy(ODD_DEFAULTS)}, odd)
    if not re.match(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$', str(o['id'] or '')):
        errs.append('id: letters, digits, "_", "." or "-", at most 41 characters')
    if not o['name']:
        o['name'] = o['id']
    if not o['worlds']:
        errs.append('worlds: at least one world')
    for w in o['worlds']:
        if project.item('worlds', w) is None:
            errs.append(f'worlds: no world {w!r} in the project')
    rig = project.item('rigs', o['rig'])
    if rig is None:
        errs.append(f'rig: no rig {o["rig"]!r} in the project')
    else:
        cam = next((s for s in rig.get('sensors', []) if s.get('type') == 'camera.pinhole'), None)
        if cam is None:
            errs.append(f'rig {rig["name"]!r} has no pinhole camera to evaluate')
        elif 'boxes2d' not in (cam.get('labels') or []):
            errs.append(f'rig {rig["name"]!r}: its camera {cam["id"]!r} must record the boxes2d label (the harness '
                        'and the trainer score and learn from its objects)')
    if o['traffic_profile'] and project.item('traffic_profiles', o['traffic_profile']) is None:
        errs.append(f'traffic_profile: no profile {o["traffic_profile"]!r} in the project')
    if not o['weathers']:
        errs.append('weathers: at least one weather')
    names = set()
    for i, we in enumerate(o['weathers']):
        if isinstance(we, str):
            we = o['weathers'][i] = {'name': we, 'preset': we}
        if not we.get('name'):
            errs.append(f'weathers[{i}]: a name is needed')
        elif we['name'] in names:
            errs.append(f'weathers[{i}]: the name {we["name"]!r} is used twice')
        names.add(we.get('name'))
        if we.get('preset', W.DEFAULT_PRESET) not in W.PRESETS:
            errs.append(f'weathers[{i}]: unknown preset {we.get("preset")!r}')
        for k in we:
            if k not in W.FIELDS and k not in ('name', 'preset', 'ranges'):
                errs.append(f'weathers[{i}]: unknown field {k!r}')
        for k, r in (we.get('ranges') or {}).items():
            if k not in W.FIELDS or not (isinstance(r, (list, tuple)) and len(r) == 2 and float(r[0]) <= float(r[1])):
                errs.append(f'weathers[{i}].ranges.{k}: a CARLA weather field and [lo, hi] with lo ≤ hi')
    bad = [c for c in o['classes'] if c not in CLASS_ID]
    if bad or not o['classes']:
        errs.append(f'classes: not Falcon classes: {", ".join(map(str, bad))}' if bad else 'classes: at least one')
    rb = [float(x) for x in o['range_bins']]
    if len(rb) < 1 or rb[0] != 0.0 or any(b <= a for a, b in zip(rb, rb[1:])):
        errs.append('range_bins: increasing edges in metres starting at 0 (the last bin is open-ended)')
    o['range_bins'] = rb
    ev, wc, se, ge = o['evaluation'], o['weak_cells'], o['search'], o['generate']
    for sect, keys in ((ev, ('drives_per_cell', 'record_every', 'every', 'n_boot')), (wc, ('min_objects', 'max_cells')),
                       (se, ('budget', 'batch')), (ge, ('max_drives', 'reruns_per_failure', 'max_failures'))):
        for k in keys:
            if int(sect[k]) < 1:
                errs.append(f'{k}: at least 1')
    if wc.get('fallback') not in ('none', 'lowest'):
        errs.append('weak_cells.fallback: none or lowest')
    if se['method'] not in ('bayes', 'cmaes', 'adaptive', 'random'):
        errs.append('search.method: bayes, cmaes, adaptive or random')
    return o, errs


def odd_key(odd: dict, project) -> str:
    """What the evaluation set depends on: the worlds, weathers, traffic profile and rig as the project defines them,
    and the evaluation's drive settings. The same key = the same evaluation set."""
    ev = odd['evaluation']
    return PV.sha256_json({
        'worlds': [project.item('worlds', w) for w in odd['worlds']], 'weathers': odd['weathers'],
        'traffic': project.item('traffic_profiles', odd['traffic_profile']) if odd['traffic_profile'] else None,
        'rig': project.item('rigs', odd['rig']),
        'drives': {k: ev[k] for k in ('drives_per_cell', 'drive_s', 'fps', 'record_every', 'seed')}})


def example_odd(project) -> tuple:
    """The Bengaluru, Indian mixed traffic, monsoon dusk ODD for this project: (odd, [what the project still needs])."""
    missing = []
    osm = [w for w in project.doc.get('worlds', []) if w.get('kind') == 'osm']
    blr = [w for w in osm if re.search(r'bengal|bangal|blr', (w.get('name', '') + ' ' + w['id']).lower())] or osm
    if not blr:
        missing.append('an imported Bengaluru world: Worlds ▸ Import a place (OpenStreetMap), e.g. central Bengaluru')
    if project.item('traffic_profiles', 'tp-india-mixed') is None:
        missing.append('the Indian mixed-traffic profile: Worlds ▸ Add the Indian mixed-traffic profile')
    rig = next((r for r in project.doc.get('rigs', []) for s in r.get('sensors', [])
                if s.get('type') == 'camera.pinhole' and 'boxes2d' in (s.get('labels') or [])), None)
    if rig is None:
        missing.append('a rig whose pinhole camera records the boxes2d label (Rigs page)')
    odd = {
        'schema': ODD_SCHEMA, 'id': 'blr-monsoon-dusk', 'name': 'Bengaluru, Indian mixed traffic, monsoon dusk',
        'rig': rig['id'] if rig else 'rig-default', 'worlds': [w['id'] for w in blr[:1]] or ['world-bengaluru'],
        'traffic_profile': 'tp-india-mixed',
        'weathers': [
            {'name': 'monsoon dusk', 'preset': 'MidRainSunset', 'wetness': 80.0,
             'ranges': {'sun_altitude_angle': [-2.0, 6.0], 'precipitation': [50.0, 90.0],
                        'precipitation_deposits': [40.0, 80.0]}},
            {'name': 'monsoon dusk, downpour', 'preset': 'HardRainSunset', 'sun_altitude_angle': 2.0,
             'fog_density': 10.0, 'wetness': 100.0},
        ],
        'classes': list(DEFAULT_CLASSES), 'range_bins': list(DEFAULT_RANGE_BINS),
    }
    return _merge(ODD_DEFAULTS, odd), missing


def odd_cells(odd: dict) -> list:
    """[{'world', 'weather' (the ODD's entry), 'label'}] — world × weather entry."""
    return [{'world': w, 'weather': copy.deepcopy(we), 'label': f'{w} · {we["name"]}'}
            for w in odd['worlds'] for we in odd['weathers']]


def sample_weather(entry: dict, seed: int) -> dict:
    """A concrete weather of an ODD weather entry: its preset and fixed fields, and each ranged field drawn
    uniformly from its range with a generator seeded by (seed, field)."""
    from . import weather as W
    out = {k: v for k, v in entry.items() if k in W.FIELDS or k in ('preset', 'name')}
    for k, (lo, hi) in sorted((entry.get('ranges') or {}).items()):
        out[k] = round(random.Random(f'{seed}:{k}').uniform(float(lo), float(hi)), 3)
    return out


# ================================================================================================ files
def loop_dir(ws: str, *parts) -> str:
    return os.path.join(ws, 'loop', *parts)


def iteration_dir(ws: str, n: int) -> str:
    return loop_dir(ws, f'{int(n):03d}')


def rel(ws: str, path: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(ws))


def absp(ws: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ws, path)


def iterations(ws: str) -> list:
    """Every iteration record of the workspace, by number."""
    d = loop_dir(ws)
    out = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name, 'iteration.json')
            if name.isdigit() and os.path.isfile(p):
                try:
                    out.append(PV.read_json(p))
                except (OSError, ValueError):
                    continue
    return out


def load_iteration(ws: str, n: int) -> dict:
    return PV.read_json(os.path.join(iteration_dir(ws, n), 'iteration.json'))


def save_iteration(ws: str, it: dict):
    it['updated'] = PV.utc_now()
    PV.write_json(os.path.join(iteration_dir(ws, it['n']), 'iteration.json'), it)


def _hash_outputs(ws: str, paths) -> dict:
    out = {}
    for p in paths:
        a = absp(ws, p)
        out[rel(ws, a)] = PV.sha256_file(a)
    return out


def _outputs_ok(ws: str, outputs: dict) -> bool:
    for r, h in (outputs or {}).items():
        p = absp(ws, r)
        try:
            if not os.path.isfile(p) or PV.sha256_file(p) != h:
                return False
        except OSError:
            return False
    return True


def _upstream(it: dict, step: str) -> str | None:
    """The hash of every earlier step's outputs: a step is current only while all it was made from is."""
    i = STEPS.index(step)
    if i == 0:
        return None
    return PV.sha256_json([it['steps'][s].get('outputs') or {} for s in STEPS[:i]])


def step_done(ws: str, it: dict, step: str) -> bool:
    """Done, its outputs present with the recorded hashes, and made from the current upstream outputs."""
    rec = it['steps'][step]
    return rec.get('state') == 'done' and _outputs_ok(ws, rec.get('outputs')) and \
        rec.get('upstream') == _upstream(it, step)


def next_step(ws: str, it: dict | None) -> str | None:
    if it is None:
        return 'pick'
    for s in STEPS:
        if not step_done(ws, it, s):
            return s
    return None


# ================================================================================================ models
def model_file(m: dict, ws: str) -> str | None:
    if m.get('runtime') == 'python' or not m.get('path'):
        return None
    from .models import resolve_path
    return resolve_path(m, ws)


def model_fingerprint(m: dict, ws: str) -> str:
    """The model file's SHA-256, or for a plug-in the hash of its registration (its code is the package's)."""
    f = model_file(m, ws)
    if f and os.path.isfile(f):
        return PV.sha256_file(f)
    keep = {k: m.get(k) for k in ('runtime', 'entry', 'kind', 'classes', 'class_map', 'options')}
    return 'registration:' + PV.sha256_json(keep)


def training_run_of(m: dict) -> tuple:
    """(training folder, its run.json) when Falcon trained the model; else raises Stopped with the reason."""
    if m.get('runtime') == 'python':
        what = ('the synthetic detector is not a model (ground truth with planted weaknesses)'
                if 'synthetic' in str(m.get('entry', '')) else f'{m.get("name")} is a plug-in')
        raise Stopped(f'{what}: Falcon cannot retrain it. Register a detector trained by Falcon (Training ▸ '
                      'Detectors, then Register as a model) and start a new iteration with it.')
    d = (m.get('source') or {}).get('training_run')
    rj = os.path.join(d, 'run.json') if d else ''
    if not d or not os.path.isfile(rj):
        raise Stopped(f'{m.get("name")} was not trained by Falcon (no training run.json beside its registration): '
                      'Falcon cannot retrain an imported model — its training code and data are not here.')
    run = PV.read_json(rj)
    if run.get('schema') != 'falcon.training/1' or run.get('status') != 'ok' or \
            not (run.get('provenance') or {}).get('settings'):
        raise Stopped(f'{rj} is not a finished Falcon training with its settings: cannot continue from it')
    return d, run


# ================================================================================================ service helpers
def _check(cancel):
    if cancel is not None and cancel():
        raise Cancelled()


def _wait_jobs(svc, jobs: list, cancel=None, progress=None, what='drives'):
    """Poll the queue until the jobs end; a cancel cancels them (each worker tidies up) and raises Cancelled."""
    from .jobs import DONE, FINAL
    try:
        while any(j.state not in FINAL for j in jobs):
            svc.queue.poll()
            if cancel is not None and cancel():
                raise Cancelled()
            if progress and jobs:
                done = sum(1 for j in jobs if j.state in FINAL)
                run = [j.progress for j in jobs if j.state not in FINAL and j.state == 'running']
                progress((done + sum(run)) / len(jobs), f'{done} of {len(jobs)} {what}')
            time.sleep(0.2)
    except (Cancelled, KeyboardInterrupt):
        for j in jobs:
            if j.state not in FINAL:
                svc.queue.cancel(j)
        t0 = time.monotonic()
        while any(j.state not in FINAL for j in jobs) and time.monotonic() - t0 < 30:
            svc.queue.poll()
            time.sleep(0.1)
        raise Cancelled() from None
    bad = [j for j in jobs if j.state != DONE]
    if bad:
        why = []
        for j in bad[:3]:
            err = ''
            if j.spec.run_dir and os.path.isfile(os.path.join(j.spec.run_dir, 'run.json')):
                try:
                    err = PV.read_json(os.path.join(j.spec.run_dir, 'run.json')).get('error') or ''
                except (OSError, ValueError):
                    err = ''
            tail = [t for lvl, t in j.log if lvl in ('error', 'stderr') and t][-1:] if not err else []
            why.append(f'{j.spec.title}: {j.state}' + (f' ({err or tail[0]})' if err or tail else ''))
        raise LoopError(f'{len(bad)} of {len(jobs)} {what} did not finish: ' + '; '.join(why))


# ================================================================================================ the evaluation set
def eval_spec(odd: dict, key: str) -> dict:
    ev = odd['evaluation']
    return {'id': f'loop-eval-{key[:12]}', 'name': f'Loop evaluation set: {odd["name"]}', 'rig': odd['rig'],
            'fps': float(ev['fps']), 'record_every': int(ev['record_every']), 'drive_s': float(ev['drive_s']),
            'worlds': list(odd['worlds']), 'weathers': [w['name'] for w in odd['weathers']], 'scenarios': [],
            'targets': {}, 'bins': {'distance_m': list(odd['range_bins']) + [SPEC_FAR_M]}, 'bin_targets': {},
            'traffic_profile': odd['traffic_profile'] or None, 'seed': int(ev['seed']),
            'max_drives': len(odd_cells(odd)) * int(ev['drives_per_cell'])}


def eval_plan(odd: dict) -> dict:
    """The evaluation set's drives: drives_per_cell per cell, at fixed seeds."""
    ev = odd['evaluation']
    drives = []
    i = 0
    for cell in odd_cells(odd):
        for _ in range(int(ev['drives_per_cell'])):
            seed = EVAL_SEED_BASE + int(ev['seed']) * 10000 + i
            drives.append({'cell': {'world': cell['world'], 'weather': sample_weather(cell['weather'], seed),
                                    'scenario': None, 'label': cell['label']}, 'seed': seed, 'index': i})
            i += 1
    return {'drives': drives}


def _drive_cfgs(project, spec, plan, svc):
    from ..sim import engine as E
    return E.drive_configs(project, spec, plan, svc.port, svc.tm_port, float(svc.settings.get('render_budget_mpx')))


def evalset_path(ws: str, key: str) -> str:
    return loop_dir(ws, 'evalsets', f'loop-eval-{key[:12]}.json')


def read_evalset(ws: str, key: str) -> dict | None:
    """The evaluation set's manifest when it is complete and every drive's run.json still has its recorded hash."""
    p = evalset_path(ws, key)
    if not os.path.isfile(p):
        return None
    doc = PV.read_json(p)
    for d in doc['drives']:
        rj = os.path.join(absp(ws, d['run']), 'run.json')
        if not os.path.isfile(rj) or PV.sha256_file(rj) != d['run_json_sha256']:
            return None
    return doc


def ensure_evalset(project, odd: dict, svc, cancel=None, progress=None, log=None) -> dict:
    """Record the evaluation set if needed (cached drives are reused) and return its manifest."""
    ws = project.workspace
    key = odd_key(odd, project)
    doc = read_evalset(ws, key)
    if doc is not None:
        return doc
    spec = eval_spec(odd, key)
    plan = eval_plan(odd)
    cfgs = _drive_cfgs(project, spec, plan, svc)
    todo = [c for c, hit in cfgs if not hit]
    if log:
        log(f'evaluation set {spec["id"]}: {len(cfgs)} drives, {len(cfgs) - len(todo)} already recorded')
    jobs = [svc.submit_drive(c, title=f'Loop evaluation drive {c["dataset"]["index"] + 1} of {len(cfgs)}')
            for c in todo]
    _wait_jobs(svc, jobs, cancel, progress, 'evaluation drives')
    drives = []
    for (c, _), d in zip(cfgs, plan['drives']):
        drives.append({'run': rel(ws, c['out']), 'key': c['cache_key'], 'cell': d['cell']['label'],
                       'world': d['cell']['world'], 'weather': d['cell']['weather'], 'seed': d['seed'],
                       'run_json_sha256': PV.sha256_file(os.path.join(c['out'], 'run.json'))})
    doc = {'schema': EVALSET_SCHEMA, 'id': spec['id'], 'odd_key': key, 'odd': odd['id'], 'spec': spec,
           'drives': drives, 'created': PV.utc_now(),
           'rule': 'held out: never a training, validation or test source of the loop (assert_no_leak)',
           'provenance': PV.block({'project': project.path}, int(odd['evaluation']['seed']), {'spec': spec})}
    PV.write_json(evalset_path(ws, key), doc)
    return doc


def evalset_runs(ws: str, doc: dict) -> list:
    return [absp(ws, d['run']) for d in doc['drives']]


def _version_runs(src: dict) -> list:
    path = src['path']
    v = PV.read_json(path)
    wsv = src.get('workspace') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(path)))))
    out = []
    for e in v.get('runs') or ():
        out.append(absp(wsv, e['run']))
    return out


def source_runs(src) -> list:
    """The drive folders a training source reads (a falcon run, or a dataset version's runs); [] for real sets."""
    if isinstance(src, str):
        from .training import parse_source
        src = parse_source(src)
    if src.get('kind') == 'falcon':
        return [os.path.abspath(src['path'])]
    if src.get('kind') == 'version':
        return _version_runs(src)
    return []


def assert_no_leak(sources: list, evalset: dict, ws: str):
    """Refuse (LoopError) when a training source is, contains or lists a drive of the evaluation set (by folder or by
    cache key)."""
    eval_dirs = {os.path.realpath(p) for p in evalset_runs(ws, evalset)}
    eval_keys = {d['key'] for d in evalset['drives']}
    bad = []
    for s in sources:
        for r in source_runs(s):
            rp = os.path.realpath(r)
            key = None
            try:
                key = PV.read_json(os.path.join(r, 'drive.json')).get('cache_key')
            except (OSError, ValueError):
                pass
            if rp in eval_dirs or (key is not None and key in eval_keys) or \
                    any(e.startswith(rp + os.sep) for e in eval_dirs):      # is one, or a folder holding one
                bad.append(r)
    if bad:
        raise LoopError('refused: the evaluation set is never trained on, but these training sources are (or hold) '
                        'evaluation drives: ' + ', '.join(sorted(set(bad))[:5]))


# ================================================================================================ weak cells
def _edges(odd_bins) -> tuple:
    return tuple(float(x) for x in odd_bins) + (math.inf,)


def bin_label(lo, hi) -> str:
    return f'{lo:g}–{hi:g} m' if hi not in (None, math.inf) and hi < SPEC_FAR_M else f'≥ {lo:g} m'


def _cell_key(cls, lo) -> str:
    return f'{cls}|{float(lo):g}'


def _cell_metric(classes, edges):
    from . import metrics as M

    def fn(records):
        s = M.coco_summarise(records)
        out = {}
        for b in (s.get('breakdowns') or {}).get('range_m', []):
            for c, v in (b.get('per_class_AP') or {}).items():
                if c in classes:
                    out[_cell_key(c, b['lo'])] = v
        return out
    return fn


def _groups(frames: list, cell_of: dict) -> dict:
    out = {ALL: list(frames)}
    for f in frames:
        c = cell_of.get(f['drive'])
        if c is not None:
            out.setdefault(c, []).append(f)
    return out


def _count(frames, cls, lo, hi) -> int:
    n = 0
    for f in frames:
        for g in f.get('gt') or ():
            r = g.get('range_m')
            if g.get('class') == cls and not g.get('ignore') and r is not None and lo <= float(r) < hi:
                n += 1
    return n


def cell_table(frames: list, cell_of: dict, classes, range_bins, n_boot: int = 1000, seed: int = 0) -> list:
    """AP per class × range bin × condition (each cell of the ODD, and ALL) with a percentile bootstrap interval over
    drives (over frames with a single drive): rows {'class', 'bin': [lo, hi], 'condition', 'AP', 'ci_low', 'ci_high',
    'n_objects', 'n_units', 'unit'}."""
    from . import metrics as M
    edges = _edges(range_bins)
    fn = _cell_metric(set(classes), edges)
    rows = []
    for cond, fr in _groups(frames, cell_of).items():
        if not fr:
            continue
        recs = M.coco_match(fr, breakdowns={'range_m': edges})
        unit = 'drive' if len({f['drive'] for f in fr}) > 1 else 'image'
        try:
            boot = M.bootstrap(fn, recs, n=int(n_boot), seed=int(seed), unit=unit)
        except ValueError:
            continue
        mets = boot.get('metrics') or {}
        for c in classes:
            for lo, hi in zip(edges, edges[1:]):
                n = _count(fr, c, lo, hi)
                r = mets.get(_cell_key(c, lo)) or {}
                rows.append({'class': c, 'bin': [lo, None if hi == math.inf else hi], 'condition': cond,
                             'AP': r.get('value'), 'ci_low': r.get('ci_low'), 'ci_high': r.get('ci_high'),
                             'n_objects': n, 'n_units': boot.get('n_units'), 'unit': unit})
    return rows


def rank_weak(rows: list, overall_ap, min_objects: int, max_cells: int, fallback: str = 'none') -> list:
    """The weak cells (WEAK_RULE), ranked; each row gets 'weak' and, when chosen, 'rank' (1-based). With fallback
    'lowest' and no weak cell (a model uniformly weak — AP 0 everywhere — or uniformly strong), the cells with
    enough objects and the lowest interval upper ends are chosen instead, each marked 'fallback'."""
    cand, enough = [], []
    for r in rows:
        r['weak'] = bool(overall_ap is not None and r['n_objects'] >= int(min_objects) and r['ci_high'] is not None
                         and r['ci_high'] < overall_ap)
        r.pop('rank', None)
        r.pop('fallback', None)
        if r['weak']:
            cand.append(r)
        if r['n_objects'] >= int(min_objects) and r['ci_high'] is not None:
            enough.append(r)
    use_fallback = not cand and fallback == 'lowest'
    if use_fallback:
        cand = enough
    cand.sort(key=lambda r: (r['ci_high'], -r['n_objects'], r['class'], r['bin'][0], r['condition'] != ALL,
                             r['condition']))
    chosen, seen = [], set()
    for r in cand:
        k = (r['class'], r['bin'][0])
        if k in seen:
            continue
        seen.add(k)
        r['rank'] = len(chosen) + 1
        if use_fallback:
            r['fallback'] = True
        chosen.append(r)
        if len(chosen) >= int(max_cells):
            break
    return chosen


def headline(evaluation: dict, metric: str = 'AP') -> dict:
    row = next((r for r in evaluation['scores']['table'] if r['metric'] == metric), None) or {}
    return {'value': row.get('value'), 'ci_low': row.get('ci_low'), 'ci_high': row.get('ci_high')}


# ================================================================================================ the search
def _study_class():
    from ..sim.studio import Study

    class CellStudy(Study):
        """A study run in an ODD cell's weather, with a given rig document (the loop's) and frame rate."""

        def __init__(self, svc, project, scenario, name, world, rig_id, seed, weather, rig_doc, fps, traffic=0):
            self.weather, self.rig_doc, self.fps = copy.deepcopy(weather), copy.deepcopy(rig_doc), float(fps)
            super().__init__(svc, project, scenario, name, world, rig_id, seed, traffic)
            PV.write_json(os.path.join(self.dir, 'study.json'),
                          {'name': name, 'scenario': scenario, 'world': world, 'rig': rig_id, 'seed': seed,
                           'traffic': traffic, 'weather': weather, 'rig_doc': rig_doc, 'fps': fps,
                           'made_by': 'loop'})

        def config(self, values, k):
            from . import datasets as D
            from . import weather as W
            from .drive import scenario_config
            cfg = scenario_config(self.project, self.scn, self.world, self.rig, '/pending', self.svc.port,
                                  self.svc.tm_port, seed=self.seed, values=values, fps=self.fps,
                                  traffic_vehicles=self.traffic,
                                  render_budget_mpx=float(self.svc.settings.get('render_budget_mpx')),
                                  workspace=self.project.workspace)
            cfg['weather'] = W.resolve(self.weather)
            cfg['weather_source'] = copy.deepcopy(self.weather)
            cfg['rig'] = copy.deepcopy(self.rig_doc)
            key = D.drive_key(cfg)
            cfg['out'] = os.path.join(self.dir, 'runs', key[:16])
            cfg['name'] = f'{self.name} #{k}'
            cfg['cache_key'] = key
            return cfg
    return CellStudy


def unlabelled_rig(rig: dict) -> dict:
    r = copy.deepcopy(rig)
    for s in r.get('sensors', []):
        if s.get('labels'):
            s['labels'] = []
    return r


def scenario_for(cls: str, fixed: dict, key: str) -> dict:
    from . import scenarios as SC
    scn = SC.builtin(key)
    for k, v in (fixed or {}).items():
        scn['params'][k] = {'dist': 'fixed', 'value': v, 'unit': '', 'about': f'held at {v!r} for the {cls} class'}
    return scn


def search_condition(cell: dict, table: list, odd: dict) -> dict:
    """The world and weather a weak cell's search runs in: the cell's own, or for ALL the condition with the lowest
    AP of that class and bin (else the first cell)."""
    cells = {c['label']: c for c in odd_cells(odd)}
    cond = cell['condition']
    if cond == ALL:
        cand = [r for r in table if r['class'] == cell['class'] and r['bin'][0] == cell['bin'][0]
                and r['condition'] != ALL and r['AP'] is not None and r['n_objects'] > 0]
        cand.sort(key=lambda r: (r['AP'], -r['n_objects'], r['condition']))
        cond = cand[0]['condition'] if cand else next(iter(cells))
    return cells[cond]


# ================================================================================================ costs
def _drive_estimate(throughput, steps: int, mpx: float, n: int) -> dict | None:
    if n <= 0:
        return {'seconds': 0.0, 'label': 'measured', 'source': 'nothing to record (cached)'}
    e = throughput.estimate(steps, mpx)
    return {'seconds': e['seconds'] * n, 'label': e['label'], 'source': f'{n} drive(s) × {e["seconds"]:.0f} s at '
            f'{mpx:.3g} Mpx per step; {e["source"]}'}


def _combine(parts: list, note: str = '') -> dict:
    """One estimate from parts: seconds summed when every part has them; the weakest label wins."""
    order = ['measured', 'extrapolated from measured', 'typical', 'not estimated']
    secs = [p.get('seconds') for p in parts]
    known = all(s is not None for s in secs)
    lab = max((p.get('label', 'not estimated') for p in parts), key=lambda x: order.index(x) if x in order else 3,
              default='measured')
    out = {'seconds': sum(secs) if known else None, 'label': lab if known else 'not estimated',
           'parts': parts, 'source': '; '.join(p.get('source', '') for p in parts if p.get('source'))}
    if note:
        out['note'] = note
    return out


def _eval_seconds(ws: str, m: dict, frames: int) -> dict:
    """Seconds for an evaluation of frames frames: from an earlier loop evaluation of the same model file (measured
    wall time per frame), else from an evaluation.json latency of the same model (inference only)."""
    fp = model_fingerprint(m, ws)
    for it in reversed(iterations(ws)):
        for s in ('reevaluate', 'evaluate'):
            rec = it['steps'].get(s) or {}
            meas = rec.get('measured') or {}
            if rec.get('state') == 'done' and meas.get('frames') and meas.get('model_sha256') == fp and \
                    not rec.get('reused_from') and meas.get('eval_s'):
                per = meas['eval_s'] / meas['frames']
                return {'seconds': per * frames, 'label': 'measured', 'source': f'{frames} frames × {per:.3f} s '
                        f'(measured: loop iteration {it["n"]}, step {STEPS.index(s) + 1}, this model)'}
    return {'seconds': None, 'label': 'not estimated', 'source': 'this model has not been evaluated by the loop on '
            'this machine yet: its time per frame is measured by this step'}


def _training_throughput(ws: str, arch: str, device: str, own_run: dict | None) -> dict:
    cands = []
    if own_run is not None:
        cands.append(('the model\'s own training', own_run))
    tdir = os.path.join(ws, 'training')
    if os.path.isdir(tdir):
        for name in sorted(os.listdir(tdir), reverse=True):
            p = os.path.join(tdir, name, 'run.json')
            try:
                r = PV.read_json(p)
            except (OSError, ValueError):
                continue
            cands.append((f'training {name}', r))
    for what, r in cands:
        s = (r.get('provenance') or {}).get('settings') or {}
        t = r.get('timings') or {}
        if r.get('status') == 'ok' and s.get('arch') == arch and (r.get('device') or {}).get('type') == device \
                and t.get('images_per_s'):
            return {'images_per_s': float(t['images_per_s']), 'peak_gpu_mib': t.get('peak_gpu_mib'),
                    'val_s': t.get('val_s'), 'label': 'measured',
                    'source': f'measured: {what} ({arch} on {device}, {t["images_per_s"]:.1f} images/s'
                              + (f', peak {t["peak_gpu_mib"]:.0f} MiB' if t.get('peak_gpu_mib') else '') + ')'}
    if device == 'cuda' and arch in TYPICAL_TRAINING:
        ty = TYPICAL_TRAINING[arch]
        return {**ty, 'val_s': None, 'label': 'typical', 'source': TYPICAL_TRAINING_SOURCE}
    return {'images_per_s': None, 'peak_gpu_mib': None, 'label': 'not estimated',
            'source': f'no training of {arch} on {device} has been measured here, and the reference figures are '
                      'for CUDA only'}


def estimate(project, it: dict | None, step: str, throughput, odd: dict | None = None,
             budget_mpx: float | None = None) -> dict:
    """The cost of running step next (seconds, label, source and parts; memory for training) from what is known now."""
    from .budget import rig_mpx
    ws = project.workspace
    if odd is None:
        odd = (it or {}).get('odd') or normalise_odd(project.doc.get('loop', {}).get('odd'), project)[0]
    if odd is None:
        return {'seconds': None, 'label': 'not estimated', 'source': 'no ODD'}
    rig = project.item('rigs', odd['rig']) or {}
    mpx = rig_mpx(rig)
    ev = odd['evaluation']
    eval_steps = max(int(round(float(ev['drive_s']) * float(ev['fps']))), 1)
    eval_frames_per_drive = eval_steps // max(int(ev['record_every']), 1) // max(int(ev['every']), 1)
    n_eval = len(odd_cells(odd)) * int(ev['drives_per_cell'])
    if step == 'pick':
        return {'seconds': 0.0, 'label': 'measured', 'source': 'no computation: the model and the ODD are recorded'}
    if step in ('evaluate', 'reevaluate'):
        parts = []
        key = odd_key(odd, project)
        if step == 'evaluate' and read_evalset(ws, key) is None:
            parts.append(_drive_estimate(throughput, eval_steps, mpx, n_eval))
            parts[-1]['source'] = 'record the evaluation set: ' + parts[-1]['source']
        m = None
        if step == 'reevaluate' and it is not None:
            m = ((it['steps'].get('retrain') or {}).get('result') or {}).get('registration')
        elif it is not None:
            m = it.get('model')
        else:
            m = project.item('models', project.doc.get('loop', {}).get('model') or '')
        if step == 'evaluate' and it is not None and _reusable_before(ws, it) is not None:
            return {'seconds': 0.0, 'label': 'measured', 'source': 'reused: the previous iteration evaluated this '
                    'model on this evaluation set'}
        if m is None:
            parts.append({'seconds': None, 'label': 'not estimated', 'source': 'the model is not known yet'})
        else:
            parts.append(_eval_seconds(ws, m, eval_frames_per_drive * n_eval))
        return _combine(parts)
    if step in ('weak_cells', 'compare'):
        prev = [(i2['steps'].get(step) or {}).get('measured', {}).get('wall_s') for i2 in iterations(ws)]
        prev = [p for p in prev if p is not None]
        if prev:
            return {'seconds': prev[-1], 'label': 'measured', 'source': f'measured: the last iteration\'s step '
                    f'{STEPS.index(step) + 1} ({prev[-1]:.1f} s); a bootstrap over the predictions, no simulation'}
        return {'seconds': None, 'label': 'not estimated', 'source': 'a bootstrap over the evaluation\'s predictions '
                '(CPU only, no simulation or training); measured by this step'}
    if step == 'search':
        cells = _chosen_cells(ws, it)
        if cells is None:
            return {'seconds': None, 'label': 'not estimated', 'source': 'the weak cells are not known yet'}
        se = odd['search']
        parts = []
        for k, c in enumerate(cells):
            key = (SCENARIOS_FOR_CLASS.get(c['class']) or [(None, {})])[0][0]
            if key is None:
                parts.append({'seconds': 0.0, 'label': 'measured', 'source': f'no scenario for {c["class"]}'})
                continue
            from . import scenarios as SC
            dur = float(SC.LIBRARY[key].get('duration_s', 12.0))
            steps = int(round(dur * float(se['fps']))) + 5
            sdir = os.path.join(ws, 'studies', f'loop-{it["n"]:03d}-{k + 1}-{key}', 'results.jsonl')
            done = 0
            if os.path.isfile(sdir):                   # a resumed search: its finished runs are not run again
                with open(sdir, encoding='utf-8') as fh:
                    done = sum(1 for x in fh if x.strip())
            p = _drive_estimate(throughput, steps, rig_mpx(unlabelled_rig(rig)), max(int(se['budget']) - done, 0))
            p['source'] = f'{key}: ' + p['source']
            parts.append(p)
        return _combine(parts) if parts else {'seconds': 0.0, 'label': 'measured', 'source': 'no weak cell'}
    if step == 'generate':
        if it is None or not step_done(ws, it, 'search'):
            return {'seconds': None, 'label': 'not estimated', 'source': 'the search has not run yet'}
        try:
            plan = targeted_plan(project, it, odd, throughput)
        except (LoopError, OSError, ValueError, KeyError) as e:
            return {'seconds': None, 'label': 'not estimated', 'source': str(e)}
        from ..sim import engine as E
        cached = 0
        if plan['drives']:
            if budget_mpx is None:                     # part of a drive's config, so of its cache key
                from .settings import Settings
                budget_mpx = float(Settings().get('render_budget_mpx'))
            cfgs = E.drive_configs(project, plan['spec'], plan, 0, 0, budget_mpx)   # ports are not in the key
            cached = sum(1 for _, hit in cfgs if hit)
        rig_doc = project.item('rigs', odd['rig']) or {}
        g = odd['generate']
        p0 = _drive_estimate(throughput, max(int(round(float(g['drive_s']) * float(g['fps']))), 1), rig_mpx(rig_doc),
                             len(plan['drives']) - cached)
        parts = [dict(p0, source=f'targeted dataset: {len(plan["drives"])} drive(s), {cached} already recorded; '
                                 + p0['source'])]
        n_re, steps_re = _rerun_count(ws, it, odd)
        if n_re:
            p = _drive_estimate(throughput, steps_re, mpx, n_re)
            p['source'] = 're-run the failing cases with labels: ' + p['source']
            parts.append(p)
        return _combine(parts)
    if step == 'retrain':
        m = (it or {}).get('model')
        if m is None:
            return {'seconds': None, 'label': 'not estimated', 'source': 'the model is not known yet'}
        try:
            d, run = training_run_of(m)
        except Stopped as e:
            return {'seconds': 0.0, 'label': 'measured', 'source': f'the step will stop: {e}'}
        s = run['provenance']['settings']
        device = (run.get('device') or {}).get('type') or 'cpu'
        tp = _training_throughput(ws, s['arch'], device, run)
        n_old = int(((run.get('data') or {}).get('train') or {}).get('n_images') or 0)
        n_new = _new_images(ws, it, odd)
        epochs = int(odd['retrain'].get('epochs') or s['epochs'])
        imgs = (n_old + n_new) * epochs
        mem = {'peak_gpu_mib': tp.get('peak_gpu_mib'), 'label': tp['label'] if tp.get('peak_gpu_mib') else
               ('not used (CPU)' if device == 'cpu' else 'not estimated'), 'source': tp['source']}
        if not tp.get('images_per_s'):
            return {'seconds': None, 'label': 'not estimated', 'source': tp['source'], 'memory': mem,
                    'images': imgs}
        secs = imgs / tp['images_per_s'] + float(tp.get('val_s') or 0.0)
        return {'seconds': secs, 'label': tp['label'], 'images': imgs, 'memory': mem,
                'source': f'({n_old} original + {n_new} new images) × {epochs} epochs at {tp["images_per_s"]:.1f} '
                          f'images/s' + (f' + validation {tp["val_s"]:.0f} s' if tp.get('val_s') else '') +
                          f'; {tp["source"]}'}
    raise ValueError(step)


def _chosen_cells(ws, it):
    if it is None or not step_done(ws, it, 'weak_cells'):
        return None
    return PV.read_json(os.path.join(iteration_dir(ws, it['n']), 'weak_cells.json'))['chosen']


def _search_doc(ws, it):
    return PV.read_json(os.path.join(iteration_dir(ws, it['n']), 'search.json'))


def _rerun_count(ws, it, odd) -> tuple:
    from . import scenarios as SC
    doc = _search_doc(ws, it)
    n = 0
    steps = 0
    for s in doc['cells']:
        k = len(s.get('rerun_cases') or [])
        n += k * int(odd['generate']['reruns_per_failure'])
        if s.get('scenario'):
            steps = max(steps, int(round(float(SC.LIBRARY[s['scenario']].get('duration_s', 12.0)) *
                                         float(odd['search']['fps']))) + 5)
    return n, steps


def _new_images(ws, it, odd) -> int:
    """Images the retrain adds: the generate step's recorded camera frames, or before it ran, its plan's."""
    if it is not None and step_done(ws, it, 'generate'):
        return int(PV.read_json(os.path.join(iteration_dir(ws, it['n']), 'generate.json')).get('frames') or 0)
    g = odd['generate']
    per = int(float(g['drive_s']) * float(g['fps']) / max(int(g['record_every']), 1))
    return per * int(g['max_drives'])


# ================================================================================================ generation
def targeted_spec(odd: dict, n: int, chosen: list) -> dict:
    g = odd['generate']
    cells = {c['label']: c for c in odd_cells(odd)}
    conds = sorted({c['condition'] for c in chosen})
    used = list(cells.values()) if ALL in conds else [cells[c] for c in conds if c in cells]
    targets, bt = {}, {}
    edges = list(odd['range_bins']) + [SPEC_FAR_M]
    for c in chosen:
        targets[c['class']] = targets.get(c['class'], 0) + int(g['objects_per_cell'])
        row = bt.setdefault(c['class'], [0] * (len(edges) - 1))
        row[edges.index(float(c['bin'][0]))] += int(g['objects_per_cell'])
    return {'id': f'loop-{int(n):03d}-targeted', 'name': f'Loop {n}: targeted data ({odd["name"]})',
            'rig': odd['rig'], 'fps': float(g['fps']), 'record_every': int(g['record_every']),
            'drive_s': float(g['drive_s']), 'worlds': sorted({c['world'] for c in used}),
            'weathers': [w for w in odd['weathers'] if any(w['name'] == c['weather']['name'] for c in used)],
            'scenarios': [], 'targets': targets, 'bins': {'distance_m': edges}, 'bin_targets': {'distance_m': bt},
            'traffic_profile': odd['traffic_profile'] or None, 'seed': int(g['seed']) * 1000 + int(n),
            'max_drives': int(g['max_drives']), 'made_by': f'loop iteration {n}', 'cells': [c['label'] for c in used]}


def _evalset_yield(ws: str, evalset: dict, odd: dict, chosen: list) -> dict:
    """Objects of each chosen cell's class in its range bin per recorded frame of the evaluation drives (those drives'
    conditions: the cell's, or all for ALL)."""
    from . import datasets as D
    edges = list(odd['range_bins']) + [SPEC_FAR_M]
    rows = {d['cell']: [] for d in evalset['drives']}
    for d in evalset['drives']:
        rows[d['cell']].append(D.coverage_of_run(absp(ws, d['run']), {'distance_m': edges}))
    out = {}
    for c in chosen:
        sel = [r for k, rs in rows.items() for r in rs if c['condition'] in (ALL, k)]
        frames = sum(r['frames'] for r in sel)
        k = edges.index(float(c['bin'][0]))
        n = sum((r['bins'].get('distance_m', {}).get(c['class']) or [0] * (len(edges) - 1))[k] for r in sel)
        y = n / frames if frames else 0.0
        out[c['class']] = min(out.get(c['class'], math.inf), y) if c['class'] in out else y
    return out


def targeted_plan(project, it: dict, odd: dict, throughput) -> dict:
    """The targeted dataset's plan (core.datasets.plan) with the cells' sampled weathers; its estimate is the plan's."""
    from . import datasets as D
    from .budget import rig_mpx
    ws = project.workspace
    chosen = _chosen_cells(ws, it) or []
    evalset = PV.read_json(absp(ws, it['evalset']))
    spec = targeted_spec(odd, it['n'], chosen)
    if not chosen:
        return {'spec': spec, 'drives': [], 'estimate': None, 'yield': {}, 'needed': {}}
    y = _evalset_yield(ws, evalset, odd, chosen)
    rig = project.item('rigs', odd['rig'])
    p = D.plan(spec, {}, y or None, throughput, rig_mpx(rig))
    p['yield_source'] = ('measured on the loop\'s evaluation set: objects of the cell\'s class in its range bin per '
                         'recorded frame, in the cell\'s conditions')
    cells = {c['label']: c for c in odd_cells(odd)}
    used = [cells[l] for l in spec['cells']]
    for d in p['drives']:
        c = used[d['index'] % len(used)]
        d['cell'] = {'world': c['world'], 'weather': sample_weather(c['weather'], d['seed']), 'scenario': None,
                     'label': c['label']}
    p['spec'] = spec
    return p


# ================================================================================================ the steps
class Runner:
    """Runs the steps of an iteration with a CarlaService (its queue runs every drive and the training)."""

    def __init__(self, project, svc, emit=None, cancel=None):
        self.project, self.svc = project, svc
        self.ws = project.workspace
        self.em = emit
        self.cancel = cancel
        self.lo, self.hi = 0.0, 1.0

    # -- reporting
    def log(self, text, level='info'):
        if self.em is not None:
            self.em.log(text, level)

    def progress(self, f, text=''):
        if self.em is not None:
            self.em.progress(self.lo + (self.hi - self.lo) * min(max(f, 0.0), 1.0), text)

    def sub(self, a, b):
        lo, hi = self.lo, self.hi
        return lambda f, t='': self.em.progress(lo + (hi - lo) * (a + (b - a) * min(max(f, 0.0), 1.0)), t) \
            if self.em is not None else None

    # -- iterations
    def settings(self) -> tuple:
        """(the model to start a new iteration with, where it came from, the normalised ODD) from the project."""
        loop = self.project.doc.get('loop') or {}
        odd, errs = normalise_odd(loop.get('odd'), self.project)
        if errs:
            raise Stopped('the ODD is not usable: ' + '; '.join(errs))
        mid = loop.get('model')
        m = self.project.item('models', mid) if mid else None
        if m is None:
            raise Stopped('no model is chosen for the loop (or it is no longer in the project)' if not mid else
                          f'the loop\'s model {mid!r} is not in the project')
        if m.get('kind') != 'detector':
            raise Stopped(f'{m["name"]} is a {m.get("kind")}: the loop evaluates detectors (2D boxes)')
        its = iterations(self.ws)
        last = its[-1] if its else None
        if last is not None and last['status'] == 'complete' and last.get('origin_model') == mid and \
                last.get('odd_sha256') == PV.sha256_json(odd):
            reg = ((last['steps'].get('retrain') or {}).get('result') or {}).get('registration')
            if reg:
                return reg, mid, odd, f'the model retrained by iteration {last["n"]}'
        return m, mid, odd, f'the project\'s model {mid!r}'

    def new_iteration(self) -> dict:
        m, origin, odd, why = self.settings()
        its = iterations(self.ws)
        n = (its[-1]['n'] + 1) if its else 1
        for old in its:
            if old['status'] in ('running', 'failed', 'cancelled'):
                old['status'] = 'abandoned'
                save_iteration(self.ws, old)
        it = {'schema': ITERATION_SCHEMA, 'n': n, 'created': PV.utc_now(), 'status': 'running',
              'project': self.project.path, 'origin_model': origin, 'model': copy.deepcopy(m), 'model_why': why,
              'model_sha256': model_fingerprint(m, self.ws), 'odd': odd, 'odd_sha256': PV.sha256_json(odd),
              'odd_key': odd_key(odd, self.project), 'evalset': None,
              'steps': {s: {'state': 'pending'} for s in STEPS}, 'summary': {},
              'provenance': PV.block({'project': self.project.path, 'model': model_file(m, self.ws)},
                                     None, {'odd': odd})}
        os.makedirs(iteration_dir(self.ws, n), exist_ok=True)
        save_iteration(self.ws, it)
        return it

    def current(self, start_new: bool = True) -> dict:
        """The iteration to work on: the latest unfinished one, else a new one. Refused (Stopped) when the latest
        stopped for a reason a new iteration would meet again, or when the project's model or ODD changed under an
        unfinished one."""
        its = iterations(self.ws)
        last = its[-1] if its else None
        if last is not None and last['status'] in ('running', 'failed', 'cancelled'):
            loop = self.project.doc.get('loop') or {}
            odd, errs = normalise_odd(loop.get('odd'), self.project)
            if not errs and (loop.get('model') != last.get('origin_model') or
                             PV.sha256_json(odd) != last.get('odd_sha256')):
                raise Stopped(f'iteration {last["n"]} was started with model {last.get("origin_model")!r} and ODD '
                              f'{last["odd"]["id"]!r}; the project now asks for others. Finish it, or start a new '
                              'iteration (falcon loop PROJECT new; the Loop page\'s New iteration)')
            return last
        if last is not None and last['status'] == 'stopped':
            loop = self.project.doc.get('loop') or {}
            odd, errs = normalise_odd(loop.get('odd'), self.project)
            if errs or (loop.get('model') == last.get('origin_model') and PV.sha256_json(odd) == last['odd_sha256']):
                raise Stopped(f'iteration {last["n"]} stopped: {last.get("stopped")}')
        if not start_new:
            raise Stopped('no iteration to continue')
        return self.new_iteration()

    # -- running
    def run(self, until: str = 'compare', one: bool = False, it: dict | None = None) -> dict:
        """Run the iteration's steps up to and including until (or only the next one). Returns the iteration."""
        it = it or self.current()
        last = STEPS.index(until)
        thr = self.svc.throughput()
        ran = 0
        for k, s in enumerate(STEPS[:last + 1]):
            if step_done(self.ws, it, s):          # checked as it goes: a step run again invalidates what follows
                continue
            if one and ran:
                break
            ran += 1
            self.lo, self.hi = (k / (last + 1), (k + 1) / (last + 1)) if not one else (0.0, 1.0)
            rec = it['steps'][s]
            est = estimate(self.project, it, s, thr, it['odd'], float(self.svc.settings.get('render_budget_mpx')))
            rec.update(state='running', started=PV.utc_now(), estimate=est, error=None, upstream=_upstream(it, s),
                       runs=int(rec.get('runs') or 0) + 1)
            it['status'] = 'running'
            save_iteration(self.ws, it)
            self.log(f'step {STEPS.index(s) + 1} ({STEP_TITLES[s]}): estimated {describe_estimate(est)}')
            self.progress(0.0, STEP_TITLES[s])
            t0 = time.monotonic()
            try:
                out = getattr(self, f'step_{s}')(it, rec)
            except Stopped as e:
                rec.update(state='stopped', error=str(e), finished=PV.utc_now(),
                           measured={'wall_s': round(time.monotonic() - t0, 3)})
                it['status'], it['stopped'] = 'stopped', str(e)
                save_iteration(self.ws, it)
                raise
            except Cancelled:
                rec.update(state='cancelled', finished=PV.utc_now(),
                           measured={'wall_s': round(time.monotonic() - t0, 3)})
                it['status'] = 'cancelled'
                save_iteration(self.ws, it)
                raise
            except Exception as e:          # noqa: BLE001 (recorded, then re-raised)
                rec.update(state='failed', error=f'{type(e).__name__}: {e}' if not isinstance(e, LoopError) else
                           str(e), finished=PV.utc_now(), measured={'wall_s': round(time.monotonic() - t0, 3)})
                it['status'] = 'failed'
                save_iteration(self.ws, it)
                raise
            rec['measured'] = dict(rec.get('measured') or {}, wall_s=round(time.monotonic() - t0, 3))
            rec['outputs'] = _hash_outputs(self.ws, out)
            rec.update(state='done', finished=PV.utc_now())
            if all(step_done(self.ws, it, x) for x in STEPS):
                it['status'] = 'complete'
                it['finished'] = PV.utc_now()
            save_iteration(self.ws, it)
            self.progress(1.0, f'{STEP_TITLES[s]}: done')
        return it

    def _file(self, it, *parts):
        return os.path.join(iteration_dir(self.ws, it['n']), *parts)

    # -- 1
    def step_pick(self, it, rec):
        m = it['model']
        f = model_file(m, self.ws)
        if f and not os.path.isfile(f):
            raise LoopError(f'the model file {f} is missing (download it on the Models page)')
        doc = {'schema': 'falcon.loop_pick/1', 'n': it['n'], 'model': m, 'model_sha256': it['model_sha256'],
               'model_why': it.get('model_why'), 'odd': it['odd'], 'odd_sha256': it['odd_sha256'],
               'odd_key': it['odd_key'], 'provenance': it['provenance']}
        rec['decisions'] = [f'model {m.get("name")} ({it.get("model_why")})', f'ODD {it["odd"]["name"]}: '
                            f'{len(odd_cells(it["odd"]))} cell(s), classes {", ".join(it["odd"]["classes"])}']
        p = self._file(it, 'pick.json')
        PV.write_json(p, doc)
        return [p]

    # -- 2 and 7
    def _evaluate(self, it, m, out_dir, rec, a=0.0):
        from . import harness as H
        evalset = PV.read_json(absp(self.ws, it['evalset']))
        runs = evalset_runs(self.ws, evalset)
        ev = it['odd']['evaluation']
        rig = self.project.item('rigs', it['odd']['rig']) or {}
        cam = next((s['id'] for s in rig.get('sensors', []) if s.get('type') == 'camera.pinhole'), None)
        locks = []
        try:
            if m.get('runtime') != 'python':
                from .training import gpu_locks
                locks = gpu_locks(self.em, self.cancel)
            t0 = time.monotonic()
            doc = H.evaluate(runs, m, out_dir, self.ws, cam, int(ev['every']), int(ev['n_boot']), int(ev['seed']),
                             progress=self.sub(a, 1.0), cancel=self.cancel)
            eval_s = time.monotonic() - t0
        except H.HarnessError as e:
            if str(e) == 'cancelled':
                raise Cancelled() from None
            raise LoopError(f'evaluation: {e}') from None
        finally:
            for lk in locks:
                lk.release()
        rec['measured'] = {'frames': doc['n_frames'], 'eval_s': round(eval_s, 3),
                           'model_sha256': model_fingerprint(m, self.ws), 'latency_ms': doc['latency_ms']}
        hd = headline(doc)
        rec['result'] = {'AP': hd, 'AP50': headline(doc, 'AP50'), 'n_frames': doc['n_frames'],
                         'note': doc['scores'].get('note'), 'synthetic_model': doc.get('synthetic_model'),
                         'dir': rel(self.ws, out_dir)}
        return [os.path.join(out_dir, 'evaluation.json'), os.path.join(out_dir, 'predictions.jsonl')]

    def step_evaluate(self, it, rec):
        evalset = ensure_evalset(self.project, it['odd'], self.svc, self.cancel, self.sub(0.0, 0.7), self.log)
        it['evalset'] = rel(self.ws, evalset_path(self.ws, it['odd_key']))
        save_iteration(self.ws, it)
        prev = _reusable_before(self.ws, it)
        evs = absp(self.ws, it['evalset'])
        if prev is not None:
            src = absp(self.ws, prev[1])
            rec['reused_from'] = {'iteration': prev[0], 'dir': prev[1]}
            ed = PV.read_json(os.path.join(src, 'evaluation.json'))
            rec['result'] = {'AP': headline(ed), 'AP50': headline(ed, 'AP50'), 'n_frames': ed['n_frames'],
                             'note': ed['scores'].get('note'), 'synthetic_model': ed.get('synthetic_model'),
                             'dir': prev[1]}
            rec['decisions'] = [f'reused iteration {prev[0]}\'s evaluation of this model (same file hash, same '
                                'evaluation set)']
            return [evs, os.path.join(src, 'evaluation.json'), os.path.join(src, 'predictions.jsonl')]
        out = self._evaluate(it, it['model'], self._file(it, 'before'), rec, 0.7)
        rec['decisions'] = [f'evaluation set {os.path.basename(evs)[:-5]}: {len(PV.read_json(evs)["drives"])} drives']
        return [evs] + out

    # -- 3
    def step_weak_cells(self, it, rec):
        from . import harness as H
        odd = it['odd']
        before = absp(self.ws, it['steps']['evaluate']['result']['dir'])
        ed = PV.read_json(os.path.join(before, 'evaluation.json'))
        frames = H.read_predictions(os.path.join(before, 'predictions.jsonl'))
        evalset = PV.read_json(absp(self.ws, it['evalset']))
        cell_of = {os.path.basename(absp(self.ws, d['run'])): d['cell'] for d in evalset['drives']}
        classes = [c for c in odd['classes'] if c in H.model_classes(it['model'])]
        if not classes:
            raise Stopped(f'the model names none of the ODD\'s classes ({", ".join(odd["classes"])})')
        self.progress(0.1, 'AP per class × range bin × condition (bootstrap over drives)')
        rows = cell_table(frames, cell_of, classes, odd['range_bins'], int(odd['evaluation']['n_boot']),
                          int(odd['evaluation']['seed']))
        overall = headline(ed)['value']
        wcs = odd['weak_cells']
        chosen = rank_weak(rows, overall, wcs['min_objects'], wcs['max_cells'], wcs.get('fallback', 'none'))
        fb = any(c.get('fallback') for c in chosen)
        doc = {'schema': 'falcon.loop_weak_cells/1', 'n': it['n'], 'rule': WEAK_RULE,
               'min_objects': odd['weak_cells']['min_objects'], 'max_cells': odd['weak_cells']['max_cells'],
               'overall_AP': headline(ed), 'classes': classes, 'range_bins': odd['range_bins'],
               'note': ed['scores'].get('note'), 'rows': rows, 'chosen': chosen, 'fallback_used': fb,
               'fallback': wcs.get('fallback', 'none'),
               'provenance': PV.block({'evaluation': os.path.join(before, 'evaluation.json'),
                                       'predictions': os.path.join(before, 'predictions.jsonl')},
                                      int(odd['evaluation']['seed']), {'n_boot': odd['evaluation']['n_boot']})}
        p = self._file(it, 'weak_cells.json')
        PV.write_json(p, doc)
        rec['result'] = {'chosen': [_cell_name(c) for c in chosen], 'n_rows': len(rows)}
        rec['decisions'] = ([f'{"lowest (fallback: no cell is weak by the rule)" if fb else "weak"}: {_cell_name(c)} '
                             f'— AP {_f(c["AP"])} [{_f(c["ci_low"])}, {_f(c["ci_high"])}], overall {_f(overall)}, '
                             f'{c["n_objects"]} objects' for c in chosen] or
                            [f'no cell is weak by the rule (heuristic; overall AP {_f(overall)}): nothing to search or '
                             'generate. A model uniformly weak or strong has no weak cell; the ODD\'s '
                             'weak_cells.fallback "lowest" targets its lowest cells instead'])
        return [p]

    # -- 4
    def step_search(self, it, rec):
        odd = it['odd']
        se = odd['search']
        wc = PV.read_json(self._file(it, 'weak_cells.json'))
        rig = self.project.item('rigs', odd['rig'])
        Study = _study_class()
        out_cells, outs, decisions = [], [], []
        chosen = wc['chosen']
        for k, cell in enumerate(chosen):
            _check(self.cancel)
            cond = search_condition(cell, wc['rows'], odd)
            weather = sample_weather(cond['weather'], int(se['seed']) * 1000 + it['n'])
            entry = {'cell': _cell_name(cell), 'class': cell['class'], 'bin': cell['bin'],
                     'condition': cond['label'], 'world': cond['world'], 'weather': weather, 'scenario': None,
                     'tried': [], 'objective': OBJECTIVE_NOTE}
            cands = SCENARIOS_FOR_CLASS.get(cell['class']) or []
            if not cands:
                entry['note'] = f'no built-in scenario for the class {cell["class"]}: the targeted dataset alone ' \
                                'serves this cell'
            for key, fixed in cands:
                scn = scenario_for(cell['class'], fixed, key)
                name = f'loop-{it["n"]:03d}-{k + 1}-{key}'
                st = Study(self.svc, self.project, scn, name, cond['world'], odd['rig'],
                           int(se['seed']) + it['n'], weather, unlabelled_rig(rig), se['fps'],
                           int(se['traffic_vehicles']))
                prog = self.sub(k / len(chosen), (k + 1) / len(chosen))
                try:
                    res = st.search(se['method'], int(se['budget']), int(se['batch']), float(se['threshold']),
                                    lambda n_: prog(n_ / max(int(se['budget']), 1),
                                                    f'{key}: {n_} of {se["budget"]} runs'))
                except RuntimeError as e:
                    entry['tried'].append({'scenario': key, 'error': str(e)})
                    self.log(f'{key} could not run in {cond["label"]}: {e}', 'warning')
                    continue
                by_values = {PV.canonical_json(r['values']): r for r in st.results()}
                fails = [dict(f, run=rel(self.ws, by_values[PV.canonical_json(f['params'])]['run']))
                         for f in res['failures'] if PV.canonical_json(f['params']) in by_values]
                ranked = sorted(res['history'], key=lambda h: (-h['value'], h['i']))
                cases = fails[:int(odd['generate']['max_failures'])]
                note = None
                if not cases:
                    cases = [dict(h, run=rel(self.ws, by_values[PV.canonical_json(h['params'])]['run']))
                             for h in ranked[:int(odd['generate']['max_failures'])]
                             if PV.canonical_json(h['params']) in by_values]
                    note = (f'no case reached the threshold {se["threshold"]}: the {len(cases)} most critical cases '
                            'are re-run instead')
                entry.update(scenario=key, fixed=fixed, study=rel(self.ws, st.dir), method=se['method'],
                             budget=int(se['budget']), evaluated=len(res['history']), seed=st.seed,
                             best=ranked[0]['value'] if ranked else None, failures=fails, rerun_cases=cases,
                             rerun_note=note)
                outs.append(os.path.join(st.dir, f'search-{se["method"]}.json'))
                decisions.append(f'{_cell_name(cell)}: {key} in {cond["label"]}, {len(res["history"])} runs, '
                                 f'{len(fails)} failing (criticality ≥ {se["threshold"]})')
                break
            else:
                if cands:
                    decisions.append(f'{_cell_name(cell)}: no matched scenario could run in {cond["label"]}')
            out_cells.append(entry)
        doc = {'schema': 'falcon.loop_search/1', 'n': it['n'], 'objective': OBJECTIVE_NOTE,
               'mapping': {k: [s for s, _ in v] for k, v in SCENARIOS_FOR_CLASS.items()}, 'cells': out_cells,
               'provenance': PV.block({'weak_cells': self._file(it, 'weak_cells.json')}, int(se['seed']), se)}
        p = self._file(it, 'search.json')
        PV.write_json(p, doc)
        rec['decisions'] = decisions or ['no weak cell: no search']
        rec['result'] = {'cells': len(out_cells), 'failures': sum(len(c.get('failures') or []) for c in out_cells)}
        return [p] + outs

    # -- 5
    def step_generate(self, it, rec):
        from ..sim import engine as E
        odd = it['odd']
        thr = self.svc.throughput()
        plan = targeted_plan(self.project, it, odd, thr)
        spec = plan['spec']
        evalset = PV.read_json(absp(self.ws, it['evalset']))
        eval_keys = {d['key'] for d in evalset['drives']}
        cfgs = _drive_cfgs(self.project, spec, plan, self.svc) if plan['drives'] else []
        clash = [c['out'] for c, _ in cfgs if c['cache_key'] in eval_keys]
        if clash:
            raise LoopError(f'refused: {len(clash)} planned drive(s) are evaluation drives (same cache key)')
        jobs = []
        if cfgs:
            sub = E.submit(self.svc, self.project, spec, plan)
            jobs = sub['queued']
            self.log(f'targeted dataset {spec["id"]}: {len(cfgs)} drives, {sub["cached"]} already recorded')
        _wait_jobs(self.svc, jobs, self.cancel, self.sub(0.0, 0.6), 'targeted drives')
        new_runs = [c['out'] for c, _ in cfgs]
        # the failing cases, re-run with the labelled rig
        sd = _search_doc(self.ws, it)
        rig = self.project.item('rigs', odd['rig'])
        Study = _study_class()
        reruns = []
        for k, cell in enumerate(sd['cells']):
            if not cell.get('scenario') or not cell.get('rerun_cases'):
                continue
            scn = scenario_for(cell['class'], cell.get('fixed') or {}, cell['scenario'])
            for j in range(int(odd['generate']['reruns_per_failure'])):
                _check(self.cancel)
                seed = int(cell['seed']) + 7919 * j
                name = f'loop-{it["n"]:03d}-{k + 1}-{cell["scenario"]}-labelled-s{seed}'
                st = Study(self.svc, self.project, scn, name, cell['world'], odd['rig'], seed, cell['weather'], rig,
                           odd['search']['fps'], int(odd['search']['traffic_vehicles']))
                res = st.sweep([c['params'] for c in cell['rerun_cases']])
                for c, r in zip(cell['rerun_cases'], res):
                    if not r or r.get('status') != 'ok':
                        raise LoopError(f're-running a failing case of {cell["scenario"]} failed: '
                                        f'{(r or {}).get("error")}')
                    if r['key'] in eval_keys:
                        raise LoopError('refused: a re-run is an evaluation drive (same cache key)')
                    reruns.append({'run': rel(self.ws, r['run']), 'scenario': cell['scenario'], 'seed': seed,
                                   'values': c['params'], 'criticality_at_search': c['value'],
                                   'criticality': r.get('objective'), 'same_seed_as_search': j == 0})
            self.progress(0.6 + 0.4 * (k + 1) / len(sd['cells']), 'failing cases re-run with labels')
        all_runs = new_runs + [absp(self.ws, r['run']) for r in reruns]
        frames = 0
        from .exporters import ExportError, Run
        for r in all_runs:
            try:
                run = Run(r)
            except ExportError as e:
                raise LoopError(str(e)) from None
            cam = next((c for c, s in run.cams.items() if s.get('type') == 'camera.pinhole'), None)
            frames += len([x for x in run.camera_steps(cam) if run.objects(x, cam) is not None]) if cam else 0
        doc = {'schema': 'falcon.loop_generate/1', 'n': it['n'], 'spec': spec,
               'plan': {k: plan.get(k) for k in ('needed', 'yield', 'yield_source', 'estimate', 'capped',
                                                   'per_drive_frames', 'unreachable')},
               'drives': [{'run': rel(self.ws, c['out']), 'key': c['cache_key'], 'cell': c['dataset']['cell']['label'],
                           'seed': c['seed']} for c, _ in cfgs],
               'reruns': reruns, 'runs': [rel(self.ws, r) for r in all_runs], 'frames': frames,
               'provenance': PV.block({'search': self._file(it, 'search.json')}, spec['seed'], {'spec': spec})}
        p = self._file(it, 'generate.json')
        PV.write_json(p, doc)
        rec['result'] = {'drives': len(cfgs), 'reruns': len(reruns), 'frames': frames}
        rec['decisions'] = [f'targeted dataset {spec["id"]}: {len(cfgs)} drive(s) for '
                            + (', '.join(f'{v} {c}' for c, v in spec['targets'].items()) or 'nothing')
                            + f' in the weak range bins (yield measured on the evaluation set)',
                            f'{len(reruns)} failing-case re-run(s) with labels']
        if plan.get('capped'):
            rec['decisions'].append(f'capped at max_drives = {spec["max_drives"]}')
        return [p] + [os.path.join(r, 'run.json') for r in all_runs]

    # -- 6
    def step_retrain(self, it, rec):
        from . import training as T
        from .jobs import FINAL
        odd = it['odd']
        m = it['model']
        tdir, run = training_run_of(m)
        prev = copy.deepcopy(run['provenance']['settings'])
        gen = PV.read_json(self._file(it, 'generate.json'))
        new = [{'kind': 'falcon', 'path': absp(self.ws, r)} for r in gen['runs']]
        evalset = PV.read_json(absp(self.ws, it['evalset']))
        assert_no_leak(list(prev.get('train') or []) + list(prev.get('val') or []) + list(prev.get('test') or [])
                       + new, evalset, self.ws)
        if not new:
            raise Stopped('the generate step made no new data: nothing to retrain on')
        cfg = {k: v for k, v in prev.items() if k not in ('id',)}
        cfg['train'] = list(prev.get('train') or []) + new
        cfg['export'] = True
        if odd['retrain'].get('epochs'):
            cfg['epochs'] = int(odd['retrain']['epochs'])
        cfg_path = self._file(it, 'retrain-config.json')
        PV.write_json(cfg_path, cfg)
        name = re.sub(r'[^A-Za-z0-9_.-]', '-', f'loop-{it["n"]:03d}-{m.get("id", "model")}')[:60]
        out = os.path.join(self.ws, 'training', name)
        argv = ['--config', cfg_path, '--out', out, '--workspace', self.ws]
        job = self.svc.queue.submit(T.job_spec(argv, f'Loop {it["n"]}: retrain {m.get("name")}'))
        prog = self.sub(0.0, 1.0)
        try:
            while job.state not in FINAL:
                self.svc.queue.poll()
                _check(self.cancel)
                prog(job.progress, job.text)
                time.sleep(0.2)
        except (Cancelled, KeyboardInterrupt):
            self.svc.queue.cancel(job)
            t0 = time.monotonic()
            while job.state not in FINAL and time.monotonic() - t0 < 60:
                self.svc.queue.poll()
                time.sleep(0.1)
            raise Cancelled() from None
        if job.state != 'done':
            err = ''
            try:
                err = PV.read_json(os.path.join(out, 'run.json')).get('error') or ''
            except (OSError, ValueError):
                err = '; '.join(t for lvl, t in job.log[-5:] if t)
            raise LoopError(f'the training {job.state}: {err}')
        reg = PV.read_json(os.path.join(out, 'model.json'))
        reg['name'] = f'{m.get("name")} + loop {it["n"]}'
        trun = PV.read_json(os.path.join(out, 'run.json'))
        t = trun.get('timings') or {}
        rec['measured'] = {'images_per_s': t.get('images_per_s'), 'peak_gpu_mib': t.get('peak_gpu_mib'),
                           'train_s': t.get('train_s'), 'device': (trun.get('device') or {}).get('type')}
        rec['result'] = {'registration': reg, 'training': rel(self.ws, out),
                         'best_val_AP': (trun.get('summary') or {}).get('best_val_AP'),
                         'onnx_verified': ((trun.get('summary') or {}).get('verification') or {}).get('ok')}
        rec['decisions'] = [f'continued from {rel(self.ws, tdir)}\'s settings ({prev.get("arch")}, init '
                            f'{prev.get("init")}, {cfg["epochs"]} epochs, seed {prev.get("seed")}) on its '
                            f'{len(prev.get("train") or [])} source(s) + {len(new)} new drive(s)',
                            'checked: no source is or holds an evaluation drive']
        return [cfg_path, os.path.join(out, 'run.json'), os.path.join(out, 'model.json'),
                os.path.join(out, 'model.onnx')]

    # -- 7
    def step_reevaluate(self, it, rec):
        reg = it['steps']['retrain']['result']['registration']
        return self._evaluate(it, reg, self._file(it, 'after'), rec)

    # -- 8
    def step_compare(self, it, rec):
        from . import harness as H
        odd = it['odd']
        a = absp(self.ws, it['steps']['evaluate']['result']['dir'])
        b = absp(self.ws, it['steps']['reevaluate']['result']['dir'])
        fa = H.read_predictions(os.path.join(a, 'predictions.jsonl'))
        fb = H.read_predictions(os.path.join(b, 'predictions.jsonl'))
        ea = PV.read_json(os.path.join(a, 'evaluation.json'))
        eb = PV.read_json(os.path.join(b, 'evaluation.json'))
        n_boot, seed = int(odd['evaluation']['n_boot']), int(odd['evaluation']['seed'])
        self.progress(0.1, 'paired bootstrap over drives')
        cmp = H.compare(fa, fb, 'detector', n_boot, seed)
        cmp.update(schema='falcon.comparison/1', a={'dir': a, 'model': ea['model']}, b={'dir': b, 'model': eb['model']},
                   provenance=PV.block({'a': os.path.join(a, 'evaluation.json'),
                                        'b': os.path.join(b, 'evaluation.json')}, seed))
        cdir = self._file(it, 'compare')
        os.makedirs(cdir, exist_ok=True)
        PV.write_json(os.path.join(cdir, 'comparison.json'), cmp)
        self.progress(0.5, 'the weak cells, before and after')
        wc = PV.read_json(self._file(it, 'weak_cells.json'))
        evalset = PV.read_json(absp(self.ws, it['evalset']))
        cell_of = {os.path.basename(absp(self.ws, d['run'])): d['cell'] for d in evalset['drives']}
        after_rows = cell_table(fb, cell_of, wc['classes'], odd['range_bins'], n_boot, seed)
        cells = compare_cells(fa, fb, cell_of, wc['chosen'], after_rows, odd['range_bins'], n_boot, seed)
        summary = summarise(ea, eb, cmp, cells)
        doc = {'schema': 'falcon.loop_cells_comparison/1', 'n': it['n'], 'cells': cells, 'summary': summary,
               'method': 'before and after on the same evaluation set: each number with its 95 % bootstrap interval '
                         'over drives; the difference with a paired bootstrap over drives (both resampled together)',
               'provenance': PV.block({'comparison': os.path.join(cdir, 'comparison.json')}, seed)}
        PV.write_json(os.path.join(cdir, 'cells.json'), doc)
        it['summary'] = summary
        rec['result'] = {'regressions': cmp['regressions'], 'improvements': cmp['improvements']}
        rec['decisions'] = [f'{r["label"]}: {_f(r["before"]["value"])} → {_f(r["after"]["value"])}, difference '
                            f'{_f(r["diff"]["value"], signed=True)} [{_f(r["diff"]["ci_low"], signed=True)}, '
                            f'{_f(r["diff"]["ci_high"], signed=True)}] — {r["verdict"]}' for r in summary['rows']]
        return [os.path.join(cdir, 'comparison.json'), os.path.join(cdir, 'cells.json')]


def _reusable_before(ws: str, it: dict):
    """(iteration, dir) of an earlier finished iteration whose step 7 evaluated this model file on this evaluation
    set, or None."""
    for old in reversed(iterations(ws)):
        if old['n'] >= it['n'] or old.get('odd_key') != it['odd_key']:
            continue
        rec = old['steps'].get('reevaluate') or {}
        if rec.get('state') == 'done' and (rec.get('measured') or {}).get('model_sha256') == it['model_sha256'] and \
                _outputs_ok(ws, rec.get('outputs')):
            return old['n'], rec['result']['dir']
    return None


def _cell_name(c: dict) -> str:
    cond = 'all conditions' if c['condition'] == ALL else c['condition']
    return f'{c["class"]}, {bin_label(c["bin"][0], c["bin"][1] if c["bin"][1] is not None else math.inf)}, {cond}'


def _f(v, signed=False) -> str:
    if v is None:
        return '—'
    return f'{v:+.3f}' if signed else f'{v:.3f}'


def compare_cells(fa, fb, cell_of, chosen, after_rows, range_bins, n_boot, seed) -> list:
    """For each chosen weak cell: before (its row), after (the same cell of the after evaluation) and the paired
    bootstrap of the difference over drives."""
    from . import metrics as M
    edges = _edges(range_bins)
    by_a = {f['image_id']: f for f in fa}
    out = []
    groups = {}
    for c in chosen:
        groups.setdefault(c['condition'], []).append(c)
    for cond, cs in groups.items():
        items = [{'image_id': f['image_id'], 'drive': f['drive'], 'a': by_a[f['image_id']], 'b': f} for f in fb
                 if f['image_id'] in by_a and (cond == ALL or cell_of.get(f['drive']) == cond)]
        res = {}
        if items:
            ra = {r['image_id']: r for r in M.coco_match([it['a'] for it in items], breakdowns={'range_m': edges})}
            rb = {r['image_id']: r for r in M.coco_match([it['b'] for it in items], breakdowns={'range_m': edges})}
            for it in items:
                it['ra'], it['rb'] = ra[it['image_id']], rb[it['image_id']]
            fn = _cell_metric({c['class'] for c in cs}, edges)
            unit = 'drive' if len({it['drive'] for it in items}) > 1 else 'image'
            res = M.paired_bootstrap(lambda its, w: fn([x['r' + w] for x in its]), items, 'a', 'b', n=int(n_boot),
                                     seed=int(seed), unit=unit).get('metrics') or {}
        for c in cs:
            k = _cell_key(c['class'], c['bin'][0])
            aft = next((r for r in after_rows if r['class'] == c['class'] and r['bin'][0] == c['bin'][0]
                        and r['condition'] == c['condition']), {})
            d = res.get(k) or {}
            out.append({'cell': _cell_name(c), 'class': c['class'], 'bin': c['bin'], 'condition': c['condition'],
                        'before': {'value': c['AP'], 'ci_low': c['ci_low'], 'ci_high': c['ci_high'],
                                   'n_objects': c['n_objects']},
                        'after': {'value': aft.get('AP'), 'ci_low': aft.get('ci_low'), 'ci_high': aft.get('ci_high'),
                                  'n_objects': aft.get('n_objects')},
                        'diff': {'value': d.get('diff'), 'ci_low': d.get('ci_low'), 'ci_high': d.get('ci_high'),
                                 'p_b_gt_a': d.get('p_b_gt_a')}})
    return out


def verdict(diff: dict) -> str:
    lo, hi = diff.get('ci_low'), diff.get('ci_high')
    if lo is not None and lo > 0:
        return 'improvement'
    if hi is not None and hi < 0:
        return 'regression'
    return 'no detectable change at 95 %'


def summarise(ea: dict, eb: dict, cmp: dict, cells: list) -> dict:
    """The before and after rows a chart and a report show: AP and AP50 over everything, then each weak cell."""
    rows = []
    by = {r['metric']: r for r in cmp['rows']}
    for metric in ('AP', 'AP50'):
        d = by.get(metric) or {}
        diff = {'value': d.get('diff'), 'ci_low': d.get('ci_low'), 'ci_high': d.get('ci_high')}
        rows.append({'label': f'{metric}, all', 'kind': 'headline', 'before': headline(ea, metric),
                     'after': headline(eb, metric), 'diff': diff, 'verdict': verdict(diff)})
    for c in cells:
        rows.append({'label': f'AP, {c["cell"]}', 'kind': 'weak cell', 'before': c['before'], 'after': c['after'],
                     'diff': c['diff'], 'verdict': verdict(c['diff'])})
    return {'rows': rows, 'unit': cmp.get('unit'), 'n_units': cmp.get('n_units'), 'n_frames': cmp.get('n_frames'),
            'note': ea['scores'].get('note'), 'regressions': cmp['regressions'],
            'improvements': cmp['improvements']}


def describe_estimate(e: dict | None) -> str:
    """An estimate in words for the page and the CLI."""
    from .budget import fmt_duration
    if not e:
        return 'not estimated'
    if e.get('seconds') is None:
        return f'not estimated — {e.get("source", "")}'
    s = f'about {fmt_duration(e["seconds"])} ({e.get("label")})'
    mem = e.get('memory')
    if mem and mem.get('peak_gpu_mib'):
        s += f', peak GPU memory about {mem["peak_gpu_mib"]:.0f} MiB ({mem["label"]})'
    return s + (f' — {e["source"]}' if e.get('source') else '')


# ================================================================================================ the project
def iteration_entry(ws: str, it: dict) -> dict:
    """The project's loop.iterations entry of an iteration record (a summary; the record is the truth)."""
    rows = {r['label']: r for r in (it.get('summary') or {}).get('rows', [])}
    ap = rows.get('AP, all')
    reg = ((it['steps'].get('retrain') or {}).get('result') or {}).get('registration')
    return {'n': it['n'], 'path': rel(ws, os.path.join(iteration_dir(ws, it['n']), 'iteration.json')),
            'status': it['status'], 'created': it['created'], 'updated': it.get('updated'),
            'model': it['model'].get('id'), 'model_after': reg.get('id') if reg else None, 'odd': it['odd']['id'],
            'odd_sha256': it['odd_sha256'], 'stopped': it.get('stopped'),
            'AP_before': ap['before'] if ap else (it['steps']['evaluate'].get('result') or {}).get('AP'),
            'AP_after': ap['after'] if ap else None, 'AP_diff': ap['diff'] if ap else None}


def sync_edits(project) -> list:
    """The edits that bring the project up to date with the workspace's iteration records: loop.iterations, and the
    retrained models and targeted dataset specs the loop made (added when missing)."""
    from .project import Edit
    ws = project.workspace
    its = iterations(ws)
    edits = []
    entries = [iteration_entry(ws, it) for it in its]
    if (project.doc.get('loop') or {}).get('iterations') != entries:
        edits.append(Edit('set', ['loop', 'iterations'], entries, text='Record the loop\'s iterations'))
    have_models = {m['id'] for m in project.doc.get('models', [])}
    have_specs = {s['id'] for s in project.doc.get('dataset_specs', [])}
    for it in its:
        reg = ((it['steps'].get('retrain') or {}).get('result') or {}).get('registration')
        if reg and reg.get('id') not in have_models and it['steps']['retrain'].get('state') == 'done':
            edits.append(Edit('insert', ['models'], reg, text=f'Register {reg["name"]}'))
            have_models.add(reg['id'])
        gp = os.path.join(iteration_dir(ws, it['n']), 'generate.json')
        if it['steps'].get('generate', {}).get('state') == 'done' and os.path.isfile(gp):
            spec = {k: v for k, v in PV.read_json(gp)['spec'].items() if k != 'cells'}
            if spec['id'] not in have_specs:
                edits.append(Edit('insert', ['dataset_specs'], spec, text=f'Add the dataset spec {spec["name"]}'))
                have_specs.add(spec['id'])
    return edits


# ================================================================================================ the command
def _print_status(project, out=print):
    ws = project.workspace
    its = iterations(ws)
    loop = project.doc.get('loop') or {}
    odd, errs = normalise_odd(loop.get('odd'), project)
    out(f'model: {loop.get("model") or "not chosen"}; ODD: {(odd or {}).get("name") or "not set"}')
    for e in errs:
        out(f'  ODD: {e}')
    if not its:
        out('no iteration yet; the next step is 1 (Pick the model and the ODD)')
        return
    it = its[-1]
    out(f'iteration {it["n"]} ({it["status"]}): model {it["model"].get("name")}, ODD {it["odd"]["name"]}')
    for k, s in enumerate(STEPS, 1):
        rec = it['steps'][s]
        state = 'done' if step_done(ws, it, s) else rec.get('state', 'pending')
        if rec.get('state') == 'done' and state == 'done':
            pass
        elif rec.get('state') == 'done':
            state = 'stale (outputs changed: will run again)'
        meas = (rec.get('measured') or {}).get('wall_s')
        out(f'  {k}. {STEP_TITLES[s]:28s} {state:10s}' + (f' measured {meas:.1f} s' if meas is not None else '')
            + (f' — {rec["error"]}' if rec.get('error') else ''))
    if it.get('stopped'):
        out(f'stopped: {it["stopped"]}')


def main_loop(argv) -> int:
    """`falcon loop PROJECT [status | next | run [--until STEP] | show N | new | set --model ID --odd FILE|example]`:
    protocol lines on stdout for next and run; 0 ok, 2 failed, 3 cancelled, 6 stopped (refused)."""
    import argparse
    import signal
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon loop', description='the closed loop (docs/loop.md)')
    ap.add_argument('project')
    ap.add_argument('action', nargs='?', default='status', choices=('status', 'next', 'run', 'show', 'new', 'set'))
    ap.add_argument('n', nargs='?', type=int, help='show: the iteration')
    ap.add_argument('--until', default='compare', help='run: the last step, 1–8 or its name (default 8, compare)')
    ap.add_argument('--model', help='set: the model id to start iterations with')
    ap.add_argument('--odd', help='set: an ODD JSON file, or "example" for the Bengaluru monsoon-dusk ODD')
    ap.add_argument('--json', action='store_true', help='show: the whole record as JSON')
    ap.add_argument('--no-save', action='store_true', help='do not write the project file (the GUI records the '
                                                           'iterations itself)')
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

    def save_synced():
        if a.no_save:
            return
        fresh = Project.load(a.project)           # the file as it is now, so nothing written meanwhile is lost
        for e in sync_edits(fresh):
            fresh.apply(e)
        if fresh.dirty:
            fresh.save()

    if a.action == 'status':
        _print_status(proj)
        it = (iterations(ws) or [None])[-1]
        nxt = next_step(ws, it) if it is not None and it['status'] in ('running', 'failed', 'cancelled') else 'pick'
        if nxt:
            from .budget import Throughput
            from .settings import Settings
            thr = Throughput.from_settings(Settings().get('measured'))
            est = estimate(proj, it if nxt != 'pick' else None, nxt, thr)
            print(f'next: step {STEPS.index(nxt) + 1} ({STEP_TITLES[nxt]}), {describe_estimate(est)}')
        return P.OK
    if a.action == 'show':
        try:
            it = load_iteration(ws, a.n if a.n is not None else iterations(ws)[-1]['n'])
        except (OSError, ValueError, IndexError):
            print(f'failed: no iteration {a.n}', flush=True)
            return P.FAILED
        if a.json:
            print(json.dumps(it, indent=2, ensure_ascii=False))
            return P.OK
        print(f'iteration {it["n"]} ({it["status"]}), {it["created"]}: model {it["model"].get("name")} '
              f'(sha256 {it["model_sha256"][:16]}…), ODD {it["odd"]["name"]} (sha256 {it["odd_sha256"][:16]}…)')
        for k, s in enumerate(STEPS, 1):
            rec = it['steps'][s]
            print(f'  {k}. {STEP_TITLES[s]} — {rec.get("state")}; estimated {describe_estimate(rec.get("estimate"))}'
                  + (f'; measured {rec["measured"]["wall_s"]:.1f} s' if (rec.get('measured') or {}).get('wall_s')
                     is not None else ''))
            for d in rec.get('decisions') or ():
                print(f'       {d}')
            if rec.get('error'):
                print(f'       {rec["error"]}')
        return P.OK
    if a.action == 'set':
        if a.model:
            if proj.item('models', a.model) is None:
                print(f'failed: no model {a.model!r} in the project', flush=True)
                return P.FAILED
            proj.doc['loop']['model'] = a.model
        if a.odd:
            if a.odd == 'example':
                odd, missing = example_odd(proj)
                for x in missing:
                    print(f'the example needs {x}')
            else:
                try:
                    odd = PV.read_json(a.odd)
                except (OSError, ValueError) as e:
                    print(f'failed: {e}', flush=True)
                    return P.FAILED
            proj.doc['loop']['odd'] = odd
        _, errs = normalise_odd(proj.doc['loop'].get('odd'), proj)
        for e in errs:
            print(f'ODD: {e}')
        proj.save()
        return P.OK
    # next, run, new
    from .jobs import JobQueue, SubprocessLauncher
    from .settings import Settings
    from ..sim.service import CarlaService
    if a.action == 'run' and a.until not in STEPS:
        try:
            a.until = STEPS[int(a.until) - 1]
        except (ValueError, IndexError):
            print(f'failed: --until is 1–8 or one of {", ".join(STEPS)}', flush=True)
            return P.FAILED
    em = P.Emitter()
    em.hello('loop')
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
    r = Runner(proj, svc, em, lambda: state['cancel'])
    code = P.OK
    try:
        if a.action == 'new':
            it = r.new_iteration()
            em.log(f'iteration {it["n"]} started: model {it["model"].get("name")}, ODD {it["odd"]["name"]}')
            it = r.run('pick', it=it)
        else:
            it = r.run(a.until if a.action == 'run' else 'compare', one=a.action == 'next')
        em.result('ok', iteration=it['n'], iteration_status=it['status'],
                  next=next_step(ws, it), path=os.path.join(iteration_dir(ws, it['n']), 'iteration.json'))
    except Stopped as e:
        print(f'stopped: {e}', flush=True)
        em.result('stopped', error=str(e))
        code = P.REFUSED
    except Cancelled:
        em.result('cancelled')
        code = P.CANCELLED
    except (LoopError, OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        code = P.FAILED
    finally:
        state['cancel'] = True                   # a late signal no longer raises (the shutdown cancels what runs)
        svc.shutdown()
        try:
            save_synced()
        except (OSError, ProjectError) as e:
            print(f'warning: the project file was not updated: {e}', flush=True)
        signal.signal(signal.SIGTERM, old[0])
        signal.signal(signal.SIGINT, old[1])
    return code
