"""Model-in-the-loop rare-case search: a perception objective for scenario searches (numpy at import; the model's
runtime, OpenCV and the harness inside the functions). docs/rare-cases-mil.md is the manual.

A rare-case search (sim.studio.Study, core.search) maximises an objective over a scenario's parameters. The default
objective is the run's criticality (core.criticality). This module adds:

- "perception": each evaluation runs the scenario with the labelled rig (its camera records boxes2d), then — after the
  drive, in the app's Python — runs a registered model (core.models.Runner through core.harness.collect) on the
  recorded frames and scores how the model failed on the objects that matter (below);
- "both": perception failure AND criticality, min(perception, criticality): a case is a combined failure exactly
  when both reach the threshold (the fuzzy "and"; the product would put two 0.7s under a 0.5 threshold).

The objects that matter (RELEVANT): the scenario's actors (its critical actors: the director reports each one's CARLA
id, which the labels' track ids are), and objects of the target classes (spec target_classes) whose range lies in
spec range_m [lo, hi). Only ground truth the harness scores counts: an object the model cannot name, with fewer than
harness.MIN_PIXELS visible pixels or more than harness.MAX_OCCLUDED of its box hidden is not "present" in a frame.

Matching (per frame, as the harness's failure gallery): per class, detections scoring at least spec min_score
(default 0.5) in score order, each taking the unmatched ground-truth box of its class with the highest IoU ≥ spec iou
(default 0.5), present objects before ignored ones. An object is DETECTED in a frame when a detection took it. A
detection of the wrong class is a miss.

Frame times: frame t = (step + 1) / fps — the time of the trajectories the criticality is computed on (a step's
camera frame is of the tick that ends it).

Per critical actor a (those present before their critical moment):
- critical moment t_c: its collision time; else the time of its minimum time to collision; else of its minimum
  distance (core.criticality's per-actor summary);
- window: [t_c − window_s, t_c] (default 3 s); window_recall = detected / present frames in it; window_miss = 1 −
  window_recall (None when it is not present in the window: hidden then, so not asked);
- first visible t_v: its first present frame at or before t_c; first detected t_d: its first detected frame in
  [t_v, t_c]; detection delay t_d − t_v (the worst-case time to first detection; None when never detected by t_c);
- lost lead: needed = min(lead_s, t_c − t_v + Δ) with Δ the frame period (the warning time it could have given, up to
  lead_s, default 3 s); lead = t_c − t_d + Δ (0 when never detected); lost_lead = clip(1 − lead / needed, 0, 1) —
  the share of the warning time the model could have given that it did not;
- failure_a = max(window_miss, lost_lead): the worse of the two.

Over the run: window_miss and lost_lead are their maximum over the critical actors (the worst-perceived actor);
miss_rate = 1 − detected / present object-frames over every relevant object and frame; the composite PERCEPTION
FAILURE (heuristic) = max over critical actors of failure_a, or miss_rate when no critical actor was present before
its critical moment, or 0 (said so: "nothing the model could be asked to see") when no relevant object was present.
spec metric picks which of perception (default), window_miss, lost_lead or miss_rate is the search's value; all of
them are recorded. Everything lies in [0, 1], higher = worse, so a search maximises it and failure ≥ threshold reads as
for the criticality.

The GPU. The CARLA worker records; the model runs after the batch's drives, when the queue has released the CARLA
lock. An ONNX or TorchScript model takes the CARLA and GPU locks (core.training.gpu_locks) while it runs, so a CARLA
job and a GPU job never run at full load together; a Python plug-in (the synthetic detector) needs no GPU and no lock.
The model is loaded once per scoring pass and released after it.

The search space can be widened with weather fields ("weather.<field>", WEATHER_RANGES): the synthetic detector's
planted weakness is in fog, which no scenario parameter sets. A value "weather.fog_density" overrides that field of
the run's weather (the world's, or the loop cell's); the scenario's own parameters are unchanged.

Limits. The composite is a heuristic, not a standard; recall over a few frames is coarse (a 3 s window at 20 fps is 60
frames); the critical moment comes from the criticality of the ego's run, so a scenario whose critical actor never
nears the ego gets its moment at the closest approach; objects the model cannot name are not scored.
"""
from __future__ import annotations

import copy
import os

SCHEMA = 'falcon.perception_score/1'
OBJECTIVES = ('criticality', 'perception', 'both')
METRICS = ('perception', 'window_miss', 'lost_lead', 'miss_rate')
DEFAULTS = {'objective': 'criticality', 'metric': 'perception', 'iou': 0.5, 'min_score': 0.5, 'window_s': 3.0,
            'lead_s': 3.0, 'target_classes': [], 'range_m': [0.0, 50.0], 'every': 1, 'camera': None,
            'count': 'avoidable', 'feasibility': True}
COUNTS = ('avoidable', 'all')         # the criticality objective: avoidable failures only (core.rss), or any
WEATHER_PREFIX = 'weather.'
WEATHER_RANGES = {'fog_density': (0.0, 100.0), 'sun_altitude_angle': (-10.0, 90.0), 'precipitation': (0.0, 100.0),
                  'precipitation_deposits': (0.0, 100.0), 'wetness': (0.0, 100.0), 'cloudiness': (0.0, 100.0)}
DEFINITION = ('perception failure (heuristic) = max over the scenario\'s actors of max(window miss, lost lead): '
              'window miss = 1 − recall in [t_c − window_s, t_c]; lost lead = 1 − (t_c − first detection + Δ) / '
              'min(lead_s, t_c − first visible + Δ); t_c the actor\'s collision, minimum-TTC or minimum-distance time; '
              'detected = a detection of its class scoring ≥ min_score at IoU ≥ iou; else the miss rate over the '
              'relevant objects')
BOTH_DEFINITION = 'min(perception failure, criticality): a combined failure needs both to reach the threshold'


class MilError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ the objective spec
def normalise_spec(spec: dict | None) -> dict:
    """A validated objective spec: {'objective', 'model' (a registration, for perception and both), 'metric', 'iou',
    'min_score', 'window_s', 'lead_s', 'target_classes', 'range_m', 'every', 'camera'}."""
    s = copy.deepcopy(DEFAULTS)
    s.update({k: copy.deepcopy(v) for k, v in (spec or {}).items() if v is not None})
    if s['objective'] not in OBJECTIVES:
        raise MilError(f'objective must be one of {", ".join(OBJECTIVES)}, not {s["objective"]!r}')
    if s['metric'] not in METRICS:
        raise MilError(f'metric must be one of {", ".join(METRICS)}, not {s["metric"]!r}')
    for k in ('iou', 'min_score'):
        s[k] = float(s[k])
        if not 0.0 < s[k] <= 1.0:
            raise MilError(f'{k} must lie in (0, 1]')
    for k in ('window_s', 'lead_s'):
        s[k] = float(s[k])
        if not s[k] > 0:
            raise MilError(f'{k} must be positive')
    s['every'] = int(s['every'])
    if s['every'] < 1:
        raise MilError('every must be at least 1')
    lo, hi = (float(x) for x in s['range_m'])
    if not 0.0 <= lo < hi:
        raise MilError('range_m is [lo, hi) metres with 0 ≤ lo < hi')
    s['range_m'] = [lo, hi]
    s['target_classes'] = sorted({str(c) for c in s['target_classes'] or ()})
    if s['count'] not in COUNTS:
        raise MilError(f'count must be one of {", ".join(COUNTS)}, not {s["count"]!r}')
    s['feasibility'] = bool(s['feasibility'])
    if s['objective'] == 'criticality':
        s.pop('model', None)
    elif not isinstance(s.get('model'), dict):
        raise MilError(f'the {s["objective"]} objective needs a registered model (core.models)')
    return s


def needs_model(spec: dict) -> bool:
    return spec.get('objective', 'criticality') != 'criticality'


def spec_key(spec: dict, workspace: str = '') -> str:
    """What a scored value depends on: the objective's settings and the model's file (or registration) hash."""
    from . import provenance as PV
    s = normalise_spec(spec)
    if s['objective'] == 'criticality':
        return 'criticality' if s['count'] == 'all' else 'criticality-rss'
    s.pop('count')                    # RSS counting and feasibility act on the criticality objective only: a
    s.pop('feasibility')              # perception study's key (and its scores) stay as they were
    m = s.pop('model')
    s['model_sha256'] = model_fingerprint(m, workspace)
    return PV.sha256_json(s)[:16]


def model_fingerprint(m: dict, workspace: str) -> str:
    from .loop import model_fingerprint as fp
    return fp(m, workspace)


def evaluable(m: dict | None, workspace: str = '') -> str | None:
    """None when the model can be run for the perception objective, else why not (in words)."""
    import importlib.util
    if not isinstance(m, dict):
        return 'no model is chosen'
    if m.get('kind') != 'detector':
        return f'{m.get("name") or m.get("id")} is a {m.get("kind")}: the perception objective scores 2D detectors'
    from .models import resolve_path, validate
    bad = validate(m)
    if bad:
        return '; '.join(bad)
    rt = m['runtime']
    if rt == 'python':
        mod = m['entry'].partition(':')[0]
        try:
            ok = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            ok = False
        return None if ok else f'its plug-in module {mod!r} cannot be imported'
    if not os.path.isfile(resolve_path(m, workspace)):
        return f'its file {resolve_path(m, workspace)} is missing (download it on the Models page)'
    need = 'onnxruntime' if rt == 'onnx' else 'torch'
    if importlib.util.find_spec(need) is None:
        return f'{need} is not installed'
    return None


# ------------------------------------------------------------------------------------------------ weather dimensions
def weather_space(fields) -> list:
    """Search-space entries "weather.<field>" for CARLA weather fields: names (their WEATHER_RANGES), "name:lo:hi"
    strings, or a {name: [lo, hi]} mapping (a loop cell's ranges)."""
    from . import weather as W
    items = []
    if isinstance(fields, dict):
        items = [(k, v) for k, v in sorted(fields.items())]
    else:
        for f in fields or ():
            name, *rng = str(f).split(':')
            items.append((name, rng or None))
    out = []
    for name, rng in items:
        if name not in W.FIELDS:
            raise MilError(f'{name!r} is not a CARLA weather field ({", ".join(W.FIELDS)})')
        if rng:
            lo, hi = (float(x) for x in rng)
        elif name in WEATHER_RANGES:
            lo, hi = WEATHER_RANGES[name]
        else:
            raise MilError(f'give {name}\'s range as {name}:LO:HI')
        if not hi > lo:
            raise MilError(f'{name}: hi must be above lo')
        out.append({'name': WEATHER_PREFIX + name, 'lo': lo, 'hi': hi})
    return out


def split_values(values: dict) -> tuple:
    """(the scenario's parameter values, the weather overrides) of a search's parameter set."""
    scn, wx = {}, {}
    for k, v in (values or {}).items():
        if k.startswith(WEATHER_PREFIX):
            wx[k[len(WEATHER_PREFIX):]] = float(v)
        else:
            scn[k] = v
    return scn, wx


def apply_weather(cfg: dict, overrides: dict) -> dict:
    """cfg with the weather fields overridden (its weather_source spec updated and resolved again)."""
    from . import weather as W
    if overrides:
        spec = dict(cfg.get('weather_source') or {})
        spec.update(overrides)
        cfg['weather'] = W.resolve(spec)
        cfg['weather_source'] = spec
    return cfg


# ------------------------------------------------------------------------------------------------ scoring
def frame_hits(f: dict, iou: float = 0.5, min_score: float = 0.5) -> list:
    """Per ground-truth object of a harness frame record, whether a detection took it (module doc: Matching)."""
    from . import metrics as M
    gt, hits = f.get('gt') or [], [False] * len(f.get('gt') or [])
    dets = [d for d in f.get('det') or () if d['score'] >= min_score]
    for c in {g['class'] for g in gt}:
        gi = [j for j, g in enumerate(gt) if g['class'] == c]
        di = sorted((d for d in dets if d['class'] == c), key=lambda d: -d['score'])
        if not di:
            continue
        ious = M.box_iou([d['bbox'] for d in di], [gt[j]['bbox'] for j in gi])
        taken = [False] * len(gi)
        for k in range(len(di)):
            for want_ignored in (0, 1):                 # present objects first, as COCO matches
                best, bj = iou, -1
                for jj, j in enumerate(gi):
                    if not taken[jj] and int(gt[j].get('ignore', 0)) == want_ignored and ious[k, jj] >= best:
                        best, bj = ious[k, jj], jj
                if bj >= 0:
                    taken[bj] = True
                    hits[gi[bj]] = True
                    break
    return hits


def critical_moments(scenario: dict | None) -> dict:
    """{scenario actor id: {'t', 'why'}}: its collision time, else its minimum-TTC time, else its minimum-distance
    time (the run's criticality summary)."""
    out = {}
    for aid, a in (((scenario or {}).get('criticality') or {}).get('actors') or {}).items():
        if a.get('collision') and a.get('collision_t') is not None:
            out[aid] = {'t': float(a['collision_t']), 'why': 'collision'}
        elif a.get('min_ttc_s') is not None and a.get('min_ttc_t') is not None:
            out[aid] = {'t': float(a['min_ttc_t']), 'why': 'minimum time to collision'}
        elif a.get('min_distance_t') is not None:
            out[aid] = {'t': float(a['min_distance_t']), 'why': 'minimum distance'}
    return out


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def perception_score(frames: list, scenario: dict | None, fps: float, spec: dict | None = None) -> dict:
    """The perception failure of one scenario run from its harness frame records (module doc). Returns the chosen
    metric as 'value' and every measure, per actor too."""
    s = dict(DEFAULTS, **{k: v for k, v in (spec or {}).items() if v is not None})
    iou, min_score = float(s['iou']), float(s['min_score'])
    window_s, lead_s = float(s['window_s']), float(s['lead_s'])
    targets = set(s.get('target_classes') or ())
    lo, hi = (float(x) for x in s.get('range_m') or (0.0, 50.0))
    ids = {str(k): int(v) for k, v in ((scenario or {}).get('actor_ids') or {}).items()}
    by_track = {v: k for k, v in ids.items()}
    moments = critical_moments(scenario)
    frames = sorted(frames, key=lambda f: f['step'])
    steps = [f['step'] for f in frames]
    period = (min(b - a for a, b in zip(steps, steps[1:])) if len(steps) > 1 else 1) / float(fps)
    seen = {aid: [] for aid in ids}                     # actor -> [(t, hit)] over its present frames
    cls = {}
    tgt_present = tgt_hits = 0
    tgt_objects = set()
    for f in frames:
        t = (f['step'] + 1) / float(fps)
        hits = frame_hits(f, iou, min_score)
        for g, h in zip(f.get('gt') or (), hits):
            if int(g.get('ignore', 0)):
                continue
            tid = g.get('track_id')
            aid = by_track.get(tid) if isinstance(tid, int) else None
            if aid is not None:
                seen[aid].append((t, bool(h)))
                cls[aid] = g['class']
            elif g['class'] in targets and g.get('range_m') is not None and lo <= float(g['range_m']) < hi:
                tgt_present += 1
                tgt_hits += bool(h)
                tgt_objects.add(str(tid))
    actors = []
    for aid in sorted(ids):
        pres = seen[aid]
        mom = moments.get(aid)
        row = {'actor': aid, 'track_id': ids[aid], 'class': cls.get(aid), 'frames': len(pres),
               'hits': sum(h for _, h in pres), 't_critical': mom['t'] if mom else None,
               'why_critical': mom['why'] if mom else None, 'scored': False}
        actors.append(row)
        if mom is None:
            row['note'] = 'no criticality summary for this actor'
            continue
        tc = mom['t']
        before = [(t, h) for t, h in pres if t <= tc + 1e-9]
        if not before:
            row['note'] = ('never present (visible enough to be scored, of a class the model names) before its '
                           'critical moment')
            continue
        tv = before[0][0]
        td = next((t for t, h in before if h), None)
        win = [(t, h) for t, h in before if t >= tc - window_s - 1e-9]
        needed = min(lead_s, tc - tv + period)
        lead = (tc - td + period) if td is not None else 0.0
        lost = min(max(1.0 - lead / needed, 0.0), 1.0)
        wrec = (sum(h for _, h in win) / len(win)) if win else None
        wmiss = None if wrec is None else 1.0 - wrec
        row.update(scored=True, first_visible_t=_r(tv), first_detected_t=_r(td),
                   detection_delay_s=_r(td - tv) if td is not None else None, lead_s=_r(lead),
                   needed_lead_s=_r(needed), lost_lead=_r(lost, 6), window=[_r(tc - window_s), _r(tc)],
                   window_frames=len(win), window_recall=_r(wrec, 6), window_miss=_r(wmiss, 6),
                   failure=_r(max(lost, wmiss if wmiss is not None else 0.0), 6))
    scored = [a for a in actors if a['scored']]
    present = sum(a['frames'] for a in actors) + tgt_present
    detected = sum(a['hits'] for a in actors) + tgt_hits
    miss_rate = (1.0 - detected / present) if present else None
    wm = [a['window_miss'] for a in scored if a['window_miss'] is not None]
    out = {'schema': SCHEMA, 'metric': s['metric'], 'n_frames': len(frames), 'frame_period_s': _r(period, 6),
           'window_miss': max(wm) if wm else None, 'lost_lead': max((a['lost_lead'] for a in scored), default=None),
           'miss_rate': _r(miss_rate, 6), 'relevant_object_frames': present, 'detected_object_frames': detected,
           'targets': {'classes': sorted(targets), 'range_m': [lo, hi], 'objects': len(tgt_objects),
                       'frames': tgt_present, 'hits': tgt_hits},
           'actors': actors, 'worst_actor': None, 'note': None, 'definition': DEFINITION,
           'settings': {k: s[k] for k in ('iou', 'min_score', 'window_s', 'lead_s', 'target_classes', 'range_m')}}
    if scored:
        worst = max(scored, key=lambda a: (a['failure'], a['actor']))
        out['perception'] = worst['failure']
        out['worst_actor'] = worst['actor']
    elif miss_rate is not None:
        out['perception'] = _r(miss_rate, 6)
        out['note'] = 'no scenario actor was present before its critical moment: the miss rate of the relevant objects'
    else:
        out['perception'] = 0.0
        out['note'] = 'nothing the model could be asked to see: no relevant object was present in any frame'
    v = out.get(s['metric']) if s['metric'] != 'perception' else out['perception']
    if v is None:
        out['note'] = (out['note'] + '; ' if out['note'] else '') + f'{s["metric"]} undefined here: counted as 0'
        v = 0.0
    out['value'] = float(v)
    return out


def objective_value(spec: dict, criticality: float, perception: dict | None, rss: dict | None = None) -> float:
    """The search's value: the criticality (by default counting avoidable failures only: core.rss.rss_objective of
    the run's RSS label), the perception failure (its chosen metric), or min of both."""
    kind = spec.get('objective', 'criticality')
    if kind == 'criticality':
        if spec.get('count', 'avoidable') == 'avoidable':
            from .rss import rss_objective
            return rss_objective(criticality, rss)
        return float(criticality)
    if perception is None:
        raise MilError('the perception objective needs the run scored by the model')
    if kind == 'perception':
        return float(perception['value'])
    return float(min(perception['value'], criticality))


def describe(spec: dict) -> str:
    """The objective in words (study.json, the search's records and the reports)."""
    kind = spec.get('objective', 'criticality')
    if kind == 'criticality':
        base = ('the scenario\'s criticality (core.criticality; 1 for a collision): where driving the scenario becomes '
                'critical, not where a perception model fails')
        if spec.get('count', 'avoidable') == 'avoidable':
            from .rss import NON_AVOIDABLE_CAP
            return (base + f'; counting avoidable failures only — a run whose critical event the ego is not '
                    f'responsible for under RSS (core.rss: the other road user\'s, kinematically unavoidable, or not '
                    f'dangerous) is capped at {NON_AVOIDABLE_CAP:g}')
        return base + '; every failure counted (RSS labels shown, not applied)'
    m = spec.get('model') or {}
    what = f'{spec.get("metric", "perception")} of {m.get("name") or m.get("id")} ({DEFINITION})'
    if kind == 'perception':
        return 'perception failure: ' + what
    return f'both: {BOTH_DEFINITION}; perception: {what}'


def _fps(run_dir: str) -> float:
    from .provenance import read_json
    for name in ('dataset.json', 'drive.json'):
        try:
            v = read_json(os.path.join(run_dir, name)).get('fps')
        except (OSError, ValueError):
            continue
        if v:
            return float(v)
    return 20.0


def score_runs(runs: list, spec: dict, workspace: str = '', emit=None, cancel=None, meta_run: str | None = None,
               progress=None, keep_frames: bool = False) -> dict:
    """{run folder: perception_score(...)} for scenario runs recorded with labels, the model loaded once. An ONNX or
    TorchScript model holds the CARLA and GPU locks while it runs (module doc); a plug-in takes none. meta_run: the
    frame identity given to the model instead of the run's folder name (core.rootcause scores counterfactual re-runs
    as the case itself)."""
    from . import harness as H
    from .models import Runner
    from .provenance import read_json
    s = normalise_spec(spec)
    m = s['model']
    why = evaluable(m, workspace)
    if why:
        raise MilError(f'the model cannot be evaluated: {why}')
    locks = []
    out = {}
    try:
        if m.get('runtime') != 'python':
            from .training import gpu_locks
            locks = gpu_locks(emit, cancel)
        runner = Runner(m, workspace)
        for k, r in enumerate(runs):
            try:
                col = H.collect([r], m, workspace, s.get('camera'), 1, cancel=cancel, runner=runner,
                                meta_run=meta_run)
            except H.HarnessError as e:
                raise MilError(f'{os.path.basename(r)}: {e}') from None
            run = read_json(os.path.join(r, 'run.json'))
            sc = perception_score(col['frames'], run.get('scenario'), _fps(r), s)
            sc.update(providers=col['providers'], synthetic_model=col['synthetic'],
                      latency_ms_median=_r(col['latency_ms']['median'], 3), camera=(col['cameras'] or [None])[0])
            if keep_frames:
                sc['frames'] = col['frames']
            out[r] = sc
            if progress:
                progress((k + 1) / len(runs))
    finally:
        for lk in locks:
            lk.release()
    return out


def check_rig(rig: dict, camera: str | None = None) -> str | None:
    """None when the rig records what the perception objective scores (a pinhole camera with boxes2d), else why."""
    cams = [x for x in (rig or {}).get('sensors', []) if x.get('type') == 'camera.pinhole']
    cam = next((x for x in cams if x['id'] == camera), None) if camera else (cams[0] if cams else None)
    if cam is None:
        return f'the rig {(rig or {}).get("name")!r} has no pinhole camera' + (f' {camera!r}' if camera else '')
    if 'boxes2d' not in (cam.get('labels') or []):
        return (f'the rig {rig.get("name")!r}: its camera {cam["id"]!r} must record the boxes2d label (the model is '
                'scored against its objects)')
    return None
