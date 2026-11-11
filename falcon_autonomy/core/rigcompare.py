"""Rig variants compared by perception (numpy at import): what a detector finds around the vehicle with each rig, on
the same scenario runs, with intervals and a paired test (docs/rig-compare.md). The recording is sim.rigstudy's; this
module scores what was recorded, so it runs in any process and re-scores a study without driving it again.

Every rig records the same scenarios with the same seeds, in kinematic motion and without background traffic: the
road users are at the same place at every step whichever rig looks at them, so the variants' numbers pair object by
object. A registered detector (core.models, run by core.harness.collect) sees every camera's frames; its detections
are matched to the frame's ground truth per class at IoU 0.5 among those scoring at least the threshold (0.5), greedy
in score order, as COCO matches; a detection on an *ignored* object (fewer than 20 visible pixels or more than 90 %
hidden: core.harness's rule) is neither a hit nor a false positive.

The numbers (each a ratio of counts; every count is summed per unit, then over units):
* precision — matched detections / detections, over every camera's frames;
* camera recall — matched objects / scored objects, over every camera's frames (each camera's view of an object counts);
* the road users around the vehicle — every vehicle and walker of actors.jsonl at each recorded step whose centre is
  within `radius` (8 m) of the CARLA car's box — are the objects of the rig-level numbers:
  - seen: some camera scores it (it is not ignored there); recall: some camera detects it (blind spots count as
    misses); recall given seen: detected / seen;
  - stitched recall (rigs from Vantage): the camera whose image the stitched bird's-eye view shows at the object's
    azimuth — the owner of its sector between the variant's seams (core.vantagerig) — detects it. Falcon renders no
    stitched image: core.lens's seam-blend tables feather the cube faces *inside one fisheye*, not a surround view's
    cameras. A detector run on Vantage's stitched image could differ.
  - by sector (8 × 45° about the body centre, Vantage's azimuth: counter-clockwise from ahead), by clearance from the
    car's box (0–1, 1–2, 2–4, 4–8 m), and in a band of ±`band_deg` (10°) about each seam of every rig compared — so
    two variants are compared on the same objects near the seams of either.
* AP (COCO, every camera's frames pooled) with core.harness's drive-level interval.

Units and intervals. A unit is one scenario run (scenario × seed) when there are at least MIN_UNITS of them, else a
block of `block_s` (2 s) of one run: the 95 % interval is a cluster bootstrap over units (core.metrics.bootstrap),
and blocks of one run are correlated, so with blocks the interval is too narrow and the report says so. A paired
comparison of B against A counts only the objects present in both runs (matched by step, actor type and position to
5 cm) and gives B − A with its paired bootstrap interval (core.metrics.paired_bootstrap, the same units resampled for
both). The verdict is not that interval's: with the handful of runs a study has, the percentile bootstrap claims a
difference too often (measured on null data, 6 runs of 60 objects: 11 % of the time at a nominal 5 %), and a pair
has some forty numbers. It is a cluster-robust paired t-test over the units (cluster_test: the delta-method CR1
standard error of the difference of the two ratios, units − 1 degrees of freedom; 3.8 % on the same null data, 92 %
power against a 0.15 drop) with Holm's adjustment over all the pair's numbers: worse or better when the adjusted p is
below 0.05, else "no detectable change at 95 %", which is not "no change". Holm over nested numbers (a sector and the
seam band inside it) is conservative.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time

import numpy as np

from . import metrics as M

SCHEMA = 'falcon.rig_comparison/1'
STUDY_SCHEMA = 'falcon.rig_study/1'
RADIUS_M = 8.0
BAND_DEG = 10.0
SCORE_THRESHOLD = 0.5
IOU = 0.5
MIN_UNITS = 5
BLOCK_S = 2.0
CLEARANCE_BINS = (0.0, 1.0, 2.0, 4.0, 8.0)
HEADLINE = ('precision', 'camera_recall', 'seen', 'recall', 'recall_seen', 'stitched_recall')
NOTE_TEXT_MAX = 400           # Vantage's design-log entry text limit (vantage.assistant.memory.TEXT_MAX)
CATEGORY_CLASS = (('human.pedestrian', 'pedestrian'), ('vehicle.bicycle', 'bicycle'),
                  ('vehicle.motorcycle', 'motorcycle'), ('vehicle.bus', 'bus'), ('vehicle.truck', 'truck'),
                  ('vehicle.construction', 'truck'), ('vehicle.trailer', 'truck'), ('vehicle.emergency', 'car'),
                  ('vehicle.car', 'car'))

# scenarios for a surround view: road users close round a stopped or creeping car, crossing every seam. Falcon's
# scenario format (core.scenarios); the director runs them kinematically.
SVM_SCENARIOS = {
    'svm_children_round': {
        'id': 'svm_children_round', 'name': 'Children close round a stopped car',
        'about': 'The ego stands still. A child crosses just ahead of the bumper, another just behind the car, and a '
                 'cyclist and a scooter pass along both sides about a metre off the body: every seam of a surround '
                 'view is crossed.',
        'needs': [], 'duration_s': 10.0,
        'params': {'front': {'dist': 'uniform', 'lo': 3.2, 'hi': 4.2, 'unit': 'm',
                             'about': 'the front child\'s path ahead of the ego\'s origin'},
                   'rear': {'dist': 'uniform', 'lo': 3.0, 'hi': 4.0, 'unit': 'm',
                            'about': 'the rear child\'s path behind the ego\'s origin'},
                   'walk': {'dist': 'uniform', 'lo': 0.9, 'hi': 1.4, 'unit': 'm/s', 'about': 'the children\'s pace'},
                   'side': {'dist': 'uniform', 'lo': 1.6, 'hi': 2.0, 'unit': 'm',
                            'about': 'the two-wheelers\' offset from the ego\'s lane centre'},
                   'ride': {'dist': 'uniform', 'lo': 8.0, 'hi': 14.0, 'unit': 'km/h',
                            'about': 'the two-wheelers\' speed'}},
        'ego': {'speed_kmh': 0, 'driver': 'none', 'route': 'straight'},
        'actors': [{'id': 'child_front', 'kind': 'child', 'place': {'lane': 'kerb', 'ahead_m': '$front'},
                    'heading': 'across', 'speed_kmh': 0, 'start': 'stopped'},
                   {'id': 'child_rear', 'kind': 'child', 'place': {'lane': 'kerb', 'ahead_m': '-$rear'},
                    'heading': 'across', 'speed_kmh': 0, 'start': 'stopped'},
                   {'id': 'cyclist', 'kind': 'bicycle', 'place': {'lane': 'ego', 'ahead_m': -14.0,
                                                                  'lateral_m': '$side'},
                    'heading': 'lane', 'speed_kmh': '$ride', 'start': 'moving'},
                   {'id': 'scooter', 'kind': 'motorcycle', 'place': {'lane': 'ego', 'ahead_m': -18.0,
                                                                     'lateral_m': '-$side'},
                    'heading': 'lane', 'speed_kmh': '$ride', 'start': 'moving'}],
        'timeline': [{'trigger': {'type': 'time', 't_s': 0.5},
                      'actions': [{'actor': 'child_front', 'do': 'walk', 'speed_mps': '$walk'},
                                  {'actor': 'child_rear', 'do': 'walk', 'speed_mps': '$walk'}]}],
        'pass': {'no_collision': False}},
    'svm_creep_past_kerb': {
        'id': 'svm_creep_past_kerb', 'name': 'Creeping past a child at the kerb',
        'about': 'The ego creeps along at walking pace past a child standing at the kerb and a car parked beyond it: '
                 'the child goes from the front cameras along the side to the rear.',
        'needs': [], 'duration_s': 10.0,
        'params': {'speed': {'dist': 'uniform', 'lo': 4.0, 'hi': 7.0, 'unit': 'km/h', 'about': 'the ego\'s speed'},
                   'child_at': {'dist': 'uniform', 'lo': 5.0, 'hi': 8.0, 'unit': 'm',
                                'about': 'where the child stands, ahead of the ego'},
                   'in_lane': {'dist': 'uniform', 'lo': 0.0, 'hi': 0.4, 'unit': 'm',
                               'about': 'how far the child stands into the lane'}},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'child', 'kind': 'child', 'place': {'lane': 'kerb', 'ahead_m': '$child_at',
                                                              'lateral_m': '$in_lane'},
                    'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'},
                   {'id': 'parked', 'kind': 'car', 'place': {'lane': 'kerb', 'ahead_m': '$child_at + 9'},
                    'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [],
        'pass': {'no_collision': True}},
}
DEFAULT_SCENARIOS = ('svm_children_round', 'svm_creep_past_kerb')


class CompareError(ValueError):
    pass


def scenario(key: str, project=None) -> dict:
    """A scenario by key: a project scenario, one of SVM_SCENARIOS, or a built-in (core.scenarios.LIBRARY)."""
    from . import scenarios as SC
    s = project.item('scenarios', key) if project is not None else None
    if s is None and key in SVM_SCENARIOS:
        s = copy.deepcopy(SVM_SCENARIOS[key])
    if s is None and key in SC.LIBRARY:
        s = SC.builtin(key)
    if s is None:
        raise CompareError(f'no scenario {key!r} (the surround-view ones: {", ".join(SVM_SCENARIOS)}; or a built-in or '
                           'project scenario id)')
    return s


# =================================================================================================== geometry
def _ego_frame(row):
    from ..sim.frames import carla_matrix
    return carla_matrix(row['pitch'], row['yaw'], row['roll']), np.array([row['x'], row['y'], row['z']], float)


def class_of_category(cat: str) -> str:
    for pre, cls in CATEGORY_CLASS:
        if str(cat or '').startswith(pre):
            return cls
    return 'other'


def rig_geometry(rig: dict) -> dict:
    """Where a rig's seams turn: {'centre_x' (Falcon frame), 'seams': [{'a','b','angle_deg','source'}] sorted
    counter-clockwise, 'cams' {tag: camera id}} (no seams for a rig not from Vantage)."""
    v = rig.get('vantage') or {}
    cams = {s['name']: s['id'] for s in rig.get('sensors', []) if s.get('type', '').startswith('camera.')}
    seams = sorted([{k: s.get(k) for k in ('a', 'b', 'angle_deg', 'source')} for s in v.get('seams') or []
                    if s.get('a') in cams and s.get('b') in cams], key=lambda s: s['angle_deg'])
    return {'centre_x': (v.get('centre') or {}).get('falcon_x'), 'seams': seams, 'cams': cams}


def owner(seams: list, az: float) -> str | None:
    """The camera tag the stitched view shows at azimuth az: the b of the last seam at or before it (counter-clockwise),
    as Vantage's sectors are assigned."""
    if not seams:
        return None
    angles = np.array([s['angle_deg'] for s in seams])
    i = (int(np.searchsorted(angles, az, side='right')) - 1) % len(seams)
    return seams[i]['b']


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def present_objects(run_dir: str, radius: float = RADIUS_M, centre_x: float | None = None) -> dict:
    """{step: [object]} the road users within radius of the CARLA car's box at each recorded step (actors.jsonl):
    {'id', 'type_id', 'class', 'key' (step, type and world position to 5 cm: the same in another run of the same
    scenario and seed), 'x', 'y' (vehicle frame, CARLA axes), 'az' (degrees about (centre_x, 0), counter-clockwise from
    ahead as Vantage's), 'clearance', 'sector'}."""
    from .vantagerig import sector_of
    path = os.path.join(run_dir, 'actors.jsonl')
    if not os.path.isfile(path):
        raise CompareError(f'{run_dir}: no actors.jsonl (record the run with a scenario)')
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            ego = next((a for a in d['actors'] if a.get('role') == 'hero'), None)
            if ego is None:
                continue
            Re, te = _ego_frame(ego)
            ec, ee = np.asarray(ego.get('bbox_center') or [0, 0, 0], float), np.asarray(ego['extent'], float)
            cx = float(ec[0]) if centre_x is None else float(centre_x)
            rows = []
            for a in d['actors']:
                if a is ego or not str(a.get('type_id', '')).startswith(('vehicle.', 'walker.')) and \
                        a.get('role') != 'falcon_prop':
                    continue
                Ra, ta = _ego_frame(a)
                pw = Ra @ np.asarray(a.get('bbox_center') or [0, 0, 0], float) + ta
                pv = Re.T @ (pw - te)
                dx = max(abs(pv[0] - ec[0]) - ee[0], 0.0)
                dy = max(abs(pv[1] - ec[1]) - ee[1], 0.0)
                clr = math.hypot(dx, dy)
                if clr > radius:
                    continue
                az = math.degrees(math.atan2(-pv[1], pv[0] - cx))
                rows.append({'id': a['id'], 'type_id': a['type_id'], 'class': class_of_category(a.get('category')),
                             'key': f'{d["step"]}:{a["type_id"]}:{round(pw[0] / 0.05)}:{round(pw[1] / 0.05)}',
                             'x': round(float(pv[0]), 3), 'y': round(float(pv[1]), 3), 'az': round(az, 2),
                             'clearance': round(clr, 3), 'sector': sector_of(az)})
            out[int(d['step'])] = rows
    return out


# =================================================================================================== matching
def match_frame(f: dict, thr: float = SCORE_THRESHOLD, iou: float = IOU) -> tuple[list, int, int]:
    """(hit per GT, true positives, false positives) of a frame record (core.harness.collect's): detections scoring
    at least thr, greedy per class in score order at IoU iou, non-ignored GT first; a detection on an ignored GT
    counts neither way."""
    gts, hit = f['gt'], [False] * len(f['gt'])
    tp = fp = 0
    for c in {d['class'] for d in f['det']}:
        gi = [j for j, g in enumerate(gts) if g['class'] == c]
        dets = sorted([d for d in f['det'] if d['class'] == c and d['score'] >= thr], key=lambda d: -d['score'])
        if not dets:
            continue
        ious = M.box_iou([d['bbox'] for d in dets], [gts[j]['bbox'] for j in gi]) if gi else None
        for k in range(len(dets)):
            best, bj = iou, -1
            for n, j in enumerate(gi):
                if not hit[j] and not gts[j]['ignore'] and ious[k, n] >= best:
                    best, bj = ious[k, n], j
            if bj >= 0:
                hit[bj] = True
                tp += 1
            elif not any(gts[j]['ignore'] and ious[k, n] >= iou for n, j in enumerate(gi)):
                fp += 1
    return hit, tp, fp


# =================================================================================================== collection
def collect_rig(rig: dict, runs: list, m: dict, workspace: str = '', every: int = 1, runner=None, progress=None,
                cancel=None) -> list:
    """Every camera's frames of the rig's runs through the model: core.harness frame records, each with 'camera' and
    'run' (the run's label: its folder name) and its GT's 'track_id' kept; image_id unique per camera."""
    from . import harness as H
    from .exporters import Run
    out = []
    cams = [s['id'] for s in rig['sensors'] if s.get('type', '').startswith('camera.')]
    jobs = [(r, c) for r in runs for c in cams]
    for k, (r, cid) in enumerate(jobs):
        if cancel and cancel():
            raise CompareError('cancelled')
        run = Run(r)
        if cid not in run.cams:
            raise CompareError(f'{r}: no camera {cid!r}')
        try:
            col = H.collect([r], m, workspace, cid, every, runner=runner, cancel=cancel)
        except H.HarnessError as e:
            raise CompareError(f'{os.path.basename(r)} / {cid}: {e}') from None
        for f in col['frames']:
            f['camera'], f['run'] = cid, os.path.basename(os.path.normpath(r))
            f['image_id'] = f'{f["drive"]}/{cid}/{f["step"]}'
            f.pop('image', None)
            out.append(f)
        if progress:
            progress((k + 1) / len(jobs), f'{os.path.basename(r)} / {cid}')
    return out


def units_of(labels: list, fps: float, steps: dict, block_s: float = BLOCK_S) -> tuple[str, dict]:
    """('run' or 'block', {(run label, step): unit id}): whole runs with MIN_UNITS of them or more, else blocks of
    block_s seconds."""
    by_run = len(labels) >= MIN_UNITS
    out = {}
    per = max(int(round(block_s * fps)), 1)
    for lab in labels:
        for st in steps.get(lab, ()):
            out[(lab, st)] = lab if by_run else f'{lab}#{st // per}'
    return ('run' if by_run else 'block'), out


def _bins():
    return [f'clearance[{a:g},{b:g})' for a, b in zip(CLEARANCE_BINS[:-1], CLEARANCE_BINS[1:])]


def _clr_bin(c):
    for a, b in zip(CLEARANCE_BINS[:-1], CLEARANCE_BINS[1:]):
        if a <= c < b:
            return f'clearance[{a:g},{b:g})'
    return None


def object_outcomes(frames: list, present: dict, geo: dict, thr: float = SCORE_THRESHOLD) -> tuple[dict, dict]:
    """Per (run label, step): ({object key: outcome}, frame counts {'tp', 'fp', 'gt', 'hit'}) for one rig. An outcome:
    {'obj' (present_objects' row), 'seen', 'det', 'owner', 'det_owner'}; frame counts over every camera."""
    by = {}
    tag_of = {v: k for k, v in geo['cams'].items()}
    for f in frames:
        hit, tp, fp = match_frame(f, thr)
        k = (f['run'], int(f['step']))
        e = by.setdefault(k, {'counts': {'tp': 0, 'fp': 0, 'gt': 0, 'hit': 0}, 'seen': {}, 'det': {}})
        c = e['counts']
        c['tp'] += tp
        c['fp'] += fp
        for g, h in zip(f['gt'], hit):
            if g['ignore']:
                continue
            c['gt'] += 1
            c['hit'] += int(h)
            tid = g.get('track_id')
            if isinstance(tid, (int, np.integer)):
                e['seen'].setdefault(int(tid), set()).add(f['camera'])
                if h:
                    e['det'].setdefault(int(tid), set()).add(f['camera'])
    objs, counts = {}, {}
    for (lab, step), e in by.items():
        counts[(lab, step)] = e['counts']
        rows = {}
        for o in (present.get(lab) or {}).get(step, []):
            seen, det = e['seen'].get(o['id'], set()), e['det'].get(o['id'], set())
            own_tag = owner(geo['seams'], o['az'])
            own = geo['cams'].get(own_tag) if own_tag else None
            rows[f'{lab}|{o["key"]}'] = {'obj': o, 'seen': bool(seen), 'det': bool(det), 'owner': tag_of.get(own),
                              'det_owner': (own in det) if own else None, 'cams_seen': sorted(seen),
                              'cams_det': sorted(det)}
        objs[(lab, step)] = rows
    return objs, counts


def _add(c, k, num, den=1):
    v = c.setdefault(k, [0, 0])
    v[0] += int(num)
    v[1] += int(den)


def unit_counts(objs: dict, counts: dict, units: dict, bands: list, keys=None, stitched: bool = True) -> dict:
    """{unit: {metric: [numerator, denominator]}} for one rig; keys (a set of object keys) restricts the object
    metrics to those objects (a paired comparison's shared objects)."""
    out = {}
    for k, cnt in counts.items():
        u = units.get(k)
        if u is None:
            continue
        c = out.setdefault(u, {})
        _add(c, 'precision', cnt['tp'], cnt['tp'] + cnt['fp'])
        _add(c, 'camera_recall', cnt['hit'], cnt['gt'])
        for key, o in (objs.get(k) or {}).items():
            if keys is not None and key not in keys:
                continue
            ob = o['obj']
            names = ['', f'.{ob["sector"]}']
            b = _clr_bin(ob['clearance'])
            if b:
                names.append(f'.{b}')
            names += [f'.seam[{bd["id"]}]' for bd in bands if abs(_wrap(ob['az'] - bd['angle_deg'])) <= bd['half']]
            for nm in names:
                _add(c, f'seen{nm}', o['seen'])
                _add(c, f'recall{nm}', o['det'])
                if stitched and o['det_owner'] is not None:
                    _add(c, f'stitched_recall{nm}', o['det_owner'])
            if o['seen']:
                _add(c, 'recall_seen', o['det'])
    return out


def ratios(items: list, which: str | None = None) -> dict:
    """The metrics of a list of unit-count dicts (or of their [which] part): sums of numerators / denominators."""
    tot = {}
    for it in items:
        c = it[which] if which else it['c']
        for k, (n, d) in c.items():
            t = tot.setdefault(k, [0, 0])
            t[0] += n
            t[1] += d
    return {k: (n / d if d else None) for k, (n, d) in tot.items()}


def _verdict(lo, hi) -> str:
    if hi is not None and hi < 0:
        return 'worse'
    if lo is not None and lo > 0:
        return 'better'
    return 'no detectable change'


def seam_bands(rigs: list, band_deg: float = BAND_DEG) -> list:
    """The bands about every rig's seams: [{'id', 'a', 'b', 'angle_deg', 'half', 'rigs', 'sources'}]; a seam two rigs
    share (the same cameras within 1°) is one band."""
    out = []
    for r in rigs:
        for s in rig_geometry(r)['seams']:
            same = next((b for b in out if b['a'] == s['a'] and b['b'] == s['b']
                         and abs(_wrap(b['angle_deg'] - s['angle_deg'])) < 1.0), None)
            if same:
                same['rigs'].append(r['id'])
                same['sources'].append(s['source'])
                continue
            out.append({'id': f'{s["a"]}|{s["b"]} {s["angle_deg"]:+.0f}°', 'a': s['a'], 'b': s['b'],
                        'angle_deg': float(s['angle_deg']), 'half': float(band_deg), 'rigs': [r['id']],
                        'sources': [s['source']]})
    return out


# =================================================================================================== the analysis
def analyse(rigs: list, frames: dict, runs: dict, fps: float = 20.0, radius: float = RADIUS_M,
            band_deg: float = BAND_DEG, thr: float = SCORE_THRESHOLD, n_boot: int = 1000, seed: int = 0,
            block_s: float = BLOCK_S, pairs: list | None = None, ap: bool = True) -> dict:
    """The comparison. rigs: the rig objects; frames: {rig id: collect_rig's records}; runs: {rig id: {run label: run
    folder}} (the same labels for every rig); pairs: [(rig a, rig b)] (default: every rig against the first)."""
    from . import harness as H
    ids = [r['id'] for r in rigs]
    labels = sorted(set.intersection(*[set(runs[i]) for i in ids]))
    if not labels:
        raise CompareError('the rigs share no scenario runs')
    bands = seam_bands(rigs, band_deg)
    per, notes = {}, []
    steps_all = {}
    for r in rigs:
        geo = rig_geometry(r)
        present = {lab: present_objects(runs[r['id']][lab], radius, geo['centre_x']) for lab in labels}
        fr = [f for f in frames[r['id']] if f['run'] in labels]
        objs, counts = object_outcomes(fr, present, geo, thr)
        per[r['id']] = {'geo': geo, 'objs': objs, 'counts': counts, 'frames': fr}
        for (lab, st) in counts:
            steps_all.setdefault(lab, set()).add(st)
    unit, units = units_of(labels, fps, {k: sorted(v) for k, v in steps_all.items()}, block_s)
    n_units = len(set(units.values()))
    if unit == 'block':
        notes.append(f'{len(labels)} scenario run(s): the interval is over {n_units} blocks of {block_s:g} s, which '
                     f'are correlated — it is too narrow; record {MIN_UNITS} or more runs (scenarios × seeds) for an '
                     'interval over whole runs')
    elif n_units < 20:
        notes.append(f'{n_units} runs: a bootstrap over fewer than about 20 units gives intervals that are somewhat '
                     'too narrow')
    variants = {}
    for r in rigs:
        p = per[r['id']]
        uc = unit_counts(p['objs'], p['counts'], units, bands, stitched=bool(p['geo']['seams']))
        items = [{'u': u, 'c': c} for u, c in sorted(uc.items())]
        bs = M.bootstrap(lambda its: ratios(its), items, n=n_boot, seed=seed, unit='image')
        tot = ratios(items)
        nobj = sum(len(v) for v in p['objs'].values())
        table = []
        for k, v in sorted(bs['metrics'].items(), key=lambda kv: _order(kv[0])):
            den = sum(it['c'].get(k, [0, 0])[1] for it in items)
            table.append({'metric': k, 'value': tot.get(k), 'ci_low': v['ci_low'], 'ci_high': v['ci_high'],
                          'n': den})
        doc = {'rig': r['id'], 'name': r['name'], 'n_frames': len(p['frames']), 'n_objects': nobj,
               'cameras': sorted({f['camera'] for f in p['frames']}), 'table': table,
               'seams': p['geo']['seams'], 'centre_x': p['geo']['centre_x']}
        if ap and p['frames']:
            try:
                sc = H.score(p['frames'], 'detector', n_boot=min(n_boot, 500), seed=seed)
                doc['ap'] = {row['metric']: row for row in sc['table'] if row['metric'] in H.HEADLINE}
                doc['ap_note'] = sc['note']
            except (H.HarnessError, ValueError) as e:
                doc['ap'] = {'error': str(e)}
        variants[r['id']] = doc
    out_pairs = []
    for a, b in (pairs or [(ids[0], x) for x in ids[1:]]):
        if a not in per or b not in per:
            raise CompareError(f'no rig {a if a not in per else b!r} in this comparison')
        pa, pb = per[a], per[b]
        ka = {k for v in pa['objs'].values() for k in v}
        kb = {k for v in pb['objs'].values() for k in v}
        shared = ka & kb
        ca = unit_counts(pa['objs'], pa['counts'], units, bands, shared, bool(pa['geo']['seams']))
        cb = unit_counts(pb['objs'], pb['counts'], units, bands, shared, bool(pb['geo']['seams']))
        items = [{'u': u, 'a': ca.get(u, {}), 'b': cb.get(u, {})} for u in sorted(set(ca) | set(cb))]
        rows = paired_rows(items, n_boot, seed)
        out_pairs.append({'a': a, 'b': b, 'n_objects': len(shared), 'only_a': len(ka - kb), 'only_b': len(kb - ka),
                          'rows': rows, 'worse': [r['metric'] for r in rows if r['verdict'] == 'worse'],
                          'better': [r['metric'] for r in rows if r['verdict'] == 'better']})
        if ka ^ kb:
            notes.append(f'{a} against {b}: {len(ka ^ kb)} object-steps were present in only one of the two runs (not '
                         'deterministic?) and are left out of the paired numbers')
    return {'schema': SCHEMA, 'labels': labels, 'unit': unit, 'n_units': n_units,
            'settings': {'radius_m': radius, 'band_deg': band_deg, 'score_threshold': thr, 'iou': IOU,
                         'n_boot': n_boot, 'seed': seed, 'block_s': block_s, 'fps': fps,
                         'clearance_bins_m': list(CLEARANCE_BINS)},
            'bands': bands, 'variants': variants, 'pairs': out_pairs, 'notes': notes}


def _betainc(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function I_x(a, b) (a continued fraction, Numerical Recipes §6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-12:
            break
    return math.exp(lb) * h / a


def t_two_sided(t: float, df: float) -> float:
    """P(|T| ≥ |t|) for Student's t with df degrees of freedom."""
    return _betainc(df / 2.0, 0.5, df / (df + t * t)) if math.isfinite(t) else 0.0


def cluster_test(items: list, k: str) -> dict:
    """Paired test of metric k (a ratio of sums) between 'a' and 'b' over units as clusters: the difference of the two
    ratios, its cluster-robust (CR1, delta-method) standard error over units, and Student's t with units − 1 degrees
    of freedom (a test that keeps its level with few units better than the percentile bootstrap). {'diff', 'se',
    't', 'df', 'p'}; p None with fewer than two units."""
    rows = [(it['a'].get(k, [0, 0]), it['b'].get(k, [0, 0])) for it in items]
    rows = [(a, b) for a, b in rows if a[1] or b[1]]
    U = len(rows)
    A = np.array([[a[0], a[1], b[0], b[1]] for a, b in rows], float).reshape(-1, 4)
    na, nb = A[:, 1].sum(), A[:, 3].sum()
    if U < 2 or not na or not nb:
        return {'diff': None, 'se': None, 't': None, 'df': max(U - 1, 0), 'p': None}
    ra, rb = A[:, 0].sum() / na, A[:, 2].sum() / nb
    e = (A[:, 2] - rb * A[:, 3]) / nb - (A[:, 0] - ra * A[:, 1]) / na
    se = math.sqrt(U / (U - 1) * float((e * e).sum()))
    diff = rb - ra
    if se < 1e-15:
        return {'diff': diff, 'se': 0.0, 't': None, 'df': U - 1, 'p': 1.0 if abs(diff) < 1e-15 else 0.0}
    t = diff / se
    return {'diff': diff, 'se': se, 't': t, 'df': U - 1, 'p': t_two_sided(t, U - 1)}


def holm(ps: list) -> list:
    """Holm's step-down adjustment of p-values (None stays None): the family's error rate kept at the level."""
    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    m, out, run = len(idx), [None] * len(ps), 0.0
    for j, i in enumerate(idx):
        run = max(run, min(1.0, (m - j) * ps[i]))
        out[i] = run
    return out


def paired_rows(items: list, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> list:
    """Every metric of a paired comparison: A, B, B − A with its paired bootstrap 95 % interval (core.metrics), the
    cluster-robust t-test's p and Holm's adjusted p over all the pair's metrics; the verdict is worse or better when
    the adjusted p is below alpha (direction by the sign), else 'no detectable change'."""
    res = M.paired_bootstrap(lambda its, w: ratios(its, w), items, 'a', 'b', n=n_boot, seed=seed, unit='image')
    keys = sorted(res['metrics'], key=_order)
    tests = [cluster_test(items, k) for k in keys]
    adj = holm([t['p'] for t in tests])
    rows = []
    for k, t, pa in zip(keys, tests, adj):
        v = res['metrics'][k]
        sig = pa is not None and pa < alpha and v['diff'] not in (None, 0.0)
        rows.append({'metric': k, 'a': v['a'], 'b': v['b'], 'diff': v['diff'], 'ci_low': v['ci_low'],
                     'ci_high': v['ci_high'], 'p_b_gt_a': v['p_b_gt_a'], 'se': t['se'], 'p': t['p'], 'p_holm': pa,
                     'n': sum(it['a'].get(k, [0, 0])[1] for it in items),
                     'verdict': ('worse' if v['diff'] < 0 else 'better') if sig else 'no detectable change'})
    return rows


def _order(k: str):
    base = k.split('.', 1)[0].split('[', 1)[0]
    rank = HEADLINE.index(base) if base in HEADLINE else len(HEADLINE)
    return (0 if '.' not in k else 1, rank, k)


# =================================================================================================== words
def _f(v, d=3):
    return '—' if v is None else f'{v:.{d}f}'


def _ci(r):
    return '' if r.get('ci_low') is None else f' [{_f(r["ci_low"])}, {_f(r["ci_high"])}]'


def markdown(rep: dict) -> str:
    """The report as Markdown."""
    L = [f'# Rig comparison: {rep.get("study", "")}', '']
    m = rep.get('model') or {}
    L.append(f'Model: {m.get("name") or m.get("id")}' + (' — the SYNTHETIC detector, not a model: its numbers test '
                                                          'the method, not a real detector' if
                                                          rep.get('synthetic_model') else ''))
    L.append(f'Runs: {len(rep["labels"])} ({", ".join(rep["labels"])}); unit: {rep["unit"]} '
             f'({rep["n_units"]}); objects within {rep["settings"]["radius_m"]:g} m of the car; seam bands '
             f'±{rep["settings"]["band_deg"]:g}°.')
    for n in rep.get('notes') or []:
        L.append(f'- {n}')
    L.append('')
    for v in rep['variants'].values():
        L += [f'## {v["name"]} ({v["rig"]})', '', f'{v["n_frames"]} camera frames, {v["n_objects"]} object-steps.', '',
              '| metric | value [95 % CI] | n |', '|---|---|---|']
        for r in v['table']:
            L.append(f'| {r["metric"]} | {_f(r["value"])}{_ci(r)} | {r["n"]} |')
        ap = (v.get('ap') or {}).get('AP')
        if isinstance(ap, dict):
            L.append(f'| AP (COCO, all cameras) | {_f(ap.get("value"))}{_ci(ap)} | {ap.get("n")} |')
        if v.get('seams'):
            L += ['', 'Seams: ' + ', '.join(f'{s["a"]}|{s["b"]} {s["angle_deg"]:+.1f}° ({s["source"]})'
                                           for s in v['seams'])]
        L.append('')
    for p in rep['pairs']:
        L += [f'## {p["b"]} against {p["a"]} (paired, {p["n_objects"]} shared object-steps)', '',
              '| metric | A | B | B − A [95 % CI] | p (Holm) | verdict |', '|---|---|---|---|---|---|']
        for r in p['rows']:
            L.append(f'| {r["metric"]} | {_f(r["a"])} | {_f(r["b"])} | {_f(r["diff"])}{_ci(r)} | '
                     f'{_f(r.get("p_holm"))} | {r["verdict"]} |')
        L.append('')
    for x in rep.get('limits') or []:
        L.append(f'- {x}')
    return '\n'.join(L) + '\n'


def _pair_row(p, metric):
    return next((r for r in p['rows'] if r['metric'] == metric), None)


def design_log(rep: dict, pair: tuple | None = None, date: str | None = None) -> dict:
    """A Vantage design-log export (schema vantage.design_log/1, the layout of Vantage's own export) with notes on the
    comparison — written to a file for the user to import or paste; Falcon never writes into a .vantage file."""
    pairs = rep['pairs']
    if pair:
        pairs = [p for p in pairs if (p['a'], p['b']) == tuple(pair)]
        if not pairs:
            raise CompareError(f'the comparison has no pair {pair[0]} / {pair[1]}')
    date = date or time.strftime('%Y-%m-%d')
    m = rep.get('model') or {}
    who = 'the synthetic detector (not a model)' if rep.get('synthetic_model') else (m.get('name') or m.get('id'))
    entries = []
    names = {k: v['name'] for k, v in rep['variants'].items()}
    for p in pairs:
        ra = _pair_row(p, 'recall')
        sa = _pair_row(p, 'stitched_recall')
        head = (f'Falcon perception check ({who}; {len(rep["labels"])} scenario runs, {p["n_objects"]} object-steps '
                f'within {rep["settings"]["radius_m"]:g} m): {names.get(p["b"], p["b"])} against '
                f'{names.get(p["a"], p["a"])}: ')
        parts = []
        for lab, r in (('recall', ra), ('stitched-view recall', sa)):
            if r and r['a'] is not None and r['b'] is not None:
                parts.append(f'{lab} {r["a"]:.2f} → {r["b"]:.2f} ({r["diff"]:+.2f}, 95 % CI {_f(r["ci_low"], 2)} to '
                             f'{_f(r["ci_high"], 2)}: {r["verdict"]})')
        text = head + '; '.join(parts or ['no objects scored'])
        about = set()
        seam_rows = [r for r in p['rows'] if r['metric'].startswith('recall.seam[') and r['verdict'] != 'no detectable '
                                                                                                         'change']
        for r in seam_rows[:3]:
            band = r['metric'][len('recall.seam['):-1]
            about.update(band.split(' ')[0].split('|'))
            text += f'; near seam {band}: {r["a"]:.2f} → {r["b"]:.2f} ({r["verdict"]})'
        if rep.get('unit') == 'block':
            text += '; interval over blocks of one run: too narrow'
        entries.append({'id': f'falcon-{p["a"]}-{p["b"]}'[:40], 'kind': 'note', 'text': text[:NOTE_TEXT_MAX],
                        'about': sorted(about), 'date': date, 'source': 'user'})
    return {'schema': 'vantage.design_log/1', 'entries': entries,
            'from': {'app': 'Falcon Autonomy', 'study': rep.get('study'), 'report': rep.get('path'),
                     'vantage_files': sorted({(v.get('vantage') or {}).get('file') for v in rep.get('rigs', [])
                                              if (v.get('vantage') or {}).get('file')})},
            'how': 'A Falcon note for Vantage\'s design log, in the layout of Vantage\'s own design-log export. Falcon '
                   'never writes into a .vantage file: add the entries in Vantage (Design log ▸ Add, pasting the text) '
                   'or import this file where your Vantage build reads one.'}
