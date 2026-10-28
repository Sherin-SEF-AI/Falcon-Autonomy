"""The model test harness: run a registered model (core.models) over recorded drives, score it against Falcon's ground
truth, and say where it fails (numpy at import; the model's runtime and OpenCV inside the functions).

What an evaluation does, for a detector:
1. collect(): every n-th frame of one camera of each drive goes through the model; per frame the predictions, the
   ground-truth objects (objects/<camera>/<n>.json) in the classes the model can name, the frame's conditions (the
   drive's weather, world and scenario parameters) and the model's latency are kept (predictions.jsonl), so a later
   comparison never has to run the model again.
2. score(): COCO AP/AR (core.metrics: pycocotools' numbers) with breakdowns by range, occlusion, truncation and pixel
   height, and a bootstrap CI of every number over **whole drives** (frames of one drive are not independent; with a
   single drive the CI is over frames and says so).
3. condition_grid(): AP over a grid of two conditions (e.g. fog × sun altitude), with each cell's frame count: the
   sweep heat map. Cells with too few objects are reported as such, never coloured.
4. gallery(): the frames with the most misses and confident false positives (IoU 0.5, per class), with their boxes.
5. compare(): two evaluations of the same frames, a paired bootstrap of each number's difference over drives; a
   regression is a difference whose whole CI is below zero.

GT objects that no detector could be asked to find are *ignored* (neither found nor missed, and a detection on one is
not a false positive): fewer than MIN_PIXELS visible pixels or more than MAX_OCCLUDED of the box hidden. The rule and
its numbers are in every evaluation's "method".

Segmenters are scored by their confusion matrix against the semantic labels (classes the model cannot name ignored):
per-class IoU and mIoU, with the same drive-level bootstrap.

A model's latency is measured here, on this machine, end to end per frame (pre-processing, inference, decoding), and
reported with the runtime's providers; it is not the latency on any other device.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import metrics as M

SCHEMA = 'falcon.evaluation/1'
MIN_PIXELS = 20
MAX_OCCLUDED = 0.9
HEADLINE = ('AP', 'AP50', 'AP75', 'AR100')
CONDITIONS = ('fog_density', 'precipitation', 'sun_altitude_angle', 'cloudiness', 'wetness')


class HarnessError(Exception):
    pass


def model_classes(m: dict) -> list:
    """The Falcon classes a model can name: its class_map's values, or for a plug-in its declared classes."""
    if m.get('class_map'):
        return sorted(set(m['class_map'].values()))
    return list(m.get('falcon_classes') or ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian', 'rider'))


def _xywh(b):
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def _conditions(run) -> dict:
    """A drive's conditions: its weather's parameters (as resolved for CARLA), weather preset, world and the scenario
    parameter values of a scenario run."""
    from .weather import resolve
    w = run.cfg.get('weather') or {}
    try:
        full = resolve(w)
    except (KeyError, ValueError, TypeError):
        full = dict(w)
    c = {k: full.get(k) for k in CONDITIONS if full.get(k) is not None}
    c['weather'] = (run.cfg.get('weather_source') or {}).get('preset') or w.get('preset') or 'custom'
    c['world'] = run.cfg.get('town') or ''
    for k, v in ((run.run.get('scenario') or {}).get('values') or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            c[f'scenario.{k}'] = v
    return c


def collect(runs: list, m: dict, workspace: str = '', camera: str | None = None, every: int = 1, progress=None,
            out_path: str | None = None, cancel=None) -> dict:
    """Run the model over the drives' frames: {'frames': [frame records], 'providers', 'latency_ms', 'classes'}.
    A frame record: {'image_id', 'drive', 'step', 'image', 'gt': [...], 'det': [...], 'conditions'} (detector) or
    {'image_id', 'drive', 'step', 'image', 'confusion': [[...]]} (segmenter). out_path: predictions.jsonl."""
    import cv2
    from .exporters import ExportError, Run, is_pinhole
    from .models import Runner
    from . import groundtruth as GT
    try:
        loaded = [Run(r) for r in runs]
    except ExportError as e:
        raise HarnessError(str(e)) from None
    if not loaded:
        raise HarnessError('no drives to evaluate on')
    runner = Runner(m, workspace)
    seg = m['kind'] == 'segmenter'
    classes = model_classes(m)
    cmap = m.get('class_map') or {}
    names = m.get('classes') or []
    jobs = []
    for run in loaded:
        cams = [c for c in run.cams if is_pinhole(run, c)] or list(run.cams)
        cid = camera or (cams[0] if cams else None)
        if cid not in run.cams:
            raise HarnessError(f'{run.dir}: no camera {cid!r}')
        need = 'semantic' if seg else 'objects'
        steps = [r for r in run.camera_steps(cid, every) if run.file(r, cid, need)]
        if not steps:
            raise HarnessError(f'{run.dir}: camera {cid} has no {need} labels to score against')
        jobs += [(run, cid, r) for r in steps]
    n_cls = len(GT.CLASSES)
    keep = np.zeros(256, bool)
    for c in classes:
        if c in GT.CLASS_ID:
            keep[GT.CLASS_ID[c]] = True
    seg_lut = None
    if seg:          # model class index -> Falcon class id (255: a class Falcon does not score)
        seg_lut = np.full(max(len(names), 1) + 1, 255, np.uint8)
        for i, nm in enumerate(names):
            fc = cmap.get(nm, nm)
            if fc in GT.CLASS_ID:
                seg_lut[i] = GT.CLASS_ID[fc]
    frames, lat, conds = [], [], {}
    fh = open(out_path, 'w', encoding='utf-8') if out_path else None
    try:
        for k, (run, cid, rec) in enumerate(jobs):
            if cancel and cancel():
                raise HarnessError('cancelled')
            path = run.p(run.file(rec, cid, 'rgb'))
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise HarnessError(f'cannot read {path}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            objs = run.objects(rec, cid) or []
            if run.key not in conds:
                conds[run.key] = _conditions(run)
            cond = conds[run.key]
            meta = {'objects': objs, 'weather': run.cfg.get('weather') or {}, 'run': run.name, 'step': rec['step']}
            t0 = time.perf_counter()
            out = runner.predict(rgb, meta)
            lat.append((time.perf_counter() - t0) * 1000.0)
            base = {'image_id': f'{run.key}/{rec["step"]}', 'drive': run.key, 'step': rec['step'], 'image': path,
                    'conditions': cond}
            if seg:
                pred = np.asarray(out)
                if pred.shape != rgb.shape[:2]:
                    raise HarnessError(f'the segmenter returned {pred.shape}, the image is {rgb.shape[:2]}')
                if m['runtime'] != 'python':       # model indices -> Falcon ids
                    pred = seg_lut[np.clip(pred, 0, len(seg_lut) - 1)]
                gt = cv2.imread(run.p(run.file(rec, cid, 'semantic')), cv2.IMREAD_UNCHANGED)
                gt = np.where(keep[gt], gt, 255).astype(np.uint8)
                conf = M.confusion_matrix(gt, pred.astype(np.int64), n_cls, ignore_index=255)
                fr = dict(base, confusion=conf.tolist())
            else:
                gts = []
                for o in objs:
                    if o.get('class') not in classes or not o.get('bbox'):
                        continue
                    ign = (o.get('pixels') or 0) < MIN_PIXELS or (o.get('occluded') or 0.0) > MAX_OCCLUDED
                    gts.append({'bbox': _xywh(o['bbox']), 'class': o['class'], 'ignore': int(ign),
                                'range_m': o.get('range_m'), 'occluded': o.get('occluded'),
                                'truncated': o.get('truncated'), 'track_id': o.get('track_id')})
                dets = []
                for d in out:
                    c = d['class'] if m['runtime'] == 'python' else cmap.get(d['class'], d['class'])
                    if c in classes:
                        dets.append({'bbox': _xywh(d['bbox']), 'class': c, 'score': float(d['score'])})
                fr = dict(base, gt=gts, det=dets)
            frames.append(fr)
            if fh:
                fh.write(json.dumps(fr, separators=(',', ':')) + '\n')
            if progress:
                progress((k + 1) / len(jobs), f'frame {k + 1} of {len(jobs)}')
    finally:
        if fh:
            fh.close()
    l = np.asarray(lat)
    return {'frames': frames, 'providers': list(runner.providers), 'classes': classes,
            'synthetic': bool(getattr(getattr(runner, 'plugin', None), 'synthetic', False)),
            'latency_ms': {'n': len(l), 'median': float(np.median(l)), 'p90': float(np.percentile(l, 90)),
                           'p99': float(np.percentile(l, 99)), 'mean': float(l.mean())},
            'cameras': sorted({c for _, c, _ in jobs})}


def read_predictions(path: str) -> list:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


# ================================================================================================ scoring
def _det_numbers(records) -> dict:
    s = M.coco_summarise(records)
    out = {k: s.get(k) for k in HEADLINE}
    for c, v in (s.get('per_class') or {}).items():
        out[f'AP.{c}'] = v.get('AP')
    for attr, bins in (s.get('breakdowns') or {}).items():
        for b in bins:
            out[f'AP.{attr}[{b["lo"]:g},{b["hi"]:g})'] = b.get('AP')
    return out


def _seg_numbers(frames) -> dict:
    conf = np.sum([np.asarray(f['confusion']) for f in frames], axis=0)
    s = M.segmentation_metrics(conf)
    from . import groundtruth as GT
    out = {'mIoU': s['mIoU'], 'pixel_accuracy': s['pixel_accuracy']}
    for i, v in enumerate(s['IoU']):
        if v is not None:
            out[f'IoU.{GT.CLASS_NAMES.get(i, i)}'] = v
    return out


def score(frames: list, kind: str = 'detector', n_boot: int = 1000, seed: int = 0, breakdowns=None) -> dict:
    """The numbers and their bootstrap CIs: {'unit', 'n_units', 'table': [{metric, value, ci_low, ci_high, n}],
    'full': the point evaluation}."""
    if not frames:
        raise HarnessError('no frames to score')
    unit = 'drive' if len({f['drive'] for f in frames}) > 1 else 'image'
    if kind == 'segmenter':
        boot = M.bootstrap(_seg_numbers, frames, n=n_boot, seed=seed, unit=unit)
        full = _seg_numbers(frames)
    else:
        bd = M.DEFAULT_BREAKDOWNS if breakdowns is None else breakdowns
        recs = M.coco_match(frames, breakdowns=bd)
        boot = M.bootstrap(_det_numbers, recs, n=n_boot, seed=seed, unit=unit)
        full = M.coco_summarise(recs)
    return {'unit': unit, 'n_units': boot.get('n_units'), 'n_boot': n_boot, 'seed': seed,
            'table': M.summary_table(boot), 'full': _clean(full)}


def _clean(o):
    """JSON-safe: numpy arrays and numbers to lists and floats, non-finite to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if k != 'meta'}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _edges(vals, n):
    v = np.asarray(sorted(set(vals)), float)
    if len(v) <= n:            # few distinct values (a sweep's grid): one cell per value
        mids = (v[:-1] + v[1:]) / 2 if len(v) > 1 else np.array([])
        return np.concatenate([[v[0] - 0.5], mids, [v[-1] + 0.5]]) if len(v) else np.array([0.0, 1.0])
    return np.quantile(v, np.linspace(0, 1, n + 1))


def condition_grid(frames: list, x: str, y: str, bins: int = 4, min_gt: int = 10) -> dict:
    """AP (COCO, all classes) per cell of two conditions; a cell with fewer than min_gt scored objects has AP None
    and says why. Bins: one per distinct value when there are few (a sweep), else quantiles."""
    xs = [f['conditions'].get(x) for f in frames]
    ys = [f['conditions'].get(y) for f in frames]
    ok = [i for i, (a, b) in enumerate(zip(xs, ys)) if isinstance(a, (int, float)) and isinstance(b, (int, float))]
    if not ok:
        raise HarnessError(f'no frame has both {x} and {y}')
    ex, ey = _edges([xs[i] for i in ok], bins), _edges([ys[i] for i in ok], bins)
    vx, vy = (sorted({v[i] for i in ok}) for v in (xs, ys))
    cells = []
    for j in range(len(ey) - 1):
        row = []
        for i in range(len(ex) - 1):
            sel = [frames[k] for k in ok if ex[i] <= xs[k] <= ex[i + 1] and ey[j] <= ys[k] <= ey[j + 1]
                   and (xs[k] < ex[i + 1] or i == len(ex) - 2) and (ys[k] < ey[j + 1] or j == len(ey) - 2)]
            n_gt = sum(1 for f in sel for g in f['gt'] if not g['ignore'])
            ap = M.coco_evaluate(sel)['AP'] if sel and n_gt >= min_gt else None
            row.append({'x': [float(ex[i]), float(ex[i + 1])], 'y': [float(ey[j]), float(ey[j + 1])],
                        'x_value': float(vx[i]) if len(vx) <= bins else None,       # a sweep's own value
                        'y_value': float(vy[j]) if len(vy) <= bins else None,
                        'n_frames': len(sel), 'n_gt': n_gt, 'AP': ap,
                        'note': None if ap is not None else f'fewer than {min_gt} objects'})
        cells.append(row)
    return {'x': x, 'y': y, 'x_edges': ex.tolist(), 'y_edges': ey.tolist(), 'cells': cells, 'min_gt': min_gt}


def _frame_errors(f: dict, iou: float = 0.5, min_score: float = 0.5) -> dict:
    """Greedy per-class matching at IoU 0.5 (score order): missed GTs and false positives of at least min_score."""
    missed, fps = [], []
    for c in {g['class'] for g in f['gt']} | {d['class'] for d in f['det']}:
        gi = [g for g in f['gt'] if g['class'] == c]
        di = sorted([d for d in f['det'] if d['class'] == c], key=lambda d: -d['score'])
        taken = [False] * len(gi)
        ious = M.box_iou([d['bbox'] for d in di], [g['bbox'] for g in gi]) if gi and di else None
        for k, d in enumerate(di):
            best, bj = iou, -1
            for j in range(len(gi)):
                if not taken[j] and ious[k, j] >= best:
                    best, bj = ious[k, j], j
            if bj >= 0:
                taken[bj] = True
            elif d['score'] >= min_score and not any(ious is not None and ious[k, j] >= iou and gi[j]['ignore']
                                                     for j in range(len(gi))):
                fps.append(d)
        missed += [g for g, t in zip(gi, taken) if not t and not g['ignore']]
    return {'missed': missed, 'false_positives': fps}


def gallery(frames: list, k: int = 24, min_score: float = 0.5) -> list:
    """The k frames with the most errors (missed objects + false positives scoring ≥ min_score), worst first, each
    with its boxes and conditions."""
    rows = []
    for f in frames:
        e = _frame_errors(f, min_score=min_score)
        n = len(e['missed']) + len(e['false_positives'])
        if n:
            rows.append({'image_id': f['image_id'], 'drive': f['drive'], 'step': f['step'], 'image': f['image'],
                         'conditions': f['conditions'], 'n_errors': n, **e})
    rows.sort(key=lambda r: (-r['n_errors'], r['image_id']))
    return rows[:k]


def compare(frames_a: list, frames_b: list, kind: str = 'detector', n_boot: int = 1000, seed: int = 0) -> dict:
    """B against A on the frames both scored (matched by image_id): a paired bootstrap over drives of every number's
    difference B − A. 'regressions' lists the numbers whose CI of the difference lies wholly below zero,
    'improvements' wholly above."""
    by_a = {f['image_id']: f for f in frames_a}
    items = [{'image_id': f['image_id'], 'drive': f['drive'], 'a': by_a[f['image_id']], 'b': f}
             for f in frames_b if f['image_id'] in by_a]
    if not items:
        raise HarnessError('the two evaluations share no frames')
    unit = 'drive' if len({it['drive'] for it in items}) > 1 else 'image'
    if kind == 'segmenter':
        def fn(its, which):
            return _seg_numbers([it[which] for it in its])
    else:
        recs_a = {r['image_id']: r for r in M.coco_match([it['a'] for it in items], breakdowns=M.DEFAULT_BREAKDOWNS)}
        recs_b = {r['image_id']: r for r in M.coco_match([it['b'] for it in items], breakdowns=M.DEFAULT_BREAKDOWNS)}
        for it in items:
            it['ra'], it['rb'] = recs_a[it['image_id']], recs_b[it['image_id']]

        def fn(its, which):
            return _det_numbers([it['r' + which] for it in its])
    res = M.paired_bootstrap(fn, items, 'a', 'b', n=n_boot, seed=seed, unit=unit)
    rows = []
    for name, r in (res.get('metrics') or {}).items():
        rows.append({'metric': name, 'a': r.get('a'), 'b': r.get('b'), 'diff': r.get('diff'),
                     'ci_low': r.get('ci_low'), 'ci_high': r.get('ci_high'), 'p_b_gt_a': r.get('p_b_gt_a')})
    reg = [r['metric'] for r in rows if r['ci_high'] is not None and r['ci_high'] < 0]
    imp = [r['metric'] for r in rows if r['ci_low'] is not None and r['ci_low'] > 0]
    return _clean({'unit': unit, 'n_units': res.get('n_units'), 'n_frames': len(items), 'n_boot': n_boot,
                   'seed': seed, 'rows': rows, 'regressions': reg, 'improvements': imp})


# ================================================================================================ the command
def evaluate(runs: list, m: dict, out_dir: str, workspace: str = '', camera: str | None = None, every: int = 1,
             n_boot: int = 1000, seed: int = 0, grid: tuple | None = None, progress=None, cancel=None) -> dict:
    """The whole evaluation into out_dir: predictions.jsonl and evaluation.json."""
    from . import provenance as PV
    os.makedirs(out_dir, exist_ok=True)

    def sub(lo, hi):
        return (lambda f, t: progress(lo + (hi - lo) * f, t)) if progress else None
    col = collect(runs, m, workspace, camera, every, sub(0.0, 0.8), os.path.join(out_dir, 'predictions.jsonl'),
                  cancel)
    frames = col['frames']
    if progress:
        progress(0.8, 'scoring')
    sc = score(frames, m['kind'], n_boot, seed)
    doc = {'schema': SCHEMA, 'model': {k: m.get(k) for k in ('id', 'name', 'kind', 'runtime', 'path', 'entry',
                                                                'licence', 'source', 'classes', 'class_map')},
           'synthetic_model': col['synthetic'], 'runs': [os.path.abspath(r) for r in runs],
           'cameras': col['cameras'], 'every': every, 'n_frames': len(frames),
           'providers': col['providers'], 'latency_ms': col['latency_ms'], 'classes': col['classes'],
           'method': {'min_pixels': MIN_PIXELS, 'max_occluded': MAX_OCCLUDED, 'bootstrap_unit': sc['unit'],
                      'n_boot': n_boot, 'seed': seed,
                      'latency': 'measured on this machine per frame: pre-processing, inference and decoding'},
           'scores': sc, 'grid': None, 'gallery': None}
    if m['kind'] == 'detector':
        doc['gallery'] = gallery(frames)
        if grid:
            try:
                doc['grid'] = condition_grid(frames, grid[0], grid[1])
            except HarnessError as e:
                doc['grid'] = {'error': str(e)}
    doc['provenance'] = PV.block({f'run{i}/run.json': os.path.join(r, 'run.json') for i, r in enumerate(runs)}
                                 | ({'model': os.path.join(workspace, 'models', m['path'])}
                                    if m.get('path') and not os.path.isabs(m['path']) else
                                    ({'model': m['path']} if m.get('path') else {})), seed,
                                 {'every': every, 'camera': camera, 'n_boot': n_boot, 'grid': list(grid or ())})
    PV.write_json(os.path.join(out_dir, 'evaluation.json'), _clean(doc))
    if progress:
        progress(1.0, 'done')
    return doc


def main_evaluate(argv) -> int:
    """`falcon evaluate PROJECT --model ID --run RUN... --out DIR [--camera C] [--every N] [--boot N] [--seed S]
    [--grid X,Y]` or `falcon evaluate --compare EVAL_A EVAL_B --out DIR`: protocol lines on stdout; 0 ok, 2 failed."""
    import argparse
    from . import protocol as P
    from . import provenance as PV
    ap = argparse.ArgumentParser(prog='falcon evaluate', description='score a registered model on recorded drives')
    ap.add_argument('project', nargs='?')
    ap.add_argument('--model')
    ap.add_argument('--run', action='append', default=[])
    ap.add_argument('--camera')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grid', help='two conditions for the heat map, e.g. fog_density,sun_altitude_angle')
    ap.add_argument('--compare', nargs=2, metavar=('EVAL_A', 'EVAL_B'))
    ap.add_argument('--out', '-o', required=True)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('evaluate')
    try:
        if a.compare:
            ea, eb = (PV.read_json(os.path.join(d, 'evaluation.json')) for d in a.compare)
            if ea['model']['kind'] != eb['model']['kind']:
                raise HarnessError('the two evaluations are of different kinds of model')
            fa, fb = (read_predictions(os.path.join(d, 'predictions.jsonl')) for d in a.compare)
            res = compare(fa, fb, ea['model']['kind'], a.boot, a.seed)
            res.update(schema='falcon.comparison/1', a={'dir': os.path.abspath(a.compare[0]), 'model': ea['model']},
                       b={'dir': os.path.abspath(a.compare[1]), 'model': eb['model']},
                       provenance=PV.block({'a': os.path.join(a.compare[0], 'evaluation.json'),
                                            'b': os.path.join(a.compare[1], 'evaluation.json')}, a.seed))
            os.makedirs(a.out, exist_ok=True)
            PV.write_json(os.path.join(a.out, 'comparison.json'), res)
            em.result('ok', summary={'regressions': res['regressions'], 'improvements': res['improvements']})
            return P.OK
        from .project import Project, ProjectError
        from .paths import workspace_for
        if not a.project or not a.model or not a.run:
            raise HarnessError('PROJECT, --model and at least one --run are needed')
        try:
            proj = Project.load(a.project)
        except (OSError, ProjectError) as e:
            raise HarnessError(str(e)) from None
        m = proj.item('models', a.model)
        if m is None:
            raise HarnessError(f'no model {a.model!r} in the project')
        grid = tuple(a.grid.split(',')) if a.grid else None
        if grid and len(grid) != 2:
            raise HarnessError('--grid is two condition names, X,Y')
        doc = evaluate(a.run, m, a.out, workspace_for(a.project), a.camera, a.every, a.boot, a.seed, grid,
                       lambda f, t: em.progress(f, t))
        head = {r['metric']: r['value'] for r in doc['scores']['table'] if r['metric'] in HEADLINE + ('mIoU',)}
        em.result('ok', summary={'frames': doc['n_frames'], **head})
        return P.OK
    except (HarnessError, ValueError, OSError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
