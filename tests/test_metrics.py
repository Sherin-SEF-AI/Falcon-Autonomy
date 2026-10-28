"""Evaluation metrics: COCO and nuScenes detection against their reference implementations (pycocotools and the
nuscenes-devkit, run in .venv-devkits), tracking (CLEAR MOT, HOTA) on hand-derived sequences, segmentation, lanes,
calibration, the bootstrap's coverage and the paired comparison."""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from falcon_autonomy.core import metrics as M
from falcon_autonomy.core.metrics import coco as C
from falcon_autonomy.core.metrics import nuscenes as N

VENV_PY = Path(__file__).resolve().parents[1] / '.venv-devkits' / 'bin' / 'python'


def _devkit_python(*modules):
    if not VENV_PY.exists():
        pytest.skip(f'no devkit venv at {VENV_PY}')
    r = subprocess.run([str(VENV_PY), '-c', 'import ' + ', '.join(modules)], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f'devkit venv lacks {modules}: {r.stderr.strip().splitlines()[-1]}')
    return str(VENV_PY)


def _run_reference(tmp_path, script: str, spec: dict, *modules):
    py = _devkit_python(*modules)
    (tmp_path / 'ref.py').write_text(script)
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    r = subprocess.run([py, str(tmp_path / 'ref.py'), str(tmp_path / 'spec.json')], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


# --- COCO --------------------------------------------------------------------------------------------------------

def synth_coco(n_images=250, seed=0, classes=('car', 'pedestrian', 'cyclist', 'truck'), n_gt=(0, 15),
               noise=(0.02, 0.12), n_fp=(0, 6), attrs=True):
    """Random boxes over all COCO size ranges, noisy detections (some of the wrong class, some duplicated), false
    positives, crowd and ignored GT, a detection class absent from the GT; Falcon's object attributes on both."""
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n_images):
        gts, dets = [], []
        for _ in range(rng.integers(n_gt[0], n_gt[1] + 1)):
            w, h = np.exp(rng.uniform(np.log(4), np.log(300), 2))
            x, y = rng.uniform(0, 1600 - w), rng.uniform(0, 900 - h)
            c = classes[rng.integers(len(classes))]
            g = {'bbox': [float(x), float(y), float(w), float(h)], 'class': c}
            if attrs:
                g.update(range_m=float(rng.uniform(1, 100)), occluded=float(rng.uniform(0, 1)),
                         truncated=float(rng.uniform(0, 0.9)))
            u = rng.random()
            if u < 0.04:
                g['iscrowd'] = 1
            elif u < 0.09:
                g['ignore'] = 1
            gts.append(g)
            for _ in range(rng.choice([0, 1, 1, 1, 2])):
                s = rng.uniform(*noise)
                bb = [x + rng.normal(0, s * w), y + rng.normal(0, s * h), w * math.exp(rng.normal(0, s)),
                      h * math.exp(rng.normal(0, s))]
                dc = c if rng.random() > 0.1 else classes[rng.integers(len(classes))]
                dets.append({'bbox': [float(v) for v in bb], 'class': dc, 'score': float(rng.uniform(0.3, 1.0) ** 0.5)})
        for _ in range(rng.integers(n_fp[0], n_fp[1] + 1)):
            w, h = np.exp(rng.uniform(np.log(4), np.log(300), 2))
            dets.append({'bbox': [float(rng.uniform(0, 1500)), float(rng.uniform(0, 800)), float(w), float(h)],
                         'class': classes[rng.integers(len(classes))], 'score': float(rng.uniform(0, 0.9))})
        if rng.random() < 0.05:
            dets.append({'bbox': [10.0, 10.0, 50.0, 50.0], 'class': 'bus', 'score': 0.5})
        if attrs:
            for d in dets:
                d.update(range_m=float(rng.uniform(1, 100)), occluded=float(rng.uniform(0, 1)),
                         truncated=float(rng.uniform(0, 0.9)))
        images.append({'image_id': i + 1, 'gt': gts, 'det': dets, 'drive': i // 25})
    return images


def to_coco_json(images, bins):
    cats = sorted({g['class'] for im in images for g in im['gt']})
    cid = {c: j + 1 for j, c in enumerate(cats)}
    anns, dts = [], []
    for im in images:
        for g in im['gt']:
            a = {'id': len(anns) + 1, 'image_id': im['image_id'], 'category_id': cid[g['class']], 'bbox': g['bbox'],
                 'area': g['bbox'][2] * g['bbox'][3], 'iscrowd': int(g.get('iscrowd', 0)),
                 'falcon_ignore': int(g.get('ignore', 0))}
            a.update({'a_' + k: g['bbox'][3] if k == 'height' else g[k] for k in bins})
            anns.append(a)
        for d in im['det']:
            if d['class'] in cid:
                a = {'image_id': im['image_id'], 'category_id': cid[d['class']], 'bbox': d['bbox'], 'score': d['score']}
                a.update({'a_' + k: d['bbox'][3] if k == 'height' else d[k] for k in bins})
                dts.append(a)
    return {'gt': {'images': [{'id': im['image_id']} for im in images], 'annotations': anns,
                   'categories': [{'id': cid[c], 'name': c} for c in cats]}, 'det': dts, 'cats': cats}


# pycocotools with Falcon's 'ignore' flag honoured (its _prepare overwrites 'ignore' with 'iscrowd'), and each bin of a
# breakdown evaluated as an area range on the attribute (GT 'area' := attribute, detection 'area' := its value).
COCO_REF = r'''
import contextlib, io, json, sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Eval(COCOeval):
    def _prepare(self):
        super()._prepare()
        for gts in self._gts.values():
            for g in gts:
                g['ignore'] = bool(g['ignore']) or bool(g['falcon_ignore'])

spec = json.load(open(sys.argv[1]))

def run(area_ranges, key=None):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = json.loads(json.dumps(spec['gt']))
        if key:
            for a in gt.dataset['annotations']:
                a['area'] = a[key]
        gt.createIndex()
        dt = gt.loadRes(json.loads(json.dumps(spec['det'])))
        if key:
            for a in dt.dataset['annotations']:
                a['area'] = a[key]
        E = Eval(gt, dt, 'bbox')
        if area_ranges is not None:
            E.params.areaRng = [list(r) for r in area_ranges]
            E.params.areaRngLbl = [str(i) for i in range(len(area_ranges))]
        E.evaluate(); E.accumulate()
        if area_ranges is None:
            E.summarize()
    P, R = E.eval['precision'], E.eval['recall']
    mv = lambda x: float(x[x > -1].mean()) if (x > -1).any() else -1.0
    out = {'ap': [mv(P[:, :, :, a, -1]) for a in range(P.shape[3])],
           'ar': [mv(R[:, :, a, -1]) for a in range(P.shape[3])],
           'per_class': [mv(P[:, :, k, 0, -1]) for k in range(P.shape[2])],
           'per_class_ap50': [mv(P[0, :, k, 0, -1]) for k in range(P.shape[2])]}
    if area_ranges is None:
        out['stats'] = [float(s) for s in E.stats]
    return out

out = {'standard': run(None)}
for attr, edges in spec['bins'].items():
    out[attr] = run([[edges[i], edges[i + 1]] for i in range(len(edges) - 1)], 'a_' + attr)
json.dump(out, sys.stdout)
'''

COCO_STAT_KEYS = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR1', 'AR10', 'AR100', 'AR_small',
                  'AR_medium', 'AR_large')


def _v(x):
    return -1.0 if x is None else x


def test_coco_matches_pycocotools_including_breakdowns(tmp_path):
    images = synth_coco(250, seed=0)
    bins = {k: list(v) for k, v in C.DEFAULT_BREAKDOWNS.items()}
    spec = to_coco_json(images, bins)
    spec['bins'] = bins
    ref = _run_reference(tmp_path, COCO_REF, spec, 'pycocotools')
    res = M.coco_evaluate(images, breakdowns=C.DEFAULT_BREAKDOWNS)
    diffs = [abs(_v(res[k]) - r) for k, r in zip(COCO_STAT_KEYS, ref['standard']['stats'])]
    diffs += [abs(_v(res['per_class'][c]['AP']) - r) for c, r in zip(spec['cats'], ref['standard']['per_class'])]
    diffs += [abs(_v(res['per_class'][c]['AP50']) - r) for c, r in zip(spec['cats'], ref['standard']['per_class_ap50'])]
    for attr in bins:
        rows = res['breakdowns'][attr]
        assert len(rows) == len(bins[attr]) - 1
        diffs += [abs(_v(b['AP']) - r) for b, r in zip(rows, ref[attr]['ap'])]
        diffs += [abs(_v(b['AR100']) - r) for b, r in zip(rows, ref[attr]['ar'])]
    assert max(diffs) < 1e-6, max(diffs)
    assert 0.2 < res['AP'] < 0.9                                    # a meaningful operating point, not all 0 or 1
    assert res['n_images'] == 250 and res['n_classes'] == 4 and 'bus' not in res['per_class']
    assert res['n_gt'] + res['n_gt_ignored'] == sum(len(im['gt']) for im in images)
    assert sum(b['n_gt'] for b in res['breakdowns']['range_m']) == res['n_gt']   # the bins partition the GT


def test_coco_crowd_ignore_and_maxdets_by_hand():
    gt = [{'bbox': [0, 0, 10, 10], 'class': 'car'}, {'bbox': [100, 0, 50, 50], 'class': 'car', 'iscrowd': 1},
          {'bbox': [300, 0, 10, 10], 'class': 'car', 'ignore': 1}]
    det = [{'bbox': [0, 0, 10, 10], 'class': 'car', 'score': 0.9},
           {'bbox': [100, 0, 10, 10], 'class': 'car', 'score': 0.8},     # inside the crowd: ignored, not an FP
           {'bbox': [105, 5, 10, 10], 'class': 'car', 'score': 0.7},     # a crowd can absorb many
           {'bbox': [300, 0, 10, 10], 'class': 'car', 'score': 0.95}]    # on an ignored GT: ignored
    r = M.coco_evaluate([{'image_id': 1, 'gt': gt, 'det': det}])
    assert r['AP'] == pytest.approx(1.0) and r['AR100'] == pytest.approx(1.0)
    assert r['n_gt'] == 1 and r['n_gt_ignored'] == 2
    # an FP ranked above the TP: precision at every recall point is 1/2
    det2 = det[:1] + [{'bbox': [500, 500, 10, 10], 'class': 'car', 'score': 0.99}]
    r2 = M.coco_evaluate([{'image_id': 1, 'gt': gt, 'det': det2}])
    assert r2['AP'] == pytest.approx(0.5, abs=1e-9)
    assert r2['AR1'] == 0.0 and r2['AR10'] == pytest.approx(1.0)      # maxDets 1 keeps only the FP
    # no detections: AP 0; a class with only ignored GT: no AP (None, the reference's -1)
    r3 = M.coco_evaluate([{'image_id': 1, 'gt': gt + [{'bbox': [0, 0, 5, 5], 'class': 'ped', 'ignore': 1}], 'det': []}])
    assert r3['AP'] == 0.0 and r3['per_class']['ped']['AP'] is None


def test_coco_breakdown_attributes_unmatched_detections_to_their_nearest_gt():
    gt = [{'bbox': [0, 0, 20, 20], 'class': 'car', 'range_m': 10.0},
          {'bbox': [200, 0, 20, 20], 'class': 'car', 'range_m': 50.0}]
    det = [{'bbox': [0, 0, 20, 20], 'class': 'car', 'score': 0.9},       # TP of the near GT
           {'bbox': [212, 0, 20, 20], 'class': 'car', 'score': 0.95},    # IoU 0.25 with the far GT: an FP there
           {'bbox': [900, 900, 20, 20], 'class': 'car', 'score': 0.99}]  # overlaps nothing: unassigned
    img = [{'image_id': 1, 'gt': gt, 'det': det}]
    near, far = M.coco_evaluate(img, breakdowns={'range_m': [0, 30, 100]})['breakdowns']['range_m']
    assert near['n_gt'] == 1 and near['n_det'] == 2                     # its TP + the unassigned FP
    assert far['n_gt'] == 1 and far['n_det'] == 2                       # its FP + the unassigned FP
    assert near['AP50'] == pytest.approx(0.5, abs=1e-9) and far['AP50'] == 0.0
    near_i, _ = M.coco_evaluate(img, breakdowns={'range_m': [0, 30, 100]}, unassigned='ignore')['breakdowns']['range_m']
    assert near_i['n_det'] == 1 and near_i['AP50'] == pytest.approx(1.0)
    # a detection's own value wins over its nearest GT's
    det[1]['range_m'] = 12.0
    near2, far2 = M.coco_evaluate(img, breakdowns={'range_m': [0, 30, 100]})['breakdowns']['range_m']
    assert near2['n_det'] == 3 and far2['n_det'] == 1


def test_coco_speed_500_images_20_boxes():
    images = synth_coco(500, seed=3, n_gt=(20, 20), noise=(0.02, 0.1), n_fp=(2, 4), attrs=False)
    assert 18 <= np.mean([len(im['det']) for im in images]) <= 30
    t = time.perf_counter()
    r = M.coco_evaluate(images)
    dt = time.perf_counter() - t
    assert r['n_gt'] + r['n_gt_ignored'] == 10000
    assert dt < 3.0, f'{dt:.2f} s'


# --- nuScenes ----------------------------------------------------------------------------------------------------

NUSC_SIZE = {'car': (1.9, 4.5, 1.6), 'pedestrian': (0.7, 0.7, 1.8), 'bicycle': (0.6, 1.8, 1.4),
             'barrier': (2.5, 0.5, 1.0), 'traffic_cone': (0.4, 0.4, 0.8)}
NUSC_ATTR = {'car': ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped'],
             'pedestrian': ['pedestrian.moving', 'pedestrian.standing'],
             'bicycle': ['cycle.with_rider', 'cycle.without_rider'], 'barrier': [''], 'traffic_cone': ['']}


def synth_nuscenes(n=150, seed=1, velocity=True):
    rng = np.random.default_rng(seed)
    cls = list(NUSC_SIZE)
    out = []
    for i in range(n):
        gts, dets = [], []
        for _ in range(rng.integers(0, 12)):
            c = cls[rng.integers(len(cls))]
            t = rng.uniform(-50, 50, 2)
            sz = np.array(NUSC_SIZE[c]) * np.exp(rng.normal(0, 0.1, 3))
            yaw = rng.uniform(-math.pi, math.pi)
            v = rng.normal(0, 3, 2) if (velocity and rng.random() > 0.1) else None
            a = NUSC_ATTR[c][rng.integers(len(NUSC_ATTR[c]))] if rng.random() > 0.1 else ''
            gts.append({'translation': [float(t[0]), float(t[1]), 0.5], 'size': [float(x) for x in sz],
                        'yaw': float(yaw), 'class': c, 'velocity': None if v is None else [float(x) for x in v],
                        'attribute': a})
            if rng.random() < 0.8:
                tt = t + rng.normal(0, rng.choice([0.2, 0.7, 1.5, 3.0]), 2)
                d = {'translation': [float(tt[0]), float(tt[1]), 0.4],
                     'size': [float(x) for x in sz * np.exp(rng.normal(0, 0.15, 3))],
                     'yaw': float(yaw + rng.normal(0, 0.4) + (math.pi if rng.random() < 0.1 else 0)), 'class': c,
                     'attribute': NUSC_ATTR[c][rng.integers(len(NUSC_ATTR[c]))],
                     'score': float(np.round(rng.uniform(0.2, 1), 2))}   # rounded: equal scores occur
                if velocity:
                    d['velocity'] = [float(x) for x in (np.zeros(2) if v is None else v) + rng.normal(0, 1, 2)]
                dets.append(d)
        for _ in range(rng.integers(0, 5)):
            c = cls[rng.integers(len(cls))]
            t = rng.uniform(-50, 50, 2)
            dets.append({'translation': [float(t[0]), float(t[1]), 0.0], 'size': list(NUSC_SIZE[c]), 'yaw': 0.3,
                         'class': c, 'velocity': [0.0, 0.0] if velocity else None, 'attribute': NUSC_ATTR[c][0],
                         'score': float(np.round(rng.uniform(0, 0.8), 2))})
        out.append({'image_id': f's{i:04d}', 'gt': gts, 'det': dets, 'drive': i // 30})
    return out


NUSC_REF = r'''
import json, math, sys
import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import DETECTION_NAMES, TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig, DetectionMetrics

spec = json.load(open(sys.argv[1]))

def box(tok, b, score):
    t = list(b['translation'])
    v = b.get('velocity') or [math.nan, math.nan]
    q = [math.cos(b['yaw'] / 2), 0.0, 0.0, math.sin(b['yaw'] / 2)]
    return DetectionBox(sample_token=tok, translation=t, size=b['size'], rotation=q, velocity=v, ego_translation=t,
                        num_pts=10, detection_name=b['class'], detection_score=score,
                        attribute_name=b.get('attribute') or '')

out = []
for samples in spec['datasets']:
    gt, pr = EvalBoxes(), EvalBoxes()
    for s in samples:
        gt.add_boxes(s['image_id'], [box(s['image_id'], b, -1.0) for b in s['gt']])
        pr.add_boxes(s['image_id'], [box(s['image_id'], b, b['score']) for b in s['det']])
    cfg = DetectionConfig({c: 1000 for c in DETECTION_NAMES}, 'center_distance', [0.5, 1.0, 2.0, 4.0], 2.0, 0.1, 0.1,
                          500, 5)
    metrics = DetectionMetrics(cfg)       # mAP and NDS over the devkit's 10 classes (5 without GT: AP 0, errors 1)
    res = {'ap': {}, 'tp': {}}
    for c in DETECTION_NAMES:
        for th in cfg.dist_ths:
            md = accumulate(gt, pr, c, center_distance, th)
            ap = calc_ap(md, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(c, th, ap)
            res['ap'][f'{c}|{th}'] = ap
            if th == cfg.dist_th_tp:
                for m in TP_METRICS:      # DetectionEval.evaluate's undefined cases
                    if (c == 'traffic_cone' and m in ('attr_err', 'vel_err', 'orient_err')) or \
                       (c == 'barrier' and m in ('attr_err', 'vel_err')):
                        tp = float('nan')
                    else:
                        tp = calc_tp(md, cfg.min_recall, m)
                    metrics.add_label_tp(c, m, tp)
                    res['tp'][f'{c}|{m}'] = None if math.isnan(tp) else tp
    res['mAP'] = metrics.mean_ap
    res['tp_errors'] = metrics.tp_errors
    res['NDS'] = metrics.nd_score
    out.append(res)
json.dump(out, sys.stdout)
'''


def test_nuscenes_matches_the_devkit(tmp_path):
    datasets = [synth_nuscenes(150, seed=1), synth_nuscenes(60, seed=2, velocity=False)]
    ref = _run_reference(tmp_path, NUSC_REF, {'datasets': datasets}, 'nuscenes')
    all_classes = [k.split('|')[0] for k in ref[0]['ap']][::4]
    assert len(all_classes) == 10 and set(NUSC_SIZE) < set(all_classes)
    diffs = []
    for samples, rf in zip(datasets, ref):
        res = M.nuscenes_evaluate(samples, classes=all_classes)
        for key, ap in rf['ap'].items():
            c, th = key.split('|')
            diffs.append(abs(res['per_class'][c]['AP_by_dist'][float(th)] - ap))
        for key, tp in rf['tp'].items():
            c, m = key.split('|')
            mine = res['per_class'][c][m]
            if tp is None:
                assert math.isnan(mine), key
            else:
                diffs.append(abs(mine - tp))
        diffs.append(abs(res['mAP'] - rf['mAP']))
        diffs.append(abs(res['NDS'] - rf['NDS']))
        for m, name in N.TP_NAMES.items():
            if res[name] is not None:
                diffs.append(abs(res[name] - rf['tp_errors'][m]))
    assert max(diffs) < 1e-6, max(diffs)
    full, novel = (M.nuscenes_evaluate(d) for d in datasets)
    assert 0.1 < full['mAP'] < 0.9 and full['mAVE'] is not None
    assert novel['mAVE'] is None and ref[1]['tp_errors']['vel_err'] == 1.0   # no velocities: reported None, scored 0


def test_nuscenes_by_hand():
    car = {'size': [2.0, 4.0, 1.5], 'class': 'car', 'yaw': 0.0, 'velocity': [1.0, 0.0], 'attribute': 'vehicle.moving'}
    s = {'image_id': 'a', 'gt': [dict(car, translation=[0.0, 0.0, 0.0])],
         'det': [dict(car, translation=[0.24, 0.32, 0.0], score=0.9, yaw=0.5, size=[2.0, 4.0, 3.0])]}
    r = M.nuscenes_evaluate([s])
    assert r['AP_by_dist'] == {0.5: pytest.approx(1.0), 1.0: pytest.approx(1.0), 2.0: pytest.approx(1.0),
                               4.0: pytest.approx(1.0)}
    assert r['mATE'] == pytest.approx(0.4) and r['mAOE'] == pytest.approx(0.5)
    assert r['mASE'] == pytest.approx(0.5)                              # volumes 12 and 24: IoU 1/2
    assert r['mAVE'] == pytest.approx(0.0) and r['mAAE'] == pytest.approx(0.0)
    assert r['NDS'] == pytest.approx((5 * 1 + 0.6 + 0.5 + 0.5 + 1 + 1) / 10)
    s['det'][0]['translation'] = [0.3, 0.4, 0.0]                        # 0.5 m is not < 0.5 m: missed at the tightest
    assert M.nuscenes_evaluate([s])['AP_by_dist'][0.5] == 0.0


# --- tracking ----------------------------------------------------------------------------------------------------

def box(x, y=0.0, w=10.0, h=10.0):
    return [float(x), float(y), w, h]


def seq_from(rows):
    """rows: per frame (gt {id: bbox}, det {id: bbox})."""
    return [{'gt': [{'id': k, 'bbox': b} for k, b in g.items()], 'det': [{'id': k, 'bbox': b} for k, b in d.items()]}
            for g, d in rows]


def test_perfect_tracking_scores_one():
    rows = [({'A': box(0), 'B': box(100)}, {1: box(0), 2: box(100)}) for _ in range(5)]
    r = M.tracking_evaluate(seq_from(rows))
    for k in ('HOTA', 'DetA', 'AssA', 'LocA', 'MOTA', 'MOTP'):
        assert r[k] == pytest.approx(1.0)
    assert r['IDSW'] == 0 and r['FP'] == 0 and r['FN'] == 0 and r['clear']['MT'] == 2


def test_id_swap_by_hand():
    # A, B on 4 frames; the tracker's ids swap after frame 2: every detection is right, half the association wrong
    rows = [({'A': box(0), 'B': box(100)}, {1: box(0), 2: box(100)})] * 2 + \
           [({'A': box(0), 'B': box(100)}, {2: box(0), 1: box(100)})] * 2
    r = M.tracking_evaluate(seq_from(rows))
    assert r['DetA'] == pytest.approx(1.0) and r['LocA'] == pytest.approx(1.0)
    assert r['AssA'] == pytest.approx(1 / 3)                            # TPA 2, FNA 2, FPA 2
    assert r['HOTA'] == pytest.approx(math.sqrt(1 / 3))
    assert r['IDSW'] == 2 and r['MOTA'] == pytest.approx((8 - 0 - 2) / 8)


def test_fragmentation_gap_and_false_positive_by_hand():
    # GT A on 4 frames; tracker 1 on frames 1, 2, 4 (missed on 3); a stray tracker 2 on frame 3
    rows = [({'A': box(0)}, {1: box(0)}), ({'A': box(0)}, {1: box(0)}), ({'A': box(0)}, {2: box(500)}),
            ({'A': box(0)}, {1: box(0)})]
    r = M.tracking_evaluate(seq_from(rows))
    assert r['DetA'] == pytest.approx(3 / 5)                            # TP 3, FN 1, FP 1
    assert r['AssA'] == pytest.approx(3 / 4)                            # TPA 3 / (4 + 3 − 3)
    assert r['HOTA'] == pytest.approx(math.sqrt(0.45))
    c = r['clear']
    assert (c['TP'], c['FN'], c['FP'], c['IDSW'], c['Frag']) == (3, 1, 1, 0, 1)
    assert c['MOTA'] == pytest.approx(0.5) and (c['MT'], c['PT'], c['ML']) == (0, 1, 0)
    # TrackEval's rule: a frame without any tracker output does not break the track, so no fragment
    rows[2] = ({'A': box(0)}, {})
    assert M.clear_mot(seq_from(rows))['Frag'] == 0
    # two trackers covering one GT in turn: one IDSW, AssA 1/2, no fragmentation
    rows = [({'A': box(0)}, {1: box(0)})] * 2 + [({'A': box(0)}, {2: box(0)})] * 2
    r = M.tracking_evaluate(seq_from(rows))
    assert r['AssA'] == pytest.approx(0.5) and r['IDSW'] == 1 and r['Frag'] == 0 and r['MOTA'] == pytest.approx(0.75)


def test_hota_over_localisation_thresholds_by_hand():
    iou = 70 / 130                                                     # boxes shifted by 3 px: IoU 0.538
    rows = [({'A': box(0)}, {1: box(3)}), ({'A': box(0)}, {1: box(0)})]
    r = M.hota(seq_from(rows))
    lo = np.array(r['alphas']) < iou                                    # α = 0.05 … 0.50: both frames match
    assert lo.sum() == 10
    det = np.where(lo, 1.0, 1 / 3)                                      # above: TP 1, FN 1, FP 1
    ass = np.where(lo, 1.0, 1 / 3)                                      # TPA 1 / (2 + 2 − 1)
    loc = np.where(lo, (iou + 1) / 2, 1.0)
    assert r['DetA'] == pytest.approx(det.mean()) and r['AssA'] == pytest.approx(ass.mean())
    assert r['LocA'] == pytest.approx(loc.mean()) and r['HOTA'] == pytest.approx(np.sqrt(det * ass).mean())
    assert r['HOTA(0)'] == pytest.approx(1.0) and r['LocA(0)'] == pytest.approx((iou + 1) / 2)


def test_clear_keeps_existing_tracks_before_similarity():
    # centre similarity 1 − d / 2: in frame 2 the pure-similarity assignment would swap the ids
    c = lambda x: [float(x), 0.0]
    seq = [{'gt': [{'id': 'A', 'center': c(0)}, {'id': 'B', 'center': c(10)}],
            'det': [{'id': 1, 'center': c(0)}, {'id': 2, 'center': c(10)}]},
           {'gt': [{'id': 'A', 'center': c(0)}, {'id': 'B', 'center': c(1)}],
            'det': [{'id': 1, 'center': c(0.8)}, {'id': 2, 'center': c(0.2)}]}]
    r = M.clear_mot(seq, similarity='center', zero_distance=2.0)
    assert r['IDSW'] == 0 and r['MOTA'] == pytest.approx(1.0)
    assert r['MOTP'] == pytest.approx((1 + 1 + 0.6 + 0.6) / 4)


def test_swapping_ids_lowers_assa_not_deta():
    rng = np.random.default_rng(5)
    for trial in range(5):
        n_obj, n_fr = 4, 12
        start = rng.uniform(0, 1000, (n_obj, 2))
        rows_ok, rows_sw = [], []
        for f in range(n_fr):
            g = {k: box(*(start[k] + 3 * f)) for k in range(n_obj)}
            d = {k + 10: box(*(start[k] + 3 * f + rng.normal(0, 1, 2))) for k in range(n_obj) if rng.random() > 0.1}
            rows_ok.append((g, d))
            sw = {(21 if k == 10 else 20 if k == 11 else k) if f >= n_fr // 2 else k: b for k, b in d.items()}
            rows_sw.append((g, sw))
        a, b = M.hota(seq_from(rows_ok)), M.hota(seq_from(rows_sw))
        assert b['DetA'] == pytest.approx(a['DetA'], abs=1e-12) and b['LocA'] == pytest.approx(a['LocA'], abs=1e-12)
        assert b['AssA'] < a['AssA'] - 0.05 and b['HOTA'] < a['HOTA']
    # several sequences combine by their counts
    s1, s2 = seq_from(rows_ok), seq_from(rows_sw)
    both = M.hota([s1, s2])
    assert both['TP'] == [x + y for x, y in zip(M.hota(s1)['TP'], M.hota(s2)['TP'])]
    assert min(M.hota(s1)['HOTA'], M.hota(s2)['HOTA']) <= both['HOTA'] <= max(M.hota(s1)['HOTA'], M.hota(s2)['HOTA'])


def test_linear_assignment_matches_scipy():
    lsa = pytest.importorskip('scipy.optimize').linear_sum_assignment
    rng = np.random.default_rng(0)
    for _ in range(500):
        n, m = rng.integers(0, 9, 2)
        c = rng.random((n, m)) * rng.choice([1, 100])
        if rng.random() < 0.3:
            c = np.round(c, 1)                                         # ties
        r, k = M.linear_assignment(c)
        r2, k2 = lsa(c)
        assert len(r) == len(r2) and len(set(k.tolist())) == len(k)
        assert c[r, k].sum() == pytest.approx(c[r2, k2].sum(), abs=1e-9)
        assert np.all(np.diff(r) > 0)
    c = rng.random((30, 40))
    assert c[M.linear_assignment(c, maximize=True)].sum() == pytest.approx(c[lsa(c, maximize=True)].sum())


# --- segmentation ------------------------------------------------------------------------------------------------

def test_segmentation_against_a_direct_count():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 5, (40, 60))
    gt[rng.random(gt.shape) < 0.1] = 255
    pred = np.where(rng.random(gt.shape) < 0.7, np.minimum(gt, 4), rng.integers(0, 5, gt.shape))
    pred[gt == 255] = rng.integers(0, 5, (gt == 255).sum())
    gt[gt == 3] = 2                                                     # class 3 never in the GT
    pred[pred == 4] = 0                                                 # class 4 never predicted
    conf = M.confusion_matrix(gt, pred, 6)
    r = M.segmentation_metrics(conf)
    keep = gt != 255
    for c in range(6):
        inter = np.sum((gt == c) & (pred == c) & keep)
        union = np.sum(((gt == c) | (pred == c)) & keep)
        assert (r['IoU'][c] is None) == (union == 0)
        if union:
            assert r['IoU'][c] == pytest.approx(inter / union)
    assert r['IoU'][5] is None and r['IoU'][4] is not None and r['IoU'][3] is not None
    assert r['mIoU'] == pytest.approx(np.mean([v for v in r['IoU'] if v is not None]))
    assert r['pixel_accuracy'] == pytest.approx(np.mean(gt[keep] == pred[keep]))
    assert r['n_pixels'] == keep.sum()
    halves = [M.confusion_matrix(gt[:20], pred[:20], 6), M.confusion_matrix(gt[20:], pred[20:], 6)]
    assert M.segmentation_metrics(halves)['mIoU'] == pytest.approx(r['mIoU'])
    with pytest.raises(ValueError):
        M.confusion_matrix(gt, pred + 3, 6)


# --- lanes -------------------------------------------------------------------------------------------------------

def test_tusimple_by_hand():
    ys = list(range(160, 720, 10))
    n = len(ys)
    gt = [[300.0] * n, [600.0] * n]
    r = M.tusimple([{'gt': gt, 'det': [[310.0] * n, [640.0] * n], 'y_samples': ys}])
    assert r['accuracy'] == pytest.approx(0.5) and r['FP'] == pytest.approx(0.5) and r['FN'] == pytest.approx(0.5)
    # rows where neither lane exists count as correct (the reference's quirk)
    half = [300.0] * (n // 2) + [-2.0] * (n - n // 2)
    assert M.tusimple_bench([half], [half], ys) == (1.0, 0.0, 0.0)
    # a slanted lane gets the wider threshold 20 / cos θ
    slope = 0.5
    slant = [300.0 + slope * (y - 160) for y in ys]
    off = 20.0 / math.cos(math.atan(slope)) - 0.5                       # 21.86 px: inside the widened threshold
    acc, fp, fn = M.tusimple_bench([[x + off for x in slant]], [slant], ys)
    assert acc == pytest.approx(1.0) and fp == 0 and fn == 0
    assert M.tusimple_bench([[300.0] * n] * 3, [[300.0] * n] * 0, ys) == (0.0, 0.0, 1.0)


def test_culane_by_hand_and_against_opencv():
    lane = [[800.0, 590.0 - y] for y in range(0, 300, 10)]
    shifted = [[x + 40.0, y] for x, y in lane]
    r = M.culane([{'gt': [lane], 'det': [lane, shifted]}])
    assert (r['TP'], r['FP'], r['FN']) == (1, 1, 0) and r['F1'] == pytest.approx(2 / 3)
    iou = M.lane_iou_matrix([[[x + 10.0, y] for x, y in lane]], [lane])[0, 0]
    assert iou == pytest.approx(20 / 40, abs=0.03)                      # 30 px lanes 10 px apart: ~(30−10)/(30+10)
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = np.cumsum(np.c_[rng.normal(0, 3, 30), -np.full(30, 10.0)], axis=0) + [800, 589]
        b = a + [rng.uniform(-25, 25), 0]
        masks = []
        for ln in (a, b):
            img = np.zeros((590, 1640), np.uint8)
            p = ln.astype(np.int32)
            for p1, p2 in zip(p[:-1], p[1:]):
                cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), color=255, thickness=30)
            masks.append(img > 0)
        ref = (masks[0] & masks[1]).sum() / (masks[0] | masks[1]).sum()
        assert M.lane_iou_matrix([a], [b])[0, 0] == pytest.approx(ref, abs=0.03)


# --- calibration -------------------------------------------------------------------------------------------------

def test_calibration_by_hand():
    r = M.calibration([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 0])
    assert r['ECE'] == pytest.approx(0.5 * 0.1 + 0.5 * 0.4) and r['MCE'] == pytest.approx(0.4)
    assert sum(b['count'] for b in r['bins']) == 4 and len(r['bins']) == 15
    assert M.calibration([0.0, 1.0], [0, 1])['ECE'] == 0.0              # the end points fall in the end bins
    rng = np.random.default_rng(0)
    p = rng.random(200000)
    assert M.calibration(p, rng.random(p.size) < p)['ECE'] < 0.01       # calibrated by construction
    img = {'image_id': 1, 'gt': [{'bbox': [0, 0, 10, 10], 'class': 'car'}],
           'det': [{'bbox': [0, 0, 10, 10], 'class': 'car', 'score': 0.8},
                   {'bbox': [1, 0, 10, 10], 'class': 'car', 'score': 0.6},   # a duplicate: FP
                   {'bbox': [50, 0, 10, 10], 'class': 'car', 'score': 0.3}]}
    d = M.detection_calibration([img])
    s, y = M.detection_correctness([img])
    assert sorted(zip(s.tolist(), y.tolist())) == [(0.3, False), (0.6, False), (0.8, True)]
    assert d['n'] == 3 and d['ECE'] == pytest.approx((0.2 + 0.6 + 0.3) / 3)


# --- uncertainty -------------------------------------------------------------------------------------------------

def _mean(items):
    return float(np.mean(items))


def test_bootstrap_ci_covers_the_truth_at_about_the_nominal_rate():
    rng = np.random.default_rng(1)
    hits = 0
    reps = 200
    for k in range(reps):
        x = list(rng.normal(0.3, 1.0, 30))
        r = M.bootstrap(_mean, x, n=300, seed=k)
        hits += r['ci_low'] <= 0.3 <= r['ci_high']
    assert 0.88 <= hits / reps <= 0.99, hits / reps                     # nominal 0.95; percentile slightly under


def test_bootstrap_is_seeded_and_resamples_whole_drives():
    rng = np.random.default_rng(2)
    drive_effect = rng.normal(0, 1, 10)
    items = [{'drive': d, 'x': drive_effect[d] + rng.normal(0, 0.1)} for d in range(10) for _ in range(20)]
    fn = lambda its: float(np.mean([i['x'] for i in its]))
    a = M.bootstrap(fn, items, n=200, seed=7)
    assert a == M.bootstrap(fn, items, n=200, seed=7)
    assert a['unit'] == 'drive' and a['n_units'] == 10 and a['n_items'] == 200
    b = M.bootstrap(fn, items, n=200, seed=7, unit='image')
    assert (a['ci_high'] - a['ci_low']) > 3 * (b['ci_high'] - b['ci_low'])   # correlated frames: images understate
    c = M.bootstrap(fn, items, n=200, seed=7, method='bca')
    assert c['method'] == 'bca' and c['ci_low'] < c['value'] < c['ci_high']


def test_paired_bootstrap_is_narrower_than_unpaired_for_correlated_models():
    rng = np.random.default_rng(3)
    difficulty = rng.normal(0.5, 0.2, 80)
    items = [{'a': d + rng.normal(0, 0.02), 'b': d + 0.02 + rng.normal(0, 0.02)} for d in difficulty]
    fn = lambda its, m: float(np.mean([i[m] for i in its]))
    p = M.paired_bootstrap(fn, items, 'a', 'b', n=400, seed=0)
    ua = M.bootstrap(lambda its: fn(its, 'a'), items, n=400, seed=1)
    ub = M.bootstrap(lambda its: fn(its, 'b'), items, n=400, seed=2)
    unpaired_width = 2 * 1.96 * math.hypot(ua['sd'], ub['sd'])
    assert p['ci_high'] - p['ci_low'] < 0.3 * unpaired_width
    assert p['diff'] == pytest.approx(p['b'] - p['a']) and p['ci_low'] > 0 and p['p_b_gt_a'] == 1.0
    assert p['unit'] == 'image' and p['n_units'] == 80


def test_bootstrap_of_coco_records_and_summary_table():
    images = synth_coco(60, seed=4, attrs=False)
    recs = M.coco_match(images)
    full = M.coco_summarise(recs)
    r = M.bootstrap(lambda rs: {k: M.coco_summarise(rs)[k] for k in ('AP', 'AP50', 'AR100')}, recs, n=40, seed=0)
    assert r['unit'] == 'drive' and r['n_units'] == 3
    assert r['metrics']['AP']['value'] == pytest.approx(full['AP'])
    assert r['metrics']['AP']['ci_low'] <= full['AP'] <= r['metrics']['AP']['ci_high']
    rows = M.summary_table(r)
    assert [x['metric'] for x in rows] == ['AP', 'AP50', 'AR100'] and all(x['n'] == 3 for x in rows)
    rows = M.summary_table({'coco': r, 'mIoU': 0.5, 'notes': 'x', 'per_class': {'car': 1.0}}, n=60)
    assert rows[0]['metric'] == 'coco.AP' and rows[-1] == {'metric': 'mIoU', 'value': 0.5, 'ci_low': None,
                                                           'ci_high': None, 'n': 60}
    assert len(rows) == 4
    p = M.paired_bootstrap(lambda its, m: float(np.mean([i[m] for i in its])), [{'a': 1, 'b': 2}] * 5, 'a', 'b', n=10)
    assert M.summary_table({'speed': p})[0]['metric'] == 'speed (B − A)'


def test_metrics_package_imports_numpy_only():
    code = ('import sys; import falcon_autonomy.core.metrics; '
            'bad = [m for m in ("scipy", "cv2", "pycocotools", "nuscenes", "sklearn", "torch") if m in sys.modules]; '
            'print(bad)')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == '[]'
