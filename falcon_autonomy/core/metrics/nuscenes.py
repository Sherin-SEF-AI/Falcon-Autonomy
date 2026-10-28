"""nuScenes detection metrics (numpy): mAP over centre-distance thresholds, the TP errors mATE, mASE, mAOE, mAVE,
mAAE, and the nuScenes detection score NDS.

Input. A list of samples (frames), each {'image_id', 'gt': [box], 'det': [box + 'score'], 'drive' (optional)}, a
box being {'translation': (x, y[, z]) m, 'size': (w, l, h) m, 'yaw': rad about z, 'class', 'velocity': (vx, vy) m/s
(optional), 'attribute': str (optional), 'ego_dist' (optional, for class_range)}. Boxes of one sample share one frame
(nuScenes evaluates in the global frame); only differences are used, so any frame common to GT and detections works.
Every entry of the list is its own sample.

Definitions (Caesar et al. 2020, nuScenes: A multimodal dataset for autonomous driving, CVPR, §4.1; the reference is
nuscenes-devkit 1.2's nuscenes.eval.detection: algo.accumulate, calc_ap, calc_tp and DetectionMetrics, reproduced to
float precision):
- Matching per class and distance threshold d ∈ {0.5, 1, 2, 4} m: predictions ranked by descending score (equal scores:
  the later prediction first, as the reference's sort), each takes the untaken GT of its class in its sample with the
  smallest centre distance on the ground plane (‖Δ(x, y)‖), a match when that distance < d.
- Precision and recall per prediction, interpolated onto 101 recall points 0:.01:1 (0 beyond the reached recall).
  AP = mean over recall points > 0.1 of max(0, precision − 0.1) / (1 − 0.1) (min_recall = min_precision = 0.1);
  mAP = the mean over classes and thresholds.
- TP errors at d = 2 m, for matched pairs: ATE = centre distance (m); ASE = 1 − IoU of the boxes' sizes aligned
  (same centre and yaw: Π min(size) / (Π size_gt + Π size_det − Π min(size))); AOE = |smallest yaw difference| (rad;
  period π for 'barrier', else 2π); AVE = ‖Δ velocity (x, y)‖ (m/s); AAE = 1 − attribute accuracy (NaN where the GT
  has no attribute). Each is a running mean over the matches in score order, interpolated onto the recall points via
  the confidence, and averaged over recall points from 0.1 to the highest reached (1 when that is below 0.1, NaN where
  the class does not define it: vel/attr/orient for 'traffic_cone', vel/attr for 'barrier'); mTP = the mean over
  classes ignoring NaN.
- NDS = (5 · mAP + Σ_TP (1 − min(1, mTP))) / 10 over the five TP errors.

Where data are missing: without velocities (or attributes) on any matched pair, the running mean is all NaN and the
reference sets that error to 1 at every recall; mAVE (mAAE) is then reported None, and NDS counts its term as 0, as
the reference's score does. A class listed in `classes` with no GT gets AP 0 and errors 1 (the reference's
no-predictions case); by default the classes are those with GT.

Limits. The reference's input filtering is not done here except class_range (keep boxes whose ego distance — 'ego_dist'
or ‖(x, y)‖ of 'translation', i.e. translations relative to the ego — is below the class's range); nuScenes also drops
GT boxes without lidar or radar points and detections inside bike racks.
"""
from __future__ import annotations

import math

import numpy as np

DIST_THS = (0.5, 1.0, 2.0, 4.0)
DIST_TH_TP = 2.0
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
MEAN_AP_WEIGHT = 5
N_POINTS = 101
TP_METRICS = ('trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err')
TP_NAMES = {'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE', 'vel_err': 'mAVE', 'attr_err': 'mAAE'}
YAW_PERIOD = {'barrier': math.pi}                                   # nuScenes: a barrier is symmetric front/back
TP_UNDEFINED = {'traffic_cone': ('attr_err', 'vel_err', 'orient_err'), 'barrier': ('attr_err', 'vel_err')}
CLASS_RANGE = {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'construction_vehicle': 50, 'pedestrian': 40,
               'motorcycle': 40, 'bicycle': 40, 'traffic_cone': 30, 'barrier': 30}   # the devkit's detection_cvpr_2019


def angle_diff(x, y, period):
    """The signed smallest angle from y to x with the given period (the devkit's angle_diff), vectorised."""
    d = (np.asarray(x, dtype=float) - y + period / 2) % period - period / 2
    return np.where(d > np.pi, d - 2 * np.pi, d)


def cummean(x: np.ndarray) -> np.ndarray:
    """Running mean ignoring NaN; all NaN gives ones (the devkit's cummean)."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).sum() == len(x):
        return np.ones(len(x))
    s = np.nancumsum(x)
    n = np.cumsum(~np.isnan(x))
    return np.divide(s, n, out=np.zeros_like(s), where=n != 0)


def _box_arrays(boxes):
    n = len(boxes)
    t = np.array([list(b['translation'][:2]) for b in boxes], dtype=float).reshape(n, 2)
    size = np.array([list(b['size']) for b in boxes], dtype=float).reshape(n, 3)
    if np.any(size <= 0):
        raise ValueError('box sizes must be > 0')
    yaw = np.array([float(b.get('yaw', 0.0)) for b in boxes])
    vel = np.array([list(b['velocity'][:2]) if b.get('velocity') is not None else [math.nan, math.nan]
                    for b in boxes], dtype=float).reshape(n, 2)
    attr = [b.get('attribute') or '' for b in boxes]
    return t, size, yaw, vel, attr


def _in_range(b, class_range) -> bool:
    if not class_range or b['class'] not in class_range:
        return True
    d = b.get('ego_dist')
    d = math.hypot(*b['translation'][:2]) if d is None else float(d)
    return d < class_range[b['class']]


def nuscenes_match(samples, dist_ths=DIST_THS, dist_th_tp=DIST_TH_TP, yaw_period=None, class_range=None) -> list:
    """Per-sample matching (the sample-local half of the evaluation): records that nuscenes_summarise() accumulates;
    resample these for a bootstrap."""
    ap_ths = tuple(float(d) for d in dist_ths)
    dist_ths = ap_ths if float(dist_th_tp) in ap_ths else ap_ths + (float(dist_th_tp),)
    yaw_period = YAW_PERIOD if yaw_period is None else yaw_period
    records = []
    for s in samples:
        gts = [b for b in (s.get('gt') or ()) if _in_range(b, class_range)]
        dets = [b for b in (s.get('det') or ()) if _in_range(b, class_range)]
        per_class = {}
        for c in {b['class'] for b in gts} | {b['class'] for b in dets}:
            G = [b for b in gts if b['class'] == c]
            P = [b for b in dets if b['class'] == c]
            scores = np.array([float(b['score']) for b in P])
            order = np.lexsort((np.arange(len(P)), scores))[::-1]  # descending (score, index), the devkit's sort
            gt_t, gt_s, gt_y, gt_v, gt_a = _box_arrays(G)
            pr_t, pr_s, pr_y, pr_v, pr_a = _box_arrays(P)
            dist = np.linalg.norm(pr_t[:, None, :] - gt_t[None, :, :], axis=-1) if len(P) and len(G) else \
                np.zeros((len(P), len(G)))
            tp, err = {}, None
            for th in dist_ths:
                taken = np.zeros(len(G), dtype=bool)
                hit = np.zeros(len(P), dtype=bool)
                gt_of = np.full(len(P), -1)
                for i in order:
                    if not len(G):
                        break
                    dd = np.where(taken, np.inf, dist[i])
                    j = int(np.argmin(dd))                   # the first of equal minima, as the strict < loop
                    if dd[j] < th:
                        taken[j] = True
                        hit[i] = True
                        gt_of[i] = j
                tp[th] = hit
                if th == dist_th_tp:
                    e = np.full((len(P), len(TP_METRICS)), np.nan)
                    i = np.nonzero(hit)[0]
                    j = gt_of[i]
                    if len(i):
                        mn = np.prod(np.minimum(gt_s[j], pr_s[i]), axis=1)
                        iou = mn / (np.prod(gt_s[j], axis=1) + np.prod(pr_s[i], axis=1) - mn)
                        e[i, 0] = dist[i, j]
                        e[i, 1] = 1 - iou
                        e[i, 2] = np.abs(angle_diff(gt_y[j], pr_y[i], yaw_period.get(c, 2 * np.pi)))
                        e[i, 3] = np.linalg.norm(pr_v[i] - gt_v[j], axis=1)
                        e[i, 4] = [math.nan if gt_a[jj] == '' else 1.0 - float(gt_a[jj] == pr_a[ii])
                                   for ii, jj in zip(i, j)]
                    err = e
            per_class[c] = {'scores': scores, 'tp': tp, 'err': err, 'n_gt': len(G),
                            'has_vel': bool(np.isfinite(err[:, 3]).any()),
                            'has_attr': bool(np.isfinite(err[:, 4]).any())}
        records.append({'image_id': s.get('image_id'), 'drive': s.get('drive'), 'per_class': per_class,
                        'dist_ths': dist_ths, 'ap_ths': ap_ths, 'dist_th_tp': float(dist_th_tp)})
    return records


def _metric_data(records, c, th, with_err: bool):
    """The devkit's accumulate() for one class and threshold: (precision, confidence, {tp metric: curve}) on the
    101 recall points, or None for its no-predictions case."""
    pcs = [r['per_class'][c] for r in records if c in r['per_class']]
    npos = sum(pc['n_gt'] for pc in pcs)
    if npos == 0 or not pcs:
        return None
    scores = np.concatenate([pc['scores'] for pc in pcs])
    tp = np.concatenate([pc['tp'][th] for pc in pcs])
    if not tp.any():
        return None
    sample = np.concatenate([np.full(len(pc['scores']), k) for k, pc in enumerate(pcs)])
    local = np.concatenate([np.arange(len(pc['scores'])) for pc in pcs])
    order = np.lexsort((local, sample, scores))[::-1]           # descending score, then later in the list first
    tp_s = tp[order]
    conf = scores[order]
    tpc = np.cumsum(tp_s).astype(float)
    fpc = np.cumsum(~tp_s).astype(float)
    prec = tpc / (fpc + tpc)
    rec = tpc / float(npos)
    rec_interp = np.linspace(0, 1, N_POINTS)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf_i = np.interp(rec_interp, rec, conf, right=0)
    errs = {}
    if with_err:
        err = np.concatenate([pc['err'] for pc in pcs], axis=0)[order][tp_s]
        mconf = conf[tp_s]
        for k, name in enumerate(TP_METRICS):
            tmp = cummean(err[:, k])
            errs[name] = np.interp(conf_i[::-1], mconf[::-1], tmp[::-1])[::-1]
    return prec, conf_i, errs


def calc_ap(prec, min_recall=MIN_RECALL, min_precision=MIN_PRECISION) -> float:
    """The devkit's calc_ap on a 101-point precision curve."""
    p = np.copy(prec)[round(100 * min_recall) + 1:] - min_precision
    p[p < 0] = 0
    return float(np.mean(p)) / (1.0 - min_precision)


def calc_tp(curve, conf, min_recall=MIN_RECALL) -> float:
    """The devkit's calc_tp: the mean of a TP-error curve from min_recall to the highest recall reached."""
    first = round(100 * min_recall) + 1
    nz = np.nonzero(conf)[0]
    last = nz[-1] if len(nz) else 0
    return 1.0 if last < first else float(np.mean(curve[first:last + 1]))


def nuscenes_summarise(records, classes=None, min_recall=MIN_RECALL, min_precision=MIN_PRECISION,
                       tp_undefined=None) -> dict:
    """mAP, the TP errors and NDS from nuscenes_match() records."""
    if not records:
        raise ValueError('no samples')
    dist_ths, ap_ths, th_tp = records[0]['dist_ths'], records[0]['ap_ths'], records[0]['dist_th_tp']
    tp_undefined = TP_UNDEFINED if tp_undefined is None else tp_undefined
    if classes is None:
        classes = sorted({c for r in records for c, pc in r['per_class'].items() if pc['n_gt']}, key=str)
    per_class = {}
    for c in classes:
        aps = {}
        tps = {}
        for th in dist_ths:
            md = _metric_data(records, c, th, th == th_tp)
            aps[th] = 0.0 if md is None else calc_ap(md[0], min_recall, min_precision)
            if th == th_tp:
                for name in TP_METRICS:
                    if name in tp_undefined.get(c, ()):
                        tps[name] = math.nan
                    else:
                        tps[name] = 1.0 if md is None else calc_tp(md[2][name], md[1], min_recall)
        pcs = [r['per_class'][c] for r in records if c in r['per_class']]
        per_class[c] = {'AP': float(np.mean([aps[th] for th in ap_ths])), 'AP_by_dist': {th: aps[th] for th in ap_ths},
                        **tps,
                        'n_gt': sum(pc['n_gt'] for pc in pcs), 'n_det': sum(len(pc['scores']) for pc in pcs),
                        'n_tp': sum(int(pc['tp'][th_tp].sum()) for pc in pcs),
                        'has_vel': any(pc['has_vel'] for pc in pcs), 'has_attr': any(pc['has_attr'] for pc in pcs)}
    mean_ap = float(np.mean([per_class[c]['AP_by_dist'][th] for c in classes for th in ap_ths])) if classes else 0.0
    out = {'mAP': mean_ap, 'AP_by_dist': {th: float(np.mean([per_class[c]['AP_by_dist'][th] for c in classes]))
                                          if classes else 0.0 for th in ap_ths}}
    scores = []
    for name in TP_METRICS:
        vals = np.array([per_class[c][name] for c in classes], dtype=float)
        err = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
        has = {'vel_err': 'has_vel', 'attr_err': 'has_attr'}.get(name)
        missing = has is not None and not any(per_class[c][has] for c in classes)
        out[TP_NAMES[name]] = None if (missing or math.isnan(err)) else err
        scores.append(0.0 if math.isnan(err) else max(0.0, 1.0 - err))
    out['NDS'] = (MEAN_AP_WEIGHT * mean_ap + sum(scores)) / (MEAN_AP_WEIGHT + len(TP_METRICS))
    out['per_class'] = per_class
    out['n_samples'] = len(records)
    out['n_classes'] = len(classes)
    out['n_gt'] = sum(v['n_gt'] for v in per_class.values())
    out['n_det'] = sum(v['n_det'] for v in per_class.values())
    out['dist_ths'] = list(ap_ths)
    return out


def nuscenes_evaluate(samples, classes=None, dist_ths=DIST_THS, dist_th_tp=DIST_TH_TP, class_range=None,
                      yaw_period=None, tp_undefined=None) -> dict:
    """nuScenes detection metrics of a list of samples: {'mAP', 'AP_by_dist', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE',
    'NDS', 'per_class': {class: {'AP', 'AP_by_dist', 'trans_err', ..., 'n_gt', 'n_det', 'n_tp'}}, 'n_samples',
    'n_classes', 'n_gt', 'n_det'}. class_range: {class: m} (CLASS_RANGE for nuScenes' own)."""
    return nuscenes_summarise(nuscenes_match(samples, dist_ths, dist_th_tp, yaw_period, class_range), classes,
                              tp_undefined=tp_undefined)
