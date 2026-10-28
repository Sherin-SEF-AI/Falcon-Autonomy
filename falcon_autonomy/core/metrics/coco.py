"""COCO box detection metrics (numpy): AP, AP50, AP75, AR@maxDets, by class, by COCO area and by any GT attribute.

Input. A list of images, each {'image_id', 'gt': [{'bbox': [x, y, w, h], 'class', 'ignore' (optional), 'iscrowd'
(optional), 'area' (optional, default w·h), extra attributes such as 'range_m', 'occluded', 'truncated'}], 'det':
[{'bbox': [x, y, w, h], 'class', 'score', optional attributes}], 'drive' (optional, the bootstrap's unit)}. Pixels;
a class is any hashable label. Every entry of the list is its own image (a bootstrap resample may repeat one).

Definitions (Lin et al. 2014, Microsoft COCO: Common Objects in Context, ECCV; the reference implementation is
pycocotools' COCOeval, iouType 'bbox', which this module reproduces number for number):
- IoU of boxes [x, y, w, h]: intersection / union, with no +1 on the sizes; for a crowd GT intersection / det area.
- Per image and class, the detections sorted by descending score (stable) and cut to maxDets[-1]; at each IoU
  threshold t in .50:.05:.95 each detection in turn takes the not-yet-taken GT (crowd GTs can be taken again) of
  highest IoU ≥ t, preferring a not-ignored GT to an ignored one; among equal IoUs the later GT wins (as the
  reference's loop does). A detection matched to an ignored GT is ignored (neither TP nor FP).
- A GT is ignored when 'ignore' or 'iscrowd' is set, or when its value lies outside the evaluated range (below).
- Accumulation per class: the detections of all images (each image cut to maxDet) sorted by descending score
  (stable, in image order), cumulative TP and FP; recall = TP / #not-ignored GT, precision = TP / (TP + FP + ε);
  precision made monotone from the right; precision sampled at the 101 recall points 0:.01:1 (the first detection
  reaching each recall, 0 beyond the last). AP = the mean over IoU thresholds, recall points and classes with at
  least one not-ignored GT; AR@k = the final recall with k detections per image, averaged likewise.
- Reported: AP, AP50, AP75, AP_<area> and AR_<area> (maxDets[-1]) for COCO's area ranges small [0, 32²], medium
  [32², 96²], large [96², 1e10] (inclusive at both ends, as the reference), AR<k> for each maxDets k, per class.

Breakdowns (an extension built on COCO's area-range rule, not part of COCO): for an attribute and bins (edges
e0 < e1 < ...; bin i holds values in [e_i, e_{i+1}), half-open), each bin is evaluated as COCO evaluates an area range:
GTs whose value is outside the bin (or missing) are ignored, so a detection matched to one is ignored too; an
unmatched detection is ignored when its value is outside the bin. A detection's value: its own entry under the
attribute when it has one (e.g. a detector's range estimate); for 'height' its own box height (pixel-height bins: the
COCO rule exactly, like area); otherwise the value of the GT of any class in its image with the highest IoU > 0 to it
(its nearest GT). A detection with no such value (no overlapping GT, or that GT lacks the attribute) is, by
`unassigned`: 'fp' (default, conservative) counted in every bin — it may be a false positive at any range — or
'ignore', in none. The default attributes are those of core.groundtruth's objects: 'range_m' (camera to box centre,
m), 'occluded' and 'truncated' (shares, 0–1), and 'height' (bbox h, px).

Limits. Boxes only (no segmentation masks or keypoints); detections whose class has no GT anywhere are not scored
(COCO scores only the dataset's categories, and a category without GT has no AP); when two detections of different
images have exactly equal scores, they are ranked in list order where the reference ranks them in sorted image-id
order (an input sorted by image_id gives identical numbers). None where the reference reports -1 (nothing to score).
"""
from __future__ import annotations

import math

import numpy as np

IOU_THRS = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)     # as pycocotools Params
REC_THRS = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
MAX_DETS = (1, 10, 100)
AREA_RANGES = {'all': (0.0, 1e5 ** 2), 'small': (0.0, 32.0 ** 2), 'medium': (32.0 ** 2, 96.0 ** 2),
               'large': (96.0 ** 2, 1e5 ** 2)}
DEFAULT_BREAKDOWNS = {'range_m': (0.0, 20.0, 40.0, 60.0, 80.0, math.inf),
                      'occluded': (0.0, 0.1, 0.5, math.inf),
                      'truncated': (0.0, 0.15, 0.3, 0.5, math.inf),
                      'height': (0.0, 25.0, 40.0, 80.0, 160.0, math.inf)}


def box_iou(dets, gts, crowd=None) -> np.ndarray:
    """IoU of boxes [x, y, w, h]: (D, 4), (G, 4) -> (D, G); for crowd GTs intersection / detection area
    (pycocotools' maskUtils.iou on boxes)."""
    d = np.asarray(dets, dtype=float).reshape(-1, 4)
    g = np.asarray(gts, dtype=float).reshape(-1, 4)
    w = np.minimum((d[:, 2] + d[:, 0])[:, None], (g[:, 2] + g[:, 0])[None]) - np.maximum(d[:, 0, None], g[None, :, 0])
    h = np.minimum((d[:, 3] + d[:, 1])[:, None], (g[:, 3] + g[:, 1])[None]) - np.maximum(d[:, 1, None], g[None, :, 1])
    hit = (w > 0) & (h > 0)
    inter = np.where(hit, w * h, 0.0)
    da = (d[:, 2] * d[:, 3])[:, None]
    ga = (g[:, 2] * g[:, 3])[None, :]
    union = da + ga - inter
    if crowd is not None:
        union = np.where(np.asarray(crowd, dtype=bool)[None, :], np.broadcast_to(da, union.shape), union)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(hit, inter / np.where(hit, union, 1.0), 0.0)


def _greedy_match(ious: np.ndarray, gt_ig: np.ndarray, crowd: np.ndarray, thrs: np.ndarray):
    """COCOeval.evaluateImg's matching for all ranges and thresholds at once. ious (D, G), detections in score order;
    gt_ig (A, G) per range; crowd (G,). Returns matched (A, T, D) and ignored-by-match (A, T, D)."""
    A, G = gt_ig.shape
    D, T = ious.shape[0], len(thrs)
    matched = np.zeros((A, T, D), dtype=bool)
    dt_ig = np.zeros((A, T, D), dtype=bool)
    if G == 0 or D == 0:
        return matched, dt_ig
    thr = np.minimum(thrs, 1 - 1e-10)[None, :, None]          # (1, T, 1)
    lowest = thr.min()
    taken = np.zeros((A, T, G), dtype=bool)
    once = ~crowd[None, None, :]
    ig = gt_ig[:, None, :]
    for d in range(D):
        row = ious[d]
        if row.max() < lowest:
            continue
        ok = (row[None, None, :] >= thr) & ~(taken & once)
        s = np.where(ok, row[None, None, :], -1.0)
        s_reg = np.where(ig, -1.0, s)
        s_ign = np.where(ig, s, -1.0)
        has_reg = s_reg.max(-1) >= 0
        has_ign = s_ign.max(-1) >= 0
        j_reg = G - 1 - np.argmax(s_reg[..., ::-1], axis=-1)     # the last of equal maxima, as the reference
        j_ign = G - 1 - np.argmax(s_ign[..., ::-1], axis=-1)
        hit = has_reg | has_ign
        m = np.where(has_reg, j_reg, j_ign)
        a_i, t_i = np.nonzero(hit)
        taken[a_i, t_i, m[hit]] = True
        matched[:, :, d] = hit
        dt_ig[:, :, d] = hit & ~has_reg
    return matched, dt_ig


def _ranges(area_ranges: dict, breakdowns: dict | None):
    """The evaluated ranges: [(key, attr, lo, hi, inclusive)]; area ranges first."""
    out = [(('area', name), 'area', float(lo), float(hi), True) for name, (lo, hi) in area_ranges.items()]
    for attr, edges in (breakdowns or {}).items():
        e = [float(x) for x in edges]
        if len(e) < 2 or any(b <= a for a, b in zip(e, e[1:])):
            raise ValueError(f'bins of {attr!r} must be at least two increasing edges; got {edges}')
        out += [(('bin', attr, i), attr, e[i], e[i + 1], False) for i in range(len(e) - 1)]
    return out


def _outside(vals: np.ndarray, lo: float, hi: float, inclusive: bool) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        inside = (vals >= lo) & (vals <= hi) if inclusive else (vals >= lo) & (vals < hi)
    return ~inside                                                # NaN (missing) is outside every range


def _num(x) -> float:
    return math.nan if x is None else float(x)


def coco_match(images, iou_thrs=IOU_THRS, max_dets=MAX_DETS, area_ranges=None, breakdowns=None,
               unassigned: str = 'fp') -> list:
    """Per-image matching (the expensive, image-local half of COCOeval): a list of records, one per image, that
    coco_summarise() accumulates. Bootstrap over images or drives by resampling these records."""
    if unassigned not in ('fp', 'ignore'):
        raise ValueError("unassigned is 'fp' or 'ignore'")
    thrs = np.asarray(iou_thrs, dtype=float)
    max_dets = tuple(int(m) for m in max_dets)
    area_ranges = AREA_RANGES if area_ranges is None else area_ranges
    ranges = _ranges(area_ranges, breakdowns)
    meta = {'iou_thrs': thrs, 'max_dets': max_dets, 'ranges': [r[0] for r in ranges],
            'range_bounds': [(r[2], r[3]) for r in ranges]}
    records = []
    for img in images:
        gts, dets = list(img.get('gt') or ()), list(img.get('det') or ())
        gb = np.array([g['bbox'] for g in gts], dtype=float).reshape(-1, 4)
        db = np.array([d['bbox'] for d in dets], dtype=float).reshape(-1, 4)
        g_cls = [g['class'] for g in gts]
        d_cls = [d['class'] for d in dets]
        crowd = np.array([bool(g.get('iscrowd', 0)) for g in gts], dtype=bool)
        g_ign = np.array([bool(g.get('ignore', 0)) for g in gts], dtype=bool) | crowd
        g_area = np.array([_num(g['area']) if g.get('area') is not None else b[2] * b[3] for g, b in zip(gts, gb)])
        d_area = db[:, 2] * db[:, 3]                              # pycocotools' loadRes: area = w·h
        nearest = None
        g_out, d_out = [], []
        for _key, attr, lo, hi, inclusive in ranges:
            if attr == 'area':
                gv, dv = g_area, d_area
            else:
                if attr == 'height':
                    gv = gb[:, 3].copy()
                else:
                    gv = np.array([_num(g.get(attr)) for g in gts])
                dv = np.array([_num(d.get(attr)) if d.get(attr) is not None else
                               (b[3] if attr == 'height' else math.nan) for d, b in zip(dets, db)])
                need = np.isnan(dv)
                if need.any() and len(gts):
                    if nearest is None:
                        iou_all = box_iou(db, gb)
                        nearest = np.where(iou_all.max(1) > 0, iou_all.argmax(1), -1)
                    has = need & (nearest >= 0)
                    dv[has] = gv[nearest[has]]
            g_out.append(_outside(gv, lo, hi, inclusive))
            dout = _outside(dv, lo, hi, inclusive)
            if attr != 'area' and unassigned == 'fp':
                dout &= ~np.isnan(dv)                             # no value: counted in every bin
            d_out.append(dout)
        g_out = np.array(g_out, dtype=bool).reshape(len(ranges), len(gts))
        d_out = np.array(d_out, dtype=bool).reshape(len(ranges), len(dets))
        per_class = {}
        for c in set(g_cls) | set(d_cls):
            gi = np.array([i for i, k in enumerate(g_cls) if k == c], dtype=np.int64)
            di = np.array([i for i, k in enumerate(d_cls) if k == c], dtype=np.int64)
            scores = np.array([float(dets[i]['score']) for i in di])
            order = np.argsort(-scores, kind='mergesort')[:max_dets[-1]]
            di, scores = di[order], scores[order]
            gt_ig = g_ign[gi][None, :] | g_out[:, gi]
            ious = box_iou(db[di], gb[gi], crowd[gi])
            matched, dt_ig = _greedy_match(ious, gt_ig, crowd[gi], thrs)
            dt_ig |= ~matched & d_out[:, None, di]
            per_class[c] = {'scores': scores, 'matched': matched, 'dt_ig': dt_ig,
                            'npig': (~gt_ig).sum(1), 'n_gt': len(gi), 'n_gt_ignored': int(g_ign[gi].sum())}
        records.append({'image_id': img.get('image_id'), 'drive': img.get('drive'), 'per_class': per_class,
                        'n_det': len(dets), 'meta': meta})
    return records


def _sort_key(c):
    return (type(c).__name__, c) if isinstance(c, (int, float, str)) else (type(c).__name__, repr(c))


def _mean_valid(x) -> float | None:
    v = np.asarray(x)[np.asarray(x) > -1]
    return float(v.mean()) if v.size else None


def coco_accumulate(records, rec_thrs=REC_THRS):
    """COCOeval.accumulate over match records: (classes, precision (T, R, K, A) at maxDets[-1], recall (T, K, A, M),
    meta); -1 where a class has no not-ignored GT in a range."""
    if not records:
        raise ValueError('no images')
    meta = records[0]['meta']
    thrs, max_dets, A = meta['iou_thrs'], meta['max_dets'], len(meta['ranges'])
    T, R, M = len(thrs), len(rec_thrs), len(max_dets)
    classes = sorted({c for r in records for c, pc in r['per_class'].items() if pc['n_gt']}, key=_sort_key)
    K = len(classes)
    precision = -np.ones((T, R, K, A))
    recall = -np.ones((T, K, A, M))
    for k, c in enumerate(classes):
        pcs = [r['per_class'][c] for r in records if c in r['per_class']]
        npig = np.sum([pc['npig'] for pc in pcs], axis=0)
        scores = np.concatenate([pc['scores'] for pc in pcs])
        rank = np.concatenate([np.arange(len(pc['scores'])) for pc in pcs])
        matched = np.concatenate([pc['matched'] for pc in pcs], axis=2)
        dt_ig = np.concatenate([pc['dt_ig'] for pc in pcs], axis=2)
        for m, md in enumerate(max_dets):
            sel = rank < md
            order = np.argsort(-scores[sel], kind='mergesort')
            for a in range(A):
                if npig[a] == 0:
                    continue
                tm = matched[a][:, sel][:, order]
                ti = dt_ig[a][:, sel][:, order]
                tp = np.cumsum(tm & ~ti, axis=1).astype(float)
                fp = np.cumsum(~tm & ~ti, axis=1).astype(float)
                nd = tp.shape[1]
                rc = tp / npig[a]
                recall[:, k, a, m] = rc[:, -1] if nd else 0.0
                if m != M - 1:
                    continue
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1] if nd else pr
                for t in range(T):
                    inds = np.searchsorted(rc[t], rec_thrs, side='left')
                    q = np.zeros(R)
                    ok = inds < nd
                    q[ok] = pr[t, inds[ok]]
                    precision[t, :, k, a] = q
    return classes, precision, recall, meta


def coco_summarise(records, rec_thrs=REC_THRS) -> dict:
    """The COCO numbers from coco_match() records (see the module docstring for the keys)."""
    classes, P, Rc, meta = coco_accumulate(records, rec_thrs)
    thrs, max_dets, keys = meta['iou_thrs'], meta['max_dets'], meta['ranges']
    t50 = np.nonzero(np.isclose(thrs, 0.5, atol=1e-12, rtol=0))[0]
    t75 = np.nonzero(np.isclose(thrs, 0.75, atol=1e-12, rtol=0))[0]
    first_t = np.zeros(len(thrs), dtype=bool)
    first_t[0] = True

    def ap(a, t=None, k=None):
        x = P[:, :, :, a] if t is None else P[t][:, :, :, a]
        if k is not None:
            x = x[..., k]
        return _mean_valid(x)

    def ar(a, m=-1, k=None):
        x = Rc[:, :, a, m]
        return _mean_valid(x if k is None else x[:, k])

    out = {}
    a_all = keys.index(('area', 'all')) if ('area', 'all') in keys else 0
    out['AP'] = ap(a_all)
    out['AP50'] = ap(a_all, t50) if len(t50) else None
    out['AP75'] = ap(a_all, t75) if len(t75) else None
    for a, key in enumerate(keys):
        if key[0] == 'area' and key[1] != 'all':
            out[f'AP_{key[1]}'] = ap(a)
    for m, md in enumerate(max_dets):
        out[f'AR{md}'] = ar(a_all, m)
    for a, key in enumerate(keys):
        if key[0] == 'area' and key[1] != 'all':
            out[f'AR_{key[1]}'] = ar(a)

    def n_counted(a, c=None):
        """Detections scored (TP or FP) at the first IoU threshold with maxDets[-1], and not-ignored GTs."""
        nd = ng = 0
        for r in records:
            for cc, pc in r['per_class'].items():
                if (c is None and cc in classes) or cc == c:
                    nd += int((~pc['dt_ig'][a, 0]).sum())
                    ng += int(pc['npig'][a])
        return nd, ng

    per_class = {}
    for k, c in enumerate(classes):
        nd = sum(len(r['per_class'][c]['scores']) for r in records if c in r['per_class'])
        per_class[c] = {'AP': ap(a_all, k=k), 'AP50': ap(a_all, t50, k) if len(t50) else None,
                        'AP75': ap(a_all, t75, k) if len(t75) else None, f'AR{max_dets[-1]}': ar(a_all, k=k),
                        'n_gt': n_counted(a_all, c)[1],
                        'n_gt_ignored': sum(r['per_class'][c]['n_gt_ignored'] for r in records if c in r['per_class']),
                        'n_det': nd}
    out['per_class'] = per_class
    breakdowns = {}
    for a, key in enumerate(keys):
        if key[0] != 'bin':
            continue
        lo, hi = meta['range_bounds'][a]
        nd, ng = n_counted(a)
        breakdowns.setdefault(key[1], []).append({
            'lo': lo, 'hi': hi, 'AP': ap(a), 'AP50': ap(a, t50) if len(t50) else None,
            'AP75': ap(a, t75) if len(t75) else None, f'AR{max_dets[-1]}': ar(a),
            'per_class_AP': {c: ap(a, k=k) for k, c in enumerate(classes)}, 'n_gt': ng, 'n_det': nd})
    if breakdowns:
        out['breakdowns'] = breakdowns
    out['n_images'] = len(records)
    out['n_classes'] = len(classes)
    out['n_gt'] = sum(v['n_gt'] for v in per_class.values())
    out['n_gt_ignored'] = sum(v['n_gt_ignored'] for v in per_class.values())
    out['n_det'] = sum(r['n_det'] for r in records)
    out['iou_thrs'] = [float(t) for t in thrs]
    out['max_dets'] = list(max_dets)
    return out


def coco_evaluate(images, iou_thrs=IOU_THRS, max_dets=MAX_DETS, area_ranges=None, breakdowns=None,
                  unassigned: str = 'fp') -> dict:
    """COCO box detection metrics of a list of images: {'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large', 'per_class': {class: {...}}, 'breakdowns':
    {attr: [{'lo', 'hi', 'AP', ..., 'n_gt', 'n_det'}]} (when asked), 'n_images', 'n_classes', 'n_gt',
    'n_gt_ignored', 'n_det'}. breakdowns: {attr: bin edges} (DEFAULT_BREAKDOWNS for Falcon's attributes)."""
    return coco_summarise(coco_match(images, iou_thrs, max_dets, area_ranges, breakdowns, unassigned))
