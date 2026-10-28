"""Lane detection metrics (numpy): TuSimple accuracy / FP / FN and CULane F1.

TuSimple (TuSimple lane detection benchmark 2017, the reference is its evaluate/lane.py LaneEval.bench, reproduced
including its quirks). Input: images {'gt': [lane], 'det': [lane], 'y_samples': [y]}, a lane being the x of the lane at
each y sample (negative = no point there). Per image and GT lane, the threshold is 20 px / cos θ, θ the lane's angle
from the vertical (least-squares slope of x over y of its points). A predicted lane's accuracy for a GT lane = the
share of the y samples where |x_pred − x_gt| < threshold, counting rows where both have no point as correct (as the
reference does: both are set to −100). Each GT lane takes its best predicted lane (predictions may be reused); it is
matched when that accuracy ≥ 0.85. Per image: accuracy = Σ best accuracies / max(min(4, #GT), 1), FP = (#pred −
#matched) / #pred, FN = #unmatched / max(min(4, #GT), 1); with more than 4 GT lanes the worst accuracy is dropped and
one FN forgiven. An image with more than #GT + 2 predictions scores (0, 0, 1). The reported values are the means over
images. (The reference's 200 ms runtime rule is not applied.)

CULane (Pan et al. 2018, Spatial As Deep: Spatial CNN for Traffic Scene Understanding, AAAI; the metric of the
official evaluate tool as reimplemented by LaneATT's culane_metric, Tabelini et al. 2021). Input: images {'gt':
[lane], 'det': [lane]}, a lane a polyline [[x, y], ...] in pixels of a 1640 × 590 image (configurable). Each lane is
rasterised 30 px wide; the lane IoU is |A ∩ B| / |A ∪ B| of the rasterised pixels; predictions and GT are matched by
an optimal assignment on IoU (assignment.py) and a pair is a TP when IoU > 0.5. Precision, recall and
F1 = 2·TP / (2·TP + FP + FN) from the counts summed over images.
Rasterisation here: the points cast to integers (as the reference), each pixel centre within 15 px of a segment (a
capsule); the reference draws segments with OpenCV's 30 px thick lines (also round-capped) after a cubic-spline
resampling of the points, so give densely sampled lanes; IoUs agree to about a percent, not exactly.

Limits. TuSimple's accuracy rewards rows where neither lane exists and its FP/FN are per-image rates; CULane's F1
depends on the rasterisation width and the image size; neither checks lane type or colour.
"""
from __future__ import annotations

import numpy as np

from .assignment import linear_assignment

TUSIMPLE_PIXEL_THRESH = 20.0
TUSIMPLE_PT_THRESH = 0.85
CULANE_WIDTH = 30
CULANE_IOU = 0.5
CULANE_SHAPE = (590, 1640)                       # (height, width)


def _tusimple_angle(xs: np.ndarray, ys: np.ndarray) -> float:
    ok = xs >= 0
    x, y = xs[ok], ys[ok]
    if len(x) <= 1:
        return 0.0
    vy = np.sum((y - y.mean()) ** 2)
    k = np.sum((y - y.mean()) * (x - x.mean())) / vy if vy > 0 else 0.0
    return float(np.arctan(k))


def _line_accuracy(pred: np.ndarray, gt: np.ndarray, thresh: float) -> float:
    p = np.where(pred >= 0, pred, -100.0)
    g = np.where(gt >= 0, gt, -100.0)
    return float(np.sum(np.abs(p - g) < thresh) / len(g))


def tusimple_bench(pred, gt, y_samples, pixel_thresh=TUSIMPLE_PIXEL_THRESH, pt_thresh=TUSIMPLE_PT_THRESH):
    """(accuracy, FP rate, FN rate) of one image, TuSimple's LaneEval.bench without the runtime rule."""
    ys = np.asarray(y_samples, dtype=float)
    pred = [np.asarray(p, dtype=float) for p in pred]
    gt = [np.asarray(g, dtype=float) for g in gt]
    if any(len(p) != len(ys) for p in pred) or any(len(g) != len(ys) for g in gt):
        raise ValueError('every lane needs one x per y sample')
    if len(gt) + 2 < len(pred):
        return 0.0, 0.0, 1.0
    threshs = [pixel_thresh / np.cos(_tusimple_angle(g, ys)) for g in gt]
    accs, fn, matched = [], 0.0, 0.0
    for g, th in zip(gt, threshs):
        best = max((_line_accuracy(p, g, th) for p in pred), default=0.0)
        if best < pt_thresh:
            fn += 1
        else:
            matched += 1
        accs.append(best)
    fp = len(pred) - matched
    if len(gt) > 4 and fn > 0:
        fn -= 1
    s = sum(accs)
    if len(gt) > 4:
        s -= min(accs)
    den = max(min(4.0, len(gt)), 1.0)
    return s / den, (fp / len(pred) if len(pred) > 0 else 0.0), fn / den


def tusimple(images, pixel_thresh=TUSIMPLE_PIXEL_THRESH, pt_thresh=TUSIMPLE_PT_THRESH) -> dict:
    """TuSimple's means over images: {'accuracy', 'FP', 'FN', 'n_images', 'n_gt_lanes', 'n_det_lanes'}."""
    if not images:
        raise ValueError('no images')
    r = np.array([tusimple_bench(im.get('det') or [], im.get('gt') or [], im['y_samples'], pixel_thresh, pt_thresh)
                  for im in images])
    return {'accuracy': float(r[:, 0].mean()), 'FP': float(r[:, 1].mean()), 'FN': float(r[:, 2].mean()),
            'n_images': len(images), 'n_gt_lanes': sum(len(im.get('gt') or []) for im in images),
            'n_det_lanes': sum(len(im.get('det') or []) for im in images)}


def lane_pixels(lane, shape=CULANE_SHAPE, width: float = CULANE_WIDTH) -> np.ndarray:
    """Flat indices (sorted, unique) of the pixels of a polyline drawn width px wide (capsules around its segments)
    in an image of shape (height, width)."""
    H, W = shape
    p = np.asarray(lane, dtype=float).reshape(-1, 2).astype(np.int64).astype(float)
    r = width / 2.0
    out = []
    for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
        xa, xb = int(max(0, np.floor(min(x0, x1) - r))), int(min(W - 1, np.ceil(max(x0, x1) + r)))
        ya, yb = int(max(0, np.floor(min(y0, y1) - r))), int(min(H - 1, np.ceil(max(y0, y1) + r)))
        if xa > xb or ya > yb:
            continue
        X, Y = np.meshgrid(np.arange(xa, xb + 1, dtype=float), np.arange(ya, yb + 1, dtype=float))
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        u = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1) if L2 > 0 else np.zeros_like(X)
        d2 = (X - x0 - u * dx) ** 2 + (Y - y0 - u * dy) ** 2
        ok = d2 <= r * r
        out.append((Y[ok] * W + X[ok]).astype(np.int64))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def lane_iou_matrix(pred, gt, shape=CULANE_SHAPE, width: float = CULANE_WIDTH) -> np.ndarray:
    """(P, G) IoU of rasterised lanes."""
    P = [lane_pixels(x, shape, width) for x in pred]
    G = [lane_pixels(x, shape, width) for x in gt]
    out = np.zeros((len(P), len(G)))
    for i, a in enumerate(P):
        for j, b in enumerate(G):
            inter = len(np.intersect1d(a, b, assume_unique=True))
            union = len(a) + len(b) - inter
            out[i, j] = inter / union if union else 0.0
    return out


def culane(images, width: float = CULANE_WIDTH, iou_thresh: float = CULANE_IOU, shape=CULANE_SHAPE) -> dict:
    """CULane's counts and F1 over images: {'F1', 'precision', 'recall', 'TP', 'FP', 'FN', 'n_images'}."""
    tp = fp = fn = 0
    for im in images:
        pred, gt = im.get('det') or [], im.get('gt') or []
        if not pred or not gt:
            fp += len(pred)
            fn += len(gt)
            continue
        iou = lane_iou_matrix(pred, gt, shape, width)
        r, c = linear_assignment(iou, maximize=True)
        t = int(np.sum(iou[r, c] > iou_thresh))
        tp += t
        fp += len(pred) - t
        fn += len(gt) - t
    return {'F1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
            'precision': tp / (tp + fp) if tp + fp else None, 'recall': tp / (tp + fn) if tp + fn else None,
            'TP': tp, 'FP': fp, 'FN': fn, 'n_images': len(images)}
