"""Confidence calibration (numpy): expected and maximum calibration error, and the reliability diagram's data.

Definitions (Naeini, Cooper & Hauskrecht 2015, Obtaining well calibrated probabilities using Bayesian binning, AAAI;
Guo, Pleiss, Sun & Weinberger 2017, On calibration of modern neural networks, ICML; the reference is the latter's
temperature_scaling ECE): the confidences in [0, 1] are put into n equal-width bins (lo, hi] (a confidence of exactly
0 into the first bin); per bin, accuracy = the share correct and confidence = the mean confidence.
- ECE = Σ_bins (n_bin / N) · |accuracy − confidence|; MCE = the largest |accuracy − confidence| over non-empty bins.
- For a detector, correctness is COCO matching at IoU 0.5 (coco.py: per image and class, by descending score, against
  not-ignored GT; up to maxDets per image and class): a detection is correct when it is a TP; detections matched to
  ignored GT are left out (as COCO leaves them out of AP). This is the detection ECE of Küppers et al. 2020
  (Multivariate confidence calibration for object detection, CVPR workshops) without its box-position bins.

Limits. Equal-width binning is biased for small bins and depends on n_bins (15 by default, as Guo et al.); a
detection's correctness depends on the IoU threshold chosen; missed GTs (false negatives) do not enter it.
"""
from __future__ import annotations

import numpy as np

from .coco import coco_match

N_BINS = 15


def calibration(confidences, correct, n_bins: int = N_BINS) -> dict:
    """{'ECE', 'MCE', 'n', 'accuracy', 'mean_confidence', 'bins': [{'lo', 'hi', 'count', 'confidence', 'accuracy',
    'gap'}]} of confidences in [0, 1] and their correctness (bool or 0/1)."""
    c = np.asarray(confidences, dtype=float).reshape(-1)
    y = np.asarray(correct, dtype=float).reshape(-1)
    if c.shape != y.shape:
        raise ValueError('confidences and correct differ in length')
    if c.size and (np.nanmin(c) < 0 or np.nanmax(c) > 1 or np.isnan(c).any()):
        raise ValueError('confidences must lie in [0, 1]')
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, c, side='left') - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    s_conf = np.bincount(idx, weights=c, minlength=n_bins)
    s_acc = np.bincount(idx, weights=y, minlength=n_bins)
    nz = count > 0
    conf_b = np.divide(s_conf, count, out=np.zeros(n_bins), where=nz)
    acc_b = np.divide(s_acc, count, out=np.zeros(n_bins), where=nz)
    gap = np.abs(acc_b - conf_b)
    n = int(c.size)
    return {'ECE': float((count * gap).sum() / n) if n else None,
            'MCE': float(gap[nz].max()) if nz.any() else None, 'n': n,
            'accuracy': float(y.mean()) if n else None, 'mean_confidence': float(c.mean()) if n else None,
            'bins': [{'lo': float(edges[i]), 'hi': float(edges[i + 1]), 'count': int(count[i]),
                      'confidence': float(conf_b[i]) if nz[i] else None, 'accuracy': float(acc_b[i]) if nz[i] else None,
                      'gap': float(gap[i]) if nz[i] else None} for i in range(n_bins)]}


def detection_correctness(images, iou: float = 0.5, max_dets: int = 100):
    """(scores, correct) of the detections of a list of COCO-style images (see coco.py) matched at one IoU; the
    detections matched to ignored GT, and those of classes without any GT, are left out."""
    recs = coco_match(images, iou_thrs=(iou,), max_dets=(max_dets,), area_ranges={'all': (0.0, 1e5 ** 2)})
    gt_classes = {c for r in recs for c, pc in r['per_class'].items() if pc['n_gt']}
    s, y = [], []
    for r in recs:
        for c, pc in r['per_class'].items():
            if c not in gt_classes:
                continue
            keep = ~pc['dt_ig'][0, 0]
            s.append(pc['scores'][keep])
            y.append(pc['matched'][0, 0][keep])
    if not s:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return np.concatenate(s), np.concatenate(y)


def detection_calibration(images, iou: float = 0.5, n_bins: int = N_BINS, max_dets: int = 100) -> dict:
    """calibration() of a detector's scores, a detection being correct when it matches a GT at IoU ≥ iou."""
    s, y = detection_correctness(images, iou, max_dets)
    out = calibration(s, y, n_bins)
    out['iou'] = float(iou)
    return out
