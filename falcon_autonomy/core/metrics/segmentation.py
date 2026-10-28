"""Semantic segmentation metrics (numpy): the confusion matrix, per-class IoU, mIoU, pixel accuracy.

Definitions (Everingham et al. 2015, The PASCAL Visual Object Classes Challenge: A Retrospective, IJCV 111, §4.3;
Cordts et al. 2016, The Cityscapes Dataset, CVPR; as computed by mmsegmentation's IoUMetric and torchmetrics'
JaccardIndex):
- Confusion matrix M (C × C): M[g, p] = the number of pixels of ground-truth class g predicted as p, over the pixels
  whose ground truth is not ignore_index. Matrices of several images add.
- IoU_c = M[c, c] / (Σ_p M[c, p] + Σ_g M[g, c] − M[c, c]): TP / (TP + FP + FN). None when the class is in neither the
  ground truth nor the prediction (union 0).
- mIoU = the mean of IoU_c over the classes with union > 0 (a class only predicted counts, with IoU 0).
- Pixel accuracy = trace(M) / ΣM; mean class accuracy = the mean over classes with GT pixels of M[c, c] / Σ_p M[c, p].

Limits. Plain IoU, not Cityscapes' instance-level iIoU; mIoU over the classes present in the evaluated pixels, so
it is comparable only between evaluations of the same images (pass a confusion matrix summed over the whole set).
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(gt, pred, n_classes: int, ignore_index: int | None = 255, out=None) -> np.ndarray:
    """The (n_classes, n_classes) int64 confusion matrix of label images gt and pred (same shape), rows = GT;
    added to out when given."""
    g = np.asarray(gt).reshape(-1)
    p = np.asarray(pred).reshape(-1)
    if g.shape != p.shape:
        raise ValueError(f'gt and pred differ in size: {np.shape(gt)} vs {np.shape(pred)}')
    if ignore_index is not None:
        keep = g != ignore_index
        g, p = g[keep], p[keep]
    g = g.astype(np.int64)
    p = p.astype(np.int64)
    if g.size and (g.min() < 0 or g.max() >= n_classes):
        raise ValueError(f'ground-truth labels outside 0..{n_classes - 1} (and not ignore_index)')
    if p.size and (p.min() < 0 or p.max() >= n_classes):
        raise ValueError(f'predicted labels outside 0..{n_classes - 1}')
    m = np.bincount(g * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    if out is not None:
        out += m
        return out
    return m


def segmentation_metrics(conf) -> dict:
    """{'IoU': [per class, None where absent], 'mIoU', 'pixel_accuracy', 'mean_class_accuracy', 'n_pixels',
    'n_classes_present', 'gt_pixels', 'pred_pixels'} from a confusion matrix or a list of them (summed)."""
    M = np.asarray(conf, dtype=np.int64)
    if M.ndim == 3:
        M = M.sum(0)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f'a confusion matrix is square; got shape {M.shape}')
    tp = np.diag(M).astype(float)
    gt_n = M.sum(1)
    pr_n = M.sum(0)
    union = gt_n + pr_n - tp
    present = union > 0
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=present)
    has_gt = gt_n > 0
    acc_c = np.divide(tp, gt_n, out=np.zeros_like(tp), where=has_gt)
    total = int(M.sum())
    return {'IoU': [float(v) if ok else None for v, ok in zip(iou, present)],
            'mIoU': float(iou[present].mean()) if present.any() else None,
            'pixel_accuracy': float(tp.sum() / total) if total else None,
            'mean_class_accuracy': float(acc_c[has_gt].mean()) if has_gt.any() else None,
            'n_pixels': total, 'n_classes_present': int(present.sum()),
            'gt_pixels': [int(v) for v in gt_n], 'pred_pixels': [int(v) for v in pr_n]}
