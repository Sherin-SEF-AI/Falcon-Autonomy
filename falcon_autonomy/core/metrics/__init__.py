"""Evaluation metrics of perception and tracking outputs (numpy only): what a model run in Falcon is scored with.

Each metric returns a dict of numbers and the counts they rest on; each module's docstring gives the exact definition,
the paper or standard, the reference implementation it reproduces, and its limits.
- coco: COCO box detection (AP@[.50:.05:.95], AP50, AP75, AR@{1, 10, 100}, by class, by COCO area range) as
  pycocotools' COCOeval computes it, and breakdowns by any GT attribute (range_m, occluded, truncated, pixel height)
  under COCO's area-range rule.
- nuscenes: nuScenes detection (mAP over centre distances 0.5/1/2/4 m, mATE, mASE, mAOE, mAVE, mAAE, NDS) as the
  nuscenes-devkit computes it.
- segmentation: confusion matrix, per-class IoU, mIoU, pixel accuracy.
- tracking: CLEAR MOT and HOTA as TrackEval computes them; assignment: the Hungarian method they use.
- lanes: TuSimple accuracy / FP / FN and CULane F1.
- calibration: ECE, MCE and reliability-diagram bins, also for detection scores (correct = matched at IoU 0.5).
- uncertainty: bootstrap CIs over images or whole drives (cluster bootstrap), paired model comparisons, and
  summary_table() rows for the UI.
The expensive, image-local half of a detection evaluation (coco_match, nuscenes_match) is separate from the
accumulation (coco_summarise, nuscenes_summarise), so a bootstrap resamples the matched records instead of
re-matching boxes.
"""
from __future__ import annotations

from .assignment import linear_assignment
from .calibration import calibration, detection_calibration, detection_correctness
from .coco import (AREA_RANGES, DEFAULT_BREAKDOWNS, IOU_THRS, MAX_DETS, REC_THRS, box_iou, coco_accumulate,
                   coco_evaluate, coco_match, coco_summarise)
from .lanes import culane, lane_iou_matrix, lane_pixels, tusimple, tusimple_bench
from .nuscenes import CLASS_RANGE, DIST_THS, nuscenes_evaluate, nuscenes_match, nuscenes_summarise
from .segmentation import confusion_matrix, segmentation_metrics
from .tracking import ALPHAS, clear_mot, hota, prepare_sequence, tracking_evaluate
from .uncertainty import bootstrap, paired_bootstrap, summary_table

__all__ = ['ALPHAS', 'AREA_RANGES', 'CLASS_RANGE', 'DEFAULT_BREAKDOWNS', 'DIST_THS', 'IOU_THRS', 'MAX_DETS', 'REC_THRS',
           'bootstrap', 'box_iou', 'calibration', 'clear_mot', 'coco_accumulate', 'coco_evaluate', 'coco_match',
           'coco_summarise', 'confusion_matrix', 'culane', 'detection_calibration', 'detection_correctness', 'hota',
           'lane_iou_matrix', 'lane_pixels', 'linear_assignment', 'nuscenes_evaluate', 'nuscenes_match',
           'nuscenes_summarise', 'paired_bootstrap', 'prepare_sequence', 'segmentation_metrics', 'summary_table',
           'tracking_evaluate', 'tusimple', 'tusimple_bench']
