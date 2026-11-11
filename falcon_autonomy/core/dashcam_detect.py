"""Road users in dashcam frames: a detector run in batches, its COCO classes mapped to Falcon's kinds, riders folded
into their two-wheelers, and the autorickshaw heuristic (numpy at import; torch and transformers only inside the
RT-DETRv2 runner).

Detectors (make_detector):
    rtdetr-v2-r18 | rtdetr-v2-r50   RT-DETRv2 (core.perception_models; Apache-2.0) through transformers'
                    RTDetrV2ForObjectDetection, loaded offline from its verified folder; frames in batches (batch 8),
                    FP16 on CUDA (FP32 on the CPU), every box above score 0.1 kept (ByteTrack's second association
                    uses the low-score ones; core.bytetrack).
    file:PATH       detections read from a file — {"frames": {"<video frame n>": [{"box": [x1, y1, x2, y2],
                    "score": s, "label": COCO name}]}} in the processed frames' pixels — for detections made
                    elsewhere and for the tests (a ground-truth sidecar of a synthetic clip). NOT a model.
YOLO (any Ultralytics model) is refused (AGPL-3.0); `ultralytics` is never imported.

Classes. COCO → Falcon: person → pedestrian, bicycle, car, motorcycle, bus, truck, cow → cattle; everything else
(trains, animals but cows, traffic lights, …) is dropped. A person whose box overlaps a motorcycle or bicycle box by
at least 30 % of its own area, whose horizontal centre lies within it and whose feet are within its lower part is its
rider (heuristic): folded into the two-wheeler, not a pedestrian.

Autorickshaws. COCO has no autorickshaw, so a detector trained on it calls one a car, a truck or a motorcycle. The
heuristic (autorickshaw()): a car, truck or motorcycle box whose size in metres — its pixel width and height at the
flat-ground range of its bottom edge (core.dashcam_depth) — is 1.1–1.6 m wide and 1.4–2.0 m tall, taller than wide
(height/width 1.05–1.6), becomes an autorickshaw with `class_source: "heuristic"` (a real class needs a detector
fine-tuned on labelled autorickshaws; Falcon's simulated ones are stand-ins: docs/indian-assets.md). Every box keeps
its detector class in `coco` and says where its kind came from (`class_source`: "model" or "heuristic").
"""
from __future__ import annotations

import json
import os

import numpy as np

COCO_TO_FALCON = {'person': 'pedestrian', 'bicycle': 'bicycle', 'car': 'car', 'motorcycle': 'motorcycle',
                  'bus': 'bus', 'truck': 'truck', 'cow': 'cattle'}
TWO_WHEELERS = ('motorcycle', 'bicycle')
AUTO_FROM = ('car', 'truck', 'motorcycle')
AUTO_WIDTH_M = (1.1, 1.6)
AUTO_HEIGHT_M = (1.4, 2.0)
AUTO_RATIO = (1.05, 1.6)
RIDER_OVERLAP = 0.3
LOW_SCORE = 0.1


class DetectError(Exception):
    pass


def map_detections(raw: list) -> list:
    """[{'box', 'score', 'label' (COCO name)}] → Falcon detections [{'box', 'score', 'coco', 'kind',
    'class_source'}] (unmapped classes dropped)."""
    out = []
    for d in raw:
        kind = COCO_TO_FALCON.get(str(d.get('label', '')).lower())
        if kind is None:
            continue
        out.append({'box': [float(v) for v in d['box']], 'score': float(d.get('score', 1.0)),
                    'coco': str(d['label']).lower(), 'kind': kind, 'class_source': 'model'})
    return out


def fold_riders(dets: list) -> tuple:
    """(detections without riders, riders folded): a pedestrian over a two-wheeler (module doc)."""
    wheels = [d for d in dets if d['kind'] in TWO_WHEELERS]
    keep, folded = [], 0
    for d in dets:
        if d['kind'] == 'pedestrian' and wheels:
            x1, y1, x2, y2 = d['box']
            area = max((x2 - x1) * (y2 - y1), 1e-6)
            rider = False
            for w in wheels:
                a1, b1, a2, b2 = w['box']
                inter = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
                if (inter / area >= RIDER_OVERLAP and a1 <= (x1 + x2) / 2 <= a2 and
                        b1 + 0.3 * (b2 - b1) <= y2 <= b2 + 0.1 * (b2 - b1)):
                    rider = True
                    break
            if rider:
                folded += 1
                continue
        keep.append(d)
    return keep, folded


def autorickshaw(det: dict, cam) -> dict:
    """The heuristic (module doc): the detection relabelled autorickshaw when its metric size fits; it records the
    size it measured either way (`size_m`) for a box it considered."""
    if det['kind'] not in AUTO_FROM:
        return det
    from .dashcam_depth import geo_point
    g = geo_point(cam, det['box'])
    if g is None:
        return det
    x1, y1, x2, y2 = det['box']
    rng = g['forward']
    w = (x2 - x1) * rng / cam.f
    h = (y2 - y1) * rng / cam.f
    out = dict(det, size_m=[round(w, 2), round(h, 2)])
    ratio = h / max(w, 1e-6)
    if (AUTO_WIDTH_M[0] <= w <= AUTO_WIDTH_M[1] and AUTO_HEIGHT_M[0] <= h <= AUTO_HEIGHT_M[1] and
            AUTO_RATIO[0] <= ratio <= AUTO_RATIO[1] and not g['cut']):
        out.update(kind='autorickshaw', class_source='heuristic',
                   heuristic=f'{det["coco"]} box {w:.2f} m wide, {h:.2f} m tall at {rng:.1f} m (width '
                             f'{AUTO_WIDTH_M[0]}–{AUTO_WIDTH_M[1]} m, height {AUTO_HEIGHT_M[0]}–{AUTO_HEIGHT_M[1]} m,'
                             f' height/width {AUTO_RATIO[0]}–{AUTO_RATIO[1]})')
    return out


def postprocess(raw: list, cam=None) -> tuple:
    """Raw detections of one frame → (Falcon detections, riders folded): mapped, riders folded, the autorickshaw
    heuristic applied (with a camera)."""
    dets, folded = fold_riders(map_detections(raw))
    if cam is not None:
        dets = [autorickshaw(d, cam) for d in dets]
    return dets, folded


# ------------------------------------------------------------------------------------------------ detectors
class FileDetector:
    """Detections read from a file (module doc): NOT a model."""
    is_model = False

    def __init__(self, path: str):
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
        self.frames = {int(k): v for k, v in (doc.get('frames') or {}).items()}
        self.name = f'file:{os.path.basename(path)}'
        self.info = {'detector': 'file', 'path': os.path.abspath(path), 'is_model': False,
                     'note': 'detections read from a file, not a model'}

    def __call__(self, frames: list) -> list:
        """frames: [(n, rgb)] → per frame [{'box', 'score', 'label'}]."""
        return [list(self.frames.get(int(n), [])) for n, _ in frames]


class RTDetrV2:
    """RT-DETRv2 through transformers (module doc), offline from its verified folder."""
    is_model = True

    def __init__(self, name: str = 'rtdetr-v2-r18', models_dir: str = '', device: str = 'auto', fp16: bool = True,
                 threshold: float = LOW_SCORE, gpu_memory_fraction: float = 0.5):
        from . import perception_models as PM
        folder = PM.load_folder(name, models_dir)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(float(gpu_memory_fraction), self.device.index or 0)
        self.dtype = torch.float16 if fp16 and self.device.type == 'cuda' else torch.float32
        self.processor = RTDetrImageProcessor.from_pretrained(folder, local_files_only=True)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(folder, local_files_only=True).to(
            self.device, self.dtype).eval()
        self.labels = {int(k): v for k, v in self.model.config.id2label.items()}
        self.threshold = float(threshold)
        self.name = name
        self.info = {'detector': name, 'repo': PM.CANDIDATES[name]['repo'],
                     'revision': PM.CANDIDATES[name]['revision'], 'licence': PM.CANDIDATES[name]['licence'],
                     'device': str(self.device), 'dtype': str(self.dtype).replace('torch.', ''),
                     'threshold': self.threshold, 'is_model': True}

    def __call__(self, frames: list) -> list:
        import torch
        imgs = [rgb for _, rgb in frames]
        enc = self.processor(images=imgs, return_tensors='pt')
        x = enc['pixel_values'].to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model(pixel_values=x)
        out.logits = out.logits.float()
        out.pred_boxes = out.pred_boxes.float()
        sizes = [im.shape[:2] for im in imgs]
        res = self.processor.post_process_object_detection(out, threshold=self.threshold, target_sizes=sizes)
        frames_out = []
        for r in res:
            frames_out.append([{'box': [float(v) for v in b], 'score': float(s), 'label': self.labels[int(c)]}
                               for s, c, b in zip(r['scores'].cpu().numpy(), r['labels'].cpu().numpy(),
                                                  r['boxes'].cpu().numpy())])
        return frames_out


def make_detector(spec: str, models_dir: str = '', device: str = 'auto', fp16: bool = True):
    """'rtdetr-v2-r18' (default), 'rtdetr-v2-r50' or 'file:PATH'. A YOLO or another refused model: DetectError."""
    from . import perception_models as PM
    spec = spec or 'rtdetr-v2-r18'
    if spec.startswith('file:'):
        return FileDetector(spec[5:])
    why = PM.refused_reason(spec)
    if why:
        raise DetectError(why)
    if spec in ('rtdetr-v2-r18', 'rtdetr-v2-r50'):
        try:
            return RTDetrV2(spec, models_dir, device, fp16)
        except PM.PerceptionModelError as e:
            raise DetectError(str(e)) from None
    raise DetectError(f'unknown detector {spec!r}: rtdetr-v2-r18, rtdetr-v2-r50 or file:PATH')


def boxes_array(dets: list) -> np.ndarray:
    return np.array([d['box'] for d in dets], float).reshape(-1, 4)
