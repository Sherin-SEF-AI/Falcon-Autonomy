"""Models: registration, runners and output decoders for the test harness (numpy at import; onnxruntime, torch and
OpenCV only inside the runners).

A model registration (the project's "models"):

    {"id", "name", "kind": "detector" | "segmenter",
     "runtime": "onnx" | "torchscript" | "python",
     "path": file (onnx, torchscript; relative paths are in the workspace's models/ folder) | "entry": "module:Class",
     "input": {"width", "height", "layout": "NCHW" | "NHWC", "scale": 1/255, "mean": [..], "std": [..],
               "rgb": true, "letterbox": false},
     "output": {"format": "boxes_scores_labels" | "yolo" | "logits", "score_threshold": 0.3, "nms_iou": 0.5,
                "boxes": "xyxy" | "cxcywh", "normalised": false},
     "classes": [model class names in index order], "class_map": {model class: Falcon class},
     "licence": "Apache-2.0", "source": {"url", "sha256"}}

Runtimes. ONNX models run through onnxruntime with CUDA when its CUDA provider is installed (onnxruntime-gpu),
else on the CPU — the providers used are reported with every evaluation. TorchScript models run through PyTorch
(CUDA when available) and are optional. "python" models are plug-ins (falcon_autonomy.plugins) with
predict(images) -> per image a list of {'bbox' [x0 y0 x1 y1] pixels, 'class', 'score'} (a detector) or an H × W class
map (a segmenter).

Downloads only on request: fetch() downloads a registered model's pinned URL, checks its SHA-256 and refuses on a
mismatch; the licence is shown before. No model is bundled, and no AGPL-licensed code (e.g. Ultralytics) is a
dependency: "yolo" here is only the output layout (rows of cx cy w h + class scores, as YOLOv5/v8-style exports
write it), decoded with Falcon's own code.
"""
from __future__ import annotations

import hashlib
import importlib
import os

import numpy as np

KINDS = ('detector', 'segmenter')
RUNTIMES = ('onnx', 'torchscript', 'python')
FORMATS = ('boxes_scores_labels', 'yolo', 'logits')
PERMISSIVE = ('Apache-2.0', 'MIT', 'BSD-2-Clause', 'BSD-3-Clause', 'CC0-1.0', 'CC-BY-4.0')


def validate(m: dict) -> list:
    out = []
    if m.get('kind') not in KINDS:
        out.append(f'kind must be one of {KINDS}')
    if m.get('runtime') not in RUNTIMES:
        out.append(f'runtime must be one of {RUNTIMES}')
    if m.get('runtime') in ('onnx', 'torchscript') and not m.get('path'):
        out.append('an ONNX or TorchScript model needs a path')
    if m.get('runtime') == 'python' and ':' not in (m.get('entry') or ''):
        out.append('a Python model needs an entry "module:Class"')
    fmt = (m.get('output') or {}).get('format')
    if m.get('runtime') != 'python' and fmt not in FORMATS:
        out.append(f'output format must be one of {FORMATS}')
    lic = m.get('licence') or ''
    if 'AGPL' in lic.upper():
        out.append('an AGPL-licensed model is not allowed in Falcon\'s core (its licence would bind the app)')
    return out


def resolve_path(m: dict, workspace: str) -> str:
    p = m.get('path') or ''
    return p if os.path.isabs(p) else os.path.join(workspace, 'models', p)


def fetch(m: dict, workspace: str, progress=None) -> str:
    """Download a registered model's pinned URL into the workspace and check its SHA-256 (only when asked)."""
    import urllib.request
    src = m.get('source') or {}
    if not src.get('url') or not src.get('sha256'):
        raise ValueError('the model has no pinned source (url and sha256)')
    path = resolve_path(m, workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    h = hashlib.sha256()
    with urllib.request.urlopen(src['url'], timeout=60) as r, open(tmp, 'wb') as fh:
        total = int(r.headers.get('Content-Length') or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
            h.update(b)
            got += len(b)
            if progress and total:
                progress(got / total)
    if h.hexdigest() != src['sha256'].lower():
        os.remove(tmp)
        raise ValueError(f'SHA-256 mismatch: expected {src["sha256"]}, got {h.hexdigest()} — not installed')
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------------------------ preprocessing
def preprocess(img: np.ndarray, spec: dict) -> tuple[np.ndarray, dict]:
    """An RGB uint8 image (H, W, 3) to the model's input tensor (1, …) and the transform back to image pixels."""
    import cv2
    H, W = img.shape[:2]
    w, h = int(spec.get('width', W)), int(spec.get('height', H))
    x = img if spec.get('rgb', True) else img[:, :, ::-1]
    if spec.get('letterbox'):
        k = min(w / W, h / H)
        nw, nh = int(round(W * k)), int(round(H * k))
        canvas = np.full((h, w, 3), 114, np.uint8)
        px, py = (w - nw) // 2, (h - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x, back = canvas, {'kx': k, 'ky': k, 'px': px, 'py': py}
    else:
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) if (w, h) != (W, H) else x
        back = {'kx': w / W, 'ky': h / H, 'px': 0, 'py': 0}
    t = x.astype(np.float32) * float(spec.get('scale', 1.0 / 255.0))
    if spec.get('mean') is not None:
        t = (t - np.asarray(spec['mean'], np.float32)) / np.asarray(spec.get('std', [1, 1, 1]), np.float32)
    if spec.get('layout', 'NCHW') == 'NCHW':
        t = t.transpose(2, 0, 1)
    back.update(model_w=w, model_h=h)
    return t[None].astype(np.float32), back


def to_image(boxes: np.ndarray, back: dict) -> np.ndarray:
    b = np.asarray(boxes, np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - back['px']) / back['kx']
    b[:, [1, 3]] = (b[:, [1, 3]] - back['py']) / back['ky']
    return b


# ------------------------------------------------------------------------------------------------ decoders
def nms(boxes, scores, iou=0.5) -> np.ndarray:
    """Greedy non-maximum suppression (xyxy); indices kept, by score."""
    b = np.asarray(boxes, float)
    order = np.argsort(-np.asarray(scores, float))
    keep = []
    area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    while len(order):
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(b[i, 0], b[order[1:], 0])
        yy0 = np.maximum(b[i, 1], b[order[1:], 1])
        xx1 = np.minimum(b[i, 2], b[order[1:], 2])
        yy1 = np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.maximum(xx1 - xx0, 0) * np.maximum(yy1 - yy0, 0)
        o = inter / np.maximum(area[i] + area[order[1:]] - inter, 1e-12)
        order = order[1:][o <= iou]
    return np.asarray(keep, int)


def _xyxy(b, fmt, w, h, normalised):
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if fmt == 'cxcywh':
        b = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
    if normalised:
        b = b * np.array([w, h, w, h])
    return b


def decode_detections(outputs: list, m: dict, back: dict) -> list:
    """A detector's raw outputs to [{'bbox' [x0 y0 x1 y1] image pixels, 'class', 'score'}] in Falcon's classes."""
    o = m.get('output') or {}
    fmt = o.get('format')
    thr = float(o.get('score_threshold', 0.3))
    classes = list(m.get('classes') or [])
    W, H = back['model_w'], back['model_h']
    if fmt == 'boxes_scores_labels':
        boxes = _xyxy(outputs[0], o.get('boxes', 'xyxy'), W, H, o.get('normalised', False))
        scores = np.asarray(outputs[1], float).reshape(-1)
        labels = np.asarray(outputs[2]).reshape(-1).astype(int)
    elif fmt == 'yolo':
        p = np.asarray(outputs[0], float)
        p = p[0] if p.ndim == 3 else p
        if p.shape[0] == 4 + len(classes) and p.shape[1] != 4 + len(classes):
            p = p.T                                     # (4 + nc, N) as YOLOv8 exports write it
        cls_scores = p[:, 4:]
        labels = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(p)), labels]
        boxes = _xyxy(p[:, :4], 'cxcywh', W, H, o.get('normalised', False))
    else:
        raise ValueError(f'not a detector output format: {fmt!r}')
    keep = scores >= thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    out = []
    for c in np.unique(labels):                         # class-wise NMS
        idx = np.flatnonzero(labels == c)
        for k in idx[nms(boxes[idx], scores[idx], float(o.get('nms_iou', 0.5)))]:
            out.append((k, int(c)))
    img_boxes = to_image(boxes, back)
    cmap = m.get('class_map') or {}
    dets = []
    for k, c in out:
        name = classes[c] if c < len(classes) else str(c)
        falcon = cmap.get(name, name)
        if falcon is None:
            continue                                     # a model class Falcon does not evaluate
        dets.append({'bbox': [float(v) for v in img_boxes[k]], 'class': falcon, 'score': float(scores[k])})
    return dets


def decode_segmentation(outputs: list, m: dict, back: dict, size: tuple) -> np.ndarray:
    """Logits (1, C, h, w) or (C, h, w) to an H × W map of Falcon class ids (class_map: model class → Falcon tag)."""
    import cv2
    logits = np.asarray(outputs[0])
    logits = logits[0] if logits.ndim == 4 else logits
    lab = np.argmax(logits, axis=0).astype(np.int32)
    H, W = size
    if back['px'] or back['py']:
        lab = lab[back['py']:back['model_h'] - back['py'], back['px']:back['model_w'] - back['px']]
    lab = cv2.resize(lab.astype(np.uint16), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    classes = list(m.get('classes') or [])
    cmap = m.get('class_map') or {}
    lut = np.full(max(len(classes), int(lab.max()) + 1), 255, np.int32)
    for i, name in enumerate(classes):
        v = cmap.get(name)
        if v is not None:
            lut[i] = int(v)
    return lut[lab]


# ------------------------------------------------------------------------------------------------ runners
class Runner:
    """Loads a registered model once; predict(rgb image) -> detections or a class map. `providers` says where it
    ran (onnxruntime's execution providers, or torch's device)."""

    def __init__(self, m: dict, workspace: str = ''):
        problems = validate(m)
        if problems:
            raise ValueError('; '.join(problems))
        self.m = m
        rt = m['runtime']
        if rt == 'onnx':
            import onnxruntime as ort
            avail = ort.get_available_providers()
            prov = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in avail]
            self.sess = ort.InferenceSession(resolve_path(m, workspace), providers=prov)
            self.providers = self.sess.get_providers()
            self.input_name = self.sess.get_inputs()[0].name
        elif rt == 'torchscript':
            import torch
            self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.net = torch.jit.load(resolve_path(m, workspace), map_location=self.dev).eval()
            self.providers = [f'torch:{self.dev}']
        else:
            mod, _, cls = m['entry'].partition(':')
            self.plugin = getattr(importlib.import_module(mod), cls)(**(m.get('params') or {}))
            self.providers = ['python']

    def raw(self, x):
        if self.m['runtime'] == 'onnx':
            return self.sess.run(None, {self.input_name: x})
        import torch
        with torch.no_grad():
            y = self.net(torch.from_numpy(x).to(self.dev))
        y = y if isinstance(y, (list, tuple)) else [y]
        return [t.detach().cpu().numpy() for t in y]

    def predict(self, rgb: np.ndarray, meta: dict | None = None):
        if self.m['runtime'] == 'python':
            return self.plugin.predict(rgb, meta or {})
        x, back = preprocess(rgb, self.m.get('input') or {})
        outs = self.raw(x)
        if self.m['kind'] == 'segmenter':
            return decode_segmentation(outs, self.m, back, rgb.shape[:2])
        return decode_detections(outs, self.m, back)
