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

Precision (the low-VRAM profile, core.profiles): Runner(precision='fp16') runs an ONNX model converted to FP16 by
onnxruntime.transformers.float16.convert_float_to_float16 with keep_io_types (inputs and outputs stay FP32), cached
as <sha256 of the FP32 file>.fp16.onnx in Falcon's cache folder; a TorchScript model .half() (under autocast on CUDA).
A model that does not load in FP16 runs in FP32 and says why (Runner.precision, .precision_note). fp16_gate() decides
'auto': the model runs on the evaluated frames in both precisions and FP16 is kept only when its AP (a segmenter's
mIoU) is at most FP16_MAX_DROP_PT points below FP32's; every evaluation records the precision it ran at and why.

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
PRECISIONS = ('fp32', 'fp16')
FP16_MAX_DROP_PT = 0.5          # 'auto' keeps FP16 when AP (mIoU) drops by at most this many points


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
    """An RGB uint8 image (H, W, 3) to the model's input tensor (1, …) and the transform back to image pixels:
    geometry() then normalise() (split so that core.deploy can do the first on the host and the second on a device,
    which runs a verbatim copy of this module)."""
    x, back = geometry(img, spec)
    return normalise(x, spec), back


def geometry(img: np.ndarray, spec: dict) -> tuple[np.ndarray, dict]:
    """The geometric half of preprocess(): the model's channel order, resized or letterboxed to its input size, still
    uint8 (h, w, 3); and the transform back to image pixels."""
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
    back.update(model_w=w, model_h=h)
    return x, back


def normalise(x: np.ndarray, spec: dict) -> np.ndarray:
    """The arithmetic half of preprocess() (numpy only): scale, mean and std, layout; a uint8 (h, w, 3) image in the
    model's channel order and size to its float32 input tensor (1, …)."""
    t = x.astype(np.float32) * float(spec.get('scale', 1.0 / 255.0))
    if spec.get('mean') is not None:
        t = (t - np.asarray(spec['mean'], np.float32)) / np.asarray(spec.get('std', [1, 1, 1]), np.float32)
    if spec.get('layout', 'NCHW') == 'NCHW':
        t = t.transpose(2, 0, 1)
    return t[None].astype(np.float32)


def to_image(boxes: np.ndarray, back: dict) -> np.ndarray:
    b = np.asarray(boxes, np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - back['px']) / back['kx']
    b[:, [1, 3]] = (b[:, [1, 3]] - back['py']) / back['ky']
    return b


# ------------------------------------------------------------------------------------------------ decoders
def nms(boxes, scores, iou=0.5) -> np.ndarray:
    """Greedy non-maximum suppression (xyxy); indices kept, by score — equal scores in index order (a stable sort:
    numpy's default quicksort orders ties differently between versions, and a device runs another numpy)."""
    b = np.asarray(boxes, float)
    order = np.argsort(-np.asarray(scores, float), kind='stable')
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
    """Logits (1, C, h, w) or (C, h, w) to an H × W map of Falcon class ids. class_map: model class → Falcon class,
    by name (as detectors' maps: 'car') or by id (14); a model class without one is 255 (not scored)."""
    import cv2
    from .groundtruth import CLASS_ID
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
        v = cmap.get(name, name if name in CLASS_ID else None)
        if isinstance(v, str):
            v = CLASS_ID.get(v)
        if v is not None:
            lut[i] = int(v)
    return lut[lab]


# ------------------------------------------------------------------------------------------------ FP16
def fp16_path(src: str) -> str:
    """Where the FP16 conversion of an ONNX file is cached: <cache>/fp16/<sha256 of the file>.fp16.onnx."""
    from .paths import cache_dir
    from .provenance import sha256_file
    return os.path.join(cache_dir(), 'fp16', f'{sha256_file(src)}.fp16.onnx')


def fp16_onnx(src: str) -> str:
    """The FP16 conversion of an ONNX model (onnxruntime.transformers.float16.convert_float_to_float16,
    keep_io_types: FP32 inputs and outputs), made once and cached (fp16_path)."""
    out = fp16_path(src)
    if os.path.isfile(out):
        return out
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m16 = convert_float_to_float16(onnx.load(src), keep_io_types=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + f'.part{os.getpid()}'
    onnx.save(m16, tmp)
    os.replace(tmp, out)
    return out


# ------------------------------------------------------------------------------------------------ runners
class Runner:
    """Loads a registered model once; predict(rgb image) -> detections or a class map. `providers` says where it
    ran (onnxruntime's execution providers, or torch's device); `precision` what it runs at ('fp32' or 'fp16'; a
    plug-in: 'native') and `precision_note` why it is not what was asked."""

    def __init__(self, m: dict, workspace: str = '', precision: str = 'fp32'):
        problems = validate(m)
        if problems:
            raise ValueError('; '.join(problems))
        if precision not in PRECISIONS:
            raise ValueError(f'precision must be one of {PRECISIONS} (\'auto\' is decided by fp16_gate)')
        self.m = m
        self.requested = precision
        self.precision, self.precision_note = 'fp32', ''
        rt = m['runtime']
        if rt == 'onnx':
            import onnxruntime as ort
            avail = ort.get_available_providers()
            prov = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in avail]
            path = resolve_path(m, workspace)
            self.sess = None
            if precision == 'fp16':
                try:
                    self.sess = ort.InferenceSession(fp16_onnx(path), providers=prov)
                    self.precision, self.weights = 'fp16', fp16_path(path)
                except Exception as e:                     # noqa: BLE001 (no converter, an op without FP16)
                    self.precision_note = f'FP16 did not load ({type(e).__name__}: {str(e)[:160]}): FP32'
            if self.sess is None:
                self.sess = ort.InferenceSession(path, providers=prov)
                self.weights = path
            self.providers = self.sess.get_providers()
            self.input_name = self.sess.get_inputs()[0].name
        elif rt == 'torchscript':
            import torch
            self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.net = torch.jit.load(resolve_path(m, workspace), map_location=self.dev).eval()
            if precision == 'fp16':
                self.net = self.net.half()
                self.precision = 'fp16'
            self.providers = [f'torch:{self.dev}']
        else:
            mod, _, cls = m['entry'].partition(':')
            self.plugin = getattr(importlib.import_module(mod), cls)(**(m.get('params') or {}))
            self.providers = ['python']
            self.precision = 'native'
            if precision == 'fp16':
                self.precision_note = 'a plug-in runs at its own precision'

    def provenance(self) -> dict:
        return {'precision': self.precision, 'requested': self.requested, 'note': self.precision_note or None}

    def raw(self, x):
        if self.m['runtime'] == 'onnx':
            return self.sess.run(None, {self.input_name: x})
        import contextlib

        import torch
        t = torch.from_numpy(x).to(self.dev)
        half = self.precision == 'fp16'
        if half:
            t = t.half()
        amp = torch.autocast('cuda', dtype=torch.float16) if half and self.dev == 'cuda' else \
            contextlib.nullcontext()
        with torch.no_grad(), amp:
            y = self.net(t)
        y = y if isinstance(y, (list, tuple)) else [y]
        return [(t.float() if t.is_floating_point() else t).detach().cpu().numpy() for t in y]

    def predict(self, rgb: np.ndarray, meta: dict | None = None):
        if self.m['runtime'] == 'python':
            return self.plugin.predict(rgb, meta or {})
        x, back = preprocess(rgb, self.m.get('input') or {})
        outs = self.raw(x)
        if self.m['kind'] == 'segmenter':
            return decode_segmentation(outs, self.m, back, rgb.shape[:2])
        return decode_detections(outs, self.m, back)


def fp16_gate(runs: list, m: dict, workspace: str = '', camera: str | None = None, every: int = 1,
              max_drop_pt: float = FP16_MAX_DROP_PT, progress=None, cancel=None) -> dict:
    """'auto' precision: the model on the same frames in FP32 and FP16 (core.harness.collect), scored against the
    ground truth (AP; a segmenter's mIoU); FP16 kept when it is at most max_drop_pt points below FP32. {'precision',
    'requested': 'auto', 'metric', 'fp32', 'fp16', 'drop_pt', 'max_drop_pt', 'frames', 'reason', 'collected':
    {precision: harness.collect's result}} — the kept collection is the evaluation's, nothing runs a third time."""
    from . import harness as H
    out = {'requested': 'auto', 'max_drop_pt': float(max_drop_pt), 'metric': None, 'fp32': None, 'fp16': None,
           'drop_pt': None}

    def sub(lo, hi):
        return (lambda f, t: progress(lo + (hi - lo) * f, t)) if progress else None
    r32 = Runner(m, workspace, 'fp32')
    c32 = H.collect(runs, m, workspace, camera, every, sub(0.0, 0.5), None, cancel, runner=r32)
    r16 = Runner(m, workspace, 'fp16')
    if r16.precision != 'fp16':
        return dict(out, precision='fp32', frames=len(c32['frames']), collected={'fp32': c32},
                    reason=r16.precision_note or 'FP16 is not available for this runtime')
    c16 = H.collect(runs, m, workspace, camera, every, sub(0.5, 1.0), None, cancel, runner=r16)
    seg = m['kind'] == 'segmenter'
    metric = 'mIoU' if seg else 'AP'

    def value(col):
        sc = H.score(col['frames'], m['kind'], n_boot=10)
        v = sc['full'].get(metric)
        return None if v is None else float(v)
    v32, v16 = value(c32), value(c16)
    out.update(metric=metric, fp32=v32, fp16=v16, frames=len(c32['frames']),
               collected={'fp32': c32, 'fp16': c16})
    if v32 is None or v16 is None:
        return dict(out, precision='fp32', reason=f'no {metric} to compare (no ground truth the model can name): '
                                                  'FP32 kept')
    drop = round((v32 - v16) * 100.0, 3)
    out['drop_pt'] = drop
    if drop > max_drop_pt:
        return dict(out, precision='fp32', reason=f'FP16 {metric} {v16 * 100:.2f} is {drop:.2f} points below FP32\'s '
                                                  f'{v32 * 100:.2f} (more than {max_drop_pt:g}): FP32')
    return dict(out, precision='fp16', reason=f'FP16 {metric} {v16 * 100:.2f} within {max_drop_pt:g} points of FP32\'s '
                                              f'{v32 * 100:.2f}: FP16')


def main_model(argv) -> int:
    """`falcon model fetch PROJECT ID`: download a registered model's pinned weights into the project's workspace
    (its URL, licence and SHA-256 printed first); protocol lines on stdout; 0 ok, 2 failed."""
    import argparse
    from . import protocol as P
    from .paths import workspace_for
    from .project import Project, ProjectError
    ap = argparse.ArgumentParser(prog='falcon model', description='registered models')
    ap.add_argument('action', choices=('fetch',))
    ap.add_argument('project')
    ap.add_argument('model')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('model')
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    m = proj.item('models', a.model)
    if m is None:
        print(f'failed: no model {a.model!r} in the project', flush=True)
        return P.FAILED
    problems = validate(m)
    if problems:
        print(f'failed: {"; ".join(problems)}', flush=True)
        return P.FAILED
    src = m.get('source') or {}
    em.log(f'downloading {src.get("url")} (licence {m.get("licence") or "not stated"}; SHA-256 {src.get("sha256")})')
    try:
        path = fetch(m, workspace_for(a.project), lambda f: em.progress(f, 'download'))
    except (OSError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    em.result('ok', path=path)
    return P.OK
