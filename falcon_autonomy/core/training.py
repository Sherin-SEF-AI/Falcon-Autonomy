"""The training lab: fine-tune a perception model on Falcon drives, dataset versions and real labelled sets, with
provenance, and export it as a model the harness runs (standard library + numpy at import; torch, torchvision, OpenCV
and ONNX only inside the functions, so importing this module from the GUI stays cheap).

Models. 2D box detectors built from torchvision (BSD-3-Clause) parts, mirroring torchvision 0.24's own builders:

    fasterrcnn_mobilenet_v3_large_fpn   Faster R-CNN, MobileNetV3-Large FPN (torchvision's model)
    fasterrcnn_mobilenet_v3_small_fpn   the same design on MobileNetV3-Small, composed here from torchvision parts
                                        (no COCO weights exist for it; small and fast, used by the tests)
    fcos_resnet50_fpn                   FCOS, ResNet-50 FPN (torchvision's model)
    retinanet_resnet50_fpn_v2           RetinaNet v2, ResNet-50 FPN, GIoU loss (torchvision's model)

No Ultralytics or other AGPL code. Initialisation is one of: "scratch" (random weights, trainable BatchNorm),
"imagenet" (the backbone's ImageNet-1k classification weights; FrozenBatchNorm, 3 trainable backbone stages, as
torchvision does) or "coco" (the whole detector's COCO weights; its class head is cut down to Falcon's classes, each
row initialised from the matching COCO class — person → pedestrian, car, truck, bus, motorcycle, bicycle, train,
traffic light, stop sign → traffic_sign — and a class COCO lacks, such as rider, initialised fresh). Pretrained weights
are never downloaded by themselves: WEIGHTS pins each file's URL (download.pytorch.org) and full SHA-256, which were
computed on the downloaded files (their first 8 hex digits are torchvision's filename suffix); fetch_weights()
downloads one only when asked (licence shown first) and every file is checked against its SHA-256 before it is
loaded, wherever it was found (the workspace's models/pretrained/, a folder given, or torch's hub cache).

Data. A source is "KIND:PATH?key=value&…" (parse_source) or the same as a dict:
- falcon: a recorded drive (falcon-raw); camera= (default the first pinhole camera), every= (every n-th frame),
  select= (a JSON file whose "ids" lists the frames to keep: an acquisition's choice, core.activelearn).
- version: a dataset version file (core.datasets; datasets/<spec>/versions/vNNN.json) and split= train | val | test;
  each run's run.json must still have the SHA-256 the version recorded (else refused: the data changed).
- kitti_object, coco, bdd100k, nuscenes: read in place through core.realgap.read_set (labels=, camera=, limit=,
  seed=); class names mapped to Falcon's by realgap.NAME_ALIASES (Van → car, Cyclist → bicycle, person →
  pedestrian, …); classes not trained are left out of both training and validation, as the harness leaves out
  classes a model cannot name.
Falcon ground truth gets the harness's ignore rule (core.harness): an object with fewer than MIN_PIXELS visible
pixels or more than MAX_OCCLUDED of its box hidden is *ignored* — not a training target, and in validation neither
found nor missed (a detection on it is not a false positive). torchvision's losses have no ignore regions, so the
pixels of an ignored object count as background in training: the model is not asked to find what the harness does
not score, but it is pushed (weakly) not to. Real sets carry no visible-pixel counts: a real box smaller than
MIN_PIXELS in area (its box area bounds its visible pixels) is ignored; occlusion is not known and not applied.
Every image is resized to the model's input size exactly as core.models.preprocess does at inference (a stretch, or a
letterbox when chosen), so training, validation and the exported model see the same pixels.

Training. SGD with momentum (lr: torchvision's reference recipes, 0.02 per 16 images for Faster R-CNN and 0.01 for
FCOS and RetinaNet, scaled linearly with the batch), linear warm-up, then a cosine or step schedule set per
iteration, gradient clipping by global norm, automatic mixed precision (float16 with a gradient scaler) on CUDA, a
checkpoint after every epoch and on a cancel (resumed from exactly where it stopped, mid-epoch included), validation
every epoch with COCO AP (core.metrics: pycocotools' numbers) on the validation sources, the best epoch by AP kept,
optional early stopping (patience in epochs). Progress, log and metric lines go out through the protocol (Emitter).

Reproducibility. Seeded: the image order and flips of an epoch are functions of (seed, epoch); torch's generator is
re-seeded from (seed, epoch, iteration) before every iteration (so a resumed run continues the same random stream);
numpy and random are seeded; cuDNN is deterministic with benchmarking off and torch.use_deterministic_algorithms(True,
warn_only=True). On the CPU two runs with the same settings, data and thread count give bit-for-bit the same weights
(tested), and a run cancelled and resumed gives the weights of the run that was not (tested). On CUDA it depends on
the model: Faster R-CNN is NOT bit for bit — torchvision's RoIAlign backward accumulates with atomic adds and has no
deterministic CUDA implementation (torch warns; the warning is kept in run.json "determinism") — so two GPU runs
differ in the last bits and drift apart (measured: validation AP 0.2625 against 0.2639 after 180 iterations, largest
weight difference 0.0023). FCOS and RetinaNet use no such operation and were measured bit for bit on CUDA with AMP
(two and three runs of 30 iterations, identical weights) — on one machine, driver and library version; another GPU or
cuDNN may choose other kernels. Data-loader workers draw no random numbers, so their count does not change results.

GPU sharing. A training on CUDA is a needs_gpu job (job_spec()); run from the CLI it also takes the GPU lock itself
and waits while a CARLA job holds the CARLA lock (and holds it, so CARLA jobs queue behind the training). The batch
is capped at MAX_BATCH and the process's share of GPU memory at gpu_memory_fraction (default 0.6, a heuristic that
leaves room for the desktop); the peak memory allocated is measured and kept in run.json.

Output: training/<id>/ — run.json (provenance: inputs hashed, settings, seed, versions, device, measured timings,
peak GPU memory, determinism notes), data.json (every image used, its unit and domain), metrics.jsonl (one line per
epoch), checkpoints/last.pt (resume) and best.pt, model.onnx (opset 17, a fixed input size; loads in onnxruntime
1.28) and model.json: a ready registration in core.models' format (kind detector, runtime onnx, output
boxes_scores_labels) with the check that the ONNX model, run through core.models.Runner, gives the torch model's
detections (verification in run.json).

Limits: 2D boxes only (no segmenter in this release); one fixed input size per model; no multi-GPU; augmentation is
a horizontal flip only; COCO and ImageNet pretrained weights carry their training data's terms, and real sets their
own (DATA_TERMS, as published by their owners — check them before shipping a model).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

SCHEMA = 'falcon.training/1'
DEFAULT_CLASSES = ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian', 'rider')
MAX_BATCH = 32
OPSET = 17
SOURCE_KINDS = ('falcon', 'version', 'kitti_object', 'coco', 'bdd100k', 'nuscenes')
REAL_KINDS = ('kitti_object', 'coco', 'bdd100k', 'nuscenes')
TORCHVISION_TERMS = 'BSD-3-Clause (torchvision)'

# Pinned pretrained weights. The SHA-256 of each file was computed on the file downloaded from its URL (2026-10-28);
# its first 8 hex digits match torchvision's filename suffix. "box_map" is torchvision's published COCO val2017 number,
# not measured here.
WEIGHTS = {
    'fasterrcnn_mobilenet_v3_large_fpn-coco': {
        'url': 'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth',
        'sha256': 'fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533', 'bytes': 77844807,
        'what': 'detector', 'arch': 'fasterrcnn_mobilenet_v3_large_fpn', 'data': 'COCO 2017 train',
        'licence': TORCHVISION_TERMS + '; trained on COCO 2017 (annotations CC-BY-4.0, images under Flickr terms)',
        'published': 'box mAP 32.8 on COCO val2017 (torchvision\'s table)'},
    'fcos_resnet50_fpn-coco': {
        'url': 'https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth',
        'sha256': '99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7', 'bytes': 129612099,
        'what': 'detector', 'arch': 'fcos_resnet50_fpn', 'data': 'COCO 2017 train',
        'licence': TORCHVISION_TERMS + '; trained on COCO 2017 (annotations CC-BY-4.0, images under Flickr terms)',
        'published': 'box mAP 39.2 on COCO val2017 (torchvision\'s table)'},
    'retinanet_resnet50_fpn_v2-coco': {
        'url': 'https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth',
        'sha256': '5905b1c544219215e544dbe319720397bc4e68de61a733a59350d7976645b769', 'bytes': 153130989,
        'what': 'detector', 'arch': 'retinanet_resnet50_fpn_v2', 'data': 'COCO 2017 train',
        'licence': TORCHVISION_TERMS + '; trained on COCO 2017 (annotations CC-BY-4.0, images under Flickr terms)',
        'published': 'box mAP 41.5 on COCO val2017 (torchvision\'s table)'},
    'mobilenet_v3_large-imagenet1k-v1': {
        'url': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
        'sha256': '8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04', 'bytes': 22139423,
        'what': 'backbone', 'arch': 'mobilenet_v3_large', 'data': 'ImageNet-1k',
        'licence': TORCHVISION_TERMS + '; ImageNet-1k training data under its own terms',
        'published': 'top-1 74.0 % on ImageNet-1k val (torchvision\'s table)'},
    'mobilenet_v3_small-imagenet1k-v1': {
        'url': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
        'sha256': '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f', 'bytes': 10306551,
        'what': 'backbone', 'arch': 'mobilenet_v3_small', 'data': 'ImageNet-1k',
        'licence': TORCHVISION_TERMS + '; ImageNet-1k training data under its own terms',
        'published': 'top-1 67.7 % on ImageNet-1k val (torchvision\'s table)'},
    'resnet50-imagenet1k-v1': {
        'url': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
        'sha256': '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a', 'bytes': 102530333,
        'what': 'backbone', 'arch': 'resnet50', 'data': 'ImageNet-1k',
        'licence': TORCHVISION_TERMS + '; ImageNet-1k training data under its own terms',
        'published': 'top-1 76.1 % on ImageNet-1k val (torchvision\'s table)'},
    'resnet50-imagenet1k-v2': {
        'url': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        'sha256': '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca', 'bytes': 102540417,
        'what': 'backbone', 'arch': 'resnet50', 'data': 'ImageNet-1k',
        'licence': TORCHVISION_TERMS + '; ImageNet-1k training data under its own terms',
        'published': 'top-1 80.9 % on ImageNet-1k val (torchvision\'s table)'},
}
for _k, _w in WEIGHTS.items():
    _w['file'] = os.path.basename(_w['url'])

ARCHS = {
    'fasterrcnn_mobilenet_v3_large_fpn': {
        'family': 'fasterrcnn', 'backbone': 'mobilenet_v3_large', 'coco': 'fasterrcnn_mobilenet_v3_large_fpn-coco',
        'imagenet': ('mobilenet_v3_large-imagenet1k-v1',), 'lr16': 0.02, 'nms_iou': 0.5,
        'note': 'torchvision\'s Faster R-CNN MobileNetV3-Large FPN'},
    'fasterrcnn_mobilenet_v3_small_fpn': {
        'family': 'fasterrcnn', 'backbone': 'mobilenet_v3_small', 'coco': None,
        'imagenet': ('mobilenet_v3_small-imagenet1k-v1',), 'lr16': 0.02, 'nms_iou': 0.5,
        'note': 'the MobileNetV3-Large FPN design on MobileNetV3-Small, composed by Falcon from torchvision parts; '
                'no COCO weights exist for it'},
    'fcos_resnet50_fpn': {
        'family': 'fcos', 'backbone': 'resnet50', 'coco': 'fcos_resnet50_fpn-coco',
        'imagenet': ('resnet50-imagenet1k-v1', 'resnet50-imagenet1k-v2'), 'lr16': 0.01, 'nms_iou': 0.6,
        'note': 'torchvision\'s FCOS ResNet-50 FPN'},
    'retinanet_resnet50_fpn_v2': {
        'family': 'retinanet_v2', 'backbone': 'resnet50', 'coco': 'retinanet_resnet50_fpn_v2-coco',
        'imagenet': ('resnet50-imagenet1k-v1', 'resnet50-imagenet1k-v2'), 'lr16': 0.01, 'nms_iou': 0.5,
        'note': 'torchvision\'s RetinaNet v2 ResNet-50 FPN (GroupNorm head, GIoU loss)'},
}
INITS = ('scratch', 'imagenet', 'coco')

# Terms of the real sets as their owners publish them (heuristic summary, 2026; check the current terms): a model
# trained on them may inherit these.
DATA_TERMS = {
    'kitti_object': 'KITTI: CC BY-NC-SA 3.0 (non-commercial)',
    'nuscenes': 'nuScenes: CC BY-NC-SA 4.0 for non-commercial use; commercial use needs a licence from Motional',
    'bdd100k': 'BDD100K: its own licence (research use; see bdd-data.berkeley.edu)',
    'coco': 'COCO: annotations CC BY 4.0; images under their Flickr licences',
}

DEFAULTS = {
    'arch': 'fasterrcnn_mobilenet_v3_large_fpn', 'init': 'coco', 'backbone_weights': None,
    'classes': list(DEFAULT_CLASSES), 'size': [640, 384], 'letterbox': False,
    'epochs': 12, 'batch': 8, 'lr': None, 'momentum': 0.9, 'weight_decay': 1e-4, 'schedule': 'cosine',
    'warmup_iters': None, 'clip_norm': 10.0, 'amp': True, 'device': 'auto', 'workers': 2, 'seed': 0, 'hflip': True,
    'patience': 0, 'trainable_layers': None, 'gpu_memory_fraction': 0.6, 'score_threshold': 0.05,
    'detections_per_image': 100, 'export': True, 'opset': OPSET, 'verify_images': 8, 'val_batch': 8,
    'select': 'best', 'box_head_size': 1024, 'keep_last': True, 'train': [], 'val': [], 'test': [],
}
# settings that do not change the result (left out of the hash that guards a resume)
VOLATILE = ('workers', 'verify_images', 'val_batch', 'export', 'gpu_memory_fraction', 'weights_dirs', 'id',
            'keep_last')


class TrainingError(Exception):
    pass


class Cancelled(Exception):
    pass


class _Quiet:
    """An Emitter that says nothing (library use)."""

    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass

    def metric(self, *a, **k):
        pass


# ================================================================================================ pretrained weights
def weights_dirs(workspace: str = '', extra=()) -> list:
    """Where a pinned weights file may already be: folders given, the workspace's models/pretrained/, torch's hub
    cache (TORCH_HOME / XDG_CACHE_HOME, as torch resolves it). Every file found is checked by its SHA-256."""
    out = [d for d in extra if d]
    if workspace:
        out.append(os.path.join(workspace, 'models', 'pretrained'))
    th = os.environ.get('TORCH_HOME') or os.path.join(os.environ.get('XDG_CACHE_HOME') or
                                                      os.path.expanduser('~/.cache'), 'torch')
    out.append(os.path.join(th, 'hub', 'checkpoints'))
    return out


def find_weights(name: str, dirs: list) -> tuple:
    """(path, notes): the first file of the pinned weights `name` whose SHA-256 is right, or (None, notes)."""
    from .provenance import sha256_file
    w = WEIGHTS[name]
    notes = []
    for d in dirs:
        p = os.path.join(d, w['file'])
        if not os.path.isfile(p):
            continue
        h = sha256_file(p)
        if h == w['sha256']:
            return p, notes
        notes.append(f'{p}: SHA-256 {h} is not the pinned {w["sha256"]} — not used')
    return None, notes


def fetch_weights(name: str, workspace: str, progress=None) -> str:
    """Download pinned weights into the workspace's models/pretrained/ (only when asked); refused on a SHA-256
    mismatch (core.models.fetch)."""
    from .models import fetch
    if name not in WEIGHTS:
        raise TrainingError(f'unknown weights {name!r}; one of {", ".join(WEIGHTS)}')
    w = WEIGHTS[name]
    return fetch({'path': os.path.join('pretrained', w['file']), 'source': {'url': w['url'], 'sha256': w['sha256']}},
                 workspace, progress)


def weights_needed(settings: dict) -> str | None:
    """The pinned weights a training's init needs (None from scratch)."""
    a = ARCHS[settings['arch']]
    if settings['init'] == 'scratch':
        return None
    if settings['init'] == 'coco':
        if not a['coco']:
            raise TrainingError(f'{settings["arch"]} has no COCO weights ({a["note"]}); use imagenet or scratch')
        return a['coco']
    name = settings.get('backbone_weights') or a['imagenet'][0]
    if name not in a['imagenet']:
        raise TrainingError(f'{name} is not a backbone of {settings["arch"]}; one of {", ".join(a["imagenet"])}')
    return name


# ================================================================================================ data
def parse_source(text: str) -> dict:
    """'KIND:PATH?key=value&key=value' → {'kind', 'path', key: value}; numbers are read as numbers."""
    kind, sep, rest = str(text).partition(':')
    if not sep or kind not in SOURCE_KINDS:
        raise TrainingError(f'a source is KIND:PATH[?key=value&…] with KIND one of {", ".join(SOURCE_KINDS)}; '
                            f'got {text!r}')
    path, _, query = rest.partition('?')
    d = {'kind': kind, 'path': os.path.abspath(os.path.expanduser(path))}
    for kv in filter(None, query.split('&')):
        k, eq, v = kv.partition('=')
        if not eq:
            raise TrainingError(f'source option {kv!r} is not key=value')
        try:
            d[k] = int(v)
        except ValueError:
            d[k] = v
    return d


def _harness_ignore(o: dict) -> bool:
    from .harness import MAX_OCCLUDED, MIN_PIXELS
    return (o.get('pixels') or 0) < MIN_PIXELS or (o.get('occluded') or 0.0) > MAX_OCCLUDED


def _falcon_samples(run_dir: str, classes, camera=None, every=1, domain='synthetic', source='falcon') -> list:
    """One sample per camera frame with its own objects file: the boxes of the trained classes, the harness's ignore
    rule applied (core.harness MIN_PIXELS, MAX_OCCLUDED)."""
    from .exporters import ExportError, Run, is_pinhole
    try:
        run = Run(run_dir)
    except ExportError as e:
        raise TrainingError(str(e)) from None
    cams = [c for c in run.cams if is_pinhole(run, c)] or list(run.cams)
    cid = camera or (cams[0] if cams else None)
    if cid not in run.cams:
        raise TrainingError(f'{run.dir}: no camera {cid!r}')
    keep = set(classes)
    out = []
    for rec in run.camera_steps(cid, int(every or 1)):
        objs = run.objects(rec, cid)
        if objs is None:
            continue
        sel = [o for o in objs if o.get('class') in keep and o.get('bbox')]
        out.append({'id': f'{run.dir}#{cid}#{rec["step"]}', 'image': run.p(run.file(rec, cid, 'rgb')),
                    'boxes': [[float(v) for v in o['bbox']] for o in sel], 'classes': [o['class'] for o in sel],
                    'ignore': [_harness_ignore(o) for o in sel], 'unit': run.dir, 'domain': domain,
                    'source': source})
    if not out:
        raise TrainingError(f'{run.dir}: camera {cid} has no object labels (record with the boxes2d label)')
    return out


def _kitti_units(path: str, opt) -> dict:
    """{image basename: raw drive} from KITTI's devkit mapping (train_rand.txt and train_mapping.txt), when found:
    the object benchmark's training images are frames of raw drives, which are not independent."""
    cands = [opt] if isinstance(opt, str) and opt not in ('auto', 'image', 'parent') else []
    cands += [path, os.path.join(path, 'devkit', 'mapping'), os.path.join(os.path.dirname(path), 'devkit', 'mapping'),
              os.path.join(os.path.dirname(path), 'devkit_object', 'mapping')]
    for d in cands:
        r, m = os.path.join(d, 'train_rand.txt'), os.path.join(d, 'train_mapping.txt')
        if os.path.isfile(r) and os.path.isfile(m):
            rand = [int(x) for x in open(r, encoding='utf-8').read().replace('\n', ',').split(',') if x.strip()]
            lines = [x.split() for x in open(m, encoding='utf-8') if x.strip()]
            out = {}
            for i, k in enumerate(rand):
                if 0 < k <= len(lines) and len(lines[k - 1]) >= 2:
                    out[f'{i:06d}'] = lines[k - 1][1]
            return out
    return {}


def _unit_of(kind: str, image: str, mode, kitti_map: dict, unit_map: dict) -> str | None:
    """The resampling unit of a real image: a sequence when the set says which (KITTI's raw drive, nuScenes' log,
    a folder per sequence, a map given), else None (the image is its own unit)."""
    base = os.path.splitext(os.path.basename(image))[0]
    if mode == 'image':
        return None
    if mode == 'parent':
        return os.path.basename(os.path.dirname(image))
    if unit_map:
        return unit_map.get(os.path.basename(image)) or unit_map.get(base)
    if kind == 'kitti_object':
        return kitti_map.get(base)
    if kind == 'nuscenes' and '__' in base:
        return base.split('__')[0]            # the log (a recording session): coarser than a scene, so conservative
    return None


def _real_samples(src: dict, classes) -> tuple:
    from .harness import MIN_PIXELS
    from .realgap import GapError, read_set
    try:
        s = read_set(src['kind'], src['path'], labels=src.get('labels'), camera=src.get('camera'),
                     limit=src.get('limit'), seed=int(src.get('seed', 0)))
    except GapError as e:
        raise TrainingError(str(e)) from None
    if not s['labelled']:
        raise TrainingError(f'{src["path"]}: the {src["kind"]} set has no labels to train or score against')
    mode = src.get('units', 'auto')
    unit_map = {}
    if isinstance(mode, str) and mode.startswith('map:'):
        with open(mode[4:], encoding='utf-8') as fh:
            unit_map = json.load(fh)
    kitti_map = _kitti_units(src['path'], mode) if src['kind'] == 'kitti_object' and mode not in ('image',
                                                                                                    'parent') else {}
    keep = set(classes)
    out = []
    for it in s['items']:
        sel = [o for o in it['objects'] or () if o.get('class') in keep]
        boxes = [[float(v) for v in o['bbox']] for o in sel]
        unit = _unit_of(src['kind'], it['image'], mode, kitti_map, unit_map)
        out.append({'id': f'{src["kind"]}:{it["image"]}', 'image': it['image'], 'boxes': boxes,
                    'classes': [o['class'] for o in sel],
                    'ignore': [(b[2] - b[0]) * (b[3] - b[1]) < MIN_PIXELS for b in boxes],
                    'unit': unit if unit is not None else f'image:{it["image"]}', 'sequence': unit is not None,
                    'domain': 'real', 'source': src['kind']})
    notes = list(s.get('notes') or ())
    seq = sum(1 for x in out if x['sequence'])
    notes.append(f'{src["kind"]}: resampling unit {"sequence" if seq == len(out) else "image"} '
                 f'({seq} of {len(out)} images have a sequence)')
    return out, notes


def load_source(src, classes) -> tuple:
    """A source (dict or 'KIND:PATH?…') → (samples, notes, provenance inputs {role: path})."""
    src = parse_source(src) if isinstance(src, str) else dict(src)
    kind = src.get('kind')
    if kind == 'falcon':
        smp = _falcon_samples(src['path'], classes, src.get('camera'), src.get('every', 1))
        ins = {f'falcon:{src["path"]}/run.json': os.path.join(src['path'], 'run.json')}
        notes = []
        if src.get('select'):                     # an acquisition's choice of frames (core.activelearn)
            from .provenance import read_json
            keep = set(read_json(src['select']).get('ids') or ())
            n0 = len(smp)
            smp = [x for x in smp if x['id'] in keep]
            ins[f'select:{src["path"]}'] = src['select']
            notes.append(f'{src["path"]}: {len(smp)} of {n0} frames selected by {src["select"]}')
        return smp, notes, ins
    if kind == 'version':
        return _version_samples(src, classes)
    if kind in REAL_KINDS:
        smp, notes = _real_samples(src, classes)
        ins = {f'{kind}:labels': src['labels']} if src.get('labels') else {}
        return smp, notes, ins
    raise TrainingError(f'unknown source kind {kind!r}; one of {", ".join(SOURCE_KINDS)}')


def _version_samples(src: dict, classes) -> tuple:
    from .provenance import read_json, sha256_file
    doc = read_json(src['path'])
    split = src.get('split', 'train')
    if split not in (doc.get('splits') or {}):
        raise TrainingError(f'{src["path"]}: no split {split!r} (has {", ".join(doc.get("splits") or {})})')
    ws = src.get('workspace') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(src['path'])))))
    pinned = {e['run']: e.get('run_json_sha256') for e in doc.get('runs') or ()}
    out, ins = [], {f'version:{src["path"]}': src['path']}
    for rel in doc['splits'][split]:
        d = rel if os.path.isabs(rel) else os.path.join(ws, rel)
        want = pinned.get(rel)
        if want and sha256_file(os.path.join(d, 'run.json')) != want:
            raise TrainingError(f'{d}: its run.json changed since dataset version {doc.get("version")} was made')
        out += _falcon_samples(d, classes, src.get('camera'), src.get('every', 1), source=f'version:{split}')
        ins[f'falcon:{d}/run.json'] = os.path.join(d, 'run.json')
    if not out:
        raise TrainingError(f'{src["path"]}: split {split!r} is empty')
    note = f'dataset version {doc.get("name")} v{doc.get("version")}, split {split} (by {doc.get("split_by")})'
    return out, [note], ins


def load_sources(srcs: list, classes) -> tuple:
    samples, notes, ins = [], [], {}
    for s in srcs or ():
        a, b, c = load_source(s, classes)
        samples += a
        notes += b
        ins.update(c)
    return samples, notes, ins


def manifest(samples: list) -> dict:
    """What a set of samples is, for data.json and provenance: per image its id, unit, domain and box counts; and a
    SHA-256 over the samples' labels and image sizes in bytes (Falcon images are pinned by their run.json's output
    hashes; real images are identified by path and size, not hashed — that would read the whole set)."""
    from .provenance import sha256_json
    rows, key = [], []
    for s in samples:
        try:
            size = os.path.getsize(s['image'])
        except OSError:
            size = None
        rows.append({'id': s['id'], 'unit': s['unit'], 'domain': s['domain'], 'source': s['source'],
                     'boxes': len(s['boxes']), 'ignored': int(sum(s['ignore']))})
        key.append([s['id'], size, s['boxes'], s['classes'], s['ignore']])
    return {'n_images': len(samples), 'n_boxes': sum(r['boxes'] for r in rows),
            'n_ignored': sum(r['ignored'] for r in rows),
            'domains': {d: sum(1 for r in rows if r['domain'] == d) for d in sorted({r['domain'] for r in rows})},
            'sha256': sha256_json(key), 'images': rows}


# ================================================================================================ settings
def normalise(settings: dict) -> dict:
    s = {**DEFAULTS, **{k: v for k, v in (settings or {}).items() if v is not None or k in ('lr', 'warmup_iters')}}
    if s['arch'] not in ARCHS:
        raise TrainingError(f'unknown arch {s["arch"]!r}; one of {", ".join(ARCHS)}')
    if s['init'] not in INITS:
        raise TrainingError(f'init is one of {", ".join(INITS)}')
    s['classes'] = list(s['classes'])
    if not s['classes'] or len(set(s['classes'])) != len(s['classes']):
        raise TrainingError('classes must be a non-empty list without repeats')
    from .groundtruth import CLASS_ID
    bad = [c for c in s['classes'] if c not in CLASS_ID]
    if bad:
        raise TrainingError(f'not Falcon classes: {", ".join(bad)}')
    s['size'] = [int(v) for v in s['size']]
    if len(s['size']) != 2 or min(s['size']) < 32:
        raise TrainingError('size is [width, height], each at least 32 px')
    if not 1 <= int(s['batch']) <= MAX_BATCH:
        raise TrainingError(f'batch must be 1–{MAX_BATCH} (GPU politeness: the cap is MAX_BATCH)')
    if int(s['epochs']) < 1:
        raise TrainingError('epochs must be at least 1')
    if s['select'] not in ('best', 'last'):
        raise TrainingError('select is best (the epoch of the best validation AP) or last')
    if s['schedule'] not in ('cosine', 'step'):
        raise TrainingError('schedule is cosine or step')
    if not 0.05 <= float(s['gpu_memory_fraction']) <= 1.0:
        raise TrainingError('gpu_memory_fraction must lie in [0.05, 1]')
    if s['lr'] is None:
        s['lr'] = ARCHS[s['arch']]['lr16'] * int(s['batch']) / 16.0
    weights_needed(s)
    return s


def settings_key(s: dict, data_sha: str) -> str:
    from .provenance import sha256_json
    return sha256_json({'settings': {k: v for k, v in s.items() if k not in VOLATILE}, 'data': data_sha})


def lr_at(it: int, total: int, base: float, warmup: int, schedule: str) -> float:
    """The learning rate of iteration it (0-based) of total: linear warm-up from base/warmup, then cosine to 0 or
    steps of ×0.1 at 2/3 and 8/9 of the iterations."""
    if warmup and it < warmup:
        return base * (it + 1) / warmup
    if schedule == 'cosine':
        t = (it - warmup) / max(total - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
    return base * (0.1 ** sum(it >= m for m in (int(total * 2 / 3), int(total * 8 / 9))))


def _mix(*xs) -> int:
    """A 63-bit seed from integers (splitmix64-style mixing), for per-iteration seeding."""
    h = 0x9E3779B97F4A7C15
    for x in xs:
        h ^= (int(x) + 0x9E3779B97F4A7C15 + (h << 6) + (h >> 2)) & 0xFFFFFFFFFFFFFFFF
        h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
    return h & 0x7FFFFFFFFFFFFFFF


# ================================================================================================ the model
def _coco_rows(classes) -> dict:
    """{Falcon class: COCO index in torchvision's 91-entry category list} for the classes COCO has."""
    from torchvision.models._meta import _COCO_CATEGORIES
    from .realgap import alias
    out = {}
    for i, name in enumerate(_COCO_CATEGORIES):
        c = alias(name)
        if c in classes and c not in out:
            out[c] = i
    return out


def _load_state(path: str):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def build_model(s: dict, weights_path: str | None = None):
    """The detector for settings s: (model, info). weights_path: the verified file of weights_needed(s)."""
    import torch
    from torch import nn
    from torchvision.ops import misc as misc_ops
    a = ARCHS[s['arch']]
    fam, classes = a['family'], s['classes']
    K = len(classes)
    trained = s['init'] != 'scratch'
    norm = misc_ops.FrozenBatchNorm2d if trained else nn.BatchNorm2d
    n_out = 91 if s['init'] == 'coco' else (K + 1 if fam == 'fasterrcnn' else K)
    W, H = s['size']
    common = {'min_size': min(W, H), 'max_size': max(W, H)}       # the input is already W × H: no resize inside
    info = {'arch': s['arch'], 'init': s['init'], 'family': fam, 'weights': None, 'head': None}
    torch.manual_seed(_mix(s['seed'], 0xB01D))
    if fam == 'fasterrcnn':
        from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.models.detection.backbone_utils import _mobilenet_extractor
        body = (mobilenet_v3_large if a['backbone'] == 'mobilenet_v3_large' else mobilenet_v3_small)(
            weights=None, norm_layer=norm)
        if s['init'] == 'imagenet':
            body.load_state_dict(_load_state(weights_path))
        n_stages = 2 + sum(1 for b in body.features if getattr(b, '_is_cn', False))
        tl = s['trainable_layers'] if s['trainable_layers'] is not None else (3 if trained else n_stages)
        backbone = _mobilenet_extractor(body, True, int(tl))
        anchors = AnchorGenerator(((32, 64, 128, 256, 512),) * 3, ((0.5, 1.0, 2.0),) * 3)
        heads = {'num_classes': n_out}
        if int(s['box_head_size']) != 1024:          # torchvision's TwoMLPHead is 1024 wide (COCO weights need it)
            if s['init'] == 'coco':
                raise TrainingError('box_head_size must stay 1024 with COCO weights')
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
            r = int(s['box_head_size'])
            heads = {'num_classes': None, 'box_head': TwoMLPHead(backbone.out_channels * 7 * 7, r),
                     'box_predictor': FastRCNNPredictor(r, n_out)}
        model = FasterRCNN(backbone, rpn_anchor_generator=anchors, box_score_thresh=s['score_threshold'],
                           box_detections_per_img=s['detections_per_image'], **heads, **common)
    else:
        from torchvision.models import resnet50
        from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
        from torchvision.ops.feature_pyramid_network import LastLevelP6P7
        body = resnet50(weights=None, norm_layer=norm if fam == 'fcos' else nn.BatchNorm2d)
        if s['init'] == 'imagenet':
            body.load_state_dict(_load_state(weights_path))
        tl = s['trainable_layers'] if s['trainable_layers'] is not None else (3 if trained else 5)
        if fam == 'fcos':
            from torchvision.models.detection import FCOS
            backbone = _resnet_fpn_extractor(body, int(tl), returned_layers=[2, 3, 4],
                                             extra_blocks=LastLevelP6P7(256, 256))
            model = FCOS(backbone, n_out, score_thresh=s['score_threshold'],
                         detections_per_img=s['detections_per_image'], **common)
        else:
            from functools import partial
            from torchvision.models.detection import RetinaNet
            from torchvision.models.detection.retinanet import RetinaNetHead, _default_anchorgen
            backbone = _resnet_fpn_extractor(body, int(tl), returned_layers=[2, 3, 4],
                                             extra_blocks=LastLevelP6P7(2048, 256))
            ag = _default_anchorgen()
            head = RetinaNetHead(backbone.out_channels, ag.num_anchors_per_location()[0], n_out,
                                 norm_layer=partial(nn.GroupNorm, 32))
            head.regression_head._loss_type = 'giou'
            model = RetinaNet(backbone, n_out, anchor_generator=ag, head=head, score_thresh=s['score_threshold'],
                              detections_per_img=s['detections_per_image'], **common)
    if s['init'] == 'coco':
        model.load_state_dict(_load_state(weights_path))      # strict: the file must be this architecture's
        info['head'] = _cut_head(model, fam, classes, s['seed'])
    info['parameters'] = int(sum(p.numel() for p in model.parameters()))
    info['trainable_parameters'] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    info['trainable_backbone_stages'] = int(tl)
    info['norm'] = 'FrozenBatchNorm2d' if trained and fam != 'retinanet_v2' else 'BatchNorm2d'
    return model, info


def _cut_head(model, fam: str, classes: list, seed: int) -> dict:
    """Cut a COCO detector's class head down to Falcon's classes: each kept row is the matching COCO class's;
    a class COCO lacks gets a fresh row (torchvision's initialisation, seeded)."""
    import torch
    from torch import nn
    rows = _coco_rows(classes)
    g = torch.Generator().manual_seed(_mix(seed, 0xC0C0))
    K = len(classes)
    with torch.no_grad():
        if fam == 'fasterrcnn':
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            old = model.roi_heads.box_predictor
            new = FastRCNNPredictor(old.cls_score.in_features, K + 1)
            nn.init.normal_(new.cls_score.weight, std=0.01, generator=g)
            nn.init.zeros_(new.cls_score.bias)
            nn.init.normal_(new.bbox_pred.weight, std=0.001, generator=g)
            nn.init.zeros_(new.bbox_pred.bias)
            src = [0] + [rows.get(c) for c in classes]           # row 0: background, from COCO's background
            ow = old.bbox_pred.weight.view(91, 4, -1)
            ob = old.bbox_pred.bias.view(91, 4)
            nw = new.bbox_pred.weight.view(K + 1, 4, -1)
            nb = new.bbox_pred.bias.view(K + 1, 4)
            for j, r in enumerate(src):
                if r is None:
                    continue
                new.cls_score.weight[j] = old.cls_score.weight[r]
                new.cls_score.bias[j] = old.cls_score.bias[r]
                nw[j] = ow[r]
                nb[j] = ob[r]
            model.roi_heads.box_predictor = new
        else:
            ch = model.head.classification_head
            old = ch.cls_logits
            A = old.out_channels // 91
            new = nn.Conv2d(old.in_channels, A * K, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(new.weight, std=0.01, generator=g)
            nn.init.constant_(new.bias, -math.log((1 - 0.01) / 0.01))
            ow = old.weight.view(A, 91, *old.weight.shape[1:])
            ob = old.bias.view(A, 91)
            nw = new.weight.view(A, K, *new.weight.shape[1:])
            nb = new.bias.view(A, K)
            for j, c in enumerate(classes):
                r = rows.get(c)
                if r is not None:
                    nw[:, j] = ow[:, r]
                    nb[:, j] = ob[:, r]
            ch.cls_logits = new
            ch.num_classes = K
    from torchvision.models._meta import _COCO_CATEGORIES
    return {'from_coco': {c: _COCO_CATEGORIES[r] for c, r in rows.items()},
            'fresh': [c for c in classes if c not in rows]}


def label_names(s: dict) -> list:
    """The model's class names in output-index order (Faster R-CNN reserves index 0 for the background)."""
    return (['__background__'] if ARCHS[s['arch']]['family'] == 'fasterrcnn' else []) + list(s['classes'])


# ================================================================================================ data loading
def input_spec(s: dict) -> dict:
    """core.models' input block of the trained model: the image as RGB in [0, 1], stretched (or letterboxed) to the
    model's size; the model normalises inside (torchvision's GeneralizedRCNNTransform, ImageNet mean and std)."""
    W, H = s['size']
    return {'width': W, 'height': H, 'layout': 'NCHW', 'scale': 1.0 / 255.0, 'mean': None, 'std': None, 'rgb': True,
            'letterbox': bool(s['letterbox'])}


def load_rgb(path: str) -> np.ndarray:
    import cv2
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise TrainingError(f'cannot read image {path}')
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def to_model(boxes, back: dict) -> np.ndarray:
    """Image-pixel boxes [x0 y0 x1 y1] to the model's input pixels (the inverse of core.models.to_image)."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]] * back['kx'] + back['px']
    b[:, [1, 3]] = b[:, [1, 3]] * back['ky'] + back['py']
    return b


def _flip(seed: int, epoch: int, i: int) -> bool:
    return (_mix(seed, epoch, i, 0xF11B) >> 11) / float(1 << 52) < 0.5


class DetectionData:
    """A map-style dataset of samples for torch's DataLoader: index (i, epoch) → (image tensor 3 × H × W in [0, 1],
    target {'boxes' (N, 4) model pixels, 'labels' (N,)}, i). Ignored objects are not targets. A plain class (no torch
    at import) that pickles, for the data-loader's worker processes."""

    def __init__(self, samples: list, spec: dict, labels: dict, hflip: bool, seed: int):
        self.samples, self.spec, self.labels, self.hflip, self.seed = samples, spec, labels, hflip, seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        import torch
        from .models import preprocess
        i, epoch = key
        s = self.samples[i]
        x, back = preprocess(load_rgb(s['image']), self.spec)
        x = x[0]
        W, H = self.spec['width'], self.spec['height']
        keep = [k for k, ig in enumerate(s['ignore']) if not ig]
        b = to_model([s['boxes'][k] for k in keep], back)
        lab = np.asarray([self.labels[s['classes'][k]] for k in keep], np.int64)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        ok = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)
        b, lab = b[ok], lab[ok]
        if self.hflip and _flip(self.seed, epoch, i):
            x = np.ascontiguousarray(x[:, :, ::-1])
            b = np.c_[W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]]
        return (torch.from_numpy(x), {'boxes': torch.as_tensor(b, dtype=torch.float32).reshape(-1, 4),
                                      'labels': torch.as_tensor(lab, dtype=torch.int64)}, i)


class EpochBatches:
    """The batch sampler: the batches of the current epoch from `start`, in an order that is a function of (seed,
    epoch) only (so a resumed run sees the same batches)."""

    def __init__(self, n: int, batch: int, seed: int):
        self.n, self.batch, self.seed = n, batch, seed
        self.epoch, self.start = 0, 0

    def order(self, epoch: int) -> list:
        perm = np.random.default_rng(_mix(self.seed, epoch, 0x5EED)).permutation(self.n)
        return [perm[j:j + self.batch].tolist() for j in range(0, self.n, self.batch)]

    def __len__(self):
        return max(-(-self.n // self.batch) - self.start, 0)

    def __iter__(self):
        for b in self.order(self.epoch)[self.start:]:
            yield [(i, self.epoch) for i in b]


def _collate(batch):
    return tuple(zip(*batch))


def _worker_init(worker_id):
    import cv2
    cv2.setNumThreads(1)


# ================================================================================================ prediction
def predict(model, samples: list, s: dict, device, batch: int = 8, cancel=None) -> list:
    """The model's detections on samples as harness frames: {'image_id', 'drive' (the resampling unit), 'gt':
    [{'bbox' xywh, 'class', 'ignore'}], 'det': [{'bbox' xywh, 'class', 'score'}]} in image pixels."""
    import torch
    from .models import preprocess, to_image
    names = label_names(s)
    spec = input_spec(s)
    model.eval()
    out = []
    with torch.inference_mode():
        for j in range(0, len(samples), batch):
            if cancel and cancel():
                raise Cancelled()
            chunk = samples[j:j + batch]
            xs, backs = [], []
            for sm in chunk:
                x, back = preprocess(load_rgb(sm['image']), spec)
                xs.append(torch.from_numpy(x[0]).to(device))
                backs.append(back)
            res = model(xs)
            for sm, back, r in zip(chunk, backs, res):
                bx = to_image(r['boxes'].float().cpu().numpy(), back)
                sc = r['scores'].float().cpu().numpy()
                lb = r['labels'].cpu().numpy()
                det = [{'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        'class': names[int(l)], 'score': float(c)}
                       for b, c, l in zip(bx, sc, lb) if 0 <= int(l) < len(names) and names[int(l)] in s['classes']]
                gt = [{'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]], 'class': c, 'ignore': int(ig)}
                      for b, c, ig in zip(sm['boxes'], sm['classes'], sm['ignore'])]
                out.append({'image_id': sm['id'], 'drive': sm['unit'], 'gt': gt, 'det': det})
    return out


def score_frames(frames: list) -> dict:
    """COCO numbers of frames (core.metrics, pycocotools' definitions): AP, AP50, AP75, AR100 and AP per class."""
    from . import metrics as M
    if not any(not g['ignore'] for f in frames for g in f['gt']):
        return {'AP': None, 'AP50': None, 'AP75': None, 'AR100': None, 'per_class': {}, 'n_gt': 0,
                'note': 'no scored object in the validation set'}
    r = M.coco_evaluate(frames)
    return {'AP': r['AP'], 'AP50': r['AP50'], 'AP75': r['AP75'], 'AR100': r['AR100'],
            'per_class': {c: v['AP'] for c, v in r['per_class'].items()}, 'n_gt': r['n_gt'],
            'n_gt_ignored': r['n_gt_ignored'], 'n_det': r['n_det']}


# ================================================================================================ export
class _Export:
    """Wraps a torchvision detector as images (1, 3, H, W) → (boxes (N, 4) xyxy, scores (N,), labels (N,)): the
    boxes_scores_labels layout core.models decodes."""

    @staticmethod
    def wrap(model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                o = self.m([x[0]])[0]
                return o['boxes'], o['scores'], o['labels']
        return Wrapped(model).eval()


def export_onnx(model, path: str, s: dict, example: np.ndarray, opset: int = OPSET) -> dict:
    """Export the detector (on the CPU) to ONNX with a fixed input size, traced on an example input (1, 3, H, W);
    checked by onnx.checker. Returns {'path', 'opset', 'sha256', 'bytes', 'seconds'}."""
    import copy
    import warnings
    import onnx
    import torch
    from .provenance import sha256_file
    t0 = time.perf_counter()
    m = _Export.wrap(copy.deepcopy(model).cpu().eval())
    tmp = path + '.part'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.onnx.export(m, (torch.from_numpy(example),), tmp, opset_version=int(opset), input_names=['images'],
                          output_names=['boxes', 'scores', 'labels'], dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(tmp))
    os.replace(tmp, path)
    return {'path': path, 'opset': int(opset), 'sha256': sha256_file(path), 'bytes': os.path.getsize(path),
            'seconds': round(time.perf_counter() - t0, 3)}


def registration(s: dict, onnx_path: str, run_id: str, name: str | None = None, sha256: str | None = None,
                 data_terms: list | None = None) -> dict:
    """A model registration (core.models' format) for the exported ONNX file."""
    a = ARCHS[s['arch']]
    terms = list(data_terms or ())
    w = weights_needed(s)
    if w:
        terms.insert(0, f'{w}: {WEIGHTS[w]["licence"]}')
    import re
    from .provenance import sha256_bytes
    rid = re.sub(r'[^A-Za-z0-9_.-]', '-', str(run_id)).lstrip('_.-') or 'trained'
    if len(rid) > 64:                    # the project schema's id: at most 64 characters
        rid = rid[:55] + '-' + sha256_bytes(str(run_id).encode())[:8]
    return {'id': rid, 'name': name or f'{s["arch"]} ({run_id})', 'kind': 'detector',
            'runtime': 'onnx', 'path': os.path.abspath(onnx_path), 'input': input_spec(s),
            'output': {'format': 'boxes_scores_labels', 'score_threshold': float(s['score_threshold']),
                       'nms_iou': float(a['nms_iou']), 'boxes': 'xyxy', 'normalised': False},
            'classes': label_names(s), 'class_map': {c: c for c in s['classes']},
            'licence': 'BSD-3-Clause', 'licence_note': 'torchvision architecture code (BSD-3-Clause); the weights also '
                                                       'carry the terms of the data they were trained on: '
                                                       + ('; '.join(terms) if terms else 'Falcon drives only'),
            'source': {'training_run': os.path.dirname(os.path.abspath(onnx_path)), 'sha256': sha256}}


def verify_onnx(model, reg: dict, samples: list, s: dict, n: int = 8, box_tol: float = 0.5,
                score_tol: float = 1e-3) -> dict:
    """Run the exported model through core.models.Runner and the torch model (CPU, float32) on the same images and
    decode both with core.models.decode_detections: every detection must have a partner of the same class within
    box_tol px (sub-pixel; measured differences are kept — typically 1e-5 px, up to 0.1 px for a barely trained model
    whose box deltas are large, as onnxruntime's RoiAlign and convolutions round differently) and score_tol (a
    detection may lack a partner only when its score lies within score_tol of the threshold)."""
    import copy
    import torch
    from .models import Runner, decode_detections, preprocess
    runner = Runner(reg)
    m = _Export.wrap(copy.deepcopy(model).cpu().eval())
    thr = float(reg['output']['score_threshold'])
    n_t = n_o = unmatched = 0
    max_box = max_score = 0.0
    for sm in samples[:n]:
        rgb = load_rgb(sm['image'])
        x, back = preprocess(rgb, reg['input'])
        with torch.inference_mode():
            outs = [t.numpy() for t in m(torch.from_numpy(x))]
        dt = decode_detections(outs, reg, back)
        do = runner.predict(rgb)
        n_t += len(dt)
        n_o += len(do)
        used = [False] * len(do)
        for d in dt:
            best, bj = None, -1
            for j, e in enumerate(do):
                if used[j] or e['class'] != d['class']:
                    continue
                dist = max(abs(p - q) for p, q in zip(d['bbox'], e['bbox']))
                if best is None or dist < best:
                    best, bj = dist, j
            if bj >= 0 and best <= box_tol * 4:
                used[bj] = True
                max_box = max(max_box, best)
                max_score = max(max_score, abs(d['score'] - do[bj]['score']))
            elif abs(d['score'] - thr) > score_tol:
                unmatched += 1
        unmatched += sum(1 for j, e in enumerate(do) if not used[j] and abs(e['score'] - thr) > score_tol)
    ok = unmatched == 0 and max_box <= box_tol and max_score <= score_tol
    return {'images': min(n, len(samples)), 'torch_detections': n_t, 'onnx_detections': n_o, 'unmatched': unmatched,
            'max_box_diff_px': max_box, 'max_score_diff': max_score, 'box_tol_px': box_tol, 'score_tol': score_tol,
            'providers': list(runner.providers), 'ok': bool(ok)}


# ================================================================================================ the training
def _versions() -> dict:
    import cv2
    import torch
    import torchvision
    v = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__,
         'opencv': cv2.__version__, 'cuda': torch.version.cuda, 'cudnn': None}
    try:
        v['cudnn'] = torch.backends.cudnn.version()
    except Exception:            # noqa: BLE001 (no cuDNN)
        pass
    for mod in ('onnx', 'onnxruntime'):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            v[mod] = None
    return v


def _device(s: dict):
    import torch
    want = s['device']
    if want == 'auto':
        want = 'cuda' if torch.cuda.is_available() else 'cpu'
    if want.startswith('cuda') and not torch.cuda.is_available():
        raise TrainingError('device cuda asked for, but torch sees no CUDA device')
    if want == 'cuda':
        want = f'cuda:{torch.cuda.current_device()}'      # an index: the memory cap and statistics need one
    return torch.device(want)


def _save(path: str, obj):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _read_metrics(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _write_metrics(path: str, rows: list):
    from .paths import atomic_write
    atomic_write(path, ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows))


def train(settings: dict, out_dir: str, data: dict | None = None, workspace: str = '', emitter=None, cancel=None,
          argv: list | None = None, weights_dirs_extra=()) -> dict:
    """Train (or resume, or return the finished result of) a detector into out_dir. data: {'train': samples, 'val':
    samples, 'test': samples, 'notes': [...], 'inputs': {role: path}} instead of loading settings['train'] / ['val'] /
    ['test'] (the mixed experiment). A test set is predicted once, with the chosen model, into
    test_predictions.jsonl. Returns the summary (also in run.json "summary"). Raises TrainingError (bad input) or
    Cancelled (after saving a checkpoint and run.json with status cancelled). The process-wide state a training sets
    (deterministic algorithms, cuDNN flags, the global random generators) is restored afterwards."""
    import random
    import torch
    saved = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, random.getstate(),
             np.random.get_state(), torch.get_rng_state())
    try:
        return _train(settings, out_dir, data, workspace, emitter, cancel, argv, weights_dirs_extra)
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved[2], saved[3]
        random.setstate(saved[4])
        np.random.set_state(saved[5])
        torch.set_rng_state(saved[6])


def _train(settings, out_dir, data, workspace, emitter, cancel, argv, weights_dirs_extra) -> dict:
    from . import protocol as P
    from . import provenance as PV
    em = emitter or _Quiet()
    cancel = cancel or (lambda: False)
    s = normalise(settings)
    os.makedirs(out_dir, exist_ok=True)
    run_path = os.path.join(out_dir, 'run.json')
    ck_dir = os.path.join(out_dir, 'checkpoints')
    # ---- data
    if data is None:
        tr, notes, ins = load_sources(s['train'], s['classes'])
        va, n2, i2 = load_sources(s['val'], s['classes'])
        te, n3, i3 = load_sources(s['test'], s['classes'])
        notes, ins = notes + n2 + n3, {**ins, **i2, **i3}
    else:
        tr, va, te = (list(data.get(k) or ()) for k in ('train', 'val', 'test'))
        notes, ins = list(data.get('notes') or ()), dict(data.get('inputs') or {})
    if not tr:
        raise TrainingError('no training images')
    man = {'train': manifest(tr), 'val': manifest(va) if va else None, 'test': manifest(te) if te else None}
    data_sha = PV.sha256_json([m and m['sha256'] for m in man.values()])
    key = settings_key(s, data_sha)
    prev = PV.read_json(run_path) if os.path.isfile(run_path) else None
    if prev is not None:
        if prev.get('settings_key') != key:
            raise TrainingError(f'{out_dir} holds a training with other settings or data: choose another folder')
        if prev.get('status') == 'ok':
            em.log(f'{out_dir}: already finished; nothing to do')
            return prev['summary']
    terms = sorted({DATA_TERMS[x['source']] for x in tr + va if x['source'] in DATA_TERMS})      # what trains it
    PV.write_json(os.path.join(out_dir, 'data.json'), {'schema': 'falcon.training_data/1', 'notes': notes,
                                                       'data_terms': terms, **man})
    wname = weights_needed(s)
    wpath = None
    if wname:
        wpath, wnotes = find_weights(wname, weights_dirs(workspace, list(weights_dirs_extra) +
                                                         list(s.get('weights_dirs') or ())))
        for x in wnotes:
            em.log(x, 'warning')
        if wpath is None:
            w = WEIGHTS[wname]
            raise TrainingError(f'the pinned weights {wname} ({w["file"]}, {w["bytes"] / 2 ** 20:.1f} MiB, '
                                f'{w["licence"]}) are not here: download them first (falcon train --fetch {wname} '
                                f'--workspace W); SHA-256 {w["sha256"]} is checked')
        ins[f'weights:{wname}'] = wpath
    # ---- torch
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    dev = _device(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    import random
    random.seed(s['seed'])
    np.random.seed(s['seed'] % (2 ** 32))
    amp = bool(s['amp']) and dev.type == 'cuda'
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(s['gpu_memory_fraction']), dev)
        torch.cuda.reset_peak_memory_stats(dev)
    run = prev if prev is not None else PV.new_run('train', argv or [], s['seed'], s, ins)
    run.update(schema=SCHEMA, status='running', settings_key=key, exit_code=None, error=None)
    run.setdefault('timings', {'train_s': 0.0, 'val_s': 0.0, 'images': 0, 'sessions': 0})
    run['timings']['sessions'] += 1
    gpu = None
    if dev.type == 'cuda':
        p = torch.cuda.get_device_properties(dev)
        gpu = {'name': p.name, 'memory_mib': round(p.total_memory / 2 ** 20), 'capability': f'{p.major}.{p.minor}'}
    run['device'] = {'type': dev.type, 'gpu': gpu, 'amp': amp, 'threads': torch.get_num_threads()}
    run['versions'] = _versions()
    run['data'] = {**{k: m and {kk: v for kk, v in m.items() if kk != 'images'} for k, m in man.items()},
                   'notes': notes, 'data_terms': terms}
    PV.write_json(run_path, run)
    model, info = build_model(s, wpath)
    model.to(dev)
    run['model'] = info
    names = label_names(s)
    labels = {c: names.index(c) for c in s['classes']}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=s['lr'], momentum=s['momentum'], weight_decay=s['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ds = DetectionData(tr, input_spec(s), labels, bool(s['hflip']), int(s['seed']))
    sampler = EpochBatches(len(tr), int(s['batch']), int(s['seed']))
    ipe = len(sampler.order(0))
    total = ipe * int(s['epochs'])
    warm = s['warmup_iters'] if s['warmup_iters'] is not None else min(200, max(1, total // 10))
    st = {'epoch': 0, 'k': 0, 'best_ap': None, 'best_epoch': None, 'bad': 0, 'loss_sum': 0.0, 'loss_n': 0,
          'parts': {}, 'skipped': 0, 'stopped_early': False}
    metrics_path = os.path.join(out_dir, 'metrics.jsonl')
    last = os.path.join(ck_dir, 'last.pt')
    if os.path.isfile(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck.get('settings_key') != key:
            raise TrainingError(f'{last} is from another training')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        st.update(ck['state'])
        em.log(f'resuming at epoch {st["epoch"] + 1}, iteration {st["k"]} of {ipe}')
    rows = [r for r in _read_metrics(metrics_path) if r['epoch'] < st['epoch'] + 1]
    _write_metrics(metrics_path, rows)
    import warnings
    caught_ref = []

    def checkpoint(path=last):
        _save(path, {'settings_key': key, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                     'scaler': scaler.state_dict(), 'state': dict(st)})

    def finish(status, code, error=None):
        run['timings']['peak_gpu_mib'] = (round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)
                                          if dev.type == 'cuda' else None)
        run['timings']['peak_gpu_reserved_mib'] = (round(torch.cuda.max_memory_reserved(dev) / 2 ** 20, 1)
                                                   if dev.type == 'cuda' else None)
        run['timings']['images_per_s'] = (round(run['timings']['images'] / run['timings']['train_s'], 2)
                                          if run['timings']['train_s'] else None)
        warned = sorted({str(w.message).split('. ')[0][:200] for c in caught_ref for w in c
                         if 'deterministic' in str(w.message)} | set(run.get('determinism', {}).get(
                             'nondeterministic_ops_warned') or ()))
        run['determinism'] = {
            'seeded': True, 'cpu_bit_for_bit': 'yes (same settings, data and thread count; tested)',
            'cuda_bit_for_bit': (None if dev.type != 'cuda' else
                                 'no: an operation without a deterministic CUDA implementation ran (below)' if warned
                                 else 'expected (no operation without a deterministic implementation ran; measured '
                                      'bit for bit for FCOS and RetinaNet on one machine)'),
            'nondeterministic_ops_warned': warned}
        run.update(status=status, exit_code=code, error=error, finished=PV.utc_now())
        PV.write_json(run_path, run)

    loader = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('default')
            caught_ref.append(caught)
            if st['epoch'] < int(s['epochs']) and not st['stopped_early']:
                loader = torch.utils.data.DataLoader(
                    ds, batch_sampler=sampler, num_workers=int(s['workers']), collate_fn=_collate,
                    persistent_workers=int(s['workers']) > 0, worker_init_fn=_worker_init,
                    pin_memory=dev.type == 'cuda')
            while st['epoch'] < int(s['epochs']) and not st['stopped_early']:
                e = st['epoch']
                sampler.epoch, sampler.start = e, st['k']
                model.train()
                t0 = time.perf_counter()
                n_img = 0
                for imgs, tgts, _idx in loader:
                    if cancel():
                        run['timings']['train_s'] += time.perf_counter() - t0
                        run['timings']['images'] += n_img
                        raise Cancelled()
                    k = st['k']
                    it = e * ipe + k
                    torch.manual_seed(_mix(s['seed'], e, k))
                    lr = lr_at(it, total, float(s['lr']), int(warm), s['schedule'])
                    for g in opt.param_groups:
                        g['lr'] = lr
                    imgs = [x.to(dev, non_blocking=True) for x in imgs]
                    tgts = [{kk: v.to(dev, non_blocking=True) for kk, v in t.items()} for t in tgts]
                    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                        losses = model(imgs, tgts)
                    loss = sum(losses.values())
                    opt.zero_grad(set_to_none=True)
                    if not torch.isfinite(loss):
                        st['skipped'] += 1
                        if st['skipped'] > 20:
                            raise TrainingError('the loss was not finite 20 times: lower the learning rate')
                    else:
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(params, float(s['clip_norm']))
                        scaler.step(opt)
                        scaler.update()
                        st['loss_sum'] += float(loss.detach())
                        st['loss_n'] += 1
                        for kk, v in losses.items():
                            st['parts'][kk] = st['parts'].get(kk, 0.0) + float(v.detach())
                    st['k'] = k + 1
                    n_img += len(imgs)
                    em.progress((it + 1) / total, f'epoch {e + 1} of {s["epochs"]}, iteration {k + 1} of {ipe}, '
                                                  f'loss {float(loss.detach()):.3f}')
                train_s = time.perf_counter() - t0
                run['timings']['train_s'] += train_s
                run['timings']['images'] += n_img
                row = {'epoch': e + 1, 'iterations': ipe, 'lr_end': lr_at((e + 1) * ipe - 1, total, float(s['lr']),
                                                                          int(warm), s['schedule']),
                       'train_loss': st['loss_sum'] / st['loss_n'] if st['loss_n'] else None,
                       'loss_parts': {kk: v / max(st['loss_n'], 1) for kk, v in st['parts'].items()},
                       'skipped_nonfinite': st['skipped'], 'train_s': round(train_s, 3),
                       'images_per_s': round(n_img / train_s, 2) if train_s > 0 and n_img else None,
                       'time': PV.utc_now()}
                if va:
                    t1 = time.perf_counter()
                    frames = predict(model, va, s, dev, int(s['val_batch']), cancel)
                    sc = score_frames(frames)
                    row['val'] = sc
                    row['val_s'] = round(time.perf_counter() - t1, 3)
                    run['timings']['val_s'] += row['val_s']
                    ap = sc['AP']
                    em.metric('val_AP', ap, '', 'measured', epoch=e + 1)
                    if ap is not None and (st['best_ap'] is None or ap > st['best_ap']):
                        st['best_ap'], st['best_epoch'], st['bad'] = ap, e + 1, 0
                        _save(os.path.join(ck_dir, 'best.pt'), {'settings_key': key, 'model': model.state_dict(),
                                                                'epoch': e + 1, 'AP': ap})
                    else:
                        st['bad'] += 1
                    if s['patience'] and st['bad'] >= int(s['patience']):
                        st['stopped_early'] = True
                        em.log(f'early stop after epoch {e + 1}: no better validation AP for {st["bad"]} epochs')
                em.metric('train_loss', row['train_loss'], '', 'measured', epoch=e + 1)
                rows.append(row)
                _write_metrics(metrics_path, rows)
                st.update(epoch=e + 1, k=0, loss_sum=0.0, loss_n=0, parts={}, skipped=0)
                checkpoint()
                PV.write_json(run_path, run)
    except Cancelled:
        checkpoint()                    # where it stopped: mid-epoch, or before an epoch's validation
        run['summary'] = {'status': 'cancelled', 'epoch': st['epoch'], 'iteration': st['k']}
        finish('cancelled', P.CANCELLED)
        em.log(f'cancelled: checkpoint saved at epoch {st["epoch"] + 1}, iteration {st["k"]}')
        raise
    except TrainingError as ex:
        finish('failed', P.FAILED, str(ex))
        raise
    except Exception as ex:             # noqa: BLE001 (recorded in run.json, then re-raised)
        if isinstance(ex, torch.OutOfMemoryError) or 'out of memory' in str(ex):
            finish('failed', P.FAILED, f'out of GPU memory: {ex}')
            raise TrainingError(f'out of GPU memory (lower --batch or --size, or raise --gpu-memory-fraction): '
                                f'{str(ex).splitlines()[0]}') from ex
        finish('crashed', P.CRASHED, f'{type(ex).__name__}: {ex}')
        raise
    finally:
        loader = None
    # ---- the final model: the best epoch by validation AP (or the last), tested, exported
    try:
        summary = _finish_model(model, s, out_dir, st, tr, va, te, terms, dev, em, cancel, run)
    except Cancelled:
        run['summary'] = {'status': 'cancelled', 'epoch': st['epoch'], 'iteration': st['k']}
        finish('cancelled', P.CANCELLED)
        raise
    except Exception as e:              # noqa: BLE001 (the export's failure is the training's)
        finish('failed', P.FAILED, f'{type(e).__name__}: {e}')
        raise TrainingError(f'after training: {type(e).__name__}: {e}') from e
    run['summary'] = summary
    if not s['keep_last'] and os.path.isfile(last):
        os.remove(last)                 # only a resume needs it; best.pt and model.onnx stay
    run['outputs'] = PV.hash_tree(out_dir, skip=('run.json',))
    finish('ok', P.OK)
    return summary


def _finish_model(model, s, out_dir, st, tr, va, te, terms, dev, em, cancel, run) -> dict:
    import torch
    from . import provenance as PV
    best = os.path.join(out_dir, 'checkpoints', 'best.pt')
    chosen = 'last'
    if va and s['select'] == 'best' and os.path.isfile(best):
        model.load_state_dict(torch.load(best, map_location=dev, weights_only=False)['model'])
        chosen = f'best (epoch {st["best_epoch"]}, validation AP {st["best_ap"]:.4f})'
    summary = {'status': 'ok', 'epochs_run': st['epoch'], 'stopped_early': st['stopped_early'], 'chosen': chosen,
               'best_epoch': st['best_epoch'], 'best_val_AP': st['best_ap'], 'onnx': None, 'registration': None,
               'verification': None, 'test': None}
    if te:
        em.progress(1.0, f'testing on {len(te)} images', force=True)
        frames = predict(model, te, s, dev, int(s['val_batch']), cancel)
        with open(os.path.join(out_dir, 'test_predictions.jsonl'), 'w', encoding='utf-8') as fh:
            for f in frames:
                fh.write(json.dumps(f, separators=(',', ':')) + '\n')
        summary['test'] = score_frames(frames)
    if s['export']:
        from .models import preprocess
        em.progress(1.0, 'exporting to ONNX', force=True)
        example = preprocess(load_rgb((va or tr)[0]['image']), input_spec(s))[0]
        ex = export_onnx(model, os.path.join(out_dir, 'model.onnx'), s, example, int(s['opset']))
        rid = os.path.basename(os.path.normpath(out_dir))
        reg = registration(s, ex['path'], rid, sha256=ex['sha256'], data_terms=terms)
        ver = verify_onnx(model, reg, va or tr, s, int(s['verify_images']))
        reg['verification'] = ver
        PV.write_json(os.path.join(out_dir, 'model.json'), reg)
        summary.update(onnx=ex, registration=os.path.join(out_dir, 'model.json'), verification=ver)
        run['timings']['export_s'] = ex['seconds']
        if not ver['ok']:
            em.log(f'the ONNX model does not reproduce the torch model\'s detections within tolerance: {ver}',
                   'warning')
    return summary


# ================================================================================================ GPU politeness
def gpu_locks(emitter=None, cancel=None, held_by_queue: bool = False, interval: float = 5.0):
    """For a training run from the CLI on CUDA: wait while a CARLA job holds the CARLA lock, then hold it (CARLA jobs
    of any Falcon process queue behind the training) and the GPU lock (unless a JobQueue already holds that for this
    process: held_by_queue). Returns the held locks (release them at the end)."""
    from .jobs import FileLock
    em = emitter or _Quiet()
    carla, gpu = FileLock('carla-gpu0.lock'), FileLock('compute-gpu0.lock')
    said = False
    while True:
        if cancel and cancel():
            raise Cancelled()
        if carla.acquire():
            if held_by_queue or gpu.acquire():
                return [carla] + ([] if held_by_queue else [gpu])
            carla.release()
            who = gpu.holder()
            why = f'waiting for the GPU lock (held by process {who or "?"})'
        else:
            why = f'waiting for the CARLA job to finish (process {carla.holder() or "?"} holds the CARLA lock)'
        if not said:
            em.log(why)
            said = True
        time.sleep(interval)


def job_spec(argv: list, title: str = 'Training', memory_gb: float = 6.0):
    """The JobSpec of a training (`falcon train …` arguments): a needs_gpu job, so the JobQueue never runs it while a
    CARLA job holds the GPU; the process knows the queue holds the GPU lock (FALCON_GPU_LOCK_HELD)."""
    import sys
    from .jobs import JobSpec
    env = dict(os.environ, FALCON_GPU_LOCK_HELD='1')
    return JobSpec('train', title, [sys.executable, '-m', 'falcon_autonomy', 'train'] + list(argv), env=env,
                   needs_gpu=True, memory_gb=memory_gb, resumable=True)


# ================================================================================================ the command
class CancelFlag:
    """Set by SIGTERM / SIGINT or a {"cmd": "cancel"} line on stdin."""

    def __init__(self):
        self.set = False

    def __call__(self):
        return self.set

    def install(self):
        import signal
        from .protocol import CommandReader

        def on_sig(*_):
            self.set = True
        signal.signal(signal.SIGTERM, on_sig)
        signal.signal(signal.SIGINT, on_sig)
        CommandReader(lambda d: setattr(self, 'set', True) if d.get('cmd') == 'cancel' else None).start()
        return self


def _size(text: str) -> list:
    w, _, h = str(text).lower().partition('x')
    try:
        return [int(w), int(h)]
    except ValueError:
        raise TrainingError(f'size is WIDTHxHEIGHT, e.g. 640x384; got {text!r}') from None


def main_train(argv) -> int:
    """`falcon train (--out DIR | --workspace W [--id ID]) --train SRC... [--val SRC...] [--version V.json]
    [--arch A] [--init scratch|imagenet|coco] [...]`, `falcon train --weights` (the pinned weights) or
    `falcon train --fetch NAME --workspace W`: protocol lines on stdout; 0 ok, 2 failed, 3 cancelled."""
    import argparse
    from . import protocol as P
    from . import provenance as PV
    ap = argparse.ArgumentParser(prog='falcon train', description='fine-tune a detector (see docs/training.md)')
    ap.add_argument('--config', help='a JSON file of settings (the flags below override it)')
    ap.add_argument('--out', '-o', default='', help='the training folder (default: WORKSPACE/training/ID)')
    ap.add_argument('--workspace', default='', help='the workspace (training/, models/pretrained/)')
    ap.add_argument('--id', default='', help='the training\'s id (default: time and arch)')
    ap.add_argument('--train', action='append', default=None, metavar='KIND:PATH[?k=v&…]',
                    help=f'a training source (repeatable); KIND one of {", ".join(SOURCE_KINDS)}')
    ap.add_argument('--val', action='append', default=None, metavar='KIND:PATH[?k=v&…]', help='a validation source')
    ap.add_argument('--test', action='append', default=None, metavar='KIND:PATH[?k=v&…]',
                    help='a held-out test source, predicted once with the chosen model')
    ap.add_argument('--select', choices=('best', 'last'), help='the exported epoch: best validation AP (default) or '
                                                               'the last')
    ap.add_argument('--version', help='a dataset version file: trains on its train split, validates on its val split')
    ap.add_argument('--arch', choices=tuple(ARCHS))
    ap.add_argument('--init', choices=INITS)
    ap.add_argument('--backbone-weights', choices=tuple(k for k, w in WEIGHTS.items() if w['what'] == 'backbone'))
    ap.add_argument('--classes', help='comma-separated Falcon classes')
    ap.add_argument('--size', help='the model\'s input, WIDTHxHEIGHT (default 640x384)')
    ap.add_argument('--letterbox', action='store_true', default=None)
    for k, t in (('epochs', int), ('batch', int), ('lr', float), ('seed', int), ('workers', int), ('patience', int),
                 ('warmup-iters', int), ('trainable-layers', int), ('gpu-memory-fraction', float)):
        ap.add_argument(f'--{k}', type=t)
    ap.add_argument('--schedule', choices=('cosine', 'step'))
    ap.add_argument('--device', help='auto (default), cpu, cuda or cuda:N')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--no-hflip', action='store_true')
    ap.add_argument('--weights-dir', action='append', default=[], help='another folder holding pinned weights')
    ap.add_argument('--weights', action='store_true', help='list the pinned pretrained weights and exit')
    ap.add_argument('--fetch', metavar='NAME', help='download pinned weights into WORKSPACE/models/pretrained')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    if a.weights:
        for k, w in WEIGHTS.items():
            print(f'{k:42s} {w["bytes"] / 2 ** 20:7.1f} MiB  {w["what"]:8s} sha256 {w["sha256"]}\n'
                  f'{"":42s} {w["url"]}\n{"":42s} {w["licence"]}; {w["published"]}')
        return P.OK
    em = P.Emitter()
    em.hello('train')
    if a.fetch:
        if not a.workspace or a.fetch not in WEIGHTS:
            print(f'failed: --fetch NAME needs --workspace and NAME one of {", ".join(WEIGHTS)}', flush=True)
            return P.FAILED
        w = WEIGHTS[a.fetch]
        em.log(f'downloading {w["url"]} ({w["bytes"] / 2 ** 20:.1f} MiB; licence {w["licence"]}; SHA-256 '
               f'{w["sha256"]})')
        try:
            path = fetch_weights(a.fetch, a.workspace, lambda f: em.progress(f, 'download'))
        except (OSError, ValueError) as e:
            print(f'failed: {e}', flush=True)
            return P.FAILED
        em.result('ok', path=path)
        return P.OK
    flag = CancelFlag().install()
    locks = []
    out = ''
    try:
        s = PV.read_json(a.config) if a.config else {}
        over = {'arch': a.arch, 'init': a.init, 'backbone_weights': a.backbone_weights, 'epochs': a.epochs,
                'batch': a.batch, 'lr': a.lr, 'seed': a.seed, 'workers': a.workers, 'patience': a.patience,
                'warmup_iters': a.warmup_iters, 'trainable_layers': a.trainable_layers,
                'gpu_memory_fraction': a.gpu_memory_fraction, 'schedule': a.schedule, 'device': a.device,
                'letterbox': a.letterbox, 'select': a.select}
        s.update({k: v for k, v in over.items() if v is not None})
        if a.classes:
            s['classes'] = [c.strip() for c in a.classes.split(',') if c.strip()]
        if a.size:
            s['size'] = _size(a.size)
        if a.no_amp:
            s['amp'] = False
        if a.no_hflip:
            s['hflip'] = False
        if a.train:
            s['train'] = list(a.train)
        if a.val:
            s['val'] = list(a.val)
        if a.test:
            s['test'] = list(a.test)
        if a.version:
            v = os.path.abspath(a.version)
            s.setdefault('train', []).append({'kind': 'version', 'path': v, 'split': 'train'})
            s.setdefault('val', []).append({'kind': 'version', 'path': v, 'split': 'val'})
        s['train'] = [parse_source(x) if isinstance(x, str) else x for x in s.get('train') or ()]
        s['val'] = [parse_source(x) if isinstance(x, str) else x for x in s.get('val') or ()]
        s['test'] = [parse_source(x) if isinstance(x, str) else x for x in s.get('test') or ()]
        s = normalise(s)
        if not s['train']:
            raise TrainingError('at least one --train source (or --version) is needed')
        out = a.out or (os.path.join(a.workspace, 'training', a.id or f'{time.strftime("%Y%m%d-%H%M%S")}-{s["arch"]}')
                        if a.workspace else '')
        if not out:
            raise TrainingError('--out DIR or --workspace W is needed')
        if _device(s).type == 'cuda':
            locks = gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
        res = train(s, out, workspace=a.workspace, emitter=em, cancel=flag, argv=['train'] + list(argv),
                    weights_dirs_extra=a.weights_dir)
        em.result('ok', run_json=os.path.join(out, 'run.json'), summary={
            'best_val_AP': res.get('best_val_AP'), 'epochs': res.get('epochs_run'),
            'registration': res.get('registration'), 'onnx_verified': (res.get('verification') or {}).get('ok'),
            'test_AP': (res.get('test') or {}).get('AP')})
        return P.OK
    except Cancelled:
        em.result('cancelled', run_json=os.path.join(out, 'run.json') if out else None)
        return P.CANCELLED
    except (TrainingError, ValueError, OSError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        return P.FAILED
    finally:
        for lk in locks:
            lk.release()
