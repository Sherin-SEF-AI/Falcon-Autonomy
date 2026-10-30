"""Learned sim-to-real image translation with the labels kept: train an unpaired translator from Falcon's rendered
frames to a real camera's look, translate recorded drives into new falcon-raw runs whose every label stays valid,
measure how far each translated frame drifted from its input, and measure the gap and the downstream effect (numpy at
import; OpenCV in the functions; torch only inside the functions that train or run a network, through
core.translate_net).

Method. CUT — contrastive unpaired translation (Park, Efros, Zhang and Zhu, ECCV 2020) — in Falcon's own
implementation (core.translate_net): a ResNet generator, a PatchGAN discriminator with the least-squares loss and the
PatchNCE loss that makes each output patch match the input patch at the same place; identity NCE on real images
(nce_idt). Geometry-preserving by design: no warping, flow or attention layers; local receptive fields; nearest-
neighbour upsampling (no checkerboard edges); the output is a residual in the logit of the input with a zero-
initialised head, so the untrained generator is the identity. Falcon adds a conditioning input: the frame's own
semantic label, as one-hot channels of GROUPS (road, sidewalk, structure, pole/sign, vegetation, terrain, sky,
person, vehicle, other) — the generator only ever translates simulated frames, which always have labels, so it knows
which pixels are road or car without having to guess (a real image, in the identity pass, gets all-zero channels:
"unknown"). Optionally (depth) one channel more: the inverse range, min(1, DEPTH_MIN_M / range) from the run's range
image, 0 for the sky and for a real image (whose all-zero label tells "unknown" from "sky": depth needs the label;
the identity pass then differs from the translation pass by eleven zero channels instead of ten — it regularises
the "unknown" mode either way). This is the "label as a conditioning input" form of semantic consistency; the other
form, a loss from a frozen segmenter on the output (CyCADA, Hoffman et al. 2018), is not implemented (it needs a
segmenter trained on real data, which Falcon does not download). CycleGAN (Zhu et al. 2017) is not implemented: two
generators double the cost, and cycle consistency lets a generator hide the input in imperceptible signal instead of
keeping it in place (Chu, Zhmoginov and Sandler 2017) — the one-sided PatchNCE ties output to input where it is. A diffusion + ControlNet
path is not implemented: it needs a model download whose licence the user must accept.

Data. Simulated frames: Falcon runs (one pinhole camera each, its rgb and semantic label; every= n-th frame). Real
frames: any core.realgap set read in place (kitti_raw, kitti_object, nuscenes, bdd100k, coco, folder, dashcam) —
nothing is copied. Every image is resized to work_width (640 px, as the gap measures) and training takes random crops
(crop, default 192 px, clamped to the smallest image) with a seeded horizontal flip.

Training. Adam (lr 2·10⁻⁴, β 0.5, 0.999) for G with the projection heads and for D; constant, then decaying linearly to
0 from decay_from of the iterations; λ_GAN 1, λ_NCE 10 (the output's NCE and the identity NCE averaged, as CUT does;
CUT's own default is 1 and FastCUT's 10 — measured on CARLA → KITTI, λ 1 repainted content and every held-out frame
exceeded the drift threshold, λ 10 kept Town10HD's frames under it: docs/translation.md), 256 patches per tap,
τ 0.07, 12 000 iterations of batch 4. bfloat16 autocast on CUDA. Seeded: crops, flips and patch draws are functions of (seed,
iteration); on the CPU two runs repeat bit for bit (tested); on CUDA cuDNN's kernels are not guaranteed to. A
checkpoint every ckpt_every iterations and on a cancel; the same command resumes; a finished training is not run
again; a folder made with other settings or data is refused. Every probe_every iterations eight fixed simulated frames
are translated whole and their drift measured (metrics.jsonl "probe"): the training curve of label preservation.

Label preservation (measured, not assumed). Per translated frame, at the work width, on grayscale:
- structure: the mean of SSIM's structure term (Wang et al. 2004), (σxy + C3) / (σx σy + C3), Gaussian window
  σ 1.5 px, C3 = (0.03 · 255)² / 2 — insensitive to brightness and contrast, falls when edges move or vanish;
- edges: the input's strong edges (Canny with thresholds at its own gradient-magnitude percentile, top 5 %) found in
  the output's edges (top 15 %) within 1.5 px — edge recall, "lost structure" — and the output's strong edges found
  in the input's — edge precision, "added structure"; thresholds from each image's own gradients, so a tone change
  alone does not move edges;
- objects: per labelled object (THINGS, at least MIN_BOX_PX tall at the work width) its box's mean structure; the
  frame keeps the worst;
- lights (night frames only: the input's median gray below NIGHT_GRAY): the input's light sources (blobs of at least
  LIGHT_MIN_PX at gray ≥ LIGHT_HI) found in the output (gray ≥ LIGHT_LO within LIGHT_TOL_PX) — light recall, lights
  removed — and the output's light sources found in the input — light precision, lights invented. Measured: at night
  the structure and edge terms see neither (every light removed: drift 0.27; twelve lamps painted into the dark: 0.04
  on the user's night dashcam frames), because the dark areas' local contrast sits under SSIM's constant C3;
- drift = 1 − min(structure, edge recall, edge precision, worst object's structure[, light recall, light precision]);
- reported beside the drift, not in it: the dynamic actors' (is_actor: vehicles, two-wheelers, pedestrians, movable
  objects, animals) worst box structure, and their boxes' edge recall and precision pooled over the frame.
A frame whose drift exceeds DRIFT_THRESHOLD is flagged; by default its translation is refused and the raw frame kept
(on_drift "raw"), or kept and listed (on_drift "flag"). The threshold is a heuristic, calibrated on known transforms
of real CARLA frames (docs/translation.md): JPEG at quality 75 gives drift 0.04, Falcon's appearance matching 0.06,
a 1.5 px blur 0.23, the whole image shifted by 1 px 0.25 and by 2 px 0.52; DRIFT_THRESHOLD = 0.35 lies between a
1-px and a 2-px shift. Optionally a registered segmenter (core.models) is run on the input and the output: pixel
agreement, and mIoU against the frame's semantic label for both — a measurement, not a gate, and none is bundled
(NVIDIA's SegFormer weights, proposed for an IoU gate, allow non-commercial use only).

Foreground pinning (pin_actors, `apply --pin-actors`). After translation, inside each dynamic actor's instance mask
(the run's instance image, the objects' ids) the translated pixels are replaced by a colour map of the input fitted
per object from the input's to the translation's pixels there (fit_colour: a gain and an offset per channel, least
squares with a ridge towards gain 1 and Huber reweighting; an offset only for objects under PIN_MIN_PX pixels or
when the map would flatten the object's luminance), with a PIN_FEATHER_PX ring blended outside the mask.
The actor's tone and colour follow the translator; its silhouette and every edge inside it are the input's, because
each output pixel is a function of the same input pixel alone. What the translator changed there beyond colour
(texture, structure) is discarded — the fit's residual (rms, 8-bit levels) says how much.

Translated runs. translate_run() makes a new falcon-raw run: every per-frame file of the source is hard-linked (or
copied across file systems) and the run's own records (frames.jsonl, actors.jsonl, …) copied, except the translated
cameras' rgb, which are written new at their full resolution (a frame without its semantic label, which a
conditioned translator needs — or its range image for a depth-conditioned one, or its instance image and objects
when pinning — is kept raw and said so) — the
network runs at the work width and its change (output − input, upsampled) is added to the full-resolution input, so
the labels' pixel grid is untouched (native=True runs the network at full resolution instead). run.json keeps the
source's (its start time, rig and settings, so timestamps and calibration are the source's) with worker "translate",
the source, the model and its weights' SHA-256, the drift of every frame (translation.jsonl) and the data terms the
model inherits. Exporters, the harness, the gap and training read it as any drive.

Terms. The translator's weights are made from the real images it was trained on: a model trained on KITTI inherits
KITTI's terms (CC BY-NC-SA 3.0: attribution, non-commercial, share-alike), and so do the images it makes. The model's
translator.json and every translated run's run.json say so (DATA_TERMS).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import numpy as np

SCHEMA = 'falcon.translator/2'          # /2: arch.depth_min_m (the inverse-range channel, or None); /1 has none
SCHEMAS = ('falcon.translator/1', SCHEMA)
RUN_SCHEMA = 'falcon.translation/1'
CHECK_SCHEMA = 'falcon.translation_check/1'
RESULTS_SCHEMA = 'falcon.translation_results/1'
METHODS = ('cut',)
REAL_KINDS = ('kitti_raw', 'kitti_object', 'nuscenes', 'bdd100k', 'coco', 'folder', 'dashcam')
WORK_WIDTH = 640                      # = core.realgap.WORK_WIDTH
EDGE_STRONG, EDGE_WEAK, EDGE_TOL_PX = 0.05, 0.15, 1.5
MIN_BOX_PX = 16
DRIFT_THRESHOLD = 0.35                # heuristic: between a 1-px (0.25) and a 2-px (0.52) shift of a whole frame
C3 = (0.03 * 255) ** 2 / 2
NIGHT_GRAY = 60                       # a frame whose median gray is below this is a night frame (the light terms apply)
LIGHT_HI, LIGHT_LO, LIGHT_TOL_PX, LIGHT_MIN_PX = 200, 100, 4, 4    # LO: a light may dim or brighten by 1 EV
GROUPS = (('road', (1, 24, 27)), ('sidewalk', (2, 25)), ('structure', (3, 4, 5, 26, 28)), ('pole_sign', (6, 7, 8)),
          ('vegetation', (9,)), ('terrain', (10,)), ('sky', (11,)), ('person', (12, 13)),
          ('vehicle', (14, 15, 16, 17, 18, 19)), ('other', (0, 20, 21, 22, 23, 29)))
DATA_TERMS = {
    'kitti_raw': 'KITTI: CC BY-NC-SA 3.0 (attribution, non-commercial, share-alike)',
    'kitti_object': 'KITTI: CC BY-NC-SA 3.0 (attribution, non-commercial, share-alike)',
    'nuscenes': 'nuScenes: CC BY-NC-SA 4.0 for non-commercial use; commercial use needs a licence from Motional',
    'bdd100k': 'BDD100K: its own licence (research use; see bdd-data.berkeley.edu)',
    'coco': 'COCO: annotations CC BY 4.0; images under their Flickr licences',
    'folder': 'a folder of images: its terms are unknown to Falcon — check them',
    'dashcam': 'your own dashcam footage (core.dashcam): your terms — it shows where and when you drove; keep the '
               'model and its images private unless you mean to share that',
}
DEPTH_MIN_M = 2.0                     # inverse-range channel: min(1, DEPTH_MIN_M / range); 0 where nothing is in range
# foreground pinning: the objects whose pixels may change only by a colour map of their own (the dynamic actors; a
# CARLA "dynamic" object is a movable one — Falcon's cattle, cart and vendor proxies, bins, animals)
PIN_CLASSES = ('car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'rider', 'pedestrian', 'dynamic')
PIN_FEATHER_PX = 1                    # the blend ring outside each pinned mask (0: a hard edge)
PIN_MIN_PX = 48                       # fewer pixels than this: a per-channel offset instead of the colour matrix
PIN_RIDGE = 1e-3                      # per pixel, colours in [0, 1]: pulls a channel's gain to 1 where the object's
#                                       colours barely spread in it (≈ a spread under 8 grey levels)
PIN_SAMPLE = 20000                    # pixels per object the fit uses (an even stride over the mask)
PIN_HUBER = 1.345                     # Huber's constant (× the residuals' robust scale) for the reweighted fit
PIN_MIN_GAIN = 0.25                   # heuristic: a fitted map whose luminance slope inside the mask is below this (the
#                                       translator's content there is unrelated to the input's) is not followed: the
#                                       per-channel median offset instead
LUMA = np.array([0.299, 0.587, 0.114])
PIN_DEFAULT = None                    # translate_run / `apply` without --pin-actors or --no-pin-actors: pin every
#                                       camera that has instance images, translate the others unpinned (measured:
#                                       pinning lowered the drift and cost no realism, docs/translation.md)
DEFAULTS = {
    'method': 'cut', 'sim': [], 'real': [], 'camera': None, 'every': 1, 'max_images': 2000,
    'work_width': WORK_WIDTH, 'crop': 192, 'batch': 4, 'iterations': 12000, 'lr': 2e-4, 'beta1': 0.5, 'beta2': 0.999,
    'decay_from': 0.5, 'ngf': 64, 'n_blocks': 9, 'ndf': 64, 'lambda_gan': 1.0, 'lambda_nce': 10.0, 'nce_idt': True,
    'num_patches': 256, 'tau': 0.07, 'condition': True, 'depth': False, 'hflip': True, 'seed': 0, 'device': 'auto',
    'amp': True, 'log_every': 100, 'ckpt_every': 1000, 'probe_every': 1000, 'probes': 8, 'gpu_memory_fraction': 0.6,
    'keep_checkpoint': False,
}
# settings that do not change the result (left out of the key that guards a resume)
VOLATILE = ('log_every', 'ckpt_every', 'gpu_memory_fraction', 'keep_checkpoint', 'id')


class TranslateError(Exception):
    pass


# ================================================================================================ label preservation
def _gray(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def working(rgb: np.ndarray, width: int = WORK_WIDTH, nearest: bool = False) -> np.ndarray:
    import cv2
    h, w = rgb.shape[:2]
    if w == width:
        return rgb
    size = (width, max(1, round(h * width / w)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def edge_map(gray: np.ndarray, top: float) -> np.ndarray:
    """Canny edges with thresholds from the image's own gradient magnitudes: high at the (1 − top) percentile, low at
    half of it (Gaussian σ 1 first), so the edge set does not follow the image's contrast."""
    import cv2
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    hi = max(float(np.percentile(np.hypot(gx, gy), 100.0 * (1.0 - top))), 1.0)
    return cv2.Canny(g, 0.5 * hi, hi, L2gradient=True) > 0


def _near(edges: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)


def structure_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SSIM's structure term per pixel (Gaussian window σ 1.5): the local correlation of two grayscale images."""
    import cv2
    x, y = a.astype(np.float32), b.astype(np.float32)

    def G(z):
        return cv2.GaussianBlur(z, (0, 0), 1.5)
    mx, my = G(x), G(y)
    sxx = np.maximum(G(x * x) - mx * mx, 0)
    syy = np.maximum(G(y * y) - my * my, 0)
    sxy = G(x * y) - mx * my
    return (sxy + C3) / (np.sqrt(sxx * syy) + C3)


def light_match(ga: np.ndarray, gb: np.ndarray) -> tuple:
    """(recall, precision) of light sources between two grayscale frames: the share of a's blobs (≥ LIGHT_MIN_PX
    pixels at ≥ LIGHT_HI) that touch b's pixels at ≥ LIGHT_LO dilated by LIGHT_TOL_PX, and the reverse. 1.0 when a
    side has no blob."""
    import cv2
    k = np.ones((2 * LIGHT_TOL_PX + 1, 2 * LIGHT_TOL_PX + 1), np.uint8)

    def found(g_from, g_in):
        n, lab, st, _ = cv2.connectedComponentsWithStats((g_from >= LIGHT_HI).astype(np.uint8), connectivity=8)
        keep = np.nonzero(st[1:, cv2.CC_STAT_AREA] >= LIGHT_MIN_PX)[0] + 1
        if not len(keep):
            return 1.0
        near = cv2.dilate((g_in >= LIGHT_LO).astype(np.uint8), k) > 0
        hit = np.unique(lab[near & (lab > 0)])
        return float(np.isin(keep, hit).mean())
    return found(ga, gb), found(gb, ga)


def is_actor(o: dict, classes=PIN_CLASSES) -> bool:
    """A dynamic actor (pinned by pin_actors, measured by the actor terms): its class is one of classes, or its
    category is an animal's."""
    return o.get('class') in classes or str(o.get('category') or '').startswith('animal')


def frame_drift(src: np.ndarray, out: np.ndarray, objects=None, scale: float = 1.0) -> dict:
    """How far a translated frame moved from its input (both RGB uint8 at the same size, the work width): structure,
    edge recall and precision, each labelled object's box structure (objects' boxes × scale), the drift. Reported
    beside it, not in it: the dynamic actors' (is_actor) worst box structure and their boxes' edge recall and
    precision, pooled over every actor box of the frame (None without an actor box or without strong edges there)."""
    if src.shape != out.shape:
        raise TranslateError(f'input {src.shape} and output {out.shape} differ in size')
    ga, gb = _gray(src), _gray(out)
    s = structure_map(ga, gb)
    ea, eb = edge_map(ga, EDGE_STRONG), edge_map(gb, EDGE_STRONG)
    da, db = _near(edge_map(ga, EDGE_WEAK)), _near(edge_map(gb, EDGE_WEAK))
    recall = float((db[ea] <= EDGE_TOL_PX).mean()) if ea.any() else 1.0
    precision = float((da[eb] <= EDGE_TOL_PX).mean()) if eb.any() else 1.0
    H, W = ga.shape
    boxes = []
    hits = np.zeros(4, np.int64)              # actor boxes: input edges, of them found; output edges, of them found
    from .realgap import THINGS
    for o in objects or ():
        if o.get('class') not in THINGS or not o.get('bbox'):
            continue
        x0, y0, x1, y1 = (float(v) * scale for v in o['bbox'])
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(math.ceil(x1)), W), min(int(math.ceil(y1)), H)
        if y1 - y0 < MIN_BOX_PX or x1 - x0 < 2:
            continue
        actor = is_actor(o)
        boxes.append({'id': o.get('id'), 'class': o['class'], 'actor': actor,
                      'structure': float(s[y0:y1, x0:x1].mean())})
        if actor:
            a_, b_ = ea[y0:y1, x0:x1], eb[y0:y1, x0:x1]
            hits += (int(a_.sum()), int((db[y0:y1, x0:x1][a_] <= EDGE_TOL_PX).sum()), int(b_.sum()),
                     int((da[y0:y1, x0:x1][b_] <= EDGE_TOL_PX).sum()))
    worst = min((b['structure'] for b in boxes), default=None)
    worst_actor = min((b['structure'] for b in boxes if b['actor']), default=None)
    st = float(s.mean())
    l_rec = l_prec = None
    if float(np.median(ga)) < NIGHT_GRAY:                            # a night frame: its lights count too
        l_rec, l_prec = light_match(ga, gb)
    drift = 1.0 - min(st, recall, precision, worst if worst is not None else 1.0,
                      *(v for v in (l_rec, l_prec) if v is not None))
    return {'structure': st, 'edge_recall': recall, 'edge_precision': precision, 'worst_box': worst,
            'light_recall': l_rec, 'light_precision': l_prec, 'worst_actor': worst_actor,
            'actor_edge_recall': float(hits[1] / hits[0]) if hits[0] else None,
            'actor_edge_precision': float(hits[3] / hits[2]) if hits[2] else None,
            'n_boxes': len(boxes), 'boxes': boxes, 'drift': max(drift, 0.0)}


MEASURES = ('drift', 'structure', 'edge_recall', 'edge_precision', 'worst_box', 'light_recall', 'light_precision',
            'worst_actor', 'actor_edge_recall', 'actor_edge_precision')


def drift_summary(rows: list, threshold: float = DRIFT_THRESHOLD) -> dict:
    """Median, 90th percentile and maximum of each measure over frames, and the frames over the threshold."""
    out = {'frames': len(rows), 'threshold': threshold,
           'flagged': sum(1 for r in rows if r.get('drift') is not None and r['drift'] > threshold)}
    for k in MEASURES:
        v = np.asarray([r[k] for r in rows if r.get(k) is not None], float)
        out[k] = ({'median': float(np.median(v)), 'p10': float(np.percentile(v, 10)),
                   'p90': float(np.percentile(v, 90)), 'min': float(v.min()), 'max': float(v.max()), 'n': len(v)}
                  if len(v) else {'n': 0})
    return out


def semantic_ids(sem: np.ndarray) -> np.ndarray:
    """CARLA semantic tags (H × W uint8) → GROUPS indices."""
    lut = np.full(256, len(GROUPS) - 1, np.uint8)
    for i, (_name, tags) in enumerate(GROUPS):
        lut[list(tags)] = i
    return lut[sem]


def one_hot(sem: np.ndarray | None, shape: tuple) -> np.ndarray:
    """K × H × W float32 one-hot of the semantic groups; all zero when there is no label ("unknown")."""
    K = len(GROUPS)
    if sem is None:
        return np.zeros((K,) + tuple(shape), np.float32)
    g = semantic_ids(sem)
    return (np.arange(K, dtype=np.uint8)[:, None, None] == g[None]).astype(np.float32)


def inverse_range(rng: np.ndarray | None, shape: tuple, d_min: float = DEPTH_MIN_M) -> np.ndarray:
    """The depth channel: H × W float32 min(1, d_min / range) from a range PNG (uint16 = round(range_m · 256), 0 =
    nothing within 256 m, core.groundtruth), so 1 at d_min metres or nearer, 0.1 at 10 d_min, 0 for the sky; all zero
    when there is no range image (a real image: "unknown", together with its all-zero label channels)."""
    if rng is None:
        return np.zeros(tuple(shape), np.float32)
    from .groundtruth import RANGE_SCALE
    r = rng.astype(np.float32) / RANGE_SCALE
    return np.where(r > 0, np.minimum(1.0, d_min / np.maximum(r, 1e-6)), 0.0).astype(np.float32)


# ================================================================================================ foreground pinning
def fit_colour(x: np.ndarray, y: np.ndarray) -> tuple:
    """A robust per-channel affine colour map y_c ≈ g_c · x_c + o_c (x, y: N × 3 colours in [0, 1]), returned as M
    (4 × 3: diag(g) over the offsets o) for apply_colour. Per channel: least squares with a ridge pulling the gain (not
    the offset) to 1 — PIN_RIDGE per pixel, so a channel the object's colours barely spread in keeps its gain —
    reweighted three times with Huber weights (PIN_HUBER × 1.4826 × the residuals' median), the gain clamped at 0 (no
    channel inverted). Up to PIN_SAMPLE pixels (an even stride). Fewer than PIN_MIN_PX pixels: gain 1 and the
    per-channel median offset ('offset'); the same when the fitted map's luminance slope (the regression of the mapped
    luma on the input's, where the input's luma spreads over more than two grey levels) is below PIN_MIN_GAIN — a map
    that would flatten the object ('fallback'). A full 3 × 3 colour matrix was measured too: its cross-channel terms
    fitted to re-textured objects changed their grey structure, and pinned drift was higher (docs/translation.md).
    Returns (M, 'affine' | 'offset' | 'fallback')."""
    n = len(x)
    M = np.vstack([np.eye(3), np.zeros((1, 3))])

    def offset(kind):
        M[:3] = np.eye(3)
        M[3] = np.median(y - x, 0) if n else 0.0
        return M, kind
    if n < PIN_MIN_PX:
        return offset('offset')
    if n > PIN_SAMPLE:
        idx = np.linspace(0, n - 1, PIN_SAMPLE).astype(np.int64)
        x, y = x[idx], y[idx]
    x, y = x.astype(np.float64), y.astype(np.float64)
    for c in range(3):
        X = np.stack([x[:, c], np.ones(len(x))], 1)
        w = np.ones(len(x))
        for _ in range(4):
            lam = np.diag([PIN_RIDGE, 0.0]) * w.sum()
            th = np.linalg.solve(X.T @ (X * w[:, None]) + lam, X.T @ (y[:, c] * w) + lam @ np.array([1.0, 0.0]))
            r = np.abs(y[:, c] - X @ th)
            sc = 1.4826 * float(np.median(r))
            if sc <= 1e-9:
                break
            w = np.minimum(1.0, PIN_HUBER * sc / np.maximum(r, 1e-12))
        if th[0] < 0:                             # no inverted channel: the best offset at gain 0
            th = np.array([0.0, float(np.sum(w * y[:, c]) / np.sum(w))])
        M[c, c], M[3, c] = th
    gi = x @ LUMA
    gm = (x @ M[:3] + M[3]) @ LUMA
    v = float(gi.var())
    if v > (2 / 255) ** 2 and float(((gi - gi.mean()) * (gm - gm.mean())).mean()) / v < PIN_MIN_GAIN:
        return offset('fallback')
    return M, 'affine'


def apply_colour(rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    """uint8 N × 3 (or H × W × 3) colours through the map M (fit_colour), rounded and clipped."""
    x = rgb.reshape(-1, 3).astype(np.float64) / 255.0
    y = x @ M[:3] + M[3]
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8).reshape(rgb.shape)


def pin_actors(src: np.ndarray, out: np.ndarray, instance: np.ndarray, objects, feather: int = PIN_FEATHER_PX,
               classes=PIN_CLASSES) -> tuple:
    """Foreground pinning: inside each dynamic actor's instance mask (is_actor; instance: H × W ids, an object's
    'id' at its pixels, the run's instance PNG) the translation is replaced by a colour map of the input fitted from
    the input's to the translation's pixels there (fit_colour) — the tone and colour follow the translator, every
    pixel's position and every edge inside the mask is the input's (a pixel is a function of the same pixel of the
    input alone, one function per object). In a ring of `feather` px outside the masks, the translation is blended
    towards the nearest mask's map of the input (weight 1 − d / (feather + 1) at distance d px); elsewhere the
    translation is untouched. src, out: H × W × 3 uint8 at the instance image's size.
    Returns (pinned frame, {'objects', 'pixels', 'fits': [{'id', 'class', 'pixels', 'kind', 'rms', 'map' (M)}]}),
    rms the fit's residual in 8-bit levels: how much of the translator's change inside the mask was not a colour map
    (texture, structure), which pinning discards."""
    import cv2
    if src.shape != out.shape or src.shape[:2] != instance.shape[:2]:
        raise TranslateError(f'pinning: input {src.shape}, translation {out.shape} and instance image '
                             f'{instance.shape} differ in size')
    res = out.copy()
    H, W = instance.shape[:2]
    best = np.full((H, W), np.inf, np.float32)          # distance to the nearest pinned mask so far (0: inside one;
    #                                                     masks never overlap: an instance image has one id a pixel)
    fits = []
    total = 0
    for o in objects or ():
        if not is_actor(o, classes) or o.get('id') is None:
            continue
        m = instance == int(o['id'])
        n = int(m.sum())
        if not n:
            continue
        x, y = src[m], out[m]
        M, kind = fit_colour(x.astype(np.float64) / 255.0, y.astype(np.float64) / 255.0)
        mapped = apply_colour(x, M)
        res[m] = mapped
        fits.append({'id': o['id'], 'class': o.get('class'), 'pixels': n, 'kind': kind,
                     'rms': float(np.sqrt(np.mean((mapped.astype(np.float32) - y) ** 2))), 'map': M})
        total += n
        best[m] = 0.0
        if feather <= 0:
            continue
        ys, xs = np.nonzero(m)
        y0, y1 = max(int(ys.min()) - feather - 1, 0), min(int(ys.max()) + feather + 2, H)
        x0, x1 = max(int(xs.min()) - feather - 1, 0), min(int(xs.max()) + feather + 2, W)
        d = cv2.distanceTransform((~m[y0:y1, x0:x1]).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ring = (d > 0) & (d < feather + 1) & (d < best[y0:y1, x0:x1])
        if not ring.any():
            continue
        best[y0:y1, x0:x1][ring] = d[ring]
        a = (1.0 - d[ring] / (feather + 1.0))[:, None]
        win_src, win_out = src[y0:y1, x0:x1], out[y0:y1, x0:x1]
        blend = a * apply_colour(win_src[ring], M).astype(np.float32) + (1.0 - a) * win_out[ring]
        res[y0:y1, x0:x1][ring] = np.clip(np.round(blend), 0, 255).astype(np.uint8)
    return res, {'objects': len(fits), 'pixels': total, 'fits': fits}


# ================================================================================================ data
def load_range(path: str) -> np.ndarray:
    import cv2
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 2:
        raise TranslateError(f'cannot read range image {path} (a one-channel uint16 PNG)')
    return im


def load_instance(path: str) -> np.ndarray:
    import cv2
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 2:
        raise TranslateError(f'cannot read instance image {path} (a one-channel uint16 PNG)')
    return im


def load_image(path: str) -> np.ndarray:
    from .realgap import load_rgb
    return load_rgb(path)


def load_label(path: str) -> np.ndarray:
    import cv2
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise TranslateError(f'cannot read label {path}')
    return im if im.ndim == 2 else im[..., 2]          # an RGB-encoded label: CARLA's tag is the red channel


def _parse(src) -> dict:
    """'KIND:PATH?k=v&…' or a run path (falcon) → a source dict."""
    if isinstance(src, dict):
        return dict(src)
    text = str(src)
    kind, sep, rest = text.partition(':')
    if not sep or kind not in REAL_KINDS + ('falcon',):
        kind, rest = 'falcon', text
    path, _, query = rest.partition('?')
    d = {'kind': kind, 'path': os.path.abspath(os.path.expanduser(path))}
    for kv in filter(None, query.split('&')):
        k, eq, v = kv.partition('=')
        if not eq:
            raise TranslateError(f'source option {kv!r} is not key=value')
        try:
            d[k] = int(v)
        except ValueError:
            d[k] = v
    return d


def camera_of(run, camera=None) -> str:
    from .exporters import is_pinhole
    cams = [c for c in run.cams if is_pinhole(run, c)]
    cid = camera or (cams[0] if cams else None)
    if cid not in run.cams:
        raise TranslateError(f'{run.dir}: no pinhole camera {cid!r}')
    return cid


def sim_frames(src, camera=None, every=1) -> list:
    """A Falcon run's frames of one camera: [{'image', 'semantic', 'range', 'objects' (paths or None), 'step'}]."""
    from .exporters import ExportError, Run
    s = _parse(src)
    try:
        run = Run(s['path'])
    except ExportError as e:
        raise TranslateError(str(e)) from None
    cid = camera_of(run, s.get('camera') or camera)
    out = []
    for rec in run.camera_steps(cid, int(s.get('every') or every or 1)):
        sem, obj, rng = run.own(rec, cid, 'semantic'), run.own(rec, cid, 'objects'), run.own(rec, cid, 'range')
        out.append({'image': run.p(run.file(rec, cid, 'rgb')), 'semantic': run.p(sem) if sem else None,
                    'range': run.p(rng) if rng else None, 'objects': run.p(obj) if obj else None, 'step': rec['step'],
                    'run': run.dir, 'camera': cid})
    if not out:
        raise TranslateError(f'{run.dir}: camera {cid} has no frames')
    return out


def real_frames(src) -> tuple:
    from .realgap import GapError, read_set
    s = _parse(src)
    if s['kind'] not in REAL_KINDS:
        raise TranslateError(f'a real set is KIND:PATH with KIND one of {", ".join(REAL_KINDS)}; got {src!r}')
    try:
        st = read_set(s['kind'], s['path'], labels=s.get('labels'), camera=s.get('camera'), limit=s.get('limit'),
                      seed=int(s.get('seed', 0)))
    except GapError as e:
        raise TranslateError(str(e)) from None
    return [it['image'] for it in st['items']], s, st.get('notes') or []


def terms_of(real_sources: list) -> list:
    return sorted({DATA_TERMS.get(_parse(s)['kind'], 'unknown terms') for s in real_sources})


def licence_note(terms: list) -> str:
    return ('The generator\'s code is Falcon\'s (GPL-3.0-or-later). Its weights were trained on real images under: '
            + '; '.join(terms) + '. The weights, and every image they translate, inherit those terms — '
            'a model trained on a non-commercial set may not be used commercially.')


# ================================================================================================ training
def normalise(settings: dict) -> dict:
    s = {**DEFAULTS, **{k: v for k, v in (settings or {}).items() if v is not None}}
    if s['method'] not in METHODS:
        raise TranslateError(f'method is one of {", ".join(METHODS)} (CycleGAN and diffusion are not implemented: '
                             f'docs/translation.md)')
    s['sim'] = [_parse(x) for x in s['sim']]
    s['real'] = [_parse(x) for x in s['real']]
    if not s['sim'] or any(x['kind'] != 'falcon' for x in s['sim']):
        raise TranslateError('sim needs at least one Falcon run')
    if not s['real'] or any(x['kind'] not in REAL_KINDS for x in s['real']):
        raise TranslateError(f'real needs at least one real set ({", ".join(REAL_KINDS)})')
    for k in ('iterations', 'batch', 'crop', 'work_width', 'ngf', 'n_blocks', 'ndf', 'num_patches', 'seed', 'probes',
              'max_images', 'every'):
        s[k] = int(s[k])
    if s['iterations'] < 1 or s['batch'] < 1 or s['crop'] < 16 or s['work_width'] < 32:
        raise TranslateError('iterations ≥ 1, batch ≥ 1, crop ≥ 16 and work_width ≥ 32')
    s['depth'] = bool(s['depth'])
    if s['depth'] and not s['condition']:
        raise TranslateError('the depth channel needs the label channels (--depth without --no-condition): a real '
                             'image\'s all-zero label is what tells its zero depth ("unknown") from the sky\'s')
    return s


def settings_key(s: dict, data_sha: str) -> str:
    """The key that guards a resume and a finished folder: the settings that change the result, and the data. depth
    off is left out, so folders made before the depth option keep their key."""
    from .provenance import sha256_json
    return sha256_json({'settings': {k: v for k, v in s.items() if k not in VOLATILE and not (k == 'depth' and not v)},
                        'data': data_sha})


def _mix(*xs) -> int:
    from .training import _mix as mix
    return mix(*xs)


class _Pool:
    """Images (and labels, and range images) at the work width, in memory, with seeded crops."""

    def __init__(self, items: list, width: int, labels: bool, depth: bool = False):
        self.images, self.labels, self.ranges = [], [], []
        for it in items:
            self.images.append(working(load_image(it['image']), width))
            if labels:
                if not it.get('semantic'):
                    raise TranslateError(f'{it["image"]}: no semantic label (the conditioned generator needs one; '
                                         f'record with the semantic label or use --no-condition)')
                self.labels.append(working(load_label(it['semantic']), width, nearest=True))
            if depth:
                if not it.get('range'):
                    raise TranslateError(f'{it["image"]}: no range image (the depth channel needs one; record with '
                                         f'the depth label or train without --depth)')
                self.ranges.append(working(load_range(it['range']), width, nearest=True))
        self.min_h = min(im.shape[0] for im in self.images)
        self.min_w = min(im.shape[1] for im in self.images)

    def crops(self, rng, n: int, ch: int, cw: int, flip: bool):
        xs, cs, ds = [], [], []
        for _ in range(n):
            k = int(rng.integers(len(self.images)))
            im = self.images[k]
            y0 = int(rng.integers(im.shape[0] - ch + 1))
            x0 = int(rng.integers(im.shape[1] - cw + 1))
            f = flip and bool(rng.integers(2))
            x = im[y0:y0 + ch, x0:x0 + cw]
            lab = self.labels[k][y0:y0 + ch, x0:x0 + cw] if self.labels else None
            rng_ = self.ranges[k][y0:y0 + ch, x0:x0 + cw] if self.ranges else None
            if f:
                x = x[:, ::-1]
                lab = lab[:, ::-1] if lab is not None else None
                rng_ = rng_[:, ::-1] if rng_ is not None else None
            xs.append(np.ascontiguousarray(x))
            cs.append(lab)
            ds.append(rng_)
        return xs, cs, ds


def to_tensor(ims: list, dev):
    import torch
    x = np.stack(ims).astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev)


def cond_tensor(labels: list, shape: tuple, dev, ranges: list | None = None, d_min: float | None = None):
    """B × K × H × W one-hot of the labels' semantic groups (as one_hot(), built on the device); a None label is all
    zero. With d_min (a depth-conditioned generator) one channel more: inverse_range() of the range images (a None
    range is all zero)."""
    import torch
    K = len(GROUPS)
    g = np.stack([semantic_ids(lab) if lab is not None else np.full(shape, K, np.uint8) for lab in labels])
    oh = torch.nn.functional.one_hot(torch.from_numpy(g.astype(np.int64)).to(dev), K + 1)[..., :K]
    c = oh.permute(0, 3, 1, 2).float()
    if d_min is not None:
        rs = ranges if ranges is not None else [None] * len(labels)
        d = np.stack([inverse_range(r, shape, d_min) for r in rs])[:, None]
        c = torch.cat([c, torch.from_numpy(d).to(dev)], 1)
    return c.contiguous()


def from_tensor(t) -> np.ndarray:
    x = t.detach().float().clamp(-1, 1).cpu().numpy().transpose(0, 2, 3, 1)
    return np.clip(np.round((x + 1.0) * 127.5), 0, 255).astype(np.uint8)


def _lr(it: int, total: int, base: float, decay_from: float) -> float:
    start = int(total * decay_from)
    if it < start:
        return base
    return base * max(0.0, 1.0 - (it - start) / max(1, total - start))


def _build(s: dict, dev):
    from .translate_net import PatchDiscriminator, PatchSampler, ResnetGenerator
    G = ResnetGenerator((len(GROUPS) + (1 if s['depth'] else 0)) if s['condition'] else 0, s['ngf'],
                        s['n_blocks']).to(dev)
    D = PatchDiscriminator(s['ndf']).to(dev)
    Fs = PatchSampler(G.tap_channels()).to(dev)
    return G, D, Fs


def _device(s: dict):
    from .training import _device as dv
    from .training import TrainingError
    try:
        return dv({'device': s['device']})
    except TrainingError as e:
        raise TranslateError(str(e)) from None


def _cond_of(pool: _Pool, i: int, cond: bool, d_min, dev):
    """The condition tensor of one whole pool image (None for an unconditioned generator)."""
    if not cond:
        return None
    return cond_tensor([pool.labels[i]], pool.images[i].shape[:2], dev,
                       [pool.ranges[i]] if pool.ranges else None, d_min)


def _probe(G, pool: _Pool, idx: list, cond: bool, dev, objects: list, d_min=None) -> dict:
    """Translate the probe frames whole (eval mode) and measure their drift. objects: per probe (objects, the
    camera's width their boxes are in)."""
    import torch
    rows = []
    G.eval()
    with torch.no_grad():
        for (objs, width), i in zip(objects, idx):
            im = pool.images[i]
            out = from_tensor(G(to_tensor([im], dev), _cond_of(pool, i, cond, d_min, dev)))[0]
            rows.append(frame_drift(im, out, objs, im.shape[1] / width))
    G.train()
    sm = drift_summary(rows)
    return {'drift_median': sm['drift'].get('median'), 'drift_max': sm['drift'].get('max'),
            'structure_median': sm['structure'].get('median'), 'edge_recall_median': sm['edge_recall'].get('median'),
            'edge_precision_median': sm['edge_precision'].get('median'), 'flagged': sm['flagged']}


def _read_objects(frame: dict) -> tuple:
    """A frame's objects and the camera width their boxes are in."""
    import cv2
    from .provenance import read_json
    im = cv2.imread(frame['image'], cv2.IMREAD_UNCHANGED)
    return (read_json(frame['objects']).get('objects', []) if frame.get('objects') else [],
            float(im.shape[1]) if im is not None else float(WORK_WIDTH))


def train(settings: dict, out_dir: str, emitter=None, cancel=None, argv=None) -> dict:
    """Train (or resume, or return the finished result of) a translator into out_dir. Returns translator.json's
    document. Raises TranslateError (bad input) or training.Cancelled (after a checkpoint and run.json status
    cancelled). Process-wide state (deterministic flags, random generators) is restored afterwards."""
    import random
    import torch
    saved = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, random.getstate(),
             np.random.get_state(), torch.get_rng_state())
    try:
        return _train(settings, out_dir, emitter, cancel, argv)
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved[2], saved[3]
        random.setstate(saved[4])
        np.random.set_state(saved[5])
        torch.set_rng_state(saved[6])


def _train(settings, out_dir, emitter, cancel, argv) -> dict:
    import torch
    from . import provenance as PV
    from .exporters import Run
    from .training import Cancelled, _Quiet, _read_metrics, _save, _versions, _write_metrics
    em = emitter or _Quiet()
    cancel = cancel or (lambda: False)
    s = normalise(settings)
    os.makedirs(out_dir, exist_ok=True)
    tj = os.path.join(out_dir, 'translator.json')
    # ---- data
    sim, inputs, run_names = [], {}, []
    for src in s['sim']:
        fr = sim_frames(src, s['camera'], s['every'])
        sim += fr
        inputs[f'sim:{src["path"]}/run.json'] = os.path.join(src['path'], 'run.json')
        run_names.append(os.path.basename(os.path.normpath(src['path'])))
    real, notes = [], []
    for src in s['real']:
        ims, _, nt = real_frames(src)
        real += ims
        notes += nt
    rng0 = np.random.default_rng(_mix(s['seed'], 7))
    if len(sim) > s['max_images']:
        sim = [sim[i] for i in np.sort(rng0.choice(len(sim), s['max_images'], replace=False))]
    if len(real) > s['max_images']:
        real = [real[i] for i in np.sort(rng0.choice(len(real), s['max_images'], replace=False))]
    data_doc = {'sim': [{'image': f['image'], 'semantic': f['semantic'], **({'range': f['range']} if s['depth'] else
                                                                            {})} for f in sim], 'real': real}
    data_sha = PV.sha256_json(data_doc)
    key = settings_key(s, data_sha)
    if os.path.isfile(tj):
        doc = PV.read_json(tj)
        if doc.get('key') == key and doc.get('status') == 'ok':
            em.log('this translator has finished already: nothing to do')
            return doc
    ck = os.path.join(out_dir, 'checkpoints', 'last.pt')
    if os.path.isfile(ck):
        state = torch.load(ck, map_location='cpu', weights_only=False)
        if state.get('key') != key:
            raise TranslateError(f'{out_dir} holds a translator trained with other settings or data; use another '
                                 f'folder')
    else:
        state = None
        if os.path.isfile(tj) or os.path.isfile(os.path.join(out_dir, 'generator.pt')):
            raise TranslateError(f'{out_dir} holds another translator; use another folder')
    terms = terms_of(s['real'])
    PV.write_json(os.path.join(out_dir, 'data.json'), {'schema': 'falcon.translator_data/1', **data_doc,
                                                       'sha256': data_sha, 'terms': terms})
    # ---- set-up
    dev = _device(s)
    torch.manual_seed(s['seed'])
    np.random.seed(s['seed'] % 2 ** 32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(s['gpu_memory_fraction']), dev)
        torch.cuda.reset_peak_memory_stats(dev)
    amp = bool(s['amp']) and dev.type == 'cuda'
    em.progress(0.0, f'loading {len(sim)} simulated and {len(real)} real images', force=True)
    d_min = DEPTH_MIN_M if s['depth'] else None
    sp = _Pool(sim, s['work_width'], s['condition'], s['depth'])
    rp = _Pool([{'image': p} for p in real], s['work_width'], False)
    ch = min(s['crop'], sp.min_h, rp.min_h) // 4 * 4
    cw = min(s['crop'], sp.min_w, rp.min_w) // 4 * 4
    if ch < 16 or cw < 16:
        raise TranslateError(f'the images are too small for training crops ({ch} × {cw} px at the work width)')
    probe_idx = sorted({int(i) for i in np.linspace(0, len(sim) - 1, min(s['probes'], len(sim)))})
    probe_obj = [_read_objects(sim[i]) for i in probe_idx]
    G, D, Fs = _build(s, dev)
    optG = torch.optim.Adam(list(G.parameters()) + list(Fs.parameters()), s['lr'], betas=(s['beta1'], s['beta2']))
    optD = torch.optim.Adam(D.parameters(), s['lr'], betas=(s['beta1'], s['beta2']))
    it0 = 0
    from .translate_net import lsgan, nce_loss
    mpath = os.path.join(out_dir, 'metrics.jsonl')
    rows = _read_metrics(mpath)
    run = PV.new_run('translate-train', argv or ['translate', 'train'], s['seed'], {k: v for k, v in s.items()},
                     inputs)
    run.update(versions=_versions(), device={'type': dev.type, 'gpu': torch.cuda.get_device_name(dev)
                                             if dev.type == 'cuda' else None, 'amp': 'bfloat16' if amp else None},
               key=key, notes=list(notes))
    if state is not None:
        G.load_state_dict(state['G'])
        D.load_state_dict(state['D'])
        Fs.load_state_dict(state['F'])
        optG.load_state_dict(state['optG'])
        optD.load_state_dict(state['optD'])
        it0 = int(state['it'])
        rows = [r for r in rows if r.get('it', 0) <= it0]
        run['started'] = state.get('started', run['started'])
        run['notes'].append(f'resumed at iteration {it0}')
        em.log(f'resuming at iteration {it0} of {s["iterations"]}')
    seconds = float(state.get('seconds', 0.0)) if state else 0.0
    gen = torch.Generator()                       # the patch draws (CPU), re-seeded every iteration

    def checkpoint(it):
        _save(ck, {'key': key, 'it': it, 'G': G.state_dict(), 'D': D.state_dict(), 'F': Fs.state_dict(),
                   'optG': optG.state_dict(), 'optD': optD.state_dict(), 'started': run['started'],
                   'seconds': seconds})
        _write_metrics(mpath, rows)

    total = s['iterations']
    acc, n_acc, t_log = {}, 0, time.monotonic()
    imgs_log = 0
    for it in range(it0, total):
        if cancel():
            checkpoint(it)
            run.update(status='cancelled', exit_code=3, finished=PV.utc_now())
            PV.write_json(os.path.join(out_dir, 'run.json'), run)
            raise Cancelled()
        t_it = time.monotonic()
        rng = np.random.default_rng(_mix(s['seed'], 11, it))
        xs, cs, ds = sp.crops(rng, s['batch'], ch, cw, s['hflip'])
        ys, _, _ = rp.crops(rng, s['batch'], ch, cw, s['hflip'])
        gen.manual_seed(_mix(s['seed'], 13, it))
        torch.manual_seed(_mix(s['seed'], 17, it))
        lr = _lr(it, total, s['lr'], s['decay_from'])
        for o in (optG, optD):
            for g in o.param_groups:
                g['lr'] = lr
        x, y = to_tensor(xs, dev), to_tensor(ys, dev)
        c = cond_tensor(cs, (ch, cw), dev, ds, d_min) if s['condition'] else None
        with torch.autocast(dev.type, torch.bfloat16, enabled=amp):
            fake = G(x, c)
            idt = G(y, None) if s['nce_idt'] else None
        # D: real KITTI patches against translated ones
        for p in D.parameters():
            p.requires_grad_(True)
        with torch.autocast(dev.type, torch.bfloat16, enabled=amp):
            loss_d = 0.5 * (lsgan(D(fake.detach()).float(), False) + lsgan(D(y).float(), True))
        optD.zero_grad(set_to_none=True)
        loss_d.backward()
        optD.step()
        # G (and the projection heads)
        for p in D.parameters():
            p.requires_grad_(False)
        with torch.autocast(dev.type, torch.bfloat16, enabled=amp):
            loss_gan = lsgan(D(fake).float(), True) * s['lambda_gan']
            nce = nce_loss(G, Fs, x, c, fake, s['num_patches'], s['tau'], gen).float()
            nce_y = (nce_loss(G, Fs, y, None, idt, s['num_patches'], s['tau'], gen).float()
                     if s['nce_idt'] else None)
            loss_nce = (nce + nce_y) * 0.5 if nce_y is not None else nce
            loss_g = loss_gan + s['lambda_nce'] * loss_nce
        optG.zero_grad(set_to_none=True)
        loss_g.backward()
        optG.step()
        if not torch.isfinite(loss_g) or not torch.isfinite(loss_d):
            raise TranslateError(f'the losses diverged at iteration {it} (G {float(loss_g)}, D {float(loss_d)})')
        for k, v in (('loss_G', loss_g), ('loss_D', loss_d), ('loss_G_GAN', loss_gan), ('nce', nce),
                     ('nce_idt', nce_y)):
            if v is not None:
                acc[k] = acc.get(k, 0.0) + float(v.detach())
        n_acc += 1
        imgs_log += s['batch']
        seconds += time.monotonic() - t_it
        done = it + 1
        if done % s['log_every'] == 0 or done == total:
            dt = max(time.monotonic() - t_log, 1e-9)
            row = {'it': done, 'lr': lr, **{k: v / n_acc for k, v in acc.items()},
                   'images_per_s': imgs_log / dt}
            if done % s['probe_every'] == 0 or done == total:
                row['probe'] = _probe(G, sp, probe_idx, s['condition'], dev, probe_obj, d_min)
            rows.append(row)
            em.metric('loss_G', row['loss_G'], step=done)
            em.metric('images_per_s', round(row['images_per_s'], 1), 'images/s', step=done)
            if 'probe' in row:
                em.metric('probe_drift_median', row['probe']['drift_median'], step=done)
            acc, n_acc, t_log, imgs_log = {}, 0, time.monotonic(), 0
        if done % s['ckpt_every'] == 0 and done < total:
            checkpoint(done)
        em.progress(done / total, f'iteration {done}/{total}')
    # ---- outputs
    _write_metrics(mpath, rows)
    gp = os.path.join(out_dir, 'generator.pt')
    _save(gp, {'state_dict': {k: v.detach().cpu() for k, v in G.state_dict().items()},
               'arch': {'cond_ch': G.cond_ch, 'ngf': G.ngf, 'n_blocks': G.n_blocks, 'depth_min_m': d_min}})
    prev = _previews(G, sp, probe_idx[:4], s['condition'], dev, os.path.join(out_dir, 'previews'), d_min)
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == 'cuda' else None
    speeds = [r['images_per_s'] for r in rows if r.get('images_per_s')]
    doc = {'schema': SCHEMA, 'status': 'ok', 'key': key, 'id': os.path.basename(os.path.normpath(out_dir)),
           'method': s['method'], 'paper': 'Park, Efros, Zhang, Zhu: Contrastive Learning for Unpaired Image-to-Image '
                                            'Translation (ECCV 2020)',
           'arch': {'generator': 'ResNet (Falcon: residual in the input\'s logit, nearest upsampling)',
                    'cond': ([g for g, _ in GROUPS] + (['inverse_range'] if s['depth'] else [])
                             if s['condition'] else None), 'depth_min_m': d_min, 'ngf': s['ngf'],
                    'n_blocks': s['n_blocks'], 'discriminator': f'PatchGAN 70 × 70, ndf {s["ndf"]}'},
           'work_width': s['work_width'], 'crop': [ch, cw], 'iterations': total,
           'weights': {'file': 'generator.pt', 'sha256': PV.sha256_file(gp), 'bytes': os.path.getsize(gp)},
           'data': {'sim_runs': run_names, 'sim_images': len(sim), 'real_images': len(real),
                    'real': [{'kind': x['kind'], 'path': x['path']} for x in s['real']], 'sha256': data_sha},
           'terms': terms, 'licence_note': licence_note(terms), 'settings': {k: v for k, v in s.items()},
           'cost': {'train_s': round(seconds, 1), 'images_per_s_median': float(np.median(speeds)) if speeds else None,
                    'peak_gpu_mib': peak, 'device': run['device']},
           'probe': rows[-1].get('probe') if rows else None, 'previews': prev, 'drift_threshold': DRIFT_THRESHOLD}
    PV.write_json(tj, doc)
    if not s['keep_checkpoint'] and os.path.isdir(os.path.dirname(ck)):
        shutil.rmtree(os.path.dirname(ck))
    run.update(status='ok', exit_code=0, finished=PV.utc_now(), timings={
        'train_s': round(seconds, 1), 'images_per_s_median': doc['cost']['images_per_s_median'],
        'peak_gpu_mib': peak}, summary={'translator': tj, 'probe': doc['probe']},
        outputs=PV.hash_tree(out_dir))
    PV.write_json(os.path.join(out_dir, 'run.json'), run)
    return doc


def _previews(G, pool, idx, cond, dev, out, d_min=None) -> list:
    """Before (top) and after (bottom) of a few probe frames at half the work width: JPEGs."""
    import cv2
    import torch
    os.makedirs(out, exist_ok=True)
    names = []
    G.eval()
    with torch.no_grad():
        for k, i in enumerate(idx):
            im = pool.images[i]
            o = from_tensor(G(to_tensor([im], dev), _cond_of(pool, i, cond, d_min, dev)))[0]
            half = lambda a: cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)  # noqa
            both = np.concatenate([half(im), half(o)], 0)
            n = f'preview{k}.jpg'
            cv2.imwrite(os.path.join(out, n), cv2.cvtColor(both, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            names.append(os.path.join('previews', n))
    G.train()
    return names


# ================================================================================================ applying a model
class Translator:
    """A trained translator, loaded once: __call__(rgb, semantic[, native, range]) translates one frame at its own
    resolution. Models of either schema load: a falcon.translator/1 model (before the depth option) has no depth
    channel."""

    def __init__(self, model_dir: str, device: str = 'auto'):
        import torch
        from . import provenance as PV
        from .translate_net import ResnetGenerator
        tj = os.path.join(model_dir, 'translator.json')
        if not os.path.isfile(tj):
            raise TranslateError(f'{model_dir} is not a finished translator (no translator.json)')
        self.doc = PV.read_json(tj)
        if self.doc.get('schema', SCHEMAS[0]) not in SCHEMAS:
            raise TranslateError(f'{tj}: schema {self.doc.get("schema")!r} is not one this Falcon reads '
                                 f'({", ".join(SCHEMAS)})')
        self.dir = os.path.abspath(model_dir)
        gp = os.path.join(model_dir, self.doc['weights']['file'])
        sha = PV.sha256_file(gp)
        if sha != self.doc['weights']['sha256']:
            raise TranslateError(f'{gp} does not match the SHA-256 in translator.json')
        self.sha256 = sha
        st = torch.load(gp, map_location='cpu', weights_only=True)
        a = st['arch']
        self.G = ResnetGenerator(a['cond_ch'], a['ngf'], a['n_blocks'])
        self.G.load_state_dict(st['state_dict'])
        self.dev = _device({'device': device})
        self.G = self.G.eval().to(self.dev)
        self.cond = bool(a['cond_ch'])
        self.depth = a.get('depth_min_m')          # the inverse-range channel's d_min, None without the channel
        self.width = int(self.doc['work_width'])
        self.torch = torch

    def net(self, rgb: np.ndarray, sem: np.ndarray | None, rng: np.ndarray | None = None) -> np.ndarray:
        if self.cond and sem is None:
            raise TranslateError('this translator is conditioned on the semantic label; the frame has none')
        if self.depth is not None and rng is None:
            raise TranslateError('this translator is conditioned on depth (the range image); the frame has none')
        with self.torch.no_grad():
            c = cond_tensor([sem], rgb.shape[:2], self.dev, [rng], self.depth) if self.cond else None
            return from_tensor(self.G(to_tensor([rgb], self.dev), c))[0]

    def __call__(self, rgb: np.ndarray, sem: np.ndarray | None = None, native: bool = False,
                 rng: np.ndarray | None = None) -> np.ndarray:
        """The translated frame at the input's size. At the work width the network's change is added to the input
        (upsampled, bicubic); native runs the network at the input's own resolution. rng: the range image (uint16
        PNG values), for a depth-conditioned translator."""
        import cv2
        if native or rgb.shape[1] == self.width:
            return self.net(rgb, sem, rng)
        small = working(rgb, self.width)
        ssem = working(sem, self.width, nearest=True) if sem is not None else None
        srng = working(rng, self.width, nearest=True) if rng is not None else None
        delta = self.net(small, ssem, srng).astype(np.float32) - small.astype(np.float32)
        up = cv2.resize(delta, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        return np.clip(np.round(rgb.astype(np.float32) + up), 0, 255).astype(np.uint8)


def _link(src: str, dst: str, mode: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == 'hard':
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _write_image(path: str, rgb: np.ndarray):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part' + os.path.splitext(path)[1]
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.lower().endswith(('.jpg', '.jpeg')) else []
    if not cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params):
        raise TranslateError(f'cannot write {path}')
    os.replace(tmp, path)


def translate_run(model_dir: str, src_dir: str, out_dir: str, cameras=None, threshold: float = DRIFT_THRESHOLD,
                  on_drift: str = 'raw', link: str = 'hard', native: bool = False, device: str = 'auto',
                  translator: Translator | None = None, emitter=None, cancel=None, argv=None,
                  pin: bool | None = PIN_DEFAULT, feather: int = PIN_FEATHER_PX) -> dict:
    """A translated copy of a recorded run (see the module docstring). pin: foreground pinning (pin_actors) with the
    frame's instance image and objects, a ring of feather px — True on every camera (one without instance images is
    refused), None on the cameras that have them, False nowhere. Returns its run.json. Raises TranslateError or
    training.Cancelled (the partial folder is removed)."""
    from . import provenance as PV
    from .exporters import ExportError, Run, is_pinhole
    from .training import Cancelled, _Quiet
    em = emitter or _Quiet()
    cancel = cancel or (lambda: False)
    if on_drift not in ('raw', 'flag'):
        raise TranslateError('on_drift is "raw" (keep the raw frame) or "flag" (keep the translation, listed)')
    try:
        run = Run(src_dir)
    except ExportError as e:
        raise TranslateError(str(e)) from None
    if os.path.abspath(out_dir) == run.dir:
        raise TranslateError('the translated run needs a folder of its own')
    tr = translator or Translator(model_dir, device)
    cams = list(cameras) if cameras else [c for c in run.cams if is_pinhole(run, c)]
    if not cams:
        raise TranslateError(f'{run.dir}: no pinhole camera to translate')
    for c in cams:
        if c not in run.cams:
            raise TranslateError(f'{run.dir}: no camera {c!r}')
        if tr.cond and not any(run.file(r, c, 'semantic') for r in run.frames):
            raise TranslateError(f'{run.dir}: camera {c} has no semantic label, which this translator is '
                                 f'conditioned on')
        if tr.depth is not None and not any(run.file(r, c, 'range') for r in run.frames):
            raise TranslateError(f'{run.dir}: camera {c} has no range image, which this translator is conditioned on')
        if pin and not any(run.file(r, c, 'instance') for r in run.frames):
            raise TranslateError(f'{run.dir}: camera {c} has no instance image, which pinning needs (record with the '
                                 f'instance label, or translate without pinning)')
    pin_cams = (set() if pin is False else
                {c for c in cams if any(run.file(r, c, 'instance') for r in run.frames)})
    for c in cams:
        if pin is None and c not in pin_cams:
            em.log(f'camera {c} has no instance image: translated without pinning its actors', 'warning')
    settings = {'model': tr.dir, 'model_sha256': tr.sha256, 'cameras': cams, 'threshold': threshold,
                'on_drift': on_drift, 'native': bool(native), 'work_width': tr.width}
    if pin_cams:                                  # (absent when off: the key of a run made before pinning holds)
        settings['pin_actors'] = {'classes': list(PIN_CLASSES), 'feather_px': int(feather),
                                  'cameras': sorted(pin_cams)}
    key = PV.sha256_json({'source': PV.sha256_file(run.p('run.json')), **settings})
    rj = os.path.join(out_dir, 'run.json')
    if os.path.isfile(rj):
        old = PV.read_json(rj)
        if (old.get('translation') or {}).get('key') == key and old.get('status') == 'ok':
            em.log(f'{out_dir} is this translation already: nothing to do')
            return old
        raise TranslateError(f'{out_dir} exists and is not this translation; choose another folder')
    if os.path.exists(out_dir):
        if not os.path.isfile(os.path.join(out_dir, '.translation-partial')):
            raise TranslateError(f'{out_dir} exists; choose another folder')
        shutil.rmtree(out_dir)                    # an interrupted translation of ours: start again
    os.makedirs(out_dir)
    open(os.path.join(out_dir, '.translation-partial'), 'w').close()
    t0 = time.monotonic()
    try:
        # which rgb files are translated, and with which label and objects
        todo = {}
        for rec in run.frames:
            for c in cams:
                rel = run.own(rec, c, 'rgb')
                if rel and rel not in todo:
                    todo[rel] = (c, rec)
        skip = set(todo) | {'run.json', 'dataset.json', 'translation.jsonl', '.translation-partial'}
        for d, dirs, files in os.walk(run.dir):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(d, f)
                rel = os.path.relpath(p, run.dir)
                if rel not in skip:      # the run's own records are copied; per-frame files are hard links
                    _link(p, os.path.join(out_dir, rel), link if d != run.dir else 'copy')
        rows = []
        width_of = {c: int(run.cams[c].get('width') or 0) for c in cams}
        n = len(todo)
        for k, (rel, (c, rec)) in enumerate(sorted(todo.items())):
            if cancel():
                raise Cancelled()
            rgb = load_image(run.p(rel))
            srel, orel = run.own(rec, c, 'semantic'), run.own(rec, c, 'objects')
            rrel, irel = run.own(rec, c, 'range'), run.own(rec, c, 'instance')
            sem = load_label(run.p(srel)) if srel else None
            reason = ('no semantic label' if tr.cond and sem is None else
                      'no range image' if tr.depth is not None and rrel is None else
                      'no instance image or objects (pinning needs both)' if c in pin_cams and (
                          irel is None or orel is None) else None)
            if reason:                                 # a frame without what the translation needs: kept raw, said so
                _link(run.p(rel), os.path.join(out_dir, rel), link)
                rows.append({'camera': c, 'step': rec['step'], 'file': rel, 'flagged': False, 'used': 'raw',
                             'reason': reason, 'drift': None})
                continue
            out = tr(rgb, sem, native, load_range(run.p(rrel)) if tr.depth is not None else None)
            objs = PV.read_json(run.p(orel)).get('objects', []) if orel else []
            pinned = None
            if c in pin_cams:
                out, pinned = pin_actors(rgb, out, load_instance(run.p(irel)), objs, feather)
            a, b = working(rgb, WORK_WIDTH), working(out, WORK_WIDTH)
            m = frame_drift(a, b, objs, WORK_WIDTH / float(width_of[c] or rgb.shape[1]))
            flagged = m['drift'] > threshold
            used = 'raw' if flagged and on_drift == 'raw' else 'translated'
            if used == 'raw':
                _link(run.p(rel), os.path.join(out_dir, rel), link)
            else:
                _write_image(os.path.join(out_dir, rel), out)
            rows.append({'camera': c, 'step': rec['step'], 'file': rel, 'flagged': flagged, 'used': used,
                         **{kk: m[kk] for kk in MEASURES + ('n_boxes',)},
                         **({'pinned': pinned['objects'], 'pin_rms_max': max((f['rms'] for f in pinned['fits']),
                                                                             default=None)} if pinned else {})})
            em.progress((k + 1) / n, f'{k + 1}/{n} frames')
        with open(os.path.join(out_dir, 'translation.jsonl'), 'w', encoding='utf-8') as fh:
            for r in rows:
                fh.write(json.dumps(r, separators=(',', ':')) + '\n')
        summ = drift_summary(rows, threshold)
        summ['refused'] = sum(1 for r in rows if r['used'] == 'raw' and r.get('reason') is None)
        summ['unlabelled'] = sum(1 for r in rows if r.get('reason') == 'no semantic label')
        summ['missing'] = {k: sum(1 for r in rows if r.get('reason') == k)
                           for k in sorted({r['reason'] for r in rows if r.get('reason')})}
        if pin_cams:
            pr = [r['pin_rms_max'] for r in rows if r.get('pin_rms_max') is not None]
            summ['pinned'] = {'objects': sum(r.get('pinned') or 0 for r in rows),
                              'frames': sum(1 for r in rows if r.get('pinned')),
                              'rms_max_median': float(np.median(pr)) if pr else None}
        ds = dict(run.ds)
        ds['name'] = os.path.basename(os.path.normpath(out_dir))
        ds['translation'] = {'source': run.ds.get('name') or run.name, 'model': tr.doc.get('id'),
                             'cameras': cams, 'terms': tr.doc.get('terms')}
        PV.write_json(os.path.join(out_dir, 'dataset.json'), ds)
        doc = dict(run.run)
        doc.update(name=os.path.basename(os.path.normpath(out_dir)), worker='translate',
                   argv=list(argv or ['translate', 'apply']), status='ok', exit_code=0, error=None,
                   notes=list(run.run.get('notes') or ()) + [
                       f'translated from {run.dir} by {tr.doc.get("id")} ({tr.doc.get("method")}); every label is '
                       f'the source\'s; started/finished are the source drive\'s times'],
                   translation={'schema': RUN_SCHEMA, 'key': key, 'source': run.dir, 'source_name': run.name,
                                'source_run_json_sha256': PV.sha256_file(run.p('run.json')),
                                'model': tr.dir, 'model_id': tr.doc.get('id'), 'method': tr.doc.get('method'),
                                'weights_sha256': tr.sha256, 'settings': settings, 'drift': summ,
                                'frames': 'translation.jsonl', 'terms': tr.doc.get('terms'),
                                'licence_note': tr.doc.get('licence_note'), 'translated_at': PV.utc_now(),
                                'seconds': round(time.monotonic() - t0, 2),
                                'link': link, 'untranslated_cameras': [c for c in run.cams if c not in cams]})
        os.remove(os.path.join(out_dir, '.translation-partial'))
        doc['outputs'] = PV.hash_tree(out_dir)
        doc['translation']['inputs'] = PV.block({'source/run.json': run.p('run.json'),
                                                 'model/translator.json': os.path.join(tr.dir, 'translator.json'),
                                                 'model/generator.pt': os.path.join(tr.dir, 'generator.pt')},
                                                None, settings)['inputs']
        PV.write_json(rj, doc)
        return doc
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


# ================================================================================================ checking
def _rows(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _miou(pred: np.ndarray, label: np.ndarray) -> float | None:
    ious = []
    valid = pred != 255
    for c in np.unique(pred[valid]):
        p, g = (pred == c) & valid, (label == c) & valid
        u = (p | g).sum()
        if u:
            ious.append((p & g).sum() / u)
    return float(np.mean(ious)) if ious else None


def check_runs(translated: list, real: str | None = None, features: bool = False, workspace: str = '',
               images: int = 240, seed: int = 0, segmenter: dict | None = None, emitter=None) -> dict:
    """Recompute every translated frame's drift against its source; with a real set, the gap (core.realgap) of the
    sources and of the translations against it, the same frames on both sides; with a segmenter registration, its
    pixel agreement and mIoU against the labels on the raw and the translated frames."""
    from . import provenance as PV
    from . import realgap as RG
    from .exporters import Run
    from .training import _Quiet
    em = emitter or _Quiet()
    runs, sources, rows = [], [], []
    seg = None
    if segmenter:
        from .models import Runner
        seg = Runner(segmenter, workspace)
    segs = []
    for t in translated:
        rd = PV.read_json(os.path.join(t, 'run.json'))
        info = rd.get('translation')
        if not info:
            raise TranslateError(f'{t} is not a translated run (no "translation" in run.json)')
        src = info['source']
        if not os.path.isfile(os.path.join(src, 'run.json')):
            raise TranslateError(f'{t}: its source run {src} is gone')
        if PV.sha256_file(os.path.join(src, 'run.json')) != info['source_run_json_sha256']:
            raise TranslateError(f'{t}: its source run {src} changed since the translation')
        runs.append(os.path.abspath(t))
        sources.append(src)
        rs, rt = Run(src), Run(t)
        used = {r['file']: r.get('used') for r in _rows(os.path.join(t, info.get('frames') or 'translation.jsonl'))}
        for c in info['settings']['cameras']:
            w = float(rs.cams[c].get('width') or 0)
            recs = rs.camera_steps(c)
            for k, rec in enumerate(recs):
                rel = rs.file(rec, c, 'rgb')
                a = load_image(rs.p(rel))
                b = load_image(rt.p(rel))
                orel = rs.own(rec, c, 'objects')
                objs = PV.read_json(rs.p(orel)).get('objects', []) if orel else []
                aw, bw = working(a), working(b)
                m = frame_drift(aw, bw, objs, WORK_WIDTH / (w or a.shape[1]))
                rows.append({'run': rt.name, 'camera': c, 'step': rec['step'], 'used': used.get(rel),
                             **{kk: m[kk] for kk in MEASURES}})
                if seg is not None and k % max(1, len(recs) // 20) == 0:
                    srel = rs.own(rec, c, 'semantic')
                    lab = load_label(rs.p(srel)) if srel else None
                    pa, pb = np.asarray(seg.predict(a)), np.asarray(seg.predict(b))
                    segs.append({'agreement': float((pa == pb).mean()),
                                 'miou_raw': _miou(pa, lab) if lab is not None else None,
                                 'miou_translated': _miou(pb, lab) if lab is not None else None})
        em.progress(len(runs) / (len(translated) + (2 if real else 0)), f'drift: {rt.name}')
    thr = PV.read_json(os.path.join(runs[0], 'run.json'))['translation']['settings']['threshold']
    doc = {'schema': CHECK_SCHEMA, 'runs': runs, 'sources': sources, 'drift': drift_summary(rows, thr),
           'kept_raw': sum(1 for r in rows if r['used'] == 'raw'), 'segmenter': None, 'gap': None,
           'method': {'drift': '1 − min(structure, edge recall, edge precision, worst object box structure; on a '
                               f'night frame (median gray < {NIGHT_GRAY}) also light recall and precision), at '
                               f'{WORK_WIDTH} px width (core.translate)', 'threshold': f'{thr} (heuristic)'}}
    if seg is not None:
        def mean(k):
            v = [s[k] for s in segs if s[k] is not None]
            return float(np.mean(v)) if v else None
        doc['segmenter'] = {'model': segmenter.get('id') or segmenter.get('name'), 'frames': len(segs),
                            'agreement': mean('agreement'), 'miou_raw': mean('miou_raw'),
                            'miou_translated': mean('miou_translated')}
    if real:
        kind, sep, path = real.partition(':')
        if not sep:
            raise TranslateError('the real set is KIND:PATH')
        rset = RG.read_set(kind, path, limit=images, seed=seed)
        share = max(1, images // len(runs))

        def side(paths):
            sets = [RG.read_set('falcon', p, limit=share, seed=seed + i) for i, p in enumerate(paths)]
            return sets
        em.progress(0.6, 'gap of the raw frames')
        before = RG.study(side(sources), rset, workspace, features, False, seed)
        em.progress(0.8, 'gap of the translated frames')
        after = RG.study(side(runs), rset, workspace, features, False, seed)
        rows_b = {r['stat']: r for r in before['gap']['appearance']}
        rows_a = {r['stat']: r for r in after['gap']['appearance']}
        doc['gap'] = {'real': {'kind': kind, 'path': os.path.abspath(path), 'images': rset['n_total'],
                               'used': len(rset['items'])},
                      'appearance': [{'stat': k, 'real_median': rows_b[k]['real_median'],
                                      'raw_median': rows_b[k]['sim_median'], 'raw_shift': rows_b[k]['shift'],
                                      'translated_median': rows_a[k]['sim_median'],
                                      'translated_shift': rows_a[k]['shift']}
                                     for k in rows_b if k in rows_a],
                      'night': _night_rows(before, after),
                      'features_raw': before['features'], 'features_translated': after['features'],
                      'objects': (before['gap'].get('objects') or {}).get('class_share_tvd')}
    em.progress(1.0, 'done', force=True)
    return doc


def _night_rows(before: dict, after: dict) -> list | None:
    """The night statistics' gaps (core.realgap.NIGHT_STATS) before and after translation, when both studies have
    them (a night real set)."""
    b, a = before['gap'].get('night'), after['gap'].get('night')
    if not b or not a:
        return None
    rb, ra = {r['stat']: r for r in b}, {r['stat']: r for r in a}
    return [{'stat': k, 'real_median': rb[k]['real_median'], 'raw_median': rb[k]['sim_median'],
             'raw_shift': rb[k]['shift'], 'translated_median': ra[k]['sim_median'], 'translated_shift': ra[k]['shift']}
            for k in rb if k in ra]


# ================================================================================================ the experiment
def _align(raw: list, translated: list) -> list:
    """The translated samples in the raw samples' order, matched by (source run, camera, step)."""
    from .provenance import read_json
    src_of = {}
    for s in translated:
        if s['unit'] not in src_of:
            info = read_json(os.path.join(s['unit'], 'run.json')).get('translation')
            if not info:
                raise TranslateError(f'{s["unit"]} is not a translated run')
            src_of[s['unit']] = os.path.abspath(info['source'])
    by = {}
    for s in translated:
        _, cam, step = s['id'].rsplit('#', 2)
        by[(src_of[s['unit']], cam, step)] = s
    out = []
    for s in raw:
        _, cam, step = s['id'].rsplit('#', 2)
        t = by.get((os.path.abspath(s['unit']), cam, step))
        if t is None:
            raise TranslateError(f'{s["id"]} has no translation among the translated runs: each raw drive needs its '
                                 f'translated run')
        if t['boxes'] != s['boxes'] or t['classes'] != s['classes']:
            raise TranslateError(f'{t["id"]}: its labels differ from its source\'s')
        out.append(t)
    return out


def arm_runs(seeds: list, shares: list) -> list:
    out = []
    for sd in seeds:
        out.append(('real only', 0.0, sd, f'real-only-seed{sd}'))
        for sh in shares:
            out.append(('raw', sh, sd, f'raw{round(sh * 100):03d}-seed{sd}'))
            out.append(('translated', sh, sd, f'translated{round(sh * 100):03d}-seed{sd}'))
    return out


def run_experiment(root: str, exp_id: str, definition: dict, emitter=None, cancel=None, workspace: str = '',
                   weights_dirs_extra=()) -> dict:
    """Translated synthetic against raw synthetic (and real only), the mixed experiment's design (core.experiments):
    every arm trains on the same budget of N images for the same iterations; a "raw" and a "translated" arm at the same
    share take the very same frames (the same seeded permutation), one raw and one translated; every run is scored on
    the same held-out real test set. definition: a mixed definition (synthetic: the raw drives) plus "translated":
    their translated runs, and "shares" without 0 (the real-only arm is always run)."""
    from . import experiments as X
    from . import training as T
    em = emitter or T._Quiet()
    cancel = cancel or (lambda: False)
    d = dict(definition or {})
    trans = [T.parse_source(x) if isinstance(x, str) else dict(x) for x in d.pop('translated', None) or ()]
    if not trans:
        raise TranslateError('translated needs the translated runs of the synthetic drives')
    shares = sorted({round(float(x) / 100.0 if float(x) > 1 else float(x), 6) for x in d.get('shares') or [0.5]} -
                    {0.0})
    if not shares:
        raise TranslateError('give at least one synthetic share above 0')
    m = X.normalise_mixed({**d, 'shares': [0.0] + shares})
    fixed = dict(m, kind='translated', translated=trans, train={k: v for k, v in m['train'].items()
                                                                if k not in T.VOLATILE})
    exp = X.Experiment.create(root, exp_id, fixed, m.get('name'))
    syn, real, val, test, notes, ins = X._pools(m, em)
    tsyn, n5, i5 = T.load_sources(trans, m['train']['classes'])
    tsyn = _align(syn, tsyn)
    notes += n5
    ins.update(i5)
    t_ids, t_units = {s['id'] for s in test}, {s['unit'] for s in test}
    if t_ids & {s['id'] for s in tsyn} or t_units & {s['unit'] for s in tsyn}:
        raise TranslateError('the test set shares images or sequences with the translated data')
    N = X._subsets(m, len(syn), len(real))
    seeds = [int(m['seed']) + j for j in range(int(m['seeds']))]
    plan = arm_runs(seeds, shares)
    ipe = -(-N // int(m['train']['batch']))
    em.log(f'{len(plan)} runs ({len(seeds)} seeds × real only + raw and translated at {len(shares)} share(s)); {N} '
           f'training images and {ipe * m["train"]["epochs"]} iterations per run; {len(test)} real test images')
    for k, (arm, sh, sd, name) in enumerate(plan):
        tr = exp.run(name)
        if tr.status == 'ok' and os.path.isfile(os.path.join(tr.dir, 'training', 'test_predictions.jsonl')):
            continue
        if cancel():
            raise T.Cancelled()
        n_syn = round(sh * N)
        rp, sp = X._perm(sd, 1, len(real)), X._perm(sd, 2, len(syn))
        pool = tsyn if arm == 'translated' else syn
        train_set = [real[i] for i in rp[:N - n_syn]] + [pool[i] for i in sp[:n_syn]]
        s = dict(m['train'], seed=sd)
        tr.start({'arm': arm, 'share_synthetic': sh, 'seed': sd, 'n_real': N - n_syn, 'n_synthetic': n_syn,
                  'budget_images': N, 'iterations': ipe * s['epochs']},
                 {'arm': arm, 'seed': sd, 'kind': 'translated'}, sd, ins)
        sub = X._SubEmitter(em, k / len(plan), (k + 1) / len(plan), name)
        try:
            summ = T.train(s, os.path.join(tr.dir, 'training'),
                           data={'train': train_set, 'val': val, 'test': test, 'inputs': ins,
                                 'notes': notes + [f'{arm} {round(sh * 100)} %, seed {sd}: {N - n_syn} real + '
                                                   f'{n_syn} synthetic']},
                           workspace=workspace, emitter=sub, cancel=cancel,
                           argv=['translate', 'experiment', exp.id, name], weights_dirs_extra=weights_dirs_extra)
        except T.Cancelled:
            tr.finish('cancelled')
            raise
        except T.TrainingError as e:
            tr.finish('failed', str(e))
            raise
        X._record_run(tr, summ, m)
        tr.finish('ok')
    em.progress(1.0, 'analysing', force=True)
    return analyse(exp, m, seeds, shares, N, ipe * int(m['train']['epochs']), notes)


def analyse(exp, m: dict, seeds: list, shares: list, N: int, iterations: int, notes=()) -> dict:
    """Per arm the mean over seeds of COCO AP (with a bootstrap interval over the test set's units); translated
    against raw at each share, and each against real only, paired on the same resamples; the combined verdict with
    the seed spread (core.experiments' method)."""
    from . import experiments as X
    from . import metrics as M
    from . import provenance as PV
    plan = arm_runs(seeds, shares)
    paths = {name: os.path.join(exp.path, 'runs', name, 'training', 'test_predictions.jsonl')
             for _a, _s, _d, name in plan}
    missing = [n for n, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise TranslateError(f'runs not finished: {", ".join(missing)}')
    frames = {n: {f['image_id']: f for f in X._frames(p)} for n, p in paths.items()}
    ids = sorted(next(iter(frames.values())))
    recs = {n: {r['image_id']: r for r in M.coco_match([fr[i] for i in ids])} for n, fr in frames.items()}
    first = frames[next(iter(frames))]
    items = [{'image_id': i, 'drive': first[i]['drive'], 'r': {n: recs[n][i] for n in recs}} for i in ids]
    arms = [('real only', 0.0)] + [(a, sh) for sh in shares for a in ('raw', 'translated')]
    label = {(a, sh): a if a == 'real only' else f'{a} {round(sh * 100)} %' for a, sh in arms}
    runs_of = {key: [name for a, sh, _d, name in plan if (a, sh) == key] for key in arms}
    n_boot, bseed = int(m['n_boot']), int(m['boot_seed'])

    def numbers(its, key):
        per = []
        for name in runs_of[key]:
            try:
                s = M.coco_summarise([it['r'][name] for it in its])
            except ValueError:
                s = {}
            per.append([s.get(k) for k in X.HEADLINE])
        out = {}
        for j, k in enumerate(X.HEADLINE):
            v = [p[j] for p in per if p[j] is not None]
            out[k] = float(np.mean(v)) if len(v) == len(per) else None
        return out

    boot = M.bootstrap(lambda its: {f'{label[a]}|{k}': v for a in arms for k, v in numbers(its, a).items()}, items,
                       n=n_boot, seed=bseed, unit='drive')
    rows = {}
    for key in arms:
        per_seed = []
        for name in runs_of[key]:
            s = M.coco_summarise([recs[name][i] for i in ids])
            per_seed.append({'run': name, **{k: s.get(k) for k in X.HEADLINE}})
        spread = {}
        for k in X.HEADLINE:
            v = [p[k] for p in per_seed if p[k] is not None]
            sd_ = float(np.std(v, ddof=1)) if len(v) > 1 else None
            spread[k] = {'sd': sd_, 'se': sd_ / math.sqrt(len(v)) if sd_ is not None else None}
        rows[key] = {'arm': label[key], 'kind': key[0], 'share_synthetic': key[1],
                     **{k: {kk: boot['metrics'][f'{label[key]}|{k}'][kk] for kk in ('value', 'ci_low', 'ci_high')}
                        for k in X.HEADLINE}, 'per_seed': per_seed, 'seed_spread': spread}

    def compare(a, b, what):
        pb = M.paired_bootstrap(lambda its, key: numbers(its, key), items, a, b, n=n_boot, seed=bseed, unit='drive')
        r = pb['metrics']['AP']
        se_a, se_b = rows[a]['seed_spread']['AP']['se'], rows[b]['seed_spread']['AP']['se']
        comb = None
        if r['diff'] is not None and r['sd'] is not None and se_a is not None and se_b is not None:
            h = 1.96 * math.sqrt(r['sd'] ** 2 + se_a ** 2 + se_b ** 2)
            comb = [r['diff'] - h, r['diff'] + h]
        lo, hi = comb if comb else (r['ci_low'], r['ci_high'])
        if lo is None or hi is None:
            verdict = 'not measurable (nothing to score)'
        elif lo > 0:
            verdict = f'better than {what}'
        elif hi < 0:
            verdict = f'worse than {what}'
        else:
            verdict = 'no detectable difference at 95 %'
        if comb is None and len(seeds) == 1:
            verdict += ' (one seed: training noise not measured)'
        return {'a': label[a], 'b': label[b], 'metric': 'AP', 'diff': r['diff'], 'ci_low': r['ci_low'],
                'ci_high': r['ci_high'], 'sd_boot': r['sd'], 'combined_ci': comb, 'verdict': verdict}
    comparisons = []
    for sh in shares:
        comparisons.append(compare(('raw', sh), ('translated', sh), 'raw synthetic'))
        for a in ('raw', 'translated'):
            comparisons.append(compare(('real only', 0.0), (a, sh), 'real only'))
    warn = []
    if boot['n_units'] < X.MIN_UNITS:
        warn.append(f'only {boot["n_units"]} test units: the intervals are not reliable')
    if len(seeds) == 1:
        warn.append('one seed per arm: the seed-to-seed spread is not measured')
    doc = {'schema': RESULTS_SCHEMA, 'experiment': exp.id, 'budget_images': N, 'iterations_per_run': iterations,
           'seeds': seeds, 'shares': shares, 'classes': m['train']['classes'],
           'test': {'images': len(ids), 'objects': sum(1 for i in ids for g in first[i]['gt'] if not g['ignore']),
                    'unit': X._unit_description(list(first.values())), 'n_units': boot['n_units']},
           'n_boot': n_boot, 'boot_seed': bseed, 'arms': [rows[k] for k in arms], 'comparisons': comparisons,
           'warnings': warn, 'notes': list(notes),
           'method': {'design': 'core.experiments\' mixed design; a raw and a translated arm at one share take the '
                                'same frames (one seeded permutation), raw or translated',
                      'interval': f'95 % percentile bootstrap over test units ({n_boot} resamples), every run '
                                  f'resampled together; paired differences on the same resamples',
                      'combined': 'diff ± 1.96 √(sd_boot² + se_a² + se_b²) (normal approximation)'},
           'provenance': PV.block(dict(paths), bseed, {'n_boot': n_boot})}
    PV.write_json(os.path.join(exp.path, 'results.json'), doc)
    return doc


def table(res: dict) -> str:
    """A translated-vs-raw results.json as text."""
    lines = [f'{"arm":>18s}  {"AP [95 % CI]":26s} {"seed sd":>8s}', '-' * 60]
    for r in res['arms']:
        ap = r['AP']
        ci = (f'{ap["value"]:.4f} [{ap["ci_low"]:.4f}, {ap["ci_high"]:.4f}]' if ap['value'] is not None and
              ap['ci_low'] is not None else '—')
        sd = r['seed_spread']['AP']['sd']
        lines.append(f'{r["arm"]:>18s}  {ci:26s} {"—" if sd is None else f"{sd:.4f}":>8s}')
    lines.append('')
    for c in res['comparisons']:
        if c['diff'] is None:
            continue
        lines.append(f'{c["b"]} − {c["a"]}: AP {c["diff"]:+.4f} [{c["ci_low"]:+.4f}, {c["ci_high"]:+.4f}] → '
                     f'{c["verdict"]}')
    t = res['test']
    lines.append(f'test: {t["images"]} images, {t["objects"]} objects, unit {t["unit"]} ({t["n_units"]}); '
                 f'{len(res["seeds"])} seed(s); {res["budget_images"]} images and {res["iterations_per_run"]} '
                 f'iterations per run')
    lines += [f'note: {w}' for w in res.get('warnings') or ()]
    return '\n'.join(lines)


# ================================================================================================ jobs and the command
def job_spec(argv: list, title: str, memory_gb: float = 6.0, run_dir: str = ''):
    """A `falcon translate …` job: needs_gpu (the JobQueue never runs it while a CARLA job holds the GPU, and the
    process knows the queue holds the GPU lock)."""
    import sys
    from .jobs import JobSpec
    env = dict(os.environ, FALCON_GPU_LOCK_HELD='1')
    return JobSpec('translate', title, [sys.executable, '-m', 'falcon_autonomy', 'translate'] + list(argv), env=env,
                   needs_gpu=True, memory_gb=memory_gb, run_dir=run_dir, resumable=argv[:1] == ['train'])


def default_name(src_dir: str, model_dir: str) -> str:
    return f'{os.path.basename(os.path.normpath(src_dir))}-tr-{os.path.basename(os.path.normpath(model_dir))}'


def main_translate(argv) -> int:
    """`falcon translate train|apply|check|experiment|results …`: protocol lines on stdout; 0 ok, 2 failed,
    3 cancelled."""
    import argparse
    from . import protocol as P
    from . import provenance as PV
    from . import training as T
    ap = argparse.ArgumentParser(prog='falcon translate', description='learned sim-to-real image translation with '
                                                                      'the labels kept (docs/translation.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    tr = sub.add_parser('train', help='train a translator (CUT) from Falcon runs to a real set (resumable)')
    tr.add_argument('--out', '-o', default='', help='the translator folder (default WORKSPACE/translators/ID)')
    tr.add_argument('--workspace', default='')
    tr.add_argument('--id', default='')
    tr.add_argument('--config', help='a JSON file of settings (the flags below override it)')
    tr.add_argument('--sim', action='append', default=None, metavar='RUN[?camera=C&every=N]')
    tr.add_argument('--real', action='append', default=None, metavar='KIND:PATH[?limit=N]',
                    help=f'KIND one of {", ".join(REAL_KINDS)}')
    for k, t in (('iterations', int), ('batch', int), ('crop', int), ('work-width', int), ('ngf', int),
                 ('n-blocks', int), ('lr', float), ('seed', int), ('max-images', int), ('probe-every', int),
                 ('log-every', int), ('ckpt-every', int), ('lambda-nce', float), ('gpu-memory-fraction', float)):
        tr.add_argument(f'--{k}', type=t)
    tr.add_argument('--device')
    tr.add_argument('--no-condition', action='store_true', help='do not give the generator the semantic label')
    tr.add_argument('--depth', action='store_true', help=f'give the generator the inverse range too (min(1, '
                                                         f'{DEPTH_MIN_M:g} m / range), from the range image; measured '
                                                         f'not to help, docs/translation.md)')
    tr.add_argument('--no-idt', action='store_true', help='no identity NCE on real images (FastCUT-like)')
    tr.add_argument('--no-amp', action='store_true')
    tr.add_argument('--keep-checkpoint', action='store_true')
    a_ = sub.add_parser('apply', help='translate recorded runs into new runs, every label kept')
    a_.add_argument('--model', required=True, help='a translator folder')
    a_.add_argument('--run', action='append', required=True, help='a Falcon run (repeatable)')
    a_.add_argument('--out', default='', help='the translated run\'s folder (one --run only)')
    a_.add_argument('--out-root', default='', help='where translated runs go (default WORKSPACE/runs)')
    a_.add_argument('--workspace', default='')
    a_.add_argument('--camera', action='append', default=None, help='cameras to translate (default every pinhole)')
    a_.add_argument('--threshold', type=float, default=DRIFT_THRESHOLD, help=f'drift threshold (heuristic, default '
                                                                             f'{DRIFT_THRESHOLD})')
    a_.add_argument('--on-drift', choices=('raw', 'flag'), default='raw',
                    help='a frame over the threshold: keep the raw frame (default) or keep the translation, listed')
    a_.add_argument('--native', action='store_true', help='run the network at the camera\'s full resolution')
    a_.add_argument('--pin-actors', action=argparse.BooleanOptionalAction, default=PIN_DEFAULT,
                    help='foreground pinning: inside each dynamic actor\'s instance mask only a colour map of the '
                         'input (default: on every camera that has instance images)')
    a_.add_argument('--pin-feather', type=int, default=PIN_FEATHER_PX, help='the blend ring outside a pinned mask, px')
    a_.add_argument('--copy', action='store_true', help='copy the source\'s files instead of hard-linking them')
    a_.add_argument('--device', default='auto')
    ck = sub.add_parser('check', help='the drift of translated runs and, with --real, the gap before and after')
    ck.add_argument('runs', nargs='+')
    ck.add_argument('--real', help='KIND:PATH of the real set for the gap')
    ck.add_argument('--features', action='store_true', help='Fréchet and kernel distances (falcon gap --fetch first)')
    ck.add_argument('--workspace', default='')
    ck.add_argument('--images', type=int, default=240)
    ck.add_argument('--seed', type=int, default=0)
    ck.add_argument('--segmenter', help='a segmenter registration (model JSON, core.models)')
    ck.add_argument('--out', '-o', default='')
    ex = sub.add_parser('experiment', help='translated synthetic against raw synthetic on a real test set')
    ex.add_argument('--root', default='')
    ex.add_argument('--workspace', default='')
    ex.add_argument('--id', required=True)
    ex.add_argument('--config')
    ex.add_argument('--synthetic', action='append', default=None, metavar='falcon:RUN', help='the raw drives')
    ex.add_argument('--translated', action='append', default=None, metavar='falcon:RUN', help='their translations')
    ex.add_argument('--real-train', action='append', default=None)
    ex.add_argument('--real-val', action='append', default=None)
    ex.add_argument('--real-test', action='append', default=None)
    ex.add_argument('--shares', help='synthetic shares in %%, e.g. 50 (the real-only arm is always run)')
    ex.add_argument('--budget', type=int)
    ex.add_argument('--seeds', type=int)
    ex.add_argument('--boot', type=int)
    for k, t in (('epochs', int), ('batch', int), ('workers', int)):
        ex.add_argument(f'--{k}', type=t)
    ex.add_argument('--arch', choices=tuple(T.ARCHS))
    ex.add_argument('--init', choices=T.INITS)
    ex.add_argument('--size')
    ex.add_argument('--classes')
    ex.add_argument('--device')
    ex.add_argument('--no-export', action='store_true', help='do not export each run\'s model to ONNX')
    rs = sub.add_parser('results', help='a translated-vs-raw experiment\'s table')
    rs.add_argument('experiment')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('translate')
    if a.action == 'results':
        try:
            print(table(PV.read_json(os.path.join(a.experiment, 'results.json'))), flush=True)
        except (OSError, ValueError, KeyError) as e:
            print(f'failed: {e}', flush=True)
            return P.FAILED
        em.result('ok')
        return P.OK
    from .experiments import ExperimentError
    flag = T.CancelFlag().install()
    locks = []
    try:
        if a.action == 'train':
            s = PV.read_json(a.config) if a.config else {}
            over = {'iterations': a.iterations, 'batch': a.batch, 'crop': a.crop, 'work_width': a.work_width,
                    'ngf': a.ngf, 'n_blocks': a.n_blocks, 'lr': a.lr, 'seed': a.seed, 'max_images': a.max_images,
                    'probe_every': a.probe_every, 'log_every': a.log_every, 'ckpt_every': a.ckpt_every,
                    'lambda_nce': a.lambda_nce, 'gpu_memory_fraction': a.gpu_memory_fraction, 'device': a.device}
            s.update({k: v for k, v in over.items() if v is not None})
            if a.sim:
                s['sim'] = list(a.sim)
            if a.real:
                s['real'] = list(a.real)
            for flag_, k in ((a.no_condition, 'condition'), (a.no_idt, 'nce_idt'), (a.no_amp, 'amp')):
                if flag_:
                    s[k] = False
            if a.keep_checkpoint:
                s['keep_checkpoint'] = True
            if a.depth:
                s['depth'] = True
            s = normalise(s)
            out = a.out or (os.path.join(a.workspace, 'translators', a.id or time.strftime('%Y%m%d-%H%M%S-cut'))
                            if a.workspace else '')
            if not out:
                raise TranslateError('--out DIR or --workspace W is needed')
            if _device(s).type == 'cuda':
                locks = T.gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
            doc = train(s, out, em, flag, ['translate'] + list(argv))
            print(f'translator {out}: {doc["cost"]["train_s"]} s, {doc["cost"]["images_per_s_median"]} images/s, '
                  f'probe drift median {(doc.get("probe") or {}).get("drift_median")}', flush=True)
            em.result('ok', translator=os.path.join(out, 'translator.json'), terms=doc['terms'])
            return P.OK
        if a.action == 'apply':
            if a.out and len(a.run) != 1:
                raise TranslateError('--out names one translated run; with several --run use --out-root')
            root = a.out_root or (os.path.join(a.workspace, 'runs') if a.workspace else '')
            if not a.out and not root:
                raise TranslateError('--out, --out-root or --workspace is needed')
            if _device({'device': a.device}).type == 'cuda':
                locks = T.gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
            model = Translator(a.model, a.device)
            em.log(f'translator {model.doc.get("id")}: {model.doc.get("licence_note")}')
            made = []
            for i, r in enumerate(a.run):
                out = a.out or os.path.join(root, default_name(r, a.model))
                sub_em = _Slice(em, i / len(a.run), (i + 1) / len(a.run))
                doc = translate_run(a.model, r, out, a.camera, a.threshold, a.on_drift, 'copy' if a.copy else 'hard',
                                    a.native, a.device, model, sub_em, flag, ['translate'] + list(argv),
                                    pin=a.pin_actors, feather=a.pin_feather)
                d = doc['translation']['drift']
                print(f'{out}: {d["frames"]} frames, drift median {d["drift"].get("median", float("nan")):.3f}, '
                      f'{d["flagged"]} over {d["threshold"]} ({d["refused"]} kept raw)', flush=True)
                em.send({'t': 'recording', 'path': out})
                made.append(out)
            em.result('ok', runs=made)
            return P.OK
        if a.action == 'check':
            seg = PV.read_json(a.segmenter) if a.segmenter else None
            if a.features and not a.workspace:
                raise TranslateError('--features needs --workspace (falcon gap --fetch)')
            if a.features and _device({'device': 'auto'}).type == 'cuda':      # the feature network runs on CUDA
                locks = T.gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
            doc = check_runs(a.runs, a.real, a.features, a.workspace, a.images, a.seed, seg, em)
            doc['provenance'] = PV.block({f'run{i}/run.json': os.path.join(p, 'run.json') for i, p in
                                         enumerate(a.runs)}, a.seed, {k: v for k, v in vars(a).items()})
            if a.out:
                PV.write_json(os.path.join(a.out, 'check.json'), doc)
            print(check_text(doc), flush=True)
            em.result('ok', check=os.path.join(a.out, 'check.json') if a.out else None)
            return P.OK
        # experiment
        d = PV.read_json(a.config) if a.config else {}
        over = {'synthetic': a.synthetic, 'translated': a.translated, 'real_train': a.real_train,
                'real_val': a.real_val, 'real_test': a.real_test, 'budget_images': a.budget, 'seeds': a.seeds,
                'n_boot': a.boot}
        d.update({k: v for k, v in over.items() if v is not None})
        if a.shares:
            d['shares'] = [float(x) for x in a.shares.split(',') if x.strip()]
        trn = dict(d.get('train') or {})
        trn.update({k: v for k, v in {'epochs': a.epochs, 'batch': a.batch, 'workers': a.workers, 'arch': a.arch,
                                      'init': a.init, 'device': a.device}.items() if v is not None})
        if a.size:
            trn['size'] = T._size(a.size)
        if a.classes:
            trn['classes'] = [c.strip() for c in a.classes.split(',') if c.strip()]
        if a.no_export:
            trn['export'] = False
        d['train'] = trn
        root = a.root or (os.path.join(a.workspace, 'experiments') if a.workspace else '')
        if not root:
            raise TranslateError('--root DIR or --workspace W is needed')
        if T._device(T.normalise(dict(trn, train=[], val=[], test=[]))).type == 'cuda':
            locks = T.gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
        res = run_experiment(root, a.id, d, em, flag, a.workspace)
        print(table(res), flush=True)
        em.result('ok', results=os.path.join(root, a.id, 'results.json'))
        return P.OK
    except T.Cancelled:
        em.result('cancelled')
        return P.CANCELLED
    except (TranslateError, ExperimentError, T.TrainingError, ValueError, OSError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        return P.FAILED
    finally:
        for lk in locks:
            lk.release()


class _Slice:
    """An emitter whose progress fills a slice of the whole."""

    def __init__(self, em, lo, hi):
        self.em, self.lo, self.hi = em, lo, hi

    def progress(self, f, text='', force=False):
        self.em.progress(self.lo + (self.hi - self.lo) * float(f), text, force=force)

    def log(self, text, level='info'):
        self.em.log(text, level)

    def metric(self, *a, **k):
        self.em.metric(*a, **k)


def check_text(doc: dict) -> str:
    d = doc['drift']
    lines = [f'drift over {d["frames"]} frames: median {d["drift"].get("median", float("nan")):.3f}, 90th percentile '
             f'{d["drift"].get("p90", float("nan")):.3f}, max {d["drift"].get("max", float("nan")):.3f}; '
             f'{d["flagged"]} over {d["threshold"]} (heuristic); {doc.get("kept_raw", 0)} frames were refused when '
             f'translated and are the raw frames',
             f'structure median {d["structure"].get("median", float("nan")):.3f}; edge recall '
             f'{d["edge_recall"].get("median", float("nan")):.3f}; edge precision '
             f'{d["edge_precision"].get("median", float("nan")):.3f}; worst object box median '
             f'{_f(d["worst_box"].get("median"))} ({d["worst_box"]["n"]} frames with objects)']
    if (d.get('worst_actor') or {}).get('n'):
        lines.append(f'dynamic actors ({d["worst_actor"]["n"]} frames): worst box structure median '
                     f'{_f(d["worst_actor"].get("median"))}; their boxes\' edge recall '
                     f'{_f((d.get("actor_edge_recall") or {}).get("median"))}, precision '
                     f'{_f((d.get("actor_edge_precision") or {}).get("median"))}')
    if doc.get('segmenter'):
        s = doc['segmenter']
        lines.append(f'segmenter {s["model"]}: agreement {s["agreement"]}, mIoU raw {s["miou_raw"]} → translated '
                     f'{s["miou_translated"]} ({s["frames"]} frames)')
    g = doc.get('gap')
    if g:
        lines.append(f'gap against {g["real"]["kind"]} ({g["real"]["used"]} images):')
        for r in sorted(g['appearance'], key=lambda r: -(r['raw_shift'] or 0)):
            lines.append(f'  {r["stat"]:15s} real {r["real_median"]:8.3f}  raw {r["raw_median"]:8.3f} (shift '
                         f'{_f(r["raw_shift"])})  translated {r["translated_median"]:8.3f} (shift '
                         f'{_f(r["translated_shift"])})')
        if g.get('night'):
            lines.append('  night statistics:')
            for r in sorted(g['night'], key=lambda r: -(r['raw_shift'] or 0)):
                lines.append(f'  {r["stat"]:15s} real {r["real_median"]:8.3f}  raw {r["raw_median"]:8.3f} (shift '
                             f'{_f(r["raw_shift"])})  translated {r["translated_median"]:8.3f} (shift '
                             f'{_f(r["translated_shift"])})')
        fr, ft = g.get('features_raw'), g.get('features_translated')
        if fr and ft:
            lines.append(f'  FD raw {fr["fd"]:.1f} → translated {ft["fd"]:.1f} (real-vs-real floor {fr["fd_floor"]:.1f}'
                         f'); KD {fr["kd"]:.4f} → {ft["kd"]:.4f} (floor {fr["kd_floor"]:.4f}); {fr["n_sim"]} vs '
                         f'{fr["n_real"]} images')
    return '\n'.join(lines)


def _f(v) -> str:
    return '—' if v is None else f'{v:.2f}'
