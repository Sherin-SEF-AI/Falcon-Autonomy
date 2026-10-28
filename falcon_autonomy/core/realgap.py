"""The sim-to-real gap: how far Falcon's camera images and labels are from a real dataset's, measured and stated
(numpy at import; OpenCV, scipy and torch only inside the functions that need them).

Real sets are read in place — nothing is copied into the project:

    folder        any folder of images (recursively; .png .jpg .jpeg), no labels
    kitti_object  KITTI 2D/3D object: image_2/*.png and label_2/*.txt
    kitti_raw     a KITTI raw drive: image_02/data/*.png (colour camera), no labels (tracklets are not read)
    nuscenes      a nuScenes release folder (v1.0-mini, v1.0-trainval, …): one camera's key frames; boxes are the
                  annotations' 3D boxes projected into the image (the corners in front of the camera, clipped)
    bdd100k       an image folder and BDD100K's detection labels JSON (weather and time of day from its attributes)
    coco          an image folder and a COCO instances JSON (class names mapped through NAME_ALIASES)
    falcon        a Falcon run folder (falcon-raw): one camera's RGB and its objects

Measured, per image at a common working width (WORK_WIDTH px, so that resolution does not dominate):
CIELAB mean and spread, saturation, clipped shares, a noise estimate (Immerkær 1996), sharpness (variance of the
Laplacian, less its noise share) and the slope of the radially averaged power spectrum. Per object: class shares,
objects per image, box height as a share of the image height and range (labelled sets that give one). Each
statistic's gap is the 1-Wasserstein distance between the two sets' distributions, also divided by the real set's
interquartile range (the "shift", unitless; a heuristic scale for ranking, not a test).

Fréchet and kernel distances (FD, KD — the method of FID and KID) are computed in the pooled feature space of a fixed
network: torchvision's ResNet-50 ImageNet-1k weights (V2), downloaded only on request from a pinned URL whose SHA-256
is checked. These are **not** comparable with published FID numbers (those use a specific Inception network) and
both depend on the number of images; each is therefore reported next to its floor — the same distance between two
random halves of the real set, at the same set sizes.

Appearance matching (optional) is a fixed, non-learned transform of the simulated images: CIELAB histogram matching
from the simulated set's pooled histograms to the real set's (one lookup table per channel for the whole set), a
Gaussian blur fitted to the real set's sharpness and noise added up to its noise level. Its effect on every
statistic above is measured, not assumed. Learned style transfer is not part of Falcon.
"""
from __future__ import annotations

import glob
import json
import math
import os
import warnings

import numpy as np

SCHEMA = 'falcon.gap/1'
KINDS = ('folder', 'kitti_object', 'kitti_raw', 'nuscenes', 'bdd100k', 'coco', 'falcon')
IMAGE_EXT = ('.png', '.jpg', '.jpeg')
WORK_WIDTH = 640
MIN_BOX_SHARE = 0.015          # objects lower than this share of the image height are left out on both sides
THINGS = ('car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'rider', 'pedestrian', 'traffic_light',
          'traffic_sign')
NAME_ALIASES = {
    'car': 'car', 'van': 'car', 'vehicle.car': 'car', 'vehicle.emergency.police': 'car', 'taxi': 'car',
    'truck': 'truck', 'vehicle.truck': 'truck', 'vehicle.trailer': 'truck', 'vehicle.construction': 'truck',
    'bus': 'bus', 'vehicle.bus.rigid': 'bus', 'vehicle.bus.bendy': 'bus',
    'train': 'train', 'tram': 'train',
    'motorcycle': 'motorcycle', 'motor': 'motorcycle', 'vehicle.motorcycle': 'motorcycle', 'motorbike': 'motorcycle',
    'bicycle': 'bicycle', 'bike': 'bicycle', 'vehicle.bicycle': 'bicycle', 'cyclist': 'bicycle',
    'rider': 'rider',
    'pedestrian': 'pedestrian', 'person': 'pedestrian', 'person_sitting': 'pedestrian',
    'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian', 'human.pedestrian.police_officer': 'pedestrian',
    'traffic light': 'traffic_light', 'traffic_light': 'traffic_light',
    'traffic sign': 'traffic_sign', 'traffic_sign': 'traffic_sign', 'stop sign': 'traffic_sign',
}
FEATURES = {
    'resnet50-imagenet1k-v2': {
        'url': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        'sha256': '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca',
        'licence': 'BSD-3-Clause (torchvision); ImageNet-1k training data under its own terms',
        'file': 'resnet50-11ad3fa6.pth', 'dims': 2048, 'input': 224, 'mb': 97.8,
    },
}
DEFAULT_FEATURES = 'resnet50-imagenet1k-v2'
IMAGE_STATS = ('L_mean', 'L_std', 'a_mean', 'b_mean', 'ab_std', 'saturation', 'dark_share', 'bright_share',
               'noise_sigma', 'sharpness', 'spectrum_slope')
STAT_NOTES = {
    'L_mean': 'CIELAB lightness, mean (0–100)', 'L_std': 'CIELAB lightness, spread (contrast)',
    'a_mean': 'CIELAB a (green–red), mean', 'b_mean': 'CIELAB b (blue–yellow), mean',
    'ab_std': 'chroma spread', 'saturation': 'HSV saturation, mean (0–1)',
    'dark_share': 'share of pixels at L < 2 (crushed blacks)', 'bright_share': 'share of pixels with a channel at 255',
    'noise_sigma': 'noise estimate, 8-bit levels (Immerkær 1996; texture raises it)',
    'sharpness': 'variance of the Laplacian less its noise share (log10)',
    'spectrum_slope': 'power-spectrum slope (natural images ≈ −2)',
}


class GapError(Exception):
    pass


# ================================================================================================ reading sets
def alias(name) -> str | None:
    return NAME_ALIASES.get(str(name or '').strip().lower())


def _sample(items: list, limit: int | None, seed: int) -> list:
    if not limit or len(items) <= limit:
        return items
    idx = np.sort(np.random.default_rng(seed).choice(len(items), int(limit), replace=False))
    return [items[i] for i in idx]


def read_set(kind: str, path: str, labels: str | None = None, camera: str | None = None, limit: int | None = None,
             seed: int = 0) -> dict:
    """A set of images (and their objects when labelled): {'kind', 'path', 'labelled', 'items': [{'image',
    'objects': [{'class', 'bbox' [x0 y0 x1 y1] px, 'range_m'?}] | None, 'conditions': {}}], 'notes'}. `limit`
    takes a seeded random subset (in the set's order)."""
    if kind not in KINDS:
        raise GapError(f'unknown set kind {kind!r}; one of {KINDS}')
    if not os.path.exists(path):
        raise GapError(f'{path} does not exist')
    reader = {'folder': _folder, 'kitti_object': _kitti_object, 'kitti_raw': _kitti_raw, 'nuscenes': _nuscenes,
              'bdd100k': _bdd, 'coco': _coco, 'falcon': _falcon}[kind]
    items, notes = reader(path, labels, camera)
    if not items:
        raise GapError(f'no images found in {path} as {kind}')
    return {'kind': kind, 'path': os.path.abspath(path), 'labels': labels, 'camera': camera,
            'labelled': items[0]['objects'] is not None, 'n_total': len(items),
            'items': _sample(items, limit, seed), 'notes': notes}


def _images_under(root: str) -> list:
    out = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        out += [os.path.join(d, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXT)]
    return out


def _folder(path, _labels, _camera):
    return [{'image': p, 'objects': None, 'conditions': {}} for p in _images_under(path)], []


def _kitti_raw(path, _labels, _camera):
    ims = sorted(glob.glob(os.path.join(path, 'image_02', 'data', '*.png')))
    if not ims:      # a date folder holding several drives
        ims = sorted(glob.glob(os.path.join(path, '*', 'image_02', 'data', '*.png')))
    return ([{'image': p, 'objects': None, 'conditions': {}} for p in ims],
            ['KITTI raw: the left colour camera (image_02); tracklet labels are not read'])


def _kitti_object(path, _labels, _camera):
    items = []
    for p in sorted(glob.glob(os.path.join(path, 'image_2', '*.png'))):
        lab = os.path.join(path, 'label_2', os.path.splitext(os.path.basename(p))[0] + '.txt')
        objs = []
        if os.path.isfile(lab):
            for line in open(lab, encoding='utf-8'):
                f = line.split()
                if len(f) < 15:
                    continue
                c = alias(f[0])
                if c is None:            # Misc, DontCare
                    continue
                x, y, z = float(f[11]), float(f[12]), float(f[13])
                objs.append({'class': c, 'bbox': [float(v) for v in f[4:8]],
                             'range_m': math.sqrt(x * x + y * y + z * z)})
        items.append({'image': p, 'objects': objs, 'conditions': {}})
    return items, ['KITTI: Cyclist counts as bicycle, Van as car, Tram as train; Misc and DontCare left out; range '
                   'to the box\'s bottom centre']


def _quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _nuscenes(path, _labels, camera):
    rel = None
    for v in ('v1.0-mini', 'v1.0-trainval', 'v1.0-test'):
        if os.path.isfile(os.path.join(path, v, 'sample_data.json')):
            rel = v
            break
    if rel is None:
        raise GapError(f'{path}: no v1.0-*/sample_data.json (a nuScenes release folder)')
    camera = camera or 'CAM_FRONT'

    def tab(name):
        with open(os.path.join(path, rel, name + '.json'), encoding='utf-8') as fh:
            return json.load(fh)
    sensors = {s['token']: s['channel'] for s in tab('sensor')}
    cs = {c['token']: c for c in tab('calibrated_sensor')}
    poses = {e['token']: e for e in tab('ego_pose')}
    cats = {c['token']: c['name'] for c in tab('category')}
    inst = {i['token']: cats.get(i['category_token']) for i in tab('instance')}
    samples = {s['token']: s for s in tab('sample')}
    scenes = {s['token']: s for s in tab('scene')}
    anns = {}
    for a in tab('sample_annotation'):
        anns.setdefault(a['sample_token'], []).append(a)
    items = []
    sds = [d for d in tab('sample_data') if d.get('is_key_frame') and sensors.get(cs[d['calibrated_sensor_token']]
                                                                                    ['sensor_token']) == camera]
    sds.sort(key=lambda d: d['timestamp'])
    corners0 = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
    for d in sds:
        c = cs[d['calibrated_sensor_token']]
        e = poses[d['ego_pose_token']]
        K = np.asarray(c['camera_intrinsic'], float)
        Re, te = _quat(e['rotation']), np.asarray(e['translation'], float)
        Rc, tc = _quat(c['rotation']), np.asarray(c['translation'], float)
        W, H = int(d.get('width') or 1600), int(d.get('height') or 900)
        objs = []
        for a in anns.get(d['sample_token'], ()):
            cl = alias(inst.get(a['instance_token']))
            if cl is None:
                continue
            w, l, h = a['size']
            P = corners0 * np.array([l, w, h]) @ _quat(a['rotation']).T + np.asarray(a['translation'], float)
            P = ((P - te) @ Re - tc) @ Rc                       # global -> ego -> camera
            ctr = ((np.asarray(a['translation'], float) - te) @ Re - tc) @ Rc
            P = P[P[:, 2] > 0.1]
            if ctr[2] <= 0.5 or len(P) == 0:
                continue
            uv = P @ K.T
            uv = uv[:, :2] / uv[:, 2:3]
            x0, y0 = np.clip(uv.min(0), 0, [W, H])
            x1, y1 = np.clip(uv.max(0), 0, [W, H])
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            objs.append({'class': cl, 'bbox': [float(x0), float(y0), float(x1), float(y1)],
                         'range_m': float(np.linalg.norm(ctr))})
        desc = (scenes.get(samples.get(d['sample_token'], {}).get('scene_token'), {}).get('description') or '').lower()
        cond = {'night': 'night' in desc, 'rain': 'rain' in desc}
        items.append({'image': os.path.join(path, d['filename']), 'objects': objs, 'conditions': cond})
    return items, [f'nuScenes {rel}, {camera} key frames: boxes are the annotated 3D boxes projected (corners in '
                   f'front of the camera, clipped to the image) — looser than drawn 2D boxes; night and rain from the '
                   f'scene descriptions']


def _bdd(path, labels, _camera):
    if not labels:
        raise GapError('a BDD100K set needs its labels JSON (labels=…)')
    with open(labels, encoding='utf-8') as fh:
        doc = json.load(fh)
    items = []
    for fr in doc:
        p = os.path.join(path, fr['name'])
        if not os.path.isfile(p):
            continue
        objs = []
        for lab in fr.get('labels') or ():
            c, b = alias(lab.get('category')), lab.get('box2d')
            if c and b:
                objs.append({'class': c, 'bbox': [b['x1'], b['y1'], b['x2'], b['y2']]})
        at = fr.get('attributes') or {}
        items.append({'image': p, 'objects': objs,
                      'conditions': {k: at[k] for k in ('weather', 'timeofday', 'scene') if k in at}})
    return items, ['BDD100K: 2D boxes as labelled; no range']


def _coco(path, labels, _camera):
    if not labels:
        raise GapError('a COCO set needs its instances JSON (labels=…)')
    with open(labels, encoding='utf-8') as fh:
        doc = json.load(fh)
    names = {c['id']: alias(c['name']) for c in doc.get('categories', ())}
    by = {}
    for a in doc.get('annotations', ()):
        c = names.get(a['category_id'])
        if c and not a.get('iscrowd'):
            x, y, w, h = a['bbox']
            by.setdefault(a['image_id'], []).append({'class': c, 'bbox': [x, y, x + w, y + h]})
    items = [{'image': os.path.join(path, im['file_name']), 'objects': by.get(im['id'], []), 'conditions': {}}
             for im in sorted(doc.get('images', ()), key=lambda i: i['id'])]
    unmapped = sorted({c['name'] for c in doc.get('categories', ()) if alias(c['name']) is None})
    return items, ['COCO: categories without a Falcon class left out' + (f' ({", ".join(unmapped[:8])}'
                                                                        f'{" …" if len(unmapped) > 8 else ""})'
                                                                        if unmapped else '')]


def _falcon(path, _labels, camera):
    from .exporters import ExportError, Run
    try:
        run = Run(path)
    except ExportError as e:
        raise GapError(str(e)) from None
    cams = [c for c, s in run.cams.items() if s.get('type') == 'camera.pinhole'] or list(run.cams)
    if not cams:
        raise GapError(f'{path}: the run has no camera')
    cid = camera or cams[0]
    if cid not in run.cams:
        raise GapError(f'{path}: no camera {cid!r} (has {", ".join(run.cams)})')
    w = run.cfg.get('weather') or {}
    cond = {'weather': w.get('preset') or w.get('name'), 'night': (w.get('sun_altitude_angle', 45) or 45) < 0}
    items = []
    for rec in run.camera_steps(cid):
        objs = run.objects(rec, cid)
        if objs is not None:
            objs = [{'class': o['class'], 'bbox': o['bbox'], 'range_m': o.get('range_m')}
                    for o in objs if o.get('class') in THINGS and o.get('bbox')]
        items.append({'image': run.p(run.file(rec, cid, 'rgb')), 'objects': objs, 'conditions': cond})
    return items, [f'Falcon run {run.name}, camera {cid}']


# ================================================================================================ statistics
def load_rgb(path: str) -> np.ndarray:
    import cv2
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise GapError(f'cannot read image {path}')
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def working(rgb: np.ndarray, width: int = WORK_WIDTH) -> np.ndarray:
    import cv2
    h, w = rgb.shape[:2]
    if w == width:
        return rgb
    return cv2.resize(rgb, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkær's fast noise estimate (1996), in the image's levels."""
    import cv2
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    g = gray.astype(np.float32)
    r = cv2.filter2D(g, -1, M, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    h, w = g.shape
    return float(math.sqrt(math.pi / 2) * np.abs(r).sum() / (6.0 * (w - 2) * (h - 2)))


def sharpness(gray: np.ndarray, sigma: float | None = None) -> float:
    """log10 of the Laplacian's variance less what white noise of the estimated level gives it (20 σ² for this
    kernel): the texture's share, so that noise is not counted as sharpness. Floored at 10^-3."""
    import cv2
    sigma = noise_sigma(gray) if sigma is None else sigma
    v = float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=1).var()) - 20.0 * sigma * sigma
    return math.log10(max(v, 1e-3))


def spectrum_slope(gray: np.ndarray) -> float:
    """Slope of log power against log frequency, radially averaged, 0.02–0.4 cycles per pixel (Hann window)."""
    g = gray.astype(np.float64)
    g = g - g.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    edges = np.geomspace(0.02, 0.4, 16)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.any():
            xs.append(math.log10(math.sqrt(lo * hi)))
            ys.append(math.log10(P[m].mean() + 1e-12))
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 2 else float('nan')


HIST_RANGE = ((0.0, 100.0), (-128.0, 128.0), (-128.0, 128.0))     # CIELAB L, a, b
HIST_BINS = 256


def image_stats(rgb: np.ndarray, hists: list | None = None) -> dict:
    """The appearance statistics of one RGB uint8 image (at the working width); hists (three arrays of HIST_BINS)
    accumulates the image's CIELAB histograms."""
    import cv2
    im = working(rgb)
    lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    if hists is not None:
        for c, (lo, hi) in enumerate(HIST_RANGE):
            hists[c] += np.histogram(lab[..., c], HIST_BINS, (lo, hi))[0]
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return {'L_mean': float(L.mean()), 'L_std': float(L.std()), 'a_mean': float(a.mean()), 'b_mean': float(b.mean()),
            'ab_std': float(math.sqrt((a.var() + b.var()) / 2)), 'saturation': float(hsv[..., 1].mean() / 255.0),
            'dark_share': float((L < 2).mean()), 'bright_share': float((im == 255).any(-1).mean()),
            'noise_sigma': (ns := noise_sigma(gray)), 'sharpness': sharpness(gray, ns),
            'spectrum_slope': spectrum_slope(gray)}


def _summary(v) -> dict:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {'n': 0}
    p = np.percentile(v, [10, 25, 50, 75, 90])
    return {'n': int(len(v)), 'mean': float(v.mean()), 'p10': float(p[0]), 'p25': float(p[1]), 'median': float(p[2]),
            'p75': float(p[3]), 'p90': float(p[4])}


def set_stats(s: dict, transform=None, progress=None, keep_images: bool = False) -> dict:
    """Per-image appearance statistics and the objects' statistics of a set read by read_set. transform(rgb) ->
    rgb is applied first (appearance matching). keep_images keeps the (transformed) images for feature
    extraction."""
    per = {k: [] for k in IMAGE_STATS}
    hists = [np.zeros(HIST_BINS, np.int64) for _ in HIST_RANGE]
    heights, ranges, per_image, classes = [], [], [], {}
    images = [] if keep_images else None
    sizes = []
    n = len(s['items'])
    for i, it in enumerate(s['items']):
        rgb = load_rgb(it['image'])
        if transform is not None:
            rgb = transform(rgb, i)
        for k, v in image_stats(rgb, hists).items():
            per[k].append(v)
        H, W = rgb.shape[:2]
        sizes.append((W, H))
        if images is not None:
            images.append(working(rgb))         # kept at the working width (memory); stats are taken there anyway
        if it['objects'] is not None:
            kept = 0
            for o in it['objects']:
                x0, y0, x1, y1 = o['bbox']
                if (y1 - y0) / H < MIN_BOX_SHARE:
                    continue
                kept += 1
                classes[o['class']] = classes.get(o['class'], 0) + 1
                heights.append((y1 - y0) / H)
                if o.get('range_m') is not None:
                    ranges.append(float(o['range_m']))
            per_image.append(kept)
        if progress:
            progress((i + 1) / n, f'{i + 1}/{n} images')
    out = {'n_images': n, 'image_sizes': sorted({f'{w}x{h}' for w, h in sizes}), 'per_image': per,
           'summary': {k: _summary(v) for k, v in per.items()}, 'lab_hist': [h.tolist() for h in hists]}
    if s.get('labelled'):
        tot = sum(classes.values())
        out['objects'] = {'n': tot, 'class_share': {c: classes[c] / tot for c in sorted(classes)} if tot else {},
                          'box_height_share': heights, 'range_m': ranges, 'per_image': per_image,
                          'summary': {'box_height_share': _summary(heights), 'range_m': _summary(ranges),
                                      'per_image': _summary(per_image)}}
    if images is not None:
        out['images'] = images
    return out


def _w1(a, b) -> float:
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b))


def _gap_row(name, a, b, note='') -> dict | None:
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    b = np.asarray([x for x in b if x is not None and np.isfinite(x)], float)
    if len(a) == 0 or len(b) == 0:
        return None
    iqr = float(np.subtract(*np.percentile(b, [75, 25])))
    w1 = _w1(a, b)
    return {'stat': name, 'note': note, 'sim_median': float(np.median(a)), 'real_median': float(np.median(b)),
            'w1': w1, 'shift': w1 / iqr if iqr > 1e-9 else None, 'n_sim': int(len(a)), 'n_real': int(len(b))}


def compare(sim: dict, real: dict) -> dict:
    """The gap between two set_stats results: appearance rows sorted by shift (largest first) and, when both are
    labelled, the objects' gaps (class shares: total-variation distance)."""
    rows = [r for r in (_gap_row(k, sim['per_image'][k], real['per_image'][k], STAT_NOTES[k]) for k in IMAGE_STATS)
            if r]
    rows.sort(key=lambda r: -(r['shift'] if r['shift'] is not None else -1))
    out = {'appearance': rows}
    if 'objects' in sim and 'objects' in real:
        so, ro = sim['objects'], real['objects']
        cls = sorted(set(so['class_share']) | set(ro['class_share']))
        share = {c: {'sim': so['class_share'].get(c, 0.0), 'real': ro['class_share'].get(c, 0.0)} for c in cls}
        obj = {'class_share': share,
               'class_share_tvd': 0.5 * sum(abs(v['sim'] - v['real']) for v in share.values()),
               'rows': [r for r in (_gap_row('box_height_share', so['box_height_share'], ro['box_height_share'],
                                             'box height / image height'),
                                    _gap_row('per_image', so['per_image'], ro['per_image'], 'objects per image'),
                                    _gap_row('range_m', so['range_m'], ro['range_m'], 'object range, m'))
                        if r]}
        out['objects'] = obj
    return out


# ================================================================================================ features
def features_path(workspace: str, name: str = DEFAULT_FEATURES) -> str:
    return os.path.join(workspace, 'models', 'features', FEATURES[name]['file'])


def fetch_features(workspace: str, name: str = DEFAULT_FEATURES, progress=None) -> str:
    """Download the feature extractor's weights (only when asked) and check their SHA-256."""
    from .models import fetch
    f = FEATURES[name]
    return fetch({'path': features_path(workspace, name), 'source': {'url': f['url'], 'sha256': f['sha256']}},
                 workspace, progress)


class Features:
    """The fixed feature extractor: pooled ResNet-50 features (2048) of the whole image squashed to 224 × 224 (no
    crop: a road scene's edges matter), ImageNet normalisation; CUDA when available."""

    def __init__(self, workspace: str, name: str = DEFAULT_FEATURES):
        import torch
        import torchvision
        from .provenance import sha256_file
        p = features_path(workspace, name)
        if not os.path.isfile(p):
            raise GapError(f'the feature extractor {name} is not downloaded ({p}); fetch it first')
        if sha256_file(p) != FEATURES[name]['sha256']:
            raise GapError(f'{p} does not match its pinned SHA-256')
        net = torchvision.models.resnet50(weights=None)
        net.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
        net.fc = torch.nn.Identity()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.eval().to(self.dev)
        self.name = name
        self.size = FEATURES[name]['input']
        self.torch = torch

    def __call__(self, images: list, batch: int = 32) -> np.ndarray:
        import cv2
        torch = self.torch
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        out = []
        for i in range(0, len(images), batch):
            x = np.stack([(cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32)
                           / 255.0 - mean) / std for im in images[i:i + batch]]).transpose(0, 3, 1, 2)
            with torch.no_grad():
                out.append(self.net(torch.from_numpy(np.ascontiguousarray(x)).to(self.dev)).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def frechet(f1: np.ndarray, f2: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to two feature sets (the method of FID)."""
    from scipy import linalg
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    with warnings.catch_warnings():        # fewer images than feature dimensions: singular covariances, expected
        warnings.simplefilter('ignore')
        cov = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(cov).all():
        eps = np.eye(len(s1)) * 1e-6
        cov = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    cov = np.real(cov)
    d = mu1 - mu2
    return float(d @ d + np.trace(s1) + np.trace(s2) - 2 * np.trace(cov))


def kernel_distance(f1: np.ndarray, f2: np.ndarray, subsets: int = 100, size: int = 1000, seed: int = 0) -> tuple:
    """Unbiased squared MMD with the cubic polynomial kernel (the method of KID): (mean, std over subsets). Subsets
    hold `size` images, or half the smaller set when it has fewer than 2 × size (so that they differ)."""
    rng = np.random.default_rng(seed)
    d = f1.shape[1]
    n = min(len(f1), len(f2))
    m = size if n >= 2 * size else max(2, n // 2)
    vals = []
    for _ in range(subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def feature_gap(f_sim: np.ndarray, f_real: np.ndarray, seed: int = 0) -> dict:
    """FD and KD of the simulated set against one random half of the real set, and their floor: the same between the
    two real halves, at the same set sizes (so the bias of small sets is the same on both)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(f_real))
    half = len(f_real) // 2
    if half < 10:
        raise GapError('feature distances need at least 20 real images')
    ra, rb = f_real[idx[:half]], f_real[idx[half:2 * half]]
    n = min(half, len(f_sim))
    fs = f_sim[rng.choice(len(f_sim), n, replace=False)]
    ra_n = ra[:n]
    kd = kernel_distance(fs, rb, seed=seed)
    kd0 = kernel_distance(ra_n, rb, seed=seed)
    return {'fd': frechet(fs, rb), 'fd_floor': frechet(ra_n, rb), 'kd': kd[0], 'kd_std': kd[1], 'kd_floor': kd0[0],
            'kd_floor_std': kd0[1], 'n_sim': int(n), 'n_real': int(len(rb))}


# ================================================================================================ appearance matching
def fit_appearance(sim: dict, real: dict) -> dict:
    """A fixed transform of simulated images towards the real set, from the two sets' statistics: histogram
    matching of each CIELAB channel at the level of the sets (one lookup table from the simulated set's pooled
    histogram to the real set's, the same for every image, so that images keep their differences), a blur fitted to
    the real median sharpness, noise added up to the real median noise level."""
    import cv2
    rs, ss = real['summary'], sim['summary']
    luts = []
    for c, (lo, hi) in enumerate(HIST_RANGE):
        centres = lo + (np.arange(HIST_BINS) + 0.5) * (hi - lo) / HIST_BINS
        cs = np.cumsum(sim['lab_hist'][c]) / max(1, np.sum(sim['lab_hist'][c]))
        ct = np.cumsum(real['lab_hist'][c]) / max(1, np.sum(real['lab_hist'][c]))
        ct_u, first = np.unique(ct, return_index=True)          # strictly increasing for the inverse
        luts.append(np.round(np.interp(cs, ct_u, centres[first]), 4).tolist())
    gap = rs['noise_sigma']['median'] ** 2 - ss['noise_sigma']['median'] ** 2
    p = {'lab_lut': luts, 'blur_sigma': 0.0, 'noise_sigma': round(math.sqrt(gap), 3) if gap > 0 else 0.0}
    target = rs['sharpness']['median']
    rng = np.random.default_rng(0)
    if 'images' in sim:
        spread = sim['images'][::max(1, len(sim['images']) // 12)][:12]
        probe = [apply_appearance(working(im), {'lab_lut': luts}).astype(np.float32) for im in spread]
        noise = [rng.normal(0, p['noise_sigma'], im.shape[:2]).astype(np.float32)[..., None] for im in probe]

        def sharp(sig):            # as apply_appearance does it, with its noise
            v = []
            for im, nz in zip(probe, noise):
                b = cv2.GaussianBlur(im, (0, 0), sig) if sig > 0 else im
                g = cv2.cvtColor(np.clip(np.round(b + nz), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                v.append(sharpness(g))
            return float(np.median(v))
        lo, hi = 0.0, 3.0
        if sharp(hi) > target:
            p['blur_note'] = ('not blurred: the real set is softer than the largest blur tried (σ 3 px at the working '
                              'width)')
        else:
            for _ in range(12):                # sharpness falls monotonically with the blur
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if sharp(mid) > target else (lo, mid)
            p['blur_sigma'] = round((lo + hi) / 2, 3) if lo > 0 else 0.0
    return p


def apply_appearance(rgb: np.ndarray, p: dict, seed: int = 0) -> np.ndarray:
    import cv2
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    for c, (lo, hi) in enumerate(HIST_RANGE):
        centres = lo + (np.arange(HIST_BINS) + 0.5) * (hi - lo) / HIST_BINS
        lab[..., c] = np.interp(lab[..., c], centres, p['lab_lut'][c]).astype(np.float32)
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0, 0, 255)
    if p.get('blur_sigma'):
        # the blur is fitted at the working width; scaled to this image's width
        out = cv2.GaussianBlur(out, (0, 0), p['blur_sigma'] * rgb.shape[1] / WORK_WIDTH)
    if p.get('noise_sigma'):
        out = out + np.random.default_rng(seed).normal(0, p['noise_sigma'], out.shape[:2])[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ================================================================================================ the whole study
def _strip(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k not in ('images', 'per_image', 'lab_hist')} | (
        {'objects': {k: v for k, v in stats['objects'].items() if k in ('n', 'class_share', 'summary')}}
        if 'objects' in stats else {})


def study(sim_sets: list, real_set: dict, workspace: str = '', features: bool = False, match: bool = False,
          seed: int = 0, progress=None) -> dict:
    """The gap document: both sets' statistics, the appearance and object gaps, the feature distances (with
    features) and, with match, the same after appearance matching."""
    def prog(lo, hi, label):
        return (lambda f, t: progress(lo + (hi - lo) * f, f'{label}: {t}')) if progress else None
    sim = {'kind': 'falcon', 'labelled': all(s['labelled'] for s in sim_sets),
           'items': [it for s in sim_sets for it in s['items']]}
    keep = features or match
    s_stats = set_stats(sim, progress=prog(0.0, 0.35, 'simulated'), keep_images=keep)
    r_stats = set_stats(real_set, progress=prog(0.35, 0.7, 'real'), keep_images=features)
    doc = {'schema': SCHEMA, 'sim': {'sources': [{'path': s['path'], 'camera': s.get('camera'), 'n_total':
                                                  s['n_total'], 'notes': s['notes']} for s in sim_sets],
                                     **_strip(s_stats)},
           'real': {'kind': real_set['kind'], 'path': real_set['path'], 'n_total': real_set['n_total'],
                    'notes': real_set['notes'], **_strip(r_stats)},
           'gap': compare(s_stats, r_stats), 'features': None, 'matched': None,
           'method': {'work_width': WORK_WIDTH, 'min_box_share': MIN_BOX_SHARE,
                      'shift': 'W1 / real IQR (heuristic scale)',
                      'seed': seed}}
    extractor = None
    if features:
        extractor = Features(workspace)
        f_real = extractor(r_stats['images'])
        f_sim = extractor(s_stats['images'])
        doc['features'] = {'extractor': extractor.name, 'device': extractor.dev,
                           'licence': FEATURES[extractor.name]['licence'], **feature_gap(f_sim, f_real, seed)}
    if progress:
        progress(0.8, 'features' if features else 'statistics done')
    if match:
        p = fit_appearance(s_stats, r_stats)
        per = {k: [] for k in IMAGE_STATS}
        matched = []
        for i, im in enumerate(s_stats['images']):          # the kept simulated images, transformed (no re-read)
            mi = apply_appearance(im, p, seed + i)
            for k, v in image_stats(mi).items():
                per[k].append(v)
            if features:
                matched.append(mi)
        m_stats = {'per_image': per, 'summary': {k: _summary(v) for k, v in per.items()}}
        doc['matched'] = {'params': p, 'summary': m_stats['summary'],
                          'gap': compare(m_stats, r_stats)['appearance']}
        if features:
            doc['matched']['features'] = feature_gap(extractor(matched), f_real, seed)
    if progress:
        progress(1.0, 'done')
    return doc


# ================================================================================================ the command
def main_gap(argv) -> int:
    """`falcon gap --sim RUN... --real KIND:PATH --out DIR [--labels FILE] [--camera C] [--real-camera C]
    [--images N] [--seed S] [--features --workspace W] [--match]`, or `falcon gap --fetch --workspace W` (download the
    feature extractor, its URL, size and licence shown first): 0 on success, 2 on failure."""
    import argparse
    from . import provenance as PV
    ap = argparse.ArgumentParser(prog='falcon gap', description='measure the sim-to-real gap of camera images')
    ap.add_argument('--sim', action='append', default=[], help='a Falcon run folder (repeatable)')
    ap.add_argument('--camera', help='the runs\' camera (default: the first pinhole camera)')
    ap.add_argument('--real', default='', metavar='KIND:PATH', help=f'the real set; KIND one of {", ".join(KINDS)}')
    ap.add_argument('--labels', help='the real set\'s labels JSON (bdd100k, coco)')
    ap.add_argument('--real-camera', help='the real set\'s camera (nuscenes: a channel, default CAM_FRONT)')
    ap.add_argument('--images', type=int, default=300, help='images per side, a seeded random subset (default 300)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--features', action='store_true', help='Fréchet and kernel distances (needs the extractor '
                                                           'fetched into the workspace: falcon gap --fetch)')
    ap.add_argument('--match', action='store_true', help='also measure after appearance matching')
    ap.add_argument('--workspace', default='', help='the workspace holding models/features/')
    ap.add_argument('--out', '-o', default='')
    ap.add_argument('--fetch', action='store_true', help='download the feature extractor into the workspace')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if a.fetch:
        if not a.workspace:
            print('failed: --fetch needs --workspace', flush=True)
            return 2
        f = FEATURES[DEFAULT_FEATURES]
        print(f'downloading {DEFAULT_FEATURES} ({f["mb"]} MB) from {f["url"]}; licence: {f["licence"]}', flush=True)
        try:
            p = fetch_features(a.workspace, progress=lambda x: print(f'{100 * x:5.1f} % download', flush=True))
        except (OSError, ValueError) as e:
            print(f'failed: {e}', flush=True)
            return 2
        print(f'installed {p} (SHA-256 checked)', flush=True)
        return 0
    if not a.sim or not a.out:
        print('failed: --sim and --out are needed', flush=True)
        return 2
    kind, sep, path = a.real.partition(':')
    if not sep:
        print('failed: --real is KIND:PATH', flush=True)
        return 2
    if a.features and not a.workspace:
        print('failed: --features needs --workspace', flush=True)
        return 2
    settings = {k: v for k, v in vars(a).items() if k != 'out'}
    run = PV.new_run('gap', ['gap'] + list(argv), a.seed, settings,
                     {f'sim{i}/run.json': os.path.join(p, 'run.json') for i, p in enumerate(a.sim)}
                     | ({'real/labels': a.labels} if a.labels else {}))
    os.makedirs(a.out, exist_ok=True)

    def prog(f, text):
        print(f'{100 * f:5.1f} % {text}', flush=True)
    try:
        sims = [read_set('falcon', p, camera=a.camera, limit=a.images, seed=a.seed + i) for i, p in enumerate(a.sim)]
        if len(sims) > 1:      # the whole simulated side holds --images, shared out over the runs
            share = max(1, a.images // len(sims))
            sims = [dict(s, items=_sample(s['items'], share, a.seed + i)) for i, s in enumerate(sims)]
        real = read_set(kind, path, labels=a.labels, camera=a.real_camera, limit=a.images, seed=a.seed)
        doc = study(sims, real, a.workspace, a.features, a.match, a.seed, prog)
    except (GapError, OSError, ValueError) as e:
        run.update(status='failed', exit_code=2, error=str(e), finished=PV.utc_now())
        PV.write_json(os.path.join(a.out, 'run.json'), run)
        print(f'failed: {e}', flush=True)
        return 2
    doc['provenance'] = run['provenance']
    PV.write_json(os.path.join(a.out, 'gap.json'), doc)
    run.update(status='ok', exit_code=0, finished=PV.utc_now(), outputs=PV.hash_tree(a.out))
    PV.write_json(os.path.join(a.out, 'run.json'), run)
    top = doc['gap']['appearance'][:3]
    print('largest appearance gaps: ' + ', '.join(f'{r["stat"]} (shift {r["shift"]:.2f})' for r in top
                                                 if r['shift'] is not None), flush=True)
    if doc['features']:
        f = doc['features']
        print(f'FD {f["fd"]:.1f} (floor {f["fd_floor"]:.1f}), KD {f["kd"]:.4f} ± {f["kd_std"]:.4f} (floor '
              f'{f["kd_floor"]:.4f}), {f["n_sim"]} vs {f["n_real"]} images', flush=True)
    print(f'wrote {os.path.join(a.out, "gap.json")}', flush=True)
    return 0
