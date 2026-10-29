"""Posed camera images for a reconstruction, read from a log's source in place (numpy and OpenCV; scipy for the point
initialisation): a Falcon run, a KITTI raw drive or a nuScenes release — never copied into the project.

A Capture is the frames of one camera (each: an image, its intrinsics K = (fx, fy, cx, cy) in pixel-edge coordinates,
its pose and a mask of the pixels a static reconstruction may learn from), the points to start the Gaussians from,
and what was read, left out or assumed.

Frames (docs/reconstruction.md, "Frames"). The reconstruction's world frame is right-handed with z up and is the
source's own: a Falcon run's CARLA world mirrored (x, −y, z: core.exporters.rh), KITTI raw's local east-north-up about
the first OXTS fix (the transverse Mercator of core.logreplay.sources, altitude from the first fix), nuScenes' global
frame. CARLA's convention of any of these is the mirror S = diag(1, −1, 1), as in log replay, so an ego path shifted
here is written in a falcon-raw run exactly as log replay would write it. Per frame: viewmat (4 × 4, world → the
camera's optical frame: x right, y down, z forward), ego (world ← the vehicle frame: CARLA's vehicle origin mirrored
for a Falcon run, the OXTS unit for KITTI, nuScenes' ego frame) and mount (vehicle ← optical).

Sources:

* **falcon** — frames.jsonl and rgb/<camera>/; the pose from the ego pose and the camera's actual mount (its mount
  faults included, as CARLA rendered it); K from the camera's field of view; pinhole cameras only (a fisheye would
  need another projection). Masks: every pixel whose instance belongs to an actor (an object with a CARLA actor id:
  every vehicle and walker, moving or not) and the ego's own body (class 29) is left out; map meshes (parked cars
  included) stay. Lidar: every sweep's static returns (actor id 0, not the ego, not weather noise) in the world through
  frames.jsonl's lidar_pose. The run's semantic labels are kept for voting the Gaussians' classes, its range labels
  for evaluation only (never for training, unless init='range' is asked for — a simulator's privilege, labelled).
* **kitti_raw** — image_02/ (the left colour camera, rectified) or image_03/; poses from OXTS as
  core.logreplay.sources reads them (R = Rz(yaw)·Ry(pitch)·Rx(roll), the KITTI devkit's convertOxtsToPose) and the
  calibration in the date folder: cam2 ← world = [I | K⁻¹P_rect_02's offset] · R_rect_00 · T_cam0←velo ·
  T_velo←imu · T_imu←world (the devkit's chain). Velodyne sweeps (velodyne_points/) for the points, coloured from the
  same frame's image. **Not masked**: the raw drives' tracklets are a separate download (not read), so moving
  vehicles stay in the images and blur the reconstruction where they pass — said in every result. The velodyne sweeps
  are not motion-compensated (KITTI raw's own limit).
* **nuscenes** — one camera channel (CAM_FRONT by default) of a scene, its key frames (annotated, 2 Hz); the pose
  from the sample_data's ego_pose and the calibrated sensor; masks from the key frame's annotated boxes of Falcon's
  actor classes (core.logreplay.ros2.falcon_class), each 3D box projected (its edges clipped at the near plane) and
  filled, dilated by 2 px; LIDAR_TOP key-frame sweeps (.pcd.bin, five floats a point) for the points. Supported by
  the format and tested on synthetic files only: no nuScenes data is on this machine.

Initialisation (init): 'lidar' (the source's lidar, voxel-averaged), 'sfm' (structure from the known poses: SIFT
features (Lowe 2004) matched between frames about baseline_m apart with Lowe's ratio test, triangulated with the
known projection matrices and kept when both reprojection errors are below 1.5 px, the ray angle above 1° and the
depth within 0.5–200 m — heuristic thresholds), 'range' (Falcon runs only: the ground-truth range image, sparsely
sampled) or 'random'. 'auto' is lidar when the source has it, else sfm. Every initialisation adds a far shell of
Gaussians (marked sky) around the cameras — 1.5 × the 99th percentile of the points' distance, at least 100 m — so the
sky and the far background have something to become (heuristic).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

S3 = np.diag([1.0, -1.0, 1.0])
S4 = np.diag([1.0, -1.0, 1.0, 1.0])
# CARLA's actor axes (x forward, y right, z up) -> the optical frame (x right, y down, z forward)
CARLA_TO_OPTICAL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
DYNAMIC_CLASSES = (12, 13, 14, 15, 16, 17, 18, 19)      # CARLA tags: pedestrian, rider, car … bicycle
EGO_CLASS = 29
SKY_CLASS = 11
INIT_KINDS = ('auto', 'lidar', 'sfm', 'range', 'random')


class DataError(ValueError):
    pass


@dataclass
class Frame:
    index: int                    # position in the source's sequence (a Falcon step, a KITTI frame, a sample)
    t: float                      # seconds from the capture's first frame
    image: object                 # a path, or an (H, W, 3) uint8 RGB array
    K: tuple                      # (fx, fy, cx, cy), pixel-edge, at the image's own size
    size: tuple                   # (width, height) of the image
    viewmat: np.ndarray           # world -> optical
    ego: np.ndarray               # world <- vehicle
    mount: np.ndarray             # vehicle <- optical
    mask: object = None           # a callable () -> (H, W) bool (True: learn from it), an array, or None (all)
    semantic: object = None       # a path or an (H, W) uint8 array of classes, or None
    range: object = None          # a path (uint16 PNG, range × 256) or an (H, W) float array (m), or None
    name: str = ''

    @property
    def centre(self) -> np.ndarray:
        return -self.viewmat[:3, :3].T @ self.viewmat[:3, 3]


@dataclass
class Capture:
    kind: str
    path: str
    camera: str
    frames: list
    lidar: object = None          # a callable () -> (xyz (N, 3), rgb (N, 3) in [0, 1]) or None
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        c = np.array([f.centre for f in self.frames])
        return {'kind': self.kind, 'path': self.path, 'camera': self.camera, 'frames': len(self.frames),
                'image_size': list(self.frames[0].size) if self.frames else None,
                'path_m': round(float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))), 3) if len(c) > 1 else 0.0,
                'lidar': self.lidar is not None, 'masks': self.meta.get('masks', ''),
                'world': self.meta.get('world', '')}


# ================================================================================================ images
def load_rgb(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    import cv2
    bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
    if bgr is None:
        raise DataError(f'cannot read the image {img}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(fr: Frame):
    m = fr.mask() if callable(fr.mask) else fr.mask
    return None if m is None else np.asarray(m, bool)


def load_semantic(fr: Frame):
    if fr.semantic is None:
        return None
    if isinstance(fr.semantic, np.ndarray):
        return fr.semantic
    import cv2
    return cv2.imread(str(fr.semantic), cv2.IMREAD_UNCHANGED)


def load_range(fr: Frame):
    """The frame's range (m along each pixel's ray; 0 = nothing within 256 m), or None."""
    if fr.range is None:
        return None
    if isinstance(fr.range, np.ndarray):
        return fr.range.astype(np.float32)
    import cv2
    r = cv2.imread(str(fr.range), cv2.IMREAD_UNCHANGED)
    return None if r is None else r.astype(np.float32) / 256.0


def working_size(size, width: int | None) -> tuple:
    """(w, h) of an image of `size` scaled to `width` (never up)."""
    w, h = size
    if not width or width >= w:
        return int(w), int(h)
    return int(width), max(1, int(round(h * width / w)))


def resize_rgb(img: np.ndarray, wh) -> np.ndarray:
    import cv2
    if (img.shape[1], img.shape[0]) == tuple(wh):
        return img
    return cv2.resize(img, tuple(wh), interpolation=cv2.INTER_AREA)


def resize_mask(m, wh):
    """A learn-from mask at another size: a pixel is usable only when everything it covers is (area resampling)."""
    import cv2
    if m is None or (m.shape[1], m.shape[0]) == tuple(wh):
        return m
    return cv2.resize(m.astype(np.float32), tuple(wh), interpolation=cv2.INTER_AREA) > 0.999


def resize_labels(m, wh):
    import cv2
    if m is None or (m.shape[1], m.shape[0]) == tuple(wh):
        return m
    return cv2.resize(m, tuple(wh), interpolation=cv2.INTER_NEAREST)


def scaled_K(K, size, wh) -> tuple:
    sx, sy = wh[0] / size[0], wh[1] / size[1]
    return (K[0] * sx, K[1] * sy, K[2] * sx, K[3] * sy)


def inv(T) -> np.ndarray:
    T = np.asarray(T, float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


# ================================================================================================ sources
def parse_source(text: str) -> tuple:
    """'KIND:PATH[?k=v&…]' (or a bare path, its kind detected) -> (kind, path, options)."""
    from urllib.parse import parse_qsl
    kind, path = None, text
    head, sep, rest = text.partition(':')
    if sep and head in ('falcon', 'kitti_raw', 'nuscenes') and not (len(head) == 1 and rest.startswith('\\')):
        kind, path = head, rest
    path, _, q = path.partition('?')
    opts = dict(parse_qsl(q))
    path = os.path.abspath(os.path.expanduser(path))
    if kind is None:
        from ..logreplay import detect_kind
        kind = detect_kind(path)
        if kind == 'kitti_raw' and os.path.basename(path) == 'oxts':
            path = os.path.dirname(path)
    if kind not in ('falcon', 'kitti_raw', 'nuscenes'):
        raise DataError(f'{text}: a reconstruction reads posed images from a Falcon run, a KITTI raw drive or a '
                        f'nuScenes release (found: {kind or "nothing it knows"}); ROS 2 bags and Waymo logs are read '
                        f'for trajectories only')
    return kind, path, opts


def open_capture(source, camera: str | None = None, **opts) -> Capture:
    """A Capture from a source string (parse_source) or (kind, path)."""
    if isinstance(source, str):
        kind, path, o = parse_source(source)
        o.update({k: v for k, v in opts.items() if v is not None})
    else:
        kind, path = source
        o = dict(opts)
    camera = camera or o.pop('camera', None)
    if kind == 'falcon':
        return falcon_capture(path, camera, every=int(o.get('every', 1)))
    if kind == 'kitti_raw':
        return kitti_capture(path, camera or 'image_02', every=int(o.get('every', 1)))
    return nuscenes_capture(path, o.get('scene'), camera or 'CAM_FRONT')


# ------------------------------------------------------------------------------------------------ Falcon runs
def falcon_capture(path: str, camera: str | None = None, every: int = 1) -> Capture:
    from ..exporters import ExportError, Run, camera_T, rh
    try:
        run = Run(path)
    except ExportError as e:
        raise DataError(str(e)) from None
    pins = [c for c, s in run.cams.items() if s['type'] == 'camera.pinhole']
    cid = camera or (pins[0] if pins else None)
    if cid not in run.cams:
        raise DataError(f'{path}: no camera {camera!r} (cameras: {", ".join(run.cams) or "none"})')
    cam = run.cams[cid]
    if cam['type'] != 'camera.pinhole':
        raise DataError(f'{path}: camera {cid} is a {cam["type"]}; a reconstruction reads pinhole cameras only')
    W, H = int(cam['width']), int(cam['height'])
    Kc = cam.get('K') or [[cam['width'] / (2 * math.tan(math.radians(cam['fov']) / 2)), 0, W / 2],
                          [0, cam['width'] / (2 * math.tan(math.radians(cam['fov']) / 2)), H / 2]]
    K = (float(Kc[0][0]), float(Kc[1][1]), float(Kc[0][2]), float(Kc[1][2]))
    mount = camera_T(run, cid, actual=True)
    recs = run.camera_steps(cid, every)
    if not recs:
        raise DataError(f'{path}: camera {cid} has no frames')
    t0 = float(recs[0]['sim_t'])
    frames = []
    for r in recs:
        ego = rh(run.ego(r))
        Tw = ego @ mount
        frames.append(Frame(int(r['step']), float(r['sim_t']) - t0, run.p(run.file(r, cid, 'rgb')), K, (W, H),
                            inv(Tw), ego, mount, _falcon_mask(run, r, cid),
                            run.p(run.file(r, cid, 'semantic')) if run.own(r, cid, 'semantic') else None,
                            run.p(run.file(r, cid, 'range')) if run.own(r, cid, 'range') else None,
                            f'{run.name}/{r["step"]:06d}'))
    lidars = [s['id'] for s in run.of_type('lidar')] or \
        sorted({k.split('/')[0] for r in run.frames for k in r.get('files', {}) if k.endswith('/lidar')})
    lid = lidars[0] if lidars else None
    has_lidar = lid is not None and any(run.file(r, lid, 'lidar') and r.get('lidar_pose', {}).get(lid)
                                        for r in run.frames)
    has_inst = any(run.own(r, cid, 'instance') for r in recs)
    meta = {'world': 'the run\'s CARLA world mirrored (x, −y, z): right-handed, z up, metres',
            'town': run.cfg.get('town') or run.cfg.get('xodr') or '', 'name': run.name, 'fps': run.fps,
            'masks': 'actors (instance labels: every vehicle and walker) and the ego\'s body' if has_inst else
            'none (the run has no instance labels)',
            'carla_exact': True, 'weather': run.cfg.get('weather'), 'vehicle': run.rig.get('vehicle'),
            'sensor': cam, 'inputs': run.inputs(), 'notes': [], 'left_out': [], 'heuristic': []}
    cap = Capture('falcon', run.dir, cid, frames, None, meta)
    if has_lidar:
        cap.lidar = lambda: _falcon_lidar(run, lid, cap)
        meta['lidar'] = lid
    return cap


def _falcon_mask(run, rec, cid):
    def mk():
        import cv2
        rel_i = run.own(rec, cid, 'instance')
        rel_s = run.own(rec, cid, 'semantic')
        m = None
        if rel_i:
            inst = cv2.imread(run.p(rel_i), cv2.IMREAD_UNCHANGED)
            objs = run.objects(rec, cid) or []
            ids = np.array([o['id'] for o in objs if isinstance(o.get('actor_id'), int) or
                            (isinstance(o.get('track_id'), int))], np.int64)
            m = ~np.isin(inst, ids) if len(ids) else np.ones(inst.shape, bool)
        if rel_s:
            sem = cv2.imread(run.p(rel_s), cv2.IMREAD_UNCHANGED)
            ego = sem != EGO_CLASS
            if m is None:                     # no instance labels: the dynamic classes (parked map cars too)
                m = ego & ~np.isin(sem, DYNAMIC_CLASSES)
            else:
                m &= ego
        return m
    return mk


def _falcon_lidar(run, lid, cap, max_sweeps: int = 60):
    recs = [r for r in run.frames if run.file(r, lid, 'lidar') and (r.get('lidar_pose') or {}).get(lid)]
    recs = recs[::max(1, int(math.ceil(len(recs) / max_sweeps)))]
    by_step = {f.index: f for f in cap.frames}
    steps = np.array(sorted(by_step))
    xs, cs = [], []
    for r in recs:
        p = run.p(run.file(r, lid, 'lidar'))
        pts = np.fromfile(p, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        lp = p[:-4] + '.labels.bin'
        if os.path.isfile(lp):
            lab = np.fromfile(lp, np.uint8)
            n = len(pts)
            if len(lab) >= 5 * n:
                tag = lab[:n]
                aid = np.frombuffer(lab[n:5 * n].tobytes(), np.uint32)
                pts = pts[(aid == 0) & (tag != EGO_CLASS) & (tag != 30)]
        M = np.asarray(r['lidar_pose'][lid], float).reshape(3, 4)
        w = (pts @ M[:, :3].T + M[:, 3]) @ S3.T
        fr = by_step[int(steps[np.argmin(np.abs(steps - r['step']))])]
        xs.append(w)
        cs.append(colour_points(w, nearest_frames(cap.frames, fr.t)))
    return np.concatenate(xs), np.concatenate(cs)


# ------------------------------------------------------------------------------------------------ KITTI raw
def _kitti_calib(path: str) -> dict:
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            k, sep, v = line.partition(':')
            if not sep:
                continue
            try:
                out[k.strip()] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass
    return out


def _rt(d) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = d['R'].reshape(3, 3)
    T[:3, 3] = d['T']
    return T


def kitti_capture(path: str, camera: str = 'image_02', every: int = 1) -> Capture:
    from ..osm import local_projection
    path = os.path.abspath(path)
    if os.path.basename(path) == 'oxts':
        path = os.path.dirname(path)
    date = os.path.dirname(path)
    cc, vc, ic = (os.path.join(date, f) for f in ('calib_cam_to_cam.txt', 'calib_velo_to_cam.txt',
                                                  'calib_imu_to_velo.txt'))
    for f in (cc, vc, ic):
        if not os.path.isfile(f):
            raise DataError(f'{path}: no {os.path.basename(f)} in the date folder {date} (KITTI raw calibration)')
    if camera not in ('image_02', 'image_03'):
        raise DataError(f'KITTI raw: the colour cameras are image_02 and image_03, not {camera!r}')
    img_dir = os.path.join(path, camera, 'data')
    ox_dir = os.path.join(path, 'oxts', 'data')
    if not (os.path.isdir(img_dir) and os.path.isdir(ox_dir)):
        raise DataError(f'{path}: no {camera}/data and oxts/data (a KITTI raw *_sync drive)')
    cam = _kitti_calib(cc)
    n = camera[-1]
    P = cam[f'P_rect_0{n}'].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = cam['R_rect_00'].reshape(3, 3)
    Kcv = P[:, :3]
    off = np.eye(4)
    off[:3, 3] = np.linalg.solve(Kcv, P[:, 3])
    T_cam_velo = off @ R0 @ _rt(_kitti_calib(vc))
    T_velo_imu = _rt(_kitti_calib(ic))
    T_cam_imu = T_cam_velo @ T_velo_imu
    mount = inv(T_cam_imu)                             # imu (the vehicle frame) <- optical
    sw, sh = (int(v) for v in cam[f'S_rect_0{n}'])
    K = (float(Kcv[0, 0]), float(Kcv[1, 1]), float(Kcv[0, 2]) + 0.5, float(Kcv[1, 2]) + 0.5)
    imgs = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
    oxs = sorted(f for f in os.listdir(ox_dir) if f.endswith('.txt'))
    if len(imgs) != len(oxs):
        raise DataError(f'{path}: {len(imgs)} images but {len(oxs)} OXTS packets')
    pk = np.array([[float(v) for v in open(os.path.join(ox_dir, f), encoding='utf-8').read().split()[:6]]
                   for f in oxs])
    proj = local_projection(float(pk[0, 0]), float(pk[0, 1]))
    e, nn = proj.to_local(pk[:, 0], pk[:, 1])
    stamps = _kitti_stamps(os.path.join(path, camera, 'timestamps.txt'), len(imgs))
    frames = []
    for k in range(0, len(imgs), max(1, every)):
        roll, pitch, yaw = pk[k, 3:6]
        cr, sr, cp, spp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                                   math.sin(yaw))
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, spp], [0, 1, 0], [-spp, 0, cp]]) @ \
            np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ego = np.eye(4)
        ego[:3, :3] = R
        ego[:3, 3] = [e[k], nn[k], pk[k, 2] - pk[0, 2]]
        frames.append(Frame(k, stamps[k] - stamps[0], os.path.join(img_dir, imgs[k]), K, (sw, sh),
                            inv(ego @ mount), ego, mount, None, None, None, f'{os.path.basename(path)}/{k:010d}'))
    from ..provenance import sha256_file
    meta = {'world': 'local east-north-up about the first OXTS fix (transverse Mercator), metres; altitude from the '
                     'first fix', 'geo': {'lat0': float(pk[0, 0]), 'lon0': float(pk[0, 1])},
            'name': os.path.basename(path), 'masks': 'none: KITTI raw tracklets are not read, so moving vehicles '
                                                     'stay in the images', 'carla_exact': False,
            'inputs': {os.path.basename(f): f for f in (cc, vc, ic)},
            'hashes': {os.path.basename(f): sha256_file(f) for f in (cc, vc, ic)},
            'notes': ['images are rectified (KITTI raw *_sync); K is P_rect\'s, shifted by ½ px to pixel-edge '
                      'coordinates'],
            'left_out': [{'what': 'moving objects', 'why': 'not masked: the tracklets are a separate download'}],
            'heuristic': []}
    cap = Capture('kitti_raw', path, camera, frames, None, meta)
    vdir = os.path.join(path, 'velodyne_points', 'data')
    if os.path.isdir(vdir):
        cap.lidar = lambda: _kitti_lidar(vdir, cap, T_velo_imu)
        meta['lidar'] = 'velodyne_points'
    return cap


def _kitti_stamps(path, n) -> np.ndarray:
    import datetime as _dt
    if not os.path.isfile(path):
        return np.arange(n) * 0.1
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if s:
                day, _, frac = s.partition('.')
                t = _dt.datetime.strptime(day, '%Y-%m-%d %H:%M:%S').replace(tzinfo=_dt.timezone.utc).timestamp()
                out.append(t + (float('0.' + frac) if frac else 0.0))
    return np.array(out[:n]) if len(out) >= n else np.arange(n) * 0.1


def _kitti_lidar(vdir, cap, T_velo_imu, max_sweeps: int = 40, max_range: float = 80.0):
    files = sorted(f for f in os.listdir(vdir) if f.endswith('.bin'))
    frames = {f.index: f for f in cap.frames}
    keys = np.array(sorted(frames))
    idx = list(range(len(files)))[::max(1, int(math.ceil(len(files) / max_sweeps)))]
    T_imu_velo = inv(T_velo_imu)
    xs, cs = [], []
    for k in idx:
        pts = np.fromfile(os.path.join(vdir, files[k]), np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        pts = pts[np.linalg.norm(pts, axis=1) < max_range]
        fr = frames[int(keys[np.argmin(np.abs(keys - k))])]
        Tw = fr.ego @ T_imu_velo
        w = pts @ Tw[:3, :3].T + Tw[:3, 3]
        xs.append(w)
        cs.append(colour_points(w, nearest_frames(cap.frames, fr.t)))
    return np.concatenate(xs), np.concatenate(cs)


# ------------------------------------------------------------------------------------------------ nuScenes
def nuscenes_capture(path: str, scene: str | None = None, channel: str = 'CAM_FRONT') -> Capture:
    from ..logreplay.model import quat_matrix
    from ..logreplay.nuscenes import find_version
    from ..logreplay.ros2 import falcon_class
    path = os.path.abspath(path)
    v = find_version(path)

    def tab(name):
        p = os.path.join(path, v, name + '.json')
        if not os.path.isfile(p):
            raise DataError(f'{path}/{v}: no {name}.json')
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    scenes = tab('scene')
    sc = next((s for s in scenes if scene in (s['name'], s['token'])), None) if scene else (scenes[0] if scenes
                                                                                             else None)
    if sc is None:
        raise DataError(f'{path}: no scene {scene!r}')
    samples = {s['token']: s for s in tab('sample') if s['scene_token'] == sc['token']}
    sensors = {s['token']: s for s in tab('sensor')}
    cal = {c['token']: c for c in tab('calibrated_sensor')}
    poses = {e['token']: e for e in tab('ego_pose')}
    sds = [d for d in tab('sample_data') if d['sample_token'] in samples and d.get('is_key_frame', True)]

    def chan(d):
        return sensors[cal[d['calibrated_sensor_token']]['sensor_token']]['channel']
    cams = sorted((d for d in sds if chan(d) == channel), key=lambda d: d['timestamp'])
    if not cams:
        raise DataError(f'scene {sc["name"]}: no key frames of {channel}')

    def T_of(rot, trans):
        T = np.eye(4)
        T[:3, :3] = quat_matrix(*rot)
        T[:3, 3] = trans
        return T
    cats = {c['token']: c['name'] for c in tab('category')}
    inst = {i['token']: cats.get(i['category_token'], '') for i in tab('instance')}
    boxes = {}
    for a in tab('sample_annotation'):
        if a['sample_token'] in samples and falcon_class(inst.get(a['instance_token'], '')):
            boxes.setdefault(a['sample_token'], []).append((T_of(a['rotation'], a['translation']), a['size']))
    t0 = cams[0]['timestamp'] * 1e-6
    frames = []
    for d in cams:
        c = cal[d['calibrated_sensor_token']]
        ego = T_of(poses[d['ego_pose_token']]['rotation'], poses[d['ego_pose_token']]['translation'])
        mount = T_of(c['rotation'], c['translation'])
        Ki = np.asarray(c['camera_intrinsic'], float)
        K = (Ki[0, 0], Ki[1, 1], Ki[0, 2] + 0.5, Ki[1, 2] + 0.5)
        W, H = int(d.get('width') or 1600), int(d.get('height') or 900)
        vm = inv(ego @ mount)
        frames.append(Frame(len(frames), d['timestamp'] * 1e-6 - t0, os.path.join(path, d['filename']), K, (W, H),
                            vm, ego, mount, _box_mask(boxes.get(d['sample_token'], []), vm, K, (W, H)), None, None,
                            d['filename']))
    lids = sorted((d for d in sds if chan(d) == 'LIDAR_TOP'), key=lambda d: d['timestamp'])
    meta = {'world': 'nuScenes\' global frame (x east, y north, z up), metres', 'name': f'nuScenes {sc["name"]}',
            'masks': 'the key frame\'s annotated boxes of Falcon\'s actor classes, projected and filled',
            'carla_exact': False, 'inputs': {}, 'notes': [f'{len(frames)} key frames of {channel}'],
            'left_out': [{'what': 'non-key-frame images (sweeps)', 'why': 'their boxes would be interpolated; key '
                                                                         'frames only'}],
            'heuristic': [{'what': 'mask dilation', 'how': '2 px around each projected box'}]}
    cap = Capture('nuscenes', path, channel, frames, None, meta)
    if lids:
        def lidar():
            xs, cs = [], []
            for d in lids:
                p = os.path.join(path, d['filename'])
                if not os.path.isfile(p):
                    continue
                pts = np.fromfile(p, np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
                c = cal[d['calibrated_sensor_token']]
                T = T_of(poses[d['ego_pose_token']]['rotation'], poses[d['ego_pose_token']]['translation']) @ \
                    T_of(c['rotation'], c['translation'])
                w = pts @ T[:3, :3].T + T[:3, 3]
                keep = np.ones(len(w), bool)
                for B, size in boxes.get(d['sample_token'], []):
                    keep &= ~_in_box(w, B, size, 0.2)
                w = w[keep]
                xs.append(w)
                cs.append(colour_points(w, nearest_frames(frames, d['timestamp'] * 1e-6 - t0)))
            if not xs:
                raise DataError(f'{path}: the LIDAR_TOP files are not on disk')
            return np.concatenate(xs), np.concatenate(cs)
        cap.lidar = lidar
        meta['lidar'] = 'LIDAR_TOP'
    return cap


def _in_box(w, B, size, pad=0.0):
    wl, ll, hl = size                                   # nuScenes: width, length, height
    loc = (w - B[:3, 3]) @ B[:3, :3]
    return (np.abs(loc[:, 0]) <= ll / 2 + pad) & (np.abs(loc[:, 1]) <= wl / 2 + pad) & \
        (np.abs(loc[:, 2]) <= hl / 2 + pad)


def box_corners(B, size) -> np.ndarray:
    """(8, 3) world corners of a box: B world <- box (x along its length), size (width, length, height)."""
    wl, ll, hl = size
    c = np.array([[sx * ll / 2, sy * wl / 2, sz * hl / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return c @ B[:3, :3].T + B[:3, 3]


EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def project_box_hull(corners, viewmat, K, near=0.1):
    """Image points (pixel-edge) of a 3D box's silhouette: its corners in front of the near plane and its edges
    clipped there; None when it is wholly behind."""
    pc = corners @ viewmat[:3, :3].T + viewmat[:3, 3]
    pts = [p for p in pc if p[2] > near]
    for i, j in EDGES:
        a, b = pc[i], pc[j]
        if (a[2] > near) != (b[2] > near):
            s = (near - a[2]) / (b[2] - a[2])
            pts.append(a + s * (b - a))
    if not pts:
        return None
    p = np.array(pts)
    return np.stack([K[0] * p[:, 0] / p[:, 2] + K[2], K[1] * p[:, 1] / p[:, 2] + K[3]], -1)


def _box_mask(boxes, viewmat, K, size, dilate: int = 2):
    def mk():
        import cv2
        W, H = size
        m = np.zeros((H, W), np.uint8)
        for B, sz in boxes:
            uv = project_box_hull(box_corners(B, sz), viewmat, K)
            if uv is None:
                continue
            hull = cv2.convexHull(np.clip(uv - 0.5, -1e5, 1e5).astype(np.float32))
            cv2.fillConvexPoly(m, np.round(hull).astype(np.int32), 1)
        if dilate:
            m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
        return m == 0
    return mk


# ================================================================================================ points
def colour_points(w, fr) -> np.ndarray:
    """Colours (N, 3) in [0, 1] of world points from a frame's image — or from a list of frames, each filling the
    points the ones before it did not see (grey 0.5 where none sees them). No occlusion test: a point hidden behind
    something nearer takes the nearer surface's colour (the optimisation corrects colours quickly)."""
    out = np.full((len(w), 3), 0.5)
    todo = np.ones(len(w), bool)
    for f in (fr if isinstance(fr, (list, tuple)) else [fr]):
        if not todo.any():
            break
        img = load_rgb(f.image)
        H, W = img.shape[:2]
        K = scaled_K(f.K, f.size, (W, H))
        q = w[todo]
        pc = q @ f.viewmat[:3, :3].T + f.viewmat[:3, 3]
        z = pc[:, 2]
        ok = z > 0.1
        u = np.where(ok, K[0] * pc[:, 0] / np.where(ok, z, 1.0) + K[2], -1.0)
        v = np.where(ok, K[1] * pc[:, 1] / np.where(ok, z, 1.0) + K[3], -1.0)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(todo)[0][ok]
        out[idx] = img[v[ok].astype(int), u[ok].astype(int)] / 255.0
        todo[idx] = False
    return out


def nearest_frames(frames: list, t: float, k: int = 4, spread_s: float = 2.0) -> list:
    """The frame nearest in time to t, then frames up to spread_s earlier (they saw what is now behind)."""
    ts = np.array([f.t for f in frames])
    order = [int(np.argmin(np.abs(ts - t)))]
    for dt in np.linspace(spread_s / (k - 1), spread_s, k - 1) if k > 1 else []:
        j = int(np.argmin(np.abs(ts - (t - dt))))
        if j not in order:
            order.append(j)
    return [frames[j] for j in order]


def voxel_average(xyz, rgb, voxel: float, max_points: int | None = None, seed: int = 0):
    """One point per occupied voxel (the mean position and colour); then at most max_points, a seeded sample."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv_idx, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv_idx = inv_idx.reshape(-1)
    n = len(cnt)
    out_x = np.zeros((n, 3))
    out_c = np.zeros((n, 3))
    np.add.at(out_x, inv_idx, xyz)
    np.add.at(out_c, inv_idx, rgb)
    out_x /= cnt[:, None]
    out_c /= cnt[:, None]
    if max_points and n > max_points:
        sel = np.random.default_rng(seed).choice(n, max_points, replace=False)
        out_x, out_c = out_x[sel], out_c[sel]
    return out_x, out_c


def sfm_points(frames: list, width: int = 640, baseline_m: float = 1.0, max_pairs: int = 40, seed: int = 0,
               masks=True) -> tuple:
    """Structure from the known poses (module doc): (xyz, rgb, stats)."""
    import cv2
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    cent = np.array([f.centre for f in frames])
    pairs = []
    i = 0
    while i < len(frames) - 1:
        d = np.linalg.norm(cent[i + 1:] - cent[i], axis=1)
        j = np.nonzero(d >= baseline_m)[0]
        if not len(j):
            break
        pairs.append((i, i + 1 + int(j[0])))
        i += max(1, int(j[0]) // 2 + 1)
    if len(pairs) > max_pairs:
        sel = np.linspace(0, len(pairs) - 1, max_pairs).round().astype(int)
        pairs = [pairs[k] for k in sel]
    cache = {}

    def feats(k):
        if k not in cache:
            fr = frames[k]
            wh = working_size(fr.size, width)
            img = resize_rgb(load_rgb(fr.image), wh)
            m = resize_mask(load_mask(fr), wh) if masks else None
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp, des = sift.detectAndCompute(gray, None if m is None else m.astype(np.uint8) * 255)
            K = scaled_K(fr.K, fr.size, wh)
            Kcv = np.array([[K[0], 0, K[2] - 0.5], [0, K[1], K[3] - 0.5], [0, 0, 1.0]])
            pts = np.array([p.pt for p in kp], np.float64).reshape(-1, 2)
            cache[k] = (pts, des, Kcv @ fr.viewmat[:3], img)
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
        return cache[k]
    xs, cs = [], []
    stats = {'pairs': len(pairs), 'matches': 0, 'kept': 0}
    for i, j in pairs:
        pi, di, Pi, img = feats(i)
        pj, dj, Pj, _ = feats(j)
        if di is None or dj is None or len(di) < 8 or len(dj) < 8:
            continue
        good = [m[0] for m in matcher.knnMatch(di, dj, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            continue
        a = pi[[m.queryIdx for m in good]]
        b = pj[[m.trainIdx for m in good]]
        X = cv2.triangulatePoints(Pi, Pj, a.T, b.T)
        X = (X[:3] / np.where(np.abs(X[3]) < 1e-12, 1e-12, X[3])).T
        ok = np.ones(len(X), bool)
        rays = []
        for P, uv, fr in ((Pi, a, frames[i]), (Pj, b, frames[j])):
            h = np.c_[X, np.ones(len(X))] @ P.T
            z = (np.c_[X, np.ones(len(X))] @ fr.viewmat[:3].T)[:, 2]
            ok &= (z > 0.5) & (z < 200.0)
            err = np.linalg.norm(h[:, :2] / np.where(np.abs(h[:, 2:]) < 1e-9, 1e-9, h[:, 2:]) - uv, axis=1)
            ok &= err < 1.5
            r = X - fr.centre
            rays.append(r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9))
        ang = np.degrees(np.arccos(np.clip(np.sum(rays[0] * rays[1], 1), -1, 1)))
        ok &= ang > 1.0
        stats['matches'] += len(good)
        stats['kept'] += int(ok.sum())
        xs.append(X[ok])
        ij = np.clip(a[ok].round().astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1])
        cs.append(img[ij[:, 1], ij[:, 0]] / 255.0)
    if not xs or not sum(len(x) for x in xs):
        raise DataError('structure from the known poses found no points (too little texture or baseline)')
    return np.concatenate(xs), np.concatenate(cs), stats


def range_points(frames: list, width: int = 640, stride: int = 8, max_points: int = 400_000) -> tuple:
    """Points from the frames' range images (Falcon's ground truth, sparsely sampled; masked pixels left out)."""
    xs, cs = [], []
    for fr in frames:
        rg = load_range(fr)
        if rg is None:
            continue
        img = load_rgb(fr.image)
        m = load_mask(fr)
        H, W = rg.shape
        K = scaled_K(fr.K, fr.size, (W, H))
        vv, uu = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
        r = rg[vv, uu]
        ok = (r > 0) & (r < 250)
        if m is not None:
            ok &= m[vv, uu]
        x = (uu + 0.5 - K[2]) / K[0]
        y = (vv + 0.5 - K[3]) / K[1]
        d = np.stack([x, y, np.ones_like(x)], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        pc = d[ok] * r[ok][:, None]
        Tw = inv(fr.viewmat)
        xs.append(pc @ Tw[:3, :3].T + Tw[:3, 3])
        cs.append(img[vv[ok], uu[ok]] / 255.0)
    if not xs:
        raise DataError('the source has no range images (init=range is for Falcon runs)')
    x, c = np.concatenate(xs), np.concatenate(cs)
    if len(x) > max_points:
        sel = np.random.default_rng(0).choice(len(x), max_points, replace=False)
        x, c = x[sel], c[sel]
    return x, c


def sky_shell(frames: list, xyz, n: int = 20000, seed: int = 0) -> tuple:
    """Points on a far shell around the cameras (module doc): (xyz, rgb, radius). Directions: a Fibonacci spiral from
    10° below the horizon to the zenith; colours from the frames that see them (mean), else grey."""
    cent = np.array([f.centre for f in frames])
    c = cent.mean(0)
    dist = np.linalg.norm(np.asarray(xyz) - c, axis=1) if len(xyz) else np.array([50.0])
    radius = max(100.0, 1.5 * float(np.percentile(dist, 99)) + float(np.linalg.norm(cent - c, axis=1).max()))
    k = np.arange(n) + 0.5
    zmin = math.sin(math.radians(-10.0))
    z = zmin + (1 - zmin) * k / n
    phi = math.pi * (3 - math.sqrt(5)) * k
    r = np.sqrt(1 - z * z)
    d = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
    pts = c + radius * d
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    step = max(1, len(frames) // 12)
    for fr in frames[::step]:
        col = colour_points(pts, fr)
        seen = ~np.all(col == 0.5, axis=1)
        acc[seen] += col[seen]
        cnt[seen] += 1
    rgb = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], 0.5)
    return pts, rgb, radius


def init_points(cap: Capture, init: str = 'auto', train_frames=None, voxel: float = 0.15,
                max_points: int = 300_000, width: int = 640, seed: int = 0, sky: int = 20000) -> dict:
    """The initial points: {'xyz', 'rgb', 'sky' (bool), 'how', 'stats'}."""
    frames = train_frames or cap.frames
    if init not in INIT_KINDS:
        raise DataError(f'init is one of {", ".join(INIT_KINDS)}')
    how = init
    if init == 'auto':
        how = 'lidar' if cap.lidar is not None else 'sfm'
    stats = {}
    if how == 'lidar':
        if cap.lidar is None:
            raise DataError(f'{cap.path}: the source has no lidar for init=lidar')
        xyz, rgb = cap.lidar()
        stats['raw_points'] = int(len(xyz))
        xyz, rgb = voxel_average(xyz, rgb, voxel, max_points, seed)
    elif how == 'sfm':
        xyz, rgb, st = sfm_points(frames, width, seed=seed)
        stats.update(st)
        xyz, rgb = voxel_average(xyz, rgb, voxel / 3, max_points, seed)
    elif how == 'range':
        xyz, rgb = range_points(frames, width)
        stats['raw_points'] = int(len(xyz))
        xyz, rgb = voxel_average(xyz, rgb, voxel, max_points, seed)
    else:
        cent = np.array([f.centre for f in frames])
        lo, hi = cent.min(0) - 30.0, cent.max(0) + 30.0
        lo[2], hi[2] = cent[:, 2].min() - 3.0, cent[:, 2].max() + 15.0
        rng = np.random.default_rng(seed)
        xyz = lo + (hi - lo) * rng.random((min(max_points, 100_000), 3))
        rgb = rng.random((len(xyz), 3))
    stats['points'] = int(len(xyz))
    flags = np.zeros(len(xyz), bool)
    if sky:
        sp, sc, radius = sky_shell(frames, xyz, sky, seed)
        xyz, rgb = np.concatenate([xyz, sp]), np.concatenate([rgb, sc])
        flags = np.concatenate([flags, np.ones(len(sp), bool)])
        stats['sky_points'] = int(len(sp))
        stats['sky_radius_m'] = round(radius, 2)
    return {'xyz': xyz, 'rgb': rgb, 'sky': flags, 'how': how, 'stats': stats}
