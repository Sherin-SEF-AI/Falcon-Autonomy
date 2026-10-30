"""Dataset exporters: recorded drives (falcon-raw run folders, workers.drive) written out in the formats training and
robotics tools read — COCO, YOLO, KITTI object, nuScenes (with nuScenes-lidarseg and nuScenes-panoptic when a lidar
was recorded), a Waymo-style JSON layout, BDD100K lane markings, ROS 2 bags (MCAP) and semantic occupancy (Occ3D-
nuScenes, OpenOcc v2, SurroundOcc: core.occexport). numpy and OpenCV; core.mcap_bag
writes the bags. Ported in part from Vantage (vantage.sim.record's nuScenes tables and COCO files, its bag topics).

    export([run_dir, ...], 'nuscenes', out_dir, {'splits': {'train': [...], 'val': [...]}}) -> report

Every export writes OUT/falcon_export.json (EXPORT_SCHEMA): the format, the runs with their hashes (provenance.block
over each run's run.json, dataset.json, frames.jsonl and lanes.json), the splits, the frames used, what was left out
and why, what is heuristic, and how to load it. The report returned is that document.

Frames. A run is in CARLA's frames: left-handed, x forward, y right, z up, degrees (core.rig). Every format here is
right-handed; points map by p -> S·p and rotations by R -> S·R·S with S = diag(1, −1, 1) (core.rig.to_ros), never by
flipping Euler angles. The vehicle frame keeps Falcon's origin (CARLA's vehicle origin: the bounding box's centre at
ground height), not nuScenes' rear axle nor KITTI's IMU: each manifest says so. Falcon's intrinsics are in pixel-edge
coordinates (pixel i covers [i, i+1), cx = W/2); KITTI, nuScenes, Waymo and ROS get OpenCV's (pixel centres on
integers: cx − 0.5, cy − 0.5). 2D boxes (COCO, YOLO, Waymo camera labels, BDD vertices) stay in pixel-edge
coordinates, as COCO defines them; KITTI's are written in its pixel-centre convention, clipped to the image.

Splits: options['splits'] = {'train' | 'val' | 'test': [run dirs]}; a run in two splits is refused (leakage). Formats
with splits write them (COCO and YOLO per split, KITTI training/ with ImageSets train.txt / val.txt and testing/,
nuScenes a table version per split, Waymo training/ validation/ testing/, BDD one file per split, MCAP a folder per
split). Without splits everything is one set named 'all' (nuScenes: one version).

Limits (also in each manifest's left_out): only pinhole cameras go to KITTI, nuScenes and Waymo (they model no other
lens); 3D boxes exist only for objects some camera saw (Falcon labels boxes per camera frame); faulty runs (fault
injection) are exported as recorded, the calibration stating the plan, as a real faulty rig's would.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import math
import os
import shutil

import numpy as np

from . import groundtruth as GT
from .provenance import block, read_json, sha256_file, utc_now
from .rig import CAM_OPTICAL, VEHICLES, carla_rotation, matrix_of, quat_of, tf_list

EXPORT_SCHEMA = 'falcon.export/1'
MANIFEST = 'falcon_export.json'
FORMATS = ('coco', 'yolo', 'kitti', 'nuscenes', 'waymo', 'bdd_lanes', 'mcap', 'occ3d')
SPLITS = ('train', 'val', 'test')
S4 = np.diag([1.0, -1.0, 1.0, 1.0])
CARLA_TO_OPTICAL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
# Falcon classes with instances (objects/), in id order: COCO category ids are the class ids, YOLO's the positions
THING_CLASSES = [c for c in GT.CLASSES if c[5]]
THING_NAMES = [c[1] for c in THING_CLASSES]
KITTI_TYPE = {'car': 'Car', 'truck': 'Truck', 'bus': 'Misc', 'train': 'Tram', 'motorcycle': 'Misc',
              'bicycle': 'Cyclist', 'rider': 'Cyclist', 'pedestrian': 'Pedestrian'}
WAYMO_TYPE = {'car': 1, 'truck': 1, 'bus': 1, 'train': 1, 'motorcycle': 1, 'pedestrian': 2, 'traffic_sign': 3,
              'bicycle': 4, 'rider': 4}
WAYMO_TYPE_NAME = {0: 'TYPE_UNKNOWN', 1: 'TYPE_VEHICLE', 2: 'TYPE_PEDESTRIAN', 3: 'TYPE_SIGN', 4: 'TYPE_CYCLIST'}
NUSC_ATTRIBUTES = (
    ('vehicle.moving', 'Vehicle is moving.'), ('vehicle.stopped', 'Vehicle, with a driver/rider in/on it, is currently '
                                                                  'stationary but has an intent to move.'),
    ('vehicle.parked', 'Vehicle is stationary (usually for longer duration) with no immediate intent to move.'),
    ('cycle.with_rider', 'There is a rider on the bicycle or motorcycle.'),
    ('cycle.without_rider', 'There is NO rider on the bicycle or motorcycle.'),
    ('pedestrian.sitting_lying_down', 'The human is sitting or lying down.'),
    ('pedestrian.standing', 'The human is standing.'), ('pedestrian.moving', 'The human is moving.'))
NUSC_VISIBILITY = (('1', 'v0-40', 'visibility of whole object is between 0 and 40%'),
                   ('2', 'v40-60', 'visibility of whole object is between 40 and 60%'),
                   ('3', 'v60-80', 'visibility of whole object is between 60 and 80%'),
                   ('4', 'v80-100', 'visibility of whole object is between 80 and 100%'))
MAP_RES = 0.1                       # nuScenes map masks: 0.1 m per pixel, global (0, 0) at the bottom-left corner
MAP_MARGIN_M = 100.0
MAP_MAX_PX = 16384
RADAR_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('dyn_prop', 'i1'), ('id', '<i2'), ('rcs', '<f4'),
                        ('vx', '<f4'), ('vy', '<f4'), ('vx_comp', '<f4'), ('vy_comp', '<f4'),
                        ('is_quality_valid', 'i1'), ('ambig_state', 'i1'), ('x_rms', 'i1'), ('y_rms', 'i1'),
                        ('invalid_state', 'i1'), ('pdh0', 'i1'), ('vx_rms', 'i1'), ('vy_rms', 'i1')])
PCD_HEAD = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n'
            'FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms '
            'invalid_state pdh0 vx_rms vy_rms\nSIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\n'
            'TYPE F F F I I F F F F F I I I I I I I I\nCOUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n'
            'WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n')
LANE_DENSE_M = 0.5
LANE_MAX_M = 80.0
LANE_SEAM_PX = 0.75                 # polyline simplification (cv2.approxPolyDP) tolerance


class ExportError(Exception):
    pass


def token(*parts) -> str:
    """A stable 32-hex token like nuScenes' (md5 of the parts joined by '|')."""
    return hashlib.md5('|'.join(str(x) for x in parts).encode('utf-8')).hexdigest()


def pose_matrix(p: dict) -> np.ndarray:
    """4×4 of a CARLA pose dict {x y z roll pitch yaw} (Transform::GetMatrix: actor -> parent)."""
    T = np.eye(4)
    T[:3, :3] = carla_rotation(float(p.get('roll', 0)), float(p.get('pitch', 0)), float(p.get('yaw', 0)))
    T[:3, 3] = [float(p.get(k, 0.0)) for k in 'xyz']
    return T


def rh(T) -> np.ndarray:
    """A CARLA 4×4 transform in the mirrored, right-handed axes (x forward, y left, z up)."""
    return S4 @ np.asarray(T, float) @ S4


def rh_points(P) -> np.ndarray:
    P = np.array(P, float, copy=True)
    P[..., 1] *= -1.0
    return P


def yaw_matrix(a_rad: float) -> np.ndarray:
    c, s = math.cos(a_rad), math.sin(a_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def K_opencv(K) -> np.ndarray:
    """Falcon's K (pixel-edge coordinates) in OpenCV's (pixel centres on integers)."""
    K = np.array(K, float)
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def _r(v, n=6):
    x = round(float(v), n) + 0.0
    return 0.0 if x == 0 else x


def _step_of(rel: str) -> int:
    try:
        return int(os.path.basename(rel).split('.')[0])
    except ValueError:
        return -1


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# =========================================================================================================== a run
class Run:
    """One falcon-raw run folder: its config, dataset.json, frames and the readers of its files."""

    def __init__(self, path: str, key: str | None = None):
        self.dir = os.path.abspath(path)
        self.name = os.path.basename(os.path.normpath(self.dir))
        self.key = key or self.name
        try:
            self.run = read_json(self.p('run.json'))
            self.ds = read_json(self.p('dataset.json'))
        except (OSError, ValueError) as e:
            raise ExportError(f'{self.dir} is not a finished drive: {e}') from None
        if self.ds.get('layout') != 'falcon-raw' or self.ds.get('schema') != 'falcon.dataset/1':
            raise ExportError(f'{self.dir}: dataset.json is not a falcon-raw dataset')
        if self.run.get('status') != 'ok':
            raise ExportError(f'{self.dir}: the run did not finish ok (status {self.run.get("status")!r})')
        cfg = None
        if os.path.isfile(self.p('drive.json')):
            cfg = read_json(self.p('drive.json'))
        if not cfg or 'rig' not in cfg:
            cfg = (self.run.get('provenance') or {}).get('settings') or {}
        if 'rig' not in cfg:
            raise ExportError(f'{self.dir}: no drive config (drive.json, or run.json provenance.settings) with a rig')
        self.cfg = cfg
        self.rig = cfg['rig']
        self.sensors = {s['id']: s for s in self.rig['sensors']}
        self.cams = {s['id']: s for s in self.ds.get('sensors', []) if s['type'].startswith('camera.')}
        self.fps = float(self.ds.get('fps') or cfg.get('fps') or 20.0)
        with open(self.p('frames.jsonl'), encoding='utf-8') as fh:
            self.frames = [json.loads(x) for x in fh if x.strip()]
        started = self.run.get('started') or self.run.get('provenance', {}).get('created')
        try:
            t0 = _dt.datetime.strptime(started, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=_dt.timezone.utc)
        except (TypeError, ValueError):
            raise ExportError(f'{self.dir}: run.json has no start time') from None
        self.epoch_us = int(t0.timestamp()) * 1_000_000
        self.started = t0
        self._actor_cat = None
        self._lanes = False

    def p(self, rel: str) -> str:
        return os.path.join(self.dir, rel)

    def inputs(self) -> dict:
        return {f'{self.key}/{f}': self.p(f) for f in ('run.json', 'dataset.json', 'frames.jsonl', 'lanes.json',
                                                         'drive.json') if os.path.isfile(self.p(f))}

    def of_type(self, *types) -> list:
        return [s for s in self.rig['sensors'] if s['type'] in types]

    def us(self, rec: dict, sid: str | None = None) -> int:
        """A record's time in µs since the Unix epoch: the run's start (run.json, UTC) + its simulation time; a sensor
        with a clock_offset fault: its own (offset) stamp."""
        t = (rec.get('stamps') or {}).get(sid, rec['sim_t']) if sid else rec['sim_t']
        return self.epoch_us + int(round(float(t) * 1e6))

    def ego(self, rec: dict) -> np.ndarray:
        """CARLA world <- vehicle at the record's step."""
        return pose_matrix(rec['ego'])

    def mount(self, sid: str, actual: bool = False) -> np.ndarray:
        """CARLA vehicle <- sensor: the planned mount, or (actual=True) with its mount faults (what the labels see)."""
        s = self.sensors[sid]
        p = s['pose']
        if actual:
            from .faults import actual_pose
            p = actual_pose(p, s.get('faults'))
        return pose_matrix(p)

    def file(self, rec: dict, sid: str, kind: str) -> str | None:
        return rec['files'].get(f'{sid}/{kind}')

    def own(self, rec: dict, sid: str, kind: str) -> str | None:
        """The record's file of a stream only when it is this step's own (not a frozen repeat of an earlier one)."""
        rel = self.file(rec, sid, kind)
        return rel if rel is not None and (kind == 'lidar' or _step_of(rel) == rec['step']) else None

    def objects(self, rec: dict, cid: str) -> list | None:
        rel = self.own(rec, cid, 'objects')
        if rel is None:
            return None
        return read_json(self.p(rel)).get('objects', [])

    def lidar(self, rec: dict, sid: str):
        """(points (N, 4) float32 x y z intensity in CARLA's sensor frame, tags uint8, actor ids uint32) or None."""
        rel = self.file(rec, sid, 'lidar')
        if rel is None:
            return None
        pts = np.fromfile(self.p(rel), np.float32).reshape(-1, 4)
        n = len(pts)
        lab = np.fromfile(self.p(rel[:-4] + '.labels.bin'), np.uint8)
        if len(lab) != 5 * n:
            raise ExportError(f'{self.dir}/{rel}: its labels file does not have 5 bytes per point')
        return pts, lab[:n].copy(), lab[n:].view(np.uint32).copy()

    def lidar_pose(self, rec: dict, sid: str) -> np.ndarray:
        """CARLA world <- lidar at the sweep's end (frames.jsonl lidar_pose)."""
        T = np.eye(4)
        T[:3, :] = np.asarray(rec['lidar_pose'][sid], float).reshape(3, 4)
        return T

    def radar(self, rec: dict, sid: str):
        rel = self.file(rec, sid, 'radar')
        return None if rel is None else np.load(self.p(rel)).astype(np.float64)

    def jsonl(self, kind: str, sid: str) -> list:
        path = self.p(os.path.join(kind, f'{sid}.jsonl'))
        if not os.path.isfile(path):
            return []
        with open(path, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def lanes(self) -> dict | None:
        if self._lanes is False:
            self._lanes = read_json(self.p('lanes.json')) if os.path.isfile(self.p('lanes.json')) else None
        return self._lanes

    def actor_categories(self) -> dict:
        """{actor id: nuScenes category} of every actor some camera labelled in the run."""
        if self._actor_cat is None:
            out = {}
            for rec in self.frames:
                for cid in self.cams:
                    for o in self.objects(rec, cid) or ():
                        if o.get('actor_id') is not None and o.get('category') in GT.NUSC_INDEX:
                            out[int(o['actor_id'])] = o['category']
            self._actor_cat = out
        return self._actor_cat

    def camera_steps(self, cid: str, every: int = 1, need: str | None = None) -> list:
        """Records with this camera's own frame (frozen repeats and dropped frames skipped) and, with need, a lidar
        sweep of that sensor; every n-th of those."""
        out = [r for r in self.frames if self.own(r, cid, 'rgb') and (need is None or self.file(r, need, 'lidar'))]
        return out[::max(int(every), 1)]

    def faults(self) -> dict:
        return self.run.get('faults_applied') or {}


def camera_T(run: Run, cid: str, actual: bool = False) -> np.ndarray:
    """Right-handed vehicle <- the camera's optical frame (x right, y down, z forward)."""
    T = rh(run.mount(cid, actual))
    T[:3, :3] = T[:3, :3] @ CAM_OPTICAL
    return T


def is_pinhole(run: Run, cid: str) -> bool:
    return run.cams.get(cid, {}).get('type') == 'camera.pinhole'


def nusc_category(o: dict):
    """(nuScenes category, attribute or None) of an object, or (None, None) when nuScenes has no category for it
    (traffic lights, signs, other static things). The map's parked vehicles take their class's category and the
    attribute vehicle.parked."""
    cat = o.get('category')
    if cat == 'vehicle.parked':
        c = GT.CLASS_ID.get(o.get('class'))
        nus = next((x[3] for x in GT.CLASSES if x[0] == c), None)
        return (nus, 'vehicle.parked') if nus and nus.startswith('vehicle.') else (None, None)
    if cat in GT.NUSC_INDEX and GT.NUSC_INDEX[cat] in GT.THING_INDICES:
        return cat, None
    return None, None


# ===================================================================================================== the context
class Ctx:
    def __init__(self, fmt, out_dir, options, progress, splits):
        self.fmt, self.out, self.opt, self.progress = fmt, out_dir, options, progress
        self.splits = splits                  # [(split name or None, [Run])]
        self.left_out, self.heuristic, self.notes = {}, {}, []
        self.counts = {}
        self.frames = {}
        self.load = ''
        self.nbytes = 0
        self.nfiles = 0
        self._done, self._total = 0, max(sum(len(r) for _, r in splits), 1)

    def leave(self, what: str, why: str):
        self.left_out.setdefault(what, why)

    def heur(self, what: str, how: str):
        self.heuristic.setdefault(what, how)

    def count(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n

    def tick(self, text):
        self._done += 1
        if self.progress:
            self.progress(min(self._done / self._total, 1.0), text)

    def path(self, rel: str) -> str:
        p = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def write(self, rel: str, data):
        p = self.path(rel)
        mode = 'wb' if isinstance(data, (bytes, bytearray)) else 'w'
        with open(p, mode, **({} if mode == 'wb' else {'encoding': 'utf-8'})) as fh:
            fh.write(data)
        self.nfiles += 1
        self.nbytes += len(data)

    def json(self, rel: str, doc, indent=None):
        self.write(rel, json.dumps(doc, indent=indent, separators=(',', ':') if indent is None else None,
                                   default=_jsonable) + '\n')

    def place(self, src: str, rel: str):
        """Copy (or hard-link, or symlink: options['link']) a run's file into the export."""
        dst = self.path(rel)
        mode = self.opt.get('link', 'copy')
        if os.path.lexists(dst):
            os.remove(dst)
        if mode == 'symlink':
            os.symlink(os.path.abspath(src), dst)
        else:
            try:
                if mode != 'hardlink':
                    raise OSError
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        self.nfiles += 1
        self.nbytes += os.path.getsize(dst)


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _any_fisheye(ctx: Ctx) -> bool:
    return any(c['type'] == 'camera.fisheye' for _, rs in ctx.splits for r in rs for c in r.cams.values())


def _split_name(split, names=None) -> str:
    if split is None:
        return 'all'
    return (names or {}).get(split, split)


# =========================================================================================================== COCO
def _coco(ctx: Ctx):
    """annotations/instances_<split>.json + images/<split>/: every labelled camera frame; masks as compressed RLE."""
    every = int(ctx.opt.get('every', 1))
    cats = [{'id': c[0], 'name': c[1], 'supercategory': c[3].split('.')[0], 'nuscenes': c[3]} for c in THING_CLASSES]
    known = {c[0] for c in THING_CLASSES}
    for split, runs in ctx.splits:
        sn = _split_name(split)
        images, anns = [], []
        for run in runs:
            for cid in _cameras(ctx, run, need='objects'):
                for rec in run.camera_steps(cid, every):
                    objs = run.objects(rec, cid)
                    if objs is None:
                        continue
                    cam = run.cams[cid]
                    name = f'{run.key}__{cid}__{rec["step"]:06d}.jpg'
                    ctx.place(run.p(run.own(rec, cid, 'rgb')), f'images/{sn}/{name}')
                    iid = len(images) + 1
                    images.append({'id': iid, 'file_name': name, 'width': cam['width'], 'height': cam['height'],
                                   'falcon': {'run': run.key, 'sensor': cid, 'step': rec['step'],
                                              'timestamp_us': run.us(rec, cid),
                                              'instance': run.own(rec, cid, 'instance')}})
                    for o in objs:
                        cls = GT.CLASS_ID.get(o.get('class'))
                        if cls not in known:
                            ctx.count('objects_without_coco_category')
                            continue
                        x0, y0, x1, y1 = o['bbox']
                        anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': cls,
                                     'bbox': [x0, y0, x1 - x0, y1 - y0], 'area': int(o['pixels']), 'iscrowd': 0,
                                     'segmentation': o['rle'],
                                     'falcon': {'object_id': o['id'], 'track_id': o.get('track_id'),
                                                'actor_id': o.get('actor_id'), 'nuscenes_category': o.get('category'),
                                                'occluded': o.get('occluded'), 'truncated': o.get('truncated'),
                                                'range_m': o.get('range_m'), 'bbox_full': o.get('bbox_full')}})
            ctx.tick(f'COCO {sn}: {run.key}')
        ctx.json(f'annotations/instances_{sn}.json', {
            'info': {'description': f'Falcon Autonomy drives ({sn})', 'version': EXPORT_SCHEMA,
                     'date_created': utc_now()},
            'licenses': [], 'images': images, 'annotations': anns, 'categories': cats})
        ctx.frames[sn] = len(images)
        ctx.count('images', len(images))
        ctx.count('annotations', len(anns))
    ctx.leave('masks as polygons', 'segmentation is COCO compressed RLE (iscrowd 0), exact from the instance image; '
                                   'no polygon conversion')
    ctx.leave('keypoints, captions', 'not in Falcon\'s labels')
    ctx.load = 'pycocotools: COCO("annotations/instances_<split>.json"); images in images/<split>/'
    ctx.frames_doc = {'bbox': '[x, y, width, height] in pixel-edge coordinates (COCO\'s): pixel i covers [i, i+1)',
                      'category_id': 'Falcon class id (= CARLA semantic tag); categories: the classes with instances'}


def _cameras(ctx: Ctx, run: Run, need: str = 'rgb', pinhole: bool = False) -> list:
    want = ctx.opt.get('cameras')
    out = []
    for cid, cam in run.cams.items():
        if want and cid not in want:
            continue
        if need not in cam.get('streams', []):
            ctx.leave(f'camera {cid} ({run.key})', f'it recorded no {need}')
            continue
        if pinhole and cam['type'] != 'camera.pinhole':
            ctx.leave(f'camera {cid} ({run.key})', f'a {cam["type"]}: {ctx.fmt} models pinhole cameras only')
            continue
        out.append(cid)
    return out


# =========================================================================================================== YOLO
def _yolo(ctx: Ctx):
    """images/<split>/ and labels/<split>/<name>.txt (class cx cy w h, normalised), data.yaml."""
    every = int(ctx.opt.get('every', 1))
    index = {c[1]: i for i, c in enumerate(THING_CLASSES)}
    written = []
    for split, runs in ctx.splits:
        sn = _split_name(split)
        n = 0
        for run in runs:
            for cid in _cameras(ctx, run, need='objects'):
                cam = run.cams[cid]
                W, H = float(cam['width']), float(cam['height'])
                for rec in run.camera_steps(cid, every):
                    objs = run.objects(rec, cid)
                    if objs is None:
                        continue
                    name = f'{run.key}__{cid}__{rec["step"]:06d}'
                    ctx.place(run.p(run.own(rec, cid, 'rgb')), f'images/{sn}/{name}.jpg')
                    lines = []
                    for o in objs:
                        k = index.get(o.get('class'))
                        x0, y0, x1, y1 = (float(v) for v in o['bbox'])
                        if k is None or x1 <= x0 or y1 <= y0:
                            ctx.count('objects_left_out')
                            continue
                        cx, cy = min(max((x0 + x1) / 2 / W, 0.0), 1.0), min(max((y0 + y1) / 2 / H, 0.0), 1.0)
                        lines.append(f'{k} {cx:.6f} {cy:.6f} {min((x1 - x0) / W, 1.0):.6f} '
                                     f'{min((y1 - y0) / H, 1.0):.6f}')
                    ctx.write(f'labels/{sn}/{name}.txt', ''.join(x + '\n' for x in lines))
                    n += 1
                    ctx.count('boxes', len(lines))
            ctx.tick(f'YOLO {sn}: {run.key}')
        ctx.frames[sn] = n
        ctx.count('images', n)
        written.append(sn)
    yaml = ['# Falcon Autonomy export (falcon_export.json has the provenance)', f'path: {os.path.abspath(ctx.out)}']
    for sn in written:
        key = {'all': 'train'}.get(sn, sn)
        yaml.append(f'{key}: images/{sn}')
    yaml.append(f'nc: {len(THING_CLASSES)}')
    yaml.append('names:')
    yaml += [f'  {i}: {c[1]}' for i, c in enumerate(THING_CLASSES)]
    ctx.write('data.yaml', '\n'.join(yaml) + '\n')
    ctx.leave('segmentation (YOLO-seg polygons)', 'not written: COCO has the exact masks')
    ctx.load = 'Ultralytics: YOLO(...).train(data="data.yaml"); without splits the one set is listed as train'
    ctx.frames_doc = {'labels': 'class cx cy w h, normalised by the image width and height (pixel-edge coordinates), '
                                'clipped to [0, 1]; class = the position in data.yaml names'}


# ========================================================================================================== KITTI
def _kitti_calib(run: Run, cid: str, lid: str | None) -> tuple[dict, np.ndarray | None]:
    K = K_opencv(run.cams[cid]['K'])
    P = np.zeros((3, 4))
    P[:, :3] = K
    T_v_c = camera_T(run, cid)
    lines = {f'P{i}': P for i in range(4)}
    lines['R0_rect'] = np.eye(3)
    Tv = None
    if lid is not None:
        Tv = np.linalg.inv(T_v_c) @ rh(run.mount(lid))
        lines['Tr_velo_to_cam'] = Tv[:3, :]
        imu = next((s for s in run.of_type('imu')), None)
        T_imu = rh(run.mount(imu['id'])) if imu else np.eye(4)
        lines['Tr_imu_to_velo'] = (np.linalg.inv(rh(run.mount(lid))) @ T_imu)[:3, :]
    else:
        lines['Tr_velo_to_cam'] = np.linalg.inv(T_v_c)[:3, :]          # no lidar: the vehicle frame stands in
        lines['Tr_imu_to_velo'] = np.eye(4)[:3, :]
    return lines, Tv


def _kitti_label(o: dict, run: Run, cid: str, W: int, H: int) -> str | None:
    typ = KITTI_TYPE.get(o.get('class'))
    b = o.get('box3d')
    if typ is None or not b:
        return None
    T_c_v = np.linalg.inv(camera_T(run, cid, actual=True))
    c = rh_points(b['center'])
    L, Wd, Hh = b['size']
    yaw = -math.radians(b['yaw_deg'])
    bottom = c - np.array([0.0, 0.0, Hh / 2.0])
    loc = T_c_v[:3, :3] @ bottom + T_c_v[:3, 3]
    fwd = T_c_v[:3, :3] @ np.array([math.cos(yaw), math.sin(yaw), 0.0])
    ry = _wrap(math.atan2(-fwd[2], fwd[0]))
    alpha = _wrap(ry - math.atan2(loc[0], loc[2]))
    x0, y0, x1, y1 = o['bbox']
    left, top = max(x0 - 0.5, 0.0), max(y0 - 0.5, 0.0)
    right, bottom_px = min(x1 - 0.5, W - 1.0), min(y1 - 0.5, H - 1.0)
    occ = o.get('occluded')
    occ_i = 3 if occ is None else 0 if occ <= 0.1 else 1 if occ <= 0.5 else 2
    trunc = min(max(float(o.get('truncated') or 0.0), 0.0), 1.0)
    return (f'{typ} {trunc:.2f} {occ_i:d} {alpha:.2f} {left:.2f} {top:.2f} {right:.2f} {bottom_px:.2f} '
            f'{Hh:.2f} {Wd:.2f} {L:.2f} {loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f} {ry:.2f}')


def _kitti(ctx: Ctx):
    """training/ (train and val, ImageSets/train.txt and val.txt) and testing/: image_2 (PNG), label_2, calib,
    velodyne; one camera (options['camera'], else the first labelled pinhole) and the first lidar."""
    import cv2
    every = int(ctx.opt.get('every', 1))
    sets = {}
    for split, runs in ctx.splits:
        part = 'testing' if split == 'test' else 'training'
        for run in runs:
            cams = _cameras(ctx, run, need='objects', pinhole=True)
            cid = ctx.opt.get('camera') or (cams[0] if cams else None)
            if cid not in cams:
                ctx.leave(f'run {run.key}', f'no labelled pinhole camera {cid or ""} for KITTI')
                ctx.tick(run.key)
                continue
            for other in cams:
                if other != cid:
                    ctx.leave(f'camera {other} ({run.key})', 'KITTI object has one camera (image_2)')
            lids = run.of_type('lidar')
            lid = lids[0]['id'] if lids else None
            for extra in lids[1:]:
                ctx.leave(f'lidar {extra["id"]} ({run.key})', 'KITTI object has one lidar (velodyne)')
            calib, _ = _kitti_calib(run, cid, lid)
            calib_text = ''.join(f'{k}: {" ".join(f"{v:.12e}" for v in np.asarray(m).ravel())}\n'
                                 for k, m in calib.items())
            W, H = run.cams[cid]['width'], run.cams[cid]['height']
            for rec in run.camera_steps(cid, every, need=lid):
                objs = run.objects(rec, cid)
                if objs is None:
                    continue
                idx = len(sets.setdefault(part, []))
                name = f'{idx:06d}'
                img = cv2.imread(run.p(run.own(rec, cid, 'rgb')), cv2.IMREAD_COLOR)
                ok, png = cv2.imencode('.png', img)
                ctx.write(f'{part}/image_2/{name}.png', png.tobytes())
                lines = [x for x in (_kitti_label(o, run, cid, W, H) for o in objs) if x]
                ctx.count('objects_left_out', len(objs) - len(lines))
                ctx.count('labels', len(lines))
                ctx.write(f'{part}/label_2/{name}.txt', ''.join(x + '\n' for x in lines))
                ctx.write(f'{part}/calib/{name}.txt', calib_text)
                if lid is not None:
                    pts, _, _ = run.lidar(rec, lid)
                    ctx.write(f'{part}/velodyne/{name}.bin', rh_points(pts).astype(np.float32).tobytes())
                sets[part].append((split, name, run.key, cid, rec['step']))
            ctx.tick(f'KITTI: {run.key}')
    for part, rows in sets.items():
        by = {}
        for split, name, *_ in rows:
            by.setdefault(_split_name(split, {'test': 'test'}), []).append(name)
        for sn, names in by.items():
            ctx.write(f'{part}/ImageSets/{sn}.txt', ''.join(n + '\n' for n in names))
        ctx.json(f'{part}/falcon_frames.json', [{'index': n, 'split': s, 'run': r, 'camera': c, 'step': st}
                                                for s, n, r, c, st in rows])
        ctx.frames[part] = len(rows)
    ctx.count('frames', sum(len(v) for v in sets.values()))
    ctx.leave('P0, P1, P3', 'Falcon exports one camera: they repeat P2 (KITTI readers need the keys)')
    ctx.leave('score (16th field)', 'ground truth has none')
    ctx.leave('traffic lights, signs and other classes', 'KITTI object has no such type (DontCare regions not written)')
    ctx.leave('Van, Person_sitting', 'not told apart in Falcon\'s classes: vans are Car, sitting people Pedestrian')
    if not any(r.of_type('imu') for _, rs in ctx.splits for r in rs):
        ctx.leave('Tr_imu_to_velo', 'no IMU in the rig: identity')
    ctx.heur('occluded (0–3)', 'from Falcon\'s occluded share: ≤ 0.1 → 0, ≤ 0.5 → 1, else 2; unknown → 3')
    ctx.heur('rotation_y, alpha', 'the box\'s yaw seen in the camera frame (exact for a level camera; KITTI assumes '
                                  'its y axis is vertical)')
    ctx.load = 'KITTI object devkit layout: training/{image_2,label_2,calib,velodyne}, ImageSets/{train,val}.txt'
    ctx.frames_doc = {
        'camera': 'image_2 is the exported camera; the rectified reference frame is its optical frame (R0_rect = I); '
                  'P2 = K [I | 0] with K in OpenCV pixel-centre coordinates',
        'velodyne': 'the lidar\'s frame, right-handed (x forward, y left, z up): CARLA\'s with y mirrored; float32 '
                    'x y z intensity (CARLA\'s intensity: atmospheric attenuation in [0, 1], not reflectance)',
        'labels': 'location = the bottom centre of the 3D box in the camera frame; dimensions h w l; bbox in pixel-'
                  'centre coordinates clipped to the image',
        'class_map': dict(KITTI_TYPE)}


# ======================================================================================================== nuScenes
def _channel(s: dict) -> str:
    pre = {'camera.pinhole': 'CAM', 'camera.fisheye': 'CAM', 'lidar': 'LIDAR', 'radar': 'RADAR'}[s['type']]
    name = ''.join(c if c.isalnum() else '_' for c in s['name'].upper())
    return name if name.startswith(pre + '_') or name == pre else f'{pre}_{name}'


def _drivable_mask(lanes: dict | None, offset, w_px: int, h_px: int) -> np.ndarray:
    """uint8 mask (0 / 255) of the map's driving lanes in the shifted global frame (nuScenes' map convention)."""
    import cv2
    mask = np.zeros((h_px, w_px), np.uint8)
    if not lanes:
        return mask
    col = {c: i for i, c in enumerate(lanes['columns'])}
    groups = {}
    for p in lanes['points']:
        groups.setdefault((p[col['road']], p[col['section']], p[col['lane']]), []).append(p)
    step = float(lanes.get('step_m', 2.0))
    polys = []
    for pts in groups.values():
        pts.sort(key=lambda p: p[col['s']])
        for a, b in zip(pts, pts[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 2.5 * step:
                continue
            quad = []
            for p, sgn in ((a, -1), (b, -1), (b, 1), (a, 1)):
                yaw = math.radians(p[col['yaw']])
                h = sgn * p[col['width']] / 2.0
                x, y = p[0] - math.sin(yaw) * h, p[1] + math.cos(yaw) * h
                gx, gy = x + offset[0], -y + offset[1]
                quad.append((gx / MAP_RES, h_px - gy / MAP_RES))
            polys.append(np.round(np.asarray(quad) * 4).astype(np.int32))
    if polys:
        cv2.fillPoly(mask, polys, 255, lineType=cv2.LINE_8, shift=2)
    return mask


def _nusc_offset(run: Run, lanes_too: bool = True):
    """(global offset (3,), mask width, height px): the global frame is CARLA's world mirrored (x, −y, z) and shifted
    so that the ego's whole drive, MAP_MARGIN_M around it, has positive coordinates (a nuScenes map mask starts at the
    global origin)."""
    xy = np.array([[r['ego']['x'], -r['ego']['y']] for r in run.frames], float)
    lo, hi = xy.min(0) - MAP_MARGIN_M, xy.max(0) + MAP_MARGIN_M
    off = np.array([-math.floor(lo[0]), -math.floor(lo[1]), 0.0])
    w = int(min(math.ceil((hi[0] - lo[0]) / MAP_RES) + 1, MAP_MAX_PX))
    h = int(min(math.ceil((hi[1] - lo[1]) / MAP_RES) + 1, MAP_MAX_PX))
    return off, w, h


def _ring(xyz: np.ndarray, s: dict) -> np.ndarray:
    """Beam index of each lidar point from its elevation (CARLA's channels are evenly spaced from lower_fov to
    upper_fov); 0 = the lowest beam."""
    n = int(s.get('channels', 1))
    lo, hi = float(s.get('lower_fov', -25.0)), float(s.get('upper_fov', 15.0))
    el = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    if n <= 1 or hi <= lo:
        return np.zeros(len(xyz), np.float32)
    return np.clip(np.round((el - lo) / (hi - lo) * (n - 1)), 0, n - 1).astype(np.float32)


def _radar_xyz(det: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CARLA radar detections (velocity, azimuth, altitude, depth; radians) -> (xyz in CARLA's sensor frame, unit
    rays)."""
    az, al, d = det[:, 1], det[:, 2], det[:, 3]
    ray = np.stack([np.cos(al) * np.cos(az), np.cos(al) * np.sin(az), np.sin(al)], 1)
    return ray * d[:, None], ray


def _sensor_velocity(run: Run, rec: dict, sid: str) -> np.ndarray:
    """A sensor's velocity (m/s) in its own right-handed frame at a record's step: the ego's velocity and yaw rate by
    central differences of frames.jsonl's poses, plus ω × the mount's lever arm (pitch and roll rates neglected)."""
    i = rec['step']
    fr = run.frames
    k = next((j for j, r in enumerate(fr) if r['step'] == i), 0)
    a, b = fr[max(k - 1, 0)], fr[min(k + 1, len(fr) - 1)]
    dt = b['sim_t'] - a['sim_t']
    if dt <= 0:
        return np.zeros(3)
    Ta, Tb, T = rh(run.ego(a)), rh(run.ego(b)), rh(run.ego(rec))
    v_w = (Tb[:3, 3] - Ta[:3, 3]) / dt
    dyaw = _wrap(math.atan2(Tb[1, 0], Tb[0, 0]) - math.atan2(Ta[1, 0], Ta[0, 0]))
    M = rh(run.mount(sid))
    v_v = T[:3, :3].T @ v_w + np.cross([0.0, 0.0, dyaw / dt], M[:3, 3])
    return M[:3, :3].T @ v_v


def _ego_box(run: Run):
    dims = VEHICLES.get(run.rig.get('vehicle'))
    if dims is None:
        return None
    L, W, H = dims
    return np.array([0.0, 0.0, H / 2.0]), [L + 0.2, W + 0.2, H + 0.2]


def lidar_labels(run: Run, tags: np.ndarray, ids: np.ndarray, xyz_vehicle: np.ndarray, inst_of: dict):
    """(nuScenes-lidarseg index uint8, nuScenes-panoptic uint16) per point: an actor some camera labelled takes its
    category, every other point its tag's (GT.NUSC_OF_TAG); the ego's points are vehicle.ego: those of the actor id
    most points inside the rig vehicle's box (+ 0.1 m) carry (a motion-compensated sweep moves the ego's own hits of
    its earlier slices out of its box, so the box alone does not find them all). Panoptic: index · 1000 + the actor's
    instance number (inst_of: actor id -> 1…999, grown here) for things; 0 for stuff and for things without an actor
    id (the map's parked vehicles)."""
    seg = GT.NUSC_OF_TAG[tags].astype(np.uint8)
    cats = run.actor_categories()
    if len(ids):
        for aid in np.unique(ids[ids > 0]):
            cat = cats.get(int(aid))
            if cat is not None:
                seg[ids == aid] = GT.NUSC_INDEX[cat]
    eb = _ego_box(run)
    if eb is not None and len(xyz_vehicle):
        inside = GT.in_box(xyz_vehicle, eb[0], eb[1], np.eye(3)) & (ids > 0)
        if inside.any():
            vals, n = np.unique(ids[inside], return_counts=True)
            ego_id = vals[np.argmax(n)]
            seg[(ids == ego_id) | inside] = GT.NUSC_INDEX['vehicle.ego']
    inst = np.zeros(len(seg), np.uint16)
    thing = np.isin(seg, sorted(GT.THING_INDICES)) & (ids > 0)
    for aid in np.unique(ids[thing]):
        k = inst_of.get(int(aid))
        if k is None and len(inst_of) < 999:
            k = inst_of[int(aid)] = len(inst_of) + 1
        if k:
            inst[thing & (ids == aid)] = k
    return seg, (seg.astype(np.uint16) * GT.PANOPTIC_DIV + inst).astype(np.uint16)


def _nuscenes(ctx: Ctx):
    version = ctx.opt.get('version', GT.NUSCENES_VERSION)
    khz = ctx.opt.get('keyframe_hz')
    versions = []
    for split, runs in ctx.splits:
        ver = version if split is None else f'{version}-{split}'
        versions.append(ver)
        T = {k: [] for k in ('attribute', 'calibrated_sensor', 'category', 'ego_pose', 'instance', 'log', 'map',
                             'sample', 'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility',
                             'lidarseg', 'panoptic')}
        tok = lambda *p: token('falcon', *p)                                        # noqa: E731
        T['category'] = GT.nuscenes_categories(lambda n: tok('category', n))
        cat_tok = {r['name']: r['token'] for r in T['category']}
        T['attribute'] = [{'token': tok('attribute', n), 'name': n, 'description': d} for n, d in NUSC_ATTRIBUTES]
        att_tok = {r['name']: r['token'] for r in T['attribute']}
        T['visibility'] = [{'token': t, 'level': lv, 'description': d} for t, lv, d in NUSC_VISIBILITY]
        sensors = {}
        for run in runs:
            _nusc_run(ctx, run, ver, T, tok, cat_tok, att_tok, sensors, khz)
            ctx.tick(f'nuScenes {ver}: {run.key}')
        T['sensor'] = list(sensors.values())
        if not T['lidarseg']:
            T.pop('lidarseg')
            T.pop('panoptic')
        for name, rows in T.items():
            ctx.json(f'{ver}/{name}.json', rows, indent=1)
        ctx.frames[ver] = len(T['sample'])
        ctx.count('samples', len(T['sample']))
        ctx.count('sample_data', len(T['sample_data']))
        ctx.count('sample_annotations', len(T['sample_annotation']))
        ctx.count('instances', len(T['instance']))
        ctx.count('lidarseg', len(T.get('lidarseg', [])))
    ctx.versions = versions
    if _any_fisheye(ctx):
        ctx.leave('fisheye cameras', 'nuScenes models pinhole cameras only')
    ctx.leave('GNSS, IMU', 'nuScenes has no table for them (the CAN bus expansion is another format)')
    ctx.leave('attributes other than vehicle.parked', 'moving / stopped / standing are not labelled by Falcon')
    ctx.leave('traffic lights, signs', 'no nuScenes category: not annotated (their lidar points are static.manmade)')
    ctx.leave('radar rcs, *_rms, pdh0', 'CARLA\'s radar gives none: written 0 (dyn_prop 4 = unknown, ambig_state 3, '
                                       'invalid_state 0: the devkit\'s default filters keep every detection)')
    ctx.heur('radar vx_comp, vy_comp', 'the radial velocity with the radar\'s own motion removed: the ego\'s velocity '
                                       'and yaw rate by central differences of its poses')
    ctx.leave('3D boxes of objects no camera saw', 'Falcon labels boxes per camera frame')
    ctx.heur('visibility', 'the best camera\'s (1 − occluded) · (1 − truncated), binned as nuScenes\' levels')
    ctx.heur('lidar ring', 'from each point\'s elevation: CARLA\'s beams are evenly spaced (0 = the lowest)')
    ctx.heur('vehicle.ego points', 'the points of the actor id most lidar points inside the rig vehicle\'s box '
                                   '(+ 0.1 m) carry: Falcon does not store the ego\'s actor id')
    ctx.load = (f'NuScenes(version=<one of {versions}>, dataroot=<this folder>); nuScenes-lidarseg and -panoptic '
                'load with it (devkit 1.1+)')
    ctx.frames_doc = {
        'global': 'CARLA\'s world mirrored (x, −y, z), shifted by each log\'s offset (manifest runs[].global_offset_m) '
                  'so the drive and its map mask have positive coordinates',
        'ego': 'right-handed vehicle frame (x forward, y left, z up) at CARLA\'s vehicle origin (bounding-box centre '
               'at ground height), NOT nuScenes\' rear axle',
        'cameras': 'calibrated_sensor rotation = the optical frame (x right, y down, z forward); camera_intrinsic in '
                   'OpenCV pixel-centre coordinates',
        'lidar': '.pcd.bin float32 x y z intensity ring in the sensor frame (right-handed), motion-compensated to the '
                 'sweep\'s end, which is its timestamp; intensity is CARLA\'s (attenuation, 0–1)',
        'radar': '.pcd (nuScenes\' 18 fields): x y z in the sensor frame; vx vy = CARLA\'s relative radial velocity '
                 '((target − radar) · ray, positive receding) as a vector along the ray',
        'time': 'µs since the Unix epoch: the run\'s start (run.json started, UTC) + simulation time',
        'keyframes': 'every lidar sweep of the first lidar (options keyframe_hz thins them), else every camera frame'}


def _nusc_run(ctx, run: Run, ver, T, tok, cat_tok, att_tok, sensors, khz):
    import cv2
    off, mw, mh = _nusc_offset(run)
    Toff = np.eye(4)
    Toff[:3, 3] = off
    ctx.run_info.setdefault(run.key, {})['global_offset_m'] = off.tolist()
    cams = _cameras(ctx, run, pinhole=True)
    lids = [s['id'] for s in run.of_type('lidar')]
    rads = [s['id'] for s in run.of_type('radar')]
    chans = {sid: _channel(run.sensors[sid]) for sid in cams + lids + rads}
    cs_tok = {}
    for sid, ch in chans.items():
        mod = 'camera' if sid in cams else 'lidar' if sid in lids else 'radar'
        sensors.setdefault(ch, {'token': tok('sensor', ch), 'channel': ch, 'modality': mod})
        Tm = camera_T(run, sid) if mod == 'camera' else rh(run.mount(sid))
        cs_tok[sid] = tok('calibrated_sensor', run.key, sid)
        T['calibrated_sensor'].append({
            'token': cs_tok[sid], 'sensor_token': sensors[ch]['token'],
            'translation': [_r(v) for v in Tm[:3, 3]], 'rotation': [_r(q, 10) for q in quat_of(Tm[:3, :3])],
            'camera_intrinsic': K_opencv(run.cams[sid]['K']).tolist() if mod == 'camera' else []})
    log_tok, scene_tok, map_tok = tok('log', run.key), tok('scene', run.key), tok('map', run.key)
    # keyframes: the first lidar's sweeps, else the first camera's frames; thinned to keyframe_hz
    key_src = lids[0] if lids else (cams[0] if cams else None)
    kinds = 'lidar' if lids else 'rgb'
    keys, last_t = [], None
    for rec in run.frames:
        if key_src is None:
            break
        if (run.own(rec, key_src, 'rgb') if kinds == 'rgb' else run.file(rec, key_src, 'lidar')) is None:
            continue
        if khz and last_t is not None and rec['sim_t'] - last_t < 1.0 / float(khz) - 1e-6:
            continue
        keys.append(rec)
        last_t = rec['sim_t']
    if not keys:
        ctx.leave(f'run {run.key}', 'no keyframes (no lidar sweep or camera frame)')
        return
    key_steps = {r['step']: i for i, r in enumerate(keys)}
    s_tok = [tok('sample', run.key, r['step']) for r in keys]
    s_us = [run.us(r) for r in keys]
    for i, r in enumerate(keys):
        T['sample'].append({'token': s_tok[i], 'timestamp': s_us[i], 'scene_token': scene_tok,
                            'prev': s_tok[i - 1] if i else '', 'next': s_tok[i + 1] if i + 1 < len(keys) else ''})
    by_chan = {sid: [] for sid in chans}
    inst_rows, ann_by_inst, inst_of = {}, {}, {}
    for rec in run.frames:
        step = rec['step']
        T_g_e = Toff @ rh(run.ego(rec))
        is_key = step in key_steps
        sample_i = key_steps[step] if is_key else int(min(np.searchsorted(s_us, run.us(rec)), len(keys) - 1))
        sweeps = {}
        for sid, ch in chans.items():
            if sid in cams:
                rel = run.own(rec, sid, 'rgb')
                if rel is None:
                    continue
                cam = run.cams[sid]
                ext, w, h = 'jpg', cam['width'], cam['height']
            elif sid in lids:
                if run.file(rec, sid, 'lidar') is None:
                    continue
                ext, w, h = 'pcd.bin', 0, 0
            else:
                if run.file(rec, sid, 'radar') is None:
                    continue
                ext, w, h = 'pcd', 0, 0
            us = run.us(rec, sid)
            sd = tok('sample_data', run.key, sid, step)
            fname = f'{"samples" if is_key else "sweeps"}/{ch}/{run.key}__{ch}__{us}.{ext}'
            if sid in cams:
                ctx.place(run.p(rel), fname)
                Te = T_g_e
            elif sid in lids:
                pts, tags, ids = run.lidar(rec, sid)
                xyz = rh_points(pts[:, :3])
                ctx.write(fname, np.c_[xyz, pts[:, 3], _ring(xyz, run.sensors[sid])].astype(np.float32).tobytes())
                Te = Toff @ rh(run.lidar_pose(rec, sid) @ np.linalg.inv(run.mount(sid)))
                sweeps[sid] = (pts, tags, ids)
                if is_key:
                    Pv = pts[:, :3].astype(np.float64) @ run.mount(sid)[:3, :3].T + run.mount(sid)[:3, 3]
                    seg, pan = lidar_labels(run, tags, ids, Pv, inst_of)
                    rl, rp = f'lidarseg/{ver}/{sd}_lidarseg.bin', f'panoptic/{ver}/{sd}_panoptic.npz'
                    ctx.write(rl, seg.tobytes())
                    buf = io.BytesIO()
                    np.savez_compressed(buf, data=pan)
                    ctx.write(rp, buf.getvalue())
                    T['lidarseg'].append({'token': sd, 'sample_data_token': sd, 'filename': rl})
                    T['panoptic'].append({'token': sd, 'sample_data_token': sd, 'filename': rp})
            else:
                det = run.radar(rec, sid)
                xyz, ray = _radar_xyz(det)
                xyz, ray = rh_points(xyz), rh_points(ray)
                a = np.zeros(max(len(det), 1), RADAR_DTYPE)
                if len(det):
                    a['x'], a['y'], a['z'] = xyz.T
                    vel = det[:, 0:1] * ray                        # CARLA's relative radial velocity, as a vector
                    a['vx'], a['vy'] = vel[:, 0], vel[:, 1]
                    v_s = _sensor_velocity(run, rec, sid)          # the radar's own motion, in its frame
                    comp = (det[:, 0] + ray @ v_s)[:, None] * ray
                    a['vx_comp'], a['vy_comp'] = comp[:, 0], comp[:, 1]
                    a['id'] = np.arange(len(det))
                else:                                              # nuScenes writes an empty scan as one NaN point
                    for f in ('x', 'y', 'z', 'vx', 'vy', 'vx_comp', 'vy_comp', 'rcs'):
                        a[f] = np.nan
                a['dyn_prop'], a['ambig_state'], a['is_quality_valid'] = 4, 3, 1
                ctx.write(fname, PCD_HEAD.format(n=len(a)).encode('ascii') + a.tobytes() + b'\n')   # as nuScenes' files
                sweeps[sid] = det
                Te = T_g_e
            T['ego_pose'].append({'token': sd, 'timestamp': us, 'rotation': [_r(q, 10) for q in quat_of(Te[:3, :3])],
                                  'translation': [_r(v) for v in Te[:3, 3]]})
            by_chan[sid].append({'token': sd, 'sample_token': s_tok[sample_i], 'ego_pose_token': sd,
                                 'calibrated_sensor_token': cs_tok[sid], 'timestamp': us,
                                 'fileformat': ext.split('.')[0], 'is_key_frame': is_key, 'height': h, 'width': w,
                                 'filename': fname, 'prev': '', 'next': ''})
        if is_key:
            _nusc_annotations(ctx, run, rec, cams, sweeps, T_g_e, s_tok[key_steps[step]], tok, cat_tok, att_tok,
                              inst_rows, ann_by_inst)
    for rows in by_chan.values():
        for i, r in enumerate(rows):
            r['prev'] = rows[i - 1]['token'] if i else ''
            r['next'] = rows[i + 1]['token'] if i + 1 < len(rows) else ''
        T['sample_data'].extend(rows)
    for it, anns in ann_by_inst.items():
        for i, a in enumerate(anns):
            a['prev'] = anns[i - 1]['token'] if i else ''
            a['next'] = anns[i + 1]['token'] if i + 1 < len(anns) else ''
            T['sample_annotation'].append(a)
        row = inst_rows[it]
        row.update(nbr_annotations=len(anns), first_annotation_token=anns[0]['token'],
                   last_annotation_token=anns[-1]['token'])
        T['instance'].append(row)
    mask = _drivable_mask(run.lanes(), off, mw, mh)
    ok, png = cv2.imencode('.png', mask)
    ctx.write(f'maps/{map_tok}.png', png.tobytes())
    if run.lanes() is None:
        ctx.leave(f'map of {run.key}', 'the run has no lanes.json: its mask is empty (unknown, not undrivable)')
    T['map'].append({'token': map_tok, 'log_tokens': [log_tok], 'category': 'semantic_prior',
                     'filename': f'maps/{map_tok}.png'})
    town = run.cfg.get('town') or os.path.basename(run.cfg.get('xodr') or '') or 'world'
    T['log'].append({'token': log_tok, 'logfile': run.key, 'vehicle': run.rig.get('vehicle', ''),
                     'date_captured': run.started.strftime('%Y-%m-%d'), 'location': f'carla-{town}'})
    T['scene'].append({'token': scene_tok, 'log_token': log_tok, 'nbr_samples': len(keys),
                       'first_sample_token': s_tok[0], 'last_sample_token': s_tok[-1], 'name': f'scene-{run.key}',
                       'description': f'Falcon drive {run.cfg.get("name", run.key)}, seed {run.cfg.get("seed")}, '
                                      f'{len(run.frames)} steps at {run.fps:g} fps'})


def _nusc_annotations(ctx, run, rec, cams, sweeps, T_g_e, sample_tok, tok, cat_tok, att_tok, inst_rows, ann_by_inst):
    """The keyframe's boxes: every object the cameras labelled at this very step, once per track."""
    seen = {}
    for cid in run.cams:
        for o in run.objects(rec, cid) or ():
            vis = (1.0 - float(o.get('occluded') or 0.0)) * (1.0 - float(o.get('truncated') or 0.0))
            k = str(o.get('track_id'))
            if k not in seen:
                seen[k] = (o, vis)
            elif vis > seen[k][1]:
                seen[k] = (o, vis)
    pts_v = []
    for sid, data in sweeps.items():
        M = run.mount(sid)
        if run.sensors[sid]['type'] == 'lidar':
            P = data[0][:, :3].astype(np.float64)
            pts_v.append(('lidar', P @ M[:3, :3].T + M[:3, 3]))
        else:
            P, _ = _radar_xyz(data)
            pts_v.append(('radar', P @ M[:3, :3].T + M[:3, 3]))
    for k, (o, vis) in seen.items():
        cat, att = nusc_category(o)
        b = o.get('box3d')
        if cat is None or not b:
            ctx.count('objects_not_annotated')
            continue
        R = GT.box_matrix(b['yaw_deg'])
        n_l = sum(int(GT.in_box(P, b['center'], b['size'], R).sum()) for m, P in pts_v if m == 'lidar')
        n_r = sum(int(GT.in_box(P, b['center'], b['size'], R).sum()) for m, P in pts_v if m == 'radar')
        c = T_g_e[:3, :3] @ rh_points(b['center']) + T_g_e[:3, 3]
        Rg = T_g_e[:3, :3] @ yaw_matrix(-math.radians(b['yaw_deg']))
        it = tok('instance', run.key, k)
        inst_rows.setdefault(it, {'token': it, 'category_token': cat_tok[cat]})
        L, W, H = b['size']
        level = 1 if vis < 0.4 else 2 if vis < 0.6 else 3 if vis < 0.8 else 4
        ann_by_inst.setdefault(it, []).append({
            'token': tok('sample_annotation', run.key, k, rec['step']), 'sample_token': sample_tok,
            'instance_token': it, 'visibility_token': str(level),
            'attribute_tokens': [att_tok[att]] if att else [], 'translation': [_r(v) for v in c],
            'size': [_r(W, 4), _r(L, 4), _r(H, 4)], 'rotation': [_r(q, 10) for q in quat_of(Rg)],
            'prev': '', 'next': '', 'num_lidar_pts': n_l, 'num_radar_pts': n_r})


# ========================================================================================================== Waymo
def _waymo_camera_name(s: dict) -> tuple[int, str]:
    y = (float(s['pose'].get('yaw', 0.0)) + 180.0) % 360.0 - 180.0       # CARLA: positive to the right
    if abs(y) <= 22.5:
        return 1, 'FRONT'
    if -67.5 <= y < -22.5:
        return 2, 'FRONT_LEFT'
    if 22.5 < y <= 67.5:
        return 3, 'FRONT_RIGHT'
    if -135.0 <= y < -67.5:
        return 4, 'SIDE_LEFT'
    if 67.5 < y <= 135.0:
        return 5, 'SIDE_RIGHT'
    return 0, 'UNKNOWN'


def _waymo(ctx: Ctx):
    """<split>/<segment>/: context.json (calibrations) and frames.jsonl (a Frame per line: pose, images, lasers,
    laser_labels, camera_labels) with the images (JPEG) and point clouds (.npy) beside them."""
    every = int(ctx.opt.get('every', 1))
    names = {'train': 'training', 'val': 'validation', 'test': 'testing'}
    for split, runs in ctx.splits:
        sn = _split_name(split, names)
        for run in runs:
            seg = f'segment-{run.key}'
            base = f'{sn}/{seg}'
            cams = _cameras(ctx, run, pinhole=True)
            lids = [s['id'] for s in run.of_type('lidar')]
            cam_cal = []
            cam_name = {}
            for cid in cams:
                num, nm = _waymo_camera_name(run.sensors[cid])
                cam_name[cid] = num
                K = K_opencv(run.cams[cid]['K'])
                T_v_body = rh(run.mount(cid))                                    # camera body: x fwd, y left, z up
                cam_cal.append({'name': num, 'name_text': nm, 'falcon_sensor': cid,
                                'intrinsic': [K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0, 0.0, 0.0, 0.0],
                                'extrinsic': {'transform': T_v_body.ravel().tolist()},
                                'width': run.cams[cid]['width'], 'height': run.cams[cid]['height'],
                                'rolling_shutter_direction': 'GLOBAL_SHUTTER'})
            las_cal = []
            for k, lid in enumerate(lids):
                s = run.sensors[lid]
                n = int(s.get('channels', 1))
                inc = np.radians(np.linspace(float(s['lower_fov']), float(s['upper_fov']), n)).tolist()
                las_cal.append({'name': k + 1, 'name_text': ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR'][k]
                                if k < 5 else 'UNKNOWN', 'falcon_sensor': lid, 'beam_inclinations': inc,
                                'beam_inclination_min': inc[0], 'beam_inclination_max': inc[-1],
                                'extrinsic': {'transform': rh(run.mount(lid)).ravel().tolist()}})
            ctx.json(f'{base}/context.json', {
                'name': run.key, 'camera_calibrations': cam_cal, 'laser_calibrations': las_cal,
                'stats': {'location': f'carla-{run.cfg.get("town") or "world"}',
                          'weather': (run.cfg.get('weather_source') or {}).get('preset', ''),
                          'time_of_day': ''}}, indent=1)
            lines = []
            for rec in run.frames[::max(every, 1)]:
                pose = rh(run.ego(rec))
                fr = {'context': {'name': run.key}, 'timestamp_micros': run.us(rec), 'falcon_step': rec['step'],
                      'pose': {'transform': pose.ravel().tolist()}, 'images': [], 'lasers': [], 'laser_labels': [],
                      'camera_labels': [], 'no_label_zones': []}
                boxes = {}
                for cid in cams:
                    rel = run.own(rec, cid, 'rgb')
                    if rel is None:
                        continue
                    dst = f'{base}/images/{cid}/{rec["step"]:06d}.jpg'
                    ctx.place(run.p(rel), dst)
                    fr['images'].append({'name': cam_name[cid], 'image': os.path.relpath(dst, base),
                                         'pose': {'transform': pose.ravel().tolist()},
                                         'pose_timestamp': run.us(rec) / 1e6, 'camera_trigger_time':
                                         run.us(rec, cid) / 1e6, 'camera_readout_done_time': run.us(rec, cid) / 1e6})
                    labels = []
                    for o in run.objects(rec, cid) or ():
                        t = WAYMO_TYPE.get(o.get('class'))
                        if t is None:
                            continue
                        x0, y0, x1, y1 = o['bbox']
                        labels.append({'box': {'center_x': (x0 + x1) / 2, 'center_y': (y0 + y1) / 2,
                                               'length': x1 - x0, 'width': y1 - y0}, 'type': t,
                                       'id': str(o.get('track_id'))})
                        if o.get('box3d') and str(o.get('track_id')) not in boxes:
                            boxes[str(o.get('track_id'))] = (o, t)
                    fr['camera_labels'].append({'name': cam_name[cid], 'labels': labels})
                clouds = []
                for k, lid in enumerate(lids):
                    if run.file(rec, lid, 'lidar') is None:
                        continue
                    pts, _, _ = run.lidar(rec, lid)
                    M = run.mount(lid)
                    Pv = pts[:, :3].astype(np.float64) @ M[:3, :3].T + M[:3, 3]
                    clouds.append(Pv)
                    dst = f'{base}/lasers/{lid}/{rec["step"]:06d}.npy'
                    buf = io.BytesIO()
                    np.save(buf, np.c_[rh_points(Pv), pts[:, 3]].astype(np.float32))
                    ctx.write(dst, buf.getvalue())
                    fr['lasers'].append({'name': k + 1, 'points': os.path.relpath(dst, base),
                                         'fields': ['x', 'y', 'z', 'intensity'], 'frame': 'vehicle'})
                for tid, (o, t) in boxes.items():
                    b = o['box3d']
                    c = rh_points(b['center'])
                    npts = sum(int(GT.in_box(P, b['center'], b['size'], GT.box_matrix(b['yaw_deg'])).sum())
                               for P in clouds)
                    fr['laser_labels'].append({
                        'box': {'center_x': c[0], 'center_y': c[1], 'center_z': c[2], 'length': b['size'][0],
                                'width': b['size'][1], 'height': b['size'][2],
                                'heading': _wrap(-math.radians(b['yaw_deg']))},
                        'type': t, 'id': tid, 'num_lidar_points_in_box': npts,
                        'detection_difficulty_level': 0, 'tracking_difficulty_level': 0})
                if fr['images'] or fr['lasers']:
                    lines.append(json.dumps(fr, separators=(',', ':'), default=_jsonable))
            ctx.write(f'{base}/frames.jsonl', ''.join(x + '\n' for x in lines))
            ctx.frames[f'{sn}/{seg}'] = len(lines)
            ctx.count('frames', len(lines))
            ctx.tick(f'Waymo-style {sn}: {run.key}')
    ctx.leave('TFRecord / protobuf / Parquet', 'this is a documented JSON layout mirroring Waymo Open\'s Frame and '
                                              'Label fields, not Waymo\'s files: the Waymo tools cannot read it')
    ctx.leave('range images (lasers)', 'point clouds in the vehicle frame (.npy) instead; Waymo stores range images')
    ctx.leave('label metadata (speed, acceleration), difficulty levels', 'not computed: difficulty 0 (UNKNOWN)')
    ctx.leave('3D labels of objects no camera saw', 'Falcon labels boxes per camera frame')
    if _any_fisheye(ctx):
        ctx.leave('fisheye cameras', 'Waymo\'s calibration models pinhole cameras')
    ctx.leave('time_of_day', 'not derived from the weather\'s sun angle')
    ctx.heur('camera names', 'Waymo\'s CameraName enum from each camera\'s mount yaw (FRONT within ±22.5°, …)')
    ctx.load = 'json per line: frames.jsonl; context.json has the calibrations (no Waymo devkit reads this layout)'
    ctx.frames_doc = {'vehicle': 'right-handed (x forward, y left, z up) at CARLA\'s vehicle origin; heading in '
                                 'radians counter-clockwise from x',
                      'global': 'CARLA\'s world mirrored (x, −y, z); pose.transform = vehicle -> global, row-major 4×4',
                      'camera_extrinsic': 'camera body (x forward, y left, z up) -> vehicle, as Waymo\'s',
                      'intrinsic': '[f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3] in OpenCV pixel-centre coordinates',
                      'camera_labels': 'box centre and size in pixel-edge coordinates; length along x, width along y',
                      'type': {WAYMO_TYPE_NAME[v]: [k for k, x in WAYMO_TYPE.items() if x == v]
                               for v in sorted(set(WAYMO_TYPE.values()))}}


# ===================================================================================================== BDD lanes
def lane_category(mtype: str, color: str):
    """(BDD100K lane category, laneStyle) of a CARLA lane marking, or (None, None) for none (or grass)."""
    t = str(mtype)
    if t in ('NONE', 'Grass', ''):
        return None, None
    if t == 'Curb':
        return 'road curb', 'solid'
    col = {'Standard': 'white', 'White': 'white', 'Yellow': 'yellow'}.get(str(color), 'other')
    double = t in ('SolidSolid', 'SolidBroken', 'BrokenSolid', 'BrokenBroken')
    style = 'dashed' if t in ('Broken', 'BrokenBroken', 'BottsDots') else 'solid'
    if t == 'Other':
        return 'single other', 'solid'
    return f'{"double" if double else "single"} {col}', style


def lane_lines(lanes: dict) -> list:
    """The map's lane boundaries as world polylines (CARLA frame, every LANE_DENSE_M) with their BDD category and
    style: [(points (N, 3), category, style)]. A boundary two lanes share is kept once."""
    col = {c: i for i, c in enumerate(lanes['columns'])}
    step = float(lanes.get('step_m', 2.0))
    groups = {}
    for p in lanes['points']:
        groups.setdefault((p[col['road']], p[col['section']], p[col['lane']]), []).append(p)
    pieces = []
    for pts in groups.values():
        pts.sort(key=lambda p: p[col['s']])
        for sgn, tcol, ccol in ((-1.0, 'left_type', 'left_color'), (1.0, 'right_type', 'right_color')):
            cur, key = [], None
            for i, p in enumerate(pts):
                k = lane_category(p[col[tcol]], p[col[ccol]])
                gap = i and math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) > 2.5 * step
                if (k != key or gap) and cur:
                    pieces.append((cur, key))
                    cur = [cur[-1]] if not gap and key and k == key else []
                key = k
                yaw = math.radians(p[col['yaw']])
                h = sgn * p[col['width']] / 2.0
                cur.append((p[0] - math.sin(yaw) * h, p[1] + math.cos(yaw) * h, p[2]))
            if cur:
                pieces.append((cur, key))
    out, cells = [], set()
    for pts, (cat, style) in pieces:
        if cat is None or len(pts) < 2:
            continue
        P = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        n = max(int(s[-1] / LANE_DENSE_M) + 1, 2)
        si = np.linspace(0.0, s[-1], n)
        D = np.stack([np.interp(si, s, P[:, k]) for k in range(3)], 1)
        cc = {(int(round(x / 0.25)), int(round(y / 0.25))) for x, y in D[:, :2]}
        near = sum(1 for (i, j) in cc if any((i + a, j + b) in cells for a in (-1, 0, 1) for b in (-1, 0, 1)))
        if near >= 0.8 * len(cc):
            continue                          # the neighbouring lane's side of the same line
        cells |= cc
        out.append((D, cat, style))
    return out


def project_lanes(run: Run, rec: dict, cid: str, lines: list, max_m: float, rng=None) -> list:
    """The lane lines visible in a camera frame: [(vertices [[x, y]] pixel-edge coordinates, category, style)]. A
    point is visible when it is within the lens's field and the image, within max_m, and (with a range image) not
    behind a nearer surface (its range ≤ the pixel's + max(0.5 m, 3 %))."""
    import cv2
    cam = run.cams[cid]
    W, H = cam['width'], cam['height']
    T_w_v = run.ego(rec)
    M = run.mount(cid, actual=True)
    T_c_w = np.linalg.inv(T_w_v @ M)
    out = []
    for D, cat, style in lines:
        Pa = D @ T_c_w[:3, :3].T + T_c_w[:3, 3]
        Po = Pa @ CARLA_TO_OPTICAL.T
        u, v, _, ok = GT.project(cam['lens'], Po)
        dist = np.linalg.norm(Po, axis=1)
        vis = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (dist <= max_m)
        if rng is not None and vis.any():
            iu = np.clip(u[vis].astype(int), 0, W - 1)
            iv = np.clip(v[vis].astype(int), 0, H - 1)
            r = rng[iv, iu]
            vis[np.flatnonzero(vis)[(r <= 0) | (dist[vis] > r + np.maximum(0.5, 0.03 * r))]] = False
        idx = np.flatnonzero(vis)
        if len(idx) < 2:
            continue
        for run_idx in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run_idx) < 2:
                continue
            pts = np.stack([u[run_idx], v[run_idx]], 1).astype(np.float32)
            simp = cv2.approxPolyDP(pts.reshape(-1, 1, 2), LANE_SEAM_PX, False).reshape(-1, 2)
            if len(simp) >= 2 and np.linalg.norm(simp[-1] - simp[0]) >= 2.0:
                out.append(([[round(float(x), 2), round(float(y), 2)] for x, y in simp], cat, style))
    return out


def _bdd_lanes(ctx: Ctx):
    """labels/lane_<split>.json (BDD100K lane marking labels, Scalabel form) + images/<split>/."""
    import cv2
    every = int(ctx.opt.get('every', 1))
    max_m = float(ctx.opt.get('lane_max_m', LANE_MAX_M))
    for split, runs in ctx.splits:
        sn = _split_name(split)
        frames = []
        for run in runs:
            lanes = run.lanes()
            if lanes is None:
                ctx.leave(f'run {run.key}', 'no lanes.json (a drive without camera ground truth)')
                ctx.tick(run.key)
                continue
            lines = lane_lines(lanes)
            for cid in _cameras(ctx, run):
                for rec in run.camera_steps(cid, every):
                    rel_r = run.own(rec, cid, 'range')
                    rng = None
                    if rel_r is not None:
                        rng = cv2.imread(run.p(rel_r), cv2.IMREAD_UNCHANGED).astype(np.float32) / GT.RANGE_SCALE
                    else:
                        ctx.leave(f'occlusion test ({cid}, {run.key})', 'no range image: lines behind vehicles kept')
                    name = f'{run.key}__{cid}__{rec["step"]:06d}.jpg'
                    ctx.place(run.p(run.own(rec, cid, 'rgb')), f'images/{sn}/{name}')
                    labels = []
                    for verts, cat, style in project_lanes(run, rec, cid, lines, max_m, rng):
                        labels.append({'id': str(len(labels)), 'category': cat,
                                       'attributes': {'laneDirection': 'parallel', 'laneStyle': style},
                                       'poly2d': [{'vertices': verts, 'types': 'L' * len(verts), 'closed': False}]})
                    frames.append({'name': name, 'videoName': f'{run.key}__{cid}', 'frameIndex': rec['step'],
                                   'timestamp': run.us(rec, cid) // 1000, 'labels': labels})
                    ctx.count('lane_labels', len(labels))
            ctx.tick(f'BDD lanes {sn}: {run.key}')
        ctx.json(f'labels/lane_{sn}.json', frames)
        ctx.frames[sn] = len(frames)
        ctx.count('images', len(frames))
    ctx.leave('crosswalks (laneDirection vertical)', 'lanes.json holds the driving lanes\' boundaries only')
    ctx.leave('lane masks (PNG)', 'only the polylines are written')
    ctx.leave('frame attributes (weather, scene, timeofday)', 'not mapped from the drive config')
    ctx.heur('lane style of mixed markings', 'SolidBroken / BrokenSolid are written double, solid')
    ctx.load = 'BDD100K / Scalabel frames list: labels/lane_<split>.json; images/<split>/'
    ctx.frames_doc = {'vertices': 'pixel-edge coordinates; each line is a map lane boundary (lanes.json, CARLA\'s '
                                  'markings) projected through the camera\'s actual mount and lens, every '
                                  f'{LANE_DENSE_M} m up to {max_m:g} m, hidden parts removed with the range image',
                      'categories': 'single/double white/yellow/other, road curb (CARLA marking type and colour)'}


# =========================================================================================================== MCAP
def _link(s: dict) -> str:
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in s['name']) or s['id']


def _mcap(ctx: Ctx):
    """One ROS 2 bag per run: /tf_static, /tf (map -> base_link), per camera image (compressed or raw) and
    camera_info, per lidar PointCloud2 (x y z intensity tag actor_id), per radar PointCloud2, Imu, NavSatFix."""
    import cv2
    from . import mcap_bag as B
    raw = ctx.opt.get('image', 'compressed') == 'raw'
    for split, runs in ctx.splits:
        for run in runs:
            rel = run.key if split is None else f'{split}/{run.key}'
            bag = B.BagWriter(os.path.join(ctx.out, rel), compression=ctx.opt.get('compression', 'none'))
            t0 = run.epoch_us * 1000
            ns = lambda rec, sid=None: run.us(rec, sid) * 1000                       # noqa: E731
            bag.add_topic('/tf_static', 'tf2_msgs/msg/TFMessage')
            bag.add_topic('/tf', 'tf2_msgs/msg/TFMessage')
            objects = str(ctx.opt.get('objects', 'none')).lower() in ('1', 'true', 'yes', 'ground_truth')
            if objects:                                  # the actors' boxes (core.logreplay reads them back)
                from .logreplay import ros2 as LR
                LR.register_vision_msgs()
                bag.add_topic(LR.OBJECTS_TOPIC, 'vision_msgs/msg/Detection3DArray')
                actor_rows = LR.actor_rows_by_step(run.dir)
            tfs = []
            for t in tf_list(run.rig):
                Tm = np.eye(4)
                x, y, z, w = t['rotation']
                Tm[:3, :3] = matrix_of((w, x, y, z))
                Tm[:3, 3] = t['translation']
                tfs.append(B.transform_stamped(Tm, t0, t['parent'], t['child']))
            bag.write('/tf_static', t0, {'transforms': tfs})
            topics = {}
            for s in run.rig['sensors']:
                L = _link(s)
                if s['type'].startswith('camera.') and s['id'] in run.cams:
                    topics[s['id']] = (f'/{L}/image' + ('' if raw else '/compressed'), f'/{L}/camera_info')
                    bag.add_topic(topics[s['id']][0], 'sensor_msgs/msg/Image' if raw else
                                  'sensor_msgs/msg/CompressedImage')
                    if _camera_info_model(run.cams[s['id']]) is not None:
                        bag.add_topic(topics[s['id']][1], 'sensor_msgs/msg/CameraInfo')
                    else:
                        ctx.leave(f'camera_info of {s["id"]}', 'its lens model has no ROS distortion model')
                elif s['type'] in ('lidar', 'radar'):
                    topics[s['id']] = (f'/{L}/points' if s['type'] == 'lidar' else f'/{L}/detections',)
                    bag.add_topic(topics[s['id']][0], 'sensor_msgs/msg/PointCloud2')
                elif s['type'] == 'imu':
                    topics[s['id']] = (f'/{L}/imu',)
                    bag.add_topic(topics[s['id']][0], 'sensor_msgs/msg/Imu')
                elif s['type'] == 'gnss':
                    topics[s['id']] = (f'/{L}/fix',)
                    bag.add_topic(topics[s['id']][0], 'sensor_msgs/msg/NavSatFix')
            pc_dt = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'tag', 'actor_id'],
                              'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', '<u4'],
                              'offsets': [0, 4, 8, 12, 16, 20], 'itemsize': 24})
            rd_dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('velocity', '<f4')])
            for rec in run.frames:
                t = ns(rec)
                bag.write('/tf', t, {'transforms': [B.transform_stamped(rh(run.ego(rec)), t, 'map', 'base_link')]})
                if objects and rec['step'] in actor_rows:
                    bag.write(LR.OBJECTS_TOPIC, t, {'header': B.header(t, 'map'), 'detections': [
                        dict(LR.detection3d(a), header=B.header(t, 'map')) for a in actor_rows[rec['step']]]})
                for sid, tp in topics.items():
                    s = run.sensors[sid]
                    fid = f'{_link(s)}_optical' if s['type'].startswith('camera.') else f'{_link(s)}_link'
                    if s['type'].startswith('camera.'):
                        r = run.file(rec, sid, 'rgb')
                        if r is None:
                            continue
                        ts = ns(rec, sid)
                        if raw:
                            img = cv2.imread(run.p(r), cv2.IMREAD_COLOR)
                            bag.write(tp[0], ts, B.image_msg(img, ts, fid, 'bgr8'))
                        else:
                            with open(run.p(r), 'rb') as fh:
                                bag.write(tp[0], ts, B.compressed_image(fh.read(), ts, fid, 'jpeg'))
                        ci = _camera_info_model(run.cams[sid])
                        if ci is not None:
                            cam = run.cams[sid]
                            bag.write(tp[1], ts, B.camera_info(cam['width'], cam['height'], ci[1], ci[0], ci[2],
                                                               ts, fid))
                    elif s['type'] == 'lidar' and run.file(rec, sid, 'lidar') is not None:
                        pts, tags, ids = run.lidar(rec, sid)
                        a = np.zeros(len(pts), pc_dt)
                        xyz = rh_points(pts[:, :3])
                        a['x'], a['y'], a['z'], a['intensity'] = xyz[:, 0], xyz[:, 1], xyz[:, 2], pts[:, 3]
                        a['tag'], a['actor_id'] = tags, ids
                        bag.write(tp[0], ns(rec, sid), B.pointcloud2(a, ns(rec, sid), fid))
                    elif s['type'] == 'radar' and run.file(rec, sid, 'radar') is not None:
                        det = run.radar(rec, sid)
                        xyz = rh_points(_radar_xyz(det)[0])
                        a = np.zeros(len(det), rd_dt)
                        a['x'], a['y'], a['z'], a['velocity'] = xyz[:, 0], xyz[:, 1], xyz[:, 2], det[:, 0]
                        bag.write(tp[0], ns(rec, sid), B.pointcloud2(a, ns(rec, sid), fid))
                ctx.count('steps')
            for s in run.of_type('imu'):
                ga, gg = float(s.get('noise_accel_stddev', 0.0)) ** 2, float(s.get('noise_gyro_stddev', 0.0)) ** 2
                for row in run.jsonl('imu', s['id']):
                    t = t0 + int(round(row['sim_t'] * 1e9))
                    a, g = row['accel'], row['gyro']
                    bag.write(topics[s['id']][0], t, {
                        'header': B.header(t, f'{_link(s)}_link'),
                        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 0.0},
                        'orientation_covariance': [-1.0] + [0.0] * 8,
                        'angular_velocity': {'x': -g[0], 'y': g[1], 'z': -g[2]},
                        'angular_velocity_covariance': [gg, 0, 0, 0, gg, 0, 0, 0, gg],
                        'linear_acceleration': {'x': a[0], 'y': -a[1], 'z': a[2]},
                        'linear_acceleration_covariance': [ga, 0, 0, 0, ga, 0, 0, 0, ga]})
            for s in run.of_type('gnss'):
                for row in run.jsonl('gnss', s['id']):
                    t = t0 + int(round(row['sim_t'] * 1e9))
                    m_lat = 111320.0
                    m_lon = 111320.0 * math.cos(math.radians(row['lat']))
                    sl = float(s.get('noise_lat_stddev', 0.0)) * m_lat
                    so = float(s.get('noise_lon_stddev', 0.0)) * m_lon
                    sa = float(s.get('noise_alt_stddev', 0.0))
                    bag.write(topics[s['id']][0], t, {
                        'header': B.header(t, f'{_link(s)}_link'), 'status': {'status': 0, 'service': 1},
                        'latitude': row['lat'], 'longitude': row['lon'], 'altitude': row['alt'],
                        'position_covariance': [sl * sl, 0, 0, 0, so * so, 0, 0, 0, sa * sa],
                        'position_covariance_type': 2})
            from .v2x import mcap_cpm                    # roadside units' CPMs (docs/v2x.md)
            mcap_cpm(bag, run.dir, t0)
            res = bag.close()
            ctx.bags = getattr(ctx, 'bags', {})
            ctx.bags[rel] = {'messages': res['messages'], 'topics': res['topics'], 'types': res['types'],
                             'file': os.path.relpath(res['file'], ctx.out)}
            ctx.nfiles += 2
            ctx.nbytes += res['bytes']
            ctx.frames[rel] = len(run.frames)
            ctx.count('messages', res['messages'])
            ctx.tick(f'MCAP: {run.key}')
    if str(ctx.opt.get('objects', 'none')).lower() in ('1', 'true', 'yes', 'ground_truth'):
        ctx.leave('ground truth labels', 'the bag carries the sensors, the transforms and the actors\' boxes '
                                         '(/falcon/objects); the lidar clouds carry CARLA\'s tag and actor id per point')
    else:
        ctx.leave('ground truth (labels, boxes)', 'the bag carries the sensors and transforms; the lidar clouds carry '
                                                  'CARLA\'s tag and actor id per point (option objects=true adds the '
                                                  'actors\' boxes)')
    ctx.leave('IMU orientation', 'CARLA\'s IMU gives none (covariance[0] = −1, REP 145); its compass is not converted')
    ctx.heur('NavSatFix covariance', 'CARLA\'s noise in degrees taken to metres at 111 320 m per degree')
    ctx.load = '`ros2 bag info <bag>`, `ros2 bag play <bag>`, Foxglove, or the mcap / mcap-ros2-support packages'
    ctx.frames_doc = {'map': 'CARLA\'s world mirrored (x, −y, z): ROS\'s right-handed axes',
                      'base_link': 'CARLA\'s vehicle origin, ROS axes (x forward, y left, z up); the URDF export\'s',
                      '<sensor>_link': 'the sensor body, REP 103; cameras\' images in <sensor>_optical (z forward)',
                      'clouds': 'x y z in the sensor link (CARLA\'s with y mirrored); lidar intensity CARLA\'s, tag '
                                'the CARLA semantic tag, actor_id CARLA\'s (0 for the map\'s meshes); radar velocity '
                                'CARLA\'s: (target − radar) · ray, positive receding',
                      'imu': 'linear acceleration (x, −y, z) and angular velocity (−x, y, −z) of CARLA\'s numbers: a '
                             'mirrored frame flips the pseudo-vector\'s other components',
                      'time': 'ns since the Unix epoch: the run\'s start (run.json, UTC) + simulation time'}


def _camera_info_model(cam: dict):
    """(distortion model, K OpenCV, d) for a camera's CameraInfo, or None when ROS has no model for its lens."""
    lens = cam.get('lens') or {}
    K = K_opencv(cam['K'])
    if cam['type'] == 'camera.pinhole' or lens.get('model') == 'rectilinear':
        return 'plumb_bob', K, [0.0] * 5
    if lens.get('model') in ('equidistant', 'kannala_brandt'):
        return 'equidistant', K, [float(v) for v in (lens.get('k') or [0, 0, 0, 0])][:4]
    return None


# =========================================================================================================== export
def _occ3d(ctx: Ctx):
    """Semantic occupancy: Occ3D-nuScenes, OpenOcc v2 or SurroundOcc (core.occexport; option variant)."""
    from .occexport import export_occ
    export_occ(ctx, Run, {'rh': rh, 'quat_of': quat_of, 'camera_T': camera_T, 'K_opencv': K_opencv, 'token': token,
                          'ExportError': ExportError})


_EXPORTERS = {'coco': _coco, 'yolo': _yolo, 'kitti': _kitti, 'nuscenes': _nuscenes, 'waymo': _waymo,
              'bdd_lanes': _bdd_lanes, 'mcap': _mcap, 'occ3d': _occ3d}


def _resolve_splits(run_dirs, options) -> list:
    sp = options.get('splits')
    if not sp:
        dirs = [os.path.abspath(d) for d in run_dirs]
        if not dirs:
            raise ExportError('no runs to export')
        if len(set(dirs)) != len(dirs):
            raise ExportError('a run is listed twice')
        return [(None, dirs)]
    bad = sorted(set(sp) - set(SPLITS))
    if bad:
        raise ExportError(f'unknown split(s) {", ".join(bad)}: use {", ".join(SPLITS)}')
    where = {}
    for name in SPLITS:
        for d in sp.get(name) or []:
            a = os.path.abspath(d)
            if a in where:
                raise ExportError(f'{a} is in {where[a]} and {name}: a run may be in one split only (leakage)')
            where[a] = name
    for d in run_dirs:
        if os.path.abspath(d) not in where:
            raise ExportError(f'{d} is in no split')
    return [(name, [d for d, n in where.items() if n == name]) for name in SPLITS if sp.get(name)]


def _prepare_out(out_dir: str, overwrite: bool):
    if os.path.isfile(out_dir):
        raise ExportError(f'{out_dir} is a file')
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not os.path.isfile(os.path.join(out_dir, MANIFEST)):
            raise ExportError(f'{out_dir} holds files Falcon did not export: choose an empty folder')
        if not overwrite:
            raise ExportError(f'{out_dir} holds an earlier export (options overwrite: true replaces it)')
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def export(run_dirs: list, fmt: str, out_dir: str, options: dict | None = None, progress=None) -> dict:
    """Export runs (falcon-raw folders) as `fmt` (FORMATS) into out_dir; returns the manifest (falcon_export.json).

    options: 'splits' {train|val|test: [run dirs]}; 'cameras' [ids]; 'every' n (camera formats: every n-th step);
    'link' copy | hardlink | symlink (images); 'overwrite' (replace an earlier export in out_dir); nuscenes:
    'version' (default v1.0-falcon), 'keyframe_hz'; kitti: 'camera'; bdd_lanes: 'lane_max_m'; mcap: 'image'
    compressed | raw, 'compression' none | zstd, 'objects' true (every actor's box per step as a
    vision_msgs/Detection3DArray on /falcon/objects, frame 'map'); occ3d: 'variant' occ3d | openocc | surroundocc,
    'lidar', 'every', 'aggregate' (static sweeps either side), 'camera_ray_step' (core.occexport). Raises
    ExportError."""
    options = dict(options or {})
    if fmt not in _EXPORTERS:
        raise ExportError(f'unknown format {fmt!r}: one of {", ".join(FORMATS)}')
    splits = _resolve_splits(run_dirs, options)
    names, runs = {}, []
    for split, dirs in splits:
        rs = []
        for d in dirs:
            base = os.path.basename(os.path.normpath(d))
            key = base if base not in names.values() else f'{base}-{len(names) + 1}'
            names[d] = key
            rs.append(Run(d, key))
        runs.append((split, rs))
    out_dir = os.path.abspath(out_dir)
    if any(out_dir == r.dir or out_dir.startswith(r.dir + os.sep) for _, rs in runs for r in rs):
        raise ExportError('the export folder may not be inside a run')
    _prepare_out(out_dir, bool(options.get('overwrite')))
    ctx = Ctx(fmt, out_dir, options, progress, runs)
    ctx.run_info, ctx.frames_doc = {}, {}
    try:
        _EXPORTERS[fmt](ctx)
    except ExportError:
        raise
    except (OSError, ValueError, KeyError) as e:
        raise ExportError(f'{fmt} export failed: {type(e).__name__}: {e}') from e
    if not any(ctx.frames.values()):
        raise ExportError(f'nothing to export as {fmt}: no run has frames it can use (manifest left_out: '
                          + '; '.join(f'{k}: {v}' for k, v in ctx.left_out.items()) + ')')
    inputs = {}
    for _, rs in runs:
        for r in rs:
            inputs.update(r.inputs())
    settings = {k: v for k, v in options.items() if k != 'splits'}
    doc = {
        'schema': EXPORT_SCHEMA, 'format': fmt, 'created': utc_now(),
        'provenance': block(inputs, None, settings),
        'runs': [dict({'key': r.key, 'dir': r.dir, 'split': split, 'name': r.cfg.get('name'), 'seed': r.cfg.get('seed'),
                       'steps': len(r.frames), 'fps': r.fps, 'town': r.cfg.get('town') or r.cfg.get('xodr'),
                       'run_json_sha256': sha256_file(r.p('run.json')),
                       'config_sha256': r.run.get('config_sha256'), 'faults_applied': r.faults() or None},
                      **ctx.run_info.get(r.key, {})) for split, rs in runs for r in rs],
        'splits': {(s or 'all'): [r.key for r in rs] for s, rs in runs},
        'frames_written': ctx.frames, 'counts': ctx.counts, 'files': ctx.nfiles, 'bytes': ctx.nbytes,
        'coordinates': ctx.frames_doc, 'left_out': [{'what': k, 'why': v} for k, v in ctx.left_out.items()],
        'heuristic': [{'what': k, 'how': v} for k, v in ctx.heuristic.items()], 'notes': ctx.notes,
        'load': ctx.load}
    if fmt == 'nuscenes':
        doc['versions'] = ctx.versions
    if fmt == 'mcap':
        doc['bags'] = getattr(ctx, 'bags', {})
    if any(r.faults() for _, rs in runs for r in rs):
        doc['notes'].append('some runs have injected faults (runs[].faults_applied): the data is as recorded and the '
                            'calibration states the plan, as a real faulty rig\'s would')
    with open(os.path.join(out_dir, MANIFEST), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False, default=_jsonable)
        fh.write('\n')
    if progress:
        progress(1.0, f'{fmt}: done')
    return doc



OCC_VARIANTS = ('occ3d', 'openocc', 'surroundocc')
CHOICES = FORMATS[:-1] + OCC_VARIANTS          # what Data ▸ Browse ▸ Export offers: the occupancy variants by name


def export_argv(run_dir: str, choice: str, out: str) -> list:
    """`falcon export …` for one of CHOICES (an occupancy variant becomes --format occ3d --option variant=V), as
    Data ▸ Browse ▸ Export queues it (the arguments after `python -m falcon_autonomy`)."""
    if choice in OCC_VARIANTS:
        return ['export', run_dir, '--format', 'occ3d', '--out', out, '--option', f'variant={choice}']
    if choice not in FORMATS:
        raise ValueError(f'unknown export format {choice!r}: one of {", ".join(CHOICES)}')
    return ['export', run_dir, '--format', choice, '--out', out]


def main_export(argv) -> int:
    """`falcon export RUN... --format F --out DIR [--split-json FILE] [--option KEY=VALUE]... [--overwrite]`:
    0 on success, 2 on failure."""
    ap = argparse.ArgumentParser(prog='falcon export', description='export recorded drives as a dataset format')
    ap.add_argument('runs', nargs='*', help='run folders (falcon-raw); may be empty with --split-json')
    ap.add_argument('--format', '-f', required=True, choices=FORMATS)
    ap.add_argument('--out', '-o', required=True, help='the export folder (empty, or an earlier export)')
    ap.add_argument('--split-json', help='JSON file {"train": [run dirs], "val": [...], "test": [...]}')
    ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE',
                    help='an export option (VALUE parsed as JSON when it can be), e.g. every=5, cameras=["cam-front"]')
    ap.add_argument('--overwrite', action='store_true', help='replace an earlier export in --out')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    opts = {}
    for kv in a.option:
        k, eq, v = kv.partition('=')
        if not eq:
            print(f'failed: option {kv!r} is not KEY=VALUE', flush=True)
            return 2
        try:
            opts[k] = json.loads(v)
        except ValueError:
            opts[k] = v
    if a.overwrite:
        opts['overwrite'] = True
    if a.split_json:
        try:
            opts['splits'] = read_json(a.split_json)
        except (OSError, ValueError) as e:
            print(f'failed: cannot read {a.split_json}: {e}', flush=True)
            return 2

    def prog(f, text):
        print(f'{100 * f:5.1f} % {text}', flush=True)
    try:
        doc = export(a.runs, a.format, a.out, opts, prog)
    except ExportError as e:
        print(f'failed: {e}', flush=True)
        return 2
    print(f'exported {len(doc["runs"])} run(s) as {a.format} to {a.out}: {doc["files"]} files, '
          f'{doc["bytes"] / 1e6:.1f} MB; {len(doc["left_out"])} item(s) left out (see {MANIFEST})', flush=True)
    return 0
