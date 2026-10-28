"""Cameras with their ground truth (runs in CARLA's Python): a camera's planned views (core.lens), the RGB image
remapped through its lens, and — when it records labels — CARLA's instance and depth cameras beside every view, read
into the output image's label layers and objects by core.groundtruth (Vantage's measured method).

Per recorded frame and camera (falcon-raw):
* rgb/<cam>/<n>.jpg — the image (a fisheye: its views remapped, Unreal's vignette removed, seams feathered);
* semantic/<cam>/<n>.png — uint8 class per pixel: CARLA's tag (the instance camera's red channel, measured equal to
  the semantic camera's), the ego's own pixels as class 29 'ego', traffic-light heads and signs as their class where
  their boxes claim the pixels;
* instance/<cam>/<n>.png — uint16: 0 none, else the object's id in objects.json;
* range/<cam>/<n>.png — uint16 = round(range_m × 256) along each pixel's own ray (0: nothing within 256 m);
* objects/<cam>/<n>.json — every object seen: id, track_id (the CARLA actor id, the same in every frame; map objects
  'level:<k>'), class, category (nuScenes), bbox (tight, from its pixels), bbox_full (its 3D box through the lens),
  truncated, occluded, range_m, pixels, box3d (vehicle frame: center, size [l w h], yaw), and its mask as COCO RLE.

Imported (OpenDRIVE) worlds. Measured on CARLA 0.9.16 (central Bengaluru, a car 10 m ahead): the road and sidewalk
mesh generate_opendrive_world builds is drawn by the RGB camera but NOT by the label cameras — the depth camera sees
nothing there (99.998 % of the lower third at its 1000 m far limit) and the instance and semantic cameras tag it sky
(11) — while actors are seen normally (the car: 1346 instance pixels, 1340 depth pixels at a median 6.1 m). The
semantic lidar does hit the mesh, tagged 0. These worlds are flat (OSM has no elevation), so the ground is found
geometrically: a pixel tagged sky whose ray meets the map's ground plane within GROUND_MAX_M is given that point's
range and the class of the map's lanes there (a 0.25 m raster: driving lanes → road, sidewalks → sidewalk, other lane
types → ground). The frame's objects.json reports how many pixels ("ground_relabelled").
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..core import groundtruth as GT
from ..core import lens as LENS
from ..core.rig import carla_rotation
from .live import transform_matrix

GT_LABELS = ('semantic', 'instance', 'depth', 'boxes2d', 'boxes3d')
GROUND_MAX_M = 250.0
LEVEL_LABELS = (('Car', 'vehicle.parked', (2.5, 7.5), True), ('Truck', 'vehicle.parked', (4.0, 15.0), True),
                ('Bus', 'vehicle.parked', (6.0, 20.0), True), ('Motorcycle', 'vehicle.parked', (1.2, 3.0), True),
                ('Bicycle', 'vehicle.parked', (1.0, 2.5), True), ('TrafficLight', 'static_object.traffic_light',
                                                                   (0.0, 3.0), False),
                ('TrafficSigns', 'static_object.traffic_sign', (0.0, 4.0), False))


def wants_gt(s: dict) -> bool:
    return any(k in (s.get('labels') or []) for k in GT_LABELS)


class LaneRaster:
    """The map's lanes rasterised top-down (CARLA world x, y), to name ground points: 1 road (driving lanes and their
    junctions), 2 sidewalk, 25 ground (other lane types: shoulders, parking, borders)."""
    CELL = 0.25

    def __init__(self, carla, cmap, step=1.0):
        wps = cmap.generate_waypoints(step)
        pts, tags = [], []
        LT = carla.LaneType
        for wp in wps:
            t = wp.transform
            yaw = math.radians(t.rotation.yaw)
            rx, ry = -math.sin(yaw), math.cos(yaw)
            tag = 1 if wp.lane_type == LT.Driving else 2 if wp.lane_type == LT.Sidewalk else 25
            half = wp.lane_width / 2.0
            for o in np.arange(-half, half + 1e-6, self.CELL):
                for a in np.arange(-step / 2, step / 2 + 1e-6, self.CELL):
                    pts.append((t.location.x + rx * o + math.cos(yaw) * a, t.location.y + ry * o + math.sin(yaw) * a))
                    tags.append(tag)
        self.z0 = float(np.median([wp.transform.location.z for wp in wps])) if wps else 0.0   # flat worlds
        P = np.asarray(pts, float) if pts else np.zeros((0, 2))
        self.x0, self.y0 = (P.min(0) - 2.0) if len(P) else (0.0, 0.0)
        ij = np.floor((P - (self.x0, self.y0)) / self.CELL).astype(int)
        self.w, self.h = (ij.max(0) + 2) if len(ij) else (1, 1)
        self.grid = np.zeros((self.h, self.w), np.uint8)
        order = np.argsort(-np.asarray(tags))           # roads painted last: a road wins where lanes overlap
        self.grid[ij[order, 1], ij[order, 0]] = np.asarray(tags, np.uint8)[order]

    def lookup(self, xy) -> np.ndarray:
        ij = np.floor((np.asarray(xy) - (self.x0, self.y0)) / self.CELL).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.w) & (ij[:, 1] >= 0) & (ij[:, 1] < self.h)
        out = np.full(len(ij), 25, np.uint8)
        out[ok] = self.grid[ij[ok, 1], ij[ok, 0]]
        out[ok & (out == 0)] = 25
        return out


class CameraRecorder:
    """One rig camera: its plan, its view sensors, and writing a frame with its ground truth."""

    def __init__(self, w, world, lib, ego, s: dict, cameras, pose_actual: dict):
        self.w, self.s, self.cameras = w, s, cameras
        self.plan = LENS.plan_camera(dict(s, pose=pose_actual))      # the views at the actual mount (faults)
        self.gt = wants_gt(s)
        self.cg = GT.CameraGT(self.plan) if self.gt else None
        self.pose = GT.camera_pose(pose_actual)          # the actual mount (faults included) for the labels
        self.views = []                                  # per view: {'rgb': sensor, 'instance': .., 'depth': ..}
        fisheye = s['type'] == 'camera.fisheye'
        for v in self.plan['views']:
            vp = v['pose']
            tf = w.carla.Transform(w.carla.Location(x=vp['x'], y=vp['y'], z=vp['z']),
                                   w.carla.Rotation(pitch=vp['pitch'], yaw=vp['yaw'], roll=vp['roll']))
            d = {'rgb': cameras.spawn(world, lib, 'sensor.camera.rgb', v['width'], v['height'], v['fov'], tf, ego,
                                      attrs=v.get('attributes') if fisheye else None)}
            if self.gt:
                for kind, bp in (('instance', 'sensor.camera.instance_segmentation'), ('depth', 'sensor.camera.depth')):
                    d[kind] = cameras.spawn(world, lib, bp, v['width'], v['height'], v['fov'], tf, ego,
                                            attrs=LENS.LABEL_VIEW_ATTRIBUTES if fisheye else None)
            self.views.append(d)
        self.devig = LENS.devignette_layers(self.plan) if fisheye else None
        self.o2v = LENS.optical_to_vehicle(pose_actual)
        self.assoc = {}
        self.relabelled = 0

    def kinds(self):
        return ['rgb'] + (['semantic', 'instance', 'range', 'objects'] if self.gt else [])

    def collect(self, frame):
        """Every view's images of this frame (by frame number)."""
        return [{k: self.cameras.get(sen.id, frame) for k, sen in d.items()} for d in self.views]

    def write(self, imgs, n: int, out_dir: str, boxes: dict, raster=None, image_faults=None) -> dict:
        """(a worker thread) The frame's files: {kind: relative path}."""
        files = {}
        sid = self.s['id']
        rgb_views = [np.frombuffer(v['rgb'].raw_data, np.uint8).reshape(v['rgb'].height, v['rgb'].width, 4)[:, :, :3]
                     for v in imgs]
        if self.s['type'] == 'camera.fisheye':
            img = LENS.remap_image(self.plan, rgb_views, self.devig)
        else:
            img = np.ascontiguousarray(rgb_views[0])
        if image_faults is not None:
            from ..core import faults as F
            fl, rng, K, omega = image_faults
            img, _ = F.apply_image(img, fl, rng, K, omega)
        files['rgb'] = _write(out_dir, 'rgb', sid, n, 'jpg', img)
        if not self.gt:
            return files
        cg = self.cg
        at = self.plan['atlas']
        ah, aw = int(at['height']), int(at['width'])
        atl = {}
        for kind in ('instance', 'depth'):
            a32 = np.zeros((ah, aw), np.uint32)
            for vi, v in enumerate(imgs):
                y0 = int(at['offsets'][vi])
                im = v[kind]
                a32[y0:y0 + im.height, :im.width] = GT.packed(im.raw_data, im.width, im.height)
            atl[kind] = a32
        tags, raw = GT.unpack_instance(cg.gather(atl['instance']))
        sem = GT.CLASS_OF_TAG[tags]
        rng = cg.range_from_packed(atl['depth'], atl['instance'])
        if raster is not None:
            sem, rng = self.relabel_ground(sem, tags, rng, boxes['T_ego'], raster)
        found, sem = GT.objects_in_frame(sem, raw, rng, cg.rays, self.pose, self.plan['lens'], boxes['known'],
                                         boxes['level'], boxes['ego_id'], self.assoc, parts=boxes['parts'])
        h, w = sem.shape
        inst = np.zeros((h, w), np.uint16)
        objs = []
        for f in sorted(found, key=lambda f: f['obj']['reg']):
            o = f['obj']
            inst.ravel()[f['pix']] = o['reg']
            geo = GT.object_geometry(o, self.pose, self.plan['lens'], w, h, cg.rays, rng, pix=f['pix'])
            Rb = np.asarray(o['R'], float)
            objs.append({'id': int(o['reg']), 'track_id': o.get('track_id'), 'actor_id': o.get('actor_id'),
                         'category': o['category'], 'class': GT.CLASS_NAMES.get(f['class'], 'unlabelled'),
                         'bbox': f['bbox'], 'pixels': f['pixels'], **geo,
                         'box3d': {'center': [round(float(x), 4) for x in o['center']],
                                   'size': [round(float(x), 4) for x in o['size']],
                                   'yaw_deg': round(math.degrees(math.atan2(Rb[1, 0], Rb[0, 0])), 4),
                                   'frame': 'vehicle'},
                         'rle': GT.rle_of_pixels(f['pix'], h, w)})
        files['semantic'] = _write(out_dir, 'semantic', sid, n, 'png', sem.astype(np.uint8))
        files['instance'] = _write(out_dir, 'instance', sid, n, 'png', inst)
        files['range'] = _write(out_dir, 'range', sid, n, 'png', GT.range_png(rng))
        rel = os.path.join('objects', sid, f'{n:06d}.json')
        os.makedirs(os.path.join(out_dir, os.path.dirname(rel)), exist_ok=True)
        with open(os.path.join(out_dir, rel), 'w', encoding='utf-8') as fh:
            json.dump({'objects': objs, 'width': w, 'height': h, 'ground_relabelled': self.relabelled}, fh,
                      separators=(',', ':'), default=_jsonable)
        files['objects'] = rel
        return files

    def relabel_ground(self, sem, tags, rng, T_ego, raster):
        """Pixels tagged sky whose ray meets the map's ground plane (within GROUND_MAX_M): their range is that point's
        and their class the map's lanes' there. Returns (semantic, range)."""
        m = tags == 11
        idx = np.flatnonzero(m.ravel())
        self.relabelled = 0
        if not len(idx):
            return sem, rng
        rays = self.cg.rays.reshape(-1, 3)[idx]           # optical, unit
        D = (T_ego[:3, :3] @ (self.o2v @ rays.T)).T       # in the world
        O = T_ego[:3, :3] @ self.pose['pos'] + T_ego[:3, 3]
        down = D[:, 2] < -1e-4
        t = np.where(down, (raster.z0 - O[2]) / np.where(down, D[:, 2], -1.0), np.inf)
        hit = down & (t > 0) & (t <= GROUND_MAX_M)
        if not hit.any():
            return sem, rng
        P = O + D[hit] * t[hit, None]
        sem, rng = sem.copy(), rng.copy()
        sem.ravel()[idx[hit]] = raster.lookup(P[:, :2])
        rng.ravel()[idx[hit]] = t[hit].astype(rng.dtype)
        self.relabelled = int(hit.sum())
        return sem, rng

def _write(out_dir, kind, sid, n, ext, arr):
    import cv2
    rel = os.path.join(kind, sid, f'{n:06d}.{ext}')
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, arr, [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == 'jpg' else [])
    if not ok:
        raise OSError(f'could not write {path}')
    return rel


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def scene_boxes(w, world, ego, level: list) -> dict:
    """The objects a camera's ground truth can name at this step, in the ego's vehicle frame: every vehicle and walker
    of the world (track id = actor id), the map's parked vehicles, and the lights' heads and signs (parts)."""
    et = ego.get_transform()
    T_ego = transform_matrix(et.location, et.rotation)
    Ti = np.linalg.inv(T_ego)
    known = {}
    for a in world.get_actors():
        if not a.type_id.startswith(('vehicle.', 'walker.')):
            continue
        try:
            t = a.get_transform()
            bb = a.bounding_box
        except RuntimeError:
            continue
        M = transform_matrix(t.location, t.rotation)
        c_w = M[:3, :3] @ np.array([bb.location.x, bb.location.y, bb.location.z]) + M[:3, 3]
        known[a.id] = {'key': ('actor', a.id), 'actor_id': a.id, 'track_id': a.id, 'reg': int(a.id) % 60000 or 1,
                       'category': GT.category_of(a.type_id, dict(getattr(a, 'attributes', {}) or {})),
                       'center': Ti[:3, :3] @ c_w + Ti[:3, 3], 'R': Ti[:3, :3] @ M[:3, :3],
                       'size': [2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z], 'instance_token': None}
    lv, parts = [], []
    ex = T_ego[:3, 3]
    for k, e in enumerate(level):
        if math.hypot(e['loc'][0] - ex[0], e['loc'][1] - ex[1]) > 100.0:
            continue
        Rw = carla_rotation(e['rot'][2], e['rot'][0], e['rot'][1])
        o = {'key': ('level', k), 'level_label': e['label'], 'actor_id': e.get('actor_id'),
             'track_id': f'level:{k}', 'reg': 60000 + k, 'category': e['category'],
             'center': Ti[:3, :3] @ np.asarray(e['loc']) + Ti[:3, 3], 'R': Ti[:3, :3] @ Rw, 'size': list(e['size']),
             'instance_token': None}
        if e['label'] in GT.PART_CLASS:
            o['part_class'] = GT.PART_CLASS[e['label']]
            parts.append(o)
        else:
            lv.append(o)
    return {'known': known, 'level': lv, 'parts': parts, 'ego_id': int(ego.id), 'T_ego': T_ego}


def level_boxes(w, world) -> list:
    """The map's own objects (level geometry, not actors) from World.get_level_bbs, in CARLA's world (read before any
    vehicle is spawned: the call lists vehicles' boxes too): parked vehicles, traffic-light heads, signs."""
    COL = getattr(w.carla, 'CityObjectLabel', None)
    if COL is None or not hasattr(world, 'get_level_bbs'):
        return []
    owner = {}
    try:
        for a in world.get_actors().filter('traffic.traffic_light'):
            for b in (a.get_light_boxes() if hasattr(a, 'get_light_boxes') else []):
                owner[tuple(np.round([b.location.x, b.location.y, b.location.z], 2))] = int(a.id)
    except RuntimeError:
        pass
    out = []
    for label, cat, (lo, hi), annotated in LEVEL_LABELS:
        lab = getattr(COL, label, None)
        if lab is None:
            continue
        try:
            bbs = list(world.get_level_bbs(lab))
        except RuntimeError:
            continue
        for b in bbs:
            L, W, H = 2 * b.extent.x, 2 * b.extent.y, 2 * b.extent.z
            if annotated and not (lo <= L <= hi and 0.3 <= W <= 4.0 and 0.5 <= H <= 5.0):
                continue
            if not annotated and max(L, W, H) > hi:
                continue
            out.append({'label': label, 'category': cat, 'loc': [b.location.x, b.location.y, b.location.z],
                        'rot': (b.rotation.pitch, b.rotation.yaw, b.rotation.roll), 'size': [L, W, H],
                        'actor_id': owner.get(tuple(np.round([b.location.x, b.location.y, b.location.z], 2)))
                        if label == 'TrafficLight' else None})
    return out

