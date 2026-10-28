"""Ground truth of a simulated drive (schema falcon.ground_truth/1): the class table, the reader, the mask loaders and
the numeric helpers a worker labels its camera images and point clouds with. numpy only at import (OpenCV is imported
by the PNG loaders when they are called), so it runs in CARLA's own Python as well as in Falcon. Ported from Vantage
(vantage.core.groundtruth); a camera is described by core.lens.plan_camera()'s plan (its views, atlas and remap
tables) and its lens block.

Frames: everything here is in CARLA's vehicle frame (x forward, y right, z up, left-handed; metres) unless said
otherwise. A camera's pose for the functions below is {'R': its CARLA rotation matrix (actor axes -> vehicle, as
carla.Transform.get_matrix), 'pos': (3,)} (camera_pose() makes it from a pose dict); its rays are in its optical frame
(OpenCV: x right, y down, z along the view; core.lens.CARLA_TO_OPTICAL from the actor axes). A box is {'center',
'size' (full length, width, height), 'R' (its CARLA rotation matrix; box_matrix(yaw) for a yaw)}.

What a drive with ground truth holds (default layout; every path in the index is relative to the output folder OUT)
-------------------------------------------------------------------------------------------------------------------
* OUT/<INDEX> (ground_truth/index.json by default): {'schema', 'classes', 'cameras': {TAG: {'channel', 'width',
  'height', 'frames': [...]}}, 'lidars': {TAG: {'channel', 'sweeps': [...]}}, 'conventions', 'notes'}; every camera
  frame lists its image, its label images and its objects (see FRAME_KEYS and OBJECT_KEYS), every lidar sweep its
  nuScenes-lidarseg and -panoptic files.
* Per camera frame: <name>.semantic.png (uint8, a class id per pixel), .instance.png (uint16, 0 = no object, else the
  object's id), .range.png (uint16 = round(range_m · 256) along the pixel's ray, 0 = nothing within 256 m).
* COCO detection files per camera (bbox [x, y, w, h], area, category, segmentation as compressed RLE, iscrowd 0).
* OUT/nuscenes/lidarseg/<NUSCENES_VERSION>/<token>_lidarseg.bin (uint8, the nuScenes-lidarseg category index of every
  point of the sweep's .pcd.bin, in its order) and panoptic/<NUSCENES_VERSION>/<token>_panoptic.npz ('data': uint16,
  category index × 1000 + instance id), with lidarseg.json, panoptic.json and category.json (with 'index') in
  <NUSCENES_VERSION>/ (v1.0-falcon), the layouts of nuScenes-lidarseg and nuScenes-panoptic (devkit 1.1+).

Classes
-------
CARLA 0.9.16 tags every mesh with one of 29 labels (carla.CityObjectLabel, values 0–28, read from the 0.9.16 client
module; the semantic camera writes the tag in its red channel). The class id is the tag. Each class maps to the
closest of nuScenes-lidarseg's 32 categories (NUSCENES_LIDARSEG, with their indices), and has CARLA's CityScapes
palette colour (checked against ColorConverter.CityScapesPalette on a 0.9.16 server). One class is added: 'ego'
(id 29), the pixels and points of the vehicle carrying the rig (nuScenes 'vehicle.ego'), told apart by its actor id.
A point or object of an actor the worker knows (a vehicle, a walker, a prop it spawned) takes the actor's own nuScenes
category (an ambulance is vehicle.emergency.ambulance, a child human.pedestrian.child, a cone
movable_object.trafficcone); everything else takes its class's.

Measured on a CARLA 0.9.16 server (Town10HD, 2026-10-27) and used here
-----------------------------------------------------------------------
* sensor.camera.instance_segmentation: R = the semantic tag (the same as the semantic camera's at every pixel of a
  640 × 480 frame), G + 256·B = the CARLA actor id for actors (vehicles, walkers, props, traffic lights), an unrelated
  id for level meshes (the map's parked cars, signs, buildings).
* sensor.camera.depth: planar depth (along the view's axis) in metres = 1000 · (R + 256 G + 65536 B) / (2²⁴ − 1).
* sensor.lidar.ray_cast_semantic with the ray-cast lidar's attributes, spawned on the same step: the same points per
  channel, directions within 3·10⁻⁶ degrees, ranges differing only by the ray-cast lidar's noise (1 σ 1.5 cm set,
  1.52 cm measured); object_idx = the CARLA actor id (0 for level meshes), object_tag = the semantic tag. Each point is
  24 bytes: x, y, z, cos_inc_angle (float32), object_idx, object_tag (uint32).
* The map's 15 traffic lights (Town10HD) have 62 heads: TrafficLight.get_light_boxes, the same boxes as the level's
  TrafficLight boxes. Of the pixels whose 3D point lies inside a head box 89.5 % are tagged traffic_light, 9.5 % pole,
  1 % sign, 99.8 % carry the light's actor id (45 views); inside the 147 sign boxes 85.6 % sign, 11.2 % pole. A head or
  a sign is therefore an object of its own whose box claims the static pixels inside it (claim_parts): they become
  its class. (No light of the map is tagged Other: its 287 Other objects are power lines, street lights, banners.)

The range image
---------------
Each output pixel's range is the planar depth of its view at the pixel's exact ray, times |(a, b, 1)| of that ray: the
inverse depth of the four view pixels around it interpolated bilinearly (exact on any plane, a pinhole view's 1/z
being linear in its pixel coordinates) when they are one surface (the same instance and tag, or all the ground's
meshes), else the nearest view pixel's (an object's edge is never blurred into a value between). On a real 10 s drive
it agrees with the lidar on the ground to a median of −0.9 mm (median absolute 2.2 cm, the lidar's noise carried
along the camera's grazing rays).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..sim.frames import CARLA_TO_OPTICAL, carla_matrix
from .lens import r_of_theta

SCHEMA = 'falcon.ground_truth/1'
INDEX = os.path.join('ground_truth', 'index.json')       # the default place of the index in an output folder
NUSCENES_VERSION = 'v1.0-falcon'
LIDARSEG_DIR = 'lidarseg'
PANOPTIC_DIR = 'panoptic'
RANGE_SCALE = 256.0                    # range PNG: uint16 = round(range_m · 256)
RANGE_MAX_M = 65535 / RANGE_SCALE      # 255.996 m; farther (or nothing) is 0
EGO_CLASS = 29
PANOPTIC_DIV = 1000                    # nuScenes-panoptic: category index · 1000 + instance id
FRAME_KEYS = ('sample_data_token', 'timestamp', 'image', 'semantic', 'instance', 'range', 'objects')
OBJECT_KEYS = ('id', 'actor_id', 'instance_token', 'category', 'class', 'bbox', 'pixels', 'bbox_full', 'truncated',
               'occluded', 'range_m', 'box3d')

# nuScenes-lidarseg categories (v1.0, category.json 'index'), with nuScenes' own short descriptions
NUSCENES_LIDARSEG = (
    (0, 'noise', 'Any lidar return that does not correspond to a physical object, such as dust, vapor, noise, fog, '
                 'raindrops, smoke and reflections.'),
    (1, 'animal', 'All animals, e.g. cats, rats, dogs, deer, birds.'),
    (2, 'human.pedestrian.adult', 'Adult subcategory.'),
    (3, 'human.pedestrian.child', 'Child subcategory.'),
    (4, 'human.pedestrian.construction_worker', 'Construction worker.'),
    (5, 'human.pedestrian.personal_mobility', 'A small electric or self-propelled vehicle, e.g. skateboard, segway, '
                                              'or scooters, on which the person typically travels in a upright '
                                              'position.'),
    (6, 'human.pedestrian.police_officer', 'Police officer.'),
    (7, 'human.pedestrian.stroller', 'Strollers.'),
    (8, 'human.pedestrian.wheelchair', 'Wheelchairs.'),
    (9, 'movable_object.barrier', 'Temporary road barrier placed in the scene in order to redirect traffic.'),
    (10, 'movable_object.debris', 'Movable object that is left on the driveable surface.'),
    (11, 'movable_object.pushable_pullable', 'Objects that a pedestrian may push or pull.'),
    (12, 'movable_object.trafficcone', 'All types of traffic cone.'),
    (13, 'static_object.bicycle_rack', 'Area or device intended to park or secure the bicycles in a row.'),
    (14, 'vehicle.bicycle', 'Human or electric powered 2-wheeled vehicle designed to travel at lower speeds either on '
                            'road surface, sidewalks or bike paths.'),
    (15, 'vehicle.bus.bendy', 'Bendy bus subcategory.'),
    (16, 'vehicle.bus.rigid', 'Rigid bus subcategory.'),
    (17, 'vehicle.car', 'Vehicle designed primarily for personal use, e.g. sedans, hatch-backs, wagons, vans, mini-vans, '
                        'SUVs and jeeps.'),
    (18, 'vehicle.construction', 'Vehicles primarily designed for construction.'),
    (19, 'vehicle.emergency.ambulance', 'All types of ambulances.'),
    (20, 'vehicle.emergency.police', 'All types of police vehicles.'),
    (21, 'vehicle.motorcycle', 'Gasoline or electric powered 2-wheeled vehicle designed to move rapidly (at the speed '
                               'of standard cars) on the road surface.'),
    (22, 'vehicle.trailer', 'Any vehicle trailer, both for trucks, cars and motorcycles.'),
    (23, 'vehicle.truck', 'Vehicles primarily designed to haul cargo.'),
    (24, 'flat.driveable_surface', 'All paved or unpaved surfaces that a car can drive on with no concern of traffic '
                                   'rules.'),
    (25, 'flat.other', 'All other forms of horizontal ground-level structures that do not belong to any of '
                       'driveable_surface, curb, sidewalk and terrain.'),
    (26, 'flat.sidewalk', 'Sidewalk, pedestrian walkways, bike paths, etc.'),
    (27, 'flat.terrain', 'Natural horizontal surfaces such as ground level horizontal vegetation (< 20 cm tall), '
                         'grass, rolling hills, soil, sand and gravel.'),
    (28, 'static.manmade', 'Includes man-made structures but not limited to: buildings, walls, guard rails, fences, '
                           'poles, drainages, hydrants, flags, banners, street signs, electric circuit boxes, traffic '
                           'lights, parking meters and stairs with less than 20 cm in height.'),
    (29, 'static.other', 'Points in the background that are not distinguishable, or objects that do not match any of '
                         'the above labels.'),
    (30, 'static.vegetation', 'Any vegetation in the frame that is higher than the ground, including bushes, plants, '
                              'potted plants, trees, etc.'),
    (31, 'vehicle.ego', 'The vehicle on which the cameras, radar and lidar are mounted, that is sometimes visible at the '
                        'bottom of the image.'),
)
NUSC_INDEX = {name: i for i, name, _ in NUSCENES_LIDARSEG}
NUSC_NAME = {i: name for i, name, _ in NUSCENES_LIDARSEG}
THING_INDICES = frozenset(range(1, 24))      # nuScenes-panoptic things: the categories with instances (not ego)

# (id = CARLA tag, name, CARLA's CityObjectLabel name, nuScenes-lidarseg category, CityScapes palette colour RGB, thing)
CLASSES = (
    (0, 'unlabelled', 'NONE', 'noise', (0, 0, 0), False),
    (1, 'road', 'Roads', 'flat.driveable_surface', (128, 64, 128), False),
    (2, 'sidewalk', 'Sidewalks', 'flat.sidewalk', (244, 35, 232), False),
    (3, 'building', 'Buildings', 'static.manmade', (70, 70, 70), False),
    (4, 'wall', 'Walls', 'static.manmade', (102, 102, 156), False),
    (5, 'fence', 'Fences', 'static.manmade', (190, 153, 153), False),
    (6, 'pole', 'Poles', 'static.manmade', (153, 153, 153), False),
    (7, 'traffic_light', 'TrafficLight', 'static.manmade', (250, 170, 30), True),
    (8, 'traffic_sign', 'TrafficSigns', 'static.manmade', (220, 220, 0), True),
    (9, 'vegetation', 'Vegetation', 'static.vegetation', (107, 142, 35), False),
    (10, 'terrain', 'Terrain', 'flat.terrain', (152, 251, 152), False),
    (11, 'sky', 'Sky', 'noise', (70, 130, 180), False),
    (12, 'pedestrian', 'Pedestrians', 'human.pedestrian.adult', (220, 20, 60), True),
    (13, 'rider', 'Rider', 'vehicle.bicycle', (255, 0, 0), True),
    (14, 'car', 'Car', 'vehicle.car', (0, 0, 142), True),
    (15, 'truck', 'Truck', 'vehicle.truck', (0, 0, 70), True),
    (16, 'bus', 'Bus', 'vehicle.bus.rigid', (0, 60, 100), True),
    (17, 'train', 'Train', 'static.other', (0, 80, 100), True),
    (18, 'motorcycle', 'Motorcycle', 'vehicle.motorcycle', (0, 0, 230), True),
    (19, 'bicycle', 'Bicycle', 'vehicle.bicycle', (119, 11, 32), True),
    (20, 'static', 'Static', 'static.manmade', (110, 190, 160), False),
    (21, 'dynamic', 'Dynamic', 'movable_object.pushable_pullable', (170, 120, 50), True),
    (22, 'other', 'Other', 'static.other', (55, 90, 80), False),
    (23, 'water', 'Water', 'flat.other', (45, 60, 150), False),
    (24, 'road_line', 'RoadLines', 'flat.driveable_surface', (157, 234, 50), False),
    (25, 'ground', 'Ground', 'flat.other', (81, 0, 81), False),
    (26, 'bridge', 'Bridge', 'static.manmade', (150, 100, 100), False),
    (27, 'rail_track', 'RailTrack', 'flat.other', (230, 150, 140), False),
    (28, 'guard_rail', 'GuardRail', 'static.manmade', (180, 165, 180), False),
    (EGO_CLASS, 'ego', '', 'vehicle.ego', (0, 255, 255), False),
)
CLASS_NAMES = {c[0]: c[1] for c in CLASSES}
CLASS_ID = {c[1]: c[0] for c in CLASSES}
CLASS_NOTES = {
    'rider': 'nuScenes counts a rider as part of the bicycle or motorcycle: a rider\'s points take its vehicle\'s '
             'category when the worker knows the actor, else vehicle.bicycle.',
    'train': 'nuScenes has no rail vehicle: static.other.',
    'sky': 'Never hit by a lidar; noise in nuScenes terms.',
    'ego': 'Not a CARLA tag: the pixels and points of the vehicle carrying the rig (told apart by its actor id; '
           'semantic images without the instance camera show it as car).',
}
# CARLA tag -> class id / nuScenes-lidarseg index (unknown tags: unlabelled / noise)
CLASS_OF_TAG = np.zeros(256, np.uint8)
NUSC_OF_TAG = np.zeros(256, np.uint8)
for _c in CLASSES:
    if _c[2]:
        CLASS_OF_TAG[_c[0]] = _c[0]
        NUSC_OF_TAG[_c[0]] = NUSC_INDEX[_c[3]]
THING_TAGS = frozenset(c[0] for c in CLASSES if c[5])
VEHICLE_TAGS = frozenset((13, 14, 15, 16, 17, 18, 19))
# what kind of level boxes (World.get_level_bbs) an unknown instance of a tag is looked for in (parked vehicles; the
# traffic lights' heads and the signs are boxes that claim their pixels: claim_parts)
LEVEL_LABEL_OF_TAG = {14: 'Car', 15: 'Truck', 16: 'Bus', 18: 'Motorcycle', 19: 'Bicycle', 13: 'Bicycle'}
# a traffic light's head box or a sign's box takes over the static pixels whose 3D point lies inside it (measured on
# 0.9.16, Town10HD, 45 views of the 62 heads: 89.5 % of the pixels inside a head box are tagged traffic_light, 9.5 %
# pole, 1 % sign; inside the 147 sign boxes 85.6 % sign, 11.2 % pole, 3.0 % traffic light)
PART_TAGS = frozenset((0, 6, 7, 8, 20, 22))
# the ground's meshes (road, its painted lines, sidewalk, terrain, …): one surface for the range image
GROUND_TAGS = frozenset((1, 2, 10, 23, 24, 25, 27))
GROUND_LUT = np.zeros(256, bool)
GROUND_LUT[list(GROUND_TAGS)] = True
PART_CLASS = {'TrafficLight': 7, 'TrafficSigns': 8}


def classes_doc() -> list:
    """The class table as index.json writes it."""
    out = []
    for cid, name, label, nus, col, thing in CLASSES:
        row = {'id': cid, 'name': name, 'carla_tags': [cid] if label else [], 'carla_label': label or None,
               'nuscenes': nus, 'nuscenes_index': NUSC_INDEX[nus], 'color': list(col), 'thing': thing}
        if name in CLASS_NOTES:
            row['note'] = CLASS_NOTES[name]
        out.append(row)
    return out


def palette() -> np.ndarray:
    """(256, 3) uint8 RGB colour per class id (unknown ids black)."""
    p = np.zeros((256, 3), np.uint8)
    for c in CLASSES:
        p[c[0]] = c[4]
    return p


def nuscenes_categories(token_fn, extra=()) -> list:
    """category.json rows for nuScenes-lidarseg: every one of the 32 categories with its 'index' (token_fn(name) gives
    its token), then any extra (name, description) a dataset uses that is not among them (index None is not allowed
    by the devkit: such names get the indices after 31)."""
    rows = [{'token': token_fn(n), 'name': n, 'description': d, 'index': i} for i, n, d in NUSCENES_LIDARSEG]
    k = len(rows)
    for n, d in extra:
        if n not in NUSC_INDEX:
            rows.append({'token': token_fn(n), 'name': n, 'description': d, 'index': k})
            k += 1
    return rows


# =========================================================================== actors: categories
def category_of(type_id: str, attrs: dict | None = None) -> str:
    """nuScenes category of a CARLA actor by its blueprint id and attributes (base_type, age)."""
    attrs = attrs or {}
    tid = str(type_id)
    base = str(attrs.get('base_type', '')).lower()
    if tid.startswith('walker.'):
        return 'human.pedestrian.child' if str(attrs.get('age', '')).lower() == 'child' else 'human.pedestrian.adult'
    if tid.startswith('static.prop.'):
        p = tid[len('static.prop.'):]
        if 'cone' in p:
            return 'movable_object.trafficcone'
        if 'barrier' in p or 'warning' in p:
            return 'movable_object.barrier'
        if any(k in p for k in ('sign', 'trafficlight')):
            return 'static.manmade'
        return 'movable_object.pushable_pullable'
    if tid.startswith('traffic.'):
        return 'static.manmade'
    if 'ambulance' in tid:
        return 'vehicle.emergency.ambulance'
    if 'police' in tid:
        return 'vehicle.emergency.police'
    if base == 'bicycle':
        return 'vehicle.bicycle'
    if base == 'motorcycle':
        return 'vehicle.motorcycle'
    if base == 'truck' or 'firetruck' in tid:
        return 'vehicle.truck'
    if base == 'bus' or 'fusorosa' in tid:
        return 'vehicle.bus.rigid'
    return 'vehicle.car'


# =========================================================================== CARLA images
def bgra(raw, width: int, height: int) -> np.ndarray:
    """A CARLA image's raw_data as (h, w, 4) uint8 (B, G, R, A)."""
    return np.frombuffer(raw, np.uint8).reshape(int(height), int(width), 4)


def decode_depth(img) -> np.ndarray:
    """Planar depth in metres (float32) of a sensor.camera.depth image (h, w, 4 BGRA): 1000 · (R + 256 G + 65536 B) /
    (2²⁴ − 1); 1000 m where nothing was hit (CARLA's far plane)."""
    a = np.asarray(img)
    n = a[..., 2].astype(np.float64) + a[..., 1].astype(np.float64) * 256.0 + a[..., 0].astype(np.float64) * 65536.0
    return (n * (1000.0 / 16777215.0)).astype(np.float32)


def decode_instance(img) -> tuple[np.ndarray, np.ndarray]:
    """(tag uint8, id uint16) of a sensor.camera.instance_segmentation image: R = the semantic tag, G + 256·B = the
    actor id (see the module doc)."""
    a = np.asarray(img)
    return a[..., 2].copy(), (a[..., 1].astype(np.uint16) | (a[..., 0].astype(np.uint16) << 8))


def packed(raw, width: int, height: int) -> np.ndarray:
    """A CARLA image's raw_data viewed (no copy) as (h, w) uint32: B | G << 8 | R << 16 | A << 24 (little-endian)."""
    return np.frombuffer(raw, '<u4').reshape(int(height), int(width))


def unpack_instance(x) -> tuple[np.ndarray, np.ndarray]:
    """(tag uint8, actor id uint16) of packed instance-camera pixels (see packed)."""
    x = np.asarray(x, np.uint32)
    return ((x >> 16) & 255).astype(np.uint8), (((x >> 8) & 255) | ((x & 255) << 8)).astype(np.uint16)


def unpack_depth(x) -> np.ndarray:
    """Planar depth (float32, metres) of packed depth-camera pixels: decode_depth's formula, R + 256 G + 65536 B by
    swapping the bytes (exact in float32: under 2²⁴)."""
    x = np.asarray(x, np.uint32)
    n = ((x & np.uint32(0xFF)) << np.uint32(16)) | (x & np.uint32(0xFF00)) | ((x >> np.uint32(16)) & np.uint32(0xFF))
    return n.astype(np.float32) * np.float32(1000.0 / 16777215.0)


def encode_depth(depth_m) -> np.ndarray:
    """The inverse of decode_depth (BGRA uint8), for synthetic images (the tests)."""
    n = np.clip(np.rint(np.asarray(depth_m, float) / 1000.0 * 16777215.0), 0, 16777215).astype(np.int64)
    out = np.empty(np.shape(depth_m) + (4,), np.uint8)
    out[..., 2] = n & 255
    out[..., 1] = (n >> 8) & 255
    out[..., 0] = (n >> 16) & 255
    out[..., 3] = 255
    return out


SEMANTIC_LIDAR = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'), ('idx', '<u4'), ('tag', '<u4')])


def semantic_lidar(raw) -> np.ndarray:
    """A SemanticLidarMeasurement's raw_data as a structured array (x, y, z, cos, idx, tag)."""
    return np.frombuffer(raw, SEMANTIC_LIDAR)


# =========================================================================== lens projection (core.lens's model)
def project(lens: dict, P) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optical-frame points or directions (K, 3) -> (u, v, theta, ok): u, v in the image's pixel-edge coordinates
    (pixel (i, j) covers [i, i+1) × [j, j+1)), theta the angle off the axis, ok where the lens model holds (theta
    within its field; a rectilinear lens: in front). lens: a plan's lens block (core.lens.Intrinsics.to_dict), or
    {'model': 'pinhole', 'K' (OpenCV coordinates), 'max_theta_deg'}."""
    P = np.asarray(P, float).reshape(-1, 3)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    max_t = math.radians(float(lens['max_theta_deg']))
    if lens.get('model') == 'pinhole':
        K = np.asarray(lens['K'], float)
        ok = (z > 1e-6) & (theta <= min(max_t, math.radians(89.0)))
        zz = np.where(ok, z, 1.0)
        return K[0, 0] * x / zz + K[0, 2] + 0.5, K[1, 1] * y / zz + K[1, 2] + 0.5, theta, ok
    r = r_of_theta(theta, float(lens['f_px']), lens['model'], lens.get('k', (0, 0, 0, 0)))
    s = np.where(rho > 1e-12, r / np.where(rho > 1e-12, rho, 1.0), 0.0)
    u = float(lens['cx']) + x * s
    v = float(lens['cy']) + y * s * float(lens.get('aspect', 1.0))
    ok = theta <= max_t
    if lens['model'] == 'rectilinear':
        ok &= z > 1e-6
    return u, v, theta, ok


def camera_pose(pose: dict) -> dict:
    """{'R': the CARLA rotation matrix (actor axes -> vehicle), 'pos': (3,)} of a pose dict {x y z roll pitch yaw}."""
    return {'R': carla_matrix(float(pose.get('pitch', 0.0)), float(pose.get('yaw', 0.0)), float(pose.get('roll', 0.0))),
            'pos': np.array([float(pose.get(k, 0.0)) for k in ('x', 'y', 'z')])}


def _o2v(cam_pose) -> np.ndarray:
    """Optical -> vehicle rotation of a camera pose."""
    return np.asarray(cam_pose['R'], float) @ CARLA_TO_OPTICAL.T


# =========================================================================== boxes
def box_matrix(yaw_deg: float = 0.0, R=None) -> np.ndarray:
    """A box's rotation: R as given, else a yaw about z (CARLA's yaw: the same matrix as carla_matrix(0, yaw, 0))."""
    if R is not None:
        return np.asarray(R, float)
    a = math.radians(float(yaw_deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_edges(center, size, R, n: int = 17) -> np.ndarray:
    """Points along a box's 12 edges (n per edge, corners included): (12 n, 3)."""
    L, W, H = (float(v) / 2.0 for v in size)
    corners = np.array([[sx * L, sy * W, sz * H] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count('1') == 1]
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.concatenate([corners[a] + t * (corners[b] - corners[a]) for a, b in pairs])
    return pts @ np.asarray(R, float).T + np.asarray(center, float)


def box_faces(center, size, R, n: int = 9) -> np.ndarray:
    """A grid of n × n points on each of a box's 6 faces (cell centres): (6 n², 3)."""
    half = np.asarray(size, float) / 2.0
    g = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    a, b = np.meshgrid(g, g)
    a, b = a.ravel(), b.ravel()
    out = []
    for ax in range(3):
        o1, o2 = [k for k in range(3) if k != ax]
        for sgn in (-1.0, 1.0):
            p = np.zeros((len(a), 3))
            p[:, ax] = sgn
            p[:, o1] = a
            p[:, o2] = b
            out.append(p * half)
    return np.concatenate(out) @ np.asarray(R, float).T + np.asarray(center, float)


def ray_box(o, D, center, size, R) -> tuple[np.ndarray, np.ndarray]:
    """Entry and exit distances (K,) of rays o + t·D (D unit, (K, 3)) through a box (inf where they miss; entry 0 when
    o is inside)."""
    Rb = np.asarray(R, float)
    oo = (np.asarray(o, float) - np.asarray(center, float)) @ Rb
    dd = np.asarray(D, float) @ Rb
    half = np.asarray(size, float) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-half - oo) / dd
        t2 = (half - oo) / dd
    lo = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
    hi = np.where(np.isnan(t2), np.inf, np.maximum(t1, t2))
    # a ray parallel to a slab: inside it always (nan above: 0/0 or ±inf), else never
    par = np.abs(dd) < 1e-12
    inside = np.abs(oo) <= half
    lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
    hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
    tin = np.max(lo, axis=-1)
    tout = np.min(hi, axis=-1)
    hit = (tout >= np.maximum(tin, 0.0))
    return np.where(hit, np.maximum(tin, 0.0), np.inf), np.where(hit, tout, np.inf)


def in_box(P, center, size, R, margin: float = 0.0) -> np.ndarray:
    q = (np.asarray(P, float) - np.asarray(center, float)) @ np.asarray(R, float)
    return np.all(np.abs(q) <= np.asarray(size, float) / 2.0 + margin, axis=-1)


# =========================================================================== camera ground truth
class CameraGT:
    """A camera's label images from its pinhole views (core.lens.plan_camera's views and remap tables): every output
    pixel reads the nearest view pixel (labels are never interpolated or feathered; where the cube faces are feathered
    the pixel reads the face with the higher weight), and knows its own ray (optical frame) and, per view pixel, the
    factor that turns the view's planar depth into the range along that ray."""

    def __init__(self, plan: dict):
        at = plan['atlas']
        self.views = plan['views']
        self.offsets = [int(o) for o in at['offsets']]
        self.aw, self.ah = int(at['width']), int(at['height'])
        rm = plan['remap']
        if rm is None:
            raise ValueError('the plan has no remap tables (plan_camera(..., tables=True))')
        mx = np.asarray(rm['map_x'], np.float64)
        my = np.asarray(rm['map_y'], np.float64)
        self.h, self.w = mx.shape
        sx, sy = mx.copy(), my.copy()
        b = rm.get('blend')
        self.blend_moved = 0
        if b is not None:
            idx = np.asarray(b['idx'], np.int64)
            bx = np.asarray(b['x'], np.float64)
            by = np.asarray(b['y'], np.float64)
            bw = np.asarray(b['w'], np.float64)
            j = np.argmax(bw, axis=1)
            sel = j > 0
            if sel.any():
                fx, fy = sx.reshape(-1), sy.reshape(-1)
                fx[idx[sel]] = bx[sel, j[sel] - 1]
                fy[idx[sel]] = by[sel, j[sel] - 1]
                self.blend_moved = int(sel.sum())
        ok = sx >= -0.5
        ix = np.clip(np.floor(sx + 0.5).astype(np.int64), 0, max(self.aw - 1, 0))
        iy = np.clip(np.floor(sy + 0.5).astype(np.int64), 0, max(self.ah - 1, 0))
        self.ok = ok
        idt = np.int32 if self.aw * self.ah < 2 ** 31 - 1 else np.int64      # (a full-size camera: tens of MB less)
        self.src = np.where(ok, iy * self.aw + ix, -1).ravel().astype(idt)
        self.sel = np.flatnonzero(self.src >= 0).astype(idt)
        self.src_sel = self.src[self.sel]
        # each pixel's view: by its atlas row
        k = np.searchsorted(np.asarray(self.offsets), iy, side='right') - 1
        k = np.clip(k, 0, max(len(self.views) - 1, 0))
        rays = np.zeros((self.h, self.w, 3))
        scale = np.zeros((self.h, self.w))
        for vi, v in enumerate(self.views):
            m = ok & (k == vi)
            if not m.any():
                continue
            f = float(v['f_px'])
            a = (sx[m] - (v['width'] / 2.0 - 0.5)) / f
            bb = (sy[m] - self.offsets[vi] - (v['height'] / 2.0 - 0.5)) / f
            dv = np.stack([a, bb, np.ones_like(a)], -1)
            n = np.linalg.norm(dv, axis=1)
            rays[m] = (dv / n[:, None]) @ np.asarray(v['optical'], float).T
            # range = planar depth of the (nearest) view pixel × |(a, b, 1)| of this pixel's own ray
            scale[m] = n
        self.rays = rays.astype(np.float32)
        self.range_scale = scale.astype(np.float32).ravel()
        # the four view pixels around each output pixel's exact position (their centres on integers) and its bilinear
        # weights there, for the range: inside one view, else the nearest pixel only
        fx, fy = sx.ravel()[self.sel], sy.ravel()[self.sel]
        x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        kk = k.ravel()[self.sel]
        offs = np.asarray(self.offsets + [self.ah], np.int64)
        widths = np.array([int(v['width']) for v in self.views] or [0], np.int64)
        inside = (x0 >= 0) & (x0 + 1 < widths[kk]) & (y0 >= offs[kk]) & (y0 + 1 < offs[kk + 1])
        base = np.where(inside, y0 * self.aw + x0, self.src_sel)
        self.nb4 = np.stack([base, base + inside, base + self.aw * inside, base + (self.aw + 1) * inside]).astype(idt)
        ax, ay = np.where(inside, fx - x0, 0.0), np.where(inside, fy - y0, 0.0)
        self.w4 = np.stack([(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay]).astype(np.float32)

    def gather(self, atlas: np.ndarray, fill=0) -> np.ndarray:
        """(h, w) of the atlas (ah, aw) value each output pixel reads (fill where it sees nothing)."""
        a = np.asarray(atlas).reshape(-1)
        out = np.full(self.h * self.w, fill, a.dtype)
        out[self.sel] = a[self.src_sel]
        return out.reshape(self.h, self.w)

    def range_of(self, depth_out: np.ndarray, far_m: float = 999.0) -> np.ndarray:
        """Range (float32, metres) along each output pixel's ray from the planar depth its view pixel has (depth_out:
        (h, w), already read through gather); 0 where nothing was hit or the pixel sees nothing."""
        z = np.asarray(depth_out, np.float32).reshape(-1)
        out = np.zeros(self.h * self.w, np.float32)
        zs = z[self.sel]
        out[self.sel] = np.where(zs < far_m, zs * self.range_scale[self.sel], 0.0)
        return out.reshape(self.h, self.w)

    def range_interp(self, depth_atlas: np.ndarray, seg_atlas=None, far_m: float = 999.0,
                     spread: float = 0.05) -> np.ndarray:
        """Range (float32, metres) along each output pixel's ray, its planar depth read at the ray's exact position:
        the inverse depth of the four view pixels around it interpolated bilinearly (exact on a plane: a pinhole view's
        1/z is linear in its pixel coordinates) when all four are one surface (the same instance and tag in seg_atlas,
        the instance camera's packed pixels; without it, depths within `spread` of each other), else the nearest view
        pixel's (never a mix across an object's edge). The ground's meshes (GROUND_TAGS: road, its lines, sidewalk,
        …) count as one surface: four samples cannot tell a curb's step from the ground's slope, and at a step the
        nearest pixel errs as much. 0 where nothing was hit or the pixel sees nothing."""
        d = np.asarray(depth_atlas, np.float32).reshape(-1)
        return self._range(d[self.nb4], d[self.src_sel], seg_atlas, far_m, spread)

    def range_from_packed(self, depth32, seg_atlas=None, far_m: float = 999.0, spread: float = 0.05) -> np.ndarray:
        """range_interp from the depth camera's packed views (uint32 atlas, see packed): decoded at the pixels read
        only."""
        p = np.asarray(depth32, np.uint32).reshape(-1)
        return self._range(unpack_depth(p[self.nb4]), unpack_depth(p[self.src_sel]), seg_atlas, far_m, spread)

    def _range(self, z4, zn, seg_atlas, far_m, spread):
        z4 = np.asarray(z4, np.float32)
        zn = np.asarray(zn, np.float32)
        if seg_atlas is not None:
            g = np.asarray(seg_atlas).reshape(-1)
            g4 = g[self.nb4]
            same = np.all(g4 == g[self.src_sel][None, :], axis=0)
            same |= np.all(GROUND_LUT[(g4.astype(np.uint32) >> np.uint32(16)) & np.uint32(255)], axis=0)
        else:
            same = z4.max(0) <= z4.min(0) * np.float32(1.0 + spread)
        ok = same & np.all(z4 < far_m, axis=0) & np.all(z4 > 0, axis=0)
        zi = 1.0 / np.sum(self.w4 / np.where(z4 > 0, z4, np.float32(1.0)), axis=0)
        z = np.where(ok, zi, zn)
        out = np.zeros(self.h * self.w, np.float32)
        out[self.sel] = np.where(z < far_m, z * self.range_scale[self.sel], 0.0)
        return out.reshape(self.h, self.w)

    def range_m(self, depth_atlas: np.ndarray, far_m: float = 999.0) -> np.ndarray:
        """Range (float32, metres) along each output pixel's ray from the views' planar depth (0: nothing hit, or the
        pixel sees nothing)."""
        d = np.asarray(depth_atlas, np.float32).reshape(-1)
        out = np.zeros(self.h * self.w, np.float32)
        z = d[self.src_sel]
        out[self.sel] = np.where(z < far_m, z * self.range_scale[self.sel], 0.0)
        return out.reshape(self.h, self.w)


def range_png(range_m: np.ndarray) -> np.ndarray:
    """uint16 range image: round(range · 256), 0 where nothing is within RANGE_MAX_M."""
    r = np.asarray(range_m, np.float64)
    return np.where((r > 0) & (r < RANGE_MAX_M), np.rint(r * RANGE_SCALE), 0).astype(np.uint16)


def claim_parts(sem: np.ndarray, rng, rays, cam_pose: dict, lens: dict, parts: list, margin: float = 0.03) -> dict:
    """{part key: flat pixel indices}: the pixels whose 3D point (the range along the pixel's ray) lies inside a part's
    box (a traffic light's head, a sign; vehicle frame), static tags only (PART_TAGS: a vehicle or walker passing in
    front keeps its pixels); the nearest box first, each pixel to one box. Only the pixels inside the box's projection
    are tested."""
    out = {}
    if rng is None or not parts:
        return out
    h, w = sem.shape
    pos = np.asarray(cam_pose['pos'], float)
    O2V = _o2v(cam_pose)
    V2O = O2V.T
    claimed = np.zeros(h * w, bool)
    static = np.isin(sem, list(PART_TAGS)).ravel()
    r_all = np.asarray(rng, np.float32).ravel()
    rays_flat = np.asarray(rays).reshape(-1, 3)
    order = sorted(parts, key=lambda o: float(np.linalg.norm(np.asarray(o['center'], float) - pos)))
    for o in order:
        E = (box_edges(o['center'], o['size'], o['R'], 5) - pos) @ V2O.T
        u, v, _, ok = project(lens, E)
        if not ok.any():
            continue
        x0, x1 = max(int(math.floor(u[ok].min())) - 1, 0), min(int(math.ceil(u[ok].max())) + 1, w)
        y0, y1 = max(int(math.floor(v[ok].min())) - 1, 0), min(int(math.ceil(v[ok].max())) + 1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pix = (yy * w + xx).ravel()
        pix = pix[static[pix] & ~claimed[pix] & (r_all[pix] > 0)]
        if not len(pix):
            continue
        P = (rays_flat[pix] * r_all[pix, None]) @ O2V.T + pos
        pix = pix[in_box(P, o['center'], o['size'], o['R'], margin)]
        if len(pix):
            claimed[pix] = True
            out[o['key']] = pix
    return out


def objects_in_frame(sem: np.ndarray, inst_raw: np.ndarray, rng, rays, cam_pose: dict, lens: dict, known: dict,
                     level_boxes: list, ego_id: int, assoc: dict, min_level_share: float = 0.5, parts=(),
                     near_px: int = 3) -> tuple:
    """The objects a camera frame sees and its instance image.

    sem (h, w) uint8 tags, inst_raw (h, w) uint16 ids as CARLA wrote them, rng (h, w) range in metres (or None), rays
    (h, w, 3) optical rays (CameraGT.rays), cam_pose {'R': 3×3 CARLA rotation (actor -> vehicle), 'pos': (3,)} (the
    camera's actual mount, camera_pose()), lens: the plan's lens block; known: {raw id: object} for the actors the
    worker knows, level_boxes: [object] of the map's level boxes ('level_label'), each object {'key', 'center', 'size',
    'R' (vehicle frame at the frame's time), 'actor_id', 'category', ...}; assoc: a cache raw id -> object key (or
    None) kept by the caller across frames (level meshes keep their ids); parts: the traffic lights' heads and the
    signs (objects with 'part_class'), each claiming the pixels inside its box (claim_parts): they take its class and
    make it an object of their own, whatever their ids.
    Returns ([{'obj', 'pix' (flat pixel indices), 'bbox' (tight, pixel edges), 'pixels', 'class' (the majority class of
    its pixels, a rider counted as its two-wheeler), 'raw_ids'}], the semantic image with the ego's pixels as class
    'ego' and the parts' pixels as theirs)."""
    h, w = sem.shape
    sem = sem.copy()
    ego_px = inst_raw == ego_id if ego_id is not None else np.zeros_like(sem, bool)
    ego_px &= np.isin(sem, list(VEHICLE_TAGS))
    sem[ego_px] = EGO_CLASS
    claims = claim_parts(sem, rng, rays, cam_pose, lens, list(parts))
    part_of = {o['key']: o for o in parts}
    # a light's own pixels of its class just outside its heads' boxes (a visor's edge): to the head of that light
    # whose claimed pixels' box is nearest, within near_px
    by_actor = {}
    for key, pix in claims.items():
        if part_of[key].get('actor_id') is not None:
            by_actor.setdefault(int(part_of[key]['actor_id']), []).append(key)
    if by_actor:
        taken = np.zeros(h * w, bool)
        for pix in claims.values():
            taken[pix] = True
        flat_ids, flat_sem = inst_raw.ravel(), sem.ravel()
        for aid, keys in by_actor.items():
            extra = np.flatnonzero((flat_ids == aid) & ~taken & (flat_sem == part_of[keys[0]]['part_class']))
            if not len(extra):
                continue
            ys, xs = extra // w, extra % w
            dist = []
            for k in keys:
                py, px = claims[k] // w, claims[k] % w
                dx = np.maximum(np.maximum(px.min() - xs, xs - px.max()), 0)
                dy = np.maximum(np.maximum(py.min() - ys, ys - py.max()), 0)
                dist.append(np.maximum(dx, dy))
            dist = np.stack(dist)
            best = np.argmin(dist, 0)
            ok = dist[best, np.arange(len(extra))] <= near_px
            for j, k in enumerate(keys):
                add = extra[ok & (best == j)]
                if len(add):
                    claims[k] = np.concatenate([claims[k], add])
    claimed = np.zeros(h * w, bool)
    raw_tags = {}
    for key, pix in claims.items():
        raw_tags[key] = np.bincount(sem.ravel()[pix], minlength=EGO_CLASS + 1)
        sem.ravel()[pix] = part_of[key]['part_class']
        claimed[pix] = True
    flat_ids = inst_raw.ravel()
    order = np.argsort(flat_ids, kind='stable')         # the pixels grouped by id in one sort
    ids, starts, counts = np.unique(flat_ids[order], return_index=True, return_counts=True)
    thing = np.isin(sem.ravel(), list(THING_TAGS))
    by_key = {o['key']: o for o in level_boxes}
    by_key.update({o['key']: o for o in known.values()})
    groups = {}
    O2V = _o2v(cam_pose)
    pos = np.asarray(cam_pose['pos'], float)
    for rid, s0, n in zip(ids.tolist(), starts.tolist(), counts.tolist()):
        if rid == ego_id:
            continue
        allpix = order[s0:s0 + n]
        if claims:
            allpix = allpix[~claimed[allpix]]
            if not len(allpix):
                continue
        obj = known.get(rid)
        if obj is not None:
            pix = allpix                                # an actor the worker knows: all its pixels (a prop sign too)
        else:
            if not level_boxes:
                continue
            pix = allpix[thing[allpix]]
            if not len(pix):
                continue
        if obj is None:
            if rid not in assoc and rng is not None:
                tags = sem.ravel()[pix]
                tag = int(np.bincount(tags, minlength=256).argmax())
                label = LEVEL_LABEL_OF_TAG.get(tag)
                r = np.asarray(rng).ravel()[pix]
                good = r > 0
                cand = [o for o in level_boxes if o.get('level_label') == label] if label else []
                if cand and good.sum() >= 3:
                    P = (np.asarray(rays).reshape(-1, 3)[pix[good]] * r[good, None]) @ O2V.T + pos
                    hits = [int(in_box(P, o['center'], o['size'], o['R'], 0.25).sum()) for o in cand]
                    best = int(np.argmax(hits))
                    assoc[rid] = cand[best]['key'] if hits[best] >= min_level_share * len(P) else None
                else:
                    assoc[rid] = None
            key = assoc.get(rid)
            obj = by_key.get(key) if key is not None else None
        if obj is None:
            continue
        g = groups.setdefault(obj['key'], (obj, [], []))
        g[1].append(pix)
        g[2].append(int(rid))
    for key, pix in claims.items():
        groups[key] = (part_of[key], [pix], sorted(set(np.unique(inst_raw.ravel()[pix]).tolist())))
    out = []
    flat_sem = sem.ravel()
    for key, (obj, parts_, rids) in groups.items():
        pix = np.concatenate(parts_)
        ys, xs = pix // w, pix % w
        tags = np.bincount(flat_sem[pix], minlength=256)
        if tags[13] and (tags[18] or tags[19]):        # a rider counts as its two-wheeler
            tags[18 if tags[18] >= tags[19] else 19] += tags[13]
            tags[13] = 0
        out.append({'obj': obj, 'pix': pix, 'bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    'pixels': int(len(pix)), 'class': int(tags.argmax()), 'raw_ids': sorted(rids),
                    'raw_tags': raw_tags.get(key)})
    return out, sem


def object_geometry(obj: dict, cam_pose: dict, lens: dict, w: int, h: int, rays=None, rng=None, visible=None,
                    occluder_tol_m: float = 0.3, pix=None) -> dict:
    """bbox_full (the box's edges through the lens, unclipped; None when none of it is within the lens's field),
    truncated (the share of the box's surface points outside the image or the lens's field), occluded (of the pixels
    whose ray meets the box, the share hidden by something nearer than the box: those over the object's visible
    pixels plus them; None without a range image), range_m (camera to box centre). The visible pixels: a mask
    (visible, h × w) or their flat indices (pix)."""
    pos = np.asarray(cam_pose['pos'], float)
    O2V = _o2v(cam_pose)
    V2O = O2V.T
    c, size, Rb = np.asarray(obj['center'], float), np.asarray(obj['size'], float), np.asarray(obj['R'], float)
    E = (box_edges(c, size, Rb) - pos) @ V2O.T
    u, v, _, ok = project(lens, E)
    full = None
    if ok.any():
        full = [float(u[ok].min()), float(v[ok].min()), float(u[ok].max()), float(v[ok].max())]
    F = (box_faces(c, size, Rb) - pos) @ V2O.T
    fu, fv, _, fok = project(lens, F)
    inside = fok & (fu >= 0) & (fu < w) & (fv >= 0) & (fv < h)
    truncated = float(1.0 - inside.mean())
    occluded = None
    if rng is not None and rays is not None and (visible is not None or pix is not None):
        if full is not None:
            x0, y0 = max(int(math.floor(full[0])) - 1, 0), max(int(math.floor(full[1])) - 1, 0)
            x1, y1 = min(int(math.ceil(full[2])) + 1, w), min(int(math.ceil(full[3])) + 1, h)
        else:
            x0, y0, x1, y1 = 0, 0, 0, 0
        n_vis = int(np.count_nonzero(visible)) if pix is None else int(len(pix))
        n_occ = 0
        if x1 > x0 and y1 > y0:
            D = np.asarray(rays)[y0:y1, x0:x1].reshape(-1, 3) @ O2V.T
            tin, _ = ray_box(pos, D, c, size, Rb)
            r = np.asarray(rng)[y0:y1, x0:x1].reshape(-1)
            if pix is None:
                vis = np.asarray(visible)[y0:y1, x0:x1].reshape(-1)
            else:
                win = np.zeros((y1 - y0, x1 - x0), bool)
                py, px = np.asarray(pix) // w, np.asarray(pix) % w
                m = (py >= y0) & (py < y1) & (px >= x0) & (px < x1)
                win[py[m] - y0, px[m] - x0] = True
                vis = win.reshape(-1)
            hit = np.isfinite(tin)
            n_occ = int(np.count_nonzero(hit & ~vis & (r > 0) & (r < tin - occluder_tol_m)))
        occluded = float(n_occ / (n_vis + n_occ)) if n_vis + n_occ else 1.0
    return {'bbox_full': [round(x, 2) for x in full] if full is not None else None, 'truncated': round(truncated, 4),
            'occluded': round(occluded, 4) if occluded is not None else None,
            'range_m': round(float(np.linalg.norm(c - pos)), 3)}


# =========================================================================== COCO run-length encoding
def rle_encode(mask: np.ndarray) -> dict:
    """COCO's compressed RLE of a binary mask (h, w): {'size': [h, w], 'counts': str} (column-major runs starting
    with the zeros, LEB128-like characters as pycocotools writes them)."""
    m = np.asarray(mask, bool)
    h, w = m.shape
    flat = m.ravel(order='F')
    if flat.size:
        change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        counts = np.diff(np.r_[0, change, flat.size]).tolist()
        if flat[0]:
            counts = [0] + counts                # the runs start with the zeros
    else:
        counts = []
    return _rle_string(counts, h, w)


def rle_of_pixels(pix, h: int, w: int) -> dict:
    """rle_encode of the mask holding the flat (row-major) pixel indices pix, without making the mask: its runs from the
    sorted column-major indices (as long as the object, not the image)."""
    pix = np.asarray(pix, np.int64)
    if not len(pix):
        return _rle_string([h * w], h, w)
    col = np.sort((pix % w) * h + pix // w)
    brk = np.flatnonzero(np.diff(col) != 1) + 1
    starts = col[np.r_[0, brk]]
    ends = col[np.r_[brk - 1, len(col) - 1]] + 1
    counts = []
    pos = 0
    for a, b in zip(starts.tolist(), ends.tolist()):
        counts += [a - pos, b - a]
        pos = b
    if pos < h * w:
        counts.append(h * w - pos)
    return _rle_string(counts, h, w)


def _rle_string(counts, h, w) -> dict:
    s = []
    for i, x in enumerate(counts):
        x = int(x)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            c = x & 0x1f
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            s.append(chr(c + 48))
    return {'size': [int(h), int(w)], 'counts': ''.join(s)}


def rle_counts(rle: dict) -> list:
    s = rle['counts']
    if isinstance(s, list):
        return [int(v) for v in s]
    counts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle['size']
    flat = np.zeros(h * w, bool)
    pos, val = 0, False
    for n in rle_counts(rle):
        if val:
            flat[pos:pos + n] = True
        pos += n
        val = not val
    return flat.reshape((h, w), order='F')


# =========================================================================== reading a drive's ground truth
def read(out_dir, index: str = INDEX) -> dict | None:
    """The ground-truth index of a drive (out_dir/index, falcon.ground_truth/1) with '_root' set to out_dir (every
    path in it is relative to out_dir), or None."""
    path = os.path.join(str(out_dir), index)
    try:
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
        return None
    doc['_root'] = os.path.abspath(str(out_dir))
    return doc


def path_of(doc: dict, rel: str) -> str:
    return os.path.join(doc.get('_root', ''), rel)


def _imread(path, flags):
    import cv2
    img = cv2.imread(path, flags)
    if img is None:
        raise OSError(f'cannot read {path}')
    return img


def load_semantic(doc: dict, frame: dict) -> np.ndarray:
    """(h, w) uint8 class ids of a camera frame."""
    import cv2
    return _imread(path_of(doc, frame['semantic']), cv2.IMREAD_UNCHANGED)


def load_instance(doc: dict, frame: dict) -> np.ndarray:
    """(h, w) uint16 object ids (frame['objects'][k]['id']; 0 = none)."""
    import cv2
    return _imread(path_of(doc, frame['instance']), cv2.IMREAD_UNCHANGED)


def load_range(doc: dict, frame: dict) -> np.ndarray:
    """(h, w) float32 range in metres along each pixel's ray (0 = nothing within RANGE_MAX_M)."""
    import cv2
    return _imread(path_of(doc, frame['range']), cv2.IMREAD_UNCHANGED).astype(np.float32) / RANGE_SCALE


def load_lidarseg(doc: dict, sweep: dict) -> np.ndarray:
    """uint8 nuScenes-lidarseg category index per point of a lidar sweep (in its .pcd.bin's order)."""
    return np.fromfile(path_of(doc, sweep['lidarseg']), np.uint8)


def load_panoptic(doc: dict, sweep: dict) -> np.ndarray:
    """uint16 per point: category index · 1000 + instance id (0 for stuff)."""
    with np.load(path_of(doc, sweep['panoptic'])) as z:
        return z['data']


def frames(doc: dict, tag: str) -> list:
    return (doc.get('cameras', {}).get(tag) or {}).get('frames', [])


def summary(doc: dict) -> dict:
    """Counts for the assistant and logs."""
    cams = doc.get('cameras', {})
    objs = [o for c in cams.values() for f in c.get('frames', []) for o in f.get('objects', [])]
    by_class = {}
    for o in objs:
        by_class[o['class']] = by_class.get(o['class'], 0) + 1
    return {'cameras': {t: len(c.get('frames', [])) for t, c in cams.items()},
            'lidars': {t: len(L.get('sweeps', [])) for t, L in doc.get('lidars', {}).items()},
            'objects': len(objs), 'objects_by_class': by_class,
            'actors': len({o['actor_id'] for o in objs if o.get('actor_id') is not None})}


# =========================================================================== viewing a drive's ground truth
# (numpy; what a ground-truth viewer shows: the label images in colour, a lidar sweep from above)
_PALETTE = palette()


def load_image(doc: dict, frame: dict) -> np.ndarray:
    """(h, w, 3) uint8 RGB: the camera image a frame's labels belong to (the same size, pixel for pixel)."""
    import cv2
    return np.ascontiguousarray(_imread(path_of(doc, frame['image']), cv2.IMREAD_COLOR)[:, :, ::-1])


def colour_semantic(sem: np.ndarray) -> np.ndarray:
    """(h, w, 3) uint8 RGB: every pixel in its class's colour (the classes table: CARLA's CityScapes palette, the ego
    cyan, an unknown id black)."""
    return _PALETTE[np.asarray(sem, np.uint8)]


def object_colours(ids) -> np.ndarray:
    """(n, 3) uint8 RGB, one colour per object id, the same for an id in every frame of a drive: the hue steps by the
    golden ratio of a turn with the id, the brightness alternates, so neighbouring ids differ; id 0 (no object) is
    black."""
    ids = np.asarray(ids, np.int64).ravel()
    h = (ids * 0.6180339887498949 + 0.11) % 1.0 * 6.0
    s = np.where(ids % 3 == 0, 0.55, 0.75)
    v = np.where(ids % 2 == 0, 0.98, 0.78)
    k = np.floor(h).astype(np.int64) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    rgb = np.select([k[:, None] == j for j in range(6)],
                    [np.stack(c, -1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))])
    out = np.round(rgb * 255).astype(np.uint8)
    out[ids == 0] = 0
    return out


def colour_instance(inst: np.ndarray) -> np.ndarray:
    """(h, w, 3) uint8 RGB: every object's pixels in its own colour (object_colours), pixels of no object black."""
    inst = np.asarray(inst)
    u, inv = np.unique(inst, return_inverse=True)
    return object_colours(u)[inv.reshape(inst.shape)]


def classes_in(sem: np.ndarray) -> list:
    """The classes a semantic image holds, most pixels first: [{'id', 'name', 'color' (RGB), 'pixels', 'share'}]."""
    n = np.bincount(np.asarray(sem, np.uint8).ravel(), minlength=256)
    tot = max(int(n.sum()), 1)
    return [{'id': int(i), 'name': CLASS_NAMES.get(int(i), f'class {i}'), 'color': tuple(int(x) for x in _PALETTE[i]),
             'pixels': int(n[i]), 'share': n[i] / tot} for i in np.argsort(-n, kind='stable') if n[i]]


def range_limits(rng: np.ndarray) -> tuple[float, float]:
    """A log range scale for a range image: from its 1st to its 99th percentile of the pixels with a range, widened to
    round values of the 1-2-5 series (at least 0.5 m to 20 m apart); (1, 100) when nothing has a range."""
    v = np.asarray(rng, np.float32)
    v = v[v > 0]
    if not v.size:
        return 1.0, 100.0
    lo, hi = (float(x) for x in np.percentile(v, [1.0, 99.0]))
    steps = [m * 10.0 ** e for e in range(-1, 4) for m in (1.0, 2.0, 5.0)]
    lo = max([s for s in steps if s <= max(lo, 0.5)] or [0.5])
    hi = min([s for s in steps if s >= max(hi, lo * 20.0)] or [500.0])
    return lo, hi


def range_position(rng: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """Each pixel's place on a log range scale: 1 at near_m and nearer, 0 at far_m and beyond (near is the stronger
    end); NaN where the range image holds nothing (0: no surface within RANGE_MAX_M)."""
    r = np.asarray(rng, np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (math.log(far_m) - np.log(np.maximum(r, 1e-6))) / (math.log(far_m) - math.log(near_m))
    return np.where(r > 0, np.clip(t, 0.0, 1.0), np.nan)


def scale_ticks(lo: float, hi: float) -> list:
    """Round values of the 1-2-5 series from lo to hi (both included) for a log scale's labels."""
    out = [lo]
    for e in range(-2, 5):
        for m in (1.0, 2.0, 5.0):
            x = m * 10.0 ** e
            if lo * 1.15 < x < hi / 1.15:
                out.append(x)
    return out + [hi]


LIDARSEG_COLOUR_CLASS = {'static.other': 'other', 'vehicle.bicycle': 'bicycle', 'flat.other': 'ground'}
NOISE_RGB = (128, 128, 128)            # nuScenes 'noise' points (unlabelled): grey, not the semantic images' black


def lidarseg_palette() -> np.ndarray:
    """(32, 3) uint8 RGB per nuScenes-lidarseg category index: the colour of the class of the classes table mapped to
    it (the first in the table, or the one the category is named after: LIDARSEG_COLOUR_CLASS), so a lidar point has
    the colour its surface has in the semantic images; a category no CARLA class maps to takes its family's (a child
    or a worker the pedestrian's, an ambulance or a trailer the truck's, a cone the dynamic class's); noise grey."""
    out = np.zeros((len(NUSCENES_LIDARSEG), 3), np.uint8)
    have = set()
    for c in CLASSES:
        i = NUSC_INDEX[c[3]]
        pref = LIDARSEG_COLOUR_CLASS.get(c[3])
        if i not in have and (pref is None or pref == c[1]):
            out[i] = c[4]
            have.add(i)
    out[NUSC_INDEX['noise']] = NOISE_RGB
    have.add(NUSC_INDEX['noise'])
    family = (('human.', 'pedestrian'), ('animal', 'other'), ('movable_object.', 'dynamic'),
              ('static_object.', 'static'), ('vehicle.bus', 'bus'), ('vehicle.', 'truck'), ('flat.', 'ground'),
              ('static.', 'static'))
    for i, name, _ in NUSCENES_LIDARSEG:
        if i in have:
            continue
        cls = next((c for pre, c in family if name.startswith(pre)), 'other')
        out[i] = CLASSES[CLASS_ID[cls]][4]
    return out


def sweep_near(doc: dict, tag: str, timestamp: int) -> dict | None:
    """The labelled sweep of lidar tag whose stamp is nearest timestamp (µs), or None."""
    sw = [s for s in (doc.get('lidars', {}).get(tag) or {}).get('sweeps', []) if s.get('lidarseg')]
    return min(sw, key=lambda s: abs(int(s['timestamp']) - int(timestamp))) if sw else None


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def lidar_points(doc: dict, sweep: dict, version: str | None = None) -> np.ndarray:
    """(n, 3) float32: a labelled sweep's points (its nuScenes .pcd.bin, in the order of its lidarseg labels) in CARLA's
    vehicle frame: through the sweep's calibrated_sensor (the mount the calibration states, in nuScenes' right-handed
    ego axes) and then mirrored to CARLA's axes (y → −y). The nuScenes version folder is `version`, else the one in
    the sweep's lidarseg path (nuscenes/lidarseg/<version>/…), else NUSCENES_VERSION. The tables are read once per
    index and kept on it."""
    parts = sweep['lidarseg'].replace('\\', '/').split('/')
    if version is None:
        version = parts[2] if len(parts) > 3 and parts[0] == 'nuscenes' else NUSCENES_VERSION
    nus = doc.get('_nuscenes')
    if nus is None or nus[0] != version:
        tables = os.path.join(doc.get('_root', ''), 'nuscenes', version)
        with open(os.path.join(tables, 'sample_data.json'), encoding='utf-8') as fh:
            sd = {r['token']: (r['filename'], r['calibrated_sensor_token']) for r in json.load(fh)}
        with open(os.path.join(tables, 'calibrated_sensor.json'), encoding='utf-8') as fh:
            cs = {r['token']: r for r in json.load(fh)}
        nus = doc['_nuscenes'] = (version, sd, cs)
    filename, cs_token = nus[1][sweep['sample_data_token']]
    a = np.fromfile(os.path.join(doc.get('_root', ''), 'nuscenes', filename), np.float32)
    if a.size % 5:
        raise ValueError(f'{filename}: not a nuScenes .pcd.bin (x, y, z, intensity, ring)')
    a = a.reshape(-1, 5)
    c = nus[2][cs_token]
    R = _quat_matrix(c['rotation'])
    P = a[:, :3] @ R.T.astype(np.float32) + np.asarray(c['translation'], np.float32)
    return (P * np.array([1.0, -1.0, 1.0], np.float32)).astype(np.float32)


def box_footprint(box3d: dict) -> np.ndarray:
    """(4, 2) float: a 3D box's corners seen from above (x, y in CARLA's vehicle frame; yaw_deg CARLA's), front right
    first (CARLA's y is to the right), then rear right, rear left, front left."""
    c = np.asarray(box3d['center'], float)
    L, W = float(box3d['size'][0]), float(box3d['size'][1])
    a = math.radians(float(box3d.get('yaw_deg', 0.0)))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    loc = np.array([[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]])
    return loc @ R.T + c[:2]


def topdown(P: np.ndarray, colours: np.ndarray, half_m: float = 40.0, px: int = 640, dot: int = 1) -> np.ndarray:
    """(px, px, 4) uint8 RGBA: points (CARLA's vehicle frame) seen from above, x forward up, y (right) to the right,
    the vehicle's origin at the centre, half_m to each edge; each point a dot × dot square in its colour (its pixel at
    the top left), a higher point over a lower one; alpha 0 where no point falls."""
    P = np.asarray(P, np.float32)
    out = np.zeros((px, px, 4), np.uint8)
    if not len(P):
        return out
    s = px / (2.0 * half_m)
    col0 = np.floor(px / 2.0 + P[:, 1] * s).astype(np.int64)
    row0 = np.floor(px / 2.0 - P[:, 0] * s).astype(np.int64)
    rgb0 = np.asarray(colours)
    lins, zs, rgbs = [], [], []
    for dr in range(max(int(dot), 1)):
        for dc in range(max(int(dot), 1)):
            row, col = row0 + dr, col0 + dc
            keep = (col >= 0) & (col < px) & (row >= 0) & (row < px)
            lins.append(row[keep] * px + col[keep])
            zs.append(P[keep, 2])
            rgbs.append(rgb0[keep])
    lin, z, rgb = np.concatenate(lins), np.concatenate(zs), np.concatenate(rgbs)
    order = np.argsort(-z, kind='stable')                  # highest first: np.unique keeps each pixel's first
    lin, first = np.unique(lin[order], return_index=True)
    out.reshape(-1, 4)[lin, :3] = rgb[order][first]
    out.reshape(-1, 4)[lin, 3] = 255
    return out
