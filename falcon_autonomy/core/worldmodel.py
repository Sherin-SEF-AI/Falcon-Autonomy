"""World-model data derived from a recorded run (falcon-raw): occupancy, BEV maps, scene flow, future trajectories,
ego actions and multi-camera clips, all aligned to the run's steps (schema falcon.worldmodel/1). numpy only.

    derive(run_dir, out_dir, options) -> manifest        everything below, with OUT/worldmodel.json

Inputs (docs/dataset-format.md): frames.jsonl (the ego's pose and speed per step, lidar sweeps with `lidar_pose`, the
camera files, `control` on live recordings), actors.jsonl (every vehicle and walker per step: pose, velocity, box,
nuScenes category, role; the ego is role 'hero'), lidar/<id>/<n>.bin + .labels.bin, lanes.json, dataset.json (fps).
Every product is computed from the recorded state only; nothing is simulated again and no image is interpolated.

Frames and units. CARLA's throughout: x forward, y right, z up — LEFT-handed — metres, seconds, degrees (core.rig).
"The ego frame at step i" is the ego's CARLA vehicle frame at that step (origin: CARLA's vehicle origin, the box's
centre at ground height; the full pose from frames.jsonl, roll and pitch included). A yaw relative to the ego is the
world yaw minus the ego's (planar: exact when roll and pitch are 0, as on CARLA's flat towns). +y is the ego's RIGHT.

Time. One step = 1 / fps of simulation time (frames.jsonl `sim_t`, from the first recorded step). Every product names
the step (and sim_t) it belongs to; OUT/timeline.jsonl lists, per step, every file of every product for that step.

Products (files under OUT; each in the manifest with its parameters):

* Occupancy — occupancy/<lidar>/<sweep>.npz per lidar sweep, in the ego frame at the sweep's step, on a grid of
  `res` m voxels over x_range × y_range × z_range (default 0.2 m, ±40 m, z −2…4 m). Arrays, indexed [ix, iy, iz]
  from (x_min, y_min, z_min) (ix along x forward, iy along y right):
  - semantics uint8: the class per voxel (core.groundtruth CLASSES: CARLA 0.9.16 tags 0–28, 29 ego), EMPTY = 255.
    From the labelled lidar: the majority class of the sweep's points in the voxel (a point's tag → class; ties to the
    lower class id); then every actor's 3D box (actors.jsonl at that step) fills the voxels whose centre lies inside it
    with the actor's class (by its nuScenes category: class_of_category), overriding the points (a two-wheeler's box
    is its vehicle class whole, rider included, as nuScenes counts riders). The ego's points and box are left out
    unless options include_ego (then class 29).
  - observed bool: the lidar visibility mask, à la Occ3D (Tian et al. 2023, "Occ3D: A Large-Scale 3D Occupancy
    Prediction Benchmark for Autonomous Driving", NeurIPS D&B) and OpenOccupancy (Wang et al. 2023, ICCV): every
    return is ray-cast from the sensor's origin to its point through the grid (an exact voxel traversal, Amanatides &
    Woo 1987, "A Fast Voxel Traversal Algorithm for Ray Tracing"); the voxels a ray passes through before its hit's
    voxel are observed free, the hit's voxel observed.
  - state uint8: 0 UNOBSERVED, 1 FREE (observed and not occupied), 2 OCCUPIED (a lidar class or an actor box; an
    actor's voxels hidden behind its visible face are occupied and not observed).
  - bev_semantics uint8 (image convention below): the class of the highest occupied voxel of each column, EMPTY 255.
  - bev_state uint8: 2 a column with an occupied voxel of a non-ground class (ground: GROUND_TAGS), else 1 when some
    voxel of the column is observed, else 0.
  Limits: CARLA's lidar returns nothing for a ray that hits nothing within its range, so free space is known only
  along rays that hit something; the sweep's points are motion-compensated for the ego only (workers/lidar), so a
  moving actor's points lie where it was at their slice of the sweep (≤ one rotation earlier).
* BEV semantic maps — bev/<step>.npz per recorded step, ego-centred, ego heading UP: a raster of `bev_res` m pixels
  covering `bev_forward` m ahead to `bev_back` m behind and `bev_left` m left to `bev_right` m right. Image convention
  (all BEV arrays here): row 0 is the farthest forward, column 0 the farthest left; pixel (r, c) has its centre at
  x = forward − (r + ½)·res, y = −left + (c + ½)·res in the ego frame — +y (right) is to the RIGHT of the image.
  Arrays: bits uint8 (a bit per channel, BEV_CHANNELS, several may be set), instance int32 (the actor id drawn last
  at the pixel: actors are drawn far to near, the ego last; 0 none). Channels: drivable (the driving lanes of
  lanes.json, each lane's area between consecutive waypoints), centerline (the lanes' centre lines), lane_boundary
  (lane edges with a painted marking, type not NONE), vehicle / pedestrian / cyclist / other (actors' box footprints
  by category, bev_class_of; a footprint that covers no pixel centre takes the pixel of its centre), ego. Lines are
  drawn one pixel wide (half-width res/2) through a world raster at res/2. Limits: the map is lanes.json's driving
  lanes only (no sidewalks, no crossings); without lanes.json the map channels are empty and the manifest says so;
  z is ignored (a bridge over a road draws as both).
* Scene flow — flow/<lidar>/<sweep>.npy per sweep that has a next sweep: float32 (N, 3), one row per point of the
  sweep's .bin in its order: the point's motion from this sweep's time to the next's, in the world, expressed in the
  sensor frame of THIS sweep (a free vector: rotated, not translated). A point whose actor id (its .labels.bin) is an
  actor of actors.jsonl at both steps moves rigidly with that actor: flow = (A₁·A₀⁻¹ − I)·p_world, A the actor's world
  pose at each sweep's step; every other point (the map: id 0; traffic lights; an actor missing at either step) is
  static: 0. The ego's own points move with the ego. Limit: as above, a point is taken where it was recorded.
* Future trajectories — trajectories.npz: for every agent at every step (actors.jsonl, the ego apart), its next
  `horizon_s` s at `traj_rate_hz` (default 3 s at 10 Hz: offsets 0.1 … 3.0 s) in the ego frame at that step: x, y, z
  (m) and yaw relative to the ego (deg). Poses between steps are interpolated linearly in time (positions; yaw along
  the shorter arc) — exact when the rate divides the fps; a future point is valid only when the agent is present at
  the steps on both sides of it (an agent destroyed or out of the run's time has valid False). Arrays: t_offsets (N,),
  agent_step (M,), agent_id (M,), present (M, 4) its pose now, future (M, N, 4), valid (M, N); and the ego's own
  ego_step (S,), ego_future (S, N, 4), ego_valid (S, N) in the ego frame at each step.
* Ego actions — ego_actions.jsonl, one line per step: `control` (throttle, steer, brake, hand_brake, reverse) when the
  run recorded it (drives and live recordings: frames.jsonl `control`; null on a step without control), else null
  and control_source 'not recorded' (drives recorded before 0.6). Always, DERIVED from the pose sequence by finite
  differences (np.gradient: central inside, one-sided at the ends): speed_mps (frames.jsonl's, recorded),
  accel_mps2 (d speed / dt), yaw_rate_dps (d yaw / dt, yaw unwrapped; positive turns right, CARLA's sign),
  curvature_1pm (yaw rate in rad/s ÷ speed; positive to the right; null below CURVATURE_MIN_SPEED), lat_accel_mps2
  (speed × yaw rate in rad/s).
* Multi-camera clips — clips.json: fixed-rate clips for video / world-model training: `clip_rate_hz`,
  `clip_length_s`, `clip_stride_s`. A clip is every k-th step (k = fps / rate) over its length, starting every stride;
  per frame its step, sim_t, t in the clip, every camera's image path (rgb), the lidar sweep if one ends there, the
  ego's pose and its actions (the ego_actions row). A rate that does not divide the drive's fps (k not an integer),
  a stride that is not a whole number of steps, or a length that is not a whole number of clip frames is REFUSED
  (ValueError): images are never interpolated or resampled. A clip with a frame missing a camera's image (a step
  not recorded, a dropped frame) is left out (counted in the manifest) unless clip_require_complete is false; a
  frozen frame (fault injection) is kept and flagged.

Provenance: worldmodel.json carries core.provenance.block over the run's run.json, frames.jsonl, lanes.json and
actors.jsonl, with the options as settings.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import groundtruth as GT
from .rig import carla_rotation

SCHEMA = 'falcon.worldmodel/1'
MANIFEST = 'worldmodel.json'
EMPTY = 255
UNOBSERVED, FREE, OCCUPIED = 0, 1, 2
BEV_CHANNELS = ('drivable', 'centerline', 'lane_boundary', 'vehicle', 'pedestrian', 'cyclist', 'other', 'ego')
BEV_BIT = {name: 1 << k for k, name in enumerate(BEV_CHANNELS)}
CURVATURE_MIN_SPEED = 0.5             # m/s: below it curvature is not stated (yaw rate / speed blows up)
PRODUCTS = ('occupancy', 'bev', 'flow', 'trajectories', 'actions', 'clips')
DEFAULTS = {
    'products': list(PRODUCTS),
    'res': 0.2, 'x_range': [-40.0, 40.0], 'y_range': [-40.0, 40.0], 'z_range': [-2.0, 4.0], 'include_ego': False,
    'bev_res': 0.25, 'bev_forward': 40.0, 'bev_back': 40.0, 'bev_left': 40.0, 'bev_right': 40.0,
    'horizon_s': 3.0, 'traj_rate_hz': 10.0,
    'clip_rate_hz': 10.0, 'clip_length_s': 1.0, 'clip_stride_s': 0.5, 'clip_require_complete': True,
}


# ============================================================================================ classes by category
def class_of_category(cat: str) -> int:
    """core.groundtruth class id (CARLA tag) of an actor by its nuScenes category: what fills its box."""
    c = str(cat or '')
    if c.startswith('human.'):
        return 12
    if c in ('vehicle.bicycle',):
        return 19
    if c == 'vehicle.motorcycle':
        return 18
    if c.startswith('vehicle.bus'):
        return 16
    if c in ('vehicle.truck', 'vehicle.trailer', 'vehicle.construction'):
        return 15
    if c.startswith('vehicle.'):
        return 14                     # cars, ambulances, police cars
    if c.startswith('movable_object.'):
        return 21
    return 22


def bev_class_of(cat: str) -> str:
    """BEV channel of an actor by its nuScenes category: pedestrian (human.*), cyclist (a bicycle or motorcycle and its
    rider), vehicle (every other vehicle.*), other."""
    c = str(cat or '')
    if c.startswith('human.'):
        return 'pedestrian'
    if c in ('vehicle.bicycle', 'vehicle.motorcycle'):
        return 'cyclist'
    if c.startswith('vehicle.'):
        return 'vehicle'
    return 'other'


# ============================================================================================ reading a run
def pose_matrix(p: dict) -> np.ndarray:
    """4 × 4 world matrix of a CARLA pose {x, y, z, roll, pitch, yaw} (Transform::GetMatrix)."""
    M = np.eye(4)
    M[:3, :3] = carla_rotation(float(p.get('roll', 0.0)), float(p.get('pitch', 0.0)), float(p['yaw']))
    M[:3, 3] = (float(p['x']), float(p['y']), float(p.get('z', 0.0)))
    return M


def actor_box(a: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre (3,), rotation (3, 3), half extents (3,)) of an actors.jsonl row's box in the world."""
    M = pose_matrix(a)
    c = M[:3, :3] @ np.asarray(a.get('bbox_center') or (0.0, 0.0, 0.0), float) + M[:3, 3]
    return c, M[:3, :3], np.asarray(a['extent'], float)


def _jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def load_run(run_dir: str) -> dict:
    """{'dir', 'run', 'dataset', 'frames', 'actors': {step: [rows]} or None, 'lanes' or None, 'fps', 'ego_id'}."""
    def opt_json(name):
        p = os.path.join(run_dir, name)
        if not os.path.isfile(p):
            return None
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    frames = _jsonl(os.path.join(run_dir, 'frames.jsonl'))
    if not frames:
        raise ValueError(f'{run_dir}: frames.jsonl has no steps')
    ds = opt_json('dataset.json') or {}
    fps = float(ds.get('fps') or 0.0)
    if fps <= 0:
        dt = np.diff([f['sim_t'] for f in frames])
        if not len(dt):
            raise ValueError(f'{run_dir}: one step and no dataset.json: the step length is unknown')
        fps = 1.0 / float(np.median(dt))
    ap = os.path.join(run_dir, 'actors.jsonl')
    actors = {int(r['step']): r['actors'] for r in _jsonl(ap)} if os.path.isfile(ap) else None
    data = {'dir': run_dir, 'run': opt_json('run.json') or {}, 'dataset': ds, 'frames': frames, 'actors': actors,
            'lanes': opt_json('lanes.json'), 'fps': fps}
    data['ego_id'] = find_ego_id(data)
    return data


def find_ego_id(data: dict):
    """The ego's actor id: the actor with role 'hero' (actors.jsonl), else the actor at the ego's pose of frames.jsonl
    (within 5 cm) at the first step; None without actors.jsonl."""
    acts = data.get('actors')
    if not acts:
        return None
    for rows in acts.values():
        for a in rows:
            if a.get('role') == 'hero':
                return a['id']
    f0 = data['frames'][0]
    best = min(acts.get(f0['step'], []), default=None,
               key=lambda a: math.hypot(a['x'] - f0['ego']['x'], a['y'] - f0['ego']['y']))
    if best is not None and math.hypot(best['x'] - f0['ego']['x'], best['y'] - f0['ego']['y']) < 0.05:
        return best['id']
    return None


def lidar_ids(data: dict) -> list:
    return sorted({k.split('/')[0] for f in data['frames'] for k in f.get('files', {}) if k.endswith('/lidar')})


def lidar_sweeps(data: dict, lid: str) -> list:
    """[(sweep index, step, sim_t, path, 4 × 4 world matrix of the sensor at the sweep's end)] in order."""
    out = []
    for f in data['frames']:
        rel = f.get('files', {}).get(f'{lid}/lidar')
        if rel is None or lid not in (f.get('lidar_pose') or {}):
            continue
        M = np.eye(4)
        M[:3, :] = np.asarray(f['lidar_pose'][lid], float).reshape(3, 4)
        n = int(os.path.splitext(os.path.basename(rel))[0])
        out.append((n, f['step'], f['sim_t'], rel, M))
    return out


def read_sweep(run_dir: str, rel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xyz (N, 3) float64 in the sensor frame, tag (N,) uint8, actor id (N,) uint32) of a sweep."""
    p = os.path.join(run_dir, rel)
    a = np.fromfile(p, np.float32).reshape(-1, 4)
    n = len(a)
    lab = np.fromfile(p[:-4] + '.labels.bin', np.uint8)
    if len(lab) != 5 * n:
        raise ValueError(f'{rel}: {len(lab)} label bytes for {n} points (expected {5 * n})')
    return a[:, :3].astype(np.float64), lab[:n].copy(), lab[n:].view(np.uint32).copy()


# ============================================================================================ occupancy
class Grid:
    """A voxel grid in some frame: res m, x/y/z ranges [min, max) (each a whole number of voxels, rounded up)."""

    def __init__(self, res=0.2, x_range=(-40.0, 40.0), y_range=(-40.0, 40.0), z_range=(-2.0, 4.0)):
        self.res = float(res)
        self.lo = np.array([x_range[0], y_range[0], z_range[0]], float)
        span = np.array([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]], float)
        if self.res <= 0 or (span <= 0).any():
            raise ValueError('a grid needs res > 0 and every range max > min')
        self.shape = tuple(int(v) for v in np.ceil(span / self.res - 1e-9))

    def doc(self) -> dict:
        return {'res_m': self.res, 'origin_m': [float(v) for v in self.lo], 'shape': list(self.shape),
                'index': '[ix, iy, iz]: voxel centre = origin + (index + 0.5) * res (x forward, y right, z up)'}

    def index(self, P) -> np.ndarray:
        return np.floor((np.asarray(P, float) - self.lo) / self.res).astype(np.int64)

    def inside(self, ijk) -> np.ndarray:
        return np.all((ijk >= 0) & (ijk < np.asarray(self.shape)), axis=-1)

    def centres(self, ijk) -> np.ndarray:
        return self.lo + (np.asarray(ijk, float) + 0.5) * self.res


def traverse(grid: Grid, origin, ends) -> tuple[np.ndarray, np.ndarray]:
    """Exact voxel traversal (Amanatides & Woo 1987) of the segments origin → each end, batched over the rays.
    Returns (free, hit): bool grids — the voxels a ray passes through before its end's voxel, and the end voxels (in
    the grid). The origin must lie inside the grid (a lidar on the ego, the grid around it)."""
    shape = np.asarray(grid.shape)
    o = (np.asarray(origin, float) - grid.lo) / grid.res
    if not ((o >= 0) & (o < shape)).all():
        raise ValueError('the ray origin lies outside the grid')
    e = (np.asarray(ends, float).reshape(-1, 3) - grid.lo) / grid.res
    free = np.zeros(grid.shape, bool)
    hit = np.zeros(grid.shape, bool)
    if not len(e):
        return free, hit
    end = np.floor(e).astype(np.int64)
    ein = grid.inside(end)
    hit[end[ein, 0], end[ein, 1], end[ein, 2]] = True
    d = e - o
    cur = np.broadcast_to(np.floor(o).astype(np.int64), e.shape).copy()
    step = np.sign(d).astype(np.int64)
    ad = np.abs(d)
    with np.errstate(divide='ignore', invalid='ignore'):
        tdelta = np.where(ad > 0, 1.0 / ad, np.inf)
        nxt = np.where(step > 0, np.floor(o) + 1.0 - o, o - np.floor(o))
        tmax = np.where(ad > 0, nxt / ad, np.inf)
    act = np.flatnonzero(np.any(cur != end, axis=1))
    rows = np.arange(len(e))
    while len(act):
        c = cur[act]
        free[c[:, 0], c[:, 1], c[:, 2]] = True        # active rays are inside the grid, short of their end voxel
        tm = tmax[act]
        ax = np.argmin(tm, axis=1)
        t = tm[rows[:len(act)], ax]
        cur[act, ax] += step[act, ax]
        tmax[act, ax] += tdelta[act, ax]
        c = cur[act]
        done = (t >= 1.0) | np.all(c == end[act], axis=1) | ~grid.inside(c)
        act = act[~done]
    return free, hit


def fill_boxes(grid: Grid, boxes) -> tuple[np.ndarray, np.ndarray]:
    """(mask, class grid): the voxels whose centre lies inside each box {'center', 'R', 'extent' (half), 'class'}
    (in the grid's frame); later boxes win where they overlap."""
    mask = np.zeros(grid.shape, bool)
    cls = np.full(grid.shape, EMPTY, np.uint8)
    for b in boxes:
        c, R, ext = np.asarray(b['center'], float), np.asarray(b['R'], float), np.asarray(b['extent'], float)
        corners = c + (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * ext) @ R.T
        i0 = np.clip(grid.index(corners.min(0)), 0, np.asarray(grid.shape) - 1)
        i1 = np.clip(grid.index(corners.max(0)), 0, np.asarray(grid.shape) - 1)
        if (grid.index(corners.max(0)) < 0).any() or (grid.index(corners.min(0)) >= np.asarray(grid.shape)).any():
            continue
        ii, jj, kk = np.meshgrid(*(np.arange(i0[a], i1[a] + 1) for a in range(3)), indexing='ij')
        ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
        local = (grid.centres(ijk) - c) @ R
        inside = np.all(np.abs(local) <= ext + 1e-9, axis=1)
        sel = ijk[inside]
        mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
        cls[sel[:, 0], sel[:, 1], sel[:, 2]] = int(b['class'])
    return mask, cls


def occupancy(points, classes, origin, grid: Grid, boxes=()) -> dict:
    """Semantic occupancy with its lidar visibility of one sweep (see the module doc), all in one frame (the grid's).
    points (N, 3) and their class ids (N,) — the returns to vote and to ray-cast; origin: the sensor's position;
    boxes: [{'center', 'R', 'extent' (half sizes), 'class'}] filled over the points."""
    P = np.asarray(points, float).reshape(-1, 3)
    cl = np.asarray(classes, np.int64).reshape(-1)
    sem = np.full(grid.shape, EMPTY, np.uint8)
    ijk = grid.index(P)
    ok = grid.inside(ijk)
    if ok.any():
        flat = np.ravel_multi_index(tuple(ijk[ok].T), grid.shape)
        key = flat * 256 + cl[ok]
        u, cnt = np.unique(key, return_counts=True)
        vox, c = u // 256, u % 256
        order = np.lexsort((c, -cnt, vox))            # per voxel: most points first, then the lower class id
        vox, c = vox[order], c[order]
        first = np.r_[True, vox[1:] != vox[:-1]]
        sem.ravel()[vox[first]] = c[first].astype(np.uint8)
    free, hit = traverse(grid, origin, P)
    bmask, bcls = fill_boxes(grid, boxes)
    sem[bmask] = bcls[bmask]
    occ = sem != EMPTY
    observed = free | hit
    state = np.where(occ, OCCUPIED, np.where(observed, FREE, UNOBSERVED)).astype(np.uint8)
    return dict(semantics=sem, observed=observed, state=state, **bev_of_grid(sem, observed))


def bev_of_grid(sem: np.ndarray, observed: np.ndarray) -> dict:
    """bev_semantics and bev_state (image convention: row 0 farthest forward, column 0 farthest left) of a 3D grid."""
    occ = sem != EMPTY
    nz = sem.shape[2]
    top = np.where(occ.any(2), nz - 1 - np.argmax(occ[:, :, ::-1], axis=2), -1)
    ii, jj = np.indices(top.shape)
    bev_sem = np.where(top >= 0, sem[ii, jj, np.maximum(top, 0)], EMPTY).astype(np.uint8)
    obstacle = (occ & ~GT.GROUND_LUT[sem]).any(2)
    bev_state = np.where(obstacle, OCCUPIED, np.where(observed.any(2), FREE, UNOBSERVED)).astype(np.uint8)
    return {'bev_semantics': bev_sem[::-1, :].copy(), 'bev_state': bev_state[::-1, :].copy()}


def sweep_occupancy(data: dict, lid: str, sweep, grid: Grid, include_ego=False) -> dict:
    """occupancy() of one recorded sweep (lidar_sweeps() row) in the ego frame at its step."""
    _n, step, _t, rel, M_ws = sweep
    frame = next(f for f in data['frames'] if f['step'] == step)
    T_we = pose_matrix(frame['ego'])
    T_es = np.linalg.inv(T_we) @ M_ws
    xyz, tag, ids = read_sweep(data['dir'], rel)
    P = xyz @ T_es[:3, :3].T + T_es[:3, 3]
    cls = GT.CLASS_OF_TAG[tag].astype(np.int64)
    ego = data.get('ego_id')
    if ego is not None:
        on_ego = ids == np.uint32(ego)
        if include_ego:
            cls[on_ego] = GT.EGO_CLASS
        else:
            P, cls = P[~on_ego], cls[~on_ego]
    boxes = []
    Ti = np.linalg.inv(T_we)
    for a in (data.get('actors') or {}).get(step, []):
        is_ego = a['id'] == ego
        if is_ego and not include_ego:
            continue
        c, R, ext = actor_box(a)
        boxes.append({'center': Ti[:3, :3] @ c + Ti[:3, 3], 'R': Ti[:3, :3] @ R, 'extent': ext,
                      'class': GT.EGO_CLASS if is_ego else class_of_category(a.get('category'))})
    return occupancy(P, cls, T_es[:3, 3], grid, boxes)


# ============================================================================================ BEV semantic maps
class BevSpec:
    """An ego-centred raster, ego heading up: res m pixels, forward/back/left/right extents in m."""

    def __init__(self, res=0.25, forward=40.0, back=40.0, left=40.0, right=40.0):
        self.res, self.forward, self.back, self.left, self.right = float(res), float(forward), float(back), \
            float(left), float(right)
        self.h = int(math.ceil((self.forward + self.back) / self.res - 1e-9))
        self.w = int(math.ceil((self.left + self.right) / self.res - 1e-9))

    def doc(self) -> dict:
        return {'res_m': self.res, 'forward_m': self.forward, 'back_m': self.back, 'left_m': self.left,
                'right_m': self.right, 'height': self.h, 'width': self.w,
                'pixel_centre': 'x = forward - (row + 0.5) * res, y = -left + (col + 0.5) * res (ego frame; +y '
                                'right, so the image shows the ego heading up with its right on the right)',
                'channels': {name: BEV_BIT[name] for name in BEV_CHANNELS}}

    def pixel(self, xy) -> np.ndarray:
        """(row, col) floats of ego-frame (x, y) points."""
        xy = np.asarray(xy, float)
        return np.stack([(self.forward - xy[..., 0]) / self.res - 0.5, (xy[..., 1] + self.left) / self.res - 0.5],
                        -1)

    def centres(self) -> np.ndarray:
        r, c = np.indices((self.h, self.w))
        return np.stack([self.forward - (r + 0.5) * self.res, -self.left + (c + 0.5) * self.res], -1)


def _fill_convex(img, value, poly_rc, op='or'):
    """Pixels of img (h, w) whose centre lies in the convex polygon poly_rc ((k, 2) row, col of pixel CENTRES as
    BevSpec.pixel gives) take value (bitwise or when op == 'or', else assigned). Returns the mask's pixel count."""
    P = np.asarray(poly_rc, float)
    h, w = img.shape
    r0, c0 = max(int(math.floor(P[:, 0].min())), 0), max(int(math.floor(P[:, 1].min())), 0)
    r1, c1 = min(int(math.ceil(P[:, 0].max())), h - 1), min(int(math.ceil(P[:, 1].max())), w - 1)
    if r1 < r0 or c1 < c0:
        return 0
    rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    pts = np.stack([rr.ravel(), cc.ravel()], 1).astype(float)
    e = np.roll(P, -1, 0) - P
    cross = e[None, :, 0] * (pts[:, None, 1] - P[None, :, 1]) - e[None, :, 1] * (pts[:, None, 0] - P[None, :, 0])
    m = np.all(cross >= -1e-9, 1) | np.all(cross <= 1e-9, 1)
    if not m.any():
        return 0
    sub = img[r0:r1 + 1, c0:c1 + 1].reshape(-1)
    if op == 'or':
        sub[m] |= value
    else:
        sub[m] = value
    img[r0:r1 + 1, c0:c1 + 1] = sub.reshape(r1 - r0 + 1, c1 - c0 + 1)
    return int(m.sum())


def _segment_quad(a, b, half):
    """A thin quad of half-width `half` around segment a–b (2D)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = np.linalg.norm(d)
    u = d / n if n > 1e-9 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]]) * half
    a2, b2 = a - u * half, b + u * half
    return np.array([a2 + v, b2 + v, b2 - v, a2 - v])


class LaneMap:
    """lanes.json rasterised once in the world (x, y), at `cell` m over `bounds` (xmin, ymin, xmax, ymax): the bits
    drivable, centerline and lane_boundary; sampled per frame at the BEV's pixel centres (nearest cell)."""

    def __init__(self, lanes: dict, bounds, cell: float):
        self.cell = float(cell)
        self.x0, self.y0 = float(bounds[0]), float(bounds[1])
        self.w = int(math.ceil((bounds[2] - bounds[0]) / self.cell)) + 1
        self.h = int(math.ceil((bounds[3] - bounds[1]) / self.cell)) + 1
        self.grid = np.zeros((self.h, self.w), np.uint8)      # [iy, ix]
        cols = lanes['columns']
        ix = {k: cols.index(k) for k in ('x', 'y', 'yaw', 'width', 'road', 'section', 'lane', 's', 'left_type',
                                         'right_type')}
        step = float(lanes.get('step_m', 2.0))
        groups = {}
        for r in lanes['points']:
            groups.setdefault((r[ix['road']], r[ix['section']], r[ix['lane']]), []).append(r)
        half_line = self.cell                                  # one BEV pixel wide (cell = res / 2)
        margin = step + 10.0
        for rows in groups.values():
            rows.sort(key=lambda r: r[ix['s']])
            P = np.array([[r[ix['x']], r[ix['y']]] for r in rows], float)
            yaw = np.radians([r[ix['yaw']] for r in rows])
            wd = np.array([r[ix['width']] for r in rows], float)
            right = np.stack([-np.sin(yaw), np.cos(yaw)], 1)
            fwd = np.stack([np.cos(yaw), np.sin(yaw)], 1)
            L, R = P - right * (wd / 2)[:, None], P + right * (wd / 2)[:, None]
            near = (P[:, 0] > bounds[0] - margin) & (P[:, 0] < bounds[2] + margin) & \
                (P[:, 1] > bounds[1] - margin) & (P[:, 1] < bounds[3] + margin)
            for k in range(len(rows)):
                if not near[k]:
                    continue
                # each waypoint's own stretch (±step/2 along it): covers lane ends and isolated waypoints
                hs = fwd[k] * step / 2
                self._poly(BEV_BIT['drivable'], [L[k] - hs, L[k] + hs, R[k] + hs, R[k] - hs])
                if k + 1 < len(rows) and np.linalg.norm(P[k + 1] - P[k]) <= 1.5 * step:
                    self._poly(BEV_BIT['drivable'], [L[k], L[k + 1], R[k + 1], R[k]])
                    self._poly(BEV_BIT['centerline'], _segment_quad(P[k], P[k + 1], half_line))
                    if str(rows[k][ix['left_type']]).upper() != 'NONE':
                        self._poly(BEV_BIT['lane_boundary'], _segment_quad(L[k], L[k + 1], half_line))
                    if str(rows[k][ix['right_type']]).upper() != 'NONE':
                        self._poly(BEV_BIT['lane_boundary'], _segment_quad(R[k], R[k + 1], half_line))

    def _poly(self, bit, xy):
        xy = np.asarray(xy, float)
        rc = np.stack([(xy[:, 1] - self.y0) / self.cell - 0.5, (xy[:, 0] - self.x0) / self.cell - 0.5], 1)
        _fill_convex(self.grid, bit, rc)

    def sample(self, world_xy) -> np.ndarray:
        ix = np.floor((world_xy[..., 0] - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((world_xy[..., 1] - self.y0) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        out = np.zeros(ix.shape, np.uint8)
        out[ok] = self.grid[iy[ok], ix[ok]]
        return out


def lane_map_for(lanes: dict | None, ego_poses, spec: BevSpec):
    """A LaneMap covering every BEV the ego poses will draw (None without lanes)."""
    if not lanes or not lanes.get('points'):
        return None
    xy = np.array([[p['x'], p['y']] for p in ego_poses], float)
    r = math.hypot(max(spec.forward, spec.back), max(spec.left, spec.right)) + 2.0
    return LaneMap(lanes, (xy[:, 0].min() - r, xy[:, 1].min() - r, xy[:, 0].max() + r, xy[:, 1].max() + r),
                   spec.res / 2.0)


def bev_raster(ego_pose: dict, actors, spec: BevSpec, lane_map: LaneMap | None = None, ego_id=None) -> dict:
    """One BEV map (module doc): {'bits': (h, w) uint8, 'instance': (h, w) int32}. ego_pose: the ego's CARLA pose;
    actors: actors.jsonl rows (the ego among them is drawn in the ego channel; or ego_id names it)."""
    T = pose_matrix(ego_pose)
    Ti = np.linalg.inv(T)
    bits = np.zeros((spec.h, spec.w), np.uint8)
    inst = np.zeros((spec.h, spec.w), np.int32)
    if lane_map is not None:
        C = spec.centres()
        Pw = np.concatenate([C, np.zeros(C.shape[:-1] + (1,))], -1) @ T[:3, :3].T + T[:3, 3]
        bits |= lane_map.sample(Pw[..., :2])
    rows = []
    for a in actors or ():
        c, R, ext = actor_box(a)
        ce = Ti[:3, :3] @ c + Ti[:3, 3]
        Re = Ti[:3, :3] @ R
        corners = np.array([[sx * ext[0], sy * ext[1], 0.0] for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        foot = (corners @ Re.T + ce)[:, :2]
        is_ego = (ego_id is not None and a['id'] == ego_id) or (ego_id is None and a.get('role') == 'hero')
        rows.append((is_ego, -float(np.hypot(*ce[:2])), a, foot))
    rows.sort(key=lambda r: (r[0], r[1]))                     # far to near, the ego last
    for is_ego, _d, a, foot in rows:
        rc = spec.pixel(foot)
        bit = BEV_BIT['ego' if is_ego else bev_class_of(a.get('category'))]
        if _fill_convex(bits, bit, rc) == 0:                   # smaller than a pixel: the pixel of its centre
            r, c = np.floor(rc.mean(0) + 0.5).astype(int)
            if 0 <= r < spec.h and 0 <= c < spec.w:
                bits[r, c] |= bit
                inst[r, c] = int(a['id'])
            continue
        _fill_convex(inst, int(a['id']), rc, op='set')
    return {'bits': bits, 'instance': inst}


# ============================================================================================ scene flow
def scene_flow(points, ids, M_ws, poses_a: dict, poses_b: dict) -> np.ndarray:
    """Per point (N, 3, sensor frame of sweep a, M_ws its world matrix) its motion to sweep b: (N, 3) float32 in that
    sensor frame. ids: the points' actor ids; poses_a / poses_b: {actor id: 4 × 4 world matrix} at the two sweeps'
    times. A point of an actor in both moves with it rigidly; every other point is static (0)."""
    P = np.asarray(points, float).reshape(-1, 3)
    ids = np.asarray(ids).reshape(-1).astype(np.int64)
    flow = np.zeros_like(P)
    Pw = P @ M_ws[:3, :3].T + M_ws[:3, 3]
    for aid in np.unique(ids):
        aid = int(aid)
        if aid == 0 or aid not in poses_a or aid not in poses_b:
            continue
        sel = ids == aid
        D = np.asarray(poses_b[aid], float) @ np.linalg.inv(np.asarray(poses_a[aid], float))
        flow[sel] = Pw[sel] @ D[:3, :3].T + D[:3, 3] - Pw[sel]
    return (flow @ M_ws[:3, :3]).astype(np.float32)            # world vectors -> sensor axes (R^T v)


def _poses_at(data: dict, step: int) -> dict:
    return {a['id']: pose_matrix(a) for a in (data.get('actors') or {}).get(step, [])}


# ============================================================================================ future trajectories
def _tracks(timeline) -> dict:
    """{agent id: (steps, times, xyz (n, 3), yaw unwrapped deg)} from [{'step', 'sim_t', 'actors'}]."""
    acc = {}
    for rec in timeline:
        for a in rec.get('actors') or ():
            acc.setdefault(a['id'], []).append((rec['step'], rec['sim_t'], a['x'], a['y'], a.get('z', 0.0), a['yaw']))
    out = {}
    for aid, rows in acc.items():
        rows.sort()
        A = np.asarray(rows, float)
        out[aid] = (A[:, 0].astype(np.int64), A[:, 1], A[:, 2:5], np.degrees(np.unwrap(np.radians(A[:, 5]))))
    return out


def _sample(track, times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xyz (n, 3), yaw (n,), valid (n,)) of a track at times: linear between the steps around each time, valid
    only where those two samples are consecutive steps (or the time is a sample's own)."""
    steps, t, xyz, yaw = track
    times = np.asarray(times, float)
    j = np.searchsorted(t, times - 1e-9, side='left')
    exact = (j < len(t)) & (np.abs(t[np.minimum(j, len(t) - 1)] - times) <= 1e-6)
    j0 = np.clip(j - 1, 0, len(t) - 1)
    j1 = np.clip(j, 0, len(t) - 1)
    valid = exact | ((j > 0) & (j < len(t)) & (steps[j1] - steps[j0] == 1))
    a = np.where(exact, j1, j0)
    b = np.where(exact, j1, j1)
    span = np.where(b > a, t[b] - t[a], 1.0)
    w = np.clip(np.where(b > a, (times - t[a]) / span, 0.0), 0.0, 1.0)
    P = xyz[a] + (xyz[b] - xyz[a]) * w[:, None]
    Y = yaw[a] + (yaw[b] - yaw[a]) * w
    return P, Y, valid


def _wrap(deg):
    return (np.asarray(deg, float) + 180.0) % 360.0 - 180.0


def future_trajectories(timeline, horizon_s: float = 3.0, rate_hz: float = 10.0, ego_id=None) -> dict:
    """Every agent's and the ego's future in the ego frame at each step (module doc). timeline: [{'step', 'sim_t',
    'ego': pose, 'actors': actors.jsonl rows}] in step order; the ego (ego_id, else role 'hero') is not an agent."""
    n = int(round(horizon_s * rate_hz))
    if n < 1:
        raise ValueError('the horizon holds no sample at this rate')
    offs = (np.arange(n) + 1) / float(rate_hz)
    tl = []
    for rec in timeline:
        acts = [a for a in rec.get('actors') or () if not (a['id'] == ego_id or (ego_id is None and
                                                                                  a.get('role') == 'hero'))]
        tl.append(dict(rec, actors=acts))
    tracks = _tracks(tl)
    ego_track = _tracks([{'step': r['step'], 'sim_t': r['sim_t'], 'actors': [dict(r['ego'], id=-1)]}
                         for r in timeline])[-1]
    a_step, a_id, a_now, a_fut, a_val = [], [], [], [], []
    e_fut, e_val = [], []
    for rec in tl:
        T = pose_matrix(rec['ego'])
        Ri, ti = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
        yaw0 = float(rec['ego']['yaw'])
        times = rec['sim_t'] + offs
        P, Y, v = _sample(ego_track, times)
        e_fut.append(np.c_[P @ Ri.T + ti, _wrap(Y - yaw0)])
        e_val.append(v)
        for a in rec['actors']:
            P, Y, v = _sample(tracks[a['id']], times)
            now = np.r_[Ri @ np.array([a['x'], a['y'], a.get('z', 0.0)]) + ti, _wrap(a['yaw'] - yaw0)]
            a_step.append(rec['step'])
            a_id.append(a['id'])
            a_now.append(now)
            a_fut.append(np.c_[P @ Ri.T + ti, _wrap(Y - yaw0)])
            a_val.append(v)
    return {'t_offsets': offs.astype(np.float32),
            'agent_step': np.asarray(a_step, np.int64), 'agent_id': np.asarray(a_id, np.int64),
            'present': np.asarray(a_now, np.float32).reshape(-1, 4),
            'future': np.asarray(a_fut, np.float32).reshape(-1, n, 4), 'valid': np.asarray(a_val, bool).reshape(-1, n),
            'ego_step': np.asarray([r['step'] for r in tl], np.int64),
            'ego_future': np.asarray(e_fut, np.float32).reshape(-1, n, 4),
            'ego_valid': np.asarray(e_val, bool).reshape(-1, n)}


# ============================================================================================ ego actions
CONTROL_KEYS = ('throttle', 'steer', 'brake', 'hand_brake', 'reverse')


def ego_actions(frames) -> list:
    """One row per step (module doc): the recorded control when the run has it, and the derived kinematics."""
    t = np.array([f['sim_t'] for f in frames], float)
    speed = np.array([f.get('speed_mps', 0.0) for f in frames], float)
    yaw = np.degrees(np.unwrap(np.radians([f['ego']['yaw'] for f in frames])))
    if len(t) >= 2:
        acc = np.gradient(speed, t)
        yr = np.gradient(yaw, t)
    else:
        acc, yr = np.zeros(len(t)), np.zeros(len(t))
    recorded = any('control' in f for f in frames)
    out = []
    for k, f in enumerate(frames):
        w = math.radians(yr[k])
        c = f.get('control') if recorded else None
        out.append({'step': f['step'], 'sim_t': f['sim_t'], 'speed_mps': round(float(speed[k]), 4),
                    'accel_mps2': round(float(acc[k]), 4), 'yaw_rate_dps': round(float(yr[k]), 4),
                    'curvature_1pm': round(w / speed[k], 6) if speed[k] >= CURVATURE_MIN_SPEED else None,
                    'lat_accel_mps2': round(float(speed[k] * w), 4),
                    'control': {k2: c.get(k2) for k2 in CONTROL_KEYS if k2 in c} if isinstance(c, dict) else None,
                    'control_source': ('recorded' if isinstance(c, dict) else 'recorded: none applied') if recorded
                    else 'not recorded',
                    'derived': ['accel_mps2', 'yaw_rate_dps', 'curvature_1pm', 'lat_accel_mps2']})
    return out


# ============================================================================================ clips
def _whole(x, what):
    k = int(round(x))
    if k < 1 or abs(x - k) > 1e-6 * max(1.0, abs(x)):
        raise ValueError(what)
    return k


def camera_ids(frames) -> list:
    return sorted({k.split('/')[0] for f in frames for k in f.get('files', {}) if k.endswith('/rgb')})


def clip_manifest(frames, fps: float, rate_hz: float, length_s: float, stride_s: float, actions=None,
                  require_complete: bool = True) -> dict:
    """Fixed-rate multi-camera clips (module doc). Refuses (ValueError) a rate that does not divide fps, a stride
    that is not whole steps and a length that is not whole clip frames — images are never interpolated."""
    k = _whole(fps / rate_hz, f'clip rate {rate_hz:g} Hz does not divide the drive\'s {fps:g} Hz: refused (frames '
                              'would have to be interpolated or unevenly spaced; images are never interpolated)')
    stride = _whole(stride_s * fps, f'clip stride {stride_s:g} s is not a whole number of steps at {fps:g} Hz: refused')
    n = _whole(length_s * rate_hz, f'clip length {length_s:g} s is not a whole number of frames at {rate_hz:g} Hz: '
                                   'refused')
    cams = camera_ids(frames)
    by_step = {f['step']: f for f in frames}
    act = {a['step']: a for a in actions or ()}
    steps = sorted(by_step)
    clips, dropped = [], 0
    if steps:
        s0, s_last = steps[0], steps[-1]
        start = s0
        while start + (n - 1) * k <= s_last:
            fr, complete = [], True
            for j in range(n):
                s = start + j * k
                f = by_step.get(s)
                if f is None:
                    complete = False
                    break
                files = f.get('files', {})
                imgs = {c: files.get(f'{c}/rgb') for c in cams}
                if f.get('recorded') is False or any(v is None for v in imgs.values()):
                    complete = False
                faults = f.get('faults') or {}
                row = {'step': s, 'sim_t': f['sim_t'], 't': round(j / rate_hz, 6), 'images': imgs,
                       'ego': f['ego'], 'speed_mps': f.get('speed_mps'), 'action': act.get(s)}
                lid = {kk.split('/')[0]: v for kk, v in files.items() if kk.endswith('/lidar')}
                if lid:
                    row['lidar'] = lid
                frozen = sorted(c for c in cams if faults.get(c) == 'freeze')
                if frozen:
                    row['frozen'] = frozen
                fr.append(row)
            if complete or (not require_complete and fr):
                clips.append({'clip': len(clips), 'start_step': start, 'start_t': by_step[start]['sim_t'],
                              'complete': complete, 'frames': fr})
            else:
                dropped += 1
            start += stride
    return {'rate_hz': rate_hz, 'length_s': length_s, 'stride_s': stride_s, 'frames_per_clip': n,
            'step_stride': k, 'cameras': cams, 'clips': clips, 'dropped_incomplete': dropped,
            'rule': 'every k-th recorded step (k = fps / rate), no interpolation; incomplete clips '
                    + ('left out' if require_complete else 'kept, complete: false')}


# ============================================================================================ derive
def _settings(options) -> dict:
    o = dict(DEFAULTS)
    o.update(options or {})
    unknown = sorted(set(o) - set(DEFAULTS))
    if unknown:
        raise ValueError(f'unknown options: {", ".join(unknown)}')
    bad = sorted(set(o['products']) - set(PRODUCTS))
    if bad:
        raise ValueError(f'unknown products: {", ".join(bad)} (known: {", ".join(PRODUCTS)})')
    return o


def _savez(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def derive(run_dir: str, out_dir: str, options: dict | None = None) -> dict:
    """Every product (module doc) of one run into out_dir; returns the manifest written to OUT/worldmodel.json.
    options: DEFAULTS' keys. Invalid options (an unknown key, a clip rate that does not divide the fps) are refused
    with ValueError before anything is written. A product whose input the run lacks (no lidar, no actors.jsonl) is
    skipped and the manifest says why."""
    from .provenance import block, write_json
    o = _settings(options)
    data = load_run(run_dir)
    fps = data['fps']
    frames = data['frames']
    if 'clips' in o['products']:
        clip_manifest([], fps, o['clip_rate_hz'], o['clip_length_s'], o['clip_stride_s'])   # refuse early
    os.makedirs(out_dir, exist_ok=True)
    have_actors = data['actors'] is not None
    lids = lidar_ids(data)
    products, skipped = {}, {}
    per_step = {f['step']: {'step': f['step'], 'sim_t': f['sim_t'], 'frame': f.get('frame')} for f in frames}

    def need_actors(name):
        if not have_actors:
            skipped[name] = 'the run has no actors.jsonl (every actor\'s pose per step)'
        return have_actors

    if 'occupancy' in o['products'] and lids:
        grid = Grid(o['res'], o['x_range'], o['y_range'], o['z_range'])
        n = 0
        for lid in lids:
            for sw in lidar_sweeps(data, lid):
                occ = sweep_occupancy(data, lid, sw, grid, o['include_ego'])
                rel = os.path.join('occupancy', lid, f'{sw[0]:06d}.npz')
                _savez(os.path.join(out_dir, rel), **occ)
                per_step[sw[1]].setdefault('occupancy', {})[lid] = rel
                n += 1
        products['occupancy'] = {'files': 'occupancy/<lidar>/<sweep>.npz', 'sweeps': n, 'grid': grid.doc(),
                                 'frame': 'ego frame at the sweep\'s step', 'empty': EMPTY,
                                 'state': {'unobserved': UNOBSERVED, 'free': FREE, 'occupied': OCCUPIED},
                                 'arrays': ['semantics', 'observed', 'state', 'bev_semantics', 'bev_state'],
                                 'actor_boxes': have_actors, 'include_ego': o['include_ego'],
                                 'classes': 'core.groundtruth classes (dataset.json labels.semantic.classes)'}
        if not have_actors:
            products['occupancy']['note'] = 'no actors.jsonl: lidar votes only, no boxes filled'
    elif 'occupancy' in o['products']:
        skipped['occupancy'] = 'the run has no lidar sweeps'

    if 'bev' in o['products'] and need_actors('bev'):
        spec = BevSpec(o['bev_res'], o['bev_forward'], o['bev_back'], o['bev_left'], o['bev_right'])
        rec_frames = [f for f in frames if f.get('recorded', True)]
        lm = lane_map_for(data['lanes'], [f['ego'] for f in rec_frames], spec)
        for f in rec_frames:
            b = bev_raster(f['ego'], data['actors'].get(f['step'], []), spec, lm, data['ego_id'])
            rel = os.path.join('bev', f'{f["step"]:06d}.npz')
            _savez(os.path.join(out_dir, rel), **b)
            per_step[f['step']]['bev'] = rel
        products['bev'] = {'files': 'bev/<step>.npz', 'frames': len(rec_frames), 'spec': spec.doc(),
                           'arrays': ['bits', 'instance'], 'map': 'lanes.json' if lm is not None else
                           'none: the run has no lanes.json, map channels empty',
                           'actor_classes': {'pedestrian': 'human.*', 'cyclist': 'vehicle.bicycle, vehicle.motorcycle',
                                             'vehicle': 'other vehicle.*', 'other': 'everything else'}}

    if 'flow' in o['products'] and lids and need_actors('flow'):
        n = 0
        for lid in lids:
            sws = lidar_sweeps(data, lid)
            for a, b in zip(sws, sws[1:]):
                xyz, _tag, ids = read_sweep(run_dir, a[3])
                fl = scene_flow(xyz, ids, a[4], _poses_at(data, a[1]), _poses_at(data, b[1]))
                rel = os.path.join('flow', lid, f'{a[0]:06d}.npy')
                os.makedirs(os.path.dirname(os.path.join(out_dir, rel)), exist_ok=True)
                np.save(os.path.join(out_dir, rel), fl)
                per_step[a[1]].setdefault('flow', {})[lid] = {'file': rel, 'to_step': b[1],
                                                              'dt_s': round(b[2] - a[2], 6)}
                n += 1
        products['flow'] = {'files': 'flow/<lidar>/<sweep>.npy', 'pairs': n, 'dtype': 'float32 (N, 3)',
                            'frame': 'sensor frame of the first sweep (a vector)', 'units': 'm over the sweep gap',
                            'rule': 'rigid motion of the point\'s actor between the sweeps\' steps; static 0'}
    elif 'flow' in o['products'] and not lids:
        skipped['flow'] = 'the run has no lidar sweeps'

    actions = ego_actions(frames)
    if 'trajectories' in o['products'] and need_actors('trajectories'):
        tl = [{'step': f['step'], 'sim_t': f['sim_t'], 'ego': f['ego'], 'actors': data['actors'].get(f['step'], [])}
              for f in frames]
        tr = future_trajectories(tl, o['horizon_s'], o['traj_rate_hz'], data['ego_id'])
        _savez(os.path.join(out_dir, 'trajectories.npz'), **tr)
        for k, s in enumerate(tr['ego_step']):
            per_step[int(s)]['trajectories'] = {'file': 'trajectories.npz', 'ego_row': k}
        products['trajectories'] = {'file': 'trajectories.npz', 'agents_rows': int(len(tr['agent_id'])),
                                    'horizon_s': o['horizon_s'], 'rate_hz': o['traj_rate_hz'],
                                    'columns': ['x', 'y', 'z', 'yaw_rel_deg'], 'frame': 'ego frame at the step',
                                    'interpolation': 'linear in time between consecutive steps' +
                                    (' (none needed: the rate divides the fps)' if
                                     abs(fps / o['traj_rate_hz'] - round(fps / o['traj_rate_hz'])) < 1e-6 else '')}
    if 'actions' in o['products']:
        with open(os.path.join(out_dir, 'ego_actions.jsonl'), 'w', encoding='utf-8') as fh:
            for k, a in enumerate(actions):
                fh.write(json.dumps(a, separators=(',', ':')) + '\n')
                per_step[a['step']]['actions'] = {'file': 'ego_actions.jsonl', 'line': k}
        products['actions'] = {'file': 'ego_actions.jsonl', 'steps': len(actions),
                               'control': actions[0]['control_source'] if actions else 'not recorded',
                               'derived': 'finite differences of the pose sequence (np.gradient)'}
    if 'clips' in o['products']:
        cm = clip_manifest(frames, fps, o['clip_rate_hz'], o['clip_length_s'], o['clip_stride_s'], actions,
                           o['clip_require_complete'])
        write_json(os.path.join(out_dir, 'clips.json'), dict(cm, schema=SCHEMA, fps=fps))
        products['clips'] = {'file': 'clips.json', 'clips': len(cm['clips']), 'cameras': cm['cameras'],
                             'dropped_incomplete': cm['dropped_incomplete'], 'step_stride': cm['step_stride']}
    with open(os.path.join(out_dir, 'timeline.jsonl'), 'w', encoding='utf-8') as fh:
        for s in sorted(per_step):
            fh.write(json.dumps(per_step[s], separators=(',', ':')) + '\n')
    ins = {k: os.path.join(run_dir, n) for k, n in (('run', 'run.json'), ('frames', 'frames.jsonl'),
                                                    ('lanes', 'lanes.json'), ('actors', 'actors.jsonl'))}
    manifest = {'schema': SCHEMA, 'run': os.path.abspath(run_dir), 'fps': fps, 'steps': len(frames),
                'ego_id': data['ego_id'], 'lidars': lids, 'cameras': camera_ids(frames),
                'coordinates': 'CARLA frames (x forward, y right, z up; left-handed), metres, seconds, degrees; '
                               'the ego frame at a step is the ego\'s vehicle frame then',
                'timeline': 'timeline.jsonl', 'products': products, 'skipped': skipped,
                'provenance': block(ins, (data['run'].get('provenance') or {}).get('seed'), o)}
    write_json(os.path.join(out_dir, MANIFEST), manifest)
    return manifest
