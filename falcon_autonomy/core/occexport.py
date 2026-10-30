"""Semantic-occupancy exports in the layouts occupancy-prediction code reads: Occ3D-nuScenes, OpenOcc v2 (with
per-voxel flow) and SurroundOcc (numpy; `falcon export --format occ3d --option variant=…`, core.exporters).

Formats, as verified against their public code (2026-10-30; the URLs are in docs/dataset-format.md):

* occ3d — Occ3D-nuScenes (Tian et al. 2023, NeurIPS Datasets & Benchmarks; github.com/Tsinghua-MARS-Lab/Occ3D;
  loaded by the CVPR 2023 occupancy challenge's LoadOccGTFromFile and scored by its Metric_mIoU):
  gts/<scene>/<token>/labels.npz with semantics, mask_lidar, mask_camera — uint8 (200, 200, 16) indexed [x, y, z],
  0.4 m voxels over [−40, −40, −1, 40, 40, 5.4] m in the ego frame (x forward, y LEFT, z up: nuScenes' right-handed
  axes); classes 0–16 the nuScenes-lidarseg 16-class mapping (0 'others', the devkit's ignore class) and 17 free;
  the masks 0/1 (the challenge's evaluate_miou reads them with .astype(bool)). annotations.json: train_split,
  val_split, scene_infos{scene: {token: {timestamp, camera_sensor{cam: {img_path, intrinsic, extrinsic, ego_pose}},
  ego_pose, gt_path, next, prev}}}.
* openocc — OpenOcc v2 (OpenDriveLab/OccNet, the CVPR 2024 occupancy-and-flow challenge): openocc_v2/<scene>/<token>/
  labels.npz with semantics uint8 (200, 200, 16) on the same grid, 17 classes in OpenOcc's order (car, truck,
  trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier, driveable_surface,
  other_flat, sidewalk, terrain, manmade, vegetation) and free = 16 — it has no 'others' class — and flow
  float32 (200, 200, 16, 2) (its ray metric reshapes the ground truth to [200, 200, 16, 2]; its AVE is in m/s). The
  masks are written too (the OpenOcc loader reads only semantics and flow). Its info .pkl files (mmdet3d pickles) are
  not written: annotations.json carries the same facts.
* surroundocc — SurroundOcc (Wei et al. 2023, ICCV; github.com/weiyithu/SurroundOcc): gts/<scene>__<token>.npy, a
  float32 (N, 4) array of the occupied voxels' [ix, iy, iz, label] — 0.5 m voxels over [−50, −50, −5, 50, 50, 3] m
  (200 × 200 × 16), labels 1–16 in its class order (barrier … vegetation: the devkit's coarse indices) and 0 the
  ignore class (its LoadOccupancy maps 0 to 255); free voxels are not stored. No masks (SurroundOcc has none).

Not written: OpenOccupancy (nuScenes-Occupancy: its file layout was not verified here), the OpenOcc pickles, Occ3D-
Waymo. The flow of OpenOcc is written as verified (shape, m/s); its frame is not stated by its code — Falcon writes
the voxel's actor velocity in the world, expressed in the ego frame of the sample (x forward, y left), as nuScenes
box velocities are; 0 for the static scene.

How a sample is made (one per lidar sweep, every n-th), in Falcon's ego frame at the sweep's step (core.worldmodel):
the labelled lidar points (and, with aggregate=k, the static points of the k sweeps either side, moved into this
frame — Occ3D's multi-frame densification, for the static scene only) vote the class of each voxel (majority, ties to
the lower class); every actor's 3D box (actors.jsonl) fills its voxels with its class by its nuScenes category,
overriding the points (two-wheelers whole, rider included); the ego is left out. mask_lidar: a ray from each sweep's
sensor origin to each of its points (Amanatides & Woo traversal) — the voxels it passes and the one it ends in
(Occ3D Section 3.3.2). mask_camera: from every pinhole camera's centre a ray through the pixel grid (sampled so
neighbouring rays are at most camera_ray_step × the voxel size apart at the grid's far corner) to the grid's edge,
stopping at the first occupied voxel: the voxels it passes and that first occupied one are observed, everything
behind is not (Occ3D: "the first occupied voxel on each ray is observed"). Then y is mirrored (Falcon's CARLA ego
frame is y right; the formats' is y left): index iy ↦ ny − 1 − iy.

Class mapping (Falcon → Occ3D / devkit coarse, via each tag's nuScenes-lidarseg category, core.groundtruth): road and
road lines → driveable_surface; sidewalk → sidewalk; terrain → terrain; ground, water, rail track → other_flat;
building, wall, fence, pole, traffic light and sign, static, bridge, guard rail → manmade; vegetation → vegetation;
the boxes' categories: vehicle.car (autorickshaw and cycle-rickshaw stand-ins too) → car, truck → truck, bus → bus,
motorcycle → motorcycle, bicycle (with rider) → bicycle, human.pedestrian.* → pedestrian; ambulance and police cars,
animals (the cattle proxy), pushable objects (handcart, vendor proxies), train, CARLA's 'dynamic' and 'other' → the
devkit's ignore class: Occ3D 'others', SurroundOcc 0 (ignored), OpenOcc FREE (it has no such class — counted in the
manifest). Unrepresentable: CARLA has no construction vehicle, trailer, traffic cone or temporary barrier actor
(those classes never occur), and no tarpaulin, debris or bicycle rack. Lidar points of CARLA's 'unlabelled', sky
and Falcon's weather noise vote for nothing (they are not objects), but their rays count for mask_lidar.

Limits: one sweep (or k static ones) is far sparser than Occ3D's whole-sequence aggregation and mesh densification;
free space is known only along rays that hit something (CARLA returns nothing for a miss); a moving actor's points
lie where it was during its slice of the sweep, its box where it is at the step; the vehicle origin is CARLA's
(the box centre at ground height), not nuScenes' rear axle.
"""
from __future__ import annotations

import io
import math
import os

import numpy as np

from . import groundtruth as GT
from . import worldmodel as WM

# the nuScenes-lidarseg 16-class mapping (nuscenes-devkit, eval/lidarseg/utils.py LidarsegClassMapper.get_fine2coarse)
COARSE = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
          'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
          'vegetation')
FINE2COARSE = {
    'noise': 'others', 'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'others', 'human.pedestrian.stroller': 'others',
    'human.pedestrian.personal_mobility': 'others', 'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian', 'animal': 'others', 'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle', 'vehicle.bicycle': 'bicycle', 'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus', 'vehicle.truck': 'truck', 'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'others', 'vehicle.emergency.police': 'others', 'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier', 'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'others', 'movable_object.debris': 'others',
    'static_object.bicycle_rack': 'others', 'flat.driveable_surface': 'driveable_surface', 'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain', 'flat.other': 'other_flat', 'static.manmade': 'manmade',
    'static.vegetation': 'vegetation', 'static.other': 'others', 'vehicle.ego': 'others'}
OPENOCC_CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                   'traffic_cone', 'barrier', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                   'vegetation', 'free')
NO_VOTE_TAGS = (0, 11, GT.WEATHER_NOISE_CLASS)      # unlabelled, sky, weather noise: not objects
UNREPRESENTABLE = ('construction_vehicle', 'trailer', 'traffic_cone', 'barrier')
VARIANTS = {
    'occ3d': {'range': (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4), 'res': 0.4, 'free': 17,
              'classes': COARSE + ('free',), 'root': 'gts', 'about': 'Occ3D-nuScenes'},
    'openocc': {'range': (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4), 'res': 0.4, 'free': 16,
                'classes': OPENOCC_CLASSES, 'root': 'openocc_v2', 'flow': True, 'about': 'OpenOcc v2'},
    'surroundocc': {'range': (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0), 'res': 0.5, 'free': None,
                    'classes': ('ignore',) + COARSE[1:], 'root': 'gts', 'sparse': True, 'about': 'SurroundOcc'},
}
CHANNELS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT')
CARLA_TO_OPTICAL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def coarse_of_category(cat) -> int:
    """The devkit's coarse index (0 'others' … 16) of a nuScenes category."""
    return COARSE.index(FINE2COARSE.get(str(cat or ''), 'others'))


def _tag_lut() -> np.ndarray:
    """CARLA tag (as written in Falcon's lidar labels) → coarse index; −1: votes for nothing."""
    lut = np.full(256, -1, np.int16)
    for c in GT.CLASSES:
        if c[0] in NO_VOTE_TAGS or c[0] == GT.EGO_CLASS:
            continue
        lut[c[0]] = COARSE.index(FINE2COARSE.get(c[3], 'others'))
    return lut


TAG_LUT = _tag_lut()


def target_lut(variant: str) -> np.ndarray:
    """Coarse index → the variant's label (OpenOcc: 'others' becomes FREE; SurroundOcc: the same indices, 0
    ignore)."""
    if variant == 'openocc':
        return np.array([OPENOCC_CLASSES.index(c) if c in OPENOCC_CLASSES else 16 for c in COARSE], np.uint8)
    return np.arange(len(COARSE), dtype=np.uint8)


def grid_of(variant: str) -> WM.Grid:
    v = VARIANTS[variant]
    r = v['range']
    return WM.Grid(v['res'], (r[0], r[3]), (r[1], r[4]), (r[2], r[5]))


# ================================================================================================ ray casting
def cast_to_first_occupied(grid: WM.Grid, occupied: np.ndarray, origin, dirs) -> np.ndarray:
    """The voxels rays from origin along dirs (N, 3) pass through, up to and including the first occupied one,
    until they leave the grid (an exact voxel traversal, Amanatides & Woo 1987). The origin may lie outside the grid
    (the ray is advanced to where it enters). Returns a bool grid."""
    seen = np.zeros(grid.shape, bool)
    shape = np.asarray(grid.shape)
    d = np.asarray(dirs, float).reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    o = (np.asarray(origin, float) - grid.lo) / grid.res                 # in voxel units
    # enter the box [0, shape]: slab intersection
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (0.0 - o) / d
        t2 = (shape - o) / d
    tmin = np.nanmax(np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2)), axis=1)
    tmax = np.nanmin(np.where(np.isnan(t1), np.inf, np.maximum(t1, t2)), axis=1)
    keep = (tmax > np.maximum(tmin, 0.0))
    d, tmin = d[keep], np.maximum(tmin[keep], 0.0)
    if not len(d):
        return seen
    p = o + d * (tmin[:, None] + 1e-9)
    cur = np.clip(np.floor(p).astype(np.int64), 0, shape - 1)
    step = np.where(d > 0, 1, -1).astype(np.int64)
    ad = np.abs(d)
    with np.errstate(divide='ignore', invalid='ignore'):
        tdelta = np.where(ad > 0, 1.0 / ad, np.inf)
        nxt = np.where(step > 0, cur + 1.0 - p, p - cur)
        tnext = np.where(ad > 0, nxt / ad, np.inf)
    act = np.arange(len(d))
    while len(act):
        c = cur[act]
        seen[c[:, 0], c[:, 1], c[:, 2]] = True
        stop = occupied[c[:, 0], c[:, 1], c[:, 2]]
        act, c = act[~stop], c[~stop]
        if not len(act):
            break
        tn = tnext[act]
        ax = np.argmin(tn, axis=1)
        cur[act, ax] += step[act, ax]
        tnext[act, ax] += tdelta[act, ax]
        c = cur[act]
        inside = np.all((c >= 0) & (c < shape), axis=1)
        act = act[inside]
    return seen


def camera_rays(K, W: int, H: int, R_vehicle_cam, angle_step: float) -> np.ndarray:
    """Ray directions (N, 3) in the vehicle frame (CARLA axes) through a pinhole camera's pixel grid, sampled every
    max(1, ⌊angle_step · f⌋) pixels (f the focal length in pixels: the angle between neighbouring rays at the image
    centre ≤ angle_step), pixel centres in Falcon's pixel-edge convention."""
    K = np.asarray(K, float)
    f = min(K[0, 0], K[1, 1])
    s = max(1, int(math.floor(angle_step * f)))
    us = np.arange(0.5, W, s)
    vs = np.arange(0.5, H, s)
    uu, vv = np.meshgrid(us, vs)
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 0)
    d_opt = np.linalg.inv(K) @ pix                                   # optical: x right, y down, z forward
    d_cam = CARLA_TO_OPTICAL.T @ d_opt                                # CARLA camera: x forward, y right, z up
    return (np.asarray(R_vehicle_cam, float) @ d_cam).T


# ================================================================================================ one sample
def sample_grids(data: dict, lid: str, sweeps: list, k: int, variant: str, cameras: list, aggregate: int = 0,
                 camera_ray_step: float = 1.0) -> dict:
    """The grids of the k-th sweep (core.worldmodel.lidar_sweeps rows) in Falcon's ego frame (y right): coarse
    class per voxel (−1 free), mask_lidar, mask_camera, flow (vx, vy in the ego frame, CARLA axes) and counts.
    cameras: [(id, K, W, H, 4 × 4 vehicle ← camera)]."""
    grid = grid_of(variant)
    _n, step, _t, _rel, _M = sweeps[k]
    frame = next(f for f in data['frames'] if f['step'] == step)
    T_we = WM.pose_matrix(frame['ego'])
    T_ew = np.linalg.inv(T_we)
    ego = data.get('ego_id')
    actors_now = (data.get('actors') or {}).get(step, [])
    votes_p, votes_c = [], []
    lidar_seen = np.zeros(grid.shape, bool)
    no_origin = 0
    for j in range(max(k - aggregate, 0), min(k + aggregate, len(sweeps) - 1) + 1):
        _nj, sj, _tj, relj, Mj = sweeps[j]
        xyz, tag, ids = WM.read_sweep(data['dir'], relj)
        T = T_ew @ Mj                                           # this sample's ego ← sweep j's sensor
        P = xyz @ T[:3, :3].T + T[:3, 3]
        keep = np.ones(len(P), bool)
        if ego is not None:
            keep &= ids != np.uint32(ego)
        if j != k:                                              # neighbours: the static scene only
            movers = {a['id'] for a in (data.get('actors') or {}).get(sj, [])}
            keep &= ~np.isin(ids, np.fromiter(movers, np.uint32, len(movers))) if movers else keep
        P, tag = P[keep], tag[keep]
        try:
            free, hit = WM.traverse(grid, T[:3, 3], P)
            lidar_seen |= free | hit
        except ValueError:                                      # the sensor outside the grid (a tall mount)
            no_origin += 1
        cls = TAG_LUT[tag]
        ok = cls >= 0
        votes_p.append(P[ok])
        votes_c.append(cls[ok])
    P = np.concatenate(votes_p) if votes_p else np.zeros((0, 3))
    C = np.concatenate(votes_c) if votes_c else np.zeros(0, np.int16)
    sem = np.full(grid.shape, -1, np.int16)
    ijk = grid.index(P)
    ok = grid.inside(ijk)
    if ok.any():
        flat = np.ravel_multi_index(tuple(ijk[ok].T), grid.shape)
        key = flat * 32 + C[ok].astype(np.int64)
        u, cnt = np.unique(key, return_counts=True)
        vox, c = u // 32, u % 32
        order = np.lexsort((c, -cnt, vox))
        vox, c = vox[order], c[order]
        first = np.r_[True, vox[1:] != vox[:-1]]
        sem.ravel()[vox[first]] = c[first]
    flow = np.zeros(grid.shape + (2,), np.float32)
    R_ew = T_ew[:3, :3]
    n_boxes = 0
    for a in actors_now:
        if a['id'] == ego:
            continue
        c, R, ext = WM.actor_box(a)
        box = {'center': R_ew @ c + T_ew[:3, 3], 'R': R_ew @ R, 'extent': ext, 'class': 0}
        m, _ = WM.fill_boxes(grid, [box])
        if not m.any():
            continue
        n_boxes += 1
        sem[m] = coarse_of_category(a.get('category'))
        v = R_ew @ np.array([a.get('vx', 0.0), a.get('vy', 0.0), a.get('vz', 0.0)], float)
        flow[m] = v[:2].astype(np.float32)
    occupied = sem >= 0
    cam_seen = np.zeros(grid.shape, bool)
    angle = camera_ray_step * grid.res / math.hypot(*(np.abs([grid.lo[0], grid.lo[1]]) + 0.0))
    for cid, K, Wpx, Hpx, T_vc in cameras:
        dirs = camera_rays(K, Wpx, Hpx, T_vc[:3, :3], angle)
        cam_seen |= cast_to_first_occupied(grid, occupied, T_vc[:3, 3], dirs)
    return {'grid': grid, 'coarse': sem, 'mask_lidar': lidar_seen, 'mask_camera': cam_seen, 'flow': flow,
            'boxes': n_boxes, 'no_origin': no_origin, 'step': step, 'frame': frame}


def to_format(g: dict, variant: str) -> dict:
    """The variant's arrays from sample_grids()' (y mirrored into the formats' y-left frame)."""
    v = VARIANTS[variant]
    lut = target_lut(variant)
    coarse = g['coarse'][:, ::-1, :]
    occ = coarse >= 0
    if v.get('sparse'):
        idx = np.argwhere(occ)
        lab = lut[coarse[occ]]
        return {'occ': np.concatenate([idx, lab[:, None]], 1).astype(np.float32)}
    sem = np.full(coarse.shape, v['free'], np.uint8)
    sem[occ] = lut[coarse[occ]]
    out = {'semantics': sem, 'mask_lidar': g['mask_lidar'][:, ::-1, :].astype(np.uint8),
           'mask_camera': g['mask_camera'][:, ::-1, :].astype(np.uint8)}
    if v.get('flow'):
        fl = g['flow'][:, ::-1, :, :].copy()
        fl[..., 1] *= -1.0                                         # y left
        out['flow'] = fl.astype(np.float32)
    return out


# ================================================================================================ the exporter
def export_occ(ctx, Run, helpers) -> None:
    """core.exporters' entry for format 'occ3d' (options: variant occ3d | openocc | surroundocc, lidar, every,
    aggregate, camera_ray_step, cameras)."""
    rh, quat_of, camera_T, K_opencv, token = (helpers[k] for k in ('rh', 'quat_of', 'camera_T', 'K_opencv', 'token'))
    variant = ctx.opt.get('variant', 'occ3d')
    if variant not in VARIANTS:
        raise helpers['ExportError'](f'variant must be one of {", ".join(VARIANTS)}, not {variant!r}')
    V = VARIANTS[variant]
    every = max(int(ctx.opt.get('every', 1)), 1)
    aggregate = max(int(ctx.opt.get('aggregate', 0)), 0)
    ray_step = float(ctx.opt.get('camera_ray_step', 1.0))
    ann = {'train_split': [], 'val_split': [], 'test_split': [], 'scene_infos': {}}
    counts = np.zeros(len(V['classes']), np.int64)
    masked = {'lidar': 0, 'camera': 0, 'voxels': 0}
    others_as_free = 0
    for split, runs in ctx.splits:
        for run in runs:
            data = WM.load_run(run.dir)
            lids = WM.lidar_ids(data)
            lid = ctx.opt.get('lidar') or (lids[0] if lids else None)
            if lid is None or lid not in lids:
                ctx.leave(f'{run.key}', 'no labelled lidar sweeps: occupancy is derived from the lidar')
                ctx.tick(run.key)
                continue
            if data.get('actors') is None:
                ctx.leave(f'{run.key}: actor boxes', 'no actors.jsonl: occupancy from the lidar points only')
            want = ctx.opt.get('cameras')
            cams = []
            for cid, c in run.cams.items():
                if c.get('type') != 'camera.pinhole' or (want and cid not in want):
                    if c.get('type') != 'camera.pinhole':
                        ctx.leave(f'{run.key}/{cid}', 'not a pinhole camera: not in mask_camera')
                    continue
                cams.append((cid, c['K'], int(c['width']), int(c['height']), run.mount(cid, actual=True)))
            if not cams and variant != 'surroundocc':
                ctx.leave(f'{run.key}: mask_camera', 'no pinhole camera in the rig: mask_camera is all 0')
            sweeps = WM.lidar_sweeps(data, lid)
            name = {None: 'train_split', 'train': 'train_split', 'val': 'val_split', 'test': 'test_split'}[split]
            ann[name].append(run.key)
            infos = ann['scene_infos'].setdefault(run.key, {})
            toks = [token(run.key, s[1]) for s in sweeps]
            written = 0
            for k in range(0, len(sweeps), every):
                g = sample_grids(data, lid, sweeps, k, variant, cams, aggregate, ray_step)
                arrays = to_format(g, variant)
                tok = toks[k]
                if V.get('sparse'):
                    rel = f'{V["root"]}/{run.key}__{tok}.npy'
                    buf = io.BytesIO()
                    np.save(buf, arrays['occ'])
                    ctx.write(rel, buf.getvalue())
                    lab = arrays['occ'][:, 3].astype(np.int64)
                    counts += np.bincount(lab, minlength=len(counts))[:len(counts)]
                else:
                    rel = f'{V["root"]}/{run.key}/{tok}/labels.npz'
                    buf = io.BytesIO()
                    np.savez_compressed(buf, **arrays)
                    ctx.write(rel, buf.getvalue())
                    counts += np.bincount(arrays['semantics'].ravel(), minlength=len(counts))[:len(counts)]
                    masked['lidar'] += int(arrays['mask_lidar'].sum())
                    masked['camera'] += int(arrays['mask_camera'].sum())
                    masked['voxels'] += int(arrays['semantics'].size)
                if variant == 'openocc':
                    others_as_free += int((g['coarse'] == 0).sum())
                if g['no_origin']:
                    ctx.leave(f'{run.key}: rays of {lid}', 'the sensor lies outside the grid: those sweeps add no '
                                                           'free space to mask_lidar')
                rec = g['frame']
                T_ge = rh(run.ego(rec))
                cam_info = {}
                for i, (cid, K, Wpx, Hpx, _T) in enumerate(cams):
                    img = run.own(rec, cid, 'rgb') or run.file(rec, cid, 'rgb')
                    ch = CHANNELS[i] if i < len(CHANNELS) else f'CAM_{cid.upper()}'
                    path = None
                    if img:
                        path = f'samples/{ch}/{run.key}__{rec["step"]:06d}{os.path.splitext(img)[1]}'
                        ctx.place(run.p(img), path)
                    Tc = camera_T(run, cid, actual=True)
                    cam_info[ch] = {'img_path': path, 'intrinsic': K_opencv(K).tolist(),
                                    'extrinsic': {'translation': Tc[:3, 3].tolist(),
                                                  'rotation': list(quat_of(Tc[:3, :3]))},
                                    'ego_pose': {'translation': T_ge[:3, 3].tolist(),
                                                 'rotation': list(quat_of(T_ge[:3, :3]))}, 'falcon_camera': cid}
                infos[tok] = {'timestamp': str(run.us(rec)), 'camera_sensor': cam_info,
                              'ego_pose': {'translation': T_ge[:3, 3].tolist(), 'rotation': list(quat_of(T_ge[:3, :3]))},
                              'gt_path': rel, 'step': rec['step'], 'sim_t': rec['sim_t'],
                              'prev': toks[k - every] if k - every >= 0 else '',
                              'next': toks[k + every] if k + every < len(sweeps) else ''}
                written += 1
                ctx.count('samples')
            ctx.frames[run.key] = written
            ctx.tick(f'{run.key}: {written} occupancy samples')
    ctx.json('annotations.json', ann, indent=1)
    names = list(V['classes'])
    ctx.counts['voxels_per_class'] = {names[i]: int(counts[i]) for i in range(len(names)) if counts[i]}
    if masked['voxels']:
        ctx.counts['observed_share'] = {'lidar': round(masked['lidar'] / masked['voxels'], 5),
                                        'camera': round(masked['camera'] / masked['voxels'], 5)}
    if others_as_free:
        ctx.leave('OpenOcc: the "others" class', f'OpenOcc v2 has no general-object class: {others_as_free} voxels of '
                                                 'animals, pushable objects, emergency vehicles, trains … are written '
                                                 'as free (the occ3d variant keeps them as "others")')
    for c in UNREPRESENTABLE:
        ctx.leave(f'class {c}', 'CARLA has no actor of this kind: the class never occurs')
    ctx.heur('mask_camera ray density', f'one ray per ⌊{ray_step:g} · voxel / far-corner distance · f⌋ pixels '
                                        '(a far voxel narrower than that angle can fall between rays)')
    if aggregate:
        ctx.heur('aggregation', f'the static points of {aggregate} sweep(s) on either side, moved into each sample\'s '
                                'ego frame (actors\' points from the sample\'s own sweep only)')
    ctx.frames_doc = {'ego': f'{V["about"]}: x forward, y left, z up (right-handed); origin Falcon\'s vehicle origin '
                             '(CARLA\'s: the box centre at ground height), not nuScenes\' rear axle',
                      'grid': {'range_m': list(V['range']), 'voxel_m': V['res'], 'shape': list(grid_of(variant).shape),
                               'index': '[ix, iy, iz] from (x_min, y_min, z_min)'},
                      'classes': names, 'free': V['free']}
    ctx.load = {
        'occ3d': "np.load('gts/<scene>/<token>/labels.npz'): semantics, mask_lidar, mask_camera — uint8 (200, 200, 16)"
                 ", masks via .astype(bool) (the CVPR 2023 occupancy challenge's loader); annotations.json as Occ3D's",
        'openocc': "np.load('openocc_v2/<scene>/<token>/labels.npz'): semantics uint8 (200, 200, 16), flow float32 "
                   "(200, 200, 16, 2) m/s; annotations.json in place of the OpenOcc pickles",
        'surroundocc': "np.load('gts/<scene>__<token>.npy'): float32 (N, 4) [ix, iy, iz, label] (SurroundOcc's "
                       "LoadOccupancy); annotations.json gives each sample's gt_path"}[variant]
