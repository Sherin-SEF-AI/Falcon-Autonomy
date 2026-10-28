"""Lidar recording with point-for-point labels, and the checks that prove them (runs in CARLA's Python; numpy).

Every lidar of the rig is spawned twice on the same step with the same attributes: sensor.lidar.ray_cast (the
recorded points, with its range noise) and sensor.lidar.ray_cast_semantic (no noise; per point x y z cos_inc
object_idx object_tag, 24 bytes). Measured on CARLA 0.9.16 (Vantage's bridge, Town10HD): the two cast the same rays
in the same order — the same points per channel, directions within 3·10⁻⁶° — so each recorded point takes the
semantic point at its index. This is checked on every step, not assumed (the proof, in run.json "lidar"): the points
per channel must be equal and every direction within 0.01°; the range difference must look like the ray-cast lidar's
noise (its RMS is reported against the noise set). A step that fails the check falls back to matching per channel by
azimuth (within half a column) and is counted as such.

A lidar delivers a slice of its rotation every simulation step. Slices are collected into sweeps of one rotation
(1 / rotation_frequency); each slice's points are moved from the sensor's pose at their step into its pose at the
sweep's end through the vehicle's poses (motion compensation), so a sweep is one point cloud in one frame.

Files (falcon-raw): lidar/<sensor>/<sweep>.bin — float32 x y z intensity (CARLA's sensor frame: x forward, y right,
z up) — and lidar/<sensor>/<sweep>.labels.bin — uint8 tag, then uint32 object id (the CARLA actor id; 0 for map
meshes), stored as two consecutive arrays of the sweep's length.

Checks (run.json "lidar_checks"):
* box_label_share — of the points inside a vehicle's 3D box, the share labelled a vehicle class (car, truck, bus,
  train, motorcycle, bicycle, rider). A lidar return lies ON the surface it hit, so the box is grown by BOX_MARGIN_M
  outward (sides and top), and its bottom raised by BOX_GROUND_M so that ground returns under the vehicle are not
  counted as its own. The acceptance asks ≥ 99 %.
* range_vs_camera — for a camera with a depth label camera on the same step: each lidar point seen by it, its range
  from the camera against the camera's range along the same ray (planar depth × |ray|, inverse depth interpolated
  within one surface only); the median absolute difference. The acceptance asks ≤ 2 cm. The two sensors stand
  apart, so a point whose camera range is more than OCCLUDED_M shorter is hidden from the camera by a nearer surface
  and is counted apart (hidden_from_camera), not compared.
"""
from __future__ import annotations

import math
import os
import queue
from types import SimpleNamespace

import numpy as np

from ..core.labels import VEHICLE_TAGS
from .live import transform_matrix

DIR_TOL_DEG = 0.01
BOX_MARGIN_M = 0.05
BOX_GROUND_M = 0.10
OCCLUDED_M = 0.5
SEMANTIC_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'), ('idx', '<u4'),
                           ('tag', '<u4')])
LIDAR_ATTRS = ('channels', 'range', 'points_per_second', 'rotation_frequency', 'upper_fov', 'lower_fov',
               'horizontal_fov')
VEHICLE_OR_RIDER = np.array(sorted(set(VEHICLE_TAGS) | {13}), np.int64)


def mount_matrix(p: dict) -> np.ndarray:
    """A sensor's 4 × 4 pose in the vehicle frame (CARLA's Transform::GetMatrix of its rig pose)."""
    return transform_matrix(SimpleNamespace(x=p['x'], y=p['y'], z=p['z']),
                            SimpleNamespace(roll=p['roll'], pitch=p['pitch'], yaw=p['yaw']))


def lidar_blueprints(lib, s: dict, role: str):
    """(ray-cast, semantic) blueprints with the rig's attributes, every step (sensor_tick 0)."""
    out = []
    for bp_id in ('sensor.lidar.ray_cast', 'sensor.lidar.ray_cast_semantic'):
        bp = lib.find(bp_id)
        for k in LIDAR_ATTRS:
            if k in s and bp.has_attribute(k):
                bp.set_attribute(k, str(s[k]))
        if bp_id.endswith('ray_cast'):
            if bp.has_attribute('noise_stddev'):
                bp.set_attribute('noise_stddev', str(s.get('noise_stddev', 0.0)))
            for k in ('dropoff_general_rate', 'dropoff_intensity_limit', 'dropoff_zero_intensity'):
                if bp.has_attribute(k):
                    bp.set_attribute(k, '0.0' if k != 'dropoff_general_rate' else str(s.get(k, 0.0)))
        bp.set_attribute('sensor_tick', '0.0')
        bp.set_attribute('role_name', role)
        out.append(bp)
    return out


def counts_of(meas, nch) -> np.ndarray:
    return np.array([meas.get_point_count(ch) for ch in range(nch)], np.int64)


def match(raw, raw_counts, sem, sem_counts, columns: int, hfov: float):
    """(tag, idx, exact: bool, stats) — labels for every recorded point from the semantic slice."""
    P = raw[:, :3].astype(np.float64)
    S = np.frombuffer(sem.raw_data, SEMANTIC_DTYPE)
    Q = np.c_[S['x'], S['y'], S['z']].astype(np.float64) if len(S) else np.zeros((0, 3))
    stats = {'points': int(len(P))}
    if np.array_equal(raw_counts, sem_counts) and len(Q) == len(P):
        rp, rq = np.linalg.norm(P, axis=1), np.linalg.norm(Q, axis=1)
        if len(P):
            cosang = np.sum(P * Q, axis=1) / np.maximum(rp * rq, 1e-9)
            err = float(np.degrees(np.arccos(np.clip(cosang.min(), -1.0, 1.0))))
        else:
            err = 0.0
        if err <= DIR_TOL_DEG:
            stats.update(max_dir_err_deg=err, range_diff_sq=float(np.sum((rp - rq) ** 2)), range_n=int(len(P)))
            return S['tag'].astype(np.uint8), S['idx'].astype(np.uint32), True, stats
    stats['count_mismatch'] = int(abs(int(sem_counts.sum()) - int(raw_counts.sum())))
    tag = np.zeros(len(P), np.uint8)
    idx = np.zeros(len(P), np.uint32)
    half = hfov / max(columns, 1) / 2.0
    a0 = np.r_[0, np.cumsum(raw_counts)]
    b0 = np.r_[0, np.cumsum(sem_counts)]
    for ch in range(len(raw_counts)):
        p_, q_ = P[a0[ch]:a0[ch + 1]], Q[b0[ch]:b0[ch + 1]]
        if not len(p_) or not len(q_):
            continue
        ap = np.degrees(np.arctan2(p_[:, 1], p_[:, 0]))
        aq = np.degrees(np.arctan2(q_[:, 1], q_[:, 0]))
        o = np.argsort(aq)
        j = np.clip(np.searchsorted(aq[o], ap), 1, max(len(o) - 1, 1)) if len(o) > 1 else np.zeros(len(ap), int)
        cand = np.stack([o[np.maximum(j - 1, 0)], o[np.minimum(j, len(o) - 1)]], 1)
        d = np.abs(((aq[cand] - ap[:, None]) + 180.0) % 360.0 - 180.0)
        best = cand[np.arange(len(ap)), np.argmin(d, 1)]
        ok = np.min(d, 1) <= half
        sl = slice(a0[ch], a0[ch + 1])
        tag[sl] = np.where(ok, S['tag'][b0[ch]:b0[ch + 1]][best], 0)
        idx[sl] = np.where(ok, S['idx'][b0[ch]:b0[ch + 1]][best], 0)
    return tag, idx, False, stats


class LidarRecorder:
    """One rig lidar: its two sensors, the sweep being collected, the proof statistics and the checks."""

    def __init__(self, w, world, lib, ego, s: dict, cameras, role, raster=None):
        self.w, self.s, self.ego = w, s, ego
        from .scene import camera_transform
        tf = camera_transform(w.carla, s['pose'])
        bp_raw, bp_sem = lidar_blueprints(lib, s, role)
        self.raw = w.own(world.spawn_actor(bp_raw, tf, attach_to=ego))
        self.sem = w.own(world.spawn_actor(bp_sem, tf, attach_to=ego))
        for a in (self.raw, self.sem):
            q = queue.Queue()
            cameras.queues[a.id] = q
            a.listen(q.put)
        self.cameras = cameras
        self.raster = raster                # an imported world's lanes: its mesh's points come tagged 0
        self.ground_relabelled = 0
        self.physics = None                 # core.sensorphysics.Session when the run asks for weather physics
        self.nch = int(s['channels'])
        self.steps_per_sweep = 1
        self.mount = mount_matrix(s['pose'])
        self.slices = []                 # (points xyz float64, intensity, tag, idx, M_world_sensor at that step)
        self.n = 0
        self.stats = {'slices': 0, 'exact_slices': 0, 'fallback_slices': 0, 'points': 0, 'max_dir_err_deg': 0.0,
                      'range_diff_sq': 0.0, 'range_n': 0, 'count_mismatch': 0}
        self.box_in = 0
        self.box_vehicle = 0
        self.range_diffs = []
        self.range_hidden = 0

    def set_rate(self, fps: float):
        """Steps per rotation at this simulation rate (a sweep is that many slices)."""
        self.steps_per_sweep = max(int(round(fps / float(self.s['rotation_frequency']))), 1)

    def step(self, frame, sim_t, out_dir, rec):
        """Collect this step's slice; write a sweep when a rotation is complete. Adds the files to rec['files']."""
        cams = self.cameras
        m_raw = cams.get(self.raw.id, frame)
        m_sem = cams.get(self.sem.id, frame)
        raw = np.frombuffer(m_raw.raw_data, np.float32).reshape(-1, 4)
        rc, sc = counts_of(m_raw, self.nch), counts_of(m_sem, self.nch)
        cols = int(round(self.s['points_per_second'] / max(self.nch, 1) / max(self.s['rotation_frequency'], 1e-6)))
        tag, idx, exact, st = match(raw, rc, m_sem, sc, cols, float(self.s.get('horizontal_fov', 360.0)))
        S = self.stats
        S['slices'] += 1
        S['points'] += st['points']
        if exact:
            S['exact_slices'] += 1
            S['max_dir_err_deg'] = max(S['max_dir_err_deg'], st['max_dir_err_deg'])
            S['range_diff_sq'] += st['range_diff_sq']
            S['range_n'] += st['range_n']
        else:
            S['fallback_slices'] += 1
            S['count_mismatch'] += st.get('count_mismatch', 0)
        et = self.ego.get_transform()
        M_ws = transform_matrix(et.location, et.rotation) @ self.mount
        if self.raster is not None:
            # measured: a generated OpenDRIVE world's road mesh is hit by the semantic lidar with tag 0 and id 0;
            # such points within 0.3 m of the map's ground plane take the class of its lanes there
            Pw = raw[:, :3].astype(np.float64) @ M_ws[:3, :3].T + M_ws[:3, 3]
            g = (tag == 0) & (idx == 0) & (np.abs(Pw[:, 2] - self.raster.z0) < 0.3)
            if g.any():
                tag = tag.copy()
                tag[g] = self.raster.lookup(Pw[g, :2])
                self.ground_relabelled += int(g.sum())
        self.box_check(raw[:, :3].astype(np.float64), tag, idx, M_ws)
        xyz, inten = raw[:, :3].astype(np.float64), raw[:, 3].copy()
        if self.physics is not None:                     # core.sensorphysics, on the slice in the sensor's frame
            xyz, inten, tag, idx, st = self.physics.lidar(self.s, xyz, inten, tag, idx)
            if st is not None:
                eff = [k for k, n in (('lidar_dropout', st['dropped']), ('lidar_clutter', st['clutter'])) if n]
                if self.physics.cond['alpha_lidar_per_m'] > 0:
                    eff.insert(0, 'lidar_attenuation')
                if eff:
                    rec.setdefault('physics', {})[self.s['id']] = eff
        self.slices.append((xyz, inten, tag, idx, M_ws))
        if len(self.slices) >= self.steps_per_sweep:
            self.write_sweep(out_dir, rec)
        return m_raw

    def write_sweep(self, out_dir, rec):
        M_end = self.slices[-1][4]
        inv = np.linalg.inv(M_end)
        pts, its, tags, ids = [], [], [], []
        for P, I, T, D, M in self.slices:
            A = inv @ M                                          # this slice's sensor frame into the sweep's
            pts.append(P @ A[:3, :3].T + A[:3, 3])
            its.append(I)
            tags.append(T)
            ids.append(D)
        xyz = np.concatenate(pts) if pts else np.zeros((0, 3))
        rel = os.path.join('lidar', self.s['id'], f'{self.n:06d}.bin')
        path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.c_[xyz, np.concatenate(its) if its else np.zeros(0)].astype(np.float32).tofile(path)
        with open(path[:-4] + '.labels.bin', 'wb') as fh:
            fh.write((np.concatenate(tags) if tags else np.zeros(0, np.uint8)).astype(np.uint8).tobytes())
            fh.write((np.concatenate(ids) if ids else np.zeros(0, np.uint32)).astype(np.uint32).tobytes())
        rec['files'][f'{self.s["id"]}/lidar'] = rel
        rec.setdefault('lidar_pose', {})[self.s['id']] = [round(float(v), 6) for v in M_end[:3, :].ravel()]
        self.slices = []
        self.n += 1

    def box_check(self, P, tag, idx, M_ws):
        """Points inside a vehicle's 3D box (see the module doc): how many are labelled a vehicle class."""
        if not len(P):
            return
        Pw = P @ M_ws[:3, :3].T + M_ws[:3, 3]
        for a in self.w.world.get_actors().filter('vehicle.*'):
            if a.id == self.ego.id:
                continue
            try:
                t = a.get_transform()
                bb = a.bounding_box
            except RuntimeError:
                continue
            M = transform_matrix(t.location, t.rotation)
            local = (Pw - M[:3, 3]) @ M[:3, :3]
            c = np.array([bb.location.x, bb.location.y, bb.location.z])
            e = np.array([bb.extent.x, bb.extent.y, bb.extent.z]) + BOX_MARGIN_M
            d = local - c
            inside = (np.abs(d[:, 0]) <= e[0]) & (np.abs(d[:, 1]) <= e[1]) & (d[:, 2] <= e[2]) & \
                (d[:, 2] >= -bb.extent.z + BOX_GROUND_M)
            if inside.any():
                self.box_in += int(inside.sum())
                self.box_vehicle += int(np.isin(tag[inside].astype(np.int64), VEHICLE_OR_RIDER).sum())

    def range_check(self, raw_xyz, cam_sensor: dict, depth_img, M_cam_from_lidar):
        """Lidar ranges against a camera's depth along the same rays (see the module doc)."""
        P = raw_xyz @ M_cam_from_lidar[:3, :3].T + M_cam_from_lidar[:3, 3]       # camera frame: x fwd, y right, z up
        W, H, fov = cam_sensor['width'], cam_sensor['height'], cam_sensor['fov']
        f = W / (2.0 * math.tan(math.radians(fov) / 2.0))
        front = P[:, 0] > 0.5
        P = P[front]
        u = f * P[:, 1] / P[:, 0] + W / 2.0
        v = -f * P[:, 2] / P[:, 0] + H / 2.0
        ok = (u >= 0.5) & (u < W - 1.5) & (v >= 0.5) & (v < H - 1.5)
        P, u, v = P[ok], u[ok], v[ok]
        if not len(P):
            return
        bgra = np.frombuffer(depth_img.raw_data, np.uint8).reshape(H, W, 4).astype(np.float64)
        depth = 1000.0 * (bgra[:, :, 2] + 256.0 * bgra[:, :, 1] + 65536.0 * bgra[:, :, 0]) / (2 ** 24 - 1)
        # pixel centres at integer + 0.5: interpolate inverse depth within one surface (all four within 2 %)
        x0, y0 = np.floor(u - 0.5).astype(int), np.floor(v - 0.5).astype(int)
        fx, fy = u - 0.5 - x0, v - 0.5 - y0
        d00, d10 = depth[y0, x0], depth[y0, x0 + 1]
        d01, d11 = depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]
        dd = np.stack([d00, d10, d01, d11], 1)
        same = (dd.max(1) - dd.min(1)) <= 0.02 * dd.min(1)
        inv = (1 - fx) * (1 - fy) / d00 + fx * (1 - fy) / d10 + (1 - fx) * fy / d01 + fx * fy / d11
        z = 1.0 / inv
        ray = np.sqrt(1.0 + ((u - W / 2.0) / f) ** 2 + ((v - H / 2.0) / f) ** 2)
        r_cam = z * ray
        r_lidar = np.linalg.norm(P, axis=1)
        good = same & (z < 900.0)
        diff = r_lidar[good] - r_cam[good]
        # the camera and the lidar stand apart: a lidar hit the camera sees a nearer surface in front of is hidden
        # from the camera (parallax), not a disagreement — counted apart
        hidden = diff > OCCLUDED_M
        self.range_hidden += int(hidden.sum())
        self.range_diffs.extend(diff[~hidden].tolist()[:5000])

    def report(self) -> dict:
        S = dict(self.stats)
        rms = math.sqrt(S['range_diff_sq'] / S['range_n']) if S['range_n'] else None
        S.pop('range_diff_sq', None)
        S.pop('range_n', None)
        out = {'sensor': self.s['id'], 'sweeps': self.n,
               'match': dict(S, range_diff_rms_m=rms, noise_set_m=float(self.s.get('noise_stddev', 0.0)),
                             method='by order when the points per channel are equal and every direction is within '
                                    f'{DIR_TOL_DEG}°; else per channel by azimuth (counted as fallback)'),
               'box_label_share': None if not self.box_in else round(self.box_vehicle / self.box_in, 5),
               'box_points': self.box_in, 'ground_relabelled_points': self.ground_relabelled,
               'box_rule': f'grown {BOX_MARGIN_M} m outward, bottom raised {BOX_GROUND_M} m; the ego excluded'}
        if self.range_diffs:
            d = np.asarray(self.range_diffs)
            out['range_vs_camera'] = {'points': int(len(d)), 'median_m': float(np.median(d)),
                                      'median_abs_m': float(np.median(np.abs(d))),
                                      'p90_abs_m': float(np.percentile(np.abs(d), 90)),
                                      'hidden_from_camera': self.range_hidden,
                                      'rule': f'points the camera sees a surface more than {OCCLUDED_M} m nearer than '
                                              'the lidar hit are hidden from it (parallax) and left out'}
        return out

