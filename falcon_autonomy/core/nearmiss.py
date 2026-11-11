"""Near-miss mining: the moments in a dashcam clip where another road user came critically close, measured with
core.criticality and ranked (numpy; scipy for the smoothing derivatives; OpenCV for the odometry; ffmpeg for clips).

Input: a clip's perception record (core.dashcam_ingest pass 1 — every processed frame's tracked road users with their
fused range in the vehicle frame, x forward, y left, and its σ; core.dashcam_depth) and its GPS track (core.dashcam).

Stages (mine_clip):
1. Ego motion from the GPS (ego_motion): the ~1 Hz fixes in local east-north metres (core.osm.local_projection about
   the clip's first fix), with the speed and course over ground as velocity measurements, through a constant-velocity
   Kalman filter and a Rauch–Tung–Striebel smoother (rts: σ 3 m position, 0.5 m/s velocity, 2 m/s² process
   acceleration) at every processed frame's time; the ego's acceleration and jerk from the smoothed speed by
   Savitzky–Golay derivatives (about 1 s windows, polynomial order 2 and 3; scipy.signal.savgol_filter). A frame more
   than 2.5 s from any fix has no ego state.
2. Candidates (candidates), cheaply, from the raw ranges: a track inside the ego's corridor (|lateral| < 2.5 m, ahead)
   closing with a range-rate TTC under 4 s; any track nearer than 6 m; or the ego braking harder than 3 m/s² with a
   track in view. Each becomes a window of ±3 s, overlapping windows merged.
3. Visual odometry on the candidate windows only (vo_heading): the window's frames decoded again, Lucas–Kanade
   corners off the tracked boxes, per frame pair the rotation of the road's homography (decomposed; the solution
   whose normal is the road's) or, failing that, of the essential matrix (pair_yaw), the yaw increments
   integrated from the GPS heading at the window's start — the GPS's 1 Hz course misses a swerve inside a second;
   the odometry does not. Positions follow by integrating the GPS speed along that
   heading from the smoothed position at the window's start. Refused (the GPS kept, said why) with too few
   inliers in more than 30 % of the pairs, or when the integrated heading departs from the GPS's by more than 20°.
4. Trajectories: each track's points attached to the ego pose of their frame (a vehicle's centre half its typical
   length beyond its near face, along the ray — as core.incident does), smoothed by the same RTS smoother
   (measurement σ along the ray the fused range σ, across it 0.3 m + 1 % of the range; process 3 m/s² for vehicles, 1
   m/s² for walkers) → (N, 8) arrays (t, x, y, yaw, vx, vy, length, width; sizes typical for the kind,
   core.scenario_plan.SIZES) for the ego and the actor.
5. Measures (score_actor): core.criticality's ttc_series, pet, required_deceleration_series and min_distance; the
   ego's peak deceleration and jerk in the window; the criticality score (criticality_terms → criticality_score). A
   bootstrap over the depth scale — every range multiplied by a factor drawn uniformly in [0.9, 1.1], 50 draws, the
   trajectories and measures recomputed — gives the score's median and 5–95 % interval. Events are ranked by that
   median (then the point score).

Outputs (export_event, for the top 20 across a folder): the event's video stretch with 5 s either side cut by
`ffmpeg -c copy` (no re-encoding: it starts at the key frame at or before the cut), event.json, a
falcon.dashcam_detections/1 file whose boxes carry their fused `range_m` and σ, and posed/posed.json (each frame's
time, the ego's pose in local east-north-up metres about the GPS origin, the camera's view matrix and intrinsics)
— the reconstruction core.incident reads, so make_scenario() turns the event into a Falcon scenario, a parametric
variations file and an OpenSCENARIO file on a world whose lanes it is on, and variations_spec() queues the rare-case
search over those variations (sim.studio). Event ids are hashes (never the clip's name, which carries its date).

Privacy: everything here is derived from the user's footage and stays on the machine (assistant.privacy marks the
ingest folder and every event; summary() is counts and ids only).

Limits: one camera, monocular: ranges rest on a flat road, a known camera height and the estimated horizon (or a
metric depth network); sizes are typical, not measured; the ego's speed is the GPS's (1 Hz, smoothed); the GPS-to-
video offset is about ±0.5 s; events do not span clips; the criticality score's scales are heuristic
(core.criticality).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

SCHEMA = 'falcon.nearmiss/1'
GPS_SIGMA_M = 3.0
GPS_SPEED_SIGMA = 0.5
EGO_ACC = 2.0
ACTOR_ACC = {'walker': 1.0, 'vehicle': 3.0}
CORRIDOR_M = 2.5
CANDIDATE_TTC_S = 4.0
CANDIDATE_RANGE_M = 6.0
HARD_BRAKE_MPS2 = 3.0
WINDOW_PAD_S = 3.0
MIN_OBS = 3
BOOT = 50
DEPTH_SCALE = 0.10
CLIP_PAD_S = 5.0
TOP = 20
VO_MIN_INLIERS = 20
VO_MAX_FAIL = 0.3
VO_MAX_DEPART_DEG = 20.0
WALKERS = ('pedestrian', 'child')
THRESHOLDS = {'corridor_half_width_m': CORRIDOR_M, 'candidate_ttc_s': CANDIDATE_TTC_S,
              'candidate_range_m': CANDIDATE_RANGE_M, 'hard_brake_mps2': HARD_BRAKE_MPS2,
              'window_pad_s': WINDOW_PAD_S, 'min_observations': MIN_OBS, 'bootstrap': BOOT,
              'depth_scale_spread': DEPTH_SCALE, 'clip_pad_s': CLIP_PAD_S, 'top': TOP,
              'gps_sigma_m': GPS_SIGMA_M, 'gps_speed_sigma_mps': GPS_SPEED_SIGMA, 'ego_process_acc_mps2': EGO_ACC,
              'actor_process_acc_mps2': ACTOR_ACC, 'vo_min_inliers': VO_MIN_INLIERS,
              'vo_max_failed_share': VO_MAX_FAIL, 'vo_max_departure_deg': VO_MAX_DEPART_DEG}


class NearMissError(Exception):
    pass


# ================================================================================================ smoothing
def rts(times, meas, q_acc: float, x0=None, P0=None) -> tuple:
    """A constant-velocity Kalman filter on (x, y, vx, vy) and its Rauch–Tung–Striebel smoother. meas: per time a
    list of (H (m × 4), z (m,), R (m × m)) — possibly empty. Returns the smoothed means (N, 4) and covariances."""
    t = np.asarray(times, float)
    n = len(t)
    if n == 0:
        return np.zeros((0, 4)), np.zeros((0, 4, 4))
    x = np.zeros(4) if x0 is None else np.asarray(x0, float)
    P = np.diag([1e4, 1e4, 1e2, 1e2]) if P0 is None else np.asarray(P0, float)
    xs, Ps, xp, Pp, Fs = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros((n, 4)), np.zeros((n, 4, 4)), []
    for k in range(n):
        dt = t[k] - t[k - 1] if k else 0.0
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = q_acc ** 2
        Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
        x, P = F @ x, F @ P @ F.T + Q
        Fs.append(F)
        xp[k], Pp[k] = x, P
        for H, z, R in meas[k] or ():
            H, z, R = np.asarray(H, float), np.asarray(z, float), np.asarray(R, float)
            S = H @ P @ H.T + R
            K = np.linalg.solve(S, H @ P).T
            x = x + K @ (z - H @ x)
            P = (np.eye(4) - K @ H) @ P
        xs[k], Ps[k] = x, P
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ Fs[k + 1].T @ np.linalg.pinv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps


def savgol_derivs(v, dt: float) -> tuple:
    """(first, second) derivatives of a uniformly sampled series by Savitzky–Golay filters over about 1 s
    (orders 2 and 3)."""
    from scipy.signal import savgol_filter
    v = np.asarray(v, float)
    n = len(v)
    w = int(round(1.0 / dt)) | 1
    w = max(5, w)
    if n < 5:
        return np.zeros(n), np.zeros(n)
    w = min(w, n if n % 2 else n - 1)
    d1 = savgol_filter(v, w, min(2, w - 1), deriv=1, delta=dt)
    d2 = savgol_filter(v, w, min(3, w - 1), deriv=2, delta=dt) if w >= 5 else np.zeros(n)
    return d1, d2


# ================================================================================================ ego motion
def ego_motion(trk: dict, times) -> dict:
    """The ego's smoothed state at `times` (video seconds) from the clip's GPS track (module doc, stage 1): arrays t,
    x, y (east, north metres about the first fix), vx, vy, speed, yaw (degrees, counter-clockwise from east), accel,
    jerk, ok (a fix within 2.5 s), and the origin (lat0, lon0)."""
    from .dashcam import MAX_GAP_S
    from .osm import local_projection
    fx = [f for f in trk.get('fixes') or () if f.get('lat') is not None]
    times = np.asarray(times, float)
    if len(fx) < 2:
        raise NearMissError('fewer than two GPS fixes: no ego motion')
    pr = local_projection(fx[0]['lat'], fx[0]['lon'])
    ft = np.array([f['t'] for f in fx])
    e, n = pr.to_local(np.array([f['lat'] for f in fx]), np.array([f['lon'] for f in fx]))
    grid = np.unique(np.r_[ft, times])
    meas = []
    at = {round(float(t), 6): i for i, t in enumerate(ft)}
    Hp = np.eye(2, 4)
    Hv = np.c_[np.zeros((2, 2)), np.eye(2)]
    for t in grid:
        i = at.get(round(float(t), 6))
        m = []
        if i is not None:
            m.append((Hp, [e[i], n[i]], np.eye(2) * GPS_SIGMA_M ** 2))
            sp, crs = fx[i].get('speed_kmh'), fx[i].get('course_deg')
            if sp is not None and crs is not None and sp / 3.6 > 0.5:
                v = sp / 3.6
                m.append((Hv, [v * math.sin(math.radians(crs)), v * math.cos(math.radians(crs))],
                          np.eye(2) * GPS_SPEED_SIGMA ** 2))
            elif sp is not None and sp / 3.6 <= 0.5:
                m.append((Hv, [0.0, 0.0], np.eye(2) * GPS_SPEED_SIGMA ** 2))
        meas.append(m)
    x0 = [e[0], n[0], 0.0, 0.0]
    xs, _ = rts(grid, meas, EGO_ACC, x0=x0, P0=np.diag([GPS_SIGMA_M ** 2] * 2 + [100.0] * 2))
    idx = np.searchsorted(grid, times)
    idx = np.clip(idx, 0, len(grid) - 1)
    S = xs[idx]
    speed = np.hypot(S[:, 2], S[:, 3])
    yaw = np.degrees(np.arctan2(S[:, 3], S[:, 2]))
    for k in range(len(yaw)):                               # standing still: hold the last heading
        if speed[k] < 0.5 and k:
            yaw[k] = yaw[k - 1]
    yaw = np.degrees(np.unwrap(np.radians(yaw)))
    gap = np.min(np.abs(times[:, None] - ft[None, :]), axis=1) if len(times) else np.zeros(0)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    acc, jerk = savgol_derivs(speed, dt)
    return {'t': times, 'x': S[:, 0], 'y': S[:, 1], 'vx': S[:, 2], 'vy': S[:, 3], 'speed': speed, 'yaw': yaw,
            'accel': acc, 'jerk': jerk, 'ok': gap <= MAX_GAP_S, 'lat0': float(fx[0]['lat']),
            'lon0': float(fx[0]['lon']), 'proj': pr.proj, 'heading': 'gps'}


def ego_pose(ego: dict, k: int) -> np.ndarray:
    """world ← vehicle (4 × 4) at frame k: east-north-up, x forward."""
    a = math.radians(ego['yaw'][k])
    E = np.eye(4)
    E[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    E[:2, 3] = [ego['x'][k], ego['y'][k]]
    return E


# ================================================================================================ candidates
def _tracks(perc: dict) -> dict:
    """{track id: [(frame k, t, box record)]} of a perception record."""
    out = {}
    for k, fr in enumerate(perc['frames']):
        for b in fr['boxes']:
            if b.get('track') is None or b.get('forward') is None:
                continue
            out.setdefault(int(b['track']), []).append((k, float(fr['t']), b))
    return out


def candidates(perc: dict, ego: dict) -> list:
    """Windows [{'t0', 't1', 'why': [..], 'tracks': [ids]}] (module doc, stage 2)."""
    times = np.array([f['t'] for f in perc['frames']])
    hits = []
    for tid, rows in _tracks(perc).items():
        if len(rows) < MIN_OBS:
            continue
        t = np.array([r[1] for r in rows])
        f = np.array([r[2]['forward'] for r in rows])
        lft = np.array([r[2]['left'] for r in rows])
        rng = np.hypot(f, lft)
        if len(rng) >= 3:
            rng_s = np.r_[rng[0], [np.median(rng[i - 1:i + 2]) for i in range(1, len(rng) - 1)], rng[-1]]
        else:
            rng_s = rng
        rate = np.gradient(rng_s, t) if len(t) > 1 else np.zeros(len(t))
        for i in range(len(t)):
            why = None
            closing = -rate[i]
            if f[i] > 0 and abs(lft[i]) < CORRIDOR_M and closing > 0.5 and rng_s[i] / closing < CANDIDATE_TTC_S:
                why = 'closing in the corridor'
            elif rng_s[i] < CANDIDATE_RANGE_M:
                why = 'near'
            if why:
                hits.append((t[i], why, tid))
    in_view = {round(float(f['t']), 6) for f in perc['frames'] if f['boxes']}
    for k, t in enumerate(ego['t']):
        if ego['ok'][k] and ego['accel'][k] < -HARD_BRAKE_MPS2 and round(float(t), 6) in in_view:
            hits.append((float(t), 'ego braking hard', None))
    hits.sort(key=lambda h: h[0])
    out = []
    lo, hi = (float(times[0]), float(times[-1])) if len(times) else (0.0, 0.0)
    for t, why, tid in hits:
        a, b = max(lo, t - WINDOW_PAD_S), min(hi, t + WINDOW_PAD_S)
        if out and a <= out[-1]['t1']:
            w = out[-1]
            w['t1'] = max(w['t1'], b)
        else:
            out.append({'t0': a, 't1': b, 'why': [], 'tracks': []})
            w = out[-1]
        if why not in w['why']:
            w['why'].append(why)
        if tid is not None and tid not in w['tracks']:
            w['tracks'].append(tid)
    return out


# ================================================================================================ odometry
def _yaw_of(R, M) -> float:
    """The vehicle's yaw change (degrees, counter-clockwise from above) of a camera rotation R (points: frame 1 →
    frame 2 camera coordinates): in the vehicle frame static points turn against the vehicle."""
    Rv = M.T @ R @ M
    return -math.degrees(math.atan2(Rv[1, 0], Rv[0, 0]))


def pair_yaw(q0, q1, cam, ground_v: float | None = None) -> tuple:
    """(yaw change in degrees or NaN, inliers, method) between two frames' matched points (OpenCV pixels). First the
    road: a homography (RANSAC, 1 px) of the points below the horizon, decomposed (cv2.decomposeHomographyMat), the
    solution whose plane normal is the road's (within 25°, the camera's mount) — exact for the road plane, with or
    without translation; else the essential matrix of all the points (RANSAC, 1 px) and its pose (recoverPose)."""
    import cv2
    K = np.array([[cam.f, 0, cam.cx], [0, cam.f, cam.cy], [0, 0, 1.0]])
    M = cam.mount()
    n_road = M @ np.array([0.0, 0.0, 1.0])                       # the road's normal (up) in camera coordinates
    gv = cam.horizon_v + 5.0 if ground_v is None else ground_v
    g = q0[:, 1] > gv
    if g.sum() >= VO_MIN_INLIERS:
        H, inl = cv2.findHomography(q0[g], q1[g], cv2.RANSAC, 1.0)
        if H is not None and inl is not None and int(inl.sum()) >= VO_MIN_INLIERS:
            _n, Rs, _ts, Ns = cv2.decomposeHomographyMat(H, K)
            best = max(range(len(Rs)), key=lambda k: abs(float(Ns[k].ravel() @ n_road)))
            if abs(float(Ns[best].ravel() @ n_road)) >= math.cos(math.radians(25.0)):
                return _yaw_of(Rs[best], M), int(inl.sum()), 'road homography'
    E, inl = cv2.findEssentialMat(q0, q1, K, cv2.RANSAC, 0.999, 1.0)
    if E is not None and E.shape == (3, 3):
        n_in, R, _t, _m = cv2.recoverPose(E, q0, q1, K, mask=inl)
        if n_in >= VO_MIN_INLIERS:
            return _yaw_of(R, M), int(n_in), 'essential matrix'
    return float('nan'), 0, None


def vo_heading(grays: list, cam, boxes_per_frame=None) -> dict:
    """Yaw increments between consecutive frames (module doc, stage 3; pair_yaw): {'dyaw_deg': [..] (NaN where a
    pair failed), 'inliers': [..], 'methods': {..}, 'failed': n}. Lucas–Kanade corners off the tracked boxes, kept
    when tracked back to within 0.5 px."""
    import cv2
    out = {'dyaw_deg': [], 'inliers': [], 'failed': 0, 'methods': {}}
    for i in range(1, len(grays)):
        a, b = grays[i - 1], grays[i]
        mask = np.full(a.shape, 255, np.uint8)
        for bx in (boxes_per_frame[i - 1] if boxes_per_frame else ()):
            x1, y1, x2, y2 = (int(round(v)) for v in bx)
            mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
        p0 = cv2.goodFeaturesToTrack(a, 600, 0.001, 7, mask=mask)
        dy, n_in, how = float('nan'), 0, None
        if p0 is not None and len(p0) >= VO_MIN_INLIERS:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=3)
            bk, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, winSize=(21, 21), maxLevel=3)
            fb = np.linalg.norm(bk.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
            q0, q1 = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
            if len(q0) >= VO_MIN_INLIERS:
                if float(np.median(np.linalg.norm(q1 - q0, axis=1))) < 0.05:
                    dy, n_in, how = 0.0, len(q0), 'standing still'
                else:
                    dy, n_in, how = pair_yaw(q0, q1, cam)
        if not math.isfinite(dy):
            out['failed'] += 1
        elif how:
            out['methods'][how] = out['methods'].get(how, 0) + 1
        out['dyaw_deg'].append(dy)
        out['inliers'].append(int(n_in))
    return out


def apply_vo(ego: dict, ks: np.ndarray, vo: dict) -> dict:
    """The ego over frames ks with the odometry's heading (module doc, stage 3), or unchanged with the reason."""
    d = np.asarray(vo['dyaw_deg'], float)
    n = len(d)
    info = {'pairs': n, 'failed': int(vo['failed'])}
    if n == 0 or vo['failed'] > VO_MAX_FAIL * n:
        info['used'] = False
        info['why'] = f'{vo["failed"]} of {n} frame pairs without a pose'
        return info
    gps_d = np.diff(ego['yaw'][ks])
    d = np.where(np.isfinite(d), d, gps_d)
    yaw = ego['yaw'][ks[0]] + np.r_[0.0, np.cumsum(d)]
    depart = float(np.max(np.abs(yaw - ego['yaw'][ks])))
    info['departure_from_gps_deg'] = round(depart, 2)
    if depart > VO_MAX_DEPART_DEG:
        info['used'] = False
        info['why'] = f'the odometry\'s heading departs {depart:.0f}° from the GPS\'s'
        return info
    t = ego['t'][ks]
    sp = ego['speed'][ks]
    x, y = [ego['x'][ks[0]]], [ego['y'][ks[0]]]
    for i in range(1, len(ks)):
        dt = t[i] - t[i - 1]
        a = math.radians((yaw[i] + yaw[i - 1]) / 2.0)
        v = (sp[i] + sp[i - 1]) / 2.0
        x.append(x[-1] + v * dt * math.cos(a))
        y.append(y[-1] + v * dt * math.sin(a))
    x, y = np.array(x), np.array(y)
    info['position_departure_m'] = round(float(np.max(np.hypot(x - ego['x'][ks], y - ego['y'][ks]))), 3)
    ego['yaw'][ks] = yaw
    ego['x'][ks], ego['y'][ks] = x, y
    ego['vx'][ks], ego['vy'][ks] = sp * np.cos(np.radians(yaw)), sp * np.sin(np.radians(yaw))
    info['used'] = True
    return info


# ================================================================================================ trajectories
def _size(kind):
    from .scenario_plan import EGO_SIZE, SIZES
    return SIZES.get(kind, EGO_SIZE)


def ego_traj(ego: dict, ks) -> np.ndarray:
    from .scenario_plan import EGO_SIZE
    ks = np.asarray(ks, int)
    return np.c_[ego['t'][ks], ego['x'][ks], ego['y'][ks], ego['yaw'][ks], ego['vx'][ks], ego['vy'][ks],
                 np.full(len(ks), EGO_SIZE[0]), np.full(len(ks), EGO_SIZE[1])]


def actor_traj(rows: list, ego: dict, kind: str, scale: float = 1.0) -> np.ndarray:
    """(N, 8) of one track (module doc, stage 4); rows [(frame k, t, box)], scale multiplies every range."""
    L, W = _size(kind)
    walker = kind in WALKERS
    times, meas = [], []
    for k, t, b in rows:
        f, lft = b['forward'] * scale, b['left'] * scale
        r = math.hypot(f, lft) or 1e-6
        u = np.array([f, lft]) / r
        if not walker:
            f, lft = f + u[0] * L / 2.0, lft + u[1] * L / 2.0
        a = math.radians(ego['yaw'][k])
        Rw = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p = np.array([ego['x'][k], ego['y'][k]]) + Rw @ np.array([f, lft])
        ur = Rw @ u
        uc = np.array([-ur[1], ur[0]])
        s_r = max(float(b.get('sigma') or 1.0) * scale, 0.1)
        s_c = 0.3 + 0.01 * r
        R = s_r ** 2 * np.outer(ur, ur) + s_c ** 2 * np.outer(uc, uc)
        times.append(t)
        meas.append([(np.eye(2, 4), p, R)])
    p0 = meas[0][0][1]
    xs, _ = rts(times, meas, ACTOR_ACC['walker' if walker else 'vehicle'], x0=[p0[0], p0[1], 0.0, 0.0],
                P0=np.diag([meas[0][0][2][0, 0] + 1, meas[0][0][2][1, 1] + 1, 100.0, 100.0]))
    sp = np.hypot(xs[:, 2], xs[:, 3])
    yaw = np.where(sp > 1.0, np.degrees(np.arctan2(xs[:, 3], xs[:, 2])),
                   np.array([ego['yaw'][k] for k, _, _ in rows]))
    return np.c_[np.array(times), xs[:, 0], xs[:, 1], yaw, xs[:, 2], xs[:, 3], np.full(len(times), L),
                 np.full(len(times), W)]


def measures(e: np.ndarray, a: np.ndarray) -> dict:
    """core.criticality's measures of one actor against the ego and the criticality score."""
    from . import criticality as CR
    t, tt = CR.ttc_series(e, a)
    k = int(np.argmin(tt)) if len(tt) else None
    min_ttc = float(tt[k]) if k is not None and math.isfinite(tt[k]) else None
    pet = CR.pet(e, a)
    td, dr = CR.required_deceleration_series(e, a)
    fin = np.where(np.isfinite(dr), dr, -1.0)
    kd = int(np.argmax(fin)) if len(fin) and fin.max() > 0 else None
    dmin, dmin_t = CR.min_distance(e, a)
    col = CR.collisions(e, a)
    collided = bool(col['times'])
    terms = CR.criticality_terms(min_ttc, pet, float(dr[kd]) if kd is not None else 0.0, dmin)
    return {'min_ttc_s': min_ttc, 'min_ttc_t': float(t[k]) if min_ttc is not None else None, 'pet_s': pet,
            'max_drac_mps2': float(dr[kd]) if kd is not None else 0.0,
            'min_distance_m': None if not math.isfinite(dmin) else float(dmin), 'min_distance_t': dmin_t,
            'overlap': collided, 'terms': terms, 'score': CR.criticality_score(terms, collided)}


def score_actor(rows: list, ego: dict, ks: np.ndarray, kind: str, seed: int = 0) -> dict:
    """measures() at the measured scale and over the depth-scale bootstrap (module doc, stage 5)."""
    e = ego_traj(ego, ks)
    a = actor_traj(rows, ego, kind)
    m = measures(e, a)
    rng = np.random.default_rng(seed)
    scores = [measures(e, actor_traj(rows, ego, kind, float(s)))['score']
              for s in rng.uniform(1.0 - DEPTH_SCALE, 1.0 + DEPTH_SCALE, BOOT)]
    m['score_median'] = float(np.median(scores))
    m['score_ci'] = [float(np.percentile(scores, 5)), float(np.percentile(scores, 95))]
    m['trajectory'] = {'actor': np.round(a, 4).tolist(), 'ego': np.round(e, 4).tolist(),
                       'columns': ['t', 'x', 'y', 'yaw_deg', 'vx', 'vy', 'length', 'width']}
    return m


def event_id(clip_path: str, t0: float, track) -> str:
    return hashlib.sha1(f'{os.path.realpath(clip_path)}|{t0:.2f}|{track}'.encode()).hexdigest()[:10]


def mine_clip(perc: dict, trk: dict, clip_path: str, cam, decode=None, emitter=None) -> dict:
    """Stages 1–5 for one clip (module doc): {'schema', 'events' (best first), 'windows', 'ego': summary}.
    decode(t0, t1) → [(n, t, gray)] decodes a window for the odometry (None: GPS heading only)."""
    times = np.array([f['t'] for f in perc['frames']], float)
    out = {'schema': SCHEMA, 'events': [], 'windows': [], 'thresholds': THRESHOLDS}
    if len(times) < 2:
        out['note'] = 'fewer than two processed frames'
        return out
    try:
        ego = ego_motion(trk, times)
    except NearMissError as e:
        out['note'] = str(e)
        return out
    out['ego'] = {'frames_with_state': int(ego['ok'].sum()), 'max_speed_mps': round(float(ego['speed'].max()), 2),
                  'peak_decel_mps2': round(float(max(0.0, -ego['accel'].min())), 2)}
    tracks = _tracks(perc)
    wins = candidates(perc, ego)
    for w in wins:
        ks = np.flatnonzero((times >= w['t0'] - 1e-9) & (times <= w['t1'] + 1e-9) & ego['ok'])
        if len(ks) < MIN_OBS:
            continue
        w['vo'] = {'used': False, 'why': 'no decoder'}
        if decode is not None:
            try:
                fr = decode(float(times[ks[0]]), float(times[ks[-1]]))
                by_t = {round(t, 3): g for _n, t, g in fr}
                grays = [by_t.get(round(float(times[k]), 3)) for k in ks]
                if all(g is not None for g in grays):
                    boxes = [[b['box'] for b in perc['frames'][k]['boxes']] for k in ks]
                    w['vo'] = apply_vo(ego, ks, vo_heading(grays, cam, boxes))
                else:
                    w['vo'] = {'used': False, 'why': 'the window\'s frames could not be decoded again'}
            except Exception as e:             # noqa: BLE001 (OpenCV, ffmpeg: the GPS heading is kept, said why)
                w['vo'] = {'used': False, 'why': f'{type(e).__name__}: {e}'}
        kset = set(int(k) for k in ks)
        for tid, rows in tracks.items():
            rr = [r for r in rows if r[0] in kset]
            if len(rr) < MIN_OBS:
                continue
            kinds = [r[2].get('kind') for r in rr]
            kind = max(set(kinds), key=kinds.count)
            src = 'heuristic' if any(r[2].get('class_source') == 'heuristic' and r[2].get('kind') == kind
                                     for r in rr) else 'model'
            eid = event_id(clip_path, w['t0'], tid)
            m = score_actor(rr, ego, ks, kind, seed=int(eid[:8], 16))
            sl = slice(int(ks[0]), int(ks[-1]) + 1)
            dec = -ego['accel'][sl]
            m.update(id=eid, track=int(tid), kind=kind, class_source=src, window_s=[round(w['t0'], 3),
                                                                                  round(w['t1'], 3)],
                     observations=len(rr), why=list(w['why']), vo=w['vo'],
                     ego_peak_decel_mps2=round(float(max(0.0, dec.max())), 3),
                     ego_peak_jerk_mps3=round(float(np.abs(ego['jerk'][sl]).max()), 3),
                     depth_sources={s: sum(1 for r in rr if r[2].get('source') == s)
                                    for s in sorted({r[2].get('source') for r in rr})},
                     frames=[int(perc['frames'][k]['n']) for k in ks])
            out['events'].append(m)
        out['windows'].append(w)
    out['events'].sort(key=lambda m: (-m['score_median'], -m['score']))
    out['ego_track'] = {k: (np.round(ego[k], 4).tolist() if isinstance(ego[k], np.ndarray) else ego[k])
                        for k in ('t', 'x', 'y', 'yaw', 'vx', 'vy', 'speed', 'accel', 'lat0', 'lon0', 'proj')}
    out['ego_track']['ok'] = ego['ok'].astype(bool).tolist()
    return out


def rank(events: list, top: int | None = None) -> list:
    out = sorted(events, key=lambda m: (-m['score_median'], -m['score'], m['id']))
    for i, m in enumerate(out):
        m['rank'] = i + 1
    return out[:top] if top else out


# ================================================================================================ exporting an event
def cut_clip(clip: str, t0: float, t1: float, out_path: str) -> dict:
    """The video stretch [t0 − 5 s, t1 + 5 s] by stream copy (no re-encoding)."""
    start = max(0.0, t0 - CLIP_PAD_S)
    dur = (t1 + CLIP_PAD_S) - start
    tmp = out_path + '.part.mp4'
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-ss', f'{start:.3f}', '-i', clip,
           '-t', f'{dur:.3f}', '-map', '0:v:0', '-c', 'copy', '-avoid_negative_ts', 'make_zero', tmp]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.isfile(tmp):
        raise NearMissError(f'ffmpeg could not cut the clip: {r.stderr[-300:]}')
    os.replace(tmp, out_path)
    return {'start_s': round(start, 3), 'seconds': round(dur, 3), 'how': 'ffmpeg -c copy (starts at the key frame '
                                                                          'at or before the cut)'}


def export_event(ev: dict, mined: dict, perc: dict, cam, clip_path: str, out_dir: str, cut: bool = True) -> dict:
    """An event's folder (module doc): event.json, detections.json, posed/posed.json, clip.mp4 and run.json (the
    privacy marker)."""
    from .incident import DETECTIONS_SCHEMA
    from .provenance import utc_now, write_json
    os.makedirs(os.path.join(out_dir, 'posed'), exist_ok=True)
    eg = {k: np.asarray(v) if isinstance(v, list) else v for k, v in mined['ego_track'].items()}
    frames_n = set(ev['frames'])
    ks = [k for k, f in enumerate(perc['frames']) if int(f['n']) in frames_n]
    t0 = float(perc['frames'][ks[0]]['t'])
    posed, det = [], []
    for k in ks:
        fr = perc['frames'][k]
        E = ego_pose(eg, k)
        posed.append({'index': int(fr['n']), 't': round(float(fr['t']) - t0, 6), 'video_t': float(fr['t']),
                      'K': [round(v, 6) for v in cam.K], 'size': [cam.width, cam.height],
                      'viewmat': cam.view(E).reshape(-1).tolist(), 'ego': E.reshape(-1).tolist()})
        boxes = []
        for b in fr['boxes']:
            if b.get('track') is None or b.get('forward') is None:
                continue
            boxes.append({'box': [round(v, 2) for v in b['box']], 'kind': b['kind'], 'track': f't{b["track"]}',
                          'range_m': round(math.hypot(b['forward'], b['left']), 3),
                          'range_sigma_m': round(float(b.get('sigma') or 0.0), 3),
                          'class_source': b.get('class_source', 'model')})
        det.append({'index': int(fr['n']), 'boxes': boxes})
    write_json(os.path.join(out_dir, 'posed', 'posed.json'), {
        'schema': 'falcon.posed/1', 'name': f'near-miss {ev["id"]}', 'camera': 'dashcam-front',
        'world': 'local east-north-up about the clip\'s first GPS fix, metres; z = 0 on the road under the camera; '
                 'the ego from the smoothed GPS (and the odometry\'s heading where used: event.json)',
        'geo': {'lat0': eg['lat0'], 'lon0': eg['lon0'], 'proj': eg.get('proj')}, 'source': {'format': 'dashcam'},
        'camera_model': cam.doc(), 'frames': posed})
    write_json(os.path.join(out_dir, 'detections.json'), {
        'schema': DETECTIONS_SCHEMA, 'source': {'format': 'dashcam', 'kind': 'nearmiss'},
        'note': 'boxes tracked by ByteTrack; range_m is the fused monocular range (core.dashcam_depth)', 'frames': det})
    event = {k: v for k, v in ev.items() if k != 'trajectory'}
    event.update(schema=SCHEMA, camera=cam.doc(), clip_cut=None)
    if cut:
        event['clip_cut'] = cut_clip(clip_path, ev['window_s'][0], ev['window_s'][1], os.path.join(out_dir,
                                                                                                    'clip.mp4'))
    write_json(os.path.join(out_dir, 'event.json'), event)
    write_json(os.path.join(out_dir, 'trajectories.json'), ev['trajectory'])
    with open(os.path.join(out_dir, 'dashcam.json'), 'w', encoding='utf-8') as fh:
        json.dump({'schema': 'falcon.dashcam_recon/1', 'made_by': 'core.nearmiss'}, fh)
    write_json(os.path.join(out_dir, 'run.json'), {
        'worker': 'dashcam-nearmiss', 'status': 'ok', 'created': utc_now(), 'source': {'format': 'dashcam'},
        'summary': {k: ev.get(k) for k in ('rank', 'kind', 'score', 'score_median', 'min_ttc_s', 'pet_s')}})
    return event


# ================================================================================================ scenarios
def make_scenario(event_dir: str, project, world_id: str, name: str = '') -> dict:
    """The event as a Falcon scenario, its variations and OpenSCENARIO (core.incident, with the event's fused
    ranges), in the workspace's incidents/<name>/. Returns {'name', 'folder', summary…}."""
    from . import incident as INC
    name = name or f'nearmiss-{os.path.basename(os.path.normpath(event_dir)).split("-")[-1]}'
    inc = INC.extract(event_dir, project, world_id, detections=os.path.join(event_dir, 'detections.json'))
    out = INC.incident_dir(project.workspace, name)
    summ = INC.write(inc, out, name)
    return {'name': name, 'folder': out, **summ}


def variations_spec(project_path: str, workspace: str, name: str, budget: int = 40, method: str = 'bayes',
                    fake_carla: str = ''):
    """The local rare-case search over an incident's variations as a queued job (sim.studio through
    `falcon dashcam incident search`'s argv; placement local: dashcam-derived scenarios never leave the machine)."""
    from . import incident as INC
    from .jobs import JobSpec
    argv = [sys.executable, '-m', 'falcon_autonomy'] + INC.search_argv(project_path, workspace, name, budget, method,
                                                                       fake_carla=fake_carla)
    return JobSpec('study', f'Search the variations of {name}', argv, memory_gb=0.5,
                   meta={'source': {'format': 'dashcam'}, 'incident': name})
