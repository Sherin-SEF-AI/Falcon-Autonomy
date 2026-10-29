"""Monocular visual odometry and mapping for a camera with unknown intrinsics (a dashcam), with OpenCV, numpy and
scipy only — no COLMAP, OpenSfM or pycolmap (docs/reconstruction.md, "From a dashcam").

    intr = Intrinsics.from_hfov(960, 540, 100.0)               # a pinhole guess (plumb_line() for k1)
    tracks = track(grays, mask, intr)                          # KLT tracks, epipolar outliers dropped
    dist = speed_distance(gps_t, gps_speed)                    # metres travelled between two times
    vo = run_vo(tracks, intr, times=times, distance=dist)      # keyframes, PnP, triangulation, local BA (metric)
    gb = global_ba(vo, tracks, intr)                           # every keyframe (free=('f', 'k1'): self-calibration)
    al = align_to_gps(vo.centres(), times, gps_t, gps_xy, up)  # similarity to the GPS track: the honest residuals
    transform(vo, al['s'], al['R'], b); gps_aided_ba(...)      # geolocated poses pulled towards the GPS

The camera model: a pinhole with one focal length f (square pixels), the principal point at the image centre (an
assumption: not estimated) and radial distortion x_d = x·(1 + k1·r² + k2·r⁴) of the normalised coordinates (OpenCV's
radial terms; no tangential terms). Pixel coordinates here are OpenCV's (a pixel's centre is an integer); the
reconstruction's frames are written in Falcon's pixel-edge convention (+½) by the caller.

The pipeline:

* **Tracking** — Shi–Tomasi corners (cv2.goodFeaturesToTrack) followed through every frame by pyramidal Lucas–Kanade
  (cv2.calcOpticalFlowPyrLK), kept when tracking back lands within fb_px of the start; replenished in empty image
  areas. Between consecutive frames the undistorted tracks are fitted with an essential matrix (RANSAC,
  track_epi_px) and the outliers dropped: independently moving points that leave their epipolar lines (crossing
  traffic, a swerving two-wheeler). Points moving along their epipolar lines — traffic ahead or oncoming, driving
  the same line as the camera — are **not** caught (the epipolar test cannot see them); a mask (the bonnet, a
  burnt-in overlay, the corridor of road ahead) keeps corners off parts of the image.
* **Initialisation** — the first frame and the first later one with enough parallax (the tracks moved init_px, at
  least init_min_points triangulated at a median ray angle above init_angle_deg, half the tracks inliers):
  essential matrix (RANSAC), cv2.recoverPose, triangulation; the baseline is the unit of length — or, given the
  distance travelled (the GPS speed over ground), metres.
* **Keyframes** — a new keyframe when the tracks have moved kf_px (median, since the last keyframe), when fewer than
  kf_track_ratio of the last keyframe's mapped tracks survive, or after max_kf_gap frames.
* **Tracking the map** — every frame's pose by PnP (cv2.solvePnPRansac on the undistorted points, then
  Levenberg–Marquardt refinement) against the triangulated points its tracks carry. With the distance travelled,
  the pose is first predicted (constant velocity, the step set to that distance) and the points more than gate_px
  from where the prediction puts them are left out — and dropped from the map after 3 such frames: points on a
  vehicle the camera follows pass every epipolar test but drag the PnP towards standing still (measured on the
  user's highway footage: the odometry's scale fell sixfold while it overtook a tanker); a PnP step disagreeing with
  that distance by more than −40 … +60 % keeps its rotation and takes the predicted position.
* **Mapping** — at each keyframe the tracks without a point are triangulated between the keyframe and the earliest
  keyframe of the window that saw them, kept when both reprojection errors are below tri_px, both depths positive and
  the rays meet at more than min_angle_deg.
* **Bundle adjustment** (bundle_adjust) — Levenberg–Marquardt with analytic derivatives, a Huber loss by reweighting
  and the points eliminated by the Schur complement; optional priors on camera positions and on distances between
  cameras. The local one: the last `window` keyframes (the oldest two fixed, or one when distances hold the scale)
  and their points; observations still above outlier_px afterwards are dropped. The global one: every keyframe and
  point, optionally with f, k1 and k2 free (self-calibration). The frames between keyframes are then re-placed by
  PnP or, when that strays from the keyframes' path, interpolated (replace_frames).
* **Intrinsics** — self_calibrate (VO + a BA with f and k1 free, repeated), plumb_line (k1 from straight edges) and
  the caller's focal search. On forward driving f is only weakly determined: on KITTI raw drive 0015 the odometry's
  error changed by under 0.3 m over 363 m with f off by −20 … +25 %, and the reprojection error not at all.
* **GPS** — align_to_gps (a robust similarity after levelling; the residuals are the independent check),
  gps_aided_ba (position priors: the poses a reconstruction uses).

Measured on KITTI raw (global-shutter camera, calibrated): drive 0005, 154 frames, 69 m: 0.65 m RMS against OXTS
after a similarity; drive 0015, 297 frames, 363 m: 5.1 m without and 1.42 m with the speed held by the OXTS speed.
Rolling shutter, motion blur and auto-exposure are not modelled.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULTS = {
    'max_corners': 3000, 'min_distance': 7, 'quality': 0.01, 'block': 7, 'fb_px': 2.0, 'epi_px': 1.5,
    'lk_window': 21, 'lk_levels': 4, 'replenish_below': 0.7,
    'kf_px': None, 'kf_track_ratio': 0.6, 'max_kf_gap': 12, 'window': 12, 'min_angle_deg': 0.75, 'tri_px': 2.0,
    'track_epi_px': 3.0, 'gate_px': 6.0, 'max_dev_rel': 0.5, 'max_dev_deg': 2.0,
    'pnp_px': 2.0, 'init_px': None, 'init_angle_deg': 1.5, 'init_min_points': 60, 'huber_px': 1.5, 'outlier_px': 4.0,
    'local_ba_iterations': 30, 'global_ba_iterations': 60, 'max_depth_ratio': 400.0,
}


class VOError(RuntimeError):
    pass


# ================================================================================================ the camera
@dataclass
class Intrinsics:
    width: int
    height: int
    f: float
    k1: float = 0.0
    k2: float = 0.0
    cx: float | None = None      # OpenCV convention; None: the image centre
    cy: float | None = None

    def __post_init__(self):
        if self.cx is None:
            self.cx = (self.width - 1) / 2.0
        if self.cy is None:
            self.cy = (self.height - 1) / 2.0

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float, k1=0.0, k2=0.0) -> 'Intrinsics':
        return cls(width, height, width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0)), k1, k2)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, 0.0, 0.0])

    @property
    def hfov_deg(self) -> float:
        """The horizontal field of the undistorted pinhole with this focal length (not the lens's own field)."""
        return math.degrees(2 * math.atan(self.width / (2 * self.f)))

    def scaled(self, width: int, height: int) -> 'Intrinsics':
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.f * sx, self.k1, self.k2, (self.cx + 0.5) * sx - 0.5,
                          (self.cy + 0.5) * sy - 0.5)

    def normalise(self, uv) -> np.ndarray:
        """Distorted pixels (N, 2) -> undistorted normalised coordinates (N, 2)."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        xd = np.c_[(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f]
        if not len(uv) or (self.k1 == 0 and self.k2 == 0):
            return xd
        rd = np.linalg.norm(xd, axis=1)
        r = rd.copy()
        rmax = self.monotonic_radius()
        for _ in range(30):                      # Newton on r·(1 + k1 r² + k2 r⁴) = r_d
            g = r * (1 + self.k1 * r ** 2 + self.k2 * r ** 4) - rd
            dg = 1 + 3 * self.k1 * r ** 2 + 5 * self.k2 * r ** 4
            r = np.clip(r - g / np.where(np.abs(dg) < 1e-9, 1e-9, dg), 0.0, rmax)
        scale = np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)
        return xd * scale[:, None]

    def monotonic_radius(self) -> float:
        """The undistorted radius up to which r·(1 + k1 r² + k2 r⁴) still grows (beyond it a strong barrel term folds
        the image back); inf when it always grows."""
        rs = np.linspace(0.0, 10.0, 4001)[1:]
        d = 1 + 3 * self.k1 * rs ** 2 + 5 * self.k2 * rs ** 4
        bad = np.nonzero(d <= 0)[0]
        return float(rs[bad[0] - 1]) if len(bad) else float('inf')

    def project_normalised(self, xy) -> np.ndarray:
        xy = np.asarray(xy, np.float64)
        r2 = np.sum(xy * xy, -1, keepdims=True)
        d = 1 + self.k1 * r2 + self.k2 * r2 * r2
        return np.c_[self.f * xy[:, 0] * d[:, 0] + self.cx, self.f * xy[:, 1] * d[:, 0] + self.cy]

    def doc(self) -> dict:
        return {'width': self.width, 'height': self.height, 'f': round(self.f, 4), 'cx': round(self.cx, 4),
                'cy': round(self.cy, 4), 'k1': round(self.k1, 6), 'k2': round(self.k2, 6),
                'hfov_deg_pinhole': round(self.hfov_deg, 3), 'convention': 'OpenCV pixel centres'}


# ================================================================================================ geometry
def skew(v) -> np.ndarray:
    v = np.asarray(v, float)
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1), np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rodrigues(r) -> np.ndarray:
    """Rotation vectors (N, 3) -> matrices (N, 3, 3), vectorised."""
    r = np.asarray(r, float).reshape(-1, 3)
    th = np.linalg.norm(r, axis=1)
    small = th < 1e-8
    k = r / np.where(small, 1.0, th)[:, None]
    Kx = skew(k)
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    R = np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)
    if small.any():
        R[small] = np.eye(3) + skew(r[small])
    return R


def rvec_of(R) -> np.ndarray:
    import cv2
    return cv2.Rodrigues(np.asarray(R, float))[0].reshape(3)


def pose_vec(R, t) -> np.ndarray:
    return np.r_[rvec_of(R), np.asarray(t, float).reshape(3)]


def pose_mat(p) -> np.ndarray:
    """A pose vector [rvec, t] (camera <- world) as a 4 × 4 matrix."""
    T = np.eye(4)
    T[:3, :3] = rodrigues(p[:3])[0]
    T[:3, 3] = p[3:6]
    return T


def centre_of(p) -> np.ndarray:
    R = rodrigues(p[:3])[0]
    return -R.T @ np.asarray(p[3:6])


def triangulate(T0, T1, x0, x1):
    """Points (N, 3) from normalised coordinates in two cameras (4 × 4 camera <- world): the linear DLT."""
    import cv2
    X = cv2.triangulatePoints(T0[:3], T1[:3], np.asarray(x0, float).T, np.asarray(x1, float).T)
    w = np.where(np.abs(X[3]) < 1e-12, 1e-12, X[3])
    return (X[:3] / w).T


def ray_angle_deg(X, c0, c1) -> np.ndarray:
    a = X - c0
    b = X - c1
    cos = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def reproj_px(T, X, xn, f) -> tuple:
    """(error in pixels of normalised coordinates scaled by f, depth) of world points in a camera."""
    pc = X @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    p = pc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
    return f * np.linalg.norm(p - xn, axis=1), z


# ================================================================================================ tracking
@dataclass
class Tracks:
    """Per frame: the ids of the tracks it sees and their distorted pixel positions."""
    ids: list
    uv: list
    size: tuple
    stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.ids)

    def lookup(self, i: int) -> dict:
        return dict(zip(self.ids[i].tolist(), range(len(self.ids[i]))))


def track(grays, mask=None, intr: Intrinsics | None = None, settings: dict | None = None, cancel=None) -> Tracks:
    """KLT tracks through a sequence of grayscale images (module doc). mask: (H, W) bool, True where corners may be."""
    import cv2
    s = dict(DEFAULTS, **(settings or {}))
    n = len(grays)
    H, W = grays[0].shape[:2]
    m8 = None if mask is None else (np.asarray(mask, bool).astype(np.uint8) * 255)
    lk = dict(winSize=(s['lk_window'], s['lk_window']), maxLevel=s['lk_levels'],
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    next_id = 0
    ids_out, uv_out = [], []
    st = {'detected': 0, 'lost_fb': 0, 'epipolar_outliers': 0, 'per_frame': []}
    t0 = time.perf_counter()

    def detect(img, have_uv):
        nonlocal next_id
        want = s['max_corners'] - len(have_uv)
        if want <= 0:
            return np.zeros((0,), np.int64), np.zeros((0, 2))
        m = np.full((H, W), 255, np.uint8) if m8 is None else m8.copy()
        for u, v in np.round(have_uv).astype(int):
            cv2.circle(m, (int(u), int(v)), s['min_distance'], 0, -1)
        p = cv2.goodFeaturesToTrack(img, want, s['quality'], s['min_distance'], mask=m, blockSize=s['block'])
        if p is None:
            return np.zeros((0,), np.int64), np.zeros((0, 2))
        p = p.reshape(-1, 2).astype(np.float64)
        new = np.arange(next_id, next_id + len(p))
        next_id += len(p)
        st['detected'] += len(p)
        return new, p

    ids, uv = detect(grays[0], np.zeros((0, 2)))
    ids_out.append(ids)
    uv_out.append(uv)
    for i in range(1, n):
        if cancel and cancel():
            raise KeyboardInterrupt()
        prev, cur = grays[i - 1], grays[i]
        if len(uv):
            p0 = uv.astype(np.float32).reshape(-1, 1, 2)
            p1, ok1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
            pb, ok2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
            fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
            q = p1.reshape(-1, 2).astype(np.float64)
            good = (ok1.reshape(-1) == 1) & (ok2.reshape(-1) == 1) & (fb < s['fb_px']) & \
                (q[:, 0] >= 0) & (q[:, 0] <= W - 1) & (q[:, 1] >= 0) & (q[:, 1] <= H - 1)
            if m8 is not None:
                qi = np.clip(np.round(q).astype(int), 0, [W - 1, H - 1])
                good &= m8[qi[:, 1], qi[:, 0]] > 0
            st['lost_fb'] += int((~good).sum())
            ids1, uv0, uv1 = ids[good], uv[good], q[good]
            if intr is not None and len(uv1) >= 12:
                disp = np.median(np.linalg.norm(uv1 - uv0, axis=1))
                if disp > 0.5:
                    a, b = intr.normalise(uv0), intr.normalise(uv1)
                    E, inl = cv2.findEssentialMat(a, b, np.eye(3), cv2.RANSAC, 0.999, s['track_epi_px'] / intr.f)
                    if E is not None and inl is not None:
                        keep = inl.reshape(-1) > 0
                        st['epipolar_outliers'] += int((~keep).sum())
                        ids1, uv1 = ids1[keep], uv1[keep]
            ids, uv = ids1, uv1
        if len(uv) < s['replenish_below'] * s['max_corners']:
            nid, nuv = detect(cur, uv)
            ids, uv = np.r_[ids, nid], np.r_[uv, nuv]
        ids_out.append(ids.astype(np.int64))
        uv_out.append(uv)
        st['per_frame'].append(len(ids))
    st['tracks'] = int(next_id)
    st['seconds'] = round(time.perf_counter() - t0, 3)
    st['mean_tracked'] = round(float(np.mean(st['per_frame'])), 1) if st['per_frame'] else 0.0
    del st['per_frame']
    return Tracks(ids_out, uv_out, (W, H), st)


# ================================================================================================ bundle adjustment
INTR_KEYS = ('f', 'k1', 'k2')


def _project_jac(R, t, X, cam_i, pt_i, f, k1, k2, cx, cy, want_jac=True):
    """Projections (M, 2) of the observations and, when want_jac, their derivatives: with respect to a left
    increment of the camera's pose — [R | t] ← [exp(ω) | δt]·[R | t] — (M, 2, 6), the point (M, 2, 3) and (f, k1,
    k2) (M, 2, 3);
    `front` marks the observations in front of their camera."""
    Rm = R[cam_i]
    pc = np.einsum('mij,mj->mi', Rm, X[pt_i]) + t[cam_i]
    front = pc[:, 2] > 1e-6
    z = np.where(front, pc[:, 2], 1.0)
    xn, yn = pc[:, 0] / z, pc[:, 1] / z
    r2 = xn * xn + yn * yn
    d = 1 + k1 * r2 + k2 * r2 * r2
    q = np.stack([f * xn * d + cx, f * yn * d + cy], -1)
    if not want_jac:
        return q, front, None
    m = len(cam_i)
    dd = 2 * (k1 + 2 * k2 * r2)                        # d(d)/d(r²) · 2
    Jn = np.empty((m, 2, 2))                           # d(u, v) / d(xn, yn)
    Jn[:, 0, 0] = f * (d + xn * dd * xn)
    Jn[:, 0, 1] = f * xn * dd * yn
    Jn[:, 1, 0] = f * yn * dd * xn
    Jn[:, 1, 1] = f * (d + yn * dd * yn)
    Jp = np.zeros((m, 2, 3))                           # d(xn, yn) / d(pc)
    Jp[:, 0, 0] = 1 / z
    Jp[:, 0, 2] = -xn / z
    Jp[:, 1, 1] = 1 / z
    Jp[:, 1, 2] = -yn / z
    Juv = Jn @ Jp                                      # (M, 2, 3): d(u, v) / d(pc)
    Jc = np.empty((m, 2, 6))
    Jc[:, :, :3] = -Juv @ skew(pc)                     # the pose updated as exp(ω)·[R | t]: d pc / dω = −[pc]×
    Jc[:, :, 3:] = Juv
    JX = Juv @ Rm
    Ji = np.stack([np.stack([xn * d, f * xn * r2, f * xn * r2 * r2], -1),
                   np.stack([yn * d, f * yn * r2, f * yn * r2 * r2], -1)], 1)
    return q, front, (Jc, JX, Ji)


def _huber(e, c):
    """(cost per observation, IRLS weight) of the Huber loss on the 2-vector error's norm e."""
    cost = np.where(e <= c, e * e, 2 * c * e - c * c)
    w = np.where(e <= c, 1.0, c / np.maximum(e, 1e-12))
    return cost, w


def bundle_adjust(poses, points, cam_i, pt_i, uv, intr: Intrinsics, fixed=(0,), free_intr=(), huber_px=1.5,
                  max_iterations=50, tol=1e-6, priors=None, distances=None) -> dict:
    """Minimise the robust reprojection error of observations uv (M, 2; distorted pixels) of points[pt_i] in
    poses[cam_i] (poses (C, 6) [rvec, t] camera <- world; points (P, 3)) — the Huber loss of scale huber_px on each
    observation's error — over the poses not in `fixed`, every point and the intrinsics named in free_intr ('f', 'k1',
    'k2').

    Levenberg–Marquardt with analytic derivatives, Huber by iteratively reweighted least squares and the points
    eliminated by the Schur complement (their 3 × 3 blocks inverted, the reduced camera-and-intrinsics system solved
    densely; scipy.sparse for the products). scipy.optimize.least_squares was tried first: with a Huber loss its
    trust-region solver (sparse Jacobian, LSMR) stalled on a synthetic 8-camera problem (58 → 40 px RMS in 100
    evaluations where the solution is 0.26 px), so this is the textbook solver instead (Triggs et al., *Bundle
    Adjustment — A Modern Synthesis*, 2000).

    priors: optional position priors on camera centres, (cams (K,), targets (K, 3), sigmas (K, 3)) — each adds
    (c − target)/σ per axis (σ = inf: that axis free) to the least squares, squared (no robust loss); the GPS-aided
    adjustment uses them. With a pose updated as [exp(ω)R | exp(ω)t + δt], the centre c = −Rᵀt changes by −Rᵀδt to
    first order (ω drops out), so a prior touches the translation block only.
    distances: optional priors on the distance between camera centres, (a (K,), b (K,), metres (K,), σ (K,)) —
    (|c_b − c_a| − d)/σ each; the dashcam's GPS speed over ground integrated between two frames gives them, which
    holds a monocular map's scale without using the GPS positions.

    Returns the new poses, points and intrinsics, the RMS and median error before and after (pixels), the residuals,
    the intrinsics' standard deviations when free (Gauss–Newton covariance: the inverse of the reduced system at the
    solution × the inliers' residual variance — a lower bound: correlated errors such as rolling shutter or a wrong
    lens model are not in it), iterations and seconds."""
    import scipy.linalg as sla
    import scipy.sparse as sp
    t0 = time.perf_counter()
    P = np.asarray(poses, float).reshape(-1, 6)
    R = rodrigues(P[:, :3])
    t = P[:, 3:].copy()
    X = np.asarray(points, float).reshape(-1, 3).copy()
    cam_i, pt_i = np.asarray(cam_i, np.int64), np.asarray(pt_i, np.int64)
    uv = np.asarray(uv, float).reshape(-1, 2)
    k = {'f': intr.f, 'k1': intr.k1, 'k2': intr.k2}
    fi = [INTR_KEYS.index(key) for key in INTR_KEYS if key in free_intr]
    n_c, n_p, m = len(P), len(X), len(cam_i)
    free = np.ones(n_c, bool)
    free[[j for j in fixed if j < n_c]] = False
    col = -np.ones(n_c, np.int64)
    col[free] = np.arange(int(free.sum())) * 6
    n_cam = 6 * int(free.sum())
    n_s = n_cam + len(fi)
    obs_free = col[cam_i] >= 0

    def errors(R, t, X, k):
        q, front, _ = _project_jac(R, t, X, cam_i, pt_i, k['f'], k['k1'], k['k2'], intr.cx, intr.cy, False)
        e = np.linalg.norm(q - uv, axis=1)
        return q - uv, np.where(front, e, 1e6), front

    if priors is not None:
        pc_, ptgt, psig = (np.asarray(priors[0], np.int64), np.asarray(priors[1], float),
                           np.asarray(priors[2], float))
        pw = np.where(np.isfinite(psig), 1.0 / np.where(np.isfinite(psig), psig, 1.0), 0.0)

    def prior_res(R, t):
        c = -np.einsum('kji,kj->ki', R[pc_], t[pc_])
        return (c - ptgt) * pw

    if distances is not None:
        da, db, dd, ds_ = (np.asarray(v) for v in distances)
        da, db = da.astype(np.int64), db.astype(np.int64)

    def dist_res(R, t):
        ca = -np.einsum('kji,kj->ki', R[da], t[da])
        cb = -np.einsum('kji,kj->ki', R[db], t[db])
        v = cb - ca
        L = np.linalg.norm(v, axis=1)
        return (L - dd) / ds_, v / np.maximum(L, 1e-12)[:, None]

    def robust_cost(R, t, X, k):
        _, e, front = errors(R, t, X, k)
        c = float(np.sum(_huber(e, huber_px)[0]))
        if priors is not None:
            c += float(np.sum(prior_res(R, t) ** 2))
        if distances is not None:
            c += float(np.sum(dist_res(R, t)[0] ** 2))
        return c

    def normal(R, t, X, k):
        """The weighted normal equations' blocks: U (n_s × n_s dense), W (n_s × 3P sparse), V's 3 × 3 blocks, g."""
        q, front, (Jc, JX, Ji) = _project_jac(R, t, X, cam_i, pt_i, k['f'], k['k1'], k['k2'], intr.cx, intr.cy)
        res = q - uv
        _, w = _huber(np.linalg.norm(res, axis=1), huber_px)
        sw = np.sqrt(np.where(front, w, 0.0))[:, None]
        rw = (res * sw).reshape(-1)
        rr, cc, vv = [], [], []
        base = (2 * np.arange(m))[:, None]
        for o in (0, 1):
            ro = base + o
            rr.append(np.repeat(ro[obs_free], 6, 1).reshape(-1))
            cc.append((col[cam_i[obs_free]][:, None] + np.arange(6)).reshape(-1))
            vv.append((Jc[obs_free, o, :] * sw[obs_free]).reshape(-1))
            if fi:
                rr.append(np.repeat(ro, len(fi), 1).reshape(-1))
                cc.append(np.tile(n_cam + np.arange(len(fi)), m))
                vv.append((Ji[:, o, fi] * sw).reshape(-1))
            rr.append(np.repeat(ro, 3, 1).reshape(-1))
            cc.append((n_s + 3 * pt_i[:, None] + np.arange(3)).reshape(-1))
            vv.append((JX[:, o, :] * sw).reshape(-1))
        J = sp.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                          shape=(2 * m, n_s + 3 * n_p))
        H = (J.T @ J).tocsr()
        g = J.T @ rw
        Vc = H[n_s:, n_s:].tocoo()
        Vb = np.zeros((n_p, 3, 3))
        Vb[Vc.row // 3, Vc.row % 3, Vc.col % 3] = Vc.data
        U = H[:n_s, :n_s].toarray()
        gs = np.array(g[:n_s])
        if priors is not None:
            r = prior_res(R, t)                                   # (K, 3)
            for kk, ci in enumerate(pc_):
                if col[ci] < 0:
                    continue
                Jt = -R[ci].T * pw[kk][:, None]                   # d(weighted residual) / d(δt)
                sl = slice(col[ci] + 3, col[ci] + 6)
                U[sl, sl] += Jt.T @ Jt
                gs[sl] += Jt.T @ r[kk]
        if distances is not None:
            r, u = dist_res(R, t)
            for kk in range(len(da)):
                parts = []
                for ci, sign in ((db[kk], 1.0), (da[kk], -1.0)):
                    if col[ci] >= 0:                              # d|c_b − c_a| / dδt = ±uᵀ(−Rᵀ)
                        parts.append((col[ci] + 3, sign * -(u[kk] @ R[ci].T) / ds_[kk]))
                for c1, j1 in parts:
                    gs[c1:c1 + 3] += j1 * r[kk]
                    for c2, j2 in parts:
                        U[c1:c1 + 3, c2:c2 + 3] += np.outer(j1, j2)
        return U, H[:n_s, n_s:].tocsr(), Vb, gs, g[n_s:]

    r0, e0, front0 = errors(R, t, X, k)
    rms_before = float(np.sqrt(np.mean(np.sum(r0[front0] ** 2, 1)))) if front0.any() else float('inf')
    med_before = float(np.median(e0))
    lam = 1e-3
    cost = robust_cost(R, t, X, k)
    it = 0
    for it in range(1, max_iterations + 1):
        U, Wm, Vb, gs, gp = normal(R, t, X, k)
        improved, rel = False, 0.0
        for _ in range(10):
            Vis = _block_diag(np.linalg.inv(Vb + lam * (Vb * np.eye(3)) + 1e-9 * np.eye(3)))
            if n_s:
                S = U + lam * np.diag(np.diag(U) + 1e-9) - (Wm @ Vis @ Wm.T).toarray()
                b = gs - Wm @ (Vis @ gp)
                try:
                    ds = -sla.cho_solve(sla.cho_factor(S), b)
                except (np.linalg.LinAlgError, sla.LinAlgError):
                    ds = -np.linalg.lstsq(S, b, rcond=None)[0]
                dp = -(Vis @ (gp + Wm.T @ ds))
            else:
                ds = np.zeros(0)
                dp = -(Vis @ gp)
            R2, t2, k2 = R.copy(), t.copy(), dict(k)
            dc = ds[:n_cam].reshape(-1, 6)
            dR = rodrigues(dc[:, :3])
            R2[free] = dR @ R[free]
            t2[free] = np.einsum('nij,nj->ni', dR, t[free]) + dc[:, 3:]
            for j, kk in enumerate(fi):
                k2[INTR_KEYS[kk]] = k[INTR_KEYS[kk]] + ds[n_cam + j]
            X2 = X + dp.reshape(-1, 3)
            c2 = robust_cost(R2, t2, X2, k2)
            if c2 < cost:
                rel = (cost - c2) / max(cost, 1e-12)
                R, t, X, k, cost = R2, t2, X2, k2, c2
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 5
        if not improved or rel < tol:
            break
    r1, e1, front1 = errors(R, t, X, k)
    out_p = np.c_[np.array([rvec_of(Rk) for Rk in R]).reshape(-1, 3), t]
    out = {'poses': out_p, 'points': X, 'intr': Intrinsics(intr.width, intr.height, k['f'], k['k1'], k['k2'],
                                                             intr.cx, intr.cy),
           'rms_before': rms_before, 'rms_after': float(np.sqrt(np.mean(np.sum(r1[front1] ** 2, 1)))),
           'median_before': med_before, 'median_after': float(np.median(e1)), 'residuals': r1,
           'iterations': it, 'nfev': it, 'seconds': time.perf_counter() - t0, 'observations': int(m),
           'cameras': int(free.sum()), 'points_n': int(n_p)}
    if fi:
        U, Wm, Vb, _, _ = normal(R, t, X, k)
        S = U - (Wm @ _block_diag(np.linalg.inv(Vb + 1e-9 * np.eye(3))) @ Wm.T).toarray()
        out['intr_std'] = _intr_std(S, fi, n_cam, e1, front1, huber_px, n_s + 3 * n_p)
    return out


def _block_diag(B) -> 'object':
    """A CSR matrix of 3 × 3 blocks on the diagonal (fast for many blocks)."""
    import scipy.sparse as sp
    n = len(B)
    rows = (3 * np.arange(n)[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
    cols = (3 * np.arange(n)[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1)
    return sp.csr_matrix((B.reshape(-1), (rows.reshape(-1), cols.reshape(-1))), shape=(3 * n, 3 * n))


def _intr_std(S, fi, n_cam, e, front, huber_px, n_params) -> dict:
    keys = [INTR_KEYS[j] for j in fi]
    if S is None:
        return {k: None for k in keys}
    inl = front & (e < 3 * huber_px)
    dof = max(1, 2 * int(inl.sum()) - n_params)
    s2 = float(np.sum(e[inl] ** 2) / dof)
    try:
        C = np.linalg.inv(S)
    except np.linalg.LinAlgError as ex:
        return {k: None for k in keys} | {'error': str(ex)}
    return {key: float(math.sqrt(max(C[n_cam + j, n_cam + j], 0.0) * s2)) for j, key in enumerate(keys)}



# ================================================================================================ the odometry
@dataclass
class VOResult:
    poses: dict                   # frame -> pose vector [rvec, t] (camera <- world), VO units
    points: dict                  # track id -> (3,) world point
    keyframes: list
    stats: dict
    lost_at: int | None = None
    times: object = None          # the frames' times (s), when a distance prior is used
    distance: object = None       # (t_a, t_b) -> (metres, σ): the distance travelled between two times, or None

    def centres(self, frames=None) -> np.ndarray:
        fs = sorted(self.poses) if frames is None else frames
        return np.array([centre_of(self.poses[f]) for f in fs])

    def point_array(self):
        ids = np.array(sorted(self.points), np.int64)
        return ids, (np.array([self.points[i] for i in ids]) if len(ids) else np.zeros((0, 3)))


def _pnp(Xw, xn, f, guess=None, px=2.0):
    import cv2
    if len(Xw) < 12:
        return None, None
    ok, rv, tv, inl = cv2.solvePnPRansac(Xw.astype(np.float64), xn.astype(np.float64), np.eye(3), None,
                                         rvec=None if guess is None else guess[:3].copy().reshape(3, 1),
                                         tvec=None if guess is None else guess[3:].copy().reshape(3, 1),
                                         useExtrinsicGuess=guess is not None, iterationsCount=200,
                                         reprojectionError=px / f, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 10:
        return None, None
    inl = inl.reshape(-1)
    rv, tv = cv2.solvePnPRefineLM(Xw[inl].astype(np.float64), xn[inl].astype(np.float64), np.eye(3), None, rv, tv)
    return np.r_[rv.reshape(3), tv.reshape(3)], inl


def run_vo(tracks: Tracks, intr: Intrinsics, settings: dict | None = None, emitter=None, cancel=None, times=None,
           distance=None) -> VOResult:
    """Keyframe VO over the tracks (module doc). Stops (lost_at) if a frame cannot be placed by PnP.

    With `times` and `distance` ((t_a, t_b) -> (metres, σ): e.g. the GPS speed over ground integrated), the map is
    metric from the start (the initial baseline is that distance) and every bundle adjustment holds the distances
    between consecutive keyframes to it (bundle_adjust's distance priors): the scale no longer drifts freely. The
    GPS positions are not used, so the alignment's residuals stay an independent check."""
    import cv2
    s = dict(DEFAULTS, **(settings or {}))
    W, H = tracks.size
    kf_px = s['kf_px'] or 0.02 * W
    init_px = s['init_px'] or 0.01 * W
    n = len(tracks)
    norm = [intr.normalise(u) for u in tracks.uv]
    look = [tracks.lookup(i) for i in range(n)]
    poses, points = {}, {}
    kfs = []
    st = {'local_ba': [], 'pnp_inliers': [], 'triangulated': 0, 'init': None, 'outliers_removed': 0, 'gated_out': 0,
          'scale_overrides': 0}
    gate_fail = {}
    t0 = time.perf_counter()

    def common(i, j):
        a, b = look[i], look[j]
        ids = np.array(sorted(set(a) & set(b)), np.int64)
        return ids, np.array([a[k] for k in ids], np.int64), np.array([b[k] for k in ids], np.int64)

    # ---- initialisation
    start = 0
    init_ok = False
    for j in range(1, n):
        ids, ia, ib = common(start, j)
        if len(ids) < s['init_min_points']:
            start = j - 1 if j - 1 > start else start
            continue
        disp = np.median(np.linalg.norm(tracks.uv[j][ib] - tracks.uv[start][ia], axis=1))
        if disp < init_px:
            continue
        a, b = norm[start][ia], norm[j][ib]
        E, inl = cv2.findEssentialMat(a, b, np.eye(3), cv2.RANSAC, 0.999, s['epi_px'] / intr.f)
        if E is None or E.shape != (3, 3):
            continue
        _, R, t, inl2 = cv2.recoverPose(E, a, b, np.eye(3), mask=inl.copy())
        keep = inl2.reshape(-1) > 0
        T0, T1 = np.eye(4), np.eye(4)
        T1[:3, :3], T1[:3, 3] = R, t.reshape(3)
        X = triangulate(T0, T1, a[keep], b[keep])
        e0, z0 = reproj_px(T0, X, a[keep], intr.f)
        e1, z1 = reproj_px(T1, X, b[keep], intr.f)
        ang = ray_angle_deg(X, np.zeros(3), -R.T @ t.reshape(3))
        good = (z0 > 0) & (z1 > 0) & (e0 < s['tri_px']) & (e1 < s['tri_px']) & (ang > s['min_angle_deg'])
        if good.sum() < s['init_min_points'] or keep.sum() < 0.5 * len(ids) or \
                np.median(ang[good]) < s['init_angle_deg']:
            continue
        sc = 1.0
        if distance is not None:
            sc = max(distance(times[start], times[j])[0], 1e-3)
        poses[start], poses[j] = pose_vec(np.eye(3), np.zeros(3)), pose_vec(R, sc * t.reshape(3))
        for tid, x in zip(ids[keep][good], X[good]):
            points[int(tid)] = sc * x
        kfs = [start, j]
        st['init'] = {'frames': [start, j], 'points': int(good.sum()), 'median_angle_deg':
                      round(float(np.median(ang[good])), 3), 'median_disp_px': round(float(disp), 2)}
        init_ok = True
        break
    if not init_ok:
        raise VOError('could not initialise: no frame pair with enough parallax and tracks (is the camera moving?)')
    # frames between start and the second keyframe: PnP later
    lost = None
    last_kf_mapped = len(points)

    def local_ba():
        win = kfs[-s['window']:]
        idx = {f: k for k, f in enumerate(win)}
        obs_c, obs_p, obs_uv, pids = [], [], [], {}
        for f in win:
            for tid, r in look[f].items():
                if tid in points:
                    pids.setdefault(tid, len(pids))
                    obs_c.append(idx[f])
                    obs_p.append(pids[tid])
                    obs_uv.append(tracks.uv[f][r])
        if not obs_c:
            return
        obs_c, obs_p, obs_uv = np.array(obs_c), np.array(obs_p), np.array(obs_uv)
        cnt = np.bincount(obs_p, minlength=len(pids))
        keep = cnt[obs_p] >= 2
        if keep.sum() < 20:
            return
        P0 = np.array([poses[f] for f in win])
        rev = {v: k for k, v in pids.items()}
        X0 = np.array([points[rev[k]] for k in range(len(pids))])
        fixed = (0, 1) if len(win) > 2 and distance is None else (0,)
        r = bundle_adjust(P0, X0, obs_c[keep], obs_p[keep], obs_uv[keep], intr, fixed, (), s['huber_px'],
                          s['local_ba_iterations'], distances=_dist_priors(win, times, distance))
        for k, f in enumerate(win):
            poses[f] = r['poses'][k]
        for k in range(len(pids)):
            if cnt[k] >= 2:
                points[rev[k]] = r['points'][k]
        err = np.linalg.norm(r['residuals'], axis=1)
        bad = {rev[int(p)] for p in obs_p[keep][err > s['outlier_px']]}
        for tid in bad:
            points.pop(tid, None)
        st['outliers_removed'] += len(bad)
        st['local_ba'].append({'kf': win[-1], 'rms_before': round(r['rms_before'], 4),
                               'rms_after': round(r['rms_after'], 4), 'seconds': round(r['seconds'], 4),
                               'observations': r['observations'], 'nfev': r['nfev']})

    def triangulate_new(k):
        nonlocal last_kf_mapped
        win = kfs[-s['window']:-1]
        Tk = pose_mat(poses[k])
        ck = centre_of(poses[k])
        cand = [tid for tid in look[k] if tid not in points]
        added = 0
        by_kf = {}
        for tid in cand:
            for f in win:
                if tid in look[f]:
                    by_kf.setdefault(f, []).append(tid)
                    break
        for f, tids in by_kf.items():
            Tf = pose_mat(poses[f])
            a = np.array([norm[f][look[f][t]] for t in tids])
            b = np.array([norm[k][look[k][t]] for t in tids])
            X = triangulate(Tf, Tk, a, b)
            e0, z0 = reproj_px(Tf, X, a, intr.f)
            e1, z1 = reproj_px(Tk, X, b, intr.f)
            base = np.linalg.norm(ck - centre_of(poses[f]))
            ang = ray_angle_deg(X, centre_of(poses[f]), ck)
            good = (z0 > 0) & (z1 > 0) & (e0 < s['tri_px']) & (e1 < s['tri_px']) & (ang > s['min_angle_deg']) & \
                (z1 < s['max_depth_ratio'] * max(base, 1e-9))
            for tid, x in zip(np.array(tids)[good], X[good]):
                points[int(tid)] = x
            added += int(good.sum())
        st['triangulated'] += added

    for i in range(kfs[1] + 1, n):
        if cancel and cancel():
            raise KeyboardInterrupt()
        have = [(tid, r) for tid, r in look[i].items() if tid in points]
        if len(have) < 12:
            lost = i
            break
        Xw = np.array([points[t] for t, _ in have])
        xn = np.array([norm[i][r] for _, r in have])
        guess = poses[i - 1] if i - 1 in poses else None
        pred = None
        if distance is not None and i - 1 in poses and i - 2 in poses:
            pred = _predict(poses[i - 2], poses[i - 1], distance(times[i - 1], times[i])[0])
            err, z = reproj_px(pose_mat(pred), Xw, xn, intr.f)
            ok = (err < s['gate_px']) & (z > 0)
            for (tid, _), g in zip(have, ok):
                if not g:
                    gate_fail[tid] = gate_fail.get(tid, 0) + 1
                    if gate_fail[tid] >= 3:                      # moving with the traffic, or badly placed
                        points.pop(tid, None)
                        st['gated_out'] += 1
            if ok.sum() >= 20:
                Xw, xn = Xw[ok], xn[ok]
            guess = pred
        p, inl = _pnp(Xw, xn, intr.f, guess, s['pnp_px'])
        if p is None:
            lost = i
            break
        if pred is not None:
            d = distance(times[i - 1], times[i])[0]
            step = np.linalg.norm(centre_of(p) - centre_of(poses[i - 1]))
            if d > 0.05 and not 0.6 < step / d < 1.6:      # PnP disagrees with the GPS speed: keep its rotation only
                R = rodrigues(p[:3])[0]
                p = np.r_[p[:3], -R @ centre_of(pred)]
                st['scale_overrides'] += 1
        poses[i] = p
        st['pnp_inliers'].append(len(inl))
        last = kfs[-1]
        ids, ia, ib = common(last, i)
        disp = np.median(np.linalg.norm(tracks.uv[i][ib] - tracks.uv[last][ia], axis=1)) if len(ids) else 1e9
        mapped_now = len(have)
        if disp > kf_px or mapped_now < s['kf_track_ratio'] * last_kf_mapped or i - last >= s['max_kf_gap']:
            if disp < 0.25 * kf_px and mapped_now >= s['kf_track_ratio'] * last_kf_mapped:
                continue                          # standing still: no keyframe however long it lasts
            kfs.append(i)
            triangulate_new(i)
            local_ba()
            last_kf_mapped = sum(1 for t in look[i] if t in points)
            if emitter:
                emitter.progress(i / n, f'visual odometry: frame {i} of {n}, {len(kfs)} keyframes, {len(points)} '
                                        f'points')
    # frames before the second keyframe that were not placed
    for i in range(kfs[0] + 1, kfs[1]):
        have = [(tid, r) for tid, r in look[i].items() if tid in points]
        if len(have) >= 12:
            p, _ = _pnp(np.array([points[t] for t, _ in have]), np.array([norm[i][r] for _, r in have]), intr.f,
                        None, s['pnp_px'])
            if p is not None:
                poses[i] = p
    st['keyframes'] = len(kfs)
    st['points'] = len(points)
    st['frames_placed'] = len(poses)
    st['seconds'] = round(time.perf_counter() - t0, 3)
    st['local_ba_seconds'] = round(sum(b['seconds'] for b in st['local_ba']), 3)
    st['mean_pnp_inliers'] = round(float(np.mean(st['pnp_inliers'])), 1) if st['pnp_inliers'] else 0
    del st['pnp_inliers']
    return VOResult(poses, points, kfs, st, lost, None if times is None else np.asarray(times, float), distance)


def _predict(p2, p1, d: float) -> np.ndarray:
    """The next pose by constant velocity from the two before it, the step rescaled to d metres."""
    T2, T1 = pose_mat(p2), pose_mat(p1)
    Tp = (T1 @ np.linalg.inv(T2)) @ T1
    c1 = -T1[:3, :3].T @ T1[:3, 3]
    cp = -Tp[:3, :3].T @ Tp[:3, 3]
    v = cp - c1
    L = np.linalg.norm(v)
    c = c1 + (v / L * d if L > 1e-9 else np.zeros(3))
    return np.r_[rvec_of(Tp[:3, :3]), -Tp[:3, :3] @ c]


def _dist_priors(frames, times, distance):
    """Distance priors between consecutive frames of a list (indices into it), or None."""
    if distance is None or times is None or len(frames) < 2:
        return None
    a = np.arange(len(frames) - 1)
    d = [distance(times[frames[k]], times[frames[k + 1]]) for k in a]
    return a, a + 1, np.array([v[0] for v in d]), np.array([max(v[1], 1e-3) for v in d])


def observations(vo: VOResult, tracks: Tracks, frames=None, min_obs: int = 2):
    """(frame list, cam_i, pt ids, pt_i, uv) of every mapped track seen in the given frames (default: keyframes)."""
    frames = list(vo.keyframes if frames is None else frames)
    cam_i, tids, uv = [], [], []
    for k, f in enumerate(frames):
        for tid, r in tracks.lookup(f).items():
            if tid in vo.points:
                cam_i.append(k)
                tids.append(tid)
                uv.append(tracks.uv[f][r])
    cam_i, tids, uv = np.array(cam_i, np.int64), np.array(tids, np.int64), np.array(uv).reshape(-1, 2)
    uniq, pt_i, cnt = np.unique(tids, return_inverse=True, return_counts=True)
    keep = cnt[pt_i] >= min_obs
    uniq2, pt_i2 = np.unique(tids[keep], return_inverse=True)
    return frames, cam_i[keep], uniq2, pt_i2.reshape(-1), uv[keep]


def global_ba(vo: VOResult, tracks: Tracks, intr: Intrinsics, free=(), settings: dict | None = None,
              rounds: int = 2) -> dict:
    """Bundle adjustment of every keyframe and point (the first two keyframes fixed), optionally with intrinsics free;
    observations above outlier_px are dropped between rounds. Updates vo in place (keyframe poses and points); the
    other frames are re-placed by PnP against the adjusted points with the adjusted intrinsics."""
    s = dict(DEFAULTS, **(settings or {}))
    out = {'rounds': []}
    cur = intr
    drop = set()
    t0 = time.perf_counter()
    for rd in range(rounds):
        frames, cam_i, tids, pt_i, uv = observations(vo, tracks)
        if drop:
            keep = np.array([(int(f), int(t)) not in drop for f, t in zip(np.array(frames)[cam_i], tids[pt_i])])
            frames, cam_i, tids, pt_i, uv = _refilter(frames, cam_i[keep], tids, pt_i[keep], uv[keep])
        P0 = np.array([vo.poses[f] for f in frames])
        X0 = np.array([vo.points[int(t)] for t in tids])
        r = bundle_adjust(P0, X0, cam_i, pt_i, uv, cur, (0,) if vo.distance else (0, 1), free, s['huber_px'],
                          s['global_ba_iterations'], distances=_dist_priors(frames, vo.times, vo.distance))
        for k, f in enumerate(frames):
            vo.poses[f] = r['poses'][k]
        for k, t in enumerate(tids):
            vo.points[int(t)] = r['points'][k]
        cur = r['intr']
        err = np.linalg.norm(r['residuals'], axis=1)
        for k in np.nonzero(err > s['outlier_px'])[0]:
            drop.add((int(frames[cam_i[k]]), int(tids[pt_i[k]])))
        row = {k: r[k] for k in ('rms_before', 'rms_after', 'median_before', 'median_after', 'seconds', 'nfev',
                                 'observations', 'cameras', 'points_n')}
        row['intr'] = cur.doc()
        if 'intr_std' in r:
            row['intr_std'] = r['intr_std']
        out['rounds'].append(row)
    # points whose every observation was dropped would be unconstrained: forget them
    seen = {}
    for (f, t) in drop:
        seen[t] = seen.get(t, 0) + 1
    out['dropped_observations'] = len(drop)
    out['intr'] = cur
    out['intr_std'] = out['rounds'][-1].get('intr_std')
    out['seconds'] = time.perf_counter() - t0
    replace_frames(vo, tracks, cur, s)
    return out


def _refilter(frames, cam_i, tids, pt_i, uv):
    used = np.unique(pt_i)
    remap = -np.ones(len(tids), np.int64)
    remap[used] = np.arange(len(used))
    pt2 = remap[pt_i]
    cnt = np.bincount(pt2, minlength=len(used))
    keep = cnt[pt2] >= 2
    used2 = np.unique(pt2[keep])
    remap2 = -np.ones(len(used), np.int64)
    remap2[used2] = np.arange(len(used2))
    return frames, cam_i[keep], tids[used][used2], remap2[pt2[keep]], uv[keep]


def interpolated_pose(vo: VOResult, i: int) -> np.ndarray | None:
    """Frame i's pose interpolated between the keyframes around it (centre linearly, rotation by slerp, in frame
    index — the frames are evenly spaced in time), or None outside the keyframes."""
    from scipy.spatial.transform import Rotation, Slerp
    kf = sorted(vo.keyframes)
    j = int(np.searchsorted(kf, i))
    if j == 0 or j >= len(kf):
        return None
    a, b = kf[j - 1], kf[j]
    w = (i - a) / (b - a)
    Ra, Rb = rodrigues(vo.poses[a][:3])[0], rodrigues(vo.poses[b][:3])[0]
    R = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))([w]).as_matrix()[0]
    c = (1 - w) * centre_of(vo.poses[a]) + w * centre_of(vo.poses[b])
    return np.r_[rvec_of(R), -R @ c]


def replace_frames(vo: VOResult, tracks: Tracks, intr: Intrinsics, settings: dict | None = None) -> dict:
    """Re-place every frame between keyframes by PnP against the map's points, starting from its pose interpolated
    between the keyframes around it; the PnP pose is kept when it has at least 30 inliers, a median reprojection error
    no larger than the interpolation's and lies within max_dev_rel × the keyframe step and max_dev_deg of it (a frame
    with few or wrong points otherwise jumps), else the interpolation is used. Frames outside the keyframes keep
    their pose from the odometry. Returns counts."""
    s = dict(DEFAULTS, **(settings or {}))
    kf = set(vo.keyframes)
    out = {'pnp': 0, 'interpolated': 0}
    for i in sorted(vo.poses):
        if i in kf:
            continue
        ip = interpolated_pose(vo, i)
        if ip is None:
            continue
        lk = tracks.lookup(i)
        have = [(t, r) for t, r in lk.items() if t in vo.points]
        p = None
        if len(have) >= 12:
            Xw = np.array([vo.points[t] for t, _ in have])
            xn = intr.normalise(np.array([tracks.uv[i][r] for _, r in have]))
            p, inl = _pnp(Xw, xn, intr.f, ip, s['pnp_px'])
            if p is not None:
                dc = np.linalg.norm(centre_of(p) - centre_of(ip))
                dr = math.degrees(np.linalg.norm(rvec_of(rodrigues(p[:3])[0] @ rodrigues(ip[:3])[0].T)))
                e_p = np.median(reproj_px(pose_mat(p), Xw, xn, intr.f)[0])
                e_i = np.median(reproj_px(pose_mat(ip), Xw, xn, intr.f)[0])
                # PnP when it explains the frame's tracks better and stays near the keyframes' path
                if len(inl) >= 30 and e_p <= e_i and dc <= s['max_dev_rel'] * _kf_step(vo, i) and \
                        dr <= s['max_dev_deg']:
                    vo.poses[i] = p
                    out['pnp'] += 1
                    continue
        vo.poses[i] = ip
        out['interpolated'] += 1
    return out


def _kf_step(vo: VOResult, i: int) -> float:
    """The length of the keyframe step around frame i (in the map's units), for scale-free thresholds."""
    kf = sorted(vo.keyframes)
    j = int(np.clip(np.searchsorted(kf, i), 1, len(kf) - 1))
    return float(np.linalg.norm(centre_of(vo.poses[kf[j]]) - centre_of(vo.poses[kf[j - 1]])))


def reprojection_rms(vo: VOResult, tracks: Tracks, intr: Intrinsics, frames=None) -> float:
    frames, cam_i, tids, pt_i, uv = observations(vo, tracks, frames, 1)
    P = np.array([vo.poses[f] for f in frames])
    X = np.array([vo.points[int(t)] for t in tids])
    R = rodrigues(P[:, :3])
    pc = np.einsum('mij,mj->mi', R[cam_i], X[pt_i]) + P[cam_i, 3:]
    ok = pc[:, 2] > 1e-6
    q = intr.project_normalised(pc[ok, :2] / pc[ok, 2:3])
    return float(np.sqrt(np.mean(np.sum((q - uv[ok]) ** 2, 1))))


def reprojection_stats(vo: VOResult, tracks: Tracks, intr: Intrinsics, frames=None, inlier_px: float = 4.0) -> dict:
    """The reprojection error of every mapped track in the frames: median, RMS over the inliers (below inlier_px)
    and the inliers' share — the all-observations RMS is dominated by the tracks on moving traffic."""
    frames, cam_i, tids, pt_i, uv = observations(vo, tracks, frames, 1)
    P = np.array([vo.poses[f] for f in frames])
    X = np.array([vo.points[int(t)] for t in tids])
    R = rodrigues(P[:, :3])
    pc = np.einsum('mij,mj->mi', R[cam_i], X[pt_i]) + P[cam_i, 3:]
    ok = pc[:, 2] > 1e-6
    e = np.full(len(pc), np.inf)
    e[ok] = np.linalg.norm(intr.project_normalised(pc[ok, :2] / pc[ok, 2:3]) - uv[ok], axis=1)
    inl = e < inlier_px
    return {'median_px': round(float(np.median(e)), 4), 'inlier_rms_px': round(float(np.sqrt(np.mean(e[inl] ** 2))), 4)
            if inl.any() else None, 'inlier_share': round(float(inl.mean()), 4), 'observations': int(len(e)),
            'inlier_px': inlier_px}


# ================================================================================================ intrinsics
def focal_profile(tracks: Tracks, hfovs, settings: dict | None = None, k1: float = 0.0) -> list:
    """The focal search: VO and a global BA (intrinsics fixed) for each horizontal field in hfovs; the final robust
    reprojection error per candidate. Its minimum is an estimate of f and its curvature a measure of how well the
    sequence determines f (flat: it does not — a straight drive with no rotation leaves f unobservable)."""
    out = []
    W, H = tracks.size
    for h in hfovs:
        intr = Intrinsics.from_hfov(W, H, float(h), k1)
        try:
            vo = run_vo(tracks, intr, settings)
            g = global_ba(vo, tracks, intr, (), settings, rounds=1)
            out.append({'hfov_deg': float(h), 'f': intr.f, 'rms': g['rounds'][-1]['rms_after'],
                        'median': g['rounds'][-1]['median_after'], 'frames': len(vo.poses)})
        except VOError as e:
            out.append({'hfov_deg': float(h), 'f': intr.f, 'rms': None, 'error': str(e)})
    return out


def self_calibrate(tracks: Tracks, start: Intrinsics, free=('f', 'k1'), rounds: int = 4,
                   settings: dict | None = None, emitter=None) -> tuple:
    """Self-calibration: VO with the current intrinsics, then a global BA with `free` intrinsics; repeated (the VO's
    structure depends on the intrinsics it was run with) until f changes by less than 0.2 % or `rounds` are done.
    Returns (intrinsics, history). On synthetic streets (tests) it lands within ±3 % of the true f from a start 15 %
    off; its formal standard deviation is far smaller than that (see bundle_adjust), so the caller reports the spread
    over independent segments as the uncertainty."""
    cur = start
    hist = []
    for r in range(rounds):
        if emitter:
            emitter.progress(0.0, f'self-calibration round {r + 1}: f = {cur.f:.1f} px, k1 = {cur.k1:+.4f}', force=True)
        vo = run_vo(tracks, cur, settings)
        g = global_ba(vo, tracks, cur, free, settings)
        new = g['intr']
        hist.append({'round': r + 1, 'start_f': round(cur.f, 3), 'intr': new.doc(), 'formal_std': g['intr_std'],
                     'rms_px': round(g['rounds'][-1]['rms_after'], 4),
                     'median_px': round(g['rounds'][-1]['median_after'], 4), 'frames': len(vo.poses),
                     'keyframes': len(vo.keyframes), 'seconds': round(g['seconds'] + vo.stats['seconds'], 2)})
        done = abs(new.f - cur.f) < 0.002 * cur.f
        cur = new
        if done:
            break
    return cur, hist


def plumb_line(grays, mask, intr: Intrinsics, k1_grid=None, every: int = 20, min_points: int = 60,
               straight_px: float = 0.6, seed: int = 0) -> dict:
    """The radial distortion k1 (at intr's f and principal point) that makes image edges straight (the plumb-line
    method, Brown 1971): Canny edge chains of at least min_points pixels from every `every`-th frame; a chain counts
    when, for some k1 of the grid, its undistorted points fit a line within straight_px (RMS) and its residual
    depends on k1 (it is off the centre); the estimate minimises the sum of the chains' residuals (capped at 3 px).
    Its spread is a bootstrap over the chains (200 draws). Curved things in the world (a curving kerb, a bent lamp
    post) that happen to straighten for some k1 bias it: a heuristic estimate."""
    import cv2
    k1_grid = np.linspace(-0.3, 0.15, 46) if k1_grid is None else np.asarray(k1_grid, float)
    # only models that do not fold the image back within its corners (a strong barrel term at a long focal length
    # would): beyond that radius the undistortion is clamped and a chain squeezed onto it looks straight
    corner = math.hypot(max(intr.cx, intr.width - 1 - intr.cx), max(intr.cy, intr.height - 1 - intr.cy)) / intr.f
    valid = []
    for k1 in k1_grid:
        rm = Intrinsics(intr.width, intr.height, intr.f, float(k1)).monotonic_radius()
        valid.append(rm == float('inf') or rm * (1 + k1 * rm * rm) > 1.02 * corner)
    k1_grid = k1_grid[np.array(valid, bool)]
    m8 = None if mask is None else np.asarray(mask, bool)
    chains = []
    for g in grays[::max(1, every)]:
        e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.2), 40, 120)
        if m8 is not None:
            e[~m8] = 0
        cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            q = c.reshape(-1, 2).astype(float)
            q = q[:len(q) // 2]                     # a thin edge's contour runs out and back
            if len(q) >= min_points and np.ptp(q, 0).max() >= min_points:
                chains.append(q[::2])
    out = {'chains': len(chains), 'grid': [float(k1_grid[0]), float(k1_grid[-1]), len(k1_grid)]}
    if len(chains) < 10:
        out.update(k1=None, why='too few long edges')
        return out
    R = np.zeros((len(chains), len(k1_grid)))
    for j, k1 in enumerate(k1_grid):
        it = Intrinsics(intr.width, intr.height, intr.f, float(k1), 0.0, intr.cx, intr.cy)
        for i, q in enumerate(chains):
            x = it.normalise(q) * it.f
            sv = np.linalg.svd(x - x.mean(0), compute_uv=False)
            R[i, j] = sv[-1] / math.sqrt(len(x))
    sel = (R.min(1) < straight_px) & (R.max(1) - R.min(1) > 0.5)
    if sel.sum() < 10:
        out.update(k1=None, why='too few straight off-centre edges', used=int(sel.sum()))
        return out
    cost = np.minimum(R[sel], 3.0).sum(0)
    j = int(np.argmin(cost))
    rng = np.random.default_rng(seed)
    idx = np.nonzero(sel)[0]
    boot = [k1_grid[int(np.argmin(np.minimum(R[rng.choice(idx, len(idx))], 3.0).sum(0)))] for _ in range(200)]
    out.update(k1=float(k1_grid[j]), std=float(np.std(boot)), used=int(sel.sum()),
               at_grid_edge=j in (0, len(cost) - 1),
               cost_at_0=float(cost[int(np.argmin(np.abs(k1_grid)))]), cost_min=float(cost[j]))
    return out


# ================================================================================================ alignment
def umeyama(src, dst, w=None, scale: bool = True) -> tuple:
    """The similarity dst ≈ s·R·src + t minimising the weighted squared error (Umeyama 1991): (s, R, t)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    w = w / max(w.sum(), 1e-12)
    ms, md = w @ src, w @ dst
    a, b = src - ms, dst - md
    C = (b * w[:, None]).T @ a
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var = float(np.sum(w * np.sum(a * a, 1)))
    s = float(np.trace(np.diag(D) @ S) / var) if scale and var > 1e-12 else 1.0
    t = md - s * R @ ms
    return s, R, t


def robust_similarity(src, dst, scale: bool = True, c: float = 2.0, iters: int = 30, trials: int = 200,
                      seed: int = 0) -> dict:
    """A similarity robust to outliers: RANSAC over minimal pairs (the most points within 3c), then iteratively
    reweighted least squares with Huber weights of scale c (in dst's units)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    rng = np.random.default_rng(seed)
    best, best_n = umeyama(src, dst, None, scale), -1
    if n >= 4:
        for _ in range(trials):
            idx = rng.choice(n, 2 if src.shape[1] == 2 else 3, replace=False)
            if np.linalg.norm(src[idx[0]] - src[idx[1]]) < 1e-9:
                continue
            m = umeyama(src[idx], dst[idx], None, scale)
            e = np.linalg.norm(m[0] * src @ m[1].T + m[2] - dst, axis=1)
            k = int((e < 3 * c).sum())
            if k > best_n:
                best, best_n = m, k
    s, R, t = best
    w = np.ones(n)
    for _ in range(iters):
        e = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        w = np.where(e <= c, 1.0, c / np.maximum(e, 1e-12))
        w[e > 6 * c] = 0.0
        if w.sum() < 2:
            w = np.ones(n)
        s, R, t = umeyama(src, dst, w, scale)
    e = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
    return {'s': s, 'R': R, 't': t, 'residuals': e, 'weights': w}


def level_rotation(up) -> np.ndarray:
    """The rotation taking the unit vector `up` to +z (the shortest one)."""
    u = np.asarray(up, float) / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    c = float(u @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    Vx = skew(v)
    return np.eye(3) + Vx + Vx @ Vx * (1 / (1 + c))


def ground_plane(points, centres, down, thresh_rel: float = 0.01, max_tilt_deg: float = 25.0, trials: int = 400,
                 seed: int = 0) -> dict | None:
    """The road's plane among the map points (RANSAC): a plane whose normal is within max_tilt_deg of the cameras'
    mean `down` direction (reversed), lying below every camera, with the most points within thresh_rel × the median
    camera-to-point distance. Returns {'normal' (up), 'd' (n·x + d = 0), 'height' (the cameras' mean height above it,
    in the map's units), 'inliers'} or None."""
    X = np.asarray(points, float)
    C = np.asarray(centres, float)
    if len(X) < 30:
        return None
    up0 = -np.asarray(down, float) / np.linalg.norm(down)
    rel = X - C.mean(0)
    below = rel @ up0 < 0
    Xb = X[below]
    if len(Xb) < 30:
        return None
    scale = float(np.median(np.linalg.norm(rel, axis=1)))
    th = thresh_rel * scale
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    cos_max = math.cos(math.radians(max_tilt_deg))
    for _ in range(trials):
        p = Xb[rng.choice(len(Xb), 3, replace=False)]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        L = np.linalg.norm(nrm)
        if L < 1e-12:
            continue
        nrm /= L
        if nrm @ up0 < 0:
            nrm = -nrm
        if nrm @ up0 < cos_max:
            continue
        d = -nrm @ p[0]
        h = C @ nrm + d
        if (h <= 0).any():
            continue
        k = int((np.abs(Xb @ nrm + d) < th).sum())
        if k > best_n:
            best, best_n = (nrm, d), k
    if best is None:
        return None
    nrm, d = best
    inl = Xb[np.abs(Xb @ nrm + d) < th]
    c0 = inl.mean(0)                                    # least-squares refit on the inliers
    _, _, Vt = np.linalg.svd(inl - c0)
    n2 = Vt[-1] if Vt[-1] @ nrm > 0 else -Vt[-1]
    d2 = -n2 @ c0
    return {'normal': n2, 'd': float(d2), 'height': float(np.mean(C @ n2 + d2)),
            'height_std': float(np.std(C @ n2 + d2)), 'inliers': int(len(inl)), 'candidates': int(len(Xb))}


def align_to_gps(centres, times, gps_t, gps_xy, up, lags=None, c: float = 2.0, seed: int = 0,
                 lag_gain: float = 1.25, lag_min_gain_m: float = 0.5) -> dict:
    """The similarity world ← VO from the camera centres (N, 3, VO units, at `times` s) and the GPS fixes (gps_xy
    (M, 2) metres east/north at gps_t s on the video's clock): the VO frame is levelled (`up` → +z), then a robust 2D
    similarity (scale, heading, offset; robust_similarity, Huber scale c metres) maps the levelled centres —
    interpolated at each fix's time + lag — onto the fixes, for every lag in `lags` (s, default ±1 s); the lag with
    the smallest median residual wins when it beats no lag by a factor lag_gain and by lag_min_gain_m metres (the
    GPS's latency against the video is not known, and on a straight road at constant speed — or an arc — a latency
    is indistinguishable from an offset along the track, so the fit alone cannot be trusted with it; measured on a
    synthetic straight drive, float32 GPS quantisation alone moved the best lag to 0.6 s). Returns s, R (3 × 3,
    world ← VO), t (z left 0), the lag (and every lag's median), each fix's residual and their RMS / median / 90th
    percentile / maximum (metres)."""
    C = np.asarray(centres, float)
    times = np.asarray(times, float)
    gt, gxy = np.asarray(gps_t, float), np.asarray(gps_xy, float)
    Rl = level_rotation(up)
    L = C @ Rl.T
    lags = np.round(np.arange(-1.0, 1.01, 0.1), 3) if lags is None else np.asarray(lags, float)
    fits = []
    for lag in lags:
        tt = gt + lag
        ok = (tt >= times[0]) & (tt <= times[-1])
        if ok.sum() < 3:
            continue
        src = np.c_[np.interp(tt[ok], times, L[:, 0]), np.interp(tt[ok], times, L[:, 1])]
        r = robust_similarity(src, gxy[ok], True, c, seed=seed)
        fits.append(dict(r, lag=float(lag), median=float(np.median(r['residuals'])), used=ok))
    best = min(fits, key=lambda r: r['median']) if fits else None
    zero = next((r for r in fits if abs(r['lag']) < 1e-9), None)
    if best is not None and zero is not None and (zero['median'] <= lag_gain * best['median'] or
                                                  zero['median'] - best['median'] < lag_min_gain_m):
        best = zero                     # a latency that hardly improves the fit is not believed (see the doc)
    if best is None:
        raise VOError('the GPS fixes do not overlap the VO\'s time span')
    R2 = best['R']
    Rz = np.eye(3)
    Rz[:2, :2] = R2
    R = Rz @ Rl
    t = np.r_[best['t'], 0.0]
    e = best['residuals']
    path = float(np.sum(np.linalg.norm(np.diff(gxy[best['used']], axis=0), axis=1)))
    return {'lag_profile': [[round(r['lag'], 3), round(r['median'], 3)] for r in fits],
            's': float(best['s']), 'R': R, 't': t, 'lag_s': best['lag'], 'heading_deg': math.degrees(
        math.atan2(R2[1, 0], R2[0, 0])), 'residuals_m': e, 'fixes': int(best['used'].sum()),
        'rms_m': float(np.sqrt(np.mean(e ** 2))), 'median_m': float(np.median(e)),
        'p90_m': float(np.percentile(e, 90)), 'max_m': float(e.max()), 'gps_path_m': path,
        'down_weighted': int((best['weights'] < 0.999).sum())}


def transform(vo: VOResult, s: float, A, b):
    """Apply the similarity x' = s·A·x + b to the map in place (poses become metric when s is the GPS scale)."""
    A, b = np.asarray(A, float), np.asarray(b, float)
    for i, p in vo.poses.items():
        T = pose_mat(p)
        R2 = T[:3, :3] @ A.T
        vo.poses[i] = pose_vec(R2, s * T[:3, 3] - R2 @ b)
    for k, x in vo.points.items():
        vo.points[k] = s * A @ x + b


def speed_distance(gps_t, speed_mps, rel_sigma: float = 0.03, abs_sigma: float = 0.05, lag: float = 0.0):
    """(t_a, t_b) -> (metres travelled, σ): the GPS speed over ground (1 Hz; linearly interpolated, shifted by lag)
    integrated between two times. σ = rel_sigma × distance + abs_sigma: a consumer receiver's Doppler speed is good to
    about 0.1 m/s, but its timing against the video is known only to about ±0.5 s, which shows when the car speeds
    up or brakes (heuristic)."""
    gt, sv = np.asarray(gps_t, float), np.asarray(speed_mps, float)

    def dist(ta, tb):
        n = max(2, int(abs(tb - ta) / 0.05) + 1)
        tt = np.linspace(ta, tb, n) + lag
        v = np.interp(tt, gt, sv)
        d = float(np.sum((v[1:] + v[:-1]) / 2 * np.diff(tt)))
        return abs(d), rel_sigma * abs(d) + abs_sigma
    return dist


def interp_positions(t_query, gps_t, gps_xy) -> tuple:
    """GPS positions (K, 2) linearly interpolated at t_query, and which queries lie within the fixes' span."""
    t_query = np.asarray(t_query, float)
    ok = (t_query >= gps_t[0]) & (t_query <= gps_t[-1])
    return np.c_[np.interp(t_query, gps_t, gps_xy[:, 0]), np.interp(t_query, gps_t, gps_xy[:, 1])], ok


def gps_aided_ba(vo: VOResult, tracks: Tracks, intr: Intrinsics, times, gps_t, gps_xy, lag: float,
                 height: float, sigma_xy: float = 3.0, sigma_z: float = 2.0, free=(), settings: dict | None = None
                 ) -> dict:
    """A global bundle adjustment of a metric, world-aligned map (transform()) with every keyframe's centre pulled
    towards the GPS (interpolated at its time − lag, align_to_gps's lag; σ sigma_xy metres east and north) and
    towards `height` metres above the road plane z = 0 (σ sigma_z: OpenStreetMap-flat roads are an assumption; slopes
    bend it). Removes
    the odometry's scale drift at the price of the GPS's own error. No camera is held: the priors fix the gauge
    (holding the first keyframe at its similarity-aligned place, which is off by that alignment's residual, bent the
    solution: 6.9 m RMS against the fixes where the free gauge gave 4.5 m, on a 20 s highway segment). Updates vo in
    place and re-places every frame by PnP (or interpolation; replace_frames)."""
    s = dict(DEFAULTS, **(settings or {}))
    times = np.asarray(times, float)
    frames, cam_i, tids, pt_i, uv = observations(vo, tracks)
    xy, ok = interp_positions(times[frames] - lag, gps_t, gps_xy)      # fix time + lag = the frames' time
    k = np.nonzero(ok)[0]
    tgt = np.c_[xy[k], np.full(len(k), height)]
    sig = np.tile([sigma_xy, sigma_xy, sigma_z], (len(k), 1))
    P0 = np.array([vo.poses[f] for f in frames])
    X0 = np.array([vo.points[int(t)] for t in tids])
    out = {'rounds': []}
    cur = intr
    for rd in range(2):
        r = bundle_adjust(P0, X0, cam_i, pt_i, uv, cur, (), free, s['huber_px'], s['global_ba_iterations'],
                          priors=(k, tgt, sig))
        P0, X0, cur = r['poses'], r['points'], r['intr']
        err = np.linalg.norm(r['residuals'], axis=1)
        keep = err <= s['outlier_px']
        out['rounds'].append({kk: r[kk] for kk in ('rms_before', 'rms_after', 'median_before', 'median_after',
                                                    'seconds', 'iterations', 'observations')} |
                             {'intr': cur.doc(), 'intr_std': r.get('intr_std')})
        if rd == 0 and not keep.all():
            frames_, cam_i, tids_, pt_i, uv = _refilter(frames, cam_i[keep], tids, pt_i[keep], uv[keep])
            X0 = X0[np.searchsorted(tids, tids_)]
            tids = tids_
    for j, f in enumerate(frames):
        vo.poses[f] = P0[j]
    for j, t in enumerate(tids):
        vo.points[int(t)] = X0[j]
    replace_frames(vo, tracks, cur, s)
    out['intr'] = cur
    out['priors'] = int(len(k))
    return out


def heading_change(R_cw_list) -> np.ndarray:
    """Cumulative yaw (degrees, about +z of a levelled frame) of camera-to-world rotations' optical axes."""
    yaw = np.array([math.atan2(R[1, 2], R[0, 2]) for R in R_cw_list])
    return np.degrees(np.unwrap(yaw))
