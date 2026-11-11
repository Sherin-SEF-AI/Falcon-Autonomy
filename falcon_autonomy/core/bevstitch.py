"""The bird's-eye surround view of a rig's cameras: remap tables over a ground grid, seams, feathered blend weights, the
body mask, exposure gains from the ground two cameras share, and the stitcher itself — torch grid_sample on a CUDA GPU
(half precision, every camera of one image size in one batch) or OpenCV's remap on the CPU (numpy at import; OpenCV
and torch are imported by the functions that use them). Ported from Vantage core/stitch.py (its tables, blend weights
and gain compensation) for Falcon's rigs; the exported tables are read by Vantage's Apache-2.0 reference stitcher,
copied unchanged into falcon_autonomy/runtime/ (bev_stitch.py, LICENSE-APACHE-2.0) so it can run on the vehicle (a
Jetson's OpenCV remaps on its GPU with --cuda). docs/bev.md is the manual.

    lut = build_lut(rig)                                    # a project's rig with two or more cameras
    st = Stitcher(lut)                                      # CUDA when there is one, else OpenCV on the CPU
    res = st.stitch({cam_id: rgb_uint8, …}, gains='auto')   # res.image (3, H, W), res.owner, res.valid
    export(lut, 'bev-tables/')                              # bev_lut.json/.npz/.txt, per-camera .bin, stitcher/

Frames
------
The ground grid is in CARLA's vehicle frame (core.rig: x forward, y right, z up, origin on the ground under the
vehicle's bounding-box centre), the ground the plane z = 0. The image is the view from above with the front at the
top: row 0 is the front edge (x1), column 0 the left edge (y0, the most negative y); pixel (j, i) covers x in
[x1 − (j + 1)·res, x1 − j·res), y in [y0 + i·res, y0 + (i + 1)·res). A camera's ray through a ground point is taken
into its optical frame by its pose (core.lens.optical_to_vehicle) and through its lens model (core.lens.Intrinsics:
every model Falcon renders, fisheyes included); the tables hold, per ground pixel, where each camera samples it in
pixel-edge coordinates of the sensor's own size (−1: it does not), as Vantage's export does. A pinhole camera's
ground mapping is a homography (ground_homography: H = K [r₁ r₂ t] of the optical frame's rotation columns and
translation), computed in closed form; every other lens goes through its model per point.

Seams and weights
-----------------
Azimuths are Vantage's: about the body centre (centre_x, 0), counter-clockwise from ahead seen from above, i.e. with
y to the left (az = atan2(−y, x − centre_x) in CARLA's frame), degrees in (−180, 180]. The seams come, in this order,
from the rig's 'vantage' block (core.vantagerig: the seams set in Vantage or estimated there, and the body centre),
else from Vantage's own rule applied here to the rig's cameras (estimate_seams: each camera's density-weighted mean
azimuth over the ground it resolves best within SEAM_ROI_M of the body, then for each neighbouring pair the ray that
maximises the shared, well-resolved ground — vantagerig.estimate_seams's rule, with Falcon's lens models and the body
box as the only occluder). A seam hands the sector after it, counter-clockwise, to its camera b — the owner of an
azimuth is core.rigcompare.owner's. Across each seam line the weights are a linear ramp blend_m wide (Vantage's
blend_weights: 0.2 m by default, a rig's own blend_m when it came from Vantage); a camera contributes only where it
sees the point (inside its lens's field and image, and the ray from the camera to the point not through the body
box); a point its owner cannot see goes to the camera that resolves it best (Vantage's pixel density, px/cm); the
weights sum to 1 wherever any camera sees the ground.

The body: the CARLA car's box (core.vantagerig.CARLA_VEHICLES, its width less the mirrors as Vantage takes it; a rig
from Vantage brings the project's body) is masked (drawn in BODY_RGB), and blocks the rays of every camera from
z = 0.2 m to its roof — except within OCCLUDER_SKIP_M of the camera, so a camera mounted on the body's surface does not
block itself.

Exposure gains (gains='auto'): Brown and Lowe's gain compensation over the ground two cameras share (well exposed
pixels only), clipped to GAIN_MIN–GAIN_MAX, as Vantage's estimate_gains and the runtime's --balance.

The stitcher
------------
* torch (CUDA by default): each camera's table is a sampling grid in grid_sample's normalised coordinates (with
  align_corners=False a pixel-edge u maps to 2u/W − 1 whatever the image's resolution, so frames of any size with the
  sensor's aspect are read correctly); the frames of every camera of one size go through one grid_sample in half
  precision, are weighted and summed on the GPU. The output is a (3, H, W) tensor in 0–255 on the device.
* cv2 (the CPU fallback, numpy and OpenCV only): cv2.remap per camera with the tables in OpenCV's fixed-point form
  (convertMaps), as the runtime does. OpenCV quantises sub-pixel positions to 1/32 px, so the two differ by a level
  or two of 255 at sharp edges (measured in tests/test_bevstitch.py).
A camera without a frame is left out and its ground shared among the others that see it (the runtime's rule).

Limits (said in every result): a bird's-eye view is a projection onto the flat ground: anything above the ground —
cars, kerbs, people, poles — is smeared outwards along each camera's rays, and differently by each camera, so it
doubles or vanishes at a seam. A bowl (Vantage core/bowl.py) is not built here. The ground is taken as the plane under
the vehicle's origin: a slope, a camber or the body's pitch moves it.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import lens as LENS

SCHEMA = 'falcon.bev_lut/1'
RES_M = 0.02                    # ground metres per stitched pixel
MARGIN_M = 4.0                  # the ground shown beyond the body's box on every side
BLEND_M = 0.2                   # Vantage's default seam blend width
SEAM_CELL_M = 0.05              # the seam estimate's ground grid (vantagerig's)
SEAM_ROI_M = 2.0                # the seam estimate's ground: this far round the body's outline (Vantage's default)
SEAM_MIN_RADIUS_M = 0.3
OCCLUDER_SKIP_M = 0.3           # the body box does not block a camera's ray within this distance of the camera
GAIN_MIN, GAIN_MAX = 0.4, 2.5   # the runtime's and Vantage's clip of the solved exposure gains
BODY_RGB = (38, 38, 36)
EMPTY_RGB = (20, 20, 20)
DEFAULT_VEHICLE = 'vehicle.lincoln.mkz_2020'
LIMITS_NOTE = ('the ground plane only: anything above the ground is smeared outwards along each camera\'s rays and '
               'doubles or vanishes at a seam (no bowl); the ground is taken as flat under the vehicle')


class BevError(ValueError):
    pass


# ================================================================================================ the rig
def cameras_of(rig: dict) -> list:
    """The rig's camera sensors (pinhole and fisheye), in the rig's order."""
    return [s for s in rig.get('sensors') or [] if str(s.get('type', '')).startswith('camera.')]


def body_box(rig: dict) -> dict:
    """The body on the ground in CARLA's vehicle frame: {'x0', 'x1' (rear, front), 'hw' (half width without mirrors),
    'height', 'source'}. A rig from Vantage with the project's body uses it (placed by the CARLA car's rear axle);
    otherwise the CARLA car's bounding box (core.vantagerig.CARLA_VEHICLES, else core.rig.VEHICLES), else
    the Lincoln MKZ 2020's."""
    from .rig import VEHICLES
    from .vantagerig import CARLA_VEHICLES
    v = rig.get('vantage') or {}
    body, veh = v.get('body'), v.get('vehicle') or {}
    if body and veh.get('rear_axle_x') is not None and body.get('rear_x') is not None:
        rax = float(veh['rear_axle_x'])
        return {'x0': float(body['rear_x']) + rax, 'x1': float(body['front_x']) + rax,
                'hw': float(body['width']) / 2.0, 'height': float(body.get('height') or 1.5),
                'source': 'the Vantage project\'s body'}
    bp = rig.get('vehicle') or DEFAULT_VEHICLE
    if bp in CARLA_VEHICLES:
        _, L, W, H = CARLA_VEHICLES[bp][:4]
        return {'x0': -L / 2.0, 'x1': L / 2.0, 'hw': 0.46 * W, 'height': H, 'source': f'{bp}\'s CARLA box'}
    if bp in VEHICLES:
        L, W, H = VEHICLES[bp]
        return {'x0': -L / 2.0, 'x1': L / 2.0, 'hw': 0.46 * W, 'height': H, 'source': f'{bp}\'s CARLA box'}
    L, W, H = CARLA_VEHICLES[DEFAULT_VEHICLE][1:4]
    return {'x0': -L / 2.0, 'x1': L / 2.0, 'hw': 0.46 * W, 'height': H,
            'source': f'{bp} is not a known CARLA car: {DEFAULT_VEHICLE}\'s box assumed'}


def _pose_of(s: dict) -> tuple:
    """(R optical -> vehicle (CARLA numbers), position (3,)) of a sensor."""
    p = dict({'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}, **(s.get('pose') or {}))
    return LENS.optical_to_vehicle(p), np.array([float(p['x']), float(p['y']), float(p['z'])])


def ground_homography(sensor: dict) -> np.ndarray:
    """A pinhole camera's ground homography: [u, v, 1]ᵀ ~ H [x, y, 1]ᵀ for a ground point (x, y, 0) of CARLA's vehicle
    frame, (u, v) pixel-edge coordinates. H = K [r₁ r₂ t] with r₁, r₂ the first two columns of the vehicle → optical
    rotation and t = −Rᵀ·position; points behind the camera have w ≤ 0."""
    if sensor.get('type') != 'camera.pinhole':
        raise BevError(f'{sensor.get("id")}: a ground homography is a pinhole camera\'s (this is {sensor.get("type")})')
    intr = LENS.intrinsics(sensor)
    R, pos = _pose_of(sensor)
    Rt = R.T                                                  # vehicle -> optical
    K = np.array([[intr.f_px, 0.0, intr.cx], [0.0, intr.f_px * intr.aspect, intr.cy], [0.0, 0.0, 1.0]])
    return K @ np.column_stack([Rt[:, 0], Rt[:, 1], -Rt @ pos])


def _occluded(pos, P, box, skip=OCCLUDER_SKIP_M):
    """Whether the segment from pos to each point P (k, 3) passes through the body box (a slab test), leaving out the
    first `skip` metres from the camera and the last 2 % (the point itself)."""
    lo = np.array([box['x0'], -box['hw'], 0.2])
    hi = np.array([box['x1'], box['hw'], box['height']])
    d = P - pos
    ln = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    t0 = np.maximum(0.02, skip / ln)
    t1 = np.full(len(P), 0.98)
    for a in range(3):
        da = d[:, a]
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = (lo[a] - pos[a]) / da
            tb = (hi[a] - pos[a]) / da
        tmin, tmax = np.minimum(ta, tb), np.maximum(ta, tb)
        par = np.abs(da) < 1e-12
        inside = (pos[a] >= lo[a]) & (pos[a] <= hi[a])
        tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
        tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
        t0, t1 = np.maximum(t0, tmin), np.minimum(t1, tmax)
    return t0 < t1


def project_ground(sensor: dict, P: np.ndarray, box: dict | None = None):
    """Where a camera sees ground points P (k, 3) of CARLA's vehicle frame: (u, v) pixel-edge coordinates of the
    sensor's own size, ok (inside the lens's field and the image, in front, and — with box — not behind the body),
    theta (rad off the axis), range (m) and the method ('homography' for a pinhole, 'lens' otherwise)."""
    intr = LENS.intrinsics(sensor)
    R, pos = _pose_of(sensor)
    d = P - pos
    rng = np.linalg.norm(d, axis=1)
    d_opt = d @ R                                              # Rᵀ·d per row: vehicle -> optical
    if sensor.get('type') == 'camera.pinhole':
        H = ground_homography(sensor)
        q = np.c_[P[:, 0], P[:, 1], np.ones(len(P))] @ H.T
        w = q[:, 2]
        front = w > 1e-9
        ws = np.where(front, w, 1.0)
        u, v = q[:, 0] / ws, q[:, 1] / ws
        ok = front & (u >= 0) & (u <= intr.width) & (v >= 0) & (v <= intr.height)
        theta = np.arctan2(np.hypot(d_opt[:, 0], d_opt[:, 1]), d_opt[:, 2])
        method = 'homography'
    else:
        u, v, theta, ok = intr.project_dirs(d_opt)
        method = 'lens'
    if box is not None:
        ok = ok & ~_occluded(pos, P, box)
    return u, v, ok, theta, rng, method


def density(sensor: dict, theta, rng):
    """Vantage's pixel density on a target facing the camera (px/cm; core.vantagerig._density)."""
    from .vantagerig import _density
    return _density(LENS.intrinsics(sensor), theta, rng)


# ================================================================================================ seams
def azimuth(X, Y, centre_x: float):
    """Vantage's azimuth (degrees, counter-clockwise from ahead, y to the left) of CARLA-frame points about the body
    centre."""
    return np.degrees(np.arctan2(-np.asarray(Y, float), np.asarray(X, float) - centre_x))


def estimate_seams(rig: dict, box: dict | None = None, cell: float = SEAM_CELL_M, roi_m: float = SEAM_ROI_M) -> dict:
    """Vantage's seam rule on the rig's own cameras (module doc): {'centre_x', 'seams': [{'a', 'b' (sensor ids),
    'a_name', 'b_name', 'angle_deg', 'source': 'estimated', 'density'}] counter-clockwise}. Ported from
    core.vantagerig.estimate_seams, which needs a Vantage project; here the cameras are Falcon sensors."""
    box = box or body_box(rig)
    cams = cameras_of(rig)
    cx = (box['x0'] + box['x1']) / 2.0
    out = {'centre_x': cx, 'cell_m': cell, 'seams': []}
    if len(cams) < 2:
        return out
    pad = roi_m + 0.5
    xs = np.arange(box['x0'] - pad + cell / 2, box['x1'] + pad, cell)
    ys = np.arange(-box['hw'] - pad + cell / 2, box['hw'] + pad, cell)
    X, Y = np.meshgrid(xs, ys)
    dx = np.maximum(np.maximum(box['x0'] - X, X - box['x1']), 0.0)
    dy = np.maximum(np.abs(Y) - box['hw'], 0.0)
    body = (dx == 0) & (dy == 0)
    roi = (np.hypot(dx, dy) <= roi_m) & ~body
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    V, Dn = [], []
    for c in cams:
        _, _, ok, th, rng, _ = project_ground(c, P, box)
        Dn.append(np.where(ok, density(c, th, rng), 0.0).reshape(X.shape))
        V.append(ok.reshape(X.shape))
    V, Dn = np.stack(V), np.stack(Dn)
    az = np.radians(azimuth(X, Y, cx))
    rad = np.hypot(Y, X - cx)
    best = np.argmax(Dn, 0)
    cover = V.any(0)
    azs = []
    for i, c in enumerate(cams):
        m = roi & cover & (best == i)
        if m.sum() > 4:
            w = Dn[i][m]
            azs.append(math.atan2(float((np.sin(az[m]) * w).sum()), float((np.cos(az[m]) * w).sum())))
        else:
            p = c.get('pose') or {}
            azs.append(math.atan2(-float(p.get('y', 0.0)), float(p.get('x', 0.0)) - cx))
    order = list(np.argsort(azs))
    rmax = float(rad[roi].max()) if roi.any() else 5.0
    radii = np.arange(SEAM_MIN_RADIUS_M, rmax + cell, cell)
    x0, y0 = float(X[0, 0] - cell / 2), float(Y[0, 0] - cell / 2)
    ny, nx = X.shape

    def sample(ang):                                           # ang in Vantage's azimuth (y left)
        Xs = cx + np.cos(ang)[..., None] * radii
        Ys = -np.sin(ang)[..., None] * radii
        ii = ((Xs - x0) / cell).astype(int)
        jj = ((Ys - y0) / cell).astype(int)
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        return np.clip(ii, 0, nx - 1), np.clip(jj, 0, ny - 1), ok
    n = len(cams)
    for k in range(n):
        ia, ib = order[k], order[(k + 1) % n]
        a0 = azs[ia]
        span = (azs[ib] - a0) % (2 * math.pi)
        if span < 1e-3:
            span = 2 * math.pi / n
        cand = a0 + span * np.linspace(0.08, 0.92, 41)
        ii, jj, ok = sample(cand)
        inroi = ok & roi[jj, ii]
        both = V[ia][jj, ii] & V[ib][jj, ii] & inroi
        dmin = np.minimum(Dn[ia][jj, ii], Dn[ib][jj, ii])
        frac = both.sum(1) / np.maximum(inroi.sum(1), 1)
        dsc = np.where(both, np.clip(dmin, 0, 10), 0).sum(1) / np.maximum(inroi.sum(1), 1) / 10
        score = frac + 0.25 * dsc - 0.02 * np.abs(np.linspace(-1, 1, len(cand)))
        ang = float(cand[int(np.argmax(score))])
        ii, jj, ok = sample(np.array(ang))
        inr = ok & roi[jj, ii]
        da, db = Dn[ia][jj, ii], Dn[ib][jj, ii]
        dd = np.where(V[ia][jj, ii] & V[ib][jj, ii], np.minimum(da, db), np.maximum(da, db))
        out['seams'].append({'a': cams[ia]['id'], 'b': cams[ib]['id'], 'a_name': cams[ia].get('name', cams[ia]['id']),
                             'b_name': cams[ib].get('name', cams[ib]['id']),
                             'angle_deg': round(math.degrees(math.atan2(math.sin(ang), math.cos(ang))), 3),
                             'source': 'estimated', 'density': round(float(np.median(dd[inr])), 3) if inr.any()
                             else 0.0})
    out['seams'].sort(key=lambda s: s['angle_deg'])
    return out


def rig_seams(rig: dict, box: dict | None = None) -> dict:
    """The seams the stitched view uses (module doc): the rig's 'vantage' block's when every seam names cameras of the
    rig (its tags are the sensors' names), else estimate_seams. {'centre_x', 'seams' (sorted by angle, a/b sensor
    ids), 'source': 'vantage' | 'estimated' | 'one camera'}."""
    box = box or body_box(rig)
    cams = cameras_of(rig)
    if len(cams) < 2:
        return {'centre_x': (box['x0'] + box['x1']) / 2.0, 'seams': [], 'source': 'one camera'}
    v = rig.get('vantage') or {}
    by_name = {s.get('name', s['id']): s['id'] for s in cams}
    ids = {s['id'] for s in cams}
    vs = v.get('seams') or []
    cx = (v.get('centre') or {}).get('falcon_x')
    if vs and cx is not None:
        seams = []
        for s in vs:
            a = by_name.get(s.get('a'), s.get('a') if s.get('a') in ids else None)
            b = by_name.get(s.get('b'), s.get('b') if s.get('b') in ids else None)
            if a is None or b is None:
                seams = None
                break
            seams.append({'a': a, 'b': b, 'a_name': s.get('a'), 'b_name': s.get('b'),
                          'angle_deg': float(s['angle_deg']), 'source': s.get('source') or 'vantage'})
        if seams:
            seams.sort(key=lambda s: s['angle_deg'])
            return {'centre_x': float(cx), 'seams': seams, 'source': 'vantage'}
    est = estimate_seams(rig, box)
    return {'centre_x': est['centre_x'], 'seams': est['seams'], 'source': 'estimated'}


def owner_index(az_deg, seams: list, ids: list) -> np.ndarray:
    """The index (into ids) of the camera owning each azimuth: core.rigcompare.owner vectorised (the b of the last seam
    at or before it, counter-clockwise); −1 without seams."""
    az = np.asarray(az_deg, float)
    if not seams:
        return np.full(az.shape, 0 if len(ids) == 1 else -1, np.int64)
    angles = np.array([s['angle_deg'] for s in seams])
    i = (np.searchsorted(angles, az, side='right') - 1) % len(seams)
    bidx = np.array([ids.index(s['b']) for s in seams])
    return bidx[i]


def blend_weights(X, Y, centre_x: float, seams: list, ids: list, blend_m: float, vis: dict, dens: dict,
                  body) -> dict:
    """Each camera's weight at CARLA-frame ground points (module doc; Vantage's blend_weights with its azimuth): the
    owner by sector, a linear ramp blend_m wide across each seam line, only where the camera sees the point, holes
    filled by the best-density camera that sees it, nothing under the body. {id: float32 like X}, summing to 1 where
    any camera sees the point."""
    shape = np.shape(X)
    W = {u: np.zeros(shape, np.float32) for u in ids}
    Xc, Yc = np.asarray(X, float) - centre_x, -np.asarray(Y, float)        # Vantage's axes: y to the left
    seams = [s for s in seams if s['a'] in W and s['b'] in W]
    if seams:
        own = owner_index(np.degrees(np.arctan2(Yc, Xc)), seams, ids)
        for i, u in enumerate(ids):
            W[u][own == i] = 1.0
        half = blend_m / 2.0
        if half > 0:
            for s in seams:
                a = math.radians(s['angle_deg'])
                along = Xc * math.cos(a) + Yc * math.sin(a)
                d = -Xc * math.sin(a) + Yc * math.cos(a)          # + = the counter-clockwise side (camera b)
                zone = (along > 0) & (np.abs(d) < half)
                wb = np.clip(0.5 + d / (2 * half), 0, 1)
                W[s['a']][zone] = (1 - wb[zone])
                W[s['b']][zone] = wb[zone]
    elif ids:
        W[ids[0]][:] = 1.0
    for u in W:
        W[u] *= vis[u]
    tot = sum(W.values()) if W else np.zeros(shape)
    hole = (tot < 1e-6) & ~body
    if ids:
        D = np.stack([dens[u] for u in ids])
        best = np.argmax(D, 0)
        anyv = D.max(0) > 0
        for i, u in enumerate(ids):
            W[u][hole & anyv & (best == i)] = 1.0
        tot = sum(W.values())
    with np.errstate(invalid='ignore', divide='ignore'):
        for u in W:
            W[u] = np.where(tot > 0, W[u] / np.where(tot > 0, tot, 1), 0).astype(np.float32)
    return W


# ================================================================================================ the tables
@dataclass
class StitchLUT:
    x0: float                          # the ground's rear edge (m, CARLA vehicle frame)
    x1: float                          # its front edge
    y0: float                          # its left edge (y right: the most negative y)
    y1: float
    res: float
    nx: int                            # columns (across, left to right)
    ny: int                            # rows (front to back)
    ids: list                          # camera sensor ids
    names: list
    sensor: dict                       # id -> (width, height) the tables' pixel coordinates refer to
    map_x: dict                        # id -> (ny, nx) float32 pixel-edge u, −1 where the camera does not see
    map_y: dict
    weight: dict                       # id -> (ny, nx) float32, summing to 1 where covered
    body: np.ndarray                   # (ny, nx) bool
    covered: np.ndarray                # (ny, nx) bool
    owner: np.ndarray                  # (ny, nx) int16: the camera with the largest weight (index into ids), −1 none
    centre_x: float
    seams: list
    seam_source: str
    blend_m: float
    box: dict
    poses: dict = field(default_factory=dict)
    method: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def world_xy(self):
        """(X, Y) (ny, nx): the CARLA-frame ground coordinates of each pixel's centre."""
        xs = self.x1 - (np.arange(self.ny) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.nx) + 0.5) * self.res
        Y, X = np.meshgrid(ys, xs)
        return X, Y

    def pixel_of(self, x, y):
        """(row, column) float pixel coordinates (centres at +0.5) of CARLA-frame ground points."""
        return (self.x1 - np.asarray(x, float)) / self.res, (np.asarray(y, float) - self.y0) / self.res

    def summary(self) -> dict:
        return {'schema': SCHEMA, 'size': [self.nx, self.ny], 'resolution_m': self.res,
                'extent_m': {'x0': self.x0, 'x1': self.x1, 'y0': self.y0, 'y1': self.y1},
                'cameras': [{'id': i, 'name': n, 'sensor_size': list(self.sensor[i]), 'method': self.method.get(i),
                             'covers': round(float((self.weight[i] > 0).mean()), 4)}
                            for i, n in zip(self.ids, self.names)],
                'covered_share': round(float((self.covered & ~self.body).sum() / max((~self.body).sum(), 1)), 4),
                'centre_x': round(self.centre_x, 4), 'seam_source': self.seam_source,
                'seams': [{k: s.get(k) for k in ('a', 'b', 'a_name', 'b_name', 'angle_deg', 'source')}
                          for s in self.seams], 'blend_m': self.blend_m, 'body': self.box, 'notes': list(self.notes),
                'limits': LIMITS_NOTE}


def default_extent(box: dict, margin: float = MARGIN_M) -> tuple:
    """(x0, x1, y0, y1): the body's box and margin metres on every side."""
    return (box['x0'] - margin, box['x1'] + margin, -box['hw'] - margin, box['hw'] + margin)


def build_lut(rig: dict, res: float = RES_M, extent=None, margin: float = MARGIN_M, blend_m: float | None = None,
              cameras=None) -> StitchLUT:
    """The remap tables and weights of a rig's cameras over the ground (module doc). extent: (x0, x1, y0, y1) in
    CARLA's vehicle frame (default: the body's box and margin metres round it); cameras: sensor ids to use (default
    every camera of the rig); blend_m: the seam ramp's width (default the rig's Vantage blend_m, else BLEND_M)."""
    cams = cameras_of(rig)
    if cameras:
        want = list(cameras)
        missing = [c for c in want if c not in {s['id'] for s in cams}]
        if missing:
            raise BevError(f'the rig has no camera {", ".join(missing)}')
        cams = [s for s in cams if s['id'] in want]
    if not cams:
        raise BevError(f'rig {rig.get("id")!r} has no cameras')
    res = float(res)
    if not 0.002 <= res <= 0.5:
        raise BevError('the resolution is 0.002–0.5 m per pixel')
    box = body_box(rig)
    x0, x1, y0, y1 = extent if extent is not None else default_extent(box, margin)
    nx, ny = int(round((y1 - y0) / res)), int(round((x1 - x0) / res))
    if nx * ny > 16_000_000:
        raise BevError(f'{nx} × {ny} px is too large a bird\'s-eye view (at most 16 Mpx)')
    rig_c = dict(rig, sensors=cams)
    sm = rig_seams(rig_c, box)
    if blend_m is None:
        blend_m = float((rig.get('vantage') or {}).get('blend_m') or BLEND_M)
    lut = StitchLUT(x0, x0 + ny * res, y0, y0 + nx * res, res, nx, ny, [c['id'] for c in cams],
                    [c.get('name', c['id']) for c in cams], {}, {}, {}, {}, np.zeros((ny, nx), bool),
                    np.zeros((ny, nx), bool), np.full((ny, nx), -1, np.int16), float(sm['centre_x']), sm['seams'],
                    sm['source'], float(blend_m), box)
    X, Y = lut.world_xy()
    lut.body = (X >= box['x0']) & (X <= box['x1']) & (np.abs(Y) <= box['hw'])
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    vis, dens = {}, {}
    for c in cams:
        u, v, ok, th, rng, method = project_ground(c, P, box)
        ok &= ~lut.body.ravel()
        intr = LENS.intrinsics(c)
        lut.sensor[c['id']] = (intr.width, intr.height)
        lut.map_x[c['id']] = np.where(ok, u, -1).reshape(ny, nx).astype(np.float32)
        lut.map_y[c['id']] = np.where(ok, v, -1).reshape(ny, nx).astype(np.float32)
        vis[c['id']] = ok.reshape(ny, nx)
        dens[c['id']] = np.where(ok, density(c, th, rng), 0.0).reshape(ny, nx)
        lut.poses[c['id']] = dict(c.get('pose') or {})
        lut.method[c['id']] = method
        if not ok.any():
            lut.notes.append(f'{c.get("name", c["id"])} sees none of the ground grid')
    lut.weight = blend_weights(X, Y, lut.centre_x, sm['seams'], lut.ids, lut.blend_m, vis, dens, lut.body)
    stack = np.stack([lut.weight[i] for i in lut.ids])
    lut.covered = stack.sum(0) > 0
    lut.owner = np.where(lut.covered, np.argmax(stack, 0), -1).astype(np.int16)
    if sm['source'] == 'estimated' and len(cams) > 1:
        lut.notes.append('seams estimated by Vantage\'s rule on these cameras (the rig did not come from Vantage)')
    lut.notes.append(f'body: {box["source"]}')
    return lut


# ================================================================================================ gains
def warp_cpu(lut: StitchLUT, images: dict) -> tuple:
    """Each camera's frame resampled onto the ground grid by OpenCV: ({id: float32 (ny, nx, 3)}, {id: bool valid})."""
    import cv2
    out, valid = {}, {}
    for i in lut.ids:
        img = images.get(i)
        if img is None:
            continue
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        mx, my = _cv_maps(lut, i, img.shape[1], img.shape[0])
        out[i] = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        valid[i] = lut.map_x[i] >= 0
    return out, valid


def _cv_maps(lut, i, w, h):
    sw, sh = lut.sensor[i]
    ok = lut.map_x[i] >= 0
    mx = np.where(ok, lut.map_x[i] * (w / sw) - 0.5, -1).astype(np.float32)
    my = np.where(ok, lut.map_y[i] * (h / sh) - 0.5, -1).astype(np.float32)
    return mx, my


def overlap_means(lut: StitchLUT, warped: dict, valid: dict, step: int = 2) -> list:
    """[(id a, id b, n, mean a (3,), mean b (3,))] over the ground both cameras see, well exposed pixels only."""
    ids = [u for u in lut.ids if u in warped]
    sl = (slice(None, None, step), slice(None, None, step))
    free = ~lut.body[sl]
    out = []
    for k, a in enumerate(ids):
        for b in ids[k + 1:]:
            m = valid[a][sl] & valid[b][sl] & free
            if m.sum() < 50:
                continue
            A, B = warped[a][sl][m], warped[b][sl][m]
            la, lb = A.mean(1), B.mean(1)
            ok = (la > 8) & (la < 247) & (lb > 8) & (lb < 247)
            if ok.sum() < 50:
                continue
            out.append((a, b, int(ok.sum()), A[ok].mean(0), B[ok].mean(0)))
    return out


def estimate_gains(lut: StitchLUT, warped: dict, valid: dict, sigma_n: float = 10.0, sigma_g: float = 0.5) -> dict:
    """Per-camera, per-channel gains making overlapping cameras agree (Brown and Lowe 2007's gain compensation:
    minimise Σ N_ab ((g_a I_ab − g_b I_ba)² / σ_n² + (1 − g)² / σ_g²)), clipped to GAIN_MIN–GAIN_MAX: {'gains':
    {id: (3,) float32}, 'clipped': {id: (3,) the solved gains of a camera clipped}, 'pairs': n}."""
    ids = [u for u in lut.ids if u in warped]
    idx = {u: k for k, u in enumerate(ids)}
    n = len(ids)
    pairs = overlap_means(lut, warped, valid)
    g = np.ones((n, 3))
    for ch in range(3):
        A = np.zeros((n, n))
        bvec = np.zeros(n)
        for a, b, N, ma, mb in pairs:
            i, j = idx[a], idx[b]
            Ia, Ib = ma[ch], mb[ch]
            A[i, i] += N * (Ia * Ia / sigma_n ** 2 + 1 / sigma_g ** 2)
            A[j, j] += N * (Ib * Ib / sigma_n ** 2 + 1 / sigma_g ** 2)
            A[i, j] -= N * Ia * Ib / sigma_n ** 2
            A[j, i] -= N * Ia * Ib / sigma_n ** 2
            bvec[i] += N / sigma_g ** 2
            bvec[j] += N / sigma_g ** 2
        for i in range(n):
            if A[i, i] == 0:
                A[i, i], bvec[i] = 1.0, 1.0
        g[:, ch] = np.linalg.solve(A, bvec) if n else g[:, ch]
    gc = np.clip(g, GAIN_MIN, GAIN_MAX)
    return {'gains': {u: gc[idx[u]].astype(np.float32) for u in ids},
            'clipped': {u: g[idx[u]].astype(np.float32) for u in ids if np.any(gc[idx[u]] != g[idx[u]])},
            'pairs': len(pairs)}


# ================================================================================================ the stitcher
@dataclass
class StitchResult:
    image: object                      # (3, H, W): a torch tensor on the stitcher's device, or a numpy array (cv2)
    owner: np.ndarray                  # (H, W) int16: the camera shown (index into ids), −1 none
    valid: np.ndarray                  # (H, W) bool: ground some delivered camera sees (not the body)
    ids: list
    backend: str
    ms: float
    gains: dict | None = None
    clipped: dict | None = None

    def rgb(self) -> np.ndarray:
        """(H, W, 3) uint8."""
        img = self.image
        if not isinstance(img, np.ndarray):
            img = img.float().clamp(0, 255).round().to('cpu').numpy()
        return np.ascontiguousarray(np.clip(np.rint(img), 0, 255).astype(np.uint8).transpose(1, 2, 0))


def resolve_device(device: str = 'auto') -> tuple:
    """(backend 'torch' | 'cv2', device name): 'auto' is torch on CUDA when torch imports and sees a GPU, else OpenCV on
    the CPU; 'cuda' requires it; 'cpu' is OpenCV; 'torch-cpu' is torch's grid_sample on the CPU (float32)."""
    if device in ('cpu', 'cv2'):
        return 'cv2', 'cpu'
    if device == 'torch-cpu':
        return 'torch', 'cpu'
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if device == 'cuda' and not cuda:
        raise BevError('no CUDA GPU (torch sees none): use --device cpu')
    if device not in ('auto', 'cuda'):
        raise BevError(f'unknown device {device!r} (auto, cuda, cpu or torch-cpu)')
    return ('torch', 'cuda') if cuda else ('cv2', 'cpu')


class Stitcher:
    """Stitch frames with a StitchLUT (module doc). device: 'auto' | 'cuda' | 'cpu' | 'torch-cpu'; half: half precision
    on CUDA (the default)."""

    def __init__(self, lut: StitchLUT, device: str = 'auto', half: bool = True):
        self.lut = lut
        self.backend, self.device = resolve_device(device)
        self.half = bool(half) and self.device == 'cuda'
        self._cv = {}
        self._torch = None
        if self.backend == 'torch':
            self._prepare_torch()
        self.body = lut.body
        self.gains: dict = {}

    # ------------------------------------------------------------------ torch
    def _prepare_torch(self):
        import torch
        lut = self.lut
        dt = torch.float16 if self.half else torch.float32
        dev = torch.device(self.device)
        grids, weights = [], []
        for i in lut.ids:
            sw, sh = lut.sensor[i]
            ok = lut.map_x[i] >= 0
            gx = np.where(ok, 2.0 * lut.map_x[i] / sw - 1.0, -2.0)
            gy = np.where(ok, 2.0 * lut.map_y[i] / sh - 1.0, -2.0)
            grids.append(np.stack([gx, gy], -1).astype(np.float32))
            weights.append(lut.weight[i].astype(np.float32))
        # the frames, the sampling grid and the weights in the working precision, the sums in float32
        self._torch = {'torch': torch, 'dev': dev, 'dt': dt,
                       'grid': torch.as_tensor(np.stack(grids), device=dev),
                       'weight': torch.as_tensor(np.stack(weights), device=dev).to(dt)[:, None],
                       'body': torch.as_tensor(lut.body, device=dev),
                       'empty': torch.tensor(EMPTY_RGB, dtype=dt, device=dev)[:, None, None],
                       'body_rgb': torch.tensor(BODY_RGB, dtype=dt, device=dev)[:, None, None]}

    def _frames_torch(self, images: dict):
        """{id: (3, h, w) tensor on the device, in the working precision, 0–255}."""
        T = self._torch
        torch = T['torch']
        out = {}
        for i in self.lut.ids:
            img = images.get(i)
            if img is None:
                continue
            if isinstance(img, np.ndarray):
                a = img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)
                t = torch.from_numpy(np.ascontiguousarray(a[..., :3])).to(T['dev'], non_blocking=True)
                t = t.permute(2, 0, 1)
            else:
                t = img.to(T['dev'])
                if t.ndim == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3, 4):
                    t = t[..., :3].permute(2, 0, 1)
            out[i] = t.to(T['dt'])
        return out

    def _stitch_torch(self, images: dict, gains):
        T = self._torch
        torch = T['torch']
        F = torch.nn.functional
        lut = self.lut
        frames = self._frames_torch(images)
        idx = {u: k for k, u in enumerate(lut.ids)}
        H, W = lut.ny, lut.nx
        acc = torch.zeros(3, H, W, dtype=torch.float32, device=T['dev'])     # sums in float32 (half's 11 bits of
        wsum = torch.zeros(1, H, W, dtype=torch.float32, device=T['dev'])    # mantissa would lose a quarter level)
        groups = {}
        for u, t in frames.items():
            groups.setdefault(tuple(t.shape), []).append(u)
        for shape, us in groups.items():
            k = [idx[u] for u in us]
            batch = torch.stack([frames[u] for u in us])
            grid = T['grid'][k].to(batch.dtype)            # grid_sample takes one dtype (module doc: half's error)
            smp = F.grid_sample(batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            w = T['weight'][k]
            if gains:
                g = torch.as_tensor(np.stack([np.broadcast_to(np.asarray(gains.get(u, 1.0), np.float32), (3,))
                                              for u in us]), dtype=T['dt'], device=T['dev'])[:, :, None, None]
                smp = smp * g
            acc += (smp.float() * w.float()).sum(0)
            wsum += w.float().sum(0)
        cov = wsum > 0
        out = torch.where(cov, acc / torch.where(cov, wsum, torch.ones_like(wsum)), T['empty'].float())
        out = torch.where(T['body'][None], T['body_rgb'].float(), out)
        valid = cov[0] & ~T['body']
        return out, valid

    # ------------------------------------------------------------------ cv2
    def _stitch_cv(self, images: dict, gains, warped=None):
        import cv2
        lut = self.lut
        out = np.zeros((lut.ny, lut.nx, 3), np.float32)
        wsum = np.zeros((lut.ny, lut.nx), np.float32)
        for i in lut.ids:
            img = images.get(i)
            if img is None:
                continue
            if warped is not None and i in warped:
                smp = warped[i]
            else:
                img = np.asarray(img)
                if img.ndim == 2:
                    img = np.repeat(img[..., None], 3, axis=2)
                img = img[..., :3]
                key = (i, img.shape[1], img.shape[0])
                if key not in self._cv:
                    mx, my = _cv_maps(lut, i, img.shape[1], img.shape[0])
                    self._cv[key] = cv2.convertMaps(mx, my, cv2.CV_16SC2)
                m1, m2 = self._cv[key]
                smp = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            if gains:
                smp = smp * np.asarray(gains.get(i, 1.0), np.float32)
            out += smp * lut.weight[i][..., None]
            wsum += lut.weight[i]
        cov = wsum > 0
        out /= np.where(cov, wsum, 1.0)[..., None]
        out[~cov] = EMPTY_RGB
        out[lut.body] = BODY_RGB
        return out.transpose(2, 0, 1), cov & ~lut.body

    # ------------------------------------------------------------------ the API
    def stitch(self, images: dict, gains=None, sync: bool = True) -> StitchResult:
        """images: {camera id: (h, w, 3) uint8 RGB (numpy, or a torch tensor) at the sensor's aspect}; gains: None,
        'auto' (estimated from this frame's overlaps, on the CPU) or {id: scalar or (3,)}. sync: wait for the GPU
        before timing."""
        t0 = time.perf_counter()
        est = None
        warped = None
        if isinstance(gains, str):
            if gains != 'auto':
                raise BevError(f'gains is None, \'auto\' or a dict, not {gains!r}')
            np_images = {u: (v if isinstance(v, np.ndarray) else v.detach().to('cpu').numpy())
                         for u, v in images.items()}
            warped, valid = warp_cpu(self.lut, np_images)
            est = estimate_gains(self.lut, warped, valid)
            gains = est['gains']
            self.gains = dict(gains)
        if self.backend == 'torch':
            img, valid = self._stitch_torch(images, gains)
            if sync and self.device == 'cuda':
                self._torch['torch'].cuda.synchronize()
            valid = valid.to('cpu').numpy()
        else:
            img, valid = self._stitch_cv(images, gains, warped)
        delivered = [k for k, u in enumerate(self.lut.ids) if images.get(u) is not None]
        owner = self.lut.owner if len(delivered) == len(self.lut.ids) else \
            _owner_of(self.lut, delivered)
        return StitchResult(img, owner, valid, list(self.lut.ids), f'{self.backend}:{self.device}'
                            + (':fp16' if self.half else ''), (time.perf_counter() - t0) * 1e3,
                            None if gains is None else {u: np.asarray(g, np.float32).reshape(-1).tolist()
                                                        if np.ndim(g) else float(g) for u, g in gains.items()},
                            None if est is None else {u: g.tolist() for u, g in est['clipped'].items()})


def _owner_of(lut: StitchLUT, delivered: list) -> np.ndarray:
    stack = np.stack([lut.weight[lut.ids[k]] for k in delivered]) if delivered else np.zeros((1, lut.ny, lut.nx))
    best = np.array(delivered)[np.argmax(stack, 0)] if delivered else np.zeros((lut.ny, lut.nx), int)
    return np.where(stack.sum(0) > 0, best, -1).astype(np.int16)


# ================================================================================================ overlays
def seam_overlay(lut: StitchLUT, rgb: np.ndarray, colour=(255, 200, 0)) -> np.ndarray:
    """The stitched image (H, W, 3) uint8 with the seam rays and the borders between owners drawn on it."""
    import cv2
    out = np.ascontiguousarray(rgb.copy())
    own = lut.owner
    edge = np.zeros(own.shape, bool)
    edge[:, 1:] |= own[:, 1:] != own[:, :-1]
    edge[1:, :] |= own[1:, :] != own[:-1, :]
    edge &= ~lut.body
    out[edge] = (0.5 * out[edge] + 0.5 * np.array(colour)).astype(np.uint8)
    r0, c0 = lut.pixel_of(lut.centre_x, 0.0)
    reach = max(lut.nx, lut.ny) * 2
    for s in lut.seams:
        a = math.radians(s['angle_deg'])
        x, y = lut.centre_x + reach * lut.res * math.cos(a), -reach * lut.res * math.sin(a)
        r1, c1 = lut.pixel_of(x, y)
        cv2.line(out, (int(c0), int(r0)), (int(c1), int(r1)), colour, 1, cv2.LINE_AA)
    return out


# ================================================================================================ export
def _safe(names: list) -> list:
    import re
    out, used = [], set()
    for n in names:
        base = re.sub(r'[^A-Za-z0-9_-]+', '_', n or '').strip('_') or 'cam'
        t, k = base, 2
        while t.lower() in used:
            t, k = f'{base}_{k}', k + 1
        used.add(t.lower())
        out.append(t)
    return out


def lut_files(lut: StitchLUT, rig: dict | None = None) -> list:
    """[(file name, bytes)] in the layout Vantage's export writes and its runtime reads (Vantage core/stitch.py
    lut_files): bev_lut.json, bev_lut.npz (<name>_map (H, W, 2) float32, <name>_weight uint8, body_mask, covered),
    bev_lut.txt (the C++ reference stitcher's manifest), <name>_map.bin, <name>_weight.bin and body_mask.bin."""
    import io
    files = []
    arrays = {'body_mask': lut.body.astype(np.uint8), 'covered': lut.covered.astype(np.uint8)}
    cams = []
    tags = _safe(lut.names)
    for i, name, safe in zip(lut.ids, lut.names, tags):
        mp = np.stack([lut.map_x[i], lut.map_y[i]], -1).astype(np.float32)
        wt = np.round(lut.weight[i] * 255).astype(np.uint8)
        arrays[f'{safe}_map'] = mp
        arrays[f'{safe}_weight'] = wt
        files.append((f'{safe}_map.bin', mp.tobytes()))
        files.append((f'{safe}_weight.bin', wt.tobytes()))
        p = lut.poses.get(i, {})
        cams.append({'tag': name, 'id': i, 'tag_safe': safe, 'map': f'{safe}_map.bin', 'weight': f'{safe}_weight.bin',
                     'sensor_width': lut.sensor[i][0], 'sensor_height': lut.sensor[i][1], 'method': lut.method.get(i),
                     'pose': {k: round(float(p.get(k, 0.0)), 4) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}})
    files.append(('body_mask.bin', lut.body.astype(np.uint8).tobytes()))
    txt = ['# falcon.bev_lut/1 (Vantage\'s vantage.bev_lut/1 layout) manifest for the reference stitcher; see '
           'bev_lut.json', f'width {lut.nx}', f'height {lut.ny}', f'resolution {lut.res:.6g}',
           f'extent {lut.x0:.6g} {lut.y0:.6g} {lut.x1:.6g} {lut.y1:.6g}']
    txt += [f'camera {c["tag_safe"]} {c["map"]} {c["weight"]} {c["sensor_width"]} {c["sensor_height"]}' for c in cams]
    txt.append('body body_mask.bin')
    for c in cams:
        c.pop('tag_safe')
    files.append(('bev_lut.txt', ('\n'.join(txt) + '\n').encode('utf-8')))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    files.append(('bev_lut.npz', buf.getvalue()))
    from .. import __version__
    meta = {
        'schema': SCHEMA, 'layout': 'vantage.bev_lut/1', 'falcon_version': __version__,
        'rig': (rig or {}).get('id', ''), 'vehicle': (rig or {}).get('vehicle', ''),
        'frame': 'CARLA\'s vehicle frame: x forward, y right, z up, origin on the ground under the bounding-box centre',
        'width': lut.nx, 'height': lut.ny, 'resolution_m_per_px': lut.res,
        'extent_m': {'x0': lut.x0, 'x1': lut.x1, 'y0': lut.y0, 'y1': lut.y1},
        'image_axes': 'row 0 = x1 (the front at the top), column 0 = y0 (the left); pixel centres at +0.5',
        'map_format': 'float32, height x width x 2 (x, y) in SENSOR pixel coordinates (pixel centre = i + 0.5); -1 '
                      'where the camera does not contribute. Scale by (image_width / sensor_width) for other '
                      'resolutions and subtract 0.5 for cv2.remap.',
        'weight_format': 'uint8, height x width, 0..255; weights of all cameras sum to 255 where covered',
        'cameras': cams, 'centre_x': lut.centre_x, 'seam_source': lut.seam_source,
        'seams': [{k: s.get(k) for k in ('a', 'b', 'a_name', 'b_name', 'angle_deg', 'source')} for s in lut.seams],
        'azimuth': 'degrees about (centre_x, 0), counter-clockwise from ahead seen from above (y to the left)',
        'blend_m': lut.blend_m, 'body': lut.box, 'notes': list(lut.notes), 'limits': LIMITS_NOTE,
        'usage_python': 'out = sum(cv2.remap(img[c], mapx[c], mapy[c], cv2.INTER_LINEAR) * w[c] / 255)',
        'runtime': 'stitcher/bev_stitch.py (Apache-2.0, numpy and OpenCV): python3 stitcher/bev_stitch.py --frames '
                   'DIR --out bev.png --balance  (frames named after the camera tags)',
    }
    files.insert(0, ('bev_lut.json', json.dumps(meta, indent=1).encode('utf-8')))
    return files


RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runtime')
RUNTIME_FILES = (('bev_stitch.py', 'bev_stitch.py'), ('LICENSE-APACHE-2.0', 'LICENSE'))


def export(lut: StitchLUT, out_dir: str, rig: dict | None = None, runtime: bool = True) -> list:
    """Write the tables (lut_files) into out_dir and, with runtime, Vantage's Apache-2.0 reference stitcher into
    out_dir/stitcher/ (bev_stitch.py and its LICENSE; it finds the tables in its parent folder). Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in lut_files(lut, rig):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    if runtime:
        sd = os.path.join(out_dir, 'stitcher')
        os.makedirs(sd, exist_ok=True)
        for src, dst in RUNTIME_FILES:
            with open(os.path.join(RUNTIME_DIR, src), 'rb') as fh:
                data = fh.read()
            p = os.path.join(sd, dst)
            with open(p, 'wb') as fh:
                fh.write(data)
            paths.append(p)
    return paths


def load_runtime():
    """The bundled runtime module (falcon_autonomy/runtime/bev_stitch.py), imported from its file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('falcon_bev_runtime', os.path.join(RUNTIME_DIR, 'bev_stitch.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================================================ synthetic rigs
def surround_rig(width: int = 1920, height: int = 1080, fov: float = 190.0, vehicle: str = DEFAULT_VEHICLE,
                 model: str = 'equidistant', pitch: float = -40.0) -> dict:
    """A four-fisheye surround-view rig on a CARLA car (front grille, rear, both mirrors; each pitched down), for
    `falcon bev bench` and the tests: a typical layout, not a product's."""
    from .vantagerig import CARLA_VEHICLES
    L, W, H = (CARLA_VEHICLES.get(vehicle) or CARLA_VEHICLES[DEFAULT_VEHICLE])[1:4]
    lens = {'model': model, 'fov': float(fov), 'f_px': None, 'k': [0.0, 0.0, 0.0, 0.0]}
    mounts = (('front', L / 2 + 0.02, 0.0, 0.65, 0.0), ('rear', -L / 2 - 0.02, 0.0, 0.95, 180.0),
              ('left', 0.9, -0.46 * W - 0.08, 1.05, -90.0), ('right', 0.9, 0.46 * W + 0.08, 1.05, 90.0))
    sensors = [{'id': f'svm-{n}', 'name': n, 'type': 'camera.fisheye', 'width': int(width), 'height': int(height),
                'lens': dict(lens), 'pose': {'x': round(x, 4), 'y': round(y, 4), 'z': z, 'roll': 0.0,
                                             'pitch': float(pitch), 'yaw': yaw}} for n, x, y, z, yaw in mounts]
    return {'id': 'surround', 'name': 'Surround view (four fisheyes)', 'vehicle': vehicle, 'sensors': sensors}


def render_ground(sensor: dict, texture, size=None, supersample: int = 2):
    """A camera's image of the flat ground (z = 0) painted by texture(x, y) -> (k, C) float values (CARLA vehicle
    frame), through the lens model's inverse (core.lens.Intrinsics.unproject), supersampled; rays that miss the ground
    (above the horizon or outside the field) are 0. Returns (h, w, C) float32 at size (default the sensor's)."""
    intr = LENS.intrinsics(sensor)
    w, h = size or (intr.width, intr.height)
    R, pos = _pose_of(sensor)
    ss = max(1, int(supersample))
    acc = None
    for a in range(ss):
        for b in range(ss):
            uu, vv = np.meshgrid((np.arange(w) + (a + 0.5) / ss) * intr.width / w,
                                 (np.arange(h) + (b + 0.5) / ss) * intr.height / h)
            d, th = intr.unproject(uu, vv)
            dv = d.reshape(-1, 3) @ R.T                          # optical -> vehicle
            hit = (dv[:, 2] < -1e-6) & (th.reshape(-1) <= intr.max_theta)
            t = np.where(hit, -pos[2] / np.where(hit, dv[:, 2], -1.0), 0.0)
            X = pos[0] + t * dv[:, 0]
            Y = pos[1] + t * dv[:, 1]
            val = np.asarray(texture(X, Y), np.float64).reshape(len(X), -1)
            val[~hit] = 0.0
            acc = val if acc is None else acc + val
    return (acc / (ss * ss)).reshape(h, w, -1).astype(np.float32)


def checkerboard(square_m: float = 0.5, lo: float = 40.0, hi: float = 215.0):
    """texture(x, y) -> (k, 3): a checkerboard of square_m squares (render_ground)."""
    def tex(x, y):
        c = ((np.floor(np.asarray(x) / square_m) + np.floor(np.asarray(y) / square_m)) % 2).astype(np.float64)
        v = lo + (hi - lo) * c
        return np.stack([v, v * 0.9 + 10, v * 0.8 + 20], -1)
    return tex


# ================================================================================================ runs
def run_frames(run_dir: str, cameras: list, step: int):
    """{camera id: RGB uint8} of a recorded run's step (frames.jsonl's files '<camera>/rgb'), cameras without a frame
    at that step left out."""
    import cv2
    from .exporters import ExportError, Run
    try:
        run = Run(run_dir)
    except ExportError as e:
        raise BevError(str(e)) from None
    rec = next((r for r in run.frames if int(r['step']) == int(step)), None)
    if rec is None:
        raise BevError(f'{run_dir}: no step {step}')
    out = {}
    for c in cameras:
        rel = run.file(rec, c, 'rgb')
        if rel is None:
            continue
        bgr = cv2.imread(run.p(rel), cv2.IMREAD_COLOR)
        if bgr is not None:
            out[c] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out


def run_rig(run_dir: str) -> dict:
    from .exporters import ExportError, Run
    try:
        return Run(run_dir).rig
    except ExportError as e:
        raise BevError(str(e)) from None


def stitch_run(run_dir: str, out_dir: str, res: float = RES_M, margin: float = MARGIN_M, blend_m=None,
               balance: bool = False, device: str = 'auto', every: int = 1, steps=None, seams: bool = False,
               emitter=None, cancel=None) -> dict:
    """Stitch every (every-th) step of a recorded run whose rig has two or more cameras into out_dir/<step>.png, with
    bev.json (the tables' summary, timings, gains). seams: draw the seams on the images."""
    import cv2
    from .exporters import ExportError, Run
    from .provenance import write_json
    try:
        run = Run(run_dir)
    except ExportError as e:
        raise BevError(str(e)) from None
    cams = [s['id'] for s in cameras_of(run.rig)]
    if len(cams) < 2:
        raise BevError(f'{run_dir}: its rig has {len(cams)} camera(s); a surround view needs two or more')
    lut = build_lut(run.rig, res, margin=margin, blend_m=blend_m)
    st = Stitcher(lut, device)
    recs = [r for r in run.frames if any(run.file(r, c, 'rgb') for c in cams)]
    if steps is not None:
        want = set(int(s) for s in steps)
        recs = [r for r in recs if int(r['step']) in want]
    recs = recs[::max(1, int(every))]
    if not recs:
        raise BevError(f'{run_dir}: no recorded camera frames to stitch')
    os.makedirs(out_dir, exist_ok=True)
    ms, rows = [], []
    for n, r in enumerate(recs):
        if cancel and cancel():
            raise KeyboardInterrupt
        frames = run_frames(run_dir, cams, r['step'])
        res_ = st.stitch(frames, gains='auto' if balance else None)
        ms.append(res_.ms)
        img = res_.rgb()
        if seams:
            img = seam_overlay(lut, img)
        rel = f'{int(r["step"]):06d}.png'
        cv2.imwrite(os.path.join(out_dir, rel), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append({'step': int(r['step']), 'file': rel, 'cameras': sorted(frames), 'ms': round(res_.ms, 3),
                     'gains': res_.gains, 'gains_clipped': res_.clipped})
        if emitter is not None:
            emitter.progress((n + 1) / len(recs), f'step {r["step"]} ({n + 1} of {len(recs)})')
    doc = {'schema': SCHEMA + '#run', 'run': os.path.abspath(run_dir), 'tables': lut.summary(),
           'backend': st.backend + ':' + st.device, 'frames': rows,
           'ms_median': round(float(np.median(ms)), 3),
           'fps_median': round(1000.0 / max(float(np.median(ms)), 1e-9), 2),
           'timing_note': 'per frame, reading the PNGs from disk excluded; the first frame includes set-up',
           'limits': LIMITS_NOTE}
    write_json(os.path.join(out_dir, 'bev.json'), doc)
    return {'out': out_dir, 'frames': len(rows), 'fps_median': doc['fps_median'], 'backend': doc['backend'],
            'seam_source': lut.seam_source}


def bench(rig: dict | None = None, frames: int = 50, device: str = 'auto', res: float = RES_M,
          size=(1920, 1080), margin: float = MARGIN_M, warmup: int = 5) -> dict:
    """Frames per second of the stitcher on synthetic frames of a rig's cameras (default surround_rig at size): the
    frames arrive as numpy arrays in host memory (their upload to the GPU is timed), the stitched image stays on the
    device (its download is not timed); 'resident' times frames already on the GPU."""
    rig = rig or surround_rig(*size)
    lut = build_lut(rig, res, margin=margin)
    rng = np.random.default_rng(0)
    imgs = {i: rng.integers(0, 256, (lut.sensor[i][1], lut.sensor[i][0], 3), dtype=np.uint8) for i in lut.ids}
    st = Stitcher(lut, device)
    out = {'backend': st.backend, 'device': st.device, 'half': st.half, 'cameras': len(lut.ids),
           'camera_size': [list(lut.sensor[i]) for i in lut.ids][0], 'bev_size': [lut.nx, lut.ny],
           'resolution_m': lut.res, 'frames': int(frames)}
    for _ in range(warmup):
        st.stitch(imgs)
    t0 = time.perf_counter()
    for _ in range(int(frames)):
        st.stitch(imgs, sync=False)
    if st.device == 'cuda':
        import torch
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out['fps'] = round(frames / dt, 2)
    out['ms_per_frame'] = round(1e3 * dt / frames, 3)
    if st.backend == 'torch' and st.device == 'cuda':
        import torch
        dev_imgs = {i: torch.from_numpy(v).to('cuda').permute(2, 0, 1).contiguous() for i, v in imgs.items()}
        for _ in range(warmup):
            st.stitch(dev_imgs)
        t0 = time.perf_counter()
        for _ in range(int(frames)):
            st.stitch(dev_imgs, sync=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out['fps_resident'] = round(frames / dt, 2)
        out['peak_gpu_memory_gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        out['gpu'] = torch.cuda.get_device_name(0)
    return out


# ================================================================================================ the CLI
def _load_rig(spec: str, project: str | None):
    """A rig: an id in the project (--project), or a rig JSON file (falcon.rig/1 or a bare rig object)."""
    if project:
        from .project import Project
        p = Project.load(project)
        r = next((x for x in p.doc['rigs'] if x['id'] == spec or x.get('name') == spec), None)
        if r is None:
            raise BevError(f'{project}: no rig {spec!r} (rigs: {", ".join(x["id"] for x in p.doc["rigs"])})')
        return r
    if spec and os.path.isfile(spec):
        from .provenance import read_json
        doc = read_json(spec)
        return doc.get('rig', doc)
    if spec in ('surround', ''):
        return surround_rig()
    raise BevError(f'{spec}: give a rig JSON file, or a rig id with --project')


def main_bev(argv) -> int:
    """`falcon bev stitch RUN | export RIG | bench` (docs/bev.md). Exit codes: 0 ok, 2 failed or a usage error, 3
    cancelled."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon bev', description='the bird\'s-eye surround view of a rig\'s cameras '
                                                                '(docs/bev.md)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--res', type=float, default=RES_M, help='ground metres per pixel (default %(default)s)')
        p.add_argument('--margin', type=float, default=MARGIN_M,
                       help='metres of ground shown beyond the body on every side (default %(default)s)')
        p.add_argument('--blend', type=float, help='the seam blend width, m (default: the rig\'s, else 0.2)')
    s = sub.add_parser('stitch', help='stitch a recorded run\'s camera frames into bird\'s-eye images')
    s.add_argument('run', help='a falcon-raw run whose rig has two or more cameras')
    s.add_argument('--out', '-o', help='the folder for <step>.png and bev.json (default RUN/bev)')
    s.add_argument('--every', type=int, default=1)
    s.add_argument('--step', type=int, action='append', help='only this step (repeatable)')
    s.add_argument('--balance', action='store_true', help='estimate and apply exposure gains per frame')
    s.add_argument('--seams', action='store_true', help='draw the seams on the images')
    s.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu', 'torch-cpu'))
    common(s)
    e = sub.add_parser('export', help='write a rig\'s remap tables and the Apache-2.0 reference stitcher')
    e.add_argument('rig', help='a rig id (with --project) or a rig JSON file')
    e.add_argument('--project', help='the project (.falcon.json) holding the rig')
    e.add_argument('--out', '-o', required=True)
    e.add_argument('--no-runtime', action='store_true', help='the tables only, without stitcher/')
    common(e)
    b = sub.add_parser('bench', help='frames per second of the stitcher on synthetic frames')
    b.add_argument('--rig', default='surround', help='a rig id (with --project) or JSON file (default: a synthetic '
                                                     'four-fisheye rig)')
    b.add_argument('--project')
    b.add_argument('--size', default='1920x1080', help='WIDTHxHEIGHT of the synthetic rig\'s fisheyes')
    b.add_argument('--frames', type=int, default=50)
    b.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu', 'torch-cpu'))
    common(b)
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return P.OK if ex.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello(f'bev {a.cmd}')
    locks = []
    try:
        if a.cmd == 'stitch':
            if resolve_device(a.device)[1] == 'cuda':
                locks = _gpu_locks(em)
            out = a.out or os.path.join(a.run, 'bev')
            res = stitch_run(a.run, out, a.res, a.margin, a.blend, a.balance, a.device, a.every, a.step, a.seams,
                             emitter=em)
            em.result('ok', out=out, summary=res)
            return P.OK
        if a.cmd == 'export':
            rig = _load_rig(a.rig, a.project)
            lut = build_lut(rig, a.res, margin=a.margin, blend_m=a.blend)
            paths = export(lut, a.out, rig, runtime=not a.no_runtime)
            em.result('ok', out=a.out, files=len(paths), summary=lut.summary())
            return P.OK
        w, _, h = a.size.lower().partition('x')
        rig = surround_rig(int(w), int(h)) if a.rig == 'surround' and not a.project else _load_rig(a.rig, a.project)
        if resolve_device(a.device)[1] == 'cuda':
            locks = _gpu_locks(em)
        res = bench(rig, a.frames, a.device, a.res, margin=a.margin)
        print(json.dumps(res, indent=1), flush=True)
        em.result('ok', summary=res)
        return P.OK
    except KeyboardInterrupt:
        em.result('cancelled')
        return P.CANCELLED
    except (BevError, ValueError, OSError, KeyError) as ex:
        print(f'failed: {ex}', flush=True)
        em.result('failed', error=str(ex))
        return P.FAILED
    finally:
        for lk in locks:
            lk.release()


def _gpu_locks(em):
    from .training import CancelFlag, gpu_locks
    return gpu_locks(em, CancelFlag(), held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
