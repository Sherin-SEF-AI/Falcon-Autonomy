"""A reconstruction from a dashcam clip: a short segment of a consumer dashcam's video (no calibration, ~1 Hz GPS)
posed by monocular visual odometry, made metric and geolocated by its GPS, reconstructed by core.recon and placed on
an imported world's map (docs/reconstruction.md, "From a dashcam"). numpy, OpenCV and scipy; torch for the training.

    from_dashcam('~/Videos/dashcam-videos/Movie_F/clip.MP4', '~/FalconWorkspace/recon/dashcam-x',
                 start=34, seconds=20)

The steps, each recorded in dashcam.json:

1. **Frames and GPS** from core.dashcam (the dashcam ingest: its GPS track on the video's clock and its frame
   extraction): every frame at `rate` per second over [start, start + seconds), decoded at `image_size`; the fixes
   in the window.
2. **A static mask**: pixels that do not move while the car does — the bonnet, a burnt-in text overlay — found as
   the pixels whose mean absolute difference between frames 0.5 s apart stays below static_level grey levels,
   keeping the part that touches the bottom edge, plus a fixed bottom band (this camera's bonnet reflects the scene
   and is not found); corners are never taken there and the posed images are cropped above it. A corridor of the
   road ahead is left out of the odometry (not of the images): the traffic the camera follows sits there.
3. **Intrinsics**: self-calibrated on the segment (core.recon.vo.self_calibrate: f and the radial k1 free; the
   principal point at the image centre is assumed) from a start of start_hfov degrees, or given (`intrinsics`: a JSON
   file written by an earlier run, or 'f=…,k1=…,k2=…' in the clip's native pixels).
4. **Visual odometry** with those intrinsics (core.recon.vo.run_vo), a global bundle adjustment, every frame
   re-placed by PnP.
5. **The road plane** among the points (vo.ground_plane) gives "up" and the camera's height in VO units.
6. **The GPS alignment** (vo.align_to_gps): a robust similarity from the levelled VO track to the fixes, with the
   GPS latency searched over ±1.5 s. Its residuals are the honest measure of the poses' metric accuracy at the
   GPS's scale (the GPS has metres of error itself).
7. **Posed images**: the frames undistorted to a pinhole and cropped to the valid area above the bonnet, written with
   their poses in a local east-north-up frame about the segment's first fix (z = 0 on the road plane) as a 'posed'
   source (core.recon.data.posed_capture) with the odometry's points to start from and, per frame, a mask of
   independently moving pixels (dense Farnebäck flow off its epipolar line by more than motion_px — catches crossing
   traffic, not traffic moving along the camera's own line).
8. **Training** (core.recon.train) and **novel views**: each of a few frames re-rendered with the ego moved
   sideways and the camera raised / lowered (renders/novel-*.png).
9. **On the map**: the geolocated track converted into the imported world's frame (core.dashcam.to_world), whether it
   is inside the import box, its distance from it, and — when it is near the imported lanes — log replay's map match
   (core.logreplay.mapmatch) and the share of the reconstruction's road-plane points that fall on the imported lanes.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import vo as V

SCHEMA = 'falcon.dashcam_recon/1'
DEFAULTS = {
    'rate': None, 'image_every': 3, 'image_size': [1920, 1080], 'vo_width': 960, 'out_width': 960,
    'intrinsics': 'auto', 'start_hfov': 110.0, 'hfov_grid': [90, 100, 110, 120, 130], 'static_level': 3.0,
    'bottom_band': 0.09, 'traffic_corridor': [0.34, 0.66, 0.55, 1.0],
    'motion_masks': True, 'motion_px': 2.0, 'motion_max_share': 0.4, 'crop_margin_px': 6, 'max_point_range_m': 150.0,
    'train': True,
    'iterations': 7000, 'width': 640, 'seed': 0, 'device': 'auto', 'project': None, 'world': None,
    'novel_lateral_m': [-1.0, -0.5, 0.5, 1.0], 'novel_up_m': [-0.5, 0.5], 'novel_frames': 3, 'keep_raw': False,
    'vo': {}, 'gps_sigma_m': 3.0, 'height_sigma_m': 2.0, 'camera_height_m': 1.3,
}
NATIVE = (3840, 2160)


class DashcamReconError(ValueError):
    pass


class _Quiet:
    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass

    def metric(self, *a, **k):
        pass


def default_out(clip: str, start: float) -> str:
    stem = os.path.splitext(os.path.basename(clip))[0]
    return os.path.expanduser(f'~/FalconWorkspace/recon/dashcam-{stem}-{start:g}s')


def normalise(s: dict | None) -> dict:
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in (s or {}).items():
        if v is None:
            continue
        if k not in DEFAULTS:
            raise DashcamReconError(f'unknown setting {k!r}')
        out[k] = v
    return out


# ================================================================================================ frames
def read_segment(clip: str, start: float, seconds: float, rate, size, raw_dir: str, vo_width: int,
                 image_every: int = 3, emitter=None, cancel=None) -> dict:
    """Frames through core.dashcam (every frame when rate is None, else `rate` a second): grayscale copies at
    vo_width for the odometry, and every image_every-th frame saved as JPEG in raw_dir at `size` for the
    reconstruction; the GPS fixes within the window (±2 s), with times relative to `start`."""
    import cv2
    from .. import dashcam as DC
    em = emitter or _Quiet()
    trk = DC.track(clip)
    if not trk['fixes']:
        raise DashcamReconError(f'{clip}: no GPS fixes (the alignment needs them)')
    pr = DC.probe(clip)
    if pr.get('duration_s') and start + seconds > pr['duration_s'] + 1e-6:
        raise DashcamReconError(f'{clip} is {pr["duration_s"]:.1f} s long; {start} + {seconds} s is past its end')
    os.makedirs(raw_dir, exist_ok=True)
    W, H = int(size[0]), int(size[1])
    vw = int(vo_width)
    vh = int(round(H * vw / W))
    frames, grays = [], []
    info = {}
    t0 = time.perf_counter()
    n_exp = int(round(seconds * (rate or pr.get('fps') or 30.0)))
    for n, t, rgb in DC.iter_frames(clip, rate, (W, H), start, seconds, info=info):
        if cancel and cancel():
            raise KeyboardInterrupt()
        if t >= start + seconds - 1e-6:
            break
        k = len(frames)
        p = None
        if k % max(1, image_every) == 0:
            p = os.path.join(raw_dir, f'{n:06d}.jpg')
            cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        grays.append(cv2.cvtColor(cv2.resize(rgb, (vw, vh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY))
        frames.append({'n': int(n), 't': float(t) - start, 'raw': p})
        if len(frames) % 60 == 0:
            em.progress(0.02 * len(frames) / max(n_exp, 1), f'decoded {len(frames)} of {n_exp} frames')
    if len(frames) < 10:
        raise DashcamReconError(f'{clip}: only {len(frames)} frames in [{start}, {start + seconds}) s')
    fx = [dict(f, t=f['t'] - start) for f in trk['fixes'] if start - 2.0 <= f['t'] <= start + seconds + 2.0]
    return {'frames': frames, 'grays': grays, 'fixes': fx, 'track': trk, 'probe': pr, 'decode': info,
            'decode_s': round(time.perf_counter() - t0, 2), 'vo_size': (vw, vh)}


def static_mask(grays, times, level: float = 3.0, gap_s: float = 0.5, scale: float = 0.5, bottom: float = 0.0,
                corridor=None) -> dict:
    """Pixels that do not change while the car moves (module doc): the mean absolute difference between frames
    gap_s apart (at `scale`, blurred 5 × 5) below `level` grey levels, the part connected to the bottom edge, filled
    downwards per column (smoothed across columns). Returns {'mask' (True: usable), 'hood_rows' (per column, the first
    masked row; the height where none), 'share', 'pairs'}. Pairs where the image hardly changes are skipped (the car
    standing still says nothing). A bonnet that reflects the moving scene changes too and is not found (dense flow
    did no better: on a moving scene it smears motion ~15 px into a still band); `bottom` (a fraction of the height)
    masks a fixed band as well. corridor (x0, x1, y0, y1 as fractions) is a box left out of the odometry only
    ('odometry' in the result), not of the images: the road ahead, where the traffic the camera follows sits
    (heuristic)."""
    import cv2
    n = len(grays)
    H, W = grays[0].shape
    times = np.asarray(times, float)
    pairs = []
    for i in range(n):
        j = int(np.searchsorted(times, times[i] + gap_s))
        if j < n:
            pairs.append((i, j))
    pairs = pairs[::max(1, len(pairs) // 24)]
    diffs = []
    for i, j in pairs:
        a = cv2.resize(grays[i], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(grays[j], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        d = cv2.absdiff(a, b).astype(np.float32)
        if np.median(d) >= 2 * level:
            diffs.append(d)
    h, w = int(round(H * scale)), int(round(W * scale))
    if not diffs:
        rows0 = np.full(W, int(round(H * (1 - bottom))), np.int64)
        return _with_corridor({'mask': np.arange(H)[:, None] < rows0[None, :], 'hood_rows': rows0,
                               'share': round(float(bottom), 4), 'pairs': 0}, corridor)
    acc = cv2.blur(np.mean(np.stack(diffs), 0), (5, 5))
    still = (acc < level).astype(np.uint8)
    still = cv2.morphologyEx(still, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, lab = cv2.connectedComponents(still)
    edge = set(np.unique(lab[-1, :]).tolist()) - {0}
    fixed = np.isin(lab, list(edge)) if edge else np.zeros((h, w), bool)
    rows = np.full(w, h, np.float64)
    for u in range(w):
        col = fixed[:, u]
        if col[-1]:
            above = np.nonzero(~col)[0]
            rows[u] = float(above.max()) + 1 if len(above) else 0.0
    k = max(3, (w // 30) | 1)
    pad = np.r_[np.full(k // 2, rows[0]), rows, np.full(k // 2, rows[-1])]
    rows = np.array([np.median(pad[u:u + k]) for u in range(w)])           # a bonnet's edge is smooth
    rows_full = np.round(np.interp((np.arange(W) + 0.5) * scale - 0.5, np.arange(w), rows) / scale).astype(np.int64)
    rows_full = np.clip(rows_full, 0, H)
    rows_full = np.minimum(rows_full, int(round(H * (1 - bottom))))
    m = np.arange(H)[:, None] < rows_full[None, :]
    return _with_corridor({'mask': m, 'hood_rows': rows_full, 'share': round(float((~m).mean()), 4),
                           'pairs': len(diffs)}, corridor)


def _with_corridor(res: dict, corridor) -> dict:
    m = res['mask'].copy()
    if corridor:
        H, W = m.shape
        x0, x1, y0, y1 = corridor
        m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = False
    res['odometry'] = m
    return res


# ================================================================================================ intrinsics
def parse_intrinsics(spec: str, native=NATIVE) -> V.Intrinsics | None:
    """'auto' -> None; a JSON file (written by an earlier run: native-pixel f, k1, k2) or 'f=…,k1=…,k2=…' (native
    pixels) -> Intrinsics at the native size."""
    if not spec or spec == 'auto':
        return None
    if os.path.isfile(os.path.expanduser(spec)):
        with open(os.path.expanduser(spec), encoding='utf-8') as fh:
            d = json.load(fh)
        d = d.get('native', d)
        return V.Intrinsics(int(d['width']), int(d['height']), float(d['f']), float(d.get('k1', 0)),
                            float(d.get('k2', 0)))
    vals = {}
    for part in spec.split(','):
        k, _, v = part.partition('=')
        vals[k.strip()] = float(v)
    if 'f' not in vals:
        raise DashcamReconError(f'intrinsics {spec!r}: give f=PIXELS (native) and optionally k1=, k2=')
    return V.Intrinsics(native[0], native[1], vals['f'], vals.get('k1', 0.0), vals.get('k2', 0.0))


# ================================================================================================ undistortion
def pinhole_crop(intr: V.Intrinsics, hood_rows=None, out_width: int = 960, margin_px: int = 6) -> dict:
    """The output pinhole of an undistorted, cropped image: the largest rectangle (horizontally centred) inside the
    undistorted image of the frame's border, cut above the bonnet / overlay (hood_rows: per column of intr's image).
    Returns {'K' (fx, fy, cx, cy) pixel-edge, 'size', 'bounds' (x0, x1, y0, y1 normalised)}."""
    W, H = intr.width, intr.height
    e = np.linspace(0, 1, 200)
    left = intr.normalise(np.c_[np.zeros_like(e), e * (H - 1)])
    right = intr.normalise(np.c_[np.full_like(e, W - 1), e * (H - 1)])
    top = intr.normalise(np.c_[e * (W - 1), np.zeros_like(e)])
    bot = intr.normalise(np.c_[e * (W - 1), np.full_like(e, H - 1)])
    x1 = min(-left[:, 0].max(), right[:, 0].min())
    x0 = -x1
    y0 = top[:, 1].max()
    y1 = bot[:, 1].min()
    if hood_rows is not None:
        us = np.arange(len(hood_rows), dtype=float)
        hr = np.asarray(hood_rows, float) - margin_px
        hood = intr.normalise(np.c_[us, np.clip(hr, 0, H - 1)])
        inside = (hood[:, 0] >= x0) & (hood[:, 0] <= x1)
        if inside.any():
            y1 = min(y1, float(hood[inside, 1].min()))
    if y1 <= y0 or x1 <= 0:
        raise DashcamReconError('the undistorted image has no valid area (implausible distortion)')
    f = out_width / (x1 - x0)
    h = int(round(f * (y1 - y0)))
    return {'K': (f, f, -f * x0, -f * y0), 'size': (int(out_width), h), 'bounds': (x0, x1, y0, y1)}


def undistort_maps(intr: V.Intrinsics, pin: dict, supersample: int = 2) -> tuple:
    """cv2.remap maps from the distorted image (intr's size) to the pinhole at `supersample` × its size."""
    W, H = pin['size']
    fx, fy, cx, cy = pin['K']
    s = supersample
    u, v = np.meshgrid((np.arange(W * s) + 0.5) / s, (np.arange(H * s) + 0.5) / s)
    xn = np.c_[((u - cx) / fx).ravel(), ((v - cy) / fy).ravel()]
    q = intr.project_normalised(xn)
    return q[:, 0].reshape(H * s, W * s).astype(np.float32), q[:, 1].reshape(H * s, W * s).astype(np.float32)


def undistort(img, maps, size, interp=None):
    import cv2
    out = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR if interp is None else interp,
                    borderMode=cv2.BORDER_REPLICATE)
    if (out.shape[1], out.shape[0]) != tuple(size):
        out = cv2.resize(out, tuple(size), interpolation=cv2.INTER_AREA if interp is None else interp)
    return out


# ================================================================================================ moving pixels
def motion_mask(g0, g1, intr: V.Intrinsics, T01: np.ndarray, px: float = 2.0, scale: float = 0.5) -> np.ndarray:
    """Pixels of g0 (True: static) whose dense flow to g1 leaves its epipolar line by more than px (at g0's size),
    given T01 (camera 1 <- camera 0). Flow along the epipolar line — traffic moving in the camera's own direction —
    passes the test: not caught."""
    import cv2
    H, W = g0.shape
    a = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(g1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 15, 3, 5, 1.1, 0)
    h, w = a.shape
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float64)
    p0 = np.c_[(uu.ravel() + 0.5) / scale - 0.5, (vv.ravel() + 0.5) / scale - 0.5]
    p1 = p0 + flow.reshape(-1, 2).astype(np.float64) / scale
    x0, x1 = intr.normalise(p0), intr.normalise(p1)
    t = T01[:3, 3]
    if np.linalg.norm(t) < 1e-9:
        return np.ones((H, W), bool)
    E = V.skew(t) @ T01[:3, :3]
    h0 = np.c_[x0, np.ones(len(x0))]
    h1 = np.c_[x1, np.ones(len(x1))]
    l1 = h0 @ E.T
    d = np.abs(np.sum(h1 * l1, 1)) / np.maximum(np.linalg.norm(l1[:, :2], axis=1), 1e-12) * intr.f
    bad = (d > px).reshape(h, w).astype(np.uint8)
    bad = cv2.morphologyEx(bad, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bad = cv2.dilate(bad, np.ones((7, 7), np.uint8))
    bad = cv2.resize(bad, (W, H), interpolation=cv2.INTER_NEAREST)
    return bad == 0


# ================================================================================================ the frames' poses
def ego_frames(viewmats: dict, times: dict, window_s: float = 0.5) -> dict:
    """world <- vehicle per frame: x along the horizontal direction of travel (central differences over ±window_s;
    the optical axis when the car stands still), z up, the origin on the road plane under the camera; and the mount
    (vehicle <- optical)."""
    ks = sorted(viewmats)
    C = {k: -viewmats[k][:3, :3].T @ viewmats[k][:3, 3] for k in ks}
    ts = np.array([times[k] for k in ks])
    out = {}
    for j, k in enumerate(ks):
        a = int(np.searchsorted(ts, ts[j] - window_s))
        b = min(len(ks) - 1, int(np.searchsorted(ts, ts[j] + window_s)))
        d = C[ks[b]] - C[ks[a]]
        if np.hypot(d[0], d[1]) < 0.3:
            d = viewmats[k][2, :3]                  # the optical axis in the world
        yaw = math.atan2(d[1], d[0])
        ego = np.eye(4)
        ego[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        ego[:3, 3] = [C[k][0], C[k][1], 0.0]
        cam = np.linalg.inv(viewmats[k])
        out[k] = (ego, np.linalg.inv(ego) @ cam)
    return out


# ================================================================================================ on the map
def place_on_map(project: str | None, world_id: str | None, lat, lon, course, ground_xyz_ll=None) -> dict:
    """The segment on an imported world (module doc, step 9). ground_xyz_ll: road-plane points as (lat, lon) arrays
    for the share-on-lanes heuristic."""
    from .. import dashcam as DC
    out = {'project': project, 'world': world_id}
    if not project or not world_id:
        out['note'] = 'no project / world given'
        return out
    try:
        w = DC.world_geo(os.path.expanduser(project), world_id)
    except Exception as e:                            # a missing project or a town world: say why
        out['note'] = f'cannot read the world: {e}'
        return out
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x, y, yaw = DC.to_world(w, lat, lon, course)
    out.update(name=w.get('name'), lat0=w['lat0'], lon0=w['lon0'], bbox=w.get('bbox'),
               carla_xy_first=[round(float(x[0]), 2), round(float(y[0]), 2)],
               carla_xy_last=[round(float(x[-1]), 2), round(float(y[-1]), 2)])
    cx, cy = float(np.mean(x)), float(np.mean(y))
    out['distance_from_centre_m'] = round(math.hypot(cx, cy), 1)
    out['bearing_from_centre_deg'] = round((math.degrees(math.atan2(cx, -cy))) % 360, 1)   # from north, clockwise
    if w.get('bbox'):
        s_, w_, n_, e_ = w['bbox']
        inside = (lat >= s_) & (lat <= n_) & (lon >= w_) & (lon <= e_)
        out['inside_box_share'] = round(float(inside.mean()), 3)
        trk = {'clip': 'segment', 'fixes': [{'t': float(k), 'lat': float(a), 'lon': float(b), 'speed_kmh': None,
                                              'course_deg': float(c) if c is not None else None}
                                             for k, (a, b, c) in enumerate(zip(lat, lon, course))]}
        m = DC.match_clip(w, trk)
        out['box_distance_m'] = m.get('box_distance_m')
        out['lane_distance_m'] = m.get('lane_distance_m')
    L = w.get('lanes')
    if L is None or not len(L) or out.get('lane_distance_m') is None or out['lane_distance_m'] > 50:
        out['on_map'] = False
        out['note'] = ('outside the imported area: no imported lanes within 50 m, so no street or lane alignment can '
                       'be measured')
        return out
    from ..logreplay import mapmatch as MM
    out['on_map'] = True
    mm = MM.match(x, y, yaw, L, w.get('lanes_from') or '')
    out['match'] = {k: mm.get(k) for k in ('verdict', 'why', 'share_on_lane_in_direction', 'share_on_lane',
                                           'median_lateral_m', 'p95_lateral_m')}
    out['street'] = _street(w, float(np.median(x)), float(np.median(y)))
    if ground_xyz_ll is not None and len(ground_xyz_ll[0]):
        gx, gy, _ = DC.to_world(w, ground_xyz_ll[0], ground_xyz_ll[1])
        from scipy.spatial import cKDTree
        tree = cKDTree(L[:, :2])
        d, j = tree.query(np.c_[gx, gy])
        a = np.radians(L[j, 2])
        lateral = np.abs((gx - L[j, 0]) * -np.sin(a) + (gy - L[j, 1]) * np.cos(a))
        on = lateral <= L[j, 3] / 2.0 + 0.25
        out['road_points_on_lanes'] = {'points': int(len(gx)), 'share': round(float(on.mean()), 3),
                                       'median_lateral_m': round(float(np.median(lateral)), 2),
                                       'how': 'road-plane points of the reconstruction (within 0.15 m of the plane, '
                                              'within 25 m of the path) whose lateral distance to the nearest '
                                              'imported driving-lane centre point is within its half width + 0.25 m '
                                              '(heuristic: kerbs, verges and parked cars on the plane count against '
                                              'it; lanes the drive did not use count for it)'}
    return out


def place_existing(rec_dir: str, project: str, world_id: str) -> dict:
    """Place an already posed segment (rec_dir/posed) on an imported world after the fact — e.g. once the world that
    covers the footage has been imported. The same measures as step 9, from the saved poses and points; dashcam.json's
    'map' (and its summary) is replaced and the earlier placement kept under 'map_before'."""
    from ..osm import local_projection
    from ..provenance import read_json, write_json
    posed = os.path.join(rec_dir, 'posed')
    with open(os.path.join(posed, 'posed.json'), encoding='utf-8') as fh:
        doc = json.load(fh)
    geo = doc['geo']
    proj = local_projection(float(geo['lat0']), float(geo['lon0']))
    cams = [np.linalg.inv(np.asarray(f['viewmat'], float).reshape(4, 4)) for f in doc['frames']]
    cw = np.array([c[:3, 3] for c in cams])
    egos = [np.asarray(f['ego'], float).reshape(4, 4) for f in doc['frames']]
    head = [(90.0 - math.degrees(math.atan2(e[1, 0], e[0, 0]))) % 360.0 for e in egos]
    lat_c, lon_c = proj.to_geo(cw[:, 0], cw[:, 1])
    glat = glon = np.zeros(0)
    pp = os.path.join(posed, doc.get('points') or 'points.npz')
    if os.path.isfile(pp):
        from scipy.spatial import cKDTree
        X = np.load(pp)['xyz'].astype(float)
        if len(X):
            dmin, _ = cKDTree(cw).query(X)
            near = (np.abs(X[:, 2]) < 0.15) & (dmin < 25.0)
            if near.any():
                glat, glon = proj.to_geo(X[near, 0], X[near, 1])
    m = place_on_map(project, world_id, lat_c, lon_c, head, (glat, glon))
    m['placed_after'] = {'how': 'falcon recon place: the saved poses (posed.json) and points (points.npz)',
                         'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    rp = os.path.join(rec_dir, 'dashcam.json')
    rep = read_json(rp) if os.path.isfile(rp) else {}
    if rep.get('map'):
        rep['map_before'] = rep['map']
    rep['map'] = m
    if isinstance(rep.get('summary'), dict):
        rep['summary']['map'] = {k: m.get(k) for k in ('on_map', 'distance_from_centre_m', 'bearing_from_centre_deg',
                                                        'box_distance_m', 'street', 'note')}
    write_json(rp, rep)
    return m


def _polyline_distance(xs, ys, x: float, y: float) -> float:
    """The distance from (x, y) to a polyline's segments."""
    if not len(xs):
        return float('inf')
    if len(xs) == 1:
        return float(math.hypot(xs[0] - x, ys[0] - y))
    a = np.c_[xs[:-1], ys[:-1]]
    b = np.c_[xs[1:], ys[1:]]
    ab = b - a
    t = np.clip(np.sum((np.array([x, y]) - a) * ab, 1) / np.maximum(np.sum(ab * ab, 1), 1e-12), 0, 1)
    q = a + t[:, None] * ab
    return float(np.min(np.hypot(q[:, 0] - x, q[:, 1] - y)))


def _street(w: dict, x: float, y: float) -> str | None:
    """The name of the imported road nearest a CARLA-frame point (the world's graph.json), or None."""
    xodr = w.get('xodr') or ''
    gp = os.path.join(os.path.dirname(xodr), 'graph.json')
    if not os.path.isfile(gp):
        return None
    with open(gp, encoding='utf-8') as fh:
        g = json.load(fh)
    best, name = float('inf'), None
    for r in g.get('roads', []):
        xs, ys = np.asarray(r['x'], float), -np.asarray(r['y'], float)       # graph.json: OpenDRIVE x, y (north)
        d = _polyline_distance(xs, ys, x, y)
        if d < best:
            best, name = d, r.get('name')
    return name if best < 30 else None


def up_vector(vo: V.VOResult, frames) -> np.ndarray:
    """"Up" in the odometry's frame: the cameras' mean up axis (−y) made perpendicular to the mean direction of
    travel (roads are close to level over a segment; the camera's roll on its mount is assumed small)."""
    up = -np.mean([V.rodrigues(vo.poses[i][:3])[0].T @ np.array([0.0, 1.0, 0.0]) for i in frames], 0)
    C = vo.centres(frames)
    d = C[-1] - C[0]
    if np.linalg.norm(d) > 1e-9:
        d = d / np.linalg.norm(d)
        up = up - (up @ d) * d
    return up / np.linalg.norm(up)


def road_plane(points, centres, height_range=(0.5, 3.0), lateral_m=8.0, thresh_m=0.12, seed=0) -> dict | None:
    """The road plane in a metric, levelled map (z up): RANSAC over the points between height_range below the
    cameras and within lateral_m of their path, normal within 15° of vertical. {'normal', 'd', 'height' (the cameras'
    mean height above it), 'height_std', 'inliers', 'candidates'} or None (heuristic thresholds)."""
    from scipy.spatial import cKDTree
    X, C = np.asarray(points, float), np.asarray(centres, float)
    if len(X) < 30:
        return None
    dist, j = cKDTree(C[:, :2]).query(X[:, :2])
    below = C[j, 2] - X[:, 2]
    cand = X[(dist < lateral_m) & (below > height_range[0]) & (below < height_range[1])]
    if len(cand) < 20:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(300):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n = n / L * (1 if n[2] >= 0 else -1)
        if n[2] < math.cos(math.radians(15)):
            continue
        d = -n @ p[0]
        k = int((np.abs(cand @ n + d) < thresh_m).sum())
        if k > best_n:
            best, best_n = (n, d), k
    if best is None or best_n < 15:
        return None
    n, d = best
    inl = cand[np.abs(cand @ n + d) < thresh_m]
    c0 = inl.mean(0)
    _, _, Vt = np.linalg.svd(inl - c0)
    n2 = Vt[-1] if Vt[-1][2] > 0 else -Vt[-1]
    d2 = -n2 @ c0
    h = C @ n2 + d2
    return {'normal': n2, 'd': float(d2), 'height': float(np.mean(h)), 'height_std': float(np.std(h)),
            'inliers': int(len(inl)), 'candidates': int(len(cand)), 'tilt_deg': round(math.degrees(math.acos(
                min(1.0, n2[2]))), 2)}


def pose_segment(tr: V.Tracks, intr: V.Intrinsics, times, fixes: list, s: dict, emitter=None, cancel=None,
                 free=(), fuse: bool = True) -> dict:
    """Steps 4–6 of the module doc on tracked frames: the odometry (scale held by the GPS speed), its similarity to
    the GPS positions (the honest residuals), then the GPS-aided adjustment (positions and speed) that gives the
    poses the reconstruction uses, in a local east-north-up frame about the first fix. Returns a dict (see the code)
    with `vo` holding the final metric poses."""
    from ..osm import local_projection
    times = np.asarray(times, float)
    gt = np.array([f['t'] for f in fixes], float)
    spd = np.array([(f['speed_kmh'] or 0.0) / 3.6 for f in fixes])
    dist = V.speed_distance(gt, spd)
    t0 = time.perf_counter()
    vo = V.run_vo(tr, intr, s['vo'], emitter, cancel, times=times, distance=dist)
    t_vo = time.perf_counter() - t0
    placed = sorted(vo.poses)
    span = float(times[placed[-1]] - times[placed[0]]) if placed else 0.0
    if len(placed) < 30 or span < 3.0:       # too short to align with ~1 Hz fixes: say why rather than fail later
        moving = spd[(gt >= times[0]) & (gt <= times[-1])]
        raise V.VOError(
            f'the odometry posed only {len(placed)} of {len(times)} frames ({span:.1f} s'
            + (f': it started at frame {placed[0]}' if placed else '')
            + (f' and lost its track at frame {vo.lost_at}' if vo.lost_at is not None else '')
            + f') — too little to align with the GPS. It tracked {tr.stats.get("mean_tracked", 0):.0f} points a frame '
              f'on average'
            + (f' at a median GPS speed of {3.6 * float(np.median(moving)):.0f} km/h' if len(moving) else '')
            + '; rain and wipers on the windscreen, darkness and slow or stopped traffic (too little parallax to '
              'start) all defeat it. Choose a drier or faster stretch.')
    gb = V.global_ba(vo, tr, intr, (), s['vo'])
    C = vo.centres(placed)
    up = up_vector(vo, placed)
    lat0, lon0 = fixes[0]['lat'], fixes[0]['lon']
    proj = local_projection(lat0, lon0)
    ge, gn = proj.to_local(np.array([f['lat'] for f in fixes]), np.array([f['lon'] for f in fixes]))
    gxy = np.c_[ge, gn]
    al = V.align_to_gps(C, times[placed], gt, gxy, up, seed=s['seed'])
    vo_only = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in al.items()
               if k not in ('R', 't', 'residuals_m')}
    vo_only['residuals_m'] = [round(float(v), 3) for v in al['residuals_m']]
    vo_only['vo_path_m'] = round(al['s'] * float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))), 2)
    vo_only['heading'] = heading_check(vo, placed, times, fixes, al['R'])
    vo_only['placed_frames'] = len(placed)
    vo_only['lost_at_frame'] = vo.lost_at
    if not fuse:
        return {'vo': vo, 'vo_only': vo_only, 'global_ba': gb, 'seconds': {'vo': round(t_vo, 2)}}
    # metric, levelled, geolocated; the cameras at an assumed height until the road plane is found
    V.transform(vo, al['s'], al['R'], np.r_[al['t'][:2], 0.0])
    ids, X = vo.point_array()
    rp = road_plane(X, vo.centres(placed))
    rp_found = rp
    if rp is not None and not 0.8 <= rp['height'] <= 2.5:      # not a car's windscreen camera: not the road
        rp = None
    h = rp['height'] if rp else s['camera_height_m']
    zshift = -(rp['d'] if rp else float(np.mean(vo.centres(placed)[:, 2])) - h)
    V.transform(vo, 1.0, np.eye(3), np.array([0.0, 0.0, zshift]))
    t1 = time.perf_counter()
    fu = V.gps_aided_ba(vo, tr, intr, times, gt, gxy, al['lag_s'], h, s['gps_sigma_m'], s['height_sigma_m'], free,
                        s['vo'])
    t_fuse = time.perf_counter() - t1
    placed = [i for i in sorted(vo.poses) if min(vo.keyframes) <= i <= max(vo.keyframes)]
    Cf = vo.centres(placed)
    tq = gt + al['lag_s']                                  # fix time + lag = the frames' time
    ok = (tq >= times[placed][0]) & (tq <= times[placed][-1])
    fx_ = np.c_[np.interp(tq[ok], times[placed], Cf[:, 0]), np.interp(tq[ok], times[placed], Cf[:, 1])]
    res = np.linalg.norm(fx_ - gxy[ok], axis=1)
    ids, X = vo.point_array()
    rp2 = road_plane(X, Cf)
    fused = {'rms_m': round(float(np.sqrt(np.mean(res ** 2))), 3), 'median_m': round(float(np.median(res)), 3),
             'max_m': round(float(res.max()), 3), 'fixes': int(ok.sum()), 'priors': fu['priors'],
             'rounds': fu['rounds'], 'sigma_xy_m': s['gps_sigma_m'], 'sigma_z_m': s['height_sigma_m'],
             'heading': heading_check(vo, placed, times, fixes, np.eye(3)),
             'reprojection_all_frames': V.reprojection_stats(vo, tr, fu['intr'], placed),
             'note': 'residuals of the GPS-aided poses against the same fixes: not independent (the fixes are in the '
                     'adjustment); the odometry-only alignment above is the independent check'}
    return {'vo': vo, 'intr': fu['intr'], 'proj': proj, 'lat0': lat0, 'lon0': lon0, 'gt': gt, 'gxy': gxy,
            'lag_s': al['lag_s'], 'vo_only': vo_only, 'fused': fused, 'global_ba': gb,
            'road_plane': (None if rp2 is None else {k: (round(v, 4) if isinstance(v, float) else v)
                                                      for k, v in rp2.items() if k not in ('normal',)}),
            'camera_height_m': round(h, 3), 'height_from': 'road plane' if rp else 'assumed',
            'road_plane_found': (None if rp_found is None else {k: (round(v, 4) if isinstance(v, float) else v)
                                                                for k, v in rp_found.items() if k != 'normal'}),
            'seconds': {'vo': round(t_vo, 2), 'global_ba': round(gb['seconds'], 2), 'fused_ba': round(t_fuse, 2)}}


def heading_check(vo: V.VOResult, placed, times, fixes, R_world) -> dict | None:
    """The camera's yaw change against the GPS course change over the segment (moving fixes only): a wrong focal
    length scales the odometry's rotations, so on a turn their ratio checks f; the course is the direction of travel
    (a lane change or a camera yawed on its mount shows as a few degrees)."""
    gt = np.array([f['t'] for f in fixes], float)
    okc = (gt >= times[placed][0]) & (gt <= times[placed][-1]) & \
        (np.array([f['speed_kmh'] or 0 for f in fixes]) > 10)
    if okc.sum() < 2:
        return None
    yaw = V.heading_change([R_world @ V.rodrigues(vo.poses[i][:3])[0].T for i in placed])
    crs = -np.degrees(np.unwrap(np.radians([f['course_deg'] or 0 for f in fixes])))
    tt = gt[okc]
    yv = np.interp(tt, times[placed], yaw)
    yv = yv - yv[0]
    yg = crs[okc] - crs[okc][0]
    return {'gps_course_change_deg': round(float(yg[-1]), 2), 'camera_yaw_change_deg': round(float(yv[-1]), 2),
            'rms_difference_deg': round(float(np.sqrt(np.mean((yv - yg) ** 2))), 2), 'over_s': round(float(tt[-1] -
                                                                                                         tt[0]), 1)}


# ================================================================================================ the pipeline
def focal_search(tr: V.Tracks, size, times, fixes: list, s: dict, hfovs, k1: float = 0.0, emitter=None,
                 cancel=None, min_turn_deg: float = 30.0) -> dict:
    """The focal search: the odometry run with each candidate horizontal field (k1 fixed). On a segment that turns
    by at least min_turn_deg (GPS course) the candidates are scored by how well the camera's yaw follows the course
    (RMS difference over the segment): a rotation's image motion is f × the angle, so a wrong f scales the turn the
    odometry sees — and each candidate's f × (camera yaw change / course change) is a direct estimate ('turn_f').
    Otherwise by the odometry-only alignment to the GPS (median residual): weak, since the odometry of a camera moving
    forward is nearly insensitive to f (on KITTI raw drive 0015, f off by −20 … +25 % changed the trajectory error by
    under 0.3 m). Candidates whose odometry placed under 90 % of the frames are left out."""
    rows = []
    for h in hfovs:
        if cancel and cancel():
            raise KeyboardInterrupt()
        intr = V.Intrinsics.from_hfov(size[0], size[1], float(h), k1)
        if emitter:
            emitter.progress(0.1, f'focal search: {h:g}° ({intr.f:.1f} px)', force=True)
        try:
            r = pose_segment(tr, intr, times, fixes, s, fuse=False)
            vo_only = r['vo_only']
            hd = vo_only.get('heading') or {}
            turn_f = None
            if abs(hd.get('gps_course_change_deg') or 0.0) >= min_turn_deg:
                turn_f = round(intr.f * hd['camera_yaw_change_deg'] / hd['gps_course_change_deg'], 2)
            rows.append({'hfov_deg': float(h), 'f': round(intr.f, 3), 'median_m': vo_only['median_m'],
                         'rms_m': vo_only['rms_m'], 'placed': vo_only['placed_frames'],
                         'reprojection_median_px': round(r['global_ba']['rounds'][-1]['median_after'], 4),
                         'heading_rms_deg': hd.get('rms_difference_deg'), 'turn_f': turn_f})
        except V.VOError as e:
            rows.append({'hfov_deg': float(h), 'f': round(intr.f, 3), 'error': str(e)})
    ok = [r for r in rows if r.get('median_m') is not None and r['placed'] >= 0.9 * max(x.get('placed', 0)
                                                                                          for x in rows)]
    if not ok:
        raise DashcamReconError('the odometry failed for every focal length tried')
    turn = all(r.get('turn_f') is not None for r in ok)
    key = 'heading_rms_deg' if turn else 'median_m'
    best = min(ok, key=lambda r: r[key])
    near = [r['hfov_deg'] for r in ok if r[key] <= 1.2 * best[key]]
    return {'profile': rows, 'best_hfov_deg': best['hfov_deg'], 'best_f': best['f'], 'scored_by': key,
            'within_20pct_of_best_deg': [min(near), max(near)],
            'turn_f_median': (round(float(np.median([r['turn_f'] for r in ok])), 2) if turn else None)}


def from_dashcam(clip: str, out_dir: str, start: float, seconds: float, settings: dict | None = None,
                 emitter=None, cancel=None, argv=None) -> dict:
    """The whole pipeline (module doc). Returns the summary (also in dashcam.json and run.json)."""
    import cv2
    from .. import dashcam as DC
    from .. import provenance as PV
    from .data import POSED_SCHEMA
    em = emitter or _Quiet()
    s = normalise(settings)
    clip = os.path.abspath(os.path.expanduser(clip))
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    posed = os.path.join(out_dir, 'posed')
    raw = os.path.join(out_dir, 'raw')
    for d in ('images', 'masks'):
        os.makedirs(os.path.join(posed, d), exist_ok=True)
    timings = {}
    rep = {'schema': SCHEMA, 'clip': clip, 'start_s': float(start), 'seconds': float(seconds), 'settings': s,
           'notes': [], 'limits': []}
    run = PV.new_run('recon.from_dashcam', argv or [], s['seed'], s)
    run['provenance']['inputs'] = {'clip': DC.fingerprint(clip)}
    rp = os.path.join(out_dir, 'dashcam.json')
    PV.write_json(os.path.join(out_dir, 'run.json'), run)
    t_all = time.perf_counter()

    # 1. frames and GPS
    em.progress(0.0, f'decoding {seconds:g} s of {os.path.basename(clip)} from {start:g} s', force=True)
    seg = read_segment(clip, start, seconds, s['rate'], s['image_size'], raw, s['vo_width'], s['image_every'], em,
                       cancel)
    frames, grays, fx = seg['frames'], seg['grays'], seg['fixes']
    times = np.array([f['t'] for f in frames])
    timings['decode_s'] = seg['decode_s']
    rep['frames'] = {'decoded': len(frames), 'images_every': s['image_every'], 'decode': seg['decode']}
    if len(fx) < 4:
        raise DashcamReconError(f'only {len(fx)} GPS fixes in the window: the alignment needs at least 4')
    sp = [f['speed_kmh'] or 0 for f in fx]
    rep['gps'] = {'fixes': len(fx), 'speed_kmh_min_max': [round(min(sp), 1), round(max(sp), 1)],
                  'course_change_deg': round(float(np.degrees(np.ptp(np.unwrap(np.radians(
                      [f['course_deg'] or 0 for f in fx]))))), 1)}
    if min(sp) < 5:
        rep['notes'].append('the car is (nearly) stopped somewhere in the window: the odometry cannot move there')
    vw, vh = seg['vo_size']
    # 2. the static mask (bonnet, overlay) and the odometry's traffic corridor
    t1 = time.perf_counter()
    st = static_mask(grays, times, s['static_level'], bottom=s['bottom_band'], corridor=s['traffic_corridor'])
    rep['static_mask'] = {'share': st['share'], 'hood_top_row_min': int(st['hood_rows'].min()), 'of_height': vh,
                          'bottom_band': s['bottom_band'], 'traffic_corridor': s['traffic_corridor'],
                          'how': f'mean |difference| of frames 0.5 s apart < {s["static_level"]} grey levels, '
                                 f'the part touching the bottom edge, filled downwards, and the bottom '
                                 f'{100 * s["bottom_band"]:g} % (this camera\'s bonnet reflects the scene, so the '
                                 f'difference does not find it); the corridor is left out of the odometry only '
                                 f'(heuristics)'}
    # 3. intrinsics
    given = parse_intrinsics(s['intrinsics'])
    guess = V.Intrinsics.from_hfov(vw, vh, s['start_hfov']) if given is None else given.scaled(vw, vh)
    em.progress(0.04, f'tracking {len(grays)} frames', force=True)
    tr = V.track(grays, st['odometry'], guess, s['vo'], cancel)
    timings['track_s'] = tr.stats['seconds']
    rep['tracks'] = tr.stats
    if given is None:
        pl = V.plumb_line(grays, st['mask'], guess)
        k1 = pl['k1'] if pl.get('k1') is not None and abs(pl['k1']) > 2 * pl['std'] and \
            not pl.get('at_grid_edge') else 0.0
        pl['used_k1'] = k1                 # taken only when inside the grid and 2 spreads away from 0
        fs = focal_search(tr, (vw, vh), times, fx, s, s['hfov_grid'], k1, em, cancel)
        intr = V.Intrinsics.from_hfov(vw, vh, fs['best_hfov_deg'], k1)
        if fs.get('turn_f_median'):        # a turn measures f directly (and nearly whatever f the odometry assumed)
            intr = V.Intrinsics(vw, vh, fs['turn_f_median'], k1)
        rep['calibration'] = {'plumb_line': pl, 'focal_search': fs,
                              'how': 'k1 by the plumb-line method at the starting focal length (used when it differs '
                                     'from 0 by more than twice its bootstrap spread), then f by the focal search: on '
                                     'a turn of 30° or more the median of the candidates\' turn estimates f × (camera '
                                     'yaw change / GPS course change), else the candidate with the best odometry-only '
                                     'alignment to the GPS; principal point at the image centre and k2 = 0 assumed'}
        timings['calibration_s'] = round(time.perf_counter() - t1 - timings['track_s'], 2)
    else:
        intr = guess
        rep['calibration'] = {'how': f'given: {s["intrinsics"]}'}
    # 4–6. odometry, the honest alignment, the GPS-aided poses
    em.progress(0.2, 'visual odometry and the GPS alignment', force=True)
    ps = pose_segment(tr, intr, times, fx, s, em, cancel)
    vo, intr = ps['vo'], ps['intr']
    timings.update({f'{k}_s': v for k, v in ps['seconds'].items()})
    rep['intrinsics'] = {'vo': intr.doc(), 'native': intr.scaled(*NATIVE).doc()}
    PV.write_json(os.path.join(out_dir, 'intrinsics.json'),
                  {'schema': 'falcon.dashcam_intrinsics/1', 'native': rep['intrinsics']['native'],
                   'how': rep['calibration'].get('how'), 'clip': clip, 'start_s': start, 'seconds': seconds,
                   'note': 'pass this file as --intrinsics to reuse the estimate for other segments of the camera'})
    rep['vo'] = {k: v for k, v in vo.stats.items() if k != 'local_ba'}
    lb = vo.stats['local_ba']
    rep['vo']['local_ba'] = {'runs': len(lb), 'seconds_mean': round(float(np.mean([b['seconds'] for b in lb])), 4),
                             'rms_before_mean_px': round(float(np.mean([b['rms_before'] for b in lb])), 3),
                             'rms_after_mean_px': round(float(np.mean([b['rms_after'] for b in lb])), 3)} \
        if lb else None
    rep['vo']['per_frame_ms'] = round(1000 * (tr.stats['seconds'] + vo.stats['seconds'] - vo.stats['local_ba_seconds'])
                                      / len(grays), 2)
    rep['vo']['global_ba'] = [{k: r[k] for k in ('rms_before', 'rms_after', 'median_before', 'median_after',
                                                  'seconds', 'iterations', 'observations', 'cameras', 'points_n')
                               if k in r} for r in ps['global_ba']['rounds']]
    rep['vo']['lost_at_frame'] = vo.lost_at
    if vo.lost_at is not None:
        rep['notes'].append(f'the odometry lost track at t = {times[vo.lost_at]:.2f} s: the segment ends there')
    rep['alignment_vo_only'] = ps['vo_only']
    rep['alignment_gps_aided'] = ps['fused']
    rep['ground'] = {'camera_height_m': ps['camera_height_m'], 'from': ps['height_from'],
                     'road_plane_before_fusion': ps['road_plane_found'], 'road_plane_after_fusion': ps['road_plane'],
                     'how': 'RANSAC plane (12 cm) among the points 0.5–3 m below the cameras within 8 m of their '
                            'path; believed when the cameras sit 0.8–2.5 m above it, else 1.3 m is assumed'}
    lat0, lon0, proj = ps['lat0'], ps['lon0'], ps['proj']
    kf0, kf1 = min(vo.keyframes), max(vo.keyframes)
    placed = [i for i in sorted(vo.poses) if kf0 <= i <= kf1]      # outside the keyframes nothing adjusted them
    vms = {i: V.pose_mat(vo.poses[i]) for i in placed}
    egos = ego_frames(vms, {i: float(times[i]) for i in placed})
    # 7. posed images (the decoded frames that were saved and placed)
    em.progress(0.3, 'undistorting and writing the posed images', force=True)
    t1 = time.perf_counter()
    W0, H0 = int(s['image_size'][0]), int(s['image_size'][1])
    intr_img = intr.scaled(W0, H0)
    hood_img = np.interp(np.arange(W0) * vw / W0, np.arange(vw), st['hood_rows'] * H0 / vh)
    pin = pinhole_crop(intr_img, hood_img, int(s['out_width']), s['crop_margin_px'] * H0 / vh)
    maps = undistort_maps(intr_img, pin)
    mask_maps = undistort_maps(intr, pin, 1)            # the odometry's frames to the same pinhole
    recs, n_masked, n_dropped = [], [], 0
    use = [i for i in placed if frames[i]['raw']]
    for i in use:
        if cancel and cancel():
            raise KeyboardInterrupt()
        img = cv2.imread(frames[i]['raw'], cv2.IMREAD_COLOR)
        und = undistort(img, maps, pin['size'])
        name = f'{frames[i]["n"]:06d}'
        cv2.imwrite(os.path.join(posed, 'images', name + '.jpg'), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mrel = None
        if s['motion_masks']:
            j = min(i + 3, placed[-1]) if i + 3 in vo.poses else (i - 3 if i - 3 in vo.poses else None)
            if j is not None and j != i:
                T01 = vms[j] @ np.linalg.inv(vms[i])
                mm = motion_mask(grays[i], grays[j], intr, T01, s['motion_px'])
                mu = undistort(mm.astype(np.uint8) * 255, mask_maps, pin['size'], cv2.INTER_NEAREST)
                share = float((mu < 128).mean())
                if share <= s['motion_max_share']:
                    n_masked.append(share)
                    mrel = f'masks/{name}.png'
                    cv2.imwrite(os.path.join(posed, mrel), mu)
                else:                           # the test failed on this frame (poor relative pose): not used
                    n_dropped += 1
        ego, mount = egos[i]
        recs.append({'index': int(frames[i]['n']), 't': round(float(times[i] - times[use[0]]), 6),
                     'video_t': round(float(times[i] + start), 6), 'image': f'images/{name}.jpg', 'mask': mrel,
                     'K': [round(float(v), 6) for v in pin['K']], 'size': list(pin['size']),
                     'viewmat': vms[i].reshape(-1).tolist(), 'ego': ego.reshape(-1).tolist(),
                     'mount': mount.reshape(-1).tolist(), 'name': f'{os.path.basename(clip)}#{frames[i]["n"]}'})
    timings['undistort_s'] = round(time.perf_counter() - t1, 2)
    rep['posed'] = {'images': len(recs), 'size': list(pin['size']), 'K_pixel_edge': [round(v, 3) for v in pin['K']],
                    'hfov_deg': round(math.degrees(2 * math.atan(pin['size'][0] / 2 / pin['K'][0])), 2)}
    if n_masked:
        rep['motion_masks'] = {'mean_share': round(float(np.mean(n_masked)), 4),
                               'max_share': round(float(np.max(n_masked)), 4), 'frames': len(n_masked),
                               'frames_left_unmasked': n_dropped,
                               'unmasked_when_share_above': s['motion_max_share'],
                               'how': f'dense Farnebäck flow to the frame 0.1 s later more than {s["motion_px"]} px '
                                      f'(at {vw} px wide) off its epipolar line, opened and dilated 3 px; traffic '
                                      f'moving along the camera\'s own line is not caught'}
    seg_name = f'{os.path.splitext(os.path.basename(clip))[0]}@{start:g}+{seconds:g}s'
    gxy = ps['gxy']
    doc = {'schema': POSED_SCHEMA, 'name': seg_name, 'camera': 'dashcam-front',
           'world': (f'local east-north-up about the segment\'s first GPS fix ({lat0:.6f}, {lon0:.6f}; transverse '
                     f'Mercator), metres; z = 0 on the road plane under the camera; poses from the GPS-aided '
                     f'adjustment of the odometry'),
           'geo': {'lat0': lat0, 'lon0': lon0, 'proj': proj.proj},
           'source': {'clip': clip, 'start_s': start, 'seconds': seconds},
           'masks': 'moving pixels off their epipolar lines (dense flow); the bonnet and overlay cropped away'
           if s['motion_masks'] else 'none (the bonnet and overlay cropped away)',
           'notes': ['poses from monocular visual odometry (scale held by the GPS speed), adjusted towards the ~1 Hz '
                     'GPS positions (dashcam.json has the residuals)', 'intrinsics estimated or given (dashcam.json)'],
           'left_out': [{'what': 'traffic moving along the camera\'s line', 'why': 'the epipolar test cannot see it'},
                        {'what': 'rolling shutter, exposure changes', 'why': 'not modelled'}],
           'heuristic': [{'what': 'static mask, corridor, motion mask, road plane', 'how': 'dashcam.json'}],
           'points': 'points.npz', 'frames': recs,
           'gps': [{'t': round(float(f['t'] - times[use[0]]), 3), 'e': round(float(e), 3), 'n': round(float(n), 3)}
                   for f, (e, n) in zip(fx, gxy)]}
    with open(os.path.join(posed, 'posed.json'), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh)
    # the odometry's points, coloured from the posed images
    from scipy.spatial import cKDTree
    from . import data as D
    cap = D.posed_capture(posed)
    _, Xw = vo.point_array()
    cw = vo.centres(placed)
    dmin, _ = cKDTree(cw).query(Xw)
    keep = (dmin < s['max_point_range_m']) & (Xw[:, 2] > -3.0)
    Xk = Xw[keep]
    rgb = D.colour_points(Xk, D.nearest_frames(cap.frames, cap.frames[len(cap.frames) // 2].t, 6,
                                               spread_s=seconds / 2)) if len(Xk) else np.zeros((0, 3))
    np.savez_compressed(os.path.join(posed, 'points.npz'), xyz=Xk.astype(np.float32), rgb=rgb.astype(np.float32))
    rep['points'] = {'mapped': int(len(Xw)), 'kept': int(len(Xk)), 'kept_within_m_of_path': s['max_point_range_m']}
    if not s['keep_raw']:
        import shutil
        shutil.rmtree(raw, ignore_errors=True)
    # 9. on the map
    lat_c, lon_c = proj.to_geo(cw[:, 0], cw[:, 1])
    head = [(90.0 - math.degrees(math.atan2(egos[i][0][1, 0], egos[i][0][0, 0]))) % 360.0 for i in placed]
    near = (np.abs(Xk[:, 2]) < 0.15) & (dmin[keep] < 25.0)
    glat, glon = proj.to_geo(Xk[near, 0], Xk[near, 1]) if near.any() else (np.zeros(0), np.zeros(0))
    rep['geo'] = {'lat0': lat0, 'lon0': lon0, 'start': [round(float(lat_c[0]), 6), round(float(lon_c[0]), 6)],
                  'end': [round(float(lat_c[-1]), 6), round(float(lon_c[-1]), 6)]}
    rep['map'] = place_on_map(s['project'], s['world'], lat_c, lon_c, head, (glat, glon))
    timings['poses_total_s'] = round(time.perf_counter() - t_all, 2)
    rep['timings'] = timings
    summary = {'frames': len(placed), 'images': len(recs), 'intrinsics_native': rep['intrinsics']['native'],
               'vo_only_gps_residual_m': {k: ps['vo_only'][k] for k in ('rms_m', 'median_m', 'max_m', 's', 'lag_s')},
               'gps_aided_residual_m': {k: ps['fused'][k] for k in ('rms_m', 'median_m', 'max_m')},
               'camera_height_m': ps['camera_height_m'],
               'map': {k: rep['map'].get(k) for k in ('on_map', 'distance_from_centre_m', 'bearing_from_centre_deg',
                                                       'box_distance_m', 'street', 'note')}}
    lost = ps['vo_only'].get('lost_at_frame')
    if lost is not None:                     # the odometry stopped early: the segment is shorter than asked
        at = round(float(times[lost]) + start, 2) if 0 <= lost < len(times) else None
        summary['lost'] = {'at_frame': int(lost), 'at_video_s': at, 'frames_decoded': len(times),
                           'note': 'the odometry lost its track (too few points it could place: rain, a wiper '
                                   'it could not bridge, a close vehicle filling the view, a stop); only the frames before it are posed and '
                                   'reconstructed' + (f' — the rest can be its own segment (--start {at:g})'
                                                      if at is not None else '')}
        em.progress(0.35, f'the odometry lost its track at frame {lost} of {len(times)}: only the frames before it are '
                          f'used', force=True)
    rep['summary'] = summary
    PV.write_json(rp, rep)
    # 8. training and novel views
    if s['train']:
        from .train import train
        em.progress(0.35, 'training the reconstruction', force=True)
        t1 = time.perf_counter()
        res = train(('posed', posed), out_dir, {'iterations': s['iterations'], 'width': s['width'],
                                                'seed': s['seed'], 'device': s['device'], 'semantic': False},
                    em, cancel, argv=['recon', 'train', '--source', f'posed:{posed}'])
        rep['recon'] = res
        rep['heldout_by_horizon'] = heldout_by_horizon(out_dir, s['device'])
        em.progress(0.95, 'rendering novel views', force=True)
        rep['novel_views'] = novel_views(out_dir, s, s['device'])
        timings['recon_s'] = round(time.perf_counter() - t1, 2)
        summary['recon'] = {'heldout': res.get('heldout'), 'heldout_by_horizon': rep['heldout_by_horizon'],
                            'gaussians': res.get('gaussians')}
        summary['novel_views'] = rep['novel_views']['files'][:3]
    from ..provenance import read_json
    rj = read_json(os.path.join(out_dir, 'run.json')) if os.path.isfile(os.path.join(out_dir, 'run.json')) else run
    rj['dashcam'] = {'report': rp, 'summary': summary, 'timings': timings}
    if not s['train']:
        rj.update(status='ok', exit_code=0, finished=PV.utc_now())
    PV.write_json(os.path.join(out_dir, 'run.json'), rj)
    rep['summary'] = summary
    PV.write_json(rp, rep)
    return summary


def heldout_by_horizon(rec_dir: str, device: str = 'auto') -> dict:
    """The held-out frames' PSNR / SSIM split at the horizon: over the pixels whose ray points below the horizontal
    (road, verges, traffic, roadside) and over those above it (mostly sky). A dashcam's sky is a large, smooth part of
    every frame that a reconstruction reproduces easily, so the whole-image number flatters the scene below it."""
    import torch
    from . import data as D
    from . import evaluate as EV
    from .scene import Recon
    rec = Recon.load(rec_dir, device)
    cap = rec.capture()
    by = {f.index: f for f in cap.frames}
    wh = tuple(rec.work_size)
    rows = {'below': [], 'above': []}
    share = []
    for fd in rec.split('test'):
        f = by.get(fd['index'])
        if f is None:
            continue
        K = D.scaled_K(f.K, f.size, wh)
        out = EV._render(rec.gs, EV._t(f.viewmat, rec.device), K, wh)
        img = out['image'].clamp(0, 1)
        gt = EV._t(D.resize_rgb(D.load_rgb(f.image), wh), rec.device) / 255.0
        u, v = np.meshgrid(np.arange(wh[0]) + 0.5, np.arange(wh[1]) + 0.5)
        d = np.stack([(u - K[2]) / K[0], (v - K[3]) / K[1], np.ones_like(u)], -1) @ f.viewmat[:3, :3]   # world dirs
        below = d[..., 2] < 0
        m = D.resize_mask(D.load_mask(f), wh)
        share.append(float(below.mean()))
        for key, part in (('below', below), ('above', ~below)):
            mm = part if m is None else part & m
            if mm.mean() > 0.02:
                rows[key].append(EV.compare(img, gt, torch.as_tensor(mm, device=rec.device)))
    out = {'what': 'held-out frames split at the horizon (each pixel\'s ray below / above the horizontal)',
           'below_share_of_pixels': round(float(np.mean(share)), 3) if share else None}
    for key, r in rows.items():
        if r:
            out[key] = {'frames': len(r), 'psnr': round(float(np.mean([x['psnr'] for x in r])), 3),
                        'ssim': round(float(np.mean([x['ssim'] for x in r])), 4)}
    return out


def novel_views(rec_dir: str, s: dict, device: str = 'auto') -> dict:
    """A few frames (spread over the segment, held-out ones first) rendered at their own pose, shifted sideways
    (novel_lateral_m) and with the camera raised / lowered (novel_up_m), saved as one strip per frame and as single
    images in renders/."""
    import cv2
    from .scene import Recon
    rec = Recon.load(rec_dir, device)
    test = [k for k, f in enumerate(rec.frames) if f['split'] == 'test']
    pick = [test[int(round(j))] for j in np.linspace(0, len(test) - 1, min(int(s['novel_frames']), len(test)))] \
        if test else list(range(0, len(rec.frames), max(1, len(rec.frames) // 3)))[:3]
    d = os.path.join(rec_dir, 'renders')
    os.makedirs(d, exist_ok=True)
    files, t0 = [], time.perf_counter()
    for k in pick:
        f = rec.frame(k)
        row = []
        for lab, kw in ([('own pose', {})] + [(f'{v:+.1f} m right', {'lateral_m': v}) for v in s['novel_lateral_m']]
                        + [(f'{v:+.1f} m up', {'up_m': v}) for v in s['novel_up_m']]):
            out = rec.render(rec.camera(f, **kw), semantic=False)
            img = cv2.cvtColor(out['image'], cv2.COLOR_RGB2BGR)
            tag = lab.replace(' ', '').replace('+', 'p').replace('-', 'm').replace('.', '_')
            p = os.path.join(d, f'novel-{f["index"]:06d}-{tag}.png')
            cv2.imwrite(p, img)
            files.append(p)
            cv2.putText(img, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            row.append(img)
        cv2.imwrite(os.path.join(d, f'novel-{f["index"]:06d}-strip.jpg'), np.vstack(row))
    return {'frames': [rec.frame(k)['index'] for k in pick], 'files': files,
            'seconds': round(time.perf_counter() - t0, 2),
            'note': 'no reference exists at these poses (the dashcam drove one path): judge them by eye; see the '
                    'held-out numbers for poses on the path'}
