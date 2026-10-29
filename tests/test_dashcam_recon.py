"""Reconstruction from a dashcam (core.recon.vo, core.recon.fromdashcam): monocular visual odometry, bundle adjustment,
intrinsics self-calibration, the GPS alignment, the posed-image source and placing a segment on an imported map —
on synthetic streets rendered here along known trajectories (a textured ground, walls and a far wall, ray-cast with a
known pinhole and radial distortion). No real footage is read or written. CPU only."""
import json
import math
import os

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from falcon_autonomy.core.recon import vo as V  # noqa: E402


# ================================================================================================ a synthetic street
def texture(seed=0, n=1024):
    """A seamless random texture (blurred noise of several scales): plenty of corners for KLT."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((n, n), np.float32)
    for cell, amp in ((64, 0.5), (16, 0.35), (4, 0.25)):
        g = rng.random((n // cell, n // cell)).astype(np.float32)
        acc += amp * cv2.resize(np.tile(g, (3, 3)), (3 * n, 3 * n), interpolation=cv2.INTER_CUBIC)[n:2 * n, n:2 * n]
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return acc


def bilinear(tex, x, y):
    """Bilinear samples of a square texture at (x, y) texels, wrapped."""
    n = tex.shape[0]
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    x0, y0 = x0.astype(np.int64) % n, y0.astype(np.int64) % n
    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
    return (tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy) + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy


class Street:
    """A street along +x in a right-handed world (z up): ground z = 0, walls at y = ±half_width up to 8 m, a far
    wall at x = end. Rendered by ray casting through a distorted pinhole (Intrinsics)."""

    def __init__(self, seed=0, half_width=7.0, end=400.0, texels_per_m=24.0):
        self.tex = [texture(seed + k) for k in range(3)]
        self.hw, self.end, self.tpm = half_width, end, texels_per_m
        self.rays = {}

    def render(self, intr: V.Intrinsics, R_wc, c, supersample=2):
        W, H = intr.width * supersample, intr.height * supersample
        big = intr.scaled(W, H)
        key = (W, H, big.f, big.k1, big.k2)
        if key not in self.rays:                   # every pixel's undistorted ray, once per camera
            u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
            self.rays[key] = big.normalise(np.c_[u.ravel(), v.ravel()])
        xn = self.rays[key]
        d = np.c_[xn, np.ones(len(xn))] @ np.asarray(R_wc).T
        t_best = np.full(len(d), np.inf)
        val = np.full(len(d), 0.85, np.float32)
        planes = []
        with np.errstate(divide='ignore', invalid='ignore'):
            planes.append((np.where(d[:, 2] < -1e-9, -c[2] / d[:, 2], np.inf), 0, (0, 1)))
            for side in (-self.hw, self.hw):
                planes.append((np.where(np.abs(d[:, 1]) > 1e-9, (side - c[1]) / d[:, 1], np.inf), 1, (0, 2)))
            planes.append((np.where(d[:, 0] > 1e-9, (self.end - c[0]) / d[:, 0], np.inf), 2, (1, 2)))
        for t, k, axes in planes:
            p = c + d * np.where(np.isfinite(t) & (t > 0), t, 0)[:, None]
            ok = np.isfinite(t) & (t > 0) & (t < t_best)
            if k == 1:
                ok &= (p[:, 2] >= 0) & (p[:, 2] <= 8.0)
            if not ok.any():
                continue
            n = self.tex[k].shape[0]
            val[ok] = bilinear(self.tex[k], p[ok, axes[0]] * self.tpm, p[ok, axes[1]] * self.tpm)
            t_best[ok] = t[ok]
        img = (val.reshape(H, W) * 220 + 20).astype(np.float32)
        img = cv2.resize(img, (intr.width, intr.height), interpolation=cv2.INTER_AREA)
        return np.clip(img, 0, 255).astype(np.uint8)


def cam_rotation(yaw, pitch=0.0):
    """camera -> world: the optical axis at heading yaw (rad, from +x towards +y), pitched down by pitch (rad)."""
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.cross(f, left)
    return np.stack([-left, -up, f], 1)          # columns: optical x (right), y (down), z (forward)


def trajectory(n=60, fps=10.0, speed=6.0, turn_deg=0.0, height=1.4, pitch_deg=3.0, weave=0.0):
    """Camera centres and camera -> world rotations along the street: forward at `speed` m/s, the heading turning
    by turn_deg in total (a gentle S-weave of amplitude `weave` degrees on top)."""
    ts = np.arange(n) / fps
    yaw = np.radians(turn_deg) * ts / ts[-1] + np.radians(weave) * np.sin(2 * math.pi * ts / ts[-1])
    pos = np.zeros((n, 3))
    for k in range(1, n):
        pos[k] = pos[k - 1] + speed / fps * np.array([math.cos(yaw[k]), math.sin(yaw[k]), 0.0])
    pos[:, 2] = height
    pos[:, 1] += 0.0
    return ts, pos, [cam_rotation(y, math.radians(pitch_deg)) for y in yaw]


@pytest.fixture(scope='module')
def street_seq():
    intr = V.Intrinsics.from_hfov(480, 270, 100.0, k1=-0.08)
    st = Street(seed=3)
    ts, pos, rots = trajectory(n=50, speed=5.0, weave=6.0)
    imgs = [st.render(intr, R, c) for R, c in zip(rots, pos)]
    return {'intr': intr, 'ts': ts, 'pos': pos, 'rots': rots, 'imgs': imgs, 'street': st}


def vo_centres(vo, frames):
    return np.array([V.centre_of(vo.poses[f]) for f in frames])


# ================================================================================================ geometry
def test_rodrigues_matches_opencv():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, (20, 3))
    r[0] = 0.0
    r[1] = 1e-10
    R = V.rodrigues(r)
    for k in range(20):
        assert np.allclose(R[k], cv2.Rodrigues(r[k])[0], atol=1e-12)


def test_normalise_inverts_projection():
    intr = V.Intrinsics(640, 360, 300.0, -0.2, 0.05)
    rng = np.random.default_rng(1)
    xy = rng.uniform(-0.8, 0.8, (200, 2)) * [1, 0.5]
    uv = intr.project_normalised(xy)
    assert np.allclose(intr.normalise(uv), xy, atol=1e-8)
    s = intr.scaled(1280, 720)
    assert s.f == pytest.approx(600.0) and s.cx == pytest.approx((319.5 + 0.5) * 2 - 0.5)


def test_umeyama_and_robust_similarity():
    rng = np.random.default_rng(2)
    src = rng.normal(0, 10, (40, 2))
    a = math.radians(37.0)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    dst = 3.7 * src @ R.T + [100.0, -50.0]
    s, R2, t = V.umeyama(src, dst)
    assert s == pytest.approx(3.7) and np.allclose(R2, R) and np.allclose(t, [100, -50])
    noisy = dst + rng.normal(0, 2.0, dst.shape)
    noisy[:5] += 60.0                              # five gross outliers (GPS multipath)
    r = V.robust_similarity(src, noisy, c=2.0)
    assert r['s'] == pytest.approx(3.7, rel=0.02)
    assert abs(math.degrees(math.atan2(r['R'][1, 0], r['R'][0, 0])) - 37.0) < 1.0
    assert (r['weights'][:5] < 0.2).all()


def test_level_rotation():
    rng = np.random.default_rng(3)
    for _ in range(5):
        u = rng.normal(0, 1, 3)
        u /= np.linalg.norm(u)
        R = V.level_rotation(u)
        assert np.allclose(R @ u, [0, 0, 1], atol=1e-12) and np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(V.level_rotation([0, 0, -1.0]) @ [0, 0, -1.0], [0, 0, 1])


# ================================================================================================ bundle adjustment
def test_bundle_adjustment_reduces_reprojection_error_and_recovers_focal():
    rng = np.random.default_rng(4)
    intr_true = V.Intrinsics(640, 360, 320.0, -0.1)
    X = np.c_[rng.uniform(-8, 8, 300), rng.uniform(-3, 2, 300), rng.uniform(6, 40, 300)]
    poses = []
    for k in range(8):                              # forward motion with some yaw and pitch: f is observable
        R = V.rodrigues(np.array([[0.03 * math.sin(k), 0.06 * k - 0.2, 0.01 * k]]))[0]
        c = np.array([0.4 * math.sin(k), 0.1 * k, 1.2 * k])
        poses.append(V.pose_vec(R, -R @ c))
    poses = np.array(poses)
    cam_i, pt_i, uv = [], [], []
    for k, p in enumerate(poses):
        T = V.pose_mat(p)
        pc = X @ T[:3, :3].T + T[:3, 3]
        q = intr_true.project_normalised(pc[:, :2] / pc[:, 2:3])
        ok = (pc[:, 2] > 1) & (q[:, 0] > 0) & (q[:, 0] < 640) & (q[:, 1] > 0) & (q[:, 1] < 360)
        cam_i += [k] * int(ok.sum())
        pt_i += list(np.nonzero(ok)[0])
        uv.append(q[ok] + rng.normal(0, 0.3, (int(ok.sum()), 2)))
    uv = np.concatenate(uv)
    p0 = poses.copy()
    p0[2:] += rng.normal(0, 0.01, p0[2:].shape)
    X0 = X + rng.normal(0, 0.1, X.shape)
    start = V.Intrinsics(640, 360, 290.0, 0.0)
    r = V.bundle_adjust(p0, X0, cam_i, pt_i, uv, start, fixed=(0, 1), free_intr=('f', 'k1'), max_iterations=100)
    assert r['rms_after'] < 0.1 * r['rms_before']
    assert r['median_after'] < 0.6                       # ~ the 0.3 px noise
    assert r['intr'].f == pytest.approx(320.0, rel=0.01)
    assert r['intr'].k1 == pytest.approx(-0.1, abs=0.01)
    assert 0 < r['intr_std']['f'] < 3.0                  # a finite, small standard deviation


# ================================================================================================ odometry
def test_vo_recovers_the_trajectory_up_to_scale(street_seq):
    s = street_seq
    intr = s['intr']
    tr = V.track(s['imgs'], None, intr)
    assert tr.stats['mean_tracked'] > 200
    vo = V.run_vo(tr, intr)
    assert vo.lost_at is None and len(vo.poses) >= len(s['imgs']) - 1
    V.global_ba(vo, tr, intr)
    fr = sorted(vo.poses)
    est = vo_centres(vo, fr)
    true = s['pos'][fr]
    sim = V.umeyama(est, true)
    err = np.linalg.norm(sim[0] * est @ sim[1].T + sim[2] - true, axis=1)
    path = np.sum(np.linalg.norm(np.diff(true, axis=0), axis=1))
    assert err.max() < 0.02 * path, (err.max(), path)     # within 2 % of the 24.5 m path
    # orientation (gauge-free): every frame's rotation relative to the first agrees with the true one
    R0 = V.rodrigues(vo.poses[fr[0]][:3])[0]
    for f in fr[1:]:
        rel_est = V.rodrigues(vo.poses[f][:3])[0] @ R0.T            # camera f <- camera 0
        rel_true = s['rots'][f].T @ s['rots'][fr[0]]
        ang = math.degrees(math.acos(np.clip((np.trace(rel_est.T @ rel_true) - 1) / 2, -1, 1)))
        assert ang < 1.5, (f, ang)               # measured: a pitch drift of up to 0.9° over 50 frames
    assert V.reprojection_rms(vo, tr, intr) < 1.0


def test_self_calibration_recovers_focal_and_distortion(street_seq):
    """With rotation in the sequence (a gentle weave) f and k1 are observable: from a start 15 % off and no
    distortion, VO + a global BA with f and k1 free, repeated, lands within 5 % of f and 0.03 of k1."""
    s = street_seq
    true = s['intr']
    start = V.Intrinsics(true.width, true.height, true.f * 0.85, 0.0)
    tr = V.track(s['imgs'], None, start)
    intr, hist = V.self_calibrate(tr, start, ('f', 'k1'), rounds=4)
    assert intr.f == pytest.approx(true.f, rel=0.05), hist
    assert intr.k1 == pytest.approx(true.k1, abs=0.03)
    assert hist[-1]['rms_px'] < 1.5 and hist[0]['start_f'] == pytest.approx(0.85 * true.f, rel=1e-3)


def test_speed_priors_make_the_map_metric(street_seq):
    """Given the distance travelled between frames (as the GPS speed gives it), the odometry is metric from the
    start: the similarity to the true track has a scale within 3 % of 1, without using any position."""
    s = street_seq
    speed = np.r_[np.linalg.norm(np.diff(s['pos'], axis=0), axis=1) * 10.0, [5.0]]
    dist = V.speed_distance(s['ts'], speed)
    d, sig = dist(0.0, 1.0)
    assert d == pytest.approx(np.sum(np.linalg.norm(np.diff(s['pos'][:11], axis=0), axis=1)), rel=0.02)
    assert sig == pytest.approx(0.03 * d + 0.05)
    tr = V.track(s['imgs'], None, s['intr'])
    vo = V.run_vo(tr, s['intr'], times=s['ts'], distance=dist)
    V.global_ba(vo, tr, s['intr'])
    fr = sorted(vo.poses)
    sc, R, t = V.umeyama(vo.centres(fr), s['pos'][fr])
    assert sc == pytest.approx(1.0, abs=0.03)


# ================================================================================================ GPS alignment
def test_gps_alignment_recovers_scale_heading_and_lag():
    """A known track seen by the odometry in its own frame (scaled by 0.37, rotated, tilted), against 1 Hz GPS
    fixes with 2 m noise and one 40 m outlier: the alignment finds the scale within 3 % and the heading within 2°, and
    the outlier is the largest residual. The latency (0.8 s here) is only weakly determined — at constant speed it
    mostly trades against an offset along the track, and on a curve a wrong one turns the heading — so it is checked
    with 0.5 m noise, to 0.3 s."""
    rng = np.random.default_rng(5)
    ts = np.arange(0, 20.0, 1 / 30)
    yaw = np.radians(70.0) * (ts / ts[-1]) ** 2             # a tightening left curve, speeding up from 6 to 16 m/s
    speed = 6.0 + 10.0 * ts / ts[-1]                        # (an arc at constant speed would hide the latency:
    pos = np.zeros((len(ts), 3))                            # a time shift there is a rotation about its centre)
    for k in range(1, len(ts)):
        pos[k] = pos[k - 1] + speed[k] / 30 * np.array([math.cos(yaw[k]), math.sin(yaw[k]), 0.0])
    tilt = V.rodrigues(np.array([[0.3, -0.2, 1.1]]))[0]   # the odometry's frame: any rotation, any scale
    vo_c = (pos - pos[0]) @ tilt.T / 0.37 + [5.0, -2.0, 1.0]
    up_vo = tilt @ np.array([0.0, 0.0, 1.0])
    gt = np.arange(0.0, 20.0, 1.0)
    gps = np.c_[np.interp(gt, ts, pos[:, 0]), np.interp(gt, ts, pos[:, 1])] + [300.0, -120.0]
    gps += rng.normal(0, 2.0, gps.shape)
    gps[7] += [40.0, 0.0]
    al = V.align_to_gps(vo_c, ts, gt, gps, up_vo, lags=[0.0])
    assert al['s'] == pytest.approx(0.37, rel=0.03)
    err = al['R'] @ tilt                                    # world <- odometry against the truth (tiltᵀ)
    assert math.degrees(math.acos(np.clip((np.trace(err) - 1) / 2, -1, 1))) < 2.0
    assert int(np.argmax(al['residuals_m'])) == 7 and al['median_m'] < 3.0
    assert np.allclose(al['R'] @ up_vo, [0, 0, 1], atol=1e-9)
    true_xy = np.c_[np.interp(gt - 0.8, ts, pos[:, 0]), np.interp(gt - 0.8, ts, pos[:, 1])] + [300.0, -120.0]
    quiet = true_xy + rng.normal(0, 0.5, true_xy.shape)
    al2 = V.align_to_gps(vo_c, ts, gt, quiet, up_vo, lags=np.arange(-1.5, 1.51, 0.1), lag_gain=1.0, lag_min_gain_m=0)
    assert al2['lag_s'] == pytest.approx(-0.8, abs=0.3)      # fix time + lag = the odometry's time
    prof = dict((round(a, 1), m) for a, m in al2['lag_profile'])
    al3 = V.align_to_gps(vo_c, ts, gt, quiet, up_vo, lags=np.arange(-1.5, 1.51, 0.1))
    assert al3['lag_s'] == pytest.approx(al2['lag_s'] if prof[0.0] - prof[round(al2['lag_s'], 1)] >= 0.5 else 0.0)


# ================================================================================================ the camera model
def grid_image(intr: V.Intrinsics, spacing=40, vertical=False):
    """An image of separate straight lines (horizontal, or vertical) in the undistorted image, seen through intr's
    distortion."""
    W, H = intr.width, intr.height
    big = np.full((H * 2, W * 2), 40, np.uint8)
    for c in range(spacing // 2, (W if vertical else H) * 2, spacing):
        if vertical:
            cv2.line(big, (c, 0), (c, H * 2 - 1), 220, 5)
        else:
            cv2.line(big, (0, c), (W * 2 - 1, c), 220, 5)
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    xn = intr.normalise(np.c_[u.ravel(), v.ravel()])        # every distorted pixel's ray, in the 2× pinhole
    mx = (xn[:, 0] * intr.f + W).reshape(H, W).astype(np.float32)
    my = (xn[:, 1] * intr.f + H).reshape(H, W).astype(np.float32)
    return cv2.remap(big, mx, my, cv2.INTER_LINEAR, borderValue=40)


def test_plumb_line_recovers_k1():
    intr = V.Intrinsics.from_hfov(480, 270, 100.0, k1=-0.05)    # (−0.12 would fold this wide image back)
    imgs = [grid_image(intr), grid_image(intr, vertical=True)]
    est = V.plumb_line(imgs, None, V.Intrinsics(480, 270, intr.f), every=1)
    assert est['k1'] == pytest.approx(-0.05, abs=0.015), est
    assert est['used'] >= 10
    pin = V.Intrinsics(480, 270, intr.f)
    flat = V.plumb_line([grid_image(pin), grid_image(pin, vertical=True)], None, pin, every=1)
    assert abs(flat['k1']) <= 0.02


def test_undistorted_crop_is_valid_and_straightens_lines():
    from falcon_autonomy.core.recon import fromdashcam as F
    intr = V.Intrinsics.from_hfov(480, 270, 100.0, k1=-0.06)
    img = grid_image(intr, vertical=True)
    hood = np.full(480, 250)
    pin = F.pinhole_crop(intr, hood, 400, 4)
    W, H = pin['size']
    maps = F.undistort_maps(intr, pin)
    assert maps[0].min() >= -0.5 and maps[0].max() <= 479.5 and maps[1].min() >= -0.5
    assert maps[1].max() <= 246.5                           # above the bonnet (row 250 − 4 px margin)
    und = F.undistort(img, maps, pin['size'])
    assert und.shape == (H, W)
    # a vertical line near the left edge is vertical again: followed row by row, its centre barely moves
    def follow(im, x0):
        xs = []
        for r in im.astype(float):
            lo = max(0, x0 - 6)
            x0 = lo + int(np.argmax(np.convolve(r[lo:x0 + 7], np.ones(3), 'same')))
            xs.append(x0)
        return np.array(xs)
    first = lambda r: int(np.nonzero(r > 150)[0][0]) + 2      # noqa: E731
    xs = follow(und[3:H - 3], first(und[3]))
    xr = follow(img[3:240], first(img[3]))
    assert np.ptp(xs) <= 2 < np.ptp(xr)                    # straight after, bent before


def test_static_mask_finds_a_still_band(street_seq):
    from falcon_autonomy.core.recon import fromdashcam as F
    imgs = [im.copy() for im in street_seq['imgs']]
    band = (texture(9, 512)[:30, :480] * 120 + 20).astype(np.uint8)
    for im in imgs:
        im[-30:, :] = band                                 # a (textured) bonnet that never moves
        cv2.putText(im, '2026/03/21 FAKE', (8, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 230, 1)
    st = F.static_mask(imgs, street_seq['ts'], 3.0, corridor=(0.4, 0.6, 0.6, 1.0))
    assert (st['hood_rows'] <= 240 + 6).all() and (st['hood_rows'] >= 230).all()
    assert st['mask'][100, 240] and not st['mask'][265, 240]
    assert not st['odometry'][200, 240] and st['mask'][200, 240]   # the corridor: the odometry's only


# ================================================================================================ end to end
LAT0, LON0 = 12.9716, 77.5946


def make_dashcam_clip(folder, n=80, fps=10, speed=6.0, weave=4.0, north=-1.75, intr=None):
    """A synthetic dashcam clip: the street rendered along a known track (east along y = `north` through the
    imported test world's centre), encoded with ffmpeg and given a GPS index of 1 Hz fixes from the true track in
    core.dashcam's layout. Returns (path, the true camera centres in local east-north-up, intrinsics)."""
    import datetime as dt
    import subprocess
    from falcon_autonomy.core import dashcam as DC
    from falcon_autonomy.core.osm import local_projection
    intr = intr or V.Intrinsics.from_hfov(480, 270, 100.0, k1=-0.04)
    st = Street(seed=11, half_width=9.0)
    ts, pos, rots = trajectory(n=n, fps=fps, speed=speed, weave=weave)
    pos[:, 0] -= 20.0
    pos[:, 1] += north
    frames = [st.render(intr, R, c) for R, c in zip(rots, pos)]
    os.makedirs(folder, exist_ok=True)
    utc0 = dt.datetime(2026, 3, 21, 6, 30, 0, tzinfo=dt.timezone.utc)
    path = os.path.join(folder, (utc0 + dt.timedelta(hours=5, minutes=30)).strftime('%Y%m%d%H%M%S') + '_000001F.MP4')
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                          '-s', f'{intr.width}x{intr.height}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-g', '5',
                          '-bf', '0', '-crf', '12', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', path],
                         stdin=subprocess.PIPE)
    p.communicate(b''.join(f.tobytes() for f in frames))
    assert p.returncode == 0
    proj = local_projection(LAT0, LON0)
    blocks = []
    for k in range(int(n / fps)):
        j = min(int(round(k * fps)), n - 1)
        d = pos[min(j + 1, n - 1)] - pos[max(j - 1, 0)]
        course = (90.0 - math.degrees(math.atan2(d[1], d[0]))) % 360
        lat, lon = proj.to_geo(pos[j, 0], pos[j, 1])
        blocks.append(DC.encode_block(utc0 + dt.timedelta(seconds=k), lat, lon, speed / 0.514444, course))
    DC.add_gps(path, blocks)
    return path, pos, intr


@pytest.fixture(scope='module')
def dash_clip(tmp_path_factory):
    import shutil
    if not shutil.which('ffmpeg'):
        pytest.skip('needs ffmpeg')
    return make_dashcam_clip(str(tmp_path_factory.mktemp('dashclip') / 'Movie_F'))


@pytest.fixture(scope='module')
def test_world(tmp_path_factory):
    """A project with an imported world: a straight two-lane road along +x (east) through (LAT0, LON0)."""
    from falcon_autonomy.core.osm import bbox_around
    from falcon_autonomy.core.project import Project
    d = tmp_path_factory.mktemp('dashproj')
    p = Project.new('dash')
    p.save(str(d / 'dash.falcon.json'))
    wdir = os.path.join(p.workspace, 'worlds', 'w-test')
    os.makedirs(wdir, exist_ok=True)
    proj = f'+proj=tmerc +lat_0={LAT0} +lon_0={LON0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    with open(os.path.join(wdir, 'world.xodr'), 'w', encoding='utf-8') as fh:
        fh.write(f"""<?xml version="1.0"?>
<OpenDRIVE><header revMajor="1" revMinor="4"><geoReference><![CDATA[{proj}]]></geoReference></header>
<road id="1" length="400" junction="-1"><planView><geometry s="0" x="-200" y="0" hdg="0" length="400"><line/></geometry>
</planView><lanes><laneSection s="0"><left><lane id="1" type="driving"><width sOffset="0" a="3.5" b="0" c="0" d="0"/>
</lane></left><center><lane id="0" type="none"/></center><right><lane id="-1" type="driving">
<width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road></OpenDRIVE>""")
    with open(os.path.join(wdir, 'graph.json'), 'w', encoding='utf-8') as fh:
        json.dump({'roads': [{'id': '1', 'x': [-200.0, 200.0], 'y': [0.0, 0.0], 'name': 'Test Road', 'left': 1,
                              'right': 1}]}, fh)
    p.doc['worlds'].append({'id': 'w-test', 'name': 'Test centre', 'kind': 'osm', 'xodr': 'worlds/w-test/world.xodr',
                            'osm': {'bbox': list(bbox_around(LAT0, LON0, 1000.0)), 'centre': [LAT0, LON0],
                                    'handedness': 'RHT'}})
    p.save(str(d / 'dash.falcon.json'))
    return str(d / 'dash.falcon.json')


def test_from_dashcam_end_to_end(dash_clip, test_world, tmp_path):
    """The synthetic clip through the whole pipeline on the CPU: decoded through core.dashcam, posed (the given
    intrinsics), aligned to its GPS, reconstructed (a few steps), rendered from novel poses and placed on the test
    world's road (right-hand lane, 1.75 m south of the reference line)."""
    from falcon_autonomy.core.recon import data as D
    from falcon_autonomy.core.recon import fromdashcam as F
    from falcon_autonomy.core.recon.scene import Recon
    clip, pos, intr = dash_clip
    nat = intr.scaled(*F.NATIVE)
    out = str(tmp_path / 'rec')
    res = F.from_dashcam(clip, out, 0.0, 7.5, {
        'image_size': [480, 270], 'vo_width': 480, 'out_width': 320, 'image_every': 2,
        'intrinsics': f'f={nat.f},k1={nat.k1}', 'iterations': 30, 'width': 96, 'device': 'cpu', 'seed': 0,
        'project': test_world, 'world': 'w-test', 'novel_frames': 1, 'novel_lateral_m': [1.0], 'novel_up_m': [0.5]})
    rep = json.load(open(os.path.join(out, 'dashcam.json'), encoding='utf-8'))
    assert rep['frames']['decoded'] == 75 and res['images'] >= 30
    assert rep['alignment_vo_only']['median_m'] < 1.0, rep['alignment_vo_only']   # exact fixes, float32 coordinates
    assert rep['alignment_vo_only']['s'] == pytest.approx(1.0, abs=0.05)          # the speed priors: metric already
    assert rep['alignment_gps_aided']['median_m'] < 1.0
    assert 1.0 < rep['ground']['camera_height_m'] < 1.8                            # the street's camera: 1.4 m
    m = rep['map']
    assert m['on_map'] and m['street'] == 'Test Road' and m['inside_box_share'] == 1.0
    assert m['match']['share_on_lane'] > 0.8
    cap = D.open_capture(f'posed:{out}/posed')
    assert cap.kind == 'posed' and cap.points is not None and len(cap.frames) == res['images']
    # the posed cameras sit where the true ones were (to the GPS's sub-metre exactness here)
    true = pos[[int(f.index) for f in cap.frames]]
    est = np.array([f.centre for f in cap.frames])
    from falcon_autonomy.core.osm import local_projection
    lat0, lon0 = rep['geo']['lat0'], rep['geo']['lon0']
    e0, n0 = local_projection(LAT0, LON0).to_local(lat0, lon0)
    assert np.median(np.linalg.norm(est[:, :2] + [e0, n0] - true[:, :2], axis=1)) < 1.0
    rec = Recon.load(out, 'cpu')
    assert rec.views['source']['kind'] == 'posed' and rec.run['summary']['init'] == 'points'
    assert os.path.isfile(os.path.join(out, 'evaluation.json')) and os.path.isfile(os.path.join(out, 'intrinsics.json'))
    files = rep['novel_views']['files']
    assert len(files) == 3 and all(os.path.isfile(f) for f in files)
    assert F.parse_intrinsics(os.path.join(out, 'intrinsics.json')).f == pytest.approx(nat.f, rel=1e-6)


def test_cli_from_dashcam_no_train(dash_clip, tmp_path, capsys):
    from falcon_autonomy.core.recon.commands import main_recon
    clip, _, intr = dash_clip
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps({'image_size': [480, 270], 'vo_width': 480, 'out_width': 320, 'hfov_grid': [90, 100]}))
    out = str(tmp_path / 'cli')
    rc = main_recon(['from-dashcam', '--clip', clip, '--start', '0', '--seconds', '6', '--out', out, '--config',
                     str(cfg), '--no-train', '--image-every', '3'])
    assert rc == 0, capsys.readouterr().out[-2000:]
    rep = json.load(open(os.path.join(out, 'dashcam.json'), encoding='utf-8'))
    fs = rep['calibration']['focal_search']                  # no intrinsics given: the plumb line and the search
    assert [r['hfov_deg'] for r in fs['profile']] == [90.0, 100.0] and fs['best_hfov_deg'] in (90.0, 100.0)
    assert fs['scored_by'] == 'median_m'                     # a straight drive: no turn to score the yaw by
    assert not os.path.isfile(os.path.join(out, 'point_cloud.ply'))
    assert main_recon(['from-dashcam', '--clip', str(tmp_path / 'nope.mp4'), '--out', out]) == 2


# ================================================================================================ the Replay page
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


@pytest.mark.gui
def test_replay_page_from_dashcam(window, qapp, dash_clip, tmp_path):
    """The reconstruction section's *From a dashcam*: the job it queues (a GPU job of `falcon recon from-dashcam`
    with the clip, the window and the panel's settings), and once a reconstruction is in its folder (made here
    in-process with small settings, as the job would) the view opens it and shows the poses' numbers."""
    from falcon_autonomy.core.jobs import DONE
    from falcon_autonomy.core.recon import fromdashcam as F
    from falcon_autonomy.ui.pages.replay import ReplayPage
    page = ReplayPage(window)
    window.stack.addWidget(page)
    window.stack.setCurrentWidget(page)
    panel = page.recon
    clip, _, intr = dash_clip
    assert panel.from_dashcam() is None and 'choose a dashcam clip' in panel.note.text()
    panel.dash_clip.setText(clip)
    panel.dash_start.setValue(0.0)
    panel.dash_seconds.setValue(6.0)
    panel.iters.setValue(20)
    panel.width_px.setValue(64)
    panel.device.setCurrentIndex(panel.device.findData('cpu'))
    out = os.path.join(window.workspace(), 'recon', f'dashcam-{os.path.splitext(os.path.basename(clip))[0]}-0s')
    argv = panel.dashcam_argv(out)
    for k, v in (('--clip', clip), ('--start', '0'), ('--seconds', '6'), ('--out', out), ('--iterations', '20'),
                 ('--width', '64'), ('--device', 'cpu')):
        assert argv[argv.index(k) + 1] == v
    assert argv[0] == 'from-dashcam' and '--project' not in argv          # an unsaved project: nothing to place on
    from falcon_autonomy.core.recon.train import job_spec
    spec = job_spec(argv, 'dashcam')
    assert spec.needs_gpu and spec.argv[-len(argv):] == argv
    nat = intr.scaled(*F.NATIVE)
    F.from_dashcam(clip, out, 0.0, 6.0, {'image_size': [480, 270], 'vo_width': 480, 'out_width': 320,
                                          'intrinsics': f'f={nat.f},k1={nat.k1}', 'iterations': 20, 'width': 64,
                                          'device': 'cpu', 'novel_frames': 1, 'novel_lateral_m': [1.0],
                                          'novel_up_m': []})

    class Finished:
        state = DONE

        def summary(self):
            return ''
    job = Finished()
    panel.dash_dir = out
    panel.jobs['dashcam'] = job
    panel.on_finished(job)
    assert pump(qapp, lambda: panel.last is not None, 60)
    assert panel.rec_dir == out and 'odometry against the GPS' in panel.info.vals['Poses'].text()
    assert 'renders' in panel.info.vals['Novel views'].text() and 'PSNR' in panel.info.vals['Held out'].text()
    panel.lateral.setValue(1.0)
    panel.lateral.valueChanged.emit(1.0)
    assert pump(qapp, lambda: panel.last['lateral_m'] == 1.0, 60)
