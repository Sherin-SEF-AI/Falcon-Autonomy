"""Sensor faults of known size (numpy; OpenCV inside the image functions).

A sensor's "faults" list in the rig names what goes wrong and by how much. The outputs keep stating the PLANNED
calibration (pose, intrinsics), as a real faulty rig's would; run.json "faults_applied" says what was done, and every
frame a fault touched carries it in frames.jsonl.

Kinds (FAULTS):
* mount_rotation  {deg: [roll, pitch, yaw]} — the sensor is turned on its mount (CARLA spawns it turned).
* mount_translation {m: [x, y, z]} — moved on its mount, in the vehicle frame.
* clock_offset    {ms} — data captured at t is stamped t + ms.
* drop            {rate} (0–0.9, seeded; a timed one up to 1: the sensor dies) — frames never delivered.
* freeze          {from_s, to_s} — frames in [from, to) repeat the last one before (a stuck sensor).
* noise           {sigma} — Gaussian noise of sigma 8-bit levels on the image (measured back by the tests).
* dirt            {cover} (0–0.5) — opaque dark blotches, blurred, covering that share of the image (seeded).
* rain            {cover} (0–0.5) — water drops on the lens: each drop shows a blurred, magnified patch of the image
                    behind it (refraction, heuristic), covering that share.
* glare           {strength} (0–1), {u, v} in 0–1 of the image — a veiling glow around a point (heuristic optics).
* rolling_shutter {readout_ms} — rows exposed over the readout time; rotation-only model: each row is re-projected
                    by the camera's own rotation during its offset from the middle row (translation parallax ignored).
* compression     {quality} (1–100) — JPEG at that quality.
* motion_blur     {intensity, max_distortion} — CARLA's own camera attributes (motion_blur_intensity,
                    motion_blur_max_distortion), set at spawn.
* occlusion       {cover} (0–0.9) — mud on the lens: one opaque, dark, soft-edged patch covering that share of the
                    image at a seeded place (camera).
* sector_loss     {from_deg, to_deg} — a lidar returns nothing in that azimuth sector of its own frame (degrees, CARLA's
                    x forward = 0, clockwise seen from above = y right positive; from > to wraps through 180).
* packet_drop     {rate} (0–1) — a lidar loses that share of its 1° azimuth packets, seeded per sweep.
* multipath       {jump_m, period_s} — GNSS fixes offset by jump_m metres in a seeded direction, a new direction (and a
                    new offset) every period_s seconds (multipath jumps; heuristic model).
* dropout         {} — a sensor delivers nothing (an IMU or GNSS receiver, a camera, a lidar going silent).

Where they act: the image faults in recorded drives and stack runs; sector_loss, packet_drop, multipath and dropout on
the data a stack (workers.stack) and the hardware in the loop (workers.hil) receive — not in a recorded drive's files.

Timed onset. Any fault but the spawn-time ones (mount, clock, motion blur) may carry "onset_s" and "until_s"
(simulation seconds from the run's first recorded frame): it acts only in [onset_s, until_s) — a sensor that fails
mid-drive. TIMED lists the kinds that can; a trigger rule's "fault" action (core.triggers) starts one at its firing.
frames.jsonl marks every frame a timed fault touched, as any other fault's.

Labels are of the scene, not of the faulty image: image faults change the RGB only; mount faults move every label
camera with the image (they are one rigid sensor), so labels stay pixel-aligned with the faulty image.
"""
from __future__ import annotations

import math

import numpy as np

FAULTS = ('mount_rotation', 'mount_translation', 'clock_offset', 'drop', 'freeze', 'noise', 'dirt', 'rain', 'glare',
          'rolling_shutter', 'compression', 'motion_blur', 'occlusion', 'sector_loss', 'packet_drop', 'multipath',
          'dropout')
IMAGE_FAULTS = ('noise', 'dirt', 'rain', 'glare', 'rolling_shutter', 'compression', 'occlusion')
LIDAR_ONLY = ('sector_loss', 'packet_drop')
GNSS_ONLY = ('multipath',)
STREAM_FAULTS = ('sector_loss', 'packet_drop', 'multipath', 'dropout')   # act on a stack's or a device's data only
SPAWN_TIME = ('mount_rotation', 'mount_translation', 'clock_offset', 'motion_blur')
TIMED = ('drop', 'freeze', 'noise', 'dirt', 'rain', 'glare', 'rolling_shutter', 'compression', 'occlusion',
         'sector_loss', 'packet_drop', 'multipath', 'dropout')
CAMERA_ONLY = IMAGE_FAULTS + ('motion_blur',)
CAMERAS = ('camera.pinhole', 'camera.fisheye')
LIMITS = {'deg': 30.0, 'm': 2.0, 'ms': 1000.0, 'rate': 0.9, 'cover': 0.5, 'sigma': 50.0, 'readout_ms': 100.0}


def validate(faults: list) -> list:
    out = []
    for f in faults or []:
        k = f.get('kind')
        if k not in FAULTS:
            out.append(f'unknown fault {k!r}')
            continue
        timed = 'onset_s' in f or 'until_s' in f        # a timed drop may lose every frame (the sensor dies)
        for key, lim in (('rate', 1.0 if timed or k == 'packet_drop' else LIMITS['rate']),
                         ('cover', 0.9 if k == 'occlusion' else LIMITS['cover']),
                         ('sigma', LIMITS['sigma']), ('ms', LIMITS['ms']), ('readout_ms', LIMITS['readout_ms'])):
            if key in f and not 0 <= abs(float(f[key])) <= lim:
                out.append(f'{k}: {key} {f[key]} outside 0–{lim}')
        for key, lim in (('deg', LIMITS['deg']), ('m', LIMITS['m'])):
            if key in f and (len(f[key]) != 3 or max(abs(float(v)) for v in f[key]) > lim):
                out.append(f'{k}: {key} needs three values within ±{lim}')
        if k == 'occlusion' and not 0 < float(f.get('cover', 0.3)) <= 0.9:
            out.append('occlusion: cover must be in (0, 0.9]')
        if k == 'sector_loss':
            for key in ('from_deg', 'to_deg'):
                if not isinstance(f.get(key), (int, float)) or not -360 <= f[key] <= 360:
                    out.append(f'sector_loss: {key} (degrees, −360…360) is needed')
        if k == 'multipath' and (not 0 < float(f.get('jump_m', 5.0)) <= 100 or float(f.get('period_s', 2.0)) <= 0):
            out.append('multipath: jump_m in (0, 100] m and period_s > 0')
        if k == 'compression' and not 1 <= int(f.get('quality', 0)) <= 100:
            out.append('compression: quality must be 1–100')
        if ('onset_s' in f or 'until_s' in f) and k in SPAWN_TIME:
            out.append(f'{k}: is set when the sensor is spawned; it cannot have onset_s / until_s')
        for key in ('onset_s', 'until_s'):
            if key in f and (not isinstance(f[key], (int, float)) or f[key] < 0):
                out.append(f'{k}: {key} must be a number of seconds ≥ 0')
        if isinstance(f.get('onset_s'), (int, float)) and isinstance(f.get('until_s'), (int, float)) and \
                f['until_s'] <= f['onset_s']:
            out.append(f'{k}: until_s must be after onset_s')
    return out


def applies_to(kind: str, sensor_type: str) -> bool:
    """Whether a fault kind can act on a sensor of this type (image faults: cameras only)."""
    if kind in CAMERA_ONLY:
        return sensor_type in CAMERAS
    if kind in LIDAR_ONLY:
        return sensor_type == 'lidar'
    if kind in GNSS_ONLY:
        return sensor_type == 'gnss'
    return True


def active(f: dict, sim_t: float) -> bool:
    """Whether a fault acts at sim_t: always, or within its [onset_s, until_s) when it has them."""
    if f.get('onset_s') is not None and sim_t < float(f['onset_s']) - 1e-9:
        return False
    if f.get('until_s') is not None and sim_t >= float(f['until_s']) - 1e-9:
        return False
    return True


def actual_pose(pose: dict, faults: list) -> dict:
    """The pose CARLA spawns the sensor at: the planned one with its mount faults."""
    p = dict(pose)
    for f in faults or []:
        if f.get('kind') == 'mount_rotation':
            r, pi, y = f['deg']
            p.update(roll=p['roll'] + r, pitch=p['pitch'] + pi, yaw=p['yaw'] + y)
        elif f.get('kind') == 'mount_translation':
            x, y, z = f['m']
            p.update(x=p['x'] + x, y=p['y'] + y, z=p['z'] + z)
    return p


def clock_offset_s(faults: list) -> float:
    return sum(float(f.get('ms', 0.0)) for f in faults or [] if f.get('kind') == 'clock_offset') / 1000.0


def delivery(faults: list, step: int, sim_t: float, rng) -> str:
    """'ok', 'drop' or 'freeze' for a frame (the drop draws from rng: seeded, one draw per frame)."""
    d = 'ok'
    for f in faults or []:
        if not active(f, sim_t):
            continue                          # a timed fault before its onset or after its end: no draw either
        if f.get('kind') == 'dropout':
            d = 'drop'                        # the sensor is silent (no draw: nothing random about it)
        elif f.get('kind') == 'drop' and rng.random() < float(f.get('rate', 0.0)):
            d = 'drop'
        elif f.get('kind') == 'freeze' and d == 'ok' and (
                ('from_s' not in f and 'to_s' not in f and ('onset_s' in f or 'until_s' in f))
                or float(f.get('from_s', 0)) <= sim_t < float(f.get('to_s', 0))):
            d = 'freeze'
    return d


def apply_image(img: np.ndarray, faults: list, rng, K=None, omega=None) -> tuple[np.ndarray, list]:
    """The image faults in a fixed order (optics, then shutter, then sensor noise, then compression). img: H × W × 3
    uint8 BGR. K: the 3 × 3 intrinsics (rolling shutter); omega: the camera's angular velocity, rad/s, in its optical
    frame (x right, y down, z forward). Returns (image, [kinds applied])."""
    order = ('occlusion', 'dirt', 'rain', 'glare', 'rolling_shutter', 'noise', 'compression')
    by = {f['kind']: f for f in faults or [] if f.get('kind') in IMAGE_FAULTS}
    done = []
    out = img
    for k in order:
        if k not in by:
            continue
        f = by[k]
        if k == 'occlusion':
            out = occlusion(out, float(f.get('cover', 0.3)), rng)
        elif k == 'dirt':
            out = dirt(out, float(f.get('cover', 0.1)), rng)
        elif k == 'rain':
            out = rain(out, float(f.get('cover', 0.1)), rng)
        elif k == 'glare':
            out = glare(out, float(f.get('strength', 0.5)), float(f.get('u', 0.5)), float(f.get('v', 0.3)))
        elif k == 'rolling_shutter':
            if K is None or omega is None:
                continue
            out = rolling_shutter(out, float(f.get('readout_ms', 30.0)) / 1000.0, K, omega)
        elif k == 'noise':
            out = noise(out, float(f.get('sigma', 3.0)), rng)
        elif k == 'compression':
            out = compression(out, int(f.get('quality', 50)))
        done.append(k)
    return out, done


def noise(img, sigma, rng):
    n = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + n, 0, 255).astype(np.uint8)


def _blob_mask(h, w, cover, rng, rmin, rmax):
    """A mask of seeded discs covering about `cover` of the image (discs added until the share is reached)."""
    import cv2
    m = np.zeros((h, w), np.uint8)
    target = cover * h * w
    area = 0
    for _ in range(10000):
        if area >= target:
            break
        r = int(rng.uniform(rmin, rmax) * min(h, w))
        cv2.circle(m, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), max(r, 1), 255, -1)
        area = int(np.count_nonzero(m))
    return m


def dirt(img, cover, rng):
    import cv2
    h, w = img.shape[:2]
    m = _blob_mask(h, w, cover, rng, 0.01, 0.06).astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), max(min(h, w) * 0.006, 1.0))
    tint = np.array([30.0, 38.0, 45.0])                   # a brown-grey film, BGR
    out = img.astype(np.float32) * (1 - 0.92 * m[..., None]) + tint * 0.92 * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def rain(img, cover, rng):
    import cv2
    h, w = img.shape[:2]
    out = img.copy()
    blurred = cv2.GaussianBlur(img, (0, 0), max(min(h, w) * 0.01, 1.5))
    target = cover * h * w
    covered = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(5000):
        if covered.sum() >= target:
            break
        r = max(int(rng.uniform(0.008, 0.035) * min(h, w)), 2)
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
        sy, sx = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
        d = np.hypot(sx - cx, sy - cy)
        inside = d <= r
        # a drop is a lens: it shows an inverted, demagnified patch of the scene around it (heuristic)
        src_x = np.clip(cx - (sx - cx) * 2.5, 0, w - 1).astype(int)
        src_y = np.clip(cy - (sy - cy) * 2.5, 0, h - 1).astype(int)
        patch = out[y0:y1, x0:x1]
        patch[inside] = blurred[src_y[inside], src_x[inside]]
        covered[y0:y1, x0:x1] |= inside
    return out


def glare(img, strength, u, v):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - u * w, yy - v * h) / max(h, w)
    glow = strength * 255.0 * (np.exp(-(r / 0.08) ** 2) + 0.35 * np.exp(-r / 0.35))
    return np.clip(img.astype(np.float32) + glow[..., None], 0, 255).astype(np.uint8)


def rolling_shutter(img, readout_s, K, omega):
    """Rows exposed from −readout/2 (top) to +readout/2 (bottom) around the middle row; each row seen through the
    rotation the camera turns during its offset (exp of the angular velocity × time), re-projected with K."""
    import cv2
    h, w = img.shape[:2]
    K = np.asarray(K, float)
    Ki = np.linalg.inv(K)
    om = np.asarray(omega, float)
    map_x = np.empty((h, w), np.float32)
    map_y = np.empty((h, w), np.float32)
    xs = np.arange(w, dtype=np.float64)
    for row in range(h):
        t = (row / max(h - 1, 1) - 0.5) * readout_s
        theta = om * t
        a = np.linalg.norm(theta)
        if a < 1e-12:
            Rm = np.eye(3)
        else:
            k = theta / a
            Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rm = np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx
        # the pixel of this row seen at time t shows the ray that, in the middle row's frame, is Rm @ ray
        rays = Ki @ np.vstack([xs, np.full(w, row, float), np.ones(w)])
        src = K @ (Rm @ rays)
        map_x[row] = (src[0] / src[2]).astype(np.float32)
        map_y[row] = (src[1] / src[2]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compression(img, quality):
    import cv2
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def occlusion(img, cover, rng):
    """Mud on the lens: one dark, opaque ellipse with a soft edge covering about `cover` of the image, centred at a
    seeded place in the image's middle half."""
    import cv2
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    ar = rng.uniform(0.6, 1.6)                                   # its aspect ratio
    area = cover * h * w
    a = math.sqrt(area * ar / math.pi)
    b = area / (math.pi * a)
    c = (int(rng.uniform(0.25, 0.75) * w), int(rng.uniform(0.25, 0.75) * h))
    cv2.ellipse(m, c, (max(int(a), 1), max(int(b), 1)), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(min(h, w) * 0.01, 1.0))
    tint = np.array([22.0, 30.0, 38.0])                          # mud, BGR
    out = img.astype(np.float32) * (1 - m[..., None]) + tint * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def active_of(faults: list, sim_t: float, kinds=None) -> list:
    """The faults acting at sim_t (of the given kinds)."""
    return [f for f in faults or [] if active(f, sim_t) and (kinds is None or f.get('kind') in kinds)]


def silenced(faults: list, sim_t: float) -> bool:
    """Whether a dropout (or a total drop) silences the sensor at sim_t."""
    return any(f.get('kind') == 'dropout' or (f.get('kind') == 'drop' and float(f.get('rate', 0.0)) >= 1.0)
               for f in active_of(faults, sim_t))


def apply_points(points: np.ndarray, faults: list, sim_t: float, rng) -> np.ndarray:
    """A lidar's points (N × ≥ 3 in its frame, CARLA axes) with the sector_loss and packet_drop faults acting at sim_t
    applied (packet_drop draws one 360-value mask from rng per call: one per sweep)."""
    P = np.asarray(points)
    act = active_of(faults, sim_t, ('sector_loss', 'packet_drop'))
    if not act or not len(P):
        return P
    az = (np.degrees(np.arctan2(P[:, 1], P[:, 0])) + 360.0) % 360.0
    keep = np.ones(len(P), bool)
    for f in act:
        if f['kind'] == 'sector_loss':
            a0, a1 = float(f['from_deg']) % 360.0, float(f['to_deg']) % 360.0
            inside = (az >= a0) & (az < a1) if a0 <= a1 else (az >= a0) | (az < a1)
            keep &= ~inside
        else:
            lost = rng.random(360) < float(f.get('rate', 0.5))
            keep &= ~lost[np.floor(az).astype(int) % 360]
    return P[keep]


def gnss_offset(faults: list, sim_t: float, seed: int, sid: str = '') -> tuple:
    """(north, east) metres a multipath fault adds to a fix at sim_t: jump_m in a direction drawn per period from a
    generator seeded by (seed, sensor, period index) — the same at every call for the same time."""
    import zlib
    n = e = 0.0
    for f in active_of(faults, sim_t, ('multipath',)):
        k = int(math.floor((sim_t - float(f.get('onset_s', 0.0))) / float(f.get('period_s', 2.0))))
        g = np.random.default_rng([int(seed), zlib.crc32(sid.encode()), k])
        ang = g.uniform(0, 2 * math.pi)
        r = float(f.get('jump_m', 5.0)) * g.uniform(0.5, 1.0)
        n += r * math.cos(ang)
        e += r * math.sin(ang)
    return n, e

