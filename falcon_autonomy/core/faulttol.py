"""Sensor fault and calibration-drift tolerance campaigns: how far can each fault go before perception or the stack
fails? (numpy at import; OpenCV, the model runtime and the simulator only inside the functions; docs/fault-tolerance.md
is the manual.)

core.faults injects a fault of a chosen size; this module grades it. Each fault MODEL has ONE severity in a physical
unit, a stated model, and is deterministic given the seed. A campaign bisects, per (sensor, fault), the severity at
which a criterion flips from pass to fail, and writes a tolerance table: sensor × fault -> the largest tolerated
severity with its bracket [last pass, first fail].

The fault models (MODELS; `falcon faulttol models` prints them):

    cam_yaw / cam_pitch / cam_roll   deg   the camera is turned on its mount while the stack keeps the nominal
                                           calibration. Offline: the recorded image is re-projected by the pure-rotation
                                           homography K Rᵀ K⁻¹ (exact for a rotation about the optical centre; what
                                           the turned camera would newly see at the image's edge is black)
    lidar_yaw                        deg   the lidar is turned about its vertical axis; its points are delivered in the
                                           turned sensor's frame (core.faults.rotate_points: an exact rotation)
    lidar_sector                     deg   no returns in an azimuth sector of this width centred straight ahead
    clock_offset                     ms    data captured at t is stamped t + offset: the stack pairs it with the world
                                           of a later moment. Offline the nearest recorded frame is used, so the
                                           severity is resolved to one frame period (stated per cell as `resolution`)
    dirty_lens                       share of the image covered by opaque dirt (core.faults.dirt; one seeded mask for
                                           the whole run — dirt stays where it is — that grows with the severity)
    frozen                           s     one freeze of this duration, starting at a seeded time in the run: the frames
                                           inside repeat the last one before it
    dropped                          rate  each frame (a lidar: each 1° azimuth packet) is lost with this probability;
                                           the stack holds the last camera frame delivered
    noise                            σ     Gaussian sensor noise of σ 8-bit levels (core.faults.noise)
    exposure                         EV    over-exposure by this many stops: gain 2^EV in linear light (gamma 2.2
                                           assumed), clipped
    gnss_bias                        m     every fix offset north by this distance (core.faults 'bias')

Common random numbers: every severity of a cell uses the same seed, the same frames (or scenario seeds) and nested
random draws (a frame dropped at rate 0.2 is also dropped at 0.3; the dirt mask at 10 % is inside the one at 20 %),
so the response differs between severities by the fault alone.

The criteria (the response is a number; the cell fails when it exceeds the threshold):

  offline, on recorded runs (no simulator):
    ap_drop         a registered detector (core.models) run on the faulted images, scored against the labels the
                    NOMINAL calibration and stamps give (core.metrics COCO AP, AP50 by default): AP without the fault
                    minus AP with it, in points
    recall_drop     no model: the share of labelled objects whose box in the delivered image no longer overlaps its
                    nominal box with IoU ≥ 0.5 (what a perfect detector would lose), in points. Geometric and timing
                    faults only (cam_*, clock_offset, frozen, dropped); the image-quality faults need a model and are
                    refused by name without one
    retention_drop  lidar: the share of actor-tagged points (the recorded per-point actor ids) that are lost or land
                    further than `lidar_tol_m` from where the nominal calibration puts them, in points. clock_offset
                    displaces every point by the ego's speed × the offset (a static-scene approximation: the actors'
                    own motion is ignored)
    position_error  GNSS: the bias against the in-lane margin (lane width − vehicle width) / 2 — analytic, the bias is
                    the error; reported so the table and the calibration statement have the row

  closed loop, through sim.service.study_service (stack runs, core.stack / workers.stack; the fault starts at onset_s):
    criticality     the run's criticality (core.criticality), the worst over the seeds, ≥ threshold
    fail            the run failed its scenario's pass checks (any seed)
    degraded        the stack's health monitor declared a failure (driver follower with degraded_mode: the size at
                    which the test stack STOPS tolerating the fault and starts its minimal-risk manoeuvre)
    stl             a signal-temporal-logic requirement violated (core.stlobjective): response −ρ, fails when ρ < 0
  Closed-loop faults are the ones the stack worker applies to the data a stack receives (CLOSED): lidar_yaw,
  lidar_sector, dropped (lidar packets), gnss_bias, dirty_lens, noise; the others are refused by name. The built-in
  follower drives from ground truth: a fault reaches its behaviour only through its degraded mode. A ROS 2 stack
  receives the faulted data itself.

The search (tolerance()): the response at severity 0 (must pass, else 'fails_nominal') and on a coarse grid up to the
model's maximum; if no grid point fails, 'tolerates_max'. The verdicts must be monotone (no pass above a fail): if they
are, the bracket between the last passing and the first failing grid point is bisected; if not, the cell says so
('not_monotone') and reports the LOWEST failing severity the grid found — a bisection would be meaningless there.

What is derived, not measured: calibration_statements() restates each tolerance as the recalibration or
synchronisation accuracy it implies ("extrinsic yaw of camera X must hold within ±0.8°"). They are derived from this
stack, these runs or scenarios, this criterion and these fault models — not a general requirement, and not a margin:
no safety factor is applied.

    falcon faulttol models [--json]
    falcon faulttol run PROJECT --name N --runs RUN [RUN …] [--model ID] [--sensor ID …] [--fault F …]
    falcon faulttol run PROJECT --name N --closed-loop --scenario S [--criterion degraded|criticality|fail|stl] …
    falcon faulttol show DIR [--json]

Output: <workspace>/faulttol/<name>/faulttol.json (SCHEMA) and tolerance.csv.
"""
from __future__ import annotations

import csv
import json
import math
import os
import zlib

import numpy as np

SCHEMA = 'falcon.faulttol/1'
CAMERAS = ('camera.pinhole', 'camera.fisheye')
GEOMETRIC = ('cam_yaw', 'cam_pitch', 'cam_roll', 'clock_offset', 'frozen', 'dropped')
DEFAULTS = {'ap_drop_pt': 5.0, 'recall_drop_pt': 5.0, 'retention_drop_pt': 10.0, 'lidar_tol_m': 0.5,
            'lane_width_m': 3.5, 'ego_width_m': 1.9, 'criticality': 0.5, 'iou': 0.5, 'ap_metric': 'AP50',
            'grid': 5, 'steps': 6, 'max_frames': 60, 'onset_s': 1.0}

# name -> the model: which sensor types, the unit, the largest severity tried, the smallest bracket worth bisecting to
MODELS = {
    'cam_yaw': {'title': 'Camera yaw misalignment', 'sensors': ('camera.pinhole',), 'unit': 'deg', 'max': 5.0,
                'resolution': 0.02, 'model': 'camera turned about its vertical axis; the stack keeps the nominal '
                'calibration; image re-projected by the rotation homography'},
    'cam_pitch': {'title': 'Camera pitch misalignment', 'sensors': ('camera.pinhole',), 'unit': 'deg', 'max': 5.0,
                  'resolution': 0.02, 'model': 'camera pitched on its mount; nominal calibration kept; rotation '
                  'homography'},
    'cam_roll': {'title': 'Camera roll misalignment', 'sensors': ('camera.pinhole',), 'unit': 'deg', 'max': 10.0,
                 'resolution': 0.05, 'model': 'camera rolled about its optical axis; nominal calibration kept; '
                 'rotation homography'},
    'lidar_yaw': {'title': 'Lidar extrinsic drift (yaw)', 'sensors': ('lidar',), 'unit': 'deg', 'max': 5.0,
                  'resolution': 0.02, 'model': 'lidar turned about its vertical axis; points delivered in the turned '
                  'frame (exact rotation)'},
    'lidar_sector': {'title': 'Lidar sector dropout', 'sensors': ('lidar',), 'unit': 'deg', 'max': 180.0,
                     'resolution': 1.0, 'model': 'no returns in an azimuth sector of this width centred ahead'},
    'clock_offset': {'title': 'Sensor clock offset', 'sensors': CAMERAS + ('lidar',), 'unit': 'ms', 'max': 500.0,
                     'resolution': 1.0, 'model': 'data captured at t stamped t + offset; offline: the nearest '
                     'recorded frame (a camera), ego speed × offset (a lidar)'},
    'dirty_lens': {'title': 'Dirty lens', 'sensors': CAMERAS, 'unit': 'share', 'max': 0.5, 'resolution': 0.005,
                   'model': 'opaque dirt blotches covering this share of the image; one mask per run, nested in '
                   'severity'},
    'frozen': {'title': 'Frozen frame', 'sensors': CAMERAS, 'unit': 's', 'max': 2.0, 'resolution': 0.01,
               'model': 'one freeze of this duration at a seeded time; frames inside repeat the last one before'},
    'dropped': {'title': 'Dropped frames', 'sensors': CAMERAS + ('lidar',), 'unit': 'rate', 'max': 0.9,
                'resolution': 0.01, 'model': 'each frame (lidar: each 1° packet) lost with this probability; the last '
                'camera frame is held'},
    'noise': {'title': 'Image noise', 'sensors': CAMERAS, 'unit': 'σ (8-bit levels)', 'max': 50.0, 'resolution': 0.25,
              'model': 'Gaussian noise of this standard deviation on every channel'},
    'exposure': {'title': 'Exposure error', 'sensors': CAMERAS, 'unit': 'EV', 'max': 3.0, 'resolution': 0.02,
                 'model': 'over-exposure: gain 2^EV in linear light (gamma 2.2 assumed), clipped'},
    'gnss_bias': {'title': 'GNSS bias', 'sensors': ('gnss',), 'unit': 'm', 'max': 20.0, 'resolution': 0.01,
                  'model': 'every fix offset north by this distance'},
}
CLOSED = ('lidar_yaw', 'lidar_sector', 'dropped', 'gnss_bias', 'dirty_lens', 'noise')
CLOSED_CRITERIA = ('criticality', 'fail', 'degraded', 'stl')
CLOSED_MAX = {'dirty_lens': 0.5}           # core.faults.LIMITS: the largest the stack worker accepts
STATUSES = ('bracketed', 'tolerates_max', 'fails_nominal', 'not_monotone', 'refused')
STATEMENT_NOTE = ('derived from this stack, these runs or scenarios, this criterion and these fault models; not a '
                  'general requirement, and no safety factor is applied')


class FaultTolError(Exception):
    """A campaign that cannot run as asked (the message says what and why)."""


def applicable(fault: str, sensor_type: str) -> bool:
    return fault in MODELS and sensor_type in MODELS[fault]['sensors']


def cell_rng(seed: int, sensor: str, fault: str, *more):
    """The generator of one (sensor, fault) cell — the same at every severity (common random numbers)."""
    return np.random.default_rng([int(seed), zlib.crc32(sensor.encode()), zlib.crc32(fault.encode()),
                                  *[int(m) for m in more]])


# ====================================================================================================== the search
def tolerance(evaluate, threshold: float, lo: float, hi: float, grid: int = 5, steps: int = 6,
              resolution: float = 0.0) -> dict:
    """The largest tolerated severity of one cell. evaluate(severity) -> response (a float; deterministic); the cell
    FAILS at a severity when its response > threshold. Returns {'status' (STATUSES), 'tolerated', 'bracket':
    [last pass, first fail], 'monotone', 'curve': [[severity, response, failed]] sorted, 'evaluations', 'note'}."""
    seen = {}

    def f(s):
        s = float(s)
        if s not in seen:
            seen[s] = float(evaluate(s))
        return seen[s] > threshold

    def done(status, tolerated, bracket, note=''):
        curve = [[s, seen[s], bool(seen[s] > threshold)] for s in sorted(seen)]
        fails = [c[2] for c in curve]
        mono = not any(a and not b for a, b in zip(fails, fails[1:]))
        return {'status': status, 'tolerated': tolerated, 'bracket': bracket, 'monotone': mono, 'curve': curve,
                'evaluations': len(seen), 'threshold': float(threshold), 'note': note}
    if f(lo):
        return done('fails_nominal', None, [None, float(lo)],
                    'the criterion fails without the fault: there is no tolerance to measure')
    pts = [lo + (hi - lo) * k / max(int(grid) - 1, 1) for k in range(max(int(grid), 2))]
    pts[-1] = float(hi)
    flags = [f(p) for p in pts]
    if not any(flags):
        return done('tolerates_max', float(hi), [float(hi), None],
                    f'no failure up to the largest severity tried ({hi:g})')
    first = flags.index(True)
    if any(not x for x in flags[first:]):
        return done('not_monotone', float(pts[first - 1]), [float(pts[first - 1]), float(pts[first])],
                    'the response is not monotone in the severity (a pass above a fail on the grid): the lowest '
                    'failing severity the coarse grid found is reported, not bisected')
    a, b = float(pts[first - 1]), float(pts[first])
    for _ in range(max(int(steps), 0)):
        if b - a <= max(float(resolution), 1e-12):
            break
        m = 0.5 * (a + b)
        if f(m):
            b = m
        else:
            a = m
    return done('bracketed', a, [a, b])


# ============================================================================================= camera fault models
def camera_K(sensor: dict) -> np.ndarray:
    from . import lens as L
    if sensor.get('type') != 'camera.pinhole':
        raise FaultTolError(f'{sensor.get("id")}: the misalignment model re-projects a pinhole image only')
    return np.asarray(L.intrinsics(sensor).K, float)


def misalign_homography(K, deg) -> np.ndarray:
    """H with u_turned ~ H u_nominal: where a pixel of the nominal image lies in the image of the camera turned by
    deg = [roll, pitch, yaw] on its mount (CARLA's convention; a rotation about the optical centre)."""
    from .faults import mount_rotation_matrix
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])      # optical (x right, y down, z fwd) <- CARLA
    Ro = A @ mount_rotation_matrix(deg) @ A.T                                # nominal optical <- turned optical
    K = np.asarray(K, float)
    return K @ Ro.T @ np.linalg.inv(K)


def misalign_image(img, K, deg):
    import cv2
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, misalign_homography(K, deg), (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def misalign_box(box, H, w, h):
    """An [x0, y0, x1, y1] box of the nominal image as it lies in the turned camera's image (the bounding box of its
    warped corners, clipped to the image); None when it left the image."""
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    c = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]]) @ np.asarray(H, float).T
    if np.any(c[:, 2] <= 1e-9):
        return None
    c = c[:, :2] / c[:, 2:3]
    b = [max(float(c[:, 0].min()), 0.0), max(float(c[:, 1].min()), 0.0),
         min(float(c[:, 0].max()), float(w)), min(float(c[:, 1].max()), float(h))]
    return b if b[2] > b[0] and b[3] > b[1] else None


def deg_of(fault: str, severity: float) -> list:
    return {'cam_roll': [severity, 0.0, 0.0], 'cam_pitch': [0.0, severity, 0.0],
            'cam_yaw': [0.0, 0.0, severity], 'lidar_yaw': [0.0, 0.0, severity]}[fault]


def exposure(img, ev: float):
    """Over-exposure by ev stops: gain 2^ev in linear light (display gamma 2.2 assumed), clipped."""
    x = (img.astype(np.float32) / 255.0) ** 2.2 * (2.0 ** float(ev))
    return (np.clip(x, 0.0, 1.0) ** (1.0 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def delivered_index(fault: str, severity: float, times, rng) -> list:
    """For the timing faults: per label frame k, the index of the recorded frame the stack receives as frame k
    (clock_offset: the frame captured offset earlier, nearest; frozen: one seeded freeze; dropped: hold the last
    frame delivered). times: the frames' simulation times (one run, ascending). The draws do not depend on the
    severity (common random numbers)."""
    t = np.asarray(times, float)
    n = len(t)
    u = rng.random(n)                                   # drawn first, always: the same stream at every severity
    start = float(rng.uniform(0.2, 0.5))
    idx = list(range(n))
    if n == 0 or severity <= 0:
        return idx
    if fault == 'clock_offset':
        return [int(np.argmin(np.abs(t - (tk - severity / 1000.0)))) for tk in t]
    if fault == 'frozen':
        t0 = t[0] + start * (t[-1] - t[0])
        k0 = max(int(np.searchsorted(t, t0)) - 1, 0)
        return [k0 if t0 <= tk < t0 + severity else k for k, tk in enumerate(t)]
    if fault == 'dropped':
        last = 0
        for k in range(n):
            if k == 0 or u[k] >= severity:
                last = k
            idx[k] = last
        return idx
    raise FaultTolError(f'{fault} is not a timing fault')


class CameraCell:
    """One camera of the recorded runs, loaded once: its frames with their own image and objects, per run."""

    def __init__(self, runs: list, cid: str, max_frames: int = 60):
        from .exporters import ExportError, Run
        self.cid = cid
        self.seqs = []                                   # per run: [(Run, record, objects)]
        try:
            loaded = [Run(r) for r in runs]
        except ExportError as e:
            raise FaultTolError(str(e)) from None
        for run in loaded:
            if cid not in run.cams:
                continue
            recs = [r for r in run.camera_steps(cid) if run.file(r, cid, 'objects')]
            if recs:
                self.seqs.append([(run, r, run.objects(r, cid) or []) for r in recs])
        total = sum(len(s) for s in self.seqs)
        if not total:
            raise FaultTolError(f'camera {cid}: no frames with object labels (boxes2d) in the runs')
        every = max(int(math.ceil(total / max(int(max_frames), 1))), 1)
        self.every = every
        self.sensor = self.seqs[0][0][0].sensors[cid]
        self.w, self.h = int(self.sensor['width']), int(self.sensor['height'])
        self.period_ms = 1000.0 / self.seqs[0][0][0].fps
        self._img = {}

    def frames(self):
        """(sequence index, frame index) of the label frames scored (every n-th, to stay within max_frames)."""
        return [(i, k) for i, s in enumerate(self.seqs) for k in range(0, len(s), self.every)]

    def image(self, i, k):
        import cv2
        if (i, k) not in self._img:
            run, rec, _ = self.seqs[i][k]
            p = run.p(run.file(rec, self.cid, 'rgb'))
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise FaultTolError(f'cannot read {p}')
            if len(self._img) > 400:
                self._img.clear()
            self._img[(i, k)] = img
        return self._img[(i, k)]

    def delivered(self, fault, severity, seed):
        """{(i, k): j}: the frame of sequence i delivered as label frame k (timing faults; identity otherwise)."""
        out = {}
        for i, s in enumerate(self.seqs):
            if fault in ('clock_offset', 'frozen', 'dropped'):
                idx = delivered_index(fault, severity, [r['sim_t'] for _, r, _ in s],
                                      cell_rng(seed, self.cid, fault, i))
            else:
                idx = list(range(len(s)))
            out.update({(i, k): j for k, j in enumerate(idx)})
        return out

    def faulted_image(self, fault, severity, seed, i, k, j):
        """The BGR image the stack receives as label frame (i, k)."""
        from . import faults as F
        img = self.image(i, j)
        if severity <= 0:
            return img
        if fault in ('cam_yaw', 'cam_pitch', 'cam_roll'):
            return misalign_image(img, camera_K(self.sensor), deg_of(fault, severity))
        if fault == 'dirty_lens':
            return F.dirt(img, severity, cell_rng(seed, self.cid, fault))       # the same mask on every frame
        if fault == 'noise':
            return F.noise(img, severity, cell_rng(seed, self.cid, fault, i, k))
        if fault == 'exposure':
            return exposure(img, severity)
        return img

    def recall_drop(self, fault, severity, seed, iou_min=0.5) -> float:
        """The share (in points) of labelled objects whose box in the delivered image misses its nominal box."""
        from . import harness as HN
        if fault not in GEOMETRIC:
            raise FaultTolError(f'{fault} needs a model (it changes the image, not where things are)')
        H = misalign_homography(camera_K(self.sensor), deg_of(fault, severity)) if fault.startswith('cam_') else None
        deliv = self.delivered(fault, severity, seed)
        n = lost = 0
        for i, k in self.frames():
            j = deliv[(i, k)]
            there = {o.get('track_id'): o for o in self.seqs[i][j][2] if o.get('bbox')}
            for o in self.seqs[i][k][2]:
                if not o.get('bbox') or (o.get('pixels') or 0) < HN.MIN_PIXELS or \
                        (o.get('occluded') or 0.0) > HN.MAX_OCCLUDED:
                    continue
                n += 1
                if H is not None:
                    seen = misalign_box(o['bbox'], H, self.w, self.h)
                else:
                    seen = (there.get(o.get('track_id')) or {}).get('bbox') if j != k else o['bbox']
                lost += iou(o['bbox'], seen) < iou_min
        if not n:
            raise FaultTolError(f'camera {self.cid}: no labelled objects large enough to score in the frames')
        return 100.0 * lost / n

    def ap(self, fault, severity, seed, runner, model: dict, metric: str = 'AP50') -> float:
        """The detector's AP (0–100) on the faulted images against the nominal labels."""
        import cv2
        from . import harness as HN
        from . import metrics as M
        classes = HN.model_classes(model)
        cmap = model.get('class_map') or {}
        deliv = self.delivered(fault, severity, seed)
        frames = []
        for i, k in self.frames():
            run, rec, objs = self.seqs[i][k]
            img = self.faulted_image(fault, severity, seed, i, k, deliv[(i, k)])
            meta = {'objects': objs, 'weather': run.cfg.get('weather') or {}, 'run': run.name, 'step': rec['step']}
            out = runner.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), meta)
            gts = [{'bbox': HN._xywh(o['bbox']), 'class': o['class'],
                    'ignore': int((o.get('pixels') or 0) < HN.MIN_PIXELS or
                                  (o.get('occluded') or 0.0) > HN.MAX_OCCLUDED)}
                   for o in objs if o.get('class') in classes and o.get('bbox')]
            dets = []
            for d in out:
                c = d['class'] if model['runtime'] == 'python' else cmap.get(d['class'], d['class'])
                if c in classes:
                    dets.append({'bbox': HN._xywh(d['bbox']), 'class': c, 'score': float(d['score'])})
            frames.append({'image_id': f'{i}/{k}', 'drive': run.key, 'step': rec['step'], 'gt': gts, 'det': dets})
        v = M.coco_summarise(M.coco_match(frames)).get(metric)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise FaultTolError(f'camera {self.cid}: {metric} is undefined on these frames (no labelled objects of '
                                'the model\'s classes)')
        return 100.0 * float(v)


# ============================================================================================== lidar fault models
class LidarCell:
    """One lidar of the recorded runs: the actor-tagged points of its sweeps and the ego's speed at each."""

    def __init__(self, runs: list, sid: str, max_frames: int = 60):
        from .exporters import ExportError, Run
        self.sid = sid
        self.sweeps = []                                 # (points N × 3 of actors, ego speed m/s)
        try:
            loaded = [Run(r) for r in runs]
        except ExportError as e:
            raise FaultTolError(str(e)) from None
        cand = []
        for run in loaded:
            if sid not in run.sensors:
                continue
            recs = [r for r in run.frames if run.file(r, sid, 'lidar')]
            for a, r in enumerate(recs):
                b = recs[a + 1] if a + 1 < len(recs) else (recs[a - 1] if a else None)
                v = r.get('speed_mps')
                if v is None and b is not None and r.get('ego') and b.get('ego') and b['sim_t'] != r['sim_t']:
                    v = math.hypot(b['ego']['x'] - r['ego']['x'], b['ego']['y'] - r['ego']['y']) / \
                        abs(b['sim_t'] - r['sim_t'])
                cand.append((run, r, float(v or 0.0)))
        every = max(int(math.ceil(len(cand) / max(int(max_frames), 1))), 1)
        for run, r, v in cand[::every]:
            got = run.lidar(r, sid)
            if got is None:
                continue
            pts, _tags, ids = got
            keep = ids != 0
            if keep.any():
                self.sweeps.append((pts[keep, :3].astype(np.float64), v))
        if not self.sweeps:
            raise FaultTolError(f'lidar {sid}: no actor-tagged points in the runs\' sweeps')

    def retention_drop(self, fault, severity, seed, tol_m=0.5) -> float:
        """The share (points) of actor-tagged points lost or displaced by more than tol_m."""
        from .faults import rotate_points
        n = kept = 0
        for a, (P, v) in enumerate(self.sweeps):
            n += len(P)
            if severity <= 0:
                kept += len(P)
                continue
            az = (np.degrees(np.arctan2(P[:, 1], P[:, 0])) + 360.0) % 360.0
            if fault == 'lidar_yaw':
                Q = rotate_points(P, deg_of(fault, severity))
                kept += int(np.count_nonzero(np.linalg.norm(Q - P, axis=1) <= tol_m))
            elif fault == 'lidar_sector':
                off = np.minimum(az, 360.0 - az)
                kept += int(np.count_nonzero(off > severity / 2.0))
            elif fault == 'dropped':
                u = cell_rng(seed, self.sid, fault, a).random(360)
                kept += int(np.count_nonzero(u[np.floor(az).astype(int) % 360] >= severity))
            elif fault == 'clock_offset':
                kept += len(P) if v * severity / 1000.0 <= tol_m else 0
            else:
                raise FaultTolError(f'{fault} is not a lidar fault')
        return 100.0 * (1.0 - kept / max(n, 1))


# ================================================================================================= offline campaign
def _cell(sensor: dict, fault: str, criterion: str, threshold: float, unit_response: str, **extra) -> dict:
    m = MODELS[fault]
    return dict({'sensor': sensor['id'], 'sensor_type': sensor['type'], 'fault': fault, 'title': m['title'],
                 'unit': m['unit'], 'model': m['model'], 'max': m['max'], 'criterion': criterion,
                 'threshold': threshold, 'response_unit': unit_response}, **extra)


def _refused(sensor, fault, why) -> dict:
    m = MODELS[fault]
    return {'sensor': sensor['id'], 'sensor_type': sensor['type'], 'fault': fault, 'title': m['title'],
            'unit': m['unit'], 'model': m['model'], 'max': m['max'], 'status': 'refused', 'tolerated': None,
            'bracket': [None, None], 'monotone': None, 'curve': [], 'evaluations': 0, 'note': why}


def _pick(rig: dict, sensors, faults) -> list:
    """[(sensor, fault)] of the rig that the request names (all applicable when it names none)."""
    ids = {s['id'] for s in rig.get('sensors') or []}
    for s in sensors or ():
        if s not in ids:
            raise FaultTolError(f'no sensor {s!r} in the rig ({", ".join(sorted(ids))})')
    for f in faults or ():
        if f not in MODELS:
            raise FaultTolError(f'unknown fault {f!r}: one of {", ".join(MODELS)}')
    out = [(s, f) for s in rig.get('sensors') or [] if not sensors or s['id'] in sensors
           for f in MODELS if (not faults or f in faults) and applicable(f, s['type'])]
    if not out:
        raise FaultTolError('no (sensor, fault) pair to test: the faults asked for do not apply to the sensors')
    return out


def offline_campaign(runs: list, sensors=None, faults=None, model: dict | None = None, workspace: str = '',
                     seed: int = 0, options: dict | None = None, progress=None, emit=None) -> list:
    """The cells of an offline campaign on recorded runs (module doc). model: a registered detector for ap_drop
    (None: recall_drop on the geometric faults; the image-quality faults are refused by name)."""
    from .exporters import ExportError, Run
    o = dict(DEFAULTS, **(options or {}))
    if not runs:
        raise FaultTolError('an offline campaign needs recorded runs')
    try:
        rig = Run(runs[0]).rig
    except ExportError as e:
        raise FaultTolError(str(e)) from None
    pairs = _pick(rig, sensors, faults)
    runner, locks = None, []
    if model is not None:
        from . import milsearch as MS
        from .models import Runner
        why = MS.evaluable(model, workspace)
        if why or model.get('kind') != 'detector':
            raise FaultTolError(f'the model cannot be used: {why or "it is not a detector"}')
        if model.get('runtime') != 'python':
            from .training import gpu_locks
            locks = gpu_locks(emit)
        runner = Runner(model, workspace)
    cache, cells = {}, []
    try:
        for n, (s, f) in enumerate(pairs):
            m = MODELS[f]
            hi, res = m['max'], m['resolution']
            try:
                if s['type'] in CAMERAS:
                    cam = cache.get(s['id']) or cache.setdefault(s['id'], CameraCell(runs, s['id'], o['max_frames']))
                    if f == 'clock_offset':
                        res = cam.period_ms
                    if runner is not None:
                        base = cam.ap(f, 0.0, seed, runner, model, o['ap_metric'])
                        c = _cell(s, f, 'ap_drop', o['ap_drop_pt'], f'{o["ap_metric"]} points lost',
                                  baseline=round(base, 3), model_id=model.get('id'),
                                  synthetic_model=bool(getattr(getattr(runner, 'plugin', None), 'synthetic', False)))

                        def ev(x, cam=cam, f=f, base=base):
                            return base - cam.ap(f, x, seed, runner, model, o['ap_metric'])
                    elif f in GEOMETRIC:
                        c = _cell(s, f, 'recall_drop', o['recall_drop_pt'],
                                  f'points of objects displaced beyond IoU {o["iou"]:g}')

                        def ev(x, cam=cam, f=f):
                            return cam.recall_drop(f, x, seed, o['iou'])
                    else:
                        cells.append(_refused(s, f, f'{f} changes the image, not where things are: it needs a '
                                                    'detector (--model)'))
                        continue
                elif s['type'] == 'lidar':
                    lid = cache.get(s['id']) or cache.setdefault(s['id'], LidarCell(runs, s['id'], o['max_frames']))
                    c = _cell(s, f, 'retention_drop', o['retention_drop_pt'],
                              f'points of actor returns lost or moved > {o["lidar_tol_m"]:g} m')

                    def ev(x, lid=lid, f=f):
                        return lid.retention_drop(f, x, seed, o['lidar_tol_m'])
                else:                                                    # gnss_bias: analytic
                    margin = (o['lane_width_m'] - o['ego_width_m']) / 2.0
                    c = _cell(s, f, 'position_error', margin, 'm of position error', analytic=True)

                    def ev(x):
                        return float(x)
                c.update(tolerance(ev, c['threshold'], 0.0, hi, o['grid'], o['steps'], res))
                c['resolution'] = res
            except FaultTolError as e:
                c = _refused(s, f, str(e))
            cells.append(c)
            if progress:
                progress((n + 1) / len(pairs), f'{s["id"]} · {f}: {c["status"]}')
    finally:
        for lk in locks:
            lk.release()
    return cells


# ============================================================================================= closed-loop campaign
def closed_fault(sensor: dict, fault: str, severity: float, onset_s: float) -> dict | None:
    """The timed fault (core.faults, as a stack run takes it) of a model at a severity; None at severity 0."""
    if fault not in CLOSED:
        raise FaultTolError(f'{fault} cannot be injected into a closed-loop run in this build (the stack worker '
                            f'applies: {", ".join(CLOSED)})')
    if severity <= 0:
        return None
    base = {'sensor': sensor['id'], 'onset_s': float(onset_s)}
    if fault == 'lidar_yaw':
        return dict(base, kind='extrinsic_drift', deg=[0.0, 0.0, float(severity)])
    if fault == 'lidar_sector':
        return dict(base, kind='sector_loss', from_deg=-severity / 2.0, to_deg=severity / 2.0)
    if fault == 'dropped':
        if sensor['type'] != 'lidar':
            raise FaultTolError('dropped frames reach a closed-loop run as lidar packet drops only')
        return dict(base, kind='packet_drop', rate=float(severity))
    if fault == 'gnss_bias':
        return dict(base, kind='bias', north_m=float(severity), east_m=0.0)
    if fault == 'dirty_lens':
        return dict(base, kind='dirt', cover=float(severity))
    return dict(base, kind='noise', sigma=float(severity))


def closed_response(run: dict, run_dir: str, criterion: str, stl_spec: dict | None = None) -> float:
    """One finished stack run's response under a closed-loop criterion (module doc)."""
    if run.get('status') != 'ok':
        raise FaultTolError(f'a run did not finish ({run.get("status")}): {run.get("error")}')
    sc = run.get('scenario') or {}
    if criterion == 'criticality':
        return float((sc.get('criticality') or {}).get('criticality') or 0.0)
    if criterion == 'fail':
        return 0.0 if sc.get('passed', True) else 1.0
    if criterion == 'degraded':
        return 1.0 if run.get('degraded') else 0.0
    from . import stlobjective as SO
    s = SO.score_run(run_dir, stl_spec)
    rho = SO.fnum(s.get('rho')) if s.get('rho') is not None else None
    if rho is None:
        raise FaultTolError(f'the requirement cannot be evaluated on a run: {s.get("error") or s.get("state")}')
    return -float(rho)


def closed_campaign(svc, project, scenario: dict, out_dir: str, world: str, rig_id: str, sensors=None, faults=None,
                    criterion: str = 'degraded', threshold: float | None = None, stl_spec: dict | None = None,
                    seeds=(0,), values: dict | None = None, driver: str = 'follower', degraded_mode: bool = True,
                    fps: float = 20.0, options: dict | None = None, progress=None) -> list:
    """The cells of a closed-loop campaign: per severity one stack run per seed (the same seeds at every severity),
    cached in out_dir/runs/<key>; the response is the worst over the seeds."""
    from . import stack as ST
    from ..sim.stacksuite import run_one
    o = dict(DEFAULTS, **(options or {}))
    if criterion not in CLOSED_CRITERIA:
        raise FaultTolError(f'criterion {criterion!r}: one of {", ".join(CLOSED_CRITERIA)}')
    if criterion == 'stl' and not stl_spec:
        raise FaultTolError('the stl criterion needs a requirement')
    rig = project.item('rigs', rig_id)
    if rig is None:
        raise FaultTolError(f'no rig {rig_id!r}')
    thr = float(threshold) if threshold is not None else (o['criticality'] - 1e-9 if criterion == 'criticality'
                                                          else 0.0 if criterion == 'stl' else 0.5)
    pairs = [(s, f) for s, f in _pick(rig, sensors, faults or [f for f in CLOSED])
             if not (f == 'dropped' and s['type'] != 'lidar' and not faults)]
    cells = []
    for n, (s, f) in enumerate(pairs):
        m = MODELS[f]
        runs_made = []

        def ev(x, s=s, f=f):
            worst = -math.inf
            for seed in seeds:
                flt = closed_fault(s, f, x, o['onset_s'])
                try:
                    cfg = ST.stack_config(project, world, rig_id, '/pending', svc.port, svc.tm_port, driver,
                                          scenario, seed=int(seed), values=values, fps=fps,
                                          render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                                          workspace=project.workspace, faults=[flt] if flt else None,
                                          degraded_mode=degraded_mode)
                except ValueError as e:
                    raise FaultTolError(str(e)) from None
                key = ST.run_key(cfg)
                cfg['out'] = os.path.join(out_dir, 'runs', key[:16])
                cfg['cache_key'] = key
                cfg['name'] = f'{s["id"]} {f} {x:g}'
                rp = os.path.join(cfg['out'], 'run.json')
                run = None
                if os.path.isfile(rp):
                    with open(rp, encoding='utf-8') as fh:
                        run = json.load(fh)
                    if run.get('status') != 'ok':
                        run = None
                if run is None:
                    run = run_one(svc, cfg, title=cfg['name'])
                runs_made.append(os.path.relpath(cfg['out'], out_dir))
                worst = max(worst, closed_response(run, cfg['out'], criterion, stl_spec))
            return worst
        try:
            closed_fault(s, f, 1e-6, 0.0)                              # refused by name before anything runs
            c = _cell(s, f, criterion, thr, {'criticality': 'criticality (worst seed)', 'fail': '1 = a run failed',
                                             'degraded': '1 = the stack declared a failure',
                                             'stl': '−ρ (worst seed)'}[criterion])
            c.update(tolerance(ev, thr, 0.0, min(m['max'], CLOSED_MAX.get(f, m['max'])), o['grid'], o['steps'],
                               m['resolution']))
            c['resolution'] = m['resolution']
            c['runs'] = sorted(set(runs_made))
        except FaultTolError as e:
            c = _refused(s, f, str(e))
        cells.append(c)
        if progress:
            progress((n + 1) / len(pairs), f'{s["id"]} · {f}: {c["status"]}')
    return cells


# ======================================================================================================= the result
def margin(cell: dict, requirements: dict | None) -> float | None:
    """tolerated / required for a cell whose fault has a requirement (≥ 1: met); None without one."""
    req = (requirements or {}).get(cell['fault'])
    if not req or cell.get('tolerated') is None:
        return None
    return float(cell['tolerated']) / float(req)


def requirement_status(cell: dict, requirements: dict | None) -> tuple:
    """(word, tone) of a cell against the user's requirement: met / not met / no requirement / not measured."""
    req = (requirements or {}).get(cell['fault'])
    if cell.get('status') == 'refused':
        return 'not measured', ''
    if cell.get('status') == 'fails_nominal':
        return 'fails without the fault', 'critical'
    if not req:
        return 'no requirement', ''
    ok = cell.get('tolerated') is not None and float(cell['tolerated']) >= float(req)
    return ('met', 'good') if ok else ('not met', 'critical')


def most_fragile(cells: list, requirements: dict | None = None) -> dict | None:
    """The sensor with the least room: by tolerated / required where requirements are set (the lowest margin), else
    by tolerated / the largest severity tried (a rough ordering: the faults' units differ)."""
    best = None
    basis = 'requirement' if any(margin(c, requirements) is not None for c in cells) else 'tested range'
    for c in cells:
        if c.get('status') == 'refused':
            continue
        if c.get('status') == 'fails_nominal':
            v = 0.0
        elif basis == 'requirement':
            v = margin(c, requirements)
            if v is None:
                continue
        else:
            v = float(c['tolerated']) / float(c['max'])
        if best is None or v < best['value']:
            best = {'sensor': c['sensor'], 'fault': c['fault'], 'value': round(v, 4), 'basis': basis}
    if best:
        best['about'] = ('tolerated severity ÷ required severity (below 1: the requirement is not met)'
                         if basis == 'requirement' else
                         'tolerated severity ÷ the largest severity tried, a rough ordering across faults of '
                         'different units; set requirements for a meaningful one')
    return best


def calibration_statements(cells: list) -> list:
    """Per sensor, what the tolerances imply for recalibration and synchronisation (DERIVED; module doc)."""
    words = {'cam_yaw': 'extrinsic yaw must hold within ±{v:.2f}°', 'cam_pitch': 'extrinsic pitch within ±{v:.2f}°',
             'cam_roll': 'extrinsic roll within ±{v:.2f}°', 'lidar_yaw': 'extrinsic yaw must hold within ±{v:.2f}°',
             'clock_offset': 'its stamps must be synchronised within {v:.0f} ms',
             'gnss_bias': 'its position bias must stay under {v:.2f} m',
             'dirty_lens': 'the lens must be cleaned before {p:.0f} % of it is covered',
             'lidar_sector': 'a blocked sector ahead must stay narrower than {v:.0f}°',
             'frozen': 'a stream freeze must be detected within {v:.2f} s',
             'dropped': 'frame loss must stay under {p:.0f} %', 'noise': 'image noise must stay under σ {v:.1f}',
             'exposure': 'exposure error must stay under {v:.2f} EV'}
    out = {}
    for c in cells:
        if c.get('status') == 'refused':
            continue
        if c.get('status') == 'fails_nominal':
            text = f'{c["title"].lower()}: the criterion already fails without the fault'
        else:
            v = float(c['tolerated'])
            text = words[c['fault']].format(v=v, p=100.0 * v)
            if c['status'] == 'tolerates_max':
                text += ' (no failure found up to that, the largest tried: the true tolerance is at least this)'
            elif c['status'] == 'not_monotone':
                text += ' (the response was not monotone: the lowest failure a coarse grid found)'
            if c['criterion'] in CLOSED_CRITERIA:
                text += f' [closed loop, {c["criterion"]}]'
        out.setdefault(c['sensor'], []).append(text)
    return [{'sensor': s, 'statement': f'{s}: ' + '; '.join(t), 'derived': True, 'note': STATEMENT_NOTE}
            for s, t in out.items()]


SENS_BAND = 0.10        # the band tried around a requirement the user stated (no document gives a range), 4.2


def sensitivity(cell: dict, requirements: dict | None, rel: float = SENS_BAND) -> dict:
    """Whether a cell's requirement verdict (met / not met) holds at the ends of two bands (4.2, pure: nothing is
    run): the requirement ±rel (the user's number: no document gives it a range) and the tolerance's own bracket
    [last pass, first fail] — where between the grid's points the true tolerance lies. A cell with no requirement,
    not measured, or failing without the fault has nothing to re-evaluate (applicable False, with why).

    Returns {'fault', 'sensor', 'unit', 'required', 'tolerated', 'ends': [{'what', 'tolerated', 'required',
    'verdict'}], 'nominal', 'flips', 'applicable', 'why'}."""
    req = (requirements or {}).get(cell.get('fault'))
    out = {'fault': cell.get('fault'), 'sensor': cell.get('sensor'), 'unit': cell.get('unit', ''),
           'required': req, 'tolerated': cell.get('tolerated'), 'rel': rel, 'ends': [], 'nominal': None,
           'flips': False, 'applicable': False, 'why': ''}
    if cell.get('status') == 'refused':
        out['why'] = 'not measured: ' + (cell.get('note') or 'refused')
        return out
    if cell.get('status') == 'fails_nominal':
        out['why'] = 'the criterion fails without the fault: no tolerance to compare'
        return out
    if not req:
        out['why'] = 'no requirement set for this fault: nothing to judge against'
        return out
    tol = float(cell['tolerated'])
    b = cell.get('bracket') or [None, None]
    lo_t = float(b[0]) if b[0] is not None else tol
    hi_t = float(b[1]) if b[1] is not None else None          # tolerates_max: no failure found, no upper end

    def word(t, r):
        return 'met' if t >= r else 'not met'
    req = float(req)
    out['applicable'] = True
    out['nominal'] = word(tol, req)
    ends = [(f'requirement −{100 * rel:g} %', tol, req * (1 - rel)),
            (f'requirement +{100 * rel:g} %', tol, req * (1 + rel)),
            ('tolerance at the last pass', lo_t, req)]
    if hi_t is not None:
        ends.append(('tolerance just below the first fail', hi_t, req))
    for what, t, r in ends:
        out['ends'].append({'what': what, 'tolerated': round(t, 6), 'required': round(r, 6), 'verdict': word(t, r)})
    out['flips'] = any(e['verdict'] != out['nominal'] for e in out['ends'])
    out['why'] = ('the verdict changes inside the band: the margin is thinner than the uncertainty' if out['flips']
                  else 'the verdict holds at every end')
    return out


def sensitivity_text(s: dict) -> str:
    """sensitivity() as lines for a tooltip or the command line."""
    if not s['applicable']:
        return f'{s["sensor"]} · {s["fault"]}: {s["why"]}'
    u = s['unit']
    lines = [f'{s["sensor"]} · {s["fault"]}: tolerated {s["tolerated"]:.4g} {u}, required {float(s["required"]):.4g} '
             f'{u} — {s["nominal"]}']
    for e in s['ends']:
        lines.append(f'  {e["what"]}: {e["tolerated"]:.4g} vs {e["required"]:.4g} {u} → {e["verdict"]}'
                     + (' FLIPS' if e['verdict'] != s['nominal'] else ''))
    lines.append(s['why'])
    return '\n'.join(lines)


def flags(doc: dict, requirements: dict | None = None) -> list:
    """What a campaign leaves to act on (4.2), most severe first: a requirement not met, a cell failing without the
    fault, a requirement met inside its ±SENS_BAND (a thin margin: no safety factor is applied), a cell not measured
    (a missing check), a non-monotone response, and the measured faults that have no requirement (no verdict at
    all). Each: {'kind', 'tone', 'sensor', 'fault', 'text', 'field' (what "Jump to config" focuses: requirement,
    scenario or the row), 'sensitivity' (whether a sensitivity check applies)}."""
    reqs = dict(requirements if requirements is not None else doc.get('requirements') or {})
    out, no_req = [], []
    for c in doc.get('cells') or []:
        st, req = c.get('status'), reqs.get(c['fault'])
        base = {'sensor': c['sensor'], 'fault': c['fault']}
        what = f'{c["sensor"]} · {c["title"].lower()}'
        if st == 'refused':
            out.append(dict(base, kind='missing', tone='warning', field='row', sensitivity=False,
                            text=f'{what}: not measured — {c.get("note") or "refused"}'))
            continue
        if st == 'fails_nominal':
            out.append(dict(base, kind='fails_nominal', tone='critical', field='scenario', sensitivity=False,
                            text=f'{what}: the criterion fails without the fault (the scenario or the criterion, '
                                 'not the sensor)'))
            continue
        if st == 'not_monotone':
            out.append(dict(base, kind='not_monotone', tone='warning', field='row', sensitivity=False,
                            text=f'{what}: the response is not monotone; the tolerance shown is the lowest failure '
                                 'the grid found'))
        if not req:
            no_req.append(c)
            continue
        m = margin(c, reqs)
        if m is not None and m < 1.0:
            out.append(dict(base, kind='not_met', tone='critical', field='requirement', sensitivity=True,
                            text=f'{what}: tolerates {float(c["tolerated"]):.3g} {c["unit"]}, the requirement is '
                                 f'{float(req):.3g} {c["unit"]} — not met'))
        elif m is not None and m < 1.0 + SENS_BAND:
            out.append(dict(base, kind='thin_margin', tone='warning', field='requirement', sensitivity=True,
                            text=f'{what}: met with {100 * (m - 1):.0f} % to spare, inside ±{100 * SENS_BAND:g} % — '
                                 'no safety factor is applied'))
    if no_req:
        c = no_req[0]
        out.append({'kind': 'no_requirement', 'tone': 'warning', 'sensor': c['sensor'], 'fault': c['fault'],
                    'field': 'requirement', 'sensitivity': False,
                    'text': f'{len(no_req)} measured cell(s) have no requirement, so no verdict: '
                            + ', '.join(sorted({x['fault'] for x in no_req}))})
    rank = {'critical': 0, 'warning': 1}
    return sorted(out, key=lambda f: rank.get(f['tone'], 2))


def campaign_fidelity(cells, out_dir: str = '', runs=None) -> tuple:
    """(fidelity, source fidelity) of a campaign (core.fidelity, 4.2): an offline one is `recorded` with its runs'
    combined level beside it; a closed-loop one is the combined level of its stack runs (out_dir/runs/…)."""
    from . import fidelity as F
    if runs is not None:
        return 'recorded', F.combine(F.of_run(r) for r in runs)
    paths = sorted({rr for c in cells or [] for rr in c.get('runs') or []})
    return F.combine(F.of_run(os.path.join(out_dir, rr)) for rr in paths), ''


def result(cells: list, mode: str, name: str = '', seed: int = 0, requirements: dict | None = None,
           inputs: dict | None = None, fidelity: str = 'unknown', source_fidelity: str = '') -> dict:
    from .provenance import utc_now
    tol = {}
    for c in cells:
        tol.setdefault(c['sensor'], {})[c['fault']] = {
            'tolerated': c.get('tolerated'), 'bracket': c.get('bracket'), 'unit': c['unit'], 'status': c['status']}
    return {'schema': SCHEMA, 'name': name, 'mode': mode, 'created': utc_now(), 'seed': int(seed),
            'inputs': inputs or {}, 'requirements': dict(requirements or {}), 'cells': cells, 'tolerance': tol,
            'fidelity': fidelity, **({'source_fidelity': source_fidelity} if source_fidelity else {}),
            'most_fragile': most_fragile(cells, requirements), 'calibration': calibration_statements(cells),
            'method': 'per cell: the response at 0 and on a coarse grid, a monotonicity check of the verdicts, then '
                      'bisection of the bracket; common random numbers across severities (core.faulttol)',
            'limits': 'tolerances of this stack on these runs or scenarios under these fault models; one fault at '
                      'a time (no combinations); simulated sensors'}


def table_rows(doc: dict) -> list:
    rows = []
    for c in doc.get('cells') or []:
        b = c.get('bracket') or [None, None]
        word, _tone = requirement_status(c, doc.get('requirements'))
        rows.append({'sensor': c['sensor'], 'fault': c['fault'], 'unit': c['unit'], 'tolerated': c.get('tolerated'),
                     'last_pass': b[0], 'first_fail': b[1], 'status': c['status'],
                     'criterion': c.get('criterion', ''), 'threshold': c.get('threshold', ''),
                     'monotone': c.get('monotone'), 'required': (doc.get('requirements') or {}).get(c['fault'], ''),
                     'requirement': word, 'note': c.get('note', '')})
    return rows


def _g(v, nd=4):
    return '—' if v is None else f'{float(v):.{nd}g}'


def table_text(doc: dict) -> str:
    lines = [f'{"sensor":12s} {"fault":13s} {"tolerated":>10s} {"bracket [pass, fail]":>24s}  {"unit":10s} status']
    for r in table_rows(doc):
        lines.append(f'{r["sensor"][:12]:12s} {r["fault"]:13s} {_g(r["tolerated"]):>10s} '
                     f'{"[" + _g(r["last_pass"]) + ", " + _g(r["first_fail"]) + "]":>24s}  {r["unit"][:10]:10s} '
                     f'{r["status"].replace("_", " ")}' + (f' · {r["requirement"]}' if r['required'] else '') +
                     (f' — {r["note"]}' if r['status'] in ('refused', 'not_monotone', 'fails_nominal') else ''))
    mf = doc.get('most_fragile')
    if mf:
        lines.append(f'most fragile: {mf["sensor"]} ({mf["fault"]}; {mf["value"]:g} of the {mf["basis"]})')
    for s in doc.get('calibration') or []:
        lines.append(f'derived — {s["statement"]}')
    if doc.get('calibration'):
        lines.append(f'({STATEMENT_NOTE})')
    return '\n'.join(lines)


def write(doc: dict, out_dir: str) -> str:
    from .provenance import write_json
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'faulttol.json')
    write_json(p, doc)
    rows = table_rows(doc)
    with open(os.path.join(out_dir, 'tolerance.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ['sensor'])
        w.writeheader()
        w.writerows(rows)
    return p


def load(path: str) -> dict:
    p = path if path.endswith('.json') else os.path.join(path, 'faulttol.json')
    with open(p, encoding='utf-8') as fh:
        doc = json.load(fh)
    if doc.get('schema') != SCHEMA:
        raise FaultTolError(f'{p} is not a fault tolerance result')
    return doc


def campaign_dir(workspace: str, name: str) -> str:
    return os.path.join(workspace, 'faulttol', name)


def campaigns(workspace: str) -> list:
    d = os.path.join(workspace, 'faulttol')
    return sorted(n for n in (os.listdir(d) if os.path.isdir(d) else [])
                  if os.path.isfile(os.path.join(d, n, 'faulttol.json')))


def parse_requirements(items) -> dict:
    """['cam_yaw=0.5', …] -> {fault: required severity}."""
    out = {}
    for it in items or []:
        k, _, v = str(it).partition('=')
        if k not in MODELS:
            raise FaultTolError(f'--require {it}: unknown fault {k!r}')
        try:
            out[k] = float(v)
        except ValueError:
            raise FaultTolError(f'--require {it}: FAULT=SEVERITY') from None
        if out[k] <= 0:
            raise FaultTolError(f'--require {it}: the severity must be positive')
    return out


# ============================================================================================================= CLI
def main_faulttol(argv) -> int:
    """`falcon faulttol models|run|show …` (module doc): 0 ok, 2 failed, 6 refused."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon faulttol', description='sensor fault and calibration-drift tolerance '
                                 'campaigns (docs/fault-tolerance.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    m = sub.add_parser('models', help='the graded fault models: unit, range, what is modelled')
    m.add_argument('--json', action='store_true')
    r = sub.add_parser('run', help='a campaign: offline on recorded runs (--runs), or closed loop (--closed-loop)')
    r.add_argument('project')
    r.add_argument('--name', required=True, help='the campaign\'s folder under <workspace>/faulttol/')
    r.add_argument('--runs', nargs='*', default=[], help='recorded run folders (offline mode)')
    r.add_argument('--model', default='', help='a registered detector: the camera criterion becomes its AP drop')
    r.add_argument('--sensor', action='append', default=[], help='a sensor id (repeatable; default: all)')
    r.add_argument('--fault', action='append', default=[], help='a fault model (repeatable; default: all that apply)')
    r.add_argument('--require', action='append', default=[], metavar='FAULT=SEVERITY',
                   help='a requirement the table is judged against (repeatable)')
    r.add_argument('--ap-drop', type=float, default=DEFAULTS['ap_drop_pt'], help='AP points lost that count as a fail')
    r.add_argument('--ap-metric', default=DEFAULTS['ap_metric'], choices=('AP', 'AP50', 'AP75'))
    r.add_argument('--recall-drop', type=float, default=DEFAULTS['recall_drop_pt'],
                   help='points of objects displaced beyond IoU 0.5 that count as a fail (no model)')
    r.add_argument('--retention-drop', type=float, default=DEFAULTS['retention_drop_pt'],
                   help='points of lidar actor returns lost or displaced that count as a fail')
    r.add_argument('--lidar-tol', type=float, default=DEFAULTS['lidar_tol_m'], help='metres a lidar point may move')
    r.add_argument('--max-frames', type=int, default=DEFAULTS['max_frames'], help='frames scored per sensor')
    r.add_argument('--grid', type=int, default=DEFAULTS['grid'], help='coarse grid points per cell')
    r.add_argument('--steps', type=int, default=DEFAULTS['steps'], help='bisection steps per cell')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--closed-loop', action='store_true', help='stack runs through the simulator instead of --runs')
    r.add_argument('--scenario', default='', help='closed loop: the scenario (a built-in key or a project id)')
    r.add_argument('--criterion', default='degraded', choices=CLOSED_CRITERIA)
    r.add_argument('--threshold', type=float, default=None, help='closed loop: the response above which a cell fails')
    r.add_argument('--stl', default='', help='closed loop, criterion stl: the formula')
    r.add_argument('--seeds', default='0', help='closed loop: scenario seeds, comma-separated (the same at every '
                                                'severity)')
    r.add_argument('--driver', default='follower', choices=('follower', 'autopilot'))
    r.add_argument('--no-degraded-mode', action='store_true', help='the follower without its health monitor')
    r.add_argument('--onset', type=float, default=DEFAULTS['onset_s'], help='closed loop: the fault\'s onset (s)')
    r.add_argument('--world', default='')
    r.add_argument('--rig', default='')
    r.add_argument('--fps', type=float, default=20.0)
    r.add_argument('--fake-carla', default='')
    s = sub.add_parser('show', help='a campaign\'s tolerance table: falcon faulttol show DIR')
    s.add_argument('campaign')
    s.add_argument('--json', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if a.action == 'models':
        if a.json:
            print(json.dumps(MODELS, indent=2))
        else:
            for k, v in MODELS.items():
                print(f'{k:13s} 0–{v["max"]:g} {v["unit"]:17s} {", ".join(v["sensors"])}: {v["model"]}'
                      + ('' if k in CLOSED else '  [offline only]'))
        return P.OK
    if a.action == 'show':
        try:
            doc = load(a.campaign)
        except (OSError, ValueError, FaultTolError) as e:
            print(f'failed: {e}', flush=True)
            return P.FAILED
        print(json.dumps(doc, indent=2) if a.json else table_text(doc), flush=True)
        return P.OK
    return _run(a)


def _run(a) -> int:
    from . import protocol as P
    from .project import Project, ProjectError
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    em = P.Emitter()
    opts = {'ap_drop_pt': a.ap_drop, 'ap_metric': a.ap_metric, 'recall_drop_pt': a.recall_drop,
            'retention_drop_pt': a.retention_drop, 'lidar_tol_m': a.lidar_tol, 'max_frames': a.max_frames,
            'grid': a.grid, 'steps': a.steps, 'onset_s': a.onset}
    out = campaign_dir(proj.workspace, a.name)
    svc = None
    try:
        req = parse_requirements(a.require)
        if a.closed_loop == bool(a.runs):
            raise FaultTolError('give recorded runs (--runs) or --closed-loop with --scenario, not both or neither')

        def progress(frac, text):
            em.progress(frac, text, force=True)
        em.hello('faulttol')
        if a.closed_loop:
            from . import scenarios as SC
            from ..sim.service import study_service
            scn = proj.item('scenarios', a.scenario) or (SC.builtin(a.scenario) if a.scenario in SC.LIBRARY else None)
            if scn is None:
                raise FaultTolError(f'no scenario {a.scenario!r}')
            spec = None
            if a.criterion == 'stl':
                from . import stlobjective as SO
                try:
                    spec = SO.normalise({'stl': a.stl or None})
                except SO.STLObjectiveError as e:
                    raise FaultTolError(str(e)) from None
            world = a.world or proj.doc['worlds'][0]['id']
            rig = a.rig or proj.doc['rigs'][0]['id']
            seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
            svc = study_service(None, proj.workspace, fake_carla=a.fake_carla)
            cells = closed_campaign(svc, proj, scn, out, world, rig, a.sensor, a.fault, a.criterion, a.threshold,
                                    spec, seeds, None, a.driver, not a.no_degraded_mode, a.fps, opts, progress)
            inputs = {'scenario': scn.get('id') or a.scenario, 'world': world, 'rig': rig, 'seeds': seeds,
                      'driver': a.driver, 'degraded_mode': not a.no_degraded_mode, 'criterion': a.criterion,
                      'backend': svc.backend}
            mode = 'closed_loop'
        else:
            model = None
            if a.model:
                model = proj.item('models', a.model)
                if model is None:
                    raise FaultTolError(f'no model {a.model!r} in the project')
            cells = offline_campaign(a.runs, a.sensor, a.fault, model, proj.workspace, a.seed, opts, progress, em)
            inputs = {'runs': [os.path.relpath(os.path.abspath(x), proj.workspace) for x in a.runs],
                      'model': a.model or None, 'options': opts}
            mode = 'offline'
        fid, src = campaign_fidelity(cells, out) if mode == 'closed_loop' else campaign_fidelity(cells, runs=a.runs)
        doc = result(cells, mode, a.name, a.seed, req, inputs, fid, src)
        path = write(doc, out)
    except FaultTolError as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    except (RuntimeError, OSError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if svc is not None:
            svc.shutdown()
    for line in table_text(doc).splitlines():
        em.log(line)
    mf = doc.get('most_fragile') or {}
    em.result('ok', summary={'campaign': a.name, 'cells': len(cells), 'path': path,
                             'refused': sum(c['status'] == 'refused' for c in cells),
                             'most_fragile': mf.get('sensor'), 'fault': mf.get('fault')})
    return P.OK
