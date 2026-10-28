"""Render budget and cost estimates (standard library only).

Render budget. What a CARLA camera costs is the pixels it renders. A pinhole camera renders its image once, plus
once per label camera beside it (semantic, instance, depth: CARLA renders each as its own camera at the same pose and
tick). A fisheye renders the pinhole views core.lens plans for it — the cube faces its rays need (five for a 190°
lens, one when every ray is within 60° of its axis), each N = 2·⌈f·tan(45° + 1.5°) + 1.5⌉ pixels on a side for a
centre focal length f — and remaps them. On the reference machine a 0.9.16 server capped at 12 GB crashed at 306 Mpx
rendered per step and ran at 77: runs over RENDER_BUDGET_MPX are refused before they start.

Cost. Seconds per simulation step = overhead + rendered Mpx / throughput. The throughput is measured on this
machine (the first-run smoke drive, and every drive after it, updates it); until then the reference machine's figure
is used and labelled typical.
"""
from __future__ import annotations

import math

RENDER_BUDGET_MPX = 200.0
LABEL_CAMERAS = {'semantic': 'semantic_segmentation', 'instance': 'instance_segmentation', 'depth': 'depth',
                 'boxes2d': 'instance_segmentation'}      # 2D tight boxes come from the instance image

# The reference machine (RTX 5080 16 GB, CARLA 0.9.16 capped at 12 GB): six fisheyes at half size (960 × 768) with
# instance and depth ground truth, 77 Mpx rendered per step: 602 labelled camera frames (100 steps) in 114 s.
TYPICAL = {
    'mpx_per_s': 77.0 * 602 / 6 / 114.0,
    'overhead_s': 0.05,
    'source': 'typical: the reference machine (RTX 5080), six half-size fisheyes with ground truth, 602 labelled '
              'frames in 114 s; not measured on this machine yet',
    'measured': False,
}


def label_cameras(labels) -> list:
    """The CARLA label cameras a set of label kinds needs (each rendered once)."""
    return sorted({LABEL_CAMERAS[k] for k in labels or () if k in LABEL_CAMERAS})


def fisheye_focal_px(s: dict) -> float:
    """A fisheye's focal length at its centre, px/rad: the lens model's (f_px) when the sensor has one, else
    (heuristic) an equidistant lens whose image circle fits the shorter image side."""
    if s.get('f_px'):
        return float(s['f_px'])
    w, h = int(s.get('width', 0)), int(s.get('height', 0))
    return (min(w, h) / 2.0) / math.radians(max(float(s.get('fov', 190.0)), 1.0) / 2.0)


def sensor_mpx(s: dict) -> float:
    """Megapixels a sensor renders per step (image + label cameras); 0 for lidar, radar, GNSS and IMU, which
    CARLA ray-casts or computes rather than renders."""
    t = s.get('type', '')
    if not t.startswith('camera.'):
        return 0.0
    n = 1 + len(label_cameras(s.get('labels')))
    w, h = int(s.get('width', 0)), int(s.get('height', 0))
    if t == 'camera.fisheye':
        # the pinhole views it is planned with (core.lens: the cube faces its rays need, sized to its lens), so the
        # budget and the renderer never disagree
        from . import lens
        return lens.camera_mpx(s)
    return w * h * n / 1e6


def rig_mpx(rig: dict) -> float:
    return sum(sensor_mpx(s) for s in rig.get('sensors', []))


def check_budget(rig: dict, budget: float = RENDER_BUDGET_MPX) -> tuple[bool, str]:
    mpx = rig_mpx(rig)
    if budget > 0 and mpx > budget:
        return False, (f'the cameras would render {mpx:.0f} Mpx per step (images and label cameras), over the budget '
                       f'of {budget:g} Mpx (a 0.9.16 server capped at 12 GB crashed at 306 on the reference '
                       'machine): lower image sizes or label layers, or raise the budget in Settings for a bigger '
                       'server')
    return True, f'{mpx:.1f} Mpx per step of a {budget:g} Mpx budget'


class Throughput:
    """Seconds per step = overhead_s + Mpx / mpx_per_s.

    Fitted to the drives measured on this machine (points: (Mpx per step, seconds per step, source)): with points at
    two or more clearly different rendering loads, a least-squares line (overhead ≥ 0); with one load only, no
    overhead can be told apart, so all of the time is put on rendering. Without any, TYPICAL. An estimate for a load
    far outside the measured ones (more than twice the largest, or under half the smallest) is labelled extrapolated.
    """

    MAX_POINTS = 30

    def __init__(self, mpx_per_s: float, overhead_s: float, source: str, measured: bool, points=()):
        self.mpx_per_s, self.overhead_s = float(mpx_per_s), float(overhead_s)
        self.source, self.measured = source, bool(measured)
        self.points = [tuple(p) for p in points]

    @classmethod
    def from_settings(cls, measured: dict | None) -> 'Throughput':
        if measured and measured.get('points'):
            return cls.fit(measured['points'])
        d = TYPICAL
        return cls(d['mpx_per_s'], d['overhead_s'], d['source'], False)

    @classmethod
    def fit(cls, points) -> 'Throughput':
        pts = [tuple(p) for p in points][-cls.MAX_POINTS:]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        n = len(pts)
        latest = pts[-1][2] if len(pts[-1]) > 2 else ''
        src = f'measured on this machine ({n} drive{"s" if n != 1 else ""}; latest: {latest})'
        if n >= 2 and max(x) > 1.2 * max(min(x), 1e-6):
            mx, my = sum(x) / n, sum(y) / n
            sxx = sum((xi - mx) ** 2 for xi in x)
            b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
            a = my - b * mx
            if b > 0 and a >= 0:
                return cls(1.0 / b, a, src, True, pts)
        # one load (or a fit that makes no sense): all of the time on rendering, at the mean measured rate
        rate = sum(xi for xi in x) / max(sum(y), 1e-9)
        return cls(max(rate, 1e-3), 0.0, src, True, pts)

    def add(self, mpx_per_step: float, seconds_per_step: float, source: str) -> 'Throughput':
        prev = self.points if self.measured else []
        return Throughput.fit(prev + [(float(mpx_per_step), float(seconds_per_step), source)])

    def to_dict(self) -> dict:
        return {'mpx_per_s': self.mpx_per_s, 'overhead_s': self.overhead_s, 'source': self.source,
                'measured': self.measured, 'points': [list(p) for p in self.points]}

    def seconds_per_step(self, mpx: float) -> float:
        return self.overhead_s + mpx / max(self.mpx_per_s, 1e-6)

    def label_for(self, mpx: float) -> str:
        if not self.measured:
            return 'typical'
        xs = [p[0] for p in self.points]
        if xs and (mpx > 2.0 * max(xs) or mpx < 0.5 * min(xs)):
            return 'extrapolated from measured'
        return 'measured'

    def estimate(self, steps: int, mpx: float, setup_s: float = 20.0) -> dict:
        """{'seconds', 'per_step_s', 'source', 'label'}: label is 'measured', 'extrapolated from measured' or
        'typical'."""
        per = self.seconds_per_step(mpx)
        return {'seconds': setup_s + steps * per, 'per_step_s': per, 'source': self.source,
                'label': self.label_for(mpx)}


def fmt_duration(s: float) -> str:
    s = max(float(s), 0.0)
    if s < 90:
        return f'{s:.0f} s'
    if s < 5400:
        return f'{s / 60:.0f} min'
    return f'{s / 3600:.1f} h'
