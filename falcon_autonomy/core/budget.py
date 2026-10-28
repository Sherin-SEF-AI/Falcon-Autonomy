"""Render budget and cost estimates (standard library only).

Render budget. What a CARLA camera costs is the pixels it renders. A pinhole camera renders its image once, plus
once per label camera beside it (semantic, instance, depth: CARLA renders each as its own camera at the same pose and
tick). A fisheye renders FISHEYE_VIEWS pinhole views (a cube's faces; their size from fisheye_face_px) and remaps
them, so it costs about five pinholes. On the reference machine a 0.9.16 server capped at 12 GB crashed at 306 Mpx
rendered per step and ran at 77: runs over RENDER_BUDGET_MPX are refused before they start.

Cost. Seconds per simulation step = overhead + rendered Mpx / throughput. The throughput is measured on this
machine (the first-run smoke drive, and every drive after it, updates it); until then the reference machine's figure
is used and labelled typical.
"""
from __future__ import annotations

import math

RENDER_BUDGET_MPX = 200.0
FISHEYE_VIEWS = 5
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


SEAM_BLEND_DEG = 1.5            # cube faces overlap this far past their seams (feathered there)
FACE_MARGIN_PX = 1.5


def fisheye_focal_px(s: dict) -> float:
    """A fisheye's focal length at its centre, px/rad: the lens model's (f_px) when the sensor has one, else
    (heuristic) an equidistant lens whose image circle fits the shorter image side."""
    if s.get('f_px'):
        return float(s['f_px'])
    w, h = int(s.get('width', 0)), int(s.get('height', 0))
    return (min(w, h) / 2.0) / math.radians(max(float(s.get('fov', 190.0)), 1.0) / 2.0)


def fisheye_face_px(f_px: float) -> int:
    """Cube-face size that samples a fisheye's centre at its own density: a face reaches half the seam band past each
    seam, and its centre density (N/2 − margin) / tan(45° + band/2) is at least f (Vantage's sizing; six 1920 × 1536
    fisheyes at f = 900 px/rad with instance and depth come to the 306 Mpx that crashed a 12 GB server)."""
    t = math.tan(math.radians(45.0 + SEAM_BLEND_DEG / 2.0))
    return 2 * int(math.ceil(f_px * t + FACE_MARGIN_PX))


def sensor_mpx(s: dict) -> float:
    """Megapixels a sensor renders per step (image + label cameras); 0 for lidar, radar, GNSS and IMU, which
    CARLA ray-casts or computes rather than renders."""
    t = s.get('type', '')
    if not t.startswith('camera.'):
        return 0.0
    n = 1 + len(label_cameras(s.get('labels')))
    w, h = int(s.get('width', 0)), int(s.get('height', 0))
    if t == 'camera.fisheye':
        f = fisheye_face_px(fisheye_focal_px(s))
        return FISHEYE_VIEWS * f * f * n / 1e6
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
    """Seconds per step = overhead_s + Mpx / mpx_per_s, from a measurement on this machine or TYPICAL."""

    def __init__(self, mpx_per_s: float, overhead_s: float, source: str, measured: bool):
        self.mpx_per_s, self.overhead_s = float(mpx_per_s), float(overhead_s)
        self.source, self.measured = source, bool(measured)

    @classmethod
    def from_settings(cls, measured: dict | None) -> 'Throughput':
        d = measured if measured and measured.get('mpx_per_s') else TYPICAL
        return cls(d['mpx_per_s'], d.get('overhead_s', TYPICAL['overhead_s']), d.get('source', ''),
                   d.get('measured', False))

    @classmethod
    def measure(cls, steps: int, wall_s: float, mpx_per_step: float, source: str,
                previous: 'Throughput | None' = None) -> 'Throughput':
        """From a run: steps simulated in wall_s at mpx_per_step. With little rendering the run says more about the
        overhead than the throughput, so the previous throughput is kept and only the overhead updated."""
        per = wall_s / max(steps, 1)
        prev = previous or cls.from_settings(None)
        if mpx_per_step < 1.0:
            return cls(prev.mpx_per_s, max(per - mpx_per_step / prev.mpx_per_s, 0.0), source, True)
        over = min(prev.overhead_s, per * 0.5)
        return cls(mpx_per_step / max(per - over, 1e-3), over, source, True)

    def to_dict(self) -> dict:
        return {'mpx_per_s': self.mpx_per_s, 'overhead_s': self.overhead_s, 'source': self.source,
                'measured': self.measured}

    def seconds_per_step(self, mpx: float) -> float:
        return self.overhead_s + mpx / max(self.mpx_per_s, 1e-6)

    def estimate(self, steps: int, mpx: float, setup_s: float = 20.0) -> dict:
        """{'seconds', 'per_step_s', 'source', 'label'}: label is 'measured' or 'typical'."""
        per = self.seconds_per_step(mpx)
        return {'seconds': setup_s + steps * per, 'per_step_s': per, 'source': self.source,
                'label': 'measured' if self.measured else 'typical'}


def fmt_duration(s: float) -> str:
    s = max(float(s), 0.0)
    if s < 90:
        return f'{s:.0f} s'
    if s < 5400:
        return f'{s / 60:.0f} min'
    return f'{s / 3600:.1f} h'
