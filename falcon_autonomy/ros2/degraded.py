"""Degraded mode of Falcon's test stack: watch the sensors' health from their data, and on a failure carry out a
minimal-risk manoeuvre — a jerk-limited stop in the lane, held (numpy only; part of Falcon Autonomy,
GPL-3.0-or-later). Imported by follower_stack.py inside the ROS 2 bridge image (this folder is mounted there) and by
the stack worker's built-in follower (workers.stack), so both react by the same rules.

Detection (Monitor; every threshold heuristic, chosen for Falcon's simulated sensors, not tuned on a real stack):

* lidar: no sweep for lidar_timeout_s of simulation time (packets lost, the sensor dead); a sweep with fewer than
  lidar_drop_ratio × the running median point count of the last sweeps (packet drop); or an empty azimuth sector
  wider than lidar_sector_deg in a sweep, where the previous sweeps had points (sector loss);
* IMU: no message for imu_timeout_s;
* GNSS: no fix for gnss_timeout_s, or a fix that moves more than gnss_jump_m further than the odometry did since the
  previous one (a multipath jump);
* camera: an image whose share of dark, texture-less 8 × 8 blocks (mean < camera_dark_level, std < camera_flat_std)
  exceeds camera_occluded_share (mud, an occluding film), or camera_frozen_frames identical images in a row while
  the vehicle moves (a standing vehicle in a still scene can see identical images; the fake carla, flat-shaded at
  64 × 36, gave 5 identical images in a row even moving — hence 10).

Detection latches: once degraded, the stack stays degraded for the run (it has no recovery logic).

Reaction (MRM): the acceleration demand ramps from the current one down to −mrm_decel at mrm_jerk (both heuristic:
a controlled stop, not emergency braking), the lane-keeping steering continues, and at standstill the vehicle holds
(full brake). The caller turns the demand into its own control (a throttle / brake pair or an acceleration command).
"""
from __future__ import annotations

import math

import numpy as np

DEFAULTS = {
    'lidar_timeout_s': 0.35, 'lidar_drop_ratio': 0.5, 'lidar_sector_deg': 45.0, 'lidar_history': 5,
    'imu_timeout_s': 0.25, 'gnss_timeout_s': 0.5, 'gnss_jump_m': 4.0,
    'camera_dark_level': 70.0, 'camera_flat_std': 6.0, 'camera_occluded_share': 0.2, 'camera_frozen_frames': 10,
    'mrm_decel': 2.5, 'mrm_jerk': 4.0, 'stopped_mps': 0.1,
}


class Monitor:
    """Sensor health from the data a stack receives (module doc). Feed it what arrives; check(t) says whether the
    stack is degraded and why (latched)."""

    def __init__(self, params: dict | None = None, expect=('lidar',)):
        self.p = dict(DEFAULTS, **(params or {}))
        self.expect = set(expect)              # the sensors whose silence counts ('lidar', 'imu', 'gnss', 'camera')
        self.last = {}                         # kind -> simulation time of the last message
        self.counts = []
        self.sectors = None
        self.gnss_prev = None
        self.frozen = 0
        self.prev_img = None
        self.degraded = None                   # (t, why) once degraded

    def _fail(self, t, why):
        if self.degraded is None:
            self.degraded = (float(t), why)

    def lidar(self, t: float, pts) -> None:
        """A full sweep: points (N × ≥ 2; x, y in the sensor frame)."""
        P = np.asarray(pts, float)
        P = P.reshape(-1, P.shape[-1]) if P.ndim > 1 else np.zeros((0, 3))
        self.last['lidar'] = float(t)
        n = len(P)
        hist = self.counts[-int(self.p['lidar_history']):]
        if len(hist) >= 3 and n < self.p['lidar_drop_ratio'] * float(np.median(hist)):
            self._fail(t, f'lidar: {n} points, below {self.p["lidar_drop_ratio"]:g} × the recent median '
                          f'{int(np.median(hist))}')
        self.counts.append(n)
        occ = np.zeros(360, bool)
        if n:
            az = (np.degrees(np.arctan2(P[:, 1], P[:, 0])) + 360.0) % 360.0
            occ[np.floor(az).astype(int) % 360] = True
        recent = self.sectors or []
        if len(recent) >= 3:
            had = np.sum(recent[-3:], axis=0) >= 2           # degrees with points in 2 of the last 3 sweeps
            run = _widest_gap(had, occ)
            if run > self.p['lidar_sector_deg']:
                self._fail(t, f'lidar: an empty sector of {run}° where earlier sweeps had points')
        self.sectors = (recent + [occ])[-int(self.p['lidar_history']):]

    def imu(self, t: float) -> None:
        self.last['imu'] = float(t)

    def gnss(self, t: float, xy_m, odom_xy_m) -> None:
        """A fix as local metres (x, y), with the odometry's position at the same time (same frame up to a
        constant offset: only displacements are compared)."""
        self.last['gnss'] = float(t)
        cur = (np.asarray(xy_m, float), np.asarray(odom_xy_m, float))
        if self.gnss_prev is not None:
            moved = float(np.linalg.norm(cur[0] - self.gnss_prev[0]))
            odom = float(np.linalg.norm(cur[1] - self.gnss_prev[1]))
            if moved - odom > self.p['gnss_jump_m']:
                self._fail(t, f'gnss: the fix jumped {moved:.1f} m while the odometry moved {odom:.1f} m')
        self.gnss_prev = cur

    def camera(self, t: float, img, moving: bool = True) -> None:
        """An image (H × W × 3 uint8); moving: whether the vehicle moves (frozen images count only then)."""
        self.last['camera'] = float(t)
        a = np.asarray(img)
        g = a.mean(axis=2) if a.ndim == 3 else a.astype(float)
        H, W = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
        if H and W:
            b = g[:H, :W].reshape(H // 8, 8, W // 8, 8)
            mean, std = b.mean(axis=(1, 3)), b.std(axis=(1, 3))
            share = float(((mean < self.p['camera_dark_level']) & (std < self.p['camera_flat_std'])).mean())
            if share > self.p['camera_occluded_share']:
                self._fail(t, f'camera: {share:.0%} of the image dark and texture-less (occluded)')
        if moving and self.prev_img is not None and self.prev_img.shape == a.shape and \
                np.array_equal(self.prev_img, a):
            self.frozen += 1
            if self.frozen >= self.p['camera_frozen_frames']:
                self._fail(t, f'camera: {self.frozen + 1} identical images (frozen)')
        else:
            self.frozen = 0
        self.prev_img = a.copy()

    def check(self, t: float):
        """(degraded: bool, why or None). Timeouts are judged here, from the last message of each expected sensor."""
        for kind, key in (('lidar', 'lidar_timeout_s'), ('imu', 'imu_timeout_s'), ('gnss', 'gnss_timeout_s'),
                          ('camera', 'lidar_timeout_s')):
            if kind in self.expect and kind in self.last and t - self.last[kind] > self.p[key] + 1e-9:
                self._fail(t, f'{kind}: nothing for {t - self.last[kind]:.2f} s')
        return (self.degraded is not None, self.degraded[1] if self.degraded else None)


def _widest_gap(had: np.ndarray, occ: np.ndarray) -> int:
    """The widest azimuth span (degrees) between two degrees that had points and still have them, with at least one
    degree inside it that had points and has none now (degrees that never had points — a sparse lidar's own gaps
    between its columns — neither break nor make a gap)."""
    kept = np.flatnonzero(had & occ)
    lost = had & ~occ
    if not lost.any():
        return 0
    if not len(kept):
        return 360
    best = 0
    for a, b in zip(kept, np.r_[kept[1:], kept[0] + 360]):
        span = int(b - a - 1)
        if span > best and lost[np.arange(a + 1, b) % 360].any():
            best = span
    return best


class MRM:
    """The minimal-risk manoeuvre's longitudinal demand: from the acceleration in effect at its start, down to
    −mrm_decel at mrm_jerk, held until standstill; then hold (brake)."""

    def __init__(self, params: dict | None = None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.t0 = None
        self.a0 = 0.0

    def accel(self, t: float, v: float, a_now: float = 0.0) -> tuple:
        """(acceleration demand m/s², hold: bool)."""
        if self.t0 is None:
            self.t0, self.a0 = float(t), min(float(a_now), 0.0)
        if v <= self.p['stopped_mps']:
            return 0.0, True
        a = max(self.a0 - self.p['mrm_jerk'] * (t - self.t0), -self.p['mrm_decel'])
        return a, False


def throttle_brake(a: float, hold: bool, accel_full: float = 3.0, brake_full: float = 8.0) -> dict:
    """An acceleration demand as Falcon's kinematic model's pedals (throttle a / 3 m/s², brake −a / 8 m/s²: the
    director's ACCEL and BRAKE), or a full stop held."""
    if hold:
        return {'throttle': 0.0, 'brake': 1.0, 'hand_brake': True}
    if a >= 0:
        return {'throttle': min(a / accel_full, 1.0), 'brake': 0.0}
    return {'throttle': 0.0, 'brake': min(-a / brake_full, 1.0)}


def mrm_steps(v0: float, params: dict | None = None, dt: float = 0.05) -> float:
    """The time (s) the MRM takes from v0 to standstill (for planning the verifier's window; ramp included)."""
    p = dict(DEFAULTS, **(params or {}))
    ramp = p['mrm_decel'] / p['mrm_jerk']
    dv_ramp = 0.5 * p['mrm_decel'] * ramp
    if v0 <= dv_ramp:
        return math.sqrt(2.0 * v0 / p['mrm_jerk'])
    return ramp + (v0 - dv_ramp) / p['mrm_decel']
