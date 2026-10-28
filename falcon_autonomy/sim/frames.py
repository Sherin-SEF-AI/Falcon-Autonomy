"""Frames: Vantage's vehicle frame (ISO 8855) and CARLA's (Unreal Engine 4's), and the conversions between them.

Vantage: right-handed, x forward, y left, z up, metres; a pose's rotation is intrinsic Z-Y-X yaw / pitch / roll in
degrees with pitch positive *down* (core.mathutil.rot_zyx: R = Rz(yaw)·Ry(pitch)·Rx(roll), standard right-handed
rotations); a sensor's own frame is x forward, y left, z up (REP 103); a camera's optical frame x right, y down, z
along the view.

CARLA 0.9.16: left-handed, x forward, y right, z up, metres. carla.Rotation(pitch, yaw, roll) in degrees: yaw about z
(positive turns x towards +y, to the right seen from above), pitch about y (positive raises the nose), roll about x
(positive lowers the right side). Its matrix (LibCarla geom/Transform.h, Transform::GetMatrix, the same as
Rotation::RotateVector) maps a point in the actor's own axes into the parent's:

    [[cp·cy, cy·sp·sr − sy·cr, −cy·sp·cr − sy·sr],
     [cp·sy, sy·sp·sr + cy·cr, −sy·sp·cr + cy·sr],
     [sp,    −cp·sr,            cp·cr          ]]

(c/s: cos/sin of p = pitch, y = yaw, r = roll). Read as numbers on CARLA's axes it is Rz(yaw)·Ry(−pitch)·Rx(−roll).
A point maps between the frames by mirroring y: p_carla = S·p_vantage with S = diag(1, −1, 1), and a rotation by
conjugation, R_carla = S·R_vantage·S. Since S·Rz(a)·S = Rz(−a), S·Ry(a)·S = Ry(a) and S·Rx(a)·S = Rx(−a):

    carla yaw = −yaw,  carla pitch = −pitch,  carla roll = +roll

for a Vantage pose (yaw, pitch, roll). carla_rotation() gets them from any rotation matrix by that conjugation and
CARLA's own Euler extraction, so a rotation that is not a plain yaw/pitch/roll triple (a cube face of a camera) maps
the same way. A pseudo-vector (an angular velocity) mirrors with the sign of the determinant: ω_vantage =
(−ωx, ωy, −ωz) of CARLA's numbers.

Sensor data CARLA returns (checked against its source and a running 0.9.16 server, see vantage.sim.carla_bridge):
lidar points and radar detections in the sensor's own CARLA axes (x forward, y right, z up), camera images with the
optical axis along the actor's +x, image right its +y and image down its −z.
"""
from __future__ import annotations

import math

import numpy as np

S = np.diag([1.0, -1.0, 1.0])            # mirrors y: Vantage (right-handed) <-> CARLA (left-handed)
# camera: CARLA actor axes (x forward, y right, z up) -> optical (x right, y down, z forward)
CARLA_TO_OPTICAL = np.array([[0.0, 1.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [1.0, 0.0, 0.0]])
# Vantage body (x forward, y left, z up) -> optical; the same as core.mathutil.BODY_TO_OPTICAL
BODY_TO_OPTICAL = np.array([[0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0]])


def carla_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """CARLA's rotation matrix for a carla.Rotation (degrees), as Transform::GetMatrix computes it."""
    p, y, r = math.radians(pitch), math.radians(yaw), math.radians(roll)
    cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                     [sp, -cp * sr, cp * cr]])


def carla_euler(M: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) degrees of a rotation matrix in CARLA's axes, the inverse of carla_matrix (yaw and roll in
    (−180, 180]; at pitch ±90° the roll goes into the yaw)."""
    M = np.asarray(M, float)
    sp = max(-1.0, min(1.0, float(M[2, 0])))
    pitch = math.degrees(math.asin(sp))
    if abs(sp) < 1.0 - 1e-12:
        yaw = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        roll = math.degrees(math.atan2(-M[2, 1], M[2, 2]))
    else:                                   # gimbal lock: roll 0 (then M[0,1] = −sin yaw, M[1,1] = cos yaw)
        yaw = math.degrees(math.atan2(-M[0, 1], M[1, 1]))
        roll = 0.0
    return pitch, _wrap(yaw), _wrap(roll)


def _wrap(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a + 0.0


def carla_rotation(R_vehicle: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) of the carla.Rotation for a body -> vehicle rotation in Vantage's frame."""
    return carla_euler(S @ np.asarray(R_vehicle, float) @ S)


def vantage_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """The body -> vehicle rotation in Vantage's frame of a carla.Rotation."""
    return S @ carla_matrix(pitch, yaw, roll) @ S


def carla_location(p) -> list:
    """A Vantage point (x, y, z) in CARLA's axes."""
    x, y, z = (float(v) for v in p)
    return [x, -y + 0.0, z]


def to_vantage(P) -> np.ndarray:
    """Points (K, 3) or a vector in CARLA's axes -> Vantage's (mirror y)."""
    P = np.array(P, float, copy=True)
    P[..., 1] *= -1.0
    return P


def angular_to_vantage(w) -> np.ndarray:
    """An angular velocity (pseudo-vector) in CARLA's axes -> Vantage's: (−ωx, ωy, −ωz)."""
    w = np.array(w, float, copy=True)
    w[..., 0] *= -1.0
    w[..., 2] *= -1.0
    return w


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform_matrix(location, rotation) -> np.ndarray:
    """4x4 of a CARLA (location, (pitch, yaw, roll)) in CARLA's axes."""
    T = np.eye(4)
    T[:3, :3] = carla_matrix(*rotation)
    T[:3, 3] = [float(v) for v in location]
    return T


def vantage_transform(location, rotation) -> np.ndarray:
    """4x4 in Vantage's axes (ENU for CARLA's world: x east, y north, z up) of a CARLA (location, (pitch, yaw, roll))."""
    T = np.eye(4)
    T[:3, :3] = vantage_matrix(*rotation)
    T[:3, 3] = to_vantage(location)
    return T


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis × angle (radians), robust near 0 and 180°."""
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if ang > math.pi - 1e-6:
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(ax))
        ax = (R[:, i] + np.eye(3)[:, i]) / (2.0 * max(ax[i], 1e-12))
        return ax / (np.linalg.norm(ax) or 1.0) * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return v * ang


def quat_wxyz(R: np.ndarray) -> list:
    """[w, x, y, z] of a rotation matrix (w >= 0)."""
    m = np.asarray(R, float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    q = [w / n, x / n, y / n, z / n]
    return [-v for v in q] if q[0] < 0 else q
