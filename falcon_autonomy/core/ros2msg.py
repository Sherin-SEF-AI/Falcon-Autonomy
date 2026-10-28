"""ROS 2 messages for Falcon's ROS 2 bridge (numpy and the standard library; no ROS): the extra message definitions,
a CDR decoder, the conversions from CARLA's numbers to ROS messages, and the control commands a stack sends back.

Encoding reuses core.mcap_bag: register() adds the definitions below to mcap_bag.MSG_DEFS, so mcap_bag.encode() writes
them as plain CDR (XCDR1, little-endian, the encapsulation header 00 01 00 00) exactly as rclpy serialises them, and
the bridge node publishes those bytes as they are (rclpy's Publisher.publish(bytes), a serialized message). decode()
is the inverse, for the control messages the node forwards raw (a subscription with raw=True): it reads XCDR1 in
either byte order (00 01 little-endian, 00 00 big-endian), aligning each primitive to its own size relative to the
payload after the header, as encode() writes.

Definitions added (field layouts of ROS 2 Humble's packages, checked against the ROS 2 Humble apt packages
ros-humble-autoware-control-msgs 1.13 / -autoware-vehicle-msgs 1.13 / -ackermann-msgs 2.0.2 by a byte-level round
trip in the falcon-ros2-bridge:humble container, tests/test_stack_ros2.py):
    rosgraph_msgs/Clock, geometry_msgs/{Point, Pose, PoseWithCovariance, Twist, TwistWithCovariance},
    nav_msgs/Odometry, autoware_control_msgs/{Control, Lateral, Longitudinal},
    autoware_vehicle_msgs/{VelocityReport, SteeringReport, GearReport, ControlModeReport},
    ackermann_msgs/{AckermannDrive, AckermannDriveStamped}, carla_msgs/CarlaEgoVehicleControl
    (carla-simulator/ros-carla-msgs, the message CARLA's own ROS 2 interface and its ROS bridge take).

Frames (the same as Falcon's MCAP export, core.exporters "_mcap"): "map" is CARLA's world mirrored (x, −y, z), ROS's
right-handed axes (REP 103); "base_link" is CARLA's vehicle origin with x forward, y left, z up; "<sensor>_link" a
sensor body; "<sensor>_optical" a camera's optical frame (z forward, x right, y down). Lidar points are CARLA's sensor
frame with y mirrored; the IMU's linear acceleration is (x, −y, z) and its angular velocity (−x, y, −z) of CARLA's
numbers (a mirrored frame flips a pseudo-vector's other components); CARLA's IMU gives no orientation
(orientation_covariance[0] = −1, REP 145). Steering: ROS angles are positive to the left, CARLA's steer positive to the
right.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from . import mcap_bag as B

EXTRA_DEFS = {
    'rosgraph_msgs/msg/Clock': 'builtin_interfaces/Time clock\n',
    'geometry_msgs/msg/Point': 'float64 x\nfloat64 y\nfloat64 z\n',
    'geometry_msgs/msg/Pose': 'geometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n',
    'geometry_msgs/msg/PoseWithCovariance': 'geometry_msgs/Pose pose\nfloat64[36] covariance\n',
    'geometry_msgs/msg/Twist': 'geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular\n',
    'geometry_msgs/msg/TwistWithCovariance': 'geometry_msgs/Twist twist\nfloat64[36] covariance\n',
    'nav_msgs/msg/Odometry': ('std_msgs/Header header\nstring child_frame_id\n'
                              'geometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist\n'),
    'autoware_control_msgs/msg/Lateral': ('builtin_interfaces/Time stamp\nbuiltin_interfaces/Time control_time\n'
                                          'float32 steering_tire_angle\nfloat32 steering_tire_rotation_rate\n'
                                          'bool is_defined_steering_tire_rotation_rate\n'),
    'autoware_control_msgs/msg/Longitudinal': ('builtin_interfaces/Time stamp\nbuiltin_interfaces/Time control_time\n'
                                               'float32 velocity\nfloat32 acceleration\nfloat32 jerk\n'
                                               'bool is_defined_acceleration\nbool is_defined_jerk\n'),
    'autoware_control_msgs/msg/Control': ('builtin_interfaces/Time stamp\nbuiltin_interfaces/Time control_time\n'
                                          'autoware_control_msgs/Lateral lateral\n'
                                          'autoware_control_msgs/Longitudinal longitudinal\n'),
    'autoware_vehicle_msgs/msg/VelocityReport': ('std_msgs/Header header\nfloat32 longitudinal_velocity\n'
                                                 'float32 lateral_velocity\nfloat32 heading_rate\n'),
    'autoware_vehicle_msgs/msg/SteeringReport': 'builtin_interfaces/Time stamp\nfloat32 steering_tire_angle\n',
    'autoware_vehicle_msgs/msg/GearReport': ('uint8 NONE = 0\nuint8 NEUTRAL = 1\nuint8 DRIVE = 2\nuint8 REVERSE = 20\n'
                                             'uint8 PARK = 22\nbuiltin_interfaces/Time stamp\nuint8 report\n'),
    'autoware_vehicle_msgs/msg/ControlModeReport': ('uint8 NO_COMMAND = 0\nuint8 AUTONOMOUS = 1\nuint8 MANUAL = 4\n'
                                                    'builtin_interfaces/Time stamp\nuint8 mode\n'),
    'ackermann_msgs/msg/AckermannDrive': ('float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\n'
                                          'float32 acceleration\nfloat32 jerk\n'),
    'ackermann_msgs/msg/AckermannDriveStamped': 'std_msgs/Header header\nackermann_msgs/AckermannDrive drive\n',
    'carla_msgs/msg/CarlaEgoVehicleControl': ('std_msgs/Header header\nfloat32 throttle\nfloat32 steer\n'
                                              'float32 brake\nbool hand_brake\nbool reverse\nint32 gear\n'
                                              'bool manual_gear_shift\n'),
}
GEAR_DRIVE, GEAR_REVERSE, GEAR_PARK = 2, 20, 22
MODE_AUTONOMOUS = 1


def register():
    """Add EXTRA_DEFS to core.mcap_bag's definitions (idempotent; an existing definition is never replaced)."""
    for k, v in EXTRA_DEFS.items():
        B.MSG_DEFS.setdefault(k, v)


register()


def encode(msg_type: str, msg: dict) -> bytes:
    """core.mcap_bag.encode. rclpy sometimes ends a message with up to 3 more zero bytes (a 13-byte GearReport is
    serialised in 16, a 78-byte Control in 78: measured in ROS 2 Humble); readers stop at the last field, so the
    message is the same either way."""
    return B.encode(msg_type, msg)


# ------------------------------------------------------------------------------------------------------------ decode
class _Cursor:
    def __init__(self, data: bytes):
        if len(data) < 4 or data[0] != 0 or data[1] not in (0, 1):
            raise ValueError('not a plain-CDR message (encapsulation header 00 00 or 00 01 expected)')
        self.d = memoryview(data)
        self.o = 4
        self.e = '<' if data[1] == 1 else '>'

    def align(self, n):
        self.o += (-(self.o - 4)) % n

    def take(self, code, size, count=None):
        """One primitive (count None) or a list of count of them."""
        self.align(size)
        k = 1 if count is None else count
        end = self.o + size * k
        if end > len(self.d):
            raise ValueError('CDR message too short')
        v = struct.unpack_from(f'{self.e}{k}{code}', self.d, self.o)
        self.o = end
        return v[0] if count is None else list(v)

    def raw(self, n):
        end = self.o + n
        if end > len(self.d):
            raise ValueError('CDR message too short')
        b = bytes(self.d[self.o:end])
        self.o = end
        return b


def _dvalue(c: _Cursor, base: str):
    if base == 'string':
        n = c.take('I', 4)
        s = c.raw(n)                                  # the length counts the terminating NUL
        return s[:-1].decode('utf-8') if n else ''
    if base in B._PRIM:
        code, size = B._PRIM[base]
        v = c.take(code, size)
        return bool(v) if base == 'bool' else v
    return _decode(c, base)


def _decode(c: _Cursor, t: str) -> dict:
    out = {}
    for name, base, n, _default in B._spec(t):
        if n is None:
            out[name] = _dvalue(c, base)
            continue
        count = c.take('I', 4) if n < 0 else n
        if base in B._BYTES:
            out[name] = c.raw(count)
        elif base in B._PRIM and base != 'bool':
            code, size = B._PRIM[base]
            out[name] = c.take(code, size, count) if count else []
        else:
            out[name] = [_dvalue(c, base) for _ in range(count)]
    return out


def decode(msg_type: str, data: bytes) -> dict:
    """CDR bytes (with the encapsulation header) -> the message as a dict, as core.mcap_bag.encode takes it."""
    return _decode(_Cursor(bytes(data)), B._norm(msg_type))


def stamp_ns(t: dict) -> int:
    """A builtin_interfaces/Time dict -> integer nanoseconds."""
    return int(t.get('sec', 0)) * 1_000_000_000 + int(t.get('nanosec', 0))


# ------------------------------------------------------------------------------------------- CARLA -> ROS geometry
S4 = np.diag([1.0, -1.0, 1.0, 1.0])


def carla_matrix(x, y, z, roll, pitch, yaw) -> np.ndarray:
    """CARLA's Transform::GetMatrix (4 × 4) of a pose (metres, degrees)."""
    from .rig import carla_rotation
    T = np.eye(4)
    T[:3, :3] = carla_rotation(roll, pitch, yaw)
    T[:3, 3] = (x, y, z)
    return T


def rh(T) -> np.ndarray:
    """A CARLA 4 × 4 transform in the mirrored, right-handed axes (as core.exporters.rh)."""
    return S4 @ np.asarray(T, float) @ S4


def clock(t_ns: int) -> dict:
    return {'clock': B.stamp(t_ns)}


def odometry(ego: dict, vel_world, ang_vel_world_deg, t_ns: int, frame='map', child='base_link') -> dict:
    """nav_msgs/Odometry of the ego: its pose in `frame` (CARLA's world mirrored), its twist in `child` (base_link)
    as REP 105 asks. ego: {x, y, z, roll, pitch, yaw} (CARLA); vel_world (m/s) and ang_vel_world_deg (deg/s, CARLA's
    get_angular_velocity) in CARLA's world axes. Covariances are zero (ground truth)."""
    T = rh(carla_matrix(ego['x'], ego['y'], ego['z'], ego.get('roll', 0.0), ego.get('pitch', 0.0), ego['yaw']))
    R = T[:3, :3]
    v = np.array([vel_world[0], -vel_world[1], vel_world[2]], float)            # world, mirrored
    w = np.radians(np.array([-ang_vel_world_deg[0], ang_vel_world_deg[1], -ang_vel_world_deg[2]], float))
    vb, wb = R.T @ v, R.T @ w
    q = B.quat_xyzw(R)
    return {'header': B.header(t_ns, frame), 'child_frame_id': child,
            'pose': {'pose': {'position': {'x': float(T[0, 3]), 'y': float(T[1, 3]), 'z': float(T[2, 3])},
                              'orientation': q}, 'covariance': [0.0] * 36},
            'twist': {'twist': {'linear': {'x': float(vb[0]), 'y': float(vb[1]), 'z': float(vb[2])},
                                'angular': {'x': float(wb[0]), 'y': float(wb[1]), 'z': float(wb[2])}},
                      'covariance': [0.0] * 36}}


def ego_tf(ego: dict, t_ns: int, frame='map', child='base_link') -> dict:
    T = rh(carla_matrix(ego['x'], ego['y'], ego['z'], ego.get('roll', 0.0), ego.get('pitch', 0.0), ego['yaw']))
    return {'transforms': [B.transform_stamped(T, t_ns, frame, child)]}


def link_name(s: dict) -> str:
    """A sensor's frame name stem, as the rig's URDF/TF export and the MCAP export name it."""
    return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in s['name']) or s['id']


def static_tf(rig: dict, t_ns: int) -> dict:
    """tf2_msgs/TFMessage of the rig's mounts (core.rig.tf_list: base_link -> <sensor>_link [-> <sensor>_optical])."""
    from .rig import matrix_of, tf_list
    out = []
    for t in tf_list(rig):
        T = np.eye(4)
        x, y, z, w = t['rotation']
        T[:3, :3] = matrix_of((w, x, y, z))
        T[:3, 3] = t['translation']
        out.append(B.transform_stamped(T, t_ns, t['parent'], t['child']))
    return {'transforms': out}


def image_bgr(bgra: bytes, width: int, height: int, t_ns: int, frame_id: str, encoding: str = 'bgr8') -> dict:
    """sensor_msgs/Image of CARLA's BGRA camera data: 'bgr8' (the alpha dropped, one copy) or 'bgra8' as it is."""
    a = np.frombuffer(bgra, np.uint8).reshape(int(height), int(width), 4)
    if encoding == 'bgr8':
        a = np.ascontiguousarray(a[:, :, :3])
    elif encoding == 'rgb8':
        a = np.ascontiguousarray(a[:, :, 2::-1])
    elif encoding != 'bgra8':
        raise ValueError(f'image encoding {encoding!r}: bgr8, rgb8 or bgra8')
    return {'header': B.header(t_ns, frame_id), 'height': int(height), 'width': int(width), 'encoding': encoding,
            'is_bigendian': 0, 'step': int(a.strides[0]), 'data': a.tobytes()}


def camera_info(s: dict, t_ns: int, frame_id: str) -> dict:
    """sensor_msgs/CameraInfo of a pinhole camera of the rig: K from its width, height and field of view (OpenCV
    pixel centres), plumb_bob with zero distortion (CARLA renders an ideal pinhole)."""
    from .rig import camera_K
    K = np.array(camera_K(s), float)
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return B.camera_info(int(s['width']), int(s['height']), K, 'plumb_bob', [0.0] * 5, t_ns, frame_id)


POINT_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])


def pointcloud(xyzi: np.ndarray, t_ns: int, frame_id: str) -> dict:
    """sensor_msgs/PointCloud2 (x y z intensity, float32) of CARLA lidar points (N × 4, CARLA's sensor frame):
    y mirrored into the sensor link's ROS axes."""
    P = np.asarray(xyzi, np.float32).reshape(-1, 4)
    a = np.empty(len(P), POINT_DTYPE)
    a['x'], a['y'], a['z'], a['intensity'] = P[:, 0], -P[:, 1], P[:, 2], P[:, 3]
    return B.pointcloud2(a, t_ns, frame_id)


def imu(accel, gyro, t_ns: int, frame_id: str, accel_var: float = 0.0, gyro_var: float = 0.0) -> dict:
    """sensor_msgs/Imu of CARLA's accelerometer (m/s², sensor axes) and gyroscope (rad/s): no orientation."""
    a, g = accel, gyro
    return {'header': B.header(t_ns, frame_id), 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 0.0},
            'orientation_covariance': [-1.0] + [0.0] * 8,
            'angular_velocity': {'x': -float(g[0]), 'y': float(g[1]), 'z': -float(g[2])},
            'angular_velocity_covariance': [gyro_var, 0, 0, 0, gyro_var, 0, 0, 0, gyro_var],
            'linear_acceleration': {'x': float(a[0]), 'y': -float(a[1]), 'z': float(a[2])},
            'linear_acceleration_covariance': [accel_var, 0, 0, 0, accel_var, 0, 0, 0, accel_var]}


def navsatfix(lat: float, lon: float, alt: float, t_ns: int, frame_id: str, s: dict | None = None) -> dict:
    """sensor_msgs/NavSatFix; the covariance from the rig's noise in degrees at 111 320 m per degree (heuristic)."""
    s = s or {}
    m_lat, m_lon = 111320.0, 111320.0 * math.cos(math.radians(lat))
    sl, so = float(s.get('noise_lat_stddev', 0.0)) * m_lat, float(s.get('noise_lon_stddev', 0.0)) * m_lon
    sa = float(s.get('noise_alt_stddev', 0.0))
    return {'header': B.header(t_ns, frame_id), 'status': {'status': 0, 'service': 1}, 'latitude': float(lat),
            'longitude': float(lon), 'altitude': float(alt), 'position_covariance': [sl * sl, 0, 0, 0, so * so, 0, 0,
                                                                                   0, sa * sa],
            'position_covariance_type': 2}


def velocity_report(speed_long: float, speed_lat: float, yaw_rate_left: float, t_ns: int) -> dict:
    return {'header': B.header(t_ns, 'base_link'), 'longitudinal_velocity': float(speed_long),
            'lateral_velocity': float(speed_lat), 'heading_rate': float(yaw_rate_left)}


def steering_report(tire_angle_left_rad: float, t_ns: int) -> dict:
    return {'stamp': B.stamp(t_ns), 'steering_tire_angle': float(tire_angle_left_rad)}


def gear_report(reverse: bool, t_ns: int) -> dict:
    return {'stamp': B.stamp(t_ns), 'report': GEAR_REVERSE if reverse else GEAR_DRIVE}


def control_mode_report(t_ns: int) -> dict:
    return {'stamp': B.stamp(t_ns), 'mode': MODE_AUTONOMOUS}


# ---------------------------------------------------------------------------------------------- control commands
def command_of(msg_type: str, m: dict) -> dict:
    """A decoded control message -> Falcon's command:
        {'kind': 'direct', 'throttle', 'steer' (CARLA: positive right), 'brake', 'hand_brake', 'reverse', 'stamp_ns'}
        {'kind': 'target', 'steer_rad' (tyre angle, positive left), 'speed' (m/s, negative: reverse),
         'accel' (m/s², None when undefined), 'stamp_ns'}
    autoware_control_msgs/Control: lateral.steering_tire_angle, longitudinal.velocity and .acceleration (when
    is_defined_acceleration), stamp = Control.stamp. ackermann_msgs/AckermannDriveStamped: drive.steering_angle,
    .speed, .acceleration (0 is taken as undefined: the message has no flag), stamp = header.stamp.
    carla_msgs/CarlaEgoVehicleControl: passed through (CARLA's convention), stamp = header.stamp."""
    t = B._norm(msg_type)
    if t == 'autoware_control_msgs/msg/Control':
        lon = m['longitudinal']
        return {'kind': 'target', 'steer_rad': float(m['lateral']['steering_tire_angle']),
                'speed': float(lon['velocity']),
                'accel': float(lon['acceleration']) if lon.get('is_defined_acceleration') else None,
                'stamp_ns': stamp_ns(m['stamp'])}
    if t == 'ackermann_msgs/msg/AckermannDriveStamped':
        d = m['drive']
        return {'kind': 'target', 'steer_rad': float(d['steering_angle']), 'speed': float(d['speed']),
                'accel': float(d['acceleration']) if d['acceleration'] != 0.0 else None,
                'stamp_ns': stamp_ns(m['header']['stamp'])}
    if t == 'carla_msgs/msg/CarlaEgoVehicleControl':
        return {'kind': 'direct', 'throttle': float(m['throttle']), 'steer': float(m['steer']),
                'brake': float(m['brake']), 'hand_brake': bool(m['hand_brake']), 'reverse': bool(m['reverse']),
                'stamp_ns': stamp_ns(m['header']['stamp'])}
    raise ValueError(f'{msg_type} is not a control message Falcon\'s bridge takes')
