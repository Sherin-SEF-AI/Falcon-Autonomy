"""Falcon's ROS 2 bridge: the topic plan, the link between the stack worker and the bridge node, the control law, and
the commands that start the node (standard library + numpy; no ROS, no carla). docs/stack-testing.md is the guide.

Three processes. The **stack worker** (falcon_autonomy.workers.stack, CARLA's Python) steps CARLA in synchronous mode
and encodes every sensor reading of a step as a ROS 2 message (CDR, core.ros2msg); the **bridge node**
(falcon_autonomy/ros2/falcon_bridge_node.py, rclpy — in the falcon-ros2-bridge:humble container or natively on a ROS 2
machine) publishes those bytes as they are and forwards the stack's control messages raw; the **stack under test**
(Autoware, or the small follower falcon_autonomy/ros2/follower_stack.py) is any ROS 2 software.

The link (worker <-> node): TCP; the worker listens on 127.0.0.1:<port> (or the host it is given), the node connects.
Port 0 in a config means "a free one": the runner (sim.stacksuite.run_one) picks it just before the run starts and
fills it into the run's config and the launch commands (a CARLA server may listen anywhere in Falcon's port range).
Framing is the live link's (core.livelink.pack: >II header length, payload length; a JSON header; the payload).

    worker -> node
      {"t": "hello", "protocol": 1, "topics": [{"topic", "type", "qos": "reliable" | "best_effort" | "latched"}],
       "control": {"topic", "type"} | null, "run": "<run name>"}
      {"t": "step", "step": i, "stamp_ns": t, "msgs": [{"topic", "nbytes"}, ...], "control": true | false}
            payload: the messages' CDR bytes in that order; the node publishes /clock first, then the rest in order
      {"t": "bye", "reason": "..."}
    node -> worker
      {"t": "ready", "node": "falcon_bridge", "ros_distro", "rmw", "control": {"topic", "type", "ok", "why"},
       "missing": [{"topic", "type", "why"}]}   topics whose message package the node's ROS lacks (not published)
      {"t": "published", "step": i, "publish_us": µs}   after publishing step i (the node's own cost)
      {"t": "control", "step": i, "topic", "type", "recv_ns": node clock}   payload: the control message's CDR
            every control message the stack sends, tagged with the last step the node published

Synchronous, closed loop. After step i is published the worker waits for the control of step i: the first control
message tagged step i whose stamp is not older than step i's stamp (a stamp of 0 counts as "for the step it was
received in"). Older messages are stale (counted, ignored). If none arrives within deadline_s of wall clock the step
is a **missed deadline**: the worker records it and applies on_miss ("hold" the last control, or "brake"), and the
simulation goes on — it never advances while a deadline is still open. The first step waits start_wait_s instead (the
stack starting up); it is not counted as a miss unless that passes too.

Control law (ControlMapper). carla_msgs/CarlaEgoVehicleControl is applied as it is. A target command (Autoware's
Control, Ackermann) becomes throttle / steer / brake: steer = −tyre angle / max steer angle (ROS positive left, CARLA
positive right; max steer: 70° in kinematic motion — the director's bicycle model — else the front wheels' measured
max_steer_angle); the longitudinal demand a = accel (when defined) + K_V · (speed − v), K_V = 1.0 1/s (heuristic),
gives throttle a / 3.0 m/s² or brake −a / 8.0 m/s² — exactly the director's kinematic model (full throttle 3 m/s², full
brake 8 m/s²), a heuristic approximation in physics motion. A negative target speed drives in reverse.
"""
from __future__ import annotations

import math
import os
import shlex

from .livelink import HEAD, Reader, pack           # noqa: F401  (the link's framing)

PROTOCOL = 1
DEFAULT_PORT = 0                # 0: a free port, chosen when the run starts (free_port); a CARLA server may listen
                                # on any port of Falcon's range, so no fixed default is safe
IMAGE = 'falcon-ros2-bridge:humble'
BASE_IMAGE = 'ros:humble-ros-base'
NS = '/falcon/ego'
V2X_NS = '/falcon/v2x'


def ros_name(s: str) -> str:
    """An id as a ROS name token (letters, digits, underscores)."""
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in s) or '_'

CONTROL_TYPES = {
    'autoware': ('autoware_control_msgs/msg/Control', '/control/command/control_cmd'),
    'ackermann': ('ackermann_msgs/msg/AckermannDriveStamped', f'{NS}/ackermann_cmd'),
    'carla': ('carla_msgs/msg/CarlaEgoVehicleControl', f'{NS}/vehicle_control_cmd'),
}
ON_MISS = ('hold', 'brake')

K_V = 1.0                      # 1/s, speed error -> acceleration demand (heuristic)
ACCEL_FULL_THROTTLE = 3.0      # m/s² at throttle 1: the director's kinematic model (heuristic in physics motion)
DECEL_FULL_BRAKE = 8.0         # m/s² at brake 1: likewise
KINEMATIC_MAX_STEER_DEG = 70.0  # the director's bicycle model (workers.director.Director.STEER_DEG)


def bridge_defaults() -> dict:
    """A bridge section of a stack config (every field documented in docs/stack-testing.md)."""
    return {'host': '127.0.0.1', 'port': DEFAULT_PORT, 'control': 'autoware', 'control_topic': '',
            'deadline_s': 0.5, 'start_wait_s': 60.0, 'connect_wait_s': 120.0, 'on_miss': 'hold',
            'image_encoding': 'bgr8', 'namespace': NS, 'autoware_status': True, 'topics': {},
            'qos': 'reliable', 'ego_tf': True, 'odometry': True}


def topic_plan(rig: dict, bridge: dict, rsus=()) -> list:
    """[{'topic', 'type', 'qos', 'kind', 'sensor'}] the worker publishes for a rig: /clock, /tf_static (latched), /tf
    (map -> base_link, ground truth; unless ego_tf is false — a stack that localises publishes it), the odometry
    (ground truth; unless odometry is false), each pinhole camera's image and camera_info, each lidar's points, each
    IMU's and GNSS's readings, each roadside unit's CPM (rsus: their ids; bridge v2x false drops them) as
    udp_msgs/UdpPacket (BTP-B + UPER, what the etsi_its_messages conversion node decodes) and as JSON
    (std_msgs/String), and Autoware's vehicle status topics when autoware_status. bridge['topics'] renames any of them
    ({default topic: new topic})."""
    ns = (bridge.get('namespace') or NS).rstrip('/')
    qos = bridge.get('qos', 'reliable')
    out = [{'topic': '/clock', 'type': 'rosgraph_msgs/msg/Clock', 'qos': qos, 'kind': 'clock'},
           {'topic': '/tf_static', 'type': 'tf2_msgs/msg/TFMessage', 'qos': 'latched', 'kind': 'tf_static'}]
    if bridge.get('ego_tf', True):                   # off for a stack that localises (and publishes map -> base_link)
        out.append({'topic': '/tf', 'type': 'tf2_msgs/msg/TFMessage', 'qos': qos, 'kind': 'tf'})
    if bridge.get('odometry', True):
        out.append({'topic': f'{ns}/odometry', 'type': 'nav_msgs/msg/Odometry', 'qos': qos, 'kind': 'odometry'})
    for s in rig.get('sensors', []):
        name = ''.join(c if c.isalnum() or c == '_' else '_' for c in s['name']) or s['id']
        t = s['type']
        if t == 'camera.pinhole':
            out.append({'topic': f'{ns}/{name}/image', 'type': 'sensor_msgs/msg/Image', 'qos': qos,
                        'kind': 'image', 'sensor': s['id']})
            out.append({'topic': f'{ns}/{name}/camera_info', 'type': 'sensor_msgs/msg/CameraInfo', 'qos': qos,
                        'kind': 'camera_info', 'sensor': s['id']})
        elif t == 'lidar':
            out.append({'topic': f'{ns}/{name}/points', 'type': 'sensor_msgs/msg/PointCloud2', 'qos': qos,
                        'kind': 'points', 'sensor': s['id']})
        elif t == 'imu':
            out.append({'topic': f'{ns}/{name}/imu', 'type': 'sensor_msgs/msg/Imu', 'qos': qos, 'kind': 'imu',
                        'sensor': s['id']})
        elif t == 'gnss':
            out.append({'topic': f'{ns}/{name}/fix', 'type': 'sensor_msgs/msg/NavSatFix', 'qos': qos,
                        'kind': 'fix', 'sensor': s['id']})
    if bridge.get('v2x', True):
        for rid in rsus or ():
            n = ros_name(rid)
            out += [{'topic': f'{V2X_NS}/{n}/cpm', 'type': 'udp_msgs/msg/UdpPacket', 'qos': qos, 'kind': 'cpm',
                     'station': rid},
                    {'topic': f'{V2X_NS}/{n}/cpm_json', 'type': 'std_msgs/msg/String', 'qos': qos, 'kind': 'cpm_json',
                     'station': rid}]
    if bridge.get('autoware_status', True):
        out += [{'topic': '/vehicle/status/velocity_status', 'type': 'autoware_vehicle_msgs/msg/VelocityReport',
                 'qos': qos, 'kind': 'velocity_status'},
                {'topic': '/vehicle/status/steering_status', 'type': 'autoware_vehicle_msgs/msg/SteeringReport',
                 'qos': qos, 'kind': 'steering_status'},
                {'topic': '/vehicle/status/gear_status', 'type': 'autoware_vehicle_msgs/msg/GearReport',
                 'qos': qos, 'kind': 'gear_status'},
                {'topic': '/vehicle/status/control_mode', 'type': 'autoware_vehicle_msgs/msg/ControlModeReport',
                 'qos': qos, 'kind': 'control_mode'}]
    ren = bridge.get('topics') or {}
    for e in out:
        e['topic'] = ren.get(e['topic'], e['topic'])
    return out


def free_port(host: str = '127.0.0.1') -> int:
    """A TCP port free on host now (the kernel's choice), for a run whose bridge port is 0."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def control_spec(bridge: dict) -> dict:
    kind = bridge.get('control', 'autoware')
    if kind not in CONTROL_TYPES:
        raise ValueError(f'control {kind!r}: one of {", ".join(CONTROL_TYPES)}')
    typ, topic = CONTROL_TYPES[kind]
    return {'kind': kind, 'type': typ, 'topic': bridge.get('control_topic') or topic}


def check_bridge(bridge: dict) -> list:
    """Problems with a bridge section (empty: fine)."""
    out = []
    try:
        control_spec(bridge)
    except ValueError as e:
        out.append(str(e))
    if bridge.get('on_miss', 'hold') not in ON_MISS:
        out.append(f'on_miss {bridge.get("on_miss")!r}: one of {", ".join(ON_MISS)}')
    for k in ('deadline_s', 'start_wait_s'):
        if not float(bridge.get(k, 1.0)) > 0:
            out.append(f'{k} must be positive')
    port = int(bridge.get('port', DEFAULT_PORT))
    if port != 0 and (not 1024 <= port <= 65535 or port in range(2000, 2011) or port == 8100):
        out.append(f'port {port}: 0 (a free one) or 1024–65535, not one of CARLA\'s defaults (2000–2010, 8100)')
    if bridge.get('image_encoding', 'bgr8') not in ('bgr8', 'rgb8', 'bgra8'):
        out.append('image_encoding: bgr8, rgb8 or bgra8')
    return out


class ControlMapper:
    """A stack's command (core.ros2msg.command_of) -> CARLA's VehicleControl fields (module docstring: the law)."""

    def __init__(self, max_steer_deg: float = KINEMATIC_MAX_STEER_DEG, k_v: float = K_V,
                 accel_full: float = ACCEL_FULL_THROTTLE, decel_full: float = DECEL_FULL_BRAKE):
        self.max_steer = math.radians(max(float(max_steer_deg), 1.0))
        self.k_v, self.accel_full, self.decel_full = float(k_v), float(accel_full), float(decel_full)

    def __call__(self, cmd: dict, speed_mps: float) -> dict:
        if cmd['kind'] == 'direct':
            return {'throttle': _clip(cmd['throttle'], 0.0, 1.0), 'steer': _clip(cmd['steer'], -1.0, 1.0),
                    'brake': _clip(cmd['brake'], 0.0, 1.0), 'hand_brake': bool(cmd.get('hand_brake')),
                    'reverse': bool(cmd.get('reverse'))}
        reverse = cmd['speed'] < 0
        target = abs(float(cmd['speed']))
        v = abs(float(speed_mps))
        a = (float(cmd['accel']) * (-1.0 if reverse else 1.0) if cmd.get('accel') is not None else 0.0)
        a += self.k_v * (target - v)
        steer = _clip(-float(cmd['steer_rad']) / self.max_steer, -1.0, 1.0)
        if target < 0.01 and v < 0.05 and a <= 0:
            return {'throttle': 0.0, 'steer': steer, 'brake': 1.0, 'hand_brake': False, 'reverse': reverse}
        if a >= 0:
            return {'throttle': _clip(a / self.accel_full, 0.0, 1.0), 'steer': steer, 'brake': 0.0,
                    'hand_brake': False, 'reverse': reverse}
        return {'throttle': 0.0, 'steer': steer, 'brake': _clip(-a / self.decel_full, 0.0, 1.0),
                'hand_brake': False, 'reverse': reverse}

    def tyre_angle(self, steer: float) -> float:
        """CARLA's steer (positive right) -> the tyre angle in ROS's convention (rad, positive left)."""
        return -float(steer) * self.max_steer


def _clip(v, lo, hi):
    return min(max(float(v), lo), hi)


class Deadlines:
    """The worker's bookkeeping of control deadlines: accept(step, step_stamp_ns, stamp_ns) says whether a control
    message is the one for that step; the counts go to run.json "bridge"."""

    def __init__(self):
        self.missed = 0
        self.stale = 0
        self.accepted = 0
        self.waits_ms: list = []
        self.missed_steps: list = []

    def accept(self, step: int, msg_step: int, step_stamp_ns: int, stamp_ns: int) -> bool:
        if msg_step != step or (stamp_ns and stamp_ns < step_stamp_ns):
            self.stale += 1
            return False
        self.accepted += 1
        return True

    def miss(self, step: int):
        self.missed += 1
        if len(self.missed_steps) < 1000:
            self.missed_steps.append(step)

    def report(self) -> dict:
        import numpy as np
        w = np.asarray(self.waits_ms, float)
        q = (lambda p: round(float(np.percentile(w, p)), 3)) if len(w) else (lambda p: None)
        return {'accepted': self.accepted, 'missed': self.missed, 'stale': self.stale,
                'missed_steps': self.missed_steps, 'wait_ms': {'p50': q(50), 'p95': q(95), 'p99': q(99),
                                                               'max': q(100)}, 'source': 'measured'}


# ------------------------------------------------------------------------------------- starting the node and a stack
def ros_source() -> str:
    """The folder falcon_autonomy/ros2 (the node and the follower; mounted read-only into the container)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ros2')


def docker_run(name: str, args: list, domain_id: int = 0, image: str = IMAGE, udp_only: bool = True) -> list:
    """docker run argv for a ROS 2 process in the bridge image: host networking (DDS), the source mounted read-only,
    ROS_DOMAIN_ID, and (udp_only) the UDP-only Fast DDS profile."""
    argv = ['docker', 'run', '--rm', '--name', name, '--network', 'host', '--ipc', 'host',
            '-v', f'{ros_source()}:/falcon/ros2:ro', '-e', f'ROS_DOMAIN_ID={int(domain_id)}']
    if udp_only:
        argv += ['-e', 'FASTRTPS_DEFAULT_PROFILES_FILE=/opt/falcon/fastdds-udp.xml']
    return argv + [image] + [str(a) for a in args]


def bridge_command(port: int, host: str = '127.0.0.1', name: str = 'falcon-bridge', domain_id: int = 0,
                   docker: bool = True, udp_only: bool = True) -> list:
    """The bridge node's command: in the bridge image (docker), or natively on a ROS 2 machine (python3)."""
    args = ['python3', '/falcon/ros2/falcon_bridge_node.py' if docker else
            os.path.join(ros_source(), 'falcon_bridge_node.py'), '--connect', f'{host}:{port}']
    return docker_run(name, args, domain_id, udp_only=udp_only) if docker else args


def follower_command(control: str = 'autoware', speed_mps: float = 8.0, name: str = 'falcon-follower',
                     domain_id: int = 0, docker: bool = True, udp_only: bool = True, extra=()) -> list:
    """The test stack (falcon_autonomy/ros2/follower_stack.py): cruise along the start heading, brake for lidar
    points in its corridor."""
    args = ['python3', '/falcon/ros2/follower_stack.py' if docker else os.path.join(ros_source(),
                                                                                    'follower_stack.py'),
            '--control', control, '--speed', f'{float(speed_mps):g}'] + [str(a) for a in extra]
    return docker_run(name, args, domain_id, udp_only=udp_only) if docker else args


def cpm_converter_command(rsu_id: str, name: str = 'falcon-cpm-converter', domain_id: int = 0, udp_only: bool = True,
                          docker: bool = True) -> list:
    """The etsi_its_messages conversion node (ika-rwth-aachen, MIT; in the bridge image) decoding one roadside unit's
    UPER CPMs into etsi_its_cpm_ts_msgs/CollectivePerceptionMessage on /falcon/v2x/<rsu>/cpm_ts: its UDP input is
    remapped to Falcon's /falcon/v2x/<rsu>/cpm (a BTP-B header, then the CPM: the node's defaults)."""
    n = ros_name(rsu_id)
    args = ['ros2', 'run', 'etsi_its_conversion', 'etsi_its_conversion_node', '--ros-args',
            '-r', f'__node:={ros_name(name)}', '-r', f'~/udp/in:={V2X_NS}/{n}/cpm',
            '-r', f'~/cpm_ts/out:={V2X_NS}/{n}/cpm_ts', '-p', 'udp2ros_etsi_types:=[cpm_ts]',
            '-p', 'ros2udp_etsi_types:=[cpm_ts]', '-r', f'~/udp/out:={V2X_NS}/{n}/cpm_echo']
    return docker_run(name, args, domain_id, udp_only=udp_only) if docker else args


def shell(argv: list) -> str:
    return ' '.join(shlex.quote(str(a)) for a in argv)


# ----------------------------------------------------------------------------------------------- Autoware's interface
# Falcon's topics against the interfaces of Autoware (Universe, autoware_msgs 1.x as released for ROS 2 Humble). NOT
# run here: Autoware is not installed on the reference machine; check each against your Autoware version's
# launch files (sensor kit, vehicle interface) before relying on it.
AUTOWARE_MAPPING = [
    # (Falcon topic or pattern, Falcon type, Autoware topic, Autoware type, note)
    ('/control/command/control_cmd', 'autoware_control_msgs/msg/Control', '/control/command/control_cmd',
     'autoware_control_msgs/msg/Control', 'Falcon subscribes Autoware\'s control output (bridge control "autoware")'),
    ('/vehicle/status/velocity_status', 'autoware_vehicle_msgs/msg/VelocityReport', '/vehicle/status/velocity_status',
     'autoware_vehicle_msgs/msg/VelocityReport', 'published by Falcon (the vehicle interface\'s role)'),
    ('/vehicle/status/steering_status', 'autoware_vehicle_msgs/msg/SteeringReport', '/vehicle/status/steering_status',
     'autoware_vehicle_msgs/msg/SteeringReport', 'tyre angle of the control applied (positive left)'),
    ('/vehicle/status/gear_status', 'autoware_vehicle_msgs/msg/GearReport', '/vehicle/status/gear_status',
     'autoware_vehicle_msgs/msg/GearReport', 'DRIVE or REVERSE'),
    ('/vehicle/status/control_mode', 'autoware_vehicle_msgs/msg/ControlModeReport', '/vehicle/status/control_mode',
     'autoware_vehicle_msgs/msg/ControlModeReport', 'always AUTONOMOUS'),
    (f'{NS}/<lidar>/points', 'sensor_msgs/msg/PointCloud2', '/sensing/lidar/top/pointcloud_raw',
     'sensor_msgs/msg/PointCloud2', 'x y z intensity only: Autoware\'s pointcloud_raw_ex fields (ring, return type, '
                                    'time) are not produced'),
    (f'{NS}/<imu>/imu', 'sensor_msgs/msg/Imu', '/sensing/imu/tamagawa/imu_raw', 'sensor_msgs/msg/Imu',
     'no orientation (covariance[0] = −1)'),
    (f'{NS}/<gnss>/fix', 'sensor_msgs/msg/NavSatFix', '/sensing/gnss/ublox/nav_sat_fix', 'sensor_msgs/msg/NavSatFix',
     'CARLA\'s geo-reference of the map'),
    (f'{NS}/<camera>/image', 'sensor_msgs/msg/Image', '/sensing/camera/camera0/image_rect_color',
     'sensor_msgs/msg/Image', 'an ideal pinhole: already rectified'),
    (f'{NS}/<camera>/camera_info', 'sensor_msgs/msg/CameraInfo', '/sensing/camera/camera0/camera_info',
     'sensor_msgs/msg/CameraInfo', ''),
    (f'{NS}/odometry', 'nav_msgs/msg/Odometry', '/localization/kinematic_state', 'nav_msgs/msg/Odometry',
     'ground truth: only when Autoware\'s localisation is bypassed (else let it localise from the sensors)'),
    ('/clock', 'rosgraph_msgs/msg/Clock', '/clock', 'rosgraph_msgs/msg/Clock', 'run Autoware with use_sim_time:=true'),
]


def autoware_remaps(rig: dict) -> dict:
    """{Falcon topic: Autoware topic} for a rig (the first camera, lidar, IMU and GNSS), to put in a bridge's
    'topics' so the node publishes under Autoware's names."""
    out = {}
    first = {}
    for s in rig.get('sensors', []):
        first.setdefault(s['type'], s)
    name = lambda s: ''.join(c if c.isalnum() or c == '_' else '_' for c in s['name']) or s['id']  # noqa: E731
    if 'lidar' in first:
        out[f'{NS}/{name(first["lidar"])}/points'] = '/sensing/lidar/top/pointcloud_raw'
    if 'imu' in first:
        out[f'{NS}/{name(first["imu"])}/imu'] = '/sensing/imu/tamagawa/imu_raw'
    if 'gnss' in first:
        out[f'{NS}/{name(first["gnss"])}/fix'] = '/sensing/gnss/ublox/nav_sat_fix'
    if 'camera.pinhole' in first:
        c = name(first['camera.pinhole'])
        out[f'{NS}/{c}/image'] = '/sensing/camera/camera0/image_rect_color'
        out[f'{NS}/{c}/camera_info'] = '/sensing/camera/camera0/camera_info'
    return out
