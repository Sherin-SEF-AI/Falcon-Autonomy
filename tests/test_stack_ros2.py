"""Falcon's bridge against real ROS 2 (marker ros2: FALCON_ROS2=1, Docker and the falcon-ros2-bridge:humble image).

* Falcon's CDR encoding is what rclpy serialises (rclpy.serialization.serialize_message in ROS 2 Humble) for every
  message type the bridge publishes or takes: every data byte the same (rclpy leaves alignment padding uninitialised,
  so padding bytes may differ, and sometimes ends a message with up to 3 zero bytes more).
* Closed loop over real DDS: the stack worker (against the fake carla), the bridge node and the test follower, each in
  its own process (the node and the follower in containers named falcon-test-*, removed afterwards), with each of the
  three control message types.
"""
import json
import os
import socket
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest

from conftest import ROOT, worker_env
from falcon_autonomy.core import mcap_bag as B
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import ros2msg as R
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import stack as ST
from falcon_autonomy.core import stackbridge as SB

pytestmark = pytest.mark.ros2

SERIALISE = r'''
import json, sys
from rclpy.serialization import serialize_message
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.utilities import get_message
cases = json.load(open(sys.argv[1]))
out = {}
for name, (typ, fields) in cases.items():
    m = get_message(typ)()
    set_message_fields(m, fields)
    out[name] = serialize_message(m).hex()
print(json.dumps(out))
'''


def _jsonable(m):
    if isinstance(m, dict):
        return {k: _jsonable(v) for k, v in m.items()}
    if isinstance(m, (bytes, bytearray)):
        return list(m)
    if isinstance(m, list):
        return [_jsonable(v) for v in m]
    return m


def test_cdr_is_rclpys(tmp_path):
    pts = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, -5.0, 6.0, 0.25]], np.float32)
    bgra = (np.arange(4 * 3 * 4) % 251).astype(np.uint8)
    cases = {
        'clock': ('rosgraph_msgs/msg/Clock', R.clock(12_000_000_345)),
        'control': ('autoware_control_msgs/msg/Control', {
            'stamp': {'sec': 3, 'nanosec': 4}, 'control_time': {'sec': 5, 'nanosec': 6},
            'lateral': {'stamp': {'sec': 1, 'nanosec': 2}, 'steering_tire_angle': 0.25,
                        'steering_tire_rotation_rate': -0.5, 'is_defined_steering_tire_rotation_rate': True},
            'longitudinal': {'velocity': 5.5, 'acceleration': -1.25, 'jerk': 0.125, 'is_defined_acceleration': True,
                             'is_defined_jerk': False}}),
        'ackermann': ('ackermann_msgs/msg/AckermannDriveStamped', {
            'header': B.header(123456789012, 'base_link'),
            'drive': {'steering_angle': 0.1, 'steering_angle_velocity': 0.2, 'speed': -2.0, 'acceleration': 0.5,
                      'jerk': 0.25}}),
        'carla': ('carla_msgs/msg/CarlaEgoVehicleControl', {
            'header': B.header(5, 'x'), 'throttle': 0.5, 'steer': -0.25, 'brake': 0.125, 'hand_brake': True,
            'reverse': False, 'gear': -1, 'manual_gear_shift': True}),
        'velocity': ('autoware_vehicle_msgs/msg/VelocityReport', R.velocity_report(3.0, -0.125, 0.0625, 9)),
        'steering': ('autoware_vehicle_msgs/msg/SteeringReport', R.steering_report(0.1, 9)),
        'gear': ('autoware_vehicle_msgs/msg/GearReport', R.gear_report(True, 9)),
        'mode': ('autoware_vehicle_msgs/msg/ControlModeReport', R.control_mode_report(9)),
        'odometry': ('nav_msgs/msg/Odometry', R.odometry({'x': 1.0, 'y': 2.0, 'z': 0.1, 'yaw': 30.0},
                                                         (2.0, 1.0, 0.0), (0, 0, 10.0), 10 ** 9)),
        'imu': ('sensor_msgs/msg/Imu', R.imu((0.1, 0.2, 9.81), (0.01, 0.02, 0.03), 5, 'imu_link', 0.01, 0.001)),
        'fix': ('sensor_msgs/msg/NavSatFix', R.navsatfix(12.97, 77.59, 920.0, 5, 'gnss_link',
                                                         {'noise_lat_stddev': 1e-6})),
        'points': ('sensor_msgs/msg/PointCloud2', R.pointcloud(pts, 7, 'lidar_link')),
        'image': ('sensor_msgs/msg/Image', R.image_bgr(bgra.tobytes(), 3, 4, 7, 'cam_optical')),
        'camera_info': ('sensor_msgs/msg/CameraInfo', R.camera_info({'type': 'camera.pinhole', 'width': 640,
                                                                     'height': 480, 'fov': 70.0}, 7, 'cam_optical')),
        'tf': ('tf2_msgs/msg/TFMessage', R.ego_tf({'x': 1.0, 'y': -2.0, 'z': 0.5, 'roll': 1.0, 'pitch': 2.0,
                                                   'yaw': 33.0}, 11)),
    }
    (tmp_path / 'cases.json').write_text(json.dumps({k: [t, _jsonable(m)] for k, (t, m) in cases.items()}))
    (tmp_path / 'ser.py').write_text(SERIALISE)
    name = f'falcon-test-cdr-{uuid.uuid4().hex[:8]}'
    try:
        r = subprocess.run(['docker', 'run', '--rm', '--name', name, '-v', f'{tmp_path}:/work:ro', SB.IMAGE,
                            'python3', '/work/ser.py', '/work/cases.json'], capture_output=True, text=True,
                           timeout=180)
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
    assert r.returncode == 0, r.stderr[-2000:]
    ros = json.loads(r.stdout.strip().splitlines()[-1])
    for k, (t, m) in cases.items():
        mine, theirs = R.encode(t, m), bytes.fromhex(ros[k])
        # padding bytes are not initialised by rclpy's serialiser (seen: 0x76 after a bool), so the bytes are compared
        # where they carry data: the same length, the same fields read back from either, and our encoding of what was
        # read from rclpy's bytes is ours exactly
        # rclpy may end a message with up to 3 more bytes (to a multiple of 4: GearReport, not Control)
        assert 0 <= len(theirs) - len(mine) <= 3 and not any(theirs[len(mine):]), k
        assert R.decode(t, theirs) == R.decode(t, mine), k
        assert R.encode(t, R.decode(t, theirs)) == mine, k


def _free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.mark.parametrize('control', ['autoware', 'ackermann', 'carla'])
def test_closed_loop_over_dds(project, tmp_path, control):
    """The fake-carla worker, Falcon's bridge node and the follower over real DDS: every step's control arrives
    within its deadline, the follower stops short of the stalled car, and all of it is recorded."""
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    from test_stack import stack_rig
    stack_rig(project)
    port = _free_port()
    domain = 100 + int(uuid.uuid4().int % 100)
    cfg = ST.stack_config(project, 'world-town10', 'rig-default', str(tmp_path / 'run'), 2310, 8310, 'ros2',
                          SC.builtin('stalled_vehicle'), seed=3,
                          values={'speed': 30.0, 'distance': 60.0, 'kind': 'car'},
                          bridge={'port': port, 'control': control, 'deadline_s': 2.0, 'start_wait_s': 60.0,
                                  'connect_wait_s': 90.0})
    os.makedirs(cfg['out'])
    p = os.path.join(cfg['out'], 'stack.json')
    with open(p, 'w') as fh:
        json.dump(cfg, fh)
    tag = uuid.uuid4().hex[:8]
    names = [f'falcon-test-{tag}-bridge', f'falcon-test-{tag}-follower']
    bridge = SB.bridge_command(port, name=names[0], domain_id=domain)
    follower = SB.follower_command(control, 30.0 / 3.6, name=names[1], domain_id=domain)
    procs = []
    try:
        worker = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'stack', p],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())
        for argv in (bridge, follower):
            procs.append(subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        out, err = worker.communicate(timeout=600)
        time.sleep(1.0)
    finally:
        for n in names:
            subprocess.run(['docker', 'stop', '-t', '5', n], capture_output=True, timeout=60)
        logs = [pr.communicate(timeout=60)[0] for pr in procs]
        for n in names:
            subprocess.run(['docker', 'rm', '-f', n], capture_output=True, timeout=60)
    assert worker.returncode == P.OK, err[-3000:] + '\n'.join(logs)
    run = json.load(open(os.path.join(cfg['out'], 'run.json')))
    br = run['bridge']
    n = run['performance']['steps']
    assert n == 280 and br['missed'] == 0 and br['accepted'] == n, br
    assert run['bridge_node']['ros_distro'] == 'humble' and br['control']['type'] == SB.CONTROL_TYPES[control][0]
    ov = run['scenario']['criticality']['overall']
    assert not ov['collision'] and ov['min_distance_m'] < 15.0
    line = next((x for x in logs[1].splitlines() if x.startswith('{"follower"')), None)
    assert line, logs[1][-3000:]
    summary = json.loads(line)['follower']
    assert summary['steps'] >= n and summary['braking'] > 0
