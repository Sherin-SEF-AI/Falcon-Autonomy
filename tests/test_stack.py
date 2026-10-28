"""Stack testing without ROS: message encoding (byte-level CDR), the bridge link and the control law, the stack worker
in closed loop against the fake carla — with the bridge node's own logic (BridgeCore) and the test follower's decisions
(Brain) on an in-process stand-in for rclpy — suites (resume, results, intervals), replay, and the safety-case report.
ROS 2 itself (the falcon-ros2-bridge:humble container) is exercised by tests/test_stack_ros2.py (marker ros2)."""
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading

import numpy as np
import pytest

from conftest import ROOT, worker_env
from falcon_autonomy.core import mcap_bag as B
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import ros2msg as R
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import stack as ST
from falcon_autonomy.core import stackbridge as SB

sys.path.insert(0, os.path.join(ROOT, 'falcon_autonomy', 'ros2'))
import falcon_bridge_node as NODE      # noqa: E402  (standalone: no rclpy at import)
import follower_stack as FOLLOWER     # noqa: E402


# --------------------------------------------------------------------------------------------- messages and CDR
def test_known_cdr_bytes():
    # rosgraph_msgs/Clock {sec 1, nanosec 2}: header 00 01 00 00, int32, uint32
    assert R.encode('rosgraph_msgs/msg/Clock', R.clock(1_000_000_002)) == bytes.fromhex('00010000' '01000000'
                                                                                       '02000000')
    # autoware Control: two Time (16 bytes), Lateral (Time, Time, 2 float32, bool -> pad to 4), Longitudinal
    b = R.encode('autoware_control_msgs/msg/Control', {
        'stamp': {'sec': 3, 'nanosec': 4}, 'lateral': {'steering_tire_angle': 0.5},
        'longitudinal': {'velocity': 2.0, 'acceleration': -1.0, 'is_defined_acceleration': True}})
    assert b[:4] == b'\x00\x01\x00\x00' and struct.unpack_from('<iI', b, 4) == (3, 4)
    o = 4 + 16 + 16                                     # Control.stamp, control_time, Lateral.stamp, control_time
    assert struct.unpack_from('<ff?', b, o) == (0.5, 0.0, False)
    o2 = o + 9 + 3 + 16                                 # bool, pad to 4, Longitudinal stamps
    assert struct.unpack_from('<fff??', b, o2) == (2.0, -1.0, 0.0, True, False) and len(b) == o2 + 14
    # big-endian CDR decodes too
    be = bytes.fromhex('00000000') + struct.pack('>iI', 7, 9)
    assert R.decode('rosgraph_msgs/msg/Clock', be) == {'clock': {'sec': 7, 'nanosec': 9}}


@pytest.mark.parametrize('typ,msg', [
    ('autoware_control_msgs/msg/Control', {'stamp': {'sec': 1, 'nanosec': 2}, 'lateral': {'steering_tire_angle': -0.2},
                                           'longitudinal': {'velocity': 5.5, 'acceleration': 0.25,
                                                            'is_defined_acceleration': True}}),
    ('ackermann_msgs/msg/AckermannDriveStamped', {'header': B.header(123456789, 'base_link'),
                                                  'drive': {'steering_angle': 0.1, 'speed': -2.0}}),
    ('carla_msgs/msg/CarlaEgoVehicleControl', {'header': B.header(5, ''), 'throttle': 0.5, 'steer': -0.25,
                                               'brake': 0.0, 'hand_brake': True, 'reverse': False, 'gear': -1}),
    ('autoware_vehicle_msgs/msg/VelocityReport', R.velocity_report(3.0, -0.1, 0.05, 9)),
    ('autoware_vehicle_msgs/msg/SteeringReport', R.steering_report(0.1, 9)),
    ('autoware_vehicle_msgs/msg/GearReport', R.gear_report(True, 9)),
    ('autoware_vehicle_msgs/msg/ControlModeReport', R.control_mode_report(9)),
    ('nav_msgs/msg/Odometry', R.odometry({'x': 1.0, 'y': 2.0, 'z': 0.1, 'yaw': 30.0}, (2.0, 1.0, 0.0), (0, 0, 10.0),
                                         10 ** 9)),
    ('sensor_msgs/msg/Imu', R.imu((0.1, 0.2, 9.81), (0.01, 0.02, 0.03), 5, 'imu_link', 0.01, 0.001)),
    ('sensor_msgs/msg/NavSatFix', R.navsatfix(12.97, 77.59, 920.0, 5, 'gnss_link', {'noise_lat_stddev': 1e-6})),
])
def test_round_trip_against_the_encoder(typ, msg):
    b = R.encode(typ, msg)
    d = R.decode(typ, b)
    assert R.encode(typ, d) == b


def test_pointcloud_image_and_camera_info():
    pts = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, -5.0, 6.0, 0.25]], np.float32)
    m = R.pointcloud(pts, 7, 'lidar_link')
    d = R.decode('sensor_msgs/msg/PointCloud2', R.encode('sensor_msgs/msg/PointCloud2', m))
    a = np.frombuffer(d['data'], R.POINT_DTYPE)
    assert d['point_step'] == 16 and d['width'] == 2 and [f['name'] for f in d['fields']] == ['x', 'y', 'z',
                                                                                               'intensity']
    assert a['y'].tolist() == [-2.0, 5.0] and a['x'].tolist() == [1.0, 4.0]          # CARLA's y mirrored
    bgra = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    im = R.image_bgr(bgra.tobytes(), 3, 2, 1, 'cam_optical')
    assert im['encoding'] == 'bgr8' and im['step'] == 9 and im['data'] == bgra[:, :, :3].tobytes()
    ci = R.camera_info({'type': 'camera.pinhole', 'width': 1280, 'height': 720, 'fov': 90.0}, 1, 'cam_optical')
    assert ci['k'][0] == pytest.approx(640.0) and ci['k'][2] == pytest.approx(639.5) and ci['k'][5] == 359.5


def test_odometry_frames_follow_rep103():
    # CARLA yaw 90° (towards +y, CARLA's right) is −90° in the mirrored map frame; moving forward at 3 m/s
    o = R.odometry({'x': 0.0, 'y': 10.0, 'z': 0.0, 'yaw': 90.0}, (0.0, 3.0, 0.0), (0.0, 0.0, 0.0), 0)
    q = o['pose']['pose']['orientation']
    yaw = math.atan2(2 * (q['w'] * q['z'] + q['x'] * q['y']), 1 - 2 * (q['y'] ** 2 + q['z'] ** 2))
    assert yaw == pytest.approx(-math.pi / 2) and o['pose']['pose']['position']['y'] == -10.0
    assert o['twist']['twist']['linear']['x'] == pytest.approx(3.0) and abs(o['twist']['twist']['linear']['y']) < 1e-9
    # a right turn in CARLA (yaw rate +) is a negative yaw rate in ROS
    o = R.odometry({'x': 0, 'y': 0, 'z': 0, 'yaw': 0.0}, (1.0, 0, 0), (0, 0, 30.0), 0)
    assert o['twist']['twist']['angular']['z'] == pytest.approx(-math.radians(30.0))


def test_control_law():
    m = SB.ControlMapper(70.0)
    u = m(R.command_of('autoware_control_msgs/msg/Control', {
        'stamp': {'sec': 0, 'nanosec': 0}, 'lateral': {'steering_tire_angle': math.radians(35.0)},
        'longitudinal': {'velocity': 10.0, 'acceleration': 0.0, 'is_defined_acceleration': True}}), 9.0)
    assert u['steer'] == pytest.approx(-0.5) and u['throttle'] == pytest.approx(1.0 / 3.0) and u['brake'] == 0
    u = m({'kind': 'target', 'steer_rad': 0.0, 'speed': 0.0, 'accel': -2.0, 'stamp_ns': 0}, 3.0)
    assert u['brake'] == pytest.approx((2.0 + 3.0) / 8.0) and u['throttle'] == 0
    assert m({'kind': 'target', 'steer_rad': 0, 'speed': 0, 'accel': None, 'stamp_ns': 0}, 0.0)['brake'] == 1.0
    assert m({'kind': 'target', 'steer_rad': 0, 'speed': -2, 'accel': None, 'stamp_ns': 0}, 0.0)['reverse']
    u = m(R.command_of('carla_msgs/msg/CarlaEgoVehicleControl', {'header': B.header(0, ''), 'throttle': 2.0,
                                                                 'steer': 0.3, 'brake': 0.0, 'hand_brake': False,
                                                                 'reverse': False, 'gear': 0,
                                                                 'manual_gear_shift': False}), 0.0)
    assert u['throttle'] == 1.0 and u['steer'] == 0.3
    assert m.tyre_angle(-0.5) == pytest.approx(math.radians(35.0))
    with pytest.raises(ValueError):
        R.command_of('sensor_msgs/msg/Imu', {})


def test_deadlines_accept_only_the_steps_own_control():
    d = SB.Deadlines()
    assert not d.accept(5, 4, 100, 0)                   # tagged for an earlier step
    assert not d.accept(5, 5, 100, 50)                  # stamped before the step: stale
    assert d.accept(5, 5, 100, 0) and d.accept(5, 5, 100, 100)
    d.miss(6)
    r = d.report()
    assert r['accepted'] == 2 and r['stale'] == 2 and r['missed'] == 1 and r['missed_steps'] == [6]


def test_topic_plan_and_checks(project):
    rig = dict(project.doc['rigs'][0])
    rig['sensors'] = rig['sensors'] + [dict(id='l1', name='top', type='lidar', pose={}),
                                       dict(id='i1', name='imu', type='imu', pose={})]
    plan = SB.topic_plan(rig, SB.bridge_defaults())
    topics = {e['topic']: e['type'] for e in plan}
    assert topics['/falcon/ego/front/image'] == 'sensor_msgs/msg/Image'
    assert topics['/falcon/ego/top/points'] == 'sensor_msgs/msg/PointCloud2'
    assert topics['/vehicle/status/velocity_status'] == 'autoware_vehicle_msgs/msg/VelocityReport'
    ren = SB.autoware_remaps(rig)
    plan = SB.topic_plan(rig, dict(SB.bridge_defaults(), topics=ren))
    assert '/sensing/lidar/top/pointcloud_raw' in {e['topic'] for e in plan}
    plan = SB.topic_plan(rig, dict(SB.bridge_defaults(), ego_tf=False, odometry=False, autoware_status=False))
    assert not {'/tf', '/falcon/ego/odometry', '/vehicle/status/gear_status'} & {e['topic'] for e in plan}
    assert SB.check_bridge(dict(SB.bridge_defaults(), port=2000, control='x', on_miss='no'))
    assert not SB.check_bridge(SB.bridge_defaults())
    argv = SB.bridge_command(2330, domain_id=7)
    assert argv[:2] == ['docker', 'run'] and 'ROS_DOMAIN_ID=7' in argv and argv[-2:] == ['--connect',
                                                                                         '127.0.0.1:2330']


def test_node_framing_is_the_live_links():
    from falcon_autonomy.core import livelink
    h = {'t': 'step', 'step': 3, 'msgs': []}
    assert NODE.pack(h, b'xy') == livelink.pack(h, b'xy')


def test_follower_brain_brakes_for_points_in_its_corridor():
    b = FOLLOWER.Brain(speed=10.0)
    delta, target, acc, gap = b.act(0.0, 0.0, 0.0, 10.0, np.array([[15.0, 0.2, 1.0], [8.0, 5.0, 1.0]]))
    assert gap == pytest.approx(12.4) and target < 10.0 and acc < 0 and abs(delta) < 1e-9
    delta, target, acc, gap = b.act(5.0, 1.0, 0.0, 5.0, None)              # off its line to the left: steer right
    assert gap is None and target == 10.0 and acc > 0 and delta < 0
    _, _, acc, _ = b.act(5.0, 0.0, 0.0, 5.0, np.array([[5.0, 0.0, 1.0]]))
    assert acc == -8.0


# ------------------------------------------------------------------------------------- an in-process ROS stand-in
class Bus:
    """rclpy's part in the loop, in this process: publishers deliver CDR bytes to the stack; the stack's control
    goes to the node's raw subscription."""

    def __init__(self, speed=8.0, control='autoware', silent_after=None, stale_every=0):
        self.speed, self.control_kind, self.silent_after, self.stale_every = speed, control, silent_after, stale_every
        self.types, self.counts, self.cb = {}, {}, None
        self.brain = FOLLOWER.Brain(speed)
        self.mount = {}
        self.points = None
        self.steps = 0
        self.first = {}

    # adapter
    def publisher(self, topic, typ, qos):
        self.types[topic] = typ
        return lambda data, t=topic: self.deliver(t, data)

    def subscribe(self, topic, typ, cb):
        self.cb = cb
        return True, ''

    def info(self):
        return {'ros_distro': 'none (test)', 'rmw': 'in-process'}

    # the stack
    def deliver(self, topic, data):
        typ = self.types[topic]
        self.counts[topic] = self.counts.get(topic, 0) + 1
        m = R.decode(typ, data)
        self.first.setdefault(topic, m)
        if topic == '/tf_static':
            for t in m['transforms']:
                tr = t['transform']
                q = tr['rotation']
                self.mount[t['child_frame_id']] = (np.array([tr['translation'][k] for k in 'xyz']),
                                                   FOLLOWER.quat_matrix((q['x'], q['y'], q['z'], q['w'])))
        elif typ == 'sensor_msgs/msg/PointCloud2':
            a = np.frombuffer(m['data'], R.POINT_DTYPE)
            P_ = np.stack([a['x'], a['y'], a['z']], 1).astype(float)
            t, Rm = self.mount[m['header']['frame_id']]
            self.points = P_ @ Rm.T + t
        elif typ == 'nav_msgs/msg/Odometry':
            self.steps += 1
            if self.silent_after is not None and self.steps > self.silent_after:
                return
            p, q = m['pose']['pose']['position'], m['pose']['pose']['orientation']
            yaw = math.atan2(2 * (q['w'] * q['z'] + q['x'] * q['y']), 1 - 2 * (q['y'] ** 2 + q['z'] ** 2))
            v = math.hypot(m['twist']['twist']['linear']['x'], m['twist']['twist']['linear']['y'])
            delta, target, acc, _ = self.brain.act(p['x'], p['y'], yaw, v, self.points)
            st = m['header']['stamp']
            if self.stale_every and self.steps % self.stale_every == 0:     # one stale message first
                old = {'sec': st['sec'] - 1, 'nanosec': st['nanosec']}
                self.cb(R.encode('autoware_control_msgs/msg/Control', {'stamp': old}))
            self.cb(R.encode('autoware_control_msgs/msg/Control', {
                'stamp': st, 'lateral': {'stamp': st, 'steering_tire_angle': delta},
                'longitudinal': {'stamp': st, 'velocity': target, 'acceleration': acc,
                                 'is_defined_acceleration': True}}))


def serve(port, bus, result):
    sock = NODE.connect('127.0.0.1', port, 60.0)
    if sock is None:
        result['code'] = 'no worker'
        return
    result['code'] = NODE.serve_one(sock, bus, lambda m: result.setdefault('log', []).append(m))


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def stack_rig(project):
    rig = project.doc['rigs'][0]
    rig['sensors'][0].update(width=64, height=36, labels=[])
    rig['sensors'] += [{'id': 'lid', 'name': 'top', 'type': 'lidar', 'pose': {'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0,
                                                                              'pitch': 0, 'yaw': 0},
                        'channels': 8, 'range': 60.0, 'points_per_second': 40000, 'rotation_frequency': 20.0,
                        'upper_fov': 5.0, 'lower_fov': -20.0, 'horizontal_fov': 360.0, 'noise_stddev': 0.0},
                       {'id': 'imu', 'name': 'imu', 'type': 'imu', 'pose': {'x': 0, 'y': 0, 'z': 1, 'roll': 0,
                                                                             'pitch': 0, 'yaw': 0}},
                       {'id': 'gnss', 'name': 'gnss', 'type': 'gnss', 'pose': {'x': 0, 'y': 0, 'z': 1.5, 'roll': 0,
                                                                                'pitch': 0, 'yaw': 0}}]
    return rig


def run_worker(cfg, tmp_path, name='stack.json'):
    os.makedirs(cfg['out'], exist_ok=True)              # as sim.stacksuite.submit_stack: the config in its folder
    p = os.path.join(cfg['out'], 'stack.json')
    with open(p, 'w') as fh:
        json.dump(cfg, fh)
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'stack', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=300)
    run = json.loads(open(os.path.join(cfg['out'], 'run.json')).read())
    return r, run


def ros2_config(project, tmp_path, scenario='stalled_vehicle', values=None, **bridge):
    stack_rig(project)
    port = free_port()
    b = dict({'port': port, 'deadline_s': 5.0, 'start_wait_s': 30.0, 'connect_wait_s': 30.0}, **bridge)
    cfg = ST.stack_config(project, 'world-town10', 'rig-default', str(tmp_path / 'run'), 2310, 8310, 'ros2',
                          SC.builtin(scenario) if scenario else None, seed=3, values=values, seconds=2.0, bridge=b)
    return cfg, port


def test_closed_loop_through_the_bridge_logic(project, tmp_path):
    """The worker, the node's logic and the follower's decisions in closed loop: the stack brakes for the stalled
    car its lidar sees and stops short of it; every step's control arrived in time; topics and frames as planned."""
    cfg, port = ros2_config(project, tmp_path, values={'speed': 30.0, 'distance': 60.0, 'kind': 'car'})
    bus, res = Bus(speed=30.0 / 3.6), {}
    th = threading.Thread(target=serve, args=(port, bus, res), daemon=True)
    th.start()
    r, run = run_worker(cfg, tmp_path)
    th.join(30)
    assert r.returncode == P.OK, r.stderr + json.dumps(run.get('error'))
    br = run['bridge']
    n = run['performance']['steps']
    assert n == 280 and br['missed'] == 0 and br['accepted'] == n and br['stale'] == 0
    assert res['code'] == 0 and bus.steps == n
    # every planned topic was published once per step (the lidar: one full sweep per step at 20 Hz), /tf_static once
    assert bus.counts['/tf_static'] == 1 and bus.counts['/clock'] == n
    for t in ('/falcon/ego/front/image', '/falcon/ego/top/points', '/falcon/ego/imu/imu', '/falcon/ego/gnss/fix',
              '/falcon/ego/odometry', '/vehicle/status/velocity_status'):
        assert bus.counts[t] == n, t
    img = bus.first['/falcon/ego/front/image']
    assert (img['width'], img['height'], img['encoding']) == (64, 36, 'bgr8')
    assert img['header']['frame_id'] == 'front_optical'
    assert bus.first['/clock']['clock'] == img['header']['stamp']             # one time base: the frame's
    sc = run['scenario']
    ov = sc['criticality']['overall']
    assert not ov['collision'] and sc['checks'][0] == {'name': 'no collision', 'ok': True, 'detail': 'none'}
    assert 2.0 < ov['min_distance_m'] < 10.0                  # it stopped short of the stalled car
    frames = [json.loads(x) for x in open(tmp_path / 'run' / 'frames.jsonl')]
    assert frames[0]['control']['throttle'] == 0 and frames[0]['control']['brake'] == 0      # the first tick coasts
    assert max(f['control']['brake'] for f in frames) > 0.5 and frames[-1]['speed_mps'] < 0.5
    assert set(br['costs_ms']) >= {'tick_ms', 'encode_ms', 'send_ms', 'publish_ms', 'wait_ms'}


def test_missed_deadlines_and_stale_controls_are_recorded(project, tmp_path):
    cfg, port = ros2_config(project, tmp_path, 'lead_brake', values={'speed': 30.0, 'gap': 20.0, 'decel': 4.0,
                                                                     'brake_at': 3.0},
                            deadline_s=0.05, on_miss='brake', max_missed=0)
    bus, res = Bus(speed=30 / 3.6, silent_after=100, stale_every=10), {}
    th = threading.Thread(target=serve, args=(port, bus, res), daemon=True)
    th.start()
    r, run = run_worker(cfg, tmp_path)
    th.join(30)
    assert r.returncode == P.OK, r.stderr
    n = run['performance']['steps']
    br = run['bridge']
    assert br['missed'] == n - 100 and br['missed_steps'][0] == 100 and br['stale'] == 10
    chk = run['scenario']['checks'][-1]
    assert chk['name'] == 'missed deadlines ≤ 0' and not chk['ok'] and not run['scenario']['passed']
    frames = [json.loads(x) for x in open(tmp_path / 'run' / 'frames.jsonl')]
    assert frames[101]['bridge']['missed'] and frames[102]['control']['brake'] == 1.0      # on_miss: brake


def test_port_zero_is_for_the_runner(project, tmp_path):
    cfg, _ = ros2_config(project, tmp_path, port=0)
    r, run = run_worker(cfg, tmp_path)
    assert r.returncode == P.FAILED and 'the runner picks a free port' in run['error']


def test_no_bridge_node_fails_with_a_reason(project, tmp_path):
    cfg, _ = ros2_config(project, tmp_path, connect_wait_s=1.0)
    r, run = run_worker(cfg, tmp_path)
    assert r.returncode == P.FAILED and 'no bridge node connected' in run['error']


def test_the_bridge_reports_a_control_type_it_cannot_take(project, tmp_path):
    cfg, port = ros2_config(project, tmp_path)

    class NoType(Bus):
        def subscribe(self, topic, typ, cb):
            return False, f'{typ} is not installed here'
    res = {}
    th = threading.Thread(target=serve, args=(port, NoType(), res), daemon=True)
    th.start()
    r, run = run_worker(cfg, tmp_path)
    th.join(30)
    assert r.returncode == P.FAILED and 'is not installed here' in run['error']


def test_baseline_and_replay_repeat_bit_for_bit(project, tmp_path):
    """A follower baseline run, then its replay from the recorded controls: the same trajectories."""
    stack_rig(project)
    cfg = ST.stack_config(project, 'world-town10', 'rig-default', str(tmp_path / 'base'), 2310, 8310, 'follower',
                          SC.builtin('lead_brake'), seed=4)
    r, run = run_worker(cfg, tmp_path, 'a.json')
    assert r.returncode == P.OK, r.stderr
    assert run['driver'] == 'follower' and 'bridge' not in run and run['scenario']['checks']
    from falcon_autonomy.sim.stacksuite import compare_trajectories, replay_config
    rcfg = replay_config(str(tmp_path / 'base'), str(tmp_path / 'replay'))
    r, rrun = run_worker(rcfg, tmp_path, 'b.json')
    assert r.returncode == P.OK, r.stderr
    cmp = compare_trajectories(str(tmp_path / 'base'), str(tmp_path / 'replay'))
    assert cmp['same_actors'] and cmp['max_m'] == 0.0
    assert rrun['scenario']['criticality'] == run['scenario']['criticality']
    assert rrun['provenance']['inputs']['replay_of']['sha256']


def test_a_baseline_needs_a_scenario(project, tmp_path):
    cfg = ST.stack_config(project, 'world-town10', 'rig-default', str(tmp_path / 'f'), 2310, 8310, 'follower')
    r, run = run_worker(cfg, tmp_path)
    assert r.returncode == P.FAILED and 'needs a scenario' in run['error']


# ------------------------------------------------------------------------------------------------------- suites
def test_wilson_interval():
    lo, hi = ST.wilson(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-3)       # the textbook value
    lo, hi = ST.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert all(math.isnan(x) for x in ST.wilson(0, 0))


def test_suite_validation(project):
    doc = ST.new_suite('s', 'ros2', ['lead_brake', 'nope'])
    doc['stack']['bridge'] = {'port': 2005}
    doc['entries'][0]['values'] = {'bogus': 1}
    errs = ST.validate_suite(doc, project)
    assert any('nope' in e for e in errs) and any('bogus' in e for e in errs) and any('2005' in e for e in errs)
    assert not ST.validate_suite(ST.new_suite('ok'), project)


def test_suite_runs_resumes_and_reports(project, tmp_path, fake_carla):
    """A follower suite through the CLI: results, table, intervals; a second run resumes (nothing repeated); the
    safety-case report from it, with an ODD and a study."""
    project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)
    suite = ST.new_suite('baseline', 'follower', ['lead_brake', 'stalled_vehicle'], [0, 1])
    suite['entries'][1]['values'] = {'distance': 40.0, 'speed': 60.0}
    suite['criteria']['min_ttc_s'] = 2.5
    sp = tmp_path / 'suite.json'
    sp.write_text(json.dumps(suite))
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'stack', '--fake-carla', fake_carla, *a],
                              capture_output=True, text=True, env=env, timeout=600)
    d = tmp_path / 'suite-out'
    r = cli('suite', 'run', path, str(sp), '--dir', str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    rows = ST.read_results(str(d))
    assert len(rows) == 4 and all(x['status'] == 'ok' for x in rows)
    res = json.load(open(d / 'results.json'))
    assert res['overall']['runs'] == 4 and res['planned'] == 4
    for g in res['scenarios'].values():
        lo, hi = g['pass_rate_ci95']
        assert 0 <= lo <= g['pass_rate'] <= hi <= 1 and g['interval'].startswith('Wilson')
    assert (d / 'results.csv').read_text().splitlines()[0].startswith('scenario,seed,status,passed')
    runj = json.load(open(d / 'run.json'))
    assert runj['provenance']['inputs']['suite']['sha256'] and len(runj['runs']) == 4
    stack_json = json.load(open(os.path.join(rows[0]['run'], 'stack.json')))
    assert stack_json['scenario']['pass']['min_ttc_s'] == 2.5
    mtimes = {x['run']: os.path.getmtime(os.path.join(x['run'], 'run.json')) for x in rows}
    r = cli('suite', 'run', path, str(sp), '--dir', str(d))
    assert r.returncode == 0 and len(open(d / 'results.jsonl').readlines()) == 4
    assert all(os.path.getmtime(os.path.join(k, 'run.json')) == v for k, v in mtimes.items())
    # the safety case, with a (hand-made) study holding one failing parameter set
    st = tmp_path / 'study'
    st.mkdir()
    (st / 'study.json').write_text(json.dumps({'scenario': {'id': 'lead_brake'}}))
    (st / 'results.jsonl').write_text('\n'.join(json.dumps({'status': 'ok', 'values': {'gap': g, 'decel': 8.0},
                                                            'objective': o, 'run': f'/x/{g}'})
                                                for g, o in ((8.0, 1.0), (30.0, 0.1), (20.0, 0.2))) + '\n')
    (st / 'search-bayes.json').write_text(json.dumps({'threshold': 0.5}))
    odd = tmp_path / 'odd.json'
    assert cli('odd', 'new', '--out', str(odd)).returncode == 0
    out = tmp_path / 'report'
    r = cli('report', str(d), '--odd', str(odd), '--out', str(out), '--study', str(st))
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(out / 'safety-case.json'))
    html_ = (out / 'safety-case.html').read_text()
    assert 'not a certification claim' in html_.lower() and '<svg' in html_ and 'http' not in html_.replace(
        'http-equiv', '')
    cov = rep['coverage']['dimensions']
    assert cov['scenario']['counts'][cov['scenario']['labels'].index('lead_brake')] == 2
    assert 'cut_in' in cov['scenario']['gaps'] and cov['actors']['counts'][0] >= 2      # car: lead_brake's lead
    assert cov['handedness']['gaps'] == ['LHT']
    kinds = {x['kind'] for x in rep['residual_risks']}
    assert {'odd_gap', 'simulation_limit', 'triggering_condition'} <= kinds
    trig = next(t for t in rep['triggering_conditions'] if t['source'].startswith('study'))
    assert trig['failures'] == 1 and trig['parameters']['gap'] == {'tried': [8.0, 30.0], 'failing': [8.0, 8.0]}
    assert rep['provenance']['inputs']['odd']['sha256'] and rep['limits']['sotif_not_covered']


def test_odd_validation_and_coordinates():
    from falcon_autonomy.core import safetycase as SCASE
    odd = SCASE.example_odd()
    assert not SCASE.validate_odd(odd)
    bad = dict(odd, dimensions=dict(odd['dimensions'], x={'kind': 'range', 'lo': 0, 'hi': 10, 'bins': [5, 1]}))
    assert any(e.startswith('x:') for e in SCASE.validate_odd(bad))
    cfg = {'town': 'Town10HD_Opt', 'handedness': 'RHT', 'weather': {'precipitation': 70.0, 'sun_altitude_angle': 45.0},
           'traffic': {'vehicles': 3}, 'scenario': SC.sample(SC.builtin('cut_in'), 1)}
    c = SCASE.run_coordinates(cfg)
    assert c['road_type'] == ['straight', 'multi_lane'] and c['actors'] == ['car'] and c['scenario'] == ['cut_in']
    cov = SCASE.coverage(odd, [c])
    assert cov['dimensions']['precipitation']['outside'] == 1                      # 70 % is beyond the ODD's 60
    assert cov['dimensions']['sun_altitude_deg']['counts'] == [0, 1, 0]


def test_cli_topics_and_bridge(project, tmp_path):
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'stack', 'topics', path], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and '/falcon/ego/front/image' in r.stdout and '/control/command/control_cmd' in r.stdout
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'stack', 'bridge', '--port', '2331'],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and 'falcon_bridge_node.py --connect 127.0.0.1:2331' in r.stdout
