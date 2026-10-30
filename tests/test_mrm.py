"""Mid-run sensor faults and the minimal-risk manoeuvre: the new timed fault kinds (core.faults), the test stack's
health monitor and MRM (ros2/degraded.py), the MRM verifier on constructed runs (core.mrm), and stack runs against
the fake carla — the built-in follower with and without its degraded mode, the ROS 2 follower's decisions through
the bridge logic — where the verifier passes when the stack reacts and fails when it does not; suites and the
safety case carry the result."""
import json
import math
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from conftest import ROOT
from falcon_autonomy.core import faults as F
from falcon_autonomy.core import mrm as M
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import stack as ST
from falcon_autonomy.ros2 import degraded as D

sys.path.insert(0, os.path.join(ROOT, 'falcon_autonomy', 'ros2'))
import follower_stack as FOLLOWER     # noqa: E402


# ------------------------------------------------------------------------------------------------- the fault kinds
def ring(n=3600):
    a = np.radians(np.arange(n) * 360.0 / n)
    return np.c_[10 * np.cos(a), 10 * np.sin(a), np.zeros(n), np.ones(n)].astype(np.float32)


def test_lidar_sector_loss_and_packet_drop():
    P_ = ring()
    f = [{'kind': 'sector_loss', 'from_deg': -30.0, 'to_deg': 30.0, 'onset_s': 2.0}]
    assert len(F.apply_points(P_, f, 1.0, np.random.default_rng(0))) == 3600          # before its onset
    out = F.apply_points(P_, f, 2.0, np.random.default_rng(0))
    az = np.degrees(np.arctan2(out[:, 1], out[:, 0]))
    assert len(out) == 3000 and not np.any((az >= -30) & (az < 30))
    wrap = F.apply_points(P_, [{'kind': 'sector_loss', 'from_deg': 170, 'to_deg': -170}], 0.0, None)
    assert len(wrap) == 3400                                                           # through 180°
    drop = F.apply_points(P_, [{'kind': 'packet_drop', 'rate': 0.5}], 0.0, np.random.default_rng(1))
    assert 1400 < len(drop) < 2200 and len(drop) % 10 == 0                             # whole 1° packets


def test_gnss_multipath_occlusion_and_dropout():
    f = [{'kind': 'multipath', 'jump_m': 8.0, 'period_s': 1.0, 'onset_s': 1.0}]
    assert F.gnss_offset(f, 0.5, 3, 'g') == (0.0, 0.0)
    a, b = F.gnss_offset(f, 1.2, 3, 'g'), F.gnss_offset(f, 1.9, 3, 'g')
    assert a == b and 4.0 <= math.hypot(*a) <= 8.0                                     # one jump per period
    assert F.gnss_offset(f, 2.1, 3, 'g') != a
    img = np.full((90, 160, 3), 200, np.uint8)
    out, done = F.apply_image(img, [{'kind': 'occlusion', 'cover': 0.4}], np.random.default_rng(0))
    share = float(np.mean(out.mean(axis=2) < 100))
    assert done == ['occlusion'] and 0.3 < share < 0.5
    assert F.silenced([{'kind': 'dropout', 'onset_s': 3.0}], 3.0) and not F.silenced([{'kind': 'dropout',
                                                                                       'onset_s': 3.0}], 2.9)
    assert F.delivery([{'kind': 'dropout', 'onset_s': 1.0}], 0, 1.5, None) == 'drop'
    assert F.validate([{'kind': 'sector_loss'}]) and F.validate([{'kind': 'occlusion', 'cover': 0.95}])
    assert F.applies_to('sector_loss', 'lidar') and not F.applies_to('sector_loss', 'camera.pinhole')
    assert F.applies_to('multipath', 'gnss') and F.applies_to('dropout', 'imu')


# ------------------------------------------------------------------------------------------------- the monitor
def test_monitor_detects_each_failure_and_latches():
    rng = np.random.default_rng(0)

    def sweep(cut=None, n=2000):
        a = rng.uniform(-math.pi, math.pi, n)
        P_ = np.c_[10 * np.cos(a), 10 * np.sin(a)]
        if cut:
            d = (np.degrees(a) + 360) % 360
            P_ = P_[~((d >= cut[0]) & (d < cut[1]))]
        return P_

    def healthy(m, t0=0.0, k=4):
        for i in range(k):
            m.lidar(t0 + 0.1 * i, sweep())
    m = D.Monitor(expect=('lidar',))
    healthy(m)
    assert m.check(0.3) == (False, None)
    assert m.check(0.3 + 0.36)[0] and 'nothing for' in m.degraded[1]                  # a silent lidar
    m = D.Monitor()
    healthy(m)
    m.lidar(0.4, sweep(n=600))
    assert m.check(0.4)[0] and 'below' in m.degraded[1]                                # packet drop
    m = D.Monitor()
    healthy(m)
    m.lidar(0.4, sweep(cut=(80, 140)))
    assert 'empty sector' in m.check(0.4)[1]
    m.lidar(0.5, sweep())
    assert m.check(0.5)[0]                                                            # latched
    m = D.Monitor(expect=('imu', 'gnss'))
    for i in range(5):
        m.imu(0.05 * i)
        m.gnss(0.05 * i, (0.5 * i, 0.0), (0.5 * i, 0.0))
    assert not m.check(0.2)[0]
    m.gnss(0.25, (2.5 + 6.0, 0.0), (2.5, 0.0))                                        # a 6 m jump
    assert 'jumped' in m.check(0.25)[1]
    m = D.Monitor(expect=('imu',))
    m.imu(0.0)
    assert m.check(0.3)[0] and m.degraded[1].startswith('imu')
    m = D.Monitor(expect=('camera',))
    bright = np.random.default_rng(1).integers(80, 255, (90, 160, 3), dtype=np.uint8)
    m.camera(0.0, bright)
    assert not m.check(0.0)[0]
    mud, _ = F.apply_image(bright, [{'kind': 'occlusion', 'cover': 0.5}], np.random.default_rng(2))
    m.camera(0.05, mud)
    assert 'occluded' in m.check(0.05)[1]
    m = D.Monitor(expect=('camera',))
    for i in range(11):
        m.camera(0.05 * i, bright)
    assert 'frozen' in m.check(0.55)[1]


def test_mrm_profile_is_jerk_limited_and_holds():
    mrm = D.MRM({'mrm_decel': 2.5, 'mrm_jerk': 4.0})
    a = [mrm.accel(t, 10.0)[0] for t in (0.0, 0.25, 0.5, 1.0)]
    assert a == pytest.approx([0.0, -1.0, -2.0, -2.5])
    assert mrm.accel(5.0, 0.05) == (0.0, True)
    assert D.throttle_brake(-2.0, False) == {'throttle': 0.0, 'brake': 0.25}
    assert D.throttle_brake(0.0, True)['brake'] == 1.0
    assert D.mrm_steps(10.0) == pytest.approx(2.5 / 4 + (10.0 - 0.5 * 2.5 * 2.5 / 4) / 2.5)


# ------------------------------------------------------------------------------------------------- the verifier
def frames_of(v0, onset, decel=None, jerk=4.0, react=0.4, T=14.0, dt=0.05, offset=0.0, drift=0.0):
    """A run's frames: cruising at v0, from onset + react a stop ramping (jerk) to decel, else cruising on."""
    rows, v, a = [], v0, 0.0
    for k in range(int(T / dt) + 1):
        t = k * dt
        if decel is not None and t >= onset + react and v > 0:
            a = max(a - jerk * dt, -decel)
            v = max(v + a * dt, 0.0)
        rows.append({'sim_t': round(t, 4), 'speed_mps': v, 'lane_offset_m': offset + drift * max(t - onset, 0.0),
                     'lane_width_m': 3.5, 'ego_width_m': 2.0, 'kerb_sign': 1.0})
    return rows


def test_a_controlled_stop_passes_and_each_failure_is_named():
    good = M.verify(frames_of(8.0, 3.0, decel=2.5), 3.0)
    assert good['passed'], good['checks']
    assert good['measured']['reaction_s'] <= 1.5 and good['measured']['standstill_s'] < good['measured']['window_s']
    none = M.verify(frames_of(8.0, 3.0), 3.0)
    failed = {c['name'] for c in none['checks'] if not c['ok']}
    assert not none['passed'] and any(n.startswith('reacts') for n in failed) and any(n.startswith('standstill')
                                                                                       for n in failed)
    harsh = M.verify(frames_of(8.0, 3.0, decel=8.0, jerk=200.0), 3.0)
    failed = {c['name'] for c in harsh['checks'] if not c['ok']}
    assert 'deceleration ≤ 4 m/s²' in failed and '|jerk| ≤ 10 m/s³' in failed
    late = M.verify(frames_of(8.0, 3.0, decel=2.5, react=3.0), 3.0)
    assert not late['passed'] and not late['checks'][0]['ok']
    out = M.verify(frames_of(8.0, 3.0, decel=2.5, drift=-0.3), 3.0)                    # drifts into the next lane
    assert not next(c for c in out['checks'] if c['name'].startswith('stays'))['ok']
    shoulder = M.verify(frames_of(8.0, 3.0, decel=2.5, drift=0.25), 3.0)              # pulls over to the kerb side
    lane = next(c for c in shoulder['checks'] if c['name'].startswith('stays'))
    assert lane['ok'] and 'shoulder' in lane['detail']
    hit = M.verify(frames_of(8.0, 3.0, decel=2.5), 3.0, collisions=[1.0, 6.0])
    assert not hit['passed'] and 'contact at 6.00 s' in hit['checks'][-1]['detail']
    restart = frames_of(8.0, 3.0, decel=2.5)
    for r in restart[-20:]:
        r['speed_mps'] = 1.0
    assert 'moved again' in M.verify(restart, 3.0)['checks'][1]['detail']


def test_fault_specs_are_validated_against_the_rig(project):
    rig = stack_rig(project)
    assert M.validate_faults([{'sensor': 'lid', 'kind': 'sector_loss', 'from_deg': 0, 'to_deg': 90,
                               'onset_s': 2.0, 'for_s': 3}], rig) == []
    bad = M.validate_faults([{'sensor': 'nope', 'kind': 'dropout', 'onset_s': 1},
                             {'sensor': 'lid', 'kind': 'occlusion', 'onset_s': 1},
                             {'sensor': 'imu', 'kind': 'mount_rotation', 'onset_s': 1},
                             {'sensor': 'gnss', 'kind': 'multipath'}], rig)
    assert len(bad) == 4 and 'no sensor' in bad[0] and 'cannot act on lid' in bad[1]
    with pytest.raises(ValueError, match='timed fault'):
        ST.stack_config(project, 'world-town10', 'rig-default', '/tmp/x', 2310, 8310, 'follower',
                        SC.builtin('stalled_vehicle'), mrm=True)


# ------------------------------------------------------------------------------------------------- stack runs
def stack_rig(project):
    from test_stack import stack_rig as rig_of
    return rig_of(project)


def follower_run(project, tmp_path, name, faults, degraded):
    from test_stack import run_worker
    stack_rig(project)
    cfg = ST.stack_config(project, 'world-town10', 'rig-default', str(tmp_path / name), 2310, 8310, 'follower',
                          SC.builtin('lead_brake'), seed=3, values=CRUISE, faults=faults, mrm=True,
                          degraded_mode=degraded)
    return run_worker(cfg, tmp_path)


LIDAR_DEAD = [{'sensor': 'lid', 'kind': 'dropout', 'onset_s': 3.0}]
CRUISE = {'speed': 30.0, 'gap': 30.0, 'decel': 3.0, 'brake_at': 100.0}     # the lead never brakes: both cruise


def test_the_follower_without_a_degraded_mode_fails_the_mrm(project, tmp_path):
    r, run = follower_run(project, tmp_path, 'plain', LIDAR_DEAD, False)
    assert r.returncode == P.OK, r.stderr[-2000:]
    v = run['mrm']
    assert not v['passed'] and v['onset_s'] == 3.0 and 'degraded' not in run
    assert not run['scenario']['passed'] and run['scenario']['checks'][-1]['name'].startswith('minimal risk')
    frames = [json.loads(x) for x in open(tmp_path / 'plain' / 'frames.jsonl')]
    assert frames[0].get('faults') is None and frames[-1]['faults'] == {'lid': ['dropout']}
    assert 'lane_offset_m' in frames[10] and frames[-1]['speed_mps'] > 7.0             # it drove on


@pytest.mark.parametrize('faults, why', [
    (LIDAR_DEAD, 'lidar: nothing for'),
    ([{'sensor': 'lid', 'kind': 'sector_loss', 'from_deg': -60.0, 'to_deg': 60.0, 'onset_s': 3.0}], 'empty sector'),
    ([{'sensor': 'imu', 'kind': 'dropout', 'onset_s': 3.0}], 'imu: nothing for'),
    ([{'sensor': 'gnss', 'kind': 'multipath', 'jump_m': 12.0, 'period_s': 1.0, 'onset_s': 3.0}], 'gnss: the fix'),
    ([{'sensor': 'cam-front', 'kind': 'occlusion', 'cover': 0.6, 'onset_s': 3.0}], 'camera'),
])
def test_the_follower_in_degraded_mode_stops_in_its_lane(project, tmp_path, faults, why):
    r, run = follower_run(project, tmp_path, 'degraded', faults, True)
    assert r.returncode == P.OK, r.stderr[-2000:]
    assert run['degraded']['t'] >= 3.0 and why in run['degraded']['why'], run.get('degraded')
    v = run['mrm']
    assert v['passed'], v['checks']
    assert v['measured']['reaction_s'] <= 1.5 and v['measured']['max_decel_mps2'] <= 4.0
    frames = [json.loads(x) for x in open(tmp_path / 'degraded' / 'frames.jsonl')]
    assert frames[-1]['speed_mps'] < 0.2 and frames[-1]['health'] == 'degraded' and frames[0]['health'] == 'ok'


class DegradedBus:
    """test_stack.Bus with the follower's degraded mode: the points, IMU and fixes go to its health monitor."""

    def __new__(cls, degraded):
        from test_stack import Bus
        from falcon_autonomy.core import ros2msg as R

        class _B(Bus):
            def __init__(self):
                super().__init__(speed=30.0 / 3.6)
                self.brain = FOLLOWER.Brain(30.0 / 3.6, degraded=degraded)

            def deliver(self, topic, data):
                typ = self.types[topic]
                m = R.decode(typ, data)
                st = (m.get('header') or {}).get('stamp')
                t = st['sec'] + st['nanosec'] * 1e-9 if st else None
                if typ == 'sensor_msgs/msg/PointCloud2':
                    a = np.frombuffer(m['data'], R.POINT_DTYPE)
                    self.brain.sense_points(t, np.stack([a['x'], a['y']], 1).astype(float))
                elif typ == 'sensor_msgs/msg/Imu':
                    self.brain.sense_imu(t)
                elif typ == 'nav_msgs/msg/Odometry':
                    self.t = t
                    self.odom = m['pose']['pose']['position']
                elif typ == 'sensor_msgs/msg/NavSatFix':
                    o = getattr(self, 'odom', {'x': 0.0, 'y': 0.0})
                    self.brain.sense_fix(t, m['latitude'], m['longitude'], (o['x'], o['y']))
                super().deliver(topic, data)
        return _B()


def test_the_ros2_follower_reacts_through_the_bridge_only_in_degraded_mode(project, tmp_path):
    """The bridge worker with the ROS 2 follower's own decisions (Brain) behind the node's logic: its lidar dies at
    3 s; with --degraded-mode it stops in its lane and the MRM check passes, without it it drives on and fails."""
    from test_stack import ros2_config, run_worker, serve
    import follower_stack
    out = {}
    for degraded in (False, True):
        cfg, port = ros2_config(project, tmp_path / str(degraded), scenario='lead_brake', values=CRUISE)
        cfg = dict(cfg, faults=LIDAR_DEAD, mrm={})
        cfg['steps'] = 280
        # the stand-in rclpy: Brain.act gets the step's simulation time as the real follower gives it
        bus, res = DegradedBus(degraded), {}
        orig = bus.brain.act
        bus.brain.act = lambda x, y, yaw, v, pts, t=None, b=bus: orig(x, y, yaw, v, pts, getattr(b, 't', None))
        th = threading.Thread(target=serve, args=(port, bus, res), daemon=True)
        th.start()
        r, run = run_worker(cfg, tmp_path)
        th.join(30)
        assert r.returncode == P.OK, r.stderr[-2000:]
        out[degraded] = run
    assert not out[False]['mrm']['passed'] and out[True]['mrm']['passed'], out[True]['mrm']['checks']
    assert follower_stack.Brain(1.0).monitor is None


def test_a_suite_reports_the_mrm_per_case_and_in_the_safety_case(project, tmp_path, fake_carla):
    stack_rig(project)
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)
    suite = ST.new_suite('mrm', 'follower', ['lead_brake'], [0, 1])
    suite['degraded_mode'] = True
    suite['entries'][0].update(values=CRUISE, mrm=True, faults=[{'sensor': 'lid', 'kind': 'dropout', 'onset_s': 2.0}])
    suite['entries'].append({'scenario': 'lead_brake', 'seeds': [0], 'values': CRUISE})
    sp = tmp_path / 'suite.json'
    sp.write_text(json.dumps(suite))
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'stack', '--fake-carla', fake_carla, *a],
                              capture_output=True, text=True, env=env, timeout=900)
    d = tmp_path / 'out'
    r = cli('suite', 'run', path, str(sp), '--dir', str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.load(open(d / 'results.json'))
    assert res['mrm']['overall']['runs'] == 2 and res['mrm']['overall']['passed'] == 2
    assert len(res['mrm']['cases']) == 2 and all(c['passed'] for c in res['mrm']['cases'])
    assert 'mrm_passed' in (d / 'results.csv').read_text().splitlines()[0]
    odd = tmp_path / 'odd.json'
    assert cli('odd', 'new', '--out', str(odd)).returncode == 0
    out = tmp_path / 'report'
    r = cli('report', str(d), '--odd', str(odd), '--out', str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(out / 'safety-case.json'))
    assert rep['mrm']['overall']['passed'] == 2
    html_ = (out / 'safety-case.html').read_text()
    assert 'Minimal-risk manoeuvre after sensor faults' in html_


def test_hil_session_streams_the_faults_and_judges_the_mrm(project, tmp_path):
    """A HIL session with the simulated device: the lidar goes silent and the camera gets mud at 1 s; the streamed
    data carry them (latency.jsonl), and the device's default stack — which has no degraded mode — fails the MRM."""
    from test_hil import _config, _run_worker
    out = str(tmp_path / 'hil')
    cfg = _config(project, out, 'raw-tcp', 'auto', 'udp', seconds=4.0,
                  faults=[{'sensor': 'lidar', 'kind': 'dropout', 'onset_s': 1.0},
                          {'sensor': 'camera', 'kind': 'occlusion', 'cover': 0.5, 'onset_s': 1.0}], mrm=True)
    assert {f['sensor'] for f in cfg['faults']} == {cfg['camera']['sensor']['id'], cfg['lidar']['sensor']['id']}
    r = _run_worker(cfg, out)
    assert r.returncode == P.OK, r.stderr[-3000:]
    run = json.load(open(os.path.join(out, 'run.json')))
    recs = [json.loads(x) for x in open(os.path.join(out, 'latency.jsonl'))]
    lid = cfg['lidar']['sensor']['id']
    assert recs[0].get('faults') is None and recs[-1]['faults'][lid] == ['dropout'] and recs[-1]['lidar_points'] == 0
    assert run['mrm']['onset_s'] == 1.0 and run['mrm']['passed'] is False
    with pytest.raises(ValueError, match='timed fault'):
        _config(project, out, 'raw-tcp', 'auto', 'udp', mrm=True)


def test_the_cli_takes_faults_the_mrm_and_the_degraded_mode(project, tmp_path, fake_carla):
    from falcon_autonomy.sim.stacksuite import parse_fault
    assert parse_fault('lid:sector_loss:3:from_deg=-60,to_deg=60,for=5') == {
        'sensor': 'lid', 'kind': 'sector_loss', 'onset_s': 3.0, 'from_deg': -60.0, 'to_deg': 60.0, 'for_s': 5.0}
    with pytest.raises(ValueError):
        parse_fault('lid:dropout')
    stack_rig(project)
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'stack', '--fake-carla', fake_carla, *a],
                              capture_output=True, text=True, env=env, timeout=600)
    sp = tmp_path / 's.json'
    r = cli('suite', 'new', 'x', '--out', str(sp), '--scenarios', 'lead_brake', '--seeds', '0', '--fault',
            'imu:dropout:2', '--mrm', '--degraded-mode')
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(sp.read_text())
    assert doc['degraded_mode'] and doc['entries'][0]['mrm'] and doc['entries'][0]['faults'][0]['kind'] == 'dropout'
    out = tmp_path / 'one'
    r = cli('run', path, '--driver', 'follower', '--scenario', 'lead_brake', '--seed', '1', '--fault',
            'imu:dropout:3', '--mrm', '--degraded-mode', '--out', str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    run = json.load(open(out / 'run.json'))
    assert run['degraded']['why'].startswith('imu') and 'mrm' in run
