"""Hardware in the loop: the device-facing wire format (framing, UDP chunks, the H.264 SEI, PointCloud2 in CDR), the
emulated sensor clocks, the clock-offset estimate on synthetic clocks with a known offset and drift, the statistics,
the default stack, and whole sessions against the fake carla with a simulated device (the agent as its own process),
over each transport; deployment through stand-in ssh/scp programs that run here. The device agent in Python 3.10 (the
ros:humble-ros-base container: JetPack 6's Python and numpy) and its ROS 2 lidar are opt-in (marker ros2,
FALCON_ROS2=1)."""
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import FAKE, worker_env
from falcon_autonomy.core import hil as H
from falcon_autonomy.core import protocol as P
from falcon_autonomy.hil import device_agent as A


# ============================================================================================ wire format
def test_framing_round_trip_over_a_socket():
    a, b = socket.socketpair()
    msgs = [({'t': 'x', 'n': 1}, b''), ({'t': 'y', 'u': 'é'}, bytes(range(256)) * 100)]
    for h, p in msgs:
        a.sendall(A.pack(h, p))
    for h, p in msgs:
        h2, p2 = A.read_msg(b)
        assert h2 == h and bytes(p2) == p
    assert A.unpack(A.pack({'t': 'z'}, b'abc'))[0] == {'t': 'z'}
    a.close()
    with pytest.raises(ConnectionError):
        A.read_msg(b)


def test_udp_chunks_reassemble_and_incomplete_messages_are_dropped():
    rng = np.random.default_rng(0)
    m1 = A.pack({'t': 'lidar', 'frame': 7}, rng.bytes(50_000))
    m2 = A.pack({'t': 'lidar', 'frame': 8}, rng.bytes(3_000))
    c1, c2 = A.chunks(7, m1), A.chunks(8, m2)
    assert all(len(c) <= A.CHUNK_BYTES for c in c1) and len(c1) == -(-len(m1) // (A.CHUNK_BYTES - A.CHUNK.size))
    ra = A.Reassembler()
    order = list(rng.permutation(len(c1)))
    got = [ra.feed(c1[i]) for i in order]
    assert got[-1] == m1 and all(g is None for g in got[:-1])          # out of order: still whole
    ra2 = A.Reassembler()
    for c in c1[:-1]:                                                    # one chunk of 7 lost
        assert ra2.feed(c) is None
    out = [ra2.feed(c) for c in c2]
    assert out[-1] == m2 and ra2.incomplete == 1 and not ra2.parts
    assert ra2.feed(b'short') is None and ra2.bad == 1


def test_sei_inserted_before_the_first_slice_and_read_back():
    aud, sps, pps = b'\x00\x00\x00\x01\x09\xf0', b'\x00\x00\x00\x01\x67\x42\x00\x1f', b'\x00\x00\x00\x01\x68\xce'
    idr = b'\x00\x00\x01\x65' + bytes([0x88, 0x84, 0x00, 0x00, 0x03, 0x01]) * 50
    au = aud + sps + pps + idr
    meta = {'f': 1234567, 's': 61.85, 'h': 61850000123, 'pad': 'x' * 300}     # > 255 bytes: multi-byte size coding
    out = A.insert_sei(au, meta)
    types = [t for *_, t in A.nal_units(out)]
    assert types == [9, 7, 8, 6, 5]                                      # AUD, SPS, PPS, SEI, IDR slice
    assert A.read_sei(out) == meta and A.read_sei(au) is None
    assert out.endswith(idr) and out.startswith(aud + sps + pps)
    # emulation prevention: no start code inside the SEI whatever the payload
    sei = A.sei_nal({'f': 0, 'z': '\x00\x00\x01'.encode().decode()})
    assert b'\x00\x00\x01' not in sei[4:] and b'\x00\x00\x00' not in sei[4:]
    assert A.read_sei(A.insert_sei(idr, {'f': 0, 'z': '\x00\x00\x01'})) == {'f': 0, 'z': '\x00\x00\x01'}


def test_pointcloud2_from_mcap_bag_decodes_on_the_device():
    from falcon_autonomy.core import mcap_bag as M
    rng = np.random.default_rng(1)
    xyzi = rng.normal(size=(997, 4)).astype(np.float32)
    pts = np.frombuffer(xyzi.tobytes(), np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')]))
    cdr = M.encode('sensor_msgs/msg/PointCloud2', M.pointcloud2(pts, 12_345_678_901, 'lidar'))
    out, meta = A.decode_pointcloud2(cdr)
    assert np.array_equal(out, xyzi) and meta['stamp_ns'] == 12_345_678_901 and meta['frame_id'] == 'lidar'
    assert meta['fields'] == ['x', 'y', 'z', 'intensity']


# ============================================================================================ clocks
def test_emulated_sensor_clock_is_seeded_and_recoverable():
    c1, c2 = H.EmulatedClock(3.0, 50.0, 20.0, seed=4), H.EmulatedClock(3.0, 50.0, 20.0, seed=4)
    t = np.arange(400) * 0.05
    s1, s2 = [c1.stamp(x) for x in t], [c2.stamp(x) for x in t]
    assert s1 == s2
    fit = H.fit_clock(t, s1)
    assert abs(fit['offset_ms'] - 3.0) < 0.01 and abs(fit['drift_ppm'] - 50.0) < 1.0 and 15 < fit['jitter_us'] < 25
    assert H.EmulatedClock().stamp(1.5) == 1_500_000_000


@pytest.mark.parametrize('offset_ms,drift_ppm,asym', [(1234.5, 40.0, 0.0), (-8e6, -30.0, 0.0), (5.0, 0.0, 0.3)])
def test_offset_estimate_on_synthetic_clocks(offset_ms, drift_ppm, asym):
    """A device clock d = h·(1 + drift) + offset; pings with random queueing delays (and, in the last case, an
    asymmetric path: the uplink always 0.3 ms slower). The truth must lie within the stated uncertainty."""
    rng = np.random.default_rng(7)
    b = drift_ppm * 1e-6
    dev = lambda h: h * (1 + b) + offset_ms * 1e6               # noqa: E731
    S, h = [], 5.0e13
    for _ in range(600):                                        # 60 s at 10 Hz
        up = 80e3 + rng.exponential(200e3) + asym * 1e6
        down = 80e3 + rng.exponential(200e3)
        t0 = h
        t1 = dev(t0 + up)
        t2 = t1 + 20e3 * (1 + b)
        t3 = t0 + up + 20e3 + down
        S.append((int(t0), int(t1), int(t2), int(t3)))
        h += 1e8
    est = H.estimate_offset(S)
    truth = offset_ms * 1e6 + b * est['t_ref_ns']
    assert abs(est['offset_ns'] - truth) <= est['uncertainty_ns']
    assert abs(est['drift_ppm'] - drift_ppm) < 0.5
    if asym == 0.0:
        assert abs(est['offset_ns'] - truth) < 20e3 and est['trusted']
    else:                                                       # the asymmetry is invisible to the exchange
        assert abs(abs(est['offset_ns'] - truth) - 0.15e6) < 20e3
    for d in (S[10][1], S[500][2]):                             # to_host inverts the fitted line exactly
        hh = H.to_host(d, est)
        assert abs(hh + H.offset_at(est, hh) - d) < 1.0
    assert H.estimate_offset([]) == {}


def test_summary_waterfall_and_deadlines():
    """Synthetic records with known stage durations: the cascade adds up, statuses are counted."""
    recs = []
    for f in range(100):
        t = 1_000_000_000 + f * 50_000_000
        ms = 1_000_000
        r = {'frame': f, 'sim_t': f * 0.05, 'tick_start': t, 'tick_end': t + 20 * ms, 'ready': t + 21 * ms,
             'gw_recv': t + 22 * ms, 'cam_sent': t + 25 * ms, 'dev_cam_rx': t + 26 * ms + 777,
             'dev_dec_end': t + 28 * ms + 777, 'dev_stack_start': t + 28 * ms + 777, 'dev_stack_end': t + 30 * ms + 777,
             'dev_ctrl_sent': t + 30 * ms + 777, 'ctrl_rx': t + 31 * ms, 'ctrl_worker': t + 32 * ms,
             'applied': t + 50 * ms, 'status': 'on_time' if f % 10 else 'late', 'cam_hw_t': int(f * 0.05e9)}
        recs.append(r)
    est = {'offset_ns': 777.0, 't_ref_ns': 0, 'drift_ppm': 0.0, 'uncertainty_ns': 10.0, 'trusted': True}
    s = H.summarise(recs, est, 50_000_000)
    assert [w['key'] for w in s['waterfall']][:4] == ['sim_step', 'sensor_read', 'handoff', 'encode']
    assert s['stages']['uplink']['median'] == pytest.approx(1.0) and s['stages']['decode']['median'] == 2.0
    tot = sum(w['median_ms'] for w in s['waterfall'])
    assert tot == pytest.approx(50.0) and s['measures']['e2e']['median'] == pytest.approx(50.0)
    assert s['measures']['network_rtt']['median'] == pytest.approx(2.0)   # round trip 6 − on device 4
    assert s['deadline']['on_time'] == 90 and s['deadline']['late'] == 10 and s['deadline']['deadline_misses'] == 10
    assert s['jitter']['step_interval_ms']['std'] == 0.0
    assert s['sensor_clocks']['camera']['recovered']['drift_ppm'] == pytest.approx(0.0, abs=1e-3)


# ============================================================================================ the stack
def test_default_stack_cruises_and_brakes_for_an_obstacle():
    st = A.BrakeForObstacles({}, {'lidar': {'pose': {'x': 0.0, 'z': 2.4}, 'rotation_frequency': 20.0}})
    ground = np.c_[np.linspace(3, 30, 50), np.zeros(50), np.full(50, -2.4), np.ones(50)].astype(np.float32)
    u = st.act(None, ground, {'sim_t': 0.05, 'speed_mps': 2.0})
    assert u['throttle'] > 0 and u['brake'] == 0 and u['info']['obstacle_m'] is None     # ground is not an obstacle
    wall = np.c_[np.full(30, 9.0), np.linspace(-1, 1, 30), np.full(30, -1.4), np.ones(30)].astype(np.float32)
    u = st.act(None, wall, {'sim_t': 0.10, 'speed_mps': 6.0})
    assert u['brake'] > 0.2 and u['throttle'] == 0                     # 6.4 m ahead at 6 m/s
    side = wall + np.array([0, 3.0, 0, 0], np.float32)                 # beside the corridor
    u = st.act(None, side, {'sim_t': 0.20, 'speed_mps': 3.0})
    assert u['brake'] == 0 and u['throttle'] > 0
    u = st.act(np.zeros((8, 8, 3), np.uint8), None, {'sim_t': 2.0, 'speed_mps': 3.0})
    assert u['brake'] == 1.0 and u['info']['reason'] == 'no lidar'


# ============================================================================================ sessions
def _gst_has(*elements):
    G = A.gst()
    return G is not None and all(G.ElementFactory.find(e) is not None for e in elements)


def _config(project, out, camera='h264-rtp', encoder='x264enc', lidar='tcp', seconds=3.0, device='local', **kw):
    p = project
    p.doc['traffic_profiles'][0]['vehicles'] = 2
    return H.hil_config(p, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=seconds, fps=10.0, width=192,
                        height=108, camera=camera, encoder=encoder, lidar=lidar,
                        lidar_overrides={'points_per_second': 20000, 'channels': 16, 'rotation_frequency': 10.0},
                        sim_device={'clock_offset_ms': -2500.0, 'clock_drift_ppm': 35.0}, warmup_steps=3,
                        camera_clock={'offset_ms': 3.0, 'drift_ppm': 50.0, 'jitter_us': 20.0}, device=device,
                        gateway_python=sys.executable, **kw)


def _run_worker(cfg, out, env_extra=None, timeout=120):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'hil.json')
    with open(path, 'w') as fh:
        json.dump(cfg, fh)
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'hil', path], env=worker_env(env_extra),
                       capture_output=True, text=True, timeout=timeout)
    return r


def _check_session(out, cfg, transport):
    run = json.load(open(os.path.join(out, 'run.json')))
    assert run['status'] == 'ok' and run['provenance']['seed'] == 0 and run['provenance']['inputs']['config']['sha256']
    h = run['hil']
    recs = [json.loads(x) for x in open(os.path.join(out, 'latency.jsonl'))]
    assert len(recs) == 30 == h['frames'] and h['deadline']['frames'] + h['deadline']['after_end'] == 30
    host = ['tick_start', 'tick_end', 'ready', 'gw_recv', 'cam_sent', 'ctrl_rx', 'ctrl_worker']
    dev = ['dev_cam_rx', 'dev_dec_end', 'dev_stack_start', 'dev_stack_end', 'dev_ctrl_sent']
    full = [r for r in recs if all(r.get(k) is not None for k in host + dev)]
    assert len(full) >= 27, h['drops']                              # loopback: (nearly) every frame closes the loop
    for r in full:                                                   # stamps in causal order on each clock
        assert [r[k] for k in host] == sorted(r[k] for k in host)
        assert [r[k] for k in dev] == sorted(r[k] for k in dev)
        assert r['ctrl_rx'] - r['cam_sent'] > r['dev_ctrl_sent'] - r['dev_cam_rx']    # round trip > time on device
        assert r['lidar_points'] > 0 and set(r['control']) == {'throttle', 'steer', 'brake'}
        if r['status'] in ('on_time', 'late'):                       # applied at a step of the session
            assert r['applied'] >= r['ctrl_worker'] and r['applied_frame'] > r['frame']
    assert h['deadline']['on_time'] >= 20, h['deadline']             # 10 Hz and sub-ms loopback: on time
    # controls were applied: the ego, held at first, accelerated under the stack's throttle
    speeds = [r['speed_mps'] for r in recs]
    assert speeds[0] < 0.1 and max(speeds) > 0.5
    chk = h['clock_offset_check']
    assert chk['within_uncertainty'] and abs(chk['error_ns']) < 200e3, chk
    assert h['one_way_trusted'] and h['stages']['uplink']['median'] < 20.0
    cam = h['sensor_clocks']['camera']['recovered']
    assert abs(cam['offset_ms'] - 3.0) < 0.05 and abs(cam['drift_ppm'] - 50.0) < 20 and cam['jitter_us'] < 40
    assert h['session']['device']['mode'] == 'local' and h['session']['camera']['transport'] == transport
    for f in ('report.html', 'waterfall.png', 'clock.jsonl', 'device.json', 'gateway.json', 'session.json'):
        assert os.path.isfile(os.path.join(out, f)), f
    html = open(os.path.join(out, 'report.html'), encoding='utf-8').read()
    assert 'Latency waterfall' in html and '<svg' in html and 'simulated device' in html
    return h


SESSIONS = [('h264-rtp', 'x264enc', 'tcp', ('x264enc', 'rtph264pay', 'avdec_h264')),
            ('h264-rtp', 'nvh264enc', 'tcp', ('nvh264enc', 'rtph264pay', 'avdec_h264')),
            ('raw-tcp', 'auto', 'udp', ())]


@pytest.mark.parametrize('camera,encoder,lidar,needs', SESSIONS, ids=['x264', 'nvenc', 'raw-udp'])
def test_session_with_a_simulated_device(project, tmp_path, camera, encoder, lidar, needs):
    if needs and not _gst_has(*needs):
        pytest.skip(f'GStreamer element(s) missing here: {needs}')
    out = str(tmp_path / 'hil')
    cfg = _config(project, out, camera, encoder, lidar)
    r = _run_worker(cfg, out)
    assert r.returncode == P.OK, r.stderr[-3000:]
    h = _check_session(out, cfg, camera)
    used = h['session']['camera']['encoder_used']
    assert used == (encoder if camera == 'h264-rtp' else 'raw')
    if camera == 'h264-rtp':
        assert h['drops']['camera_decoded'] >= 27 and h['session']['device']['decoder']
    # the report can be rewritten from the files alone
    from falcon_autonomy.hil.report import write_report
    files = write_report(out, str(tmp_path / 'again'))
    assert os.path.isfile(files['html'])


def test_session_stop_command_and_untrusted_without_pings(project, tmp_path):
    """A session until stopped ends cleanly on stdin "stop"; seconds 0 means no step count."""
    out = str(tmp_path / 'hil')
    cfg = _config(project, out, 'raw-tcp', 'auto', 'tcp', seconds=0.0)
    os.makedirs(out)
    path = os.path.join(out, 'hil.json')
    json.dump(cfg, open(path, 'w'))
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'hil', path], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())
    try:
        t0 = time.monotonic()
        seen = False
        for line in p.stdout:
            if '"t":"hil"' in line:
                msg = json.loads(line)
                assert msg['waterfall'] and msg['e2e']['median'] is not None
                seen = True
                break
            assert time.monotonic() - t0 < 60
        assert seen
        p.stdin.write('{"cmd": "stop"}\n')
        p.stdin.flush()
        p.wait(60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == P.OK, p.stderr.read()[-2000:]
    run = json.load(open(os.path.join(out, 'run.json')))
    assert run['hil']['frames'] > 5 and run['summary']['report']


def test_unreachable_device_fails_clearly(project, tmp_path):
    from falcon_autonomy.hil.gateway import free_base_port
    out = str(tmp_path / 'hil')
    cfg = _config(project, out, 'raw-tcp', 'auto', 'tcp', device='remote', device_port=free_base_port())
    r = _run_worker(cfg, out, {'FALCON_HIL_ADDRESS': '127.0.0.1'}, timeout=90)
    assert r.returncode == P.FAILED
    run = json.load(open(os.path.join(out, 'run.json')))
    assert 'no device agent answered' in run['error'] and '127.0.0.1' not in run['error']


def test_config_never_holds_the_address_and_rejects_bad_choices(project, tmp_path, settings):
    with pytest.raises(ValueError):
        H.hil_config(project, 'world-town10', 'rig-default', str(tmp_path), 2310, 8310, camera='mjpeg')
    with pytest.raises(ValueError):
        H.hil_config(project, 'world-town10', 'rig-default', str(tmp_path), 2310, 8310, width=100, height=50)
    cfg = H.hil_config(project, 'world-town10', 'rig-default', str(tmp_path), 2310, 8310, device='remote')
    settings.set('hil_device', {'address': '10.1.2.3', 'ssh_user': 'nvidia'})
    dev = H.device_settings(settings)
    assert dev['address'] == '10.1.2.3' and dev['port'] == 7710 and dev['remote_dir'] == '~/falcon-hil'
    assert '10.1.2.3' not in json.dumps(cfg) and cfg['lidar']['sensor']['id'] == 'lidar-hil' and cfg['notes']
    assert 'hil_device' not in project.doc

    class Svc:                                                # submit_session refuses a remote session without one
        pass
    with pytest.raises(ValueError):
        H.submit_session(Svc(), cfg, '')


# ============================================================================================ deployment
@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """ssh and scp stand-ins that run "on the device" here: ssh runs its command with bash (HOME = a scratch folder),
    scp copies; both record their arguments."""
    home, bin_ = tmp_path / 'device-home', tmp_path / 'bin'
    home.mkdir()
    bin_.mkdir()
    log = tmp_path / 'ssh.log'
    (bin_ / 'ssh').write_text(f'#!/bin/sh\necho "ssh $*" >> {log}\nfor last; do :; done\n'
                              f'HOME={home} exec bash -c "$last"\n')
    (bin_ / 'scp').write_text(f'#!/bin/sh\necho "scp $*" >> {log}\nfor last; do :; done\nsrc=""\n'
                              'for a; do [ "$a" = "$last" ] || src="$a"; done\n'
                              f'cp "$src" "{home}/${{last#*:}}"\n')
    for f in ('ssh', 'scp'):
        os.chmod(bin_ / f, stat.S_IRWXU)
    monkeypatch.setenv('PATH', f'{bin_}{os.pathsep}{os.environ["PATH"]}')
    key = tmp_path / 'id_ed25519'
    key.write_text('not a real key')
    from falcon_autonomy.hil.gateway import free_base_port
    dev = dict(H.DEVICE_DEFAULTS, address='127.0.0.1', ssh_user='nvidia', ssh_key=str(key), port=free_base_port(),
               python=sys.executable)
    return dev, home, log


def test_deploy_check_start_session_stop(fake_ssh, project, tmp_path, settings):
    """The acceptance flow with ssh standing in: copy, check, start the agent for one session, a remote session
    through the job queue (the address only in the worker's environment), the agent gone afterwards."""
    from falcon_autonomy.core.gymenv import make_service
    from falcon_autonomy.hil import deploy as D
    dev, home, log = fake_ssh
    D.copy(dev)
    assert (home / 'falcon-hil' / 'device_agent.py').read_bytes() == open(A.__file__, 'rb').read()
    env = D.check(dev)
    assert env['protocol'] == A.PROTOCOL and env['numpy']
    cmds = log.read_text()
    assert 'BatchMode=yes' in cmds and f'-i {dev["ssh_key"]}' in cmds and 'StrictHostKeyChecking=accept-new' in cmds
    pid = D.start(dev, once=True, extra=['--listen', '127.0.0.1'])
    assert pid > 0
    try:
        out = str(tmp_path / 'remote')
        cfg = _config(project, out, 'raw-tcp', 'auto', 'tcp', device='remote', device_port=dev['port'])
        svc = make_service(settings, fake_carla=FAKE, lock_dir=str(tmp_path / 'locks'))
        job = H.submit_session(svc, cfg, dev['address'])
        assert svc.queue.wait(job, 120) and job.state == 'done', job.log[-10:]
        svc.shutdown()
        run = json.load(open(os.path.join(out, 'run.json')))
        assert run['hil']['session']['device']['mode'] == 'remote' and run['hil']['deadline']['on_time'] >= 20
        assert 'clock_offset_check' not in run['hil']                    # a real device: no known truth
        for f in os.listdir(out):                 # the device's address is in no file of the run (CARLA's host is)
            if f.endswith(('.json', '.jsonl', '.html')):
                text = open(os.path.join(out, f), encoding='utf-8').read()
                assert '"address": "127.0.0.1"' not in text and 'FALCON_HIL_ADDRESS' not in text, f
        assert run['hil']['session']['device']['address'].startswith('from the user')
        assert (home / 'falcon-hil' / 'agent.log').read_text().count('session ended') == 1
        t0 = time.monotonic()                                            # --once: it exits after its session
        while time.monotonic() - t0 < 10 and os.path.exists(f'/proc/{pid}'):
            time.sleep(0.1)
        assert not os.path.exists(f'/proc/{pid}')
    finally:
        if D.stop(dev, pid):                                             # never leave it running
            pytest.fail('the agent was still running after its one session')
    with pytest.raises(D.DeployError):
        D.validate(dict(dev, address='-oProxyCommand=x'))
    with pytest.raises(D.DeployError):
        D.validate(dict(dev, ssh_key=str(tmp_path / 'missing')))
    assert '<address>' in D._clean('ssh: connect to host 127.0.0.1 port 22: refused', dev)


def test_cli_device_and_report(tmp_path, monkeypatch, project):
    from falcon_autonomy.cli import main
    from falcon_autonomy.core.settings import Settings
    key = tmp_path / 'k'
    key.write_text('x')
    assert main(['hil', 'device', '--address', 'orin.local', '--ssh-user', 'nvidia', '--ssh-key', str(key)]) == 0
    dev = H.device_settings(Settings())
    assert dev['address'] == 'orin.local' and dev['ssh_key'] == str(key)
    assert main(['hil', 'report', str(tmp_path / 'nothing')]) == P.FAILED
    assert main(['hil', 'nonsense']) == P.FAILED


# ============================================================================================ JetPack's Python (opt-in)
def _humble_image() -> bool:
    if os.environ.get('FALCON_ROS2') != '1' or shutil.which('docker') is None:
        return False
    return subprocess.run(['docker', 'image', 'inspect', 'ros:humble-ros-base'], capture_output=True,
                          timeout=30).returncode == 0


ROS = pytest.mark.skipif(not _humble_image(), reason='container tests need FALCON_ROS2=1, Docker and the official '
                                                      'ros:humble-ros-base image')


def _container_agent(name, port, extra='', pre=''):
    agent_dir = os.path.dirname(os.path.abspath(A.__file__))
    cmd = (f'{pre}python3 /agent/device_agent.py --listen 127.0.0.1 --port {port} --once {extra}')
    return subprocess.Popen(['docker', 'run', '--rm', '--name', name, '--network', 'host', '-v',
                             f'{agent_dir}:/agent:ro', 'ros:humble-ros-base', 'bash', '-c', cmd],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait_port(port, timeout=60):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            socket.create_connection(('127.0.0.1', port), 0.5).close()
            return True
        except OSError:
            time.sleep(0.3)
    return False


@pytest.mark.ros2
@ROS
def test_agent_in_jetpack6_python_container(project, tmp_path):
    """The agent under Python 3.10 and numpy 1.21 (Ubuntu 22.04's, as on JetPack 6), in a container on the host's
    network (127.0.0.1 only): a raw-camera session closes the loop."""
    from falcon_autonomy.hil.gateway import free_base_port
    port = free_base_port()
    p = _container_agent('falcon-test-hil-agent', port)
    try:
        assert _wait_port(port), p.stdout.read() if p.poll() is not None else 'agent not listening'
        out = str(tmp_path / 'hil')
        cfg = _config(project, out, 'raw-tcp', 'auto', 'tcp', device='remote', device_port=port)
        r = _run_worker(cfg, out, {'FALCON_HIL_ADDRESS': '127.0.0.1'})
        assert r.returncode == P.OK, r.stderr[-3000:]
        h = json.load(open(os.path.join(out, 'run.json')))['hil']
        assert h['session']['device']['env']['python'].startswith('3.10') and h['deadline']['on_time'] >= 20
        p.wait(30)
    finally:
        subprocess.run(['docker', 'rm', '-f', 'falcon-test-hil-agent'], capture_output=True)


@pytest.mark.ros2
@ROS
def test_lidar_republished_as_pointcloud2_over_ros2(project, tmp_path):
    """--ros2-topic: the gateway encodes PointCloud2 (core.mcap_bag), the agent republishes it with rclpy; a ROS 2
    subscriber in the same container receives every message with the fields intact."""
    from falcon_autonomy.hil.gateway import free_base_port
    port = free_base_port()
    sub = tmp_path / 'sub.py'
    sub.write_text(
        'import json, rclpy, time\nfrom rclpy.qos import qos_profile_sensor_data\n'
        'from sensor_msgs.msg import PointCloud2\nrclpy.init()\nn = rclpy.create_node("falcon_test_sub")\nseen = []\n'
        'n.create_subscription(PointCloud2, "/falcon/lidar", lambda m: seen.append((m.width, [f.name for f in '
        'm.fields], m.header.frame_id)), qos_profile_sensor_data)\nt0 = time.time()\n'
        'while time.time() - t0 < 60 and len(seen) < 30:\n    rclpy.spin_once(n, timeout_sec=0.1)\n'
        'json.dump(seen, open("/out/seen.json", "w"))\n')
    agent_dir = os.path.dirname(os.path.abspath(A.__file__))
    cmd = ('source /opt/ros/humble/setup.bash && { python3 /out/sub.py & SUB=$!; } && sleep 3 && '
           f'python3 /agent/device_agent.py --listen 127.0.0.1 --port {port} --once; wait $SUB')
    p = subprocess.Popen(['docker', 'run', '--rm', '--name', 'falcon-test-hil-ros2', '--network', 'host', '-v',
                          f'{agent_dir}:/agent:ro', '-v', f'{tmp_path}:/out', 'ros:humble-ros-base', 'bash', '-c',
                          cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert _wait_port(port), p.stdout.read() if p.poll() is not None else 'agent not listening'
        out = str(tmp_path / 'hil')
        cfg = _config(project, out, 'raw-tcp', 'auto', 'tcp', device='remote', device_port=port,
                      ros2_topic='/falcon/lidar')
        r = _run_worker(cfg, out, {'FALCON_HIL_ADDRESS': '127.0.0.1'})
        assert r.returncode == P.OK, r.stderr[-3000:]
        dev = json.load(open(os.path.join(out, 'device.json')))
        assert dev['stats']['ros2']['published'] == 30
        t0 = time.monotonic()
        while not (tmp_path / 'seen.json').exists() and time.monotonic() - t0 < 70:
            time.sleep(0.5)
        seen = json.load(open(tmp_path / 'seen.json'))
        assert len(seen) >= 28 and all(f == ['x', 'y', 'z', 'intensity'] and fid == 'lidar' for _, f, fid in seen)
    finally:
        subprocess.run(['docker', 'rm', '-f', 'falcon-test-hil-ros2'], capture_output=True)
