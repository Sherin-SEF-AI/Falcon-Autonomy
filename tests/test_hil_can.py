"""Hardware in the loop over CAN, PTP-style clocks and V4L2, end to end with the fake carla and a simulated device: the
chassis model on a simulated CAN link (no vcan, no privileges) driving the ego from the device's CAN commands, the
"can" stage of the waterfall from the kernel's receive timestamps, both clock sources and their disagreement; the
waterfall's CAN stages on synthetic records; the device agent still standard library and numpy; `falcon hil setup`
printing, never running; the CLI's can / ptp / v4l2 commands. On a real vcan0 (marker vcan, FALCON_VCAN=1) the same
session runs over SocketCAN."""
import json
import os
import subprocess
import sys

import pytest

from conftest import worker_env
from falcon_autonomy.core import hil as H
from falcon_autonomy.core import protocol as P
from falcon_autonomy.hil import device_agent as A


def _config(project, out, can, seconds=3.0, **kw):
    project.doc['traffic_profiles'][0]['vehicles'] = 1
    return H.hil_config(project, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=seconds, fps=10.0,
                        width=192, height=108, camera='raw-tcp', lidar='tcp',
                        lidar_overrides={'points_per_second': 20000, 'channels': 16, 'rotation_frequency': 10.0},
                        sim_device={'clock_offset_ms': 1500.0, 'clock_drift_ppm': 20.0}, warmup_steps=3,
                        device='local', gateway_python=sys.executable, can=can, **kw)


def _run(cfg, out, timeout=150):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'hil.json')
    with open(path, 'w') as fh:
        json.dump(cfg, fh)
    return subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'hil', path], env=worker_env(),
                          capture_output=True, text=True, timeout=timeout)


def _check_can_session(out, fd):
    run = json.load(open(os.path.join(out, 'run.json')))
    assert run['status'] == 'ok', run.get('error')
    h = run['hil']
    recs = [json.loads(x) for x in open(os.path.join(out, 'latency.jsonl'))]
    # the CAN stages replace the session's way back; the "can" stage is the kernel's timestamp to the worker
    keys = [k for k in h['stages']]
    assert keys[-4:] == ['reply', 'can_bus', 'can', 'wait_step'] and 'downlink' not in keys
    can = h['stages']['can']
    assert can['n'] >= 20 and 0.0 <= can['median'] < 20.0 and can['clock'] == 'host', can
    assert h['stages']['can_bus']['n'] >= 20 and h['stages']['reply']['n'] >= 20
    with_can = [r for r in recs if r.get('can_rx') is not None]
    assert len(with_can) >= 20
    for r in with_can:
        assert r['can_rx'] <= r['can_read'] == r['ctrl_worker']          # kernel time, then the bridge's read
        assert r['dev_stack_end'] <= r['dev_can_tx']                     # the stack, then onto the bus
        if r['status'] in ('on_time', 'late'):
            assert r['applied'] >= r['ctrl_worker'] and r['applied_frame'] > r['frame']
    assert h['deadline']['on_time'] >= 15, h['deadline']
    speeds = [r['speed_mps'] for r in recs]
    assert speeds[0] < 0.1 and max(speeds) > 0.3                        # the chassis drove the ego on CAN commands
    c = json.load(open(os.path.join(out, 'can.json')))
    st = c['stats']
    assert c['fd'] == fd and st['commands'] >= 40 and st['tx_frames'] >= 100 and not c['aborted']
    assert st['rejected_crc'] == st['rejected_counter'] == 0 and st['error_frames'] == 0
    assert st['saturated'] == 0, c['saturated_signals']                 # every signal within its range
    assert c['chassis']['surface'] == 'wet_asphalt' and c['timestamps'].startswith('kernel')
    dev = json.load(open(os.path.join(out, 'device.json')))
    dc = dev['stats']['can']
    assert dc['fd'] == fd and dc['tx_frames'] >= 40 and dc['rx_frames'] >= 50 and dc['error_frames'] == 0
    assert run['can']['interface'].startswith(('sim:', 'vcan')) and run['can_vehicle']['max_steer_deg'] == 70.0
    # both clock sources, the one used (auto, a simulated device: shared realtime) and their disagreement
    cs = h['clock_sources']
    assert cs['used'] == 'ptp' and cs['ptp']['basis'] == 'shared realtime' and cs['ntp_style']
    assert cs['consistent'] and abs(cs['disagreement_ns']) < cs['ptp']['uncertainty_ns'] + \
        cs['ntp_style']['uncertainty_ns']
    for k in ('ptp', 'ntp_style'):
        assert cs['checks'][k]['within_uncertainty'], cs['checks']
    assert abs(cs['checks']['ptp']['error_ns']) < 50e3
    html = open(os.path.join(out, 'report.html'), encoding='utf-8').read()
    assert 'CAN frame to the simulation' in html and 'Clock sources' in html and 'CAN bus' in html
    return h


@pytest.mark.parametrize('fd', [False, True], ids=['classical', 'fd'])
def test_can_session_on_a_simulated_link(project, tmp_path, fd):
    out = str(tmp_path / 'hil')
    can = {'interface': f'sim:test-{os.getpid()}-{int(fd)}', 'fd': 'on' if fd else 'off', 'control': 'can',
           'chassis': {'surface': 'wet_asphalt'}}
    cfg = _config(project, out, can)
    assert cfg['can']['device_interface'] == can['interface'] and cfg['clock_source'] == 'auto'
    r = _run(cfg, out)
    assert r.returncode == P.OK, r.stderr[-3000:]
    _check_can_session(out, fd)


@pytest.mark.vcan
def test_can_session_on_vcan0(project, tmp_path):
    """The same over SocketCAN (needs vcan0 up: falcon hil setup prints how)."""
    if not os.path.exists('/sys/class/net/vcan0'):
        pytest.skip('vcan0 does not exist')
    out = str(tmp_path / 'hil')
    fd = A.interface_mtu('vcan0') == 72
    cfg = _config(project, out, {'interface': 'vcan0', 'control': 'can', 'chassis': {'surface': 'wet_asphalt'}})
    r = _run(cfg, out)
    assert r.returncode == P.OK, r.stderr[-3000:]
    _check_can_session(out, fd)


def test_a_can_interface_not_allowed_is_refused(project, tmp_path, settings):
    with pytest.raises(ValueError, match='not in the CAN interfaces'):
        _config(project, tmp_path / 'x', {'interface': 'can0'}, settings=settings)
    with pytest.raises(ValueError, match='chassis'):
        _config(project, tmp_path / 'x', {'interface': 'sim:x', 'chassis': {'surface': 'lava'}})
    # a config written by hand past the allow-list: the worker refuses it from the user's settings
    out = str(tmp_path / 'hil')
    cfg = _config(project, out, {'interface': 'sim:x'})
    cfg['can']['interface'] = 'can7'
    r = _run(cfg, out, timeout=60)
    assert r.returncode == P.REFUSED, r.stderr[-2000:]
    assert 'not allowed' in json.load(open(os.path.join(out, 'run.json')))['error']


# ============================================================================================ the stages
def test_can_waterfall_stages_on_synthetic_records():
    est = {'offset_ns': 1_000_000.0, 't_ref_ns': 0, 'drift_ppm': 0.0, 'uncertainty_ns': 1000.0, 'trusted': True}
    recs = []
    for k in range(20):
        t = 1_000_000_000 + k * 100_000_000
        recs.append({'frame': k, 'sim_t': k * 0.1, 'tick_start': t, 'tick_end': t + 5_000_000,
                     'ready': t + 6_000_000, 'gw_recv': t + 7_000_000, 'cam_sent': t + 8_000_000,
                     'dev_cam_rx': t + 9_000_000 + 1_000_000, 'dev_dec_end': t + 10_000_000 + 1_000_000,
                     'dev_stack_start': t + 11_000_000 + 1_000_000, 'dev_stack_end': t + 12_000_000 + 1_000_000,
                     'dev_can_tx': t + 12_500_000 + 1_000_000, 'can_rx': t + 13_000_000,
                     'can_read': t + 13_200_000, 'ctrl_worker': t + 13_200_000, 'applied': t + 100_000_000,
                     'status': 'on_time'})
    s = H.summarise(recs, est, 100_000_000, cfg={'can': {'control': 'can'}})
    assert [w['key'] for w in s['waterfall']][-4:] == ['reply', 'can_bus', 'can', 'wait_step']
    assert s['stages']['can']['median'] == pytest.approx(0.2) and s['stages']['can_bus']['median'] == \
        pytest.approx(0.5) and s['stages']['reply']['median'] == pytest.approx(0.5)
    assert s['stages']['can_bus']['one_way'] and s['stages']['can_bus']['trusted']
    # without CAN control the classic stages stand
    s2 = H.summarise(recs, est, 100_000_000, cfg={'can': {'control': 'record'}})
    assert 'can' not in s2['stages'] and 'downlink' in s2['stages']
    assert H.stages_for(None) is H.STAGES and H.stages_for({'can': {'control': 'can'}}) is H.CAN_STAGES


# ============================================================================================ the device agent
def test_device_agent_needs_only_the_standard_library_and_numpy(tmp_path):
    """On the Jetson the agent runs from its folder with its siblings (falcon hil deploy's copies) — nothing of
    Falcon, nothing beyond the standard library and numpy, even with CAN and V4L2 loaded."""
    from falcon_autonomy.hil import deploy as D
    import shutil
    for src, name in D.FILES:
        shutil.copy(src, tmp_path / name)
    code = ('import sys, json\n'
            f'sys.path.insert(0, {str(tmp_path)!r})\n'
            'base = set(sys.modules)\n'
            'import device_agent as A\n'
            'C = A._canmsgs(); g = A._sibling("gmsl"); v = A._sibling("v4l2out")\n'
            'assert C.__file__.endswith("falcon_canmsgs.py") and g.V is v\n'
            'new = {m.split(".")[0] for m in set(sys.modules) - base}\n'
            'ours = ("device_agent", "falcon_canmsgs", "gmsl", "v4l2out")\n'
            'bad = sorted(m for m in new if m not in sys.stdlib_module_names and m != "numpy"\n'
            '             and m not in ours and not m.startswith("_"))\n'
            'print(json.dumps(bad))\n')
    r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, cwd=str(tmp_path),
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == []
    assert 'falcon_autonomy' not in r.stdout


def test_device_can_maps_the_stacks_output_to_commands():
    import threading
    from falcon_autonomy.core import canmsgs as C

    class Ses:
        stopping = threading.Event()
        clock = A.DeviceClock()
        notes = []

        def count(self, k):
            pass

        def note(self, t):
            self.notes.append(t)
    name = f'sim:agent-{os.getpid()}'
    host = A.CanLink(name, 'off', side='a')
    try:
        dc = A.DeviceCan(Ses(), name, 'off', {'steer_max_deg': 500.0, 'brake_full_bar': 80.0})
        t = dc.command(70001, {'throttle': 0.25, 'steer': 0.1, 'brake': 0.5})
        assert t > 0
        got = {}
        for _ in range(2):
            raw, ts = host.recv(1.0)
            fr = C.unpack_frame(raw)
            got[C.BY_ID[fr.can_id].name] = C.decode(C.BY_ID[fr.can_id], fr.data)
            assert ts is not None
        s, lg = got['CMD_STEER'], got['CMD_LONG']
        assert s['STEER_REQ'] == pytest.approx(-50.0) and s['ENABLE'] == 1 and s['STEER_MODE'] == 1
        assert lg['BRAKE_REQ'] == pytest.approx(40.0) and lg['THROTTLE_REQ'] == pytest.approx(25.0)
        assert lg['STEP'] == 70001 & 0xFFFF and s['STEP'] == 70001 & 0xFFF and lg['_crc_ok'] and s['_crc_ok']
        # torque mode and an explicit pressure from the stack
        dc.command(5, {'steer_torque_nm': 2.5, 'brake_bar': 12.0})
        got = {}
        for _ in range(2):
            fr = C.unpack_frame(host.recv(1.0)[0])
            got[C.BY_ID[fr.can_id].name] = C.decode(C.BY_ID[fr.can_id], fr.data)
        assert got['CMD_STEER']['STEER_MODE'] == 2 and got['CMD_STEER']['STEER_TORQUE_REQ'] == pytest.approx(2.5)
        assert got['CMD_LONG']['BRAKE_REQ'] == pytest.approx(12.0)
        # the chassis's feedback is decoded for the stack
        host.send(C.pack_frame(C.frame('BRAKE', {'BRAKE_PRESSURE': 33.3, 'VEHICLE_SPEED': 50.0})))
        dc.start()
        import time
        t0 = time.monotonic()
        while 'BRAKE_PRESSURE' not in dc.snapshot() and time.monotonic() - t0 < 3:
            time.sleep(0.02)
        assert dc.snapshot()['BRAKE_PRESSURE'] == pytest.approx(33.3, abs=0.01)
        Ses.stopping.set()
        dc.stop()
    finally:
        host.close()


# ============================================================================================ the CLI
def test_hil_setup_prints_and_runs_nothing(monkeypatch, capsys):
    from falcon_autonomy.cli import main

    def refuse(*a, **k):
        raise AssertionError(f'setup ran something: {a}')
    monkeypatch.setattr(subprocess, 'run', refuse)
    monkeypatch.setattr(subprocess, 'Popen', refuse)
    monkeypatch.setattr(os, 'system', refuse)
    assert main(['hil', 'setup', '--can-interface', 'can1', '--ptp-interface', 'enp1s0']) == 0
    text = capsys.readouterr().out
    for want in ('sudo modprobe vcan', 'sudo ip link add dev vcan0 type vcan',
                 'sudo modprobe v4l2loopback devices=2 video_nr=40,41 card_label=falcon exclusive_caps=1',
                 'sudo ip link set can1 up type can bitrate 500000 dbitrate 2000000 fd on',
                 'falcon hil can confirm can1', 'sudo ptp4l -f', 'never runs these'):
        assert want in text, want
    assert main(['hil', 'setup', '--can-interface', 'can0;rm']) == P.FAILED      # not an interface name


def test_cli_can_ptp_v4l2_commands(tmp_path, capsys, monkeypatch):
    from falcon_autonomy.cli import main
    from falcon_autonomy.core.settings import Settings
    out = tmp_path / 'f.dbc'
    assert main(['hil', 'can', 'dbc', '--out', str(out)]) == 0
    from falcon_autonomy.core import canmsgs as C
    assert out.read_text() == C.dbc_text()
    assert main(['hil', 'ptp', 'conf', '--profile', 'gptp', '--interface', 'enp1s0']) == 0
    text = capsys.readouterr().out
    assert 'transportSpecific       0x1' in text and 'network_transport       L2' in text and '[enp1s0]' in text
    assert main(['hil', 'ptp', 'conf', '--interface', 'bad name']) == P.FAILED
    assert main(['hil', 'can', 'confirm', 'can3', '--statement', 'sure']) == P.FAILED
    assert main(['hil', 'can', 'confirm', 'can3', '--statement', 'bench network, not a vehicle']) == 0
    cfg = Settings().get('hil_can')
    assert 'can3' in cfg['interfaces_allowed']
    assert cfg['confirmed']['can3']['statement'] == 'bench network, not a vehicle'
    assert main(['hil', 'can', 'revoke', 'can3']) == 0
    assert 'can3' not in Settings().get('hil_can')['interfaces_allowed']
    assert main(['hil', 'can', 'interfaces']) == 0
    assert main(['hil', 'v4l2', 'devices']) == 0
    assert main(['hil', 'v4l2', 'pattern', str(tmp_path / 'no-such-video')]) == P.FAILED
    assert main(['hil', 'can', 'bench', 'can9', '--seconds', '0.1']) == P.FAILED        # not allowed
    for args in (['hil', 'setup', '--help'], ['hil', 'can', 'bench', '--help'], ['hil', 'ptp', 'conf', '--help'],
                 ['hil', 'v4l2', 'pattern', '--help'], ['hil', 'session', '--help']):
        assert main(args) == 0, args
    from falcon_autonomy.hil.cli import doctor_rows
    rows = doctor_rows(Settings())
    names = [r[0] for r in rows]
    assert 'can_raw module' in names and 'pmc' in names and 'v4l2loopback' in names
    assert all(len(r) == 3 for r in rows)


def test_can_bench_on_a_simulated_link(settings):
    """`falcon hil can bench` on a sim link: the chassis model as the plant, a device answering at 50 Hz."""
    import threading
    import time
    from falcon_autonomy.core import canmsgs as C
    from falcon_autonomy.hil import canbridge as B
    name = f'sim:bench-{os.getpid()}'
    dev = A.CanLink(name, 'off', side='b')
    stop = threading.Event()

    def device():
        cnt = C.Counters()
        while not stop.wait(0.02):
            for n, v in (('CMD_STEER', {'STEER_REQ': 90.0, 'STEER_MODE': 1, 'ENABLE': 1}),
                         ('CMD_LONG', {'THROTTLE_REQ': 30.0, 'ENABLE': 1})):
                dev.send(C.pack_frame(C.frame(n, v, cnt.next(n))))
    t = threading.Thread(target=device, daemon=True)
    t.start()
    try:
        out = B.bench(name, settings, 1.0, 'off', {'surface': 'dry_asphalt'})
    finally:
        stop.set()
        t.join(2)
        dev.close()
    assert out['mode'] == 'active' and out['commands'] > 60 and out['tx_frames'] > 400 and not out['aborted'], out
    assert out['bus_load'] and 0.05 < out['bus_load'] < 0.25
