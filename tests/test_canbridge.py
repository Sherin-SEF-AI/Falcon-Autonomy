"""The CAN layer (core.canmsgs, hil.canbridge): the kernel's frame layouts (classical and CAN FD) through an in-process
fake bus with kernel receive timestamps; every message of the table encoded and decoded, its CRC and alive counter;
the generated DBC (the committed resource is current, and reads back to the table), a user's DBC (both byte orders)
mapped onto the bridge's roles; the bridge's interlocks — the command watchdog's safe hold, recovery, ENABLE, rejected
frames, an error frame aborting, the bus left silent — and the interface allow-list; python-can through a stand-in
module. Nothing here opens a network interface."""
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

from falcon_autonomy.core import canmsgs as C
from falcon_autonomy.core.chassis import Chassis
from falcon_autonomy.hil import canbridge as B
from falcon_autonomy.hil import device_agent as A

RESOURCE = os.path.join(os.path.dirname(C.__file__), '..', 'resources', 'can', 'falcon_chassis.dbc')


# ============================================================================================ frames and the codec
def test_frame_layouts_round_trip():
    fr = C.Frame(0x123, b'\x01\x02\x03')
    raw = C.pack_frame(fr)
    assert len(raw) == C.CAN_MTU and C.unpack_frame(raw) == fr
    fd = C.Frame(0x0C0, bytes(range(40)), fd=True, brs=True)
    raw = C.pack_frame(fd)
    back = C.unpack_frame(raw)
    assert len(raw) == C.CANFD_MTU and back.fd and back.brs
    assert back.data == bytes(range(40)) + bytes(8)                    # 48: the next CAN FD length
    ext = C.Frame(0x18FEF100, b'\xaa' * 8, extended=True)
    assert C.unpack_frame(C.pack_frame(ext)) == ext
    with pytest.raises(ValueError):
        C.pack_frame(C.Frame(1, bytes(9)))
    with pytest.raises(ValueError):
        C.unpack_frame(b'short')
    assert C.fd_length(9) == 12 and C.fd_length(64) == 64
    err = C.unpack_frame(C.pack_frame(C.Frame(0x040 | 0x004, bytes([0, 0x30, 0, 0, 0, 0, 0, 0]), error=True)))
    assert err.error and C.describe_error(err) == ['controller', 'bus_off', 'rx_passive', 'tx_passive']


def test_crc_and_counters():
    assert C.crc8_j1850(b'123456789') == 0x4B                          # the published check value
    cnt = C.Counters()
    assert [cnt.next('X') for _ in range(18)] == list(range(16)) + [0, 1]
    assert C.counter_ok(None, 5) and C.counter_ok(15, 0) and C.counter_ok(3, 6)
    assert not C.counter_ok(3, 3) and not C.counter_ok(3, 7)            # a repeat, a jump of four


def test_every_message_round_trips_and_its_signals_fit():
    rng = np.random.default_rng(3)
    for m in C.TABLE:
        used = set()
        for s in m.signals:
            bits = set(range(s.start, s.start + s.length))
            assert not bits & used and max(bits) < 8 * m.length, (m.name, s.name)
            used |= bits
        assert m.protected and m.signals[-1].name == 'CRC' and m.signals[-1].start == 8 * (m.length - 1)
        vals = {}
        for s in m.signals:
            if s.name in ('COUNTER', 'CRC'):
                continue
            lo, hi = s.phys_range()
            vals[s.name] = float(rng.uniform(lo, hi))
        data, sat = C.encode(m, vals, counter=11)
        assert len(data) == m.length and not sat
        back = C.decode(m, data)
        assert back['_crc_ok'] and back['COUNTER'] == 11
        for k, v in vals.items():
            assert back[k] == pytest.approx(v, abs=m.signal(k).scale / 2 + 1e-9), (m.name, k)
        bad = bytearray(data)
        bad[0] ^= 0x01
        assert not C.decode(m, bytes(bad))['_crc_ok']
        moved = C.Message(m.can_id + 1, m.name, m.length, m.sender, m.period_ms, m.signals)
        assert not C.decode(moved, data)['_crc_ok']                     # the identifier is in the CRC
    data, sat = C.encode(C.BY_NAME['CMD_LONG'], {'BRAKE_REQ': 1e6, 'THROTTLE_REQ': 150.0})
    assert set(sat) == {'BRAKE_REQ', 'THROTTLE_REQ'}
    v = C.decode(C.BY_NAME['CMD_LONG'], data)
    assert v['THROTTLE_REQ'] == 100.0 and v['BRAKE_REQ'] == pytest.approx(655.35)


def test_motorola_signals_round_trip():
    s = C.Signal('M', 7, 12, signed=True, little_endian=False)
    buf = bytearray(8)
    C.set_raw(s, buf, -1234 & 0xFFF)
    assert C.get_raw(s, bytes(buf)) == -1234
    assert C._motorola_positions(7, 12) == [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12]
    s2 = C.Signal('N', 13, 10, little_endian=False)                    # starts mid-byte, crosses into the next
    buf = bytearray(8)
    C.set_raw(s2, buf, 0x2A5)
    assert C.get_raw(s2, bytes(buf)) == 0x2A5


# ============================================================================================ the DBC
def test_the_committed_dbc_is_the_generated_one():
    with open(RESOURCE, encoding='utf-8') as fh:
        assert fh.read() == C.dbc_text(), 'resources/can/falcon_chassis.dbc is stale: falcon hil can dbc --out it'


def test_the_dbc_reads_back_to_the_table():
    msgs, notes = C.parse_dbc(C.dbc_text())
    assert not notes and len(msgs) == len(C.TABLE)
    for m, t in zip(msgs, C.TABLE):
        assert (m.can_id, m.name, m.length, m.sender, m.period_ms, m.fd) == \
            (t.can_id, t.name, t.length, t.sender, t.period_ms, t.fd)
        for s, ts in zip(m.signals, t.signals):
            assert (s.name, s.start, s.length, s.signed, s.little_endian, s.unit) == \
                (ts.name, ts.start, ts.length, ts.signed, ts.little_endian, ts.unit)
            assert s.scale == pytest.approx(ts.scale) and s.offset == ts.offset


def test_the_dbc_reads_in_cantools_when_installed():
    cantools = pytest.importorskip('cantools')
    db = cantools.database.load_file(RESOURCE)
    m = db.get_message_by_name('CHASSIS_FD')
    assert m.frame_id == 0x0C0 and m.length == 64 and m.is_fd
    data = db.encode_message('CMD_LONG', {'BRAKE_REQ': 12.5, 'THROTTLE_REQ': 20, 'ENABLE': 1, 'STEP': 9,
                                          'COUNTER': 0, 'CRC': 0})
    assert C.decode(C.BY_NAME['CMD_LONG'], data)['BRAKE_REQ'] == pytest.approx(12.5)


USER_DBC = '''VERSION ""
BU_: ECU GW
BO_ 768 VEH_STATE: 8 ECU
 SG_ WhlSpdFL : 7|16@0+ (0.01,0) [0|655.35] "km/h" GW
 SG_ YawRt : 23|16@0- (0.01,0) [-327.68|327.67] "deg/s" GW
 SG_ Alive : 48|4@1+ (1,0) [0|15] "" GW
BO_ 769 BRK_CMD: 8 GW
 SG_ PressReq : 0|16@1+ (0.1,0) [0|6553.5] "bar" ECU
 SG_ Mux M : 16|8@1+ (1,0) [0|0] "" ECU
 SG_ Muxed m1 : 24|8@1+ (1,0) [0|0] "" ECU
'''


def test_a_user_dbc_mapped_onto_the_roles(tmp_path):
    p = tmp_path / 'car.dbc'
    p.write_text(USER_DBC)
    msgs, notes, reader = C.load_dbc(str(p))
    if reader == 'falcon':
        assert any('multiplexed' in n for n in notes)
    rm = C.RoleMap(msgs, {'SPEED_FL': 'VEH_STATE.WhlSpdFL', 'YAW_RATE': 'VEH_STATE.YawRt',
                          'COUNTER': 'VEH_STATE.Alive', 'BRAKE_REQ': 'BRK_CMD.PressReq'})
    assert rm.check() == []
    frames = rm.encode_out({'SPEED_FL': 87.65, 'YAW_RATE': -12.34}, C.Counters())
    assert len(frames) == 1 and frames[0].can_id == 768
    data = frames[0].data
    ws = next(s for s in msgs[0].signals if s.name == 'WhlSpdFL')
    assert C.get_raw(ws, data) * 0.01 == pytest.approx(87.65)
    assert data[0] == (8765 >> 8) and data[1] == 8765 & 0xFF            # big-endian on the wire
    cmd = bytearray(8)
    C.set_raw(next(s for s in msgs[1].signals if s.name == 'PressReq'), cmd, 455)
    assert rm.decode_in(C.Frame(769, bytes(cmd))) == {'BRAKE_REQ': pytest.approx(45.5)}
    assert rm.decode_in(C.Frame(1, b'')) is None
    bad = C.RoleMap(msgs, {'SPEED_FL': 'VEH_STATE.Nope', 'WHAT': 'X.Y', 'YAW_RATE': 'BRK_CMD.PressReq'})
    problems = ' | '.join(bad.check())
    assert 'no signal Nope' in problems and 'not a role' in problems and "'bar'" in problems
    assert 'no BRAKE_REQ' in problems


# ============================================================================================ buses
def test_fake_bus_carries_classical_and_fd_frames_with_kernel_timestamps():
    a, b = B.fake_pair(fd=True)
    try:
        sent = [C.Frame(0x0A0, bytes(range(8))), C.frame('CHASSIS_FD', {'YAW_RATE': 3.25, 'SIM_TIME': 12.5}, 7)]
        for fr in sent:
            a.send(fr)
        got = [b.recv(1.0) for _ in sent]
        assert got[0].frame.data == bytes(range(8)) and not got[0].frame.fd
        assert got[1].frame.fd and got[1].frame.brs and len(got[1].frame.data) == 64
        v = C.decode(C.BY_NAME['CHASSIS_FD'], got[1].frame.data)
        assert v['_crc_ok'] and v['YAW_RATE'] == pytest.approx(3.25) and v['SIM_TIME'] == pytest.approx(12.5)
        for r in got:                                                  # the kernel's time, then ours
            assert r.t_kernel_ns is not None and r.t_kernel_ns <= r.t_read_ns + 1_000_000
            assert r.t_read_ns - r.t_kernel_ns < 1_000_000_000
        assert b.recv(0.01) is None
    finally:
        a.close()
        b.close()
    a, b = B.fake_pair(fd=False)
    with pytest.raises(ValueError):
        a.send(C.Frame(1, b'', fd=True))                               # no FD frames on a classical bus
    a.close()
    b.close()


def test_simulated_link_between_two_ends():
    name = f'sim:t-{os.getpid()}'
    x, y = A.CanLink(name, 'on', 'a'), A.CanLink(name, 'on', 'b')
    try:
        assert x.fd and y.fd and x.kind == 'sim'
        x.send(C.pack_frame(C.Frame(5, b'hi', fd=True)))
        raw, ts = y.recv(1.0)
        assert C.unpack_frame(raw).data[:2] == b'hi' and ts is not None
        with pytest.raises(OSError):
            A.CanLink(name, 'on', 'a')                                 # one process per end
    finally:
        x.close()
        y.close()
    lone = A.CanLink(f'sim:lone-{os.getpid()}', 'off', 'a')
    lone.send(C.pack_frame(C.Frame(1, b'x')))                          # nobody there: lost, as on an empty bus
    assert lone.lost == 1
    lone.close()


def test_python_can_adapter_through_a_stand_in(monkeypatch):
    sent = []

    class Msg:
        def __init__(self, arbitration_id=0, data=b'', is_extended_id=False, is_fd=False, bitrate_switch=False,
                     is_error_frame=False, is_remote_frame=False, timestamp=0.0):
            self.__dict__.update(locals())

    class Bus:
        def __init__(self, **kw):
            self.kw = kw
            self.q = [Msg(0x0B1, bytes(8), timestamp=time.time()), Msg(0x40, bytes(8), is_error_frame=True)]

        def send(self, m):
            sent.append(m)

        def recv(self, timeout):
            return self.q.pop(0) if self.q else None

        def shutdown(self):
            sent.append('shutdown')
    monkeypatch.setitem(sys.modules, 'can', types.SimpleNamespace(Bus=Bus, Message=Msg))
    bus = B.PythonCanBus('slcan', '/dev/ttyACM0', fd=True)
    assert bus.bus.kw['fd'] and bus.bus.kw['data_bitrate'] == 2000000 and not bus.bus.kw['receive_own_messages']
    bus.send(C.Frame(0x0C0, bytes(64), fd=True, brs=True))
    assert sent[0].is_fd and sent[0].bitrate_switch and sent[0].arbitration_id == 0x0C0
    r = bus.recv(0.1)
    assert r.frame.can_id == 0x0B1 and r.t_kernel_ns is not None
    assert bus.recv(0.1).frame.error and bus.recv(0.1) is None
    bus.close()
    assert sent[-1] == 'shutdown'


# ============================================================================================ the bridge's interlocks
class Clock:
    def __init__(self):
        self.t = 1_000_000_000

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += int(ms * 1e6)


def _bridge(fd=False, **kw):
    a, b = B.fake_pair(fd=fd)
    clk = Clock()
    br = B.Bridge(a, Chassis({'surface': 'dry_asphalt'}), clock=clk, **kw)
    return br, b, clk


def _cmd(br, dev, cnt, step, steer=0.0, brake=0.0, throttle=10.0, enable=1, fd=False):
    names = (('CMD_FD',) if fd else ('CMD_STEER', 'CMD_LONG'))
    for n in names:
        v = {'STEER_REQ': steer, 'STEER_MODE': 1, 'ENABLE': enable, 'BRAKE_REQ': brake, 'THROTTLE_REQ': throttle,
             'STEP': step}
        dev.send(C.frame(n, v, cnt.next(n)))
        br.handle(br.bus.recv(1.0))


def _drain(bus):
    out = []
    while True:
        r = bus.recv(0.01)
        if r is None:
            return out
        out.append(r.frame)


def test_watchdog_holds_the_chassis_and_control_comes_back_after_valid_commands():
    br, dev, clk = _bridge()
    cnt = C.Counters()
    br.tick()
    assert br.mode == 'idle' and br.ch.cmd['safe']                    # held before any command
    for k in range(B.RECOVER):
        clk.advance(20)
        _cmd(br, dev, cnt, k, steer=30.0, throttle=20.0)
        br.tick()
    assert br.mode == 'active' and br.ch.cmd['steer_req_deg'] == 30.0 and br.ch.cmd['throttle_pct'] == 20.0
    assert br.arrivals[2][0] is not None                               # the kernel's time of step 2's command
    clk.advance(99)
    br.tick()
    assert br.mode == 'active'
    clk.advance(2)                                                     # 101 ms without a command
    br.tick()
    assert br.mode == 'safe_hold' and br.stats['watchdog_trips'] == 1 and br.ch.cmd['safe']
    assert br.ch.cmd['brake_bar'] == 40.0 and br.ch.cmd['throttle_pct'] == 0.0
    clk.advance(500)
    br.tick()
    assert br.stats['watchdog_trips'] == 1                             # one trip, not one per tick
    fb = [f for f in _drain(dev) if f.can_id == C.BY_NAME['CHASSIS_STATUS'].can_id]
    assert C.decode(C.BY_NAME['CHASSIS_STATUS'], fb[-1].data)['MODE'] == C.MODES['safe_hold']
    for k in range(B.RECOVER - 1):                                     # control comes back only after RECOVER
        clk.advance(20)
        _cmd(br, dev, cnt, 10 + k)
        br.tick()
        assert br.mode == 'safe_hold'
    clk.advance(20)
    _cmd(br, dev, cnt, 20)
    br.tick()
    assert br.mode == 'active'
    clk.advance(20)
    _cmd(br, dev, cnt, 21, enable=0)                                   # the device lets go: held, not a trip
    br.tick()
    assert br.mode == 'safe_hold' and br.stats['watchdog_trips'] == 1 and br.stats['disabled'] == 1


def test_bad_frames_are_rejected_and_do_not_feed_the_watchdog():
    br, dev, clk = _bridge()
    cnt = C.Counters()
    for k in range(B.RECOVER):
        clk.advance(20)
        _cmd(br, dev, cnt, k)
    br.tick()
    assert br.mode == 'active'
    m = C.BY_NAME['CMD_LONG']
    data, _ = C.encode(m, {'BRAKE_REQ': 99.0, 'ENABLE': 1}, cnt.next('CMD_LONG'))
    bad = bytearray(data)
    bad[-1] ^= 0xFF
    dev.send(C.Frame(m.can_id, bytes(bad)))
    br.handle(br.bus.recv(1.0))
    assert br.stats['rejected_crc'] == 1 and br.parts['long'][1]['BRAKE_REQ'] != 99.0
    same = (cnt.c['CMD_LONG'] - 1) % 16                                # the last accepted counter again: stuck
    dev.send(C.Frame(m.can_id, C.encode(m, {'BRAKE_REQ': 99.0, 'ENABLE': 1}, same)[0]))
    br.handle(br.bus.recv(1.0))
    assert br.stats['rejected_counter'] == 1
    dev.send(C.Frame(0x7FF, b'\x00'))                                  # someone else's frame
    br.handle(br.bus.recv(1.0))
    assert br.stats['foreign_frames'] == 1
    clk.advance(150)                                                   # only rejected frames since: the watchdog trips
    br.tick()
    assert br.mode == 'safe_hold'


def test_feedback_frames_classical_and_fd():
    br, dev, clk = _bridge()
    br.tick()
    clk.advance(10)
    br.tick()
    names = [C.BY_ID[f.can_id].name for f in _drain(dev)]
    assert names.count('WHEEL_TICKS') == 2 and names.count('CHASSIS_STATUS') == 1     # the status at 10 Hz
    assert set(names) == set(C.FEEDBACK) | {'CHASSIS_STATUS'}
    br, dev, clk = _bridge(fd=True)
    br.tick()
    fr = _drain(dev)
    assert [C.BY_ID[f.can_id].name for f in fr] == ['CHASSIS_FD', 'CHASSIS_STATUS'] and fr[0].fd and fr[0].brs
    cnt = C.Counters()
    for k in range(B.RECOVER):                                         # commands as one FD frame
        clk.advance(20)
        _cmd(br, dev, cnt, 100000 + k, brake=12.0, fd=True)
    br.tick()
    assert br.mode == 'active' and br.ch.cmd['brake_bar'] == 12.0 and (100002 & 0xFFFF) in br.arrivals


def test_an_error_frame_aborts_and_the_bus_goes_silent():
    br, dev, clk = _bridge()
    br.tick()
    _drain(dev)
    dev.send(C.Frame(0x040, bytes(8), error=True))                     # bus-off
    br.handle(br.bus.recv(1.0))
    assert br.aborted and 'bus_off' in br.aborted and br.mode == 'aborted' and br.ch.cmd['safe']
    clk.advance(10)
    br.tick()
    assert _drain(dev) == []                                           # nothing more is sent
    sc = B.SessionCan.__new__(B.SessionCan)
    sc.bridge = br
    with pytest.raises(B.BusError):
        sc.check()


def test_the_threads_hold_trip_and_stop_silent():
    """The bridge on its own threads at 100 Hz: a device answering at 50 Hz takes control, stops, the watchdog trips
    within about 100 ms; after stop() nothing more reaches the bus."""
    a, dev = B.fake_pair()
    br = B.Bridge(a, Chassis()).start()
    cnt = C.Counters()
    seen = []
    stop_dev = threading.Event()

    def reader():
        while not stop_dev.is_set():
            try:
                r = dev.recv(0.05)
            except OSError:                                            # the bridge's end closed: nothing more comes
                time.sleep(0.05)
                continue
            if r is not None:
                seen.append((time.monotonic(), r.frame))
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for k in range(15):
            for n in ('CMD_STEER', 'CMD_LONG'):
                dev.send(C.frame(n, {'ENABLE': 1, 'THROTTLE_REQ': 10.0, 'STEP': k, 'STEER_MODE': 1}, cnt.next(n)))
            time.sleep(0.02)
        assert br.snapshot()['mode'] == 'active'
        t_last = time.monotonic()
        while br.snapshot()['mode'] == 'active' and time.monotonic() - t_last < 2:
            time.sleep(0.005)
        tripped = time.monotonic() - t_last
        assert br.snapshot()['mode'] == 'safe_hold' and br.snapshot()['watchdog_trips'] == 1
        assert 0.08 < tripped < 0.4, tripped
    finally:
        br.stop()
    n_at_stop = len(seen)
    time.sleep(0.15)
    stop_dev.set()
    t.join(1)
    assert len(seen) == n_at_stop                                      # silent after stop
    assert len([f for _, f in seen if f.can_id == 0x0A0]) > 20         # it did send at about 100 Hz before
    dev.close()


# ============================================================================================ the allow-list
@pytest.fixture
def hil_settings(settings, monkeypatch):
    from falcon_autonomy.core import settings as S
    monkeypatch.setitem(S.DEFAULTS, 'hil_can', S.DEFAULTS.get('hil_can') or {'interfaces_allowed': ['vcan*']})
    monkeypatch.setattr(B, 'is_virtual', lambda name: name.startswith('vcan'))
    return settings


def test_the_allow_list(hil_settings):
    st = hil_settings
    assert B.check_interface('vcan0', st)[0] and B.check_interface('sim:x', st)[0]
    ok, why, _ = B.check_interface('can0', st)
    assert not ok and 'not in the CAN interfaces' in why
    with pytest.raises(PermissionError):
        B.open_bus('can0', st)
    st.set('hil_can', {'interfaces_allowed': ['vcan*', 'can*']})
    ok, why, _ = B.check_interface('can0', st)                         # allowed by pattern, but a real adapter
    assert not ok and 'bench network, not a vehicle' in why
    with pytest.raises(ValueError):
        B.confirm_bench(st, 'can0', 'yes')
    with pytest.raises(ValueError):
        B.confirm_bench(st, 'sim:x', B.CONFIRM_TEXT)
    B.confirm_bench(st, 'can0', 'Bench network, not a vehicle')
    ok, why, _ = B.check_interface('can0', st)
    assert ok and 'confirmed' in why and st.get('hil_can')['confirmed']['can0']['at']
    assert not B.check_interface('can1', st)[0]                        # each adapter on its own
    assert not B.check_interface('python-can:slcan:/dev/ttyACM0', st)[0]
    B.confirm_bench(st, 'python-can:slcan:/dev/ttyACM0', B.CONFIRM_TEXT)
    assert B.check_interface('python-can:slcan:/dev/ttyACM0', st)[0]
    B.revoke_bench(st, 'can0')
    assert not B.check_interface('can0', st)[0]
    st.set('hil_can', {'interfaces_allowed': ['vcan*']})
    B.confirm_bench(st, 'can5', B.CONFIRM_TEXT)                        # confirming adds it to the allowed list
    assert 'can5' in st.get('hil_can')['interfaces_allowed'] and B.check_interface('can5', st)[0]
