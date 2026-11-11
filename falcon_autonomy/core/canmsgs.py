"""The CAN messages between Falcon's chassis emulation and a device (docs/hardware.md#can): the message table, its
codec, the Linux SocketCAN frame layout, error frames, the generated DBC (resources/can/falcon_chassis.dbc) and a
reader of a user's DBC mapped onto Falcon's signal roles.

Standard library only, Python ≥ 3.10: falcon hil deploy copies this file verbatim beside the device agent as
falcon_canmsgs.py, so the host's bridge (hil.canbridge) and the device encode and decode with one definition.

Frames. A classical frame is Linux's struct can_frame (16 bytes: can_id u32, len u8, 3 pad bytes, 8 data bytes); a
CAN FD frame is struct canfd_frame (72 bytes: can_id u32, len u8, flags u8, 2 reserved, 64 data bytes; flags BRS = bit-rate
switch, ESI). Both in the host's byte order, as the kernel reads and writes them on an AF_CAN raw socket.

Messages. Every signal is little-endian (Intel; DBC "@1"), standard 11-bit identifiers. Every message of the table
ends with an alive counter (4 bits, the low nibble of the byte before the last) and a CRC-8 (the last byte): SAE J1850
(polynomial 0x1D, initial 0xFF, final XOR 0xFF — the CRC of AUTOSAR's E2E profile 1) over the identifier's two bytes
(little-endian) then every byte of the message but the CRC, so a frame copied onto another identifier fails its check.
Feedback (FALCON_CHASSIS → DEVICE) every 10 ms, the status every 100 ms; the device's commands every 20 ms (a command
older than 100 ms trips the bridge's watchdog: hil.canbridge). With CAN FD (the interface's MTU is 72) the feedback
goes as one 64-byte frame (CHASSIS_FD, bit-rate switched) and a command may come as CMD_FD; the classical messages
stay valid in both.

Signal conventions: ISO 8855 (x forward, y left, z up): yaw rate and steering angle positive to the left; the steering
angle is the steering wheel's; wheel ticks are the edges of an ABS tone wheel counted since the bridge started,
wrapping at the signal's width — they come from the wheels' rotation, so a slipping or locked wheel shows in them.
"""
from __future__ import annotations

import fnmatch
import re
import struct
from dataclasses import dataclass, field

# ============================================================================================ Linux SocketCAN frames
CAN_MTU, CANFD_MTU = 16, 72
CAN_EFF_FLAG, CAN_RTR_FLAG, CAN_ERR_FLAG = 0x80000000, 0x40000000, 0x20000000
CAN_SFF_MASK, CAN_EFF_MASK, CAN_ERR_MASK = 0x7FF, 0x1FFFFFFF, 0x1FFFFFFF
CANFD_BRS, CANFD_ESI, CANFD_FDF = 0x01, 0x02, 0x04
CAN_FRAME = struct.Struct('=IB3x8s')
CANFD_FRAME = struct.Struct('=IBBxx64s')
FD_LENGTHS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)
# linux/can/error.h: the class bits of an error frame's identifier, and data[1]'s controller states
ERR_CLASSES = ((0x001, 'tx_timeout'), (0x002, 'lost_arbitration'), (0x004, 'controller'), (0x008, 'protocol_violation'),
               (0x010, 'transceiver'), (0x020, 'no_ack'), (0x040, 'bus_off'), (0x080, 'bus_error'),
               (0x100, 'restarted'), (0x200, 'error_counters'))
ERR_CRTL = ((0x01, 'rx_overflow'), (0x02, 'tx_overflow'), (0x04, 'rx_warning'), (0x08, 'tx_warning'),
            (0x10, 'rx_passive'), (0x20, 'tx_passive'), (0x40, 'back_to_active'))


@dataclass
class Frame:
    """One CAN frame: a standard (11-bit) or extended (29-bit) identifier, its data, FD or classical."""
    can_id: int
    data: bytes = b''
    fd: bool = False
    brs: bool = False
    extended: bool = False
    rtr: bool = False
    error: bool = False

    @property
    def dlc_len(self) -> int:
        return len(self.data)


def fd_length(n: int) -> int:
    """The smallest CAN FD payload length that holds n bytes."""
    for k in FD_LENGTHS:
        if k >= n:
            return k
    raise ValueError(f'{n} bytes do not fit a CAN FD frame (64 at most)')


def pack_frame(fr: Frame) -> bytes:
    """struct can_frame (16 bytes) or struct canfd_frame (72 bytes) of a frame."""
    cid = fr.can_id & (CAN_EFF_MASK if fr.extended else CAN_SFF_MASK)
    if fr.extended:
        cid |= CAN_EFF_FLAG
    if fr.rtr:
        cid |= CAN_RTR_FLAG
    if fr.error:
        cid = (fr.can_id & CAN_ERR_MASK) | CAN_ERR_FLAG
    if fr.fd:
        n = fd_length(len(fr.data))
        return CANFD_FRAME.pack(cid, n, (CANFD_BRS if fr.brs else 0) | CANFD_FDF,
                                bytes(fr.data).ljust(64, b'\x00'))
    if len(fr.data) > 8:
        raise ValueError(f'a classical frame holds 8 bytes, not {len(fr.data)}')
    return CAN_FRAME.pack(cid, len(fr.data), bytes(fr.data).ljust(8, b'\x00'))


def unpack_frame(raw) -> Frame:
    """A frame from the kernel's 16- or 72-byte layout."""
    raw = bytes(raw)
    if len(raw) == CANFD_MTU:
        cid, n, flags, data = CANFD_FRAME.unpack(raw)
        fd, brs = True, bool(flags & CANFD_BRS)
    elif len(raw) == CAN_MTU:
        cid, n, data = CAN_FRAME.unpack(raw)
        fd, brs = False, False
        n = min(n, 8)
    else:
        raise ValueError(f'a CAN frame is 16 or 72 bytes, not {len(raw)}')
    err = bool(cid & CAN_ERR_FLAG)
    ext = bool(cid & CAN_EFF_FLAG)
    ident = cid & CAN_ERR_MASK if err else cid & (CAN_EFF_MASK if ext else CAN_SFF_MASK)
    return Frame(ident, data[:n], fd=fd, brs=brs, extended=ext, rtr=bool(cid & CAN_RTR_FLAG), error=err)


def describe_error(fr: Frame) -> list:
    """What an error frame says (linux/can/error.h): its classes, and the controller's state when it reports one."""
    out = [name for bit, name in ERR_CLASSES if fr.can_id & bit]
    if fr.can_id & 0x004 and len(fr.data) > 1:
        out += [name for bit, name in ERR_CRTL if fr.data[1] & bit]
    return out or ['unspecified']


# ============================================================================================ CRC and counters
def _crc8_table():
    t = []
    for b in range(256):
        c = b
        for _ in range(8):
            c = ((c << 1) ^ 0x1D) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        t.append(c)
    return tuple(t)


_CRC8 = _crc8_table()


def crc8_j1850(data: bytes, crc: int = 0xFF) -> int:
    """CRC-8 SAE J1850: polynomial 0x1D, initial 0xFF, final XOR 0xFF (check value of b'123456789' is 0x4B)."""
    for b in data:
        crc = _CRC8[crc ^ b]
    return crc ^ 0xFF


def frame_crc(can_id: int, body: bytes) -> int:
    return crc8_j1850(bytes((can_id & 0xFF, (can_id >> 8) & 0xFF)) + bytes(body))


# ============================================================================================ the message table
@dataclass
class Signal:
    name: str
    start: int                      # DBC start bit: Intel (little-endian) = the LSB's bit; Motorola = the MSB's
    length: int
    scale: float = 1.0
    offset: float = 0.0
    signed: bool = False
    unit: str = ''
    about: str = ''
    little_endian: bool = True
    minimum: float | None = None
    maximum: float | None = None
    values: dict = field(default_factory=dict)          # raw → name (DBC VAL_)

    def raw_range(self) -> tuple:
        if self.signed:
            return -(1 << (self.length - 1)), (1 << (self.length - 1)) - 1
        return 0, (1 << self.length) - 1

    def phys_range(self) -> tuple:
        lo, hi = self.raw_range()
        a, b = lo * self.scale + self.offset, hi * self.scale + self.offset
        mn = self.minimum if self.minimum is not None else min(a, b)
        mx = self.maximum if self.maximum is not None else max(a, b)
        return mn, mx


@dataclass
class Message:
    can_id: int
    name: str
    length: int
    sender: str
    period_ms: int
    signals: list
    about: str = ''
    fd: bool = False
    extended: bool = False

    def signal(self, name: str) -> Signal:
        for s in self.signals:
            if s.name == name:
                return s
        raise KeyError(f'{self.name} has no signal {name}')

    @property
    def protected(self) -> bool:
        """Ends with Falcon's alive counter and CRC-8."""
        names = [s.name for s in self.signals]
        return 'COUNTER' in names and 'CRC' in names


def _tail(n: int) -> list:
    """The alive counter and the CRC of an n-byte message (its last two bytes)."""
    return [Signal('COUNTER', 8 * (n - 2), 4, about='alive counter, +1 per frame, modulo 16'),
            Signal('CRC', 8 * (n - 1), 8, about='CRC-8 SAE J1850 over the identifier (2 bytes, little-endian) and '
                                                'the bytes before this one')]


_MODES = {0: 'idle', 1: 'active', 2: 'safe_hold', 3: 'aborted'}
_SURFACES = {0: 'dry_asphalt', 1: 'wet_asphalt', 2: 'waterlogged', 3: 'gravel', 4: 'snow', 5: 'ice'}
_STEER_MODES = {0: 'off', 1: 'angle', 2: 'torque'}


def _feedback() -> list:
    W = ('FL', 'FR', 'RL', 'RR')
    ticks = [Signal(f'TICKS_{w}', 12 * i, 12, unit='ticks', about=f'{w} wheel: tone-wheel edges since start, wraps at '
                                                                     '4096') for i, w in enumerate(W)]
    speeds = [Signal(f'SPEED_{w}', 12 * i, 12, 0.0625, unit='km/h', about=f'{w} wheel speed (its rotation × the '
                                                                           'rolling radius: slip shows)')
              for i, w in enumerate(W)]
    return [
        Message(0x0A0, 'WHEEL_TICKS', 8, 'FALCON_CHASSIS', 10,
                ticks + [Signal('QUALIFIER', 52, 4, about='bit per wheel FL FR RL RR: 1 = valid')] + _tail(8),
                'Wheel tick counters (ABS tone wheel edges) of the four wheels'),
        Message(0x0A1, 'WHEEL_SPEEDS', 8, 'FALCON_CHASSIS', 10, speeds + _tail(8),
                'Wheel speeds from the wheels\' rotation'),
        Message(0x0A2, 'YAW_ACCEL', 8, 'FALCON_CHASSIS', 10,
                [Signal('YAW_RATE', 0, 16, 0.01, signed=True, unit='deg/s', about='yaw rate, left positive'),
                 Signal('LAT_ACCEL', 16, 16, 0.001, signed=True, unit='m/s2', about='lateral acceleration, left '
                                                                                     'positive'),
                 Signal('LONG_ACCEL', 32, 16, 0.001, signed=True, unit='m/s2', about='longitudinal acceleration'),
                 Signal('STATUS', 52, 4, about='0 = valid')] + _tail(8),
                'Yaw rate and accelerations (the vehicle\'s, ISO 8855)'),
        Message(0x0A3, 'STEERING', 8, 'FALCON_CHASSIS', 10,
                [Signal('STEER_ANGLE', 0, 16, 0.1, signed=True, unit='deg', about='steering wheel angle, left '
                                                                                  'positive'),
                 Signal('STEER_RATE', 16, 16, 1.0, signed=True, unit='deg/s', about='steering wheel angle rate'),
                 Signal('EPS_TORQUE', 32, 16, 0.01, signed=True, unit='Nm', about='torque the EPS applies'),
                 Signal('EPS_STATUS', 52, 4, values={0: 'ok', 1: 'safe_hold', 2: 'fault'})] + _tail(8),
                'The steering (EPS) as it is'),
        Message(0x0A4, 'BRAKE', 8, 'FALCON_CHASSIS', 10,
                [Signal('BRAKE_PRESSURE', 0, 16, 0.01, unit='bar', about='brake line pressure'),
                 Signal('BRAKE_REQ_ECHO', 16, 16, 0.01, unit='bar', about='the pressure being requested (after the '
                                                                          'watchdog)'),
                 Signal('VEHICLE_SPEED', 32, 16, 0.01, unit='km/h', about='vehicle speed (over ground)'),
                 Signal('BRAKE_STATUS', 52, 4, values={0: 'released', 1: 'applied', 2: 'safe_hold'})] + _tail(8),
                'The brake as it is, and the vehicle speed'),
        Message(0x0A5, 'CHASSIS_STATUS', 8, 'FALCON_CHASSIS', 100,
                [Signal('MODE', 0, 4, values=dict(_MODES)), Signal('SURFACE', 4, 4, values=dict(_SURFACES)),
                 Signal('MU_EFF', 8, 8, 0.01, about='the tyre–road friction coefficient in effect'),
                 Signal('WATCHDOG_TRIPS', 16, 16, about='commands that went stale (> 100 ms) since start'),
                 Signal('CMD_REJECTED', 32, 16, about='command frames rejected (CRC, counter, range)'),
                 Signal('HYDROPLANING', 52, 1, about='1 = the speed is past the hydroplaning onset')] + _tail(8),
                'The emulation\'s state'),
    ]


def _commands() -> list:
    return [
        Message(0x0B0, 'CMD_STEER', 8, 'DEVICE', 20,
                [Signal('STEER_REQ', 0, 16, 0.1, signed=True, unit='deg', about='steering wheel angle request '
                                                                                '(angle mode), left positive'),
                 Signal('STEER_TORQUE_REQ', 16, 16, 0.01, signed=True, unit='Nm', about='EPS torque request '
                                                                                        '(torque mode)'),
                 Signal('STEER_MODE', 32, 2, values=dict(_STEER_MODES)),
                 Signal('ENABLE', 34, 1, about='1 = the device asks to be in control'),
                 Signal('STEP', 36, 12, about='the simulation step the command answers (low 12 bits)')] + _tail(8),
                'Steering command'),
        Message(0x0B1, 'CMD_LONG', 8, 'DEVICE', 20,
                [Signal('BRAKE_REQ', 0, 16, 0.01, unit='bar', about='brake pressure request'),
                 Signal('THROTTLE_REQ', 16, 8, 0.5, unit='%', maximum=100.0, about='accelerator pedal'),
                 Signal('ENABLE', 24, 1, about='1 = the device asks to be in control'),
                 Signal('STEP', 32, 16, about='the simulation step the command answers (low 16 bits)')] + _tail(8),
                'Longitudinal command (brake, throttle)'),
    ]


def _fd() -> list:
    W = ('FL', 'FR', 'RL', 'RR')
    fb = [Signal(f'TICKS_{w}', 16 * i, 16, unit='ticks', about='wraps at 65536') for i, w in enumerate(W)]
    fb += [Signal(f'SPEED_{w}', 64 + 16 * i, 16, 0.01, unit='km/h') for i, w in enumerate(W)]
    fb += [Signal('YAW_RATE', 128, 16, 0.01, signed=True, unit='deg/s'),
           Signal('LAT_ACCEL', 144, 16, 0.001, signed=True, unit='m/s2'),
           Signal('LONG_ACCEL', 160, 16, 0.001, signed=True, unit='m/s2'),
           Signal('STEER_ANGLE', 176, 16, 0.1, signed=True, unit='deg'),
           Signal('STEER_RATE', 192, 16, 1.0, signed=True, unit='deg/s'),
           Signal('EPS_TORQUE', 208, 16, 0.01, signed=True, unit='Nm'),
           Signal('BRAKE_PRESSURE', 224, 16, 0.01, unit='bar'),
           Signal('VEHICLE_SPEED', 240, 16, 0.01, unit='km/h'),
           Signal('MODE', 256, 4, values=dict(_MODES)), Signal('SURFACE', 260, 4, values=dict(_SURFACES)),
           Signal('MU_EFF', 264, 8, 0.01), Signal('HYDROPLANING', 272, 1),
           Signal('SIM_TIME', 288, 32, 0.001, unit='s', about='simulation time of the feedback, modulo 2^32 ms')]
    cmd = [Signal('STEER_REQ', 0, 16, 0.1, signed=True, unit='deg'),
           Signal('STEER_TORQUE_REQ', 16, 16, 0.01, signed=True, unit='Nm'),
           Signal('STEER_MODE', 32, 2, values=dict(_STEER_MODES)), Signal('ENABLE', 34, 1),
           Signal('BRAKE_REQ', 40, 16, 0.01, unit='bar'), Signal('THROTTLE_REQ', 56, 8, 0.5, unit='%', maximum=100.0),
           Signal('STEP', 64, 32, about='the simulation step the command answers')]
    return [Message(0x0C0, 'CHASSIS_FD', 64, 'FALCON_CHASSIS', 10, fb + _tail(64),
                    'All feedback in one CAN FD frame (bit-rate switched)', fd=True),
            Message(0x0C1, 'CMD_FD', 64, 'DEVICE', 20, cmd + _tail(64), 'Steering and longitudinal command in one '
                                                                           'CAN FD frame', fd=True)]


TABLE = _feedback() + _commands() + _fd()
BY_ID = {m.can_id: m for m in TABLE}
BY_NAME = {m.name: m for m in TABLE}
FEEDBACK = ('WHEEL_TICKS', 'WHEEL_SPEEDS', 'YAW_ACCEL', 'STEERING', 'BRAKE')
COMMANDS = ('CMD_STEER', 'CMD_LONG', 'CMD_FD')
STATUS = 'CHASSIS_STATUS'
MODES = {v: k for k, v in _MODES.items()}
SURFACES = {v: k for k, v in _SURFACES.items()}
STEER_MODES = {v: k for k, v in _STEER_MODES.items()}


# ============================================================================================ codec
def _bits_intel(data: bytes, start: int, length: int) -> int:
    return (int.from_bytes(data, 'little') >> start) & ((1 << length) - 1)


def _motorola_positions(start: int, length: int) -> list:
    """The bit positions (byte·8 + bit, bit 0 = LSB of its byte) of a big-endian (Motorola) signal, MSB first, from
    its DBC start bit (the MSB's position) — the DBC "sawtooth" numbering."""
    pos, out = start, []
    for _ in range(length):
        out.append(pos)
        pos = pos + 15 if pos % 8 == 0 else pos - 1
    return out


def get_raw(sig: Signal, data: bytes) -> int:
    if sig.little_endian:
        v = _bits_intel(data, sig.start, sig.length)
    else:
        v = 0
        for p in _motorola_positions(sig.start, sig.length):
            v = (v << 1) | ((data[p // 8] >> (p % 8)) & 1)
    if sig.signed and v & (1 << (sig.length - 1)):
        v -= 1 << sig.length
    return v


def set_raw(sig: Signal, buf: bytearray, raw: int):
    raw &= (1 << sig.length) - 1
    if sig.little_endian:
        n = int.from_bytes(buf, 'little')
        mask = ((1 << sig.length) - 1) << sig.start
        n = (n & ~mask) | (raw << sig.start)
        buf[:] = n.to_bytes(len(buf), 'little')
        return
    for i, p in enumerate(_motorola_positions(sig.start, sig.length)):
        bit = (raw >> (sig.length - 1 - i)) & 1
        buf[p // 8] = (buf[p // 8] & ~(1 << (p % 8))) | (bit << (p % 8))


def to_raw(sig: Signal, value: float) -> tuple:
    """(raw, saturated): the physical value as the signal's integer, clipped to what it can hold."""
    lo, hi = sig.raw_range()
    mn, mx = sig.phys_range()
    v = min(max(float(value), mn), mx)
    raw = int(round((v - sig.offset) / sig.scale))
    clipped = min(max(raw, lo), hi)
    return clipped, clipped != raw or v != float(value)


def encode(msg: Message, values: dict, counter: int = 0) -> tuple:
    """(data bytes, saturated signal names). Missing signals are 0; the counter and the CRC are filled in."""
    buf = bytearray(msg.length)
    sat = []
    for s in msg.signals:
        if s.name in ('COUNTER', 'CRC'):
            continue
        raw, was = to_raw(s, values.get(s.name, 0.0))
        if was:
            sat.append(s.name)
        set_raw(s, buf, raw)
    if msg.protected:
        set_raw(msg.signal('COUNTER'), buf, counter & 0xF)
        buf[-1] = frame_crc(msg.can_id, bytes(buf[:-1]))
    return bytes(buf), sat


def decode(msg: Message, data: bytes) -> dict:
    """The signals' physical values; '_crc_ok' when the message is protected (False: do not use it)."""
    data = bytes(data).ljust(msg.length, b'\x00')[:msg.length]
    out = {s.name: get_raw(s, data) * s.scale + s.offset for s in msg.signals}
    if msg.protected:
        out['_crc_ok'] = data[-1] == frame_crc(msg.can_id, data[:-1])
        out['COUNTER'] = int(out['COUNTER'])
    return out


def frame(name: str, values: dict, counter: int = 0, brs: bool = True) -> Frame:
    m = BY_NAME[name]
    data, _ = encode(m, values, counter)
    return Frame(m.can_id, data, fd=m.fd, brs=m.fd and brs, extended=m.extended)


class Counters:
    """Alive counters per message: next(name) → 0, 1, …, 15, 0 …"""

    def __init__(self):
        self.c: dict = {}

    def next(self, name: str) -> int:
        v = self.c.get(name, -1) + 1 & 0xF
        self.c[name] = v
        return v


def counter_ok(prev, cur: int) -> bool:
    """An alive counter moved on by 1 to 3 (up to two frames lost is tolerated; a repeat or a jump is not)."""
    return prev is None or 1 <= (cur - prev) % 16 <= 3


# ============================================================================================ the DBC
_NS = ('NS_DESC_', 'CM_', 'BA_DEF_', 'BA_', 'VAL_', 'CAT_DEF_', 'CAT_', 'FILTER', 'BA_DEF_DEF_', 'EV_DATA_',
       'ENVVAR_DATA_', 'SGTYPE_', 'SGTYPE_VAL_', 'BA_DEF_SGTYPE_', 'BA_SGTYPE_', 'SIG_TYPE_REF_', 'VAL_TABLE_',
       'SIG_GROUP_', 'SIG_VALTYPE_', 'SIGTYPE_VALTYPE_', 'BO_TX_BU_', 'BA_DEF_REL_', 'BA_REL_', 'BA_DEF_DEF_REL_',
       'BU_SG_REL_', 'BU_EV_REL_', 'BU_BO_REL_', 'SG_MUL_VAL_')


def _num(x: float) -> str:
    return f'{x:.10g}'


def _q(text: str) -> str:
    return text.replace('\\', '\\\\').replace('"', '\\"')


def dbc_text(table: list | None = None) -> str:
    """The message table as a DBC file (Vector's format; cantools, SavvyCAN and CANdb++ read it). CAN FD messages
    carry VFrameFormat 14 (StandardCAN_FD)."""
    table = table or TABLE
    nodes = sorted({m.sender for m in table} | {'DEVICE', 'FALCON_CHASSIS'})
    out = ['VERSION "falcon_chassis 1"', '', '', 'NS_ :'] + [f'\t{n}' for n in _NS] + ['', 'BS_:', '',
                                                                                       f'BU_: {" ".join(nodes)}', '']
    for m in table:
        rx = 'DEVICE' if m.sender == 'FALCON_CHASSIS' else 'FALCON_CHASSIS'
        out.append(f'BO_ {m.can_id | (CAN_EFF_FLAG if m.extended else 0)} {m.name}: {m.length} {m.sender}')
        for s in m.signals:
            mn, mx = s.phys_range()
            out.append(f' SG_ {s.name} : {s.start}|{s.length}@{1 if s.little_endian else 0}{"-" if s.signed else "+"} '
                       f'({_num(s.scale)},{_num(s.offset)}) [{_num(mn)}|{_num(mx)}] "{_q(s.unit)}" {rx}')
        out.append('')
    out.append('')
    out.append('CM_ "Falcon Autonomy chassis emulation <-> device (docs/hardware.md). Generated from '
               'falcon_autonomy/core/canmsgs.py: do not edit.";')
    for m in table:
        if m.about:
            out.append(f'CM_ BO_ {m.can_id} "{_q(m.about)}";')
        for s in m.signals:
            if s.about:
                out.append(f'CM_ SG_ {m.can_id} {s.name} "{_q(s.about)}";')
    out += ['BA_DEF_ BO_ "GenMsgCycleTime" INT 0 65535;',
            'BA_DEF_ BO_ "VFrameFormat" ENUM "StandardCAN","ExtendedCAN","reserved","reserved","reserved","reserved",'
            '"reserved","reserved","reserved","reserved","reserved","reserved","reserved","reserved","StandardCAN_FD",'
            '"ExtendedCAN_FD";',
            'BA_DEF_ "BusType" STRING ;',
            'BA_DEF_DEF_ "GenMsgCycleTime" 0;',
            'BA_DEF_DEF_ "VFrameFormat" "StandardCAN";',
            'BA_DEF_DEF_ "BusType" "CAN";',
            'BA_ "BusType" "CAN FD";']
    for m in table:
        out.append(f'BA_ "GenMsgCycleTime" BO_ {m.can_id} {m.period_ms};')
        if m.fd:
            out.append(f'BA_ "VFrameFormat" BO_ {m.can_id} {15 if m.extended else 14};')
    for m in table:
        for s in m.signals:
            if s.values:
                vals = ' '.join(f'{k} "{_q(v)}"' for k, v in sorted(s.values.items()))
                out.append(f'VAL_ {m.can_id} {s.name} {vals} ;')
    return '\n'.join(out) + '\n'


_BO = re.compile(r'^BO_\s+(\d+)\s+(\w+)\s*:\s*(\d+)\s+(\w+)')
_SG = re.compile(r'^SG_\s+(\w+)\s*(M|m\d+)?\s*:\s*(\d+)\|(\d+)@([01])([+-])\s*\(([^,]+),([^)]+)\)\s*'
                 r'\[([^|]*)\|([^\]]*)\]\s*"((?:[^"\\]|\\.)*)"')
_FF = re.compile(r'^BA_\s+"VFrameFormat"\s+BO_\s+(\d+)\s+(\d+)')
_CT = re.compile(r'^BA_\s+"GenMsgCycleTime"\s+BO_\s+(\d+)\s+(\d+)')


def parse_dbc(text: str) -> tuple:
    """(messages, notes) from a DBC file's text: BO_ and SG_ lines (both byte orders, signed or not), the cycle
    time and CAN FD frame format attributes. Multiplexed signals are skipped (noted): with cantools installed, a user
    DBC is read by cantools instead (load_dbc)."""
    msgs, notes, cur = {}, [], None
    fmt, cyc = {}, {}
    for line in text.splitlines():
        s = line.strip()
        m = _BO.match(s)
        if m:
            raw = int(m.group(1))
            ext = bool(raw & CAN_EFF_FLAG)
            cid = raw & (CAN_EFF_MASK if ext else CAN_SFF_MASK)
            cur = Message(cid, m.group(2), int(m.group(3)), m.group(4), 0, [], extended=ext)
            msgs[cid] = cur
            continue
        m = _SG.match(s)
        if s.startswith('SG_ ') and m is None:
            notes.append(f'a signal line was not understood: {s[:60]}')
            continue
        if m and cur is not None:
            if m.group(2):
                notes.append(f'{cur.name}.{m.group(1)} is multiplexed: skipped (install cantools to read it)')
                continue

            def f(x):
                return float(x) if x.strip() else None
            cur.signals.append(Signal(m.group(1), int(m.group(3)), int(m.group(4)), float(m.group(7)),
                                      float(m.group(8)), m.group(6) == '-', m.group(11),
                                      little_endian=m.group(5) == '1', minimum=f(m.group(9)), maximum=f(m.group(10))))
            continue
        if not s.startswith('SG_ '):
            cur = None if s.startswith(('BO_TX_BU_', 'CM_', 'BA_', 'VAL_')) else cur
        m = _FF.match(s)
        if m:
            fmt[int(m.group(1)) & CAN_EFF_MASK] = int(m.group(2))
        m = _CT.match(s)
        if m:
            cyc[int(m.group(1)) & CAN_EFF_MASK] = int(m.group(2))
    for cid, msg in msgs.items():
        msg.fd = fmt.get(cid, 0) in (14, 15)
        msg.period_ms = cyc.get(cid, 0)
        for sg in msg.signals:
            if sg.minimum == sg.maximum == 0.0:              # DBC writes [0|0] for "no range given"
                sg.minimum = sg.maximum = None
    return list(msgs.values()), notes


def load_dbc(path: str) -> tuple:
    """A user's DBC: (messages, notes, reader) — through cantools when it is installed (multiplexing, every
    attribute), else this module's reader."""
    try:
        import cantools                                   # optional (the hil extra)
    except ImportError:
        with open(path, encoding='utf-8', errors='replace') as fh:
            msgs, notes = parse_dbc(fh.read())
        return msgs, notes, 'falcon'
    db = cantools.database.load_file(path, strict=False)
    msgs = []
    for m in db.messages:
        sigs = [Signal(s.name, s.start, s.length, float(s.scale), float(s.offset), bool(s.is_signed), s.unit or '',
                       little_endian=s.byte_order == 'little_endian', minimum=s.minimum, maximum=s.maximum)
                for s in m.signals if not s.multiplexer_ids]
        msgs.append(Message(m.frame_id, m.name, m.length, (m.senders or [''])[0], int(m.cycle_time or 0), sigs,
                            fd=bool(getattr(m, 'is_fd', False)), extended=bool(m.is_extended_frame)))
    return msgs, [], 'cantools'


# ============================================================================================ roles: a user's DBC
# the bridge's signals by role: what it sends (feedback) and what it reads (commands); a user's DBC is mapped onto them
OUT_ROLES = ('TICKS_FL', 'TICKS_FR', 'TICKS_RL', 'TICKS_RR', 'SPEED_FL', 'SPEED_FR', 'SPEED_RL', 'SPEED_RR',
             'YAW_RATE', 'LAT_ACCEL', 'LONG_ACCEL', 'STEER_ANGLE', 'STEER_RATE', 'EPS_TORQUE', 'BRAKE_PRESSURE',
             'VEHICLE_SPEED')
IN_ROLES = ('STEER_REQ', 'STEER_TORQUE_REQ', 'STEER_MODE', 'ENABLE', 'BRAKE_REQ', 'THROTTLE_REQ', 'STEP')
ROLE_UNITS = {'SPEED_FL': 'km/h', 'SPEED_FR': 'km/h', 'SPEED_RL': 'km/h', 'SPEED_RR': 'km/h', 'VEHICLE_SPEED': 'km/h',
              'YAW_RATE': 'deg/s', 'STEER_ANGLE': 'deg', 'STEER_RATE': 'deg/s', 'EPS_TORQUE': 'Nm',
              'BRAKE_PRESSURE': 'bar', 'LAT_ACCEL': 'm/s2', 'LONG_ACCEL': 'm/s2', 'STEER_REQ': 'deg',
              'STEER_TORQUE_REQ': 'Nm', 'BRAKE_REQ': 'bar', 'THROTTLE_REQ': '%'}


class RoleMap:
    """A user's DBC mapped onto the bridge's roles: {role: 'MESSAGE.SIGNAL'}. Feedback roles are encoded into their
    messages (their other signals 0; no counter or CRC unless mapped as COUNTER: 'MESSAGE.SIGNAL', since a DBC does
    not say how a vehicle protects its frames); command roles are decoded from theirs. Units are taken as the DBC's:
    map a signal whose unit differs and the check() says so."""

    def __init__(self, messages: list, mapping: dict):
        self.by_name = {m.name: m for m in messages}
        self.by_id = {m.can_id: m for m in messages}
        self.mapping = dict(mapping)
        self.out: dict = {}                   # message name → [(role, signal)]
        self.inp: dict = {}                   # can_id → [(role, signal)]
        self.counters: dict = {}              # message name → counter signal
        for role, ref in self.mapping.items():
            mname, _, sname = str(ref).partition('.')
            m = self.by_name.get(mname)
            if m is None:
                continue
            try:
                sig = m.signal(sname)
            except KeyError:
                continue
            if role == 'COUNTER' or role.startswith('COUNTER:'):
                self.counters[mname] = sig
            elif role in OUT_ROLES:
                self.out.setdefault(mname, []).append((role, sig))
            elif role in IN_ROLES:
                self.inp.setdefault(m.can_id, []).append((role, sig))

    def check(self) -> list:
        """Problems with the mapping: unknown roles, missing messages or signals, unit mismatches."""
        bad = []
        for role, ref in self.mapping.items():
            if role not in OUT_ROLES + IN_ROLES and not role.startswith('COUNTER'):
                bad.append(f'{role}: not a role ({", ".join(OUT_ROLES + IN_ROLES)})')
                continue
            mname, _, sname = str(ref).partition('.')
            m = self.by_name.get(mname)
            if m is None:
                bad.append(f'{role}: no message {mname} in the DBC')
                continue
            try:
                sig = m.signal(sname)
            except KeyError:
                bad.append(f'{role}: {mname} has no signal {sname}')
                continue
            want = ROLE_UNITS.get(role)
            if want and sig.unit and sig.unit.replace('²', '2') != want:
                bad.append(f'{role}: {mname}.{sname} is in {sig.unit!r}, the role in {want!r} (not converted)')
        if not any(r in self.mapping for r in ('BRAKE_REQ',)):
            bad.append('no BRAKE_REQ mapped: the device cannot brake through this DBC')
        return bad

    def encode_out(self, state: dict, counters: Counters) -> list:
        frames = []
        for mname, pairs in self.out.items():
            m = self.by_name[mname]
            buf = bytearray(m.length)
            for role, sig in pairs:
                raw, _ = to_raw(sig, state.get(role, 0.0))
                set_raw(sig, buf, raw)
            if mname in self.counters:
                set_raw(self.counters[mname], buf, counters.next(mname))
            frames.append(Frame(m.can_id, bytes(buf), fd=m.fd, brs=m.fd, extended=m.extended))
        return frames

    def decode_in(self, fr: Frame) -> dict | None:
        pairs = self.inp.get(fr.can_id)
        if not pairs:
            return None
        m = self.by_id[fr.can_id]
        data = bytes(fr.data).ljust(m.length, b'\x00')
        return {role: get_raw(sig, data) * sig.scale + sig.offset for role, sig in pairs}


def matches(name: str, patterns) -> bool:
    """Whether an interface name matches one of the shell patterns (vcan*, can0 …)."""
    return any(fnmatch.fnmatchcase(name, p) for p in patterns or ())
