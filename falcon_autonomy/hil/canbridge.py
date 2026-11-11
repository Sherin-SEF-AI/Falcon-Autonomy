"""The chassis on a CAN bus (docs/hardware.md#can): Falcon's chassis model (core.chassis) as the ECUs a device talks
to — its feedback out every 10 ms (wheel ticks and speeds, yaw rate, steering, brake pressure; the status every 100 ms),
its commands in (steering, brake, throttle), with the safety interlocks below. Standard library (numpy through the
chassis); runs in the HIL session worker (CARLA's Python) and in `falcon hil can bench`.

Buses (one interface: send(Frame), recv(timeout) → Received | None, close(); fd says CAN FD):
* SocketCAN — AF_CAN / CAN_RAW (stdlib), error frames received (CAN_RAW_ERR_FILTER), CAN FD frames only when the
  interface's MTU is 72 (CAN_RAW_FD_FRAMES), kernel receive timestamps (SO_TIMESTAMPNS, CLOCK_REALTIME, put on
  CLOCK_MONOTONIC by the offset between the two read at that moment): vcan, a USB-CAN adapter with a SocketCAN driver
  (gs_usb, peak_usb, kvaser_usb …), the Jetson's mttcan.
* sim:NAME — a simulated point-to-point link between two processes on this machine (abstract Unix datagram sockets;
  no file, no privileges, no network interface): the simulated device's CAN without vcan.
* python-can:INTERFACE:CHANNEL — through python-can (optional, the hil extra) for an adapter without a SocketCAN
  driver (slcan, pcan on Windows …); its timestamps are the adapter's, taken as realtime.
* fake_pair() — two buses on a socket pair, for tests.

Safety interlocks:
* the interface must match settings hil_can.interfaces_allowed (default vcan*); a real adapter (not under
  /sys/devices/virtual) also needs a confirmation that it is a bench network, not a vehicle (confirm_bench(), from
  Settings ▸ Hardware or the Hardware page), recorded with its time;
* command watchdog: no valid command for 100 ms → safe hold (no throttle, steering held, the brake at safe_brake_bar;
  a held stop once stationary), counted; control returns after RECOVER valid commands in a row. Before the first
  command, and while a command says ENABLE 0, the same hold. A command is valid when its CRC checks, its alive counter
  moved on by 1–3, its ranges hold;
* an error frame (bus-off, error-passive, no acknowledgement …) or ABORT_SEND_FAILURES failed sends in a row aborts
  the bridge: the chassis holds, the session fails with the reason;
* on exit the bridge stops sending before it closes: no periodic job is left in the kernel (raw sockets only, no BCM),
  the bus goes silent.

The "can" latency stage (core.hil.CAN_STAGES): the kernel's receive timestamp of the first frame carrying a step's
command → the moment the worker had it.
"""
from __future__ import annotations

import errno
import os
import socket
import threading
import time
from dataclasses import dataclass

from ..core import canmsgs as C
from . import device_agent as A

PERIOD_S = 0.010
STATUS_PERIOD_S = 0.100
WATCHDOG_S = 0.100
RECOVER = 3
ABORT_SEND_FAILURES = 50
CONFIRM_TEXT = 'bench network, not a vehicle'


class BusError(RuntimeError):
    """The bus is not healthy (an error frame, sends that keep failing): the bridge aborts."""


@dataclass
class Received:
    frame: C.Frame
    t_kernel_ns: int | None           # the kernel's receive time, on CLOCK_MONOTONIC (None: not available)
    t_read_ns: int                    # when recv() returned, CLOCK_MONOTONIC


def realtime_to_mono(rt_ns: int | None) -> int | None:
    if rt_ns is None:
        return None
    return int(rt_ns - (time.time_ns() - time.monotonic_ns()))


# ============================================================================================ buses
class RawBus:
    """A bus of Linux frame structs over a datagram socket (SocketCAN, a simulated link, a test's socket pair)."""

    def __init__(self, link, name: str):
        self.link, self.name = link, name
        self.fd = bool(link.fd)
        self.kind = link.kind

    def send(self, fr: C.Frame):
        if fr.fd and not self.fd:
            raise ValueError(f'a CAN FD frame on {self.name}, which is classical CAN')
        self.link.send(C.pack_frame(fr))

    def recv(self, timeout: float = 0.1) -> Received | None:
        got = self.link.recv(timeout)
        if got is None:
            return None
        raw, ts = got
        t_read = time.monotonic_ns()
        return Received(C.unpack_frame(raw), realtime_to_mono(ts), t_read)

    def close(self):
        self.link.close()


class _PairLink:
    kind = 'fake'

    def __init__(self, sock, fd):
        self.sock, self.fd, self.spec = sock, fd, 'fake'
        self.sock.setsockopt(socket.SOL_SOCKET, A.SO_TIMESTAMPNS, 1)

    send = A.CanLink.send
    recv = A.CanLink.recv
    close = A.CanLink.close
    peer = None


def fake_pair(fd: bool = False) -> tuple:
    """Two connected buses in this process (an AF_UNIX SOCK_SEQPACKET pair carrying the kernel's frame layout)."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    return RawBus(_PairLink(a, fd), 'fake-a'), RawBus(_PairLink(b, fd), 'fake-b')


class PythonCanBus:
    """An adapter through python-can (optional). Timestamps are the adapter's (python-can's float seconds, taken as
    CLOCK_REALTIME): state them as such."""
    kind = 'python-can'

    def __init__(self, interface: str, channel: str, fd: bool = False, bitrate: int = 500000,
                 data_bitrate: int = 2000000):
        try:
            import can
        except ImportError as e:
            raise RuntimeError('python-can is not installed (pip install python-can, the hil extra); a SocketCAN '
                               'adapter needs no python-can') from e
        self.can = can
        kw = {'interface': interface, 'channel': channel, 'receive_own_messages': False, 'bitrate': int(bitrate)}
        if fd:
            kw.update(fd=True, data_bitrate=int(data_bitrate))
        self.bus = can.Bus(**kw)
        self.fd = bool(fd)
        self.name = f'python-can:{interface}:{channel}'

    def send(self, fr: C.Frame):
        self.bus.send(self.can.Message(arbitration_id=fr.can_id, data=bytes(fr.data), is_extended_id=fr.extended,
                                       is_fd=fr.fd, bitrate_switch=fr.brs))

    def recv(self, timeout: float = 0.1) -> Received | None:
        m = self.bus.recv(timeout)
        if m is None:
            return None
        t_read = time.monotonic_ns()
        if m.is_error_frame:
            fr = C.Frame(m.arbitration_id, bytes(m.data or b''), error=True)
        else:
            fr = C.Frame(m.arbitration_id, bytes(m.data or b''), fd=bool(m.is_fd), brs=bool(m.bitrate_switch),
                         extended=bool(m.is_extended_id), rtr=bool(m.is_remote_frame))
        ts = int(m.timestamp * 1e9) if m.timestamp else None
        return Received(fr, realtime_to_mono(ts), t_read)

    def close(self):
        try:
            self.bus.shutdown()
        except Exception:                                # noqa: BLE001 (closing: nothing to do about it)
            pass


# ============================================================================================ the allow-list
def is_virtual(name: str) -> bool:
    return os.path.exists(f'/sys/devices/virtual/net/{name}')


def can_settings(settings) -> dict:
    d = dict((settings.get('hil_can') if settings is not None else None) or {})
    d.setdefault('interfaces_allowed', ['vcan*'])
    d.setdefault('confirmed', {})
    return d


def check_interface(spec: str, settings) -> tuple:
    """(allowed, why, kind): may the bridge open spec? sim: links always (nothing leaves this machine); a SocketCAN
    interface when it matches hil_can.interfaces_allowed and is virtual or was confirmed as a bench network; a
    python-can adapter only when confirmed."""
    cfg = can_settings(settings)
    if spec.startswith('sim:'):
        return True, 'a simulated link on this machine', 'sim'
    confirmed = cfg.get('confirmed') or {}
    if spec.startswith('python-can:'):
        if spec in confirmed:
            return True, f'confirmed as a {CONFIRM_TEXT} on {confirmed[spec].get("at")}', 'python-can'
        return False, (f'{spec} is a real adapter: confirm it is a {CONFIRM_TEXT} first (Settings ▸ Hardware, or '
                       '`falcon hil can confirm`)'), 'python-can'
    if not C.matches(spec, cfg['interfaces_allowed']):
        return False, (f'{spec} is not in the CAN interfaces Falcon may open ({", ".join(cfg["interfaces_allowed"])};'
                       ' Settings ▸ Hardware)'), 'socketcan'
    if is_virtual(spec):
        return True, 'a virtual interface', 'socketcan'
    if spec in confirmed:
        return True, f'confirmed as a {CONFIRM_TEXT} on {confirmed[spec].get("at")}', 'socketcan'
    return False, (f'{spec} is a real CAN adapter: confirm it is a {CONFIRM_TEXT} first (Settings ▸ Hardware, or '
                   '`falcon hil can confirm`)'), 'socketcan'


def confirm_bench(settings, spec: str, statement: str) -> dict:
    """Record that spec is a bench network, not a vehicle (statement must say so), and allow it. Returns the new
    hil_can settings."""
    if statement.strip().lower() != CONFIRM_TEXT:
        raise ValueError(f'the confirmation must read "{CONFIRM_TEXT}"')
    import re
    if not (re.match(r'^[A-Za-z0-9_.:-]{1,15}$', spec or '') and not spec.startswith('sim:')) and \
            not re.match(r'^python-can:[A-Za-z0-9_]+:[^\s]+$', spec or ''):
        raise ValueError(f'{spec!r} is not a CAN interface name (or python-can:INTERFACE:CHANNEL)')
    cfg = can_settings(settings)
    cfg['confirmed'] = dict(cfg.get('confirmed') or {})
    cfg['confirmed'][spec] = {'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'statement': CONFIRM_TEXT}
    if not spec.startswith('python-can:') and not C.matches(spec, cfg['interfaces_allowed']):
        cfg['interfaces_allowed'] = list(cfg['interfaces_allowed']) + [spec]
    settings.set('hil_can', cfg)
    return cfg


def revoke_bench(settings, spec: str) -> dict:
    cfg = can_settings(settings)
    cfg['confirmed'] = {k: v for k, v in (cfg.get('confirmed') or {}).items() if k != spec}
    cfg['interfaces_allowed'] = [p for p in cfg['interfaces_allowed'] if p != spec]
    settings.set('hil_can', cfg)
    return cfg


def open_bus(spec: str, settings, fd='auto', side: str = 'a'):
    """The bus spec names, after the allow-list (check_interface). fd: 'auto' (FD when the MTU is 72), 'on', 'off'."""
    ok, why, kind = check_interface(spec, settings)
    if not ok:
        raise PermissionError(why)
    if kind == 'python-can':
        _, iface, chan = spec.split(':', 2)
        return PythonCanBus(iface, chan, fd=fd in (True, 'on'))
    link = A.CanLink(spec, fd if spec.startswith('sim:') else ('auto' if fd == 'auto' else fd == 'on'), side=side)
    return RawBus(link, spec)


def interfaces() -> list:
    """The CAN interfaces of this machine (device_agent.can_interfaces: sysfs only)."""
    return A.can_interfaces()


# ============================================================================================ the bridge
class Bridge:
    """The chassis behind a bus (module doc). tick(now) and handle(received) do the work; start() runs them on two
    threads (100 Hz paced, and a reader), stop() ends both with the bus silent. A test may drive tick/handle itself
    with its own clock."""

    def __init__(self, bus, chassis, codec=None, period_s: float = PERIOD_S, watchdog_s: float = WATCHDOG_S,
                 status_period_s: float = STATUS_PERIOD_S, clock=time.monotonic_ns):
        self.bus, self.ch = bus, chassis
        self.codec = codec                                 # None: Falcon's table; a canmsgs.RoleMap: a user's DBC
        self.period_ns = int(period_s * 1e9)
        self.watchdog_ns = int(watchdog_s * 1e9)
        self.status_ns = int(status_period_s * 1e9)
        self.clock = clock
        self.lock = threading.Lock()
        self.counters = C.Counters()
        self.prev_counter: dict = {}
        self.parts: dict = {}                              # 'steer' | 'long' → (t, signals)
        self.mode = 'idle'
        self.good_run = 0
        self.last_tick = None
        self.next_status = 0
        self.stopping = threading.Event()
        self.aborted = None
        self.send_failures = 0
        self.threads: list = []
        self.arrivals: dict = {}                           # STEP & 0xFFFF → (t_kernel, t_read) of its first frame
        self.last_step = None
        self.on_command = None                             # callback(signals, received) after a valid command
        self.stats = {'tx_frames': 0, 'tx_bytes': 0, 'rx_frames': 0, 'commands': 0, 'rejected_crc': 0,
                      'rejected_counter': 0, 'rejected_range': 0, 'watchdog_trips': 0, 'error_frames': 0,
                      'send_failures': 0, 'saturated': 0, 'disabled': 0, 'foreign_frames': 0}
        self.errors: list = []
        self.saturated_signals: dict = {}
        self.trips: list = []                              # watchdog trips, seconds since the first tick
        self.t_first = None
        self.ch.safe_hold()

    # ------------------------------------------------------------------ in
    def handle(self, rx: Received):
        fr = rx.frame
        if fr.error:
            why = C.describe_error(fr)
            with self.lock:
                self.stats['error_frames'] += 1
                self.errors.append(why)
            self.abort('CAN error frame: ' + ', '.join(why))
            return
        with self.lock:
            self.stats['rx_frames'] += 1
        if self.codec is not None:
            sig = self.codec.decode_in(fr)
            if sig is None:
                with self.lock:
                    self.stats['foreign_frames'] += 1
                return
            self._accept(('steer', 'long'), sig, rx)
            return
        m = C.BY_ID.get(fr.can_id)
        if m is None or m.name not in C.COMMANDS:
            with self.lock:
                self.stats['foreign_frames'] += 1
            return
        v = C.decode(m, fr.data)
        with self.lock:
            if not v['_crc_ok']:
                self.stats['rejected_crc'] += 1
                return
            if not C.counter_ok(self.prev_counter.get(m.name), v['COUNTER']):
                self.prev_counter[m.name] = v['COUNTER']
                self.stats['rejected_counter'] += 1
                self.good_run = 0
                return
            self.prev_counter[m.name] = v['COUNTER']
        if v.get('STEER_MODE', 0) not in (0, 1, 2):
            with self.lock:
                self.stats['rejected_range'] += 1
            return
        parts = {'CMD_STEER': ('steer',), 'CMD_LONG': ('long',), 'CMD_FD': ('steer', 'long')}[m.name]
        self._accept(parts, v, rx)

    def _accept(self, parts, sig, rx: Received):
        now = self.clock()
        with self.lock:
            for p in parts:
                self.parts[p] = (now, sig)
            self.stats['commands'] += 1
            step = sig.get('STEP')
            if step is not None and 'long' in parts:
                key = int(step) & 0xFFFF                   # CMD_LONG's 16 bits (CMD_FD's 32 cut to them)
                if key != self.last_step:                  # the first frame of a step's command (it repeats)
                    self.last_step = key
                    self.arrivals[key] = (rx.t_kernel_ns, rx.t_read_ns)
                    if len(self.arrivals) > 4096:          # never taken: the oldest go
                        self.arrivals.pop(next(iter(self.arrivals)))
            ready = all(p in self.parts for p in ('steer', 'long'))
            enable = all(int(self.parts[p][1].get('ENABLE', 1)) == 1 for p in self.parts)
            if ready and 'long' in parts:              # one count per complete command (its CMD_LONG / CMD_FD)
                if enable:
                    self.good_run += 1
                else:
                    self.stats['disabled'] += 1
                    self.good_run = 0
        if self.on_command is not None:
            self.on_command(sig, rx)

    # ------------------------------------------------------------------ the 100 Hz tick
    def _command_age(self, now: int):
        if not all(p in self.parts for p in ('steer', 'long')):
            return None
        return now - min(self.parts['steer'][0], self.parts['long'][0])

    def tick(self, now: int | None = None):
        """Step the chassis to now, apply the newest valid command or the safe hold, and send the feedback."""
        now = self.clock() if now is None else now
        with self.lock:
            if self.t_first is None:
                self.t_first = now
            age = self._command_age(now)
            fresh = age is not None and age <= self.watchdog_ns
            enabled = fresh and all(int(self.parts[p][1].get('ENABLE', 1)) == 1 for p in ('steer', 'long'))
            if self.aborted:
                mode = 'aborted'
            elif self.mode == 'active' and not fresh:
                mode = 'safe_hold'
                self.stats['watchdog_trips'] += 1
                self.good_run = 0
                if len(self.trips) < 50:                   # when, for the record (seconds since the first tick)
                    self.trips.append(round((now - self.t_first) / 1e9, 3))
            elif self.mode in ('idle', 'safe_hold') and enabled and self.good_run >= RECOVER:
                mode = 'active'
            elif self.mode == 'active' and not enabled:
                mode = 'safe_hold'
            else:
                mode = self.mode
            self.mode = mode
            if mode == 'active':
                s, lg = self.parts['steer'][1], self.parts['long'][1]
                sm = {0: 'off', 1: 'angle', 2: 'torque'}.get(int(s.get('STEER_MODE', 1)), 'off')
                self.ch.command(sm, float(s.get('STEER_REQ', 0.0)), float(s.get('STEER_TORQUE_REQ', 0.0)),
                                float(lg.get('BRAKE_REQ', 0.0)), float(lg.get('THROTTLE_REQ', 0.0)))
            elif not self.ch.cmd.get('safe'):
                self.ch.safe_hold()
            dt = 0.0 if self.last_tick is None else (now - self.last_tick) / 1e9
            self.last_tick = now
            if dt > 0:
                self.ch.step(min(dt, 0.1))
            fb = self.ch.feedback()
            fb['MODE'] = C.MODES[mode]
            fb['WATCHDOG_TRIPS'] = self.stats['watchdog_trips']
            fb['CMD_REJECTED'] = self.stats['rejected_crc'] + self.stats['rejected_counter'] + \
                self.stats['rejected_range']
            fb['EPS_STATUS'] = 0 if mode == 'active' else 1
            fb['BRAKE_STATUS'] = 2 if mode != 'active' else int(fb['BRAKE_PRESSURE'] > 0.5)
            frames = self._feedback_frames(fb, now)
        if mode == 'aborted':
            return
        for fr in frames:
            self._send(fr)

    def _feedback_frames(self, fb: dict, now: int) -> list:
        if self.codec is not None:
            return self.codec.encode_out(fb, self.counters)
        out = []
        if self.bus.fd:
            names = ['CHASSIS_FD']
        else:
            names = list(C.FEEDBACK)
        if now >= self.next_status:
            names.append(C.STATUS)
            self.next_status = now + self.status_ns
        for n in names:
            m = C.BY_NAME[n]
            vals = dict(fb)
            for s in m.signals:
                if s.name.startswith('TICKS_'):
                    vals[s.name] = int(vals[s.name]) % (1 << s.length)      # the counters wrap
            data, sat = C.encode(m, vals, self.counters.next(n))
            if sat:
                self.stats['saturated'] += 1
                for s in sat:                              # which signals hit their range, for the record
                    self.saturated_signals[s] = self.saturated_signals.get(s, 0) + 1
            out.append(C.Frame(m.can_id, data, fd=m.fd, brs=m.fd))
        return out

    def _send(self, fr: C.Frame):
        try:
            self.bus.send(fr)
            self.send_failures = 0
            with self.lock:
                self.stats['tx_frames'] += 1
                self.stats['tx_bytes'] += len(fr.data)
        except OSError as e:
            self.send_failures += 1
            with self.lock:
                self.stats['send_failures'] += 1
            if e.errno not in (errno.ENOBUFS, errno.EAGAIN) or self.send_failures >= ABORT_SEND_FAILURES:
                self.abort(f'sending failed {self.send_failures} times in a row ({e.strerror or e}): is another node '
                           'on the bus acknowledging, and is the bit rate right?')

    def abort(self, why: str):
        with self.lock:
            if self.aborted is None:
                self.aborted = why
            self.mode = 'aborted'
            self.ch.safe_hold()
        self.stopping.set()

    # ------------------------------------------------------------------ threads
    def start(self):
        for fn, name in ((self._reader, 'falcon-can-rx'), (self._ticker, 'falcon-can-tx')):
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            self.threads.append(t)
        return self

    def _reader(self):
        while not self.stopping.is_set():
            try:
                rx = self.bus.recv(0.05)
            except OSError as e:
                if not self.stopping.is_set():
                    self.abort(f'reading the bus failed: {e}')
                break
            except ValueError:
                continue
            if rx is not None:
                self.handle(rx)

    def _ticker(self):
        nxt = time.monotonic_ns()
        while not self.stopping.is_set():
            self.tick()
            nxt += self.period_ns
            delay = (nxt - time.monotonic_ns()) / 1e9
            if delay > 0:
                self.stopping.wait(delay)
            else:
                nxt = time.monotonic_ns()                    # late: no burst to catch up

    def stop(self):
        """Stop sending, then close: the bus is left silent."""
        self.stopping.set()
        for t in self.threads:
            t.join(2.0)
        self.bus.close()

    def snapshot(self) -> dict:
        with self.lock:
            return {'mode': self.mode, 'aborted': self.aborted, **self.stats, 'fd': self.bus.fd}

    def bus_load(self, seconds: float, bitrate: int = 500000) -> float | None:
        """The share of a classical bus's bit rate the bridge's frames took (nominal frame bits, no stuffing)."""
        if seconds <= 0 or self.bus.fd:
            return None
        bits = self.stats['tx_frames'] * 47 + self.stats['tx_bytes'] * 8
        return bits / seconds / bitrate


# ============================================================================================ in a HIL session
class SessionCan:
    """The CAN loop of a HIL session (workers.hil): the chassis model with CARLA as the plant behind the bus; the
    device's commands drive the actuators, whose output the worker applies to the ego, and the ego's motion drives
    the wheels whose signals go back. cfg: core.hil's 'can' section."""

    def __init__(self, cfg: dict, settings):
        from ..core.chassis import Chassis
        self.cfg = cfg
        self.bus = open_bus(cfg['interface'], settings, cfg.get('fd', 'auto'), side='a')
        codec = None
        if cfg.get('dbc') and cfg.get('dbc_map'):
            msgs, _notes, _reader = C.load_dbc(cfg['dbc'])
            codec = C.RoleMap(msgs, cfg['dbc_map'])
        self.chassis = Chassis(cfg.get('chassis') or {}, plant='carla')
        self.bridge = Bridge(self.bus, self.chassis, codec)
        self.t0 = time.monotonic()
        self.bridge.start()

    def set_motion(self, speed, yaw_rate_deg_s, ax, ay):
        with self.bridge.lock:
            self.chassis.set_motion(speed, yaw_rate_deg_s, ax, ay)

    def actuation(self) -> dict:
        with self.bridge.lock:
            return dict(self.chassis.actuation(), mode=self.bridge.mode)

    def take_arrival(self, frame: int):
        """(kernel time, read time) of the first command frame for simulation frame `frame` (taken: asked once), or
        None. The kernel time is None when the bus gives none."""
        with self.bridge.lock:
            got = self.bridge.arrivals.pop(int(frame) & 0xFFFF, None)
        if got is None:
            return None
        return got[0], got[1]

    def check(self):
        if self.bridge.aborted:
            raise BusError(self.bridge.aborted)

    def finish(self) -> dict:
        self.bridge.stop()
        wall = time.monotonic() - self.t0
        snap = self.bridge.snapshot()
        return {'interface': self.cfg['interface'], 'kind': self.bus.kind, 'fd': self.bus.fd, 'stats': snap,
                'errors': self.bridge.errors[:20], 'aborted': self.bridge.aborted,
                'saturated_signals': dict(self.bridge.saturated_signals), 'watchdog_trip_s': list(self.bridge.trips),
                'seconds': round(wall, 3),
                'bus_load': self.bridge.bus_load(wall, int(self.cfg.get('bitrate', 500000))),
                'chassis': self.chassis.describe(), 'watchdog_ms': WATCHDOG_S * 1000, 'period_ms': PERIOD_S * 1000,
                'timestamps': 'kernel (SO_TIMESTAMPNS)' if self.bus.kind in ('socketcan', 'sim') else
                'adapter (python-can)'}


# ============================================================================================ the bench
def bench(spec: str, settings, seconds: float = 10.0, fd='auto', chassis: dict | None = None, progress=None) -> dict:
    """The chassis model alone on a bus (plant 'model'), for a device on the bench: `falcon hil can bench`. Runs
    seconds (0: until KeyboardInterrupt), returns the counters."""
    from ..core.chassis import Chassis
    bus = open_bus(spec, settings, fd, side='a')
    ch = Chassis(chassis or {}, plant='model')
    br = Bridge(bus, ch).start()
    t0 = time.monotonic()
    try:
        while not br.stopping.is_set() and (seconds <= 0 or time.monotonic() - t0 < seconds):
            time.sleep(0.5)
            if progress is not None:
                progress(br.snapshot(), ch.feedback())
    except KeyboardInterrupt:
        pass
    finally:
        br.stop()
    wall = time.monotonic() - t0
    return {'interface': spec, 'seconds': round(wall, 2), **br.snapshot(), 'bus_load': br.bus_load(wall),
            'errors': br.errors[:20], 'saturated_signals': dict(br.saturated_signals),
            'chassis': {k: v for k, v in ch.describe().items() if k != 'params'}}


# ============================================================================================ doctor
def doctor_rows(settings=None) -> list:
    """(name, ok, detail) for `falcon doctor` (informational: CAN is optional)."""
    rows = []
    try:
        with open('/proc/modules', encoding='ascii') as fh:
            mods = fh.read()
    except OSError:
        mods = ''
    for mod in ('can_raw', 'vcan'):
        ok = f'\n{mod} ' in '\n' + mods
        rows.append((f'{mod} module', ok, 'loaded' if ok else 'not loaded (falcon hil setup prints how)'))
    ifs = interfaces()
    cfg = can_settings(settings)
    for i in ifs:
        allowed, why, _ = check_interface(i['name'], settings)
        rows.append((f'CAN {i["name"]}', i['up'],
                     f'{"virtual" if i["virtual"] else "adapter"}, {"CAN FD" if i["fd"] else "classical"}, '
                     f'{"up" if i["up"] else "down"}; {"allowed" if allowed else "not allowed: " + why}'))
    if not ifs:
        rows.append(('CAN interfaces', False, f'none (allowed: {", ".join(cfg["interfaces_allowed"])})'))
    for mod, what in (('can', 'python-can (optional, non-SocketCAN adapters)'), ('cantools', 'cantools (optional, '
                                                                                             'a user DBC)')):
        import importlib.util
        ok = importlib.util.find_spec(mod) is not None
        rows.append((what.split(' ')[0], ok, 'installed' if ok else f'not installed: {what}'))
    return rows
