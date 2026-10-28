#!/usr/bin/env python3
"""Falcon Autonomy's device agent for hardware-in-the-loop sessions — one file, copied to the device and run there.

    python3 device_agent.py [--listen 0.0.0.0] [--port 7710] [--once] [--stack module:Class] [--decoder auto|NAME]
    python3 device_agent.py --check            # what this device has (JSON on stdout), then exit

Needs Python ≥ 3.10, the standard library and numpy; for an H.264 camera also GStreamer and its Python bindings
(python3-gi, gir1.2-gstreamer-1.0) — all part of JetPack 6 (L4T r36, Ubuntu 22.04). It imports nothing from Falcon:
the host's gateway (falcon_autonomy.hil.gateway) imports this module for the wire format, so both ends share one
definition. Licence: GPL-3.0-or-later, as Falcon Autonomy.

What it does: it listens on TCP and UDP port P. A session (one at a time) is a TCP connection from the host's
gateway that starts with "hello"; the agent sets up its receivers, replies "ready" with what it runs on, receives the
camera and the lidar of every simulation step, runs the stack on each (camera, lidar) pair and sends the control back
on the session connection with its own timestamps. It executes nothing it receives: the stack is chosen on this
command line, never by the host. Run it on a trusted network only (the streams are neither encrypted nor
authenticated).

Wire format (docs/hardware.md):

* Framed messages (the session connection, TCP streams, reassembled UDP messages): >II (header length, payload
  length, big-endian uint32), a UTF-8 JSON header with a type "t", then the payload bytes — the same framing as
  Falcon's live link.
* Session, host → device: {"t": "hello", "protocol": 1, "session": {...}}, then {"t": "bye"}.
  Device → host: {"t": "ready", "env": {...}, "decoder": ...}, {"t": "control", "frame", "throttle", "steer",
  "brake", "dev": {stamps}, "missing": [...], "info": {...}} per stack run, {"t": "stats", ...} after "bye".
* Streams over TCP: a connection whose first message is {"t": "stream", "kind": "camera" | "lidar"}, then one message
  per step: {"t": "camera", "frame", "sim_t", "hw_t", "w", "h", "format": "bgr8"} + the pixels, or {"t": "lidar",
  "frame", "sim_t", "hw_t", "n", "encoding": "xyzi" | "pointcloud2", "speed_mps"} + float32 x y z intensity per
  point (the sensor's frame: x forward, y right, z up) or a sensor_msgs/PointCloud2 in CDR.
* Camera over RTP: H.264 (RFC 6184, payload type 96, SPS/PPS with every IDR) to UDP port camera.rtp_port. Each access
  unit carries a user-data-unregistered SEI message (UUID SEI_UUID) holding {"f": frame, "s": sim_t, "h": hw_t} as
  JSON, so a decoded frame is matched to its step exactly; decoders ignore it.
* Lidar over UDP: a framed message cut into datagrams of at most CHUNK_BYTES: CHUNK header (magic "FLDR", message id
  = the frame, chunk index, chunk count, total length) then the bytes. A message is used only when every chunk
  arrived; an incomplete one is dropped when a newer one completes.
* Clock pings over UDP to port P: PING (magic "FPNG", sequence, t0) answered at once with PONG ("FPON", sequence, t0,
  t1 = receive time, t2 = send time, both on this device's clock) — the host estimates the offset NTP-style.

Timestamps are int64 nanoseconds of CLOCK_MONOTONIC (time.monotonic_ns), optionally passed through an emulated clock
(--clock-offset-ms, --clock-drift-ppm) so that the host's offset estimate can be checked against a known truth.

The stack: a class with __init__(self, params: dict, session: dict) and act(self, camera, lidar, meta) -> dict with
throttle, steer, brake (0..1, −1..1, 0..1). camera is an H × W × 3 or × 4 uint8 array (BGR or BGRx) or None; lidar
an N × 4 float32 array (x y z intensity, sensor frame) or None; meta the step's {frame, sim_t, speed_mps, hw_t…}.
The default, BrakeForObstacles, drives at a target speed and brakes for anything in its lidar corridor ahead — a
trivial controller that closes the loop honestly: its control depends on what the device received.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import socket
import struct
import sys
import threading
import time

PROTOCOL = 1
AGENT_VERSION = '1.0'
DEFAULT_PORT = 7710
HEAD = struct.Struct('>II')
MAX_HEADER = 1 << 20
MAX_PAYLOAD = 256 << 20
PING = struct.Struct('>4sIq')
PONG = struct.Struct('>4sIqqq')
CHUNK = struct.Struct('>4sIHHI')
CHUNK_BYTES = 1400                       # per datagram, header included (fits a 1500-byte Ethernet MTU)
SEI_UUID = b'FalconAutonomy/1'           # 16 bytes, no zero byte
RTP_CAPS = ('application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,'
            'payload=(int)96')
DECODERS = ('nvv4l2decoder', 'avdec_h264', 'nvh264dec', 'openh264dec')   # in order of preference (auto)
ELEMENTS = ('udpsrc', 'rtph264depay', 'h264parse', 'appsrc', 'appsink', 'videoconvert', 'nvvidconv') + DECODERS


def log(text: str):
    print(f'{time.strftime("%H:%M:%S")} agent: {text}', file=sys.stderr, flush=True)


# ============================================================================================ framing
def pack(header: dict, payload=b'') -> bytes:
    h = json.dumps(header, separators=(',', ':')).encode('utf-8')
    return HEAD.pack(len(h), len(payload)) + h + bytes(payload)


def unpack(data) -> tuple:
    """One framed message held whole in data: (header, payload)."""
    hl, pl = HEAD.unpack_from(data, 0)
    if HEAD.size + hl + pl != len(data):
        raise ValueError(f'framed message of {len(data)} bytes says {hl} + {pl}')
    return json.loads(bytes(data[HEAD.size:HEAD.size + hl]).decode('utf-8')), memoryview(data)[HEAD.size + hl:]


def recv_exact(sock, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError('the connection closed')
        got += k
    return buf


def read_msg(sock) -> tuple:
    """The next framed message from a stream socket: (header, payload); ConnectionError when it closes."""
    hl, pl = HEAD.unpack(recv_exact(sock, HEAD.size))
    if hl > MAX_HEADER or pl > MAX_PAYLOAD:
        raise ValueError(f'message too large ({hl} + {pl} bytes)')
    header = json.loads(recv_exact(sock, hl).decode('utf-8'))
    return header, (recv_exact(sock, pl) if pl else bytearray())


def chunks(msg_id: int, data: bytes, size: int = CHUNK_BYTES) -> list:
    """A framed message as UDP datagrams (CHUNK header + bytes)."""
    body = size - CHUNK.size
    n = max((len(data) + body - 1) // body, 1)
    if n > 65535:
        raise ValueError(f'a message of {len(data)} bytes needs more than 65535 datagrams')
    return [CHUNK.pack(b'FLDR', msg_id & 0xFFFFFFFF, i, n, len(data)) + data[i * body:(i + 1) * body]
            for i in range(n)]


class Reassembler:
    """Datagrams back into messages. feed() returns the message's bytes when its last chunk arrives; a message still
    incomplete when a newer one completes is dropped and counted."""

    def __init__(self):
        self.parts: dict = {}
        self.incomplete = 0
        self.bad = 0

    def feed(self, dgram: bytes):
        if len(dgram) < CHUNK.size:
            self.bad += 1
            return None
        magic, mid, i, n, total = CHUNK.unpack_from(dgram, 0)
        if magic != b'FLDR' or i >= n:
            self.bad += 1
            return None
        p = self.parts.setdefault(mid, {'n': n, 'total': total, 'got': {}})
        p['got'][i] = dgram[CHUNK.size:]
        if len(p['got']) < p['n']:
            return None
        del self.parts[mid]
        for old in [k for k in self.parts if k < mid]:
            del self.parts[old]
            self.incomplete += 1
        data = b''.join(p['got'][k] for k in range(p['n']))
        if len(data) != p['total']:
            self.bad += 1
            return None
        return data


# ============================================================================================ H.264 SEI
def _escape(b: bytes) -> bytes:
    """Emulation prevention: a 0x03 after two zero bytes whenever the next byte is 0–3 (H.264 7.4.1)."""
    out = bytearray()
    zeros = 0
    for x in b:
        if zeros >= 2 and x <= 3:
            out.append(3)
            zeros = 0
        out.append(x)
        zeros = zeros + 1 if x == 0 else 0
    return bytes(out)


def sei_nal(meta: dict) -> bytes:
    """A user-data-unregistered SEI NAL unit (payload type 5) carrying meta as JSON after SEI_UUID, with its start
    code."""
    data = SEI_UUID + json.dumps(meta, separators=(',', ':')).encode('utf-8')
    body = bytearray([5])
    size = len(data)
    while size >= 255:
        body.append(255)
        size -= 255
    body.append(size)
    body += data
    body.append(0x80)                                    # rbsp_trailing_bits
    return b'\x00\x00\x00\x01\x06' + _escape(bytes(body))


def nal_units(au: bytes):
    """(start of the start code, start of the NAL, end, nal_unit_type) of each NAL unit of an Annex B access unit."""
    i = au.find(b'\x00\x00\x01')
    while i >= 0:
        sc = i - 1 if i > 0 and au[i - 1] == 0 else i
        s = i + 3
        j = au.find(b'\x00\x00\x01', s)
        e = len(au) if j < 0 else j
        while j >= 0 and e > s and au[e - 1] == 0:          # the next start code's leading zero, trailing zeros
            e -= 1
        if s < len(au):
            yield sc, s, e, au[s] & 0x1F
        i = j


def insert_sei(au: bytes, meta: dict) -> bytes:
    """The access unit with an SEI carrying meta before its first slice (after any AUD, SPS and PPS)."""
    for sc, _s, _e, t in nal_units(au):
        if 1 <= t <= 5:
            return au[:sc] + sei_nal(meta) + au[sc:]
    return au + sei_nal(meta)


def read_sei(au: bytes):
    """The meta of Falcon's SEI in an access unit (before its first slice), or None."""
    for _sc, s, e, t in nal_units(au):
        if 1 <= t <= 5:
            return None
        if t != 6:
            continue
        p = au[s + 1:e].replace(b'\x00\x00\x03', b'\x00\x00')
        i = 0
        while len(p) - i > 1:
            pt = 0
            while i < len(p) and p[i] == 255:
                pt += 255
                i += 1
            if i >= len(p):
                break
            pt += p[i]
            i += 1
            sz = 0
            while i < len(p) and p[i] == 255:
                sz += 255
                i += 1
            if i >= len(p):
                break
            sz += p[i]
            i += 1
            body = p[i:i + sz]
            i += sz
            if pt == 5 and body[:16] == SEI_UUID:
                try:
                    return json.loads(body[16:].decode('utf-8'))
                except ValueError:
                    return None
    return None


# ============================================================================================ PointCloud2 (CDR)
class _Cdr:
    """A reader of plain little-endian CDR (XCDR1): primitives aligned to their size after the 4-byte header."""

    def __init__(self, b, pos=4):
        self.b, self.pos = b, pos

    def _align(self, n):
        self.pos = 4 + ((self.pos - 4 + n - 1) // n) * n

    def _take(self, fmt, n):
        self._align(n)
        v = struct.unpack_from('<' + fmt, self.b, self.pos)[0]
        self.pos += n
        return v

    def u8(self):
        v = self.b[self.pos]
        self.pos += 1
        return v

    def u32(self):
        return self._take('I', 4)

    def i32(self):
        return self._take('i', 4)

    def string(self):
        n = self.u32()
        s = bytes(self.b[self.pos:self.pos + n - 1]).decode('utf-8')
        self.pos += n
        return s

    def raw(self, n):
        v = self.b[self.pos:self.pos + n]
        self.pos += n
        return v


_PF_DTYPE = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}


def decode_pointcloud2(cdr):
    """sensor_msgs/PointCloud2 in CDR (as Falcon's gateway encodes it with core.mcap_bag) → (N × 4 float32 x y z
    intensity, {stamp_ns, frame_id, width, fields})."""
    import numpy as np
    if bytes(cdr[:2]) != b'\x00\x01':
        raise ValueError('only little-endian plain CDR is read')
    r = _Cdr(cdr)
    sec, nsec = r.i32(), r.u32()
    frame_id = r.string()
    height, width = r.u32(), r.u32()
    fields = []
    for _ in range(r.u32()):
        name = r.string()
        off = r.u32()
        dt = r.u8()
        cnt = r.u32()
        fields.append((name, off, dt, cnt))
    bigend = r.u8()
    step = r.u32()
    r.u32()                                              # row_step
    data = r.raw(r.u32())
    if bigend:
        raise ValueError('big-endian point clouds are not read')
    dt = np.dtype({'names': [f[0] for f in fields], 'formats': [_PF_DTYPE[f[2]] for f in fields],
                   'offsets': [f[1] for f in fields], 'itemsize': step})
    a = np.frombuffer(data, dt, count=height * width)
    out = np.zeros((len(a), 4), np.float32)
    for k, name in enumerate(('x', 'y', 'z', 'intensity')):
        if name in a.dtype.names:
            out[:, k] = a[name]
    return out, {'stamp_ns': sec * 1_000_000_000 + nsec, 'frame_id': frame_id, 'width': width,
                 'fields': [f[0] for f in fields]}


# ============================================================================================ the clock
class DeviceClock:
    """This device's clock for every stamp it sends: CLOCK_MONOTONIC, optionally through an emulated offset and drift
    (simulated devices: lets the host's offset estimate be checked against a known truth)."""

    def __init__(self, offset_ms: float = 0.0, drift_ppm: float = 0.0):
        self.offset_ns = int(round(float(offset_ms) * 1e6))
        self.drift_ppm = float(drift_ppm)

    @property
    def emulated(self) -> bool:
        return bool(self.offset_ns or self.drift_ppm)

    def now(self) -> int:
        t = time.monotonic_ns()
        if self.drift_ppm:
            t += int(t * self.drift_ppm * 1e-6)
        return t + self.offset_ns


# ============================================================================================ what this device has
def _read(path, n=200) -> str:
    try:
        with open(path, 'rb') as fh:
            return fh.read(n).replace(b'\x00', b'').decode('utf-8', 'replace').strip()
    except OSError:
        return ''


def gst():
    """GStreamer's module (initialised), or None when PyGObject or GStreamer is missing."""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
    except (ImportError, ValueError):
        return None
    if not Gst.is_initialized():
        Gst.init(None)
    return Gst


def pick_decoder(prefer: str = 'auto') -> str:
    G = gst()
    if G is None:
        return ''
    if prefer and prefer != 'auto':
        return prefer if G.ElementFactory.find(prefer) is not None else ''
    return next((d for d in DECODERS if G.ElementFactory.find(d) is not None), '')


def environment(decoder: str = 'auto') -> dict:
    """What this device runs on (no host name, no addresses)."""
    env = {'agent_version': AGENT_VERSION, 'protocol': PROTOCOL, 'python': platform.python_version(),
           'platform': platform.platform(terse=True), 'machine': platform.machine(), 'cpus': os.cpu_count(),
           'model': _read('/proc/device-tree/model'), 'l4t': _read('/etc/nv_tegra_release').split(',')[0],
           'clock': 'CLOCK_MONOTONIC'}
    try:
        import numpy
        env['numpy'] = numpy.__version__
    except ImportError:
        env['numpy'] = None
    G = gst()
    env['gstreamer'] = G.version_string() if G is not None else None
    env['elements'] = {e: G.ElementFactory.find(e) is not None for e in ELEMENTS} if G is not None else {}
    env['decoder'] = pick_decoder(decoder)
    env['rclpy'] = importlib.util.find_spec('rclpy') is not None
    return env


# ============================================================================================ the default stack
class BrakeForObstacles:
    """Drive at a target speed, brake for anything in the lidar's corridor ahead (the vehicle frame: x forward, the
    lidar's mount from the session). The lidar's latest full rotation is kept (a step may bring a slice of one).
    params: target_speed_mps (6), corridor_half_width_m (1.4), min_height_m (0.3 above the ground), max_height_m
    (2.5), front_m (2.6: nearer points are the ego's own body), range_m (40), decel_mps2 (4: planned braking),
    margin_m (6: stopping gap), kp (0.4). With no lidar for 0.5 s it brakes."""

    def __init__(self, params: dict, session: dict):
        p = {'target_speed_mps': 6.0, 'corridor_half_width_m': 1.4, 'min_height_m': 0.3, 'max_height_m': 2.5,
             'front_m': 2.6, 'range_m': 40.0, 'decel_mps2': 4.0, 'margin_m': 6.0, 'kp': 0.4}
        p.update({k: float(v) for k, v in (params or {}).items()})
        self.p = p
        lid = session.get('lidar') or {}
        pose = lid.get('pose') or {}
        self.mount = (float(pose.get('x', 0.0)), float(pose.get('y', 0.0)), float(pose.get('z', 2.4)))
        self.rotation_s = 1.0 / max(float(lid.get('rotation_frequency', 10.0)), 1e-3)
        self.slices: list = []                           # (sim_t, points in the vehicle frame)
        self.last_lidar = None

    def act(self, camera, lidar, meta: dict) -> dict:
        import numpy as np
        p = self.p
        st = float(meta.get('sim_t', 0.0))
        v = float(meta.get('speed_mps', 0.0))
        info = {'speed_mps': round(v, 3)}
        if camera is not None:
            info['camera_mean'] = round(float(camera[::16, ::16, :3].mean()), 2)
        if lidar is not None:
            P = np.asarray(lidar, np.float32)[:, :3] + np.asarray(self.mount, np.float32)
            keep = (P[:, 0] > p['front_m']) & (P[:, 0] < p['range_m']) & \
                (np.abs(P[:, 1]) < p['corridor_half_width_m']) & (P[:, 2] > p['min_height_m']) & \
                (P[:, 2] < p['max_height_m'])
            self.slices.append((st, P[keep]))
            self.slices = [s for s in self.slices if s[0] > st - self.rotation_s + 1e-6]
            self.last_lidar = st
        if self.last_lidar is None or st - self.last_lidar > 0.5:
            info['reason'] = 'no lidar'
            return {'throttle': 0.0, 'steer': 0.0, 'brake': 1.0, 'info': info}
        near = [float(s[1][:, 0].min()) for s in self.slices if len(s[1])]
        d = min(near) if near else None
        info['obstacle_m'] = None if d is None else round(d - p['front_m'], 2)
        gap = None if d is None else d - p['front_m']
        if gap is not None and gap < p['margin_m']:
            info['reason'] = 'obstacle within the margin'
            return {'throttle': 0.0, 'steer': 0.0, 'brake': 1.0, 'info': info}
        if gap is not None:
            need = v * v / (2.0 * max(gap - p['margin_m'], 0.1))       # deceleration needed to stop at the margin
            if need > 0.5 * p['decel_mps2']:
                info['reason'] = 'braking for an obstacle'
                return {'throttle': 0.0, 'steer': 0.0, 'brake': min(max(need / 8.0, 0.2), 1.0), 'info': info}
        err = p['target_speed_mps'] - v
        info['reason'] = 'cruise'
        return {'throttle': min(max(p['kp'] * err, 0.0), 0.6), 'steer': 0.0,
                'brake': min(max(-p['kp'] * (err + 0.5), 0.0), 1.0), 'info': info}


def load_stack(spec: str, params: dict, session: dict):
    if not spec or spec == 'brake_for_obstacles':
        return BrakeForObstacles(params, session)
    mod, _, cls = spec.partition(':')
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return getattr(importlib.import_module(mod), cls or 'Stack')(params, session)


# ============================================================================================ a session
class H264Receiver:
    """RTP/H.264 in: udpsrc → rtph264depay → access units (their SEI read, receive time stamped) → appsrc → h264parse →
    decoder → BGRx frames (decode end stamped). No jitter buffer (it would add its latency); a lost packet loses its
    frame, which the stack then runs without."""

    def __init__(self, ses, port: int, decoder: str):
        G = gst()
        if G is None:
            raise RuntimeError('the camera is H.264 but PyGObject/GStreamer is not available here '
                               '(sudo apt install python3-gi gir1.2-gstreamer-1.0)')
        self.G, self.ses = G, ses
        self.decoder = pick_decoder(decoder)
        if not self.decoder:
            raise RuntimeError(f'no H.264 decoder found (tried {", ".join(DECODERS)})')
        conv = 'nvvidconv' if self.decoder == 'nvv4l2decoder' else 'videoconvert'
        self.rx = G.parse_launch(
            f'udpsrc name=src port={int(port)} buffer-size=4194304 caps="{RTP_CAPS}" ! rtph264depay ! '
            'video/x-h264,stream-format=byte-stream,alignment=au ! appsink name=sink sync=false max-buffers=32 '
            'drop=false emit-signals=false')
        self.dec = G.parse_launch(
            'appsrc name=src is-live=true format=time do-timestamp=false '
            'caps=video/x-h264,stream-format=byte-stream,alignment=au ! h264parse ! '
            f'{self.decoder} name=dec ! {conv} ! video/x-raw,format=BGRx ! appsink name=sink sync=false '
            'max-buffers=8 drop=false emit-signals=false')
        dec = self.dec.get_by_name('dec')
        for prop, val in (('thread-type', 'slice'), ('enable-max-performance', 'true'), ('disable-dpb', 'true')):
            if dec.find_property(prop) is not None:
                G.util_set_object_arg(dec, prop, val)
        self.src = self.dec.get_by_name('src')
        self.inflight: dict = {}                         # pts → (frame, meta, rx, decode start, bytes)
        self.lock = threading.Lock()
        self.delta = None
        self.period = int(ses.period_ns)
        self.threads = [threading.Thread(target=self._receive, daemon=True, name='agent-rtp'),
                        threading.Thread(target=self._decoded, daemon=True, name='agent-decode')]

    def start(self):
        for p in (self.dec, self.rx):
            if p.set_state(self.G.State.PLAYING) == self.G.StateChangeReturn.FAILURE:
                raise RuntimeError('the H.264 receiver did not start')
        for t in self.threads:
            t.start()

    def _errors(self, pipe):
        msg = pipe.get_bus().timed_pop_filtered(0, self.G.MessageType.ERROR)
        if msg is not None:
            err, dbg = msg.parse_error()
            self.ses.note(f'GStreamer: {err.message} ({dbg})')

    def _receive(self):
        sink = self.rx.get_by_name('sink')
        G, ses = self.G, self.ses
        while not ses.stopping.is_set():
            s = sink.emit('try-pull-sample', 100_000_000)
            if s is None:
                self._errors(self.rx)
                continue
            t_rx = ses.clock.now()
            buf = s.get_buffer()
            au = buf.extract_dup(0, buf.get_size())
            meta = read_sei(au)
            if meta is None or 'f' not in meta:
                ses.count('camera_without_sei')
                continue
            frame = int(meta['f'])
            pts = frame * self.period
            out = G.Buffer.new_wrapped(au)
            out.pts = pts
            out.duration = self.period
            with self.lock:
                self.inflight[pts] = (frame, meta, t_rx, ses.clock.now(), len(au))
            ses.arrived(frame, 'cam_rx', t_rx, len(au))
            self.src.emit('push-buffer', out)

    def _decoded(self):
        import numpy as np
        sink = self.dec.get_by_name('sink')
        ses = self.ses
        while not ses.stopping.is_set():
            s = sink.emit('try-pull-sample', 100_000_000)
            if s is None:
                self._errors(self.dec)
                continue
            t_dec = ses.clock.now()
            buf = s.get_buffer()
            st = s.get_caps().get_structure(0)
            w, h = st.get_value('width'), st.get_value('height')
            with self.lock:
                if self.delta is None and self.inflight:     # a decoder may shift timestamps: learnt once
                    self.delta = buf.pts - min(self.inflight)
                key = buf.pts - (self.delta or 0)
                item = self.inflight.pop(key, None)
                for k in [k for k in self.inflight if k < key]:
                    self.inflight.pop(k)                 # older units the decoder dropped (a lost reference)
                    ses.count('camera_not_decoded')
            if item is None:
                continue
            frame, meta, t_rx, t_dec0, n = item
            img = np.frombuffer(buf.extract_dup(0, buf.get_size()), np.uint8)[:h * w * 4].reshape(h, w, 4)
            ses.arrived(frame, 'dec_end', t_dec)
            ses.offer(frame, 'camera', img, {'sim_t': meta.get('s'), 'hw_t': meta.get('h')},
                      {'cam_rx': t_rx, 'dec_start': t_dec0, 'dec_end': t_dec})

    def stop(self):
        for p in (self.rx, self.dec):
            p.set_state(self.G.State.NULL)
        for t in self.threads:
            t.join(2.0)


class Session:
    """One session: the receivers, the frame sets waiting for the stack, the stack's thread, the counters."""

    def __init__(self, agent, conn, hello: dict):
        self.agent, self.conn = agent, conn
        self.cfg = hello.get('session') or {}
        self.clock = agent.clock
        self.wlock = threading.Lock()
        self.cv = threading.Condition()
        self.pending: dict = {}                          # frame → {'camera': ..., 'lidar': ..., 'first': t}
        self.last_run = -1
        self.stopping = threading.Event()
        self.counts: dict = {}
        self.notes: list = []
        self.arrivals: dict = {}                         # frame → {cam_rx, dec_end, lidar_rx, cam_bytes, lidar_bytes}
        self.period_ns = int(self.cfg.get('period_ns', 50_000_000))
        self.sync_timeout_ns = int(float(self.cfg.get('sync_timeout_ms', self.period_ns / 1e6)) * 1e6)
        self.video = None
        self.udp = None
        self.ros = None
        self.streams: list = []
        st = self.cfg.get('stack') or {}
        self.stack = load_stack(agent.stack_spec, dict(st.get('params') or {}, **agent.stack_params), self.cfg)
        self.stack_name = agent.stack_spec or 'brake_for_obstacles'

    def count(self, k, n=1):
        self.counts[k] = self.counts.get(k, 0) + n

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)
            log(text)

    def send(self, header: dict, payload=b''):
        data = pack(header, payload)
        with self.wlock:
            self.conn.sendall(data)

    def arrived(self, frame, key, t, nbytes=None):
        a = self.arrivals.setdefault(int(frame), {})
        a[key] = t
        if nbytes is not None:
            a[key.split('_')[0] + '_bytes'] = nbytes

    # ---------------------------------------------------------------- receivers
    def start(self):
        cam, lid = self.cfg.get('camera') or {}, self.cfg.get('lidar') or {}
        if cam.get('transport') == 'h264-rtp':
            self.video = H264Receiver(self, int(cam['rtp_port']), self.agent.decoder)
            self.video.start()
        if lid.get('transport') == 'udp':
            self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
            self.udp.bind((self.agent.listen, int(lid['udp_port'])))
            self.udp.settimeout(0.2)
            threading.Thread(target=self._lidar_udp, daemon=True, name='agent-lidar-udp').start()
        if lid.get('ros2_topic'):
            try:
                self.ros = Ros2Out(lid['ros2_topic'], self.clock)
            except Exception as e:                       # noqa: BLE001 (reported; the session goes on)
                self.note(f'ROS 2 publishing is off: {type(e).__name__}: {e}')
        threading.Thread(target=self._stack_loop, daemon=True, name='agent-stack').start()

    def attach(self, kind: str, sock):
        self.streams.append(sock)
        threading.Thread(target=self._stream, args=(kind, sock), daemon=True, name=f'agent-{kind}').start()

    def _stream(self, kind, sock):
        import numpy as np
        try:
            while not self.stopping.is_set():
                h, payload = read_msg(sock)
                t_rx = self.clock.now()
                if h.get('t') == 'camera':
                    frame = int(h['frame'])
                    self.arrived(frame, 'cam_rx', t_rx, len(payload))
                    img = np.frombuffer(payload, np.uint8).reshape(int(h['h']), int(h['w']), 3)
                    t_dec = self.clock.now()
                    self.arrived(frame, 'dec_end', t_dec)
                    self.offer(frame, 'camera', img, h, {'cam_rx': t_rx, 'dec_start': t_rx, 'dec_end': t_dec})
                elif h.get('t') == 'lidar':
                    self._lidar(h, payload, t_rx)
        except (ConnectionError, OSError, ValueError) as e:
            if not self.stopping.is_set():
                self.note(f'{kind} stream ended: {e}')

    def _lidar_udp(self):
        ra = Reassembler()
        while not self.stopping.is_set():
            try:
                dg = self.udp.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            data = ra.feed(dg)
            if data is not None:
                t_rx = self.clock.now()
                h, payload = unpack(data)
                self._lidar(h, payload, t_rx)
        self.counts['lidar_udp_incomplete'] = ra.incomplete
        self.counts['lidar_udp_bad'] = ra.bad

    def _lidar(self, h, payload, t_rx):
        import numpy as np
        frame = int(h['frame'])
        self.arrived(frame, 'lidar_rx', t_rx, len(payload))
        if self.ros is not None and h.get('encoding') == 'pointcloud2':
            self.ros.publish(bytes(payload), frame)
        if h.get('encoding') == 'pointcloud2':
            pts, _ = decode_pointcloud2(payload)
        else:
            pts = np.frombuffer(payload, np.float32).reshape(-1, 4)
        self.offer(frame, 'lidar', pts, h, {'lidar_rx': t_rx})

    # ---------------------------------------------------------------- the stack
    def offer(self, frame: int, kind: str, data, meta: dict, stamps: dict):
        with self.cv:
            if frame <= self.last_run:
                self.count(f'{kind}_stale')              # its step's set already ran (or was superseded)
                return
            p = self.pending.setdefault(frame, {'first': self.clock.now(), 'stamps': {}, 'meta': {}})
            p[kind] = data
            p['stamps'].update(stamps)
            p['meta'].update({k: meta[k] for k in ('sim_t', 'hw_t', 'speed_mps') if k in meta and meta[k] is not None})
            p['meta'][f'{kind}_hw_t'] = meta.get('hw_t')
            self.cv.notify()

    def _choose(self):
        now = self.clock.now()
        full = [f for f, p in self.pending.items() if 'camera' in p and 'lidar' in p]
        if full:
            f = max(full)
        else:
            due = [f for f, p in self.pending.items() if now - p['first'] >= self.sync_timeout_ns]
            if not due:
                return None
            f = max(due)
        parts = self.pending.pop(f)
        for g in [g for g in self.pending if g < f]:
            self.pending.pop(g)
            self.count('sets_superseded')
        self.last_run = f
        return f, parts

    def _stack_loop(self):
        while not self.stopping.is_set():
            with self.cv:
                got = self._choose()
                if got is None:
                    self.cv.wait(0.002)
                    continue
            frame, parts = got
            try:
                self._run(frame, parts)
            except (ConnectionError, OSError):
                break
            except Exception as e:                       # noqa: BLE001 (a stack's bug: counted, the session goes on)
                self.count('stack_errors')
                self.note(f'the stack failed: {type(e).__name__}: {e}')

    def _run(self, frame, parts):
        missing = [k for k in ('camera', 'lidar') if k not in parts]
        for k in missing:
            self.count(f'sets_without_{k}')
        meta = dict(parts['meta'], frame=frame)
        t0 = self.clock.now()
        u = self.stack.act(parts.get('camera'), parts.get('lidar'), meta) or {}
        t1 = self.clock.now()
        self.count('sets_run')
        dev = dict(parts['stamps'], stack_start=t0, stack_end=t1)
        ctrl = {'t': 'control', 'frame': frame, 'throttle': float(u.get('throttle', 0.0)),
                'steer': float(u.get('steer', 0.0)), 'brake': float(u.get('brake', 0.0)), 'missing': missing,
                'info': u.get('info') or {}, 'dev': dev}
        dev['ctrl_sent'] = self.clock.now()
        self.send(ctrl)

    # ---------------------------------------------------------------- the end
    def stop(self):
        self.stopping.set()
        with self.cv:
            self.cv.notify_all()
        if self.video is not None:
            self.video.stop()
        for s in self.streams + ([self.udp] if self.udp is not None else []):
            try:
                s.close()
            except OSError:
                pass
        if self.ros is not None:
            self.ros.close()

    def stats(self) -> dict:
        keys = ('cam_rx', 'dec_end', 'lidar_rx', 'cam_bytes', 'lidar_bytes')
        arr = [[f] + [a.get(k) for k in keys] for f, a in sorted(self.arrivals.items())]
        return {'t': 'stats', 'counts': self.counts, 'notes': self.notes, 'stack': self.stack_name,
                'decoder': self.video.decoder if self.video is not None else None,
                'arrival_fields': ['frame'] + list(keys), 'arrivals': arr,
                'ros2': None if self.ros is None else self.ros.stats()}


class Ros2Out:
    """Republishes each lidar message (already sensor_msgs/PointCloud2 in CDR, from the host) on a ROS 2 topic with
    the sensor-data QoS, unchanged (rclpy's publish of serialised bytes). Needs ROS 2's environment sourced."""

    def __init__(self, topic: str, clock):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node('falcon_hil_agent')
        self.pub = self.node.create_publisher(PointCloud2, topic, qos_profile_sensor_data)
        self.topic, self.clock = topic, clock
        self.n = 0
        self.publish_ns: list = []

    def publish(self, cdr: bytes, frame: int):
        t0 = self.clock.now()
        self.pub.publish(cdr)
        self.publish_ns.append(self.clock.now() - t0)
        self.n += 1

    def stats(self) -> dict:
        p = sorted(self.publish_ns)
        return {'topic': self.topic, 'published': self.n,
                'publish_median_ms': round(p[len(p) // 2] / 1e6, 3) if p else None}

    def close(self):
        try:
            self.node.destroy_node()
            self.rclpy.shutdown()
        except Exception:                                # noqa: BLE001
            pass


# ============================================================================================ the agent
class Agent:
    def __init__(self, listen: str, port: int, stack: str = '', stack_params: dict | None = None,
                 decoder: str = 'auto', clock: DeviceClock | None = None):
        self.listen, self.port = listen, int(port)
        self.stack_spec, self.stack_params = stack, dict(stack_params or {})
        self.decoder = decoder
        self.clock = clock or DeviceClock()
        self.session = None
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp.bind((listen, self.port))
        self.tcp.listen(8)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind((listen, self.port))

    def pings(self):
        """Clock pings: answered at once, receive and send stamped on this device's clock."""
        while not self.done.is_set():
            try:
                data, addr = self.udp.recvfrom(64)
            except OSError:
                return
            t1 = self.clock.now()
            if len(data) != PING.size:
                continue
            magic, seq, t0 = PING.unpack(data)
            if magic != b'FPNG':
                continue
            try:
                self.udp.sendto(PONG.pack(b'FPON', seq, t0, t1, self.clock.now()), addr)
            except OSError:
                pass

    def serve(self, once: bool = False):
        threading.Thread(target=self.pings, daemon=True, name='agent-pings').start()
        log(f'listening on {self.listen}:{self.port} (TCP and UDP); clock '
            f'{"emulated" if self.clock.emulated else "CLOCK_MONOTONIC"}; '
            f'stack {self.stack_spec or "brake_for_obstacles"}')
        try:
            while not self.done.is_set():
                try:
                    conn, _addr = self.tcp.accept()
                except OSError:
                    break
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(target=self.handle, args=(conn, once), daemon=True, name='agent-conn').start()
        finally:
            self.close()

    def handle(self, conn, once):
        try:
            conn.settimeout(30.0)
            h, _ = read_msg(conn)
            conn.settimeout(None)
        except (ConnectionError, OSError, ValueError) as e:
            log(f'a connection sent no valid first message: {e}')
            conn.close()
            return
        if h.get('t') == 'stream':
            with self.lock:
                ses = self.session
            if ses is None:
                conn.close()
                return
            ses.attach(h.get('kind', ''), conn)
            return
        if h.get('t') != 'hello':
            conn.close()
            return
        with self.lock:
            busy = self.session is not None
        if busy or int(h.get('protocol', 0)) != PROTOCOL:
            why = 'busy with another session' if busy else \
                f'protocol {h.get("protocol")} (this agent speaks {PROTOCOL})'
            try:
                conn.sendall(pack({'t': 'error', 'error': why}))
            finally:
                conn.close()
            return
        self.run_session(conn, h)
        if once:
            self.done.set()
            self.close()

    def run_session(self, conn, hello):
        try:
            ses = Session(self, conn, hello)
            ses.start()
        except Exception as e:                           # noqa: BLE001 (told to the host, then closed)
            log(f'session refused: {type(e).__name__}: {e}')
            try:
                conn.sendall(pack({'t': 'error', 'error': f'{type(e).__name__}: {e}'}))
            except OSError:
                pass
            conn.close()
            return
        with self.lock:
            self.session = ses
        log('session started')
        try:
            ses.send({'t': 'ready', 'env': environment(self.decoder), 'stack': ses.stack_name,
                      'decoder': ses.video.decoder if ses.video is not None else None,
                      'clock': {'emulated': self.clock.emulated}})
            while True:
                h, _ = read_msg(conn)
                if h.get('t') == 'bye':
                    break
        except (ConnectionError, OSError, ValueError) as e:
            ses.note(f'the session connection ended: {e}')
        finally:
            ses.stop()
            try:
                ses.send(ses.stats())
            except OSError:
                pass
            conn.close()
            with self.lock:
                self.session = None
            log(f'session ended: {json.dumps(ses.counts)}')

    def close(self):
        self.done.set()
        for s in (self.tcp, self.udp):
            try:
                s.shutdown(socket.SHUT_RDWR)            # wakes a thread blocked in accept() (close alone does not)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='device_agent.py', description='Falcon Autonomy HIL device agent')
    ap.add_argument('--listen', default='0.0.0.0', help='address to listen on (default all)')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT, help='TCP and UDP port (sessions, streams, pings)')
    ap.add_argument('--once', action='store_true', help='exit after one session')
    ap.add_argument('--check', action='store_true', help='print what this device has as JSON and exit')
    ap.add_argument('--stack', default='', help='module:Class of the stack (default: brake_for_obstacles)')
    ap.add_argument('--stack-param', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--decoder', default='auto', help=f'H.264 decoder element (auto: first of {", ".join(DECODERS)})')
    ap.add_argument('--clock-offset-ms', type=float, default=0.0, help='emulate a clock offset (simulated devices)')
    ap.add_argument('--clock-drift-ppm', type=float, default=0.0, help='emulate a clock drift (simulated devices)')
    a = ap.parse_args(argv)
    if a.check:
        print(json.dumps(environment(a.decoder), indent=1))
        return 0
    params = {}
    for kv in a.stack_param:
        k, _, v = kv.partition('=')
        params[k] = v
    agent = Agent(a.listen, a.port, a.stack, params, a.decoder, DeviceClock(a.clock_offset_ms, a.clock_drift_ppm))
    try:
        agent.serve(once=a.once)
    except KeyboardInterrupt:
        agent.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
