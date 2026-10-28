"""The live link: how a live worker streams frames to the GUI over a local (Unix-domain) socket (standard library).

The GUI listens on a socket path; the live worker connects to it. Each message is

    >II  header length, payload length (big-endian uint32)
    header  UTF-8 JSON with a type "t"
    payload bytes (a JPEG for "frame", empty otherwise)

Message types:
    {"t": "map", "x": [...], "y": [...], "yaw": [...], "step_m": 3.0, "name": "Town10HD_Opt", "rule": "RHT"}
          the road network once at the start: lane waypoints every step_m metres (x, y in CARLA's frame, yaw °)
    {"t": "frame", "frame": 1203, "sim_t": 60.15, "wall_t": 61.2, "view": "chase", "w": 1280, "h": 720,
     "ticks_per_s": 20.0, "rtf": 1.0, "sent": 1200, "dropped": 3, "ego": {...}, "speed_kmh": 31.4,
     "control": {...}, "mode": "manual", "actors": [[id, kind, x, y, yaw, half_len, half_wid], ...],
     "boxes": [[id, kind, x0, y0, x1, y1], ...], "recording": "…/rec-001" | null, "events": [...]}
          one view frame (the payload is its JPEG); boxes are full ground-truth boxes projected into the view
    {"t": "bye", "reason": "..."}

Frames are "latest only": the worker keeps one frame waiting to be sent; when the GUI has not taken it before the
next one is ready it is replaced, and counted in "dropped". The simulation itself never drops a step (synchronous
mode); "rtf" is simulated seconds per wall second over the last two seconds.
"""
from __future__ import annotations

import json
import socket
import struct
import threading

HEAD = struct.Struct('>II')
MAX_HEADER = 64 << 20
MAX_PAYLOAD = 64 << 20


def pack(header: dict, payload: bytes = b'') -> bytes:
    h = json.dumps(header, separators=(',', ':')).encode('utf-8')
    return HEAD.pack(len(h), len(payload)) + h + payload


class Reader:
    """Incremental decoder: feed(bytes) returns the complete (header, payload) messages so far."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes) -> list:
        self.buf += data
        out = []
        while len(self.buf) >= HEAD.size:
            hl, pl = HEAD.unpack_from(self.buf, 0)
            if hl > MAX_HEADER or pl > MAX_PAYLOAD:
                raise ValueError(f'live link message too large ({hl} + {pl} bytes)')
            end = HEAD.size + hl + pl
            if len(self.buf) < end:
                break
            header = json.loads(bytes(self.buf[HEAD.size:HEAD.size + hl]).decode('utf-8'))
            payload = bytes(self.buf[HEAD.size + hl:end])
            del self.buf[:end]
            out.append((header, payload))
        return out


class Sender:
    """The worker's side: a thread sending the latest message; put() replaces a message not yet sent (dropped)."""

    def __init__(self, path: str, timeout: float = 10.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)
        self.sock.settimeout(None)
        self.cv = threading.Condition()
        self.slot = None
        self.urgent: list = []             # never dropped (map, bye)
        self.sent = 0
        self.dropped = 0
        self.closed = False
        self.error = None
        self.t = threading.Thread(target=self._run, daemon=True, name='falcon-livelink')
        self.t.start()

    def put(self, header: dict, payload: bytes = b''):
        with self.cv:
            if self.slot is not None:
                self.dropped += 1
            self.slot = pack(header, payload)
            self.cv.notify()

    def put_urgent(self, header: dict, payload: bytes = b''):
        with self.cv:
            self.urgent.append(pack(header, payload))
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while not self.closed and self.slot is None and not self.urgent:
                    self.cv.wait()
                if self.urgent:
                    msg = self.urgent.pop(0)
                    frame = False
                elif self.slot is not None:
                    msg, self.slot = self.slot, None
                    frame = True
                else:
                    return
            try:
                self.sock.sendall(msg)
            except OSError as e:
                self.error = e
                return
            if frame:
                self.sent += 1

    def close(self, reason=''):
        try:
            self.put_urgent({'t': 'bye', 'reason': reason})
        except OSError:
            pass
        with self.cv:
            self.closed = True
            self.cv.notify()
        self.t.join(5.0)
        try:
            self.sock.close()
        except OSError:
            pass
