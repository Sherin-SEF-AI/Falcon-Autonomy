"""Cameras that reach the device as V4L2 capture nodes — a GMSL2 deserialiser on the Jetson AGX Orin (MAX9296 /
MAX96712 boards appear as /dev/videoN through the Jetson's VI driver), or a v4l2loopback node on this machine — and
the interface for putting simulated frames onto a GMSL2 link (docs/hardware.md#gmsl2). Standard library and numpy;
copied beside the device agent (falcon hil deploy).

    Grabber             open() · grab(timeout) → Grabbed | None · close()
    V4L2CaptureGrabber  a capture node by memory-mapped streaming I/O: VIDIOC_QUERYCAP, G_FMT (or S_FMT when a size
                        and format are given), REQBUFS (MMAP), QUERYBUF + mmap per buffer, QBUF, STREAMON; then per
                        frame select() → DQBUF → the image, its sequence number and the driver's buffer timestamp →
                        QBUF. The timestamp's clock and source come from the buffer's flags (CLOCK_MONOTONIC on Linux
                        drivers that say so; start or end of exposure) and are reported, never assumed.
    Injector            inject(image, frame) — puts a simulated frame on a camera link, so that the device's own
                        capture path (deserialiser, VI, its driver) carries it.
    FakeInjector        the only Injector here: an in-memory queue read back through its grabber() after a set
                        latency. A real GMSL2 injector needs serializer-injection hardware (a frame grabber/injector
                        board feeding a MAX9295 serializer, for example), which this project does not have; nothing
                        here talks to one.

What a GMSL2 path adds and what is not modelled: the serializer/deserializer's own latency (sub-millisecond, fixed per
link), the camera's exposure and rolling shutter (the session stamps the simulation time; core.faults models a rolling
shutter for recorded data), the ISP when the deserialiser carries raw Bayer (Jetson's ISP via libargus is not V4L2:
Bayer through VI gives the raw mosaic, which this grabber does not demosaic — use a YUV camera or the ISP's own path).
"""
from __future__ import annotations

import mmap
import os
import select
import struct
import time
from dataclasses import dataclass

import numpy as np

try:
    from . import v4l2out as V
except ImportError:                                      # beside the agent on the device
    import v4l2out as V

READ_FORMATS = ('YUYV', 'UYVY', 'RGB3', 'BGR3', 'NV12', 'GREY')


@dataclass
class Grabbed:
    image: np.ndarray                 # H × W × 3 BGR uint8
    sequence: int                     # the driver's frame counter
    t_buffer_ns: int | None           # the driver's timestamp (its clock in clock), ns
    t_dequeued_ns: int                # CLOCK_MONOTONIC when DQBUF returned
    t_converted_ns: int               # CLOCK_MONOTONIC when the BGR image was ready
    clock: str                        # 'monotonic', 'copy' (passed through from the writer) or 'unknown'
    source: str                       # 'start_of_exposure', 'end_of_frame' or 'unknown'
    barcode: int | None = None        # the frame number painted by v4l2out.paint_barcode, when present


class Grabber:
    """A camera on the device (module doc)."""

    def open(self):
        raise NotImplementedError

    def grab(self, timeout: float = 1.0) -> Grabbed | None:
        raise NotImplementedError

    def close(self):
        pass

    def describe(self) -> dict:
        return {}


class V4L2CaptureGrabber(Grabber):
    """A V4L2 capture node by mmap streaming I/O (module doc). The system calls are parameters so that a test can
    stand in for the kernel."""

    def __init__(self, device: str, width: int = 0, height: int = 0, fmt: str = '', buffers: int = 4,
                 read_barcode: bool = True, *, opener=os.open, ioctl=None, mapper=None, selector=None, close=os.close):
        self.device, self.width, self.height, self.fmt = device, int(width), int(height), fmt
        self.nbuf = int(buffers)
        self.want_barcode = read_barcode
        self._open, self._ioctl = opener, ioctl or V._ioctl_default()
        self._mmap = mapper or (lambda fd, length, offset: mmap.mmap(fd, length, mmap.MAP_SHARED,
                                                                     mmap.PROT_READ | mmap.PROT_WRITE, offset=offset))
        self._select = selector or select.select
        self._close = close
        self.fd = None
        self.maps: list = []
        self.streaming = False
        self.frames = 0
        self.cap: dict = {}
        self.format: dict = {}

    def open(self):
        try:
            self.fd = self._open(self.device, os.O_RDWR | os.O_NONBLOCK)
        except OSError as e:
            raise V.V4L2Error(f'cannot open {self.device}: {e.strerror or e}') from e
        try:
            cap = bytearray(V.CAP.size)
            self._ioctl(self.fd, V.VIDIOC_QUERYCAP, cap, True)
            self.cap = V.unpack_cap(cap)
            if not self.cap['capture']:
                why = ' (multi-planar capture is not read here)' if self.cap['mplane'] else ''
                raise V.V4L2Error(f'{self.device} ({self.cap["card"]}) is not a single-plane V4L2 capture '
                                  f'device{why}')
            if not self.cap['streaming']:
                raise V.V4L2Error(f'{self.device} does not support streaming I/O')
            if self.width and self.height and self.fmt:
                f = V.pack_format(V.BUF_TYPE_CAPTURE, self.width, self.height, self.fmt)
                self._ioctl(self.fd, V.VIDIOC_S_FMT, f, True)
            else:
                f = bytearray(V.FMT_SIZE)
                struct.pack_into('I', f, 0, V.BUF_TYPE_CAPTURE)
                self._ioctl(self.fd, V.VIDIOC_G_FMT, f, True)
            self.format = V.unpack_format(f)
            if self.format['format'] not in READ_FORMATS:
                raise V.V4L2Error(f'{self.device} delivers {self.format["format"]}, not one of '
                                  f'{", ".join(READ_FORMATS)}')
            self.width, self.height, self.fmt = self.format['width'], self.format['height'], self.format['format']
            rq = bytearray(V.REQBUFS.pack(self.nbuf, V.BUF_TYPE_CAPTURE, V.MEMORY_MMAP, 0, 0))
            self._ioctl(self.fd, V.VIDIOC_REQBUFS, rq, True)
            n = V.REQBUFS.unpack(rq)[0]
            if n < 2:
                raise V.V4L2Error(f'{self.device} granted {n} buffers (2 at least are needed)')
            for i in range(n):
                b = self._buffer(i)
                self._ioctl(self.fd, V.VIDIOC_QUERYBUF, b, True)
                f = V.BUFFER.unpack(b)
                self.maps.append(self._mmap(self.fd, f[11], f[10]))       # length, m.offset
            for i in range(n):
                self._ioctl(self.fd, V.VIDIOC_QBUF, self._buffer(i), True)
            self._ioctl(self.fd, V.VIDIOC_STREAMON, bytearray(struct.pack('I', V.BUF_TYPE_CAPTURE)), True)
            self.streaming = True
        except OSError as e:
            self.close()
            raise V.V4L2Error(f'{self.device}: {e}') from e
        except V.V4L2Error:
            self.close()
            raise
        return self

    @staticmethod
    def _buffer(index: int = 0) -> bytearray:
        return bytearray(V.BUFFER.pack(index, V.BUF_TYPE_CAPTURE, 0, 0, 0, 0, 0, b'', 0, V.MEMORY_MMAP, 0, 0, 0, 0))

    def grab(self, timeout: float = 1.0) -> Grabbed | None:
        r, _, _ = self._select([self.fd], [], [], timeout)
        if not r:
            return None
        b = self._buffer()
        try:
            self._ioctl(self.fd, V.VIDIOC_DQBUF, b, True)
        except BlockingIOError:
            return None
        t_dq = time.monotonic_ns()
        idx, _t, used, flags, _field, sec, usec, _tc, seq, *_ = V.BUFFER.unpack(b)
        data = bytes(self.maps[idx][:used or len(self.maps[idx])])
        self._ioctl(self.fd, V.VIDIOC_QBUF, self._buffer(idx), True)
        img = V.to_bgr(data, self.width, self.height, self.fmt, self.format.get('bytesperline', 0))
        t_conv = time.monotonic_ns()
        kind = flags & V.BUF_FLAG_TIMESTAMP_MASK
        clock = {V.BUF_FLAG_TIMESTAMP_MONOTONIC: 'monotonic', V.BUF_FLAG_TIMESTAMP_COPY: 'copy'}.get(kind, 'unknown')
        src = 'start_of_exposure' if flags & V.BUF_FLAG_TSTAMP_SRC_SOE else 'end_of_frame'
        self.frames += 1
        ts = sec * 1_000_000_000 + usec * 1000 if (sec or usec) else None
        return Grabbed(img, int(seq), ts, t_dq, t_conv, clock, src if clock == 'monotonic' else 'unknown',
                       V.read_barcode(img) if self.want_barcode else None)

    def close(self):
        if self.fd is None:
            return
        if self.streaming:
            try:
                self._ioctl(self.fd, V.VIDIOC_STREAMOFF, bytearray(struct.pack('I', V.BUF_TYPE_CAPTURE)), True)
            except OSError:
                pass
            self.streaming = False
        for m in self.maps:
            try:
                m.close()
            except (OSError, ValueError, BufferError):
                pass
        self.maps = []
        try:
            self._close(self.fd)
        except OSError:
            pass
        self.fd = None

    def describe(self) -> dict:
        return {'device': self.device, 'card': self.cap.get('card'), 'driver': self.cap.get('driver'),
                'width': self.width, 'height': self.height, 'format': self.fmt, 'buffers': len(self.maps)}


class Injector:
    """Puts a simulated frame onto a camera link (module doc)."""

    def inject(self, image: np.ndarray, frame: int):
        raise NotImplementedError

    def close(self):
        pass


class FakeInjector(Injector):
    """The only injector here: frames queued in memory, delivered by grabber() after latency_s (a stand-in for a
    serializer-injection board; it models a fixed link latency, nothing of GMSL2's protocol)."""

    def __init__(self, latency_s: float = 0.0, depth: int = 4):
        import collections
        import threading
        self.latency_ns = int(float(latency_s) * 1e9)
        self.q = collections.deque(maxlen=int(depth))
        self.cv = threading.Condition()
        self.dropped = 0
        self.seq = 0

    def inject(self, image: np.ndarray, frame: int):
        with self.cv:
            if len(self.q) == self.q.maxlen:
                self.dropped += 1
            self.seq += 1
            self.q.append((time.monotonic_ns(), self.seq, np.ascontiguousarray(np.asarray(image)[:, :, :3]), frame))
            self.cv.notify_all()

    def grabber(self) -> Grabber:
        inj = self

        class _G(Grabber):
            def open(self):
                return self

            def grab(self, timeout: float = 1.0):
                end = time.monotonic() + timeout
                with inj.cv:
                    while not inj.q:
                        left = end - time.monotonic()
                        if left <= 0:
                            return None
                        inj.cv.wait(left)
                    t_in, seq, img, frame = inj.q.popleft()
                wait = (t_in + inj.latency_ns - time.monotonic_ns()) / 1e9
                if wait > 0:
                    time.sleep(wait)
                t = time.monotonic_ns()
                return Grabbed(img, seq, t_in + inj.latency_ns, t, t, 'monotonic', 'end_of_frame', frame)

            def describe(self):
                return {'device': 'fake injector', 'latency_ms': inj.latency_ns / 1e6}
        return _G()
