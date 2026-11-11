"""The camera of a HIL session as a V4L2 video device (docs/hardware.md#the-camera-as-a-v4l2-device): frames written
into a V4L2 *output* device — a v4l2loopback node on this machine, which any program then opens as a camera — with
their stamps on a sidecar UDP stream and, optionally, the frame number painted as a barcode into the top rows so that
a reader can measure the path's latency frame by frame. Standard library and numpy (GStreamer optional); also copied
beside the device agent (falcon hil deploy), whose V4L2 capture (hil.gmsl) shares these definitions.

Writer (V4L2Writer): open the node, VIDIOC_QUERYCAP (it must be an output device), VIDIOC_S_FMT for
V4L2_BUF_TYPE_VIDEO_OUTPUT with the size and pixel format (YUYV, UYVY, RGB3 or NV12; the driver's answer is checked),
then one write() of sizeimage bytes per frame — v4l2loopback's read/write path, no buffer queue. The alternative
(GstV4L2Writer) is GStreamer's `appsrc ! videoconvert ! v4l2sink`. Colour conversion from the session's BGRA is
ITU-R BT.601, limited range (what a USB or GMSL camera's YUV usually is).

Stamps (StampSender): one UDP datagram per frame, JSON {"f": frame, "s": sim_t, "h": hw_t, "w": the write() start on
this machine's CLOCK_MONOTONIC (ns), "n": the sequence of writes}: a reader pairs them with what it captured, by the
barcode or by the order.

Barcode: rows 0 to BARCODE_ROWS − 1 carry START (1010) then the frame number (32 bits, most significant first) then an
8-bit check (the bytes' sum XOR 0xA5), each bit a cell of equal width, white (255) or black (0) in R = G = B so that
no chroma is touched — it survives YUV 4:2:2 and 4:2:0 subsampling; read_barcode() decodes it from any converted frame.
The painted rows are part of the image the stack sees: it is off unless asked for.

The ioctl layouts are those of the 64-bit kernel ABI (x86-64, aarch64: the Jetson); a 32-bit Python is refused.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import struct
import time

import numpy as np

if struct.calcsize('P') != 8:                                  # pragma: no cover
    raise ImportError('the V4L2 structures here are the 64-bit ABI\'s')

# ---------------------------------------------------------------------------------------------- the kernel's ABI
_IOC_WRITE, _IOC_READ = 1, 2


def _ioc(d: int, nr: int, size: int) -> int:
    return (d << 30) | (size << 16) | (ord('V') << 8) | nr


CAP = struct.Struct('16s32s32sIII12x')                         # struct v4l2_capability (104 bytes)
PIX = struct.Struct('IIIIIIIIIIII')                            # struct v4l2_pix_format (48 bytes)
FMT_SIZE = 208                                                 # struct v4l2_format: u32 type, pad, 200-byte union
REQBUFS = struct.Struct('IIIIB3x')                             # struct v4l2_requestbuffers (20 bytes)
BUFFER = struct.Struct('IIIII4xqq16sIIQIII4x')                 # struct v4l2_buffer (88 bytes, 64-bit)
VIDIOC_QUERYCAP = _ioc(_IOC_READ, 0, CAP.size)
VIDIOC_G_FMT = _ioc(_IOC_READ | _IOC_WRITE, 4, FMT_SIZE)
VIDIOC_S_FMT = _ioc(_IOC_READ | _IOC_WRITE, 5, FMT_SIZE)
VIDIOC_REQBUFS = _ioc(_IOC_READ | _IOC_WRITE, 8, REQBUFS.size)
VIDIOC_QUERYBUF = _ioc(_IOC_READ | _IOC_WRITE, 9, BUFFER.size)
VIDIOC_QBUF = _ioc(_IOC_READ | _IOC_WRITE, 15, BUFFER.size)
VIDIOC_DQBUF = _ioc(_IOC_READ | _IOC_WRITE, 17, BUFFER.size)
VIDIOC_STREAMON = _ioc(_IOC_WRITE, 18, 4)
VIDIOC_STREAMOFF = _ioc(_IOC_WRITE, 19, 4)
BUF_TYPE_CAPTURE, BUF_TYPE_OUTPUT = 1, 2
MEMORY_MMAP = 1
FIELD_NONE = 1
COLORSPACE_SMPTE170M, COLORSPACE_SRGB = 1, 8
CAP_VIDEO_CAPTURE, CAP_VIDEO_OUTPUT = 0x1, 0x2
CAP_VIDEO_CAPTURE_MPLANE, CAP_VIDEO_OUTPUT_MPLANE = 0x1000, 0x2000
CAP_READWRITE, CAP_STREAMING, CAP_DEVICE_CAPS = 0x01000000, 0x04000000, 0x80000000
BUF_FLAG_TIMESTAMP_MASK = 0xE000
BUF_FLAG_TIMESTAMP_MONOTONIC, BUF_FLAG_TIMESTAMP_COPY = 0x2000, 0x4000
BUF_FLAG_TSTAMP_SRC_SOE = 0x10000


def fourcc(code: str) -> int:
    return struct.unpack('<I', code.encode('ascii'))[0]


def fourcc_str(v: int) -> str:
    return struct.pack('<I', v).decode('ascii', 'replace')


# per format: bytes per line from the width, the image's size from (width, height)
FORMATS = {
    'YUYV': (lambda w: 2 * w, lambda w, h: 2 * w * h),
    'UYVY': (lambda w: 2 * w, lambda w, h: 2 * w * h),
    'RGB3': (lambda w: 3 * w, lambda w, h: 3 * w * h),
    'BGR3': (lambda w: 3 * w, lambda w, h: 3 * w * h),
    'NV12': (lambda w: w, lambda w, h: w * h * 3 // 2),
    'GREY': (lambda w: w, lambda w, h: w * h),
}
WRITE_FORMATS = ('YUYV', 'UYVY', 'RGB3', 'NV12')


def pack_format(buf_type: int, width: int, height: int, fmt: str, bytesperline: int = 0,
                sizeimage: int = 0) -> bytearray:
    """struct v4l2_format for a single-plane pixel format."""
    b = bytearray(FMT_SIZE)
    struct.pack_into('I', b, 0, buf_type)
    PIX.pack_into(b, 8, int(width), int(height), fourcc(fmt), FIELD_NONE, int(bytesperline), int(sizeimage),
                  COLORSPACE_SMPTE170M if fmt in ('YUYV', 'UYVY', 'NV12') else COLORSPACE_SRGB, 0, 0, 0, 0, 0)
    return b


def unpack_format(b) -> dict:
    t = struct.unpack_from('I', b, 0)[0]
    w, h, pf, field, bpl, size, cs, *_ = PIX.unpack_from(b, 8)
    return {'type': t, 'width': w, 'height': h, 'format': fourcc_str(pf), 'field': field, 'bytesperline': bpl,
            'sizeimage': size, 'colorspace': cs}


def unpack_cap(b) -> dict:
    drv, card, bus, ver, caps, dcaps = CAP.unpack_from(b, 0)
    s = lambda x: x.split(b'\0', 1)[0].decode('utf-8', 'replace')     # noqa: E731
    use = dcaps if caps & CAP_DEVICE_CAPS else caps
    return {'driver': s(drv), 'card': s(card), 'bus': s(bus), 'version': f'{ver >> 16}.{(ver >> 8) & 255}.{ver & 255}',
            'capabilities': caps, 'device_caps': use,
            'capture': bool(use & CAP_VIDEO_CAPTURE), 'output': bool(use & CAP_VIDEO_OUTPUT),
            'mplane': bool(use & (CAP_VIDEO_CAPTURE_MPLANE | CAP_VIDEO_OUTPUT_MPLANE)),
            'readwrite': bool(use & CAP_READWRITE), 'streaming': bool(use & CAP_STREAMING)}


# ---------------------------------------------------------------------------------------------- pixels (BT.601)
def _bgr(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError('a frame is H × W × 3 (BGR) or × 4 (BGRA) uint8')
    return a[:, :, :3]


def _ycc(bgr: np.ndarray):
    f = bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    cb = 128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0
    cr = 128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0
    return y, cb, cr


def _u8(x) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def convert(img, fmt: str) -> bytes:
    """A BGR(A) frame as the bytes of one V4L2 image in fmt (width must be even for the YUV formats, height too for
    NV12)."""
    bgr = _bgr(img)
    h, w = bgr.shape[:2]
    if fmt == 'RGB3':
        return np.ascontiguousarray(bgr[:, :, ::-1]).tobytes()
    if fmt == 'BGR3':
        return np.ascontiguousarray(bgr).tobytes()
    if w % 2 or (fmt == 'NV12' and h % 2):
        raise ValueError(f'{fmt} needs an even width{" and height" if fmt == "NV12" else ""} ({w} × {h})')
    y, cb, cr = _ycc(bgr)
    if fmt in ('YUYV', 'UYVY'):
        u = (cb[:, 0::2] + cb[:, 1::2]) / 2.0
        v = (cr[:, 0::2] + cr[:, 1::2]) / 2.0
        out = np.empty((h, w * 2), np.uint8)
        if fmt == 'YUYV':
            out[:, 0::4], out[:, 1::4], out[:, 2::4], out[:, 3::4] = _u8(y[:, 0::2]), _u8(u), _u8(y[:, 1::2]), _u8(v)
        else:
            out[:, 0::4], out[:, 1::4], out[:, 2::4], out[:, 3::4] = _u8(u), _u8(y[:, 0::2]), _u8(v), _u8(y[:, 1::2])
        return out.tobytes()
    if fmt == 'NV12':
        u = (cb[0::2, 0::2] + cb[0::2, 1::2] + cb[1::2, 0::2] + cb[1::2, 1::2]) / 4.0
        v = (cr[0::2, 0::2] + cr[0::2, 1::2] + cr[1::2, 0::2] + cr[1::2, 1::2]) / 4.0
        uv = np.empty((h // 2, w), np.uint8)
        uv[:, 0::2], uv[:, 1::2] = _u8(u), _u8(v)
        return _u8(y).tobytes() + uv.tobytes()
    if fmt == 'GREY':
        return _u8(y).tobytes()
    raise ValueError(f'pixel format {fmt} is not one of {", ".join(FORMATS)}')


def _rgb_from_ycc(y, cb, cr) -> np.ndarray:
    y = 1.164 * (y.astype(np.float32) - 16.0)
    cb, cr = cb.astype(np.float32) - 128.0, cr.astype(np.float32) - 128.0
    return np.dstack([_u8(y + 2.017 * cb), _u8(y - 0.392 * cb - 0.813 * cr), _u8(y + 1.596 * cr)])   # B, G, R


def to_bgr(data, width: int, height: int, fmt: str, bytesperline: int = 0) -> np.ndarray:
    """One captured V4L2 image as H × W × 3 BGR uint8."""
    w, h = int(width), int(height)
    bpl = int(bytesperline) or FORMATS[fmt][0](w)
    a = np.frombuffer(data, np.uint8)
    if fmt in ('RGB3', 'BGR3'):
        img = a[:bpl * h].reshape(h, bpl)[:, :3 * w].reshape(h, w, 3)
        return np.ascontiguousarray(img[:, :, ::-1] if fmt == 'RGB3' else img)
    if fmt in ('YUYV', 'UYVY'):
        p = a[:bpl * h].reshape(h, bpl)[:, :2 * w]
        yi, ui, vi = (0, 1, 3) if fmt == 'YUYV' else (1, 0, 2)
        y = p[:, yi::2]
        u = np.repeat(p[:, ui::4], 2, axis=1)
        v = np.repeat(p[:, vi::4], 2, axis=1)
        return _rgb_from_ycc(y, u, v)
    if fmt == 'NV12':
        y = a[:bpl * h].reshape(h, bpl)[:, :w]
        uv = a[bpl * h:bpl * h + bpl * (h // 2)].reshape(h // 2, bpl)[:, :w]
        u = np.repeat(np.repeat(uv[:, 0::2], 2, axis=0), 2, axis=1)
        v = np.repeat(np.repeat(uv[:, 1::2], 2, axis=0), 2, axis=1)
        return _rgb_from_ycc(y, u, v)
    if fmt == 'GREY':
        y = a[:bpl * h].reshape(h, bpl)[:, :w]
        return np.dstack([y, y, y])
    raise ValueError(f'pixel format {fmt} is not read')


# ---------------------------------------------------------------------------------------------- the barcode
BARCODE_ROWS = 2
START = (1, 0, 1, 0)
BARCODE_BITS = len(START) + 32 + 8


def _check(n: int) -> int:
    return (sum(int(n & 0xFFFFFFFF).to_bytes(4, 'big')) & 0xFF) ^ 0xA5


def barcode_bits(n: int) -> list:
    n &= 0xFFFFFFFF
    return list(START) + [(n >> (31 - i)) & 1 for i in range(32)] + [(_check(n) >> (7 - i)) & 1 for i in range(8)]


def paint_barcode(img: np.ndarray, n: int, rows: int = BARCODE_ROWS) -> np.ndarray:
    """img (H × W × 3|4, written in place) with frame number n in its top rows; the cell width is W // 44 (≥ 1)."""
    w = img.shape[1]
    cell = max(w // BARCODE_BITS, 1)
    if cell * BARCODE_BITS > w:
        raise ValueError(f'a barcode needs a frame {BARCODE_BITS} pixels wide at least')
    for i, bit in enumerate(barcode_bits(n)):
        img[:rows, i * cell:(i + 1) * cell, :3] = 255 if bit else 0
    return img


def read_barcode(bgr: np.ndarray):
    """The frame number painted by paint_barcode, or None (no start pattern, a failed check)."""
    w = bgr.shape[1]
    cell = max(w // BARCODE_BITS, 1)
    if cell * BARCODE_BITS > w:
        return None
    row = bgr[0, :cell * BARCODE_BITS, :3].astype(np.float32).mean(axis=1)
    mid = row.reshape(BARCODE_BITS, cell)[:, cell // 4: cell - cell // 4 or cell].mean(axis=1)
    bits = [int(x > 128) for x in mid]
    if tuple(bits[:4]) != START:
        return None
    n = 0
    for b in bits[4:36]:
        n = (n << 1) | b
    chk = 0
    for b in bits[36:]:
        chk = (chk << 1) | b
    return n if chk == _check(n) else None


# ---------------------------------------------------------------------------------------------- the writers
class V4L2Error(RuntimeError):
    pass


def _ioctl_default():
    import fcntl
    return fcntl.ioctl


class V4L2Writer:
    """Frames into a V4L2 output device by write() (module doc). The system calls are parameters so that a test
    can stand in for the kernel."""

    def __init__(self, device: str, width: int, height: int, fmt: str = 'YUYV', barcode: bool = False, *,
                 opener=os.open, ioctl=None, write=os.write, close=os.close):
        if fmt not in WRITE_FORMATS:
            raise V4L2Error(f'pixel format {fmt} is not one of {", ".join(WRITE_FORMATS)}')
        self.device, self.width, self.height, self.fmt, self.barcode = device, int(width), int(height), fmt, barcode
        self._ioctl, self._write, self._close = ioctl or _ioctl_default(), write, close
        try:
            self.fd = opener(device, os.O_RDWR)
        except OSError as e:
            raise V4L2Error(f'cannot open {device}: {e.strerror or e} (falcon hil setup prints how to create '
                            'v4l2loopback devices)') from e
        try:
            cap = bytearray(CAP.size)
            self._ioctl(self.fd, VIDIOC_QUERYCAP, cap, True)
            self.cap = unpack_cap(cap)
            if not self.cap['output']:
                raise V4L2Error(f'{device} ({self.cap["card"]}, {self.cap["driver"]}) is not a V4L2 output device')
            bpl, size = FORMATS[fmt][0](self.width), FORMATS[fmt][1](self.width, self.height)
            f = pack_format(BUF_TYPE_OUTPUT, self.width, self.height, fmt, bpl, size)
            self._ioctl(self.fd, VIDIOC_S_FMT, f, True)
            got = unpack_format(f)
            if (got['width'], got['height'], got['format']) != (self.width, self.height, fmt):
                raise V4L2Error(f'{device} took {got["width"]} × {got["height"]} {got["format"]} instead of '
                                f'{self.width} × {self.height} {fmt}')
            self.format = got
            self.sizeimage = got['sizeimage'] or size
            self.bytesperline = got['bytesperline'] or bpl
        except OSError as e:
            self.close()
            raise V4L2Error(f'{device}: {e}') from e
        except V4L2Error:
            self.close()
            raise
        self.frames = 0
        self.short_writes = 0

    def write(self, img, n: int) -> dict:
        """One frame (BGR or BGRA): converted, the barcode painted when on, written whole. Its stamps (monotonic ns)."""
        t0 = time.monotonic_ns()
        a = np.asarray(img)
        if a.shape[0] != self.height or a.shape[1] != self.width:
            raise V4L2Error(f'a {a.shape[1]} × {a.shape[0]} frame for a {self.width} × {self.height} device')
        if self.barcode:
            a = paint_barcode(np.array(a[:, :, :3]), n)
        data = convert(a, self.fmt)
        if self.bytesperline != FORMATS[self.fmt][0](self.width) and self.fmt != 'NV12':
            rows = np.frombuffer(data, np.uint8).reshape(self.height, -1)
            pad = np.zeros((self.height, self.bytesperline), np.uint8)
            pad[:, :rows.shape[1]] = rows
            data = pad.tobytes()
        data = data.ljust(self.sizeimage, b'\x00')[:self.sizeimage]
        t1 = time.monotonic_ns()
        try:
            k = self._write(self.fd, data)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                k = 0
            else:
                raise V4L2Error(f'writing to {self.device}: {e}') from e
        t2 = time.monotonic_ns()
        if k != len(data):
            self.short_writes += 1
        self.frames += 1
        return {'enc_start': t0, 'enc_end': t1, 'cam_sent': t1, 'cam_sent_end': t2, 'cam_bytes': len(data),
                'written': k}

    def close(self):
        fd, self.fd = getattr(self, 'fd', None), None
        if fd is not None:
            try:
                self._close(fd)
            except OSError:
                pass

    def describe(self) -> dict:
        return {'method': 'ioctl+write', 'device': self.device, 'format': self.fmt, 'width': self.width,
                'height': self.height, 'sizeimage': self.sizeimage, 'card': self.cap.get('card'),
                'driver': self.cap.get('driver'), 'barcode': self.barcode}


class GstV4L2Writer:
    """The same through GStreamer: appsrc (BGRx) ! videoconvert ! the format ! v4l2sink (no clock sync)."""

    GST_FORMATS = {'YUYV': 'YUY2', 'UYVY': 'UYVY', 'RGB3': 'RGB', 'NV12': 'NV12'}

    def __init__(self, device: str, width: int, height: int, fmt: str = 'YUYV', barcode: bool = False,
                 fps: float = 20.0):
        try:
            from .device_agent import gst
        except ImportError:                                    # beside the agent on the device
            from device_agent import gst
        G = gst()
        if G is None:
            raise V4L2Error('GStreamer\'s Python bindings are not importable here: use the ioctl writer')
        if G.ElementFactory.find('v4l2sink') is None:
            raise V4L2Error('the GStreamer element v4l2sink is missing (gst-plugins-good\'s video4linux2)')
        if fmt not in self.GST_FORMATS:
            raise V4L2Error(f'pixel format {fmt} is not one of {", ".join(self.GST_FORMATS)}')
        self.G, self.device, self.width, self.height, self.fmt, self.barcode = G, device, int(width), int(height), \
            fmt, barcode
        num = int(round(float(fps) * 1000))
        self.pipe = G.parse_launch(
            f'appsrc name=src is-live=true format=time do-timestamp=true block=false '
            f'caps=video/x-raw,format=BGRx,width={self.width},height={self.height},framerate={num}/1000 ! '
            f'videoconvert ! video/x-raw,format={self.GST_FORMATS[fmt]} ! v4l2sink device={device} sync=false '
            'async=false')
        self.src = self.pipe.get_by_name('src')
        if self.pipe.set_state(G.State.PLAYING) == G.StateChangeReturn.FAILURE:
            raise V4L2Error(f'v4l2sink could not open {device}')
        self.frames = 0

    def write(self, img, n: int) -> dict:
        t0 = time.monotonic_ns()
        a = np.asarray(img)
        if a.shape[2] == 3:
            a = np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
        if self.barcode:
            a = paint_barcode(np.array(a), n)
        buf = self.G.Buffer.new_wrapped(np.ascontiguousarray(a).tobytes())
        t1 = time.monotonic_ns()
        self.src.emit('push-buffer', buf)
        self.frames += 1
        return {'enc_start': t0, 'enc_end': t1, 'cam_sent': t1, 'cam_sent_end': time.monotonic_ns(),
                'cam_bytes': a.nbytes}

    def close(self):
        self.pipe.set_state(self.G.State.NULL)

    def describe(self) -> dict:
        return {'method': 'gstreamer v4l2sink', 'device': self.device, 'format': self.fmt, 'width': self.width,
                'height': self.height, 'barcode': self.barcode}


class StampSender:
    """The sidecar stamp stream: one UDP datagram per frame (module doc)."""

    def __init__(self, host: str, port: int):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.n = 0
        self.errors = 0

    def send(self, frame: int, sim_t: float, hw_t: int, written_ns: int):
        self.n += 1
        msg = json.dumps({'f': int(frame), 's': sim_t, 'h': hw_t, 'w': int(written_ns), 'n': self.n},
                         separators=(',', ':')).encode()
        try:
            self.sock.sendto(msg, self.addr)
        except OSError:
            self.errors += 1

    def close(self):
        self.sock.close()


def open_writer(cfg: dict, fps: float = 20.0):
    """The writer a session's camera config asks for: {device, format, method: ioctl | gstreamer, barcode}."""
    args = (cfg['device'], int(cfg['width']), int(cfg['height']), cfg.get('format', 'YUYV'), bool(cfg.get('barcode')))
    if cfg.get('method') == 'gstreamer':
        return GstV4L2Writer(*args, fps=fps)
    return V4L2Writer(*args)


# ---------------------------------------------------------------------------------------------- what is here
def devices() -> list:
    """The V4L2 nodes of this machine: [{device, name, loopback}] (sysfs; nothing is opened)."""
    out = []
    try:
        names = sorted(os.listdir('/sys/class/video4linux'), key=lambda s: (len(s), s))
    except OSError:
        return out
    for n in names:
        try:
            with open(f'/sys/class/video4linux/{n}/name', encoding='utf-8') as fh:
                name = fh.read().strip()
        except OSError:
            name = ''
        virtual = os.path.realpath(f'/sys/class/video4linux/{n}').startswith('/sys/devices/virtual/')
        out.append({'device': f'/dev/{n}', 'name': name,
                    'loopback': virtual and os.path.isdir('/sys/module/v4l2loopback')})
    return out


def doctor_rows() -> list:
    """(name, ok, detail) for `falcon doctor`: v4l2loopback and Falcon's two nodes (informational)."""
    loaded = os.path.isdir('/sys/module/v4l2loopback')
    devs = {d['device']: d for d in devices()}
    rows = [('v4l2loopback', loaded, 'loaded' if loaded else 'not loaded (optional: falcon hil setup prints how)')]
    for n in (40, 41):
        d = devs.get(f'/dev/video{n}')
        rows.append((f'/dev/video{n}', d is not None,
                     f'{d["name"] or "?"}' if d else 'absent (optional: the V4L2 camera output)'))
    return rows
