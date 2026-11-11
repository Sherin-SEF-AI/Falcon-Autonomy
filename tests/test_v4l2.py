"""The camera as a V4L2 device (hil.v4l2out, hil.gmsl): the kernel ABI's structure sizes and ioctl numbers; BT.601
conversion to and from YUYV, UYVY, RGB3 and NV12; the frame-number barcode surviving every conversion; the writer
(QUERYCAP, S_FMT, one write() per frame) and the mmap-streaming capture grabber (REQBUFS, QUERYBUF, mmap, QBUF,
STREAMON, DQBUF with its timestamp and flags) against a stand-in kernel — and the two joined through a stand-in
loopback; the fake GMSL injector; the sidecar stamp stream. On a real v4l2loopback node (marker v4l2, FALCON_V4L2=1)
the writer and the grabber run against the kernel."""
import json
import os
import socket
import time

import numpy as np
import pytest

from falcon_autonomy.hil import gmsl as G
from falcon_autonomy.hil import v4l2out as V


def _image(w=320, h=180, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    base[:, :, 1] = np.linspace(40, 200, h)[:, None]
    return np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)


def test_abi_sizes_and_ioctl_numbers():
    assert (V.CAP.size, V.PIX.size, V.REQBUFS.size, V.BUFFER.size, V.FMT_SIZE) == (104, 48, 20, 88, 208)
    # the numbers videodev2.h gives on 64-bit Linux
    assert V.VIDIOC_QUERYCAP == 0x80685600 and V.VIDIOC_G_FMT == 0xC0D05604 and V.VIDIOC_S_FMT == 0xC0D05605
    assert V.VIDIOC_REQBUFS == 0xC0145608 and V.VIDIOC_QUERYBUF == 0xC0585609
    assert V.VIDIOC_QBUF == 0xC058560F and V.VIDIOC_DQBUF == 0xC0585611
    assert V.VIDIOC_STREAMON == 0x40045612 and V.VIDIOC_STREAMOFF == 0x40045613
    assert V.fourcc('YUYV') == 0x56595559 and V.fourcc_str(V.fourcc('NV12')) == 'NV12'
    f = V.pack_format(V.BUF_TYPE_OUTPUT, 640, 360, 'YUYV', 1280, 460800)
    assert V.unpack_format(f) == {'type': 2, 'width': 640, 'height': 360, 'format': 'YUYV', 'field': 1,
                                  'bytesperline': 1280, 'sizeimage': 460800, 'colorspace': V.COLORSPACE_SMPTE170M}


@pytest.mark.parametrize('fmt,tol', [('RGB3', 0), ('YUYV', 12), ('UYVY', 12), ('NV12', 14)])
def test_conversions_round_trip(fmt, tol):
    img = _image()
    data = V.convert(img, fmt)
    assert len(data) == V.FORMATS[fmt][1](320, 180)
    back = V.to_bgr(data, 320, 180, fmt)
    assert back.shape == img.shape
    err = np.abs(back.astype(int) - img.astype(int))
    assert np.median(err) <= max(tol / 4, 0) + 1 and np.percentile(err, 99) <= tol + 1, (fmt, err.max())
    bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    assert V.convert(bgra, fmt) == data                                # BGRA in: the alpha is dropped
    with pytest.raises(ValueError):
        V.convert(img[:, :319], 'YUYV')                                # odd width


def test_grey_and_bt601_levels():
    white = np.full((2, 2, 3), 255, np.uint8)
    black = np.zeros((2, 2, 3), np.uint8)
    assert V.convert(white, 'YUYV') == bytes([235, 128, 235, 128]) * 2       # limited range: Y 16–235
    assert V.convert(black, 'YUYV') == bytes([16, 128, 16, 128]) * 2
    assert V.convert(white, 'GREY') == bytes([235]) * 4


@pytest.mark.parametrize('fmt', ['RGB3', 'YUYV', 'UYVY', 'NV12'])
def test_barcode_survives_every_format(fmt):
    for n in (0, 1, 123456789, 0xFFFFFFFF):
        img = V.paint_barcode(_image(seed=n % 7).copy(), n)
        assert V.read_barcode(V.to_bgr(V.convert(img, fmt), 320, 180, fmt)) == n
    assert V.read_barcode(_image()) is None                            # no start pattern
    img = V.paint_barcode(_image().copy(), 77)
    cell = 320 // V.BARCODE_BITS
    img[:2, 40 * cell:41 * cell] = 255 - img[:2, 40 * cell:41 * cell]  # one bit of the check flipped
    assert V.read_barcode(img) is None
    with pytest.raises(ValueError):
        V.paint_barcode(np.zeros((4, 30, 3), np.uint8), 1)


# ============================================================================================ a stand-in kernel
class FakeMap(bytearray):
    def close(self):
        self.closed = True


class Kernel:
    """Enough of V4L2 for one output node and one capture node joined like v4l2loopback: write() on the output
    queues a frame; the capture side's DQBUF delivers it into a queued buffer with a monotonic timestamp."""

    def __init__(self, caps_out=V.CAP_VIDEO_OUTPUT, caps_cap=V.CAP_VIDEO_CAPTURE | V.CAP_STREAMING, adjust=None):
        self.fds = {}
        self.calls = []
        self.frames = []
        self.caps = {'/dev/video40': caps_out, '/dev/video41': caps_cap}
        self.fmt = None
        self.adjust = adjust
        self.queued = []
        self.maps = {}
        self.length = 0
        self.seq = 0
        self.streaming = False
        self.closed = []

    def open(self, path, flags):
        if path not in self.caps:
            raise FileNotFoundError(2, 'No such file or directory')
        fd = 100 + len(self.fds)
        self.fds[fd] = path
        return fd

    def close(self, fd):
        self.closed.append(fd)

    def ioctl(self, fd, req, buf, mutate=True):
        self.calls.append(req)
        if req == V.VIDIOC_QUERYCAP:
            c = self.caps[self.fds[fd]]
            V.CAP.pack_into(buf, 0, b'v4l2 loopback', b'falcon', b'platform:v4l2loopback-000', 0x060800,
                            c | V.CAP_DEVICE_CAPS, c)
        elif req == V.VIDIOC_S_FMT:
            f = V.unpack_format(buf)
            if self.adjust:
                f = dict(f, **self.adjust)
            size = V.FORMATS[f['format']][1](f['width'], f['height'])
            self.fmt = dict(f, sizeimage=size, bytesperline=V.FORMATS[f['format']][0](f['width']))
            buf[:] = V.pack_format(f['type'], f['width'], f['height'], f['format'], self.fmt['bytesperline'], size)
        elif req == V.VIDIOC_G_FMT:
            f = self.fmt
            buf[:] = V.pack_format(V.BUF_TYPE_CAPTURE, f['width'], f['height'], f['format'], f['bytesperline'],
                                   f['sizeimage'])
        elif req == V.VIDIOC_REQBUFS:
            n, t, mem, *_ = V.REQBUFS.unpack(buf)
            self.length = self.fmt['sizeimage']
            buf[:] = V.REQBUFS.pack(min(n, 3), t, mem, 0, 0)
        elif req == V.VIDIOC_QUERYBUF:
            f = list(V.BUFFER.unpack(buf))
            f[10], f[11] = f[0] * 4096 * 1024, self.length
            buf[:] = V.BUFFER.pack(*f)
        elif req == V.VIDIOC_QBUF:
            self.queued.append(V.BUFFER.unpack(buf)[0])
        elif req == V.VIDIOC_DQBUF:
            if not self.frames or not self.queued:
                raise BlockingIOError(11, 'try again')
            data, t = self.frames.pop(0)
            idx = self.queued.pop(0)
            self.maps[idx][:len(data)] = data
            self.seq += 1
            f = list(V.BUFFER.unpack(buf))
            f[0], f[2], f[3] = idx, len(data), V.BUF_FLAG_TIMESTAMP_MONOTONIC
            f[5], f[6], f[8] = t // 1_000_000_000, (t % 1_000_000_000) // 1000, self.seq
            buf[:] = V.BUFFER.pack(*f)
        elif req == V.VIDIOC_STREAMON:
            self.streaming = True
        elif req == V.VIDIOC_STREAMOFF:
            self.streaming = False
        else:
            raise OSError(25, 'Inappropriate ioctl')
        return 0

    def write(self, fd, data):
        self.frames.append((bytes(data), time.monotonic_ns()))
        return len(data)

    def mmap(self, fd, length, offset):
        m = FakeMap(length)
        self.maps[offset // (4096 * 1024)] = m
        return m

    def select(self, r, w, x, timeout):
        return (r if self.frames else []), [], []


def _writer(k, **kw):
    return V.V4L2Writer('/dev/video40', kw.pop('w', 320), kw.pop('h', 180), kw.pop('fmt', 'YUYV'), opener=k.open,
                        ioctl=k.ioctl, write=k.write, close=k.close, **kw)


def _grabber(k, **kw):
    return G.V4L2CaptureGrabber('/dev/video41', opener=k.open, ioctl=k.ioctl, mapper=k.mmap, selector=k.select,
                                close=k.close, **kw)


def test_writer_against_a_stand_in_kernel():
    k = Kernel()
    w = _writer(k, fmt='NV12', barcode=True)
    assert k.calls[:2] == [V.VIDIOC_QUERYCAP, V.VIDIOC_S_FMT] and w.sizeimage == 320 * 180 * 3 // 2
    assert w.describe()['card'] == 'falcon' and w.describe()['driver'] == 'v4l2 loopback'
    st = w.write(np.dstack([_image(), np.zeros((180, 320), np.uint8)]), 4242)
    assert st['written'] == w.sizeimage and st['enc_start'] <= st['enc_end'] <= st['cam_sent_end']
    data, _ = k.frames[0]
    assert V.read_barcode(V.to_bgr(data, 320, 180, 'NV12')) == 4242
    with pytest.raises(V.V4L2Error):
        w.write(_image(160, 90), 1)                                   # the wrong size
    w.close()
    assert k.closed
    with pytest.raises(V.V4L2Error, match='not a V4L2 output device'):
        _writer(Kernel(caps_out=V.CAP_VIDEO_CAPTURE))
    with pytest.raises(V.V4L2Error, match='took 640'):
        _writer(Kernel(adjust={'width': 640}))
    with pytest.raises(V.V4L2Error, match='cannot open'):
        V.V4L2Writer('/dev/video99', 320, 180, opener=Kernel().open, ioctl=Kernel().ioctl)
    with pytest.raises(V.V4L2Error):
        V.V4L2Writer('/dev/video40', 320, 180, 'MJPG')


def test_writer_then_grabber_through_a_stand_in_loopback():
    k = Kernel()
    w = _writer(k, fmt='YUYV', barcode=True)
    g = _grabber(k).open()
    assert g.describe()['format'] == 'YUYV' and g.describe()['buffers'] == 3 and k.streaming
    assert V.VIDIOC_REQBUFS in k.calls and V.VIDIOC_STREAMON in k.calls
    sent = {}
    for n in range(5):
        sent[n] = w.write(_image(seed=n), 1000 + n)['cam_sent']
    for n in range(5):
        got = g.grab(0.1)
        assert got.barcode == 1000 + n and got.sequence == n + 1 and got.image.shape == (180, 320, 3)
        assert got.clock == 'monotonic' and got.source == 'end_of_frame'
        assert sent[n] <= got.t_buffer_ns + 1000 <= got.t_dequeued_ns + 1000 <= got.t_converted_ns + 1000
    assert g.grab(0.01) is None                                       # nothing waiting
    g.close()
    assert not k.streaming and all(getattr(m, 'closed', False) for m in k.maps.values())
    with pytest.raises(V.V4L2Error, match='not a single-plane'):
        _grabber(Kernel(caps_cap=V.CAP_VIDEO_OUTPUT)).open()
    k2 = Kernel(caps_cap=V.CAP_VIDEO_CAPTURE)                          # no streaming I/O
    with pytest.raises(V.V4L2Error, match='streaming'):
        _grabber(k2).open()


def test_fake_injector_delivers_after_its_latency():
    inj = G.FakeInjector(latency_s=0.02)
    g = inj.grabber().open()
    t0 = time.monotonic_ns()
    inj.inject(_image(), 7)
    got = g.grab(1.0)
    assert got.barcode == 7 and got.t_dequeued_ns - t0 >= 19_000_000 and got.image.shape == (180, 320, 3)
    assert g.grab(0.01) is None
    for k in range(6):                                                 # a full queue drops the oldest
        inj.inject(_image(), k)
    assert inj.dropped == 2 and g.grab(0.5).barcode == 2
    with pytest.raises(NotImplementedError):
        G.Injector().inject(None, 0)


def test_sidecar_stamps_over_udp():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(('127.0.0.1', 0))
    rx.settimeout(2.0)
    tx = V.StampSender('127.0.0.1', rx.getsockname()[1])
    tx.send(12, 1.25, 1250000000, 999)
    d = json.loads(rx.recv(512))
    assert d == {'f': 12, 's': 1.25, 'h': 1250000000, 'w': 999, 'n': 1}
    tx.close()
    rx.close()


def test_session_config_for_a_v4l2_camera(project, tmp_path):
    from falcon_autonomy.core import hil as H
    cfg = H.hil_config(project, 'world-town10', 'rig-default', str(tmp_path), 2310, 8310, camera='v4l2', width=320,
                       height=180, v4l2={'device': '/dev/video40', 'format': 'NV12'})
    v = cfg['camera']['v4l2']
    assert v['format'] == 'NV12' and v['barcode'] and v['capture_device'] == '/dev/video40' and v['width'] == 320
    for bad in ({'format': 'MJPG'}, {'device': '/tmp/x'}, {'method': 'magic'}):
        with pytest.raises(ValueError):
            H.hil_config(project, 'world-town10', 'rig-default', str(tmp_path), 2310, 8310, camera='v4l2', width=320,
                         height=180, v4l2=bad)
    with pytest.raises(ValueError):
        H.hil_config(project, 'world-town10', 'rig-default', str(tmp_path), 2310, 8310, v4l2={'format': 'YUYV'})
    remote = H.hil_config(project, 'world-town10', 'rig-default', str(tmp_path), 2310, 8310, camera='v4l2',
                          width=320, height=180, device='remote')
    assert 'capture_device' not in remote['camera']['v4l2']           # the device reads its own (--camera-device)


# ============================================================================================ real devices (opt-in)
def _loopback_pair():
    devs = {d['device']: d for d in V.devices()}
    return all(p in devs for p in ('/dev/video40',)) and os.access('/dev/video40', os.W_OK)


@pytest.mark.v4l2
def test_real_v4l2loopback_write_and_capture():
    """falcon hil setup's v4l2loopback (exclusive_caps=1): the writer on /dev/video40, the grabber on the same node
    once a writer has set its format; the barcode and the latency write → buffer timestamp."""
    if not _loopback_pair():
        pytest.skip('/dev/video40 (v4l2loopback) is not there or not writable')
    w = V.V4L2Writer('/dev/video40', 640, 360, 'YUYV', barcode=True)
    w.write(_image(640, 360), 0)
    g = G.V4L2CaptureGrabber('/dev/video40').open()
    lat = []
    try:
        for n in range(1, 40):
            t = w.write(_image(640, 360, n), n)['cam_sent']
            got = g.grab(1.0)
            assert got is not None and got.barcode is not None
            if got.t_buffer_ns:
                lat.append((got.t_buffer_ns - t) / 1e6)
            time.sleep(0.02)
    finally:
        g.close()
        w.close()
    assert lat and np.median(lat) < 20.0


@pytest.mark.v4l2
def test_session_with_the_camera_through_v4l2loopback(project, tmp_path):
    """A whole session: the camera written into /dev/video40, the simulated device reading it back as a camera,
    frames matched by their barcode."""
    import subprocess
    import sys
    from conftest import worker_env
    from falcon_autonomy.core import hil as H
    from falcon_autonomy.core import protocol as P
    if not _loopback_pair():
        pytest.skip('/dev/video40 (v4l2loopback) is not there or not writable')
    out = str(tmp_path / 'hil')
    os.makedirs(out)
    cfg = H.hil_config(project, 'world-town10', 'rig-default', out, 2310, 8310, seconds=3.0, fps=10.0, width=320,
                       height=180, camera='v4l2', lidar_overrides={'points_per_second': 20000, 'channels': 16},
                       warmup_steps=3, gateway_python=sys.executable)
    path = os.path.join(out, 'hil.json')
    json.dump(cfg, open(path, 'w'))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'hil', path], env=worker_env(),
                       capture_output=True, text=True, timeout=150)
    assert r.returncode == P.OK, r.stderr[-3000:]
    h = json.load(open(os.path.join(out, 'run.json')))['hil']
    assert h['drops']['camera_received'] >= 20 and h['stages']['uplink']['n'] >= 20


@pytest.mark.orin
def test_gmsl_capture_on_the_orin():
    """On the Jetson with a GMSL2 camera (FALCON_ORIN=1, FALCON_GMSL_DEVICE=/dev/videoN): frames with driver
    timestamps on CLOCK_MONOTONIC."""
    dev = os.environ.get('FALCON_GMSL_DEVICE', '')
    if not dev or not os.path.exists(dev):
        pytest.skip('FALCON_GMSL_DEVICE names no device here')
    g = G.V4L2CaptureGrabber(dev, read_barcode=False).open()
    try:
        frames = [g.grab(2.0) for _ in range(10)]
    finally:
        g.close()
    frames = [f for f in frames if f is not None]
    assert len(frames) >= 8 and all(f.clock == 'monotonic' for f in frames)
    seq = [f.sequence for f in frames]
    assert seq == sorted(seq)
