"""The HIL gateway: between the session worker (CARLA's Python, on this machine) and the device agent (on the device,
or a simulated one started here). Runs in the app's Python, which has GStreamer's bindings; CARLA's venv does not.

    python -m falcon_autonomy.hil.gateway --fd N --config RUN/hil.json

The worker starts it with one end of a socket pair (fd N) and talks to it in the live link's framing
(falcon_autonomy.hil.device_agent.pack). Worker → gateway: {"t": "frame", "frame", "sim_t", "speed_mps", "cam": {w, h,
hw_t, bytes}, "lidar": {n, hw_t, bytes}} with the camera's BGRA pixels then the lidar's float32 x y z i as payload;
{"t": "bye"}. Gateway → worker: {"t": "ready", "device": env, ...}, {"t": "gw", "frame", stamps} per frame once both
streams are sent, {"t": "control", ...} as the device's controls arrive (with ctrl_rx), {"t": "clock", "estimate"}
once a second, {"t": "log"|"error", "text"}, and {"t": "done"} after it wrote its files (device.json, clock.jsonl,
gateway.json) into the run folder.

Device side (device_agent's wire format): the session on TCP port P, clock pings on UDP P, the camera as H.264 over
RTP to UDP P + 1 (or raw BGR over a TCP stream), the lidar over a TCP stream (or UDP P + 2 in chunks). Encoders: NVENC
(nvh264enc: preset p1, tune ultra-low-latency, zero latency, no B-frames, CBR) or x264 (x264enc: tune zerolatency,
speed-preset ultrafast, no B-frames, sliced threads); an IDR every gop frames with SPS/PPS repeated, so a receiver
recovers within a second of a loss. Each sender keeps only the newest frame waiting: when the network or the encoder
falls behind, an older frame is replaced and counted ("dropped by the gateway"), never queued up.

Local mode ("simulated device"): the gateway starts device_agent.py in this Python on 127.0.0.1 on free ports, with
the config's emulated clock (offset, drift) and decoder, its log in the run folder, and stops it at the end. Remote
mode: the address comes from FALCON_HIL_ADDRESS (the user's settings, never a file) and the agent must be running
there (falcon hil deploy --start). Messages never name the address.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time

from . import device_agent as A

READY_TIMEOUT_S = 30.0
CONNECT_TIMEOUT_S = 15.0


def now() -> int:
    return time.monotonic_ns()


def free_base_port(host='127.0.0.1') -> int:
    """A port P with TCP P and UDP P, P + 1, P + 2 free (the agent's session, camera and lidar ports)."""
    rnd = random.SystemRandom()
    for _ in range(200):
        p = rnd.randint(20000, 60000)
        socks = []
        try:
            t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socks.append(t)
            t.bind((host, p))
            for k in range(3):
                u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                socks.append(u)
                u.bind((host, p + k))
            return p
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError('no free ports for the simulated device')


class Slot:
    """The newest item waiting for a sender thread; put() replaces one not yet taken (counted as dropped)."""

    def __init__(self):
        self.cv = threading.Condition()
        self.item = None
        self.closed = False
        self.dropped = 0

    def put(self, item):
        with self.cv:
            if self.item is not None:
                self.dropped += 1
            self.item = item
            self.cv.notify()

    def take(self):
        with self.cv:
            while self.item is None and not self.closed:
                self.cv.wait(0.2)
            it, self.item = self.item, None
            return it

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()


class H264Sender:
    """BGRx frames → encoder → access units (SEI with the step's frame, sim_t, hw_t inserted) → RTP/UDP."""

    def __init__(self, gw, w, h, fps, encoder, bitrate_kbps, gop, host, port):
        G = A.gst()
        if G is None:
            raise RuntimeError(f'the camera is H.264 but GStreamer\'s Python bindings are not importable in '
                               f'{sys.executable}: choose the raw transport, or install python3-gi')
        self.G, self.gw = G, gw
        if encoder == 'auto':
            encoder = 'nvh264enc' if G.ElementFactory.find('nvh264enc') is not None else 'x264enc'
        if G.ElementFactory.find(encoder) is None:
            raise RuntimeError(f'the GStreamer element {encoder} is not available')
        self.encoder = encoder
        num = int(round(float(fps) * 1000))
        caps = f'video/x-raw,format=BGRx,width={int(w)},height={int(h)},framerate={num}/1000'
        conv = '' if encoder == 'nvh264enc' else 'videoconvert ! video/x-raw,format=I420 ! '
        self.enc = G.parse_launch(
            f'appsrc name=src is-live=true format=time do-timestamp=false block=false caps={caps} ! {conv}'
            f'{encoder} name=enc ! h264parse config-interval=-1 ! '
            'video/x-h264,stream-format=byte-stream,alignment=au ! appsink name=sink sync=false emit-signals=false '
            'max-buffers=0 drop=false')
        e = self.enc.get_by_name('enc')
        props = ({'preset': 'p1', 'tune': 'ultra-low-latency', 'zerolatency': 'true', 'bframes': '0',
                  'rc-mode': 'cbr', 'bitrate': str(int(bitrate_kbps)), 'gop-size': str(int(gop))}
                 if encoder == 'nvh264enc' else
                 {'tune': 'zerolatency', 'speed-preset': 'ultrafast', 'bframes': '0', 'bitrate': str(int(bitrate_kbps)),
                  'key-int-max': str(int(gop)), 'byte-stream': 'true', 'sliced-threads': 'true', 'threads': '4'})
        self.props = {}
        for k, v in props.items():
            if e.find_property(k) is not None:
                G.util_set_object_arg(e, k, v)
                self.props[k] = v
        self.rtp = G.parse_launch(
            'appsrc name=src is-live=true format=time do-timestamp=false block=false '
            'caps=video/x-h264,stream-format=byte-stream,alignment=au ! rtph264pay name=pay pt=96 mtu=1400 '
            f'config-interval=-1 ! udpsink host={host} port={int(port)} sync=false async=false')
        pay = self.rtp.get_by_name('pay')
        if pay.find_property('aggregate-mode') is not None:
            G.util_set_object_arg(pay, 'aggregate-mode', 'zero-latency')
        self.src = self.enc.get_by_name('src')
        self.rsrc = self.rtp.get_by_name('src')
        self.period = int(round(1e9 / float(fps)))
        self.inflight: dict = {}                          # pts → (frame, sei meta)
        self.lock = threading.Lock()
        self.f0 = None
        self.delta = None
        for p in (self.rtp, self.enc):
            if p.set_state(G.State.PLAYING) == G.StateChangeReturn.FAILURE:
                raise RuntimeError(f'the {encoder} pipeline did not start')
        self.thread = threading.Thread(target=self._pull, daemon=True, name='gw-encoded')
        self.thread.start()

    def push(self, frame: int, bgrx: bytes, meta: dict):
        if self.f0 is None:
            self.f0 = frame
        pts = (frame - self.f0) * self.period
        buf = self.G.Buffer.new_wrapped(bgrx)
        buf.pts = pts
        buf.duration = self.period
        with self.lock:
            self.inflight[pts] = (frame, meta)
        self.gw.stamp(frame, enc_start=now())
        self.src.emit('push-buffer', buf)

    def _pull(self):
        G, gw = self.G, self.gw
        sink = self.enc.get_by_name('sink')
        while not gw.stopping.is_set():
            s = sink.emit('try-pull-sample', 100_000_000)
            if s is None:
                msg = self.enc.get_bus().timed_pop_filtered(0, G.MessageType.ERROR)
                if msg is not None:
                    gw.error(f'the encoder failed: {msg.parse_error()[0].message}')
                continue
            t_enc = now()
            buf = s.get_buffer()
            with self.lock:
                if self.delta is None and self.inflight:     # encoders may shift timestamps (GstVideoEncoder's
                    self.delta = buf.pts - min(self.inflight)  # offset for negative DTS): learnt on the first unit
                key = buf.pts - (self.delta or 0)
                item = self.inflight.pop(key, None)
                for k in [k for k in self.inflight if k < key]:
                    self.inflight.pop(k)
                    gw.counts['encoder_dropped'] = gw.counts.get('encoder_dropped', 0) + 1
            if item is None:
                gw.counts['encoder_unmatched'] = gw.counts.get('encoder_unmatched', 0) + 1
                continue
            frame, meta = item
            au = A.insert_sei(buf.extract_dup(0, buf.get_size()), meta)
            out = G.Buffer.new_wrapped(au)
            out.pts, out.duration = key, self.period
            t_sent = now()
            self.rsrc.emit('push-buffer', out)
            gw.stamp(frame, enc_end=t_enc, cam_sent=t_sent, cam_sent_end=now(), cam_bytes=len(au), cam_done=True)

    def stop(self):
        for p in (self.enc, self.rtp):
            p.set_state(self.G.State.NULL)
        self.thread.join(2.0)


class Gateway:
    def __init__(self, cfg: dict, wsock, address: str = ''):
        self.cfg = cfg
        self.out = cfg['out']
        self.ws = wsock
        self.wlock = threading.Lock()
        self.address = address
        self.local = cfg['device']['mode'] == 'local'
        self.port = int(cfg['device']['port'])
        self.stopping = threading.Event()
        self.agent_proc = None
        self.sess = None
        self.slock = threading.Lock()
        self.frames: dict = {}                            # frame → gateway stamps until both streams are sent
        self.flock = threading.Lock()
        self.counts: dict = {}
        self.samples: list = []                           # clock pings (t0, t1, t2, t3)
        self.device: dict = {}
        self.stats = None
        self.stats_ev = threading.Event()
        self.video = None
        self.cam_slot, self.lidar_slot = Slot(), Slot()
        self.cam_sock = self.lidar_sock = self.lidar_udp = None
        self.threads: list = []

    # ---------------------------------------------------------------- to the worker
    def to_worker(self, header: dict, payload=b''):
        data = A.pack(header, payload)
        with self.wlock:
            try:
                self.ws.sendall(data)
            except OSError:
                self.stopping.set()

    def log(self, text, level='info'):
        self.to_worker({'t': 'log', 'level': level, 'text': text})

    def error(self, text):
        self.to_worker({'t': 'error', 'text': text})

    def stamp(self, frame: int, **kw):
        """Gateway stamps of a frame; once the camera and the lidar are both sent they go to the worker."""
        with self.flock:
            st = self.frames.setdefault(frame, {})
            st.update(kw)
            done = st.get('cam_done') and st.get('lidar_done')
            if done:
                self.frames.pop(frame)
        if done:
            st.pop('cam_done')
            st.pop('lidar_done')
            self.to_worker(dict(st, t='gw', frame=frame))

    # ---------------------------------------------------------------- the device
    def start_local_agent(self):
        self.port = free_base_port()
        sd = self.cfg.get('sim_device') or {}
        cmd = [self.cfg.get('gateway_python') or sys.executable, os.path.abspath(A.__file__), '--listen',
               '127.0.0.1', '--port', str(self.port), '--once', '--clock-offset-ms',
               str(float(sd.get('clock_offset_ms', 0.0))), '--clock-drift-ppm',
               str(float(sd.get('clock_drift_ppm', 0.0))),
               '--decoder', str(sd.get('decoder', 'auto'))]
        st = self.cfg.get('stack') or {}
        if st.get('name'):
            cmd += ['--stack', st['name']]
        env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
        log = open(os.path.join(self.out, 'device-agent.log'), 'w', encoding='utf-8')
        self.agent_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=False)
        log.close()
        self.address = '127.0.0.1'

    def connect(self) -> socket.socket:
        t0 = time.monotonic()
        last = None
        while time.monotonic() - t0 < CONNECT_TIMEOUT_S:
            if self.agent_proc is not None and self.agent_proc.poll() is not None:
                raise RuntimeError(f'the simulated device exited at once (code {self.agent_proc.returncode}); see '
                                   'device-agent.log in the run folder')
            try:
                s = socket.create_connection((self.address, self.port), 2.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(None)
                return s
            except OSError as e:
                last = e
                time.sleep(0.1)
        raise RuntimeError(f'no device agent answered on port {self.port} within {CONNECT_TIMEOUT_S:.0f} s ({last}); '
                           'is it running? (falcon hil deploy --start)')

    def hello(self):
        cam, lid = self.cfg['camera'], self.cfg['lidar']
        cs, ls = cam['sensor'], lid['sensor']
        session = {
            'period_ns': int(round(1e9 / float(self.cfg['fps']))), 'fps': float(self.cfg['fps']),
            'sync_timeout_ms': float(self.cfg['sync_timeout_ms']),
            'camera': {'transport': cam['transport'], 'width': int(cs['width']), 'height': int(cs['height']),
                       'rtp_port': self.port + 1, 'fov': float(cs['fov'])},
            'lidar': {'transport': lid['transport'], 'udp_port': self.port + 2, 'encoding': lid['encoding'],
                      'ros2_topic': lid.get('ros2_topic') or '', 'channels': int(ls['channels']),
                      'rotation_frequency': float(ls['rotation_frequency']), 'pose': ls['pose']},
            'stack': {'params': (self.cfg.get('stack') or {}).get('params') or {}},
        }
        self.sess = self.connect()
        self.sess.sendall(A.pack({'t': 'hello', 'protocol': A.PROTOCOL, 'session': session}))
        self.sess.settimeout(READY_TIMEOUT_S)
        try:
            h, _ = A.read_msg(self.sess)
        except socket.timeout as e:
            raise RuntimeError(f'the device agent did not get ready within {READY_TIMEOUT_S:.0f} s') from e
        self.sess.settimeout(None)
        if h.get('t') == 'error':
            raise RuntimeError(f'the device agent refused the session: {h.get("error")}')
        if h.get('t') != 'ready':
            raise RuntimeError(f'the device agent answered {h.get("t")!r} instead of ready')
        self.device = h

    def open_streams(self):
        cam, lid = self.cfg['camera'], self.cfg['lidar']
        cs = cam['sensor']
        if cam['transport'] == 'h264-rtp':
            self.video = H264Sender(self, cs['width'], cs['height'], self.cfg['fps'], cam['encoder'],
                                    cam['bitrate_kbps'], cam['gop'], self.address, self.port + 1)
        else:
            self.cam_sock = self.connect()
            self.cam_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
            self.cam_sock.sendall(A.pack({'t': 'stream', 'kind': 'camera'}))
        if lid['transport'] == 'tcp':
            self.lidar_sock = self.connect()
            self.lidar_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
            self.lidar_sock.sendall(A.pack({'t': 'stream', 'kind': 'lidar'}))
        else:
            self.lidar_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.lidar_udp.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
            self.lidar_udp.connect((self.address, self.port + 2))

    # ---------------------------------------------------------------- threads
    def _start(self, fn, name):
        t = threading.Thread(target=fn, daemon=True, name=name)
        t.start()
        self.threads.append(t)

    def ping_once(self, u, seq) -> bool:
        t0 = now()
        try:
            u.send(A.PING.pack(b'FPNG', seq, t0))
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                data = u.recv(64)
                t3 = now()
                if len(data) == A.PONG.size:
                    magic, s, a, t1, t2 = A.PONG.unpack(data)
                    if magic == b'FPON' and s == seq and a == t0:
                        self.samples.append((t0, t1, t2, t3))
                        return True
        except OSError:
            pass
        return False

    def pings(self, u):
        from ..core.hil import estimate_offset
        seq = 1000
        last = time.monotonic()
        while not self.stopping.wait(0.1):
            seq += 1
            self.ping_once(u, seq)
            if time.monotonic() - last >= 1.0:
                last = time.monotonic()
                self.to_worker({'t': 'clock', 'estimate': estimate_offset(self.samples)})

    def session_reader(self):
        while True:
            try:
                h, _ = A.read_msg(self.sess)
            except (ConnectionError, OSError, ValueError) as e:
                if not self.stopping.is_set():
                    self.error(f'the device closed the session: {e}')
                    self.stopping.set()
                break
            t = h.get('t')
            if t == 'control':
                h['ctrl_rx'] = now()
                self.to_worker(h)
            elif t == 'stats':
                self.stats = h
                self.stats_ev.set()
                break
            elif t == 'error':
                self.error(f'the device: {h.get("error")}')

    def camera_sender(self):
        import numpy as np
        while not self.stopping.is_set():
            it = self.cam_slot.take()
            if it is None:
                continue
            frame, h, data = it
            meta = {'f': frame, 's': h['sim_t'], 'h': h['cam']['hw_t']}
            if self.video is not None:
                self.video.push(frame, bytes(data), meta)
                continue
            w, hh = int(h['cam']['w']), int(h['cam']['h'])
            t0 = now()
            bgr = np.frombuffer(data, np.uint8).reshape(hh, w, 4)[:, :, :3].tobytes()
            msg = A.pack({'t': 'camera', 'frame': frame, 'sim_t': h['sim_t'], 'hw_t': h['cam']['hw_t'], 'w': w,
                          'h': hh, 'format': 'bgr8', 'speed_mps': h.get('speed_mps')}, bgr)
            t1 = now()
            try:
                self.cam_sock.sendall(msg)
            except OSError as e:
                self.error(f'the camera stream failed: {e}')
                self.stopping.set()
                break
            self.stamp(frame, enc_start=t0, enc_end=t1, cam_sent=t1, cam_sent_end=now(), cam_bytes=len(msg),
                       cam_done=True)

    def lidar_sender(self):
        import numpy as np
        lid = self.cfg['lidar']
        enc = lid['encoding']
        while not self.stopping.is_set():
            it = self.lidar_slot.take()
            if it is None:
                continue
            frame, h, data = it
            if enc == 'pointcloud2':
                from ..core import mcap_bag as M
                pts = np.frombuffer(data, np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')]))
                payload = M.encode('sensor_msgs/msg/PointCloud2', M.pointcloud2(pts, int(h['lidar']['hw_t']), 'lidar'))
            else:
                payload = data
            msg = A.pack({'t': 'lidar', 'frame': frame, 'sim_t': h['sim_t'], 'hw_t': h['lidar']['hw_t'],
                          'n': int(h['lidar']['n']), 'encoding': enc, 'speed_mps': h.get('speed_mps')}, payload)
            t_sent = now()
            try:
                if self.lidar_sock is not None:
                    self.lidar_sock.sendall(msg)
                else:
                    for dg in A.chunks(frame, msg):
                        self.lidar_udp.send(dg)
            except OSError as e:
                self.error(f'the lidar stream failed: {e}')
                self.stopping.set()
                break
            self.stamp(frame, lidar_sent=t_sent, lidar_sent_end=now(), lidar_bytes=len(msg), lidar_done=True)

    # ---------------------------------------------------------------- the run
    def run(self) -> int:
        code = 0
        try:
            if self.local:
                self.start_local_agent()
            elif not self.address:
                raise RuntimeError('no device address (FALCON_HIL_ADDRESS is empty)')
            self.hello()
            self.open_streams()
            u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            u.connect((self.address, self.port))
            u.settimeout(0.5)
            for k in range(30):                          # a burst before the first frame: an estimate from the start
                self.ping_once(u, k)
                time.sleep(0.01)
            from ..core.hil import estimate_offset
            self.to_worker({'t': 'ready', 'device': self.device.get('env') or {}, 'stack': self.device.get('stack'),
                            'decoder': self.device.get('decoder'), 'clock': estimate_offset(self.samples),
                            'encoder': self.video.encoder if self.video is not None else 'raw',
                            'encoder_props': self.video.props if self.video is not None else {},
                            'device_clock': self.device.get('clock') or {}})
            self._start(lambda: self.pings(u), 'gw-pings')
            self._start(self.session_reader, 'gw-session')
            self._start(self.camera_sender, 'gw-camera')
            self._start(self.lidar_sender, 'gw-lidar')
            self.from_worker()
        except Exception as e:                           # noqa: BLE001 (told to the worker, which fails the run)
            self.error(f'{type(e).__name__}: {e}')
            code = 2
        finally:
            self.finish()
        return code

    def from_worker(self):
        while not self.stopping.is_set():
            try:
                h, payload = A.read_msg(self.ws)
            except (ConnectionError, OSError, ValueError):
                break
            t_recv = now()
            if h.get('t') == 'bye':
                break
            if h.get('t') != 'frame':
                continue
            frame = int(h['frame'])
            nc = int(h['cam']['bytes'])
            self.stamp(frame, gw_recv=t_recv)
            view = memoryview(payload)
            self.cam_slot.put((frame, h, view[:nc]))
            self.lidar_slot.put((frame, h, view[nc:]))

    def finish(self):
        self.stopping.set()
        for s in (self.cam_slot, self.lidar_slot):
            s.close()
        if self.sess is not None:
            try:
                self.sess.sendall(A.pack({'t': 'bye'}))
            except OSError:
                pass
            self.stats_ev.wait(5.0)
        if self.video is not None:
            self.video.stop()
        for s in (self.sess, self.cam_sock, self.lidar_sock, self.lidar_udp):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        for t in self.threads:
            t.join(3.0)
        if self.agent_proc is not None:
            try:
                self.agent_proc.wait(4.0)
            except subprocess.TimeoutExpired:
                self.agent_proc.terminate()
                try:
                    self.agent_proc.wait(2.0)
                except subprocess.TimeoutExpired:
                    self.agent_proc.kill()
        from ..core.provenance import write_json
        counts = dict(self.counts, camera_dropped=self.cam_slot.dropped, lidar_dropped=self.lidar_slot.dropped)
        try:
            with open(os.path.join(self.out, 'clock.jsonl'), 'w', encoding='utf-8') as fh:
                for s in self.samples:
                    fh.write(json.dumps({'t0': s[0], 't1': s[1], 't2': s[2], 't3': s[3]}) + '\n')
            write_json(os.path.join(self.out, 'device.json'),
                       {'ready': {k: self.device.get(k) for k in ('env', 'stack', 'decoder', 'clock')},
                        'stats': self.stats, 'mode': 'local' if self.local else 'remote',
                        'agent_exit': None if self.agent_proc is None else self.agent_proc.returncode})
            write_json(os.path.join(self.out, 'gateway.json'),
                       {'python': sys.version.split()[0], 'counts': counts,
                        'encoder': self.video.encoder if self.video is not None else 'raw',
                        'encoder_props': self.video.props if self.video is not None else {},
                        'gstreamer': (A.gst().version_string() if self.video is not None else None)})
        except OSError as e:
            self.error(f'could not write the gateway\'s files: {e}')
        self.to_worker({'t': 'done', 'counts': counts})


def main(argv=None) -> int:
    # the worker blocks SIGINT and SIGTERM for its threads and a child inherits the mask: take them back
    if hasattr(signal, 'pthread_sigmask'):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    ap = argparse.ArgumentParser(prog='python -m falcon_autonomy.hil.gateway')
    ap.add_argument('--fd', type=int, required=True, help='the worker\'s end of the socket pair')
    ap.add_argument('--config', required=True)
    a = ap.parse_args(argv)
    with open(a.config, encoding='utf-8') as fh:
        cfg = json.load(fh)
    ws = socket.socket(fileno=a.fd)
    return Gateway(cfg, ws, os.environ.get('FALCON_HIL_ADDRESS', '')).run()


if __name__ == '__main__':
    sys.exit(main())
