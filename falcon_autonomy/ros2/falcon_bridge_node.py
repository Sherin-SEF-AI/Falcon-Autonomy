#!/usr/bin/env python3
"""Falcon's ROS 2 bridge node: publishes the ROS 2 messages Falcon's stack worker sends for every simulation step and
forwards the stack's control messages back to it (rclpy; ROS 2 Humble or later). Part of Falcon Autonomy (GPL-3.0-or-
later); this file is standalone — it needs only rclpy and the message packages — so it can be copied onto any ROS 2
machine, or run in the falcon-ros2-bridge:humble image (falcon_autonomy/ros2/Dockerfile).

    python3 falcon_bridge_node.py --connect 127.0.0.1:2330 [--wait 120] [--serve]

It connects to the worker (TCP) and speaks the link of falcon_autonomy/core/stackbridge.py: a "hello" names the topics
(type, QoS) and the control topic; then per step a "step" header with the messages' CDR bytes as its payload. The node
publishes those bytes as serialized messages (Publisher.publish(bytes): no decoding, no copying into Python message
objects) — /clock first, then the rest in the worker's order — and replies "published" with the time it took. The
control topic is subscribed raw (raw=True): every message is sent back as it came, tagged with the step last published,
and the worker decides which one is the step's control (stackbridge.Deadlines).

QoS: "reliable" (RELIABLE, KEEP_LAST 10, VOLATILE), "best_effort" (BEST_EFFORT, KEEP_LAST 10), "latched" (RELIABLE,
KEEP_LAST 1, TRANSIENT_LOCAL: /tf_static). The control subscription is RELIABLE, KEEP_LAST 10.

--serve keeps the node up after a run ends and connects again for the next (a batch of runs); without it the node
exits when the worker says bye or closes the link. Exit codes: 0 a run ended normally, 2 no worker within --wait,
3 the link broke.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time

HEAD = struct.Struct('>II')         # header length, payload length (core.livelink's framing)
MAX_PART = 256 << 20


def pack(header: dict, payload: bytes = b'') -> bytes:
    h = json.dumps(header, separators=(',', ':')).encode('utf-8')
    return HEAD.pack(len(h), len(payload)) + h + payload


def recv_exact(sock, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise EOFError('the link closed')
        got += k
    return bytes(buf)


def recv_message(sock):
    hl, pl = HEAD.unpack(recv_exact(sock, HEAD.size))
    if hl > MAX_PART or pl > MAX_PART:
        raise ValueError(f'message too large ({hl} + {pl} bytes)')
    header = json.loads(recv_exact(sock, hl).decode('utf-8'))
    return header, (recv_exact(sock, pl) if pl else b'')


class BridgeCore:
    """The node's logic without ROS: an adapter makes publishers (callable(bytes)) and the control subscription.

    adapter.publisher(topic, type, qos) -> callable(bytes)
    adapter.subscribe(topic, type, callback(bytes)) -> (ok: bool, why: str)
    adapter.info() -> {'ros_distro', 'rmw'}
    """

    def __init__(self, adapter, send):
        self.adapter, self.send = adapter, send
        self.pubs: dict = {}
        self.step = -1
        self.control = None
        self.control_type = ''
        self.lock = threading.Lock()
        self.ended = ''

    def handle(self, header: dict, payload: bytes):
        t = header.get('t')
        if t == 'hello':
            self.hello(header)
        elif t == 'step':
            self.publish_step(header, payload)
        elif t == 'bye':
            self.ended = header.get('reason') or 'bye'
        else:
            raise ValueError(f'unknown message {t!r}')

    def hello(self, h: dict):
        if int(h.get('protocol', 0)) != 1:
            raise ValueError(f'protocol {h.get("protocol")} (this node speaks 1)')
        for e in h.get('topics', []):
            if e['topic'] not in self.pubs:
                self.pubs[e['topic']] = self.adapter.publisher(e['topic'], e['type'], e.get('qos', 'reliable'))
        c = h.get('control')
        ctl = None
        if c:
            ok, why = self.adapter.subscribe(c['topic'], c['type'], self.on_control)
            if ok:
                self.control = c['topic']
            ctl = {'topic': c['topic'], 'type': c['type'], 'ok': ok, 'why': why}
            self.control_type = c['type']
        self.send({'t': 'ready', 'node': 'falcon_bridge', 'control': ctl, **self.adapter.info()})

    def publish_step(self, h: dict, payload: bytes):
        t0 = time.monotonic()
        with self.lock:
            self.step = int(h['step'])
        view = memoryview(payload)
        o = 0
        parts = []
        for m in h.get('msgs', []):
            n = int(m['nbytes'])
            parts.append((m['topic'], bytes(view[o:o + n])))
            o += n
        if o != len(payload):
            raise ValueError(f'step {h["step"]}: the payload has {len(payload)} bytes, the header names {o}')
        parts.sort(key=lambda p: p[0] != '/clock')          # /clock first, the rest in order (sort is stable)
        for topic, data in parts:
            pub = self.pubs.get(topic)
            if pub is None:
                raise ValueError(f'step {h["step"]}: topic {topic} was not in the hello')
            pub(data)
        self.send({'t': 'published', 'step': self.step, 'publish_us': int((time.monotonic() - t0) * 1e6)})

    def on_control(self, data: bytes):
        if self.ended:
            return
        with self.lock:
            step = self.step
        try:
            self.send({'t': 'control', 'step': step, 'topic': self.control, 'type': self.control_type,
                       'recv_ns': time.monotonic_ns()}, data)
        except OSError:
            self.ended = self.ended or 'the link closed'


class RclpyAdapter:
    def __init__(self, node):
        self.node = node
        self.pubs: dict = {}            # kept across runs (--serve): one publisher per topic
        self.subs: dict = {}            # topic -> [subscription, current callback]

    def _qos(self, kind: str):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        if kind == 'latched':
            return QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        rel = ReliabilityPolicy.BEST_EFFORT if kind == 'best_effort' else ReliabilityPolicy.RELIABLE
        return QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST, reliability=rel,
                          durability=DurabilityPolicy.VOLATILE)

    def publisher(self, topic, typ, qos):
        from rosidl_runtime_py.utilities import get_message
        key = (topic, typ, qos)
        if key not in self.pubs:
            self.pubs[key] = self.node.create_publisher(get_message(typ), topic, self._qos(qos))
        return self.pubs[key].publish

    def subscribe(self, topic, typ, cb):
        try:
            from rosidl_runtime_py.utilities import get_message
            cls = get_message(typ)
        except (AttributeError, ModuleNotFoundError, ValueError) as e:
            return False, f'{typ} is not installed here ({e})'
        if topic in self.subs:
            self.subs[topic][1] = cb                     # the next run's link
            return True, ''
        entry = [None, cb]
        entry[0] = self.node.create_subscription(cls, topic, lambda data: entry[1](data), self._qos('reliable'),
                                                 raw=True)
        self.subs[topic] = entry
        return True, ''

    def info(self):
        import os
        try:
            from rclpy.utilities import get_rmw_implementation_identifier
            rmw = get_rmw_implementation_identifier()
        except ImportError:
            rmw = os.environ.get('RMW_IMPLEMENTATION', '')
        return {'ros_distro': os.environ.get('ROS_DISTRO', ''), 'rmw': rmw}


def connect(host: str, port: int, wait_s: float):
    t0 = time.monotonic()
    while True:
        try:
            s = socket.create_connection((host, port), timeout=5.0)
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError:
            if time.monotonic() - t0 > wait_s:
                return None
            time.sleep(0.2)


def serve_one(sock, adapter, log) -> int:
    lock = threading.Lock()

    def send(header, payload=b''):
        with lock:
            sock.sendall(pack(header, payload))
    core = BridgeCore(adapter, send)
    try:
        while not core.ended:
            core.handle(*recv_message(sock))
    except EOFError:
        log('the worker closed the link')
        return 0
    except (OSError, ValueError) as e:
        log(f'link error: {e}')
        return 3
    finally:
        sock.close()
    log(f'run ended: {core.ended}')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Falcon Autonomy ROS 2 bridge node')
    ap.add_argument('--connect', default='127.0.0.1:2330', help='the stack worker\'s host:port')
    ap.add_argument('--wait', type=float, default=120.0, help='seconds to wait for a worker')
    ap.add_argument('--serve', action='store_true', help='stay up and serve the next run too')
    a = ap.parse_args(argv)
    host, _, port = a.connect.rpartition(':')
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    node = rclpy.create_node('falcon_bridge')
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True, name='falcon-bridge-executor').start()

    def log(msg):
        node.get_logger().info(msg)
    code = 0
    adapter = RclpyAdapter(node)
    try:
        while True:
            sock = connect(host or '127.0.0.1', int(port), a.wait)
            if sock is None:
                log(f'no worker at {a.connect} within {a.wait:g} s')
                code = 2 if code == 0 else code
                break
            log(f'connected to {a.connect}')
            code = serve_one(sock, adapter, log)
            if not a.serve:
                break
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
