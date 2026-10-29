"""A stand-in ROS 2 stack for the CI tests (no ROS): it plays Falcon's bridge node — falcon_bridge_node's own logic
(BridgeCore over the worker's link) with an in-process adapter instead of rclpy — and answers each odometry message
with an Autoware Control: steer straight, hold --speed (m/s). --delay-ms waits that long before answering (a slow
stack: the control wait the latency gate measures); --state FILE counts the invocations, and --then-speed is used from
the second invocation on (a stack that fails at first and passes on a retry: the flaky-run policy).

    python3 tests/data/ci_stack.py --connect 127.0.0.1:{port} --speed 0 [--delay-ms 5] [--state F --then-speed S]
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'falcon_autonomy', 'ros2')]

import falcon_bridge_node as NODE                 # noqa: E402  (standalone: no rclpy at import)
from falcon_autonomy.core import ros2msg as R     # noqa: E402


class Adapter:
    def __init__(self, speed: float, delay_s: float):
        self.speed, self.delay_s = speed, delay_s
        self.types, self.cb = {}, None

    def publisher(self, topic, typ, qos):
        self.types[topic] = typ
        return lambda data, t=topic: self.deliver(t, data)

    def subscribe(self, topic, typ, cb):
        self.cb = cb
        return True, ''

    def info(self):
        return {'ros_distro': 'none (CI test stand-in)', 'rmw': 'in-process'}

    def deliver(self, topic, data):
        if self.types[topic] != 'nav_msgs/msg/Odometry' or self.cb is None:
            return
        st = R.decode('nav_msgs/msg/Odometry', data)['header']['stamp']
        if self.delay_s:
            time.sleep(self.delay_s)
        acc = 3.0 if self.speed > 0 else -8.0
        self.cb(R.encode('autoware_control_msgs/msg/Control', {
            'stamp': st, 'lateral': {'stamp': st, 'steering_tire_angle': 0.0},
            'longitudinal': {'stamp': st, 'velocity': self.speed, 'acceleration': acc,
                             'is_defined_acceleration': True}}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--connect', required=True)
    ap.add_argument('--speed', type=float, default=0.0)
    ap.add_argument('--delay-ms', type=float, default=0.0)
    ap.add_argument('--state', default='')
    ap.add_argument('--then-speed', type=float, default=None)
    a = ap.parse_args(argv)
    speed = a.speed
    if a.state:
        n = int(open(a.state).read() or 0) if os.path.isfile(a.state) else 0
        with open(a.state, 'w') as fh:
            fh.write(str(n + 1))
        if n >= 1 and a.then_speed is not None:
            speed = a.then_speed
    host, _, port = a.connect.rpartition(':')
    sock = NODE.connect(host, int(port), 60.0)
    if sock is None:
        print('no worker to connect to', flush=True)
        return 1
    return NODE.serve_one(sock, Adapter(speed, a.delay_ms / 1000.0), lambda m: print(m, flush=True))


if __name__ == '__main__':
    sys.exit(main())
