#!/usr/bin/env python3
"""A tiny ROS 2 driving stack to test Falcon's bridge end to end (rclpy, numpy; ROS 2 Humble or later). Part of Falcon
Autonomy (GPL-3.0-or-later). It is a test object, not a driver: it holds the line it started on and brakes for what its
lidar sees ahead.

    python3 follower_stack.py --control autoware|ackermann|carla --speed 8 [--odom T] [--points T] [--control-topic T]

Per odometry message (Falcon's bridge publishes one per simulation step, after the step's sensors) it publishes one
control message stamped with the odometry's stamp, so the bridge takes it as that step's control:

* lateral: pure pursuit (look-ahead max(6 m, 1 s · speed), wheelbase 2.9 m) back to the straight line through its
  first pose along its first heading;
* longitudinal: the nearest lidar point in its corridor (|y| < --corridor m of base_link, 0.3 m < z < 3 m, x beyond
  --front m) gives a gap; the allowed speed is sqrt(2 · --decel · (gap − --stop-gap)), the set speed the lesser of it
  and --speed; acceleration demand 1.5 · (set − v), within [−8, 2.5] m/s², and −8 m/s² once the gap is below
  --stop-gap. Points are taken to base_link through /tf_static (the lidar's mount). DDS delivers each topic on its
  own, so a step's cloud (large) often arrives after its odometry (small): the follower waits for the cloud of the
  odometry's stamp up to --points-wait wall seconds (0.2), then acts on the latest cloud it has, counted as
  "stale points".

Control messages: autoware_control_msgs/Control on /control/command/control_cmd (velocity, acceleration with
is_defined_acceleration, steering_tire_angle), ackermann_msgs/AckermannDriveStamped on /falcon/ego/ackermann_cmd
(speed, acceleration, steering_angle) or carla_msgs/CarlaEgoVehicleControl on /falcon/ego/vehicle_control_cmd
(throttle a / 3 m/s², brake −a / 8 m/s², steer −δ / 70°). It runs on simulation time (use_sim_time) and prints a
summary on exit (SIGINT / SIGTERM, or --max-steps).

--degraded-mode: it also watches its sensors' health from their data (degraded.py beside it: the lidar's sweeps,
the IMU's and GNSS's messages found under /falcon) and, once a failure is detected, carries out the minimal-risk
manoeuvre: target speed 0, a jerk-limited deceleration to --mrm-decel, the line held, then standing still. Without
it, it ignores failures (the baseline the MRM verifier should fail).
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time

import numpy as np

try:                                                    # beside this file: in the bridge image (/falcon/ros2) and here
    from degraded import MRM, Monitor
except ImportError:                                     # imported as falcon_autonomy.ros2.follower_stack
    from .degraded import MRM, Monitor

WHEELBASE_M = 2.9
TOPICS = {'autoware': '/control/command/control_cmd', 'ackermann': '/falcon/ego/ackermann_cmd',
          'carla': '/falcon/ego/vehicle_control_cmd'}


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cloud_xyz(msg) -> np.ndarray:
    """N × 3 float of a PointCloud2's x, y, z (float32 fields)."""
    off = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(n, msg.point_step)
    cols = [buf[:, off[k]:off[k] + 4].copy().view('<f4').reshape(n) for k in ('x', 'y', 'z')]
    return np.stack(cols, 1).astype(float)


class Brain:
    """The decisions, without ROS (unit-testable)."""

    def __init__(self, speed, corridor=1.3, front=2.6, stop_gap=6.0, decel=4.0, degraded=False,
                 degraded_params=None):
        self.speed, self.corridor, self.front, self.stop_gap, self.decel = speed, corridor, front, stop_gap, decel
        self.line = None
        self.monitor = Monitor(degraded_params, ('lidar',)) if degraded else None
        self.mrm = MRM(degraded_params) if degraded else None
        self.a_prev = 0.0
        self.fix0 = None

    # degraded mode: what arrives (simulation seconds), for the health monitor
    def sense_points(self, t, pts_sensor):
        if self.monitor is not None:
            self.monitor.lidar(t, pts_sensor)

    def sense_imu(self, t):
        if self.monitor is not None:
            self.monitor.expect.add('imu')
            self.monitor.imu(t)

    def sense_fix(self, t, lat, lon, odom_xy):
        if self.monitor is None:
            return
        self.monitor.expect.add('gnss')
        if self.fix0 is None:
            self.fix0 = (lat, lon)
        north = math.radians(lat - self.fix0[0]) * 6378137.0
        east = math.radians(lon - self.fix0[1]) * 6378137.0 * math.cos(math.radians(self.fix0[0]))
        self.monitor.gnss(t, (north, east), odom_xy)

    def degraded(self):
        return self.monitor is not None and self.monitor.degraded is not None

    def gap(self, pts_base: np.ndarray | None):
        if pts_base is None or not len(pts_base):
            return None
        m = ((pts_base[:, 0] > self.front) & (np.abs(pts_base[:, 1]) < self.corridor) & (pts_base[:, 2] > 0.3)
             & (pts_base[:, 2] < 3.0))
        return float(pts_base[m, 0].min() - self.front) if m.any() else None

    def act(self, x, y, yaw, v, pts_base, t=None):
        """-> (steering angle rad, positive left; set speed m/s; acceleration m/s²; gap m or None). t: simulation
        seconds (degraded mode judges the sensors' silence by it)."""
        if self.line is None:
            self.line = (x, y, yaw)
        x0, y0, h = self.line
        d = np.array([math.cos(h), math.sin(h)])
        s = (x - x0) * d[0] + (y - y0) * d[1]
        la = max(6.0, 1.0 * v)
        tx, ty = x0 + d[0] * (s + la), y0 + d[1] * (s + la)
        dx, dy = tx - x, ty - y
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        delta = math.atan2(2.0 * WHEELBASE_M * ly, lx * lx + ly * ly)
        gap = self.gap(pts_base)
        target = self.speed
        if gap is not None:
            target = min(target, math.sqrt(max(0.0, 2.0 * self.decel * (gap - self.stop_gap))))
        a = min(max(1.5 * (target - v), -8.0), 2.5)
        if gap is not None and gap < self.stop_gap:
            a = -8.0
        if self.monitor is not None and t is not None and self.monitor.check(t)[0]:
            m, hold = self.mrm.accel(t, v, self.a_prev)     # the minimal-risk manoeuvre: stop in the lane, hold
            # its sensors are not to be trusted: the speed set where it is (a speed-tracking bridge adds nothing), the
            # deceleration the manoeuvre's; standing, speed 0 and the brake held
            target, a = (0.0, -self.mrm.p['mrm_decel']) if hold else (v, m)
        self.a_prev = a
        return delta, target, a, gap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Falcon test stack: line holding and lidar braking')
    ap.add_argument('--control', choices=sorted(TOPICS), default='autoware')
    ap.add_argument('--control-topic', default='')
    ap.add_argument('--speed', type=float, default=8.0, help='cruise speed, m/s')
    ap.add_argument('--odom', default='/falcon/ego/odometry')
    ap.add_argument('--points', default='', help='PointCloud2 topic; default: the first one found under /falcon')
    ap.add_argument('--corridor', type=float, default=1.3)
    ap.add_argument('--front', type=float, default=2.6)
    ap.add_argument('--stop-gap', type=float, default=6.0)
    ap.add_argument('--decel', type=float, default=4.0)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--points-wait', type=float, default=0.2, help='wall seconds to wait for the step\'s cloud')
    ap.add_argument('--degraded-mode', action='store_true', help='detect sensor failures and stop in the lane')
    ap.add_argument('--mrm-decel', type=float, default=2.5, help='the minimal-risk manoeuvre\'s deceleration, m/s²')
    a = ap.parse_args(argv)
    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from tf2_msgs.msg import TFMessage
    rclpy.init()
    node = rclpy.create_node('falcon_follower', parameter_overrides=[Parameter('use_sim_time', value=True)])
    brain = Brain(a.speed, a.corridor, a.front, a.stop_gap, a.decel, a.degraded_mode, {'mrm_decel': a.mrm_decel})
    state = {'pts': None, 'pts_stamp': -1, 'mount': {}, 'steps': 0, 'stale_points': 0, 'no_points': 0,
             'braking': 0, 'min_gap': None}
    reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    if a.control == 'autoware':
        from autoware_control_msgs.msg import Control as Msg
    elif a.control == 'ackermann':
        from ackermann_msgs.msg import AckermannDriveStamped as Msg
    else:
        from carla_msgs.msg import CarlaEgoVehicleControl as Msg
    pub = node.create_publisher(Msg, a.control_topic or TOPICS[a.control], reliable)

    def on_tf(m):
        for t in m.transforms:
            tr = t.transform
            state['mount'][t.child_frame_id] = (np.array([tr.translation.x, tr.translation.y, tr.translation.z]),
                                                quat_matrix((tr.rotation.x, tr.rotation.y, tr.rotation.z,
                                                             tr.rotation.w)), t.header.frame_id)
    node.create_subscription(TFMessage, '/tf_static', on_tf, latched)

    def sim_s(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def on_points(m):
        P = cloud_xyz(m)
        brain.sense_points(sim_s(m.header.stamp), P)          # the sensor frame: azimuths as the lidar sees them
        mt = state['mount'].get(m.header.frame_id)
        if mt is not None and mt[2] == 'base_link':
            P = P @ mt[1].T + mt[0]
        elif mt is None:
            return                                       # its mount is not known yet
        state['pts'] = P
        state['pts_stamp'] = m.header.stamp.sec * 10 ** 9 + m.header.stamp.nanosec
        pend = state.get('pending')
        if pend is not None and state['pts_stamp'] >= pend[0].header.stamp.sec * 10 ** 9 + \
                pend[0].header.stamp.nanosec:
            act(*pend)

    def subscribe_points(topic):
        node.create_subscription(PointCloud2, topic, on_points, reliable)
        node.get_logger().info(f'points from {topic}')
        state['points_topic'] = topic

    if a.points:
        subscribe_points(a.points)

    def on_odom(m):
        stamp = m.header.stamp.sec * 10 ** 9 + m.header.stamp.nanosec
        if state.get('pending') is not None:              # a newer step came: act on the waiting one now
            act(*state.pop('pending'))
        if 'points_topic' in state and state['pts_stamp'] < stamp:
            state['pending'] = (m, time.monotonic())       # wait (briefly) for this step's cloud
            return
        act(m, None)

    def act(m, _since):
        state['pending'] = None
        if 'points_topic' not in state:
            for topic, types in node.get_topic_names_and_types():
                if topic.startswith('/falcon') and 'sensor_msgs/msg/PointCloud2' in types:
                    subscribe_points(topic)
                    break
        if a.degraded_mode and 'health_topics' not in state:      # the IMU and GNSS too, for their health
            from sensor_msgs.msg import Imu, NavSatFix
            state['health_topics'] = []
            for topic, types in node.get_topic_names_and_types():
                if topic.startswith('/falcon') and 'sensor_msgs/msg/Imu' in types:
                    node.create_subscription(Imu, topic, lambda mm: brain.sense_imu(sim_s(mm.header.stamp)),
                                             reliable)
                    state['health_topics'].append(topic)
                elif topic.startswith('/falcon') and 'sensor_msgs/msg/NavSatFix' in types:
                    node.create_subscription(NavSatFix, topic, lambda mm: brain.sense_fix(
                        sim_s(mm.header.stamp), mm.latitude, mm.longitude, state.get('odom_xy', (0.0, 0.0))),
                        reliable)
                    state['health_topics'].append(topic)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        v = math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)
        stamp = m.header.stamp.sec * 10 ** 9 + m.header.stamp.nanosec
        if state['pts'] is None:
            state['no_points'] += 1
        elif state['pts_stamp'] < stamp:
            state['stale_points'] += 1
        state['odom_xy'] = (p.x, p.y)
        delta, target, acc, gap = brain.act(p.x, p.y, yaw, v, state['pts'], sim_s(m.header.stamp))
        if brain.degraded() and 'degraded' not in state:
            state['degraded'] = {'t': brain.monitor.degraded[0], 'why': brain.monitor.degraded[1]}
            node.get_logger().warning(f'degraded: {brain.monitor.degraded[1]}: minimal-risk manoeuvre')
        if gap is not None:
            state['min_gap'] = gap if state['min_gap'] is None else min(state['min_gap'], gap)
        if acc < 0:
            state['braking'] += 1
        out = Msg()
        if a.control == 'autoware':
            out.stamp = m.header.stamp
            out.lateral.stamp = out.longitudinal.stamp = m.header.stamp
            out.lateral.steering_tire_angle = float(delta)
            out.longitudinal.velocity = float(target)
            out.longitudinal.acceleration = float(acc)
            out.longitudinal.is_defined_acceleration = True
        elif a.control == 'ackermann':
            out.header.stamp = m.header.stamp
            out.header.frame_id = 'base_link'
            out.drive.steering_angle = float(delta)
            out.drive.speed = float(target)
            out.drive.acceleration = float(acc) if acc != 0 else 1e-6
        else:
            out.header.stamp = m.header.stamp
            out.throttle = float(min(max(acc / 3.0, 0.0), 1.0))
            out.brake = float(min(max(-acc / 8.0, 0.0), 1.0))
            out.steer = float(min(max(-delta / math.radians(70.0), -1.0), 1.0))
        pub.publish(out)
        state['steps'] += 1
        if a.max_steps and state['steps'] >= a.max_steps:
            state['stop'] = True
    node.create_subscription(Odometry, a.odom, on_odom, reliable)

    def stop(*_):
        state['stop'] = True
    signal.signal(signal.SIGTERM, stop)                  # spin in short slices so that the flag is seen
    signal.signal(signal.SIGINT, stop)
    try:
        while not state.get('stop') and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            pend = state.get('pending')
            if pend is not None and time.monotonic() - pend[1] > a.points_wait:
                act(*pend)                                   # the cloud did not come: act on the last one
    except KeyboardInterrupt:
        pass
    finally:
        print(json.dumps({'follower': {k: v for k, v in state.items() if k not in ('pts', 'mount', 'stop', 'pending')}},
                         default=str), flush=True)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
