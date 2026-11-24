#!/usr/bin/env python3
"""The runtime monitor's ROS 2 node (rclpy; ROS 2 Humble): subscribes to the topics of mapping.json, samples the
mapped signals at a fixed rate, steps the monitors of rules.json (falcon_monitor.py beside this file) and publishes

    <alarm_topic>      std_msgs/msg/Bool     true at a step where a rule's verdict is `violated`
    <verdict_topic>    std_msgs/msg/String   the step as JSON: per rule verdict, robustness, worst, violations, pending
    /diagnostics       diagnostic_msgs/msg/DiagnosticArray   one status per rule: OK (ok), WARN (unknown, pending),
                                                             ERROR (violated)

`falcon monitor gen` copies this file verbatim into a monitor package as falcon_monitor_node.py. It imports rclpy, the
message packages named in mapping.json and falcon_monitor — nothing of Falcon's.

    python3 falcon_monitor_node.py [--mapping mapping.json] [--rules rules.json]

Time: the node's clock (ROS time: simulation time with use_sim_time), in seconds from the first tick; that is the rule
signal `t`. A signal with no mapping, never received, or older than the mapping's stale_s is unknown at that sample
(an unknown never makes a violation: core.stl's three-valued semantics). When the clock jumps back (a simulation
restarted) the monitors are reset.
"""
import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import falcon_monitor as FM  # noqa: E402 (the generated runtime beside this file)
except ImportError:              # inside Falcon's own tree, not a generated package: the same file under its name there
    from . import monitor_runtime as FM  # noqa: E402

LEVEL = {'ok': 'OK', 'unknown': 'WARN', 'pending': 'WARN', 'violated': 'ERROR'}


def message_class(type_name: str):
    """'std_msgs/msg/Float64' → the message class."""
    pkg, _msg, name = type_name.split('/')
    return getattr(importlib.import_module(f'{pkg}.msg'), name)


def _json_num(v):
    return None if v is None or v != v else ('inf' if v == float('inf') else '-inf' if v == float('-inf') else v)


class Runner:
    """The node's work, on any object with rclpy's Node methods (create_subscription, create_publisher,
    create_timer, get_clock, get_logger)."""

    def __init__(self, node, mapping: dict, rules: dict):
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        from std_msgs.msg import Bool, String
        self.node, self.mapping, self.rules = node, mapping, rules
        self.Bool, self.String = Bool, String
        self.DiagnosticArray, self.DiagnosticStatus, self.KeyValue = DiagnosticArray, DiagnosticStatus, KeyValue
        self.mapper = FM.SignalMapper(mapping)
        self.set = FM.MonitorSet(rules)
        self.name = str(rules.get('name') or 'falcon_monitor')
        self.unmapped = sorted(s for s in self.set.signals if s != 't' and s not in self.mapper.entries)
        if self.unmapped:
            node.get_logger().warning('signals with no mapping (unknown to the rules that read them): '
                                      + ', '.join(self.unmapped))
        self.subs = []
        for sig, topic, typ, _field in self.mapper.inputs():
            self.subs.append(node.create_subscription(message_class(typ), topic,
                                                      lambda msg, s=sig: self.mapper.on_message(s, msg, self.now()),
                                                      10))
        self.alarm_pub = node.create_publisher(Bool, mapping.get('alarm_topic', '/falcon/monitor/alarm'), 10)
        self.verdict_pub = node.create_publisher(String, mapping.get('verdict_topic', '/falcon/monitor/verdict'), 10)
        self.diag_pub = node.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.t0 = None
        self.resets = 0
        self.timer = node.create_timer(1.0 / float(mapping.get('rate_hz', 20.0)), self.tick)

    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        now = self.now()
        if self.t0 is None:
            self.t0 = now
        try:
            res = self.set.step(now - self.t0, self.mapper.sample(now))
        except FM.MonitorError as e:                  # the clock went back, or a buffer limit: start again
            self.node.get_logger().error(f'monitor reset: {e}')
            self.set = FM.MonitorSet(self.rules)
            self.mapper.latest.clear()
            self.t0 = None
            self.resets += 1
            return
        self.publish(res)

    def publish(self, res: dict):
        alarm = self.Bool()
        alarm.data = bool(res['alarm'])
        self.alarm_pub.publish(alarm)
        doc = {'t': res['t'], 'alarm': res['alarm'], 'resets': self.resets, 'rules': {
            rid: {'verdict': r['verdict'], 'robustness': _json_num(r['robustness']), 'worst': _json_num(r['worst']),
                  'violations': r['violations'], 'pending': r['pending']} for rid, r in res['rules'].items()}}
        s = self.String()
        s.data = json.dumps(doc)
        self.verdict_pub.publish(s)
        arr = self.DiagnosticArray()
        try:
            arr.header.stamp = self.node.get_clock().now().to_msg()
        except AttributeError:
            pass
        for rid, r in res['rules'].items():
            st = self.DiagnosticStatus()
            st.level = getattr(self.DiagnosticStatus, LEVEL[r['verdict']])   # the message's own constants (a byte)
            st.name = f'falcon_monitor: {rid}'
            st.message = r['verdict']
            st.hardware_id = self.name
            for k in ('robustness', 'worst', 'violations', 'pending'):
                kv = self.KeyValue()
                kv.key, kv.value = k, str(_json_num(r[k]))
                st.values.append(kv)
            arr.status.append(st)
        self.diag_pub.publish(arr)


def load(path: str) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Falcon runtime monitor node')
    ap.add_argument('--mapping', default=os.path.join(HERE, 'mapping.json'))
    ap.add_argument('--rules', default=os.path.join(HERE, 'rules.json'))
    a, ros_args = ap.parse_known_args(argv)
    import rclpy
    rclpy.init(args=ros_args)
    node = rclpy.create_node('falcon_monitor')
    Runner(node, load(a.mapping), load(a.rules))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:                             # noqa: BLE001 (already shut down by a signal)
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
