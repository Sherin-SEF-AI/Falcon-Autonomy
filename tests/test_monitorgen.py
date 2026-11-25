"""Runtime monitors generated from the rules (core.monitorgen, core.monitor_runtime; docs/runtime-monitors.md): the
online evaluator against core.stl.evaluate on random formulas and signals (exact equality, unknowns and infinities
included) and on the built-in rule sets; verdict delays and what is refused; the generated package (standard library
only, hashes pinned), its equivalence check on a run recorded on the fake carla, the node on a fake rclpy, the
deployment on a fake transport (never sudo), the GSN evidence kind, the command, the tools and the Hardware tab."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import types

import numpy as np
import pytest

from falcon_autonomy.core import gsn as G
from falcon_autonomy.core import monitor_runtime as RT
from falcon_autonomy.core import monitorgen as MG
from falcon_autonomy.core import rules as R
from falcon_autonomy.core import stl as S

from conftest import FAKE

SIG = ('a', 'b', 'c')


def _expr(rnd, d):
    r = rnd.random()
    if d <= 0 or r < 0.4:
        return rnd.choice(SIG + (str(rnd.choice([0, 1, 2.5, 0.5])),))
    if r < 0.6:
        return f'({_expr(rnd, d - 1)} {rnd.choice("+-*/")} {_expr(rnd, d - 1)})'
    if r < 0.7:
        return f'abs({_expr(rnd, d - 1)})'
    if r < 0.8:
        return f'{rnd.choice(["min", "max"])}({_expr(rnd, d - 1)}, {_expr(rnd, d - 1)})'
    if r < 0.9:
        return f'default({_expr(rnd, d - 1)}, {_expr(rnd, d - 1)})'
    return f'-{_expr(rnd, d - 1)}'


def _interval(rnd):
    lo = rnd.choice([0, 0, 0.1, 0.5, 1])
    return f'[{lo},{lo + rnd.choice([0, 0.3, 1, 2.5])}]'


def _formula(rnd, d):
    r = rnd.random()
    if d <= 0 or r < 0.25:
        if rnd.random() < 0.8:
            return f'{_expr(rnd, 1)} {rnd.choice(["<=", "<", ">=", ">"])} {_expr(rnd, 1)}'
        return rnd.choice(['a', 'true', 'false'])
    if r < 0.35:
        return f'not ({_formula(rnd, d - 1)})'
    if r < 0.55:
        return f'(({_formula(rnd, d - 1)}) {rnd.choice(["and", "or", "->"])} ({_formula(rnd, d - 1)}))'
    if r < 0.85:
        op = rnd.choice('GFHO')
        b = '[0,inf]' if op in 'HO' and rnd.random() < 0.2 else '[1,inf]' if op in 'HO' and rnd.random() < 0.1 \
            else _interval(rnd)
        return f'{op}{b} ({_formula(rnd, d - 1)})'
    return f'(({_formula(rnd, d - 1)}) {rnd.choice(["until", "since"])}{_interval(rnd)} ({_formula(rnd, d - 1)}))'


def _signals(rnd, names, n):
    odd = [0.0, 1.0, math.nan, math.inf, -math.inf]
    return {s: np.array([rnd.choice(odd) if rnd.random() < 0.12 else rnd.uniform(-3, 3) for _ in range(n)])
            for s in names}


def _rule(node, rid='x'):
    return {'id': rid, 'formula': node.text(), 'tree': MG.tree(node), 'signals': sorted(S.names(node))}


# ================================================================================================ the evaluator
def test_the_online_evaluator_equals_the_offline_one_on_random_formulas():
    """The key property: per sample, exactly core.stl.evaluate's robustness — nested bounded future and past
    operators, until and since, unbounded once / historically, Kleene unknowns, ±inf, non-uniform steps."""
    rnd = random.Random(41)
    temporal = 0
    for _ in range(500):
        node = S.parse(_formula(rnd, 3))
        n = rnd.randint(1, 60)
        t = np.cumsum([rnd.choice([0.05, 0.1, 0.1, 0.25, 0.5, 0.013]) for _ in range(n)])
        c = MG.compare(RT, _rule(node), _signals(rnd, SIG, n), t)
        assert c['agree'], (node.text(), c['first_disagreement'], c['online_state'], c['offline_state'])
        assert c['max_abs_diff'] == 0.0 and c['samples'] == n
        temporal += RT.is_temporal(MG.tree(node))
    assert temporal > 250                                     # most of them exercise the streaming operators


def test_the_builtin_rule_sets_are_monitorable_and_equal():
    rnd = random.Random(7)
    for name in ('india', 'generic'):
        rules, src = MG.compile_rules(name)
        assert src['kind'] == 'rules' and len(rules) == len(R.load_ruleset(name)['rules'])
        n = 400
        t = np.arange(n) * 0.05
        names = sorted({s for r in rules for s in r['signals']})
        sig = _signals(rnd, names, n)
        for k in ('passing', 'pass_on_left', 'pass_on_right', 'on_lane', 'in_junction'):
            if k in sig:
                sig[k] = np.array([float(rnd.random() < 0.05) for _ in range(n)])
        sig['t'] = t.copy()
        for r in rules:
            c = MG.compare(RT, r, sig, t)
            assert c['agree'], (name, r['id'], c['first_disagreement'])
        delays = {r['id']: r['horizon_s']['future'] for r in rules}
        keep = 'keep_left' if name == 'india' else 'keep_right'
        assert delays[keep] == 10.0 and all(v == 0.0 for k, v in delays.items() if k != keep)
        assert all(r['sha'].startswith('stl-') and len(r['sha']) == 20 for r in rules)
        assert rules[0]['spec_key'].startswith('stl-')        # the falsifier's key of the same rule


def test_verdicts_wait_for_the_horizon_and_no_longer():
    node = S.parse('eventually[0,1] (a >= 1)')
    m = RT.Monitor(_rule(node))
    assert m.delay_s == 1.0
    out = [m.step(0.25 * k, {'a': 1.0 if k == 6 else 0.0}) for k in range(12)]
    assert [r['verdict'] for r in out[:5]] == ['pending'] * 5      # t = 1.0 is in sample 0's window: wait for 1.25
    assert out[5]['decided'] == [(0, 0.0, -1.0)] and out[5]['verdict'] == 'violated'
    assert out[6]['decided'] == [(1, 0.25, -1.0)]                   # 1.5 (the 1) is just outside [0.25, 1.25]
    assert out[7]['decided'] == [(2, 0.5, 0.0)] and out[7]['verdict'] == 'ok'
    assert out[-1]['pending'] == 5 and m.violations == 2
    end = m.finish()
    assert len(end['decided']) == 5 and end['decided'][-1][2] == -1.0 and end['pending'] == 0
    # a formula with no future operator is decided at its own sample; a missing signal is unknown, never a violation
    m = RT.Monitor(_rule(S.parse('once[0,1] (a >= 1) or b >= 0')))
    r = m.step(0.0, {'a': 0.0})
    assert r['verdict'] == 'unknown' and r['robustness'] is None and r['pending'] == 0 and m.state() == 'not evaluable'
    assert m.step(0.5, {'a': 2.0})['verdict'] == 'ok'
    with pytest.raises(RT.MonitorError, match='must increase'):
        m.step(0.5, {'a': 2.0})
    # memory: a window holds its samples and no more, however long the monitor runs
    m = RT.Monitor(_rule(S.parse('historically[0,2] (a >= 0) and eventually[0,1] (a >= 0)')))
    for k in range(5000):
        m.step(0.05 * k, {'a': 1.0})
    w = [x for x in (m.root.a, m.root.b)]
    assert all(len(x.t) <= 42 and len(x.mono) <= 42 and len(x.v) <= 42 for x in w)
    with pytest.raises(RT.MonitorError, match='buffer limit'):
        m = RT.Monitor(_rule(S.parse('eventually[0,100] (a >= 0)')), max_buffer=50)
        for k in range(200):
            m.step(0.05 * k, {'a': 1.0})


def test_what_cannot_be_monitored_is_refused_with_a_reason():
    for text, why in (('eventually[0,inf] (speed_mps >= 1)', 'unbounded future'),
                      ('speed_mps >= 0 until (lead_gap_m >= 1)', 'until'),
                      ('speed_mps >= 0 since (lead_gap_m >= 1)', 'since'),
                      ('always[0,5] (always (speed_mps >= 0))', 'unbounded future'),
                      ('pet_s >= 2', 'whole run'), ('accel_mps2 >= -3', 'central difference'),
                      ('G (warp >= 1)', 'unknown signal'), ('G (speed_mps >= ', 'column')):
        with pytest.raises(MG.MonitorGenError, match=why):
            MG.compile_stl(text)
    rules, src = MG.compile_stl('G (once (speed_mps >= 1) -> lead_gap_m >= 2)', 'gap')
    assert src['outer_always_removed'] and rules[0]['formula'].startswith('(O[0,inf]')
    assert rules[0]['horizon_s'] == {'past': None, 'future': 0.0}
    with pytest.raises(MG.MonitorGenError, match='no rule nope'):
        MG.compile_rules('india', ['nope'])


# ================================================================================================ the package
RULES = '''ruleset mine "Mine"
handedness RHT
let gap_s = {gap}
rule follow "Keep a distance" heuristic:
    speed_mps < 1 or lead_gap_m >= gap_s * speed_mps
rule slow "Stay slow" heuristic:
    speed_kmh <= 60 or eventually[0,1] (speed_kmh <= 60)
'''


@pytest.fixture(scope='module')
def bench(tmp_path_factory):
    """A project on the fake carla with one recorded lead_brake run, and a package of the india rules."""
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.sim.service import study_service
    from falcon_autonomy.sim.studio import Study
    base = tmp_path_factory.mktemp('mon')
    p = Project.new('Monitors')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(base / 'mon.falcon.json'))
    svc = study_service(workspace=p.workspace, fake_carla=FAKE, lock_dir=str(base / 'locks'))
    try:
        scn = SC.builtin('lead_brake')
        st = Study(svc, p, scn, 'mon', 'world-town10', 'rig-default', seed=3, fps=10)
        run = st.evaluate([SC.sample_params(scn, 5)])[0]['run']
    finally:
        svc.shutdown()
    pkg = os.path.join(p.workspace, 'monitors', 'india')
    rules, src = MG.compile_rules('india')
    man = MG.generate(pkg, rules, src)
    return {'project': p, 'run': run, 'pkg': pkg, 'man': man, 'base': base}


def test_the_package_is_self_contained(bench):
    pkg, man = bench['pkg'], bench['man']
    assert man['schema'] == MG.SCHEMA and man['name'] == 'india' and len(man['hash']) == 64
    for f in ('falcon_monitor.py', 'falcon_monitor_node.py', 'rules.json', 'mapping.json', 'README.md',
              'manifest.json', 'falcon-monitor-india.service'):
        assert os.path.isfile(os.path.join(pkg, f)), f
    assert 'mapping.json' not in man['files'] and MG.changed_files(pkg, man) == []
    for f in ('falcon_monitor.py', 'falcon_monitor_node.py'):
        src = open(os.path.join(pkg, f), encoding='utf-8').read()
        assert re.search(r'^\s*(import|from)\s+(falcon_autonomy|numpy)', src, re.M) is None
    # the module alone, in a bare interpreter (no site packages, not Falcon's folder): only the standard library
    code = ('import sys, json; sys.path.insert(0, sys.argv[1]); import falcon_monitor as m; '
            's = m.MonitorSet(m.load_rules(sys.argv[1] + "/rules.json")); '
            'r = s.step(0.0, {"speed_mps": 20.0, "lead_gap_m": 5.0}); '
            'assert not any(k.split(".")[0] in ("falcon_autonomy", "numpy") for k in sys.modules); '
            'print(json.dumps([r["alarm"], r["rules"]["following_distance"]["verdict"], '
            'r["rules"]["following_distance"]["robustness"], r["rules"]["keep_left"]["verdict"]]))')
    out = subprocess.run([sys.executable, '-I', '-S', '-c', code, pkg], capture_output=True, text=True, cwd='/',
                         timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [True, 'violated', -19.0, 'pending']
    mp = json.load(open(os.path.join(pkg, 'mapping.json')))
    assert mp['signals']['speed_kmh'] == {'from': 'speed_mps', 'scale': 3.6}
    assert mp['signals']['lead_gap_m']['topic'] == '/falcon/signals/lead_gap_m' and 't' not in mp['signals']
    unit = open(os.path.join(pkg, 'falcon-monitor-india.service')).read()
    assert 'sudo' not in unit and '%h/falcon-monitor/india/falcon_monitor_node.py' in unit
    # a mapping the user edited survives a regeneration; a changed pinned file is refused
    mp['signals']['speed_mps'] = {'topic': '/odom', 'type': 'nav_msgs/msg/Odometry', 'field': 'twist.twist.linear.x'}
    json.dump(mp, open(os.path.join(pkg, 'mapping.json'), 'w'))
    man2 = MG.generate(pkg, *MG.compile_rules('india'))
    assert man2['mapping_kept'] and man2['hash'] == man['hash']
    assert json.load(open(os.path.join(pkg, 'mapping.json')))['signals']['speed_mps']['topic'] == '/odom'
    with pytest.raises(MG.MonitorGenError, match='name'):
        MG.generate(str(bench['base'] / 'bad name'), *MG.compile_rules('india'))


def test_the_generated_monitor_equals_the_offline_checker_on_a_recorded_run(bench):
    pkg, run = bench['pkg'], bench['run']
    doc = MG.check(pkg, [run])
    assert doc['agree'] and doc['schema'] == MG.EQ_SCHEMA and doc['stale_rules'] == []
    assert doc['manifest_hash'] == bench['man']['hash']
    row = doc['runs'][0]
    tr = R.trace_from_run(run)
    off = {r['id']: r for r in R.check(tr, R.load_ruleset('india'))['rules']}
    assert row['samples'] == len(tr.t) > 50
    states = set()
    for rid, c in row['rules'].items():
        assert c['agree'] and c['first_disagreement'] is None and c['max_abs_diff'] == 0.0
        assert c['online_state'] == c['offline_state'] == c['check_state'] == off[rid]['state']
        if off[rid]['robustness'] is not None:
            assert c['online_robustness'] == pytest.approx(off[rid]['robustness'], abs=1e-4)
        states.add(c['online_state'])
    assert {'satisfied', 'not evaluable'} <= states             # light states are not recorded in a drive
    assert row['rules']['keep_left']['max_pending'] > 50        # its verdicts waited 10 s; the others did not
    assert row['rules']['speed_limit']['max_pending'] == 0
    assert row['rules']['speed_limit']['decided_before_end'] == row['samples']
    assert json.load(open(os.path.join(pkg, 'equivalence.json')))['agree']
    st = MG.status(pkg)
    assert st['status'] == 'agreed' and all(r['status'] == 'agreed' for r in st['rules'])
    # a monitor that computes something else is caught, with the first sample where it differs
    mod, rules, _man = MG.load_package(pkg)
    rule = dict(next(r for r in rules['rules'] if r['id'] == 'following_distance'))
    rule['tree'] = MG.tree(S.parse('speed_mps < 1 or lead_gap_m >= 3 * speed_mps'))
    sig = R.signals(tr)
    c = MG.compare(mod, rule, sig, tr.t)
    fd = c['first_disagreement']
    assert not c['agree'] and fd['i'] is not None and fd['online'] != fd['offline'] and fd['t'] == tr.t[fd['i']]
    # a run that cannot be read is an error of that run, and no agreement
    d2 = MG.check(pkg, [str(bench['base'] / 'nowhere')], write=False)
    assert not d2['agree'] and d2['runs'][0]['error']
    # a tampered package is refused
    p = os.path.join(pkg, 'rules.json')
    keep = open(p).read()
    open(p, 'w').write(keep.replace('speed_mps', 'speed_kmh', 1))
    with pytest.raises(MG.MonitorGenError, match='changed since it was generated: rules.json'):
        MG.load_package(pkg)
    open(p, 'w').write(keep)


# ================================================================================================ the node
def _fake_ros(monkeypatch):
    """rclpy's message packages as plain classes (no ROS 2 on this machine: the node is tested against these)."""
    def cls(cname, **fields):
        def init(self):
            for k, v in fields.items():
                setattr(self, k, v() if callable(v) else v)
        return type(cname, (), {'__init__': init})
    std = types.ModuleType('std_msgs.msg')
    std.Bool, std.String, std.Float64 = cls('Bool', data=False), cls('String', data=''), cls('Float64', data=0.0)
    diag = types.ModuleType('diagnostic_msgs.msg')
    diag.KeyValue = cls('KeyValue', key='', value='')
    diag.DiagnosticStatus = cls('DiagnosticStatus', level=b'\x00', name='', message='', hardware_id='', values=list)
    diag.DiagnosticStatus.OK, diag.DiagnosticStatus.WARN, diag.DiagnosticStatus.ERROR = b'\x00', b'\x01', b'\x02'
    diag.DiagnosticArray = cls('DiagnosticArray', header=lambda: types.SimpleNamespace(stamp=None), status=list)
    for name, mod in (('std_msgs', types.ModuleType('std_msgs')), ('std_msgs.msg', std),
                      ('diagnostic_msgs', types.ModuleType('diagnostic_msgs')), ('diagnostic_msgs.msg', diag)):
        monkeypatch.setitem(sys.modules, name, mod)
    return std


class _Node:
    def __init__(self):
        self.now_ns, self.subs, self.pubs, self.timers, self.logs = 0, {}, {}, [], []

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=self.now_ns, to_msg=lambda: 0))

    def get_logger(self):
        return types.SimpleNamespace(warning=self.logs.append, error=self.logs.append)

    def create_subscription(self, typ, topic, cb, qos):
        self.subs[topic] = (typ, cb)

    def create_publisher(self, typ, topic, qos):
        self.pubs[topic] = out = []
        return types.SimpleNamespace(publish=out.append)

    def create_timer(self, period, cb):
        self.timers.append((period, cb))


def test_the_node_on_a_fake_rclpy(bench, monkeypatch, tmp_path):
    std = _fake_ros(monkeypatch)
    rules, src = MG.compile_rules('generic', ['following_distance', 'speed_limit'])
    pkg = str(tmp_path / 'gen')
    MG.generate(pkg, rules, src)
    monkeypatch.syspath_prepend(pkg)
    monkeypatch.delitem(sys.modules, 'falcon_monitor', raising=False)
    spec = importlib.util.spec_from_file_location('_node_under_test', os.path.join(pkg, 'falcon_monitor_node.py'))
    node_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node_mod)
    assert node_mod.FM.__file__ == os.path.join(pkg, 'falcon_monitor.py')       # the package's own runtime
    mapping, rdoc = node_mod.load(os.path.join(pkg, 'mapping.json')), node_mod.load(os.path.join(pkg, 'rules.json'))
    del mapping['signals']['speed_limit_kmh']                                    # not mapped: unknown, and said so
    node = _Node()
    run = node_mod.Runner(node, mapping, rdoc)
    assert set(node.subs) == {'/falcon/signals/lead_gap_m', '/falcon/signals/speed_mps'}
    assert node.timers[0][0] == pytest.approx(0.05) and 'speed_limit_kmh' in node.logs[0]

    def send(topic, v, at):
        node.now_ns = int(at * 1e9)
        m = std.Float64()
        m.data = v
        node.subs[topic][1](m)

    def tick(at):
        node.now_ns = int(at * 1e9)
        node.timers[0][1]()
        return json.loads(node.pubs['/falcon/monitor/verdict'][-1].data), node.pubs['/falcon/monitor/alarm'][-1].data
    v, alarm = tick(100.0)                                         # nothing received yet: unknown, no alarm
    assert v['rules']['following_distance']['verdict'] == 'unknown' and not alarm and v['t'] == 0.0
    send('/falcon/signals/speed_mps', 10.0, 100.01)
    send('/falcon/signals/lead_gap_m', 30.0, 100.01)
    v, alarm = tick(100.05)
    assert v['rules']['following_distance'] == {'verdict': 'ok', 'robustness': 10.0, 'worst': 10.0, 'violations': 0,
                                                'pending': 0} and not alarm
    assert v['rules']['speed_limit']['verdict'] == 'ok'            # default(speed_limit_kmh, 50): 36 km/h is within
    send('/falcon/signals/lead_gap_m', 12.0, 100.06)
    v, alarm = tick(100.10)
    assert v['rules']['following_distance']['verdict'] == 'violated' and alarm
    assert v['rules']['following_distance']['robustness'] == -8.0
    st = {s.name: s for s in node.pubs['/diagnostics'][-1].status}
    assert st['falcon_monitor: following_distance'].level == b'\x02'
    assert st['falcon_monitor: speed_limit'].level == b'\x00' and st['falcon_monitor: speed_limit'].hardware_id == 'gen'
    v, alarm = tick(101.0)                                         # the inputs went stale: unknown again, no alarm
    assert v['rules']['following_distance']['verdict'] == 'unknown' and not alarm
    assert v['rules']['following_distance']['violations'] == 1
    n = len(node.pubs['/falcon/monitor/verdict'])
    tick(50.0)                                                     # the clock jumped back: reset, nothing published
    assert len(node.pubs['/falcon/monitor/verdict']) == n and run.resets == 1 and 'monitor reset' in node.logs[-1]
    v, _ = tick(50.05)
    assert v['t'] == 0.0 and v['rules']['following_distance']['violations'] == 0
    assert v['rules']['following_distance']['verdict'] == 'unknown'       # what was received before the jump is gone
    # the mapper alone: a field path, scale and offset, booleans, derived signals, constants
    M = node_mod.FM.SignalMapper({'stale_s': 1.0, 'signals': {
        'speed_mps': {'topic': '/odom', 'type': 'nav_msgs/msg/Odometry', 'field': 'twist.twist.linear.x'},
        'speed_kmh': {'from': 'speed_mps', 'scale': 3.6}, 'on_lane': {'topic': '/lane', 'field': 'data'},
        'lanes_left': {'const': 0}, 'lead_gap_m': {'topic': '/gap', 'field': 'ranges.1', 'offset': -1.0}}})
    odom = types.SimpleNamespace(twist=types.SimpleNamespace(twist=types.SimpleNamespace(
        linear=types.SimpleNamespace(x=5.0))))
    M.on_message('speed_mps', odom, 1.0)
    M.on_message('on_lane', types.SimpleNamespace(data=True), 1.0)
    M.on_message('lead_gap_m', types.SimpleNamespace(ranges=[3.0, 9.0]), 1.0)
    s = M.sample(1.5)
    assert s == {'speed_mps': 5.0, 'speed_kmh': 18.0, 'on_lane': 1.0, 'lanes_left': 0.0, 'lead_gap_m': 8.0}
    M.on_message('lead_gap_m', types.SimpleNamespace(nothing=1), 1.6)
    s = M.sample(2.5)
    assert math.isnan(s['speed_mps']) and math.isnan(s['speed_kmh']) and math.isnan(s['lead_gap_m'])
    with pytest.raises(node_mod.FM.MonitorError, match='derived from'):
        node_mod.FM.SignalMapper({'signals': {'x': {'from': 'y'}}})


# ================================================================================================ deployment
class _Transport:
    def __init__(self, answer='falcon-monitor ok 12\n'):
        self.commands, self.files, self.answer = [], [], answer

    def ssh(self, remote):
        self.commands.append(remote)
        return self.answer if 'falcon_monitor' in remote else ''

    def put(self, local, remote_dir, name):
        assert os.path.isfile(local)
        self.files.append((remote_dir, name))


def test_deployment_on_a_fake_transport_never_runs_sudo_or_systemctl(bench, capsys):
    """No device is reached in the tests: the transport is a fake that records what would be sent."""
    pkg = bench['pkg']
    tr = _Transport()
    out = MG.deploy(pkg, transport=tr)
    assert out['remote_dir'] == '~/falcon-monitor/india' and out['selftest'] == 'falcon-monitor ok 12'
    assert tr.commands[0].startswith('if [ -d "$HOME"/falcon-monitor/india ]')      # keeps the version it replaces
    assert tr.commands[1] == 'mkdir -p "$HOME"/falcon-monitor/india'
    assert {n for _d, n in tr.files} >= {'falcon_monitor.py', 'falcon_monitor_node.py', 'rules.json', 'mapping.json',
                                         'manifest.json', 'falcon-monitor-india.service'}
    assert all(d == '~/falcon-monitor/india' for d, _n in tr.files)
    assert not any('sudo' in c or 'systemctl' in c for c in tr.commands) and len(tr.commands) == 3
    assert any(c.startswith('systemctl --user enable --now falcon-monitor-india.service') for c in out['commands'])
    assert sum('sudo' in c for c in out['commands']) == 1 and out['commands'][-2].startswith('#')
    with pytest.raises(MG.MonitorGenError, match='did not load'):
        MG.deploy(pkg, transport=_Transport('Traceback: no module'))
    assert MG.main_monitor(['deploy', pkg, '--dry-run']) == 0
    text = capsys.readouterr().out
    assert 'then copy' in text and 'Nothing was installed or started' in text and 'systemctl --user' in text
    # without a device in the settings the real transport refuses before anything is sent
    assert MG.main_monitor(['deploy', pkg]) == 2
    assert 'address is not set' in capsys.readouterr().err


# ================================================================================================ the safety case
def test_a_monitor_is_gsn_evidence(tmp_path):
    ws = str(tmp_path / 'ws')
    os.makedirs(ws)
    rf = str(tmp_path / 'mine.rules')
    open(rf, 'w').write(RULES.format(gap=2))
    pkg = os.path.join(ws, 'monitors', 'mine')
    man = MG.generate(pkg, *MG.compile_rules(rf))
    assert [r['id'] for r in man['rules']] == ['follow', 'slow'] and man['rules'][1]['horizon_s']['future'] == 1.0
    doc = {'schema': G.SCHEMA, 'title': 't', 'template': 'blank', 'created': 'now',
           'nodes': [{'id': 'G1', 'type': 'Goal', 'text': 'the rules are watched on the vehicle'},
                     {'id': 'Sn1', 'type': 'Solution', 'text': 'the runtime monitor'}],
           'links': [{'from': 'G1', 'to': 'Sn1', 'kind': 'SupportedBy'}]}
    assert G.validate(doc) == []
    ev = G.bind(ws, doc, 'Sn1', 'monitor', 'monitors/mine')
    assert ev['kind'] == 'monitor' and ev['path'] == os.path.join('monitors', 'mine', 'manifest.json')
    assert G.validate(doc) == []
    v = G.verify(ws, doc)
    assert v['nodes']['Sn1']['status'] == 'amber' and 'no equivalence check' in v['nodes']['Sn1']['why']
    assert v['top'] == {'G1': 'amber'}
    # an equivalence check on a synthetic trace
    mod, rules, _m = MG.load_package(pkg)
    n = 80
    t = np.arange(n) * 0.1
    sig = {'speed_mps': np.linspace(0, 20, n), 'speed_kmh': np.linspace(0, 72, n), 'lead_gap_m': np.full(n, 30.0)}
    res = {r['id']: MG.compare(mod, r, sig, t) for r in rules['rules']}
    assert all(c['agree'] for c in res.values()) and res['follow']['online_state'] == 'violated'
    assert res['slow']['decided_before_end'] == n - 11            # the last second waits for finish()

    def record(agree):
        json.dump({'schema': MG.EQ_SCHEMA, 'at': 'now', 'manifest_hash': man['hash'], 'agree': agree, 'runs': [],
                   'rules': {k: {'agree': agree, 'runs': 1, 'samples': n,
                                 'first_disagreement': None if agree else {'i': 3, 't': 0.3, 'online': 1.0,
                                                                           'offline': 2.0}} for k in res}},
                  open(os.path.join(pkg, 'equivalence.json'), 'w'))
    record(True)
    v = G.verify(ws, doc)
    assert v['nodes']['Sn1']['status'] == 'green' and v['top'] == {'G1': 'green'}
    assert '2 rule(s) monitored' in v['nodes']['Sn1']['why']
    record(False)
    v = G.verify(ws, doc)
    assert v['nodes']['Sn1']['status'] == 'red' and 'sample 3' in v['nodes']['Sn1']['why']
    record(True)
    # the rule's formula changes: the monitor on the vehicle no longer watches the rule that is checked offline
    open(rf, 'w').write(RULES.format(gap=3))
    assert MG.stale_rules(MG.read_manifest(pkg)) == ['follow']
    v = G.verify(ws, doc)
    assert v['nodes']['Sn1']['status'] == 'amber' and 'formula of follow changed' in v['nodes']['Sn1']['why']
    assert MG.status(pkg)['rules'][0]['status'] == 'stale' and MG.status(pkg)['rules'][1]['status'] == 'agreed'
    # generated again: a new manifest (the pinned hash differs: re-bind), the old check no longer counts
    man2 = MG.generate(pkg, *MG.compile_rules(rf))
    assert man2['hash'] != man['hash'] and not os.path.exists(os.path.join(pkg, 'equivalence.json'))
    v = G.verify(ws, doc)
    assert v['nodes']['Sn1']['status'] == 'amber' and 're-bind' in v['nodes']['Sn1']['why']
    G.bind(ws, doc, 'Sn1', 'monitor', 'monitors/mine')
    assert 'no equivalence check' in G.verify(ws, doc)['nodes']['Sn1']['why']
    os.remove(rf)                                                  # the rule set is gone: every rule is stale
    assert MG.stale_rules(MG.read_manifest(pkg)) == ['follow', 'slow']


# ================================================================================================ the command
def test_the_commands(bench, capsys, tmp_path):
    p, run = bench['project'], bench['run']
    out = str(tmp_path / 'ttc')
    assert MG.main_monitor(['gen', '--stl', 'G (ttc_min_s >= 1.5 or eventually[0,2] (speed_mps <= 0.5))',
                            '--out', out]) == 0
    text = capsys.readouterr().out
    assert 'verdict delay 2 s' in text and 'ttc_min_s' in text
    assert MG.main_monitor(['check', run, '--monitor', out, '--json']) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['agree'] and doc['rules']['ttc']['samples'] > 50
    assert MG.main_monitor(['check', run, '--monitor', out]) == 0
    assert 'equals the offline checker' in capsys.readouterr().out
    assert MG.main_monitor(['show', out]) == 0 and 'agreed' in capsys.readouterr().out
    assert MG.main_monitor(['gen', '--rules', 'generic', '--rule', 'solid_line', '--project', p.path, '--name',
                            'solid']) == 0
    capsys.readouterr()
    assert MG.main_monitor(['list', '--project', p.path]) == 0
    text = capsys.readouterr().out
    assert 'solid  unchecked' in text and 'india' in text
    assert MG.main_monitor(['bench', out, '--steps', '2000', '--json']) == 0
    b = json.loads(capsys.readouterr().out)
    assert 0 < b['median_us'] <= b['p99_us'] <= b['max_us'] and b['steps'] == 2000 and 'this machine' in b['where']
    assert MG.main_monitor(['bench', '--rules', 'india', '--steps', '500']) == 0 and 'µs' in capsys.readouterr().out
    assert MG.main_monitor(['gen', '--stl', 'F (speed_mps >= 1)', '--out', out]) == 2
    assert 'unbounded future' in capsys.readouterr().err
    assert MG.main_monitor(['gen', '--rules', 'india', '--stl', 'x', '--out', out]) == 2
    assert MG.main_monitor(['check', run, '--monitor', str(tmp_path / 'none')]) == 2
    capsys.readouterr()
    from falcon_autonomy import cli
    assert cli.LAZY['monitor'][:2] == ('core.monitorgen', 'main_monitor')


# ================================================================================================ the tools
def test_the_monitor_tools(bench):
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    p = bench['project']
    assert REGISTRY['formal_monitor_gen'].tier == 'act' and REGISTRY['formal_monitor_check'].group == 'formal'
    ctx = TOOLS.context(p.path, FAKE)
    try:
        r = call(ctx, 'formal_monitor_gen', {'name': 'gap', 'ruleset': 'generic', 'rules': ['following_distance']})
        assert r['path'] == 'monitors/gap' and r['rules'][0]['sha'].startswith('stl-') and r['rules'][0]['delay_s'] == 0
        assert 'monitors/gap' in REGISTRY['formal_monitor_gen'].preview(ctx, name='gap', ruleset='generic')
        assert 'unbounded future' in call(ctx, 'formal_monitor_gen', {'name': 'f', 'formula': 'F (speed_mps > 1)'})[
            'error']
        assert 'one of' in call(ctx, 'formal_monitor_gen', {'name': 'f'})['error']
        s = call(ctx, 'formal_monitor_check', {'name': 'gap'})
        assert s['status'] == 'unchecked' and s['rules'][0]['status'] == 'unchecked'
        run = os.path.relpath(bench['run'], p.workspace)
        s = call(ctx, 'formal_monitor_check', {'name': 'gap', 'runs': [run]})
        assert s['agree'] and s['status'] == 'agreed' and s['errors'] == [] and s['checked_at']
        assert 'no monitor' in call(ctx, 'formal_monitor_check', {'name': 'nope'})['error']
    finally:
        ctx.service.shutdown()


# ================================================================================================ the tab
_APP = []


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    from falcon_autonomy.ui.app import make_app
    from falcon_autonomy.ui.theme import THEME
    app = QApplication.instance() or make_app()
    _APP.append(app)
    THEME.apply(app, 'dark')
    return app


@pytest.mark.gui
def test_the_runtime_monitors_tab_offscreen(qapp, bench, tmp_path):
    from types import SimpleNamespace
    from falcon_autonomy.ui.monitor_tab import MonitorTab
    from falcon_autonomy.ui.theme import THEME
    ws = str(tmp_path / 'ws')
    os.makedirs(ws)
    win = SimpleNamespace(workspace=lambda: ws)
    tab = MonitorTab(SimpleNamespace(win=win))
    assert tab.tree.topLevelItemCount() == 0 and 'Generate' in tab.note.text() and not tab.check_btn.isEnabled()
    assert tab.gen_btn.property('role') == 'primary' and tab.check_btn.property('role') is None
    tab.ruleset.setCurrentIndex(tab.ruleset.findData('india'))
    man = tab.generate()
    assert man['name'] == 'india' and tab.package.currentText() == 'india'
    n = tab.tree.topLevelItemCount()
    assert n == len(man['rules'])
    rows = {tab.tree.topLevelItem(i).text(0): tab.tree.topLevelItem(i) for i in range(n)}
    assert rows['keep_left'].text(1) == 'NOT CHECKED' and rows['keep_left'].text(2) == '10 s'
    assert rows['keep_left'].text(3).startswith('stl-')
    text = tab.command_text()
    assert 'falcon monitor deploy' in text and 'systemctl --user enable --now falcon-monitor-india.service' in text
    assert tab.dry_cmd.command.endswith('--dry-run') and tab.rollback_cmd.command.startswith('falcon monitor rollback')
    assert tab.deploy_cmd.copy_btn.isEnabled()                       # text with a Copy button: Falcon runs none
    doc = tab.check_run(bench['run'])
    assert doc['agree']
    it = next(tab.tree.topLevelItem(i) for i in range(n) if tab.tree.topLevelItem(i).text(0) == 'speed_limit')
    assert it.text(1) == 'EQUAL' and it.foreground(1).color().name() == THEME.status('good')
    assert 'equal to the offline checker' in it.toolTip(1) and 'india: equal, checked' in tab.note.text()
    tab.grab()
    THEME.apply(qapp, 'light')                                     # the status colour follows the theme
    it = next(tab.tree.topLevelItem(i) for i in range(n) if tab.tree.topLevelItem(i).text(0) == 'speed_limit')
    assert it.foreground(1).color().name() == THEME.status('good')
    THEME.apply(qapp, 'dark')
    tab.check_run(str(tmp_path / 'nowhere'))
    assert 'could not be read' in tab.note.text()
    # the Hardware page hosts it through its one hook line
    src = open(os.path.join(os.path.dirname(MG.__file__), '..', 'ui', 'pages', 'hardware.py')).read()
    assert "optional_tab(pv, 'ui.monitor_tab', 'MonitorTab', 'Runtime monitors', self)" in src
