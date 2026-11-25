"""Runtime safety monitors generated from the rules: the formulas core.rules checks offline, compiled into an online
monitor for the vehicle computer, so the rule that failed a run in simulation is the rule watching the car
(docs/runtime-monitors.md is the manual; numpy only inside the functions that read runs).

    falcon monitor gen (--rules india|generic|FILE [--rule ID ...] | --stl FORMULA) --out DIR | --project P [--name N]
    falcon monitor check RUN... --monitor DIR [--xodr FILE] [--json]      0 agree, 1 a disagreement, 2 failed
    falcon monitor deploy DIR [--dry-run]                                 copy to the device over SSH; prints commands
    falcon monitor rollback DIR [--dry-run]                               restore the version the last deploy replaced
    falcon monitor bench (DIR | --rules SET) [--steps N]                  the per-step cost on THIS machine
    falcon monitor show DIR | falcon monitor list --project P

What is monitored. A rule of a rule set must hold at every sample (core.rules: G φ): the monitor evaluates φ at each
sample with core.stl's quantitative semantics. A free formula (--stl) is monitored the same way — at every sample; an
outer `always` without a bound (G φ) is taken as that "at every sample" and removed. Bounded future operators delay
the verdict by the formula's future horizon; an unbounded future operator inside the formula is REFUSED
(monitor_runtime.MonitorError says which), as is an unbounded since; once / historically may be unbounded. The
built-in rule sets use only once[0,c] and eventually[0,c] (keep_left / keep_right: a 10 s delay) and are all
monitorable.

The package `gen` writes (self-contained; nothing in it imports falcon_autonomy):

    falcon_monitor.py            core/monitor_runtime.py verbatim: the evaluator (standard library only)
    rules.json                   the compiled rules as data: per rule its tree, signals, horizon and hash
    falcon_monitor_node.py       core/monitor_node.py verbatim: the rclpy node (topics → signals → verdicts)
    mapping.json                 topics → signals; the user edits it (default: /falcon/signals/<name>, Float64)
    falcon-monitor-<name>.service   a systemd USER unit (no root)
    README.md, manifest.json     the manifest pins every file's SHA-256 and each rule's formula hash

Hashes. A rule's `sha` is "stl-" + the first 16 hex digits of the SHA-256 of {formula (core.stl's normal form, the
constants substituted), signals_version} in canonical JSON — the same construction as core.stlobjective.spec_key,
without the falsifier's scale and smoothing (they do not change what is monitored); `spec_key` beside it is that
falsification key itself, so a monitor and a falsification study of the same requirement can be joined.

Equivalence (`check`, the key test). A recorded run's signals (core.rules.signals of its trace) are replayed through
the GENERATED module, one sample at a time, then finish(); every sample's robustness must equal core.stl.evaluate's on
the whole run exactly (same floats; unknown where unknown), the run's verdict must be core.rules.check's, and what the
monitor decided BEFORE the end must already be the final value (causality). The result (agreement, the first
disagreement with its sample, time and both values) is written to DIR/equivalence.json.

GSN. A Solution of the safety case (core.gsn) can bind a monitor as evidence (kind `monitor`, ref the package
folder): its manifest's SHA-256 is pinned; verification is amber when a rule's formula has changed since the monitor
was generated (or no equivalence check has been made on this manifest), red when the latest check disagreed, green
when it agreed.

Limits, stated: the deployment is tested with a fake transport only (no device was reached in the tests); the node
was run against a fake rclpy, not a ROS 2 installation; the per-step cost is measured on the machine that runs
`bench`, not on the Orin; the monitor is only as good as the signals mapped to it — the rule signals (lead_gap_m,
lane_margin_m, …) are Falcon's simulator-side definitions (core.rules), and the vehicle stack must publish quantities
that mean the same. pet_s (reads the whole run) and accel_mps2 (a central difference) are non-causal offline: a
monitor of a formula that reads them is refused.
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import sys
import time

from . import monitor_runtime as RT

SCHEMA = 'falcon.monitor/1'
EQ_SCHEMA = 'falcon.monitor_equivalence/1'
NONCAUSAL = {'pet_s': 'reads the whole run (an actor passing later counts)',
             'accel_mps2': 'is a central difference (it reads the next sample)'}
FILES = ('falcon_monitor.py', 'rules.json', 'falcon_monitor_node.py', 'mapping.json', 'README.md')
REMOTE_ROOT = '~/falcon-monitor'
_NAME = re.compile(r'^[A-Za-z][\w.-]{0,47}$')
_HERE = os.path.dirname(os.path.abspath(__file__))


class MonitorGenError(ValueError):
    pass


# ================================================================================================ compiling
def tree(node) -> dict:
    """A core.stl.Node as the runtime's JSON tree (inf as "inf")."""
    def num(v):
        return ('inf' if v > 0 else '-inf') if math.isinf(v) else float(v)
    d = {'op': node.op}
    if node.args:
        d['args'] = [tree(a) for a in node.args]
    if node.op == 'num':
        d['value'] = num(node.value)
    if node.name is not None and node.op != 'num':
        d['name'] = node.name
    if node.op in RT.TEMPORAL:
        d['lo'], d['hi'] = num(node.lo), num(node.hi)
    return d


def formula_sha(node) -> str:
    from .provenance import sha256_json
    from .rules import SIGNALS_VERSION
    return 'stl-' + sha256_json({'formula': node.text(), 'signals_version': SIGNALS_VERSION})[:16]


def _secs(v: float):
    return None if math.isinf(v) else float(v)


def compile_node(node, rid: str, title: str = '') -> dict:
    """One formula φ (to hold at every sample) → the runtime's compiled rule. MonitorGenError when it cannot be
    monitored (an unbounded future operator, a non-causal signal)."""
    from . import stl as S
    used = sorted(S.names(node))
    bad = [f'{s} {NONCAUSAL[s]}' for s in used if s in NONCAUSAL]
    if bad:
        raise MonitorGenError(f'{rid}: cannot be monitored online: ' + '; '.join(bad))
    t = tree(node)
    try:
        RT.Monitor({'id': rid, 'tree': t})
    except RT.MonitorError as e:
        raise MonitorGenError(f'{rid}: {e}') from None
    past, future = S.horizon(node)
    return {'id': rid, 'title': title or rid, 'formula': node.text(), 'tree': t, 'signals': used,
            'horizon_s': {'past': _secs(past), 'future': _secs(future)}, 'sha': formula_sha(node)}


def compile_rules(ruleset, ids=None) -> tuple:
    """(compiled rules, source) of a rule set's rules (all, or `ids`)."""
    from . import rules as R
    try:
        rs = R.load_ruleset(ruleset)
    except R.RulesError as e:
        raise MonitorGenError(str(e)) from None
    by = {r['id']: r for r in rs['rules']}
    ids = list(ids or by)
    missing = [i for i in ids if i not in by]
    if missing:
        raise MonitorGenError(f'no rule {", ".join(missing)} in the set {rs["id"]} ({", ".join(by)})')
    out = []
    for i in ids:
        c = compile_node(by[i]['node'], i, by[i]['title'])
        c['tags'] = list(by[i]['tags'])
        try:
            from .stlobjective import spec_key
            if isinstance(ruleset, str) and ruleset:
                c['spec_key'] = spec_key({'rule': f'{ruleset}:{i}'})
        except Exception:                                    # noqa: BLE001 (the link is optional)
            pass
        out.append(c)
    src = {'kind': 'rules', 'ruleset': ruleset if isinstance(ruleset, str) else rs['id'], 'ruleset_id': rs['id'],
           'ruleset_sha256': rs['sha256'], 'handedness': rs.get('handedness'), 'rule_ids': ids}
    return out, src


def compile_stl(text: str, rid: str = 'requirement', consts: dict | None = None) -> tuple:
    """(compiled rules, source) of a free formula; an outer unbounded `always` is the "at every sample" (module doc)."""
    from . import rules as R
    from . import stl as S
    try:
        node = R.parse_formula(text, consts)
    except (S.STLError, R.RulesError) as e:
        raise MonitorGenError(str(e)) from None
    whole = node.text()
    stripped = False
    if node.op == 'G' and node.lo == 0 and math.isinf(node.hi):
        node, stripped = node.args[0], True
    c = compile_node(node, rid, rid)
    try:
        from .stlobjective import spec_key
        c['spec_key'] = spec_key({'stl': whole})
    except Exception:                                        # noqa: BLE001
        pass
    return [c], {'kind': 'stl', 'stl': whole, 'outer_always_removed': stripped}


def default_mapping(rules: list, rate_hz: float = 20.0) -> dict:
    """The default mapping: every signal the rules read on /falcon/signals/<name> (std_msgs/msg/Float64, field
    data); speed_kmh derived from speed_mps when both are read or only speed_kmh is; t is the node's own clock."""
    names = sorted({s for r in rules for s in r['signals']} - {'t'})
    sig = {}
    for n in names:
        sig[n] = {'topic': f'/falcon/signals/{n}', 'type': 'std_msgs/msg/Float64', 'field': 'data'}
    if 'speed_kmh' in sig:
        sig.setdefault('speed_mps', {'topic': '/falcon/signals/speed_mps', 'type': 'std_msgs/msg/Float64',
                                     'field': 'data'})
        sig['speed_kmh'] = {'from': 'speed_mps', 'scale': 3.6}
    return {'schema': 'falcon.monitor_mapping/1', 'rate_hz': float(rate_hz), 'stale_s': 0.5,
            'alarm_topic': '/falcon/monitor/alarm', 'verdict_topic': '/falcon/monitor/verdict',
            'about': ('signals: NAME → {topic, type, field, scale, offset} (a message field, value*scale+offset) | '
                      '{from: NAME, scale, offset} (derived) | {const: number}. Examples: {"topic": "/odom", "type": '
                      '"nav_msgs/msg/Odometry", "field": "twist.twist.linear.x"}; Autoware: {"topic": '
                      '"/vehicle/status/velocity_status", "type": "autoware_auto_vehicle_msgs/msg/VelocityReport", '
                      '"field": "longitudinal_velocity"}. A signal older than stale_s seconds is unknown.'),
            'signals': sig}


def unit_text(name: str) -> str:
    d = f'%h/falcon-monitor/{name}'
    return f'''[Unit]
Description=Falcon runtime monitor ({name})
After=network-online.target

[Service]
Type=simple
WorkingDirectory={d}
ExecStart=/bin/bash -c 'source /opt/ros/humble/setup.bash && exec python3 {d}/falcon_monitor_node.py'
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
'''


def install_commands(name: str) -> list:
    """What the user runs on the device to install and start the unit (Falcon never runs them, and never sudo)."""
    d, unit = f'{REMOTE_ROOT}/{name}', f'falcon-monitor-{name}.service'
    return [f'mkdir -p ~/.config/systemd/user && cp {d}/{unit} ~/.config/systemd/user/',
            'systemctl --user daemon-reload',
            f'systemctl --user enable --now {unit}',
            f'systemctl --user status {unit}',
            '# to keep it running without a login session (needs root; run it yourself):',
            'sudo loginctl enable-linger "$USER"']


def readme_text(man: dict) -> str:
    rows = '\n'.join(f'| `{r["id"]}` | {r["title"]} | `{r["sha"]}` | '
                     f'{0 if not r["horizon_s"]["future"] else r["horizon_s"]["future"]:g} s | '
                     f'{", ".join(r["signals"])} |' for r in man['rules'])
    cmds = '\n'.join('    ' + c for c in install_commands(man['name']))
    return f'''# Falcon runtime monitor: {man['name']}

Generated by `falcon monitor gen` ({man['created']}). It evaluates, online, the rules below with the same
quantitative semantics Falcon's offline checker uses; `falcon monitor check RUN --monitor <this folder>` on the
machine that generated it compares the two sample by sample.

| rule | title | formula hash | verdict delay | signals |
|---|---|---|---|---|
{rows}

Files: `falcon_monitor.py` (the evaluator, Python standard library only), `rules.json` (the compiled rules),
`falcon_monitor_node.py` (the ROS 2 node, rclpy), `mapping.json` (topics → signals: EDIT THIS for your stack),
`falcon-monitor-{man['name']}.service` (a systemd user unit), `manifest.json` (hashes).

## Signals

Each signal is a number per sample; the default mapping expects `std_msgs/msg/Float64` on `/falcon/signals/<name>`.
Point an entry at your stack's own topic and field instead (`mapping.json` explains the forms). The signals are
Falcon's definitions (see docs/traffic-rules.md in Falcon): what you publish must mean the same thing. A signal that
is not mapped, never received or stale is *unknown*: it never raises an alarm, and the rule reports `unknown`.

## Outputs

- `/falcon/monitor/alarm` (`std_msgs/msg/Bool`): true at a step where a rule is violated.
- `/falcon/monitor/verdict` (`std_msgs/msg/String`, JSON): per rule `verdict` (ok, violated, unknown, pending),
  `robustness`, the worst robustness so far, the violated samples so far and the samples still pending.
- `/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`): one status per rule (OK, WARN, ERROR).

A rule with a verdict delay reports sample *t* only once *t + delay* has passed: until then it is `pending`.

## Run

    source /opt/ros/humble/setup.bash
    python3 falcon_monitor_node.py            # --ros-args -p use_sim_time:=true   in simulation

As a service (a user unit; the folder must be `{REMOTE_ROOT}/{man['name']}`, where `falcon monitor deploy` puts it):

{cmds}

Use in Python without ROS:

    import falcon_monitor as fm
    mon = fm.MonitorSet(fm.load_rules('rules.json'))
    out = mon.step(t_seconds, {{'speed_mps': 4.2, 'lead_gap_m': 11.0}})   # out['alarm'], out['rules'][id]['verdict']

## Limits

This monitor is evidence, not a safety mechanism qualified to any standard. It watches only what its signals carry.
It was generated and checked on a development machine; its cost on this computer has not been measured by Falcon.
'''


def manifest_hash(man: dict) -> str:
    from .provenance import sha256_json
    return sha256_json({'rules': [r['sha'] for r in man['rules']], 'files': man['files'],
                        'runtime': man['runtime']})


def generate(out_dir: str, rules: list, source: dict, name: str = '', rate_hz: float = 20.0) -> dict:
    """Write the package (module doc) into out_dir and return its manifest. An existing mapping.json is KEPT (the
    user edits it) unless it no longer parses."""
    from .. import __version__
    from .provenance import sha256_file, utc_now, write_json
    from .rules import SIGNALS_VERSION
    name = name or os.path.basename(os.path.abspath(out_dir))
    if not _NAME.match(name):
        raise MonitorGenError(f'the monitor\'s name {name!r}: a letter, then letters, digits, . _ - (at most 48)')
    if not rules:
        raise MonitorGenError('no rules to monitor')
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(os.path.join(_HERE, 'monitor_runtime.py'), os.path.join(out_dir, 'falcon_monitor.py'))
    shutil.copyfile(os.path.join(_HERE, 'monitor_node.py'), os.path.join(out_dir, 'falcon_monitor_node.py'))
    write_json(os.path.join(out_dir, 'rules.json'), {'schema': 'falcon.monitor_rules/1', 'name': name,
                                                      'rules': rules})
    mp = os.path.join(out_dir, 'mapping.json')
    kept = False
    if os.path.isfile(mp):
        try:
            with open(mp, encoding='utf-8') as fh:
                RT.SignalMapper(json.load(fh))
            kept = True
        except (OSError, ValueError):
            kept = False
    if not kept:
        write_json(mp, default_mapping(rules, rate_hz))
    unit = f'falcon-monitor-{name}.service'
    with open(os.path.join(out_dir, unit), 'w', encoding='utf-8') as fh:
        fh.write(unit_text(name))
    man = {'schema': SCHEMA, 'name': name, 'created': utc_now(), 'falcon': __version__, 'runtime': RT.VERSION,
           'signals_version': SIGNALS_VERSION, 'source': source,
           'rules': [{k: r.get(k) for k in ('id', 'title', 'formula', 'sha', 'spec_key', 'signals', 'horizon_s',
                                            'tags')} for r in rules],
           'unit': unit, 'mapping_kept': kept, 'files': {},
           'notes': ['the deployment and the ROS 2 node are tested with fakes only (docs/runtime-monitors.md)']}
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as fh:
        fh.write(readme_text(man))
    # mapping.json is the user's to edit: it is not pinned
    man['files'] = {f: sha256_file(os.path.join(out_dir, f)) for f in FILES + (unit,) if f != 'mapping.json'}
    man['hash'] = manifest_hash(man)
    write_json(os.path.join(out_dir, 'manifest.json'), man)
    eq = os.path.join(out_dir, 'equivalence.json')
    if os.path.isfile(eq):                                 # a check of an older generation says nothing of this one
        try:
            with open(eq, encoding='utf-8') as fh:
                if json.load(fh).get('manifest_hash') != man['hash']:
                    os.remove(eq)
        except (OSError, ValueError):
            pass
    return man


# ================================================================================================ the package
def read_manifest(pkg: str) -> dict:
    p = os.path.join(pkg, 'manifest.json') if os.path.isdir(pkg) else pkg
    try:
        with open(p, encoding='utf-8') as fh:
            man = json.load(fh)
    except (OSError, ValueError) as e:
        raise MonitorGenError(f'no monitor package at {pkg} ({e})') from None
    if man.get('schema') != SCHEMA:
        raise MonitorGenError(f'{p} is not a monitor manifest ({SCHEMA})')
    return man


def changed_files(pkg: str, man: dict) -> list:
    """The pinned files whose bytes are no longer the manifest's (or are missing)."""
    from .provenance import sha256_file
    out = []
    for f, sha in (man.get('files') or {}).items():
        p = os.path.join(pkg, f)
        if not os.path.isfile(p) or sha256_file(p) != sha:
            out.append(f)
    return out


def load_package(pkg: str) -> tuple:
    """(the package's own falcon_monitor module, its rules document, its manifest). The files' hashes are checked
    against the manifest first: a changed file is refused, so what runs here is what `gen` wrote."""
    import importlib.util
    man = read_manifest(pkg)
    bad = changed_files(pkg, man)
    if bad:
        raise MonitorGenError(f'{pkg}: changed since it was generated: {", ".join(bad)} (generate it again)')
    spec = importlib.util.spec_from_file_location(f'_falcon_monitor_{abs(hash(os.path.abspath(pkg)))}',
                                                  os.path.join(pkg, 'falcon_monitor.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.load_rules(os.path.join(pkg, 'rules.json')), man


def current_shas(man: dict) -> dict:
    """{rule id: the formula hash the rule has NOW (None: the rule or its set is gone)} — a free formula is its own
    source and cannot change."""
    src = man.get('source') or {}
    if src.get('kind') != 'rules':
        return {r['id']: r['sha'] for r in man['rules']}
    from . import rules as R
    try:
        rs = R.load_ruleset(src.get('ruleset'))
    except (R.RulesError, OSError):
        return {r['id']: None for r in man['rules']}
    by = {r['id']: r for r in rs['rules']}
    return {r['id']: (formula_sha(by[r['id']]['node']) if r['id'] in by else None) for r in man['rules']}


def stale_rules(man: dict) -> list:
    """The rules whose formula changed (or vanished) since the monitor was generated."""
    now = current_shas(man)
    return [r['id'] for r in man['rules'] if now.get(r['id']) != r['sha']]


# ================================================================================================ equivalence
def _same(a: float, b: float) -> bool:
    return (a != a and b != b) or a == b


def replay(mod, rule: dict, sig: dict, t) -> dict:
    """One compiled rule stepped over a trace by the module `mod` (a generated falcon_monitor, or monitor_runtime):
    {'series': per-sample robustness as decided, 'early': how many samples were decided before the end, 'state',
    'worst', 'violations', 'max_pending'}."""
    names = [k for k in rule['signals'] if k in sig]
    cols = {k: [float(x) for x in sig[k]] for k in names}
    m = mod.Monitor(rule)
    series, max_pending = [], 0
    for i, ti in enumerate(t):
        r = m.step(float(ti), {k: cols[k][i] for k in names})
        series.extend(v for _i, _t, v in r['decided'])
        max_pending = max(max_pending, r['pending'])
    early = len(series)
    series.extend(v for _i, _t, v in m.finish()['decided'])
    return {'series': series, 'early': early, 'state': m.state(), 'worst': m.worst, 'violations': m.violations,
            'max_pending': max_pending}


def compare(mod, rule: dict, sig: dict, t) -> dict:
    """The online monitor against core.stl.evaluate on the same samples: exact equality per sample."""
    import numpy as np
    from . import stl as S
    node = S.parse(rule['formula'])
    off = S.evaluate(node, sig, t)
    on = replay(mod, rule, sig, t)
    first, diff = None, 0.0
    n = len(off)
    if len(on['series']) != n:
        first = {'i': min(len(on['series']), n), 't': None, 'online': None, 'offline': None,
                 'why': f'the monitor decided {len(on["series"])} of {n} samples'}
    else:
        for i in range(n):
            a, b = on['series'][i], float(off[i])
            if not _same(a, b):
                if first is None:
                    first = {'i': i, 't': float(t[i]), 'online': _j(a), 'offline': _j(b)}
                if a == a and b == b and math.isfinite(a) and math.isfinite(b):
                    diff = max(diff, abs(a - b))
                else:
                    diff = math.inf
    known = off[~np.isnan(off)]
    off_state = 'not evaluable' if not len(known) else 'violated' if float(known.min()) < 0 else 'satisfied'
    off_rob = float(known.min()) if len(known) else None
    agree = first is None and on['state'] == off_state and (
        (on['worst'] is None and off_rob is None) or (on['worst'] is not None and off_rob is not None
                                                      and on['worst'] == off_rob))
    return {'agree': bool(agree), 'samples': n, 'decided_before_end': on['early'], 'max_pending': on['max_pending'],
            'first_disagreement': first, 'max_abs_diff': _j(diff), 'online_state': on['state'],
            'offline_state': off_state, 'online_robustness': _j(on['worst']), 'offline_robustness': _j(off_rob),
            'violated_samples': on['violations']}


def _j(v):
    if v is None:
        return None
    v = float(v)
    return None if v != v else ('inf' if v == math.inf else '-inf' if v == -math.inf else v)


def check_trace(pkg: str, tr, loaded: tuple | None = None) -> dict:
    """The package's monitors against the offline checker on one trace (core.rules.Trace)."""
    from . import rules as R
    mod, rules, man = loaded or load_package(pkg)
    sig = R.signals(tr)
    out = {'samples': int(len(tr.t)), 'rules': {}}
    if len(tr.t) > 1:
        import numpy as np
        if float(np.min(np.diff(tr.t))) <= RT.MIN_STEP_S:
            raise MonitorGenError(f'the run\'s sample times do not increase by more than {RT.MIN_STEP_S:g} s')
    for rule in rules['rules']:
        out['rules'][rule['id']] = compare(mod, rule, sig, tr.t)
    # the run as core.rules.check judges it, when the monitored rules are still the set's
    src = man.get('source') or {}
    if src.get('kind') == 'rules' and not stale_rules(man):
        res = R.check(tr, R.load_ruleset(src['ruleset']))
        by = {r['id']: r for r in res['rules']}
        for rid, row in out['rules'].items():
            c = by.get(rid)
            if c is None:
                continue
            row['check_state'] = c['state']
            if c['state'] != row['online_state']:
                row['agree'] = False
                row['first_disagreement'] = row['first_disagreement'] or {
                    'i': None, 't': None, 'online': row['online_state'], 'offline': c['state'],
                    'why': 'the run\'s verdict differs from core.rules.check'}
    out['agree'] = all(r['agree'] for r in out['rules'].values())
    return out


def check(pkg: str, runs: list, xodr: str | None = None, write: bool = True) -> dict:
    """`falcon monitor check`: every run replayed through the generated monitor and compared (module doc); writes
    DIR/equivalence.json."""
    from . import rules as R
    from .provenance import utc_now, write_json
    loaded = load_package(pkg)
    man = loaded[2]
    doc = {'schema': EQ_SCHEMA, 'at': utc_now(), 'monitor': man['name'], 'manifest_hash': man['hash'],
           'stale_rules': stale_rules(man), 'runs': [],
           'method': 'the generated falcon_monitor.py stepped sample by sample over core.rules.signals of each '
                     'run, then finish(); compared with core.stl.evaluate per sample (exact) and with '
                     'core.rules.check\'s verdict'}
    for run in runs:
        try:
            tr = R.trace_from_run(run, xodr)
            row = check_trace(pkg, tr, loaded)
        except (R.RulesError, OSError, ValueError, KeyError) as e:
            row = {'error': str(e), 'agree': False, 'rules': {}, 'samples': 0}
        row['run'] = os.path.abspath(run)
        doc['runs'].append(row)
    per = {}
    for r in man['rules']:
        rows = [(x['run'], x['rules'][r['id']]) for x in doc['runs'] if r['id'] in x.get('rules', {})]
        bad = next(((run, y) for run, y in rows if not y['agree']), None)
        per[r['id']] = {'agree': bool(rows) and bad is None, 'runs': len(rows),
                        'samples': sum(y['samples'] for _r, y in rows),
                        'first_disagreement': None if bad is None else dict(bad[1]['first_disagreement'] or {},
                                                                            run=bad[0])}
    doc['rules'] = per
    doc['agree'] = bool(doc['runs']) and all(x['agree'] for x in doc['runs'])
    from . import fidelity as F                     # an offline replay of recorded runs (4.2), theirs beside it
    doc['fidelity'] = 'recorded'
    doc['source_fidelity'] = F.combine(F.of_run(x['run']) for x in doc['runs'] if not x.get('error'))
    if write:
        write_json(os.path.join(pkg, 'equivalence.json'), doc)
    return doc


def read_equivalence(pkg: str) -> dict | None:
    try:
        with open(os.path.join(pkg, 'equivalence.json'), encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def status(pkg: str) -> dict:
    """A package for the page, the tools and the safety case: per rule {'status', 'word', 'why'} with status
    agreed | disagreed | unchecked | stale, and the package's own (the worst)."""
    man = read_manifest(pkg)
    stale = set(stale_rules(man))
    eq = read_equivalence(pkg)
    fresh = eq is not None and eq.get('manifest_hash') == man['hash']
    rows = []
    for r in man['rules']:
        e = ((eq or {}).get('rules') or {}).get(r['id']) if fresh else None
        if r['id'] in stale:
            st, why = 'stale', 'the rule\'s formula changed since the monitor was generated: generate it again'
        elif e is None or not e.get('runs'):
            st, why = 'unchecked', 'no equivalence check on this monitor yet (falcon monitor check)'
        elif e['agree']:
            st, why = 'agreed', f'equal to the offline checker on {e["samples"]} samples of {e["runs"]} run(s)'
        else:
            fd = e.get('first_disagreement') or {}
            st = 'disagreed'
            why = (f'differs from the offline checker at sample {fd.get("i")} (t = {fd.get("t")} s): online '
                   f'{fd.get("online")}, offline {fd.get("offline")}')
        rows.append({'id': r['id'], 'title': r['title'], 'sha': r['sha'], 'status': st, 'why': why,
                     'delay_s': (r.get('horizon_s') or {}).get('future') or 0.0, 'signals': r['signals']})
    order = ('disagreed', 'stale', 'unchecked', 'agreed')
    worst = min((x['status'] for x in rows), key=order.index)
    return {'name': man['name'], 'path': os.path.abspath(pkg), 'hash': man['hash'], 'created': man['created'],
            'status': worst, 'rules': rows, 'checked_at': (eq or {}).get('at') if fresh else None,
            'changed_files': changed_files(pkg, man), 'source': man.get('source'),
            'check_fidelity': (eq or {}).get('source_fidelity') or 'unknown' if fresh else ''}


def gsn_evidence(manifest_path: str) -> dict:
    """core.gsn's reading of a bound monitor: {'status': green | amber | red, 'why'}."""
    pkg = os.path.dirname(os.path.abspath(manifest_path))
    try:
        s = status(pkg)
    except MonitorGenError as e:
        return {'status': 'amber', 'why': str(e)}
    by = {}
    for r in s['rules']:
        by.setdefault(r['status'], []).append(r['id'])
    if s['changed_files']:
        return {'status': 'amber', 'why': f'the package\'s files changed since it was generated: '
                                          f'{", ".join(s["changed_files"])}'}
    if by.get('disagreed'):
        r = next(x for x in s['rules'] if x['status'] == 'disagreed')
        return {'status': 'red', 'why': f'{r["id"]}: the monitor {r["why"]}'}
    if by.get('stale'):
        return {'status': 'amber', 'why': f'the formula of {", ".join(by["stale"])} changed since the monitor was '
                                          'generated: generate, check and re-bind'}
    if by.get('unchecked'):
        return {'status': 'amber', 'why': f'no equivalence check on this monitor yet ({", ".join(by["unchecked"])}): '
                                          'falcon monitor check'}
    n = len(s['rules'])
    return {'status': 'green', 'why': f'{n} rule(s) monitored; the monitor equals the offline checker '
                                      f'(checked {s["checked_at"]})'}


def list_packages(workspace: str) -> list:
    d = os.path.join(workspace, 'monitors')
    out = []
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if os.path.isfile(os.path.join(d, n, 'manifest.json')):
                out.append(os.path.join(d, n))
    return out


# ================================================================================================ deployment
class SshTransport:
    """The device over SSH with the user's key, through hil.deploy (BatchMode, keys only; nothing waits for a
    prompt). put() copies one file into the remote folder; ssh() runs one command and returns its output."""

    def __init__(self, dev: dict):
        from ..hil import deploy as HD
        self.HD, self.dev = HD, dev
        HD.validate(dev)

    def ssh(self, remote: str) -> str:
        return self.HD.ssh(self.dev, remote)

    def put(self, local: str, remote_dir: str, name: str):
        HD, dev = self.HD, self.dev
        target = remote_dir[2:] if remote_dir.startswith('~/') else remote_dir
        cmd = [shutil.which('scp') or 'scp', '-P', str(int(dev.get('ssh_port') or 22))] + HD._opts(dev) + \
            ['-q', local, f'{dev["ssh_user"]}@{dev["address"]}:{target.rstrip("/")}/{name}']
        r = HD.run(cmd, 120.0)
        if r.returncode != 0:
            raise HD.DeployError(f'copying {name} failed: {HD._clean(r.stderr or r.stdout, dev)}')


def previous_dir(name: str) -> str:
    """Where deploy keeps the version it replaces on the device (rollback swaps it back)."""
    return f'{REMOTE_ROOT}/.previous/{name}'


def _remote(path: str) -> str:
    """A ~/… device path for a shell command: $HOME quoted apart, the rest shell-quoted."""
    return '"$HOME"/' + shlex.quote(path[2:]) if path.startswith('~/') else shlex.quote(path)


def validate_package(pkg: str) -> dict:
    """Everything deploy needs, checked HERE without a device or a network (`deploy --dry-run`): the manifest
    (schema, name, hash), every pinned file present with the manifest's SHA-256, mapping.json read by the evaluator's
    own SignalMapper, rules.json loaded by the package's own falcon_monitor.py, and the systemd unit equal to the
    one `gen` writes (its ExecStart and WorkingDirectory in the folder deploy copies to, no root, a user unit).
    Returns {'ok', 'checks': [{'name', 'ok', 'detail'}], 'files': [{'name', 'bytes', 'sha256'}], 'name',
    'remote_dir', 'previous_dir', 'unit', 'commands', 'manifest_hash'}."""
    from .provenance import sha256_file
    checks = []

    def ck(name, ok, detail=''):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
        return ok
    try:
        man = read_manifest(pkg)
    except MonitorGenError as e:
        ck('manifest', False, str(e))
        return {'ok': False, 'checks': checks, 'files': [], 'name': '', 'remote_dir': '', 'previous_dir': '',
                'unit': '', 'commands': [], 'manifest_hash': ''}
    name = str(man.get('name') or '')
    ck('manifest', bool(_NAME.match(name)) and man.get('hash') == manifest_hash(man),
       f'{SCHEMA}, name {name!r}, hash {str(man.get("hash"))[:16]}' if man.get('hash') == manifest_hash(man)
       else 'the manifest\'s hash is not the hash of its contents: generate the package again')
    bad = changed_files(pkg, man)
    ck('pinned files', not bad, f'{len(man.get("files") or {})} files match the manifest\'s SHA-256' if not bad
       else 'changed or missing since it was generated: ' + ', '.join(bad))
    try:
        with open(os.path.join(pkg, 'mapping.json'), encoding='utf-8') as fh:
            mp = RT.SignalMapper(json.load(fh))
        ck('mapping.json', True, f'{len(mp.entries)} signal(s) mapped (yours to edit; not pinned)')
    except (OSError, ValueError, RT.MonitorError) as e:
        ck('mapping.json', False, f'cannot be read by the monitor: {e}')
    if not bad:
        try:
            mod, rd, _m = load_package(pkg)
            n = len(mod.MonitorSet(rd).monitors)
            ck('rules.json', n == len(man['rules']), f'the package\'s own evaluator loads {n} rule(s)')
        except Exception as e:                                     # noqa: BLE001 (what the device would hit)
            ck('rules.json', False, f'the package\'s evaluator cannot load it: {type(e).__name__}: {e}')
    unit = man.get('unit') or f'falcon-monitor-{name}.service'
    try:
        with open(os.path.join(pkg, unit), encoding='utf-8') as fh:
            text = fh.read()
        want = f'%h/falcon-monitor/{name}'
        ok = text == unit_text(name)
        why = 'as gen writes it: a user unit, ' + f'WorkingDirectory {want}, no root' if ok else \
            'differs from the unit gen writes for this name (generate again)'
        if 'sudo' in text or 'User=root' in text:
            ok, why = False, 'the unit asks for root'
        ck('systemd unit', ok, f'{unit}: {why}')
    except OSError as e:
        ck('systemd unit', False, f'{unit} cannot be read: {e}')
    files = []
    for f in list(man.get('files') or {}) + ['mapping.json', 'manifest.json']:
        fp = os.path.join(pkg, f)
        if os.path.isfile(fp) and f not in [x['name'] for x in files]:
            files.append({'name': f, 'bytes': os.path.getsize(fp), 'sha256': sha256_file(fp)})
    rd = f'{REMOTE_ROOT}/{name}'
    return {'ok': all(c['ok'] for c in checks), 'checks': checks, 'files': files, 'name': name, 'remote_dir': rd,
            'previous_dir': previous_dir(name), 'unit': unit, 'commands': install_commands(name),
            'manifest_hash': man.get('hash', '')}


def deploy(pkg: str, transport=None, dev: dict | None = None, dry: bool = False) -> dict:
    """Copy the package to ~/falcon-monitor/<name> on the device and load it there once with the device's Python
    (no ROS needed for that): {'remote_dir', 'copied', 'selftest', 'commands', 'previous'}. The version already
    there (if any) is first kept in ~/falcon-monitor/.previous/<name> (rollback() restores it). Nothing is installed,
    enabled or started, and nothing runs as root: `commands` is what the user runs there. dry: validate_package()
    only — no SSH, no network."""
    v = validate_package(pkg)
    if dry:
        return dict(v, files=[f['name'] for f in v['files']], file_info=v['files'], copied=[], selftest=None,
                    dry_run=True, previous=None)
    man = read_manifest(pkg)
    bad = changed_files(pkg, man)
    if bad:
        raise MonitorGenError(f'{pkg}: changed since it was generated: {", ".join(bad)}')
    name = man['name']
    rd, prev = f'{REMOTE_ROOT}/{name}', previous_dir(name)
    files = [f for f in list(man['files']) + ['mapping.json', 'manifest.json']
             if os.path.isfile(os.path.join(pkg, f))]
    out = {'remote_dir': rd, 'previous_dir': prev, 'files': files, 'copied': [], 'selftest': None,
           'commands': install_commands(name), 'dry_run': False, 'previous': None}
    if transport is None:
        transport = SshTransport(dev or {})
    q, qp = _remote(rd), _remote(prev)
    kept = transport.ssh(f'if [ -d {q} ]; then rm -rf {qp} && mkdir -p "$(dirname {qp})" && cp -a {q} {qp} && '
                         f'echo falcon-kept; else echo falcon-none; fi')
    out['previous'] = prev if 'falcon-kept' in (kept or '') else None
    transport.ssh(f'mkdir -p {q}')
    for f in files:
        transport.put(os.path.join(pkg, f), rd, f)
        out['copied'].append(f)
    out['selftest'] = _selftest(transport, q, dev)
    if 'falcon-monitor ok' not in out['selftest']:
        raise MonitorGenError(f'the device\'s Python did not load the monitor: {out["selftest"][:200]}')
    return out


def _selftest(transport, quoted_dir: str, dev: dict | None) -> str:
    py = shlex.quote(str((dev or {}).get('python') or 'python3'))
    test = ('import falcon_monitor as m; s = m.MonitorSet(m.load_rules("rules.json")); '
            'print("falcon-monitor ok", len(s.monitors))')
    got = transport.ssh(f'cd {quoted_dir} && {py} -c {shlex.quote(test)}')
    return got.strip().splitlines()[-1] if got and got.strip() else ''


def rollback(pkg: str, transport=None, dev: dict | None = None, dry: bool = False) -> dict:
    """Put back the version deploy kept on the device (4.2): ~/falcon-monitor/<name> and .previous/<name> are swapped
    (so a second rollback returns to the newer one), the restored package is loaded once with the device's Python,
    and the user unit is restarted with `systemctl --user try-restart` — which restarts it only if it is running and
    starts nothing that was stopped; no root. Like deploy, SSH is used only when the user runs it. dry: the plan
    only. Returns {'remote_dir', 'previous_dir', 'restored' (the restored manifest's hash, or ''), 'selftest',
    'restart', 'commands'}."""
    man = read_manifest(pkg)
    name = man['name']
    rd, prev = f'{REMOTE_ROOT}/{name}', previous_dir(name)
    unit = man.get('unit') or f'falcon-monitor-{name}.service'
    q, qp, qt = _remote(rd), _remote(prev), _remote(f'{REMOTE_ROOT}/.rollback-{name}')
    swap = (f'[ -d {qp} ] || {{ echo falcon-no-previous; exit 0; }}; rm -rf {qt} && '
            f'{{ [ -d {q} ] && mv {q} {qt} || true; }} && mv {qp} {q} && {{ [ -d {qt} ] && mv {qt} {qp} || true; }} '
            f'&& echo falcon-swapped')
    restart = f'systemctl --user try-restart {shlex.quote(unit)}'
    out = {'remote_dir': rd, 'previous_dir': prev, 'unit': unit, 'restored': '', 'selftest': None, 'restart': None,
           'dry_run': bool(dry), 'plan': [swap, f'cd {q} && <the device\'s python> -c <load the rules>', restart],
           'commands': [f'systemctl --user status {unit}']}
    if dry:
        return out
    if transport is None:
        transport = SshTransport(dev or {})
    got = transport.ssh(swap) or ''
    if 'falcon-swapped' not in got:
        raise MonitorGenError(f'nothing to roll back to: the device has no {prev} (deploy keeps the version it '
                              'replaces from 4.2 on)')
    try:
        out['restored'] = json.loads(transport.ssh(f'cat {q}/manifest.json') or '{}').get('hash', '')
    except ValueError:
        out['restored'] = ''
    out['selftest'] = _selftest(transport, q, dev)
    if 'falcon-monitor ok' not in out['selftest']:
        raise MonitorGenError(f'the restored monitor does not load on the device: {out["selftest"][:200]} (the '
                              'newer one is kept in ' + prev + ': roll back again to return to it)')
    out['restart'] = (transport.ssh(restart) or '').strip()
    return out


# ================================================================================================ the cost
def bench(rules: list, steps: int = 20000, seed: int = 0, mod=None) -> dict:
    """The per-step cost of the monitors on THIS machine: every rule stepped over `steps` synthetic samples at 20 Hz
    (each signal a seeded random walk, 2 in 100 samples unknown), timed with perf_counter_ns around each step of the
    whole set. Returns microseconds: median, p95, p99, max, mean."""
    import platform
    import random
    mod = mod or RT
    rnd = random.Random(seed)
    ms = mod.MonitorSet({'rules': rules})
    names = ms.signals
    val = {k: rnd.uniform(0, 10) for k in names}
    samples = []
    for _ in range(steps):
        s = {}
        for k in names:
            val[k] += rnd.gauss(0, 0.5)
            s[k] = math.nan if rnd.random() < 0.02 else val[k]
        samples.append(s)
    cost = []
    clock = time.perf_counter_ns
    for i, s in enumerate(samples):
        t0 = clock()
        ms.step(i * 0.05, s)
        cost.append(clock() - t0)
    cost.sort()

    def q(p):
        return cost[min(int(p * len(cost)), len(cost) - 1)] / 1000.0
    return {'steps': steps, 'rules': len(rules), 'median_us': q(0.5), 'p95_us': q(0.95), 'p99_us': q(0.99),
            'max_us': cost[-1] / 1000.0, 'mean_us': sum(cost) / len(cost) / 1000.0,
            'python': platform.python_version(), 'machine': platform.machine(),
            'where': 'this machine (not the vehicle computer)'}


# ================================================================================================ the command
def _out_dir(a) -> str:
    if a.out:
        return os.path.abspath(a.out)
    if a.project:
        from .project import Project
        ws = Project.load(a.project).workspace
        return os.path.join(ws, 'monitors', a.name or (a.rules if a.rules in ('india', 'generic') else 'monitor'))
    raise MonitorGenError('give --out DIR (or --project P: <workspace>/monitors/<name>)')


def table(s: dict) -> str:
    L = [f'{s["name"]}  {s["status"]}  ({s["path"]})']
    for r in s['rules']:
        L.append(f'  {r["id"]:22s} {r["status"]:10s} {r["sha"]}  delay {r["delay_s"]:g} s  {r["why"]}')
    return '\n'.join(L)


def main_monitor(argv) -> int:
    """`falcon monitor gen|check|deploy|bench|show|list …` (module doc): 0 ok, 1 the monitor disagrees with the
    offline checker (check), 2 failed."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon monitor', description='runtime monitors generated from the rules '
                                                                     '(docs/runtime-monitors.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    g = sub.add_parser('gen', help='write a monitor package for a rule set\'s rules or a formula')
    g.add_argument('--rules', default='', help='india, generic or a rule file')
    g.add_argument('--rule', action='append', default=[], help='a rule id of the set (repeatable; default: all)')
    g.add_argument('--stl', default='', help='a free formula over the rule signals, monitored at every sample')
    g.add_argument('--name', default='', help='the monitor\'s name (default: the folder\'s)')
    g.add_argument('--out', default='', help='the package folder')
    g.add_argument('--project', default='', help='or a project: <workspace>/monitors/<name>')
    g.add_argument('--rate', type=float, default=20.0, help='the node\'s sampling rate in Hz (default 20)')
    c = sub.add_parser('check', help='replay runs through the generated monitor and compare with the offline checker')
    c.add_argument('runs', nargs='+', help='recorded run folders')
    c.add_argument('--monitor', required=True, help='the package folder')
    c.add_argument('--xodr', default=None, help='the map\'s OpenDRIVE file ("none": no file)')
    c.add_argument('--json', action='store_true')
    d = sub.add_parser('deploy', help='copy the package to the device over SSH (settings: the HIL device); prints '
                                      'the systemctl commands, runs none')
    d.add_argument('monitor')
    d.add_argument('--dry-run', action='store_true', help='validate the package here (files, unit, manifest) and '
                                                          'print what would be copied and enabled; no SSH')
    rb = sub.add_parser('rollback', help='restore the version the last deploy replaced on the device and restart '
                                         'its user unit if it is running (over SSH; no root)')
    rb.add_argument('monitor')
    rb.add_argument('--dry-run', action='store_true', help='print what would be run on the device; no SSH')
    b = sub.add_parser('bench', help='the monitor\'s per-step cost on this machine')
    b.add_argument('monitor', nargs='?', default='')
    b.add_argument('--rules', default='', help='a rule set instead of a package')
    b.add_argument('--steps', type=int, default=20000)
    b.add_argument('--json', action='store_true')
    s = sub.add_parser('show', help='a package: its rules, hashes and equivalence status')
    s.add_argument('monitor')
    s.add_argument('--json', action='store_true')
    ls = sub.add_parser('list', help='the workspace\'s monitor packages')
    ls.add_argument('--project', required=True)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    from ..hil.deploy import DeployError
    from .project import ProjectError
    try:
        if a.action == 'gen':
            if bool(a.rules) == bool(a.stl):
                raise MonitorGenError('give --rules SET or --stl FORMULA, one of them')
            out = _out_dir(a)
            if a.rules:
                rules, src = compile_rules(a.rules, a.rule or None)
            else:
                rules, src = compile_stl(a.stl, a.name or os.path.basename(out))
            man = generate(out, rules, src, a.name, a.rate)
            print(f'monitor {man["name"]}: {len(man["rules"])} rule(s) in {out}')
            for r in man['rules']:
                fut = (r['horizon_s'] or {}).get('future') or 0
                print(f'  {r["id"]:22s} {r["sha"]}  verdict delay {fut:g} s  reads {", ".join(r["signals"])}')
            print(f'manifest {man["hash"][:16]}; next: falcon monitor check RUN --monitor {out}')
            return 0
        if a.action == 'check':
            doc = check(a.monitor, a.runs, '' if a.xodr == 'none' else a.xodr)
            if a.json:
                print(json.dumps(doc, indent=1))
            else:
                for rid, r in doc['rules'].items():
                    word = 'agrees' if r['agree'] else 'DISAGREES'
                    line = f'{rid:22s} {word}  {r["samples"]} samples, {r["runs"]} run(s)'
                    fd = r.get('first_disagreement')
                    if fd:
                        line += (f'; first at sample {fd.get("i")} (t = {fd.get("t")} s) of {fd.get("run")}: online '
                                 f'{fd.get("online")}, offline {fd.get("offline")}'
                                 + (f' ({fd["why"]})' if fd.get('why') else ''))
                    print(line)
                for x in doc['runs']:
                    if x.get('error'):
                        print(f'{x["run"]}: could not be read: {x["error"]}')
                if doc['stale_rules']:
                    print('formula changed since the monitor was generated: ' + ', '.join(doc['stale_rules']))
                print('the generated monitor equals the offline checker' if doc['agree'] else
                      'the generated monitor does NOT equal the offline checker')
            return 0 if doc['agree'] else 1
        if a.action in ('deploy', 'rollback'):
            dev = None
            if not a.dry_run:
                from .hil import device_settings
                from .settings import Settings
                dev = device_settings(Settings())
            if a.action == 'rollback':
                out = rollback(a.monitor, dev=dev, dry=a.dry_run)
                if a.dry_run:
                    print(f'would swap {out["remote_dir"]} and {out["previous_dir"]} on the device, load the '
                          f'restored monitor once and restart {out["unit"]} if it is running:')
                    for cmd in out['plan']:
                        print('  ' + cmd)
                    print('Nothing was run (dry run: no SSH).')
                else:
                    print(f'restored {out["previous_dir"]} as {out["remote_dir"]} (manifest '
                          f'{(out["restored"] or "?")[:16]}); the device loaded it: {out["selftest"]}; '
                          f'try-restart: {out["restart"] or "done (it restarts only a running unit)"}')
                    print(f'The newer version is now in {out["previous_dir"]}: roll back again to return to it.')
                return 0
            out = deploy(a.monitor, dev=dev, dry=a.dry_run)
            if a.dry_run:
                for c in out['checks']:
                    print(f'  {"ok  " if c["ok"] else "FAIL"} {c["name"]}: {c["detail"]}')
                if not out['ok']:
                    print('not deployable as it is (nothing was sent: a dry run uses no SSH)')
                    return 2
                print(f'would keep the version on the device (if any) in {out["previous_dir"]}, then copy '
                      f'{len(out["files"])} files to {out["remote_dir"]}:')
                for f in out['file_info']:
                    print(f'  {f["name"]:34s} {f["bytes"]:>8d} B  sha256 {f["sha256"][:16]}')
                print('and load it once with the device\'s python3. Nothing would be installed, enabled or started; '
                      'nothing was sent (dry run: no SSH).')
            else:
                print(f'copied {len(out["copied"])} files to {out["remote_dir"]}; the device loaded the monitor: '
                      f'{out["selftest"]}' + (f'; the version it replaced is kept in {out["previous"]}'
                                              if out['previous'] else ''))
            print('Nothing was installed or started. On the device, run (this enables the unit):')
            for cmd in out['commands']:
                print('  ' + cmd)
            return 0
        if a.action == 'bench':
            if bool(a.monitor) == bool(a.rules):
                raise MonitorGenError('give a package folder or --rules SET, one of them')
            if a.monitor:
                mod, rd, _man = load_package(a.monitor)
                out = bench(rd['rules'], a.steps, mod=mod)
            else:
                out = bench(compile_rules(a.rules)[0], a.steps)
            if a.json:
                print(json.dumps(out, indent=1))
            else:
                print(f'{out["rules"]} rule(s), {out["steps"]} steps on {out["where"]} (Python {out["python"]}, '
                      f'{out["machine"]}): per step median {out["median_us"]:.1f} µs, p95 {out["p95_us"]:.1f} µs, '
                      f'p99 {out["p99_us"]:.1f} µs, max {out["max_us"]:.1f} µs')
            return 0
        if a.action == 'show':
            st = status(a.monitor)
            print(json.dumps(st, indent=1) if a.json else table(st))
            return 0
        if a.action == 'list':
            from .project import Project
            pk = list_packages(Project.load(a.project).workspace)
            for p in pk:
                print(table(status(p)))
            if not pk:
                print('no monitor packages (falcon monitor gen --project …)')
            return 0
    except (MonitorGenError, RT.MonitorError, DeployError, ProjectError, OSError) as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return 2
    return 2
