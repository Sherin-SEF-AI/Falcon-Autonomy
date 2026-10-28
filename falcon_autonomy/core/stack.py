"""Stack testing: the stack worker's config, scenario suites with pass/fail criteria, and their results (standard
library + numpy; the runner is falcon_autonomy.sim.stacksuite, the worker falcon_autonomy.workers.stack).

A suite (schema falcon.suite/1; docs/stack-testing.md):

    {"schema": "falcon.suite/1", "name": "...", "about": "...",
     "stack": {"driver": "ros2" | "follower" | "autopilot",
               "bridge": {...stackbridge.bridge_defaults() fields...},
               "launch": [[argv...], ...],          ros2: commands started for each run (the bridge node, the stack);
                                                    "{port}", "{name}", "{run}", "{domain}" are filled in
               "domain_id": 0},
     "world": "world-town10", "rig": "rig-default", "fps": 20.0, "motion": "kinematic", "traffic_vehicles": 0,
     "criteria": {"no_collision": true, "min_ttc_s": 1.0, "max_missed_deadlines": 0},
     "entries": [{"scenario": "<built-in key or project scenario id>",
                  "seeds": [0, 1, 2] | "samples": 10, "seed": 0,      seeds, or that many from seed (seed·1000003+i)
                  "values": {...},                                    fixed parameter values (others sampled)
                  "criteria": {...}}]}                                per entry, over the suite's

Criteria replace the scenario's own "pass" where given: no_collision (boxes or the collision sensor), min_ttc_s (the
run's minimum time to collision), max_missed_deadlines (ros2 only: missed control deadlines). A run passes when every
check passes. A run that did not finish (the worker failed: no bridge node, no server) is not a result: it is recorded
in results.jsonl with its status and run again on resume.

Runs. Each (entry, seed) is a stack run in <suite>/runs/<key[:16]>/ (run_key: the run's config without where it is
written, so the same run is never repeated), with stack.json and run.json; results.jsonl holds one line per finished
run. Every run is replayable: `falcon stack replay RUN` re-runs it with the controls it recorded (bit for bit in
kinematic motion).

Results (summarise): per scenario the runs, passes, the pass rate with its Wilson 95 % interval (Wilson 1927), the mean
criticality with a percentile bootstrap 95 % interval over runs (core.metrics.uncertainty.bootstrap, seeded), the
minimum time to collision, collisions and missed deadlines; the same overall.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os

from . import scenarios as SC
from .provenance import sha256_json

STACK_SCHEMA = 'falcon.stack_config/1'
SUITE_SCHEMA = 'falcon.suite/1'
DRIVERS = ('ros2', 'follower', 'autopilot')
Z95 = 1.959963984540054


def stack_config(project, world_id: str, rig_id: str, out_dir: str, port: int, tm_port: int, driver: str = 'ros2',
                 scenario: dict | None = None, seed: int = 0, values: dict | None = None, seconds: float = 20.0,
                 fps: float = 20.0, motion: str = 'kinematic', bridge: dict | None = None,
                 traffic_vehicles: int = 0, render_budget_mpx: float = 200.0, workspace: str = '',
                 criteria: dict | None = None, name: str = '') -> dict:
    """A stack worker config: a drive config (core.drive) with the driver, the motion and the bridge; a scenario is
    sampled with the seed (and the given values), its pass criteria replaced by `criteria` where given."""
    from .drive import drive_config, scenario_config
    from .stackbridge import bridge_defaults
    if scenario is not None:
        cfg = scenario_config(project, scenario, world_id, rig_id, out_dir, port, tm_port, seed=seed,
                              values=values, fps=fps, traffic_vehicles=traffic_vehicles,
                              render_budget_mpx=render_budget_mpx, workspace=workspace)
        crit = dict(criteria or {})
        if crit:
            p = dict(cfg['scenario'].get('pass') or {})
            for k in ('no_collision', 'min_ttc_s'):
                if k in crit:
                    p[k] = crit[k]
            cfg['scenario']['pass'] = {k: v for k, v in p.items() if v is not None}
        cfg['scenario']['motion'] = motion
    else:
        cfg = drive_config(project, world_id, rig_id, out_dir, port, tm_port, seconds=seconds, fps=fps, seed=seed,
                           render_budget_mpx=render_budget_mpx, workspace=workspace)
        cfg['traffic'] = dict(cfg['traffic'], vehicles=int(traffic_vehicles))
    cfg['schema'] = STACK_SCHEMA
    cfg['driver'] = driver
    cfg['motion'] = motion
    cfg['name'] = name or cfg.get('name', 'stack run')
    b = dict(bridge_defaults(), **(bridge or {}))
    if criteria and criteria.get('max_missed_deadlines') is not None:
        b['max_missed'] = int(criteria['max_missed_deadlines'])
    cfg['bridge'] = b
    return cfg


def run_key(cfg: dict) -> str:
    """The cache key of a stack run: its config without where it runs (folder, ports, host, name)."""
    c = {k: v for k, v in cfg.items() if k not in ('out', 'port', 'tm_port', 'host', 'name')}
    b = dict(c.get('bridge') or {})
    for k in ('host', 'port', 'connect_wait_s'):
        b.pop(k, None)
    c['bridge'] = b
    return sha256_json(c)


# ------------------------------------------------------------------------------------------------------------ suites
def new_suite(name: str, driver: str = 'follower', scenarios=('lead_brake', 'stalled_vehicle'), seeds=(0, 1, 2),
              world: str = 'world-town10', rig: str = 'rig-default') -> dict:
    return {'schema': SUITE_SCHEMA, 'name': name, 'about': '', 'stack': {'driver': driver, 'bridge': {},
                                                                         'launch': [], 'domain_id': 0},
            'world': world, 'rig': rig, 'fps': 20.0, 'motion': 'kinematic', 'traffic_vehicles': 0,
            'criteria': {'no_collision': True, 'min_ttc_s': 1.0, 'max_missed_deadlines': 0},
            'entries': [{'scenario': s, 'seeds': list(seeds)} for s in scenarios]}


def validate_suite(doc: dict, project=None) -> list:
    out = []
    if doc.get('schema') != SUITE_SCHEMA:
        out.append(f'schema {doc.get("schema")!r}, expected {SUITE_SCHEMA}')
    st = doc.get('stack') or {}
    if st.get('driver') not in DRIVERS:
        out.append(f'stack.driver {st.get("driver")!r}: one of {", ".join(DRIVERS)}')
    if st.get('driver') == 'ros2':
        from .stackbridge import bridge_defaults, check_bridge
        out += [f'stack.bridge: {p}' for p in check_bridge(dict(bridge_defaults(), **(st.get('bridge') or {})))]
    if doc.get('motion', 'kinematic') not in ('kinematic', 'physics'):
        out.append('motion: kinematic or physics')
    if not doc.get('entries'):
        out.append('no entries')
    for k, e in enumerate(doc.get('entries') or []):
        scn = resolve_scenario(e.get('scenario'), project)
        if scn is None:
            out.append(f'entry {k}: no scenario {e.get("scenario")!r}')
            continue
        if not e.get('seeds') and not e.get('samples'):
            out.append(f'entry {k}: give seeds or samples')
        bad = sorted(set(e.get('values') or {}) - set(scn.get('params') or {}))
        if bad:
            out.append(f'entry {k}: {scn["id"]} has no parameter(s) {", ".join(bad)}')
        problems = SC.validate(scn)
        out += [f'entry {k}: {p}' for p in problems]
    if project is not None:
        if project.item('worlds', doc.get('world')) is None:
            out.append(f'no world {doc.get("world")!r} in the project')
        if project.item('rigs', doc.get('rig')) is None:
            out.append(f'no rig {doc.get("rig")!r} in the project')
    return out


def resolve_scenario(key, project=None) -> dict | None:
    if not key:
        return None
    if project is not None:
        s = project.item('scenarios', key)
        if s is not None:
            return copy.deepcopy(s)
    return SC.builtin(key) if key in SC.LIBRARY else None


def entry_seeds(e: dict) -> list:
    if e.get('seeds'):
        return [int(s) for s in e['seeds']]
    base = int(e.get('seed', 0))
    return [base * 1000003 + i for i in range(int(e.get('samples', 0)))]


def plan_runs(doc: dict, project, suite_dir: str, port: int, tm_port: int, render_budget_mpx: float = 200.0) -> list:
    """[(run spec, stack config)] of a suite, in order: spec {'entry', 'scenario', 'seed', 'key', 'dir'}."""
    st = doc['stack']
    out = []
    for k, e in enumerate(doc['entries']):
        scn = resolve_scenario(e['scenario'], project)
        crit = dict(doc.get('criteria') or {}, **(e.get('criteria') or {}))
        for seed in entry_seeds(e):
            cfg = stack_config(project, doc['world'], doc['rig'], '/pending', port, tm_port, st['driver'], scn,
                               seed=seed, values=e.get('values'), fps=float(doc.get('fps', 20.0)),
                               motion=doc.get('motion', 'kinematic'), bridge=st.get('bridge'),
                               traffic_vehicles=int(doc.get('traffic_vehicles', 0)),
                               render_budget_mpx=render_budget_mpx, workspace=project.workspace,
                               criteria=crit, name=f'{doc["name"]}: {scn["name"]} (seed {seed})')
            key = run_key(cfg)
            cfg['out'] = os.path.join(suite_dir, 'runs', key[:16])
            cfg['cache_key'] = key
            out.append(({'entry': k, 'scenario': scn['id'], 'scenario_name': scn['name'], 'seed': seed,
                         'key': key, 'dir': cfg['out']}, cfg))
    return out


def result_row(spec: dict, run: dict) -> dict:
    """A results.jsonl line from a run's spec and its run.json."""
    scn = run.get('scenario') or {}
    crit = scn.get('criticality') or {}
    overall = crit.get('overall') or {}
    bridge = run.get('bridge') or {}
    collided = bool(overall.get('collision')) or any(a.get('collision') for a in (crit.get('actors') or {}).values())
    return {'key': spec['key'], 'entry': spec['entry'], 'scenario': spec['scenario'],
            'scenario_name': spec.get('scenario_name', spec['scenario']), 'seed': spec['seed'],
            'run': spec['dir'], 'status': run.get('status'), 'error': run.get('error'),
            'passed': scn.get('passed') if run.get('status') == 'ok' else None, 'checks': scn.get('checks') or [],
            'values': scn.get('values'), 'criticality': overall.get('criticality', crit.get('criticality')),
            'min_ttc_s': overall.get('min_ttc_s'), 'min_distance_m': overall.get('min_distance_m'),
            'collision': collided, 'missed_deadlines': bridge.get('missed'),
            'steps_per_s': (run.get('performance') or {}).get('steps_per_s'), 'driver': run.get('driver')}


def read_results(suite_dir: str) -> list:
    """The finished runs of a suite (the last line per key wins)."""
    p = os.path.join(suite_dir, 'results.jsonl')
    if not os.path.isfile(p):
        return []
    rows = {}
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r['key']] = r
    return list(rows.values())


def wilson(k: int, n: int, z: float = Z95) -> tuple:
    """Wilson's score interval of a proportion k / n (95 % by default); (nan, nan) for n = 0."""
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _group(rows: list, seed: int) -> dict:
    from .metrics.uncertainty import bootstrap
    ok = [r for r in rows if r.get('status') == 'ok' and r.get('passed') is not None]
    n = len(ok)
    k = sum(1 for r in ok if r['passed'])
    lo, hi = wilson(k, n)
    crit = [float(r['criticality']) for r in ok if r.get('criticality') is not None]
    ttc = [float(r['min_ttc_s']) for r in ok if r.get('min_ttc_s') is not None and math.isfinite(r['min_ttc_s'])]
    g = {'runs': n, 'passed': k, 'failed': n - k, 'pass_rate': (k / n) if n else None,
         'pass_rate_ci95': [lo, hi] if n else None, 'interval': 'Wilson score, 95 %',
         'collisions': sum(1 for r in ok if r.get('collision')),
         'missed_deadlines': sum(int(r.get('missed_deadlines') or 0) for r in ok),
         'min_ttc_s': min(ttc) if ttc else None, 'not_finished': len(rows) - n}
    if crit:
        mean = sum(crit) / len(crit)
        if len(crit) > 1:
            b = bootstrap(lambda xs: sum(xs) / len(xs), crit, n=1000, seed=seed, unit='image')
            g['criticality'] = {'mean': mean, 'ci95': [b['ci_low'], b['ci_high']],
                                'interval': 'percentile bootstrap over runs, 1000 resamples', 'max': max(crit)}
        else:
            g['criticality'] = {'mean': mean, 'ci95': None, 'interval': 'one run: no interval', 'max': max(crit)}
    return g


def summarise(rows: list, seed: int = 0) -> dict:
    """Per scenario and overall (module docstring)."""
    by = {}
    for r in rows:
        by.setdefault(r['scenario'], []).append(r)
    per = {s: dict(_group(rs, seed), name=rs[0].get('scenario_name', s)) for s, rs in sorted(by.items())}
    return {'scenarios': per, 'overall': _group(rows, seed),
            'failures': sorted([{k: r.get(k) for k in ('scenario', 'seed', 'values', 'run', 'checks', 'criticality',
                                                       'min_ttc_s', 'collision', 'missed_deadlines')}
                                for r in rows if r.get('status') == 'ok' and r.get('passed') is False],
                               key=lambda r: -(r.get('criticality') or 0.0))}


TABLE_COLUMNS = ('scenario', 'seed', 'status', 'passed', 'criticality', 'min_ttc_s', 'min_distance_m', 'collision',
                 'missed_deadlines', 'steps_per_s', 'run')


def write_table(rows: list, path: str):
    """The results table as CSV (one row per run, TABLE_COLUMNS)."""
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(TABLE_COLUMNS)
        for r in sorted(rows, key=lambda r: (r['scenario'], r['seed'])):
            w.writerow(['' if r.get(c) is None else r.get(c) for c in TABLE_COLUMNS])
