"""4.1 assistant tools: runtime monitors (formal_monitor_*) and latency in the loop (analysis_latency_*). Registered
at import (see tools/__init__).

    formal_monitor_gen       act   a monitor package generated in the workspace (monitors/<name>) from a rule set's
                                   rules or a formula (core.monitorgen.generate)
    formal_monitor_check     act   runs replayed through the generated monitor and compared with the offline checker;
                                   the result is recorded beside the package (equivalence.json: what the safety case
                                   reads), so it is an act tool
    analysis_latency_budget  read  the workspace's latency budgets (latency/<name>/latency-budget.json) and profiles
    analysis_latency_run     act   `falcon latency budget` queued (a cost estimate and the command on its card)

Runs are read through the privacy guard (C.resolve_run). Nothing here reaches a device: deployment is the user's
(`falcon monitor deploy`, the Hardware page).
"""
from __future__ import annotations

import os

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact

MINE = {
    'formal_monitor_gen': 'Generate a runtime monitor package (pure-Python + ROS 2 node) from rules or an STL formula.',
    'formal_monitor_check': 'Replay runs through a generated monitor; compare its verdicts with the offline checker.',
    'analysis_latency_budget': 'Latency budgets: per scenario the largest latency that still passes; fails only when '
                               'late.',
    'analysis_latency_run': 'Queue a latency-budget sweep: bisect the latency at which scenarios start to fail.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)

_NAME = r'^[A-Za-z][\w.-]{0,47}$'


def _MG():
    from ...core import monitorgen as MG
    return MG


def _pkg(ctx, name: str) -> str:
    p = os.path.join(C.workspace(ctx), 'monitors', name)
    if not os.path.isfile(os.path.join(p, 'manifest.json')):
        have = [os.path.basename(x) for x in _MG().list_packages(C.workspace(ctx))]
        raise ToolError(f'no monitor {name!r} in the workspace ({", ".join(have) or "none yet: formal_monitor_gen"})')
    return p


def _brief(st: dict) -> dict:
    return {'name': st['name'], 'status': st['status'], 'manifest': st['hash'][:16], 'checked_at': st['checked_at'],
            'rules': [{'id': r['id'], 'status': r['status'], 'sha': r['sha'], 'delay_s': r['delay_s'],
                       'why': C.clip(r['why'], 200)} for r in st['rules']]}


def _gen_preview(ctx, name, ruleset='', rules=None, formula=''):
    what = f'formula {C.clip(formula, 120)}' if formula else f'{ruleset}: {", ".join(rules or []) or "every rule"}'
    return f'write the monitor package monitors/{name} ({what})'


@tool('formal_monitor_gen', 'formal', 'act',
      'Generate a runtime monitor package in the workspace (monitors/<name>): the rules of a rule set (india, '
      'generic; all or the ids given) or one STL formula over the rule signals, compiled for online checking on the '
      'vehicle computer — a standard-library Python evaluator, an rclpy node, a topic mapping, a systemd user unit '
      'and a manifest with each rule\'s formula hash. Bounded future operators delay the verdict by their horizon; '
      'an unbounded future operator is refused. Returns the rules with hash, delay and signals. Nothing is deployed.',
      C.obj({'name': C.s('the monitor\'s name', pattern=_NAME),
             'ruleset': C.s('india or generic (or give formula)', enum=['india', 'generic']),
             'rules': C.arr(C.s('rule id'), 'rule ids of the set (default: all)', maxItems=40),
             'formula': C.s('an STL formula monitored at every sample', maxLength=2000)}, ['name']),
      preview=_gen_preview)
def formal_monitor_gen(ctx, name, ruleset='', rules=None, formula=''):
    MG = _MG()
    if bool(ruleset) == bool(formula):
        raise ToolError('give one of ruleset or formula')
    try:
        comp, src = MG.compile_rules(ruleset, rules or None) if ruleset else MG.compile_stl(formula, name)
        man = MG.generate(os.path.join(C.workspace(ctx), 'monitors', name), comp, src, name)
    except (MG.MonitorGenError, OSError) as e:
        raise ToolError(C.clip(str(e), 400)) from None
    return {'monitor': name, 'path': f'monitors/{name}', 'manifest': man['hash'][:16],
            'rules': [{'id': r['id'], 'sha': r['sha'], 'delay_s': (r['horizon_s'] or {}).get('future') or 0.0,
                       'signals': r['signals']} for r in man['rules']],
            'next': 'formal_monitor_check with runs; deployment is the user\'s (falcon monitor deploy)'}


def _check_preview(ctx, name, runs=None):
    return f'replay {len(runs or [])} run(s) through monitors/{name} and record the comparison (equivalence.json)'


@tool('formal_monitor_check', 'formal', 'act',
      'Equivalence of a generated monitor with the offline checker: each run (ids as runs_search gives them, ≤ 10) '
      'is replayed through the package\'s own module sample by sample and compared with core.stl\'s robustness (exact) '
      'and core.rules.check\'s verdict. Returns per rule agreed / disagreed with the first disagreement (run, '
      'sample, time, both values), and rules whose formula changed since generation (stale). Recorded as '
      'equivalence.json, which a GSN Solution of kind monitor reads. With no runs: the package\'s current status.',
      C.obj({'name': C.s('the monitor\'s name', pattern=_NAME),
             'runs': C.arr(C.s('run id'), 'run ids', maxItems=10)}, ['name']),
      preview=_check_preview)
def formal_monitor_check(ctx, name, runs=None):
    MG = _MG()
    dirs = [C.resolve_run(ctx, r) for r in runs or []]           # each through the privacy guard, before anything else
    pkg = _pkg(ctx, name)
    try:
        if runs:
            doc = MG.check(pkg, dirs)
            out = _brief(MG.status(pkg))
            out['agree'] = doc['agree']
            out['errors'] = [{'run': C.run_id(ctx, x['run']), 'error': C.clip(x['error'], 200)}
                             for x in doc['runs'] if x.get('error')]
            for r in out['rules']:
                fd = (doc['rules'].get(r['id']) or {}).get('first_disagreement')
                if fd:
                    r['first_disagreement'] = {k: (C.run_id(ctx, v) if k == 'run' and v else v)
                                               for k, v in fd.items()}
            return out
        return _brief(MG.status(pkg))
    except MG.MonitorGenError as e:
        raise ToolError(C.clip(str(e), 400)) from None


# ================================================================================================ latency
def _LL():
    from ...core import latencyloop as LL
    return LL


def _case(r: dict) -> dict:
    out = {'case': r['case'], 'status': r['status'], 'budget': C.rounded(r.get('budget'), 3),
           'first_failure': C.rounded(r.get('first_failure'), 3), 'runs': r['runs'],
           'baseline_criticality': C.rounded(r.get('baseline_criticality'), 3)}
    for k in ('budget_ms', 'over_budget_share', 'non_monotone'):
        if r.get(k) is not None:
            out[k] = ({a: C.rounded(b, 2) for a, b in r[k].items()} if isinstance(r[k], dict) else
                      C.rounded(r[k], 4) if isinstance(r[k], float) else r[k])
    return out


@tool('analysis_latency_budget', 'analysis', 'read',
      'Latency budgets of the workspace (latency in the loop, core.latencyloop). Without a name: the budgets and '
      'latency profiles there are. With a name: per scenario case its status (fails only when late / fails without '
      'latency / no failure up to the maximum), the budget (the largest latency that still passes; ms for a constant '
      'sweep, the profile\'s scale otherwise, with its median and p99 in ms), where it first fails, and the list of '
      'cases that fail only when the computer is late. Latencies are rounded up to whole simulation steps.',
      C.obj({'name': C.s('a budget\'s name', maxLength=64)}))
def analysis_latency_budget(ctx, name=''):
    LL = _LL()
    ws = C.workspace(ctx)
    if not name:
        profs = []
        for p in LL.list_profiles(ws):
            d = C.read_json(p, {}) or {}
            if d.get('schema') == LL.PROFILE_SCHEMA:
                s = d['stats']
                profs.append({'name': d['name'], 'measured': bool(d.get('measured')),
                              'source': (d.get('source') or {}).get('kind'),
                              'device': C.clip((d.get('source') or {}).get('device') or '', 80),
                              'median_ms': C.rounded(s['median'], 2), 'p99_ms': C.rounded(s['p99'], 2),
                              'max_ms': C.rounded(s['max'], 2)})
        buds = []
        for p in LL.list_budgets(ws):
            d = C.read_json(p, {}) or {}
            buds.append({'name': d.get('name'), 'unit': d.get('unit'), 'cases': len(d.get('cases') or []),
                         'fails_only_when_late': len(d.get('fails_only_when_late') or [])})
        return {'budgets': buds, 'profiles': profs}
    d = C.read_json(os.path.join(LL.budget_dir(ws, os.path.basename(name)), 'latency-budget.json'))
    if not d or d.get('schema') != LL.BUDGET_SCHEMA:
        raise ToolError(f'no latency budget {name!r} (analysis_latency_budget without a name lists them)')
    prof = d.get('profile') or d.get('reference_profile')
    return {'name': d['name'], 'sweep': d['sweep'], 'unit': d['unit'], 'grid_step': d['grid_step'],
            'grid_max': d['grid_max'], 'step_ms': d['step_ms'], 'criterion': d['criterion'], 'mode': d['mode'],
            'profile': None if not prof else {'name': prof.get('name'), 'measured': prof.get('measured'),
                                              'median_ms': C.rounded(prof['stats']['median'], 2),
                                              'p99_ms': C.rounded(prof['stats']['p99'], 2)},
            'fails_only_when_late': d['fails_only_when_late'], 'assumes': C.clip(d['assumes'], 240),
            **C.capped([_case(r) for r in d['cases']], 40)}


def _latency(ctx, scenarios, name='', profile='', max_ms=500.0, max_scale=4.0, samples=1, world='', rig='', seed=0,
             dry=False):
    from .jobs import _fake, _saved, _scenario, _world_rig
    LL = _LL()
    scns = [_scenario(ctx, s) for s in scenarios]
    w, r = _world_rig(ctx, world, rig)
    path = (C.project(ctx).path if dry else _saved(ctx)) or '<the project file>'
    name = name or ('latency-' + '-'.join(s['id'] for s in scns))[:60]
    argv = ['latency', 'budget', path]
    for s in scns:
        argv += ['--scenario', s['id']]
    argv += ['--name', name, '--world', w['id'], '--rig', r['id'], '--seed', str(int(seed)), '--samples',
             str(int(samples))]
    if profile:
        pp = os.path.join(C.workspace(ctx), 'latency', 'profiles', os.path.basename(profile) + '.json')
        if not os.path.isfile(pp):
            have = [os.path.basename(x)[:-5] for x in LL.list_profiles(C.workspace(ctx))]
            raise ToolError(f'no latency profile {profile!r} ({", ".join(have) or "none: falcon latency profile"})')
        argv += ['--profile', pp, '--max-scale', str(float(max_scale))]
    else:
        argv += ['--constant', '--max-ms', str(float(max_ms))]
    if _fake(ctx):
        argv += ['--fake-carla', _fake(ctx)]
    return 'study', f'Latency budget {name}', argv, {'memory_gb': 0.5, 'resumable': True}, name, scns, r


def _runs_per_case(a) -> int:
    import math
    n = (float(a.get('max_scale', 4.0)) / 0.25) if a.get('profile') else (float(a.get('max_ms', 500.0)) / 50.0)
    return 2 + int(math.ceil(math.log2(max(n, 2))))


def _latency_estimate(ctx, **a):
    from .jobs import _drive_estimate
    _k, _t, _argv, _kw, _name, scns, r = _latency(ctx, dry=True, **a)
    n = len(scns) * int(a.get('samples', 1)) * _runs_per_case(a)
    return _drive_estimate(ctx, r, int(max(float(s.get('duration_s', 12)) for s in scns) * 20), n)


def _latency_preview(ctx, **a):
    return C.command_line(_latency(ctx, dry=True, **a)[2])


@tool('analysis_latency_run', 'analysis', 'act',
      'Queue a latency-budget sweep (`falcon latency budget`, resumable): for each scenario the latency between the '
      'computed and the applied control is raised — a constant latency in whole simulation steps up to max_ms, or a '
      'saved latency profile\'s scale up to max_scale — with common random numbers, and the largest latency that '
      'still passes is bisected (failures assumed monotone). Returns the job and the budget\'s name; when it ends, '
      'analysis_latency_budget with that name.',
      C.obj({'scenarios': C.arr(C.s('scenario key or id'), 'scenarios', minItems=1, maxItems=12),
             'name': C.s('the budget\'s name', pattern=_NAME),
             'profile': C.s('a saved latency profile\'s name (else a constant latency is swept)', maxLength=64),
             'max_ms': C.n('constant sweep: up to this latency in ms (default 500)', exclusiveMinimum=0, maximum=5000),
             'max_scale': C.n('profile sweep: up to this scale (default 4)', exclusiveMinimum=0, maximum=64),
             'samples': C.i('parameter sets per scenario (default 1)', minimum=1, maximum=20),
             'world': C.s('world id'), 'rig': C.s('rig id'), 'seed': C.i('seed')}, ['scenarios']),
      estimate=_latency_estimate, preview=_latency_preview)
def analysis_latency_run(ctx, **a):
    kind, title, argv, kw, name, _scns, _r = _latency(ctx, **a)
    out = C.cli_job(ctx, kind, title, argv, **kw)
    out['budget'] = name
    return out
