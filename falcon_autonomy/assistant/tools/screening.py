"""4.1 assistant tools: two-speed search (research_prescreen*) and the release gate (formal_gate_*). Registered at
import (see tools/__init__).

    research_prescreen_run   act   `falcon prescreen run` queued: a fast kinematic simulator screens a scenario's
                                   parameters, CARLA confirms the most promising (core.fastsim; a cost estimate in
                                   CARLA runs and the command on its card)
    research_prescreen_show  read  a two-speed study's fidelity gap, trust word and confirmed failures; or, with no
                                   study yet, which scenarios the fast simulator runs and why not the others
    formal_gate_run          act   `falcon gate run` queued: two versions on identical scenarios and seeds, a verdict
                                   with anytime-valid confidence (core.releasegate)
    formal_gate_show         read  a gate's verdict, bounds, trials used and the newly failing scenarios with their
                                   run pairs

Both act tools queue a job that drives CARLA (the read/act tiers and the approval card are the registry's); neither
moves anything off this machine unless `on` names the cluster, which the commands refuse for dashcam-derived
scenarios and worlds before queuing a run. Run folders are returned as run ids (C.run_id), never as home paths.
"""
from __future__ import annotations

import os

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact

MINE = {
    'research_prescreen_run': 'Queue a two-speed search: a fast simulator screens a scenario, CARLA confirms the top '
                              'candidates.',
    'research_prescreen_show': 'A two-speed search\'s fidelity gap, trust and failures; or which scenarios the fast '
                               'simulator runs.',
    'formal_gate_run': 'Queue a release gate: versions A and B on identical scenarios and seeds until a verdict is '
                       'clear.',
    'formal_gate_show': 'A release gate\'s verdict, confidence bounds, trials used and newly failing scenarios.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)


def _rid(ctx, path):
    """A run folder as the run id the other tools take; outside the workspace only its folder name."""
    if not path:
        return None
    rel = C.run_id(ctx, path)
    return os.path.basename(os.path.normpath(path)) if rel.startswith('..') else rel


# ================================================================================================ two-speed search
def _prescreen(ctx, scenario, budget=40, batch=8, threshold=0.5, screen=10000, controls=0.2, min_spearman=0.5,
               count='avoidable', seed=0, world='', rig='', on='local', dry=False):
    from ...core import fastsim as F
    from .jobs import _fake, _saved, _scenario, _world_rig
    scn = _scenario(ctx, scenario)
    why = F.unsupported(scn)
    if why:
        raise ToolError(f'the fast simulator cannot run {scn["id"]}: {why}. Use jobs_search (every run in CARLA).')
    w, r = _world_rig(ctx, world, rig)
    path = (C.project(ctx).path if dry else _saved(ctx)) or '<the project file>'
    name = F.default_name(scn['id'])
    argv = F.prescreen_argv(path, scn['id'], name, int(budget), int(batch), float(threshold), int(seed), w['id'],
                            r['id'], int(screen), float(controls), float(min_spearman), 'bayes-pof', count,
                            _fake(ctx), on or 'local')
    return 'study', f'Two-speed search {scn["name"]} ({budget} CARLA runs)', argv, {'memory_gb': 1.5,
                                                                                  'resumable': True}, name


def _prescreen_estimate(ctx, **a):
    from .jobs import _drive_estimate, _scenario, _world_rig
    scn = _scenario(ctx, a['scenario'])
    _w, r = _world_rig(ctx, a.get('world', ''), a.get('rig', ''))
    est = _drive_estimate(ctx, r, int(float(scn.get('duration_s', 12)) * 20), int(a.get('budget', 40)))
    est['what'] += f'; plus the screen of {int(a.get("screen", 10000))} cheap candidates on the CPU (seconds to a ' \
                   'minute)'
    return est


def _prescreen_preview(ctx, **a):
    return C.command_line(_prescreen(ctx, dry=True, **a)[2])


@tool('research_prescreen_run', 'research', 'act',
      'Queue a two-speed failure search (`falcon prescreen run`, resumable; docs/two-speed-search.md): a batched '
      'kinematic simulator that mirrors the director\'s kinematic mode screens `screen` parameter sets of the '
      'scenario, then CARLA confirms the most promising in rounds within `budget` CARLA runs; a share of every round '
      '(`controls`, default 0.2) is uniform random controls, from which the fidelity gap is estimated (Spearman rank '
      'correlation, false-negative risk with a Clopper–Pearson bound). Below `min_spearman` the screen is not trusted '
      'and a plain search (bayes-pof) takes over. Refused for a scenario the fast simulator cannot run (a junction, '
      'opposite lanes, replayed or level-k actors). Writes the ordinary study of the scenario (analysis_study reads '
      'it) plus prescreen.json; read with research_prescreen_show.',
      C.obj({'scenario': C.s('scenario key or id'),
             'budget': C.i('CARLA runs in all (default 40)', minimum=4, maximum=5000),
             'batch': C.i('CARLA runs per round (default 8)', minimum=2, maximum=64),
             'threshold': C.n('failure threshold of the criticality (default 0.5)', minimum=0, maximum=1),
             'screen': C.i('cheap candidates screened (default 10000)', minimum=100, maximum=200000),
             'controls': C.n('share of each round spent on random controls (default 0.2)', minimum=0, maximum=0.9),
             'min_spearman': C.n('rank correlation below which the screen is not trusted (default 0.5)', minimum=-1,
                                 maximum=1),
             'count': C.s('count avoidable failures only (RSS) or all', enum=['avoidable', 'all']),
             'seed': C.i('seed'), 'world': C.s('world id'), 'rig': C.s('rig id'),
             'on': C.s('where the CARLA runs go: local (default), cluster or cluster:POOL')}, ['scenario']),
      estimate=_prescreen_estimate, preview=_prescreen_preview)
def research_prescreen_run(ctx, **a):
    kind, title, argv, kw, study = _prescreen(ctx, **a)
    out = C.cli_job(ctx, kind, title, argv, **kw)
    out['study'] = study
    return out


@tool('research_prescreen_show', 'research', 'read',
      'A two-speed search\'s result (prescreen.json of the scenario\'s study): CARLA runs spent, candidates screened '
      'and the measured throughput, the fidelity gap (Spearman over all pairs and over the random controls, the '
      'isotonic residual, the false-negative risk with its 95 % bound and how many controls it rests on), the trust '
      'word (trusted, not trusted, not judged yet), the fallback if it happened, and the confirmed failures (CARLA '
      'runs, with the cheap value beside each). The cheap value is a screen, never a result. With coverage true (or '
      'no scenario): which built-in scenarios the fast simulator runs and why it refuses the others.',
      C.obj({'scenario': C.s('scenario key or id (its study is read)'),
             'coverage': C.b('list which scenarios the fast simulator runs'),
             'failures': C.i('failures listed (default 10)', minimum=0, maximum=50)}))
def research_prescreen_show(ctx, scenario='', coverage=False, failures=10):
    from ...core import fastsim as F
    if coverage or not scenario:
        cov = F.coverage()
        return {'covered': [k for k, v in cov.items() if v is None],
                'refused': {k: C.clip(v, 200) for k, v in cov.items() if v},
                'limits': 'one straight road, nominal boxes, kinematic motion only, no weather, no RSS label'}
    from .jobs import _scenario
    scn = _scenario(ctx, scenario)
    name = F.default_name(scn['id'])
    d = F.read_prescreen(os.path.join(C.workspace(ctx), 'studies', name))
    if d is None:
        why = F.unsupported(scn)
        raise ToolError(f'no two-speed result for {scn["id"]} yet' + (f' (and the fast simulator cannot run it: {why})'
                                                                     if why else ': research_prescreen_run'))
    f = d['fidelity']
    fn = f['false_negative']
    return {'study': name, 'state': d.get('state'), 'carla_runs': d['spent'], 'budget': d['budget'],
            'screened': d['screened'], 'throughput_per_s': (d.get('throughput') or {}).get('per_s'),
            'threshold': d['threshold'], 'trust': d['trust'],
            'spearman': C.rounded(f['spearman'], 3), 'spearman_controls': C.rounded(f['spearman_controls'], 3),
            'pairs': f['pairs'], 'controls': f['controls'], 'min_spearman': f['min_spearman'],
            'residual_rmse': f['residual_rmse'], 'selected': f.get('selected'),
            'false_negative': {'dropped_controls': fn['dropped_controls'], 'failed': fn['failed'],
                               'upper95': C.rounded(fn['upper95'], 3)},
            'fallback': d.get('fallback'),
            'failures': [{'value': C.rounded(x['value'], 3), 'cheap': C.rounded(x['low'], 3), 'role': x['role'],
                          'values': {k: C.rounded(v, 3) if isinstance(v, float) else v
                                     for k, v in x['values'].items()},
                          'run': _rid(ctx, x.get('run'))}
                         for x in (d.get('failures') or [])[:int(failures)]],
            'failures_total': len(d.get('failures') or []), 'limits': d.get('limits')}


# ================================================================================================ the release gate
def _gate(ctx, name, a, b, scenarios, budget=200, batch=8, alpha=0.05, margin=0.05, non_inferiority=None,
          primary='failure', seed=0, world='', rig='', on='local', dry=False):
    from ...core import releasegate as G
    from .jobs import _fake, _saved, _scenario, _world_rig
    scns = [_scenario(ctx, s) for s in scenarios]
    w, r = _world_rig(ctx, world, rig)
    proj = C.project(ctx)
    base = os.path.dirname(os.path.abspath(proj.path)) if proj.path else ''
    try:
        va, vb = G.parse_version(a, base), G.parse_version(b, base)
    except G.GateError as e:
        raise ToolError(str(e)) from None
    if va['kind'] != vb['kind']:
        raise ToolError('both versions must be stacks (follower, autopilot, stack:SUITE.json) or both models '
                        '(model:ID)')
    for v in (va, vb):
        if v['kind'] == 'model':
            C.find(ctx, 'models', v['model'], 'model')
    path = (proj.path if dry else _saved(ctx)) or '<the project file>'
    argv = G.gate_argv(path, name, a, b, [s['id'] for s in scns], int(budget), int(batch), float(alpha),
                       float(margin), None if non_inferiority is None else float(non_inferiority), primary,
                       int(seed), w['id'], r['id'], _fake(ctx), on or 'local')
    return 'study', f'Release gate {name}: {vb["label"]} against {va["label"]} (≤ {budget} pairs)', argv, \
        {'memory_gb': 0.5, 'resumable': True}, scns, va


def _gate_estimate(ctx, **a):
    from .jobs import _drive_estimate, _scenario, _world_rig
    scns = [_scenario(ctx, s) for s in a['scenarios']]
    _w, r = _world_rig(ctx, a.get('world', ''), a.get('rig', ''))
    steps = int(max(float(s.get('duration_s', 12)) for s in scns) * 20)
    n = int(a.get('budget', 200))
    per = 1 if str(a.get('a', '')).startswith('model:') else 2
    est = _drive_estimate(ctx, r, steps, n * per)
    est['what'] = f'at most {n} paired trials ({per} run(s) each; it stops as soon as the verdict is clear): ' \
                  + est['what']
    return est


def _gate_preview(ctx, **a):
    return C.command_line(_gate(ctx, dry=True, **a)[2])


@tool('formal_gate_run', 'formal', 'act',
      'Queue a release gate (`falcon gate run`, resumable; docs/release-gate.md): the old version `a` and the new '
      'version `b` — follower, autopilot, stack:SUITE.json (a ROS 2 stack\'s suite file) or model:ID (both the same '
      'kind) — run the same scenarios with the same seeds (common random numbers), in batches, until the verdict is '
      'clear or `budget` paired trials are spent. Verdict: better, worse, equivalent (no difference within ±margin), '
      'non_inferior (when non_inferiority is given) or cant_tell; its confidence holds at every look (a mixture test '
      'of the discordant pairs, empirical-Bernstein confidence sequences; total error alpha). The verdict holds for '
      'the tested scenario distribution only. Writes gates/<name>/gate.json; read with formal_gate_show.',
      C.obj({'name': C.s('the gate\'s name (its folder under gates/)', pattern=r'^[A-Za-z0-9][\w.-]{0,63}$'),
             'a': C.s('the old version'), 'b': C.s('the new version'),
             'scenarios': C.arr(C.s('scenario key or id'), 'the tested scenarios, taken in turn', minItems=1,
                                maxItems=30),
             'budget': C.i('paired trials at most (default 200)', minimum=2, maximum=5000),
             'batch': C.i('paired trials between looks (default 8)', minimum=1, maximum=64),
             'alpha': C.n('total error probability (default 0.05)', exclusiveMinimum=0, maximum=0.5),
             'margin': C.n('equivalence margin on the failure-rate difference (default 0.05)', exclusiveMinimum=0,
                           maximum=1),
             'non_inferiority': C.n('also stop when B is not worse by more than this', exclusiveMinimum=0, maximum=1),
             'primary': C.s('the outcome the verdict is read from', enum=['failure', 'value']),
             'seed': C.i('seed'), 'world': C.s('world id'), 'rig': C.s('rig id'),
             'on': C.s('where the runs go: local (default), cluster or cluster:POOL')},
            ['name', 'a', 'b', 'scenarios']),
      estimate=_gate_estimate, preview=_gate_preview)
def formal_gate_run(ctx, **a):
    kind, title, argv, kw, _scns, _va = _gate(ctx, **a)
    out = C.cli_job(ctx, kind, title, argv, **kw)
    out['gate'] = a['name']
    return out


@tool('formal_gate_show', 'formal', 'read',
      'A release gate\'s result (gates/<name>/gate.json): the verdict and its word, the paired trials used and run, '
      'the failures under A and B, the discordant pairs with θ\'s interval, the risk difference P(B fails) − P(A '
      'fails) and the value difference with their confidence-sequence bounds (valid at every look), whether the '
      'difference is inside the margin, and the scenarios B newly fails (and newly passes), each with its seed, '
      'parameter values and its pair of runs. With no name: the workspace\'s gates. The verdict holds for the tested '
      'scenario distribution only.',
      C.obj({'name': C.s('the gate\'s name', maxLength=64),
             'rows': C.i('newly failing / passing rows listed (default 10)', minimum=0, maximum=50)}))
def formal_gate_show(ctx, name='', rows=10):
    from ...core import releasegate as G
    ws = C.workspace(ctx)
    base = os.path.join(ws, 'gates')
    if not name:
        out = []
        for n in sorted(os.listdir(base)) if os.path.isdir(base) else ():
            d = G.read_gate(os.path.join(base, n))
            if d:
                out.append({'name': n, 'verdict': d['verdict'], 'state': d.get('state'), 'trials': d['trials_used'],
                            'a': (d.get('versions') or {}).get('a', {}).get('label'),
                            'b': (d.get('versions') or {}).get('b', {}).get('label')})
        return {'gates': out}
    if os.sep in name or name.startswith('.'):
        raise ToolError('a gate is named, not given by path')
    d = G.read_gate(os.path.join(base, name))
    if d is None:
        raise ToolError(f'no gate {name!r} (formal_gate_show with no name lists them; formal_gate_run makes one)')

    def row(x):
        return {'trial': x['i'], 'scenario': x['scenario'], 'seed': x['seed'],
                'values': {k: C.rounded(v, 3) if isinstance(v, float) else v for k, v in x['values'].items()},
                'value_a': C.rounded(x['value_a'], 3), 'value_b': C.rounded(x['value_b'], 3),
                'run_a': _rid(ctx, x['a']), 'run_b': _rid(ctx, x['b'])}
    n = int(rows)
    return {'name': name, 'state': d.get('state'), 'verdict': d['verdict'], 'word': d['word'],
            'versions': {k: (v or {}).get('label') for k, v in (d.get('versions') or {}).items()},
            'trials_used': d['trials_used'], 'trials_run': d['trials_run'], 'budget': d['budget'],
            'alpha': d['alpha'], 'primary': d['primary'], 'margin': d['margin'],
            'non_inferiority': d.get('non_inferiority'), 'within_margin': d['within_margin'],
            'failures': d['failures'],
            'discordant': {k: d['discordant'][k] for k in ('b_only', 'a_only', 'n', 'theta', 'interval')},
            'risk_difference': {k: C.rounded(d['risk'][k], 4) for k in ('estimate', 'lower', 'upper')},
            'value_difference': {k: C.rounded(d['value'][k], 4) for k in ('estimate', 'lower', 'upper')},
            'newly_failing': [row(x) for x in d['newly_failing'][:n]], 'newly_failing_total': len(d['newly_failing']),
            'newly_passing': [row(x) for x in d['newly_passing'][:n]], 'newly_passing_total': len(d['newly_passing']),
            'skipped': len(d.get('skipped') or ()), 'scope': d['scope']}
