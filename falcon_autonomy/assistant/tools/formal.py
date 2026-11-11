"""Formal tools (the formal group): signal temporal logic requirements, their falsification and the living GSN safety
case (docs/falsification.md, docs/gsn.md).

    formal_stl_check       read  a formula parsed by core.stl against the rule signals: signals, horizon, normal form
    formal_stl_compile     act   a requirement (formula or SET:ID rule) checked, saved by name in safety/formulas.json
    formal_stl_robustness  read  runs' robustness ρ against a requirement (core.stlobjective.score_run)
    formal_falsify_stl     act   `falcon falsify run` queued (a cost estimate and the command on its card)
    formal_gsn_tree        read  the safety case verified now (not logged): statuses, why, evidence
    formal_gsn_verify      act   verify and log; optionally re-bind one Solution first (its old evidence logged)

Runs are read through the privacy guard (C.resolve_run); the safety case and formulas live in the workspace.
formal_causal_probe (the introspection workstream's) is not here.

A module adds its tools' one-line descriptions to _compact.SHORT itself (MINE below; tools/__init__ explains why)."""
from __future__ import annotations

import math

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact

MINE = {
    'formal_stl_check': 'Parse an STL formula over the rule signals: its signals, horizon and normal form, or the '
                        'error.',
    'formal_stl_compile': 'Save a named STL requirement (formula or SET:ID rule) in the workspace after checking it.',
    'formal_stl_robustness': 'Robustness ρ of runs against a requirement: ρ, verdict, deciding time, unknown signals.',
    'formal_falsify_stl': 'Queue a falsification study: search a scenario for runs violating a requirement.',
    'formal_gsn_tree': 'The living GSN safety case: each node\'s status (green/amber/red/grey), why and evidence.',
    'formal_gsn_verify': 'Re-verify the GSN safety case (logged); optionally re-bind one Solution to its result.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)


def _secs(v: float):
    return None if math.isinf(v) else round(float(v), 3)


@tool('formal_stl_check', 'formal', 'read',
      'Check a signal temporal logic formula (core.stl\'s language: always[a,b], eventually[a,b], until, since, '
      'and/or/not/->, comparisons of expressions over signals) against the signals the traffic rules can read '
      '(core.rules.SIGNALS). Returns ok, the formula in normal form (full parentheses), the signals it reads with '
      'their units, and its horizon — how many seconds it looks back and ahead (null: unbounded); or ok false with '
      'the parse error and its column, or the unknown signal names. Reads nothing from runs; changes nothing.',
      C.obj({'formula': C.s('the formula, e.g. "always[0,10] (lead_gap_m >= 5)"', minLength=1, maxLength=2000),
             'consts': C.obj(description='named constants substituted into the formula (name → number)',
                             additionalProperties={'type': 'number'})}, required=['formula']))
def formal_stl_check(ctx, formula, consts=None):
    from ...core import stl as S
    from ...core.rules import SIGNALS
    try:
        node = S.parse(formula, consts or {})
    except S.STLError as e:
        return {'ok': False, 'error': C.clip(str(e), 300)}
    except (ValueError, TypeError) as e:          # a constant that is not a number
        raise ToolError(f'consts: {e}') from None
    used = sorted(S.names(node))
    unknown = [n for n in used if n not in SIGNALS]
    if unknown:
        return {'ok': False, 'error': f'unknown signal(s) {", ".join(unknown)}', 'unknown': unknown,
                'known': sorted(SIGNALS)}
    past, future = S.horizon(node)
    return {'ok': True, 'normalised': C.clip(node.text(), 2000),
            'signals': [{'name': n, 'unit': SIGNALS[n][0]} for n in used],
            'horizon_s': {'past': _secs(past), 'future': _secs(future)}}


# ================================================================================================ the falsifier
_REQ = {'formula': C.s('an STL formula over the rule signals (or give rule or name)', maxLength=2000),
        'rule': C.s('a rule as SET:ID[,ID], e.g. india:following_distance', maxLength=300),
        'name': C.s('a requirement saved with formal_stl_compile', maxLength=64)}


def _SO():
    from ...core import stlobjective as SO
    return SO


def _requirement(ctx, formula='', rule='', name='', scale=1.0, smooth=0.0) -> dict:
    SO = _SO()
    if sum(bool(x) for x in (formula, rule, name)) != 1:
        raise ToolError('give one of formula, rule or name')
    try:
        spec = SO.named(C.workspace(ctx), name) if name else {'stl': formula or None, 'rule': rule or None,
                                                              'scale': scale, 'smooth': smooth}
        return SO.normalise(spec)
    except SO.STLObjectiveError as e:
        raise ToolError(str(e)) from None


def _stl_compile_preview(ctx, name, formula='', rule='', scale=1.0, smooth=0.0, about=''):
    s = _requirement(ctx, formula, rule, '', scale, smooth)
    return f'save requirement {name}: {s["formula"]} (scale {s["scale"]:g}, β {s["smooth"]:g}) in safety/formulas.json'


@tool('formal_stl_compile', 'formal', 'act',
      'Check a requirement — an STL formula over the rule signals (core.rules.SIGNALS, incl. ttc_min_s, ttc_lead_s, '
      'drac_mps2, pet_s, lat_clear_m, lon_gap_m) or a rule set\'s rule(s) SET:ID — and save it under a name in the '
      'workspace\'s safety/formulas.json (replacing one of that name), for formal_falsify_stl and '
      'formal_stl_robustness. Returns its normal form, signals and key; or the parse error with its column.',
      C.obj({'name': C.s('the name to save it under', pattern=r'^[A-Za-z][\w.-]{0,63}$'),
             'formula': _REQ['formula'], 'rule': _REQ['rule'],
             'scale': C.n('ρ scale of the search value σ(−ρ/scale) (default 1)', exclusiveMinimum=0),
             'smooth': C.n('β of the soft-min robustness the searcher sees (0 exact)', minimum=0),
             'about': C.s('what it requires, in words', maxLength=500)}, ['name']),
      preview=_stl_compile_preview)
def formal_stl_compile(ctx, name, formula='', rule='', scale=1.0, smooth=0.0, about=''):
    SO = _SO()
    s = _requirement(ctx, formula, rule, '', scale, smooth)
    try:
        e = SO.save_formula(C.workspace(ctx), name, s, about)
    except SO.STLObjectiveError as ex:
        raise ToolError(str(ex)) from None
    return {'saved': name, 'formula': C.clip(e['formula'], 600), 'signals': e['signals'], 'key': e['key'],
            'path': 'safety/formulas.json'}


@tool('formal_stl_robustness', 'formal', 'read',
      'The robustness ρ of recorded runs (run ids as runs_search gives them, ≤ 20) against a requirement (a formula, '
      'a rule SET:ID or a saved name): ρ (< 0 violated, in the predicate\'s units; "inf"/"-inf" possible; null '
      'unknown), the verdict, the time that decides it, the share of the run evaluated and the signals not recorded. '
      'series adds a down-sampled ρ(t) (≤ 60 points).',
      C.obj({'runs': C.arr(C.s('run id'), 'run ids', minItems=1, maxItems=20), **_REQ,
             'series': C.b('include ρ(t), down-sampled')}, ['runs']))
def formal_stl_robustness(ctx, runs, formula='', rule='', name='', series=False):
    SO = _SO()
    dirs = [C.resolve_run(ctx, r) for r in runs]           # each through the privacy guard, before anything else
    s = _requirement(ctx, formula, rule, name)
    out = []
    for d in dirs:
        sc = SO.score_run(d, s, series=bool(series))
        row = {'run': C.run_id(ctx, d), 'rho': sc['rho'], 'state': sc['state'], 'decided_at_s': sc['worst_t'],
               'evaluated_share': sc['evaluated_share'], 'missing_signals': sc['missing_signals']}
        if sc.get('error'):
            row['error'] = C.clip(sc['error'], 200)
        if series and sc.get('t'):
            k = max(1, len(sc['t']) // 60)
            row['t'] = sc['t'][::k]
            row['rho_t'] = [x if isinstance(x, str) else C.rounded(x, 3) for x in sc['rho_series'][::k]]
        out.append(row)
    return {'formula': C.clip(s['formula'], 600), 'runs': out}


def _falsify(ctx, scenario, formula='', rule='', name='', method='cmaes', budget=60, batch=4, scale=1.0,
             smooth=0.0, margin=1.0, world='', rig='', seed=0, dry=False):
    from .jobs import _fake, _saved, _scenario, _world_rig
    SO = _SO()
    s = _requirement(ctx, formula, rule, name, scale, smooth)
    scn = _scenario(ctx, scenario)
    w, r = _world_rig(ctx, world, rig)
    path = (C.project(ctx).path if dry else _saved(ctx)) or '<the project file>'
    study = SO.study_name(scn['id'], s)
    argv = ['falsify', 'run', path, '--scenario', scn['id'], '--study', study, '--method', method, '--budget',
            str(int(budget)), '--batch', str(int(batch)), '--margin', str(float(margin)), '--world', w['id'],
            '--rig', r['id'], '--seed', str(int(seed))]
    argv += ['--stl', s['stl']] if s['stl'] else ['--rule', s['rule']]
    argv += ['--scale', str(s['scale']), '--smooth', str(s['smooth'])]
    if _fake(ctx):
        argv += ['--fake-carla', _fake(ctx)]
    return 'study', f'Falsify {scn["name"]} ({method}, {budget})', argv, {'memory_gb': 0.5, 'resumable': True}, study


def _falsify_estimate(ctx, **a):
    from .jobs import _drive_estimate, _scenario, _world_rig
    scn = _scenario(ctx, a['scenario'])
    _w, r = _world_rig(ctx, a.get('world', ''), a.get('rig', ''))
    return _drive_estimate(ctx, r, int(float(scn.get('duration_s', 12)) * 20), int(a.get('budget', 60)))


def _falsify_preview(ctx, **a):
    _k, _t, argv, _kw, _s = _falsify(ctx, dry=True, **a)
    return C.command_line(argv)


@tool('formal_falsify_stl', 'formal', 'act',
      'Queue a falsification study (`falcon falsify run`, resumable): search a scenario\'s parameters for runs that '
      'violate a requirement (a formula, a rule SET:ID or a saved name), maximising σ(−ρ/scale) so ρ < 0 ⇔ value > '
      '0.5. Methods: cmaes (default), cem, bayes, bayes-pof, adaptive, random. Returns the job and the study id; when '
      'it ends, analysis_study on it or formal_stl_robustness on its runs; falsify.json in the study holds the best '
      'ρ, the falsified runs and the near misses (0 ≤ ρ < margin).',
      C.obj({'scenario': C.s('scenario key or id'), **_REQ,
             'method': C.s('search method', enum=['cmaes', 'cem', 'bayes', 'bayes-pof', 'adaptive', 'random']),
             'budget': C.i('runs (default 60)', minimum=4, maximum=2000),
             'batch': C.i('runs per batch (default 4)', minimum=1, maximum=64),
             'scale': C.n('ρ scale (default 1)', exclusiveMinimum=0), 'smooth': C.n('soft-min β (0 exact)', minimum=0),
             'margin': C.n('near-miss margin of ρ (default 1)', minimum=0), 'world': C.s('world id'),
             'rig': C.s('rig id'), 'seed': C.i('seed')}, ['scenario']),
      estimate=_falsify_estimate, preview=_falsify_preview)
def formal_falsify_stl(ctx, **a):
    kind, title, argv, kw, study = _falsify(ctx, **a)
    out = C.cli_job(ctx, kind, title, argv, **kw)
    out['study'] = study
    return out


# ================================================================================================ the safety case
def _G():
    from ...core import gsn as G
    return G


@tool('formal_gsn_tree', 'formal', 'read',
      'The workspace\'s living GSN safety case (safety/gsn.json, core.gsn) verified now (not logged): each node\'s '
      'id, type, text, status (green holds; amber evidence missing or changed — re-bind; red a claim violated; grey '
      'undeveloped; null for context nodes) and why, with AND propagation up the SupportedBy links; Solutions\' '
      'evidence (kind, ref, claim, pinned hash prefix); the top goals; the recent events. node: one node only.',
      C.obj({'node': C.s('one node id', maxLength=32),
             'events': C.i('recent events (default 5)', minimum=0, maximum=50)}))
def formal_gsn_tree(ctx, node='', events=5):
    G = _G()
    ws = C.workspace(ctx)
    try:
        doc = G.load(ws)
    except G.GSNError as e:
        raise ToolError(str(e)) from None
    v = G.verify(ws, doc)
    kids = G.children(doc)
    rows = []
    for n in doc['nodes']:
        if node and n['id'] != node:
            continue
        r = v['nodes'][n['id']]
        row = {'id': n['id'], 'type': n['type'], 'text': C.clip(n.get('text', ''), 200), 'status': r['status'],
               'why': C.clip(r.get('why') or '', 200), 'children': kids.get(n['id'], [])}
        ev = n.get('evidence')
        if ev:
            row['evidence'] = {'kind': ev['kind'], 'ref': ev['ref'], 'claim': ev.get('claim'),
                               'sha256': (ev.get('sha256') or '')[:12] or None, 'bound_at': ev.get('bound_at')}
        rows.append(row)
    if node and not rows:
        raise ToolError(f'no node {node!r}')
    return {'title': doc.get('title'), 'template': doc.get('template'), 'top': v['top'], 'counts': v['counts'],
            'nodes': rows, 'events': [{k: e.get(k) for k in ('at', 'event', 'by', 'node', 'top')}
                                      for e in G.events(ws, int(events))] if events else []}


def _gsn_verify_preview(ctx, rebind=''):
    return (f're-bind {rebind} to its result as it is now (pins the new hash; the previous evidence is in the event '
            'log), then verify' if rebind else 'verify the safety case and append the result to '
                                               'safety/gsn-events.jsonl')


@tool('formal_gsn_verify', 'formal', 'act',
      'Re-verify the GSN safety case and append the verification to safety/gsn-events.jsonl; with rebind, first '
      're-bind that Solution to its result as it is now (accepting a changed result: pins its new SHA-256, keeps the '
      'claim; the previous evidence is logged so it can be restored). Returns the top goals, the counts and the '
      'nodes that are not green.',
      C.obj({'rebind': C.s('a Solution id to re-bind first', maxLength=32)}), preview=_gsn_verify_preview)
def formal_gsn_verify(ctx, rebind=''):
    import copy
    G = _G()
    ws = C.workspace(ctx)
    try:
        doc = G.load(ws)
        if rebind:
            n = G.node(doc, rebind)
            old = copy.deepcopy(n.get('evidence'))
            if not old:
                raise ToolError(f'{rebind} has no evidence to re-bind')
            ev = G.bind(ws, doc, rebind, old['kind'], old['ref'], keep_claim=True)
            G.save(ws, doc)
            G.log_event(ws, 'bind', 'assistant', node=rebind, evidence=ev, previous=old, gsn_sha256=G.doc_sha(doc))
        v = G.verify_and_log(ws, 'assistant', doc=doc)
    except G.GSNError as e:
        raise ToolError(str(e)) from None
    bad = [{'id': k, 'status': r['status'], 'why': C.clip(r['why'], 200)} for k, r in v['nodes'].items()
           if r['status'] not in (None, 'green')]
    return {'top': v['top'], 'counts': v['counts'], 'not_green': bad[:30], 'rebound': rebind or None}
