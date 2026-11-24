"""4.1 assistant tools: fault tolerance campaigns (research_faulttol_*) and regulation packs (formal_regpack_*).
Registered at import (see tools/__init__).

    research_faulttol_run   act   `falcon faulttol run` queued: offline on recorded runs, or closed loop on a scenario
    research_faulttol_show  read  a campaign's tolerance table, most fragile sensor and derived calibration statements
    formal_regpack_run      act   `falcon regpack run` queued: a pack's cases as stack runs, scored
    formal_regpack_show     read  the packs, one pack's cases and unverified values, or a result's verdict and cases

A regulation pack is a simulation pre-check, not a type-approval or a rating: every formal_regpack_* result carries
core.regpacks.DISCLAIMER and the values that were not verified in the document's text, so the model cannot present
a pass as an approval or an unverified number as the regulation's. Runs are read through the privacy guard
(C.resolve_run).
"""
from __future__ import annotations

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact

MINE = {
    'research_faulttol_run': 'Queue a sensor fault tolerance campaign: bisect how far each fault goes before failing.',
    'research_faulttol_show': 'A fault campaign\'s tolerance table (sensor × fault), most fragile sensor, statements.',
    'formal_regpack_run': 'Queue a regulation pack (UN R157, Euro NCAP AEB) as scored stack runs: a pre-check only.',
    'formal_regpack_show': 'Regulation packs: the list, a pack\'s cases and unverified values, or a result\'s verdict.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)

_NAME = r'^[A-Za-z0-9][\w.-]{0,63}$'


def _FT():
    from ...core import faulttol as FT
    return FT


def _RP():
    from ...core import regpacks as RP
    return RP


# ============================================================================================== fault tolerance
def _faulttol(ctx, name, runs=None, scenario='', sensors=None, faults=None, model='', criterion='degraded',
              require=None, seed=0, dry=False):
    from .jobs import _fake, _model, _saved, _scenario
    FT = _FT()
    if bool(runs) == bool(scenario):
        raise ToolError('give recorded runs (offline) or a scenario (closed loop), not both or neither')
    for f in faults or []:
        if f not in FT.MODELS:
            raise ToolError(f'unknown fault {f!r}: one of {", ".join(FT.MODELS)}')
    try:
        FT.parse_requirements([f'{k}={v}' for k, v in (require or {}).items()])
    except FT.FaultTolError as e:
        raise ToolError(str(e)) from None
    if not dry:
        C.out_dir(ctx, 'faulttol', name)
    path = (C.project(ctx).path if dry else _saved(ctx)) or '<the project file>'
    argv = ['faulttol', 'run', path, '--name', name, '--seed', str(int(seed))]
    for s in sensors or []:
        argv += ['--sensor', s]
    for f in faults or []:
        argv += ['--fault', f]
    for k, v in (require or {}).items():
        argv += ['--require', f'{k}={float(v):g}']
    if runs:
        argv += ['--runs'] + [C.resolve_run(ctx, r) for r in runs]       # each through the privacy guard
        if model:
            argv += ['--model', _model(ctx, model)['id']]
        return 'faulttol', f'Fault tolerance {name} (offline, {len(runs)} runs)', argv, {'memory_gb': 2.0}
    scn = _scenario(ctx, scenario)
    if criterion not in FT.CLOSED_CRITERIA or criterion == 'stl':
        raise ToolError('criterion: degraded, criticality or fail (stl from the command line)')
    argv += ['--closed-loop', '--scenario', scn.get('id') or scenario, '--criterion', criterion]
    if _fake(ctx):
        argv += ['--fake-carla', _fake(ctx)]
    return 'faulttol', f'Fault tolerance {name} (closed loop, {scn["name"]})', argv, {'memory_gb': 0.5}


def _faulttol_preview(ctx, **a):
    _k, _t, argv, _kw = _faulttol(ctx, dry=True, **a)
    return C.command_line(argv)


def _faulttol_estimate(ctx, **a):
    FT = _FT()
    if a.get('runs'):
        return {'seconds': None, 'what': 'offline: no simulator; seconds per cell without a model, the model\'s '
                                         'time × about 11 evaluations × the frames per cell with one'}
    from .jobs import _drive_estimate, _scenario, _world_rig
    scn = _scenario(ctx, a['scenario'])
    _w, r = _world_rig(ctx, '', '')
    cells = len(a.get('faults') or FT.CLOSED) * max(len(a.get('sensors') or []), 1)
    return _drive_estimate(ctx, r, int(float(scn.get('duration_s', 12)) * 20),
                           cells * (FT.DEFAULTS['grid'] + FT.DEFAULTS['steps']))


@tool('research_faulttol_run', 'research', 'act',
      'Queue a sensor fault and calibration-drift tolerance campaign (`falcon faulttol run`, core.faulttol): for each '
      '(sensor, fault) the severity at which a criterion flips is bracketed by bisection, with common random numbers '
      'and a monotonicity check. Offline (runs: recorded run ids; no simulator): cameras by the share of labelled '
      'objects displaced beyond IoU 0.5 (or, with model, a detector\'s AP drop), lidars by actor-point retention. '
      'Closed loop (scenario): stack runs with the fault from 1 s; criterion degraded (the stack\'s health monitor '
      'trips), criticality or fail. Faults: cam_yaw, cam_pitch, cam_roll, lidar_yaw, lidar_sector, clock_offset, '
      'dirty_lens, frozen, dropped, noise, exposure, gnss_bias (closed loop: lidar_yaw, lidar_sector, dropped, '
      'gnss_bias, dirty_lens, noise). Returns the job; when it ends, research_faulttol_show on the name.',
      C.obj({'name': C.s('the campaign\'s name (a new folder under faulttol/)', pattern=_NAME),
             'runs': C.arr(C.s('run id'), 'offline: recorded run ids', maxItems=20),
             'scenario': C.s('closed loop: scenario key or id'),
             'sensors': C.arr(C.s('sensor id'), 'sensors (default all)', maxItems=16),
             'faults': C.arr(C.s('fault model'), 'fault models (default all that apply)', maxItems=12),
             'model': C.s('offline: a registered detector id (AP drop)'),
             'criterion': C.s('closed loop criterion', enum=['degraded', 'criticality', 'fail']),
             'require': C.obj(description='required severity per fault, e.g. {"cam_yaw": 0.5}',
                              additionalProperties={'type': 'number'}),
             'seed': C.i('seed')}, ['name']),
      estimate=_faulttol_estimate, preview=_faulttol_preview)
def research_faulttol_run(ctx, **a):
    kind, title, argv, kw = _faulttol(ctx, **a)
    out = C.cli_job(ctx, kind, title, argv, needs_carla=bool(a.get('scenario')), **kw)
    out['campaign'] = a['name']
    return out


@tool('research_faulttol_show', 'research', 'read',
      'A fault tolerance campaign (faulttol/<name>/faulttol.json): per (sensor, fault) the largest tolerated '
      'severity with its unit and bracket [last pass, first fail], the status (bracketed, tolerates_max: no failure '
      'up to the largest tried, fails_nominal, not_monotone: the lowest failure of a coarse grid, refused with why), '
      'the criterion, the user\'s requirement and whether it is met, the most fragile sensor, and the DERIVED '
      'calibration statements (of this stack and these runs, not general requirements). cell "SENSOR:FAULT" adds '
      'that cell\'s response curve. Without a name: the campaigns and the fault models.',
      C.obj({'name': C.s('campaign name', maxLength=64), 'cell': C.s('SENSOR:FAULT for its curve', maxLength=120)}))
def research_faulttol_show(ctx, name='', cell=''):
    FT = _FT()
    ws = C.workspace(ctx)
    if not name:
        return {'campaigns': FT.campaigns(ws),
                'fault_models': [{'fault': k, 'unit': v['unit'], 'max': v['max'], 'sensors': list(v['sensors']),
                                  'closed_loop': k in FT.CLOSED, 'model': v['model']} for k, v in FT.MODELS.items()]}
    d = C.resolve_in(ctx, 'faulttol', name, 'fault tolerance campaign', must=('faulttol.json',))
    try:
        doc = FT.load(d)
    except (OSError, ValueError, FT.FaultTolError) as e:
        raise ToolError(str(e)) from None
    rows = []
    for c, r in zip(doc['cells'], FT.table_rows(doc)):
        row = {k: (C.rounded(v) if isinstance(v, float) else v) for k, v in r.items() if v not in ('', None) or
               k == 'tolerated'}
        if row.get('note'):
            row['note'] = C.clip(row['note'], 200)
        if cell and cell == f'{c["sensor"]}:{c["fault"]}':
            row['curve'] = [[C.rounded(x), C.rounded(y), f] for x, y, f in c.get('curve') or []][:40]
            row['response_unit'] = c.get('response_unit')
        rows.append(row)
    if cell and not any('curve' in r for r in rows):
        have = ', '.join(c['sensor'] + ':' + c['fault'] for c in doc['cells'][:12])
        raise ToolError(f'no cell {cell!r}: SENSOR:FAULT of {have}')
    return {'campaign': name, 'mode': doc['mode'], 'inputs': C.no_media(doc.get('inputs') or {}), **C.capped(rows),
            'most_fragile': doc.get('most_fragile'),
            'derived_calibration': [C.clip(s['statement'], 400) for s in doc.get('calibration') or []],
            'derived_note': FT.STATEMENT_NOTE, 'limits': doc.get('limits')}


# ============================================================================================= regulation packs
def _regpack(ctx, pack, name, cases=None, groups=None, driver='follower', duration_scale=1.0, world='', rig='',
             dry=False):
    from .jobs import _fake, _saved, _world_rig
    RP = _RP()
    try:
        p = RP.pack(pack)
        chosen = RP.select(p, cases, groups) if p['kind'] != 'checklist' else []
    except RP.RegPackError as e:
        raise ToolError(str(e)) from None
    if not dry:
        C.out_dir(ctx, 'regpacks', name)
    w, r = _world_rig(ctx, world, rig)
    path = (C.project(ctx).path if dry else _saved(ctx)) or '<the project file>'
    argv = ['regpack', 'run', path, pack, '--name', name, '--driver', driver, '--world', w['id'], '--rig', r['id']]
    for c in cases or []:
        argv += ['--case', c]
    for g in groups or []:
        argv += ['--group', g]
    if float(duration_scale) != 1.0:
        argv += ['--duration-scale', f'{float(duration_scale):g}']
    if _fake(ctx):
        argv += ['--fake-carla', _fake(ctx)]
    n = sum(RP.runnable(c) for c in chosen)
    return 'regpack', f'Regulation pack {pack}: {n} runs (pre-check)', argv, {'memory_gb': 0.5}, chosen, r


def _regpack_preview(ctx, **a):
    out = _regpack(ctx, dry=True, **a)
    return C.command_line(out[2]) + '   # a simulation pre-check, not a type-approval or a rating'


def _regpack_estimate(ctx, **a):
    from .jobs import _drive_estimate
    RP = _RP()
    _k, _t, _argv, _kw, chosen, r = _regpack(ctx, dry=True, **a)
    run = [c for c in chosen if RP.runnable(c)]
    if not run:
        return {'seconds': 0.0, 'what': 'nothing runs (a checklist, or every chosen case is refused by name)'}
    steps = int(sum(float(c['scenario']['duration_s']) for c in run) / len(run) * 20)
    return _drive_estimate(ctx, r, steps, len(run))


@tool('formal_regpack_run', 'formal', 'act',
      'Queue a regulation test pack (`falcon regpack run`, core.regpacks): r157 (UN R157 ALKS up to 60 km/h: lane '
      'keeping, obstacles, following distance, cut-in and cut-out against the regulation\'s driver performance '
      'model, the MRM), ncap_aeb_c2c (Euro NCAP 2023 CCRs/CCRm/CCRb grids scored by impact-speed band) or iso34502 '
      '(a coverage checklist; nothing runs). ncap_lss lists its grid but every case is refused by name. Each case '
      'is one stack run with the chosen driver, scored from the run. THE RESULT IS A SIMULATION PRE-CHECK: not a '
      'type-approval, not a Euro NCAP rating; Falcon is not a qualified tool — say so whenever you report it. '
      'Returns the job; then formal_regpack_show with the result name.',
      C.obj({'pack': C.s('the pack', enum=['r157', 'ncap_aeb_c2c', 'ncap_lss', 'iso34502']),
             'name': C.s('the result\'s name (a new folder under regpacks/)', pattern=_NAME),
             'cases': C.arr(C.s('case id'), 'case ids (default all)', maxItems=120),
             'groups': C.arr(C.s('case group'), 'case groups, e.g. cut-in, CCRs', maxItems=10),
             'driver': C.s('the system under test', enum=['follower', 'autopilot']),
             'duration_scale': C.n('shorten the 5-minute lane keeping (result: incomplete)', exclusiveMinimum=0,
                                   maximum=1),
             'world': C.s('world id'), 'rig': C.s('rig id')}, ['pack', 'name']),
      estimate=_regpack_estimate, preview=_regpack_preview)
def formal_regpack_run(ctx, **a):
    RP = _RP()
    kind, title, argv, kw, _chosen, _r = _regpack(ctx, **a)
    out = C.cli_job(ctx, kind, title, argv, needs_carla=a['pack'] not in ('iso34502', 'ncap_lss'), **kw)
    out.update(result=a['name'], disclaimer=RP.DISCLAIMER)
    return out


@tool('formal_regpack_show', 'formal', 'read',
      'Regulation packs. No argument: the packs (cases, how many run, unverified values, sources). pack: its cases '
      'with clause and whether mandatory, informative or refused, and its values with their document, clause and '
      'whether verified in the document\'s text. result: a finished run (regpacks/<name>/regpack.json) — the '
      'verdict or score, each case\'s result with its clause and what was measured, the unverified values, Falcon\'s '
      'own choices and the limits. Always carries the disclaimer: a simulation pre-check, not a type-approval or a '
      'rating; never present an unverified value as the regulation\'s.',
      C.obj({'pack': C.s('a pack id', maxLength=32), 'result': C.s('a result name', maxLength=64),
             'group': C.s('only this case group', maxLength=40)}))
def formal_regpack_show(ctx, pack='', result='', group=''):
    RP = _RP()
    if result:
        d = C.resolve_in(ctx, 'regpacks', result, 'regulation pack result', must=('regpack.json',))
        try:
            doc = RP.load(d)
        except (OSError, ValueError, RP.RegPackError) as e:
            raise ToolError(str(e)) from None
        rows = [{'case': r['id'], 'clause': r['clause'], 'result': r['word'], 'mandatory': r['mandatory'],
                 'uses_unverified_value': bool(r.get('unverified')),
                 'detail': C.clip(r.get('why') or '; '.join(c['detail'] for c in r.get('checks') or []), 220),
                 **({'note': C.clip(r['note'], 200)} if r.get('note') else {})}
                for r in doc['cases'] if not group or r['group'] == group]
        out = {'disclaimer': doc['disclaimer'], 'pack': doc['pack'], 'result': result, 'verdict': doc['verdict'],
               'verdict_note': doc.get('verdict_note'), 'score': doc.get('score'), 'counts': doc.get('counts'),
               **C.capped(rows, 60, 'narrow with group'), 'unverified_values': doc['unverified'][:20],
               'falcon_choices': [c['note'] for c in doc.get('choices') or []][:8], 'limits': doc.get('limits'),
               'report': 'regpacks/' + result + '/report.html'}
        if doc['kind'] == 'checklist':
            out['items'] = [{k: i[k] for k in ('item', 'status', 'scenarios', 'note')} for i in doc['items']
                            if not group or i['category'] == group]
        return out
    if not pack:
        return {'disclaimer': RP.DISCLAIMER, 'packs': [RP.summary(p) for p in RP.PACKS],
                'results': RP.results(C.workspace(ctx)) if ctx.project is not None else []}
    try:
        p = RP.pack(pack)
    except RP.RegPackError as e:
        raise ToolError(str(e)) from None
    cases = [{'case': c['id'], 'group': c['group'], 'clause': c['clause'],
              'kind': 'refused' if c.get('refuse') else 'mandatory' if c['mandatory'] else 'informative',
              **({'why': C.clip(c['refuse'], 160)} if c.get('refuse') else {})}
             for c in p['cases'] if not group or c['group'] == group]
    vals = [{'name': n, 'value': C.clip(RP._val(v['value']), 120), 'unit': v['unit'],
             'document': RP.SOURCES[v['source']]['title'][:60], 'clause': v['clause'],
             'verified': v['verified'], **({'why': RP.BASIS_TEXT[v['basis']]} if v['verified'] is False else {})}
            for n, v in RP.values_of(p) if '.' not in n]
    return {'disclaimer': RP.DISCLAIMER, 'pack': pack, 'title': p['title'], 'about': p['about'],
            'groups': sorted({c['group'] for c in p['cases']}), **C.capped(cases, 40, 'narrow with group'),
            'values': vals, 'sources': [RP.SOURCES[s] for s in p['sources']], 'limits': p['limits'],
            **({'items': [{k: i[k] for k in ('item', 'status', 'scenarios')} for i in p['items']]}
               if p['kind'] == 'checklist' else {})}
