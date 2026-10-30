"""Run tools: search the workspace's recorded runs, read one compactly, its frames and events, compare runs, measure a
dataset's health; tags and drive sets (project edits); and the trash (core.trash), the only way a run leaves the list.

Everything comes from what the runs wrote (the run index, core.runindex, reads run.json and frames.jsonl; core.runcompare
derives events; core.datahealth measures labels) — a fact a run did not record is None ("not recorded"). Runs derived
from the user's dashcam footage are private: searches leave them out (and say how many) and every tool naming one is
refused (privacy.guard). Results carry ids and statistics, never pixels, points or audio.
"""
from __future__ import annotations

import os

from ..privacy import private_reason
from ..tool_api import ToolError, tool
from . import _common as C

ROW_KEYS = ('name', 'started', 'status', 'kind', 'world', 'map', 'rig', 'weather', 'tod', 'scenario', 'seed', 'fps',
            'frames', 'duration_s', 'distance_m', 'speed_mean_kmh', 'collisions', 'min_ttc_s', 'passed', 'events',
            'tags', 'rule_violations', 'odd_outside_s')


def _entries(ctx) -> tuple:
    """(the index's entries with tags, inside the workspace and not private; how many were left out as private)."""
    from ...core import runindex as RI
    ws = C.workspace(ctx)
    es = RI.with_tags(RI.build(ws), C.doc(ctx), ws)
    out, hidden = [], 0
    for e in es:
        if not C.inside(e['path'], ws):
            continue
        if e.get('dashcam_route') or private_reason(e['path']):
            hidden += 1
            continue
        out.append(e)
    return out, hidden


def _row(ctx, e) -> dict:
    r = {'id': e['rel']}
    r.update({k: C.rounded(e.get(k), 3) for k in ROW_KEYS if e.get(k) not in (None, [], {})})
    if e.get('error'):
        r['error'] = C.clip(e['error'], 160)
    return r


@tool('runs_search', 'runs', 'read',
      'Search the workspace\'s runs in the run query language (docs/runs.md): terms ANDed, key:value substring '
      '(world: map: weather: tod: kind: status: scenario: rig: name: folder:), membership (events:collision, tag:x), '
      'numbers (dur>20, frames>=100, speed>50, dist, ttc<1.5, collisions>0, rain, fog, sun, seed=3), dates '
      '(after:2026-10-28 before:), -term negates, other words are free text. Or list a drive set\'s runs. Returns '
      'compact rows with each run\'s id (use it in the other run tools), newest first unless sorted. Runs from the '
      'user\'s dashcam footage are never listed.',
      C.obj({'query': C.s('the query (empty: every run)'), 'drive_set': C.s('a drive set id instead of a query'),
             'sort': C.s('sort key', enum=['date', 'name', 'world', 'weather', 'duration', 'status', 'frames']),
             'limit': C.i('rows (default 20, at most 50)', minimum=1, maximum=50)}))
def runs_search(ctx, query='', drive_set='', sort='date', limit=20):
    from ...core import runindex as RI
    es, hidden = _entries(ctx)
    try:
        if drive_set:
            _i, ds = C.find(ctx, 'drive_sets', drive_set, 'drive set')
            want = set(RI.resolve_set(ds, es, C.workspace(ctx)))
            rows = [e for e in es if e['path'] in want]
        else:
            rows = RI.search(es, query or '')
    except RI.QueryError as e:
        raise ToolError(f'query: {e}') from e
    rows = RI.sort(rows, sort)
    out = C.capped([_row(ctx, e) for e in rows], int(limit), 'narrow the query or raise limit (≤ 50)')
    if hidden:
        out['private_left_out'] = hidden
    return out


def _labels_present(run_dir: str) -> dict:
    """Which label and sensor streams the run holds, with file counts (names only)."""
    out = {}
    for top in ('rgb', 'semantic', 'instance', 'range', 'flow', 'objects', 'lidar', 'radar', 'gnss', 'imu', 'audio'):
        d = os.path.join(run_dir, top)
        if not os.path.isdir(d):
            continue
        subs = {}
        for s in sorted(os.listdir(d))[:12]:
            sd = os.path.join(d, s)
            subs[s] = len(os.listdir(sd)) if os.path.isdir(sd) else 1
        out[top] = subs
    for f in ('frames.jsonl', 'actors.jsonl', 'trajectories.json', 'lanes.json', 'cpm.jsonl', 'odd-monitor.json',
              'dataset.json', 'drive.json'):
        if os.path.isfile(os.path.join(run_dir, f)):
            out.setdefault('files', []).append(f)
    return out


@tool('runs_get', 'runs', 'read',
      'One run, compactly: what ran (worker, status, error, start/end, name), its settings (world, rig, map, seed, '
      'step, steps, weather preset and values, scenario and its sampled parameter values, traffic counts), '
      'measured performance (steps/s, real-time factor, Mpx per step), the index summary (distance, speeds, '
      'collisions, min TTC, events, tags), the scenario checks and criticality summary, trigger rules fired, the ODD '
      'monitor\'s time outside, which sensor and label streams it holds (file counts), and its notes.',
      C.obj({'run': C.s('a run id from runs_search')}, ['run']))
def runs_get(ctx, run):
    from ...core import runindex as RI
    d = C.resolve_run(ctx, run)
    r = C.read_json(os.path.join(d, 'run.json'), {})
    prov = r.get('provenance') or {}
    st = prov.get('settings') or {}
    e = RI.with_tags([RI.summarise(d, C.workspace(ctx)) or {}], C.doc(ctx), C.workspace(ctx))[0]
    traffic = st.get('traffic') or {}
    scn = st.get('scenario') or {}
    out = {'id': C.run_id(ctx, d), 'name': r.get('name'), 'worker': r.get('worker'), 'status': r.get('status'),
           'exit_code': r.get('exit_code'), 'error': C.clip(r.get('error'), 300) if r.get('error') else None,
           'started': r.get('started'), 'finished': r.get('finished'), 'app_version': prov.get('app_version'),
           'settings': {'world': (st.get('project') or {}).get('world'), 'rig': (st.get('project') or {}).get('rig'),
                        'town': st.get('town') or None, 'imported_world': bool(st.get('xodr')),
                        'seed': st.get('seed', prov.get('seed')), 'fps': st.get('fps'), 'steps': st.get('steps'),
                        'weather_preset': (st.get('weather_source') or {}).get('preset'),
                        'weather': C.rounded(st.get('weather'), 2),
                        'traffic': {'vehicles': traffic.get('vehicles'), 'walkers': traffic.get('walkers'),
                                    'profile': (st.get('project') or {}).get('traffic_profile')} if traffic else None,
                        'scenario': {'id': scn.get('id'), 'name': scn.get('name'),
                                     'values': C.rounded(scn.get('values'), 3)} if scn else None,
                        'sensors': [f'{s.get("id")} ({s.get("type")})' for s in (st.get('rig') or {}).get(
                            'sensors') or []][:16]},
           'performance': C.rounded({k: v for k, v in (r.get('performance') or {}).items()
                                     if isinstance(v, (int, float))}, 3),
           'summary': {k: C.rounded(e.get(k), 3) for k in ROW_KEYS if e.get(k) not in (None, [], {})},
           'streams': _labels_present(d)}
    sres = r.get('scenario') or {}
    if sres:
        crit = sres.get('criticality') or {}
        out['scenario_result'] = C.rounded({'passed': sres.get('passed'), 'checks': sres.get('checks'),
                                            'criticality': crit.get('criticality'),
                                            'overall': crit.get('overall')}, 3)
    if (r.get('trigger_rules') or {}).get('fired'):
        out['trigger_rules_fired'] = r['trigger_rules']['fired']
    if r.get('odd_monitor'):
        out['odd_monitor'] = C.rounded({k: v for k, v in r['odd_monitor'].items() if not isinstance(v, (list, dict))})
    if r.get('faults_applied'):
        out['faults_applied'] = C.no_media(r['faults_applied'])[:10] if isinstance(r['faults_applied'], list) else \
            r['faults_applied']
    notes = r.get('notes') or []
    if notes:
        out['notes'] = [C.clip(n, 240) for n in notes[:8]]
    return C.no_media(out)


@tool('runs_frames', 'runs', 'read',
      'A run\'s recorded frames as numbers, a window at a time: per frame the step, sim time, ego speed (km/h), '
      'position and yaw, controls, which sensor files it has (names), and for a camera the labelled objects\' count '
      'by class. Never images: show one in the window with ui_show_frame.',
      C.obj({'run': C.s('a run id'), 'start': C.i('first frame index (default 0)', minimum=0),
             'count': C.i('frames (default 10, at most 40)', minimum=1, maximum=40),
             'camera': C.s('a camera sensor id for object counts (default: the first with objects)')}, ['run']))
def runs_frames(ctx, run, start=0, count=10, camera=''):
    d = C.resolve_run(ctx, run)
    rows = C.read_jsonl(os.path.join(d, 'frames.jsonl'))
    if not rows:
        raise ToolError('the run recorded no frames (no frames.jsonl)')
    objs = os.path.join(d, 'objects')
    cams = sorted(os.listdir(objs)) if os.path.isdir(objs) else []
    cam = camera or (cams[0] if cams else '')
    if camera and camera not in cams:
        raise ToolError(f'no objects for camera {camera!r} (cameras with objects: {", ".join(cams) or "none"})')
    out = []
    t0 = float(rows[0].get('sim_t') or 0.0)
    for k, f in enumerate(rows[start:start + count], start):
        ego = f.get('ego') or {}
        row = {'index': k, 'step': f.get('step'), 't_s': round(float(f.get('sim_t') or 0.0) - t0, 3),
               'speed_kmh': round(float(f['speed_mps']) * 3.6, 2) if f.get('speed_mps') is not None else None,
               'ego': C.rounded({kk: ego.get(kk) for kk in ('x', 'y', 'yaw') if kk in ego}, 2),
               'control': C.rounded(f.get('control'), 3), 'files': sorted((f.get('files') or {}).keys())[:12]}
        if f.get('teleport'):
            row['teleport'] = True
        rel = (f.get('files') or {}).get(f'{cam}/objects') if cam else None
        if rel and not C._bad_name(rel) and os.path.isfile(os.path.join(d, rel)):
            items = (C.read_json(os.path.join(d, rel), {}) or {}).get('objects') or []
            byc = {}
            for o in items:
                byc[o.get('class', '?')] = byc.get(o.get('class', '?'), 0) + 1
            row['objects'] = byc
        out.append(row)
    return {'run': C.run_id(ctx, d), 'frames_total': len(rows), 'camera': cam or None, 'items': out}


@tool('runs_events', 'runs', 'read',
      'A run\'s events on its own clock (core.runcompare): collisions, the minimum TTC, trigger rules fired, '
      'teleports (put back on the road) and intervals outside the ODD — each with its time in seconds.',
      C.obj({'run': C.s('a run id')}, ['run']))
def runs_events(ctx, run):
    from ...core import runcompare as RC
    d = C.resolve_run(ctx, run)
    ev = [C.rounded(x, 3) for x in RC.events(d)]
    return dict(C.capped(ev, 40), run=C.run_id(ctx, d))


@tool('runs_compare', 'runs', 'read',
      'Compare 2–4 runs side by side, as Data ▸ Compare does (core.runcompare): duration, step, frames, distance, '
      'mean and max speed, collisions, minimum TTC, teleports, time outside the ODD, rule violations and (with '
      'rules true, slower: the Indian traffic rules are checked) lane margins; each cell with its difference from '
      'the first run. "not recorded" where a run did not record it.',
      C.obj({'runs': C.arr(C.s('run id'), '2–4 run ids', minItems=2, maxItems=4),
             'rules': C.b('also compute the traffic-rule metrics (slow)')}, ['runs']))
def runs_compare(ctx, runs, rules=False):
    from ...core import runcompare as RC
    from ...core import runindex as RI
    dirs = [C.resolve_run(ctx, r) for r in runs]
    ws = C.workspace(ctx)
    built = {e['path']: e for e in RI.build(ws)}
    entries = [built.get(os.path.abspath(d)) or RI.summarise(d, ws) for d in dirs]
    rm = [RC.rule_metrics(d) for d in dirs] if rules else None
    tab = RC.table(entries, rm)
    return {'runs': [C.run_id(ctx, d) for d in dirs],
            'rows': [{'metric': r['label'], 'unit': r['unit'],
                      'values': [c['text'] + (f' ({c["delta"]})' if c.get('delta') else '') for c in r['cells']]}
                     for r in tab]}


@tool('runs_health', 'runs', 'read',
      'A dataset\'s health over runs (Data ▸ Health, core.datahealth): frames read, class balance (objects per '
      'class), empty and labelled frames, tiny/degenerate/outside labels, near-duplicate frames (perceptual hash), '
      'coverage of the condition grid with its empty cells, and suggestions. Reads up to max_frames frames; takes '
      'seconds to minutes on large sets.',
      C.obj({'runs': C.arr(C.s('run id'), 'run ids', minItems=1, maxItems=200),
             'drive_set': C.s('or a drive set id'),
             'max_frames': C.i('frames to read at most (default 1500)', minimum=50, maximum=20000)}))
def runs_health(ctx, runs=None, drive_set='', max_frames=1500):
    from ...core import datahealth as DH
    from ...core import runindex as RI
    if drive_set:
        es, _ = _entries(ctx)
        _i, ds = C.find(ctx, 'drive_sets', drive_set, 'drive set')
        dirs = [C.checked(ctx, p) for p in RI.resolve_set(ds, es, C.workspace(ctx))]
    else:
        dirs = [C.resolve_run(ctx, r) for r in runs or []]
    if not dirs:
        raise ToolError('give runs or a drive set with runs')
    rep = DH.analyse(dirs, C.workspace(ctx), C.doc(ctx), max_frames=int(max_frames))
    keep = ('runs', 'frames', 'frames_read', 'every', 'classes', 'objects', 'labelled_frames', 'empty_frames',
            'coverage', 'suggestions', 'method')
    out = {k: rep.get(k) for k in keep if k in rep}
    lab = rep.get('labels') or {}
    out['labels'] = {k: lab.get(k) for k in ('tiny', 'degenerate', 'outside', 'tiny_rule') if k in lab}
    dup = rep.get('duplicates') or {}
    out['duplicates'] = {k: dup.get(k) for k in ('hashed', 'near_duplicates', 'share', 'bits') if k in dup}
    if isinstance(out.get('runs'), list):
        out['runs'] = len(out['runs'])
    return C.no_media(C.rounded(out, 3))


# ================================================================================================= tags and sets
def _paths(ctx, runs):
    return [C.resolve_run(ctx, r) for r in runs]


def _tag_edit(ctx, runs, tags):
    from ...core import runindex as RI
    return RI.tags_edit(C.doc(ctx), C.workspace(ctx), _paths(ctx, runs), tags)


@tool('runs_tag', 'runs', 'act',
      'Give runs exactly these tags (an empty list removes their tags): the project\'s run_tags, one undoable edit '
      '(a recorded run is never modified). Tags are searchable with tag:NAME.',
      C.obj({'runs': C.arr(C.s('run id'), 'run ids', minItems=1, maxItems=200),
             'tags': C.arr(C.s('tag'), 'the tags (empty: remove)')}, ['runs', 'tags']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _tag_edit(ctx, **a)))
def runs_tag(ctx, runs, tags):
    return C.apply_edit(ctx, _tag_edit(ctx, runs, tags))


def _set_edit(ctx, name, runs=None, query=''):
    from ...core import runindex as RI
    try:
        if query:
            return RI.drive_set_edit(C.doc(ctx), C.workspace(ctx), name, 'query', query=query)
        if not runs:
            raise ToolError('give runs or a query')
        return RI.drive_set_edit(C.doc(ctx), C.workspace(ctx), name, 'list', _paths(ctx, runs))
    except RI.QueryError as e:
        raise ToolError(f'query: {e}') from e


@tool('runs_drive_set_add', 'runs', 'act',
      'Save a drive set (a named group of runs used by training, evaluation and health): a fixed list of runs, or a '
      'saved query re-evaluated whenever it is used. One undoable project edit; remove one with project_remove kind '
      'drive_set. Returns its id.',
      C.obj({'name': C.s('its name'), 'runs': C.arr(C.s('run id'), 'run ids (a fixed list)', maxItems=500),
             'query': C.s('a run query (a saved query instead of a list)')}, ['name']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _set_edit(ctx, **a)[0]))
def runs_drive_set_add(ctx, name, runs=None, query=''):
    e, ds = _set_edit(ctx, name, runs, query)
    out = C.apply_edit(ctx, e)
    out['drive_set'] = ds['id']
    return out


# ================================================================================================= trash
def _refresh_window(ctx):
    if ctx.win is not None:
        p = ctx.win.page('data')
        if p is not None:
            p.browse.refresh()
    ctx.emit('runs_changed', {})


def _busy(ctx, dirs):
    from ...core.jobs import FINAL
    q = C.need_queue(ctx)
    for j in q.jobs:
        if j.state in FINAL:
            continue
        if j.spec.run_dir and os.path.realpath(j.spec.run_dir) in dirs:
            raise ToolError(f'job {j.id} ({j.spec.title}) is still writing {os.path.basename(j.spec.run_dir)}')
        used = [a for a in j.spec.argv if isinstance(a, str) and any(a == d or a.endswith(':' + d) for d in dirs)]
        if used:
            raise ToolError(f'job {j.id} ({j.spec.title}) is still reading {os.path.basename(used[0].split(":")[-1])}')


def _trash_preview(ctx, runs):
    from ...core import trash as TR
    ws = C.workspace(ctx)
    lines = []
    for r in runs:
        d = C.resolve_run(ctx, r)
        try:
            TR.check_run(ws, d)
        except TR.TrashError as e:
            raise ToolError(str(e)) from e
        lines.append(f'move {C.run_id(ctx, d)} → trash/<UTC stamp>-{os.path.basename(d)}/')
    return '\n'.join(lines) + '\n(restorable with runs_restore; nothing is deleted)'


@tool('runs_trash', 'runs', 'trash',
      'Move recorded runs (runs/<name>) into the workspace\'s trash folder, each with a manifest of where it came '
      'from, its catalogue entry and tags; the catalogue and run index forget it until it is restored. Nothing is '
      'deleted; runs_restore puts a run back. A study\'s runs and a run a job is still writing are refused. Always '
      'asks the user.',
      C.obj({'runs': C.arr(C.s('run id'), 'run ids', minItems=1, maxItems=100)}, ['runs']),
      estimate=lambda ctx, **a: {'seconds': 1.0, 'what': 'folders are moved within the workspace'},
      preview=_trash_preview)
def runs_trash(ctx, runs):
    from ...core import trash as TR
    dirs = [C.resolve_run(ctx, r) for r in runs]
    _busy(ctx, set(dirs))
    cat = C.need_service(ctx).catalog if (ctx.service is not None or ctx.win is not None) else None
    try:
        mans = TR.trash(C.workspace(ctx), dirs, C.doc(ctx), catalog=cat)
    except TR.TrashError as e:
        raise ToolError(str(e)) from e
    out = []
    for m in mans:
        moved = {'from': os.path.join(C.workspace(ctx), m['original']),
                 'to': os.path.join(TR.trash_dir(C.workspace(ctx)), m['item'], m['name']), 'item': m['item']}
        ctx.emit('trashed', {'from': moved['from'], 'to': moved['to']})       # the activity log can undo it
        out.append(moved)
    _refresh_window(ctx)
    return {'trashed': out, 'restore': 'runs_restore with the item'}


@tool('runs_trash_list', 'runs', 'read',
      'The workspace\'s trash: each trashed run\'s item name (for runs_restore), where it came from, when it was '
      'trashed (UTC), its size and the tags it had.')
def runs_trash_list(ctx):
    from ...core import trash as TR
    items = []
    for m in TR.list_items(C.workspace(ctx)):
        items.append({'item': m['item'], 'run': m['original'], 'trashed': m['trashed'],
                      'size_mb': round(m['size_bytes'] / 1e6, 1), 'tags': m.get('tags') or []})
    return C.capped(items, 40)


@tool('runs_restore', 'runs', 'act',
      'Put a trashed run back where it was (refused when something is there now) and index it again, with its '
      'catalogue links.', C.obj({'item': C.s('a trash item from runs_trash_list')}, ['item']),
      preview=lambda ctx, item: f'move trash/{item}/… back to its original place in runs/')
def runs_restore(ctx, item):
    from ...core import trash as TR
    ws = C.workspace(ctx)
    try:
        man = TR.read_manifest(ws, item)
        C.checked(ctx, os.path.join(TR.trash_dir(ws), item))
        cat = C.need_service(ctx).catalog if (ctx.service is not None or ctx.win is not None) else None
        dest = TR.restore(ws, item, catalog=cat)
    except TR.TrashError as e:
        raise ToolError(str(e)) from e
    _refresh_window(ctx)
    return {'restored': C.run_id(ctx, dest), 'was': man['original']}
