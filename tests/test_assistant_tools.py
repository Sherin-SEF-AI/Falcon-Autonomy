"""The assistant's tools (falcon_autonomy.assistant.tools) end to end against the fake carla: every schema valid and
documented, project edits (undoable in the window), a drive queued, waited for, searched and summarised, the trash and
restore, the privacy refusals (a dashcam-derived run, paths outside the workspace), what is never allowed, result sizes,
the window tools ("show me"), and `falcon assistant`."""
import json
import os
import threading
import time

import pytest

from conftest import FAKE
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401 (the GUI tests' fixtures)

from falcon_autonomy.assistant import tools as TOOLS
from falcon_autonomy.assistant.tool_api import GROUPS, NEVER, REGISTRY, ToolContext, call


@pytest.fixture
def ctx(project):
    events = []
    c = TOOLS.context(project.path, FAKE, emit=lambda kind, payload: events.append((kind, payload)))
    c.events = events
    yield c
    c.service.shutdown(grace_s=10.0)


def _wait(c, job, timeout=180):
    r = call(c, 'jobs_wait', {'jobs': [job], 'timeout_s': timeout})
    assert r['finished'] == [job], r
    return r['jobs'][0]


# ------------------------------------------------------------------------------------------------ the registry
def test_registry_schemas_and_doc():
    import jsonschema
    from falcon_autonomy.assistant.cli import doc_markdown, schema_tokens
    from falcon_autonomy.assistant.tools._compact import DETAILS, SHORT
    assert len(REGISTRY) >= 60
    assert not {'plan_update', 'plan_wait_jobs'} & set(REGISTRY)              # the runtime's own meta-tools
    for t in REGISTRY.values():
        jsonschema.Draft202012Validator.check_schema(t.params)
        jsonschema.Draft202012Validator.check_schema(DETAILS[t.name]['params'])
        assert t.description == SHORT[t.name] and 15 < len(t.description) <= 120, t.name
        assert t.description.count('. ') <= 1 and len(DETAILS[t.name]['description']) > 40, t.name
        assert t.name.split('_')[0] == t.group, t.name
        assert t.name not in NEVER, t.name
        if t.tier != 'read':
            assert t.preview is not None, f'{t.name} acts without a preview'
        assert t.gui == (t.group == 'ui'), t.name
    assert {t.group for t in REGISTRY.values()} == set(GROUPS)
    assert sum(t.tier == 'trash' for t in REGISTRY.values()) == 1
    toks = schema_tokens()                     # Groq's base tier: ~8,000 tokens a minute; groups load on demand
    assert toks['all'] < 10000 and max(v for k, v in toks.items() if k != 'all') < 3500, toks
    with open(os.path.join(os.path.dirname(__file__), '..', 'docs', 'assistant-tools.md'), encoding='utf-8') as fh:
        assert fh.read() == doc_markdown(), 'docs/assistant-tools.md is stale: falcon assistant tools --doc > it'


def test_never_and_unknown(ctx):
    for name in NEVER:
        assert 'never allowed' in call(ctx, name, {})['error']
    # the compact schemas leave bounds and additionalProperties out; the full schema is still enforced
    assert 'invalid arguments' in call(ctx, 'runs_search', {'limit': 500})['error']
    assert 'invalid arguments' in call(ctx, 'runs_search', {'colour': 'red'})['error']
    assert 'minimum' not in json.dumps(REGISTRY['runs_search'].params)
    assert 'no tool' in call(ctx, 'runs_delete', {})['error']
    assert 'needs Falcon' in call(ctx, 'ui_current', {})['error']             # window tools: not from the CLI


# ------------------------------------------------------------------------------------------------ project
def test_project_edits_from_the_command_line(ctx, project):
    from falcon_autonomy.core.project import Project
    r = call(ctx, 'project_world_add', {'town': 'Town03', 'weather_preset': 'WetNoon'})
    wid = r['world']
    assert 'fog_density' in REGISTRY['project_world_set'].preview(ctx, world=wid, weather={'fog_density': 40})
    assert 'edit' in call(ctx, 'project_world_set', {'world': wid, 'weather': {'fog_density': 40}})
    assert 'imported world' in call(ctx, 'project_world_set', {'world': wid, 'speed_limits': 'carla'})['error']
    assert 'within' in call(ctx, 'project_world_set', {'world': wid, 'weather': {'cloudiness': 900}})['error']
    s = call(ctx, 'project_sensor_add', {'rig': 'rig-default', 'type': 'audio_array', 'fields': {'mics': 4}})
    assert s['sensor'].startswith('audio_array')
    assert 'edit' in call(ctx, 'project_sensor_set', {'rig': 'rig-default', 'sensor': 'cam-front',
                                                      'fields': {'fov': 100}, 'pose': {'z': 1.7}})
    scn = call(ctx, 'project_scenario_create', {'copy_of': 'lead_brake'})['scenario']
    assert 'edit' in call(ctx, 'project_scenario_set', {'scenario': scn, 'add_actor': {'kind': 'walker'}})
    bad = call(ctx, 'project_scenario_set', {'scenario': scn, 'timeline': [{'trigger': {'type': 'nope'}}]})
    assert 'scenario check' in bad['error']
    assert 'edit' in call(ctx, 'project_traffic_set', {'profile': 'tp-default', 'vehicles': 4,
                                                       'lateral_filtering': {'enabled': True}})
    assert call(ctx, 'project_trigger_add', {'rule': 'when ego.speed > 40 -> marker fast'})['rule'] == 'rule1'
    err = call(ctx, 'project_trigger_add', {'rule': 'when ego.speed >> 40 -> marker x'})['error']
    assert 'column' in err                                           # the parser's own error, with its column
    assert call(ctx, 'project_trigger_set', {'id': 'rule1', 'enabled': False})['edit']
    assert call(ctx, 'project_trigger_check', {})['ok']
    rsu = call(ctx, 'project_rsu_place', {'world': 'world-town10', 'x': 10, 'y': 5})['rsu']
    assert 'imported' in call(ctx, 'project_rsu_junctions', {'world': 'world-town10'})['error']
    assert call(ctx, 'project_odd_set', {'target': 'safety_case', 'example': True})['written'] == 'odd.json'
    assert 'at least one rig' in call(ctx, 'project_remove', {'kind': 'rig', 'id': 'rig-default'})['error']
    assert call(ctx, 'project_remove', {'kind': 'rsu', 'id': rsu})['edit']
    saved = Project.load(project.path)                               # the CLI saves every edit, schema-checked
    d = saved.doc
    assert d['worlds'][1]['weather'] == {'preset': 'WetNoon', 'fog_density': 40.0}
    assert d['rigs'][0]['sensors'][0]['fov'] == 100 and d['rigs'][0]['sensors'][1]['mics'] == 4
    assert d['trigger_rules'][0]['enabled'] is False and d['infrastructure']['rsus'] == []
    assert d['traffic_profiles'][0]['lateral_filtering']['enabled'] is True
    assert len(d['scenarios'][0]['actors']) == 2
    ov = call(ctx, 'project_overview', {})
    assert [w['id'] for w in ov['worlds']] == ['world-town10', wid]


# ------------------------------------------------------------------------------------------------ runs and jobs
def test_drive_search_summarise_trash_restore(ctx):
    est = REGISTRY['jobs_drive'].estimate(ctx, seconds=1)
    assert est['seconds'] > 0 and 'steps' in est['what']
    assert 'record a drive' in REGISTRY['jobs_drive'].preview(ctx, seconds=1)
    r = call(ctx, 'jobs_drive', {'seconds': 1, 'seed': 3})
    run = r['run']
    j = _wait(ctx, r['job'])
    assert j['state'] == 'done', call(ctx, 'jobs_get', {'job': r['job'], 'log_lines': 30})
    rows = call(ctx, 'runs_search', {'query': 'seed=3 dur>0.5'})
    assert [x['id'] for x in rows['items']] == [run]
    assert call(ctx, 'runs_search', {'query': 'seed=4'})['total'] == 0
    assert 'query' in call(ctx, 'runs_search', {'query': 'dur>abc'})['error']
    g = call(ctx, 'runs_get', {'run': run})
    assert g['status'] == 'ok' and g['settings']['seed'] == 3 and g['settings']['steps'] == 20
    assert g['streams']['rgb'] == {'cam-front': 20} and g['performance']['steps'] == 20
    f = call(ctx, 'runs_frames', {'run': run, 'start': 2, 'count': 3})
    assert [x['index'] for x in f['items']] == [2, 3, 4] and f['frames_total'] == 20
    assert all('objects' in x for x in f['items'])                   # counts per class, never pixels
    assert 'items' in call(ctx, 'runs_events', {'run': run})
    cmp_ = call(ctx, 'runs_compare', {'runs': [run, run]})
    assert any(row['metric'] == 'Frames' and row['values'][0] == '20' for row in cmp_['rows'])
    assert call(ctx, 'runs_tag', {'runs': [run], 'tags': ['night']})['edit']
    assert call(ctx, 'runs_search', {'query': 'tag:night'})['total'] == 1
    ds = call(ctx, 'runs_drive_set_add', {'name': 'mine', 'runs': [run]})['drive_set']
    assert call(ctx, 'runs_search', {'drive_set': ds})['total'] == 1
    # the trash: moved, out of the index and catalogue, then back
    assert 'trash/' in REGISTRY['runs_trash'].preview(ctx, runs=[run])
    t = call(ctx, 'runs_trash', {'runs': [run]})
    item = t['trashed'][0]['item']
    ws = ctx.workspace
    moved = {'from': os.path.join(ws, run), 'to': os.path.join(ws, 'trash', item, os.path.basename(run))}
    assert {k: t['trashed'][0][k] for k in ('from', 'to')} == moved and os.path.isdir(moved['to'])
    assert ('trashed', moved) in ctx.events                          # the activity log can undo it
    assert not os.path.exists(os.path.join(ws, run))
    assert os.path.isfile(os.path.join(ws, 'trash', item, 'manifest.json'))
    assert call(ctx, 'runs_search', {})['total'] == 0
    assert ctx.service.catalog.get('run:' + os.path.basename(run)) is None
    lst = call(ctx, 'runs_trash_list', {})
    assert lst['items'][0]['item'] == item and lst['items'][0]['tags'] == ['night']
    assert call(ctx, 'runs_restore', {'item': item})['restored'] == run
    assert call(ctx, 'runs_search', {})['total'] == 1 and not os.path.exists(os.path.join(ws, 'trash', item))
    assert ctx.service.catalog.get('run:' + os.path.basename(run)) is not None
    assert 'not a trash item' in call(ctx, 'runs_restore', {'item': '../x'})['error']


def test_jobs_wait_times_out_and_cancel(ctx):
    r = call(ctx, 'jobs_drive', {'seconds': 600})
    t0 = time.monotonic()
    w = call(ctx, 'jobs_wait', {'jobs': [r['job']], 'timeout_s': 0.5})
    assert w['timed_out'] and w['finished'] == [] and time.monotonic() - t0 < 5
    assert 'error' in call(ctx, 'runs_trash', {'runs': [r['run']]})          # refused while its job writes it
    c = call(ctx, 'jobs_cancel', {'job': r['job']})
    assert c['job'] == r['job']
    j = _wait(ctx, r['job'], 60)
    assert j['state'] in ('cancelled', 'failed')
    assert 'finished already' in call(ctx, 'jobs_cancel', {'job': r['job']})['error']
    assert 'no job' in call(ctx, 'jobs_get', {'job': 999})['error']


def test_job_previews_are_the_pages_command_lines(ctx):
    pv = REGISTRY['jobs_search'].preview(ctx, scenario='lead_brake', budget=8, count='all', feasibility=False)
    assert pv.startswith('falcon search ') and '--count all' in pv and '--no-feasibility' in pv
    assert '--name search-lead_brake' in pv
    pv = REGISTRY['jobs_sweep'].preview(ctx, scenario='cut_in', samples=5)
    assert '--samples 5' in pv and '--name sweep-cut_in' in pv
    assert 'needs a registered model' in call(ctx, 'jobs_search', {'scenario': 'lead_brake',
                                                                   'objective': 'perception'})['error']
    assert 'no run' in call(ctx, 'jobs_export', {'run': 'runs/none', 'format': 'coco'})['error']
    from falcon_autonomy.core.exporters import export_argv
    assert export_argv('R', 'openocc', 'O') == ['export', 'R', '--format', 'occ3d', '--out', 'O', '--option',
                                                 'variant=openocc']


# ------------------------------------------------------------------------------------------------ privacy and paths
def _fake_run(ws, name, settings):
    d = os.path.join(ws, 'runs', name)
    os.makedirs(os.path.join(d, 'rgb', 'cam'))
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump({'schema': 'falcon.run/1', 'worker': 'drive', 'status': 'ok', 'name': name,
                   'started': '2026-10-01T00:00:00Z', 'provenance': {'settings': settings}}, fh)
    with open(os.path.join(d, 'frames.jsonl'), 'w') as fh:
        fh.write(json.dumps({'step': 0, 'frame': 0, 'sim_t': 0.0, 'files': {}}) + '\n')
    return d


def _args_for(t, run_id):
    from falcon_autonomy.assistant.tools._compact import DETAILS
    full = DETAILS[t.name]['params']                 # with its bounds (minItems …)
    out = {}
    props = full['properties']
    for k in list(full.get('required') or ()) + [k for k in ('run', 'runs') if k in props]:
        sch = props[k]
        if k == 'run':
            out[k] = run_id
        elif k == 'runs':
            out[k] = [run_id] * max(sch.get('minItems', 1), 1)
        elif 'enum' in sch:
            out[k] = sch['enum'][0]
        elif sch.get('type') == 'integer':
            out[k] = 0
        elif sch.get('type') == 'number':
            out[k] = 1.0
        elif sch.get('type') == 'array':
            out[k] = ['x']
        else:
            out[k] = 'x'
    return out


def test_private_runs_and_paths_outside_are_refused(ctx):
    ws = ctx.workspace
    _fake_run(ws, 'plain', {'seed': 1})
    _fake_run(ws, 'route', {'seed': 2, 'dashcam_route': {'route_sha256': 'x'}})
    s = call(ctx, 'runs_search', {})
    assert [x['id'] for x in s['items']] == ['runs/plain'] and s['private_left_out'] == 1
    with_run = [t for t in REGISTRY.values() if {'run', 'runs'} & set(t.params.get('properties') or {})
                and not t.gui]
    assert len(with_run) >= 20
    for t in with_run:
        args = _args_for(t, 'runs/route')
        r = call(ctx, t.name, args)
        assert 'error' in r, (t.name, r)
        # refused for the run itself, or before it for another id that does not exist here (no model, translator,
        # reconstruction or real set in this workspace)
        assert 'private' in r['error'] or t.name in ('jobs_evaluate', 'jobs_train_translator', 'jobs_translate_apply',
                                                     'jobs_hybrid'), (t.name, r)
    assert 'private' in call(ctx, 'runs_get', {'run': 'route'})['error']           # by its folder name too
    assert 'private' in call(ctx, 'runs_health', {'runs': ['runs/route']})['error']
    assert 'private' in call(ctx, 'runs_drive_set_add', {'name': 'x', 'runs': ['runs/route']})['error']
    for bad in ('../../etc', '/etc', 'runs/../../x', '~/x'):
        assert 'not a run id' in call(ctx, 'runs_get', {'run': bad})['error'], bad
    outside = os.path.join(os.path.dirname(ws), 'elsewhere')
    _fake_run(os.path.dirname(ws), 'elsewhere-run', {'seed': 3})
    os.symlink(os.path.join(os.path.dirname(ws), 'runs', 'elsewhere-run'), os.path.join(ws, 'runs', 'link'))
    assert 'outside' in call(ctx, 'runs_get', {'run': 'runs/link'})['error']
    assert not os.path.exists(outside)
    assert 'never reaches' in call(ctx, 'jobs_gap', {'runs': ['runs/plain'], 'real': 'dashcam:x'})['error']


def test_result_sizes_are_capped(ctx):
    ws = ctx.workspace
    for k in range(40):
        _fake_run(ws, f'r{k:02d}', {'seed': k})
    s = call(ctx, 'runs_search', {'limit': 5})
    assert len(s['items']) == 5 and s['total'] == 40 and s['more'] == 35
    assert len(call(ctx, 'runs_search', {})['items']) == 20
    for name, args in (('project_options', {}), ('project_scenarios', {}), ('knowledge_tools', {}),
                       ('knowledge_cli', {}), ('knowledge_manual', {'query': 'scenario sweep', 'limit': 8}),
                       ('runs_search', {'limit': 50}), ('system_machine', {})):
        n = len(json.dumps(call(ctx, name, args)))
        assert n < 40000, (name, n)
    m = call(ctx, 'knowledge_manual', {'query': 'trigger rules cooldown', 'limit': 2})
    assert len(m['sections']) == 2 and all(len(x['text']) <= 1200 for x in m['sections'])
    assert m['sections'][0]['link'].startswith('docs/')
    assert call(ctx, 'knowledge_setting', {'key': 'hil_device'})['settings'][0]['value'] == '(private: not shown)'
    assert call(ctx, 'knowledge_setting', {'key': 'rpc_port'})['settings'][0]['value'] == 2310
    assert any('export' in r['command'] for r in call(ctx, 'knowledge_cli', {'filter': 'export'})['items'])
    assert call(ctx, 'system_server', {})['state'] == 'fake'


# ------------------------------------------------------------------------------------------------ the command line
def test_falcon_assistant_cli(project, capsys):
    from falcon_autonomy.cli import main
    assert main(['assistant', 'tools', '--json']) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r['name'] for r in rows} == set(REGISTRY)
    assert main(['assistant', 'call', 'project_overview', '--project', project.path]) == 0
    assert json.loads(capsys.readouterr().out)['name'] == 'Test'
    assert main(['assistant', 'call', 'project_world_add', '{"town": "Town01"}', '--project', project.path]) == 2
    assert 'allow-act' in json.loads(capsys.readouterr().out)['error']
    assert main(['assistant', 'call', 'project_world_add', '{"town": "Town01"}', '--project', project.path,
                 '--allow-act']) == 0
    capsys.readouterr()
    from falcon_autonomy.core.project import Project
    assert Project.load(project.path).doc['worlds'][-1]['town'] == 'Town01'
    assert main(['assistant', 'call', 'runs_trash', '{"runs": ["x"]}', '--project', project.path,
                 '--allow-act']) == 2
    assert 'window' in json.loads(capsys.readouterr().out)['error']
    assert main(['assistant', 'call', 'shell', '{}']) == 2
    assert 'never' in json.loads(capsys.readouterr().out)['error']


def test_scenario_sweep_export_and_analysis(ctx):
    """A scenario run and a two-sample sweep through the job tools, read back by the analysis tools; an export job
    (the CLI's code path, queued as Data ▸ Browse queues it) and a report run to their end."""
    r = call(ctx, 'jobs_scenario_run', {'scenario': 'lead_brake', 'seed': 1})
    assert _wait(ctx, r['job'])['state'] == 'done'
    cr = call(ctx, 'analysis_criticality', {'run': r['run']})
    assert 'overall' in cr['criticality'] and cr['run'] == r['run']
    g = call(ctx, 'runs_get', {'run': r['run']})
    assert g['settings']['scenario']['id'] == 'lead_brake' and 'passed' in g['scenario_result']
    e = call(ctx, 'jobs_export', {'run': r['run'], 'format': 'coco', 'name': 'lb-coco'})
    assert e['command'].startswith('falcon export ')
    assert _wait(ctx, e['job'])['state'] == 'done'
    assert os.path.isfile(os.path.join(ctx.workspace, 'exports', 'lb-coco', 'falcon_export.json'))
    assert call(ctx, 'analysis_outputs', {'kind': 'export'})['items'][0]['id'] == 'lb-coco'
    s = call(ctx, 'jobs_sweep', {'scenario': 'lead_brake', 'samples': 2, 'name': 'sw'})
    assert _wait(ctx, s['job'], 600)['state'] == 'done', call(ctx, 'jobs_get', {'job': s['job'], 'log_lines': 40})
    st = call(ctx, 'analysis_study', {'study': 'sw'})
    assert st['runs'] == 2 and st['scenario'] == 'lead_brake' and sum(st['rss_labels'].values()) == 2
    names = [p['name'] for p in st['space']]
    fm = call(ctx, 'analysis_failure_map', {'study': 'sw', 'x': names[0], 'y': names[1], 'bins': 4})
    assert sum(c for row in fm['count'] for c in row if c) == 2
    assert 'not a searched parameter' in call(ctx, 'analysis_failure_map', {'study': 'sw', 'x': 'nope',
                                                                            'y': names[1]})['error']
    rp = call(ctx, 'jobs_replay_failure', {'study': 'sw', 'index': 1})
    assert _wait(ctx, rp['job'])['state'] == 'done'
    rows = call(ctx, 'runs_search', {'query': 'kind:scenario', 'limit': 50})['items']
    assert rp['run'] in [x['id'] for x in rows] and len(rows) >= 2
    assert call(ctx, 'analysis_report_sources', {})['sources']['study'] == ['sw']
    rep = call(ctx, 'jobs_report', {'kind': 'study', 'source': 'sw'})
    assert _wait(ctx, rep['job'])['state'] == 'done'
    assert call(ctx, 'analysis_report_sources', {})['reports'][0]['kind'] == 'study'


def test_import_a_place_place_a_pole_and_edit_the_map(ctx, project):
    """An OSM area through the project tools: imported by its job (CARLA's Python: the fake here), added to the
    project, a roadside unit at one of its junctions, a map edit checked against its OSM data, re-converted and taken
    in — the Worlds page's and map editor's flow."""
    import shutil
    from falcon_autonomy.core.project import Project
    os.makedirs(os.path.join(ctx.workspace, 'osm'))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'data', 'blr_small.osm'), os.path.join(ctx.workspace, 'osm'))
    assert 'no' in call(ctx, 'project_world_import_osm', {'source': 'missing.osm'})['error']
    r = call(ctx, 'project_world_import_osm', {'source': 'blr_small.osm'})
    assert _wait(ctx, r['job'])['state'] == 'done', call(ctx, 'jobs_get', {'job': r['job'], 'log_lines': 30})
    wid = r['world']
    assert call(ctx, 'project_world_add_imported', {'world': wid})['world'] == wid
    w = call(ctx, 'project_get', {'kind': 'world', 'id': wid})['world']
    assert w['kind'] == 'osm' and w['osm']['handedness'] == 'LHT' and w['name'] == 'Blr Small'
    assert call(ctx, 'project_rsu_place', {'world': wid, 'x': 0.0, 'y': 0.0, 'preset': 'pole-camera'})['rsu']
    assert 'check' in call(ctx, 'project_map_edit', {'world': wid, 'edit': {'op': 'maxspeed', 'way': 1,
                                                                           'kmh': 40}})['error']
    assert 'edit' in call(ctx, 'project_map_edit', {'world': wid, 'edit': {'op': 'maxspeed', 'way': 23029715,
                                                                          'kmh': 20}})
    a = call(ctx, 'jobs_apply_map_edits', {'world': wid})
    assert _wait(ctx, a['job'])['state'] == 'done', call(ctx, 'jobs_get', {'job': a['job'], 'log_lines': 30})
    assert call(ctx, 'project_world_add_imported', {'world': wid})['edit'].startswith('Apply map edits')
    nw = Project.load(project.path).item('worlds', wid)
    assert nw['osm']['applied_edits'] and nw['xodr'].startswith(f'worlds/{wid}/edit-')
    assert call(ctx, 'project_world_set', {'world': wid, 'speed_limits': 'carla', 'keep_traffic': False})['edit']


def test_roadside_units_at_junctions(ctx):
    """Junction candidates and a pole at a junction's corner of an OpenDRIVE world (a SUMO conversion with three
    junctions, tests/data/sumo_small.xodr, as tests/test_v2x.py uses)."""
    import shutil
    from falcon_autonomy.assistant.tools import _common as TC
    from falcon_autonomy.core.project import Edit
    d = os.path.join(ctx.workspace, 'worlds', 'w-sumo')
    os.makedirs(d)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'data', 'sumo_small.xodr'), os.path.join(d, 'world.xodr'))
    TC.apply_edit(ctx, Edit('insert', ['worlds'], {'id': 'w-sumo', 'name': 'SUMO sample', 'kind': 'xodr',
                                                   'xodr': 'worlds/w-sumo/world.xodr',
                                                   'weather': {'preset': 'ClearNoon'}, 'traffic_profile': 'tp-default'}))
    js = call(ctx, 'project_rsu_junctions', {'world': 'w-sumo'})
    assert js['total'] == 3 and all(j['units_here'] == 0 for j in js['items'])
    j0 = js['items'][0]['id']
    assert 'junction' in REGISTRY['project_rsu_place'].preview(ctx, world='w-sumo', junction=str(j0))
    p = call(ctx, 'project_rsu_place', {'world': 'w-sumo', 'junction': str(j0)})
    rsu = call(ctx, 'project_get', {'kind': 'rsu', 'id': p['rsu']})['rsu']
    assert str(rsu['junction']['id']) == str(j0) and rsu['world'] == 'w-sumo'
    assert call(ctx, 'project_rsu_junctions', {'world': 'w-sumo'})['items'][0]['units_here'] == 1
    assert call(ctx, 'project_rsu_set', {'rsu': p['rsu'], 'height_m': 8, 'cpm': {'rate_hz': 5}})['edit']
    assert call(ctx, 'project_get', {'kind': 'rsu', 'id': p['rsu']})['rsu']['height_m'] == 8.0
    assert 'no junction' in call(ctx, 'project_rsu_place', {'world': 'w-sumo', 'junction': 'nope'})['error']


# ------------------------------------------------------------------------------------------------ the window
@pytest.mark.gui
def test_window_tools_edit_undo_drive_and_show_me(qapp, window):
    from falcon_autonomy.assistant.privacy import redact
    c = ToolContext(project=window.project, settings=window.settings, queue=window.queue, service=window.service,
                    win=window, workspace=window.workspace(), redact=redact)
    n0 = window.undo.count()
    r = call(c, 'project_world_set', {'world': 'world-town10', 'weather_preset': 'HardRainNoon'})
    assert 'Undo' in r['undo'] and window.undo.count() == n0 + 1
    assert window.project.doc['worlds'][0]['weather'] == {'preset': 'HardRainNoon'}
    window.undo.undo()                                          # one tool call, one undo step
    assert window.project.doc['worlds'][0]['weather'] == {'preset': 'ClearNoon'}
    assert call(c, 'ui_open_page', {'page': 'data', 'tab': 'Record'}) == {'page': 'data', 'tab': 'Record'}
    got = {}                                   # a read tool runs off the GUI thread: it looks at widgets through it
    th = threading.Thread(target=lambda: got.update(call(c, 'ui_current', {})))
    th.start()
    assert pump(qapp, lambda: not th.is_alive(), 10) and got['page'] == 'data', got
    cur = call(c, 'ui_current', {})
    assert cur['page'] == 'data' and cur['tab'] == 'Record' and 'Browse' in cur['pages']['data']
    names = {x['name'] for x in call(c, 'ui_controls', {'page': 'data', 'filter': 'record'})['items']}
    assert 'record.go_btn' in names and 'record.world' in names
    h = call(c, 'ui_highlight', {'page': 'data', 'control': 'record.go_btn', 'seconds': 1})
    assert h['highlighted'] == 'record.go_btn'
    assert window.findChildren(type(window.centralWidget()), 'assistantHighlight')
    assert call(c, 'ui_highlight', {'page': 'data', 'control': 'tab:Browse', 'seconds': 1})['page'] == 'data'
    assert window.page('data').tabs.tabText(window.page('data').tabs.currentIndex()) == 'Browse'
    assert 'no control' in call(c, 'ui_highlight', {'page': 'data', 'control': 'nope'})['error']
    # a drive queued from the window, waited for off the GUI thread while the event loop runs
    r = call(c, 'jobs_drive', {'seconds': 1})
    assert 'off the window' in call(c, 'jobs_wait', {'jobs': [r['job']], 'timeout_s': 5})['error']
    out = {}
    th = threading.Thread(target=lambda: out.update(call(c, 'jobs_wait', {'jobs': [r['job']], 'timeout_s': 150})))
    th.start()
    assert pump(qapp, lambda: not th.is_alive(), 170)
    assert out['finished'] == [r['job']] and out['jobs'][0]['state'] == 'done', out
    s = call(c, 'ui_show_run', {'run': r['run'], 'step': 5})
    assert s['shown'] == r['run'] and window.current_key() == 'data'
    assert call(c, 'ui_show_frame', {'index': 3})['frame_index'] == 3
    assert call(c, 'ui_current', {})['browse_run'] == r['run']
    t = call(c, 'runs_trash', {'runs': [r['run']]})
    pump(qapp, timeout=0.3)
    assert window.page('data').browse.runs.topLevelItemCount() == 0             # Browse forgot it
    call(c, 'runs_restore', {'item': t['trashed'][0]['item']})
    pump(qapp, timeout=0.3)
    assert window.page('data').browse.runs.topLevelItemCount() == 1
    pump(qapp, timeout=1.2)                                     # the outlines remove themselves
    assert not window.findChildren(type(window.centralWidget()), 'assistantHighlight')
