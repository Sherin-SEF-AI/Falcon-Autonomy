"""The assistant's lab tools (tools/lab.py) and the results analysis_outputs reads with an id, against the fake carla:
their schemas and the compact forms' rules, previews that are the pages' command lines, the refusals (a private run or
source, paths outside the workspace, ids that do not exist, the Orin when it is not configured, a model not downloaded,
a plug-in runner), and end to end: a HIL session on the simulated device, CI mode on a suite with a baseline, a log
replayed in CARLA, a VLM evaluation with the oracle baseline, an imported world re-imported without map edits."""
import json
import os
import shutil

import pytest

from conftest import FAKE
from test_assistant_tools import _fake_run, _wait

from falcon_autonomy.assistant import tools as TOOLS
from falcon_autonomy.assistant.privacy import redact_text
from falcon_autonomy.assistant.tool_api import REGISTRY, call

LAB = ('lab_hil', 'lab_deploy', 'lab_ci', 'lab_replay', 'lab_adversary', 'lab_experiment', 'lab_vlm')


@pytest.fixture
def ctx(project):
    c = TOOLS.context(project.path, FAKE)
    yield c
    c.service.shutdown(grace_s=10.0)


def test_lab_tools_are_registered_compact_and_previewed():
    from falcon_autonomy.assistant.agent import GROUP_ABOUT, guess_groups
    from falcon_autonomy.assistant.cli import schema_tokens
    from falcon_autonomy.assistant.tools._compact import DETAILS, minify
    assert {n for n, t in REGISTRY.items() if t.group == 'lab'} == set(LAB)
    for n in LAB:
        t = REGISTRY[n]
        assert t.tier == 'act' and t.preview is not None and t.estimate is not None and not t.gui, n
        assert len(DETAILS[n]['description']) > 100
    assert 'lab' in GROUP_ABOUT and guess_groups('deploy it to the Orin and run CI')[0] == 'lab'
    toks = schema_tokens()
    assert toks['all'] < 10000 and toks['lab'] < 3500, toks
    # the compact form drops what only repeats the name, a boolean's default, or an enum's own values
    s = minify({'type': 'object', 'properties': {
        'repeat': {'type': 'string', 'enum': ['once', 'every', 'cooldown'], 'description': 'once, every or cooldown'},
        'enabled': {'type': 'boolean', 'description': 'on (default true)'},
        'name': {'type': 'string', 'description': 'a new name'},
        'kind': {'type': 'string', 'enum': ['a', 'b'], 'description': 'the kind of thing it is, for the job'},
        'lane_width': {'type': 'number', 'description': 'metres (default 3.35)'}}})['properties']
    assert [k for k, v in s.items() if 'description' in v] == ['kind', 'lane_width']
    assert s['lane_width']['description'] == 'metres'


def test_previews_are_the_pages_command_lines_and_refusals(ctx, project):
    ws = ctx.workspace
    _fake_run(ws, 'plain', {'seed': 1})
    _fake_run(ws, 'route', {'seed': 2, 'dashcam_route': {'route_sha256': 'x'}})
    pv = REGISTRY['lab_adversary'].preview(ctx, action='train', updates=7)
    assert pv.startswith(f'falcon adversary train {project.path} --out {ws}/adversary/train-') and \
        pv.endswith(f'--ego follower --action lateral --updates 7 --fake-carla {FAKE}')
    _fake_run(ws, 'held', {'seed': 3})
    assert 'test_runs' in call(ctx, 'lab_experiment', {'kind': 'active', 'name': 'al', 'runs': ['plain']})['error']
    assert 'cannot also' in call(ctx, 'lab_experiment', {'kind': 'active', 'name': 'al', 'runs': ['plain'],
                                                         'test_runs': ['plain']})['error']
    pv = REGISTRY['lab_experiment'].preview(ctx, kind='active', name='al', runs=['runs/plain'], test_runs=['held'])
    assert pv.startswith(f'falcon experiment active --workspace {ws} --id al --strategies random,entropy,kcenter,'
                         'hybrid --initial 32 --budget 32 --rounds 3 --seeds 3 --epochs 6')
    assert pv.endswith(f'--pool falcon:{ws}/runs/plain --test falcon:{ws}/runs/held')
    pv = REGISTRY['lab_vlm'].preview(ctx, runs=['runs/plain'])
    assert pv.startswith(f'falcon vlm eval --runner oracle --out {ws}/vlm/evaluations/') and \
        pv.endswith(f'--every 5 --max-frames 0 --run {ws}/runs/plain')
    assert 'HIL session on the simulated device' in REGISTRY['lab_hil'].preview(ctx, seconds=2)
    assert REGISTRY['lab_hil'].estimate(ctx, seconds=2)['seconds'] is None
    # private runs and sources, outside paths, missing ids
    for name, args in (('lab_vlm', {'runs': ['runs/route']}), ('lab_deploy', {'model': 'x', 'runs': ['route']}),
                       ('lab_experiment', {'kind': 'active', 'name': 'a', 'runs': ['runs/route']}),
                       ('lab_replay', {'action': 'compare', 'runs': ['runs/plain', 'runs/route']})):
        assert 'private' in call(ctx, name, args)['error'], name
    assert 'private' in call(ctx, 'lab_replay', {'action': 'import', 'source': 'ros2:runs/route'})['error']
    for bad in ('ros2:../x', 'ros2:/etc', 'ros2:~/x'):
        assert 'not a path inside' in call(ctx, 'lab_replay', {'action': 'import', 'source': bad})['error']
    assert 'kind must be' in call(ctx, 'lab_replay', {'action': 'import', 'source': 'dashcam:x'})['error']
    assert 'no log' in call(ctx, 'lab_replay', {'action': 'run', 'log': 'nope'})['error']
    assert 'no suite' in call(ctx, 'lab_ci', {'suite': 'nope', 'name': 'c'})['error']
    assert 'no model' in call(ctx, 'lab_deploy', {'model': 'nope', 'runs': ['runs/plain']})['error']
    ctx.project.doc['models'] += [{'id': 'det-onnx', 'name': 'det', 'kind': 'detector', 'runtime': 'onnx'},
                                  {'id': 'det-py', 'name': 'py', 'kind': 'detector', 'runtime': 'python'}]
    assert 'not an ONNX detector' in call(ctx, 'lab_deploy', {'model': 'det-py', 'runs': ['runs/plain']})['error']
    assert 'not configured' in call(ctx, 'lab_deploy', {'model': 'det-onnx', 'runs': ['runs/plain'],
                                                        'device': 'orin'})['error']
    assert 'INT8 needs' in call(ctx, 'lab_deploy', {'model': 'det-onnx', 'runs': ['runs/plain'],
                                                    'precisions': ['int8']})['error']
    pv = REGISTRY['lab_deploy'].preview(ctx, model='det-onnx', runs=['runs/plain'])
    assert pv.startswith(f'falcon deploy all {project.path} --model det-onnx --out {ws}/deploy/') and \
        pv.endswith(f'--device sim --precision fp32,fp16 --run {ws}/runs/plain')
    assert 'training id' in call(ctx, 'lab_adversary', {'action': 'eval'})['error']
    from falcon_autonomy.core.vlm_models import CANDIDATES
    assert 'listed candidates' in call(ctx, 'lab_vlm', {'runs': ['runs/plain'], 'runner': 'hf:nope'})['error']
    if CANDIDATES:
        assert 'not downloaded' in call(ctx, 'lab_vlm', {'runs': ['runs/plain'],
                                                         'runner': 'hf:' + next(iter(CANDIDATES))})['error']
    assert 'runner is' in call(ctx, 'lab_vlm', {'runs': ['runs/plain'], 'runner': 'plugin:m:C'})['error']
    assert 'real_train' in call(ctx, 'lab_experiment', {'kind': 'mixed', 'name': 'm', 'runs': ['plain']})['error']
    assert 'never reaches' in call(ctx, 'lab_experiment', {'kind': 'mixed', 'name': 'm', 'runs': ['plain'],
                                                           'real_train': 'dashcam:x', 'real_test': 'kitti:y'})['error']
    assert 'analysis_study' in call(ctx, 'analysis_outputs', {'kind': 'study', 'id': 'x'})['error']
    assert 'no ci' in call(ctx, 'analysis_outputs', {'kind': 'ci', 'id': 'x'})['error']
    assert not os.path.exists(os.path.join(ws, 'deploy')) and not os.path.exists(os.path.join(ws, 'ci-configs'))


def test_hil_session_on_the_simulated_device(ctx):
    r = call(ctx, 'lab_hil', {'seconds': 2})
    assert r['run'].startswith('runs/') and 'hil' in r['run']
    j = _wait(ctx, r['job'], 240)
    assert j['state'] == 'done', call(ctx, 'jobs_get', {'job': r['job'], 'log_lines': 40})
    lst = call(ctx, 'analysis_outputs', {'kind': 'hil'})
    assert lst['items'][0]['id'] == os.path.basename(r['run'])
    h = call(ctx, 'analysis_outputs', {'kind': 'hil', 'id': r['run']})
    assert h['status'] == 'ok' and h['device']['mode'] == 'local' and h['frames'] > 0
    assert h['measures_ms']['e2e']['median'] is not None and 'on_time' in h['deadline']


def test_ci_mode_on_a_suite_with_a_baseline(ctx):
    s = call(ctx, 'jobs_stack_suite', {'name': 's1', 'scenarios': ['lead_brake'], 'seeds': [0]})
    assert _wait(ctx, s['job'], 300)['state'] == 'done'
    pv = REGISTRY['lab_ci'].preview(ctx, suite='s1', name='c1')
    ws = ctx.workspace
    assert pv == f'falcon ci run {ws}/ci-configs/c1.json --out {ws}/ci/c1 --quiet --fake-carla {FAKE}'
    c = call(ctx, 'lab_ci', {'suite': 's1', 'name': 'c1', 'min_pass_rate': 0.0, 'rule': 'estimate'})
    assert c['ci'] == 'c1' and c['command'] == redact_text(pv)          # results are redacted (~ for the home)
    cfg = json.load(open(os.path.join(ws, 'ci-configs', 'c1.json')))
    assert cfg['subject']['suite'] == os.path.join(ws, 'suites', 's1', 'suite-request.json')
    assert cfg['carla'] == {'mode': 'fake'} and cfg['gates'] == {'pass_rate': {'min': 0.0, 'rule': 'estimate'}}
    assert _wait(ctx, c['job'], 300)['state'] == 'done', call(ctx, 'jobs_get', {'job': c['job'], 'log_lines': 40})
    assert call(ctx, 'analysis_outputs', {'kind': 'ci'})['items'][0]['id'] == 'c1'
    res = call(ctx, 'analysis_outputs', {'kind': 'ci', 'id': 'c1'})
    assert res['verdict'] == 'passed' and res['runs'] == 1 and any(g['status'] == 'passed' for g in res['gates'])
    assert 'exists already' in call(ctx, 'lab_ci', {'suite': 's1', 'name': 'c1'})['error']
    c2 = call(ctx, 'lab_ci', {'suite': 's1', 'name': 'c2', 'min_pass_rate': 0.0, 'rule': 'estimate',
                              'baseline': 'c1'})
    _wait(ctx, c2['job'], 300)
    res2 = call(ctx, 'analysis_outputs', {'kind': 'ci', 'id': 'c2'})
    assert res2['comparison']['pairs'] == 1 and res2['comparison']['regressions'] == []


def test_a_log_listed_read_and_replayed(ctx):
    from test_logreplay import straight_log
    straight_log().save(os.path.join(ctx.workspace, 'logs', 'straight'))
    assert call(ctx, 'analysis_outputs', {'kind': 'log'})['items'][0]['id'] == 'straight'
    lg = call(ctx, 'analysis_outputs', {'kind': 'log', 'id': 'straight'})
    assert lg['actors'] == 2 and lg['duration_s'] == 10.0 and lg['map']['name'] == 'Town10HD'
    pv = REGISTRY['lab_replay'].preview(ctx, action='run', log='straight', driver='follower')
    assert pv.startswith('replay straight in ') and 'driver follower' in pv
    assert REGISTRY['lab_replay'].estimate(ctx, action='run', log='straight')['seconds'] > 0
    r = call(ctx, 'lab_replay', {'action': 'run', 'log': 'straight', 'driver': 'follower'})
    j = _wait(ctx, r['job'], 300)
    assert j['state'] == 'done', call(ctx, 'jobs_get', {'job': r['job'], 'log_lines': 40})
    assert os.path.isfile(os.path.join(ctx.workspace, r['run'], 'run.json'))
    pv = REGISTRY['lab_replay'].preview(ctx, action='compare', runs=[r['run'], r['run']])
    assert pv.endswith(f'--out {ctx.workspace}/{r["run"]}-comparison')


def test_vlm_oracle_evaluation_scores_as_text(ctx):
    d = call(ctx, 'jobs_drive', {'seconds': 1.5, 'seed': 3})
    assert _wait(ctx, d['job'])['state'] == 'done'
    v = call(ctx, 'lab_vlm', {'runs': [d['run']], 'every': 5})
    assert _wait(ctx, v['job'], 300)['state'] == 'done', call(ctx, 'jobs_get', {'job': v['job'], 'log_lines': 40})
    assert call(ctx, 'analysis_outputs', {'kind': 'vlm'})['items'][0]['id'] == v['evaluation']
    sc = call(ctx, 'analysis_outputs', {'kind': 'vlm', 'id': v['evaluation']})
    text = json.dumps(sc)
    assert 'rows' not in sc and 'base64' not in text and '.png' not in text and len(text) < 16000


def test_an_imported_world_without_edits_is_re_imported(ctx):
    os.makedirs(os.path.join(ctx.workspace, 'osm'))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'data', 'blr_small.osm'), os.path.join(ctx.workspace, 'osm'))
    r = call(ctx, 'project_world_import_osm', {'source': 'blr_small.osm'})
    assert _wait(ctx, r['job'])['state'] == 'done'
    wid = r['world']
    call(ctx, 'project_world_add_imported', {'world': wid})
    assert REGISTRY['jobs_apply_map_edits'].preview(ctx, world=wid).startswith('re-import Blr Small with the '
                                                                               'current converter')
    a = call(ctx, 'jobs_apply_map_edits', {'world': wid})
    assert a['reimport'] and a['world'] != wid and a['title'].startswith('Re-import: ')
    assert _wait(ctx, a['job'])['state'] == 'done', call(ctx, 'jobs_get', {'job': a['job'], 'log_lines': 30})
    assert call(ctx, 'project_world_add_imported', {'world': a['world']})['world'] == a['world']
    assert {wid, a['world']} <= {w['id'] for w in ctx.project.doc['worlds']}
    town = ctx.project.doc['worlds'][0]['id']
    assert 'OpenStreetMap' in call(ctx, 'jobs_apply_map_edits', {'world': town})['error']
