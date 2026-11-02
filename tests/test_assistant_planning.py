"""The assistant's research tools (tools/research.py) on synthetic data and the fake carla: the planner benchmark
queued (previewed as its command line) and read back with intervals; an incident made from a reconstruction id with
counts only, its folder and its search's study private; refusals; the group's schema size."""
import json
import os
import shutil

import pytest

from conftest import FAKE
from test_assistant_tools import _wait
from test_incident import recon, world_project  # noqa: F401 (the synthetic reconstruction and world)

from falcon_autonomy.assistant import tools as TOOLS
from falcon_autonomy.assistant.tool_api import REGISTRY, call

RESEARCH = ('research_bench_run', 'research_bench_results', 'research_incident')


@pytest.fixture
def ctx(project):
    c = TOOLS.context(project.path, FAKE)
    yield c
    c.service.shutdown(grace_s=10.0)


def test_research_group_registered_and_small():
    from falcon_autonomy.assistant.agent import GROUP_ABOUT, guess_groups
    from falcon_autonomy.assistant.cli import schema_tokens
    from falcon_autonomy.assistant.tools._compact import DETAILS
    assert {n for n, t in REGISTRY.items() if t.group == 'research'} == set(RESEARCH)
    for n in RESEARCH:
        t = REGISTRY[n]
        assert (t.tier == 'read') == (n == 'research_bench_results') and len(DETAILS[n]['description']) > 100
        if t.tier == 'act':
            assert t.preview is not None and t.estimate is not None
    assert 'research' in GROUP_ABOUT and guess_groups('benchmark my planner by the PDM score')[0] == 'research'
    assert schema_tokens()['research'] < 3500


def test_benchmark_queued_and_read(ctx, project):
    pv = REGISTRY['research_bench_run'].preview(ctx, planners=['none'], routes='falcon-pdm-mini', name='b1')
    assert pv.startswith('falcon bench run') and '--planner none' in pv and '--set falcon-pdm-mini' in pv
    assert 'no policy' in call(ctx, 'research_bench_run', {'planners': ['policy:nope']})['error']
    assert 'unknown planner' in call(ctx, 'research_bench_run', {'planners': ['magic']})['error']
    assert 'invalid' in call(ctx, 'research_bench_run', {'routes': 'other'})['error']
    # a small benchmark made through sim.benchrun is read back with its intervals
    from falcon_autonomy.core import planbench as PB
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.benchrun import Bench
    from falcon_autonomy.sim.service import CarlaService
    svc = CarlaService(Settings(), JobQueue(SubprocessLauncher()), project.workspace, fake_carla=FAKE)
    try:
        b = Bench(svc, project, 'b1', 'falcon-pdm-mini', ['t10-stalled_vehicle-s1'])
        out = b.run(['follower', 'none'])
    finally:
        svc.shutdown()
    assert out['done'] == 2
    PB.results(b.dir, 100)
    lst = call(ctx, 'research_bench_results', {})
    assert [r['name'] for r in lst['items']] == ['b1'] and lst['items'][0]['scored']
    r = call(ctx, 'research_bench_results', {'name': 'b1'})
    assert set(r['planners']) == {'follower', 'none'} and len(r['planners']['none']['pdms']) == 3
    assert r['comparisons'][0]['b_minus_a'] == 'none − follower'
    assert 'error' in call(ctx, 'research_bench_results', {'name': '../x'})
    q = call(ctx, 'research_bench_run', {'planners': ['none'], 'routes': 'falcon-pdm-mini', 'name': 'b1'})
    assert q['benchmark'] == 'b1' and 'job' in q
    ctx.queue.cancel(ctx.queue.get(q['job']))


def test_incident_counts_only_and_private(recon, world_project):  # noqa: F811
    rec, det = recon
    world_project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])   # the search's drives: fast
    world_project.save()
    c = TOOLS.context(world_project.path, FAKE)
    try:
        dst = os.path.join(world_project.workspace, 'recon', 'dashcam-synthetic')
        shutil.copytree(rec, dst)
        pv = REGISTRY['research_incident'].preview(c, recon='dashcam-synthetic', world='w-test', search=4)
        assert 'private' in pv and '4-run' in pv
        # without a detections file: the ego only; and a 4-run search of its variations queued
        r = call(c, 'research_incident', {'recon': 'dashcam-synthetic', 'world': 'w-test', 'search': 4})
        assert 'error' not in r, r
        assert r['private'] is True and r['actors'] == 0 and r['matched_share'] == 1.0 and 'note' in r
        assert r['incident'].startswith('incident-') and 'synthetic' not in json.dumps(r)
        text = json.dumps(r)
        for word in ('"x"', 'track', 'lat', 'lon', 'route', '.json', 'recon/'):
            assert word not in text, word
        inc_dir = os.path.join(world_project.workspace, 'incidents', r['incident'])
        assert os.path.isfile(os.path.join(inc_dir, 'variations.json'))
        assert call(c, 'analysis_outputs', {'kind': 'reconstruction'}).get('private_left_out') == 1
        j = _wait(c, r['search_job'], timeout=600)
        assert j['state'] == 'done', j
        st = call(c, 'analysis_outputs', {'kind': 'study'})
        assert st['items'] == [] and st.get('private_left_out') == 1
        sid = os.listdir(os.path.join(world_project.workspace, 'studies'))[0]
        assert sid == f'{r["incident"]}-search'
        assert 'private' in call(c, 'analysis_study', {'study': sid})['error']
        # refusals
        assert 'not a reconstruction id' in call(c, 'research_incident', {'recon': '../x', 'world': 'w-test'})['error']
        assert 'no reconstruction' in call(c, 'research_incident', {'recon': 'nope', 'world': 'w-test'})['error']
        assert 'no world' in call(c, 'research_incident', {'recon': 'dashcam-synthetic', 'world': 'w-none'})['error']
        e = call(c, 'research_incident', {'recon': 'dashcam-synthetic', 'world': 'w-test', 'start_s': 50.0,
                                          'end_s': 51.0})['error']
        assert 'IncidentError' in e and 'posed' not in e
    finally:
        c.service.shutdown(grace_s=10.0)
