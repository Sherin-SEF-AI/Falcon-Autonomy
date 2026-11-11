"""The failure rate and the operating domain's coverage for the assistant: analysis_outputs kinds failure_rate and
coverage, jobs_search method estimate and scenario cell:<cell> — inside existing tools, so the schemas stay within
the token budget — with previews that are the pages' command lines, refusals, and a cell's scenario added when the job
is queued."""
import json
import os

import numpy as np
import pytest

from conftest import FAKE
from test_oddcoverage import _cfg, _run, _study

from falcon_autonomy.assistant import tools as TOOLS
from falcon_autonomy.assistant.tool_api import REGISTRY, call
from falcon_autonomy.core import failrate as F


@pytest.fixture
def ctx(project):
    c = TOOLS.context(project.path, FAKE)
    yield c
    c.service.shutdown(grace_s=10.0)


def _workspace(project):
    ws = project.workspace
    rows = [(_run(ws, f'studies/s1/runs/r{i}', _cfg(project, 'lead_brake', {'speed': 30.0, 'gap': 10 + i,
                                                                            'decel': 5.0, 'brake_at': 3.0})), 0.1)
            for i in range(3)]
    _study(ws, 's1', rows)
    d = os.path.join(ws, 'studies', 'rate-lead_brake')
    os.makedirs(d)
    est = F.summary(np.full(400, 1e-5), np.arange(400) < 100)
    with open(os.path.join(d, 'failure-rate.json'), 'w') as fh:
        json.dump({'schema': F.SCHEMA, 'scenario': 'lead_brake', 'scenario_name': 'Lead vehicle brakes',
                   'estimate': est, 'runs': {'total': 700, 'final': 400, 'lost': 0}, 'stages': [],
                   'judge': {'kind': 'objective', 'threshold': 0.5, 'about': 'the objective'},
                   'rate': F.exposure_rate(est, {'per': 'km', 'encounters': 0.1}), 'nominal': [],
                   'by_cell': {'clear/straight/car/20-40/day': {'runs': 300, 'failures': 80, 'p': 3e-6,
                                                                'status': 'ok', 'ci95': [2e-6, 4e-6]}},
                   'plain_mc': {'runs': 100, 'failures': 0}, 'mc_runs_for_same_rel_se': 4e7}, fh)
    return ws


def test_schemas_stay_within_the_budget():
    from falcon_autonomy.assistant.cli import schema_tokens
    toks = schema_tokens()
    assert toks['all'] < 18000 and toks['jobs'] < 3500 and toks['analysis'] < 3500, toks
    assert 'estimate' in json.dumps(REGISTRY['jobs_search'].params)
    assert {'failure_rate', 'coverage'} <= set(REGISTRY['analysis_outputs'].params['properties']['kind']['enum'])


def test_reading_estimates_and_coverage(ctx, project):
    _workspace(project)
    lst = call(ctx, 'analysis_outputs', {'kind': 'failure_rate'})
    assert lst['total'] == 1 and lst['items'][0]['id'] == 'rate-lead_brake'
    assert lst['items'][0]['p_per_encounter'] == 2.5e-6 and lst['items'][0]['status'] == 'ok'
    one = call(ctx, 'analysis_outputs', {'kind': 'failure_rate', 'id': 'rate-lead_brake'})
    assert one['headline'].startswith('P(fail per encounter) = 2.5e-06')
    assert one['rate'].startswith('2.5 failures per 10^7 km')
    assert one['by_cell']['items'][0]['p'] == 3e-6 and one['plain_mc_runs_for_same_rel_se'] == 4e7
    assert 'error' in call(ctx, 'analysis_outputs', {'kind': 'failure_rate', 'id': 'nope'})
    cov = call(ctx, 'analysis_outputs', {'kind': 'coverage', 'limit': 3})
    assert cov['runs'] == 3 and cov['counts']['too_few'] == 1 and cov['domain']['cells'] == 6240
    assert len(cov['next_to_fill']) == 3 and cov['next_to_fill'][0]['cell'] == 'clear/straight/car/20-40/day'
    assert cov['weather_x_light']['clear']['day'].startswith('3 runs')
    cell = call(ctx, 'analysis_outputs', {'kind': 'coverage', 'id': 'clear/straight/car/20-40/day'})
    assert cell['runs'] == 3 and cell['needed'] == 56 and cell['estimates'][0]['p'] == 3e-6
    assert cell['fill'] == 'jobs_search with scenario "cell:clear/straight/car/20-40/day"'
    assert 'error' in call(ctx, 'analysis_outputs', {'kind': 'coverage', 'id': 'rain/night'})


def test_estimate_and_cell_searches_are_queued_as_the_page_does(ctx, project):
    pv = REGISTRY['jobs_search'].preview(ctx, scenario='lead_brake', method='estimate', budget=60)
    assert pv.startswith(f'falcon failrate run {project.path} --scenario lead_brake --name rate-lead_brake ')
    assert '--stage-size 12 --final 24' in pv and '--max-stages 3' in pv
    cid = 'fog/straight/pedestrian/20-40/night'
    n0 = len(project.doc['scenarios'])
    pv = REGISTRY['jobs_search'].preview(ctx, scenario=f'cell:{cid}', method='bayes-pof', budget=8)
    assert pv.startswith('falcon search ') and '--scenario cov-pedestrian_crossing-' in pv
    assert len(project.doc['scenarios']) == n0                    # a preview changes nothing
    assert 'no scenario' in call(ctx, 'jobs_search', {'scenario': 'cell:clear/two_way/cattle/0-20/day'})['error']
    assert 'error' in call(ctx, 'jobs_search', {'scenario': 'cell:fog', 'method': 'estimate'})
    r = call(ctx, 'jobs_search', {'scenario': f'cell:{cid}', 'method': 'estimate', 'budget': 50})
    assert r['study'].startswith('rate-cov-pedestrian_crossing-') and 'failrate run' in r['command']
    from falcon_autonomy.core.project import Project
    added = [s for s in Project.load(project.path).doc['scenarios'] if s.get('coverage_cell') == cid]
    assert len(added) == 1 and added[0]['weather']['fog_density'] == 50.0
    call(ctx, 'jobs_cancel', {'job': r['job']})
    w = call(ctx, 'jobs_wait', {'jobs': [r['job']], 'timeout_s': 60})
    assert w['finished'] == [r['job']]
