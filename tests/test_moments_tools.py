"""The assistant's moment and campaign tools (assistant/tools/moments.py): moments searched locally with private
runs only counted, the exposure table, and a campaign that exists only after its one approval card. No model, no
network, no CARLA; the runner job is held, never started."""
import json
import os

import pytest

from falcon_autonomy.assistant import tool_api as T
from falcon_autonomy.assistant import tools as TOOLS  # noqa: F401 (registers the tools)
from falcon_autonomy.assistant.privacy import redact
from falcon_autonomy.core import autoresearch as R
from falcon_autonomy.core import campaign as K
from falcon_autonomy.core import moments as M
from test_campaign import GOAL, MODE, suggestion
from test_moments import NIGHT_RAIN, actor, lin, write_run
from test_research_autopilot import InlineLauncher, make_queue


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    ws = str(tmp_path / 'ws')
    os.makedirs(ws)
    write_run(ws, 'cutin-night', 10.0, [actor('motorcycle', lambda t: 14 + 9 * t, lin(-3.5, 0.0, 2.0, 4.0),
                                              size=(2.2, 0.8))], weather=NIGHT_RAIN)
    write_run(ws, 'cutin-day', 10.0, [actor('car', lambda t: 22 + 9 * t, lin(3.5, 0.0, 2.0, 4.0))])
    write_run(ws, 'route', 10.0, [actor('motorcycle', lambda t: 14 + 9 * t, lin(-3.5, 0.0, 2.0, 4.0),
                                        size=(2.2, 0.8))], weather=NIGHT_RAIN, settings={'dashcam_route': {'id': 'r'}})
    monkeypatch.setattr(R, 'suggestions', lambda workspace, project=None, limit=8: [MODE, suggestion()])
    q = make_queue(tmp_path, InlineLauncher(hold=True))
    c = T.ToolContext(queue=q, workspace=ws, redact=redact)
    c.ws = os.path.realpath(ws)
    return c


def test_the_tools_are_registered_in_their_groups_and_tiers():
    want = {'runs_moments_search': ('runs', 'read'), 'runs_moments_stats': ('runs', 'read'),
            'research_campaign_new': ('research', 'act'), 'research_campaign_status': ('research', 'read'),
            'research_campaign_brief': ('research', 'read')}
    for name, (group, tier) in want.items():
        t = T.REGISTRY[name]
        assert (t.group, t.tier) == (group, tier) and len(t.description) <= 110, name
    t = T.REGISTRY['research_campaign_new']
    assert t.approval is not None and t.preview is not None


def test_moment_search_is_local_and_private_runs_are_only_counted(ctx):
    r = T.call(ctx, 'runs_moments_search', {'query': 'two-wheeler cuts in from the left at night in rain'})
    assert r['items'] == [] and 'no moment index' in r['note'] and r['understood'].startswith('type cut_in · actor')
    M.index(ctx.ws, xodr='')
    r = T.call(ctx, 'runs_moments_search', {'query': 'two-wheeler cuts in from the left at night in rain, zebra'})
    assert r['understood'] == 'type cut_in · actor motorcycle or bicycle · side left · light night · weather rain'
    assert r['not_understood'] == ['zebra'] and r['full_matches'] == 1 and r['private_left_out'] == 1
    assert [(os.path.basename(x['run']), x.get('match')) for x in r['items']] == [('cutin-night', None),
                                                                                 ('cutin-day', 0.0)]
    top = r['items'][0]
    assert top['type'] == 'cut_in' and top['actors'] == ['motorcycle'] and top['side'] == 'left' and top['tod'] == 'night'
    assert r['items'][1]['missed'] == ['actors', 'side', 'tod', 'weather']
    assert 'route' not in json.dumps(r)                              # the dashcam-derived run: a count, nothing else
    r = T.call(ctx, 'runs_moments_search', {'types': ['cut_in'], 'actors': ['two_wheeler'], 'strict': True,
                                            'where': ['gap<25']})
    assert r['total'] == 1 and r['private_left_out'] == 1
    assert 'unknown actor class' in T.call(ctx, 'runs_moments_search', {'actors': ['dragon']})['error']
    assert 'FIELD<NUMBER' in T.call(ctx, 'runs_moments_search', {'where': ['ttc is low']})['error']
    assert T.call(ctx, 'runs_moments_search', {'side': 'up'})['error'] == 'invalid arguments'


def test_moment_stats_leave_private_sources_out(ctx):
    M.index(ctx.ws, xodr='')
    r = T.call(ctx, 'runs_moments_stats', {})
    assert r['runs'] == 2 and r['private_left_out'] == 1 and 'not real traffic' in r['note']
    assert [(x['type'], x['count']) for x in r['items']] == [('cut_in', 2)]
    assert r['items'][0]['per_hour'] == pytest.approx(2 / (16 / 3600), rel=1e-3)
    by = T.call(ctx, 'runs_moments_stats', {'by': 'tod'})
    assert {x['tod'] for x in by['items']} == {'day', 'night'}


def test_a_campaign_exists_only_after_its_one_approval_card(ctx):
    t = T.REGISTRY['research_campaign_new']
    args = {'goal': GOAL, 'gpu_hours': 1.0, 'until': '+6h', 'max_jobs': 5, 'name': 'night'}
    g = t.approval(ctx, **args)
    assert g['ask'] is True and g['allow_all'] is False and 'one approval' in g['why']
    card = t.preview(ctx, **args)
    for part in ('Campaign night: Find why night pedestrian recall dropped', '1 GPU-hours', 'at most 5 jobs',
                 'clusters, moments, rootcause, ablation', '[ablation] Ablation: rain at night', 'Never done unattended'):
        assert part in card, part
    assert not os.path.exists(os.path.join(ctx.ws, 'campaigns'))       # the card wrote nothing
    for ap in ({}, {'how': 'auto'}):                                    # a permission mode is not the user's approval
        ctx.approval = ap
        r = T.call(ctx, 'research_campaign_new', args)
        assert 'approves its card' in r['error'] and ctx.queue.jobs == []
    assert not os.path.exists(os.path.join(ctx.ws, 'campaigns'))
    assert 'no campaign in this workspace' in T.call(ctx, 'research_campaign_status', {})['error']
    ctx.approval = {'how': 'approved', 'options': {}}
    r = T.call(ctx, 'research_campaign_new', args)
    assert r['campaign'] == 'night' and r['state'] == 'approved' and [p['kind'] for p in r['plan']] == [
        'clusters', 'moments', 'ablation']
    rec = K.load(ctx.ws, 'night')
    assert rec['approval']['how'] == 'approved' and rec['scope_sha256'] == K.scope_hash(rec)
    job = ctx.queue.jobs[0]
    assert r['job'] == job.id and job.spec.kind == 'campaign'
    assert job.spec.argv[-5:] == ['falcon_autonomy', 'campaign', 'run', ctx.ws, 'night']
    assert 'exists already' in T.call(ctx, 'research_campaign_new', args)['error'] and len(ctx.queue.jobs) == 1
    st = T.call(ctx, 'research_campaign_status', {})
    assert st['id'] == 'night' and st['state'] == 'approved' and st['left']['gpu_hours'] == 1.0
    assert [s['state'] for s in st['studies']] == ['planned'] * 3 and st['log'][-1]['decision'] == 'approved'
    b = T.call(ctx, 'research_campaign_brief', {'campaign': 'night'})
    assert b['final'] is False and '# Morning brief' in b['markdown'] and '**Confidence: none**' in b['markdown']
    # kinds never allowed, a goal naming the footage folder
    assert 'cloud spend' in T.call(ctx, 'research_campaign_new', dict(args, name='c2', kinds=['cloud']))['error']
    r = T.call(ctx, 'research_campaign_new', dict(args, name='c3', goal='Find what is in ~/Videos/dashcam/clip.mp4'))
    assert 'private location' in r['error'] and len(ctx.queue.jobs) == 1
    ctx.queue.cancel(job)
