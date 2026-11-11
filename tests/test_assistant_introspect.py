"""The introspection tools of the formal group (assistant/tools/introspect.py): formal_causal_probe queues a bias
audit, an attention record or an intervention exactly as Models ▸ Introspection does, with a preview and an
estimate; formal_causal_result reads the outputs back as numbers; refusals for missing or non-token policies and bad
interventions; their descriptions are compact."""
import json

import pytest

from conftest import FAKE
from falcon_autonomy.assistant import tools as TOOLS
from falcon_autonomy.assistant.tool_api import REGISTRY, call
from falcon_autonomy.assistant.tools._compact import DETAILS, SHORT


@pytest.fixture
def ctx(project):
    c = TOOLS.context(project.path, FAKE)
    yield c
    c.service.shutdown(grace_s=10.0)


def _wait(c, job, timeout=600):
    r = call(c, 'jobs_wait', {'jobs': [job], 'timeout_s': timeout})
    assert r['finished'] == [job], r
    return r['jobs'][0]


def test_registered_compact_and_documented():
    from falcon_autonomy.assistant.tools import introspect as M
    for name in ('formal_causal_probe', 'formal_causal_result'):
        t = REGISTRY[name]
        assert t.group == 'formal' and t.description == SHORT[name] == M.MINE[name]
        assert 15 < len(t.description) <= 120 and t.description.count('. ') <= 1
        assert len(DETAILS[name]['description']) > 200
    assert REGISTRY['formal_causal_probe'].tier == 'act' and REGISTRY['formal_causal_probe'].preview is not None
    assert REGISTRY['formal_causal_result'].tier == 'read'
    assert 'minimum' not in json.dumps(REGISTRY['formal_causal_probe'].params)          # compacted


def test_audit_probe_queued_and_read(ctx):
    t = REGISTRY['formal_causal_probe']
    a = {'kind': 'audit', 'policy': 'none', 'arms': ['brake_light'], 'seeds': 2}
    pv = t.preview(ctx, **a)
    assert pv.startswith('falcon introspect audit') and '--arms brake_light' in pv and '--seeds 2' in pv
    assert '8 CARLA episodes' in t.estimate(ctx, **a)['what']
    r = call(ctx, 'formal_causal_probe', a)
    assert r['command'].startswith('falcon introspect audit') and 'formal_causal_result' in r['read']
    j = _wait(ctx, r['job'])
    assert j['state'] == 'done', call(ctx, 'jobs_get', {'job': r['job'], 'log_lines': 30})
    lst = call(ctx, 'formal_causal_result', {})
    oid = lst['items'][0]['id']
    assert oid.startswith('audit-') and lst['items'][0]['kind'] == 'audit' and lst['total'] == 1
    res = call(ctx, 'formal_causal_result', {'id': oid})
    assert res['kind'] == 'audit' and res['ego'] == 'none' and res['seeds'] == 2
    bl = res['arms']['brake_light']
    assert set(bl['outcomes']) == {'min_ttc_s', 'brake_onset_s', 'collision'} and bl['flagged'] is False


def test_refusals(ctx, project, tmp_path):
    assert 'no policy' in call(ctx, 'formal_causal_probe', {'kind': 'attention', 'policy': 'ghost'})['error']
    assert 'invalid arguments' in call(ctx, 'formal_causal_probe', {'kind': 'audit', 'policy': 'none',
                                                                    'seeds': 1})['error']
    assert 'invalid arguments' in call(ctx, 'formal_causal_probe', {'kind': 'probe', 'policy': 'x'})['error']
    # a policy that is not a token policy: no attention, an audit is fine
    import os
    d = os.path.join(project.workspace, 'policies', 'mlp1')
    os.makedirs(d)
    with open(os.path.join(d, 'policy.json'), 'w') as fh:
        json.dump({'schema': 'falcon.policy/1', 'net': {'kind': 'mlp'}}, fh)
    e = call(ctx, 'formal_causal_probe', {'kind': 'attention', 'policy': 'mlp1'})['error']
    assert 'token policy' in e and 'audit works on any ego' in e
    assert 'token policy' in call(ctx, 'formal_causal_probe', {'kind': 'intervene', 'policy': 'mlp1'})['error']
    pv = REGISTRY['formal_causal_probe'].preview(ctx, kind='audit', policy='mlp1')
    assert 'policies/mlp1' in pv and '--arms brake_light,velocity,autorickshaw' in pv
    assert 'no introspect output' in call(ctx, 'formal_causal_result', {'id': 'nope'})['error']


def test_intervene_needs_frames_and_valid_ops(ctx, project, tmp_path):
    import os
    from test_tokenpolicy import make_policy
    pol = make_policy(tmp_path, seed=2, name='tok')
    dst = os.path.join(project.workspace, 'policies', 'tok')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.rename(pol.path, dst)
    assert 'needs frames' in call(ctx, 'formal_causal_probe', {'kind': 'intervene', 'policy': 'tok'})['error']
    out = os.path.join(project.workspace, 'introspect', 'attention-1')
    os.makedirs(out)
    open(os.path.join(out, 'frames.npz'), 'wb').close()
    e = call(ctx, 'formal_causal_probe', {'kind': 'intervene', 'policy': 'tok', 'frames': 'attention-1',
                                          'ops': [{'op': 'mask_class', 'class': 'tram'}]})['error']
    assert 'class must be' in e
    pv = REGISTRY['formal_causal_probe'].preview(ctx, kind='intervene', policy='tok', frames='attention-1',
                                                 ops=[{'op': 'zero_head', 'layer': 1, 'head': 3}])
    assert pv.startswith('falcon introspect intervene') and 'zero_head' in pv
    pv = REGISTRY['formal_causal_probe'].preview(ctx, kind='attention', policy='tok', scenario='cut_in')
    assert '--scenario cut_in' in pv
