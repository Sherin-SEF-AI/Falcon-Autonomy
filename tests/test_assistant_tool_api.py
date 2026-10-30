"""The assistant's tool contract (falcon_autonomy.assistant.tool_api): registration rules, argument validation, what is
never allowed, and how failures reach the model."""
import pytest

from falcon_autonomy.assistant import tool_api as T


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(T, 'REGISTRY', {})
    return T.REGISTRY


def test_registration_rules(reg):
    T.register(T.Tool('runs_count', 'runs', 'read', 'how many runs', {'type': 'object', 'properties': {}},
                      lambda ctx: {'n': 3}))
    for bad, why in ((dict(tier='write'), 'tier'), (dict(group='misc'), 'group'), (dict(name='shell'), 'never'),
                     (dict(name='runs_count'), 'twice'), (dict(params={'type': 'array'}), 'object')):
        kw = dict(name='x_y', group='runs', tier='read', description='d', params={'type': 'object'},
                  fn=lambda ctx: {})
        kw.update(bad)
        with pytest.raises(ValueError, match=why):
            T.register(T.Tool(**kw))
    assert T.schemas() == [{'type': 'function', 'function': {'name': 'runs_count', 'description': 'how many runs',
                                                            'parameters': {'type': 'object', 'properties': {}}}}]


def test_calls_validate_redact_and_contain_failures(reg):
    seen = []

    @T.tool('runs_get', 'runs', 'read', 'one run', {'type': 'object', 'properties': {'run': {'type': 'string'}},
                                                     'required': ['run'], 'additionalProperties': False})
    def runs_get(ctx, run):
        if run == 'missing':
            raise T.ToolError('no run named missing')
        if run == 'boom':
            raise RuntimeError('/home/someone/secret path')
        seen.append(run)
        return {'run': run, 'path': '/home/someone/x'}

    @T.tool('ui_open', 'ui', 'act', 'open a page', gui=True)
    def ui_open(ctx):
        return {}

    ctx = T.ToolContext(redact=lambda o: {k: (v.replace('/home/someone', '~') if isinstance(v, str) else v)
                                          for k, v in o.items()})
    assert T.call(ctx, 'runs_get', {'run': 'a'}) == {'run': 'a', 'path': '~/x'}
    assert T.call(ctx, 'runs_get', {})['problems'] == ["(arguments): 'run' is a required property"]
    assert T.call(ctx, 'runs_get', {'run': 1, 'x': 2})['error'] == 'invalid arguments'
    assert T.call(ctx, 'runs_get', {'run': 'missing'}) == {'error': 'no run named missing'}
    boom = T.call(ctx, 'runs_get', {'run': 'boom'})
    assert boom == {'error': 'RuntimeError in runs_get', 'internal': True}         # no message, no path
    assert T.call(ctx, 'shell', {'cmd': 'ls'})['error'].startswith('shell is never allowed')
    assert T.call(ctx, 'nope', {}) == {'error': "no tool named 'nope'"}
    assert 'needs Falcon' in T.call(ctx, 'ui_open', {})['error']                  # no window: refused
    assert [s['function']['name'] for s in T.schemas(gui=False)] == ['runs_get']
    assert seen == ['a']


def test_redaction_and_the_private_footage(tmp_path, monkeypatch):
    import json
    import os
    from falcon_autonomy.assistant import privacy as PV
    monkeypatch.setattr(PV, 'HOME', str(tmp_path / 'home'))
    home = PV.HOME
    out = PV.redact({'path': f'{home}/FalconWorkspace/runs/a', 'log': ['key gsk_' + 'A' * 40 + ' ok',
                                                                       'Authorization: Bearer ' + 'x' * 30],
                     'n': 3, 'cfg': 'api_key=abc123 token: zz'})
    assert out['path'] == '~/FalconWorkspace/runs/a' and out['n'] == 3
    assert 'gsk_' not in out['log'][0] and '[secret]' in out['log'][0] and 'xxxx' not in out['log'][1]
    assert 'abc123' not in out['cfg'] and 'zz' not in out['cfg']
    os.makedirs(os.path.join(home, 'Videos', 'clips'))
    assert PV.private_reason(os.path.join(home, 'Videos', 'clips', 'a.mp4'))
    runs = tmp_path / 'ws' / 'runs'

    def run(name, doc, extra=None):
        d = runs / name
        d.mkdir(parents=True)
        (d / 'run.json').write_text(json.dumps(doc))
        if extra:
            (d / extra).write_text('{}')
        return str(d)
    plain = run('plain', {'worker': 'drive', 'settings': {'seed': 1}})
    route = run('route', {'worker': 'drive', 'settings': {'dashcam_route': {'route_sha256': 'x'}}})
    replayed = run('replayed', {'worker': 'replay',
                                'provenance': {'settings': {'dashcam_route': {'route_sha256': 'y'}}}})
    extract = run('extract', {'worker': 'dashcam-extract'})
    recon = run('recon', {'worker': 'recon'}, extra='dashcam.json')
    real = run('trans', {'worker': 'translate', 'real': ['dashcam:/somewhere']})
    assert PV.private_reason(plain) is None
    for d in (route, replayed, extract, recon, real):
        assert PV.private_reason(d), d
    os.makedirs(os.path.join(route, 'rgb', 'front'))
    assert PV.private_reason(os.path.join(route, 'rgb', 'front', '000001.jpg'))     # a file inside one: too
    with pytest.raises(T.ToolError, match='private'):
        PV.guard(extract)
