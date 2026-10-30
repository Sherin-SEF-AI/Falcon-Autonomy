"""The assistant's suggestions, "Explain this error", "Ask about this" and show-me mode, without Qt (a fake Groq for the
model's part): every trigger from a fixture workspace, each candidate's tool call valid against the tool's full schema,
the offline wording, the model's wording kept only when its numbers are the facts', dismissals kept per project, the
pacing, private data never suggested or asked about, the explanation offline and from the fast model with its action,
and show-me's ui calls run without a card and with a pause."""
import json
import os
import time

import pytest

from fake_groq import FAKE_KEY, FakeGroq, text, tools
from falcon_autonomy.assistant import explain as X
from falcon_autonomy.assistant import suggest as S
from falcon_autonomy.assistant import tool_api as T


def _w(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        if isinstance(doc, list):
            fh.write(''.join(json.dumps(r) + '\n' for r in doc))
        else:
            json.dump(doc, fh)


_N = [0]


def _run(ws, name, world='world-town10', rig='rig-default', weather=None, steps=200, extra=None, **st):
    d = os.path.join(ws, 'runs', name)
    settings = {'project': {'world': world, 'rig': rig}, 'fps': 20,
                'weather': weather or {'sun_altitude_angle': 45.0, 'cloudiness': 0.0, 'precipitation': 0.0},
                'weather_source': {'preset': 'ClearNoon'}, **st}
    _N[0] += 1
    _w(os.path.join(d, 'run.json'), dict({'name': name, 'status': 'ok', 'worker': 'drive',
                                          'started': f'2026-10-30T10:{_N[0] % 60:02d}:00Z',
                                          'provenance': {'settings': settings},
                                          'performance': {'steps': steps, 'sim_s': steps / 20}}, **(extra or {})))
    return d


@pytest.fixture
def doc():
    from falcon_autonomy.core.project import Project
    return Project.new('Suggest').doc


@pytest.fixture
def ws(tmp_path):
    p = tmp_path / 'ws'
    p.mkdir()
    return str(p)


def _state(ws, doc, **kw):
    return S.State(workspace=ws, doc=doc, now=time.time(), **kw)


def _one(cands, trigger):
    got = [s for s in cands if s.trigger == trigger]
    assert len(got) == 1, [s.key for s in cands]
    return got[0]


def _valid(s):
    """The suggestion's call is a registered tool with arguments its FULL schema accepts."""
    import jsonschema
    from falcon_autonomy.assistant import tools  # noqa: F401 (registers them)
    from falcon_autonomy.assistant.tools._compact import DETAILS
    t = T.REGISTRY[s.tool]
    assert not T.check_args(t, s.args), (s.tool, s.args)
    jsonschema.validate(s.args, DETAILS[s.tool]['params'])
    assert s.tool not in T.NEVER


# ------------------------------------------------------------------------------------------------ the triggers
def test_server_crash_failed_and_finished_jobs_and_an_idle_gpu(ws, doc):
    now = time.time()
    jobs = [{'id': 4, 'kind': 'drive', 'title': 'Drive Town 10', 'state': 'failed', 'exit_code': 5,
             'exit': 'No CARLA server answered at the start.', 'error': 'connection refused', 'needs_carla': True,
             'needs_gpu': False, 'wait_reason': '', 'run_dir': '', 'finished': now - 30, 'elapsed_s': 3.0,
             'page': 'data'},
            {'id': 5, 'kind': 'export', 'title': 'Export r as coco', 'state': 'done', 'exit_code': 0, 'exit': '',
             'error': '', 'needs_carla': False, 'needs_gpu': False, 'wait_reason': '', 'run_dir': '',
             'finished': now - 10, 'elapsed_s': 12.4, 'page': 'data'},
            {'id': 6, 'kind': 'drive', 'title': 'Drive 2', 'state': 'waiting', 'needs_carla': True,
             'wait_reason': 'the CARLA lock is held by process 4242', 'finished': None}]
    st = _state(ws, doc, server={'state': 'crashed', 'error': 'signal 11\nmore'}, jobs=jobs, gpu_util=3.0)
    cands = S.gather(st)
    crash = _one(cands, 'server_crashed')
    assert crash.tool == 'system_server' and 'signal 11' in crash.why and crash is cands[0]   # the most urgent first
    f = _one(cands, 'job_failed')
    assert f.args == {'job': 4, 'log_lines': 30} and 'connection refused' in f.why and 'data' in f.pages
    d = _one(cands, 'job_done')
    assert d.tool == 'ui_open_job' and d.args == {'job': 5} and '12 s' in d.why
    g = _one(cands, 'gpu_idle')
    assert g.tool == 'system_queue' and '1 CARLA job(s) wait while the GPU is at 3 %' in g.why and '4242' in g.why
    for s in (crash, f, d, g):
        _valid(s)
        assert s.shown() == s.why and not s.phrased                  # offline: the deterministic wording
    busy = _state(ws, doc, jobs=jobs[2:], gpu_util=55.0)
    assert not [s for s in S.gather(busy) if s.trigger == 'gpu_idle']
    old = [dict(jobs[1], finished=now - 3600)]
    assert not [s for s in S.gather(_state(ws, doc, jobs=old)) if s.trigger == 'job_done']


def test_avoidable_failures_replay_the_worst(ws, doc):
    d = os.path.join(ws, 'studies', 'search-lead_brake')
    _w(os.path.join(d, 'study.json'), {'scenario': 'lead_brake', 'world': 'world-town10'})
    _w(os.path.join(d, 'search-bayes.json'), {'threshold': 0.5, 'method': 'bayes'})
    rows = [{'values': {'gap_m': 10}, 'objective': 0.7, 'criticality': 0.7, 'rss': {'label': 'avoidable'}},
            {'values': {'gap_m': 4}, 'objective': 0.912, 'criticality': 0.912, 'rss': {'label': 'avoidable'}},
            {'values': {'gap_m': 3}, 'objective': 0.0, 'criticality': 0.95, 'rss': {'label': 'other'}},
            {'values': {'gap_m': 30}, 'objective': 0.3, 'criticality': 0.3, 'rss': {'label': 'avoidable'}}]
    _w(os.path.join(d, 'results.jsonl'), rows)
    s = _one(S.gather(_state(ws, doc)), 'avoidable')
    assert s.args == {'study': 'search-lead_brake', 'index': 1} and s.tool == 'jobs_replay_failure'
    assert '2 of 4 runs of study search-lead_brake (lead_brake)' in s.why and 'criticality 0.912' in s.why
    assert set(s.pages) == {'rarecases', 'home'}
    _valid(s)


def test_failed_mrm_checks_ask_why_on_the_worst_run(ws, doc):
    r1, r2 = _run(ws, 'scenario-lead-1'), _run(ws, 'scenario-lead-2')
    base = {'status': 'ok', 'scenario': 'lead_brake'}
    _w(os.path.join(ws, 'suites', 'nightly', 'results.jsonl'), [
        dict(base, key='a', seed=0, run=r1, criticality=0.4,
             mrm={'passed': False, 'checks': [{'name': 'stopped_in_time', 'ok': False}, {'name': 'lane', 'ok': True}]}),
        dict(base, key='b', seed=1, run=r2, criticality=0.8,
             mrm={'passed': False, 'checks': [{'name': 'stopped_in_time', 'ok': False}]}),
        dict(base, key='c', seed=2, run=r2, criticality=0.1, mrm={'passed': True, 'checks': []})])
    s = _one(S.gather(_state(ws, doc)), 'mrm_failed')
    assert s.tool == 'jobs_why' and s.args == {'run': 'runs/scenario-lead-2'}
    assert '2 of 3 runs of suite nightly failed the minimal-risk-manoeuvre check' in s.why
    assert 'stopped_in_time, 2 times' in s.why and 'seed 1' in s.why
    _valid(s)


def test_worlds_whose_lights_control_no_lane_or_an_old_left_hand_import(ws, doc):
    doc['worlds'] += [{'id': 'w-blr', 'name': 'Bengaluru', 'kind': 'osm', 'xodr': 'worlds/w-blr/map.xodr'},
                      {'id': 'w-old', 'name': 'Old Bengaluru', 'kind': 'osm', 'xodr': 'worlds/w-old/map.xodr'},
                      {'id': 'w-ok', 'name': 'Fine', 'kind': 'osm', 'xodr': 'worlds/w-ok/map.xodr'}]
    _w(os.path.join(ws, 'worlds', 'w-blr', 'import.json'), {'handedness': 'LHT', 'source': 'a.osm',
                                                            'signal_lanes': {'oriented': 23, 'no_lane': 23}})
    _w(os.path.join(ws, 'worlds', 'w-old', 'import.json'), {'handedness': 'LHT', 'source': 'a.osm'})
    _w(os.path.join(ws, 'worlds', 'w-ok', 'import.json'), {'handedness': 'LHT', 'source': 'a.osm',
                                                           'signal_lanes': {'oriented': 7, 'no_lane': 0}})
    got = {s.args['world']: s for s in S.gather(_state(ws, doc)) if s.trigger == 'world_lights'}
    assert set(got) == {'w-blr', 'w-old'}
    assert '23 of 23 traffic-light records control no lane' in got['w-blr'].why
    assert 'converted by an earlier version' in got['w-old'].why and got['w-old'].tool == 'jobs_apply_map_edits'
    _valid(got['w-blr'])


def test_coverage_gaps_from_the_run_index_and_a_first_drive(ws, doc):
    first = _one(S.gather(_state(ws, doc)), 'first_drive')
    assert first.args == {'world': 'world-town10', 'rig': 'rig-default', 'seconds': 10.0}
    _valid(first)
    _run(ws, 'drive-1')
    _run(ws, 'drive-route', dashcam_route={'clip': 'x'})          # private: left out of the counts
    cands = S.gather(_state(ws, doc))
    assert not [s for s in cands if s.trigger == 'first_drive']
    s = _one(cands, 'coverage_gap')
    assert s.args == {'world': 'world-town10', 'weather_preset': 'CloudyNoon', 'seconds': 30.0}
    assert 'Town 10 has no frames at day, cloudy: 11 of 12 condition cells are empty over 200 frames in 1 drives' \
        in s.why and 'drive-route' not in json.dumps(s.to_dict())
    _valid(s)


def test_near_duplicates_drift_refusals_and_a_missing_evaluation(ws, doc):
    r = _run(ws, 'drive-stopped')
    health = {'duplicates': {'hashed': 200, 'near_duplicates': 88, 'share': 0.44, 'bits': 4,
                             'examples': [{'run': r, 'step': 3, 'like': {}, 'bits': 1}]}}
    tr = _run(ws, 'drive-stopped-tr-cut1', extra={'translation': {'drift': {'frames': 120, 'threshold': 0.35,
                                                                            'flagged': 0, 'refused': 119}}})
    doc['models'] = [{'id': 'm-synth', 'name': 'Synthetic detector'}]
    cands = S.gather(_state(ws, doc, health=health))
    dup = _one(cands, 'duplicates')
    assert dup.args == {'runs': ['runs/drive-stopped'], 'tags': ['near-duplicates']}
    assert '88 of 200 sampled frames (44 %) are near-duplicates' in dup.why
    dr = _one(cands, 'drift_refused')
    assert dr.args == {'run': os.path.relpath(tr, ws)} and '119 of 120 frames' in dr.why and '0.35' in dr.why
    ev = _one(cands, 'no_evaluation')
    assert ev.args['model'] == 'm-synth' and set(ev.args['runs']) <= {'runs/drive-stopped',
                                                                      'runs/drive-stopped-tr-cut1'}
    for s in (dup, dr, ev):
        _valid(s)


def test_no_scenarios_or_rules_and_their_calls_work(project, doc, ws):
    from conftest import FAKE
    from falcon_autonomy.assistant import tools as TOOLS
    cands = S.gather(_state(ws, doc))
    scn, rule = _one(cands, 'no_scenarios'), _one(cands, 'no_rules')
    _valid(scn)
    _valid(rule)
    ctx = TOOLS.context(project.path, FAKE)
    try:
        r1 = T.call(ctx, scn.tool, scn.args)
        r2 = T.call(ctx, rule.tool, rule.args)
    finally:
        ctx.service.shutdown(grace_s=5.0)
    assert 'error' not in r1 and 'error' not in r2, (r1, r2)          # the example rule parses
    assert S.gather(_state(ws, ctx.project.doc), (S.t_no_scenarios, S.t_no_rules)) == []


def test_private_studies_and_worlds_are_never_suggested(ws, doc):
    d = os.path.join(ws, 'studies', 'search-x')
    _w(os.path.join(d, 'study.json'), {'scenario': 'lead_brake'})
    priv = _run(ws, 'route-run', dashcam_route={'clip': 'y'})
    _w(os.path.join(d, 'results.jsonl'), [{'objective': 0.9, 'criticality': 0.9, 'rss': {'label': 'avoidable'},
                                           'run': priv}])
    doc['worlds'].append({'id': 'w-dc', 'name': 'From the dashcam', 'kind': 'osm', 'xodr': 'worlds/w-dc/m.xodr'})
    _w(os.path.join(ws, 'worlds', 'w-dc', 'import.json'), {'handedness': 'LHT', 'source': 'dashcam:route'})
    cands = S.gather(_state(ws, doc))
    assert not [s for s in cands if s.trigger in ('avoidable', 'world_lights')], [s.key for s in cands]
    assert 'route-run' not in json.dumps([s.to_dict() for s in cands])


# ------------------------------------------------------------------------------------------------ model wording
@pytest.fixture
def srv():
    s = FakeGroq().start()
    yield s
    s.stop()


@pytest.fixture
def svc(tmp_path, srv):
    from falcon_autonomy.assistant.keystore import KeyStore
    from falcon_autonomy.assistant.service import AssistantService
    from falcon_autonomy.core.settings import Secrets, Settings
    s = AssistantService(Settings(str(tmp_path / 'settings.json')),
                         KeyStore(Secrets(str(tmp_path / 'secrets.json')), use_keyring=False))
    s.set_cfg(enabled=True, base_url=srv.base_url)
    s.accept_notice()
    s.keys.set(FAKE_KEY)
    return s


def _quick(content, model='qwen/qwen3.8-27b'):
    return {'json': {'model': model, 'choices': [{'message': {'role': 'assistant', 'content': content},
                                                  'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': 90, 'completion_tokens': 40, 'total_tokens': 130}}}


def test_model_wording_ranked_and_grounded_else_the_deterministic_one(ws, doc, srv, svc):
    st = _state(ws, doc, server={'state': 'crashed', 'error': ''},
                jobs=[{'id': 4, 'kind': 'drive', 'title': 'Drive', 'state': 'failed', 'exit': 'Finished.',
                       'error': 'boom', 'finished': time.time(), 'page': 'data'}])
    cands = S.gather(st)
    crash, failed = _one(cands, 'server_crashed'), _one(cands, 'job_failed')
    srv.push(_quick('```json\n' + json.dumps({'suggestions': [
        {'key': failed.key, 'text': 'Job 4 failed with "boom": read its last 30 log lines.'},
        {'key': crash.key, 'text': 'The server crashed 3 times today — check its log.'},     # 3: not in its facts
        {'key': 'made-up', 'text': 'x'}]}) + '\n```'))
    res = S.phrase(svc, cands, {'page': 'home'})
    assert res['ranked'] == [failed.key, crash.key] and set(res['texts']) == {failed.key}
    assert crash.key in res['rejected']
    S.apply_phrasing(cands, res)
    assert failed.phrased and failed.shown().startswith('Job 4 failed') and failed.rank == 0
    assert not crash.phrased and crash.shown() == crash.why                      # the grounding check failed
    assert S.for_page(cands, 'home')[0] is failed                                # the model's ranking first
    sent = srv.chats()[-1]['body']
    assert sent['model'] == 'qwen/qwen3.8-27b' and sent.get('stream') in (False, None)
    assert FAKE_KEY not in json.dumps(sent)
    srv.push(_quick('not json at all'))
    assert S.phrase(svc, cands) == {'ranked': [], 'texts': {}, 'rejected': {}}


def test_dismissals_are_kept_per_project_and_the_pacing(ws, doc, tmp_path):
    cands = S.gather(_state(ws, doc))
    top = S.for_page(cands, 'home')
    assert len(top) <= S.MAX_PER_PAGE and top
    store = S.Store(ws)
    store.dismiss(top[0].key)
    again = S.Store(ws)                                    # a restart: still dismissed
    assert again.is_dismissed(top[0].key) and top[0] not in S.for_page(cands, 'home', again.dismissed)
    other = tmp_path / 'other-ws'
    other.mkdir()
    assert not S.Store(str(other)).dismissed                # another project: its own
    pc = S.Pacer()
    assert pc.gather_due(100.0)
    pc.gathered(100.0)
    assert not pc.gather_due(110.0) and not pc.gather_due(101.0, forced=True) and pc.gather_due(104.0, forced=True)
    assert pc.gather_due(130.0)
    assert pc.phrase_due(cands, 100.0)
    pc.phrased(cands, {'ranked': [cands[0].key], 'texts': {cands[0].key: 'worded'}}, 100.0)
    assert not pc.phrase_due(cands, 500.0)                  # nothing new: not asked again
    fresh = S.gather(_state(ws, doc))
    pc.recall(fresh)
    assert fresh[0].shown() == 'worded' and fresh[0].rank == 0
    fresh[1].why += ' (changed)'
    assert not pc.phrase_due(fresh, 150.0) and pc.phrase_due(fresh, 221.0)   # a changed why, after the interval


# ------------------------------------------------------------------------------------------------ explain, ask
class _Job:
    """A failed core.jobs.Job, as the queue keeps it."""

    def __init__(self, run_dir='', log=None, code=5):
        from falcon_autonomy.core.jobs import Job, JobSpec
        j = Job(JobSpec('derive', 'Synthetic audio of drive-1', ['x'], run_dir=run_dir))
        j.state, j.exit_code = 'failed', code
        j.log = log or [('info', 'starting'), ('error', f'{os.path.expanduser("~")}/ws/runs/x: the rig has no '
                                                         'microphone array (audio_array)')]
        self.job = j


def test_explain_offline_and_from_the_fast_model(ws, doc, srv, svc):
    j = _Job().job
    ctx = X.job_context(j, ws)
    assert ctx['job'] == j.id and ctx['exit'].startswith('No CARLA server') and '~/ws/runs/x' in ctx['error']
    assert os.path.expanduser('~') + '/' not in json.dumps(ctx)              # redacted
    fixes = X.known_fixes(ctx, doc)
    keys = [f['key'] for f in fixes]
    assert keys[:2] == ['server', 'audio_array']
    audio = fixes[1]
    assert audio['tool'] == 'project_sensor_add' and audio['args'] == {'type': 'audio_array', 'rig': 'rig-default'}
    off = X.offline(ctx, fixes, [])
    assert off['text'].startswith('Cause: No CARLA server answered') and off['cta'] is fixes[0] and not off['model']
    srv.push(_quick('Cause: the rig has no microphone array, so the audio cannot be derived.\nFix: add an '
                    'audio_array sensor to rig-default and run it again in 45 minutes.\nAction: audio_array'))
    res = X.explain(svc, ctx, fixes, [{'link': 'docs/audio.md', 'heading': 'Audio', 'text': 'the array'}])
    assert res['cta'] is audio and 'Action:' not in res['text'] and res['model'] == 'qwen/qwen3.8-27b'
    assert res['ungrounded'] == ['45']                                       # "45 minutes": not in the log
    body = json.dumps(srv.chats()[-1]['body'])
    assert 'microphone array' in body and os.path.expanduser('~') + '/' not in body
    srv.push(_quick('Cause: unclear from the log.\nAction: none'))
    assert X.explain(svc, ctx, fixes, [])['cta'] is None


def test_explain_reads_nothing_of_a_private_job_and_history_records(ws):
    priv = _run(ws, 'route', dashcam_route={'clip': 'z'})
    j = _Job(run_dir=priv).job
    ctx = X.job_context(j, ws)
    assert ctx.get('private') and 'log_tail' not in ctx and not ctx['error']
    rec = {'title': 'Export', 'kind': 'export', 'status': 'failed', 'exit': 'The run failed.', 'error': 'bad format',
           'result': '', 'log_text': 'info\nerror: bad format'}
    hc = X.job_context(rec, ws)
    assert hc['log_tail'] == ['info', 'error: bad format'] and hc['error'] == 'bad format'


def test_ask_about_names_objects_by_id_and_refuses_private_ones(ws, tmp_path):
    r = _run(ws, 'drive-1')
    text, ctx = X.about('run', ws, run=r)
    assert ctx == {'about': 'run', 'run': 'runs/drive-1'} and 'runs/drive-1' in text
    assert ws not in json.dumps(ctx) and ws not in text                        # ids, never paths
    _t, c = X.about('case', ws, study='search-lead_brake', index=3, run=r)
    assert c == {'about': 'case', 'study': 'search-lead_brake', 'index': 3, 'run': 'runs/drive-1'}
    _t, c = X.about('suite_result', ws, suite=os.path.join(ws, 'suites', 'nightly'), scenario='lead_brake', seed=2,
                    run=r)
    assert c['suite'] == 'nightly' and c['seed'] == 2
    _t, c = X.about('job', ws, job=7, run_dir=r)
    assert c == {'about': 'job', 'job': 7}
    _t, c = X.about('evaluation', ws, evaluation=os.path.join(ws, 'evaluations', 'e1'))
    assert c == {'about': 'evaluation', 'evaluation': 'e1'}
    priv = _run(ws, 'route', dashcam_route={'clip': 'z'})
    for kind, ids in (('run', {'run': priv}), ('job', {'job': 1, 'run_dir': priv}),
                      ('run', {'run': os.path.join(os.path.expanduser('~'), 'Videos', 'clip')}),
                      ('run', {'run': str(tmp_path / 'elsewhere')})):
        with pytest.raises(X.Refused):
            X.about(kind, ws, **ids)


# ------------------------------------------------------------------------------------------------ show-me
@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(T, 'REGISTRY', {})
    return T.REGISTRY


def test_show_me_navigates_without_cards_and_pauses(tmp_path, srv, svc, reg, monkeypatch):
    from falcon_autonomy.assistant import agent as A
    from falcon_autonomy.assistant.session import Session
    opened = []
    T.register(T.Tool('ui_open_page', 'ui', 'act', 'Go to a page.', {'type': 'object', 'properties': {
        'page': {'type': 'string'}}, 'required': ['page']}, lambda ctx, page: opened.append(page) or {'page': page},
        preview=lambda ctx, page: f'go to {page}'))
    T.register(T.Tool('ui_highlight', 'ui', 'act', 'Outline a control.', {'type': 'object', 'properties': {
        'page': {'type': 'string'}, 'control': {'type': 'string'}, 'seconds': {'type': 'number'}},
        'required': ['page', 'control']}, lambda ctx, **a: opened.append(a['control']) or {'highlighted': 1},
        preview=lambda ctx, **a: 'outline'))
    monkeypatch.setattr(A, 'SHOW_ME_PAUSE_S', 0.3)
    monkeypatch.setattr(A, 'SHOW_ME_MAX_S', 0.4)

    class Win(A.Host):
        has_window = True

        def __init__(self):
            super().__init__(T.ToolContext(workspace=str(tmp_path)))
            self.events = []

        def event(self, kind, payload):
            self.events.append((kind, payload, time.monotonic()))
            if kind == 'approval':
                payload.decide('reject')                 # a card would be refused: show-me must not need one
    host = Win()
    ag = A.Agent(svc, Session(str(tmp_path)), host, 'ask')
    ag.show_me = True
    srv.push(tools(('ui_open_page', {'page': 'data'}), content='First, Data ▸ Record:'),
             tools(('ui_highlight', {'page': 'data', 'control': 'record.go_btn', 'seconds': 4}),
                   content='This button starts the drive.'),
             text('Press **Record** when the settings are right.'))
    t0 = time.monotonic()
    assert ag.start('How do I record a night drive?')
    assert ag.wait(20)
    assert opened == ['data', 'record.go_btn']
    kinds = [k for k, _p, _t in host.events]
    assert 'approval' not in kinds
    pauses = [p for k, p, _t in host.events if k == 'show_me']
    assert pauses == [{'tool': 'ui_open_page', 'pause_s': 0.3}, {'tool': 'ui_highlight', 'pause_s': 0.4}]
    assert time.monotonic() - t0 >= 0.7
    first = srv.chats()[0]['body']
    assert 'Show-me mode is on' in first['messages'][0]['content'] and 'such a question: show it' in \
        first['messages'][0]['content']
    assert 'ui' in ag.session.loaded
    done = [p for k, p, _t in host.events if k == 'tool_done']
    assert [d['approval'] for d in done] == ['show_me', 'show_me']
    ag.show_me = False                                    # off: ui act calls ask again (and here are refused)
    srv.push(tools(('ui_open_page', {'page': 'home'})), text('ok'))
    assert ag.start('go home') and ag.wait(20)
    assert opened[-1] != 'home' and 'approval' in [k for k, _p, _t in host.events]
    assert 'Show-me mode is on' not in srv.chats()[-1]['body']['messages'][0]['content']
    ag.close()
