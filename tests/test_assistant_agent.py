"""The assistant's runtime against a fake Groq server (no network, no Qt): the consent gate, where the key lives (and
where it never goes), the permission modes, Autopilot across a job finishing, parallel reads, Stop, malformed tool
calls, the plan, grounding, persistence, the audit log, private paths and attachments, and undo."""
import json
import os
import stat
import threading
import time

import pytest

from fake_groq import FAKE_KEY, FakeGroq, hang, text, tools
from falcon_autonomy.assistant import agent as A
from falcon_autonomy.assistant import privacy
from falcon_autonomy.assistant import tool_api as T
from falcon_autonomy.assistant.keystore import KeyStore
from falcon_autonomy.assistant.service import AssistantService
from falcon_autonomy.assistant.session import Audit, Session
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.jobs import JobQueue, JobSpec, Launcher
from falcon_autonomy.core.settings import Secrets, Settings


@pytest.fixture
def srv():
    s = FakeGroq().start()
    yield s
    s.stop()


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(T, 'REGISTRY', {})
    return T.REGISTRY


class _Handle:
    def __init__(self, job, q):
        self.job, self.q = job, q

    def cancel(self):
        self.q.on_exit(self.job, P.CANCELLED)

    def kill(self):
        self.cancel()

    def send(self, cmd):
        return True


class HoldLauncher(Launcher):
    """Starts nothing: a job runs until the test ends it with queue.on_exit."""

    def start(self, job, queue):
        return _Handle(job, queue)


@pytest.fixture
def queue(tmp_path):
    return JobQueue(HoldLauncher(), lock_dir=str(tmp_path / 'locks'),
                    meminfo_fn=lambda: {'available_gb': 64.0, 'total_gb': 64.0})


class Recorder(A.Host):
    """A headless host: records every event, answers approval cards by a rule."""

    def __init__(self, ctx, decide=None):
        super().__init__(ctx)
        self.events = []
        self.decide = decide or (lambda req: 'approve')
        self.cards = []

    def event(self, kind, payload):
        self.events.append((kind, payload))
        if kind == 'approval':
            self.cards.append(payload)
            payload.decide(self.decide(payload))

    def kinds(self, kind):
        return [p for k, p in self.events if k == kind]


def make(tmp_path, srv, queue=None, mode='ask', decide=None, enabled=True, consent=True, key=True, tpm=0):
    st = Settings(str(tmp_path / 'config' / 'settings.json'))
    ks = KeyStore(Secrets(str(tmp_path / 'config' / 'secrets.json')), use_keyring=False)
    ws = str(tmp_path / 'proj.falcon')
    svc = AssistantService(st, ks, workspace_fn=lambda: ws)
    svc.set_cfg(base_url=srv.base_url, enabled=enabled, tpm_limit=tpm)          # 0: no per-minute limit
    if consent:
        svc.accept_notice()
    if key:
        ks.set(FAKE_KEY)
    host = Recorder(T.ToolContext(queue=queue, workspace=ws), decide)
    ag = A.Agent(svc, Session(ws), host, mode)
    return svc, ag, host, ws


def run(ag, content='hello', timeout=10.0):
    assert ag.start(content)
    assert ag.wait(timeout), 'the turn did not end'


def bodies(srv):
    return [r['body'] for r in srv.chats()]


def test_nothing_is_sent_before_consent(tmp_path, srv, reg):
    svc, ag, host, ws = make(tmp_path, srv, enabled=False, consent=False)
    run(ag)
    assert host.kinds('error')[0]['kind'] == 'consent'
    from falcon_autonomy.assistant.provider import ProviderError
    for fn in (svc.models, lambda: svc.quick('hi'), svc.test_connection):
        with pytest.raises(ProviderError) as ei:
            fn()
        assert ei.value.kind == 'consent'
    svc.set_cfg(enabled=True)                                         # on, but the notice not accepted: still nothing
    run(ag)
    assert srv.requests == []
    svc.accept_notice()
    srv.push(text('Hi.'))
    run(ag, 'again')
    assert len(srv.chats()) == 1 and host.kinds('assistant')[-1]['content'] == 'Hi.'


def test_the_key_stays_in_the_secrets_file(tmp_path, srv, reg):
    T.register(T.Tool('runs_list', 'runs', 'read', 'list runs', {'type': 'object', 'properties': {}},
                      lambda ctx: {'runs': ['a', 'b'], 'note': f'key {FAKE_KEY}'}))
    svc, ag, host, ws = make(tmp_path, srv)
    srv.push(tools(('runs_list', {})), text('There are 2 runs.'))
    run(ag)
    sec = tmp_path / 'config' / 'secrets.json'
    assert FAKE_KEY in sec.read_text() and stat.S_IMODE(os.stat(sec).st_mode) == 0o600
    for root, _dirs, files in os.walk(tmp_path):
        for f in files:
            p = os.path.join(root, f)
            if p != str(sec):
                with open(p, 'rb') as fh:
                    assert FAKE_KEY.encode() not in fh.read(), p
    assert FAKE_KEY not in json.dumps(host.events, default=str)
    assert all(FAKE_KEY not in r['raw'].decode() for r in srv.chats())      # the tool result's copy was redacted
    assert os.path.isfile(os.path.join(ws, 'assistant', 'conversation.json'))
    audit = Audit(ws).read()
    assert len(audit) == 2 and all(len(a['payload_sha256']) == 64 and 'messages' not in a for a in audit)
    assert audit[0]['tools'] == ['runs_list'] and audit[1]['prompt_tokens'] == 120


def _act_tools(calls):
    T.register(T.Tool('project_rename', 'project', 'act', 'rename the project',
                      {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
                      lambda ctx, name: calls.append(name) or {'renamed': name},
                      preview=lambda ctx, name: f'name: "Old" -> "{name}"',
                      estimate=lambda ctx, name: {'seconds': 0.0, 'what': 'one project edit'}))
    T.register(T.Tool('runs_trash', 'runs', 'trash', 'move a run to the trash',
                      {'type': 'object', 'properties': {'run': {'type': 'string'}}, 'required': ['run']},
                      lambda ctx, run: calls.append('trash:' + run) or {'moved': run}))


def test_ask_mode_waits_for_approval_and_reject_reaches_the_model(tmp_path, srv, reg):
    calls = []
    _act_tools(calls)
    answers = iter(['reject', 'approve'])
    svc, ag, host, ws = make(tmp_path, srv, mode='ask', decide=lambda req: next(answers))
    srv.push(tools(('project_rename', {'name': 'A'})), text('You declined.'),
             tools(('project_rename', {'name': 'B'})), text('Renamed.'))
    run(ag, 'rename to A')
    assert calls == [] and host.cards[0].preview == 'name: "Old" -> "A"'
    assert host.cards[0].estimate == {'seconds': 0.0, 'what': 'one project edit'} and host.cards[0].allow_all
    sent = json.dumps(bodies(srv)[1]['messages'])
    assert 'declined' in sent
    run(ag, 'rename to B')
    assert calls == ['B'] and len(host.cards) == 2
    assert [a['kind'] for a in ag.session.activity] == ['other']


def test_approve_all_and_auto_mode_but_trash_always_asks(tmp_path, srv, reg):
    calls = []
    _act_tools(calls)
    svc, ag, host, ws = make(tmp_path, srv, mode='ask', decide=lambda req: 'approve_all')
    srv.push(tools(('project_rename', {'name': 'A'})), tools(('project_rename', {'name': 'B'})), text('Done.'))
    run(ag)
    assert calls == ['A', 'B'] and len(host.cards) == 1 and ag.session.allowed == ['project_rename']
    ag.mode = 'auto'
    srv.push(tools(('runs_trash', {'run': 'r1'})), text('Trashed.'))
    run(ag)
    assert host.cards[-1].tool == 'runs_trash' and not host.cards[-1].allow_all
    assert calls[-1] == 'trash:r1'
    ag.mode = 'autopilot'
    host.decide = lambda req: 'approve_all'                            # "all" is not offered for trash: still one
    srv.push(tools(('runs_trash', {'run': 'r2'})), tools(('runs_trash', {'run': 'r3'})), text('Both.'))
    run(ag)
    assert [c.tool for c in host.cards[-2:]] == ['runs_trash', 'runs_trash'] and 'runs_trash' not in ag.session.allowed


def test_autopilot_continues_when_its_job_finishes(tmp_path, srv, reg, queue):
    def queue_drive(ctx, seconds):
        job = ctx.queue.submit(JobSpec('drive', f'Drive {seconds} s', ['true'], memory_gb=0.01))
        return {'job_id': job.id}
    T.register(T.Tool('jobs_drive', 'jobs', 'act', 'queue a drive', {'type': 'object', 'properties': {
        'seconds': {'type': 'integer'}}, 'required': ['seconds']}, queue_drive))
    svc, ag, host, ws = make(tmp_path, srv, queue=queue, mode='autopilot')
    srv.push(tools(('plan_update', {'items': [{'text': 'Record a drive'}, {'text': 'Report it'}]})),
             tools(('jobs_drive', {'seconds': 30})), text('The drive is running.'),
             tools(('plan_update', {'items': [{'text': 'Record a drive', 'done': True},
                                              {'text': 'Report it', 'done': True}]})),
             text('The drive finished.'))
    assert ag.start('record a drive and report')
    deadline = time.monotonic() + 10
    while not host.kinds('waiting') or not host.kinds('waiting')[-1]:
        assert time.monotonic() < deadline, host.events
        time.sleep(0.02)
    job = queue.jobs[0]
    assert ag.busy and host.kinds('waiting')[-1] == [job.id] and len(srv.chats()) == 3
    queue.on_exit(job, P.OK)                                           # the queue says it finished: no polling
    assert ag.wait(10)
    assert host.kinds('plan')[0] == [{'text': 'Record a drive', 'done': False}, {'text': 'Report it', 'done': False}]
    msgs = bodies(srv)[3]['messages']
    ev = [m['content'] for m in msgs if m['role'] == 'user' and m['content'].startswith('[Falcon] Job')]
    assert ev and f'Job {job.id} "Drive 30 s" finished: done' in ev[0]
    assert [a['kind'] for a in ag.session.activity] == ['job'] and ag.session.activity[0]['job_id'] == job.id
    assert host.kinds('turn')[-1] == {'state': 'done'} and len(srv.chats()) == 5
    A.undo_activity(ag.session.activity[0], queue=queue)               # finished: nothing to cancel, said so


def test_plan_wait_jobs_in_ask_mode_and_stop_while_waiting(tmp_path, srv, reg, queue):
    job = queue.submit(JobSpec('train', 'Train', ['true'], memory_gb=0.01))
    svc, ag, host, ws = make(tmp_path, srv, queue=queue, mode='ask')
    srv.push(tools(('plan_wait_jobs', {'job_ids': [job.id, 999]})), text('I will wait.'))
    assert ag.start('tell me when training ends')
    deadline = time.monotonic() + 10
    while not host.kinds('waiting'):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    t0 = time.monotonic()
    ag.stop()
    assert ag.wait(3) and time.monotonic() - t0 < 2
    assert host.kinds('turn')[-1] == {'state': 'stopped'}
    res = json.loads([m for m in ag.session.messages if m.get('_name') == 'plan_wait_jobs'][0]['content'])
    assert res['scheduled'] == [job.id] and res['unknown'] == [999]


def test_stop_mid_stream(tmp_path, srv, reg):
    svc, ag, host, ws = make(tmp_path, srv)
    srv.push(hang('Looking', seconds=30))
    assert ag.start('hi')
    assert srv.hung.wait(5)
    t0 = time.monotonic()
    ag.stop()
    assert ag.wait(3) and time.monotonic() - t0 < 2
    assert host.kinds('turn')[-1] == {'state': 'stopped'} and host.kinds('delta') == ['Looking']


def test_parallel_reads_malformed_json_and_dangling_calls(tmp_path, srv, reg):
    gate = threading.Barrier(2, timeout=3)

    def slow(ctx, which):
        gate.wait()                                   # both reads must be running at once to get past this
        return {'which': which, 'value': 0.734}
    T.register(T.Tool('runs_get', 'runs', 'read', 'one run', {'type': 'object', 'properties': {
        'which': {'type': 'string'}}, 'required': ['which']}, slow))
    svc, ag, host, ws = make(tmp_path, srv)
    srv.push(tools(('runs_get', {'which': 'a'}), ('runs_get', {'which': 'b'}), ('runs_get', '{"which": "c"')),
             text('AP 0.73 on a and b (73.4 %); recall 0.91 is a guess, 12 runs.'))
    run(ag)
    done = host.kinds('tool_done')
    assert [d['summary'] for d in done[:2]] == ['which, value', 'which, value']
    assert 'not valid JSON' in done[2]['summary'] and done[2]['error']
    final = host.kinds('assistant')[-1]
    assert final['ungrounded'] == ['0.91', '12']                    # 0.73 and 73.4 % come from the results
    msgs = bodies(srv)[1]['messages']
    assert sum(m['role'] == 'tool' for m in msgs) == 3
    # a turn stopped between a call and its result: the next turn closes the call before it is sent
    ag.session.add({'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'x1', 'type': 'function', 'function': {'name': 'runs_get', 'arguments': '{}'}}]})
    srv.push(text('ok'))
    run(ag, 'next')
    sent = bodies(srv)[-1]['messages']
    i = next(k for k, m in enumerate(sent) if m.get('tool_calls') and m['tool_calls'][0]['id'] == 'x1')
    assert sent[i + 1]['role'] == 'tool' and sent[i + 1]['tool_call_id'] == 'x1'
    assert all(not k.startswith('_') for m in sent for k in m)       # Falcon's own notes never go out


def test_conversation_persists_and_clears(tmp_path, srv, reg):
    svc, ag, host, ws = make(tmp_path, srv)
    srv.push(text('First answer.'))
    run(ag, 'first')
    again = Session(ws)
    assert [m['role'] for m in again.messages] == ['user', 'assistant']
    assert again.messages[1]['content'] == 'First answer.' and again.usage['requests'] == 1
    srv.push(text('Second.'))
    ag2 = A.Agent(svc, again, host)
    run(ag2, 'second')
    assert [m['role'] for m in bodies(srv)[-1]['messages']] == ['system', 'user', 'assistant', 'user']
    again.clear()
    assert Session(ws).messages == []
    open(again.path, 'w').write('{not json')
    s3 = Session(ws)
    assert s3.messages == [] and s3.notes and os.path.isfile(again.path + '.damaged')


def test_context_too_long_squeezes_then_explains(tmp_path, srv, reg):
    from fake_groq import error
    T.register(T.Tool('docs_search', 'knowledge', 'read', 'docs', {'type': 'object', 'properties': {}},
                      lambda ctx: {'text': 'x' * 5000}))
    svc, ag, host, ws = make(tmp_path, srv)
    srv.push(*[tools(('docs_search', {}))] * 6, error(400, 'Please reduce the length of the messages'),
             text('Short now.'))
    run(ag)
    last = bodies(srv)[-1]['messages']
    tool_msgs = [m['content'] for m in last if m['role'] == 'tool']
    assert sum('elided' in c for c in tool_msgs) == 2 and host.kinds('assistant')[-1]['content'] == 'Short now.'


def test_outbound_redaction_of_tool_results(tmp_path, srv, reg):
    T.register(T.Tool('runs_where', 'runs', 'read', 'where', {'type': 'object', 'properties': {}},
                      lambda ctx: {'path': os.path.join(privacy.HOME, 'FalconWork', 'runs', 'r1'),
                                   'cfg': 'password=hunter2'}))
    svc, ag, host, ws = make(tmp_path, srv)
    srv.push(tools(('runs_where', {})), text('It is in ~/FalconWork/runs/r1.'))
    run(ag)
    raw = srv.chats()[1]['raw'].decode()
    assert '~/FalconWork/runs/r1' in raw and privacy.HOME + '/' not in raw and 'hunter2' not in raw


def test_private_paths_and_attachments_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, 'HOME', str(tmp_path / 'home'))
    vid = tmp_path / 'home' / 'Videos'
    vid.mkdir(parents=True)
    content, why = A.compose(f'summarise {vid}/clip.mp4 for me')
    assert content is None and 'private' in why
    run_dir = tmp_path / 'ws' / 'runs' / 'r'
    run_dir.mkdir(parents=True)
    (run_dir / 'run.json').write_text(json.dumps({'worker': 'dashcam-extract'}))
    (run_dir / 'notes.txt').write_text('frames')
    assert 'private' in A.compose('x', attachments=[str(run_dir / 'notes.txt')])[1]
    img = tmp_path / 'a.png'
    img.write_bytes(b'\x89PNG')
    assert 'not a text file' in A.compose('x', attachments=[str(img)])[1]
    binf = tmp_path / 'a.log'
    binf.write_bytes(b'ab\x00cd')
    assert 'not a text file' in A.compose('x', attachments=[str(binf)])[1]
    ok = tmp_path / 'error.log'
    ok.write_text('Traceback: boom')
    content, why = A.compose('why?', {'page': 'data'}, [str(ok)])
    assert why is None and 'Attached file `error.log`' in content and 'boom' in content and '"page": "data"' in content


def test_undo_restores_trash_and_cancels_jobs(tmp_path, queue):
    ws = tmp_path / 'ws'
    (ws / 'trash').mkdir(parents=True)
    (ws / 'trash' / 'r1').mkdir()
    entry = {'kind': 'trash', 'from': str(ws / 'runs' / 'r1'), 'to': str(ws / 'trash' / 'r1')}
    assert A.undo_activity(entry, workspace=str(ws)) == 'Restored r1.'
    assert (ws / 'runs' / 'r1').is_dir() and not (ws / 'trash' / 'r1').exists()
    outside = {'kind': 'trash', 'from': str(ws / 'x'), 'to': str(tmp_path / 'elsewhere')}
    assert 'not in this workspace' in A.undo_activity(outside, workspace=str(ws))
    job = queue.submit(JobSpec('drive', 'D', ['true'], memory_gb=0.01))
    assert A.undo_activity({'kind': 'job', 'job_id': job.id}, queue=queue) == f'Cancelled job {job.id}.'
    assert job.state == 'cancelled'
    assert 'cannot be undone' in A.undo_activity({'kind': 'other'})


def test_quick_uses_the_fast_model_and_counts_cost(tmp_path, srv, reg):
    svc, ag, host, ws = make(tmp_path, srv)
    totals = []
    svc.usage_listeners.append(totals.append)
    srv.push({'json': {'model': 'qwen/qwen3.8-27b', 'choices': [{'message': {'role': 'assistant', 'content':
              ' The port is taken. '}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1000000,
              'completion_tokens': 0}}})
    assert svc.explain_error('OSError: [Errno 98] Address already in use') == 'The port is taken.'
    b = bodies(srv)[0]
    assert b['model'] == 'qwen/qwen3.8-27b' and 'stream' not in b and b['reasoning_effort'] == 'none'
    assert totals[-1]['cost'] == pytest.approx(0.80) and totals[-1]['requests'] == 1
    assert Audit(ws).read()[-1]['purpose'] == 'quick'


def _answer(content='ok', model='openai/gpt-oss-20b'):
    return {'json': {'model': model, 'choices': [{'message': {'role': 'assistant', 'content': content},
                                                  'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': 50, 'completion_tokens': 5}}}


def test_fast_model_falls_back_when_qwen_is_not_listed(tmp_path, srv, reg):
    from fake_groq import MODELS
    svc, ag, host, ws = make(tmp_path, srv)
    told = []
    svc.notice_listeners.append(told.append)
    srv.models = [m for m in MODELS if not m['id'].startswith('qwen/')]
    assert 'Falcon uses openai/gpt-oss-20b instead' in svc.test_connection()       # the list is remembered
    srv.push(_answer('a'), _answer('b'))
    assert svc.quick('one') == 'a' and svc.quick('two') == 'b'
    assert [b['model'] for b in bodies(srv)] == ['openai/gpt-oss-20b'] * 2
    assert bodies(srv)[0]['reasoning_effort'] == 'low'                             # gpt-oss's own parameters
    assert len(told) == 1 and 'qwen/qwen3.8-27b' in told[0] and 'Preview' in told[0]


def test_fast_model_falls_back_when_the_server_says_it_is_gone(tmp_path, srv, reg):
    from fake_groq import error
    svc, ag, host, ws = make(tmp_path, srv)
    told = []
    svc.notice_listeners.append(told.append)
    srv.push(error(404, 'The model `qwen/qwen3.8-27b` does not exist or you do not have access to it.',
                   'model_not_found'), _answer('fine'), _answer('again'))
    assert svc.quick('x') == 'fine' and svc.quick('y') == 'again'
    assert [b['model'] for b in bodies(srv)] == ['qwen/qwen3.8-27b', 'openai/gpt-oss-20b', 'openai/gpt-oss-20b']
    assert len(told) == 1
    svc.set_cfg(fast_model='openai/gpt-oss-20b')
    srv.push(error(404, 'The model `openai/gpt-oss-20b` does not exist', 'model_not_found'))
    from falcon_autonomy.assistant.provider import ProviderError
    with pytest.raises(ProviderError) as ei:                                         # the fallback itself gone: said
        svc.quick('z')
    assert ei.value.kind == 'model'


@pytest.mark.parametrize('model,parallel,extra', [
    ('openai/gpt-oss-120b', False, {'include_reasoning': False, 'reasoning_effort': 'medium'}),
    ('qwen/qwen3.8-27b', True, {'reasoning_format': 'hidden', 'reasoning_effort': 'medium'}),
    ('local/some-model', None, {}),
])
def test_parallel_tool_calls_and_reasoning_per_model(tmp_path, srv, reg, model, parallel, extra):
    svc, ag, host, ws = make(tmp_path, srv)
    svc.set_cfg(main_model=model)
    srv.push(text('ok'))
    run(ag)
    b = bodies(srv)[0]
    assert b.get('parallel_tool_calls') == parallel and ('parallel_tool_calls' in b) == (parallel is not None)
    for k in ('include_reasoning', 'reasoning_effort', 'reasoning_format'):
        assert b.get(k) == extra.get(k), k
    if model.startswith('openai/gpt-oss'):
        assert 'reasoning_format' not in b                    # gpt-oss does not take it


def _groups_tools():
    for name, group, about in (('runs_search', 'runs', 'Search recorded runs by query.'),
                               ('runs_get', 'runs', 'One run\'s summary.'),
                               ('analysis_metrics', 'analysis', 'Evaluation metrics of a model on a drive set.'),
                               ('project_worlds', 'project', 'The project\'s worlds.')):
        T.register(T.Tool(name, group, 'read', about, {'type': 'object', 'properties': {}},
                          lambda ctx, _n=name: {'tool': _n, 'value': 0.5}))


def _sent_tools(body):
    return sorted(t['function']['name'] for t in body.get('tools') or [])


def test_tool_schemas_on_demand(tmp_path, srv, reg):
    _groups_tools()
    svc, ag, host, ws = make(tmp_path, srv)
    core = sorted(A.CORE)
    srv.push(tools(('tools_find', {'query': 'evaluation metrics'})), tools(('tools_load', {'groups': ['analysis']})),
             tools(('analysis_metrics', {})), text('Metric 0.5.'))
    run(ag, 'hello there')
    b = bodies(srv)
    assert _sent_tools(b[0]) == core and _sent_tools(b[1]) == core                 # only the core until loaded
    found = json.loads([m for m in ag.session.messages if m.get('_name') == 'tools_find'][0]['content'])
    assert found['tools'][0]['name'] == 'analysis_metrics' and found['groups']['runs']['tools'] == 2
    assert _sent_tools(b[2]) == sorted(core + ['analysis_metrics']) and ag.session.loaded == ['analysis']
    assert 'analysis' in b[2]['messages'][0]['content']                              # the system prompt says so
    assert Session(ws).loaded == ['analysis']                                        # kept with the conversation
    # the user's words make a group obviously relevant: loaded with the message
    srv.push(text('Two runs.'))
    run(ag, 'search my recorded runs')
    assert set(_sent_tools(bodies(srv)[-1])) >= {'runs_search', 'runs_get'}
    # a tool called by name before its group was loaded runs, and its group comes with the next step
    ag.session.loaded = []
    srv.push(tools(('project_worlds', {})), text('One world.'))
    run(ag, 'next')
    assert 'project_worlds' in _sent_tools(bodies(srv)[-1]) and 'project' in ag.session.loaded
    assert any(k == 'tools' for k, _p in host.events)


def test_tokens_per_minute_trimming_pacing_and_refusal(tmp_path, srv, reg, monkeypatch):
    from falcon_autonomy.assistant.provider import Usage, estimate_tokens
    from fake_groq import error
    T.register(T.Tool('runs_big', 'runs', 'read', 'A large result.', {'type': 'object', 'properties': {}},
                      lambda ctx: {'text': 'x' * 3000}))
    svc, ag, host, ws = make(tmp_path, srv, tpm=2500)
    monkeypatch.setattr(svc, 'tpm_wait', lambda tokens: 0.0)         # pacing is checked on its own below
    srv.push(*[tools(('runs_big', {}))] * 4, text('Done.'))
    run(ag, 'look at it')
    last = bodies(srv)[-1]
    assert estimate_tokens(last['messages'], last.get('tools')) <= 2500
    assert sum('elided' in m['content'] for m in last['messages'] if m['role'] == 'tool') >= 2
    assert any(k == 'notice' and 'per minute' in p for k, p in host.events)
    # pacing: the last minute's tokens count against the limit
    svc.set_cfg(tpm_limit=8000)
    svc._window.clear()
    svc.add_usage('m', Usage(5000, 0))
    real = AssistantService.tpm_wait
    assert real(svc, 2000) == 0.0 and 55.0 < real(svc, 4000) <= 60.1
    waits = iter([0.05, 0.0])
    monkeypatch.setattr(svc, 'tpm_wait', lambda tokens: next(waits))
    srv.push(text('Paced.'))
    run(ag, 'again')
    assert [p['limit'] for k, p in host.events if k == 'tpm_wait'] == [8000]
    # a 413 from the server: trimmed and sent again
    monkeypatch.setattr(svc, 'tpm_wait', lambda tokens: 0.0)
    srv.push(error(413, 'Request too large for model `openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000, '
                        'Requested 9100'), text('Smaller now.'))
    run(ag, 'once more')
    a, b = bodies(srv)[-2:]
    assert len(json.dumps(b)) < len(json.dumps(a))                                  # strictly smaller
    assert host.kinds('assistant')[-1]['content'] == 'Smaller now.'
    # a step larger than the limit even when trimmed: refused before sending
    svc.set_cfg(tpm_limit=300)
    n = len(srv.chats())
    run(ag, 'y ' * 1500)
    assert host.kinds('error')[-1]['kind'] == 'too_large' and len(srv.chats()) == n
