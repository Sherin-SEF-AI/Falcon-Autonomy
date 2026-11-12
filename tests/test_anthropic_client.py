"""The Claude provider (assistant.anthropic_client) on the official SDK against a fake Claude API (no network, never the
real key): the request as the Claude API reference says it must be (adaptive thinking left implicit, effort explicit,
no sampling parameters, tool_choice auto, strict tools with eager input streaming, refusal fallback on by default,
cache breakpoints), streaming with thinking blocks, the stop reasons (tool_use, end_turn, max_tokens, pause_turn,
refusal), fallback blocks, the typed errors (429 with retry-after, 529, 401, 400) with request ids in the audit, Stop,
and — through the agent — a conversation that is append-only, sends Claude's thinking blocks back unchanged, groups a
step's tool results in one message and reads its cached prefix on the next step."""
import json
import threading
import time

import pytest

pytest.importorskip('anthropic')

from fake_anthropic import (FAKE_KEY, FakeAnthropic, error, fallback, redacted, reply, text,  # noqa: E402
                            thinking, tool_use)
from falcon_autonomy.assistant import agent as A  # noqa: E402
from falcon_autonomy.assistant import anthropic_client as C  # noqa: E402
from falcon_autonomy.assistant import provider as PR  # noqa: E402
from falcon_autonomy.assistant import tool_api as T  # noqa: E402
from falcon_autonomy.assistant.keystore import KeyStore  # noqa: E402
from falcon_autonomy.assistant.service import AssistantService  # noqa: E402
from falcon_autonomy.assistant.session import Audit, Session  # noqa: E402
from falcon_autonomy.core.settings import Secrets, Settings  # noqa: E402

OPUS, SONNET = 'claude-opus-5-5', 'claude-sonnet-5-5'
SYS = 'You are a test assistant. ' * 60                     # long enough for the fake's cache minimum


@pytest.fixture
def srv():
    s = FakeAnthropic().start()
    yield s
    s.stop()


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(T, 'REGISTRY', {})
    return T.REGISTRY


def client(srv, **kw):
    kw.setdefault('sleep', lambda s: None)
    return C.AnthropicClient(FAKE_KEY, srv.base_url, **kw)


def fn_tool(name, params, desc='a tool'):
    return {'type': 'function', 'function': {'name': name, 'description': desc, 'parameters': params}}


# ======================================================================= the request
def test_request_follows_the_reference(srv):
    srv.push(reply([thinking('let me see'), text('Three runs.')]))
    tools = [fn_tool('runs_search', {'type': 'object', 'properties': {
                 'query': {'type': 'string', 'maxLength': 200}, 'limit': {'type': 'integer', 'minimum': 1}}}),
             fn_tool('project_set', {'type': 'object', 'properties': {'value': {'type': 'object'}}}),
             fn_tool('a_first', {'type': 'object', 'properties': {}})]
    deltas = []
    res = client(srv, effort='high').chat([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': 'hi'}],
                                          OPUS, tools=tools, on_delta=deltas.append, temperature=0.3)
    assert res.content == 'Three runs.' and ''.join(deltas) == 'Three runs.' and res.finish_reason == 'stop'
    req = srv.messages()[0]
    b = req['body']
    assert b['model'] == OPUS and b['output_config'] == {'effort': 'high'} and b['stream'] is True
    assert 'thinking' not in b                                      # adaptive: left implicit, never disabled
    assert not {'temperature', 'top_p', 'top_k'} & set(b)           # these models take none
    assert b['tool_choice'] == {'type': 'auto'}
    assert [t['name'] for t in b['tools']] == ['a_first', 'project_set', 'runs_search']      # sorted: stable bytes
    assert all(t['eager_input_streaming'] is True for t in b['tools'])
    rs = next(t for t in b['tools'] if t['name'] == 'runs_search')
    assert rs['strict'] is True and rs['input_schema']['additionalProperties'] is False
    assert 'maxLength' not in json.dumps(rs) and 'minimum' not in json.dumps(rs)    # Falcon validates these itself
    ps = next(t for t in b['tools'] if t['name'] == 'project_set')
    assert 'strict' not in ps                                       # a free-form object cannot be strict
    assert b['fallbacks'] == 'default' and req['headers'].get('anthropic-beta') == C.FALLBACK_BETA
    assert b['system'] == [{'type': 'text', 'text': SYS, 'cache_control': {'type': 'ephemeral'}}]
    assert b['cache_control'] == {'type': 'ephemeral'}              # automatic caching of the conversation's tail
    assert req['headers']['Authorization'] == f'Bearer {FAKE_KEY}' and FAKE_KEY not in req['raw'].decode()
    assert res.native[0] == {'type': 'thinking', 'thinking': 'let me see', 'signature': 'sig-FAKE-abc123=='}


def test_fallback_off_effort_and_disallowed_models(srv):
    srv.push(reply([text('ok')]))
    client(srv, fallback=False).chat([{'role': 'user', 'content': 'x'}], SONNET, extra={'effort': 'low'})
    req = srv.messages()[0]
    assert 'fallbacks' not in req['body'] and 'anthropic-beta' not in req['headers']
    assert req['body']['output_config'] == {'effort': 'low'} and 'tools' not in req['body']
    for model in ('claude-opus-5', 'claude-opus-5-5-20261001', 'claude-haiku-5-5', ''):
        with pytest.raises(PR.ProviderError) as ei:
            client(srv).chat([{'role': 'user', 'content': 'x'}], model)
        assert ei.value.kind == 'model'
    with pytest.raises(PR.ProviderError, match='forced tool choice'):
        client(srv).chat([{'role': 'user', 'content': 'x'}], OPUS, tools=[fn_tool('a', {'type': 'object'})],
                         tool_choice='required')
    assert len(srv.messages()) == 1                                 # refused before anything was sent
    assert C.effort_of('', 'high') == 'high' and C.effort_of('xhigh', 'low') == 'xhigh'
    assert C.effort_of('', 'nonsense') == 'medium'


def test_models_are_filtered_to_the_two(srv):
    ms = client(srv).models()
    assert [m['id'] for m in ms] == [OPUS, SONNET] and ms[0]['context_window'] == 1000000


def test_strict_schema_rules():
    ok, why = C.strict_schema({'type': 'object', 'properties': {'items': {'type': 'array', 'maxItems': 3, 'items': {
        'type': 'object', 'properties': {'text': {'type': 'string', 'minLength': 1}}, 'required': ['text']}}}})
    assert why == '' and ok['properties']['items']['items']['additionalProperties'] is False
    assert 'maxItems' not in json.dumps(ok) and ok['additionalProperties'] is False
    assert C.strict_schema({'type': 'object', 'properties': {'v': {'type': 'object'}}})[0] is None
    assert 'additional' in C.strict_schema({'type': 'object', 'properties': {}, 'additionalProperties': True})[1]
    assert C.strict_schema({'type': 'object', 'properties': {'x': {'oneOf': []}}})[0] is None


# ======================================================================= stop reasons and blocks
def test_tool_use_raw_json_is_kept_strictly(srv):
    srv.push(reply([tool_use('runs_get', {'run': 'r1'}, 'toolu_1'), tool_use('runs_get', '{"run": "r2"', 'toolu_2')],
                   stop='tool_use'))
    res = client(srv).chat([{'role': 'user', 'content': 'x'}], OPUS, tools=[fn_tool('runs_get', {
        'type': 'object', 'properties': {'run': {'type': 'string'}}})])
    assert res.finish_reason == 'tool_calls' and [c.id for c in res.tool_calls] == ['toolu_1', 'toolu_2']
    assert json.loads(res.tool_calls[0].arguments) == {'run': 'r1'}
    assert res.tool_calls[1].arguments == '{"run": "r2"'           # truncated: kept raw, never the tolerant parse


def test_refusal_runs_nothing_and_says_why(srv):
    srv.push(reply([text('Sure, here'), tool_use('runs_get', {'run': 'x'})], stop='refusal',
                   stop_details={'type': 'refusal', 'category': 'cyber', 'explanation': 'not this'}))
    res = client(srv).chat([{'role': 'user', 'content': 'x'}], OPUS, tools=[fn_tool('runs_get', {
        'type': 'object', 'properties': {'run': {'type': 'string'}}})])
    assert res.finish_reason == 'refusal' and res.tool_calls == []
    assert res.stop_details['category'] == 'cyber' and res.stop_details['explanation'] == 'not this'


def test_fallback_blocks_and_iterations(srv):
    its = [{'type': 'message', 'model': OPUS, 'input_tokens': 100, 'output_tokens': 0, 'cache_read_input_tokens': 0,
            'cache_creation_input_tokens': 0},
           {'type': 'fallback_message', 'model': 'claude-opus-5', 'input_tokens': 100, 'output_tokens': 20,
            'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}]
    srv.push(reply([text('partial '), thinking('x'), tool_use('runs_get', {'run': 'a'}, 'toolu_a'), fallback(),
                    text('served'), tool_use('runs_get', {'run': 'b'}, 'toolu_b')], stop='tool_use',
                   model='claude-opus-5', iterations=its))
    res = client(srv).chat([{'role': 'user', 'content': 'x'}], OPUS, tools=[fn_tool('runs_get', {
        'type': 'object', 'properties': {'run': {'type': 'string'}}})])
    assert res.fallback == {'from': OPUS, 'to': 'claude-opus-5', 'category': 'cyber'} and res.model == 'claude-opus-5'
    assert [c.id for c in res.tool_calls] == ['toolu_b']             # the declined model's call is not this answer's
    assert [m for m, _u in res.iterations] == [OPUS, 'claude-opus-5']
    echoed = C.echo_blocks(res.native)                               # the API's rule for echoing a fallback turn
    assert [b['type'] for b in echoed] == ['text', 'fallback', 'text', 'tool_use']
    assert res.native[3]['from'] == {'model': OPUS}                  # the SDK's from_ goes back as "from"


def test_typed_errors_retries_and_request_ids(srv, tmp_path):
    recs = []
    srv.push(error(429, 'rate_limit_error', 'slow down', {'retry-after': '3'}), error(529, 'overloaded_error', 'busy'),
             reply([text('ok')]))
    waits, retries = [], []
    c = client(srv, sleep=waits.append, on_request=recs.append)
    c.on_retry = lambda st, w, a: retries.append((st, a))
    res = c.chat([{'role': 'user', 'content': 'x'}], OPUS)
    assert res.content == 'ok' and res.attempts == 3 and len(srv.messages()) == 3
    assert 3.0 <= waits[0] <= 3.3 and retries == [(429, 1), (529, 2)]
    assert recs[-1]['request_id'].startswith('req_fake_') and recs[-1]['provider'] == 'anthropic'
    for resp, kind, words in ((error(401, 'authentication_error', 'invalid x-api-key'), 'auth', 'API key'),
                              (error(400, 'invalid_request_error', 'messages: bad'), 'bad_request', 'rejected'),
                              (error(404, 'not_found_error', 'model: x'), 'model', 'not available'),
                              (error(403, 'permission_error', 'nope'), 'forbidden', 'refused'),
                              (error(400, 'invalid_request_error', 'prompt is too long: 1200000 tokens'), 'context',
                               'too long'),
                              (error(400, 'invalid_request_error', 'messages.3.content.0: Invalid `signature` in '
                                     '`thinking` block. The block is bound to a different conversation.'), 'binding',
                               'reasoning')):
        srv.push(resp)
        n = len(srv.messages())
        with pytest.raises(PR.ProviderError) as ei:
            client(srv, on_request=recs.append).chat([{'role': 'user', 'content': 'x'}], OPUS)
        e = ei.value
        assert e.kind == kind and words in str(e) and FAKE_KEY not in str(e) and 'Traceback' not in str(e)
        assert e.request_id.startswith('req_fake_') and len(srv.messages()) == n + 1        # none retried
    srv.push(*[error(500, 'api_error', 'boom')] * 3)
    with pytest.raises(PR.ProviderError) as ei:
        client(srv, retries=2).chat([{'role': 'user', 'content': 'x'}], OPUS)
    assert ei.value.kind == 'server'
    srv.push(reply([text('half '), text('more')], error_after=1))  # overloaded mid-stream after text: not retried
    with pytest.raises(PR.ProviderError) as ei:
        client(srv).chat([{'role': 'user', 'content': 'x'}], OPUS)
    assert ei.value.kind == 'server' and ei.value.status == -1
    Audit(str(tmp_path)).write(dict(recs[-1], extra='dropped'))
    got = Audit(str(tmp_path)).read()[-1]
    assert 'request_id' in got and 'extra' not in got


def test_stop_cancels_mid_stream(srv):
    srv.push(reply([text('Let me look'), text('never')], hang=20))
    tok = PR.CancelToken()
    got = []

    def on_delta(d):
        got.append(d)
        threading.Timer(0.1, tok.set).start()
    t0 = time.monotonic()
    with pytest.raises(PR.Cancelled):
        client(srv).chat([{'role': 'user', 'content': 'x'}], OPUS, on_delta=on_delta, cancel=tok)
    assert got and time.monotonic() - t0 < 3.0


def test_gate_and_key(srv):
    def gate():
        raise PR.ProviderError('consent', 'not accepted')
    c = C.AnthropicClient(FAKE_KEY, srv.base_url, gate=gate)
    for fn in (c.models, lambda: c.chat([{'role': 'user', 'content': 'x'}], OPUS)):
        with pytest.raises(PR.ProviderError, match='not accepted'):
            fn()
    with pytest.raises(PR.ProviderError) as ei:
        C.AnthropicClient('', srv.base_url).chat([{'role': 'user', 'content': 'x'}], OPUS)
    assert ei.value.kind == 'no_key'
    assert srv.requests == []


def test_outbound_text_is_redacted_but_claudes_blocks_are_not(srv):
    from falcon_autonomy.assistant import privacy
    home = privacy.HOME
    srv.push(reply([text('ok')]))
    blocks = [thinking(f'thought about {home}/x', 'sig-1'), redacted('ENC-1'), text('earlier')]
    msgs = [{'role': 'user', 'content': f'see {home}/proj and token=hunter2'},
            {'role': 'assistant', 'content': 'earlier', '_anthropic': blocks},
            {'role': 'user', 'content': 'and now'}]
    client(srv).chat(msgs, OPUS)
    sent = srv.messages()[0]['body']['messages']
    assert '~/proj' in sent[0]['content'] and 'hunter2' not in sent[0]['content']
    assert sent[1]['content'] == blocks                               # byte for byte: a changed block is a 400


# ======================================================================= through the agent
class Recorder(A.Host):
    def __init__(self, ctx, decide=None):
        super().__init__(ctx)
        self.events = []
        self.decide = decide or (lambda req: 'approve')

    def event(self, kind, payload):
        self.events.append((kind, payload))
        if kind == 'approval':
            payload.decide(self.decide(payload))

    def kinds(self, kind):
        return [p for k, p in self.events if k == kind]

    def context_summary(self):
        return {'project': 'demo', 'page': getattr(self, 'page', 'home')}


def make(tmp_path, srv, mode='auto', fallback_on=True, effort=''):
    st = Settings(str(tmp_path / 'config' / 'settings.json'))
    ks = KeyStore(Secrets(str(tmp_path / 'config' / 'secrets.json')), use_keyring=False)
    ws = str(tmp_path / 'proj.falcon')
    svc = AssistantService(st, ks, workspace_fn=lambda: ws)
    svc.set_cfg(enabled=True, provider='anthropic', reasoning_effort='high')
    svc.set_cfg(base_url=srv.base_url, anthropic={'fallback': fallback_on, 'effort': effort})
    svc.accept_notice()
    svc.claude_keys.set(FAKE_KEY)
    host = Recorder(T.ToolContext(workspace=ws))
    return svc, A.Agent(svc, Session(ws), host, mode), host, ws


def run(ag, content='hello', timeout=15.0):
    assert ag.start(content)
    assert ag.wait(timeout), 'the turn did not end'


def _reads(reg):
    def get(ctx, run):
        if run == 'bad':
            raise T.ToolError('no run named bad')
        return {'run': run, 'ap': 0.734}
    T.register(T.Tool('runs_get', 'runs', 'read', 'one run', {'type': 'object', 'properties': {
        'run': {'type': 'string', 'maxLength': 40}}, 'required': ['run']}, get))
    T.register(T.Tool('project_notes', 'project', 'read', 'notes', {'type': 'object', 'properties': {
        'v': {'type': 'object'}}}, lambda ctx, v=None: {'ok': True}))


def test_agent_conversation_is_append_only_and_cached(tmp_path, srv, reg):
    _reads(reg)
    svc, ag, host, ws = make(tmp_path, srv)
    assert svc.cfg()['main_model'] == OPUS and svc.cfg()['fast_model'] == SONNET
    srv.push(reply([thinking('plan', 'sig-A'), text('Looking.'), tool_use('runs_get', {'run': 'r1'}, 'toolu_1'),
                    tool_use('runs_get', {'run': 'bad'}, 'toolu_2')], stop='tool_use'),
             reply([thinking('done', 'sig-B'), text('Run r1 has AP 0.734.')]))
    run(ag, 'how is r1?')
    host.page = 'data'                                                # the page changes between messages
    srv.push(reply([text('Yes.')]))
    run(ag, 'thanks')
    b = srv.bodies()
    assert len(b) == 3
    # the system prompt and the tools are byte-identical every step (the cache's prefix) …
    assert all(x['system'] == b[0]['system'] and x['tools'] == b[0]['tools'] for x in b)
    assert b[0]['system'][0]['text'] == A.SYSTEM_CLAUDE and 'cache_control' in b[0]['system'][0]
    names = [t['name'] for t in b[0]['tools']]
    assert names == sorted(names) and 'tools_find' not in names and 'plan_update' in names
    assert ag.session.native['tools'] == names
    # … so the second step reads it from the cache
    usages = host.kinds('usage')
    assert usages[0]['cache_write'] > 0 and usages[1]['cache_read'] > 0
    # each request's messages begin with the previous request's messages, unchanged (append-only)
    for prev, nxt in zip(b, b[1:]):
        assert nxt['messages'][:len(prev['messages'])] == prev['messages']
    m1 = b[1]['messages']
    assert m1[0] == {'role': 'user', 'content': 'how is r1?'}
    assert m1[1]['role'] == 'system' and 'Permission mode: Auto' in m1[1]['content']     # a note, not the prompt
    assert m1[2]['role'] == 'assistant' and m1[2]['content'][0] == {'type': 'thinking', 'thinking': 'plan',
                                                                    'signature': 'sig-A'}
    results = m1[3]
    assert results['role'] == 'user' and [r['tool_use_id'] for r in results['content']] == ['toolu_1', 'toolu_2']
    assert 'is_error' not in results['content'][0] and results['content'][1]['is_error'] is True
    m2 = b[2]['messages']
    notes = [m for m in m2 if m['role'] == 'system']
    assert len(notes) == 2 and '"page": "data"' in notes[1]['content']    # the new page: a new note, appended
    assert m2[-2] == {'role': 'user', 'content': 'thanks'} and m2[-1]['role'] == 'system'
    assert host.kinds('assistant')[-1]['content'] == 'Yes.'
    final = [p for p in host.kinds('assistant') if p.get('final')][0]
    assert final['ungrounded'] == []                                 # 0.734 came from the tool
    assert not any(k == 'tools' for k, _ in host.events)            # nothing "loaded": Claude has every tool
    # the conversation as saved round-trips Claude's blocks
    again = Session(ws)
    assert again.messages[2]['_anthropic'][0]['signature'] == 'sig-A'
    assert again.native['tools'] == names
    assert ag.session.usage['cache_read'] > 0 and ag.session.usage['cost'] > 0
    assert b[0]['output_config'] == {'effort': 'high'}                # Falcon's reasoning effort, explicit


def test_agent_refusal_max_tokens_pause_and_invalid_json(tmp_path, srv, reg):
    _reads(reg)
    calls = []
    T.register(T.Tool('runs_tag', 'runs', 'read', 'tag', {'type': 'object', 'properties': {
        'run': {'type': 'string'}}, 'required': ['run']}, lambda ctx, run: calls.append(run) or {'ok': True}))
    svc, ag, host, ws = make(tmp_path, srv)
    # a refusal: its tool call never runs, the user is told the category, and it is never sent back
    srv.push(reply([text('Sure'), tool_use('runs_tag', {'run': 'x'})], stop='refusal',
                   stop_details={'type': 'refusal', 'category': 'bio', 'explanation': 'declined'}))
    run(ag, 'first')
    assert calls == [] and 'category: bio' in host.kinds('error')[-1]['message']
    # cut at max_tokens with a tool call: not run, an error result goes back; then pause_turn goes back as it is
    srv.push(reply([tool_use('runs_tag', '{"run": "y', 'toolu_c')], stop='max_tokens'),
             reply([thinking('', 'sig-P')], stop='pause_turn'),
             reply([tool_use('runs_tag', '{"run": "z"', 'toolu_j')], stop='tool_use'),
             reply([text('Done.')]))
    run(ag, 'second')
    assert calls == []                                               # neither the cut nor the malformed call ran
    b = srv.bodies()
    assert len(b) == 5
    second = b[1]['messages']
    # the refused answer is not in the history; the note sent with it was never answered, and a system message may
    # not stand before a user message, so it goes again after the new message
    assert second[:2] == [{'role': 'user', 'content': 'first'}, {'role': 'user', 'content': 'second'}]
    assert second[2]['role'] == 'system' and second[2]['content'] == b[0]['messages'][1]['content']
    cut = b[2]['messages'][-1]
    assert cut['role'] == 'user' and cut['content'][0]['is_error'] is True and 'cut off' in cut['content'][0]['content']
    assert b[3]['messages'][-1] == {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': '',
                                                                      'signature': 'sig-P'}]}
    bad = b[4]['messages'][-1]['content'][0]
    assert bad['tool_use_id'] == 'toolu_j' and bad['is_error'] is True and 'JSON' in bad['content']
    for prev, nxt in zip(b[1:], b[2:]):
        assert nxt['messages'][:len(prev['messages'])] == prev['messages']


def test_agent_fallback_notice_cost_and_binding_recovery(tmp_path, srv, reg):
    _reads(reg)
    svc, ag, host, ws = make(tmp_path, srv)
    its = [{'type': 'message', 'model': OPUS, 'input_tokens': 1000, 'output_tokens': 0, 'cache_read_input_tokens': 0,
            'cache_creation_input_tokens': 0},
           {'type': 'fallback_message', 'model': OPUS, 'input_tokens': 1000, 'output_tokens': 100,
            'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}]
    srv.push(reply([fallback(OPUS, OPUS, 'cyber'), thinking('t', 'sig-F'), text('Answered.')], iterations=its))
    run(ag, 'q1')
    assert any('declined this step' in n and 'category: cyber' in n for n in host.kinds('notice'))
    assert svc.totals['cost'] == pytest.approx((2000 * 4 + 100 * 20) / 1e6)      # per attempt, at the published prices
    # the API says earlier thinking is bound to another conversation: once, without it, and remembered
    srv.push(error(400, 'invalid_request_error', 'messages.2.content.1: Invalid `signature` in `thinking` block. The '
                   'block is bound to a different conversation.'), reply([text('ok')]), reply([text('ok2')]))
    run(ag, 'q2')
    b = srv.bodies()
    assert any(bl['type'] == 'thinking' for bl in b[1]['messages'][2]['content'])
    assert not any(bl['type'] == 'thinking' for bl in b[2]['messages'][2]['content'])
    assert b[2]['messages'][2]['content'][0]['type'] == 'fallback'
    assert ag.session.native['drop_thinking_before'] > 0
    run(ag, 'q3')
    assert not any(bl['type'] == 'thinking' for bl in srv.bodies()[3]['messages'][2]['content'])


def test_service_provider_routing_prices_and_ready(tmp_path, srv):
    st = Settings(str(tmp_path / 'settings.json'))
    ks = KeyStore(Secrets(str(tmp_path / 'secrets.json')), use_keyring=False)
    svc = AssistantService(st, ks)
    svc.set_cfg(main_model='openai/gpt-oss-120b', enabled=True)
    svc.accept_notice()
    svc.set_cfg(provider='anthropic')
    assert svc.cfg()['main_model'] == OPUS and svc.cfg()['tpm_limit'] == 0
    assert 'Claude API key' in svc.ready()[1]
    svc.set_cfg(main_model=SONNET, tpm_limit=50000)                  # routed to Claude's own settings
    svc.set_cfg(provider='groq')
    assert svc.cfg()['main_model'] == 'openai/gpt-oss-120b' and svc.cfg()['tpm_limit'] == 8000
    svc.set_cfg(provider='anthropic')
    assert svc.cfg()['main_model'] == SONNET and svc.cfg()['tpm_limit'] == 50000
    svc.claude_keys.set(FAKE_KEY)
    assert svc.ready() == (True, '') and not ks.has()                # the two keys are kept apart
    assert Secrets(str(tmp_path / 'secrets.json')).get('anthropic_api_key') == FAKE_KEY
    assert svc.params(OPUS) == {'effort': 'medium'} and svc.params(SONNET, quick=True) == {'effort': 'low'}
    assert OPUS not in svc.cfg()['prices']                           # the published prices, unless the user's
    assert svc.cost_of(OPUS, PR.Usage(1_000_000, 1_000_000, 1_000_000, 1_000_000)) == pytest.approx(29.2)
    assert svc.cost_of(SONNET, PR.Usage(0, 0, 1_000_000, 0)) == pytest.approx(0.10)
    svc.set_cfg(base_url=srv.base_url)
    srv.push(reply([text('Short answer.')], model=SONNET))
    assert svc.quick('explain') == 'Short answer.'
    body = srv.bodies()[0]
    assert body['model'] == SONNET and body['output_config'] == {'effort': 'low'} and 'tools' not in body
    assert body['system'][0]['text'].startswith('You are Falcon') and 'cache_control' not in body['system'][0]
    assert 'Anthropic offers 2 of the models' in svc.test_connection()
    srv.push(reply([], stop='refusal', stop_details={'type': 'refusal', 'category': 'cyber'}))
    with pytest.raises(PR.ProviderError) as ei:
        svc.quick('x')
    assert ei.value.kind == 'refusal' and 'cyber' in str(ei.value)


def test_at_most_twenty_strict_tools_act_tools_first():
    """The Messages API refuses more than 20 strict tools in one request (found against the real API: a 400 for 115);
    with Falcon's whole registry only 20 are strict, act tools first, and the bytes do not depend on input order."""
    import json as _json
    from falcon_autonomy.assistant import tools  # noqa: F401 (registers them)
    from falcon_autonomy.assistant.tool_api import REGISTRY, schemas
    allt = schemas()
    defs, loose = C.tool_defs(allt)
    strict = [d['name'] for d in defs if d.get('strict')]
    assert len(strict) == C.MAX_STRICT == 20 and len(defs) == len(allt)
    acts = sorted(n for n in strict if REGISTRY[n].tier != 'read')
    eligible_acts = sorted(n for n, t in REGISTRY.items() if t.tier != 'read' and n in {d['name'] for d in defs}
                           and C.strict_schema((REGISTRY[n].params or {}))[0] is not None)
    assert acts == eligible_acts[:len(acts)]                         # act tools take the slots first, by name
    assert any('20 strict tools' in why for why in loose.values())
    again, _ = C.tool_defs(list(reversed(allt)))
    assert _json.dumps(defs, sort_keys=True) == _json.dumps(again, sort_keys=True)
