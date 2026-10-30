"""The assistant's chat-completions client against a fake Groq server (no network): streaming, tool calls assembled
from split and interleaved chunks, retries honouring retry-after, every failure as a sentence (never the key),
redaction of what is sent, Stop mid-stream, the model list, prices."""
import json
import socket
import threading
import time

import pytest

from fake_groq import FAKE_KEY, FakeGroq, error, hang, text, tools
from falcon_autonomy.assistant import groq as G
from falcon_autonomy.assistant import provider as PR


@pytest.fixture
def srv():
    s = FakeGroq().start()
    yield s
    s.stop()


def client(srv, **kw):
    kw.setdefault('sleep', lambda s: None)
    return G.GroqClient(srv.base_url, FAKE_KEY, **kw)


def test_stream_text_usage_and_deltas(srv):
    srv.push(text('Falcon found three runs.', pieces=4, prompt_tokens=321, completion_tokens=12))
    seen = []
    res = client(srv).chat([{'role': 'user', 'content': 'hi'}], 'openai/gpt-oss-120b', on_delta=seen.append,
                           extra=PR.model_params('openai/gpt-oss-120b', 'high'))
    assert res.content == 'Falcon found three runs.' and ''.join(seen) == res.content and len(seen) >= 4
    assert res.finish_reason == 'stop' and (res.usage.prompt, res.usage.completion) == (321, 12)
    body = srv.chats()[0]['body']
    assert body['stream'] is True and body['reasoning_effort'] == 'high' and body['include_reasoning'] is False
    assert srv.chats()[0]['headers']['Authorization'] == f'Bearer {FAKE_KEY}'


def test_parallel_tool_calls_split_across_chunks(srv):
    srv.push(tools(('runs_search', {'query': 'weather:rain', 'limit': 5}), ('runs_get', {'run': 'r1'}), split=4))
    res = client(srv).chat([{'role': 'user', 'content': 'x'}], 'm', tools=[{'type': 'function', 'function': {
        'name': 'runs_search', 'parameters': {'type': 'object'}}}])
    assert [c.name for c in res.tool_calls] == ['runs_search', 'runs_get']
    assert json.loads(res.tool_calls[0].arguments) == {'query': 'weather:rain', 'limit': 5}
    assert json.loads(res.tool_calls[1].arguments) == {'run': 'r1'}
    assert res.tool_calls[0].id == 'call_0_runs_search' and res.finish_reason == 'tool_calls'
    assert srv.chats()[0]['body']['tool_choice'] == 'auto'


def test_malformed_tool_arguments_are_kept_raw(srv):
    srv.push(tools(('runs_get', '{"run": "r1"'), split=2))
    res = client(srv).chat([{'role': 'user', 'content': 'x'}], 'm', tools=[{'type': 'function', 'function': {
        'name': 'runs_get', 'parameters': {'type': 'object'}}}])
    assert res.tool_calls[0].arguments == '{"run": "r1"'          # the agent tells the model, it does not guess


def test_429_honours_retry_after_then_succeeds(srv):
    srv.push(error(429, 'Rate limit reached for requests', headers={'retry-after': '3'}), text('ok'))
    waits, retries = [], []
    c = client(srv, sleep=waits.append)
    c.on_retry = lambda st, w, a: retries.append((st, a))
    res = c.chat([{'role': 'user', 'content': 'x'}], 'm')
    assert res.content == 'ok' and res.attempts == 2 and len(srv.chats()) == 2
    assert len(waits) == 1 and 3.0 <= waits[0] <= 3.3 and retries == [(429, 1)]


def test_5xx_backoff_with_jitter_then_a_sentence(srv):
    srv.push(*[error(503, 'Service Unavailable')] * 4)
    waits = []
    with pytest.raises(PR.ProviderError) as ei:
        client(srv, sleep=waits.append, retries=3, backoff_s=1.0).chat([{'role': 'user', 'content': 'x'}], 'm')
    assert ei.value.kind == 'server' and '503' in str(ei.value) and len(srv.chats()) == 4
    assert len(waits) == 3 and all(0.25 <= w <= cap for w, cap in zip(waits, (1.0, 2.0, 4.0)))


@pytest.mark.parametrize('resp,kind,words', [
    (error(401, 'Invalid API Key', 'invalid_api_key'), 'auth', 'API key'),
    (error(404, 'The model `qwen/qwen3-32b` has been decommissioned', 'model_decommissioned'), 'model', 'retired'),
    (error(413, 'Request too large for model in organization on tokens per minute (TPM): Limit 8000, Requested 9100'),
     'too_large', 'tokens-per-minute'),
    (error(400, 'Please reduce the length of the messages or completion. context_length_exceeded'), 'context',
     'too long'),
    (error(400, 'This model does not support tools'), 'no_tools', 'does not take tools'),
    (error(400, 'Failed to call a function. Please adjust your prompt.', 'tool_use_failed'), 'tool_use_failed',
     'could not parse'),
])
def test_failures_are_sentences_without_the_key(srv, resp, kind, words):
    srv.push(resp)
    with pytest.raises(PR.ProviderError) as ei:
        client(srv).chat([{'role': 'user', 'content': 'x'}], 'qwen/qwen3-32b')
    e = ei.value
    assert e.kind == kind and words in str(e) and FAKE_KEY not in str(e) and 'Traceback' not in str(e)
    assert len(srv.chats()) == 1                                   # none of these is retried


def test_unknown_optional_parameter_is_dropped_once(srv):
    srv.push(error(400, "property 'reasoning_effort' is unsupported for this model"), text('fine'))
    res = client(srv).chat([{'role': 'user', 'content': 'x'}], 'local-model',
                           extra={'reasoning_effort': 'low', 'include_reasoning': False})
    assert res.content == 'fine'
    first, second = (r['body'] for r in srv.chats())
    assert 'reasoning_effort' in first and 'reasoning_effort' not in second and second['include_reasoning'] is False


def test_network_down_is_a_sentence():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()                                                      # nothing listens there now
    c = G.GroqClient(f'http://127.0.0.1:{port}/v1', FAKE_KEY, retries=1, sleep=lambda s: None)
    with pytest.raises(PR.ProviderError) as ei:
        c.chat([{'role': 'user', 'content': 'x'}], 'm')
    assert ei.value.kind == 'network' and 'Cannot reach' in str(ei.value)


def test_outbound_payload_is_redacted(srv, monkeypatch):
    from falcon_autonomy.assistant import privacy
    home = privacy.HOME
    srv.push(text('ok'))
    msgs = [{'role': 'user', 'content': f'look at {home}/proj/runs/a and use token=hunter2 and {FAKE_KEY}'},
            {'role': 'tool', 'tool_call_id': 'c', 'content': json.dumps({'path': f'{home}/x', 'n': 3})}]
    client(srv).chat(msgs, 'm')
    raw = srv.chats()[0]['raw'].decode()
    assert home + '/' not in raw and '~/proj/runs/a' in raw and 'hunter2' not in raw
    assert raw.count(FAKE_KEY) == 0 and '[secret]' in raw
    assert srv.chats()[0]['headers']['Authorization'].endswith(FAKE_KEY)     # the key goes in the header only


def test_stop_cancels_mid_stream(srv):
    srv.push(hang('Let me look', seconds=20))
    tok = PR.CancelToken()
    got = []

    def on_delta(d):
        got.append(d)
        threading.Timer(0.1, tok.set).start()
    t0 = time.monotonic()
    with pytest.raises(PR.Cancelled):
        client(srv).chat([{'role': 'user', 'content': 'x'}], 'm', on_delta=on_delta, cancel=tok)
    assert got == ['Let me look'] and time.monotonic() - t0 < 3.0


def test_gate_blocks_every_request(srv):
    def gate():
        raise PR.ProviderError('consent', 'not accepted')
    c = G.GroqClient(srv.base_url, FAKE_KEY, gate=gate)
    for fn in (lambda: c.models(), lambda: c.chat([{'role': 'user', 'content': 'x'}], 'm')):
        with pytest.raises(PR.ProviderError, match='not accepted'):
            fn()
    assert srv.requests == []


def test_models_and_audit_records(srv):
    recs = []
    c = G.GroqClient(srv.base_url, FAKE_KEY, on_request=recs.append)
    ms = c.models()
    assert [m['id'] for m in ms] == ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.8-27b']
    assert ms[0]['context_window'] == 131072
    srv.push(text('ok', prompt_tokens=10, completion_tokens=2))
    c.chat([{'role': 'user', 'content': 'secret words'}], 'm')
    assert [r['purpose'] for r in recs] == ['models', 'chat']
    assert recs[1]['prompt_tokens'] == 10 and len(recs[1]['payload_sha256']) == 64
    assert 'secret words' not in json.dumps(recs)
    bad = G.GroqClient(srv.base_url, 'gsk_wrongwrongwrongwrong00')
    with pytest.raises(PR.ProviderError) as ei:
        bad.models()
    assert ei.value.kind == 'auth' and 'gsk_wrong' not in str(ei.value)


def test_model_params_prices_and_cost():
    assert PR.model_params('openai/gpt-oss-20b', quick=True) == {'reasoning_effort': 'low', 'include_reasoning': False}
    assert PR.model_params('qwen/qwen3-32b', quick=True) == {'reasoning_effort': 'none'}
    assert PR.model_params('qwen/qwen3.8-27b') == {'reasoning_format': 'hidden', 'reasoning_effort': 'medium'}
    assert PR.model_params('llama-3.1-8b-instant') == {}
    p = PR.parse_prices('# mine\nopenai/gpt-oss-120b  0.15  0.60\nlocal/x 0 0\n')
    assert p == {'openai/gpt-oss-120b': [0.15, 0.6], 'local/x': [0.0, 0.0]}
    assert PR.parse_prices(PR.format_prices(p)) == p
    assert PR.cost(p, 'openai/gpt-oss-120b', PR.Usage(1_000_000, 500_000)) == pytest.approx(0.45)
    assert PR.cost(p, 'unknown', PR.Usage(10, 10)) is None
    with pytest.raises(ValueError, match='two prices'):
        PR.parse_prices('model 1')
    assert PR.scrubber(FAKE_KEY)(f'x {FAKE_KEY} y') == 'x [secret] y'
