"""The Claude API as a provider of the assistant, on Anthropic's official Python SDK (`anthropic`, the extra
`falcon-autonomy[claude]`); the same interface the agent uses of groq.GroqClient: chat() with tools, deltas streamed to
the panel, models(), usage and cost, Stop.

Models: only claude-opus-5-5 (the main agent) and claude-sonnet-5-5 (the fast lane) — exact ids; any other is refused
before anything is sent, and the model list (models()) is filtered to them.

What every request is (docs/assistant.md, "Claude"; checked against the Claude API reference shipped with the SDK):
- streamed with client.beta.messages.stream(...) and the SDK's get_final_message(); text deltas go to on_delta;
- adaptive thinking: `thinking` is never sent (Opus 5.5 always thinks; disabled and budget_tokens are 400s), and
  output_config.effort is always set (Opus 5.5's default is medium: Falcon sends the user's effort explicitly);
- no temperature, top_p or top_k (removed on these models; the SDK has no such arguments), no assistant prefill;
- tool_choice auto only (forced any/tool are 400s on these models); every tool has eager_input_streaming, and strict
  when its schema can be strict (strict_schema: objects closed with additionalProperties false, the keywords strict
  mode does not support left to Falcon's own validation; a free-form object argument cannot be — that tool goes
  without strict, and tool_defs says why). Every input is parsed from the raw streamed JSON (json.loads, not the SDK's
  tolerant parser) and validated against the tool's schema before it runs (tool_api.call), so a truncated or invalid
  input becomes an error result, never a call;
- the stop reason is read before the content: tool_use (the tool calls), end_turn, max_tokens (cut: its tool calls are
  not run), pause_turn (sent back as it is to continue), refusal with stop_details' category and explanation (nothing
  of a refused response runs);
- server-side refusal fallback: fallbacks "default" under the beta server-side-fallback-2026-07-01, on unless the user
  turns it off (Settings ▸ Assistant ▸ Provider) — a request Claude's safety classifiers decline is re-run by
  Anthropic on the model it recommends for that category, and the answer says which model served it;
- prompt caching: the tools (sorted by name, the same every step) then the frozen system prompt with a cache
  breakpoint on it, then the conversation with automatic caching (top-level cache_control) for its growing tail;
- the conversation is sent as Claude returned it: an assistant turn's content blocks (thinking blocks included) are
  kept with the message unchanged (_anthropic) and sent back byte for byte; earlier blocks are never edited or
  reordered; page and mode notes are mid-conversation system messages (to_native), never edits of the system prompt;
  after a mid-output fallback the blocks before the fallback block are echoed as the API requires (echo_blocks);
- tool results of one step go back together in one user message, a failed tool's as is_error.

Failures are the SDK's typed errors mapped to provider.ProviderError sentences (with the request id, which the audit
log keeps): RateLimitError waits for its retry-after; overloaded (529) and 5xx back off with full jitter; a connection
failing before anything streamed is retried; BadRequest, Authentication, PermissionDenied, NotFound, RequestTooLarge
are not. The SDK's own retries are off (max_retries=0) so Stop and the panel's "waiting" line see every retry.

Text only: what leaves the machine passes privacy.redact (the user's messages, tool results, notes; Claude's own
blocks came from the server and go back unchanged). The key is read from Falcon's key store only when a request is
made, sent only in the request header, and scrubbed from every error.
"""
from __future__ import annotations

import copy
import hashlib
import json
import queue
import random
import socket
import threading

from . import privacy
from .provider import Cancelled, CancelToken, ChatResult, ProviderError, ToolCall, Usage, scrubber

MAIN_MODEL = 'claude-opus-5-5'
FAST_MODEL = 'claude-sonnet-5-5'
ALLOWED = (MAIN_MODEL, FAST_MODEL)
DEFAULT_BASE_URL = 'https://api.anthropic.com'
FALLBACK_BETA = 'server-side-fallback-2026-07-01'
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
# $ per million tokens: input, output, cache read, 5-minute cache write (Anthropic's published prices)
PRICES = {MAIN_MODEL: [4.0, 20.0, 0.20, 5.0], FAST_MODEL: [2.0, 10.0, 0.10, 2.50]}
AGENT_MAX_TOKENS = 32000          # thinking counts toward it: room for the thinking and a long answer
QUICK_MIN_TOKENS = 4000           # the fast lane's floor (a one-line answer still thinks first)
JSON_RETRIES = 2                  # a stream whose tool input the SDK cannot parse at all is asked again this often
USER_AGENT = 'FalconAutonomy-assistant'

# JSON Schema keywords strict tool use does not support: dropped from what is sent (Falcon validates them itself)
STRICT_DROP = ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf', 'minLength', 'maxLength',
               'pattern', 'minItems', 'maxItems', 'uniqueItems', 'minProperties', 'maxProperties', 'default',
               'examples')
STRICT_FORMATS = ('date-time', 'time', 'date', 'duration', 'email', 'hostname', 'uri', 'ipv4', 'ipv6', 'uuid')
STRICT_UNSUPPORTED = ('oneOf', 'not', 'if', 'then', 'else', 'patternProperties', 'dependentSchemas',
                      'propertyNames', 'unevaluatedProperties', 'prefixItems', 'contains')
INTERNAL_BLOCKS = ('thinking', 'redacted_thinking', 'tool_use', 'server_tool_use')


def allowed(model: str) -> bool:
    return model in ALLOWED


def effort_of(setting: str, falcon_effort: str = 'medium') -> str:
    """The output_config.effort sent: the Claude setting when it names a level, else Falcon's reasoning effort
    (low, medium, high) — always explicit."""
    if setting in EFFORTS:
        return setting
    return falcon_effort if falcon_effort in EFFORTS else 'medium'


# --------------------------------------------------------------------------------------------- tools
def strict_schema(schema, path: str = '$'):
    """(schema for strict tool use, '') or (None, why it cannot be strict). Objects with properties are closed
    (additionalProperties false); unsupported constraints are dropped (Falcon's own validation keeps them); a
    free-form object, an open additionalProperties or an unsupported construct cannot be strict."""
    if isinstance(schema, list):
        out = []
        for i, s in enumerate(schema):
            v, why = strict_schema(s, f'{path}[{i}]')
            if v is None:
                return None, why
            out.append(v)
        return out, ''
    if not isinstance(schema, dict):
        return schema, ''
    bad = [k for k in STRICT_UNSUPPORTED if k in schema]
    if bad:
        return None, f'{path} uses {bad[0]}, which strict mode does not support'
    out = {}
    for k, v in schema.items():
        if k in STRICT_DROP or (k == 'format' and v not in STRICT_FORMATS):
            continue
        if k == 'properties' and isinstance(v, dict):
            props = {}
            for pk, pv in v.items():
                sv, why = strict_schema(pv, f'{path}.{pk}')
                if sv is None:
                    return None, why
                props[pk] = sv
            out[k] = props
        elif k in ('items', 'anyOf', 'allOf', '$defs', 'definitions'):
            if k in ('$defs', 'definitions') and isinstance(v, dict):
                defs = {}
                for dk, dv in v.items():
                    sv, why = strict_schema(dv, f'{path}/{k}/{dk}')
                    if sv is None:
                        return None, why
                    defs[dk] = sv
                out[k] = defs
            else:
                sv, why = strict_schema(v, f'{path}/{k}')
                if sv is None:
                    return None, why
                out[k] = sv
        else:
            out[k] = copy.deepcopy(v)
    is_object = out.get('type') == 'object' or (isinstance(out.get('type'), list) and 'object' in out['type'])
    if is_object:
        if 'properties' not in out:
            return None, f'{path} is a free-form object (a scenario, a settings patch …), which strict mode cannot ' \
                         'express'
        ap = out.get('additionalProperties', False)
        if ap is not False:
            return None, f'{path} allows additional properties, which strict mode does not'
        out['additionalProperties'] = False
    return out, ''


def tool_defs(tools: list) -> tuple:
    """OpenAI-format tool schemas as the Messages API takes them, sorted by name (the same bytes every step, so the
    prompt cache holds), each with eager_input_streaming, and strict when its schema can be: (defs, {name: why not
    strict})."""
    defs, loose = [], {}
    for t in tools or []:
        fn = t.get('function') if isinstance(t, dict) and 'function' in t else t
        name = fn.get('name')
        params = fn.get('parameters') or fn.get('input_schema') or {'type': 'object', 'properties': {}}
        d = {'name': name, 'description': fn.get('description') or '', 'eager_input_streaming': True}
        sv, why = strict_schema(params)
        if sv is not None:
            d['input_schema'] = sv
            d['strict'] = True
        else:
            d['input_schema'] = copy.deepcopy(params)
            loose[name] = why
        defs.append(d)
    defs.sort(key=lambda d: d['name'])
    return defs, loose


# --------------------------------------------------------------------------------------------- the conversation
def echo_blocks(blocks: list) -> list:
    """An assistant turn's content blocks as they go back: unchanged — except after a mid-output fallback, where the
    thinking, redacted_thinking and tool_use blocks before the last fallback block are left out (the API's rule for
    echoing a fallback turn; the fallback block stays where it was)."""
    blocks = [b for b in blocks or [] if isinstance(b, dict)]
    last = max((i for i, b in enumerate(blocks) if b.get('type') == 'fallback'), default=-1)
    if last < 0:
        return blocks
    known = ('text', 'fallback', 'server_tool_result', 'web_search_tool_result')
    return [b for i, b in enumerate(blocks) if i >= last or b.get('type') in known]


def _parsed(arguments: str) -> dict:
    try:
        v = json.loads(arguments or '{}')
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def _delivered(msgs: list, i: int) -> bool:
    """A mid-conversation system note may go: it is the last message, or the next message is the assistant's (a note
    left before a user message was never answered — the API rejects it there — and a note followed by a newer one is
    superseded by it)."""
    for m in msgs[i + 1:]:
        if m.get('_skip'):
            continue
        return m.get('role') == 'assistant'
    return True


def to_native(messages: list, drop_thinking_before: int = 0, cache_system: bool = False) -> tuple:
    """(system, messages) for the Messages API from Falcon's conversation (session messages with their '_' notes):
    the first system message is the top-level system prompt (frozen; with a cache breakpoint when cache_system); a
    later one is a mid-conversation system message; a Claude turn is its stored blocks (echo_blocks); a turn from
    another provider becomes text and tool_use blocks; consecutive tool results become one user message of
    tool_result blocks (is_error for a failed tool). Deterministic: the same conversation gives the same bytes.
    drop_thinking_before: thinking blocks of turns before this index are left out (the one recovery after the API
    said they were bound to another conversation)."""
    system, out = [], []
    msgs = list(messages or [])
    start = 0
    if msgs and msgs[0].get('role') == 'system':
        text = privacy.redact_text(str(msgs[0].get('content') or ''))
        block = {'type': 'text', 'text': text}
        if cache_system:
            block['cache_control'] = {'type': 'ephemeral'}
        system = [block]
        start = 1
    pending: list = []                         # tool_result blocks of the current step

    def flush():
        if pending:
            out.append({'role': 'user', 'content': list(pending)})
            pending.clear()
    for i in range(start, len(msgs)):
        m = msgs[i]
        role = m.get('role')
        if m.get('_skip'):
            continue
        if role == 'tool':
            r = {'type': 'tool_result', 'tool_use_id': str(m.get('tool_call_id') or ''),
                 'content': privacy.redact_text(str(m.get('content') or ''))}
            if m.get('_error'):
                r['is_error'] = True
            pending.append(r)
            continue
        flush()
        if role == 'system':
            if _delivered(msgs, i):
                out.append({'role': 'system', 'content': privacy.redact_text(str(m.get('content') or ''))})
            continue
        if role == 'user':
            out.append({'role': 'user', 'content': privacy.redact_text(str(m.get('content') or ''))})
            continue
        if role != 'assistant':
            continue
        if isinstance(m.get('_anthropic'), list):
            blocks = echo_blocks(m['_anthropic'])
            if i < drop_thinking_before:
                blocks = [b for b in blocks if b.get('type') not in ('thinking', 'redacted_thinking')]
        else:                                  # a turn from another provider: its text and tool calls
            blocks = []
            if m.get('content'):
                blocks.append({'type': 'text', 'text': str(m['content'])})
            for tc in m.get('tool_calls') or []:
                fn = tc.get('function') or {}
                blocks.append({'type': 'tool_use', 'id': str(tc.get('id') or ''), 'name': str(fn.get('name') or ''),
                               'input': _parsed(fn.get('arguments'))})
        if blocks:
            out.append({'role': 'assistant', 'content': blocks})
    flush()
    return system, out


# --------------------------------------------------------------------------------------------- the client
class AnthropicClient:
    native = True                              # sends the conversation as Claude returned it (to_native)

    def __init__(self, key: str = '', base_url: str = DEFAULT_BASE_URL, gate=None, on_request=None,
                 retries: int = 4, backoff_s: float = 1.0, max_wait_s: float = 60.0, connect_timeout_s: float = 15.0,
                 read_timeout_s: float = 600.0, sleep=None, fallback: bool = True, effort: str = 'medium',
                 drop_thinking_before: int = 0):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip('/')
        self._key = key or ''
        self.gate = gate
        self.on_request = on_request
        self.retries, self.backoff_s, self.max_wait_s = int(retries), float(backoff_s), float(max_wait_s)
        self.connect_timeout_s, self.read_timeout_s = connect_timeout_s, read_timeout_s
        self.fallback = bool(fallback)
        self.effort = effort if effort in EFFORTS else 'medium'
        self.drop_thinking_before = int(drop_thinking_before or 0)
        self.scrub = scrubber(self._key)
        self._sleep = sleep
        self.on_retry = None                   # on_retry(status, wait_s, attempt)
        self.on_limits = None                  # (the Claude API's limits are not paced from headers here)
        self.loose_tools: dict = {}            # the last request's tools sent without strict, and why

    # ------------------------------------------------------------------ the SDK
    def _sdk(self):
        try:
            import anthropic
        except ImportError:
            raise ProviderError('protocol', 'The Claude provider needs Anthropic\'s Python package: pip install '
                                '"falcon-autonomy[claude]" (or anthropic>=2).') from None
        timeout = anthropic.Timeout(self.read_timeout_s, connect=self.connect_timeout_s)
        return anthropic.Anthropic(api_key=self._key, base_url=self.base_url, max_retries=0, timeout=timeout,
                                   default_headers={'User-Agent': USER_AGENT})

    def _check_gate(self):
        if self.gate is not None:
            self.gate()
        if not self._key:
            raise ProviderError('no_key', 'No Claude API key: paste yours in Settings ▸ Assistant ▸ Provider.')

    def _host(self) -> str:
        import urllib.parse
        return urllib.parse.urlsplit(self.base_url).netloc or self.base_url

    # ------------------------------------------------------------------ models
    def models(self, cancel: CancelToken | None = None) -> list:
        """The models this key may use, filtered to the two Falcon uses: [{'id', 'context_window', 'active',
        'max_completion_tokens', 'display_name'}]."""
        self._check_gate()
        cancel = cancel or CancelToken()

        def once(attempt):
            sdk = self._sdk()
            try:
                return [m for m in sdk.models.list(limit=100)]
            except Exception as e:             # noqa: BLE001 (mapped to a sentence)
                raise self._error(e, '', False) from None
            finally:
                sdk.close()
        try:
            got = self._with_retries(once, cancel)
        except ProviderError as e:
            if e.kind != 'cancelled':
                self._record('models', '', b'', e.status or e.kind, request_id=e.request_id)
            raise
        self._record('models', '', b'', 200)
        out = []
        for m in got:
            mid = str(getattr(m, 'id', '') or '')
            if mid not in ALLOWED:
                continue
            out.append({'id': mid, 'context_window': getattr(m, 'max_input_tokens', None),
                        'max_completion_tokens': getattr(m, 'max_tokens', None),
                        'display_name': getattr(m, 'display_name', mid),
                        'active': getattr(m, 'lifecycle', 'active') != 'retired', 'owned_by': 'Anthropic'})
        return sorted(out, key=lambda m: m['id'])

    # ------------------------------------------------------------------ chat
    def request(self, messages: list, model: str, tools: list | None = None, tool_choice: str | None = None,
                parallel_tool_calls: bool | None = None, max_completion_tokens: int | None = None,
                extra: dict | None = None) -> dict:
        """The Messages API request chat() streams (without the key): for the audit fingerprint and the tests."""
        if not allowed(model):
            raise ProviderError('model', f'Falcon uses Claude as {MAIN_MODEL} (the main agent) or {FAST_MODEL} (the '
                                f'fast lane) only; {model or "no model"} is not one of them. Choose one in Settings ▸ '
                                'Assistant.')
        if tool_choice not in (None, 'auto', 'none'):
            raise ProviderError('bad_request', 'Claude Opus 5.5 and Sonnet 5.5 do not take a forced tool choice: '
                                'Falcon sends auto.')
        cache = bool(tools)
        system, native = to_native(messages, self.drop_thinking_before, cache_system=cache)
        if not native or native[0].get('role') != 'user':
            raise ProviderError('bad_request', 'Nothing to send: a conversation with Claude starts with a message '
                                'from you.')
        effort = effort_of(str((extra or {}).get('effort') or ''), self.effort)
        floor = QUICK_MIN_TOKENS if not tools else AGENT_MAX_TOKENS
        params = {'model': model, 'max_tokens': max(int(max_completion_tokens or 0), floor), 'messages': native,
                  'output_config': {'effort': effort}}
        if system:
            params['system'] = system
        if tools:
            defs, self.loose_tools = tool_defs(tools)
            params['tools'] = defs
            choice = {'type': 'none' if tool_choice == 'none' else 'auto'}
            if parallel_tool_calls is False:
                choice['disable_parallel_tool_use'] = True
            params['tool_choice'] = choice
        if cache:
            params['cache_control'] = {'type': 'ephemeral'}       # automatic caching of the growing conversation
        if self.fallback:
            params['betas'] = [FALLBACK_BETA]
            params['fallbacks'] = 'default'
        return params

    def chat(self, messages: list, model: str, tools: list | None = None, tool_choice: str | None = None,
             parallel_tool_calls: bool | None = None, stream: bool = True, temperature: float | None = None,
             max_completion_tokens: int | None = None, extra: dict | None = None, on_delta=None,
             cancel: CancelToken | None = None, purpose: str = 'chat') -> ChatResult:
        """One Claude turn, streamed (stream and temperature are accepted for the shared interface: Claude is always
        streamed here, and these models take no temperature)."""
        del stream, temperature
        cancel = cancel or CancelToken()
        params = self.request(messages, model, tools, tool_choice, parallel_tool_calls, max_completion_tokens, extra)
        self._check_gate()
        body = json.dumps({k: v for k, v in params.items() if k != 'betas'}, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':')).encode('utf-8')
        attempts = [0]
        json_fails = [0]

        def once(attempt):
            attempts[0] = attempt + 1
            while True:
                try:
                    return self._stream(params, cancel, on_delta)
                except ProviderError as e:
                    if e.kind != 'tool_json' or json_fails[0] >= JSON_RETRIES:
                        raise
                    json_fails[0] += 1          # unparseable tool input: no tool_use id to answer, so ask again
        try:
            res = self._with_retries(once, cancel)
        except ProviderError as e:
            if e.kind == 'tool_json':
                e = ProviderError('tool_use_failed', 'Claude wrote tool input that could not be read, and asking '
                                  'again did not help. Ask again in other words.', request_id=e.request_id)
            if e.kind != 'cancelled':
                self._record(purpose, model, body, e.status or e.kind, attempts=attempts[0], request_id=e.request_id)
            raise e from None
        res.attempts = attempts[0]
        self._record(purpose, model, body, 200, res.usage, [t.name for t in res.tool_calls], attempts[0],
                     res.request_id, res.stop_reason)
        return res

    # ------------------------------------------------------------------ one streamed attempt
    def _stream(self, params, cancel: CancelToken, on_delta) -> ChatResult:
        """Stream on a worker thread; this thread hands on the deltas and returns at once on Stop (the socket is shut
        down, so the worker's read ends too)."""
        q: queue.Queue = queue.Queue()
        state = {'sock': None, 'abandon': False}

        def work():
            sdk = None
            try:
                sdk = self._sdk()
                raw: dict = {}
                with sdk.beta.messages.stream(**params) as st:
                    try:
                        state['sock'] = st.response.extensions['network_stream'].get_extra_info('socket')
                    except Exception:          # noqa: BLE001 (no socket to shut: Stop still returns at once)
                        state['sock'] = None
                    if state['abandon']:
                        return
                    for ev in st:
                        if state['abandon']:
                            return
                        t = getattr(ev, 'type', '')
                        if t == 'text':
                            q.put(('delta', ev.text))
                        elif t == 'content_block_delta' and getattr(ev.delta, 'type', '') == 'input_json_delta':
                            raw[ev.index] = raw.get(ev.index, '') + (ev.delta.partial_json or '')
                    final = st.get_final_message()
                    q.put(('done', final, raw, st.request_id))
            except BaseException as e:         # noqa: BLE001 (handed to the caller)
                q.put(('error', e))
            finally:
                if sdk is not None:
                    try:
                        sdk.close()
                    except Exception:          # noqa: BLE001
                        pass

        def close():
            state['abandon'] = True
            s = state['sock']
            if s is not None:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        cancel.on_cancel(close)
        th = threading.Thread(target=work, name='falcon-claude-stream', daemon=True)
        th.start()
        streamed = False
        try:
            while True:
                try:
                    item = q.get(timeout=0.1)
                except queue.Empty:
                    if cancel.is_set():
                        raise Cancelled() from None
                    continue
                if cancel.is_set():
                    raise Cancelled()
                if item[0] == 'delta':
                    streamed = True
                    if on_delta is not None and item[1]:
                        on_delta(item[1])
                    continue
                if item[0] == 'error':
                    raise self._error(item[1], params.get('model', ''), streamed) from None
                return self._result(item[1], item[2], item[3], params.get('model', ''))
        finally:
            cancel.off(close)

    def _result(self, final, raw: dict, request_id: str, requested: str) -> ChatResult:
        blocks = [b.to_dict(mode='json') for b in final.content]
        stop = str(final.stop_reason or '')
        res = ChatResult(model=str(getattr(final, 'model', '') or requested), native=blocks, stop_reason=stop,
                         request_id=request_id or getattr(final, '_request_id', '') or '')
        sd = getattr(final, 'stop_details', None)
        res.stop_details = sd.to_dict(mode='json') if sd is not None and hasattr(sd, 'to_dict') else None
        last_fb = max((i for i, b in enumerate(blocks) if b.get('type') == 'fallback'), default=-1)
        res.content = '\n\n'.join(b.get('text') or '' for b in blocks if b.get('type') == 'text' and b.get('text'))
        calls = []
        for i, b in enumerate(blocks):
            if b.get('type') != 'tool_use' or i < last_fb:
                continue                       # before a fallback: the declined model's, not this answer's
            text = raw.get(i, '')
            if not text.strip():
                text = json.dumps(b.get('input') or {}, ensure_ascii=False)
            calls.append(ToolCall(str(b.get('id') or f'toolu_{i}'), str(b.get('name') or ''), text))
        res.finish_reason = {'end_turn': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length',
                             'model_context_window_exceeded': 'length', 'refusal': 'refusal',
                             'pause_turn': 'pause', 'stop_sequence': 'stop'}.get(stop, stop or 'stop')
        res.tool_calls = [] if stop == 'refusal' else calls       # nothing of a refused response runs
        u = final.usage
        res.usage = Usage(int(getattr(u, 'input_tokens', 0) or 0), int(getattr(u, 'output_tokens', 0) or 0),
                          int(getattr(u, 'cache_read_input_tokens', 0) or 0),
                          int(getattr(u, 'cache_creation_input_tokens', 0) or 0))
        its = getattr(u, 'iterations', None) or []
        ran_fallback = False
        for it in its:
            t = getattr(it, 'type', '')
            if t == 'fallback_message':
                ran_fallback = True
            if t in ('message', 'fallback_message'):
                res.iterations.append((str(getattr(it, 'model', '') or res.model),
                                       Usage(int(it.input_tokens or 0), int(it.output_tokens or 0),
                                             int(it.cache_read_input_tokens or 0),
                                             int(it.cache_creation_input_tokens or 0))))
        if last_fb >= 0:
            fb = blocks[last_fb]
            res.fallback = {'from': (fb.get('from') or {}).get('model', requested),
                            'to': (fb.get('to') or {}).get('model', res.model),
                            'category': (fb.get('trigger') or {}).get('category')}
        elif ran_fallback and stop != 'refusal':
            res.fallback = {'from': requested, 'to': res.model, 'category': None}        # a sticky fallback turn
        return res

    # ------------------------------------------------------------------ failures
    def _error(self, e, model: str, streamed: bool) -> ProviderError:
        """An SDK exception as a sentence (the chain: rate limit, overloaded and 5xx, bad request, authentication and
        permission, not found, too large, connection)."""
        if isinstance(e, ProviderError):
            return e
        try:
            import anthropic
        except ImportError:                    # pragma: no cover (the client exists only with the SDK)
            return ProviderError('protocol', f'{type(e).__name__}')
        if isinstance(e, ValueError) and 'tool parameter JSON' in str(e):
            return ProviderError('tool_json', 'Claude\'s tool input could not be parsed.')
        if isinstance(e, anthropic.APIStatusError):
            rid = getattr(e, 'request_id', '') or ''
            body = getattr(e, 'body', None)
            err = body.get('error') if isinstance(body, dict) else None
            msg = str(err.get('message') or '') if isinstance(err, dict) else str(getattr(e, 'message', '') or '')
            msg = self.scrub(privacy.redact_text(msg))[:400]
            etype = getattr(e, 'type', None) or (err.get('type') if isinstance(err, dict) else '') or ''
            status = int(getattr(e, 'status_code', 0) or 0)
            said = f' Anthropic said: {msg}' if msg else ''
            ref = f' (request {rid})' if rid else ''
            if isinstance(e, anthropic.RateLimitError) or etype == 'rate_limit_error':
                ra = None
                try:
                    v = e.response.headers.get('retry-after')
                    ra = float(v) if v is not None else None
                except (ValueError, AttributeError):
                    ra = None
                out = ProviderError('rate', 'Anthropic\'s rate limit for this key was reached and waiting did not '
                                    'clear it. Wait a minute and ask again; your limits are on console.anthropic.com.'
                                    + ref, 429, ra, msg, rid)
            elif isinstance(e, (anthropic.OverloadedError, anthropic.InternalServerError,
                                anthropic.ServiceUnavailableError, anthropic.DeadlineExceededError)) or \
                    etype in ('overloaded_error', 'api_error') or status >= 500:
                code = status if status >= 500 else (529 if etype == 'overloaded_error' else 500)
                out = ProviderError('server', f'The Claude API is overloaded or failing ({code}) and retries did not '
                                    'help. Try again in a few minutes.' + ref, code, None, msg, rid)
            elif isinstance(e, anthropic.AuthenticationError):
                out = ProviderError('auth', 'Anthropic refused the API key (401). Check the Claude key in Settings ▸ '
                                    'Assistant ▸ Provider (console.anthropic.com ▸ API keys).' + ref, 401, None, msg,
                                    rid)
            elif isinstance(e, anthropic.PermissionDeniedError):
                out = ProviderError('forbidden', 'Anthropic refused this request for this key (403): its organisation '
                                    f'or workspace may not have access to {model or "this model"}.' + said + ref, 403,
                                    None, msg, rid)
            elif isinstance(e, anthropic.NotFoundError):
                out = ProviderError('model', f'The model {model or "?"} is not available to this key (404).' + said
                                    + ref, 404, None, msg, rid)
            elif isinstance(e, anthropic.RequestTooLargeError):
                out = ProviderError('context', 'The conversation is too large for the Claude API (413). Clear it '
                                    '(Clear in the Assistant panel) and ask again.' + ref, 413, None, msg, rid)
            elif status == 402 or etype == 'billing_error':
                out = ProviderError('billing', 'Anthropic refused the request for a billing reason (402): check the '
                                    'account\'s plan and credits on console.anthropic.com.' + ref, 402, None, msg, rid)
            else:
                low = msg.lower()
                if 'bound to a different conversation' in low or ('signature' in low and 'thinking' in low):
                    out = ProviderError('binding', 'Claude said earlier reasoning in this conversation no longer '
                                        'matches it.' + said + ref, status, None, msg, rid)
                elif 'prompt is too long' in low or 'context window' in low or 'too long' in low:
                    out = ProviderError('context', f'The conversation is too long for {model or "the model"}\'s '
                                        'context window. Clear it (Clear in the Assistant panel) and ask again.' + ref,
                                        status, None, msg, rid)
                else:
                    out = ProviderError('bad_request', f'The Claude API rejected the request ({status}).' + said + ref,
                                        status, None, msg, rid)
            if streamed and out.kind in ('server', 'rate'):
                out.status = -1                # part of the answer is already shown: not retried
            return out
        if isinstance(e, anthropic.APIConnectionError):
            why = 'timed out' if isinstance(e, anthropic.APITimeoutError) else self.scrub(str(e))[:160]
            out = ProviderError('network', f'Cannot reach {self._host()} ({why}). Is the network up?')
            if streamed:
                out.status = -1
            return out
        return ProviderError('protocol', f'The Claude API call failed ({type(e).__name__}: '
                             f'{self.scrub(privacy.redact_text(str(e)))[:200]}).')

    # ------------------------------------------------------------------ retries, audit
    def _wait(self, seconds: float, cancel: CancelToken):
        if self._sleep is not None:
            self._sleep(seconds)
            if cancel.is_set():
                raise Cancelled()
            return
        if cancel.wait(seconds):
            raise Cancelled()

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_wait_s) * (1.0 + 0.1 * random.random())
        cap = min(self.max_wait_s, self.backoff_s * (2 ** attempt))
        return max(0.25 * self.backoff_s, random.uniform(0, cap))

    def _with_retries(self, fn, cancel: CancelToken):
        attempt = 0
        while True:
            if cancel.is_set():
                raise Cancelled()
            try:
                return fn(attempt)
            except Cancelled:
                raise
            except ProviderError as e:
                retryable = (e.kind in ('rate', 'server') and e.status != -1) or \
                    (e.kind == 'network' and e.status is None)
                if not retryable or attempt >= self.retries:
                    raise
                wait = self._delay(attempt, e.retry_after)
                if self.on_retry is not None:
                    self.on_retry(e.status, wait, attempt + 1)
                self._wait(wait, cancel)
                attempt += 1

    def _record(self, purpose, model, body: bytes, status, usage: Usage | None = None, tools=(), attempts=1,
                request_id: str = '', stop_reason: str = ''):
        if self.on_request is None:
            return
        try:
            self.on_request({'purpose': purpose, 'model': model, 'provider': 'anthropic',
                             'payload_sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body), 'status': status,
                             'attempts': attempts, 'prompt_tokens': usage.prompt if usage else 0,
                             'completion_tokens': usage.completion if usage else 0,
                             'cache_read_tokens': usage.cache_read if usage else 0,
                             'cache_write_tokens': usage.cache_write if usage else 0, 'tools': list(tools),
                             'request_id': request_id, 'stop_reason': stop_reason})
        except Exception:                      # noqa: BLE001 (an audit write failing never stops the answer)
            pass
