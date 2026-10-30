"""An OpenAI-compatible chat-completions client on the standard library (http.client) — Groq by default
(https://api.groq.com/openai/v1), any compatible server by its base URL. No SDK in the GUI process.

GroqClient(base_url, key, gate, on_request).chat(messages, model, tools=…, stream=True, on_delta=…, cancel=…):
    every payload passes privacy.redact before it leaves the machine (home paths to ~, anything shaped like a
    credential to [secret]); gate() runs first and raises when the user has not enabled the assistant and accepted
    its notice (nothing is sent before); on_request(record) gets what the audit log keeps (model, the payload's
    SHA-256, token counts — never the payload).
    Streaming reads server-sent events: content deltas, tool-call deltas assembled by index (the id and name arrive
    once, the arguments in pieces), the finish reason, usage (a final chunk's usage, or Groq's x_groq.usage).
    Retries: 429, 498 and 5xx, and a connection that fails before anything streamed, up to `retries` times with
    exponential backoff and full jitter — or exactly the server's retry-after (plus up to 10 % jitter) when it gives
    one. A 400 naming an optional parameter (reasoning_effort, stream_options, …) is retried once without it.
    cancel (provider.CancelToken) closes the socket in flight: a blocked read returns at once.
models(): GET /models — the live list (id, context window, whether active).
Failures are provider.ProviderError with a sentence for the user; the key never appears in one.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import random
import socket
import ssl
import urllib.parse

from . import privacy
from .provider import (RETRY_STATUS, Cancelled, CancelToken, ChatResult, ProviderError, ToolCall, Usage, classify,
                       rejected_params, scrubber)

DEFAULT_BASE_URL = 'https://api.groq.com/openai/v1'
USER_AGENT = 'FalconAutonomy-assistant'


class GroqClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, key: str = '', gate=None, on_request=None,
                 retries: int = 4, backoff_s: float = 1.0, max_wait_s: float = 60.0, connect_timeout_s: float = 15.0,
                 read_timeout_s: float = 180.0, sleep=None):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip('/')
        self._key = key or ''
        self.gate = gate
        self.on_request = on_request
        self.retries, self.backoff_s, self.max_wait_s = int(retries), float(backoff_s), float(max_wait_s)
        self.connect_timeout_s, self.read_timeout_s = connect_timeout_s, read_timeout_s
        self.scrub = scrubber(self._key)
        self._sleep = sleep                    # tests: record the waits instead of sleeping
        self.on_retry = None                   # on_retry(status, wait_s, attempt): the UI says why it waits

    # ------------------------------------------------------------------ public
    def models(self, cancel: CancelToken | None = None) -> list:
        """The server's models: [{'id', 'context_window', 'active', 'owned_by'}], sorted by id."""
        doc = self._json('GET', '/models', None, cancel or CancelToken(), model='')
        out = []
        for m in (doc.get('data') or []) if isinstance(doc, dict) else []:
            if isinstance(m, dict) and m.get('id'):
                out.append({'id': str(m['id']), 'context_window': m.get('context_window'),
                            'active': m.get('active', True), 'owned_by': m.get('owned_by', ''),
                            'max_completion_tokens': m.get('max_completion_tokens')})
        return sorted(out, key=lambda m: m['id'])

    def chat(self, messages: list, model: str, tools: list | None = None, tool_choice: str | None = None,
             parallel_tool_calls: bool | None = None, stream: bool = True, temperature: float | None = None,
             max_completion_tokens: int | None = None, extra: dict | None = None, on_delta=None,
             cancel: CancelToken | None = None, purpose: str = 'chat') -> ChatResult:
        cancel = cancel or CancelToken()
        payload = {'model': model, 'messages': messages}
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = tool_choice or 'auto'
            if parallel_tool_calls is not None:
                payload['parallel_tool_calls'] = bool(parallel_tool_calls)
        if temperature is not None:
            payload['temperature'] = float(temperature)
        if max_completion_tokens:
            payload['max_completion_tokens'] = int(max_completion_tokens)
        if stream:
            payload['stream'] = True
            payload['stream_options'] = {'include_usage': True}
        payload.update(extra or {})
        stripped = False
        while True:
            try:
                return self._chat_once(payload, cancel, on_delta, purpose)
            except ProviderError as e:
                drop = rejected_params(e.detail or e.message, payload) if e.status == 400 and not stripped else []
                if not drop:
                    raise
                stripped = True                 # a server that does not know an optional parameter: once without
                for p in drop:
                    payload.pop(p, None)

    # ------------------------------------------------------------------ internals
    def _check_gate(self):
        if self.gate is not None:
            self.gate()                        # raises ProviderError('consent' | 'no_key')
        if not self._key:
            raise ProviderError('no_key', 'No API key: paste yours in Settings ▸ Assistant.')

    def _body(self, payload) -> bytes:
        return json.dumps(privacy.redact(payload), ensure_ascii=False, separators=(',', ':')).encode('utf-8')

    def _record(self, purpose, model, body, status, usage: Usage | None = None, tools=(), attempts=1):
        if self.on_request is None:
            return
        try:
            self.on_request({'purpose': purpose, 'model': model, 'payload_sha256': hashlib.sha256(body).hexdigest(),
                             'bytes': len(body), 'status': status, 'attempts': attempts,
                             'prompt_tokens': usage.prompt if usage else 0,
                             'completion_tokens': usage.completion if usage else 0, 'tools': list(tools)})
        except Exception:                      # noqa: BLE001 (an audit write failing never stops the answer)
            pass

    def _connect(self, cancel: CancelToken):
        u = urllib.parse.urlsplit(self.base_url)
        if u.scheme not in ('http', 'https') or not u.hostname:
            raise ProviderError('network', f'The server URL {self.base_url!r} is not an http(s) address.')
        port = u.port or (443 if u.scheme == 'https' else 80)
        if u.scheme == 'https':
            conn = http.client.HTTPSConnection(u.hostname, port, timeout=self.connect_timeout_s,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(u.hostname, port, timeout=self.connect_timeout_s)
        return conn, u.path.rstrip('/')

    def _open(self, method, path, body, cancel: CancelToken, accept='application/json'):
        """Send one request; returns (conn, response) with the status read. Raises ProviderError('network')."""
        conn, prefix = self._connect(cancel)

        def close():                           # from the thread that stops: wake the reader, it closes
            try:
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        cancel.on_cancel(close)
        headers = {'Authorization': f'Bearer {self._key}', 'Accept': accept, 'User-Agent': USER_AGENT}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        try:
            conn.request(method, prefix + path, body=body, headers=headers)
            if conn.sock is not None:
                conn.sock.settimeout(self.read_timeout_s)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException, ValueError, AttributeError) as e:
            cancel.off(close)
            conn.close()
            if cancel.is_set():
                raise Cancelled() from None
            raise ProviderError('network', f'Cannot reach {self._host()} ({self._why(e)}). Is the network up?') \
                from None
        return conn, resp, close

    def _host(self) -> str:
        return urllib.parse.urlsplit(self.base_url).netloc or self.base_url

    def _why(self, e) -> str:
        if isinstance(e, socket.timeout):
            return 'timed out'
        if isinstance(e, socket.gaierror):
            return 'the name does not resolve'
        if isinstance(e, ConnectionRefusedError):
            return 'connection refused'
        return self.scrub(type(e).__name__ + (f': {e}' if str(e) else ''))[:160]

    def _error_of(self, resp, body: bytes, model: str) -> ProviderError:
        msg, code = '', ''
        try:
            doc = json.loads(body.decode('utf-8', 'replace'))
            err = doc.get('error') if isinstance(doc, dict) else None
            if isinstance(err, dict):
                msg, code = str(err.get('message') or ''), str(err.get('code') or err.get('type') or '')
            elif isinstance(err, str):
                msg = err
        except ValueError:
            msg = body.decode('utf-8', 'replace')[:300]
        msg = self.scrub(privacy.redact_text(msg))
        e = classify(resp.status, msg, model, code)
        e.detail = msg + (f' [{code}]' if code else '')
        ra = resp.getheader('retry-after')
        try:
            e.retry_after = float(ra) if ra is not None else None
        except ValueError:
            e.retry_after = None
        return e

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
        return max(0.25 * self.backoff_s, random.uniform(0, cap))      # full jitter

    def _with_retries(self, fn, cancel: CancelToken):
        """fn(attempt) -> result; retried on the statuses and failures that clear with time."""
        attempt = 0
        while True:
            if cancel.is_set():
                raise Cancelled()
            try:
                return fn(attempt)
            except Cancelled:
                raise
            except ProviderError as e:
                retryable = e.status in RETRY_STATUS or (e.kind == 'network' and e.status is None)
                if e.kind in ('too_large',) or not retryable or attempt >= self.retries:
                    raise
                wait = self._delay(attempt, e.retry_after)
                if self.on_retry is not None:
                    self.on_retry(e.status, wait, attempt + 1)
                self._wait(wait, cancel)
                attempt += 1

    def _json(self, method, path, payload, cancel: CancelToken, model: str, purpose: str = 'models'):
        self._check_gate()
        body = self._body(payload) if payload is not None else None
        try:
            out = self._json_retried(method, path, body, cancel, model)
        except ProviderError as e:
            if e.kind != 'cancelled':
                self._record(purpose, model, body or b'', e.status or e.kind)
            raise
        self._record(purpose, model, body or b'', 200)
        return out

    def _json_retried(self, method, path, body, cancel: CancelToken, model: str):

        def once(attempt):
            conn, resp, close = self._open(method, path, body, cancel)
            try:
                raw = resp.read()
            except (OSError, http.client.HTTPException, ValueError, AttributeError) as e:
                if cancel.is_set():
                    raise Cancelled() from None
                raise ProviderError('network', f'The connection to {self._host()} broke ({self._why(e)}).') \
                    from None
            finally:
                cancel.off(close)
                conn.close()
            if resp.status >= 400:
                raise self._error_of(resp, raw, model)
            try:
                return json.loads(raw.decode('utf-8'))
            except ValueError:
                raise ProviderError('protocol', f'{self._host()} answered with something that is not JSON — is the '
                                    'server URL an OpenAI-compatible API (…/v1)?') from None
        return self._with_retries(once, cancel)

    def _chat_once(self, payload, cancel, on_delta, purpose) -> ChatResult:
        self._check_gate()
        body = self._body(payload)
        model = payload.get('model', '')
        attempts = [0]

        def once(attempt):
            attempts[0] = attempt + 1
            if not payload.get('stream'):
                conn, resp, close = self._open('POST', '/chat/completions', body, cancel)
                try:
                    raw = resp.read()
                except (OSError, http.client.HTTPException, ValueError, AttributeError) as e:
                    if cancel.is_set():
                        raise Cancelled() from None
                    raise ProviderError('network', f'The connection to {self._host()} broke ({self._why(e)}).') \
                        from None
                finally:
                    cancel.off(close)
                    conn.close()
                if resp.status >= 400:
                    raise self._error_of(resp, raw, model)
                return self._parse_full(raw, model)
            return self._stream(body, model, cancel, on_delta)
        try:
            res = self._with_retries(once, cancel)
        except ProviderError as e:
            if e.kind != 'cancelled':
                self._record(purpose, model, body, e.status or e.kind, attempts=attempts[0])
            raise
        res.attempts = attempts[0]
        self._record(purpose, model, body, 200, res.usage, [t.name for t in res.tool_calls], attempts[0])
        return res

    def _parse_full(self, raw: bytes, model: str) -> ChatResult:
        try:
            doc = json.loads(raw.decode('utf-8'))
            ch = (doc.get('choices') or [{}])[0]
            msg = ch.get('message') or {}
        except (ValueError, AttributeError, IndexError):
            raise ProviderError('protocol', f'{self._host()} answered with something that is not a chat '
                                'completion.') from None
        calls = [ToolCall(tc.get('id') or f'call_{i}', (tc.get('function') or {}).get('name') or '',
                          (tc.get('function') or {}).get('arguments') or '')
                 for i, tc in enumerate(msg.get('tool_calls') or [])]
        return ChatResult(content=msg.get('content') or '', reasoning=msg.get('reasoning') or '', tool_calls=calls,
                          finish_reason=ch.get('finish_reason') or '',
                          usage=Usage.of(doc.get('usage') or (doc.get('x_groq') or {}).get('usage')),
                          model=doc.get('model') or model)

    def _stream(self, body, model, cancel: CancelToken, on_delta) -> ChatResult:
        conn, resp, close = self._open('POST', '/chat/completions', body, cancel, accept='text/event-stream')
        res = ChatResult(model=model)
        calls: dict = {}
        streamed = False
        try:
            if resp.status >= 400:
                raise self._error_of(resp, resp.read(), model)
            while True:
                try:
                    line = resp.readline()
                except (OSError, http.client.HTTPException, ValueError, AttributeError) as e:
                    if cancel.is_set():
                        raise Cancelled() from None
                    err = ProviderError('network', f'The connection to {self._host()} broke mid-answer '
                                        f'({self._why(e)}).')
                    if streamed:
                        err.status = -1                       # not retried: part of the answer is already shown
                    raise err from None
                if cancel.is_set():
                    raise Cancelled()
                if not line:
                    break
                line = line.decode('utf-8', 'replace').strip()
                if not line or line.startswith(':') or not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk.get('error'), dict):
                    err = chunk['error']
                    msg = self.scrub(privacy.redact_text(str(err.get('message') or '')))
                    e = classify(400 if err.get('code') == 'tool_use_failed' else 500, msg, model,
                                 str(err.get('code') or ''))
                    e.detail = msg
                    if streamed:
                        e.status = -1
                    raise e
                u = chunk.get('usage') or (chunk.get('x_groq') or {}).get('usage')
                if u:
                    res.usage = Usage.of(u)
                for ch in chunk.get('choices') or []:
                    d = ch.get('delta') or {}
                    if d.get('content'):
                        res.content += d['content']
                        streamed = True
                        if on_delta is not None:
                            on_delta(d['content'])
                    if d.get('reasoning'):
                        res.reasoning += d['reasoning']
                    for tc in d.get('tool_calls') or []:
                        i = tc.get('index', len(calls))
                        c = calls.setdefault(i, ToolCall('', ''))
                        if tc.get('id'):
                            c.id = tc['id']
                        fn = tc.get('function') or {}
                        if fn.get('name'):
                            c.name = fn['name']
                        if fn.get('arguments'):
                            c.arguments += fn['arguments']
                        streamed = True
                    if ch.get('finish_reason'):
                        res.finish_reason = ch['finish_reason']
        finally:
            cancel.off(close)
            conn.close()
        res.tool_calls = [calls[i] for i in sorted(calls)]
        for k, c in enumerate(res.tool_calls):
            c.id = c.id or f'call_{k}'
        return res
