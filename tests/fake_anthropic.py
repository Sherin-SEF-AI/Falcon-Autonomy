"""A fake Claude API for the assistant's tests: a local http.server on a free port speaking the Messages API as the
official SDK reads it — GET /v1/models (paged list), POST /v1/messages streamed as server-sent events (message_start,
content_block_start/delta/stop for text, thinking with its signature, redacted thinking, tool_use with input_json_delta
pieces, fallback blocks; message_delta with the stop reason, stop_details and usage with iterations; message_stop),
errors as the API sends them ({"type": "error", "error": {type, message}} with a request-id header, retry-after on a
429), an error event mid-stream, and a stream that hangs (for Stop). Nothing leaves the machine.

It also models the prompt cache the way the API keys it — a prefix of the exact bytes of tools and system up to the
breakpoint: a request whose tools and system are byte-identical to an earlier one's reads that prefix from the cache
(cache_read_input_tokens), else writes it (cache_creation_input_tokens). Every request is recorded (method, path,
headers, the raw body and the parsed one), so a test reads exactly what was sent.

    srv = FakeAnthropic().start()
    srv.push(reply([text('Hello')]))
    client = AnthropicClient(FAKE_KEY, srv.base_url)
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RESERVED = set(range(2000, 2011)) | {8100, 2310, 2311, 2312, 8310}
FAKE_KEY = 'sk-ant-FAKEFAKEFAKE-testkey-0000000000'        # obviously fake; shaped like a key for the redaction tests

MODELS = [{'type': 'model', 'id': 'claude-opus-5-5', 'display_name': 'Claude Opus 5.5',
           'created_at': '2026-09-01T00:00:00Z', 'max_input_tokens': 1000000, 'max_tokens': 128000,
           'lifecycle': 'active'},
          {'type': 'model', 'id': 'claude-sonnet-5-5', 'display_name': 'Claude Sonnet 5.5',
           'created_at': '2026-09-01T00:00:00Z', 'max_input_tokens': 1000000, 'max_tokens': 128000,
           'lifecycle': 'active'},
          {'type': 'model', 'id': 'claude-opus-5', 'display_name': 'Claude Opus 5',
           'created_at': '2026-05-01T00:00:00Z',
           'max_input_tokens': 1000000, 'max_tokens': 128000, 'lifecycle': 'active'},
          {'type': 'model', 'id': 'claude-haiku-5-5', 'display_name': 'Claude Haiku 5.5',
           'created_at': '2026-09-01T00:00:00Z', 'max_input_tokens': 200000, 'max_tokens': 64000,
           'lifecycle': 'active'}]


# --------------------------------------------------------------------------------------------- blocks of a reply
def text(t: str) -> dict:
    return {'type': 'text', 'text': t}


def thinking(t: str = '', signature: str = 'sig-FAKE-abc123==') -> dict:
    return {'type': 'thinking', 'thinking': t, 'signature': signature}


def redacted(data: str = 'ENCRYPTED-FAKE') -> dict:
    return {'type': 'redacted_thinking', 'data': data}


def tool_use(name: str, args, ident: str = '') -> dict:
    """A tool_use block; args a dict (sent as its JSON in pieces) or a raw string (sent as it is: a malformed input)."""
    raw = args if isinstance(args, str) else json.dumps(args)
    return {'type': 'tool_use', 'id': ident or f'toolu_{name}', 'name': name, '_raw': raw}


def fallback(frm: str = 'claude-opus-5-5', to: str = 'claude-opus-5', category: str = 'cyber') -> dict:
    return {'type': 'fallback', 'from': {'model': frm}, 'to': {'model': to},
            'trigger': {'type': 'refusal', 'category': category}}


def reply(blocks: list, stop: str = 'end_turn', output_tokens: int = 30, model: str = '', stop_details=None,
          iterations=None, pieces: int = 3, usage: dict | None = None, error_after: int | None = None,
          hang: float = 0.0) -> dict:
    """One streamed response. error_after: send an overloaded error event after that many blocks; hang: stop after the
    first block and wait (for Stop)."""
    return {'blocks': blocks, 'stop': stop, 'output_tokens': output_tokens, 'model': model,
            'stop_details': stop_details, 'iterations': iterations, 'pieces': pieces, 'usage': usage,
            'error_after': error_after, 'hang': hang}


def error(status: int, etype: str, message: str, headers: dict | None = None) -> dict:
    return {'status': status, 'json': {'type': 'error', 'error': {'type': etype, 'message': message}},
            'headers': headers or {}}


def _split(s: str, n: int) -> list:
    k = max(1, -(-len(s) // max(1, n)))
    return [s[i:i + k] for i in range(0, len(s), k)] or ['']


class FakeAnthropic:
    def __init__(self):
        self.script: list = []
        self.requests: list = []
        self.models = list(MODELS)
        self.lock = threading.Lock()
        self.cached: set = set()             # the cached prefixes (bytes of tools + system)
        self.httpd = None
        self.released = threading.Event()
        self.hung = threading.Event()
        self._n = 0

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}'

    def push(self, *responses):
        with self.lock:
            self.script.extend(responses)
        return self

    def messages(self) -> list:
        return [r for r in self.requests if r['path'].split('?')[0] == '/v1/messages']

    def bodies(self) -> list:
        return [r['body'] for r in self.messages()]

    # ------------------------------------------------------------------ the cache model
    def _cache(self, body: dict) -> tuple:
        """(read, written) tokens: the bytes of tools then system (as sent), a hit when seen before."""
        prefix = json.dumps(body.get('tools'), separators=(',', ':')) + json.dumps(body.get('system'),
                                                                                   separators=(',', ':'))
        n = len(prefix) // 4
        has_marker = any(isinstance(b, dict) and b.get('cache_control') for b in body.get('system') or []) or \
            bool(body.get('cache_control'))
        if not has_marker or n < 128:
            return 0, 0
        with self.lock:
            hit = prefix in self.cached
            self.cached.add(prefix)
        return (n, 0) if hit else (0, n)

    def start(self):
        srv = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def _rid(self) -> str:
                with srv.lock:
                    srv._n += 1
                    return f'req_fake_{srv._n:04d}'

            def _record(self, body=None, raw=b''):
                with srv.lock:
                    srv.requests.append({'method': self.command, 'path': self.path, 'headers': dict(self.headers),
                                         'body': body, 'raw': raw})

            def _json(self, status, doc, headers=None, rid=''):
                data = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('request-id', rid or self._rid())
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _authorised(self) -> bool:
                return self.headers.get('Authorization', '') == f'Bearer {FAKE_KEY}' or \
                    self.headers.get('x-api-key', '') == FAKE_KEY

            def do_GET(self):                                   # noqa: N802
                self._record()
                if not self._authorised():
                    return self._json(401, {'type': 'error', 'error': {'type': 'authentication_error',
                                                                       'message': 'invalid x-api-key'}})
                if self.path.split('?')[0] == '/v1/models':
                    return self._json(200, {'data': srv.models, 'has_more': False,
                                            'first_id': srv.models[0]['id'], 'last_id': srv.models[-1]['id']})
                self._json(404, {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'not found'}})

            def do_POST(self):                                  # noqa: N802
                n = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(n)
                try:
                    body = json.loads(raw.decode('utf-8'))
                except ValueError:
                    body = None
                self._record(body, raw)
                if not self._authorised():
                    return self._json(401, {'type': 'error', 'error': {'type': 'authentication_error',
                                                                       'message': 'invalid x-api-key'}})
                n_strict = sum(1 for t in (body or {}).get('tools') or [] if isinstance(t, dict) and t.get('strict'))
                if n_strict > 20:                               # the real API's limit (measured against it)
                    return self._json(400, {'type': 'error', 'error': {
                        'type': 'invalid_request_error',
                        'message': f'Too many strict tools ({n_strict}). The maximum number of strict tools supported '
                                   'is 20. Try reducing the number of tools marked as strict.'}})
                with srv.lock:
                    resp = srv.script.pop(0) if srv.script else error(500, 'api_error', 'no scripted answer')
                if 'blocks' not in resp:
                    return self._json(resp['status'], resp['json'], resp.get('headers'))
                self._stream(body or {}, resp)

            def _event(self, name, doc):
                data = f'event: {name}\ndata: {json.dumps(doc)}\n\n'.encode()
                self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                self.wfile.flush()

            def _stream(self, body, r):
                read, wrote = srv._cache(body)
                inp = len(json.dumps(body.get('messages'))) // 4
                model = r['model'] or body.get('model', '')
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Transfer-Encoding', 'chunked')
                self.send_header('request-id', self._rid())
                self.end_headers()
                u0 = {'input_tokens': inp, 'output_tokens': 1, 'cache_read_input_tokens': read,
                      'cache_creation_input_tokens': wrote}
                if r['usage']:
                    u0.update(r['usage'])
                try:
                    self._event('message_start', {'type': 'message_start', 'message': {
                        'id': 'msg_fake', 'type': 'message', 'role': 'assistant', 'model': model, 'content': [],
                        'stop_reason': None, 'stop_sequence': None, 'usage': u0}})
                    for i, b in enumerate(r['blocks']):
                        if r['error_after'] is not None and i == r['error_after']:
                            self._event('error', {'type': 'error', 'error': {'type': 'overloaded_error',
                                                                             'message': 'Overloaded'}})
                            self.wfile.write(b'0\r\n\r\n')
                            return
                        self._block(i, b, r['pieces'])
                        if r['hang'] and i == 0:
                            srv.hung.set()
                            srv.released.wait(r['hang'])
                    delta = {'stop_reason': r['stop'], 'stop_sequence': None}
                    if r['stop_details'] is not None:
                        delta['stop_details'] = r['stop_details']
                    ud = {'output_tokens': r['output_tokens']}
                    if r['iterations'] is not None:
                        ud['iterations'] = r['iterations']
                    self._event('message_delta', {'type': 'message_delta', 'delta': delta, 'usage': ud})
                    self._event('message_stop', {'type': 'message_stop'})
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

            def _block(self, i, b, pieces):
                t = b['type']
                if t == 'text':
                    self._event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                        'content_block': {'type': 'text', 'text': ''}})
                    for p in _split(b['text'], pieces):
                        self._event('content_block_delta', {'type': 'content_block_delta', 'index': i,
                                                            'delta': {'type': 'text_delta', 'text': p}})
                elif t == 'thinking':
                    self._event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                        'content_block': {'type': 'thinking', 'thinking': '',
                                                                          'signature': ''}})
                    if b['thinking']:
                        self._event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                            'type': 'thinking_delta', 'thinking': b['thinking']}})
                    self._event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                        'type': 'signature_delta', 'signature': b['signature']}})
                elif t == 'tool_use':
                    self._event('content_block_start', {'type': 'content_block_start', 'index': i, 'content_block': {
                        'type': 'tool_use', 'id': b['id'], 'name': b['name'], 'input': {}}})
                    for p in _split(b['_raw'], pieces):
                        self._event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                            'type': 'input_json_delta', 'partial_json': p}})
                else:                                            # fallback, redacted_thinking: whole at the start
                    self._event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                        'content_block': {k: v for k, v in b.items()}})
                self._event('content_block_stop', {'type': 'content_block_stop', 'index': i})
                time.sleep(0.002)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        assert self.port not in RESERVED
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.released.set()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
