"""A fake Groq (OpenAI-compatible) server for the assistant's tests: a local http.server on a free port (never one of
CARLA's or Falcon's), serving GET /models and POST /chat/completions from a script of responses — streamed chunks
(content, tool calls split across chunks, parallel tool calls, usage in x_groq), plain JSON, errors (401, 404, 413, 429
with retry-after, 500), malformed tool-call JSON, and a stream that hangs (for Stop). Every request is recorded
(method, path, headers, body) so a test can count them and read exactly what was sent.

    srv = FakeGroq().start()
    srv.push(text('Hello'))                 # the next /chat/completions answer
    client = GroqClient(srv.base_url, FAKE_KEY)
    ...
    srv.stop()
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RESERVED = set(range(2000, 2011)) | {8100, 2310, 2311, 2312, 8310}
FAKE_KEY = 'gsk_FAKEFAKEFAKEtestkey0000'          # obviously fake: the prefix the redaction tests need, nothing more

MODELS = [{'id': 'openai/gpt-oss-120b', 'object': 'model', 'owned_by': 'OpenAI', 'active': True,
           'context_window': 131072, 'max_completion_tokens': 65536},
          {'id': 'openai/gpt-oss-20b', 'object': 'model', 'owned_by': 'OpenAI', 'active': True,
           'context_window': 131072, 'max_completion_tokens': 65536},
          {'id': 'qwen/qwen3.8-27b', 'object': 'model', 'owned_by': 'Alibaba Cloud', 'active': True,
           'context_window': 131072, 'max_completion_tokens': 16384}]


def _chunk(delta=None, finish=None, usage=None, model='openai/gpt-oss-120b'):
    c = {'id': 'chatcmpl-fake', 'object': 'chat.completion.chunk', 'model': model,
         'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}]}
    if usage:
        c['x_groq'] = {'usage': usage}
    return c


def text(content: str, pieces: int = 3, prompt_tokens: int = 120, completion_tokens: int = 30) -> dict:
    """A streamed answer: the content in `pieces` deltas, then the finish and the usage."""
    n = max(1, len(content) // pieces)
    parts = [content[i:i + n] for i in range(0, len(content), n)] or ['']
    chunks = [_chunk({'role': 'assistant', 'content': ''})] + [_chunk({'content': p}) for p in parts]
    chunks.append(_chunk({}, 'stop', {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                                      'total_tokens': prompt_tokens + completion_tokens}))
    return {'stream': chunks}


def tools(*calls, split: int = 3, content: str = '', prompt_tokens: int = 200, completion_tokens: int = 20) -> dict:
    """A streamed answer calling tools: calls are (name, arguments as a string or a dict); each call's arguments
    arrive in `split` pieces, and the calls' pieces interleave (parallel tool calls, assembled by index)."""
    chunks = [_chunk({'role': 'assistant', 'content': content or None})]
    pieces = []
    for i, (name, args) in enumerate(calls):
        a = args if isinstance(args, str) else json.dumps(args)
        n = max(1, -(-len(a) // split))
        parts = [a[k:k + n] for k in range(0, len(a), n)] or ['']
        pieces.append([{'index': i, 'id': f'call_{i}_{name}', 'type': 'function',
                        'function': {'name': name, 'arguments': parts[0]}}] +
                      [{'index': i, 'function': {'arguments': p}} for p in parts[1:]])
    while any(pieces):
        for p in pieces:
            if p:
                chunks.append(_chunk({'tool_calls': [p.pop(0)]}))
    chunks.append(_chunk({}, 'tool_calls', {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                                            'total_tokens': prompt_tokens + completion_tokens}))
    return {'stream': chunks}


def error(status: int, message: str, code: str = '', headers: dict | None = None) -> dict:
    return {'status': status, 'json': {'error': {'message': message, 'type': 'invalid_request_error',
                                                 **({'code': code} if code else {})}}, 'headers': headers or {}}


def hang(first: str = 'Thinking', seconds: float = 20.0) -> dict:
    """A stream that sends one delta and then nothing (until the client hangs up or `seconds` pass)."""
    return {'stream': [_chunk({'role': 'assistant', 'content': ''}), _chunk({'content': first})], 'hang': seconds}


class FakeGroq:
    def __init__(self):
        self.script: list = []
        self.requests: list = []
        self.models = list(MODELS)
        self.lock = threading.Lock()
        self.httpd = None
        self.thread = None
        self.released = threading.Event()
        self.hung = threading.Event()

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}/openai/v1'

    def push(self, *responses):
        with self.lock:
            self.script.extend(responses)
        return self

    def chats(self) -> list:
        return [r for r in self.requests if r['path'].endswith('/chat/completions')]

    def start(self):
        srv = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def _record(self, body=None):
                with srv.lock:
                    srv.requests.append({'method': self.command, 'path': self.path, 'headers': dict(self.headers),
                                         'body': body, 'raw': getattr(self, '_raw', b'')})

            def _send_json(self, status, doc, headers=None):
                data = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):                                   # noqa: N802
                self._record()
                if self.path.endswith('/models'):
                    if self.headers.get('Authorization', '') != f'Bearer {FAKE_KEY}':
                        return self._send_json(401, {'error': {'message': 'Invalid API Key',
                                                               'type': 'invalid_request_error',
                                                               'code': 'invalid_api_key'}})
                    return self._send_json(200, {'object': 'list', 'data': srv.models})
                self._send_json(404, {'error': {'message': 'not found'}})

            def do_POST(self):                                  # noqa: N802
                n = int(self.headers.get('Content-Length') or 0)
                self._raw = self.rfile.read(n)
                try:
                    body = json.loads(self._raw.decode('utf-8'))
                except ValueError:
                    body = None
                self._record(body)
                with srv.lock:
                    resp = srv.script.pop(0) if srv.script else error(500, 'the fake server has no scripted answer')
                if 'stream' not in resp:
                    return self._send_json(resp.get('status', 200), resp.get('json', {}), resp.get('headers'))
                self.send_response(resp.get('status', 200))
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                try:
                    for c in resp['stream']:
                        self._chunked(f'data: {json.dumps(c)}\n\n'.encode())
                        time.sleep(resp.get('delay', 0.0))
                    if resp.get('hang'):
                        srv.hung.set()
                        srv.released.wait(resp['hang'])
                    self._chunked(b'data: [DONE]\n\n')
                    self._chunked(b'')
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

            def _chunked(self, data: bytes):
                self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        assert self.port not in RESERVED
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.released.set()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
