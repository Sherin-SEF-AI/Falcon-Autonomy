"""Falcon's MCP server: an MCP client (Claude Code, Claude Desktop, any other) uses Falcon through the assistant's
tools. Ported from Vantage mcp/server.py (GPL).

    falcon --mcp [--mcp-project FILE.falcon.json] [--mcp-mode auto|window|file]
    python3 -m falcon_autonomy.mcp …

It speaks MCP over stdio (JSON-RPC 2.0, one message per line) with the Python standard library only; nothing but
protocol messages reaches stdout. Each call goes to the Falcon window that is open (through its local bridge, token
protected: the change shows there as one undo step "Claude · <tool>"), or with no window to the project file (one
undoable step per act call, saved, the file as it was kept as .bak). Which window: the one that started this server
(FALCON_MCP_PID), else the one with --mcp-project open, else the newest.

What it offers (MCP 2024-11-05 to 2025-11-25):
- tools: the assistant's read and act tools (mcp.service: trash tools and the never-list refused), annotated
  (read-only, destructive, idempotent), each result whole as text and as structuredContent;
- resources (falcon://status, overview, runs, queue) and a template (falcon://run/{run}) (mcp.catalog);
- cancellation (notifications/cancelled): a call still waiting its turn never runs, a running one is dropped (its
  answer is not sent; in the window the request to cancel reaches the bridge);
- progress (notifications/progress every PROGRESS_S seconds while a call runs, when the client gives a token);
- logging (notifications/message, once the client sets a level with logging/setLevel).

Calls run on worker threads in the order they arrive: reads side by side, a call that changes something waits for
every call before it and holds back the ones after it — a client that sends an edit and then a read gets the read
after the edit.

Register it with Claude Code once (docs/mcp.md):

    claude mcp add --scope user falcon -- python3 -m falcon_autonomy --mcp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

from . import catalog as CT
from . import service as SV
from .session import FileSession, LinkError, LiveLink, same_file, windows

VERSIONS = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')
LEVELS = ('debug', 'info', 'notice', 'warning', 'error', 'critical', 'alert', 'emergency')
PROGRESS_S = 2.0
INSTRUCTIONS = (' Resources give the project by URI (falcon://overview, falcon://runs, falcon://run/<run id>, '
                'falcon://queue, falcon://status). falcon_status says where the calls go; undo_last_change takes this '
                'client\'s last step back.')


class Router:
    """Which session a call goes to: the open window (the one that started this server, or the one with the project
    open, or the newest), else the file."""

    def __init__(self, project=None, mode='auto', fake_carla=''):
        self.mode = mode
        self.project = os.path.abspath(os.path.expanduser(project)) if project else ''
        self.file = FileSession(fake_carla)
        self.link = None
        self.sticky_file = mode == 'file'
        self._tried = False
        self._lock = threading.RLock()

    def session(self):
        with self._lock:
            if not self.sticky_file:
                ws = windows()
                if ws:
                    here = os.environ.get('FALCON_MCP_PID', '')
                    pick = next((w for w in ws if here and str(w.get('pid')) == here), None) or \
                        next((w for w in ws if self.project and same_file(w.get('path', ''), self.project)), None) or \
                        next((w for w in ws if self.link is not None and int(w.get('pid', 0)) == self.link.pid),
                             None) or ws[0]
                    if self.link is None or self.link.pid != int(pick.get('pid', 0)):
                        if self.link is not None:
                            self.link.close()
                        self.link = LiveLink(pick)
                    else:
                        self.link.info = pick
                    return self.link
                if self.mode == 'window':
                    raise LinkError('no Falcon window is open that lets MCP clients in (Settings ▸ Assistant ▸ '
                                    'Provider ▸ MCP)')
            if self.file.ctx is None and self.project and not self._tried:
                self._tried = True
                r = self.file.open(self.project)
                if 'error' in r:
                    print(f'falcon mcp: {r["error"]}', file=sys.stderr)
            return self.file

    def open(self, args) -> dict:
        with self._lock:
            path = (args or {}).get('path')
            if not path:
                self.sticky_file = self.mode == 'file'
                return {'result': 'calls go to the Falcon window again' if windows() and not self.sticky_file else
                        'no Falcon window is open: give path (a *.falcon.json file) to work on a file'}
            r = self.file.open(path)
            if 'error' not in r:
                self.sticky_file = True
                r['note'] = ('calls go to this file now, even with a Falcon window open; open_project without a path '
                             'goes back to the window')
            return r

    def call(self, name, args, cancel=None) -> dict:
        args = args if isinstance(args, dict) else {}
        if name == 'open_project':
            out = self.open(args)
            return SV.text(out, 'error' in out)
        try:
            s = self.session()
        except LinkError as e:
            return SV.error(str(e))
        if isinstance(s, LiveLink):
            try:
                return s.call(name, args, cancel)
            except LinkError as e:
                with self._lock:
                    self.link = None
                return SV.error(f'{e}. Call the tool again to use whatever is open now.')
        return SV.dispatch(s, name, args)

    def writes(self, name: str) -> bool:
        """Whether a call changes something (it waits for every call before it, and holds back the ones after)."""
        if name in ('undo_last_change', 'redo_last_change', 'open_project'):
            return True
        t = SV._registry().get(name)
        return t is not None and t.tier != 'read'


class _Order:
    """Arrival order: a read starts once every earlier write has finished; a write once every earlier call has. A
    call takes its ticket as it arrives (take, on the reading thread) and waits for its turn on its worker (wait)."""

    def __init__(self):
        self._c = threading.Condition()
        self._n = 0
        self._open: dict = {}

    def take(self, write: bool) -> int:
        with self._c:
            t = self._n
            self._n += 1
            self._open[t] = 'w' if write else 'r'
            return t

    def wait(self, t: int):
        with self._c:
            write = self._open.get(t) == 'w'
            while any(k < t and (write or v == 'w') for k, v in self._open.items()):
                self._c.wait()

    def enter(self, write: bool) -> int:
        t = self.take(write)
        self.wait(t)
        return t

    def leave(self, t):
        with self._c:
            self._open.pop(t, None)
            self._c.notify_all()


class Server:
    def __init__(self, router: Router, out):
        self.router = router
        self.out = out
        self._wlock = threading.Lock()
        self.level = None
        self.inflight: dict = {}                 # request id -> cancel event
        self.threads: list = []
        self.order = _Order()
        self.client: dict = {}

    # ---------------------------------------------------------------- output
    def send(self, doc):
        line = json.dumps(doc, ensure_ascii=False, default=str) + '\n'
        with self._wlock:
            try:
                self.out.write(line)
                self.out.flush()
            except (OSError, ValueError, AttributeError):       # the client went away
                pass

    def notify(self, method, params):
        self.send({'jsonrpc': '2.0', 'method': method, 'params': params})

    def log(self, level, data):
        if self.level is not None and LEVELS.index(level) >= LEVELS.index(self.level):
            self.notify('notifications/message', {'level': level, 'logger': 'falcon', 'data': data})

    # ---------------------------------------------------------------- running tools
    def run_tool(self, name, args, cancel, ticket=None):
        if ticket is None:
            ticket = self.order.take(self.router.writes(name))
        t0 = time.time()
        try:
            self.order.wait(ticket)
            if cancel is not None and cancel.is_set():
                return SV.error('cancelled before it ran')
            self.log('debug', {'tool': name, 'event': 'start'})
            res = self.router.call(name, args, cancel)
        finally:
            self.order.leave(ticket)
        dt = time.time() - t0
        if res.get('isError'):
            body = next((c.get('text', '') for c in res.get('content', []) if c.get('type') == 'text'), '')
            self.log('warning', {'tool': name, 'event': 'error', 'seconds': round(dt, 2), 'error': body[:500]})
        else:
            self.log('info', {'tool': name, 'event': 'done', 'seconds': round(dt, 2)})
        return res

    def _heartbeat(self, token, label, done, t0):
        while not done.wait(PROGRESS_S):
            el = time.time() - t0
            self.notify('notifications/progress', {'progressToken': token, 'progress': round(el, 1),
                                                   'message': f'{label}: running for {el:.0f} s'})

    def _threaded(self, mid, fn, progress=None, label=''):
        ev = threading.Event()
        self.inflight[mid] = ev

        def run():
            done = threading.Event()
            if progress is not None:
                threading.Thread(target=self._heartbeat, args=(progress, label, done, time.time()),
                                 daemon=True).start()
            try:
                reply = {'jsonrpc': '2.0', 'id': mid, 'result': fn(ev)}
            except CT.NotFound as e:
                reply = {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32002, 'message': str(e)}}
            except (ValueError, TypeError) as e:
                reply = {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32602, 'message': str(e)}}
            except Exception as e:             # noqa: BLE001 (the server keeps running)
                traceback.print_exc(file=sys.stderr)
                reply = {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32603,
                                                                'message': f'{type(e).__name__} in the server'}}
            finally:
                done.set()
                self.inflight.pop(mid, None)
            if not ev.is_set():                # a cancelled request gets no answer
                self.send(reply)
        th = threading.Thread(target=run, daemon=True)
        self.threads = [t for t in self.threads if t.is_alive()] + [th]
        th.start()

    def _read(self, uri, cancel, ticket=None):
        name, args = CT.resolve(uri)
        res = self.run_tool(name, args, cancel, ticket)
        if res.get('isError'):
            body = next((c.get('text', '') for c in res.get('content', []) if c.get('type') == 'text'), '')
            raise CT.NotFound(f'{uri}: {body[:400]}')
        return {'contents': CT.contents(uri, res)}

    # ---------------------------------------------------------------- the protocol
    def handle(self, msg):
        if not isinstance(msg, dict) or msg.get('jsonrpc') != '2.0':
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'invalid request'}}
        if 'method' not in msg:                # a response to something we sent (we send no requests): ignore
            return None
        mid, method, params = msg.get('id'), msg.get('method'), msg.get('params') or {}
        if mid is None:                        # a notification
            if method == 'notifications/cancelled':
                ev = self.inflight.get(params.get('requestId'))
                if ev is not None:
                    ev.set()
            return None
        meta = params.get('_meta') or {}
        token = meta.get('progressToken') if isinstance(meta, dict) else None
        try:
            if method == 'initialize':
                want = params.get('protocolVersion')
                self.client = params.get('capabilities') or {}
                from .. import __version__
                result = {'protocolVersion': want if want in VERSIONS else VERSIONS[0],
                          'capabilities': {'tools': {'listChanged': False},
                                           'resources': {'subscribe': False, 'listChanged': False},
                                           'logging': {}},
                          'serverInfo': {'name': 'falcon', 'title': 'Falcon Autonomy', 'version': __version__},
                          'instructions': SV.PROTOCOL_NOTE + INSTRUCTIONS}
            elif method == 'ping':
                result = {}
            elif method == 'tools/list':
                result = {'tools': SV.listing()}
            elif method == 'tools/call':
                name, args = str(params.get('name', '')), params.get('arguments') or {}
                ticket = self.order.take(self.router.writes(name))          # its place: the order calls arrived in
                self._threaded(mid, lambda ev: self.run_tool(name, args, ev, ticket), token, name)
                return None
            elif method == 'resources/list':
                result = {'resources': CT.RESOURCES}
            elif method == 'resources/templates/list':
                result = {'resourceTemplates': CT.TEMPLATES}
            elif method == 'resources/read':
                uri = str(params.get('uri', ''))
                name, _args = CT.resolve(uri)  # an unknown URI is answered at once
                ticket = self.order.take(self.router.writes(name))
                self._threaded(mid, lambda ev: self._read(uri, ev, ticket), token, uri)
                return None
            elif method == 'prompts/list':
                result = {'prompts': []}
            elif method == 'logging/setLevel':
                lv = params.get('level')
                if lv not in LEVELS:
                    return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32602,
                                                                   'message': f'level: one of {", ".join(LEVELS)}'}}
                self.level = lv
                result = {}
            else:
                return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': f'no method {method}'}}
        except CT.NotFound as e:
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32002, 'message': str(e)}}
        except ValueError as e:
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32602, 'message': str(e)}}
        except Exception as e:                 # noqa: BLE001 (the server keeps running)
            traceback.print_exc(file=sys.stderr)
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32603, 'message': f'{type(e).__name__}'}}
        return {'jsonrpc': '2.0', 'id': mid, 'result': result}

    def serve(self, inp):
        for raw in inp:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send({'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'parse error'}})
                continue
            for m in (msg if isinstance(msg, list) else [msg]):
                reply = self.handle(m)
                if reply is not None:
                    self.send(reply)
        for th in list(self.threads):          # the input ended: answer what is still running first
            th.join()
        self.router.file.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='falcon --mcp', description='Falcon\'s MCP server on stdin and stdout '
                                 '(docs/mcp.md): an MCP client such as Claude Code uses Falcon through its tools.')
    ap.add_argument('--mcp', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--mcp-project', '--project', dest='project', default='',
                    help='the project file (*.falcon.json) to work on when no Falcon window is open')
    ap.add_argument('--mcp-mode', '--mode', dest='mode', choices=('auto', 'window', 'file'), default='auto',
                    help='auto: the open window, else the file (default); window: only a window; file: only the file')
    ap.add_argument('--fake-carla', default='', help=argparse.SUPPRESS)
    try:
        a, _rest = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    out = sys.stdout
    sys.stdout = sys.stderr                    # only protocol messages may reach stdout
    try:
        out.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    try:
        Server(Router(a.project, a.mode, a.fake_carla), out).serve(sys.stdin)
    finally:
        sys.stdout = out
    return 0
