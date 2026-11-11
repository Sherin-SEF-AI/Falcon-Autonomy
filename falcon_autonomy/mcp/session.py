"""Where an MCP client's tool calls go: the Falcon window that is open (LiveLink, through the window's bridge,
ui.mcp_bridge) or a project file (FileSession, here in the MCP process). Ported from Vantage mcp/session.py (GPL).

A window that lets MCP clients in (Settings ▸ Assistant ▸ Provider, on by default) listens on 127.0.0.1 only and
writes <config>/mcp/<pid>.json (the directory 0700, the file 0600: readable by the user alone) with its port and a
random token; every request carries the token, compared in constant time. Without a window, FileSession opens the
project file with the same tools the command line has (assistant.tools.context): each act call is one undoable step,
saved as it is made, after the file as it was is copied to <name>.bak the first time; a file changed on disk since it
was opened (by Falcon's window, an editor, git) is never overwritten — open_project reads it again."""
from __future__ import annotations

import copy
import itertools
import json
import os
import shutil
import socket
import sys
import threading
import time

PREFIX = 'Claude ·'                   # the undo steps an MCP client made, in the window and in a file session


def bridge_dir() -> str:
    from ..core.paths import config_dir
    return os.path.join(config_dir(), 'mcp')


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def windows() -> list:
    """The open Falcon windows that let MCP clients in, newest first (a file left by a window that is gone is
    removed)."""
    out = []
    d = bridge_dir()
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        if not n.endswith('.json'):
            continue
        f = os.path.join(d, n)
        try:
            st = os.stat(f)
            if st.st_uid != os.getuid() or st.st_mode & 0o077:
                continue                      # not the user's own, or readable by others: not trusted
            with open(f, encoding='utf-8') as fh:
                doc = json.load(fh)
            if not _alive(int(doc.get('pid', 0))):
                os.remove(f)
                continue
            out.append(doc)
        except (OSError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda doc: -float(doc.get('started', 0)))


def same_file(a: str, b: str) -> bool:
    try:
        return bool(a) and bool(b) and os.path.samefile(a, b)
    except OSError:
        return bool(a) and bool(b) and os.path.abspath(a) == os.path.abspath(b)


class LinkError(Exception):
    pass


# =========================================================================== the window
_CALLS = itertools.count(1)


class LiveLink:
    """Requests to one window's bridge: one JSON line each way, on a small pool of connections so calls can run side
    by side and a cancel reaches the window while a call still runs."""
    mode = 'window'

    def __init__(self, info: dict, timeout: float = 900.0):
        self.info = info
        self.timeout = timeout
        self._free: list = []
        self._lock = threading.Lock()
        self._n = 0

    @property
    def pid(self) -> int:
        return int(self.info.get('pid', 0))

    def _take(self):
        with self._lock:
            if self._free:
                return self._free.pop()
        try:
            sock = socket.create_connection(('127.0.0.1', int(self.info['port'])), timeout=10)
        except (OSError, KeyError, ValueError) as e:
            raise LinkError(f'the Falcon window (pid {self.pid}) does not answer: {e}') from None
        sock.settimeout(self.timeout)
        return [sock, b'']

    def _give(self, conn):
        with self._lock:
            if len(self._free) < 4:
                self._free.append(conn)
                return
        try:
            conn[0].close()
        except OSError:
            pass

    def request(self, method: str, **params) -> dict:
        conn = self._take()
        with self._lock:
            self._n += 1
            n = self._n
        msg = {'token': self.info.get('token', ''), 'id': n, 'method': method, **params}
        sock = conn[0]
        try:
            sock.sendall((json.dumps(msg) + '\n').encode('utf-8'))
            while b'\n' not in conn[1]:
                chunk = sock.recv(1 << 20)
                if not chunk:
                    raise OSError('the window closed the connection')
                conn[1] += chunk
        except OSError as e:
            try:
                sock.close()
            except OSError:
                pass
            raise LinkError(f'lost the Falcon window (pid {self.pid}): {e}') from None
        line, conn[1] = conn[1].split(b'\n', 1)
        self._give(conn)
        doc = json.loads(line.decode('utf-8'))
        if 'error' in doc:
            raise LinkError(str(doc['error']))
        return doc.get('result') or {}

    def call(self, name, args, cancel=None) -> dict:
        """A tool call in the window; cancel (a threading.Event) set while it runs is passed on to the window."""
        cid = f'{os.getpid()}-{next(_CALLS)}'
        if cancel is None:
            return self.request('call', name=name, args=args or {}, call_id=cid)
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                if cancel.wait(0.2):
                    try:
                        self.request('cancel', call_id=cid)
                    except LinkError:
                        pass
                    return
        threading.Thread(target=watch, daemon=True).start()
        try:
            return self.request('call', name=name, args=args or {}, call_id=cid)
        finally:
            stop.set()

    def revision(self):
        r = self.request('ping')
        return r.get('path'), int(r.get('undo_index', 0))

    def close(self):
        with self._lock:
            conns, self._free = self._free, []
        for c in conns:
            try:
                c[0].close()
            except OSError:
                pass


# =========================================================================== a file
class FileSession:
    """A project file worked on without the window: Falcon's tools on it as the command line runs them; each act call
    one undoable step, saved as it is made (a backup first), never over a file changed on disk."""
    mode = 'file'

    def __init__(self, fake_carla: str = ''):
        self.fake_carla = fake_carla
        self.ctx = None                       # assistant.tool_api.ToolContext of the open project
        self.path = ''
        self.undo_stack: list = []            # [(text, doc before, doc after)]
        self.redo_stack: list = []
        self.rev = 0
        self._mtime = 0.0
        self._backed_up: set = set()
        self._lock = threading.RLock()        # one act call at a time on the file
        self._pump = None

    # ---------------------------------------------------------------- opening
    def open(self, path: str) -> dict:
        from ..assistant import tools
        from ..core.project import ProjectError
        from ..core.paths import PROJECT_SUFFIX
        path = os.path.abspath(os.path.expanduser(str(path or '')))
        if not path.endswith(PROJECT_SUFFIX) or not os.path.isfile(path):
            return {'error': f'{path} is not a Falcon project file (*.falcon.json)'}
        open_in = [w for w in windows() if same_file(w.get('path', ''), path)]
        if open_in:
            return {'error': f'{os.path.basename(path)} is open in a Falcon window (pid {open_in[0]["pid"]}): the '
                             'calls go there'}
        try:
            ctx = tools.context(path, self.fake_carla)
        except (OSError, ProjectError, ValueError) as e:
            return {'error': f'cannot open {path}: {e}'}
        with self._lock:
            self.close()
            self.ctx, self.path = ctx, path
            from ..assistant import privacy
            ctx.redact = privacy.redact
            ctx.approval = {'how': 'mcp'}
            self.undo_stack, self.redo_stack, self.rev = [], [], 0
            self._mtime = os.path.getmtime(path)
        return {'opened': path, 'project': ctx.project.doc.get('name', ''), 'workspace': ctx.workspace}

    def close(self):
        ctx, self.ctx = self.ctx, None
        if ctx is not None:
            for j in list(getattr(ctx.queue, 'jobs', [])):
                if j.state not in ('done', 'failed', 'cancelled'):
                    ctx.queue.cancel(j)
            try:
                ctx.service.shutdown()
            except Exception:                 # noqa: BLE001 (closing anyway)
                pass

    def _need(self):
        if self.ctx is None:
            raise RuntimeError('no project: open Falcon (the calls then go to its window), or call open_project with '
                               'a project file, or start the server with --mcp-project FILE')

    def where(self) -> str:
        return f'the project file {self.path}' if self.path else 'no project yet'

    # ---------------------------------------------------------------- changes
    def _check_disk(self):
        p = self.path
        if p and os.path.exists(p) and self._mtime and abs(os.path.getmtime(p) - self._mtime) > 1e-3:
            raise RuntimeError(f'{os.path.basename(p)} changed on disk since it was opened here: call open_project '
                               'with it again before changing it (nothing was changed)')
        if p and any(same_file(w.get('path', ''), p) for w in windows()):
            raise RuntimeError(f'{os.path.basename(p)} is now open in a Falcon window: the calls go there (nothing '
                               'was changed here)')

    def _backup(self):
        p = self.path
        if p and p not in self._backed_up and os.path.exists(p):
            shutil.copy2(p, p + '.bak')
            self._backed_up.add(p)

    @staticmethod
    def _body(doc) -> str:
        return json.dumps({k: v for k, v in doc.items() if k not in ('modified', 'app_version')}, sort_keys=True,
                          default=str)

    def _save(self):
        proj = self.ctx.project
        proj.save()
        self._mtime = os.path.getmtime(proj.path)
        self.path = proj.path

    def call_tool(self, name: str, args: dict, tier: str) -> dict:
        """Run one of Falcon's tools on the file: an act call is one undoable step (named PREFIX + tool), saved."""
        from ..assistant import tool_api as T
        from .service import gate_refusal
        self._need()
        t = T.REGISTRY.get(name)
        g = gate_refusal(t, self.ctx, args) if t is not None else None
        if g:
            return {'error': g}
        if tier != 'act':
            return T.call(self.ctx, name, args)
        with self._lock:
            self._check_disk()
            self._backup()
            proj = self.ctx.project
            before = copy.deepcopy(proj.doc)
            res = T.call(self.ctx, name, args)
            if self._body(proj.doc) != self._body(before):
                self.undo_stack.append((f'{PREFIX} {name}', before, copy.deepcopy(proj.doc)))
                self.redo_stack.clear()
                self.rev += 1
                self._save()
                if isinstance(res, dict):
                    res = dict(res, undo='undo_last_change takes this step back (the file as it was first is kept '
                                         f'as {os.path.basename(self.path)}.bak)')
            elif os.path.exists(self.path):
                self._mtime = os.path.getmtime(self.path)
            self._start_pump()
        return res

    def _start_pump(self):
        """Run the jobs a call queued (the command line's loop: the queue polled until they end)."""
        q = self.ctx.queue
        if not any(j.state not in ('done', 'failed', 'cancelled') for j in getattr(q, 'jobs', [])):
            return
        if self._pump is not None and self._pump.is_alive():
            return

        def pump():
            while self.ctx is not None and self.ctx.queue is q and \
                    any(j.state not in ('done', 'failed', 'cancelled') for j in q.jobs):
                try:
                    q.poll()
                except Exception as e:         # noqa: BLE001 (a job's problem is the job's)
                    print(f'falcon mcp: the job queue: {type(e).__name__}: {e}', file=sys.stderr)
                time.sleep(0.25)
        self._pump = threading.Thread(target=pump, name='falcon-mcp-jobs', daemon=True)
        self._pump.start()

    def undo(self) -> str:
        self._need()
        with self._lock:
            if not self.undo_stack:
                return 'nothing to undo: no change was made here'
            self._check_disk()
            text, before, after = self.undo_stack.pop()
            self.ctx.project.doc = copy.deepcopy(before)
            self.redo_stack.append((text, before, after))
            self.rev += 1
            self._save()
            return f'undid: {text}'

    def redo(self) -> str:
        self._need()
        with self._lock:
            if not self.redo_stack:
                return 'nothing to redo'
            self._check_disk()
            text, before, after = self.redo_stack.pop()
            self.ctx.project.doc = copy.deepcopy(after)
            self.undo_stack.append((text, before, after))
            self.rev += 1
            self._save()
            return f'redid: {text}'

    def status(self) -> dict:
        out = {'where': self.where(), 'mode': 'file'}
        if self.ctx is not None:
            jobs = [{'job': j.id, 'title': j.spec.title, 'state': j.state} for j in getattr(self.ctx.queue, 'jobs', [])]
            out.update(project=self.ctx.project.doc.get('name', ''), workspace=self.ctx.workspace,
                       steps=[t for t, _b, _a in self.undo_stack], backup=(self.path + '.bak')
                       if self.path in self._backed_up else '', jobs=jobs[-20:],
                       note='jobs queued here run in the MCP server\'s process and stop with it; for long jobs open '
                            'the project in Falcon\'s window')
        return out

    def revision(self):
        return self.path, self.rev
