"""The window's side of MCP (docs/mcp.md): Falcon's MCP server (falcon_autonomy.mcp, started by Claude Code or another
MCP client) sends each tool call here, and it runs on this window's project. Ported from Vantage ui/mcp_bridge.py
(GPL).

The bridge listens on 127.0.0.1 only, on a free port, and writes <config>/mcp/<pid>.json (the port and a random token;
the directory 0700, the file 0600 — readable by the user alone) for the server to find; every request carries the
token, compared in constant time, and a wrong one closes the connection. A call runs on a worker thread; whatever
touches the project or the window is handed to the GUI thread (mcp.service.WindowSession): an act call's edits are
one undo step "Claude · <tool>" in Edit ▸ Undo, like the user's own. Settings ▸ Assistant ▸ Provider turns it off
(assistant.mcp_bridge, on by default); the file is removed when the window closes.

install(win) starts it for a window (MainWindow calls it once); the bridge is win.mcp_bridge.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QHostAddress, QTcpServer

from ..mcp import service as SV
from ..mcp.session import bridge_dir

MAX_LINE = 32 << 20


class _Invoker(QObject):
    """Runs a function on the GUI thread for a worker thread and hands back its result (or its exception)."""
    job = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.job.connect(self._run)

    def _run(self, item):
        fn, box, done = item
        try:
            box['value'] = fn()
        except BaseException as e:             # noqa: BLE001 (raised again in the worker)
            box['error'] = e
        finally:
            done.set()

    def call(self, fn, timeout: float = 600.0):
        if threading.current_thread() is threading.main_thread():
            return fn()
        box, done = {}, threading.Event()
        self.job.emit((fn, box, done))
        if not done.wait(timeout):
            raise RuntimeError('Falcon\'s window did not answer in time')
        if 'error' in box:
            raise box['error']
        return box.get('value')


def enabled(win) -> bool:
    try:
        return bool(win.assistant.cfg().get('mcp_bridge', True))
    except Exception:                          # noqa: BLE001
        return True


class Bridge(QObject):
    """The local server the MCP process talks to."""
    _reply = pyqtSignal(object, object)        # (socket, reply): written on the GUI thread

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.server = None
        self.token = ''
        self.file = ''
        self.invoker = _Invoker()
        log = getattr(getattr(win, 'log', None), 'append_line', None)
        self.session = SV.WindowSession(win, self.invoker.call, log=log)
        self._buf: dict = {}
        self._running: dict = {}               # call id -> cancel event
        self._reply.connect(self._write)

    @property
    def running(self) -> bool:
        return self.server is not None and self.server.isListening()

    def start(self) -> bool:
        if self.running:
            return True
        self.server = QTcpServer(self)
        if not self.server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost), 0):
            self.server = None
            return False
        self.server.newConnection.connect(self._accept)
        self.token = secrets.token_hex(24)
        self.publish()
        try:
            self.win.projectChanged.connect(self._project_changed)
        except (AttributeError, TypeError):
            pass
        return True

    def stop(self):
        if self.server is not None:
            self.server.close()
            self.server.deleteLater()
            self.server = None
        try:
            self.win.projectChanged.disconnect(self._project_changed)
        except (AttributeError, TypeError, RuntimeError):
            pass
        if self.file:
            try:
                os.remove(self.file)
            except OSError:
                pass
            self.file = ''

    def _project_changed(self, path):
        if path is None:                       # another project: its path in the file (the server routes by it)
            self.publish()

    def publish(self):
        """Write <config>/mcp/<pid>.json: 0600 in a 0700 directory, atomically."""
        if not self.running:
            return
        d = bridge_dir()
        os.makedirs(d, mode=0o700, exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
        self.file = os.path.join(d, f'{os.getpid()}.json')
        p = self.win.project
        doc = {'pid': os.getpid(), 'port': int(self.server.serverPort()), 'token': self.token,
               'project': p.doc.get('name', ''), 'path': p.path, 'started': time.time()}
        tmp = self.file + '.tmp'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh)
        os.replace(tmp, self.file)

    # ---------------------------------------------------------------- connections
    def _accept(self):
        while self.server is not None and self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            self._buf[id(sock)] = b''
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda s=sock: (self._buf.pop(id(s), None), s.deleteLater()))

    def _read(self, sock):
        buf = self._buf.get(id(sock), b'') + bytes(sock.readAll())
        if len(buf) > MAX_LINE:
            sock.abort()
            return
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            self._handle(sock, line)
        self._buf[id(sock)] = buf

    def _handle(self, sock, line):
        try:
            req = json.loads(line.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return self._write(sock, {'id': None, 'error': 'not JSON'})
        rid = req.get('id')
        if not hmac.compare_digest(str(req.get('token', '')).encode(), self.token.encode()):
            self._write(sock, {'id': rid, 'error': 'refused: wrong token'})
            sock.disconnectFromHost()
            return
        method = req.get('method')
        if method == 'ping':
            p, u = self.win.project, self.win.undo
            return self._write(sock, {'id': rid, 'result': {'project': p.doc.get('name', ''), 'path': p.path,
                                                             'undo_index': int(u.index())}})
        if method == 'cancel':
            ev = self._running.get(str(req.get('call_id', '')))
            if ev is not None:
                ev.set()
            return self._write(sock, {'id': rid, 'result': {'cancelled': ev is not None}})
        if method != 'call':
            return self._write(sock, {'id': rid, 'error': f'no method {method!r}'})
        name, args = str(req.get('name', '')), req.get('args') or {}
        cid = str(req.get('call_id') or f'{id(sock)}:{rid}')
        ev = threading.Event()
        self._running[cid] = ev
        threading.Thread(target=self._work, args=(sock, rid, name, args, cid, ev), daemon=True,
                         name='falcon-mcp-call').start()

    def _work(self, sock, rid, name, args, cid, ev):
        try:
            out = SV.error('cancelled before it ran') if ev.is_set() else SV.dispatch(self.session, name, args)
        except Exception as e:                 # noqa: BLE001 (the window keeps running)
            out = SV.error(f'{type(e).__name__} in Falcon\'s window')
        finally:
            self._running.pop(cid, None)
        try:
            self._reply.emit(sock, {'id': rid, 'result': out})
        except RuntimeError:                   # the window is closing
            pass

    def _write(self, sock, doc):
        try:
            if sock.state() == sock.SocketState.ConnectedState:
                sock.write((json.dumps(doc, ensure_ascii=False, default=str) + '\n').encode('utf-8'))
        except RuntimeError:                   # the socket is gone
            pass


def install(win):
    """Start the bridge for a window when the setting allows it (win.mcp_bridge); stopped when the app quits."""
    from PyQt6.QtWidgets import QApplication
    b = Bridge(win)
    win.mcp_bridge = b
    if enabled(win):
        b.start()
    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(b.stop)
    return b


def set_enabled(win, on: bool):
    """Settings: let MCP clients in (start) or not (stop, the file removed)."""
    b = getattr(win, 'mcp_bridge', None) or install(win)
    if on:
        b.start()
    else:
        b.stop()
