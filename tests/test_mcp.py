"""Falcon's MCP server (falcon_autonomy.mcp): the stdio protocol (versions, tools with annotations, resources,
structured content, errors as JSON-RPC), the refusals (trash, the never-list, a tool that asks on a card, a window tool
without a window), a project file worked on without the window (one undoable step per act call, a backup the first
time, a file changed on disk left alone), arrival order, cancellation and progress, a real stdio subprocess, and the
open window through its token-protected bridge (ui.mcp_bridge) with each act call one undo step."""
import json
import os
import stat
import subprocess
import sys
import threading
import time

import pytest

from conftest import FAKE, ROOT, worker_env
from falcon_autonomy.assistant import tool_api as T
from falcon_autonomy.core.project import Project
from falcon_autonomy.mcp import catalog as CT
from falcon_autonomy.mcp import server as S
from falcon_autonomy.mcp import service as SV
from falcon_autonomy.mcp import session as SE


class Out:
    """stdout for an in-process server: the lines it writes, parsed."""

    def __init__(self):
        self.lines, self.lock = [], threading.Lock()

    def write(self, s):
        with self.lock:
            self.lines.extend(json.loads(x) for x in s.splitlines() if x.strip())

    def flush(self):
        pass

    def reply(self, mid, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                got = [x for x in self.lines if x.get('id') == mid and 'method' not in x]
            if got:
                return got[0]
            time.sleep(0.01)
        raise AssertionError(f'no reply to {mid}')


def req(mid, method, **params):
    return {'jsonrpc': '2.0', 'id': mid, 'method': method, 'params': params}


def doc_of(result):
    return json.loads(result['content'][0]['text'])


@pytest.fixture
def server(project):
    out = Out()
    srv = S.Server(S.Router(project.path, 'file', FAKE), out)
    yield srv, out, project.path
    srv.router.file.close()


def call(srv, out, mid, tool, **args):
    assert srv.handle(req(mid, 'tools/call', name=tool, arguments=args)) is None     # answered from a worker
    return out.reply(mid)['result']


# ======================================================================= protocol
def test_initialize_lists_tools_resources_and_answers_errors():
    srv = S.Server(S.Router(mode='file'), out=None)
    r = srv.handle(req(1, 'initialize', protocolVersion='2025-06-18', capabilities={}))['result']
    assert r['protocolVersion'] == '2025-06-18' and r['serverInfo']['name'] == 'falcon'
    assert 'dashcam' in r['instructions'] and r['capabilities']['tools'] == {'listChanged': False}
    r = srv.handle(req(2, 'initialize', protocolVersion='1999-01-01'))['result']
    assert r['protocolVersion'] == S.VERSIONS[0]
    tools = srv.handle(req(3, 'tools/list'))['result']['tools']
    names = {t['name'] for t in tools}
    assert {'project_overview', 'runs_search', 'jobs_drive', 'project_world_add', 'falcon_status',
            'undo_last_change', 'open_project'} <= names
    trash = {t.name for t in T.REGISTRY.values() if t.tier == 'trash'}
    assert trash and not names & trash and not names & set(T.NEVER)
    ov = next(t for t in tools if t['name'] == 'project_overview')
    assert ov['annotations']['readOnlyHint'] is True and ov['inputSchema']['type'] == 'object'
    add = next(t for t in tools if t['name'] == 'project_world_add')
    assert add['annotations']['readOnlyHint'] is False and add['annotations']['destructiveHint'] is False
    assert 'undoable' in add['description']
    assert {x['uri'] for x in srv.handle(req(4, 'resources/list'))['result']['resources']} >= {
        'falcon://status', 'falcon://overview', 'falcon://runs', 'falcon://queue'}
    assert srv.handle(req(5, 'resources/templates/list'))['result']['resourceTemplates'][0]['uriTemplate'] == \
        'falcon://run/{run}'
    assert srv.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None
    assert srv.handle(req(6, 'nope'))['error']['code'] == -32601
    assert srv.handle(req(7, 'resources/read', uri='falcon://nothing'))['error']['code'] == -32002
    assert srv.handle(req(8, 'logging/setLevel', level='loud'))['error']['code'] == -32602
    assert srv.handle({'id': 9})['error']['code'] == -32600
    assert CT.resolve('falcon://run/runs/drive%2D1') == ('runs_get', {'run': 'runs/drive-1'})


def test_refusals_trash_never_cards_and_window_tools(server, monkeypatch):
    srv, out, path = server
    r = call(srv, out, 1, 'runs_trash', runs=['runs/x'])
    assert r['isError'] and 'only from Falcon\'s window' in doc_of(r)['error']
    r = call(srv, out, 2, 'shell', command='rm -rf /')
    assert r['isError'] and 'never allowed' in doc_of(r)['error']
    r = call(srv, out, 3, 'ui_open_page', page='data')
    assert r['isError'] and 'needs Falcon\'s window' in doc_of(r)['error']
    ran = []
    monkeypatch.setitem(T.REGISTRY, 'jobs_card', T.Tool(
        'jobs_card', 'jobs', 'act', 'asks every time', {'type': 'object', 'properties': {}},
        lambda ctx: ran.append(1) or {'ok': True}, approval=lambda ctx: {'ask': True, 'why': 'it queues compute'}))
    r = call(srv, out, 4, 'jobs_card')
    assert r['isError'] and 'cannot answer that card' in doc_of(r)['error'] and ran == []
    assert 'jobs_card' in {t['name'] for t in SV.listing()}
    with open(path, encoding='utf-8') as fh:
        assert json.load(fh) == Project.load(path).doc            # nothing was written


def test_file_session_one_step_per_call_backup_undo_and_disk_check(server):
    srv, out, path = server
    with open(path, encoding='utf-8') as fh:
        original = fh.read()
    r = call(srv, out, 1, 'project_overview')
    assert not r['isError'] and r['structuredContent'] == doc_of(r)          # structured content: the same JSON
    n0 = len(Project.load(path).doc['worlds'])
    r = call(srv, out, 2, 'project_world_add', town='Town03', name='Third')
    assert not r['isError'], r
    assert 'undo_last_change' in doc_of(r)['undo']
    assert len(Project.load(path).doc['worlds']) == n0 + 1                   # saved as it was made
    with open(path + '.bak', encoding='utf-8') as fh:
        assert fh.read() == original                                         # the file as it was, kept once
    call(srv, out, 3, 'project_world_add', town='Town04', name='Fourth')
    with open(path + '.bak', encoding='utf-8') as fh:
        assert fh.read() == original
    st = doc_of(call(srv, out, 4, 'falcon_status'))
    assert st['mode'] == 'file' and st['steps'] == ['Claude · project_world_add'] * 2
    r = call(srv, out, 5, 'undo_last_change')
    assert not r['isError'] and len(Project.load(path).doc['worlds']) == n0 + 1
    r = call(srv, out, 6, 'redo_last_change')
    assert len(Project.load(path).doc['worlds']) == n0 + 2
    time.sleep(0.02)
    p = Project.load(path)                                                   # someone else saves the file
    p.doc['name'] = 'Edited elsewhere'
    p.save()
    os.utime(path, (time.time() + 5, time.time() + 5))
    r = call(srv, out, 7, 'project_world_add', town='Town05')
    assert r['isError'] and 'changed on disk' in doc_of(r)['error']
    assert Project.load(path).doc['name'] == 'Edited elsewhere'              # left alone
    r = call(srv, out, 8, 'open_project', path=path)
    assert not r['isError']
    assert not call(srv, out, 9, 'project_world_add', town='Town05')['isError']
    r = srv.handle(req(10, 'resources/read', uri='falcon://overview'))
    assert r is None
    contents = out.reply(10)['result']['contents']
    assert contents[0]['mimeType'] == 'application/json' and json.loads(contents[0]['text'])
    r = call(srv, out, 11, 'open_project', path=str(path) + '.bak')
    assert r['isError'] and 'not a Falcon project file' in doc_of(r)['error']


def test_arrival_order_cancellation_and_progress(server, monkeypatch):
    srv, out, _path = server
    monkeypatch.setattr(S, 'PROGRESS_S', 0.05)
    gate, log = threading.Event(), []

    def slow(ctx):
        log.append('read start')
        gate.wait(10)
        log.append('read end')
        return {'ok': True}

    def write(ctx):
        log.append('write')
        return {'ok': True}
    monkeypatch.setitem(T.REGISTRY, 'runs_slow', T.Tool('runs_slow', 'runs', 'read', 'slow', {
        'type': 'object', 'properties': {}}, slow))
    monkeypatch.setitem(T.REGISTRY, 'jobs_write', T.Tool('jobs_write', 'jobs', 'act', 'w', {
        'type': 'object', 'properties': {}}, write))
    srv.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': 'runs_slow', 'arguments': {}, '_meta': {'progressToken': 'p1'}}})
    srv.handle(req(2, 'tools/call', name='jobs_write', arguments={}))           # waits for the read before it
    srv.handle(req(3, 'tools/call', name='runs_slow', arguments={}))            # waits for the write
    srv.handle({'jsonrpc': '2.0', 'method': 'notifications/cancelled', 'params': {'requestId': 3}})
    time.sleep(0.3)
    assert log == ['read start']
    prog = [x for x in out.lines if x.get('method') == 'notifications/progress']
    assert prog and prog[0]['params']['progressToken'] == 'p1'
    gate.set()
    assert not out.reply(1)['result']['isError'] and not out.reply(2)['result']['isError']
    time.sleep(0.3)
    assert log == ['read start', 'read end', 'write']                          # the cancelled call never ran
    assert not [x for x in out.lines if x.get('id') == 3]                      # and was never answered


def test_stdio_server_in_a_process(tmp_path, project):
    path = project.path
    msgs = [req(1, 'initialize', protocolVersion='2025-11-25', capabilities={}),
            {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
            req(2, 'tools/call', name='project_world_add', arguments={'town': 'Town02', 'name': 'From MCP'}),
            req(3, 'resources/read', uri='falcon://status'),
            req(4, 'tools/call', name='runs_trash', arguments={'runs': ['x']})]
    env = worker_env({'XDG_CONFIG_HOME': str(tmp_path / 'cfg'), 'XDG_CACHE_HOME': str(tmp_path / 'cache')})
    p = subprocess.run([sys.executable, '-m', 'falcon_autonomy.mcp', '--mcp-project', path, '--fake-carla', FAKE],
                       input='\n'.join(json.dumps(m) for m in msgs) + '\n', capture_output=True, text=True, env=env,
                       timeout=300, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    lines = [json.loads(x) for x in p.stdout.splitlines()]
    assert sorted(d['id'] for d in lines) == [1, 2, 3, 4]                        # stdout holds the protocol only
    by = {d['id']: d for d in lines}
    assert by[1]['result']['serverInfo']['name'] == 'falcon'
    assert not by[2]['result']['isError'], by[2]
    assert any(w.get('name') == 'From MCP' for w in Project.load(path).doc['worlds'])
    assert os.path.isfile(path + '.bak')
    st = json.loads(by[3]['result']['contents'][0]['text'])
    assert st['mode'] == 'file' and st['steps'] == ['Claude · project_world_add']
    assert by[4]['result']['isError']


def test_windows_are_found_only_from_the_users_own_private_files(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    d = SE.bridge_dir()
    os.makedirs(d)
    good = os.path.join(d, f'{os.getpid()}.json')
    with open(good, 'w') as fh:
        json.dump({'pid': os.getpid(), 'port': 1, 'token': 't', 'path': '', 'started': 1}, fh)
    os.chmod(good, 0o600)
    loose = os.path.join(d, '1.json')
    with open(loose, 'w') as fh:
        json.dump({'pid': 1, 'port': 2, 'token': 't', 'started': 2}, fh)
    os.chmod(loose, 0o644)                                                     # readable by others: ignored
    dead = os.path.join(d, '999999999.json')
    with open(dead, 'w') as fh:
        json.dump({'pid': 999999999, 'port': 3, 'token': 't', 'started': 3}, fh)
    os.chmod(dead, 0o600)
    assert [w['port'] for w in SE.windows()] == [1] and not os.path.exists(dead)


# ======================================================================= the window, through its bridge
@pytest.fixture(scope='module')
def qapp():
    pytest.importorskip('PyQt6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from falcon_autonomy.ui.app import make_app
    return make_app()


def pump(app, cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_window_bridge_token_and_one_undo_step_per_call(qapp, tmp_path, project, monkeypatch):
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.ui import tasks
    from falcon_autonomy.ui.mainwindow import MainWindow
    win = MainWindow(Settings(str(tmp_path / 'settings.json')), project, fake_carla=FAKE)
    try:
        b = win.mcp_bridge
        assert b.running and os.path.isfile(b.file)
        assert stat.S_IMODE(os.stat(b.file).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(os.path.dirname(b.file)).st_mode) == 0o700
        info = next(w for w in SE.windows() if w['pid'] == os.getpid())
        assert info['path'] == project.path and len(info['token']) == 48

        def in_thread(fn):
            box = {}

            def run():
                try:
                    box['v'] = fn()
                except Exception as e:                                   # noqa: BLE001 (raised below)
                    box['e'] = e
            th = threading.Thread(target=run, daemon=True)
            th.start()
            assert pump(qapp, lambda: not th.is_alive()), 'the bridge did not answer'
            if 'e' in box:
                raise box['e']
            return box['v']
        bad = SE.LiveLink(dict(info, token='0' * 48))
        with pytest.raises(SE.LinkError, match='wrong token'):
            in_thread(lambda: bad.request('ping'))
        link = SE.LiveLink(info)
        n0, i0 = len(win.project.doc['worlds']), win.undo.index()
        r = in_thread(lambda: link.call('project_world_add', {'town': 'Town03', 'name': 'In the window'}))
        assert not r['isError'], r
        assert len(win.project.doc['worlds']) == n0 + 1
        assert win.undo.index() == i0 + 1 and win.undo.text(i0) == 'Claude · project_world_add'
        r = in_thread(lambda: link.call('project_overview', {}))
        assert not r['isError'] and win.undo.index() == i0 + 1                 # a read makes no step
        r = in_thread(lambda: link.call('runs_trash', {'runs': ['x']}))
        assert r['isError']
        st = doc_of(in_thread(lambda: link.call('falcon_status', {})))
        assert st['mode'] == 'window' and st['steps'][-1]['text'] == 'Claude · project_world_add'
        r = in_thread(lambda: link.call('undo_last_change', {}))
        assert not r['isError'] and len(win.project.doc['worlds']) == n0
        win.undo.redo()
        from falcon_autonomy.core.project import Edit
        win.edit(Edit('set', ['name'], 'User rename', text='rename'))           # the user's own step on top
        r = in_thread(lambda: link.call('undo_last_change', {}))
        assert r['isError'] and 'the user\'s' in doc_of(r)['result'] and win.project.doc['name'] == 'User rename'
        router = S.Router(project.path)                                        # the server picks the window
        assert isinstance(in_thread(router.session), SE.LiveLink)
        published = b.file
        win.shutdown()
        assert not os.path.exists(published) and not b.running                # the window closed: its file goes
    finally:
        win.shutdown()
        win.deleteLater()
        tasks.wait_all(30)
        qapp.processEvents()
