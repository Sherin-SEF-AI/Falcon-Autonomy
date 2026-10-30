"""The Assistant panel offscreen, against a fake Groq server: the empty state and shortcuts, Settings ▸ Assistant (the
masked key, where it is stored, consent, Test connection), a conversation with an approval card, a project edit undone
from the activity log, Autopilot continuing after a real job, Stop with Esc, a private path refused, the status bar,
and the panel fitting a 1280 px window in both themes."""
import os
import sys
import time

import pytest
from PyQt6.QtCore import Qt

pytestmark = pytest.mark.gui

from conftest import FAKE  # noqa: E402
from fake_groq import FAKE_KEY, FakeGroq, hang, text, tools  # noqa: E402


@pytest.fixture(scope='session')
def qapp():
    from falcon_autonomy.ui.app import make_app
    from falcon_autonomy.ui.theme import THEME, load_fonts
    app = make_app()
    load_fonts()
    THEME.apply(app, 'dark')
    yield app


def pump(app, cond=lambda: False, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@pytest.fixture(autouse=True)
def slot_exceptions(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, 'excepthook', lambda t, v, tb: seen.append((t, v)))
    yield seen
    assert not seen, f'exceptions in Qt slots: {seen}'


@pytest.fixture
def srv():
    s = FakeGroq().start()
    yield s
    s.stop()


@pytest.fixture
def reg(monkeypatch):
    from falcon_autonomy.assistant import tool_api as T
    monkeypatch.setattr(T, 'REGISTRY', {})
    return T.REGISTRY


@pytest.fixture
def window(qapp, tmp_path, project):
    from falcon_autonomy.assistant.keystore import KeyStore
    from falcon_autonomy.core.settings import Secrets, Settings
    from falcon_autonomy.ui import tasks
    from falcon_autonomy.ui.mainwindow import MainWindow
    win = MainWindow(Settings(str(tmp_path / 'settings.json')), project, fake_carla=FAKE)
    win.assistant.keys = KeyStore(Secrets(str(tmp_path / 'secrets.json')), use_keyring=False)
    win.show()
    pump(qapp, timeout=0.2)
    yield win
    win.shutdown()
    assert tasks.wait_all(30)
    pump(qapp, timeout=0.2)
    win._closing = True
    win.close()
    win.deleteLater()
    pump(qapp, timeout=0.2)


def ready(win, srv):
    svc = win.assistant
    svc.set_cfg(enabled=True, base_url=srv.base_url)
    svc.accept_notice()
    svc.keys.set(FAKE_KEY)
    win.show_assistant(True)
    return win.assistant_panel


def test_empty_state_shortcuts_and_palette(window, qapp):
    from falcon_autonomy.ui.palette import CommandPalette
    panel, dock = window.assistant_panel, window.assistant_dock
    assert not dock.isVisible() and window.assistant_badge.text() == 'Assistant off'
    assert window.actions_['assistant_panel'].shortcut().toString() == 'Ctrl+Shift+A'
    window.actions_['assistant_panel'].trigger()
    assert dock.isVisible() and panel.pages.currentIndex() == 0
    assert panel.empty_btn.text() == 'Set up the assistant' and 'sends text only' in panel.empty_msg.text()
    assert not panel.send_btn.isEnabled() and 'Turn it on' in panel.send_btn.toolTip()
    panel.empty_btn.click()
    assert window.stack.currentWidget().key == 'settings'
    pal = CommandPalette(window)
    pal.input.setText('ask falcon')
    assert pal.list.item(0).text().startswith('Ask Falcon…')
    pal.close()
    window.actions_['assistant_panel'].trigger()
    assert not dock.isVisible()


def test_settings_key_consent_and_test_connection(window, qapp, srv, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QLineEdit, QMessageBox
    from falcon_autonomy.ui import assistant_dock as D
    page = window.go('settings')
    assert page.as_key.echoMode() == QLineEdit.EchoMode.Password and not page.as_on.isChecked()
    page.as_eye.setChecked(True)
    assert page.as_key.echoMode() == QLineEdit.EchoMode.Normal
    page.as_eye.setChecked(False)
    asked = []
    monkeypatch.setattr(D.ConsentDialog, 'exec', lambda self: asked.append(1) or QDialog.DialogCode.Rejected)
    page.as_url.setText(srv.base_url)
    page.as_key.setText(FAKE_KEY)
    page.assistant_test()                                   # the notice declined: nothing goes out
    assert asked and srv.requests == [] and 'not accepted' in page.as_status.text()
    monkeypatch.setattr(D.ConsentDialog, 'exec', lambda self: QDialog.DialogCode.Accepted)
    page.assistant_test()
    assert pump(qapp, lambda: page.as_status.text().startswith(('✓', '⚠')), 10), page.as_status.text()
    assert page.as_status.text().startswith('✓ Connected: the server offers 3 models')
    assert page.as_main.findText('openai/gpt-oss-20b') >= 0 and page.as_on.isChecked()
    page.as_prices.setPlainText('openai/gpt-oss-120b 1 2\nbroken line')
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: asked.append('warned'))
    page.apply()
    assert asked[-1] == 'warned' and not window.assistant.keys.has()          # nothing saved on a bad table
    page.as_prices.setPlainText('openai/gpt-oss-120b 1 2')
    page.apply()
    svc = window.assistant
    assert svc.keys.get() == FAKE_KEY and page.as_key.text() == '' and 'stored in your secrets file' in \
        page.as_key.placeholderText()
    assert svc.cfg()['prices'] == {'openai/gpt-oss-120b': [1.0, 2.0]} and svc.consented()
    settings_text = open(window.settings.path).read()
    assert FAKE_KEY not in settings_text and '"enabled": true' in settings_text
    assert oct(os.stat(tmp_path / 'secrets.json').st_mode & 0o777) == '0o600'
    assert window.assistant_badge.text() == 'Assistant · gpt-oss-120b'
    assert FAKE_KEY not in window.log.toPlainText()
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)
    page.assistant_remove_key()
    assert not svc.keys.has() and not page.as_remove.isEnabled()
    assert 'No API key' in window.assistant_panel.svc.ready()[1]


def test_conversation_approval_undo_and_persistence(window, qapp, srv, reg):
    from falcon_autonomy.assistant import tool_api as T
    from falcon_autonomy.assistant.session import Session
    from falcon_autonomy.core.project import Edit
    from falcon_autonomy.ui import assistant_dock as D

    def rename(ctx, name):
        ctx.win.edit(Edit('set', ['name'], name, text='rename the project'))
        return {'name': name}
    T.register(T.Tool('project_rename', 'project', 'act', 'Rename the project. Undoable.',
                      {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}, rename,
                      preview=lambda ctx, name: f'name: "{ctx.project.doc["name"]}" → "{name}"',
                      estimate=lambda ctx, name: {'seconds': 0.0, 'what': 'one project edit'}, gui=True))
    T.register(T.Tool('runs_count', 'runs', 'read', 'count runs', {'type': 'object', 'properties': {}},
                      lambda ctx: {'runs': 7}))
    panel = ready(window, srv)
    old = window.project.doc['name']
    srv.push(tools(('runs_count', {}), content='I will rename it.'), tools(('project_rename', {'name': 'Bengaluru'})),
             text('Renamed to Bengaluru; it has 7 runs and an AP of 0.42.', prompt_tokens=1500, completion_tokens=40))
    assert panel.send('rename the project to Bengaluru')
    assert pump(qapp, lambda: panel.approvals, 10), [w for w in panel.findChildren(D.Bubble)]
    card = next(iter(panel.approvals.values()))
    assert card.req.preview == f'name: "{old}" → "Bengaluru"' and card.all is not None
    assert window.project.doc['name'] == old                 # nothing happened before the approval
    card.approve.click()
    assert pump(qapp, lambda: not panel.agent.busy, 10)
    pump(qapp, timeout=0.2)
    assert window.project.doc['name'] == 'Bengaluru'
    bubbles = [b for b in panel.findChildren(D.Bubble) if b.kind == 'assistant' and b.isVisible()]
    assert bubbles[-1].ungrounded == ['0.42'] and 'Not from Falcon' in bubbles[-1].text()
    cards = [c for c in panel.findChildren(D.ToolCard)]
    assert [c.name for c in cards] == ['runs_count', 'project_rename'] and cards[1].approval == 'approved'
    assert card.outcome.text() == 'Approved'
    assert 'Assistant · gpt-oss-120b · ' in window.assistant_badge.text() and 'tokens' in window.assistant_badge.text()
    acts = panel.session.activity
    assert [a['kind'] for a in acts] == ['edit']
    assert panel.activity.topLevelItemCount() == 1
    assert panel.undo(acts[0]['id']) == 'Undid 1 project edit(s).' and window.project.doc['name'] == old
    assert panel.undo(acts[0]['id']) == 'Already undone.'
    again = Session(window.workspace())
    assert [m['role'] for m in again.messages][:2] == ['user', 'assistant'] and again.activity[0]['undone']
    panel.load_session()                                     # a restart: the conversation is drawn again
    assert [c.name for c in panel.findChildren(D.ToolCard) if c.isVisible() or True][-2:] == \
        ['runs_count', 'project_rename']
    for root, _d, files in os.walk(window.workspace()):
        for f in files:
            with open(os.path.join(root, f), 'rb') as fh:
                assert FAKE_KEY.encode() not in fh.read()


def test_autopilot_continues_after_a_real_job(window, qapp, srv, reg):
    from falcon_autonomy.assistant import tool_api as T
    from falcon_autonomy.core.jobs import JobSpec
    from falcon_autonomy.ui import assistant_dock as D

    def queue(ctx):
        job = ctx.queue.submit(JobSpec('check', 'Short check', [sys.executable, '-c', 'import time; time.sleep(0.3)'],
                                       memory_gb=0.01))
        return {'job_id': job.id}
    T.register(T.Tool('jobs_check', 'jobs', 'act', 'queue a short check', {'type': 'object', 'properties': {}},
                      queue))
    panel = ready(window, srv)
    panel.mode.setCurrentIndex(2)
    assert panel.agent.mode == 'autopilot' and window.assistant.cfg()['mode'] == 'autopilot'
    srv.push(tools(('jobs_check', {})), text('The check is running.'), text('The check finished.'))
    assert panel.send('run the check and tell me')
    assert pump(qapp, lambda: len(srv.chats()) == 3 and not panel.agent.busy, 15), panel.status.text()
    chips = panel.findChildren(D.JobChip)
    assert len(chips) == 1 and 'done' in chips[0].title.text()
    events = [b.raw for b in panel.findChildren(D.Bubble) if b.kind == 'event']
    assert any('finished: done' in e for e in events)
    chips[0].show_in_panel()
    assert window.jobs_dock.isVisible()


def test_stop_with_escape_and_private_paths(window, qapp, srv, reg, tmp_path, monkeypatch):
    from PyQt6.QtTest import QTest
    from falcon_autonomy.assistant import privacy
    from falcon_autonomy.ui import assistant_dock as D
    panel = ready(window, srv)
    srv.push(hang('Looking', seconds=30))
    assert panel.send('hello')
    assert pump(qapp, srv.hung.is_set, 5)
    assert pump(qapp, lambda: panel.send_btn.text() == 'Stop', 5)
    QTest.keyClick(panel.composer, Qt.Key.Key_Escape)
    assert pump(qapp, lambda: not panel.agent.busy and panel.send_btn.text() == 'Send', 5)
    assert panel.status.text() == 'Stopped.'
    monkeypatch.setattr(privacy, 'HOME', str(tmp_path / 'home'))
    (tmp_path / 'home' / 'Videos').mkdir(parents=True)
    n = len(srv.chats())
    assert not panel.send(f'what is in {tmp_path}/home/Videos/trip.mp4')
    errs = [b.raw for b in panel.findChildren(D.Bubble) if b.kind == 'error']
    assert errs and 'private' in errs[-1] and len(srv.chats()) == n


def test_panel_fits_1280_in_both_themes(window, qapp, srv, reg):
    from falcon_autonomy.ui import assistant_dock as D
    panel = ready(window, srv)
    srv.push(text('Run `r_1` has **0.5** recall; see job 1.'))
    assert panel.send('x')
    assert pump(qapp, lambda: not panel.agent.busy, 10)
    window.assistant_dock.hide()
    window.resize(1280, 720)
    pump(qapp, timeout=0.2)
    window.show_assistant(True)                              # no room beside the pages at 1280: it floats over them
    pump(qapp, timeout=0.3)
    dock = window.assistant_dock
    assert window.width() <= 1280 and dock.isFloating() and panel.minimumSizeHint().width() <= 340
    assert window.frameGeometry().contains(dock.geometry().topLeft()) and panel.width() >= 300
    dock.hide()
    window.resize(1600, 900)
    pump(qapp, timeout=0.2)
    window.show_assistant(True)                              # room: docked at the right
    pump(qapp, timeout=0.3)
    assert not dock.isFloating() and window.width() <= 1600 and panel.width() >= 300
    window.set_theme('light')
    pump(qapp, timeout=0.1)
    b = [x for x in panel.findChildren(D.Bubble) if x.kind == 'assistant'][-1]
    assert D.MARK['light'] in b.text() and not panel.grab().isNull()
    window.set_theme('dark')
    assert D.MARK['dark'] in b.text()
    html = D.render_html('see job 1 and `x`', ['0.5'], window)
    assert 'falcon://job/1' not in html                      # no job 1 in this session's queue: not a link


def test_ask_and_propose_hooks(window, qapp, srv, reg, tmp_path):
    from falcon_autonomy.assistant import tool_api as T
    from falcon_autonomy.ui import assistant_dock as D
    ws = window.workspace()
    os.makedirs(os.path.join(ws, 'trash'), exist_ok=True)
    os.makedirs(os.path.join(ws, 'runs', 'r9'), exist_ok=True)

    def trash(ctx, run):
        src, dst = os.path.join(ctx.workspace, 'runs', run), os.path.join(ctx.workspace, 'trash', run)
        os.replace(src, dst)
        ctx.emit('trashed', {'from': src, 'to': dst})
        return {'moved': run}
    T.register(T.Tool('runs_trash', 'runs', 'trash', 'Move a run to the workspace trash.',
                      {'type': 'object', 'properties': {'run': {'type': 'string'}}, 'required': ['run']}, trash,
                      preview=lambda ctx, run: f'runs/{run} → trash/{run}'))
    assert not window.assistant.ask('find rainy runs')            # not set up: left in the box, nothing sent
    assert window.assistant_panel.composer.toPlainText() == 'find rainy runs' and srv.requests == []
    panel = ready(window, srv)
    card = window.assistant.propose({'tool': 'runs_trash', 'args': {'run': 'r9'}, 'why': 'It failed to record.'})
    assert isinstance(card, D.ApprovalCard) and card.all is None and 'trash/r9' in card.req.preview
    card.approve.click()
    assert pump(qapp, lambda: os.path.isdir(os.path.join(ws, 'trash', 'r9')) and panel.session.activity, 10)
    assert pump(qapp, lambda: any(b.kind == 'event' for b in panel.findChildren(D.Bubble)), 5)
    e = panel.session.activity[-1]
    assert e['kind'] == 'trash' and panel.undo(e['id']) == 'Restored r9.'
    assert os.path.isdir(os.path.join(ws, 'runs', 'r9'))
    srv.push(text('Searching.'))
    assert window.assistant.ask('find rainy runs', {'page': 'data'})
    assert pump(qapp, lambda: not panel.agent.busy, 10)
    sent = srv.chats()[-1]['body']['messages']
    assert sent[-1]['content'].startswith('find rainy runs') and '"page": "data"' in sent[-1]['content']
    assert any('ran runs_trash from a suggestion' in m['content'] for m in sent if m['role'] == 'user')
