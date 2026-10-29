"""The command palette's search (runs, scenarios, worlds, rigs, models, evaluations, reports, settings, the manual,
the job history) and the job history and notifications: grouped ranking, opening each kind on its page, keyboard
movement across groups; a finished job kept in the workspace with its log, read back after a restart, opened from
the History tab; notifications decided by the rule and never sent offscreen."""
import os
import sys

import pytest

from falcon_autonomy.core.jobhistory import SCHEMA, JobHistory
from falcon_autonomy.core.jobs import JobSpec
from falcon_autonomy.ui import search as S
from falcon_autonomy.ui.notify import LONG_S, message, should_notify


# ------------------------------------------------------------------------------------------------------- ranking
def _it(kind, title, sub=''):
    return {'kind': kind, 'title': title, 'sub': sub, 'ident': title, 'open': lambda w: None}


def test_rank_groups_best_first_and_drops_scattered_noise():
    items = [_it('command', 'Go to Data'), _it('run', 'drive-town10-0001', 'ok · drive'),
             _it('scenario', 'Lead vehicle brakes'), _it('doc', 'Left-hand traffic: what was verified'),
             _it('setting', 'Render budget', 'runs rendering more per step are refused'),
             _it('doc', 'Data format', 'dataset-format.md')]
    g = S.rank('lead', items)
    assert [k for k, _ in g] == ['scenario']                     # 'Left-hand traffic…' is a scattered match: noise
    g = S.rank('data', items)                                    # a title starting with it beats one containing it
    assert g == [('doc', [items[5]]), ('command', [items[0]])]
    g = S.rank('refused', items)                                 # only in a subtitle: found, at a third
    assert g == [('setting', [items[4]])]
    assert S.rank('', items) == [('command', [items[0]])]        # an empty query: the commands only
    many = [_it('run', f'run-{k:03d}') for k in range(30)]
    assert len(S.rank('run', many)[0][1]) == 8                   # a group shows its best eight


def test_workspace_listings_are_cached_until_the_folder_changes(tmp_path):
    from types import SimpleNamespace
    win = SimpleNamespace(workspace=lambda: str(tmp_path))
    assert S.evaluation_entries(win) == []
    ev = tmp_path / 'evaluations' / '20261029-120000-yolo'
    ev.mkdir(parents=True)
    (ev / 'evaluation.json').write_text('{}')
    first = S.evaluation_entries(win)
    assert [e['ident'] for e in first] == [str(ev)] and S.rank('yolo', first)[0][0] == 'evaluation'
    assert S.evaluation_entries(win) is first                   # cached: the folder did not change
    (tmp_path / 'evaluations' / 'not-one').mkdir()              # the folder changed: listed again
    assert S.evaluation_entries(win) is not first and len(S.evaluation_entries(win)) == 1
    docs = S.doc_entries()
    assert any(e['title'] == 'Lidar labels and their proof' for e in docs)
    assert not any(e['title'].startswith('#') for e in docs)    # code blocks' comment lines are not headings


# ------------------------------------------------------------------------------------------------------- notifications
class _Job:
    def __init__(self, elapsed, state='done', exit_code=0, title='Record a drive'):
        self.elapsed, self.state, self.exit_code = elapsed, state, exit_code

        class _S:
            pass
        self.spec = _S()
        self.spec.title = title


def test_notification_rule():
    long = _Job(LONG_S + 1)
    assert should_notify(long, False, True, 'xcb') == (True, '')
    assert should_notify(long, True, True, 'xcb')[0] is False             # the window is in front
    assert should_notify(long, False, False, 'xcb')[1] == 'turned off in Settings'
    assert should_notify(long, False, True, 'offscreen')[0] is False      # no desktop
    assert should_notify(_Job(LONG_S - 1), False, True, 'wayland')[0] is False
    assert message(_Job(42.4, 'failed', 3))[0] == 'Record a drive failed'
    assert message(_Job(42.4))[1] == 'after 42 s'


def test_history_file_survives_damage_and_trims(tmp_path, monkeypatch):
    import falcon_autonomy.core.jobhistory as JH

    class J:
        def __init__(self, k):
            self.id, self.state, self.exit_code = k, 'done', 0
            self.submitted = self.started = 1_700_000_000.0 + k
            self.finished = self.started + 2.0
            self.elapsed = 2.0
            self.log = [('info', f'line {k}'), ('error', 'bad thing')]
            self.result = {'path': f'/x/report-{k}.html'}

            class Sp:
                kind, title, run_dir = 'report', f'Report {k}', ''
            self.spec = Sp()
    h = JobHistory(str(tmp_path))
    monkeypatch.setattr(JH, 'KEEP', 3)
    for k in range(7):
        rec = h.record(J(k), 'reports')
    assert rec['result'] == '/x/report-6.html' and rec['error'] == '' and rec['page'] == 'reports'
    with open(h.path, 'a', encoding='utf-8') as fh:
        fh.write('{not json\n')
    rows = h.read()
    assert [r['title'] for r in rows] == ['Report 6', 'Report 5', 'Report 4']     # past 2 × 3: the newest 3 kept
    assert all(r['schema'] == SCHEMA for r in rows)
    logs = sorted(os.listdir(os.path.join(h.dir, 'logs')))
    assert len(logs) == 3                                         # the dropped entries' logs went with them
    assert 'line 6' in h.log_text(rows[0]) and 'error: bad thing' in h.log_text(rows[0])
    assert JobHistory('').record(J(9)) is None and JobHistory('').read() == []


# ------------------------------------------------------------------------------------------------------- in the app
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


@pytest.mark.gui
def test_palette_opens_each_kind_on_its_page(window, qapp, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from falcon_autonomy.ui.palette import CommandPalette
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.save()
    window.page('worlds').add_world()
    doc = window.project.doc
    doc['worlds'][1]['name'] = 'Harbour district'
    doc['rigs'][0]['name'] = 'Roof rig'
    window.projectChanged.emit(None)

    def pick(query, kind):
        pal = CommandPalette(window)
        pal.input.setText(query)
        for i in range(pal.list.count()):
            e = pal.list.item(i).data(Qt.ItemDataRole.UserRole)
            if e['kind'] == kind:
                pal.list.setCurrentRow(i)
                pal.run_current()
                return e
        raise AssertionError(f'{query!r}: no {kind} among {[pal.list.item(i).text() for i in range(pal.list.count())]}')
    pick('cut-in', 'scenario')
    sc = window.page('scenarios')
    assert window.stack.currentWidget() is sc and sc.current()[1] == 'cut_in'
    pick('harbour', 'world')
    assert window.stack.currentWidget().key == 'worlds' and window.page('worlds').list.currentRow() == 1
    pick('roof', 'rig')
    assert window.stack.currentWidget().key == 'rigs'
    st = window.page('settings')
    pick('render budget', 'setting')
    assert window.stack.currentWidget() is st
    assert any(t == 'Render budget' for t, _n, _f in st.fields) and st.focus_field('Render budget')
    pick('job notifications', 'setting')
    e = pick('Lidar labels', 'doc')
    hp = window.page('help')
    assert window.stack.currentWidget() is hp and hp.list.currentItem().data(256) == e['ident'][0]
    assert hp.view.textCursor().selectedText() == e['title']           # scrolled to the heading
    pal = CommandPalette(window)
    pal.input.setText('a')
    groups = [pal.list.item(i).data(Qt.ItemDataRole.UserRole + 1) for i in range(pal.list.count())]
    assert sum(g is not None for g in groups) >= 3                      # several groups, each headed once
    QTest.keyClick(pal, Qt.Key.Key_Down)
    QTest.keyClick(pal, Qt.Key.Key_PageDown)
    assert pal.list.currentRow() == 9
    assert pal.build_s < 2.0
    pal.list.resize(600, 400)
    assert not pal.grab().isNull()
    pal.input.setText('zzqqxx')
    assert pal.list.count() == 0 and 'Nothing matches' in pal.hint.text()
    pal.close()


@pytest.mark.gui
def test_job_history_kept_opened_and_searched(window, qapp, project, tmp_path):
    from PyQt6.QtCore import Qt
    from falcon_autonomy.ui.palette import CommandPalette
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.save()
    window.set_project(window.project)                    # the workspace now beside the project file
    hist = window.history
    assert hist.path.startswith(window.workspace()) and hist.read() == []
    data = window.go('data')
    data.record.seconds.setValue(0.5)
    data.record.record()
    job = window.queue.jobs[-1]
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120), job.log
    bad = window.queue.submit(JobSpec('check', 'A job that fails', [sys.executable, '-c',
                                      "import sys; print('it broke', file=sys.stderr); sys.exit(2)"]))
    assert pump(qapp, lambda: bad.state in ('done', 'failed', 'cancelled'), 30)
    pump(qapp, timeout=0.2)
    rows = JobHistory(window.workspace()).read()          # read back as after a restart
    assert [r['title'] for r in rows] == ['A job that fails', job.spec.title]
    assert rows[0]['status'] == 'failed' and rows[0]['exit_code'] == 2 and 'it broke' in rows[0]['error']
    assert rows[1]['status'] == job.state and rows[1]['result'] == job.spec.run_dir and rows[1]['kind'] == 'drive'
    assert rows[1]['page'] == 'data'
    assert job.spec.title in hist.log_text(rows[1]) and 'it broke' in hist.log_text(rows[0])
    panel = window.jobs_dock.widget().history
    assert panel.tree.topLevelItemCount() == 2 and window.jobs_dock.widget().tabs.tabText(1) == 'History'
    panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
    assert panel.log_btn.isEnabled()
    dlg = panel.show_log(panel.tree.currentItem())
    from PyQt6.QtWidgets import QPlainTextEdit
    assert 'it broke' in dlg.findChild(QPlainTextEdit).toPlainText()
    dlg.close()
    panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
    window.go('home')
    panel.open(panel.tree.currentItem())
    assert window.stack.currentWidget() is data and data.browse.run_dir == job.spec.run_dir
    assert window.notifier.sent == []                     # offscreen: never a notification
    window.go('home')
    pal = CommandPalette(window)
    pal.input.setText('A job that fails')
    kinds = [pal.list.item(i).data(Qt.ItemDataRole.UserRole)['kind'] for i in range(pal.list.count())]
    assert 'job' in kinds
    name = next(r['name'] for r in window.service.catalog.list('run') if r['path'] == job.spec.run_dir)
    pal.input.setText(name)
    e = pal.list.item(0).data(Qt.ItemDataRole.UserRole)
    assert e['kind'] == 'run' and e['ident'] == job.spec.run_dir
    pal.run_current()
    assert window.stack.currentWidget() is data
    st = window.go('settings')
    assert st.notify.isChecked()
    st.notify.setChecked(False)
    st.apply()
    assert window.settings.get('notify_jobs') is False
