"""The application offscreen: navigation, the command palette, undo and redo, recording and browsing a drive
through the GUI's job path, cancel, save and open, themes. The window fixture waits for every background worker and
task before the window goes (an analysis thread emitting into a deleted window segfaults)."""
import json
import os
import time

import pytest

pytestmark = pytest.mark.gui

from conftest import FAKE  # noqa: E402


@pytest.fixture(scope='session')
def qapp():
    from falcon_autonomy.ui.app import make_app
    from falcon_autonomy.ui.theme import THEME, load_fonts
    app = make_app()
    load_fonts()
    THEME.apply(app, 'dark')
    return app


def pump(app, cond=lambda: False, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@pytest.fixture
def window(qapp, tmp_path, project):
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.ui import tasks
    from falcon_autonomy.ui.mainwindow import MainWindow
    win = MainWindow(Settings(str(tmp_path / 'settings.json')), project, fake_carla=FAKE)
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


def test_pages_and_navigation(window, qapp):
    keys = [p.key for p in window.pages]
    assert keys == ['home', 'worlds', 'live', 'data', 'settings', 'help']
    assert window.go('data') is window.stack.currentWidget()
    window.actions_['page_worlds'].trigger()
    assert window.stack.currentWidget().key == 'worlds'


def test_command_palette_filters_and_runs(window, qapp):
    from falcon_autonomy.ui.palette import CommandPalette, score
    assert score('gtd', 'Go to Data') is not None and score('xyz', 'Go to Data') is None
    assert score('data', 'Go to Data') > score('gtd', 'Go to Data')
    pal = CommandPalette(window)
    pal.input.setText('go to data')
    assert pal.list.item(0).text().startswith('Go to Data')
    pal.run_current()
    assert window.stack.currentWidget().key == 'data'


def test_undo_redo_and_slider_drags_merge(window, qapp):
    page = window.go('worlds')
    doc = window.project.doc
    page.set_preset('HardRainNoon')
    assert doc['worlds'][0]['weather'] == {'preset': 'HardRainNoon'} and window.project.dirty
    s = page.sliders['fog_density']
    for v in (10, 20, 30, 40):                     # a drag
        s.setValue(v)
        page.set_weather('fog_density', float(v), merge=True)
    assert doc['worlds'][0]['weather']['fog_density'] == 40
    assert window.undo.count() == 2                # the preset, and the whole drag as one step
    window.undo.undo()
    assert doc['worlds'][0]['weather'] == {'preset': 'HardRainNoon'}
    assert page.sliders['fog_density'].value() == 7           # the preset's fog, shown again
    window.undo.undo()
    assert doc['worlds'][0]['weather'] == {'preset': 'ClearNoon'}
    window.undo.redo()
    window.undo.redo()
    assert doc['worlds'][0]['weather']['fog_density'] == 40
    page.add_world()
    assert len(doc['worlds']) == 2 and page.list.count() == 2
    window.undo.undo()
    assert len(doc['worlds']) == 1


def test_record_and_browse_a_drive(window, qapp):
    data = window.go('data')
    rec = data.record
    rec.seconds.setValue(1.0)
    assert 'typical' in rec.cost.vals['Estimate'].text()
    rec.record()
    job = window.queue.jobs[-1]
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120), job.log
    assert job.state == 'done', job.log
    assert pump(qapp, lambda: data.browse.runs.topLevelItemCount() == 1, 5)
    data.show_run(job.spec.run_dir)
    b = data.browse
    assert b.slider.maximum() == 19 and b.sensor.currentText() == 'cam-front'
    assert pump(qapp, lambda: b.cache is not None, 10)
    assert b.legend_grid.count() >= 6 and 'Measured' in b.info.vals
    b.opacity.setValue(0)
    b.slider.setValue(10)
    assert pump(qapp, lambda: b.cache is not None and b.view.pixmap() is not None, 10)
    home = window.page('home')
    home.refresh()
    assert home.runs.topLevelItemCount() == 1


def test_cancel_from_the_jobs_dock(window, qapp, project):
    from falcon_autonomy.core.drive import drive_config
    cfg = drive_config(project, 'world-town10', 'rig-default', os.path.join(window.service.runs_dir(), 'long'),
                       2310, 8310, seconds=120.0)
    job = window.service.submit_drive(cfg)
    assert pump(qapp, lambda: 'step' in (job.text or ''), 60)
    window.queue.cancel(job)
    assert pump(qapp, lambda: job.state == 'cancelled', 60), (job.state, job.log)
    assert sorted(os.listdir(job.spec.run_dir)) == ['drive.json', 'run.json']       # the run's input is kept
    assert json.load(open(os.path.join(job.spec.run_dir, 'run.json')))['status'] == 'cancelled'


def test_save_and_open(window, qapp, tmp_path):
    window.go('worlds').set_preset('ClearNight')
    path = str(tmp_path / 'saved.falcon.json')
    window.project.path = path
    assert window.save() and not window.project.dirty
    assert '●' not in window.windowTitle()
    window.project.dirty = False
    assert window.open_project(path)
    assert window.project.doc['worlds'][0]['weather']['preset'] == 'ClearNight'
    assert window.settings.get('recent_projects')[0] == path
    assert window.workspace().endswith('saved.falcon')


def test_themes(window, qapp):
    window.set_theme('light')
    assert window.settings.get('theme') == 'light'
    window.set_theme('dark')


def test_settings_reserved_port_needs_confirmation(window, qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    page = window.go('settings')
    page.port.setValue(2000)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.No)
    page.apply()
    assert window.settings.get('rpc_port') == 2310
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)
    page.apply()
    assert window.settings.get('rpc_port') == 2000 and window.settings.get('allowed_ports') == [2000]
    assert window.service.external


def test_help_is_searchable(window, qapp):
    page = window.go('help')
    assert page.list.count() >= 3
    page.search.setText('exit code')
    assert page.list.count() >= 1 and '(' in page.list.item(0).text()


def test_wizard_pages_load(window, qapp):
    from falcon_autonomy.ui.wizard import FirstRunWizard, MachinePage
    wz = FirstRunWizard(window)
    mp = next(wz.page(i) for i in wz.pageIds() if isinstance(wz.page(i), MachinePage))
    mp.initializePage()
    assert pump(qapp, lambda: 'Memory' in mp.kv.vals, 15)
    wz.done(0)


def test_live_session_in_the_gui(window, qapp):
    page = window.go('live')
    page.size.addItem('96 × 64')
    from falcon_autonomy.ui.pages import live as L
    L.VIEW_SIZES['96 × 64'] = (96, 64)
    page.size.setCurrentText('96 × 64')
    page.mode_btns['manual'].setChecked(True)
    page.mode_btns['autopilot'].setChecked(False)
    page.start()
    assert pump(qapp, lambda: page.view.img is not None and page.shown > 5, 60), (page.job.state, page.job.log)
    assert page.top.map is not None and page.top.actors
    assert 'steps/s' in page.stats.vals['Simulation'].text()
    from PyQt6.QtCore import Qt
    page.view.keys.add(Qt.Key.Key_W)
    assert pump(qapp, lambda: (page.view.hdr.get('control') or {}).get('throttle') == 0.7, 20)
    page.view.keys.clear()
    page.rec_btn.setChecked(True)
    page.toggle_record()
    page.fire_event('pedestrian', distance_m=20.0, side='right')
    assert pump(qapp, lambda: page.view.hdr.get('recording'), 20)
    pump(qapp, timeout=1.0)
    page.rec_btn.setChecked(False)
    page.toggle_record()
    assert pump(qapp, lambda: page.recordings, 20)
    page.stop()
    assert pump(qapp, lambda: page.job.state == 'done', 60), page.job.log
    rows = window.service.catalog.list('run')
    assert any(r['path'] == page.recordings[0] for r in rows)
    data = window.go('data')
    data.show_run(page.recordings[0])
    assert data.browse.sensor.currentText() == 'cam-front'
