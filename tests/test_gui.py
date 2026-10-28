"""The application offscreen: navigation, the command palette, undo and redo, recording and browsing a drive
through the GUI's job path, cancel, save and open, themes. The window fixture waits for every background worker and
task before the window goes (an analysis thread emitting into a deleted window segfaults)."""
import json
import os
import time

import pytest
from PyQt6.QtCore import Qt

pytestmark = pytest.mark.gui

from conftest import FAKE  # noqa: E402


@pytest.fixture(scope='session')
def qapp():
    from falcon_autonomy.ui.app import make_app
    from falcon_autonomy.ui.theme import THEME, load_fonts
    from PyQt6.QtCore import QThreadPool
    app = make_app()
    load_fonts()
    THEME.apply(app, 'dark')
    yield app
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()


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
    """An exception raised in a Qt slot (printed, or fatal in a frozen bundle) fails the test."""
    import sys
    seen = []
    monkeypatch.setattr(sys, 'excepthook', lambda t, v, tb: seen.append((t, v)))
    yield seen
    assert not seen, f'exceptions in Qt slots: {seen}'


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
    assert keys == ['loop', 'worlds', 'live', 'rigs', 'scenarios', 'data', 'rarecases', 'models', 'training',
                    'stack', 'replay', 'hardware', 'reports', 'settings', 'help']
    assert window.stack.currentWidget().key == 'loop'           # the home page
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
    status = window.page('loop').status                 # the Loop page's Status tab (what Home showed)
    status.refresh()
    assert status.runs.topLevelItemCount() == 1


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


def test_import_a_place(window, qapp, tmp_path):
    from falcon_autonomy.core import osm as O
    from falcon_autonomy.ui.osmimport import ImportDialog
    src = os.path.join(os.path.dirname(__file__), 'data', 'blr_small.osm')
    dlg = ImportDialog(window, src)
    assert pump(qapp, lambda: dlg.data is not None, 15)
    assert dlg.picker.lines and 'roads' in dlg.report.vals['State'].text()
    dlg.size.setValue(200)
    dlg.bbox_from_centre()
    assert dlg.picker.rect_m is not None and 'N 12.9' in dlg.bbox_lbl.text()
    dlg.convert()
    assert pump(qapp, lambda: dlg.job.state in ('done', 'failed', 'cancelled'), 60), dlg.job.log
    assert dlg.job.state == 'done', dlg.job.log
    assert dlg.add_btn.isEnabled() and dlg.graph.lines
    assert 'left-hand' in dlg.report.vals['Traffic'].text()
    assert 'agree' in dlg.report.vals['Lanes against OSM'].text()
    n = len(window.project.doc['worlds'])
    dlg.add_world()
    worlds = window.project.doc['worlds']
    assert len(worlds) == n + 1 and worlds[-1]['kind'] == 'osm' and worlds[-1]['osm']['handedness'] == 'LHT'
    from falcon_autonomy.core.project import validate
    assert validate(window.project.doc) == []
    page = window.go('worlds')
    page.list.setCurrentRow(page.list.count() - 1)
    assert page.place.isVisibleTo(page) and page.graph.lines and not page.town.isEnabled()
    assert 'not verified' in page.verify_kv.vals['Handedness'].text()
    page.verify()
    assert pump(qapp, lambda: page.verify_job.state in ('done', 'failed'), 120), page.verify_job.log
    # the fake keeps its own right-hand road whatever the file says: the verification reports what it measured
    assert 'RHT' in page.verify_kv.vals['Handedness'].text()
    window.undo.undo()
    assert len(window.project.doc['worlds']) == n


def test_rig_editor(window, qapp):
    page = window.go('rigs')
    assert page.view.backend == 'software'                    # offscreen: the software 3D view
    rig = window.project.doc['rigs'][0]
    page.add_sensor('lidar')
    assert rig['sensors'][-1]['type'] == 'lidar' and page.sensors.count() == 2
    page.sensors.setCurrentRow(0)
    page.set_pose(0, {'yaw': 25.0})
    assert rig['sensors'][0]['pose']['yaw'] == 25.0
    page.view.poseDropped.emit(rig['sensors'][0]['id'], dict(rig['sensors'][0]['pose'], x=2.0, y=0.4))
    assert rig['sensors'][0]['pose']['x'] == 2.0
    window.undo.undo()
    assert rig['sensors'][0]['pose']['x'] == 1.5
    page.toggle_label(0, 'instance', True)
    assert rig['sensors'][0]['labels'] == ['instance', 'semantic']
    from falcon_autonomy.core import budget
    assert f'{budget.rig_mpx(rig):.1f} of 200 Mpx' in page.budget_lbl.text()
    page.set_field(0, 'width', 8192)
    page.set_field(0, 'height', 8192)
    assert 'over the render budget' in page.problems.text()
    page.set_faults(0, [{'kind': 'noise', 'sigma': 3.0}])
    assert rig['sensors'][0]['faults'][0]['kind'] == 'noise'
    window.undo.undo()
    page.view.resize(500, 400)
    img = page.view.grab()
    assert not img.isNull()
    for _ in range(5):
        window.undo.undo()
    assert len(rig['sensors']) == 1


def test_datasets_tab_and_browser_layers(window, qapp, tmp_path):
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.save()
    window.service.set_workspace(window.workspace())
    window.project.doc['rigs'][0]['sensors'][0]['labels'] = ['semantic', 'boxes2d']
    data = window.go('data')
    tab = data.datasets
    tab.add_spec()
    tab.set('drive_s', 0.5)
    tab.set('record_every', 2)
    tab.set('targets', {'car': 10})
    tab.make_plan()
    assert tab.run_btn.isEnabled() and 'heuristic' in tab.plan_kv.vals['Yield'].text()
    assert 'typical' in tab.plan_kv.vals['Cost'].text()
    tab.run()
    jobs = [j for j in window.queue.jobs if j.spec.kind == 'drive']
    assert jobs and pump(qapp, lambda: all(j.state in ('done', 'failed') for j in jobs), 120)
    assert all(j.state == 'done' for j in jobs), [j.log for j in jobs]
    _, spec = tab.current()
    tab.show_coverage(spec)
    assert pump(qapp, lambda: tab.bars.rows and tab.bars.rows[0][1] > 0, 20)
    tab.new_version()
    assert tab.versions.count() == 1 and 'v1' in tab.versions.item(0).text()
    run_dir = jobs[0].spec.run_dir
    data.show_run(run_dir)
    b = data.browse
    b.layer.setCurrentText('boxes')
    assert pump(qapp, lambda: b.objects.topLevelItemCount() > 0, 20)
    b.layer.setCurrentText('range')
    assert pump(qapp, lambda: b.cache is not None and b.cache[1] is not None, 20)


def test_scenario_studio_and_rare_cases(window, qapp, tmp_path):
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    window.save()
    window.service.set_workspace(window.workspace())
    page = window.go('scenarios')
    assert page.list.count() == 11 and not page.editable()
    page.list.setCurrentRow(0)
    page.copy_builtin()
    assert page.editable() and window.project.doc['scenarios'][0]['name'].endswith('(copy)')
    page.set_actor_field(0, 'ahead_m', 25.0)
    assert window.project.doc['scenarios'][0]['actors'][0]['place']['ahead_m'] == 25.0
    window.undo.undo()
    page.sketch.resize(600, 260)
    assert not page.sketch.grab().isNull()
    page.samples.setValue(2)
    page.sweep()
    job = window.queue.jobs[-1]
    assert pump(qapp, lambda: job.state in ('done', 'failed'), 300), job.log
    assert job.state == 'done', job.log
    page.load_results()
    assert page.results.topLevelItemCount() == 2
    rc = window.go('rarecases')
    rc.scenario.setCurrentIndex(rc.scenario.findData('lead_brake'))
    rc.budget.setValue(4)
    rc.batch.setValue(2)
    rc.start()
    assert pump(qapp, lambda: rc.job.state in ('done', 'failed'), 300), rc.job.log
    assert rc.job.state == 'done', rc.job.log
    rc.load()
    assert len(rc.curve.values) == 4 and rc.px.count() >= 2
    rc.heat.resize(400, 300)
    assert not rc.heat.grab().isNull()


def test_models_page_evaluates_compares_and_measures_the_gap(window, qapp, tmp_path):
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.project.doc['rigs'][0]['sensors'][0].update(width=320, height=180,
                                                       labels=['semantic', 'instance', 'depth', 'boxes2d'])
    window.save()
    window.service.set_workspace(window.workspace())
    data = window.go('data')
    data.record.seconds.setValue(1.0)
    data.record.record()
    job = window.queue.jobs[-1]
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120), job.log
    assert job.state == 'done', job.log
    page = window.go('models')
    page.add_synthetic()
    assert page.models.topLevelItemCount() == 1 and page.models.topLevelItem(0).text(4) == 'plug-in'
    assert window.project.doc['models'][0]['entry'].endswith(':SyntheticDetector')
    assert page.runs.count() == 1
    page.runs.item(0).setCheckState(Qt.CheckState.Checked)
    page.boot.setValue(50)

    def run_job(fn):
        fn()
        j = window.queue.jobs[-1]
        assert pump(qapp, lambda: j.state in ('done', 'failed', 'cancelled'), 180), j.log
        assert j.state == 'done', j.log
        assert pump(qapp, lambda: j.id not in page.jobs, 5)
        return j
    j = run_job(page.evaluate)
    assert not j.spec.needs_gpu                       # a plug-in: no GPU lock
    page.every.setValue(4)
    run_job(page.evaluate)
    assert page.ev.count() == 2 and page.table.topLevelItemCount() > 4
    assert 'SYNTHETIC' in page.ev_info.vals['Note'].text()
    page.cmp_a.setCurrentIndex(1)
    page.cmp_b.setCurrentIndex(0)
    run_job(page.compare)
    assert page.cmp.count() == 1 and page.cmp_table.topLevelItemCount() > 4
    run_dir = page.runs.item(0).data(256)
    page.real_kind.setCurrentIndex(page.real_kind.findData('folder'))
    page.real_path.setText(os.path.join(run_dir, 'rgb'))
    page.gap_images.setValue(20)
    page.match.setChecked(True)
    run_job(page.gap)
    assert page.gapsel.count() == 1 and page.gap_table.topLevelItemCount() == 11
    assert page.gap_info.vals['Fréchet distance'].text() == 'not measured'
    for w in (page.gridview, page.frame):
        w.resize(400, 260)
        assert not w.grab().isNull()
    if page.gal.topLevelItemCount():
        page.gal.setCurrentItem(page.gal.topLevelItem(0))
        assert page.frame.pixmap() is not None and not page.frame.pixmap().isNull()


def test_training_page_clones_scores_and_replays(window, qapp, tmp_path):
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.project.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=['semantic'])
    window.save()
    window.service.set_workspace(window.workspace())
    data = window.go('data')
    data.record.seconds.setValue(2.0)
    for _ in range(2):                    # behaviour cloning splits by drive: two at least
        data.record.record()
        job = window.queue.jobs[-1]
        assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120), job.log
        assert job.state == 'done', job.log
        time.sleep(1.1)                   # run folders are named by the second
    page = window.go('training')
    assert page.det_runs.count() == page.bc_runs.count() == 2

    def run_job(fn, timeout=300):
        n = len(window.queue.jobs)
        fn()
        assert len(window.queue.jobs) == n + 1
        j = window.queue.jobs[-1]
        assert pump(qapp, lambda: j.state in ('done', 'failed', 'cancelled'), timeout), j.log
        assert j.state == 'done', j.log
        assert pump(qapp, lambda: j.id not in page.jobs, 5)
        return j
    # detectors: the job it would submit (not run here: it would use the GPU)
    captured = []
    page._submit = lambda spec: captured.append(spec)
    page.det_runs.item(0).setCheckState(Qt.CheckState.Checked)
    page.train()
    spec = captured[0]
    assert spec.needs_gpu and spec.env['FALCON_GPU_LOCK_HELD'] == '1'
    assert spec.argv[3:5] == ['train', '--workspace'] and f'falcon:{page.det_runs.item(0).data(256)}' in spec.argv
    del page._submit
    assert page.weights.topLevelItemCount() == len(__import__('falcon_autonomy.core.training',
                                                              fromlist=['WEIGHTS']).WEIGHTS)
    # a policy from the drive, scored on one scenario route, and an infraction replayed if it made one
    for i in range(2):
        page.bc_runs.item(i).setCheckState(Qt.CheckState.Checked)
    page.bc_epochs.setValue(3)
    run_job(page.bc)
    assert page.policies.topLevelItemCount() == 1 and page.lb_policy.count() == 1
    page.lb_seeds.setValue(1)
    page.lb_free.setValue(1)
    k = next(i for i in range(page.lb_scn.count()) if page.lb_scn.item(i).data(256) == 'lead_brake')
    page.lb_scn.item(k).setCheckState(Qt.CheckState.Checked)
    run_job(page.score_policy)
    assert page.lbs.count() == 1 and page.infr.topLevelItemCount() >= 2
    assert page.lb_info.vals['Driving score'].text()[0].isdigit()
    rows = [page.infr.topLevelItem(i) for i in range(page.infr.topLevelItemCount())]
    with_inf = [r for r in rows if r.data(0, 256) is not None]
    if with_inf:
        page.infr.setCurrentItem(with_inf[0])
        run_job(page.replay)
        page.flip.resize(400, 260)
        assert page.flip.frames and not page.flip.grab().isNull()


def test_loop_page_runs_steps_and_shows_them(window, qapp, tmp_path):
    from falcon_autonomy.core import loop as L
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.project.doc['rigs'][0]['sensors'][0].update(width=160, height=96,
                                                       labels=['semantic', 'instance', 'depth', 'boxes2d'])
    window.project.doc['traffic_profiles'][0].update(vehicles=10, walkers=10)
    window.save()
    window.service.set_workspace(window.workspace())
    page = window.go('loop')
    assert page.tabs.tabText(1) == 'Status' and page.status.runs is not None
    assert 'pending' in page.steps.topLevelItem(0).text(2) and page.chart.rows == []
    page.use_example()                              # the default project lacks what the example needs: said so
    assert 'Bengaluru' in page.odd_kv.vals['Problems'].text()
    window.undo.undo()
    window.go('models').add_synthetic()
    page = window.go('loop')
    assert page.model.count() == 1
    page.model.setCurrentIndex(0)
    page.set_model()
    odd = {'id': 'gui', 'name': 'GUI test', 'rig': 'rig-default', 'worlds': ['world-town10'],
           'traffic_profile': 'tp-default', 'weathers': [{'name': 'clear', 'preset': 'ClearNoon'}],
           'classes': ['car', 'pedestrian'], 'range_bins': [0, 20, 40],
           'evaluation': {'drives_per_cell': 2, 'drive_s': 1.0, 'record_every': 2, 'n_boot': 50},
           'weak_cells': {'min_objects': 3, 'max_cells': 1, 'fallback': 'lowest'}}
    page.set_odd(odd)
    assert window.project.doc['loop']['model'] == window.project.doc['models'][0]['id']
    assert page.odd_kv.vals['Problems'].text() == 'none'
    assert pump(qapp, lambda: page.next_cost.text().startswith('Next:'), 10)
    assert 'step 1' in page.next_cost.text()
    page.until.setValue(3)
    job = page.run_until()
    assert job is not None and job.spec.argv[3:7] == ['loop', window.project.path, 'run', '--until']
    assert not page.run_next.isEnabled() and page.run_until() is None          # one loop job at a time
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 240), job.log
    assert job.state == 'done', job.log
    assert pump(qapp, lambda: page.it is not None and page.it['n'] == 1, 5)
    assert [page.steps.topLevelItem(i).text(2) for i in range(8)] == ['done'] * 3 + ['pending'] * 5
    assert 'measured' in page.steps.topLevelItem(1).toolTip(3) or 'typical' in page.steps.topLevelItem(1).toolTip(3)
    assert page.cells.topLevelItemCount() > 0 and page.hist.topLevelItemCount() == 1
    ents = window.project.doc['loop']['iterations']                 # recorded in the project as undoable edits
    assert len(ents) == 1 and ents[0]['n'] == 1 and window.project.dirty
    assert pump(qapp, lambda: 'step 4' in page.next_cost.text(), 10), page.next_cost.text()
    # the chart, from a before and after with intervals
    rows = [{'label': 'AP, all', 'before': {'value': 0.4, 'ci_low': 0.3, 'ci_high': 0.5},
             'after': {'value': 0.6, 'ci_low': 0.5, 'ci_high': 0.7}, 'diff': {}, 'verdict': 'improvement'}]
    page.chart.set_rows(rows)
    page.chart.resize(600, 200)
    assert not page.chart.grab().isNull()
    reports = window.page('reports')
    page.it = L.load_iteration(window.workspace(), 1)
    page.report()
    assert window.stack.currentWidget() is reports
    rj = window.queue.jobs[-1]
    assert pump(qapp, lambda: rj.state in ('done', 'failed'), 60), rj.log
    assert rj.state == 'done', rj.log
    assert pump(qapp, lambda: reports.list.topLevelItemCount() == 1, 5)
    out = reports.export_to(str(tmp_path / 'exported'), [reports.list.topLevelItem(0).data(0, 256)])
    assert len(out) == 1 and open(out[0], encoding='utf-8').read().startswith('<!doctype html>')
    reports.kind.setCurrentIndex(reports.kind.findData('evaluation'))
    assert reports.src.count() == 1 and reports.src.itemText(0) == 'loop 001 before'
    window.set_theme('light')
    assert not page.chart.grab().isNull()
    window.set_theme('dark')
