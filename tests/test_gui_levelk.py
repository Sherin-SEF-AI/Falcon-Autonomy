"""Worlds ▸ Traffic profile: the behaviour model (CARLA's traffic manager or level-k road users), the level-k parameters
per kind, and Load calibrated parameters… (a falcon levelk calibrate result), each change one undoable edit; parameters
calibrated on dashcam footage mark the profile private. Offscreen, on test_gui's session application and window."""
import json
import os

import pytest

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)


def _tp(window, ed):
    return window.project.item('traffic_profiles', ed.pid)


def test_behaviour_model_and_level_k_parameters_are_undoable_edits(window, qapp):
    from falcon_autonomy.core import levelk as LK
    from falcon_autonomy.core.project import levelk_params, traffic_model, validate
    page = window.go('worlds')
    ed = page.traffic_editor
    tp = _tp(window, ed)
    assert traffic_model(tp) == 'tm' and ed.model_box.currentData() == 'tm' and not ed.levelk_spins
    n0 = window.undo.count()
    ed.model_box.setCurrentIndex(ed.model_box.findData('levelk'))
    ed.model_box.activated.emit(ed.model_box.currentIndex())
    pump(qapp, timeout=0.2)
    tp = _tp(window, ed)
    assert tp['behaviour']['model'] == 'levelk' and window.undo.count() == n0 + 1
    ed.show_profile(ed.pid)
    assert set(ed.levelk_spins) == {(t, k) for t, _ in ed.LEVELK_KINDS for k in ('k', 'lambda', 'alpha')}
    assert ed.levelk_spins[('two_wheeler', 'alpha')].value() == pytest.approx(LK.PARAM_DEFAULTS['alpha'])
    assert 'lateral filtering is not used' in ed.notes.text()
    w = ed.levelk_spins[('two_wheeler', 'alpha')]
    w.setValue(0.7)
    w.editingFinished.emit()
    pump(qapp, timeout=0.2)
    assert levelk_params(_tp(window, ed)) == {'two_wheeler': {'alpha': 0.7}}
    ed.show_profile(ed.pid)
    w = ed.levelk_spins[('two_wheeler', 'k')]
    w.setValue(2)
    w.editingFinished.emit()
    pump(qapp, timeout=0.2)
    tp = _tp(window, ed)
    assert levelk_params(tp) == {'two_wheeler': {'alpha': 0.7, 'k': 2}} and window.undo.count() == n0 + 3
    assert LK.profile_params(tp)['two_wheeler'] == {'alpha': 0.7, 'level': 2}
    assert validate(window.project.doc) == []
    ed.show_profile(ed.pid)
    w = ed.levelk_spins[('two_wheeler', 'k')]
    w.editingFinished.emit()                                   # unchanged: no edit
    pump(qapp, timeout=0.2)
    assert window.undo.count() == n0 + 3
    for _ in range(3):
        window.undo.undo()
    pump(qapp, timeout=0.2)
    assert traffic_model(_tp(window, ed)) == 'tm' and not levelk_params(_tp(window, ed))
    window.undo.redo()
    assert _tp(window, ed)['behaviour']['model'] == 'levelk'
    ed.show_profile(ed.pid)
    ed.set_model(window.project.doc['traffic_profiles'].index(_tp(window, ed)), 'tm')
    assert _tp(window, ed)['behaviour']['model'] == 'tm'


def test_load_calibrated_parameters_marks_the_profile_private(window, qapp, tmp_path):
    from falcon_autonomy.assistant.privacy import levelk_reason
    from falcon_autonomy.core.project import levelk_params, traffic_model, validate
    page = window.go('worlds')
    ed = page.traffic_editor
    cal = tmp_path / 'levelk' / 'calibration.json'
    os.makedirs(cal.parent)
    block = {'two_wheeler': {'k': 1, 'lambda': 6.0, 'alpha': 0.65, 'v_des': 9.1, 't_gap': 2.3}}
    cal.write_text(json.dumps({'schema': 'falcon.levelk_nearmiss/1', 'worker': 'dashcam-levelk-calibrate',
                               'source': {'format': 'dashcam'},
                               'behaviour': {'model': 'levelk', 'levelk': block,
                                             'levelk_source': {'format': 'dashcam', 'kind': 'nearmiss'}}}))
    n0 = window.undo.count()
    assert ed.load_calibrated(str(cal))
    pump(qapp, timeout=0.2)
    tp = _tp(window, ed)
    assert window.undo.count() == n0 + 1
    assert traffic_model(tp) == 'levelk' and levelk_params(tp) == block
    assert tp['behaviour']['levelk_source']['format'] == 'dashcam' and levelk_reason(tp)
    assert validate(window.project.doc) == []
    ed.show_profile(ed.pid)
    assert 'stays on this machine' in ed.notes.text()
    assert ed.levelk_spins[('two_wheeler', 'alpha')].value() == pytest.approx(0.65)
    window.undo.undo()
    pump(qapp, timeout=0.2)
    assert not levelk_reason(_tp(window, ed)) and traffic_model(_tp(window, ed)) == 'tm'
    # a plain parameters file (no dashcam marker) loads without the mark; a file without parameters is refused
    plain = tmp_path / 'plain.json'
    plain.write_text(json.dumps({'levelk': {'motorcycle': {'k': 2, 'alpha': 0.4}}}))
    assert ed.load_calibrated(str(plain))
    tp = _tp(window, ed)
    assert levelk_params(tp) == {'two_wheeler': {'k': 2, 'alpha': 0.4}} and not levelk_reason(tp)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'something': 1}))
    n1 = window.undo.count()
    assert not ed.load_calibrated(str(bad)) and window.undo.count() == n1
    assert 'Could not load' in ed.notes.text()
    # the calibrated ones, forgotten in one edit
    assert ed.load_calibrated(str(cal))
    i = window.project.doc['traffic_profiles'].index(_tp(window, ed))
    ed.forget_calibrated(i)
    tp = _tp(window, ed)
    assert not levelk_reason(tp) and not levelk_params(tp) and traffic_model(tp) == 'levelk'


def test_replay_drive_a_policy_with_level_k_road_users(window, qapp, tmp_path):
    """Replay ▸ reconstruction: Level-k road users beside Drive a policy here queues the closed loop with
    --actors levelk (and the default stays logged)."""
    pytest.importorskip('torch')
    from test_recon_levelk import NEAR, write_crossing_recon
    from falcon_autonomy.core.recon.scene import Recon
    from falcon_autonomy.ui.pages.replay import ReplayPage
    recon = write_crossing_recon(str(tmp_path / 'recon'), **NEAR)
    page = ReplayPage(window)
    window.stack.addWidget(page)
    panel = page.recon
    panel.device.setCurrentIndex(panel.device.findData('cpu'))
    panel.rec_dir = recon
    panel.rec = Recon.load(recon, 'cpu')
    assert panel.actors.currentData() == 'logged'
    panel.closedloop(str(tmp_path / 'cl-logged'))
    argv = panel.jobs['closedloop'].spec.argv
    assert argv[argv.index('--actors') + 1] == 'logged'
    from falcon_autonomy.core.jobs import FINAL
    first = panel.jobs['closedloop']
    window.queue.cancel(first)
    assert pump(qapp, lambda: first.state in FINAL, 60)
    panel.actors.setCurrentIndex(panel.actors.findData('levelk'))
    d = panel.closedloop(str(tmp_path / 'cl-levelk'))
    argv = panel.jobs['closedloop'].spec.argv
    assert argv[argv.index('--actors') + 1] == 'levelk' and argv[argv.index('--out') + 1] == d
    assert argv[argv.index('closedloop') + 1] == recon
    window.queue.cancel(panel.jobs['closedloop'])
