"""4.2 groundwork: core.fidelity, the shared widgets (fidelity chip, ids, command block, details, action callout) and
ui.context (the run context and Send to)."""
import json
import os
import sys

import pytest

from falcon_autonomy.core import fidelity as F

HERE = os.path.dirname(__file__)


def test_levels_and_lookup(tmp_path):
    assert F.label('fake') == 'Fake simulator' and F.tone('fake') == 'warning'
    assert F.label('nonsense') == F.label('unknown')
    assert F.of_doc({'provenance': {'fidelity': 'carla'}}) == 'carla'
    assert F.of_doc({'fidelity': 'kinematic'}) == 'kinematic'
    assert F.of_doc({'provenance': {'fidelity': 'bogus'}}) == 'unknown'
    assert F.of_doc(None) == 'unknown'
    run = tmp_path / 'r1'
    run.mkdir()
    (run / 'run.json').write_text(json.dumps({'provenance': {'fidelity': 'fake'}}))
    assert F.of_run(str(run)) == 'fake'
    assert F.of_run(str(run / 'run.json')) == 'fake'
    assert F.of_run(str(tmp_path / 'missing')) == 'unknown'


def test_combine_and_comparable():
    assert F.combine([]) == 'unknown'
    assert F.combine(['carla', 'carla', 'unknown']) == 'carla'
    assert F.combine(['carla', 'fake']) == 'mixed'
    assert F.comparable('carla', 'carla') == (True, '')
    ok, why = F.comparable('carla', 'fake')
    assert not ok and 'Fake simulator' in why
    ok, why = F.comparable('carla', 'unknown')
    assert ok and why
    assert not F.comparable('mixed', 'mixed')[0]


def test_of_module_reads_the_fake_flag():
    sys.path.insert(0, os.path.join(HERE, 'fake_carla'))
    try:
        sys.modules.pop('carla', None)
        import carla
        assert carla.FALCON_FAKE and F.of_module(carla) == 'fake'
    finally:
        sys.path.pop(0)
        sys.modules.pop('carla', None)

    class Real:
        __version__ = '0.9.16'

    class UE5:
        __version__ = '0.10.0'
    assert F.of_module(Real) == 'carla' and F.of_module(UE5) == 'carla-ue5'


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_widgets(qapp):
    from PyQt6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem

    from falcon_autonomy.ui import widgets as W
    c = W.fidelity_chip('fake')
    assert c.text() == 'Fake simulator' and c.property('tone') == 'warning' and 'test double' in c.toolTip()
    assert W.short_id('abcdefghijklmnopqrstuvwxyz', 4) == 'abcd…wxyz'
    assert W.short_id('short') == 'short'
    lb = W.IdLabel('20261029-105219-live-0123456789abcdef', keep=6)
    assert '…' in lb.text() and lb.full.endswith('abcdef')
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(1, 1), QPointF(1, 1), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    lb.mousePressEvent(ev)
    assert QApplication.clipboard().text() == lb.full

    t = QTreeWidget()
    t.setColumnCount(2)
    it = QTreeWidgetItem(['x' * 40, 'value'])
    it.setData(0, Qt.ItemDataRole.UserRole + 1, 'the-full-id')
    t.addTopLevelItem(it)
    W.copyable_ids(t, 0)
    t.itemClicked.emit(it, 0)
    assert QApplication.clipboard().text() == 'the-full-id'
    t.itemClicked.emit(it, 1)                      # not an id column: nothing copied
    assert QApplication.clipboard().text() == 'the-full-id'
    W.fidelity_item(it, 1, 'kinematic')
    assert it.text(1) == 'Kinematic'

    cb = W.CommandBlock('falcon monitor deploy pkg --dry-run', 'what it does')
    cb.copy_btn.click()
    assert QApplication.clipboard().text() == 'falcon monitor deploy pkg --dry-run'
    d = W.Details('a long explanation')
    assert not d.expanded and d.body.isHidden()
    d.toggle_btn.click()
    assert d.expanded and 'Details ▾' == d.toggle_btn.text()
    hits = []
    ac = W.action_callout('Value not verified', 'warning', [('Jump to config', lambda: hits.append(1), 'tip')])
    ac.buttons[0].click()
    assert hits == [1] and ac.buttons[0].toolTip() == 'tip'


def test_context_and_send_to(qapp):
    from PyQt6.QtCore import QObject

    from falcon_autonomy.ui import context as C

    class Win(QObject):
        pass
    win = Win()
    ctx = C.context_of(win)
    seen = []
    ctx.changed.connect(seen.append)
    ctx.set(run='/w/runs/20261029-live', seed=3)
    assert ctx.get('seed') == '3' and ctx.pill_text('run') == 'Run 20261029-live'
    ctx.set(seed=3)                                  # unchanged: no signal
    assert len(seen) == 1
    ctx.clear('seed')
    assert 'seed' not in ctx.values and len(seen) == 2
    with pytest.raises(KeyError):
        ctx.set(colour='red')
    got = []
    C.register_target(win, 'replay', 'Replay', got.append)
    C.register_target(win, 'data', 'Data', got.append)
    m = C.send_to_menu(win, '/w/runs/r2', exclude='data')
    assert [a.text() for a in m.actions()] == ['Replay']
    m.actions()[0].trigger()
    assert got == ['/w/runs/r2'] and ctx.get('run') == '/w/runs/r2'
    ctx.clear()
    assert ctx.values == {} and seen[-1] == {}
