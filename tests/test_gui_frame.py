"""The lab-console frame offscreen: the top bar, the navigator's counts and sliding selection, the shared page header,
the building blocks (cards, headline tiles, meters, interval bars, capped form fields) and the animations — each
one runs and leaves nothing behind, and Reduce motion turns them all off."""
import pytest

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)


@pytest.fixture
def motion_on():
    from falcon_autonomy.ui import motion
    was = motion.enabled()
    motion.set_enabled(True)
    yield motion
    motion.set_enabled(was)


def test_top_bar_navigator_counts_and_header(window, qapp, project):
    from falcon_autonomy.ui.frame import COUNT_ROLE
    top = window.top
    assert top.project.text() == project.doc['name'] and window.menuBar() is window._menubar
    assert [a.text() for a in window.menuBar().actions()][:2] == ['&File', '&Edit']
    counts = {window.pages[i].key: window.nav.item(r).data(COUNT_ROLE) for r, i in window._nav_rows()}
    assert counts['worlds'] == len(project.doc['worlds']) and counts['rigs'] == len(project.doc['rigs'])
    assert counts['live'] == '' and counts['data'] == 0           # no runs in the test project
    for key in ('worlds', 'models', 'training', 'settings'):
        page = window.go(key)
        pump(qapp, timeout=0.05)
        assert window.header.title.text() == page.title
        assert window.header.about.full == ' '.join((getattr(page, 'header_note', '') or page.about).split())
    assert window.header.title.text() == 'Settings' and 'settings.json' in window.header.about.full
    # a page draws no title of its own any more: the header names it once
    assert not [lb for p in window.pages for lb in p.findChildren(type(window.header.title))
                if lb.property('page') and lb.text() == p.title]
    window.go('home')
    pump(qapp, timeout=0.05)
    assert window.status.dot.c.isValid() and top.server.text.text()


def test_building_blocks(qapp):
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QComboBox, QLineEdit, QSpinBox, QTreeWidget, QTreeWidgetItem, QWidget
    from falcon_autonomy.ui.widgets import (FIELD_MAX, INTERVAL_ROLE, Form, IntervalDelegate, Meter, Section, Stat,
                                            callout, chip)
    host = QWidget()
    f = Form()
    host.setLayout(f)
    edit, combo, spin = f.row('Name', QLineEdit()), f.row('Kind', QComboBox()), f.row('n', QSpinBox())
    host.resize(1600, 200)
    host.show()
    pump(qapp, timeout=0.05)
    assert all(w.maximumWidth() == FIELD_MAX and w.width() <= FIELD_MAX for w in (edit, combo, spin))
    assert edit.x() < 300                                       # left-aligned under its label, not pushed right
    st = Stat('Latest AP')
    st.set('0.243', sub='95 % CI 0.215 – 0.336', tone='warning')
    assert st.value.text() == '0.243' and not st.unit.isVisibleTo(st) and 'color' in st.sub.styleSheet()
    m = Meter(80)
    m.set(1.7)
    assert m.frac == 1.0
    m.set(None)
    assert m.frac == 0.0
    assert chip('LHT', 'accent').property('tone') == 'accent'
    assert callout('Synthetic').label.text() == 'Synthetic'
    outer, inner = Section('Outer'), Section('Inner')
    outer.add(inner)
    assert outer.objectName() == inner.objectName() == 'section' and inner.title == 'Inner'
    t = QTreeWidget()
    t.setColumnCount(2)
    t.setItemDelegateForColumn(1, IntervalDelegate(t))
    for v in ((0.24, 0.21, 0.34), (None, None, None), (1.5, None, None)):
        it = QTreeWidgetItem(['m', ''])
        it.setData(1, INTERVAL_ROLE, v)
        t.addTopLevelItem(it)
    t.resize(400, 200)
    t.render(QPixmap(400, 200))                                 # paints every row (out-of-axis values clamp)
    for w in (st, m, outer):
        w.render(QPixmap(300, 120))
    host.close()


def test_metric_labels():
    from falcon_autonomy.ui.pages.models import metric_label as L
    assert [L(k) for k in ('AP', 'AP.car', 'AP.range_m[0,20)', 'AP.range_m[80,inf)', 'AP.height[160,inf)',
                           'AP.occluded[0.1,0.5)')] == \
        ['AP', 'AP · car', 'AP · range 0–20 m', 'AP · range ≥ 80 m', 'AP · height ≥ 160 px', 'AP · occluded 0.1–0.5']


def test_animations_run_and_leave_nothing_behind(window, qapp, motion_on):
    from PyQt6.QtWidgets import QLabel
    motion = motion_on
    window.go('home')
    pump(qapp, timeout=0.3)
    window.go('worlds')                                         # the crossfade: a snapshot over the stack, then gone
    shots = [c for c in window.stack.findChildren(QLabel) if c.objectName() == 'motionSnapshot']
    assert len(shots) == 1 and shots[0].graphicsEffect() is not None
    nav = window.nav
    target = nav._target(nav.currentRow())
    assert nav.anim.state() == nav.anim.State.Running and nav.pill != target
    assert pump(qapp, lambda: not [c for c in window.stack.findChildren(QLabel)
                                   if c.objectName() == 'motionSnapshot'], 2)
    assert pump(qapp, lambda: nav.pill == target, 2)
    page = window.page('worlds')                                # its cards faded in, their effects removed after
    from falcon_autonomy.ui.widgets import Section
    assert pump(qapp, lambda: all(s.graphicsEffect() is None for s in page.findChildren(Section)), 2)
    d = motion.Dot('#2fbf71')
    d.pulse(True)
    assert d.anim.state() == d.anim.State.Running
    d.pulse(False)
    assert d.anim.state() == d.anim.State.Stopped and d.t == 0.0
    # the hover easing tints a button while the pointer is over it, and clears once it has left
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication, QPushButton
    b = QPushButton('Go')
    b.setProperty('role', 'primary')
    b.show()
    QApplication.sendEvent(b, QEvent(QEvent.Type.Enter))
    assert pump(qapp, lambda: 'background' in b.styleSheet(), 1)
    QApplication.sendEvent(b, QEvent(QEvent.Type.Leave))
    assert pump(qapp, lambda: b.styleSheet() == '', 1)
    b.setEnabled(False)
    assert b.styleSheet() == ''
    b.close()
    # Reduce motion: nothing animates, and Settings keeps the choice
    settings = window.go('settings')
    assert pump(qapp, lambda: not [c for c in window.stack.findChildren(QLabel)
                                   if c.objectName() == 'motionSnapshot'] and nav.anim.state() != nav.anim.State.Running, 2)
    settings.motion.setChecked(True)
    assert not motion.enabled() and window.settings.get('reduce_motion') is True
    window.go('home')
    assert not [c for c in window.stack.findChildren(QLabel) if c.objectName() == 'motionSnapshot']
    assert nav.anim.state() == nav.anim.State.Stopped and nav.pill == nav._target(nav.currentRow())
    d.pulse(True)
    assert d.anim.state() == d.anim.State.Stopped
    settings.motion.setChecked(False)
    assert motion.enabled() and window.settings.get('reduce_motion') is False


def test_home_tiles_go_to_their_pages(window, qapp, project):
    home = window.go('home')
    home.refresh()
    tiles = home.count_btns
    assert tiles['worlds'][0].value.text() == str(len(project.doc['worlds']))
    assert tiles['runs'][0].value.text() == '0' and home.ap_tile.value.text() == '—'
    tiles['scenarios'][0].click()
    assert window.stack.currentWidget().key == 'scenarios'
    window.go('home')
    tiles['runs'][0].click()
    assert window.stack.currentWidget().key == 'data'


def test_assistant_wait_strip_counts_down_and_offers_the_accounts_limit(window, qapp):
    panel = window.assistant_panel             # (its chat page may be behind the set-up page: own flags checked)
    svc = window.assistant
    svc.set_cfg(tpm_limit=8000)
    panel.on_event('tpm_wait', {'wait_s': 30.0, 'tokens': 7433, 'limit': 8000, 'account': 250000})
    assert (not panel.wait_bar.isHidden()) and 0.9 < panel.wait_bar.frac <= 1.0
    assert panel.status.text().startswith('Waiting 30 s') and '8,000 tokens a minute' in panel.status.text()
    assert (not panel.status_action.isHidden()) and '250,000' in panel.status_action.text()
    panel.status_action.click()                                 # the account's limit, kept in Settings ▸ Assistant
    assert svc.cfg()['tpm_limit'] == 250000 and not (not panel.wait_bar.isHidden())
    assert 'now 250,000 tokens a minute' in panel.status.text()
    panel.on_event('tpm_wait', {'wait_s': 30.0, 'tokens': 9000, 'limit': 8000, 'account': None})
    assert 'Settings' in panel.status_action.text()             # no account limit known: where to change it
    panel.on_event('tpm_wait', {'wait_s': 0.0, 'tokens': 9000, 'limit': 8000, 'account': None})
    assert not (not panel.wait_bar.isHidden()) and not (not panel.status_action.isHidden())
    panel.on_event('retry', {'status': 429, 'wait_s': 0.4, 'attempt': 2})
    assert panel.status.text().startswith('Rate limited — retrying in')
    assert pump(qapp, lambda: not (not panel.wait_bar.isHidden()), 2)
    assert panel.send_btn.property('role') == 'primary' and not (not panel.working.isHidden())
