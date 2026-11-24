"""The Moments section of the Replay page and the Overnight campaign tab of the Loop page, offscreen: a query
understood and shown back, hits with their severity, a hit opened at its time, made a scenario and put in a drive set;
a campaign previewed, approved once and run by its runner job to a brief. No CARLA, no model."""
import os

import pytest

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401
from test_moments import NIGHT_RAIN, actor, lin, write_run  # noqa: E402


def _workspace(window, tmp_path):
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.save()
    ws = window.workspace()
    os.makedirs(ws, exist_ok=True)
    write_run(ws, 'cutin-night', 10.0, [actor('motorcycle', lambda t: 14 + 9 * t, lin(-3.5, 0.0, 2.0, 4.0),
                                              size=(2.2, 0.8))], weather=NIGHT_RAIN)
    write_run(ws, 'cutin-day', 10.0, [actor('car', lambda t: 22 + 9 * t, lin(3.5, 0.0, 2.0, 4.0))])
    write_run(ws, 'headon', 10.0, [actor('car', lambda t: 120 - 10 * t, 0.0, yaw=180.0)], dur=7.0)
    return ws


def test_moments_section_searches_opens_and_makes_a_scenario(window, qapp, tmp_path):
    from falcon_autonomy.ui.widgets import Meter
    ws = _workspace(window, tmp_path)
    page = window.go('replay')
    tab = page.moments
    assert tab is not None and tab.isVisibleTo(page)
    assert tab.search() == [] and 'No moment index' in tab.note.text()
    tab.index()
    assert pump(qapp, lambda: not tab.busy, 30)
    assert '3 source(s): 3 read' in tab.note.text() or 'moment(s)' in tab.note.text()
    tab.query.setText('two-wheeler cuts in from the left at night in rain, zebra')
    hits = tab.search()
    assert tab.understood.text() == ('understood as: type cut_in · actor motorcycle or bicycle · side left · light '
                                     'night · weather rain')
    assert tab.understood.property('mono') is True and tab.unknown.property('mono') is True
    assert tab.unknown.isVisibleTo(tab) and tab.unknown.text() == 'not understood: zebra'
    assert len(hits) == 2 and tab.tree.topLevelItemCount() == 2
    it = tab.tree.topLevelItem(0)
    assert [it.text(k) for k in (0, 1, 2)] == ['cut-in', 'motorcycle', 'left'] and it.text(4).endswith('cutin-night')
    assert it.text(6) == '0.30 low' and it.text(5).endswith(' s')
    assert tab.tree.itemWidget(it, 6).findChild(Meter).frac == pytest.approx(0.3)
    assert 'partial: not actors, side, tod, weather' in tab.tree.topLevelItem(1).text(4)
    assert '2 moment(s), 1 matching every term' in tab.note.text()
    # every moment, most severe first; the collision's meter is full
    tab.query.setText('')
    tab.search()
    assert not tab.unknown.isVisibleTo(tab) and tab.understood.text() == 'understood as: (everything)'
    top = tab.tree.topLevelItem(0)
    assert top.text(0) == 'collision' and top.text(6) == '1.00 collision'
    assert tab.tree.itemWidget(top, 6).findChild(Meter).frac == 1.0
    # a hit: the run on the Data page at its time
    tab.query.setText('motorcycle cut in')
    tab.search()
    it = tab.tree.topLevelItem(0)
    tab.tree.setCurrentItem(it)
    m = tab.open(it)
    data = window.go('data')
    assert m['type'] == 'cut_in' and data.tabs.currentWidget() is data.browse
    assert data.browse.slider.value() == round(m['t0'] / 0.05)
    # a scenario and a drive set from the hits
    window.go('replay')
    res = tab.make_scenario('gui-cut-in')
    assert res is not None and os.path.isfile(os.path.join(ws, 'moments', 'scenarios', 'gui-cut-in', 'moment.xosc'))
    assert 'Scenario written' in tab.note.text()
    assert tab.make_scenario('../bad') is None and 'letters, digits' in tab.note.text()
    ds = tab.drive_set('cut-ins')
    assert len(ds['runs']) == 1 and window.project.doc['drive_sets'][-1]['name'] == 'cut-ins'
    window.undo.undo()
    assert not any(d.get('name') == 'cut-ins' for d in window.project.doc.get('drive_sets') or ())


def test_campaign_tab_previews_asks_once_and_shows_the_brief(window, qapp, tmp_path):
    from falcon_autonomy.core import campaign as K
    ws = _workspace(window, tmp_path)
    page = window.go('loop')
    tab = page.campaign
    assert tab is not None and page.tabs.indexOf(tab) >= 0
    page.tabs.setCurrentWidget(tab)
    assert tab.start_btn.property('role') == 'primary' and tab.preview_btn.property('role') is None
    assert [k for k, cb in tab.kinds.items() if cb.isChecked()] == list(K.DEFAULT_KINDS)
    assert tab.state.text() == 'no campaign yet' and not tab.stop_btn.isEnabled()
    assert tab.preview() is None and 'goal' in tab.note.text()                   # no goal yet
    tab.goal.setText('Find out how often a motorcycle cuts in at night')
    tab.gpu.setValue(0.5)
    tab.until.setText('+2h')
    tab.jobs.setValue(3)
    for k, cb in tab.kinds.items():
        cb.setChecked(k == 'moments')
    text = tab.preview()
    assert 'Budget: 0.5 GPU-hours' in text and 'at most 3 jobs' in text and 's1 [moments]' in text
    assert tab.card.isVisibleTo(tab) and tab.card.toPlainText() == text
    assert not os.path.exists(os.path.join(ws, 'campaigns'))                       # a preview writes nothing
    asked = []
    tab.confirm = lambda t: asked.append(t) or False
    n = len(window.queue.jobs)
    assert tab.start() is None and len(asked) == 1 and 'Not approved' in tab.note.text()
    assert len(window.queue.jobs) == n and not os.path.exists(os.path.join(ws, 'campaigns'))
    tab.confirm = lambda t: asked.append(t) or True
    job = tab.start()
    assert job is not None and job.spec.kind == 'campaign' and len(asked) == 2 and 'Never done unattended' in asked[1]
    cid = tab.cid
    rec = K.load(ws, cid)
    assert rec['approval'] == {'how': 'approved', 'by': 'user', 'time': rec['approval']['time']}
    assert tab.pick.count() == 1 and tab.state.text() in ('approved — waiting for its runner', 'running')
    assert tab.stop_btn.isEnabled() and tab.sweep.isVisibleTo(tab)
    assert tab.b_gpu.text.text() == '0.00 of 0.5 GPU-h' and tab.b_jobs.text.text().endswith('of 3')
    assert 'min left' in tab.b_time.text.text()
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120)
    assert job.state == 'done', job.log
    st = tab.refresh()
    assert st['state'] == 'done' and tab.state.text() == 'done' and not tab.stop_btn.isEnabled()
    assert not tab.sweep.isVisibleTo(tab) and not tab.timer.isActive()
    assert tab.b_jobs.text.text() == '1 of 3' and tab.b_jobs.meter.frac == pytest.approx(1 / 3)
    decisions = [tab.log.topLevelItem(i).text(1) for i in range(tab.log.topLevelItemCount())]
    assert [d for d in decisions if d != 'needs_you'] == ['new', 'plan', 'approved', 'start', 'run', 'result', 'stop']
    assert 'within budget' in tab.log.topLevelItem(decisions.index('run')).text(2)
    assert tab.needs.isVisibleTo(tab) and tab.needs.text().startswith('Needs you: Failure modes')
    brief = tab.brief.toPlainText()
    assert 'Morning brief' in brief and 'Confidence: low' in brief and '1 recorded moment(s) match' in brief
    # a second campaign, stopped before its runner starts: the brief says nothing ran
    tab.goal.setText('Find why pedestrians are missed in rain')
    job2 = tab.start()
    window.queue.cancel(job2)
    assert tab.pick.count() == 2 and tab.cid != cid
    st = tab.stop()
    assert st['state'] == 'stopped' and tab.state.text() == 'stopped' and '(nothing ran)' in tab.brief.toPlainText()
