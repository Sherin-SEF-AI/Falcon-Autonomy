"""The scenario timeline editor: the plan (core.scenario_plan) against the director's rules worked by hand, the
timeline's validation, and the editor on the Scenarios page — dragging a trigger and a duration (snapped), the form,
adding, duplicating and removing events and actions, every change one undoable edit, nothing written until something
changes (an untouched scenario saves byte for byte), invalid input refused or flagged inline, OpenSCENARIO export of
an edited scenario still read back."""
import copy
import json
import math

import pytest

from falcon_autonomy.core import scenario_plan as PL
from falcon_autonomy.core import scenarios as SC


# ------------------------------------------------------------------------------------------------------- the plan
def test_plan_time_trigger_and_brake_ramp():
    scn = SC.concrete(SC.builtin('lead_brake'), {'speed': 36.0, 'gap': 20.0, 'decel': 5.0, 'brake_at': 3.0})
    pl = PL.plan(scn, dt=0.05)
    assert pl['fired'] == [(0, 3.0)]
    (i, j, t, end), = pl['actions']
    assert (i, j, t) == (0, 0, 3.0) and end == pytest.approx(3.0 + 10.0 / 5.0)       # 10 m/s at 5 m/s²
    lead = dict(zip(pl['t'], pl['tracks']['lead']))
    assert lead[3.0][0] == pytest.approx(20.0 + 10.0 * 3.0, abs=1e-6)               # at speed until the brake
    assert lead[6.0][0] == pytest.approx(50.0 + 10.0 ** 2 / (2 * 5.0), abs=0.3)     # then stopped 10 m on
    assert dict(zip(pl['t'], pl['tracks']['ego']))[6.0][0] == pytest.approx(60.0)   # the ego holds its speed
    assert 'not a simulation' in pl['note']


def test_plan_distance_trigger_walk_and_ahead_trigger_lane_change():
    ped = SC.concrete(SC.builtin('pedestrian_crossing'), {k: (v.get('lo', 1.0) + v.get('hi', 1.0)) / 2
                                                          for k, v in SC.builtin('pedestrian_crossing')['params']
                                                          .items()})
    pl = PL.plan(ped, dt=0.05)
    a = ped['actors'][0]
    trig = ped['timeline'][0]['trigger']
    (i, t), = pl['fired']
    ego_v = float(ped['ego']['speed_kmh']) / 3.6
    y = PL.lane_offset(a, False)
    x_ped = float(a['place']['ahead_m'])
    # the first step at which the ego's centre is within lt_m of the pedestrian's
    k = next(k for k in range(10000) if math.hypot(x_ped - k * 0.05 * ego_v, y) < float(trig['lt_m']))
    assert t == pytest.approx(k * 0.05, abs=1e-6)
    tr = dict(zip(pl['t'], pl['tracks'][a['id']]))
    assert tr[t][1] == pytest.approx(y) and tr[round(t + 1.0, 2)][1] < y          # it walks across, away from the kerb
    cut = SC.concrete(SC.builtin('cut_in'), {'speed': 36.0, 'cutter_speed': 72.0, 'start_offset': 0.0,
                                             'cut_gap': 5.0, 'cut_time': 2.0})
    pl = PL.plan(cut, dt=0.05)
    # its rear more than 5 m ahead of the ego's front: (20 − 10) m/s × t − 4.6 m > 5 m first at t = 1.0 s
    assert pl['fired'] == [(0, 1.0)] and pl['actions'] == [(0, 0, 1.0, 3.0)]
    ys = dict(zip(pl['t'], pl['tracks']['cutter']))
    assert ys[1.0][1] == pytest.approx(-PL.LANE_W) and ys[2.0][1] == pytest.approx(-PL.LANE_W / 2)
    assert ys[3.0][1] == pytest.approx(0.0, abs=1e-9)

def test_plan_lanes_follow_the_director():
    W = PL.LANE_W
    assert PL.lane_offset({'place': {'lane': 'adjacent'}}, True) == -W
    assert PL.lane_offset({'place': {'lane': 'opposite'}}, True) == -2 * W
    assert PL.lane_offset({'place': {'lane': 'opposite'}}, False) == -W
    assert PL.lane_offset({'kind': 'car', 'place': {'lane': 'kerb'}}, False) == pytest.approx(W / 2 + 1.0 + 0.2)
    assert PL.lane_offset({'place': {'lane': 'sidewalk', 'lateral_m': 0.5}}, False) == pytest.approx(W / 2 + 2.5)
    assert PL.lane_offset({'place': {'lane': 'line'}}, False) == -W / 2


def test_validation_names_bad_triggers_and_actions():
    s = SC.builtin('lead_brake')
    s['timeline'] = [{'trigger': {'type': 'when'}, 'actions': []},
                     {'trigger': {'type': 'distance', 'actor': 'lead'}, 'actions': [{'actor': 'lead', 'do': 'fly'}]}]
    probs = SC.validate(s)
    assert any("timeline 0: unknown trigger 'when'" in p for p in probs)
    assert any('timeline 1: a distance trigger needs lt_m' in p for p in probs)
    assert any("timeline 1: unknown action 'fly'" in p for p in probs)
    for k in SC.LIBRARY:
        assert SC.validate(SC.builtin(k)) == [], k


# ------------------------------------------------------------------------------------------------------- the editor
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


def _editor(window, tmp_path):
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.save()
    page = window.go('scenarios')
    page.list.setCurrentRow(0)                                   # Lead vehicle brakes (built-in)
    ed = page.timeline
    assert not ed.editable and not ed.add_btn.isEnabled() and 'Copy the built-in' in ed.add_btn.toolTip()
    page.copy_builtin()
    page.tabs.setCurrentWidget(ed)
    assert page.top.currentWidget() is ed.plan_panel             # the plan replaces the sketch on this tab
    assert ed.editable and ed.timeline() == window.project.doc['scenarios'][0]['timeline']
    return page, ed


@pytest.mark.gui
def test_untouched_scenario_is_not_rewritten(window, qapp, tmp_path):
    page, ed = _editor(window, tmp_path)
    window.save()
    before = open(window.project.path, 'rb').read()
    n = window.undo.count()
    for i in range(page.list.count()):                           # look at everything, change nothing
        page.list.setCurrentRow(i)
        ed.select(0)
        ed.axis.set_playhead(2.0)
        ed.trig_value.editingFinished.emit()                     # the same value back: no edit
    page.list.setCurrentRow(page.list.count() - 1)
    assert window.undo.count() == n
    window.save()
    assert open(window.project.path, 'rb').read() == before


@pytest.mark.gui
def test_drag_trigger_and_duration_snapped_and_undoable(window, qapp, tmp_path):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    page, ed = _editor(window, tmp_path)
    doc = window.project.doc['scenarios'][0]
    ed.trig_value.setText('3')
    ed.commit_trigger_value()
    assert doc['timeline'][0]['trigger']['t_s'] == 3                    # a number, not the text
    ax = ed.axis
    ax.resize(700, 200)
    ax.show()
    pump(qapp, timeout=0.1)
    x, y = int(ax.x_of(3.0)), int(ax.row_y(0) + 30 / 2 - 4)
    assert ax.hit(QPoint(x, y).toPointF())[0] == 'trigger'
    ed.snap_box.setCurrentIndex(2)                                     # 0.5 s
    QTest.mousePress(ax, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
    QTest.mouseMove(ax, QPoint(int(ax.x_of(4.37)), y))
    QTest.mouseRelease(ax, Qt.MouseButton.LeftButton, pos=QPoint(int(ax.x_of(4.37)), y))
    assert doc['timeline'][0]['trigger']['t_s'] == 4.5
    window.undo.undo()
    assert doc['timeline'][0]['trigger']['t_s'] == 3
    window.undo.undo()
    assert doc['timeline'][0]['trigger']['t_s'] == '$brake_at'          # back to the parameter
    # a lane change's duration, by its right end
    ed.change_action_kind(0, 'lane_change')
    a = doc['timeline'][0]['actions'][0]
    assert a == {'actor': 'lead', 'do': 'lane_change', 'to': 'ego', 'duration_s': 2.0}
    ed.set_trigger_field(0, 't_s', 1.0)
    pump(qapp, timeout=0.05)
    t0, t1, draggable = ax.action_span(0, 0)
    assert draggable and t0 == pytest.approx(1.0) and t1 == pytest.approx(3.0)
    ed.snap_box.setCurrentIndex(1)                                     # 0.1 s
    yb = int(ax.row_y(0) + 30 / 2 + 4)
    QTest.mousePress(ax, Qt.MouseButton.LeftButton, pos=QPoint(int(ax.x_of(3.0)), yb))
    QTest.mouseMove(ax, QPoint(int(ax.x_of(4.23)), yb))
    QTest.mouseRelease(ax, Qt.MouseButton.LeftButton, pos=QPoint(int(ax.x_of(4.23)), yb))
    assert doc['timeline'][0]['actions'][0]['duration_s'] == pytest.approx(3.2)
    s0, x1 = ax.scale(), ax.x_of(1.0)
    ax.zoom(2.0, x1)                                                   # about the pointer: 1 s stays put
    assert ax.scale() == pytest.approx(2 * s0) and ax.x_of(1.0) == pytest.approx(x1)
    ax.fit()
    assert not ax.grab().isNull() and not ed.planview.grab().isNull()


@pytest.mark.gui
def test_form_events_actions_and_inline_problems(window, qapp, tmp_path):
    page, ed = _editor(window, tmp_path)
    doc = window.project.doc['scenarios'][0]
    n0 = window.undo.index()
    ed.add_event()
    assert len(doc['timeline']) == 2 and ed.current == 1
    assert doc['timeline'][1] == {'trigger': {'type': 'time', 't_s': 1.0},
                                  'actions': [{'actor': 'lead', 'do': 'brake', 'decel': 6.0}]}
    ed.trig_type.setCurrentText('distance')
    ed.change_trigger_type()
    assert doc['timeline'][1]['trigger'] == {'type': 'distance', 'actor': 'lead', 'lt_m': 20.0}
    assert not ed.form.rows[id(ed.trig_actor)][0].isHidden()
    ed.trig_value.setText('$gap / 2')
    ed.commit_trigger_value()
    assert doc['timeline'][1]['trigger']['lt_m'] == '$gap / 2'
    ed.add_action()
    assert doc['timeline'][1]['actions'][-1] == {'actor': 'lead', 'do': 'speed', 'kmh': 30.0, 'accel': 3.0}
    ed.actions.item(1, 2).setText('kmh=$speed * 0.5, accel=1.5')          # the cell edit commits
    assert doc['timeline'][1]['actions'][1] == {'actor': 'lead', 'do': 'speed', 'kmh': '$speed * 0.5', 'accel': 1.5}
    ed.actions.item(1, 2).setText('kmh 40')                              # not name=value: refused, said why
    assert doc['timeline'][1]['actions'][1]['kmh'] == '$speed * 0.5' and 'not name=value' in ed.problems.text()
    ed.set_action_field(1, 0, 'actor', 'ghost')                          # allowed, flagged inline
    assert 1 in ed.bad_events and "unknown actor 'ghost'" in ed.problems.text()
    assert "unknown actor 'ghost'" in page.problems.text()
    ed.duplicate_event()
    assert len(doc['timeline']) == 3 and doc['timeline'][2] == doc['timeline'][1] and ed.current == 2
    ed.remove_event()
    assert len(doc['timeline']) == 2
    ed.actions.setCurrentCell(1, 2)
    ed.remove_action()
    assert len(doc['timeline'][1]['actions']) == 1
    tl = copy.deepcopy(doc['timeline'])
    tl[0]['trigger'] = {'type': 'ahead', 'actor': 'lead'}                # a required field missing: not applied
    assert not ed.commit(tl, 'bad') and 'needs gt_m' in ed.problems.text()
    while window.undo.index() > n0:
        window.undo.undo()
    assert doc['timeline'] == SC.builtin('lead_brake')['timeline']


@pytest.mark.gui
def test_edited_scenario_exports_to_openscenario(window, qapp, tmp_path):
    from falcon_autonomy.core import openscenario as X
    page, ed = _editor(window, tmp_path)
    ed.add_event()
    ed.select(0)
    ed.change_action_kind(0, 'speed')
    scn = window.project.doc['scenarios'][0]
    assert SC.validate(scn) == []
    conc = SC.sample(scn, 3)
    back = X.import_xosc(X.export_xosc(conc))
    assert len(back['timeline']) == 2
    assert [a['do'] for e in back['timeline'] for a in e['actions']] == ['speed', 'brake']
    json.dumps(scn)                                                       # still plain JSON
    ed.play_btn.setChecked(True)
    pump(qapp, lambda: ed.axis.playhead > 0.2, 3)
    ed.play_btn.setChecked(False)
    assert ed.axis.playhead > 0.0 and 't = ' in ed.plan_status.text()
