"""Models ▸ Introspection (ui/introspect_tab.py) offscreen: the tab is on the Models page, lists token policies and
egos, shows a recording's frames with their tokens and the 3D overlay, an audit's table, queues its jobs with the
right command lines; the 3D view's overlay paints; the adversary panel's level-k baseline."""
import json
import os

import pytest

from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401 (the GUI tests' fixtures)

pytestmark = pytest.mark.gui


@pytest.fixture
def tab(window):
    from falcon_autonomy.ui.pages.models import ModelsPage
    p = ModelsPage(window)
    p.resize(1300, 800)
    p.show()
    assert p.introspect is not None                                       # the optional tab is in this build
    yield p.introspect
    p.close()
    p.deleteLater()


def _write(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(doc, fh)


def test_tab_lists_and_shows_results(tab, window, qapp):
    ws = window.workspace()
    _write(os.path.join(ws, 'policies', 'tok', 'policy.json'), {'schema': 'falcon.policy/1', 'net': {'kind': 'token'}})
    _write(os.path.join(ws, 'policies', 'mlp', 'policy.json'), {'schema': 'falcon.policy/1', 'net': {'kind': 'mlp'}})
    frame = {'frame': 0, 'action': [0.01, -0.5], 'by_type': {'patch': 0.5, 'object': 0.3, 'route': 0.1, 'ego': 0.1},
             'tokens': [{'index': 0, 'type': 'patch', 'row': 0, 'col': 0, 'relevance': 0.2, 'plan_attention': 0.1},
                        {'index': 40, 'type': 'object', 'id': 11, 'class': 'car', 'brake_light': True,
                         'relevance': 0.3, 'plan_attention': 0.4}],
             'points': [{'x': 11.5, 'y': 0.5, 'z': 1.2, 'weight': 0.2}],
             'objects': [{'id': 11, 'class': 'car', 'x': 15.0, 'y': 0.2, 'yaw': 0.0, 'length': 4.6, 'width': 2.0,
                          'weight': 0.3}]}
    _write(os.path.join(ws, 'introspect', 'attention-20260101-000000', 'attention.json'),
           {'frames': [frame, dict(frame, frame=5)], 'policy': os.path.join(ws, 'policies', 'tok')})
    open(os.path.join(ws, 'introspect', 'attention-20260101-000000', 'frames.npz'), 'wb').close()
    o = {'cue_effect': {'mean': -3.0, 'ci95': [-3.5, -2.5]}, 'cause_effect': {'mean': 0.0, 'ci95': [0.0, 0.0]},
         'R': 100.0, 'R_ci95': [100.0, 100.0], 'flagged': True}
    _write(os.path.join(ws, 'introspect', 'audit-20260101-000000', 'audit.json'),
           {'ego': 'tok', 'seeds': [0, 1], 'flagged': ['brake_light'],
            'arms': {'brake_light': {'available': True, 'flagged': True, 'outcomes': {'brake_onset_s': o}},
                     'velocity': {'available': False, 'why': 'black box'}}})
    tab.refresh()
    assert [tab.policy.itemText(i) for i in range(tab.policy.count())] == [os.path.join('policies', 'tok')]
    egos = [tab.ego.itemData(i) for i in range(tab.ego.count())]
    assert egos[-2:] == ['follower', 'none'] and len(egos) == 4
    assert tab.frames.count() == 2 and tab.tokens.topLevelItemCount() == 2
    assert tab.tokens.topLevelItem(0).text(1) == 'car #11 (brake light)'                  # most relevant first
    assert tab.view.overlay['boxes'][0]['id'] == 11 and len(tab.view.overlay['points']) == 1
    assert tab.table.topLevelItemCount() == 2 and tab.table.topLevelItem(0).text(5) == 'yes'
    assert 'shortcut flagged on brake_light' in tab.state.vals['Audit'].text()
    tab.view.grab()                                                       # the overlay paints
    tab.view.set_overlay(None)
    assert tab.view.overlay is None


def test_tab_queues_its_jobs(tab, window, qapp, monkeypatch):
    ws = window.workspace()
    _write(os.path.join(ws, 'policies', 'tok', 'policy.json'), {'schema': 'falcon.policy/1', 'net': {'kind': 'token'}})
    tab.refresh()
    sent = []
    monkeypatch.setattr(window.queue, 'submit', lambda spec: sent.append(spec) or type('J', (), {
        'id': len(sent), 'state': 'queued'})())
    tab.record()
    tab.audit()
    tab.train()
    argv = [s.argv[3:] for s in sent]
    assert argv[0][:3] == ['introspect', 'attention', window.project.path] and '--policy' in argv[0]
    assert argv[1][:2] == ['introspect', 'audit'] and '--arms' in argv[1]
    assert argv[1][argv[1].index('--arms') + 1] == 'brake_light,velocity,autorickshaw'
    assert argv[2][:3] == ['policy', 'bc', '--arch'] and '--fake-carla' in argv[2]
    tab.op.setCurrentIndex(tab.op.count() - 1)
    tab.layer.setValue(2)
    tab.head.setValue(7)
    assert tab.selected_op() == {'op': 'zero_head', 'layer': 2, 'head': 7}


def test_tab_without_a_window(qapp):
    from falcon_autonomy.ui.introspect_tab import IntrospectTab
    t = IntrospectTab(type('Page', (), {})())
    assert t.win is None and t.frames.count() == 0
    t.record()
    t.audit()
    t.train()
    t.intervene()                                                         # nothing to run, no dialog, no error
    t.deleteLater()


def test_adversary_panel_levelk_baseline(window, qapp):
    from falcon_autonomy.ui.pages.rarecases import RareCasesPage
    p = RareCasesPage(window)
    try:
        sec = p.adversary
        assert sec.baselines() == []
        sec.levelk.setChecked(True)
        b = sec.baselines()
        assert b[0] == '--baselines' and b[1].split(',')[-1] == 'levelk' and 'random' in b[1]
    finally:
        p.close()
        p.deleteLater()
