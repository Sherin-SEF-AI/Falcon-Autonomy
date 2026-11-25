"""4.2 studies (workstream S), offscreen: fidelity beside every study result, the two-speed search's trust threshold
line and its "why correlation is low" panel, the release gate's refusal across fidelity and its Compare in Replay,
the Moments section's fidelity column, filter, Open in Replay and Send-to target, and the Rare cases page's Send-to
target that preselects a run without starting anything. No CARLA, no model, no network."""
import json
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytestmark = pytest.mark.gui

from conftest import FAKE  # noqa: E402
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401

from falcon_autonomy.core import fastsim as F  # noqa: E402
from falcon_autonomy.core import releasegate as G  # noqa: E402
from falcon_autonomy.core import scenarios as SC  # noqa: E402


def _win(ws, **kw):
    from falcon_autonomy.ui.context import RunContext

    class Q:
        def __init__(self):
            self.specs = []

        def submit(self, spec):
            self.specs.append(spec)
            return NS(state='queued', summary=lambda: 'queued')
    doc = {'models': [], 'scenarios': [], 'worlds': [{'id': 'world-town10', 'name': 'Town10'}],
           'rigs': [{'id': 'rig-default', 'name': 'Rig', 'sensors': [{'type': 'camera.pinhole', 'name': 'front',
                                                                      'width': 64, 'height': 36}]}]}
    w = NS(project=NS(path=os.path.join(ws, 'p.falcon.json'), dirty=False, doc=doc,
                      item=lambda sec, i: next((x for x in doc.get(sec, ()) if x['id'] == i), None)),
           workspace=lambda: ws, queue=Q(), fake_carla=FAKE, save=lambda: True, send_targets={}, **kw)
    w.context = RunContext()
    return w


def _swapped_pairs(n_extra=0):
    """14 pairs the screen orders right but for two swaps at the largest gaps (ρ ≈ 0.75), runs on the fake."""
    high = np.arange(14) / 13.0
    low = high.copy()
    for a, b in ((3, 10), (5, 8)):
        low[a], low[b] = high[b], high[a]
    gaps = [5, 6, 7, 40, 8, 30, 9, 10, 28, 11, 38, 12, 13, 14]
    return [{'low': float(lo), 'high': float(hi), 'role': 'top' if i % 4 else 'control', 'round': i // 4,
             'predicted': None, 'values': {'gap': float(g), 'speed': 20.0}, 'run': f'/ws/studies/s/runs/{i:02d}aa'
             f'bbccddeeff{i:02d}', 'fidelity': 'fake'} for i, (lo, hi, g) in enumerate(zip(low, high, gaps))][
        :14 - n_extra]


def _prescreen_doc(pairs, trust='not trusted', fidelity='fake'):
    rho = F.spearman([p['low'] for p in pairs], [p['high'] for p in pairs])
    fails = [dict(p, value=p['high']) for p in pairs if p['high'] >= 0.5]
    return {'schema': F.PRESCREEN_SCHEMA, 'study': 'search-lead_brake', 'state': 'finished', 'threshold': 0.5,
            'budget': 14, 'spent': len(pairs), 'screened': 1000, 'throughput': {'per_s': 900.0, 'device': 'cpu'},
            'trust': trust, 'fidelity': fidelity, 'screen_fidelity': 'kinematic', 'fallback': None,
            'gap': {'pairs': len(pairs), 'spearman': rho, 'spearman_controls': None, 'controls': 4,
                    'min_spearman': 0.5, 'min_pairs': 12, 'trust': trust, 'isotonic': None, 'residual_rmse': None,
                    'false_negative': {'dropped_controls': 0, 'failed': 0, 'upper95': None}},
            'pairs': pairs, 'failures': fails}


def _grab(w, name):
    """Paint the widget (every chart and chip draws); with FALCON_UI42_PNG=<folder>, save it in both themes there."""
    out = os.environ.get('FALCON_UI42_PNG')
    if not out:
        w.grab()
        return
    from PyQt6.QtWidgets import QApplication

    from falcon_autonomy.ui.theme import THEME
    for theme in ('dark', 'light'):
        THEME.apply(QApplication.instance(), theme)
        QApplication.instance().processEvents()
        w.grab().save(os.path.join(out, name.replace('.png', f'-{theme}.png')))
    THEME.apply(QApplication.instance(), 'dark')


# ================================================================================================ two-speed search
def test_the_trust_threshold_line_follows_the_box_and_the_low_correlation_panel(qapp, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

    from falcon_autonomy.ui.context import register_target
    from falcon_autonomy.ui.prescreen_tab import PrescreenTab
    ws = str(tmp_path)
    win = _win(ws)
    sent = []
    register_target(win, 'replay', 'Replay', sent.append)
    thr, seed, rig = QDoubleSpinBox(), QSpinBox(), QComboBox()
    thr.setValue(0.5)
    rig.addItem('Rig', 'rig-default')
    page = NS(win=win, scenarios=lambda: [('lead_brake', SC.builtin('lead_brake'))], threshold=thr, seed=seed,
              rig=rig, load=lambda: None)
    tab = PrescreenTab(page)
    # the dashed trust line starts at the box's default and follows it live
    assert tab.rho_trace.threshold == 0.5 and tab.rho_trace.threshold_label() == 'trust at ρ ≥ 0.50'
    tab.min_rho.setValue(0.3)
    assert tab.rho_trace.threshold == pytest.approx(0.3) and tab.rho_trace.threshold_label() == 'trust at ρ ≥ 0.30'
    assert not tab.why_low.isVisibleTo(tab) and tab.fid_screen.level == 'kinematic'
    # pairs arrive: the running ρ grows with them
    os.makedirs(tab.study_dir())
    path = os.path.join(tab.study_dir(), 'prescreen.json')
    with open(path, 'w') as fh:
        json.dump(_prescreen_doc(_swapped_pairs(n_extra=4), trust='not judged yet'), fh)
    tab.load()
    assert len(tab.rho_trace.series) == 10 - 2 and tab.rho_trace.min_pairs == 12
    with open(path, 'w') as fh:
        json.dump(_prescreen_doc(_swapped_pairs()), fh)
    tab.load()
    assert len(tab.rho_trace.series) == 12 and tab.rho_trace.series[-1][1] == pytest.approx(0.745, abs=1e-3)
    # not trusted: the panel shows, its rows say why, the worst pairs are listed with their runs
    assert tab.why_low.isVisibleTo(tab) and tab.diagnosis['parameter']['name'] == 'gap'
    kv = {k: v.text() for k, v in tab.why_kv.vals.items()}
    assert kv['ρ'].startswith('+0.75 over 14 pairs · trust at ≥ 0.30') and 'gap' in kv['Parameter']
    assert kv['Worst pair'].startswith('cheap ') and kv['Camera'] == 'front 64 × 36'
    assert tab.disagree.topLevelItemCount() == 3 and tab.disagree.topLevelItem(0).text(0) == '7'
    full = tab.disagree.topLevelItem(0).data(4, Qt.ItemDataRole.UserRole + 1)
    from falcon_autonomy.ui.widgets import short_id
    assert full.startswith('/ws/studies/s/runs/') and tab.disagree.topLevelItem(0).text(4) == short_id(
        os.path.basename(full)) and tab.disagree.topLevelItem(0).toolTip(4) == full
    assert 'Spearman over a dozen pairs is itself noisy' in tab.why_details.body.text()
    # trusted and above the box's threshold: no panel; under it: the panel, judged against the box
    with open(path, 'w') as fh:
        json.dump(_prescreen_doc(_swapped_pairs(), trust='trusted'), fh)
    tab.load()
    assert not tab.why_low.isVisibleTo(tab)
    tab.min_rho.setValue(0.8)
    assert tab.why_low.isVisibleTo(tab) and tab.rho_trace.threshold_label() == 'trust at ρ ≥ 0.80'
    # fidelity: the confirmed runs' chip, the screen's, and each failure's row
    assert tab.fid_runs.level == 'fake' and tab.fid_runs.chip.text() == 'Fake simulator'
    assert tab.failures.headerItem().text(4) == 'Fidelity' and tab.failures.topLevelItem(0).text(4) == 'Fake simulator'
    with open(path, 'w') as fh:
        json.dump(_prescreen_doc(_swapped_pairs(), trust='trusted', fidelity='mixed'), fh)
    tab.load()
    assert tab.fid_runs.level == 'mixed' and 'Mixed fidelity' in tab.fid_runs.note.text()
    # a failure row: the window's active run, and Send to ▸ Replay
    it = tab.failures.topLevelItem(0)
    tab.failures.setCurrentItem(it)
    run = it.data(0, 256)['run']
    assert win.context.get('run') == run
    menu = tab.failures.run_menu(it)
    sub = next(a.menu() for a in menu.actions() if a.menu() is not None)
    assert [a.text() for a in sub.actions()] == ['Replay']
    sub.actions()[0].trigger()
    assert sent == [run]
    tab.resize(520, 2000)
    _grab(tab, 'prescreen.png')
    tab.shutdown()


# ================================================================================================ release gate
class _Side:
    def __init__(self, tag, fails, level):
        self.tag, self.fails, self.level = tag, fails, level

    def run(self, trials):
        return [{'ok': True, 'failed': bool(self.fails(t)), 'run': f'/ws/gates/g/runs/{self.tag}/{t["i"]:04d}'
                 '0123456789abcdef', 'value': 1.0 if self.fails(t) else 0.1, 'fidelity': self.level(t)}
                for t in trials]


def test_the_gate_refuses_across_fidelity_flags_mixed_and_compares_in_replay(qapp, tmp_path):
    from PyQt6.QtCore import Qt

    from falcon_autonomy.ui.gate_tab import GateTab
    from falcon_autonomy.ui.theme import THEME
    from falcon_autonomy.ui.widgets import short_id
    ws = str(tmp_path)
    compared = []
    replay = NS(compare_runs=lambda a, b: compared.append((a, b)))
    win = _win(ws, go=lambda key: replay if key == 'replay' else None, page=lambda key: replay)
    tab = GateTab(NS(win=win))
    tab.name.setText('rel')
    scn = [SC.builtin('lead_brake')]
    G.Gate(tab.gate_dir(), _Side('a', lambda t: False, lambda t: 'carla'),
           _Side('b', lambda t: t['i'] % 2 == 0, lambda t: 'fake'), scn, budget=48).run()
    tab.load()
    assert tab.s_verdict.value.text() == 'no verdict' and THEME.status('critical') in tab.s_verdict.value.styleSheet()
    assert tab.fid_a.level == 'carla' and tab.fid_b.level == 'fake' and tab.fid_note.isVisibleTo(tab)
    assert tab.fid_note.label.text().startswith('No verdict. CARLA vs Fake simulator')
    assert 'holds for the tested scenario distribution only' in tab.scope.text()
    assert 'lead_brake' in tab.scope_details.body.text()
    # mixed within each version: the verdict stands, with the critical sentence
    lv = lambda t: 'fake' if t['i'] % 2 else 'carla'                     # noqa: E731
    tab.name.setText('mixed')
    G.Gate(tab.gate_dir(), _Side('a', lambda t: False, lv), _Side('b', lambda t: t['i'] % 3 == 0, lv), scn,
           budget=48).run()
    tab.load()
    assert tab.s_verdict.value.text() == 'worse' and tab.fid_a.level == 'mixed'
    assert tab.fid_note.label.text().startswith('Mixed fidelity: version A') and tab.fid_note.property('tone') == \
        'critical'
    # a newly failing row: the pair's fidelity, short copyable ids, Compare in Replay
    it = tab.changes.topLevelItem(0)
    x = it.data(0, 256)
    assert it.text(0) == 'newly fails' and it.text(5) in ('CARLA', 'Fake simulator')
    assert it.text(7) == short_id(os.path.basename(x['b'])) and it.data(7, Qt.ItemDataRole.UserRole + 1) == x['b']
    btn = tab.changes.itemWidget(it, 8)
    assert btn.text() == 'Compare in Replay' and btn.isEnabled()
    btn.click()
    assert compared == [(x['a'], x['b'])]
    tab.changes.setCurrentItem(it)
    assert win.context.get('run') == x['b']
    tab.resize(1000, 1700)
    _grab(tab, 'gate.png')
    # a Replay page without compare_runs: the button says why and is off
    win2 = _win(ws, go=lambda key: NS(), page=lambda key: NS())
    tab2 = GateTab(NS(win=win2))
    tab2.name.setText('mixed')
    tab2.load()
    b2 = tab2.changes.itemWidget(tab2.changes.topLevelItem(0), 8)
    assert not b2.isEnabled() and 'cannot compare two runs' in b2.toolTip()
    assert tab2.compare_in_replay(x) is False and 'cannot compare' in tab2.note.text()
    tab.shutdown()
    tab2.shutdown()


# ================================================================================================ moments
def test_moments_fidelity_filter_open_in_replay_and_the_send_to_target(qapp, tmp_path):
    from test_moments import actor, write_run

    from falcon_autonomy.ui.moments_tab import MomentsTab
    ws = str(tmp_path / 'ws')
    os.makedirs(ws)
    for name, lv in (('real', 'carla'), ('fake', 'fake')):
        write_run(ws, name, 10.0, [actor('car', lambda t: 120 - 10 * t, 0.0, yaw=180.0)], dur=7.0,
                  extra={'provenance': {'settings': {'weather': {'sun_altitude_angle': 45.0}, 'fps': 20},
                                        'fidelity': lv}})
    loaded, went = [], []
    replay = NS(load_run=loaded.append, load_log=loaded.append)
    win = _win(ws, go=lambda key: went.append(key) or replay)
    tab = MomentsTab(NS(win=win, load_log=loaded.append))
    assert 'moments' in win.send_targets and win.send_targets['moments'][0] == 'Moments'
    # a run sent here before any index: indexed, then its moments listed
    real = os.path.join(ws, 'runs', 'real')
    win.send_targets['moments'][1](real)
    assert went == ['replay']
    assert pump(qapp, lambda: not tab.busy, 30)
    assert tab.run_row.isVisibleTo(tab) and tab.hits and {m['run'] for m in tab.hits} == {real}
    assert 'its moments' in tab.run_label.text()
    tab.show_all()
    assert not tab.run_row.isVisibleTo(tab)
    tab.index()                                                  # the other run too
    assert pump(qapp, lambda: not tab.busy, 30)
    # "time to collision" is a TTC condition, never the collision type
    tab.query.setText('time to collision under 2 s')
    tab.search()
    assert tab.understood.text() == 'understood as: min TTC < 2 s' and not tab.unknown.isVisibleTo(tab)
    # per-hit fidelity, and the box filters by it
    tab.query.setText('collision')
    tab.search()
    assert {tab.tree.topLevelItem(i).text(7) for i in range(tab.tree.topLevelItemCount())} == {'CARLA',
                                                                                             'Fake simulator'}
    tab.fidelity.setCurrentIndex(tab.fidelity.findData('fake'))
    assert tab.understood.text() == 'understood as: type collision · fidelity fake'
    assert {m['fidelity'] for m in tab.hits} == {'fake'} and tab.tree.topLevelItemCount() >= 1
    # Open in Replay: the page's run loader; a hit row selects its run and has Send to ▸
    it = tab.tree.topLevelItem(0)
    tab.tree.setCurrentItem(it)
    assert win.context.get('run') == os.path.join(ws, 'runs', 'fake')
    m = tab.open_in_replay(it)
    assert m is not None and loaded == [m['run']]
    menu = tab.tree.run_menu(it)
    assert [a.text() for a in menu.actions()][:2] == ['Open on the Data page', 'Open in Replay']
    win.go = lambda key: NS()                                    # a Replay page that cannot load a run
    assert tab.open_in_replay(it) is None and 'cannot load a run' in tab.note.text()
    tab.resize(640, 700)
    _grab(tab, 'moments.png')


# ================================================================================================ rare cases
def test_the_rare_cases_target_preselects_a_run_and_starts_nothing(window, qapp, tmp_path):
    from falcon_autonomy.ui.context import send_to_menu
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.save()
    ws = window.workspace()
    page = window.go('rarecases')
    assert 'rarecases' in window.send_targets and 'moments' in window.send_targets
    run = os.path.join(ws, 'runs', 'cut-in-0007')
    os.makedirs(run)
    world, rig = window.project.doc['worlds'][0]['id'], window.project.doc['rigs'][0]['id']
    with open(os.path.join(run, 'run.json'), 'w') as fh:
        json.dump({'schema': 'falcon.run/1', 'status': 'ok', 'provenance': {
            'fidelity': 'fake', 'settings': {'seed': 7, 'scenario': {'id': 'cut_in', 'name': 'Cut-in'},
                                             'project': {'world': world, 'rig': rig}}}}, fh)
    jobs = len(window.queue.jobs)
    window.go('data')
    m = send_to_menu(window, run, exclude='data')
    act = next(a for a in m.actions() if a.text() == 'Rare cases')
    act.trigger()
    assert window.current_key() == 'rarecases' and window.context.get('run') == run
    assert page.scenario.currentData() == 'cut_in' and page.seed.value() == 7
    assert window.context.get('seed') == '7' and window.context.get('scenario') == 'cut_in'
    assert page.from_run is not None and page.from_run.isVisibleTo(page)
    assert [b.text() for b in page.from_run.buttons] == ['Estimate its failure rate', 'Search its variations']
    assert 'Nothing runs until you choose' in page.from_run.label.text()
    assert len(window.queue.jobs) == jobs                                # nothing started
    # a run of a scenario this project does not have: said, nothing preselected
    other = os.path.join(ws, 'runs', 'x')
    os.makedirs(other)
    with open(os.path.join(other, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok', 'provenance': {'settings': {'scenario': {'id': 'nope'}}}}, fh)
    assert page.preselect(other)['scenario'] is None and 'not one of this project' in page.from_run.label.text()
    # the study's fidelity beside the curve, each case's in its row
    d = page.study_dir()
    os.makedirs(d, exist_ok=True)
    from falcon_autonomy.core import search as S
    vals = S.to_params(page.space, np.full(len(page.space), 0.5))
    rows = [{'key': f'k{i}', 'values': vals, 'run': os.path.join(d, 'runs', f'r{i}'),
             'status': 'ok', 'criticality': 0.9, 'objective': 0.9, 'fidelity': 'fake'} for i in range(2)]
    with open(os.path.join(d, 'results.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(r) + '\n' for r in rows))
    page.load()
    assert page.fid.level == 'fake' and page.failures.topLevelItem(0).text(7) == 'Fake simulator'
    page.failures.setCurrentItem(page.failures.topLevelItem(0))
    assert window.context.get('run') == rows[0]['run'] or window.context.get('run') == rows[1]['run']
    menu = page.failures.run_menu(page.failures.topLevelItem(0))       # one menu: the case's actions, Send to, Ask
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts[:2] == ['Replay the case', 'Browse its run'] and texts[-1] == 'Ask about this'
    assert any(a.menu() is not None and 'Moments' in [b.text() for b in a.menu().actions()] for a in menu.actions())
    window.resize(1440, 900)
    _grab(window, 'rarecases.png')
