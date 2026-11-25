"""4.2 Runs & Replay offscreen: fidelity beside every run (Data, Home, Replay), Send to between Data, Home and Replay,
the synchronized side-by-side comparison of two runs with its divergences (core.logreplay.sync,
ReplayPage.compare_runs), and Data ▸ Compare's not-like-for-like callout."""
import json
import os

import numpy as np
import pytest
from PyQt6.QtCore import Qt

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


def make_run(d, name='run', swerve_at=None, swerve_m=3.0, fidelity='fake', extra_actor_at=None, n=101, dt=0.1,
             status='ok', min_ttc=None, world='w-town10', started='2026-11-20T10:00:00Z'):
    """A Falcon run folder as a worker writes it (run.json, frames.jsonl, trajectories.json): the ego straight along
    x at 8 m/s, a lead vehicle in the next lane; optionally the ego swerving sideways from `swerve_at` s, and an
    actor (a pedestrian, id 99) appearing at `extra_actor_at` s. fidelity None: a run written before 4.2."""
    os.makedirs(d, exist_ok=True)
    t = np.arange(n) * dt
    ex = -40 + 8 * t
    ey = np.zeros(n)
    if swerve_at is not None:
        ey = np.where(t >= swerve_at, np.minimum((t - swerve_at) * 4.0, swerve_m), 0.0)

    def rows(x, y, vx, L, W):
        return [[float(t[i]), float(x[i]), float(y[i]), 0.0, vx, 0.0, L, W] for i in range(n)]
    actors = {'ego': rows(ex, ey, 8.0, 4.6, 2.0), '7': rows(-15 + 7 * t, np.full(n, 3.5), 7.0, 4.5, 1.9)}
    if extra_actor_at is not None:
        k = int(round(extra_actor_at / dt))
        actors['99'] = [[float(t[i]), 10.0, -6.0 + 0.5 * (t[i] - t[k]), 90.0, 0.0, 0.5, 0.6, 0.6] for i in range(k, n)]
    with open(os.path.join(d, 'trajectories.json'), 'w', encoding='utf-8') as fh:
        json.dump({'dt': dt, 'columns': ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'length', 'width'], 'actors': actors}, fh)
    with open(os.path.join(d, 'frames.jsonl'), 'w', encoding='utf-8') as fh:
        for i in range(n):
            fh.write(json.dumps({'step': i, 'frame': 100 + i, 'sim_t': float(t[i]), 'speed_mps': 8.0,
                                 'ego': {'x': float(ex[i]), 'y': float(ey[i]), 'z': 0.0, 'yaw': 0.0, 'pitch': 0.0,
                                         'roll': 0.0}, 'files': {}}) + '\n')
    run = {'name': name, 'worker': 'drive', 'status': status, 'started': started, 'finished': started,
           'provenance': {'seed': 3, 'app_version': '4.1.0',
                          'settings': {'project': {'world': world, 'rig': 'rig-default'}, 'town': 'Town10HD_Opt',
                                       'fps': 1 / dt, 'seed': 3,
                                       'scenario': {'id': 'lead-brake', 'name': 'Lead brakes'}}},
           'performance': {'steps': n, 'sim_s': float(t[-1]), 'wall_s': 2.0},
           'summary': {'steps': n, 'realtime_factor': 5.0}}
    if fidelity:
        run['provenance']['fidelity'] = fidelity
    if min_ttc is not None:
        run['scenario'] = {'criticality': {'overall': {'min_ttc_s': min_ttc}, 'actors': {}}, 'passed': min_ttc > 1.5}
    with open(os.path.join(d, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump(run, fh)
    return d


# ------------------------------------------------------------------------------------------------ the clock
def test_synchronize_finds_a_planted_divergence(tmp_path):
    from falcon_autonomy.core.logreplay import sync as S
    a = make_run(str(tmp_path / 'a'))
    b = make_run(str(tmp_path / 'b'), swerve_at=4.0, extra_actor_at=7.0)
    r = S.synchronize(a, b)
    ev = r['events']
    assert ev, 'no divergence found'
    # the ego swerves at 4.0 s at 4 m/s sideways: more than 1 m apart from 4.25 s on (the first grid step past it)
    assert 4.2 <= ev[0]['t'] <= 4.4 and 'gap' in ev[0]['reasons']
    assert r['summary']['max_gap_m'] == pytest.approx(3.0, abs=1e-6)
    # identical runs: nothing diverges
    same = S.synchronize(a, make_run(str(tmp_path / 'a2')))
    assert same['events'] == [] and same['summary']['first_divergence_t'] is None
    # an actor present in one run only, with the egos together
    c = make_run(str(tmp_path / 'c'), extra_actor_at=7.0)
    r2 = S.synchronize(a, c)
    assert r2['events'] and r2['events'][0]['t'] == pytest.approx(7.0, abs=0.06) and \
        'actors' in r2['events'][0]['reasons']
    # two rates: the grid is the finer step, and the same swerve is found at the same time
    d = make_run(str(tmp_path / 'd'), swerve_at=4.0, n=201, dt=0.05)
    r3 = S.synchronize(a, d)
    assert r3['summary']['dt'] == pytest.approx(0.05) and 4.2 <= r3['events'][0]['t'] <= 4.35
    # a run with frames.jsonl only (no trajectories.json): the ego alone, said in the notes
    e = make_run(str(tmp_path / 'e'), swerve_at=4.0)
    os.remove(os.path.join(e, 'trajectories.json'))
    r4 = S.synchronize(a, e)
    assert r4['b']['source'] == 'frames.jsonl' and any('heuristic' in n for n in r4['notes'])
    assert 4.2 <= r4['events'][0]['t'] <= 4.4
    with pytest.raises(S.SyncError):
        S.synchronize(a, str(tmp_path))
    # the replay comparison records its fidelity (core.fidelity): one level, and each run's
    from falcon_autonomy.core import fidelity as F
    assert F.of_doc({'fidelity': 'mixed'}) == 'mixed'


def test_log_import_and_neural_closed_loop_record_fidelity(tmp_path):
    from falcon_autonomy.core.logreplay import import_log
    run = make_run(str(tmp_path / 'r'), fidelity='carla')
    log = import_log('falcon', run)
    assert log.meta['fidelity'] == 'carla'                   # a Falcon run's log keeps the run's own fidelity
    log.save(str(tmp_path / 'log'))
    from falcon_autonomy.core.logreplay import Log
    assert Log.load(str(tmp_path / 'log')).meta['fidelity'] == 'carla'
    from falcon_autonomy.ui.pages.replay import log_fidelity
    assert log_fidelity({'source': {'format': 'nuscenes'}}) == 'recorded'      # a log imported before 4.2
    assert log_fidelity({'source': {'format': 'falcon', 'path': run}}) == 'carla'
    assert log_fidelity({}) == 'unknown'
    src = open(os.path.join(os.path.dirname(__import__('falcon_autonomy').__file__), 'core', 'recon',
                            'closedloop.py'), encoding='utf-8').read()
    assert "['fidelity'] = 'neural'" in src and "'fidelity': 'neural'" in src


def _runs(window, specs):
    """Run folders in the window's workspace (newest last in specs), catalogued as a finished job would be."""
    ws = window.workspace()
    out = []
    for k, (name, kw) in enumerate(specs):
        d = make_run(os.path.join(ws, 'runs', name), name=name, started=f'2026-11-2{k}T10:00:00Z', **kw)
        window.service.catalog.add_run(d)
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ Replay
def test_compare_runs_marks_divergences_and_jumps(window, qapp):
    a, b = _runs(window, [('base', {'fidelity': 'carla'}), ('cand', {'fidelity': 'carla', 'swerve_at': 4.0,
                                                                     'extra_actor_at': 7.0})])
    rp = window.go('replay')
    rp.compare_runs(a, b)
    assert pump(qapp, lambda: rp.sync.data is not None, 20), rp.sync.here.text()
    sy = rp.sync
    assert rp.right.currentIndex() == 1 and window.context.get('run') == b
    assert sy.slider.value() == 0 and sy.slider.spans and len(sy.events) >= 1
    sy.jump_first()
    k = sy.slider.value()
    assert k == sy.events[0]['start'] and 4.2 <= sy.data['grid'][k] <= 4.4
    assert 'diverging here' in sy.here.text() and sy.view_a.diverging and sy.view_b.k == k
    sy.slider.setValue(0)
    sy.next_divergence()
    assert sy.slider.value() == sy.events[0]['start']
    sy.slider.setValue(len(sy.data['grid']) - 1)
    sy.prev_divergence()
    assert sy.slider.value() == sy.events[-1]['start']
    # the numbers: both CARLA, like for like; what both recorded (they are not replays of one log)
    assert pump(qapp, lambda: rp.table.topLevelItemCount() > 0, 5)
    rows = {rp.table.topLevelItem(i).text(0): rp.table.topLevelItem(i).text(3)
            for i in range(rp.table.topLevelItemCount())}
    assert rows['Ego position, largest difference (m)'].startswith('3.000')
    assert rp.cmp_fid.text() == 'CARLA' and not sy.likeness.isVisibleTo(rp)
    sy.view_a.grab()
    sy.slider.grab()                                                     # paints (ticks) without error


def test_replay_fidelity_mismatch_and_unknown(window, qapp):
    a, b, c = _runs(window, [('real', {'fidelity': 'carla'}), ('fake', {'fidelity': 'fake'}),
                             ('old', {'fidelity': None})])
    rp = window.go('replay')
    rp.compare_runs(a, b)
    assert pump(qapp, lambda: rp.sync.data is not None, 20)
    assert rp.sync.likeness.isVisibleTo(rp) and 'Not like for like' in rp.sync.likeness.label.text()
    assert rp.cmp_fid.text() == 'Mixed fidelity'
    rp.compare_runs(a, c)
    assert pump(qapp, lambda: rp.sync.data is not None and rp.sync.paths[1] == c, 20)
    assert not rp.sync.likeness.isVisibleTo(rp) and rp.sync.likeness_note.isVisibleTo(rp)
    assert rp.sync.fids['B'].text() == 'Fidelity unknown'


# ------------------------------------------------------------------------------------------------ Data and Home
def test_data_fidelity_column_header_chip_and_send_to(window, qapp):
    from falcon_autonomy.ui.pages.data import COL
    a, b = _runs(window, [('drive-fake', {'fidelity': 'fake', 'min_ttc': 2.5}), ('drive-old', {'fidelity': None})])
    data = window.go('data')
    data.refresh()
    br = data.browse
    rows = {br.runs.topLevelItem(i).data(0, 256): br.runs.topLevelItem(i) for i in range(br.runs.topLevelItemCount())}
    assert rows[a].text(COL['Fidelity']) == 'Fake simulator' and 'test double' in rows[a].toolTip(COL['Fidelity'])
    assert rows[b].text(COL['Fidelity']) == 'Fidelity unknown'
    assert rows[a].text(COL['Min TTC']) == '2.50 s' and rows[b].text(COL['Min TTC']) == '—'
    assert rows[a].data(0, Qt.ItemDataRole.UserRole + 1) == a                # the Run cell copies the folder
    # choosing a run: the header's chip, and the window's context (with the run's seed, scenario, world)
    br.runs.setCurrentItem(rows[a])
    assert br.head_fid.text() == 'Fake simulator' and br.head_id.full == a
    assert window.context.get('run') == a and window.context.get('seed') == '3'
    assert window.context.get('scenario') == 'lead-brake' and window.context.get('world') == 'w-town10'
    # Send to ▸ in the run's context menu: Replay among the targets, Data itself not
    br.runs.clearSelection()
    rows[a].setSelected(True)
    m = br.build_menu(rows[a])
    titles = [x.text() for x in m.send_to.actions()]
    assert 'Replay' in titles and 'Data' not in titles
    # Send to ▸ Replay: the run imported as a log (kind falcon), with the run's own fidelity
    rp = window.page('replay')
    next(x for x in m.send_to.actions() if x.text() == 'Replay').trigger()
    assert window.stack.currentWidget() is rp
    assert pump(qapp, lambda: rp.log is not None and rp.log.meta['source']['format'] == 'falcon', 20), \
        rp.info.vals['Log'].text()
    assert rp.log.meta['fidelity'] == 'fake' and rp.log_fid.text() == 'Fake simulator'
    # Send to ▸ Data from elsewhere: Browse with the run's row selected
    from falcon_autonomy.ui.context import targets
    targets(window)['data'][1](b)
    assert window.stack.currentWidget() is data and br.run_dir == b and br.runs.currentItem().data(0, 256) == b


def test_home_recent_runs_fidelity_and_menu(window, qapp):
    a, b = _runs(window, [('n-1', {'fidelity': 'neural'}), ('n-2', {'fidelity': 'carla'})])
    home = window.go('home')
    home.refresh()
    by = {home.runs.topLevelItem(i).data(0, 256): home.runs.topLevelItem(i)
          for i in range(home.runs.topLevelItemCount())}
    assert by[a].text(6) == 'Neural render' and by[b].text(6) == 'CARLA'
    assert home.runs.height() >= home.runs.header().height() + 2 * home.runs.sizeHintForRow(0)   # every row whole
    m = home.build_run_menu(by[a])
    sub = next(x.menu() for x in m.actions() if x.menu() is not None)
    assert {'Data', 'Replay'} <= {x.text() for x in sub.actions()}
    assert home.runs._ask_target(by[a]) == ('run', {'run': a})
    home.open_run(by[b])
    assert window.context.get('run') == b and window.stack.currentWidget().key == 'data'


def test_runs_compare_not_like_for_like_callout(window, qapp):
    a, b, c = _runs(window, [('cmp-carla', {'fidelity': 'carla'}), ('cmp-fake', {'fidelity': 'fake'}),
                             ('cmp-carla2', {'fidelity': 'carla'})])
    data = window.go('data')
    data.compare_runs([a, b])
    cmp_ = data.compare
    assert cmp_.likeness.isVisibleTo(cmp_) and 'CARLA vs Fake simulator' in cmp_.likeness.label.text()
    assert 'Not like for like' in cmp_.likeness.label.text()
    first = cmp_.table.topLevelItem(0)
    assert first.text(0) == 'Fidelity' and first.text(1) == 'CARLA' and first.text(2) == 'Fake simulator'
    data.compare_runs([a, c])
    assert not cmp_.likeness.isVisibleTo(cmp_) and not cmp_.likeness_note.isVisibleTo(cmp_)
    # A and B side by side in Replay, one click
    rp = cmp_.side_by_side()
    assert rp is window.stack.currentWidget() and rp.sync.paths == (a, c)
    assert pump(qapp, lambda: rp.sync.data is not None, 20)
