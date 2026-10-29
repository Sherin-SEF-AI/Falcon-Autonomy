"""The run index and its query language, drive sets and tags, side-by-side comparison and dataset health
(core.runindex, core.runcompare, core.datahealth; Data ▸ Browse, Compare and Health; the Drive set bars)."""
import json
import os

import numpy as np
import pytest

from falcon_autonomy.core import datahealth as DH
from falcon_autonomy.core import runcompare as RC
from falcon_autonomy.core import runindex as RI


# ------------------------------------------------------------------------------------------------------- fixtures
def make_run(root, name, *, fps=20.0, steps=40, speed=10.0, world='world-town10', preset='ClearNoon', weather=None,
             status='ok', started='2026-10-28T10:00:00Z', scenario=None, collision_t=None, teleport_at=None,
             images=True, objects=None, worker='drive', odd_out=None):
    """A run folder as the drive worker writes it (run.json, frames.jsonl, dataset.json, images and labels)."""
    import cv2
    from falcon_autonomy.core import weather as W
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    w = W.resolve(weather or {'preset': preset})
    st = {'name': name, 'fps': fps, 'seed': 3, 'town': 'Town10HD_Opt', 'xodr': '', 'weather': w,
          'weather_source': weather or {'preset': preset}, 'project': {'world': world, 'rig': 'rig-default'}}
    run = {'schema': 'falcon.run/1', 'worker': worker, 'status': status, 'started': started, 'name': name,
           'provenance': {'app_version': 'test', 'seed': 3, 'settings': st},
           'performance': {'steps': steps, 'sim_s': steps / fps}, 'notes': []}
    if scenario:
        st['scenario'] = {'id': scenario[0], 'name': scenario[0], 'values': scenario[1]}
        actors = {'lead': {'collision': collision_t is not None, 'collision_t': collision_t,
                           'min_ttc_s': 1.25, 'min_ttc_t': 1.0}}
        run['scenario'] = {'criticality': {'actors': actors, 'overall': {'min_ttc_s': 1.25}, 'other_collisions': []},
                           'passed': collision_t is None, 'fired': [[0, 0.5]]}
    if odd_out:
        run['odd_monitor'] = {'time_outside_s': odd_out[1] - odd_out[0]}
        with open(os.path.join(d, 'odd-monitor.json'), 'w') as fh:
            json.dump({'segments': [{'t0': 0.0, 't1': odd_out[0], 'inside': True},
                                    {'t0': odd_out[0], 't1': odd_out[1], 'inside': False, 'reasons': ['rain']}]}, fh)
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump(run, fh)
    with open(os.path.join(d, 'dataset.json'), 'w') as fh:
        json.dump({'sensors': [{'id': 'cam', 'width': 64, 'height': 36}]}, fh)
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    with open(os.path.join(d, 'frames.jsonl'), 'w') as fh:
        for i in range(steps):
            rec = {'step': i, 'frame': 100 + i, 'sim_t': round(5.0 + i / fps, 6), 'speed_mps': speed,
                   'ego': {'x': i * speed / fps, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}, 'files': {}}
            if teleport_at is not None and i == teleport_at:
                rec['teleport'] = {'from': [0, 0], 'to': [1, 1]}
            if images:
                os.makedirs(os.path.join(d, 'rgb', 'cam'), exist_ok=True)
                img = np.full((36, 64, 3), 90, np.uint8) if i < steps // 2 else \
                    rng.integers(0, 255, (36, 64, 3), dtype=np.uint8)
                rel = os.path.join('rgb', 'cam', f'{i:06d}.jpg')
                cv2.imwrite(os.path.join(d, rel), img)
                rec['files']['cam/rgb'] = rel
            if objects is not None:
                os.makedirs(os.path.join(d, 'objects', 'cam'), exist_ok=True)
                rel = os.path.join('objects', 'cam', f'{i:06d}.json')
                with open(os.path.join(d, rel), 'w') as of:
                    json.dump({'objects': objects(i)}, of)
                rec['files']['cam/objects'] = rel
            fh.write(json.dumps(rec) + '\n')
    return d


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / 'ws'
    runs = w / 'runs'
    make_run(str(runs), 'a-clear', started='2026-10-27T09:00:00Z', steps=40)
    make_run(str(runs), 'b-rain', preset='SoftRainNight', started='2026-10-28T09:00:00Z', steps=60, speed=5.0,
             teleport_at=30, odd_out=(1.0, 2.0))
    make_run(str(w / 'studies' / 'search-x' / 'runs'), 'c-scn', started='2026-10-29T09:00:00Z', steps=20,
             scenario=('lead_brake', {'speed': 30.0, 'gap': 20.0}), collision_t=0.8)
    make_run(str(runs), 'd-failed', status='failed', started='2026-10-29T10:00:00Z', steps=5, images=False)
    return str(w)


# ------------------------------------------------------------------------------------------------------- the index
def test_summaries_come_from_what_the_run_recorded(ws):
    es = {e['id']: e for e in RI.build(ws)}
    assert set(es) == {'a-clear', 'b-rain', 'c-scn', 'd-failed'}
    a, b, c = es['a-clear'], es['b-rain'], es['c-scn']
    assert a['kind'] == 'drive' and a['weather'] == 'ClearNoon' and a['weather_kind'] == 'clear' and a['tod'] == 'day'
    assert a['duration_s'] == 2.0 and a['frames'] == 40 and a['speed_max_kmh'] == 36.0
    assert a['distance_m'] == pytest.approx(19.5)                 # 39 steps of 0.5 m
    assert a['collisions'] is None and a['min_ttc_s'] is None and a['events'] == []
    assert b['weather_kind'] == 'rain' and b['tod'] == 'night' and b['teleports'] == [1.5]
    assert set(b['events']) == {'teleport', 'odd_exit'} and b['odd_outside_s'] == 1.0
    assert b['distance_m'] == pytest.approx(58 * 0.25)            # the jump at the teleport is not distance
    assert c['kind'] == 'search' and c['scenario'] == 'lead_brake' and c['collisions'] == 1
    assert c['min_ttc_s'] == 1.25 and {'collision', 'failed_check'} <= set(c['events'])
    assert c['rel'] == os.path.join('studies', 'search-x', 'runs', 'c-scn')
    assert [e['id'] for e in RI.build(ws)][:2] == ['d-failed', 'c-scn']            # newest first


def test_the_cache_is_reused_and_refreshed_when_a_run_changes(ws, monkeypatch):
    RI.build(ws)
    assert os.path.isfile(os.path.join(ws, RI.CACHE_FILE))
    calls = []
    real = RI.summarise
    monkeypatch.setattr(RI, 'summarise', lambda d, w='': calls.append(d) or real(d, w))
    RI.build(ws)
    assert calls == []                                            # nothing changed: nothing re-read
    p = os.path.join(ws, 'runs', 'a-clear', 'frames.jsonl')
    with open(p, 'a') as fh:
        fh.write(json.dumps({'step': 40, 'sim_t': 7.0, 'speed_mps': 20.0, 'files': {}}) + '\n')
    es = {e['id']: e for e in RI.build(ws)}
    assert [os.path.basename(c) for c in calls] == ['a-clear'] and es['a-clear']['speed_max_kmh'] == 72.0


def test_the_query_language(ws):
    es = RI.with_tags(RI.build(ws), {'run_tags': {os.path.join('runs', 'a-clear'): ['golden', 'night-set']}}, ws)

    def q(text):
        return sorted(e['id'] for e in RI.search(es, text))
    assert q('') == ['a-clear', 'b-rain', 'c-scn', 'd-failed']
    assert q('weather:rain') == ['b-rain'] and q('weather:SoftRain') == ['b-rain']
    assert q('tod:night') == ['b-rain'] and q('status:ok -kind:search') == ['a-clear', 'b-rain']
    assert q('events:collision') == ['c-scn'] and q('events:teleport') == ['b-rain']
    assert q('dur>2') == ['b-rain'] and q('dur>=2 dur<3') == ['a-clear'] and q('frames=20') == ['c-scn']
    assert q('after:2026-10-28') == ['b-rain', 'c-scn', 'd-failed'] and q('before:2026-10-28') == ['a-clear']
    assert q('after:2026-10-29T09:30') == ['d-failed']
    assert q('tag:golden') == ['a-clear'] and q('tag:night') == ['a-clear'] and q('-tag:golden status:ok') == \
        ['b-rain', 'c-scn']
    assert q('ttc<2') == ['c-scn'] and q('collisions>0') == ['c-scn'] and q('speed>30') == ['a-clear', 'c-scn', 'd-failed'] and q('speed>40') == []
    assert q('"b-rain"') == ['b-rain'] and q('lead') == ['c-scn'] and q('scenario:lead world:town10') == ['c-scn']
    for bad, why in (('dur>abc', 'not a number'), ('world>3', 'not a number field'), ('after:yesterday', 'not a date'),
                     ('"open', 'quotes')):
        with pytest.raises(RI.QueryError, match=why):
            RI.parse(bad)
    assert [e['id'] for e in RI.sort(es, 'duration', reverse=False)] == ['d-failed', 'c-scn', 'a-clear', 'b-rain']


def test_drive_sets_resolve_lists_and_saved_queries(ws):
    es = RI.build(ws)
    a = os.path.join(ws, 'runs', 'a-clear')
    lst = {'kind': 'list', 'runs': [RI.key_of(a, ws), 'runs/gone']}
    assert RI.key_of(a, ws) == os.path.join('runs', 'a-clear') and RI.path_of('runs/x', ws) == os.path.join(ws, 'runs/x')
    assert RI.resolve_set(lst, es, ws) == [a]                     # a run no longer on disk drops out
    qs = {'kind': 'query', 'query': 'status:ok dur>=2'}
    assert sorted(os.path.basename(p) for p in RI.resolve_set(qs, es, ws)) == ['a-clear', 'b-rain']
    make_run(os.path.join(ws, 'runs'), 'e-new', steps=80, started='2026-10-30T00:00:00Z')
    assert 'e-new' in [os.path.basename(p) for p in RI.resolve_set(qs, RI.build(ws), ws)]   # re-evaluated
    assert RI.describe_set(qs) == 'query: status:ok dur>=2' and RI.describe_set(lst) == '2 runs'


def test_falcon_runs_cli(ws, capsys):
    from falcon_autonomy.cli import main
    assert main(['runs', 'search', ws, 'weather:rain']) == 0
    out = capsys.readouterr().out
    assert 'runs/b-rain' in out and '1 run(s)' in out
    assert main(['runs', 'search', ws, 'dur>x']) == 2


# ------------------------------------------------------------------------------------------------------- comparison
def test_runs_at_different_rates_align_by_time(tmp_path):
    a = make_run(str(tmp_path), 'fast', fps=20.0, steps=40, images=False)          # 2 s
    b = make_run(str(tmp_path), 'slow', fps=10.0, steps=10, images=False)          # 1 s
    ta, tb = RC.times(RC.load_frames(a)), RC.times(RC.load_frames(b))
    assert ta[0] == 0.0 and tb[-1] == pytest.approx(0.9)
    assert RC.frame_at(ta, 0.52) == 10 and RC.frame_at(tb, 0.52) == 5            # the same instant, not the same step
    assert RC.frame_at(tb, 1.5) is None and RC.frame_at(ta, 1.5) == 30           # past the short run's end
    assert RC.frame_at(tb, 0.99) == 9                                              # within a step of its end


def test_the_comparison_table_says_what_was_not_recorded(ws):
    es = {e['id']: e for e in RI.build(ws)}
    rows = {r['key']: r for r in RC.table([es['a-clear'], es['c-scn']], [None, {'violations': 2,
                                                                                 'lane_margin_min': None,
                                                                                 'lane_margin_median': None}])}
    assert rows['duration_s']['cells'][1]['delta'] == '-1.0' and rows['duration_s']['cells'][0]['text'] == '2.0 s'
    assert rows['collisions']['cells'][0]['text'] == 'not recorded (no scenario)'
    assert rows['collisions']['cells'][1]['value'] == 1 and rows['collisions']['cells'][1]['delta'] is None
    assert rows['rule_violations']['cells'][0]['text'] == 'checking…' and rows['rule_violations']['cells'][1]['text'] \
        == '2'
    assert rows['lane_margin_min']['cells'][1]['text'] == 'not recorded (no lane map)'
    assert rows['odd_outside_s']['cells'][0]['text'] == 'not monitored'
    ev = RC.events(os.path.join(ws, 'studies', 'search-x', 'runs', 'c-scn'))
    assert [(e['kind'], e['t']) for e in ev] == [('trigger', 0.5), ('collision', 0.8), ('min_ttc', 1.0)]
    ev = RC.events(os.path.join(ws, 'runs', 'b-rain'))
    assert [(e['kind'], e['t'], e['t1']) for e in ev] == [('odd_exit', 1.0, 2.0), ('teleport', 1.5, None)]


# ------------------------------------------------------------------------------------------------------- health
def test_dataset_health_measures_balance_coverage_duplicates_and_labels(tmp_path):
    from falcon_autonomy.core.project import Project

    def objs(i):
        if i % 10 == 0:
            return []                                                              # 4 of 40 frames empty
        out = [{'class': 'car', 'bbox': [10, 10, 30, 30]}]
        if i % 4 == 0:
            out.append({'class': 'pedestrian', 'bbox': [2, 2, 5, 6]})              # tiny: 3 × 4 px
        if i == 5:
            out.append({'class': 'car', 'bbox': [40, 10, 40, 20]})                 # no area
        if i == 7:
            out.append({'class': 'car', 'bbox': [70, 10, 90, 30]})                 # outside a 64 px image
        return out
    a = make_run(str(tmp_path), 'h1', steps=40, objects=objs, scenario=('lead_brake', {'speed': 22.0, 'gap': 12.0}))
    b = make_run(str(tmp_path), 'h2', steps=20, preset='ClearNight', objects=lambda i: [{'class': 'car',
                                                                                         'bbox': [0, 0, 20, 20]}])
    doc = Project.new('T').doc
    r = DH.analyse([a, b], '', doc)
    assert r['frames'] == 60 and r['frames_read'] == 60 and r['labelled_frames'] == 60
    assert r['classes']['car'] == 36 + 2 + 20 and r['classes']['pedestrian'] == 8      # every 4th, not the empty
    assert r['empty_frames'] == 4
    assert r['labels']['tiny'] == 8 and r['labels']['degenerate'] == 1 and r['labels']['outside'] == 1
    dup = r['duplicates']                         # the first half of each run is one flat grey frame
    assert dup['hashed'] == 60 and dup['near_duplicates'] >= 20 + 10 - 1
    cells = {(c['world'], c['tod'], c['weather']): c['frames'] for c in r['coverage']['cells']}
    assert cells[('world-town10', 'day', 'clear')] == 40 and cells[('world-town10', 'night', 'clear')] == 20
    assert r['coverage']['of'] == 12 and r['coverage']['empty'] == 10
    s = r['suggestions'][0]
    assert s['frames'] == 0 and s['preset'] in DH.PRESET_FOR.values()
    v = {h['param']: h for h in r['values']}
    assert v['speed']['lo'] == 20 and v['speed']['hi'] == 60 and v['speed']['range_from'] == 'the scenario'
    assert sum(v['speed']['counts']) == 1 and v['speed']['empty_bins'] == 9
    few = DH.analyse([a, b], '', doc, max_frames=20)
    assert few['every'] == 3 and few['frames_read'] == 20


def test_dhash_and_hamming():
    g = np.tile(np.arange(64, dtype=np.uint8), (36, 1))
    h = DH.dhash(g)
    assert h == 2 ** 64 - 1                                      # brighter to the right everywhere
    assert list(DH.hamming(h, np.array([h, 0], np.uint64))) == [0, 64]


# ------------------------------------------------------------------------------------------------------- the GUI
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


def _gui_runs(window):
    runs = window.service.runs_dir()
    a = make_run(runs, 'g-a', started='2026-10-27T09:00:00Z', steps=40, fps=20.0,
                 objects=lambda i: [{'class': 'car', 'bbox': [4, 4, 20, 20]}])
    b = make_run(runs, 'g-b', started='2026-10-28T09:00:00Z', steps=10, fps=10.0, preset='HardRainNight',
                 teleport_at=5)
    for d in (a, b):                          # finished jobs are catalogued; these were written directly
        window.service.catalog.add_run(d)
    return a, b


@pytest.mark.gui
def test_browse_searches_sorts_and_tags(window, qapp, monkeypatch):
    from PyQt6.QtCore import Qt
    a, b = _gui_runs(window)
    data = window.go('data')
    data.refresh()
    br = data.browse
    assert br.runs.topLevelItemCount() == 2 and br.runs.topLevelItem(0).data(0, 256) == b    # newest first
    br.filter.setText('weather:rain')
    assert br.visible_runs() == [b] and br.query_state.text() == '1 of 2 runs'
    br.filter.setText('dur>abc')
    assert br.query_state.text().startswith('⚠') and not br.save_query_btn.isEnabled()
    br.filter.setText('')
    br.runs.sortByColumn(4, Qt.SortOrder.AscendingOrder)                  # duration: 1 s before 2 s
    assert br.runs.topLevelItem(0).data(0, 256) == b
    br.runs.clearSelection()
    br.runs.topLevelItem(1).setSelected(True)
    monkeypatch.setattr('PyQt6.QtWidgets.QInputDialog.getText', lambda *a, **k: ('golden, day ', True))
    br.edit_tags()
    key = os.path.join('runs', 'g-a')
    assert window.project.doc['run_tags'] == {key: ['day', 'golden']}
    assert pump(qapp, lambda: any(br.runs.topLevelItem(i).text(6) == 'day, golden'
                                  for i in range(br.runs.topLevelItemCount())), 3)
    br.filter.setText('tag:golden')
    assert br.visible_runs() == [a]
    window.undo.undo()                                                    # tags are a project edit: undoable
    assert 'run_tags' not in window.project.doc
    assert open(os.path.join(a, 'run.json')).read().count('golden') == 0   # the run itself is never touched


@pytest.mark.gui
def test_drive_sets_from_browse_check_drives_in_models_and_training(window, qapp, monkeypatch):
    from PyQt6.QtCore import Qt
    a, b = _gui_runs(window)
    data = window.go('data')
    data.refresh()
    br = data.browse
    br.runs.clearSelection()
    for i in range(br.runs.topLevelItemCount()):
        if br.runs.topLevelItem(i).data(0, 256) == a:
            br.runs.topLevelItem(i).setSelected(True)
    monkeypatch.setattr('PyQt6.QtWidgets.QInputDialog.getText', lambda *a, **k: ('Golden', True))
    br.save_selection_set()
    br.filter.setText('weather:rain')
    monkeypatch.setattr('PyQt6.QtWidgets.QInputDialog.getText', lambda *a, **k: ('Rainy', True))
    br.save_query_set()
    sets = window.project.doc['drive_sets']
    assert [(s['name'], s['kind']) for s in sets] == [('Golden', 'list'), ('Rainy', 'query')]
    assert sets[0]['runs'] == [os.path.join('runs', 'g-a')] and sets[1]['query'] == 'weather:rain'
    assert data.show_drive_set(sets[1]['id']) and br.filter.text() == 'weather:rain'
    assert data.select_drive_set(sets[0]['id']) and br.selected_runs() == [a] and br.filter.text() == ''
    assert not data.show_drive_set('ds-missing')
    models = window.go('models')
    models.refresh()
    bar = models.run_sets
    assert bar.combo.count() == 3                                          # the prompt and the two sets
    bar.combo.setCurrentIndex(bar.combo.findData(sets[0]['id']))
    assert bar.apply(sets[0]) == (1, 1) and models.checked_runs() == [a]
    assert bar.apply(sets[1]) == (1, 1) and models.checked_runs() == [b]
    tr = window.go('training')
    tr.refresh()
    for i in range(tr.det_runs.count()):
        tr.det_runs.item(i).setCheckState(Qt.CheckState.Checked)
    from falcon_autonomy.ui.drivesets import DriveSetBar
    tbar = next(w for w in tr.findChildren(DriveSetBar) if w.runs is tr.det_runs)
    assert tbar.combo.count() == 3
    monkeypatch.setattr('PyQt6.QtWidgets.QInputDialog.getText', lambda *a, **k: ('Both', True))
    tbar.save()
    assert sorted(window.project.doc['drive_sets'][-1]['runs']) == [os.path.join('runs', 'g-a'),
                                                                     os.path.join('runs', 'g-b')]
    window.undo.undo()
    assert len(window.project.doc['drive_sets']) == 2


@pytest.mark.gui
def test_compare_two_runs_on_one_clock(window, qapp):
    a, b = _gui_runs(window)
    data = window.go('data')
    data.refresh()
    data.compare_runs([a, b])
    cmp = data.compare
    assert data.tabs.currentWidget() is cmp and [r['slot'] for r in cmp.runs] == ['A', 'B']
    assert cmp.body.isVisible() and cmp.slider.maximum() == 1950                      # the longer run: 1.95 s
    assert pump(qapp, lambda: all(r['rules'] is not None for r in cmp.runs), 60)
    assert pump(qapp, lambda: cmp.views[0].pixmap() is not None and not cmp.views[0].pixmap().isNull(), 10)
    cmp.seek(1.5)
    assert 'ended at 0.90 s' in cmp.captions[1].text() and 't 1.50 s' in cmp.captions[0].text()
    labels = [cmp.table.topLevelItem(i).text(0) for i in range(cmp.table.topLevelItemCount())]
    assert 'Duration' in labels and 'Rule violations' in labels
    dur = cmp.table.topLevelItem(labels.index('Duration'))
    assert dur.text(1) == '2.0 s' and dur.text(2) == '1.0 s  (-1.0)'
    col = cmp.table.topLevelItem(labels.index('Collisions'))
    assert col.text(1) == 'not recorded (no scenario)'
    rows = cmp.timeline.rows
    assert [r[0] for r in rows] == ['A', 'B'] and any(e['kind'] == 'teleport' for e in rows[1][2])
    cmp.timeline.seek.emit(0.5)
    assert abs(cmp.slider.value() - 500) <= 1


@pytest.mark.gui
def test_health_suggests_and_record_takes_the_weather(window, qapp):
    a, b = _gui_runs(window)
    data = window.go('data')
    data.refresh()
    h = data.health
    h.source.setCurrentIndex(0)                                           # all recorded drives
    h.analyse()
    assert pump(qapp, lambda: h.report is not None, 30)
    r = h.report
    assert r['frames'] == 50 and r['classes'] == {'car': 40}
    assert h.classes.topLevelItemCount() == 1 and h.sugg.topLevelItemCount() > 0
    assert 'Near duplicates' in h.summary.vals and 'Coverage' in h.summary.vals
    it = h.sugg.topLevelItem(0)
    s = it.data(0, 256)
    h.record_cell(it)
    rec = data.record
    assert data.tabs.currentIndex() == 0 and rec.world.currentData() == s['world']
    assert rec.weather.currentData() == s['preset'] and 'health check' in rec.note.text()
    rec.seconds.setValue(0.5)
    rec.record()
    job = window.queue.jobs[-1]
    cfg = json.load(open(os.path.join(job.spec.run_dir, 'drive.json')))
    assert cfg['weather_source'] == {'preset': s['preset'], 'override': 'Data ▸ Record'}
    from falcon_autonomy.core import weather as W
    assert cfg['weather'] == W.resolve({'preset': s['preset']})
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120), job.log
    assert job.state == 'done', job.log
    assert window.project.doc['worlds'][0]['weather'] == {'preset': 'ClearNoon'}       # the world is unchanged
