"""The closed loop (core.loop) against the fake carla: the weak-cell rule on hand-made frames with known answers; the
ODD (validation, the Bengaluru monsoon-dusk example, seeded weather ranges, the evaluation set's key); the evaluation
set is never trained on (refused by folder, by a folder holding one, by a dataset version and by cache key); then a
whole loop with tiny drives — the synthetic detector, which the retrain step refuses to retrain (it says why), and a
tiny scratch detector Falcon trained on the CPU, which goes through all eight steps: costs recorded before each step,
the retrain continuing from the model's own settings on its sources plus the new drives and none of the evaluation
drives, before and after on the same set with paired intervals; resume runs only what changed; the next iteration
reuses the retrained model's evaluation; the project's loop.iterations; the command."""
import copy
import json
import os
import subprocess
import sys

import pytest

from falcon_autonomy.core import loop as L
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import provenance as PV

from conftest import FAKE, ROOT

SYN = {'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
       'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector', 'licence': 'GPL-3.0-or-later'}
TINY = {'arch': 'fasterrcnn_mobilenet_v3_small_fpn', 'init': 'scratch', 'size': [128, 64], 'batch': 4, 'epochs': 1,
        'workers': 0, 'device': 'cpu', 'box_head_size': 64, 'verify_images': 2, 'export': True}


# ------------------------------------------------------------------------------------------------ weak cells
def _frames(conditions):
    """Hand-made frames: per drive two cars at 10 m (always found) and pedestrians at 10 m (found) and 50 m (found
    in condition A, all missed in B); in B also 3 bicycles at 30 m, all missed (too few to be a finding)."""
    frames = []
    for d, cond in enumerate(conditions):
        for step in range(5):
            gt, det = [], []
            for k in range(2):
                gt.append({'bbox': [10 + 40 * k, 10, 30, 20], 'class': 'car', 'ignore': 0, 'range_m': 10.0})
            gt.append({'bbox': [100, 10, 10, 25], 'class': 'pedestrian', 'ignore': 0, 'range_m': 10.0})
            gt.append({'bbox': [130, 10, 6, 12], 'class': 'pedestrian', 'ignore': 0, 'range_m': 50.0})
            for g in gt:
                if not (cond == 'B' and g['class'] == 'pedestrian' and g['range_m'] == 50.0):
                    det.append({'bbox': list(g['bbox']), 'class': g['class'], 'score': 0.9})
            if cond == 'B' and step < 3 and d == 1:
                gt.append({'bbox': [200, 10, 8, 8], 'class': 'bicycle', 'ignore': 0, 'range_m': 30.0})
            frames.append({'image_id': f'd{d}/{step}', 'drive': f'd{d}', 'step': step, 'gt': gt, 'det': det,
                           'conditions': {}})
    return frames


def test_weak_cells_on_hand_made_frames():
    from falcon_autonomy.core import metrics as M
    conds = ['A', 'B', 'A', 'B', 'A', 'B']
    frames = _frames(conds)
    cell_of = {f'd{d}': c for d, c in enumerate(conds)}
    rows = L.cell_table(frames, cell_of, ['car', 'pedestrian', 'bicycle'], [0, 20, 40, 60], n_boot=200, seed=0)
    by = {(r['class'], r['bin'][0], r['condition']): r for r in rows}
    assert by[('pedestrian', 40.0, 'B')]['AP'] == 0.0 and by[('pedestrian', 40.0, 'B')]['ci_high'] == 0.0
    assert by[('pedestrian', 40.0, 'B')]['n_objects'] == 15 and by[('pedestrian', 40.0, 'B')]['unit'] == 'drive'
    assert by[('pedestrian', 40.0, 'A')]['AP'] == pytest.approx(1.0)
    assert by[('car', 0.0, L.ALL)]['AP'] == pytest.approx(1.0) and by[('car', 0.0, L.ALL)]['n_objects'] == 60
    assert by[('pedestrian', 40.0, L.ALL)]['AP'] == pytest.approx(0.5, abs=0.02)      # half of them missed
    assert by[('bicycle', 20.0, 'B')]['AP'] == 0.0 and by[('bicycle', 20.0, 'B')]['n_objects'] == 3
    overall = M.coco_evaluate(frames)['AP']
    assert 0.5 < overall < 1.0
    chosen = L.rank_weak(rows, overall, min_objects=10, max_cells=3)
    # the surest shortfall first; the "all conditions" pedestrian cell is the same class × bin: not chosen twice;
    # the bicycles are missed too but are 3 objects — not a finding
    assert [(c['class'], c['bin'][0], c['condition'], c['rank']) for c in chosen] == [('pedestrian', 40.0, 'B', 1)]
    assert not by[('bicycle', 20.0, 'B')]['weak'] and not by[('bicycle', 20.0, L.ALL)]['weak']
    assert not any(r['weak'] for r in rows if r['class'] == 'car')
    assert L.rank_weak(rows, overall, min_objects=100, max_cells=3) == []
    assert L.rank_weak(rows, None, min_objects=1, max_cells=3) == []
    # a model weak everywhere has no weak cell; the fallback targets the lowest cells, and says so
    assert L.rank_weak(rows, 0.0, min_objects=10, max_cells=1) == []
    (fb,) = L.rank_weak(rows, 0.0, min_objects=10, max_cells=1, fallback='lowest')
    assert fb['fallback'] and (fb['class'], fb['bin'][0], fb['condition']) == ('pedestrian', 40.0, 'B')
    chosen = L.rank_weak(rows, overall, min_objects=10, max_cells=3)
    # a paired comparison of the cell: a model that now finds them is an improvement
    fixed = copy.deepcopy(frames)
    for f in fixed:
        f['det'] = [{'bbox': list(g['bbox']), 'class': g['class'], 'score': 0.9} for g in f['gt']]
    after = L.cell_table(fixed, cell_of, ['pedestrian'], [0, 20, 40, 60], n_boot=200)
    (c,) = L.compare_cells(frames, fixed, cell_of, chosen, after, [0, 20, 40, 60], 200, 0)
    assert c['before']['value'] == 0.0 and c['after']['value'] == pytest.approx(1.0)
    assert c['diff']['value'] == pytest.approx(1.0) and c['diff']['ci_low'] > 0
    assert L.verdict(c['diff']) == 'improvement' and L.verdict({'ci_low': -0.1, 'ci_high': 0.2}).startswith('no ')


# ------------------------------------------------------------------------------------------------ the ODD
def test_odd_validation_example_weather_and_key(project):
    odd, missing = L.example_odd(project)
    assert any('Bengaluru' in m for m in missing) and any('Indian' in m for m in missing)
    assert odd['traffic_profile'] == 'tp-india-mixed' and odd['weathers'][0]['name'] == 'monsoon dusk'
    _, errs = L.normalise_odd(odd, project)
    assert any('world' in e for e in errs) and any('boxes2d' in e for e in errs)
    # the project gets what the example needs: an imported place, the Indian profile, a labelled camera
    from falcon_autonomy.core.traffic import preset
    project.doc['worlds'].append({'id': 'world-blr', 'name': 'Central Bengaluru', 'kind': 'osm',
                                  'osm': {'source': 'blr.osm', 'handedness': 'LHT'}, 'xodr': 'worlds/blr.xodr',
                                  'traffic_profile': 'tp-india-mixed', 'weather': {'preset': 'ClearNoon'}})
    project.doc['traffic_profiles'].append(preset('tp-india-mixed'))
    project.doc['rigs'][0]['sensors'][0]['labels'] = ['semantic', 'instance', 'depth', 'boxes2d']
    odd, missing = L.example_odd(project)
    assert missing == [] and odd['worlds'] == ['world-blr']
    norm, errs = L.normalise_odd(odd, project)
    assert errs == [] and norm['evaluation']['n_boot'] == 1000 and norm['range_bins'] == [0, 20, 40, 60, 80]
    assert [c['label'] for c in L.odd_cells(norm)] == ['world-blr · monsoon dusk',
                                                       'world-blr · monsoon dusk, downpour']
    from falcon_autonomy.core.project import validate
    doc = copy.deepcopy(project.doc)
    doc['loop']['odd'] = norm
    assert validate(doc) == []                       # the ODD lives in the project's loop.odd
    # the dusk: each drive draws the ranged fields in their ranges, the same for the same seed
    a, b = L.sample_weather(norm['weathers'][0], 7), L.sample_weather(norm['weathers'][0], 7)
    assert a == b and -2 <= a['sun_altitude_angle'] <= 6 and 50 <= a['precipitation'] <= 90 and 'ranges' not in a
    assert L.sample_weather(norm['weathers'][0], 8) != a
    from falcon_autonomy.core.weather import describe
    assert 'rain' in describe(a)
    # the evaluation set's key follows what changes its drives, not the search's settings
    k = L.odd_key(norm, project)
    assert L.odd_key(dict(norm, search=dict(norm['search'], budget=99)), project) == k
    assert L.odd_key(dict(norm, evaluation=dict(norm['evaluation'], drives_per_cell=9)), project) != k
    project.doc['rigs'][0]['sensors'][0]['width'] = 320
    assert L.odd_key(norm, project) != k
    bad, errs = L.normalise_odd({'id': 'x y', 'worlds': ['nope'], 'rig': 'nope', 'weathers': [{'name': 'w',
                                 'preset': 'Sunny', 'ranges': {'fog_density': [9, 1]}}], 'classes': ['unicorn'],
                                 'range_bins': [5, 3]}, project)
    assert len(errs) >= 6 and L.normalise_odd(None, project)[1]


def test_the_evaluation_set_is_never_trained_on(tmp_path):
    ws = str(tmp_path)
    ev = [os.path.join(ws, 'datasets', 'loop-eval-x', 'drives', f'k{i}') for i in range(2)]
    other = os.path.join(ws, 'runs', 'mine')
    for d in ev + [other]:
        os.makedirs(d)
    PV.write_json(os.path.join(other, 'drive.json'), {'cache_key': 'other'})
    evalset = {'drives': [{'run': os.path.relpath(d, ws), 'key': f'key{i}'} for i, d in enumerate(ev)]}
    L.assert_no_leak([f'falcon:{other}', {'kind': 'kitti_object', 'path': '/data/kitti'}], evalset, ws)
    with pytest.raises(L.LoopError, match='never trained on'):
        L.assert_no_leak([{'kind': 'falcon', 'path': ev[1]}], evalset, ws)
    with pytest.raises(L.LoopError):
        L.assert_no_leak([f'falcon:{os.path.dirname(ev[0])}'], evalset, ws)       # a folder holding them
    copy_dir = os.path.join(ws, 'runs', 'copy')                                    # a copy: the same cache key
    os.makedirs(copy_dir)
    PV.write_json(os.path.join(copy_dir, 'drive.json'), {'cache_key': 'key0'})
    with pytest.raises(L.LoopError):
        L.assert_no_leak([f'falcon:{copy_dir}'], evalset, ws)
    ver = os.path.join(ws, 'datasets', 'spec', 'versions', 'v001.json')
    PV.write_json(ver, {'runs': [{'run': os.path.relpath(other, ws)}, {'run': os.path.relpath(ev[0], ws)}]})
    with pytest.raises(L.LoopError):
        L.assert_no_leak([{'kind': 'version', 'path': ver, 'split': 'val'}], evalset, ws)


# ------------------------------------------------------------------------------------------------ the whole loop
def _odd():
    return {'schema': L.ODD_SCHEMA, 'id': 'tiny', 'name': 'Town 10, clear and fog', 'rig': 'rig-default',
            'worlds': ['world-town10'], 'traffic_profile': 'tp-default',
            'weathers': [{'name': 'clear', 'preset': 'ClearNoon'},
                         {'name': 'fog', 'preset': 'ClearNoon', 'fog_density': 90.0,
                          'ranges': {'sun_altitude_angle': [10.0, 40.0]}}],
            'classes': ['car', 'pedestrian'], 'range_bins': [0, 20, 40],
            'evaluation': {'drives_per_cell': 2, 'drive_s': 1.0, 'fps': 20.0, 'record_every': 1, 'seed': 3,
                           'n_boot': 100},
            'weak_cells': {'min_objects': 5, 'max_cells': 1},
            'search': {'budget': 2, 'batch': 2, 'fps': 5.0, 'method': 'random'},
            'generate': {'objects_per_cell': 20, 'drive_s': 1.0, 'record_every': 1, 'max_drives': 2,
                         'reruns_per_failure': 1, 'max_failures': 1}}


@pytest.fixture(scope='module')
def loopws(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    base = tmp_path_factory.mktemp('loop')
    p = Project.new('Loop')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=96, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.doc['traffic_profiles'][0].update(vehicles=10, walkers=10)
    p.doc['models'] = [SYN]
    p.doc['loop'] = {'model': 'syn', 'odd': _odd(), 'iterations': []}
    p.save(str(base / 'loop.falcon.json'))
    return {'path': p.path, 'base': base}


def _runner(path, settings_path):
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    proj = Project.load(path)
    svc = CarlaService(Settings(str(settings_path)), JobQueue(SubprocessLauncher()), proj.workspace, fake_carla=FAKE)
    return proj, svc, L.Runner(proj, svc)


def test_the_loop_with_the_synthetic_detector_stops_at_retrain(loopws, tmp_path):
    proj, svc, r = _runner(loopws['path'], tmp_path / 'settings.json')
    ws = proj.workspace
    try:
        with pytest.raises(L.Stopped, match='synthetic detector is not a model'):
            r.run('compare')
    finally:
        svc.shutdown()
    (it,) = L.iterations(ws)
    assert it['status'] == 'stopped' and it['steps']['retrain']['state'] == 'stopped'
    for s in L.STEPS[:5]:
        rec = it['steps'][s]
        assert rec['state'] == 'done' and L.step_done(ws, it, s), (s, rec)
        assert 'estimate' in rec and rec['estimate']['label'] in ('measured', 'typical', 'not estimated',
                                                                   'extrapolated from measured'), (s, rec)
        assert rec['measured']['wall_s'] >= 0 and rec['outputs']
    # costs: the drives from the (typical: a fake server measures nothing) throughput; the first evaluation of a
    # model says it is not estimated, and why
    ev_est = it['steps']['evaluate']['estimate']
    assert ev_est['parts'][0]['label'] == 'typical' and ev_est['parts'][0]['seconds'] > 0
    assert ev_est['parts'][1]['label'] == 'not estimated' and 'measured by this step' in ev_est['parts'][1]['source']
    assert it['steps']['search']['estimate']['seconds'] > 0
    assert it['steps']['generate']['estimate']['seconds'] is not None
    # the evaluation set: two drives per cell at fixed seeds, recorded once, every drive hashed
    evs = PV.read_json(os.path.join(ws, it['evalset']))
    assert len(evs['drives']) == 4 and {d['cell'] for d in evs['drives']} == {'world-town10 · clear',
                                                                              'world-town10 · fog'}
    assert [d['seed'] for d in evs['drives']] == [L.EVAL_SEED_BASE + 30000 + i for i in range(4)]
    fogs = [d['weather'] for d in evs['drives'] if d['cell'].endswith('fog')]
    assert all(10 <= w['sun_altitude_angle'] <= 40 and w['fog_density'] == 90 for w in fogs) and fogs[0] != fogs[1]
    assert L.read_evalset(ws, it['odd_key']) is not None
    # weak cells and the search
    wc = PV.read_json(os.path.join(L.iteration_dir(ws, 1), 'weak_cells.json'))
    assert wc['rule'] == L.WEAK_RULE and 'heuristic' in wc['rule'] and wc['chosen'], wc['rows']
    cell = wc['chosen'][0]
    assert cell['ci_high'] < wc['overall_AP']['value'] and cell['n_objects'] >= 5
    sd = PV.read_json(os.path.join(L.iteration_dir(ws, 1), 'search.json'))
    (sc,) = sd['cells']
    assert sc['scenario'] == L.SCENARIOS_FOR_CLASS[cell['class']][0][0] and sc['evaluated'] == 2
    assert 'not the model' in sd['objective'] and sc['rerun_cases']
    study = PV.read_json(os.path.join(ws, sc['study'], 'study.json'))
    assert study['rig_doc']['sensors'][0]['labels'] == [] and study['weather']['preset'] == 'ClearNoon'
    # generated: the targeted drives and the failing case re-run with labels (the same seed and values)
    gen = PV.read_json(os.path.join(L.iteration_dir(ws, 1), 'generate.json'))
    assert gen['drives'] and gen['reruns'] and gen['reruns'][0]['same_seed_as_search']
    assert gen['plan']['yield_source'].startswith('measured on the loop')
    eval_dirs = {os.path.join(ws, d['run']) for d in evs['drives']}
    assert not eval_dirs & {os.path.join(ws, x) for x in gen['runs']}
    rerun = json.load(open(os.path.join(ws, gen['reruns'][0]['run'], 'drive.json')))
    assert 'boxes2d' in rerun['rig']['sensors'][0]['labels'] and rerun['seed'] == sc['seed']
    assert gen['frames'] > 0
    # the record says why it stopped; a new run refuses to repeat it until the model or the ODD changes
    assert 'cannot retrain' in it['stopped']
    proj2, svc2, r2 = _runner(loopws['path'], tmp_path / 'settings.json')
    try:
        with pytest.raises(L.Stopped, match='stopped'):
            r2.run('compare')
    finally:
        svc2.shutdown()
    # the project's record
    for e in L.sync_edits(proj2):
        proj2.apply(e)
    assert proj2.doc['loop']['iterations'][0]['status'] == 'stopped'
    assert any(s['id'] == 'loop-001-targeted' for s in proj2.doc['dataset_specs'])
    proj2.save()


def test_the_loop_retrains_a_falcon_model_and_compares(loopws, tmp_path):
    from falcon_autonomy.core import training as T
    from falcon_autonomy.core.project import Project
    from test_workers import _cfg, _run
    pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    proj = Project.load(loopws['path'])
    ws = proj.workspace
    # a tiny detector trained by Falcon on two drives of its own (not the evaluation set's)
    srcs = []
    os.makedirs(os.path.join(ws, 'runs'), exist_ok=True)
    for name, seed in (('train-a', 11), ('train-b', 12)):
        out = os.path.join(ws, 'runs', name)
        code, _, err = _run(_cfg(proj, out, seconds=1.0, seed=seed))
        assert code == P.OK, err
        srcs.append({'kind': 'falcon', 'path': out})
    tdir = os.path.join(ws, 'training', 'tiny')
    T.train(dict(TINY, train=srcs[:1], val=srcs[1:]), tdir)
    reg = PV.read_json(os.path.join(tdir, 'model.json'))
    reg['id'] = 'tiny'
    proj.doc['models'].append(reg)
    proj.doc['loop']['model'] = 'tiny'
    proj.doc['loop']['odd']['weak_cells']['fallback'] = 'lowest'     # a barely trained model: AP 0 everywhere
    proj.save()
    proj, svc, r = _runner(loopws['path'], tmp_path / 'settings.json')
    try:
        it = r.run('compare')
    finally:
        svc.shutdown()
    assert it['n'] == 2 and it['status'] == 'complete', {s: it['steps'][s].get('error') for s in L.STEPS}
    assert all(L.step_done(ws, it, s) for s in L.STEPS)
    # the evaluation set is the first iteration's (same ODD): no drive recorded again
    first = L.load_iteration(ws, 1)
    assert it['evalset'] == first['evalset']
    # retrained from the model's own settings on its sources + the new drives, never on the evaluation set
    rt = it['steps']['retrain']
    trun = PV.read_json(os.path.join(ws, rt['result']['training'], 'run.json'))
    s = trun['provenance']['settings']
    assert s['arch'] == TINY['arch'] and s['seed'] == 0 and s['init'] == 'scratch' and s['epochs'] == 1
    gen = PV.read_json(os.path.join(L.iteration_dir(ws, 2), 'generate.json'))
    paths = [x['path'] for x in s['train']]
    assert paths[0] == srcs[0]['path'] and paths[1:] == [os.path.join(ws, x) for x in gen['runs']]
    assert [x['path'] for x in s['val']] == [srcs[1]['path']]
    evs = PV.read_json(os.path.join(ws, it['evalset']))
    eval_dirs = {os.path.realpath(os.path.join(ws, d['run'])) for d in evs['drives']}
    assert not eval_dirs & {os.path.realpath(p) for p in paths + [x['path'] for x in s['val']]}
    assert rt['estimate']['label'] in ('measured', 'not estimated') and rt['measured']['images_per_s'] > 0
    assert rt['result']['registration']['runtime'] == 'onnx'
    # before and after on the same set, with intervals and the paired difference
    cmp = PV.read_json(os.path.join(L.iteration_dir(ws, 2), 'compare', 'comparison.json'))
    assert cmp['schema'] == 'falcon.comparison/1' and cmp['unit'] == 'drive' and cmp['n_units'] == 4
    rows = it['summary']['rows']
    assert rows[0]['label'] == 'AP, all' and rows[0]['before']['value'] is not None
    assert rows[0]['diff']['value'] == pytest.approx(rows[0]['after']['value'] - rows[0]['before']['value'])
    assert all(r['verdict'] in ('improvement', 'regression', 'no detectable change at 95 %') for r in rows)
    wc = PV.read_json(os.path.join(L.iteration_dir(ws, 2), 'weak_cells.json'))
    assert len(rows) == 2 + len(wc['chosen']) and wc['chosen']
    # resume: nothing runs again; a step whose output changed runs again, and what follows it — from the tools'
    # caches: no drive is recorded again (the study's results and the targeted drives are reused)
    runs = {x: it['steps'][x]['runs'] for x in L.STEPS}
    n_runs = sum(len(fs) for _, _, fs in os.walk(os.path.join(ws, 'studies')))
    proj, svc, r = _runner(loopws['path'], tmp_path / 'settings.json')
    try:
        assert r.run('compare', it=it)['steps'] == it['steps'] and len(svc.queue.jobs) == 0
        wcp = os.path.join(L.iteration_dir(ws, 2), 'weak_cells.json')
        PV.write_json(wcp, dict(PV.read_json(wcp), tampered=True))
        assert L.next_step(ws, L.load_iteration(ws, 2)) == 'weak_cells'
        it2 = r.run('compare', it=L.load_iteration(ws, 2))
        assert it2['status'] == 'complete' and [j.spec.kind for j in svc.queue.jobs] == ['train']
        assert sum(len(fs) for _, _, fs in os.walk(os.path.join(ws, 'studies'))) == n_runs
        assert [it2['steps'][x]['runs'] - runs[x] for x in L.STEPS] == [0, 0, 1, 1, 1, 1, 1, 1]
        assert it2['steps']['retrain']['result']['training'] == rt['result']['training']
        # the estimates knew what the tools had cached: the search's runs and the targeted drives
        assert it2['steps']['search']['estimate']['seconds'] == 0
        gen_est = it2['steps']['generate']['estimate']['parts'][0]
        assert gen_est['seconds'] == 0 and 'already recorded' in gen_est['source']
        # the next iteration starts from the retrained model and reuses its evaluation
        it3 = r.run('evaluate')
        assert it3['n'] == 3 and it3['model']['id'] == rt['result']['registration']['id']
        assert it3['steps']['evaluate']['reused_from']['iteration'] == 2 and len(svc.queue.jobs) == 1
    finally:
        svc.shutdown()
    # the project's loop.iterations and the retrained model, through the edits the GUI and the CLI apply
    edits = L.sync_edits(proj)
    for e in edits:
        proj.apply(e)
    ents = proj.doc['loop']['iterations']
    assert [e['n'] for e in ents] == [1, 2, 3] and ents[1]['status'] == 'complete' and ents[1]['AP_diff']
    assert any(m['id'] == rt['result']['registration']['id'] for m in proj.doc['models'])
    proj.save()                                      # schema-valid
    assert L.sync_edits(Project.load(loopws['path'])) == []


def test_the_loop_command(loopws, tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)

    def cli(*args):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'loop', loopws['path'], *args],
                              capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    r = cli('status')
    assert r.returncode == 0 and 'iteration 3' in r.stdout and 'next: step' in r.stdout, r.stdout + r.stderr
    r = cli('show', '2')
    assert r.returncode == 0 and '8. Before and after' in r.stdout and 'estimated' in r.stdout
    r = cli('show', '2', '--json')
    assert json.loads(r.stdout)['schema'] == L.ITERATION_SCHEMA
    r = cli('next', '--fake-carla', FAKE)             # iteration 3's weak cells (its evaluation was reused)
    msgs = [P.parse_line(x) for x in r.stdout.splitlines()]
    res = [m for m in msgs if m['t'] == 'result']
    assert r.returncode == 0 and res and res[-1]['status'] == 'ok' and res[-1]['next'] == 'search', r.stdout
    assert any(m['t'] == 'log' and 'estimated' in m['text'] for m in msgs)
    r = cli('run', '--until', '9')
    assert r.returncode == P.FAILED
    odd = json.load(open(loopws['path']))['loop']['odd']
    r = cli('set', '--odd', 'example')
    assert r.returncode == 0 and 'Bengaluru' in r.stdout and 'ODD:' in r.stdout
    odd['search']['budget'] = 3
    (tmp_path / 'odd.json').write_text(json.dumps(odd))
    assert cli('set', '--odd', str(tmp_path / 'odd.json')).returncode == 0
    r = cli('run', '--fake-carla', FAKE)               # the ODD changed under an unfinished iteration: refused
    assert r.returncode == P.REFUSED and 'new iteration' in r.stdout


def test_the_iteration_report(loopws, tmp_path):
    from falcon_autonomy.core import reports as R
    from falcon_autonomy.core.project import Project
    from test_reports import self_contained
    ws = Project.load(loopws['path']).workspace
    out = str(tmp_path / 'it.html')
    meta = R.write_report('iteration', L.iteration_dir(ws, 2), out)
    doc = open(out, encoding='utf-8').read()
    self_contained(doc)
    it = L.load_iteration(ws, 2)
    for r in it['summary']['rows']:
        for k in ('before', 'after'):
            if r[k]['value'] is not None:
                assert f'{r[k]["value"]:.3f}' in doc
        assert f'{r["diff"]["value"]:+.3f}' in doc and r['verdict'] in doc
    assert 'class="b"' in doc and 'class="a"' in doc and '(circle)' in doc and '(square)' in doc
    for s in L.STEPS:
        assert L.STEP_TITLES[s] in doc
        assert (it['steps'][s].get('estimate') or {}).get('label', '—') in doc
    assert 'not the model' in doc and 'loop-002-targeted' in doc and it['model_sha256'] in doc
    assert 'never trained on' in doc and os.path.join(L.iteration_dir(ws, 2), 'iteration.json') in meta['inputs']
    stopped = R.write_report('iteration', os.path.join(L.iteration_dir(ws, 1), 'iteration.json'),
                             str(tmp_path / 'one.html'))
    assert 'cannot retrain' in open(tmp_path / 'one.html', encoding='utf-8').read() and stopped['kind'] == 'iteration'
