"""Model-in-the-loop rare-case search (core.milsearch, sim.studio): the perception failure on hand-made frames with
known answers (matching, the critical window, the lost lead, the miss rate, the combined objective); weather
dimensions; the probability-of-failure acquisition on the planted objective; a perception search against the fake
carla with the synthetic detector — scored after the drives, resumed, re-scored for another objective without driving
again; the GPU lock held while a GPU model runs; the loop's step-4 option; the command."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKE
from falcon_autonomy.core import milsearch as MS
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import search as S

SYN = {'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
       'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector', 'licence': 'GPL-3.0-or-later',
       'params': {'h50': 6.0, 'h_scale': 1.5}}


# ------------------------------------------------------------------------------------------------ hand-made frames
def _box(x, w=20.0, h=40.0):
    return [x, 10.0, w, h]


def _frames(n=60, visible_from=10, detected_from=30, extra=None):
    """fps 10: steps 0..n-1 at t = (step + 1) / 10. The pedestrian (track 101) present from visible_from, detected
    (a box on it, score 0.9) from detected_from; a parked car (track 'level:1') at 30 m, never detected."""
    out = []
    for s in range(n):
        gt, det = [], []
        if s >= visible_from:
            gt.append({'bbox': _box(100), 'class': 'pedestrian', 'ignore': 0, 'range_m': 20.0, 'track_id': 101})
            if s >= detected_from:
                det.append({'bbox': _box(101), 'class': 'pedestrian', 'score': 0.9})
        gt.append({'bbox': _box(300, 40, 20), 'class': 'car', 'ignore': 0, 'range_m': 30.0, 'track_id': 'level:1'})
        for e in (extra or {}).get(s, ()):
            det.append(e)
        out.append({'image_id': f'd/{s}', 'drive': 'd', 'step': s, 'gt': gt, 'det': det, 'conditions': {}})
    return out


SCN = {'id': 'pedestrian_crossing', 'actor_ids': {'ped': 101},
       'criticality': {'criticality': 0.62, 'actors': {'ped': {'collision': False, 'collision_t': None,
                                                              'min_ttc_s': 1.2, 'min_ttc_t': 5.0,
                                                              'min_distance_t': 5.6}}}}


def test_perception_failure_on_hand_made_frames():
    sc = MS.perception_score(_frames(), SCN, 10.0, {'window_s': 3.0, 'lead_s': 3.0})
    (a,) = sc['actors']
    # t_c = 5.0 (minimum TTC); visible from t = 1.1, first detected at t = 3.1
    assert a['t_critical'] == 5.0 and a['why_critical'] == 'minimum time to collision'
    assert a['first_visible_t'] == pytest.approx(1.1) and a['first_detected_t'] == pytest.approx(3.1)
    assert a['detection_delay_s'] == pytest.approx(2.0)
    # window [2.0, 5.0]: steps 19..49 (31 frames), detected in 20 of them
    assert a['window_frames'] == 31 and a['window_recall'] == pytest.approx(20 / 31, abs=1e-6)
    # lead 5.0 − 3.1 + 0.1 = 2.0 of the needed min(3, 5.0 − 1.1 + 0.1) = 3: a third lost
    assert a['needed_lead_s'] == pytest.approx(3.0) and a['lead_s'] == pytest.approx(2.0)
    assert a['lost_lead'] == pytest.approx(1 / 3, abs=1e-6)
    assert a['failure'] == pytest.approx(11 / 31, abs=1e-6) == sc['perception'] == sc['value']
    assert sc['worst_actor'] == 'ped' and sc['metric'] == 'perception'
    # the miss rate: the pedestrian's 50 present frames (30 detected); the car is not a target class
    assert sc['miss_rate'] == pytest.approx(1 - 30 / 50) and sc['targets']['frames'] == 0
    # with the car's class as a target within range it counts, never found
    sc2 = MS.perception_score(_frames(), SCN, 10.0, {'target_classes': ['car'], 'range_m': [0, 50],
                                                     'metric': 'miss_rate'})
    assert sc2['targets'] == {'classes': ['car'], 'range_m': [0.0, 50.0], 'objects': 1, 'frames': 60, 'hits': 0}
    assert sc2['value'] == sc2['miss_rate'] == pytest.approx(1 - 30 / 110, abs=1e-6)
    assert MS.perception_score(_frames(), SCN, 10.0, {'target_classes': ['car'], 'range_m': [0, 25]}
                               )['targets']['frames'] == 0        # out of the range bin
    # never detected: the whole lead lost; detected from its first frame: none lost
    never = MS.perception_score(_frames(detected_from=99), SCN, 10.0)
    assert never['lost_lead'] == 1.0 == never['perception'] and never['actors'][0]['first_detected_t'] is None
    at_once = MS.perception_score(_frames(detected_from=10), SCN, 10.0)
    assert at_once['lost_lead'] == 0.0 and at_once['window_miss'] == 0.0 and at_once['perception'] == 0.0
    # a detection of the wrong class, or scoring under min_score, is a miss
    wrong = {s: [{'bbox': _box(100), 'class': 'car', 'score': 0.9}] for s in range(60)}
    low = {s: [{'bbox': _box(100), 'class': 'pedestrian', 'score': 0.3}] for s in range(60)}
    assert MS.perception_score(_frames(detected_from=99, extra=wrong), SCN, 10.0)['perception'] == 1.0
    assert MS.perception_score(_frames(detected_from=99, extra=low), SCN, 10.0)['perception'] == 1.0
    assert MS.perception_score(_frames(detected_from=99, extra=low), SCN, 10.0,
                               {'min_score': 0.25})['perception'] == 0.0
    # nothing present: 0, said so; the combined objective is the smaller of the two
    empty = MS.perception_score([], SCN, 10.0)
    assert empty['value'] == 0.0 and 'nothing the model' in empty['note']
    spec = {'objective': 'both'}
    assert MS.objective_value(spec, 0.62, sc) == pytest.approx(11 / 31, abs=1e-6)
    assert MS.objective_value(spec, 0.2, sc) == 0.2
    assert MS.objective_value({'objective': 'criticality'}, 0.62, None) == 0.62


def test_matching_takes_present_objects_first_and_one_detection_each():
    f = {'gt': [{'bbox': [0, 0, 10, 10], 'class': 'car', 'ignore': 1},
                {'bbox': [1, 0, 10, 10], 'class': 'car', 'ignore': 0},
                {'bbox': [50, 0, 10, 10], 'class': 'car', 'ignore': 0}],
         'det': [{'bbox': [0.5, 0, 10, 10], 'class': 'car', 'score': 0.9}]}
    assert MS.frame_hits(f) == [False, True, False]
    f['det'].append({'bbox': [0.5, 0, 10, 10], 'class': 'car', 'score': 0.8})
    assert MS.frame_hits(f) == [True, True, False]                  # the second takes the ignored one


def test_spec_weather_dimensions_and_evaluability(tmp_path):
    with pytest.raises(MS.MilError, match='needs a registered model'):
        MS.normalise_spec({'objective': 'perception'})
    with pytest.raises(MS.MilError, match='metric'):
        MS.normalise_spec({'objective': 'perception', 'model': SYN, 'metric': 'AP'})
    s = MS.normalise_spec({'objective': 'both', 'model': SYN, 'target_classes': ['car', 'car', 'bus']})
    assert s['target_classes'] == ['bus', 'car'] and s['range_m'] == [0.0, 50.0]
    assert MS.spec_key({'objective': 'criticality'}) == 'criticality-rss'          # avoidable failures (core.rss)
    assert MS.spec_key({'objective': 'criticality', 'count': 'all'}) == 'criticality'
    assert MS.spec_key(dict(s, count='all')) == MS.spec_key(s)      # RSS counting leaves perception keys alone
    assert MS.spec_key(s) != MS.spec_key(dict(s, window_s=2.0)) != MS.spec_key(dict(s, model=dict(SYN, params={})))
    sp = MS.weather_space(['fog_density', 'sun_altitude_angle:-5:10'])
    assert sp == [{'name': 'weather.fog_density', 'lo': 0.0, 'hi': 100.0},
                  {'name': 'weather.sun_altitude_angle', 'lo': -5.0, 'hi': 10.0}]
    assert MS.weather_space({'precipitation': [50, 90]})[0] == {'name': 'weather.precipitation', 'lo': 50.0,
                                                                'hi': 90.0}
    with pytest.raises(MS.MilError):
        MS.weather_space(['fog'])
    scn, wx = MS.split_values({'gap': 10.0, 'weather.fog_density': 40.0})
    assert scn == {'gap': 10.0} and wx == {'fog_density': 40.0}
    cfg = MS.apply_weather({'weather_source': {'preset': 'ClearNoon'}}, wx)
    assert cfg['weather']['fog_density'] == 40.0 and cfg['weather']['cloudiness'] == 5.0
    assert cfg['weather_source'] == {'preset': 'ClearNoon', 'fog_density': 40.0}
    assert MS.evaluable(SYN) is None
    assert 'segmenter' in MS.evaluable(dict(SYN, kind='segmenter'))
    assert 'missing' in MS.evaluable({'id': 'x', 'kind': 'detector', 'runtime': 'onnx', 'path': 'nope.onnx',
                                      'output': {'format': 'yolo'}}, str(tmp_path))
    assert 'cannot be imported' in MS.evaluable(dict(SYN, entry='no_such_module:X'))
    assert MS.check_rig({'name': 'r', 'sensors': [{'id': 'c', 'type': 'camera.pinhole', 'labels': []}]})
    assert MS.check_rig({'name': 'r', 'sensors': [{'id': 'c', 'type': 'camera.pinhole',
                                                   'labels': ['boxes2d']}]}) is None


def test_probability_of_failure_maps_the_planted_region():
    """The threshold-aware acquisition against expected improvement on the planted objective (seeds 0–2, 120
    evaluations): as fast to the first failures, many more failures found (it maps the region)."""
    f = S.planted_objective()
    r = S.compare(f, f.space, budget=120, seeds=range(3), threshold=0.5, methods=['bayesopt', 'bayes_pof'])['methods']
    ei, pof = r['bayesopt'], r['bayes_pof']
    print(f'\nEI: 5th failure {ei["to_n_failures_mean"]:.1f}, {ei["failures_at_budget"]:.1f} found; '
          f'PoF: {pof["to_n_failures_mean"]:.1f}, {pof["failures_at_budget"]:.1f} found')
    assert all(v is not None for v in pof['to_n_failures'])
    assert pof['to_n_failures_mean'] < 1.5 * ei['to_n_failures_mean']
    assert pof['failures_at_budget'] > 1.5 * ei['failures_at_budget']
    s = S.make('bayes_pof', f.space, 4, threshold=0.7)
    assert s.threshold == 0.7 and S.from_state(s.state()).threshold == 0.7


# ------------------------------------------------------------------------------------------------ against the fake
@pytest.fixture(scope='module')
def mil(tmp_path_factory):
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    base = tmp_path_factory.mktemp('mil')
    p = Project.new('MIL')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=90, labels=['boxes2d'])
    p.doc['models'] = [SYN]
    p.save(str(base / 'mil.falcon.json'))
    svc = CarlaService(Settings(str(base / 'settings.json')), JobQueue(SubprocessLauncher()), p.workspace,
                       fake_carla=FAKE)
    yield {'project': p, 'svc': svc, 'base': base}
    svc.shutdown()


def _study(mil, objective, name='mil', **kw):
    from falcon_autonomy.sim.studio import Study
    return Study(mil['svc'], mil['project'], SC.builtin('lead_brake'), name, 'world-town10', 'rig-default', seed=11,
                 objective=objective, extra_space=MS.weather_space(['fog_density']), fps=10, **kw)


def test_a_perception_search_scores_resumes_and_rescores(mil):
    st = _study(mil, {'objective': 'perception', 'model': SYN})
    out = st.search('random', 2, 2, 0.8)
    rows = st.results()
    assert len(rows) == 2 and len(out['history']) == 2 and out['objective']['kind'] == 'perception'
    for r in rows:
        assert r['objective_key'] == st.okey and r['objective_kind'] == 'perception'
        assert r['objective'] == r['perception']['value'] and 0.0 <= r['objective'] <= 1.0
        assert r['perception']['actors'][0]['actor'] == 'lead' and r['perception']['synthetic_model']
        assert os.path.isfile(os.path.join(st.dir, 'perception', f'{os.path.basename(r["run"])}-{st.okey}.json'))
        cfg = json.load(open(os.path.join(r['run'], 'drive.json')))
        assert cfg['weather']['fog_density'] == pytest.approx(r['values']['weather.fog_density'])
        assert 'weather.fog_density' not in cfg['scenario']['values'] and cfg['fps'] == 10.0
        run = json.load(open(os.path.join(r['run'], 'run.json')))
        assert set(run['scenario']['actor_ids']) == {'lead'}
        assert run['scenario']['blueprints']['lead'].startswith('vehicle.')
    doc = json.load(open(os.path.join(st.dir, 'study.json')))
    assert doc['objective']['kind'] == 'perception' and doc['objective']['model']['id'] == 'syn'
    assert doc['objective']['model_sha256'].startswith('registration:')
    assert [p['name'] for p in doc['space']][-1] == 'weather.fog_density'
    # resumed with a larger budget: one more run, the two finished ones read back
    runs = lambda: sorted(os.listdir(os.path.join(st.dir, 'runs')))     # noqa: E731
    assert len(runs()) == 2
    out = _study(mil, {'objective': 'perception', 'model': SYN}).search('random', 3, 2, 0.8)
    assert len(out['history']) == 3 and len(runs()) == 3
    # another objective on the same study: re-scored from the runs' frames, nothing driven again
    both = _study(mil, {'objective': 'both', 'model': SYN})
    rows = both.current_results()
    assert len(rows) == 3 and len(runs()) == 3 and both.okey != st.okey
    for r in rows:
        assert r['objective_kind'] == 'both' and r['objective_key'] == both.okey
        assert r['objective'] == pytest.approx(min(r['perception']['value'], r['criticality']))
    # back to the criticality: the drives are the same (same rig, same weather), so no new run either
    crit = _study(mil, None)
    rows = crit.current_results()
    from falcon_autonomy.core.rss import rss_objective           # by default only avoidable failures count
    assert len(rows) == 3 and len(runs()) == 3
    assert all(r['rss'] and r['objective'] == rss_objective(r['criticality'], r['rss']) for r in rows)
    rows = _study(mil, {'objective': 'criticality', 'count': 'all'}).current_results()
    assert len(rows) == 3 and all(r['objective'] == r['criticality'] for r in rows) and len(runs()) == 3


def test_a_study_refuses_what_it_cannot_score(mil):
    p = mil['project']
    with pytest.raises(ValueError, match='segmenter'):
        _study(mil, {'objective': 'perception', 'model': dict(SYN, kind='segmenter')}, name='bad')
    p.doc['rigs'][0]['sensors'][0]['labels'] = []
    try:
        with pytest.raises(ValueError, match='boxes2d'):
            _study(mil, {'objective': 'perception', 'model': SYN}, name='bad')
    finally:
        p.doc['rigs'][0]['sensors'][0]['labels'] = ['boxes2d']


def test_a_gpu_model_runs_under_the_gpu_lock(mil, monkeypatch, tmp_path):
    """An ONNX model: score_runs takes the CARLA and GPU locks while it runs and releases them after."""
    from falcon_autonomy.core import models
    from falcon_autonomy.core.jobs import FileLock
    st = _study(mil, {'objective': 'perception', 'model': SYN})
    run = st.results()[0]['run']
    held = []

    class Fake:
        providers = ['CUDAExecutionProvider']

        def __init__(self, m, ws=''):
            pass

        def predict(self, rgb, meta):
            held.append((FileLock('compute-gpu0.lock').acquire(), FileLock('carla-gpu0.lock').acquire()))
            return [{'bbox': o['bbox'], 'class': o['class'], 'score': 1.0} for o in meta['objects'] if o.get('bbox')]
    monkeypatch.setattr(models, 'Runner', Fake)
    monkeypatch.setattr(MS, 'evaluable', lambda m, ws='': None)
    onnx = {'id': 'gpu', 'name': 'GPU', 'kind': 'detector', 'runtime': 'onnx', 'path': 'x.onnx',
            'output': {'format': 'yolo'}, 'falcon_classes': ['car']}
    sc = MS.score_runs([run], {'objective': 'perception', 'model': onnx})[run]
    assert held and all(h == (False, False) for h in held)          # both locks held by the scoring pass
    lk = FileLock('compute-gpu0.lock')
    assert lk.acquire()                                               # and released after it
    lk.release()
    assert sc['perception'] == 0.0 and sc['actors'][0]['window_recall'] == 1.0     # it sees every box


def test_the_loops_search_can_use_the_model(mil):
    from falcon_autonomy.core import loop as L
    odd = {'search': {'objective': 'perception'}}
    assert L.search_objective(odd, SYN, '') == ('perception', None)
    kind, why = L.search_objective(odd, dict(SYN, kind='segmenter'), '')
    assert kind == 'criticality' and 'segmenter' in why
    assert L.search_objective({'search': {}}, SYN, '') == ('criticality', None)       # the default
    cell = {'class': 'car', 'bin': [20.0, 40.0]}
    cond = {'weather': {'name': 'dusk', 'preset': 'ClearSunset', 'ranges': {'sun_altitude_angle': [-2, 6]}}}
    spec, extra = L.cell_objective('perception', cell, cond, odd, SYN)
    assert spec['target_classes'] == ['car'] and spec['range_m'] == [20.0, 40.0]
    assert extra == [{'name': 'weather.sun_altitude_angle', 'lo': -2.0, 'hi': 6.0}]
    assert L.cell_objective('perception', {'class': 'car', 'bin': [80.0, None]}, cond, odd, SYN)[0][
        'range_m'] == [80.0, L.SPEC_FAR_M]
    assert L.cell_objective('criticality', cell, cond, odd, SYN) == (None, [])
    # the cell's study: the labelled rig, the searched weather field on top of the cell's weather
    Study = L._study_class()
    p = mil['project']
    st = Study(mil['svc'], p, SC.builtin('lead_brake'), 'loop-cell', 'world-town10', 'rig-default', 11,
               dict(cond['weather'], sun_altitude_angle=2.0), p.doc['rigs'][0], 10.0, 0, spec, extra)
    vals = dict(SC.sample_params(SC.builtin('lead_brake'), 3), **{'weather.sun_altitude_angle': 5.5})
    cfg = st.config(vals, 0)
    assert cfg['weather']['sun_altitude_angle'] == 5.5 and cfg['fps'] == 10.0
    assert 'boxes2d' in cfg['rig']['sensors'][0]['labels']
    doc = json.load(open(os.path.join(st.dir, 'study.json')))
    assert doc['made_by'] == 'loop' and doc['objective']['kind'] == 'perception'


def test_the_command(mil):
    p = mil['project']
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *a], capture_output=True, text=True,
                              env=env, timeout=300)
    r = cli('search', p.path, '--scenario', 'lead_brake', '--name', 'c', '--objective', 'perception',
            '--fake-carla', FAKE)
    assert r.returncode == P.REFUSED and 'needs --model' in r.stdout
    r = cli('sweep', p.path, '--scenario', 'lead_brake', '--name', 'c', '--samples', '1', '--seed', '4',
            '--objective', 'both', '--model', 'syn', '--weather', 'fog_density:40:60', '--fps', '10',
            '--fake-carla', FAKE)
    assert r.returncode == 0, r.stdout + r.stderr
    (row,) = [json.loads(x) for x in open(os.path.join(p.workspace, 'studies', 'c', 'results.jsonl'))]
    assert 40 <= row['values']['weather.fog_density'] <= 60 and row['objective_kind'] == 'both'
    assert row['objective'] == pytest.approx(min(row['perception']['value'], row['criticality']))
    assert np.isfinite(row['objective'])
