"""The model test harness (core.harness) on drives the worker records against the fake carla: an oracle detector (the
ground truth itself) must score AP 1 with a CI at 1; the synthetic detector (planted weaknesses) scores lower and is
flagged as a regression against the oracle by the paired bootstrap; the heat map and the gallery on hand-made frames
with known answers; a segmenter's IoU against the confusion matrix worked out from the labels; the command end to
end."""
import json
import os

import numpy as np
import pytest

from falcon_autonomy.core import harness as H
from falcon_autonomy.core import protocol as P

cv2 = pytest.importorskip('cv2')

SYN = {'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
       'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector', 'licence': 'GPL-3.0-or-later'}
ORACLE = {'id': 'oracle', 'name': 'Oracle', 'kind': 'detector', 'runtime': 'python', 'entry': 'test_harness:Oracle',
          'licence': 'GPL-3.0-or-later'}
ROAD = {'id': 'road', 'name': 'All road', 'kind': 'segmenter', 'runtime': 'python', 'entry': 'test_harness:AllRoad',
        'licence': 'GPL-3.0-or-later', 'falcon_classes': ['road', 'car', 'sidewalk']}


class Oracle:
    """Returns the frame's ground-truth boxes with score 1."""
    synthetic = True

    def predict(self, rgb, meta):
        return [{'bbox': o['bbox'], 'class': o['class'], 'score': 1.0} for o in meta['objects'] if o.get('bbox')]


class AllRoad:
    def predict(self, rgb, meta):
        return np.ones(rgb.shape[:2], np.uint8)            # Falcon class 1, road, everywhere


@pytest.fixture(scope='module')
def drives(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('drives')
    p = Project.new('Test')
    cam = p.doc['rigs'][0]['sensors'][0]
    cam.update(width=480, height=270, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.doc['traffic_profiles'][0]['vehicles'] = 10
    p.doc['models'] = [SYN, ORACLE, ROAD]
    p.save(str(base / 'test.falcon.json'))
    out = []
    for name, seed, fog in (('a', 3, 0.0), ('b', 4, 0.0), ('c', 5, 40.0)):
        code, _, err = _run(_cfg(p, base / name, seconds=1.0, seed=seed, weather={**_weather(), 'fog_density': fog}))
        assert code == P.OK, err
        out.append(str(base / name))
    return {'project': str(base / 'test.falcon.json'), 'runs': out, 'base': base}


def _weather():
    from falcon_autonomy.core.weather import resolve
    return resolve({'preset': 'ClearNoon'})


def test_oracle_scores_one_and_synthetic_is_a_regression(drives, tmp_path):
    ev_o = H.evaluate(drives['runs'], ORACLE, str(tmp_path / 'o'), n_boot=200)
    ev_s = H.evaluate(drives['runs'], SYN, str(tmp_path / 's'), n_boot=200, grid=('fog_density', 'cloudiness'))
    t_o = {r['metric']: r for r in ev_o['scores']['table']}
    assert ev_o['n_frames'] == 60 and ev_o['scores']['unit'] == 'drive' and ev_o['scores']['n_units'] == 3
    assert t_o['AP']['value'] == pytest.approx(1.0) and t_o['AP']['ci_low'] == pytest.approx(1.0)
    assert ev_o['gallery'] == [] and ev_o['providers'] == ['python']
    n_gt = sum(1 for f in H.read_predictions(str(tmp_path / 'o' / 'predictions.jsonl')) for g in f['gt']
               if not g['ignore'])
    assert n_gt > 50
    t_s = {r['metric']: r for r in ev_s['scores']['table']}
    assert t_s['AP']['value'] < 0.9 and t_s['AP']['ci_low'] <= t_s['AP']['value'] <= t_s['AP']['ci_high']
    assert ev_s['synthetic_model'] and ev_s['latency_ms']['n'] == 60
    assert any(k.startswith('AP.range_m[') for k in t_s)
    g = ev_s['gallery']
    assert g and g[0]['n_errors'] >= g[-1]['n_errors'] and g[0]['n_errors'] == len(g[0]['missed']) + len(
        g[0]['false_positives'])
    grid = ev_s['grid']
    assert grid['x'] == 'fog_density' and len(grid['x_edges']) == 3          # two distinct fog values: two columns
    assert sum(c['n_frames'] for row in grid['cells'] for c in row) == 60
    saved = json.loads((tmp_path / 's' / 'evaluation.json').read_text())
    assert saved['schema'] == H.SCHEMA and 'run0/run.json' in saved['provenance']['inputs']
    cmp = H.compare(H.read_predictions(str(tmp_path / 'o' / 'predictions.jsonl')),
                    H.read_predictions(str(tmp_path / 's' / 'predictions.jsonl')), n_boot=200)
    assert 'AP' in cmp['regressions'] and cmp['n_frames'] == 60 and cmp['unit'] == 'drive'
    same = H.compare(H.read_predictions(str(tmp_path / 's' / 'predictions.jsonl')),
                     H.read_predictions(str(tmp_path / 's' / 'predictions.jsonl')), n_boot=100)
    assert same['regressions'] == [] and same['improvements'] == []
    assert next(r for r in same['rows'] if r['metric'] == 'AP')['diff'] == pytest.approx(0.0)


def _frame(i, fog, sun, hit):
    gt = [{'bbox': [10 + 30 * k, 10, 20, 20], 'class': 'car', 'ignore': 0} for k in range(4)]
    det = [dict(g, score=0.9) for g in gt] if hit else []
    return {'image_id': f'd/{i}', 'drive': 'd', 'step': i, 'image': '', 'gt': gt, 'det': det,
            'conditions': {'fog_density': fog, 'sun_altitude_angle': sun}}


def test_condition_grid_and_gallery_on_known_frames():
    frames = [_frame(i, fog, sun, hit=not (fog > 50 and sun < 0))
              for i, (fog, sun) in enumerate([(f, s) for f in (0, 80) for s in (-10, 45) for _ in range(5)])]
    g = H.condition_grid(frames, 'fog_density', 'sun_altitude_angle')
    cells = {(c['x'][0] < 40, c['y'][0] < 15): c for row in g['cells'] for c in row}
    assert cells[(False, True)]['AP'] == pytest.approx(0.0)                 # fog 80, night: all missed
    assert cells[(True, True)]['AP'] == pytest.approx(1.0) and cells[(False, False)]['AP'] == pytest.approx(1.0)
    assert all(c['n_frames'] == 5 and c['n_gt'] == 20 for c in cells.values())
    sparse = H.condition_grid(frames[:2], 'fog_density', 'sun_altitude_angle')
    assert sparse['cells'][0][0]['AP'] is None and 'fewer than' in sparse['cells'][0][0]['note']
    gal = H.gallery(frames, k=3)
    assert len(gal) == 3 and all(r['n_errors'] == 4 and len(r['missed']) == 4 for r in gal)
    fp = _frame(99, 0, 0, True)
    fp['det'].append({'bbox': [300, 300, 20, 20], 'class': 'car', 'score': 0.95})
    fp['det'].append({'bbox': [400, 300, 20, 20], 'class': 'car', 'score': 0.2})              # below min_score
    (row,) = H.gallery([fp])
    assert row['n_errors'] == 1 and row['false_positives'][0]['bbox'][0] == 300
    with pytest.raises(H.HarnessError):
        H.condition_grid(frames, 'fog_density', 'no_such_condition')


def test_segmenter_iou_against_the_labels(drives, tmp_path):
    from falcon_autonomy.core import groundtruth as GT
    ev = H.evaluate(drives['runs'][:1], ROAD, str(tmp_path / 'seg'), n_boot=50)
    t = {r['metric']: r for r in ev['scores']['table']}
    # worked out from the semantic images: road IoU = road pixels / scored pixels (the model says road everywhere)
    frames = H.read_predictions(str(tmp_path / 'seg' / 'predictions.jsonl'))
    road = scored = 0
    for f in frames:
        sem = cv2.imread(f['image'].replace('/rgb/', '/semantic/').replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)
        keep = np.isin(sem, [GT.CLASS_ID[c] for c in ROAD['falcon_classes']])
        road += int(((sem == 1) & keep).sum())
        scored += int(keep.sum())
    assert scored > 0 and t['IoU.road']['value'] == pytest.approx(road / scored)
    assert t['IoU.car']['value'] == pytest.approx(0.0)
    assert ev['scores']['unit'] == 'image'


def test_evaluate_command(drives, tmp_path):
    from falcon_autonomy.cli import main
    a, b = tmp_path / 'a', tmp_path / 'b'
    args = ['--run', drives['runs'][0], '--run', drives['runs'][1], '--boot', '50', '--every', '2']
    assert main(['evaluate', drives['project'], '--model', 'oracle', '--out', str(a)] + args) == P.OK
    assert main(['evaluate', drives['project'], '--model', 'syn', '--out', str(b)] + args) == P.OK
    assert json.loads((a / 'evaluation.json').read_text())['n_frames'] == 20
    assert main(['evaluate', '--compare', str(a), str(b), '--out', str(tmp_path / 'cmp')]) == P.OK
    cmp = json.loads((tmp_path / 'cmp' / 'comparison.json').read_text())
    assert cmp['schema'] == 'falcon.comparison/1' and cmp['b']['model']['id'] == 'syn'
    assert main(['evaluate', drives['project'], '--model', 'nope', '--out', str(tmp_path / 'x')] + args) == P.FAILED
    assert main(['evaluate', drives['project'], '--model', 'syn', '--out', str(tmp_path / 'y')]) == P.FAILED
    assert main(['evaluate', '--compare', str(a), str(tmp_path / 'seg-missing'), '--out', str(tmp_path / 'z')]) \
        == P.FAILED
    assert not os.path.exists(tmp_path / 'z' / 'comparison.json')
