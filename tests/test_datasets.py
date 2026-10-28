"""The data engine's core: plans, cache keys, coverage, splits without leakage, versions."""
import json
import os

import pytest

from falcon_autonomy.core import datasets as D
from falcon_autonomy.core.budget import Throughput


def test_plan_from_the_prior_then_from_measured_yield():
    spec = D.new_spec('ds', 'test', 'rig-default', ['w1', 'w2'])
    spec.update(weathers=['ClearNoon', 'HardRainNoon'], targets={'car': 1000, 'pedestrian': 300, 'train': 5},
                drive_s=30.0, record_every=5)
    p = D.plan(spec, throughput=Throughput.from_settings({}), mpx=1.8)
    assert p['per_drive_frames'] == 120 and 'heuristic' in p['yield_source']
    assert p['unreachable'] == ['train'] and p['estimate']['label'] == 'typical'
    assert len(p['drives']) == 3                                 # car: 1000 / (4 × 120) → 3 drives
    assert [d['cell']['weather']['preset'] for d in p['drives'][:4]] == ['ClearNoon', 'HardRainNoon', 'ClearNoon']
    p2 = D.plan(spec, have={'car': 900, 'pedestrian': 300}, yield_={'car': 0.5, 'pedestrian': 0.1})
    assert len(p2['drives']) == 2 and p2['needed']['pedestrian'] == 0 and 'measured' in p2['yield_source']


def test_drive_key_ignores_where_it_is_written():
    a = {'seed': 1, 'town': 'T', 'out': '/a', 'port': 1}
    b = {'seed': 1, 'town': 'T', 'out': '/b', 'port': 2}
    assert D.drive_key(a) == D.drive_key(b) != D.drive_key(dict(a, seed=2))


def _run(tmp, name, world, objs_per_frame):
    d = tmp / name
    (d / 'objects').mkdir(parents=True)
    (d / 'drive.json').write_text(json.dumps({'project': {'world': world}, 'weather_source': {'preset': 'ClearNoon'}}))
    with open(d / 'frames.jsonl', 'w') as fh:
        for i, objs in enumerate(objs_per_frame):
            (d / 'objects' / f'{i}.json').write_text(json.dumps({'objects': objs}))
            fh.write(json.dumps({'step': i, 'files': {'cam/objects': f'objects/{i}.json'}}) + '\n')
    return str(d)


def test_coverage_counts_bins_and_cells(tmp_path):
    car = {'class': 'car', 'range_m': 15.0, 'occluded': 0.1, 'truncated': 0.0}
    far = {'class': 'car', 'range_m': 90.0, 'occluded': 0.8, 'truncated': 0.6}
    ped = {'class': 'pedestrian', 'range_m': 30.0, 'occluded': 0.3, 'truncated': 0.0}
    r1 = D.coverage_of_run(_run(tmp_path, 'a', 'w1', [[car, ped], [car, far]]))
    r2 = D.coverage_of_run(_run(tmp_path, 'b', 'w2', [[ped]]))
    spec = D.new_spec('ds', 't', 'r', ['w1', 'w2'])
    cov = D.coverage(spec, [r1, r2])
    assert cov['have'] == {'car': 3, 'pedestrian': 2} and cov['frames'] == 3
    assert cov['bins']['distance_m']['car'] == [2, 0, 0, 1] and cov['bins']['occlusion']['car'] == [2, 0, 0, 1]
    assert cov['cells']['w1 / ClearNoon']['objects']['car'] == 3
    assert cov['fill']['car'] == pytest.approx(3 / 1000)
    assert D.measured_yield([r1, r2]) == {'car': 1.0, 'pedestrian': pytest.approx(2 / 3)}


def test_splits_never_leak():
    units = [f'drive{i}' for i in range(20)]
    s = D.split(units, {'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=1)
    flat = [u for v in s.values() for u in v]
    assert sorted(flat) == sorted(units) and len(set(flat)) == len(flat)
    assert len(s['train']) == 14 and len(s['val']) == 3 and len(s['test']) == 3
    assert D.split(units, {'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=1) == s


def test_version_is_hashed(tmp_path):
    d = tmp_path / 'runs' / 'r1'
    d.mkdir(parents=True)
    (d / 'run.json').write_text('{}')
    spec = D.new_spec('ds', 't', 'r', ['w1'])
    v = D.make_version(spec, [str(d)], {'train': ['runs/r1']}, 'drive', 1, str(tmp_path))
    assert v['runs'][0]['run'] == os.path.join('runs', 'r1') and len(v['sha256']) == 64
