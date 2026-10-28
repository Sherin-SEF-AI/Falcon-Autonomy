"""Experiments (core.experiments): local tracking and its queries; the MLflow file-store layout, checked file by file;
the mixed sim/real experiment on tiny data (drives the worker records against the fake carla as the synthetic pool,
KITTI-format sets made from other drives as the real ones; a tiny model from scratch on the CPU) — arms × seeds runs,
a finished run skipped on resume, every arm compared with the real-only arm on the same test images, the seed spread,
the test set's sequences as the bootstrap's unit, and leakage refused."""
import json
import os

import pytest

from falcon_autonomy.core import experiments as E
from falcon_autonomy.core import protocol as P

pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

TINY = {'arch': 'fasterrcnn_mobilenet_v3_small_fpn', 'init': 'scratch', 'size': [128, 64], 'batch': 2, 'epochs': 1,
        'workers': 0, 'device': 'cpu', 'box_head_size': 64, 'export': False}


def _yaml(path):
    """MLflow's meta.yaml as written (flat key: value lines)."""
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def test_tracking_queries_and_mlflow_layout(tmp_path):
    exp = E.Experiment.create(str(tmp_path / 'experiments'), 'demo', {'kind': 'generic', 'seed': 1}, 'A demo')
    assert E.Experiment.create(str(tmp_path / 'experiments'), 'demo', {'kind': 'generic', 'seed': 1}).id == 'demo'
    with pytest.raises(E.ExperimentError, match='another definition'):
        E.Experiment.create(str(tmp_path / 'experiments'), 'demo', {'kind': 'generic', 'seed': 2})
    art = tmp_path / 'weights.bin'
    art.write_bytes(b'0123456789')
    for name, lr, aps in (('r1', 0.01, [0.1, 0.3]), ('r2', 0.02, [0.2, 0.25]), ('r3', 0.03, [0.5])):
        r = exp.run(name).start({'lr': lr, 'model': {'arch': 'x', 'size': [64, 32]}}, {'arm': name[-1]})
        for step, ap in enumerate(aps, 1):
            r.log_metrics({'val.AP': ap, 'loss': 1 - ap}, step=step, timestamp=1700000000000 + step)
        r.log_metric('AP.range_m[0,20)', 0.4, step=1)
        r.log_artifact(str(art))
        r.finish('failed' if name == 'r3' else 'ok')
    runs = exp.runs()
    assert [r['name'] for r in runs] == ['r1', 'r2', 'r3'] and runs[0]['metrics']['val.AP'] == 0.3
    assert exp.best('val.AP')['name'] == 'r1'                      # r3 failed: not a candidate
    assert exp.best('loss', 'min')['name'] == 'r1' and exp.best('val.AP', tags={'arm': '2'})['name'] == 'r2'
    assert exp.best('nothing') is None and len(exp.runs('ok')) == 2
    lines = [json.loads(x) for x in open(os.path.join(exp.run('r1').dir, 'metrics.jsonl'))]
    assert lines[0] == {'key': 'val.AP', 'value': 0.1, 'step': 1, 'timestamp': 1700000000001}
    assert exp.run('r1').artifacts()[0]['bytes'] == 10
    # MLflow's file store, file by file
    res = E.export_mlflow(exp.path, str(tmp_path / 'mlruns'))
    edir = tmp_path / 'mlruns' / res['experiment_id']
    em = _yaml(edir / 'meta.yaml')
    assert res['experiment_id'].isdigit() and em['experiment_id'] == res['experiment_id']
    assert em['name'] == 'A demo' and em['lifecycle_stage'] == 'active'
    assert em['artifact_location'] == f'file://{edir}' and isinstance(em['creation_time'], int)
    rid = res['runs']['r1']
    rdir = edir / rid
    rm = _yaml(rdir / 'meta.yaml')
    assert rm['run_id'] == rm['run_uuid'] == rid and len(rid) == 32 and rm['experiment_id'] == res['experiment_id']
    assert rm['run_name'] == 'r1' and rm['status'] == 3 and rm['source_type'] == 4 and rm['tags'] == []
    assert rm['artifact_uri'] == f'file://{rdir}/artifacts' and rm['end_time'] >= rm['start_time']
    assert _yaml(edir / res['runs']['r3'] / 'meta.yaml')['status'] == 4          # FAILED
    assert (rdir / 'metrics' / 'val.AP').read_text() == '1700000000001 0.1 1\n1700000000002 0.3 2\n'
    assert (rdir / 'metrics' / 'AP.range_m_0_20_').is_file()                     # a name MLflow accepts
    assert (rdir / 'params' / 'lr').read_text() == '0.01'
    assert (rdir / 'params' / 'model.size').read_text() == '[64,32]'
    assert (rdir / 'tags' / 'mlflow.runName').read_text() == 'r1' and (rdir / 'tags' / 'arm').read_text() == '1'
    assert (rdir / 'artifacts' / 'weights.bin').read_bytes() == b'0123456789'
    again = E.export_mlflow(exp.path, str(tmp_path / 'mlruns'))                 # stable ids: same folders
    assert again['runs'] == res['runs'] and sorted(os.listdir(edir)) == sorted(list(res['runs'].values()) +
                                                                               ['meta.yaml'])


@pytest.fixture(scope='module')
def pools(tmp_path_factory):
    from test_training import kitti_from_runs, record
    base = tmp_path_factory.mktemp('mixed')
    syn = record(base, (('s1', 3), ('s2', 4)))
    real = record(base, (('r1', 11), ('r2', 12), ('t1', 13), ('t2', 14)))
    kitti_from_runs(real[:2], str(base / 'kitti_train'), every=2)
    kitti_from_runs(real[2:], str(base / 'kitti_test'), every=2)
    return {'syn': syn, 'train': str(base / 'kitti_train'), 'test': str(base / 'kitti_test'), 'base': base}


def _definition(pools, **kw):
    d = {'synthetic': [f'falcon:{p}?every=2' for p in pools['syn']], 'real_train': [f'kitti_object:{pools["train"]}'],
         'real_test': [f'kitti_object:{pools["test"]}'], 'shares': [0, 50, 100], 'budget_images': 4, 'seeds': 2,
         'val_share': 0.5, 'n_boot': 40, 'train': TINY}
    d.update(kw)
    return d


def test_mixed_experiment_arms_seeds_resume_and_paired_intervals(pools, tmp_path):
    root = str(tmp_path / 'experiments')
    msgs = []

    class Em:
        def progress(self, f, text='', force=False):
            msgs.append(('progress', f))

        def log(self, text, level='info'):
            msgs.append(('log', text))

        def metric(self, *a, **k):
            msgs.append(('metric', a[0]))
    res = E.run_mixed(root, 'mix', _definition(pools), Em())
    assert res['runs_trained'] == 6 and res['runs_skipped'] == 0
    assert [a['arm'] for a in res['arms']] == ['0%', '50%', '100%'] and res['base_arm'] == '0%'
    assert [(a['n_real'], a['n_synthetic']) for a in res['arms']] == [(4, 0), (2, 2), (0, 4)]
    assert res['test']['unit'] == 'sequence' and res['test']['n_units'] == 2 and res['test']['images'] == 10
    assert any('not reliable' in w for w in res['warnings'])        # two test sequences are too few
    assert res['seeds'] == [0, 1] and res['iterations_per_run'] == 2
    exp = E.Experiment(os.path.join(root, 'mix'))
    names = {r['name'] for r in exp.runs('ok')}
    assert names == {E.run_name(s, k) for s in (0, 0.5, 1) for k in (0, 1)} | {'analysis'}
    for arm in res['arms']:
        assert len(arm['per_seed']) == 2 and {p['seed'] for p in arm['per_seed']} == {0, 1}
        assert arm['AP']['value'] == pytest.approx(sum(p['AP'] for p in arm['per_seed']) / 2)
        assert arm['AP']['ci_low'] <= arm['AP']['value'] + 1e-12 <= arm['AP']['ci_high'] + 2e-12
        assert arm['seed_spread']['AP']['sd'] is not None
    assert res['arms'][0]['vs_real_only'] is None
    base = res['arms'][0]['AP']['value']
    for arm in res['arms'][1:]:
        c = arm['vs_real_only']['AP']
        assert c['diff'] == pytest.approx(arm['AP']['value'] - base)       # against the real-only arm
        assert c['combined_ci'] is not None and c['verdict']
    # every run trained the same budget for the same iterations, and was tested on the same real images
    for r in exp.runs('ok'):
        if r['name'] == 'analysis':
            continue
        tr = json.loads(open(os.path.join(exp.run(r['name']).dir, 'training', 'run.json')).read())
        assert tr['data']['train']['n_images'] == 4 and tr['data']['val']['domains'] == {'real': 5}
        assert tr['data']['test']['n_images'] == 10 and tr['data']['test']['domains'] == {'real': 10}
        assert tr['timings']['images'] == 4 and not os.path.isfile(os.path.join(exp.run(r['name']).dir, 'training',
                                                                                'checkpoints', 'last.pt'))
        assert r['metrics']['test.AP'] is not None and 'test.AP.ci_low' in r['metrics']
        want = {'0%': {'real': 4}, '50%': {'real': 2, 'synthetic': 2}, '100%': {'synthetic': 4}}[r['tags']['arm']]
        assert tr['data']['train']['domains'] == want
    # resume: finished runs are skipped; a run that did not finish is trained again, alone
    again = E.run_mixed(root, 'mix', _definition(pools, train=dict(TINY, workers=2, export=True)))   # volatile
    assert again['runs_trained'] == 0 and again['runs_skipped'] == 6 and again['key'] == res['key']
    one = exp.run(E.run_name(0.5, 1))
    preds = os.path.join(one.dir, 'training', 'test_predictions.jsonl')
    before = open(preds).read()
    one.finish('cancelled')
    os.remove(os.path.join(one.dir, 'training', 'run.json'))
    os.remove(preds)
    third = E.run_mixed(root, 'mix', _definition(pools))
    assert third['runs_trained'] == 1 and third['runs_skipped'] == 5
    assert open(preds).read() == before                            # retrained from scratch: the CPU repeats bit for bit
    assert third['key'] == res['key'] and third['arms'][1]['AP'] == res['arms'][1]['AP']
    with pytest.raises(E.ExperimentError, match='another definition'):
        E.run_mixed(root, 'mix', _definition(pools, seeds=3))
    # the command: the table, the best run, the MLflow export
    from falcon_autonomy.core.experiments import main_experiment
    assert main_experiment(['results', exp.path]) == P.OK
    assert main_experiment(['best', exp.path, '--metric', 'test.AP']) == P.OK
    assert main_experiment(['export', exp.path, '--mlflow', str(tmp_path / 'mlruns'), '--no-artifacts']) == P.OK
    eid = next(d for d in os.listdir(tmp_path / 'mlruns'))
    assert len([d for d in os.listdir(tmp_path / 'mlruns' / eid) if d != 'meta.yaml']) == 7
    assert E.table(res).splitlines()[2].split()[0] == '0%'


def test_mixed_experiment_refuses_leakage_and_bad_definitions(pools, tmp_path):
    with pytest.raises(E.ExperimentError, match='share'):
        E.run_mixed(str(tmp_path), 'leak', _definition(pools, real_test=[f'kitti_object:{pools["train"]}']))
    with pytest.raises(E.ExperimentError, match='include 0'):
        E.normalise_mixed(_definition(pools, shares=[25, 50]))
    with pytest.raises(E.ExperimentError, match='budget'):
        E.run_mixed(str(tmp_path), 'big', _definition(pools, budget_images=500))
    with pytest.raises(E.ExperimentError, match='real sets'):
        E.normalise_mixed(_definition(pools, real_test=[f'falcon:{pools["syn"][0]}']))


def test_analysis_on_known_predictions(tmp_path):
    """The analysis on predictions with known answers: the real-only arm finds nothing (AP 0), the synthetic arm finds
    every object with seed 0 and all but one with seed 1; three test sequences and two lone images are the units."""
    m = {'shares': [0.0, 1.0], 'n_boot': 300, 'boot_seed': 0, 'train': {'classes': ['car']}}
    exp = E.Experiment.create(str(tmp_path), 'known', {'kind': 'mixed'})
    units = ['seqA', 'seqA', 'seqB', 'seqB', 'seqC', 'seqC', 'image:x', 'image:y']
    frames = []
    for i, u in enumerate(units):
        gt = [{'bbox': [10 + 30 * k, 10, 20, 20], 'class': 'car', 'ignore': 0} for k in range(3)]
        frames.append({'image_id': f'im{i}', 'drive': u, 'gt': gt, 'det': []})

    def write(share, seed, fr):
        d = os.path.join(exp.path, 'runs', E.run_name(share, seed), 'training')
        os.makedirs(d)
        with open(os.path.join(d, 'test_predictions.jsonl'), 'w') as fh:
            fh.writelines(json.dumps(f) + '\n' for f in fr)
    for seed in (0, 1):
        write(0.0, seed, frames)
        found = []
        for i, f in enumerate(frames):
            dets = [dict(g, score=0.9) for g in f['gt']]
            if seed == 1 and i == 0:
                dets = dets[1:]                                    # seed 1 misses one object
            found.append(dict(f, det=[{'bbox': d['bbox'], 'class': 'car', 'score': d['score']} for d in dets]))
        write(1.0, seed, found)
    res = E.analyse(exp, m, [0, 1], 8, 4)
    assert res['test'] == {'images': 8, 'objects': 24, 'unit': 'sequence where the set gives one, else image',
                           'n_units': 5}
    real, syn = res['arms']
    assert real['AP']['value'] == 0.0 and real['AP']['ci_low'] == real['AP']['ci_high'] == 0.0
    ap = [p['AP'] for p in syn['per_seed']]
    assert ap[0] == pytest.approx(1.0) and 0.9 < ap[1] < 1.0 and syn['AP']['value'] == pytest.approx(sum(ap) / 2)
    import statistics
    assert syn['seed_spread']['AP']['sd'] == pytest.approx(statistics.stdev(ap))
    c = syn['vs_real_only']['AP']
    assert c['diff'] == pytest.approx(syn['AP']['value']) and 0 < c['ci_low'] <= c['ci_high'] <= 1.0
    se = statistics.stdev(ap) / 2 ** 0.5
    h = 1.96 * (c['sd_boot'] ** 2 + se ** 2) ** 0.5
    assert c['combined_ci'] == pytest.approx([c['diff'] - h, c['diff'] + h])
    assert c['verdict'] == 'better than real only' and c['p_arm_gt_real_only'] == 1.0
    # one seed: the verdict says the training noise was not measured
    single = E.Experiment.create(str(tmp_path), 'one', {'kind': 'mixed'})
    _copy_runs(exp.path, single.path, seed=0)
    one = E.analyse(single, m, [0], 8, 4)
    assert one['arms'][1]['vs_real_only']['AP']['combined_ci'] is None
    assert 'training noise not measured' in one['arms'][1]['vs_real_only']['AP']['verdict']
    assert any('one seed' in w for w in one['warnings'])


def _copy_runs(src, dst, seed):
    import shutil
    for n in os.listdir(os.path.join(src, 'runs')):
        if n.endswith(f'-seed{seed}'):
            shutil.copytree(os.path.join(src, 'runs', n), os.path.join(dst, 'runs', n))
