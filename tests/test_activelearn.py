"""Active learning (core.activelearn) on tiny CPU detectors from scratch and drives recorded against the fake carla:
the strategies' choices by hand (uncertainty order and ties, k-centre greedy, the hybrid's short-list, random seeded);
the frame scores of a Faster R-CNN against the entropy and margin worked out by hand from known class logits; the
experiment in rounds (round 0 shared, nested labelled sets, choices never already labelled, resume without
retraining); the analysis on predictions with known answers (AP per round, paired differences, the area under the
curve, the frames to reach random's final AP); the command; the loop's acquisition of generated frames and the training
source that keeps only them; the Training page's tab. One GPU smoke test runs only with FALCON_GPU_TESTS=1."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (GUI fixtures; Qt slot exceptions fail tests)

from falcon_autonomy.core import activelearn as A
from falcon_autonomy.core import protocol as P

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

TINY = {'arch': 'fasterrcnn_mobilenet_v3_small_fpn', 'init': 'scratch', 'size': [128, 64], 'batch': 2, 'epochs': 1,
        'workers': 0, 'device': 'cpu', 'box_head_size': 64}


@pytest.fixture(scope='module')
def drives(tmp_path_factory):
    from test_training import record
    base = tmp_path_factory.mktemp('al-drives')
    return record(base, (('p1', 3), ('p2', 4), ('p3', 5), ('t1', 3), ('t2', 5)))


def _definition(drives, **kw):
    d = {'pool': [f'falcon:{x}' for x in drives[:3]], 'test': [f'falcon:{x}' for x in drives[3:]], 'initial': 4,
         'budget': 4, 'rounds': 2, 'strategies': ['random', 'entropy', 'hybrid'], 'seeds': 2, 'n_boot': 60,
         'train': dict(TINY)}
    d.update(kw)
    return d


# ================================================================================================ choosing
def test_select_by_hand():
    sc = {'entropy': np.array([0.1, 0.9, 0.5, 0.9, 0.2, 0.7]), 'margin': np.array([0.3, 0.1, 0.8, 0.2, 0.8, 0.0]),
          'det_mean': np.zeros(6), 'det_max': np.array([0, 0, 0, 0, 0, 1.0]),
          'features': np.array([[0.0, 0], [1, 0], [2, 0], [10, 0], [11, 0], [5, 0]], np.float32)}
    assert A.select('entropy', sc, [], [0, 1, 2, 3, 4, 5], 2) == [1, 3]           # 0.9, 0.9 (ties: lower index)
    assert A.select('entropy', sc, [1], [0, 2, 3, 4, 5], 2) == [3, 5]
    assert A.select('margin', sc, [], range(6), 2) == [2, 4]
    assert A.select('det_max', sc, [], range(6), 1) == [5]
    # k-centre greedy: labelled at x = 0; the farthest first (11), then the farthest from {0, 11} (5)
    assert A.kcenter_greedy(sc['features'], [0], [1, 2, 3, 4, 5], 2) == [4, 5]
    assert A.select('kcenter', sc, [0], [1, 2, 3, 4, 5], 2) == [4, 5]
    # with nothing labelled the first is the farthest from the candidates' mean (4.83: x = 11)
    assert A.kcenter_greedy(sc['features'], [], [0, 1, 2, 3, 4, 5], 1) == [4]
    # hybrid: the 2 × 1 most uncertain by entropy (1, 3), then k-centre among them from the labelled x = 0: 3
    assert A.select('hybrid', sc, [0], [1, 2, 3, 4, 5], 1, hybrid_factor=2) == [3]
    a = A.select('random', sc, [], range(6), 3, np.random.default_rng(5))
    assert a == A.select('random', sc, [], range(6), 3, np.random.default_rng(5)) and len(set(a)) == 3
    assert A.select('entropy', sc, [], [2], 5) == [2]                              # never more than the candidates
    with pytest.raises(A.ActiveError, match='unknown strategy'):
        A.select('mc_dropout', sc, [], range(6), 1)


def test_frame_scores_by_hand(drives):
    """A Faster R-CNN whose class head gives every proposal the logits [2, 0, …, 0] (background 2): every proposal's
    foreground probability is 1 / (e² + K) ≥ 0.05, its entropy and margin follow by hand."""
    from falcon_autonomy.core import training as T
    s = T.normalise(dict(TINY, score_threshold=0.05))
    model, _ = T.build_model(s)
    K = len(s['classes'])

    class Fixed(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            sc, bb = self.inner(x)
            z = torch.zeros_like(sc)
            z[:, 0] = 2.0
            return z, bb
    model.roi_heads.box_predictor = Fixed(model.roi_heads.box_predictor)
    samples = T.load_source(f'falcon:{drives[0]}', s['classes'])[0][:3]
    out = A.score_frames(model, samples, s, torch.device('cpu'), 'max', batch=2)
    q = np.array([np.e ** 2] + [1.0] * K)
    q /= q.sum()
    H = -(q * np.log(q)).sum() / np.log(K + 1)
    assert np.allclose(out['entropy'], H) and np.allclose(out['margin'], 1 - (q[0] - q[1]))
    assert np.all(out['proposals'] > 0) and out['features'].shape[0] == 3
    assert np.allclose(np.linalg.norm(out['features'], axis=1), 1.0, atol=1e-5)
    s2 = A.score_frames(model, samples, s, torch.device('cpu'), 'sum', batch=2)
    assert np.allclose(s2['entropy'], H * out['proposals'])
    # det_mean / det_max: 1 − score over the model's own detections at least DET_FLOOR
    model.eval()
    from falcon_autonomy.core.models import preprocess
    with torch.inference_mode():
        dets = model([torch.from_numpy(preprocess(T.load_rgb(x['image']), T.input_spec(s))[0][0]) for x in samples])
    for k, d in enumerate(dets):
        sc = d['scores'].numpy()
        sc = sc[sc >= A.DET_FLOOR]
        assert out['det_max'][k] == pytest.approx(float((1 - sc).max()) if len(sc) else 0.0)


# ================================================================================================ the experiment
def test_experiment_rounds_resume_and_paired_intervals(drives, tmp_path):
    from falcon_autonomy.core.experiments import Experiment
    d = _definition(drives)
    res = A.run_active(str(tmp_path / 'exp'), 'al', d)
    exp = Experiment(str(tmp_path / 'exp' / 'al'))
    names = set(exp.run_names())
    assert {'seed0-r0', 'seed1-r0', 'entropy-seed0-r2', 'hybrid-seed1-r1', 'random-seed1-r2', 'analysis'} <= names
    assert len([n for n in names if n != 'analysis']) == 2 * (1 + 3 * 2)
    for st in ('random', 'entropy', 'hybrid'):
        lab0 = json.load(open(os.path.join(exp.path, 'runs', 'seed0-r0', 'labelled.json')))['ids']
        prev = set(lab0)
        for r in (1, 2):
            run = os.path.join(exp.path, 'runs', f'{st}-seed0-r{r}')
            sel = json.load(open(os.path.join(run, 'selection.json')))
            lab = json.load(open(os.path.join(run, 'labelled.json')))['ids']
            assert len(sel['ids']) == 4 and not set(sel['ids']) & prev        # never a frame already in training
            assert lab0 == json.load(open(os.path.join(exp.path, 'runs', 'seed0-r0', 'labelled.json')))['ids']
            assert set(lab) == prev | set(sel['ids']) and len(lab) == 4 + 4 * r
            assert set(sel['labelled_before']) == prev and sel['strategy'] == st and sel['round'] == r
            if st != 'random':
                assert sel['model']['weights_sha256'] and set(sel['scores']) == set(A.UNCERTAINTY)
            prev = set(lab)
    # checkpoints are gone once no selection needs them
    assert not [f for f in os.listdir(os.path.join(exp.path, 'runs', 'seed0-r0', 'training', 'checkpoints'))
                if f.endswith('.pt')]
    assert res['schema'] == A.SCHEMA and res['frames'] == [4, 8, 12] and res['strategies'] == ['random', 'entropy',
                                                                                              'hybrid']
    by = {s['strategy']: s for s in res['results']}
    assert by['entropy']['rounds'][0]['vs_random']['verdict'] == 'the same model (round 0 is shared)'
    assert by['random']['rounds'][0]['AP']['value'] == by['hybrid']['rounds'][0]['AP']['value']
    for s in res['results']:
        assert len(s['rounds']) == 3 and len(s['rounds'][2]['per_seed']) == 2
    assert res['test']['n_units'] == 2 and res['runs_trained'] == 14
    assert 'labels are free' in res['labels'] and 'Falcon' in res['labels']
    again = A.run_active(str(tmp_path / 'exp'), 'al', dict(d, score_batch=4))     # volatile: the same experiment
    assert again['runs_trained'] == 0 and again['key'] == res['key']
    with pytest.raises(Exception, match='another definition'):
        A.run_active(str(tmp_path / 'exp'), 'al', dict(d, budget=3))
    text = A.table(res)
    assert 'random' in text and 'area under the curve' in text


def test_refusals(drives, tmp_path):
    with pytest.raises(A.ActiveError, match='shares images or drives'):
        A.run_active(str(tmp_path), 'leak', _definition(drives, test=[f'falcon:{drives[0]}']))
    with pytest.raises(A.ActiveError, match='need 104 pool frames'):
        A.run_active(str(tmp_path), 'big', _definition(drives, initial=4, budget=50, rounds=2))
    with pytest.raises(A.ActiveError, match='must include random'):
        A.normalise_active(_definition(drives, strategies=['entropy']))
    with pytest.raises(A.ActiveError, match='unknown strategies'):
        A.normalise_active(_definition(drives, strategies=['random', 'bald']))


def test_analysis_on_known_predictions(tmp_path):
    """Random finds nothing until round 2 (AP 0, 0, 1); entropy finds everything from round 1 (0, 1, 1); two seeds
    alike; three test drives. By hand: the areas under the curves over frames 4–12 are 0.25 and 0.75; entropy reaches
    random's final AP at 8 frames (a third saved)."""
    from falcon_autonomy.core.experiments import Experiment
    exp = Experiment.create(str(tmp_path), 'known', {'kind': 'active'})
    frames = []
    for i, u in enumerate(['dA', 'dA', 'dB', 'dB', 'dC', 'dC']):
        gt = [{'bbox': [10 + 30 * k, 10, 20, 20], 'class': 'car', 'ignore': 0} for k in range(2)]
        frames.append({'image_id': f'im{i}', 'drive': u, 'gt': gt, 'det': []})
    found = [dict(f, det=[{'bbox': g['bbox'], 'class': 'car', 'score': 0.9} for g in f['gt']]) for f in frames]

    def write(name, fr):
        d = os.path.join(exp.path, 'runs', name, 'training')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'test_predictions.jsonl'), 'w') as fh:
            fh.writelines(json.dumps(f) + '\n' for f in fr)
    for sd in (0, 1):
        write(f'seed{sd}-r0', frames)
        write(f'random-seed{sd}-r1', frames)
        write(f'random-seed{sd}-r2', found)
        write(f'entropy-seed{sd}-r1', found)
        write(f'entropy-seed{sd}-r2', found)
    m = {'strategies': ['random', 'entropy'], 'n_boot': 200, 'boot_seed': 0, 'initial': 4, 'budget': 4,
         'aggregate': 'max', 'train': {'classes': ['car']}}
    res = A.analyse(exp, m, [0, 1], [4, 8, 12])
    rnd, ent = res['results']
    assert [r['AP']['value'] for r in rnd['rounds']] == pytest.approx([0, 0, 1])
    assert [r['AP']['value'] for r in ent['rounds']] == pytest.approx([0, 1, 1])
    assert rnd['aulc']['value'] == pytest.approx(0.25) and ent['aulc']['value'] == pytest.approx(0.75)
    c = ent['aulc_vs_random']
    assert c['diff'] == pytest.approx(0.5) and c['combined_ci'] == pytest.approx([0.5, 0.5])
    assert c['verdict'] == 'better than random: more data-efficient (only 3 test unit(s): not reliable)'
    r1 = ent['rounds'][1]['vs_random']
    assert r1['diff'] == pytest.approx(1.0) and r1['verdict'].startswith('better than random (only 3')
    assert ent['rounds'][2]['vs_random']['verdict'] == 'no detectable difference at 95 %'
    reach = ent['frames_to_reach_random_final']
    assert reach['value'] == pytest.approx(8.0) and reach['saving'] == pytest.approx(1 / 3)
    assert rnd['frames_to_reach_random_final']['value'] == pytest.approx(12.0)
    assert res['test'] == {'images': 6, 'objects': 12, 'unit': 'sequence', 'n_units': 3}
    assert any('3 test units' in w for w in res['warnings'])
    assert A.analyse(exp, m, [0, 1], [4, 8, 12])['key'] == res['key']               # cached
    assert A._frames_to_reach([4, 8, 12], [0.1, 0.3, 0.5], 0.4) == pytest.approx(10.0)
    assert A._frames_to_reach([4, 8, 12], [0.1, 0.3, 0.35], 0.4) is None


def test_cli(drives, tmp_path):
    from conftest import worker_env
    ws = tmp_path / 'ws'
    cfg = tmp_path / 'train.json'
    cfg.write_text(json.dumps({'train': TINY}))
    args = [sys.executable, '-m', 'falcon_autonomy', 'experiment', 'active', '--workspace', str(ws), '--id', 'cli',
            '--config', str(cfg), '--pool', f'falcon:{drives[0]}', '--pool', f'falcon:{drives[1]}', '--test',
            f'falcon:{drives[3]}', '--strategies', 'random,det_max', '--initial', '4', '--budget', '3', '--rounds',
            '1', '--seeds', '1', '--boot', '30']
    r = subprocess.run(args, capture_output=True, text=True, env=worker_env(), timeout=300)
    assert r.returncode == P.OK, r.stdout + r.stderr
    msgs = [P.parse_line(x) for x in r.stdout.splitlines()]
    assert msgs[0]['t'] == 'hello' and msgs[-1]['t'] == 'result' and msgs[-1]['status'] == 'ok'
    assert set(msgs[-1]['summary']) == {'random', 'det_max'} and 'area under the curve' in r.stdout
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'experiment', 'results',
                        str(ws / 'experiments' / 'cli')], capture_output=True, text=True, env=worker_env(), timeout=60)
    assert r.returncode == P.OK and 'det_max' in r.stdout and 'training frames' in r.stdout
    r = subprocess.run(args[:-2] + ['--strategies', 'random,nonsense'], capture_output=True, text=True,
                       env=worker_env(), timeout=60)
    assert r.returncode == P.FAILED and 'unknown strategies' in r.stdout


# ================================================================================================ the loop and sources
def test_loop_acquisition_and_the_selected_source(drives, tmp_path, project):
    from falcon_autonomy.core import loop as L
    from falcon_autonomy.core import training as T
    tdir = str(tmp_path / 'model')
    T.train(dict(TINY, export=False, train=[f'falcon:{drives[0]}?every=2'], val=[]), tdir)
    sel = A.select_frames(tdir, drives[1:3], 5, 'entropy', out=str(tmp_path / 'acq.json'))
    assert sel['chosen'] == 5 and sel['frames'] == 20 and len(sel['ids']) == 5 and 'free' in sel['note']
    smp, notes, ins = T.load_source({'kind': 'falcon', 'path': drives[1], 'select': str(tmp_path / 'acq.json')},
                                    T.DEFAULT_CLASSES)
    smp2 = T.load_source({'kind': 'falcon', 'path': drives[2], 'select': str(tmp_path / 'acq.json')},
                         T.DEFAULT_CLASSES)[0]
    assert sorted(x['id'] for x in smp + smp2) == sorted(sel['ids']) and 'selected' in notes[0]
    assert any(k.startswith('select:') for k in ins)
    # the loop's step 5: generate.acquisition picks the frames; step 6 trains on them only
    runner = L.Runner(project, svc=None)
    it = {'n': 1, 'model': {'name': 'tiny', 'runtime': 'onnx', 'source': {'training_run': tdir}}}
    os.makedirs(L.iteration_dir(project.workspace, 1), exist_ok=True)
    odd = {'generate': {'acquisition': {'strategy': 'hybrid', 'frames': 6}, 'seed': 2}}
    doc, rec = {}, {}
    got = runner._acquire(it, odd, drives[1:3], 20, doc, rec)
    assert got == {'chosen': 6} and doc['acquisition']['done'] and 'hybrid: 6 of 20' in doc['acquisition']['decision']
    acq = json.load(open(os.path.join(L.iteration_dir(project.workspace, 1), 'acquisition.json')))
    assert acq['strategy'] == 'hybrid' and len(acq['ids']) == 6
    doc = {}
    assert runner._acquire(it, {'generate': {'acquisition': {'strategy': 'entropy', 'frames': 50}}}, drives[1:3], 20,
                           doc, rec) is None and 'all kept' in doc['acquisition']['why']
    doc = {}
    plug = {'n': 1, 'model': {'name': 'syn', 'runtime': 'python', 'entry': 'x:synthetic'}}
    assert runner._acquire(plug, odd, drives[1:3], 20, doc, rec) is None
    assert 'every generated frame is kept' in doc['acquisition']['decision']
    assert runner._acquire(it, {'generate': {}}, drives[1:3], 20, {}, rec) is None              # the default: off
    o, errs = L.normalise_odd({'generate': {'acquisition': {'strategy': 'bald', 'frames': 0}}}, project)
    assert any('generate.acquisition' in e for e in errs)
    o, errs = L.normalise_odd({'generate': {'acquisition': {'strategy': 'entropy', 'frames': 10}}}, project)
    assert not any('generate.acquisition' in e for e in errs)


# ================================================================================================ the page
def test_training_page_active_tab(window, qapp, drives, tmp_path):
    from falcon_autonomy.core.provenance import write_json
    page = window.go('training')
    exp = os.path.join(window.workspace(), 'experiments', 'al-page')
    os.makedirs(exp)
    write_json(os.path.join(exp, 'experiment.json'), {'schema': 'falcon.experiment/1', 'id': 'al-page',
                                                      'kind': 'active', 'definition': {}})
    ap = lambda v, lo, hi: {'value': v, 'ci_low': lo, 'ci_high': hi}           # noqa: E731
    res = {'schema': A.SCHEMA, 'frames': [4, 8, 12], 'seeds': [0, 1], 'labels': 'labels are free',
           'test': {'images': 6, 'objects': 12, 'unit': 'sequence', 'n_units': 3}, 'warnings': [],
           'results': [{'strategy': 'random', 'rounds': [{'frames': f, 'AP': ap(v, v - .1, v + .1)} for f, v in
                                                         ((4, .2), (8, .3), (12, .4))],
                        'aulc': {'value': 0.3}, 'aulc_vs_random': None,
                        'frames_to_reach_random_final': {'value': 12.0, 'saving': 0.0}},
                       {'strategy': 'entropy', 'rounds': [{'frames': f, 'AP': ap(v, v - .1, v + .1)} for f, v in
                                                          ((4, .2), (8, .4), (12, .45))],
                        'aulc': {'value': 0.36}, 'frames_to_reach_random_final': {'value': 8.0, 'saving': 1 / 3},
                        'aulc_vs_random': {'diff': 0.06, 'ci_low': 0.02, 'ci_high': 0.1, 'combined_ci': [0.01, 0.11],
                                           'verdict': 'better than random: more data-efficient'}}]}
    write_json(os.path.join(exp, 'results.json'), res)
    page.load_active()
    page.als.setCurrentIndex(page.als.findData(exp))
    pump(qapp, timeout=0.2)
    assert page.al_table.topLevelItemCount() == 2 and 'saves 33 %' in page.al_table.topLevelItem(1).text(4)
    assert 'more data-efficient' in page.al_table.topLevelItem(1).text(3)
    page.al_chart.grab()                                              # paints the curves
    page.tabs.setCurrentIndex(1)
    page.load_experiments()
    page.exps.setCurrentIndex(page.exps.findData(exp))                # the mixed tab shows it as text, not an error
    pump(qapp, timeout=0.2)
    assert 'training frames' in page.exp_text.toPlainText()


# ================================================================================================ GPU
@pytest.mark.skipif(os.environ.get('FALCON_GPU_TESTS') != '1', reason='GPU tests need FALCON_GPU_TESTS=1')
def test_gpu_smoke(drives, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    from falcon_autonomy.core import training as T
    flag = T.CancelFlag()
    locks = T.gpu_locks(None, flag)
    try:
        res = A.run_active(str(tmp_path / 'gpu'), 'gpu', _definition(drives, seeds=1, rounds=1,
                                                                     train=dict(TINY, device='cuda')))
    finally:
        for lk in locks:
            lk.release()
    assert res['runs_trained'] == 1 + 3 and res['results'][1]['rounds'][1]['AP']['value'] is not None
