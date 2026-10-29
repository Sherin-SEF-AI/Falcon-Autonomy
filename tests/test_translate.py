"""Learned sim-to-real translation (core.translate, core.translate_net) on the CPU with tiny networks, drives recorded
against the fake carla and synthetic "real" images (never a real dataset's): the drift measures on transforms of known
size; the generator starts as the identity at any size; PatchNCE prefers aligned patches; a training cancelled and
resumed ends with the weights of one that was not (bit for bit), writes its model card with the data's terms and
refuses another training in its folder; a translated run keeps every label file (hard links, the same bytes), loads
as a run, exports to COCO and trains a detector as its source does, refuses or flags frames over the drift threshold;
tampered weights are refused; the check measures drift and the gap before and after; the translated-against-raw
experiment runs its arms on the same frames; the command's exit codes; the Training page's tab. One GPU smoke test
runs only with FALCON_GPU_TESTS=1."""
import json
import os

import numpy as np
import pytest

from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import translate as TR
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (the GUI fixtures)

cv2 = pytest.importorskip('cv2')
torch = pytest.importorskip('torch')

TINY = {'iterations': 6, 'batch': 2, 'crop': 32, 'work_width': 96, 'ngf': 4, 'n_blocks': 1, 'ndf': 8,
        'num_patches': 16, 'device': 'cpu', 'log_every': 2, 'probe_every': 2, 'ckpt_every': 100, 'probes': 3}


def _scene(rng, h=96, w=160):
    """A synthetic street-like image: a sky band, a road trapezoid, boxes and texture (no real data)."""
    im = np.zeros((h, w, 3), np.uint8)
    im[: h // 3] = (rng.integers(120, 200), rng.integers(150, 210), 230)
    im[h // 3:] = (90, 100, 85)
    pts = np.array([[w // 2 - 6, h // 3], [w // 2 + 6, h // 3], [w - 1, h - 1], [0, h - 1]], np.int32)
    cv2.fillPoly(im, [pts], (70, 70, 72))
    for _ in range(4):
        x, y = int(rng.integers(0, w - 30)), int(rng.integers(h // 3, h - 20))
        cv2.rectangle(im, (x, y), (x + int(rng.integers(10, 30)), y + int(rng.integers(8, 18))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return np.clip(im + rng.normal(0, 6, im.shape), 0, 255).astype(np.uint8)


def _real_folder(path, n=16, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(path, exist_ok=True)
    for i in range(n):
        im = _scene(rng, 80, 200)                          # another aspect ratio, as a real camera's would be
        im = cv2.convertScaleAbs(im, alpha=1.4, beta=-30)  # darker, more contrast
        cv2.imwrite(os.path.join(path, f'{i:03d}.png'), im)
    return path


@pytest.fixture(scope='module')
def drives(tmp_path_factory):
    from test_training import record
    base = tmp_path_factory.mktemp('translate-drives')
    a, b = record(base, (('a', 3), ('b', 4)))
    real = _real_folder(str(base / 'real'))
    return {'a': a, 'b': b, 'real': real, 'base': base}


@pytest.fixture(scope='module')
def model(drives, tmp_path_factory):
    out = str(tmp_path_factory.mktemp('translator') / 'tr')
    doc = TR.train(dict(TINY, sim=[drives['a']], real=[f'folder:{drives["real"]}']), out)
    return out, doc


# ================================================================================================ drift
def test_drift_measures_on_transforms_of_known_size():
    rng = np.random.default_rng(1)
    im = TR.working(_scene(rng, 360, 640))
    objs = [{'class': 'car', 'bbox': [100, 200, 220, 280]}, {'class': 'road', 'bbox': [0, 0, 10, 10]},
            {'class': 'pedestrian', 'bbox': [300, 200, 310, 205]}]        # not a thing; too small
    same = TR.frame_drift(im, im.copy(), objs)
    assert same['drift'] == pytest.approx(0.0, abs=1e-6) and same['n_boxes'] == 1
    assert same['structure'] == pytest.approx(1.0, abs=1e-6) and same['edge_recall'] == same['edge_precision'] == 1.0
    tone = TR.frame_drift(im, (255 * (im / 255.0) ** 0.6).astype(np.uint8), objs)
    assert tone['drift'] < 0.1                                          # a tone curve moves nothing
    M = np.float32([[1, 0, 3], [0, 1, 0]])
    shifted = cv2.warpAffine(im, M, im.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
    sh = TR.frame_drift(im, shifted, objs)
    assert sh['drift'] > TR.DRIFT_THRESHOLD and sh['structure'] < 0.8
    local = im.copy()                                                   # only the car's box is scrambled
    local[200:280, 100:220] = np.random.default_rng(2).integers(0, 255, (80, 120, 3))
    loc = TR.frame_drift(im, local, objs)
    assert loc['worst_box'] < 0.5 < loc['structure'] and loc['drift'] > TR.DRIFT_THRESHOLD
    s = TR.drift_summary([same, tone, sh, loc])
    assert s['frames'] == 4 and s['flagged'] == 2 and s['drift']['max'] == pytest.approx(max(sh['drift'],
                                                                                           loc['drift']))
    with pytest.raises(TR.TranslateError):
        TR.frame_drift(im, im[:-1], objs)


def test_semantic_groups_and_one_hot():
    sem = np.arange(30, dtype=np.uint8).reshape(5, 6)
    g = TR.semantic_ids(sem)
    names = [n for n, _ in TR.GROUPS]
    assert names[g[0, 1]] == 'road' and names[g[2, 2]] == 'vehicle' and names[g[1, 5]] == 'sky'
    assert sorted({t for _, tags in TR.GROUPS for t in tags}) == list(range(30))        # every tag has a group
    oh = TR.one_hot(sem, sem.shape)
    assert oh.shape == (len(TR.GROUPS), 5, 6) and (oh.sum(0) == 1).all()
    assert not TR.one_hot(None, (5, 6)).any()                          # unlabelled: "unknown"


# ================================================================================================ the networks
def test_generator_starts_as_the_identity_and_keeps_the_size():
    from falcon_autonomy.core.translate_net import PatchDiscriminator, ResnetGenerator
    torch.manual_seed(0)
    G = ResnetGenerator(len(TR.GROUPS), 4, 1)
    x = torch.rand(2, 3, 37, 50) * 1.6 - 0.8                            # odd sizes
    y = G(x, None)
    assert y.shape == x.shape and torch.allclose(y, x, atol=1e-5)       # zero-initialised head: the identity
    feats = G.encoder_features(x)
    assert [f.shape[1] for f in feats] == G.tap_channels() and len(feats) == 5
    assert PatchDiscriminator(8)(x).shape[1] == 1
    for name, m in G.named_modules():                                  # nothing that can move pixels
        assert not isinstance(m, (torch.nn.ConvTranspose2d, torch.nn.AdaptiveAvgPool2d)), name


def test_patch_nce_prefers_aligned_patches():
    from falcon_autonomy.core.translate_net import patch_nce
    torch.manual_seed(0)
    k = torch.nn.functional.normalize(torch.randn(1, 64, 32), dim=-1)
    aligned = patch_nce(k.clone(), k)
    shuffled = patch_nce(k[:, torch.randperm(64)], k)
    assert aligned < 0.5 < shuffled


# ================================================================================================ training
def test_training_resumes_bit_for_bit_and_writes_its_model_card(drives, model, tmp_path):
    from falcon_autonomy.core.training import Cancelled
    out, doc = model
    assert doc['status'] == 'ok' and doc['method'] == 'cut' and doc['data']['sim_images'] == 10
    assert doc['terms'] == [TR.DATA_TERMS['folder']] and 'inherit' in doc['licence_note']
    assert doc['weights']['sha256'] and os.path.isfile(os.path.join(out, 'generator.pt'))
    assert len(doc['previews']) == 3 and all(os.path.isfile(os.path.join(out, p)) for p in doc['previews'])
    assert not os.path.exists(os.path.join(out, 'checkpoints'))       # a finished training keeps no optimizer state
    rows = [json.loads(x) for x in open(os.path.join(out, 'metrics.jsonl'))]
    assert [r['it'] for r in rows] == [2, 4, 6] and all('probe' in r and r['images_per_s'] > 0 for r in rows)
    run = json.loads(open(os.path.join(out, 'run.json')).read())
    assert run['status'] == 'ok' and run['provenance']['inputs'] and run['outputs']['generator.pt']
    # cancelled at iteration 3 and resumed: the very same weights
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        return calls['n'] == 4
    other = str(tmp_path / 'resumed')
    s = dict(TINY, sim=[drives['a']], real=[f'folder:{drives["real"]}'])
    with pytest.raises(Cancelled):
        TR.train(s, other, cancel=cancel)
    assert json.load(open(os.path.join(other, 'run.json')))['status'] == 'cancelled'
    assert os.path.isfile(os.path.join(other, 'checkpoints', 'last.pt'))
    TR.train(s, other)
    a = torch.load(os.path.join(out, 'generator.pt'), weights_only=True)['state_dict']
    b = torch.load(os.path.join(other, 'generator.pt'), weights_only=True)['state_dict']
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(v, torch.zeros_like(v)) for k, v in a.items() if k.startswith('head'))   # it learned
    assert TR.train(s, other)['key'] == doc['key']                    # finished: not trained again
    with pytest.raises(TR.TranslateError, match='another translator'):
        TR.train(dict(s, seed=1), other)
    with pytest.raises(TR.TranslateError):
        TR.normalise(dict(s, method='cyclegan'))
    with pytest.raises(TR.TranslateError):
        TR.normalise(dict(s, real=[f'falcon:{drives["b"]}']))


# ================================================================================================ applying
def _files(root):
    out = {}
    for d, _dirs, fs in os.walk(root):
        for f in fs:
            out[os.path.relpath(os.path.join(d, f), root)] = os.path.join(d, f)
    return out


def test_a_translated_run_keeps_every_label(drives, model, tmp_path):
    from falcon_autonomy.core import training as T
    from falcon_autonomy.core.exporters import Run, export
    out_m, _ = model
    out = str(tmp_path / 'a-tr')
    doc = TR.translate_run(out_m, drives['a'], out, threshold=1.0)
    src, dst = _files(drives['a']), _files(out)
    rgb = {r['file'] for r in (json.loads(x) for x in open(os.path.join(out, 'translation.jsonl')))}
    assert set(src) | {'translation.jsonl'} == set(dst) and len(rgb) == 10
    for rel, p in src.items():
        if rel in ('run.json', 'dataset.json') or rel in rgb:
            continue
        assert open(p, 'rb').read() == open(dst[rel], 'rb').read(), rel
        assert os.path.samefile(p, dst[rel]) == (os.sep in rel), rel      # per-frame files linked, records copied
    for rel in rgb:                                                     # new pixels, the same grid
        a, b = cv2.imread(src[rel]), cv2.imread(dst[rel])
        assert a.shape == b.shape and not os.path.samefile(src[rel], dst[rel])
    tr = doc['translation']
    assert doc['worker'] == 'translate' and doc['status'] == 'ok' and tr['source'] == os.path.abspath(drives['a'])
    assert tr['drift']['frames'] == 10 and tr['weights_sha256'] and tr['terms'] == [TR.DATA_TERMS['folder']]
    assert doc['started'] == json.load(open(os.path.join(drives['a'], 'run.json')))['started']
    run = Run(out)                                                     # loads as any drive
    assert run.frames == Run(drives['a']).frames
    a = T.load_source(f'falcon:{drives["a"]}', T.DEFAULT_CLASSES)[0]
    b = T.load_source(f'falcon:{out}', T.DEFAULT_CLASSES)[0]
    assert [(s['boxes'], s['classes'], s['ignore']) for s in a] == [(s['boxes'], s['classes'], s['ignore'])
                                                                    for s in b]
    res = export([out], 'coco', str(tmp_path / 'coco'))
    assert res and os.listdir(str(tmp_path / 'coco'))
    from falcon_autonomy.core.realgap import read_set
    assert len(read_set('falcon', out)['items']) == 10
    # the same translation again: nothing to do; another into the same folder: refused
    assert TR.translate_run(out_m, drives['a'], out, threshold=1.0)['translation']['key'] == tr['key']
    with pytest.raises(TR.TranslateError):
        TR.translate_run(out_m, drives['a'], out, threshold=0.5)
    with pytest.raises(TR.TranslateError):
        TR.translate_run(out_m, drives['a'], drives['a'])


def test_frames_over_the_threshold_are_refused_or_flagged(drives, model, tmp_path):
    out_m, _ = model
    raw = TR.translate_run(out_m, drives['b'], str(tmp_path / 'raw'), threshold=-1.0, on_drift='raw')
    d = raw['translation']['drift']
    assert d['flagged'] == d['refused'] == d['frames'] == 10
    for r in (json.loads(x) for x in open(tmp_path / 'raw' / 'translation.jsonl')):
        assert r['used'] == 'raw' and r['flagged']
        assert open(os.path.join(drives['b'], r['file']), 'rb').read() == open(tmp_path / 'raw' / r['file'],
                                                                              'rb').read()
    flag = TR.translate_run(out_m, drives['b'], str(tmp_path / 'flag'), threshold=-1.0, on_drift='flag')
    assert flag['translation']['drift']['flagged'] == 10 and flag['translation']['drift']['refused'] == 0
    with pytest.raises(TR.TranslateError):
        TR.translate_run(out_m, drives['b'], str(tmp_path / 'x'), on_drift='drop')


def test_tampered_weights_are_refused(model, tmp_path):
    import shutil
    out_m, _ = model
    copy = str(tmp_path / 'm')
    shutil.copytree(out_m, copy)
    with open(os.path.join(copy, 'generator.pt'), 'ab') as fh:
        fh.write(b'x')
    with pytest.raises(TR.TranslateError, match='SHA-256'):
        TR.Translator(copy, 'cpu')
    with pytest.raises(TR.TranslateError):
        TR.Translator(str(tmp_path), 'cpu')


def test_full_resolution_output_adds_the_networks_change(model):
    tr = TR.Translator(model[0], 'cpu')
    rng = np.random.default_rng(3)
    rgb = _scene(rng, 192, 320)                                         # twice the work width
    sem = np.full(rgb.shape[:2], 1, np.uint8)
    out = tr(rgb, sem)
    small = TR.working(rgb, tr.width)
    delta = tr.net(small, TR.working(sem, tr.width, nearest=True)).astype(np.float32) - small
    up = cv2.resize(delta, (320, 192), interpolation=cv2.INTER_CUBIC)
    assert out.shape == rgb.shape
    assert np.abs(out.astype(np.float32) - np.clip(rgb + up, 0, 255)).max() <= 0.5 + 1e-3
    assert tr(rgb, sem, native=True).shape == rgb.shape
    with pytest.raises(TR.TranslateError):
        tr(rgb, None)                                                   # conditioned: needs the label


# ================================================================================================ checking
class Thresholds:
    """A plug-in segmenter (test only): road where the image is dark grey, else sky or building by brightness."""

    def predict(self, rgb, meta):
        g = rgb.mean(-1)
        return np.where(g < 90, 1, np.where(g > 170, 11, 3)).astype(np.int32)


def test_check_measures_drift_the_gap_and_a_segmenter(drives, model, tmp_path):
    out_m, _ = model
    out = str(tmp_path / 'chk')
    TR.translate_run(out_m, drives['a'], out, threshold=1.0)
    seg = {'id': 'thr', 'name': 'Thresholds', 'kind': 'segmenter', 'runtime': 'python',
           'entry': 'test_translate:Thresholds', 'licence': 'GPL-3.0-or-later'}
    doc = TR.check_runs([out], f'folder:{drives["real"]}', images=10, segmenter=seg)
    assert doc['schema'] == TR.CHECK_SCHEMA and doc['drift']['frames'] == 10
    assert doc['runs'] == [os.path.abspath(out)] and doc['sources'] == [os.path.abspath(drives['a'])]
    stats = {r['stat'] for r in doc['gap']['appearance']}
    assert {'L_mean', 'sharpness', 'noise_sigma'} <= stats and doc['gap']['features_raw'] is None
    s = doc['segmenter']
    assert 0 <= s['agreement'] <= 1 and s['miou_raw'] is not None and s['frames'] >= 1
    text = TR.check_text(doc)
    assert 'drift over 10 frames' in text and 'L_mean' in text
    with pytest.raises(TR.TranslateError):
        TR.check_runs([drives['a']])                                    # not a translated run


# ================================================================================================ the experiment
def test_translated_against_raw_experiment(drives, model, tmp_path):
    from test_training import TINY as DET, kitti_from_runs
    from falcon_autonomy.core.experiments import Experiment
    out_m, _ = model
    base = tmp_path
    tra = str(base / 'a-tr')
    TR.translate_run(out_m, drives['a'], tra, threshold=1.0)
    # labelled "real" sets: KITTI-format folders made from fake drives (stand-ins, not real data)
    from test_training import record
    os.makedirs(base / 'r')
    rt, = record(base / 'r', (('rt', 5),))
    kitti_from_runs([rt], str(base / 'real-train'))
    kitti_from_runs([drives['b']], str(base / 'real-test'))
    d = {'synthetic': [f'falcon:{drives["a"]}'], 'translated': [f'falcon:{tra}'],
         'real_train': [f'kitti_object:{base / "real-train"}?units=image'],
         'real_test': [f'kitti_object:{base / "real-test"}?units=image'], 'shares': [50], 'budget_images': 8,
         'seeds': 1, 'n_boot': 50, 'val_share': 0.0, 'train': dict(DET, epochs=1)}
    res = TR.run_experiment(str(base / 'exps'), 'tvr', d)
    assert res['schema'] == TR.RESULTS_SCHEMA and [r['arm'] for r in res['arms']] == ['real only', 'raw 50 %',
                                                                                      'translated 50 %']
    assert res['test']['objects'] > 0 and all(r['AP']['value'] is not None for r in res['arms'])
    assert {(c['a'], c['b']) for c in res['comparisons']} == {('raw 50 %', 'translated 50 %'),
                                                             ('real only', 'raw 50 %'),
                                                             ('real only', 'translated 50 %')}
    exp = Experiment(str(base / 'exps' / 'tvr'))
    runs = {r['name']: r for r in exp.runs()}
    assert {'real-only-seed0', 'raw050-seed0', 'translated050-seed0'} <= set(runs)
    da = json.load(open(base / 'exps' / 'tvr' / 'runs' / 'raw050-seed0' / 'training' / 'data.json'))
    db = json.load(open(base / 'exps' / 'tvr' / 'runs' / 'translated050-seed0' / 'training' / 'data.json'))
    steps = lambda dd: [i['id'].rsplit('#', 1)[1] for i in dd['train']['images'] if i['domain'] == 'synthetic']  # noqa
    assert steps(da) == steps(db) and len(steps(da)) == 4                # the same frames, raw or translated
    assert 'translated 50 % − raw 50 %' in TR.table(res)
    # a translated run that is not the raw drive's: refused
    with pytest.raises(TR.TranslateError):
        TR.run_experiment(str(base / 'exps'), 'bad', dict(d, synthetic=[f'falcon:{drives["b"]}']))


# ================================================================================================ the command
def test_command_exit_codes(drives, model, tmp_path, capsys):
    out_m, _ = model
    assert TR.main_translate(['apply', '--model', str(tmp_path / 'none'), '--run', drives['a'], '--out',
                              str(tmp_path / 'x'), '--device', 'cpu']) == P.FAILED
    assert TR.main_translate(['results', str(tmp_path)]) == P.FAILED
    assert TR.main_translate(['train', '--out', str(tmp_path / 't'), '--real', f'folder:{drives["real"]}',
                              '--device', 'cpu']) == P.FAILED                 # no --sim
    out = str(tmp_path / 'cli-tr')
    assert TR.main_translate(['apply', '--model', out_m, '--run', drives['b'], '--out', out, '--device', 'cpu',
                              '--copy']) == P.OK
    assert TR.main_translate(['check', out, '--out', str(tmp_path / 'c')]) == P.OK
    assert os.path.isfile(tmp_path / 'c' / 'check.json')
    from falcon_autonomy.cli import main
    assert main(['translate', 'results', str(tmp_path)]) == P.FAILED      # dispatched by the CLI
    assert 'drift over 10 frames' in capsys.readouterr().out


# ================================================================================================ the page
def test_training_page_translation_tab(window, qapp, drives, model, monkeypatch):
    import shutil
    from PyQt6.QtCore import Qt
    page = window.go('training')
    tab = page.translation
    ws = window.workspace()
    shutil.copytree(model[0], os.path.join(ws, 'translators', 'tiny'))
    run = os.path.join(ws, 'runs', 'a-tr-tiny')
    TR.translate_run(os.path.join(ws, 'translators', 'tiny'), drives['a'], run, threshold=1.0)
    window.service.catalog.add_run(run)
    page.refresh()
    pump(qapp, timeout=0.2)
    assert tab.models.findData(os.path.join(ws, 'translators', 'tiny')) >= 0
    assert 'inherit' in tab.info.vals['Terms'].text() and tab.curve.topLevelItemCount() == 3
    assert tab.translated.currentData() == run and len(tab.viewer.rows) == 10
    assert not tab.viewer.after.pixmap().isNull() and 'drift' in tab.viewer.caption.text()
    tab.viewer.slider.setValue(5)
    assert 'step' in tab.viewer.caption.text()
    assert 'No gap measured yet' in tab.check_text.toPlainText()
    names = [tab.apply_runs.item(i).text() for i in range(tab.apply_runs.count())]
    assert 'a-tr-tiny' in names                                         # a translated run is a drive for training
    # the actions submit `falcon translate` GPU jobs
    sent = []
    monkeypatch.setattr(page, '_submit', lambda spec: sent.append(spec) or type('J', (), {'id': len(sent)})())
    monkeypatch.setattr(page, '_ready', lambda: True)
    for i in range(tab.apply_runs.count()):
        if tab.apply_runs.item(i).data(256) == run:
            tab.apply_runs.item(i).setCheckState(Qt.CheckState.Checked)
    tab.apply()
    tab.check()
    tab.train_runs.item(0).setCheckState(Qt.CheckState.Checked)
    tab.real.setText(f'folder:{drives["real"]}')
    tab.train()
    kinds = [s.argv[s.argv.index('translate') + 1] for s in sent]
    assert kinds == ['apply', 'check', 'train'] and all(s.needs_gpu for s in sent)
    assert sent[0].run_dir and '--threshold' in sent[0].argv and '--no-condition' not in sent[2].argv


# ================================================================================================ GPU
@pytest.mark.skipif(os.environ.get('FALCON_GPU_TESTS') != '1', reason='GPU tests need FALCON_GPU_TESTS=1')
def test_gpu_smoke(drives, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    from falcon_autonomy.core import training as T
    locks = T.gpu_locks(None, T.CancelFlag())
    try:
        doc = TR.train(dict(TINY, device='cuda', iterations=20, ngf=16, n_blocks=2, sim=[drives['a']],
                            real=[f'folder:{drives["real"]}']), str(tmp_path / 'gpu'))
    finally:
        for lk in locks:
            lk.release()
    assert doc['cost']['device']['type'] == 'cuda' and doc['cost']['peak_gpu_mib'] > 0
    assert doc['cost']['device']['amp'] == 'bfloat16'
