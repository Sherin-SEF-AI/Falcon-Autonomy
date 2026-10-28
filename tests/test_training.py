"""The training lab (core.training) on drives the worker records against the fake carla, on the CPU, with a tiny model
built from scratch and nothing downloaded: the data adapters against the objects JSON and a KITTI-format set; the
harness's ignore rule applied exactly as the harness applies it; a training that is cancelled mid-epoch and resumed
ends with the very weights of one that was not (seeded, bit for bit on the CPU), writes run.json, metrics and
checkpoints, and exports an ONNX model that core.models.Runner runs to the torch model's detections and the harness
evaluates; the COCO head cut; the command with its exit codes. One GPU smoke test runs only with FALCON_GPU_TESTS=1."""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from falcon_autonomy.core import harness as H
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import training as T

cv2 = pytest.importorskip('cv2')
torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

TINY = {'arch': 'fasterrcnn_mobilenet_v3_small_fpn', 'init': 'scratch', 'size': [128, 64], 'batch': 4, 'epochs': 2,
        'workers': 0, 'device': 'cpu', 'box_head_size': 64, 'verify_images': 2, 'export': False}
ORACLE = {'id': 'oracle', 'name': 'Oracle', 'kind': 'detector', 'runtime': 'python', 'entry': 'test_training:Oracle',
          'licence': 'GPL-3.0-or-later', 'falcon_classes': list(T.DEFAULT_CLASSES)}
KITTI_NAME = {'car': 'Car', 'truck': 'Truck', 'pedestrian': 'Pedestrian', 'bicycle': 'Cyclist', 'bus': 'Misc'}


class Oracle:
    """The frame's ground-truth boxes with score 1 (to read the harness's ground truth)."""
    synthetic = True

    def predict(self, rgb, meta):
        return [{'bbox': o['bbox'], 'class': o['class'], 'score': 1.0} for o in meta['objects'] if o.get('bbox')]


def record(base, names_seeds, seconds=0.5):
    """Drives recorded by the worker against the fake carla: a 160 × 96 camera with box labels."""
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.weather import resolve
    from test_workers import _cfg, _run
    p = Project.new('Test')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=96, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.doc['traffic_profiles'][0]['vehicles'] = 10
    out = []
    for name, seed in names_seeds:
        code, _, err = _run(_cfg(p, os.path.join(str(base), name), seconds=seconds, seed=seed,
                                 weather=resolve({'preset': 'ClearNoon'})))
        assert code == P.OK, err
        out.append(os.path.join(str(base), name))
    return out


def kitti_from_runs(runs, out, mapping=True, every=1):
    """A KITTI-object-format folder made from drives (a stand-in for a real set): image_2/*.png, label_2/*.txt with
    KITTI's type names (every other car written as Van), a DontCare line per image, and — with mapping — the devkit's
    train_rand.txt / train_mapping.txt naming each image's drive. Returns [(image, run, objects)]."""
    from falcon_autonomy.core.exporters import Run
    os.makedirs(os.path.join(out, 'image_2'), exist_ok=True)
    os.makedirs(os.path.join(out, 'label_2'), exist_ok=True)
    rows, maps = [], []
    i = 0
    for r in runs:
        run = Run(r)
        cid = next(iter(run.cams))
        for rec in run.camera_steps(cid, every):
            objs = run.objects(rec, cid) or []
            img = cv2.imread(run.p(run.file(rec, cid, 'rgb')))
            name = f'{i:06d}'
            cv2.imwrite(os.path.join(out, 'image_2', name + '.png'), img)
            lines = []
            for k, o in enumerate(objs):
                t = KITTI_NAME.get(o['class'], 'Misc')
                if t == 'Car' and k % 2:
                    t = 'Van'
                x0, y0, x1, y1 = o['bbox']
                lines.append(f'{t} 0.00 0 0.0 {x0:.2f} {y0:.2f} {x1:.2f} {y1:.2f} 1.5 1.6 3.9 0.0 0.0 '
                             f'{o.get("range_m") or 10.0:.2f} 0.0')
            lines.append('DontCare -1 -1 -10 0.00 0.00 10.00 10.00 -1 -1 -1 -1000 -1000 -1000 -10')
            with open(os.path.join(out, 'label_2', name + '.txt'), 'w') as fh:
                fh.write('\n'.join(lines) + '\n')
            rows.append((os.path.join(out, 'image_2', name + '.png'), r, objs))
            maps.append(f'2011_09_26 {os.path.basename(r)}_sync {rec["step"]:010d}')
            i += 1
    if mapping:
        with open(os.path.join(out, 'train_mapping.txt'), 'w') as fh:
            fh.write('\n'.join(maps) + '\n')
        with open(os.path.join(out, 'train_rand.txt'), 'w') as fh:
            fh.write(','.join(str(k + 1) for k in range(len(maps))) + '\n')
    return rows


@pytest.fixture(scope='module')
def drives(tmp_path_factory):
    base = tmp_path_factory.mktemp('train-drives')
    a, b, c = record(base, (('a', 3), ('b', 4), ('c', 5)))
    return {'a': a, 'b': b, 'c': c, 'base': base}


def test_falcon_source_matches_the_objects_json(drives):
    from falcon_autonomy.core.exporters import Run
    smp, notes, ins = T.load_source(f'falcon:{drives["a"]}', T.DEFAULT_CLASSES)
    run = Run(drives['a'])
    cid = next(iter(run.cams))
    assert len(smp) == len(run.camera_steps(cid)) == 10 and list(ins.values()) == [os.path.join(drives['a'],
                                                                                                'run.json')]
    n_boxes = 0
    for s, rec in zip(smp, run.camera_steps(cid)):
        objs = [o for o in json.loads(open(run.p(run.file(rec, cid, 'objects'))).read())['objects']
                if o['class'] in T.DEFAULT_CLASSES and o.get('bbox')]
        assert s['boxes'] == [[float(v) for v in o['bbox']] for o in objs]
        assert s['classes'] == [o['class'] for o in objs]
        assert s['ignore'] == [o['pixels'] < H.MIN_PIXELS or o['occluded'] > H.MAX_OCCLUDED for o in objs]
        assert s['unit'] == run.dir and s['domain'] == 'synthetic' and s['image'] == run.p(run.file(rec, cid, 'rgb'))
        n_boxes += len(objs)
    assert n_boxes > 10
    every = T.load_source({'kind': 'falcon', 'path': drives['a'], 'every': 3}, ['car'])[0]
    assert len(every) == 4 and all(set(s['classes']) <= {'car'} for s in every)


def test_ignore_rule_is_the_harness_rule(drives):
    """Ground truth as training sees it, frame by frame, equals the ground truth the harness scores (boxes, classes,
    ignore flags); ignored objects are not training targets."""
    runs = [drives[k] for k in 'abc']
    col = H.collect(runs, ORACLE)
    smp = T.load_sources([f'falcon:{r}' for r in runs], T.DEFAULT_CLASSES)[0]
    assert len(col['frames']) == len(smp)
    n_ign = 0
    for f, s in zip(col['frames'], smp):
        assert s['id'].endswith(f'#{f["step"]}')
        mine = [([b[0], b[1], b[2] - b[0], b[3] - b[1]], c, int(ig)) for b, c, ig in zip(s['boxes'], s['classes'],
                                                                                       s['ignore'])]
        theirs = [(g['bbox'], g['class'], g['ignore']) for g in f['gt']]
        assert mine == theirs
        n_ign += sum(s['ignore'])
    assert n_ign > 0                                      # the rule is exercised
    s = T.normalise(TINY)
    names = T.label_names(s)
    ds = T.DetectionData(smp, T.input_spec(s), {c: names.index(c) for c in s['classes']}, False, 0)
    for i, sm in enumerate(smp):
        x, tgt = ds[(i, 0)][:2]
        assert tuple(x.shape) == (3, 64, 128) and float(x.max()) <= 1.0
        assert len(tgt['boxes']) == sum(1 for ig in sm['ignore'] if not ig)
        assert all(names[int(l)] in s['classes'] and int(l) > 0 for l in tgt['labels'])    # 0 is the background
    # a flip mirrors the boxes about the image's centre line
    flip = T.DetectionData(smp, T.input_spec(s), {c: names.index(c) for c in s['classes']}, True, 0)
    k = next(i for i in range(len(smp)) if T._flip(0, 0, i) and len(ds[(i, 0)][1]['boxes']))
    b0, b1 = ds[(k, 0)][1]['boxes'], flip[(k, 0)][1]['boxes']
    assert torch.allclose(b1[:, [0, 2]], 128 - b0[:, [2, 0]]) and torch.equal(b1[:, [1, 3]], b0[:, [1, 3]])


def test_kitti_source_classes_boxes_and_sequences(drives, tmp_path):
    rows = kitti_from_runs([drives['c']], str(tmp_path / 'kitti'))
    smp, notes, _ = T.load_source(f'kitti_object:{tmp_path / "kitti"}', T.DEFAULT_CLASSES)
    assert len(smp) == len(rows) and any('sequence' in n for n in notes)
    for s, (img, run, objs) in zip(smp, rows):
        want = [o for o in objs if KITTI_NAME.get(o['class'], 'Misc') != 'Misc' and o['class'] in T.DEFAULT_CLASSES]
        assert s['classes'] == [o['class'] for o in want]         # Van → car, Cyclist → bicycle; DontCare, Misc out
        assert np.allclose(s['boxes'], [o['bbox'] for o in want], atol=0.01)
        assert s['ignore'] == [(o['bbox'][2] - o['bbox'][0]) * (o['bbox'][3] - o['bbox'][1]) < H.MIN_PIXELS
                               for o in want]
        assert s['unit'] == os.path.basename(run) + '_sync' and s['sequence'] and s['domain'] == 'real'
    plain = T.load_source({'kind': 'kitti_object', 'path': str(tmp_path / 'kitti'), 'units': 'image'},
                          T.DEFAULT_CLASSES)[0]
    assert all(s['unit'].startswith('image:') and not s['sequence'] for s in plain)
    with pytest.raises(T.TrainingError):
        T.parse_source('kitti_raw:/somewhere')


def _weights(path):
    return torch.load(path, map_location='cpu', weights_only=False)['model']


def test_training_resumes_bit_for_bit_and_exports(drives, tmp_path):
    s = dict(TINY, train=[f'falcon:{drives["a"]}?every=2', f'falcon:{drives["b"]}?every=2'],
             val=[f'falcon:{drives["c"]}?every=3'])
    a = T.train(s, str(tmp_path / 'a'))                              # uninterrupted
    assert a['status'] == 'ok' and a['epochs_run'] == 2 and a['onnx'] is None
    # interrupted after the first iteration of epoch 2 (cancel is asked before each iteration and each val batch)
    calls = itertools.count(1)
    with pytest.raises(T.Cancelled):
        T.train(s, str(tmp_path / 'b'), cancel=lambda: next(calls) >= 6)
    run = json.loads((tmp_path / 'b' / 'run.json').read_text())
    assert run['status'] == 'cancelled' and run['exit_code'] == P.CANCELLED
    assert run['summary'] == {'status': 'cancelled', 'epoch': 1, 'iteration': 1}
    assert (tmp_path / 'b' / 'checkpoints' / 'last.pt').is_file()
    assert len((tmp_path / 'b' / 'metrics.jsonl').read_text().splitlines()) == 1
    b = T.train(dict(s, export=True), str(tmp_path / 'b'))           # resumed; export does not change the training
    wa, wb = _weights(tmp_path / 'a' / 'checkpoints' / 'last.pt'), _weights(tmp_path / 'b' / 'checkpoints' /
                                                                            'last.pt')
    assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
    ma = [json.loads(x) for x in (tmp_path / 'a' / 'metrics.jsonl').read_text().splitlines()]
    mb = [json.loads(x) for x in (tmp_path / 'b' / 'metrics.jsonl').read_text().splitlines()]
    assert [r['train_loss'] for r in ma] == [r['train_loss'] for r in mb] and [r['epoch'] for r in mb] == [1, 2]
    assert [r['val']['AP'] for r in ma] == [r['val']['AP'] for r in mb] and mb[0]['iterations'] == 3
    run = json.loads((tmp_path / 'b' / 'run.json').read_text())
    assert run['status'] == 'ok' and run['timings']['sessions'] == 2 and run['timings']['images'] == 20
    assert run['timings']['images_per_s'] > 0 and run['device']['type'] == 'cpu' and run['versions']['torch']
    assert run['data']['train']['n_images'] == 10 and run['data']['val']['n_images'] == 4
    assert all(v['sha256'] for v in run['provenance']['inputs'].values()) and run['provenance']['seed'] == 0
    assert run['determinism']['cuda_bit_for_bit'] is None and 'model.onnx' in run['outputs']
    data = json.loads((tmp_path / 'b' / 'data.json').read_text())
    assert len(data['train']['images']) == 10 and data['train']['domains'] == {'synthetic': 10}
    # the exported model: a registration core.models runs, with the torch model's detections
    from falcon_autonomy.core.models import Runner, validate
    reg = json.loads((tmp_path / 'b' / 'model.json').read_text())
    assert validate(reg) == [] and reg['output']['format'] == 'boxes_scores_labels' and reg['runtime'] == 'onnx'
    assert reg['classes'][0] == '__background__' and reg['class_map']['car'] == 'car'
    from falcon_autonomy.core.project import Project, validate as validate_project
    proj = Project.new('T')
    proj.doc['models'] = [reg, T.registration(T.normalise(s), reg['path'], 'x' * 80)]   # ids the schema accepts
    assert validate_project(proj.doc) == [] and len(proj.doc['models'][1]['id']) == 64
    ver = b['verification']
    assert ver['ok'] and ver['torch_detections'] > 0 and ver['torch_detections'] == ver['onnx_detections']
    assert ver['max_box_diff_px'] <= 0.5 and ver['max_score_diff'] <= 1e-3
    import onnx
    assert onnx.load(reg['path']).opset_import[0].version == T.OPSET
    # independently of verify_onnx: Runner against the torch model on a frame
    model, _ = T.build_model(T.normalise(s))
    model.load_state_dict(_weights(tmp_path / 'b' / 'checkpoints' / 'best.pt'))
    smp = T.load_source(f'falcon:{drives["c"]}?every=5', T.DEFAULT_CLASSES)[0]
    tdet = T.predict(model, smp[:1], T.normalise(s), torch.device('cpu'))[0]['det']
    odet = Runner(reg).predict(T.load_rgb(smp[0]['image']))
    assert len(tdet) == len(odet) > 0
    left = list(odet)
    for d in tdet:                                    # each torch detection has its ONNX twin
        xyxy = np.array([d['bbox'][0], d['bbox'][1], d['bbox'][0] + d['bbox'][2], d['bbox'][1] + d['bbox'][3]])
        j = min(range(len(left)), key=lambda j: (left[j]['class'] != d['class'],
                                                 np.abs(np.array(left[j]['bbox']) - xyxy).max()))
        e = left.pop(j)
        assert e['class'] == d['class'] and abs(d['score'] - e['score']) < 1e-3
        assert np.abs(np.array(e['bbox']) - xyxy).max() < 0.5
    # the harness evaluates the trained model like any registered one
    ev = H.evaluate([drives['c']], reg, str(tmp_path / 'ev'), every=5, n_boot=20)
    assert ev['providers'] == ['CPUExecutionProvider'] and ev['n_frames'] == 2
    # a finished training is not run again; other settings in the same folder are refused
    assert T.train(dict(s, export=True), str(tmp_path / 'b')) == run['summary']
    with pytest.raises(T.TrainingError, match='other settings'):
        T.train(dict(s, seed=1), str(tmp_path / 'b'))


def test_pretrained_weights_are_never_fetched_by_themselves(tmp_path):
    s = dict(TINY, arch='fasterrcnn_mobilenet_v3_large_fpn', init='coco', box_head_size=1024,
             train=['falcon:/nowhere'])
    assert T.weights_needed(T.normalise(s)) == 'fasterrcnn_mobilenet_v3_large_fpn-coco'
    path, _ = T.find_weights('fasterrcnn_mobilenet_v3_large_fpn-coco', [str(tmp_path)])
    assert path is None
    (tmp_path / 'fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth').write_bytes(b'not the weights')
    path, notes = T.find_weights('fasterrcnn_mobilenet_v3_large_fpn-coco', [str(tmp_path)])
    assert path is None and 'not the pinned' in notes[0]               # a wrong file is refused, never used
    with pytest.raises(T.TrainingError, match='no COCO weights'):
        T.normalise(dict(TINY, init='coco'))
    for w in T.WEIGHTS.values():
        assert len(w['sha256']) == 64 and w['url'].startswith('https://download.pytorch.org/models/')
        assert os.path.splitext(w['file'])[0].rsplit('-', 1)[1] == w['sha256'][:8]       # torchvision's suffix


def test_coco_head_is_cut_to_falcon_classes(tmp_path):
    """A COCO detector's class head keeps the rows of the matching COCO classes (car ← car, pedestrian ← person) and
    starts a class COCO lacks (rider) fresh: checked on a stand-in COCO file of the right architecture."""
    import torchvision
    torch.manual_seed(0)
    coco = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None,
                                                                          num_classes=91)
    p = tmp_path / 'coco.pth'
    torch.save(coco.state_dict(), p)
    s = T.normalise(dict(TINY, arch='fasterrcnn_mobilenet_v3_large_fpn', init='coco', box_head_size=1024,
                         classes=['car', 'pedestrian', 'rider']))
    model, info = T.build_model(s, str(p))
    assert info['head'] == {'from_coco': {'car': 'car', 'pedestrian': 'person'}, 'fresh': ['rider']}
    assert info['norm'] == 'FrozenBatchNorm2d' and info['trainable_parameters'] < info['parameters']
    old, new = coco.roi_heads.box_predictor, model.roi_heads.box_predictor
    assert new.cls_score.out_features == 4
    for j, r in ((0, 0), (1, 3), (2, 1)):                            # background, car, person
        assert torch.equal(new.cls_score.weight[j], old.cls_score.weight[r])
        assert torch.equal(new.bbox_pred.weight[4 * j:4 * j + 4], old.bbox_pred.weight[4 * r:4 * r + 4])
    assert not any(torch.equal(new.cls_score.weight[3], old.cls_score.weight[r]) for r in range(91))
    # a dense head (RetinaNet's): anchors × classes rows
    import types
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    ch = RetinaNetClassificationHead(256, 9, 91)
    w0 = ch.cls_logits.weight.detach().clone()
    T._cut_head(types.SimpleNamespace(head=types.SimpleNamespace(classification_head=ch)), 'retinanet_v2',
                ['pedestrian', 'rider', 'car'], 0)
    assert ch.num_classes == 3 and ch.cls_logits.out_channels == 27
    nw, ow = ch.cls_logits.weight.view(9, 3, 256, 3, 3), w0.view(9, 91, 256, 3, 3)
    assert torch.equal(nw[:, 0], ow[:, 1]) and torch.equal(nw[:, 2], ow[:, 3])


def test_lr_schedule():
    assert T.lr_at(0, 100, 0.1, 10, 'cosine') == pytest.approx(0.01)
    assert T.lr_at(9, 100, 0.1, 10, 'cosine') == pytest.approx(0.1)
    assert T.lr_at(99, 100, 0.1, 10, 'cosine') < 1e-4
    assert T.lr_at(70, 100, 0.1, 0, 'step') == pytest.approx(0.01) and T.lr_at(95, 100, 0.1, 0, 'step') == \
        pytest.approx(0.001)


def _cli(args, env=None, stdin=None):
    from conftest import worker_env
    return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'train'] + args, capture_output=True, text=True,
                          env=worker_env(env), input=stdin, timeout=180)


def test_cli_trains_and_cancels(drives, tmp_path):
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps(dict(TINY, epochs=1, export=True)))
    args = ['--config', str(cfg), '--train', f'falcon:{drives["a"]}?every=5', '--val', f'falcon:{drives["c"]}?every=5']
    r = _cli(args + ['--out', str(tmp_path / 'ok')])
    assert r.returncode == P.OK, r.stdout + r.stderr
    msgs = [P.parse_line(x) for x in r.stdout.splitlines()]
    assert msgs[0]['t'] == 'hello' and msgs[0]['worker'] == 'train' and any(m['t'] == 'progress' for m in msgs)
    assert any(m['t'] == 'metric' and m['name'] == 'val_AP' for m in msgs)
    res = msgs[-1]
    assert res['t'] == 'result' and res['status'] == 'ok' and res['summary']['onnx_verified'] is True
    r = _cli(args + ['--out', str(tmp_path / 'cancel')], stdin='{"cmd": "cancel"}\n')
    assert r.returncode == P.CANCELLED, r.stdout + r.stderr
    assert json.loads((tmp_path / 'cancel' / 'run.json').read_text())['status'] == 'cancelled'
    assert (tmp_path / 'cancel' / 'checkpoints' / 'last.pt').is_file()
    r = _cli(['--out', str(tmp_path / 'x'), '--train', 'nonsense'])
    assert r.returncode == P.FAILED and 'KIND:PATH' in r.stdout


@pytest.mark.skipif(os.environ.get('FALCON_GPU_TESTS') != '1', reason='GPU tests need FALCON_GPU_TESTS=1')
def test_gpu_smoke(drives, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    s = dict(TINY, device='cuda', epochs=1, export=True, train=[f'falcon:{drives["a"]}'],
             val=[f'falcon:{drives["c"]}?every=5'])
    res = T.train(s, str(tmp_path / 'gpu'))
    run = json.loads((tmp_path / 'gpu' / 'run.json').read_text())
    assert res['verification']['ok'] and run['device']['amp'] is True and run['device']['gpu']['name']
    assert run['timings']['peak_gpu_mib'] > 0
