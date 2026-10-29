"""TensorRT deployment (core.deploy, hil/engine_device.py) on the SIMULATED DEVICE: the device tool run on this machine
with onnxruntime on the CPU standing in for TensorRT (not TensorRT, not the Orin — the pipeline only). Fixtures are
real: small falcon-raw drives written here with known boxes (grey squares on black, a car each) and a real ONNX
detector built here with onnx.helper (a 1 × 1 convolution to grey and a 16 × 16 average pool: a square's window scores
its brightness), so the whole flow — frames, copy, calibration, build, profile, accuracy against the host's ONNX
reference with paired intervals, the gate, the report — runs end to end and its numbers can be checked. The SSH path
runs through stand-in ssh/scp programs that execute "on the device" here. Parsers of TensorRT's outputs (trtexec
layer info, build logs) are checked on hand-written samples of their documented formats, and the INA3221 power
sampler on a hand-made sysfs tree: both labelled as such, and no number from them is presented as a measurement."""
import json
import os
import stat
import sys

import numpy as np
import pytest

from falcon_autonomy.core import deploy as DP
from falcon_autonomy.core import models as MD
from falcon_autonomy.core import protocol as P
from falcon_autonomy.hil import engine_device as T

onnx = pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')
cv2 = pytest.importorskip('cv2')
from onnx import TensorProto, helper  # noqa: E402

W, H, MW, MH = 128, 96, 64, 48            # image and model input sizes
SQ = 32                                   # a car: a SQ × SQ grey square on the 16-pixel grid of the image


def blob_detector(path):
    """A YOLO-layout ONNX detector: grey = 1 × 1 conv (BT.601 weights), score = 16 × 16 average pool with stride 8 over
    the 64 × 48 input (35 windows), boxes = the windows (constants); one class, car."""
    inp = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, MH, MW])
    gw = helper.make_tensor('gw', TensorProto.FLOAT, [1, 3, 1, 1], [0.299, 0.587, 0.114])
    gy, gx = (MH - 16) // 8 + 1, (MW - 16) // 8 + 1
    boxes = np.array([[8 * j + 8, 8 * i + 8, 16, 16] for i in range(gy) for j in range(gx)], np.float32).T[None]
    bx = helper.make_tensor('bx', TensorProto.FLOAT, list(boxes.shape), boxes.ravel().tolist())
    shp = helper.make_tensor('shp', TensorProto.INT64, [3], [1, 1, gy * gx])
    nodes = [helper.make_node('Conv', ['images', 'gw'], ['grey']),
             helper.make_node('AveragePool', ['grey'], ['pool'], kernel_shape=[16, 16], strides=[8, 8]),
             helper.make_node('Reshape', ['pool', 'shp'], ['score']),
             helper.make_node('Concat', ['bx', 'score'], ['output0'], axis=1)]
    g = helper.make_graph(nodes, 'blob', [inp], [helper.make_tensor_value_info('output0', TensorProto.FLOAT,
                                                                                [1, 5, gy * gx])],
                          initializer=[gw, bx, shp])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path


def registration(path):
    return {'id': 'blob', 'name': 'Blob detector (test)', 'kind': 'detector', 'runtime': 'onnx', 'path': path,
            'input': {'width': MW, 'height': MH, 'layout': 'NCHW', 'scale': 1 / 255, 'rgb': True},
            'output': {'format': 'yolo', 'score_threshold': 0.3, 'nms_iou': 0.5}, 'classes': ['car'],
            'class_map': {'car': 'car'}, 'licence': 'GPL-3.0-or-later'}


def make_drive(path, seed, n=6):
    """A minimal finished falcon-raw drive: n frames of grey squares (cars, brightness 0.35–1) on black, with their
    boxes as ground truth, and a small pedestrian the detector cannot name."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(path, 'rgb', 'cam'))
    os.makedirs(os.path.join(path, 'objects', 'cam'))
    cam = {'id': 'cam', 'name': 'front', 'type': 'camera.pinhole', 'width': W, 'height': H, 'fov': 90.0,
           'pose': {'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}}
    json.dump({'schema': 'falcon.run/1', 'worker': 'drive', 'status': 'ok', 'started': '2026-10-29T00:00:00Z',
               'provenance': {}}, open(os.path.join(path, 'run.json'), 'w'))
    json.dump({'schema': 'falcon.dataset/1', 'layout': 'falcon-raw', 'fps': 10.0, 'frames': n, 'sensors': [cam]},
              open(os.path.join(path, 'dataset.json'), 'w'))
    json.dump({'rig': {'sensors': [cam]}, 'weather': {'preset': 'ClearNoon'}, 'town': 'Synthetic', 'fps': 10.0},
              open(os.path.join(path, 'drive.json'), 'w'))
    cells = [(a, b) for a in range(W // 16 - 1) for b in range(H // 16 - 1)]
    with open(os.path.join(path, 'frames.jsonl'), 'w') as fh:
        for k in range(n):
            img = np.zeros((H, W, 3), np.uint8)
            objs, used = [], set()
            for c in rng.permutation(len(cells))[:int(rng.integers(1, 4))]:
                a, b = cells[c]
                if any(abs(a - u) < 3 and abs(b - v) < 3 for u, v in used):
                    continue
                used.add((a, b))
                v = int(rng.uniform(0.35, 1.0) * 255)
                x0, y0 = 16 * a, 16 * b
                img[y0:y0 + SQ, x0:x0 + SQ] = v
                objs.append({'id': len(objs), 'track_id': len(objs), 'class': 'car', 'bbox': [x0, y0, x0 + SQ, y0 + SQ],
                             'pixels': SQ * SQ, 'occluded': 0.0, 'truncated': 0.0, 'range_m': 10.0})
            objs.append({'id': 99, 'track_id': 99, 'class': 'pedestrian', 'bbox': [2, 2, 8, 14], 'pixels': 72,
                         'occluded': 0.0, 'truncated': 0.0, 'range_m': 30.0})
            cv2.imwrite(os.path.join(path, 'rgb', 'cam', f'{k:06d}.png'), img[:, :, ::-1])
            json.dump({'objects': objs}, open(os.path.join(path, 'objects', 'cam', f'{k:06d}.json'), 'w'))
            files = {'cam/rgb': f'rgb/cam/{k:06d}.png', 'cam/objects': f'objects/cam/{k:06d}.json'}
            fh.write(json.dumps({'step': k, 'frame': k, 'sim_t': k / 10.0, 'files': files,
                                 'ego': {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}}) + '\n')
    return path


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    base = tmp_path_factory.mktemp('deploy')
    model = blob_detector(str(base / 'blob.onnx'))
    p = Project.new('Deploy test')
    p.doc['models'] = [registration(model)]
    proj = str(base / 'deploy.falcon.json')
    p.save(proj)
    runs = [make_drive(str(base / f'ev{i}'), seed=i) for i in range(5)]
    calib = [make_drive(str(base / f'cal{i}'), seed=100 + i, n=8) for i in range(2)]
    return {'base': base, 'model': model, 'm': registration(model), 'project': proj, 'runs': runs, 'calib': calib}


@pytest.fixture(scope='module')
def built(world):
    """The whole simulated flow once, through the command line (`falcon deploy all`)."""
    out = str(world['base'] / 'dep')
    argv = ['all', world['project'], '--model', 'blob', '--out', out, '--calib-count', '10', '--runs', '60',
            '--warmup-ms', '50', '--min-seconds', '0.2', '--idle-s', '0', '--boot', '200', '--threads', '1']
    for r in world['runs']:
        argv += ['--run', r]
    for r in world['calib']:
        argv += ['--calib-run', r]
    assert DP.main_deploy(argv) == P.OK
    return out, DP.read_doc(out)


# ============================================================================================ the same rules
def test_geometry_then_normalise_is_preprocess_bit_for_bit():
    """The host prepares frames with geometry(), the device normalises them with the copy of the same module."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (90, 160, 3), np.uint8)
    for spec in ({'width': 64, 'height': 48}, {'width': 64, 'height': 64, 'letterbox': True},
                 {'width': 160, 'height': 90, 'rgb': False, 'layout': 'NHWC', 'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}):
        x, back = MD.preprocess(img, spec)
        g, back2 = MD.geometry(img, spec)
        assert g.dtype == np.uint8 and back == back2
        assert np.array_equal(MD.normalise(g, spec), x)


def test_simulated_flow_end_to_end(built, world):
    out, doc = built
    assert doc['simulated'] and 'not TensorRT, not the Orin' in doc['label']
    assert doc['device']['mode'] == 'sim' and doc['device']['backend'] == 'onnxruntime'
    assert doc['plan']['eval_frames'] == 30 and doc['plan']['calib_frames'] == 10
    assert set(doc['engines']) == {'fp32', 'fp16', 'int8'}
    # the device got a verbatim copy of core/models.py and of the tool
    sim = os.path.join(out, 'simdevice')
    assert open(os.path.join(sim, 'falcon_models.py'), 'rb').read() == open(MD.__file__, 'rb').read()
    assert open(os.path.join(sim, 'engine_device.py'), 'rb').read() == open(T.__file__, 'rb').read()
    for name, e in doc['engines'].items():
        b = e['build']
        assert b['simulated'] and b['label'] == T.SIMULATED and os.path.isfile(os.path.join(out, 'engines', name,
                                                                                            'build.log'))
        p = e['profile']
        assert p['simulated'] and p['latency_ms']['n'] >= 60 and p['latency_ms']['p95'] >= p['latency_ms']['median']
        assert p['power']['available'] is False and 'not measured' in p['power']['why']    # no INA3221 here
        assert p['throughput_qps'] > 0
    int8 = doc['engines']['int8']
    assert int8['build']['layers']['quantize_nodes'] > 0 and doc['calibration']['frames'] == 10
    assert 'entropy' in doc['calibration']['method']
    # accuracy: the simulated fp32 engine IS the reference's model and decoder: identical detections, ratio 1
    ref = doc['reference']
    assert ref['frames'] == 30 and ref['providers'] == ['CPUExecutionProvider']
    fp32 = doc['engines']['fp32']['accuracy']
    assert fp32['agreement']['identical_detections'] == 30
    r = fp32['vs_reference']['metrics']['AP_ratio']
    assert r['value'] == pytest.approx(1.0) and r['ci_low'] == pytest.approx(1.0) and r['ci_high'] == pytest.approx(1.0)
    assert fp32['vs_reference']['unit'] == 'drive' and fp32['vs_reference']['n_units'] == 5
    assert 0.3 < ref['scores']['AP']['value'] < 1.0                  # dim squares rank below bright false positives
    for name in ('fp16', 'int8'):
        a = doc['engines'][name]['accuracy']
        assert a['scores']['AP']['value'] is not None and a['vs_reference']['metrics']['AP_ratio']['ci_low'] is not None
    # the gate: evaluated per engine, never a release decision on the simulated device; energy is not measured here
    g = doc['gate']
    assert g['release_decision'] is False and 'SIMULATED' in g['note']
    for name, v in g['engines'].items():
        assert v['criteria']['energy']['status'] == 'not measured'
        assert v['verdict'] in ('incomplete', 'fail')
    assert g['engines']['fp32']['criteria']['accuracy']['status'] == 'pass'
    html = open(os.path.join(out, 'report.html'), encoding='utf-8').read()
    assert 'SIMULATED DEVICE' in html and 'Release gate' in html and 'fp16' in html
    from falcon_autonomy.core.reports import read_meta
    assert read_meta(os.path.join(out, 'report.html'))['kind'] == 'deploy'


def test_gate_rules(built):
    _, doc = built
    strict = dict(DP.DEFAULT_GATE, accuracy={'metric': 'AP', 'min_relative': 1.01, 'rule': 'ci_lower',
                                             'min_drives': 5},
                  latency={'max_p95_ms': 1e-6}, energy=None)
    res = DP.evaluate_gate(doc, strict)
    for v in res['engines'].values():
        assert v['verdict'] == 'fail' and v['criteria']['latency']['status'] == 'fail'
        assert v['criteria']['energy']['status'] == 'not set'
    loose = dict(DP.DEFAULT_GATE, accuracy={'metric': 'AP', 'min_relative': 0.0, 'rule': 'point'},
                 latency={'max_p95_ms': 1e6}, energy=None)
    assert DP.evaluate_gate(doc, loose)['engines']['fp32']['verdict'] == 'pass'
    assert DP.evaluate_gate(doc, dict(loose, energy={'max_mj_per_frame': 1.0}, missing='fail')
                            )['engines']['fp32']['verdict'] == 'fail'               # unmeasured counts as failing
    few = dict(loose, accuracy={'metric': 'AP', 'min_relative': 0.5, 'min_drives': 10})
    c = DP.evaluate_gate(doc, few)['engines']['fp32']['criteria']['accuracy']
    assert c['status'] == 'not measured' and '10 drives' in c['why']
    floor = dict(loose, accuracy={'metric': 'AP', 'min_relative': 0.5, 'min_reference': 0.99})
    c = DP.evaluate_gate(doc, floor)['engines']['int8']['criteria']['accuracy']
    assert c['status'] == 'not measured' and 'min_reference' in c['why']     # a ratio to a tiny AP says nothing
    with pytest.raises(DP.DeployError):
        DP.evaluate_gate(doc, {'schema': 'x'})


def test_cli_stages_rerun_and_write_the_default_gate(built, tmp_path):
    out, _ = built
    g = str(tmp_path / 'gate.json')
    assert DP.main_deploy(['gate', '--write-default', g]) == P.OK
    gate = json.load(open(g))
    assert gate['schema'] == DP.GATE_SCHEMA and gate['accuracy']['rule'] == 'ci_lower'
    gate['latency']['max_p95_ms'] = 1e6
    json.dump(gate, open(g, 'w'))
    assert DP.main_deploy(['gate', out, '--gate', g]) == P.OK
    doc = DP.read_doc(out)
    assert doc['gate']['gate']['latency']['max_p95_ms'] == 1e6
    assert all(v['criteria']['latency']['status'] == 'pass' for v in doc['gate']['engines'].values())
    assert DP.main_deploy(['report', out, '--out', str(tmp_path / 'r.html')]) == P.OK
    assert DP.main_deploy(['accuracy', out, '--boot', '50']) == P.OK              # from deploy.json alone
    assert DP.main_deploy(['gate', str(tmp_path)]) == P.FAILED                    # not a deploy folder


def test_build_refuses(world, tmp_path):
    m = world['m']
    with pytest.raises(DP.DeployError, match='calibration drives'):
        DP.build(m, '', world['runs'][:1], str(tmp_path / 'a'), precisions=['int8'])
    with pytest.raises(DP.DeployError, match='ONNX detector'):
        DP.build(dict(m, kind='segmenter'), '', world['runs'][:1], str(tmp_path / 'b'))
    with pytest.raises(DP.DeployError, match='not here'):
        DP.build(dict(m, path=str(tmp_path / 'missing.onnx')), '', world['runs'][:1], str(tmp_path / 'c'))


def test_calibration_is_cached_and_shared_drives_are_noted(world, tmp_path):
    out = str(tmp_path / 'c')
    doc = DP.build(world['m'], '', world['runs'][:2], out, calib_runs=world['runs'][1:3], precisions=['int8'],
                   calib_count=4, threads=1)
    assert not doc['calibration']['reused'] and any('also evaluation drives' in n for n in doc['notes'])
    ev = {f['image_id'] for f in json.load(open(os.path.join(out, 'bundle', 'eval', 'frames.json')))['frames']}
    cal = {f['image_id'] for f in json.load(open(os.path.join(out, 'bundle', 'calib', 'frames.json')))['frames']}
    assert cal and not (ev & cal)                                     # never an evaluation frame
    doc2 = DP.build(world['m'], '', world['runs'][:2], out, calib_runs=world['runs'][1:3], precisions=['int8'],
                    calib_count=4, threads=1)
    assert doc2['calibration']['reused'] and doc2['calibration']['cache'] == doc['calibration']['cache']


# ============================================================================================ over SSH
@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """ssh and scp stand-ins that run "on the device" here (HOME = a scratch folder); scp copies both ways."""
    home, bin_ = tmp_path / 'device-home', tmp_path / 'bin'
    home.mkdir()
    bin_.mkdir()
    log = tmp_path / 'ssh.log'
    (bin_ / 'ssh').write_text(f'#!/bin/sh\necho "ssh $*" >> {log}\nfor last; do :; done\n'
                              'export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"\n'   # this Python's packages
                              f'HOME={home} exec bash -c "$last"\n')
    (bin_ / 'scp').write_text(
        f'#!/bin/bash\necho "scp $*" >> {log}\nargs=("$@")\nn=${{#args[@]}}\nsrc="${{args[$((n-2))]}}"\n'
        f'dst="${{args[$((n-1))]}}"\nfix() {{ case "$1" in *@*:*) p="${{1#*:}}"; case "$p" in /*) echo "$p";; '
        f'*) echo "{home}/$p";; esac;; *) echo "$1";; esac; }}\ncp -r "$(fix "$src")" "$(fix "$dst")"\n')
    for f in ('ssh', 'scp'):
        os.chmod(bin_ / f, stat.S_IRWXU)
    monkeypatch.setenv('PATH', f'{bin_}{os.pathsep}{os.environ["PATH"]}')
    key = tmp_path / 'id_ed25519'
    key.write_text('not a real key')
    from falcon_autonomy.core import hil as H
    dev = dict(H.DEVICE_DEFAULTS, address='192.0.2.7', ssh_user='nvidia', ssh_key=str(key), python=sys.executable)
    return dev, home, log


def test_remote_path_through_ssh_keeps_the_address_out(world, tmp_path, fake_ssh, monkeypatch):
    """The Orin path with ssh standing in (and onnxruntime, since this machine has no TensorRT): files go over scp,
    the tool runs over ssh, results come back; neither the address nor the user is in any file written."""
    dev, home, log = fake_ssh
    from falcon_autonomy.core import hil as H
    from falcon_autonomy.core.settings import Settings
    st = Settings(str(tmp_path / 'settings.json'))
    st.set('hil_device', {k: dev[k] for k in H.DEVICE_DEFAULTS})
    out = str(tmp_path / 'remote')
    doc = DP.build(world['m'], '', world['runs'][:1], out, device='orin', precisions=['fp32'], settings=st,
                   backend='onnxruntime', threads=1)
    assert doc['device']['mode'] == 'orin' and doc['device']['where'] == 'from the user\'s settings'
    assert doc['simulated']                                           # onnxruntime: labelled whatever the transport
    assert (home / 'falcon-hil' / 'deploy' / 'remote' / 'engines' / 'fp32' / 'build.json').is_file()
    doc = DP.profile(out, runs=20, warmup_ms=10, min_seconds=0.05, idle_s=0, settings=st)
    assert doc['engines']['fp32']['profile']['latency_ms']['n'] >= 20
    doc = DP.accuracy(out, world['m'], '', n_boot=50, settings=st)
    assert doc['engines']['fp32']['accuracy']['agreement']['identical_detections'] == 6
    cmds = log.read_text()
    assert 'BatchMode=yes' in cmds and 'scp' in cmds
    with pytest.raises(DP.DeployError, match='address'):                # hil.deploy's errors, as this module's
        DP.SshDevice(dict(dev, address=''), 'x')
    for d, _, files in os.walk(out):
        if 'simdevice' in d or 'bundle' in d:
            continue
        for f in files:
            if f.endswith(('.json', '.jsonl', '.log', '.html')):
                text = open(os.path.join(d, f), encoding='utf-8', errors='replace').read()
                assert '192.0.2.7' not in text and 'nvidia@' not in text, f


# ============================================================================================ TensorRT's formats
def test_trtexec_layer_info_and_build_log_parsers(tmp_path):
    """On hand-written samples of trtexec's documented --exportLayerInfo and log formats (not real Orin output)."""
    sample = {'Layers': [
        {'Name': 'conv1', 'LayerType': 'CaskConvolution',
         'Outputs': [{'Name': 'a', 'Format/Datatype': 'Channel major FP16 format where channel % 8 == 0'}]},
        {'Name': 'conv2', 'LayerType': 'CaskConvolution', 'Outputs': [{'Name': 'b', 'Format/Datatype':
                                                                         'Row major linear Int8'}]},
        {'Name': '{ForeignNode[conv3...relu3]}', 'LayerType': 'DLA', 'Outputs': [{'Name': 'c', 'Format/Datatype':
                                                                                    'Row major linear FP16'}]},
        {'Name': 'head', 'LayerType': 'Reformat', 'Outputs': [{'Name': 'd', 'Format/Datatype':
                                                                'Row major linear FP32'}]}]}
    p = tmp_path / 'layers.json'
    p.write_text(json.dumps(sample))
    L = T.parse_layers(str(p))
    assert L['by_output_precision'] == {'FP16': 2, 'INT8': 1, 'FP32': 1} and len(L['dla_subgraphs']) == 1
    log = ('&&&& RUNNING TensorRT.trtexec [TensorRT v100300] # trtexec --onnx=model.onnx\n'
           "[W] [TRT] Layer 'Resize_12' (RESIZE): Unsupported on DLA. Switching this layer's device type to GPU.\n"
           '[I] [TRT] Total Host Persistent Memory: 123456\n[I] Engine built in 41.5 sec.\n')
    b = T.parse_build_log(log)
    assert b['tensorrt'] == '10.3.0' and b['engine_built_s'] == 41.5 and b['warnings'] == 1
    assert b['dla_fallback'][0]['layer'] == 'Resize_12' and b['memory_lines']
    assert T.parse_layers(str(tmp_path / 'missing.json')) == {'available': False}


def test_ina3221_sampler_and_energy_on_a_hand_made_sysfs(tmp_path):
    """The power path on a hand-made hwmon tree (constant values written here; L4T r36's file layout): rails, NC
    channels skipped, V × I, the sampling thread, and the profile summary's estimated energy."""
    hw = tmp_path / 'bus' / 'i2c' / 'drivers' / 'ina3221' / '1-0040' / 'hwmon' / 'hwmon3'
    hw.mkdir(parents=True)
    for k, (name, mv, ma) in enumerate((('VDD_GPU_SOC', 5000, 1200), ('VDD_CPU_CV', 5000, 400), ('NC', 0, 0)), 1):
        (hw / f'in{k}_label').write_text(name + '\n')
        (hw / f'in{k}_input').write_text(f'{mv}\n')
        (hw / f'curr{k}_input').write_text(f'{ma}\n')
    (hw / 'update_interval').write_text('8\n')
    pw = T.Power(str(tmp_path), interval_ms=5)
    assert [r['label'] for r in pw.rails] == ['VDD_GPU_SOC', 'VDD_CPU_CV'] and pw.update_ms == 8
    assert pw.read() == {'VDD_GPU_SOC': 6000.0, 'VDD_CPU_CV': 2000.0}
    pw.start()
    import time
    time.sleep(0.06)
    samples = pw.stop()
    assert len(samples) >= 3
    raw = {'latency_ms': [2.0] * 10, 'samples': samples, 'window_ns': [samples[0]['t_ns'], samples[-1]['t_ns']],
           'power_source': pw.source, 'idle_samples': [{'t_ns': 0, 'mw': {'VDD_GPU_SOC': 1000.0,
                                                                        'VDD_CPU_CV': 1000.0}}], 'wall_s': 0.02}
    s = DP.summarise_profile(raw)
    assert s['power']['run']['sum_mw'] == pytest.approx(8000.0)
    assert s['power']['energy_mj_per_inference'] == pytest.approx(16.0)          # 8 W × 2 ms
    assert s['power']['dynamic_energy_mj_per_inference'] == pytest.approx(12.0)  # (8 − 2) W × 2 ms
    assert 'ESTIMATED' in s['power']['energy_definition']
    assert T.Power(str(tmp_path / 'nothing')).available is False


def test_deploy_tab_lists_engines(window, qapp, built, monkeypatch):
    out, _ = built
    page = window.go('models')
    tab = page.deploy
    monkeypatch.setattr(DP, 'list_deployments', lambda ws: [out])
    tab.load_results()
    assert tab.sel.count() == 1
    assert tab.table.topLevelItemCount() == 3
    assert 'SIMULATED' in tab.info.vals['Device'].text()
    texts = [tab.table.topLevelItem(i).text(0) for i in range(3)]
    assert texts == ['fp32', 'fp16', 'int8']
    assert tab.table.topLevelItem(0).text(4) == 'not measured'                  # power: no rails here
    tab.table.setCurrentItem(tab.table.topLevelItem(2))
    assert 'gate · energy: not measured' in tab.detail.text()
    assert not tab.grab().isNull()
    tab.run_all()                                       # the test project has no ONNX detector: nothing submitted
    assert 'Choose an ONNX detector' in tab.state.vals['Job'].text()
    tab.check()                                         # the simulated device's check, through the job queue
    job = window.queue.jobs[-1]
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 60), job.log
    assert job.state == 'done', job.log
    assert pump(qapp, lambda: job.id not in tab.jobs, 5)



# ============================================================================================ JetPack's Python (opt-in)
def _humble_image() -> bool:
    import shutil
    import subprocess
    if os.environ.get('FALCON_ROS2') != '1' or shutil.which('docker') is None:
        return False
    return subprocess.run(['docker', 'image', 'inspect', 'ros:humble-ros-base'], capture_output=True,
                          timeout=30).returncode == 0


@pytest.mark.skipif(not _humble_image(), reason='needs FALCON_ROS2=1, Docker and the official ros:humble-ros-base '
                                                'image (Ubuntu 22.04: Python 3.10 and numpy 1.21, as on JetPack 6)')
def test_device_side_code_under_jetpack6_python(built, tmp_path):
    """The tool's check and the copy of core.models (normalise, decode_detections) under Python 3.10 and numpy 1.21:
    the same input tensors bit for bit and the same detections as the host's, from the host's raw outputs (no engine
    runs there: the container has neither TensorRT nor onnxruntime)."""
    import shutil
    import subprocess
    import onnxruntime as ort
    out, _ = built
    d = tmp_path / 'dev'
    shutil.copytree(os.path.join(out, 'bundle'), d)
    spec = json.load(open(d / 'spec.json'))
    idx = json.load(open(d / 'eval' / 'frames.json'))['frames'][:8]
    sess = ort.InferenceSession(str(d / 'model.onnx'), providers=['CPUExecutionProvider'])
    exp = []
    for f in idx:
        x = MD.normalise(np.load(d / 'eval' / f['file'])['x'], spec['input'])
        np.save(d / f'x_{f["i"]}.npy', x)
        outs = sess.run(None, {spec['input_name']: x})
        np.save(d / f'o_{f["i"]}.npy', outs[0])
        exp.append(dict(f, det=MD.decode_detections(outs, spec['model'], f['back'])))
    json.dump(exp, open(d / 'expected.json', 'w'))
    script = (
        'import sys, json, numpy as np\n'
        'sys.path.insert(0, "/d")\n'
        'import engine_device as T, falcon_models as FM\n'
        'env = T.environment()\n'
        'spec = json.load(open("/d/spec.json"))\n'
        'for f in json.load(open("/d/expected.json")):\n'
        '    x = FM.normalise(T.load_frame("/d/eval/" + f["file"]), spec["input"])\n'
        '    assert x.tobytes() == np.load("/d/x_%d.npy" % f["i"]).tobytes()\n'
        '    dets = FM.decode_detections([np.load("/d/o_%d.npy" % f["i"])], spec["model"], f["back"])\n'
        '    assert dets == f["det"], (dets, f["det"])\n'
        'print("ok", env["python"], env["numpy"], env["power"]["source"], env["trtexec"])\n')
    (d / 'check.py').write_text(script)
    r = subprocess.run(['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}', '-e',
                        'PYTHONDONTWRITEBYTECODE=1', '-v', f'{d}:/d', 'ros:humble-ros-base', 'bash', '-c',
                        'python3 /d/check.py && python3 /d/engine_device.py check'], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    first = r.stdout.splitlines()[0].split()
    assert first[0] == 'ok' and first[1].startswith('3.10') and first[2].startswith('1.21')
    env = json.loads(r.stdout.splitlines()[-1])
    assert env['tensorrt_python'] is None and env['power']['source'] is None

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401
