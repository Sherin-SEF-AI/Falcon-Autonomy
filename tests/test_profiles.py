"""The low-VRAM profile (core.profiles): switching profiles writes and restores settings, what the server is started
with, reconstruction caps, the doctor's suggestion, the VRAM sampler and `falcon profile`, the Performance section —
and FP16 inference (core.models): the cached ONNX conversion, TorchScript in half precision, and the AP-drop gate
deciding 'auto' on a drive recorded against the fake carla."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from falcon_autonomy.core import profiles as PR
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import sysinfo
from falcon_autonomy.core.settings import Settings


def _cli(*a, config=None, path_prepend=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    if config:
        env['XDG_CONFIG_HOME'] = str(config)
    if path_prepend:
        env['PATH'] = str(path_prepend) + os.pathsep + env.get('PATH', '')
    return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *a], capture_output=True, text=True, env=env,
                          timeout=300)


def _fake_smi(tmp_path, used_mb=(1000, 6000, 3000), total_mb=8192):
    """An nvidia-smi on PATH answering --query-gpu with a used figure that walks through used_mb, one per call."""
    d = tmp_path / 'bin'
    d.mkdir()
    state = tmp_path / 'smi-calls'
    seq = ' '.join(str(int(u)) for u in used_mb)
    (d / 'nvidia-smi').write_text(
        '#!/bin/sh\n'
        f'n=$(cat "{state}" 2>/dev/null || echo 0); echo $((n + 1)) > "{state}"\n'
        f'set -- {seq}; i=0; v=$1\n'
        'for x in "$@"; do if [ $i -le $n ]; then v=$x; fi; i=$((i + 1)); done\n'
        f'echo "0, Test GPU, {total_mb}, $v, 10, 999.0"\n')
    (d / 'nvidia-smi').chmod(0o755)
    return d


# ------------------------------------------------------------------------------------------------- profiles
def test_low_vram_writes_its_values_and_standard_restores_yours(tmp_path):
    st = Settings(str(tmp_path / 'settings.json'))
    st.set('render_budget_mpx', 150.0)
    st.set('server_quality', 'Epic')
    assert PR.current(st) == 'standard' and PR.effective(st)['differs'] == {}
    changed = PR.set_profile(st, 'low_vram')
    assert changed['render_budget_mpx'] == (150.0, 96.0) and changed['server_quality'] == ('Epic', 'Low')
    st2 = Settings(str(tmp_path / 'settings.json'))                 # saved
    assert PR.current(st2) == 'low_vram' and st2.get('texture_pool_mb') == 512
    assert st2.get('server_memory_gb') == 10.0 and st2.get('inference_precision') == 'auto'
    eff = PR.effective(st2)
    assert eff['differs'] == {} and eff['recon']['max_gaussians'] == 300_000
    assert eff['server_args'] == ['-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=512',
                                  '-ExecCmds=r.Streaming.PoolSize 512']
    st2.set('render_budget_mpx', 120.0)                               # changed by hand: stays, and is shown
    assert PR.effective(st2)['differs'] == {'render_budget_mpx': {'profile': 96.0, 'setting': 120.0}}
    PR.set_profile(st2, 'low_vram')                                   # again: the kept values are not overwritten
    PR.set_profile(st2, 'standard')
    st3 = Settings(str(tmp_path / 'settings.json'))
    assert (st3.get('render_budget_mpx'), st3.get('server_quality'), st3.get('texture_pool_mb')) == (150.0, 'Epic', 0)
    assert PR.current(st3) == 'standard' and not os.path.exists(PR.saved_path(st3))
    with pytest.raises(PR.ProfileError):
        PR.set_profile(st3, 'turbo')


def test_server_and_recon_take_the_profile(tmp_path):
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.sim import server as S
    from falcon_autonomy.sim.service import CarlaService
    st = Settings(str(tmp_path / 'settings.json'))
    PR.set_profile(st, 'low_vram')
    root = tmp_path / 'CARLA_0.9.16'
    (root / os.path.dirname(S.BINARY)).mkdir(parents=True)
    (root / S.BINARY).write_text('')
    st.set('carla_root', str(root))
    svc = CarlaService(st, JobQueue(SubprocessLauncher(), str(tmp_path / 'locks')))
    try:
        cmd = svc.ensure_server_object().command()
        assert '-quality-level=Low' in cmd and 'MemoryMax=10240M' in cmd
        assert '-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=512' in cmd
    finally:
        svc.shutdown()
    assert PR.cap_recon({'max_gaussians': 1_000_000, 'width': 320}, st) == {
        'max_gaussians': 300_000, 'width': 320, 'max_init_points': 150_000, 'gpu_memory_fraction': 0.5}
    PR.set_profile(st, 'standard')
    assert PR.cap_recon({'max_gaussians': 1_000_000}, st) == {'max_gaussians': 1_000_000}


def test_doctor_suggests_low_vram_on_a_small_gpu(monkeypatch):
    g8 = [{'index': 0, 'name': 'Small', 'memory_total_mb': 8192.0, 'memory_used_mb': 300.0, 'driver': '999'}]
    g16 = [{'index': 0, 'name': 'Big', 'memory_total_mb': 16303.0, 'memory_used_mb': 300.0}]
    ok, why = PR.suggestion(g8, 'standard')
    assert ok and 'falcon profile set low_vram' in why and '8.0 GB' in why
    assert 'suggested' not in PR.suggestion(g8, 'low_vram')[1]
    assert 'suggested' not in PR.suggestion(g16, 'standard')[1]
    assert 'no NVIDIA GPU' in PR.suggestion([], 'standard')[1]
    monkeypatch.setattr(sysinfo, 'gpus', lambda timeout=5.0: g8)
    row = next(r for r in sysinfo.checks() if r[0] == 'Performance profile')
    assert row[1] and 'low_vram' in row[2]


def test_vram_sampler_and_measure(tmp_path):
    seq = iter([1000.0, 2000.0, 5000.0, 4000.0] + [3000.0] * 1000)
    q = lambda: [{'index': 0, 'name': 'Fake', 'memory_total_mb': 8192.0, 'memory_used_mb': next(seq)}]  # noqa: E731
    s = sysinfo.VramSampler(0.05, query=q)
    res = PR.measure([sys.executable, '-c', 'import time; time.sleep(0.4)'], interval=0.05, sampler=s)
    assert res['baseline_mb'] == 1000.0 and res['peak_mb'] == 5000.0 and res['peak_delta_mb'] == 4000.0
    assert res['exit_code'] == 0 and res['gpu'] == 'Fake' and res['samples'] >= 4
    with pytest.raises(PR.ProfileError):
        PR.measure(None, 0.0, sampler=sysinfo.VramSampler(0.05, query=lambda: []))
    smi = _fake_smi(tmp_path)
    env = dict(os.environ, PATH=str(smi) + os.pathsep + os.environ['PATH'])
    out = subprocess.run([sys.executable, '-c', 'from falcon_autonomy.core import sysinfo; import json; '
                          'print(json.dumps(sysinfo.gpus()))'], capture_output=True, text=True, env=env)
    g = json.loads(out.stdout)[0]
    assert g['memory_total_mb'] == 8192.0 and g['memory_used_mb'] == 1000.0 and g['memory_free_mb'] == 7192.0


def test_profile_command(tmp_path):
    cfg = tmp_path / 'cfg'
    r = _cli('profile', 'show', config=cfg)
    assert r.returncode == 0 and r.stdout.startswith('standard:')
    r = _cli('profile', 'set', 'low_vram', config=cfg)
    assert r.returncode == 0 and 'render_budget_mpx 200.0 → 96.0' in r.stdout
    r = _cli('profile', 'show', '--json', config=cfg)
    eff = json.loads(r.stdout)
    assert eff['profile'] == 'low_vram' and eff['values']['texture_pool_mb'] == 512
    r = _cli('doctor', config=cfg)
    assert 'Performance profile' in r.stdout
    smi = _fake_smi(tmp_path, used_mb=(1000, 6000, 3000, 3000, 3000))
    r = _cli('profile', 'measure', '--interval', '0.05', '--json', '--', sys.executable, '-c',
             'import time; time.sleep(0.5)', config=cfg, path_prepend=smi)
    res = json.loads(r.stdout)
    assert r.returncode == 0 and res['peak_mb'] == 6000.0 and res['within'] and res['gpu'] == 'Test GPU'
    (tmp_path / 'smi-calls').unlink()
    r = _cli('profile', 'measure', '--interval', '0.05', '--limit-gb', '4', '--seconds', '0.3', config=cfg,
             path_prepend=smi)
    assert r.returncode == P.FAILED and 'over 4 GB' in r.stdout
    r = _cli('profile', 'set', 'standard', config=cfg)
    assert r.returncode == 0 and 'render_budget_mpx 96.0 → 200.0' in r.stdout
    assert _cli('profile', 'measure', '--help', config=cfg).returncode == 0


# ------------------------------------------------------------------------------------------------- the GUI
@pytest.fixture(scope='module')
def qapp():
    from falcon_autonomy.ui.app import make_app
    from PyQt6.QtCore import QThreadPool
    app = make_app()
    yield app
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()


@pytest.mark.gui
def test_settings_performance_section(qapp, tmp_path):
    from types import SimpleNamespace
    from falcon_autonomy.ui import settings_profile as SPR
    st = Settings(str(tmp_path / 'settings.json'))
    quality = SimpleNamespace(v='Epic', setCurrentText=lambda t: setattr(quality, 'v', t))
    page = SimpleNamespace(win=SimpleNamespace(settings=st), quality=quality)
    sec = SPR.section(page)
    assert sec.title == 'Performance' and sec.profile.currentData() == 'standard'
    sec.profile.setCurrentIndex(sec.profile.findData('low_vram'))
    assert sec.precision.currentData() == 'auto' and sec.pool.value() == 512
    sec.pool.setValue(768)
    assert sec.apply(st) == ''
    assert PR.current(st) == 'low_vram' and st.get('texture_pool_mb') == 768 and quality.v == 'Low'
    assert 'texture_pool_mb 768' in sec.state.text()
    st.save()
    sec.profile.setCurrentIndex(sec.profile.findData('standard'))
    assert sec.apply(st) == '' and st.get('server_quality') == 'Epic' and PR.current(st) == 'standard'
    sec.deleteLater()


# ------------------------------------------------------------------------------------------------- FP16
onnx = pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')


def _det_model(path, boxes, scores, labels, unsafe=False, size=(32, 64)):
    """A detector graph whose outputs are constants (boxes xyxy, scores, labels) plus 0 × the input's maximum, so it
    takes the image. unsafe: its scores are (70000 − 69999) × scores, whose constants FP16 conversion clamps to its
    largest finite value (1e4): 0 in FP16."""
    from onnx import TensorProto, helper, numpy_helper
    h, w = size
    inp = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, h, w])
    outs = [helper.make_tensor_value_info('boxes', TensorProto.FLOAT, None),
            helper.make_tensor_value_info('scores', TensorProto.FLOAT, None),
            helper.make_tensor_value_info('labels', TensorProto.INT64, None)]
    inits = [numpy_helper.from_array(np.asarray(boxes, np.float32).reshape(-1, 4), 'b0'),
             numpy_helper.from_array(np.asarray(scores, np.float32), 's0'),
             numpy_helper.from_array(np.asarray(labels, np.int64), 'l0'),
             numpy_helper.from_array(np.zeros((), np.float32), 'zero'),
             numpy_helper.from_array(np.asarray(70000.0, np.float32), 'k1'),
             numpy_helper.from_array(np.asarray(69999.0, np.float32), 'k2')]
    nodes = [helper.make_node('ReduceMax', ['images'], ['m'], keepdims=0),
             helper.make_node('Mul', ['m', 'zero'], ['z']),
             helper.make_node('Add', ['b0', 'z'], ['boxes']),
             helper.make_node('Identity', ['l0'], ['labels'])]
    if unsafe:
        nodes += [helper.make_node('Sub', ['k1', 'k2'], ['d']), helper.make_node('Add', ['d', 'z'], ['d1']),
                  helper.make_node('Mul', ['d1', 's0'], ['scores'])]
    else:
        nodes += [helper.make_node('Add', ['s0', 'z'], ['scores'])]
    g = helper.make_graph(nodes, 'det', [inp], outs, inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)]), str(path))
    return str(path)


def test_onnx_fp16_is_converted_once_and_cached(tmp_path):
    from falcon_autonomy.core import models as M
    p = _det_model(tmp_path / 'det.onnx', [[1.5, 2.25, 10.75, 20.5]], [0.9], [0])
    m = {'kind': 'detector', 'runtime': 'onnx', 'path': p, 'input': {'width': 64, 'height': 32},
         'output': {'format': 'boxes_scores_labels', 'score_threshold': 0.3}, 'classes': ['car']}
    r32, r16 = M.Runner(m), M.Runner(m, precision='fp16')
    assert r32.precision == 'fp32' and r16.precision == 'fp16' and r16.provenance()['requested'] == 'fp16'
    cached = M.fp16_path(p)
    assert os.path.isfile(cached) and cached.endswith('.fp16.onnx') and r16.weights == cached
    assert os.path.basename(cached).split('.')[0] == __import__('hashlib').sha256(open(p, 'rb').read()).hexdigest()
    t = os.path.getmtime(cached)
    M.Runner(m, precision='fp16')
    assert os.path.getmtime(cached) == t                                    # not converted again
    img = np.zeros((32, 64, 3), np.uint8)
    d32, d16 = r32.predict(img), r16.predict(img)
    assert d32[0]['score'] == pytest.approx(0.9) and d16[0]['score'] == pytest.approx(0.9, abs=1e-3)
    assert np.allclose(d32[0]['bbox'], d16[0]['bbox'], atol=0.01)
    with pytest.raises(ValueError):
        M.Runner(m, precision='auto')
    plug = M.Runner({'kind': 'detector', 'runtime': 'python', 'licence': 'MIT',
                     'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector'}, precision='fp16')
    assert plug.precision == 'native' and 'own precision' in plug.precision_note


def test_torchscript_in_half_precision(tmp_path):
    torch = pytest.importorskip('torch')
    from falcon_autonomy.core import models as M

    class Seg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)

        def forward(self, x):
            return self.conv(x)
    torch.manual_seed(0)
    path = str(tmp_path / 'seg.pt')
    torch.jit.trace(Seg(), torch.zeros(1, 3, 8, 16)).save(path)       # (torch.jit.script: not on Python 3.14)
    m = {'kind': 'segmenter', 'runtime': 'torchscript', 'path': path, 'input': {'width': 16, 'height': 8},
         'output': {'format': 'logits'}, 'classes': ['road', 'car', 'sidewalk', 'sky']}
    x = np.random.default_rng(0).random((1, 3, 8, 16)).astype(np.float32)
    r32, r16 = M.Runner(m), M.Runner(m, precision='fp16')
    assert r16.precision == 'fp16' and next(r16.net.parameters()).dtype == torch.float16
    y32, y16 = r32.raw(x)[0], r16.raw(x)[0]
    assert y16.dtype == np.float32 and np.allclose(y32, y16, atol=5e-3)


@pytest.fixture(scope='module')
def one_frame(tmp_path_factory):
    """One recorded frame with ground truth on the fake (seed 4, ten vehicles) and its cars' boxes."""
    from falcon_autonomy.core.project import Project
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('fp16')
    p = Project.new('Test')
    p.doc['rigs'][0]['sensors'][0].update(width=320, height=180, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.doc['traffic_profiles'][0]['vehicles'] = 10
    p.save(str(base / 'test.falcon.json'))
    code, _, err = _run(_cfg(p, base / 'run', seconds=0.05, seed=4))
    assert code == P.OK, err
    objs = json.load(open(base / 'run' / 'objects' / 'cam-front' / '000000.json'))
    objs = objs['objects'] if isinstance(objs, dict) else objs
    cars = [o['bbox'] for o in objs if o.get('class') == 'car' and o.get('bbox')]
    assert cars, 'the frame has no car to detect'
    return {'run': str(base / 'run'), 'cars': cars, 'base': base}


def test_fp16_gate_keeps_fp16_when_ap_holds_and_falls_back_when_it_drops(one_frame, tmp_path):
    from falcon_autonomy.core import harness as H
    cars = one_frame['cars']

    def model(name, unsafe):
        p = _det_model(tmp_path / f'{name}.onnx', cars, [0.9] * len(cars), [0] * len(cars), unsafe=unsafe,
                       size=(180, 320))
        return {'id': name, 'name': name, 'kind': 'detector', 'runtime': 'onnx', 'path': p, 'licence': 'MIT',
                'input': {'width': 320, 'height': 180}, 'classes': ['car'], 'class_map': {'car': 'car'},
                'output': {'format': 'boxes_scores_labels', 'score_threshold': 0.3, 'nms_iou': 0.99}}
    doc = H.evaluate([one_frame['run']], model('safe', False), str(tmp_path / 'e1'), n_boot=20, precision='auto')
    g = doc['precision']
    assert g['precision'] == 'fp16' and g['metric'] == 'AP' and g['fp32'] > 0.5 and g['drop_pt'] <= 0.5, g
    assert 'collected' not in g and doc['provenance']['settings']['precision'] == 'fp16'
    assert len(open(tmp_path / 'e1' / 'predictions.jsonl').readlines()) == doc['n_frames'] == 1
    doc = H.evaluate([one_frame['run']], model('unsafe', True), str(tmp_path / 'e2'), n_boot=20, precision='auto')
    g = doc['precision']
    assert g['precision'] == 'fp32' and g['fp16'] == 0.0 and g['drop_pt'] > 50 and 'points below' in g['reason'], g
    doc = H.evaluate([one_frame['run']], model('unsafe2', True), str(tmp_path / 'e3'), n_boot=20, precision='fp16')
    assert doc['precision'] == {'precision': 'fp16', 'requested': 'fp16', 'note': None}
    assert H.evaluate([one_frame['run']], model('d', False), str(tmp_path / 'e4'), n_boot=20)['precision'][
        'precision'] == 'fp32'
