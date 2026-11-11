"""The low_vram profile's caps for the dashcam near-miss perception (core.profiles.cap_perception; core.dashcam_ingest,
core.dashcam_detect, core.dashcam_depth): batches of 2, rtdetr-v2-r18 in place of r50, Depth Anything V2 at 364 px,
0.35 of the GPU, FP16 unless inference_precision is fp32 — applied by an ingest, passed to the model runners and
recorded in ingest.json and perception.json; the settings key that decides re-perception. The runners are fakes or
built from their parts (a real DPT image processor, a fake network and a fake onnxruntime): no model is downloaded
and no GPU is needed."""
import json
import os
import shutil
import sys
import types

import numpy as np
import pytest

from falcon_autonomy.core import dashcam_depth as DD
from falcon_autonomy.core import dashcam_detect as DT
from falcon_autonomy.core import dashcam_ingest as I
from falcon_autonomy.core import profiles as PR
from falcon_autonomy.core.settings import Settings


def _settings(folder, profile='standard', precision=None):
    os.makedirs(folder, exist_ok=True)
    st = Settings(os.path.join(folder, 'settings.json'))
    if profile != 'standard':
        PR.set_profile(st, profile, save=False)
    if precision:
        st.set('inference_precision', precision, save=False)
    return st


def test_cap_perception_standard_and_low_vram(tmp_path):
    std = _settings(str(tmp_path / 'std'))
    out = PR.cap_perception({'batch': 8, 'detector': 'rtdetr-v2-r50'}, std)
    assert out['batch'] == 8 and out['detector'] == 'rtdetr-v2-r50' and out['precision'] == 'fp16'
    assert out['gpu_memory_fraction'] == 0.5 and out.get('depth_input_px') is None
    assert out['profile']['name'] == 'standard' and out['profile']['caps'] == {} and out['profile']['changed'] == {}
    low = _settings(str(tmp_path / 'low'), 'low_vram')
    assert low.get('inference_precision') == 'auto'
    out = PR.cap_perception({'batch': 8, 'detector': 'rtdetr-v2-r50', 'gpu_memory_fraction': 0.5}, low)
    assert (out['batch'], out['detector'], out['depth_input_px'], out['gpu_memory_fraction'], out['precision']) == \
        (2, 'rtdetr-v2-r18', 364, 0.35, 'fp16')
    p = out['profile']
    assert p['name'] == 'low_vram' and p['caps'] == PR.PERCEPTION_CAPS and not p['kept_detector']
    assert p['changed'] == {'batch': [8, 2], 'detector': ['rtdetr-v2-r50', 'rtdetr-v2-r18'],
                            'depth_input_px': [None, 364], 'gpu_memory_fraction': [0.5, 0.35],
                            'precision': [None, 'fp16']}
    assert 'auto' in p['precision']['why']
    # smaller asks stay; r50 kept on purpose; FP32 when inference_precision says so
    out = PR.cap_perception({'batch': 1, 'detector': 'rtdetr-v2-r50', 'depth_input_px': 280,
                             'gpu_memory_fraction': 0.2}, low, keep_detector=True)
    assert (out['batch'], out['detector'], out['depth_input_px'], out['gpu_memory_fraction']) == \
        (1, 'rtdetr-v2-r50', 280, 0.2)
    assert out['profile']['kept_detector'] and 'detector' not in out['profile']['changed']
    low.set('inference_precision', 'fp32', save=False)
    out = PR.cap_perception({}, low)
    assert out['precision'] == 'fp32' and out['batch'] == 2 and I.normalise(out)['detector'] == 'rtdetr-v2-r18'
    assert PR.effective(low)['perception'] == PR.PERCEPTION_CAPS and PR.effective(std)['perception'] == {}


def test_the_settings_key_ignores_what_does_not_change_the_perception():
    base = I.normalise({})
    k = I.settings_key(base)
    same = dict(base, precision='fp32', gpu_memory_fraction=0.1, keep_detector=True, batch=2,
                profile={'name': 'low_vram'})
    assert I.settings_key(same) == k
    # an ingest made before these settings existed keeps its key (depth_input_px None is left out)
    old = {kk: v for kk, v in base.items() if kk not in I.VOLATILE and kk != 'depth_input_px'}
    from falcon_autonomy.core.provenance import sha256_json
    assert sha256_json(old) == k
    assert I.settings_key(dict(base, depth_input_px=364)) != k
    assert I.settings_key(dict(base, detector='rtdetr-v2-r50')) != k


class _Rec:
    """Records what the factories built (the fakes below stand for the real runners)."""
    made = []


class FakeRTDetr:
    is_model = True

    def __init__(self, name, models_dir='', device='auto', fp16=True, threshold=0.1, gpu_memory_fraction=0.5):
        _Rec.made.append(('detector', name, fp16, gpu_memory_fraction))
        self.inner = DT.FileDetector(os.environ['FALCON_TEST_SIDECAR'])
        self.info = {'detector': name, 'fake': True}

    def __call__(self, frames):
        _Rec.made.append(('batch', len(frames)))
        return self.inner(frames)


class FakeDA2:
    kind = 'relative'

    def __init__(self, name='da2-small', models_dir='', device='auto', fp16=True, gpu_memory_fraction=0.5,
                 input_px=None):
        _Rec.made.append(('depth', name, fp16, gpu_memory_fraction, input_px))
        self.info = {'model': name, 'fake': True, 'input_px': input_px}

    def __call__(self, frames):
        return [_DEPTHS[int(n)] for n, _ in frames]


_DEPTHS = {}


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('needs ffmpeg')
    pytest.importorskip('cv2')
    pytest.importorskip('scipy')
    from test_dashcam_nearmiss import IST, UTC0, make_clip
    root = tmp_path_factory.mktemp('perc-profile')
    folder = str(root / 'footage')
    path, side, depths = make_clip(os.path.join(folder, 'Movie_F'), UTC0.astimezone(IST))
    return folder, path, side, depths


def test_an_ingest_under_low_vram_uses_and_records_the_caps(clip, tmp_path, monkeypatch):
    from test_dashcam_nearmiss import SETTINGS
    folder, path, side, depths = clip
    _DEPTHS.clear()
    _DEPTHS.update(depths)
    monkeypatch.setenv('FALCON_TEST_SIDECAR', side)
    monkeypatch.setattr(DT, 'RTDetrV2', FakeRTDetr)
    monkeypatch.setattr(DD, 'DepthAnythingV2', FakeDA2)
    _Rec.made = []
    low = _settings(str(tmp_path / 'low'), 'low_vram')
    s = dict(SETTINGS, detector='rtdetr-v2-r50', depth='da2-small', batch=8, cut_clips=False)
    res = I.ingest(folder, s, app_settings=low, redo=True)
    assert res['counts']['perceived'] == 1 and res['events'] >= 1
    made = [m for m in _Rec.made if m[0] != 'batch']
    assert made == [('detector', 'rtdetr-v2-r18', True, 0.35), ('depth', 'da2-small', True, 0.35, 364)]
    assert max(m[1] for m in _Rec.made if m[0] == 'batch') == 2
    od = I.out_dir(folder)
    st = json.load(open(os.path.join(od, 'ingest.json')))
    perc = json.load(open(os.path.join(od, 'clips', I.clip_key(path), 'perception.json')))
    for prof in (st['settings']['profile'], perc['profile']):
        assert prof['name'] == 'low_vram' and prof['changed']['detector'] == ['rtdetr-v2-r50', 'rtdetr-v2-r18']
        assert prof['changed']['batch'] == [8, 2] and prof['precision']['used'] == 'fp16'
    assert perc['batch'] == 2 and st['settings']['depth_input_px'] == 364 and st['settings']['detector'] == \
        'rtdetr-v2-r18'
    # the same profile again: nothing perceived again; standard: perceived again (another detector and depth size)
    _Rec.made = []
    assert I.ingest(folder, s, app_settings=low)['counts']['reused'] == 1 and not _Rec.made
    std = _settings(str(tmp_path / 'std'))
    res = I.ingest(folder, s, app_settings=std)
    assert res['counts']['perceived'] == 1
    made = [m for m in _Rec.made if m[0] != 'batch']
    assert made == [('detector', 'rtdetr-v2-r50', True, 0.5), ('depth', 'da2-small', True, 0.5, None)]
    assert max(m[1] for m in _Rec.made if m[0] == 'batch') == 8
    # inference_precision fp32 under low_vram: the models in FP32
    _Rec.made = []
    low.set('inference_precision', 'fp32', save=False)
    I.ingest(folder, s, app_settings=low)
    assert [m for m in _Rec.made if m[0] != 'batch'] == [('detector', 'rtdetr-v2-r18', False, 0.35),
                                                          ('depth', 'da2-small', False, 0.35, 364)]
    perc = json.load(open(os.path.join(od, 'clips', I.clip_key(path), 'perception.json')))
    assert perc['profile']['precision']['used'] == 'fp32'


def test_depth_anything_works_at_the_capped_resolution():
    """The processor's resize target is the working resolution (a real DPT image processor, a fake network that
    returns a depth map of the input's size): 364 px for low_vram, the processor's own 518 otherwise; the map comes
    back at the frame's size either way."""
    torch = pytest.importorskip('torch')
    tr = pytest.importorskip('transformers')
    proc = tr.DPTImageProcessor(size={'height': 518, 'width': 518}, keep_aspect_ratio=True, ensure_multiple_of=14,
                                do_resize=True, resample=3, do_rescale=True, do_normalize=True)
    seen = []

    class Net(torch.nn.Module):
        def forward(self, pixel_values):
            seen.append(tuple(pixel_values.shape))
            return types.SimpleNamespace(predicted_depth=pixel_values.mean(1))

    frames = [(0, np.zeros((540, 960, 3), np.uint8)), (1, np.full((540, 960, 3), 90, np.uint8))]
    for px, want_h in ((364, 364), (None, 518)):
        da = DD.DepthAnythingV2.__new__(DD.DepthAnythingV2)
        da.processor, da.model, da.input_px = proc, Net(), px
        da.device, da.dtype = torch.device('cpu'), torch.float32
        maps = da(frames)
        assert seen[-1][0] == 2 and seen[-1][2] == want_h and seen[-1][3] % 14 == 0
        assert maps[0].shape == (540, 960)
    assert seen[0][2] * seen[0][3] < 0.55 * seen[1][2] * seen[1][3]       # about half the pixels


def test_metric3d_gets_the_profiles_share_of_the_gpu(tmp_path, monkeypatch):
    """onnxruntime's CUDA arena limited to the profile's share (low_vram), unlimited otherwise (a fake
    onnxruntime: no model, no GPU)."""
    from falcon_autonomy.core import perception_models as PM
    calls = []

    class Sess:
        def __init__(self, path, providers=None):
            calls.append(providers)

        def get_providers(self):
            return ['CUDAExecutionProvider']

        def get_inputs(self):
            return [types.SimpleNamespace(name='x')]
    fake = types.SimpleNamespace(InferenceSession=Sess,
                                 get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    monkeypatch.setitem(sys.modules, 'onnxruntime', fake)
    monkeypatch.setattr(PM, 'load_folder', lambda name, models_dir='', accept=False: str(tmp_path))
    monkeypatch.setattr(PM, 'refused_reason', lambda name: None)
    monkeypatch.setattr(DD, 'cuda_arena_limit', lambda f: int(f * 16e9) if f else None)
    m = DD.make_depth('metric3d-small', 500.0, device='cuda', accept_unstated=True, gpu_memory_fraction=0.35,
                      limit_onnx=True)
    assert calls[-1][0] == ('CUDAExecutionProvider', {'gpu_mem_limit': int(0.35 * 16e9),
                                                      'arena_extend_strategy': 'kSameAsRequested'})
    assert m.info['gpu_mem_limit_bytes'] == int(0.35 * 16e9) and 'fixed input' in m.info['input']
    DD.make_depth('metric3d-small', 500.0, device='cuda', accept_unstated=True, gpu_memory_fraction=0.5)
    assert calls[-1] == ['CUDAExecutionProvider', 'CPUExecutionProvider']


def test_the_runners_take_the_profiles_precision_and_share(monkeypatch):
    made = []
    monkeypatch.setattr(DT, 'RTDetrV2', lambda *a, **k: made.append(('det', a, k)) or 'det')
    monkeypatch.setattr(DD, 'DepthAnythingV2', lambda *a, **k: made.append(('depth', a, k)) or 'depth')
    assert DT.make_detector('rtdetr-v2-r18', '', 'cpu', False, 0.35) == 'det'
    assert made[-1] == ('det', ('rtdetr-v2-r18', '', 'cpu', False), {'gpu_memory_fraction': 0.35})
    assert DD.make_depth('da2-small', 500.0, '', 'cpu', False, False, 364, 0.35) == 'depth'
    assert made[-1] == ('depth', ('da2-small', '', 'cpu', False, 0.35, 364), {})
    with pytest.raises(DT.DetectError, match='AGPL'):
        DT.make_detector('yolov8n')


def test_the_command_takes_keep_detector(capsys):
    assert I.main_ingest(['ingest', '--help']) == 0
    text = ' '.join(capsys.readouterr().out.split())
    assert '--keep-detector' in text and 'caps it at 2' in text
