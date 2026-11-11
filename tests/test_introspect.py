"""Introspection (core.introspect): attention hooks and their shapes on a random-initialised token policy, rollout,
deterministic masking / head zeroing / activation patching / feature flips, patch rays projected to the world with
depth, object boxes; the hooks on a tiny random Qwen2-VL with eager attention (no download) with and without a LoRA
adapter; HFRunner's attention option; frames files and the CLI on the fake carla."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from falcon_autonomy.core import introspect as IN  # noqa: E402
from falcon_autonomy.core import tokenfeatures as TF  # noqa: E402
from test_tokenpolicy import make_obs, make_policy  # noqa: E402


@pytest.fixture
def pol(tmp_path):
    return make_policy(tmp_path, seed=3)


def test_hook_shapes_and_rollout(pol):
    obs = make_obs()
    pts = IN.token_hook_points(pol.net)
    assert [n for n, _ in pts] == ['layer0', 'layer1', 'layer2', 'layer3', 'plan']
    from falcon_autonomy.core.drivefeatures import FEATURE_NAMES
    with IN.AttentionRecorder(pts) as rec, torch.no_grad():
        pol.net(IN._batch(pol, [obs], FEATURE_NAMES))
    assert all(h.handles == [] for h in [rec])                             # removed on exit
    for i in range(4):
        w = rec.weights[f'layer{i}']
        assert tuple(w.shape) == (1, 8, 58, 58) and torch.allclose(w.sum(-1), torch.ones(1, 8, 58))
        assert float(w[0, :, :, 43:56].abs().max()) == 0.0                 # empty object rows: never attended
    assert tuple(rec.weights['plan'].shape) == (1, 8, 1, 58)
    R = IN.rollout([rec.weights[f'layer{i}'] for i in range(4)])
    assert R.shape == (1, 58, 58) and np.allclose(R.sum(-1), 1.0)
    single = IN.rollout([torch.full((1, 1, 2, 2), 0.5)])
    assert np.allclose(single[0], [[0.75, 0.25], [0.25, 0.75]])             # ½ (A + I)
    d = IN.attention(pol, obs)
    assert sum(r['relevance'] for r in d['tokens']) == pytest.approx(1.0, abs=1e-4)
    assert set(d['by_type']) == {'patch', 'object', 'route', 'ego'} and len(d['plan_heads']) == 8
    objs = [r for r in d['tokens'] if r['type'] == 'object' and r['id'] is not None]
    assert [r['id'] for r in objs] == [12, 11, 13] and objs[1]['brake_light'] and objs[0]['class'] == 'autorickshaw'
    assert all(r['relevance'] == 0 for r in d['tokens'] if r['type'] == 'object' and r['id'] is None)
    assert [b['id'] for b in d['objects']] == [12, 11, 13] and d['objects'][1]['length'] == pytest.approx(4.6)
    assert d['action'] == pytest.approx(pol.act(obs).tolist(), abs=1e-6)


def test_masking_and_head_zeroing_deterministic(pol):
    obs = make_obs()
    base, _ = IN.act_with(pol, obs, [])
    assert np.array_equal(base, pol.act(obs))
    m1, n1 = IN.act_with(pol, obs, [{'op': 'mask_class', 'class': 'autorickshaw'}])
    m2, _ = IN.act_with(pol, obs, [{'op': 'mask_class', 'class': 'autorickshaw'}])
    assert n1 == 1 and np.array_equal(m1, m2) and not np.array_equal(m1, base)
    # masking a token is exactly removing its row (an empty row is masked the same way)
    gone = dict(obs, objects=obs['objects'].copy())
    gone['objects'][0] = 0.0
    assert np.allclose(pol.act(gone), m1, atol=1e-6)
    same, n0 = IN.act_with(pol, obs, [{'op': 'mask_class', 'class': 'bus'}])
    assert n0 == 0 and np.array_equal(same, base)
    byid, _ = IN.act_with(pol, obs, [{'op': 'mask_object', 'id': 12}])
    assert np.array_equal(byid, m1)
    z1, _ = IN.act_with(pol, obs, [{'op': 'zero_head', 'layer': 2, 'head': 5}])
    z2, _ = IN.act_with(pol, obs, [{'op': 'zero_head', 'layer': 2, 'head': 5}])
    assert np.array_equal(z1, z2) and not np.array_equal(z1, base)
    assert np.array_equal(IN.act_with(pol, obs, [])[0], base)              # the hook was removed
    zp, _ = IN.act_with(pol, obs, [{'op': 'zero_head', 'layer': 'plan', 'head': 0}])
    assert not np.array_equal(zp, base)
    pt, _ = IN.act_with(pol, obs, [{'op': 'mask_tokens', 'type': 'patch'}])
    assert not np.array_equal(pt, base)
    with pytest.raises(ValueError, match='head 9'):
        IN.act_with(pol, obs, [{'op': 'zero_head', 'layer': 0, 'head': 9}])
    with pytest.raises(ValueError, match='class must be'):
        IN.intervene(pol, [obs], [{'op': 'mask_class', 'class': 'tram'}])


def test_patching_and_flips(pol):
    obs, other = make_obs(0), make_obs(7)
    base, _ = IN.act_with(pol, obs, [])
    same, _ = IN.act_with(pol, obs, [{'op': 'patch', 'layer': 1, 'tokens': 'all'}], counterfactual=obs)
    assert np.allclose(same, base, atol=1e-6)                              # patching with itself changes nothing
    full, _ = IN.act_with(pol, obs, [{'op': 'patch', 'layer': 3, 'tokens': 'all'}], counterfactual=other)
    assert np.allclose(full, IN.act_with(pol, other, [])[0], atol=1e-5)    # everything after the last layer: other's
    part, _ = IN.act_with(pol, obs, [{'op': 'patch', 'layer': 1, 'tokens': 'patch'}], counterfactual=other)
    assert not np.allclose(part, base)
    flipped = IN.edit_observation(obs, [{'op': 'flip', 'feature': 'brake_light', 'id': 11}])
    assert flipped['objects'][1, TF.OI['brake_light']] == 0.0 and obs['objects'][1, TF.OI['brake_light']] == 1.0
    sh = IN.edit_observation(obs, [{'op': 'speed_history', 'offset': -9.0}])
    assert sh['ego_history'].min() == 0.0
    doc = IN.intervene(pol, [obs, other], [{'op': 'flip', 'feature': 'brake_light', 'all': True}])
    assert len(doc['frames']) == 2 and doc['summary']['frames'] == 2
    assert doc['frames'][0]['delta'] == pytest.approx(np.subtract(doc['frames'][0]['intervened'],
                                                                  doc['frames'][0]['base']).tolist(), abs=1e-6)
    shift = IN.intervene(pol, [obs, other], [{'op': 'patch', 'layer': 3, 'tokens': 'all'}], shift=1)
    assert shift['frames'][0]['intervened'] == pytest.approx(IN.act_with(pol, other, [])[0].tolist(), abs=1e-5)


def test_patch_rays_to_the_world():
    cam = {'width': 128, 'height': 72, 'fov': 90.0, 'pose': {'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0,
                                                              'pitch': 0.0, 'yaw': 0.0}}
    depth = np.full((72, 128), 10.0, np.float32)
    depth[:20] = 1000.0                                                    # the sky: dropped
    pts = IN.patch_points((5, 8), cam, depth)
    assert len(pts) == 32 and all(p['row'] >= 1 for p in pts)
    assert all(p['x'] == pytest.approx(11.5) for p in pts)                 # planar depth 10 m ahead of the camera
    mid = [p for p in pts if p['col'] in (3, 4)]
    assert all(abs(p['y']) == pytest.approx(10.0 * (8.0 / 64.0), abs=1e-3) for p in mid)   # cell centres ±8 px
    left = min(pts, key=lambda p: p['y'])
    assert left['col'] == 0 and left['y'] == pytest.approx(-10.0 * 56.0 / 64.0, abs=1e-3)   # y right: left is −
    low = [p for p in pts if p['row'] == 4]
    assert all(p['z'] < 1.6 for p in low)                                  # below the axis: below the camera
    w = IN.patch_points((5, 8), cam, depth, {'x': 100.0, 'y': 50.0, 'z': 0.0, 'yaw': 90.0},
                        relevance=np.arange(40) / 40.0)
    p0 = next(p for p in w if p['row'] == 2 and p['col'] == 4)
    assert p0['x'] == pytest.approx(100.0 - 10.0 * 8.0 / 64.0, abs=1e-3) and p0['y'] == pytest.approx(61.5)
    assert p0['weight'] == pytest.approx(20 / 40)
    boxes = IN.object_boxes(make_obs(), ego={'x': 0.0, 'y': 0.0, 'yaw': 90.0})
    car = next(b for b in boxes if b['id'] == 11)
    assert car['x'] == pytest.approx(-0.2, abs=1e-4) and car['y'] == pytest.approx(15.0) and car['yaw'] == 90.0


def _tiny_vlm():
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    cfg = Qwen2VLConfig(
        text_config={'vocab_size': 128, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2,
                     'bos_token_id': 1, 'eos_token_id': 2,
                     'num_attention_heads': 4, 'num_key_value_heads': 2, 'max_position_embeddings': 256,
                     'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0, 'mrope_section': [1, 1, 2]}},
        vision_config={'depth': 1, 'embed_dim': 16, 'hidden_size': 32, 'num_heads': 2, 'mlp_ratio': 2,
                       'patch_size': 4, 'spatial_merge_size': 2, 'temporal_patch_size': 2, 'in_channels': 3},
        image_token_id=100, video_token_id=101, vision_start_token_id=102, vision_end_token_id=103,
        attn_implementation='eager')
    torch.manual_seed(0)
    m = Qwen2VLForConditionalGeneration(cfg).eval()
    t, h, w = 1, 4, 6
    ids = torch.tensor([[1, 2, 102] + [100] * (t * h * w // 4) + [103, 5, 6, 7]])
    enc = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'mm_token_type_ids': (ids == 100).int(),
           'pixel_values': torch.randn(t * h * w, 3 * 2 * 4 * 4), 'image_grid_thw': torch.tensor([[t, h, w]])}
    return m, enc, (t, h, w)


def test_tiny_qwen2_vl_hooks_eager_and_lora():
    pytest.importorskip('transformers')
    m, enc, grid = _tiny_vlm()
    pts = IN.vlm_hook_points(m)
    assert [n for n, _ in pts] == ['model.language_model.layers.0.self_attn', 'model.language_model.layers.1.self_attn']
    d = IN.vlm_attention(m, enc, 100, list(grid))
    assert d['image_tokens'] == 6 and len(d['image_relevance']) == 6 and np.isclose(sum(d['image_relevance']), 1.0)
    assert np.asarray(d['heatmap']).shape == (2, 3) and 0.0 < d['image_share'] < 1.0
    assert all(len(v) == 6 for v in d['per_layer_last'].values())
    d2 = IN.vlm_attention(m, enc, 100, list(grid))
    assert d2['image_relevance'] == d['image_relevance']                    # deterministic
    z = IN.vlm_zero_head(m, enc, pts[0][0], 1)
    assert z['kl'] > 0 and IN.vlm_zero_head(m, enc, pts[0][0], 1)['kl'] == z['kl']
    after = IN.vlm_attention(m, enc, 100, list(grid))
    assert after['image_relevance'] == d['image_relevance']                 # the zeroing hook is gone
    from falcon_autonomy.core.vlm_lora import apply_lora, lora_modules
    apply_lora(m, r=2)
    assert lora_modules(m)
    assert [n for n, _ in IN.vlm_hook_points(m)] == [n for n, _ in pts]      # the adapter's attention modules
    assert np.allclose(IN.vlm_attention(m, enc, 100, list(grid))['image_relevance'], d['image_relevance'],
                       atol=1e-6)                                            # B = 0: the adapter starts neutral
    m.config._attn_implementation = 'sdpa'
    with pytest.raises(ValueError, match='eager'):
        IN.vlm_attention(m, enc, 100, list(grid))


def test_hfrunner_asks_for_eager_attention(tmp_path, monkeypatch):
    pytest.importorskip('transformers')
    import transformers
    from falcon_autonomy.core import vlm_models as VM
    seen = {}

    class FakeModel:
        def to(self, dev):
            return self

        def eval(self):
            return self

    def fake_model(folder, **kw):
        seen.update(kw)
        return FakeModel()
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', staticmethod(fake_model))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', staticmethod(lambda folder, **kw: object()))
    (tmp_path / 'm').mkdir()
    r = VM.make_runner(f'hf:{tmp_path / "m"},attn_implementation=eager,device=cpu')
    assert seen['attn_implementation'] == 'eager' and r.attn_implementation == 'eager'
    seen.clear()
    VM.HFRunner(str(tmp_path / 'm'), device='cpu')
    assert 'attn_implementation' not in seen


def test_frames_and_cli(project, tmp_path, fake_carla, capsys):
    pol = make_policy(tmp_path, seed=1, name='p')
    rc = IN.main_introspect(['attention', project.path, '--policy', pol.path, '--scenario', 'lead_brake', '--steps',
                             '12', '--every', '4', '--out', str(tmp_path / 'att'), '--fake-carla', fake_carla])
    assert rc == 0, capsys.readouterr().out
    doc = json.load(open(tmp_path / 'att' / 'attention.json'))
    assert [r['frame'] for r in doc['frames']] == [0, 4, 8, 12]
    f0 = doc['frames'][0]
    assert f0['points'] and all(math.isfinite(p['x']) for p in f0['points'])   # depth recorded: world points
    assert any(b['id'] >= 0 for b in f0['objects']) and 'attention is not explanation' in doc['note']
    rec = IN.load_frames(str(tmp_path / 'att' / 'frames.npz'))
    assert len(rec['frames']) == 13 and rec['frames'][0]['depth'].shape == (72, 128)
    assert len(rec['actions']) == 12 and rec['egos'][0]['x'] is not None
    rc = IN.main_introspect(['intervene', '--policy', pol.path, '--frames', str(tmp_path / 'att' / 'frames.npz'),
                             '--op', '{"op": "mask_class", "class": "car"}', '--op',
                             '{"op": "zero_head", "layer": 0, "head": 2}', '--out', str(tmp_path / 'iv')])
    assert rc == 0
    iv = json.load(open(tmp_path / 'iv' / 'intervene.json'))
    assert iv['summary']['frames'] == 13 and iv['frames'][0]['tokens_hidden'] == 1
    assert IN.main_introspect(['intervene', '--policy', pol.path, '--frames', str(tmp_path / 'att' / 'frames.npz'),
                               '--op', '{"op": "nope"}', '--out', str(tmp_path / 'x')]) == 2
    assert IN.main_introspect(['attention', '--policy', pol.path, '--out', str(tmp_path / 'y')]) == 2
    assert IN.main_introspect(['what']) == 2 and IN.main_introspect(['--help']) == 0
    import falcon_autonomy.cli as cli
    assert cli.main(['introspect', 'attention', '--help']) == 0
