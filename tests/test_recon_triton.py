"""The Triton rasteriser (core.recon.raster_triton) against the pure-PyTorch reference (core.recon.raster): forward
outputs and every gradient on random scenes and on the edge cases — empty tiles, Gaussians partly off-screen and at
the near plane, opacities under 1/255, a median depth never reached, no visible Gaussian, image sizes that are not a
multiple of the tile, 4/8/16-pixel tiles, many channels (the tl.dot path) —, the early termination and the
consistency of its gradient, run-to-run determinism, the backend selection and a training run's record of it.

These run whenever a CUDA device and Triton are present (they take a few megabytes and seconds; the Triton backend
is what a CUDA training uses by default, so its agreement with the reference is checked wherever it can run) and
skip otherwise. The float64 finite-difference checks of the reference itself are in test_recon.py.

Tolerances. Both paths compute in float32 (ε = 1.2·10⁻⁷) the same terms in different orders: the reference takes a
cumulative product over a tile's whole list and sums over it with einsum; the kernels take it 32 Gaussians at a
time and form the backward's suffix sums as (total − prefix), which can lose up to the total's magnitude × ε and is
then divided by 1 − α ≥ 0.01. So outputs are compared to 2·10⁻⁵ of their largest magnitude and gradients to 10⁻⁴ of
each gradient's largest magnitude (measured differences were 1–4·10⁻⁷ of it on the random scene).
"""
import math

import pytest

torch = pytest.importorskip('torch')

from falcon_autonomy.core.recon import raster as R  # noqa: E402

HAVE = torch.cuda.is_available() and R.triton_status()[0]
need_triton = pytest.mark.skipif(not HAVE, reason='needs a CUDA device and Triton')
DEV = 'cuda'


def scene(n=1500, seed=0, z0=6.0, spread=(2.0, 1.5, 1.0), scale=(0.02, 0.22), op=(0.01, 0.99), dev=DEV):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g) * torch.tensor(spread) + torch.tensor([0.0, 0.0, z0])
    scales = torch.rand(n, 3, generator=g) * (scale[1] - scale[0]) + scale[0]
    quats = torch.randn(n, 4, generator=g)
    ops = torch.rand(n, generator=g) * (op[1] - op[0]) + op[0]
    return [t.to(dev) for t in (means, scales, quats, ops)]


def composite_both(means, scales, quats, ops, W, H, K, C=3, M=1, tile=8, t_stop=0.0, seed=1, bg=None):
    """Rasterise one projected scene with both backends from the same leaves; returns per backend (outputs, grads,
    info) for a random linear loss over every output."""
    vm = torch.eye(4, device=means.device)
    proj0 = R.project(means, scales, quats, ops, vm, K, W, H)
    g = torch.Generator().manual_seed(seed)
    feats = torch.rand(len(means), C, generator=g).to(means.device)
    mfeat = (torch.rand(len(means), M, generator=g) > 0.6).float().to(means.device) if M else None
    bg = torch.rand(C, generator=g).to(means.device) if bg is None else bg
    res = {}
    for be in ('torch', 'triton'):
        leaves = {k: proj0[k].detach().clone().requires_grad_(True) for k in ('means2d', 'conic', 'depth')}
        op = ops.detach().clone().requires_grad_(True)
        f = feats.clone().requires_grad_(True)
        mf = None if mfeat is None else mfeat.clone().requires_grad_(True)
        proj = dict(proj0, **leaves, opacity=op)
        img, dep, al, info = R.rasterize(proj, f, W, H, bg, tile=tile, median_features=mf, backend=be,
                                         t_stop=t_stop if be == 'triton' else None)
        outs = {'image': img, 'depth': dep, 'alpha': al}
        if mf is not None:
            outs['median'] = info.pop('median')
        gw = torch.Generator().manual_seed(seed + 7)
        loss = sum((o * torch.rand(o.shape, generator=gw).to(o.device)).sum() for o in outs.values())
        if loss.requires_grad:                       # the reference's graph is empty when nothing is visible
            loss.backward()
        named = {'means2d': leaves['means2d'], 'conic': leaves['conic'], 'depth': leaves['depth'], 'opacity': op,
                 'features': f, 'median_features': mf}
        grads = {k: torch.zeros_like(v) if v.grad is None else v.grad for k, v in named.items() if v is not None}
        res[be] = ({k: v.detach() for k, v in outs.items()}, grads, info)
    return res


def assert_agree(res, out_tol=2e-5, grad_tol=1e-4):
    (oa, ga, ia), (ob, gb, ib) = res['torch'], res['triton']
    assert ia['backend'] == 'torch' and ib['backend'] == 'triton' and ia['pairs'] == ib['pairs']
    for k in oa:
        scale = max(1.0, float(oa[k].abs().max()))
        err = float((oa[k] - ob[k]).abs().max())
        assert err <= out_tol * scale, (k, err, scale)
    for k in ga:
        a, b = ga[k], gb[k]
        scale = float(a.abs().max())
        err = float((a - b).abs().max())
        assert err <= grad_tol * scale + 1e-9, (k, err, scale)


@need_triton
@pytest.mark.parametrize('tile', [4, 8, 16])
def test_triton_matches_torch_on_a_random_scene(tile):
    res = composite_both(*scene(), 100, 70, (80.0, 80.0, 50.0, 35.0), tile=tile)
    assert res['torch'][2]['pairs'] > 1000
    assert_agree(res)
    # the median is reached on most pixels here, and some Gaussians have no median role: both exercised
    med = res['torch'][0]['median'][..., 0]
    assert (med > 0).float().mean() > 0.5 and (med == 0).any()


@need_triton
def test_triton_matches_torch_with_many_channels_and_odd_sizes():
    """33 composited channels (colour and 30 class votes, as a semantic render: the tl.dot path), 3 median
    channels, an image of 61 × 37 (partial tiles on two edges)."""
    res = composite_both(*scene(800, seed=3), 61, 37, (50.0, 50.0, 30.0, 18.0), C=33, M=3)
    assert_agree(res)


@need_triton
def test_triton_edge_cases():
    W, H, K = 96, 64, (70.0, 70.0, 48.0, 32.0)
    # Gaussians in one corner only: most tiles empty (background, T = 1, depth 0, median 0)
    m, s, q, o = scene(200, seed=4, z0=8.0, spread=(0.3, 0.3, 0.5))
    m[:, :2] -= torch.tensor([4.0, 3.0], device=DEV)
    res = composite_both(m, s, q, o, W, H, K)
    assert_agree(res)
    a = res['triton'][0]['alpha']
    assert (a == 0).float().mean() > 0.5
    # partly off-screen, at and behind the near plane, and too transparent to touch a pixel
    m, s, q, o = scene(600, seed=5, spread=(6.0, 4.0, 3.0))
    m[:40, 2] = torch.linspace(-1.0, R.NEAR, 40)                         # behind / on the near plane: not drawn
    m[40:80, 2] = torch.linspace(R.NEAR + 0.01, 0.4, 40)                  # just in front: huge, clamped footprints
    o[80:140] = torch.linspace(0.5 / 255, 1.5 / 255, 60)                  # under 1/255 or barely over it
    res = composite_both(m, s, q, o, W, H, K)
    assert_agree(res)
    vis = R.project(m, s, q, o, torch.eye(4, device=DEV), K, W, H)['visible']
    assert not vis[:40].any() and vis[40:80].any() and not vis[80:100].any()
    # faint everywhere: the transmittance never falls below ½, so the median is zero on every pixel
    m, s, q, o = scene(60, seed=6, op=(0.01, 0.05))
    res = composite_both(m, s, q, o, W, H, K)
    assert_agree(res)
    assert float(res['triton'][0]['median'].abs().max()) == 0.0
    # nothing visible at all: all background, every gradient zero
    m, s, q, o = scene(50, seed=7, z0=-5.0, spread=(0.5, 0.5, 0.5))
    bg = torch.tensor([0.2, 0.4, 0.6], device=DEV)
    res = composite_both(m, s, q, o, W, H, K, bg=bg)
    assert res['triton'][2]['pairs'] == 0
    assert_agree(res)
    assert torch.equal(res['triton'][0]['image'], bg.expand(H, W, 3))
    assert all(float(v.abs().max()) == 0.0 for v in res['triton'][1].values())


@need_triton
def test_render_gradients_agree_end_to_end():
    """render() from world-space parameters with degree-3 spherical harmonics, a background and a median: every
    parameter's gradient (through the projection, the covariance, the SH and the compositing) agrees."""
    m, s, q, o = scene(1200, seed=8)
    g = torch.Generator().manual_seed(9)
    sh = (torch.randn(len(m), 16, 3, generator=g) * 0.3).to(DEV)
    sky = (torch.rand(len(m), 1, generator=g) > 0.8).float().to(DEV)
    vm = torch.eye(4, device=DEV)
    vm[:3, 3] = torch.tensor([0.1, -0.05, 0.2])
    K = (90.0, 92.0, 60.3, 40.1)
    grads, outs = {}, {}
    for be in ('torch', 'triton'):
        ps = [t.clone().requires_grad_(True) for t in (m, torch.log(s), q, torch.logit(o), sh)]
        out = R.render(ps[0], torch.exp(ps[1]), ps[2], torch.sigmoid(ps[3]), ps[4], vm, K, 120, 80,
                       background=torch.tensor([0.1, 0.2, 0.3], device=DEV), sh_degree=3, median_features=sky,
                       backend=be, t_stop=0.0)
        assert out['info']['backend'] == be
        gw = torch.Generator().manual_seed(10)
        loss = sum((out[k] * torch.rand(out[k].shape, generator=gw).to(DEV)).sum()
                   for k in ('image', 'depth', 'alpha', 'median'))
        loss.backward()
        grads[be] = [p.grad for p in ps] + [out['means2d'].grad]
        outs[be] = out
    for k in ('image', 'depth', 'alpha', 'median'):
        a, b = outs['torch'][k].detach(), outs['triton'][k].detach()
        assert float((a - b).abs().max()) <= 2e-5 * max(1.0, float(a.abs().max())), k
    for name, a, b in zip(('means', 'log_scales', 'quats', 'opacity_logits', 'sh', 'means2d'), grads['torch'],
                          grads['triton']):
        assert float((a - b).abs().max()) <= 1e-4 * float(a.abs().max()), name


@need_triton
def test_early_termination_and_its_gradient():
    """64 large flat Gaussians of α = 0.3 stacked over a 16 × 16 image: after the first block of 32 every pixel's
    T = 0.7³² ≈ 1.1·10⁻⁵ < T_STOP, so each tile stops there. The output is within T × the largest feature of the
    full composite; the 32 Gaussians never composited get exactly zero gradient (the backward pass goes as far as
    the forward did), the others the reference's up to the same bound."""
    n = 64
    m = torch.zeros(n, 3, device=DEV)
    m[:, 2] = 5.0 + 0.1 * torch.arange(n, device=DEV)
    s = torch.full((n, 3), 60.0, device=DEV)
    q = torch.tensor([[1.0, 0, 0, 0]], device=DEV).repeat(n, 1)
    o = torch.full((n,), 0.3, device=DEV)
    full = composite_both(m, s, q, o, 16, 16, (16.0, 16.0, 8.0, 8.0), M=1, t_stop=0.0)
    assert_agree(full)
    res = composite_both(m, s, q, o, 16, 16, (16.0, 16.0, 8.0, 8.0), M=1, t_stop=R.T_STOP)
    (oa, ga, _), (ob, gb, _) = res['torch'], res['triton']
    t32 = 0.7 ** 32
    for k in ('image', 'depth', 'alpha'):
        fmax = 1.0 if k != 'depth' else float(m[:, 2].max())
        err = float((oa[k] - ob[k]).abs().max())
        assert 0 < err <= 1.05 * t32 * fmax + 1e-6, (k, err)
    assert torch.equal(oa['median'], ob['median'])                     # crossed ½ long before the stop
    order = torch.argsort(m[:, 2])
    late, early = order[32:], order[:32]
    for k in ('features', 'opacity', 'means2d', 'conic', 'depth'):
        assert float(gb[k][late].abs().max()) == 0.0, k
        assert float(ga[k][late].abs().max()) > 0.0, k                  # the reference composites them
    gf_a, gf_b = ga['features'][early], gb['features'][early]
    assert float((gf_a - gf_b).abs().max()) <= 1e-4 * float(gf_a.abs().max())


@need_triton
def test_triton_backward_is_deterministic():
    m, s, q, o = scene(3000, seed=11, spread=(1.0, 0.8, 0.5))              # many pairs per Gaussian and tile
    a = composite_both(m, s, q, o, 96, 64, (70.0, 70.0, 48.0, 32.0))['triton'][1]
    b = composite_both(m, s, q, o, 96, 64, (70.0, 70.0, 48.0, 32.0))['triton'][1]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_backend_selection():
    assert R.resolve_backend('torch', 'cuda') == 'torch'
    assert R.resolve_backend('auto', 'cpu') == 'torch'
    assert R.resolve_backend(None, 'cpu') == 'torch'
    with pytest.raises(ValueError, match='not a CUDA device'):
        R.resolve_backend('triton', 'cpu')
    with pytest.raises(ValueError, match='unknown'):
        R.resolve_backend('cuda', 'cpu')
    assert R.resolve_backend('auto', 'cuda', torch.float64) == 'torch'
    assert R.resolve_backend('auto', 'cuda', torch.float32, tile=12) == 'torch'
    with pytest.raises(ValueError, match='float32'):
        R.resolve_backend('triton', 'cuda', torch.float64)
    assert R.resolve_backend('auto', 'cuda') == ('triton' if R.triton_status()[0] else 'torch')
    with R.default_backend('torch'):
        assert R.resolve_backend(None, 'cuda') == 'torch'
    assert R._default_backend[0] == 'auto'
    from falcon_autonomy.core.recon.train import ReconError, normalise
    assert normalise({})['raster_backend'] == 'auto'
    with pytest.raises(ReconError):
        normalise({'raster_backend': 'cuda'})


@need_triton
def test_a_cuda_training_records_its_backend(tmp_path):
    from falcon_autonomy.core.provenance import read_json
    from falcon_autonomy.core.recon.train import train
    from test_recon import make_falcon_run
    run = make_falcon_run(str(tmp_path / 'run'), n=8, actor=False)
    s = {'iterations': 20, 'width': 64, 'device': 'cuda', 'eval_every': 10, 'log_every': 10, 'sky_points': 100,
         'densify_from': 5, 'densify_every': 5, 'renders': 1, 'semantic': False, 'gpu_memory_fraction': 0.9}
    train(f'falcon:{run}', str(tmp_path / 'a'), s)
    rj = read_json(str(tmp_path / 'a' / 'run.json'))
    assert rj['raster']['backend'] == 'triton' and rj['raster']['requested'] == 'auto'
    assert rj['raster']['early_stop_transmittance'] == R.T_STOP and rj['raster']['triton'].startswith('triton')
    train(f'falcon:{run}', str(tmp_path / 'b'), dict(s, raster_backend='torch'))
    rb = read_json(str(tmp_path / 'b' / 'run.json'))
    assert rb['raster']['backend'] == 'torch' and rb['raster']['early_stop_transmittance'] is None
    ha, hb = rj['summary']['heldout'], rb['summary']['heldout']
    assert math.isfinite(ha['psnr']) and abs(ha['psnr'] - hb['psnr']) < 1.0, (ha, hb)
