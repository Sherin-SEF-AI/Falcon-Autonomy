"""Training a reconstruction: posed images in, Gaussians out (torch at import).

    train('falcon:~/demo.falcon/runs/drive', '~/demo.falcon/recon/drive', {'iterations': 7000})

The recipe is the 3DGS paper's (Kerbl et al. 2023, §5–7), at a shorter schedule by default:

* one training image per step, in a seeded random order (a new permutation every pass over the images);
* loss (1 − λ)·L1 + λ·(1 − SSIM), λ = 0.2, over the pixels the frame's mask keeps (moving objects left out);
* Adam with the paper's learning rates — positions 1.6·10⁻⁴ × the scene's extent decaying exponentially to 1.6·10⁻⁶ ×
  the extent over the run, SH DC 2.5·10⁻³ and the rest 1/20 of it, opacity 0.05, log-scales 5·10⁻³, rotations
  10⁻³ — the extent being 1.1 × the largest distance of a training camera from their mean (the paper's);
* spherical harmonics of degree 0 at first, one more every 1000 steps up to sh_degree (3);
* adaptive density control (core.recon.gaussians.Densifier) every 100 steps from step 500 until half the run, the
  opacities reset every 3000 steps within that window.

The paper trains 30 000 steps; the default here is 7000 (the paper reports its 7000-step results too, somewhat below
its 30 000-step ones) because this rasteriser is plain PyTorch. Every setting is in run.json.

Held out: every holdout-th frame (default 8: frames 0, 8, 16, …, the Mip-NeRF 360 convention the paper follows) is
never trained on and is what the evaluation scores. Semantic classes: when the source has semantic labels (Falcon
runs), each Gaussian's classes are voted after training — its rendering weights wᵢ summed per class over the training
pixels (one backward pass per view of Σ_pixels onehot(label)·rendered features gives exactly those sums) — so a
re-rendered view can carry semantic labels (core.recon.rerender).

GPU: a CUDA training is a needs_gpu job (job_spec()) and, run from the CLI, takes Falcon's GPU lock itself
(core.training.gpu_locks). Its share of GPU memory is capped (gpu_memory_fraction, 0.6) and its peak is measured.

Reproducibility: seeded (torch, numpy, the image order, the splits' samples). On the CPU with one thread two runs
give the same Gaussians bit for bit (tested); with several CPU threads the backward pass's sums are accumulated in a
varying order, so runs differ in the last bits. On CUDA two 1500-step runs with density control were identical bit
for bit (measured on one machine, driver and torch version; docs/reconstruction.md) — another GPU or torch may differ.

Outputs (the reconstruction folder): point_cloud.ply (+ extras.npz), views.json (every frame's pose and split),
curves.jsonl (the training curve), evaluation.json (held-out PSNR / SSIM / depth), renders/ (a few held-out frames
beside their renders), run.json (provenance: the source's files hashed, settings, seed, versions, device, timings,
peak memory).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from . import data as D
from .gaussians import Adam, Densifier, Gaussians
from .metrics import psnr, ssim_loss

SCHEMA = 'falcon.recon/1'
DEFAULTS = {
    'iterations': 7000, 'width': 640, 'holdout': 8, 'init': 'auto', 'sh_degree': 3, 'seed': 0, 'device': 'auto',
    'voxel_m': 0.15, 'max_init_points': 300_000, 'sky_points': 20_000, 'max_gaussians': 600_000,
    'lambda_ssim': 0.2, 'lr_means': 1.6e-4, 'lr_means_final': 1.6e-6, 'lr_sh_dc': 2.5e-3, 'lr_opacity': 0.05,
    'lr_scales': 5e-3, 'lr_quats': 1e-3, 'densify_from': 500, 'densify_until': None, 'densify_every': 100,
    'grad_threshold': 2e-4, 'percent_dense': 0.01, 'opacity_reset_every': 3000, 'sh_up_every': 1000,
    'eval_every': 1000, 'log_every': 50, 'masks': True, 'semantic': True, 'every': 1, 'camera': None,
    'gpu_memory_fraction': 0.6, 'tile': 8, 'renders': 4, 'background': [0.0, 0.0, 0.0],
}


class Cancelled(Exception):
    pass


class ReconError(ValueError):
    pass


class _Quiet:
    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass

    def metric(self, *a, **k):
        pass


def normalise(s: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (s or {}).items() if v is not None})
    unknown = sorted(set(out) - set(DEFAULTS))
    if unknown:
        raise ReconError(f'unknown settings: {", ".join(unknown)}')
    for k in ('iterations', 'width', 'holdout', 'sh_degree', 'seed', 'max_gaussians', 'every', 'renders'):
        out[k] = int(out[k])
    if not 0 <= out['sh_degree'] <= 3:
        raise ReconError('sh_degree is 0–3')
    if out['iterations'] < 1:
        raise ReconError('iterations must be at least 1')
    if out['densify_until'] is None:
        out['densify_until'] = out['iterations'] // 2
    return out


def split_frames(frames: list, holdout: int) -> tuple:
    if holdout and holdout > 1 and len(frames) > 2:
        test = [f for k, f in enumerate(frames) if k % holdout == 0]
        train = [f for k, f in enumerate(frames) if k % holdout != 0]
        return train, test
    return list(frames), []


def scene_extent(frames: list) -> float:
    c = np.array([f.centre for f in frames])
    return max(1.0, 1.1 * float(np.linalg.norm(c - c.mean(0), axis=1).max())) if len(c) else 1.0


def load_views(frames: list, wh, device, masks=True, semantic=True):
    """Images (N, H, W, 3) uint8, masks (N, H, W) bool or None, semantic (N, H, W) uint8 or None, Ks, viewmats."""
    imgs, ms, sems = [], [], []
    for f in frames:
        imgs.append(torch.from_numpy(np.ascontiguousarray(D.resize_rgb(D.load_rgb(f.image), wh))))
        m = D.resize_mask(D.load_mask(f), wh) if masks else None
        ms.append(None if m is None else torch.from_numpy(m))
        s = D.resize_labels(D.load_semantic(f), wh) if semantic else None
        sems.append(None if s is None else torch.from_numpy(np.ascontiguousarray(s)))
    out = {'images': torch.stack(imgs).to(device),
           'K': [D.scaled_K(f.K, f.size, wh) for f in frames],
           'viewmat': torch.as_tensor(np.stack([f.viewmat for f in frames]), dtype=torch.float32, device=device)}
    out['masks'] = torch.stack([m if m is not None else torch.ones(wh[1], wh[0], dtype=torch.bool) for m in ms]) \
        .to(device) if any(m is not None for m in ms) else None
    out['semantic'] = torch.stack(sems).to(device) if sems and all(s is not None for s in sems) else None
    return out


@torch.no_grad()
def quick_psnr(gs: Gaussians, views: dict, deg: int, bg, limit: int = 8) -> float | None:
    n = len(views['images'])
    if not n:
        return None
    idx = np.linspace(0, n - 1, min(limit, n)).round().astype(int)
    H, W = views['images'].shape[1:3]
    vals = []
    for i in idx:
        out = gs.render(views['viewmat'][i], views['K'][i], W, H, deg, bg)
        gt = views['images'][i].float() / 255.0
        m = views['masks'][i] if views['masks'] is not None else None
        vals.append(psnr(out['image'].clamp(0, 1), gt, m))
    return float(np.mean(vals))


def vote_semantics(gs: Gaussians, views: dict, n_classes: int = 30, cancel=None) -> torch.Tensor:
    """Per-Gaussian class votes (N, n_classes): rendering weights summed per class over the training pixels (module
    doc); masked pixels and the ego's class do not vote."""
    H, W = views['images'].shape[1:3]
    feats = torch.zeros(len(gs), n_classes, device=gs.device, requires_grad=True)
    geo = gs.detach_copy()
    for i in range(len(views['images'])):
        if cancel and cancel():
            raise Cancelled()
        out = geo.render(views['viewmat'][i], views['K'][i], W, H, features=feats)
        lab = views['semantic'][i].long().clamp(max=n_classes - 1)
        onehot = torch.nn.functional.one_hot(lab, n_classes).to(out['image'].dtype)
        if views['masks'] is not None:
            onehot = onehot * views['masks'][i][..., None]
        onehot[..., D.EGO_CLASS if D.EGO_CLASS < n_classes else 0] = 0
        (out['image'] * onehot).sum().backward()
    return feats.grad.detach()


def train(source, out_dir: str, settings: dict | None = None, emitter=None, cancel=None, argv=None) -> dict:
    """Train a reconstruction of `source` ('KIND:PATH', core.recon.data.parse_source) into out_dir; returns the
    summary (also in run.json)."""
    from . import evaluate as EV
    from .. import provenance as PV
    from ..paths import ensure_dir
    from .scene import pick_device, write_views
    em = emitter or _Quiet()
    s = normalise(settings)
    out_dir = ensure_dir(os.path.abspath(os.path.expanduser(out_dir)))
    kind, path, opts = D.parse_source(source) if isinstance(source, str) else (source[0], source[1], {})
    if s['camera']:
        opts['camera'] = s['camera']
    if s['every'] > 1:
        opts['every'] = s['every']
    cap = D.open_capture((kind, path), opts.get('camera'), **{k: v for k, v in opts.items() if k != 'camera'})
    train_f, test_f = split_frames(cap.frames, s['holdout'])
    if not train_f:
        raise ReconError('no frames to train on')
    dev = pick_device(s['device'])
    torch.manual_seed(s['seed'])
    np.random.seed(s['seed'] % 2 ** 32)
    gen = torch.Generator(device=dev)
    gen.manual_seed(s['seed'])
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(s['gpu_memory_fraction']))
        torch.cuda.reset_peak_memory_stats(dev)
    wh = D.working_size(cap.frames[0].size, s['width'])
    run = PV.new_run('recon.train', argv or [], s['seed'], dict(s, source=f'{kind}:{path}'),
                     {f'source/{k}': v for k, v in (cap.meta.get('inputs') or {}).items()})
    run['schema_recon'] = SCHEMA
    run['source'] = dict(cap.summary(), meta={k: v for k, v in cap.meta.items() if k not in ('inputs', 'sensor')})
    run_path = os.path.join(out_dir, 'run.json')
    PV.write_json(run_path, run)
    t_start = time.perf_counter()
    em.progress(0.0, f'reading {len(train_f)} training images', force=True)
    views = load_views(train_f, wh, dev, s['masks'], s['semantic'])
    tviews = load_views(test_f, wh, dev, s['masks'], False) if test_f else None
    em.progress(0.0, 'initialising the Gaussians', force=True)
    t0 = time.perf_counter()
    pts = D.init_points(cap, s['init'], train_f, s['voxel_m'], s['max_init_points'], wh[0], s['seed'],
                        s['sky_points'])
    gs = Gaussians.from_points(pts['xyz'], pts['rgb'], s['sh_degree'], pts['sky'], device=dev)
    t_init = time.perf_counter() - t0
    extent = scene_extent(train_f)
    lrs = {'means': s['lr_means'] * extent, 'sh_dc': s['lr_sh_dc'], 'sh_rest': s['lr_sh_dc'] / 20.0,
           'opacity_logits': s['lr_opacity'], 'log_scales': s['lr_scales'], 'quats': s['lr_quats']}
    opt = Adam(gs, lrs)
    dens = Densifier(gs, extent, s['grad_threshold'], s['percent_dense'], max_gaussians=s['max_gaussians'],
                     generator=gen)
    bg = torch.tensor(s['background'], dtype=torch.float32, device=dev)
    n_train = len(train_f)
    H, W = wh[1], wh[0]
    rng = np.random.default_rng(s['seed'])
    order = []
    curves = []
    curve_path = os.path.join(out_dir, 'curves.jsonl')
    fh = open(curve_path, 'w', encoding='utf-8')
    n_iter = s['iterations']
    em.log(f'{len(gs)} Gaussians from {pts["how"]} ({pts["stats"]}); extent {extent:.1f} m; {n_train} training and '
           f'{len(test_f)} held-out frames at {W} × {H}; {dev}')
    t_loop = time.perf_counter()
    dens_log = []
    try:
        for it in range(1, n_iter + 1):
            if cancel and cancel():
                raise Cancelled()
            if not order:
                order = list(rng.permutation(n_train))
            v = int(order.pop())
            frac = (it - 1) / max(1, n_iter - 1)
            opt.lrs['means'] = math.exp((1 - frac) * math.log(s['lr_means'] * extent) +
                                        frac * math.log(s['lr_means_final'] * extent))
            deg = min(s['sh_degree'], (it - 1) // s['sh_up_every'])
            out = gs.render(views['viewmat'][v], views['K'][v], W, H, deg, bg, tile=s['tile'])
            gt = views['images'][v].float() / 255.0
            img = out['image']
            m = views['masks'][v] if views['masks'] is not None else None
            if m is not None:
                mf = m.float()
                l1 = ((img - gt).abs().mean(-1) * mf).sum() / mf.sum().clamp_min(1)
            else:
                mf = None
                l1 = (img - gt).abs().mean()
            loss = (1 - s['lambda_ssim']) * l1 + s['lambda_ssim'] * ssim_loss(img, gt, mf)
            loss.backward()
            with torch.no_grad():
                if it < s['densify_until'] and out['means2d'].grad is not None:
                    dens.add_stats(out['means2d'].grad, out['visible'], out['radius'], W, H)
                opt.step()
                opt.zero_grad()
                if s['densify_from'] <= it < s['densify_until'] and it % s['densify_every'] == 0:
                    dens_log.append(dict(dens.step(opt), step=it))
                if s['opacity_reset_every'] and it % s['opacity_reset_every'] == 0 and it < s['densify_until']:
                    dens.reset_opacity(opt)
            if it % s['log_every'] == 0 or it == n_iter:
                el = time.perf_counter() - t_loop
                row = {'step': it, 'loss': round(loss.item(), 6), 'l1': round(l1.item(), 6),
                       'psnr_train_view': round(psnr(img.detach().clamp(0, 1), gt, m), 4), 'gaussians': len(gs),
                       'seconds': round(el, 3), 'sh_degree': deg}
                if s['eval_every'] and (it % s['eval_every'] == 0 or it == n_iter) and tviews is not None:
                    row['psnr_heldout'] = round(quick_psnr(gs, tviews, deg, bg), 4)
                curves.append(row)
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                em.progress(it / n_iter, f'step {it} of {n_iter}: loss {row["loss"]:.4f}, {len(gs)} Gaussians'
                            + (f', held-out PSNR {row["psnr_heldout"]:.2f} dB' if 'psnr_heldout' in row else ''))
    except Cancelled:
        fh.close()
        run.update(status='cancelled', finished=PV.utc_now(), exit_code=3)
        PV.write_json(run_path, run)
        raise
    fh.close()
    t_train = time.perf_counter() - t_loop
    em.progress(1.0, 'saving and evaluating', force=True)
    t0 = time.perf_counter()
    if s['semantic'] and views['semantic'] is not None:
        gs.votes = vote_semantics(gs, views, cancel=cancel)
    t_vote = time.perf_counter() - t0
    gs.write_ply(os.path.join(out_dir, 'point_cloud.ply'))
    write_views(os.path.join(out_dir, 'views.json'), cap, train_f, test_f, wh,
                {k: v for k, v in opts.items() if k != 'camera'})
    del views
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    ev = EV.heldout(gs, test_f, wh, dev, bg, out_dir, renders=s['renders'], masks=s['masks']) if test_f else None
    t_eval = time.perf_counter() - t0
    doc = {'schema': 'falcon.recon_evaluation/1', 'heldout': ev,
           'lpips': 'not computed: no pinned LPIPS network is available in Falcon',
           'masks': cap.meta.get('masks'),
           'note': 'held-out frames are interpolations along the recorded path; a pose off the path is an '
                   'extrapolation (falcon recon eval --reference / --warp-shift)'}
    doc['provenance'] = PV.block({'run.json': run_path}, s['seed'])
    PV.write_json(os.path.join(out_dir, 'evaluation.json'), doc)
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == 'cuda' else None
    summary = {'gaussians': len(gs), 'sky_gaussians': int(gs.sky.sum()), 'init': pts['how'],
               'init_stats': pts['stats'], 'extent_m': round(extent, 3), 'work_size': list(wh),
               'train_frames': n_train, 'heldout_frames': len(test_f),
               'heldout': None if ev is None else ev['mean'], 'semantic_votes': gs.votes is not None,
               'densification': dens_log[-1] if dens_log else None}
    run.update(status='ok', exit_code=0, finished=PV.utc_now(), summary=summary,
               timings={'init_s': round(t_init, 2), 'train_s': round(t_train, 2), 'vote_s': round(t_vote, 2),
                        'eval_s': round(t_eval, 2), 'total_s': round(time.perf_counter() - t_start, 2),
                        'steps_per_s': round(n_iter / max(t_train, 1e-9), 3)},
               device={'type': dev.type, 'name': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu',
                       'torch': torch.__version__, 'peak_gpu_memory_gib': None if peak is None else round(peak, 3),
                       'threads': torch.get_num_threads()},
               determinism='CPU with one thread: bit for bit (tested); several CPU threads: not (sums accumulate in '
                           'a varying order); CUDA: bit for bit as measured on the reference machine')
    run['outputs'] = PV.hash_tree(out_dir, skip=('run.json',))
    PV.write_json(run_path, run)
    return summary


def job_spec(argv: list, title: str = 'Reconstruction', memory_gb: float = 6.0):
    """The JobSpec of `falcon recon …` (train, eval or rerender): a needs_gpu job (the queue holds the GPU lock for
    it: FALCON_GPU_LOCK_HELD)."""
    import sys
    from ..jobs import JobSpec
    env = dict(os.environ, FALCON_GPU_LOCK_HELD='1')
    return JobSpec('recon', title, [sys.executable, '-m', 'falcon_autonomy', 'recon'] + list(argv), env=env,
                   needs_gpu=True, memory_gb=memory_gb)
