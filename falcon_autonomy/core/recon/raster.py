"""A differentiable 3D Gaussian rasteriser with two backends for its compositing: a pure-PyTorch reference (no CUDA
kernels, no compiler: it runs wherever torch runs, on the GPU or the CPU, in float32 or float64) and Triton kernels
for a CUDA GPU (core.recon.raster_triton: float32; Triton compiles them at first use, no CUDA toolkit needed).

Written from the papers, not from any existing implementation: 3D Gaussian Splatting (Kerbl, Kopanas, Leimkühler and
Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023) for the representation,
the tile-based front-to-back alpha compositing and the 0.3 px² screen-space dilation, and EWA splatting (Zwicker,
Pfister, van Baar and Gross, "EWA Splatting", IEEE TVCG 2002) for the projection of a 3D covariance through the local
affine approximation of the perspective map. The Inria reference code (non-commercial licence) is not used.

The model, per Gaussian: a mean μ (world, metres), a scale s (3, linear; the caller passes exp of its log-scales), a
rotation q (quaternion w, x, y, z; normalised here), an opacity o in (0, 1) and a feature vector f (C channels: RGB
from spherical harmonics, or anything else — class votes, see render()). Σ = R S Sᵀ Rᵀ.

A camera: viewmat (4 × 4, world → camera, OpenCV's optical axes: x right, y down, z forward; a rotation and a
translation), K = (fx, fy, cx, cy) in **pixel-edge coordinates** (pixel i covers [i, i + 1), so its centre is i + ½;
an OpenCV principal point is this − ½), width × height.

Projection (project()): t = W μ + w (camera frame); the Jacobian of (fx x/z + cx, fy y/z + cy) at t, with x/z and
y/z clamped to 15 % beyond the image's own field on each side (the 3DGS paper's guard against the affine
approximation exploding off-screen; for a centred principal point this is its 1.3 × tan(fov/2)); Σ' = J W Σ Wᵀ Jᵀ +
0.3 I (2 × 2); the conic Σ'⁻¹. A Gaussian is drawn where z > near.

Bounds. A pixel at offset d from the projected mean gets α = min(0.99, o·exp(−½ dᵀ Σ'⁻¹ d)), and α < 1/255 is
skipped (as in the paper). So a Gaussian touches exactly the pixels inside the ellipse dᵀ Σ'⁻¹ d ≤ 2 ln(255 o): its
axis-aligned bounding box has half-extents k √Σ'ₓₓ and k √Σ'ᵧᵧ, k = √(2 ln(255 o)). Binning by that box loses no
contribution (the paper's 3σ circle is a looser bound of the same thing), and a Gaussian with o < 1/255 touches
nothing.

Rasterisation (rasterize()): Gaussians are sorted once by camera depth (their centre's z, as in the paper: popping
when the order changes is a known artefact of the method); each (Gaussian, tile) pair of a TILE × TILE pixel grid is
listed and the pairs sorted stably by tile, which keeps the depth order inside each tile. Per pixel, front to back:
Tᵢ = Πⱼ<ᵢ (1 − αⱼ), wᵢ = αᵢ Tᵢ, colour = Σ wᵢ fᵢ + T_N·background, expected depth Σ wᵢ zᵢ, alpha 1 − T_N, and on
request the median depth: the z of the Gaussian at which T falls below ½ (with that Gaussian's own features — which
surface it is), the sharper depth for labels.

Backends (rasterize(backend=…); resolve_backend()). Projection, spherical harmonics, the depth sort and the binning
are the same torch code for both; only the per-tile compositing and its gradient differ.
- 'torch', the reference: tiles are processed in chunks — tiles of similar list length together, each padded to the
  chunk's longest list — so that a chunk's (tiles × pixels × Gaussians) tensors stay under a set number of elements
  (bounded memory, whatever the scene). There is no early stop: every listed Gaussian is composited. Gradients come
  from torch's autograd through these vectorised operations; each chunk runs under activation checkpointing
  (recomputed in the backward pass) so the backward pass needs no more memory than one chunk's forward.
  tests/test_recon.py compares every gradient with central finite differences in float64.
- 'triton': one GPU program per tile walks its list 32 Gaussians at a time in registers, with hand-written backward
  kernels (their formulas are in raster_triton's doc) and no per-pair intermediates beyond one gradient row per
  (Gaussian, tile) pair, summed per Gaussian in a fixed order (bit-for-bit repeatable, no float atomics). It stops a
  tile once every pixel's T < T_STOP = 10⁻⁴ (checked every 32 Gaussians), as the CUDA implementations stop a pixel:
  the result differs from the full composite by less than 10⁻⁴ × the largest feature left out (10⁻⁴ of a colour; of
  the far depth, for the expected depth), the median is unaffected, and the backward pass goes exactly as far as the
  forward did, so its gradient is that of what was rendered. t_stop=0 composites everything, and then it agrees
  with the reference to float32 rounding (tests/test_recon_triton.py: every output and gradient, edge cases
  included).
- 'auto' (the default, changeable with set_default_backend() or default_backend()): Triton when the tensors are
  float32 on a CUDA device, the tile is 4, 8 or 16 px and Triton imports; the reference otherwise.

The hard cuts (α < 1/255, the 0.99 cap, z ≤ near) are not differentiable at their edges, as in the original method.

The projected 2D means are returned with their gradient retained (meta['means2d']) so a trainer can accumulate the
screen-space gradient that drives adaptive density control.

Speed (measured 2026-10-30, RTX 5080, torch 2.9.1 + CUDA 12.8, Triton 3.5.1; docs/reconstruction.md has the
conditions and the GPU's other users): the torch path does in vectorised tensor operations what a kernel does per
thread, evaluates every pixel of a tile against every Gaussian of its padded list and allocates (and, under
checkpointing, recomputes) its intermediates; the Triton kernels keep a tile's pixels and 32 Gaussians at a time in
registers. A training step (render, loss, backward) of a 266 000-Gaussian reconstruction at 640 × 360 took 93 ms
with the torch path and 13 ms with Triton (interleaved in one process), and whole 7000-step trainings ran 6–12 times
faster with Triton (8.8 × when the GPU was least shared), at 1.24 against 2.08 GiB peak GPU memory, with the same
held-out quality within the run-to-run spread of float32 rounding (33.65 against 33.62 dB). With the compositing in
Triton, a step is bound by the torch operations around it (projection, SH, sorting, binning), which is why they are
written to launch few kernels (covariance(), sh_to_color()).
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch.utils.checkpoint import checkpoint

TILE = 8
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
EPS2D = 0.3                    # px², the paper's screen-space dilation (a low-pass filter)
FIELD_GUARD = 0.15             # the Jacobian's x/z, y/z clamp: this share of the field beyond each image edge
NEAR = 0.05
T_STOP = 1e-4                  # the Triton path's early termination: a tile stops once every pixel's T is below this
BACKENDS = ('auto', 'torch', 'triton')
TRITON_TILES = (4, 8, 16)      # the Triton kernels need a power-of-two pixel count per tile

# real spherical-harmonics basis constants (the standard real SH, l ≤ 3)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435)


_default_backend = ['auto']
_triton_state: list = []


def triton_status() -> tuple:
    """(available, 'triton X.Y.Z' or why not): Triton and the kernels' module import (nothing is compiled yet)."""
    if not _triton_state:
        try:
            import triton
            from . import raster_triton  # noqa: F401
            _triton_state.append((True, f'triton {triton.__version__}'))
        except Exception as e:                       # ImportError, or a Triton that fails to initialise
            _triton_state.append((False, f'Triton unavailable ({type(e).__name__}: {e})'))
    return _triton_state[0]


def resolve_backend(backend: str | None, device, dtype=torch.float32, tile: int = TILE) -> str:
    """The rasteriser that runs: 'triton' or 'torch'. backend None means the default (set_default_backend(), 'auto'
    unless changed); 'auto' is Triton when the tensors are float32 on a CUDA device, the tile is 4, 8 or 16 px and
    Triton imports, else torch; 'triton' raises ValueError when it cannot run, saying why."""
    b = backend or _default_backend[0]
    if b not in BACKENDS:
        raise ValueError(f'unknown rasteriser backend {b!r} (one of {", ".join(BACKENDS)})')
    if b == 'torch':
        return 'torch'
    device = torch.device(device)
    if device.type != 'cuda':
        why = f'the tensors are on {device.type}, not a CUDA device'
    elif dtype != torch.float32:
        why = f'its kernels are float32 only, not {dtype}'
    elif tile not in TRITON_TILES:
        why = f'tiles of {tile} px are not one of {TRITON_TILES}'
    else:
        ok, msg = triton_status()
        why = None if ok else msg
    if why is None:
        return 'triton'
    if b == 'triton':
        raise ValueError(f'the Triton rasteriser cannot run: {why}')
    return 'torch'


def set_default_backend(backend: str) -> str:
    """Set the backend rasterize() uses when none is passed; returns the previous one."""
    if backend not in BACKENDS:
        raise ValueError(f'unknown rasteriser backend {backend!r} (one of {", ".join(BACKENDS)})')
    prev, _default_backend[0] = _default_backend[0], backend
    return prev


@contextlib.contextmanager
def default_backend(backend: str):
    """with default_backend('torch'): every render inside uses that backend unless one is passed."""
    prev = set_default_backend(backend)
    try:
        yield
    finally:
        set_default_backend(prev)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternions w, x, y, z (any norm) -> (N, 3, 3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Σ = R S Sᵀ Rᵀ, (N, 3, 3), from linear scales (N, 3) and quaternions (N, 4). Products of these 3 × 3 (and the
    projection's 2 × 3) matrices are written as broadcast sums, not batched matmuls: cuBLAS's batched GEMM on
    millions of tiny matrices was the largest cost of a training step once the compositing ran in Triton (measured
    2026-10-30: ~27 ms of a step, forward and backward, at 477 000 Gaussians)."""
    M = quat_to_rotmat(quats) * scales[:, None, :]
    return (M[:, :, None, :] * M[:, None, :, :]).sum(-1)


def sh_to_color(sh: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """RGB from spherical harmonics. sh (N, (D+1)², 3) — coefficient k of channel c at [:, k, c] —, dirs (N, 3) unit
    view directions (from the camera to the Gaussian), degree D ≤ 3 used. Returns SH·basis + ½, clamped at 0 (the
    paper's convention: a zero SH is mid-grey). The basis is stacked and contracted with the coefficients in one
    broadcast sum (indexing sh[:, k] sixteen times made sixteen full-size zero-filled gradients in the backward
    pass)."""
    basis = [torch.full_like(dirs[:, 0], SH_C0)]
    if degree > 0:
        x, y, z = dirs.unbind(-1)
        basis += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
        if degree > 1:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            basis += [SH_C2[0] * xy, SH_C2[1] * yz, SH_C2[2] * (2 * zz - xx - yy), SH_C2[3] * xz,
                      SH_C2[4] * (xx - yy)]
            if degree > 2:
                basis += [SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * xy * z, SH_C3[2] * y * (4 * zz - xx - yy),
                          SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
                          SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy)]
    b = torch.stack(basis, -1)
    c = (b[:, :, None] * sh[:, :len(basis)]).sum(1)
    return (c + 0.5).clamp_min(0.0)


def project(means, scales, quats, opacities, viewmat, K, width, height, near=NEAR, eps2d=EPS2D):
    """Project Gaussians into a camera. Returns a dict of (N, …) tensors: means2d (pixel-edge coordinates), depth (z),
    conic (a, b, c of Σ'⁻¹ = [[a, b], [b, c]]), cov2d (Σ'ₓₓ, Σ'ₓᵧ, Σ'ᵧᵧ), extent (the bounding box's half-sizes,
    px; not differentiable) and visible (bool: in front of the camera, opaque enough to touch a pixel, and its box
    overlapping the image)."""
    fx, fy, cx, cy = (float(v) for v in K)
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    pc = (means[:, None, :] * R[None]).sum(-1) + t             # W μ + w (a broadcast sum: covariance()'s note)
    x, y, z = pc.unbind(-1)
    front = z > near
    zs = torch.where(front, z, torch.full_like(z, near))
    lo_x, hi_x = -cx / fx, (width - cx) / fx
    lo_y, hi_y = -cy / fy, (height - cy) / fy
    gx, gy = FIELD_GUARD * (hi_x - lo_x), FIELD_GUARD * (hi_y - lo_y)
    tx = (x / zs).clamp(lo_x - gx, hi_x + gx) * zs
    ty = (y / zs).clamp(lo_y - gy, hi_y + gy) * zs
    # the rows of T = J W, J = [[fx/z, 0, −fx tx/z²], [0, fy/z, −fy ty/z²]]; Σ' = T Σ Tᵀ entry by entry
    t0 = (fx / zs)[:, None] * R[0] + (-fx * tx / (zs * zs))[:, None] * R[2]
    t1 = (fy / zs)[:, None] * R[1] + (-fy * ty / (zs * zs))[:, None] * R[2]
    cov3 = covariance(scales, quats)
    s0 = (cov3 * t0[:, None, :]).sum(-1)
    a = (t0 * s0).sum(-1) + eps2d
    b = (t1 * s0).sum(-1)
    c = (t1 * (cov3 * t1[:, None, :]).sum(-1)).sum(-1) + eps2d
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack([c / det, -b / det, a / det], -1)
    u = fx * x / zs + cx
    v = fy * y / zs + cy
    with torch.no_grad():
        k2 = 2.0 * torch.log((255.0 * opacities).clamp_min(1e-12))
        k = torch.sqrt(k2.clamp_min(0.0))
        ext = torch.stack([k * torch.sqrt(a.clamp_min(0)), k * torch.sqrt(c.clamp_min(0))], -1)
        visible = front & (k2 > 0) & (u + ext[:, 0] > 0) & (u - ext[:, 0] < width) & (v + ext[:, 1] > 0) & \
            (v - ext[:, 1] < height) & torch.isfinite(u) & torch.isfinite(v)
    return {'means2d': torch.stack([u, v], -1), 'depth': z, 'conic': conic,
            'cov2d': torch.stack([a, b, c], -1), 'extent': ext, 'visible': visible}


def _bin(means2d, ext, order, width, height, tile, groups=False):
    """The (Gaussian, tile) pairs of the depth-ordered visible Gaussians `order`, sorted by tile (depth order kept
    within a tile). Returns (pair_gauss, tile_start, tile_count, tiles_x, tiles_y) and, with groups, a dict giving
    each Gaussian's pairs (the Triton backward's fixed-order per-Gaussian sums): 'seg_rows' (the pairs' positions in
    the tile-sorted list, grouped by Gaussian), 'seg_start' and 'seg_count' (each of `order`'s segment of it) and
    'seg_gauss' (= order)."""
    dev = means2d.device
    ntx, nty = (width + tile - 1) // tile, (height + tile - 1) // tile
    with torch.no_grad():
        m = means2d.detach()[order]
        e = ext[order]
        # pixels whose centres (i + ½) fall within [u − e, u + e]
        i0 = torch.ceil(m[:, 0] - e[:, 0] - 0.5).clamp(0, width - 1).long()
        i1 = torch.floor(m[:, 0] + e[:, 0] - 0.5).clamp(0, width - 1).long()
        j0 = torch.ceil(m[:, 1] - e[:, 1] - 0.5).clamp(0, height - 1).long()
        j1 = torch.floor(m[:, 1] + e[:, 1] - 0.5).clamp(0, height - 1).long()
        tx0, tx1, ty0, ty1 = i0 // tile, i1 // tile, j0 // tile, j1 // tile
        nx = (tx1 - tx0 + 1).clamp_min(0)
        ny = (ty1 - ty0 + 1).clamp_min(0)
        cnt = nx * ny
        total = int(cnt.sum())
        if total == 0:
            z = torch.zeros(ntx * nty, dtype=torch.long, device=dev)
            e0 = torch.zeros(0, dtype=torch.long, device=dev)
            if groups:
                return e0, z, z, ntx, nty, {'seg_rows': e0, 'seg_start': e0, 'seg_count': e0, 'seg_gauss': e0}
            return e0, z, z, ntx, nty
        gid = torch.repeat_interleave(torch.arange(len(order), device=dev), cnt, output_size=total)
        first = torch.cumsum(cnt, 0) - cnt
        local = torch.arange(total, device=dev) - first[gid]
        w = nx[gid]
        tile_id = (ty0[gid] + local // w) * ntx + (tx0[gid] + local % w)
        tile_sorted, perm = torch.sort(tile_id, stable=True)
        pair_gauss = order[gid[perm]]
        # each tile's run of the sorted list (a search of the sorted ids: cheaper than bincount's atomics)
        edges = torch.searchsorted(tile_sorted, torch.arange(ntx * nty + 1, device=dev, dtype=tile_sorted.dtype))
        start = edges[:-1]
        count = edges[1:] - start
        if groups:
            rows = torch.empty_like(perm)
            rows[perm] = torch.arange(total, device=dev)
            return pair_gauss, start, count, ntx, nty, {'seg_rows': rows, 'seg_start': first, 'seg_count': cnt,
                                                        'seg_gauss': order}
    return pair_gauss, start, count, ntx, nty


def _composite(px, m2, con, op, feat, dep, valid, mfeat=None):
    """One chunk: px (T, P, 2) pixel centres; per tile-list entry m2 (T, K, 2), con (T, K, 3), op (T, K), feat (T, K,
    C), dep (T, K), valid (T, K), and optionally mfeat (T, K, M). Returns (T, P, C + 2 [+ 1 + M]): features, expected
    depth, T_final [, the median depth and the median Gaussian's mfeat]."""
    dx = px[:, :, None, 0] - m2[:, None, :, 0]
    dy = px[:, :, None, 1] - m2[:, None, :, 1]
    a, b, c = con[:, None, :, 0], con[:, None, :, 1], con[:, None, :, 2]
    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
    alpha = (op[:, None, :] * torch.exp(power)).clamp(max=ALPHA_MAX)
    keep = (power <= 0) & (alpha >= ALPHA_MIN) & valid[:, None, :]
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    trans = torch.cumprod(1.0 - alpha, dim=-1)
    t_excl = torch.cat([torch.ones_like(trans[..., :1]), trans[..., :-1]], -1)
    w = alpha * t_excl
    f = torch.einsum('tpk,tkc->tpc', w, feat)
    d = (w * dep[:, None, :]).sum(-1, keepdim=True)
    parts = [f, d, trans[..., -1:]]
    if mfeat is not None:
        cross = ((t_excl >= 0.5) & (trans < 0.5)).to(f.dtype)          # the Gaussian where T falls below ½
        parts.append(torch.einsum('tpk,tkm->tpm', cross, torch.cat([dep[..., None], mfeat], -1)))
    return torch.cat(parts, -1)


def rasterize(proj: dict, features: torch.Tensor, width: int, height: int, background=None, tile: int = TILE,
              chunk_elems: int | None = None, use_checkpoint: bool | None = None, median_features=None,
              backend: str | None = None, t_stop: float | None = None):
    """Alpha-composite projected Gaussians (project()'s dict) with per-Gaussian features (N, C). Returns
    (image (H, W, C), depth (H, W) — the expected depth Σ wᵢ zᵢ, not divided by alpha —, alpha (H, W), info: the
    number of pairs and chunks, the backend that ran, and with median_features (N, M) given 'median' (H, W, 1 + M):
    the median depth — the depth of the Gaussian at which the transmittance falls below ½ — and that Gaussian's
    median_features, zeros where it never does). background: (C,) composited behind (default zeros). backend:
    'torch', 'triton' or 'auto' (None: the default, set_default_backend(); resolve_backend()). t_stop: the Triton
    path's early-termination transmittance (default T_STOP; 0 composites every listed Gaussian, as the torch path
    always does); chunk_elems and use_checkpoint are the torch path's."""
    dev, dt = features.device, features.dtype
    C = features.shape[1]
    M = 0 if median_features is None else median_features.shape[1] + 1
    be = resolve_backend(backend, dev, dt, tile)
    vis = proj['visible']
    idx = torch.nonzero(vis, as_tuple=False).squeeze(1)
    order = idx[torch.argsort(proj['depth'].detach()[idx], stable=True)]
    binned = _bin(proj['means2d'], proj['extent'], order, width, height, tile, groups=be == 'triton')
    pair_gauss, start, count, ntx, nty = binned[:5]
    bg = torch.zeros(C, dtype=dt, device=dev) if background is None else torch.as_tensor(background, dtype=dt,
                                                                                          device=dev)
    if be == 'triton':
        from .raster_triton import Composite
        i32 = lambda t: t.to(torch.int32).contiguous()  # noqa: E731
        bins = {'pair_gauss': i32(pair_gauss), 'start': i32(start), 'count': i32(count), 'ntx': ntx, 'nty': nty}
        bins.update({k: i32(v) for k, v in binned[5].items()})
        dep = proj['depth'][:, None].to(dt)
        mf = None if median_features is None else torch.cat([dep, median_features.to(dt)], 1)
        f_out, t_out, m_out = Composite.apply(proj['means2d'], proj['conic'], proj['opacity'],
                                              torch.cat([features, dep], 1), mf, bins, tile,
                                              T_STOP if t_stop is None else float(t_stop))
        full = torch.cat([f_out, t_out[..., None], m_out], -1)
        n_chunks = 0
    else:
        full, n_chunks = _rasterize_torch(proj, features, median_features, pair_gauss, start, count, ntx, tile, C, M,
                                          chunk_elems, use_checkpoint)
    img = full.reshape(nty, ntx, tile, tile, C + 2 + M).permute(0, 2, 1, 3, 4).reshape(nty * tile, ntx * tile,
                                                                                         C + 2 + M)
    img = img[:height, :width]
    t_final = img[..., C + 1]
    color = img[..., :C] + t_final[..., None] * bg
    info = {'pairs': int(len(pair_gauss)), 'chunks': n_chunks, 'visible': int(len(order)), 'backend': be}
    if M:
        info['median'] = img[..., C + 2:]
    return color, img[..., C], 1.0 - t_final, info


def _rasterize_torch(proj, features, median_features, pair_gauss, start, count, ntx, tile, C, M, chunk_elems,
                     use_checkpoint):
    """The torch path's compositing: (tiles, TILE², C + 2 + M) per tile and pixel — features, expected depth,
    T_final [, median] —, empty tiles (0…, 0, 1, 0…); and the number of chunks."""
    dev, dt = features.device, features.dtype
    if chunk_elems is None:
        chunk_elems = 1 << 24 if dev.type == 'cuda' else 1 << 21
    if use_checkpoint is None:
        use_checkpoint = torch.is_grad_enabled() and any(
            t.requires_grad for t in (features, proj['means2d'], proj['conic'], proj['opacity'], proj['depth']))
    P = tile * tile
    ntiles = len(count)
    pieces, where = [], []
    busy = torch.nonzero(count > 0, as_tuple=False).squeeze(1)
    if len(busy):
        cnt = count[busy]
        srt = torch.argsort(cnt, descending=True)
        busy, cnt = busy[srt], cnt[srt]
        cnt_l = cnt.tolist()
        oy, ox = torch.meshgrid(torch.arange(tile, device=dev, dtype=dt), torch.arange(tile, device=dev, dtype=dt),
                                indexing='ij')
        off = torch.stack([ox.reshape(-1), oy.reshape(-1)], -1) + 0.5
        m2_all, con_all, dep_all = proj['means2d'], proj['conic'], proj['depth']
        op_all = proj['opacity']
        i = 0
        n = len(cnt_l)
        while i < n:
            K = cnt_l[i]
            nt = max(1, min(n - i, chunk_elems // max(1, K * P)))
            tiles = busy[i:i + nt]
            ks = torch.arange(K, device=dev)
            valid = ks[None, :] < cnt[i:i + nt, None]
            pidx = (start[tiles][:, None] + ks[None, :]).clamp(max=max(len(pair_gauss) - 1, 0))
            g = pair_gauss[pidx]
            org = torch.stack([(tiles % ntx) * tile, (tiles // ntx) * tile], -1).to(dt)
            px = org[:, None, :] + off[None]
            args = (px, m2_all[g], con_all[g], op_all[g], features[g], dep_all[g], valid,
                    None if median_features is None else median_features[g])
            out = checkpoint(_composite, *args, use_reentrant=False) if use_checkpoint else _composite(*args)
            pieces.append(out)
            where.append(tiles)
            i += nt
    full = torch.cat([torch.zeros(ntiles, P, C + 1, dtype=dt, device=dev),
                      torch.ones(ntiles, P, 1, dtype=dt, device=dev),
                      torch.zeros(ntiles, P, M, dtype=dt, device=dev)], -1)
    if pieces:
        full = full.index_put((torch.cat(where),), torch.cat(pieces, 0))
    return full, len(pieces)


def render(means, scales, quats, opacities, features, viewmat, K, width, height, background=None, sh_degree=None,
           tile: int = TILE, chunk_elems: int | None = None, median_features=None, backend: str | None = None,
           t_stop: float | None = None):
    """Render one view. features: (N, C) to composite as they are, or with sh_degree given (N, (D+1)², 3)
    spherical-harmonics coefficients evaluated towards this camera. Returns {'image' (H, W, C), 'depth' (H, W,
    expected), 'alpha' (H, W), 'means2d' (N, 2, with its gradient retained when it has one), 'visible', 'radius'
    (N, px: the larger bounding-box half-size), 'info'} and, with median_features (N, M), 'median' (H, W, 1 + M):
    the median depth and the median Gaussian's features (rasterize()). backend, t_stop: rasterize()'s."""
    proj = project(means, scales, quats, opacities, viewmat, K, width, height)
    proj['opacity'] = opacities
    if proj['means2d'].requires_grad:
        proj['means2d'].retain_grad()
    if sh_degree is not None:
        R, t = viewmat[:3, :3], viewmat[:3, 3]
        centre = -(R.T @ t)
        dirs = means - centre
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats = sh_to_color(features, dirs, sh_degree)
    else:
        feats = features
    img, depth, alpha, info = rasterize(proj, feats, width, height, background, tile, chunk_elems,
                                        median_features=median_features, backend=backend, t_stop=t_stop)
    radius = torch.where(proj['visible'], proj['extent'].max(-1).values, torch.zeros_like(proj['extent'][:, 0]))
    out = {'image': img, 'depth': depth, 'alpha': alpha, 'means2d': proj['means2d'], 'visible': proj['visible'],
           'radius': radius, 'info': info}
    if 'median' in info:
        out['median'] = info.pop('median')
    return out


def ray_norms(K, width, height, device=None, dtype=torch.float32) -> torch.Tensor:
    """|K⁻¹ (u, v, 1)| per pixel centre (H, W): range along a pixel's ray = its z-depth × this."""
    fx, fy, cx, cy = (float(v) for v in K)
    u = (torch.arange(width, device=device, dtype=dtype) + 0.5 - cx) / fx
    v = (torch.arange(height, device=device, dtype=dtype) + 0.5 - cy) / fy
    return torch.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)


def scaled_K(K, sx: float, sy: float | None = None) -> tuple:
    """K for an image resized by (sx, sy) (pixel-edge coordinates scale exactly)."""
    sy = sx if sy is None else sy
    fx, fy, cx, cy = K
    return (fx * sx, fy * sy, cx * sx, cy * sy)


def fov_K(width: int, height: int, fov_deg: float) -> tuple:
    """A centred pinhole K of a horizontal field of view (CARLA's camera)."""
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return (f, f, width / 2.0, height / 2.0)
