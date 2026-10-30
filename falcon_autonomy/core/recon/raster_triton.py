"""Triton kernels for the hot part of core.recon.raster: the per-tile front-to-back compositing and its gradient.

Imported only when the Triton backend runs (raster.rasterize(..., backend='triton' or 'auto' on a CUDA device with
Triton installed); raster.py stays importable without Triton. Triton (MIT licence) compiles these kernels for the GPU
at first use — no CUDA toolkit needed — and caches them (~/.triton).

What the kernels compute is exactly raster._composite's math (raster.py's module doc), per tile of TILE × TILE
pixels over the tile's depth-sorted list of Gaussians, BLOCK Gaussians at a time:

    power = −½ (a dx² + c dy²) − b dx dy,  α = min(0.99, o·e^power), α := 0 unless power ≤ 0 and α ≥ 1/255,
    Tᵢ = Πⱼ<ᵢ (1 − αⱼ),  wᵢ = αᵢ Tᵢ,  F = Σ wᵢ gᵢ,  T_N,  and optionally M = mfᵢ* of the first i* with Tᵢ*₊₁ < ½

where g is a Gaussian's composited channels (raster.rasterize passes its features with its depth appended, so F is
the colour and the expected depth together) and mf its median channels (depth and the median features). The
background is added outside (colour + T_N·bg in torch), so its gradient is torch's.

Early termination: after each block, a tile whose every pixel has T < t_stop (raster.T_STOP, 10⁻⁴; 0 turns it off)
stops; the number of Gaussians it composited is kept per tile and the backward pass goes exactly that far, so the
gradient is the exact gradient of what the forward pass computed (the stop point, like the α cuts, is a constant).
A stopped tile differs from compositing the whole list by less than t_stop × the largest |gᵢ| of the rest (the
remaining transmittance bounds every later weight's sum). The median is unaffected (T crosses ½ before 10⁻⁴).

Backward (front to back, the per-pixel suffix sums from the forward totals rather than by dividing T_N back up,
which is unstable when T_N is tiny): with upstream gradients γ_F (per channel), γ_T and γ_M per pixel,

    ∂L/∂gᵢ = Σ_pixels wᵢ γ_F,   ∂L/∂mfᵢ = Σ_pixels [i = i*] γ_M,
    ∂L/∂αᵢ = Tᵢ (γ_F·gᵢ) − (Sᵢ + γ_T T_N) / (1 − αᵢ),   Sᵢ = Σⱼ>ᵢ wⱼ (γ_F·gⱼ) = γ_F·F − Σⱼ≤ᵢ wⱼ (γ_F·gⱼ),

then through α = o·e^power where neither cut nor the 0.99 cap is active: ∂α/∂o = e^power, ∂α/∂power = α, and
∂power/∂μ = (a dx + b dy, c dy + b dx), ∂power/∂(a, b, c) = (−½dx², −dx dy, −½dy²).

Per-Gaussian gradients are sums over (Gaussian, tile) pairs. Rather than float atomics (whose summation order varies
from run to run), the backward kernel writes each pair's gradient to its own row and a second kernel sums each
Gaussian's rows in a fixed order: the result is the same bit for bit on every run (tests/test_recon_triton.py).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .raster import ALPHA_MAX, ALPHA_MIN

BLOCK = 32                     # Gaussians per inner step (a 64 × 32 tile of pixel–Gaussian pairs in registers)
SEG_ROWS = 16                  # pair rows summed per step of the per-Gaussian reduction


@triton.jit
def _mul(a, b):
    return a * b


@triton.jit
def _matmul(x, y, SMALL: tl.constexpr):
    """x (P, K) @ y (K, N) in float32 (no TF32): a broadcast sum for small N, tl.dot (IEEE precision) otherwise."""
    if SMALL:
        return tl.sum(x[:, :, None] * y[None, :, :], axis=1)
    else:
        return tl.dot(x, y, input_precision='ieee')


@triton.jit
def _matmul_tn(x, y, SMALL: tl.constexpr):
    """xᵀ @ y for x (P, K), y (P, N) -> (K, N)."""
    if SMALL:
        return tl.sum(x[:, :, None] * y[:, None, :], axis=0)
    else:
        return tl.dot(tl.trans(x), y, input_precision='ieee')


@triton.jit
def _matmul_nt(x, y, SMALL: tl.constexpr):
    """x @ yᵀ for x (P, N), y (K, N) -> (P, K)."""
    if SMALL:
        return tl.sum(x[:, None, :] * y[None, :, :], axis=2)
    else:
        return tl.dot(x, tl.trans(y), input_precision='ieee')


@triton.jit
def _alpha(px, py, g, valid, m2, con, op, alpha_min, alpha_max):
    """The (pixels, BLOCK) α of a block (the cuts applied) and what the backward pass needs of it."""
    mx = tl.load(m2 + g * 2, mask=valid, other=0.0)
    my = tl.load(m2 + g * 2 + 1, mask=valid, other=0.0)
    a = tl.load(con + g * 3, mask=valid, other=0.0)
    b = tl.load(con + g * 3 + 1, mask=valid, other=0.0)
    c = tl.load(con + g * 3 + 2, mask=valid, other=0.0)
    o = tl.load(op + g, mask=valid, other=0.0)
    dx = px[:, None] - mx[None, :]
    dy = py[:, None] - my[None, :]
    power = -0.5 * (a[None, :] * dx * dx + c[None, :] * dy * dy) - b[None, :] * dx * dy
    gau = tl.exp(power)
    raw = o[None, :] * gau
    alpha = tl.minimum(raw, alpha_max)
    keep = (power <= 0.0) & (alpha >= alpha_min) & valid[None, :]
    alpha = tl.where(keep, alpha, 0.0)
    return alpha, keep & (raw <= alpha_max), gau, dx, dy, a, b, c


@triton.jit
def _fwd_kernel(pair_gauss, tile_start, tile_count, m2, con, op, feat, mfeat,
                out_f, out_t, out_m, n_done,
                ntx, C, M, alpha_min, alpha_max, t_stop,
                TILE: tl.constexpr, BLOCK: tl.constexpr, CP: tl.constexpr, MP: tl.constexpr,
                HAS_M: tl.constexpr, SMALL_C: tl.constexpr, SMALL_M: tl.constexpr):
    tid = tl.program_id(0)
    PIX: tl.constexpr = TILE * TILE
    p = tl.arange(0, PIX)
    px = ((tid % ntx) * TILE + p % TILE).to(tl.float32) + 0.5
    py = ((tid // ntx) * TILE + p // TILE).to(tl.float32) + 0.5
    start = tl.load(tile_start + tid)
    cnt = tl.load(tile_count + tid)
    kb = tl.arange(0, BLOCK)
    cc = tl.arange(0, CP)
    cm = tl.arange(0, MP)
    T = tl.full([PIX], 1.0, tl.float32)
    acc_f = tl.zeros([PIX, CP], tl.float32)
    acc_m = tl.zeros([PIX, MP], tl.float32)
    crossed = tl.zeros([PIX], tl.int32)
    k0 = 0
    t_max = 1.0
    while (k0 < cnt) & (t_max >= t_stop):
        kk = k0 + kb
        valid = kk < cnt
        g = tl.load(pair_gauss + start + kk, mask=valid, other=0)
        alpha, _, _, _, _, _, _, _ = _alpha(px, py, g, valid, m2, con, op, alpha_min, alpha_max)
        om = 1.0 - alpha
        incl = tl.cumprod(om, axis=1)
        t_in = T[:, None] * incl                       # T after each Gaussian
        w = alpha * (T[:, None] * (incl / om))         # αᵢ Tᵢ (Tᵢ: before it)
        f = tl.load(feat + g[:, None] * C + cc[None, :], mask=valid[:, None] & (cc[None, :] < C), other=0.0)
        acc_f += _matmul(w, f, SMALL_C)
        if HAS_M:
            below = t_in < 0.5
            first = below & (tl.cumsum(below.to(tl.int32), axis=1) == 1) & (crossed == 0)[:, None]
            mf = tl.load(mfeat + g[:, None] * M + cm[None, :], mask=valid[:, None] & (cm[None, :] < M), other=0.0)
            acc_m += _matmul(first.to(tl.float32), mf, SMALL_M)
            crossed = tl.maximum(crossed, tl.max(below.to(tl.int32), axis=1))
        T = T * tl.reduce(om, 1, _mul)
        t_max = tl.max(T, axis=0)
        k0 += BLOCK
    tl.store(n_done + tid, tl.minimum(k0, cnt))
    base = tid * PIX + p
    tl.store(out_f + base[:, None] * C + cc[None, :], acc_f, mask=cc[None, :] < C)
    tl.store(out_t + base, T)
    if HAS_M:
        tl.store(out_m + base[:, None] * M + cm[None, :], acc_m, mask=cm[None, :] < M)


@triton.jit
def _bwd_kernel(pair_gauss, tile_start, n_done, m2, con, op, feat, mfeat,
                out_f, out_t, g_f, g_t, g_m, d_pair,
                ntx, C, M, D, alpha_min, alpha_max,
                TILE: tl.constexpr, BLOCK: tl.constexpr, CP: tl.constexpr, MP: tl.constexpr,
                HAS_M: tl.constexpr, SMALL_C: tl.constexpr, SMALL_M: tl.constexpr):
    tid = tl.program_id(0)
    PIX: tl.constexpr = TILE * TILE
    p = tl.arange(0, PIX)
    px = ((tid % ntx) * TILE + p % TILE).to(tl.float32) + 0.5
    py = ((tid // ntx) * TILE + p // TILE).to(tl.float32) + 0.5
    start = tl.load(tile_start + tid)
    cnt = tl.load(n_done + tid)
    kb = tl.arange(0, BLOCK)
    cc = tl.arange(0, CP)
    cm = tl.arange(0, MP)
    base = tid * PIX + p
    cmask = cc[None, :] < C
    gf = tl.load(g_f + base[:, None] * C + cc[None, :], mask=cmask, other=0.0)
    total = tl.sum(gf * tl.load(out_f + base[:, None] * C + cc[None, :], mask=cmask, other=0.0), axis=1)
    gtt = tl.load(g_t + base) * tl.load(out_t + base)          # γ_T T_N
    if HAS_M:
        gm = tl.load(g_m + base[:, None] * M + cm[None, :], mask=cm[None, :] < M, other=0.0)
    T = tl.full([PIX], 1.0, tl.float32)
    prefix = tl.zeros([PIX], tl.float32)
    crossed = tl.zeros([PIX], tl.int32)
    for k0 in range(0, cnt, BLOCK):
        kk = k0 + kb
        valid = kk < cnt
        g = tl.load(pair_gauss + start + kk, mask=valid, other=0)
        alpha, live, gau, dx, dy, a, b, c = _alpha(px, py, g, valid, m2, con, op, alpha_min, alpha_max)
        om = 1.0 - alpha
        incl = tl.cumprod(om, axis=1)
        t_ex = T[:, None] * (incl / om)
        w = alpha * t_ex
        f = tl.load(feat + g[:, None] * C + cc[None, :], mask=valid[:, None] & (cc[None, :] < C), other=0.0)
        gdot = _matmul_nt(gf, f, SMALL_C)                      # γ_F·gᵢ per pixel
        contrib = w * gdot
        suffix = total[:, None] - (prefix[:, None] + tl.cumsum(contrib, axis=1))
        d_alpha = t_ex * gdot - (suffix + gtt[:, None]) / om
        d_alpha = tl.where(live, d_alpha, 0.0)
        d_pow = d_alpha * alpha
        d_op = tl.sum(d_alpha * gau, axis=0)
        d_mx = tl.sum(d_pow * (a[None, :] * dx + b[None, :] * dy), axis=0)
        d_my = tl.sum(d_pow * (c[None, :] * dy + b[None, :] * dx), axis=0)
        d_a = tl.sum(d_pow * (-0.5 * dx * dx), axis=0)
        d_b = tl.sum(d_pow * (-dx * dy), axis=0)
        d_c = tl.sum(d_pow * (-0.5 * dy * dy), axis=0)
        row = (start + kk) * D
        tl.store(d_pair + row, d_mx, mask=valid)
        tl.store(d_pair + row + 1, d_my, mask=valid)
        tl.store(d_pair + row + 2, d_a, mask=valid)
        tl.store(d_pair + row + 3, d_b, mask=valid)
        tl.store(d_pair + row + 4, d_c, mask=valid)
        tl.store(d_pair + row + 5, d_op, mask=valid)
        d_f = _matmul_tn(w, gf, SMALL_C)                       # (BLOCK, CP)
        tl.store(d_pair + row[:, None] + 6 + cc[None, :], d_f, mask=valid[:, None] & (cc[None, :] < C))
        if HAS_M:
            below = (T[:, None] * incl) < 0.5
            first = below & (tl.cumsum(below.to(tl.int32), axis=1) == 1) & (crossed == 0)[:, None]
            d_m = _matmul_tn(first.to(tl.float32), gm, SMALL_M)
            tl.store(d_pair + row[:, None] + 6 + C + cm[None, :], d_m, mask=valid[:, None] & (cm[None, :] < M))
            crossed = tl.maximum(crossed, tl.max(below.to(tl.int32), axis=1))
        prefix += tl.sum(contrib, axis=1)
        T = T * tl.reduce(om, 1, _mul)


@triton.jit
def _segment_sum_kernel(d_pair, rows, seg_start, seg_count, seg_gauss, out, D,
                        DP: tl.constexpr, R: tl.constexpr):
    """out[seg_gauss[s]] = Σ d_pair[rows[seg_start[s] + j]] over j < seg_count[s], always in the same order."""
    s = tl.program_id(0)
    st = tl.load(seg_start + s)
    n = tl.load(seg_count + s)
    gi = tl.load(seg_gauss + s)
    cols = tl.arange(0, DP)
    r = tl.arange(0, R)
    acc = tl.zeros([R, DP], tl.float32)
    for j in range(0, n, R):
        m = (j + r) < n
        idx = tl.load(rows + st + j + r, mask=m, other=0)
        acc += tl.load(d_pair + idx[:, None] * D + cols[None, :], mask=m[:, None] & (cols[None, :] < D), other=0.0)
    tl.store(out + gi * D + cols, tl.sum(acc, axis=0), mask=cols < D)


def _p2(n: int, lo: int) -> int:
    return max(lo, triton.next_power_of_2(max(1, n)))


def _widths(n: int) -> tuple:
    """(padded width, small?): a broadcast sum up to 8 channels, tl.dot (which needs ≥ 16) above."""
    if n <= 8:
        return _p2(n, 2), True
    return _p2(n, 16), False


class Composite(torch.autograd.Function):
    """F, T, M = Composite.apply(means2d, conic, opacity, g, mf, bins, tile, t_stop): per-tile outputs (tiles, TILE²,
    C), (tiles, TILE²), (tiles, TILE², M). bins: a dict of the binning (raster._bin's, with the per-Gaussian
    grouping). mf may be None (no median)."""

    @staticmethod
    def forward(ctx, m2, con, op, g, mf, bins, tile, t_stop):
        dev = g.device
        ntiles = bins['ntx'] * bins['nty']
        P = tile * tile
        C = g.shape[1]
        M = 0 if mf is None else mf.shape[1]
        m2, con, op, g = (t.detach().contiguous() for t in (m2, con, op, g))
        mfc = mf.detach().contiguous() if mf is not None else g          # a placeholder pointer when unused
        out_f = torch.empty(ntiles, P, C, device=dev, dtype=torch.float32)
        out_t = torch.empty(ntiles, P, device=dev, dtype=torch.float32)
        out_m = torch.empty(ntiles, P, max(M, 1), device=dev, dtype=torch.float32)
        n_done = torch.empty(ntiles, device=dev, dtype=torch.int32)
        CP, sc = _widths(C)
        MP, sm = _widths(max(M, 1))
        _fwd_kernel[(ntiles,)](bins['pair_gauss'], bins['start'], bins['count'], m2, con, op, g, mfc,
                               out_f, out_t, out_m, n_done, bins['ntx'], C, max(M, 1), ALPHA_MIN, ALPHA_MAX,
                               float(t_stop), TILE=tile, BLOCK=BLOCK, CP=CP, MP=MP, HAS_M=M > 0, SMALL_C=sc,
                               SMALL_M=sm, num_warps=4)
        ctx.save_for_backward(m2, con, op, g, mfc, out_f, out_t, n_done)
        ctx.bins, ctx.tile, ctx.M = bins, tile, M
        ctx.n = m2.shape[0]
        return out_f, out_t, (out_m[..., :M] if M else out_m[..., :0])

    @staticmethod
    def backward(ctx, gf, gt, gm):
        m2, con, op, g, mfc, out_f, out_t, n_done = ctx.saved_tensors
        bins, tile, M = ctx.bins, ctx.tile, ctx.M
        dev = g.device
        C = g.shape[1]
        ntiles = bins['ntx'] * bins['nty']
        P = tile * tile
        gf = torch.zeros_like(out_f) if gf is None else gf.contiguous()
        gt = torch.zeros_like(out_t) if gt is None else gt.contiguous()
        if M:
            gm = torch.zeros(ntiles, P, M, device=dev) if gm is None else gm.contiguous()
        else:
            gm = gf                                                     # unused
        D = 6 + C + M
        npairs = len(bins['pair_gauss'])
        d_pair = torch.zeros(max(npairs, 1), D, device=dev, dtype=torch.float32)
        CP, sc = _widths(C)
        MP, sm = _widths(max(M, 1))
        _bwd_kernel[(ntiles,)](bins['pair_gauss'], bins['start'], n_done, m2, con, op, g, mfc,
                               out_f, out_t, gf, gt, gm, d_pair, bins['ntx'], C, max(M, 1), D, ALPHA_MIN,
                               ALPHA_MAX, TILE=tile, BLOCK=BLOCK, CP=CP, MP=MP, HAS_M=M > 0, SMALL_C=sc,
                               SMALL_M=sm, num_warps=4)
        out = torch.zeros(ctx.n, D, device=dev, dtype=torch.float32)
        nseg = len(bins['seg_gauss'])
        if nseg:
            _segment_sum_kernel[(nseg,)](d_pair, bins['seg_rows'], bins['seg_start'], bins['seg_count'],
                                         bins['seg_gauss'], out, D, DP=triton.next_power_of_2(D), R=SEG_ROWS,
                                         num_warps=1)
        d_mf = out[:, 6 + C:] if M else None
        return out[:, 0:2], out[:, 2:5], out[:, 5], out[:, 6:6 + C], d_mf, None, None, None
