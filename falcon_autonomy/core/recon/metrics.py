"""Image and depth measures for reconstructions (torch at import).

* PSNR = 10 log₁₀(1 / MSE) over the pixels a mask keeps, images in [0, 1].
* SSIM (Wang, Bovik, Sheikh and Simoncelli, "Image Quality Assessment: From Error Visibility to Structural
  Similarity", IEEE TIP 2004): an 11 × 11 Gaussian window of σ = 1.5, C₁ = (0.01)², C₂ = (0.03)², per channel; the
  map is computed where the whole window lies inside the image (5 px are lost at each border) and averaged over the
  pixels the mask keeps there. With no mask this is scikit-image's structural_similarity(gaussian_weights=True,
  sigma=1.5, use_sample_covariance=False, data_range=1) averaged over channels (tests/test_recon.py checks it). The
  3DGS paper's training loss uses the same window with zero padding instead (ssim_loss here): numbers from the two
  differ slightly at the borders.
* LPIPS is not computed: it needs a learned network and its calibration weights, and none is pinned in Falcon.
* Depth: the reconstruction's range along each pixel's ray (core.recon.scene.render_views: the median depth's)
  against a reference range (CARLA's range labels): the median absolute relative error, the shares within 5 % and
  10 %, and the median absolute error in metres.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _window(dtype, device, size=11, sigma=1.5):
    x = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def _filt(x, g, pad):
    """Separable Gaussian filter of (B, C, H, W); pad: zero padding (0: valid)."""
    C = x.shape[1]
    k = len(g)
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(C, 1, 1, k), padding=(0, pad), groups=C)
    return F.conv2d(x, g.view(1, 1, k, 1).expand(C, 1, k, 1), padding=(pad, 0), groups=C)


def ssim_map(a: torch.Tensor, b: torch.Tensor, pad: int = 0) -> torch.Tensor:
    """SSIM map of (H, W, C) images in [0, 1]: (H', W', C) — valid region with pad 0, the same size with pad 5."""
    g = _window(a.dtype, a.device)
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mx, my = _filt(x, g, pad), _filt(y, g, pad)
    sxx = _filt(x * x, g, pad) - mx * mx
    syy = _filt(y * y, g, pad) - my * my
    sxy = _filt(x * y, g, pad) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return m[0].permute(1, 2, 0)


def ssim(a, b, mask=None) -> float:
    """Mean SSIM (module doc) over the valid region's masked pixels."""
    m = ssim_map(a, b)
    if mask is not None:
        mk = mask[5:-5, 5:-5]
        return float(m[mk].mean()) if mk.any() else float('nan')
    return float(m.mean())


def ssim_loss(a, b, mask=None) -> torch.Tensor:
    """1 − SSIM with zero padding (the training loss), averaged over the masked pixels."""
    m = ssim_map(a, b, pad=5).mean(-1)
    if mask is not None:
        return 1.0 - (m * mask).sum() / mask.sum().clamp_min(1)
    return 1.0 - m.mean()


def psnr(a, b, mask=None) -> float:
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    mse = float(d.mean()) if d.numel() else float('nan')
    return float('inf') if mse == 0 else 10.0 * math.log10(1.0 / mse)


def depth_errors(pred_range: torch.Tensor, ref_range: torch.Tensor, valid: torch.Tensor) -> dict:
    """Range against a reference over valid pixels: median absolute relative error, shares within 5 % and 10 %."""
    p, r = pred_range[valid], ref_range[valid]
    if not p.numel():
        return {'pixels': 0}
    rel = (p - r).abs() / r
    return {'pixels': int(p.numel()), 'median_abs_rel': round(float(rel.median()), 5),
            'within_5pc': round(float((rel < 0.05).float().mean()), 5),
            'within_10pc': round(float((rel < 0.10).float().mean()), 5),
            'median_abs_m': round(float((p - r).abs().median()), 4)}
