"""The networks and losses of learned sim-to-real image translation (core.translate): Falcon's own PyTorch
implementation of CUT — contrastive unpaired translation (Park, Efros, Zhang and Zhu, "Contrastive Learning for
Unpaired Image-to-Image Translation", ECCV 2020) — with a geometry-preserving generator. Written for Falcon (GPL-3.0-
or-later); the architecture follows the paper and its BSD-licensed reference implementation's design (a ResNet
generator after Johnson et al. 2016, a 70 × 70 PatchGAN discriminator after Isola et al. 2017, least-squares GAN
losses after Mao et al. 2017, the PatchNCE loss with a two-layer projection head per layer); no code was copied.

This module imports torch at the top: core.translate imports it only inside the functions that train or run a model.

Generator (ResnetGenerator):

    input   RGB in [-1, 1] and, when conditioned, K one-hot channels of semantic groups (Falcon's labels of the frame;
            all zero for an unlabelled image: "unknown")
    stem    reflection pad 3, conv 7 × 7 → ngf, InstanceNorm, ReLU
    down    2 × (conv 3 × 3 stride 2, InstanceNorm, ReLU)       ngf → 2 ngf → 4 ngf
    blocks  n_blocks residual blocks (reflection pad, conv 3 × 3, InstanceNorm, ReLU, pad, conv, InstanceNorm)
    up      2 × (nearest-neighbour upsampling × 2, conv 3 × 3, InstanceNorm, ReLU)   — not transposed convolutions,
            whose uneven overlap prints a checkerboard of false edges (Odena et al. 2016)
    head    reflection pad 3, conv 7 × 7 → 3, initialised to zero
    output  tanh(artanh(x) + head): a residual in the logit of the input, so the untrained generator is the identity
            and training moves appearance away from it; no spatial warping, deformable or flow layer anywhere, and
            the receptive field is local (no global pooling or attention), so an output pixel is made from the
            input's neighbourhood at the same place.

Feature taps for PatchNCE (encoder_features): the (padded) input, the stem, the two downsampling stages and the middle
residual block — as CUT takes five layers from the pixels to the middle of the residual blocks.

PatchNCE: for each tap, num_patches locations are drawn at random (the same for the input and the output), the
features there go through the tap's projection head (Linear, ReLU, Linear → 256, L2-normalised) and each output
patch must pick its own input patch among the image's other sampled patches (InfoNCE, temperature 0.07, negatives
from the same image). That is what ties every output patch to the content at the same place in the input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c):
    return nn.InstanceNorm2d(c, affine=False, track_running_stats=False)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), _norm(c), nn.ReLU(True),
                                  nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), _norm(c))

    def forward(self, x):
        return x + self.body(x)


class ResnetGenerator(nn.Module):
    def __init__(self, cond_ch: int = 0, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        self.cond_ch, self.ngf, self.n_blocks = int(cond_ch), int(ngf), int(n_blocks)
        cin = 3 + self.cond_ch
        self.pad = nn.ReflectionPad2d(3)
        self.stem = nn.Sequential(nn.Conv2d(cin, ngf, 7), _norm(ngf), nn.ReLU(True))
        self.down1 = nn.Sequential(nn.Conv2d(ngf, 2 * ngf, 3, 2, 1), _norm(2 * ngf), nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(2 * ngf, 4 * ngf, 3, 2, 1), _norm(4 * ngf), nn.ReLU(True))
        self.blocks = nn.ModuleList(ResBlock(4 * ngf) for _ in range(self.n_blocks))
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(4 * ngf, 2 * ngf, 3, 1, 1), _norm(2 * ngf),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(2 * ngf, ngf, 3, 1, 1), _norm(ngf), nn.ReLU(True))
        self.head = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7))
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        self.mid = self.n_blocks // 2           # the residual block whose output is the last NCE tap

    def _input(self, x, cond):
        if self.cond_ch:
            if cond is None:
                cond = x.new_zeros((x.shape[0], self.cond_ch) + tuple(x.shape[2:]))
            x = torch.cat([x, cond], 1)
        return x

    def forward(self, x, cond=None):
        h = self.down2(self.down1(self.stem(self.pad(self._input(x, cond)))))
        for b in self.blocks:
            h = b(h)
        h = self.up(h)
        H, W = x.shape[2:]
        h = self.head(h)[:, :, :H, :W]
        # odd sizes: the encoder floors, the decoder doubles; pad what is missing at the far edges
        if h.shape[2] != H or h.shape[3] != W:
            h = F.pad(h, (0, W - h.shape[3], 0, H - h.shape[2]), mode='replicate')
        return torch.tanh(torch.atanh(x.clamp(-0.999, 0.999)) + h)

    def encoder_features(self, x, cond=None) -> list:
        """The NCE taps: padded input, stem, down1, down2, the middle residual block."""
        feats = []
        h = self.pad(self._input(x, cond))
        feats.append(h)
        h = self.stem(h)
        feats.append(h)
        h = self.down1(h)
        feats.append(h)
        h = self.down2(h)
        feats.append(h)
        for i, b in enumerate(self.blocks):
            h = b(h)
            if i == self.mid:
                feats.append(h)
                break
        return feats

    def tap_channels(self) -> list:
        return [3 + self.cond_ch, self.ngf, 2 * self.ngf, 4 * self.ngf, 4 * self.ngf]


class PatchDiscriminator(nn.Module):
    """70 × 70 PatchGAN (3 strided layers): a real/fake score per overlapping patch; least-squares loss."""

    def __init__(self, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        seq = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        m = 1
        for n in range(1, n_layers):
            m0, m = m, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * m0, ndf * m, 4, 2, 1), _norm(ndf * m), nn.LeakyReLU(0.2, True)]
        m0, m = m, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * m0, ndf * m, 4, 1, 1), _norm(ndf * m), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * m, 1, 4, 1, 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class PatchSampler(nn.Module):
    """CUT's projection heads: per tap a two-layer MLP to nc dimensions, applied to num_patches sampled locations."""

    def __init__(self, channels: list, nc: int = 256):
        super().__init__()
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(c, nc), nn.ReLU(True), nn.Linear(nc, nc)) for c in channels)
        for m in self.mlps.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, feats: list, num_patches: int = 256, ids: list | None = None, generator=None):
        out, out_ids = [], []
        for k, f in enumerate(feats):
            B, C, H, W = f.shape
            flat = f.permute(0, 2, 3, 1).reshape(B, H * W, C)
            if ids is None:
                n = min(num_patches, H * W)
                pid = torch.randperm(H * W, generator=generator)[:n].to(f.device)      # a CPU generator: seeded
            else:
                pid = ids[k]
            x = flat[:, pid, :].reshape(-1, C)                     # (B · n, C)
            x = self.mlps[k](x)
            out.append(F.normalize(x, dim=1).reshape(B, -1, x.shape[-1]))
            out_ids.append(pid)
        return out, out_ids


def patch_nce(q: torch.Tensor, k: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """InfoNCE per patch: q (B, n, D) from the output, k (B, n, D) from the input (detached). Each q's positive is the
    k at the same index; negatives are the other n − 1 patches of the same image. Mean over patches."""
    k = k.detach()
    B, n, _ = q.shape
    l_pos = (q * k).sum(-1, keepdim=True)                           # (B, n, 1)
    l_neg = torch.bmm(q, k.transpose(1, 2))                         # (B, n, n)
    eye = torch.eye(n, device=q.device, dtype=torch.bool)[None]
    l_neg = l_neg.masked_fill(eye, -10.0)                           # a patch is not its own negative
    logits = torch.cat([l_pos, l_neg], 2).reshape(B * n, n + 1) / tau
    return F.cross_entropy(logits, torch.zeros(B * n, dtype=torch.long, device=q.device))


def nce_loss(G: ResnetGenerator, Fs: PatchSampler, src, cond, out, num_patches: int, tau: float = 0.07,
             generator=None) -> torch.Tensor:
    """PatchNCE between an input and its translation: the output's patches (through the same encoder, with the same
    condition) must match the input's at the same places."""
    fk = G.encoder_features(src, cond)
    fq = G.encoder_features(out, cond)
    pk, ids = Fs(fk, num_patches, generator=generator)
    pq, _ = Fs(fq, num_patches, ids)
    return sum(patch_nce(q, k, tau) for q, k in zip(pq, pk)) / len(pq)


def lsgan(pred: torch.Tensor, real: bool) -> torch.Tensor:
    return F.mse_loss(pred, torch.full_like(pred, 1.0 if real else 0.0))
