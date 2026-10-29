"""The Gaussians of a reconstruction: parameters, initialisation from points, the standard 3DGS .ply layout, adaptive
density control and the optimiser (torch at import: this module is never imported by the GUI at start-up).

Parameters (as the 3DGS paper optimises them): means (N, 3) in metres; log-scales (N, 3); quaternions (N, 4: w, x,
y, z, unnormalised); opacity logits (N); spherical-harmonics coefficients split into the DC term sh_dc (N, 1, 3) and
the rest sh_rest (N, (D+1)² − 1, 3). Beside them, not optimised: ``sky`` (bool, N), the Gaussians placed on the far
shell (core.recon.data), which density control leaves alone (they are large by design), and ``votes`` (N, classes)
when semantic classes were voted (core.recon.train.vote_semantics).

The .ply (write_ply / read_ply) is the layout every 3DGS viewer reads: binary little-endian, one ``vertex`` element of
float properties x y z nx ny nz f_dc_0..2 f_rest_0..(3·((D+1)²−1)−1) opacity scale_0..2 rot_0..3; f_rest is channel
major (all of red's coefficients, then green's, then blue's), opacity a logit, scales logs, rot (w, x, y, z)
unnormalised; normals are written as zeros. The coordinates are the reconstruction's world frame, right-handed with z
up (docs/reconstruction.md, "Frames"). ``sky`` and ``votes`` go in extras.npz beside it.

Adaptive density control is the paper's (Kerbl et al. 2023, §5.2): every interval steps, a Gaussian whose
view-space positional gradient averaged over the views that saw it exceeds grad_threshold (in the paper's units: the
gradient with respect to normalised device coordinates, i.e. the pixel gradient × half the image size) is cloned
when small (largest scale ≤ percent_dense × the scene's extent) or split into two when large (positions drawn from
the Gaussian, scales ÷ 1.6); Gaussians with opacity below prune_opacity are removed, and after the first opacity
reset those larger than max_screen_px on screen or than 0.1 × the extent in the world; every opacity_reset steps the
opacities are capped at 0.01. Above max_gaussians no more are added (the largest gradients are densified first), so a
scene's memory stays bounded — a cap the paper does not have.

The optimiser is Adam (Kingma and Ba 2015) written here over plain tensors, so density control can grow and shrink its
moment estimates with the parameters (new rows start at zero moments).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

PARAMS = ('means', 'log_scales', 'quats', 'opacity_logits', 'sh_dc', 'sh_rest')


def inverse_sigmoid(x):
    x = torch.as_tensor(x).clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


def rgb_to_sh0(rgb):
    from .raster import SH_C0
    return (torch.as_tensor(rgb) - 0.5) / SH_C0


class Gaussians:
    """The parameters (requires_grad leaves) and per-Gaussian extras of one reconstruction."""

    def __init__(self, means, log_scales, quats, opacity_logits, sh_dc, sh_rest, sky=None, votes=None):
        self.p = {'means': means, 'log_scales': log_scales, 'quats': quats, 'opacity_logits': opacity_logits,
                  'sh_dc': sh_dc, 'sh_rest': sh_rest}
        n = means.shape[0]
        self.sky = torch.zeros(n, dtype=torch.bool, device=means.device) if sky is None else sky.to(means.device)
        self.votes = votes

    # ------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_points(cls, xyz, rgb, sh_degree: int = 3, sky=None, opacity: float = 0.1, device='cpu',
                    dtype=torch.float32, min_scale: float = 1e-3, scale_mult=None):
        """Gaussians at points (N, 3) with colours (N, 3) in [0, 1]: isotropic, each scale the root of the mean
        squared distance to its three nearest neighbours (the paper's initialisation), opacity 0.1, identity
        rotation, SH DC from the colour. scale_mult (N,) multiplies the scales (the far shell's)."""
        from scipy.spatial import cKDTree
        xyz = np.asarray(xyz, np.float64)
        n = len(xyz)
        if n == 0:
            raise ValueError('no points to start the Gaussians from')
        k = min(4, n)
        d, _ = cKDTree(xyz).query(xyz, k=k)
        d2 = np.mean(np.square(d[:, 1:]), axis=1) if k > 1 else np.ones(n)
        s = np.sqrt(np.maximum(d2, min_scale ** 2))
        if scale_mult is not None:
            s = s * np.asarray(scale_mult, np.float64)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        nrest = (sh_degree + 1) ** 2 - 1
        g = cls(t(xyz).requires_grad_(), t(np.log(s))[:, None].repeat(1, 3).requires_grad_(),
                t(np.tile([1.0, 0, 0, 0], (n, 1))).requires_grad_(),
                inverse_sigmoid(torch.full((n,), opacity, dtype=dtype)).to(device).requires_grad_(),
                rgb_to_sh0(t(np.clip(rgb, 0, 1)))[:, None, :].contiguous().requires_grad_(),
                torch.zeros(n, nrest, 3, dtype=dtype, device=device).requires_grad_(),
                None if sky is None else torch.as_tensor(np.asarray(sky, bool), device=device))
        return g

    # ------------------------------------------------------------------------------------------- views
    def __len__(self):
        return self.p['means'].shape[0]

    @property
    def device(self):
        return self.p['means'].device

    @property
    def sh_degree(self) -> int:
        return int(round(math.sqrt(self.p['sh_rest'].shape[1] + 1))) - 1

    def scales(self):
        return torch.exp(self.p['log_scales'])

    def opacities(self):
        return torch.sigmoid(self.p['opacity_logits'])

    def sh(self):
        return torch.cat([self.p['sh_dc'], self.p['sh_rest']], 1)

    def render(self, viewmat, K, width, height, sh_degree=None, background=None, features=None, **kw):
        """raster.render with these Gaussians: SH colour (up to sh_degree, default all), or `features` (N, C)."""
        from .raster import render
        if features is not None:
            return render(self.p['means'], self.scales(), self.p['quats'], self.opacities(), features, viewmat, K,
                          width, height, background, None, **kw)
        deg = self.sh_degree if sh_degree is None else min(int(sh_degree), self.sh_degree)
        return render(self.p['means'], self.scales(), self.p['quats'], self.opacities(), self.sh(), viewmat, K,
                      width, height, background, deg, **kw)

    def detach_copy(self) -> 'Gaussians':
        return Gaussians(*(self.p[k].detach().clone() for k in PARAMS), self.sky.clone(),
                         None if self.votes is None else self.votes.clone())

    def to(self, device) -> 'Gaussians':
        return Gaussians(*(self.p[k].detach().to(device) for k in PARAMS), self.sky.to(device),
                         None if self.votes is None else self.votes.to(device))

    # ------------------------------------------------------------------------------------------- .ply
    def write_ply(self, path: str):
        """The standard 3DGS layout (module doc); extras.npz (sky, votes) beside it."""
        from ..paths import atomic_write
        with torch.no_grad():
            n = len(self)
            xyz = self.p['means'].detach().cpu().numpy().astype(np.float32)
            dc = self.p['sh_dc'].detach().cpu().numpy().reshape(n, 3).astype(np.float32)
            rest = self.p['sh_rest'].detach().cpu().numpy().transpose(0, 2, 1).reshape(n, -1).astype(np.float32)
            op = self.p['opacity_logits'].detach().cpu().numpy().reshape(n, 1).astype(np.float32)
            sc = self.p['log_scales'].detach().cpu().numpy().astype(np.float32)
            rot = self.p['quats'].detach().cpu().numpy().astype(np.float32)
        names = ply_properties(self.sh_degree)
        data = np.concatenate([xyz, np.zeros((n, 3), np.float32), dc, rest, op, sc, rot], 1)
        assert data.shape[1] == len(names)
        header = 'ply\nformat binary_little_endian 1.0\n' \
                 f'comment Falcon Autonomy reconstruction (3D Gaussian Splatting layout)\nelement vertex {n}\n' + \
                 ''.join(f'property float {k}\n' for k in names) + 'end_header\n'
        atomic_write(path, header.encode('ascii') + np.ascontiguousarray(data, '<f4').tobytes())
        extras = {'sky': self.sky.cpu().numpy()}
        if self.votes is not None:
            extras['votes'] = self.votes.cpu().numpy().astype(np.float32)
        xp = os.path.join(os.path.dirname(os.path.abspath(path)), 'extras.npz')
        tmp = xp + f'.{os.getpid()}.tmp.npz'
        np.savez_compressed(tmp, **extras)
        os.replace(tmp, xp)

    @classmethod
    def read_ply(cls, path: str, device='cpu', dtype=torch.float32) -> 'Gaussians':
        names, data = read_ply_arrays(path)
        col = {k: i for i, k in enumerate(names)}
        n = data.shape[0]
        nrest = sum(1 for k in names if k.startswith('f_rest_'))
        if nrest % 3:
            raise ValueError(f'{path}: {nrest} f_rest properties is not a multiple of 3')
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)  # noqa: E731
        g = lambda *ks: data[:, [col[k] for k in ks]]  # noqa: E731
        rest = data[:, [col[f'f_rest_{i}'] for i in range(nrest)]].reshape(n, 3, nrest // 3).transpose(0, 2, 1) \
            if nrest else np.zeros((n, 0, 3), np.float32)
        xp = os.path.join(os.path.dirname(os.path.abspath(path)), 'extras.npz')
        sky = votes = None
        if os.path.isfile(xp):
            with np.load(xp) as z:
                if 'sky' in z and len(z['sky']) == n:
                    sky = torch.as_tensor(z['sky'].astype(bool))
                if 'votes' in z and len(z['votes']) == n:
                    votes = torch.as_tensor(z['votes'], device=device)
        return cls(t(g('x', 'y', 'z')), t(g('scale_0', 'scale_1', 'scale_2')),
                   t(g('rot_0', 'rot_1', 'rot_2', 'rot_3')), t(data[:, col['opacity']]),
                   t(g('f_dc_0', 'f_dc_1', 'f_dc_2')).reshape(n, 1, 3), t(rest), sky, votes)


def ply_properties(sh_degree: int) -> list:
    nrest = 3 * ((sh_degree + 1) ** 2 - 1)
    return (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(nrest)] +
            ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])


def read_ply_arrays(path: str):
    """(property names, float32 array (N, P)) of a binary little-endian PLY whose one vertex element holds floats only
    (the 3DGS layout)."""
    with open(path, 'rb') as fh:
        if fh.readline().strip() != b'ply':
            raise ValueError(f'{path}: not a PLY file')
        names, n, fmt = [], None, None
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f'{path}: the header has no end_header')
            parts = line.decode('ascii', 'replace').split()
            if not parts:
                continue
            if parts[0] == 'format':
                fmt = parts[1]
            elif parts[0] == 'element':
                if parts[1] != 'vertex' or n is not None:
                    raise ValueError(f'{path}: only one vertex element is read (found {parts[1]})')
                n = int(parts[2])
            elif parts[0] == 'property':
                if parts[1] != 'float':
                    raise ValueError(f'{path}: property {parts[-1]} is {parts[1]}, only float is read')
                names.append(parts[2])
            elif parts[0] == 'end_header':
                break
        if fmt != 'binary_little_endian' or n is None:
            raise ValueError(f'{path}: not a binary little-endian vertex PLY')
        data = np.frombuffer(fh.read(4 * n * len(names)), '<f4')
    if data.size != n * len(names):
        raise ValueError(f'{path}: truncated ({data.size} of {n * len(names)} values)')
    return names, data.reshape(n, len(names)).astype(np.float32)


# ================================================================================================ the optimiser
class Adam:
    """Adam over the Gaussians' parameters, one learning rate per parameter group (lrs: {name: lr})."""

    def __init__(self, gs: Gaussians, lrs: dict, betas=(0.9, 0.999), eps=1e-15):
        self.gs, self.lrs, self.betas, self.eps = gs, dict(lrs), betas, eps
        self.m = {k: torch.zeros_like(v) for k, v in gs.p.items()}
        self.v = {k: torch.zeros_like(v) for k, v in gs.p.items()}
        self.t = {k: 0 for k in gs.p}

    def zero_grad(self):
        for v in self.gs.p.values():
            v.grad = None

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for k, p in self.gs.p.items():
            if p.grad is None or self.lrs.get(k, 0.0) == 0.0:
                continue
            self.t[k] += 1
            t = self.t[k]
            self.m[k].mul_(b1).add_(p.grad, alpha=1 - b1)
            self.v[k].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            mh = self.m[k] / (1 - b1 ** t)
            vh = self.v[k] / (1 - b2 ** t)
            p.sub_(self.lrs[k] * mh / (vh.sqrt() + self.eps))

    def state_dict(self) -> dict:
        return {'m': self.m, 'v': self.v, 't': self.t, 'lrs': self.lrs}

    def load_state_dict(self, d: dict):
        self.m, self.v, self.t, self.lrs = d['m'], d['v'], dict(d['t']), dict(d['lrs'])

    # rows: keep a mask, or append new rows with zero moments
    def _apply(self, fn_param, fn_state):
        new = {}
        for k, p in self.gs.p.items():
            new[k] = fn_param(k, p.detach()).requires_grad_()
            self.m[k] = fn_state(self.m[k])
            self.v[k] = fn_state(self.v[k])
        self.gs.p = new

    def keep(self, mask: torch.Tensor):
        self._apply(lambda k, p: p[mask].contiguous(), lambda s: s[mask].contiguous())
        self.gs.sky = self.gs.sky[mask]
        if self.gs.votes is not None:
            self.gs.votes = self.gs.votes[mask]

    def append(self, rows: dict, sky: torch.Tensor):
        self._apply(lambda k, p: torch.cat([p, rows[k].to(p.dtype)], 0),
                    lambda s: torch.cat([s, torch.zeros((len(sky),) + s.shape[1:], dtype=s.dtype,
                                                        device=s.device)], 0))
        self.gs.sky = torch.cat([self.gs.sky, sky])
        if self.gs.votes is not None:
            self.gs.votes = torch.cat([self.gs.votes, torch.zeros(len(sky), self.gs.votes.shape[1],
                                                                  device=self.gs.votes.device)])


# ================================================================================================ density control
class Densifier:
    """The paper's adaptive density control (module doc). Call add_stats() after every backward pass and step() at
    the interval."""

    def __init__(self, gs: Gaussians, extent: float, grad_threshold=2e-4, percent_dense=0.01, prune_opacity=0.005,
                 max_screen_px=20.0, max_gaussians=1_000_000, generator=None):
        self.extent, self.thr, self.pd = float(extent), float(grad_threshold), float(percent_dense)
        self.prune_opacity, self.max_screen_px, self.max_n = float(prune_opacity), float(max_screen_px), \
            int(max_gaussians)
        self.gen = generator
        self.reset(gs)
        self.after_reset = False

    def reset(self, gs: Gaussians):
        n, dev = len(gs), gs.device
        self.grad_acc = torch.zeros(n, device=dev)
        self.seen = torch.zeros(n, device=dev)
        self.max_radius = torch.zeros(n, device=dev)

    @torch.no_grad()
    def add_stats(self, means2d_grad: torch.Tensor, visible: torch.Tensor, radius: torch.Tensor, width: int,
                  height: int):
        g = means2d_grad[visible] * torch.tensor([0.5 * width, 0.5 * height], device=means2d_grad.device,
                                                 dtype=means2d_grad.dtype)
        self.grad_acc[visible] += g.norm(dim=-1).to(self.grad_acc.dtype)
        self.seen[visible] += 1
        self.max_radius[visible] = torch.maximum(self.max_radius[visible], radius[visible].to(self.max_radius.dtype))

    @torch.no_grad()
    def step(self, opt: Adam) -> dict:
        gs = opt.gs
        n0 = len(gs)
        grads = self.grad_acc / self.seen.clamp_min(1)
        big = gs.scales().max(-1).values > self.pd * self.extent
        cand = (grads >= self.thr) & ~gs.sky
        room = max(0, self.max_n - n0)
        clone = cand & ~big
        split = cand & big
        # under the cap, the largest gradients first (a split adds one net Gaussian, a clone one)
        want = int(clone.sum() + split.sum())
        if want > room:
            idx = torch.nonzero(cand, as_tuple=False).squeeze(1)
            top = idx[torch.argsort(grads[idx], descending=True)[:room]]
            keep_c = torch.zeros_like(cand)
            keep_c[top] = True
            clone, split = clone & keep_c, split & keep_c
        p = {k: v.detach() for k, v in gs.p.items()}
        rows = {k: [p[k][clone]] for k in PARAMS}
        sky_rows = [gs.sky[clone]]
        ns = int(split.sum())
        if ns:
            from .raster import quat_to_rotmat
            s = gs.scales()[split]
            R = quat_to_rotmat(p['quats'][split])
            for _ in range(2):
                z = torch.randn(ns, 3, device=gs.device, generator=self.gen, dtype=s.dtype)
                rows['means'].append(p['means'][split] + (R @ (s * z)[..., None])[..., 0])
                rows['log_scales'].append(torch.log(s / 1.6))
                for k in ('quats', 'opacity_logits', 'sh_dc', 'sh_rest'):
                    rows[k].append(p[k][split])
                sky_rows.append(gs.sky[split])
        n_new = sum(len(r) for r in rows['means'])
        if n_new:
            opt.append({k: torch.cat(v, 0) for k, v in rows.items()}, torch.cat(sky_rows))
        # the split originals go; then pruning
        keep = torch.ones(len(gs), dtype=torch.bool, device=gs.device)
        keep[:n0] &= ~split
        op = gs.opacities()
        prune = op < self.prune_opacity
        if self.after_reset:
            mr = torch.cat([self.max_radius, torch.zeros(len(gs) - n0, device=gs.device)])
            too_big = (mr > self.max_screen_px) | (gs.scales().max(-1).values > 0.1 * self.extent)
            prune |= too_big & ~gs.sky
        keep &= ~prune
        opt.keep(keep)
        self.reset(gs)
        return {'cloned': int(clone.sum()), 'split': ns, 'pruned': int((~keep).sum()) - ns, 'n': len(gs)}

    @torch.no_grad()
    def reset_opacity(self, opt: Adam, value: float = 0.01):
        gs = opt.gs
        lim = float(inverse_sigmoid(torch.tensor(value)))
        gs.p['opacity_logits'].data.clamp_(max=lim)
        opt.m['opacity_logits'].zero_()
        opt.v['opacity_logits'].zero_()
        self.after_reset = True
