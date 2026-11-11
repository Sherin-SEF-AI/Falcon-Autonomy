"""Gaussians through any of core.lens's camera models — a fisheye far beyond 90° included — and through a lidar's
spherical beam grid (torch at import).

Two ways to render a fisheye, both differentiable:

(a) **cube faces, the reference** (render_cube): the camera's rays are planned into pinhole views exactly as CARLA
    renders a fisheye (core.lens.plan_views: one view within 60° of the axis, else the cube faces the image needs,
    each 1.5° past its seams), each view rendered by the pinhole rasteriser (core.recon.raster.render: EWA projection,
    which is exact for a pinhole up to its affine approximation), and the image read from the views through the lens
    model's remap tables (core.lens.remap_tables), bilinearly, by torch's grid_sample so the gradient flows back.
    Its error is the pinhole rasteriser's plus a bilinear resampling; a view's own field is at most 93° (a face), so
    the affine approximation never sees the fisheye's rim.
(b) **the lens fast path** (render_lens), ported from Vantage core/splat_raster.py (LensProjector, splat): every
    Gaussian's centre goes through the lens model itself, its covariance through the projection's Jacobian at the
    centre (central differences, step 10⁻³ of the range), composited front to back in range order (not depth: a
    fisheye sees behind its own image plane) on 16 × 16 tiles. One pass, no views; the local affine approximation is
    poor for a Gaussian spanning a large angle — near the camera, or towards the rim of a lens beyond ~150° where
    r(θ) bends — so (b) is an approximation of (a) there. tests/test_recon_dynamic.py measures (a) against (b): equal
    in the centre of the field, apart towards the rim.

The lidar uses (b)'s rasteriser with SphereProjector (core.recon.lidar).

Coordinates: a camera's optical frame (x right, y down, z forward) and pixel-edge image coordinates (pixel i covers
[i, i + 1)), as core.lens and core.recon.raster. Gaussians: means (N, 3) world, linear scales (N, 3), quaternions
(N, 4) w x y z, opacities (N,), features (N, C); viewmat world → optical.
"""
from __future__ import annotations

import math

import numpy as np
import torch

TILE = 16
LOWPASS = 0.3                       # px², added to every projected covariance (3DGS's dilation)
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_STOP = 1e-4


# =========================================================================== projections (Vantage core/splat_raster.py)
def _r_norm(theta, projection: str, kb):
    """Normalised image height r(θ)/f of a projection (core.lens.r_of_theta with f = 1), in torch."""
    if projection == 'rectilinear':
        return torch.tan(torch.clamp(theta, max=math.radians(89.9)))
    if projection == 'equidistant':
        return theta
    if projection == 'equisolid':
        return 2 * torch.sin(theta / 2)
    if projection == 'stereographic':
        return 2 * torch.tan(torch.clamp(theta, max=math.radians(179.0)) / 2)
    if projection == 'orthographic':
        return torch.sin(torch.clamp(theta, max=math.pi / 2))
    if projection == 'kannala_brandt':
        k1, k2, k3, k4 = (list(kb) + [0.0] * 4)[:4]
        t2 = theta * theta
        return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    raise ValueError(projection)


class LensProjector:
    """A camera's lens model on its image: optical-frame points -> (u, v, ok), pixel-edge coordinates; ok where
    θ ≤ max_theta (and in front of a rectilinear lens). Ported from Vantage core/splat_raster.py."""

    def __init__(self, fx, fy, cx, cy, projection, kb=(0.0, 0.0, 0.0, 0.0), max_theta=math.pi):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.projection = str(projection)
        self.kb = tuple(float(k) for k in (list(kb) + [0.0] * 4)[:4])
        self.max_theta = float(max_theta)

    @classmethod
    def of(cls, intr, width: int | None = None, height: int | None = None) -> 'LensProjector':
        """The projector of a core.lens.Intrinsics at an image of width × height (default its own size)."""
        sx = (width or intr.width) / intr.width
        sy = (height or intr.height) / intr.height
        return cls(intr.f_px * sx, intr.f_px * intr.aspect * sy, intr.cx * sx, intr.cy * sy, intr.model, intr.k,
                   intr.max_theta)

    def __call__(self, P):
        x, y, z = P.unbind(-1)
        rho = torch.sqrt(x * x + y * y + 1e-18)
        th = torch.atan2(rho, z)
        s = _r_norm(th, self.projection, self.kb) / rho
        u = self.cx + self.fx * x * s
        v = self.cy + self.fy * y * s
        ok = th <= self.max_theta
        if self.projection == 'rectilinear':
            ok = ok & (z > 1e-6)
        return u, v, ok


class SphereProjector:
    """A lidar's beam grid as an image (Vantage core/splat_raster.py): sensor-frame points (x forward, y left, z up)
    -> column u = (a − a_lo) / a_step, row v = (e_hi − e) / e_step, a the azimuth (degrees, counter-clockwise from
    +x) relative to az_centre wrapped to ±180°, e the elevation. Pixel (i, j)'s centre looks along azimuth a_lo +
    (i + ½)·a_step, elevation e_hi − (j + ½)·e_step."""

    def __init__(self, az_centre, a_lo, a_step, e_hi, e_step, width, height):
        self.az_centre, self.a_lo, self.a_step = float(az_centre), float(a_lo), float(a_step)
        self.e_hi, self.e_step = float(e_hi), float(e_step)
        self.width, self.height = int(width), int(height)

    def __call__(self, P):
        x, y, z = P.unbind(-1)
        h = torch.sqrt(x * x + y * y + 1e-18)
        a = torch.rad2deg(torch.atan2(y, x)) - self.az_centre
        a = torch.remainder(a + 180.0, 360.0) - 180.0
        e = torch.rad2deg(torch.atan2(z, h))
        u = (a - self.a_lo) / self.a_step
        v = (self.e_hi - e) / self.e_step
        margin = 0.25 * self.width
        ok = (e.abs() < 85.0) & (u > -margin) & (u < self.width + margin)
        return u, v, ok


def quat_to_rot(q):
    """(N, 4) w, x, y, z -> (N, 3, 3) (normalised here)."""
    from .raster import quat_to_rotmat
    return quat_to_rotmat(q)


def covariance(rot, scales):
    """Σ = R diag(s²) Rᵀ: (N, 3, 3) rotations, (N, 3) linear scales."""
    M = rot * scales[:, None, :]
    return M @ M.transpose(1, 2)


# ====================================================================== the rasteriser (Vantage core/splat_raster.py)
def _tile_lists(u, v, rad, keep, W, H, rows):
    """(tile per pair, Gaussian per pair, tiles across): the (tile, Gaussian) pairs sorted by tile, then by the
    Gaussians' order (they arrive sorted by range)."""
    dev = u.device
    TW = (W + TILE - 1) // TILE
    ty_lo, ty_hi = rows
    tx0 = torch.clamp(torch.floor((u - rad) / TILE), 0, TW - 1).long()
    tx1 = torch.clamp(torch.floor((u + rad) / TILE), 0, TW - 1).long()
    ty0 = torch.clamp(torch.floor((v - rad) / TILE), ty_lo, ty_hi - 1).long()
    ty1 = torch.clamp(torch.floor((v + rad) / TILE), ty_lo, ty_hi - 1).long()
    inside = keep & (u + rad >= 0) & (u - rad < W) & (v + rad >= ty_lo * TILE) & (v - rad < min(ty_hi * TILE, H))
    nw = (tx1 - tx0 + 1)
    cnt = torch.where(inside, nw * (ty1 - ty0 + 1), torch.zeros_like(nw))
    total = int(cnt.sum())
    if total == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, TW
    gid = torch.repeat_interleave(torch.arange(len(u), device=dev), cnt)
    first = torch.cumsum(cnt, 0) - cnt
    k = torch.arange(total, device=dev) - torch.repeat_interleave(first, cnt)
    w = nw[gid]
    tile = (ty0[gid] + k // w) * TW + tx0[gid] + k % w
    order = torch.sort(tile, stable=True).indices
    return tile[order], gid[order], TW


def _depth_gradient(P, r, J, n):
    """(n, 2): the derivative, with respect to the pixel coordinates, of the range along a pixel's ray to the plane
    through P with normal n, at P's own pixel (Vantage): ∇r = −r (n·∂d/∂p) / (n·d), kept within ±r/4 per pixel."""
    d = P / r[:, None].clamp(min=1e-9)
    a = torch.where(d[:, 2:3].abs() < 0.9, torch.tensor([0.0, 0.0, 1.0], device=P.device, dtype=P.dtype),
                    torch.tensor([1.0, 0.0, 0.0], device=P.device, dtype=P.dtype))
    t1 = torch.linalg.cross(d, a)
    t1 = t1 / torch.linalg.norm(t1, dim=1, keepdim=True).clamp(min=1e-12)
    t2 = torch.linalg.cross(d, t1)
    T = torch.stack([t1, t2], 2)
    JT = J @ T
    det = JT[:, 0, 0] * JT[:, 1, 1] - JT[:, 0, 1] * JT[:, 1, 0]
    ok = det.abs() > 1e-12
    inv = torch.stack([torch.stack([JT[:, 1, 1], -JT[:, 0, 1]], 1), torch.stack([-JT[:, 1, 0], JT[:, 0, 0]], 1)], 1)
    inv = inv / torch.where(ok, det, torch.ones_like(det))[:, None, None]
    dd = (T @ inv) / r[:, None, None].clamp(min=1e-9)
    nd = (n * d).sum(1)
    nd = torch.where(nd.abs() < 0.02, torch.full_like(nd, 0.02) * torch.sign(nd + 1e-12), nd)
    g = -r[:, None] * torch.einsum('ni,nij->nj', n, dd) / nd[:, None]
    lim = (r / 4.0)[:, None]
    return torch.where(ok[:, None], torch.maximum(torch.minimum(g, lim), -lim), torch.zeros_like(g))


def _block(pxc, pyc, uu, vv, con, op, fv, g, valid, Tcur, keep_w=False, dg=None, depth_col=None):
    """One block of a chunk of tiles: the Gaussians g (tiles × k, front to back) composited over the tiles' pixels
    with the transmittance Tcur carried in."""
    dx = pxc[:, :, None] - uu[g][:, None, :]
    dy = pyc[:, :, None] - vv[g][:, None, :]
    cg = con[g]
    power = -0.5 * (cg[:, None, :, 0] * dx * dx + cg[:, None, :, 2] * dy * dy) - cg[:, None, :, 1] * dx * dy
    al = torch.clamp(op[g][:, None, :] * torch.exp(torch.clamp(power, max=0.0)), max=ALPHA_MAX)
    al = torch.where(valid[:, None, :] & (al >= ALPHA_MIN), al, torch.zeros_like(al))
    Tb = torch.cumprod(1.0 - al, -1)
    Tex = torch.cat([torch.ones_like(Tb[..., :1]), Tb[..., :-1]], -1) * Tcur[..., None]
    wgt = al * Tex
    add = torch.einsum('cpk,ckf->cpf', wgt, fv[g])
    lin = None
    if dg is not None:
        lin = dg[g][:, None, :, 0] * dx + dg[g][:, None, :, 1] * dy
        extra = (wgt * lin).sum(-1)
        add = torch.cat([add[..., :depth_col], (add[..., depth_col] + extra)[..., None], add[..., depth_col + 1:]], -1)
    if keep_w:
        med = has = None
        if depth_col is not None:
            cross = (Tex >= 0.5) & (Tex * (1.0 - al) < 0.5)
            has = cross.any(-1)
            k = cross.to(torch.int8).argmax(-1, keepdim=True)
            dep = fv[g][:, None, :, depth_col].expand_as(al)
            if lin is not None:
                dep = dep + lin
            med = torch.gather(dep, 2, k)[..., 0]
        return add, Tb[..., -1], wgt, med, has
    return add, Tb[..., -1]


def splat(Pc, cov_c, feats, opacity, proj, width: int, height: int, bg=None, rows=None, near: float = 0.05,
          block: int = 512, max_per_tile: int = 4096, chunk_elems: int | None = None, normals=None,
          depth_col: int | None = None, lowpass: float = LOWPASS) -> dict:
    """Render Gaussians already in the projector's frame (Vantage core/splat_raster.py's splat; module doc (b)).

    Pc (N, 3) centres, cov_c (N, 3, 3) covariances, feats (N, F), opacity (N,); proj a projector; bg (H·W, F) or None.
    Returns {'image' (H·W, F): Σ fᵢ αᵢ Tᵢ + T·bg, 'alpha' (H·W,), 'cut' (tile entries left out under gradients beyond
    max_per_tile), 'pairs', and without gradients with depth_col 'median' (H·W,): the depth feature of the Gaussian
    at which the transmittance falls through ½ (NaN where it never does)}. With normals (N, 3) and depth_col, that
    feature varies across the splat as the range to the plane through the centre with that normal (first order)."""
    from torch.utils.checkpoint import checkpoint
    dev = Pc.device
    W, H = int(width), int(height)
    F = feats.shape[1]
    TH = (H + TILE - 1) // TILE
    rows = (0, TH) if rows is None else (max(int(rows[0]), 0), min(int(rows[1]), TH))
    grad = torch.is_grad_enabled() and (Pc.requires_grad or cov_c.requires_grad or feats.requires_grad or
                                        opacity.requires_grad or (bg is not None and bg.requires_grad))
    if chunk_elems is None:
        cuda = dev.type == 'cuda'
        chunk_elems = (1 << 23 if cuda else 1 << 21) if grad else (1 << 24 if cuda else 1 << 22)
    bgf = bg if bg is not None else torch.zeros(H * W, F, device=dev, dtype=feats.dtype)
    out = {'image': bgf, 'alpha': torch.zeros(H * W, device=dev, dtype=feats.dtype), 'cut': 0, 'pairs': 0}
    if depth_col is not None and not grad:
        out['median'] = torch.full((H * W,), float('nan'), device=dev, dtype=feats.dtype)
    r0, r1 = rows[0] * TILE, min(rows[1] * TILE, H)
    if len(Pc) == 0 or r1 <= r0:
        return out
    rng = torch.linalg.norm(Pc, dim=-1)
    u, v, ok = proj(Pc)
    ok = ok & (rng > near) & (opacity > ALPHA_MIN)
    idx = torch.nonzero(ok).squeeze(1)
    if len(idx) == 0:
        return out
    P = Pc[idx]
    r = rng[idx]
    h = (1e-3 * r.detach()).clamp(min=1e-5)
    eye = torch.eye(3, device=dev, dtype=P.dtype)
    Q = torch.cat([P[None] + h[None, :, None] * eye[:, None, :], P[None] - h[None, :, None] * eye[:, None, :]])
    uq, vq, _ = proj(Q.reshape(-1, 3))
    uq, vq = uq.reshape(6, -1), vq.reshape(6, -1)
    J = torch.stack([torch.stack([(uq[k] - uq[k + 3]) / (2 * h) for k in range(3)], -1),
                     torch.stack([(vq[k] - vq[k + 3]) / (2 * h) for k in range(3)], -1)], 1)
    S2 = J @ cov_c[idx] @ J.transpose(1, 2)
    a = S2[:, 0, 0] + lowpass
    b = S2[:, 0, 1]
    c = S2[:, 1, 1] + lowpass
    det = torch.clamp(a * c - b * b, min=1e-12)
    con = torch.stack([c / det, -b / det, a / det], -1)
    uu, vv = u[idx], v[idx]
    op = opacity[idx]
    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
        m = torch.sqrt(torch.clamp(2.0 * torch.log(torch.clamp(op * 255.0, min=1.0 + 1e-6)), min=0.0)).clamp(max=3.0)
        rad = torch.ceil(m * torch.sqrt(lam))
        good = torch.isfinite(rad) & torch.isfinite(uu) & torch.isfinite(vv) & (rad < 4 * max(W, H))
        order = torch.argsort(r.detach(), stable=True)
        tile, gid_s, TW = _tile_lists(uu.detach()[order], vv.detach()[order], rad[order], good[order], W, H, rows)
    gid = order[gid_s]
    out['pairs'] = int(len(gid))
    if len(gid) == 0:
        return out
    with torch.no_grad():
        ntiles = TW * TH
        counts = torch.bincount(tile, minlength=ntiles)
        starts = torch.cumsum(counts, 0) - counts
        live = torch.nonzero(counts).squeeze(1)
        live = live[torch.argsort(counts[live], descending=True, stable=True)]
        cnt_live = counts[live].tolist()
    fv = feats[idx]
    dg = None
    if normals is not None and depth_col is not None:
        with torch.no_grad():
            dg = _depth_gradient(P.detach(), r.detach(), J.detach(), normals[idx].detach())
    lx = torch.arange(TILE, device=dev, dtype=P.dtype)
    gy, gx = torch.meshgrid(lx, lx, indexing='ij')
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    pieces_i, pieces_v, pieces_a, pieces_m = [], [], [], []
    i = 0
    nl = len(live)
    while i < nl:
        nmax = cnt_live[i]
        kmax = min(nmax, max_per_tile) if grad else nmax
        kb = min(block, kmax)
        n_c = max(1, min(nl - i, chunk_elems // (TILE * TILE * kb)))
        tiles = live[i:i + n_c]
        i += n_c
        tx, ty = tiles % TW, tiles // TW
        px = (tx[:, None] * TILE + gx[None]).to(P.dtype)
        py = (ty[:, None] * TILE + gy[None]).to(P.dtype)
        pix_ok = (px < W) & (py < H)
        pxc, pyc = px + 0.5, py + 0.5
        Tcur = torch.ones(len(tiles), TILE * TILE, device=dev, dtype=P.dtype)
        acc = torch.zeros(len(tiles), TILE * TILE, F, device=dev, dtype=feats.dtype)
        med = torch.full((len(tiles), TILE * TILE), float('nan'), device=dev, dtype=feats.dtype)
        tc = counts[tiles]
        if grad:
            out['cut'] += int(torch.clamp(tc - max_per_tile, min=0).sum())
            tc = torch.clamp(tc, max=max_per_tile)
        ts = starts[tiles]
        for k0 in range(0, kmax, kb):
            kk = min(kb, kmax - k0)
            ar = torch.arange(k0, k0 + kk, device=dev)
            valid = ar[None, :] < tc[:, None]
            pos = torch.clamp(ts[:, None] + ar[None, :], max=len(gid) - 1)
            g = gid[pos]
            if grad:
                add, Tb_last = checkpoint(_block, pxc, pyc, uu, vv, con, op, fv, g, valid, Tcur, False, dg, depth_col,
                                          use_reentrant=False)
            else:
                add, Tb_last, _wgt, md, mh = _block(pxc, pyc, uu, vv, con, op, fv, g, valid, Tcur, True, dg,
                                                    depth_col)
                if md is not None:
                    med = torch.where(mh & torch.isnan(med), md, med)
            acc = acc + add
            Tcur = Tcur * Tb_last
            if not grad and bool((Tcur < T_STOP).all()):
                break
        flat = (py * W + px).long()
        pieces_i.append(flat[pix_ok])
        pieces_v.append(acc[pix_ok])
        pieces_a.append(Tcur[pix_ok])
        pieces_m.append(med[pix_ok])
    pi = torch.cat(pieces_i)
    T = torch.cat(pieces_a)
    out['image'] = bgf.index_copy(0, pi, torch.cat(pieces_v) + T[:, None] * bgf[pi])
    out['alpha'] = out['alpha'].index_copy(0, pi, 1.0 - T)
    if depth_col is not None and not grad:
        out['median'] = out['median'].index_copy(0, pi, torch.cat(pieces_m))
    return out


# =========================================================================== cameras
def _to_frame(means, quats, viewmat):
    """Centres and rotations (N, 3, 3) in the camera's (or sensor's) frame."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    Pc = means @ R.T + t
    return Pc, R[None] @ quat_to_rot(quats)


def render_lens(means, scales, quats, opacities, features, viewmat, intr, width: int | None = None,
                height: int | None = None, background=None) -> dict:
    """(b), the lens fast path (module doc): {'image' (H, W, C), 'alpha' (H, W), 'range' (H, W): the expected range
    along each ray (Σ wᵢ rᵢ / α; 0 where α is 0)}. intr: a core.lens.Intrinsics; the image is width × height (default
    the lens's own size)."""
    w, h = int(width or intr.width), int(height or intr.height)
    proj = LensProjector.of(intr, w, h)
    Pc, Rc = _to_frame(means, quats, viewmat)
    cov = covariance(Rc, scales)
    rng = torch.linalg.norm(Pc, dim=1, keepdim=True)
    feats = torch.cat([features, rng.detach() if not rng.requires_grad else rng], 1)
    bg = None
    if background is not None:
        b = torch.as_tensor(background, dtype=features.dtype, device=features.device).reshape(1, -1)
        bg = torch.cat([b, torch.zeros_like(b[:, :1])], 1).expand(w * h, -1)
    out = splat(Pc, cov, feats, opacities, proj, w, h, bg=bg)
    field = torch.as_tensor(cube_plan(intr, w, h)['face'] >= 0, device=features.device)
    img = out['image'].reshape(h, w, -1) * field[..., None]
    al = out['alpha'].reshape(h, w) * field
    C = features.shape[1]
    return {'image': img[..., :C], 'alpha': al,
            'range': torch.where(al > 1e-6, img[..., C] / al.clamp(min=1e-6), torch.zeros_like(al)),
            'pairs': out['pairs'], 'field': field}


_PLANS: dict = {}


def cube_plan(intr, width: int | None = None, height: int | None = None, oversample: float = 1.0) -> dict:
    """The pinhole views a lens image is rendered from and the sampling grid per view (cached): {'views': [View],
    'face' (H, W) int64 (−1: outside the field), 'grid' (n views, H, W, 2) grid_sample coordinates}."""
    from ..lens import output_rays, plan_views
    w, h = int(width or intr.width), int(height or intr.height)
    key = (intr.width, intr.height, round(intr.f_px, 9), intr.model, tuple(intr.k), round(intr.cx, 9),
           round(intr.cy, 9), round(intr.max_theta, 9), round(intr.aspect, 9), w, h, float(oversample))
    plan = _PLANS.get(key)
    if plan is not None:
        return plan
    d, ok = output_rays(intr, w, h)
    f_out = intr.f_px * w / intr.width
    views = plan_views(d, ok, f_out, oversample)
    D = d.reshape(-1, 3)
    okf = ok.reshape(-1)
    if len(views) == 1:
        face = np.where(okf, 0, -1)
    else:
        face = np.where(okf, np.argmax(np.stack([D @ v.F[:, 2] for v in views], 1), 1), -1)
    grids = np.full((len(views), h * w, 2), -2.0, np.float32)
    for k, v in enumerate(views):
        sel = face == k
        dv = D[sel] @ v.F
        # OpenCV centres (View.project) -> grid_sample's normalised coordinates (align_corners=False)
        uu, vy = v.project(dv)
        grids[k, sel, 0] = 2.0 * (uu + 0.5) / v.width - 1.0
        grids[k, sel, 1] = 2.0 * (vy + 0.5) / v.height - 1.0
    plan = {'views': views, 'face': face.reshape(h, w), 'grid': grids.reshape(len(views), h, w, 2), 'size': (w, h)}
    if len(_PLANS) > 32:
        _PLANS.clear()
    _PLANS[key] = plan
    return plan


def render_cube(means, scales, quats, opacities, features, viewmat, intr, width: int | None = None,
                height: int | None = None, background=None, oversample: float = 1.0) -> dict:
    """(a), the reference (module doc): {'image' (H, W, C), 'alpha' (H, W), 'range' (H, W) expected range along each
    ray, 'views' (how many pinhole views were rendered)}."""
    from .raster import ray_norms, render
    plan = cube_plan(intr, width, height, oversample)
    w, h = plan['size']
    dev, dt = features.device, features.dtype
    C = features.shape[1]
    out_img = torch.zeros(h, w, C + 2, dtype=dt, device=dev)
    face = torch.as_tensor(plan['face'], device=dev)
    for k, v in enumerate(plan['views']):
        Fm = torch.as_tensor(v.F, dtype=viewmat.dtype, device=viewmat.device)
        vm = viewmat.clone()
        vm[:3, :] = Fm.T @ viewmat[:3, :]                       # world -> the view's optical frame
        K = (v.f, v.f, v.width / 2.0, v.height / 2.0)
        r = render(means, scales, quats, opacities, features, vm, K, v.width, v.height, background)
        rng = r['depth'] * ray_norms(K, v.width, v.height, dev, dt)
        stack = torch.cat([r['image'], r['alpha'][..., None], rng[..., None]], -1).permute(2, 0, 1)[None]
        grid = torch.as_tensor(plan['grid'][k], device=dev, dtype=dt)[None]
        smp = torch.nn.functional.grid_sample(stack, grid, mode='bilinear', padding_mode='border',
                                              align_corners=False)[0].permute(1, 2, 0)
        out_img = torch.where((face == k)[..., None], smp, out_img)
    al = out_img[..., C]
    return {'image': out_img[..., :C], 'alpha': al,
            'range': torch.where(al > 1e-6, out_img[..., C + 1] / al.clamp(min=1e-6), torch.zeros_like(al)),
            'views': len(plan['views']), 'field': torch.as_tensor(plan['face'] >= 0, device=dev)}


def render_fisheye(means, scales, quats, opacities, features, viewmat, intr, width=None, height=None,
                   background=None, method: str = 'cube') -> dict:
    """A fisheye (any core.lens model) by method 'cube' (the reference, (a)) or 'lens' (the fast path, (b))."""
    if method == 'cube':
        return render_cube(means, scales, quats, opacities, features, viewmat, intr, width, height, background)
    if method == 'lens':
        return render_lens(means, scales, quats, opacities, features, viewmat, intr, width, height, background)
    raise ValueError(f'unknown fisheye method {method!r} (cube or lens)')


def fisheye_intrinsics(width: int, height: int, fov: float = 190.0, model: str = 'equidistant', f_px=None, k=None):
    """A core.lens.Intrinsics of a fisheye sensor dict (Falcon's camera.fisheye)."""
    from ..lens import intrinsics
    return intrinsics({'type': 'camera.fisheye', 'width': int(width), 'height': int(height),
                       'lens': {'model': model, 'fov': float(fov), 'f_px': f_px, 'k': list(k or [0.0] * 4)}})
