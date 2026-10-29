"""Measuring a reconstruction (torch, numpy, OpenCV, scipy).

Three measures, each stated with what it can and cannot show:

1. **Held-out frames** (heldout()): every holdout-th frame of the source, never trained on, rendered at its own pose
   and compared with its image — PSNR and SSIM (core.recon.metrics) over the pixels its mask keeps, and, where the
   source has range labels (Falcon runs), the rendered range (the median depth's, core.recon.scene.render_views)
   against them — a pixel the reconstruction calls sky or empty where CARLA has a surface within 80 m counts as a
   100 % error. These poses lie *on* the recorded path, between training frames: an interpolation, the easy case.

2. **Another run's frames** (against_run()): the frames of another Falcon run of the same CARLA world (the town is
   checked; whether the weather is the same is reported) whose cameras lie beside the reconstruction's path. Each is
   a genuine CARLA render at a pose the reconstruction never saw; its offset from the nearest training camera is
   measured in that camera's axes (lateral: to the right; along: forward; vertical: up) and the results are grouped
   by lateral offset. The reference run has its own traffic: its moving actors are masked (its instance labels, or
   without them the classes of vehicles and people — parked map cars included); ghosts of the source's own actors
   remain wherever the source's masks missed them.

3. **Depth-warped CARLA references** (lateral_shift()): for a lateral shift without a CARLA render there (no server
   is started), the held-out frame's own image is forward-warped to the shifted camera through CARLA's own range
   labels — exact geometry and the true colour of each surface — and the reconstruction rendered there is compared
   with it over the pixels the warp defines. The warp splats the full-resolution source (1280 × 720 for 640 × 360:
   about four samples a target pixel), keeps the nearest surface per pixel (z-buffer) and averages the samples
   within 2 % (+ 5 cm) of it; it leaves out pixels no sample reaches (newly visible — disoccluded — areas and the
   strip outside the source's view), pixels that show a far surface through a gap of a nearer one (see_through():
   a surface over 1.1 × (+ 0.2 m) nearer within 2 px on both sides), and pixels from masked (moving) sources. The
   sky (range 0) moves with the camera's rotation only. What the warp cannot model — view-dependent shading, and
   everything in the left-out areas — makes it an approximation of CARLA's render, so its own error is measured
   (warp_check()): the same warp to a real CARLA frame at a known offset (a frame further along the path, or another
   run beside it) against that frame's image.

PSNR and SSIM of masked comparisons: left-out pixels take the render's value in the reference (they add no error) and
are not averaged. LPIPS is not computed (no pinned network).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from . import data as D
from .metrics import depth_errors, psnr, ssim

LATERAL_BINS = (0.0, 0.25, 0.75, 1.5, 2.5, 4.5, 8.0)


def _t(a, dev):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)


def _render(gs, viewmat, K, wh, bg=None) -> dict:
    """core.recon.scene.render_views: image, range (the median depth's; 0 for nothing or sky), alpha."""
    from .scene import render_views
    return render_views(gs, viewmat, K, wh[0], wh[1], bg)


def compare(render_f: torch.Tensor, ref_f: torch.Tensor, valid: torch.Tensor | None) -> dict:
    """PSNR / SSIM of a render against a reference (H, W, 3 in [0, 1]) over valid pixels (module doc)."""
    if valid is not None:
        ref_f = torch.where(valid[..., None], ref_f, render_f)
    return {'psnr': round(psnr(render_f, ref_f, valid), 4), 'ssim': round(ssim(render_f, ref_f, valid), 5),
            'valid_share': round(float(valid.float().mean()), 4) if valid is not None else 1.0}


def _save_strip(path, *imgs):
    import cv2
    row = np.concatenate([np.asarray(i) for i in imgs], 1)
    cv2.imwrite(path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _mean(rows: list, keys=('psnr', 'ssim')) -> dict:
    out = {'frames': len(rows)}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None and math.isfinite(r[k])]
        out[k] = round(float(np.mean(v)), 4) if v else None
    d = [r['depth'] for r in rows if r.get('depth') and r['depth'].get('pixels')]
    if d:
        out['depth_median_abs_rel'] = round(float(np.median([x['median_abs_rel'] for x in d])), 5)
        out['depth_within_10pc'] = round(float(np.mean([x['within_10pc'] for x in d])), 5)
    return out


# ================================================================================================ 1. held out
@torch.no_grad()
def heldout(gs, frames: list, wh, dev, bg=None, out_dir: str | None = None, renders: int = 4, masks=True) -> dict:
    """Held-out frames rendered at their own poses (module doc, 1)."""
    import cv2
    rows = []
    show = set(np.linspace(0, len(frames) - 1, min(renders, len(frames))).round().astype(int).tolist()) \
        if renders else set()
    if out_dir and show:
        os.makedirs(os.path.join(out_dir, 'renders'), exist_ok=True)
    for k, f in enumerate(frames):
        K = D.scaled_K(f.K, f.size, wh)
        out = _render(gs, _t(f.viewmat, dev), K, wh, bg)
        img = out['image'].clamp(0, 1)
        gt = _t(D.resize_rgb(D.load_rgb(f.image), wh), dev) / 255.0
        m = D.resize_mask(D.load_mask(f), wh) if masks else None
        mt = None if m is None else torch.as_tensor(m, device=dev)
        r = dict({'index': f.index, 'name': f.name}, **compare(img, gt, mt))
        ref = D.load_range(f)
        if ref is not None:
            ref = cv2.resize(ref, tuple(wh), interpolation=cv2.INTER_NEAREST)
            rt = _t(ref, dev)
            ok = (rt > 0) & (rt < 80)
            if mt is not None:
                ok &= mt
            r['depth'] = depth_errors(out['range'], rt, ok)
        rows.append(r)
        if k in show and out_dir:
            ri = (img.cpu().numpy() * 255 + 0.5).astype(np.uint8)
            gi = (gt.cpu().numpy() * 255 + 0.5).astype(np.uint8)
            _save_strip(os.path.join(out_dir, 'renders', f'heldout-{f.index:06d}.jpg'), gi, ri)
    return {'what': 'held-out frames of the source, rendered at their own poses (interpolation along the path)',
            'mean': _mean(rows), 'frames': rows}


# ================================================================================================ 2. another run
def offsets(rec, cams: np.ndarray, fwd: np.ndarray) -> list:
    """For camera centres (M, 3) and forward axes (M, 3): (nearest training frame, lateral, along, vertical, heading
    difference in degrees, inside: whether the nearest training camera is not an end of the path)."""
    tr = rec.split('train')
    V = np.array([np.asarray(f['viewmat']).reshape(4, 4) for f in tr])
    C = np.array([-v[:3, :3].T @ v[:3, 3] for v in V])
    out = []
    for c, fw in zip(cams, fwd):
        j = int(np.argmin(np.linalg.norm(C - c, axis=1)))
        R = V[j][:3, :3]
        d = c - C[j]
        right, down, ahead = R[0], R[1], R[2]
        h0 = math.degrees(math.atan2(ahead[1], ahead[0]))
        h1 = math.degrees(math.atan2(fw[1], fw[0]))
        out.append({'nearest': tr[j]['index'], 'lateral_m': round(float(d @ right), 4),
                    'along_m': round(float(d @ ahead), 4), 'vertical_m': round(float(-(d @ down)), 4),
                    'heading_deg': round(float((h1 - h0 + 180) % 360 - 180), 3), 'inside': 0 < j < len(tr) - 1})
    return out


def _bins(rows: list) -> list:
    out = []
    for lo, hi in zip(LATERAL_BINS[:-1], LATERAL_BINS[1:]):
        sel = [r for r in rows if lo <= abs(r['lateral_m']) < hi]
        if sel:
            out.append(dict(_mean(sel), lateral_m=[lo, hi],
                            mean_abs_lateral_m=round(float(np.mean([abs(r['lateral_m']) for r in sel])), 3)))
    return out


@torch.no_grad()
def against_run(rec, ref_source: str, limit: int = 40, max_heading_deg: float = 15.0, max_along_m: float = 1.0,
                out_dir: str | None = None, renders: int = 4, allow_other_world: bool = False) -> dict:
    """The reconstruction against another Falcon run's frames beside its path (module doc, 2)."""
    import cv2
    ref = D.open_capture(ref_source)
    if ref.kind != 'falcon' or rec.views['source']['kind'] != 'falcon':
        raise D.DataError('a reference run is a Falcon run, against a reconstruction of a Falcon run')
    town = rec.views.get('town') or ''
    if not allow_other_world and (ref.meta.get('town') or '') != town:
        raise D.DataError(f'the reference run is of {ref.meta.get("town")!r}, the reconstruction of {town!r}')
    src_w = rec.run.get('source', {}).get('meta', {}).get('weather')
    same_weather = src_w is None or src_w == ref.meta.get('weather')
    cams = np.array([f.centre for f in ref.frames])
    fwd = np.array([f.viewmat[2, :3] for f in ref.frames])
    offs = offsets(rec, cams, fwd)
    cand = [k for k, o in enumerate(offs) if o['inside'] and abs(o['heading_deg']) <= max_heading_deg and
            abs(o['along_m']) <= max_along_m]
    if not cand:
        return {'reference': ref.path, 'frames': [], 'note': 'no frame of the reference run lies beside the '
                                                             'reconstruction\'s path'}
    pick = [cand[i] for i in np.linspace(0, len(cand) - 1, min(limit, len(cand))).round().astype(int)]
    pick = sorted(set(pick))
    wh = rec.work_size
    rows = []
    show = set(pick[i] for i in np.linspace(0, len(pick) - 1, min(renders, len(pick))).round().astype(int))
    for k in pick:
        f = ref.frames[k]
        K = D.scaled_K(f.K, f.size, wh)
        out = _render(rec.gs, _t(f.viewmat, rec.device), K, wh)
        img = out['image'].clamp(0, 1)
        gt = _t(D.resize_rgb(D.load_rgb(f.image), wh), rec.device) / 255.0
        m = D.resize_mask(D.load_mask(f), wh)
        mt = None if m is None else torch.as_tensor(m, device=rec.device)
        r = dict({'index': f.index, 'name': f.name}, **offs[k], **compare(img, gt, mt))
        rg = D.load_range(f)
        if rg is not None:
            rt = _t(cv2.resize(rg, tuple(wh), interpolation=cv2.INTER_NEAREST), rec.device)
            ok = (rt > 0) & (rt < 80)
            if mt is not None:
                ok &= mt
            r['depth'] = depth_errors(out['range'], rt, ok)
        rows.append(r)
        if out_dir and k in show:
            os.makedirs(os.path.join(out_dir, 'renders'), exist_ok=True)
            _save_strip(os.path.join(out_dir, 'renders', f'reference-{os.path.basename(ref.path)}-{f.index:06d}.jpg'),
                        (gt.cpu().numpy() * 255 + 0.5).astype(np.uint8),
                        (img.cpu().numpy() * 255 + 0.5).astype(np.uint8))
    return {'what': 'another CARLA run\'s frames beside the path: genuine renders at unseen poses',
            'reference': ref.path, 'same_weather': same_weather, 'candidates': len(cand), 'mean': _mean(rows),
            'by_lateral': _bins(rows), 'frames': rows}


# ================================================================================================ 3. warping
def warp(src: D.Frame, cam: dict):
    """The source frame's image forward-warped through its range labels to a camera (module doc, 3): (reference
    (H, W, 3) float in [0, 1], valid (H, W) bool, reference range (H, W) m, 0 for the sky)."""
    img = D.load_rgb(src.image).astype(np.float64) / 255.0
    rg = D.load_range(src)
    if rg is None:
        raise D.DataError(f'{src.name}: no range labels to warp with')
    Hs, Ws = rg.shape
    if img.shape[:2] != rg.shape:
        img = D.resize_rgb((img * 255).astype(np.uint8), (Ws, Hs)).astype(np.float64) / 255.0
    msk = D.load_mask(src)
    msk = np.ones((Hs, Ws), bool) if msk is None else D.resize_mask(msk, (Ws, Hs))
    K = D.scaled_K(src.K, src.size, (Ws, Hs))
    vv, uu = np.mgrid[0:Hs, 0:Ws]
    d = np.stack([(uu + 0.5 - K[2]) / K[0], (vv + 0.5 - K[3]) / K[1], np.ones((Hs, Ws))], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    r = rg.reshape(-1)
    d = d.reshape(-1, 3)
    sky = r <= 0
    T = np.asarray(cam['viewmat'], float) @ D.inv(src.viewmat)        # source optical -> target optical
    p = np.where(sky[:, None], 0.0, d * r[:, None]) @ T[:3, :3].T + np.where(sky[:, None], 0.0, T[:3, 3])
    dir_t = d @ T[:3, :3].T
    p = np.where(sky[:, None], dir_t, p)
    z = p[:, 2]
    W, H = cam['size']
    Kt = cam['K']
    ok = z > 0.05
    ok &= np.isfinite(z)
    u = np.where(ok, Kt[0] * p[:, 0] / np.where(ok, z, 1.0) + Kt[2], -1)
    v = np.where(ok, Kt[1] * p[:, 1] / np.where(ok, z, 1.0) + Kt[3], -1)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi = u[ok].astype(np.int64), v[ok].astype(np.int64)
    zt = np.where(sky[ok], np.inf, z[ok])
    pix = vi * W + ui
    zbuf = np.full(H * W, np.inf)
    np.minimum.at(zbuf, pix, zt)
    hit = np.zeros(H * W, bool)
    hit[pix] = True
    near = zt <= zbuf[pix] * 1.02 + 0.05
    near |= np.isinf(zt) & np.isinf(zbuf[pix])
    col = img.reshape(-1, 3)[ok]
    good_src = msk.reshape(-1)[ok]
    acc = np.zeros((H * W, 3))
    cnt = np.zeros(H * W)
    bad = np.zeros(H * W)
    np.add.at(acc, pix[near], col[near])
    np.add.at(cnt, pix[near], 1.0)
    np.add.at(bad, pix[near & ~good_src], 1.0)
    zb = zbuf.reshape(H, W)
    zfin = np.where(np.isinf(zb), 1e9, zb)
    valid = hit.reshape(H, W) & (cnt.reshape(H, W) > 0) & (bad.reshape(H, W) == 0) & ~see_through(zfin)
    ref = acc.reshape(H, W, 3) / np.maximum(cnt.reshape(H, W, 1), 1)
    # the reference's range along the target's rays (0: sky)
    xs = (np.arange(W) + 0.5 - Kt[2]) / Kt[0]
    ys = (np.arange(H) + 0.5 - Kt[3]) / Kt[1]
    norms = np.sqrt(1 + xs[None, :] ** 2 + ys[:, None] ** 2)
    rref = np.where(np.isinf(zb), 0.0, zb * norms)
    return ref.astype(np.float32), valid, rref.astype(np.float32)


def see_through(z: np.ndarray, ratio: float = 1.1, slack_m: float = 0.2, reach: int = 2) -> np.ndarray:
    """Pixels of a forward-warped depth map that show a far surface through a gap of a nearer one: a nearer surface
    (z < (depth − slack_m) / ratio) within `reach` pixels on both sides of it, left and right or above and below. A
    depth edge has the nearer surface on one side only, and a slanted surface (the road ahead, a wall seen at a
    grazing angle) is nearer on one side only too, so neither is flagged."""
    big = 1e12
    pad = np.pad(z, reach, constant_values=big)
    H, W = z.shape
    lim = (z - slack_m) / ratio

    def side(dy0, dy1, dx0, dx1):
        m = np.full(z.shape, big)
        for dy in range(dy0, dy1 + 1):
            for dx in range(dx0, dx1 + 1):
                m = np.minimum(m, pad[reach + dy:reach + dy + H, reach + dx:reach + dx + W])
        return m < lim
    left, right = side(-1, 1, -reach, -1), side(-1, 1, 1, reach)
    up, down = side(-reach, -1, -1, 1), side(1, reach, -1, 1)
    return (left & right) | (up & down)


def _pairs_by_distance(frames: list, dist_m: float, tol: float = 0.15) -> list:
    """(k, j) with frame j about dist_m ahead of frame k along the path (within tol × dist_m)."""
    c = np.array([f.centre for f in frames])
    out = []
    for k in range(len(frames)):
        d = np.linalg.norm(c[k + 1:] - c[k], axis=1)
        j = np.nonzero(np.abs(d - dist_m) <= tol * dist_m)[0]
        if len(j):
            out.append((k, k + 1 + int(j[0])))
    return out


@torch.no_grad()
def warp_check(cap: D.Capture, wh, distances=(0.5, 1.0), limit: int = 20, dev=None,
               targets: D.Capture | None = None, max_along_m: float = 1.0) -> list:
    """The warp's own error (module doc, 3): each frame warped to a real CARLA frame at a known offset — a frame
    distances metres further along the same path, or (targets) the frames of another run beside it (within
    max_along_m along the path of a source frame that is not an end of it) — against that frame's image."""
    dev = dev or torch.device('cpu')
    out = []
    if targets is None:
        for dist in distances:
            pairs = _pairs_by_distance(cap.frames, dist)
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, min(limit, len(pairs))).round().astype(int)] \
                if pairs else []
            rows = [_warp_row(cap.frames[k], cap.frames[j], wh, dev) for k, j in pairs]
            out.append(dict(_mean(rows), what=f'warped {dist} m forward along the path to a real CARLA frame',
                            distance_m=dist, coverage=_cov(rows), frames=rows))
        return out
    tc = np.array([f.centre for f in targets.frames])
    sc = np.array([f.centre for f in cap.frames])
    near = []
    for j in range(len(targets.frames)):
        k = int(np.argmin(np.linalg.norm(sc - tc[j], axis=1)))
        R = cap.frames[k].viewmat[:3, :3]
        dd = tc[j] - sc[k]
        if 0 < k < len(cap.frames) - 1 and abs(float(dd @ R[2])) <= max_along_m:
            near.append((j, k, round(float(dd @ R[0]), 3), round(float(dd @ R[2]), 3)))
    rows = []
    for i in sorted(set(np.linspace(0, len(near) - 1, min(limit, len(near))).round().astype(int).tolist())) \
            if near else []:
        j, k, lat, along = near[i]
        row = _warp_row(cap.frames[k], targets.frames[j], wh, dev)
        row.update(lateral_m=lat, along_m=along)
        rows.append(row)
    out.append(dict(_mean(rows), what='warped beside the path to another run\'s real CARLA frames',
                    target=targets.path, coverage=_cov(rows), frames=rows))
    return out


def _cov(rows):
    v = [r['valid_share'] for r in rows]
    return round(float(np.mean(v)), 4) if v else None


def _warp_row(src: D.Frame, dst: D.Frame, wh, dev) -> dict:
    cam = {'viewmat': dst.viewmat, 'K': D.scaled_K(dst.K, dst.size, wh), 'size': tuple(wh)}
    ref, valid, _ = warp(src, cam)
    gt = D.resize_rgb(D.load_rgb(dst.image), wh).astype(np.float32) / 255.0
    m = D.resize_mask(D.load_mask(dst), wh)
    if m is not None:
        valid = valid & m
    vt = torch.as_tensor(valid, device=dev)
    return dict({'source': src.index, 'target': dst.index}, **compare(_t(ref, dev), _t(gt, dev), vt))


@torch.no_grad()
def lateral_shift(rec, shifts=(0.5, 1.0), limit: int = 20, out_dir: str | None = None, renders: int = 2) -> dict:
    """The reconstruction at laterally shifted poses of its held-out frames against depth-warped CARLA references
    (module doc, 3). Each shift is taken to the right and to the left; shift 0 is the control (the reference is the
    frame resampled through the same warp)."""
    cap = rec.capture()
    test = rec.split('test') or rec.split('train')
    test = [test[i] for i in sorted(set(np.linspace(0, len(test) - 1, min(limit, len(test))).round().astype(int)))]
    wh = rec.work_size
    res = []
    for s in (0.0,) + tuple(x for v in shifts for x in (float(v), -float(v))):
        rows = []
        for n, f in enumerate(test):
            src = rec.source_frame(f)
            if D.load_range(src) is None:
                continue
            cam = rec.camera(f, lateral_m=s)
            ref, valid, rref = warp(src, cam)
            out = _render(rec.gs, _t(cam['viewmat'], rec.device), cam['K'], wh)
            img = out['image'].clamp(0, 1)
            vt = torch.as_tensor(valid, device=rec.device)
            r = dict({'index': f['index'], 'lateral_m': s}, **compare(img, _t(ref, rec.device), vt))
            rt = _t(rref, rec.device)
            ok = vt & (rt > 0) & (rt < 80)
            r['depth'] = depth_errors(out['range'], rt, ok)
            rows.append(r)
            if out_dir and n < renders and s >= 0:
                os.makedirs(os.path.join(out_dir, 'renders'), exist_ok=True)
                refv = np.where(valid[..., None], ref, 0.0)
                _save_strip(os.path.join(out_dir, 'renders', f'lateral-{s:+.2f}-{f["index"]:06d}.jpg'),
                            (refv * 255 + 0.5).astype(np.uint8), (img.cpu().numpy() * 255 + 0.5).astype(np.uint8))
        if rows:
            res.append(dict(_mean(rows), lateral_m=s, coverage=_cov(rows), frames=rows))
    return {'what': 'held-out frames shifted sideways, against the frame depth-warped there through CARLA\'s range '
                    'labels (an approximation of CARLA\'s render: see warp_check)', 'shifts': res}
