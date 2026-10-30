"""Hybrid worlds: CARLA's actors composited over a 3D Gaussian Splatting reconstruction of the static scene (torch,
numpy, OpenCV). docs/reconstruction.md, "Hybrid worlds", is the manual.

    hybrid_run('…/recon/drive', '…/runs/redrive', '…/runs/redrive-hybrid')     # a recorded CARLA drive, composited
    cfg = hybrid_drive_config(project, rec, '…/carla', port, tm_port, seed=7, lateral_m=1.0)   # the drive to record

The split of the world. A reconstruction of a Falcon run holds the static scene only: every CARLA actor — vehicles and
walkers, moving or stopped, and the traffic lights, whose state changes — and the ego's body were masked out of its
training (core.recon.data.actor_object_ids); the map's own meshes, parked cars included, are in it. A hybrid world
draws the same line: CARLA simulates and renders the actors (traffic, scenario actors, the ego's motion, the lights'
states), the reconstruction gives everything else.

Per camera and tick (Compositor.frame): CARLA's camera gives the actors' colours, its instance label (objects with an
actor id: actor_object_ids, and the ego class) the actor mask A, its range label the actors' range r_a; the
reconstruction rendered at the camera's pose (scene.render_views) gives the background's colour and range r_s (the
median depth's; 0: nothing or sky) — rendered at the reconstruction's working width and upsampled for a wider camera
(render_width; a reconstruction fitted at 640 px and rendered at 1280 × 720 shows gaps between its Gaussians: 27.3
against 30.0 dB on the Town 10 reconstruction's held-out frames). Methods:

* 'mask' — a pixel is CARLA's where A. CARLA's own render has already decided, exactly, which surface is nearest at
  every pixel against CARLA's static geometry — and that geometry is the reconstruction's scene when it is a
  reconstruction of a CARLA run of the same world. The hybrid then differs from CARLA's frame on background pixels
  only (actor pixels are CARLA's, identical by construction).
* 'depth' — a pixel is CARLA's where A and (r_a ≤ r_s·(1 + tolerance) or r_s = 0): the per-pixel depth test, which a
  reconstruction whose geometry is not CARLA's needs (its static surfaces may hide an actor CARLA's do not). It
  cannot show an actor CARLA's geometry hides. Its errors are the reconstruction's depth errors: a floater or a
  blurred edge nearer than an actor hides it. An actor pixel the test gives to the background is relabelled with
  the reconstruction's estimate (semantic: the voted class, or 0 without votes; instance: 0; range: the
  reconstruction's) and counted (hybrid.jsonl 'hidden_px'); objects/ keeps CARLA's boxes. tolerance (0.05) is a
  heuristic: a tyre meets the road at the road's own range, which the reconstruction gives to a few per cent.
* 'auto' (the default) — 'mask' for a reconstruction of a Falcon run used without an alignment (the same CARLA world),
  else 'depth'.

The run (hybrid_run): a falcon-raw run made from the CARLA drive as core.translate makes a translated run — every
per-frame file hard-linked (copied across file systems), the run's own records copied — with each composited
camera's rgb written new (JPEG at the drive worker's quality, 92). **Every label is CARLA's** (semantic, instance,
range, objects, flow, lidar, …: exact for the world at that tick; the background pixels' colours are the
reconstruction's estimate of those same surfaces), except the pixels the 'depth' test relabels. The reconstruction's
voted classes are measured against CARLA's labels instead of written (label agreement below). run.json keeps the
drive's (its clock, rig and settings) with worker 'recon.hybrid' and a 'hybrid' block: the source drive, the
reconstruction and its point_cloud.ply's SHA-256, the method, the alignment, what was measured; hybrid.jsonl has a
row per composited frame; hybrid-previews/ a few strips (CARLA | hybrid | the reconstruction alone | the actor mask).

Measured (when the reconstruction is of a CARLA run of the same world: CARLA's own frame at that tick and pose is
then the ground truth of the hybrid; JPEG-decoded, as recorded): PSNR and SSIM (core.recon.metrics, masked as
core.recon.evaluate.compare does) of the composite before JPEG encoding over the whole frame, the background pixels
(not A), the actor pixels (A; exact by construction under 'mask', so measured for the 'depth' alternative), and the
background within near_px of an actor (where missing shadows and the actor's edge show) — and, for a camera wider
than the reconstruction's working width, the whole frame and the background again at that width (both images
area-resampled, as its training images were); the voted classes against
CARLA's semantic labels on background pixels (accuracy, and the IoU of each class over 0.1 % of them, summed over the
frames); the reconstruction's range against CARLA's on background pixels within 80 m (median absolute relative error)
and the residual image shift between the reconstruction and CARLA's frame (phase correlation of the background, actor
pixels filled with the mean) — the alignment's check; each frame's offset from the nearest training camera
(evaluate.offsets); and the cost: milliseconds a frame to render the reconstruction and to composite.

Not modelled: shadows and lighting between the two worlds. CARLA's actors cast no shadow on the reconstruction's
background (and shadows the source drive's actors cast, which no label covers, may be faintly in it); the actors are
lit by CARLA's sun, the background by whatever lit the source. Darkening under actors from CARLA's own render would
need that render without the actors at the same tick (a second render with every actor hidden); the ratio of CARLA's
frame to the reconstruction would put the ground truth into the measured background and does not carry over to a
reconstruction whose appearance is not CARLA's — so none is applied.

Online use: Compositor.frame() is one camera and tick; Falcon's own drivers (the traffic manager, the replay's log and
follower drivers) never read the camera's pixels, so compositing the recorded drive afterwards gives exactly the
images compositing at each tick would. A policy that drives on the hybrid image would call it in its loop (not wired
into the environment worker).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import numpy as np
import torch

from . import data as D

METHODS = ('auto', 'mask', 'depth')
DEFAULTS = {'method': 'auto', 'tolerance': 0.05, 'near_px': 12, 'min_actor_px': 64, 'previews': 3, 'jpeg_quality': 92,
            'depth_max_m': 80.0}
SCHEMA = 'falcon.recon_hybrid/1'


class HybridError(ValueError):
    pass


# ================================================================================================ the compositor
def check_align(align) -> np.ndarray:
    """reconstruction world ← CARLA's world mirrored (right-handed: core.exporters.rh), a rigid 4 × 4."""
    A = np.eye(4) if align is None else np.asarray(align, float).reshape(4, 4)
    R = A[:3, :3]
    if not (np.allclose(R.T @ R, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6 and
            np.allclose(A[3], [0, 0, 0, 1])):
        raise HybridError('the alignment must be rigid (a rotation and a translation, metres): ranges are compared '
                          'between the two worlds')
    return A


class Compositor:
    """One camera and tick at a time (module doc). rec: a scene.Recon; align: reconstruction world ← CARLA's world
    (right-handed), None for a reconstruction of a CARLA run of the same world."""

    def __init__(self, rec, align=None, method: str = 'auto', tolerance: float = DEFAULTS['tolerance'],
                 render_width: int | None = 0):
        if method not in METHODS:
            raise HybridError(f'method must be one of {", ".join(METHODS)}')
        self.rec, self.dev = rec, rec.device
        self.A = check_align(align)
        self.Ainv = D.inv(self.A)
        self.exact = align is None and rec.views['source']['kind'] == 'falcon' and bool(rec.views.get('carla_exact'))
        if align is None and rec.views['source']['kind'] != 'falcon':
            raise HybridError('this reconstruction is not of a CARLA run: give the alignment of its world to the CARLA '
                              'world (reconstruction ← CARLA, right-handed, a rigid 4 × 4)')
        self.method = ('mask' if self.exact else 'depth') if method == 'auto' else method
        self.tolerance = float(tolerance)
        # 0: the reconstruction's working width (a wider camera's background is rendered there and upsampled);
        # None: the camera's own size
        self.render_width = int(rec.work_size[0]) if render_width == 0 else render_width
        self.votes = rec.gs.votes is not None
        self._feats = None

    def view(self, viewmat_carla) -> np.ndarray:
        """optical ← reconstruction world, for a camera's optical ← CARLA world (right-handed)."""
        return np.asarray(viewmat_carla, float) @ self.Ainv

    @torch.no_grad()
    def background(self, viewmat_carla, K, size) -> dict:
        """The reconstruction at the camera: image (H, W, 3) in [0, 1], range (m along each ray, 0: nothing or
        sky), alpha and, with votes, semantic (the voted class; 0 where alpha < ½) — torch tensors on the device. A
        camera wider than render_width is rendered at that width and upsampled to its size (image and alpha
        bicubic / bilinear, range and classes nearest): a 3D Gaussian reconstruction rendered far above the resolution
        it was fitted at shows gaps between its Gaussians (their 0.3 px dilation is in pixels) — module doc."""
        w, h = int(size[0]), int(size[1])
        ws = D.working_size((w, h), self.render_width) if self.render_width else (w, h)
        if ws == (w, h):
            return self._render(viewmat_carla, K, (w, h))
        import torch.nn.functional as F
        r = self._render(viewmat_carla, D.scaled_K(K, (w, h), ws), ws)

        def up(x, mode):
            x = x.permute(2, 0, 1)[None] if x.dim() == 3 else x[None, None]
            kw = {} if mode == 'nearest' else {'align_corners': False}
            y = F.interpolate(x.float(), size=(h, w), mode=mode, **kw)[0]
            return y.permute(1, 2, 0) if y.shape[0] > 1 else y[0]
        out = {'image': up(r['image'], 'bicubic').clamp(0, 1), 'alpha': up(r['alpha'], 'bilinear'),
               'range': up(r['range'], 'nearest')}
        if 'semantic' in r:
            out['semantic'] = up(r['semantic'], 'nearest').long()
        return out

    def _render(self, viewmat_carla, K, size) -> dict:
        from .scene import render_views
        gs = self.rec.gs
        if self.votes and self._feats is None:
            v = gs.votes.to(self.dev)
            self._feats = v / v.sum(-1, keepdim=True).clamp_min(1e-12)
        vm = torch.as_tensor(self.view(viewmat_carla), dtype=torch.float32, device=self.dev)
        out = render_views(gs, vm, tuple(float(k) for k in K), int(size[0]), int(size[1]), extra=self._feats)
        res = {'image': out['image'].clamp(0, 1), 'range': out['range'], 'alpha': out['alpha']}
        if self.votes:
            cls = out['extra'].argmax(-1)
            res['semantic'] = torch.where(out['alpha'] >= 0.5, cls, torch.zeros_like(cls))
        return res

    def win(self, actor: torch.Tensor, actor_range: torch.Tensor | None, bg_range: torch.Tensor,
            method: str | None = None) -> torch.Tensor:
        """The pixels that show CARLA's render."""
        method = method or self.method
        if method == 'mask':
            return actor
        if actor_range is None:
            raise HybridError('the depth test needs the drive\'s range labels (a camera with the depth label)')
        return actor & ((actor_range <= bg_range * (1.0 + self.tolerance)) | (bg_range <= 0) | (actor_range <= 0))

    @torch.no_grad()
    def frame(self, rgb, actor, actor_range, viewmat_carla, K) -> dict:
        """rgb (H, W, 3) uint8 — CARLA's camera; actor (H, W) bool; actor_range (H, W) m or None; the camera's
        optical ← CARLA world (right-handed) and K (fx, fy, cx, cy, pixel-edge). Returns image (H, W, 3) uint8 and,
        as tensors, image_f, win, actor, the background (background()) and the times (ms)."""
        h, w = rgb.shape[:2]
        sync = torch.cuda.synchronize if self.dev.type == 'cuda' else (lambda: None)
        t0 = time.perf_counter()
        bg = self.background(viewmat_carla, K, (w, h))
        sync()
        t1 = time.perf_counter()
        car = torch.as_tensor(rgb, device=self.dev).float() / 255.0
        a = torch.as_tensor(np.asarray(actor, bool), device=self.dev)
        ra = None if actor_range is None else torch.as_tensor(np.asarray(actor_range, np.float32), device=self.dev)
        win = self.win(a, ra, bg['range'])
        img = torch.where(win[..., None], car, bg['image'])
        out = (img * 255 + 0.5).to(torch.uint8).cpu().numpy()
        t2 = time.perf_counter()
        return {'image': out, 'image_f': img, 'carla_f': car, 'win': win, 'actor': a, 'actor_range': ra,
                'background': bg, 'render_ms': (t1 - t0) * 1e3, 'composite_ms': (t2 - t1) * 1e3}


# ================================================================================================ measures
def _region(img, ref, valid, min_px=1) -> dict | None:
    from .evaluate import compare
    if valid is not None and int(valid.sum()) < min_px:
        return None
    r = compare(img, ref, valid)
    if not math.isfinite(r['psnr']):
        r['psnr'], r['identical'] = None, True
    return r


def _shift_px(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> float | None:
    """The residual translation (px) between two RGB images over the kept pixels (phase correlation; the rest filled
    with the kept pixels' mean)."""
    import cv2
    if keep.mean() < 0.2:
        return None
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32) for x in (a, b))
    for g in (ga, gb):
        g[~keep] = g[keep].mean()
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, win)
    return round(math.hypot(dx, dy), 3)


class _Conf:
    """Voted class against CARLA's label, summed over frames (256 × 256)."""

    def __init__(self):
        self.m = np.zeros((256, 256), np.int64)

    def add(self, label: np.ndarray, pred: np.ndarray):
        self.m += np.bincount(label.astype(np.int64) * 256 + pred.astype(np.int64), minlength=65536).reshape(256, 256)

    def summary(self, names: dict | None = None, min_share: float = 0.001) -> dict | None:
        n = int(self.m.sum())
        if not n:
            return None
        tp = np.diag(self.m)
        gt, pr = self.m.sum(1), self.m.sum(0)
        keep = np.flatnonzero(gt >= min_share * n)
        iou = {int(c): round(float(tp[c] / max(gt[c] + pr[c] - tp[c], 1)), 4) for c in keep}
        return {'pixels': n, 'accuracy': round(float(tp.sum() / n), 5),
                'mIoU': round(float(np.mean(list(iou.values()))), 4) if iou else None, 'classes': len(iou),
                'iou': {(names or {}).get(c, str(c)): v for c, v in sorted(iou.items())}}


def _get(r, path):
    for k in path:
        r = r.get(k) if isinstance(r, dict) else None
    return r


def _summ(rows: list, conf: _Conf, names: dict) -> dict:
    def mean(path):
        v = [float(x) for x in (_get(r, path) for r in rows) if isinstance(x, (int, float)) and math.isfinite(x)]
        return round(float(np.mean(v)), 4) if v else None

    def median(key):
        v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(float(np.median(v)), 4) if v else None
    def region(*path):
        got = [x for x in (_get(r, path) for r in rows) if x]
        d = {'psnr': mean(path + ('psnr',)), 'ssim': mean(path + ('ssim',)), 'frames': len(got)}
        if got and all(x.get('identical') for x in got):
            d['identical'] = 'CARLA\'s own pixels (exact by construction)'
        return d
    out = {'frames': len(rows)}
    for reg in ('whole', 'background', 'actors', 'near_actors'):
        out[reg] = region('metrics', reg)
    if any((r.get('metrics') or {}).get('work') for r in rows):
        out['work'] = {'size': next(r['metrics']['work']['size'] for r in rows if (r.get('metrics') or {}).get('work')),
                       'whole': region('metrics', 'work', 'whole'),
                       'background': region('metrics', 'work', 'background')}
    alts = sorted({k for r in rows for k in (r.get('alternative') or {})})
    out['alternative'] = {m: {'whole': region('alternative', m, 'whole'), 'actors': region('alternative', m, 'actors'),
                              'hidden_share': mean(('alternative', m, 'hidden_share'))} for m in alts}
    out['hidden_share'] = mean(('hidden_share',))
    out['actor_share'] = mean(('actor_share',))
    out['labels'] = conf.summary(names)
    out['depth_background_median_abs_rel'] = median('depth_median_abs_rel')
    out['shift_px_median'] = median('shift_px')
    out['lateral_m'] = mean(('offset', 'lateral_m'))
    out['abs_lateral_m'] = round(float(np.mean([abs(r['offset']['lateral_m']) for r in rows if r.get('offset')])),
                                 4) if any(r.get('offset') for r in rows) else None
    for k in ('render_ms', 'composite_ms'):
        out[k] = {'mean': mean((k,)), 'median': median(k)}
    return out


# ================================================================================================ the run
def _link(src, dst, mode):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == 'hard':
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _write(path: str, arr: np.ndarray, params=()):
    """A new file in place of whatever (a hard link) is there: written beside it, then renamed over it."""
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part' + os.path.splitext(path)[1]
    if not cv2.imwrite(tmp, arr, list(params)):
        raise HybridError(f'cannot write {path}')
    os.replace(tmp, path)


def _class_names(run) -> dict:
    sem = (run.ds.get('labels') or {}).get('semantic') or {}
    out = {}
    for c in sem.get('classes') or []:
        if isinstance(c, dict) and 'id' in c:
            out[int(c['id'])] = c.get('name') or str(c['id'])
    return out


def hybrid_run(recon, drive_dir: str, out_dir: str, cameras=None, method: str = 'auto',
               tolerance: float = DEFAULTS['tolerance'], align=None, measure: bool = True, link: str = 'hard',
               settings: dict | None = None, emitter=None, cancel=None, argv=None, device: str = 'auto',
               render_width: int | None = 0) -> dict:
    """Composite a recorded CARLA drive over a reconstruction into a new falcon-raw run (module doc); returns its
    summary. render_width: 0 renders a wider camera's background at the reconstruction's working width and
    upsamples it (Compositor.background), None at the camera's own size. recon: a scene.Recon or its folder. Raises HybridError, or KeyboardInterrupt on cancel (the partial
    folder is removed)."""
    import cv2
    from .. import provenance as PV
    from ..exporters import ExportError, Run, is_pinhole
    from .scene import Recon
    s = dict(DEFAULTS, **(settings or {}))
    em = emitter or _Quiet()
    rec = recon if isinstance(recon, Recon) else Recon.load(recon, device)
    try:
        run = Run(drive_dir)
    except ExportError as e:
        raise HybridError(str(e)) from None
    if os.path.abspath(out_dir) == run.dir:
        raise HybridError('the hybrid run needs a folder of its own')
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise HybridError(f'{out_dir} exists and is not empty')
    comp = Compositor(rec, align, method, tolerance, render_width)
    town = rec.views.get('town') or ''
    if comp.exact and (run.cfg.get('town') or run.cfg.get('xodr') or '') != town:
        raise HybridError(f'the drive is of {run.cfg.get("town") or run.cfg.get("xodr")!r}, the reconstruction of '
                          f'{town!r}: a reconstruction of a CARLA run composites drives of the same world')
    cams = list(cameras) if cameras else [c for c in run.cams if is_pinhole(run, c)]
    if not cams:
        raise HybridError(f'{run.dir}: no pinhole camera (the reconstruction renders pinhole cameras only)')
    caps = {}
    for c in cams:
        if c not in run.cams or not is_pinhole(run, c):
            raise HybridError(f'{run.dir}: no pinhole camera {c!r}')
        steps = run.camera_steps(c)
        if not any(run.own(r, c, 'instance') or run.own(r, c, 'semantic') for r in steps):
            raise HybridError(f'camera {c} has no instance or semantic label: the actors\' pixels are not known')
        if comp.method == 'depth' and not any(run.own(r, c, 'range') for r in steps):
            raise HybridError(f'camera {c} has no range label, which the depth test needs')
        caps[c] = D.falcon_capture(run.dir, c)
    exact = comp.exact and measure
    src_dir = rec.views['source'].get('path') if rec.views['source']['kind'] == 'falcon' else None
    own_source = bool(src_dir) and os.path.abspath(src_dir) == run.dir
    views = _views(rec)
    names = _class_names(run)
    by_step = {r['step']: r for r in run.frames}
    alt = [m for m in ('mask', 'depth') if m != comp.method]
    settings_doc = {'method': comp.method, 'asked': method, 'tolerance': comp.tolerance, 'cameras': cams,
                    'render_width': comp.render_width,
                    'align': None if align is None else comp.A.round(9).tolist(), 'measure': exact, 'link': link,
                    **{k: s[k] for k in ('near_px', 'min_actor_px', 'jpeg_quality', 'depth_max_m')}}
    os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(out_dir, '.hybrid-partial')
    open(partial, 'w').close()
    t_start = time.perf_counter()
    try:
        todo = {}
        for c, cap in caps.items():
            for f in cap.frames:
                rel = run.own(by_step[f.index], c, 'rgb')
                if rel:
                    todo[(c, f.index)] = (rel, f)
        skip = {rel for rel, _ in todo.values()} | {'run.json', 'dataset.json', 'hybrid.jsonl', '.hybrid-partial'}
        for d, dirs, files in os.walk(run.dir):
            dirs.sort()
            for fn in sorted(files):
                p = os.path.join(d, fn)
                rel = os.path.relpath(p, run.dir)
                if rel not in skip:
                    _link(p, os.path.join(out_dir, rel), link if d != run.dir else 'copy')
        rows, confs = [], {}
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s['near_px'] + 1,) * 2)
        n_prev = max(0, int(s['previews']))
        order = sorted(todo)
        show = set(order[i] for i in np.linspace(0, len(order) - 1, min(n_prev, len(order))).round().astype(int)) \
            if n_prev and order else set()
        for k, key in enumerate(order):
            if cancel and cancel():
                raise KeyboardInterrupt
            c, step = key
            rel, f = todo[key]
            t_io = time.perf_counter()
            rgb = D.load_rgb(f.image)
            keep = D.load_mask(f)
            if keep is None:
                raise HybridError(f'{rel}: no instance or semantic label for this frame')
            actor = ~keep
            rng = D.load_range(f)
            io_ms = (time.perf_counter() - t_io) * 1e3
            out = comp.frame(rgb, actor, rng, f.viewmat, f.K)
            row = {'camera': c, 'step': int(step), 'file': rel, 'method': comp.method,
                   'actor_share': round(float(actor.mean()), 5), 'render_ms': round(out['render_ms'], 2),
                   'composite_ms': round(out['composite_ms'], 2)}
            vm = comp.view(f.viewmat)
            row['offset'] = _offset(rec, vm)
            row['split'], row['view_m'] = _nearest_view(views, vm, step)
            hidden = out['actor'] & ~out['win']
            row['hidden_px'] = int(hidden.sum())
            row['hidden_share'] = round(row['hidden_px'] / max(int(actor.sum()), 1), 5) if actor.any() else None
            t_w = time.perf_counter()
            ext = os.path.splitext(rel)[1].lower()
            params = [cv2.IMWRITE_JPEG_QUALITY, int(s['jpeg_quality'])] if ext in ('.jpg', '.jpeg') else []
            _write(os.path.join(out_dir, rel), cv2.cvtColor(out['image'], cv2.COLOR_RGB2BGR), params)
            if row['hidden_px']:
                _relabel(run, by_step[step], c, out_dir, hidden.cpu().numpy(), out['background'])
            io_ms += (time.perf_counter() - t_w) * 1e3
            row['io_ms'] = round(io_ms, 2)
            if exact:
                t_m = time.perf_counter()
                _measure(row, out, comp, alt, s, kernel, f, confs.setdefault('all', _Conf()),
                         confs.setdefault(row['split'], _Conf()))
                row['measure_ms'] = round((time.perf_counter() - t_m) * 1e3, 2)
            rows.append(row)
            if key in show:
                _preview(out_dir, c, step, out)
            em.progress((k + 1) / len(order), f'{k + 1}/{len(order)} frames')
        with open(os.path.join(out_dir, 'hybrid.jsonl'), 'w', encoding='utf-8') as fh:
            for r in rows:
                fh.write(json.dumps(r, separators=(',', ':')) + '\n')
        summary = {'all': _summ(rows, confs.get('all', _Conf()), names)}
        for sp in ('test', 'train'):              # by the split of the reconstruction's nearest view
            sel = [r for r in rows if r['split'] == sp]
            if sel:
                summary['heldout' if sp == 'test' else 'train'] = _summ(sel, confs.get(sp, _Conf()), names)
        seconds = round(time.perf_counter() - t_start, 2)
        ply = os.path.join(rec.path, 'point_cloud.ply')
        what = {'schema': SCHEMA, 'source_run': run.dir, 'source_name': run.name,
                'source_run_json_sha256': PV.sha256_file(run.p('run.json')),
                'reconstruction': rec.path, 'splat_sha256': PV.sha256_file(ply),
                'reconstruction_source': src_dir, 'own_source': own_source,
                'compositing': {'method': comp.method, 'tolerance': comp.tolerance if comp.method == 'depth' else None,
                                'rule': 'CARLA\'s pixel where the actor mask (instance objects with a CARLA actor id, '
                                        'and the ego) holds' + (', and the actor\'s range ≤ the reconstruction\'s × '
                                                                f'(1 + {comp.tolerance})' if comp.method == 'depth'
                                                                else '') + '; the reconstruction elsewhere'},
                'labels': 'CARLA\'s (exact for the world)' + ('; actor pixels the depth test hid are relabelled from '
                                                              'the reconstruction' if comp.method == 'depth' else ''),
                'not_modelled': ['shadows of CARLA\'s actors on the reconstruction', 'lighting between the two worlds'],
                'aligned': 'the same CARLA world (identity)' if align is None else 'the alignment given',
                'settings': settings_doc, 'cameras': cams, 'frames': len(rows), 'seconds': seconds,
                'measured': exact, 'summary': summary if exact else {'all': {k: summary['all'][k] for k in
                                                                             ('frames', 'render_ms', 'composite_ms',
                                                                              'hidden_share')}},
                'rows': 'hybrid.jsonl',
                'inputs': PV.block({'source/run.json': run.p('run.json'), 'recon/point_cloud.ply': ply,
                                    'recon/run.json': os.path.join(rec.path, 'run.json'),
                                    'recon/views.json': os.path.join(rec.path, 'views.json')},
                                   None, settings_doc)['inputs']}
        if not exact:
            what['why_not_measured'] = ('measuring was not asked for' if not measure else
                                        'the reconstruction is not of a CARLA run of this world, so CARLA\'s frame is '
                                        'not the truth of the hybrid')
        ds = dict(run.ds)
        ds['name'] = os.path.basename(os.path.normpath(out_dir))
        ds['hybrid'] = {k: what[k] for k in ('source_run', 'reconstruction', 'splat_sha256', 'compositing', 'labels',
                                             'not_modelled', 'cameras')}
        PV.write_json(os.path.join(out_dir, 'dataset.json'), ds)
        doc = dict(run.run)
        doc.update(name=ds['name'], worker='recon.hybrid', argv=list(argv or ['recon', 'hybrid']), status='ok',
                   exit_code=0, error=None, hybrid=what,
                   notes=list(run.run.get('notes') or ()) + [
                       f'hybrid: CARLA\'s actors from {run.dir} composited over the reconstruction {rec.path} '
                       f'({comp.method}); every label is CARLA\'s; started/finished are the drive\'s times'])
        os.remove(partial)
        doc['outputs'] = PV.hash_tree(out_dir)
        PV.write_json(os.path.join(out_dir, 'run.json'), doc)
        return {'run': os.path.abspath(out_dir), 'frames': len(rows), 'method': comp.method, 'measured': exact,
                'seconds': seconds, **({'summary': summary} if exact else {})}
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _views(rec) -> tuple:
    V = [np.asarray(f['viewmat'], float).reshape(4, 4) for f in rec.frames]
    return (np.array([-v[:3, :3].T @ v[:3, 3] for v in V]), [f['split'] for f in rec.frames],
            [f['index'] for f in rec.frames])


def _nearest_view(views, vm: np.ndarray, step: int | None = None) -> tuple:
    """(the split of the reconstruction's view whose camera is nearest — 'test': a held-out pose —, its distance).
    Views within 1 cm of the nearest (a vehicle standing still) are a tie, which the view of the same step wins (a
    re-drive's step k is the source's step k)."""
    C, splits, idx = views
    d = np.linalg.norm(C - (-vm[:3, :3].T @ vm[:3, 3]), axis=1)
    ties = np.flatnonzero(d <= d.min() + 0.01)
    j = next((int(t) for t in ties if idx[t] == step), int(ties[0]))
    return splits[j], round(float(d[j]), 4)


def _offset(rec, vm: np.ndarray) -> dict:
    from .evaluate import offsets
    c = -vm[:3, :3].T @ vm[:3, 3]
    o = offsets(rec, c[None], vm[2, :3][None])[0]
    return {k: o[k] for k in ('nearest', 'lateral_m', 'along_m', 'vertical_m', 'heading_deg')}


def _measure(row, out, comp, alt, s, kernel, f, conf_all, conf_split):
    import cv2
    from .metrics import depth_errors
    img, car, a = out['image_f'], out['carla_f'], out['actor']
    bg = ~a
    m = {'whole': _region(img, car, None), 'background': _region(img, car, bg),
         'actors': _region(img, car, a, s['min_actor_px'])}
    an = a.cpu().numpy()
    near = cv2.dilate(an.astype(np.uint8), kernel) > 0
    near_t = torch.as_tensor(near & ~an, device=comp.dev)
    m['near_actors'] = _region(img, car, near_t, s['min_actor_px'])
    h, w = an.shape
    ws = D.working_size((w, h), comp.rec.work_size[0])
    if ws != (w, h):                       # also at the reconstruction's working size (area-resampled, as trained)
        import torch.nn.functional as F

        def area(x):
            return F.interpolate(x.permute(2, 0, 1)[None], size=(ws[1], ws[0]), mode='area')[0].permute(1, 2, 0)
        bg_s = torch.as_tensor(D.resize_mask(~an, ws), device=comp.dev)
        m['work'] = {'size': list(ws), 'whole': _region(area(img), area(car), None),
                     'background': _region(area(img), area(car), bg_s)}
    row['metrics'] = m
    row['alternative'] = {}
    for meth in alt:
        if meth == 'depth' and out['actor_range'] is None:
            continue
        w2 = comp.win(a, out['actor_range'], out['background']['range'], meth)
        im2 = torch.where(w2[..., None], car, out['background']['image'])
        hid = a & ~w2
        row['alternative'][meth] = {'whole': _region(im2, car, None), 'actors': _region(im2, car, a, s['min_actor_px']),
                                    'hidden_share': round(float(hid.sum()) / max(float(a.sum()), 1.0), 5)
                                    if a.any() else None}
    bgd = out['background']
    sem = D.load_semantic(f)
    if comp.votes and sem is not None:
        keep = bg.cpu().numpy()
        conf_all.add(sem[keep], bgd['semantic'].cpu().numpy()[keep])
        if conf_split is not None:
            conf_split.add(sem[keep], bgd['semantic'].cpu().numpy()[keep])
    rng = D.load_range(f)
    if rng is not None:
        rt = torch.as_tensor(rng, device=comp.dev)
        ok = bg & (rt > 0) & (rt < s['depth_max_m'])
        de = depth_errors(bgd['range'], rt, ok)
        if de.get('pixels'):
            row['depth_median_abs_rel'] = de['median_abs_rel']
            row['depth_within_10pc'] = de['within_10pc']
    bimg = (bgd['image'] * 255 + 0.5).to(torch.uint8).cpu().numpy()
    cimg = (car * 255 + 0.5).to(torch.uint8).cpu().numpy()
    row['shift_px'] = _shift_px(bimg, cimg, ~an)


def _relabel(run, rec, cam, out_dir, hidden: np.ndarray, bgd: dict):
    """Actor pixels the depth test gave to the background take the reconstruction's labels (module doc)."""
    import cv2
    for kind in ('semantic', 'instance', 'range'):
        rel = run.own(rec, cam, kind)
        if not rel:
            continue
        lab = cv2.imread(run.p(rel), cv2.IMREAD_UNCHANGED)
        if lab is None or lab.shape[:2] != hidden.shape:
            continue
        lab = lab.copy()
        if kind == 'semantic':
            lab[hidden] = bgd['semantic'].cpu().numpy()[hidden] if 'semantic' in bgd else 0
        elif kind == 'instance':
            lab[hidden] = 0
        else:
            r = bgd['range'].cpu().numpy()[hidden]
            lab[hidden] = np.where(r < 256.0, np.round(r * 256.0), 0).astype(lab.dtype)
        _write(os.path.join(out_dir, rel), lab)


def _preview(out_dir, cam, step, out):
    import cv2
    d = os.path.join(out_dir, 'hybrid-previews')
    os.makedirs(d, exist_ok=True)
    car = (out['carla_f'] * 255 + 0.5).to(torch.uint8).cpu().numpy()
    bgi = (out['background']['image'] * 255 + 0.5).to(torch.uint8).cpu().numpy()
    mk = np.repeat((out['win'].cpu().numpy()[..., None] * 255).astype(np.uint8), 3, 2)
    strip = np.concatenate([car, out['image'], bgi, mk], 1)
    cv2.imwrite(os.path.join(d, f'{cam}-{step:06d}.jpg'), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


class _Quiet:
    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass


# ================================================================================================ the drive
def redrive_log(source_dir: str, lateral_m: float = 0.0):
    """The source drive's log (core.logreplay) with its ego path moved lateral_m to the right of its heading and
    without its actors: CARLA's traffic replaces them."""
    from ..logreplay import import_log
    log = import_log('falcon', source_dir).copy()
    n_actors = len(log.actors)
    e = log.ego
    a = np.radians(e.yaw)
    e.x, e.y = e.x - float(lateral_m) * np.sin(a), e.y + float(lateral_m) * np.cos(a)
    log.actors = []
    log.meta['name'] = f'{log.meta.get("name") or os.path.basename(source_dir)} (hybrid re-drive, ' \
                       f'{float(lateral_m):+g} m)'
    log.meta.setdefault('notes', []).append(
        f'hybrid re-drive: the ego path {float(lateral_m):+g} m to the right of its heading; the source\'s '
        f'{n_actors} actors left out (CARLA\'s traffic replaces them)')
    return log


def hybrid_drive_config(project, rec, out_dir: str, port: int, tm_port: int, world: str | None = None,
                        rig: str | None = None, lateral_m: float = 0.0, seed: int | None = None,
                        vehicles: int | None = None, walkers: int | None = None, render_budget_mpx: float = 200.0,
                        workspace: str = '') -> dict:
    """The CARLA drive of a hybrid re-drive: a log replay (workers.replay) of the reconstruction's source drive with
    its ego on the logged path (driver 'log': the camera poses are the path's, moved lateral_m to the right) and no
    logged actors, plus the traffic manager's traffic — the source's numbers unless given — seeded by seed (default:
    the source's seed + 1, so the traffic differs). A shifted path may leave the driving lanes (3 m to the right of a
    lane's centre is the kerb or the next lane), which a replay refuses unless told: a shift allows it, deliberately.
    Raises HybridError."""
    from ..logreplay.commands import replay_config
    from ..logreplay.model import LogError
    from ..logreplay.plan import make_plan, town_key, world_info
    from ..provenance import read_json
    src = rec.views['source']
    if src['kind'] != 'falcon':
        raise HybridError('a hybrid re-drive follows the source drive\'s path: the reconstruction must be of a Falcon '
                          'run')
    sdrive = read_json(os.path.join(src['path'], 'drive.json'))
    town = sdrive.get('town') or ''
    worlds = project.doc.get('worlds') or []
    if world is None:
        world = next((w['id'] for w in worlds if town and town_key(w.get('town')) == town_key(town)), None)
        if world is None:
            raise HybridError(f'no world of the project is the source drive\'s map ({town}): choose one')
    winfo = world_info(project, world, workspace)
    if town and town_key(winfo.get('town')) != town_key(town):
        raise HybridError(f'the world {world} is {winfo.get("town") or winfo.get("name")}, the reconstruction\'s '
                          f'drive {town}')
    rigs = project.doc.get('rigs') or []
    rig_doc = project.item('rigs', rig) if rig else next(
        (r for r in rigs if any(x['type'] == 'camera.pinhole' and 'instance' in (x.get('labels') or [])
                                for x in r.get('sensors', []))), None)
    if rig_doc is None:
        raise HybridError(f'no rig {rig!r} in the project' if rig else
                          'no rig of the project has a pinhole camera with the instance label (the actors\' mask)')
    cams = [x for x in rig_doc['sensors'] if x['type'] == 'camera.pinhole']
    if not any({'instance', 'depth'} <= set(x.get('labels') or []) for x in cams):
        raise HybridError(f'rig {rig_doc["id"]}: a hybrid drive needs a pinhole camera with the instance and depth '
                          f'labels (the actors\' mask and range)')
    os.makedirs(out_dir, exist_ok=True)
    log = redrive_log(src['path'], lateral_m)
    log_path = log.save(os.path.join(out_dir, 'log.json'))
    seed = int(sdrive.get('seed', 0)) + 1 if seed is None else int(seed)
    try:
        plan = make_plan(log, winfo, fps=float(sdrive.get('fps') or 20.0), seed=seed, driver='log', log_path=log_path,
                         allow_mismatch=bool(lateral_m))
    except LogError as e:
        raise HybridError(str(e)) from None
    cfg = replay_config(project, plan, world, rig_doc['id'], out_dir, port, tm_port, render_budget_mpx, workspace,
                        name=f'hybrid re-drive of {os.path.basename(src["path"])} ({float(lateral_m):+g} m, seed '
                             f'{seed})')
    tr = dict(sdrive.get('traffic') or {})
    cfg['traffic'] = dict(cfg['traffic'], vehicles=int(tr.get('vehicles', 0) if vehicles is None else vehicles),
                          walkers=int(tr.get('walkers', 0) if walkers is None else walkers))
    if sdrive.get('weather'):
        cfg['weather'], cfg['weather_source'] = sdrive['weather'], sdrive.get('weather_source')
    cfg['hybrid'] = {'reconstruction': rec.path, 'lateral_m': float(lateral_m), 'seed': seed,
                     'source_drive': src['path'],
                     'note': 'the ego replays the source\'s path (driver log); the traffic is the traffic manager\'s, '
                             'reacting to the ego and to itself'}
    return cfg
