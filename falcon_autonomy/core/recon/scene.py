"""A trained reconstruction folder and its render API: pose → image, depth (range along each ray), alpha and, where
classes were voted, semantic classes (torch at import).

    rec = Recon.load('…/recon/drive')                       # point_cloud.ply, views.json, run.json
    cam = rec.camera(rec.frame(40), lateral_m=1.0)          # the 40th frame's camera, the ego 1 m to the right
    out = rec.render(cam)                                   # {'image' uint8 (H, W, 3), 'range' (H, W) m, …}

A camera is a dict: viewmat (4 × 4, world → optical), K (fx, fy, cx, cy, pixel-edge), size (w, h), and for a camera
made from a frame its ego (world ← vehicle) and mount (vehicle ← optical). Shifts are in the vehicle's own frame:
lateral_m to the right (as log replay's lateral_m), forward_m, up_m, and yaw_deg in CARLA's sense (positive turns
right); mount replaces the camera's mount with a CARLA vehicle-frame pose {x y z roll pitch yaw} (a rig's), and fov /
size replace its intrinsics with a centred pinhole of that horizontal field.

Range: the median depth — the depth of the Gaussian at which a pixel's transmittance falls below ½ — times the ray's
length per unit depth; 0 where the transmittance never falls that far (nothing the reconstruction is sure of) or where
that Gaussian is on the far shell (sky) — falcon-raw's "nothing within 256 m". It is the reconstruction's estimate of
the geometry, not ground truth.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from . import data as D
from .gaussians import Gaussians

VIEWS_SCHEMA = 'falcon.recon_views/1'
CAM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # columns: optical x, y, z in body


def pick_device(want: str = 'auto') -> torch.device:
    if want in ('', 'auto', None):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(want)


def mount_from_carla(pose: dict) -> np.ndarray:
    """vehicle ← optical (right-handed vehicle frame) of a CARLA vehicle-frame camera pose (a rig's)."""
    from ..exporters import pose_matrix, rh
    T = rh(pose_matrix(pose))
    T[:3, :3] = T[:3, :3] @ CAM_OPTICAL
    return T


def carla_pose_of(T_rh: np.ndarray, optical: bool = False) -> dict:
    """A right-handed 4 × 4 (world ← vehicle, or vehicle ← optical when optical) as CARLA's pose dict."""
    from ..rig import carla_euler
    T = np.array(T_rh, float)
    if optical:
        T[:3, :3] = T[:3, :3] @ CAM_OPTICAL.T
    Tc = D.S4 @ T @ D.S4
    roll, pitch, yaw = carla_euler(Tc[:3, :3])
    return {'x': round(float(Tc[0, 3]), 6) + 0.0, 'y': round(float(Tc[1, 3]), 6) + 0.0,
            'z': round(float(Tc[2, 3]), 6) + 0.0, 'roll': round(roll, 6) + 0.0, 'pitch': round(pitch, 6) + 0.0,
            'yaw': round(yaw, 6) + 0.0}


def shift_matrix(lateral_m=0.0, forward_m=0.0, up_m=0.0, yaw_deg=0.0) -> np.ndarray:
    """vehicle ← shifted vehicle, in the right-handed vehicle frame (x forward, y left, z up)."""
    a = -math.radians(float(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = [float(forward_m), -float(lateral_m), float(up_m)]
    return T


def frame_doc(f: D.Frame, split: str) -> dict:
    return {'index': f.index, 't': round(float(f.t), 6), 'name': f.name, 'split': split,
            'image': f.image if isinstance(f.image, str) else None, 'K': [float(v) for v in f.K],
            'size': [int(v) for v in f.size], 'viewmat': np.asarray(f.viewmat).reshape(-1).tolist(),
            'ego': np.asarray(f.ego).reshape(-1).tolist(), 'mount': np.asarray(f.mount).reshape(-1).tolist(),
            'semantic': f.semantic if isinstance(f.semantic, str) else None,
            'range': f.range if isinstance(f.range, str) else None}


class Recon:
    """A reconstruction: its Gaussians, its frames (views.json) and its run.json."""

    def __init__(self, path: str, gs: Gaussians, views: dict, run: dict, device):
        self.path, self.gs, self.views, self.run, self.device = path, gs, views, run, device
        self.work_size = tuple(views.get('work_size') or (640, 360))
        self._capture = None

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'Recon':
        from ..provenance import read_json
        path = os.path.abspath(path)
        ply = os.path.join(path, 'point_cloud.ply')
        vj = os.path.join(path, 'views.json')
        if not (os.path.isfile(ply) and os.path.isfile(vj)):
            raise D.DataError(f'{path}: not a finished reconstruction (point_cloud.ply and views.json)')
        dev = pick_device(device)
        gs = Gaussians.read_ply(ply, device=dev)
        run = read_json(os.path.join(path, 'run.json')) if os.path.isfile(os.path.join(path, 'run.json')) else {}
        return cls(path, gs, read_json(vj), run, dev)

    # ------------------------------------------------------------------------------------------- frames
    @property
    def frames(self) -> list:
        return self.views['frames']

    def frame(self, i: int) -> dict:
        return self.frames[int(i)]

    def split(self, which: str) -> list:
        return [f for f in self.frames if f['split'] == which]

    def capture(self) -> D.Capture:
        """The source re-opened (for masks and labels)."""
        if self._capture is None:
            s = self.views['source']
            self._capture = D.open_capture((s['kind'], s['path']), s.get('camera'), **(s.get('options') or {}))
        return self._capture

    def source_frame(self, f: dict) -> D.Frame:
        for fr in self.capture().frames:
            if fr.index == f['index']:
                return fr
        raise D.DataError(f'the source no longer has frame {f["index"]}')

    # ------------------------------------------------------------------------------------------- cameras
    def camera(self, f: dict, lateral_m=0.0, forward_m=0.0, up_m=0.0, yaw_deg=0.0, mount: dict | None = None,
               fov: float | None = None, size=None) -> dict:
        ego = np.asarray(f['ego'], float).reshape(4, 4) @ shift_matrix(lateral_m, forward_m, up_m, yaw_deg)
        mt = mount_from_carla(mount) if mount else np.asarray(f['mount'], float).reshape(4, 4)
        wh = tuple(size) if size else D.working_size(f['size'], self.work_size[0])
        if fov:
            from .raster import fov_K
            K = fov_K(wh[0], wh[1], float(fov))
        else:
            K = D.scaled_K(f['K'], f['size'], wh)
        return {'viewmat': D.inv(ego @ mt), 'K': tuple(float(v) for v in K), 'size': (int(wh[0]), int(wh[1])),
                'ego': ego, 'mount': mt}

    def camera_at(self, pose: dict, fov: float = 90.0, size=None) -> dict:
        """A camera at a world pose in CARLA's convention of the reconstruction's world — {x y z roll pitch yaw} of a
        CARLA camera actor (x forward, y right, z up) — with a centred pinhole of horizontal field fov."""
        from ..exporters import pose_matrix, rh
        from .raster import fov_K
        T = rh(pose_matrix(pose))
        T[:3, :3] = T[:3, :3] @ CAM_OPTICAL
        wh = tuple(size) if size else tuple(self.work_size)
        return {'viewmat': D.inv(T), 'K': fov_K(wh[0], wh[1], float(fov)), 'size': (int(wh[0]), int(wh[1]))}

    # ------------------------------------------------------------------------------------------- rendering
    @torch.no_grad()
    def render(self, cam: dict, semantic: bool = True) -> dict:
        gs = self.gs
        vm = torch.as_tensor(cam['viewmat'], dtype=torch.float32, device=self.device)
        w, h = cam['size']
        votes = gs.votes is not None and semantic
        feats = None
        if votes:
            v = gs.votes.to(self.device)
            feats = v / v.sum(-1, keepdim=True).clamp_min(1e-12)
        out = render_views(gs, vm, cam['K'], w, h, extra=feats)
        res = {'image': (out['image'].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy(),
               'image_f': out['image'].clamp(0, 1), 'range': out['range'].cpu().numpy(),
               'alpha': out['alpha'].cpu().numpy(), 'sky': out['sky'].cpu().numpy(), 'pairs': out['pairs']}
        if votes:
            cls = out['extra'].argmax(-1)
            cls = torch.where(out['alpha'] >= 0.5, cls, torch.zeros_like(cls))
            res['semantic'] = cls.to(torch.uint8).cpu().numpy()
        return res


def render_views(gs: Gaussians, vm: torch.Tensor, K, w: int, h: int, bg=None, extra=None) -> dict:
    """One render with what a view needs: the SH colour, the range along each ray — the median depth (where the
    transmittance falls below ½) × the ray's length per unit depth; 0 where it never does (nothing) or where that
    Gaussian is on the far shell (sky) —, the alpha, the sky flag and, given extra (N, C) per-Gaussian features,
    their composite. The median depth is used rather than the expected depth Σ wᵢ zᵢ / α, which mixes surfaces at
    edges and lets the far shell's residual weight pull a surface's depth far back."""
    from .raster import ray_norms, render, sh_to_color
    R, t = vm[:3, :3], vm[:3, 3]
    dirs = gs.p['means'] - (-(R.T @ t))
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    feats = sh_to_color(gs.sh(), dirs, gs.sh_degree)
    if extra is not None:
        feats = torch.cat([feats, extra], -1)
    out = render(gs.p['means'], gs.scales(), gs.p['quats'], gs.opacities(), feats, vm, K, w, h, bg,
                 median_features=gs.sky[:, None].to(feats.dtype))
    med = out['median']
    d, sky = med[..., 0], med[..., 1] > 0.5
    rng = torch.where((d > 0) & ~sky, d * ray_norms(K, w, h, vm.device, d.dtype), torch.zeros_like(d))
    res = {'image': out['image'][..., :3], 'range': rng, 'alpha': out['alpha'], 'sky': sky & (d > 0),
           'pairs': out['info']['pairs']}
    if extra is not None:
        res['extra'] = out['image'][..., 3:]
    return res


def write_views(path: str, cap: D.Capture, train: list, test: list, work_size, options: dict):
    from ..provenance import write_json
    split = {id(f): 'train' for f in train}
    split.update({id(f): 'test' for f in test})
    write_json(path, {'schema': VIEWS_SCHEMA, 'world': cap.meta.get('world'),
                      'carla_exact': bool(cap.meta.get('carla_exact')), 'town': cap.meta.get('town'),
                      'source': {'kind': cap.kind, 'path': cap.path, 'camera': cap.camera, 'options': options},
                      'work_size': list(work_size),
                      'frames': [frame_doc(f, split.get(id(f), 'unused')) for f in cap.frames]})


def read_json_lines(path: str) -> list:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]
