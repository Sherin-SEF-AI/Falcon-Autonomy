"""Ranges to the road users in a dashcam frame, in metres, with their uncertainty (numpy and OpenCV at import; torch,
transformers or onnxruntime only inside the depth runners).

The camera (Camera). A pinhole: focal length f (pixels; from the lens's horizontal field of view unless given),
principal point at the image centre, mounted `height_m` above the road (default 1.3 m: a dashcam behind a car's
windscreen; give yours), pitched down by θ, no roll. Its horizon row v_h = c_y − f·tan θ is estimated per clip from
the focus of expansion (estimate_horizon: while the car drives straight at speed, the image flow of the static scene
radiates from the point the car moves towards, which lies on the horizon; the median over frame pairs), or given, or
— with too few usable pairs — the image centre (θ = 0), recorded as such. Lens distortion is not modelled.

Two ranges per box, then fused:

1. Geometric (flat ground): the box's bottom-centre pixel cast onto the road plane — core.incident.ground_point with
   this camera's view matrix — so a box whose bottom edge is v rows below the horizon is at forward distance
   ≈ f·H / (v − v_h). Its σ: a 2 px error in the bottom edge (σ_Z = Z²/(f·H)·2 px) and 5 % for a road that is not
   flat (heuristic).
2. From the depth network. Depth Anything V2 Small (the default; core.perception_models) gives relative depth: an
   affine-invariant inverse depth d = α/Z + β (larger is nearer). On a flat road under a camera with no roll the true
   inverse depth is linear in the row: 1/Z = cos θ·(v − v_h)/(f·H). So the network's d over the road is a plane
   d = p·u + q·v + r, fitted by RANSAC (fit_ground) to pixels below the horizon outside every detected box; with
   the horizon known, α = q·f·H/cos θ and β = r + q·v_h put every pixel in metres: Z = α/(d − β). A box's d is the
   median over the lower half of its central third. σ: the spread of α and β over a bootstrap of the RANSAC
   inliers, plus 10 % for the network (heuristic). Metric3D v2 (opt-in) gives metric depth directly; its scale is
   still checked against the camera height on the same road fit (1/Z regressed on v gives the height it implies;
   the depth is rescaled by the ratio, recorded).

Fusion (fuse): inverse-variance weighting of the two forward distances; the lateral offset follows the bottom-centre
ray at the fused distance. When they disagree by more than 3 σ combined, the box is flagged `disagree` and the
network's one is kept for a box cut by the image's bottom edge (its bottom is not its ground contact), the
geometric one otherwise; σ is then at least half their difference.

Frames: the vehicle frame of core.incident — x forward, y left, z up, the origin on the road under the camera.
"""
from __future__ import annotations

import math
import os

import numpy as np

DEFAULT_HEIGHT_M = 1.3
DEFAULT_HFOV_DEG = 110.0
GEO_PX_SIGMA = 2.0
GEO_FLAT_REL = 0.05
NET_REL = 0.10
DISAGREE_SIGMA = 3.0


class DepthError(Exception):
    pass


class Camera:
    """A dashcam's pinhole (module doc). width, height in pixels of the frames processed."""

    def __init__(self, width: int, height: int, f: float | None = None, hfov_deg: float = DEFAULT_HFOV_DEG,
                 height_m: float = DEFAULT_HEIGHT_M, horizon_v: float | None = None, cy: float | None = None):
        self.width, self.height = int(width), int(height)
        self.f = float(f) if f else self.width / 2.0 / math.tan(math.radians(hfov_deg) / 2.0)
        self.cx = self.width / 2.0
        self.cy = float(cy) if cy is not None else self.height / 2.0
        self.height_m = float(height_m)
        self.horizon_v = float(horizon_v) if horizon_v is not None else self.cy
        self.horizon_source = 'given' if horizon_v is not None else 'image centre (level camera assumed)'

    @property
    def pitch(self) -> float:
        """Radians, positive down."""
        return math.atan2(self.cy - self.horizon_v, self.f)

    @property
    def K(self) -> list:
        return [self.f, self.f, self.cx, self.cy]

    def doc(self) -> dict:
        return {'size': [self.width, self.height], 'f_px': round(self.f, 3), 'cx': self.cx, 'cy': self.cy,
                'height_m': self.height_m, 'horizon_v': round(self.horizon_v, 2),
                'pitch_deg': round(math.degrees(self.pitch), 3), 'horizon_source': self.horizon_source,
                'hfov_deg': round(math.degrees(2 * math.atan(self.width / 2.0 / self.f)), 2)}

    def mount(self) -> np.ndarray:
        """camera (OpenCV: x right, y down, z forward) ← vehicle (x forward, y left, z up), as a 3 × 3 rotation."""
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        R0 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return Rx @ R0

    def view(self, ego: np.ndarray | None = None) -> np.ndarray:
        """world → camera (4 × 4) for a vehicle pose ego (world ← vehicle, 4 × 4; identity: the vehicle frame), the
        camera height_m above the vehicle's origin."""
        E = np.eye(4) if ego is None else np.asarray(ego, float)
        T_vc = np.eye(4)                                   # vehicle ← camera
        T_vc[:3, :3] = self.mount().T
        T_vc[:3, 3] = [0.0, 0.0, self.height_m]
        return np.linalg.inv(E @ T_vc)

    def ray_vehicle(self, u, v) -> np.ndarray:
        """The pixel's ray in the vehicle frame (not normalised; its camera-z component is 1)."""
        d = np.array([(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0])
        return self.mount().T @ d


# ------------------------------------------------------------------------------------------------ geometric range
def geo_point(cam: Camera, box) -> dict | None:
    """The flat-ground point under a box's bottom centre (core.incident.ground_point), in the vehicle frame, with
    σ of its forward distance; None above the horizon."""
    from .incident import ground_point
    x1, y1, x2, y2 = (float(v) for v in box)
    u, v = (x1 + x2) / 2.0, y2
    hit = ground_point(cam.K, cam.view(), u, v)
    if hit is None:
        return None
    p, _C = hit
    fwd, left = float(p[0]), float(p[1])
    if fwd <= 0:
        return None
    sig = math.hypot(fwd * fwd / (cam.f * cam.height_m) * GEO_PX_SIGMA, GEO_FLAT_REL * fwd)
    return {'forward': fwd, 'left': left, 'sigma': sig, 'cut': y2 >= cam.height - 1.0}


# ------------------------------------------------------------------------------------------------ the road fit
def ground_samples(shape, cam: Camera, boxes=(), step: int = 8, margin_px: float = 6.0, bottom_skip: float = 0.0,
                   v_from: float | None = None):
    """Pixel coordinates (u, v) on a grid below the horizon (or below row v_from), outside every box (where the road
    should be)."""
    h, w = shape[:2]
    v0 = int(max((cam.horizon_v if v_from is None else v_from) + margin_px, 0)) + 1
    v1 = int(h * (1.0 - bottom_skip))
    us, vs = np.meshgrid(np.arange(step // 2, w, step), np.arange(v0, v1, step))
    keep = np.ones(us.shape, bool)
    for b in boxes:
        x1, y1, x2, y2 = b
        keep &= ~((us >= x1 - 2) & (us <= x2 + 2) & (vs >= y1 - 2) & (vs <= y2 + 2))
    return us[keep].astype(float), vs[keep].astype(float)


def fit_plane(depth: np.ndarray, cam: Camera, boxes=(), v_min: float | None = None, metric: bool = False,
              seed: int = 0, iters: int = 200, boot: int = 30) -> dict | None:
    """RANSAC plane d = p·u + q·v + r through the depth map's road pixels (rows below v_min — default the horizon
    plus a margin — outside every box), refitted by least squares on the inliers, with `boot` bootstrap refits of
    them. For metric depth (metric=True) the plane is of 1/Z. None when the road does not show (too few pixels, or a
    plane that does not grow nearer down the image). Independent of the horizon: scale_plane() uses it."""
    if v_min is None:
        us, vs = ground_samples(depth.shape, cam, boxes)
    else:
        us, vs = ground_samples(depth.shape, cam, boxes, margin_px=0.0, v_from=v_min)
    if len(us) < 30:
        return None
    vals = depth[vs.astype(int), us.astype(int)].astype(float)
    if metric:
        ok = vals > 0
        us, vs, vals = us[ok], vs[ok], 1.0 / np.where(ok, vals, 1.0)[ok]
    ok = np.isfinite(vals)
    us, vs, vals = us[ok], vs[ok], vals[ok]
    if len(us) < 30:
        return None
    A = np.c_[us, vs, np.ones_like(us)]
    spread = float(np.percentile(vals, 95) - np.percentile(vals, 5)) or 1e-6
    th = 0.03 * spread
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        idx = rng.choice(len(us), 3, replace=False)
        try:
            x = np.linalg.solve(A[idx], vals[idx])
        except np.linalg.LinAlgError:
            continue
        n = int((np.abs(A @ x - vals) < th).sum())
        if n > best_n and x[1] > 0:
            best, best_n = x, n
    if best is None or best_n < 20:
        return None
    inl = np.abs(A @ best - vals) < th
    x, *_ = np.linalg.lstsq(A[inl], vals[inl], rcond=None)
    if x[1] <= 0:
        return None
    Ai, vi = A[inl], vals[inl]
    bs = []
    for _ in range(boot):
        k = rng.integers(0, len(vi), len(vi))
        xb, *_ = np.linalg.lstsq(Ai[k], vi[k], rcond=None)
        bs.append([float(v) for v in xb])
    return {'p': float(x[0]), 'q': float(x[1]), 'r': float(x[2]), 'inliers': int(inl.sum()),
            'samples': int(len(us)), 'roll_slope': float(x[0] / x[1]), 'metric': bool(metric), 'boot': bs}


def scale_plane(plane: dict | None, cam: Camera) -> dict | None:
    """With the horizon known (module doc): for relative depth α = q·f·H/cos θ, β = r + p·c_x + q·v_h (Z = α/(d −
    β)) and their bootstrap σ; for metric depth the camera height the plane implies (H' = cos θ/(f·q)) and the scale
    H/H' that corrects it."""
    if plane is None:
        return None
    cos_t = math.cos(cam.pitch)
    out = {k: plane[k] for k in ('p', 'q', 'r', 'inliers', 'samples', 'roll_slope')}

    def ab(x):
        return x[1] * cam.f * cam.height_m / cos_t, x[2] + x[0] * cam.cx + x[1] * cam.horizon_v
    if plane['metric']:
        h_implied = cos_t / (cam.f * plane['q'])
        out.update(height_implied_m=float(h_implied), scale=float(cam.height_m / h_implied))
        return out
    alpha, beta = ab((plane['p'], plane['q'], plane['r']))
    boots = [ab(x) for x in plane.get('boot') or () if x[1] > 0]
    out.update(alpha=float(alpha), beta=float(beta),
               alpha_sigma=float(np.std([b[0] for b in boots])) if boots else None,
               beta_sigma=float(np.std([b[1] for b in boots])) if boots else None)
    return out


def fit_ground(depth: np.ndarray, cam: Camera, boxes=(), metric: bool = False, seed: int = 0) -> dict | None:
    """fit_plane then scale_plane with the camera's horizon."""
    return scale_plane(fit_plane(depth, cam, boxes, metric=metric, seed=seed), cam)


def box_value(depth: np.ndarray, box) -> float | None:
    """The depth map's value for a box: the median over the lower half of its central third (its rear or front
    face, where a vehicle's ground contact is)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = depth.shape[:2]
    cx0, cx1 = x1 + (x2 - x1) / 3.0, x2 - (x2 - x1) / 3.0
    a, b = int(max(0, math.floor(cx0))), int(min(w, math.ceil(cx1)))
    c, d = int(max(0, math.floor((y1 + y2) / 2.0))), int(min(h, math.ceil(y2)))
    if b <= a or d <= c:
        return None
    val = float(np.median(depth[c:d, a:b]))
    return val if math.isfinite(val) else None


def net_point(val: float | None, kind: str, fit: dict | None, cam: Camera, box) -> dict | None:
    """The box's point from the depth network's value for it (box_value; module doc), vehicle frame, with σ; None
    without a value or — for a relative map — a usable road fit."""
    if val is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    if kind == 'metric':
        scale = (fit or {}).get('scale') or 1.0
        Z = val * scale
        sZ = NET_REL * Z
    else:
        if not fit or 'alpha' not in fit or val <= fit['beta']:
            return None
        Z = fit['alpha'] / (val - fit['beta'])
        sa, sb = fit.get('alpha_sigma') or 0.0, fit.get('beta_sigma') or 0.0
        dz_da = 1.0 / (val - fit['beta'])
        dz_db = fit['alpha'] / (val - fit['beta']) ** 2
        sZ = math.sqrt((dz_da * sa) ** 2 + (dz_db * sb) ** 2 + (NET_REL * Z) ** 2)
    if not (0 < Z < 500):
        return None
    u, v = (x1 + x2) / 2.0, min(y2, cam.height - 1.0)
    ray = cam.ray_vehicle(u, v)                              # camera z = 1: scale by Z for the point
    P = ray * Z + np.array([0.0, 0.0, cam.height_m])
    return {'forward': float(P[0]), 'left': float(P[1]), 'sigma': float(sZ * abs(ray[0])), 'depth': float(Z)}


def fuse(geo: dict | None, net: dict | None) -> dict | None:
    """Inverse-variance fusion of the two forward distances (module doc); the lateral offset along the box's
    bottom-centre ray at the fused distance. {'forward', 'left', 'sigma', 'source', 'geo', 'net', 'disagree'}."""
    if geo is None and net is None:
        return None
    if net is None or geo is None:
        one = geo or net
        return {'forward': one['forward'], 'left': one['left'], 'sigma': one['sigma'],
                'source': 'geometric' if geo else 'network', 'geo': _r(geo), 'net': _r(net), 'disagree': False}
    s2 = math.hypot(geo['sigma'], net['sigma'])
    if abs(geo['forward'] - net['forward']) > DISAGREE_SIGMA * s2:
        keep = net if geo.get('cut') else geo               # a box cut by the bottom edge: not its ground contact
        return {'forward': keep['forward'], 'left': keep['left'], 'sigma': max(keep['sigma'], abs(
            geo['forward'] - net['forward']) / 2.0), 'source': 'geometric' if keep is geo else 'network',
                'geo': _r(geo), 'net': _r(net), 'disagree': True}
    wg, wn = 1.0 / geo['sigma'] ** 2, 1.0 / net['sigma'] ** 2
    fwd = (wg * geo['forward'] + wn * net['forward']) / (wg + wn)
    slope = geo['left'] / geo['forward'] if geo['forward'] > 0 else 0.0     # the bottom-centre ray's bearing
    return {'forward': fwd, 'left': slope * fwd, 'sigma': math.sqrt(1.0 / (wg + wn)), 'source': 'fused',
            'geo': _r(geo), 'net': _r(net), 'disagree': False}


def _r(x):
    return None if x is None else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}


# ------------------------------------------------------------------------------------------------ the horizon
def foe(prev_gray: np.ndarray, gray: np.ndarray, boxes=(), min_flow_px: float = 0.5,
        max_flow_rel: float | None = 0.05) -> dict | None:
    """The focus of expansion between two frames: sparse Lucas–Kanade flow on corners outside the boxes (kept when
    tracked back to within 0.5 px of the start, moving at least min_flow_px and at most max_flow_rel of the image's
    height — the road near the bottom expands too fast for the window's translation model, which then underestimates
    the flow and pulls the focus up), the point nearest every flow line (least squares, then reweighted twice to drop
    moving points). None with too little."""
    import cv2
    mask = np.full(gray.shape, 255, np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
    p0 = cv2.goodFeaturesToTrack(prev_gray, 400, 0.001, 6, mask=mask)
    if p0 is None or len(p0) < 20:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(21, 21), maxLevel=3)
    fb = np.linalg.norm(back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)          # tracked back to where it started
    a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    d = b - a
    n = np.linalg.norm(d, axis=1)
    keep = n > min_flow_px
    if max_flow_rel:
        keep &= n < max_flow_rel * gray.shape[0]
    a, d, n = a[keep], d[keep], n[keep]
    if len(a) < 15:
        return None
    dirs = d / n[:, None]
    w = np.ones(len(a))
    x = None
    for _ in range(3):
        M = np.zeros((2, 2))
        rhs = np.zeros(2)
        for p, di, wi in zip(a, dirs, w):
            P = np.eye(2) - np.outer(di, di)
            M += wi * P
            rhs += wi * P @ p
        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return None
        perp = np.abs((a - x)[:, 0] * dirs[:, 1] - (a - x)[:, 1] * dirs[:, 0])
        s = max(float(np.median(perp)) * 1.4826, 1.0)
        w = 1.0 / np.maximum(1.0, perp / (2.0 * s))
        outward = np.sum((a - x) * dirs, axis=1) > 0               # flow radiates away from the FOE
        w = w * outward
    return {'u': float(x[0]), 'v': float(x[1]), 'points': int(len(a)), 'median_flow_px': float(np.median(n))}


def estimate_horizon(foes: list, cam: Camera, min_pairs: int = 5) -> dict:
    """The horizon row from per-pair foci of expansion (module doc): their median row when at least min_pairs lie
    within the image; otherwise the image centre. Updates cam and returns what was used."""
    vs = [f['v'] for f in foes if f and 0 <= f['v'] < cam.height and 0 <= f['u'] < cam.width]
    if len(vs) >= min_pairs:
        cam.horizon_v = float(np.median(vs))
        cam.horizon_source = f'focus of expansion (median of {len(vs)} frame pairs driving straight)'
        return {'horizon_v': cam.horizon_v, 'pairs': len(vs), 'spread_px': float(np.percentile(vs, 75) -
                                                                                 np.percentile(vs, 25))}
    cam.horizon_source = f'image centre (level camera assumed: {len(vs)} usable frame pairs, {min_pairs} needed)'
    return {'horizon_v': cam.horizon_v, 'pairs': len(vs), 'spread_px': None}


# ------------------------------------------------------------------------------------------------ depth runners
class DepthAnythingV2:
    """Depth Anything V2 Small through transformers (relative depth), loaded offline from its verified folder."""
    kind = 'relative'

    def __init__(self, name: str = 'da2-small', models_dir: str = '', device: str = 'auto', fp16: bool = True,
                 gpu_memory_fraction: float = 0.5):
        from . import perception_models as PM
        folder = PM.load_folder(name, models_dir)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(float(gpu_memory_fraction), self.device.index or 0)
        self.dtype = torch.float16 if fp16 and self.device.type == 'cuda' else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(folder, local_files_only=True).to(
            self.device, self.dtype).eval()
        self.name = name
        self.info = {'model': name, 'licence': PM.CANDIDATES[name]['licence'], 'device': str(self.device),
                     'dtype': str(self.dtype).replace('torch.', ''), 'kind': self.kind}

    def __call__(self, frames: list) -> list:
        """frames [(n, rgb)] → relative depth maps (inverse depth, larger nearer) at each frame's size."""
        import torch
        images = [rgb for _n, rgb in frames]
        enc = self.processor(images=images, return_tensors='pt')
        x = enc['pixel_values'].to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model(pixel_values=x).predicted_depth.float()
        maps = []
        for k, im in enumerate(images):
            h, w = im.shape[:2]
            m = torch.nn.functional.interpolate(out[k][None, None], size=(h, w), mode='bicubic',
                                                align_corners=False)[0, 0]
            maps.append(m.cpu().numpy())
        return maps


class Metric3DOnnx:
    """Metric3D v2 ViT-S as ONNX through onnxruntime (metric depth; opt-in: core.perception_models). Its input is the
    frame resized to fit 616 × 1064 keeping the aspect, padded with the mean, normalised with ImageNet's mean and std
    in 0–255 (Metric3D's own preprocessing; read from the folder's preprocessor_config.json when it states them); its
    output is depth for a canonical camera of focal length 1000 px, rescaled here by f_resized / 1000."""
    kind = 'metric'

    def __init__(self, cam_f: float, name: str = 'metric3d-small', models_dir: str = '', device: str = 'auto',
                 accept_unstated: bool = False):
        import json
        import onnxruntime as ort
        from . import perception_models as PM
        folder = PM.load_folder(name, models_dir, accept_unstated)
        prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device in ('auto', 'cuda') else \
            ['CPUExecutionProvider']
        prov = [p for p in prov if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(os.path.join(folder, 'onnx', 'model.onnx'), providers=prov)
        cfg = {}
        pc = os.path.join(folder, 'preprocessor_config.json')
        if os.path.isfile(pc):
            with open(pc, encoding='utf-8') as fh:
                cfg = json.load(fh)
        self.mean = np.array(cfg.get('image_mean') or [123.675, 116.28, 103.53], np.float32)
        self.std = np.array(cfg.get('image_std') or [58.395, 57.12, 57.375], np.float32)
        if self.mean.max() <= 1.0:
            self.mean, self.std = self.mean * 255.0, self.std * 255.0
        self.size = (616, 1064)
        self.cam_f = float(cam_f)
        self.inp = self.sess.get_inputs()[0].name
        self.name = name
        self.info = {'model': name, 'licence': PM.CANDIDATES[name]['licence'], 'providers': self.sess.get_providers(),
                     'kind': self.kind, 'accepted_unstated_terms': True}

    def __call__(self, frames: list) -> list:
        """frames [(n, rgb)] → metric depth maps (metres along the optical axis) at each frame's size."""
        import cv2
        out = []
        H, W = self.size
        for _n, im in frames:
            h, w = im.shape[:2]
            k = min(H / h, W / w)
            nh, nw = int(round(h * k)), int(round(w * k))
            x = np.empty((H, W, 3), np.float32)
            x[:] = self.mean
            x[:nh, :nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
            pred = np.asarray(self.sess.run(None, {self.inp: x})[0], np.float32).reshape(-1, H, W)[0]
            pred = pred[:nh, :nw] * (self.cam_f * k / 1000.0)
            out.append(cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR))
        return out


def make_depth(spec: str, cam_f: float, models_dir: str = '', device: str = 'auto', accept_unstated: bool = False):
    """'da2-small' (default), 'metric3d-small' (opt-in) or 'none' (geometric ranges only)."""
    from . import perception_models as PM
    if spec in ('none', '', None):
        return None
    why = PM.refused_reason(spec)
    if why:
        raise DepthError(why)
    try:
        if spec == 'da2-small':
            return DepthAnythingV2(spec, models_dir, device)
        if spec == 'metric3d-small':
            return Metric3DOnnx(cam_f, spec, models_dir, device, accept_unstated)
    except PM.PerceptionModelError as e:
        raise DepthError(str(e)) from None
    raise DepthError(f'unknown depth model {spec!r}: da2-small, metric3d-small (opt-in) or none')
