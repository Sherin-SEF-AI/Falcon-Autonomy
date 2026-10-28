"""The camera model: lens models, intrinsics, and the pinhole views CARLA renders for a camera with their remap and
seam-blend tables (numpy only at import; OpenCV is imported by the functions that remap images). Ported from Vantage
(vantage.core.lens's projection maths, vantage.core.simrig's views and tables, the CARLA bridge's image assembly), so it
runs in CARLA's own Python as well as in Falcon.

The input is a Falcon sensor dict: {'type': 'camera.pinhole' | 'camera.fisheye', 'width', 'height', 'fov' (a pinhole's
horizontal field, degrees), 'lens': {'model', 'f_px', 'cx', 'cy', 'k': [k1, k2, k3, k4], 'fov' (the lens's whole
field, degrees)} (a fisheye), 'pose': {x y z roll pitch yaw} (CARLA's vehicle frame)}.

Lens models
-----------
r(θ) is the image height in pixels of a ray θ radians off the optical axis, f the focal length in px (px/rad at the
centre): rectilinear f·tan θ (a pinhole), equidistant f·θ, equisolid 2f·sin(θ/2), stereographic 2f·tan(θ/2),
orthographic f·sin θ, kannala_brandt f·θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸). Pixel coordinates are pixel-edge
coordinates: pixel i covers [i, i+1), the image centre is (W/2, H/2), so cx = W/2 is a centred lens (as CARLA's
pinhole and Falcon's K). OpenCV coordinates (pixel centres on integers, as cv2.remap reads) are these − 0.5.
A fisheye without f_px gets, as a heuristic, an equidistant lens whose image circle (its 'fov') fits the shorter image
side (core.budget.fisheye_focal_px, so the cost and the plan agree), whatever its model.

CARLA (0.9.16) and how a camera is rendered (measured by Vantage on a 0.9.16 server, RTX 5080, Town10HD)
--------------------------------------------------------------------------------------------------------------------
* sensor.camera.rgb is a pinhole camera only: square pixels, image_size_x/_y, fov = the horizontal field (f = W / (2
  tan(fov/2)), principal point at the image centre), BGRA 8-bit raw_data; its lens-distortion post-process (lens_k,
  lens_kcube, lens_circle_*) is no calibrated lens model. Its image axes: the optical axis along the actor's +x, image
  right its +y, image down its −z.
* A pinhole sensor is rendered as itself: one view, the camera's own size and field, the identity remap.
* Any other lens is rendered as pinhole views from one optical centre and remapped through the lens model: one view
  when every ray of the image is within SINGLE_VIEW_DEG (60°) of the axis, else the cube faces the image needs (front,
  left, right, up, down, back), each SEAM_BLEND_DEG / 2 (1.5°) past its seams, with every ray at least FACE_MARGIN_PX
  (1.5 px) inside its face. A face's size N = 2·⌈f·tan(45° + SEAM_BLEND_DEG/2) + FACE_MARGIN_PX⌉ = 2·⌈f·tan(46.5°) +
  1.5⌉ samples the lens's centre at its own density (f the lens's px/rad times oversample); views are capped at
  VIEW_MAX_PX a side.
* The remap tables hold, per output pixel, where it reads in the views stacked vertically into one atlas image (−1
  where the pixel sees nothing: outside the lens's field). cv2.remap(atlas, map_x, map_y, INTER_LINEAR,
  BORDER_CONSTANT) gives the image.
* All views of a camera render at the same instant with the same manual exposure, no motion blur, bloom or lens flare
  (each would differ per view and show as seams) and CARLA's lens effect off (VIEW_ATTRIBUTES).
* Unreal's default vignette (intensity 0.4, applied in linear light before the tone curve; no blueprint attribute)
  darkens every view towards its own edges: falcon_autonomy.sim.tonemap takes it out of each sample in linear light
  through CARLA's camera response, measured from exposure brackets (Debevec–Malik). devignette_layers() builds the
  correction for a plan (one per layer: the face each pixel reads first, then its two blend samples); Vantage applies
  it to every camera unless asked to keep the vignette.
* The views of a moving camera render slightly differently (temporal anti-aliasing reprojects each view's history
  along its own image motion; static views agree): about 4 levels where two faces meet. blend_tables feathers the faces
  across a SEAM_BLEND_DEG band at each seam; the step became under 1 level.
* Geometry: lidar points projected with the pinhole convention here (f = W / 2 tan(fov/2), centre at W/2 − 0.5 in
  OpenCV coordinates) fall on CARLA's depth image within 0.33 mm (median), and the colour image's edges lie on its
  semantic segmentation's within 0.1 px, static and moving.

Frames: a sensor's optical frame is OpenCV's (x right, y down, z along the view); CARLA_TO_OPTICAL maps CARLA's actor
axes (x forward, y right, z up) into it. A view's rotation is given relative to the sensor and as a pose in the vehicle
frame (the sensor's location, the composed rotation), both as CARLA's roll / pitch / yaw.

Render cost: a camera renders its views' pixels once for the image and once per label camera (core.budget's rule);
camera_mpx() counts them from the views actually planned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sim.frames import CARLA_TO_OPTICAL, carla_euler, carla_matrix
from . import budget

MODELS = ('rectilinear', 'equidistant', 'equisolid', 'stereographic', 'orthographic', 'kannala_brandt')
MODEL_ALIASES = {'pinhole': 'rectilinear', 'kb': 'kannala_brandt', 'kannala-brandt': 'kannala_brandt',
                 'fisheye': 'equidistant'}
# cube faces: columns are the face's optical x, y, z in the camera's optical frame (z = where the face looks)
FACES = {
    'front': np.eye(3),
    'right': np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    'left': np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    'down': np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    'up': np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    'back': np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
}
SINGLE_VIEW_DEG = 60.0          # a camera whose rays all lie within this of its axis renders as one pinhole view
FACE_MARGIN_PX = 1.5            # every ray lands at least this far inside its cube face (bilinear reads stay inside)
SEAM_BLEND_DEG = 3.0            # cube faces overlap across each seam by this much (half past each side), feathered
VIEW_MAX_PX = 4096              # largest view CARLA is asked to render (per side)
RECTILINEAR_MAX_DEG = 89.0      # a rectilinear lens's field is cut at this angle off the axis
# blueprint attributes of every view of a remapped camera, so the views agree (seams) and CARLA adds no lens effect
VIEW_ATTRIBUTES = {'lens_k': 0.0, 'lens_kcube': 0.0, 'lens_circle_multiplier': 0.0, 'lens_circle_falloff': 0.0,
                   'motion_blur_intensity': 0.0, 'motion_blur_max_distortion': 0.0, 'lens_flare_intensity': 0.0,
                   'bloom_intensity': 0.0, 'chromatic_aberration_intensity': 0.0, 'blur_amount': 0.0,
                   'blur_radius': 0.0, 'exposure_mode': 'manual'}
# the label cameras beside each view (semantic, instance, depth): the same size, field, pose and tick, no lens effect
LABEL_VIEW_ATTRIBUTES = {'lens_k': 0.0, 'lens_kcube': 0.0, 'lens_circle_multiplier': 0.0, 'lens_circle_falloff': 0.0}


# ---------------------------------------------------------------- projection maths (vantage.core.lens)
def model_name(model) -> str:
    m = str(model or '').strip().lower()
    m = MODEL_ALIASES.get(m, m)
    if m not in MODELS:
        raise ValueError(f'unknown lens model {model!r} (one of {", ".join(MODELS)})')
    return m


def r_of_theta(theta, f, model, k=(0.0, 0.0, 0.0, 0.0)):
    """Image height (the unit of f) of a ray theta radians off the axis."""
    t = np.asarray(theta, float)
    if model == 'rectilinear':
        return f * np.tan(np.minimum(t, math.radians(89.9)))
    if model == 'equidistant':
        return f * t
    if model == 'equisolid':
        return 2 * f * np.sin(t / 2)
    if model == 'stereographic':
        return 2 * f * np.tan(np.minimum(t, math.radians(179)) / 2)
    if model == 'orthographic':
        return f * np.sin(np.minimum(t, math.pi / 2))
    if model == 'kannala_brandt':
        k1, k2, k3, k4 = (list(k) + [0.0] * 4)[:4]
        t2 = t * t
        return f * t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    raise ValueError(model)


def dr_dtheta(theta, f, model, k=(0.0, 0.0, 0.0, 0.0)):
    t = np.asarray(theta, float)
    if model == 'rectilinear':
        c = np.cos(np.minimum(t, math.radians(89.9)))
        return f / (c * c)
    if model == 'equidistant':
        return f * np.ones_like(t)
    if model == 'equisolid':
        return f * np.cos(t / 2)
    if model == 'stereographic':
        c = np.cos(t / 2)
        return f / (c * c)
    if model == 'orthographic':
        return f * np.cos(np.minimum(t, math.pi / 2))
    if model == 'kannala_brandt':
        k1, k2, k3, k4 = (list(k) + [0.0] * 4)[:4]
        t2 = t * t
        return f * (1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4))))
    raise ValueError(model)


def theta_of_r(r, f, model, k=(0.0, 0.0, 0.0, 0.0)):
    """The inverse of r_of_theta (Kannala–Brandt by 12 Newton steps, clipped to [0, π])."""
    x = np.asarray(r, float) / f
    if model == 'rectilinear':
        return np.arctan(x)
    if model == 'equidistant':
        return x
    if model == 'equisolid':
        return 2 * np.arcsin(np.clip(x / 2, -1, 1))
    if model == 'stereographic':
        return 2 * np.arctan(x / 2)
    if model == 'orthographic':
        return np.arcsin(np.clip(x, -1, 1))
    if model == 'kannala_brandt':
        t = np.array(x, float, copy=True)
        for _ in range(12):
            g = r_of_theta(t, 1.0, model, k) - x
            d = dr_dtheta(t, 1.0, model, k)
            t = t - g / np.where(np.abs(d) < 1e-9, 1e-9, d)
            t = np.clip(t, 0, math.pi)
        return t
    raise ValueError(model)


@dataclass
class Intrinsics:
    """A camera's lens resolved for its image (pixel-edge coordinates, see the module doc)."""
    width: int
    height: int
    f_px: float
    model: str
    k: tuple
    cx: float
    cy: float
    max_theta: float          # radians: the lens's field (half its 'fov'), cut by the model's own limit
    aspect: float = 1.0       # fy / fx
    f_heuristic: bool = False

    def project_dirs(self, d_opt):
        """Optical-frame directions (K, 3) -> (u, v, theta, inside the field and the image)."""
        d_opt = np.asarray(d_opt, float)
        x, y, z = d_opt[..., 0], d_opt[..., 1], d_opt[..., 2]
        rho = np.hypot(x, y)
        theta = np.arctan2(rho, z)
        r = r_of_theta(theta, self.f_px, self.model, self.k)
        s = np.where(rho > 1e-12, r / np.where(rho > 1e-12, rho, 1.0), 0.0)
        u = self.cx + x * s
        v = self.cy + y * s * self.aspect
        ok = (theta <= self.max_theta) & (u >= 0) & (u <= self.width) & (v >= 0) & (v <= self.height)
        if self.model == 'rectilinear':
            ok &= z > 1e-6
        return u, v, theta, ok

    def unproject(self, u, v):
        """Pixel coordinates -> optical-frame unit directions and theta."""
        x = np.asarray(u, float) - self.cx
        y = (np.asarray(v, float) - self.cy) / self.aspect
        r = np.hypot(x, y)
        theta = theta_of_r(r, self.f_px, self.model, self.k)
        s = np.where(r > 1e-12, np.sin(theta) / np.where(r > 1e-12, r, 1.0), 0.0)
        d = np.stack([x * s, y * s, np.cos(theta)], axis=-1)
        return d, theta

    @property
    def K(self) -> list:
        """3 × 3 camera matrix (pixel-edge coordinates; a fisheye's: its focal length at the centre)."""
        return [[float(self.f_px), 0.0, float(self.cx)], [0.0, float(self.f_px * self.aspect), float(self.cy)],
                [0.0, 0.0, 1.0]]

    def to_dict(self) -> dict:
        """The lens block (JSON) core.groundtruth.project reads."""
        return {'model': self.model, 'f_px': float(self.f_px), 'k': [float(v) for v in self.k],
                'cx': float(self.cx), 'cy': float(self.cy), 'aspect': float(self.aspect),
                'max_theta_deg': math.degrees(self.max_theta), 'width': int(self.width), 'height': int(self.height),
                'f_heuristic': bool(self.f_heuristic)}


def pinhole_f(width, fov_deg) -> float:
    """CARLA's pinhole focal length: f = W / (2 tan(fov/2))."""
    return float(width) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))


def intrinsics(sensor: dict) -> Intrinsics:
    """The sensor's lens: a pinhole camera's from its fov (CARLA's), a fisheye's from its lens block."""
    t = sensor.get('type')
    w, h = int(sensor['width']), int(sensor['height'])
    if t == 'camera.pinhole':
        return Intrinsics(w, h, pinhole_f(w, sensor['fov']), 'rectilinear', (0.0, 0.0, 0.0, 0.0), w / 2.0, h / 2.0,
                          math.radians(RECTILINEAR_MAX_DEG))
    if t != 'camera.fisheye':
        raise ValueError(f'not a camera: {t!r}')
    L = sensor.get('lens') or {}
    model = model_name(L.get('model', 'equidistant'))
    fov = float(L.get('fov') or 190.0)
    f = L.get('f_px')
    heur = not f
    if heur:
        f = budget.fisheye_focal_px({'width': w, 'height': h, 'fov': fov})
    max_half = fov / 2.0
    if model == 'rectilinear':
        max_half = min(max_half, RECTILINEAR_MAX_DEG)
    elif model == 'orthographic':
        max_half = min(max_half, 90.0)
    k = tuple(float(v) for v in (list(L.get('k') or []) + [0.0] * 4)[:4])
    cx = float(L['cx']) if L.get('cx') is not None else w / 2.0
    cy = float(L['cy']) if L.get('cy') is not None else h / 2.0
    return Intrinsics(w, h, float(f), model, k, cx, cy, math.radians(max_half), float(L.get('aspect') or 1.0), heur)


def output_rays(intr: Intrinsics, w: int | None = None, h: int | None = None):
    """Optical-frame unit rays (h, w, 3) of the output image's pixel centres and whether each sees anything (within the
    lens's field): pixel i covering [i, i+1) of the sensor scaled to w × h (by default the sensor's own size)."""
    w = int(w or intr.width)
    h = int(h or intr.height)
    sx, sy = intr.width / w, intr.height / h
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    d, theta = intr.unproject((uu + 0.5) * sx, (vv + 0.5) * sy)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    return d, (theta <= intr.max_theta) & np.isfinite(theta)


# ---------------------------------------------------------------- views and tables (vantage.core.simrig)
@dataclass
class View:
    name: str
    F: np.ndarray            # columns: the view's optical x, y, z in the camera's optical frame
    width: int
    height: int
    f: float                 # px

    @property
    def fov_deg(self) -> float:
        """CARLA's fov attribute: the horizontal field over the full image width."""
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.f)))

    def project(self, d_view):
        """View-frame directions (K, 3) -> OpenCV pixel coordinates (centres on integers)."""
        z = d_view[..., 2]
        return (self.width / 2.0 - 0.5 + self.f * d_view[..., 0] / z,
                self.height / 2.0 - 0.5 + self.f * d_view[..., 1] / z)


def face_px(f_px: float) -> int:
    """A cube face's size for a lens of f_px px/rad: N = 2·⌈f·tan(45° + SEAM_BLEND_DEG/2) + FACE_MARGIN_PX⌉, so its
    centre density (N/2 − margin) / tan(45° + band/2) is at least f (before the VIEW_MAX_PX cap)."""
    t = math.tan(math.radians(45.0 + SEAM_BLEND_DEG / 2.0))
    return 2 * int(math.ceil(f_px * t + FACE_MARGIN_PX))


def plan_views(d, ok, f_out: float, oversample: float = 1.0) -> list:
    """The pinhole views a camera's rays need (see the module doc): one view within SINGLE_VIEW_DEG, else cube faces."""
    D = d[ok]
    if not len(D):
        return []
    f_v = max(f_out * oversample, 8.0)
    ang = np.degrees(np.arccos(np.clip(D[:, 2], -1, 1)))
    if float(ang.max()) <= SINGLE_VIEW_DEG:
        ax = float(np.max(np.abs(D[:, 0] / D[:, 2])))
        ay = float(np.max(np.abs(D[:, 1] / D[:, 2])))
        W = 2 * int(math.ceil(f_v * ax + FACE_MARGIN_PX + 1))
        H = 2 * int(math.ceil(f_v * ay + FACE_MARGIN_PX + 1))
        if max(W, H) > VIEW_MAX_PX:
            k = VIEW_MAX_PX / max(W, H)
            W, H, f_v = 2 * int(W * k / 2), 2 * int(H * k / 2), f_v * k
        return [View('front', FACES['front'], W, H, f_v)]
    t = math.tan(math.radians(45.0 + SEAM_BLEND_DEG / 2.0))  # a face reaches half the band past each of its seams
    N = min(face_px(f_v), VIEW_MAX_PX)                      # face centre density f = (N/2 − margin) / t ≥ f_v
    f = (N / 2.0 - FACE_MARGIN_PX) / t
    best = np.argmax(np.stack([D @ FACES[k][:, 2] for k in FACES], 1), 1)
    used = sorted(set(best.tolist()))
    names = list(FACES)
    return [View(names[i], FACES[names[i]], N, N, f) for i in used]


def remap_tables(d, ok, views: list) -> tuple[np.ndarray, np.ndarray, list]:
    """(map_x, map_y float32 (h, w), atlas offsets): where each output pixel reads in the views stacked vertically
    (view k's rows start at offsets[k]); −1 where the pixel sees nothing. cv2.remap(atlas, map_x, map_y,
    INTER_LINEAR, BORDER_CONSTANT) gives the image."""
    h, w = ok.shape
    mx = np.full((h, w), -1.0, np.float32)
    my = np.full((h, w), -1.0, np.float32)
    offsets, y0 = [], 0
    for v in views:
        offsets.append(y0)
        y0 += v.height
    if not views:
        return mx, my, offsets
    D = d[ok]
    if len(views) == 1:
        k = np.zeros(len(D), int)
    else:
        k = np.argmax(np.stack([D @ v.F[:, 2] for v in views], 1), 1)
    u = np.zeros(len(D))
    vv = np.zeros(len(D))
    for i, v in enumerate(views):
        sel = k == i
        dv = D[sel] @ v.F
        uu, vy = v.project(dv)
        u[sel] = uu
        vv[sel] = vy + offsets[i]
    mx[ok] = u.astype(np.float32)
    my[ok] = vv.astype(np.float32)
    return mx, my, offsets


def blend_tables(d, ok, views: list, offsets: list, band_deg: float = SEAM_BLEND_DEG):
    """Feathering across the cube faces' seams, or None for a single view: (idx (M,) int32, the flat indices of the
    output pixels within band_deg / 2 of a seam; x, y (M, 2) float32, where they read their second and third face in
    the atlas (−1: none); w (M, 3) float32, the weights of the face remap_tables reads and of those two, summing to 1).
    A face's weight is a smoothstep of the ray's angle inside it from its seams with the other rendered faces: ½ on a
    seam, 1 half the band inside, 0 half the band outside. The views of a camera differ slightly where CARLA renders
    them in motion (its temporal anti-aliasing reprojects each view's history along that view's own image motion):
    feathered, a difference of a few levels fades over the band instead of stepping at a line."""
    if len(views) < 2:
        return None
    D = d[ok]
    n = np.stack([v.F[:, 2] for v in views])
    s = np.full((len(D), len(views)), np.inf)
    for i in range(len(views)):
        for j in range(len(views)):
            if i != j and abs(float(n[i] @ n[j])) < 0.5:               # neighbours (not the same or opposite face)
                m = (n[i] - n[j]) / np.linalg.norm(n[i] - n[j])
                s[:, i] = np.minimum(s[:, i], np.degrees(np.arcsin(np.clip(D @ m, -1.0, 1.0))))
    t = np.clip((s + band_deg / 2.0) / band_deg, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    U = np.full(w.shape, -1.0)
    V = np.full(w.shape, -1.0)
    for i, v in enumerate(views):
        dv = D @ v.F
        front = dv[:, 2] > 1e-9
        uu, vy = v.project(np.where(front[:, None], dv, 1.0))
        inside = front & (uu >= FACE_MARGIN_PX - 0.5) & (uu <= v.width - 0.5 - FACE_MARGIN_PX) & \
            (vy >= FACE_MARGIN_PX - 0.5) & (vy <= v.height - 0.5 - FACE_MARGIN_PX)
        w[~inside, i] = 0.0
        U[:, i], V[:, i] = uu, vy + offsets[i]
    first = np.argmax(D @ n.T, 1)                                         # the face remap_tables reads
    tot = w.sum(1)
    many = (np.count_nonzero(w, 1) >= 2) & (tot > 0)
    rows = np.flatnonzero(many)
    W = w[rows] / tot[rows, None]
    k0 = first[rows]
    order = np.argsort(-W, 1)
    out_w = np.zeros((len(rows), 3), np.float32)
    out_x = np.full((len(rows), 2), -1.0, np.float32)
    out_y = np.full((len(rows), 2), -1.0, np.float32)
    r = np.arange(len(rows))
    out_w[:, 0] = W[r, k0]
    col = np.zeros(len(rows), int)
    for c in range(order.shape[1]):
        kc = order[:, c]
        take = (kc != k0) & (W[r, kc] > 0) & (col < 2)
        out_w[take, 1 + col[take]] = W[take, kc[take]]
        out_x[take, col[take]] = U[rows[take], kc[take]]
        out_y[take, col[take]] = V[rows[take], kc[take]]
        col += take
    out_w /= out_w.sum(1, keepdims=True)                     # (a weight beyond two other faces: none on a cube)
    idx = np.flatnonzero(ok.ravel())[rows].astype(np.int32)
    return idx, out_x, out_y, out_w


def views_mpx(views) -> float:
    """Megapixels a camera frame renders (one camera kind): its views' pixels."""
    return sum(int(_get(v, 'width')) * int(_get(v, 'height')) for v in views) / 1e6


def _get(v, k):
    return v[k] if isinstance(v, dict) else getattr(v, k)


# ---------------------------------------------------------------- CARLA rotations of the views
def view_rotation(F) -> dict:
    """{'roll', 'pitch', 'yaw'} of a view's CARLA camera relative to the sensor: the view's optical frame F (columns in
    the sensor's optical frame) taken into CARLA's actor axes, C2Oᵀ·F·C2O."""
    M = CARLA_TO_OPTICAL.T @ np.asarray(F, float) @ CARLA_TO_OPTICAL
    return _rot(*carla_euler(M))


def view_pose(sensor_pose: dict, F) -> dict:
    """The view's pose in the vehicle frame: the sensor's location, the sensor's rotation composed with the view's."""
    p = sensor_pose or {}
    Ms = carla_matrix(float(p.get('pitch', 0.0)), float(p.get('yaw', 0.0)), float(p.get('roll', 0.0)))
    M = Ms @ CARLA_TO_OPTICAL.T @ np.asarray(F, float) @ CARLA_TO_OPTICAL
    return dict({k: float(p.get(k, 0.0)) for k in ('x', 'y', 'z')}, **_rot(*carla_euler(M)))


def _rot(pitch, yaw, roll) -> dict:
    return {'roll': round(float(roll), 9) + 0.0, 'pitch': round(float(pitch), 9) + 0.0,
            'yaw': round(float(yaw), 9) + 0.0}


def optical_to_vehicle(pose: dict) -> np.ndarray:
    """3 × 3: a sensor's optical-frame directions into the vehicle frame (CARLA's numbers)."""
    return carla_matrix(float(pose.get('pitch', 0.0)), float(pose.get('yaw', 0.0)),
                        float(pose.get('roll', 0.0))) @ CARLA_TO_OPTICAL.T


# ---------------------------------------------------------------- the plan
def plan_camera(sensor: dict, oversample: float = 1.0, tables: bool = True) -> dict:
    """What CARLA renders for a camera sensor and how its image is made from it:

    {'sensor_id', 'type', 'image': {'width', 'height'}, 'lens' (Intrinsics.to_dict), 'K',
     'views': [{'name', 'width', 'height', 'fov' (CARLA's attribute), 'f_px', 'optical' (3 × 3, the view's optical
                frame in the sensor's), 'rotation' ({roll, pitch, yaw} relative to the sensor), 'pose' (in the vehicle
                frame), 'attributes' (the blueprint attributes to set)}],
     'atlas': {'width', 'height', 'offsets'},
     'remap': {'map_x', 'map_y' (float32 (h, w) numpy), 'blend': None or {'idx', 'x', 'y', 'w'} (numpy)} (None with
              tables=False),
     'render_mpx' (the views, one camera kind), 'seam_blend_deg', 'notes', 'summary' (JSON-able: all but the arrays)}

    A pinhole sensor is exactly one view, the camera itself (its size and fov, no rotation), with the identity remap."""
    t = sensor.get('type')
    intr = intrinsics(sensor)
    w, h = intr.width, intr.height
    pose = dict({'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}, **(sensor.get('pose') or {}))
    notes = []
    if t == 'camera.pinhole':
        views = [View('front', FACES['front'], w, h, intr.f_px)]
        fovs = [float(sensor['fov'])]
        offsets = [0]
        remap = None
        if tables:
            uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            remap = {'map_x': uu, 'map_y': vv, 'blend': None}
        coverage = 1.0
    else:
        d, ok = output_rays(intr)
        views = plan_views(d, ok, intr.f_px, oversample)
        fovs = [v.fov_deg for v in views]
        offsets = [int(o) for o in np.cumsum([0] + [v.height for v in views[:-1]])] if views else []
        remap = None
        if tables:
            mx, my, offsets = remap_tables(d, ok, views)
            b = blend_tables(d, ok, views, offsets)
            remap = {'map_x': mx, 'map_y': my,
                     'blend': None if b is None else dict(zip(('idx', 'x', 'y', 'w'), b))}
        coverage = float(ok.mean())
        if intr.f_heuristic:
            notes.append(f'{sensor.get("name", sensor.get("id", "camera"))}: no focal length given; heuristic: an '
                         f'equidistant lens whose {math.degrees(2 * intr.max_theta):g}° image circle fits the shorter '
                         f'image side (f = {intr.f_px:.1f} px/rad).')
        if any(max(v.width, v.height) >= VIEW_MAX_PX for v in views):
            notes.append(f'views capped at {VIEW_MAX_PX} px: the lens\'s centre is sampled below its own density.')
    vdocs = []
    for v, fov in zip(views, fovs):
        attrs = {'image_size_x': int(v.width), 'image_size_y': int(v.height), 'fov': float(fov)}
        if t != 'camera.pinhole':
            attrs.update(VIEW_ATTRIBUTES)
        vdocs.append({'name': v.name, 'width': int(v.width), 'height': int(v.height), 'fov': float(fov),
                      'f_px': float(v.f), 'optical': [[float(x) for x in row] for row in v.F],
                      'rotation': view_rotation(v.F), 'pose': view_pose(pose, v.F), 'attributes': attrs})
    atlas = {'width': max((v.width for v in views), default=0), 'height': int(sum(v.height for v in views)),
             'offsets': [int(o) for o in offsets]}
    plan = {'sensor_id': sensor.get('id'), 'type': t, 'image': {'width': w, 'height': h}, 'lens': intr.to_dict(),
            'K': intr.K, 'pose': pose, 'views': vdocs, 'atlas': atlas, 'remap': remap,
            'render_mpx': round(views_mpx(views), 6), 'seam_blend_deg': SEAM_BLEND_DEG if len(views) > 1 else 0.0,
            'notes': notes}
    summary = {k: plan[k] for k in ('sensor_id', 'type', 'image', 'lens', 'K', 'pose', 'views', 'atlas',
                                    'render_mpx', 'seam_blend_deg', 'notes')}
    summary['coverage'] = round(coverage, 6)
    if remap is not None:
        summary['blend_pixels'] = int(len(remap['blend']['idx'])) if remap['blend'] is not None else 0
    plan['summary'] = summary
    return plan


def camera_mpx(sensor: dict, plan: dict | None = None) -> float:
    """Megapixels the camera renders per step: its views' pixels once for the image and once per label camera
    (core.budget.label_cameras)."""
    p = plan or plan_camera(sensor, tables=False)
    return p['render_mpx'] * (1 + len(budget.label_cameras(sensor.get('labels'))))


# ---------------------------------------------------------------- making the image (Vantage's CARLA bridge)
def atlas_of(plan: dict, images) -> np.ndarray:
    """The views' images (in plan['views'] order, each (h, w) or (h, w, C)) stacked into the atlas the remap reads."""
    at = plan['atlas']
    first = np.asarray(images[0])
    out = np.zeros((int(at['height']), int(at['width'])) + first.shape[2:], first.dtype)
    for k, img in enumerate(images):
        a = np.asarray(img)
        y0 = int(at['offsets'][k])
        out[y0:y0 + a.shape[0], :a.shape[1]] = a
    return out


def blend_tuple(plan: dict):
    b = (plan.get('remap') or {}).get('blend')
    return None if b is None else tuple(b[k] for k in ('idx', 'x', 'y', 'w'))


def _sample(img, x, y, cols=1024):
    """img (h, w, C) read bilinearly at the points (x, y) (M,): (M, C). The points go through cv2.remap as a block
    of rows (it takes images under 32767 pixels a side); outside the image reads 0."""
    import cv2
    m = len(x)
    rows = max(-(-m // cols), 1)
    px = np.full(rows * cols, -1.0, np.float32)
    py = np.full(rows * cols, -1.0, np.float32)
    px[:m], py[:m] = x, y
    out = cv2.remap(img, px.reshape(rows, cols), py.reshape(rows, cols), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(-1, img.shape[2])[:m]


def feather(out, atlas, blend, devignette=None):
    """The output image (h, w, C), read from each pixel's first face, with the cube faces feathered across their seams
    (blend: idx, x, y, w of blend_tables; None leaves it as it is). devignette: a tonemap.Devignette per other face's
    samples, applied before mixing (each sample's vignette is its own view's)."""
    if blend is None or not len(blend[0]):
        return out
    idx, bx, by, bw = blend
    flat = out.reshape(-1, out.shape[2])
    acc = flat[idx].astype(np.float32) * bw[:, :1]
    for j in range(bx.shape[1]):
        smp = _sample(atlas, bx[:, j], by[:, j])
        if devignette is not None:
            smp = devignette[j].apply(smp[None])[0]
        acc += smp.astype(np.float32) * bw[:, 1 + j:2 + j]
    flat[idx] = np.clip(np.rint(acc), 0, 255).astype(out.dtype) if out.dtype == np.uint8 else acc
    return out


def devignette_layers(plan: dict) -> list:
    """[tonemap.Devignette] taking Unreal's vignette out of a camera's image (see the module doc): the first for the
    face each output pixel reads (the vignette mask at its view pixel), then one per blend sample column."""
    from ..sim import tonemap as TM
    sizes = [(v['width'], v['height']) for v in plan['views']]
    offs = plan['atlas']['offsets']
    rm = plan['remap']
    layers = [TM.Devignette(TM.mask_of_remap(rm['map_x'], rm['map_y'], offs, sizes))]
    bl = blend_tuple(plan)
    if bl is not None:
        m = TM.mask_of_remap(bl[1], bl[2], offs, sizes)
        layers += [TM.Devignette(m[:, j][None, :]) for j in range(m.shape[1])]
    return layers


def remap_image(plan: dict, images, devignette=None) -> np.ndarray:
    """The camera's image (h, w, 3) uint8 from its views' images (each (h, w, 3) or CARLA's BGRA (h, w, 4), in
    plan['views'] order; channel order kept): stacked into the atlas, remapped, the vignette taken out when devignette
    (devignette_layers) is given, the faces feathered across their seams."""
    import cv2
    atlas = atlas_of(plan, [np.asarray(a)[..., :3] for a in images])
    rm = plan['remap']
    out = cv2.remap(atlas, rm['map_x'], rm['map_y'], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if devignette is not None:
        out = devignette[0].apply(out)
    return feather(out, atlas, blend_tuple(plan), devignette[1:] if devignette is not None else None)
