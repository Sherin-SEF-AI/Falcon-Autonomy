"""Moving actors in a reconstruction (4D Gaussians; torch at import): every tracked road user is a set of Gaussians in
its own box frame, posed at any instant by its logged box composed with a learned, regularised rigid correction, and
rasterised together with the static scene in ONE pass of core.recon.raster.render, so occlusion between actors and
the scene (and between actors) is the rasteriser's depth order, as for any Gaussians.

Ported from Vantage core/splat_reproject.py (Track, _build_tracks, _posed_means: an annotated object rebuilt in its box
frame and placed at its interpolated box pose whenever a view is rendered), extended for training:

* **Tracks** (Track): box poses world ← box at the annotation times (the reconstruction's right-handed world; box x
  along the length, y to the left, z up, origin at the box centre), the size (length, width, height), the class.
  Positions are interpolated linearly and rotations by normalised linear interpolation of the quaternions (nlerp);
  an object is shown within ACTOR_PAD_S of an annotation and across gaps of at most ACTOR_GAP_S (Vantage's rule).
  An object that does not move (≤ 0.3 m over the drive, not a person: _build_tracks's by_motion) is posed where it
  was annotated nearest in time.
* **The pose** of actor a at time t: T_box(t) ∘ exp(ξ(t)), ξ = (δ, ω) a translation (box frame, m) and a rotation
  vector (rad) learned per annotation and interpolated linearly in time — the correction of a box that is a little
  off (a detector's, or an annotator's). Regularised towards zero (λ‖δ‖² + λ‖ω‖²) and towards its neighbours in time
  (the second difference), so it cannot absorb the object's motion or wander where the images say nothing.
* **Wheels** (vehicles): each wheel is a template cylinder of Gaussians (WHEEL_AROUND × WHEEL_ACROSS discs on its
  tyre surface, in the wheel's frame) at the class's wheel positions — four for a car, three for an autorickshaw
  (one in front), two for a two-wheeler — turning about its axle by φ(t) = φ₀ + s(t) / r, s(t) = ∫ v dt the distance
  the box travelled along its own heading (signed), r the wheel radius (a class default, WHEELS), φ₀ a learned phase
  per wheel; the front wheels steer by the bicycle model's angle from the box's yaw rate, δ = atan(L ψ̇ / v) (L the
  wheelbase), split between the left and right wheels by Ackermann geometry (δ_l, δ_r = atan(L / (L / tan δ ∓
  track / 2))); held at the last value below MIN_STEER_SPEED. Their colours and opacities are learned like the body's.
* **Pedestrians**: canonical Gaussians in the box frame plus a small deformation network Δμ = f(μ, t) (Fourier
  features of the canonical position and time → two hidden layers of 64 → a displacement, its last layer
  zero-initialised so it starts as a rigid body), regularised by the displacement's size, its change over 0.1 s
  (temporal) and the difference of neighbouring Gaussians' displacements (local rigidity, 4 nearest neighbours in the
  canonical pose). No body model (SMPL is refused: non-commercial); the network replays the motion the images show at
  the times they show it — it cannot make a new motion, and a limb hidden in every frame stays wherever it started.
* Every actor Gaussian has degree-0 colour (no view dependence), a lidar intensity and a ray-drop logit
  (core.recon.lidar); there is no density control for actor Gaussians (their number is fixed at initialisation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

ACTOR_PAD_S, ACTOR_GAP_S = 0.11, 0.6          # Vantage's: shown within this of an annotation, across gaps up to this
MIN_STEER_SPEED = 0.5                         # m/s: below it the steering angle is held
MAX_STEER_DEG = 40.0
WHEEL_AROUND, WHEEL_ACROSS = 16, 3
KINDS = ('car', 'autorickshaw', 'two_wheeler', 'pedestrian', 'rigid')
# class -> (wheel radius m, wheel width m, layout); layout: wheel centres as (x / length, y / width) fractions and
# whether the wheel steers. A typical vehicle of the class, not a measured one.
WHEELS = {
    'car': (0.33, 0.22, ((0.32, 0.43, True), (0.32, -0.43, True), (-0.32, 0.43, False), (-0.32, -0.43, False))),
    'autorickshaw': (0.20, 0.12, ((0.38, 0.0, True), (-0.30, 0.42, False), (-0.30, -0.42, False))),
    'two_wheeler': (0.30, 0.10, ((0.36, 0.0, True), (-0.36, 0.0, False))),
}
SH_C0 = 0.28209479177387814


def kind_of(cls: str, category: str = '') -> str:
    """The actor model of a class name or a category (nuScenes or core.groundtruth's): car (four wheels; trucks, buses
    and vans too), autorickshaw, two_wheeler (bicycles, motorcycles), pedestrian, or rigid (anything else)."""
    c = f'{cls} {category}'.lower()
    if 'pedestrian' in c or c.startswith('human') or ' human' in c or 'walker' in c or 'child' in c:
        return 'pedestrian'
    if 'rickshaw' in c or 'tuktuk' in c or 'tuk-tuk' in c:
        return 'autorickshaw'
    if any(k in c for k in ('bicycle', 'motorcycle', 'two_wheeler', 'scooter', 'cycle')):
        return 'two_wheeler'
    if any(k in c for k in ('car', 'truck', 'bus', 'van', 'vehicle', 'trailer')):
        return 'car'
    return 'rigid'


# =========================================================================== quaternions (numpy and torch)
def quat_to_rot_np(q) -> np.ndarray:
    q = np.asarray(q, float).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def rot_to_quat_np(R) -> np.ndarray:
    """(N, 3, 3) -> (N, 4) unit quaternions w ≥ 0 (Shepperd's method; Vantage core/splat_raster.py)."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    q = np.zeros((len(R), 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cases = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], 1).argmax(1)
    for c in range(4):
        m = cases == c
        if not m.any():
            continue
        r = R[m]
        if c == 0:
            s = np.sqrt(1.0 + tr[m]) * 2
            q[m] = np.stack([0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif c == 1:
            s = np.sqrt(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                             (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif c == 2:
            s = np.sqrt(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2
            q[m] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            s = np.sqrt(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2
            q[m] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def nlerp(q0, q1, f):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    q = (1 - f)[..., None] * q0 + f[..., None] * q1
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def quat_mul(a, b):
    """Hamilton product of (…, 4) torch quaternions w, x, y, z."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_rotvec(w):
    """(…, 3) rotation vectors -> (…, 4) unit quaternions (differentiable at 0)."""
    th2 = (w * w).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-24)
    half = 0.5 * th
    s = torch.where(th2 > 1e-12, torch.sin(half) / th, 0.5 - th2 / 48.0)
    return torch.cat([torch.cos(half), w * s], -1)


def quat_axis(axis: int, angle):
    """(…,) angles about the x (0), y (1) or z (2) axis -> (…, 4)."""
    h = 0.5 * angle
    z = torch.zeros_like(h)
    parts = [torch.cos(h), z, z, z]
    parts[1 + axis] = torch.sin(h)
    return torch.stack(parts, -1)


def rot_of_quat(q):
    from .raster import quat_to_rotmat
    return quat_to_rotmat(q.reshape(-1, 4)).reshape(q.shape[:-1] + (3, 3))


# =========================================================================== tracks (Vantage core/splat_reproject.py)
@dataclass
class Track:
    """An annotated object over the drive: box poses world ← box at the annotation times."""
    id: int
    cls: str
    category: str
    dynamic: bool
    size: tuple                         # (length, width, height) m
    t: np.ndarray
    centre: np.ndarray                  # (k, 3)
    quat: np.ndarray                    # (k, 4) w, x, y, z
    kind: str = ''
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.kind:
            self.kind = kind_of(self.cls, self.category)

    def index(self, t: float):
        """(j, f): t lies between annotations j and j + 1 at fraction f (clipped to the span)."""
        ts = self.t
        if len(ts) == 1:
            return 0, 0.0
        j = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
        f = float(np.clip((t - ts[j]) / max(ts[j + 1] - ts[j], 1e-9), 0.0, 1.0))
        return j, f

    def shown(self, t: float) -> bool:
        return self.pose(t) is not None

    def pose(self, t):
        """(R world ← box, centre) at t, or None when the object is not shown then (module doc)."""
        ts = self.t
        if not len(ts):
            return None
        if not self.dynamic:
            k = int(np.argmin(np.abs(ts - t)))
            return quat_to_rot_np(self.quat[k:k + 1])[0], self.centre[k].copy()
        j = int(np.searchsorted(ts, t))
        if j < len(ts) and abs(ts[j] - t) < 1e-9:
            return quat_to_rot_np(self.quat[j:j + 1])[0], self.centre[j].copy()
        if 0 < j < len(ts) and ts[j] - ts[j - 1] <= ACTOR_GAP_S:
            f = np.array([(t - ts[j - 1]) / (ts[j] - ts[j - 1])])
            q = nlerp(self.quat[j - 1:j], self.quat[j:j + 1], f)
            return quat_to_rot_np(q)[0], self.centre[j - 1] + f[0] * (self.centre[j] - self.centre[j - 1])
        k = j - 1 if j > 0 else 0
        if j >= len(ts):
            k = len(ts) - 1
        elif j > 0 and abs(ts[j] - t) < abs(ts[j - 1] - t):
            k = j
        if abs(ts[k] - t) <= ACTOR_PAD_S:
            return quat_to_rot_np(self.quat[k:k + 1])[0], self.centre[k].copy()
        return None

    def velocity(self, t: float) -> np.ndarray:
        """World velocity (m/s) at t: the slope of the annotated centres around it (0 for one annotation or a static
        object)."""
        if len(self.t) < 2 or not self.dynamic:
            return np.zeros(3)
        j, _ = self.index(t)
        dt = self.t[j + 1] - self.t[j]
        return (self.centre[j + 1] - self.centre[j]) / max(dt, 1e-9)

    def yaw(self) -> np.ndarray:
        """Heading (rad, counter-clockwise from world x) of the box's x axis at each annotation, unwrapped."""
        R = quat_to_rot_np(self.quat)
        return np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))

    def doc(self) -> dict:
        return {'id': int(self.id), 'cls': self.cls, 'category': self.category, 'kind': self.kind,
                'dynamic': bool(self.dynamic), 'size': [float(x) for x in self.size],
                't': [float(x) for x in self.t], 'centre': np.asarray(self.centre, float).tolist(),
                'quat': np.asarray(self.quat, float).tolist(), 'meta': self.meta}

    @classmethod
    def from_doc(cls, d: dict) -> 'Track':
        return cls(int(d['id']), d.get('cls', ''), d.get('category', ''), bool(d.get('dynamic', True)),
                   tuple(float(x) for x in d['size']), np.asarray(d['t'], float), np.asarray(d['centre'], float),
                   np.asarray(d['quat'], float), d.get('kind', ''), dict(d.get('meta') or {}))


def build_tracks(obs: dict, info: dict, by_motion: bool = False) -> dict:
    """{id: Track} from {id: [(t, centre (3,), quat (4,)), …]} and {id: (cls, category, dynamic, size (l, w, h))}
    (Vantage's _build_tracks). by_motion: an object moves when it is a person or its box moved more than 0.3 m."""
    out = {}
    for i, rows in obs.items():
        rows = sorted(rows, key=lambda r: r[0])
        t = np.array([r[0] for r in rows], float)
        keep = np.r_[True, np.diff(t) > 1e-6]
        cls, cat, dyn, size = info[i]
        c = np.array([r[1] for r in rows], float)[keep]
        if by_motion:
            moved = float(np.max(np.linalg.norm(c - c[0], axis=1))) if len(c) else 0.0
            dyn = moved > 0.3 or kind_of(cls, cat) == 'pedestrian'
        q = np.array([r[2] for r in rows], float)[keep]
        for k in range(1, len(q)):                      # one hemisphere along the track (nlerp takes the short way)
            if np.dot(q[k], q[k - 1]) < 0:
                q[k] = -q[k]
        out[i] = Track(int(i), cls, cat, bool(dyn), tuple(float(x) for x in size), t[keep], c, q)
    return out


def posed_means(local: np.ndarray, owner: np.ndarray, tracks: list, t: float):
    """World centres (N, 3) at time t of Gaussians in their actors' box frames (owner: index into tracks, −1 the
    static scene, unchanged) and whether each is shown (Vantage's _posed_means)."""
    P = np.asarray(local, float).copy()
    shown = np.ones(len(P), bool)
    for a, tr in enumerate(tracks):
        m = owner == a
        po = tr.pose(t)
        if po is None:
            shown[m] = False
            continue
        Rb, cb = po
        P[m] = P[m] @ Rb.T + cb
    return P, shown


# =========================================================================== wheels
def wheel_layout(track: Track) -> list:
    """The wheels of a vehicle track: [{'centre' (3,) box frame, 'radius', 'width', 'steers', 'side' (+1 left, −1
    right, 0 centre)}] by its class (WHEELS); [] for pedestrians and rigid objects."""
    spec = WHEELS.get(track.kind)
    if spec is None:
        return []
    r, wd, layout = spec
    L, W, H = track.size
    r = min(r, 0.45 * H) if H > 0 else r
    out = []
    for fx, fy, steers in layout:
        out.append({'centre': np.array([fx * L, fy * W, -H / 2 + r]), 'radius': float(r), 'width': float(wd),
                    'steers': bool(steers), 'side': int(np.sign(fy))})
    return out


def wheel_template(radius: float, width: float, around: int = WHEEL_AROUND, across: int = WHEEL_ACROSS):
    """Gaussians on a tyre's surface in the wheel's frame (axle along y): (means (n, 3), scales (n, 3), quats (n, 4)).
    Each is a disc tangent to the tread: thin radially, an arc's length around, a share of the width across."""
    a = 2 * np.pi * (np.arange(around) + 0.5) / around
    ys = (np.arange(across) + 0.5) / across * width - width / 2
    A, Y = np.meshgrid(a, ys, indexing='ij')
    A, Y = A.ravel(), Y.ravel()
    means = np.stack([radius * np.sin(A), Y, radius * np.cos(A)], 1)
    arc = 2 * np.pi * radius / around
    scales = np.tile([0.5 * arc, 0.5 * width / across, 0.15 * arc], (len(A), 1))
    # the disc's thin axis (local z) along the radius: rotate z towards (sin a, 0, cos a) about y by a
    quats = np.stack([np.cos(A / 2), np.zeros_like(A), np.sin(A / 2), np.zeros_like(A)], 1)
    return means, scales, quats


def wheel_kinematics(track: Track, wheelbase: float, track_width: float):
    """At the track's annotation times: (s (k,) the signed distance travelled along the box's heading, steer (k,)
    the bicycle model's front angle from the yaw rate, rad; module doc)."""
    t = track.t
    if len(t) < 2:
        return np.zeros(len(t)), np.zeros(len(t))
    yaw = track.yaw()
    c = track.centre
    ds = np.zeros(len(t))
    steer = np.zeros(len(t))
    for k in range(len(t) - 1):
        mid = 0.5 * (yaw[k] + yaw[k + 1])
        d = c[k + 1] - c[k]
        ds[k + 1] = d[0] * math.cos(mid) + d[1] * math.sin(mid)
        dt = max(t[k + 1] - t[k], 1e-6)
        v = ds[k + 1] / dt
        if abs(v) >= MIN_STEER_SPEED:
            rate = (yaw[k + 1] - yaw[k]) / dt
            steer[k] = math.atan(wheelbase * rate / v)
        else:
            steer[k] = steer[k - 1] if k else 0.0
    steer[-1] = steer[-2]
    lim = math.radians(MAX_STEER_DEG)
    return np.cumsum(ds), np.clip(steer, -lim, lim)


def ackermann(steer: float, wheelbase: float, track_width: float, side: int) -> float:
    """The angle of the left (side +1) or right (−1) front wheel for the bicycle model's steer (rad, + left)."""
    if side == 0 or abs(steer) < 1e-6:
        return steer
    Rt = wheelbase / math.tan(steer)                     # signed turning radius (+ left)
    return math.atan(wheelbase / (Rt - side * track_width / 2.0))


# =========================================================================== the deformation network
class Deform(torch.nn.Module):
    """Δμ = f(μ, t): Fourier features (N_FREQ octaves) of the canonical position (box frame, metres / SCALE) and the
    normalised time, two hidden layers of 64, a zero-initialised output (module doc)."""
    N_FREQ = 4
    SCALE = 1.0

    def __init__(self, hidden: int = 64):
        super().__init__()
        n_in = 4 * (1 + 2 * self.N_FREQ)
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.SiLU(), torch.nn.Linear(hidden, 3))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def encode(self, x):
        out = [x]
        for k in range(self.N_FREQ):
            out += [torch.sin((2.0 ** k) * math.pi * x), torch.cos((2.0 ** k) * math.pi * x)]
        return torch.cat(out, -1)

    def forward(self, mu, tn):
        x = torch.cat([mu / self.SCALE, torch.full_like(mu[:, :1], float(tn))], -1)
        return self.net(self.encode(x))


# =========================================================================== one actor
class ActorModel:
    """One track's Gaussians and its learned pose correction, wheels or deformation (module doc). local: the body's
    Gaussians in the box frame: means (n, 3), scales (n, 3) linear, quats (n, 4), opacity (n,), rgb (n, 3)."""

    def __init__(self, track: Track, means, scales, quats, opacity, rgb, device='cpu', wheels: bool = True,
                 deform: bool | None = None):
        self.track = track
        self.device = torch.device(device)
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)  # noqa: E731
        L, W, H = track.size
        self.wheelbase = 0.64 * L if track.kind == 'car' else (0.68 * L if track.kind == 'autorickshaw' else 0.72 * L)
        self.track_width = 0.86 * W
        self.wheels = wheel_layout(track) if wheels else []
        parts = [np.full(len(means), -1)]
        m_all, s_all, q_all = [np.asarray(means, float)], [np.asarray(scales, float)], [np.asarray(quats, float)]
        o_all, c_all = [np.asarray(opacity, float)], [np.asarray(rgb, float)]
        for k, wh in enumerate(self.wheels):
            wm, ws, wq = wheel_template(wh['radius'], wh['width'])
            m_all.append(wm)
            s_all.append(ws)
            q_all.append(wq)
            o_all.append(np.full(len(wm), 0.9))
            c_all.append(np.tile([0.12, 0.12, 0.13], (len(wm), 1)))
            parts.append(np.full(len(wm), k))
        self.part = torch.as_tensor(np.concatenate(parts), device=self.device)
        rgb = np.clip(np.concatenate(c_all), 0.01, 0.99)
        op = np.clip(np.concatenate(o_all), 0.01, 0.99)
        n = len(op)
        self.p = {'means': T(np.concatenate(m_all)).requires_grad_(),
                  'log_scales': T(np.log(np.maximum(np.concatenate(s_all), 1e-4))).requires_grad_(),
                  'quats': T(np.concatenate(q_all)).requires_grad_(),
                  'opacity_logits': T(np.log(op / (1 - op))).requires_grad_(),
                  'sh_dc': T((rgb - 0.5) / SH_C0).requires_grad_(),
                  'lidar_int': torch.zeros(n, device=self.device, requires_grad=True),
                  'lidar_drop': torch.full((n,), -3.0, device=self.device, requires_grad=True)}
        self.xi = torch.zeros(max(len(track.t), 1), 6, device=self.device, requires_grad=True)
        self.phase = torch.zeros(len(self.wheels), device=self.device, requires_grad=True)
        s_, st_ = wheel_kinematics(track, self.wheelbase, self.track_width)
        self.s_keys, self.steer_keys = s_, st_
        self.deform = None
        if deform if deform is not None else track.kind == 'pedestrian':
            self.deform = Deform().to(self.device)
            self._knn = self._neighbours()

    # ------------------------------------------------------------------ parameters
    def params(self) -> dict:
        """{group: [tensors]}: 'gaussians' (positions …), 'pose' (the correction and wheel phases), 'deform'."""
        out = {'gaussians': list(self.p.values()), 'pose': [self.xi, self.phase]}
        if self.deform is not None:
            out['deform'] = list(self.deform.parameters())
        return out

    def __len__(self):
        return int(self.p['means'].shape[0])

    def _neighbours(self, k: int = 4):
        from scipy.spatial import cKDTree
        body = (self.part < 0).cpu().numpy()
        mu = self.p['means'].detach().cpu().numpy()
        idx = np.flatnonzero(body)
        if len(idx) < 2:
            return None
        _, nb = cKDTree(mu[idx]).query(mu[idx], k=min(k + 1, len(idx)))
        return torch.as_tensor(idx[nb], device=self.device)

    # ------------------------------------------------------------------ time
    def _interp(self, keys, t: float):
        j, f = self.track.index(t)
        if len(self.track.t) == 1:
            return keys[0]
        return (1 - f) * keys[j] + f * keys[j + 1]

    def wheel_state(self, t: float) -> list:
        """[(spin angle φ rad, steer angle rad)] per wheel at t (module doc), without the learned phase."""
        if not self.wheels:
            return []
        s = float(self._interp(self.s_keys, t))
        st = float(self._interp(self.steer_keys, t))
        out = []
        for wh in self.wheels:
            ang = ackermann(st, self.wheelbase, self.track_width, wh['side']) if wh['steers'] else 0.0
            out.append((s / wh['radius'], ang))
        return out

    def pose_at(self, t: float, box=None):
        """(R world ← box (3, 3), centre (3,), q world ← box (4,)) as torch tensors with the learned correction, or
        None when the track is not shown at t. box: (R, centre) replacing the logged box at t (a controller's, in
        closed loop)."""
        po = self.track.pose(t) if box is None else box
        if po is None:
            return None
        Rb, cb = po
        xi = self._interp(self.xi, t)
        Rb_t = torch.as_tensor(Rb, dtype=torch.float32, device=self.device)
        qb = torch.as_tensor(rot_to_quat_np(Rb)[0], dtype=torch.float32, device=self.device)
        qc = quat_rotvec(xi[3:])
        q = quat_mul(qb, qc)
        R = Rb_t @ rot_of_quat(qc)
        c = torch.as_tensor(cb, dtype=torch.float32, device=self.device) + Rb_t @ xi[:3]
        return R, c, q

    def world(self, t: float, deform_t: float | None = None, box=None, velocity=None):
        """The actor's Gaussians at time t in the world: dict of means, scales, quats, opacity, rgb, lidar_int,
        lidar_drop, velocity (n, 3); None when the track is not shown. box, velocity: the box pose and velocity to
        use instead of the logged ones (closed loop); the wheels and the deformation keep the logged time's."""
        po = self.pose_at(t, box)
        if po is None:
            return None
        R, c, q = po
        mu = self.p['means']
        quats = self.p['quats']
        if self.wheels:
            mu_parts, q_parts = [], []
            body = self.part < 0
            mu_l = torch.where(body[:, None], mu, torch.zeros_like(mu))
            q_l = quats
            for k, ((spin, steer), wh) in enumerate(zip(self.wheel_state(t), self.wheels)):
                m = self.part == k
                ang = self.phase[k] + spin
                qw = quat_mul(quat_axis(2, torch.as_tensor(steer, device=self.device)), quat_axis(1, ang))
                Rw = rot_of_quat(qw)
                cen = torch.as_tensor(wh['centre'], dtype=torch.float32, device=self.device)
                mu_parts.append((m, mu[m] @ Rw.T + cen))
                q_parts.append((m, quat_mul(qw.expand(int(m.sum()), 4), quats[m])))
            for m, v in mu_parts:
                mu_l = mu_l.index_put((torch.nonzero(m).squeeze(1),), v)
            for m, v in q_parts:
                q_l = q_l.index_put((torch.nonzero(m).squeeze(1),), v)
            mu, quats = mu_l, q_l
        if self.deform is not None:
            tn = self._tnorm(t if deform_t is None else deform_t)
            mu = mu + self.deform(mu, tn)
        P = mu @ R.T + c
        qw = quat_mul(q.expand(len(quats), 4), quats)
        v = torch.as_tensor(self.track.velocity(t) if velocity is None else np.asarray(velocity, float),
                            dtype=torch.float32, device=self.device).expand(len(P), 3)
        return {'means': P, 'quats': qw, 'scales': torch.exp(self.p['log_scales']),
                'opacity': torch.sigmoid(self.p['opacity_logits']), 'rgb': (self.p['sh_dc'] * SH_C0 + 0.5).clamp(0, 1),
                'lidar_int': self.p['lidar_int'], 'lidar_drop': self.p['lidar_drop'], 'velocity': v}

    def _tnorm(self, t: float) -> float:
        ts = self.track.t
        if len(ts) < 2:
            return 0.0
        return float(np.clip((t - ts[0]) / max(ts[-1] - ts[0], 1e-6), 0.0, 1.0) * 2.0 - 1.0)

    # ------------------------------------------------------------------ regularisers
    def regulariser(self, w_trans=1.0, w_rot=10.0, w_smooth=10.0, w_def=1.0, w_temporal=1.0, w_rigid=1.0,
                    t: float | None = None) -> torch.Tensor:
        """The correction's size and roughness, and (pedestrians, at time t) the deformation's (module doc)."""
        xi = self.xi
        loss = w_trans * (xi[:, :3] ** 2).sum(1).mean() + w_rot * (xi[:, 3:] ** 2).sum(1).mean()
        if len(xi) > 2:
            loss = loss + w_smooth * ((xi[2:] - 2 * xi[1:-1] + xi[:-2]) ** 2).sum(1).mean()
        if self.deform is not None and t is not None:
            mu = self.p['means'].detach()
            d0 = self.deform(mu, self._tnorm(t))
            ts = self.track.t
            dt = 0.1 / max(ts[-1] - ts[0], 1e-6) * 2.0 if len(ts) > 1 else 0.0
            d1 = self.deform(mu, min(self._tnorm(t) + dt, 1.0))
            loss = loss + w_def * (d0 ** 2).sum(1).mean() + w_temporal * ((d1 - d0) ** 2).sum(1).mean()
            if self._knn is not None:
                nb = self._knn
                loss = loss + w_rigid * ((d0[nb[:, :1]] - d0[nb[:, 1:]]) ** 2).sum(-1).mean()
        return loss

    # ------------------------------------------------------------------ saving
    def state(self) -> dict:
        d = {'track': self.track.doc(), 'part': self.part.cpu(), 'xi': self.xi.detach().cpu(),
             'phase': self.phase.detach().cpu(), 'wheels': [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                              for k, v in w.items()} for w in self.wheels],
             'p': {k: v.detach().cpu() for k, v in self.p.items()}}
        if self.deform is not None:
            d['deform'] = {k: v.detach().cpu() for k, v in self.deform.state_dict().items()}
        return d

    @classmethod
    def from_state(cls, d: dict, device='cpu') -> 'ActorModel':
        tr = Track.from_doc(d['track'])
        p = d['p']
        nb = int((d['part'] < 0).sum())

        def host(k):                                   # the state may be loaded onto the GPU (torch.load map_location)
            return p[k][:nb].detach().cpu().numpy()
        a = cls(tr, host('means'), np.exp(host('log_scales')), host('quats'),
                1 / (1 + np.exp(-host('opacity_logits'))), host('sh_dc') * SH_C0 + 0.5, device,
                wheels=bool(d['wheels']), deform='deform' in d)
        with torch.no_grad():
            for k, v in p.items():
                a.p[k] = v.to(a.device).clone().requires_grad_()
            a.part = d['part'].to(a.device)
            a.xi = d['xi'].to(a.device).clone().requires_grad_()
            a.phase = d['phase'].to(a.device).clone().requires_grad_()
            if 'deform' in d:
                a.deform.load_state_dict(d['deform'])
        return a


def box_surface(size, n: int, seed: int = 0) -> np.ndarray:
    """n points on a box's surface (box frame, centred), the faces sampled by area."""
    L, W, H = size
    rng = np.random.default_rng(seed)
    areas = np.array([W * H, W * H, L * H, L * H, L * W, L * W])
    face = rng.choice(6, n, p=areas / areas.sum())
    u, v = rng.random(n) - 0.5, rng.random(n) - 0.5
    P = np.zeros((n, 3))
    for f, (ax, sgn) in enumerate(((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))):
        m = face == f
        dims = [L, W, H]
        others = [a for a in range(3) if a != ax]
        P[m, ax] = sgn * dims[ax] / 2
        P[m, others[0]] = u[m] * dims[others[0]]
        P[m, others[1]] = v[m] * dims[others[1]]
    return P


def init_actor(track: Track, points_local=None, rgb=None, n_surface: int = 400, seed: int = 0, device='cpu',
               wheels: bool = True) -> ActorModel:
    """An actor's Gaussians at initialisation: the lidar points that fell in its box (box frame) when there are
    enough, else samples on its box's surface (pedestrians: a shrunken box), small isotropic Gaussians, opacity 0.5;
    colours given (rgb), else mid-grey."""
    if points_local is not None and len(points_local) >= 30:
        P = np.asarray(points_local, float)
        C = np.asarray(rgb, float) if rgb is not None else np.full((len(P), 3), 0.5)
    else:
        size = track.size
        if track.kind == 'pedestrian':
            size = (0.6 * size[0], 0.6 * size[1], 0.95 * size[2])
        P = box_surface(size, n_surface, seed)
        C = np.full((len(P), 3), 0.5) if rgb is None else np.asarray(rgb, float)
    L, W, H = track.size
    s0 = max(0.02, 0.5 * math.sqrt((2 * (L * W + L * H + W * H)) / max(len(P), 1)))
    return ActorModel(track, P, np.full((len(P), 3), s0), np.tile([1.0, 0, 0, 0], (len(P), 1)),
                      np.full(len(P), 0.5), C, device, wheels=wheels)


# =========================================================================== the scene: static + actors
class DynamicModel:
    """The static Gaussians (core.recon.gaussians.Gaussians) and the actors, rendered together (module doc)."""

    def __init__(self, gs, actors: list):
        self.gs = gs
        self.actors = list(actors)

    @property
    def device(self):
        return self.gs.device

    def gaussians_at(self, t: float, centre=None, sh_degree=None, boxes: dict | None = None) -> dict:
        """Every Gaussian at time t, static first: means, scales, quats, opacity, rgb (view-dependent for the static
        ones when centre, the camera's position, is given), sky, lidar_int, lidar_drop, velocity, actor (−1 static,
        else the index into self.actors), n_static. boxes: {track id: (R, centre[, velocity])} — the actors' boxes
        at t instead of their logged ones (closed loop); a track not in it is not drawn."""
        from .raster import sh_to_color
        gs = self.gs
        means = gs.p['means']
        if centre is not None:
            dirs = means - centre
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            deg = gs.sh_degree if sh_degree is None else min(int(sh_degree), gs.sh_degree)
            rgb = sh_to_color(gs.sh(), dirs, deg)
        else:
            rgb = (gs.p['sh_dc'][:, 0] * SH_C0 + 0.5).clamp_min(0)
        n = len(gs)
        dev = means.device
        parts = {'means': [means], 'scales': [gs.scales()], 'quats': [gs.p['quats']], 'opacity': [gs.opacities()],
                 'rgb': [rgb], 'sky': [gs.sky], 'velocity': [torch.zeros(n, 3, device=dev)],
                 'actor': [torch.full((n,), -1, dtype=torch.long, device=dev)],
                 'lidar_int': [gs.p['lidar_int'] if 'lidar_int' in gs.p else torch.zeros(n, device=dev)],
                 'lidar_drop': [gs.p['lidar_drop'] if 'lidar_drop' in gs.p else torch.full((n,), -3.0, device=dev)]}
        for k, a in enumerate(self.actors):
            if boxes is not None:
                b = boxes.get(a.track.id)
                if b is None:
                    continue
                w = a.world(t, box=(b[0], b[1]), velocity=b[2] if len(b) > 2 else None)
            else:
                w = a.world(t)
            if w is None:
                continue
            m = len(w['means'])
            for key in ('means', 'scales', 'quats', 'opacity', 'rgb', 'velocity', 'lidar_int', 'lidar_drop'):
                parts[key].append(w[key])
            parts['sky'].append(torch.zeros(m, dtype=torch.bool, device=dev))
            parts['actor'].append(torch.full((m,), k, dtype=torch.long, device=dev))
        out = {k: torch.cat(v, 0) for k, v in parts.items()}
        out['n_static'] = n
        return out

    def render(self, viewmat, K, width: int, height: int, t: float, sh_degree=None, background=None,
               extra=None, tile: int = 8, boxes: dict | None = None) -> dict:
        """One pinhole view at time t (core.recon.raster.render with every Gaussian): {'image' (H, W, 3), 'alpha',
        'range' (the median depth's, 0 for nothing or sky), 'sky', 'actor' (H, W) the median Gaussian's actor index
        (−1 static or none), 'means2d', 'visible', 'radius' (all Gaussians, static first), 'n_static', 'pairs'}.
        extra: (N_static,) features are not supported here (semantic votes are the static scene's)."""
        from .raster import ray_norms, render
        R, tt = viewmat[:3, :3], viewmat[:3, 3]
        centre = -(R.T @ tt)
        G = self.gaussians_at(t, centre, sh_degree, boxes)
        med = torch.stack([G['sky'].to(G['rgb'].dtype), G['actor'].to(G['rgb'].dtype)], 1)
        out = render(G['means'], G['scales'], G['quats'], G['opacity'], G['rgb'], viewmat, K, width, height,
                     background, None, tile=tile, median_features=med)
        m = out['median']
        d, sky = m[..., 0], m[..., 1] > 0.5
        rng = torch.where((d > 0) & ~sky, d * ray_norms(K, width, height, viewmat.device, d.dtype),
                          torch.zeros_like(d))
        act = torch.where(d > 0, m[..., 2].round().long(), torch.full_like(d, -1, dtype=torch.long))
        return {'image': out['image'], 'alpha': out['alpha'], 'range': rng, 'sky': sky & (d > 0), 'actor': act,
                'means2d': out['means2d'], 'visible': out['visible'], 'radius': out['radius'],
                'n_static': G['n_static'], 'pairs': out['info']['pairs']}

    def render_fisheye(self, viewmat, intr, width, height, t: float, background=None, method: str = 'cube') -> dict:
        from .lens_raster import render_fisheye
        R, tt = viewmat[:3, :3], viewmat[:3, 3]
        G = self.gaussians_at(t, -(R.T @ tt))
        return render_fisheye(G['means'], G['scales'], G['quats'], G['opacity'], G['rgb'], viewmat, intr, width,
                              height, background, method)

    def regulariser(self, t: float, **kw) -> torch.Tensor:
        loss = torch.zeros((), device=self.device)
        for a in self.actors:
            loss = loss + a.regulariser(t=t if a.track.shown(t) else None, **kw)
        return loss

    def state(self) -> dict:
        return {'schema': 'falcon.recon_actors/1', 'actors': [a.state() for a in self.actors]}

    @staticmethod
    def actors_from_state(d: dict, device='cpu') -> list:
        return [ActorModel.from_state(x, device) for x in d.get('actors') or []]
