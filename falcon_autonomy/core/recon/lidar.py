"""A neural lidar from the same Gaussians as the images (SplatAD-like, a first cut), and a geometric radar (torch at
import).

Lidar
-----
A recorded lidar is described by its beam grid (LidarSpec): the rings' elevations, the columns' azimuths in firing
order, each column's firing time as a share of the sweep's period, the range limits. Falcon's lidar sensors give it
from their configuration (spec_of_sensor: CARLA's evenly spaced channels between lower_fov and upper_fov,
points_per_second / (channels × rotation_frequency) columns); nuScenes' LIDAR_TOP from its points (spec_of_points: the
median elevation of each ring index, the columns its densest ring had).

Rendering a sweep (render_sweep; ported from Vantage core/splat_reproject.py lidar_beams / render_lidar): the beam
grid is rendered in sectors of azimuth as an image whose pixel centres are the beams (core.recon.lens_raster's
SphereProjector and splat: each Gaussian's footprint in azimuth × elevation from the spherical projection's Jacobian,
dilated by LIDAR_LOWPASS = 0.05 px² instead of an image's 0.3 — a beam is one ray), every Gaussian with three
features composited front to back:

* range — the range along the beam to the plane of the Gaussian (its thinnest axis the normal, to first order across
  the splat; a Gaussian not flatter than half its middle scale counts as a sphere);
* intensity — σ(the Gaussian's intensity logit);
* ray drop — the Gaussian's drop logit ℓᵢ.

With weights wᵢ = αᵢ Tᵢ and α = Σ wᵢ (1 − the transmittance left over): the expected range r̂ = Σ wᵢ rᵢ / α, the
intensity Î = Σ wᵢ Iᵢ / α, the drop logit ℓ = Σ wᵢ ℓᵢ / α, and the probability that the beam returns
p_ret = α · (1 − σ(ℓ)): a beam drops when it meets nothing (the leftover transmittance) or when what it meets drops
it (glass, dark or wet surfaces, sky behind thin structures — whatever the recording shows). Actors (core.recon.
actors) are posed at each sector's mean firing time; the sensor at the sweep's pose (the recorded points are in the
sensor's frame at one pose: CARLA's partial sweep per tick, nuScenes' ego-motion-compensated sweep).

Losses (lidar_loss), on the columns the recording observed (a column with at least one return; a column whose every
beam dropped is indistinguishable from one not fired in a partial sweep, so it is left out): L1 on the range of the
beams that returned (relative to the range, so far beams do not dominate), binary cross-entropy of p_ret against
returned / dropped, L1 on the intensity of the beams that returned. A rendered return (to_points) is a beam with
p_ret > ½, at the median range (the range of the Gaussian at which the transmittance falls through ½) where there is
one, else the expected range.

Radar (render_radar): **geometric, not learned** — the radar's field rendered the same way on a 1° × 1° grid, a
detection per cell the Gaussians cover (α > ½) within its range: the range, the azimuth and elevation, the radial
velocity of what it hits (an actor Gaussian's track velocity, 0 for the static scene) relative to the sensor's own,
and an RCS **proxy** (10·log10 of α·|cos incidence|·the cell's area at that range, dBsm) — no material, multipath or
Doppler noise; nothing about a radar is fitted to recordings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

LIDAR_LOWPASS = 0.05
SECTORS = 16
MAX_COLUMNS = 4096


@dataclass
class LidarSpec:
    elevations: np.ndarray              # (R,) degrees, ring 0 the lowest
    az: np.ndarray                      # (C,) degrees, sensor frame (x forward, y left: ccw), in firing order
    frac: np.ndarray                    # (C,) each column's firing time as a share of the period
    period_s: float = 0.0               # 0: every column at the sweep's instant
    range_min: float = 0.5
    range_max: float = 120.0
    name: str = 'lidar'

    @property
    def rings(self) -> int:
        return int(len(self.elevations))

    @property
    def columns(self) -> int:
        return int(len(self.az))

    def doc(self) -> dict:
        return {'name': self.name, 'elevations_deg': [round(float(x), 5) for x in self.elevations],
                'az0_deg': float(self.az[0]), 'columns': self.columns, 'sense': int(np.sign(self.az[1] - self.az[0]))
                if self.columns > 1 else 1, 'period_s': self.period_s, 'range_min': self.range_min,
                'range_max': self.range_max}

    @classmethod
    def from_doc(cls, d: dict) -> 'LidarSpec':
        C = int(d['columns'])
        k = np.arange(C)
        az = (float(d['az0_deg']) + int(d.get('sense', 1)) * k * 360.0 / C + 180.0) % 360.0 - 180.0
        return cls(np.asarray(d['elevations_deg'], float), az, k / C, float(d.get('period_s', 0.0)),
                   float(d.get('range_min', 0.5)), float(d.get('range_max', 120.0)), d.get('name', 'lidar'))

    def directions(self) -> np.ndarray:
        """(R, C, 3) unit beam directions in the sensor frame (x forward, y left, z up)."""
        e = np.radians(self.elevations)[:, None]
        a = np.radians(self.az)[None, :]
        return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e) * np.ones_like(a)], -1)


def _spec(elev, columns, az0=0.0, sense=1, period=0.0, rmin=0.5, rmax=120.0, name='lidar'):
    C = int(min(max(columns, 8), MAX_COLUMNS))
    k = np.arange(C)
    az = (az0 + sense * k * 360.0 / C + 180.0) % 360.0 - 180.0
    return LidarSpec(np.sort(np.asarray(elev, float)), az, k / C, float(period), float(rmin), float(rmax), name)


def spec_of_sensor(s: dict, period: float = 0.0) -> LidarSpec:
    """A Falcon lidar sensor's beam grid (core.rig's lidar: channels, upper_fov, lower_fov, points_per_second,
    rotation_frequency, range). CARLA casts every channel at evenly spaced elevations."""
    ch = int(s.get('channels', 32))
    up, lo = float(s.get('upper_fov', 15.0)), float(s.get('lower_fov', -25.0))
    pps, f = float(s.get('points_per_second', 600000)), float(s.get('rotation_frequency', 10.0))
    cols = int(round(pps / max(ch * f, 1e-9)))
    elev = np.linspace(lo, up, ch) if ch > 1 else np.array([up])
    return _spec(elev, cols, 0.0, 1, period, 0.5, float(s.get('range', 100.0)), s.get('id', 'lidar'))


def spec_of_points(sweeps: list, period: float = 0.05, name: str = 'LIDAR_TOP') -> LidarSpec:
    """A recorded lidar's beam grid measured from sweeps [(n, ≥4) x y z (sensor frame, right-handed) intensity
    [ring]]: with a ring column, each ring's median elevation; without, the elevations clustered (peaks of a 0.1°
    histogram); the columns the densest ring holds in one sweep; the turning sense from the azimuth order."""
    els, rings, cnt, votes = [], [], 0, []
    for a in sweeps[:8]:
        P = np.asarray(a, float)
        if not len(P):
            continue
        h = np.hypot(P[:, 0], P[:, 1])
        e = np.degrees(np.arctan2(P[:, 2], h))
        els.append(e)
        if P.shape[1] >= 5:
            r = P[:, 4].astype(int)
            rings.append(r)
            cnt = max(cnt, int(np.bincount(np.clip(r, 0, None)).max()))
        if len(P) > 100:
            dd = np.diff(np.degrees(np.arctan2(P[:, 1], P[:, 0])))
            votes.append(np.sign(np.sum((dd + 180.0) % 360.0 - 180.0)))
    if not els:
        raise ValueError('no lidar points to measure the beam grid from')
    e = np.concatenate(els)
    if rings:
        r = np.concatenate(rings)
        elev = [float(np.median(e[r == k])) for k in np.unique(r) if (r == k).sum() >= 3]
    else:
        hist, edges = np.histogram(e, bins=np.arange(e.min() - 0.1, e.max() + 0.2, 0.1))
        pk = [i for i in range(1, len(hist) - 1) if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1] and
              hist[i] > 0.2 * np.median(hist[hist > 0])]
        elev = [0.5 * (edges[i] + edges[i + 1]) for i in pk]
        cnt = max(cnt, int(len(e) / max(len(elev), 1) / max(len(els), 1)))
    sense = 1 if not votes or np.mean(votes) >= 0 else -1
    return _spec(elev, max(cnt, 64), 0.0, sense, period, 0.5, 120.0, name)


# =========================================================================== targets
def beam_targets(spec: LidarSpec, P, intensity=None) -> dict:
    """A recorded sweep on the beam grid: P (n, 3) in the sensor frame (right-handed), intensity (n,) or None.
    {'range' (R, C) m (NaN: no return), 'intensity' (R, C), 'observed' (C,) bool (a return in the column),
    'returned' (R, C) bool}. A point goes to the nearest ring (within half the ring spacing, else left out) and the
    nearest column; two points on one beam: the nearer kept."""
    P = np.asarray(P, float)
    R, C = spec.rings, spec.columns
    rng = np.full((R, C), np.nan)
    inten = np.zeros((R, C))
    if len(P):
        r = np.linalg.norm(P, axis=1)
        e = np.degrees(np.arctan2(P[:, 2], np.hypot(P[:, 0], P[:, 1])))
        a = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
        el = spec.elevations
        ring = np.clip(np.searchsorted(el, e), 1, max(R - 1, 1)) if R > 1 else np.zeros(len(e), int)
        if R > 1:
            ring = np.where(np.abs(e - el[ring - 1]) < np.abs(e - el[ring]), ring - 1, ring)
            sp = np.diff(el)
            half = 0.5 * np.r_[sp[0], np.minimum(sp[:-1], sp[1:]), sp[-1]] if R > 2 else np.full(R, 0.5 * sp[0])
            ok = np.abs(e - el[ring]) <= half[ring] + 1e-6
        else:
            ok = np.ones(len(e), bool)
        step = 360.0 / C
        sense = 1 if C < 2 or ((spec.az[1] - spec.az[0] + 180.0) % 360.0 - 180.0) > 0 else -1
        col = np.round(((a - spec.az[0]) * sense % 360.0) / step).astype(int) % C
        ok &= (r >= spec.range_min) & (r <= spec.range_max)
        order = np.argsort(-r)                                # nearer points written last: they win
        for i in order[ok[order]]:
            rng[ring[i], col[i]] = r[i]
            if intensity is not None:
                inten[ring[i], col[i]] = float(intensity[i])
    ret = np.isfinite(rng)
    return {'range': rng, 'intensity': inten, 'observed': ret.any(0), 'returned': ret}


# =========================================================================== rendering
def _grid(spec: LidarSpec):
    el = np.asarray(spec.elevations, float)
    R, C = len(el), len(spec.az)
    e_step = float(min(np.min(np.diff(np.sort(el))) if R > 1 else 0.5, 0.5))
    e_step = max(e_step, 0.05)
    e_hi = float(el.max()) + 2.5 * e_step
    height = int(math.ceil((e_hi - (el.min() - 2 * e_step)) / e_step))
    rows = np.clip(np.floor((e_hi - el) / e_step).astype(int), 0, height - 1)
    step = 360.0 / C
    nsec = SECTORS if C >= SECTORS * 4 else max(1, C // 8)
    return rows, e_hi, e_step, height, step, nsec


def _normals_cos(Rs, s, Ps):
    """Each Gaussian's thinnest axis in the sensor frame (the ray's direction for a near-isotropic one), and
    |cos| of the incidence."""
    jmin = s.argmin(1)
    iso = s.min(1).values > 0.5 * s.median(1).values
    rr = torch.linalg.norm(Ps, dim=1).clamp(min=1e-6)
    n = torch.gather(Rs, 2, jmin[:, None, None].expand(-1, 3, 1))[..., 0]
    cs = torch.where(iso, torch.ones_like(rr), (n * Ps).sum(1).abs() / rr)
    n = torch.where(iso[:, None], Ps / rr[:, None], n)
    return n, cs


def render_sweep(gauss_at, spec: LidarSpec, T_world_sensor, t_start: float, grad: bool = False,
                 extra_feats=None) -> dict:
    """One sweep (module doc). gauss_at(t) -> dict with means, scales, quats, opacity, lidar_int, lidar_drop (and
    velocity) of every Gaussian in the world at time t (core.recon.actors.DynamicModel.gaussians_at, or a static
    scene's); T_world_sensor: world ← sensor (4 × 4, right-handed). Returns torch tensors (R, C): 'alpha', 'range'
    (expected), 'median' (no gradient; NaN where none), 'intensity', 'drop_logit', 'p_return', 'cos'."""
    from .lens_raster import SphereProjector, covariance, quat_to_rot, splat
    T = torch.as_tensor(np.asarray(T_world_sensor, float) if not torch.is_tensor(T_world_sensor) else
                        T_world_sensor.detach().cpu().numpy(), dtype=torch.float32)
    rows, e_hi, e_step, height, step, nsec = _grid(spec)
    R, C = spec.rings, spec.columns
    cache = {}
    outs = {k: [None] * C for k in ()}
    acc = {k: [] for k in ('alpha', 'range', 'median', 'intensity', 'drop_logit', 'cos')}
    cols_all = []
    with torch.set_grad_enabled(grad):
        for k in range(nsec):
            cols = np.arange(k * C // nsec, (k + 1) * C // nsec)
            if not len(cols):
                continue
            tm = t_start + float(np.mean(spec.frac[cols])) * spec.period_s
            key = round(tm, 6)
            if key not in cache:
                cache.clear()
                cache[key] = gauss_at(tm)
            G = cache[key]
            dev = G['means'].device
            Tw = T.to(dev)
            W2S = Tw[:3, :3].T
            o = Tw[:3, 3]
            Ps = (G['means'] - o) @ W2S.T
            Rs = W2S[None] @ quat_to_rot(G['quats'])
            s = G['scales']
            nrm, cs = _normals_cos(Rs, s, Ps)
            rr = torch.linalg.norm(Ps, dim=1)
            feats = torch.stack([rr if grad else rr.detach(), torch.sigmoid(G['lidar_int']), G['lidar_drop'], cs], 1)
            azc = math.degrees(math.atan2(float(np.mean(np.sin(np.radians(spec.az[cols])))),
                                          float(np.mean(np.cos(np.radians(spec.az[cols]))))))
            rel = (spec.az[cols] - azc + 180.0) % 360.0 - 180.0
            m = 2
            a_lo = float(rel.min()) - (m + 0.5) * step
            width = int(math.ceil((rel.max() - rel.min()) / step)) + 2 * m + 2
            proj = SphereProjector(azc, a_lo, step, e_hi, e_step, width, height)
            ci = np.clip(np.floor((rel - a_lo) / step).astype(int), 0, width - 1)
            res = splat(Ps, covariance(Rs, s), feats, G['opacity'], proj, width, height, normals=nrm.detach(),
                        depth_col=0, lowpass=LIDAR_LOWPASS)
            img = res['image'].reshape(height, width, -1)
            al = res['alpha'].reshape(height, width)
            ri = torch.as_tensor(rows, device=dev)[:, None]
            cj = torch.as_tensor(ci, device=dev)[None, :]
            a_ = al[ri, cj]
            den = a_.clamp(min=1e-6)
            acc['alpha'].append(a_)
            acc['range'].append(img[ri, cj, 0] / den)
            acc['intensity'].append(img[ri, cj, 1] / den)
            acc['drop_logit'].append(img[ri, cj, 2] / den)
            acc['cos'].append(img[ri, cj, 3] / den)
            med = res.get('median')
            acc['median'].append(med.reshape(height, width)[ri, cj] if med is not None else
                                 torch.full_like(a_, float('nan')))
            cols_all.append(cols)
        order = np.argsort(np.concatenate(cols_all))
        oi = torch.as_tensor(order, device=acc['alpha'][0].device)
        out = {k: torch.cat(v, 1)[:, oi] for k, v in acc.items()}
    del outs
    out['p_return'] = out['alpha'] * (1.0 - torch.sigmoid(out['drop_logit']))
    out['time'] = t_start + spec.frac * spec.period_s
    return out


def lidar_loss(out: dict, tgt: dict, w_range: float = 1.0, w_drop: float = 0.2, w_int: float = 0.2) -> tuple:
    """(loss, parts) of a rendered sweep against its recorded beam targets (module doc)."""
    dev = out['alpha'].device
    obs = torch.as_tensor(tgt['observed'], device=dev)[None, :].expand_as(out['alpha'])
    ret = torch.as_tensor(tgt['returned'], device=dev) & obs
    r_t = torch.as_tensor(np.nan_to_num(tgt['range'], nan=0.0), dtype=torch.float32, device=dev)
    zero = torch.zeros((), device=dev)
    l_rng = ((out['range'] - r_t).abs() / r_t.clamp(min=1.0))[ret].mean() if ret.any() else zero
    p = out['p_return'].clamp(1e-5, 1 - 1e-5)
    y = ret.to(p.dtype)
    l_drop = torch.nn.functional.binary_cross_entropy(p[obs], y[obs]) if obs.any() else zero
    i_t = torch.as_tensor(tgt['intensity'], dtype=torch.float32, device=dev)
    l_int = (out['intensity'] - i_t).abs()[ret].mean() if ret.any() else zero
    loss = w_range * l_rng + w_drop * l_drop + w_int * l_int
    return loss, {'range_rel_l1': float(l_rng), 'drop_bce': float(l_drop), 'intensity_l1': float(l_int)}


def to_points(spec: LidarSpec, out: dict, threshold: float = 0.5) -> dict:
    """The rendered returns: {'points' (n, 4) x y z intensity in the sensor frame (right-handed), 'ring', 'col'
    (n,), 'range' (R, C) NaN where none, 'returns', 'beams'}."""
    p = out['p_return'].detach().cpu().numpy()
    med = out['median'].detach().cpu().numpy()
    exp = out['range'].detach().cpu().numpy()
    rng = np.where(np.isfinite(med), med, exp)
    ok = (p > threshold) & (rng >= spec.range_min) & (rng <= spec.range_max)
    rng = np.where(ok, rng, np.nan)
    rr, cc = np.nonzero(ok)
    D = spec.directions()
    P = D[rr, cc] * rng[rr, cc][:, None]
    I = out['intensity'].detach().cpu().numpy()[rr, cc]
    return {'points': np.c_[P, I].astype(np.float32), 'ring': rr, 'col': cc, 'range': rng,
            'returns': int(ok.sum()), 'beams': int(ok.size)}


def compare_sweep(spec: LidarSpec, out: dict, tgt: dict) -> dict:
    """A rendered sweep against the recorded one on the observed columns: the range error where both return
    (median and mean absolute, m; the share within 0.1 m), the ray-drop accuracy, returns only recorded and only
    rendered, the intensity error."""
    pts = to_points(spec, out)
    obs = np.broadcast_to(tgt['observed'][None, :], tgt['returned'].shape)
    rec = tgt['returned'] & obs
    ren = np.isfinite(pts['range']) & obs
    both = rec & ren
    err = np.abs(pts['range'] - np.nan_to_num(tgt['range']))[both]
    inten = out['intensity'].detach().cpu().numpy()
    return {'beams_observed': int(obs.sum()), 'returns_recorded': int(rec.sum()), 'returns_rendered': int(ren.sum()),
            'both': int(both.sum()), 'only_recorded': int((rec & ~ren).sum()), 'only_rendered': int((ren & ~rec).sum()),
            'range_median_abs_m': round(float(np.median(err)), 4) if len(err) else None,
            'range_mean_abs_m': round(float(np.mean(err)), 4) if len(err) else None,
            'range_within_10cm': round(float(np.mean(err < 0.1)), 4) if len(err) else None,
            'drop_accuracy': round(float(np.mean((ren == rec)[obs])), 4) if obs.any() else None,
            'intensity_mean_abs': round(float(np.mean(np.abs(inten - tgt['intensity'])[both])), 4)
            if both.any() else None}


# =========================================================================== radar
def render_radar(gauss_at, sensor: dict, T_world_sensor, t: float, ego_velocity=(0.0, 0.0, 0.0),
                 step_deg: float = 1.0) -> dict:
    """Geometric radar detections (module doc): sensor a Falcon radar dict (horizontal_fov, vertical_fov, range);
    T_world_sensor world ← sensor (right-handed); ego_velocity the sensor's world velocity. Returns {'detections'
    (n, 4): CARLA's radar layout — velocity (m/s, + away), azimuth (rad, + to the right as CARLA's), altitude (rad,
    + up), depth (m) —, 'rcs_proxy_dbsm' (n,), 'learned': False}."""
    from .lens_raster import SphereProjector, covariance, quat_to_rot, splat
    hf, vf = float(sensor.get('horizontal_fov', 30.0)), float(sensor.get('vertical_fov', 10.0))
    rmax = float(sensor.get('range', 100.0))
    W, H = max(1, int(round(hf / step_deg))), max(1, int(round(vf / step_deg)))
    with torch.no_grad():
        G = gauss_at(t)
        dev = G['means'].device
        Tw = torch.as_tensor(np.asarray(T_world_sensor, float), dtype=torch.float32, device=dev)
        W2S = Tw[:3, :3].T
        Ps = (G['means'] - Tw[:3, 3]) @ W2S.T
        Rs = W2S[None] @ quat_to_rot(G['quats'])
        nrm, cs = _normals_cos(Rs, G['scales'], Ps)
        v_rel = (G['velocity'] - torch.as_tensor(np.asarray(ego_velocity, float), dtype=torch.float32,
                                                 device=dev)) @ W2S.T
        rr = torch.linalg.norm(Ps, dim=1)
        feats = torch.cat([rr[:, None], cs[:, None], v_rel], 1)
        proj = SphereProjector(0.0, -hf / 2.0, step_deg, vf / 2.0, step_deg, W, H)
        res = splat(Ps, covariance(Rs, G['scales']), feats, G['opacity'], proj, W, H, normals=nrm, depth_col=0,
                    lowpass=LIDAR_LOWPASS)
        img = res['image'].reshape(H, W, -1).cpu().numpy()
        al = res['alpha'].reshape(H, W).cpu().numpy()
        med = res['median'].reshape(H, W).cpu().numpy()
    den = np.maximum(al, 1e-6)
    rng = np.where(np.isfinite(med), med, img[..., 0] / den)
    ok = (al > 0.5) & (rng > 0.5) & (rng <= rmax)
    jj, ii = np.nonzero(ok)
    az = np.radians(-hf / 2.0 + (ii + 0.5) * step_deg)
    el = np.radians(vf / 2.0 - (jj + 0.5) * step_deg)
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    v = img[jj, ii, 2:5] / den[jj, ii, None]
    vr = np.sum(v * d, 1)
    cell = (np.radians(step_deg) ** 2) * rng[jj, ii] ** 2
    rcs = 10 * np.log10(np.maximum(al[jj, ii] * img[jj, ii, 1] / den[jj, ii] * cell, 1e-6))
    det = np.stack([vr, -az, el, rng[jj, ii]], 1).astype(np.float32)
    return {'detections': det, 'rcs_proxy_dbsm': rcs.astype(np.float32), 'learned': False,
            'note': 'geometric: range, angle and radial velocity of what the Gaussians cover; RCS is a proxy'}
