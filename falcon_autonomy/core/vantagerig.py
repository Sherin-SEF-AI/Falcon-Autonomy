"""Vantage projects as Falcon rigs (standard library + numpy): read a `.vantage` file, turn each of its variants into a
Falcon rig on one CARLA vehicle, and say where the variant's stitch seams lie.

Vantage (the sibling app that plans camera rigs geometrically, GPL-3+) is never imported: its project format is read
here from the file. A `.vantage` file is a zip holding project.json (and mesh.npz, the body, and thumb.png). The file
is opened read-only and never written; Falcon keeps only its path and SHA-256.

What is read (project.json, format 'vantage', version 1)
--------------------------------------------------------
* variants[]: uid, name, note, cameras[], sensors[], seams {'uidA|uidB': azimuth degrees} (the seams the user set),
  blend_m.
* a camera: uid, tag, name, x y z (metres), yaw pitch roll (degrees), module_id, lens_id, calib (a unit calibration,
  which replaces the library lens as Vantage's Project.lens_for does), visible (a hidden camera is left out).
* modules{id}: width_px, height_px, pixel_um, fps. lenses{id}: focal_mm, projection, kb (Kannala-Brandt k1-k4),
  sheet_fov (H, V, D), max_half_deg, pp_offset_px, aspect.
* a sensor (lidar, radar, gnss, imu): uid, tag, name, kind, model_id, x y z yaw pitch roll, visible; sensor_models{id}:
  channels, v_min, v_max, beams, h_fov, h_res, rate_hz, range_m, range_max_m, points_per_s, zones.
* vehicle: parts (names and roles), wheel_centres, spec; mesh.npz's vertices, faces and face_part for the body's
  extents (length, width, height, wheelbase) and its centre.
* roi: mode, distance, polygon (the ground region the seam estimate scores, as Vantage's analysis does).
Everything else (targets, photos, obstacles, brackets, ECU, streams, the analysis settings, the log, cable routing,
ports, sync lines, bracket stand-offs, housings) is ignored: none of it changes what a simulated camera sees.

Frames
------
Vantage: ISO 8855, x forward, y left, z up, origin at the rear-axle centre on the ground; yaw / pitch / roll with
pitch positive DOWN, R = Rz(yaw)·Ry(pitch)·Rx(roll) (vantage.core.mathutil.rot_zyx); a camera's body x is its optical
axis. Falcon: CARLA's vehicle frame, x forward, y right, z up, origin on the ground under the vehicle's bounding-box
centre. A pose maps by sim.frames (y mirrored; carla yaw = −yaw, pitch = −pitch, roll = roll, by conjugation) and the
rear axle is placed where the CARLA vehicle's rear axle is (CARLA_VEHICLES' rear_axle_x, measured by Vantage on a
0.9.16 server): x_falcon = x_vantage + rear_axle_x. This is how Vantage's own CARLA bridge mounts a rig.

The CARLA vehicle
-----------------
CARLA renders its own car, not the project's body. Every variant of one import goes on the same vehicle (so their runs
are comparable): the one asked for, or the 4-wheeled one nearest the body's length, width, height and wheelbase (the
mean relative difference; a CARLA box's width includes the mirrors, so the body's width is compared with 92 % of it),
Vantage's own rule (vantage.core.simrig.match_vehicle). A camera that sat on the project's body may then sit inside or
off the CARLA car's: the size error is reported.

Lenses
------
A Vantage lens is resolved as vantage.core.lens.resolve does: f = focal_mm / pixel size, the principal point at the
sensor centre plus pp_offset_px (pixel-edge coordinates, the same as Falcon's), the image-circle limit max_half_deg,
else half the datasheet diagonal + 2° (at most 95°), else the sensor's half diagonal through the model + 1°. A unit
calibration's (fx, fy, cx, cy in OpenCV pixel-centre coordinates, k1-k4) scaled to the module's resolution replaces
it, as vantage.core.calib.unit_lens does. The camera becomes a Falcon fisheye camera with that lens block (the model,
f_px, cx, cy, k, aspect, fov = twice the limit), which core.lens renders as CARLA pinhole views remapped through
exactly that model; a centred rectilinear lens becomes a pinhole camera. The image is scaled by image_scale (default
0.5, Vantage's own simulation default): sizes, f, cx and cy together.

Seams
-----
Vantage's stitched bird's-eye view hands the ground around the body to its cameras by azimuth about the body centre
(x the middle of the body's length, y = 0): seam k between neighbouring cameras a and b (ordered counter-clockwise)
gives the sector after it to b. A seam the user set in Vantage (variant.seams['uidA|uidB']) is used as it is. Vantage
does not store the seams it computes itself; those are estimated here by Vantage's own rule (vantage.core.analysis.
compute_seams: each camera's density-weighted mean azimuth where it sees the ground best, then for each neighbouring
pair the candidate ray maximising the shared, well-resolved ground) on a ground grid around the body's box, with the
body box as the only occluder (Vantage uses the body's mesh). Measured against Vantage's own analysis on its sample
Tigor project (five variants, 24 seams): see docs/rig-compare.md. Each seam says which it is ('vantage' or
'estimated').
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import zipfile

import numpy as np

from ..sim.frames import BODY_TO_OPTICAL, carla_rotation
from . import lens as LENS

FORMAT = 'vantage'
MAX_BYTES = 2_000_000_000
IMAGE_SCALE = 0.5
CAMERA_LABELS = ['semantic', 'instance', 'depth', 'boxes2d']
SEAM_CELL_M = 0.05
SEAM_ROI_M = 2.0               # Vantage's default ground region: 2 m round the body's outline
SEAM_MIN_RADIUS_M = 0.3
# blueprint -> (base type, length, width (mirrors included), height, wheelbase, rear axle x in the actor frame), m:
# measured by Vantage on a CARLA 0.9.16 server (no rendering) on 2026-10-27 (vantage.core.simrig.CARLA_VEHICLES)
CARLA_VEHICLES = {
    'vehicle.audi.a2': ('car', 3.705, 1.789, 1.549, 2.506, -1.256),
    'vehicle.audi.etron': ('car', 4.856, 2.033, 1.649, 2.900, -1.386),
    'vehicle.audi.tt': ('car', 4.181, 1.994, 1.385, 2.642, -1.400),
    'vehicle.bmw.grandtourer': ('car', 4.611, 2.242, 1.667, 2.935, -1.443),
    'vehicle.carlamotors.carlacola': ('truck', 5.204, 2.627, 2.467, 2.665, -1.206),
    'vehicle.carlamotors.european_hgv': ('truck', 7.936, 2.891, 3.462, 4.995, -2.678),
    'vehicle.carlamotors.firetruck': ('truck', 8.468, 2.891, 3.827, 4.995, -2.678),
    'vehicle.chevrolet.impala': ('car', 5.357, 2.033, 1.411, 2.982, -1.329),
    'vehicle.citroen.c3': ('car', 3.988, 1.851, 1.617, 2.684, -1.410),
    'vehicle.dodge.charger_2020': ('car', 5.008, 1.882, 1.535, 3.044, -1.433),
    'vehicle.dodge.charger_police': ('car', 4.974, 2.038, 1.542, 2.992, -1.501),
    'vehicle.dodge.charger_police_2020': ('car', 5.238, 1.930, 1.638, 3.044, -1.433),
    'vehicle.ford.ambulance': ('van', 6.366, 2.351, 2.431, 3.770, -1.661),
    'vehicle.ford.crown': ('car', 5.366, 1.801, 1.575, 2.916, -1.143),
    'vehicle.ford.mustang': ('car', 4.718, 1.895, 1.301, 2.886, -1.324),
    'vehicle.jeep.wrangler_rubicon': ('car', 3.866, 1.905, 1.878, 2.308, -1.017),
    'vehicle.lincoln.mkz_2017': ('car', 4.902, 2.128, 1.511, 2.871, -1.394),
    'vehicle.lincoln.mkz_2020': ('car', 4.892, 1.837, 1.490, 2.860, -1.389),
    'vehicle.mercedes.coupe': ('car', 5.027, 2.152, 1.651, 3.289, -1.519),
    'vehicle.mercedes.coupe_2020': ('car', 4.674, 1.812, 1.442, 2.833, -1.287),
    'vehicle.mercedes.sprinter': ('van', 5.915, 1.988, 2.561, 3.662, -1.729),
    'vehicle.micro.microlino': ('car', 2.207, 1.481, 1.376, 1.525, -0.841),
    'vehicle.mini.cooper_s': ('car', 3.806, 1.970, 1.475, 2.511, -1.367),
    'vehicle.mini.cooper_s_2021': ('car', 4.553, 2.097, 1.767, 3.010, -1.640),
    'vehicle.mitsubishi.fusorosa': ('bus', 10.273, 3.944, 4.253, 5.630, -2.615),
    'vehicle.nissan.micra': ('car', 3.633, 1.845, 1.501, 2.490, -1.317),
    'vehicle.nissan.patrol': ('car', 4.605, 1.932, 1.855, 2.839, -1.380),
    'vehicle.nissan.patrol_2021': ('car', 5.566, 2.150, 2.045, 3.279, -1.500),
    'vehicle.seat.leon': ('car', 4.193, 1.816, 1.474, 2.736, -1.420),
    'vehicle.tesla.cybertruck': ('truck', 6.274, 2.390, 2.098, 4.075, -1.907),
    'vehicle.tesla.model3': ('car', 4.792, 2.163, 1.488, 3.005, -1.386),
    'vehicle.toyota.prius': ('car', 4.514, 2.007, 1.525, 2.819, -1.399),
    'vehicle.volkswagen.t2': ('van', 4.480, 2.069, 2.038, 2.444, -1.277),
    'vehicle.volkswagen.t2_2021': ('van', 4.442, 1.775, 1.987, 2.544, -1.299),
}
SMALL_CAR = 'vehicle.citroen.c3'           # Vantage's choice for a project without a body
SECTORS = ('front', 'front-left', 'left', 'rear-left', 'rear', 'rear-right', 'right', 'front-right')
READ_FIELDS = {
    'variant': ('uid', 'name', 'note', 'cameras', 'sensors', 'seams', 'blend_m'),
    'camera': ('uid', 'tag', 'name', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'module_id', 'lens_id', 'calib', 'visible'),
    'module': ('width_px', 'height_px', 'pixel_um', 'fps'),
    'lens': ('focal_mm', 'projection', 'kb', 'sheet_fov', 'max_half_deg', 'pp_offset_px', 'aspect'),
    'sensor': ('uid', 'tag', 'name', 'kind', 'model_id', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'visible'),
    'sensor_model': ('kind', 'channels', 'v_min', 'v_max', 'beams', 'h_fov', 'h_res', 'rate_hz', 'range_m',
                     'range_max_m', 'points_per_s', 'zones'),
    'vehicle': ('name', 'parts', 'wheel_centres', 'spec', 'mesh.npz'),
    'roi': ('mode', 'distance', 'polygon'),
}


class VantageError(ValueError):
    pass


# =========================================================================================== reading the file
def _num(v, default=0.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _nums(v, n, default):
    if not isinstance(v, (list, tuple)) or len(v) < n:
        return list(default)
    out = [_num(x, math.nan) for x in list(v)[:n]]
    return list(default) if any(math.isnan(x) for x in out) else out


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path: str) -> dict:
    """A Vantage project read from its file (never written): {'name', 'path', 'sha256', 'variants': [...], 'active',
    'modules', 'lenses', 'sensor_models', 'body' (the body's extents, or None), 'roi', 'frame', 'notes'}."""
    path = os.path.abspath(os.path.expanduser(path))
    try:
        if os.path.getsize(path) > MAX_BYTES:
            raise VantageError('the file is larger than 2 GB')
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if 'project.json' not in names:
                raise VantageError('not a Vantage project (project.json missing)')
            data = json.loads(z.read('project.json').decode('utf-8'))
            mesh = z.read('mesh.npz') if 'mesh.npz' in names else None
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise VantageError(f'cannot read {os.path.basename(path)}: {e}') from None
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        raise VantageError('not a Vantage project (its format is not "vantage")')
    return parse(data, mesh, path=path, sha256=sha256_of(path))


def parse(data: dict, mesh_bytes: bytes | None = None, path: str = '', sha256: str = '') -> dict:
    """The parts of a project.json (already loaded) Falcon uses; mesh_bytes: mesh.npz's bytes, for the body."""
    notes = []
    if int(_num(data.get('version'), 1)) > 1:
        notes.append(f'format version {data.get("version")}: newer than this reader knows (1); read as version 1')
    variants = []
    for v in data.get('variants') or []:
        if not isinstance(v, dict):
            continue
        cams = [c for c in (v.get('cameras') or []) if isinstance(c, dict)]
        sens = [s for s in (v.get('sensors') or []) if isinstance(s, dict)]
        seams = {str(k): _num(a) for k, a in (v.get('seams') or {}).items()
                 if isinstance(k, str) and '|' in k and _num(a, math.nan) == _num(a, math.nan)}
        variants.append({'uid': str(v.get('uid') or f'v{len(variants)}'), 'name': str(v.get('name') or 'Variant'),
                         'note': str(v.get('note') or ''), 'cameras': cams, 'sensors': sens, 'seams': seams,
                         'blend_m': _num(v.get('blend_m'), 0.2)})
    if not variants:
        raise VantageError('the project has no variants')
    veh = data.get('vehicle') if isinstance(data.get('vehicle'), dict) else {}
    body = body_of(veh, mesh_bytes)
    if body is None:
        notes.append('no body mesh in the file: the vehicle is matched by nothing (a typical small car) and the body '
                     'centre is estimated from the wheel centres, else the rear axle')
    roi = data.get('roi') if isinstance(data.get('roi'), dict) else {}
    return {'name': str(data.get('name') or os.path.splitext(os.path.basename(path))[0] or 'untitled'),
            'path': path, 'sha256': sha256, 'version': data.get('version'),
            'variants': variants, 'active': str(data.get('active') or variants[0]['uid']),
            'modules': {str(k): m for k, m in (data.get('modules') or {}).items() if isinstance(m, dict)},
            'lenses': {str(k): x for k, x in (data.get('lenses') or {}).items() if isinstance(x, dict)},
            'sensor_models': {str(k): m for k, m in (data.get('sensor_models') or {}).items() if isinstance(m, dict)},
            'vehicle_name': str(veh.get('name') or ''), 'body': body,
            'roi': {'mode': roi.get('mode', 'offset'), 'distance': _num(roi.get('distance'), SEAM_ROI_M),
                    'polygon': [p for p in roi.get('polygon') or [] if isinstance(p, (list, tuple)) and len(p) >= 2]},
            'frame': str(data.get('frame') or 'rear_axle'), 'notes': notes}


def body_of(veh: dict, mesh_bytes: bytes | None) -> dict | None:
    """The body's box in Vantage's frame (rear-axle origin) from mesh.npz, as vantage.core.project.Vehicle.measured
    does (mirrors and ignored parts left out of the extents): {'length', 'width', 'height', 'wheelbase', 'front_x',
    'rear_x', 'centre_x', 'source'}; from the wheel centres and the datasheet spec without a mesh; None with neither."""
    wc = veh.get('wheel_centres') or {}
    wb = abs(_num(wc['FL'][0]) - _num(wc['RL'][0])) if all(k in wc for k in ('FL', 'RL')) else None
    if mesh_bytes:
        try:
            with np.load(io.BytesIO(mesh_bytes), allow_pickle=False) as npz:
                V = np.asarray(npz['vertices'], float)
                F = np.asarray(npz['faces'], int)
                fp = np.asarray(npz['face_part'], int)
        except (OSError, KeyError, ValueError):
            V = None
        if V is not None and len(F):
            parts = veh.get('parts') or []
            keep = np.array([p.get('name') != 'Mirrors' and p.get('role') != 'ignored' for p in parts] or [True], bool)
            sel = keep[np.clip(fp, 0, len(keep) - 1)]
            tri = V[F[sel]] if sel.any() else V[F]
            P = tri.reshape(-1, 3)
            lo, hi = P.min(0), P.max(0)
            out = {'length': float(hi[0] - lo[0]), 'width': float(hi[1] - lo[1]), 'height': float(hi[2] - lo[2]),
                   'front_x': float(hi[0]), 'rear_x': float(lo[0]), 'centre_x': float((hi[0] + lo[0]) / 2),
                   'wheelbase': wb, 'source': 'mesh'}
            return out
    spec = veh.get('spec') or {}
    L, W, H = (_num(spec.get(k)) for k in ('Length', 'Width', 'Height'))
    if L > 0 and W > 0 and wb:
        # the body centre about the wheelbase's middle: an assumption without a mesh
        cx = wb / 2.0
        return {'length': L, 'width': W, 'height': H or 1.5, 'front_x': cx + L / 2, 'rear_x': cx - L / 2,
                'centre_x': cx, 'wheelbase': wb, 'source': 'spec (centre assumed at the middle of the wheelbase)'}
    return None


# =========================================================================================== the CARLA vehicle
def match_vehicle(body: dict | None, choice: str = 'auto') -> dict:
    """The CARLA vehicle every variant goes on (module doc): {'blueprint', 'how', 'size_error', 'rear_axle_x',
    'carla' {length, width, height, wheelbase}, 'candidates'}."""
    want = {}
    if body:
        for k in ('length', 'width', 'height', 'wheelbase'):
            if body.get(k):
                want[k] = float(body[k])

    def score(row):
        _, L, W, H, wb, _ = row
        got = {'length': L, 'width': W * 0.92, 'height': H, 'wheelbase': wb}
        return float(np.mean([abs(got[k] - v) / v for k, v in want.items()])) if want else 0.0
    ranked = sorted(CARLA_VEHICLES.items(), key=lambda kv: (score(kv[1]), kv[0]))
    if choice and choice != 'auto':
        if not choice.startswith('vehicle.'):
            raise VantageError(f'{choice!r} is not a CARLA vehicle blueprint (vehicle.*)')
        bp, how = choice, 'chosen'
    elif want:
        bp, how = ranked[0][0], 'nearest by size'
    else:
        bp, how = SMALL_CAR, 'the project has no body: a typical small car'
    row = CARLA_VEHICLES.get(bp)
    out = {'blueprint': bp, 'how': how, 'candidates': [{'blueprint': k, 'score': round(score(r), 4)}
                                                       for k, r in ranked[:4]]}
    if row is None:
        # a blueprint not measured: its rear axle is assumed at 0.28 of its length behind the centre (no length known:
        # the CARLA cars' median, −1.39 m) — said in the notes
        out.update(carla=None, rear_axle_x=-1.39, size_error=None,
                   note=f'{bp} is not in the measured table: the rear axle is assumed 1.39 m behind its centre')
    else:
        out.update(carla={'length': row[1], 'width': row[2], 'height': row[3], 'wheelbase': row[4]},
                   rear_axle_x=row[5], size_error=round(score(row), 4) if want else None)
    return out


# =========================================================================================== lenses and poses
def _module(doc, mid):
    m = doc['modules'].get(mid)
    if m is None:
        return None
    return {'id': mid, 'name': str(m.get('name') or mid), 'width_px': int(_num(m.get('width_px'), 1920)),
            'height_px': int(_num(m.get('height_px'), 1536)), 'pixel_um': _num(m.get('pixel_um'), 3.0) or 3.0,
            'fps': _num(m.get('fps'), 30.0) or 30.0}


def _lens(doc, lid):
    x = doc['lenses'].get(lid)
    if x is None:
        return None
    return {'id': lid, 'name': str(x.get('name') or lid), 'focal_mm': _num(x.get('focal_mm'), 2.62) or 2.62,
            'projection': str(x.get('projection') or 'equisolid'), 'kb': _nums(x.get('kb'), 4, (0.0,) * 4),
            'sheet_fov': _nums(x.get('sheet_fov'), 3, (0.0,) * 3), 'max_half_deg': _num(x.get('max_half_deg')),
            'pp_offset_px': _nums(x.get('pp_offset_px'), 2, (0.0, 0.0)), 'aspect': _num(x.get('aspect'), 1.0) or 1.0}


def camera_lens(doc: dict, cam: dict) -> dict:
    """The camera's lens at the module's full resolution, as Vantage resolves it (module doc): {'width', 'height',
    'f_px', 'cx', 'cy', 'model', 'k', 'aspect', 'max_half_deg', 'fps', 'module', 'lens', 'source', 'notes'}."""
    notes = []
    mod = _module(doc, cam.get('module_id'))
    if mod is None:
        first = next(iter(doc['modules']), None)
        mod = _module(doc, first) if first else {'id': '', 'name': 'typical', 'width_px': 1920, 'height_px': 1536,
                                                 'pixel_um': 3.0, 'fps': 30.0}
        notes.append(f'module {cam.get("module_id")!r} is not in the file: {mod["name"]} used (as Vantage does)')
    ln = _lens(doc, cam.get('lens_id'))
    if ln is None:
        first = next(iter(doc['lenses']), None)
        if first is None:
            raise VantageError(f'camera {cam.get("tag")}: the file has no lenses')
        ln = _lens(doc, first)
        notes.append(f'lens {cam.get("lens_id")!r} is not in the file: {ln["name"]} used (as Vantage does)')
    W, H = mod['width_px'], mod['height_px']
    pm = mod['pixel_um'] * 1e-3
    proj = LENS.model_name(ln['projection'])
    kb = list(ln['kb'])
    focal, pp, aspect = ln['focal_mm'], list(ln['pp_offset_px']), ln['aspect']
    source = 'library lens'
    rec = cam.get('calib') if isinstance(cam.get('calib'), dict) else None
    if rec and all(k in rec for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy', 'k')):
        sx, sy = W / _num(rec['width'], W), H / _num(rec['height'], H)
        fx, fy = _num(rec['fx']) * sx, _num(rec['fy']) * sy
        cx, cy = (_num(rec['cx']) + 0.5) * sx, (_num(rec['cy']) + 0.5) * sy
        focal, proj = fx * pm, 'kannala_brandt'
        kb = (_nums(rec['k'], 4, (0.0,) * 4) if len(rec.get('k') or []) >= 4
              else (list(rec.get('k') or []) + [0.0] * 4)[:4])
        pp, aspect = [cx - W / 2, cy - H / 2], fy / fx if fx else 1.0
        source = f'unit calibration {rec.get("serial") or ""}'.strip()
    max_half = ln['max_half_deg']
    if not max_half:
        d = ln['sheet_fov'][2]
        if d:
            max_half = min(d / 2 + 2.0, 95.0)
        else:
            hd = math.hypot(W * pm, H * pm) / 2
            max_half = min(math.degrees(float(LENS.theta_of_r(hd, focal, proj, kb))) + 1.0, 95.0)
    if proj == 'rectilinear':
        max_half = min(max_half, 89.0)
    return {'width': W, 'height': H, 'f_px': focal / pm, 'cx': W / 2 + pp[0], 'cy': H / 2 + pp[1], 'model': proj,
            'k': [float(v) for v in kb] if proj == 'kannala_brandt' else [0.0] * 4, 'aspect': float(aspect),
            'max_half_deg': float(max_half), 'fps': mod['fps'], 'module': mod['id'], 'lens': ln['id'],
            'source': source, 'notes': notes}


def rot_zyx(yaw, pitch, roll) -> np.ndarray:
    """Vantage's body -> vehicle rotation (degrees, pitch positive down)."""
    y, p, r = (math.radians(float(a)) for a in (yaw, pitch, roll))
    Rz = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]])
    return Rz @ Ry @ Rx


def falcon_pose(o: dict, rear_axle_x: float) -> dict:
    """A Vantage pose (rear-axle origin, ISO 8855) in Falcon's rig frame (CARLA's vehicle frame)."""
    pitch, yaw, roll = carla_rotation(rot_zyx(_num(o.get('yaw')), _num(o.get('pitch')), _num(o.get('roll'))))
    return {'x': round(_num(o.get('x')) + rear_axle_x, 6), 'y': round(-_num(o.get('y')) + 0.0, 6),
            'z': round(_num(o.get('z')), 6), 'roll': round(roll, 6), 'pitch': round(pitch, 6), 'yaw': round(yaw, 6)}


def _slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', str(s)).strip('-').lower()[:40] or 'x'


def camera_sensor(doc: dict, cam: dict, rear_axle_x: float, image_scale: float = IMAGE_SCALE) -> tuple[dict, list]:
    """A Falcon camera for a Vantage camera, and notes."""
    L = camera_lens(doc, cam)
    s = float(image_scale)
    w, h = max(int(round(L['width'] * s)), 16), max(int(round(L['height'] * s)), 16)
    sx, sy = w / L['width'], h / L['height']
    tag = str(cam.get('tag') or cam.get('uid'))
    base = {'id': f'cam-{_slug(tag)}', 'name': tag, 'pose': falcon_pose(cam, rear_axle_x), 'width': w, 'height': h,
            'rate_hz': float(L['fps']), 'labels': list(CAMERA_LABELS),
            'vantage': {'uid': cam.get('uid'), 'name': cam.get('name'), 'module': L['module'], 'lens': L['lens'],
                        'lens_source': L['source'], 'full_size': [L['width'], L['height']]}}
    notes = [f'{tag}: {n}' for n in L['notes']]
    centred = abs(L['cx'] - L['width'] / 2) < 0.5 and abs(L['cy'] - L['height'] / 2) < 0.5
    if L['model'] == 'rectilinear' and abs(L['aspect'] - 1.0) < 1e-6 and centred:
        base.update(type='camera.pinhole', fov=math.degrees(2 * math.atan(L['width'] / (2 * L['f_px']))))
        return base, notes
    base.update(type='camera.fisheye',
                lens={'model': L['model'], 'f_px': L['f_px'] * sx, 'cx': L['cx'] * sx, 'cy': L['cy'] * sy,
                      'k': L['k'], 'aspect': L['aspect'] * sy / sx, 'fov': 2.0 * L['max_half_deg']})
    if 2.0 * L['max_half_deg'] < 60.0:
        notes.append(f'{tag}: a {2 * L["max_half_deg"]:.0f}° lens: narrower than Falcon\'s fisheye range (60–270°)')
    return base, notes


def _sensor_model(doc, s):
    kind = s.get('kind') if s.get('kind') in ('lidar', 'radar', 'gnss', 'imu') else 'lidar'
    m = doc['sensor_models'].get(s.get('model_id'))
    if m is not None and m.get('kind', kind) == kind:
        return m, None
    m = next((x for x in doc['sensor_models'].values() if x.get('kind') == kind), None)
    why = (f'model {s.get("model_id")!r} is not in the file: ' + (f'{m.get("name")} used' if m else
                                                                  'Falcon\'s typical figures used'))
    return m or {}, why


def other_sensor(doc: dict, s: dict, rear_axle_x: float) -> tuple[dict | None, list]:
    """A Falcon lidar, radar, GNSS or IMU for a Vantage sensor, and notes."""
    from .rig import new_sensor
    kind = s.get('kind') if s.get('kind') in ('lidar', 'radar', 'gnss', 'imu') else 'lidar'
    m, why = _sensor_model(doc, s)
    tag = str(s.get('tag') or s.get('uid'))
    out = new_sensor(kind, f'{kind}-{_slug(tag)}', tag, falcon_pose(s, rear_axle_x))
    notes = [f'{tag}: {why}'] if why else []
    rate = _num(m.get('rate_hz'))
    if kind == 'lidar':
        beams = [_num(b) for b in m.get('beams') or []]
        ch = len(beams) if beams else int(_num(m.get('channels'), 32))
        if ch > 256:
            notes.append(f'{tag}: {ch} channels capped at 256')
        out['channels'] = max(min(ch, 256), 1)
        out['upper_fov'] = max(beams) if beams else _num(m.get('v_max'), 15.0)
        out['lower_fov'] = min(beams) if beams else _num(m.get('v_min'), -25.0)
        out['horizontal_fov'] = _num(m.get('h_fov'), 360.0) or 360.0
        out['rotation_frequency'] = rate or 10.0
        out['range'] = _num(m.get('range_max_m')) or _num(m.get('range_m'), 100.0) or 100.0
        pps = _num(m.get('points_per_s'))
        if not pps and _num(m.get('h_res')) > 0:
            pps = out['channels'] * out['horizontal_fov'] / _num(m.get('h_res')) * out['rotation_frequency']
        out['points_per_second'] = int(min(pps or out['points_per_second'], 5_000_000))
        if beams:
            notes.append(f'{tag}: its measured beam elevations become CARLA\'s even spread from '
                         f'{out["lower_fov"]:g}° to {out["upper_fov"]:g}°')
    elif kind == 'radar':
        zones = [z for z in m.get('zones') or [] if isinstance(z, (list, tuple)) and len(z) >= 3]
        if zones:
            out['range'] = max(_num(z[0]) for z in zones)
            out['horizontal_fov'] = min(2 * max(_num(z[1]) for z in zones), 180.0)
            out['vertical_fov'] = min(2 * max(_num(z[2]) for z in zones), 90.0)
            if len(zones) > 1:
                notes.append(f'{tag}: {len(zones)} zones become one cone (the widest field, the longest range)')
        out['rate_hz'] = rate or out['rate_hz']
    else:
        out['rate_hz'] = rate or out['rate_hz']
    out['vantage'] = {'uid': s.get('uid'), 'name': s.get('name'), 'model': s.get('model_id')}
    return out, notes


# =========================================================================================== seams
def _body_box(doc, vehicle):
    """(rear_x, front_x, half_width, height) of the body in Vantage's frame: the mesh's, else the CARLA car's placed
    on the rear axle."""
    b = doc.get('body')
    if b:
        return b['rear_x'], b['front_x'], b['width'] / 2, b['height'] or 1.5
    c = vehicle.get('carla') or {'length': 4.5, 'width': 1.9, 'height': 1.5}
    ra = -vehicle['rear_axle_x']                     # the centre's x from the rear axle
    return ra - c['length'] / 2, ra + c['length'] / 2, c['width'] * 0.46, c['height']


def body_centre_x(doc: dict, vehicle: dict) -> float:
    """The x (Vantage frame) of the body centre the seams turn about."""
    r, f, _, _ = _body_box(doc, vehicle)
    return (r + f) / 2


def _density(intr: LENS.Intrinsics, theta, rng):
    """Vantage's pixel density (px/cm) on a target facing the camera: sqrt(radial × tangential sampling)."""
    th = np.maximum(theta, 1e-4)
    radial = np.maximum(LENS.dr_dtheta(th, intr.f_px, intr.model, intr.k), 1e-6)
    tang = np.maximum(LENS.r_of_theta(th, intr.f_px, intr.model, intr.k), 1e-9) / np.sin(np.minimum(th, math.pi - 1e-3))
    return np.sqrt(radial * tang * intr.aspect) / (np.maximum(rng, 1e-3) * 100.0)


def _seg_hits_box(P0, P1, lo, hi):
    """Whether each segment P0 -> P1[k] passes through the box [lo, hi] (slab test, ends excluded by 2 %)."""
    d = P1 - P0
    t0 = np.full(len(P1), 0.02)
    t1 = np.full(len(P1), 0.98)
    for a in range(3):
        da = d[:, a]
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = (lo[a] - P0[a]) / da
            tb = (hi[a] - P0[a]) / da
        tmin, tmax = np.minimum(ta, tb), np.maximum(ta, tb)
        par = np.abs(da) < 1e-12
        inside = (P0[a] >= lo[a]) & (P0[a] <= hi[a])
        tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
        tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
        t0, t1 = np.maximum(t0, tmin), np.minimum(t1, tmax)
    return t0 < t1


def _ground(doc, vehicle, cell):
    r, f, hw, _ = _body_box(doc, vehicle)
    roi = doc.get('roi') or {}
    dist = float(roi.get('distance') or SEAM_ROI_M)
    pad = dist + 0.5
    x0, x1, y0, y1 = r - pad, f + pad, -hw - pad, hw + pad
    xs = np.arange(x0 + cell / 2, x1, cell)
    ys = np.arange(y0 + cell / 2, y1, cell)
    X, Y = np.meshgrid(xs, ys)
    dx = np.maximum(np.maximum(r - X, X - f), 0.0)
    dy = np.maximum(np.abs(Y) - hw, 0.0)
    body = (dx == 0) & (dy == 0)
    if roi.get('mode') == 'polygon' and len(roi.get('polygon') or []) >= 3:
        poly = np.asarray(roi['polygon'], float)[:, :2]
        inside = np.zeros(X.shape, bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            cross = ((yi > Y) != (yj > Y)) & (X < (xj - xi) * (Y - yi) / np.where(yj != yi, yj - yi, 1e-12) + xi)
            inside ^= cross
            j = i
        roi_m = inside & ~body
    else:
        roi_m = (np.hypot(dx, dy) <= dist) & ~body
    return X, Y, body, roi_m, (r, f, hw)


def estimate_seams(doc: dict, variant: dict, vehicle: dict, cell: float = SEAM_CELL_M) -> dict:
    """The variant's seams (module doc): {'centre_x' (Vantage frame), 'seams': [{'a', 'b' (tags), 'a_uid', 'b_uid',
    'angle_deg' (azimuth about the body centre, counter-clockwise from ahead, degrees in (−180, 180]), 'source'
    ('vantage' or 'estimated'), 'density' (px/cm along it)}] in counter-clockwise order, 'unused' (seams set in Vantage
    between cameras that are not neighbours in this order)}."""
    cams = [c for c in variant['cameras'] if c.get('visible', True) is not False]
    X, Y, body, roi, (r, f, hw) = _ground(doc, vehicle, cell)
    cxv = (r + f) / 2
    out = {'centre_x': cxv, 'cell_m': cell, 'seams': [], 'unused': []}
    if len(cams) < 2:
        return out
    height = _body_box(doc, vehicle)[3]
    lo, hi = np.array([r, -hw, 0.2]), np.array([f, hw, height])
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    V, D = [], []
    for c in cams:
        L = camera_lens(doc, c)
        intr = LENS.Intrinsics(L['width'], L['height'], L['f_px'], L['model'], tuple(L['k']), L['cx'], L['cy'],
                               math.radians(L['max_half_deg']), L['aspect'])
        pos = np.array([_num(c.get('x')), _num(c.get('y')), _num(c.get('z'))])
        W2O = BODY_TO_OPTICAL @ rot_zyx(_num(c.get('yaw')), _num(c.get('pitch')), _num(c.get('roll'))).T
        d = P - pos
        _, _, theta, ok = intr.project_dirs(d @ W2O.T)
        ok &= ~_seg_hits_box(pos, P, lo, hi)
        dens = np.where(ok, _density(intr, theta, np.linalg.norm(d, axis=1)), 0.0)
        V.append(ok.reshape(X.shape))
        D.append(dens.reshape(X.shape))
    V, D = np.stack(V), np.stack(D)
    az = np.arctan2(Y, X - cxv)
    rad = np.hypot(Y, X - cxv)
    best = np.argmax(D, 0)
    cover = V.any(0)
    azs = []
    for i, c in enumerate(cams):
        m = roi & cover & (best == i)
        if m.sum() > 4:
            w = D[i][m]
            azs.append(math.atan2(float((np.sin(az[m]) * w).sum()), float((np.cos(az[m]) * w).sum())))
        else:
            azs.append(math.atan2(_num(c.get('y')), _num(c.get('x')) - cxv))
    order = list(np.argsort(azs))
    rmax = float(rad[roi].max()) if roi.any() else 5.0
    radii = np.arange(SEAM_MIN_RADIUS_M, rmax + cell, cell)
    x0, y0 = float(X[0, 0] - cell / 2), float(Y[0, 0] - cell / 2)
    ny, nx = X.shape

    def sample(ang):
        Xs = cxv + np.cos(ang)[..., None] * radii
        Ys = np.sin(ang)[..., None] * radii
        ii = ((Xs - x0) / cell).astype(int)
        jj = ((Ys - y0) / cell).astype(int)
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        return np.clip(ii, 0, nx - 1), np.clip(jj, 0, ny - 1), ok
    used = set()
    n = len(cams)
    for k in range(n):
        ia, ib = order[k], order[(k + 1) % n]
        a0 = azs[ia]
        span = (azs[ib] - a0) % (2 * math.pi)
        if span < 1e-3:
            span = 2 * math.pi / n
        ua, ub = str(cams[ia].get('uid')), str(cams[ib].get('uid'))
        key = f'{ua}|{ub}'
        if key in variant['seams']:
            ang, src = math.radians(variant['seams'][key]), 'vantage'
            used.add(key)
        else:
            cand = a0 + span * np.linspace(0.08, 0.92, 41)
            ii, jj, ok = sample(cand)
            inroi = ok & roi[jj, ii]
            both = V[ia][jj, ii] & V[ib][jj, ii] & inroi
            dmin = np.minimum(D[ia][jj, ii], D[ib][jj, ii])
            frac = both.sum(1) / np.maximum(inroi.sum(1), 1)
            dsc = np.where(both, np.clip(dmin, 0, 10), 0).sum(1) / np.maximum(inroi.sum(1), 1) / 10
            score = frac + 0.25 * dsc - 0.02 * np.abs(np.linspace(-1, 1, len(cand)))
            ang, src = float(cand[int(np.argmax(score))]), 'estimated'
        ii, jj, ok = sample(np.array(ang))
        inr = ok & roi[jj, ii]
        da, db = D[ia][jj, ii], D[ib][jj, ii]
        dd = np.where(V[ia][jj, ii] & V[ib][jj, ii], np.minimum(da, db), np.maximum(da, db))
        out['seams'].append({'a': str(cams[ia].get('tag')), 'b': str(cams[ib].get('tag')), 'a_uid': ua, 'b_uid': ub,
                             'angle_deg': round(math.degrees(math.atan2(math.sin(ang), math.cos(ang))), 3),
                             'source': src, 'density': round(float(np.median(dd[inr])), 3) if inr.any() else 0.0})
    tags = {str(c.get('uid')): str(c.get('tag')) for c in cams}
    for key, a in variant['seams'].items():
        if key not in used:
            ua, _, ub = key.partition('|')
            out['unused'].append({'key': key, 'a': tags.get(ua, ua), 'b': tags.get(ub, ub), 'angle_deg': a})
    return out


# =========================================================================================== variants as rigs
def find_variant(doc: dict, ident: str) -> dict:
    """A variant by uid, by name (case ignored) or by its 1-based number."""
    for v in doc['variants']:
        if v['uid'] == ident:
            return v
    named = [v for v in doc['variants'] if v['name'].lower() == str(ident).lower()]
    if len(named) == 1:
        return named[0]
    if str(ident).isdigit() and 1 <= int(ident) <= len(doc['variants']):
        return doc['variants'][int(ident) - 1]
    raise VantageError(f'no variant {ident!r} (variants: {", ".join(repr(v["name"]) for v in doc["variants"])})')


def variant_rig(doc: dict, variant: dict, vehicle: dict, image_scale: float = IMAGE_SCALE, rig_id: str = '',
                seams: dict | None = None) -> dict:
    """A Falcon rig (the project's rig object) for one variant, on the given vehicle (match_vehicle), with a 'vantage'
    block saying where it came from: the file and its SHA-256, the variant, the seams (Falcon's frame and Vantage's),
    the body centre, the vehicle match and every note."""
    rax = float(vehicle['rear_axle_x'])
    sensors, notes, ids = [], list(doc.get('notes') or []), set()
    for c in variant['cameras']:
        if c.get('visible', True) is False:
            notes.append(f'{c.get("tag")}: hidden in Vantage, left out')
            continue
        s, n = camera_sensor(doc, c, rax, image_scale)
        notes += n
        sensors.append(s)
    for x in variant['sensors']:
        if x.get('visible', True) is False:
            continue
        s, n = other_sensor(doc, x, rax)
        notes += n
        sensors.append(s)
    for s in sensors:                                 # ids unique within the rig
        base, k = s['id'], 2
        while s['id'] in ids:
            s['id'], k = f'{base}-{k}', k + 1
        ids.add(s['id'])
    sm = seams if seams is not None else estimate_seams(doc, variant, vehicle)
    cxv = sm['centre_x']
    if vehicle.get('size_error') is not None and vehicle['size_error'] > 0.08:
        notes.append(f'{vehicle["blueprint"]} differs from the project\'s body by {vehicle["size_error"] * 100:.0f} % '
                     'on average: cameras may sit inside or off the CARLA car\'s body')
    if vehicle.get('note'):
        notes.append(vehicle['note'])
    rid = rig_id or f'vtg-{_slug(doc["name"])[:20]}-{_slug(variant["name"])[:20]}'
    return {'id': _slug(rid)[:64], 'name': f'{doc["name"]} · {variant["name"]}', 'vehicle': vehicle['blueprint'],
            'sensors': sensors,
            'vantage': {'file': doc['path'], 'sha256': doc['sha256'], 'project': doc['name'],
                        'variant': variant['name'], 'variant_uid': variant['uid'], 'note': variant['note'],
                        'image_scale': float(image_scale), 'blend_m': variant['blend_m'],
                        'vehicle': {k: vehicle.get(k) for k in ('blueprint', 'how', 'size_error', 'rear_axle_x')},
                        'body': doc.get('body'),
                        'centre': {'vantage_x': round(cxv, 4), 'falcon_x': round(cxv + rax, 4)},
                        'seams': sm['seams'], 'seams_unused': sm['unused'], 'notes': notes}}


def rigs_for(doc: dict, variants=None, vehicle: str = 'auto', image_scale: float = IMAGE_SCALE) -> list:
    """Rigs for the chosen variants (all by default), every one on the same CARLA vehicle."""
    if not 0.05 <= float(image_scale) <= 1.0:
        raise VantageError('the image scale must be in 0.05–1')
    veh = match_vehicle(doc.get('body'), vehicle)
    chosen = [find_variant(doc, v) for v in variants] if variants else list(doc['variants'])
    return [variant_rig(doc, v, veh, image_scale) for v in chosen]


def merge_into(project_rigs: list, new_rigs: list) -> list:
    """The project's rig list with the imported rigs added; a rig of the same file and variant is replaced, other
    clashing ids get a suffix."""
    out = copy.deepcopy(project_rigs)
    for r in new_rigs:
        r = copy.deepcopy(r)
        v = r.get('vantage') or {}
        same = next((i for i, x in enumerate(out) if (x.get('vantage') or {}).get('file') == v.get('file')
                     and (x.get('vantage') or {}).get('variant_uid') == v.get('variant_uid')), None)
        if same is not None:
            r['id'] = out[same]['id']
            out[same] = r
            continue
        ids, base, k = {x['id'] for x in out}, r['id'], 2
        while r['id'] in ids:
            r['id'], k = f'{base[:60]}-{k}', k + 1
        out.append(r)
    return out


def summary(doc: dict, vehicle: str = 'auto', seams: bool = True) -> dict:
    """What the file holds, for `falcon rigs vantage` and the import dialog: per variant its cameras (tag, module,
    lens, pose), sensors and seams."""
    veh = match_vehicle(doc.get('body'), vehicle)
    rows = []
    for v in doc['variants']:
        row = {'uid': v['uid'], 'name': v['name'], 'note': v['note'], 'active': v['uid'] == doc['active'],
               'cameras': [{'tag': c.get('tag'), 'module': c.get('module_id'), 'lens': c.get('lens_id'),
                            'calibrated': bool(c.get('calib')), 'visible': c.get('visible', True) is not False,
                            **{k: _num(c.get(k)) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')}}
                           for c in v['cameras']],
               'sensors': [{'tag': s.get('tag'), 'kind': s.get('kind'), 'model': s.get('model_id')}
                           for s in v['sensors']],
               'seams_set': len(v['seams'])}
        if seams:
            sm = estimate_seams(doc, v, veh)
            row['seams'] = [{k: s[k] for k in ('a', 'b', 'angle_deg', 'source')} for s in sm['seams']]
        rows.append(row)
    return {'name': doc['name'], 'file': doc['path'], 'sha256': doc['sha256'], 'vehicle': veh, 'body': doc['body'],
            'variants': rows, 'notes': doc['notes'],
            'read': {k: list(v) for k, v in READ_FIELDS.items()}}


def sector_of(az_deg: float) -> str:
    """The 45° sector of an azimuth (Vantage's: counter-clockwise from ahead)."""
    return SECTORS[int(((float(az_deg) + 22.5) % 360.0) // 45.0)]
