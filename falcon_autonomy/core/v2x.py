"""Cooperative V2X roadside units (RSUs): the project's infrastructure, their placement at junctions, the part of a
drive config they make, and the measured benefit of their messages to the ego (standard library + numpy).
docs/v2x.md is the guide; core.cpm builds and encodes the messages; workers.rsu spawns and records the stations.

A roadside unit (project "infrastructure" → "rsus", one per pole):

    {"id", "name", "world" (the world it stands in), "x", "y" (CARLA world, metres: the pole's foot), "z" (its ground
     height; absent: the nearest lane's, read on the server), "yaw" (degrees, CARLA's: 0 = +x, positive towards +y),
     "height_m" (the mast head above the foot), "sensors": [rig sensors], "station_id" (ETSI StationId; absent: a
     number from the id), "cpm": {…CPM_DEFAULTS}, "junction": {where a placement helper put it}}

Frames. The station frame is CARLA's vehicle frame put at the pole's foot and turned by yaw: x along the RSU's
heading, y right, z up (left-handed). A sensor's pose is a rig pose (core.rig: x y z metres, roll pitch yaw degrees)
from the MAST HEAD (0, 0, height_m) — so raising the pole raises every sensor; in the recorded dataset (dataset.json,
objects' box3d) poses and boxes are in the station frame (the foot), as an ego's are in its vehicle frame. CPM objects
are in East-North-Up from the station's reference position (the foot): x = CARLA x, y = −CARLA y (CARLA's +y is South).
Only yaw turns the station (a mast stands upright), which keeps every composition exact: a sensor's world rotation is
its own with the station's yaw added.

Sensors are the rig's own definitions (core.rig.with_defaults fills what a hand-written one leaves out). Cameras always
record their ground truth (semantic is added when no label layer is asked): the CPM's visibility is read from the
instance camera. Faults and Falcon's sensor weather physics are not applied to roadside sensors.
"""
from __future__ import annotations

import copy
import json
import math
import os
import zlib

from .rig import new_sensor, validate_rig, with_defaults

KEY = 'infrastructure'
GT_LABELS = ('semantic', 'instance', 'depth', 'boxes2d', 'boxes3d')     # workers.camgt.GT_LABELS
HEIGHT_M = 6.0                       # a typical signal mast arm or camera pole (5–7 m), not a measured standard
LIMITS = {'height_m': (1.0, 30.0), 'pose_m': 10.0}
# the CPM a station sends (docs/v2x.md): rate, visibility thresholds (heuristics, the same for the ego in the measured
# benefit), optional Gaussian noise of the reported state, a camera's declared range, free-space sectors per sensor
CPM_DEFAULTS = {'rate_hz': 10.0, 'min_pixels': 20, 'min_points': 3, 'noise': {'position_m': 0.0, 'velocity_mps': 0.0},
                'camera_range_m': 100.0, 'sectors': 16, 'max_objects': 255, 'free_space': True,
                'epoch': '2025-01-01T00:00:00Z'}
PRESETS = {
    'pole-camera': 'one 1280 × 720, 90° camera at the mast head, tilted down towards the junction',
    'intersection-lidar': 'one 32-channel lidar at the mast head (10° up to 30° down, 360°, 100 m)',
    'camera-lidar': 'both',
}


def preset_sensors(preset: str, pitch_deg: float = -20.0) -> list:
    """The sensors of a preset, posed from the mast head (x forward along the RSU's yaw)."""
    if preset not in PRESETS:
        raise ValueError(f'preset {preset!r}: one of {", ".join(PRESETS)}')
    out = []
    if preset in ('pole-camera', 'camera-lidar'):
        cam = new_sensor('camera.pinhole', 'cam', 'pole camera', {'x': 0.3, 'z': 0.0, 'pitch': round(pitch_deg, 2)})
        cam.update(width=1280, height=720, fov=90.0, labels=['semantic'])
        out.append(cam)
    if preset in ('intersection-lidar', 'camera-lidar'):
        lid = new_sensor('lidar', 'lidar', 'pole lidar', {'z': 0.4})
        lid.update(channels=32, range=100.0, points_per_second=600000, rotation_frequency=10.0, upper_fov=10.0,
                   lower_fov=-30.0, horizontal_fov=360.0)
        out.append(lid)
    return out


def station_id(rsu: dict) -> int:
    """The ETSI StationId: the RSU's own, else CRC-32 of its id (stable across runs and machines)."""
    sid = rsu.get('station_id')
    return int(sid) if sid is not None else zlib.crc32(rsu['id'].encode('utf-8'))


def cpm_settings(rsu: dict) -> dict:
    c = copy.deepcopy(CPM_DEFAULTS)
    own = rsu.get('cpm') or {}
    c.update({k: v for k, v in own.items() if k != 'noise'})
    c['noise'].update(own.get('noise') or {})
    return c


def new_rsu(world_id: str, x: float, y: float, yaw: float = 0.0, preset: str = 'camera-lidar',
            height_m: float = HEIGHT_M, name: str = '', ident: str = '', pitch_deg: float = -20.0,
            z: float | None = None) -> dict:
    from .project import new_id
    r = {'id': ident or new_id('rsu'), 'name': name or 'Roadside unit', 'world': world_id, 'x': round(float(x), 3),
         'y': round(float(y), 3), 'yaw': round(float(yaw), 3), 'height_m': float(height_m),
         'sensors': preset_sensors(preset, pitch_deg)}
    if z is not None:
        r['z'] = round(float(z), 3)
    return r


def rsus(doc: dict, world_id: str | None = None) -> list:
    """The project's RSUs (in one world when world_id is given)."""
    out = ((doc or {}).get(KEY) or {}).get('rsus') or []
    return [r for r in out if world_id is None or r.get('world') == world_id]


def sensor_with_defaults(s: dict) -> dict:
    """A roadside sensor as the workers read it: the type's defaults filled in; a camera records its ground truth."""
    s = with_defaults(s)
    if s['type'].startswith('camera.'):
        if not any(k in (s.get('labels') or []) for k in GT_LABELS):
            s['labels'] = list(s.get('labels') or []) + ['semantic']
    s.pop('faults', None)
    return s


def validate_rsu(r: dict, world_ids=None) -> list:
    """Problems beyond the schema: the world, the height, sensors far from the mast head, the rig checks."""
    out = []
    name = r.get('name') or r.get('id')
    if world_ids is not None and r.get('world') not in world_ids:
        out.append(f'{name}: no world {r.get("world")!r} in the project')
    lo, hi = LIMITS['height_m']
    if not lo <= float(r.get('height_m', HEIGHT_M)) <= hi:
        out.append(f'{name}: height {r.get("height_m")} m outside {lo:g}–{hi:g} m')
    if not r.get('sensors'):
        out.append(f'{name}: no sensors')
    for s in r.get('sensors') or []:
        if s.get('type') not in ('camera.pinhole', 'camera.fisheye', 'lidar'):
            out.append(f'{name}: {s.get("name")} ({s.get("type")}) — a roadside unit records cameras and lidars')
    rig_like = {'sensors': [dict(s, pose=dict(s.get('pose') or {}, z=0.0)) for s in r.get('sensors') or []]}
    out += [f'{name}: {p}' for p in validate_rig(rig_like)]
    return out


def validate_all(doc: dict) -> list:
    out = []
    ids = [r.get('id') for r in rsus(doc)]
    for d in sorted({i for i in ids if ids.count(i) > 1}):
        out.append(f'roadside unit id {d!r} is used more than once')
    wids = {w['id'] for w in doc.get('worlds') or []}
    for r in rsus(doc):
        out += validate_rsu(r, wids)
    return out


# --------------------------------------------------------------------------------------------- frames
def mount_pose(rsu: dict, s: dict) -> dict:
    """A sensor's pose in the station frame (the pole's foot): its pose from the mast head, raised by height_m."""
    p = dict(s['pose'])
    p['z'] = float(p.get('z', 0.0)) + float(rsu.get('height_m', HEIGHT_M))
    return p


def station_xyz(rsu: dict, ground_z: float = 0.0) -> tuple:
    return float(rsu['x']), float(rsu['y']), float(rsu['z']) if rsu.get('z') is not None else float(ground_z)


def to_world(rsu: dict, p: dict, ground_z: float = 0.0) -> dict:
    """A station-frame pose (x y z roll pitch yaw) in CARLA's world: exact, the station turning only by yaw."""
    x0, y0, z0 = station_xyz(rsu, ground_z)
    a = math.radians(float(rsu.get('yaw', 0.0)))
    c, s_ = math.cos(a), math.sin(a)
    return {'x': x0 + c * p['x'] - s_ * p['y'], 'y': y0 + s_ * p['x'] + c * p['y'], 'z': z0 + p['z'],
            'roll': p.get('roll', 0.0), 'pitch': p.get('pitch', 0.0), 'yaw': float(rsu.get('yaw', 0.0)) + p.get('yaw', 0.0)}


def to_enu(rsu: dict, xyz_world, ground_z: float = 0.0):
    """CARLA world points (N, 3) in the station's East-North-Up frame from its foot (numpy)."""
    import numpy as np
    x0, y0, z0 = station_xyz(rsu, ground_z)
    P = np.asarray(xyz_world, float).reshape(-1, 3)
    return np.c_[P[:, 0] - x0, -(P[:, 1] - y0), P[:, 2] - z0]


# --------------------------------------------------------------------------------------------- junctions
def junctions(xodr_text: str, world_dir: str = '', proj_centre=None) -> list:
    """The junctions of an OpenDRIVE world as placement candidates: id, centre (CARLA x, y), radius (the farthest
    point of its connecting roads), the approaches (incoming roads, bearing from the centre in CARLA degrees), whether
    a signal stands on them, the connecting roads; named after the OSM ways meeting there when the world's area.osm is
    in world_dir (proj_centre: the import's centre, lat lon)."""
    import numpy as np
    from . import xodr as X
    od = X.parse(xodr_text)
    by_j: dict = {}
    for r in od.roads:
        if r.in_junction:
            by_j.setdefault(r.junction, []).append(r)
    names = _osm_junction_names(world_dir, proj_centre) if world_dir else []
    out = []
    for j in od.junctions:
        conn = by_j.get(j.id) or []
        pts = [X.sample_reference_line(r, 2.0)[:, :2] for r in conn]
        pts = [p for p in pts if len(p)]
        if not pts:
            continue
        P = np.concatenate(pts)
        c = P.mean(0)
        radius = float(np.hypot(*(P - c).T).max())
        incoming = sorted({cn.incoming for cn in j.connections})
        app, signals = [], 0
        for rid in incoming:
            road = od.road(rid)
            if road is None:
                continue
            signals += road.signals
            q = X.sample_reference_line(road, 2.0)[:, :2]
            if not len(q):
                continue
            k = int(np.argmin(np.hypot(*(q - c).T)))
            far = q[min(k + 5, len(q) - 1)] if k < len(q) // 2 else q[max(k - 5, 0)]
            app.append({'road': rid, 'bearing_deg': round(-math.degrees(math.atan2(far[1] - c[1], far[0] - c[0])), 1)})
        signals += sum(r.signals for r in conn)
        item = {'id': j.id, 'x': round(float(c[0]), 2), 'y': round(float(-c[1]), 2), 'radius_m': round(radius, 1),
                'approaches': app, 'signalised': signals > 0, 'connecting_roads': len(conn), 'name': '',
                'xodr_name': j.name or ''}
        if names:
            d = [math.hypot(n['x'] - c[0], n['y'] - c[1]) for n in names]
            k = int(np.argmin(d))
            if d[k] <= radius + 8.0 and names[k]['name']:
                item['name'] = names[k]['name']
        out.append(item)
    out.sort(key=lambda j: (not j['signalised'], -len(j['approaches']), j['id']))
    return out


def _osm_junction_names(world_dir: str, centre) -> list:
    """[{x, y (OpenDRIVE frame), name: 'A × B'}] of the OSM junction nodes in the world's area.osm."""
    path = os.path.join(world_dir, 'area.osm')
    if not centre or not os.path.isfile(path):
        return []
    try:
        from . import mapedit as M
        from . import osm as O
        data = O.read_osm(path)
        g = M.geometry(data, O.local_projection(float(centre[0]), float(centre[1])))
    except (OSError, ValueError, KeyError):
        return []
    way_name = {w['id']: w['name'] for w in g['ways']}
    out = []
    for j in g['junctions']:
        nm = sorted({way_name.get(w, '') for w in j.get('ways') or []} - {''})
        out.append({'x': j['xy'][0], 'y': j['xy'][1], 'name': ' × '.join(nm[:3])})
    return out


def corners(j: dict, xodr_text: str = '', lanes=None, setback_m: float = 4.0) -> list:
    """Pole positions at a junction's corners, between adjacent approaches: on each bisector, the nearest point from
    radius + setback_m outwards (1 m steps, up to 15 m more) whose distance to the nearest driving lane's edge is at
    least 0.5 m (core.cityprops.Lanes; a heuristic for "off the carriageway"), facing the centre. [{x, y, yaw,
    distance_m, clearance_m, gap_deg}] widest angular gap first."""
    if lanes is None and xodr_text:
        from .cityprops import Lanes
        lanes = Lanes(xodr_text)
    b = sorted(a['bearing_deg'] % 360.0 for a in j.get('approaches') or [])
    if len(b) < 2:
        b = [0.0, 180.0] if not b else [b[0], b[0] + 180.0]
    out = []
    for k, a0 in enumerate(b):
        a1 = b[(k + 1) % len(b)] + (360.0 if k == len(b) - 1 else 0.0)
        gap = a1 - a0
        mid = math.radians(a0 + gap / 2.0)
        best = None
        for extra in range(0, 16):
            d = j['radius_m'] + setback_m + extra
            x, y = j['x'] + d * math.cos(mid), j['y'] + d * math.sin(mid)
            clear = math.inf
            if lanes is not None:
                _, clear = lanes.nearest('driving', x, -y, radius=30.0)
            if clear >= 0.5:
                best = (x, y, d, clear)
                break
        if best is None:
            continue
        x, y, d, clear = best
        out.append({'x': round(x, 2), 'y': round(y, 2), 'yaw': round(math.degrees(math.atan2(j['y'] - y, j['x'] - x)), 2),
                    'distance_m': round(d, 2), 'clearance_m': round(clear, 2) if math.isfinite(clear) else None,
                    'gap_deg': round(gap, 1)})
    out.sort(key=lambda c: -c['gap_deg'])
    return out


def place_at_junction(world_id: str, j: dict, xodr_text: str = '', preset: str = 'camera-lidar',
                      height_m: float = HEIGHT_M, corner: int = 0, name: str = '', lanes=None) -> dict:
    """An RSU at one of a junction's corners (corners()), its camera tilted to look 10 m past the centre."""
    cs = corners(j, xodr_text, lanes)
    if not cs:
        raise ValueError(f'junction {j["id"]}: no corner clear of the driving lanes')
    c = cs[min(int(corner), len(cs) - 1)]
    pitch = -math.degrees(math.atan2(height_m, c['distance_m'] + 10.0))
    r = new_rsu(world_id, c['x'], c['y'], c['yaw'], preset, height_m,
                name or f'RSU {j.get("name") or "junction " + str(j["id"])}', pitch_deg=max(min(pitch, -8.0), -45.0))
    r['junction'] = {'id': str(j['id']), 'name': j.get('name', ''), 'signalised': bool(j.get('signalised')),
                     'corner': cs.index(c), 'distance_m': c['distance_m'], 'clearance_m': c['clearance_m'],
                     'method': 'xodr junction centre, a corner between adjacent approaches clear of the driving lanes '
                               '(heuristic)'}
    return r


def world_xodr(project, world: dict) -> tuple[str, str]:
    """(xodr text, world folder) of an imported world, or raises ValueError."""
    if world.get('kind') == 'town' or not world.get('xodr'):
        raise ValueError(f'{world.get("name")}: junction placement needs an imported (OpenDRIVE) world; place an RSU '
                         'by position in a town')
    path = os.path.join(project.workspace, world['xodr'])
    with open(path, encoding='utf-8') as fh:
        return fh.read(), os.path.dirname(path)


# --------------------------------------------------------------------------------------------- drive configs
def attach(cfg: dict, project, world_id: str):
    """Put the world's RSUs into a drive config (cfg['infrastructure']), each sensor with its defaults; absent when
    the world has none, so other configs' cache keys are unchanged."""
    rs = rsus(getattr(project, 'doc', project), world_id)
    if not rs:
        return
    cfg[KEY] = {'rsus': [dict(copy.deepcopy(r), sensors=[sensor_with_defaults(s) for s in r['sensors']],
                              station_id=station_id(r), cpm=cpm_settings(r)) for r in rs]}


def start_near(cfg: dict, rsu: dict, min_m: float = 50.0, max_m: float = 150.0, traffic_within_m: float = 200.0):
    """Ask the drive worker to start the ego on a spawn point min_m–max_m from the RSU heading towards it, and to place
    the traffic on the spawn points within traffic_within_m first (seeded order kept among them)."""
    cfg['start_near'] = {'x': float(rsu['x']), 'y': float(rsu['y']), 'min_m': float(min_m), 'max_m': float(max_m),
                         'traffic_within_m': float(traffic_within_m), 'rsu': rsu['id']}


def spawn_order(spawns_xy_yaw: list, order: list, near: dict) -> tuple[list, int]:
    """The drive's seeded spawn order with the start_near candidates first: the ego's (min_m–max_m from the point,
    heading within 60° of it), then the traffic's (within traffic_within_m), then the rest. Returns (order, the number
    of ego candidates)."""
    px, py = near['x'], near['y']
    ego, close, rest = [], [], []
    for k in order:
        x, y, yaw = spawns_xy_yaw[k]
        d = math.hypot(px - x, py - y)
        to = math.degrees(math.atan2(py - y, px - x))
        off = abs((to - yaw + 180.0) % 360.0 - 180.0)
        if near['min_m'] <= d <= near['max_m'] and off <= 60.0:
            ego.append(k)
        elif d <= near.get('traffic_within_m', 0.0):
            close.append(k)
        else:
            rest.append(k)
    return ego + close + rest, len(ego)


# --------------------------------------------------------------------------------------------- the benefit, measured
ROAD_USER = ('vehicle.', 'walker.')
LARGE = ('vehicle.bus', 'vehicle.truck', 'vehicle.trailer', 'vehicle.construction')


def _jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def ego_visibility(run_dir: str, min_pixels: int = 20, min_points: int = 3, sensors: str = 'all') -> dict:
    """{step: set of actor ids} the ego's own sensors saw: an ego camera's objects with at least min_pixels pixels,
    an ego lidar's latest sweep's points (at least min_points of an actor). RSU sensors (dataset.json 'station') are
    left out. sensors: 'all', 'camera' or 'lidar' — the ego's cameras or lidars only (a rig without the others)."""
    import numpy as np
    ds = json.load(open(os.path.join(run_dir, 'dataset.json'), encoding='utf-8'))
    rsu_ids = {s['id'] for s in ds.get('sensors') or [] if s.get('station')}
    rsu_ids |= {s['id'] for st in (ds.get(KEY) or {}).get('stations') or [] for s in st.get('sensors') or []}
    ego_id = (ds.get('ego') or {}).get('actor_id')
    out, last_lidar = {}, {}
    for f in _jsonl(os.path.join(run_dir, 'frames.jsonl')):
        seen = set()
        for key, rel in (f.get('files') or {}).items():
            sid, kind = key.rsplit('/', 1)
            if sid in rsu_ids:
                continue
            if kind == 'objects' and sensors in ('all', 'camera'):
                with open(os.path.join(run_dir, rel), encoding='utf-8') as fh:
                    for o in json.load(fh).get('objects') or []:
                        if isinstance(o.get('track_id'), int) and o.get('pixels', 0) >= min_pixels:
                            seen.add(int(o['track_id']))
            elif kind == 'lidar' and sensors in ('all', 'lidar'):
                with open(os.path.join(run_dir, rel)[:-4] + '.labels.bin', 'rb') as fh:
                    lab = fh.read()
                n = len(lab) // 5                       # uint8 tag, then uint32 actor id, per point
                ids = np.frombuffer(lab[n:5 * n], np.uint32)
                u, c = np.unique(ids[ids > 0], return_counts=True)
                last_lidar[sid] = {int(a) for a, k in zip(u, c) if k >= min_points}
        for v in last_lidar.values():
            seen |= v
        seen.discard(ego_id)
        out[int(f['step'])] = seen
    return out


def rsu_visibility(run_dir: str) -> tuple[dict, dict]:
    """({step: set of actor ids in a CPM sent at that step (any station)}, {station: [steps]}) from cpm.jsonl."""
    by_step, steps = {}, {}
    for m in _jsonl(os.path.join(run_dir, 'cpm.jsonl')):
        st = int(m['step'])
        by_step.setdefault(st, set()).update(int(o['actor_id']) for o in m.get('objects') or [])
        steps.setdefault(m['station'], []).append(st)
    return by_step, steps


def benefit(run_dir: str, radius_m: float = 50.0, near_m: float = 60.0, min_pixels: int = 20, min_points: int = 3,
            min_occlusion_s: float = 0.5, ego_sensors: str = 'all') -> dict:
    """The share of ground-truth road users within radius_m of the ego that its own sensors see, against its sensors
    and the RSUs' CPMs together, at the steps a CPM was sent (both known then); also only while the ego is within
    near_m of a station; per kind (vehicles, large vehicles, VRUs). And the time to first detection of road users
    emerging from occlusion: an occlusion episode is a stretch of at least min_occlusion_s (heuristic) during which a
    road user within radius_m is not seen by the ego, ending when the ego sees it; the gain is how much earlier a CPM
    named it within the episode (0 when no CPM did). Road users never seen by the ego while within radius_m but named
    in a CPM are counted apart. Only recorded steps count (record_every: the ego's camera labels exist there).
    Everything from the recorded files (frames.jsonl, objects, lidar labels, actors.jsonl,
    cpm.jsonl): the ego's thresholds are the stations' (min_pixels, min_points)."""
    import numpy as np
    run = json.load(open(os.path.join(run_dir, 'run.json'), encoding='utf-8'))
    ds = json.load(open(os.path.join(run_dir, 'dataset.json'), encoding='utf-8'))
    fps = float(ds.get('fps') or 20.0)
    stations = [(s['x'], s['y']) for s in (ds.get(KEY) or {}).get('stations') or []]
    ego_vis = ego_visibility(run_dir, min_pixels, min_points, ego_sensors)
    cpm_vis, _ = rsu_visibility(run_dir)
    frames = {int(f['step']): f for f in _jsonl(os.path.join(run_dir, 'frames.jsonl'))}
    ego_id = (ds.get('ego') or {}).get('actor_id')
    tally = {k: {'ego': 0, 'union': 0, 'total': 0, 'steps': 0} for k in
             ('all', 'near_station', 'vehicles', 'large_vehicles', 'vru')}
    per_step = []
    hist = {}                 # actor -> [(step, within radius, ego sees, cpm names at this step)]
    union_known = {}          # actor -> the last step a CPM named it
    for row in _jsonl(os.path.join(run_dir, 'actors.jsonl')):
        st = int(row['step'])
        f = frames.get(st)
        if f is None or 'ego' not in f or not f.get('recorded', True):
            continue                                  # the ego's cameras are written on recorded steps only
        ex, ey = f['ego']['x'], f['ego']['y']
        near = any(math.hypot(ex - sx, ey - sy) <= near_m for sx, sy in stations)
        ev = ego_vis.get(st, set())
        cv = cpm_vis.get(st)
        for a in row['actors']:
            if a['id'] == ego_id or not a['type_id'].startswith(ROAD_USER) or a.get('role') == 'hero':
                continue
            d = math.hypot(a['x'] - ex, a['y'] - ey)
            inside = d <= radius_m
            if cv is not None and a['id'] in cv:
                union_known[a['id']] = st
            hist.setdefault(a['id'], []).append((st, inside, a['id'] in ev, cv is not None and a['id'] in cv,
                                                 a.get('category', ''), a['type_id']))
            if not inside or cv is None:
                continue
            kinds = ['all'] + (['near_station'] if near else [])
            cat = a.get('category', '')
            if cat.startswith(('human.', 'vehicle.motorcycle', 'vehicle.bicycle')):
                kinds.append('vru')
            else:
                kinds.append('vehicles')
                if cat.startswith(LARGE):
                    kinds.append('large_vehicles')
            e, u = a['id'] in ev, (a['id'] in ev) or (a['id'] in cv)
            for k in kinds:
                t = tally[k]
                t['total'] += 1
                t['ego'] += int(e)
                t['union'] += int(u)
        if cv is not None:
            tot = [a for a in row['actors'] if a['id'] != ego_id and a['type_id'].startswith(ROAD_USER)
                   and a.get('role') != 'hero' and math.hypot(a['x'] - ex, a['y'] - ey) <= radius_m]
            if tot:
                e = sum(a['id'] in ev for a in tot)
                u = sum((a['id'] in ev) or (a['id'] in cv) for a in tot)
                per_step.append((len(tot), e, u, near))
            for k in tally:
                tally[k]['steps'] += 1
    shares = {}
    for k, t in tally.items():
        shares[k] = {'road_user_steps': t['total'], 'ego_share': round(t['ego'] / t['total'], 4) if t['total'] else None,
                     'ego_or_rsu_share': round(t['union'] / t['total'], 4) if t['total'] else None,
                     'gain_points': round(100.0 * (t['union'] - t['ego']) / t['total'], 2) if t['total'] else None}
    # occlusion episodes
    min_steps = max(int(round(min_occlusion_s * fps)), 1)
    episodes, never = [], []
    for aid, h in hist.items():
        h.sort()
        run_start = None
        cpm_first = None
        for st, inside, ego_sees, cpm_names, cat, tid in h:
            if inside and not ego_sees:
                if run_start is None:
                    run_start, cpm_first = st, None
                if cpm_names and cpm_first is None:
                    cpm_first = st
            elif inside and ego_sees:
                if run_start is not None and st - run_start >= min_steps:
                    episodes.append({'actor': aid, 'category': cat, 'start_step': run_start, 'ego_step': st,
                                     'cpm_step': cpm_first, 'occluded_s': round((st - run_start) / fps, 3),
                                     'gain_s': round((st - cpm_first) / fps, 3) if cpm_first is not None else 0.0})
                run_start = None
            else:
                run_start = None
        within = [x for x in h if x[1]]
        if within and not any(x[2] for x in within):
            never.append({'actor': aid, 'category': h[0][4], 'cpm_named': any(x[3] for x in within),
                          'within_s': round(len(within) / fps, 2)})
    gains = np.array([e['gain_s'] for e in episodes], float)
    covered = gains[gains > 0]
    ttfd = {'episodes': len(episodes), 'cpm_first': int((gains > 0).sum()),
            'gain_s_mean': round(float(gains.mean()), 3) if len(gains) else None,
            'gain_s_median_when_earlier': round(float(np.median(covered)), 3) if len(covered) else None,
            'gain_s_max': round(float(gains.max()), 3) if len(gains) else None,
            'never_seen_by_ego': len(never), 'never_seen_by_ego_but_in_a_cpm': sum(1 for n in never if n['cpm_named'])}
    ev = {'camera_only': 0, 'lidar_only': 0, 'both': 0}
    kinds = {}
    for st in (ds.get(KEY) or {}).get('stations') or []:
        for x in st.get('sensors') or []:
            kinds[(st['id'], str(x['cpm_sensor_id']))] = 'lidar' if x['type'] == 'lidar' else 'camera'
    sizes = []
    for m in _jsonl(os.path.join(run_dir, 'cpm.jsonl')):
        sizes.append(m.get('bytes', 0))
        for o in m.get('objects') or []:
            ks = {kinds.get((m['station'], k)) for k in o.get('evidence') or {}}
            ev['both' if ks >= {'camera', 'lidar'} else 'camera_only' if 'camera' in ks else 'lidar_only'] += 1
    return {'run': os.path.basename(os.path.normpath(run_dir)), 'seed': run.get('provenance', {}).get('seed'),
            'cpm': {'messages': len(sizes), 'bytes_mean': round(sum(sizes) / len(sizes), 1) if sizes else None,
                    'bytes_max': max(sizes) if sizes else None, 'objects_by_evidence': ev},
            'radius_m': radius_m, 'near_m': near_m, 'ego_sensors': ego_sensors, 'min_pixels': min_pixels, 'min_points': min_points,
            'min_occlusion_s': min_occlusion_s, 'evaluated_steps': tally['all']['steps'], 'shares': shares,
            'time_to_first_detection': ttfd, 'episodes': episodes[:500], 'never_seen_by_ego': never[:500],
            'method': 'ground-truth road users (actors.jsonl, the ego left out) within radius_m of the ego at the '
                      'steps a CPM was sent; seen by the ego: an ego camera\'s objects with ≥ min_pixels pixels or '
                      'the ego lidar\'s latest sweep with ≥ min_points points of the actor; by the RSUs: named in a '
                      'CPM of that step (the stations\' own instance pixels and lidar points, the same thresholds)'}


def pooled(results: list) -> dict:
    """Shares pooled over runs (road-user steps summed) and the runs' spread, per kind; TTFD episodes pooled."""
    import numpy as np
    out = {}
    for k in ('all', 'near_station', 'vehicles', 'large_vehicles', 'vru'):
        n = sum(r['shares'][k]['road_user_steps'] for r in results)
        e = sum((r['shares'][k]['ego_share'] or 0) * r['shares'][k]['road_user_steps'] for r in results)
        u = sum((r['shares'][k]['ego_or_rsu_share'] or 0) * r['shares'][k]['road_user_steps'] for r in results)
        per = [r['shares'][k]['gain_points'] for r in results if r['shares'][k]['gain_points'] is not None]
        out[k] = {'road_user_steps': n, 'ego_share': round(e / n, 4) if n else None,
                  'ego_or_rsu_share': round(u / n, 4) if n else None,
                  'gain_points': round(100 * (u - e) / n, 2) if n else None,
                  'gain_points_per_run': {'min': min(per), 'max': max(per)} if per else None}
    g = [e['gain_s'] for r in results for e in r['episodes']]
    ga = np.array(g, float)
    out['time_to_first_detection'] = {
        'episodes': len(g), 'cpm_first': int((ga > 0).sum()) if len(ga) else 0,
        'gain_s_mean': round(float(ga.mean()), 3) if len(ga) else None,
        'gain_s_median_when_earlier': round(float(np.median(ga[ga > 0])), 3) if (ga > 0).any() else None,
        'never_seen_by_ego': sum(r['time_to_first_detection']['never_seen_by_ego'] for r in results),
        'never_seen_by_ego_but_in_a_cpm': sum(r['time_to_first_detection']['never_seen_by_ego_but_in_a_cpm']
                                              for r in results)}
    return out


# --------------------------------------------------------------------------------------------- ROS 2 bags
def mcap_cpm(bag, run_dir: str, t0_ns: int) -> int:
    """A run's CPMs (cpm.jsonl) into a ROS 2 bag (core.mcap_bag.BagWriter), as the stack bridge publishes them:
    /falcon/v2x/<rsu>/cpm (udp_msgs/UdpPacket: a BTP-B header and the UPER CPM) and /falcon/v2x/<rsu>/cpm_json
    (std_msgs/String), stamped with the step's simulation time from t0_ns. Returns the messages written."""
    import base64
    from . import cpm as C
    from . import ros2msg as R
    from .stackbridge import V2X_NS, ros_name
    rows = _jsonl(os.path.join(run_dir, 'cpm.jsonl'))
    added, n = set(), 0
    for line in rows:
        tp = f'{V2X_NS}/{ros_name(line["station"])}'
        if tp not in added:
            bag.add_topic(tp + '/cpm', 'udp_msgs/msg/UdpPacket')
            bag.add_topic(tp + '/cpm_json', 'std_msgs/msg/String')
            added.add(tp)
        t = int(t0_ns) + int(round(float(line['sim_t']) * 1e9))
        bag.write(tp + '/cpm', t, R.udp_packet(C.btp_payload(base64.b64decode(line['uper'])), t))
        bag.write(tp + '/cpm_json', t, {'data': json.dumps({k: line[k] for k in ('step', 'sim_t', 'station', 'station_id',
                                                                                  'cpm', 'objects')},
                                                           separators=(',', ':'))})
        n += 2
    return n


# --------------------------------------------------------------------------------------------- the command line
def main_v2x(argv: list) -> int:
    """falcon v2x junctions|place|list|remove|drive|benefit|decode|converter … (docs/v2x.md)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog='falcon v2x', description='cooperative roadside units (docs/v2x.md)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    j = sub.add_parser('junctions', help='an imported world\'s junctions as placement candidates')
    j.add_argument('project')
    j.add_argument('--world', required=True)
    j.add_argument('--json', action='store_true')
    p = sub.add_parser('place', help='add a roadside unit at a junction (or a position)')
    p.add_argument('project')
    p.add_argument('--world', required=True)
    p.add_argument('--junction', default='', help='a junction id (falcon v2x junctions)')
    p.add_argument('--at', default='', help='X,Y[,YAW] in CARLA\'s world instead of a junction')
    p.add_argument('--corner', type=int, default=0)
    p.add_argument('--preset', default='camera-lidar', choices=sorted(PRESETS))
    p.add_argument('--height', type=float, default=HEIGHT_M)
    p.add_argument('--name', default='')
    li = sub.add_parser('list', help='the project\'s roadside units')
    li.add_argument('project')
    rm = sub.add_parser('remove', help='remove a roadside unit')
    rm.add_argument('project')
    rm.add_argument('rsu')
    dr = sub.add_parser('drive', help='a recorded drive starting near a roadside unit (its world, traffic nearby)')
    dr.add_argument('project')
    dr.add_argument('--rsu', required=True)
    dr.add_argument('--rig', default='')
    dr.add_argument('--seconds', type=float, default=30.0)
    dr.add_argument('--fps', type=float, default=20.0)
    dr.add_argument('--seed', type=int, default=0)
    dr.add_argument('--seeds', default='', help='several drives, one per seed (N,N,…), into OUT/s<seed>, in one queue')
    dr.add_argument('--vehicles', type=int, default=-1, help='traffic vehicles (default: the profile\'s)')
    dr.add_argument('--walkers', type=int, default=-1)
    dr.add_argument('--start-min', type=float, default=50.0)
    dr.add_argument('--start-max', type=float, default=150.0)
    dr.add_argument('--record-every', type=int, default=1, help='write every n-th step (the CPM period: 2 at 20 fps)')
    dr.add_argument('--out', default='')
    dr.add_argument('--fake-carla', default='', help=argparse.SUPPRESS)
    be = sub.add_parser('benefit', help='measured visibility gain of the RSUs\' CPMs in recorded drives')
    be.add_argument('runs', nargs='+')
    be.add_argument('--radius', type=float, default=50.0)
    be.add_argument('--near', type=float, default=60.0)
    be.add_argument('--min-pixels', type=int, default=20)
    be.add_argument('--min-points', type=int, default=3)
    be.add_argument('--min-occlusion', type=float, default=0.5)
    be.add_argument('--ego-sensors', default='all', choices=('all', 'camera', 'lidar'),
                    help='the ego\'s cameras or lidars only (what a rig without the others would see)')
    be.add_argument('--out', default='')
    cv = sub.add_parser('converter', help='print the command of etsi_its_messages\' conversion node decoding an RSU\'s '
                                          'CPMs into etsi_its_cpm_ts_msgs (in the bridge image, or --native)')
    cv.add_argument('rsu')
    cv.add_argument('--native', action='store_true')
    cv.add_argument('--domain', type=int, default=0)
    de = sub.add_parser('decode', help='decode a run\'s CPMs (UPER) and check them against their JSON')
    de.add_argument('run')
    de.add_argument('--show', type=int, default=0, help='print this many decoded messages')
    a = ap.parse_args(argv)
    if a.cmd == 'benefit':
        res = [benefit(r, a.radius, a.near, a.min_pixels, a.min_points, a.min_occlusion, a.ego_sensors) for r in a.runs]
        for r in res:
            s = r['shares']
            t = r['time_to_first_detection']
            print(f'{r["run"]} (seed {r["seed"]}): within {a.radius:g} m, ego {_pct(s["all"]["ego_share"])} → ego ∪ '
                  f'RSU {_pct(s["all"]["ego_or_rsu_share"])} of {s["all"]["road_user_steps"]} road-user steps; near a '
                  f'station {_pct(s["near_station"]["ego_share"])} → {_pct(s["near_station"]["ego_or_rsu_share"])}; '
                  f'occlusion episodes {t["episodes"]}, a CPM first in {t["cpm_first"]}, mean gain '
                  f'{t["gain_s_mean"]} s')
        doc = {'runs': res, 'pooled': pooled(res)}
        if len(res) > 1:
            pl = doc['pooled']
            print(f'pooled: ego {_pct(pl["all"]["ego_share"])} → {_pct(pl["all"]["ego_or_rsu_share"])}; near a station '
                  f'{_pct(pl["near_station"]["ego_share"])} → {_pct(pl["near_station"]["ego_or_rsu_share"])}')
        if a.out:
            from .provenance import write_json
            write_json(a.out, doc)
        return 0
    if a.cmd == 'decode':
        return _decode_run(a.run, a.show)
    if a.cmd == 'converter':
        from .stackbridge import cpm_converter_command, shell
        print(shell(cpm_converter_command(a.rsu, domain_id=a.domain, docker=not a.native)))
        return 0
    from .project import Edit, Project, ProjectError
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
    if a.cmd == 'list':
        for r in rsus(proj.doc):
            print(f'{r["id"]:14s} {r["name"][:40]:40s} {r["world"]:16s} x {r["x"]:9.2f} y {r["y"]:9.2f} yaw '
                  f'{r.get("yaw", 0):7.1f} h {r.get("height_m", HEIGHT_M):4.1f} m  '
                  + ', '.join(f'{s["id"]} ({s["type"]})' for s in r['sensors']))
        return 0
    if a.cmd == 'remove':
        lst = rsus(proj.doc)
        k = next((i for i, r in enumerate(lst) if r['id'] == a.rsu), None)
        if k is None:
            print(f'failed: no roadside unit {a.rsu!r}', file=sys.stderr)
            return 2
        proj.apply(Edit('remove', [KEY, 'rsus'], index=k, text=f'Remove {a.rsu}'))
        proj.save()
        return 0
    if a.cmd == 'drive':
        return _drive(proj, a)
    world = proj.item('worlds', a.world)
    if world is None:
        print(f'failed: no world {a.world!r}', file=sys.stderr)
        return 2
    if a.cmd == 'junctions':
        try:
            text, wdir = world_xodr(proj, world)
        except (ValueError, OSError) as e:
            print(f'failed: {e}', file=sys.stderr)
            return 2
        js = junctions(text, wdir, (world.get('osm') or {}).get('centre'))
        if a.json:
            print(json.dumps(js, indent=1))
        else:
            for x in js:
                print(f'{x["id"]:>6s}  x {x["x"]:9.2f} y {x["y"]:9.2f}  r {x["radius_m"]:5.1f} m  '
                      f'{len(x["approaches"])} approaches  {"signals" if x["signalised"] else "       "}  {x["name"]}')
        return 0
    if a.cmd == 'place':
        try:
            if a.at:
                v = [float(t) for t in a.at.split(',')]
                r = new_rsu(world['id'], v[0], v[1], v[2] if len(v) > 2 else 0.0, a.preset, a.height,
                            a.name or 'Roadside unit')
            else:
                text, wdir = world_xodr(proj, world)
                js = {x['id']: x for x in junctions(text, wdir, (world.get('osm') or {}).get('centre'))}
                if a.junction not in js:
                    print(f'failed: no junction {a.junction!r} (falcon v2x junctions)', file=sys.stderr)
                    return 2
                r = place_at_junction(world['id'], js[a.junction], text, a.preset, a.height, a.corner, a.name)
        except (ValueError, OSError) as e:
            print(f'failed: {e}', file=sys.stderr)
            return 2
        if KEY not in proj.doc:
            proj.apply(Edit('set', [KEY], {'rsus': []}, text='Infrastructure'))
        proj.apply(Edit('insert', [KEY, 'rsus'], r, text=f'Add {r["name"]}'))
        bad = validate_all(proj.doc)
        if bad:
            print('failed: ' + '; '.join(bad), file=sys.stderr)
            return 2
        proj.save()
        print(json.dumps({k: r[k] for k in ('id', 'name', 'x', 'y', 'yaw', 'height_m')}))
        return 0
    return 2


def _pct(v):
    return '—' if v is None else f'{100 * v:.1f} %'


def _drive(proj, a) -> int:
    import sys
    from . import protocol as P
    from .drive import drive_config, run_dir_name
    from .project import ProjectError
    r = next((x for x in rsus(proj.doc) if x['id'] == a.rsu), None)
    if r is None:
        print(f'failed: no roadside unit {a.rsu!r}', file=sys.stderr)
        return 2
    from ..cli import _run_job, _service
    svc, st = _service(proj.path, a.fake_carla)
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()] if a.seeds else [a.seed]
    base = a.out or os.path.join(svc.runs_dir(), run_dir_name(f'v2x-{r["id"]}'))
    code = P.OK
    try:
        for seed in seeds:
            out = os.path.join(base, f's{seed}') if a.seeds else base
            rig = a.rig or proj.doc['rigs'][0]['id']
            cfg = drive_config(proj, r['world'], rig, out, svc.port, svc.tm_port, a.seconds, a.fps, seed,
                               render_budget_mpx=float(st.get('render_budget_mpx')),
                               name=f'V2X {r["name"]} (seed {seed})')
            if a.vehicles >= 0:
                cfg['traffic']['vehicles'] = a.vehicles
            if a.walkers >= 0:
                cfg['traffic']['walkers'] = a.walkers
            start_near(cfg, r, a.start_min, a.start_max)
            if a.record_every > 1:
                cfg['record_every'] = int(a.record_every)
            c = _run_job(svc, svc.submit_drive(cfg, kind='drive'))
            if c == P.OK:
                print(out)
            code = code if code != P.OK else c
        return code
    except (ProjectError, RuntimeError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return P.FAILED
    finally:
        svc.shutdown()


def _decode_run(run_dir: str, show: int = 0) -> int:
    """Decode every CPM of a run's cpm.jsonl from its UPER bytes and compare with the JSON written beside it."""
    import base64
    from . import cpm as C
    rows = _jsonl(os.path.join(run_dir, 'cpm.jsonl'))
    if not rows:
        print(f'failed: {run_dir} has no cpm.jsonl', file=__import__('sys').stderr)
        return 2
    bad = 0
    for k, m in enumerate(rows):
        dec = C.decode(base64.b64decode(m['uper']))
        if dec != m['cpm']:
            bad += 1
        if k < show:
            print(json.dumps(dec, indent=1))
    size = [len(base64.b64decode(m['uper'])) for m in rows]
    print(f'{len(rows)} CPMs, {len(rows) - bad} decoded equal to their JSON; {min(size)}–{max(size)} bytes '
          f'(mean {sum(size) / len(size):.0f})')
    return 0 if bad == 0 else 2
