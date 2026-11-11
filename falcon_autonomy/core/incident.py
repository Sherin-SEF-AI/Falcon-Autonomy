"""A real incident made a scenario: a reconstructed dashcam event written out as a Falcon scenario and an
OpenSCENARIO file, placed on the matched road of a Falcon world, and the variations around it for the rare-case
search (numpy at import; scipy for the lane snapping).

Input. A reconstruction made by `falcon recon from-dashcam` (core.recon.fromdashcam): posed/posed.json gives every
posed frame's time, the vehicle's pose (world ← vehicle, local east-north-up metres about the segment's first GPS
fix: monocular visual odometry held to the GPS, core.recon.vo), the camera's view matrix and intrinsics. Other road
users come from a detections file (core.nearmiss writes one from its detector and tracker; or by hand):

    {"schema": "falcon.dashcam_detections/1",
     "frames": [{"index": video frame n | "t": seconds from the first posed frame,
                 "boxes": [{"box": [x1, y1, x2, y2] (pixels of the posed images), "kind": car | truck | bus | van |
                            motorcycle | bicycle | pedestrian | child | autorickshaw | …, "track": id (optional),
                            "range_m": horizontal distance from the camera (optional: measured, e.g. by
                            core.nearmiss; otherwise the box is cast onto the road plane)}]}]}

Without one the scenario has the ego only.

Steps.
1. The event window [start, end] of the posed frames (default: all).
2. The ego: each pose to latitude/longitude (core.osm.local_projection) and to the world's CARLA frame
   (core.dashcam.to_world), snapped to the world's lanes (core.dashcam_route.snap_to_lanes: a Viterbi pass over the
   lane centres within 8 m and 45°). The ego is put on the matched lane's centre; its offset from it (GPS noise
   dominates it) is kept in incident.json only.
3. Other road users: a box's bottom centre is cast onto the road plane (z = 0 under the camera, the reconstruction's own
   plane) — or, when the box carries a measured range_m, placed at that horizontal distance along the bottom centre's
   bearing (range_point) — then pushed half the kind's typical length further along the ray for a vehicle (the box's
   bottom edge is its near face); boxes beyond 60 m or above the horizon are dropped. Positions are taken in the
   vehicle's frame (the odometry's relative geometry) and attached to the snapped ego pose, so the conflict's geometry
   is the reconstruction's. Boxes without a track id are joined frame to frame by the nearest of the same kind within 3
   m. Tracks are median-filtered over three samples; speed and heading from central differences (the ego's heading for
   an actor moving under 1 m/s).
4. Two scenarios (core.scenarios): scenario.json replays the event — the ego (driver "replay") and every actor on
   its recorded track, kinematically, at the matched start ("start", "route_xy"; workers.director); variations.json
   is a parametric fit of the same event — each actor placed by lane and distance from the ego at t = 0 (an actor
   seen later extrapolated back at its along-road speed), its speed, its largest braking and its lane change or walk
   as timeline actions — whose parameters are uniform ranges centred on the extracted values (SPREAD), for the
   rare-case search (`falcon search … --scenario variations.json`, sim.studio, core.search, core.milsearch), the ego
   on Falcon's lane follower from the same start.
5. incident.xosc: scenario.json in OpenSCENARIO 1.2 (core.openscenario): the start as a WorldPosition, each track as a
   FollowTrajectoryAction along a Polyline of timed WorldPositions.

Privacy. Everything here is derived from the user's footage and stays on the machine: the incident folder's run.json
names its worker "dashcam-incident" and each scenario carries source {"format": "dashcam"}, which
assistant.privacy reads, so the assistant never reads the folder, a run of these scenarios or a study of them. The
folder's default name is a hash, not the clip's. summary() holds only counts, durations and shares.

Limits. Monocular ranging on a flat road: a slope, a pitching car or a box cut by the image's edge moves a road user
along the ray (a 1 px error in the bottom edge at 30 m is about 1 m at a 1,000 px focal length); heights and sizes
are typical values for the kind, not measured; occluded stretches are interpolated; the replayed ego's speed comes
from the odometry, held to the ~1 Hz GPS.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re

import numpy as np

SCHEMA = 'falcon.incident/1'
DETECTIONS_SCHEMA = 'falcon.dashcam_detections/1'
SOURCE = {'format': 'dashcam', 'kind': 'incident'}
MAX_RANGE_M = 60.0
JOIN_M = 3.0
SPREAD = {'speed': 0.25, 'gap_m': 3.0, 'gap_rel': 0.25, 'lateral_m': 0.5, 'time_s': 1.0, 'decel': 0.3, 'walk': 0.3}
WALKERS = ('pedestrian', 'child')


class IncidentError(Exception):
    pass


def _sizes():
    from .scenario_plan import EGO_SIZE, SIZES
    return SIZES, EGO_SIZE


def size_of(kind):
    sizes, ego = _sizes()
    return sizes.get(kind, ego)


# ----------------------------------------------------------------------------------------------- reading
def read_recon(rec_dir: str) -> dict:
    """posed.json's frames as arrays: t, ego (N, 4, 4), view (N, 4, 4), K (N, 4: fx, fy, cx, cy), size, index;
    the geographic origin."""
    p = os.path.join(rec_dir, 'posed', 'posed.json')
    if not os.path.isfile(p):
        raise IncidentError(f'{rec_dir}: no posed/posed.json (not a reconstruction from a dashcam, or not finished)')
    with open(p, encoding='utf-8') as fh:
        doc = json.load(fh)
    fr = doc.get('frames') or []
    if len(fr) < 2:
        raise IncidentError('fewer than two posed frames')
    geo = doc.get('geo') or {}
    if geo.get('lat0') is None:
        raise IncidentError('the reconstruction has no geographic origin (posed.json geo)')
    return {'t': np.array([float(f['t']) for f in fr]), 'ego': np.array([np.reshape(f['ego'], (4, 4)) for f in fr]),
            'view': np.array([np.reshape(f['viewmat'], (4, 4)) for f in fr]),
            'K': np.array([f['K'][:4] for f in fr], float), 'size': [tuple(f.get('size') or (0, 0)) for f in fr],
            'index': [int(f.get('index', i)) for i, f in enumerate(fr)], 'lat0': float(geo['lat0']),
            'lon0': float(geo['lon0'])}


def read_detections(path_or_doc) -> list:
    doc = path_or_doc
    if isinstance(path_or_doc, str):
        with open(path_or_doc, encoding='utf-8') as fh:
            doc = json.load(fh)
    if not isinstance(doc, dict) or doc.get('schema') != DETECTIONS_SCHEMA:
        raise IncidentError(f'a detections file is {DETECTIONS_SCHEMA} (core.incident)')
    return doc.get('frames') or []


# ----------------------------------------------------------------------------------------------- geometry
def ground_point(K, view, u: float, v: float):
    """The road-plane point (e, n) under pixel (u, v), or None above the horizon (the reconstruction's z = 0)."""
    fx, fy, cx, cy = K[:4]
    R, tvec = view[:3, :3], view[:3, 3]
    d = R.T @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    C = -R.T @ tvec
    if d[2] >= -1e-6:
        return None
    lam = -C[2] / d[2]
    p = C + lam * d
    return p[:2], C[:2]


def range_point(K, view, u: float, v: float, range_m: float):
    """The point at horizontal distance range_m from the camera along pixel (u, v)'s bearing (a box whose range was
    measured, e.g. by core.nearmiss's fused monocular depth), and the camera's position: as ground_point returns."""
    fx, fy, cx, cy = K[:4]
    R, tvec = view[:3, :3], view[:3, 3]
    d = R.T @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    C = -R.T @ tvec
    h = d[:2] / max(float(np.hypot(d[0], d[1])), 1e-9)
    return C[:2] + h * float(range_m), C[:2]


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _course(E):
    """Course (degrees from north, clockwise) of world ← vehicle matrices (x forward)."""
    return 90.0 - np.degrees(np.arctan2(E[:, 1, 0], E[:, 0, 0]))


def _medfilt(v):
    v = np.asarray(v, float)
    if len(v) < 3:
        return v
    out = v.copy()
    for i in range(1, len(v) - 1):
        out[i] = np.median(v[i - 1:i + 2])
    return out


def _grad(v, t):
    return np.gradient(v, t) if len(t) > 1 else np.zeros(len(t))


# ----------------------------------------------------------------------------------------------- actors
def actor_points(rec: dict, frames: list, keep) -> list:
    """Every usable box as (frame k, kind, track id or None, (forward, left) in the vehicle's frame)."""
    by_index = {n: k for k, n in enumerate(rec['index'])}
    out = []
    for f in frames:
        k = by_index.get(int(f['index'])) if f.get('index') is not None else None
        if k is None and f.get('t') is not None:
            k = int(np.argmin(np.abs(rec['t'] - float(f['t']))))
            gap = float(np.median(np.diff(rec['t']))) if len(rec['t']) > 1 else 1.0
            if abs(rec['t'][k] - float(f['t'])) > 0.6 * gap:
                k = None
        if k is None or not keep[k]:
            continue
        Einv = np.linalg.inv(rec['ego'][k])
        for b in f.get('boxes') or ():
            x1, y1, x2, y2 = (float(v) for v in b['box'])
            kind = str(b.get('kind') or 'car')
            if b.get('range_m') is not None:          # a measured range (core.nearmiss): along the bottom ray
                p, C = range_point(rec['K'][k], rec['view'][k], (x1 + x2) / 2.0, y2, float(b['range_m']))
            else:
                hit = ground_point(rec['K'][k], rec['view'][k], (x1 + x2) / 2.0, y2)
                if hit is None:
                    continue
                p, C = hit
            ray = p - C
            rng = float(np.hypot(*ray))
            if rng > MAX_RANGE_M or rng < 0.5:
                continue
            if kind not in WALKERS:                # the bottom edge is the near face: to the centre, half a length on
                p = p + ray / rng * size_of(kind)[0] / 2.0
            q = Einv @ np.array([p[0], p[1], 0.0, 1.0])
            out.append((k, kind, None if b.get('track') is None else str(b['track']), (float(q[0]), float(q[1]))))
    return out


def join_tracks(points: list) -> dict:
    """{track id: [(k, kind, (fwd, left))]}: given ids kept; the others joined to the nearest track of the same kind
    seen in the previous frame within JOIN_M."""
    tracks, auto = {}, 0
    last = {}
    for k in sorted({p[0] for p in points}):
        here = [p for p in points if p[0] == k]
        used = set()
        for _, kind, tid, q in here:
            if tid is None:
                best, bd = None, JOIN_M
                for oid, (kk, okind, oq) in last.items():
                    if oid in used or okind != kind or kk == k:
                        continue
                    d = math.hypot(q[0] - oq[0], q[1] - oq[1])
                    if d < bd:
                        best, bd = oid, d
                if best is None:
                    auto += 1
                    best = f'auto{auto}'
                tid = best
            used.add(tid)
            tracks.setdefault(tid, []).append((k, kind, q))
        for tid in used:
            kk, kind, q = tracks[tid][-1]
            last[tid] = (kk, kind, q)
    return tracks


# ----------------------------------------------------------------------------------------------- extraction
def _path_frame(px, py, x, y):
    """(s along the polyline, lateral to its right in CARLA's frame) of point (x, y)."""
    a = np.c_[px[:-1], py[:-1]]
    seg = np.c_[np.diff(px), np.diff(py)]
    L = np.hypot(seg[:, 0], seg[:, 1])
    s0 = np.r_[0.0, np.cumsum(L)]
    p = np.array([x, y])
    u = np.clip(np.einsum('kd,kd->k', p - a, seg) / np.maximum(L ** 2, 1e-12), 0.0, 1.0)
    q = a + seg * u[:, None]
    d = np.hypot(*(p - q).T)
    j = int(np.argmin(d))
    r = seg[j] / max(L[j], 1e-9)
    lat = (p - q[j]) @ np.array([-r[1], r[0]])
    s = s0[j] + u[j] * L[j]
    if (j == 0 and u[j] == 0.0) or (j == len(L) - 1 and u[j] == 1.0):       # before / past the ends: along the end
        s = s0[j] + (p - a[j]) @ r
        lat = (p - a[j]) @ np.array([-r[1], r[0]])
    return float(s), float(lat)


def extract(rec_dir: str, project, world_id: str, start: float | None = None, end: float | None = None,
            detections=None, workspace: str = '') -> dict:
    """The incident document (module doc, steps 1–3): the ego's and every actor's track in the world's frame and in
    the ego's road frame, the match to the world, and what was left out."""
    from .dashcam import to_world
    from .dashcam_route import snap_to_lanes, world_for
    from .osm import local_projection
    rec = read_recon(rec_dir)
    t = rec['t']
    lo = t[0] if start is None else float(start)
    hi = t[-1] if end is None else float(end)
    keep = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if keep.sum() < 2:
        raise IncidentError(f'the window [{lo:g}, {hi:g}] s holds fewer than two posed frames')
    world = world_for(project, world_id, workspace or getattr(project, 'workspace', ''))
    if not len(world['lanes']):
        raise IncidentError(f'world {world_id!r} has no lanes (its OpenDRIVE file is missing)')
    w = project.item('worlds', world_id) or {}
    lht = str((w.get('osm') or {}).get('handedness', 'RHT')).upper() == 'LHT'
    ks = -1.0 if lht else 1.0                                   # kerb side: + right in right-hand traffic
    E = rec['ego'][keep]
    tt = t[keep] - lo
    pr = local_projection(rec['lat0'], rec['lon0'])
    lat, lon = pr.to_geo(E[:, 0, 3], E[:, 1, 3])
    x, y, yaw = to_world(world, lat, lon, _course(E))
    sn = snap_to_lanes(x, y, yaw, world['lanes'])
    sx, sy = sn['x'], sn['y']
    syaw = np.where(np.isfinite(sn['yaw']), sn['yaw'], yaw)
    matched = float(np.mean(sn['matched']))
    if matched == 0.0:
        raise IncidentError(f'the event is not on world {world_id!r}\'s roads (no pose within 8 m and 45° of a lane)')
    ego_v = np.hypot(_grad(E[:, 0, 3], tt), _grad(E[:, 1, 3], tt))
    lane_w = float(np.median(world['lanes'][:, 3])) if world['lanes'].shape[1] > 3 else 3.5
    # other road users
    frames = read_detections(detections) if detections is not None else []
    idx = np.flatnonzero(keep)
    sub = {'t': t, 'ego': rec['ego'], 'view': rec['view'], 'K': rec['K'], 'index': rec['index']}
    pts = actor_points(sub, frames, keep)
    tracks = join_tracks(pts)
    pos = {int(k): j for j, k in enumerate(idx)}
    actors = []
    for n, (tid, rows) in enumerate(sorted(tracks.items(), key=lambda kv: kv[1][0][0])):
        rows = [r for r in rows if r[0] in pos]
        if len(rows) < 2:
            continue
        kinds = [r[1] for r in rows]
        kind = max(set(kinds), key=kinds.count)
        j = np.array([pos[r[0]] for r in rows])
        fwd = _medfilt([r[2][0] for r in rows])
        left = _medfilt([r[2][1] for r in rows])
        yr = np.radians(syaw[j])
        ax = sx[j] + fwd * np.cos(yr) + left * np.sin(yr)          # attached to the snapped ego (CARLA: y right)
        ay = sy[j] + fwd * np.sin(yr) - left * np.cos(yr)
        at = tt[j]
        vx, vy = _grad(ax, at), _grad(ay, at)
        sp = np.hypot(vx, vy)
        ayaw = np.where(sp > 1.0, np.degrees(np.arctan2(vy, vx)), syaw[j])
        actors.append({'id': f'a{n + 1}', 'kind': kind, 'track': {'t': at.round(3).tolist(), 'x': ax.round(3).tolist(),
                                                                  'y': ay.round(3).tolist(),
                                                                  'yaw': np.round(ayaw, 2).tolist()},
                       'speed_mps': sp.round(3).tolist(), 'boxes': len(rows)})
    route = np.c_[sx, sy]
    keep_r = [0]
    for i in range(1, len(route)):
        if np.hypot(*(route[i] - route[keep_r[-1]])) >= 0.5:
            keep_r.append(i)
    route = route[keep_r]
    doc = {'schema': SCHEMA, 'source': dict(SOURCE), 'world': world_id, 'handedness': 'LHT' if lht else 'RHT',
           'kerb_sign': ks, 'lane_width_m': round(lane_w, 3), 'window_s': [round(float(lo), 3), round(float(hi), 3)],
           'duration_s': round(float(hi - lo), 3),
           'ego': {'track': {'t': tt.round(3).tolist(), 'x': np.round(sx, 3).tolist(), 'y': np.round(sy, 3).tolist(),
                             'yaw': np.round(syaw, 2).tolist()},
                   'speed_mps': np.round(ego_v, 3).tolist(),
                   'offset_m': [None if not np.isfinite(o) else round(float(o), 3) for o in sn['offset']]},
           'route_xy': np.round(route, 3).tolist() if len(route) >= 2 else np.round(np.c_[sx, sy], 3).tolist(),
           'match': {'matched_share': round(matched, 4), 'lane_changes': int(sn['lane_changes']),
                     'median_offset_m': None if not np.isfinite(sn['offset']).any() else
                     round(float(np.nanmedian(np.abs(sn['offset']))), 3)},
           'actors': actors,
           'detections': {'frames': len(frames), 'boxes_used': len(pts), 'tracks': len(actors)},
           'method': 'core.incident (module doc)'}
    doc['fit'] = fit(doc)
    return doc


# ----------------------------------------------------------------------------------------------- the parametric fit
def fit(inc: dict) -> dict:
    """Each actor by lane, distance, lateral offset, speed and its largest braking, lane change or walk (module doc,
    step 4): {'actors': [{id, kind, place, heading, speed_kmh, start, actions: [...]}], 'ego_speed_kmh'}."""
    ego = inc['ego']['track']
    px, py = np.asarray(inc['route_xy'], float).T if len(inc['route_xy']) >= 2 else (np.asarray(ego['x']),
                                                                                      np.asarray(ego['y']))
    if len(px) < 2:
        px, py = np.r_[px, px + 1.0], np.r_[py, py]
    te = np.asarray(ego['t'], float)
    se = np.array([_path_frame(px, py, a, b)[0] for a, b in zip(ego['x'], ego['y'])])
    s0 = se[0]
    ks = float(inc.get('kerb_sign', 1.0))
    w = float(inc.get('lane_width_m') or 3.5)
    out = []
    ego_v = np.asarray(inc['ego'].get('speed_mps') or [0.0], float)
    for a in inc['actors']:
        tr = a['track']
        ta = np.asarray(tr['t'], float)
        fr = np.array([_path_frame(px, py, x, y) for x, y in zip(tr['x'], tr['y'])])
        s, lat = fr[:, 0] - s0, fr[:, 1] * ks                     # lat: + towards the kerb
        vs = _grad(s, ta)
        vl = _grad(lat, ta)
        v_along0 = float(np.median(vs[:3]))
        ahead0 = float(s[0] - v_along0 * ta[0])                   # back to t = 0 at its along-road speed
        lat0 = float(np.median(lat[:3]))
        kind = a['kind']
        walker = kind in WALKERS
        across = walker and np.median(np.abs(vl)) > np.median(np.abs(vs)) and np.max(np.abs(vl)) > 0.3
        if abs(lat0) <= w / 2.0:
            lane, lateral = 'ego', lat0
        elif lat0 < -w / 2.0:
            lane, lateral = ('opposite', 0.0) if v_along0 < -1.0 else ('adjacent', lat0 + w)
        elif walker:
            lane, lateral = ('sidewalk', lat0 - (w / 2.0 + 2.0)) if lat0 > w / 2.0 + 1.0 else ('kerb', 0.0)
        else:
            lane, lateral = 'kerb', w / 2.0 + size_of(kind)[1] / 2.0 + 0.2 - lat0
        heading = 'across' if across else ('against' if v_along0 < -1.0 and lane == 'ego' else 'lane')
        speed0 = abs(v_along0)
        row = {'id': a['id'], 'kind': kind, 'place': {'lane': lane, 'ahead_m': round(ahead0, 2),
                                                      'lateral_m': round(float(lateral), 2)},
               'heading': heading, 'speed_kmh': 0.0 if walker else round(speed0 * 3.6, 2),
               'start': 'moving' if speed0 > 0.5 and not walker else 'stopped', 'actions': []}
        if walker:
            sp = np.hypot(vs, vl)
            go = np.flatnonzero(sp > 0.5)
            if len(go):
                row['actions'].append({'do': 'walk', 't_s': round(float(ta[go[0]]), 2),
                                       'speed_mps': round(float(np.median(sp[go])), 2)})
        else:
            av = np.abs(vs)
            acc = _grad(av, ta)
            k = int(np.argmin(acc)) if len(acc) else 0
            if len(av) > 2 and acc[k] < -1.0 and av[:k + 1].max() - av[k:].min() > 2.0:
                b0 = k
                while b0 > 0 and acc[b0 - 1] < -0.5:
                    b0 -= 1
                row['actions'].append({'do': 'speed', 't_s': round(float(ta[b0]), 2),
                                       'kmh': round(float(av[k:].min()) * 3.6, 2),
                                       'accel': round(float(-acc[k]), 2)})
            if np.ptp(lat) > 1.5:
                i0 = int(np.argmax(np.abs(lat - lat[0]) > 0.3))
                near_end = np.abs(lat - lat[-1]) < 0.3
                i1 = int(np.argmax(near_end)) if near_end.any() else len(lat) - 1
                to = 'ego' if abs(lat[-1]) <= w / 2.0 else ('kerb' if lat[-1] > 0 else 'adjacent')
                row['actions'].append({'do': 'lane_change', 't_s': round(float(ta[i0]), 2), 'to': to,
                                       'duration_s': round(max(float(ta[max(i1, i0 + 1)] - ta[i0]), 0.5), 2)})
        out.append(row)
    return {'actors': out, 'ego_speed_kmh': round(float(np.median(ego_v[:3])) * 3.6, 2),
            'method': 'placement at t = 0 by lane from the lateral offset (lane width '
                      f'{w:g} m), its along-road speed, the largest braking (deceleration over 1 m/s², a drop over '
                      '2 m/s), a lateral move over 1.5 m as a lane change, a walker\'s start'}


# ----------------------------------------------------------------------------------------------- scenarios
def _param(value, spread_abs, lo=None, unit='', about='', relative=None):
    v = float(value)
    d = max(float(spread_abs), abs(v) * relative if relative else 0.0)
    a, b = v - d, v + d
    if lo is not None:
        a = max(a, lo)
    return {'dist': 'uniform', 'lo': round(a, 3), 'hi': round(max(b, a), 3), 'unit': unit, 'value': round(v, 3),
            'about': about + f' (extracted {v:.2f})'}


def scenarios(inc: dict, name: str = 'Incident') -> tuple:
    """(replay scenario, parametric scenario with its search space): module doc, step 4."""
    from . import scenarios as SC
    f = inc['fit']
    dur = max(float(inc['duration_s']), 1.0)
    ego_tr = inc['ego']['track']
    start = {'x': ego_tr['x'][0], 'y': ego_tr['y'][0], 'yaw': ego_tr['yaw'][0], 'world': inc['world']}
    base = {'schema': SC.SCENARIO_SCHEMA, 'name': name, 'needs': [], 'duration_s': round(dur, 3),
            'source': dict(SOURCE), 'start': start, 'route_xy': inc['route_xy'], 'motion': 'kinematic',
            'pass': {'no_collision': True, 'min_ttc_s': 1.0}}
    rp = copy.deepcopy(base)
    rp.update(id=_slug(name) + '-replay', about='A recorded dashcam event replayed: the ego and every road user on '
                                                'their reconstructed tracks (core.incident).',
              params={}, ego={'speed_kmh': f['ego_speed_kmh'], 'driver': 'replay', 'route': 'straight',
                              'track': ego_tr},
              actors=[], timeline=[])
    tracks = {a['id']: a['track'] for a in inc['actors']}
    for a in f['actors']:
        rp['actors'].append({'id': a['id'], 'kind': a['kind'], 'place': dict(a['place']), 'heading': a['heading'],
                             'speed_kmh': a['speed_kmh'], 'start': a['start'], 'track': tracks[a['id']]})
    vr = copy.deepcopy(base)
    params = {'ego_speed': _param(f['ego_speed_kmh'], 3.0, 0.0, 'km/h', 'the ego\'s set speed', SPREAD['speed'])}
    actors, timeline = [], []
    for a in f['actors']:
        i = a['id']
        params[f'{i}_ahead'] = _param(a['place']['ahead_m'], SPREAD['gap_m'], None, 'm', f'{i}: ahead of the ego at '
                                      f't = 0', SPREAD['gap_rel'])
        params[f'{i}_lateral'] = _param(a['place']['lateral_m'], SPREAD['lateral_m'], None, 'm', f'{i}: off its '
                                        f'lane\'s centre')
        act = {'id': i, 'kind': a['kind'], 'place': {'lane': a['place']['lane'], 'ahead_m': f'${i}_ahead',
                                                     'lateral_m': f'${i}_lateral'},
               'heading': a['heading'], 'start': a['start'], 'speed_kmh': 0.0}
        if a['speed_kmh'] > 0:
            params[f'{i}_speed'] = _param(a['speed_kmh'], 3.0, 0.0, 'km/h', f'{i}: speed', SPREAD['speed'])
            act['speed_kmh'] = f'${i}_speed'
        actors.append(act)
        for k, x in enumerate(a['actions']):
            tn = f'{i}_t{k}'
            params[tn] = _param(x['t_s'], SPREAD['time_s'], 0.0, 's', f'{i}: when it {x["do"].replace("_", " ")}s')
            entry = {'actor': i, 'do': x['do']}
            if x['do'] == 'speed':
                params[f'{i}_kmh{k}'] = _param(x['kmh'], 3.0, 0.0, 'km/h', f'{i}: speed after braking', SPREAD['speed'])
                params[f'{i}_decel{k}'] = _param(x['accel'], 0.5, 0.3, 'm/s²', f'{i}: braking', SPREAD['decel'])
                entry.update(kmh=f'${i}_kmh{k}', accel=f'${i}_decel{k}')
            elif x['do'] == 'walk':
                params[f'{i}_walk{k}'] = _param(x['speed_mps'], 0.2, 0.2, 'm/s', f'{i}: walking speed', SPREAD['walk'])
                entry.update(speed_mps=f'${i}_walk{k}')
            elif x['do'] == 'lane_change':
                params[f'{i}_lc{k}'] = _param(x['duration_s'], 0.5, 0.4, 's', f'{i}: lane-change duration',
                                              SPREAD['decel'])
                entry.update(to=x['to'], duration_s=f'${i}_lc{k}')
            timeline.append({'trigger': {'type': 'time', 't_s': f'${tn}'}, 'actions': [entry]})
    vr.update(id=_slug(name) + '-variations', about='Variations around a recorded dashcam event: each extracted value '
                                                     'the centre of a uniform range (core.incident).',
              params=params, ego={'speed_kmh': '$ego_speed', 'driver': 'follower', 'route': 'straight'},
              actors=actors, timeline=timeline)
    if any(a['place']['lane'] == 'adjacent' for a in actors):
        vr['needs'] = ['adjacent_same']
    for d in (rp, vr):
        errs = SC.validate(d)
        if errs:
            raise IncidentError('the scenario does not validate: ' + '; '.join(errs))
    return rp, vr


def _slug(s):
    return re.sub(r'[^A-Za-z0-9_-]+', '-', s).strip('-').lower()[:40] or 'incident'


def default_name(rec_dir: str, start=None, end=None) -> str:
    """A name that does not repeat the clip's (which carries its date and time)."""
    h = hashlib.sha1(f'{os.path.realpath(rec_dir)}|{start}|{end}'.encode()).hexdigest()[:8]
    return f'incident-{h}'


def summary(inc: dict) -> dict:
    """What may be said outside the machine: counts, durations and shares only (no positions, times of day, places
    or names)."""
    kinds = {}
    for a in inc['actors']:
        kinds[a['kind']] = kinds.get(a['kind'], 0) + 1
    return {'duration_s': inc['duration_s'], 'actors': len(inc['actors']), 'kinds': kinds,
            'matched_share': inc['match']['matched_share'], 'lane_changes': inc['match']['lane_changes'],
            'boxes_used': inc['detections']['boxes_used'],
            'timeline_actions': sum(len(a['actions']) for a in inc['fit']['actors'])}


def incident_dir(workspace: str, name: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name or ''):
        raise IncidentError(f'an incident name is letters, digits, _ . - (not {name!r})')
    return os.path.join(workspace, 'incidents', name)


def write(inc: dict, out_dir: str, name: str) -> dict:
    """incident.json, scenario.json, variations.json, incident.xosc and run.json (the privacy marker) in out_dir."""
    from .openscenario import export_xosc
    from .provenance import utc_now, write_json
    os.makedirs(out_dir, exist_ok=True)
    rp, vr = scenarios(inc, name)
    write_json(os.path.join(out_dir, 'incident.json'), inc)
    write_json(os.path.join(out_dir, 'scenario.json'), rp)
    write_json(os.path.join(out_dir, 'variations.json'), vr)
    from . import scenarios as SC
    with open(os.path.join(out_dir, 'incident.xosc'), 'w', encoding='utf-8') as fh:
        fh.write(export_xosc(SC.concrete(rp, {})))
    summ = summary(inc)
    summ['search_parameters'] = len(vr['params'])
    write_json(os.path.join(out_dir, 'run.json'), {
        'worker': 'dashcam-incident', 'status': 'ok', 'name': name, 'created': utc_now(), 'source': dict(SOURCE),
        'world': inc['world'], 'summary': summ,
        'files': ['incident.json', 'scenario.json', 'variations.json', 'incident.xosc']})
    return summ


def load(workspace: str, name: str) -> dict:
    d = incident_dir(workspace, name)
    p = os.path.join(d, 'incident.json')
    if not os.path.isfile(p):
        raise IncidentError(f'no incident {name!r}')
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def search_argv(project_path: str, workspace: str, name: str, budget: int = 40, method: str = 'bayes',
                seed: int = 0, fake_carla: str = '') -> list:
    """`falcon search` over the incident's variations (the arguments after `python -m falcon_autonomy`)."""
    inc = load(workspace, name)
    argv = ['search', project_path, '--scenario', os.path.join(incident_dir(workspace, name), 'variations.json'),
            '--name', f'{name}-search', '--world', inc['world'], '--method', method, '--budget', str(int(budget)),
            '--seed', str(int(seed))]
    return argv + (['--fake-carla', fake_carla] if fake_carla else [])


def main_incident(argv) -> int:
    """`falcon dashcam incident extract|search|show …` (docs/dashcam.md, "A real incident as a scenario")."""
    import argparse
    from . import protocol as P
    from .project import Project
    ap = argparse.ArgumentParser(prog='falcon dashcam incident')
    sub = ap.add_subparsers(dest='what', required=True)
    e = sub.add_parser('extract', help='a reconstructed event as a scenario and an OpenSCENARIO file')
    e.add_argument('recon', help='the reconstruction folder (falcon recon from-dashcam)')
    e.add_argument('--project', required=True)
    e.add_argument('--world', required=True, help='the imported world the event is matched to')
    e.add_argument('--from', dest='start', type=float, default=None, help='window start, s from the first posed frame')
    e.add_argument('--to', dest='end', type=float, default=None)
    e.add_argument('--detections', default=None, help=f'other road users ({DETECTIONS_SCHEMA})')
    e.add_argument('--name', default='')
    s = sub.add_parser('search', help='the rare-case search over the variations around the event')
    s.add_argument('project')
    s.add_argument('name')
    s.add_argument('--budget', type=int, default=40)
    s.add_argument('--method', default='bayes')
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--fake-carla', default='')
    sh = sub.add_parser('show', help='what an incident holds (counts only)')
    sh.add_argument('project')
    sh.add_argument('name')
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return P.OK if ex.code == 0 else P.FAILED
    try:
        if a.what == 'extract':
            proj = Project.load(a.project)
            name = a.name or default_name(a.recon, a.start, a.end)
            inc = extract(a.recon, proj, a.world, a.start, a.end, a.detections)
            out = incident_dir(proj.workspace, name)
            summ = write(inc, out, name)
            print(json.dumps({'incident': name, 'folder': out, **summ}, indent=1))
            return P.OK
        proj = Project.load(a.project)
        if a.what == 'show':
            inc = load(proj.workspace, a.name)
            print(json.dumps(dict(summary(inc), world=inc['world'], window_s=inc['window_s']), indent=1))
            return P.OK
        from ..sim.studio import main_study
        return main_study(search_argv(proj.path, proj.workspace, a.name, a.budget, a.method, a.seed,
                                      a.fake_carla))
    except Exception as ex:          # noqa: BLE001 (DashcamError, IncidentError, OSError, ValueError: one sentence)
        print(f'failed: {type(ex).__name__}: {ex}' if not isinstance(ex, (IncidentError, OSError, ValueError))
              else f'failed: {ex}', flush=True)
        return P.FAILED
