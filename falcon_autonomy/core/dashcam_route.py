"""Dashcam routes: drive a stretch of your own dashcam footage in simulation, on the same streets (numpy; scipy).

    falcon dashcam route plan CLIP... [--start S] [--seconds N] [--out DIR]
    falcon dashcam route import PROJECT ROUTE (--osm FILE | --download-osm [--yes]) [--world-id W] [--rht]
    falcon dashcam route match PROJECT ROUTE --world W
    falcon dashcam route drive PROJECT ROUTE [--world W] [--rig R] [--driver follower|log] [--fps F] [--out RUN]
    falcon dashcam route pairs ROUTE RUN [--rate R] [--frames] [--size W H]

The steps, each reusing what Falcon already has:

1. **Window** (window.json). One or more clips (core.dashcam.track), joined on their GPS's UTC — consecutive clips
   make one timeline T (seconds from the first clip's video time 0) — and a window of it. The fixes' bounding box
   plus a margin (MARGIN_M) is the area to import; a box wider than max_side_m is refused (choose a shorter window).
2. **Roads** (an imported world). The area's roads from an OpenStreetMap file the user names (.osm or .osm.pbf,
   read by core.osm's own readers: no new dependency), cut to a corridor: highway ways with a node within
   CORRIDOR_M of the track, each clipped to the box (a way running on for kilometres is cut at the box's edge, one
   node beyond it, so its piece ends there). The corridor file goes through the ordinary import (workers.osm_import:
   carla.Osm2Odr, left-hand conversion, speed breakers, lane check) and becomes a project world. Downloading the
   area instead (Overpass API, overpass-api.de) happens only when the user asks for it and confirms: the request
   sends the box to that server; the data is © OpenStreetMap contributors under the ODbL 1.0.
3. **Route** (route.json). The track every 1 / RATE_HZ s (positions interpolated between the 1 Hz fixes; no
   interpolation across a hole of more than core.dashcam.MAX_GAP_S), in the world's frame (core.dashcam.to_world).
   Its fit to the imported lanes is measured with core.logreplay.mapmatch (positions and course as recorded). The
   route the ego drives is the track snapped to lane centres: per sample the candidate lanes are the driving-lane
   centre points within mapmatch.SEARCH_M whose heading is within mapmatch.HEADING_TOL_DEG of the course, one per
   lane (deduplicated by their lateral offset), and a Viterbi pass picks one per sample minimising
   Σ (offset / GPS_SIGMA_M)² / 2 + Σ (across / OFFSET_SIGMA_M)² / 2, where `across` is how far a sample's lane-centre
   point lies across the previous one's lane (≈ 0 on the same lane, a lane width on the next): the snapped lane
   changes only when the GPS stays on another lane for a while (heuristic; a real lane change shows up late).
   Samples with no candidate keep their GPS position (counted). The path is then smoothed (SMOOTH_S), held still
   where the GPS speed is below STOP_MPS (GPS wander at a standstill), and its heading taken from the path (from the
   lane where it stands still).
4. **Drive** (a run). The route becomes a log (core.logreplay: a geo-located ego track, no actors — the footage's
   other road users are not tracked) and runs through log replay: 'follower' re-drives it with the built-in lane
   follower at the recorded speeds (default), 'log' places the ego on the route every step (frames line up with the
   footage exactly in time, the motion is not driven). The run is an ordinary replay run (drive.json, run.json,
   replay.json with the measured match against the server's own lanes); drive.json — and so run.json's provenance —
   carries "dashcam_route": the clips' fingerprints, the window and route.json's SHA-256.
5. **Pairs** (pairs/<run>/pairs.jsonl). Dashcam frames at a rate, each with the simulated frame at the same moment:
   video frame n of clip k is at T = offset_k + n / fps_k, which is replay step round((T − T₀) · fps). Each pair
   carries both images' paths (the dashcam frame written only with --frames), the GPS position, the route pose and
   the simulated ego's pose, and their distance, along and across the route (the follower's deviation: a re-driven
   ego starting from the recorded speed's first value lags behind at first). Translation and reconstruction tools can
   read it.

Limits, stated in the files too: the GPS has metres of error and the route is only as good as the snapping; the
sub-second offset between the GPS and the video is not known (core.dashcam takes block k at video time k); the
camera's mounting and lens are not the rig's (the simulated camera is the project rig's); an OpenDRIVE world has no
buildings, vegetation or textures (docs/worlds.md); traffic in the footage is not reproduced.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time

import numpy as np

from . import dashcam as DC
from .dashcam import DashcamError

WINDOW_SCHEMA = 'falcon.dashcam_route_window/1'
ROUTE_SCHEMA = 'falcon.dashcam_route/1'
PAIRS_SCHEMA = 'falcon.dashcam_pairs/1'
DEFAULT_ROUTES = os.path.expanduser('~/FalconWorkspace/dashcam-routes')
MARGIN_M = 150.0                # around the track's extent: the imported box
MAX_SIDE_M = 6000.0             # a longer box is refused
CORRIDOR_M = 80.0               # a road with a node this close to the track is imported
RATE_HZ = 10.0                  # route samples a second
GPS_SIGMA_M = 5.0               # heuristic: GPS error, for the snapping's emission cost
OFFSET_SIGMA_M = 0.5            # heuristic: the lane-centre point's sideways step between samples on one lane
SMOOTH_S = 1.0                  # the snapped path's Gaussian smoothing (σ, seconds of travel)
STOP_MPS = 0.5                  # below this GPS speed the position is held
OVERPASS_URL = 'https://overpass-api.de/api/interpreter'
OSM_LICENCE = 'ODbL-1.0 (© OpenStreetMap contributors)'
HEURISTIC = [
    {'what': 'snapping', 'how': f'Viterbi over lane candidates: GPS σ {GPS_SIGMA_M:g} m, sideways step σ '
                                f'{OFFSET_SIGMA_M:g} m per sample, at {RATE_HZ:g} Hz'},
    {'what': 'stops', 'how': f'position held where the GPS speed < {STOP_MPS:g} m/s'},
    {'what': 'smoothing', 'how': f'Gaussian, σ {SMOOTH_S:g} s of travel'},
    {'what': 'GPS-to-video offset', 'how': 'block k at video time k (core.dashcam.track); not measured'},
    {'what': 'ego box', 'how': 'a typical car, 4.6 × 2.0 × 1.5 m; the dashcam car\'s size is unknown'},
]


def _epoch(iso: str) -> float:
    return _dt.datetime.fromisoformat(iso.replace('Z', '+00:00')).timestamp()


def _r(a, n=4) -> list:
    return [None if v is None or not math.isfinite(v) else round(float(v), n) + 0.0 for v in a]


# ================================================================================================ 1. the window
def window(clips: list, start: float = 0.0, seconds: float | None = None, margin_m: float = MARGIN_M,
           max_side_m: float = MAX_SIDE_M) -> dict:
    """The window of the clips' joined GPS timeline (see the module docstring): clips, fixes on T, the box to import."""
    from .osm import bbox_around, local_projection
    rows = []
    for c in clips:
        p = c if isinstance(c, str) else c['path']
        trk = DC.track(p)
        if not trk['fixes']:
            raise DashcamError(f'{os.path.basename(p)}: no GPS fix')
        rows.append((trk, DC.probe(p), os.path.abspath(p)))
    rows.sort(key=lambda r: _epoch(r[0]['t0_utc']))
    T0 = _epoch(rows[0][0]['t0_utc'])
    cdoc, fixes = [], []
    for k, (trk, pr, p) in enumerate(rows):
        off = round(_epoch(trk['t0_utc']) - T0, 3)
        dur = float(pr.get('duration_s') or (pr.get('n_frames') or 0) / (pr.get('fps') or 30.0))
        cdoc.append({'index': k, 'name': os.path.basename(p), 'path': p, 'fingerprint': DC.fingerprint(p),
                     't0_utc': trk['t0_utc'], 'offset_s': off, 'duration_s': round(dur, 3), 'fps': pr.get('fps'),
                     'fixes': trk['n_fixes']})
        for f in trk['fixes']:
            fixes.append(dict(f, clip=k, video_t=f['t'], T=round(off + f['t'], 3)))
    fixes.sort(key=lambda f: f['T'])
    uniq = []
    for f in fixes:                                       # overlapping clips: the first fix of a moment is kept
        if uniq and abs(f['T'] - uniq[-1]['T']) < 0.5:
            continue
        uniq.append(f)
    end_all = max(c['offset_s'] + c['duration_s'] for c in cdoc)
    t0 = float(start)
    t1 = end_all if seconds is None else min(end_all, t0 + float(seconds))
    if t1 - t0 < 1.0:
        raise DashcamError(f'the window {t0:.1f}–{t1:.1f} s is shorter than a second (the clips cover 0–'
                           f'{end_all:.1f} s)')
    sel = [f for f in uniq if t0 - DC.MAX_GAP_S <= f['T'] <= t1 + DC.MAX_GAP_S]
    if len(sel) < 2:
        raise DashcamError(f'fewer than two GPS fixes in {t0:.1f}–{t1:.1f} s')
    holes = [(a['T'], b['T']) for a, b in zip(sel, sel[1:]) if b['T'] - a['T'] > 2 * DC.MAX_GAP_S]
    uncovered = _uncovered(cdoc, t0, t1)
    lat = np.array([f['lat'] for f in sel])
    lon = np.array([f['lon'] for f in sel])
    pr = local_projection(float(np.median(lat)), float(np.median(lon)))
    e, n = pr.to_local(lat, lon)
    w_e, w_n = float(e.max() - e.min()) + 2 * margin_m, float(n.max() - n.min()) + 2 * margin_m
    if max(w_e, w_n) > max_side_m:
        raise DashcamError(f'the window\'s area is {w_e / 1000:.1f} × {w_n / 1000:.1f} km with the {margin_m:g} m '
                           f'margin, over the {max_side_m / 1000:g} km limit: choose a shorter window')
    c_lat, c_lon = pr.to_geo((e.max() + e.min()) / 2, (n.max() + n.min()) / 2)
    s_, w_, n_, e_ = bbox_around(float(c_lat), float(c_lon), 1.0)
    dlat, dlon = (n_ - s_), (e_ - w_)                     # degrees per metre at the centre
    bbox = [float(c_lat - dlat * w_n / 2), float(c_lon - dlon * w_e / 2), float(c_lat + dlat * w_n / 2),
            float(c_lon + dlon * w_e / 2)]
    path_m = float(np.sum(np.hypot(np.diff(e), np.diff(n))))
    sp = [f['speed_kmh'] for f in sel if f['speed_kmh'] is not None]
    return {'schema': WINDOW_SCHEMA, 'clips': cdoc, 'start_s': round(t0, 3), 'end_s': round(t1, 3),
            'fixes': sel, 'holes': holes, 'uncovered': uncovered, 'bbox': [round(v, 7) for v in bbox],
            'centre': [round(float(c_lat), 7), round(float(c_lon), 7)], 'size_m': [round(w_e, 1), round(w_n, 1)],
            'margin_m': margin_m, 'path_m': round(path_m, 1),
            'speed_kmh': {'mean': round(float(np.mean(sp)), 1) if sp else None,
                          'max': round(float(np.max(sp)), 1) if sp else None},
            'created': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}


def _uncovered(cdoc: list, t0: float, t1: float) -> list:
    """Stretches of [t0, t1] no clip's video covers."""
    spans = sorted((c['offset_s'], c['offset_s'] + c['duration_s']) for c in cdoc)
    out, t = [], t0
    for a, b in spans:
        if b <= t:
            continue
        if a > t + 0.05:
            out.append((round(t, 3), round(min(a, t1), 3)))
        t = max(t, b)
        if t >= t1:
            break
    if t < t1 - 0.05:
        out.append((round(t, 3), round(t1, 3)))
    return [s for s in out if s[1] > s[0]]


def save_window(doc: dict, route_dir: str) -> str:
    from .provenance import write_json
    os.makedirs(route_dir, exist_ok=True)
    p = os.path.join(route_dir, 'window.json')
    write_json(p, doc)
    return p


def load(route_dir: str, name: str = 'window.json') -> dict:
    with open(os.path.join(route_dir, name), encoding='utf-8') as fh:
        return json.load(fh)


def default_route_dir(win: dict) -> str:
    stem = os.path.splitext(win['clips'][0]['name'])[0]
    return os.path.join(DEFAULT_ROUTES, f'{stem}-{win["start_s"]:g}s-{win["end_s"] - win["start_s"]:g}s')


# ================================================================================================ 2. the roads
def dense_track(win: dict, step_s: float = 1.0 / RATE_HZ) -> dict:
    """The window's track every step_s on T: t (T), lat, lon, speed (m/s), course (degrees), clip, video_t."""
    fx = win['fixes']
    T = np.array([f['T'] for f in fx])
    t0, t1 = win['start_s'], win['end_s']
    tq = np.arange(t0, t1 + 1e-9, step_s)
    tq = tq[(tq >= T[0]) & (tq <= T[-1])]
    j = np.clip(np.searchsorted(T, tq) - 1, 0, len(T) - 2)
    gap = T[j + 1] - T[j]
    ok = gap <= 2 * DC.MAX_GAP_S
    if not ok.all():
        bad = tq[~ok][0]
        raise DashcamError(f'the GPS has a hole of {gap[~ok][0]:.1f} s at T = {bad:.1f} s: choose a window without it')
    sp = np.array([(f['speed_kmh'] if f['speed_kmh'] is not None else np.nan) / 3.6 for f in fx])
    if np.isnan(sp).any():
        good = ~np.isnan(sp)
        sp = np.interp(T, T[good], sp[good]) if good.any() else np.zeros(len(T))
    crs = np.degrees(np.unwrap(np.radians([f['course_deg'] if f['course_deg'] is not None else 0.0 for f in fx])))
    clip = np.full(len(tq), -1)
    vt = np.full(len(tq), np.nan)
    for c in win['clips']:                                # the first clip whose video covers the moment
        m = (clip < 0) & (tq >= c['offset_s'] - 1e-6) & (tq <= c['offset_s'] + c['duration_s'] + 1e-6)
        clip[m] = c['index']
        vt[m] = tq[m] - c['offset_s']
    return {'t': tq, 'lat': np.interp(tq, T, [f['lat'] for f in fx]), 'lon': np.interp(tq, T, [f['lon'] for f in fx]),
            'speed': np.interp(tq, T, sp), 'course': np.interp(tq, T, crs) % 360.0, 'clip': clip, 'video_t': vt}


def clip_ways(data, bbox, track_ll, corridor_m: float = CORRIDOR_M) -> tuple:
    """(ways, counts): highway ways with a node within corridor_m of the track (lat, lon arrays), each cut to the box
    — every run of nodes inside it, with one node beyond each end — as pieces with new ids (way id × 100 + k)."""
    from scipy.spatial import cKDTree
    from .osm import local_projection
    lat, lon = track_ll
    pr = local_projection(float(np.median(lat)), float(np.median(lon)))
    te, tn = pr.to_local(np.asarray(lat), np.asarray(lon))
    tree = cKDTree(np.column_stack([te, tn]))
    s, w, n, e = bbox
    out, counts = [], {'ways_in_box': 0, 'ways_in_corridor': 0, 'pieces': 0, 'cut': 0}
    for wy in data.highways:
        refs = [r for r in wy['nodes'] if r in data.nodes]
        if len(refs) < 2:
            continue
        ll = np.array([data.nodes[r] for r in refs])
        inside = (ll[:, 0] >= s) & (ll[:, 0] <= n) & (ll[:, 1] >= w) & (ll[:, 1] <= e)
        if not inside.any():
            continue
        counts['ways_in_box'] += 1
        we, wn = pr.to_local(ll[:, 0], ll[:, 1])
        d, _ = tree.query(np.column_stack([we, wn]))
        if not (d[inside] <= corridor_m).any() and not _segment_near(we, wn, tree, corridor_m):
            continue
        counts['ways_in_corridor'] += 1
        idx = np.flatnonzero(inside)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        if len(runs) > 1 or idx[0] > 0 or idx[-1] < len(refs) - 1:
            counts['cut'] += 1
        for k, r in enumerate(runs):
            a, b = max(0, int(r[0]) - 1), min(len(refs) - 1, int(r[-1]) + 1)
            piece = refs[a:b + 1]
            if len(piece) < 2:
                continue
            out.append({'id': int(wy['id']) * 100 + k, 'nodes': piece, 'tags': dict(wy['tags'], **{
                'falcon:source_way': str(wy['id'])}), 'kind': 'highway'})
            counts['pieces'] += 1
    return out, counts


def _segment_near(we, wn, tree, r) -> bool:
    """A way whose nodes are far apart may pass the track between two of them: test points every r / 2 along it."""
    for i in range(len(we) - 1):
        L = math.hypot(we[i + 1] - we[i], wn[i + 1] - wn[i])
        if L <= r:
            continue
        f = np.linspace(0, 1, int(L / (r / 2)) + 2)
        pts = np.column_stack([we[i] + f * (we[i + 1] - we[i]), wn[i] + f * (wn[i + 1] - wn[i])])
        if (tree.query(pts)[0] <= r).any():
            return True
    return False


def corridor_osm(source: str, win: dict, out_path: str, corridor_m: float = CORRIDOR_M, progress=None) -> dict:
    """Read the source cropped to the window's box (core.osm), keep the corridor (clip_ways) and write it as OSM XML
    for the import. Returns what was read and kept."""
    from . import osm as O
    from .provenance import sha256_file
    t0 = time.monotonic()
    data = O.read_osm(source, bbox=tuple(win['bbox']), progress=progress)
    trk = dense_track(win, 1.0)
    ways, counts = clip_ways(data, win['bbox'], (trk['lat'], trk['lon']), corridor_m)
    if not ways:
        raise DashcamError('no roads near the track in this OpenStreetMap file (is the area inside the file?)')
    sub = O.subset(data, ways, bounds=win['bbox'],
                   notes=[f'cut to a {corridor_m:g} m corridor of a dashcam track (core.dashcam_route)'])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    O.write_osm(sub, out_path)
    is_file = os.path.isfile(source)
    return {'source': os.path.abspath(source), 'source_sha256': None if not is_file or os.path.getsize(source) > 64e6
            else sha256_file(source), 'source_size': os.path.getsize(source) if is_file else None,
            'corridor_m': corridor_m, 'bbox': win['bbox'], 'counts': counts, 'highways': len(sub.highways),
            'nodes': len(sub.nodes), 'read_s': round(time.monotonic() - t0, 2), 'out': out_path}


def overpass_query(bbox) -> str:
    s, w, n, e = bbox
    return (f'[out:xml][timeout:120];(way["highway"]({s:.7f},{w:.7f},{n:.7f},{e:.7f}););(._;>;);out body;')


def fetch_overpass(bbox, out_path: str, allowed: bool, url: str = OVERPASS_URL, timeout: float = 180.0) -> dict:
    """Download the box's roads from the Overpass API (sends the box to `url`). Only when the user asked for it:
    raises PermissionError otherwise."""
    if not allowed:
        raise PermissionError(f'downloading OpenStreetMap data is off: it sends the area to {url}')
    import hashlib
    import urllib.parse
    import urllib.request
    from .osm import USER_AGENT
    from .paths import atomic_write
    body = urllib.parse.urlencode({'data': overpass_query(bbox)}).encode()
    req = urllib.request.Request(url, data=body, headers={'User-Agent': USER_AGENT})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if b'<osm' not in raw[:2000]:
        raise DashcamError(f'{url} did not return OSM XML: {raw[:200]!r}')
    atomic_write(out_path, raw.decode('utf-8'))
    return {'url': url, 'query': overpass_query(bbox), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'licence': OSM_LICENCE, 'downloaded': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'seconds': round(time.monotonic() - t0, 2), 'out': out_path}


def import_config(win: dict, corridor_path: str, out_dir: str, lht: bool = True, settings: dict | None = None,
                  seed: int = 0) -> dict:
    """The OSM import's config (core.worlds.import_config) for the corridor file, centred on the window's box."""
    from .worlds import import_config as IC
    return IC(corridor_path, out_dir, win['bbox'], lht, settings, centre=win['centre'], seed=seed)


def add_world(project, world_id: str, name: str, cfg: dict, report: dict) -> dict:
    """The world entry (core.worlds.world_entry; left-hand traffic gets the project's left-hand profile) appended to
    the project document (not saved)."""
    from .worlds import world_entry
    doc = project.doc
    left = next((tp['id'] for tp in doc['traffic_profiles'] if tp.get('handedness') == 'left'), None)
    tp = left if report.get('handedness') == 'LHT' and left else 'tp-default'
    w = world_entry(world_id, name, cfg, report, None, tp)
    doc['worlds'] = [x for x in doc['worlds'] if x['id'] != world_id] + [w]
    return w


# ================================================================================================ 3. the route
def world_for(project, world_id: str, workspace: str = '') -> dict:
    """core.dashcam.world_geo for a Project object (the GUI's, possibly unsaved): lat0, lon0, bbox, lanes."""
    from .logreplay.mapmatch import lanes_from_xodr
    from .logreplay.plan import world_info
    w = world_info(project, world_id, workspace)
    if not w.get('geo'):
        raise DashcamError(f'world {world_id!r} has no geographic reference (a CARLA town is not on the map)')
    out = {'id': world_id, 'name': w.get('name'), 'lat0': w['geo']['lat0'], 'lon0': w['geo']['lon0'],
           'bbox': w.get('bbox'), 'xodr': w.get('xodr'), 'lanes': np.zeros((0, 4)), 'lanes_from': None}
    if w.get('xodr') and os.path.isfile(w['xodr']):
        with open(w['xodr'], encoding='utf-8') as fh:
            out['lanes'] = lanes_from_xodr(fh.read())
        out['lanes_from'] = os.path.basename(w['xodr'])
    return out


def snap_to_lanes(x, y, yaw, lanes: np.ndarray, gps_sigma: float = GPS_SIGMA_M,
                  offset_sigma: float = OFFSET_SIGMA_M) -> dict:
    """Lane-centre positions for a track (module docstring, step 3): {'x', 'y', 'yaw' (the lane's, NaN where
    unmatched), 'offset' (signed GPS-to-lane-centre distance, NaN where unmatched), 'matched', 'lane_changes'}."""
    from scipy.spatial import cKDTree
    from .logreplay import mapmatch as MM
    x, y, yaw = (np.asarray(v, float) for v in (x, y, yaw))
    n = len(x)
    out = {'x': x.copy(), 'y': y.copy(), 'yaw': np.full(n, np.nan), 'offset': np.full(n, np.nan),
           'matched': np.zeros(n, bool), 'lane_changes': 0}
    if not len(lanes) or not n:
        return out
    tree = cKDTree(lanes[:, :2])
    cands = []
    for i in range(n):
        idx = tree.query_ball_point((x[i], y[i]), MM.SEARCH_M)
        rows = []
        if idx:
            L = lanes[idx]
            a = np.radians(L[:, 2])
            hd = np.abs((yaw[i] - L[:, 2] + 180.0) % 360.0 - 180.0)
            lat = (x[i] - L[:, 0]) * -np.sin(a) + (y[i] - L[:, 1]) * np.cos(a)
            along = (x[i] - L[:, 0]) * np.cos(a) + (y[i] - L[:, 1]) * np.sin(a)
            ok = (hd <= MM.HEADING_TOL_DEG) & (np.abs(along) <= 1.0)
            for k in np.flatnonzero(ok)[np.argsort(np.abs(along[ok]))]:
                if all(abs(lat[k] - r[0]) > 1.0 for r in rows):           # one candidate per lane
                    rows.append((float(lat[k]), float(L[k, 2]), float(a[k]),
                                 x[i] + lat[k] * math.sin(a[k]), y[i] - lat[k] * math.cos(a[k])))
        cands.append(rows)
    # Viterbi: state = a candidate, or "none" (the GPS position kept)
    none_cost = 0.5 * (MM.SEARCH_M / gps_sigma) ** 2 + 2.0
    prev_cost, back = None, []
    for i, rows in enumerate(cands):
        emit = np.array([0.5 * (r[0] / gps_sigma) ** 2 for r in rows] + [none_cost])
        if prev_cost is None:
            cost = emit
            back.append(np.full(len(emit), -1))
        else:
            tr = np.full((len(rows) + 1, len(cands[i - 1]) + 1), 2.0)    # to or from "none": a fixed cost
            for kk, c in enumerate(rows):
                for jj, p in enumerate(cands[i - 1]):
                    tr[kk, jj] = 0.5 * (_across(c, p) / offset_sigma) ** 2
            tot = prev_cost[None, :] + tr
            b = np.argmin(tot, axis=1)
            cost = tot[np.arange(len(rows) + 1), b] + emit
            back.append(b)
        prev_cost = cost
    k = int(np.argmin(prev_cost))
    path = [k]
    for i in range(n - 1, 0, -1):
        k = int(back[i][k])
        path.append(k)
    path.reverse()
    last = None
    for i, k in enumerate(path):
        rows = cands[i]
        if k >= len(rows):
            continue
        lat, lyaw, a, sx, sy = rows[k]
        out['x'][i], out['y'][i] = sx, sy
        out['yaw'][i] = lyaw
        out['offset'][i] = lat
        out['matched'][i] = True
        if last is not None and _across(rows[k], last) > 1.5:
            out['lane_changes'] += 1
        last = rows[k]
    return out


def _across(c, p) -> float:
    """How far candidate c's lane-centre point lies across candidate p's lane (along p's lane normal): ≈ 0 on the
    same lane, about a lane width on the next one — whatever the GPS noise."""
    return abs((c[3] - p[3]) * -math.sin(p[2]) + (c[4] - p[4]) * math.cos(p[2]))


def _smooth(x, y, s, sigma_m):
    """Gaussian smoothing of a path over its arc length s (σ in metres); unchanged when the path is shorter."""
    if sigma_m <= 0 or len(x) < 3 or s[-1] < 3 * sigma_m:
        return x, y
    sg = np.arange(0.0, s[-1] + 1e-9, max(sigma_m / 4, 0.25))
    xg, yg = np.interp(sg, s, x), np.interp(sg, s, y)
    k = int(math.ceil(3 * sigma_m / (sg[1] - sg[0])))
    w = np.exp(-0.5 * ((np.arange(-k, k + 1) * (sg[1] - sg[0])) / sigma_m) ** 2)
    w /= w.sum()
    pad = lambda v: np.r_[np.full(k, v[0]), v, np.full(k, v[-1])]          # noqa: E731
    xs, ys = np.convolve(pad(xg), w, 'valid'), np.convolve(pad(yg), w, 'valid')
    return np.interp(s, sg, xs), np.interp(s, sg, ys)


def build_route(win: dict, world: dict, rate_hz: float = RATE_HZ) -> dict:
    """The route of a window in a world (core.dashcam.world_geo): the GPS track and the snapped route, with the
    measured fit (module docstring, step 3)."""
    from .logreplay import mapmatch as MM
    tr = dense_track(win, 1.0 / rate_hz)
    gx, gy, gyaw = DC.to_world(world, tr['lat'], tr['lon'], tr['course'])
    lanes = world['lanes']
    raw = MM.match(gx, gy, gyaw, lanes, world.get('lanes_from') or '')
    sn = snap_to_lanes(gx, gy, gyaw, lanes)
    x, y = sn['x'].copy(), sn['y'].copy()
    moving = tr['speed'] >= STOP_MPS
    for i in range(1, len(x)):                            # hold still while stopped
        if not moving[i]:
            x[i], y[i] = x[i - 1], y[i - 1]
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    v_mean = float(np.mean(tr['speed'][moving])) if moving.any() else 0.0
    x, y = _smooth(x, y, s, SMOOTH_S * max(v_mean, 1.0))
    dx, dy = np.gradient(x), np.gradient(y)
    yaw = np.degrees(np.arctan2(dy, dx))
    still = np.hypot(dx, dy) < 0.05
    fallback = np.where(np.isnan(sn['yaw']), gyaw, sn['yaw'])
    yaw = np.where(still, fallback, yaw)
    snapped = MM.match(x, y, yaw, lanes, world.get('lanes_from') or '')
    path_gps = float(np.sum(np.hypot(np.diff(gx), np.diff(gy))))
    path_route = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    off = np.abs(sn['offset'][sn['matched']])
    dist_t = float(np.sum(tr['speed']) / rate_hz)
    return {
        'schema': ROUTE_SCHEMA, 'rate_hz': rate_hz, 'T0': float(tr['t'][0]),
        'world': {k: world.get(k) for k in ('id', 'name', 'lat0', 'lon0', 'bbox', 'lanes_from')},
        'lane_points': int(len(lanes)),
        'samples': {'t': _r(tr['t'] - tr['t'][0], 3), 'T': _r(tr['t'], 3), 'clip': [int(v) for v in tr['clip']],
                    'video_t': _r(tr['video_t'], 3), 'lat': _r(tr['lat'], 8), 'lon': _r(tr['lon'], 8),
                    'speed_mps': _r(tr['speed'], 3), 'course_deg': _r(tr['course'], 2),
                    'gps_x': _r(gx, 3), 'gps_y': _r(gy, 3), 'x': _r(x, 3), 'y': _r(y, 3), 'yaw': _r(yaw, 3),
                    'matched': [bool(v) for v in sn['matched']]},
        'fit': {
            'gps_against_lanes': raw,
            'route_against_lanes': {k: snapped.get(k) for k in ('verdict', 'why', 'share_on_lane_in_direction',
                                                                 'median_lateral_m', 'p95_lateral_m')},
            'snapped_share': round(float(np.mean(sn['matched'])), 4),
            'gps_to_lane_m': {'median': round(float(np.median(off)), 3) if len(off) else None,
                              'p95': round(float(np.percentile(off, 95)), 3) if len(off) else None,
                              'max': round(float(off.max()), 3) if len(off) else None},
            'lane_changes': int(sn['lane_changes']),
            'path_m': {'gps': round(path_gps, 1), 'route': round(path_route, 1),
                       'speed_integral': round(dist_t, 1)},
        },
        'heuristic': HEURISTIC,
        'limits': ['the GPS has metres of error: the snapped lane is a best guess, a lane change shows late',
                   'the GPS-to-video offset (below a second) is not known',
                   'traffic in the footage is not reproduced; the simulated camera is the rig\'s, not the dashcam\'s'],
    }


def route_log(route: dict, win: dict, route_path: str = ''):
    """The route as a log (core.logreplay.Log): a geo-located ego track, t from the window's first route sample."""
    from .logreplay.model import Log, Track
    s = route['samples']
    t = np.array(s['t'], float)
    ego = Track('ego', 'car', t, s['x'], s['y'], np.zeros(len(t)), s['yaw'], None, 'dashcam car', 'typical')
    first = win['clips'][0]
    meta = {'name': f'dashcam route {os.path.splitext(first["name"])[0]} {win["start_s"]:g}–{win["end_s"]:g} s',
            'source': {'format': 'dashcam', 'path': first['path'], 'segment': f'{win["start_s"]:g}–{win["end_s"]:g} s',
                       'T0': route['T0'], 'route': route_path,
                       'clips': [{k: c[k] for k in ('name', 'fingerprint', 'offset_s', 'duration_s')}
                                 for c in win['clips']]},
            'ego_origin': 'the dashcam car (GPS antenna position unknown)',
            'map': {'kind': 'geo', 'name': route['world'].get('name'),
                    'geo': {'lat0': route['world']['lat0'], 'lon0': route['world']['lon0'], 'method': 'tmerc',
                            'source': 'the imported world\'s projection'}},
            'notes': ['the route is the GPS track snapped to lane centres (core.dashcam_route)'],
            'left_out': [{'what': 'actors', 'why': 'the footage\'s other road users are not tracked'},
                         {'what': 'images', 'why': 'a replay renders CARLA\'s own sensors; pairs link the footage'}],
            'heuristic': HEURISTIC}
    return Log(ego, [], meta)


# ================================================================================================ 4. the drive
def drive_config(project, route_dir: str, world_id: str, rig_id: str, out_dir: str, port: int, tm_port: int,
                 driver: str = 'follower', fps: float = 20.0, seed: int = 0, render_budget_mpx: float = 200.0,
                 allow_mismatch: bool = False) -> dict:
    """The replay config (core.logreplay.commands.replay_config) driving the route, with "dashcam_route"
    provenance. Writes the route's log.json."""
    from .logreplay.commands import replay_config
    from .logreplay.plan import make_plan, world_info
    from .provenance import sha256_file
    win, route = load(route_dir), load(route_dir, 'route.json')
    if route['world']['id'] != world_id:
        raise DashcamError(f'route.json was matched to world {route["world"]["id"]!r}, not {world_id!r}: run '
                           f'`falcon dashcam route match` for this world')
    rp = os.path.join(route_dir, 'route.json')
    log = route_log(route, win, rp)
    log_path = log.save(os.path.join(route_dir, 'log.json'))
    plan = make_plan(log, world_info(project, world_id), fps=fps, seed=seed, driver=driver,
                     allow_mismatch=allow_mismatch, log_path=log_path)
    cfg = replay_config(project, plan, world_id, rig_id, out_dir, port, tm_port, render_budget_mpx,
                        project.workspace, name=f'dashcam route ({driver}): {log.meta["name"]}')
    cfg['dashcam_route'] = {'route_dir': route_dir, 'route_sha256': sha256_file(rp), 'window': [win['start_s'],
                                                                                                  win['end_s']],
                            'T0': route['T0'], 'driver': driver,
                            'clips': [{k: c[k] for k in ('name', 'fingerprint', 'offset_s', 'duration_s')}
                                      for c in win['clips']]}
    return cfg


def new_run_dir(workspace: str) -> str:
    return os.path.join(workspace, 'runs', time.strftime('%Y%m%d-%H%M%S') + '-dashcam-route')


# ================================================================================================ 5. the pairs
def _rows(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def pairs(route_dir: str, run_dir: str, rate: float = 2.0, frames: bool = False, size=(640, 360),
          crop_bottom: float = 0.0, camera: str | None = None, out_dir: str | None = None, progress=None) -> dict:
    """Dashcam frames paired with the run's frames at the same moment (module docstring, step 5)."""
    from .provenance import block, read_json, sha256_file, write_json
    win, route = load(route_dir), load(route_dir, 'route.json')
    cfg = read_json(os.path.join(run_dir, 'drive.json'))
    plan = cfg['replay']['plan']
    fps_sim, steps = float(plan['fps']), int(plan['steps'])
    T0 = float(route['T0']) + float(plan.get('t_start') or 0.0)
    rows = {r['step']: r for r in _rows(os.path.join(run_dir, 'frames.jsonl'))}
    if not rows:
        raise DashcamError(f'{run_dir}: no recorded frames')
    cam_key = None
    for k in next(iter(rows.values())).get('files', {}):
        if k.endswith('/rgb') and (camera is None or k.split('/')[0] == camera):
            cam_key = k
            break
    rs = route['samples']
    rT = np.array(rs['T'], float)
    out_dir = out_dir or os.path.join(route_dir, 'pairs', os.path.basename(os.path.normpath(run_dir)))
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, 'real')
    lines, dev = [], []
    t_end = T0 + (steps - 1) / fps_sim
    for c in win['clips']:
        a = max(win['start_s'], T0, c['offset_s']) - c['offset_s']
        b = min(win['end_s'], t_end, c['offset_s'] + c['duration_s']) - c['offset_s']
        if b <= a:
            continue
        fps_v = float(c['fps'] or 30.0)
        step = max(1, int(round(fps_v / rate)))
        if frames:
            os.makedirs(real_dir, exist_ok=True)
            it = ((n, t, img) for n, t, img in DC.iter_frames(c['path'], rate=rate, size=tuple(size), start=a,
                                                               duration=b - a, crop_bottom=crop_bottom))
        else:
            n0 = int(round(a * fps_v))
            it = ((n, n / fps_v, None) for n in range(n0, int(math.floor(b * fps_v)) + 1, step))
        for n, tv, img in it:
            T = c['offset_s'] + tv
            i = int(round((T - T0) * fps_sim))
            if not 0 <= i < steps or i not in rows:
                continue
            r = rows[i]
            j = int(np.clip(np.searchsorted(rT, T), 0, len(rT) - 1))
            if j > 0 and abs(rT[j - 1] - T) < abs(rT[j] - T):
                j -= 1
            rec = {'T': round(T, 4), 'clip': c['index'], 'video_frame': int(n), 'video_t': round(tv, 4),
                   'step': i, 'sim_t': r.get('sim_t'), 'sim_image': (r.get('files') or {}).get(cam_key),
                   'real_image': None, 'lat': rs['lat'][j], 'lon': rs['lon'][j],
                   'gps_speed_mps': rs['speed_mps'][j],
                   'route': {'x': rs['x'][j], 'y': rs['y'][j], 'yaw': rs['yaw'][j]},
                   'sim_ego': {k: (r.get('ego') or {}).get(k) for k in ('x', 'y', 'yaw')},
                   'sim_speed_mps': r.get('speed_mps')}
            e = r.get('ego') or {}
            if e.get('x') is not None:
                dx, dy = e['x'] - rs['x'][j], e['y'] - rs['y'][j]
                ya = math.radians(rs['yaw'][j])
                d = math.hypot(dx, dy)
                al, ac = dx * math.cos(ya) + dy * math.sin(ya), -dx * math.sin(ya) + dy * math.cos(ya)
                rec.update(sim_minus_route_m=round(d, 3), sim_along_m=round(al, 3), sim_across_m=round(ac, 3))
                dev.append((d, al, ac))
            if img is not None:
                import cv2
                name = f'{c["index"]:02d}-{int(n):06d}.jpg'
                cv2.imwrite(os.path.join(real_dir, name), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                rec['real_image'] = os.path.join('real', name)
            lines.append(rec)
            if progress is not None and len(lines) % 20 == 0:
                progress(min(0.99, (T - T0) / max(t_end - T0, 1e-6)), f'{len(lines)} pairs')
    with open(os.path.join(out_dir, 'pairs.jsonl'), 'w', encoding='utf-8') as fh:
        for rec in lines:
            fh.write(json.dumps(rec, separators=(',', ':')) + '\n')
    rj = os.path.join(run_dir, 'run.json')
    dv = np.array(dev) if dev else np.zeros((0, 3))

    def q(v):
        return {'median': round(float(np.median(v)), 3) if len(v) else None,
                'p95': round(float(np.percentile(v, 95)), 3) if len(v) else None,
                'max': round(float(v.max()), 3) if len(v) else None}
    doc = {'schema': PAIRS_SCHEMA, 'run': run_dir, 'route_dir': route_dir, 'pairs': len(lines), 'rate': rate,
           'camera': cam_key, 'frames_written': bool(frames), 'size': list(size) if frames else None,
           'crop_bottom': crop_bottom, 'T0': T0, 'fps': fps_sim,
           'sim_minus_route_m': q(dv[:, 0]), 'sim_along_m': q(np.abs(dv[:, 1])),
           'sim_across_m': q(np.abs(dv[:, 2])),
           'rule': 'video frame n of clip k at T = offset_k + n / fps_k pairs with replay step round((T − T0) · fps); '
                   'the route sample nearest in time gives the GPS and route pose',
           'provenance': block({'route': os.path.join(route_dir, 'route.json'), 'run': rj,
                                'drive': os.path.join(run_dir, 'drive.json')}),
           'clips': [{k: c[k] for k in ('name', 'fingerprint', 'offset_s')} for c in win['clips']]}
    if os.path.isfile(rj):
        doc['run_sha256'] = sha256_file(rj)
    write_json(os.path.join(out_dir, 'pairs.json'), doc)
    return doc


# ================================================================================================ the command
def _service(fake_carla: str = ''):
    from .jobs import JobQueue, SubprocessLauncher
    from .settings import Settings
    from ..sim.service import CarlaService
    return CarlaService(Settings(), JobQueue(SubprocessLauncher()), '', fake_carla=fake_carla)


def _confirm(text: str) -> bool:
    import sys
    if not sys.stdin or not sys.stdin.isatty():
        return False
    print(text + ' [y/N] ', end='', file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in ('y', 'yes')


def route_text(route: dict) -> str:
    f = route['fit']
    g = f['gps_against_lanes']
    r = f['route_against_lanes']
    return (f'GPS against the lanes: {g.get("verdict")} ({g.get("why")}); snapped {100 * f["snapped_share"]:.1f} % of '
            f'samples, GPS-to-lane median {f["gps_to_lane_m"]["median"]} m, p95 {f["gps_to_lane_m"]["p95"]} m, '
            f'{f["lane_changes"]} lane change(s); route: {r.get("verdict")}; path GPS {f["path_m"]["gps"]} m, route '
            f'{f["path_m"]["route"]} m')


def main_route(argv) -> int:
    """`falcon dashcam route plan|import|match|drive|pairs`: protocol lines on stdout; 0 ok, 2 failed."""
    import argparse
    import sys
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon dashcam route', description='drive a stretch of your dashcam footage in '
                                 'simulation on the same streets (docs/dashcam.md)')
    sub = ap.add_subparsers(dest='what', required=True)
    pl = sub.add_parser('plan', help='a window of one or more clips: its GPS track and the area to import')
    pl.add_argument('clips', nargs='+')
    pl.add_argument('--start', type=float, default=0.0, help='seconds from the first clip\'s start')
    pl.add_argument('--seconds', type=float)
    pl.add_argument('--margin', type=float, default=MARGIN_M)
    pl.add_argument('--max-side', type=float, default=MAX_SIDE_M)
    pl.add_argument('--out', '-o', help=f'the route folder (default under {DEFAULT_ROUTES})')
    im = sub.add_parser('import', help='import the route\'s roads as a project world and match the route to them')
    im.add_argument('project')
    im.add_argument('route')
    src = im.add_mutually_exclusive_group(required=True)
    src.add_argument('--osm', help='a local OpenStreetMap file (.osm, .osm.pbf) covering the area (read only)')
    src.add_argument('--download-osm', action='store_true',
                     help=f'download the area\'s roads from {OVERPASS_URL} (asks first; ODbL data)')
    im.add_argument('--yes', action='store_true', help='with --download-osm: do not ask')
    im.add_argument('--world-id', default='')
    im.add_argument('--name', default='')
    im.add_argument('--rht', action='store_true', help='right-hand traffic (default: left-hand)')
    im.add_argument('--corridor', type=float, default=CORRIDOR_M)
    im.add_argument('--fake-carla', default='', help=argparse.SUPPRESS)
    ma = sub.add_parser('match', help='match the route to a world of the project (route.json)')
    ma.add_argument('project')
    ma.add_argument('route')
    ma.add_argument('--world', required=True)
    dr = sub.add_parser('drive', help='drive the route in CARLA (a replay run) and pair its frames with the footage')
    dr.add_argument('project')
    dr.add_argument('route')
    dr.add_argument('--world', default='')
    dr.add_argument('--rig', default='')
    dr.add_argument('--driver', default='follower', choices=('follower', 'log'))
    dr.add_argument('--fps', type=float, default=20.0)
    dr.add_argument('--seed', type=int, default=0)
    dr.add_argument('--out', '-o', default='')
    dr.add_argument('--allow-mismatch', action='store_true')
    dr.add_argument('--pair-rate', type=float, default=2.0)
    dr.add_argument('--frames', action='store_true', help='also write the paired dashcam frames')
    dr.add_argument('--keep-server', action='store_true')
    dr.add_argument('--fake-carla', default='', help=argparse.SUPPRESS)
    pa = sub.add_parser('pairs', help='pair a route run\'s frames with the footage')
    pa.add_argument('route')
    pa.add_argument('run')
    pa.add_argument('--rate', type=float, default=2.0)
    pa.add_argument('--frames', action='store_true')
    pa.add_argument('--size', type=int, nargs=2, default=(640, 360))
    pa.add_argument('--crop-bottom', type=float, default=0.0)
    pa.add_argument('--camera')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello(f'dashcam route {a.what}')
    try:
        if a.what == 'plan':
            win = window(a.clips, a.start, a.seconds, a.margin, a.max_side)
            d = a.out or default_route_dir(win)
            save_window(win, d)
            em.result('ok', summary={'route': d, 'fixes': len(win['fixes']), 'path_m': win['path_m'],
                                     'size_m': win['size_m'], 'holes': len(win['holes']),
                                     'uncovered': win['uncovered']})
            print(f'{len(win["clips"])} clip(s), {win["start_s"]:g}–{win["end_s"]:g} s: {len(win["fixes"])} fixes, '
                  f'{win["path_m"]:.0f} m; area {win["size_m"][0]:.0f} × {win["size_m"][1]:.0f} m → {d}\n'
                  f'next: falcon dashcam route import PROJECT {d} --osm FILE  (or --download-osm: asks first)',
                  file=sys.stderr)
            return P.OK
        if a.what == 'pairs':
            doc = pairs(a.route, a.run, a.rate, a.frames, tuple(a.size), a.crop_bottom, a.camera,
                        progress=lambda f, t: em.progress(f, t))
            em.result('ok', summary={k: doc[k] for k in ('pairs', 'camera', 'sim_minus_route_m')})
            return P.OK
        from .dashcam import world_geo
        from .paths import workspace_for
        from .project import Project
        proj = Project.load(a.project)
        proj_ws = workspace_for(a.project)
        win = load(a.route)
        if a.what == 'import':
            from .logreplay.commands import _wait
            from .worlds import world_dir
            wid = a.world_id or 'w-route-' + time.strftime('%Y%m%d%H%M%S')
            out = world_dir(proj_ws, wid)
            os.makedirs(out, exist_ok=True)
            source = a.osm
            if a.download_osm:
                q = (f'This sends the area {win["bbox"]} to {OVERPASS_URL} and downloads its roads '
                     f'(OpenStreetMap data, {OSM_LICENCE}). Download?')
                if not (a.yes or _confirm(q)):
                    print('not downloaded (needs your confirmation, or --yes)', file=sys.stderr)
                    em.result('failed', error='download not confirmed')
                    return P.FAILED
                got = fetch_overpass(win['bbox'], os.path.join(out, 'overpass.osm'), True)
                source = got['out']
                from .provenance import write_json
                write_json(os.path.join(out, 'download.json'), got)
            corr = corridor_osm(source, win, os.path.join(out, 'corridor.osm'), a.corridor,
                                progress=lambda f, t: em.progress(0.3 * f, t))
            from .provenance import write_json
            write_json(os.path.join(out, 'corridor.json'), corr)
            cfg = import_config(win, corr['out'], out, lht=not a.rht)
            svc = _service(a.fake_carla)
            try:
                job = svc.submit_osm_import(cfg, title='Import the dashcam route\'s roads')
                code = _wait(svc, job, em)
            finally:
                svc.shutdown()
            if code != P.OK:
                print('failed: the import did not finish; see ' + os.path.join(out, 'run.json'), file=sys.stderr)
                return code
            with open(os.path.join(out, 'import.json'), encoding='utf-8') as fh:
                rep = json.load(fh)
            add_world(proj, wid, a.name or f'Dashcam route {win["start_s"]:g}–{win["end_s"]:g} s', cfg, rep)
            proj.save()
            a.world = wid
        if not a.world and a.what == 'drive' and os.path.isfile(os.path.join(a.route, 'route.json')):
            a.world = load(a.route, 'route.json')['world']['id']            # the world the route was matched to
        world = world_geo(a.project, a.world or proj.doc['worlds'][0]['id'])
        if a.what in ('import', 'match'):
            route = build_route(win, world)
            from .provenance import write_json
            write_json(os.path.join(a.route, 'route.json'), route)
            em.result('ok', summary={'world': world['id'], 'gps_verdict': route['fit']['gps_against_lanes'].get(
                'verdict'), 'snapped_share': route['fit']['snapped_share'],
                'gps_to_lane_m': route['fit']['gps_to_lane_m']})
            print(route_text(route), file=sys.stderr)
            return P.OK
        # drive
        from .logreplay.commands import _wait, submit
        svc = _service(a.fake_carla)
        out = a.out or new_run_dir(proj_ws)
        try:
            cfg = drive_config(proj, a.route, world['id'], a.rig or proj.doc['rigs'][0]['id'], out, svc.port,
                               svc.tm_port, a.driver, a.fps, a.seed, float(svc.settings.get('render_budget_mpx')),
                               a.allow_mismatch)
            job = submit(svc, cfg, cfg['name'])
            code = _wait(svc, job, em)
        finally:
            if not a.keep_server:
                svc.shutdown()
        if code != P.OK:
            try:
                err = json.load(open(os.path.join(out, 'run.json'), encoding='utf-8')).get('error')
            except (OSError, ValueError):
                err = None
            em.result(P.status_of(code), error=err, run_json=os.path.join(out, 'run.json'))
            return code
        doc = pairs(a.route, out, a.pair_rate, a.frames)
        rp = json.load(open(os.path.join(out, 'replay.json'), encoding='utf-8'))
        em.result('ok', run_json=os.path.join(out, 'run.json'),
                  summary={'run': out, 'pairs': doc['pairs'], 'sim_minus_route_m': doc['sim_minus_route_m'],
                           'server_match': (rp.get('map_match') or {}).get('verdict')})
        return P.OK
    except (DashcamError, PermissionError, ValueError, OSError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
