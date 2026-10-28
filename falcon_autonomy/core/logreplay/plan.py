"""The replay plan: a (changed) log resampled to the replay's fixed step and placed in the target world — what the
replay worker executes (numpy; schema falcon.replay_plan/1).

Placement — where the log's coordinates go in the world, decided from the log's map block and the world:

* 'matched' — the log is in this very world: a Falcon run (or an export of one) in the same CARLA town (the `_Opt`
  variants share their towns' roads) or the same OpenDRIVE file (by SHA-256). Coordinates are used as they are.
* 'geo' — the log is geo-located (nuScenes with the devkit's reference corner, a NavSatFix bag, KITTI's OXTS) and
  the world is an imported OpenStreetMap / OpenDRIVE world with a geographic origin, and the log's area lies inside the
  world's: every position goes through latitude and longitude into the world's own projection (headings turned by
  the local difference between the two projections' north).
* 'anchored' — anything else: **no matching road**. The log is moved rigidly so the ego's first pose sits on an
  anchor (a spawn point of the world, by index, or a given pose) and replayed there; the run says so everywhere.

Whatever the placement, the replay worker measures the match against the server's own lanes (mapmatch) and records
it; a 'matched' or 'geo' placement whose measured verdict is 'none' is refused unless allowed.

Samples. Steps i = 0 … steps − 1 at t = t_start + i / fps of the (changed) log's time. Each actor is sampled as
model.Track.sample says (linear, yaw along the shorter arc, absent outside its samples unless held); its velocity is
the central difference of its own samples on the replay's grid (one-sided at the ends of each observed run). The
ego's pose likewise; its path (every metre, with the logged speed there) is what a re-driving policy follows.

Positions of actors are box centres; the ego's is its log's ego origin (meta 'ego_origin'), placed at CARLA's vehicle
origin of the ego vehicle — an offset the run states (ego_offset_m moves it forward along the ego's heading).
"""
from __future__ import annotations

import math
import os
import re

import numpy as np

from .model import Log, LogError, latlon_of, wrap_deg

PLAN_SCHEMA = 'falcon.replay_plan/1'
DRIVERS = ('log', 'follower', 'plugin')


def town_key(name) -> str:
    return re.sub(r'_Opt$', '', str(name or '').rpartition('/')[2])


def world_info(project, world_id: str, workspace: str = '') -> dict:
    """What placement needs to know of a project's world: kind, town or OpenDRIVE file (and its hash), geographic
    origin and area."""
    from ..provenance import sha256_file
    w = project.item('worlds', world_id)
    if w is None:
        raise LogError(f'no world {world_id!r} in the project')
    info = {'id': world_id, 'name': w.get('name'), 'kind': w['kind'], 'town': w.get('town') or None, 'xodr': None,
            'xodr_sha256': None, 'geo': None, 'bbox': None}
    if w['kind'] != 'town' and w.get('xodr'):
        ws = workspace or project.workspace
        p = os.path.join(ws, w['xodr']) if ws else w['xodr']
        info['xodr'] = p
        if os.path.isfile(p):
            info['xodr_sha256'] = sha256_file(p)
            with open(p, encoding='utf-8') as fh:
                head = fh.read(20000)
            m0, m1 = re.search(r'\+lat_0=([-\d.eE]+)', head), re.search(r'\+lon_0=([-\d.eE]+)', head)
            if m0 and m1 and '+proj=tmerc' in head:
                info['geo'] = {'lat0': float(m0.group(1)), 'lon0': float(m1.group(1)), 'source': 'geoReference'}
        osm = w.get('osm') or {}
        if osm.get('centre'):
            info['geo'] = {'lat0': float(osm['centre'][0]), 'lon0': float(osm['centre'][1]), 'source': 'import centre'}
        if osm.get('bbox'):
            info['bbox'] = list(osm['bbox'])
    return info


def resolve_placement(log: Log, world: dict, anchor: dict | None = None) -> dict:
    """{'mode', 'why', 'anchor'} — see the module docstring."""
    mp = log.map
    if mp.get('kind') == 'carla':
        if world.get('kind') == 'town' and mp.get('name') and town_key(mp['name']) == town_key(world.get('town')):
            return {'mode': 'matched', 'why': f'the log was recorded in {mp["name"]}, this world\'s town'}
        if mp.get('xodr_sha256') and mp.get('xodr_sha256') == world.get('xodr_sha256'):
            return {'mode': 'matched', 'why': 'the log was recorded in this world\'s OpenDRIVE file (same SHA-256)'}
    geo = mp.get('geo')
    if geo and world.get('geo'):
        lat, lon = latlon_of(geo, log.ego.x, log.ego.y)
        bb = world.get('bbox')
        if bb:
            # bbox as core.osm: (south, west, north, east)
            s, w_, n, e = bb
            inside = float(np.mean((lat >= s) & (lat <= n) & (lon >= w_) & (lon <= e)))
            if inside < 0.99:
                return {'mode': 'anchored', 'anchor': anchor or {'spawn_index': 0},
                        'why': f'no matching road: only {100 * inside:.0f} % of the ego\'s positions lie inside the '
                               f'world\'s area'}
        return {'mode': 'geo', 'why': f'the log is geo-located ({geo.get("method")}) and the world has a geographic '
                                      f'origin ({world["geo"]["source"]})'}
    why = {'nuscenes': f'no matching road: the log is in the nuScenes map {mp.get("name")!r}, which is not a CARLA '
                       'town (import an OpenStreetMap world of the place to match it)',
           'waymo': 'no matching road: Waymo publishes no geographic reference for its segments',
           'carla': f'no matching road: the log was recorded in {mp.get("name") or "another OpenDRIVE world"}, not in '
                    f'this world',
           'geo': 'no matching road: the world has no geographic origin (a CARLA town)'}.get(
        mp.get('kind'), 'no matching road: the log does not say which map it is in')
    return {'mode': 'anchored', 'anchor': anchor or {'spawn_index': 0}, 'why': why}


def geo_transform(log: Log, world_geo: dict) -> Log:
    """A copy of the log in the world's projection (positions through latitude and longitude, headings turned by the
    two projections' difference of north)."""
    from ..osm import local_projection
    out = log.copy()
    proj = local_projection(world_geo['lat0'], world_geo['lon0'])
    geo = log.map['geo']
    for tr in [out.ego] + out.actors:
        lat, lon = latlon_of(geo, tr.x, tr.y)
        e, n = proj.to_local(lat, lon)
        a = np.radians(tr.yaw)
        lat2, lon2 = latlon_of(geo, tr.x + np.cos(a), tr.y + np.sin(a))
        e2, n2 = proj.to_local(lat2, lon2)
        tr.x, tr.y = np.asarray(e, float), -np.asarray(n, float)
        tr.yaw = wrap_deg(np.degrees(np.arctan2(-(np.asarray(n2) - n), np.asarray(e2) - e)))
    out.meta['map'] = dict(log.map, placed_in={'lat0': world_geo['lat0'], 'lon0': world_geo['lon0']})
    return out


def _dense(tr, tq):
    ok, x, y, z, yaw = tr.sample(tq)
    pitch = np.interp(tq, tr.t, tr.pitch) if tr.pitch is not None and len(tr.t) > 0 else np.zeros(len(tq))
    roll = np.interp(tq, tr.t, tr.roll) if tr.roll is not None and len(tr.t) > 0 else np.zeros(len(tq))
    return ok, x, y, z, yaw, pitch, roll


def _velocity(ok, x, y, dt):
    vx, vy = np.full(len(x), np.nan), np.full(len(x), np.nan)
    i = 0
    n = len(x)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        if j > i:
            vx[i:j + 1] = np.gradient(x[i:j + 1], dt)
            vy[i:j + 1] = np.gradient(y[i:j + 1], dt)
        else:
            vx[i] = vy[i] = 0.0
        i = j + 1
    return vx, vy


def _list(a, ok=None, n=6):
    return [None if (ok is not None and not ok[k]) or not math.isfinite(v) else round(float(v), n) + 0.0
            for k, v in enumerate(a)]


def make_plan(log: Log, world: dict, *, fps: float = 20.0, changes=None, seed: int = 0, driver: str = 'log',
              policy: str | None = None, t_start: float | None = None, t_end: float | None = None,
              anchor: dict | None = None, ego_offset_m: float = 0.0, allow_mismatch: bool = False,
              log_path: str = '') -> dict:
    """The replay plan of a log (with changes) in a world (world_info()). driver: 'log' (the ego replays its logged
    trajectory), 'follower' (the built-in lane follower drives it along the logged path at the logged speeds,
    braking for what is in its corridor) or 'plugin' (policy 'module:Class', the same observation as the
    follower's)."""
    from . import changes as CH
    if driver not in DRIVERS:
        raise LogError(f'driver must be one of {", ".join(DRIVERS)}')
    if driver == 'plugin' and not policy:
        raise LogError('a plugin driver needs a policy "module:Class"')
    changed, weather, done = CH.apply(log, changes or [], seed)
    placement = resolve_placement(changed, world, anchor)
    if placement['mode'] == 'geo':
        changed = geo_transform(changed, world['geo'])
    ego = changed.ego
    a0, a1 = ego.span
    t0 = a0 if t_start is None else max(float(t_start), a0)
    t1 = a1 if t_end is None else min(float(t_end), a1)
    dt = 1.0 / float(fps)
    steps = int(math.floor((t1 - t0) / dt + 1e-9)) + 1
    if steps < 2:
        raise LogError(f'the replay window {t0:.3f}–{t1:.3f} s is shorter than two steps')
    tq = t0 + dt * np.arange(steps)
    ok, x, y, z, yaw, pitch, roll = _dense(ego, tq)
    if not ok.all():
        bad = tq[~ok]
        raise LogError(f'the logged ego has gaps longer than {ego.max_gap_s:.2f} s (first at {bad[0]:.2f} s): '
                       'replay a window without them (t_start / t_end)')
    if ego_offset_m:
        a = np.radians(yaw)
        x, y = x + ego_offset_m * np.cos(a), y + ego_offset_m * np.sin(a)
    vx, vy = _velocity(ok, x, y, dt)
    speed = np.hypot(vx, vy)
    # the ego's path every metre with the logged speed there: the route a re-driving policy follows
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    sg = np.arange(0.0, s[-1] + 1e-9, 1.0) if s[-1] >= 1.0 else np.array([0.0])
    path = {'x': _list(np.interp(sg, s, x), n=4), 'y': _list(np.interp(sg, s, y), n=4),
            'v': _list(np.interp(sg, s, speed), n=4), 'step_m': 1.0}
    actors = []
    for tr in changed.actors:
        aok, ax, ay, az, ayaw, ap, ar = _dense(tr, tq)
        if not aok.any():
            continue
        avx, avy = _velocity(aok, ax, ay, dt)
        i0, i1 = int(np.flatnonzero(aok)[0]), int(np.flatnonzero(aok)[-1]) + 1
        sl = slice(i0, i1)
        actors.append({'id': tr.id, 'class': tr.cls, 'source_class': tr.source_class,
                       'size': [round(v, 4) for v in tr.size], 'size_source': tr.size_source,
                       'blueprint': tr.blueprint, 'i0': i0,
                       'x': _list(ax[sl], aok[sl]), 'y': _list(ay[sl], aok[sl]), 'z': _list(az[sl], aok[sl]),
                       'yaw': _list(ayaw[sl], aok[sl]), 'pitch': _list(ap[sl], aok[sl], 4),
                       'roll': _list(ar[sl], aok[sl], 4), 'vx': _list(avx[sl], aok[sl], 4),
                       'vy': _list(avy[sl], aok[sl], 4)})
    wfields = None
    if weather:
        from ..weather import PRESETS
        if weather.get('preset') and weather['preset'] not in PRESETS:
            raise LogError(f'weather change: unknown preset {weather["preset"]!r}')
        wfields = weather
    from ..provenance import sha256_file
    z_mode = 'log' if placement['mode'] == 'matched' else 'ground'
    plan = {'schema': PLAN_SCHEMA, 'fps': float(fps), 'dt': dt, 'steps': steps, 't_start': round(t0, 6),
            'seed': int(seed), 'placement': dict(placement, z=z_mode, allow_mismatch=bool(allow_mismatch)),
            'world': {k: world.get(k) for k in ('id', 'name', 'kind', 'town', 'xodr', 'xodr_sha256', 'geo')},
            'log': {'name': log.meta.get('name'), 'path': log_path,
                    'sha256': sha256_file(log_path) if log_path and os.path.isfile(log_path) else None,
                    'source': {k: (log.meta.get('source') or {}).get(k) for k in ('format', 'path', 'scene',
                                                                                  'segment')},
                    'map': log.map, 'ego_origin': log.meta.get('ego_origin')},
            'changes': done, 'weather': wfields,
            'ego': {'driver': driver, 'policy': policy, 'blueprint': ego.blueprint, 'size': list(ego.size),
                    'offset_m': float(ego_offset_m),
                    'x': _list(x), 'y': _list(y), 'z': _list(z), 'yaw': _list(yaw), 'pitch': _list(pitch, n=4),
                    'roll': _list(roll, n=4), 'vx': _list(vx, n=4), 'vy': _list(vy, n=4), 'path': path},
            'actors': actors}
    return plan


def rigid_anchor(plan: dict, ax: float, ay: float, ayaw: float) -> dict:
    """The plan moved rigidly (in place) so the ego's first pose is (ax, ay, ayaw); returns the transform applied."""
    ex, ey, eyaw = plan['ego']['x'][0], plan['ego']['y'][0], plan['ego']['yaw'][0]
    d = math.radians(ayaw - eyaw)
    c, s = math.cos(d), math.sin(d)

    def move(xs, ys):
        out_x, out_y = [], []
        for x, y in zip(xs, ys):
            if x is None or y is None:
                out_x.append(None)
                out_y.append(None)
                continue
            dx, dy = x - ex, y - ey
            out_x.append(ax + c * dx - s * dy)
            out_y.append(ay + s * dx + c * dy)
        return out_x, out_y

    def turn(vs):
        return [None if v is None else float(wrap_deg(v + math.degrees(d))) for v in vs]

    def rot(vx, vy):
        return ([None if a is None else c * a - s * b for a, b in zip(vx, vy)],
                [None if b is None else s * a + c * b for a, b in zip(vx, vy)])
    for item in [plan['ego']] + plan['actors']:
        item['x'], item['y'] = move(item['x'], item['y'])
        item['yaw'] = turn(item['yaw'])
        item['vx'], item['vy'] = rot(item['vx'], item['vy'])
    p = plan['ego']['path']
    p['x'], p['y'] = move(p['x'], p['y'])
    return {'from': [ex, ey, eyaw], 'to': [ax, ay, ayaw], 'rotation_deg': math.degrees(d)}
