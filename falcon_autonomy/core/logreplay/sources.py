"""Two more log sources: KITTI raw drives (the ego from OXTS) and Falcon's own runs (falcon-raw folders) — numpy and
the standard library.

KITTI raw (`<date>_drive_<n>_sync/oxts/`): `timestamps.txt` and one `data/<n>.txt` per frame of 30 numbers, of which
lat, lon, alt, roll, pitch, yaw (radians; yaw 0 = east, counter-clockwise; pitch positive nose down) are read, as the
dataset's dataformat.txt defines them. Positions go to a transverse Mercator about the first fix (core.osm's: x east,
y north), the orientation is R = Rz(yaw)·Ry(pitch)·Rx(roll) (the KITTI devkit's convertOxtsToPose), the OXTS unit's
frame x forward, y left, z up — all right-handed, to CARLA's frame by model.rh_to_carla. Only the ego: the raw drives'
tracklets come as a separate download and are not read. The log is geo-located (its first fix), so it can match an
imported OpenStreetMap world of the same place.

Falcon runs: frames.jsonl (the ego: CARLA's vehicle origin) and actors.jsonl (every vehicle and walker: the box's
centre is the actor's transform applied to its bounding box's centre; the class from its nuScenes category; its
blueprint kept so a replay spawns the same one). The map is the run's own CARLA world: its town, or its OpenDRIVE file
with that file's SHA-256 — a replay in the same world matches it by construction (and measures it all the same).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .model import Log, LogError, Track, rebase_time, rh_to_carla
from .ros2 import falcon_class


# ------------------------------------------------------------------------------------------------ KITTI raw
def import_kitti_raw(path: str) -> Log:
    from ..osm import local_projection
    from ..provenance import sha256_file
    path = os.path.abspath(path)
    ox = os.path.join(path, 'oxts') if os.path.isdir(os.path.join(path, 'oxts')) else path
    data_dir = os.path.join(ox, 'data')
    ts_path = os.path.join(ox, 'timestamps.txt')
    if not (os.path.isdir(data_dir) and os.path.isfile(ts_path)):
        raise LogError(f'{path}: no oxts/data and oxts/timestamps.txt (a KITTI raw drive folder)')
    import datetime as _dt
    stamps = []
    with open(ts_path, encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            day, _, frac = s.partition('.')
            t = _dt.datetime.strptime(day, '%Y-%m-%d %H:%M:%S').replace(tzinfo=_dt.timezone.utc).timestamp()
            stamps.append(t + float('0.' + frac) if frac else t)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.txt'))
    if len(files) != len(stamps):
        raise LogError(f'{path}: {len(files)} OXTS packets but {len(stamps)} timestamps')
    pk = np.array([[float(v) for v in open(os.path.join(data_dir, f), encoding='utf-8').read().split()[:30]]
                   for f in files])
    lat, lon, alt, roll, pitch, yaw = (pk[:, i] for i in range(6))
    proj = local_projection(float(lat[0]), float(lon[0]))
    e, n = proj.to_local(lat, lon)
    rows = []
    for k in range(len(pk)):
        cr, sr, cp, sp, cy, sy = (math.cos(roll[k]), math.sin(roll[k]), math.cos(pitch[k]), math.sin(pitch[k]),
                                  math.cos(yaw[k]), math.sin(yaw[k]))
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ \
            np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        rows.append((stamps[k],) + rh_to_carla(R, (e[k], n[k], alt[k] - alt[0])))
    r = np.array(rows)
    ego = Track('ego', 'car', r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 6], (4.6, 1.8, 1.5), 'ego', 'typical',
                pitch=r[:, 5], roll=r[:, 4])
    t0 = rebase_time(ego, [])
    # the OXTS forward speed against the speed of the positions: a check of the import, reported
    vf = pk[:, 8]
    sp = np.hypot(np.gradient(ego.x, ego.t), np.gradient(ego.y, ego.t)) if len(ego) > 2 else np.zeros(len(ego))
    check = {'median_abs_speed_diff_mps': round(float(np.median(np.abs(sp - vf))), 4) if len(ego) > 2 else None,
             'mean_oxts_speed_mps': round(float(np.mean(vf)), 4)}
    meta = {'name': f'KITTI raw {os.path.basename(path)}',
            'source': {'format': 'kitti_raw', 'path': path, 't0_s': t0, 'packets': len(files),
                       'files': [{'path': ts_path, 'sha256': sha256_file(ts_path)}],
                       'speed_check': check},
            'ego_origin': 'the OXTS unit',
            'map': {'kind': 'geo', 'name': None,
                    'geo': {'lat0': float(lat[0]), 'lon0': float(lon[0]), 'method': 'tmerc',
                            'source': 'the first OXTS fix'}},
            'notes': [f'speed from the positions against the OXTS forward speed: median |difference| '
                      f'{check["median_abs_speed_diff_mps"]} m/s'],
            'left_out': [{'what': 'actors', 'why': 'KITTI raw tracklets are a separate download, not read'},
                         {'what': 'images and velodyne points', 'why': 'a replay renders CARLA\'s own sensors'}],
            'heuristic': [{'what': 'ego box', 'how': 'a typical car, 4.6 × 1.8 × 1.5 m (the VW Passat\'s size is not '
                                                     'in the data)'}]}
    return Log(ego, [], meta)


# ------------------------------------------------------------------------------------------------ Falcon runs
def import_falcon_run(path: str) -> Log:
    from ..provenance import read_json, sha256_file
    from ..rig import carla_rotation
    path = os.path.abspath(path)
    fp, ap = os.path.join(path, 'frames.jsonl'), os.path.join(path, 'actors.jsonl')
    if not os.path.isfile(fp):
        raise LogError(f'{path}: no frames.jsonl (a falcon-raw run folder)')
    cfg = {}
    for name in ('drive.json', 'live.json'):
        if os.path.isfile(os.path.join(path, name)):
            cfg = read_json(os.path.join(path, name))
            break
    if not cfg and os.path.isfile(os.path.join(path, 'run.json')):
        cfg = (read_json(os.path.join(path, 'run.json')).get('provenance') or {}).get('settings') or {}
    with open(fp, encoding='utf-8') as fh:
        frames = [json.loads(x) for x in fh if x.strip()]
    if not frames:
        raise LogError(f'{path}: frames.jsonl is empty')
    e = np.array([[f['sim_t'], f['ego']['x'], f['ego']['y'], f['ego']['z'], f['ego']['yaw'], f['ego']['pitch'],
                   f['ego']['roll']] for f in frames])
    per, ego_box = {}, None
    files = [fp]
    if os.path.isfile(ap):
        files.append(ap)
        with open(ap, encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                for a in row['actors']:
                    R = carla_rotation(a['roll'], a['pitch'], a['yaw'])
                    c = R @ np.asarray(a.get('bbox_center') or (0, 0, 0), float) + [a['x'], a['y'], a['z']]
                    size = (2 * a['extent'][0], 2 * a['extent'][1], 2 * a['extent'][2])
                    if a.get('role') == 'hero':
                        ego_box = (size, a.get('type_id'), a.get('bbox_center'))
                        continue
                    k = str(a['id'])
                    if k not in per:
                        per[k] = {'cls': falcon_class(a.get('category')) or falcon_class(a.get('type_id')),
                                  'cat': a.get('category'), 'type_id': a.get('type_id'), 'size': size, 'rows': []}
                    per[k]['rows'].append((row['sim_t'], c[0], c[1], c[2], a['yaw'], a['pitch'], a['roll']))
    ego = Track('ego', 'car', e[:, 0], e[:, 1], e[:, 2], e[:, 3], e[:, 4],
                ego_box[0] if ego_box else None, 'ego', 'source' if ego_box else 'typical', pitch=e[:, 5],
                roll=e[:, 6], blueprint=(ego_box[1] if ego_box else (cfg.get('rig') or {}).get('vehicle')))
    actors, skipped = [], 0
    for k in sorted(per, key=lambda s: int(s) if s.isdigit() else s):
        d = per[k]
        if d['cls'] is None:
            skipped += 1
            continue
        rr = np.array(d['rows'])
        actors.append(Track(k, d['cls'], rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3], rr[:, 4], d['size'], d['cat'],
                            pitch=rr[:, 5], roll=rr[:, 6], blueprint=d['type_id']))
    t0 = rebase_time(ego, actors)
    town, xodr = cfg.get('town') or '', cfg.get('xodr') or ''
    mp = {'kind': 'carla', 'name': town or None}
    if xodr:
        mp.update(xodr=xodr, xodr_sha256=sha256_file(xodr) if os.path.isfile(xodr) else None,
                  handedness=cfg.get('handedness'))
    if not town and not xodr:
        mp = {'kind': 'unknown', 'note': 'the run folder has no drive.json (nor run.json settings) saying its world'}
    left = [] if os.path.isfile(ap) else [{'what': 'actors', 'why': 'the run has no actors.jsonl'}]
    if skipped:
        left.append({'what': f'{skipped} actors', 'why': 'their category names no Falcon class'})
    meta = {'name': f'Falcon run {os.path.basename(path)}',
            'source': {'format': 'falcon', 'path': path, 't0_s': t0,
                       'files': [{'path': f, 'sha256': sha256_file(f)} for f in files],
                       'run': (read_json(os.path.join(path, 'run.json')).get('name')
                               if os.path.isfile(os.path.join(path, 'run.json')) else None),
                       'weather': cfg.get('weather'), 'fps': cfg.get('fps')},
            'ego_origin': 'CARLA\'s vehicle origin', 'map': mp, 'notes': [], 'left_out': left, 'heuristic': []}
    return Log(ego, actors, meta)
