"""nuScenes scenes as logs, read from the release's JSON tables directly (no devkit; numpy and the standard library).

What is read (the tables of a release folder, v1.0-mini / v1.0-trainval / v1.0-test or another v1.0-* version):

* scene → its samples (key frames, 2 Hz) and its log (the map's name: boston-seaport, singapore-onenorth, …);
* **the ego**: the ego_pose of every sample_data of one channel (LIDAR_TOP by default, about 20 Hz with its sweeps)
  whose sample is in the scene; ego_pose.translation and .rotation (w, x, y, z) are the ego frame (the rear axle's
  midpoint, on the ground) in the map's global frame;
* **the actors**: sample_annotation per instance (one per key frame it is seen in): translation (the box's centre,
  global), size (width, length, height), rotation; the category through instance → category, mapped to Falcon's
  classes (ros2.falcon_class: static objects, cones, barriers, debris, bicycle racks and animals are left out and
  counted). A replay interpolates between key frames as the devkit's NuScenes.get_boxes does for sweeps: the centre
  linearly, the rotation by slerp (yaw along the shorter arc), between annotations at most max_gap_s apart (1.05 s:
  one missed key frame — heuristic).

Frames: nuScenes' global frame is right-handed, x east and y north, z up, metres; it goes to CARLA's by
model.rh_to_carla. The map: nuScenes maps are not CARLA towns. Its reference corner (the devkit's REFERENCE_COORDINATES,
approximate by the devkit's own account) geo-locates the log, so a replay can match it with an imported OpenStreetMap
world of the same place; otherwise it replays without a matching road and says so. The global frame's metres are
kept: a transverse Mercator (WGS84) about the corner. The devkit's own export (export_poses.py, a great circle on a
sphere of the equatorial radius; model.nuscenes_latlon reproduces it exactly) stretches east-west distances by about
0.15 % at Boston — 1.5 m a kilometre from the corner — so the two disagree by that much.

Not read: images, lidar and radar data (a replay renders CARLA's own sensors), attributes, visibility, lidarseg.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .model import NUSCENES_ORIGINS, Log, LogError, Track, quat_matrix, rebase_time, rh_to_carla
from .ros2 import falcon_class

NUSC_MAX_GAP_S = 1.05


def find_version(path: str) -> str:
    for v in sorted(os.listdir(path)):
        if v.startswith('v1.0-') and os.path.isfile(os.path.join(path, v, 'scene.json')):
            return v
    raise LogError(f'{path}: no v1.0-*/scene.json (a nuScenes release folder)')


def scenes(path: str) -> list:
    """[(token, name, description, samples)] of a release."""
    v = find_version(path)
    with open(os.path.join(path, v, 'scene.json'), encoding='utf-8') as fh:
        return [(s['token'], s['name'], s.get('description', ''), s.get('nbr_samples')) for s in json.load(fh)]


def import_nuscenes(path: str, scene: str | None = None, channel: str = 'LIDAR_TOP') -> Log:
    """A nuScenes scene (by name or token; the first when None) as a Log."""
    from ..provenance import sha256_file
    path = os.path.abspath(path)
    v = find_version(path)
    files = {}

    def tab(name, required=True):
        p = os.path.join(path, v, name + '.json')
        if not os.path.isfile(p):
            if required:
                raise LogError(f'{path}/{v}: no {name}.json')
            return []
        files[name] = p
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    sc_all = tab('scene')
    if not sc_all:
        raise LogError(f'{path}/{v}: no scenes')
    sc = next((s for s in sc_all if scene in (s['name'], s['token'])), None) if scene else sc_all[0]
    if sc is None:
        raise LogError(f'no scene {scene!r} (the release has {", ".join(s["name"] for s in sc_all[:20])}'
                       f'{"…" if len(sc_all) > 20 else ""})')
    logs = {r['token']: r for r in tab('log')}
    location = (logs.get(sc['log_token']) or {}).get('location', '')
    samples = {s['token']: s for s in tab('sample') if s['scene_token'] == sc['token']}
    sensors = {s['token']: s['channel'] for s in tab('sensor')}
    cs = {c['token']: sensors.get(c['sensor_token']) for c in tab('calibrated_sensor')}
    sds = [d for d in tab('sample_data') if d['sample_token'] in samples]
    by_ch = {}
    for d in sds:
        by_ch.setdefault(cs.get(d['calibrated_sensor_token']), []).append(d)
    if channel not in by_ch:
        raise LogError(f'scene {sc["name"]}: no {channel} data (channels: {", ".join(sorted(c for c in by_ch if c))})')
    poses = {e['token']: e for e in tab('ego_pose')}
    rows = []
    for d in sorted(by_ch[channel], key=lambda d: d['timestamp']):
        e = poses.get(d['ego_pose_token'])
        if e is None:
            continue
        x, y, z, roll, pitch, yaw = rh_to_carla(quat_matrix(*e['rotation']), e['translation'])
        rows.append((d['timestamp'] * 1e-6, x, y, z, yaw, pitch, roll))
    if not rows:
        raise LogError(f'scene {sc["name"]}: no ego poses')
    r = np.array(rows)
    _, uniq = np.unique(r[:, 0], return_index=True)
    r = r[uniq]
    ego = Track('ego', 'car', r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4], None, 'ego', pitch=r[:, 5], roll=r[:, 6])
    cats = {c['token']: c['name'] for c in tab('category')}
    inst = {i['token']: cats.get(i['category_token'], '') for i in tab('instance')}
    per, skipped = {}, {}
    for a in tab('sample_annotation'):
        s = samples.get(a['sample_token'])
        if s is None:
            continue
        cat = inst.get(a['instance_token'], '')
        fc = falcon_class(cat)
        if fc is None:
            skipped[cat] = skipped.get(cat, 0) + 1
            continue
        x, y, z, roll, pitch, yaw = rh_to_carla(quat_matrix(*a['rotation']), a['translation'])
        w, l, h = a['size']
        per.setdefault(a['instance_token'], {'cls': fc, 'cat': cat, 'rows': [], 'size': (l, w, h)})
        per[a['instance_token']]['rows'].append((s['timestamp'] * 1e-6, x, y, z, yaw, pitch, roll))
    actors = []
    for tok in sorted(per):
        d = per[tok]
        rr = np.array(sorted(d['rows']))
        actors.append(Track(tok, d['cls'], rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3], rr[:, 4], d['size'], d['cat'],
                            'source', NUSC_MAX_GAP_S, pitch=rr[:, 5], roll=rr[:, 6]))
    t0 = rebase_time(ego, actors)
    geo = {'location': location, 'method': 'tmerc',
           'lat0': NUSCENES_ORIGINS[location][0], 'lon0': NUSCENES_ORIGINS[location][1],
           'accuracy': 'the devkit\'s reference corner (export_poses.py), approximate by the devkit\'s own account; '
                       'positions keep their metres (a transverse Mercator about the corner), where the devkit\'s own '
                       'spherical conversion stretches east-west distances by about 0.15 % at Boston'} \
        if location in NUSCENES_ORIGINS else None
    left = [{'what': f'{n} annotations of {c or "no category"}', 'why': 'not a Falcon actor class (static objects, '
                                                                        'cones, barriers, debris, racks, animals)'}
            for c, n in sorted(skipped.items())]
    left.append({'what': 'images, lidar and radar data', 'why': 'a replay renders CARLA\'s own sensors'})
    left.append({'what': 'attributes and visibility', 'why': 'not used by the replay'})
    meta = {'name': f'nuScenes {sc["name"]}',
            'source': {'format': 'nuscenes', 'path': path, 'version': v, 'scene': sc['name'],
                       'scene_token': sc['token'], 'description': sc.get('description', ''), 'channel': channel,
                       't0_s': t0,
                       'files': [{'path': p, 'sha256': sha256_file(p)} for _, p in sorted(files.items())]},
            'ego_origin': 'nuScenes\' ego frame: the rear axle\'s midpoint, on the ground',
            'map': {'kind': 'nuscenes', 'name': location, 'geo': geo,
                    'note': 'a nuScenes map is not a CARLA town; with its geographic reference a replay can match an '
                            'imported OpenStreetMap world of the same place'},
            'notes': [f'{len(by_ch[channel])} {channel} ego poses, {len(samples)} key frames'],
            'left_out': left,
            'heuristic': [{'what': 'interpolation gap', 'how': f'actors are interpolated between annotations at most '
                                                               f'{NUSC_MAX_GAP_S} s apart (one missed key frame)'},
                          {'what': 'ego box', 'how': 'a typical car (nuScenes does not give the ego\'s box)'}]}
    return Log(ego, actors, meta)
