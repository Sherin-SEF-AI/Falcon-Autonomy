"""Waymo Open Dataset (Perception) v2 segments as logs: the v2 format's Apache Parquet components, read with pyarrow —
no waymo-open-dataset or TensorFlow package.

The layout (the dataset's v2 documentation): a folder per component, one Parquet file per segment named after its
context name, `<root>/<component>/<segment context name>.parquet`. Columns are the component's fields flattened with
dots: keys as `key.segment_context_name`, `key.frame_timestamp_micros`, `key.laser_object_id`; values as
`[<ComponentName>].<field>.<subfield>`. What is read:

* vehicle_pose — `[VehiclePoseComponent].world_from_vehicle.transform`: the vehicle frame's pose in the world frame,
  16 numbers row-major, per frame (10 Hz): **the ego**;
* lidar_box — per frame and laser object id, `[LiDARBoxComponent].box.center.{x,y,z}`, `.box.size.{x,y,z}` (length,
  width, height), `.box.heading` (radians about the vehicle's z axis) and `.type` (1 vehicle, 2 pedestrian, 3 sign,
  4 cyclist), in the vehicle frame of that frame: **the actors**, taken to the world frame by that frame's pose;
* stats (optional) — `[StatsComponent].location`, `.time_of_day`, `.weather`: what the segment is.

camera_box holds 2D image boxes without a 3D pose: counted and left out. Columns are found by their names' endings,
so the bracketed component prefix may vary; a component without the columns above is refused, listing what it has.

Classes: vehicle → car, or bus / truck by length (≥ 10 m bus, ≥ 6.5 m truck: heuristic — Waymo has one vehicle
class); pedestrian; cyclist → bicycle; signs are left out. Frames: Waymo's world and vehicle frames are right-handed,
z up; poses go to CARLA's through model.rh_to_carla. The map: Waymo publishes no geographic reference, so a Waymo
log replays without a matching road (its location, e.g. location_sf, is recorded).

Not verified on real Waymo files: none are on this machine and Falcon downloads no datasets. The column names follow
the published v2 component definitions; the tests read files written in that layout. The Waymo Open Dataset licence
forbids redistributing its data: nothing of it is copied into a log beyond the trajectories the user's own import
makes, kept in the user's workspace.

v1 (TFRecord files of Frame protocol buffers) is not supported: reading it needs the dataset's .proto definitions,
which are not on this machine, and decoding them without them could not be verified.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np

from .model import Log, LogError, Track, rebase_time, rh_to_carla

POSE_COL = 'world_from_vehicle.transform'
SEG_COL, TS_COL, OBJ_COL = 'key.segment_context_name', 'key.frame_timestamp_micros', 'key.laser_object_id'
BOX_COLS = {'cx': 'box.center.x', 'cy': 'box.center.y', 'cz': 'box.center.z', 'l': 'box.size.x', 'w': 'box.size.y',
            'h': 'box.size.z', 'heading': 'box.heading', 'type': '.type'}
WAYMO_TYPES = {1: 'vehicle', 2: 'pedestrian', 3: 'sign', 4: 'cyclist'}


def _col(names, ending):
    hits = [n for n in names if n.endswith(ending)]
    return hits[0] if hits else None


def segments(root: str) -> list:
    return sorted(os.path.basename(p)[:-len('.parquet')] for p in glob.glob(os.path.join(root, 'vehicle_pose',
                                                                                        '*.parquet')))


def _read(root, comp, seg):
    import pyarrow.parquet as pq
    p = os.path.join(root, comp, f'{seg}.parquet')
    if not os.path.isfile(p):
        return None, None
    return pq.read_table(p), p


def vehicle_class(type_id: int, length: float):
    k = WAYMO_TYPES.get(int(type_id))
    if k == 'pedestrian':
        return 'pedestrian'
    if k == 'cyclist':
        return 'bicycle'
    if k == 'vehicle':
        return 'bus' if length >= 10.0 else ('truck' if length >= 6.5 else 'car')
    return None


def import_waymo(root: str, segment: str | None = None) -> Log:
    """A Waymo v2 segment (by context name; the only or first one when None) as a Log."""
    from ..provenance import sha256_file
    root = os.path.abspath(root)
    segs = segments(root)
    if not segs:
        raise LogError(f'{root}: no vehicle_pose/*.parquet (a Waymo Open Dataset v2 folder)')
    seg = segment or segs[0]
    if seg not in segs:
        raise LogError(f'no segment {seg!r} in {root} ({", ".join(segs[:10])})')
    files = []
    tab, p = _read(root, 'vehicle_pose', seg)
    files.append(p)
    names = tab.column_names
    pc, tc = _col(names, POSE_COL), _col(names, TS_COL)
    if pc is None or tc is None:
        raise LogError(f'{p}: no {TS_COL} and *{POSE_COL} columns (it has {", ".join(names)})')
    ts = np.asarray(tab.column(tc).to_pylist(), np.int64)
    Ts = [np.asarray(v, float).reshape(4, 4) for v in tab.column(pc).to_pylist()]
    order = np.argsort(ts)
    ts, Ts = ts[order], [Ts[i] for i in order]
    pose_by_ts = dict(zip(ts.tolist(), Ts))
    e = np.array([rh_to_carla(T[:3, :3], T[:3, 3]) for T in Ts])
    ego = Track('ego', 'car', ts * 1e-6, e[:, 0], e[:, 1], e[:, 2], e[:, 5], None, 'ego', pitch=e[:, 4], roll=e[:, 3])
    left, heur, notes = [], [], []
    actors = []
    btab, bp = _read(root, 'lidar_box', seg)
    if btab is None:
        left.append({'what': 'actors', 'why': f'no lidar_box/{seg}.parquet'})
    else:
        files.append(bp)
        bn = btab.column_names
        cols = {k: _col(bn, v) for k, v in BOX_COLS.items()}
        oc, btc = _col(bn, OBJ_COL), _col(bn, TS_COL)
        missing = [BOX_COLS[k] for k, c in cols.items() if c is None] + [x for x, c in ((OBJ_COL, oc), (TS_COL, btc))
                                                                           if c is None]
        if missing:
            raise LogError(f'{bp}: no columns ending {", ".join(missing)} (it has {", ".join(bn)})')
        d = {k: np.asarray(btab.column(c).to_pylist()) for k, c in cols.items()}
        oid = btab.column(oc).to_pylist()
        bts = np.asarray(btab.column(btc).to_pylist(), np.int64)
        per, signs, no_pose, by_len = {}, 0, 0, 0
        for i in range(len(oid)):
            T = pose_by_ts.get(int(bts[i]))
            if T is None:
                no_pose += 1
                continue
            L = float(d['l'][i])
            cls = vehicle_class(d['type'][i], L)
            if cls is None:
                signs += 1
                continue
            if WAYMO_TYPES.get(int(d['type'][i])) == 'vehicle' and cls != 'car':
                by_len += 1
            c, s = math.cos(float(d['heading'][i])), math.sin(float(d['heading'][i]))
            B = np.eye(4)
            B[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            B[:3, 3] = [d['cx'][i], d['cy'][i], d['cz'][i]]
            W = T @ B
            x, y, z, roll, pitch, yaw = rh_to_carla(W[:3, :3], W[:3, 3])
            per.setdefault(str(oid[i]), {'cls': [], 'rows': [], 'size': [], 'type': int(d['type'][i])})
            per[str(oid[i])]['rows'].append((bts[i] * 1e-6, x, y, z, yaw, pitch, roll))
            per[str(oid[i])]['size'].append((L, float(d['w'][i]), float(d['h'][i])))
            per[str(oid[i])]['cls'].append(cls)
        for k in sorted(per):
            v = per[k]
            rr = np.array(sorted(v['rows']))
            _, uniq = np.unique(rr[:, 0], return_index=True)
            rr = rr[uniq]
            cls = max(set(v['cls']), key=v['cls'].count)
            actors.append(Track(k, cls, rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3], rr[:, 4],
                                tuple(np.median(np.array(v['size']), axis=0)), f'waymo {WAYMO_TYPES.get(v["type"])}',
                                pitch=rr[:, 5], roll=rr[:, 6]))
        if signs:
            left.append({'what': f'{signs} sign boxes', 'why': 'not an actor'})
        if no_pose:
            left.append({'what': f'{no_pose} boxes', 'why': 'their frame has no vehicle pose'})
        if by_len:
            heur.append({'what': f'bus or truck for {by_len} vehicle boxes',
                         'how': 'by length (≥ 10 m bus, ≥ 6.5 m truck): Waymo has one vehicle class'})
    ctab, cp = _read(root, 'camera_box', seg)
    if ctab is not None:
        files.append(cp)
        left.append({'what': f'{ctab.num_rows} camera boxes', 'why': '2D image boxes carry no 3D pose'})
    stats = {}
    stab, sp = _read(root, 'stats', seg)
    if stab is not None:
        files.append(sp)
        for k in ('location', 'time_of_day', 'weather'):
            c = _col(stab.column_names, '.' + k)
            if c is not None:
                vals = [v for v in stab.column(c).to_pylist() if v]
                if vals:
                    stats[k] = max(set(vals), key=vals.count)
    t0 = rebase_time(ego, actors)
    if stats:
        notes.append('segment: ' + ', '.join(f'{k} {v}' for k, v in stats.items()))
    left.append({'what': 'images, lidar returns, maps', 'why': 'a replay renders CARLA\'s own sensors'})
    meta = {'name': f'Waymo {seg}',
            'source': {'format': 'waymo_v2', 'path': root, 'segment': seg, 'stats': stats, 't0_s': t0,
                       'files': [{'path': f, 'sha256': sha256_file(f)} for f in files]},
            'ego_origin': 'the origin of Waymo\'s vehicle frame',
            'map': {'kind': 'waymo', 'name': stats.get('location'), 'geo': None,
                    'note': 'Waymo publishes no geographic reference: no CARLA world can be matched to it'},
            'notes': notes, 'left_out': left,
            'heuristic': heur + [{'what': 'ego box', 'how': 'a typical car (the segment does not give it)'}],
            'licence': 'Waymo Open Dataset licence (non-commercial): the data stays the user\'s; Falcon does not '
                       'redistribute it'}
    return Log(ego, actors, meta)
