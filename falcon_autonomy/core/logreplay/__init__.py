"""Log replay (milestone 7): real logs imported as trajectories, rebuilt in CARLA, re-simulated with changes ("what
if"), and the original and changed replays compared. docs/log-replay.md is the manual.

    log = import_log('nuscenes', '~/data/nuscenes', scene='scene-0061')     # → model.Log
    log.save(workspace + '/logs/scene-0061')
    plan = make_plan(log, world_info(project, 'world-town10'), changes=[...], seed=3, driver='follower')
    cfg = replay_config(project, plan, 'world-town10', 'rig-default', out_dir, port, tm_port)   # commands.py
    compare(original_run, [changed_run])                                                     # compare.py

Modules: model (the log, frames, geography), cdr and ros2 (ROS 2 bags: MCAP and sqlite3, this package's own CDR
decoder), nuscenes (the JSON tables), waymo (v2 Parquet), sources (KITTI raw OXTS, Falcon runs), changes (the what-if
list), plan (placement and the replay plan), mapmatch (does the log run on the world's lanes), compare, commands (the
`falcon replay` command and the job). The replay itself is workers.replay (CARLA's Python).
"""
from __future__ import annotations

import os

from .model import CLASSES, LOG_SCHEMA, Log, LogError, Track  # noqa: F401
from .plan import make_plan, resolve_placement, world_info  # noqa: F401

KINDS = ('nuscenes', 'waymo', 'ros2', 'kitti_raw', 'falcon')


def detect_kind(path: str) -> str | None:
    """The kind of log a path holds, from its layout; None when it is none of KINDS."""
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(p):
        return 'ros2' if p.endswith(('.mcap', '.db3')) else None
    if not os.path.isdir(p):
        return None
    names = set(os.listdir(p))
    if 'frames.jsonl' in names:
        return 'falcon'
    if 'metadata.yaml' in names or any(n.endswith(('.mcap', '.db3')) for n in names):
        return 'ros2'
    if 'vehicle_pose' in names:
        return 'waymo'
    if any(n.startswith('v1.0-') for n in names):
        return 'nuscenes'
    if 'oxts' in names or ('data' in names and 'timestamps.txt' in names):
        return 'kitti_raw'
    return None


def falcon_export_town(bag_path: str) -> dict | None:
    """The map block of a bag Falcon exported (core.exporters 'mcap'): its manifest, falcon_export.json, one or two
    folders up, names the run's town or OpenDRIVE file."""
    import json
    d = os.path.abspath(bag_path)
    if os.path.isfile(d):
        d = os.path.dirname(d)
    key = os.path.basename(d)
    for up in (os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        m = os.path.join(up, 'falcon_export.json')
        if not os.path.isfile(m):
            continue
        with open(m, encoding='utf-8') as fh:
            doc = json.load(fh)
        for r in doc.get('runs') or []:
            if r.get('key') == key and r.get('town'):
                t = r['town']
                if t.endswith('.xodr'):
                    from ..provenance import sha256_file
                    return {'kind': 'carla', 'name': None, 'xodr': t,
                            'xodr_sha256': sha256_file(t) if os.path.isfile(t) else None,
                            'source': 'the Falcon export\'s manifest'}
                return {'kind': 'carla', 'name': t, 'source': 'the Falcon export\'s manifest'}
    return None


def import_log(kind: str | None, path: str, **opts) -> Log:
    """Import a log. kind one of KINDS (None: detected from the path). opts: nuscenes scene, channel; waymo segment;
    ros2 world_frame, ego_frame, topics, unknown_by_size, town (the map when the caller knows it)."""
    path = os.path.abspath(os.path.expanduser(path))
    kind = kind or detect_kind(path)
    if kind not in KINDS:
        raise LogError(f'{path}: not a log this version reads (nuScenes, Waymo v2, ROS 2 bag, KITTI raw, Falcon run)')
    if kind == 'nuscenes':
        from .nuscenes import import_nuscenes
        return import_nuscenes(path, opts.get('scene'), opts.get('channel') or 'LIDAR_TOP')
    if kind == 'waymo':
        from .waymo import import_waymo
        return import_waymo(path, opts.get('segment'))
    if kind == 'kitti_raw':
        from .sources import import_kitti_raw
        return import_kitti_raw(path)
    if kind == 'falcon':
        from .sources import import_falcon_run
        return import_falcon_run(path)
    from .ros2 import import_bag
    hint = {'kind': 'carla', 'name': opts['town'], 'source': 'given'} if opts.get('town') else \
        falcon_export_town(path)
    return import_bag(path, opts.get('world_frame'), opts.get('ego_frame'), opts.get('topics'),
                      bool(opts.get('unknown_by_size')), hint)
