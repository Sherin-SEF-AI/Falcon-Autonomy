"""The data engine's core (standard library + numpy): dataset specs as coverage targets, planning the drives that
fill them with their cost, coverage of what was recorded, splits without leakage, dataset versions.

A dataset spec (the project's dataset_specs) says what to collect:

    {"id", "name", "rig": rig id, "fps": 20, "record_every": 5 (keep every 5th step), "drive_s": 60,
     "worlds": [world ids], "weathers": [preset names or {"preset", …overrides}], "scenarios": [ids] (milestone 4),
     "targets": {"car": 2000, "pedestrian": 1000, …}  (objects per class, counted in recorded camera frames),
     "bins": {"distance_m": [0, 20, 40, 80, 200], "occlusion": [0, 0.25, 0.5, 0.75, 1.01],
              "truncation": [0, 0.01, 0.5, 1.01]},
     "bin_targets": {"distance_m": {"car": [500, 500, 500, 500]}},   (optional: per bin)
     "traffic_profile": id, "seed": 0, "max_drives": 200}

The plan is the cells worlds × weathers (× scenarios), each with seeded drives; how many drives a target needs comes
from the yield measured on the dataset's own earlier drives (objects per recorded frame, per class and cell), or —
before any — from a stated prior (PRIOR_YIELD, heuristic) with its label. Each drive has a cache key: the SHA-256 of its
config without the output folder, so a drive that already ran with that exact config is reused (resumable, cached).

Coverage counts objects in the recorded frames' object files (<camera>/objects in frames.jsonl: the ground truth's
objects with "class", "range_m", "occluded", "truncated"), against the targets, per bin.

Splits never put one drive, or (by place) one world, in two splits: the unit of assignment is the drive or the world,
assigned by a seeded shuffle to reach the requested shares as closely as whole units allow.
"""
from __future__ import annotations

import copy
import json
import os
import random

from .provenance import sha256_json, utc_now

DATASET_VERSION_SCHEMA = 'falcon.dataset_version/1'
DEFAULT_BINS = {'distance_m': [0, 20, 40, 80, 200], 'occlusion': [0, 0.25, 0.5, 0.75, 1.01],
                'truncation': [0, 0.01, 0.5, 1.01]}
# objects per recorded camera frame before any drive of the dataset measured it (heuristic: a forward camera in a
# CARLA town with 40 vehicles and 20 walkers)
PRIOR_YIELD = {'car': 4.0, 'truck': 0.3, 'bus': 0.2, 'motorcycle': 0.3, 'bicycle': 0.2, 'pedestrian': 1.0,
               'traffic light': 1.0, 'traffic sign': 1.0}
PRIOR_SOURCE = ('heuristic prior: a forward camera in a CARLA town with 40 vehicles and 20 walkers; replaced by '
                'the yield measured on this dataset\'s own drives as soon as one has run')


def new_spec(spec_id: str, name: str, rig_id: str, worlds: list) -> dict:
    return {'id': spec_id, 'name': name, 'rig': rig_id, 'fps': 20.0, 'record_every': 5, 'drive_s': 60.0,
            'worlds': list(worlds), 'weathers': ['ClearNoon'], 'scenarios': [],
            'targets': {'car': 1000, 'pedestrian': 300}, 'bins': copy.deepcopy(DEFAULT_BINS), 'bin_targets': {},
            'traffic_profile': None, 'seed': 0, 'max_drives': 200}


def cells(spec: dict) -> list:
    """[{'world', 'weather', 'scenario'}] — the cells the spec's drives are spread over."""
    out = []
    for w in spec.get('worlds') or []:
        for we in spec.get('weathers') or ['ClearNoon']:
            for sc in spec.get('scenarios') or [None]:
                out.append({'world': w, 'weather': we if isinstance(we, dict) else {'preset': we}, 'scenario': sc})
    return out


def drive_key(cfg: dict) -> str:
    """The cache key of a drive: its config without where it is written, the server's ports, and the ODD monitor
    (it only observes)."""
    c = {k: v for k, v in cfg.items() if k not in ('out', 'port', 'tm_port', 'host', 'name', 'odd', 'odd_reference',
                                                    'odd_every', 'odd_workspace')}
    return sha256_json(c)


def measured_yield(coverage_rows: list) -> dict:
    """{class: objects per recorded frame} from coverage rows of finished drives."""
    frames = sum(r['frames'] for r in coverage_rows)
    tot = {}
    for r in coverage_rows:
        for cls, n in r['objects'].items():
            tot[cls] = tot.get(cls, 0) + n
    return {c: n / frames for c, n in tot.items()} if frames else {}


def plan(spec: dict, have: dict | None = None, yield_: dict | None = None, throughput=None, mpx: float = 0.0) -> dict:
    """The drives still needed: {'drives': [{'cell', 'seed', 'index'}], 'per_drive_frames', 'needed', 'yield',
    'yield_source', 'estimate'}. have: objects already recorded per class; yield_: measured objects per frame."""
    have = have or {}
    measured = bool(yield_)
    y = yield_ if measured else PRIOR_YIELD
    frames_per_drive = int(spec['drive_s'] * spec['fps'] / max(int(spec.get('record_every', 1)), 1))
    needed = {c: max(int(t) - int(have.get(c, 0)), 0) for c, t in (spec.get('targets') or {}).items()}
    drives_for = {}
    for c, n in needed.items():
        per = y.get(c, 0.0) * frames_per_drive
        drives_for[c] = (0 if n == 0 else (None if per <= 0 else -(-n // max(int(per), 1)) if per >= 1 else
                                            int(n / per) + 1))
    unreachable = sorted(c for c, d in drives_for.items() if d is None)
    count = max([d for d in drives_for.values() if d is not None] or [0])
    count = min(count, int(spec.get('max_drives', 200)))
    cs = cells(spec)
    drives = []
    for i in range(count):
        cell = cs[i % len(cs)] if cs else None
        drives.append({'cell': cell, 'seed': int(spec.get('seed', 0)) * 100003 + i, 'index': i})
    est = None
    if throughput is not None and drives:
        steps = int(spec['drive_s'] * spec['fps'])
        e = throughput.estimate(steps, mpx)
        est = {'seconds': e['seconds'] * len(drives), 'per_drive_s': e['seconds'], 'label': e['label'],
               'source': e['source']}
    return {'drives': drives, 'per_drive_frames': frames_per_drive, 'needed': needed, 'unreachable': unreachable,
            'yield': y, 'yield_source': 'measured on this dataset\'s drives' if measured else PRIOR_SOURCE,
            'estimate': est, 'capped': count == int(spec.get('max_drives', 200))}


def _bin(v, edges):
    for k in range(len(edges) - 1):
        if edges[k] <= v < edges[k + 1]:
            return k
    return None


def coverage_of_run(run_dir: str, bins: dict | None = None) -> dict:
    """Objects in a recorded drive's frames: {'run', 'frames', 'objects': {class: n}, 'bins': {name: {class: [..]}},
    'world', 'weather'}. Frames without object files count as frames with no objects of known class."""
    bins = bins or DEFAULT_BINS
    idx = os.path.join(run_dir, 'frames.jsonl')
    out = {'run': run_dir, 'frames': 0, 'objects': {}, 'bins': {b: {} for b in bins}}
    try:
        with open(os.path.join(run_dir, 'drive.json'), encoding='utf-8') as fh:
            cfg = json.load(fh)
        out['world'] = (cfg.get('project') or {}).get('world')
        out['weather'] = cfg.get('weather_source')
    except (OSError, ValueError):
        pass
    if not os.path.isfile(idx):
        return out
    with open(idx, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            fr = json.loads(line)
            for key, rel in fr.get('files', {}).items():
                if not key.endswith('/objects'):
                    continue
                out['frames'] += 1
                try:
                    with open(os.path.join(run_dir, rel), encoding='utf-8') as of:
                        objs = json.load(of)
                except (OSError, ValueError):
                    continue
                for o in objs.get('objects', objs) if isinstance(objs, dict) else objs:
                    c = o.get('class')
                    if not c:
                        continue
                    out['objects'][c] = out['objects'].get(c, 0) + 1
                    for b, edges in bins.items():
                        v = {'distance_m': o.get('range_m'), 'occlusion': o.get('occluded'),
                             'truncation': o.get('truncated')}.get(b)
                        if v is None:
                            continue
                        k = _bin(float(v), edges)
                        if k is None:
                            continue
                        row = out['bins'][b].setdefault(c, [0] * (len(edges) - 1))
                        row[k] += 1
    return out


def coverage(spec: dict, rows: list) -> dict:
    """What the dataset has against what the spec asks: per class, per bin, per cell."""
    have, per_bin, per_cell = {}, {b: {} for b in (spec.get('bins') or DEFAULT_BINS)}, {}
    for r in rows:
        for c, n in r['objects'].items():
            have[c] = have.get(c, 0) + n
        for b, by in r['bins'].items():
            for c, counts in by.items():
                acc = per_bin.setdefault(b, {}).setdefault(c, [0] * len(counts))
                for k, n in enumerate(counts):
                    acc[k] += n
        key = f'{r.get("world")} / {(r.get("weather") or {}).get("preset", "?")}'
        cell = per_cell.setdefault(key, {'frames': 0, 'objects': {}})
        cell['frames'] += r['frames']
        for c, n in r['objects'].items():
            cell['objects'][c] = cell['objects'].get(c, 0) + n
    targets = spec.get('targets') or {}
    return {'have': have, 'targets': targets,
            'fill': {c: (have.get(c, 0) / t if t else 1.0) for c, t in targets.items()},
            'bins': per_bin, 'bin_targets': spec.get('bin_targets') or {}, 'cells': per_cell,
            'frames': sum(r['frames'] for r in rows), 'drives': len(rows)}


def split(units: list, shares: dict, seed: int = 0, weights: dict | None = None) -> dict:
    """{split: [units]} — each unit (a drive or a world) in exactly one split, by a seeded shuffle filling the shares
    (by weight: frames per unit, else 1) greedily: no leakage across splits by construction."""
    names = list(shares)
    total_share = sum(shares.values())
    order = list(units)
    random.Random(seed).shuffle(order)
    w = weights or {}
    total = sum(w.get(u, 1) for u in order) or 1
    got = {n: 0.0 for n in names}
    out = {n: [] for n in names}
    for u in order:
        # the split furthest below its share takes the unit
        n = min(names, key=lambda n: (got[n] / total) - shares[n] / total_share)
        out[n].append(u)
        got[n] += w.get(u, 1)
    return out


def make_version(spec: dict, runs: list, splits: dict, by: str, number: int, workspace: str) -> dict:
    """A dataset version: the runs (with their run.json hashes), the split assignment, the spec, provenance."""
    from .provenance import block, sha256_file
    entries = []
    for r in runs:
        rj = os.path.join(r, 'run.json')
        entries.append({'run': os.path.relpath(r, workspace), 'run_json_sha256': sha256_file(rj)
                        if os.path.isfile(rj) else None})
    doc = {'schema': DATASET_VERSION_SCHEMA, 'spec': spec['id'], 'name': spec['name'], 'version': number,
           'created': utc_now(), 'split_by': by, 'splits': splits, 'runs': entries, 'spec_snapshot': spec,
           'provenance': block(seed=spec.get('seed'), settings={'split_by': by})}
    doc['sha256'] = sha256_json({k: v for k, v in doc.items() if k not in ('created', 'provenance')})
    return doc
