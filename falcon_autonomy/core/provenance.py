"""Provenance: what produced an output, so every number can be traced to its run (standard library only).

Every JSON Falcon writes carries a "provenance" block:

    {"app": "Falcon Autonomy", "app_version": "0.1.0", "created": "2026-10-28T15:04:05Z",
     "inputs": {"<role>": {"path": "~/x.json", "sha256": "..."}}, "seed": 7, "settings": {...},
     "host": {"python": "3.12.13", "platform": "Linux-...", "carla": "0.9.16"}}

and every run folder a run.json (RUN_SCHEMA) saying exactly what ran: the worker, its arguments, versions, the seed,
the settings, the inputs' hashes, the outputs' hashes, timings, the status and exit code.

pins(...) is what an experiment's runs are pinned to (Falcon's version, the CARLA build, the hashes of the worlds,
rigs, traffic profiles and model weights, the seeds), so that a re-run is checked against the original and says what
changed (diff_pins; core.ablation, core.reproduce.compare_experiments).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys

from .. import APP_NAME, __version__
from .paths import atomic_write, tilde

RUN_SCHEMA = 'falcon.run/1'


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path: str, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    """The hash of a JSON value in canonical form (sorted keys, no spaces): a stable cache key for settings."""
    return sha256_bytes(canonical_json(obj).encode('utf-8'))


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)


def host_info(**extra) -> dict:
    d = {'python': sys.version.split()[0], 'platform': platform.platform(terse=True), 'machine': platform.machine()}
    d.update({k: v for k, v in extra.items() if v is not None})
    return d


def block(inputs: dict | None = None, seed=None, settings: dict | None = None, **host) -> dict:
    """A provenance block. inputs: {role: path} (hashed now; a missing file is recorded as missing)."""
    ins = {}
    for role, path in (inputs or {}).items():
        if not path:
            continue
        rec = {'path': tilde(path)}
        try:
            rec['sha256'] = sha256_file(path) if os.path.isfile(path) else None
        except OSError:
            rec['sha256'] = None
        if rec['sha256'] is None:
            rec['missing'] = not os.path.exists(path)
        ins[role] = rec
    return {'app': APP_NAME, 'app_version': __version__, 'created': utc_now(), 'inputs': ins, 'seed': seed,
            'settings': settings or {}, 'host': host_info(**host)}


def hash_tree(root: str, skip=('run.json',)) -> dict:
    """{relative path: sha256} of every file under root (run.json itself left out): outputs' hashes for run.json and
    --reproduce."""
    out = {}
    for d, _dirs, files in os.walk(root):
        _dirs.sort()
        for f in sorted(files):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root)
            if rel in skip:
                continue
            out[rel] = sha256_file(p)
    return out


def write_json(path: str, doc: dict):
    atomic_write(path, json.dumps(doc, indent=2, ensure_ascii=False, default=str) + '\n')


def read_json(path: str) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def new_run(worker: str, argv: list, seed=None, settings: dict | None = None, inputs: dict | None = None) -> dict:
    """A run.json document at the start of a run; the worker fills in status, outputs, timings and notes."""
    return {'schema': RUN_SCHEMA, 'worker': worker, 'argv': list(argv), 'status': 'running', 'exit_code': None,
            'started': utc_now(), 'finished': None, 'error': None, 'notes': [],
            'provenance': block(inputs, seed, settings)}


# ================================================================================================ pins
def weights_sha256(model: dict, workspace: str = '') -> str:
    """A registered model's weights: its file's SHA-256, or for a plug-in (runtime python) the hash of its
    registration with its parameters ('registration:…'; its code is the package's, pinned by the app version)."""
    path = model.get('path') or ''
    if path and not os.path.isabs(path) and workspace:
        path = os.path.join(workspace, 'models', path)
    if path and os.path.isfile(path):
        return sha256_file(path)
    keep = {k: model.get(k) for k in ('runtime', 'entry', 'kind', 'classes', 'class_map', 'params', 'options')}
    return 'registration:' + sha256_json(keep)


def world_sha256(world: dict, workspace: str = '') -> str:
    """A world's definition (town, weather, traffic profile, settings) and, for an imported one, its OpenDRIVE file."""
    h = {'definition': sha256_json(world)}
    x = world.get('xodr')
    if x:
        p = x if os.path.isabs(x) else os.path.join(workspace, x)
        h['xodr'] = sha256_file(p) if os.path.isfile(p) else None
    return sha256_json(h)


def pins(project=None, workspace: str = '', worlds=(), rigs=(), models=(), traffic=(), seeds=(), carla=None,
         extra: dict | None = None) -> dict:
    """What a result is pinned to, so that a re-run can say exactly what changed: Falcon's version, the CARLA build
    (the server's version as a worker read it; None until a CARLA run recorded one), the hash of each world, rig and
    traffic profile used (their project definitions) and of each model's weights, and the seeds. diff_pins compares
    two."""
    doc = getattr(project, 'doc', None) or {}

    def items(section, ids):
        return {i: next((x for x in doc.get(section) or () if x.get('id') == i), None) for i in ids}
    out = {'app': APP_NAME, 'app_version': __version__, 'carla_server': carla, 'seeds': sorted(set(seeds)),
           'worlds': {i: (world_sha256(w, workspace) if w else None) for i, w in items('worlds', worlds).items()},
           'rigs': {i: (sha256_json(r) if r else None) for i, r in items('rigs', rigs).items()},
           'traffic_profiles': {i: (sha256_json(t) if t else None)
                                for i, t in items('traffic_profiles', traffic).items()},
           'weights': {i: (weights_sha256(m, workspace) if m else None) for i, m in items('models', models).items()}}
    out.update(extra or {})
    return out


def diff_pins(a: dict, b: dict, prefix: str = '') -> list:
    """[{'key', 'a', 'b'}] for every pinned value that differs between two pins (nested keys joined by '.')."""
    out = []
    for k in sorted(set(a or {}) | set(b or {})):
        va, vb = (a or {}).get(k), (b or {}).get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff_pins(va, vb, f'{prefix}{k}.')
        elif va != vb:
            out.append({'key': f'{prefix}{k}', 'a': va, 'b': vb})
    return out
