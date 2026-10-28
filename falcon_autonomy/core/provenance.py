"""Provenance: what produced an output, so every number can be traced to its run (standard library only).

Every JSON Falcon writes carries a "provenance" block:

    {"app": "Falcon Autonomy", "app_version": "0.1.0", "created": "2026-10-28T15:04:05Z",
     "inputs": {"<role>": {"path": "~/x.json", "sha256": "..."}}, "seed": 7, "settings": {...},
     "host": {"python": "3.12.13", "platform": "Linux-...", "carla": "0.9.16"}}

and every run folder a run.json (RUN_SCHEMA) saying exactly what ran: the worker, its arguments, versions, the seed,
the settings, the inputs' hashes, the outputs' hashes, timings, the status and exit code.
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
