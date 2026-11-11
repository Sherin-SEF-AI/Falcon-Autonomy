"""What a remote job carries: its config and the content of the files it reads, never this machine's paths (standard
library; docs/cluster.md, "Bundles").

A job bundle (kind "job") is made from a JobSpec the service built for an allow-listed worker (WORKERS): the config
the worker would read (drive.json), with what only makes sense here stripped — where it is written ("out"), the server's
address and ports ("host", "port", "tm_port": the node uses its own slot's) — and every other value that is an
absolute path to a file on this machine replaced by "@input:<name>" with the file's content in the bundle. A folder
named in the config is refused (it cannot be carried), apart from DROPPED keys, which are left out and noted. The
node writes the inputs into its scratch folder and puts their paths back; it builds the worker's command itself, so a
bundle can only ever run an allow-listed worker on a config.

A study bundle (kind "study") is what a cloud burst runs: a whole study (failrate, sweep or search) with the part of the
project it needs (its world, rig and traffic profile, and the scenario) and its options (STUDY_OPTIONS only).

Refused: anything private (assistant.privacy: dashcam-derived configs, scenarios, runs and studies, the user's footage
folder) — dashcam-derived data never leaves this machine; a bundle larger than its cap (LAN_MAX for the cluster,
CLOUD_MAX for a cloud burst; the encoded, compressed size); a worker not on the allow-list.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import zlib

SCHEMA = 'falcon.bundle/1'
LAN_MAX = 5 << 20                   # 5 MiB: a bundle to a cluster node
CLOUD_MAX = 64 << 10                # 64 KiB: a cloud burst's study (kilobytes, as the plan says)
STRIP = ('out', 'host', 'port', 'tm_port')
DROPPED = {'odd_workspace': 'the ODD data monitor\'s feature weights stay on this machine: the node monitors the '
                            'ODD\'s conditions without the feature monitor'}
PLACEHOLDER = '@input:'
# the workers a node runs, the file it reads its config from, and what it needs there (the node decides, not the bundle)
WORKERS = {
    'drive': {'config': 'drive.json', 'needs_carla': True, 'memory_gb': 1.0},
}
STUDY_COMMANDS = ('failrate', 'sweep', 'search')
STUDY_OPTIONS = {
    'failrate': ('stage_size', 'final', 'rho', 'defensive', 'max_stages', 'batch', 'threshold', 'seed', 'search_seed',
                 'judge', 'count', 'fps', 'no_feasibility'),
    'sweep': ('grid', 'samples', 'seed', 'count', 'fps', 'no_feasibility'),
    'search': ('method', 'budget', 'batch', 'threshold', 'seed', 'search_seed', 'count', 'fps', 'no_feasibility'),
}


class BundleError(Exception):
    """The job cannot be carried to another machine (the message says why; it is never started)."""


# ------------------------------------------------------------------------------------------------ privacy
def _private(cfg: dict, paths) -> str | None:
    from ..assistant import privacy
    why = privacy._says_dashcam(cfg)
    if why:
        return why
    for p in list(paths) + privacy._made_from(cfg):    # and a reconstruction the config renders (a hybrid drive's)
        why = privacy.private_reason(p)
        if why:
            return why
    return None


# ------------------------------------------------------------------------------------------------ job bundles
def worker_of(spec) -> str:
    """The worker a JobSpec runs (meta['worker'], set by CarlaService.worker_spec; else read from its command)."""
    w = spec.meta.get('worker') or ''
    if not w and 'falcon_autonomy.workers' in spec.argv:
        i = spec.argv.index('falcon_autonomy.workers')
        w = spec.argv[i + 1] if i + 1 < len(spec.argv) else ''
    return w


def _file_input(path: str, inputs: dict, seen: dict) -> str:
    real = os.path.realpath(path)
    if real in seen:
        return PLACEHOLDER + seen[real]
    why = _private({}, [path])                    # every file carried: the footage, ingest folders, their events
    if why:
        raise BundleError(f'private: {why} — dashcam-derived data never leaves this machine')
    name = f'{len(inputs)}-{os.path.basename(path)}'
    with open(path, 'rb') as fh:
        data = fh.read()
    inputs[name] = {'data': base64.b64encode(data).decode('ascii'), 'sha256': hashlib.sha256(data).hexdigest(),
                    'name': os.path.basename(path)}
    seen[real] = name
    return PLACEHOLDER + name


def _carry(value, inputs: dict, seen: dict, key: str = ''):
    """The config with every absolute file path made an input; BundleError for a folder (or an unreadable file)."""
    if isinstance(value, dict):
        return {k: _carry(v, inputs, seen, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_carry(v, inputs, seen, key) for v in value]
    if isinstance(value, str) and value.startswith('/') and len(value) < 4096 and '\n' not in value:
        if os.path.isdir(value):
            raise BundleError(f'the config names a folder on this machine ({key!r}), which a bundle cannot carry')
        if os.path.isfile(value):
            try:
                return _file_input(value, inputs, seen)
            except OSError as e:
                raise BundleError(f'cannot read {key!r} for the bundle: {e.strerror}') from None
    return value


def make(spec, cap: int = LAN_MAX) -> dict:
    """The bundle of a JobSpec (BundleError when it cannot go: not allow-listed, private, too large)."""
    worker = worker_of(spec)
    if worker not in WORKERS:
        raise BundleError(f'the {worker or "?"} worker does not run on the cluster (only: {", ".join(WORKERS)}): its '
                          'job stays on this machine')
    cfg_path = spec.meta.get('config') or ''
    if not cfg_path or not os.path.isfile(cfg_path):
        raise BundleError('the job has no config file to carry')
    with open(cfg_path, encoding='utf-8') as fh:
        cfg = json.load(fh)
    extra = [p for p in spec.inputs if p]
    why = _private(cfg, [cfg_path, spec.run_dir] + extra)
    if why:
        raise BundleError(f'private: {why} — dashcam-derived data never leaves this machine')
    notes = []
    body = {k: copy.deepcopy(v) for k, v in cfg.items() if k not in STRIP}
    for k, note in DROPPED.items():
        if k in body:
            body.pop(k)
            notes.append(note)
    inputs, seen = {}, {}
    body = _carry(body, inputs, seen)
    for p in extra:                                       # files the job reads that its config does not name
        if os.path.isdir(p):
            raise BundleError('the job reads a folder on this machine, which a bundle cannot carry')
        _file_input(p, inputs, seen)
    b = {'schema': SCHEMA, 'kind': 'job', 'worker': worker, 'job_kind': spec.kind, 'title': spec.title,
         'config': body, 'inputs': inputs, 'notes': notes,
         'needs': {'backend': spec.meta.get('backend') or 'carla-0.9.16', 'vram_gb': float(spec.meta.get('vram_gb') or 0),
                   'memory_gb': float(spec.memory_gb)}}
    size(b, cap)
    return b


# ------------------------------------------------------------------------------------------------ study bundles
def make_study(project, command: str, scenario: dict, world: str, rig: str, name: str, options: dict,
               no_render: bool = False, cap: int = CLOUD_MAX) -> dict:
    """A cloud burst's study: the project reduced to what the study reads, its scenario and allow-listed options."""
    from ..core import scenarios as SC
    if command not in STUDY_COMMANDS:
        raise BundleError(f'{command} does not run as a cloud study (only: {", ".join(STUDY_COMMANDS)})')
    bad = sorted(set(options) - set(STUDY_OPTIONS[command]))
    if bad:
        raise BundleError(f'options a cloud {command} does not take: {", ".join(bad)}')
    doc = project.doc
    w = next((x for x in doc.get('worlds') or () if x.get('id') == world), None)
    r = next((x for x in doc.get('rigs') or () if x.get('id') == rig), None)
    if w is None or r is None:
        raise BundleError('the study\'s world or rig is not in the project')
    if w.get('kind') != 'town':
        raise BundleError(f'world {w.get("name")!r} is an imported map: its OpenDRIVE file stays on this machine — a '
                          'cloud study runs on a CARLA town')
    tp_id = w.get('traffic_profile') or 'tp-default'
    tps = [x for x in doc.get('traffic_profiles') or () if x.get('id') == tp_id]
    why = _private({'scenario': scenario}, [])
    if why:
        raise BundleError(f'private: {why} — dashcam-derived data never leaves this machine')
    if no_render:
        r = dict(copy.deepcopy(r), sensors=[])            # the criticality needs actor states, not images
    sub = {'schema': doc.get('schema'), 'version': doc.get('version'), 'name': doc.get('name', 'study'),
           'worlds': [w], 'rigs': [r], 'traffic_profiles': tps,
           'scenarios': [scenario] if scenario.get('id') not in SC.LIBRARY else [], 'models': []}
    for k in ('loop',):
        if k in doc:
            sub[k] = {}
    b = {'schema': SCHEMA, 'kind': 'study', 'command': command, 'name': name, 'scenario': scenario,
         'world': world, 'rig': rig, 'options': dict(options), 'no_render': bool(no_render), 'project': sub,
         'needs': {'backend': 'carla-0.9.16', 'vram_gb': 0.0, 'memory_gb': 4.0}}
    size(b, cap)
    return b


# ------------------------------------------------------------------------------------------------ encoding
def encode(b: dict) -> bytes:
    return zlib.compress(json.dumps(b, separators=(',', ':'), sort_keys=True).encode(), 9)


def decode(data: bytes, cap: int = LAN_MAX) -> dict:
    d = zlib.decompressobj()
    raw = d.decompress(data, 64 * cap + 1)
    if len(raw) > 64 * cap:
        raise BundleError('the bundle unpacks to too much')
    try:
        b = json.loads(raw)
    except ValueError as e:
        raise BundleError('the bundle is not JSON') from e
    if not isinstance(b, dict) or b.get('schema') != SCHEMA or b.get('kind') not in ('job', 'study'):
        raise BundleError('not a Falcon bundle')
    return b


def size(b: dict, cap: int) -> int:
    n = len(encode(b))
    if n > cap:
        raise BundleError(f'the bundle is {n / 1024:.0f} KiB, over its cap of {cap / 1024:.0f} KiB'
                          + (' (a cloud burst carries kilobytes: a study of a CARLA town)' if cap <= CLOUD_MAX else ''))
    return n


def sha(b: dict) -> str:
    return hashlib.sha256(encode(b)).hexdigest()


# ------------------------------------------------------------------------------------------------ the node's side
def materialise(b: dict, run_dir: str, host: str, port: int, tm_port: int) -> str:
    """Write a job bundle's inputs into run_dir/inputs and its config (paths put back, the node's address and ports,
    out = run_dir) into run_dir/<the worker's config file>; that file's path."""
    if b.get('kind') != 'job' or b.get('worker') not in WORKERS:
        raise BundleError(f'the {b.get("worker") or "?"} worker is not on this node\'s allow-list')
    inp = os.path.join(run_dir, 'inputs')
    paths = {}
    for name, item in (b.get('inputs') or {}).items():
        if not isinstance(item, dict) or os.path.basename(name) != name or name.startswith('.'):
            raise BundleError('an input with a bad name')
        data = base64.b64decode(item.get('data') or '')
        if hashlib.sha256(data).hexdigest() != item.get('sha256'):
            raise BundleError(f'input {name} does not match its hash')
        os.makedirs(inp, exist_ok=True)
        p = os.path.join(inp, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths[name] = p

    def put(v):
        if isinstance(v, dict):
            return {k: put(x) for k, x in v.items()}
        if isinstance(v, list):
            return [put(x) for x in v]
        if isinstance(v, str) and v.startswith(PLACEHOLDER):
            n = v[len(PLACEHOLDER):]
            if n not in paths:
                raise BundleError(f'the config names an input the bundle does not carry ({n})')
            return paths[n]
        return v
    cfg = put(b.get('config') or {})
    cfg.update(out=os.path.abspath(run_dir), host=host, port=int(port), tm_port=int(tm_port))
    path = os.path.join(run_dir, WORKERS[b['worker']]['config'])
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, indent=1)
    return path
