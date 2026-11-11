"""Content-addressed lineage: which bytes a dataset version and a model were made from, provably (numpy-free; pyarrow
for the indexes — the extra `lineage` — and core.cas for the hashes and the store).

Run index. index_run(RUN) hashes every recorded frame's files and writes RUN/frames.parquet — one row per (step,
sensor): `frame` the address of its data file (the camera's rgb image, the lidar sweep, the radar return; "-" when the
sensor recorded only labels), `label` the address of its label set, and each file's path, size and mtime (so the next
index re-hashes only files that changed). A label set is the canonical JSON {kind: address} of the frame's label files
(semantic, instance, range, objects, flow, the lidar's .labels.bin, the radar's flags) — hashed as bytes, so it is an
address too. run.json is never modified: the index is a file of its own, and its summary (algorithm, Merkle root,
rows) is in the parquet file's metadata (key `falcon.lineage`). Indexing is a cheap CPU job (index_job_spec) queued
after a run finishes (after_job).

Merkle root. RFC 6962 §2.1 (Certificate Transparency): leaf hash H(0x00 ‖ d), node hash H(0x01 ‖ left ‖ right) and,
for n leaves, the largest power of two k < n splits D[0:k] | D[k:n] — the same tree as pairing a level's nodes left to
right and promoting an odd last node unchanged to the next level, which is how merkle_root() computes it. The hash is
the index's algorithm (BLAKE3 or SHA-256, core.cas). A dataset's leaves are its samples' (frame, label) pairs, the
leaf bytes "<frame address>|<label address>" (ASCII), sorted bytewise (so the root does not depend on the order the
samples were listed in; a sample listed twice counts twice). The empty tree's root is H(""). audit_path() and
verify_path() are RFC 6962's PATH and RFC 9162 §2.1.3.2's check: `falcon lineage prove MODEL FRAME` shows that one
frame is among a model's training samples with ⌈log₂ n⌉ hashes.

Dataset versions. core.datasets.make_version also writes datasets/<spec>/splits/v<NNN>.parquet (write_splits): every
sample of every split with its frame and label addresses, and puts each file and label set into the workspace's store
(core.cas: reflink → hard link → copy), so the version's bytes survive the runs being changed or deleted;
materialise() rebuilds a version's files from the store (`--hardlink`: no data copied).

Models. core.training's manifest hashes every training sample (the image's bytes; the labels as trained — the
canonical JSON of its boxes, classes and ignore flags), data.json keeps the addresses and each split's Merkle root, and
the exported ONNX file carries falcon.data_merkle_root (the training split's root), falcon.val_merkle_root,
falcon.hash_alg, falcon.dataset_version and falcon.manifest_sha in its metadata_props — written before the
registration's SHA-256 is computed, so the registered file is the one that names its data. verify() recomputes it.

Commands (`falcon lineage …`): index RUN… | verify MODEL|VERSION|RUN | prove MODEL FRAME | diff A B |
materialise VERSION DIR [--hardlink] | gc WORKSPACE [--dry-run].

Limits. A frame's address is that of its file's bytes: re-encoding an image changes it; sensors' jsonl streams (GNSS,
IMU) and the run's own JSON files are not leaves. The store's hard links make a run's put files read-only (0444).
"""
from __future__ import annotations

import json
import os
import sys

from . import cas as C
from .provenance import canonical_json, read_json, utc_now

INDEX_SCHEMA = 'falcon.lineage_index/1'
SPLITS_SCHEMA = 'falcon.lineage_splits/1'
INDEX_FILE = 'frames.parquet'
META_KEY = b'falcon.lineage'
DATA_KINDS = ('rgb', 'lidar', 'radar')       # a sensor's data file; every other kind of a frame is a label
NONE = '-'
INDEX_KINDS = ('drive', 'smoke', 'live', 'scenario', 'replay', 'stack', 'gym', 'hil')   # runs after_job indexes


class LineageError(Exception):
    pass


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise LineageError('lineage indexes need pyarrow (pip install pyarrow, or the extra '
                           'falcon-autonomy[lineage])') from None
    return pa, pq


# ================================================================================================ Merkle (RFC 6962)
def _h(alg: str, data: bytes) -> bytes:
    return C.digest(data, alg)


def leaf_hash(data: bytes, alg: str) -> bytes:
    return _h(alg, b'\x00' + data)


def node_hash(left: bytes, right: bytes, alg: str) -> bytes:
    return _h(alg, b'\x01' + left + right)


def merkle_root(leaves, alg: str) -> bytes:
    """RFC 6962's Merkle tree hash of the leaves (bytes each, in the order given): pairs left to right, an odd last
    node promoted unchanged."""
    level = [leaf_hash(x, alg) for x in leaves]
    if not level:
        return _h(alg, b'')
    while len(level) > 1:
        nxt = [node_hash(level[i], level[i + 1], alg) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def _mth(hashes: list, alg: str) -> bytes:
    """The tree hash of already leaf-hashed nodes (for audit paths)."""
    level = list(hashes)
    while len(level) > 1:
        nxt = [node_hash(level[i], level[i + 1], alg) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def audit_path(leaves, m: int, alg: str) -> list:
    """RFC 6962 §2.1.1 PATH(m, D[n]): the sibling hashes from leaf m up to the root."""
    hs = [leaf_hash(x, alg) for x in leaves]
    if not 0 <= m < len(hs):
        raise LineageError(f'leaf {m} is not among {len(hs)}')

    def path(m, d):
        n = len(d)
        if n <= 1:
            return []
        k = 1
        while k * 2 < n:
            k *= 2
        if m < k:
            return path(m, d[:k]) + [_mth(d[k:], alg)]
        return path(m - k, d[k:]) + [_mth(d[:k], alg)]
    return path(m, hs)


def verify_path(leaf: bytes, m: int, n: int, path: list, root: bytes, alg: str) -> bool:
    """RFC 9162 §2.1.3.2: whether leaf (its bytes) at index m of n leaves, with the audit path, gives root."""
    if not 0 <= m < n:
        return False
    fn, sn = m, n - 1
    r = leaf_hash(leaf, alg)
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r, alg)
            while not fn & 1 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            r = node_hash(r, p, alg)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def leaf_bytes(frame: str | None, label: str | None) -> bytes:
    return f'{frame or NONE}|{label or NONE}'.encode('ascii')


def pairs_root(pairs, alg: str) -> str:
    """The address-style root ('<alg>:<hex>') of (frame, label) pairs (module doc: leaves sorted)."""
    return f'{alg}:{merkle_root(sorted(leaf_bytes(f, l) for f, l in pairs), alg).hex()}'


def _alg_of(*addresses) -> str | None:
    for a in addresses:
        if a and a != NONE:
            return C.parse(a)[0]
    return None


# ================================================================================================ hashing with a cache
class Hasher:
    """Content addresses of files, each read at most once: cached by (real path, size, mtime_ns), and primed from a
    run's frames.parquet so files indexed before are not read again."""

    def __init__(self, alg: str | None = None):
        self.alg = alg or C.default_alg()
        C.new_hasher(self.alg)
        self.cache = {}
        self.read = 0

    def prime(self, rows: list, root: str):
        for r in rows:
            for path, size, mt, addr in _files_of(r, root):
                if addr and addr != NONE and C.parse(addr)[0] == self.alg:
                    self.cache[(os.path.realpath(path), size, mt)] = addr

    def address(self, path: str) -> dict:
        st = os.stat(path)
        key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
        a = self.cache.get(key)
        if a is None:
            a = C.hash_file(path, self.alg)
            self.cache[key] = a
            self.read += 1
        return {'addr': a, 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}

    def bytes_address(self, data: bytes) -> str:
        return C.hash_bytes(data, self.alg)


def _files_of(row: dict, root: str):
    """(absolute path, bytes, mtime_ns, address) of every file a row names (its frame and its labels)."""
    if row.get('frame_path'):
        yield (os.path.join(root, row['frame_path']), row.get('frame_bytes'), row.get('frame_mtime_ns'),
               row.get('frame'))
    for _kind, f in (json.loads(row['labels']) if row.get('labels') else {}).items():
        yield os.path.join(root, f['path']), f.get('bytes'), f.get('mtime_ns'), f.get('addr')


def label_set(files: dict, hasher: Hasher) -> tuple:
    """(address, canonical bytes) of a label set {kind: address} (None, b'') when empty."""
    if not files:
        return None, b''
    data = canonical_json({k: v for k, v in sorted(files.items())}).encode('utf-8')
    return hasher.bytes_address(data), data


# ================================================================================================ run index
def _frame_records(run_dir: str):
    p = os.path.join(run_dir, 'frames.jsonl')
    if not os.path.isfile(p):
        raise LineageError(f'{run_dir}: no frames.jsonl (not a recorded run)')
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _groups(rec: dict) -> dict:
    """{sensor: {kind: relative path}} of one frames.jsonl record (an image set's record: sensor 'image')."""
    out = {}
    for key, rel in (rec.get('files') or {}).items():
        sid, _, kind = key.rpartition('/')
        if sid and rel:
            out.setdefault(sid, {})[kind] = rel
    if not out and rec.get('image'):
        out['image'] = {'rgb': rec['image']}
    return out


def scan_run(run_dir: str, hasher: Hasher) -> list:
    """The rows of a run's index (module doc), in (step, sensor) order."""
    rows = []
    for i, rec in enumerate(_frame_records(run_dir)):
        if rec.get('recorded') is False:
            continue
        step = int(rec.get('step', i))
        for sid, kinds in sorted(_groups(rec).items()):
            data_kind = next((k for k in DATA_KINDS if k in kinds), None)
            row = {'step': step, 'sim_t': float(rec.get('sim_t', rec.get('t', 0.0)) or 0.0), 'sensor': sid,
                   'frame': NONE, 'frame_path': '', 'frame_bytes': 0, 'frame_mtime_ns': 0, 'label': NONE,
                   'labels': '{}', 'missing': 0}
            labels, label_files = {}, {}
            extra = dict(kinds)
            if data_kind == 'lidar':
                extra['lidar_labels'] = kinds['lidar'][:-4] + '.labels.bin'
            for kind, rel in sorted(extra.items()):
                p = os.path.join(run_dir, rel)
                if not os.path.isfile(p):
                    if kind != 'lidar_labels':
                        row['missing'] += 1
                    continue
                a = hasher.address(p)
                if kind == data_kind:
                    row.update(frame=a['addr'], frame_path=rel, frame_bytes=a['bytes'], frame_mtime_ns=a['mtime_ns'])
                else:
                    labels[kind] = a['addr']
                    label_files[kind] = {'path': rel, 'addr': a['addr'], 'bytes': a['bytes'],
                                         'mtime_ns': a['mtime_ns']}
            addr, _ = label_set(labels, hasher)
            row['label'] = addr or NONE
            row['labels'] = json.dumps(label_files, sort_keys=True, separators=(',', ':'))
            rows.append(row)
    return rows


_INDEX_COLUMNS = (('step', 'int64'), ('sim_t', 'float64'), ('sensor', 'string'), ('frame', 'string'),
                  ('frame_path', 'string'), ('frame_bytes', 'int64'), ('frame_mtime_ns', 'int64'),
                  ('label', 'string'), ('labels', 'string'), ('missing', 'int64'))


def _write_table(path: str, rows: list, columns, meta: dict):
    pa, pq = _pa()
    arrays = {c: pa.array([r.get(c) for r in rows], type=getattr(pa, t)()) for c, t in columns}
    table = pa.table(arrays).replace_schema_metadata({META_KEY: json.dumps(meta).encode('utf-8')})
    tmp = path + '.part'
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def read_table(path: str) -> tuple:
    """(rows as dicts, the falcon.lineage metadata) of a frames.parquet or splits parquet."""
    _pa_, pq = _pa()
    t = pq.read_table(path)
    md = (t.schema.metadata or {}).get(META_KEY)
    return t.to_pylist(), (json.loads(md) if md else {})


def index_path(run_dir: str) -> str:
    return os.path.join(run_dir, INDEX_FILE)


def read_index(run_dir: str) -> tuple:
    p = index_path(run_dir)
    if not os.path.isfile(p):
        raise LineageError(f'{run_dir}: not indexed (falcon lineage index {run_dir})')
    return read_table(p)


def index_run(run_dir: str, alg: str | None = None, hasher: Hasher | None = None, reuse: bool = True) -> dict:
    """Index a recorded run (module doc): writes RUN/frames.parquet, never run.json. Files whose size and mtime are
    unchanged since the last index are not read again (reuse). Returns the summary kept in the file's metadata."""
    import time
    t0 = time.perf_counter()
    hasher = hasher or Hasher(alg)
    if reuse and os.path.isfile(index_path(run_dir)):
        try:
            old, meta = read_index(run_dir)
            if meta.get('hash_alg') == hasher.alg:
                hasher.prime(old, run_dir)
        except (LineageError, OSError, ValueError):
            pass
    before = hasher.read
    rows = scan_run(run_dir, hasher)
    from .. import __version__
    meta = {'schema': INDEX_SCHEMA, 'hash_alg': hasher.alg, 'rows': len(rows),
            'frames': sum(1 for r in rows if r['frame'] != NONE),
            'merkle_root': pairs_root(((r['frame'], r['label']) for r in rows), hasher.alg),
            'missing_files': sum(r['missing'] for r in rows), 'created': utc_now(), 'app_version': __version__,
            'files_read': hasher.read - before, 'seconds': None,
            'leaf': '"<frame address>|<label address>", sorted; RFC 6962 tree (core.lineage)'}
    meta['seconds'] = round(time.perf_counter() - t0, 3)
    _write_table(index_path(run_dir), rows, _INDEX_COLUMNS, meta)
    return meta


def index_job_spec(run_dir: str, title: str = ''):
    """The cheap CPU job that indexes a finished run (`falcon lineage index RUN`): no CARLA, no GPU."""
    from .jobs import JobSpec
    argv = [sys.executable, '-m', 'falcon_autonomy', 'lineage', 'index', run_dir]
    return JobSpec('lineage-index', title or f'Index {os.path.basename(os.path.normpath(run_dir))}', argv,
                   memory_gb=0.4, meta={'lineage_of': run_dir})


def after_job(service, job) -> object | None:
    """For CarlaService._on_job: queue the index of a run that finished well and recorded frames (INDEX_KINDS);
    None when there is nothing to index (or pyarrow is missing: the index is optional)."""
    from . import protocol as P
    s = job.spec
    if job.exit_code != P.OK or s.kind not in INDEX_KINDS or not s.run_dir:
        return None
    if not os.path.isfile(os.path.join(s.run_dir, 'frames.jsonl')):
        return None
    try:
        _pa()
    except LineageError:
        return None
    return service.queue.submit(index_job_spec(s.run_dir))


# ================================================================================================ dataset versions
_SPLIT_COLUMNS = (('split', 'string'), ('run', 'string'), ('step', 'int64'), ('sensor', 'string'),
                  ('frame', 'string'), ('frame_path', 'string'), ('label', 'string'), ('labels', 'string'))


def splits_path(workspace: str, spec_id: str, number: int) -> str:
    return os.path.join(workspace, 'datasets', spec_id, 'splits', f'v{int(number):03d}.parquet')


def write_splits(workspace: str, spec_id: str, number: int, splits: dict, store: C.Store | None = None,
                 how: str = 'auto') -> dict:
    """splits: {split: [run folder relative to the workspace]}. Indexes each run (reusing its frames.parquet), puts
    every frame and label file and every label set into the store, and writes splits/v<NNN>.parquet. Returns
    {'splits' (its path relative to the workspace), 'hash_alg', 'merkle_root', 'roots': {split: root}, 'rows',
    'objects', 'methods'}."""
    store = store or C.Store(C.store_dir(workspace))
    hasher = Hasher(store.alg)
    rows, objects = [], set()
    with_root = {}
    for split, runs in splits.items():
        for rel in runs:
            run_dir = rel if os.path.isabs(rel) else os.path.join(workspace, rel)
            if not os.path.isfile(os.path.join(run_dir, 'frames.jsonl')):
                continue
            index_run(run_dir, hasher=hasher)
            idx, _meta = read_index(run_dir)
            for r in idx:
                labels = json.loads(r['labels'] or '{}')
                if r['frame'] != NONE:
                    got = store.put(os.path.join(run_dir, r['frame_path']), how)
                    if got != r['frame']:
                        raise LineageError(f'{run_dir}/{r["frame_path"]} changed while it was stored')
                    objects.add(got)
                for f in labels.values():
                    objects.add(store.put(os.path.join(run_dir, f['path']), how))
                if labels:
                    _addr, data = label_set({k: f['addr'] for k, f in labels.items()}, hasher)
                    objects.add(store.put_bytes(data))
                rows.append({'split': split, 'run': os.path.relpath(run_dir, workspace), 'step': r['step'],
                             'sensor': r['sensor'], 'frame': r['frame'], 'frame_path': r['frame_path'],
                             'label': r['label'], 'labels': r['labels']})
                with_root.setdefault(split, []).append((r['frame'], r['label']))
    meta = {'schema': SPLITS_SCHEMA, 'spec': spec_id, 'version': int(number), 'hash_alg': store.alg,
            'rows': len(rows), 'merkle_root': pairs_root(((r['frame'], r['label']) for r in rows), store.alg),
            'roots': {s: pairs_root(p, store.alg) for s, p in with_root.items()}, 'created': utc_now()}
    p = splits_path(workspace, spec_id, number)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    _write_table(p, rows, _SPLIT_COLUMNS, meta)
    return {'splits': os.path.relpath(p, workspace), 'hash_alg': store.alg, 'merkle_root': meta['merkle_root'],
            'roots': meta['roots'], 'rows': len(rows), 'objects': len(objects),
            'methods': {k: v for k, v in store.methods.items() if v}}


def materialise(splits_parquet: str, dest: str, store: C.Store, how: str = 'hardlink', split: str | None = None
                ) -> dict:
    """A dataset version's files rebuilt from the store under dest/<split>/<run>/<path as recorded>, plus
    dest/materialised.json. Hard links by default: no data copied (the files are the store's read-only inodes)."""
    rows, meta = read_table(splits_parquet)
    used = {}
    n = 0
    for r in rows:
        if split and r['split'] != split:
            continue
        base = os.path.join(dest, r['split'], r['run'])
        files = [(r['frame_path'], r['frame'])] if r['frame'] != NONE else []
        files += [(f['path'], f['addr']) for f in json.loads(r['labels'] or '{}').values()]
        for rel, addr in files:
            m = store.materialise(addr, os.path.join(base, rel), how)
            used[m] = used.get(m, 0) + 1
            n += 1
    doc = {'schema': 'falcon.materialised/1', 'from': os.path.abspath(splits_parquet), 'split': split, 'files': n,
           'methods': used, 'merkle_root': meta.get('merkle_root'), 'hash_alg': meta.get('hash_alg'),
           'created': utc_now()}
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, 'materialised.json'), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    return doc


# ================================================================================================ roots and gc
def roots(workspace: str) -> set:
    """Every address the workspace still refers to: dataset versions' splits parquet files (frames, label files and
    label sets), run indexes and trainings' data.json. gc keeps exactly these."""
    out = set()

    def add(a):
        if a and a != NONE:
            out.add(a)
    ds = os.path.join(workspace, 'datasets')
    for spec in sorted(os.listdir(ds)) if os.path.isdir(ds) else ():
        sd = os.path.join(ds, spec, 'splits')
        for f in sorted(os.listdir(sd)) if os.path.isdir(sd) else ():
            if not f.endswith('.parquet'):
                continue
            rows, meta = read_table(os.path.join(sd, f))
            hasher = Hasher(meta.get('hash_alg') or C.default_alg()) if rows else None
            for r in rows:
                add(r['frame'])
                labels = json.loads(r['labels'] or '{}')
                for x in labels.values():
                    add(x.get('addr'))
                if labels:
                    add(label_set({k: x['addr'] for k, x in labels.items()}, hasher)[0])
                    add(r['label'])
    import glob
    for p in sorted(glob.glob(os.path.join(workspace, 'runs', '*', INDEX_FILE)) +
                    glob.glob(os.path.join(workspace, 'datasets', '*', 'drives', '*', INDEX_FILE))):
        for r in read_table(p)[0]:                   # a run's index sits at the run's top (index_path)
            add(r['frame'])
            add(r['label'])
            for x in json.loads(r['labels'] or '{}').values():
                add(x.get('addr'))
    tr = os.path.join(workspace, 'training')
    for d in sorted(os.listdir(tr)) if os.path.isdir(tr) else ():
        p = os.path.join(tr, d, 'data.json')
        if os.path.isfile(p):
            for _split, man in _data_splits(read_json(p)).items():
                for row in man.get('images') or ():
                    add(row.get('frame'))
                    add(row.get('label'))
    return out


def gc(workspace: str, dry_run: bool = False, grace_s: float = C.DEFAULT_GRACE_S) -> dict:
    store = C.Store(C.store_dir(workspace))
    keep = roots(workspace)
    out = store.gc(keep, dry_run=dry_run, grace_s=grace_s)
    out['referenced'] = len(keep)
    return out


# ================================================================================================ models
def _data_splits(doc: dict) -> dict:
    return {k: doc[k] for k in ('train', 'val', 'test') if isinstance(doc.get(k), dict)}


def onnx_metadata(path: str) -> dict:
    import onnx
    m = onnx.load(path, load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


def set_onnx_metadata(path: str, props: dict):
    """Write metadata_props (replacing keys of the same name) into an ONNX file in place."""
    import onnx
    m = onnx.load(path)
    keep = [(p.key, p.value) for p in m.metadata_props if p.key not in props]
    del m.metadata_props[:]
    for k, v in keep + sorted(props.items()):
        e = m.metadata_props.add()
        e.key, e.value = str(k), str(v)
    tmp = path + '.meta.part'
    onnx.save(m, tmp)
    os.replace(tmp, path)


def resolve_model(path: str) -> dict:
    """{'onnx', 'dir' (the training folder), 'model_json'} of a model.onnx, a model.json or a training folder."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        d = path
        mj = os.path.join(d, 'model.json')
        onnx_p = os.path.join(d, 'model.onnx')
    elif path.endswith('.json'):
        mj = path
        reg = read_json(path)
        onnx_p = reg.get('path') or ''
        d = (reg.get('source') or {}).get('training_run') or os.path.dirname(onnx_p or path)
    else:
        onnx_p, d = path, os.path.dirname(path)
        mj = os.path.join(d, 'model.json')
    if not os.path.isfile(os.path.join(d, 'data.json')):
        raise LineageError(f'{d}: no data.json (not a Falcon training folder)')
    return {'onnx': onnx_p if os.path.isfile(onnx_p) else None, 'dir': d,
            'model_json': mj if os.path.isfile(mj) else None}


def model_leaves(train_dir: str, split: str = 'train') -> tuple:
    """(sorted leaf bytes, rows, data.json) of a training's split."""
    doc = read_json(os.path.join(train_dir, 'data.json'))
    man = _data_splits(doc).get(split)
    if not man:
        raise LineageError(f'{train_dir}: data.json has no {split} split')
    rows = man.get('images') or []
    if rows and not rows[0].get('frame'):
        raise LineageError(f'{train_dir}: data.json has no per-sample hashes (trained before lineage)')
    return sorted(leaf_bytes(r.get('frame'), r.get('label')) for r in rows), rows, doc


def verify_model(path: str, reread: bool = True) -> dict:
    """A model's lineage checked: its ONNX metadata against data.json's roots, data.json's roots against its
    per-sample addresses, the registration's SHA-256 against the file and — reread — every training image still on
    disk re-hashed (labels as trained are pinned by data.json's addresses; core.training recomputes them on a re-run).
    {'ok', 'problems', 'checked', …}."""
    m = resolve_model(path)
    problems, checked = [], {}
    doc = read_json(os.path.join(m['dir'], 'data.json'))
    alg = doc.get('hash_alg')
    if not alg:
        raise LineageError(f'{m["dir"]}: data.json has no hash_alg (trained before lineage)')
    for split, man in _data_splits(doc).items():
        rows = man.get('images') or []
        root = pairs_root(((r.get('frame'), r.get('label')) for r in rows), alg)
        checked[f'{split}_root'] = root
        if man.get('merkle_root') != root:
            problems.append(f'data.json {split}: recorded root {man.get("merkle_root")} is not the samples\' {root}')
    meta = {}
    if m['onnx']:
        meta = onnx_metadata(m['onnx'])
        checked['onnx_metadata'] = {k: v for k, v in meta.items() if k.startswith('falcon.')}
        if meta.get('falcon.hash_alg') != alg:
            problems.append(f'ONNX falcon.hash_alg {meta.get("falcon.hash_alg")} is not data.json\'s {alg}')
        if meta.get('falcon.data_merkle_root') != checked.get('train_root'):
            problems.append(f'ONNX falcon.data_merkle_root {meta.get("falcon.data_merkle_root")} is not the training '
                            f'split\'s {checked.get("train_root")}')
        if doc.get('manifest_sha') and meta.get('falcon.manifest_sha') != doc.get('manifest_sha'):
            problems.append('ONNX falcon.manifest_sha is not data.json\'s')
        if m['model_json']:
            from .provenance import sha256_file
            want = (read_json(m['model_json']).get('source') or {}).get('sha256')
            have = sha256_file(m['onnx'])
            checked['onnx_sha256'] = have
            if want and want != have:
                problems.append(f'the ONNX file\'s SHA-256 {have} is not the registration\'s {want}')
    else:
        problems.append('no model.onnx beside data.json: only data.json was checked')
    if reread:
        hasher = Hasher(alg)
        changed = missing = 0
        for split, man in _data_splits(doc).items():
            for r in man.get('images') or ():
                img = r.get('image')
                if not img or not os.path.isfile(img):
                    missing += 1
                    continue
                if hasher.address(img)['addr'] != r.get('frame'):
                    changed += 1
        checked['images_reread'] = hasher.read
        checked['images_missing'] = missing
        if changed:
            problems.append(f'{changed} training image(s) on disk no longer hash to the address the model was '
                            'trained on')
    return {'ok': not problems, 'problems': problems, 'checked': checked, 'model': m, 'hash_alg': alg}


def prove(model_path: str, frame: str, split: str = 'train') -> dict:
    """An RFC 6962 audit path showing a frame (an image path, a sample id or a frame address) is a leaf of the
    model's training-split root; checked here with verify_path."""
    m = resolve_model(model_path)
    leaves, rows, doc = model_leaves(m['dir'], split)
    alg = doc['hash_alg']
    if C.is_address(frame):
        want = frame
    elif os.path.isfile(frame):
        want = C.hash_file(frame, alg)
    else:
        hit = next((r for r in rows if r.get('id') == frame), None)
        if hit is None:
            raise LineageError(f'{frame!r} is neither a file, an address nor a sample id of the {split} split')
        want = hit['frame']
    cands = [i for i, lb in enumerate(leaves) if lb.split(b'|', 1)[0].decode() == want]
    if not cands:
        return {'member': False, 'frame': want, 'split': split, 'leaves': len(leaves),
                'root': pairs_root([], alg) if not leaves else f'{alg}:{merkle_root(leaves, alg).hex()}'}
    i = cands[0]
    path = audit_path(leaves, i, alg)
    root = merkle_root(leaves, alg)
    ok = verify_path(leaves[i], i, len(leaves), path, root, alg)
    return {'member': True, 'verified': ok, 'frame': want, 'leaf': leaves[i].decode(), 'index': i,
            'leaves': len(leaves), 'path': [h.hex() for h in path], 'root': f'{alg}:{root.hex()}', 'split': split,
            'hash_alg': alg}


# ================================================================================================ diff
def leaves_of(path: str) -> dict:
    """{'kind', 'pairs': [(frame, label)], 'alg', 'label_kind'} of a model / training folder, a dataset version
    (its .json or splits .parquet) or an indexed run."""
    p = os.path.abspath(os.path.expanduser(path))
    if p.endswith('.parquet'):
        rows, meta = read_table(p)
        return {'kind': 'index', 'pairs': [(r['frame'], r['label']) for r in rows], 'alg': meta.get('hash_alg'),
                'label_kind': 'files'}
    if p.endswith('.json') and os.path.isfile(p):
        doc = read_json(p)
        if doc.get('schema') == 'falcon.dataset_version/1':
            lin = doc.get('lineage') or {}
            if not lin.get('splits'):
                raise LineageError(f'{p}: the version has no lineage (made before lineage, or without pyarrow)')
            ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p))))
            out = leaves_of(os.path.join(ws, lin['splits']))
            out['kind'] = 'version'
            return out
    if os.path.isdir(p) and os.path.isfile(index_path(p)):
        out = leaves_of(index_path(p))
        out['kind'] = 'run'
        return out
    m = resolve_model(p)
    _leaves, rows, doc = model_leaves(m['dir'])
    return {'kind': 'model', 'pairs': [(r.get('frame'), r.get('label')) for r in rows], 'alg': doc['hash_alg'],
            'label_kind': 'trained'}


def diff(a: str, b: str) -> dict:
    """What differs between two leaf sets: frames only in A or only in B, and (when both label alike) frames whose
    label differs."""
    A, B = leaves_of(a), leaves_of(b)
    if A['alg'] != B['alg']:
        raise LineageError(f'{a} is hashed with {A["alg"]} and {b} with {B["alg"]}: re-index one of them')
    fa = {}
    for f, lb in A['pairs']:
        fa.setdefault(f, set()).add(lb)
    fb = {}
    for f, lb in B['pairs']:
        fb.setdefault(f, set()).add(lb)
    only_a = sorted(set(fa) - set(fb) - {NONE, None})
    only_b = sorted(set(fb) - set(fa) - {NONE, None})
    common = set(fa) & set(fb) - {NONE, None}
    relabelled = sorted(f for f in common if fa[f] != fb[f]) if A['label_kind'] == B['label_kind'] else []
    return {'a': {'kind': A['kind'], 'samples': len(A['pairs']), 'root': pairs_root(A['pairs'], A['alg'])},
            'b': {'kind': B['kind'], 'samples': len(B['pairs']), 'root': pairs_root(B['pairs'], B['alg'])},
            'same': pairs_root(A['pairs'], A['alg']) == pairs_root(B['pairs'], B['alg']),
            'frames_only_in_a': len(only_a), 'frames_only_in_b': len(only_b), 'frames_in_both': len(common),
            'relabelled': len(relabelled), 'labels_compared': A['label_kind'] == B['label_kind'],
            'examples': {'only_in_a': only_a[:5], 'only_in_b': only_b[:5], 'relabelled': relabelled[:5]}}


# ======================================================================================= verify a version or run
def verify_version(version_json: str) -> dict:
    doc = read_json(version_json)
    lin = doc.get('lineage') or {}
    if not lin.get('splits'):
        raise LineageError(f'{version_json}: no lineage recorded')
    ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(version_json)))))
    rows, meta = read_table(os.path.join(ws, lin['splits']))
    alg = meta['hash_alg']
    problems = []
    root = pairs_root(((r['frame'], r['label']) for r in rows), alg)
    if root != lin.get('merkle_root') or root != meta.get('merkle_root'):
        problems.append(f'the splits\' root {root} is not the version\'s {lin.get("merkle_root")}')
    store = C.Store(C.store_dir(ws), alg)
    bad = missing = n = 0
    for r in rows:
        for a in [r['frame']] + [x['addr'] for x in json.loads(r['labels'] or '{}').values()]:
            if a == NONE:
                continue
            n += 1
            if not store.has(a):
                missing += 1
            elif not store.verify(a):
                bad += 1
    if missing:
        problems.append(f'{missing} object(s) missing from the store')
    if bad:
        problems.append(f'{bad} object(s) in the store no longer hash to their address')
    return {'ok': not problems, 'problems': problems, 'root': root, 'objects_checked': n, 'hash_alg': alg}


def verify_run(run_dir: str) -> dict:
    """Re-hash every file of an indexed run (no cache) and compare the root with its index's."""
    _rows, meta = read_index(run_dir)
    hasher = Hasher(meta['hash_alg'])
    rows = scan_run(run_dir, hasher)
    root = pairs_root(((r['frame'], r['label']) for r in rows), hasher.alg)
    ok = root == meta.get('merkle_root')
    return {'ok': ok, 'problems': [] if ok else [f'the run\'s files now give root {root}, the index says '
                                                 f'{meta.get("merkle_root")}'], 'root': root, 'rows': len(rows)}


# ================================================================================================ the command
def main_lineage(argv) -> int:
    """`falcon lineage index|verify|prove|diff|materialise|gc` (module doc; docs/lineage.md)."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon lineage', description='content-addressed lineage of runs, dataset '
                                                                    'versions and models (docs/lineage.md)')
    sub = ap.add_subparsers(dest='what', required=True)
    i = sub.add_parser('index', help='hash every recorded frame of runs into RUN/frames.parquet (run.json untouched)')
    i.add_argument('runs', nargs='+')
    i.add_argument('--full', action='store_true', help='re-hash every file, even unchanged ones')
    v = sub.add_parser('verify', help='check a model (training folder, model.json or model.onnx), a dataset '
                                      'version (vNNN.json) or an indexed run')
    v.add_argument('target')
    v.add_argument('--quick', action='store_true', help='models: do not re-read the training images')
    pr = sub.add_parser('prove', help='an RFC 6962 audit path: a frame is among a model\'s training samples')
    pr.add_argument('model')
    pr.add_argument('frame', help='an image path, a sample id or a frame address')
    pr.add_argument('--split', default='train', choices=('train', 'val', 'test'))
    d = sub.add_parser('diff', help='what differs between two models, versions or indexed runs')
    d.add_argument('a')
    d.add_argument('b')
    m = sub.add_parser('materialise', help='a dataset version\'s files rebuilt from the store')
    m.add_argument('version', help='datasets/<spec>/versions/vNNN.json')
    m.add_argument('dest')
    grp = m.add_mutually_exclusive_group()
    for how in ('hardlink', 'reflink', 'copy', 'symlink'):
        grp.add_argument(f'--{how}', dest='how', action='store_const', const=how)
    m.add_argument('--split', default=None)
    g = sub.add_parser('gc', help='remove store objects nothing refers to (mark and sweep)')
    g.add_argument('workspace')
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--grace', type=float, default=C.DEFAULT_GRACE_S,
                   help='keep objects linked within this many seconds (default %(default)g)')
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return P.OK if ex.code == 0 else P.FAILED
    try:
        if a.what == 'index':
            for r in a.runs:
                s = index_run(r, reuse=not a.full)
                print(json.dumps({'run': r, **{k: s[k] for k in ('rows', 'frames', 'merkle_root', 'hash_alg',
                                                                  'files_read', 'missing_files', 'seconds')}}))
            return P.OK
        if a.what == 'verify':
            t = os.path.abspath(a.target)
            if t.endswith('.json') and os.path.isfile(t) and \
                    read_json(t).get('schema') == 'falcon.dataset_version/1':
                res = verify_version(t)
            elif os.path.isdir(t) and os.path.isfile(index_path(t)) and \
                    not os.path.isfile(os.path.join(t, 'data.json')):
                res = verify_run(t)
            else:
                res = verify_model(t, reread=not a.quick)
                res.pop('model', None)
            print(json.dumps(res, indent=1))
            return P.OK if res['ok'] else P.FAILED
        if a.what == 'prove':
            res = prove(a.model, a.frame, a.split)
            print(json.dumps(res, indent=1))
            return P.OK if res.get('member') and res.get('verified') else P.FAILED
        if a.what == 'diff':
            print(json.dumps(diff(a.a, a.b), indent=1))
            return P.OK
        if a.what == 'materialise':
            t = os.path.abspath(a.version)
            doc = read_json(t)
            lin = doc.get('lineage') or {}
            if not lin.get('splits'):
                raise LineageError(f'{t}: no lineage recorded')
            ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(t))))
            res = materialise(os.path.join(ws, lin['splits']), a.dest, C.Store(C.store_dir(ws), lin['hash_alg']),
                              a.how or 'hardlink', a.split)
            print(json.dumps(res, indent=1))
            return P.OK
        res = gc(a.workspace, a.dry_run, a.grace)
        res['removed'] = len(res['removed'])
        print(json.dumps(res, indent=1))
        return P.OK
    except (LineageError, C.CASError, OSError, ValueError, KeyError) as ex:
        print(f'failed: {ex}', flush=True)
        return P.FAILED
