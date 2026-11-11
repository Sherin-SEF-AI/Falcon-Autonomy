"""The catalogue: a local SQLite index of a workspace's runs, datasets, models, evaluations and experiments.

The files on disk are the truth (each run folder's run.json, each dataset's manifest); the catalogue indexes them so
the GUI can list, filter and link them, and can be rebuilt from the workspace at any time (rebuild()). Every row keeps
the path of the JSON it was read from, so every number shown traces back to the run that produced it.

Where it lives (catalog_path): <workspace>/catalog.sqlite; in a team workspace on a shared folder (core.team) a
per-user file in the cache folder instead, rebuilt from the run folders when the workspace is opened — SQLite's locking
is not reliable over NFS or SMB (sqlite.org/faq.html#q5), and the files are the truth anyway.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading

from .provenance import read_json, utc_now

SCHEMA_VERSION = 1
KINDS = ('run', 'dataset', 'model', 'evaluation', 'experiment')

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT,
    created TEXT,
    parent TEXT,
    summary TEXT,
    doc TEXT
);
CREATE INDEX IF NOT EXISTS entries_kind ON entries(kind, created);
CREATE TABLE IF NOT EXISTS links (src TEXT, dst TEXT, role TEXT, PRIMARY KEY (src, dst, role));
"""


def catalog_path(workspace: str) -> str:
    """The catalogue's file for a workspace: inside it, or — for a team workspace (core.team.is_team) — this user's
    local copy, keyed by the workspace's real path."""
    import hashlib
    from .team import is_team
    if workspace and is_team(workspace):
        from .paths import cache_dir
        key = hashlib.sha256(os.path.realpath(workspace).encode()).hexdigest()[:16]
        return os.path.join(cache_dir(), 'catalogues', f'{os.path.basename(os.path.normpath(workspace))}-{key}.sqlite')
    return os.path.join(workspace, 'catalog.sqlite')


class Catalog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # jobs finish on launcher threads: one connection shared under a lock
        self.db = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
        self._lock = threading.RLock()
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_DDL)
        self.db.execute('INSERT OR IGNORE INTO meta VALUES (?, ?)', ('schema_version', str(SCHEMA_VERSION)))
        self.db.commit()

    def close(self):
        with self._lock:
            self.db.close()

    def _exec(self, sql, args=(), commit=False):
        with self._lock:
            cur = self.db.execute(sql, args)
            rows = cur.fetchall()
            if commit:
                self.db.commit()
            return rows

    def add(self, ident: str, kind: str, name: str, path: str, status: str = '', created: str = '',
            parent: str = '', summary: dict | None = None, doc: dict | None = None):
        if kind not in KINDS:
            raise ValueError(f'unknown catalogue kind {kind!r}')
        self._exec('INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   (ident, kind, name, path, status, created or utc_now(), parent,
                    json.dumps(summary or {}), json.dumps(doc or {}, default=str)), commit=True)

    def link(self, src: str, dst: str, role: str):
        self._exec('INSERT OR IGNORE INTO links VALUES (?, ?, ?)', (src, dst, role), commit=True)

    def remove(self, ident: str):
        self._exec('DELETE FROM entries WHERE id = ?', (ident,))
        self._exec('DELETE FROM links WHERE src = ? OR dst = ?', (ident, ident), commit=True)

    def get(self, ident: str) -> dict | None:
        rows = self._exec('SELECT * FROM entries WHERE id = ?', (ident,))
        return _row(rows[0]) if rows else None

    def list(self, kind: str | None = None, limit: int = 1000) -> list:
        if kind:
            rows = self._exec('SELECT * FROM entries WHERE kind = ? ORDER BY created DESC LIMIT ?', (kind, limit))
        else:
            rows = self._exec('SELECT * FROM entries ORDER BY created DESC LIMIT ?', (limit,))
        return [_row(r) for r in rows]

    def links(self, ident: str) -> list:
        rows = self._exec('SELECT * FROM links WHERE src = ? OR dst = ?', (ident, ident))
        return [dict(r) for r in rows]

    # -- runs
    def add_run(self, run_dir: str) -> dict | None:
        """Index a run folder by its run.json (None when it has none)."""
        p = os.path.join(run_dir, 'run.json')
        if not os.path.isfile(p):
            return None
        try:
            r = read_json(p)
        except (OSError, ValueError):
            return None
        ident = 'run:' + os.path.basename(os.path.normpath(run_dir))
        summary = {k: r.get(k) for k in ('worker', 'status', 'exit_code', 'error')}
        summary.update(r.get('summary') or {})
        self.add(ident, 'run', r.get('name') or os.path.basename(run_dir), run_dir, r.get('status', ''),
                 r.get('started', ''), '', summary, None)
        if r.get('status') == 'ok' and os.path.isfile(os.path.join(run_dir, 'model.json')):
            self.add_model(run_dir)              # a training that registered a model: its lineage row too
        return self.get(ident)

    # -- experiments (the registry: core.ablation)
    def add_experiment(self, exp_dir: str) -> dict | None:
        """Index an experiment folder (experiment.json) of any kind: its id, name, status and, for an ablation, the
        hypothesis, factors, metric and verdict (core.ablation.summary). A re-run is linked to its original."""
        from .ablation import summary
        if not os.path.isfile(os.path.join(exp_dir, 'experiment.json')):
            return None
        s = summary(exp_dir)
        ident = 'experiment:' + str(s['id'])
        self.add(ident, 'experiment', s['name'] or s['id'], exp_dir, s['status'], s['created'] or '', '',
                 {k: s.get(k) for k in ('kind', 'hypothesis', 'factors', 'metric', 'verdict', 'runs', 'planned',
                                        'rerun_of')})
        if s.get('rerun_of'):
            self.link(ident, 'experiment:' + str(s['rerun_of']), 'rerun_of')
        return self.get(ident)

    # -- dataset versions and trained models (their content lineage: core.lineage)
    def add_dataset(self, version_json: str) -> dict | None:
        """Index a dataset version (datasets/<spec>/versions/vNNN.json): its splits' sizes, its SHA-256 and its
        content lineage (Merkle root, hash algorithm, samples). Id dataset:<spec>@v<k>."""
        try:
            d = read_json(version_json)
        except (OSError, ValueError):
            return None
        if d.get('schema') != 'falcon.dataset_version/1':
            return None
        ident = f'dataset:{d.get("spec")}@v{d.get("version")}'
        lin = d.get('lineage') or {}
        summary = {'spec': d.get('spec'), 'version': d.get('version'), 'split_by': d.get('split_by'),
                   'splits': {k: len(v) for k, v in (d.get('splits') or {}).items()}, 'sha256': d.get('sha256'),
                   'merkle_root': lin.get('merkle_root'), 'hash_alg': lin.get('hash_alg'), 'samples': lin.get('rows'),
                   'split_roots': lin.get('roots')}
        self.add(ident, 'dataset', f'{d.get("name")} v{d.get("version")}', version_json, 'ok', d.get('created', ''),
                 '', summary)
        return self.get(ident)

    def add_model(self, train_dir: str) -> dict | None:
        """Index a trained model (training/<id>/ with model.json and data.json): its data's Merkle root, hash
        algorithm, dataset version and manifest SHA-256 (as the ONNX file's metadata states them), linked to the
        dataset version it was trained on (role trained_on). Id model:<training id>."""
        mj, dj = os.path.join(train_dir, 'model.json'), os.path.join(train_dir, 'data.json')
        if not os.path.isfile(mj):
            return None
        try:
            reg = read_json(mj)
            data = read_json(dj) if os.path.isfile(dj) else {}
        except (OSError, ValueError):
            return None
        ident = 'model:' + os.path.basename(os.path.normpath(train_dir))
        summary = {'registration': reg.get('id'), 'onnx_sha256': (reg.get('source') or {}).get('sha256'),
                   'merkle_root': (data.get('train') or {}).get('merkle_root'), 'hash_alg': data.get('hash_alg'),
                   'val_merkle_root': (data.get('val') or {}).get('merkle_root'),
                   'manifest_sha': data.get('manifest_sha'), 'dataset_version': data.get('dataset_version'),
                   'images': (data.get('train') or {}).get('n_images')}
        created = ''
        try:
            created = read_json(os.path.join(train_dir, 'run.json')).get('started') or ''
        except (OSError, ValueError):
            pass
        self.add(ident, 'model', reg.get('name') or reg.get('id') or ident, train_dir, 'ok', created, '', summary)
        ws =os.path.dirname(os.path.dirname(os.path.abspath(train_dir)))
        for entry in self.list('dataset'):
            dv = data.get('dataset_version') or ''
            s = entry['summary']
            if s.get('sha256') and str(s['sha256'])[:12] in dv and os.path.realpath(entry['path']).startswith(
                    os.path.realpath(ws)):
                self.link(ident, entry['id'], 'trained_on')
        return self.get(ident)

    def rebuild(self, workspace: str) -> int:
        """Re-index every run folder under workspace/runs, every experiment under workspace/experiments, every
        dataset version and every trained model (the files are the truth). Returns the count of runs."""
        self._exec("DELETE FROM entries WHERE kind IN ('dataset', 'model')", commit=True)
        ds = os.path.join(workspace, 'datasets')
        for spec in sorted(os.listdir(ds)) if os.path.isdir(ds) else ():
            vd = os.path.join(ds, spec, 'versions')
            for f in sorted(os.listdir(vd)) if os.path.isdir(vd) else ():
                if f.endswith('.json'):
                    self.add_dataset(os.path.join(vd, f))
        tr = os.path.join(workspace, 'training')
        for d in sorted(os.listdir(tr)) if os.path.isdir(tr) else ():
            self.add_model(os.path.join(tr, d))
        self._exec("DELETE FROM entries WHERE kind = 'run'", commit=True)
        runs = os.path.join(workspace, 'runs')
        n = 0
        if os.path.isdir(runs):
            for d in sorted(os.listdir(runs)):
                if self.add_run(os.path.join(runs, d)):
                    n += 1
        exps = os.path.join(workspace, 'experiments')
        self._exec("DELETE FROM entries WHERE kind = 'experiment' AND id LIKE 'experiment:%'", commit=True)
        if os.path.isdir(exps):
            for d in sorted(os.listdir(exps)):
                try:
                    self.add_experiment(os.path.join(exps, d))
                except (OSError, ValueError, KeyError):
                    continue
        return n


def _row(r) -> dict:
    d = dict(r)
    for k in ('summary', 'doc'):
        try:
            d[k] = json.loads(d[k] or '{}')
        except ValueError:
            d[k] = {}
    return d
