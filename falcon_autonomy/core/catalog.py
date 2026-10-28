"""The catalogue: a local SQLite index of a workspace's runs, datasets, models, evaluations and experiments.

The files on disk are the truth (each run folder's run.json, each dataset's manifest); the catalogue indexes them so
the GUI can list, filter and link them, and can be rebuilt from the workspace at any time (rebuild()). Every row keeps
the path of the JSON it was read from, so every number shown traces back to the run that produced it.
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
        return self.get(ident)

    def rebuild(self, workspace: str) -> int:
        """Re-index every run folder under workspace/runs (the files are the truth). Returns the count."""
        self._exec("DELETE FROM entries WHERE kind = 'run'", commit=True)
        runs = os.path.join(workspace, 'runs')
        n = 0
        if os.path.isdir(runs):
            for d in sorted(os.listdir(runs)):
                if self.add_run(os.path.join(runs, d)):
                    n += 1
        return n


def _row(r) -> dict:
    d = dict(r)
    for k in ('summary', 'doc'):
        try:
            d[k] = json.loads(d[k] or '{}')
        except ValueError:
            d[k] = {}
    return d
