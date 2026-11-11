"""The Models page's Lineage tab: which bytes each trained model and each dataset version were made from
(core.lineage, docs/lineage.md).

The table lists the workspace's trained models (training/<id>/ with data.json) and dataset versions
(datasets/<spec>/versions/vNNN.json) with their Merkle root, hash algorithm, sample count and — for a model — the
dataset version it names and whether its ONNX metadata carries the root. Verify re-checks the selected row in the
background (a model: its ONNX metadata, data.json's roots and its images re-hashed; a version: its splits against
the store); Prove a frame… shows the audit path for an image; Index runs queues the cheap index job for every
recorded run not indexed yet; Store shows the content-addressed store and what a garbage collection would remove
(nothing is removed from here: `falcon lineage gc WORKSPACE`)."""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import cas as C
from ..core import lineage as L
from ..core.provenance import read_json
from . import tasks
from .widgets import KeyValues, Section, button, label

COLUMNS = ('Kind', 'Name', 'Merkle root', 'Hash', 'Samples', 'Dataset version', 'Check')


def rows(workspace: str) -> list:
    """[{'kind', 'name', 'path', 'root', 'alg', 'samples', 'version', 'onnx'}] of the workspace's models and dataset
    versions (files only: nothing re-hashed here)."""
    out = []
    tr = os.path.join(workspace, 'training')
    for d in sorted(os.listdir(tr)) if os.path.isdir(tr) else ():
        p = os.path.join(tr, d)
        dj = os.path.join(p, 'data.json')
        if not os.path.isfile(dj):
            continue
        try:
            data = read_json(dj)
        except (OSError, ValueError):
            continue
        train = data.get('train') or {}
        onnx_p = os.path.join(p, 'model.onnx')
        out.append({'kind': 'model', 'name': d, 'path': p, 'root': train.get('merkle_root'),
                    'alg': data.get('hash_alg'), 'samples': train.get('n_images'),
                    'version': data.get('dataset_version'), 'onnx': onnx_p if os.path.isfile(onnx_p) else None})
    ds = os.path.join(workspace, 'datasets')
    for spec in sorted(os.listdir(ds)) if os.path.isdir(ds) else ():
        vd = os.path.join(ds, spec, 'versions')
        for f in sorted(os.listdir(vd)) if os.path.isdir(vd) else ():
            if not f.endswith('.json'):
                continue
            p = os.path.join(vd, f)
            try:
                v = read_json(p)
            except (OSError, ValueError):
                continue
            lin = v.get('lineage') or {}
            out.append({'kind': 'dataset version', 'name': f'{v.get("name")} v{v.get("version")}', 'path': p,
                        'root': lin.get('merkle_root'), 'alg': lin.get('hash_alg'), 'samples': lin.get('rows'),
                        'version': None, 'onnx': None})
    return out


def short(root) -> str:
    return '—' if not root else f'{root.split(":", 1)[0]}:{root.split(":", 1)[-1][:12]}…'


class LineageTab(QWidget):
    """The table on top, the selected row's details and the actions below (module doc)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.items = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        sec = Section('Lineage', 'Every model and dataset version names the bytes it was made from: a Merkle root '
                                 '(RFC 6962) over its samples\' image and label hashes, in the model\'s ONNX metadata '
                                 'and the version\'s splits file. Verify re-hashes; a proof shows one frame is among a '
                                 'model\'s training samples.')
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.itemSelectionChanged.connect(self.show_selected)
        sec.add(self.tree)
        b = QHBoxLayout()
        self.verify_btn = button('Verify', self.verify, tip='falcon lineage verify: re-check the selected model or '
                                                            'version (images re-hashed)')
        self.prove_btn = button('Prove a frame…', self.prove, tip='falcon lineage prove: the audit path for an image')
        self.index_btn = button('Index runs', self.index_runs, tip='queue falcon lineage index for every recorded run '
                                                                  'without an index (a cheap CPU job each)')
        self.store_btn = button('Store', self.show_store, tip='the content-addressed store and what a garbage '
                                                              'collection would remove (nothing is removed here)')
        self.refresh_btn = button('Refresh', self.refresh)
        for x in (self.verify_btn, self.prove_btn, self.index_btn, self.store_btn, self.refresh_btn):
            b.addWidget(x)
        b.addStretch(1)
        sec.add(b)
        self.info = KeyValues()
        sec.add(self.info)
        self.note = label('', muted=True, wrap=True)
        sec.add(self.note)
        v.addWidget(sec, 1)
        self.refresh()

    def workspace(self) -> str:
        try:
            return self.win.workspace() or ''
        except AttributeError:
            return ''

    def refresh(self):
        self.tree.clear()
        ws = self.workspace()
        self.items = rows(ws) if ws else []
        for r in self.items:
            meta = '—'
            if r['onnx']:
                try:
                    m = L.onnx_metadata(r['onnx'])
                    meta = ('root in ONNX' if m.get('falcon.data_merkle_root') == r['root'] and r['root'] else
                            'ONNX names another root' if m.get('falcon.data_merkle_root') else 'ONNX without lineage')
                except Exception:              # noqa: BLE001 (onnx missing or the file unreadable: shown, not raised)
                    meta = 'ONNX unreadable'
            it = QTreeWidgetItem([r['kind'], r['name'], short(r['root']), r['alg'] or '—',
                                  '—' if r['samples'] is None else str(r['samples']), r['version'] or '—', meta])
            it.setToolTip(2, r['root'] or 'made before lineage')
            it.setToolTip(1, r['path'])
            self.tree.addTopLevelItem(it)
        for k in range(len(COLUMNS)):
            self.tree.resizeColumnToContents(k)
        self.note.setText('' if self.items else 'No trained models or dataset versions in this workspace yet.')

    def selected(self) -> dict | None:
        it = self.tree.currentItem()
        if it is None:
            return None
        i = self.tree.indexOfTopLevelItem(it)
        return self.items[i] if 0 <= i < len(self.items) else None

    def show_selected(self):
        r = self.selected()
        if r is None:
            return
        self.info.set('Selected', r['name'], tip=r['path'])
        self.info.set('Merkle root', r['root'] or 'none (made before lineage, or without pyarrow)')

    # ---------------------------------------------------------------- actions
    def verify(self):
        r = self.selected()
        if r is None:
            self.note.setText('choose a model or a dataset version first')
            return None
        path = r['path']
        fn = (lambda: L.verify_version(path)) if r['kind'] == 'dataset version' else (lambda: L.verify_model(path))
        self.note.setText(f'verifying {r["name"]}…')
        return tasks.run(self, fn, lambda res: self.show_check(r, res), lambda e: self.note.setText(f'failed: {e}'))

    def show_check(self, r, res):
        self.info.set('Check', 'verified' if res['ok'] else 'PROBLEMS: ' + '; '.join(res['problems']))
        self.note.setText(f'{r["name"]}: ' + ('every hash checks out' if res['ok'] else
                                              f'{len(res["problems"])} problem(s)'))

    def prove(self, image: str | None = None):
        r = self.selected()
        if r is None or r['kind'] != 'model':
            self.note.setText('choose a trained model first')
            return None
        if image is None:
            image, _ = QFileDialog.getOpenFileName(self, 'An image', self.workspace(), 'Images (*.jpg *.png)')
        if not image:
            return None
        try:
            res = L.prove(r['path'], image)
        except (L.LineageError, OSError, ValueError, C.CASError) as e:
            self.note.setText(f'failed: {e}')
            return None
        if res['member']:
            self.info.set('Proof', f'leaf {res["index"] + 1} of {res["leaves"]}, {len(res["path"])} hashes to the '
                                   f'root — ' + ('verified' if res['verified'] else 'DOES NOT VERIFY'))
        else:
            self.info.set('Proof', f'not among the model\'s {res["leaves"]} training samples')
        return res

    def index_runs(self):
        ws = self.workspace()
        runs = os.path.join(ws, 'runs') if ws else ''
        todo = [os.path.join(runs, d) for d in sorted(os.listdir(runs))] if runs and os.path.isdir(runs) else []
        todo = [d for d in todo if os.path.isfile(os.path.join(d, 'frames.jsonl')) and
                not os.path.isfile(L.index_path(d))]
        for d in todo:
            self.win.queue.submit(L.index_job_spec(d))
        self.note.setText(f'{len(todo)} run(s) queued for indexing' if todo else 'every recorded run is indexed')
        return len(todo)

    def show_store(self):
        ws = self.workspace()
        if not ws:
            return None

        def work():
            st = C.Store(C.store_dir(ws)).stats()
            st['gc'] = L.gc(ws, dry_run=True)
            return st

        def done(st):
            g = st['gc']
            self.info.set('Store', f'{st["objects"]} objects, {st["bytes"] / 2 ** 20:.1f} MiB ({st["alg"]}); a '
                                   f'garbage collection would remove {len(g["removed"])} '
                                   f'({g["bytes_freed"] / 2 ** 20:.1f} MiB), keep {g["kept"]}', tip=st['root'])
        return tasks.run(self, work, done, lambda e: self.note.setText(f'failed: {e}'))
