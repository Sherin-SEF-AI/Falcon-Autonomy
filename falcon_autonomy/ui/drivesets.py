"""Drive sets and tags in the GUI: the run index of the open project's workspace (core.runindex), the "Drive set"
bar shown above every list of drives (choose a set: its drives are checked; save the checked drives as a set), and
the project edits for sets and tags (undoable, like every project change)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QInputDialog, QListWidget, QWidget

from ..core import runindex as RI
from ..core.project import Edit, new_id
from ..core.provenance import utc_now
from .widgets import button, label


def index_entries(win, tags: bool = True) -> list:
    """Every run of the workspace (and any the catalogue knows elsewhere), newest first, with the project's tags."""
    ws = win.workspace()
    cat = win.service.catalog
    extra = [r['path'] for r in cat.list('run', 5000)] if cat is not None else []
    out = RI.build(ws, extra)
    return RI.with_tags(out, win.project.doc, ws) if tags else out


def drive_sets(win) -> list:
    return list(win.project.doc.get('drive_sets') or ())


def add_set(win, name: str, kind: str, runs=(), query: str = '') -> dict:
    ws = win.workspace()
    ds = {'id': new_id('ds'), 'name': name, 'kind': kind, 'created': utc_now()}
    if kind == 'query':
        RI.parse(query)                      # raises QueryError on a malformed query
        ds['query'] = query
    else:
        ds['runs'] = [RI.key_of(p, ws) for p in runs]
    if 'drive_sets' in win.project.doc:
        win.edit(Edit('insert', ['drive_sets'], ds, text=f'Add the drive set {name}'))
    else:
        win.edit(Edit('set', ['drive_sets'], [ds], text=f'Add the drive set {name}'))
    return ds


def remove_set(win, ident: str):
    i, ds = win.project.find('drive_sets', ident)
    if ds is not None:
        win.edit(Edit('remove', ['drive_sets'], index=i, text=f'Remove the drive set {ds["name"]}'))


def resolve(win, ds: dict, entries: list | None = None) -> list:
    return RI.resolve_set(ds, entries if entries is not None else index_entries(win), win.workspace())


def set_tags(win, paths, tags: list):
    """Give these runs exactly these tags (an empty list removes them); one undoable edit."""
    ws = win.workspace()
    cur = dict(win.project.doc.get('run_tags') or {})
    clean = sorted({t.strip() for t in tags if t.strip()})
    for p in paths:
        k = RI.key_of(p, ws)
        if clean:
            cur[k] = clean
        else:
            cur.pop(k, None)
    n = len(list(paths))
    win.edit(Edit('set', ['run_tags'], cur, text=f'Tag {n} run{"s" if n != 1 else ""}'))


class DriveSetBar(QWidget):
    """Above a list of checkable drives (item data 256: the run's path): pick a drive set to check its drives, or save
    the checked drives as a new set."""

    def __init__(self, win, runs: QListWidget, parent=None):
        super().__init__(parent)
        self.win, self.runs = win, runs
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(label('Drive set', muted=True))
        self.combo = QComboBox()
        self.combo.setToolTip('Check the drives of a saved set (Data ▸ Browse makes sets from a selection or a '
                              'search; docs/runs.md)')
        self.combo.activated.connect(self._chosen)
        h.addWidget(self.combo, 1)
        self.save_btn = button('Save checked…', self.save, tip='Save the checked drives as a named drive set')
        h.addWidget(self.save_btn)
        self.status = label('', muted=True)
        h.addWidget(self.status)
        self.refresh()
        win.projectChanged.connect(self._on_project)              # sets added or removed anywhere

    def _on_project(self, _path):
        self.refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def refresh(self):
        cur = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        sets = drive_sets(self.win)
        self.combo.addItem('Choose a set…' if sets else 'No drive sets yet', None)
        for ds in sets:
            self.combo.addItem(f'{ds["name"]} ({RI.describe_set(ds)})', ds['id'])
        k = self.combo.findData(cur)
        self.combo.setCurrentIndex(k if k >= 0 else 0)
        self.combo.blockSignals(False)

    def _chosen(self, _i=None):
        ident = self.combo.currentData()
        ds = self.win.project.item('drive_sets', ident) if ident else None
        if ds is None:
            return
        self.apply(ds)

    def apply(self, ds: dict) -> tuple:
        """Check the set's drives in the list; (checked, in the set)."""
        want = set(resolve(self.win, ds))
        n = 0
        for i in range(self.runs.count()):
            it = self.runs.item(i)
            on = it.data(256) in want
            n += on
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        missing = len(want) - n
        self.status.setText(f'{n} checked' + (f'; {missing} not listed here' if missing else ''))
        self.status.setToolTip('Drives of the set that this list does not offer (not finished, or not a recorded '
                               'drive) stay unchecked.' if missing else '')
        return n, len(want)

    def checked(self) -> list:
        return [self.runs.item(i).data(256) for i in range(self.runs.count())
                if self.runs.item(i).checkState() == Qt.CheckState.Checked]

    def save(self):
        runs = self.checked()
        if not runs:
            self.status.setText('check some drives first')
            return
        name, ok = QInputDialog.getText(self, 'Save a drive set', f'A name for these {len(runs)} drives:')
        if ok and name.strip():
            ds = add_set(self.win, name.strip(), 'list', runs)
            self.refresh()
            self.combo.setCurrentIndex(self.combo.findData(ds['id']))
            self.status.setText(f'saved {len(runs)} drives')
