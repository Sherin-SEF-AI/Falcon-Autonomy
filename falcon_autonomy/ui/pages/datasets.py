"""Data ▸ Datasets: dataset specs as coverage targets, their plan and cost, running them (cached, resumable), the
coverage dashboard (what you have against what you asked for) and dataset versions with leak-free splits."""
from __future__ import annotations

import copy

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox,
                             QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import budget
from ...core import datasets as D
from ...core import weather as W
from ...core.project import Edit, new_id
from ...sim import engine as E
from .. import tasks
from ..theme import SEQ, STATUS, THEME
from ..widgets import Form, KeyValues, Section, button, label, set_enabled


class CoverageBars(QWidget):
    """Per class: what the dataset has against its target (a bar per class, the target as a tick)."""

    def __init__(self):
        super().__init__()
        self.rows = []                                   # (class, have, target)
        self.setMinimumHeight(140)

    def set_rows(self, rows):
        self.rows = rows
        self.setMinimumHeight(28 + 22 * max(len(rows), 1))
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        if not self.rows:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No finished drives yet')
            return
        x0, w = 120, self.width() - 220
        for i, (cls, have, target) in enumerate(self.rows):
            y = 12 + 22 * i
            p.setPen(QColor(THEME.tok['ink2']))
            p.drawText(QRectF(8, y, 108, 18), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, cls)
            p.fillRect(QRectF(x0, y + 3, w, 12), QColor(THEME.tok['raised']))
            frac = have / target if target else 1.0
            col = QColor(STATUS['good'] if frac >= 1 else SEQ[7])
            p.fillRect(QRectF(x0, y + 3, w * min(frac, 1.0), 12), col)
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(QRectF(x0 + w + 8, y, 100, 18), Qt.AlignmentFlag.AlignVCenter,
                       f'{have} / {target}' if target else f'{have} (no target)')


class DatasetsTab(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.win = page, page.win
        self.plan = None
        h = QHBoxLayout(self)
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.setFixedWidth(220)
        self.list.currentRowChanged.connect(lambda _: self.show_spec())
        left.addWidget(self.list, 1)
        lb = QHBoxLayout()
        lb.addWidget(button('New spec', self.add_spec))
        lb.addWidget(button('Remove', self.remove_spec))
        left.addLayout(lb)
        h.addLayout(left)
        body = QVBoxLayout()
        h.addLayout(body, 1)

        self.edit_sec = Section('Spec', 'What to collect, as targets. The engine plans drives over every world × '
                                        'weather cell, estimates their cost, and fills the targets.')
        f = Form()
        self.rig = QComboBox()
        self.rig.activated.connect(lambda _: self.set('rig', self.rig.currentData()))
        f.row('Rig', self.rig)
        self.drive_s = QDoubleSpinBox()
        self.drive_s.setRange(1, 3600)
        self.drive_s.setSuffix(' s')
        self.drive_s.editingFinished.connect(lambda: self.set('drive_s', self.drive_s.value()))
        f.row('Drive length', self.drive_s)
        self.every = QSpinBox()
        self.every.setRange(1, 100)
        self.every.editingFinished.connect(lambda: self.set('record_every', self.every.value()))
        f.row('Record every', self.every, 'steps (at 20 Hz: 5 keeps 4 frames a second)')
        self.max_drives = QSpinBox()
        self.max_drives.setRange(1, 10000)
        self.max_drives.editingFinished.connect(lambda: self.set('max_drives', self.max_drives.value()))
        f.row('At most', self.max_drives, 'drives')
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.editingFinished.connect(lambda: self.set('seed', self.seed.value()))
        f.row('Seed', self.seed)
        cells = QHBoxLayout()
        self.worlds = QListWidget()
        self.worlds.setMaximumHeight(110)
        self.worlds.itemChanged.connect(lambda _: self.set_checked('worlds', self.worlds))
        self.weathers = QListWidget()
        self.weathers.setMaximumHeight(110)
        self.weathers.itemChanged.connect(lambda _: self.set_checked('weathers', self.weathers))
        cells.addWidget(self.worlds)
        cells.addWidget(self.weathers)
        f.row('Worlds × weathers', cells)
        self.targets = QTableWidget(0, 2)
        self.targets.setHorizontalHeaderLabels(['Class', 'Objects'])
        self.targets.setMaximumHeight(150)
        self.targets.cellChanged.connect(lambda *_: self.set_targets())
        tb = QVBoxLayout()
        tb.addWidget(self.targets)
        tb.addWidget(button('Add a class', lambda: self.targets.insertRow(self.targets.rowCount())))
        f.row('Targets', tb, 'objects seen in recorded camera frames')
        self.edit_sec.add(f)
        body.addWidget(self.edit_sec)

        ps = Section('Plan and run')
        b = QHBoxLayout()
        b.addWidget(button('Plan', self.make_plan, tip='How many drives the targets need, and what they cost'))
        self.run_btn = button('Run the plan', self.run, role='primary')
        set_enabled(self.run_btn, False, 'Make a plan first')
        b.addWidget(self.run_btn)
        b.addStretch(1)
        ps.add(b)
        self.plan_kv = KeyValues()
        ps.add(self.plan_kv)
        body.addWidget(ps)

        cs = Section('Coverage', 'What the finished drives hold against the targets; per distance bin below.')
        self.bars = CoverageBars()
        cs.add(self.bars)
        self.bins = QTableWidget(0, 0)
        self.bins.setMaximumHeight(160)
        cs.add(self.bins)
        body.addWidget(cs)

        vs = Section('Versions', 'A version freezes the finished drives with a split that never puts one drive (or, '
                                 'by place, one world) in two splits.')
        vb = QHBoxLayout()
        self.split_by = QComboBox()
        self.split_by.addItems(['drive', 'place'])
        self.train = QDoubleSpinBox()
        self.train.setRange(0.05, 0.95)
        self.train.setValue(0.7)
        self.train.setSingleStep(0.05)
        vb.addWidget(label('Split by', muted=True))
        vb.addWidget(self.split_by)
        vb.addWidget(label('Train share', muted=True))
        vb.addWidget(self.train)
        vb.addWidget(button('New version', self.new_version))
        vb.addStretch(1)
        vs.add(vb)
        self.versions = QListWidget()
        self.versions.setMaximumHeight(100)
        vs.add(self.versions)
        body.addWidget(vs)
        body.addStretch(1)
        self.refresh()

    # ------------------------------------------------------------------ model
    def specs(self):
        return self.win.project.doc['dataset_specs']

    def current(self):
        i = self.list.currentRow()
        s = self.specs()
        return (i, s[i]) if 0 <= i < len(s) else (None, None)

    def set(self, key, value):
        i, s = self.current()
        if s is not None and s.get(key) != value:
            self.win.edit(Edit('set', ['dataset_specs', i, key], value, text=f'Dataset spec: {key}'))

    def set_checked(self, key, lst):
        vals = [lst.item(k).data(256) for k in range(lst.count())
                if lst.item(k).checkState() == Qt.CheckState.Checked]
        self.set(key, vals)

    def set_targets(self):
        t = {}
        for r in range(self.targets.rowCount()):
            c, n = self.targets.item(r, 0), self.targets.item(r, 1)
            if c and c.text().strip() and n:
                try:
                    t[c.text().strip()] = int(n.text())
                except ValueError:
                    continue
        self.set('targets', t)

    # ------------------------------------------------------------------ view
    def refresh(self):
        i = max(self.list.currentRow(), 0)
        self.list.blockSignals(True)
        self.list.clear()
        for s in self.specs():
            self.list.addItem(s['name'])
        self.list.setCurrentRow(min(i, self.list.count() - 1))
        self.list.blockSignals(False)
        self.show_spec()

    def show_spec(self):
        i, s = self.current()
        for w in (self.edit_sec, self.run_btn):
            w.setEnabled(s is not None)
        if s is None:
            return
        doc = self.win.project.doc
        widgets = (self.rig, self.drive_s, self.every, self.max_drives, self.seed, self.worlds, self.weathers,
                   self.targets)
        for w in widgets:
            w.blockSignals(True)
        self.rig.clear()
        for r in doc['rigs']:
            self.rig.addItem(r['name'], r['id'])
        self.rig.setCurrentIndex(max(self.rig.findData(s['rig']), 0))
        self.drive_s.setValue(float(s['drive_s']))
        self.every.setValue(int(s.get('record_every', 1)))
        self.max_drives.setValue(int(s.get('max_drives', 200)))
        self.seed.setValue(int(s.get('seed', 0)))
        for lst, items, chosen in ((self.worlds, [(w['id'], w['name']) for w in doc['worlds']], s['worlds']),
                                   (self.weathers, [(p, p) for p in W.PRESETS],
                                    [x if isinstance(x, str) else x.get('preset') for x in s['weathers']])):
            lst.clear()
            for ident, text in items:
                it = QListWidgetItem(text)
                it.setData(256, ident)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if ident in chosen else Qt.CheckState.Unchecked)
                lst.addItem(it)
        self.targets.setRowCount(0)
        for c, n in (s.get('targets') or {}).items():
            r = self.targets.rowCount()
            self.targets.insertRow(r)
            self.targets.setItem(r, 0, QTableWidgetItem(c))
            self.targets.setItem(r, 1, QTableWidgetItem(str(n)))
        for w in widgets:
            w.blockSignals(False)
        self.show_coverage(s)
        self.versions.clear()
        if self.win.project.path:
            for v in E.versions(self.win.workspace(), s['id']):
                n = {k: len(x) for k, x in v['splits'].items()}
                self.versions.addItem(f'v{v["version"]}  {v["created"]}  by {v["split_by"]}  {n}  '
                                      f'{v["sha256"][:12]}')

    def show_coverage(self, s):
        if not self.win.project.path:
            self.bars.set_rows([])
            return

        def compute():
            return D.coverage(s, E.coverage_rows(self.win.workspace(), s))

        def done(cov):
            classes = sorted(set(cov['targets']) | set(cov['have']))
            self.bars.set_rows([(c, cov['have'].get(c, 0), cov['targets'].get(c, 0)) for c in classes])
            edges = (s.get('bins') or D.DEFAULT_BINS)['distance_m']
            cols = [f'{edges[k]}–{edges[k + 1]} m' for k in range(len(edges) - 1)]
            dist = cov['bins'].get('distance_m', {})
            self.bins.setColumnCount(len(cols))
            self.bins.setHorizontalHeaderLabels(cols)
            self.bins.setRowCount(len(dist))
            self.bins.setVerticalHeaderLabels(sorted(dist))
            for r, c in enumerate(sorted(dist)):
                for k, n in enumerate(dist[c]):
                    self.bins.setItem(r, k, QTableWidgetItem(str(n)))
        tasks.run(self, compute, done)

    # ------------------------------------------------------------------ actions
    def add_spec(self):
        doc = self.win.project.doc
        s = D.new_spec(new_id('ds'), f'Dataset {len(doc["dataset_specs"]) + 1}', doc['rigs'][0]['id'],
                       [doc['worlds'][0]['id']])
        self.win.edit(Edit('insert', ['dataset_specs'], s, text='New dataset spec'))
        self.list.setCurrentRow(self.list.count() - 1)

    def remove_spec(self):
        i, s = self.current()
        if s is not None:
            self.win.edit(Edit('remove', ['dataset_specs'], index=i, text=f'Remove {s["name"]}'))

    def make_plan(self):
        i, s = self.current()
        if s is None:
            return
        if not self.win.project.path:
            QMessageBox.information(self, 'Plan', 'Save the project first: a dataset\'s drives live in its workspace.')
            return
        try:
            self.plan = E.make_plan(self.win.project, copy.deepcopy(s), self.win.service.throughput())
        except ValueError as e:
            QMessageBox.warning(self, 'Plan', str(e))
            return
        p = self.plan
        self.plan_kv.clear()
        self.plan_kv.set('Drives', f'{len(p["drives"])} to run ({p["done"]} finished already; '
                                   f'{p["per_drive_frames"]} recorded frames each)' +
                         (' — capped by "at most"' if p['capped'] else ''))
        self.plan_kv.set('Still needed', ', '.join(f'{c}: {n}' for c, n in p['needed'].items()) or 'nothing')
        if p['unreachable']:
            self.plan_kv.set('⚠ Unreachable', f'{", ".join(p["unreachable"])}: no yield measured or expected for '
                                              'these classes in these cells')
        self.plan_kv.set('Yield', p['yield_source'])
        if p['estimate']:
            e = p['estimate']
            self.plan_kv.set('Cost', f'about {budget.fmt_duration(e["seconds"])} ({e["label"]}; '
                                     f'{budget.fmt_duration(e["per_drive_s"])} per drive)', e['source'])
        set_enabled(self.run_btn, bool(p['drives']), 'The plan needs no drives')

    def run(self):
        i, s = self.current()
        if s is None or self.plan is None:
            return
        try:
            res = E.submit(self.win.service, self.win.project, s, self.plan)
        except (RuntimeError, ValueError) as e:
            QMessageBox.warning(self, 'Run', str(e))
            return
        self.plan_kv.set('Queued', f'{len(res["queued"])} drives ({res["cached"]} cached from earlier runs)')
        self.win.jobs_dock.raise_()

    def new_version(self):
        i, s = self.current()
        if s is None or not self.win.project.path:
            return
        tr = self.train.value()
        v = E.new_version(self.win.workspace(), s, self.split_by.currentText(),
                          {'train': tr, 'val': (1 - tr) / 2, 'test': (1 - tr) / 2}, int(s.get('seed', 0)))
        self.win.log.append_line(f'dataset {s["name"]} v{v["version"]}: '
                                 + ', '.join(f'{k} {len(x)}' for k, x in v['splits'].items()))
        self.show_spec()

    def on_run_finished(self, job):
        _, s = self.current()
        if s is not None and (job.spec.meta.get('config') or '').find(s['id']) >= 0:
            self.show_coverage(s)
