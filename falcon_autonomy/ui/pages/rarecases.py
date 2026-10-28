"""Rare cases: search a scenario's parameters for where it becomes critical — Bayesian optimisation, CMA-ES, adaptive
sampling or random (core.search) — and see the failing cases ranked (each replayable), how the search went, and a map
of the failure region over two parameters."""
from __future__ import annotations

import json
import os
import sys

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QMessageBox, QSpinBox, QSplitter,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import scenarios as SC
from ...core import search as S
from ...core.jobs import FINAL, JobSpec
from ..theme import SEQ, STATUS, THEME
from ..widgets import Form, KeyValues, Page, Section, button, label

METHODS = (('bayes', 'Bayesian optimisation'), ('cmaes', 'CMA-ES'), ('adaptive', 'Adaptive sampling'),
           ('random', 'Random (baseline)'))


class Curve(QWidget):
    """Best criticality so far (line) and failures found (steps) against evaluations."""

    def __init__(self):
        super().__init__()
        self.values = []
        self.threshold = 0.5
        self.setMinimumHeight(160)

    def set_values(self, values, threshold):
        self.values, self.threshold = list(values), threshold
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        n = len(self.values)
        if n == 0:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No evaluations yet')
            return
        x0, y0, w, h = 40, 10, self.width() - 60, self.height() - 34
        fails = 0
        best = 0.0
        bpts, fpts = [], []
        for i, v in enumerate(self.values):
            best = max(best, v)
            fails += v >= self.threshold
            bpts.append((i, best))
            fpts.append((i, fails))
        fmax = max(fails, 1)

        def X(i):
            return x0 + w * i / max(n - 1, 1)
        p.setPen(QColor(THEME.tok['grid2']))
        p.drawRect(QRectF(x0, y0, w, h))
        ty = y0 + h * (1 - self.threshold)
        p.setPen(QPen(QColor(STATUS['serious']), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x0, ty), QPointF(x0 + w, ty))
        p.setPen(QPen(QColor(SEQ[7]), 2))
        for (a, va), (b, vb) in zip(bpts, bpts[1:]):
            p.drawLine(QPointF(X(a), y0 + h * (1 - va)), QPointF(X(b), y0 + h * (1 - vb)))
        p.setPen(QPen(QColor(STATUS['critical']), 1.5))
        for (a, fa), (b, fb) in zip(fpts, fpts[1:]):
            p.drawLine(QPointF(X(a), y0 + h * (1 - fa / fmax)), QPointF(X(b), y0 + h * (1 - fa / fmax)))
            p.drawLine(QPointF(X(b), y0 + h * (1 - fa / fmax)), QPointF(X(b), y0 + h * (1 - fb / fmax)))
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QPointF(x0, self.height() - 6), f'evaluations (0–{n}) · blue: best criticality (0–1) · red: '
                                                   f'failures found (0–{fails}) · dashed: the failure threshold')


class Heatmap(QWidget):
    """The failure region: per cell of two parameters, the share of evaluations over the threshold (colour) and
    how many (number)."""

    def __init__(self):
        super().__init__()
        self.grid = None
        self.setMinimumHeight(220)

    def set_grid(self, grid):
        self.grid = grid
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        g = self.grid
        if not g:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Choose two parameters')
            return
        count, share = g['count'], g.get('failure_share')
        ny, nx = len(count), len(count[0]) if count else 0
        x0, y0, w, h = 60, 10, self.width() - 80, self.height() - 40
        for j in range(ny):
            for i in range(nx):
                n = count[j][i]
                r = QRectF(x0 + w * i / nx, y0 + h * (1 - (j + 1) / ny), w / nx, h / ny)
                if not n:
                    p.fillRect(r, QColor(THEME.tok['raised']))
                    continue
                f = (share[j][i] if share and share[j][i] is not None else 0.0)
                col = QColor(STATUS['critical'])
                col.setAlphaF(0.15 + 0.85 * f)
                p.fillRect(r, col)
                p.setPen(QColor(THEME.tok['ink']))
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(n))
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QPointF(x0, self.height() - 8), f'{g.get("x", "")} →   (colour: share of failures; number: '
                                                   'evaluations in the cell)')
        p.save()
        p.translate(14, y0 + h)
        p.rotate(-90)
        p.drawText(QPointF(0, 0), f'{g.get("y", "")} →')
        p.restore()


class RareCasesPage(Page):
    key, title, icon = 'rarecases', 'Rare cases', 'target'
    about = 'Search a scenario\'s parameters for where it becomes critical; replay what fails.'

    def __init__(self, win):
        super().__init__(win)
        self.job = None
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(label(self.title, title=True))
        lv.addWidget(label(self.about, muted=True, wrap=True))
        sec = Section('Search', 'Each evaluation runs the scenario in CARLA with the parameters the method asks for; '
                                'the objective is the run\'s criticality (1 for a collision). A search resumes where '
                                'it stopped, and starts from the sweeps of the same study.')
        f = Form()
        self.scenario = QComboBox()
        self.scenario.currentIndexChanged.connect(lambda _: self.load())
        self.method = QComboBox()
        for k, text in METHODS:
            self.method.addItem(text, k)
        self.method.currentIndexChanged.connect(lambda _: self.load())
        self.world = QComboBox()
        self.rig = QComboBox()
        self.budget = QSpinBox()
        self.budget.setRange(4, 5000)
        self.budget.setValue(60)
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(4)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 1.0)
        self.threshold.setValue(0.5)
        self.threshold.setSingleStep(0.05)
        self.threshold.valueChanged.connect(lambda _: self.load())
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        for text, w in (('Scenario', self.scenario), ('Method', self.method), ('World', self.world),
                        ('Rig', self.rig), ('Evaluations', self.budget), ('In batches of', self.batch),
                        ('Failure at criticality ≥', self.threshold), ('Seed', self.seed)):
            f.row(text, w)
        sec.add(f)
        b = QHBoxLayout()
        self.start_btn = button('Search', self.start, role='primary')
        b.addWidget(self.start_btn)
        b.addStretch(1)
        sec.add(b)
        self.state = KeyValues(('State',))
        sec.add(self.state)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(label('How the search went', h=True))
        self.curve = Curve()
        rv.addWidget(self.curve)
        rv.addWidget(label('Failing cases, most critical first', h=True))
        self.failures = QTreeWidget()
        self.failures.setHeaderLabels(['#', 'Criticality', 'Values', 'Run'])
        self.failures.setRootIsDecorated(False)
        self.failures.setMinimumHeight(160)
        rv.addWidget(self.failures)
        fb = QHBoxLayout()
        fb.addWidget(button('Replay the selected case', self.replay))
        fb.addWidget(button('Export it as OpenSCENARIO…', self.export_case))
        fb.addWidget(button('Browse its run', self.browse))
        fb.addStretch(1)
        rv.addLayout(fb)
        mh = QHBoxLayout()
        mh.addWidget(label('Failure region over', h=True))
        self.px = QComboBox()
        self.py = QComboBox()
        for c in (self.px, self.py):
            c.currentIndexChanged.connect(lambda _: self.draw_map())
            mh.addWidget(c)
        mh.addStretch(1)
        rv.addLayout(mh)
        self.heat = Heatmap()
        rv.addWidget(self.heat, 1)
        split.addWidget(right)
        split.setSizes([380, 900])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(2000)
        win.bridge.finished.connect(lambda j: self.load() if j is self.job else None)
        self.refresh()

    def scenarios(self):
        out = [(k, SC.builtin(k)) for k in SC.LIBRARY]
        out += [(s['id'], s) for s in self.win.project.doc['scenarios']]
        return out

    def refresh(self):
        cur = self.scenario.currentData()
        self.scenario.blockSignals(True)
        self.scenario.clear()
        for ident, s in self.scenarios():
            self.scenario.addItem(s['name'], ident)
        k = self.scenario.findData(cur)
        self.scenario.setCurrentIndex(k if k >= 0 else 0)
        self.scenario.blockSignals(False)
        doc = self.win.project.doc
        for combo, items in ((self.world, doc['worlds']), (self.rig, doc['rigs'])):
            c = combo.currentData()
            combo.clear()
            for it in items:
                combo.addItem(it['name'], it['id'])
            k = combo.findData(c)
            combo.setCurrentIndex(k if k >= 0 else 0)
        self.load()

    def study(self):
        return f'search-{self.scenario.currentData()}'

    def study_dir(self):
        return os.path.join(self.win.workspace(), 'studies', self.study())

    def start(self):
        w = self.win
        if self.job is not None and self.job.state not in FINAL:
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        argv = [sys.executable, '-m', 'falcon_autonomy', 'search', w.project.path, '--scenario',
                self.scenario.currentData(), '--name', self.study(), '--method', self.method.currentData(),
                '--budget', str(self.budget.value()), '--batch', str(self.batch.value()), '--threshold',
                str(self.threshold.value()), '--seed', str(self.seed.value()), '--world', self.world.currentData(),
                '--rig', self.rig.currentData()]
        if w.fake_carla:
            argv += ['--fake-carla', w.fake_carla]
        self.job = w.queue.submit(JobSpec('study', f'Search {self.scenario.currentText()} '
                                                   f'({self.method.currentText()}, {self.budget.value()})',
                                          argv, memory_gb=0.5))
        w.jobs_dock.raise_()

    def poll(self):
        if self.job is not None and self.job.state not in FINAL:
            self.state.set('State', self.job.summary())
            self.load()

    def rows(self):
        p = os.path.join(self.study_dir(), 'results.jsonl') if self.win.project.path else ''
        if not p or not os.path.isfile(p):
            return []
        with open(p, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def load(self):
        rows = self.rows()
        thr = self.threshold.value()
        self.curve.set_values([r.get('objective', 0.0) for r in rows], thr)
        self.failures.clear()
        hist = [{'params': r['values'], 'value': r.get('objective', 0.0), 'i': k, 'run': r['run']}
                for k, r in enumerate(rows)]
        self.history = hist
        for rank, h in enumerate(S.failures(hist, thr), 1):
            it = QTreeWidgetItem([str(rank), f'{h["value"]:.3f}',
                                  ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                            for k, v in sorted(h['params'].items())), os.path.basename(h['run'])])
            it.setData(0, 256, h)
            self.failures.addTopLevelItem(it)
        if not (self.job is not None and self.job.state not in FINAL):
            self.state.set('State', f'{len(rows)} evaluations, {self.failures.topLevelItemCount()} failures '
                                    f'(criticality ≥ {thr:g})' if rows else 'no evaluations yet')
        space = SC.search_space(dict(self.scenarios())[self.scenario.currentData()]) if self.scenario.count() else []
        names = [p['name'] for p in space]
        for c, default in ((self.px, 0), (self.py, 1)):
            cur = c.currentText()
            c.blockSignals(True)
            c.clear()
            c.addItems(names)
            if cur in names:
                c.setCurrentText(cur)
            elif len(names) > default:
                c.setCurrentIndex(default)
            c.blockSignals(False)
        self.space = space
        self.draw_map()

    def draw_map(self):
        hist = getattr(self, 'history', [])
        x, y = self.px.currentText(), self.py.currentText()
        if not hist or not x or not y or x == y:
            self.heat.set_grid(None)
            return
        g = S.failure_map(hist, self.space, x, y, bins=8, threshold=self.threshold.value())
        g = dict(g, x=x, y=y)
        self.heat.set_grid(g)

    def selected(self):
        it = self.failures.currentItem()
        return it.data(0, 256) if it is not None else None

    def replay(self):
        from ...core.drive import run_dir_name, scenario_config
        h = self.selected()
        if h is None:
            return
        w = self.win
        scn = dict(self.scenarios())[self.scenario.currentData()]
        cfg = scenario_config(w.project, scn, self.world.currentData(), self.rig.currentData(),
                              os.path.join(w.service.runs_dir(), run_dir_name('replay')), w.service.port,
                              w.service.tm_port, seed=self.seed.value(), values=h['params'],
                              workspace=w.workspace())
        w.service.submit_drive(cfg, title=f'Replay: {scn["name"]}')
        w.jobs_dock.raise_()

    def export_case(self):
        from ...core import openscenario as X
        h = self.selected()
        if h is None:
            return
        scn = dict(self.scenarios())[self.scenario.currentData()]
        path, _ = QFileDialog.getSaveFileName(self, 'Export the case', os.path.join(os.path.expanduser('~'),
                                                                                   'rare-case.xosc'),
                                              'OpenSCENARIO (*.xosc)')
        if path:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(X.export_xosc(dict(SC.concrete(scn, h['params']), seed=self.seed.value())))

    def browse(self):
        h = self.selected()
        if h is not None:
            self.win.go('data').show_run(h['run'])

    def shutdown(self):
        self.timer.stop()
