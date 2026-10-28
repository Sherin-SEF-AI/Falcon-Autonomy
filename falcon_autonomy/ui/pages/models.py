"""Models: register a perception model (ONNX, TorchScript or a Python plug-in), download its pinned weights when asked,
score it on recorded drives (core.harness: AP with drive-level CIs, a condition heat map, the worst frames), compare
two evaluations (paired bootstrap: regressions), and measure the sim-to-real gap of the drives' images against a real
dataset (core.realgap)."""
from __future__ import annotations

import json
import os
import sys
import time

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QSpinBox,
                             QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import harness as H
from ...core import models as MD
from ...core import realgap as G
from ...core.jobs import FINAL, JobSpec
from ...core.paths import tilde
from ...core.project import Edit, new_id
from ...core.provenance import read_json
from ..theme import SEQ, STATUS, THEME
from ..widgets import Form, KeyValues, Page, Section, button, label, scroll

SYNTHETIC = {'name': 'Synthetic detector (testing only)', 'kind': 'detector', 'runtime': 'python',
             'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector', 'licence': 'GPL-3.0-or-later',
             'about': 'Ground truth with planted weaknesses (small, occluded, fogged objects) — not a model.'}


def _fmt(v, nd=3):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


class RegisterDialog(QDialog):
    """A model registration (core.models' fields); validated before it is accepted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Register a model')
        v = QVBoxLayout(self)
        f = Form()
        self.name = QLineEdit()
        self.kind = QComboBox()
        self.kind.addItems(MD.KINDS)
        self.runtime = QComboBox()
        self.runtime.addItems(MD.RUNTIMES)
        self.path = QLineEdit()
        self.path.setPlaceholderText('model file (relative: the workspace\'s models/ folder)')
        pick = button('…', self.pick)
        ph = QHBoxLayout()
        ph.addWidget(self.path, 1)
        ph.addWidget(pick)
        pw = QWidget()
        pw.setLayout(ph)
        self.entry = QLineEdit()
        self.entry.setPlaceholderText('module:Class (a Python plug-in)')
        self.width_ = QSpinBox()
        self.height_ = QSpinBox()
        for s, d in ((self.width_, 640), (self.height_, 640)):
            s.setRange(16, 4096)
            s.setValue(d)
        self.fmt = QComboBox()
        self.fmt.addItems(MD.FORMATS)
        self.classes = QLineEdit()
        self.classes.setPlaceholderText('the model\'s class names in index order, comma-separated')
        self.cmap = QLineEdit()
        self.cmap.setPlaceholderText('{"person": "pedestrian", "car": "car"}')
        self.licence = QLineEdit()
        self.url = QLineEdit()
        self.sha = QLineEdit()
        for text, w in (('Name', self.name), ('Kind', self.kind), ('Runtime', self.runtime), ('File', pw),
                        ('Plug-in entry', self.entry), ('Input width', self.width_), ('Input height', self.height_),
                        ('Output layout', self.fmt), ('Classes', self.classes), ('Class map → Falcon', self.cmap),
                        ('Licence (SPDX)', self.licence), ('Download URL', self.url), ('SHA-256', self.sha)):
            f.row(text, w)
        v.addLayout(f)
        v.addWidget(label('Weights are downloaded only when you ask, from the pinned URL, and refused if their SHA-256 '
                          'differs. An AGPL-licensed model cannot be registered.', muted=True, wrap=True))
        self.err = label('', wrap=True)
        self.err.setStyleSheet(f'color: {STATUS["critical"]}')
        v.addWidget(self.err)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.check)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.model = None

    def pick(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Model file', '', 'Models (*.onnx *.pt *.ts *.torchscript);;All (*)')
        if p:
            self.path.setText(p)

    def check(self):
        try:
            cmap = json.loads(self.cmap.text()) if self.cmap.text().strip() else {}
        except ValueError as e:
            self.err.setText(f'The class map is not JSON: {e}')
            return
        m = {'name': self.name.text().strip() or 'Model', 'kind': self.kind.currentText(),
             'runtime': self.runtime.currentText(), 'licence': self.licence.text().strip(),
             'classes': [c.strip() for c in self.classes.text().split(',') if c.strip()], 'class_map': cmap}
        if m['runtime'] == 'python':
            m['entry'] = self.entry.text().strip()
        else:
            m['path'] = self.path.text().strip()
            m['input'] = {'width': self.width_.value(), 'height': self.height_.value(), 'layout': 'NCHW',
                          'scale': 1 / 255, 'rgb': True}
            m['output'] = {'format': self.fmt.currentText(), 'score_threshold': 0.3, 'nms_iou': 0.5}
        if self.url.text().strip() or self.sha.text().strip():
            m['source'] = {'url': self.url.text().strip(), 'sha256': self.sha.text().strip().lower()}
        problems = MD.validate(m)
        if problems:
            self.err.setText('; '.join(problems))
            return
        self.model = m
        self.accept()


class GridView(QWidget):
    """AP per cell of two conditions (colour; grey: too few objects to score) with each cell's frames and objects."""

    def __init__(self):
        super().__init__()
        self.grid = None
        self.setMinimumHeight(260)

    def set_grid(self, g):
        self.grid = g
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        g = self.grid
        if not g or g.get('error') or not g.get('cells'):
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       (g or {}).get('error') or 'Evaluate with two conditions to see AP over them')
            return
        cells = g['cells']
        ny, nx = len(cells), len(cells[0])
        x0, y0, w, h = 70, 10, self.width() - 90, self.height() - 46
        for j in range(ny):
            for i in range(nx):
                c = cells[j][i]
                r = QRectF(x0 + w * i / nx, y0 + h * (1 - (j + 1) / ny), w / nx - 1, h / ny - 1)
                if c['AP'] is None:
                    fill = QColor(THEME.tok['raised'])
                    text = f'{c["n_frames"]} frames\n{c.get("note") or ""}'
                else:
                    fill = QColor(SEQ[min(int(c['AP'] * (len(SEQ) - 1) + 0.5), len(SEQ) - 1)])
                    text = f'AP {c["AP"]:.2f}\n{c["n_frames"]} frames, {c["n_gt"]} objects'
                p.fillRect(r, fill)
                p.setPen(QColor('#101010') if fill.lightnessF() > 0.6 else QColor('#f4f4f4'))
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
                if j == 0:
                    p.setPen(QColor(THEME.tok['muted']))
                    p.drawText(QRectF(r.x(), y0 + h + 2, r.width(), 16), Qt.AlignmentFlag.AlignCenter,
                               f'{c["x_value"]:g}' if c.get('x_value') is not None else
                               f'{c["x"][0]:g}–{c["x"][1]:g}')
            p.setPen(QColor(THEME.tok['muted']))
            c = cells[j][0]
            p.drawText(QRectF(0, y0 + h * (1 - (j + 1) / ny), x0 - 4, h / ny), Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter, f'{c["y_value"]:g}' if c.get('y_value') is not None else
                       f'{c["y"][0]:g}–{c["y"][1]:g}')
        p.drawText(QPointF(x0, self.height() - 6), f'{g["x"]} →   (colour: AP, light to dark = 0 to 1; grey: fewer '
                                                   f'than {g.get("min_gt")} objects)   ↑ {g["y"]}')


class FrameView(QLabel):
    """A frame with its missed objects (red) and confident false positives (orange)."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.row = None

    def show_row(self, row):
        self.row = row
        self.redraw()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.redraw()

    def redraw(self):
        r = self.row
        if not r or not os.path.isfile(r.get('image') or ''):
            self.setPixmap(QPixmap())
            self.setText('Select a frame' if not r else 'The frame\'s image is not on this machine')
            return
        img = QImage(r['image'])
        p = QPainter(img)
        for items, col in ((r['missed'], STATUS['critical']), (r['false_positives'], STATUS['serious'])):
            p.setPen(QPen(QColor(col), max(2, img.width() // 400)))
            for o in items:
                x, y, w, h = o['bbox']
                p.drawRect(QRectF(x, y, w, h))
        p.end()
        self.setPixmap(QPixmap.fromImage(img).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))


class ModelsPage(Page):
    key, title, icon = 'models', 'Models', 'checklist'
    about = ('Register a perception model, score it on recorded drives with confidence intervals, see where it fails, '
             'compare versions, and measure how far the simulated images are from real ones.')

    def __init__(self, win):
        super().__init__(win)
        self.jobs = {}
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(label(self.title, title=True))
        lv.addWidget(label(self.about, muted=True, wrap=True))
        # ---- registered models
        sec = Section('Registered models', 'No model is bundled. Weights download only when you ask, from a pinned '
                                           'URL checked by SHA-256, with the licence shown first.')
        self.models = QTreeWidget()
        self.models.setHeaderLabels(['Name', 'Kind', 'Runtime', 'Licence', 'Weights'])
        self.models.setRootIsDecorated(False)
        self.models.setMinimumHeight(110)
        sec.add(self.models)
        b = QHBoxLayout()
        b.addWidget(button('Register…', self.register))
        b.addWidget(button('Add the synthetic detector', self.add_synthetic,
                           tip='A test model built from the ground truth with planted weaknesses (not a detector)'))
        b.addWidget(button('Download…', self.download))
        b.addWidget(button('Remove', self.remove))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        # ---- evaluate
        sec = Section('Evaluate', 'Every n-th frame of the checked drives goes through the model; the numbers carry a '
                                  'bootstrap 95 % CI over whole drives.')
        f = Form()
        self.model = QComboBox()
        self.every = QSpinBox()
        self.every.setRange(1, 100)
        self.every.setValue(2)
        self.boot = QSpinBox()
        self.boot.setRange(50, 10000)
        self.boot.setValue(1000)
        self.gx, self.gy = QComboBox(), QComboBox()
        for c in (self.gx, self.gy):
            c.addItem('—', '')
            for k in H.CONDITIONS:
                c.addItem(k, k)
        self.gx.setCurrentIndex(1)
        self.gy.setCurrentIndex(3)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        for text, w in (('Model', self.model), ('Every n-th frame', self.every), ('Bootstrap resamples', self.boot),
                        ('Heat map across', self.gx), ('and', self.gy), ('Seed', self.seed)):
            f.row(text, w)
        sec.add(f)
        sec.add(label('Drives', h=True))
        self.runs = QListWidget()
        self.runs.setMinimumHeight(120)
        sec.add(self.runs)
        b = QHBoxLayout()
        b.addWidget(button('Evaluate', self.evaluate, role='primary'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        # ---- compare
        sec = Section('Compare', 'B against A on the frames both scored: a paired bootstrap of each number\'s change. '
                                 'A regression is a drop whose whole CI is below zero.')
        f = Form()
        self.cmp_a, self.cmp_b = QComboBox(), QComboBox()
        f.row('A (before)', self.cmp_a)
        f.row('B (after)', self.cmp_b)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Compare', self.compare))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        # ---- sim-to-real gap
        sec = Section('Sim-to-real gap', 'The checked drives\' images against a real dataset, read in place: colour, '
                                         'brightness, noise, sharpness, objects; Fréchet and kernel distances in a '
                                         'fixed feature space, each next to its real-vs-real floor.')
        f = Form()
        self.real_kind = QComboBox()
        for k in G.KINDS:
            self.real_kind.addItem(k, k)
        self.real_path = QLineEdit()
        self.real_labels = QLineEdit()
        self.real_labels.setPlaceholderText('labels JSON (bdd100k, coco)')
        rp = QHBoxLayout()
        rp.addWidget(self.real_path, 1)
        rp.addWidget(button('…', lambda: self._pick_dir(self.real_path)))
        rw = QWidget()
        rw.setLayout(rp)
        self.gap_images = QSpinBox()
        self.gap_images.setRange(20, 5000)
        self.gap_images.setValue(300)
        self.features = QCheckBox('Fréchet / kernel distances (needs the feature extractor)')
        self.match = QCheckBox('Also measure after appearance matching')
        for text, w in (('Real set', self.real_kind), ('Folder', rw), ('Labels', self.real_labels),
                        ('Images per side', self.gap_images)):
            f.row(text, w)
        sec.add(f)
        sec.add(self.features)
        sec.add(self.match)
        b = QHBoxLayout()
        b.addWidget(button('Measure', self.gap, role='primary'))
        b.addWidget(button('Download the feature extractor…', self.fetch_features))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        self.state = KeyValues(('Job',))
        lv.addWidget(self.state)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        # ---- results
        self.tabs = QTabWidget()
        self.tabs.addTab(self._scores_tab(), 'Scores')
        self.tabs.addTab(self._grid_tab(), 'Conditions')
        self.tabs.addTab(self._gallery_tab(), 'Failures')
        self.tabs.addTab(self._compare_tab(), 'Comparison')
        self.tabs.addTab(self._gap_tab(), 'Sim-to-real')
        split.addWidget(self.tabs)
        split.setSizes([440, 860])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1500)
        win.bridge.finished.connect(self.finished)
        self.refresh()

    # ------------------------------------------------------------------ result tabs
    def _scores_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.ev = QComboBox()
        self.ev.currentIndexChanged.connect(lambda _: self.show_eval())
        v.addWidget(self.ev)
        self.ev_info = KeyValues(('Model', 'Frames', 'Bootstrap', 'Ran on', 'Latency (this machine)', 'Note'))
        v.addWidget(self.ev_info)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(['Metric', 'Value', '95 % CI', 'n'])
        self.table.setRootIsDecorated(False)
        v.addWidget(self.table, 1)
        return w

    def _grid_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.gridview = GridView()
        v.addWidget(self.gridview, 1)
        return w

    def _gallery_tab(self):
        w = QSplitter(Qt.Orientation.Vertical)
        self.gal = QTreeWidget()
        self.gal.setHeaderLabels(['Frame', 'Errors', 'Missed', 'False positives', 'Conditions'])
        self.gal.setRootIsDecorated(False)
        self.gal.currentItemChanged.connect(lambda it, _: self.frame.show_row(it.data(0, 256) if it else None))
        w.addWidget(self.gal)
        box = QWidget()
        bv = QVBoxLayout(box)
        self.frame = FrameView()
        bv.addWidget(self.frame, 1)
        bv.addWidget(label('Red: missed objects · orange: false positives scoring ≥ 0.5 (IoU 0.5, per class)',
                           muted=True))
        w.addWidget(box)
        return w

    def _compare_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.cmp = QComboBox()
        self.cmp.currentIndexChanged.connect(lambda _: self.show_compare())
        v.addWidget(self.cmp)
        self.cmp_info = KeyValues(('A', 'B', 'Paired over', 'Regressions', 'Improvements'))
        v.addWidget(self.cmp_info)
        self.cmp_table = QTreeWidget()
        self.cmp_table.setHeaderLabels(['Metric', 'A', 'B', 'B − A', '95 % CI of B − A'])
        self.cmp_table.setRootIsDecorated(False)
        v.addWidget(self.cmp_table, 1)
        return w

    def _gap_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.gapsel = QComboBox()
        self.gapsel.currentIndexChanged.connect(lambda _: self.show_gap())
        v.addWidget(self.gapsel)
        self.gap_info = KeyValues(('Simulated', 'Real', 'Fréchet distance', 'Kernel distance', 'After matching'))
        v.addWidget(self.gap_info)
        self.gap_table = QTreeWidget()
        self.gap_table.setHeaderLabels(['Statistic', 'Simulated (median)', 'Real (median)', 'Shift', 'After matching'])
        self.gap_table.setRootIsDecorated(False)
        v.addWidget(self.gap_table, 1)
        v.addWidget(label('Shift: the Wasserstein distance between the two sets\' distributions over the real set\'s '
                          'interquartile range — a heuristic scale for ranking the gaps, not a test.', muted=True,
                          wrap=True))
        return w

    # ------------------------------------------------------------------ state
    def ws(self, *parts):
        return os.path.join(self.win.workspace(), *parts)

    def new_dir(self, sub, name):
        """A new output folder under the workspace (a suffix when the name is taken)."""
        base = self.ws(sub, name)
        p, k = base, 2
        while os.path.exists(p):
            p, k = f'{base}-{k}', k + 1
        os.makedirs(p)                    # taken now, so that a second submission picks another name
        return p

    def refresh(self):
        doc = self.win.project.doc
        ws = self.win.workspace()
        cur = self.model.currentData()
        self.models.clear()
        self.model.clear()
        for m in doc['models']:
            if m.get('runtime') == 'python':
                weights = 'plug-in'
            elif m.get('path') and os.path.isfile(MD.resolve_path(m, ws)):
                weights = 'present'
            else:
                weights = 'download' if (m.get('source') or {}).get('url') else 'missing'
            it = QTreeWidgetItem([m['name'], m.get('kind', ''), m.get('runtime', ''), m.get('licence') or '—',
                                  weights])
            it.setData(0, 256, m['id'])
            self.models.addTopLevelItem(it)
            self.model.addItem(m['name'], m['id'])
        for k in range(5):
            self.models.resizeColumnToContents(k)
        k = self.model.findData(cur)
        self.model.setCurrentIndex(k if k >= 0 else 0)
        checked = {self.runs.item(i).data(256) for i in range(self.runs.count())
                   if self.runs.item(i).checkState() == Qt.CheckState.Checked}
        self.runs.clear()
        cat = self.win.service.catalog
        for r in (cat.list('run', 500) if cat is not None else []):
            if r['summary'].get('worker') not in (None, 'drive') or r['status'] != 'ok':
                continue
            it = QListWidgetItem(r['name'])
            it.setData(256, r['path'])
            it.setToolTip(tilde(r['path']))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if r['path'] in checked else Qt.CheckState.Unchecked)
            self.runs.addItem(it)
        self.load_results()

    def _dirs(self, sub, name):
        root = self.ws(sub)
        if not os.path.isdir(root):
            return []
        return sorted((d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, name))), reverse=True)

    def load_results(self):
        for combo, sub, name in ((self.ev, 'evaluations', 'evaluation.json'), (self.cmp, 'comparisons',
                                                                              'comparison.json'),
                                 (self.gapsel, 'gaps', 'gap.json')):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for d in self._dirs(sub, name):
                combo.addItem(d, os.path.join(self.ws(sub), d))
            k = combo.findData(cur)
            combo.setCurrentIndex(k if k >= 0 else 0)
            combo.blockSignals(False)
        for combo in (self.cmp_a, self.cmp_b):
            cur = combo.currentData()
            combo.clear()
            for d in self._dirs('evaluations', 'evaluation.json'):
                combo.addItem(d, os.path.join(self.ws('evaluations'), d))
            k = combo.findData(cur)
            combo.setCurrentIndex(k if k >= 0 else (1 if combo is self.cmp_a and combo.count() > 1 else 0))
        self.show_eval()
        self.show_compare()
        self.show_gap()

    def checked_runs(self):
        return [self.runs.item(i).data(256) for i in range(self.runs.count())
                if self.runs.item(i).checkState() == Qt.CheckState.Checked]

    def selected_model(self):
        it = self.models.currentItem()
        return self.win.project.item('models', it.data(0, 256)) if it else None

    # ------------------------------------------------------------------ actions
    def _add(self, m):
        m = dict(m, id=new_id('model'))
        self.win.edit(Edit('insert', ['models'], m, text=f'Register {m["name"]}'))
        self.refresh()

    def register(self):
        d = RegisterDialog(self)
        if d.exec() and d.model:
            self._add(d.model)

    def add_synthetic(self):
        self._add(SYNTHETIC)

    def remove(self):
        it = self.models.currentItem()
        if not it:
            return
        ids = [m['id'] for m in self.win.project.doc['models']]
        self.win.edit(Edit('remove', ['models'], index=ids.index(it.data(0, 256)), text='Remove a model'))
        self.refresh()

    def _ready(self) -> bool:
        w = self.win
        if not w.project.path or w.project.dirty:
            return w.save()
        return True

    def _submit(self, kind, title, argv, needs_gpu=False, memory_gb=1.0):
        job = self.win.queue.submit(JobSpec(kind, title, [sys.executable, '-m', 'falcon_autonomy'] + argv,
                                            needs_gpu=needs_gpu, memory_gb=memory_gb))
        self.jobs[job.id] = job
        self.win.jobs_dock.raise_()
        return job

    def download(self):
        m = self.selected_model()
        if not m:
            return
        src = m.get('source') or {}
        if m.get('runtime') == 'python' or not src.get('url'):
            QMessageBox.information(self, 'Download', 'This model has nothing to download (a plug-in, or no pinned '
                                                      'URL).')
            return
        if QMessageBox.question(self, 'Download the model?', f'{m["name"]}\n\nFrom: {src["url"]}\nLicence: '
                                f'{m.get("licence") or "not stated — check before use"}\nSHA-256: {src["sha256"]}\n\n'
                                'The file is refused if its SHA-256 differs.') != QMessageBox.StandardButton.Yes:
            return
        if self._ready():
            self._submit('fetch', f'Download {m["name"]}', ['model', 'fetch', self.win.project.path, m['id']],
                         memory_gb=0.3)

    def fetch_features(self):
        f = G.FEATURES[G.DEFAULT_FEATURES]
        if QMessageBox.question(self, 'Download the feature extractor?',
                                f'{G.DEFAULT_FEATURES} ({f["mb"]} MB)\n\nFrom: {f["url"]}\nLicence: {f["licence"]}\n'
                                f'SHA-256: {f["sha256"]}') != QMessageBox.StandardButton.Yes:
            return
        self._submit('fetch', 'Download the feature extractor', ['gap', '--fetch', '--workspace',
                                                                  self.win.workspace()], memory_gb=0.3)

    def evaluate(self):
        mid = self.model.currentData()
        runs = self.checked_runs()
        if not mid or not runs:
            QMessageBox.information(self, 'Evaluate', 'Choose a model and check at least one drive.')
            return
        if not self._ready():
            return
        m = self.win.project.item('models', mid)
        out = self.new_dir('evaluations', f'{time.strftime("%Y%m%d-%H%M%S")}-{mid}')
        argv = ['evaluate', self.win.project.path, '--model', mid, '--out', out, '--every', str(self.every.value()),
                '--boot', str(self.boot.value()), '--seed', str(self.seed.value())]
        for r in runs:
            argv += ['--run', r]
        if self.gx.currentData() and self.gy.currentData() and self.gx.currentData() != self.gy.currentData():
            argv += ['--grid', f'{self.gx.currentData()},{self.gy.currentData()}']
        self._submit('evaluate', f'Evaluate {m["name"]} on {len(runs)} drive(s)', argv,
                     needs_gpu=m.get('runtime') != 'python', memory_gb=2.0)

    def compare(self):
        a, b = self.cmp_a.currentData(), self.cmp_b.currentData()
        if not a or not b or a == b:
            QMessageBox.information(self, 'Compare', 'Choose two different evaluations.')
            return
        out = self.new_dir('comparisons', f'{os.path.basename(a)}-vs-{os.path.basename(b)}')
        self._submit('compare', 'Compare two evaluations', ['evaluate', '--compare', a, b, '--out', out,
                                                            '--boot', str(self.boot.value())], memory_gb=1.0)

    def gap(self):
        runs = self.checked_runs()
        path = self.real_path.text().strip()
        if not runs or not path:
            QMessageBox.information(self, 'Sim-to-real gap', 'Check at least one drive and choose the real set\'s '
                                                             'folder.')
            return
        out = self.new_dir('gaps', time.strftime('%Y%m%d-%H%M%S'))
        argv = ['gap', '--real', f'{self.real_kind.currentData()}:{path}', '--out', out, '--images',
                str(self.gap_images.value()), '--workspace', self.win.workspace()]
        for r in runs:
            argv += ['--sim', r]
        if self.real_labels.text().strip():
            argv += ['--labels', self.real_labels.text().strip()]
        if self.features.isChecked():
            argv.append('--features')
        if self.match.isChecked():
            argv.append('--match')
        self._submit('gap', f'Sim-to-real gap: {len(runs)} drive(s) vs {self.real_kind.currentData()}', argv,
                     needs_gpu=self.features.isChecked(), memory_gb=3.0)

    def _pick_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, 'Real dataset folder', edit.text())
        if d:
            edit.setText(d)

    def poll(self):
        live = [j for j in self.jobs.values() if j.state not in FINAL]
        if live:
            self.state.set('Job', ' · '.join(j.summary() for j in live))

    def finished(self, job):
        if job.id not in self.jobs:
            return
        self.jobs.pop(job.id)
        self.state.set('Job', job.summary())
        self.refresh()

    # ------------------------------------------------------------------ showing results
    def show_eval(self):
        self.table.clear()
        self.gal.clear()
        self.frame.show_row(None)
        d = self.ev.currentData()
        if not d:
            self.ev_info.clear()
            self.gridview.set_grid(None)
            return
        try:
            e = read_json(os.path.join(d, 'evaluation.json'))
        except (OSError, ValueError):
            return
        sc = e['scores']
        lat = e.get('latency_ms') or {}
        self.ev_info.set('Model', f'{e["model"].get("name")} ({e["model"].get("runtime")})')
        self.ev_info.set('Frames', f'{e["n_frames"]} from {len(e["runs"])} drive(s), every {e["every"]}')
        self.ev_info.set('Bootstrap', f'{sc["n_boot"]} resamples of whole {sc["unit"]}s ({sc["n_units"]})')
        self.ev_info.set('Ran on', ', '.join(e.get('providers') or ()))
        self.ev_info.set('Latency (this machine)', f'median {lat.get("median", 0):.3g} ms, p90 '
                                                   f'{lat.get("p90", 0):.3g} ms per frame')
        notes = (['SYNTHETIC test model: these numbers say nothing about a real detector']
                 if e.get('synthetic_model') else []) + ([sc['note']] if sc.get('note') else [])
        self.ev_info.set('Note', ' · '.join(notes))
        for r in sc['table']:
            ci = '—' if r['ci_low'] is None else f'{_fmt(r["ci_low"])} – {_fmt(r["ci_high"])}'
            self.table.addTopLevelItem(QTreeWidgetItem([r['metric'], _fmt(r['value']), ci, _fmt(r.get('n'))]))
        self.table.resizeColumnToContents(0)
        self.gridview.set_grid(e.get('grid'))
        for row in e.get('gallery') or ():
            cond = ', '.join(f'{k} {v}' for k, v in row['conditions'].items() if k in ('weather', 'fog_density',
                                                                                        'sun_altitude_angle'))
            it = QTreeWidgetItem([row['image_id'], str(row['n_errors']), str(len(row['missed'])),
                                  str(len(row['false_positives'])), cond])
            it.setData(0, 256, row)
            self.gal.addTopLevelItem(it)

    def show_compare(self):
        self.cmp_table.clear()
        d = self.cmp.currentData()
        if not d:
            self.cmp_info.clear()
            return
        try:
            c = read_json(os.path.join(d, 'comparison.json'))
        except (OSError, ValueError):
            return
        self.cmp_info.set('A', c['a']['model'].get('name') + ' — ' + os.path.basename(c['a']['dir']))
        self.cmp_info.set('B', c['b']['model'].get('name') + ' — ' + os.path.basename(c['b']['dir']))
        self.cmp_info.set('Paired over', f'{c["n_frames"]} frames, {c["n_units"]} {c["unit"]}s, {c["n_boot"]} '
                                         'resamples')
        self.cmp_info.set('Regressions', ', '.join(c['regressions']) or 'none')
        self.cmp_info.set('Improvements', ', '.join(c['improvements']) or 'none')
        reg, imp = set(c['regressions']), set(c['improvements'])
        for r in c['rows']:
            ci = '—' if r['ci_low'] is None else f'{_fmt(r["ci_low"])} – {_fmt(r["ci_high"])}'
            it = QTreeWidgetItem([r['metric'], _fmt(r['a']), _fmt(r['b']), _fmt(r['diff']), ci])
            if r['metric'] in reg or r['metric'] in imp:
                col = QColor(STATUS['critical'] if r['metric'] in reg else STATUS['good'])
                for k in range(5):
                    it.setForeground(k, col)
            self.cmp_table.addTopLevelItem(it)
        self.cmp_table.resizeColumnToContents(0)

    def show_gap(self):
        self.gap_table.clear()
        d = self.gapsel.currentData()
        if not d:
            self.gap_info.clear()
            return
        try:
            g = read_json(os.path.join(d, 'gap.json'))
        except (OSError, ValueError):
            return
        self.gap_info.set('Simulated', f'{g["sim"]["n_images"]} images from {len(g["sim"]["sources"])} drive(s)')
        self.gap_info.set('Real', f'{g["real"]["kind"]}: {g["real"]["n_images"]} of {g["real"]["n_total"]} images — '
                                  f'{tilde(g["real"]["path"])}')
        f = g.get('features')
        if f:
            self.gap_info.set('Fréchet distance', f'{f["fd"]:.1f} (real-vs-real floor {f["fd_floor"]:.1f}; '
                                                  f'{f["n_sim"]} vs {f["n_real"]} images, {f["extractor"]})')
            self.gap_info.set('Kernel distance', f'{f["kd"]:.4f} ± {f["kd_std"]:.4f} (floor {f["kd_floor"]:.4f})')
        else:
            self.gap_info.set('Fréchet distance', 'not measured')
            self.gap_info.set('Kernel distance', 'not measured')
        mt = g.get('matched')
        after = {r['stat']: r for r in (mt or {}).get('gap') or ()}
        if mt and mt.get('features'):
            self.gap_info.set('After matching', f'Fréchet {mt["features"]["fd"]:.1f}, kernel '
                                                f'{mt["features"]["kd"]:.4f}')
        else:
            self.gap_info.set('After matching', 'measured per statistic below' if mt else 'not measured')
        rows = g['gap']['appearance'] + ((g['gap'].get('objects') or {}).get('rows') or [])
        for r in rows:
            a = after.get(r['stat'])
            it = QTreeWidgetItem([r['stat'], _fmt(r['sim_median']), _fmt(r['real_median']), _fmt(r['shift'], 2),
                                  f'{_fmt(a["sim_median"])} (shift {_fmt(a["shift"], 2)})' if a else ''])
            it.setToolTip(0, r.get('note') or '')
            self.gap_table.addTopLevelItem(it)
        self.gap_table.resizeColumnToContents(0)
