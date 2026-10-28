"""Data: record a drive (its cost shown before it starts), and browse recorded runs frame by frame with their label
layers. The dataset engine, exports and coverage dashboards join this page in milestone 3."""
from __future__ import annotations

import json
import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QSlider,
                             QSpinBox, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import budget, labels
from ...core.drive import drive_config, run_dir_name
from ...core.paths import tilde
from ...core.project import ProjectError
from .. import icons, tasks
from ..widgets import Form, KeyValues, Page, Section, button, label, scroll


class ImageView(QLabel):
    """An image scaled to fit, aspect kept."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self._pm = None

    def set_rgb(self, rgb: np.ndarray | None):
        if rgb is None:
            self._pm = None
            self.clear()
            return
        h, w, _ = rgb.shape
        img = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pm = QPixmap.fromImage(img)
        self._fit()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()

    def _fit(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))


class RecordTab(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        v = QVBoxLayout(self)
        sec = Section('Record a drive', 'Synchronous mode at a fixed step; the ego and the traffic on CARLA\'s traffic '
                                        'manager; every camera\'s label cameras at its exact pose and tick. Seeded: '
                                        'the same seed on the same server reproduces the drive (falcon reproduce).')
        f = Form()
        self.world = QComboBox()
        self.rig = QComboBox()
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0.5, 3600)
        self.seconds.setValue(10)
        self.seconds.setSuffix(' s')
        self.fps = QDoubleSpinBox()
        self.fps.setRange(1, 100)
        self.fps.setValue(20)
        self.fps.setSuffix(' Hz')
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        f.row('World', self.world)
        f.row('Rig', self.rig)
        f.row('Duration', self.seconds, 'simulated time')
        f.row('Step', self.fps, 'fixed_delta_seconds = 1 / rate')
        f.row('Seed', self.seed, 'spawn choices, traffic manager, pedestrians')
        sec.add(f)
        self.cost = KeyValues(('Rendering', 'Budget', 'Estimate'))
        sec.add(self.cost)
        b = QHBoxLayout()
        self.go_btn = button('Record', self.record, role='primary')
        b.addWidget(self.go_btn)
        b.addStretch(1)
        sec.add(b)
        v.addWidget(sec)
        v.addStretch(1)
        for w in (self.world, self.rig):
            w.currentIndexChanged.connect(self.update_cost)
        for w in (self.seconds, self.fps):
            w.valueChanged.connect(self.update_cost)

    def refresh(self):
        doc = self.win.project.doc
        for combo, items in ((self.world, doc['worlds']), (self.rig, doc['rigs'])):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for it in items:
                combo.addItem(it['name'], it['id'])
            k = combo.findData(cur)
            combo.setCurrentIndex(k if k >= 0 else 0)
            combo.blockSignals(False)
        self.update_cost()

    def update_cost(self):
        rig = self.win.project.item('rigs', self.rig.currentData() or '')
        if rig is None:
            return
        mpx = budget.rig_mpx(rig)
        lim = float(self.win.settings.get('render_budget_mpx'))
        ok, why = budget.check_budget(rig, lim)
        steps = int(round(self.seconds.value() * self.fps.value()))
        t = self.win.service.throughput()
        est = t.estimate(steps, mpx)
        self.cost.set('Rendering', f'{mpx:.2f} Mpx per step ({len(rig["sensors"])} sensors, images and label '
                                   'cameras)')
        self.cost.set('Budget', ('' if ok else '⚠ ') + why)
        self.cost.set('Estimate', f'{steps} steps: about {budget.fmt_duration(est["seconds"])} '
                                  f'({est["per_step_s"] * 1000:.0f} ms per step, {est["label"]})', t.source)
        self.go_btn.setEnabled(ok)

    def record(self):
        w = self.win
        try:
            out = os.path.join(w.service.runs_dir(), run_dir_name(f'drive-{self.world.currentText()}'))
            cfg = drive_config(w.project, self.world.currentData(), self.rig.currentData(), out, w.service.port,
                               w.service.tm_port, self.seconds.value(), self.fps.value(), self.seed.value(),
                               render_budget_mpx=float(w.settings.get('render_budget_mpx')))
            w.service.submit_drive(cfg)
        except (ProjectError, RuntimeError) as e:
            QMessageBox.warning(self, 'Record a drive', str(e))
            return
        w.jobs_dock.raise_()


class BrowseTab(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.run_dir = ''
        self.frames: list = []
        self.cache = None                     # (index, sensor, rgb, label colours, tags)
        h = QHBoxLayout(self)
        split = QSplitter()
        h.addWidget(split)
        self.runs = QTreeWidget()
        self.runs.setHeaderLabels(['Run', 'Status', 'Steps'])
        self.runs.setRootIsDecorated(False)
        self.runs.currentItemChanged.connect(lambda it, _p: self.show_run(it.data(0, 256)) if it else None)
        split.addWidget(self.runs)
        right = QWidget()
        rv = QVBoxLayout(right)
        top = QHBoxLayout()
        self.sensor = QComboBox()
        self.layer = QComboBox()
        self.layer.addItems(['semantic'])
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.setFixedWidth(140)
        top.addWidget(label('Camera', muted=True))
        top.addWidget(self.sensor)
        top.addWidget(label('Layer', muted=True))
        top.addWidget(self.layer)
        top.addWidget(label('Opacity', muted=True))
        top.addWidget(self.opacity)
        top.addStretch(1)
        self.open_btn = button('Open folder', self.open_folder)
        top.addWidget(self.open_btn)
        rv.addLayout(top)
        self.view = ImageView()
        rv.addWidget(self.view, 1)
        fr = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_lbl = label('', mono=True)
        fr.addWidget(self.slider, 1)
        fr.addWidget(self.frame_lbl)
        rv.addLayout(fr)
        lower = QHBoxLayout()
        self.legend = QWidget()
        self.legend_grid = QGridLayout(self.legend)
        self.legend_grid.setContentsMargins(0, 0, 0, 0)
        leg = Section('Classes in this frame')
        leg.add(self.legend)
        lower.addWidget(leg, 1)
        info = Section('Run')
        self.info = KeyValues()
        info.add(self.info)
        lower.addWidget(info, 1)
        rv.addLayout(lower)
        split.addWidget(right)
        split.setSizes([300, 900])
        self.slider.valueChanged.connect(self.load_frame)
        self.sensor.currentIndexChanged.connect(lambda _: self.load_frame(self.slider.value()))
        self.opacity.valueChanged.connect(lambda _: self.compose())

    def refresh(self):
        cat = self.win.service.catalog
        cur = self.run_dir
        self.runs.blockSignals(True)
        self.runs.clear()
        if cat is not None:
            for r in cat.list('run', 500):
                if r['summary'].get('worker') not in (None, 'drive'):
                    continue
                it = QTreeWidgetItem([r['name'], r['status'], str(r['summary'].get('steps', ''))])
                it.setData(0, 256, r['path'])
                it.setToolTip(0, tilde(r['path']))
                sev = {'ok': 'good', 'cancelled': 'warning'}.get(r['status'], 'critical')
                it.setIcon(1, icons.status_icon(sev))
                self.runs.addTopLevelItem(it)
                if r['path'] == cur:
                    self.runs.setCurrentItem(it)
        self.runs.blockSignals(False)

    def show_run(self, path):
        self.run_dir = path or ''
        self.frames, self.cache = [], None
        self.info.clear()
        self.sensor.blockSignals(True)
        self.sensor.clear()
        self.sensor.blockSignals(False)
        if not path:
            return
        try:
            with open(os.path.join(path, 'run.json'), encoding='utf-8') as fh:
                run = json.load(fh)
        except (OSError, ValueError):
            run = {}
        perf = run.get('performance') or {}
        prov = run.get('provenance') or {}
        self.info.set('Status', f'{run.get("status", "?")}' + (f' — {run["error"]}' if run.get('error') else ''))
        self.info.set('Folder', tilde(path))
        if perf.get('steps'):
            parts = [f'{perf["steps"]} steps in {perf.get("wall_s", 0):.1f} s']
            if 'realtime_factor' in perf:
                parts.append(f'{perf["realtime_factor"]:.2f}× real time')
            if 'mpx_per_s' in perf:
                parts.append(f'{perf["mpx_per_s"]:.0f} Mpx/s')
            self.info.set('Measured', ', '.join(parts))
        host = prov.get('host') or {}
        self.info.set('Ran with', f'seed {prov.get("seed")}, CARLA {host.get("carla_server", "?")}, Falcon '
                                  f'{prov.get("app_version", "?")}')
        if run.get('config_sha256'):
            self.info.set('Config', run['config_sha256'][:16] + '…', 'SHA-256 of drive.json (canonical JSON)')
        for n in run.get('notes') or []:
            if not n.startswith('semantic labels:'):
                self.info.set('Note', n)
        try:
            with open(os.path.join(path, 'dataset.json'), encoding='utf-8') as fh:
                for wtext in json.load(fh).get('label_warnings') or []:
                    self.info.set('⚠ Labels', wtext)
        except (OSError, ValueError):
            pass
        idx = os.path.join(path, 'frames.jsonl')
        if not os.path.isfile(idx):
            self.view.set_rgb(None)
            self.slider.setRange(0, 0)
            self.frame_lbl.setText('no frames (only run.json)')
            return
        with open(idx, encoding='utf-8') as fh:
            self.frames = [json.loads(x) for x in fh if x.strip()]
        sensors = sorted({k.split('/')[0] for f in self.frames[:1] for k in f['files']})
        self.sensor.blockSignals(True)
        self.sensor.addItems(sensors)
        self.sensor.blockSignals(False)
        self.slider.setRange(0, max(len(self.frames) - 1, 0))
        self.slider.setValue(0)
        self.load_frame(0)

    def load_frame(self, i):
        if not self.frames or not 0 <= i < len(self.frames):
            return
        f = self.frames[i]
        s = self.sensor.currentText()
        self.frame_lbl.setText(f'step {f["step"]}  frame {f["frame"]}  t {f["sim_t"]:.2f} s  '
                               f'{f.get("speed_mps", 0) * 3.6:.0f} km/h')
        files = {k.split('/')[1]: os.path.join(self.run_dir, v) for k, v in f['files'].items()
                 if k.split('/')[0] == s}

        def load():
            import cv2
            rgb = cv2.imread(files['rgb'])[:, :, ::-1] if 'rgb' in files else None
            tags = cv2.imread(files['semantic'], cv2.IMREAD_UNCHANGED) if 'semantic' in files else None
            return i, s, rgb, tags
        tasks.run(self, load, self._loaded)

    def _loaded(self, res):
        i, s, rgb, tags = res
        if i != self.slider.value() or s != self.sensor.currentText():
            return                                 # a newer frame was asked for meanwhile
        self.cache = (rgb, None if tags is None else labels.colorize(tags), tags)
        self.compose()
        self.show_legend(tags)

    def compose(self):
        if not self.cache or self.cache[0] is None:
            return
        rgb, col, _ = self.cache
        a = self.opacity.value() / 100.0
        out = rgb if col is None or a <= 0 else (rgb * (1 - a) + col * a).astype(np.uint8)
        self.view.set_rgb(out)

    def show_legend(self, tags):
        while self.legend_grid.count():
            it = self.legend_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if tags is None:
            return
        vals, counts = np.unique(tags, return_counts=True)
        order = np.argsort(-counts)
        total = float(tags.size)
        for r, k in enumerate(order[:24]):
            t = int(vals[k])
            sw = QLabel()
            sw.setFixedSize(12, 12)
            c = labels.color(t)
            sw.setStyleSheet(f'background: {QColor(*c).name()}; border-radius: 2px;')
            self.legend_grid.addWidget(sw, r // 2, (r % 2) * 3)
            self.legend_grid.addWidget(label(labels.name(t)), r // 2, (r % 2) * 3 + 1)
            self.legend_grid.addWidget(label(f'{100 * counts[k] / total:.1f} %', muted=True, mono=True), r // 2,
                                       (r % 2) * 3 + 2)

    def open_folder(self):
        if self.run_dir:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.run_dir))


class DataPage(Page):
    key, title, icon = 'data', 'Data', 'dataset'
    about = 'Record drives with perfectly labelled frames, and browse what was recorded.'

    def __init__(self, win):
        super().__init__(win)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.addWidget(label(self.title, title=True))
        v.addWidget(label(self.about, muted=True))
        self.tabs = QTabWidget()
        self.record = RecordTab(self)
        self.browse = BrowseTab(self)
        self.tabs.addTab(scroll(self.record), 'Record')
        self.tabs.addTab(self.browse, 'Browse')
        v.addWidget(self.tabs, 1)
        self.refresh()

    def refresh(self):
        self.record.refresh()
        self.browse.refresh()

    def show_run(self, path):
        self.tabs.setCurrentWidget(self.browse)
        self.browse.refresh()
        self.browse.show_run(path)

    def on_run_finished(self, job):
        self.browse.refresh()
        self.record.update_cost()
