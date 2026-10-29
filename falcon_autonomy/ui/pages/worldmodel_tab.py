"""Training ▸ World model: train a learned world model (core.wmtrain: a DreamerV3-style RSSM) on checked drives and
experience folders, collect environment episodes for it, evaluate its open-loop predictions on held-out drives
against the copy-last-frame and constant-velocity baselines (a chart of IoU against the horizon per channel, and the
predicted BEV beside the truth over a horizon slider), learn a policy in its imagination, and clone an end-to-end
camera policy (core.e2e). Every action is a `falcon worldmodel …` job; training, evaluation, imagination and cloning
are needs_gpu jobs (never at full load beside CARLA). Policies made here join the Driving policies and Leaderboard
lists."""
from __future__ import annotations

import glob
import os
import time

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                             QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ...core import wmtrain as WT
from ...core.paths import tilde
from ...core.provenance import read_json
from ..theme import THEME
from ..widgets import Form, KeyValues, Section, button, label, scroll

# BEV colours, drawn in this order (later over earlier): drivable, centerline, lane boundary, vehicle, pedestrian,
# cyclist, other, ego — the map greys, the actors in distinct hues
BEV_RGB = ((70, 70, 66), (150, 150, 140), (220, 220, 210), (60, 130, 230), (240, 140, 40), (200, 70, 200),
           (80, 190, 110), (250, 250, 250))
MODELS = (('model', 'learned model'), ('copy_last', 'copy last frame'), ('constant_velocity', 'constant velocity'))


def bev_rgb(bits: np.ndarray, background=(26, 26, 25)) -> np.ndarray:
    """(H, W) packed BEV bits → (H, W, 3) uint8 RGB with BEV_RGB."""
    b = np.asarray(bits, np.uint8)
    img = np.empty(b.shape + (3,), np.uint8)
    img[:] = background
    for k, col in enumerate(BEV_RGB):
        img[(b >> k) & 1 == 1] = col
    return img


def _pixmap(rgb: np.ndarray, scale: int = 3) -> QPixmap:
    h, w = rgb.shape[:2]
    a = np.ascontiguousarray(rgb)
    qi = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qi).scaled(w * scale, h * scale, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.FastTransformation)


def _marker(p: QPainter, shape: int, x: float, y: float, r: float = 4.0):
    k = shape % 3
    if k == 0:
        p.drawEllipse(QPointF(x, y), r, r)
    elif k == 1:
        p.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))
    else:
        p.drawPolygon(QPolygonF([QPointF(x, y - r - 1), QPointF(x + r + 1, y + r), QPointF(x - r - 1, y + r)]))


class HorizonChart(QWidget):
    """IoU of one channel against the horizon: the learned model and each baseline (colour and marker shape)."""

    def __init__(self):
        super().__init__()
        self.ev, self.channel = None, 'actors'
        self.setMinimumHeight(220)

    def set_eval(self, ev, channel):
        self.ev, self.channel = ev, channel
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        ev = self.ev
        if not ev or self.channel not in ev.get('channels', []):
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No evaluation yet: evaluate the model')
            return
        K = ev['horizon']
        x0, x1, y0, y1 = 48, self.width() - 16, 12, self.height() - 52

        def X(k):
            return x0 + (x1 - x0) * (k - 1) / max(K - 1, 1)

        def Y(v):
            return y1 - (y1 - y0) * v
        p.setPen(QColor(THEME.tok['grid2']))
        for k in range(5):
            p.drawLine(QPointF(x0, Y(k / 4)), QPointF(x1, Y(k / 4)))
        p.setPen(QColor(THEME.tok['muted']))
        for k in range(5):
            p.drawText(QRectF(0, Y(k / 4) - 8, x0 - 6, 16), Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter, f'{k / 4:.2f}')
        for k in ev['report_horizons']:
            p.drawText(QRectF(X(k) - 20, y1 + 4, 40, 14), Qt.AlignmentFlag.AlignCenter, str(k))
        p.drawText(QRectF(x0, y1 + 18, x1 - x0, 14), Qt.AlignmentFlag.AlignCenter,
                   f'horizon (steps of {ev.get("dt_s", 0.05):.2f} s) — IoU of {self.channel}')
        lx, ly = x0, self.height() - 8
        for i, (key, name) in enumerate(MODELS):
            vals = (ev['iou'].get(key) or {}).get(self.channel)
            if vals is None:
                continue
            col = THEME.cam(i)
            p.setPen(QPen(col, 2))
            pts = [(X(k + 1), v) for k, v in enumerate(vals)]
            for (xa, a), (xb, b) in zip(pts, pts[1:]):
                if a is not None and b is not None:
                    p.drawLine(QPointF(xa, Y(a)), QPointF(xb, Y(b)))
            p.setBrush(col)
            p.setPen(QPen(QColor(THEME.tok['panel']), 1.0))
            for k in ev['report_horizons']:
                v = vals[k - 1]
                if v is not None:
                    _marker(p, i, X(k), Y(v))
            _marker(p, i, lx + 5, ly - 4)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QColor(THEME.tok['ink2']))
            p.drawText(QPointF(lx + 14, ly), name)
            lx += 30 + p.fontMetrics().horizontalAdvance(name)


class BevViewer(QWidget):
    """A few held-out starts: the truth, the learned model's prediction and a baseline's at horizon k (slider)."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.sample = QComboBox()
        self.sample.currentIndexChanged.connect(lambda _: self.show_frame())
        self.base = QComboBox()
        self.base.addItem('copy last frame', 'copy_last')
        self.base.addItem('constant velocity', 'constant_velocity')
        self.base.currentIndexChanged.connect(lambda _: self.show_frame())
        top.addWidget(self.sample, 2)
        top.addWidget(self.base, 1)
        v.addLayout(top)
        row = QHBoxLayout()
        self.views = []
        for t in ('Truth', 'Learned model', 'Baseline'):
            col = QVBoxLayout()
            cap = label(t, muted=True)
            img = QLabel()
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img.setMinimumSize(150, 190)
            col.addWidget(cap)
            col.addWidget(img, 1)
            row.addLayout(col, 1)
            self.views.append(img)
        v.addLayout(row, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(lambda _: self.show_frame())
        v.addWidget(self.slider)
        self.caption = label('', muted=True, wrap=True)
        v.addWidget(self.caption)
        self.z = None
        self.dt = 0.05

    def load(self, path: str, dt: float = 0.05):
        self.z, self.dt = None, dt
        self.sample.blockSignals(True)
        self.sample.clear()
        if path and os.path.isfile(path):
            try:
                with np.load(path) as z:
                    self.z = {k: z[k] for k in z.files}
            except (OSError, ValueError):
                self.z = None
        if self.z is not None:
            for k, n in enumerate(self.z['names']):
                self.sample.addItem(str(n), k)
            self.slider.setRange(0, self.z['truth'].shape[1] - 1)
            self.slider.setValue(min(5, self.z['truth'].shape[1] - 1))
        self.sample.blockSignals(False)
        self.show_frame()

    def show_frame(self):
        if self.z is None or self.sample.count() == 0:
            for w in self.views:
                w.setPixmap(QPixmap())
                w.setText('—')
            self.caption.setText('No predictions yet: evaluate the model')
            return
        j = max(self.sample.currentIndex(), 0)
        k = self.slider.value()
        base = self.base.currentData()
        if base == 'constant_velocity' and not bool(self.z['has_cv'][j]):
            base = 'copy_last'
        ok = bool(self.z['valid'][j, k])
        for w, key in zip(self.views, ('truth', 'model', base)):
            if ok:
                w.setPixmap(_pixmap(bev_rgb(self.z[key][j, k])))
            else:
                w.setPixmap(QPixmap())
                w.setText('beyond the sequence')
        self.caption.setText(f'k = {k} steps ({k * self.dt:.2f} s) ahead of the start; heading up, 0.5 m pixels '
                             f'by default. Grey: lanes; blue: vehicles; orange: pedestrians; magenta: cyclists; '
                             f'green: other; white: the ego. The model sees nothing after the start — only the '
                             f'recorded actions.')


class WorldModelTab(QWidget):
    """The tab; `page` is the Training page (its workspace, job submission and drive lists)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        sec = Section('Train a world model', 'A recurrent state-space model in the form of DreamerV3 (Hafner et al. '
                                             '2023) learns to predict the BEV, the reward and the episode\'s end '
                                             'from the actions. It holds whole drives out and is judged on them '
                                             'against copying the last frame and constant velocity.')
        f = Form()
        self.iters = QSpinBox()
        self.iters.setRange(10, 1_000_000)
        self.iters.setValue(WT.DEFAULTS['iterations'])
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.device = QComboBox()
        for d in ('auto', 'cpu', 'cuda'):
            self.device.addItem(d, d)
        f.row('Iterations', self.iters)
        f.row('Seed', self.seed)
        f.row('Device', self.device)
        sec.add(f)
        sec.add(label('Drives and experience folders', h=True))
        self.sources = QListWidget()
        self.sources.setMinimumHeight(120)
        from ..drivesets import DriveSetBar
        sec.add(DriveSetBar(self.page.win, self.sources))
        sec.add(self.sources)
        b = QHBoxLayout()
        b.addWidget(button('Train', self.train, role='primary'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        sec = Section('Collect environment episodes', 'A behaviour with exploration noise drives the learning '
                                                      'environment: episodes with the environment\'s own rewards '
                                                      'and endings, and actions the recorded drivers never took.')
        f = Form()
        self.episodes = QSpinBox()
        self.episodes.setRange(1, 10000)
        self.episodes.setValue(20)
        self.behaviour = QComboBox()
        self.behaviour.addItem('scripted lane keeping + noise (heuristic)', 'scripted')
        self.behaviour.addItem('noise only', 'random')
        f.row('Episodes', self.episodes)
        f.row('Behaviour', self.behaviour)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Collect', self.collect))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        sec = Section('Learn a policy in imagination', 'Dreamer\'s actor-critic on the chosen model\'s imagined '
                                                       'rollouts. The policy is then judged in the real '
                                                       'environment (Leaderboard), where it may well not drive.')
        f = Form()
        self.im_iters = QSpinBox()
        self.im_iters.setRange(10, 1_000_000)
        self.im_iters.setValue(WT.IMAGINE_DEFAULTS['iterations'])
        f.row('Iterations', self.im_iters)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Imagine', self.imagine))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        sec = Section('End-to-end driving', 'Behaviour cloning from the camera (and the BEV) to the recorded '
                                            'control of the checked drives (they must have recorded control). Score '
                                            'it on the Leaderboard tab: it drives with its own camera.')
        f = Form()
        self.e2e_epochs = QSpinBox()
        self.e2e_epochs.setRange(1, 1000)
        self.e2e_epochs.setValue(30)
        self.e2e_bev = QCheckBox('Camera and BEV')
        f.row('Epochs', self.e2e_epochs)
        f.row('', self.e2e_bev)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Clone', self.e2e))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rv = QVBoxLayout(right)
        self.models = QComboBox()
        self.models.currentIndexChanged.connect(lambda _: self.show_model())
        rv.addWidget(self.models)
        self.info = KeyValues(('Method', 'Data', 'Held out', 'Cost', 'Verdict'))
        rv.addWidget(self.info)
        b = QHBoxLayout()
        b.addWidget(button('Evaluate on the held-out drives', self.evaluate))
        b.addWidget(label('Channel', muted=True))
        self.channel = QComboBox()
        self.channel.currentIndexChanged.connect(lambda _: self.chart.set_eval(self.ev, self.channel.currentData()))
        b.addWidget(self.channel)
        b.addStretch(1)
        rv.addLayout(b)
        self.chart = HorizonChart()
        rv.addWidget(self.chart)
        self.viewer = BevViewer()
        rv.addWidget(self.viewer, 1)
        split.addWidget(right)
        split.setSizes([440, 820])
        self.ev = None

    # ------------------------------------------------------------------ data
    def ws(self, *parts):
        return self.page.ws(*parts)

    def refresh(self):
        self.page._fill_runs(self.sources)
        checked = set(self.page._checked(self.sources))
        for d in sorted(glob.glob(self.ws('experience', '*', 'experience.json')), reverse=True):
            p = os.path.dirname(d)
            it = QListWidgetItem(f'experience: {os.path.basename(p)}')
            it.setData(256, p)
            it.setToolTip(tilde(p))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if p in checked else Qt.CheckState.Unchecked)
            self.sources.addItem(it)
        self.page._combo_dirs(self.models, self.ws('worldmodels'), WT.MODEL_JSON)
        self.show_model()

    def show_model(self):
        d = self.models.currentData()
        self.ev = None
        if not d:
            self.info.clear()
            self.chart.set_eval(None, 'actors')
            self.viewer.load('')
            return
        try:
            doc = read_json(os.path.join(d, WT.MODEL_JSON))
        except (OSError, ValueError):
            return
        c = doc['config']
        self.info.set('Method', f'RSSM: GRU {c["deter"]}, {c["stoch"]} × {c["classes"]} categorical latents, BEV '
                                f'{c["bev_hw"][0]} × {c["bev_hw"][1]}, {len(doc["features"])} state features')
        cost = doc.get('cost') or {}
        self.info.set('Data', f'{cost.get("train_steps")} training steps from {len(doc["data"]["train"])} unit(s)')
        self.info.set('Held out', f'{len(doc["data"]["val"])} unit(s), {cost.get("val_steps")} steps' +
                      ('' if doc['data']['val'] else ' — nothing held out'))
        self.info.set('Cost', f'{cost.get("iterations")} iterations, {cost.get("train_s")} s on {cost.get("device")}'
                              + (f', peak {cost["peak_gpu_mib"]} MiB' if cost.get('peak_gpu_mib') else ''))
        ep = os.path.join(d, 'eval', 'eval.json')
        try:
            self.ev = read_json(ep)
        except (OSError, ValueError):
            self.ev = None
        self.channel.blockSignals(True)
        cur = self.channel.currentData() or 'actors'
        self.channel.clear()
        for ch in (self.ev or {}).get('channels', []):
            self.channel.addItem(ch, ch)
        k = self.channel.findData(cur)
        self.channel.setCurrentIndex(k if k >= 0 else 0)
        self.channel.blockSignals(False)
        if self.ev:
            beats = []
            for r in self.ev['report_horizons']:
                row = self.ev['verdict'][str(r)]
                won = [ch for ch, v in row.items() if v]
                beats.append(f'k={r}: {", ".join(won) if won else "none"}')
            self.info.set('Verdict', 'channels where the model beats every baseline — ' + '; '.join(beats))
        else:
            self.info.set('Verdict', 'not evaluated yet')
        self.chart.set_eval(self.ev, self.channel.currentData() or 'actors')
        self.viewer.load(os.path.join(d, 'eval', 'samples.npz'), (self.ev or {}).get('dt_s', 0.05))

    # ------------------------------------------------------------------ actions
    def _device(self):
        return self.device.currentData()

    def train(self):
        srcs = self.page._checked(self.sources)
        if not srcs:
            QMessageBox.information(self, 'World model', 'Check the drives (and experience folders) to train on.')
            return
        mid = time.strftime('%Y%m%d-%H%M%S-rssm')
        argv = ['train', '--workspace', self.page.win.workspace(), '--id', mid, '--iterations',
                str(self.iters.value()), '--seed', str(self.seed.value()), '--device', self._device()]
        for s in srcs:
            argv += ['--run', s]
        self.page._submit(WT.job_spec(argv, f'World model {mid}', needs_gpu=self._device() != 'cpu'))

    def evaluate(self):
        d = self.models.currentData()
        if not d:
            QMessageBox.information(self, 'World model', 'Train a world model first.')
            return
        argv = ['eval', '--model', d, '--device', self._device()]
        self.page._submit(WT.job_spec(argv, f'Evaluate {os.path.basename(d)}', memory_gb=3.0,
                                      needs_gpu=self._device() != 'cpu'))

    def imagine(self):
        d = self.models.currentData()
        if not d:
            QMessageBox.information(self, 'World model', 'Train a world model first.')
            return
        out = self.page.new_dir('policies', f'dreamer-{time.strftime("%Y%m%d-%H%M%S")}')
        argv = ['imagine', '--model', d, '--out', out, '--iterations', str(self.im_iters.value()), '--device',
                self._device()]
        self.page._submit(WT.job_spec(argv, f'Imagination on {os.path.basename(d)}',
                                      needs_gpu=self._device() != 'cpu'))

    def e2e(self):
        runs = [s for s in self.page._checked(self.sources) if not os.path.isfile(os.path.join(s, 'experience.json'))]
        if len(runs) < 2:
            QMessageBox.information(self, 'End-to-end driving', 'Check at least two drives with recorded control.')
            return
        out = self.page.new_dir('policies', f'e2e-{time.strftime("%Y%m%d-%H%M%S")}')
        argv = ['e2e', '--out', out, '--epochs', str(self.e2e_epochs.value()), '--device', self._device()]
        if self.e2e_bev.isChecked():
            argv.append('--bev')
        for r in runs:
            argv += ['--run', r]
        self.page._submit(WT.job_spec(argv, f'End-to-end cloning on {len(runs)} drive(s)',
                                      needs_gpu=self._device() != 'cpu'))

    def collect(self):
        if not self.page._ready():
            return
        out = self.page.new_dir('experience', time.strftime('%Y%m%d-%H%M%S'))
        argv = ['worldmodel', 'collect', self.page.win.project.path, '--out', out, '--episodes',
                str(self.episodes.value()), '--behaviour', self.behaviour.currentData(), '--start-speed', '30'] + \
            self.page._carla_args()
        self.page._cli('worldmodel', f'Collect {self.episodes.value()} episodes', argv, memory_gb=2.0)
