"""Rigs ▸ Surround view: the selected rig's cameras stitched into the bird's-eye view (core.bevstitch; docs/bev.md) —
a recorded run of that rig (two or more cameras) step by step with a frame slider, the seams drawn over it, exposure
balancing, the stitcher's frame time, and the remap tables exported for the vehicle with the Apache-2.0 reference
stitcher. Building the tables and stitching run on a worker thread; nothing heavy is imported at start-up."""
from __future__ import annotations

import json
import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .widgets import KeyValues, button, label

VIEW_PX = 360


class BevTab(QWidget):
    """Built by the Rigs page (ui.widgets.optional_tab) as BevTab(page)."""

    def __init__(self, owner=None):
        super().__init__()
        self.page = owner
        self.win = getattr(owner, 'win', None)
        self.lut = None
        self.lut_key = None
        self.stitcher = None
        self.run_dir = ''
        self.steps: list = []
        self.busy = False
        self.pending = False
        self.last = None                         # the last stitched result (tests read it)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(label('The ground around the car seen from above, stitched from the rig\'s cameras. Only the '
                          'ground is right: anything standing on it smears outwards and doubles or vanishes at a '
                          'seam (no bowl).', muted=True, small=True, wrap=True))
        row = QHBoxLayout()
        self.runs = QComboBox()
        self.runs.setMinimumWidth(180)
        self.runs.currentIndexChanged.connect(lambda _: self.open_run())
        row.addWidget(self.runs, 1)
        row.addWidget(button('Choose run…', self.choose_run, tip='a recorded run of this rig'))
        v.addLayout(row)
        self.view = QLabel()
        self.view.setMinimumHeight(220)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.view)
        self.frame = QSlider(Qt.Orientation.Horizontal)
        self.frame.valueChanged.connect(lambda _: self.request())
        v.addWidget(self.frame)
        row = QHBoxLayout()
        self.seams = QCheckBox('Seams')
        self.seams.setChecked(True)
        self.seams.toggled.connect(lambda _: self.request())
        self.balance = QCheckBox('Balance exposure')
        self.balance.toggled.connect(lambda _: self.request())
        self.device = QComboBox()
        self.device.addItem('OpenCV (CPU)', 'cpu')
        self.device.addItem('CUDA (torch, half precision)', 'cuda')
        self.device.currentIndexChanged.connect(lambda _: self.reset_stitcher())
        for w in (self.seams, self.balance, self.device):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(button('Export tables…', self.export, tip='bev_lut.json/.npz and the reference stitcher '
                                                                '(Apache-2.0) for the vehicle'))
        v.addLayout(row)
        self.info = KeyValues(('Tables', 'Seams', 'Frame time'))
        v.addWidget(self.info)
        self.note = label('', muted=True, small=True, wrap=True)
        v.addWidget(self.note)
        rigs = getattr(owner, 'rigs', None)
        if rigs is not None and hasattr(rigs, 'currentRowChanged'):
            rigs.currentRowChanged.connect(lambda _: self.refresh())
        self.refresh()

    # ------------------------------------------------------------------ the rig
    def rig(self):
        r = self.page.rig() if self.page is not None and hasattr(self.page, 'rig') else None
        return r

    def cameras(self) -> list:
        r = self.rig()
        return [s['id'] for s in (r or {}).get('sensors') or [] if str(s.get('type', '')).startswith('camera.')]

    def refresh(self):
        """The rig changed: its runs, and the tables built again when needed."""
        cams = self.cameras()
        self.lut = self.stitcher = None
        self.view.clear()
        if len(cams) < 2:
            self.note.setText(f'This rig has {len(cams)} camera(s): a surround view needs two or more.')
        else:
            self.note.setText('')
        self.refresh_runs()

    def refresh_runs(self):
        cur = self.runs.currentData()
        rig = self.rig() or {}
        self.runs.blockSignals(True)
        self.runs.clear()
        svc = getattr(self.win, 'service', None) if self.win is not None else None
        root = svc.runs_dir() if svc is not None else ''
        for d in sorted(os.listdir(root), reverse=True) if root and os.path.isdir(root) else []:
            p = os.path.join(root, d)
            if self._run_rig_id(p) == rig.get('id'):
                self.runs.addItem(d, p)
        if self.run_dir and self.runs.findData(self.run_dir) < 0 and os.path.isdir(self.run_dir):
            self.runs.addItem(os.path.basename(self.run_dir), self.run_dir)
        i = self.runs.findData(cur) if cur else -1
        self.runs.setCurrentIndex(i if i >= 0 else (0 if self.runs.count() else -1))
        self.runs.blockSignals(False)
        if self.runs.count():
            self.open_run()

    @staticmethod
    def _run_rig_id(p: str):
        dj = os.path.join(p, 'drive.json')
        if not (os.path.isfile(dj) and os.path.isfile(os.path.join(p, 'frames.jsonl'))):
            return None
        try:
            with open(dj, encoding='utf-8') as fh:
                return (json.load(fh).get('rig') or {}).get('id')
        except (OSError, ValueError):
            return None

    def choose_run(self):
        d = QFileDialog.getExistingDirectory(self, 'A recorded run of this rig')
        if d:
            self.set_run(d)

    def set_run(self, path: str):
        self.run_dir = os.path.abspath(path)
        i = self.runs.findData(self.run_dir)
        if i < 0:
            self.runs.addItem(os.path.basename(self.run_dir), self.run_dir)
            i = self.runs.count() - 1
        self.runs.blockSignals(True)
        self.runs.setCurrentIndex(i)
        self.runs.blockSignals(False)
        self.open_run()

    def open_run(self):
        p = self.runs.currentData()
        if not p:
            return
        self.run_dir = p
        cams = self.cameras()
        steps = []
        try:
            with open(os.path.join(p, 'frames.jsonl'), encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        if sum(1 for c in cams if f'{c}/rgb' in (r.get('files') or {})) >= 2:
                            steps.append(int(r['step']))
        except (OSError, ValueError) as e:
            self.note.setText(f'cannot read {p}: {e}')
            return
        self.steps = steps
        self.frame.blockSignals(True)
        self.frame.setRange(0, max(len(steps) - 1, 0))
        self.frame.setValue(0)
        self.frame.blockSignals(False)
        if not steps:
            self.note.setText(f'{os.path.basename(p)} has no step with two of this rig\'s cameras')
            return
        self.request()

    def reset_stitcher(self):
        self.stitcher = None
        self.request()

    # ------------------------------------------------------------------ stitching
    def request(self):
        from . import tasks
        rig = self.rig()
        if rig is None or len(self.cameras()) < 2 or not self.steps or not self.run_dir:
            return
        if self.busy:
            self.pending = True
            return
        self.busy = True
        step = self.steps[min(self.frame.value(), len(self.steps) - 1)]
        run, seams, balance, dev = self.run_dir, self.seams.isChecked(), self.balance.isChecked(), \
            self.device.currentData()
        key = json.dumps({k: rig.get(k) for k in ('id', 'vehicle', 'sensors', 'vantage')}, sort_keys=True,
                         default=str)

        def work():
            from ..core import bevstitch as B
            if self.lut is None or self.lut_key != key:
                self.lut, self.lut_key, self.stitcher = B.build_lut(rig), key, None
            if self.stitcher is None:
                try:
                    self.stitcher = B.Stitcher(self.lut, dev)
                except B.BevError:
                    self.stitcher = B.Stitcher(self.lut, 'cpu')
            frames = B.run_frames(run, self.lut.ids, step)
            res = self.stitcher.stitch(frames, gains='auto' if balance else None)
            img = res.rgb()
            if seams:
                img = B.seam_overlay(self.lut, img)
            return {'image': img, 'step': step, 'ms': res.ms, 'backend': res.backend, 'cameras': sorted(frames),
                    'clipped': res.clipped, 'summary': self.lut.summary()}

        def done(out):
            self.busy = False
            self.show(out)
            if self.pending:
                self.pending = False
                self.request()

        def fail(e):
            self.busy = False
            self.note.setText(f'stitching failed: {e}')
        tasks.run(self, work, done, fail)

    def show(self, out: dict):
        self.last = out
        img = np.ascontiguousarray(out['image'])
        h, w = img.shape[:2]
        q = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.view.setPixmap(QPixmap.fromImage(q).scaled(max(self.view.width(), VIEW_PX), VIEW_PX,
                                                        Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))
        s = out['summary']
        self.info.set('Tables', f'{s["size"][0]} × {s["size"][1]} px at {s["resolution_m"] * 100:.0f} cm, '
                                f'{len(s["cameras"])} cameras, {s["covered_share"] * 100:.0f} % of the ground seen')
        self.info.set('Seams', f'{s["seam_source"]}: ' + ', '.join(
            f'{x["a_name"]}|{x["b_name"]} {x["angle_deg"]:+.0f}°' for x in s['seams']))
        self.info.set('Frame time', f'{out["ms"]:.1f} ms ({1000.0 / max(out["ms"], 1e-6):.0f} fps) on '
                                    f'{out["backend"]}, step {out["step"]}')
        self.note.setText(f'gains clipped: {out["clipped"]}' if out.get('clipped') else '')

    def export(self, out_dir: str | None = None):
        rig = self.rig()
        if rig is None or len(self.cameras()) < 2:
            self.note.setText('a surround view needs two or more cameras')
            return None
        out_dir = out_dir or QFileDialog.getExistingDirectory(self, 'Export the remap tables into')
        if not out_dir:
            return None
        from ..core import bevstitch as B
        lut = self.lut if self.lut is not None else B.build_lut(rig)
        paths = B.export(lut, out_dir, rig)
        self.note.setText(f'{len(paths)} files in {out_dir} (stitcher/bev_stitch.py runs them on the vehicle)')
        return paths
