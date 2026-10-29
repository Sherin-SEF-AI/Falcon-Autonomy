"""The Replay page's reconstruction section: train a 3D Gaussian Splatting reconstruction from the imported log's own
camera images (a job on the GPU queue), look at it from poses the log never took — any frame, the ego moved sideways
with a slider — and read how it measured (core.recon; docs/reconstruction.md). torch is imported only when a
reconstruction is opened, on a worker thread."""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QSpinBox, QWidget

from ...core.jobs import DONE, FINAL
from ..widgets import Form, KeyValues, Section, SliderField, button, label

READS = {'falcon': 'falcon', 'kitti_raw': 'kitti_raw', 'nuscenes': 'nuscenes'}
VIEW_WIDTH = 480


def source_of(log) -> tuple:
    """(the reconstruction's source 'KIND:PATH[?scene=…]', or None and why) for an imported log."""
    src = (log.meta.get('source') or {}) if log is not None else {}
    kind, path = src.get('format'), src.get('path')
    if kind not in READS or not path:
        return None, (f'a reconstruction needs the log\'s camera images and their poses: Falcon runs, KITTI raw and '
                      f'nuScenes are read; {kind or "this log"} is read for trajectories only')
    if not os.path.exists(path):
        return None, f'the log\'s source is no longer at {path}'
    if kind == 'nuscenes':
        return f'nuscenes:{path}?scene={src.get("scene") or ""}', ''
    return f'{kind}:{path}', ''


class ReconPanel(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.win = page, page.win
        self.log = None
        self.source = None
        self.rec_dir = ''
        self.rec = None
        self.jobs = {}                         # 'train' | 'eval' | 'rerender' -> job
        self.busy = False
        self.pending = False
        self.last = None                       # the last rendered result (tests read it)
        from PyQt6.QtWidgets import QVBoxLayout
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        sec = Section('Reconstruction: the log\'s own appearance',
                      'A 3D Gaussian Splatting reconstruction (Kerbl et al. 2023) of the log\'s camera images — the '
                      'static scene only: moving actors are masked out where the source says where they are. Render '
                      'it from poses the log never took; extrapolation degrades quickly with distance from the path.')
        f = Form()
        self.iters = QSpinBox()
        self.iters.setRange(10, 30000)
        self.iters.setValue(7000)
        self.width_px = QSpinBox()
        self.width_px.setRange(64, 1280)
        self.width_px.setValue(640)
        self.init = QComboBox()
        for k, text in (('auto', 'Lidar if the log has it, else structure from the poses'), ('lidar', 'Lidar'),
                        ('sfm', 'Structure from the known poses'), ('random', 'Random')):
            self.init.addItem(text, k)
        self.device = QComboBox()
        for k, text in (('auto', 'The GPU when there is one'), ('cpu', 'The CPU (slow)')):
            self.device.addItem(text, k)
        for text, w in (('Steps', self.iters), ('Image width', self.width_px), ('Start from', self.init),
                        ('Device', self.device)):
            f.row(text, w)
        sec.add(f)
        b = QHBoxLayout()
        self.train_btn = button('Train from this log', self.train, role='primary')
        b.addWidget(self.train_btn)
        b.addWidget(button('Measure novel views', self.evaluate,
                           tip='held-out frames, and the ego 0.5 m and 1 m to either side against CARLA\'s range-'
                               'warped frames (Falcon runs)'))
        b.addStretch(1)
        sec.add(b)
        self.info = KeyValues(('Source', 'Reconstruction', 'Held out', 'Novel views', 'Cost'))
        sec.add(self.info)
        self.view = QLabel()
        self.view.setMinimumHeight(200)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sec.add(self.view)
        f = Form()
        self.frame = QSlider(Qt.Orientation.Horizontal)
        self.frame.valueChanged.connect(lambda _: self.request_render())
        self.lateral = SliderField(-3.0, 3.0, 'm', 1, 0.1)
        self.lateral.valueChanged.connect(lambda _: self.request_render())
        f.row('Frame', self.frame)
        f.row('Ego to the right', self.lateral)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Re-render the log from the shifted path', self.rerender,
                           tip='a falcon-raw run of rendered images (RGB, the reconstruction\'s depth, voted classes '
                               'for Falcon sources) the harness can run models on'))
        b.addStretch(1)
        sec.add(b)
        self.note = label('', muted=True, wrap=True)
        sec.add(self.note)
        v.addWidget(sec)

    # ---------------------------------------------------------------- the log and its reconstruction
    def set_log(self, log, path: str):
        self.log = log
        self.source, why = source_of(log)
        name = os.path.basename(os.path.dirname(path)) if path else 'log'
        self.rec_dir = os.path.join(self.win.workspace(), 'recon', name)
        self.info.set('Source', self.source or why)
        self.train_btn.setEnabled(self.source is not None)
        self.rec = None
        self.view.clear()
        if os.path.isfile(os.path.join(self.rec_dir, 'point_cloud.ply')):
            self.open_recon()
        else:
            self.info.set('Reconstruction', 'none yet')
            for k in ('Held out', 'Novel views', 'Cost'):
                self.info.set(k, '—')

    def _submit(self, which: str, argv: list, title: str):
        from ...core.recon.train import job_spec
        busy = self.jobs.get(which)
        if busy is not None and busy.state not in FINAL:
            return None
        job = self.win.queue.submit(job_spec(argv, title))
        self.jobs[which] = job
        self.win.jobs_dock.raise_()
        return job

    def train(self):
        if not self.source:
            return
        argv = ['train', '--source', self.source, '--out', self.rec_dir, '--iterations', str(self.iters.value()),
                '--width', str(self.width_px.value()), '--init', self.init.currentData(), '--device',
                self.device.currentData(), '--seed', str(self.page.seed.value())]
        if self._submit('train', argv, f'Reconstruct {os.path.basename(self.rec_dir)}'):
            self.info.set('Reconstruction', f'training: {self.rec_dir}')

    def evaluate(self):
        if not os.path.isfile(os.path.join(self.rec_dir, 'point_cloud.ply')):
            self.note.setText('train a reconstruction first')
            return
        argv = ['eval', self.rec_dir, '--warp-shift', '0.5', '--warp-shift', '1.0', '--warp-check', '1.0',
                '--device', self.device.currentData()]
        self._submit('eval', argv, f'Measure {os.path.basename(self.rec_dir)}')

    def rerender(self, out: str | None = None):
        if self.rec is None:
            self.note.setText('open a reconstruction first')
            return None
        from ...core.drive import run_dir_name
        lat = self.lateral.value()
        out = out or os.path.join(self.win.service.runs_dir() or os.path.join(self.win.workspace(), 'runs'),
                                  run_dir_name(f'recon-{os.path.basename(self.rec_dir)}-{lat:+.1f}m'))
        argv = ['rerender', self.rec_dir, '--out', out, '--lateral', f'{lat}', '--device', self.device.currentData()]
        job = self._submit('rerender', argv, f'Re-render {os.path.basename(self.rec_dir)} {lat:+.1f} m')
        if job:
            self.note.setText(f're-rendering into {out}')
        return out

    def on_finished(self, job):
        which = next((k for k, j in self.jobs.items() if j is job), None)
        if which is None:
            return
        self.note.setText(f'{which}: {job.state}' + (f' — {job.summary()}' if job.state != DONE else ''))
        if job.state == DONE and which in ('train', 'eval'):
            self.open_recon()

    def open_recon(self):
        from .. import tasks
        path, dev = self.rec_dir, self.device.currentData()
        self.info.set('Reconstruction', f'opening {path}…')

        def work():
            from ...core.recon.scene import Recon
            return Recon.load(path, dev)
        tasks.run(self, work, self.opened, lambda e: self.info.set('Reconstruction', f'cannot open it: {e}'))

    def opened(self, rec):
        self.rec = rec
        s = rec.run.get('summary') or {}
        tm = rec.run.get('timings') or {}
        dv = rec.run.get('device') or {}
        self.info.set('Reconstruction', f'{s.get("gaussians", len(rec.gs))} Gaussians from {s.get("init", "?")}, '
                                        f'{s.get("train_frames", "?")} training frames at {rec.work_size[0]} × '
                                        f'{rec.work_size[1]}', tip=rec.path)
        self.info.set('Cost', f'{tm.get("train_s", "?")} s training ({tm.get("steps_per_s", "?")} steps/s) on '
                              f'{dv.get("name", "?")}' + (f', peak {dv["peak_gpu_memory_gib"]} GiB'
                                                          if dv.get('peak_gpu_memory_gib') else ''))
        self.show_metrics()
        self.frame.blockSignals(True)
        self.frame.setRange(0, len(rec.frames) - 1)
        self.frame.setValue(min(self.frame.value(), len(rec.frames) - 1))
        self.frame.blockSignals(False)
        self.request_render()

    def show_metrics(self):
        from ...core.provenance import read_json
        p = os.path.join(self.rec_dir, 'evaluation.json')
        if not os.path.isfile(p):
            return
        ev = read_json(p)
        h = (ev.get('heldout') or {}).get('mean') or {}
        if h:
            d = f', depth median error {100 * h["depth_median_abs_rel"]:.1f} %' if h.get('depth_median_abs_rel') \
                is not None else ''
            self.info.set('Held out', f'{h.get("frames")} frames: PSNR {h.get("psnr")} dB, SSIM {h.get("ssim")}{d} '
                                      f'(LPIPS not computed)')
        nv = (ev.get('novel_view') or {}).get('lateral_warped') or {}
        rows = [f'{s["lateral_m"]:+.1f} m: {s["psnr"]} dB' for s in nv.get('shifts') or [] if s.get('psnr')]
        if rows:
            self.info.set('Novel views', '; '.join(rows) + ' (against range-warped CARLA frames)')

    # ---------------------------------------------------------------- the viewer
    def request_render(self):
        from .. import tasks
        if self.rec is None:
            return
        if self.busy:
            self.pending = True
            return
        self.busy = True
        rec, k, lat = self.rec, self.frame.value(), self.lateral.value()

        def work():
            f = rec.frame(k)
            w0, h0 = rec.work_size
            w = min(VIEW_WIDTH, w0)
            cam = rec.camera(f, lateral_m=lat, size=(w, max(1, round(h0 * w / w0))))
            out = rec.render(cam, semantic=False)
            return {'image': out['image'], 'frame': k, 'lateral_m': lat, 'split': f['split']}

        def done(res):
            self.busy = False
            self.show(res)
            if self.pending:
                self.pending = False
                self.request_render()

        def fail(e):
            self.busy = False
            self.note.setText(f'render failed: {e}')
        tasks.run(self, work, done, fail)

    def show(self, res):
        self.last = res
        img = np.ascontiguousarray(res['image'])
        h, w = img.shape[:2]
        q = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.view.setPixmap(QPixmap.fromImage(q).scaled(self.view.width(), max(self.view.height(), 200),
                                                        Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))
        self.note.setText(f'frame {res["frame"]} ({res["split"]}), the ego {res["lateral_m"]:+.1f} m to the right')
