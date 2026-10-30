"""The Replay page's reconstruction section: train a 3D Gaussian Splatting reconstruction from the imported log's own
camera images (a job on the GPU queue), look at it from poses the log never took — any frame, the ego moved sideways
with a slider — and read how it measured (core.recon; docs/reconstruction.md). *From a dashcam* does the same for a
time window of a dashcam clip: posed by visual odometry and its GPS (core.recon.fromdashcam), reconstructed, placed on
an imported world. *Hybrid world* composites CARLA's actors over the reconstruction (core.recon.hybrid): a recorded
drive of the same world, or a new CARLA drive of the log's path with seeded traffic and the ego at the slider's offset
(a replay job, then the composite on the GPU queue). torch is imported only when a reconstruction is opened, on a
worker thread."""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QSlider,
                             QSpinBox, QWidget)

from ...core.jobs import DONE, FINAL
from ..widgets import Form, KeyValues, Section, SliderField, button, label, set_enabled

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
        self.jobs = {}                         # 'train' | 'eval' | 'rerender' | 'dashcam' | 'hybrid' | 'hybrid-carla'
        self.hybrid_next = None                # (carla run, hybrid run): the composite to queue when CARLA's drive ends
        self.hybrid_out = ''
        self.dash_dir = ''
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
        self.backend = QComboBox()
        for k, text in (('auto', 'Triton kernels on the GPU (PyTorch otherwise)'), ('torch', 'Plain PyTorch'),
                        ('triton', 'Triton kernels (refused without a CUDA GPU)')):
            self.backend.addItem(text, k)
        self.backend.setToolTip('Triton trains 6–12× faster on the RTX 5080 at the same quality (docs/reconstruction.md);'
                                ' PyTorch is the reference implementation')
        for text, w in (('Steps', self.iters), ('Image width', self.width_px), ('Start from', self.init),
                        ('Device', self.device), ('Rasteriser', self.backend)):
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
        self.info = KeyValues(('Source', 'Reconstruction', 'Poses', 'Held out', 'Novel views', 'Cost'))
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
        v.addWidget(self._hybrid_section())
        v.addWidget(self._dashcam_section())

    # ---------------------------------------------------------------- hybrid worlds
    def _hybrid_section(self) -> Section:
        sec = Section('Hybrid world: CARLA\'s actors in the reconstruction',
                      'CARLA simulates and renders the actors — traffic, the ego\'s motion, the traffic lights — and '
                      'the reconstruction gives the static scene: CARLA\'s actor pixels are laid over the '
                      'reconstruction rendered at each camera. Labels stay CARLA\'s. Shadows and lighting between '
                      'the two are not modelled. On a reconstruction of a CARLA run each frame is measured against '
                      'CARLA\'s own.')
        f = Form()
        self.hy_drive = QComboBox()
        self.hy_method = QComboBox()
        for k, text in (('auto', 'Automatic (CARLA\'s occlusion for a CARLA run of the same world)'),
                        ('mask', 'CARLA\'s occlusion (the actor mask)'), ('depth', 'Per-pixel depth test')):
            self.hy_method.addItem(text, k)
        self.hy_seed = QSpinBox()
        self.hy_seed.setRange(0, 2 ** 31 - 1)
        self.hy_seed.setValue(7)
        self.hy_vehicles = QSpinBox()
        self.hy_vehicles.setRange(0, 200)
        self.hy_vehicles.setValue(30)
        for text, w in (('Recorded drive', self.hy_drive), ('Compositing', self.hy_method),
                        ('New traffic: seed', self.hy_seed), ('Vehicles', self.hy_vehicles)):
            f.row(text, w)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Composite this drive', self.hybrid,
                           tip='falcon recon hybrid: the recorded drive\'s actors over the reconstruction (a GPU job)'))
        b.addWidget(button('Drive the path with new traffic', self.hybrid_drive,
                           tip='a CARLA replay of the log\'s path (the ego at the slider\'s offset) with the traffic '
                               'manager\'s traffic, then its composite'))
        b.addStretch(1)
        sec.add(b)
        self.hy_info = KeyValues(('Hybrid run', 'Against CARLA', 'Cost'))
        sec.add(self.hy_info)
        return sec

    def refresh_drives(self):
        """The workspace's recorded runs with instance labels (the actors' mask)."""
        cur = self.hy_drive.currentData()
        self.hy_drive.clear()
        svc = getattr(self.win, 'service', None)
        runs = svc.runs_dir() if svc is not None else ''
        for d in sorted(os.listdir(runs), reverse=True) if runs and os.path.isdir(runs) else []:
            p = os.path.join(runs, d)
            if os.path.isfile(os.path.join(p, 'frames.jsonl')) and os.path.isdir(os.path.join(p, 'instance')):
                self.hy_drive.addItem(d, p)
        if cur:
            self.hy_drive.setCurrentIndex(max(0, self.hy_drive.findData(cur)))

    def hybrid_argv(self, drive: str, out: str) -> list:
        return ['hybrid', self.rec_dir, '--drive', drive, '--out', out, '--method', self.hy_method.currentData(),
                '--device', self.device.currentData()]

    def hybrid(self, drive: str | None = None, out: str | None = None):
        if self.rec is None:
            self.note.setText('open a reconstruction first')
            return None
        from ...core.drive import run_dir_name
        drive = drive or self.hy_drive.currentData()
        if not drive:
            self.note.setText('no recorded drive with instance labels in this workspace')
            return None
        out = out or os.path.join(os.path.dirname(drive), run_dir_name(f'hybrid-{os.path.basename(drive)}'))
        job = self._submit('hybrid', self.hybrid_argv(drive, out), f'Hybrid {os.path.basename(drive)}')
        if job:
            self.hy_info.set('Hybrid run', f'compositing into {out}')
            self.hybrid_out = out
        return out

    def hybrid_drive(self, out: str | None = None):
        """A CARLA replay of the log's path with new traffic (a CARLA job); its composite follows (on_finished)."""
        if self.rec is None:
            self.note.setText('open a reconstruction first')
            return None
        from ...core.drive import run_dir_name
        from ...core.logreplay.commands import submit
        from ...core.recon.hybrid import HybridError, hybrid_drive_config
        lat = self.lateral.value()
        svc = self.win.service
        out = out or os.path.join(svc.runs_dir() or os.path.join(self.win.workspace(), 'runs'),
                                  run_dir_name(f'hybrid-drive-{lat:+.1f}m-seed{self.hy_seed.value()}'))
        try:
            cfg = hybrid_drive_config(self.win.project, self.rec, os.path.join(out, 'carla'), svc.port, svc.tm_port,
                                      lateral_m=lat, seed=self.hy_seed.value(), vehicles=self.hy_vehicles.value(),
                                      render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                                      workspace=self.win.workspace())
        except (HybridError, ValueError, KeyError, OSError) as e:
            self.note.setText(f'cannot drive it: {e}')
            return None
        job = submit(svc, cfg, cfg['name'])
        self.jobs['hybrid-carla'] = job
        self.hybrid_next = (os.path.join(out, 'carla'), os.path.join(out, 'hybrid'))
        self.hy_info.set('Hybrid run', f'recording CARLA\'s drive into {os.path.join(out, "carla")}')
        self.win.jobs_dock.raise_()
        return out

    def show_hybrid(self, out: str):
        from ...core.provenance import read_json
        p = os.path.join(out, 'run.json')
        if not os.path.isfile(p):
            return
        hy = read_json(p).get('hybrid') or {}
        self.hy_info.set('Hybrid run', f'{hy.get("frames")} frames, {hy.get("compositing", {}).get("method")}: {out}',
                         tip=out)
        a = (hy.get('summary') or {}).get('all') or {}
        if hy.get('measured'):
            bg, wh = a.get('background') or {}, a.get('whole') or {}
            lab = a.get('labels') or {}
            self.hy_info.set('Against CARLA', f'background {bg.get("psnr")} dB / SSIM {bg.get("ssim")}, whole frame '
                                              f'{wh.get("psnr")} dB; voted classes right on {lab.get("accuracy")} of '
                                              f'the background (the labels written are CARLA\'s)')
        else:
            self.hy_info.set('Against CARLA', hy.get('why_not_measured') or '—')
        self.hy_info.set('Cost', f'render {(a.get("render_ms") or {}).get("median")} ms, composite '
                                 f'{(a.get("composite_ms") or {}).get("median")} ms a frame (medians)')

    # ---------------------------------------------------------------- from a dashcam
    def _dashcam_section(self) -> Section:
        sec = Section('From a dashcam',
                      'A time window of a dashcam clip with embedded GPS: its frames posed by monocular visual '
                      'odometry (the GPS speed holds the scale, the GPS track places it), reconstructed with the '
                      'settings above and placed on an imported world. The result opens in the view above. A '
                      'consumer camera\'s footage — traffic, rolling shutter, exposure changes — reconstructs far '
                      'worse than a simulator\'s; dashcam.json in the folder has every number.')
        f = Form()
        row = QHBoxLayout()
        self.dash_clip = QLineEdit()
        self.dash_clip.setPlaceholderText('a dashcam .MP4 with embedded GPS')
        row.addWidget(self.dash_clip, 1)
        row.addWidget(button('Choose…', self.choose_clip))
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        self.dash_start = QDoubleSpinBox()
        self.dash_start.setRange(0.0, 3600.0)
        self.dash_start.setSuffix(' s')
        self.dash_seconds = QDoubleSpinBox()
        self.dash_seconds.setRange(3.0, 120.0)
        self.dash_seconds.setValue(20.0)
        self.dash_seconds.setSuffix(' s')
        self.dash_world = QComboBox()
        for text, wd in (('Clip', w), ('From', self.dash_start), ('Length', self.dash_seconds),
                         ('Place on the world', self.dash_world)):
            f.row(text, wd)
        sec.add(f)
        b = QHBoxLayout()
        self.dash_btn = button('Reconstruct the segment', self.from_dashcam, role='primary',
                               tip='falcon recon from-dashcam: a GPU job; the poses alone take about a minute')
        b.addWidget(self.dash_btn)
        b.addStretch(1)
        sec.add(b)
        self.refresh_worlds()
        return sec

    def refresh_worlds(self):
        self.dash_world.clear()
        self.dash_world.addItem('None', None)
        proj = getattr(self.win, 'project', None)
        for wd in (proj.doc.get('worlds') or []) if proj is not None else []:
            if wd.get('kind') == 'osm':
                self.dash_world.addItem(wd.get('name') or wd['id'], wd['id'])
        if self.dash_world.count() > 1:
            self.dash_world.setCurrentIndex(1)

    def choose_clip(self):
        start = os.path.dirname(self.dash_clip.text()) or os.path.expanduser('~/Videos/dashcam-videos')
        path, _ = QFileDialog.getOpenFileName(self, 'A dashcam clip', start, 'MP4 video (*.mp4 *.MP4)')
        if path:
            self.dash_clip.setText(path)

    def dashcam_argv(self, out: str) -> list:
        argv = ['from-dashcam', '--clip', self.dash_clip.text().strip(), '--start', f'{self.dash_start.value():g}',
                '--seconds', f'{self.dash_seconds.value():g}', '--out', out, '--iterations', str(self.iters.value()),
                '--width', str(self.width_px.value()), '--device', self.device.currentData(), '--seed',
                str(self.page.seed.value())]
        proj = getattr(self.win, 'project', None)
        wid = self.dash_world.currentData()
        if wid and proj is not None and proj.path:
            argv += ['--project', proj.path, '--world', wid]
        return argv

    def from_dashcam(self):
        clip = self.dash_clip.text().strip()
        if not clip or not os.path.isfile(clip):
            self.note.setText('choose a dashcam clip first')
            return None
        stem = os.path.splitext(os.path.basename(clip))[0]
        out = os.path.join(self.win.workspace(), 'recon', f'dashcam-{stem}-{self.dash_start.value():g}s')
        job = self._submit('dashcam', self.dashcam_argv(out), f'Dashcam {stem} from {self.dash_start.value():g} s')
        if job:
            self.dash_dir = out
            self.info.set('Source', f'{clip} from {self.dash_start.value():g} s for {self.dash_seconds.value():g} s')
            self.info.set('Reconstruction', f'posing and training: {out}')
        return job

    def show_poses(self):
        from ...core.provenance import read_json
        p = os.path.join(self.rec_dir, 'dashcam.json')
        if not os.path.isfile(p):
            self.info.set('Poses', '—')
            return
        d = read_json(p)
        a, g = d.get('alignment_vo_only') or {}, d.get('alignment_gps_aided') or {}
        m = d.get('map') or {}
        where = (f'on {m.get("name")}: {m.get("street") or "a road of the import"}' if m.get('on_map') else
                 f'{m.get("distance_from_centre_m")} m from the imported area\'s centre (outside it)'
                 if m.get('distance_from_centre_m') is not None else 'not placed on a world')
        self.info.set('Poses', f'odometry against the GPS: median {a.get("median_m")} m, RMS {a.get("rms_m")} m '
                               f'(GPS-aided: median {g.get("median_m")} m); f = '
                               f'{(d.get("intrinsics") or {}).get("native", {}).get("f")} px at 3840 wide; {where}',
                      tip=p)

    # ---------------------------------------------------------------- the log and its reconstruction
    def set_log(self, log, path: str):
        self.log = log
        self.source, why = source_of(log)
        name = os.path.basename(os.path.dirname(path)) if path else 'log'
        self.rec_dir = os.path.join(self.win.workspace(), 'recon', name)
        self.info.set('Source', self.source or why)
        set_enabled(self.train_btn, self.source is not None, f'Cannot reconstruct this log: {why}')
        self.rec = None
        self.view.clear()
        if os.path.isfile(os.path.join(self.rec_dir, 'point_cloud.ply')):
            self.open_recon()
        else:
            self.info.set('Reconstruction', 'none yet')
            for k in ('Poses', 'Held out', 'Novel views', 'Cost'):
                self.info.set(k, '—')
        self.refresh_worlds()
        self.refresh_drives()

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
                self.device.currentData(), '--seed', str(self.page.seed.value()),
                '--raster-backend', self.backend.currentData()]
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
        if job.state == DONE and which == 'dashcam' and self.dash_dir:
            self.rec_dir = self.dash_dir
        if which == 'hybrid-carla' and self.hybrid_next:
            carla, out = self.hybrid_next
            self.hybrid_next = None
            if job.state == DONE:
                self.hybrid(carla, out)
            else:
                self.hy_info.set('Hybrid run', f'CARLA\'s drive did not finish ({job.state})')
        if job.state == DONE and which == 'hybrid':
            self.show_hybrid(self.hybrid_out)
            self.refresh_drives()
        if job.state == DONE and which in ('train', 'eval', 'dashcam'):
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
        self.show_poses()
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
        elif os.path.isfile(os.path.join(self.rec_dir, 'dashcam.json')):
            self.info.set('Novel views', f'rendered sideways and raised / lowered (no reference exists there): '
                                         f'{os.path.join(self.rec_dir, "renders")}')

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
