"""The Replay page's Near-misses section: a folder of your dashcam clips mined for near-misses (core.dashcam_ingest,
core.nearmiss; docs/nearmiss.md).

Choose the folder, the detector and depth model (their licence and whether they are downloaded are shown; downloading
is `falcon dashcam nearmiss fetch NAME --accept-licence`, never from here by itself), the camera's height; Ingest queues
the resumable job (a GPU job unless on the CPU) and its progress shows here and in Jobs. The ranked list shows each
event's kind, score with its interval, minimum TTC, PET, required deceleration and the ego's braking; selecting one
loads its clip into the player (play, pause, scrub). Make scenario turns the selected event into a Falcon scenario,
variations and OpenSCENARIO on the chosen imported world (a job); Queue variations queues the rare-case search over
them on this machine; Reconstruct with moving road users reconstructs the event's stretch of the clip in 3D with its
tracked road users as moving actors (core.recon.dashcam_tracks; a GPU job) and opens it in the page's Reconstruction
section, whose Drive a policy here runs a policy in closed loop through the incident. Everything here is the user's
footage: it stays local (the folder, the reconstruction and its drives are private to the assistant)."""
from __future__ import annotations

import os
import re
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QSlider,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import dashcam_ingest as I
from ..core import nearmiss as NM
from ..core import perception_models as PM
from ..core.jobs import FINAL, JobSpec
from .widgets import Form, KeyValues, button, label, set_enabled

COLUMNS = ('Rank', 'Kind', 'Score [5–95 %]', 'Min TTC', 'PET', 'DRAC', 'Ego braking')


def _s(v, unit='', nd=2):
    return '—' if v is None else f'{v:.{nd}f}{unit}'


class ClipPlayer(QWidget):
    """A small player for an event's clip: frames read on demand with OpenCV; play, pause, scrub."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.view = QLabel('No clip')
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumHeight(180)
        v.addWidget(self.view)
        row = QHBoxLayout()
        self.play_btn = button('Play', self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        row.addWidget(self.play_btn)
        row.addWidget(self.slider, 1)
        self.time = label('', muted=True)
        row.addWidget(self.time)
        v.addLayout(row)
        self.cap = None
        self.fps = 30.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)
        set_enabled(self.play_btn, False, 'choose an event with a clip')

    def load(self, path: str | None) -> bool:
        import cv2
        self.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if not path or not os.path.isfile(path):
            self.view.setText('No clip (exported with --no-clips, or not in the top events)')
            set_enabled(self.play_btn, False, 'no clip')
            return False
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not cap.isOpened() or n <= 0:
            self.view.setText('The clip cannot be read')
            return False
        self.cap, self.fps = cap, float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        set_enabled(self.play_btn, True)
        self.show_frame(0)
        return True

    def show_frame(self, k: int) -> bool:
        import cv2
        if self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, bgr = self.cap.read()
        if not ok:
            return False
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pm = QPixmap.fromImage(img)
        self.view.setPixmap(pm.scaledToWidth(min(640, max(320, self.view.width())),
                                             Qt.TransformationMode.SmoothTransformation))
        self.time.setText(f'{k / self.fps:5.1f} s')
        return True

    def toggle(self):
        if self.timer.isActive():
            self.stop()
        elif self.cap is not None:
            self.timer.start(int(1000 / max(self.fps, 1.0)))
            self.play_btn.setText('Pause')

    def stop(self):
        self.timer.stop()
        self.play_btn.setText('Play')

    def step(self):
        k = self.slider.value() + 1
        if k > self.slider.maximum():
            self.stop()
            return
        self.slider.setValue(k)


class NearMissTab(QWidget):
    """Controls, the ranked list, the player and the scenario actions (module doc)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.jobs = {}
        self.events = []
        self.incident = ''
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(label('Your dashcam clips mined for moments another road user came critically close: detected, '
                          'tracked, ranged and ranked by time to collision, PET and the braking they called for. It '
                          'all stays on this machine.', muted=True, wrap=True))
        row = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText('a folder of dashcam clips (MP4 with embedded GPS)')
        self.folder.editingFinished.connect(self.refresh)
        row.addWidget(self.folder, 1)
        row.addWidget(button('Choose…', self.choose))
        v.addLayout(row)
        f = Form()
        self.detector = QComboBox()
        self.depth = QComboBox()
        for n in ('rtdetr-v2-r18', 'rtdetr-v2-r50'):
            self.detector.addItem(n, n)
        for n, text in (('da2-small', 'da2-small'), ('metric3d-small', 'metric3d-small (opt-in)'),
                        ('none', 'none (flat-ground ranges only)')):
            self.depth.addItem(text, n)
        self.height_m = QDoubleSpinBox()
        self.height_m.setRange(0.3, 4.0)
        self.height_m.setSingleStep(0.05)
        self.height_m.setValue(I.DEFAULTS['camera_height_m'])
        self.height_m.setSuffix(' m')
        self.device = QComboBox()
        for d in ('auto', 'cuda', 'cpu'):
            self.device.addItem(d, d)
        for text, w in (('Detector', self.detector), ('Depth', self.depth), ('Camera height', self.height_m),
                        ('Device', self.device)):
            f.row(text, w)
        v.addLayout(f)
        self.models = label('', muted=True, wrap=True)
        v.addWidget(self.models)
        b = QHBoxLayout()
        self.ingest_btn = button('Ingest', self.ingest, role='primary',
                                 tip='falcon dashcam ingest: both passes, resumable; a GPU job unless on the CPU')
        b.addWidget(self.ingest_btn)
        b.addWidget(button('Refresh', self.refresh))
        b.addStretch(1)
        v.addLayout(b)
        self.status = label('', muted=True, wrap=True)
        v.addWidget(self.status)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(140)
        self.tree.itemSelectionChanged.connect(self.select)
        v.addWidget(self.tree)
        self.player = ClipPlayer()
        v.addWidget(self.player)
        self.info = KeyValues()
        v.addWidget(self.info)
        s = QHBoxLayout()
        self.world = QComboBox()
        self.scenario_btn = button('Make scenario', self.make_scenario,
                                   tip='the event as a Falcon scenario, variations and OpenSCENARIO on the chosen '
                                       'imported world (core.incident)')
        self.vary_btn = button('Queue variations', self.queue_variations,
                               tip='the rare-case search over the scenario\'s variations, on this machine')
        s.addWidget(label('World', muted=True))
        s.addWidget(self.world, 1)
        s.addWidget(self.scenario_btn)
        s.addWidget(self.vary_btn)
        v.addLayout(s)
        r = QHBoxLayout()
        self.recon_btn = button('Reconstruct with moving road users', self.reconstruct,
                                tip='the event\'s stretch of the clip reconstructed in 3D with its tracked road users '
                                    'as moving actors (falcon dashcam nearmiss reconstruct; a GPU job, on this '
                                    'machine); it opens in Reconstruction, where Drive a policy here runs closed loop '
                                    'through the incident')
        r.addWidget(self.recon_btn)
        r.addStretch(1)
        v.addLayout(r)
        self.recon_out = ''
        set_enabled(self.scenario_btn, False, 'choose an event first')
        set_enabled(self.vary_btn, False, 'make a scenario first')
        set_enabled(self.recon_btn, False, 'choose an event first')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.refresh_worlds()
        self.show_models()

    # ---------------------------------------------------------------- state
    def show_models(self):
        rows = {r['name']: r for r in PM.list_models()}
        parts = []
        for n in (self.detector.currentData(), self.depth.currentData()):
            r = rows.get(n)
            if r:
                parts.append(f'{n}: {r["licence"]}, {r["state"]}' + (' — fetch it: falcon dashcam nearmiss fetch '
                                                                        f'{n} --accept-licence' if r['state'] ==
                                                                       'not downloaded' else ''))
        parts.append('YOLO (AGPL-3.0) and Depth Anything V2 Base/Large (CC-BY-NC) are refused')
        self.models.setText('; '.join(parts))

    def refresh_worlds(self):
        self.world.clear()
        proj = getattr(self.win, 'project', None)
        for wd in (proj.doc.get('worlds') or []) if proj is not None else []:
            if wd.get('kind') == 'osm':
                self.world.addItem(wd.get('name') or wd['id'], wd['id'])

    def choose(self):
        d = QFileDialog.getExistingDirectory(self, 'A folder of dashcam clips', self.folder.text() or
                                             os.path.expanduser('~'))
        if d:
            self.folder.setText(d)
            self.refresh()

    def refresh(self):
        folder = self.folder.text().strip()
        self.tree.clear()
        self.events = I.ranking(folder) if folder and os.path.isdir(folder) else []
        for e in self.events:
            it = QTreeWidgetItem([str(e['rank']), e['kind'] + (' (heuristic)' if e.get('class_source') ==
                                                                 'heuristic' else ''),
                                  f'{e["score_median"]:.3f} [{e["score_ci"][0]:.3f}, {e["score_ci"][1]:.3f}]',
                                  _s(e['min_ttc_s'], ' s'), _s(e['pet_s'], ' s'), _s(e['max_drac_mps2'], ' m/s²', 1),
                                  _s(e['ego_peak_decel_mps2'], ' m/s²', 1)])
            it.setToolTip(0, e['id'])
            self.tree.addTopLevelItem(it)
        st = I.summary(folder) if folder and os.path.isdir(I.out_dir(folder)) else None
        if st:
            c = st['counts']
            self.status.setText(f'{st["status"]}: {c.get("clips", 0)} clips ({c.get("no_gps", 0)} without GPS, '
                                f'{c.get("rear_skipped", 0)} rear skipped), {st["events"]} events ranked')
        elif folder:
            self.status.setText('not ingested yet')
        return len(self.events)

    def selected(self) -> dict | None:
        it = self.tree.currentItem()
        if it is None:
            return None
        i = self.tree.indexOfTopLevelItem(it)
        return self.events[i] if 0 <= i < len(self.events) else None

    def select(self):
        e = self.selected()
        if e is None:
            return
        folder = self.folder.text().strip()
        try:
            d = I.event_dir(folder, e['id'])
        except I.IngestError:
            d = None
        self.player.load(os.path.join(d, 'clip.mp4') if d else None)
        self.info.set('Event', f'{e["id"]}: {e["kind"]}, window {e["window_s"][0]:.1f}–{e["window_s"][1]:.1f} s, '
                               f'{e["observations"]} observations; ' + ', '.join(e.get('why') or ()))
        vo = e.get('vo') or {}
        self.info.set('Heading', 'odometry' if vo.get('used') else f'GPS ({vo.get("why", "no odometry")})')
        set_enabled(self.scenario_btn, d is not None and self.world.count() > 0,
                    'not an exported event' if d is None else 'import a world where the footage drove first')
        set_enabled(self.recon_btn, d is not None, 'not an exported event (only the top events are exported)')

    # ---------------------------------------------------------------- jobs
    def ingest_argv(self) -> list:
        return ['--detector', self.detector.currentData(), '--depth', self.depth.currentData(), '--camera-height',
                f'{self.height_m.value():g}']

    def ingest(self):
        folder = self.folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self.status.setText('choose a folder of dashcam clips first')
            return None
        spec = I.job_spec(folder, self.ingest_argv(), self.device.currentData())
        job = self.win.queue.submit(spec)
        self.jobs['ingest'] = job
        self.timer.start(1000)
        self.win.jobs_dock.raise_()
        return job

    def poll(self):
        busy = False
        for key, job in list(self.jobs.items()):
            if job.state in FINAL:
                if key == 'ingest':
                    self.status.setText(f'ingest {job.state}')
                    self.refresh()
                elif key == 'scenario' and job.state == 'done':
                    self.info.set('Scenario', f'{self.incident}: in the project\'s incidents/ (Queue variations to '
                                              'search around it)')
                    set_enabled(self.vary_btn, True)
                elif key == 'recon':
                    self.reconstructed(job)
                del self.jobs[key]
            else:
                busy = True
                if key == 'ingest':
                    self.status.setText(f'ingesting: {job.summary()}')
        if not busy:
            self.timer.stop()

    def make_scenario(self):
        e = self.selected()
        w = self.win
        if e is None or not self.world.currentData():
            return None
        if not w.project.path or w.project.dirty:
            if not w.save():
                return None
        name = re.sub(r'[^A-Za-z0-9_.-]', '-', f'nearmiss-{e["id"]}')
        argv = [sys.executable, '-m', 'falcon_autonomy', 'dashcam', 'nearmiss', 'scenario', self.folder.text().strip(),
                e['id'], '--project', w.project.path, '--world', self.world.currentData(), '--name', name]
        job = w.queue.submit(JobSpec('incident', f'Scenario from near-miss {e["rank"]}', argv, memory_gb=0.5,
                                     meta={'source': {'format': 'dashcam'}}))
        self.jobs['scenario'] = job
        self.incident = name
        self.timer.start(1000)
        return job

    def reconstruct(self):
        """The selected event reconstructed with its tracked road users as moving actors (a GPU job, local)."""
        e = self.selected()
        folder = self.folder.text().strip()
        if e is None or not folder:
            return None
        busy = self.jobs.get('recon')
        if busy is not None and busy.state not in FINAL:
            return None
        w = self.win
        out = os.path.join(w.workspace(), 'recon', f'nearmiss-{e["id"]}')
        job = w.queue.submit(I.reconstruct_spec(folder, e['id'], out, self.device.currentData(),
                                                title=f'Reconstruct near-miss {e["rank"]} with its road users'))
        self.jobs['recon'] = job
        self.recon_out = out
        self.info.set('Reconstruction', f'posing the stretch and learning its road users: {out}')
        self.timer.start(1000)
        w.jobs_dock.raise_()
        return job

    def reconstructed(self, job):
        """A finished reconstruction opens in Replay ▸ Reconstruction (its Drive a policy here runs closed loop)."""
        if job.state != 'done':
            self.info.set('Reconstruction', f'{job.state}: {job.summary()}')
            return False
        self.info.set('Reconstruction', f'{self.recon_out} — open in Reconstruction below; Drive a policy here runs '
                                        'closed loop through the incident')
        panel = getattr(self.page, 'recon', None)
        if panel is None or not hasattr(panel, 'open_recon'):
            return False
        panel.rec_dir = self.recon_out
        panel.open_recon()
        return True

    def queue_variations(self):
        w = self.win
        if not self.incident:
            return None
        job = w.queue.submit(NM.variations_spec(w.project.path, w.workspace(), self.incident, 40,
                                                fake_carla=getattr(w, 'fake_carla', '') or ''))
        self.jobs['variations'] = job
        w.jobs_dock.raise_()
        return job
