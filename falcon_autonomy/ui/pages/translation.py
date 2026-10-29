"""Training ▸ Sim-to-real translation: train a translator (core.translate: CUT, the frame's semantic label as a
conditioning input) from checked drives to a real set, translate checked drives into new runs with every label kept,
look at a translated run frame by frame beside its source with each frame's measured drift, and measure the gap
before and after. Every action is a `falcon translate …` job (needs_gpu: never at full load beside CARLA). The data
terms a translator inherits from its real images are shown with it. A dashcam source (core.dashcam) reads the user's
own clips' GPS, draws the track beside the project's imported areas and extracts a daytime, moving image set (a
`falcon dashcam extract` job) that becomes the real images."""
from __future__ import annotations

import json
import os
import time

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QMessageBox, QPlainTextEdit, QSlider, QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ...core import dashcam as DC
from ...core import translate as TR
from ...core.jobs import DONE, JobSpec
from ...core.paths import tilde
from ...core.provenance import read_json
from .. import tasks
from ..osmimport import _Canvas
from ..theme import CAMERA, THEME
from ..widgets import Form, KeyValues, Section, button, label, scroll


def _jsonl(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]
    except (OSError, ValueError):
        return []


def _f(v, nd=3):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


class BeforeAfter(QWidget):
    """A translated run's frames beside their source's, with a slider and each frame's drift."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.before, self.after = QLabel('Source'), QLabel('Translated')
        for w in (self.before, self.after):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            w.setMinimumSize(240, 140)
            row.addWidget(w, 1)
        v.addLayout(row, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        v.addWidget(self.slider)
        self.caption = label('', muted=True, wrap=True)
        v.addWidget(self.caption)
        self.rows, self.run, self.source = [], '', ''

    def load(self, run_dir: str):
        self.run, self.rows, self.source = run_dir or '', [], ''
        if run_dir:
            try:
                info = read_json(os.path.join(run_dir, 'run.json')).get('translation') or {}
            except (OSError, ValueError):
                info = {}
            self.source = info.get('source') or ''
            self.rows = _jsonl(os.path.join(run_dir, info.get('frames') or 'translation.jsonl'))
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(self.rows) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.show_frame(0)

    def show_frame(self, k=0):
        if not self.rows:
            for w, t in ((self.before, 'Source'), (self.after, 'Translated')):
                w.setPixmap(QPixmap())
                w.setText(t)
            self.caption.setText('Choose a translated run' if not self.run else 'This run lists no translated frames')
            return
        r = self.rows[min(k, len(self.rows) - 1)]
        for w, base in ((self.before, self.source), (self.after, self.run)):
            p = os.path.join(base, r['file'])
            pm = QPixmap(p) if os.path.isfile(p) else QPixmap()
            if pm.isNull():
                w.setPixmap(QPixmap())
                w.setText(f'missing: {tilde(p)}')
            else:
                w.setPixmap(pm.scaled(w.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        state = ('over the threshold: the raw frame was kept' if r.get('used') == 'raw' else
                 'over the threshold (flagged, translation kept)' if r.get('flagged') else 'translated')
        self.caption.setText(f'{r["camera"]} step {r["step"]}: drift {_f(r.get("drift"))} — structure '
                             f'{_f(r.get("structure"))}, edge recall {_f(r.get("edge_recall"))}, edge precision '
                             f'{_f(r.get("edge_precision"))}, worst object box {_f(r.get("worst_box"))}; {state}')


class TrackMap(_Canvas):
    """Dashcam GPS tracks in local metres (north up) with the project's imported areas as boxes."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 160)
        self.tracks, self.boxes = [], []          # [[(x, y)]], [(x0, y0, x1, y1, name)]

    def set_data(self, tracks: list, worlds: list):
        from ...core.osm import local_projection
        pts = [(f['lat'], f['lon']) for t in tracks for f in t['fixes']]
        geo = [b for w in worlds for b in [w.get('bbox')] if b]
        ref = pts or [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in geo]
        self.tracks, self.boxes, self.lines = [], [], []
        if not ref:
            self.bounds = None
            self.update()
            return
        pr = local_projection(sum(p[0] for p in ref) / len(ref), sum(p[1] for p in ref) / len(ref))
        xs, ys = [], []
        for t in tracks:
            xy = [pr.to_local(f['lat'], f['lon']) for f in t['fixes']]
            if xy:
                self.tracks.append(xy)
                xs += [p[0] for p in xy]
                ys += [p[1] for p in xy]
        for w in worlds:
            b = w.get('bbox')
            if b:
                x0, y0 = pr.to_local(b[0], b[1])
                x1, y1 = pr.to_local(b[2], b[3])
                self.boxes.append((x0, y0, x1, y1, w.get('name') or w.get('id')))
                xs += [x0, x1]
                ys += [y0, y1]
        self.bounds = (min(xs), min(ys), max(xs), max(ys)) if xs else None
        self.update()

    def paint_extra(self, p):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = QColor(CAMERA['dark'][0])
        p.setPen(QPen(col, 2))
        for xy in self.tracks:
            pts = [self.to_px(x, y) for x, y in xy]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        p.setPen(QPen(QColor(CAMERA['dark'][1]), 2))
        for x0, y0, x1, y1, name in self.boxes:
            a, b = self.to_px(x0, y1), self.to_px(x1, y0)
            r = QRectF(a, b)
            if r.width() < 6:                                  # far smaller than the track: keep it visible
                r = QRectF(r.center() - QPointF(3, 3), r.center() + QPointF(3, 3))
            p.drawRect(r)
            p.setPen(QPen(QColor(THEME.tok['ink2']), 1))
            tw = p.fontMetrics().horizontalAdvance(str(name))
            at = r.bottomRight() + QPointF(4, 0)
            if at.x() + tw > self.width() - 4:                 # no room on the right: write it on the left
                at = r.bottomLeft() - QPointF(tw + 4, 0)
            p.drawText(at, str(name))
            p.setPen(QPen(QColor(CAMERA['dark'][1]), 2))


class DashcamSource(QWidget):
    """The user's dashcam as the real images: read the clips' GPS, see it beside the imported areas, extract a
    daytime, moving image set (a job), use it."""

    def __init__(self, tab):
        super().__init__()
        self.tab = tab
        self.tracks = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        f = Form()
        self.root = QLineEdit(DC.DEFAULT_ROOT)
        self.name = QLineEdit('dashcam-day')
        self.max_frames = QSpinBox()
        self.max_frames.setRange(10, 100000)
        self.max_frames.setValue(4000)
        self.day = QCheckBox('Daytime and moving frames only (sun ≥ 5°, ≥ 2 km/h)')
        self.day.setChecked(True)
        self.crop = QCheckBox('Cut the camera\'s burned-in text line (bottom 6 %)')
        self.crop.setChecked(True)
        f.row('Clips', self.root)
        f.row('Image set', self.name)
        f.row('Frames', self.max_frames)
        f.row('', self.day)
        f.row('', self.crop)
        v.addLayout(f)
        self.map = TrackMap()
        v.addWidget(self.map)
        self.summary = label('Read the clips\' GPS to see where they were recorded.', muted=True, wrap=True)
        v.addWidget(self.summary)
        b = QHBoxLayout()
        b.addWidget(button('Read GPS', self.read))
        b.addWidget(button('Extract frames', self.extract))
        b.addStretch(1)
        v.addLayout(b)

    def worlds(self) -> list:
        proj = self.tab.page.win.project
        return [{'id': w['id'], 'name': w.get('name'), 'bbox': (w.get('osm') or {}).get('bbox')}
                for w in proj.doc.get('worlds', ()) if (w.get('osm') or {}).get('bbox')]

    def read(self):
        root = self.root.text().strip()
        worlds = self.worlds()
        self.summary.setText('Reading the clips\' GPS…')

        def work():
            clips = DC.list_clips(root)
            return clips, [DC.track(c['path']) for c in clips]

        def done(res):
            clips, tracks = res
            self.tracks = tracks
            self.map.set_data(tracks, worlds)
            fixes = sum(t['n_fixes'] for t in tracks)
            km = sum(DC.clip_summary(c, t)['distance_m'] for c, t in zip(clips, tracks)) / 1000
            near = []
            for w in worlds:
                s_, w_, n_, e_ = w['bbox']
                inside = sum(1 for t in tracks if any(s_ <= f['lat'] <= n_ and w_ <= f['lon'] <= e_
                                                      for f in t['fixes']))
                d = min((DC.distance_m(f['lat'], f['lon'], min(max(f['lat'], s_), n_), min(max(f['lon'], w_), e_))
                         for t in tracks for f in t['fixes']), default=None)
                near.append(f'{inside} drive inside {w["name"] or w["id"]}'
                            + (f' (nearest fix {d / 1000:.1f} km from it)' if d else ''))
            self.summary.setText(f'{len(clips)} clips, {fixes} GPS fixes, {km:.1f} km. '
                                 + ('; '.join(near) + '. ' if near else 'No imported area in this project. ')
                                 + 'Which streets match exactly: falcon dashcam match.')
        tasks.run(self, work, done, lambda e: self.summary.setText(f'could not read the clips: {e}'))

    def extract(self):
        import sys
        out = DC.default_set_dir(self.name.text().strip() or time.strftime('%Y%m%d-%H%M%S-dashcam'))
        argv = ['dashcam', 'extract', '--root', self.root.text().strip(), '--out', out, '--max-frames',
                str(self.max_frames.value())]
        if self.day.isChecked():
            argv += ['--day', '--min-sun', '5']
        else:
            argv += ['--min-speed', '-1']
        if self.crop.isChecked():
            argv += ['--crop-bottom', '0.06']
        spec = JobSpec('dashcam', f'Extract dashcam frames to {os.path.basename(out)}',
                       [sys.executable, '-m', 'falcon_autonomy'] + argv, memory_gb=2.0, run_dir=out)
        self.tab.submit(spec)

    def use(self, folder: str):
        self.tab.real.setText(f'dashcam:{folder}')


class TranslationTab(QWidget):
    """The tab; `page` is the Training page (its workspace, job submission and drive lists)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.jobs = {}
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        sec = Section('Train a translator', 'CUT (contrastive unpaired translation, Park et al. 2020), Falcon\'s own '
                                            'implementation: the checked drives\' frames are made to look like the '
                                            'real images, with each frame\'s semantic label as an input and no '
                                            'layer that can move pixels. Trained here, on this GPU; nothing is '
                                            'downloaded.')
        f = Form()
        self.tr_id = QLineEdit('kitti-cut')
        self.real = QLineEdit()
        self.real.setPlaceholderText('KIND:PATH, e.g. kitti_raw:/data/kitti/2011_09_26 (read in place)')
        self.iters = QSpinBox()
        self.iters.setRange(10, 500000)
        self.iters.setValue(TR.DEFAULTS['iterations'])
        self.crop = QSpinBox()
        self.crop.setRange(32, 1024)
        self.crop.setValue(TR.DEFAULTS['crop'])
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.cond = QCheckBox('Give the generator the semantic label')
        self.cond.setChecked(True)
        for text, w in (('Translator', self.tr_id), ('Real images', self.real), ('Iterations', self.iters),
                        ('Crop', self.crop), ('Seed', self.seed), ('', self.cond)):
            f.row(text, w)
        sec.add(f)
        sec.add(label('A translator inherits the terms of the real images it learns from — KITTI\'s are CC BY-NC-SA '
                      '3.0 (non-commercial) — and so does every image it makes.', muted=True, wrap=True))
        sec.add(label('Or your own dashcam', h=True))
        self.dashcam = DashcamSource(self)
        sec.add(self.dashcam)
        sec.add(label('Simulated drives', h=True))
        self.train_runs = QListWidget()
        self.train_runs.setMinimumHeight(100)
        sec.add(self.train_runs)
        b = QHBoxLayout()
        b.addWidget(button('Train', self.train, role='primary'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        sec = Section('Translate drives', 'A new run per drive: the rgb translated, every label (semantic, '
                                          'instance, range, boxes, lidar, …) the source\'s. A frame whose measured '
                                          'drift exceeds the threshold (a heuristic) keeps its raw image, or is '
                                          'kept and flagged.')
        f = Form()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(TR.DRIFT_THRESHOLD)
        self.on_drift = QComboBox()
        self.on_drift.addItem('keep the raw frame', 'raw')
        self.on_drift.addItem('keep the translation, flagged', 'flag')
        f.row('Drift threshold', self.threshold)
        f.row('Over the threshold', self.on_drift)
        sec.add(f)
        sec.add(label('Drives to translate (with the translator chosen on the right)', h=True))
        self.apply_runs = QListWidget()
        self.apply_runs.setMinimumHeight(100)
        sec.add(self.apply_runs)
        b = QHBoxLayout()
        b.addWidget(button('Translate the checked drives', self.apply, role='primary'))
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
        self.info = KeyValues(('Method', 'Data', 'Terms', 'Cost', 'Probe drift', 'Status'))
        rv.addWidget(self.info)
        self.curve = QTreeWidget()
        self.curve.setHeaderLabels(['Iteration', 'Loss G', 'Loss D', 'NCE', 'Probe drift', 'Images/s'])
        self.curve.setRootIsDecorated(False)
        self.curve.setMaximumHeight(150)
        rv.addWidget(self.curve)
        rv.addWidget(label('Translated runs', h=True))
        self.translated = QComboBox()
        self.translated.currentIndexChanged.connect(lambda _: self.show_translated())
        rv.addWidget(self.translated)
        self.viewer = BeforeAfter()
        rv.addWidget(self.viewer, 1)
        f = Form()
        self.gap_real = QLineEdit()
        self.gap_real.setPlaceholderText('KIND:PATH of a real set (best: images the translator did not train on)')
        self.gap_features = QCheckBox('Fréchet and kernel distances (falcon gap --fetch first)')
        f.row('Real set', self.gap_real)
        f.row('', self.gap_features)
        rv.addLayout(f)
        b = QHBoxLayout()
        b.addWidget(button('Measure drift and the gap before and after', self.check))
        b.addStretch(1)
        rv.addLayout(b)
        self.check_text = QPlainTextEdit()
        self.check_text.setReadOnly(True)
        self.check_text.setObjectName('mono')
        self.check_text.setMaximumHeight(170)
        rv.addWidget(self.check_text)
        split.addWidget(right)
        split.setSizes([460, 800])

    # ------------------------------------------------------------------ data
    def ws(self, *parts):
        return self.page.ws(*parts)

    def refresh(self):
        for w in (self.train_runs, self.apply_runs):
            self.page._fill_runs(w)
        self.page._combo_dirs(self.models, self.ws('translators'), 'translator.json')
        cur = self.translated.currentData()
        self.translated.blockSignals(True)
        self.translated.clear()
        for d in self.translated_runs():
            self.translated.addItem(os.path.basename(d), d)
        k = self.translated.findData(cur)
        self.translated.setCurrentIndex(k if k >= 0 else 0)
        self.translated.blockSignals(False)
        self.show_model()
        self.show_translated()

    def translated_runs(self) -> list:
        root = self.ws('runs')
        out = []
        if os.path.isdir(root):
            for d in sorted(os.listdir(root), reverse=True):
                p = os.path.join(root, d, 'run.json')
                if os.path.isfile(p):
                    try:
                        if read_json(p).get('worker') == 'translate':
                            out.append(os.path.join(root, d))
                    except (OSError, ValueError):
                        pass
        return out

    def show_model(self):
        self.curve.clear()
        d = self.models.currentData()
        if not d:
            self.info.set('Status', 'No translator yet: train one on the left.')
            return
        try:
            doc = read_json(os.path.join(d, 'translator.json'))
        except (OSError, ValueError) as e:
            self.info.set('Status', f'translator.json could not be read: {e}')
            return
        data, cost, pr = doc.get('data') or {}, doc.get('cost') or {}, doc.get('probe') or {}
        self.info.set('Method', f'{doc.get("method", "?").upper()} — {doc.get("paper", "")}')
        self.info.set('Data', f'{data.get("sim_images")} simulated frames ({", ".join(data.get("sim_runs") or [])}) '
                              f'→ {data.get("real_images")} real images ('
                              f'{", ".join(tilde(r["path"]) for r in data.get("real") or [])})')
        self.info.set('Terms', doc.get('licence_note') or '; '.join(doc.get('terms') or []))
        peak = cost.get('peak_gpu_mib')
        self.info.set('Cost', f'{_f(cost.get("train_s"), 0)} s, {_f(cost.get("images_per_s_median"), 1)} images/s'
                              f'{f", peak GPU memory {peak:.0f} MiB" if isinstance(peak, (int, float)) else ""} '
                              f'({(cost.get("device") or {}).get("type", "?")})')
        self.info.set('Probe drift', f'median {_f(pr.get("drift_median"))}, max {_f(pr.get("drift_max"))} on '
                                     f'{len(doc.get("previews") or [])}+ fixed frames (threshold '
                                     f'{doc.get("drift_threshold")}, heuristic)' if pr else '—')
        self.info.set('Status', doc.get('status', '?'))
        for r in _jsonl(os.path.join(d, 'metrics.jsonl')):
            p = r.get('probe') or {}
            self.curve.addTopLevelItem(QTreeWidgetItem([str(r.get('it')), _f(r.get('loss_G')), _f(r.get('loss_D')),
                                                        _f(r.get('nce')), _f(p.get('drift_median')),
                                                        _f(r.get('images_per_s'), 1)]))

    def show_translated(self):
        d = self.translated.currentData()
        self.viewer.load(d or '')
        p = os.path.join(self.check_dir(d), 'check.json') if d else ''
        if p and os.path.isfile(p):
            try:
                self.check_text.setPlainText(TR.check_text(read_json(p)))
                return
            except (OSError, ValueError, KeyError, TypeError) as e:
                self.check_text.setPlainText(f'check.json could not be read: {e}')
                return
        if d:
            try:
                tr = read_json(os.path.join(d, 'run.json')).get('translation') or {}
                dr = tr.get('drift') or {}
                self.check_text.setPlainText(
                    f'{dr.get("frames")} frames translated by {tr.get("model_id")}: drift median '
                    f'{_f((dr.get("drift") or {}).get("median"))}, {dr.get("flagged")} over {dr.get("threshold")} '
                    f'({dr.get("refused")} kept raw).\nTerms: {tr.get("licence_note")}\nNo gap measured yet.')
            except (OSError, ValueError):
                self.check_text.setPlainText('')
        else:
            self.check_text.setPlainText('')

    def check_dir(self, run_dir: str) -> str:
        return self.ws('translation-checks', os.path.basename(os.path.normpath(run_dir)))

    # ------------------------------------------------------------------ actions
    def train(self):
        runs = self.page._checked(self.train_runs)
        real = self.real.text().strip()
        if not runs or not real:
            QMessageBox.information(self, 'Train a translator', 'Check the simulated drives and give the real images '
                                                                '(KIND:PATH).')
            return
        if not self.page._ready():
            return
        tid = self.tr_id.text().strip() or time.strftime('%Y%m%d-%H%M%S-cut')
        argv = ['train', '--workspace', self.page.win.workspace(), '--id', tid, '--real', real, '--iterations',
                str(self.iters.value()), '--crop', str(self.crop.value()), '--seed', str(self.seed.value())]
        for r in runs:
            argv += ['--sim', r]
        if not self.cond.isChecked():
            argv.append('--no-condition')
        self.submit(TR.job_spec(argv, f'Train translator {tid}'))

    def apply(self):
        model = self.models.currentData()
        runs = self.page._checked(self.apply_runs)
        if not model or not runs:
            QMessageBox.information(self, 'Translate', 'Choose a translator and check the drives to translate.')
            return
        if not self.page._ready():
            return
        root = self.ws('runs')
        for r in runs:
            out = os.path.join(root, TR.default_name(r, model))
            argv = ['apply', '--model', model, '--run', r, '--out', out, '--threshold', f'{self.threshold.value():g}',
                    '--on-drift', self.on_drift.currentData()]
            self.submit(TR.job_spec(argv, f'Translate {os.path.basename(r)}', memory_gb=3.0, run_dir=out))

    def check(self):
        d = self.translated.currentData()
        if not d:
            QMessageBox.information(self, 'Measure', 'Choose a translated run.')
            return
        argv = ['check', d, '--out', self.check_dir(d)]
        if self.gap_real.text().strip():
            argv += ['--real', self.gap_real.text().strip()]
            if self.gap_features.isChecked():
                argv += ['--features', '--workspace', self.page.win.workspace()]
        self.submit(TR.job_spec(argv, f'Check {os.path.basename(d)}', memory_gb=3.0))

    def submit(self, spec):
        job = self.page._submit(spec)
        self.jobs[job.id] = job
        return job

    def finished(self, job):
        if job.id not in self.jobs:
            return False
        self.jobs.pop(job.id)
        if job.spec.kind == 'dashcam':
            if job.state == DONE:
                self.dashcam.use(job.spec.run_dir)
            return True
        cat = self.page.win.service.catalog
        if cat is not None and job.spec.run_dir and os.path.isfile(os.path.join(job.spec.run_dir, 'run.json')):
            cat.add_run(job.spec.run_dir)
        return True
