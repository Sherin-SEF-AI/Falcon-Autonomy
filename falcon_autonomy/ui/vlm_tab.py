"""The Models page's Language tab: vision-language driving explanations scored against the simulator's truth
(core.vlm, core.vlm_eval, core.vlm_models). Pick a runner — the oracle or the noisy baseline (both test runners that
read the truth, not models), a downloaded open VLM, or a Python plug-in — and recorded drives, run the evaluation as
a job, read the per-question-type scores with their intervals, and browse frames with each question, the answer, the
truth and its score. The candidate VLMs are listed with their licences and sizes; one downloads only when asked,
after its licence and checksums are shown."""
from __future__ import annotations

import json
import os
import re
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QMessageBox, QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import vlm_eval as E
from ..core import vlm_models as VM
from ..core.jobs import FINAL
from ..core.paths import tilde
from ..core.provenance import read_json
from .theme import STATUS
from .widgets import Form, KeyValues, Section, button, label, scroll


def _fmt(v, nd=3):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


class LanguageTab(QWidget):
    """Controls on the left, results on the right (module doc)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.jobs = {}
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        # ---- candidates
        sec = Section('Vision-language models', 'Small open VLMs whose licences were read from their model cards. '
                                                'None is bundled: one downloads only when you ask, its files checked '
                                                'against the publisher\'s checksums and pinned by SHA-256 on that '
                                                'download.')
        self.cands = QTreeWidget()
        self.cands.setHeaderLabels(['Name', 'Parameters', 'Download', 'Licence', 'State'])
        self.cands.setRootIsDecorated(False)
        self.cands.setMinimumHeight(150)
        sec.add(self.cands)
        b = QHBoxLayout()
        b.addWidget(button('Download…', self.download))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        # ---- evaluate
        sec = Section('Evaluate explanations', 'Falcon\'s grounded questions about each chosen frame go to the runner; '
                                               'answers are scored against the simulator\'s facts with 95 % bootstrap '
                                               'intervals over drives.')
        f = Form()
        self.runner = QComboBox()
        self.entry = QLineEdit()
        self.entry.setPlaceholderText('module:Class (a plug-in with answer(images, prompt, meta))')
        self.noise = QDoubleSpinBox()
        self.noise.setRange(0.0, 1.0)
        self.noise.setSingleStep(0.05)
        self.noise.setValue(0.3)
        self.every = QSpinBox()
        self.every.setRange(1, 200)
        self.every.setValue(5)
        self.max_frames = QSpinBox()
        self.max_frames.setRange(0, 100000)
        self.max_frames.setSpecialValueText('all')
        self.clip = QSpinBox()
        self.clip.setRange(1, 8)
        self.boot = QSpinBox()
        self.boot.setRange(50, 10000)
        self.boot.setValue(1000)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        for text, w in (('Runner', self.runner), ('Plug-in', self.entry), ('Noise (noisy baseline)', self.noise),
                        ('Every n-th frame', self.every), ('Frames per drive', self.max_frames),
                        ('Frames per clip', self.clip), ('Bootstrap resamples', self.boot), ('Seed', self.seed)):
            f.row(text, w)
        sec.add(f)
        sec.add(label('Drives (language data are derived from each)', h=True))
        self.runs = QListWidget()
        self.runs.setMinimumHeight(110)
        sec.add(self.runs)
        b = QHBoxLayout()
        b.addWidget(button('Evaluate', self.evaluate, role='primary'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        self.state = KeyValues(('Job',))
        lv.addWidget(self.state)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        # ---- results
        right = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tv = QVBoxLayout(top)
        self.ev = QComboBox()
        self.ev.currentIndexChanged.connect(lambda _: self.show_eval())
        tv.addWidget(self.ev)
        self.info = KeyValues(('Runner', 'Licence', 'Drives', 'Questions', 'Interval', 'Note'))
        tv.addWidget(self.info)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(['Question type', 'Metric', 'Value', '95 % CI', 'Answers'])
        self.table.setRootIsDecorated(False)
        tv.addWidget(self.table, 1)
        right.addWidget(top)
        bottom = QSplitter()
        self.frames = QTreeWidget()
        self.frames.setHeaderLabels(['Frame', 'Correct'])
        self.frames.setRootIsDecorated(False)
        self.frames.currentItemChanged.connect(lambda it, _: self.show_frame(it.data(0, 256) if it else None))
        bottom.addWidget(self.frames)
        fw = QWidget()
        fv = QVBoxLayout(fw)
        self.image = QLabel()
        self.image.setMinimumSize(320, 180)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fv.addWidget(self.image, 1)
        self.qa = QTreeWidget()
        self.qa.setHeaderLabels(['Question', 'Answer', 'Truth', 'Score'])
        self.qa.setRootIsDecorated(False)
        self.qa.setWordWrap(True)
        fv.addWidget(self.qa, 1)
        bottom.addWidget(fw)
        bottom.setSizes([160, 640])
        right.addWidget(bottom)
        split.addWidget(right)
        split.setSizes([380, 820])
        self.answers = {}
        self._pix = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1500)
        self.win.bridge.finished.connect(self.finished)
        self.refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()                    # drives recorded and models downloaded since it was last shown

    # ------------------------------------------------------------------ state
    def ws(self):
        return self.win.workspace()

    def refresh(self):
        ws = self.ws()
        self.cands.clear()
        for r in VM.list_candidates(ws):
            size = f'{r["bytes"] / 2 ** 30:.2f} GiB' if r['bytes'] else '—'
            it = QTreeWidgetItem([r['name'], f'{r["params"] / 1e9:.2f} B', size, r['licence'], r['state']])
            it.setData(0, 256, r['name'])
            it.setToolTip(0, f'{r["repo"]}\n{r["refused"] or r["note"]}')
            if r['refused']:
                it.setToolTip(4, r['refused'])
                for k in range(5):
                    it.setForeground(k, self.palette().placeholderText())
            self.cands.addTopLevelItem(it)
        for k in range(5):
            self.cands.resizeColumnToContents(k)
        cur = self.runner.currentData()
        self.runner.clear()
        self.runner.addItem('Oracle — core.language\'s answers (test runner, not a model)', 'oracle')
        self.runner.addItem('Noisy baseline — seeded corruption (test runner, not a model)', 'noisy')
        for r in VM.list_candidates(ws):
            if r['state'].startswith('downloaded'):
                self.runner.addItem(f'{r["name"]} ({r["licence"]})', f'hf:{r["name"]}')
        self.runner.addItem('Plug-in (module:Class below)', 'plugin')
        k = self.runner.findData(cur)
        self.runner.setCurrentIndex(k if k >= 0 else 0)
        checked = {self.runs.item(i).data(256) for i in range(self.runs.count())
                   if self.runs.item(i).checkState() == Qt.CheckState.Checked}
        self.runs.clear()
        cat = self.win.service.catalog
        for r in (cat.list('run', 500) if cat is not None else []):
            if r['summary'].get('worker') not in (None, 'drive') or r['status'] != 'ok':
                continue
            if not os.path.isfile(os.path.join(r['path'], 'actors.jsonl')):
                continue                     # the language data need every actor's state per step
            it = QListWidgetItem(r['name'])
            it.setData(256, r['path'])
            it.setToolTip(tilde(r['path']))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if r['path'] in checked else Qt.CheckState.Unchecked)
            self.runs.addItem(it)
        self.load_results()

    def load_results(self):
        cur = self.ev.currentData()
        self.ev.blockSignals(True)
        self.ev.clear()
        for d in E.list_evaluations(self.ws()):
            self.ev.addItem(os.path.basename(d), d)
        k = self.ev.findData(cur)
        self.ev.setCurrentIndex(k if k >= 0 else 0)
        self.ev.blockSignals(False)
        self.show_eval()

    def checked_runs(self):
        return [self.runs.item(i).data(256) for i in range(self.runs.count())
                if self.runs.item(i).checkState() == Qt.CheckState.Checked]

    def runner_spec(self) -> str:
        r = self.runner.currentData() or 'oracle'
        if r == 'noisy':
            return f'noisy:p={self.noise.value():g},h={self.noise.value() / 3:.3g},seed={self.seed.value()}'
        if r == 'plugin':
            return f'plugin:{self.entry.text().strip()}'
        return r

    # ------------------------------------------------------------------ actions
    def _submit(self, argv, title, gpu=False):
        job = self.win.queue.submit(E.job_spec(argv, title, needs_gpu=gpu, memory_gb=6.0 if gpu else 1.5))
        self.jobs[job.id] = job
        self.win.jobs_dock.raise_()
        self.state.set('Job', job.summary())
        return job

    def download(self):
        it = self.cands.currentItem()
        if it is None:
            self.state.set('Job', 'Choose a model first.')
            return
        name = it.data(0, 256)
        if name in VM.REFUSED:
            QMessageBox.information(self, 'Refused', f'{name} is refused by default: {VM.REFUSED[name][3]}.')
            return
        if not self.page._ready():
            return
        if QMessageBox.question(self, 'Download the model?', VM.describe_candidate(name) + '\n\nDownload it now?') \
                != QMessageBox.StandardButton.Yes:
            return
        self._submit(['fetch', name, '--workspace', self.ws(), '--yes'], f'Download {name}')

    def evaluate(self):
        runs = self.checked_runs()
        if not runs:
            self.state.set('Job', 'Check at least one drive.')
            return
        spec = self.runner_spec()
        if spec == 'plugin:' or (spec.startswith('plugin:') and ':' not in spec[7:]):
            self.state.set('Job', 'A plug-in is module:Class.')
            return
        if not self.page._ready():
            return
        name = re.sub(r'[^A-Za-z0-9_.-]+', '-', spec.split(',')[0])[:32]
        base = os.path.join(E.evaluations_dir(self.ws()), f'{time.strftime("%Y%m%d-%H%M%S")}-{name}')
        out, k = base, 2
        while os.path.exists(out):
            out, k = f'{base}-{k}', k + 1
        os.makedirs(out)
        argv = ['eval', '--runner', spec, '--out', out, '--workspace', self.ws(), '--every', str(self.every.value()),
                '--max-frames', str(self.max_frames.value()), '--clip', str(self.clip.value()),
                '--boot', str(self.boot.value()), '--seed', str(self.seed.value())]
        for r in runs:
            argv += ['--run', r]
        self._submit(argv, f'VLM evaluation: {name} on {len(runs)} drive(s)', gpu=E.needs_gpu(spec))

    def finished(self, job):
        if job.id not in self.jobs:
            return
        self.jobs.pop(job.id)
        self.state.set('Job', job.summary())
        self.refresh()

    def poll(self):
        live = [j for j in self.jobs.values() if j.state not in FINAL]
        if live:
            self.state.set('Job', ' · '.join(j.summary() for j in live))

    # ------------------------------------------------------------------ results
    def show_eval(self):
        self.table.clear()
        self.frames.clear()
        self.answers = {}
        self.show_frame(None)
        d = self.ev.currentData()
        if not d:
            for k in self.info.vals:
                self.info.set(k, '—')
            return
        try:
            sc = read_json(os.path.join(d, 'scores.json'))
        except (OSError, ValueError):
            return
        r, agg = sc['runner'], sc['scores']
        self.info.set('Runner', f'{r.get("name")} ({r.get("kind")}; {", ".join(r.get("providers") or [])})')
        self.info.set('Licence', r.get('licence') or 'not stated')
        self.info.set('Drives', ', '.join(sc.get('drives') or []))
        self.info.set('Questions', str(sc.get('n_questions')))
        self.info.set('Interval', f'{agg["n_boot"]} resamples of {agg["unit"]}s ({agg["n_units"]})')
        notes = (['NOT A MODEL: this runner reads the ground truth'] if sc.get('not_a_model') else []) + \
            ([agg['note']] if agg.get('note') else [])
        notes.append(f'scorer {sc.get("scorer", 1)}')
        tr = sc.get('truth') or {}
        if tr.get('mode') == 'visible':
            notes.append(f'truth: visible actors only ({tr.get("n_skipped", 0)} questions skipped: nothing relevant '
                         'visible)')
        elif tr:
            notes.append('truth: every actor within the radius, visible or not')
        self.info.set('Note', ' · '.join(notes) or '—')
        for row in sc['table']:
            ci = '—' if row['ci_low'] is None else f'{_fmt(row["ci_low"])} – {_fmt(row["ci_high"])}'
            self.table.addTopLevelItem(QTreeWidgetItem([row['type'], row['metric'], _fmt(row['value']), ci,
                                                        _fmt(row.get('n'))]))
        for k in range(2):
            self.table.resizeColumnToContents(k)
        p = os.path.join(d, 'answers.jsonl')
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as fh:
                for x in fh:
                    if x.strip():
                        a = json.loads(x)
                        self.answers.setdefault((a['drive'], a['step']), []).append(a)
        for (drive, step), rows in self.answers.items():
            ok = sum(bool(a['score']['correct']) for a in rows)
            it = QTreeWidgetItem([f'{drive} · step {step}', f'{ok}/{len(rows)}'])
            it.setData(0, 256, (drive, step))
            self.frames.addTopLevelItem(it)

    def show_frame(self, key):
        self.qa.clear()
        rows = self.answers.get(key) if key else None
        if not rows:
            self.image.setPixmap(QPixmap())
            self.image.setText('Select a frame')
            return
        img = (rows[0].get('images') or [None])[-1]
        if img and os.path.isfile(img):
            self._pix = QPixmap(img)
            self.image.setPixmap(self._pix.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))
        else:
            self.image.setPixmap(QPixmap())
            self.image.setText('The frame\'s image is not on this machine')
        for a in rows:
            sc = a['score']
            truth = a['truth'].get('value')
            if a['type'] == 'brake_reason':
                truth = ', '.join(f'{r["kind"]} {r.get("id") or r.get("state") or ""}'.strip()
                                  for r in a['truth']['reasons']) or 'no reason to brake'
            verdict = 'correct' if sc['correct'] else ('unparsed' if not sc['parsed'] else 'wrong')
            if sc.get('hallucinated'):
                verdict += f' · {sc["hallucinated"]} hallucinated'
            if sc.get('grounded') is False:
                verdict += ' · wrong object'
            it = QTreeWidgetItem([a['question'], a['answer'], _fmt(truth), verdict])
            it.setToolTip(1, a['answer'])
            if not sc['correct']:
                it.setForeground(3, QColor(STATUS['critical']))
            self.qa.addTopLevelItem(it)
        self.qa.resizeColumnToContents(3)
