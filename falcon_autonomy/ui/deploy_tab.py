"""The Models page's Deploy tab: a registered ONNX detector as TensorRT engines on the Jetson AGX Orin (core.deploy,
docs/deploy.md) — or on the simulated device, onnxruntime on this machine's CPU standing in for TensorRT and labelled
so everywhere (not TensorRT, not the Orin: the pipeline only). Choose the model, the device, the precisions (and a DLA
core), the evaluation and calibration drives and a gate file; one job builds the engines on the device, profiles them
(latency, throughput, power on the INA3221 rails, estimated energy), scores them on the evaluation frames against the
host's ONNX model with paired intervals, evaluates the gate and writes the report. The table shows, per engine, its
numbers and the gate's verdict. The device's address and SSH key are the Hardware page's (the user's settings)."""
from __future__ import annotations

import os
import sys
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
                             QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import deploy as DP
from ..core.jobs import FINAL, JobSpec
from ..core.paths import tilde
from .theme import STATUS
from .widgets import Form, KeyValues, Section, button, label, scroll

COLUMNS = ('Engine', 'Median (ms)', 'p95 (ms)', 'Throughput (qps)', 'Power (W)', 'Energy (mJ, est.)', 'AP [95 %]',
           'AP / reference [95 %]', 'Gate')
VERDICT_COLOUR = {'pass': 'good', 'fail': 'critical', 'incomplete': 'warning'}


def _f(v, nd=3):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def _ci(v, c, nd=3):
    if v is None:
        return '—'
    if not c or c[0] is None:
        return _f(v, nd)
    return f'{v:.{nd}f} [{c[0]:.{nd}f}, {c[1]:.{nd}f}]'


class DeployTab(QWidget):
    """Controls on the left, deployments on the right (module doc)."""

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
        sec = Section('Deploy to the Jetson', 'TensorRT engines are built, timed and scored on the device itself '
                                              '(they are specific to its GPU and TensorRT version). The simulated '
                                              'device runs the same steps here with onnxruntime on the CPU: not '
                                              'TensorRT, not the Orin — the pipeline only.')
        f = Form()
        self.model = QComboBox()
        self.device = QComboBox()
        self.device.addItem('Simulated device (this machine, onnxruntime CPU)', 'sim')
        self.device.addItem('Jetson AGX Orin (the Hardware page\'s device)', 'orin')
        prec = QHBoxLayout()
        self.prec = {}
        for p in DP.PRECISIONS:
            self.prec[p] = QCheckBox(p.upper())
            self.prec[p].setChecked(True)
            prec.addWidget(self.prec[p])
        prec.addStretch(1)
        pw = QWidget()
        pw.setLayout(prec)
        self.dla = QComboBox()
        self.dla.addItem('none (GPU only)', None)
        self.dla.addItem('DLA 0 (GPU fallback)', 0)
        self.dla.addItem('DLA 1 (GPU fallback)', 1)
        self.calib_count = QSpinBox()
        self.calib_count.setRange(1, 100000)
        self.calib_count.setValue(DP.DEFAULT_CALIB_COUNT)
        self.every = QSpinBox()
        self.every.setRange(1, 1000)
        self.every.setValue(2)
        self.runs_n = QSpinBox()
        self.runs_n.setRange(10, 1000000)
        self.runs_n.setValue(DP.DEFAULT_RUNS)
        self.boot = QSpinBox()
        self.boot.setRange(50, 10000)
        self.boot.setValue(1000)
        self.gate = QLineEdit()
        self.gate.setPlaceholderText('a gate JSON (empty: the default gate)')
        gp = QHBoxLayout()
        gp.addWidget(self.gate, 1)
        gp.addWidget(button('…', self.pick_gate))
        gw = QWidget()
        gw.setLayout(gp)
        for text, w in (('Model', self.model), ('Device', self.device), ('Precisions', pw), ('DLA', self.dla),
                        ('Calibration frames', self.calib_count), ('Every n-th frame', self.every),
                        ('Timed runs', self.runs_n), ('Bootstrap resamples', self.boot), ('Gate', gw)):
            f.row(text, w)
        sec.add(f)
        sec.add(label('Evaluation drives', h=True))
        self.ev_runs = QListWidget()
        self.ev_runs.setMinimumHeight(90)
        sec.add(self.ev_runs)
        sec.add(label('Calibration drives (INT8; other drives than the evaluation\'s)', h=True))
        self.cal_runs = QListWidget()
        self.cal_runs.setMinimumHeight(90)
        sec.add(self.cal_runs)
        b = QHBoxLayout()
        b.addWidget(button('Build, profile, score and gate', self.run_all, role='primary'))
        b.addWidget(button('Check the device', self.check))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        self.state = KeyValues(('Job',))
        lv.addWidget(self.state)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rv = QVBoxLayout(right)
        self.sel = QComboBox()
        self.sel.currentIndexChanged.connect(lambda _: self.show_doc())
        rv.addWidget(self.sel)
        self.info = KeyValues(('Model', 'Device', 'Software', 'Power mode', 'Frames', 'Reference', 'Gate', 'Note'))
        rv.addWidget(self.info)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(list(COLUMNS))
        self.table.setRootIsDecorated(False)
        rv.addWidget(self.table, 1)
        self.detail = label('', muted=True, wrap=True, sel=True)
        self.table.currentItemChanged.connect(lambda it, _: self.show_engine(it.data(0, 256) if it else None))
        rv.addWidget(self.detail)
        bar = QHBoxLayout()
        bar.addWidget(button('Evaluate another gate…', self.regate))
        bar.addWidget(button('Write the report', self.write_report))
        bar.addStretch(1)
        rv.addLayout(bar)
        split.addWidget(right)
        split.setSizes([400, 820])
        self.doc = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1500)
        self.win.bridge.finished.connect(self.finished)
        self.refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    # ------------------------------------------------------------------ state
    def ws(self):
        return self.win.workspace()

    def refresh(self):
        cur = self.model.currentData()
        self.model.clear()
        for m in self.win.project.doc['models']:
            if m.get('kind') == 'detector' and m.get('runtime') == 'onnx':
                self.model.addItem(m['name'], m['id'])
        k = self.model.findData(cur)
        self.model.setCurrentIndex(k if k >= 0 else 0)
        cat = self.win.service.catalog
        rows = [r for r in (cat.list('run', 500) if cat is not None else [])
                if r['summary'].get('worker') in (None, 'drive', 'translate') and r['status'] == 'ok']
        for lst in (self.ev_runs, self.cal_runs):
            checked = {lst.item(i).data(256) for i in range(lst.count())
                       if lst.item(i).checkState() == Qt.CheckState.Checked}
            lst.clear()
            for r in rows:
                it = QListWidgetItem(r['name'])
                it.setData(256, r['path'])
                it.setToolTip(tilde(r['path']))
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if r['path'] in checked else Qt.CheckState.Unchecked)
                lst.addItem(it)
        self.load_results()

    def load_results(self):
        cur = self.sel.currentData()
        self.sel.blockSignals(True)
        self.sel.clear()
        for d in DP.list_deployments(self.ws()):
            self.sel.addItem(os.path.basename(d), d)
        k = self.sel.findData(cur)
        self.sel.setCurrentIndex(k if k >= 0 else 0)
        self.sel.blockSignals(False)
        self.show_doc()

    @staticmethod
    def _checked(lst):
        return [lst.item(i).data(256) for i in range(lst.count())
                if lst.item(i).checkState() == Qt.CheckState.Checked]

    # ------------------------------------------------------------------ actions
    def _submit(self, argv, title):
        job = self.win.queue.submit(JobSpec('deploy', title, [sys.executable, '-m', 'falcon_autonomy', 'deploy'] +
                                            list(argv), needs_gpu=False, memory_gb=2.0))
        self.jobs[job.id] = job
        self.win.jobs_dock.raise_()
        self.state.set('Job', job.summary())
        return job

    def pick_gate(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Gate file', self.gate.text(), 'JSON (*.json)')
        if p:
            self.gate.setText(p)

    def run_all(self):
        mid = self.model.currentData()
        runs, cal = self._checked(self.ev_runs), self._checked(self.cal_runs)
        prec = [p for p, c in self.prec.items() if c.isChecked()]
        if not mid or not runs or not prec:
            self.state.set('Job', 'Choose an ONNX detector, at least one precision and at least one evaluation drive.')
            return
        if 'int8' in prec and not cal:
            self.state.set('Job', 'INT8 needs calibration drives (other drives than the evaluation\'s).')
            return
        if not self.page._ready():
            return
        base = os.path.join(DP.deploy_dir(self.ws()), f'{time.strftime("%Y%m%d-%H%M%S")}-{mid}')
        out, k = base, 2
        while os.path.exists(out):
            out, k = f'{base}-{k}', k + 1
        os.makedirs(out)
        argv = ['all', self.win.project.path, '--model', mid, '--out', out, '--device', self.device.currentData(),
                '--precision', ','.join(prec), '--calib-count', str(self.calib_count.value()), '--every',
                str(self.every.value()), '--runs', str(self.runs_n.value()), '--boot', str(self.boot.value())]
        if self.dla.currentData() is not None:
            argv += ['--dla', str(self.dla.currentData())]
        if self.gate.text().strip():
            argv += ['--gate', self.gate.text().strip()]
        for r in runs:
            argv += ['--run', r]
        for r in cal:
            argv += ['--calib-run', r]
        where = 'the simulated device' if self.device.currentData() == 'sim' else 'the Jetson'
        self._submit(argv, f'Deploy {self.model.currentText()} on {where}')

    def check(self):
        self._submit(['check', '--device', self.device.currentData()], 'Check the deployment device')

    def regate(self):
        d = self.sel.currentData()
        if not d:
            return
        p, _ = QFileDialog.getOpenFileName(self, 'Gate file', self.gate.text(), 'JSON (*.json)')
        if p:
            self._submit(['gate', d, '--gate', p], f'Gate {os.path.basename(d)}')

    def write_report(self):
        d = self.sel.currentData()
        if d:
            self._submit(['report', d], f'Deploy report {os.path.basename(d)}')

    def finished(self, job):
        if job.id not in self.jobs:
            return
        self.jobs.pop(job.id)
        self.state.set('Job', job.summary())
        self.load_results()

    def poll(self):
        live = [j for j in self.jobs.values() if j.state not in FINAL]
        if live:
            self.state.set('Job', ' · '.join(j.summary() for j in live))

    # ------------------------------------------------------------------ results
    def show_doc(self):
        self.table.clear()
        self.detail.setText('')
        d = self.sel.currentData()
        self.doc = None
        if not d:
            for k in self.info.vals:
                self.info.set(k, '—')
            return
        try:
            doc = DP.read_doc(d)
        except DP.DeployError as e:
            self.info.set('Note', str(e))
            return
        self.doc = doc
        m, dv, env, plan = doc.get('model') or {}, doc.get('device') or {}, doc.get('env') or {}, doc.get('plan') or {}
        self.info.set('Model', f'{m.get("name")} ({m.get("licence")})')
        dev = f'{dv.get("mode")} — {dv.get("where")}; backend {dv.get("backend")}'
        self.info.set('Device', ('SIMULATED — not TensorRT, not the Orin; ' if doc.get('simulated') else '') + dev)
        self.info.set('Software', f'{env.get("model") or env.get("platform")}; L4T {env.get("l4t") or "—"}; TensorRT '
                                  f'{env.get("tensorrt_python") or "—"}; trtexec {env.get("trtexec") or "none"}')
        pm = env.get('power_mode') or {}
        locked = (env.get('clocks') or {}).get('locked')
        self.info.set('Power mode', f'{pm.get("name") or pm.get("id") or "not recorded"}; clocks '
                                    f'{ {True: "locked", False: "not locked", None: "unknown"}[locked] }')
        self.info.set('Frames', f'{plan.get("eval_frames")} evaluation ({len(plan.get("eval_drives") or [])} drives), '
                                f'{plan.get("calib_frames")} calibration')
        ref = doc.get('reference')
        ap = (ref or {}).get('scores', {}).get('AP') or {}
        self.info.set('Reference', '—' if not ref else
                      f'host ONNX (CPU): AP {_ci(ap.get("value"), [ap.get("ci_low"), ap.get("ci_high")])}')
        g = doc.get('gate') or {}
        self.info.set('Gate', '—' if not g else f'{g["gate"].get("name")}' +
                      ('' if g.get('release_decision') else ' (not a release decision: simulated device)'))
        self.info.set('Note', ' · '.join([x for x in (doc.get('notes') or [])] + ([ref['note']] if ref and
                                                                                  ref.get('note') else [])) or '—')
        for r in DP.engine_rows(doc):
            it = QTreeWidgetItem([r['name'], _f(r['median_ms']), _f(r['p95_ms']), _f(r['throughput_qps'], 1),
                                  _f(r['power_w'], 2) if r['power_w'] is not None else 'not measured',
                                  _f(r['energy_mj'], 2) if r['energy_mj'] is not None else 'not measured',
                                  _ci(r['ap'], r['ap_ci']), _ci(r['ap_ratio'], r['ap_ratio_ci']),
                                  r['verdict'] or '—'])
            it.setData(0, 256, r['name'])
            if r['error']:
                it.setToolTip(0, r['error'])
                it.setForeground(0, QColor(STATUS['critical']))
            if r['power_note']:
                for k in (4, 5):
                    it.setToolTip(k, r['power_note'])
            if r['verdict'] in VERDICT_COLOUR:
                it.setForeground(8, QColor(STATUS[VERDICT_COLOUR[r['verdict']]]))
            self.table.addTopLevelItem(it)
        for k in range(len(COLUMNS)):
            self.table.resizeColumnToContents(k)

    def show_engine(self, name):
        if not name or not self.doc:
            self.detail.setText('')
            return
        e = self.doc['engines'].get(name) or {}
        b = e.get('build') or {}
        lines = [f'{name}: ' + ('build failed — ' + b['error'] if b.get('error') else
                                f'built in {_f(b.get("build_seconds"), 1)} s, '
                                f'{_f((b.get("engine_bytes") or 0) / 2 ** 20, 2)} MiB')]
        lay = b.get('layers') or {}
        if lay.get('by_output_precision'):
            lines.append('layers by output precision: ' + ', '.join(f'{k} {v}' for k, v in
                                                                  lay['by_output_precision'].items()))
        elif lay.get('initialiser_types'):
            lines.append('simulated engine, ONNX initialisers: ' + ', '.join(f'{k} {v}' for k, v in
                                                                          lay['initialiser_types'].items()))
        fb = (b.get('log') or {}).get('dla_fallback')
        if e.get('dla_core') is not None:
            lines.append(f'fell back from DLA {e["dla_core"]} to the GPU: ' +
                         (', '.join(x.get('layer') or '?' for x in fb) if fb else 'none reported in the build log'))
        crit = (((self.doc.get('gate') or {}).get('engines') or {}).get(name) or {}).get('criteria') or {}
        for k, c in crit.items():
            lines.append(f'gate · {k}: {c["status"]}' + (f' (value {_f(c.get("value"))}, limit {c.get("limit")})'
                                                         if c.get('value') is not None else '') +
                         (f' — {c["why"]}' if c.get('why') else ''))
        self.detail.setText('\n'.join(lines))
