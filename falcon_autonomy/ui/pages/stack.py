"""Stack testing: drive Falcon's ego with an external stack through ROS 2 (Falcon's bridge), run scenario suites with
pass/fail criteria as jobs, read the results with their intervals, replay any run, and build the safety-case report
(core.stackbridge, core.stack, core.safetycase; the jobs are `falcon stack ...` commands, sim.stacksuite)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                             QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QSpinBox, QSplitter, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import faults as FL
from ...core import scenarios as SC
from ...core import stack as ST
from ...core import stackbridge as SB
from ...core.jobs import FINAL, JobSpec
from ...core.paths import tilde
from ...core.provenance import read_json, write_json
from ..widgets import Form, KeyValues, Note, Page, Section, button, label, scroll

DRIVERS = (('ros2', 'ROS 2 stack (Falcon\'s bridge)'), ('follower', 'Falcon\'s lane follower (baseline)'),
           ('autopilot', 'CARLA\'s autopilot (baseline)'))
CONTROLS = (('autoware', 'Autoware Control (autoware_control_msgs)'), ('ackermann', 'AckermannDriveStamped'),
            ('carla', 'CarlaEgoVehicleControl (steer, throttle, brake)'))



def suite_title(doc) -> str:
    """The Jobs panel's name for a suite run: "Suite <name> (<driver>)", without saying "suite" twice when the suite
    is itself named so."""
    name = str(doc.get('name') or '').strip()
    if name.lower() == 'suite' or name.lower().startswith('suite '):
        name = name[5:].strip()
    return f'Suite {name} ({doc["stack"]["driver"]})' if name else f'Suite ({doc["stack"]["driver"]})'


class StackPage(Page):
    key, title, icon = 'stack', 'Stack testing', 'link'
    about = ('Test a driving stack: connect it through ROS 2, run scenario suites with pass/fail criteria, and read '
             'the evidence against an operational design domain. A report, not a certification claim.')

    def __init__(self, win):
        super().__init__(win)
        self.jobs: dict = {}
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(24, 18, 12, 18)
        lv.addWidget(label(self.title, page=True))
        lv.addWidget(Note(self.about, limit=160))

        sec = Section('ROS 2 bridge', 'Falcon publishes the rig\'s sensors, the odometry, /tf and /clock per '
                                      'simulation step and waits for the stack\'s control of that step (or its '
                                      'deadline). The node runs in the falcon-ros2-bridge:humble image or on any '
                                      'ROS 2 machine.')
        f = Form()
        self.control = QComboBox()
        for k, t in CONTROLS:
            self.control.addItem(t, k)
        self.control.currentIndexChanged.connect(lambda _: self.show_topics())
        self.port = QSpinBox()
        self.port.setRange(0, 65535)
        self.port.setSpecialValueText('a free port per run')
        self.deadline = QDoubleSpinBox()
        self.deadline.setRange(0.01, 60.0)
        self.deadline.setValue(0.5)
        self.deadline.setSuffix(' s')
        self.on_miss = QComboBox()
        self.on_miss.addItem('hold the last control', 'hold')
        self.on_miss.addItem('brake', 'brake')
        self.test_stack = QCheckBox('Launch the test stack (bridge node + follower) with each run')
        self.test_stack.setChecked(True)
        for text, w, note in (('Control message', self.control, ''), ('Link port', self.port, ''),
                              ('Control deadline', self.deadline, 'wall clock per step; a miss is recorded'),
                              ('On a missed deadline', self.on_miss, '')):
            f.row(text, w, note)
        sec.add(f)
        sec.add(self.test_stack)
        self.bridge_state = KeyValues(('Docker', 'Bridge image'))
        sec.add(self.bridge_state)
        row = QHBoxLayout()
        row.addWidget(button('Check', self.check_bridge, tip='docker and the falcon-ros2-bridge:humble image'))
        row.addWidget(button('Build the image', self.build_image, tip=f'docker build -t {SB.IMAGE} (ROS 2 Humble, '
                                                                      'Autoware and Ackermann messages)'))
        row.addWidget(button('Copy the node\'s command', self.copy_command))
        row.addStretch(1)
        sec.add(row)
        lv.addWidget(sec)

        sec = Section('Suite', 'Scenarios × seeds, each run with the same stack and criteria. A suite resumes: '
                               'finished runs are kept, never repeated.')
        f = Form()
        self.name = QLineEdit('suite')
        self.driver = QComboBox()
        for k, t in DRIVERS:
            self.driver.addItem(t, k)
        self.world = QComboBox()
        self.rig = QComboBox()
        self.rig.currentIndexChanged.connect(lambda _: (self.show_topics(), self._fault_sensors()))
        self.motion = QComboBox()
        self.motion.addItem('kinematic (repeatable)', 'kinematic')
        self.motion.addItem('physics', 'physics')
        self.seeds = QLineEdit('0,1,2')
        self.no_collision = QCheckBox('no collision')
        self.no_collision.setChecked(True)
        self.min_ttc = QDoubleSpinBox()
        self.min_ttc.setRange(0.0, 10.0)
        self.min_ttc.setValue(1.0)
        self.min_ttc.setSuffix(' s')
        self.max_missed = QSpinBox()
        self.max_missed.setRange(0, 100000)
        for text, w in (('Name', self.name), ('Stack', self.driver), ('World', self.world), ('Rig', self.rig),
                        ('Motion', self.motion), ('Seeds', self.seeds), ('Pass when', self.no_collision),
                        ('and time to collision ≥', self.min_ttc), ('and missed deadlines ≤', self.max_missed)):
            f.row(text, w)
        sec.add(f)
        self.scn_list = QListWidget()
        self.scn_list.setMinimumHeight(150)
        sec.add(label('Scenarios', muted=True))
        sec.add(self.scn_list)
        sec.add(label('Sensor faults during each run', muted=True))
        sec.add(label('A sensor fails mid-run as the stack receives it (core.faults: mud on a camera, a lidar sector '
                      'or packets lost, GNSS multipath, a sensor gone). With the minimal-risk check on, each run is '
                      'judged on what the stack does after the first fault (docs/stack-testing.md).', muted=True,
                      wrap=True))
        self.fault_list = QTreeWidget()
        self.fault_list.setHeaderLabels(['Sensor', 'Fault', 'From s', 'For s', 'Settings'])
        self.fault_list.setRootIsDecorated(False)
        self.fault_list.setMaximumHeight(110)
        sec.add(self.fault_list)
        fr = QHBoxLayout()
        self.fault_sensor = QComboBox()
        self.fault_sensor.currentIndexChanged.connect(lambda _: self._fault_kinds())
        self.fault_kind = QComboBox()
        self.fault_onset = QDoubleSpinBox()
        self.fault_onset.setRange(0.0, 3600.0)
        self.fault_onset.setValue(3.0)
        self.fault_onset.setPrefix('from ')
        self.fault_onset.setSuffix(' s')
        self.fault_for = QDoubleSpinBox()
        self.fault_for.setRange(0.0, 3600.0)
        self.fault_for.setSpecialValueText('to the end')
        self.fault_for.setPrefix('for ')
        self.fault_for.setSuffix(' s')
        self.fault_opts = QLineEdit()
        self.fault_opts.setPlaceholderText('settings (optional), e.g. from_deg=-60,to_deg=60 or cover=0.6')
        for wdg in (self.fault_sensor, self.fault_kind, self.fault_onset, self.fault_for):
            fr.addWidget(wdg, 1)
        sec.add(fr)
        fr2 = QHBoxLayout()
        fr2.addWidget(self.fault_opts, 1)
        fr2.addWidget(button('Add', self.add_fault))
        fr2.addWidget(button('Remove', self.remove_fault))
        sec.add(fr2)
        self.mrm = QCheckBox('Judge the minimal-risk manoeuvre after the first fault')
        self.mrm.setToolTip('Each run passes when the stack reacts in time, comes to a standstill, keeps its '
                            'deceleration and jerk bounded, stays in its lane or moves to the shoulder, and does not '
                            'collide (core.mrm; the thresholds are heuristic)')
        self.degraded = QCheckBox('The test stack reacts to sensor failures (degraded mode)')
        self.degraded.setToolTip('Falcon\'s lane follower, or the ROS 2 test stack when it is launched with each run, '
                                 'watches its sensors and stops in its lane when one fails')
        sec.add(self.mrm)
        sec.add(self.degraded)
        row = QHBoxLayout()
        self.run_btn = button('Run the suite', self.run_suite, role='primary')
        row.addWidget(self.run_btn)
        row.addWidget(button('Save as JSON…', self.save_suite))
        row.addStretch(1)
        sec.add(row)
        self.state = KeyValues(('State',))
        sec.add(self.state)
        lv.addWidget(sec)

        sec = Section('CI', 'CI mode runs a suite headless on every commit of a stack and gates it (pass rate with '
                            'Wilson\'s interval, regressions by a paired bootstrap, rule violations per km, latency): '
                            '`falcon ci run CONFIG --out <workspace>/ci/NAME`. Its results in this workspace are '
                            'listed here. Posting to GitHub is a command-line step only (`falcon ci post … --post`), '
                            'never from this page.')
        self.ci_runs = QTreeWidget()
        self.ci_runs.setHeaderLabels(['Started', 'Name', 'Verdict', 'Commit', 'Gates not passed'])
        self.ci_runs.setRootIsDecorated(False)
        self.ci_runs.setMinimumHeight(110)
        sec.add(self.ci_runs)
        row = QHBoxLayout()
        row.addWidget(button('Refresh', self.load_ci))
        row.addWidget(button('Open its report', self.open_ci_report))
        row.addStretch(1)
        sec.add(row)
        row = QHBoxLayout()
        row.addWidget(button('Write a CI config…', lambda: self.write_ci(False),
                             tip='falcon-ci.yml and falcon-suite.json into a repository you choose (files that '
                                 'exist are kept)'))
        row.addWidget(button('Write the GitHub workflow…', lambda: self.write_ci(True),
                             tip='also .github/workflows/falcon.yml: a fake-carla smoke job and a self-hosted CARLA '
                                 'job (docs/ci.md)'))
        row.addStretch(1)
        sec.add(row)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 18, 24, 18)
        rv.addWidget(label('Topics in use', h=True))
        self.topics = QTreeWidget()
        self.topics.setHeaderLabels(['Direction', 'Topic', 'Type', 'Autoware'])
        self.topics.setRootIsDecorated(False)
        self.topics.setMinimumHeight(180)
        self.topics.setColumnWidth(1, 260)
        self.topics.setColumnWidth(2, 260)
        rv.addWidget(self.topics)
        rv.addWidget(label('Results', h=True))
        self.summary = KeyValues(('Runs', 'Pass rate (Wilson 95 %)', 'Collisions', 'Missed deadlines',
                                  'Minimal-risk manoeuvre', 'Outside the ODD'))
        rv.addWidget(self.summary)
        self.results = QTreeWidget()
        self.results.setHeaderLabels(['Scenario', 'Seed', 'Result', 'Criticality', 'Min TTC s', 'Missed', 'MRM',
                                      'Run'])
        self.results.setRootIsDecorated(False)
        self.results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results.setMinimumHeight(200)
        from ..assistant_dock import add_ask_menu
        add_ask_menu(self.results, win, lambda it: ('suite_result', {
            'suite': self.suite_dir(), 'scenario': it.data(0, 256).get('scenario'), 'seed': it.data(0, 256).get('seed'),
            'run': it.data(0, 256).get('run') or ''}) if it.data(0, 256) and self.suite_dir() else None)
        rv.addWidget(self.results, 1)
        row = QHBoxLayout()
        row.addWidget(button('Replay the selected run', self.replay))
        row.addWidget(button('Open its folder', self.open_run))
        row.addStretch(1)
        rv.addLayout(row)
        rv.addWidget(label('Safety-case report', h=True))
        rv.addWidget(label('Scenario coverage against the ODD, results with intervals, triggering conditions and '
                           'residual risks with their evidence, and what the report does not cover (ISO 21448 '
                           'areas, the simulation\'s limits).', muted=True, wrap=True))
        row = QHBoxLayout()
        self.odd = QLineEdit()
        self.odd.setPlaceholderText('ODD description (JSON)')
        row.addWidget(self.odd, 1)
        row.addWidget(button('Choose…', self.choose_odd))
        row.addWidget(button('New from the example', self.new_odd))
        rv.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(button('Build the report', self.build_report, role='primary'))
        row.addWidget(button('Open the report', self.open_report))
        row.addStretch(1)
        rv.addLayout(row)
        split.addWidget(right)
        split.setSizes([460, 820])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(2000)
        win.bridge.finished.connect(self._finished)
        self.refresh()

    # ------------------------------------------------------------------ the project
    def scenarios(self):
        return [(k, SC.builtin(k)) for k in SC.LIBRARY] + [(s['id'], s) for s in self.win.project.doc['scenarios']]

    def refresh(self):
        doc = self.win.project.doc
        for combo, items in ((self.world, doc['worlds']), (self.rig, doc['rigs'])):
            c = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for it in items:
                combo.addItem(it['name'], it['id'])
            k = combo.findData(c)
            combo.setCurrentIndex(k if k >= 0 else 0)
            combo.blockSignals(False)
        self._fault_sensors()
        checked = self.checked_scenarios() or ['lead_brake', 'stalled_vehicle']
        self.scn_list.clear()
        for ident, s in self.scenarios():
            it = QListWidgetItem(s['name'])
            it.setData(256, ident)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if ident in checked else Qt.CheckState.Unchecked)
            self.scn_list.addItem(it)
        if not self.odd.text() and self.win.workspace():
            p = os.path.join(self.win.workspace(), 'odd.json')
            if os.path.isfile(p):
                self.odd.setText(p)
        self.show_topics()
        self.load()
        self.load_ci()

    def checked_scenarios(self) -> list:
        return [self.scn_list.item(i).data(256) for i in range(self.scn_list.count())
                if self.scn_list.item(i).checkState() == Qt.CheckState.Checked]

    def rig_doc(self):
        return self.win.project.item('rigs', self.rig.currentData()) if self.rig.count() else None

    def bridge_doc(self) -> dict:
        return {'control': self.control.currentData(), 'port': int(self.port.value()),
                'deadline_s': float(self.deadline.value()), 'on_miss': self.on_miss.currentData()}

    def show_topics(self):
        self.topics.clear()
        rig = self.rig_doc()
        if rig is None:
            return
        b = dict(SB.bridge_defaults(), **self.bridge_doc())
        aw = SB.autoware_remaps(rig)
        fixed = {m[0]: m[2] for m in SB.AUTOWARE_MAPPING}
        for e in SB.topic_plan(rig, b):
            self.topics.addTopLevelItem(QTreeWidgetItem(['publish', e['topic'], e['type'],
                                                         aw.get(e['topic'], fixed.get(e['topic'], ''))]))
        c = SB.control_spec(b)
        self.topics.addTopLevelItem(QTreeWidgetItem(['subscribe', c['topic'], c['type'],
                                                     fixed.get(c['topic'], '')]))

    # ------------------------------------------------------------------ the bridge
    def check_bridge(self):
        docker = shutil.which('docker')
        self.bridge_state.set('Docker', docker or 'not found')
        if not docker:
            self.bridge_state.set('Bridge image', '— (needs Docker; or run the node on a ROS 2 machine)')
            return
        try:
            ok = subprocess.run([docker, 'image', 'inspect', SB.IMAGE], capture_output=True, timeout=20).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        self.bridge_state.set('Bridge image', f'{SB.IMAGE}: built' if ok else f'{SB.IMAGE}: not built')

    def build_image(self):
        argv = [sys.executable, '-m', 'falcon_autonomy', 'stack', 'image']
        self.track(self.win.queue.submit(JobSpec('stack-image', f'Build {SB.IMAGE}', argv, memory_gb=1.5)),
                   'image')

    def copy_command(self):
        port = int(self.port.value())                    # 0: the run picks one; the command needs it then
        text = SB.shell(SB.bridge_command(port or '<PORT>'))
        QGuiApplication.clipboard().setText(text)
        self.state.set('State', f'copied: {text}')

    # ------------------------------------------------------------------ suites
    def suite_doc(self) -> dict:
        seeds = [int(x) for x in self.seeds.text().replace(' ', '').split(',') if x.strip().lstrip('-').isdigit()]
        driver = self.driver.currentData()
        doc = ST.new_suite(self.name.text().strip() or 'suite', driver, self.checked_scenarios(), seeds or [0],
                           self.world.currentData() or '', self.rig.currentData() or '')
        doc['motion'] = self.motion.currentData()
        doc['criteria'] = {'no_collision': self.no_collision.isChecked(),
                           'min_ttc_s': float(self.min_ttc.value()) if self.min_ttc.value() > 0 else None,
                           'max_missed_deadlines': int(self.max_missed.value())}
        faults = self.faults()
        for e in doc['entries']:
            if faults:
                e['faults'] = [dict(f) for f in faults]
            if self.mrm.isChecked():
                e['mrm'] = True
        if self.degraded.isChecked() and driver == 'follower':
            doc['degraded_mode'] = True
        if driver == 'ros2':
            doc['stack']['bridge'] = self.bridge_doc()
            if self.test_stack.isChecked():
                from ...sim.stacksuite import test_stack_launch
                doc['stack']['launch'] = test_stack_launch(self.control.currentData(),
                                                           degraded_mode=self.degraded.isChecked())
        return doc

    # ------------------------------------------------------------------------------------------- sensor faults
    def _fault_sensors(self):
        cur = self.fault_sensor.currentData()
        self.fault_sensor.blockSignals(True)
        self.fault_sensor.clear()
        for s in (self.rig_doc() or {}).get('sensors', []):
            self.fault_sensor.addItem(f'{s["id"]} ({s["type"]})', (s['id'], s['type']))
        i = self.fault_sensor.findData(cur)
        self.fault_sensor.setCurrentIndex(max(i, 0))
        self.fault_sensor.blockSignals(False)
        self._fault_kinds()

    def _fault_kinds(self):
        cur = self.fault_kind.currentText()
        self.fault_kind.clear()
        d = self.fault_sensor.currentData()
        if d:
            self.fault_kind.addItems([k for k in FL.TIMED if FL.applies_to(k, d[1])])
        i = self.fault_kind.findText(cur)
        self.fault_kind.setCurrentIndex(max(i, 0))

    def faults(self) -> list:
        return [self.fault_list.topLevelItem(i).data(0, 256) for i in range(self.fault_list.topLevelItemCount())]

    def add_fault(self):
        from ...sim.stacksuite import parse_fault
        d = self.fault_sensor.currentData()
        if not d or not self.fault_kind.currentText():
            QMessageBox.information(self, 'Sensor faults', 'Choose a rig with sensors first.')
            return
        spec = f'{d[0]}:{self.fault_kind.currentText()}:{self.fault_onset.value():g}'
        opts = [x.strip() for x in self.fault_opts.text().split(',') if x.strip()]
        if self.fault_for.value() > 0:
            opts.append(f'for={self.fault_for.value():g}')
        if opts:
            spec += ':' + ','.join(opts)
        try:
            f = parse_fault(spec)
        except ValueError as e:
            QMessageBox.warning(self, 'Sensor faults', str(e))
            return
        errs = FL.validate([f])
        if errs:
            QMessageBox.warning(self, 'Sensor faults', '\n'.join(errs))
            return
        extra = ', '.join(f'{k}={v:g}' if isinstance(v, float) else f'{k}={v}' for k, v in f.items()
                          if k not in ('sensor', 'kind', 'onset_s', 'for_s'))
        it = QTreeWidgetItem([f['sensor'], f['kind'], f'{f["onset_s"]:g}',
                              f'{f["for_s"]:g}' if 'for_s' in f else 'to the end', extra])
        it.setData(0, 256, f)
        self.fault_list.addTopLevelItem(it)
        self.fault_opts.clear()

    def remove_fault(self):
        it = self.fault_list.currentItem()
        if it is not None:
            self.fault_list.takeTopLevelItem(self.fault_list.indexOfTopLevelItem(it))

    def suite_dir(self) -> str:
        ws = self.win.workspace()
        return os.path.join(ws, 'suites', self.name.text().strip() or 'suite') if ws else ''

    def save_suite(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save the suite', os.path.join(self.win.workspace() or '',
                                                                                   'suite.json'), 'JSON (*.json)')
        if path:
            write_json(path, self.suite_doc())

    def run_suite(self):
        w = self.win
        if any(j.state not in FINAL for j, kind in self.jobs.items() if kind == 'suite'):
            return
        doc = self.suite_doc()
        errs = ST.validate_suite(doc, w.project)
        if errs:
            QMessageBox.warning(self, 'Suite', '\n'.join(errs))
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        d = self.suite_dir()
        os.makedirs(d, exist_ok=True)
        spath = os.path.join(d, 'suite-request.json')
        write_json(spath, doc)
        argv = [sys.executable, '-m', 'falcon_autonomy'] + ST.suite_run_argv(w.project.path, spath, d, w.fake_carla)
        self.track(w.queue.submit(JobSpec('stack-suite', suite_title(doc), argv,
                                          memory_gb=0.5)), 'suite')
        w.jobs_dock.raise_()

    def track(self, job, kind):
        self.jobs[job] = kind
        self.state.set('State', f'{kind}: {job.summary()}')

    def poll(self):
        for j, kind in list(self.jobs.items()):
            if j.state not in FINAL:
                self.state.set('State', f'{kind}: {j.summary()}')
                if kind == 'suite':
                    self.load()

    def _finished(self, job):
        kind = self.jobs.get(job)
        if kind is None:
            return
        self.state.set('State', f'{kind}: {job.summary()}')
        if kind == 'image':
            self.check_bridge()
        self.load()

    def load(self):
        cur = self.selected_run()
        cur = cur['run'] if cur else None
        self.results.clear()
        d = self.suite_dir()
        rows = ST.read_results(d) if d and os.path.isdir(d) else []
        for r in sorted(rows, key=lambda r: (r['scenario'], r['seed'])):
            res = {True: 'passed', False: 'FAILED', None: r.get('status') or '?'}[r.get('passed')]
            it = QTreeWidgetItem([r.get('scenario_name', r['scenario']), str(r['seed']), res,
                                  '' if r.get('criticality') is None else f'{r["criticality"]:.3f}',
                                  '' if r.get('min_ttc_s') is None else f'{r["min_ttc_s"]:.2f}',
                                  '' if r.get('missed_deadlines') is None else str(r['missed_deadlines']),
                                  '' if not r.get('mrm') else ('passed' if r['mrm'].get('passed') else 'FAILED'),
                                  tilde(r['run'])])
            if r.get('mrm') and not r['mrm'].get('passed'):
                it.setToolTip(6, 'failed: ' + ', '.join(c['name'] for c in (r['mrm'].get('checks') or ())
                                                       if not c.get('ok')))
            it.setData(0, 256, r)
            self.results.addTopLevelItem(it)
            if r['run'] == cur:
                self.results.setCurrentItem(it)
        if not rows:
            for k in ('Runs', 'Pass rate (Wilson 95 %)', 'Collisions', 'Missed deadlines', 'Minimal-risk manoeuvre'):
                self.summary.set(k, '—')
            return
        o = ST.summarise(rows)['overall']
        self.summary.set('Runs', f'{o["runs"]} finished, {o["passed"]} passed'
                         + (f', {o["not_finished"]} did not finish' if o['not_finished'] else ''))
        ci = o['pass_rate_ci95']
        self.summary.set('Pass rate (Wilson 95 %)', '—' if o['pass_rate'] is None else
                         f'{o["pass_rate"]:.0%} ({ci[0]:.0%}–{ci[1]:.0%})')
        self.summary.set('Collisions', str(o['collisions']))
        self.summary.set('Missed deadlines', str(o['missed_deadlines']))
        m = ST.mrm_summary(rows)
        if not m:
            self.summary.set('Minimal-risk manoeuvre', 'not judged (no sensor faults with the check on)')
        else:
            g = m['overall']
            ci = g['pass_rate_ci95']
            fails = ', '.join(f'{k} ×{v}' for k, v in sorted(g['failed_checks'].items()))
            self.summary.set('Minimal-risk manoeuvre', f'{g["passed"]} of {g["runs"]} passed ({ci[0]:.0%}–{ci[1]:.0%})'
                             + (f'; failed: {fails}' if fails else ''))
        self.odd_summary(rows)

    def odd_summary(self, rows):
        """Time the suite's runs spent outside the chosen ODD (core.oddmonitor.suite_outside; cached per run)."""
        path = self.odd.text().strip()
        if not path or not os.path.isfile(path):
            self.summary.set('Outside the ODD', '— (choose an ODD)')
            return
        from ...core import oddmonitor as OM
        try:
            odd = OM.load_odd(path)
        except (OSError, ValueError, OM.OddError) as e:
            self.summary.set('Outside the ODD', f'— ({e})')
            return
        cache = self.__dict__.setdefault('_odd_cache', {})
        done = [r for r in rows if r.get('status') == 'ok' and r.get('run')]
        key = (path, os.path.getmtime(path), tuple(sorted(r['run'] for r in done)))
        if cache.get('key') != key:
            cache.update(key=key, res=OM.suite_outside(done, odd))
        res = cache['res']
        if not res['runs']:
            self.summary.set('Outside the ODD', '—')
            return
        worst = sorted(res['scenarios'].values(), key=lambda g: -g['time_outside_s'])[:2]
        self.summary.set('Outside the ODD', f'{res["time_outside_s"]:.1f} s of {res["time_s"]:.1f} s '
                         f'({res["share"]:.0%})' + ''.join(f'; {g["name"]} {g["time_outside_s"]:.1f} s'
                                                           for g in worst if g['time_outside_s']),
                         'condition monitors over each run\'s recorded frames (docs/odd-monitoring.md)')

    def selected_run(self):
        it = self.results.currentItem()
        return it.data(0, 256) if it is not None else None

    def replay(self):
        r = self.selected_run()
        if r is None:
            return
        into = r['run'].rstrip('/') + '-replay'
        if os.path.exists(into):
            QMessageBox.information(self, 'Replay', f'{tilde(into)} exists already.')
            return
        w = self.win
        argv = [sys.executable, '-m', 'falcon_autonomy', 'stack'] + (['--fake-carla', w.fake_carla] if w.fake_carla
                                                                     else []) + ['replay', r['run'], '--into', into]
        self.track(w.queue.submit(JobSpec('stack-replay', f'Replay {r["scenario"]} seed {r["seed"]}', argv,
                                          memory_gb=0.5)), 'replay')

    def open_run(self):
        r = self.selected_run()
        if r is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(r['run']))

    # ------------------------------------------------------------------ the safety case
    def choose_odd(self):
        path, _ = QFileDialog.getOpenFileName(self, 'ODD description', self.win.workspace() or '', 'JSON (*.json)')
        if path:
            self.odd.setText(path)

    def new_odd(self):
        from ...core.safetycase import example_odd
        ws = self.win.workspace()
        if not ws:
            return
        p = os.path.join(ws, 'odd.json')
        if not os.path.exists(p):
            write_json(p, example_odd())
        self.odd.setText(p)
        self.state.set('State', f'ODD: {tilde(p)} (edit it: every value of the example is illustrative)')

    def report_dir(self) -> str:
        d = self.suite_dir()
        return os.path.join(d, 'safety-case') if d else ''

    def build_report(self):
        d, odd = self.suite_dir(), self.odd.text().strip()
        if not d or not os.path.isfile(os.path.join(d, 'results.jsonl')):
            QMessageBox.information(self, 'Safety case', 'Run the suite first.')
            return
        if not os.path.isfile(odd):
            QMessageBox.information(self, 'Safety case', 'Choose an ODD description (or start from the example).')
            return
        try:
            from ...core.safetycase import validate_odd
            errs = validate_odd(read_json(odd))
        except (OSError, ValueError) as e:
            errs = [str(e)]
        if errs:
            QMessageBox.warning(self, 'ODD', '\n'.join(errs))
            return
        argv = [sys.executable, '-m', 'falcon_autonomy', 'stack', 'report', d, '--odd', odd, '--out',
                self.report_dir()]
        studies = os.path.join(self.win.workspace(), 'studies')
        if os.path.isdir(studies):
            for s in sorted(os.listdir(studies)):
                if os.path.isfile(os.path.join(studies, s, 'results.jsonl')):
                    argv += ['--study', os.path.join(studies, s)]
        self.track(self.win.queue.submit(JobSpec('stack-report', f'Safety case of {os.path.basename(d)}', argv,
                                                 memory_gb=0.5)), 'report')

    def open_report(self):
        p = os.path.join(self.report_dir(), 'safety-case.html')
        if os.path.isfile(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))
        else:
            self.state.set('State', 'no report yet: build it first')

    # ------------------------------------------------------------------ CI mode (core.ci; no posting from here)
    def ci_results(self) -> list:
        """The CI runs in <workspace>/ci/*/ci-results.json, newest first: [(folder, results)]."""
        if not self.win.workspace():
            return []
        base = os.path.join(self.win.workspace(), 'ci')
        out = []
        for d in (sorted(os.listdir(base)) if os.path.isdir(base) else ()):
            p = os.path.join(base, d, 'ci-results.json')
            try:
                out.append((os.path.join(base, d), read_json(p)))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda x: x[1].get('started') or '', reverse=True)

    def load_ci(self):
        self.ci_runs.clear()
        for d, r in self.ci_results():
            sha = (r.get('commit') or {}).get('sha') or ''
            bad = [g['name'] for g in r.get('gates') or () if g.get('status') in ('failed', 'error')]
            it = QTreeWidgetItem([r.get('started') or '', r.get('name') or os.path.basename(d),
                                  r.get('verdict') or '?', sha[:10], '; '.join(bad) or (r.get('error') or '')[:80]])
            it.setData(0, 256, d)
            it.setToolTip(4, '\n'.join(bad) or (r.get('error') or ''))
            self.ci_runs.addTopLevelItem(it)

    def open_ci_report(self):
        it = self.ci_runs.currentItem() or self.ci_runs.topLevelItem(0)
        if it is None:
            self.state.set('State', f'no CI results in {tilde(os.path.join(self.win.workspace() or "", "ci"))}')
            return
        p = os.path.join(it.data(0, 256), 'report.html')
        if os.path.isfile(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))
        else:
            self.state.set('State', f'{tilde(p)} was not written')

    def write_ci(self, github: bool, target: str = '') -> list:
        """A CI config and suite (and with github the workflow) into a repository folder the user chooses."""
        from ...core.ci import CIError
        from ...core.ci_github import init
        target = target or QFileDialog.getExistingDirectory(self, 'The repository to write the CI files into')
        if not target:
            return []
        try:
            written = init(target, 'stack', github)
        except (CIError, OSError, ValueError) as e:
            QMessageBox.warning(self, 'CI', str(e))
            return []
        self.state.set('State', ('wrote ' + ', '.join(tilde(w) for w in written)) if written else
                       'nothing written: the files exist (edit them, or use `falcon ci init --force`)')
        return written

    def summary_json(self) -> str:
        """The current suite's results.json (for tests and the log)."""
        p = os.path.join(self.suite_dir(), 'results.json')
        return json.dumps(read_json(p)) if os.path.isfile(p) else ''
