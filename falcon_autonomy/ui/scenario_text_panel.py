"""Scenarios ▸ Describe a scenario: a description in words → a scenario written by the assistant's model (or Falcon's
draft from the phrases it recognises, offline) → its preview (the JSON, the validation rounds, what the description
says that the scenario does not) → added and run once → the run checked against the description, one row per claim
(core.scenario_text, core.scenario_check; docs/scenario-text.md)."""
from __future__ import annotations

import json
import os

from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QPlainTextEdit, QTreeWidget, QTreeWidgetItem

from .widgets import Section, button, label, set_enabled

STATUS_TEXT = {'met': 'met', 'not_met': 'not met', 'not_checkable': 'not checkable'}
EXAMPLE = 'an autorickshaw cuts in from the left at a speed breaker at dusk'


class DescribeSection(Section):
    def __init__(self, page):
        super().__init__('Describe a scenario',
                         'Say what should happen; the assistant\'s model writes it as a Falcon scenario (only this '
                         'text, the scenario format and validation errors are sent), Falcon validates it and repairs '
                         'it in up to two rounds, runs it once and checks the run against each claim of the '
                         'description. Offline, Falcon drafts it from the phrases it recognises.')
        self.page, self.win = page, page.win
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText(f'e.g. {EXAMPLE}')
        self.text.setMaximumHeight(70)
        self.add(self.text)
        row = QHBoxLayout()
        self.write_btn = button('Write the scenario', self.write, role='primary')
        row.addWidget(self.write_btn)
        self.offline = QCheckBox('Falcon\'s draft only')
        self.offline.setToolTip('No model: the scenario from the phrases Falcon recognises (a library scenario with '
                                'the kind, side, light, weather and speed breaker the text names)')
        row.addWidget(self.offline)
        row.addStretch(1)
        self.add(row)
        self.status = label('', muted=True, wrap=True, small=True)
        self.add(self.status)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(150)
        self.preview.setPlaceholderText('The scenario appears here before anything is added or run.')
        self.add(self.preview)
        row2 = QHBoxLayout()
        self.run_btn = button('Add and run', self.add_and_run, tip='Add the scenario to the project (undoable) and '
                                                                   'run it once with the world, rig and seed above')
        row2.addWidget(self.run_btn)
        self.check_btn = button('Check the last run', self.check_last, tip='Check the newest run of this scenario '
                                                                           'against the description')
        row2.addWidget(self.check_btn)
        row2.addStretch(1)
        self.add(row2)
        self.claims = QTreeWidget()
        self.claims.setHeaderLabels(['Claim', 'From the text', 'Result', 'Measured'])
        self.claims.setRootIsDecorated(False)
        self.add(self.claims)
        self.record = None              # the last falcon.scenario_text/1 record
        self.pending_run = ''           # the run folder of the job "Add and run" queued
        self.report = None
        set_enabled(self.run_btn, False, 'Write a scenario first')
        set_enabled(self.check_btn, False, 'Write a scenario first')
        self.win.bridge.finished.connect(self._job_finished)

    # ------------------------------------------------------------------ writing
    def description(self) -> str:
        return ' '.join(self.text.toPlainText().split())

    def writer(self):
        """(complete, why it is offline): the assistant's main model when it is ready and not switched off here."""
        if self.offline.isChecked():
            return None, 'Falcon\'s draft (offline, as asked)'
        svc = getattr(self.win, 'assistant', None)
        if svc is None:
            return None, 'Falcon\'s draft (no assistant)'
        ok, why = svc.ready()
        if not ok:
            return None, f'Falcon\'s draft — {why}'
        return svc.scenario_writer(self.description()), ''

    def write(self, wait: bool = False):
        """Write the scenario in the background (wait: in this thread, for the tests); returns the record when
        waited for."""
        from ..core import scenario_text as T
        from . import tasks
        d = self.description()
        if not d:
            self.status.setText('Describe the scenario first.')
            return None
        world = self.page.world.currentData()
        hand = T.world_handedness(self.win.project, world)
        try:
            complete, why = self.writer()
        except Exception as e:                  # noqa: BLE001 (a refused description: a sentence)
            self.status.setText(f'Not sent: {getattr(e, "message", None) or e}')
            return None
        self.status.setText(('Writing it with ' + getattr(complete, 'model', 'the model') + '…') if complete else
                            f'Drafting: {why}.')
        set_enabled(self.write_btn, False, 'Writing…')

        def job():
            rec = T.write(d, complete, hand)
            if rec.get('scenario'):
                rec['notes'] = T.plan_notes(rec['scenario'])
            if self.win.workspace():
                T.save_record(self.win.workspace(), rec)
            return rec
        if wait:
            try:
                rec = job()
            except Exception as e:              # noqa: BLE001
                self._failed(e)
                return None
            self.show_record(rec)
            return rec
        tasks.run(self, job, self.show_record, self._failed)
        return None

    def _failed(self, e):
        set_enabled(self.write_btn, True)
        self.status.setText(f'Could not write it: {getattr(e, "message", None) or e}')

    def show_record(self, rec: dict):
        from ..core.scenario_check import claim_text
        set_enabled(self.write_btn, True)
        self.record = rec
        scn = rec.get('scenario')
        n = len(rec['attempts'])
        head = {'valid': 'Valid', 'valid_with_mismatches': 'Valid, but it does not say everything the text says',
                'invalid': 'Not valid', 'none': 'No scenario'}[rec['status']]
        by = 'the model' if rec['mode'] == 'model' else 'Falcon\'s draft'
        bits = [f'{head} ({by}, {n} attempt{"s" if n != 1 else ""}).']
        bits += [f'Mismatch: {m}' for m in rec.get('mismatches') or ()]
        bits += [f'Problem: {p}' for p in (rec.get('problems') or ())[:6]]
        bits += [f'Note: {x}' for x in rec.get('notes') or ()]
        if rec.get('unrecognised'):
            bits.append('Not recognised: ' + ', '.join(rec['unrecognised']))
        if rec['sent']['requests']:
            bits.append(f'Sent: {rec["sent"]["requests"]} request(s), {rec["sent"]["characters"]:,} characters of '
                        f'text.')
        self.status.setText('\n'.join(bits))
        self.preview.setPlainText(json.dumps(scn, indent=2) if scn else '')
        self.claims.clear()
        for c in rec['claims']:
            self.claims.addTopLevelItem(QTreeWidgetItem([claim_text(c), c.get('phrase', ''), 'not run yet', '']))
        set_enabled(self.run_btn, scn is not None, 'No valid scenario')
        set_enabled(self.check_btn, scn is not None and self._last_run() is not None, 'No run of it yet')

    # ------------------------------------------------------------------ running and checking
    def add_and_run(self):
        """Add the scenario (once) and queue its run; the claims are checked when the run ends."""
        from ..core.drive import run_dir_name, scenario_config
        from ..core.project import Edit, ProjectError
        scn = (self.record or {}).get('scenario')
        if not scn:
            return None
        w = self.win
        if w.project.item('scenarios', scn['id']) is None:
            w.edit(Edit('insert', ['scenarios'], scn, text=f'Add the scenario {scn["name"]} (from a description)'))
        out = os.path.join(w.service.runs_dir(), run_dir_name(f'scenario-{scn["id"]}'))
        try:
            cfg = scenario_config(w.project, scn, self.page.world.currentData(), self.page.rig.currentData(), out,
                                  w.service.port, w.service.tm_port, seed=self.page.seed.value(),
                                  render_budget_mpx=float(w.settings.get('render_budget_mpx')),
                                  workspace=w.workspace(), trigger_rules='project')
            job = w.service.submit_drive(cfg, title=f'{scn["name"]} (seed {self.page.seed.value()})')
        except (ProjectError, RuntimeError, ValueError, OSError) as e:
            self.status.setText(f'Added, not run: {e}')
            return None
        self.pending_run = out
        self.status.setText(f'Running {scn["name"]}: the claims are checked when the run ends.')
        return job

    def _job_finished(self, job):
        rd = getattr(job.spec, 'run_dir', '')
        if self.pending_run and rd and os.path.realpath(rd) == os.path.realpath(self.pending_run):
            run, self.pending_run = self.pending_run, ''
            if job.state == 'done':
                self.check(run)
            else:
                self.status.setText(f'The run ended {job.state}: nothing to check.')

    def _last_run(self) -> str | None:
        scn = (self.record or {}).get('scenario')
        rd = self.win.service.runs_dir() if scn else ''
        if not scn or not rd or not os.path.isdir(rd):
            return None
        best = None
        for name in sorted(os.listdir(rd)):
            p = os.path.join(rd, name, 'drive.json')
            if name.endswith(f'scenario-{scn["id"]}') and os.path.isfile(os.path.join(rd, name, 'run.json')) \
                    and os.path.isfile(p):
                best = os.path.join(rd, name)
        return best

    def check_last(self):
        run = self._last_run()
        if run:
            return self.check(run)
        self.status.setText('No finished run of this scenario yet.')
        return None

    def check(self, run_dir: str):
        """Check a run against the description and fill the claims table."""
        from ..core import scenario_check as CK
        d = (self.record or {}).get('description') or self.description()
        model = [c for c in (self.record or {}).get('claims') or () if c.get('source') == 'model']
        try:
            rep = CK.check_run(run_dir, d, model_claims=model)
        except (CK.CheckError, OSError, ValueError) as e:
            self.status.setText(f'Cannot check {os.path.basename(run_dir)}: {e}')
            return None
        self.report = rep
        self.claims.clear()
        for r in rep['claims']:
            it = QTreeWidgetItem([r['claim'], r.get('phrase', ''), STATUS_TEXT[r['status']],
                                  r['value'] or r.get('why', '')])
            it.setToolTip(3, json.dumps(r.get('measured'), indent=1, default=str)[:1500])
            self.claims.addTopLevelItem(it)
        self.status.setText(f'{os.path.basename(run_dir)}: {rep["verdict"]}.')
        set_enabled(self.check_btn, True)
        return rep
