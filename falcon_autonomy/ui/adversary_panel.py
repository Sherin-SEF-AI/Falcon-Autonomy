"""The Rare cases page's section for learned adversarial traffic (core.adversary): train adversaries against an ego
(the lane follower, constant speed, CARLA's autopilot or a driving policy of the workspace), evaluate them beside the
baselines with the same budget, and list every realistic failure they found as a case — each replayable with a chase
camera (a job; the replay checks that the same failure happens at the same step). docs/adversarial-traffic.md."""
from __future__ import annotations

import glob
import json
import os
import sys
import time

from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QMessageBox, QSpinBox, QTreeWidget, QTreeWidgetItem,
                             QWidget)

from ..core import advfeatures as AF
from ..core.jobs import FINAL, JobSpec
from .widgets import Form, KeyValues, Section, button

EGOS = (('follower', 'Lane follower'), ('none', 'Constant speed (no braking)'), ('autopilot', 'CARLA autopilot'))


class AdversarySection(Section):
    """Train, evaluate, and the adversarial cases with their replay (module doc)."""

    def __init__(self, page):
        super().__init__('Adversarial traffic (learned)',
                         'Road users trained with multi-agent PPO to make the ego fail — realistically: an adversary '
                         'that causes a collision itself (by fault attribution), leaves the lanes, drives the wrong '
                         'way beyond its role, exceeds the dynamics bounds or breaks the traffic rules its role does '
                         'not allow makes the episode unrealistic, and its failures are not counted. The evaluation '
                         'compares the learned adversaries with random and idle ones and with the scripted scenarios '
                         'searched with the same budget.')
        self.page = page
        self.win = page.win
        self.jobs: dict = {}
        f = Form()
        self.ego = QComboBox()
        self.action = QComboBox()
        for k in AF.ACTION_KINDS:
            self.action.addItem({'lateral': 'Lateral and acceleration commands', 'discrete': 'Manoeuvres (9)',
                                 'continuous': 'Steer and acceleration'}[k], k)
        self.updates = QSpinBox()
        self.updates.setRange(1, 100000)
        self.updates.setValue(50)
        self.episodes = QSpinBox()
        self.episodes.setRange(1, 10000)
        self.episodes.setValue(20)
        self.policy = QComboBox()
        for text, w in (('Ego under test', self.ego), ('Adversary actions', self.action), ('PPO updates', self.updates),
                        ('Trained adversaries', self.policy), ('Episodes per method', self.episodes)):
            f.row(text, w)
        self.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Train adversaries', self.train, role='primary', tip='falcon adversary train (a job)'))
        b.addWidget(button('Evaluate', self.evaluate, tip='falcon adversary eval: learned vs random, idle and the '
                                                          'searched scripted scenarios (a job)'))
        b.addStretch(1)
        self.add(b)
        self.state = KeyValues(('Cases',))
        self.add(self.state)
        self.cases = QTreeWidget()
        self.cases.setHeaderLabels(['#', 'Ego', 'Method', 'Failure', 'Fault', 'Role', 'Step', 'Evaluation'])
        self.cases.setRootIsDecorated(False)
        self.cases.setMinimumHeight(140)
        self.add(self.cases)
        r = QHBoxLayout()
        r.addWidget(button('Replay the selected case', self.replay))
        r.addWidget(button('Refresh', self.refresh))
        r.addStretch(1)
        self.add(r)
        self.refresh()

    # ------------------------------------------------------------------ folders
    def root(self, *parts) -> str:
        return os.path.join(self.win.workspace(), 'adversary', *parts)

    def ego_policies(self) -> list:
        ws = self.win.workspace()
        return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(ws, 'policies', '*', 'policy.json')) +
                       glob.glob(os.path.join(ws, 'policies', '*', 'policy', 'policy.json'))}, reverse=True)

    def adversary_policies(self) -> list:
        return sorted({os.path.dirname(p) for p in glob.glob(self.root('*', 'policy', 'policy.json'))},
                      reverse=True)

    def refresh(self):
        for combo, items in ((self.ego, [(t, k) for k, t in EGOS] +
                              [(f'Policy: {os.path.relpath(d, self.win.workspace())}', d)
                               for d in self.ego_policies()]),
                             (self.policy, [(os.path.relpath(d, self.root()), d) for d in self.adversary_policies()])):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for text, data in items:
                combo.addItem(text, data)
            k = combo.findData(cur)
            combo.setCurrentIndex(k if k >= 0 else 0)
            combo.blockSignals(False)
        from ..core.adversary import list_cases
        self.cases.clear()
        rows = list_cases(self.win.workspace()) if self.win.project.path else []
        for k, c in enumerate(rows, 1):
            it = QTreeWidgetItem([str(k), os.path.basename(str(c.get('ego'))), str(c.get('method')), str(c.get('kind')),
                                  str(c.get('fault')), str(c.get('role') or '—'), str(c.get('step')),
                                  os.path.basename(c['eval_dir'])])
            it.setData(0, 256, c)
            self.cases.addTopLevelItem(it)
        self.state.set('Cases', f'{len(rows)} realistic failures in {len({c["eval_dir"] for c in rows})} evaluations'
                       if rows else 'none yet: train adversaries, then evaluate them')

    # ------------------------------------------------------------------ jobs
    def _ready(self) -> bool:
        w = self.win
        if not w.project.path or w.project.dirty:
            return w.save()
        return True

    def _submit(self, title, argv, memory_gb=1.0):
        w = self.win
        if w.fake_carla:
            argv = argv + ['--fake-carla', w.fake_carla]
        job = w.queue.submit(JobSpec('adversary', title, [sys.executable, '-m', 'falcon_autonomy', 'adversary'] + argv,
                                     memory_gb=memory_gb))
        self.jobs[job.id] = job
        w.jobs_dock.raise_()
        return job

    def train(self):
        if not self._ready():
            return
        out = self.root(f'train-{time.strftime("%Y%m%d-%H%M%S")}')
        self._submit(f'Train adversaries against {self.ego.currentText()}',
                     ['train', self.win.project.path, '--out', out, '--ego', self.ego.currentData(), '--action',
                      self.action.currentData(), '--updates', str(self.updates.value())])

    def evaluate(self):
        if not self._ready():
            return
        pol = self.policy.currentData()
        if not pol:
            QMessageBox.information(self, 'Evaluate', 'Train adversaries first: no adversary policy in the workspace.')
            return
        out = self.root(f'eval-{time.strftime("%Y%m%d-%H%M%S")}')
        self._submit(f'Evaluate adversaries ({os.path.basename(os.path.dirname(pol))}) against '
                     f'{self.ego.currentText()}',
                     ['eval', self.win.project.path, '--policy', pol, '--out', out, '--ego', self.ego.currentData(),
                      '--episodes', str(self.episodes.value())])

    def selected(self):
        it = self.cases.currentItem()
        return it.data(0, 256) if it is not None else None

    def replay(self):
        c = self.selected()
        if c is None:
            QMessageBox.information(self, 'Replay', 'Choose a case first.')
            return
        if not c.get('replayable', True):
            QMessageBox.information(self, 'Replay', 'This episode cannot be replayed exactly (physics, background '
                                                    'traffic or the autopilot ego).')
            return
        if not self._ready():
            return
        out = self.root('replays', f'{os.path.basename(c["eval_dir"])}-{c["line"]}-{c["failure"]}')
        self._submit(f'Replay adversarial case: {c.get("kind")} ({c.get("role") or "—"})',
                     ['replay', self.win.project.path, '--episodes', c['episodes'], '--episode', str(c['line']),
                      '--failure', str(c['failure']), '--out', out])

    def finished(self, job):
        if job.id in self.jobs:
            self.jobs.pop(job.id)
            self.refresh()
            if job.state == 'done' and job.spec.argv[4:5] == ['replay']:
                self.show_replay(job)

    def show_replay(self, job):
        """The replay's result: whether it met the same failure, and where its frames are."""
        argv = job.spec.argv
        out = argv[argv.index('--out') + 1]
        try:
            with open(os.path.join(out, 'replay.json'), encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            return
        self.state.set('Last replay', f'{len(doc.get("frames") or ())} frames in {out}; the same failure at the same '
                                      f'step: {"yes" if doc.get("match") else "no"}')

    def busy(self) -> bool:
        return any(j.state not in FINAL for j in self.jobs.values())


def attach(page) -> QWidget:
    """The section, created for a Rare cases page (connected to the bridge's finished signal)."""
    sec = AdversarySection(page)
    page.win.bridge.finished.connect(sec.finished)
    return sec
