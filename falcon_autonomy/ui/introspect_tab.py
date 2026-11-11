"""The Models page's Introspection tab: what a token driving policy attends to (core.introspect), what changes when a
token, a head or an input is intervened on, and closed-loop bias audits of any ego (core.biasaudit). Every action is
a job (`falcon introspect …`, `falcon policy bc --arch token`) whose outputs land in the workspace's introspect/
folder: attention per frame with its tokens and the patch points in 3D (ui/view3d.py), intervention deltas, and the
audits' reliance ratios with their intervals. docs/introspection.md and docs/bias-audit.md are the guides."""
from __future__ import annotations

import glob
import json
import os
import sys
import time

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QSpinBox,
                             QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.jobs import FINAL, JobSpec
from .widgets import Form, KeyValues, Section, button, scroll

SCENARIOS = ('lead_brake', 'cut_in', 'autorickshaw_cut_in', 'free')
OPS = (('Mask the autorickshaws\' tokens', {'op': 'mask_class', 'class': 'autorickshaw'}),
       ('Turn every brake light off (flip)', {'op': 'flip', 'feature': 'brake_light', 'all': True}),
       ('Shift the speed history by −3 m/s', {'op': 'speed_history', 'offset': -3.0}),
       ('Hide the image patches', {'op': 'mask_tokens', 'type': 'patch'}),
       ('Zero a head (layer and head below)', None))
ARMS = (('brake_light', 'Brake-light shortcut'), ('velocity', 'Velocity extrapolation'),
        ('autorickshaw', 'Autorickshaw token'))


def _fmt(v, nd=3):
    if v is None:
        return '—'
    if isinstance(v, list):
        return '[' + ', '.join(_fmt(x, nd) for x in v) + ']'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


class IntrospectTab(QWidget):
    """Controls on the left, results on the right (module doc)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = getattr(page, 'win', None)        # a page without a window (a test, a preview): results only
        self.jobs: dict = {}
        self.attn: dict = {}
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        # ---- the policy and its attention
        sec = Section('Token policy', 'A small transformer over the camera\'s patches, the road users as object '
                                      'tokens, the route and the ego — cloned from a privileged expert in the '
                                      'environment. Record an episode to see what it attends to, per frame, in 3D.')
        f = Form()
        self.policy = QComboBox()
        self.scenario = QComboBox()
        for s in SCENARIOS:
            self.scenario.addItem(s, s)
        self.episodes = QSpinBox()
        self.episodes.setRange(1, 1000)
        self.episodes.setValue(12)
        f.row('Policy', self.policy)
        f.row('Scenario', self.scenario)
        f.row('Training episodes', self.episodes)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Record and attend', self.record, role='primary',
                           tip='falcon introspect attention (a job): an episode, its frames and attention'))
        b.addWidget(button('Train a token policy', self.train, tip='falcon policy bc --arch token (a job)'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        # ---- interventions
        sec = Section('Interventions', 'Change one thing on every frame of the last recording and compare the actions: '
                                       'the steering and acceleration deltas say what the policy relies on.')
        f = Form()
        self.op = QComboBox()
        for text, op in OPS:
            self.op.addItem(text, op)
        self.layer = QSpinBox()
        self.layer.setRange(0, 15)
        self.head = QSpinBox()
        self.head.setRange(0, 31)
        f.row('Intervention', self.op)
        f.row('Layer', self.layer)
        f.row('Head', self.head)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Intervene', self.intervene, tip='falcon introspect intervene (a job)'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        # ---- audits
        sec = Section('Bias audit', 'Closed-loop arms with common random numbers: does the ego brake for the brake '
                                    'light or the deceleration, extrapolate its speed history, see the autorickshaw '
                                    'only through its token? Black-box egos get the simulation-side arm.')
        f = Form()
        self.ego = QComboBox()
        self.seeds = QSpinBox()
        self.seeds.setRange(2, 500)
        self.seeds.setValue(6)
        f.row('Ego', self.ego)
        f.row('Seeds per cell', self.seeds)
        sec.add(f)
        self.arm_boxes = {}
        row = QHBoxLayout()
        for key, text in ARMS:
            cb = QCheckBox(text)
            cb.setChecked(True)
            self.arm_boxes[key] = cb
            row.addWidget(cb)
        row.addStretch(1)
        sec.add(row)
        b = QHBoxLayout()
        b.addWidget(button('Run the audit', self.audit, tip='falcon introspect audit (a job)'))
        b.addWidget(button('Refresh', self.refresh))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        # ---- results
        right = QWidget()
        rv = QVBoxLayout(right)
        self.state = KeyValues(('Attention', 'Intervention', 'Audit'))
        rv.addWidget(self.state)
        self.frames = QListWidget()
        self.frames.setMaximumHeight(110)
        self.frames.currentItemChanged.connect(lambda *_: self.show_frame())
        rv.addWidget(self.frames)
        from .view3d import make_view
        self.view = make_view()
        rv.addWidget(self.view, 2)
        self.tokens = QTreeWidget()
        self.tokens.setHeaderLabels(['Token', 'Object', 'Relevance', 'Plan attention'])
        self.tokens.setRootIsDecorated(False)
        rv.addWidget(self.tokens, 1)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(['Arm', 'Outcome', 'Cue effect [95 %]', 'Cause effect [95 %]', 'R [95 %]',
                                    'Flagged'])
        self.table.setRootIsDecorated(False)
        rv.addWidget(self.table, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 2)
        if self.win is not None:
            self.win.bridge.finished.connect(self.finished)
        self.refresh()

    # ------------------------------------------------------------------ folders
    def root(self, *parts) -> str:
        return os.path.join(self.win.workspace(), 'introspect', *parts)

    def latest(self, pattern: str) -> str:
        found = sorted(glob.glob(self.root(pattern)))
        return found[-1] if found else ''

    def token_policies(self) -> list:
        out = []
        for p in sorted(glob.glob(os.path.join(self.win.workspace(), 'policies', '*', 'policy.json'))):
            try:
                with open(p, encoding='utf-8') as fh:
                    if json.load(fh).get('net', {}).get('kind') == 'token':
                        out.append(os.path.dirname(p))
            except (OSError, ValueError):
                continue
        return out

    def all_policies(self) -> list:
        return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(self.win.workspace(), 'policies', '*',
                                                                         'policy.json')))

    def refresh(self):
        if self.win is None:
            self.state.set('Attention', 'no window: open a project in Falcon')
            return
        ws = self.win.workspace()
        for combo, items in ((self.policy, [(os.path.relpath(d, ws), d) for d in self.token_policies()]),
                             (self.ego, [(os.path.relpath(d, ws), d) for d in self.all_policies()] +
                              [('Lane follower (black box)', 'follower'), ('Constant speed (black box)', 'none')])):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for text, data in items:
                combo.addItem(text, data)
            k = combo.findData(cur)
            combo.setCurrentIndex(k if k >= 0 else 0)
            combo.blockSignals(False)
        self.load_attention()
        self.load_audit()
        iv = self.latest('intervene-*')
        self.state.set('Intervention', self.intervention_text(iv) if iv else 'none yet')

    # ------------------------------------------------------------------ results
    def load_attention(self):
        d = self.latest('attention-*')
        self.attn = {}
        self.frames.clear()
        if d and os.path.isfile(os.path.join(d, 'attention.json')):
            try:
                with open(os.path.join(d, 'attention.json'), encoding='utf-8') as fh:
                    self.attn = json.load(fh)
            except (OSError, ValueError):
                self.attn = {}
        rows = self.attn.get('frames') or []
        for r in rows:
            it = QListWidgetItem(f'frame {r["frame"]}: steer {r["action"][0]:+.3f}, accel {r["action"][1]:+.3f}')
            it.setData(256, r)
            self.frames.addItem(it)
        self.state.set('Attention', f'{len(rows)} frames in {os.path.relpath(d, self.win.workspace())}' if rows
                       else 'none yet: record and attend')
        if rows:
            self.frames.setCurrentRow(0)
        else:
            self.view.set_overlay(None)

    def show_frame(self):
        it = self.frames.currentItem()
        self.tokens.clear()
        if it is None:
            return
        r = it.data(256)
        self.view.set_overlay(r.get('points') or [], r.get('objects') or [])
        for t in sorted(r.get('tokens') or [], key=lambda x: -x.get('relevance', 0.0))[:40]:
            obj = '' if t['type'] != 'object' else (f'{t.get("class")} #{t.get("id")}' +
                                                    (' (brake light)' if t.get('brake_light') else ''))
            if t['type'] == 'patch':
                obj = f'row {t["row"]}, column {t["col"]}'
            self.tokens.addTopLevelItem(QTreeWidgetItem([t['type'], obj, f'{t.get("relevance", 0.0):.4f}',
                                                         f'{t.get("plan_attention", 0.0):.4f}']))

    def intervention_text(self, d) -> str:
        try:
            with open(os.path.join(d, 'intervene.json'), encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            return 'unreadable'
        s = doc['summary']
        return (f'{s["frames"]} frames: mean |Δsteer| {_fmt(s["steer"]["mean_abs_delta"])}, mean |Δaccel| '
                f'{_fmt(s["accel"]["mean_abs_delta"])} ({", ".join(o["op"] for o in doc["ops"])})')

    def load_audit(self):
        self.table.clear()
        d = self.latest('audit-*')
        if not d or not os.path.isfile(os.path.join(d, 'audit.json')):
            self.state.set('Audit', 'none yet')
            return
        try:
            with open(os.path.join(d, 'audit.json'), encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            self.state.set('Audit', 'unreadable')
            return
        for arm, v in doc.get('arms', {}).items():
            if not v.get('available'):
                self.table.addTopLevelItem(QTreeWidgetItem([arm, '—', v.get('why', ''), '', '', '']))
                continue
            for m, o in v['outcomes'].items():
                self.table.addTopLevelItem(QTreeWidgetItem([
                    arm, m, f'{_fmt(o["cue_effect"]["mean"])} {_fmt(o["cue_effect"]["ci95"])}',
                    f'{_fmt(o["cause_effect"]["mean"])} {_fmt(o["cause_effect"]["ci95"])}',
                    f'{_fmt(o["R"])} {_fmt(o["R_ci95"])}', 'yes' if o['flagged'] else 'no']))
        flagged = doc.get('flagged') or []
        self.state.set('Audit', f'{os.path.basename(str(doc.get("ego")))}: ' +
                       (f'shortcut flagged on {", ".join(flagged)}' if flagged else 'nothing flagged') +
                       f' ({len(doc.get("seeds") or ())} seeds per cell)')

    # ------------------------------------------------------------------ jobs
    def _ready(self) -> bool:
        w = self.win
        if w is None:
            return False
        if not w.project.path or w.project.dirty:
            return w.save()
        return True

    def _submit(self, title, argv, kind='introspect'):
        w = self.win
        if w.fake_carla and argv[0] != 'intervene':
            argv = argv + ['--fake-carla', w.fake_carla]
        job = w.queue.submit(JobSpec(kind, title, [sys.executable, '-m', 'falcon_autonomy'] + argv, memory_gb=2.0))
        self.jobs[job.id] = job
        return job

    def record(self):
        if self.win is None:                         # no window: nothing to run (and no dialog to block on)
            return
        pol = self.policy.currentData()
        if not pol:
            QMessageBox.information(self, 'Record', 'Train a token policy first: no token policy in the workspace.')
            return
        if not self._ready():
            return
        out = self.root(f'attention-{time.strftime("%Y%m%d-%H%M%S")}')
        self._submit(f'Attention of {os.path.basename(pol)}',
                     ['introspect', 'attention', self.win.project.path, '--policy', pol, '--scenario',
                      self.scenario.currentData(), '--out', out])

    def train(self):
        if self.win is None or not self._ready():
            return
        out = os.path.join(self.win.workspace(), 'policies', f'token-{time.strftime("%Y%m%d-%H%M%S")}')
        self._submit('Train a token policy', ['policy', 'bc', '--arch', 'token', self.win.project.path, '--out', out,
                                              '--episodes', str(self.episodes.value())], kind='policy')

    def selected_op(self) -> dict:
        op = self.op.currentData()
        return op if op is not None else {'op': 'zero_head', 'layer': self.layer.value(), 'head': self.head.value()}

    def intervene(self):
        if self.win is None:
            return
        d = self.latest('attention-*')
        pol = (self.attn or {}).get('policy') or self.policy.currentData()
        if not d or not os.path.isfile(os.path.join(d, 'frames.npz')) or not pol:
            QMessageBox.information(self, 'Intervene', 'Record and attend first: the intervention runs on its frames.')
            return
        out = self.root(f'intervene-{time.strftime("%Y%m%d-%H%M%S")}')
        self._submit(f'Intervene on {os.path.basename(pol)}',
                     ['introspect', 'intervene', '--policy', pol, '--frames', os.path.join(d, 'frames.npz'), '--op',
                      json.dumps(self.selected_op()), '--out', out])

    def audit(self):
        if self.win is None:
            return
        arms = [k for k, cb in self.arm_boxes.items() if cb.isChecked()]
        if not arms:
            QMessageBox.information(self, 'Audit', 'Choose at least one arm.')
            return
        if not self._ready():
            return
        out = self.root(f'audit-{time.strftime("%Y%m%d-%H%M%S")}')
        self._submit(f'Bias audit of {os.path.basename(str(self.ego.currentData()))}',
                     ['introspect', 'audit', self.win.project.path, '--policy', self.ego.currentData(), '--out', out,
                      '--arms', ','.join(arms), '--seeds', str(self.seeds.value())])

    def finished(self, job):
        if job.id in self.jobs:
            self.jobs.pop(job.id)
            self.refresh()

    def busy(self) -> bool:
        return any(j.state not in FINAL for j in self.jobs.values())
