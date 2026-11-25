"""Reports ▸ Release gate: is the new version better, worse or no different from the old one on the same scenarios
(core.releasegate; docs/release-gate.md).

Pick the old version A and the new version B — a baseline driver (follower, autopilot), a ROS 2 stack by its suite
file (stack:SUITE.json) or a registered detector (model:ID) — the scenarios of the tested distribution, the trial
budget, the error probability and the margin. *Run the gate* queues `falcon gate run` as a job (resumable): batches of
paired trials, the same scenario instance and seed under both versions, until the verdict is clear. While it runs a
sweep shows the trials against the budget. The result: the verdict as a word with its colour, the confidence band of
the primary difference over the trials (one axis; zero solid, the margin dashed; it holds at every look, so the gate
may stop the moment it clears zero or sits inside the margin), and the scenarios the new version newly fails, each
with its pair of runs. A verdict holds for the tested scenario distribution only — the section says so under it.

4.2: each version's fidelity as a chip (core.fidelity, from its trials' runs); two versions at different fidelity get
no verdict — the verdict reads "no verdict: different fidelity" with the reason in the critical colour — and a version
whose own trials mix levels keeps its verdict with a critical *Mixed fidelity* chip and a sentence. Each newly failing
row has *Compare in Replay* (the Replay page's compare_runs, when this build has it), short copyable run ids, the pair's
fidelity, and *Send to ▸* in its context menu; selecting a row makes its B run the window's active run."""
from __future__ import annotations

import os
import re
import sys

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QListWidget,
                             QListWidgetItem, QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import fidelity as FD
from ..core import releasegate as G
from ..core.jobs import FINAL, JobSpec
from . import motion
from .placement import PlacementCombo
from .study_bits import Badges, FidelitySlot, link_runs
from .theme import THEME, mono
from .widgets import Details, Form, Stat, button, callout, copyable_ids, label, short_id


class Band(QWidget):
    """The confidence band of the primary difference (B − A) over the paired trials: the lower and upper bound as
    lines with the band between them, zero as a solid line, ±margin dashed, the trial at which the verdict held
    marked. One axis. Hover reads the bounds at a trial; new trials draw in."""

    def __init__(self):
        super().__init__()
        self.lo, self.hi = [], []
        self.margin, self.decided, self.verdict, self.what, self.by = 0.05, None, 'cant_tell', 'failure', None
        self.hover = -1
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.draw = motion.Driver(self)

    def set_gate(self, d: dict | None):
        lo, hi = [], []
        if d:
            key = 'risk' if d.get('primary', 'failure') == 'failure' else 'value'
            lo, hi = list(d['path'][f'{key}_lower']), list(d['path'][f'{key}_upper'])
            self.margin, self.decided, self.verdict = float(d['margin']), d.get('decided_at'), d['verdict']
            self.what, self.by = d.get('primary', 'failure'), d.get('decided_by')
        if len(lo) != len(self.lo) and lo:
            self.draw.start()
        self.lo, self.hi = lo, hi
        self.hover = -1
        self.update()

    def _geom(self):
        return 52.0, 14.0, self.width() - 70.0, self.height() - 50.0

    def _span(self) -> float:
        """The axis' half range: one and a half times the band where it ended, the margin with room; the wide
        first trials run off the plot's edge (they are ±1 by construction)."""
        m = max(abs(self.lo[-1]), abs(self.hi[-1])) * 1.5
        return min(1.0, max(m, self.margin * 2.0, 0.05))

    def mouseMoveEvent(self, ev):                               # noqa: N802
        n = len(self.lo)
        if not n:
            return
        x0, _y0, w, _h = self._geom()
        i = int(round((ev.position().x() - x0) / max(w, 1) * max(n - 1, 1)))
        i = min(max(i, 0), n - 1)
        if i != self.hover:
            self.hover = i
            self.setToolTip(f'after {i + 1} paired trials: the difference is in [{self.lo[i]:+.3f}, '
                            f'{self.hi[i]:+.3f}]' + ('\nthe verdict held here' if self.decided == i + 1 else ''))
            self.update()

    def leaveEvent(self, _ev):                                  # noqa: N802
        self.hover = -1
        self.update()

    def paintEvent(self, _ev):                                  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self.lo)
        if not n:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No paired trials yet')
            return
        x0, y0, w, h = self._geom()
        span = self._span()

        def X(i):
            return x0 + w * i / max(n - 1, 1)

        def Y(v):
            return y0 + h * (0.5 - 0.5 * min(max(v / span, -1.0), 1.0))
        p.setFont(mono(10))
        for t in (-span, 0.0, span):                            # recessive grid, labelled in mono
            p.setPen(QPen(QColor(THEME.tok['soft']), 1))
            p.drawLine(QPointF(x0, Y(t)), QPointF(x0 + w, Y(t)))
            p.setPen(QColor(THEME.tok['off']))
            p.drawText(QRectF(0, Y(t) - 8, x0 - 8, 16),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f'{t:+.2f}')
        tone = QColor(THEME.status(G.TONES.get(self.verdict, 'warning')))
        upto = n if self.draw.value >= 1.0 else max(2, int(n * self.draw.value))
        upto = min(upto, n)
        fill = QColor(THEME.tok['ink2'])
        fill.setAlphaF(0.14)
        poly = QPolygonF([QPointF(X(i), Y(self.hi[i])) for i in range(upto)] +
                         [QPointF(X(i), Y(self.lo[i])) for i in reversed(range(upto))])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawPolygon(poly)
        p.setPen(QPen(QColor(THEME.tok['muted']), 1, Qt.PenStyle.DashLine))    # the margin
        for m in (self.margin, -self.margin):
            if abs(m) <= span:
                p.drawLine(QPointF(x0, Y(m)), QPointF(x0 + w, Y(m)))
        p.setPen(QColor(THEME.tok['muted']))
        if self.margin <= span:
            p.drawText(QRectF(x0 + w - 170, Y(self.margin) - 16, 168, 14),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f'margin ±{self.margin:g}')
        p.setPen(QPen(QColor(THEME.tok['ink2']), 1))                           # zero: no difference
        p.drawLine(QPointF(x0, Y(0.0)), QPointF(x0 + w, Y(0.0)))
        p.setPen(QPen(QColor(THEME.tok['ink']), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        for series in (self.lo, self.hi):
            for i in range(1, upto):
                p.drawLine(QPointF(X(i - 1), Y(series[i - 1])), QPointF(X(i), Y(series[i])))
        if self.decided and self.decided <= upto:                              # where the verdict held
            k = self.decided - 1
            p.setPen(QPen(tone, 1.5))
            p.drawLine(QPointF(X(k), y0), QPointF(X(k), y0 + h))
            p.setFont(mono(11))
            right = X(k) < x0 + w - 190
            p.drawText(QRectF(X(k) + 6 if right else X(k) - 186, y0 + 2, 180, 14),
                       int((Qt.AlignmentFlag.AlignLeft if right else Qt.AlignmentFlag.AlignRight)
                           | Qt.AlignmentFlag.AlignVCenter), f'{G.WORDS[self.verdict]} at {self.decided}')
            if self.by == 'discordant':                                         # the band itself has not cleared zero
                right2 = X(k) < x0 + w - 290                                    # the longer line needs its own room
                p.drawText(QRectF(X(k) + 6 if right2 else X(k) - 286, y0 + 18, 280, 14),
                           int((Qt.AlignmentFlag.AlignLeft if right2 else Qt.AlignmentFlag.AlignRight)
                               | Qt.AlignmentFlag.AlignVCenter), 'shown by the discordant pairs (θ ≠ ½)')
        if 0 <= self.hover < upto:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(THEME.tok['ink']))
            for series in (self.lo, self.hi):
                p.drawEllipse(QPointF(X(self.hover), Y(series[self.hover])), 3.5, 3.5)
        p.setFont(mono(11))
        p.setPen(QColor(THEME.tok['muted']))
        what = 'P(B fails) − P(A fails)' if self.what == 'failure' else 'value(B) − value(A)'
        p.drawText(QPointF(x0, self.height() - 8), f'paired trials 1–{n} →   band: {what}, valid at every look · '
                                                   'above zero: B is worse')


class GateTab(QWidget):
    """The Release gate section (module doc). owner: the Reports page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.job = None
        self.doc = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.intro = Badges()
        self.intro.set([('Design', 'paired trials, same seeds', 'the old version A and the new version B drive the '
                         'same scenario instances with the same seeds'),
                        ('Looks', 'after every batch', 'the gate stops when the answer is clear'),
                        ('Confidence', 'valid at every look', 'it holds however often the gate looks')])
        v.addWidget(self.intro)
        v.addWidget(Details('The old version A and the new version B drive the same scenario instances with the same '
                            'seeds; the gate looks after every batch and stops when the answer is clear. Its '
                            'confidence holds however often it looks. Two versions that ran at different fidelity '
                            '(core.fidelity: a fake or kinematic side against CARLA) get no verdict. '
                            'docs/release-gate.md.'))
        f = Form()
        self.name = QLineEdit('release')
        self.name.setToolTip('The gate\'s folder under the workspace\'s gates/; running it again resumes it')
        self.name.editingFinished.connect(self.load)
        self.a = QComboBox()
        self.b = QComboBox()
        for c in (self.a, self.b):
            c.setEditable(True)
            c.setToolTip('follower, autopilot, stack:SUITE.json (a ROS 2 stack\'s suite file) or model:ID — both '
                         'versions of the same kind')
            c.editTextChanged.connect(lambda _: self.check())
        self.scenarios = QListWidget()
        self.scenarios.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.scenarios.setMaximumHeight(130)
        self.scenarios.itemChanged.connect(lambda _: self.check())
        self.budget = QSpinBox()
        self.budget.setRange(2, 5000)
        self.budget.setValue(200)
        self.budget.setToolTip('Paired trials at most: each is two runs (one run scored twice for models)')
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(8)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.001, 0.5)
        self.alpha.setDecimals(3)
        self.alpha.setSingleStep(0.01)
        self.alpha.setValue(0.05)
        self.alpha.setToolTip('The total probability of a wrong verdict, however often the gate looks')
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.005, 1.0)
        self.margin.setDecimals(3)
        self.margin.setSingleStep(0.01)
        self.margin.setValue(0.05)
        self.margin.setToolTip('A difference inside ±margin counts as no difference. Showing that takes about '
                               '17.5 / margin identical trials at 5 %')
        self.primary = QComboBox()
        self.primary.addItem('Failures (pass criteria)', 'failure')
        self.primary.addItem('Criticality (the value)', 'value')
        self.world = QComboBox()
        self.rig = QComboBox()
        self.placement = PlacementCombo(getattr(self.win, 'settings', None))
        self.placement.currentIndexChanged.connect(lambda _: self.check())
        for text, w in (('Name', self.name), ('Old version (A)', self.a), ('New version (B)', self.b),
                        ('Scenarios', self.scenarios), ('Trials at most', self.budget), ('Per look', self.batch),
                        ('Error probability α', self.alpha), ('Margin ±', self.margin), ('Verdict from', self.primary),
                        ('World', self.world), ('Rig', self.rig), ('Runs on', self.placement)):
            f.row(text, w)
        v.addLayout(f)
        self.why = label('', small=True, wrap=True, sel=True)
        self.why.setStyleSheet(f'color: {THEME.status("critical")}')
        self.why.hide()
        v.addWidget(self.why)
        b = QHBoxLayout()
        self.start_btn = button('Run the gate', self.start, role='primary', tip='falcon gate run (a job; resumable)')
        b.addWidget(self.start_btn)
        b.addWidget(button('Refresh', self.load))
        b.addStretch(1)
        v.addLayout(b)
        self.sweep = motion.Sweep()
        v.addWidget(self.sweep)
        self.state = label('', small=True, wrap=True, sel=True)
        v.addWidget(self.state)
        stats = QHBoxLayout()
        self.s_verdict = Stat('Verdict')
        self.s_trials = Stat('Paired trials')
        self.s_diff = Stat('Difference (B − A)')
        for s in (self.s_verdict, self.s_trials, self.s_diff):
            stats.addWidget(s, 1)
        v.addLayout(stats)
        fr = QHBoxLayout()
        self.fid_a = FidelitySlot('A')
        self.fid_b = FidelitySlot('B')
        fr.addWidget(self.fid_a)
        fr.addWidget(self.fid_b)
        fr.addStretch(1)
        v.addLayout(fr)
        self.fid_note = callout('', 'critical')                  # refused, or mixed within a version
        self.fid_note.hide()
        v.addWidget(self.fid_note)
        v.addWidget(label('Confidence band over the trials', h=True))
        self.band = Band()
        v.addWidget(self.band)
        self.scope = label('', small=True, muted=True, wrap=True)
        v.addWidget(self.scope)
        self.scope_details = Details('')
        v.addWidget(self.scope_details)
        v.addWidget(label('Newly failing under the new version', h=True))
        self.changes = QTreeWidget()
        self.changes.setHeaderLabels(['Change', 'Trial', 'Scenario', 'Seed', 'Value A → B', 'Fidelity', 'Run A',
                                      'Run B', ''])
        self.changes.setRootIsDecorated(False)
        self.changes.setMinimumHeight(130)
        copyable_ids(self.changes, 6, 7)
        link_runs(self.changes, self.win, lambda it: ((it.data(0, 256) or {}).get('b') or '') if it else '',
                  exclude='reports',
                  actions=lambda it: [('Compare in Replay', lambda: self.compare_in_replay(it.data(0, 256))),
                                      ('Browse run A', lambda: self.browse('a')),
                                      ('Browse run B', lambda: self.browse('b'))])
        v.addWidget(self.changes)
        cb = QHBoxLayout()
        cb.addWidget(button('Browse run A', lambda: self.browse('a')))
        cb.addWidget(button('Browse run B', lambda: self.browse('b')))
        cb.addWidget(button('Compare the pair', self.compare, tip='Both runs side by side on the Data page'))
        cb.addStretch(1)
        v.addLayout(cb)
        self.note = label('', small=True, muted=True, wrap=True)
        v.addWidget(self.note)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.refresh()

    def showEvent(self, ev):                                      # noqa: N802
        super().showEvent(ev)
        self.placement.reload()
        self.refresh()

    # ------------------------------------------------------------------ what can be chosen
    def _doc(self) -> dict:
        try:
            return self.win.project.doc
        except AttributeError:
            return {}

    def _ws(self) -> str:
        try:
            return self.win.workspace() if self.win.project.path else ''
        except AttributeError:
            return ''

    def refresh(self):
        from ..core import scenarios as SC
        doc = self._doc()
        versions = ['follower', 'autopilot'] + [f'model:{m["id"]}' for m in doc.get('models') or ()
                                                if m.get('kind') == 'detector']
        for c, default in ((self.a, 'follower'), (self.b, 'autopilot')):
            cur = c.currentText() or default
            c.blockSignals(True)
            c.clear()
            c.addItems(versions)
            c.setEditText(cur)
            c.blockSignals(False)
        checked = set(self.selected()) or {'lead_brake', 'cut_in'}
        self.scenarios.blockSignals(True)
        self.scenarios.clear()
        items = [(k, SC.LIBRARY[k]['name']) for k in SC.LIBRARY] + \
                [(s['id'], s.get('name', s['id'])) for s in doc.get('scenarios') or ()]
        for ident, name in items:
            it = QListWidgetItem(name)
            it.setData(256, ident)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if ident in checked else Qt.CheckState.Unchecked)
            self.scenarios.addItem(it)
        self.scenarios.blockSignals(False)
        for combo, items in ((self.world, doc.get('worlds') or ()), (self.rig, doc.get('rigs') or ())):
            cur = combo.currentData()
            combo.clear()
            for it in items:
                combo.addItem(it.get('name', it['id']), it['id'])
            k = combo.findData(cur)
            combo.setCurrentIndex(k if k >= 0 else 0)
        self.check()
        self.load()

    def selected(self) -> list:
        return [self.scenarios.item(i).data(256) for i in range(self.scenarios.count())
                if self.scenarios.item(i).checkState() == Qt.CheckState.Checked]

    def refusal(self) -> str:
        """Why the gate cannot be queued as it is set ('' when it can)."""
        if not re.fullmatch(r'[A-Za-z0-9][\w.-]{0,63}', self.name.text().strip()):
            return 'The name is letters, digits, _ . - (it names a folder)'
        if not self.selected():
            return 'Choose at least one scenario'
        base = ''
        try:
            base = os.path.dirname(os.path.abspath(self.win.project.path)) if self.win.project.path else ''
        except AttributeError:
            pass
        try:
            va, vb = G.parse_version(self.a.currentText(), base), G.parse_version(self.b.currentText(), base)
        except G.GateError as e:
            return str(e)
        if va['kind'] != vb['kind']:
            return 'Both versions must be stacks (follower, autopilot, a suite file) or both models'
        ids = {m.get('id') for m in self._doc().get('models') or ()}
        for v in (va, vb):
            if v['kind'] == 'model' and v['model'] not in ids:
                return f'No model {v["model"]!r} in the project'
        from ..core import scenarios as SC
        world = next((w for w in self._doc().get('worlds') or () if w['id'] == self.world.currentData()), None)
        by_id = {s['id']: s for s in self._doc().get('scenarios') or ()}
        for sid in self.selected():
            scn = by_id.get(sid) or (SC.builtin(sid) if sid in SC.LIBRARY else None)
            why = self.placement.refusal(scn, world)
            if why:
                return why
        return ''

    def check(self):
        why = self.refusal()
        self.why.setText(why)
        self.why.setVisible(bool(why))
        self.start_btn.setEnabled(not why)

    # ------------------------------------------------------------------ the gate
    def gate_dir(self) -> str:
        nm = self.name.text().strip()
        return G.gate_dir(self._ws(), nm) if self._ws() and nm else ''

    def argv(self) -> list:
        w = self.win
        return G.gate_argv(w.project.path, self.name.text().strip(), self.a.currentText().strip(),
                           self.b.currentText().strip(), self.selected(), self.budget.value(), self.batch.value(),
                           round(self.alpha.value(), 3), round(self.margin.value(), 3), None,
                           self.primary.currentData(), 0,
                           self.world.currentData() or '', self.rig.currentData() or '',
                           getattr(w, 'fake_carla', '') or '', self.placement.placement())

    def start(self):
        w = self.win
        if self.job is not None and self.job.state not in FINAL:
            return
        why = self.refusal()
        if why:
            self.check()
            self.state.setText(f'Not queued: {why}')
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        on = self.placement.placement()
        self.job = w.queue.submit(JobSpec('study', f'Release gate {self.name.text().strip()}: '
                                                   f'{self.b.currentText().strip()} against '
                                                   f'{self.a.currentText().strip()} (≤ {self.budget.value()} pairs)'
                                                   + ('' if on == 'local' else f' on {on}'),
                                          [sys.executable, '-m', 'falcon_autonomy'] + self.argv(), memory_gb=0.5,
                                          resumable=True))
        if hasattr(w, 'jobs_dock'):
            w.jobs_dock.raise_()
        self.state.setText(self.job.summary())
        self.sweep.set(0.0)
        self.timer.start(2000)

    def poll(self):
        if self.job is None:
            self.timer.stop()
            return
        self.load()
        if self.job.state in FINAL:
            self.timer.stop()
            self.sweep.set(None)

    def load(self):
        d = G.read_gate(self.gate_dir()) if self.gate_dir() else None
        self.doc = d
        busy = self.job is not None and self.job.state not in FINAL
        if busy:
            self.sweep.set((d['trials_run'] / max(d['budget'], 1)) if d else 0.0)
        elif self.sweep.isVisible():
            self.sweep.set(None)
        self.band.set_gate(d)
        self.changes.clear()
        if d is None:
            self.state.setText(self.job.summary() if busy else 'No gate of this name yet')
            for s in (self.s_verdict, self.s_trials, self.s_diff):
                s.set('—')
            self.scope.setText('')
            self.scope_details.set_text('')
            self.fid_a.set('unknown')
            self.fid_b.set('unknown')
            self.fid_note.hide()
            return
        fv = d.get('fidelity_by_version') or {}
        mixed = {s: fv.get(s) == 'mixed' for s in ('a', 'b')}
        self.fid_a.set(fv.get('a') or 'unknown')
        self.fid_b.set(fv.get('b') or 'unknown')
        why = d.get('refused') or d.get('fidelity_note') or ''
        self.fid_note.label.setText(('No verdict. ' + d['refused']) if d.get('refused') else why)
        tone = 'critical' if d.get('refused') or any(mixed.values()) else 'warning'
        self.fid_note.label.setStyleSheet(f'color: {THEME.status(tone)}')
        self.fid_note.setProperty('tone', tone)
        self.fid_note.setVisible(bool(why))
        vs = d.get('versions') or {}
        tone = G.TONES.get(d['verdict'], 'warning')
        running = d.get('state') != 'finished'
        self.s_verdict.set(d['word'], sub=('so far — the gate is still running' if running else
                                           f'B {(vs.get("b") or {}).get("label", "")} against A '
                                           f'{(vs.get("a") or {}).get("label", "")}'), tone=tone)
        self.s_verdict.value.setStyleSheet(f'color: {THEME.status(tone)}')
        self.s_trials.set(str(d['trials_used']), f'of {d["budget"]}',
                          f'failed: A {d["failures"]["a"]}, B {d["failures"]["b"]} · {d["discordant"]["n"]} discordant')
        r = d['risk'] if d['primary'] == 'failure' else d['value']
        dc = d['discordant']
        self.s_diff.set(f'{r["lower"]:+.3f} … {r["upper"]:+.3f}',
                        sub=(f'θ in [{dc["interval"][0]:.2f}, {dc["interval"][1]:.2f}] decided · '
                             if d.get('decided_by') == 'discordant' else '')
                        + f'margin ±{d["margin"]:g}' + (' · inside the margin' if d['within_margin'] else ''))
        if d['verdict'] == 'refused':
            self.s_verdict.set('no verdict', sub='the versions ran at different fidelity', tone='critical')
            self.s_verdict.value.setStyleSheet(f'color: {THEME.status("critical")}')
        self.state.setText((self.job.summary() + ' · ' if busy else '') + G.headline(d).splitlines()[0])
        self.scope.setText('The verdict holds for the tested scenario distribution only.')
        self.scope_details.set_text('The verdict ' + d['scope'] + '. Trials are taken as independent draws of that '
                                    'distribution; in physics motion or with traffic a pair also holds run-to-run '
                                    'noise (it costs trials, not validity).')
        levels = self._trial_levels()
        can = self._replay_can_compare()
        for kind, rows in (('newly fails', d['newly_failing']), ('newly passes', d['newly_passing'])):
            for x in rows:
                la, lb = levels.get(x['i'], ('unknown', 'unknown'))
                fl = FD.label(la) if la == lb else f'{FD.label(la)} / {FD.label(lb)}'
                it = QTreeWidgetItem([kind, str(x['i']), str(x['scenario']), str(x['seed']),
                                      f'{x["value_a"]:.3f} → {x["value_b"]:.3f}', fl,
                                      short_id(os.path.basename(x['a'] or '')),
                                      short_id(os.path.basename(x['b'] or '')), ''])
                it.setForeground(0, QColor(THEME.status('critical' if kind == 'newly fails' else 'good')))
                it.setToolTip(4, ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                           for k, v in sorted((x.get('values') or {}).items())))
                it.setToolTip(5, f'A: {FD.sentence(la)}\nB: {FD.sentence(lb)}')
                if FD.tone(la if la != 'unknown' else lb):
                    it.setForeground(5, QColor(THEME.status(FD.tone(la if la != 'unknown' else lb))))
                for col, side in ((6, 'a'), (7, 'b')):
                    it.setData(col, Qt.ItemDataRole.UserRole + 1, x[side] or '')
                    it.setToolTip(col, x[side] or '')
                it.setData(0, 256, x)
                self.changes.addTopLevelItem(it)
                if kind == 'newly fails':
                    b = button('Compare in Replay', lambda _=False, x=x: self.compare_in_replay(x),
                               tip='Both runs of the pair on the Replay page' if can else
                               'This build\'s Replay page cannot compare two runs yet: use Compare the pair (the Data '
                               'page)')
                    b.setEnabled(can)
                    b.setProperty('role', 'link')
                    self.changes.setItemWidget(it, 8, b)
        for k in (0, 1, 2, 3, 4, 5):
            self.changes.resizeColumnToContents(k)

    def _row(self):
        it = self.changes.currentItem()
        return it.data(0, 256) if it is not None else None

    def _trial_levels(self) -> dict:
        """{trial: (A's level, B's level)} from trials.jsonl (core.releasegate.trial_fidelity)."""
        d = self.gate_dir()
        if not d:
            return {}
        return {t['i']: (G.trial_fidelity(t['a']), G.trial_fidelity(t['b'])) for t in G.read_trials(d)}

    def _replay_can_compare(self) -> bool:
        if not hasattr(self.win, 'page'):
            return hasattr(self.win, 'go')                     # a window without page lookup: try on click
        page = self.win.page('replay')
        return page is not None and hasattr(page, 'compare_runs')

    def compare_in_replay(self, x):
        """A newly failing pair on the Replay page (its compare_runs(run_a, run_b), workstream R)."""
        if not x or not hasattr(self.win, 'go'):
            return False
        page = self.win.go('replay')
        if page is not None and hasattr(page, 'compare_runs'):
            page.compare_runs(x['a'], x['b'])
            return True
        self.note.setText('This build\'s Replay page cannot compare two runs: Compare the pair shows them on the '
                          'Data page.')
        return False

    def browse(self, side: str):
        x = self._row()
        if x and x.get(side) and hasattr(self.win, 'go'):
            self.win.go('data').show_run(x[side])

    def compare(self):
        x = self._row()
        if not x or not hasattr(self.win, 'go'):
            return
        page = self.win.go('data')
        if hasattr(page, 'compare_runs'):
            page.compare_runs([x['a'], x['b']])
        else:
            page.show_run(x['b'])

    def shutdown(self):
        self.timer.stop()
