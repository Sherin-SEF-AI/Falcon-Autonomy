"""Rare cases ▸ Two-speed search: a fast kinematic simulator screens a scenario's parameters and CARLA confirms the
most promising (core.fastsim; docs/two-speed-search.md).

The section takes the scenario, the budget in CARLA runs and how it is split — runs per round, the share of each
round spent on random controls — the number of cheap candidates screened and the rank correlation below which the
screen is not trusted; the world, rig, seed and failure threshold are the page's. *Search at two speeds* queues
`falcon prescreen run` as a job (resumable; its study is the page's criticality search of that scenario, so the page's
curve, failing cases and failure map show its runs too). A scenario the fast simulator cannot run says why and is not
queued. While it runs a sweep shows the CARLA runs spent; the result: the scatter of the cheap against the CARLA
criticality of every confirmed run with the rank correlation and the trust word, the false-negative risk the random
controls give, and the confirmed failures (each a CARLA run, the cheap value beside it)."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QSpinBox, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ..core import fastsim as F
from ..core.jobs import FINAL, JobSpec
from . import motion
from .placement import PlacementCombo
from .theme import THEME, mono
from .widgets import Form, Stat, button, label

TRUST_TONE = {'trusted': 'good', 'not trusted': 'critical', 'not judged yet': 'warning'}
ROLE_WORDS = {'top': 'selected by the screen', 'control': 'random control', 'fallback': 'plain search (fallback)',
              'prior': 'an earlier search'}


class Scatter(QWidget):
    """Every confirmed run: its cheap criticality (x) against its CARLA criticality (y, the one axis that counts),
    the failure threshold dashed on both, the isotonic fit as a line. Selected points are filled, random controls are
    rings, a fallback's and earlier searches' are small and muted. The rank correlation and the trust word are
    written on the plot; hover reads a point. New points draw in."""

    def __init__(self):
        super().__init__()
        self.pairs, self.threshold, self.knots = [], 0.5, None
        self.rho, self.trust = None, ''
        self.hover = -1
        self.setMinimumHeight(240)
        self.setMouseTracking(True)
        self.draw = motion.Driver(self)

    def set_data(self, pairs, threshold, knots=None, rho=None, trust=''):
        pairs = list(pairs or [])
        if len(pairs) != len(self.pairs) and pairs:
            self.draw.start()
        self.pairs, self.threshold, self.knots, self.rho, self.trust = pairs, float(threshold), knots, rho, trust
        self.hover = -1
        self.update()

    def _geom(self):
        return 44.0, 14.0, self.width() - 60.0, self.height() - 52.0

    def _pos(self, lo, hi):
        x0, y0, w, h = self._geom()
        return QPointF(x0 + w * min(max(lo, 0.0), 1.0), y0 + h * (1.0 - min(max(hi, 0.0), 1.0)))

    def mouseMoveEvent(self, ev):                               # noqa: N802
        best, at = 14.0, -1
        for i, p in enumerate(self.pairs):
            q = self._pos(p['low'], p['high'])
            d = abs(q.x() - ev.position().x()) + abs(q.y() - ev.position().y())
            if d < best:
                best, at = d, i
        if at != self.hover:
            self.hover = at
            if at >= 0:
                p = self.pairs[at]
                self.setToolTip(f'cheap {p["low"]:.3f} · CARLA {p["high"]:.3f}'
                                + (' (a failure)' if p['high'] >= self.threshold else '')
                                + f'\n{ROLE_WORDS.get(p.get("role"), p.get("role"))}'
                                + (f', round {p["round"] + 1}' if (p.get('round') or 0) >= 0 else '')
                                + (f'\n{os.path.basename(p["run"])}' if p.get('run') else ''))
            else:
                self.setToolTip('')
            self.update()

    def leaveEvent(self, _ev):                                  # noqa: N802
        self.hover = -1
        self.update()

    def paintEvent(self, _ev):                                  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.pairs:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No confirmed runs yet')
            return
        x0, y0, w, h = self._geom()
        p.setFont(mono(10))
        for t in (0.0, 0.5, 1.0):                               # recessive grid, labelled in mono
            p.setPen(QPen(QColor(THEME.tok['soft']), 1))
            p.drawLine(QPointF(x0, y0 + h * (1 - t)), QPointF(x0 + w, y0 + h * (1 - t)))
            p.drawLine(QPointF(x0 + w * t, y0), QPointF(x0 + w * t, y0 + h))
            p.setPen(QColor(THEME.tok['off']))
            p.drawText(QRectF(0, y0 + h * (1 - t) - 8, x0 - 8, 16),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f'{t:.1f}')
            p.drawText(QRectF(x0 + w * t - 20, y0 + h + 4, 40, 14), int(Qt.AlignmentFlag.AlignCenter), f'{t:.1f}')
        fail = QColor(THEME.status('critical'))
        p.setPen(QPen(fail, 1, Qt.PenStyle.DashLine))
        ty = y0 + h * (1 - self.threshold)
        p.drawLine(QPointF(x0, ty), QPointF(x0 + w, ty))
        p.setPen(fail)
        p.drawText(QRectF(x0 + w - 190, ty - 16, 188, 14),
                   int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                   f'fails in CARLA ≥ {self.threshold:.2f}')
        if self.knots:                                          # the isotonic fit of CARLA on cheap
            p.setPen(QPen(QColor(THEME.tok['ink2']), 1.5))
            pts = [self._pos(0.0, self.knots[0][1])] + [self._pos(a, b) for a, b in self.knots] + \
                [self._pos(1.0, self.knots[-1][1])]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        n = len(self.pairs)
        shown = n if self.draw.value >= 1.0 else int(n * self.draw.value)
        ink, muted, base = QColor(THEME.tok['ink']), QColor(THEME.tok['muted']), QColor(THEME.tok['base'])
        for i, q in enumerate(self.pairs[:max(shown, 0)]):
            c = self._pos(q['low'], q['high'])
            col = fail if q['high'] >= self.threshold else ink
            r = 4.0 if i != self.hover else 5.5
            role = q.get('role')
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(base)
            p.drawEllipse(c, r + 1.5, r + 1.5)                  # a surface ring keeps dots apart
            if role == 'control':                               # a ring: drawn at random, not chosen
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(col, 1.6))
                p.drawEllipse(c, r, r)
            elif role == 'top':
                p.setBrush(col)
                p.drawEllipse(c, r, r)
            else:
                p.setBrush(fail if q['high'] >= self.threshold else muted)
                p.drawEllipse(c, r - 1.5, r - 1.5)
        p.setFont(mono(11))
        if self.trust:
            p.setPen(QColor(THEME.status(TRUST_TONE.get(self.trust, 'warning'))))
            rho = 'ρ —' if self.rho is None else f'ρ {self.rho:+.2f}'
            p.drawText(QRectF(x0 + 8, y0 + 4, w - 16, 16), int(Qt.AlignmentFlag.AlignLeft), f'{rho} · {self.trust}')
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QPointF(x0, self.height() - 8), 'cheap →  up: CARLA · filled: selected · ring: control · line: fit')


class PrescreenTab(QWidget):
    """The Two-speed search section (module doc). owner: the Rare cases page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.job = None
        self.doc = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(label('A fast kinematic simulator (the director\'s lane follower on a straight road, the same '
                          'criticality formula) rates thousands of parameter sets in seconds; CARLA runs only the '
                          'most promising, and a share of random controls that measure how far the two agree. The '
                          'cheap value is a screen, never a result.', muted=True, wrap=True))
        f = Form()
        self.scenario = QComboBox()
        self.scenario.currentIndexChanged.connect(lambda _: self.scenario_changed())
        self.budget = QSpinBox()
        self.budget.setRange(4, 5000)
        self.budget.setValue(40)
        self.budget.setToolTip('CARLA runs in all; the study\'s finished runs count towards it')
        self.batch = QSpinBox()
        self.batch.setRange(2, 64)
        self.batch.setValue(8)
        self.controls = QSpinBox()
        self.controls.setRange(0, 90)
        self.controls.setValue(20)
        self.controls.setSuffix(' %')
        self.controls.setToolTip('The share of every round drawn uniformly at random, whatever the screen says: the '
                                 'unbiased sample the false-negative risk is estimated from')
        self.screen = QSpinBox()
        self.screen.setRange(100, 200000)
        self.screen.setSingleStep(1000)
        self.screen.setValue(10000)
        self.screen.setToolTip('Parameter sets rated by the fast simulator before the first CARLA run')
        self.min_rho = QDoubleSpinBox()
        self.min_rho.setRange(-1.0, 1.0)
        self.min_rho.setSingleStep(0.05)
        self.min_rho.setValue(0.5)
        self.min_rho.setToolTip('Below this Spearman rank correlation between the cheap and the CARLA criticality the '
                                'screen is not trusted: a plain search (Bayesian optimisation by the probability of '
                                'failure) spends the rest of the budget')
        self.placement = PlacementCombo(getattr(self.win, 'settings', None))
        self.placement.currentIndexChanged.connect(lambda _: self.scenario_changed())
        for text, w in (('Scenario', self.scenario), ('CARLA runs', self.budget), ('Per round', self.batch),
                        ('Random controls', self.controls), ('Cheap candidates', self.screen),
                        ('Trust at ρ ≥', self.min_rho), ('Runs on', self.placement)):
            f.row(text, w)
            if isinstance(w, QSpinBox):
                w.valueChanged.connect(lambda _: self.show_split())
        v.addLayout(f)
        self.split = label('', small=True, mono=True, wrap=True)
        v.addWidget(self.split)
        self.why = label('', small=True, wrap=True, sel=True)
        self.why.setStyleSheet(f'color: {THEME.status("critical")}')
        self.why.hide()
        v.addWidget(self.why)
        b = QHBoxLayout()
        self.start_btn = button('Search at two speeds', self.start, role='primary',
                                tip='falcon prescreen run (a job; resumable)')
        b.addWidget(self.start_btn)
        b.addWidget(button('Refresh', self.load))
        b.addStretch(1)
        v.addLayout(b)
        self.sweep = motion.Sweep()
        v.addWidget(self.sweep)
        self.state = label('', small=True, wrap=True, sel=True)
        v.addWidget(self.state)
        stats = QHBoxLayout()
        self.s_rho = Stat('Rank correlation')
        self.s_fn = Stat('False-negative risk')
        self.s_runs = Stat('CARLA runs')
        for s in (self.s_rho, self.s_fn, self.s_runs):
            stats.addWidget(s, 1)
        v.addLayout(stats)
        v.addWidget(label('Cheap against CARLA criticality', h=True))
        self.scatter = Scatter()
        v.addWidget(self.scatter)
        v.addWidget(label('Confirmed failures (CARLA runs)', h=True))
        self.failures = QTreeWidget()
        self.failures.setHeaderLabels(['CARLA', 'Cheap', 'Found by', 'Values', 'Run'])
        self.failures.setRootIsDecorated(False)
        self.failures.setMinimumHeight(120)
        self.failures.itemDoubleClicked.connect(lambda it, _c: self.browse(it))
        v.addWidget(self.failures)
        fb = QHBoxLayout()
        fb.addWidget(button('Browse its run', lambda: self.browse(self.failures.currentItem())))
        fb.addStretch(1)
        v.addLayout(fb)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.refresh()

    def showEvent(self, ev):                                      # noqa: N802
        super().showEvent(ev)
        self.placement.reload()
        self.refresh()

    # ------------------------------------------------------------------ the scenario
    def _scenarios(self):
        if hasattr(self.page, 'scenarios'):
            return self.page.scenarios()
        from ..core import scenarios as SC
        return [(k, SC.builtin(k)) for k in SC.LIBRARY]

    def _scenario_doc(self):
        ident = self.scenario.currentData()
        return next((s for k, s in self._scenarios() if k == ident), None)

    def _ws(self) -> str:
        try:
            return self.win.workspace() if self.win.project.path else ''
        except AttributeError:
            return ''

    def _page_value(self, name, default):
        w = getattr(self.page, name, None)
        if w is None:
            return default
        return w.currentData() if isinstance(w, QComboBox) else w.value()

    def refresh(self):
        cur = self.scenario.currentData()
        self.scenario.blockSignals(True)
        self.scenario.clear()
        for ident, s in self._scenarios():
            self.scenario.addItem(s['name'], ident)
        k = self.scenario.findData(cur)
        self.scenario.setCurrentIndex(k if k >= 0 else 0)
        self.scenario.blockSignals(False)
        self.scenario_changed()

    def refusal(self) -> str:
        """Why the chosen scenario cannot be searched at two speeds where chosen ('' when it can)."""
        scn = self._scenario_doc()
        if scn is None:
            return 'Choose a scenario'
        why = F.unsupported(scn)
        if why:
            return f'The fast simulator cannot run this scenario: {why}. Use Search above (every run in CARLA).'
        wid = self._page_value('world', '')
        try:
            world = self.win.project.item('worlds', wid) if wid else None
        except AttributeError:
            world = None
        return self.placement.refusal(scn, world)

    def scenario_changed(self):
        why = self.refusal()
        self.why.setText(why)
        self.why.setVisible(bool(why))
        self.start_btn.setEnabled(not why)
        self.show_split()
        self.load()

    def split_counts(self) -> tuple:
        """(controls, selected) CARLA runs of the budget as the rounds will draw them."""
        budget, batch, share = self.budget.value(), self.batch.value(), self.controls.value() / 100.0
        ctrl, left = 0, budget
        while left > 0:
            n = min(batch, left)
            ctrl += 0 if n < 2 or share <= 0 else min(max(1, int(round(share * n))), n - 1)
            left -= n
        return ctrl, budget - ctrl

    def show_split(self):
        ctrl, top = self.split_counts()
        self.split.setText(f'{self.budget.value()} CARLA runs: {top} selected by the screen, {ctrl} random controls · '
                           f'{self.screen.value():,} cheap candidates screened first')

    # ------------------------------------------------------------------ the study
    def study(self) -> str:
        return F.default_name(self.scenario.currentData()) if self.scenario.currentData() else ''

    def study_dir(self) -> str:
        return os.path.join(self._ws(), 'studies', self.study()) if self._ws() and self.study() else ''

    def argv(self) -> list:
        w = self.win
        argv = F.prescreen_argv(w.project.path, self.scenario.currentData(), self.study(), self.budget.value(),
                                self.batch.value(), float(self._page_value('threshold', 0.5)),
                                int(self._page_value('seed', 0)), self._page_value('world', '') or '',
                                self._page_value('rig', '') or '', self.screen.value(),
                                self.controls.value() / 100.0, self.min_rho.value(), 'bayes-pof',
                                self._page_value('count', 'avoidable') or 'avoidable',
                                getattr(w, 'fake_carla', '') or '', self.placement.placement())
        out, i = [], 0
        while i < len(argv):                                     # a page without that choice: the project's first
            if argv[i] in ('--world', '--rig') and argv[i + 1] == '':
                i += 2
                continue
            out.append(argv[i])
            i += 1
        return out

    def start(self):
        w = self.win
        if self.job is not None and self.job.state not in FINAL:
            return
        why = self.refusal()
        if why:
            self.scenario_changed()
            self.state.setText(f'Not queued: {why}')
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        argv = self.argv()
        on = self.placement.placement()
        self.job = w.queue.submit(JobSpec('study', f'Two-speed search {self.scenario.currentText()} '
                                                   f'({self.budget.value()} CARLA runs)'
                                                   + ('' if on == 'local' else f' on {on}'),
                                          [sys.executable, '-m', 'falcon_autonomy'] + argv, memory_gb=1.5,
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
            if hasattr(self.page, 'load'):
                self.page.load()                                 # the page's curve and cases read the same study

    def load(self):
        d = F.read_prescreen(self.study_dir()) if self.study_dir() else None
        self.doc = d
        busy = self.job is not None and self.job.state not in FINAL
        if busy:
            self.sweep.set((d['spent'] / max(d['budget'], 1)) if d else 0.0)
        elif self.sweep.isVisible():
            self.sweep.set(None)
        if d is None:
            self.state.setText(self.job.summary() if busy else 'No two-speed search of this scenario yet')
            self.s_rho.set('—')
            self.s_fn.set('—')
            self.s_runs.set('—')
            self.scatter.set_data([], self._page_value('threshold', 0.5))
            self.failures.clear()
            return
        f = d['fidelity']
        fn = f['false_negative']
        rho = f['spearman']
        tone = TRUST_TONE.get(d['trust'], 'warning')
        self.s_rho.set('—' if rho is None else f'{rho:+.2f}', sub=f'{d["trust"]} · {f["pairs"]} pairs', tone=tone)
        self.s_rho.setToolTip(f'Spearman rank correlation of the cheap and the CARLA criticality over {f["pairs"]} '
                              f'confirmed runs; trusted at ≥ {f["min_spearman"]:g} over ≥ {f["min_pairs"]} pairs. Over '
                              f'the {f["controls"]} random controls alone: '
                              + ('not defined' if f['spearman_controls'] is None else f'{f["spearman_controls"]:+.2f}'))
        if fn['dropped_controls']:
            self.s_fn.set(f'≤ {100.0 * fn["upper95"]:.0f}', '%', f'{fn["failed"]} of {fn["dropped_controls"]} controls '
                          'failed', 'warning' if fn['failed'] else '')
        else:
            self.s_fn.set('—', sub='no dropped control yet')
        self.s_fn.setToolTip('Of the random controls the screen predicted below the threshold when they were drawn, '
                             'the share that failed in CARLA: the one-sided 95 % Clopper–Pearson bound. Few controls '
                             'cannot show a small risk (none failing of n bounds it near 3/n).')
        tp = d.get('throughput') or {}
        self.s_runs.set(f'{d["spent"]}', f'of {d["budget"]}', f'{d["screened"]:,} screened, {tp.get("per_s")}/s')
        fb = d.get('fallback')
        self.state.setText((self.job.summary() + ' · ' if busy else '')
                           + (f'Fell back to a plain search ({fb["method"]}) after {fb["from_run"]} runs: '
                              f'{fb["why"]}' if fb else f'The screen is {d["trust"]}')
                           + f' · {len(d.get("failures") or ())} failures confirmed')
        self.scatter.set_data(d['pairs'], d['threshold'], f.get('isotonic'), rho, d['trust'])
        before = {self.failures.topLevelItem(i).data(0, 256).get('run')
                  for i in range(self.failures.topLevelItemCount())}
        self.failures.clear()
        for x in d.get('failures') or ():
            it = QTreeWidgetItem([f'{x["value"]:.3f}', f'{x["low"]:.3f}', ROLE_WORDS.get(x['role'], x['role']),
                                  ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                            for k, v in sorted(x['values'].items())),
                                  os.path.basename(x.get('run') or '')])
            it.setData(0, 256, x)
            self.failures.addTopLevelItem(it)
            if before and x.get('run') not in before:             # a failure that just arrived eases in
                motion.flash_item(it, self.failures.columnCount())

    def browse(self, it):
        x = it.data(0, 256) if it is not None else None
        if x and x.get('run') and hasattr(self.win, 'go'):
            self.win.go('data').show_run(x['run'])

    def shutdown(self):
        self.timer.stop()
