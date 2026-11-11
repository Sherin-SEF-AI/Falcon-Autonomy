"""Rare cases ▸ Falsify: search a scenario for a run that violates a signal-temporal-logic requirement
(core.stlobjective; docs/falsification.md).

The requirement is a formula over the rule signals (core.rules.SIGNALS — the surrogate-safety ones too: ttc_min_s,
drac_mps2, pet_s, …), checked as it is typed (the parse error names its column), a rule of a rule set, or one saved in
the workspace's safety/formulas.json. The study runs as a job (`falcon falsify run`, resumable) with the chosen
scenario, method (CMA-ES, the cross-entropy method, Bayesian optimisation …) and budget, on the page's world, rig,
seed and rate. The results: the best (lowest) robustness ρ, the falsified runs (ρ < 0), the near misses (0 ≤ ρ <
margin), and the ρ(t) trace of the chosen run with the time that decides it. Every run can be browsed or replayed from
the page above (its folder is in the list)."""
from __future__ import annotations

import json
import math
import os
import sys

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QPlainTextEdit, QSpinBox, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import stlobjective as SO
from ..core.jobs import FINAL, JobSpec
from .theme import STATUS, THEME
from .widgets import Form, KeyValues, button, label

METHODS = (('cmaes', 'CMA-ES'), ('cem', 'Cross-entropy method'), ('bayes', 'Bayesian optimisation'),
           ('adaptive', 'Adaptive sampling'), ('random', 'Random (baseline)'))
EXAMPLE = 'G (ttc_min_s >= 1.5 and pet_s >= 1.0)'


class Trace(QWidget):
    """ρ(t) of one run: the robustness series with the zero line (below it the requirement is violated) and the
    deciding time marked."""

    def __init__(self):
        super().__init__()
        self.t, self.rho, self.worst = [], [], None
        self.title = ''
        self.setMinimumHeight(150)

    def set_series(self, t, rho, worst=None, title=''):
        self.t = list(t or [])
        self.rho = [SO.fnum(x) for x in rho or []]
        self.worst, self.title = worst, title
        self.update()

    def paintEvent(self, _ev):                                    # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        pts = [(t, r) for t, r in zip(self.t, self.rho) if math.isfinite(r)]
        if not pts:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Choose a run to see its ρ(t)' if not self.t
                       else 'ρ is not finite or unknown throughout this run')
            return
        x0, y0, w, h = 44, 10, self.width() - 56, self.height() - 34
        t0, t1 = min(t for t, _ in pts), max(t for t, _ in pts)
        lo, hi = min(0.0, min(r for _, r in pts)), max(0.0, max(r for _, r in pts))
        if hi - lo < 1e-9:
            hi = lo + 1.0

        def X(t):
            return x0 + w * (t - t0) / max(t1 - t0, 1e-9)

        def Y(r):
            return y0 + h * (hi - r) / (hi - lo)
        p.setPen(QColor(THEME.tok['grid2']))
        p.drawRect(QRectF(x0, y0, w, h))
        p.setPen(QPen(QColor(STATUS['critical']), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x0, Y(0.0)), QPointF(x0 + w, Y(0.0)))
        p.setPen(QPen(QColor(THEME.tok.get('accent', '#2a78d6')), 2))
        for (ta, ra), (tb, rb) in zip(pts, pts[1:]):
            p.drawLine(QPointF(X(ta), Y(ra)), QPointF(X(tb), Y(rb)))
        if self.worst is not None and t0 <= self.worst <= t1:
            p.setPen(QPen(QColor(STATUS['serious']), 1.5))
            p.drawLine(QPointF(X(self.worst), y0), QPointF(X(self.worst), y0 + h))
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QPointF(4, Y(hi) + 10), f'{hi:.3g}')
        p.drawText(QPointF(4, Y(lo)), f'{lo:.3g}')
        p.drawText(QPointF(x0, self.height() - 8), f'{self.title}  t {t0:.1f}–{t1:.1f} s · dashed: ρ = 0 (below: '
                                                   'violated) · orange: the deciding time')


class FalsifyTab(QWidget):
    """The Falsify tab (module doc). owner: the Rare cases page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.job = None
        self.spec = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(label('A requirement over the recorded signals (falcon rules signals), e.g. '
                          f'{EXAMPLE} — or a rule of a set, or one saved in the workspace.', muted=True, wrap=True))
        self.source = QComboBox()
        self.source.currentIndexChanged.connect(lambda _: self._source_changed())
        v.addWidget(self.source)
        self.formula = QPlainTextEdit()
        self.formula.setPlaceholderText(EXAMPLE)
        self.formula.setFixedHeight(64)
        self.formula.textChanged.connect(lambda: self._debounce.start(250))
        v.addWidget(self.formula)
        self.parse = label('', small=True, wrap=True, sel=True)
        v.addWidget(self.parse)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.check)
        f = Form()
        self.scenario = QComboBox()
        self.scenario.currentIndexChanged.connect(lambda _: self.load())
        self.method = QComboBox()
        for k, t in METHODS:
            self.method.addItem(t, k)
        self.budget = QSpinBox()
        self.budget.setRange(4, 5000)
        self.budget.setValue(60)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.001, 1000.0)
        self.scale.setDecimals(3)
        self.scale.setValue(SO.DEFAULT_SCALE)
        self.scale.setToolTip('ρ\'s scale in the value σ(−ρ/scale) the searcher maximises (the predicate\'s units)')
        self.scale.valueChanged.connect(lambda _: self.check())
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0.0, 1000.0)
        self.smooth.setDecimals(2)
        self.smooth.setToolTip('β > 0: the searcher sees a soft-min robustness (an approximation); the exact ρ is '
                               'always what is reported')
        self.smooth.valueChanged.connect(lambda _: self.check())
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 1000.0)
        self.margin.setValue(SO.DEFAULT_MARGIN)
        self.margin.setToolTip('a near miss: 0 ≤ ρ < margin (in the predicate\'s units)')
        self.margin.valueChanged.connect(lambda _: self.load())
        for text, w in (('Scenario', self.scenario), ('Method', self.method), ('Evaluations', self.budget),
                        ('Scale', self.scale), ('Smoothing β', self.smooth), ('Near miss below ρ', self.margin)):
            f.row(text, w)
        v.addLayout(f)
        b = QHBoxLayout()
        self.start_btn = button('Falsify', self.start, role='primary', tip='falcon falsify run (a job; resumable)')
        b.addWidget(self.start_btn)
        b.addWidget(button('Save the requirement…', self.save_formula, tip='into the workspace\'s '
                                                                          'safety/formulas.json'))
        b.addWidget(button('Refresh', self.load))
        b.addStretch(1)
        v.addLayout(b)
        self.kv = KeyValues(('State', 'Best ρ', 'Falsified', 'Near misses', 'Unknown'))
        v.addWidget(self.kv)
        self.runs = QTreeWidget()
        self.runs.setHeaderLabels(['ρ', 'Verdict', 'Decided at (s)', 'Values', 'Run'])
        self.runs.setRootIsDecorated(False)
        self.runs.setMinimumHeight(140)
        self.runs.currentItemChanged.connect(lambda it, _p: self.show_trace(it))
        v.addWidget(self.runs)
        self.trace = Trace()
        v.addWidget(self.trace)
        self.refresh()

    def showEvent(self, ev):                                      # noqa: N802 (another project, new studies)
        super().showEvent(ev)
        self.refresh()

    # ------------------------------------------------------------------ the requirement
    def refresh(self):
        cur_s, cur_src = self.scenario.currentData(), self.source.currentData()
        self.scenario.blockSignals(True)
        self.scenario.clear()
        for ident, s in self._scenarios():
            self.scenario.addItem(s['name'], ident)
        k = self.scenario.findData(cur_s)
        self.scenario.setCurrentIndex(k if k >= 0 else 0)
        self.scenario.blockSignals(False)
        self.source.blockSignals(True)
        self.source.clear()
        self.source.addItem('A formula (typed below)', 'stl:')
        from ..core import rules as R
        for set_id in R.BUILTIN:
            for r in R.load_ruleset(set_id)['rules']:
                self.source.addItem(f'Rule {set_id}: {r["title"]}', f'rule:{set_id}:{r["id"]}')
        try:
            for name in SO.load_formulas(self._ws()):
                self.source.addItem(f'Saved: {name}', f'name:{name}')
        except SO.STLObjectiveError:
            pass
        k = self.source.findData(cur_src)
        self.source.setCurrentIndex(k if k >= 0 else 0)
        self.source.blockSignals(False)
        self._source_changed()

    def _ws(self) -> str:
        try:
            return self.win.workspace() if self.win.project.path else ''
        except AttributeError:
            return ''

    def _scenarios(self):
        if hasattr(self.page, 'scenarios'):
            return self.page.scenarios()
        from ..core import scenarios as SC
        return [(k, SC.builtin(k)) for k in SC.LIBRARY]

    def _choice(self) -> tuple:
        """(kind, value) of the requirement's source: ('stl', ''), ('rule', 'SET:ID') or ('name', NAME)."""
        kind, _, val = (self.source.currentData() or 'stl:').partition(':')
        return kind, val

    def _source_changed(self):
        kind, val = self._choice()
        self.formula.setReadOnly(kind != 'stl')
        if kind == 'rule':
            from ..core import rules as R
            set_id, rid = val.split(':', 1)
            rule = next(r for r in R.load_ruleset(set_id)['rules'] if r['id'] == rid)
            self.formula.setPlainText(rule['formula'])
        elif kind == 'name':
            try:
                self.formula.setPlainText(SO.load_formulas(self._ws())[val].get('formula') or '')
            except (SO.STLObjectiveError, KeyError):
                pass
        self.check()

    def requirement(self) -> dict | None:
        kind, val = self._choice()
        if kind == 'rule':
            return {'rule': val, 'scale': self.scale.value(), 'smooth': self.smooth.value()}
        if kind == 'name':
            try:
                return SO.named(self._ws(), val)
            except SO.STLObjectiveError:
                return None
        text = self.formula.toPlainText().strip()
        return {'stl': text, 'scale': self.scale.value(), 'smooth': self.smooth.value()} if text else None

    def check(self):
        """Parse the requirement as it is typed: its normal form and signals, or the error with its column."""
        req = self.requirement()
        self.spec = None
        if req is None:
            self.parse.setText('Type a formula, or choose a rule')
            self.parse.setStyleSheet(f'color: {THEME.tok["muted"]}')
        else:
            try:
                n = SO.normalise(req)
                self.spec = n
                self.parse.setText(f'✓ {n["formula"]}   signals: {", ".join(n["signals"])}')
                self.parse.setStyleSheet(f'color: {STATUS["good"]}')
            except SO.STLObjectiveError as e:
                self.parse.setText(f'✗ {e}')
                self.parse.setStyleSheet(f'color: {STATUS["critical"]}')
        self.start_btn.setEnabled(self.spec is not None)
        self.load()

    def save_formula(self):
        from PyQt6.QtWidgets import QInputDialog
        if self.spec is None or not self._ws():
            return
        name, ok = QInputDialog.getText(self, 'Save the requirement', 'Name (letters, digits, _ . -):')
        if ok and name:
            try:
                SO.save_formula(self._ws(), name.strip(), self.requirement())
            except SO.STLObjectiveError as e:
                self.kv.set('State', str(e))
                return
            self.refresh()
            k = self.source.findData(f'name:{name.strip()}')
            if k >= 0:
                self.source.setCurrentIndex(k)

    # ------------------------------------------------------------------ the study
    def _page_value(self, name, default):
        w = getattr(self.page, name, None)
        if w is None:
            return default
        return w.currentData() if isinstance(w, QComboBox) else w.value()

    def study(self) -> str:
        if self.spec is None or not self.scenario.currentData():
            return ''
        return SO.study_name(self.scenario.currentData(), self.spec, (), self._page_value('fps', 20))

    def study_dir(self) -> str:
        nm = self.study()
        return os.path.join(self._ws(), 'studies', nm) if nm and self._ws() else ''

    def argv(self) -> list:
        w = self.win
        req = self.requirement() or {}
        argv = ['falsify', 'run', w.project.path, '--scenario', self.scenario.currentData(), '--study', self.study(),
                '--method', self.method.currentData(), '--budget', str(self.budget.value()),
                '--batch', str(self._page_value('batch', 4)), '--margin', str(self.margin.value()),
                '--seed', str(self._page_value('seed', 0))]
        for opt, name in (('--world', 'world'), ('--rig', 'rig')):
            val = self._page_value(name, '')
            if val:
                argv += [opt, val]
        fps = self._page_value('fps', 20)
        if fps != 20:
            argv += ['--fps', str(fps)]
        argv += ['--stl', req['stl']] if req.get('stl') else ['--rule', req['rule']]
        argv += ['--scale', str(req.get('scale', 1.0)), '--smooth', str(req.get('smooth') or 0.0)]
        if getattr(w, 'fake_carla', ''):
            argv += ['--fake-carla', w.fake_carla]
        return argv

    def start(self):
        w = self.win
        if self.spec is None or (self.job is not None and self.job.state not in FINAL):
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        argv = [sys.executable, '-m', 'falcon_autonomy'] + self.argv()
        self.job = w.queue.submit(JobSpec('study', f'Falsify {self.scenario.currentText()} '
                                                   f'({self.method.currentText()}, {self.budget.value()})', argv,
                                          memory_gb=0.5, resumable=True))
        if hasattr(w, 'jobs_dock'):
            w.jobs_dock.raise_()
        self.kv.set('State', self.job.summary())
        if not hasattr(self, '_poll'):
            self._poll = QTimer(self)
            self._poll.timeout.connect(self.poll)
            self._poll.start(2000)

    def poll(self):
        if self.job is not None:
            self.load()
            if self.job.state in FINAL:
                self._poll.stop()
                del self._poll

    def rows(self) -> list:
        d = self.study_dir()
        return SO.study_rows(d) if d else []

    def load(self):
        rows = [r for r in self.rows() if r.get('stl')]
        busy = self.job is not None and self.job.state not in FINAL
        s = SO.summarise(rows, self.margin.value())
        self.kv.set('State', self.job.summary() if busy else (f'{len(rows)} runs scored' if rows else
                                                              'no runs of this requirement on this scenario yet'))
        self.kv.set('Best ρ', '—' if s['best_rho'] is None else f'{s["best_rho"]}'
                    + (' — falsified' if s['falsified'] else ''))
        self.kv.set('Falsified', str(len(s['falsified'])))
        self.kv.set('Near misses', str(len(s['near_misses'])))
        self.kv.set('Unknown', str(s['unknown']) + (f' (signals not recorded: {", ".join(s["missing_signals"])})'
                                                     if s['missing_signals'] else ''))
        cur = self.runs.currentItem().data(0, 256).get('key') if self.runs.currentItem() is not None else None
        self.runs.clear()
        ordered = sorted(rows, key=lambda r: (math.isnan(SO.fnum(r['stl'].get('rho'))), SO.fnum(r['stl'].get('rho'))
                                              if not math.isnan(SO.fnum(r['stl'].get('rho'))) else 0.0))
        for r in ordered:
            st = r['stl']
            rho = SO.fnum(st.get('rho'))
            verdict = ('falsified' if rho < 0 else 'near miss' if rho < self.margin.value() else 'satisfied') \
                if not math.isnan(rho) else 'unknown'
            it = QTreeWidgetItem([str(st.get('rho')), verdict, '—' if st.get('worst_t') is None else
                                  f'{st["worst_t"]:.2f}', ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else
                                                                    f'{k}={v}' for k, v in sorted(
                                                                        (r.get('values') or {}).items())),
                                  os.path.basename(r.get('run') or '')])
            tone = {'falsified': 'critical', 'near miss': 'warning'}.get(verdict)
            if tone:
                it.setForeground(1, QColor(STATUS[tone]))
            it.setData(0, 256, r)
            self.runs.addTopLevelItem(it)
            if r.get('key') == cur:
                self.runs.setCurrentItem(it)
        if self.runs.currentItem() is None and self.runs.topLevelItemCount():
            self.runs.setCurrentItem(self.runs.topLevelItem(0))
        elif not self.runs.topLevelItemCount():
            self.trace.set_series([], [])

    def show_trace(self, it):
        if it is None:
            return
        r = it.data(0, 256)
        okey = r.get('objective_key') or ''
        p = SO.series_path(self.study_dir(), r.get('run') or '', okey)
        try:
            with open(p, encoding='utf-8') as fh:
                sc = json.load(fh)
        except (OSError, ValueError):
            self.trace.set_series([], [], title='no series saved for this run')
            return
        self.trace.set_series(sc.get('t'), sc.get('rho_series'), sc.get('worst_t'),
                              f'{os.path.basename(r.get("run") or "")}: ρ = {sc.get("rho")}')

    def shutdown(self):
        if hasattr(self, '_poll'):
            self._poll.stop()
