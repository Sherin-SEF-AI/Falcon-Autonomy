"""Rare cases ▸ Two-speed search: a fast kinematic simulator screens a scenario's parameters and CARLA confirms the
most promising (core.fastsim; docs/two-speed-search.md).

The section takes the scenario, the budget in CARLA runs and how it is split — runs per round, the share of each
round spent on random controls — the number of cheap candidates screened and the rank correlation below which the
screen is not trusted; the world, rig, seed and failure threshold are the page's. *Search at two speeds* queues
`falcon prescreen run` as a job (resumable; its study is the page's criticality search of that scenario, so the page's
curve, failing cases and failure map show its runs too). A scenario the fast simulator cannot run says why and is not
queued. While it runs a sweep shows the CARLA runs spent; the result: the scatter of the cheap against the CARLA
criticality of every confirmed run with the rank correlation and the trust word, the false-negative risk the random
controls give, and the confirmed failures (each a CARLA run, the cheap value beside it).

4.2: the result's fidelity as chips (the confirmed runs' level from their run.json, the screen's 'kinematic'), per row
in the failures; the running rank correlation over the pairs (RhoTrace) with the trust threshold as a labelled dashed
line that follows the *Trust at ρ ≥* box; and, when ρ is under it or the screen is not trusted, a compact *Why
correlation is low* panel (core.fastsim.diagnose: the pairs whose ranks disagree most — each openable and sendable —,
ties in the cheap value, too few pairs, the parameter the disagreement grows with, a tiny camera). A failure row
selects its run as the window's active one and has *Send to ▸*."""
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
from .study_bits import Badges, FidelitySlot, link_runs
from .theme import THEME, mono
from .widgets import Details, Form, KeyValues, Stat, button, copyable_ids, fidelity_item, label, short_id

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


def running_spearman(pairs, cap: int = 2000) -> list:
    """[(k, ρ over the first k pairs)] for k = 3 … n (ρ None when not defined): how the rank correlation moved as
    the pairs arrived."""
    ok = [p for p in pairs or () if p.get('high') is not None][:cap]
    lo = [float(p['low']) for p in ok]
    hi = [float(p['high']) for p in ok]
    return [(k, F.spearman(lo[:k], hi[:k])) for k in range(3, len(ok) + 1)]


class RhoTrace(QWidget):
    """The rank correlation of the cheap and the CARLA criticality over the pairs as they arrived (one axis, ρ in
    [−1, 1]): the trust threshold as a labelled dashed line (the *Trust at ρ ≥* box's value, live), the pair count
    from which trust is judged as a dotted mark, zero as a recessive line. Hover reads ρ after k pairs."""

    def __init__(self):
        super().__init__()
        self.series, self.threshold, self.min_pairs, self.trust = [], 0.5, 12, ''
        self.next_threshold = None      # the box's value when it differs from the shown study's own threshold
        self.hover = -1
        self.setMinimumHeight(150)
        self.setMouseTracking(True)

    def set_pairs(self, pairs, min_pairs: int = 12, trust: str = ''):
        self.series = running_spearman(pairs)
        self.min_pairs, self.trust = int(min_pairs or 12), trust
        self.hover = -1
        self.update()

    def set_threshold(self, v: float, next_v: float | None = None):
        """v: the threshold the shown study was judged with (or the box's value before any study); next_v: the box's
        value when it differs — drawn faint as the next search's threshold, never as this study's."""
        self.threshold = float(v)
        self.next_threshold = None if next_v is None or abs(float(next_v) - self.threshold) < 1e-9 else float(next_v)
        self.update()

    def threshold_label(self) -> str:
        return f'trust at ρ ≥ {self.threshold:.2f}'

    def _geom(self):
        return 44.0, 12.0, self.width() - 60.0, self.height() - 40.0

    def _n(self) -> int:
        return self.series[-1][0] if self.series else 0

    def _x(self, k, x0, w):
        n = max(self._n(), self.min_pairs, 4)
        return x0 + w * (k - 1) / max(n - 1, 1)

    def mouseMoveEvent(self, ev):                               # noqa: N802
        if not self.series:
            return
        x0, _y0, w, _h = self._geom()
        best, at = 1e9, -1
        for i, (k, _r) in enumerate(self.series):
            d = abs(self._x(k, x0, w) - ev.position().x())
            if d < best:
                best, at = d, i
        if at != self.hover:
            self.hover = at
            k, r = self.series[at]
            self.setToolTip(f'after {k} pairs: ρ ' + ('not defined (a side is constant)' if r is None else
                                                      f'{r:+.2f}') +
                            (f'\n(trust is judged from {self.min_pairs} pairs)' if k < self.min_pairs else ''))
            self.update()

    def leaveEvent(self, _ev):                                  # noqa: N802
        self.hover = -1
        self.update()

    def paintEvent(self, _ev):                                  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x0, y0, w, h = self._geom()

        def Y(v):
            return y0 + h * (0.5 - 0.5 * min(max(v, -1.0), 1.0))
        p.setFont(mono(10))
        for t in (-1.0, 0.0, 1.0):                              # recessive grid, labelled in mono
            p.setPen(QPen(QColor(THEME.tok['soft']), 1))
            p.drawLine(QPointF(x0, Y(t)), QPointF(x0 + w, Y(t)))
            p.setPen(QColor(THEME.tok['off']))
            p.drawText(QRectF(0, Y(t) - 8, x0 - 8, 16), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       f'{t:+.0f}' if t else '0')
        thr = QColor(THEME.tok['ink2'])                         # the trust threshold: dashed, labelled
        p.setPen(QPen(thr, 1.2, Qt.PenStyle.DashLine))
        ty = Y(self.threshold)
        p.drawLine(QPointF(x0, ty), QPointF(x0 + w, ty))
        p.setPen(thr)
        p.setFont(mono(11))
        first = next((r for _k, r in self.series if r is not None), None)
        below = (first is not None and first >= self.threshold) or self.threshold > 0.85   # away from the line
        if self.threshold < -0.85:
            below = False
        p.drawText(QRectF(x0 + 6, ty + 3 if below else ty - 17, 200, 14),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self.threshold_label())
        if self.next_threshold is not None:                     # the box changed since: the next search's, faint
            nt = Y(self.next_threshold)
            p.setPen(QPen(QColor(THEME.tok['muted']), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(x0, nt), QPointF(x0 + w, nt))
            p.setPen(QColor(THEME.tok['muted']))
            p.setFont(mono(10))
            p.drawText(QRectF(x0 + w - 206, nt + 3 if self.next_threshold > 0.85 else nt - 16, 200, 14),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       f'next search: ρ ≥ {self.next_threshold:.2f}')
        mx = self._x(self.min_pairs, x0, w)                     # the trust rule's pair count
        p.setPen(QPen(QColor(THEME.tok['muted']), 1, Qt.PenStyle.DotLine))
        p.drawLine(QPointF(mx, y0), QPointF(mx, y0 + h))
        p.setPen(QColor(THEME.tok['muted']))
        p.setFont(mono(10))
        fits = mx + 150 < x0 + w
        p.drawText(QRectF(mx + 4 if fits else mx - 154, y0 + h - 16, 150, 14),
                   int((Qt.AlignmentFlag.AlignLeft if fits else Qt.AlignmentFlag.AlignRight)
                       | Qt.AlignmentFlag.AlignVCenter), f'judged from {self.min_pairs} pairs')
        pts = [(k, r) for k, r in self.series if r is not None]
        if not pts:
            p.setPen(QColor(THEME.tok['muted']))
            p.setFont(mono(11))
            p.drawText(QRectF(x0, y0, w, h), int(Qt.AlignmentFlag.AlignCenter), 'ρ needs 3 confirmed pairs')
        else:
            tone = QColor(THEME.status(TRUST_TONE.get(self.trust, 'warning'))) if self.trust else \
                QColor(THEME.tok['ink'])
            p.setPen(QPen(QColor(THEME.tok['ink']), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            for (ka, ra), (kb, rb) in zip(pts, pts[1:]):
                p.drawLine(QPointF(self._x(ka, x0, w), Y(ra)), QPointF(self._x(kb, x0, w), Y(rb)))
            k, r = pts[-1]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(tone)
            p.drawEllipse(QPointF(self._x(k, x0, w), Y(r)), 4.0, 4.0)
            p.setPen(tone)
            p.setFont(mono(11))                                 # the value, in the half the line is not in
            ly = Y(-0.45) if r > 0 else Y(0.55)
            p.drawText(QRectF(x0 + w - 236, ly - 7, 230, 14),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       f'ρ {r:+.2f} after {k} pairs' + (f' · {self.trust}' if self.trust else ''))
            if 0 <= self.hover < len(self.series) and self.series[self.hover][1] is not None:
                kh, rh = self.series[self.hover]
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(THEME.tok['ink']))
                p.drawEllipse(QPointF(self._x(kh, x0, w), Y(rh)), 3.5, 3.5)
        p.setFont(mono(11))
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QPointF(x0, self.height() - 8), f'pairs → ({self._n()})  line: ρ of cheap vs CARLA so far')


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
        self.intro = Badges()
        self.intro.set([('Screen', 'kinematic, straight road', 'the director\'s lane follower on a straight road, the '
                         'same criticality formula: thousands of parameter sets in seconds'),
                        ('Confirms', 'CARLA', 'CARLA runs only the most promising'),
                        ('Controls', 'random share', 'a share of random controls measures how far the two agree')])
        v.addWidget(self.intro)
        v.addWidget(label('The cheap value is a screen, never a result.', muted=True, small=True))
        v.addWidget(Details('A fast kinematic simulator (the director\'s lane follower on a straight road, the same '
                            'criticality formula) rates thousands of parameter sets in seconds; CARLA runs only the '
                            'most promising, and a share of random controls that measure how far the two agree. The '
                            'cheap value is a screen, never a result. Trust is judged from 12 confirmed pairs: below '
                            'the threshold ρ the rest of the budget goes to a plain search. docs/two-speed-search.md.'))
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
        self.min_rho.valueChanged.connect(lambda v: self.threshold_changed(v))
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
        fr = QHBoxLayout()
        self.fid_runs = FidelitySlot('Confirmed runs')
        self.fid_screen = FidelitySlot('Screen')
        self.fid_screen.set(F.SCREEN_FIDELITY)
        fr.addWidget(self.fid_runs)
        fr.addWidget(self.fid_screen)
        fr.addStretch(1)
        v.addLayout(fr)
        v.addWidget(label('Rank correlation as the pairs arrived', h=True))
        self.rho_trace = RhoTrace()
        self.rho_trace.set_threshold(self.min_rho.value())
        v.addWidget(self.rho_trace)
        self.why_low = QWidget()                                 # Why correlation is low (core.fastsim.diagnose)
        wl = QVBoxLayout(self.why_low)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(6)
        wl.addWidget(label('Why correlation is low', h=True))
        self.why_kv = KeyValues()
        wl.addWidget(self.why_kv)
        self.disagree = QTreeWidget()
        self.disagree.setHeaderLabels(['Rank gap', 'Cheap', 'CARLA', 'Values', 'Run'])
        self.disagree.setRootIsDecorated(False)
        self.disagree.setMaximumHeight(110)
        self.disagree.setToolTip('The pairs whose cheap rank and CARLA rank differ most; right-click to send a run on')
        self.disagree.itemDoubleClicked.connect(lambda it, _c: self.browse(it))
        copyable_ids(self.disagree, 4)
        link_runs(self.disagree, self.win, self._run_of, exclude='rarecases',
                  actions=lambda it: [('Browse its run', lambda: self.browse(it))])
        wl.addWidget(self.disagree)
        self.why_details = Details('')
        wl.addWidget(self.why_details)
        self.why_low.hide()
        v.addWidget(self.why_low)
        v.addWidget(label('Cheap against CARLA criticality', h=True))
        self.scatter = Scatter()
        v.addWidget(self.scatter)
        v.addWidget(label('Confirmed failures (CARLA runs)', h=True))
        self.failures = QTreeWidget()
        self.failures.setHeaderLabels(['CARLA', 'Cheap', 'Found by', 'Values', 'Fidelity', 'Run'])
        self.failures.setRootIsDecorated(False)
        self.failures.setMinimumHeight(120)
        self.failures.itemDoubleClicked.connect(lambda it, _c: self.browse(it))
        copyable_ids(self.failures, 5)
        link_runs(self.failures, self.win, self._run_of, exclude='rarecases',
                  actions=lambda it: [('Browse its run', lambda: self.browse(it))])
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
            self.fid_runs.set('unknown')
            self.rho_trace.set_pairs([])
            self.show_diagnosis(None)
            return
        f = d['gap']
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
        lv = d.get('fidelity') or 'unknown'
        self.fid_runs.set(lv, 'Mixed fidelity: the confirmed runs pool simulators of different fidelity; compare '
                              'with care.' if lv == 'mixed' else '', 'critical' if lv == 'mixed' else '')
        self.fid_screen.set(d.get('screen_fidelity') or F.SCREEN_FIDELITY)
        self.rho_trace.set_pairs(d['pairs'], f.get('min_pairs') or 12, d['trust'])
        self.rho_trace.set_threshold(self._study_threshold(d), self.min_rho.value())
        self.show_diagnosis(d)
        before = {self.failures.topLevelItem(i).data(0, 256).get('run')
                  for i in range(self.failures.topLevelItemCount())}
        self.failures.clear()
        for x in d.get('failures') or ():
            it = QTreeWidgetItem([f'{x["value"]:.3f}', f'{x["low"]:.3f}', ROLE_WORDS.get(x['role'], x['role']),
                                  ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                            for k, v in sorted(x['values'].items())), '',
                                  short_id(os.path.basename(x.get('run') or ''))])
            fidelity_item(it, 4, x.get('fidelity') or 'unknown')
            it.setData(5, Qt.ItemDataRole.UserRole + 1, x.get('run') or '')
            it.setToolTip(5, x.get('run') or '')
            it.setData(0, 256, x)
            self.failures.addTopLevelItem(it)
            if before and x.get('run') not in before:             # a failure that just arrived eases in
                motion.flash_item(it, self.failures.columnCount())

    def browse(self, it):
        x = it.data(0, 256) if it is not None else None
        if x and x.get('run') and hasattr(self.win, 'go'):
            self.win.go('data').show_run(x['run'])

    @staticmethod
    def _run_of(it):
        x = it.data(0, 256) if it is not None else None
        return (x or {}).get('run') or ''

    def _study_threshold(self, d) -> float:
        """The trust threshold the shown study was judged with (its own min_spearman), else the box's value."""
        g = (d or {}).get('gap') or {}
        try:
            return float(g['min_spearman'])
        except (KeyError, TypeError, ValueError):
            return self.min_rho.value()

    def threshold_changed(self, v: float):
        """The trust threshold box sets the NEXT search's threshold: with no study shown the dashed line follows it;
        with a study shown its own threshold stays the line (its verdict was judged by it) and the box's value is a
        faint 'next search' line."""
        if self.doc is not None:
            self.rho_trace.set_threshold(self._study_threshold(self.doc), v)
        else:
            self.rho_trace.set_threshold(v)

    def _rig_doc(self):
        rid = self._page_value('rig', '')
        try:
            return self.win.project.item('rigs', rid) if rid else None
        except AttributeError:
            return None

    def show_diagnosis(self, d):
        """The *Why correlation is low* panel: shown when ρ is under the study's threshold or the screen is not
        trusted; key-value rows, the most disagreeing pairs, and the whole sentences behind *Details*."""
        self.diagnosis = None
        if not d or not d.get('pairs'):
            self.why_low.hide()
            return None
        g = d.get('gap') or {}
        thr = self._study_threshold(d)                         # the verdict's own threshold, not the box's
        dg = F.diagnose(d['pairs'], thr, int(g.get('min_pairs') or 12), self._rig_doc())
        self.diagnosis = dg
        show = dg['low'] or d.get('trust') != 'trusted'
        self.why_low.setVisible(bool(show))
        if not show:
            return dg
        self.why_kv.clear()
        rho = dg['spearman']
        self.why_kv.set('ρ', ('not defined' if rho is None else f'{rho:+.2f}') + f' over {dg["pairs"]} pairs · trust '
                        f'at ≥ {thr:.2f}' + (f' · the screen is {d.get("trust")}' if d.get('trust') else ''))
        names = {'inverted': 'Order', 'few_pairs': 'Pairs', 'saturation': 'Ties', 'disagreement': 'Worst pair',
                 'parameter': 'Parameter', 'camera': 'Camera'}
        for f in dg['findings']:
            key = names.get(f['key'], f['key'])
            short = {'few_pairs': f'{dg["pairs"]} of {dg["min_pairs"]} needed',
                     'saturation': (f'{dg["ties"]["count"]} of {dg["pairs"]} at '
                                    f'{dg["ties"]["at"]:.3g}' if dg['ties'] else ''),
                     'parameter': (f'{dg["parameter"]["name"]} (ρ {dg["parameter"]["rho"]:+.2f})'
                                   if dg['parameter'] else ''),
                     'camera': (f'{dg["cameras"][0][0]} {dg["cameras"][0][1]} × {dg["cameras"][0][2]}'
                                if dg['cameras'] else ''),
                     'inverted': 'the screen orders against CARLA'}.get(f['key'])
            if f['key'] == 'disagreement' and dg['disagreements']:
                x = dg['disagreements'][0]
                short = f'cheap {x["low"]:.3f} vs CARLA {x["high"]:.3f}'
            self.why_kv.set(key, short or f['text'], tip=f['text'])
            if f['tone']:
                self.why_kv.vals[key].setStyleSheet(f'color: {THEME.status(f["tone"])}')
        if not dg['findings']:
            self.why_kv.set('Findings', 'none of the usual causes shows in these pairs')
        self.disagree.clear()
        for x in dg['disagreements']:
            it = QTreeWidgetItem([f'{x["rank_diff"]:.0f}', f'{x["low"]:.3f}', f'{x["high"]:.3f}',
                                  ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                            for k, v in sorted(x['values'].items())),
                                  short_id(os.path.basename(x['run'] or ''))])
            it.setData(4, Qt.ItemDataRole.UserRole + 1, x['run'] or '')
            it.setToolTip(4, x['run'] or '')
            it.setData(0, 256, x)
            self.disagree.addTopLevelItem(it)
        self.disagree.setVisible(bool(dg['disagreements']))
        self.why_details.set_text('\n'.join(['• ' + f['text'] + '.' for f in dg['findings']] + [
            'How it is read: the pairs are ranked by their cheap and by their CARLA criticality; a pair far apart in '
            'the two ranks is one the screen misorders. Ties in the cheap value cannot be ordered at all. The '
            'parameter named is the one whose value best orders the rank gaps (Spearman). These are leads measured on '
            'these pairs, not proof: a Spearman over a dozen pairs is itself noisy.']))
        return dg

    def shutdown(self):
        self.timer.stop()
