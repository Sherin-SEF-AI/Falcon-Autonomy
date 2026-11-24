"""Hardware ▸ Latency budget: measured compute latency fed back into closed-loop simulation (core.latencyloop;
docs/latency-loop.md).

The histogram is a latency profile of the workspace (latency/profiles/*.json): one axis in milliseconds, the counts as
bars labelled only where they matter (the median and p99 marked on the axis in mono), a tooltip per bar; whether the
profile was MEASURED (and on what) or is synthetic is said beside it. The table is a latency budget
(latency/<name>/latency-budget.json): per scenario case the largest latency at which it still passes, where it first
fails, and its status in a word and a colour — late only (it fails only when the computer is late), fails anyway (with
no latency at all), holds (no failure up to the swept maximum). "Find the budget" queues `falcon latency budget` for
the chosen scenario: a constant latency in whole simulation steps, read against the selected profile."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QToolTip, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import latencyloop as LL
from ..core.jobs import JobSpec
from ..core.provenance import read_json
from . import motion
from .theme import THEME, mono
from .widgets import button, flow_row, label

COLUMNS = ('Case', 'Status', 'Budget', 'Fails from', 'Runs')


def word(status: str) -> tuple:
    """A budget row's status as (word, tone)."""
    if status.startswith('fails only when late'):
        return 'late only', 'warning'
    if status == 'fails without latency':
        return 'fails anyway', 'critical'
    if status == 'no failure up to the maximum':
        return 'holds', 'good'
    return 'not evaluated', ''


class Histogram(QWidget):
    """A latency profile's histogram: bars over one axis (ms), the median and p99 marked and labelled on it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.setMouseTracking(True)
        self.profile, self.hist = None, None
        self.grow = motion.Driver(self)
        THEME.changed.connect(self._themed)          # a bound slot: let go when the widget is deleted

    def _themed(self, _name):
        self.update()

    def set(self, profile: dict | None):
        self.profile = profile
        self.hist = LL.histogram(profile) if profile else None
        if profile and motion.enabled():
            self.grow.start()
        self.update()

    def _plot(self) -> QRectF:
        return QRectF(8, 8, max(self.width() - 16, 10), max(self.height() - 34, 10))

    def bar_at(self, x: float):
        """The bin under a widget x, or None."""
        if not self.hist:
            return None
        r, n = self._plot(), len(self.hist['counts'])
        k = int((x - r.left()) / (r.width() / n))
        return k if 0 <= k < n and r.left() <= x <= r.right() else None

    def tip(self, k: int) -> str:
        e, c = self.hist['edges'], self.hist['counts']
        return f'{e[k]:.2f}–{e[k + 1]:.2f} ms: {c[k]} of {sum(c)} samples'

    def mouseMoveEvent(self, ev):                                 # noqa: N802
        k = self.bar_at(ev.position().x())
        if k is None:
            QToolTip.hideText()
        else:
            QToolTip.showText(ev.globalPosition().toPoint(), self.tip(k), self)

    def paintEvent(self, _ev):                                    # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._plot()
        ink, muted = QColor(THEME.tok['ink2']), QColor(THEME.tok['muted'])
        p.setFont(mono(10))
        if not self.hist:
            p.setPen(muted)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'no latency profile')
            return
        e, c = self.hist['edges'], self.hist['counts']
        top = max(c) or 1
        w = r.width() / len(c)
        frac = self.grow.value if (self.grow.running() and motion.enabled()) else 1.0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(ink))
        for k, n in enumerate(c):
            h = r.height() * n / top * frac
            if n:
                p.drawRect(QRectF(r.left() + k * w + 1, r.bottom() - max(h, 1.0), max(w - 2, 1), max(h, 1.0)))
        p.setPen(QPen(QColor(THEME.tok['strong']), 1))
        p.drawLine(int(r.left()), int(r.bottom()), int(r.right()), int(r.bottom()))
        span = (e[-1] - e[0]) or 1.0
        st = self.profile['stats']
        p.setPen(muted)
        p.drawText(QRectF(r.left(), r.bottom() + 3, 90, 16), Qt.AlignmentFlag.AlignLeft, f'{e[0]:.1f} ms')
        p.drawText(QRectF(r.right() - 90, r.bottom() + 3, 90, 16), Qt.AlignmentFlag.AlignRight, f'{e[-1]:.1f} ms')
        for key, text in (('median', 'median'), ('p99', 'p99')):
            x = r.left() + r.width() * min(max((st[key] - e[0]) / span, 0.0), 1.0)
            p.setPen(QPen(QColor(THEME.status('warning')), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(x), int(r.top()), int(x), int(r.bottom()))
            p.setPen(QColor(THEME.tok['ink']))
            lab = f'{text} {st[key]:.1f}'
            right = x + 4 + 86 <= r.right()
            p.drawText(QRectF(x + 4 if right else x - 90, r.top() + (0 if key == 'median' else 14), 86, 14),
                       Qt.AlignmentFlag.AlignLeft if right else Qt.AlignmentFlag.AlignRight, lab)


class LatencyTab(QWidget):
    """The Latency budget section (module doc). owner: the Hardware page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.job = None
        self.doc = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(label('Simulation assumes the stack answers instantly. Here the control computed at a step is '
                          'applied a latency later (rounded up to whole steps, held in between), and each scenario\'s '
                          'budget is the largest latency at which it still passes.', muted=True, wrap=True))
        self.profile = QComboBox()
        self.profile.currentIndexChanged.connect(lambda _i: self.show_profile())
        v.addWidget(flow_row(label('Profile'), self.profile, button('Refresh', self.refresh)))
        self.hist = Histogram()
        v.addWidget(self.hist)
        self.profile_note = label('', muted=True, wrap=True)
        v.addWidget(self.profile_note)
        self.scenario = QComboBox()
        self.run_btn = button('Find the budget', self.start, role='primary',
                              tip='falcon latency budget --constant: sweep a constant latency in whole steps and '
                                  'bisect the largest that still passes')
        v.addWidget(flow_row(label('Scenario'), self.scenario, self.run_btn))
        self.budget = QComboBox()
        self.budget.currentIndexChanged.connect(lambda _i: self.show_budget())
        v.addWidget(flow_row(label('Budget'), self.budget))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(140)
        v.addWidget(self.tree)
        v.addWidget(label('Fail only when the computer is late', h=True))
        self.late = label('—', mono=True, wrap=True, sel=True)
        v.addWidget(self.late)
        self.note = label('', muted=True, wrap=True)
        v.addWidget(self.note)
        THEME.changed.connect(self._themed)          # a bound slot: let go when the tab is deleted
        self.refresh()

    def _themed(self, _name):
        self.show_budget()

    def workspace(self) -> str:
        try:
            return self.win.workspace() or ''
        except AttributeError:
            return ''

    def scenarios(self) -> list:
        from ..core import scenarios as SC
        out = [(k, SC.LIBRARY[k].get('name', k) if isinstance(SC.LIBRARY[k], dict) else k) for k in SC.LIBRARY]
        try:
            out += [(s['id'], s.get('name', s['id'])) for s in self.win.project.doc.get('scenarios') or []]
        except AttributeError:
            pass
        return out

    def refresh(self, select: str = ''):
        ws = self.workspace()
        for combo, paths, keep in ((self.profile, LL.list_profiles(ws) if ws else [], self.profile.currentData()),
                                   (self.budget, LL.list_budgets(ws) if ws else [],
                                    select or self.budget.currentData())):
            combo.blockSignals(True)
            combo.clear()
            if combo is self.profile:
                combo.addItem('(none)', '')
            for p in paths:
                combo.addItem(os.path.basename(p)[:-5] if combo is self.profile
                              else os.path.basename(os.path.dirname(p)), p)
            k = combo.findData(keep)
            combo.setCurrentIndex(max(k, 0))
            combo.blockSignals(False)
        if not self.scenario.count():
            for k, name in self.scenarios():
                self.scenario.addItem(name, k)
        self.show_profile()
        self.show_budget()

    def show_profile(self):
        p = self.profile.currentData()
        doc = None
        if p:
            try:
                doc = LL.load_profile(p)
            except LL.LatencyError as e:
                self.profile_note.setText(str(e))
        self.hist.set(doc)
        if doc:
            s, src = doc['stats'], doc['source']
            what = 'MEASURED' if doc.get('measured') else 'NOT MEASURED (synthetic or given)'
            self.profile_note.setText(f'{what}: {src.get("kind")}, {src.get("device")}. {s["n"]} values; median '
                                      f'{s["median"]:.2f} ms, p95 {s["p95"]:.2f} ms, p99 {s["p99"]:.2f} ms, max '
                                      f'{s["max"]:.2f} ms. Consecutive latencies are drawn independently.')
        elif not p:
            self.profile_note.setText('No profile selected. Make one with `falcon latency profile` from a HIL '
                                      'session, a deployment\'s on-device profiling, a file or a synthetic '
                                      'distribution.')

    def show_budget(self, fresh: bool = False):
        self.tree.clear()
        p = self.budget.currentData()
        self.doc = None
        if not p:
            self.late.setText('—')
            self.note.setText('No latency budget in this workspace yet.' if self.workspace()
                              else 'Open a project to find latency budgets.')
            return
        try:
            d = self.doc = read_json(p)
        except (OSError, ValueError) as e:
            self.note.setText(f'The budget cannot be read: {e}')
            return
        u = d['unit']
        for r in d['cases']:
            w, tone = word(r['status'])
            at_least = '≥ ' if r['first_failure'] is None else ''
            b = '—' if r['budget'] is None else f'{at_least}{r["budget"]:g} {u}'
            f = '—' if r['first_failure'] is None else f'{r["first_failure"]:g} {u}'
            it = QTreeWidgetItem([r['case'], w.upper(), b, f, str(r['runs'])])
            it.setToolTip(0, r.get('name') or r['case'])
            tip = r.get('why', '')
            if r.get('over_budget_share') is not None:
                tip += f'; {100 * r["over_budget_share"]:.1f} % of the profile\'s latencies are over this budget'
            it.setToolTip(1, tip)
            it.setForeground(1, QBrush(QColor(THEME.status(tone) if tone else THEME.tok['muted'])))
            for c in (1, 2, 3, 4):
                it.setFont(c, mono(11))
            self.tree.addTopLevelItem(it)
            if fresh and motion.enabled():
                motion.flash_item(it, len(COLUMNS))
        for c in range(len(COLUMNS)):
            self.tree.resizeColumnToContents(c)
        self.late.setText(', '.join(d['fails_only_when_late']) or 'none')
        self.note.setText(f'{d["sweep"]} up to {d["grid_max"]:g} {u}; {d["quantisation"]}; {d["assumes"]}.')

    def argv(self) -> list:
        w = self.win
        key = self.scenario.currentData()
        argv = ['latency', 'budget', w.project.path, '--scenario', key, '--constant', '--name', f'latency-{key}']
        if self.profile.currentData():
            argv += ['--profile', self.profile.currentData()]
        if getattr(w, 'fake_carla', ''):
            argv += ['--fake-carla', w.fake_carla]
        return argv

    def start(self):
        w = self.win
        if not self.scenario.currentData():
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        self.job = w.queue.submit(JobSpec('study', f'Latency budget {self.scenario.currentText()}',
                                          [sys.executable, '-m', 'falcon_autonomy'] + self.argv(),
                                          memory_gb=0.5, resumable=True))
        self.note.setText(f'Queued: {self.job.summary()}. Refresh when it has finished.')
