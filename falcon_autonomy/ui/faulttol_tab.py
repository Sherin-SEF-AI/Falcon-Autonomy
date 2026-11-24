"""Stack testing ▸ Fault tolerance: how far each sensor fault can go before the stack or perception fails
(core.faulttol; docs/fault-tolerance.md).

A campaign's table — one row per (sensor, fault): the largest tolerated severity in mono numerals with its unit, the
bracket [last pass, first fail], a meter of how much of that tolerance the user's requirement uses, and the status as
a word with its colour (met, not met, no requirement; a cell that could not be measured says why). The requirement of
the selected row's fault is set beside the table and kept in the settings (faulttol_requirements). Selecting a row
draws its response curve: one axis, the criterion's threshold as a dashed line, each evaluated severity a point that
names its value on hover. Below: the most fragile sensor and the calibration and synchronisation accuracy the
tolerances imply — derived from this stack and these runs, and labelled so.

A closed-loop campaign is started here on the page's world and rig (`falcon faulttol run … --closed-loop`, a job);
an offline one on recorded runs from the command line or the assistant."""
from __future__ import annotations

import sys

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QToolTip, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import faulttol as FT
from ..core.jobs import FINAL, JobSpec
from . import motion
from .theme import THEME, mono
from .widgets import Form, Meter, Stat, button, label

CRITERIA = (('degraded', 'The stack declares a failure'), ('criticality', 'Criticality reaches 0.5'),
            ('fail', 'A run fails its checks'))


def fmt(v, unit='') -> str:
    if v is None:
        return '—'
    return f'{float(v):.3g}' + (f' {unit}' if unit else '')


class Curve(QWidget):
    """One cell's response against the severity: a line through the evaluated points, the threshold dashed, failing
    points in the failed colour (and square, so colour is not the only cue); hover names a point."""

    def __init__(self):
        super().__init__()
        self.cell = None
        self.setMinimumHeight(170)
        self.setMouseTracking(True)
        self.draw = motion.Driver(self)
        self._pts = []                                   # screen (x, y, severity, response, failed)

    def set_cell(self, cell: dict | None):
        self.cell = cell
        self.draw.start()
        self.update()

    def paintEvent(self, _ev):                                    # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        p.setFont(mono(11))
        c = self.cell
        self._pts = []
        curve = (c or {}).get('curve') or []
        if not curve:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       'Choose a row to see its response curve' if c is None else
                       (c.get('note') or 'This cell has no evaluations'))
            return
        x0, y0, w, h = 52, 14, self.width() - 70, self.height() - 46
        thr = float(c.get('threshold') or 0.0)
        xs = [s for s, _r, _f in curve]
        ys = [r for _s, r, _f in curve]
        xa, xb = min(xs), max(max(xs), min(xs) + 1e-9)
        lo, hi = min(min(ys), thr, 0.0), max(max(ys), thr)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        hi += 0.08 * (hi - lo)

        def X(s):
            return x0 + w * (s - xa) / (xb - xa)

        def Y(r):
            return y0 + h * (hi - r) / (hi - lo)
        p.setPen(QColor(THEME.tok['grid2']))
        p.drawLine(QPointF(x0, y0 + h), QPointF(x0 + w, y0 + h))
        p.drawLine(QPointF(x0, y0), QPointF(x0, y0 + h))
        crit, good, ink = QColor(THEME.status('critical')), QColor(THEME.status('good')), QColor(THEME.tok['ink2'])
        p.setPen(QPen(crit, 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x0, Y(thr)), QPointF(x0 + w, Y(thr)))
        p.drawText(QPointF(x0 + w - 150, Y(thr) - 4), f'fails above {thr:.3g}')
        b = c.get('bracket') or [None, None]
        if b[0] is not None and b[1] is not None:                 # the bracket: a faint band between pass and fail
            band = QColor(THEME.tok['soft'])
            p.fillRect(QRectF(X(b[0]), y0, max(X(b[1]) - X(b[0]), 1.0), h), band)
        n = max(int(round(self.draw.value * (len(curve) - 1))), 0)
        p.setPen(QPen(ink, 1.5))
        for (sa, ra, _), (sb, rb, _f) in list(zip(curve, curve[1:]))[:n]:
            p.drawLine(QPointF(X(sa), Y(ra)), QPointF(X(sb), Y(rb)))
        for s, r, failed in curve[:n + 1]:
            col = crit if failed else good
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            if failed:
                p.drawRect(QRectF(X(s) - 3.5, Y(r) - 3.5, 7, 7))
            else:
                p.drawEllipse(QPointF(X(s), Y(r)), 3.5, 3.5)
            self._pts.append((X(s), Y(r), s, r, failed))
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QPointF(6, Y(hi) + 10), f'{hi:.3g}')
        p.drawText(QPointF(6, y0 + h), f'{lo:.3g}')
        p.drawText(QPointF(x0, self.height() - 22), f'{xa:.3g}')
        p.drawText(QPointF(x0 + w - 40, self.height() - 22), f'{xb:.3g} {c.get("unit", "")}')
        p.drawText(QPointF(x0, self.height() - 6),
                   f'{c.get("response_unit", "response")} against the severity · circle passes, square fails')

    def mouseMoveEvent(self, ev):                                 # noqa: N802
        pos = ev.position()
        best = min(self._pts, key=lambda q: (q[0] - pos.x()) ** 2 + (q[1] - pos.y()) ** 2, default=None)
        if best and (best[0] - pos.x()) ** 2 + (best[1] - pos.y()) ** 2 <= 144:
            QToolTip.showText(ev.globalPosition().toPoint(), self.tip(best), self)
        else:
            QToolTip.hideText()

    def tip(self, q) -> str:
        c = self.cell or {}
        return f'{q[2]:.4g} {c.get("unit", "")}: {q[3]:.4g} ({c.get("response_unit", "response")}) — ' + \
            ('fails' if q[4] else 'passes')


class FaultTolTab(QWidget):
    """The Fault tolerance section (module doc). owner: the Stack testing page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.doc = None
        self.job = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.campaign = QComboBox()
        self.campaign.setMinimumWidth(180)
        self.campaign.currentIndexChanged.connect(lambda _: self.load())
        top.addWidget(label('Campaign'))
        top.addWidget(self.campaign, 1)
        top.addWidget(button('Refresh', self.refresh))
        v.addLayout(top)
        stats = QHBoxLayout()
        self.fragile = Stat('Most fragile sensor')
        self.cells_stat = Stat('Cells measured')
        self.req_stat = Stat('Requirements met')
        for s in (self.fragile, self.cells_stat, self.req_stat):
            stats.addWidget(s, 1)
        v.addLayout(stats)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(['Sensor', 'Fault', 'Tolerated', 'Bracket [pass, fail]', 'Required',
                                    'Of the tolerance', 'Status'])
        self.table.setRootIsDecorated(False)
        self.table.setMinimumHeight(170)
        self.table.setColumnWidth(1, 190)
        self.table.setColumnWidth(3, 170)
        self.table.setColumnWidth(5, 120)
        self.table.currentItemChanged.connect(lambda it, _p: self.select(it))
        v.addWidget(self.table)
        row = QHBoxLayout()
        self.req_label = label('Requirement', muted=True)
        self.req = QDoubleSpinBox()
        self.req.setDecimals(3)
        self.req.setRange(0.0, 100000.0)
        self.req.setSpecialValueText('none')
        self.req.setToolTip('The severity of the selected row\'s fault that the system must tolerate (0: none). Kept '
                            'in the settings; the table\'s meters and statuses are judged against it.')
        self.req.setEnabled(False)
        self.req.valueChanged.connect(self.set_requirement)
        row.addWidget(self.req_label)
        row.addWidget(self.req)
        row.addStretch(1)
        v.addLayout(row)
        self.curve = Curve()
        v.addWidget(self.curve)
        self.derived = label('', small=True, wrap=True, sel=True)
        v.addWidget(self.derived)
        f = Form()
        self.scenario = QComboBox()
        self.criterion = QComboBox()
        for k, t in CRITERIA:
            self.criterion.addItem(t, k)
        self.name = QLineEdit()
        self.name.setPlaceholderText('a name for the campaign')
        f.row('Scenario', self.scenario)
        f.row('A cell fails when', self.criterion)
        f.row('Name', self.name)
        v.addLayout(f)
        b = QHBoxLayout()
        self.start_btn = button('Run a closed-loop campaign', self.start, role='primary',
                                tip='falcon faulttol run … --closed-loop (a job): stack runs with each fault at '
                                    'graded severities')
        b.addWidget(self.start_btn)
        self.state = label('', small=True, muted=True)
        b.addWidget(self.state, 1)
        v.addLayout(b)
        self.refresh()

    def showEvent(self, ev):                                      # noqa: N802
        super().showEvent(ev)
        self.refresh()

    # ------------------------------------------------------------------ data
    def _ws(self) -> str:
        try:
            return self.win.workspace() if self.win.project.path else ''
        except AttributeError:
            return ''

    def requirements(self) -> dict:
        st = getattr(self.win, 'settings', None)
        saved = dict((st.get('faulttol_requirements') if st is not None else None) or {})
        return dict((self.doc or {}).get('requirements') or {}, **saved)

    def refresh(self):
        cur = self.campaign.currentData()
        ws = self._ws()
        self.campaign.blockSignals(True)
        self.campaign.clear()
        for n in reversed(FT.campaigns(ws) if ws else []):
            self.campaign.addItem(n, n)
        k = self.campaign.findData(cur)
        self.campaign.setCurrentIndex(k if k >= 0 else 0)
        self.campaign.blockSignals(False)
        cur_s = self.scenario.currentData()
        self.scenario.clear()
        scns = self.page.scenarios() if hasattr(self.page, 'scenarios') else []
        for ident, s in scns:
            self.scenario.addItem(s['name'], ident)
        k = self.scenario.findData(cur_s)
        self.scenario.setCurrentIndex(k if k >= 0 else 0)
        self.load()

    def load(self):
        name, ws = self.campaign.currentData(), self._ws()
        self.doc = None
        if name and ws:
            try:
                self.doc = FT.load(FT.campaign_dir(ws, name))
            except (OSError, ValueError, FT.FaultTolError):
                self.doc = None
        self.fill()

    def fill(self, keep: tuple | None = None):
        doc = self.doc
        self.table.blockSignals(True)
        self.table.clear()
        self.table.blockSignals(False)
        self.curve.set_cell(None)
        self.req.setEnabled(False)
        if doc is None:
            self.fragile.set('—', sub='no campaign yet: run one below, or falcon faulttol run')
            self.cells_stat.set('—')
            self.req_stat.set('—')
            self.derived.setText('')
            return
        reqs = self.requirements()
        met = judged = 0
        for c in doc['cells']:
            word, tone = FT.requirement_status(c, reqs)
            b = c.get('bracket') or [None, None]
            it = QTreeWidgetItem([c['sensor'], c['title'], fmt(c.get('tolerated'), c['unit']),
                                  f'[{fmt(b[0])}, {fmt(b[1])}]', fmt(reqs.get(c['fault'])) if reqs.get(c['fault'])
                                  else '—', '', word])
            it.setData(0, 256, c)
            for col in (2, 3, 4):
                it.setFont(col, mono(12))
                it.setTextAlignment(col, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if tone:
                it.setForeground(6, QColor(THEME.status(tone)))
            it.setToolTip(6, c.get('note') or c['status'].replace('_', ' '))
            it.setToolTip(1, c['model'])
            self.table.addTopLevelItem(it)
            m = Meter(100)
            req = reqs.get(c['fault'])
            if req and c.get('tolerated'):
                m.set(float(req) / float(c['tolerated']))          # how much of the tolerance the requirement uses
                m.setToolTip(f'the requirement {fmt(req, c["unit"])} is {100 * float(req) / float(c["tolerated"]):.0f} '
                             f'% of the tolerated {fmt(c["tolerated"], c["unit"])}')
            elif req:
                m.set(1.0)
            self.table.setItemWidget(it, 5, m)
            if word in ('met', 'not met'):
                judged += 1
                met += word == 'met'
            if keep and keep == (c['sensor'], c['fault']):
                self.table.setCurrentItem(it)
        mf = FT.most_fragile(doc['cells'], reqs)
        if mf:
            self.fragile.set(mf['sensor'], sub=f'{mf["fault"].replace("_", " ")}: {mf["value"]:.2g} of the '
                             f'{mf["basis"]}', tone='critical' if mf['basis'] == 'requirement' and mf['value'] < 1
                             else '')
            self.fragile.sub.setToolTip(mf['about'])
        else:
            self.fragile.set('—', sub='nothing measured')
        n_ref = sum(c['status'] == 'refused' for c in doc['cells'])
        self.cells_stat.set(str(len(doc['cells']) - n_ref), sub=f'{n_ref} not measured' if n_ref else doc['mode']
                            .replace('_', ' '))
        self.req_stat.set(f'{met} of {judged}' if judged else '—', sub='set a requirement per fault below'
                          if not judged else ('all met' if met == judged else f'{judged - met} not met'),
                          tone='' if met == judged else 'critical')
        lines = [s['statement'] for s in doc.get('calibration') or []]
        self.derived.setText(('Derived — ' + '\nDerived — '.join(lines) + f'\n({FT.STATEMENT_NOTE})') if lines
                             else '')
        if not keep and self.table.topLevelItemCount():
            self.table.setCurrentItem(self.table.topLevelItem(0))

    def select(self, it):
        c = it.data(0, 256) if it is not None else None
        self.curve.set_cell(c)
        self.req.blockSignals(True)
        self.req.setEnabled(c is not None)
        if c is not None:
            self.req.setSuffix(f' {c["unit"]}')
            self.req.setSingleStep(max(float(c['max']) / 50.0, 0.001))
            self.req.setValue(float(self.requirements().get(c['fault']) or 0.0))
            self.req_label.setText(f'Requirement for {c["title"].lower()}')
        self.req.blockSignals(False)

    def set_requirement(self, value: float):
        it = self.table.currentItem()
        c = it.data(0, 256) if it is not None else None
        st = getattr(self.win, 'settings', None)
        if c is None or st is None:
            return
        reqs = dict(st.get('faulttol_requirements') or {})
        if value > 0:
            reqs[c['fault']] = float(value)
        else:
            reqs.pop(c['fault'], None)
            if self.doc is not None:
                (self.doc.get('requirements') or {}).pop(c['fault'], None)
        st.set('faulttol_requirements', reqs)
        self.fill(keep=(c['sensor'], c['fault']))

    # ------------------------------------------------------------------ a closed-loop campaign
    def argv(self) -> list:
        w = self.win
        world = self.page.world.currentData() if hasattr(self.page, 'world') else ''
        rig = self.page.rig.currentData() if hasattr(self.page, 'rig') else ''
        a = ['faulttol', 'run', w.project.path, '--name', self.name.text().strip(), '--closed-loop', '--scenario',
             str(self.scenario.currentData()), '--criterion', self.criterion.currentData()]
        if world:
            a += ['--world', world]
        if rig:
            a += ['--rig', rig]
        for k, val in (getattr(w, 'settings', None).get('faulttol_requirements') or {}).items():
            a += ['--require', f'{k}={float(val):g}']
        fake = getattr(getattr(w, 'service', None), 'fake_carla', '')
        if fake:
            a += ['--fake-carla', fake]
        return a

    def start(self):
        import re
        w = self.win
        if self.job is not None and self.job.state not in FINAL:
            return
        name = self.name.text().strip()
        if not re.match(r'^[A-Za-z0-9][\w.-]{0,63}$', name):
            self.state.setText('Give the campaign a name (letters, digits, - and _).')
            return
        if self.scenario.currentData() is None:
            self.state.setText('Choose a scenario.')
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        self.job = w.queue.submit(JobSpec('faulttol', f'Fault tolerance {name} ({self.scenario.currentText()})',
                                          [sys.executable, '-m', 'falcon_autonomy'] + self.argv(), memory_gb=0.5))
        self.state.setText(self.job.summary())
        if not hasattr(self, '_poll'):
            self._poll = QTimer(self)
            self._poll.timeout.connect(self.poll)
            self._poll.start(2000)

    def poll(self):
        if self.job is None:
            return
        self.state.setText(self.job.summary())
        if self.job.state in FINAL:
            self._poll.stop()
            del self._poll
            name = self.name.text().strip()
            self.refresh()
            k = self.campaign.findData(name)
            if k >= 0:
                self.campaign.setCurrentIndex(k)
                motion.flash_item(self.table.topLevelItem(0), self.table.columnCount())
