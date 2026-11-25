"""Stack testing ▸ Fault tolerance: how far each sensor fault can go before the stack or perception fails
(core.faulttol; docs/fault-tolerance.md).

A campaign's table — one row per (sensor, fault): the largest tolerated severity in mono numerals with its unit, the
bracket [last pass, first fail], a meter of how much of that tolerance the user's requirement uses, and the status as
a word with its colour (met, not met, no requirement; a cell that could not be measured says why). The requirement of
the selected row's fault is set beside the table and kept in the settings (faulttol_requirements). Selecting a row
draws its response curve: one axis, the criterion's threshold as a dashed line, each evaluated severity a point that
names its value on hover. Below: the most fragile sensor and the calibration and synchronisation accuracy the
tolerances imply — derived from this stack and these runs, and labelled so.

A closed-loop campaign is started here on the world and rig chosen in the section (`falcon faulttol run …
--closed-loop`, a job); an offline one on a recorded run — sent here with Send to ▸ Fault tolerance, or chosen — with
`--runs` (a job; nothing starts without a click). The campaign's fidelity chip says what produced it (core.fidelity:
an offline analysis is `recorded`, with its runs' level beside it).

4.2: the failed or missing checks (a requirement not met, a cell failing without the fault, a requirement met inside
±10 % — no safety factor is applied —, a cell not measured, faults without a requirement) are flags with next steps:
Jump to config (the row and its requirement, or the campaign's scenario), Attach evidence (the campaign as a Solution
of the safety case, undoable there) and Run sensitivity check (core.faulttol.sensitivity: the verdict at the
requirement ±10 % and at both ends of the tolerance's bracket; nothing is run). The derived accuracy is one compact
row per sensor; how it is derived (and that it is not measured) is behind Details."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QToolTip, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import faulttol as FT
from ..core.jobs import FINAL, JobSpec
from . import motion
from .theme import THEME, mono
from .robustkit import RigWorldPicker, attach_evidence, flag, run_menu, set_context_run
from .widgets import (Details, Form, IdLabel, KeyValues, Meter, Stat, button, clear_layout, fidelity_chip,
                      label)

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
        self.sens = None                                   # the last sensitivity() shown
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.campaign = QComboBox()
        self.campaign.setMinimumWidth(180)
        self.campaign.currentIndexChanged.connect(lambda _: self.load())
        top.addWidget(label('Campaign'))
        top.addWidget(self.campaign, 1)
        self.chip = fidelity_chip('unknown')
        self.chip.setVisible(False)
        top.addWidget(self.chip)
        top.addWidget(button('Refresh', self.refresh))
        v.addLayout(top)
        stats = QHBoxLayout()
        self.fragile = Stat('Most fragile sensor')
        self.cells_stat = Stat('Cells measured')
        self.req_stat = Stat('Requirements met')
        for s in (self.fragile, self.cells_stat, self.req_stat):
            stats.addWidget(s, 1)
        v.addLayout(stats)
        self.flags_box = QVBoxLayout()                    # failed or missing checks with their next steps (4.2)
        self.flags_box.setSpacing(6)
        v.addLayout(self.flags_box)
        self.sens_label = label('', small=True, wrap=True, sel=True)
        self.sens_label.setVisible(False)
        v.addWidget(self.sens_label)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(['Sensor', 'Fault', 'Tolerated', 'Bracket [pass, fail]', 'Required',
                                    'Of the tolerance', 'Status'])
        self.table.setRootIsDecorated(False)
        self.table.setMinimumHeight(170)
        self.table.setColumnWidth(1, 190)
        self.table.setColumnWidth(3, 170)
        self.table.setColumnWidth(5, 120)
        self.table.currentItemChanged.connect(lambda it, _p: self.select(it))
        run_menu(self.win, self.table, self.run_of, exclude='faulttol')
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
        v.addWidget(label('Derived accuracy (not measured)', h=True))
        self.derived = KeyValues()                        # sensor: what its tolerances imply, one row each
        v.addWidget(self.derived)
        self.derived_more = Details('', 'How it is derived')
        v.addWidget(self.derived_more)
        f = Form()
        self.where = RigWorldPicker(self.win, labels=False, what='a closed-loop campaign spawns the rig\'s sensors '
                                                                  'in every run')
        f.row('Where', self.where)
        self.scenario = QComboBox()
        self.criterion = QComboBox()
        for k, t in CRITERIA:
            self.criterion.addItem(t, k)
        self.name = QLineEdit()
        self.name.setPlaceholderText('a name for the campaign')
        f.row('Scenario', self.scenario)
        f.row('A cell fails when', self.criterion)
        f.row('Name', self.name)
        orow = QHBoxLayout()
        self.offline_run = IdLabel('', keep=14)
        self.offline_none = label('none — Send to ▸ Fault tolerance from a run, or Choose…', muted=True)
        orow.addWidget(self.offline_run)
        orow.addWidget(self.offline_none)
        orow.addStretch(1)
        orow.addWidget(button('Choose…', self.pick_run, tip='A recorded run folder of this workspace'))
        f.row('Recorded run', orow)
        self.offline_none.setToolTip('Offline: the recorded run\'s frames are faulted and re-scored; no simulator')
        v.addLayout(f)
        b = QHBoxLayout()
        self.start_btn = button('Run a closed-loop campaign', self.start, role='primary',
                                tip='falcon faulttol run … --closed-loop (a job): stack runs with each fault at '
                                    'graded severities')
        b.addWidget(self.start_btn)
        self.offline_btn = button('Analyse the recorded run', self.start_offline,
                                  tip='falcon faulttol run … --runs RUN (a job): the recorded run faulted offline')
        self.offline_btn.setEnabled(False)
        b.addWidget(self.offline_btn)
        self.state = label('', small=True, muted=True)
        b.addWidget(self.state, 1)
        v.addLayout(b)
        from .context import register_target
        register_target(self.win, 'faulttol', 'Fault tolerance (offline input)', self.take_run,
                        'Use the run as the recorded input of an offline campaign (nothing starts without a click)')
        THEME.changed.connect(self._themed)          # a bound slot: the flags' colours follow the theme
        self.refresh()

    def _themed(self, _name):
        shown = self.sens_label.isVisible()
        it = self.table.currentItem()
        c = it.data(0, 256) if it is not None else None
        self.fill(keep=(c['sensor'], c['fault']) if c else None)
        self.sens_label.setVisible(shown)

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
        self.where.refresh()
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

    def run_of(self, it) -> str:
        """A row's first run folder (closed loop: its stack runs; offline: the recorded run), or ''."""
        c = it.data(0, 256) if it is not None else None
        name, ws = self.campaign.currentData(), self._ws()
        if not (c and name and ws and self.doc):
            return ''
        if c.get('runs'):
            return os.path.join(FT.campaign_dir(ws, name), c['runs'][0])
        runs = (self.doc.get('inputs') or {}).get('runs') or []
        return os.path.join(ws, runs[0]) if runs else ''

    def _chip(self, doc):
        from ..core import fidelity as F
        lv = (doc or {}).get('fidelity')
        self.chip.setVisible(bool(doc))
        if not doc:
            return
        lv = lv or 'unknown'
        src = doc.get('source_fidelity')
        self.chip.setText(F.label(lv) + (f' · of {F.label(src)} runs' if src else ''))
        self.chip.setToolTip(F.sentence(lv) + (f' The runs it analysed: {F.sentence(src)}' if src else ''))
        tone = F.tone(src) if lv == 'recorded' and src else F.tone(lv)
        self.chip.setProperty('tone', tone)
        self.chip.setProperty('fidelity', lv)
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)

    def fill(self, keep: tuple | None = None):
        doc = self.doc
        self.table.blockSignals(True)
        self.table.clear()
        self.table.blockSignals(False)
        self.curve.set_cell(None)
        self.req.setEnabled(False)
        self._chip(doc)
        if doc is None:
            self.fragile.set('—', sub='no campaign yet: run one below, or falcon faulttol run')
            self.cells_stat.set('—')
            self.req_stat.set('—')
            self.derived.clear()
            self.derived_more.setVisible(False)
            self.fill_flags()
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
        self.derived.clear()
        for s in doc.get('calibration') or []:
            body = s['statement'].split(': ', 1)[-1]
            self.derived.set(s['sensor'], body,
                             tip=f'Derived — {s["statement"]}\n({s.get("note") or FT.STATEMENT_NOTE})')
        self.derived_more.set_text(f'Derived — {FT.STATEMENT_NOTE}. Method: {doc.get("method", "")}. Limits: '
                                   f'{doc.get("limits", "")}.')
        self.derived_more.setVisible(bool(doc.get('calibration')))
        self.fill_flags()
        if not keep and self.table.topLevelItemCount():
            self.table.setCurrentItem(self.table.topLevelItem(0))

    def derived_text(self) -> str:
        """The derived statements as one text (each 'Derived — sensor: …')."""
        return '\n'.join(f'Derived — {s["statement"]}' for s in (self.doc or {}).get('calibration') or [])

    # ------------------------------------------------------------------ flags and their next steps (4.2)
    MAX_FLAGS = 4

    def fill_flags(self):
        clear_layout(self.flags_box)
        if getattr(self, '_shown', None) != self.campaign.currentData():   # its sensitivity line goes with it
            self.sens_label.setVisible(False)
            self._shown = self.campaign.currentData()
        self.flag_list = FT.flags(self.doc, self.requirements()) if self.doc else []
        for fl in self.flag_list[:self.MAX_FLAGS]:
            acts = [('Jump to config', lambda _=False, f=fl: self.jump_to(f),
                     {'requirement': 'Select the row and focus its requirement', 'scenario': 'Focus the scenario '
                      'and criterion of the campaign', 'row': 'Select the row'}[fl['field']]),
                    ('Attach evidence', lambda _=False, f=fl: self.attach(f),
                     'Add this campaign to the safety case as a Solution (undoable there)')]
            if fl['sensitivity']:
                acts.append(('Run sensitivity check', lambda _=False, f=fl: self.run_sensitivity(f),
                             f'The verdict at the requirement ±{100 * FT.SENS_BAND:g} % and at both ends of the '
                             'tolerance\'s bracket; nothing is run'))
            self.flags_box.addWidget(flag(fl['text'], fl['tone'], acts))
        more = self.flag_list[self.MAX_FLAGS:]
        if more:
            self.flags_box.addWidget(Details('\n'.join('● ' + f['text'] for f in more),
                                             f'{len(more)} more flag(s)'))

    def _row_of(self, sensor: str, fault: str):
        for i in range(self.table.topLevelItemCount()):
            it = self.table.topLevelItem(i)
            c = it.data(0, 256)
            if c['sensor'] == sensor and c['fault'] == fault:
                return it
        return None

    def jump_to(self, fl: dict):
        """Select the flag's row and focus the field it concerns (the requirement, or the campaign's scenario)."""
        it = self._row_of(fl['sensor'], fl['fault'])
        if it is not None:
            self.table.setCurrentItem(it)
            self.table.scrollToItem(it)
        if fl['field'] == 'requirement':
            self.req.setFocus()
            self.req.selectAll()
            return self.req
        if fl['field'] == 'scenario':
            self.scenario.setFocus()
            return self.scenario
        self.table.setFocus()
        return self.table

    def run_sensitivity(self, fl: dict) -> dict:
        it = self._row_of(fl['sensor'], fl['fault'])
        c = it.data(0, 256) if it is not None else {}
        s = self.sens = FT.sensitivity(c, self.requirements())
        flipped = [e['what'] for e in s['ends'] if e['verdict'] != s['nominal']]
        head = f'Sensitivity of {fl["sensor"]} · {fl["fault"]}: '
        if not s['applicable']:
            tone, text = '', head + s['why']
        elif s['flips']:
            tone = 'warning'
            text = head + f'{s["nominal"].upper()} here, but it FLIPS at: {", ".join(flipped)}'
        else:
            tone = 'good' if s['nominal'] == 'met' else 'critical'
            text = head + f'{s["nominal"].upper()} at every end of the band (holds)'
        self.sens_label.setText(text)
        self.sens_label.setStyleSheet(f'color: {THEME.status(tone)}' if tone else '')
        self.sens_label.setToolTip(FT.sensitivity_text(s))
        self.sens_label.setVisible(True)
        return s

    def attach(self, fl: dict | None = None) -> str:
        """This campaign as a Solution of the safety case; with a requirement on the flagged cell the claim is its
        tolerance ≥ the requirement, else the campaign's hash alone."""
        name = self.campaign.currentData()
        if not (name and self.doc):
            return ''
        claim, text = None, f'Fault tolerance campaign {name}'
        if fl:
            req = self.requirements().get(fl['fault'])
            if req:
                claim = {'measure': 'field', 'path': f'tolerance.{fl["sensor"]}.{fl["fault"]}.tolerated', 'op': '>=',
                         'limit': float(req)}
                text = f'{fl["sensor"]} tolerates {fl["fault"]} up to the required {float(req):g} ({name})'
            else:
                text += f' ({fl["sensor"]} · {fl["fault"]}: {fl["kind"].replace("_", " ")})'
        return attach_evidence(self.win, 'faulttol', f'faulttol/{name}', claim, text)

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
            set_context_run(self.win, self.run_of(it))
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

    # ------------------------------------------------------------------ a recorded run (Send to ▸ Fault tolerance)
    def take_run(self, run_path: str):
        """Send to ▸ Fault tolerance: the run becomes the offline input, preselected; nothing starts."""
        if hasattr(self.win, 'go'):
            self.win.go('stack')
        self.offline_run.setText(run_path or '')
        self.offline_none.setVisible(not run_path)
        self.offline_btn.setEnabled(bool(run_path))
        if run_path and not self.name.text().strip():
            self.name.setText(f'ft-{os.path.basename(run_path.rstrip("/"))}'[:60])
        self.state.setText('Recorded run chosen: name the campaign, then Analyse the recorded run.' if run_path
                           else '')
        self.offline_btn.setFocus()

    def pick_run(self):
        from PyQt6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, 'A recorded run', os.path.join(self._ws(), 'runs'))
        if d:
            self.take_run(d)

    # ------------------------------------------------------------------ campaigns (jobs)
    def argv(self, offline: bool = False) -> list:
        w = self.win
        a = ['faulttol', 'run', w.project.path, '--name', self.name.text().strip()]
        if offline:
            a += ['--runs', self.offline_run.full]
        else:
            a += ['--closed-loop', '--scenario', str(self.scenario.currentData()), '--criterion',
                  self.criterion.currentData()]
            if self.where.world():
                a += ['--world', self.where.world()]
            if self.where.rig():
                a += ['--rig', self.where.rig()]
        for k, val in (getattr(w, 'settings', None).get('faulttol_requirements') or {}).items():
            a += ['--require', f'{k}={float(val):g}']
        fake = getattr(getattr(w, 'service', None), 'fake_carla', '') or getattr(w, 'fake_carla', '')
        if fake and not offline:
            a += ['--fake-carla', fake]
        return a

    def _submit(self, offline: bool):
        import re
        w = self.win
        if self.job is not None and self.job.state not in FINAL:
            return
        name = self.name.text().strip()
        if not re.match(r'^[A-Za-z0-9][\w.-]{0,63}$', name):
            self.state.setText('Give the campaign a name (letters, digits, - and _).')
            return
        if offline and not self.offline_run.full:
            self.state.setText('Choose a recorded run.')
            return
        if not offline and self.scenario.currentData() is None:
            self.state.setText('Choose a scenario.')
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        what = os.path.basename(self.offline_run.full.rstrip('/')) if offline else self.scenario.currentText()
        self.job = w.queue.submit(JobSpec('faulttol', f'Fault tolerance {name} ({what})',
                                          [sys.executable, '-m', 'falcon_autonomy'] + self.argv(offline),
                                          memory_gb=0.5))
        self.state.setText(self.job.summary())
        if not hasattr(self, '_poll'):
            self._poll = QTimer(self)
            self._poll.timeout.connect(self.poll)
            self._poll.start(2000)

    def start(self):
        self._submit(False)

    def start_offline(self):
        self._submit(True)

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
