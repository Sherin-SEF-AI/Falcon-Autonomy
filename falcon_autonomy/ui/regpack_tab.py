"""Reports ▸ Regulation packs: ready-made scenario suites scored after published test procedures (core.regpacks;
docs/regulation-packs.md).

The notice comes first and stays: a pack is an engineering pre-check in simulation — not a type-approval, not an
official Euro NCAP rating — and Falcon is not a qualified tool. Then the packs with how many of their cases run and
how many of their values were not verified in the document's text; the chosen pack's results in the workspace; and
the case table: each case with the clause it comes from, its result as a word with its colour (passed, failed, a
Euro NCAP colour band, refused with the reason, not run), a mark on every case that uses an unverified value, and
what was measured. Below the table the unverified values are spelled out with why. Run the pack (a job, `falcon
regpack run`, on the project's first world and rig, or a group of its cases) and export the result's report into
the workspace's reports folder, where the list above shows it."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ..core import regpacks as RP
from ..core.jobs import FINAL, JobSpec
from . import motion
from .theme import THEME, mono
from .widgets import Form, Stat, button, callout, label

UNVERIFIED_MARK = '● unverified value'


class RegPackTab(QWidget):
    """The Regulation packs section (module doc). owner: the Reports page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.doc = None
        self.job = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.notice = callout(RP.DISCLAIMER, 'warning')
        v.addWidget(self.notice)
        self.packs = QTreeWidget()
        self.packs.setHeaderLabels(['Pack', 'Cases', 'Runnable', 'Unverified values', 'Sources'])
        self.packs.setRootIsDecorated(False)
        self.packs.setFixedHeight(128)
        self.packs.setColumnWidth(0, 380)
        for pid in RP.PACKS:
            s = RP.summary(pid)
            it = QTreeWidgetItem([s['title'], str(s['cases']), str(s['runnable']), str(s['unverified']),
                                  '; '.join(s['sources'])])
            it.setData(0, 256, pid)
            it.setToolTip(0, s['about'])
            for col in (1, 2, 3):
                it.setFont(col, mono(12))
                it.setTextAlignment(col, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if s['unverified']:
                it.setForeground(3, QColor(THEME.status('warning')))
            self.packs.addTopLevelItem(it)
        self.packs.currentItemChanged.connect(lambda _it, _p: self.pack_changed())
        v.addWidget(self.packs)
        row = QHBoxLayout()
        self.result = QComboBox()
        self.result.setMinimumWidth(220)
        self.result.currentIndexChanged.connect(lambda _: self.load())
        row.addWidget(label('Result', muted=True))
        row.addWidget(self.result, 1)
        row.addWidget(button('Refresh', self.refresh))
        v.addLayout(row)
        stats = QHBoxLayout()
        self.verdict = Stat('Verdict (pre-check)')
        self.count = Stat('Cases')
        self.unver = Stat('Unverified values')
        for s in (self.verdict, self.count, self.unver):
            stats.addWidget(s, 1)
        v.addLayout(stats)
        self.cases = QTreeWidget()
        self.cases.setHeaderLabels(['Case', 'Clause', 'Result', 'Values', 'Measured / why'])
        self.cases.setRootIsDecorated(False)
        self.cases.setMinimumHeight(220)
        self.cases.setColumnWidth(0, 330)
        self.cases.setColumnWidth(1, 230)
        self.cases.setColumnWidth(2, 170)
        self.cases.setColumnWidth(3, 140)
        v.addWidget(self.cases)
        self.unverified = label('', small=True, wrap=True, sel=True)
        v.addWidget(self.unverified)
        f = Form()
        self.group = QComboBox()
        self.name = QLineEdit()
        self.name.setPlaceholderText('a name for the result')
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.01, 1.0)
        self.scale.setDecimals(2)
        self.scale.setSingleStep(0.05)
        self.scale.setValue(1.0)
        self.scale.setToolTip('Shorten the long cases (UN R157\'s 5-minute lane keeping) by this factor. The result '
                              'then says the document\'s duration was not run, and its verdict is incomplete.')
        f.row('Cases', self.group)
        f.row('Name', self.name)
        f.row('Duration factor', self.scale)
        v.addLayout(f)
        b = QHBoxLayout()
        self.run_btn = button('Run the pack', self.start, role='primary',
                              tip='falcon regpack run (a job): each case one stack run with Falcon\'s follower, '
                                  'scored from the run')
        b.addWidget(self.run_btn)
        self.export_btn = button('Export the report', self.export_report,
                                 tip='the result as a self-contained HTML report in the workspace\'s reports folder')
        b.addWidget(self.export_btn)
        self.open_btn = button('Open the report', self.open_report)
        b.addWidget(self.open_btn)
        self.state = label('', small=True, muted=True)
        b.addWidget(self.state, 1)
        v.addLayout(b)
        self.packs.setCurrentItem(self.packs.topLevelItem(0))

    def showEvent(self, ev):                                      # noqa: N802
        super().showEvent(ev)
        self.refresh()

    # ------------------------------------------------------------------ data
    def _ws(self) -> str:
        try:
            return self.win.workspace() if self.win.project.path else ''
        except AttributeError:
            return ''

    def pack_id(self) -> str:
        it = self.packs.currentItem()
        return it.data(0, 256) if it is not None else RP.PACKS[0]

    def pack_changed(self):
        p = RP.pack(self.pack_id())
        self.group.clear()
        self.group.addItem('All cases', '')
        for g in sorted({c['group'] for c in p['cases']}):
            self.group.addItem(f'Only {g}', g)
        self.run_btn.setEnabled(p['kind'] == 'checklist' or any(RP.runnable(c) for c in p['cases']))
        self.run_btn.setToolTip('' if self.run_btn.isEnabled() else
                                'No case of this pack runs in this build: each is refused by name (see the table).')
        self.refresh()

    def refresh(self):
        ws, pid, cur = self._ws(), self.pack_id(), self.result.currentData()
        self.result.blockSignals(True)
        self.result.clear()
        self.result.addItem('The pack itself (not run)', '')
        for n in reversed(RP.results(ws) if ws else []):
            try:
                if RP.load(RP.result_dir(ws, n))['pack'] == pid:
                    self.result.addItem(n, n)
            except (OSError, ValueError, RP.RegPackError, KeyError):
                continue
        k = self.result.findData(cur)
        self.result.setCurrentIndex(k if k >= 0 else (1 if self.result.count() > 1 else 0))
        self.result.blockSignals(False)
        self.load()

    def load(self):
        name, ws = self.result.currentData(), self._ws()
        self.doc = None
        if name and ws:
            try:
                self.doc = RP.load(RP.result_dir(ws, name))
            except (OSError, ValueError, RP.RegPackError):
                self.doc = None
        self.fill()

    def _add(self, title, clause, word, tone, unverified, detail, tip=''):
        it = QTreeWidgetItem([title, clause, word, UNVERIFIED_MARK if unverified else '', detail])
        if tone:
            it.setForeground(2, QColor(THEME.status(tone)))
        if unverified:
            it.setForeground(3, QColor(THEME.status('warning')))
            it.setToolTip(3, 'uses: ' + ', '.join(unverified) + ' — not verified as a number in the document\'s '
                             'text (listed below the table)')
        it.setToolTip(4, tip or detail)
        it.setFont(1, mono(12))
        self.cases.addTopLevelItem(it)
        return it

    def fill(self):
        p = RP.pack(self.pack_id())
        self.cases.clear()
        doc = self.doc
        have = doc is not None
        self.export_btn.setEnabled(have)
        self.open_btn.setEnabled(have)
        if p['kind'] == 'checklist':
            tones = {'covered': 'good', 'partial': 'warning', 'gap': 'critical'}
            for i in (doc or p)['items']:
                self._add(i['item'], f'JAMA Ver 3.0 {i["clause"]}', i['status'], tones[i['status']], [],
                          ', '.join(i['scenarios']) or i['note'], i['note'])
        elif have:
            for r in doc['cases']:
                detail = r.get('why') or '; '.join(c['detail'] for c in r.get('checks') or [] if c.get('detail'))
                self._add(r['title'], r['clause'], r['word'], r.get('tone', ''), r.get('unverified'),
                          detail + (f' [{r["note"]}]' if r.get('note') else ''))
        else:
            for c in p['cases']:
                unv = sorted(k for k, x in (c.get('values') or {}).items() if RP._is_value(x) and
                             x['verified'] is False)
                word = 'refused' if c.get('refuse') else 'not run'
                self._add(c['title'], c['clause'], word, '', unv, c.get('refuse') or c.get('note') or
                          ('mandatory' if c['mandatory'] else 'informative'))
        unv = (doc or {}).get('unverified') if have else RP.unverified(p)
        if have:
            v = doc['verdict']
            tone = {'passed': 'good', 'scored': 'good', 'failed': 'critical', 'incomplete': 'warning',
                    'partial': 'warning'}.get(v, '')
            self.verdict.set(v, sub=doc.get('verdict_note', ''), tone=tone)
            counts = doc.get('counts') or {}
            self.count.set(str(len(doc['cases']) or len(doc.get('items') or [])),
                           sub=', '.join(f'{n} {k}' for k, n in counts.items()))
        else:
            self.verdict.set('not run', sub='run the pack below, or choose a result')
            self.count.set(str(len(p['cases']) or len(p.get('items') or [])),
                           sub=f'{sum(RP.runnable(c) for c in p["cases"])} runnable')
        self.unver.set(str(len(unv)), sub='not verified in the document\'s text' if unv else
                       'every document value was read in its text', tone='warning' if unv else '')
        self.unverified.setText('\n'.join(
            f'● {u["name"]} = {u["value"]} {u["unit"]} ({RP.SOURCES[u["source"]]["title"]}, {u["clause"]}): '
            f'{u["why"]}' for u in unv))

    # ------------------------------------------------------------------ run and export
    def argv(self) -> list:
        w = self.win
        a = ['regpack', 'run', w.project.path, self.pack_id(), '--name', self.name.text().strip()]
        if self.group.currentData():
            a += ['--group', self.group.currentData()]
        if self.scale.value() < 1.0:
            a += ['--duration-scale', f'{self.scale.value():g}']
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
            self.state.setText('Give the result a name (letters, digits, - and _).')
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        self.job = w.queue.submit(JobSpec('regpack', f'Regulation pack {self.pack_id()} ({name}; a pre-check)',
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
            k = self.result.findData(name)
            if k >= 0:
                self.result.setCurrentIndex(k)
                motion.flash_item(self.cases.topLevelItem(0), self.cases.columnCount())

    def export_report(self) -> str:
        """Write the chosen result's report into the workspace's reports folder; returns its path ('' without one)."""
        from ..core import reports as R
        name, ws = self.result.currentData(), self._ws()
        if not (name and ws and self.doc is not None):
            return ''
        src = RP.result_dir(ws, name)
        out = os.path.join(R.reports_dir(ws), R.default_name('regpack', src))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        R.write_report('regpack', src, out)
        self.state.setText(f'Report written: {os.path.basename(out)}')
        if hasattr(self.page, 'refresh'):
            self.page.refresh()
        return out

    def open_report(self):
        name, ws = self.result.currentData(), self._ws()
        if name and ws:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.join(RP.result_dir(ws, name), 'report.html')))
