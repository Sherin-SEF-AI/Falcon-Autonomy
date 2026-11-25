"""Reports ▸ Regulation packs: ready-made scenario suites scored after published test procedures (core.regpacks;
docs/regulation-packs.md).

The notice comes first and stays: a pack is an engineering pre-check in simulation — not a type-approval, not an
official Euro NCAP rating — and Falcon is not a qualified tool. Then the packs with how many of their cases run and
how many of their values were not verified in the document's text; the chosen pack's results in the workspace; and
the case table: each case with the clause it comes from, its result as a word with its colour (passed, failed, a
Euro NCAP colour band, refused with the reason, not run), a mark on every case that uses an unverified value, and
what was measured. The verdict wears the result's fidelity chip (core.fidelity).

4.2: each value not verified in its document is a flag with its next steps — Jump to config (select the value in the
values table), Attach evidence (the result as a Solution of the safety case, undoable there) and Run sensitivity
check (core.regpacks.sensitivity: the value at the ends of ±10 % — the pack gives no range — and whether a case or the
verdict flips; nothing is run). The values table is compact: the value, its document and clause, how it was read;
why it is not verified is in the tooltip and Falcon's own choices are behind Details. Run the pack (a job, `falcon
regpack run`) on the world and rig chosen here; a run case's row has Send to ▸ (right click) and selecting it makes
its run the active one."""
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
from .robustkit import RigWorldPicker, attach_evidence, flag, run_menu, set_context_run
from .theme import THEME, mono
from .widgets import Details, Form, Stat, button, callout, clear_layout, fidelity_chip, label

UNVERIFIED_MARK = '● unverified value'
BASIS_WORD = {'figure': 'read from a figure', 'recalled': 'recalled'}


class RegPackTab(QWidget):
    """The Regulation packs section (module doc). owner: the Reports page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.doc = None
        self.job = None
        self.sens = {}                                   # value name -> the last sensitivity() of it
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
            it.setToolTip(4, '\n'.join(s['sources']))
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
        self.chip = fidelity_chip('unknown')
        stats.addWidget(self.chip, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(stats)
        self.flags_box = QVBoxLayout()                   # the unverified values as flags with their next steps
        self.flags_box.setSpacing(6)
        v.addLayout(self.flags_box)
        self.sens_label = label('', small=True, wrap=True, sel=True)
        self.sens_label.setVisible(False)
        v.addWidget(self.sens_label)
        self.cases = QTreeWidget()
        self.cases.setHeaderLabels(['Case', 'Clause', 'Result', 'Values', 'Measured / why'])
        self.cases.setRootIsDecorated(False)
        self.cases.setMinimumHeight(220)
        self.cases.setColumnWidth(0, 330)
        self.cases.setColumnWidth(1, 230)
        self.cases.setColumnWidth(2, 170)
        self.cases.setColumnWidth(3, 140)
        self.cases.currentItemChanged.connect(lambda it, _p: set_context_run(self.win, self.run_of(it)))
        run_menu(self.win, self.cases, self.run_of)
        v.addWidget(self.cases)
        v.addWidget(label('Values not verified in the document', h=True))
        self.values = QTreeWidget()                      # compact: value, document · clause, how it was read
        self.values.setHeaderLabels(['Value', 'Is', 'Document · clause', 'Read', 'Used by'])
        self.values.setRootIsDecorated(False)
        self.values.setMaximumHeight(120)
        self.values.setColumnWidth(0, 150)
        self.values.setColumnWidth(1, 170)
        self.values.setColumnWidth(2, 360)
        self.values.setColumnWidth(3, 130)
        v.addWidget(self.values)
        self.choices = Details('', 'Falcon\'s own choices (not any document\'s values)')
        v.addWidget(self.choices)
        f = Form()
        self.where = RigWorldPicker(self.win, labels=False, what='a case with a sensor failure — the minimum risk '
                                                                  'manoeuvre — renders the rig\'s cameras')
        f.row('Where', self.where)
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
        THEME.changed.connect(self._themed)          # a bound slot: the flags' colours follow the theme
        self.packs.setCurrentItem(self.packs.topLevelItem(0))

    def _themed(self, _name):
        shown = self.sens_label.isVisible()
        self.fill()
        self.sens_label.setVisible(shown)

    def showEvent(self, ev):                                      # noqa: N802
        super().showEvent(ev)
        self.where.refresh()
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
        self.sens = {}
        self.refresh(newest=True)

    def refresh(self, newest: bool = False):
        """The chosen pack's results (the shown one kept; newest: the newest result, as after a pack change)."""
        ws, pid, cur = self._ws(), self.pack_id(), None if newest else self.result.currentData()
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
        self.sens = {}
        self.fill()

    def run_of(self, it) -> str:
        """The run folder of a case row ('' when it was not run)."""
        r = it.data(0, 256) if it is not None else None
        name, ws = self.result.currentData(), self._ws()
        if not (r and r.get('run') and name and ws):
            return ''
        return os.path.join(RP.result_dir(ws, name), r['run'])

    def _add(self, title, clause, word, tone, unverified, detail, tip='', row=None):
        it = QTreeWidgetItem([title, clause, word, UNVERIFIED_MARK if unverified else '', detail])
        if tone:
            it.setForeground(2, QColor(THEME.status(tone)))
        if unverified:
            it.setForeground(3, QColor(THEME.status('warning')))
            it.setToolTip(3, 'uses: ' + ', '.join(unverified) + ' — not verified as a number in the document\'s '
                             'text (the values table below says how it was read)')
        it.setToolTip(4, tip or detail)
        it.setFont(1, mono(12))
        it.setData(0, 256, row)
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
                          detail + (f' [{r["note"]}]' if r.get('note') else ''), row=r)
        else:
            for c in p['cases']:
                unv = sorted(k for k, x in (c.get('values') or {}).items() if RP._is_value(x) and
                             x['verified'] is False)
                word = 'refused' if c.get('refuse') else 'not run'
                self._add(c['title'], c['clause'], word, '', unv, c.get('refuse') or c.get('note') or
                          ('mandatory' if c['mandatory'] else 'informative'))
        unv = (doc or {}).get('unverified') if have else RP.unverified(p)
        lv = (doc or {}).get('fidelity') or 'unknown'
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
        self._chip(lv if have and p['kind'] != 'checklist' else '')
        self.unver.set(str(len(unv)), sub='not verified in the document\'s text' if unv else
                       'every document value was read in its text', tone='warning' if unv else '')
        self.fill_values(p, unv)
        self.fill_flags(p, unv)

    def _chip(self, level: str):
        from ..core import fidelity as F
        self.chip.setVisible(bool(level))
        if level:
            self.chip.setText(F.label(level))
            self.chip.setToolTip(F.sentence(level))
            self.chip.setProperty('tone', F.tone(level))
            self.chip.setProperty('fidelity', level)
            self.chip.style().unpolish(self.chip)
            self.chip.style().polish(self.chip)

    def fill_values(self, p: dict, unv: list):
        self.values.clear()
        for u in unv:
            src = RP.SOURCES.get(u['source'], {})
            used = RP.cases_using(p, u['name'])
            it = QTreeWidgetItem([u['name'], f'{u["value"]} {u["unit"]}'.strip(),
                                  f'{src.get("title", u["source"])} · {u["clause"]}',
                                  BASIS_WORD.get(u['basis'], u['basis']), f'{len(used)} case(s)' if used else '—'])
            it.setData(0, 256, u['name'])
            it.setFont(0, mono(11))
            it.setFont(1, mono(11))
            it.setForeground(3, QColor(THEME.status('warning')))
            it.setToolTip(1, u['value'])
            it.setToolTip(2, f'{src.get("title", "")} {src.get("version", "")} ({src.get("date", "")}), '
                             f'{u["clause"]}; read from: {src.get("read_from", "") or "—"}')
            it.setToolTip(3, f'NOT VERIFIED: {u["why"]}' + (f'. {u["note"]}' if u.get('note') else ''))
            it.setToolTip(4, ', '.join(used) or 'no case reads it')
            self.values.addTopLevelItem(it)
        self.values.setVisible(bool(unv))
        ch = RP.choices(p)
        self.choices.set_text('\n'.join(f'● {c["note"]}' for c in ch))
        self.choices.setVisible(bool(ch))

    def fill_flags(self, p: dict, unv: list):
        clear_layout(self.flags_box)
        key = (self.pack_id(), self.result.currentData())
        if getattr(self, '_shown', None) != key:          # a sensitivity line belongs to what it was run on
            self.sens_label.setVisible(False)
            self._shown = key
        self.flag_widgets = {}
        if p['kind'] == 'checklist' and not unv:
            return
        for u in unv:
            src = RP.SOURCES.get(u['source'], {})
            used = RP.cases_using(p, u['name'])
            doc_ref = f'{src.get("title", u["source"]).split(" — ")[0]}, {u["clause"]}'
            val = f'{u["value"]} {u["unit"]}'.strip() if len(u['value']) <= 24 else \
                ('a table (see Values)' if u['value'][:1] in '{[' else 'a statement (see Values)')
            text = (f'{u["name"]} = {val} — not verified ({BASIS_WORD.get(u["basis"], u["basis"])}) · {doc_ref}'
                    + (f' · used by {len(used)} case(s)' if used else ''))
            acts = [('Jump to config', lambda _=False, n=u['name']: self.jump_to(n),
                     'Select the value in the values table'),
                    ('Run sensitivity check', lambda _=False, n=u['name']: self.run_sensitivity(n),
                     f'Re-evaluate what it decides at ±{100 * RP.SENS_BAND:g} % (the pack gives no range); '
                     'nothing is run')]
            if self.doc is not None:
                acts.insert(1, ('Attach evidence', lambda _=False, n=u['name']: self.attach(n),
                                'Add this result to the safety case as a Solution (undoable there)'))
            f = flag(text, 'warning', acts)
            f.label.setToolTip(f'NOT VERIFIED: {u["why"]}' + (f'. {u["note"]}' if u.get('note') else ''))
            self.flags_box.addWidget(f)
            self.flag_widgets[u['name']] = f

    # ------------------------------------------------------------------ the flags' next steps
    def jump_to(self, name: str):
        """Select and focus the value's row in the values table."""
        for i in range(self.values.topLevelItemCount()):
            it = self.values.topLevelItem(i)
            if it.data(0, 256) == name:
                self.values.setCurrentItem(it)
                self.values.scrollToItem(it)
                self.values.setFocus()
                motion.flash_item(it, self.values.columnCount())
                return it
        return None

    def run_sensitivity(self, name: str) -> dict:
        """core.regpacks.sensitivity of the value on the shown result (or the pack): flips or holds, in a word."""
        s = self.sens[name] = RP.sensitivity(self.pack_id(), name, self.doc)
        if not s['applicable']:
            word, tone = 'not applicable', ''
        else:
            word, tone = ('flips', 'warning') if s['flips'] else ('holds', 'good')
        lo, hi = s['low'], s['high']
        band = f'{lo} … {hi} {s["unit"]}' if isinstance(lo, (int, float)) else s['band']
        flipped = [c['id'] for c in s['cases'] if c['flips']]
        ver = s.get('verdict') or {}
        if ver and set(ver.values()) == {'—'}:
            what = 'no score to re-check (not every grid point was run)'
        elif ver:
            what = f'verdict {ver.get("nominal")}; at the low end {ver.get("low")}, at the high end {ver.get("high")}'
        else:
            what = 'the pack\'s classification only (no result chosen)' if not s['run'] else ''
        self.sens_label.setText(f'Sensitivity of {name}: {word.upper()} at {band}' +
                                (f' · {what}' if what else '') +
                                (f' · flips: {", ".join(flipped[:4])}' + ('…' if len(flipped) > 4 else '')
                                 if flipped else ''))
        self.sens_label.setStyleSheet(f'color: {THEME.status(tone)}' if tone else '')
        self.sens_label.setToolTip(RP.sensitivity_text(s))
        self.sens_label.setVisible(True)
        return s

    def attach(self, name: str = '') -> str:
        """The shown result as a Solution of the safety case: claim verdict == passed (a test pack) or the score's
        share ≥ 1 (a scored pack); the text names the unverified value the flag is about."""
        rname, ws = self.result.currentData(), self._ws()
        if not (rname and ws and self.doc is not None):
            return ''
        if self.doc.get('kind') == 'scored':
            claim = {'measure': 'field', 'path': 'score.share', 'op': '>=', 'limit': 1.0}
        else:
            claim = {'measure': 'verdict', 'equals': 'passed'}
        text = f'{self.doc.get("title", self.pack_id())}: {rname} (a simulation pre-check, not an approval)'
        if name:
            text += f'; uses {name}, not verified in the document'
        return attach_evidence(self.win, 'regpack', f'regpacks/{rname}', claim, text)

    # ------------------------------------------------------------------ run and export
    def argv(self) -> list:
        w = self.win
        a = ['regpack', 'run', w.project.path, self.pack_id(), '--name', self.name.text().strip()]
        if self.where.world():
            a += ['--world', self.where.world()]
        if self.where.rig():
            a += ['--rig', self.where.rig()]
        if self.group.currentData():
            a += ['--group', self.group.currentData()]
        if self.scale.value() < 1.0:
            a += ['--duration-scale', f'{self.scale.value():g}']
        fake = getattr(getattr(w, 'service', None), 'fake_carla', '') or getattr(w, 'fake_carla', '')
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
