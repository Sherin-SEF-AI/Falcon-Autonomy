"""The Loop page's Overnight campaign tab: a goal and a budget in, a one-page brief out (core.campaign,
docs/campaign.md).

Top: the goal, the budget fields (GPU-hours, the deadline, jobs, disk) and the scope checklist (the study kinds the
campaign may run). Preview shows the whole scope, the planned studies and their estimated costs — the text of the one
approval. Approve and start (the page's one amber action) asks once with that text; on yes it writes the campaign,
stores the approval and queues the runner as a job (`falcon campaign run`, in the Jobs dock). Nothing outside the
approved scope is done; such things appear under Needs you.

Below: the selected campaign — its state as a word in its status colour, the budgets as meters with their numbers
(used of given), a sweep line while it runs, the decision log (every decision with its reason; new rows ease in) and,
once it is finished, the brief. Stop asks the runner to stop: it cancels its job and writes the brief.

4.2: the preview is compact — the scope and budget as key-value badges and the planned studies as a table, the whole
approval text (exactly what the one approval shows) behind *Details*; what is never done unattended is one line with
its list behind *Details*. The campaign's fidelity (core.fidelity over its studies' results) is a chip beside its
state."""
from __future__ import annotations

import os

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLineEdit, QMessageBox,
                             QPlainTextEdit, QSpinBox, QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from ..core import campaign as K
from . import motion
from .study_bits import Badges, FidelitySlot
from .theme import THEME, mono
from .widgets import Details, Meter, Section, button, flow_row, label, scroll

STATE_TONE = {'draft': '', 'approved': 'warning', 'running': 'warning', 'stopped': 'serious', 'done': 'good'}
STATE_WORD = {'draft': 'draft — not approved', 'approved': 'approved — waiting for its runner', 'running': 'running',
              'stopped': 'stopped', 'done': 'done'}
LOG_COLUMNS = ('Time (UTC)', 'Decision', 'Reason')


class _Budget(QWidget):
    """A budget line: its name, a meter, and the numbers in mono ("0.50 of 2 GPU-h")."""

    def __init__(self, name: str):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.name = label(name, muted=True)
        self.name.setFixedWidth(90)
        self.meter = Meter(160)
        self.text = label('—', mono=True)
        h.addWidget(self.name)
        h.addWidget(self.meter)
        h.addWidget(self.text, 1)

    def set(self, used, given, text: str):
        self.meter.set(None if not given else float(used) / float(given))
        self.text.setText(text)


class CampaignTab(QWidget):
    """Built by the Loop page (ui.widgets.optional_tab) as CampaignTab(page)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.cid = ''
        self.rows_shown = 0
        self.confirm = self._confirm                # tests replace it: confirm(text) -> bool
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        body.setObjectName('base')
        v = QVBoxLayout(body)
        v.setContentsMargins(24, 18, 24, 18)
        v.setSpacing(10)
        outer.addWidget(scroll(body))

        sec = Section('Goal and budget', 'State what should be found out and what it may cost. Falcon plans the '
                      'studies by rule from this workspace, shows the plan and its estimated costs, and — after your '
                      'one approval — runs them unattended inside that scope. A hard accountant refuses any job past a '
                      'budget and cancels at the deadline.')
        self.goal = QLineEdit()
        self.goal.setPlaceholderText('Find why night pedestrian recall dropped')
        sec.add(self.goal)
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        self.gpu = QDoubleSpinBox()
        self.gpu.setRange(0.0, 200.0)
        self.gpu.setDecimals(1)
        self.gpu.setValue(K.DEFAULT_BUDGET['gpu_hours'])
        self.gpu.setSuffix(' GPU-h')
        self.until = QLineEdit()
        self.until.setPlaceholderText('07:00, +6h or 2027-01-15T07:00:00Z (empty: no deadline)')
        self.jobs = QSpinBox()
        self.jobs.setRange(1, 100)
        self.jobs.setValue(K.DEFAULT_BUDGET['max_jobs'])
        self.disk = QDoubleSpinBox()
        self.disk.setRange(0.1, 2000.0)
        self.disk.setDecimals(1)
        self.disk.setValue(K.DEFAULT_BUDGET['disk_gb'])
        self.disk.setSuffix(' GB')
        for k, (name, w) in enumerate((('GPU-hours', self.gpu), ('Until', self.until), ('Jobs at most', self.jobs),
                                       ('Disk', self.disk))):
            g.addWidget(label(name, muted=True), 0, k)
            g.addWidget(w, 1, k)
        g.setColumnStretch(1, 1)
        sec.add(g)
        sec.add(label('Scope: the study kinds it may run', h=True))
        self.kinds = {}
        for kind, spec in K.KINDS.items():
            cb = QCheckBox(kind + (' (GPU)' if spec['gpu'] else ''))
            cb.setChecked(kind in K.DEFAULT_KINDS)
            cb.setToolTip(spec['about'])
            self.kinds[kind] = cb
        sec.add(flow_row(*self.kinds.values()))
        sec.add(label('Some things are never done unattended, whatever is ticked.', muted=True, small=True))
        sec.add(Details('Never done unattended, whatever is ticked: ' + '; '.join(K.OUTSIDE.values()) + '; anything '
                        'on the assistant\'s never-list. They go to the brief as "needs you".'))
        row = QHBoxLayout()
        self.preview_btn = button('Preview', self.preview, tip='the plan and its estimated costs: nothing is written')
        self.start_btn = button('Approve and start', self.start, role='primary',
                                tip='asks once with the whole scope and budget, then queues falcon campaign run')
        row.addWidget(self.preview_btn)
        row.addWidget(self.start_btn)
        row.addStretch(1)
        sec.add(row)
        self.scope_badges = Badges()                    # the preview, compact: scope and budget as badges
        self.scope_badges.hide()
        sec.add(self.scope_badges)
        self.plan_tree = QTreeWidget()
        self.plan_tree.setHeaderLabels(['Study', 'Kind', 'What', 'About', 'GPU'])
        self.plan_tree.setRootIsDecorated(False)
        self.plan_tree.setMaximumHeight(150)
        self.plan_tree.hide()
        sec.add(self.plan_tree)
        self.card = QPlainTextEdit()                    # the approval's whole text, behind Details
        self.card.setReadOnly(True)
        self.card.setProperty('mono', True)
        self.card.setFont(mono(12))
        self.card.setMinimumHeight(150)
        self.card_details = Details(self.card, 'The approval\'s whole text')
        self.card_details.hide()
        sec.add(self.card_details)
        self.note = label('', muted=True, wrap=True)
        sec.add(self.note)
        v.addWidget(sec)

        sec = Section('Campaign')
        row = QHBoxLayout()
        self.pick = QComboBox()
        self.pick.currentIndexChanged.connect(self._picked)
        self.state = label('—')
        self.stop_btn = button('Stop', self.stop, tip='falcon campaign stop: its running job is cancelled and the '
                                                      'brief written')
        row.addWidget(self.pick, 1)
        row.addWidget(self.state)
        self.fid = FidelitySlot()
        self.fid.setToolTip('The fidelity of what the campaign\'s studies read (core.fidelity)')
        row.addWidget(self.fid)
        row.addWidget(self.stop_btn)
        sec.add(row)
        self.sweep = motion.Sweep()
        sec.add(self.sweep)
        self.b_gpu, self.b_jobs, self.b_disk, self.b_time = (_Budget(n) for n in ('GPU-hours', 'Jobs', 'Disk', 'Time'))
        for b in (self.b_gpu, self.b_jobs, self.b_disk, self.b_time):
            sec.add(b)
        self.needs = label('', wrap=True)
        self.needs.hide()
        sec.add(self.needs)
        sec.add(label('Decision log', h=True))
        self.log = QTreeWidget()
        self.log.setHeaderLabels(list(LOG_COLUMNS))
        self.log.setRootIsDecorated(False)
        self.log.setMinimumHeight(170)
        sec.add(self.log)
        sec.add(label('Brief', h=True))
        self.brief = QTextBrowser()
        self.brief.setMinimumHeight(260)
        self.brief.setOpenExternalLinks(False)
        sec.add(self.brief)
        v.addWidget(sec)
        v.addStretch(1)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.win.bridge.finished.connect(lambda _job: self.refresh())
        self.reload()

    # ------------------------------------------------------------------ the form
    def workspace(self) -> str:
        try:
            return self.win.workspace() or ''
        except AttributeError:
            return ''

    def budget(self) -> dict:
        u = self.until.text().strip()
        return {'gpu_hours': self.gpu.value(), 'max_jobs': self.jobs.value(), 'disk_gb': self.disk.value(),
                'deadline': K.parse_until(u) if u else None}

    def chosen_kinds(self) -> list:
        return [k for k, cb in self.kinds.items() if cb.isChecked()]

    def _draft(self, write: bool) -> dict:
        ws = self.workspace()
        os.makedirs(ws, exist_ok=True)
        proj = self.win.project
        return K.new(ws, self.goal.text(), self.budget(), self.chosen_kinds(), '', getattr(proj, 'path', '') or '',
                     proj, write=write)

    def preview(self) -> str | None:
        """The approval's text for the form as it stands (nothing written)."""
        try:
            rec = self._draft(write=False)
            text = K.approval_text(rec)
        except K.CampaignError as e:
            self.note.setText(str(e))
            return None
        self.card.setPlainText(text)
        self.card_details.show()
        self.show_scope(rec)
        self.note.setText('Nothing is written or run until you approve.')
        return text

    def show_scope(self, rec: dict):
        """The preview as badges and a table (the whole text is the card behind Details)."""
        b = rec['budget']
        studies = rec['studies']
        total = sum(float(s['est']['seconds'] or 0.0) for s in studies)
        gpu = sum(float(s['est']['gpu_seconds'] or 0.0) for s in studies) / 3600.0
        self.scope_badges.set([
            ('GPU-h', f'{float(b["gpu_hours"]):g}', 'the campaign\'s GPU-hours: the wall time of GPU-holding studies'),
            ('Until', b['deadline'] or 'no deadline', 'cancelled at the deadline'),
            ('Jobs', f'≤ {b["max_jobs"]}', 'at most this many jobs'),
            ('Disk', f'{float(b["disk_gb"]):g} GB', 'the disk budget'),
            ('Kinds', ', '.join(rec['scope']['kinds']), 'the study kinds it may run; anything else is "needs you"'),
            ('Planner', rec['planner'], 'rule-based, no model' if rec['planner'] != 'assistant' else
             'the model orders the candidates; it cannot add a study'),
            ('Planned', f'{len(studies)} · about {K._fmt_s(total)}' + (f' · {gpu:.2f} GPU-h' if gpu else ''),
             'each study is refused at its turn if it no longer fits the budget'),
        ] + ([('Needs you', str(len(rec['needs_you'])), '; '.join(n['what'] for n in rec['needs_you'][:6]),
               'warning')] if rec['needs_you'] else []))
        self.scope_badges.show()
        self.plan_tree.clear()
        for s in studies:
            h = next((h for h in rec['hypotheses'] if h['id'] == s['hypothesis']), {})
            it = QTreeWidgetItem([s['id'], s['kind'], s['title'], K._fmt_s(s['est']['seconds']),
                                  f'{s["est"]["gpu_seconds"] / 3600.0:.2f} h' if s['est']['gpu_seconds'] else 'no'])
            it.setToolTip(2, f'tests: {h.get("text", "")}')
            it.setToolTip(3, s['est']['source'])
            self.plan_tree.addTopLevelItem(it)
        for k in (0, 1, 3, 4):
            self.plan_tree.resizeColumnToContents(k)
        self.plan_tree.setVisible(bool(studies))

    def _confirm(self, text: str) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle('Approve this campaign?')
        box.setText('This is the campaign\'s one approval. Inside this scope and budget it runs unattended.')
        box.setDetailedText(text)
        box.setInformativeText(text[:1500])
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def start(self):
        """Preview, ask once, then: the campaign written, the approval stored, the runner queued. Returns the job."""
        text = self.preview()
        if text is None:
            return None
        if not self.confirm(text):
            self.note.setText('Not approved: nothing was written or run.')
            return None
        ws = self.workspace()
        try:
            rec = self._draft(write=True)
            K.approve(ws, rec['id'], {'how': 'approved', 'by': 'user'})
        except K.CampaignError as e:
            self.note.setText(str(e))
            return None
        fake = getattr(self.win.service, 'fake_carla', '') or ''
        job = self.win.queue.submit(K.runner_job_spec(ws, rec['id'], fake))
        self.note.setText(f'Campaign {rec["id"]} approved; its runner is job {job.id} (the Jobs dock). To leave it '
                          f'running without Falcon\'s window: {K.detached_command(ws, rec["id"])}')
        self.reload(rec['id'])
        return job

    # ------------------------------------------------------------------ the selected campaign
    def reload(self, select: str = ''):
        ws = self.workspace()
        recs = K.campaigns(ws) if ws and os.path.isdir(ws) else []
        recs.sort(key=lambda r: r.get('created') or '', reverse=True)
        self.pick.blockSignals(True)
        self.pick.clear()
        for r in recs:
            self.pick.addItem(f'{r["id"]} — {r["goal"][:70]}', r['id'])
        want = select or self.cid
        i = max(self.pick.findData(want), 0) if recs else -1
        self.pick.setCurrentIndex(i)
        self.pick.blockSignals(False)
        self._picked()

    def _picked(self, _i=None):
        cid = self.pick.currentData() or ''
        if cid != self.cid:
            self.cid, self.rows_shown = cid, 0
            self.log.clear()
        self.refresh()

    def showEvent(self, ev):                                    # noqa: N802
        super().showEvent(ev)
        self.reload()

    def refresh(self) -> dict | None:
        """The selected campaign as its files say now: state, budgets, new log rows, the brief when finished."""
        ws = self.workspace()
        if not self.cid:
            self.state.setText('no campaign yet')
            self.stop_btn.setEnabled(False)
            self.sweep.set(None)
            self.timer.stop()
            return None
        try:
            st = K.status(ws, self.cid)
        except K.CampaignError as e:
            self.state.setText(str(e))
            return None
        s = st['state']
        self.state.setText(STATE_WORD[s])
        lv = st.get('fidelity') or 'unknown'
        self.fid.set(lv, 'mixed fidelity' if lv == 'mixed' else '', 'critical' if lv == 'mixed' else '')
        tone = STATE_TONE[s]
        self.state.setStyleSheet(f'color: {THEME.status(tone)}' if tone else '')
        live = s in ('approved', 'running')
        self.stop_btn.setEnabled(s != 'done' and s != 'stopped')
        b, u, left = st['budget'], st['used'], st['left']
        self.b_gpu.set(u['gpu_hours'], b['gpu_hours'], f'{u["gpu_hours"]:.2f} of {float(b["gpu_hours"]):g} GPU-h')
        self.b_jobs.set(u['jobs'], b['max_jobs'], f'{u["jobs"]} of {b["max_jobs"]}')
        self.b_disk.set(u['disk_gb'], b['disk_gb'], f'{u["disk_gb"]:.2f} of {float(b["disk_gb"]):g} GB')
        if b.get('deadline'):
            rec = K.load(ws, self.cid)
            t0 = K.epoch(rec.get('started') or rec['created'])
            span = max(K.epoch(b['deadline']) - t0, 1.0)
            used = span - max(left['seconds'] or 0.0, 0.0)
            self.b_time.set(used, span, f'until {b["deadline"]}' + (
                f' · {max(left["seconds"], 0.0) / 60.0:.0f} min left' if live else ''))
        else:
            self.b_time.set(0, 0, 'no deadline')
        done = sum(1 for x in st['studies'] if x['state'] in ('done', 'failed', 'cancelled', 'refused', 'not run'))
        self.sweep.set(done / max(len(st['studies']), 1) if live else None, live=s == 'running')
        if st['needs_you']:
            self.needs.setText('Needs you: ' + '; '.join(f'{n["what"]} — {n["why"]}' for n in st['needs_you'][:6]))
            self.needs.setStyleSheet(f'color: {THEME.status("warning")}')
        self.needs.setVisible(bool(st['needs_you']))
        rows = K.read_log(ws, self.cid)
        first = self.rows_shown == 0
        for r in rows[self.rows_shown:]:
            it = QTreeWidgetItem([str(r.get('time', ''))[11:19], str(r.get('decision', '')),
                                  (f'{r["study"]}: ' if r.get('study') else '') + str(r.get('reason', ''))])
            it.setFont(0, mono(12))
            it.setToolTip(2, str(r.get('reason', '')))
            self.log.addTopLevelItem(it)
            if not first:
                motion.flash_item(it, len(LOG_COLUMNS))
        if len(rows) > self.rows_shown:
            self.log.scrollToBottom()
            for k in (0, 1):
                self.log.resizeColumnToContents(k)
        self.rows_shown = len(rows)
        if s in ('done', 'stopped') and st.get('brief'):
            with open(st['brief'], encoding='utf-8') as fh:
                md = fh.read()
            if self.brief.property('shown') != (self.cid, len(md)):
                self.brief.setMarkdown(md)
                self.brief.setProperty('shown', (self.cid, len(md)))
        elif self.brief.property('shown') is not None or not self.brief.toPlainText():
            self.brief.setPlainText('The brief is written when the campaign is finished.')
            self.brief.setProperty('shown', None)
        if live and self.isVisible():
            self.timer.start()
        else:
            self.timer.stop()
        return st

    def stop(self):
        if not self.cid:
            return None
        try:
            K.stop(self.workspace(), self.cid, why='stopped from the Loop page')
        except K.CampaignError as e:
            self.note.setText(str(e))
            return None
        return self.refresh()
