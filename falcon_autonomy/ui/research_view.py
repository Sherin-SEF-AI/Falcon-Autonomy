"""The Assistant panel's Research tab: the research autopilot's studies (core.autoresearch;
docs/research-autopilot.md) — each study's state machine (proposed → approved → running → analysed → follow-up or
reported, and stopped, failed or interrupted), its rounds, history and next step — with Stop, Undo and Resume, the
report, and the questions the app suggests from its own state.

A study waiting for its first approval card is shown as proposed (from the panel's open cards: nothing is written
before the user approves). Stop cancels the study's job, records the stop (which revokes follow-ups approved ahead)
and stops the assistant's turn; Undo takes back the newest change the assistant made for the study, through the
activity log (a queued job is cancelled); Resume asks the assistant's tool, so its card appears as the permission rules
say. The view writes nothing else: the transitions the files show are written into the study's history on refresh
(core.autoresearch.sync), as the tools do.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QHBoxLayout, QHeaderView, QTreeWidgetItem, QVBoxLayout, QWidget

from .widgets import PlaceholderTree, button, label, set_enabled

CHAIN = ('proposed', 'approved', 'running', 'analysed', 'follow_up', 'reported')
SIDE = {'interrupted': 'running', 'failed': 'running', 'stopped': None}
TEXT = {'follow_up': 'follow-up'}


def chain_text(state: str) -> str:
    """The state machine as one line, the current state in brackets (a side state named after the step it left)."""
    at = SIDE.get(state, state)
    parts = [f'[{TEXT.get(s, s)}]' if s == at and state in CHAIN else TEXT.get(s, s) for s in CHAIN]
    line = ' → '.join(parts)
    if state not in CHAIN:
        line += f'   ({state}' + (f' at {at}' if at else '') + ')'
    return line


class ResearchView(QWidget):
    poke = pyqtSignal()                                 # a study's job changed (from the queue's thread)

    def __init__(self, panel):
        super().__init__()
        self.panel, self.win = panel, panel.win
        self.rows: list = []
        self.suggested: list = []
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label('Research studies', h=True), 1)
        head.addWidget(button('Refresh', self.refresh, tip='Read the studies again from the workspace'))
        v.addLayout(head)
        self.tree = PlaceholderTree('No research study yet. Ask the assistant to study a question — "does the '
                                    'fine-tuned detector raise night recall?" — or pick one suggested below.')
        self.tree.setHeaderLabels(('Study', 'State', 'Rounds', 'Verdict'))
        self.tree.setRootIsDecorated(False)
        hd = self.tree.header()
        hd.setStretchLastSection(False)
        hd.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.itemSelectionChanged.connect(self._show)
        v.addWidget(self.tree, 2)
        self.chain = label('', small=True, mono=True, wrap=True)
        self.chain.setAccessibleName('The study\'s state')
        v.addWidget(self.chain)
        self.detail = label('', small=True, wrap=True, sel=True)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignTop)
        v.addWidget(self.detail, 1)
        row = QHBoxLayout()
        self.stop_btn = button('Stop', self.stop, tip='Cancel the study\'s job and stop the assistant')
        self.undo_btn = button('Undo', self.undo, tip='Take back the newest change the assistant made for this study')
        self.resume_btn = button('Resume', self.resume, tip='Continue the study (its card asks you first)')
        self.report_btn = button('Open report', self.open_report)
        for b in (self.stop_btn, self.undo_btn, self.resume_btn, self.report_btn):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        sh = QHBoxLayout()
        sh.addWidget(label('Suggested questions', h=True), 1)
        self.suggest_btn = button('Find', self.find_suggestions,
                                  tip='Questions from the app\'s state: failure modes, operating-domain cells, failure '
                                      'rates, bisect regressions (nothing from the dashcam footage)')
        sh.addWidget(self.suggest_btn)
        v.addLayout(sh)
        self.sug = PlaceholderTree('Press Find to look for questions worth a study.')
        self.sug.setHeaderLabels(('Question', ''))
        self.sug.setRootIsDecorated(False)
        hs = self.sug.header()
        hs.setStretchLastSection(False)
        hs.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.sug, 1)
        self._later = QTimer(self)
        self._later.setSingleShot(True)
        self._later.setInterval(300)
        self._later.timeout.connect(self.refresh)
        self.poke.connect(self.changed)
        self.win.queue.subscribe(self._on_job)
        self._buttons(None)

    # ------------------------------------------------------------------ data
    def _ws(self) -> str:
        return self.win.workspace()

    def _ctx(self):
        from ..assistant.privacy import redact
        ctx = self.panel.host.tool_context()
        ctx.redact = redact
        return ctx

    def _on_job(self, ev, job):
        if ev in ('finished', 'added') and (job.spec.meta or {}).get('study'):
            try:
                self.poke.emit()
            except RuntimeError:                        # the panel is gone (the window closing)
                pass

    def changed(self):
        """Something a study depends on changed: refresh soon, when the tab is shown."""
        if self.isVisible():
            self._later.start()

    def pending(self) -> list:
        """Studies whose first approval card is open: proposed, nothing written yet."""
        out = []
        for card in list(self.panel.approvals.values()):
            r = card.req
            if r.tool == 'research_autopilot' and (r.args or {}).get('action') == 'start' and card.outcome.isHidden():
                out.append({'study': r.args.get('study') or '?', 'state': 'proposed', 'rounds': 0, 'verdict': None,
                            'question': r.args.get('question') or '', 'why': 'waiting for your approval card'})
        return out

    def refresh(self):
        from ..assistant.tools.autopilot import job_state
        from ..core import autoresearch as R
        ws = self._ws()
        js = job_state(self._ctx())
        try:
            for rec in R.studies(ws):
                R.sync(ws, rec, js)                     # the transitions the files show, into the history
            rows = R.summary_rows(ws, js)
        except (OSError, ValueError, KeyError, R.StudyError) as e:
            rows = []
            self.detail.setText(f'The studies could not be read: {e}')
        sel = self.selected()
        self.rows = self.pending() + rows
        self.tree.clear()
        for r in self.rows:
            it = QTreeWidgetItem([r['study'], TEXT.get(r['state'], r['state']), str(r.get('rounds') or 0),
                                  r.get('verdict') or '—'])
            it.setToolTip(0, r.get('question') or '')
            it.setToolTip(1, r.get('why') or '')
            self.tree.addTopLevelItem(it)
            if sel and r['study'] == sel:
                it.setSelected(True)
        if not sel and self.rows:
            self.tree.topLevelItem(0).setSelected(True)
        self._show()

    def selected(self) -> str:
        items = self.tree.selectedItems()
        return items[0].text(0) if items else ''

    def _row(self, sid: str):
        return next((r for r in self.rows if r['study'] == sid), None)

    def _show(self):
        from ..assistant.tools.autopilot import job_state
        from ..core import autoresearch as R
        sid = self.selected()
        row = self._row(sid)
        self._buttons(row)
        if row is None:
            self.chain.setText('')
            self.detail.setText('')
            return
        self.chain.setText(chain_text(row['state']))
        if row['state'] == 'proposed' and row.get('rounds') == 0:
            self.detail.setText(f'"{row["question"]}" — the plan waits for your approval card in Chat.')
            return
        try:
            st = R.status(self._ws(), sid, job_state(self._ctx()))
        except R.StudyError as e:
            self.detail.setText(str(e))
            return
        lines = [f'"{st["question"]}"', f'{st["why"]}', f'Next: {st["next"]}', '']
        for r in st['rounds']:
            lines.append(f'Round {r["round"]} — {r["experiment"]} ({r["kind"].replace("_", " ")}): '
                         f'{r["plan"]["runs"]} runs, {r["status"]}, verdict {r["verdict"] or "—"}; approved: '
                         f'{r["approval"]["how"]}' + (f'; jobs {", ".join(str(j) for j in r["jobs"])}'
                                                      if r['jobs'] else ''))
        if st.get('follow_ups_approved'):
            lines.append('Follow-ups approved ahead (within the approved plan) until Stop.')
        lines.append('')
        for h in st['history'][-8:]:
            lines.append(f'{(h.get("time") or "")[11:16]}  {TEXT.get(h["state"], h["state"])} — {h.get("why", "")}')
        self.detail.setText('\n'.join(lines))

    def _buttons(self, row):
        st = (row or {}).get('state')
        written = bool(row) and not (st == 'proposed' and not row.get('rounds'))
        set_enabled(self.stop_btn, written and st in ('running', 'approved', 'interrupted', 'follow_up', 'failed'),
                    'Nothing to stop' if row else 'Select a study')
        set_enabled(self.undo_btn, bool(row) and self._undoable(row['study']) is not None,
                    'No change of the assistant\'s to undo for this study' if row else 'Select a study')
        set_enabled(self.resume_btn, written and st in ('stopped', 'failed', 'interrupted', 'approved'),
                    'Nothing to resume' if row else 'Select a study')
        has = written and os.path.isfile(os.path.join(self._ws(), 'experiments', row['study'], 'report.md'))
        set_enabled(self.report_btn, has, 'No report yet' if row else 'Select a study')

    def _undoable(self, sid: str):
        """The newest activity entry of the assistant's for this study that Undo can take back."""
        sess = self.panel.session
        for e in reversed(sess.activity if sess else []):
            if e.get('tool') == 'research_autopilot' and (e.get('args') or {}).get('study') == sid \
                    and not e.get('undone') and e.get('kind') in ('edit', 'job', 'trash'):
                if e.get('kind') == 'job':
                    j = self.win.queue.get(e.get('job_id'))
                    if j is None or j.state in ('done', 'failed', 'cancelled'):
                        continue
                return e
        return None

    # ------------------------------------------------------------------ actions
    def stop(self) -> str:
        from ..assistant.tools.autopilot import _cancel, job_state
        from ..core import autoresearch as R
        sid = self.selected()
        if not sid:
            return ''
        ctx = self._ctx()
        if self.panel.agent is not None and self.panel.agent.busy:
            self.panel.stop()
        try:
            rec = R.stop(self._ws(), sid, _cancel(ctx), job_state(ctx), 'user', 'stopped by the user (Research ▸ Stop)')
        except R.StudyError as e:
            return str(e)
        msg = f'Stopped {sid}' + (f' (cancelled job {", ".join(map(str, rec["stopped"]["cancelled"]))})'
                                  if rec['stopped']['cancelled'] else '') + '.'
        self.win.log.append_line(f'assistant: {msg}')
        self.panel.status.setText(msg)
        self.refresh()
        return msg

    def undo(self) -> str:
        sid = self.selected()
        e = self._undoable(sid) if sid else None
        if e is None:
            return ''
        msg = self.panel.undo(e['id'])
        self.refresh()
        return msg

    def resume(self):
        sid = self.selected()
        if sid:
            return self.panel.propose({'tool': 'research_autopilot', 'args': {'action': 'resume', 'study': sid},
                                       'why': f'Continue the research study {sid}.'})

    def open_report(self):
        sid = self.selected()
        p = os.path.join(self._ws(), 'experiments', sid, 'report.md')
        if sid and os.path.isfile(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def find_suggestions(self):
        from ..core import autoresearch as R
        from . import tasks
        ws, proj = self._ws(), self.win.project
        self.suggest_btn.setEnabled(False)

        def done(rows):
            self.suggest_btn.setEnabled(True)
            self.show_suggestions(rows)
        tasks.run(self, lambda: R.suggestions(ws, proj), done, lambda e: done([]))

    def show_suggestions(self, rows: list):
        self.suggested = list(rows)
        self.sug.clear()
        for s in rows:
            it = QTreeWidgetItem([s['question'], ''])
            it.setToolTip(0, f'{s["id"]}\n{s.get("note") or ""}')
            self.sug.addTopLevelItem(it)
            b = button('Study this', lambda _=False, s=s: self.study_this(s),
                       tip='Ask the assistant to make this a study (its plan comes on an approval card)')
            self.sug.setItemWidget(it, 1, b)

    def study_this(self, s: dict):
        return self.panel.ask(f'Make this a research study: {s["question"]} (suggestion {s["id"]}: start from its '
                              'template, adjusted to my project)')
