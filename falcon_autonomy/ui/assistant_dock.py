"""The Assistant panel (a right-side dock): a conversation with Falcon's assistant, which uses the app through its tools
(docs/assistant.md).

Chat: the mode (Ask, Auto, Autopilot) and the main model at the top; the plan checklist the model keeps; the messages —
the user's, the assistant's (Markdown, streamed; run, job and scenario ids are links that go there; numbers no tool
returned are marked "not from Falcon's data"), tool-call cards (name, arguments, result, duration; collapsed), approval
cards (what, the exact change from Tool.preview, the estimate, Approve / Approve all of this kind / Reject), the jobs it
queued with their progress, Falcon's events; then the box: Enter sends, Shift+Enter is a new line, Esc stops.
Activity: everything it changed, newest first, each with Undo (a project edit through the window's undo stack, a job
by cancelling it, runs moved to the trash by moving them back).

QtHost connects the Qt-free agent (assistant.agent) to the window: the agent's thread emits events through a signal
that lands on the GUI thread; gui tools run on the GUI thread; the undo stack is read there.
"""
from __future__ import annotations

import html
import json
import os
import re
import threading

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPalette, QTextDocument
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame, QHBoxLayout,
                             QHeaderView, QLabel, QMenu, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                             QSizePolicy, QStackedWidget, QTabWidget, QToolButton, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from ..assistant import agent as A
from ..assistant import privacy
from ..assistant.provider import Cancelled, fmt_tokens
from ..assistant.service import NOTICE
from ..assistant.session import Session, wire
from ..core.budget import fmt_duration
from . import icons
from .theme import STATUS, THEME, mono, sans
from .widgets import PlaceholderTree, button, clear_layout, label, set_enabled

MARK = {'dark': STATUS['serious'], 'light': '#a8461f'}          # "not from Falcon's data": ≥ 4.5:1 on either panel
MODE_TIPS = {
    'ask': 'Ask: every change waits for your approval on a card',
    'auto': 'Auto: changes run without asking; moving runs to the trash still asks',
    'autopilot': 'Autopilot: works through the goal across many steps and long jobs without asking; moving runs to '
                 'the trash still asks; Stop ends it',
}


# --------------------------------------------------------------------------------------------- the host
class _Bridge(QObject):
    event = pyqtSignal(str, object)
    call = pyqtSignal(object)
    usage = pyqtSignal(object)


class QtHost(A.Host):
    has_window = True

    def __init__(self, win, panel):
        super().__init__()
        self.win, self.panel = win, panel
        self.bridge = _Bridge()
        self.bridge.call.connect(self._run_call)
        self.cancelled = lambda: False

    @property
    def queue(self):
        return self.win.queue

    def tool_context(self):
        from ..assistant.tool_api import ToolContext
        w = self.win
        return ToolContext(project=w.project, settings=w.settings, queue=w.queue, service=w.service, win=w,
                           workspace=w.workspace())

    def event(self, kind, payload):
        self.bridge.event.emit(kind, payload)

    @staticmethod
    def _run_call(h):
        try:
            h['res'] = h['fn']()
        except Exception as e:                  # noqa: BLE001 (re-raised on the agent's thread)
            h['err'] = e
        h['ev'].set()

    def run_gui(self, fn):
        if threading.current_thread() is threading.main_thread():
            return fn()
        h = {'fn': fn, 'ev': threading.Event()}
        self.bridge.call.emit(h)
        while not h['ev'].wait(0.1):
            if self.cancelled():
                raise Cancelled()
        if 'err' in h:
            raise h['err']
        return h.get('res')

    def undo_index(self):
        return self.run_gui(lambda: self.win.undo.index())

    def context_summary(self) -> dict:
        def read():
            w = self.win
            return {'project': w.project.doc.get('name'), 'page': w.current_key()}
        try:
            return self.run_gui(read)
        except Cancelled:
            return {}


# --------------------------------------------------------------------------------------------- rendering
def _known_runs(win) -> dict:
    cat = getattr(win.service, 'catalog', None)
    try:
        return {r['name']: r['path'] for r in (cat.list('run', 2000) if cat is not None else [])}
    except Exception:                          # noqa: BLE001 (a catalogue being rebuilt)
        return {}


def _known_scenarios(win) -> set:
    try:
        from ..core import scenarios as SC
        return set(SC.LIBRARY) | {s.get('id') for s in win.project.doc.get('scenarios') or [] if isinstance(s, dict)}
    except Exception:                          # noqa: BLE001
        return set()


WORD = re.compile(r'[A-Za-z0-9][\w.\-]{2,}')
JOB = re.compile(r'\b([Jj]ob)\s+#?(\d+)\b')


def render_html(text: str, ungrounded=(), win=None, final=True) -> str:
    """Markdown as rich text for a QLabel; when final, run/job/scenario ids become falcon:// links and the numbers no
    tool returned are marked."""
    doc = QTextDocument()
    doc.setDefaultFont(sans(12))
    doc.setMarkdown(text or '', QTextDocument.MarkdownFeature.MarkdownDialectGitHub)
    out = doc.toHtml()
    m = re.search(r'<body[^>]*>(.*)</body>', out, re.S)
    body = m.group(1) if m else out
    if not final:
        return body
    runs = _known_runs(win) if win is not None else {}
    scns = _known_scenarios(win) if win is not None else set()
    jobs = {j.id for j in win.queue.jobs} if win is not None else set()
    link = THEME.tok['ink']
    mark = MARK.get(THEME.name, MARK['dark'])
    nums = sorted({re.sub(r'\s?%$|[kKMB]$', '', u) for u in ungrounded}, key=len, reverse=True)
    num_re = re.compile(r'(?<![\w.])(' + '|'.join(re.escape(n) for n in nums) + r')(?![\w]|\.\d)') if nums else None

    def text_node(s: str) -> str:
        def word(mw):
            w = mw.group(0)
            if w in runs:
                return f'<a href="falcon://run/{html.escape(w)}" style="color:{link}">{w}</a>'
            if w in scns:
                return f'<a href="falcon://scenario/{html.escape(w)}" style="color:{link}">{w}</a>'
            return w
        s = WORD.sub(word, s) if (runs or scns) else s
        s = JOB.sub(lambda mj: f'<a href="falcon://job/{mj.group(2)}" style="color:{link}">{mj.group(0)}</a>'
                    if int(mj.group(2)) in jobs else mj.group(0), s)
        if num_re is not None:
            s = num_re.sub(lambda mn: f'<span style="color:{mark}; text-decoration:underline" title="not from '
                           f'Falcon\'s data">{mn.group(1)}</span>', s)
        return s
    parts = re.split(r'(<[^>]+>)', body)
    inside_a = False
    for i, p in enumerate(parts):
        if p.startswith('<'):
            inside_a = (inside_a or p.lower().startswith('<a ')) and not p.lower().startswith('</a')
            continue
        if p and not inside_a:
            parts[i] = text_node(p)
    return ''.join(parts)


class Bubble(QLabel):
    """One message: the user's (boxed), the assistant's (Markdown), an event of Falcon's (muted) or an error."""
    linkTo = pyqtSignal(str)

    def __init__(self, kind: str, text: str = '', win=None):
        super().__init__()
        self.kind, self.raw, self.win, self.ungrounded, self.final = kind, text, win, [], kind != 'assistant'
        self.setWordWrap(True)
        self.setMinimumWidth(40)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                     Qt.TextInteractionFlag.LinksAccessibleByMouse |
                                     Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
        self.linkActivated.connect(self.linkTo.emit)
        if kind == 'user':
            self.setObjectName('bubbleUser')
            self.setContentsMargins(8, 6, 8, 6)
        elif kind == 'event':
            self.setProperty('muted', True)
            self.setProperty('small', True)
        elif kind == 'error':
            self.setObjectName('errorBox')
            self.setContentsMargins(8, 6, 8, 6)
        self.setAccessibleName({'user': 'Your message', 'assistant': 'Assistant', 'event': 'Falcon event',
                                'error': 'Error'}.get(kind, kind))
        self.render()

    def set_text(self, text: str, final: bool = False, ungrounded=()):
        self.raw, self.final, self.ungrounded = text, final, list(ungrounded)
        self.render()

    def render(self):
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Link, THEME.color('ink'))
        self.setPalette(pal)
        if self.kind == 'assistant':
            self.setTextFormat(Qt.TextFormat.RichText)
            body = render_html(self.raw, self.ungrounded, self.win, self.final)
            if self.final and self.ungrounded:
                mark = MARK.get(THEME.name, MARK['dark'])
                body += (f'<p style="color:{mark}; font-size:11px">Not from Falcon\'s data: '
                         f'{html.escape(", ".join(self.ungrounded))} — no tool returned these numbers.</p>')
            self.setText(body)
        else:
            self.setTextFormat(Qt.TextFormat.PlainText)
            self.setText(self.raw)


class ToolCard(QFrame):
    """A tool call: one line (name, duration, what it returned), the arguments and the result behind it."""

    def __init__(self, name: str, arguments: str, tier: str = 'read'):
        super().__init__()
        self.setObjectName('toolCard')
        self.name, self.arguments, self.tier = name, arguments, tier
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 2, 6, 2)
        v.setSpacing(2)
        self.head = QToolButton()
        self.head.setCheckable(True)
        self.head.setProperty('text-button', True)
        self.head.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.head.setIcon(icons.icon('chev-r', 12))
        self.head.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.head.toggled.connect(self._toggle)
        self.body = QLabel()
        self.body.setProperty('mono', True)
        self.body.setProperty('small', True)
        self.body.setWordWrap(True)
        self.body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.hide()
        v.addWidget(self.head)
        v.addWidget(self.body)
        self.result, self.ms, self.summary, self.error, self.approval = '', None, '', False, ''
        self._update()

    def done(self, result: str, ms: float, summary: str, error: bool, approval: str = ''):
        self.result, self.ms, self.summary, self.error, self.approval = result, ms, summary, error, approval
        self._update()

    def _update(self):
        when = '…' if self.ms is None else (f'{self.ms:.0f} ms' if self.ms < 1000 else f'{self.ms / 1000:.1f} s')
        how = {'approved': 'approved', 'approved_all': 'approved (all of this kind)', 'allowed': 'approved earlier',
               'auto': 'ran without asking', 'rejected': 'declined', 'stopped': 'stopped'}.get(self.approval, '')
        bits = [self.name, when] + ([how] if how else []) + ([self.summary] if self.summary else [])
        self.head.setText('  ' + ' · '.join(bits))
        self.head.setToolTip(' · '.join(bits))
        if self.error:
            self.head.setIcon(icons.status_icon('warning'))
        elif self.ms is not None:
            self.head.setIcon(icons.icon('chev-d' if self.head.isChecked() else 'chev-r', 12))
        try:
            args = json.dumps(json.loads(self.arguments or '{}'), indent=1, ensure_ascii=False)
        except ValueError:
            args = self.arguments
        res = self.result
        try:
            res = json.dumps(json.loads(res), indent=1, ensure_ascii=False)
        except (ValueError, TypeError):
            pass
        self.body.setText(f'arguments\n{args}\n\nresult\n{res[:6000]}' if self.ms is not None else
                          f'arguments\n{args}')

    def _toggle(self, on):
        self.body.setVisible(on)
        if not self.error and self.ms is not None:
            self.head.setIcon(icons.icon('chev-d' if on else 'chev-r', 12))


class ApprovalCard(QFrame):
    """An act or trash call waiting for the user: what it does, the exact change, the estimate, and the choice."""

    def __init__(self, req: A.Approval, on_decide):
        super().__init__()
        self.setObjectName('card')
        self.req, self.on_decide = req, on_decide
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        trash = req.tier == 'trash'
        head = QHBoxLayout()
        title = label(f'Approve {req.tool}?', h=True)
        title.setAccessibleName(f'Approval needed: {req.tool}')
        head.addWidget(title, 1)
        badge = label('moves runs to the trash' if trash else 'changes something', small=True)
        badge.setProperty('crit' if trash else 'ink2', True)
        head.addWidget(badge)
        v.addLayout(head)
        if req.description:
            d = label(req.description.split('. ')[0].rstrip('.') + '.', muted=True, wrap=True, small=True)
            d.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            v.addWidget(d)
        args = label(json.dumps(req.args, ensure_ascii=False), mono=True, small=True, wrap=True, sel=True)
        args.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.addWidget(args)
        if req.preview:
            pv = QPlainTextEdit(str(req.preview))
            pv.setReadOnly(True)
            pv.setFont(mono(11))
            pv.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            pv.setFixedHeight(min(140, pv.fontMetrics().lineSpacing() * (str(req.preview).count('\n') + 1) + 14))
            pv.setAccessibleName('The exact change')
            v.addWidget(pv)
        est = req.estimate if isinstance(req.estimate, dict) else {}
        if est:
            s = est.get('seconds')
            txt = ' — '.join(x for x in ((f'about {fmt_duration(s)}' if s else ''), str(est.get('what') or '')) if x)
            if txt:
                v.addWidget(label(txt, muted=True, small=True, wrap=True))
        self.row = QHBoxLayout()
        self.approve = button('Approve', lambda: self.decide('approve'), role='primary')
        self.approve.setAccessibleName(f'Approve {req.tool}')
        self.row.addWidget(self.approve)
        self.all = None
        if req.allow_all:
            self.all = button('Approve all of this kind', lambda: self.decide('approve_all'),
                              tip=f'Run {req.tool} without asking for the rest of this conversation')
            self.row.addWidget(self.all)
        self.reject = button('Reject', lambda: self.decide('reject'))
        self.row.addWidget(self.reject)
        self.row.addStretch(1)
        v.addLayout(self.row)
        self.outcome = label('', muted=True, small=True)
        self.outcome.hide()
        v.addWidget(self.outcome)

    def decide(self, choice):
        self.settle(choice)
        self.on_decide(choice)

    def settle(self, choice):
        for b in (self.approve, self.all, self.reject):
            if b is not None:
                b.hide()
        self.outcome.setText({'approve': 'Approved', 'approve_all': 'Approved — and the rest of this kind',
                              'reject': 'Rejected', 'stopped': 'Stopped before it ran'}.get(choice, choice))
        self.outcome.show()


class JobChip(QFrame):
    """A job the assistant queued: its title, state and progress, kept current; Jobs opens the Jobs panel."""

    def __init__(self, win, job_id):
        super().__init__()
        self.setObjectName('toolCard')
        self.win, self.job_id = win, job_id
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 4, 6, 4)
        v = QVBoxLayout()
        v.setSpacing(2)
        self.title = label('', small=True)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setProperty('inline', True)
        v.addWidget(self.title)
        v.addWidget(self.bar)
        h.addLayout(v, 1)
        b = QToolButton()
        b.setText('Jobs')
        b.setProperty('text-button', True)
        b.setToolTip('Show this job in the Jobs panel')
        b.clicked.connect(self.show_in_panel)
        h.addWidget(b)
        win.bridge.changed.connect(self._on)
        win.bridge.finished.connect(self._on)
        job = win.queue.get(job_id)
        if job is not None:
            self._on(job)

    def _on(self, job):
        if job.id != self.job_id:
            return
        st = {'running': f'running {100 * job.progress:.0f} %', 'waiting': 'waiting'}.get(job.state, job.state)
        self.title.setText(f'Job {job.id} · {job.spec.title} · {st}')
        self.title.setToolTip(job.summary())
        self.bar.setValue(int(1000 * job.progress))
        self.bar.setVisible(job.state not in ('done', 'failed', 'cancelled'))

    def show_in_panel(self):
        show_job(self.win, self.job_id)


def show_job(win, job_id):
    """The Jobs panel, with this job's row selected."""
    win.show_jobs()
    dock = win.jobs_dock.widget()
    it = dock.items.get(job_id)
    if it is not None:
        dock.tabs.setCurrentIndex(0)
        dock.tree.setCurrentItem(it)


class PlanView(QFrame):
    """The model's checklist (plan_update), live; hidden while there is none."""

    def __init__(self):
        super().__init__()
        self.setObjectName('toolCard')
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 4, 8, 6)
        v.setSpacing(2)
        self.head = label('', h=True, small=True)
        v.addWidget(self.head)
        self.items = QVBoxLayout()
        self.items.setSpacing(1)
        v.addLayout(self.items)
        self.plan: list = []
        self.hide()

    def set_plan(self, plan: list):
        self.plan = list(plan or [])
        clear_layout(self.items)
        if not self.plan:
            self.hide()
            return
        done = sum(bool(p.get('done')) for p in self.plan)
        self.head.setText(f'Plan · {done} of {len(self.plan)} done')
        for p in self.plan:
            t = html.escape(str(p.get('text', '')))
            lb = QLabel(f'<span style="color:{THEME.tok["muted"]}">✓ <s>{t}</s></span>' if p.get('done')
                        else f'○ {t}')
            lb.setTextFormat(Qt.TextFormat.RichText)
            lb.setWordWrap(True)
            lb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            lb.setAccessibleName(('done: ' if p.get('done') else 'to do: ') + str(p.get('text', '')))
            self.items.addWidget(lb)
        self.show()


class Composer(QPlainTextEdit):
    """The message box: Enter sends, Shift+Enter is a new line, Esc stops; grows to six lines."""
    send = pyqtSignal()
    stop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName('composer')
        self.setPlaceholderText('Ask Falcon… (Enter to send, Shift+Enter for a new line)')
        self.setTabChangesFocus(True)
        self.setAccessibleName('Message to the assistant')
        self.document().contentsChanged.connect(self._fit)
        self._fit()

    def keyPressEvent(self, ev):                       # noqa: N802
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.send.emit()
            return
        if ev.key() == Qt.Key.Key_Escape:
            self.stop.emit()
            return
        super().keyPressEvent(ev)

    def _fit(self):
        lines = max(2, min(6, int(self.document().size().height())))
        self.setFixedHeight(int(self.fontMetrics().lineSpacing() * lines + 14))


class ConsentDialog(QDialog):
    """The one-time notice: exactly what is sent and what never is. Accepting turns the assistant on."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('What the assistant sends')
        self.resize(560, 520)
        v = QVBoxLayout(self)
        t = label(NOTICE, wrap=True, sel=True)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        box = QWidget()
        bv = QVBoxLayout(box)
        bv.addWidget(t)
        bv.addStretch(1)
        sa.setWidget(box)
        v.addWidget(sa, 1)
        bb = QDialogButtonBox()
        self.ok = bb.addButton('Accept and turn on', QDialogButtonBox.ButtonRole.AcceptRole)
        self.ok.setProperty('role', 'primary')
        bb.addButton('Not now', QDialogButtonBox.ButtonRole.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


def ensure_consent(parent, svc) -> bool:
    """Show the notice when it has not been accepted; True when the assistant may send."""
    if svc.consented() and svc.cfg()['enabled']:
        return True
    if not svc.consented():
        d = ConsentDialog(parent)
        if d.exec() != QDialog.DialogCode.Accepted:
            return False
        svc.accept_notice()
    svc.set_cfg(enabled=True)
    return True


class PayloadPreview(QDialog):
    """Exactly what the next request sends (after redaction): the messages, and the tools by name."""

    def __init__(self, parent, payload: dict, tool_names: list):
        super().__init__(parent)
        self.setWindowTitle('What will be sent')
        self.resize(720, 560)
        v = QVBoxLayout(self)
        v.addWidget(label(f'To {payload.get("_to", "the server")}: the messages below, as JSON, and the schemas of '
                          f'{len(tool_names)} tools ({", ".join(tool_names[:12])}{" …" if len(tool_names) > 12 else ""}'
                          '). Home folders are shown as ~ and secrets as [secret], as sent.', muted=True, wrap=True))
        ed = QPlainTextEdit(json.dumps(payload['messages'], indent=1, ensure_ascii=False))
        ed.setReadOnly(True)
        ed.setFont(mono(11))
        v.addWidget(ed, 1)
        bb = QDialogButtonBox()
        send = bb.addButton('Send', QDialogButtonBox.ButtonRole.AcceptRole)
        send.setProperty('role', 'primary')
        bb.addButton('Cancel', QDialogButtonBox.ButtonRole.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


# --------------------------------------------------------------------------------------------- the panel
class AssistantPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.svc = win.assistant
        self.attachments: list = []
        self.cards: dict = {}               # tool call id -> ToolCard
        self.approvals: dict = {}           # tool call id -> ApprovalCard
        self.stream: Bubble | None = None
        self.stream_text = ''
        self.host = QtHost(win, self)
        self.host.bridge.event.connect(self.on_event)
        self.session = None
        self.agent = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(8, 4, 6, 4)
        top.setSpacing(6)
        self.mode = QComboBox()
        for m in A.MODES:
            self.mode.addItem(A.MODE_TEXT[m], m)
            self.mode.setItemData(self.mode.count() - 1, MODE_TIPS[m], Qt.ItemDataRole.ToolTipRole)
        self.mode.setToolTip('Permission mode: ' + ' · '.join(MODE_TIPS.values()))
        self.mode.setAccessibleName('Permission mode')
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.model = QComboBox()
        self.model.setToolTip('The main model: plans, calls Falcon\'s tools, reasons over results')
        self.model.setAccessibleName('Main model')
        self.model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model.setMinimumContentsLength(10)
        self.model.activated.connect(self._model_changed)
        self.menu_btn = QToolButton()
        self.menu_btn.setIcon(icons.icon('more', 16))
        self.menu_btn.setToolTip('More: preview what is sent, clear, the audit log, settings')
        self.menu_btn.setAccessibleName('More assistant actions')
        self.menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(self.menu_btn)
        self.preview_act = m.addAction('Show what will be sent before each message')
        self.preview_act.setCheckable(True)
        self.preview_act.toggled.connect(lambda on: self.svc.set_cfg(preview=bool(on)))
        m.addAction('Clear the conversation', self.clear)
        m.addAction('Open the audit log', self.open_audit)
        m.addSeparator()
        m.addAction('Settings ▸ Assistant…', self.open_settings)
        self.menu_btn.setMenu(m)
        top.addWidget(self.mode)
        top.addWidget(self.model, 1)
        top.addWidget(self.menu_btn)
        v.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        v.addWidget(self.tabs, 1)
        self.pages = QStackedWidget()
        self.tabs.addTab(self.pages, 'Chat')
        # the empty state: what to set up
        self.empty = QWidget()
        ev = QVBoxLayout(self.empty)
        ev.setContentsMargins(20, 24, 20, 24)
        ev.addStretch(1)
        ic = QLabel()
        ic.setPixmap(icons.icon('sparkle', 28).pixmap(28, 28))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ev.addWidget(ic)
        self.empty_title = label('Falcon\'s assistant', title=True)
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ev.addWidget(self.empty_title)
        self.empty_msg = label('', muted=True, wrap=True)
        self.empty_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ev.addWidget(self.empty_msg)
        row = QHBoxLayout()
        row.addStretch(1)
        self.empty_btn = button('', self._empty_action, role='primary')
        row.addWidget(self.empty_btn)
        row.addStretch(1)
        ev.addLayout(row)
        ev.addStretch(2)
        self.pages.addWidget(self.empty)
        # the chat
        chat = QWidget()
        cv = QVBoxLayout(chat)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        self.banner = label('', small=True, wrap=True)
        self.banner.setObjectName('errorBox')
        self.banner.setContentsMargins(8, 4, 8, 4)
        self.banner.hide()
        cv.addWidget(self.banner)
        self.plan = PlanView()
        cv.addWidget(self.plan)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName('panel')
        self.list = QVBoxLayout(inner)
        self.list.setContentsMargins(10, 8, 10, 8)
        self.list.setSpacing(8)
        self.list.addStretch(1)
        self.scroll.setWidget(inner)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._stick)
        self._at_bottom = True
        self.scroll.verticalScrollBar().valueChanged.connect(
            lambda val: setattr(self, '_at_bottom', val >= self.scroll.verticalScrollBar().maximum() - 24))
        cv.addWidget(self.scroll, 1)
        self.status = label('', muted=True, small=True)
        self.status.setContentsMargins(10, 2, 10, 2)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        cv.addWidget(self.status)
        box = QHBoxLayout()
        box.setContentsMargins(8, 4, 8, 8)
        box.setSpacing(6)
        self.composer = Composer()
        self.composer.send.connect(self.send)
        self.composer.stop.connect(self.stop)
        box.addWidget(self.composer, 1)
        side = QVBoxLayout()
        side.setSpacing(4)
        self.send_btn = QPushButton()
        self.send_btn.setProperty('role', 'primary')
        self.send_btn.clicked.connect(self._send_or_stop)
        self.attach_btn = QToolButton()
        self.attach_btn.setIcon(icons.icon('file', 16))
        self.attach_btn.setToolTip('Attach a text file (a log, a config); only text is ever sent')
        self.attach_btn.setAccessibleName('Attach a text file')
        self.attach_btn.clicked.connect(self.attach)
        side.addWidget(self.send_btn)
        side.addWidget(self.attach_btn, 0, Qt.AlignmentFlag.AlignRight)
        side.addStretch(1)
        box.addLayout(side)
        cv.addLayout(box)
        self.attached = label('', muted=True, small=True)
        self.attached.setContentsMargins(10, 0, 10, 4)
        self.attached.hide()
        cv.addWidget(self.attached)
        self.pages.addWidget(chat)
        # activity
        act = QWidget()
        av = QVBoxLayout(act)
        av.setContentsMargins(0, 0, 0, 0)
        self.activity = PlaceholderTree('Nothing changed yet. Every change the assistant makes — a project edit, a '
                                        'job, runs moved to the trash — is listed here with Undo.')
        self.activity.setHeaderLabels(('Time', 'Change', ''))
        self.activity.setRootIsDecorated(False)
        hd = self.activity.header()
        hd.setStretchLastSection(False)
        hd.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hd.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.activity.setColumnWidth(0, 64)
        self.activity.setColumnWidth(2, 64)
        av.addWidget(self.activity)
        self.tabs.addTab(act, 'Activity')
        self.setMinimumWidth(300)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_stream)
        THEME.changed.connect(lambda _n: self._rerender())
        self.svc.ask_handler = self.ask
        self.svc.propose_handler = self.propose
        self.load_session()
        self.refresh_state()

    # ------------------------------------------------------------------ state
    def load_session(self):
        if self.agent is not None:
            self.agent.close()
        self.session = Session(self.win.workspace())
        cfg = self.svc.cfg()
        self.agent = A.Agent(self.svc, self.session, self.host, cfg.get('mode', 'ask'))
        self.host.cancelled = lambda: self.agent.cancel.is_set()
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(max(0, A.MODES.index(self.agent.mode)))
        self.mode.blockSignals(False)
        self.rebuild()
        for n in self.session.notes:
            self.win.log.append_line(f'assistant: {n}')

    def follow_workspace(self):
        """The project was saved to a new place: keep this conversation there (unless one is already there)."""
        from ..assistant.session import folder
        new = self.win.workspace()
        if self.session.workspace == new:
            return
        if os.path.isfile(os.path.join(folder(new), 'conversation.json')) or self.agent.busy:
            self.load_session()
            return
        self.session.workspace = new
        self.session.path = os.path.join(folder(new), 'conversation.json')
        if self.session.messages:
            self.session.save()

    def refresh_state(self):
        cfg = self.svc.cfg()
        ok, why = self.svc.ready()
        has = bool(self.session and self.session.messages)
        self.pages.setCurrentIndex(1 if ok or has else 0)
        self.banner.setText(why)
        self.banner.setVisible(bool(not ok and has))
        if not cfg['enabled']:
            self.empty_msg.setText('An assistant that uses Falcon for you through its own tools: it finds runs, '
                                   'explains results with the numbers Falcon measured, sets up scenarios and queues '
                                   'jobs — asking before it changes anything. It runs on a model server with your own '
                                   'API key (Groq by default) and sends text only.')
            self.empty_btn.setText('Set up the assistant')
        elif not self.svc.consented():
            self.empty_msg.setText(why)
            self.empty_btn.setText('Read what is sent…')
        else:
            self.empty_msg.setText(why)
            self.empty_btn.setText('Open Settings ▸ Assistant')
        self.preview_act.blockSignals(True)
        self.preview_act.setChecked(bool(cfg.get('preview')))
        self.preview_act.blockSignals(False)
        ids = [m['id'] for m in cfg.get('models') or [] if isinstance(m, dict) and m.get('id')]
        if cfg['main_model'] not in ids:
            ids.insert(0, cfg['main_model'])
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItems(ids)
        self.model.setCurrentText(cfg['main_model'])
        self.model.blockSignals(False)
        self._buttons()

    def _buttons(self):
        busy = self.agent is not None and self.agent.busy
        ok, why = self.svc.ready()
        self.send_btn.setText('Stop' if busy else 'Send')
        self.send_btn.setIcon(icons.icon('stop' if busy else 'send', 14, THEME.tok['panel']))
        self.send_btn.setToolTip('Stop the assistant (Esc)' if busy else 'Send (Enter)')
        set_enabled(self.send_btn, busy or ok, why)
        set_enabled(self.composer, ok or busy, why)
        set_enabled(self.attach_btn, ok and not busy, why or 'Wait for the answer first')

    def _empty_action(self):
        cfg = self.svc.cfg()
        if cfg['enabled'] and not self.svc.consented():
            if ensure_consent(self, self.svc):
                self.refresh_state()
            return
        self.open_settings()

    def open_settings(self):
        page = self.win.go('settings')
        if page is not None and hasattr(page, 'show_assistant'):
            page.show_assistant()

    def open_audit(self):
        from ..assistant.session import Audit
        rows = Audit(self.win.workspace()).read(500)
        d = QDialog(self)
        d.setWindowTitle('Assistant audit log')
        d.resize(820, 480)
        lv = QVBoxLayout(d)
        lv.addWidget(label(f'Every request to the model server from this workspace ({len(rows)} shown, newest last): '
                           'time (UTC), purpose, model, status, tokens, tools called, the SHA-256 of the payload — '
                           'never the payload itself.', muted=True, wrap=True))
        ed = QPlainTextEdit('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) or 'No requests yet.')
        ed.setReadOnly(True)
        ed.setFont(mono(11))
        lv.addWidget(ed, 1)
        d.show()
        return d

    def _mode_changed(self, i):
        m = self.mode.itemData(i)
        if self.agent is not None:
            self.agent.mode = m
        self.svc.set_cfg(mode=m)

    def _model_changed(self, _i):
        t = self.model.currentText().strip()
        if t:
            self.svc.set_cfg(main_model=t)

    # ------------------------------------------------------------------ the list
    def _add(self, w: QWidget):
        self.list.insertWidget(self.list.count() - 1, w)
        return w

    def _stick(self, _lo, hi):
        if self._at_bottom:
            self.scroll.verticalScrollBar().setValue(hi)

    def _bubble(self, kind, text, final=True, ungrounded=()):
        b = Bubble(kind, '', self.win)
        b.linkTo.connect(self.navigate)
        b.set_text(text, final, ungrounded)
        return self._add(b)

    def rebuild(self):
        """Draw the conversation from the session (a restart, another project, a clear)."""
        clear_layout(self.list)
        self.list.addStretch(1)
        self.cards.clear()
        self.approvals.clear()
        self.stream = None
        msgs = self.session.snapshot()
        if not msgs:
            hint = label('Ask about this project or ask for something to be done — for example: "Which drives had '
                         'collisions?", "Why did the last evaluation get worse?", "Set up a rainy cut-in scenario and '
                         'search it for failures". In Ask mode every change waits for your approval.',
                         muted=True, wrap=True)
            hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self._add(hint)
        results = {m.get('tool_call_id'): m for m in msgs if m.get('role') == 'tool'}
        for m in msgs:
            role = m.get('role')
            if role == 'user':
                self._bubble('event' if m.get('_event') else 'user', m.get('_display') or m.get('content') or '')
            elif role == 'assistant':
                if m.get('content'):
                    self._bubble('assistant', m['content'], True, m.get('_ungrounded') or [])
                for tc in m.get('tool_calls') or []:
                    fn = tc.get('function') or {}
                    card = self._add(ToolCard(fn.get('name', ''), fn.get('arguments', '')))
                    r = results.get(tc.get('id'))
                    if r is not None:
                        card.done(r.get('content') or '', r.get('_ms') or 0.0, r.get('_summary') or '',
                                  bool(r.get('_error')), r.get('_approval') or '')
                    self.cards[tc.get('id')] = card
        self.plan.set_plan(self.session.plan)
        self._fill_activity()
        self._usage_text()

    def _rerender(self):
        for i in range(self.list.count()):
            w = self.list.itemAt(i).widget()
            if isinstance(w, Bubble):
                w.render()
        self.plan.set_plan(self.plan.plan)
        self._buttons()                                  # the icons are tinted per theme
        self.menu_btn.setIcon(icons.icon('more', 16))
        self.attach_btn.setIcon(icons.icon('file', 16))

    # ------------------------------------------------------------------ sending
    def _send_or_stop(self):
        if self.agent.busy:
            self.stop()
        else:
            self.send()

    def attach(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Attach a text file', self.win.workspace(),
                                           'Text (*.txt *.log *.json *.yaml *.yml *.md *.csv *.xosc *.xodr *.py);;'
                                           'All files (*)')
        if p:
            self.add_attachment(p)

    def add_attachment(self, path: str):
        _c, why = A.compose('', attachments=[path])
        if why:
            self.status.setText(why)
            return False
        self.attachments.append(path)
        self.attached.setText('Attached: ' + ', '.join(os.path.basename(a) for a in self.attachments) +
                              ' (sent with the next message)')
        self.attached.show()
        return True

    def send(self, text: str | None = None, context=None) -> bool:
        if self.agent.busy:
            return False
        if self.session.workspace != self.win.workspace():
            self.load_session()
        text = (self.composer.toPlainText() if text is None else text).strip()
        if not text and not self.attachments:
            return False
        ok, why = self.svc.ready()
        if not ok:
            if self.svc.cfg()['enabled'] and not self.svc.consented() and ensure_consent(self, self.svc):
                ok, why = self.svc.ready()
            if not ok:
                self.refresh_state()
                self.status.setText(why)
                return False
        content, refused = A.compose(text, context, self.attachments)
        if refused:
            self._bubble('error', refused)
            self.pages.setCurrentIndex(1)
            return False
        if self.svc.cfg().get('preview'):
            from ..assistant import tool_api as T
            msgs = [self.agent.system_message()] + wire(self.session.snapshot()) + [{'role': 'user',
                                                                                     'content': content}]
            names = [s['function']['name'] for s in T.schemas(gui=True)]
            d = PayloadPreview(self, {'messages': privacy.redact(msgs), '_to': self.svc.cfg()['base_url']}, names)
            if d.exec() != QDialog.DialogCode.Accepted:
                return False
        shown = text + (('\n📎 ' + ', '.join(os.path.basename(a) for a in self.attachments)) if self.attachments
                        else '')
        self.pages.setCurrentIndex(1)
        self.banner.hide()
        self._bubble('user', shown)
        self._at_bottom = True
        self.composer.clear()
        self.attachments = []
        self.attached.hide()
        started = self.agent.start(content, display=shown)
        self.status.setText('Thinking…')
        self._buttons()
        return started

    def stop(self):
        if self.agent is not None and self.agent.busy:
            self.agent.stop()
            self.status.setText('Stopping…')

    def clear(self):
        if self.agent.busy:
            self.status.setText('Stop the assistant before clearing the conversation.')
            return
        self.session.clear()
        self.agent.my_jobs.clear()
        self.rebuild()
        self.refresh_state()
        self.status.setText('Cleared.')

    def ask(self, text: str, context=None):
        """AssistantService.ask: show the panel with the request — sent at once when the assistant is ready."""
        self.win.show_assistant(True)
        if self.svc.ready()[0] and not self.agent.busy:
            return self.send(text, context)
        self.composer.setPlainText(text)
        self.composer.setFocus()
        return False

    def propose(self, action: dict):
        """AssistantService.propose: an approval card for a suggested action; approved, it runs like any tool call."""
        from ..assistant import tool_api as T
        self.win.show_assistant(True)
        self.pages.setCurrentIndex(1)
        t = T.REGISTRY.get(action['tool'])
        if t is None:
            self._bubble('error', f'Suggested action {action["tool"]!r} is not a tool Falcon has.')
            return None
        args = action.get('args') or {}
        req = A.Approval(f'proposal-{id(action)}', t.name, t.tier, args, description=action.get('why') or
                         t.description)
        req.allow_all = False
        try:
            req.preview = privacy.redact(t.preview(self.host.tool_context(), **args)) if t.preview else ''
        except Exception as e:                  # noqa: BLE001
            req.preview = f'(no preview: {type(e).__name__})'

        def decided(choice):
            if choice != 'approve':
                return
            from . import tasks
            card = self._add(ToolCard(t.name, json.dumps(args)))
            tasks.run(self, lambda: self.agent.run_proposed(t.name, args),
                      lambda res: card.done(json.dumps(res, default=str), 0.0, A.summarize(res),
                                            bool(res.get('error')), 'approved'))
        card = self._add(ApprovalCard(req, decided))
        return card

    # ------------------------------------------------------------------ the agent's events
    def on_event(self, kind, p):
        if kind == 'delta':
            if self.stream is None:
                self.stream = self._bubble('assistant', '', final=False)
                self.stream_text = ''
            self.stream_text += p
            if not self._render_timer.isActive():
                self._render_timer.start(60)
            self.status.setText('Writing…')
        elif kind == 'assistant':
            if self.stream is None and p.get('content'):
                self.stream = self._bubble('assistant', '', final=False)
            if self.stream is not None:
                self._render_timer.stop()
                self.stream.set_text(p.get('content') or self.stream_text, True, p.get('ungrounded') or [])
                if not (p.get('content') or self.stream_text):
                    self.stream.hide()
            self.stream, self.stream_text = None, ''
            if p.get('cut'):
                self._bubble('event', 'The answer was cut at the model\'s output limit.')
            self.status.setText('Working…' if not p.get('final') else '')
        elif kind == 'tool_start':
            self.cards[p['id']] = self._add(ToolCard(p['name'], p.get('arguments') or '', p.get('tier', 'read')))
            self.status.setText(f'Running {p["name"]}…')
        elif kind == 'tool_done':
            card = self.cards.get(p['id'])
            if card is None:
                card = self.cards[p['id']] = self._add(ToolCard(p['name'], p.get('arguments') or ''))
            card.done(p.get('result') or '', p.get('ms') or 0.0, p.get('summary') or '', p.get('error'),
                      p.get('approval') or '')
        elif kind == 'approval':
            card = ApprovalCard(p, p.decide)
            self.approvals[p.call_id] = card
            self._add(card)
            self._at_bottom = True
            self.status.setText(f'Waiting for your approval: {p.tool}')
            self.win.show_assistant(True)
        elif kind == 'approval_done':
            card = self.approvals.pop(p['id'], None)
            if card is not None and card.outcome.isHidden():
                card.settle(p['choice'])
        elif kind == 'plan':
            self.plan.set_plan(p)
        elif kind == 'job':
            self._add(JobChip(self.win, p))
        elif kind == 'activity':
            self._fill_activity()
        elif kind == 'event':
            self._bubble('event', p['text'])
        elif kind == 'waiting':
            self.status.setText(('Waiting for job ' + ', '.join(str(j) for j in p) +
                                 ' — the assistant continues when one finishes (Stop ends the wait)') if p else '')
        elif kind == 'retry':
            what = 'rate limited' if p.get('status') == 429 else f'server error {p.get("status") or ""}'.strip()
            self.status.setText(f'{what.capitalize()} — retrying in {p["wait_s"]:.0f} s (attempt {p["attempt"]})')
        elif kind == 'notice':
            self.status.setText(str(p))
        elif kind == 'usage':
            self._usage_text()
        elif kind == 'error':
            self._bubble('error', p['message'])
            self.win.log.append_line(f'assistant: {p["message"]}')
        elif kind == 'turn':
            if p['state'] != 'started':
                if self.stream is not None:
                    self._render_timer.stop()
                    self.stream.set_text(self.stream_text, True)
                    self.stream = None
                for card in self.approvals.values():
                    card.settle('stopped')
                self.approvals.clear()
                self.status.setText({'stopped': 'Stopped.', 'failed': '', 'done': ''}.get(p['state'], ''))
                self._usage_text()
            self._buttons()

    def _render_stream(self):
        if self.stream is not None:
            self.stream.set_text(self.stream_text, False)

    def _usage_text(self):
        u = self.session.usage if self.session else {}
        if u.get('requests'):
            self.composer.setToolTip(f'This conversation: {fmt_tokens(u["prompt"] + u["completion"])} tokens in '
                                     f'{u["requests"]} requests, about ${u["cost"]:.4f} by your price table')

    # ------------------------------------------------------------------ activity and navigation
    def _fill_activity(self):
        self.activity.clear()
        for e in reversed(self.session.activity):
            it = QTreeWidgetItem([(e.get('time') or '')[11:16], e.get('text', ''), ''])
            it.setToolTip(0, (e.get('time') or '') + ' UTC')
            it.setToolTip(1, e.get('text', '') + ('\n' + json.dumps(e.get('args'), ensure_ascii=False)
                                                  if e.get('args') else ''))
            self.activity.addTopLevelItem(it)
            b = QToolButton()
            b.setText('Undone' if e.get('undone') else 'Undo')
            b.setProperty('text-button', True)
            can = e.get('kind') in ('edit', 'job', 'trash') and not e.get('undone')
            set_enabled(b, can, 'Already undone' if e.get('undone') else 'This change cannot be undone from here',
                        {'edit': 'Undo these project edits', 'job': 'Cancel this job',
                         'trash': 'Move it back out of the trash'}.get(e.get('kind'), ''))
            b.clicked.connect(lambda _=False, ident=e.get('id'): self.undo(ident))
            self.activity.setItemWidget(it, 2, b)

    def undo(self, ident: str) -> str:
        e = next((a for a in self.session.activity if a.get('id') == ident), None)
        if e is None:
            return ''
        msg = A.undo_activity(e, self.win, self.win.queue, self.win.workspace())
        if msg.startswith(('Undid', 'Cancelled', 'Restored')):
            self.session.update_activity(ident, undone=True)
            self.session.save()
        self.win.log.append_line(f'assistant: {msg}')
        self.status.setText(msg)
        self._fill_activity()
        return msg

    def navigate(self, href: str):
        u = QUrl(href)
        if u.scheme() in ('http', 'https'):
            QDesktopServices.openUrl(u)
            return
        if u.scheme() != 'falcon':
            return
        what, ident = u.host(), u.path().lstrip('/')
        if what == 'run':
            path = _known_runs(self.win).get(ident)
            if path:
                self.win.go('data').show_run(path)
        elif what == 'job':
            show_job(self.win, int(ident))
        elif what == 'scenario':
            page = self.win.go('scenarios')
            if page is not None:
                page.select(ident)

    def shutdown(self):
        if self.agent is not None:
            self.agent.close()
        self.svc.ask_handler = self.svc.propose_handler = None


# --------------------------------------------------------------------------------------------- installing it
PANEL_W = 360


def place(win):
    """Dock the panel at the right when the window has room for it beside the pages (their minimum width); else float
    it over the window's right edge — so showing it never makes the window wider than it is (a 1280 px screen). A
    panel the user floated or moved stays where they put it."""
    dock = win.assistant_dock
    if dock.isVisible() or getattr(dock, '_user_placed', False):
        return
    room = win.width() - win.centralWidget().minimumSizeHint().width()
    for d in (win.jobs_dock, win.log_dock):
        if d.isVisible() and win.dockWidgetArea(d) in (Qt.DockWidgetArea.LeftDockWidgetArea,
                                                       Qt.DockWidgetArea.RightDockWidgetArea):
            room -= d.width()
    if room >= dock.minimumSizeHint().width() + 8:
        if dock.isFloating():
            dock.setFloating(False)
        QTimer.singleShot(0, lambda: win.resizeDocks([dock], [min(PANEL_W, room - 8)], Qt.Orientation.Horizontal))
        return
    dock.setFloating(True)
    g = win.frameGeometry()
    h = max(420, win.height() - 120)
    dock.setGeometry(g.right() - PANEL_W - 16, g.top() + 60, PANEL_W, h)
def install(win):
    """Add the Assistant dock (right side, hidden until asked), its status-bar badge, Ctrl+Shift+A to show or hide it
    and the command palette's "Ask Falcon…"."""
    from PyQt6.QtWidgets import QDockWidget
    panel = AssistantPanel(win)
    dock = QDockWidget('Assistant', win)
    dock.setObjectName('Assistant')
    dock.setWidget(panel)
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                     QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                     QDockWidget.DockWidgetFeature.DockWidgetClosable)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.hide()
    dock.topLevelChanged.connect(lambda _f: setattr(dock, '_user_placed', dock.isVisible()))
    win.assistant_panel, win.assistant_dock = panel, dock

    badge = QToolButton()
    badge.setObjectName('assistantBadge')
    badge.clicked.connect(lambda: win.show_assistant(not dock.isVisible()))
    win.statusBar().addWidget(badge)
    win.assistant_badge = badge

    def show_usage(t=None):
        c = win.assistant.cfg()
        t = t or win.assistant.totals
        if not c['enabled']:
            badge.setText('Assistant off')
            badge.setToolTip('Show the Assistant panel (Ctrl+Shift+A); set it up in Settings ▸ Assistant')
            return
        model = (t.get('model') or c['main_model']).split('/')[-1]
        if t.get('requests'):
            cost = f' · ${t["cost"]:.4f}' if t['requests'] > t.get('unpriced', 0) else ''
            badge.setText(f'Assistant · {model} · {fmt_tokens(t["prompt"] + t["completion"])} tokens{cost}')
        else:
            badge.setText(f'Assistant · {model}')
        badge.setToolTip(f'This session: {t["prompt"]} prompt and {t["completion"]} completion tokens in '
                         f'{t["requests"]} requests' + (f', about ${t["cost"]:.4f} estimated from your price table '
                                                         '(Settings ▸ Assistant)' if t['requests'] else '') +
                         '. Click to show or hide the Assistant panel (Ctrl+Shift+A).')
    panel.host.bridge.usage.connect(show_usage)
    win.assistant.usage_listeners.append(panel.host.bridge.usage.emit)    # from the agent's thread, onto the GUI's
    win.refresh_assistant_badge = show_usage
    show_usage()

    view = next((a.menu() for a in win.menuBar().actions() if a.menu() and a.text() == '&View'), None)
    win._act('assistant_panel', 'Assistant panel', lambda: win.show_assistant(not dock.isVisible()), 'Ctrl+Shift+A',
             view, tip='Show or hide the Assistant panel')
    win._act('ask', 'Ask Falcon…', lambda: win.show_assistant(True, focus=True), menu=view,
             tip='Ask the assistant to do something in Falcon')
    return panel
