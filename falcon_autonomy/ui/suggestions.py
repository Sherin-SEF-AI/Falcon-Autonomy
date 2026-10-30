"""Suggestions in the window: one-click chips on the pages they concern and the "Next best action" card on Home
(assistant.suggest finds them; docs/assistant.md, "Suggestions").

SuggestionEngine (win.suggestions) takes a snapshot of the window on the GUI thread (the project, this session's jobs,
the server, the GPU's utilisation from the status bar, the last Data ▸ Health report), runs the triggers on a
background task, and — when the assistant is on and its notice accepted — asks the fast model to rank and word the top
few, also on a task; every step is paced (assistant.suggest.Pacer) and nothing is asked when nothing new is there. A
job finishing or the project changing brings the next gather forward; a timer does the rest.

SuggestionBar: at most three chips under a page's heading, each the suggestion's title (its wording in the tooltip) and
a × that dismisses it for this project. A click opens the Assistant panel with the suggested tool call — an approval
card, or run at once when the permission mode lets it (AssistantService.propose). Settings ▸ Assistant ▸
Suggestions turns them off.
"""
from __future__ import annotations

import copy
import re
import time

from PyQt6.QtCore import QObject, QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
                             QWidgetItem)

from ..assistant import suggest as S
from . import icons, tasks

BAR_PAGES = ('worlds', 'scenarios', 'live', 'data', 'rarecases', 'training', 'stack')
TICK_MS = 10_000


class FlowLayout(QLayout):
    """Children left to right, wrapping to the next line when the width runs out (chips in a narrow column)."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self.items: list = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):                       # noqa: N802 (Qt's names)
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, i):                           # noqa: N802
        return self.items[i] if 0 <= i < len(self.items) else None

    def takeAt(self, i):                           # noqa: N802
        return self.items.pop(i) if 0 <= i < len(self.items) else None

    def expandingDirections(self):                 # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):                   # noqa: N802
        return True

    def heightForWidth(self, w):                   # noqa: N802
        return self._lay(QRect(0, 0, w, 0), True)

    def setGeometry(self, r):                      # noqa: N802
        super().setGeometry(r)
        self._lay(r, False)

    def sizeHint(self):                            # noqa: N802
        return self.minimumSize()

    def minimumSize(self):                         # noqa: N802
        s = QSize()
        for it in self.items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _lay(self, r: QRect, dry: bool) -> int:
        m = self.contentsMargins()
        x, y, line = r.x() + m.left(), r.y() + m.top(), 0
        right = r.right() - m.right()
        for it in self.items:
            if it.widget() is not None and it.widget().isHidden():
                continue
            hint = it.sizeHint()
            w = min(hint.width(), max(r.width() - m.left() - m.right(), 40))
            if x + w > right + 1 and line > 0:
                x, y, line = r.x() + m.left(), y + line + self.spacing(), 0
            if not dry:
                it.setGeometry(QRect(QPoint(x, y), QSize(w, hint.height())))
            x += w + self.spacing()
            line = max(line, hint.height())
        return y + line - r.y() + m.bottom()


class Chip(QFrame):
    """One suggestion: its title as a button (the wording in the tooltip) and a × to dismiss it."""

    def __init__(self, s: S.Suggestion, on_act, on_dismiss):
        super().__init__()
        self.setObjectName('suggestChip')
        self.s = s
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 0, 2, 0)
        h.setSpacing(0)
        self.btn = QToolButton()
        self.btn.setObjectName('suggestAct')
        self.btn.setText(s.title)
        self.btn.setIcon(icons.icon('sparkle', 12))
        self.btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn.setToolTip(f'{s.shown()}\n\nClick: {s.tool} — opens the Assistant panel with it (an approval card '
                            'when it changes something and your permission mode asks).')
        self.btn.setAccessibleName(f'Suggestion: {s.title}')
        self.btn.clicked.connect(lambda: on_act(self.s))
        self.x = QToolButton()
        self.x.setObjectName('suggestDismiss')
        self.x.setText('×')
        self.x.setToolTip('Dismiss this suggestion (for this project)')
        self.x.setAccessibleName(f'Dismiss: {s.title}')
        self.x.clicked.connect(lambda: on_dismiss(self.s))
        h.addWidget(self.btn)
        h.addWidget(self.x)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


class SuggestionBar(QWidget):
    """The chips of one page (at most suggest.MAX_PER_PAGE), hidden when there are none."""

    def __init__(self, engine, page_key: str):
        super().__init__()
        self.setObjectName('suggestBar')
        self.engine, self.key = engine, page_key
        self.flow = FlowLayout(self)
        self.chips: list = []
        self.hide()

    def update_chips(self):
        for c in self.chips:
            self.flow.removeWidget(c)
            c.deleteLater()
        self.chips = []
        for s in self.engine.for_page(self.key):
            c = Chip(s, self.engine.act, self.engine.dismiss)
            self.flow.addWidget(c)
            self.chips.append(c)
        self.setVisible(bool(self.chips))
        self.updateGeometry()


def _insert(page, bar: QWidget) -> bool:
    """Put the bar under the page's heading (and its note), or at the top of a page without one."""
    from .widgets import Note
    title = next((lb for lb in page.findChildren(QLabel) if lb.property('page')), None)
    if title is not None and title.parentWidget() is not None and title.parentWidget().layout() is not None:
        lay = title.parentWidget().layout()
        i = lay.indexOf(title)
        if i >= 0 and isinstance(lay, QVBoxLayout):
            nxt = lay.itemAt(i + 1)
            j = i + 2 if nxt is not None and isinstance(nxt.widget(), Note) else i + 1
            lay.insertWidget(j, bar)
            return True
    lay = page.layout()
    if isinstance(lay, QVBoxLayout):
        lay.insertWidget(0, bar)
        return True
    return False


GPU_PCT = re.compile(r'·\s*(\d+(?:\.\d+)?)\s*%\s*$')


class SuggestionEngine(QObject):
    changed = pyqtSignal()

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.pacer = S.Pacer()
        self.store = S.Store(win.workspace())
        self.cands: list = []
        self.busy = False
        self.phrasing = False
        self._again = False
        self._later = False
        self.stopped = False
        self.last_error = ''
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.refresh())
        self.timer.start(TICK_MS)

    # ------------------------------------------------------------------ state
    def enabled(self) -> bool:
        return bool(self.win.assistant.cfg().get('suggestions', True))

    def snapshot(self) -> S.State:
        w = self.win
        origin = getattr(w, '_job_origin', {})
        jobs = [S.job_facts(j, origin.get(j.id, '')) for j in w.queue.jobs]
        svc = w.service
        m = GPU_PCT.search(w.status.gpu.text() or '') if getattr(w, 'status', None) is not None else None
        data = w.page('data')
        health = getattr(getattr(data, 'health', None), 'report', None)
        return S.State(workspace=w.workspace(), doc=copy.deepcopy(w.project.doc), jobs=jobs,
                       server={'state': svc.server_state(), 'error': svc.server_error or ''},
                       gpu_util=float(m.group(1)) if m else None, health=health, now=time.time())

    def for_page(self, key: str, n: int = S.MAX_PER_PAGE) -> list:
        return S.for_page(self.cands, key, self.store.dismissed, n) if self.enabled() else []

    def best(self):
        rows = self.for_page('home', 1)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ the work
    def refresh(self, forced: bool = False):
        if self.stopped:
            return False
        if self.win.workspace() != self.store.workspace:
            self.store = S.Store(self.win.workspace())        # another project: its own dismissals
            self.cands = []
            self.changed.emit()
        if not self.enabled():
            if self.cands:
                self.cands = []
                self.changed.emit()
            return False
        if self.busy:
            self._again = self._again or forced
            return False
        now = time.monotonic()
        if not self.pacer.gather_due(now, forced):
            if forced and not self._later:            # an event too soon after the last gather: right after the pause
                self._later = True
                wait = max(0.0, self.pacer.min_forced_s - (now - self.pacer.last_gather))
                QTimer.singleShot(int(wait * 1000) + 50, self._run_later)
            return False
        self.pacer.gathered(now)
        self.busy = True
        st = self.snapshot()
        tasks.run(self, lambda: S.gather(st), self._gathered, self._failed)
        return True

    def _run_later(self):
        self._later = False
        self.refresh(forced=True)

    def stop(self):
        """No more gathers (the window is closing)."""
        self.timer.stop()
        self.stopped = True

    def _failed(self, e):
        self.busy = False
        self.last_error = type(e).__name__

    def _gathered(self, cands):
        self.busy = False
        if not self.enabled() or self.win.workspace() != self.store.workspace:
            cands = []                                         # turned off, or another project, while it ran
        self.pacer.recall(cands)
        self.cands = cands
        self.changed.emit()
        self.maybe_phrase()
        if self._again:
            self._again = False
            QTimer.singleShot(int(S.MIN_FORCED_S * 1000), lambda: self.refresh(forced=True))

    def maybe_phrase(self) -> bool:
        svc = self.win.assistant
        if self.phrasing or not svc.ready()[0]:
            return False
        top = [s for s in sorted(self.cands, key=lambda s: (-s.priority, s.key))
               if s.key not in self.store.dismissed][:S.PHRASE_TOP]
        now = time.monotonic()
        if not self.pacer.phrase_due(top, now):
            return False
        self.phrasing = True
        ctx = {'project': self.win.project.doc.get('name'), 'page': self.win.current_key()}

        def done(res):
            self.phrasing = False
            self.pacer.phrased(top, res, time.monotonic())
            live = {s.key for s in self.cands}
            S.apply_phrasing([s for s in self.cands if s.key in live], res)
            self.changed.emit()

        def failed(e):
            self.phrasing = False
            self.pacer.phrased(top, {}, time.monotonic())        # not asked again before PHRASE_EVERY_S
            self.last_error = getattr(e, 'message', None) or type(e).__name__
            self.win.log.append_line(f'assistant: suggestions kept their own wording ({self.last_error})')
        tasks.run(self, lambda: S.phrase(svc, top, ctx), done, failed)
        return True

    # ------------------------------------------------------------------ the user
    def act(self, s: S.Suggestion):
        return self.win.assistant.propose(s.action())

    def dismiss(self, s):
        self.store.dismiss(s.key if isinstance(s, S.Suggestion) else str(s))
        self.changed.emit()


def install(win):
    """The engine (win.suggestions), a bar on each page of BAR_PAGES, the Home card, and what brings a gather
    forward."""
    eng = SuggestionEngine(win)
    win.suggestions = eng
    win.suggestion_bars = {}
    for key in BAR_PAGES:
        page = win.page(key)
        if page is None:
            continue
        bar = SuggestionBar(eng, key)
        if _insert(page, bar):
            win.suggestion_bars[key] = bar
            eng.changed.connect(bar.update_chips)
    home = win.page('home')
    if home is not None and hasattr(home, 'show_suggestion'):
        eng.changed.connect(lambda: home.show_suggestion(eng.best()))
    win.bridge.finished.connect(lambda _job: eng.refresh(forced=True))
    win.bridge.added.connect(lambda _job: eng.refresh(forced=True))
    win.projectChanged.connect(lambda path: eng.refresh(forced=path is None))
    QTimer.singleShot(0, lambda: eng.refresh(forced=True))
    return eng
