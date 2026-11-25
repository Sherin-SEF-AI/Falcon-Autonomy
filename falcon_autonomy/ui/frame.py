"""The main window's frame: the top bar (the project, the active run/seed/rig/scenario/world as pills, the menus, the
command search, the server and the Assistant), the navigator (its selection slides to the page chosen; each page's
count beside it) and the page header every page shares (its title, what it is for, its next-step chips and the Focus
button).

The top bar never overflows: the project's folder is elided in the middle (the whole path in its tooltip) and the
context pills share what room is left — each pill's text is elided, and pills that do not fit at a readable width
fold into a "+N" chip whose tooltip lists them."""
from __future__ import annotations

import os

from PyQt6.QtCore import QEasingCurve, QRectF, QSize, Qt, QVariantAnimation
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout, QListWidget, QMenuBar, QPushButton, QSizePolicy,
                             QStyle, QStyledItemDelegate, QStyleOptionViewItem, QToolButton, QVBoxLayout, QWidget)

from ..core.paths import tilde
from . import icons, motion
from .theme import THEME, mono
from .widgets import Note

COUNT_ROLE = Qt.ItemDataRole.UserRole + 1
ICON_ROLE = Qt.ItemDataRole.UserRole + 2

SERVER_LOOK = {                      # server state -> (status colour, pulsing)
    'ready': ('good', False), 'shared': ('good', False), 'external': ('good', False), 'fake': ('muted', False),
    'starting': ('warning', True), 'stopped': ('muted', False), 'crashed': ('critical', False),
}


class TopBar(QWidget):
    """FALCON / project and its folder, the menus, the command search (Ctrl+K), the server chip and the Assistant."""

    def __init__(self, win, menubar: QMenuBar):
        super().__init__()
        self.win = win
        self.setObjectName('topBar')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(48)
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 0, 12, 0)
        h.setSpacing(10)
        self.logo = motion.Dot(THEME.tok['accent'], 8)
        h.addWidget(self.logo)
        brand = QLabel('FALCON')
        brand.setObjectName('brand')
        h.addWidget(brand)
        sep = QLabel('/')
        sep.setObjectName('crumbSep')
        h.addWidget(sep)
        self.project = QLabel()
        self.project.setObjectName('projectName')
        h.addWidget(self.project)
        self.path = ElidedLabel()
        self.path.setObjectName('projectPath')
        self.path.setMinimumWidth(140)              # the folder stays readable; pills fold first
        h.addWidget(self.path, 1)
        self.pills = ContextPills(win)
        h.addWidget(self.pills)
        h.addSpacing(6)
        h.addWidget(menubar)
        h.addSpacing(6)
        self.search = QToolButton()
        self.search.setObjectName('searchBox')
        self.search.setIcon(icons.icon('search', 14, THEME.tok['muted']))
        self.search.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.search.setText('  Search pages, runs and commands      Ctrl K')
        self.search.setToolTip('The command palette (Ctrl+K): every page and action by name')
        self.search.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search.clicked.connect(lambda: win.show_palette())
        box = QWidget()                          # the box's width (the theme's QToolButton min-width would win)
        box.setMinimumWidth(200)
        box.setMaximumWidth(460)
        box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        bl = QHBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.search)
        h.addWidget(box, 1)
        self.server = _Chip(lambda: win.go('home'))
        self.server_dot = self.server.dot
        h.addWidget(self.server)
        self.assistant = QPushButton('Assistant')
        self.assistant.setIcon(icons.icon('sparkle', 14, THEME.tok['accent_hi']))
        self.assistant.setProperty('role', 'ghost')
        self.assistant.setToolTip('Show or hide the Assistant panel (Ctrl+Shift+A)')
        self.assistant.clicked.connect(lambda: win.show_assistant(not win.assistant_dock.isVisible(), focus=True))
        h.addWidget(self.assistant)
        THEME.changed.connect(lambda _n: self._retint())

    def _retint(self):
        self.logo.set_color(THEME.tok['accent'])
        self.search.setIcon(icons.icon('search', 14, THEME.tok['muted']))
        self.assistant.setIcon(icons.icon('sparkle', 14, THEME.tok['accent_hi']))

    def set_project(self, name: str, path: str):
        self.project.setText(name)
        self.path.setText(tilde(path) if path else 'not saved yet')
        self.path.setToolTip(tilde(path) if path else 'Runs go to a temporary workspace until the project is saved')

    def overflow(self) -> int:
        """How many pixels the bar's contents need beyond its width (0: everything fits)."""
        lay = self.layout()
        need = lay.contentsMargins().left() + lay.contentsMargins().right()
        n = 0
        for i in range(lay.count()):
            it = lay.itemAt(i)
            w = it.widget()
            if w is not None and w.isHidden():
                continue
            need += it.minimumSize().width()
            n += 1
        need += lay.spacing() * max(n - 1, 0)
        return max(0, need - self.width())

    def set_server(self, state: str, text: str, tip: str = ''):
        tone, pulse = SERVER_LOOK.get(state, ('muted', False))
        self.server_dot.set_color(THEME.tok.get(tone) or THEME.color(tone).name(), pulse)
        self.server.text.setText(text)
        self.server.setToolTip(tip)


class ElidedLabel(QLabel):
    """A label whose text is cut in the middle to the width it gets (start and end kept); the whole text stays in
    `full` (and the tooltip the owner sets). It never asks for more room than its minimum."""

    def __init__(self, text: str = '', mode=Qt.TextElideMode.ElideMiddle):
        super().__init__()
        self.full = ''
        self.mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text):                                    # noqa: N802
        self.full = str(text or '')
        self._elide()

    def _elide(self):
        w = self.width()
        QLabel.setText(self, self.fontMetrics().elidedText(self.full, self.mode, w) if w > 0 else self.full)

    def natural(self) -> int:
        self.ensurePolished()                                   # the style sheet's font, not the default one
        return self.fontMetrics().horizontalAdvance(self.full) + 2

    def resizeEvent(self, ev):                                  # noqa: N802
        super().resizeEvent(ev)
        self._elide()

    def sizeHint(self):                                         # noqa: N802
        return QSize(self.natural(), super().sizeHint().height())

    def minimumSizeHint(self):                                  # noqa: N802
        return QSize(self.minimumWidth(), super().minimumSizeHint().height())


class _Pill(QFrame):
    """One context value: its text (elided to the room it gets; the whole value in the tooltip) and × to clear it."""

    def __init__(self, bar, key: str):
        super().__init__()
        self.bar, self.key = bar, key
        self.setObjectName('topChip')
        self.setProperty('pill', key)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(24)
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 0, 2, 0)
        h.setSpacing(2)
        self.text = ElidedLabel()
        self.text.setObjectName('chipText')
        self.text.setMinimumWidth(0)
        h.addWidget(self.text, 1)
        self.close_btn = QToolButton()
        self.close_btn.setText('×')
        self.close_btn.setAutoRaise(True)
        self.close_btn.setFixedSize(18, 18)
        self.close_btn.setToolTip(f'Clear the active {key}')
        self.close_btn.setAccessibleName(f'Clear the active {key}')
        self.close_btn.clicked.connect(lambda: bar.context().clear(key))
        h.addWidget(self.close_btn)
        if key == 'run':
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def chrome(self) -> int:
        """The pill's width besides its text."""
        m = self.layout().contentsMargins()
        return m.left() + m.right() + self.layout().spacing() + self.close_btn.width()

    def mouseReleaseEvent(self, ev):                            # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self.bar.open(self.key)


class ContextPills(QWidget):
    """The window's run context (ui.context.RunContext) as pills: Run <short id>, Seed, Rig, Scenario, World, each with
    × to clear it; a click on the Run pill opens the run on Data. Fits the width the top bar gives it (module doc)."""
    MIN_TEXT = 72                                               # below this a pill folds into "+N"

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)    # its pills never hold the top bar wide
        self.pills: dict = {}
        self.more = QLabel()
        self.more.setObjectName('chipText')
        h.addWidget(self.more)
        self.more.hide()
        self.hide()
        self.context().changed.connect(lambda _v: self.refresh())

    def context(self):
        from .context import context_of
        return context_of(self.win)

    def text_of(self, key: str) -> str:
        from .context import TITLES
        from .widgets import short_id
        v = self.context().get(key)
        if key == 'run':
            v = short_id(os.path.basename(v.rstrip('/')) or v, 6)
        return f'{TITLES[key]} {v}'

    def refresh(self):
        from .context import KEYS
        ctx = self.context()
        lay = self.layout()
        for k in KEYS:
            v = ctx.get(k)
            p = self.pills.get(k)
            if v and p is None:
                p = self.pills[k] = _Pill(self, k)
                lay.insertWidget(sum(1 for x in KEYS[:KEYS.index(k)] if x in self.pills), p)
            if p is None:
                continue
            if not v:
                self.pills.pop(k)
                lay.removeWidget(p)
                p.hide()
                p.deleteLater()
                continue
            p.text.setText(self.text_of(k))
            p.setToolTip(f'{k}: {v}' + ('\nClick to open it on Data' if k == 'run' else ''))
        self.setVisible(bool(self.pills))
        self.updateGeometry()
        self._fit()

    def open(self, key: str):
        v = self.context().get(key)
        if key == 'run' and v:
            page = self.win.go('data')
            if page is not None and hasattr(page, 'show_run') and os.path.isdir(v):
                page.show_run(v)

    def _ordered(self) -> list:
        from .context import KEYS
        return [self.pills[k] for k in KEYS if k in self.pills]

    def _natural(self) -> list:
        return [(p, p.text.natural(), p.chrome()) for p in self._ordered()]

    def sizeHint(self):                                         # noqa: N802
        sp = self.layout().spacing()
        w = sum(t + c for _p, t, c in self._natural()) + sp * max(len(self.pills) - 1, 0)
        return QSize(w, 26)

    def minimumSizeHint(self):                                  # noqa: N802
        return QSize(0, 26)

    def resizeEvent(self, ev):                                  # noqa: N802
        super().resizeEvent(ev)
        self._fit()

    def _fit(self):
        """Share the width among the pills: full texts when they fit, else each text cut to a common level (a short
        one keeps its own width); pills left with less than MIN_TEXT fold into "+N"."""
        rows = self._natural()
        if not rows:
            return
        sp = self.layout().spacing()
        avail = self.width()
        shown, folded = list(rows), []
        more_w = self.more.fontMetrics().horizontalAdvance('+5') + 8
        share = []
        while shown:
            room = avail - sum(c for _p, _t, c in shown) - sp * (len(shown) - 1) - (more_w + sp if folded else 0)
            texts = [t for _p, t, _c in shown]
            share = _water_fill(texts, room)
            if all(sh >= min(self.MIN_TEXT, t) - 0.5 for sh, t in zip(share, texts)) or len(shown) == 1:
                break
            folded.insert(0, shown.pop())
        for (p, _t, _c), w in zip(shown, share):
            p.show()
            p.text.setFixedWidth(max(int(w), 0))
        for p, _t, _c in folded:
            p.hide()
        self.more.setVisible(bool(folded))
        if folded:
            self.more.setText(f'+{len(folded)}')
            self.more.setToolTip('\n'.join(f'{p.key}: {self.context().get(p.key)}' for p, _t, _c in folded))


def _water_fill(widths: list, room: float) -> list:
    """Each width capped by a common level so that they sum to at most room (short ones keep their own width)."""
    if sum(widths) <= room:
        return list(widths)
    order = sorted(widths)
    left, n = max(float(room), 0.0), len(widths)
    level = 0.0
    for i, w in enumerate(order):
        if w * (n - i) <= left:
            left -= w
            continue
        level = left / (n - i)
        break
    return [min(w, level) for w in widths]


class _Chip(QFrame):
    """A rounded status chip: a dot (pulsing while something is under way) and a short text; click goes somewhere."""

    def __init__(self, on_click):
        super().__init__()
        self.setObjectName('topChip')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.on_click = on_click
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 0, 12, 0)
        h.setSpacing(4)
        self.dot = motion.Dot('', 7)
        self.text = QLabel()
        self.text.setObjectName('chipText')
        h.addWidget(self.dot)
        h.addWidget(self.text)
        self.setFixedHeight(26)

    def mouseReleaseEvent(self, ev):                            # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self.on_click()


class _NavDelegate(QStyledItemDelegate):
    """A navigator row: its icon (ink when selected), its name, and its count on the right in mono."""

    def paint(self, p: QPainter, opt: QStyleOptionViewItem, index):
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        name = index.data(ICON_ROLE)
        if name is None:                                        # a group header or a spacer
            super().paint(p, opt, index)
            return
        o = QStyleOptionViewItem(opt)
        self.initStyleOption(o, index)
        o.icon = icons.icon(name, 16, THEME.tok['ink'] if sel else THEME.tok['off'])
        o.state &= ~QStyle.StateFlag.State_HasFocus
        if sel:
            o.font.setWeight(600)
            o.palette.setColor(o.palette.ColorRole.Text, QColor(THEME.tok['ink']))
        super().paint(p, o, index)
        n = index.data(COUNT_ROLE)
        if n not in (None, ''):
            p.save()
            p.setFont(mono(11))
            p.setPen(QColor(THEME.tok['ink2'] if sel else THEME.tok['off']))
            p.drawText(opt.rect.adjusted(0, 0, -12, 0), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       str(n))
            p.restore()


class NavList(QListWidget):
    """The navigator: the selected page's pill (with an accent bar) slides to the new row."""

    def __init__(self):
        super().__init__()
        self.setObjectName('nav')
        self.setIconSize(QSize(16, 16))
        self.setItemDelegate(_NavDelegate(self))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.pill = QRectF()
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(motion.BASE)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim.valueChanged.connect(self._move)
        self.currentRowChanged.connect(self._to_row)
        self.verticalScrollBar().valueChanged.connect(lambda _v: self._to_row(self.currentRow(), animate=False))

    def _target(self, row) -> QRectF:
        it = self.item(row)
        return QRectF(self.visualItemRect(it)) if it is not None else QRectF()

    def _to_row(self, row, animate=True):
        to = self._target(row)
        if to.isNull():
            return
        if animate and motion.enabled() and not self.pill.isNull() and self.isVisible():
            self.anim.stop()
            self.anim.setStartValue(self.pill)
            self.anim.setEndValue(to)
            self.anim.start()
        else:
            self.anim.stop()
            self.pill = to
            self.viewport().update()

    def _move(self, r):
        self.pill = QRectF(r)
        self.viewport().update()

    def resizeEvent(self, ev):                                  # noqa: N802
        super().resizeEvent(ev)
        self._to_row(self.currentRow(), animate=False)

    def showEvent(self, ev):                                    # noqa: N802
        super().showEvent(ev)
        self._to_row(self.currentRow(), animate=False)

    def paintEvent(self, ev):                                   # noqa: N802
        if not self.pill.isNull():
            p = QPainter(self.viewport())
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(THEME.tok['nav_sel']))
            r = self.pill.adjusted(0, 1, 0, -1)
            p.drawRoundedRect(r, 4, 4)
            p.setBrush(QColor(THEME.tok['accent']))
            p.drawRect(QRectF(r.left(), r.top() + 8, 2, r.height() - 16))
            p.end()
        super().paintEvent(ev)


class PageHeader(QWidget):
    """The open page's title and what it is for (its first sentence; *more* reads the rest), with the page's
    next-step chips under it."""

    def __init__(self):
        super().__init__()
        self.setObjectName('pageHeader')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 14, 24, 10)
        v.setSpacing(2)
        self.title = QLabel()
        self.title.setObjectName('pageTitle')
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.title)
        row.addStretch(1)
        self.focus_btn = QToolButton()
        self.focus_btn.setObjectName('focusButton')
        self.focus_btn.setCheckable(True)
        self.focus_btn.setAutoRaise(True)
        self.focus_btn.setIcon(icons.icon('maximise', 14, THEME.tok['muted']))
        self.focus_btn.setToolTip('Focus: the page across the whole window — navigator, panels and side lists '
                                  'hidden (Ctrl+Shift+F; Esc leaves)')
        self.focus_btn.setAccessibleName('Focus mode')
        row.addWidget(self.focus_btn)
        v.addLayout(row)
        THEME.changed.connect(lambda _n: self.focus_btn.setIcon(icons.icon('maximise', 14, THEME.tok['muted'])))
        self.compact = False
        self.about = Note('', limit=200)
        v.addWidget(self.about)
        self.chips = QVBoxLayout()
        self.chips.setContentsMargins(0, 6, 0, 0)
        v.addLayout(self.chips)
        self.bars: dict = {}                      # page key -> its suggestion bar (shown while the page is)
        self.key = ''

    def set_compact(self, on: bool):
        """Focus mode: the title alone; what the page is for and its chips come back on leaving."""
        self.compact = bool(on)
        self.about.setVisible(not on)
        for k, bar in self.bars.items():
            bar.set_active(k == self.key)
            if on:
                bar.setVisible(False)

    def show_page(self, page):
        self.key = page.key
        self.title.setText(page.title)
        self.about.setText(getattr(page, 'header_note', '') or page.about)
        self.about.expanded = False
        self.about._render()
        for k, bar in self.bars.items():
            bar.set_active(k == page.key)
            if self.compact:
                bar.setVisible(False)

    def add_bar(self, key, bar):
        self.bars[key] = bar
        self.chips.addWidget(bar)
        bar.set_active(key == self.key)
