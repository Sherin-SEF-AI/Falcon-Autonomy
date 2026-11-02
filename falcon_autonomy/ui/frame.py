"""The main window's frame: the top bar (the project, the menus, the command search, the server and the Assistant),
the navigator (its selection slides to the page chosen; each page's count beside it) and the page header every page
shares (its title, what it is for, and its next-step chips)."""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QRectF, QSize, Qt, QVariantAnimation
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QMenuBar, QPushButton, QSizePolicy, QStyle,
                             QStyledItemDelegate, QStyleOptionViewItem, QToolButton, QVBoxLayout, QWidget)

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
        self.setFixedHeight(46)
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 0, 12, 0)
        h.setSpacing(10)
        self.logo = motion.Dot(THEME.tok['accent'], 10)
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
        self.path = QLabel()
        self.path.setObjectName('projectPath')
        self.path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.path.setMinimumWidth(40)
        h.addWidget(self.path, 1)
        h.addSpacing(6)
        h.addWidget(menubar)
        h.addStretch(1)
        self.search = QToolButton()
        self.search.setObjectName('searchBox')
        self.search.setIcon(icons.icon('search', 14, THEME.tok['muted']))
        self.search.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.search.setText('  Search pages, runs and commands…')
        self.search.setToolTip('The command palette (Ctrl+K): every page and action by name')
        self.search.setMinimumWidth(220)
        self.search.setMaximumWidth(460)
        self.search.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search.clicked.connect(lambda: win.show_palette())
        h.addWidget(self.search, 2)
        h.addStretch(1)
        self.server = _Chip(lambda: win.go('home'))
        self.server_dot = self.server.dot
        h.addWidget(self.server)
        self.assistant = QPushButton('Assistant')
        self.assistant.setIcon(icons.icon('sparkle', 14, THEME.tok['accent_hi']))
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

    def set_server(self, state: str, text: str, tip: str = ''):
        tone, pulse = SERVER_LOOK.get(state, ('muted', False))
        self.server_dot.set_color(THEME.tok.get(tone) or THEME.color(tone).name(), pulse)
        self.server.text.setText(text)
        self.server.setToolTip(tip)


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
        self.setFixedHeight(28)

    def mouseReleaseEvent(self, ev):                            # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self.on_click()


class _NavDelegate(QStyledItemDelegate):
    """A navigator row: its icon (accent when selected), its name, and its count on the right in mono."""

    def paint(self, p: QPainter, opt: QStyleOptionViewItem, index):
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        name = index.data(ICON_ROLE)
        if name is None:                                        # a group header or a spacer
            super().paint(p, opt, index)
            return
        o = QStyleOptionViewItem(opt)
        self.initStyleOption(o, index)
        o.icon = icons.icon(name, 16, THEME.tok['accent_hi'] if sel else THEME.tok['muted'])
        o.state &= ~QStyle.StateFlag.State_HasFocus
        if sel:
            o.font.setWeight(600)
            o.palette.setColor(o.palette.ColorRole.Text, QColor(THEME.tok['ink']))
        super().paint(p, o, index)
        n = index.data(COUNT_ROLE)
        if n not in (None, ''):
            p.save()
            p.setFont(mono(11))
            p.setPen(QColor(THEME.tok['ink2'] if sel else THEME.tok['muted']))
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
            p.setBrush(QColor(THEME.tok['accent_soft']))
            r = self.pill.adjusted(0, 1, 0, -1)
            p.drawRoundedRect(r, 6, 6)
            p.setBrush(QColor(THEME.tok['accent']))
            p.drawRoundedRect(QRectF(r.left(), r.top() + 7, 3, r.height() - 14), 1.5, 1.5)
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
        v.setContentsMargins(24, 16, 24, 10)
        v.setSpacing(3)
        self.title = QLabel()
        self.title.setObjectName('pageTitle')
        v.addWidget(self.title)
        self.about = Note('', limit=200)
        v.addWidget(self.about)
        self.chips = QVBoxLayout()
        self.chips.setContentsMargins(0, 6, 0, 0)
        v.addLayout(self.chips)
        self.bars: dict = {}                      # page key -> its suggestion bar (shown while the page is)
        self.key = ''

    def show_page(self, page):
        self.key = page.key
        self.title.setText(page.title)
        self.about.setText(getattr(page, 'header_note', '') or page.about)
        self.about.expanded = False
        self.about._render()
        for k, bar in self.bars.items():
            bar.set_active(k == page.key)

    def add_bar(self, key, bar):
        self.bars[key] = bar
        self.chips.addWidget(bar)
        bar.set_active(key == self.key)
