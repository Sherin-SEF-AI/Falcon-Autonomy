"""The command palette (Ctrl+K): every action and page by name, and the project's and workspace's things — runs,
scenarios, worlds, rigs, models, evaluations, reports, drive sets, settings, the manual's headings, the job history —
filtered as you type (subsequence match), grouped by kind, the best group first. Enter opens the highlighted entry on
its page; the arrow keys move through every group."""
from __future__ import annotations

import time

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QStyle, QStyledItemDelegate,
                             QVBoxLayout)

from . import icons
from .theme import THEME

HEADER_H = 22
GROUP_ROLE = Qt.ItemDataRole.UserRole + 1
SUB_ROLE = Qt.ItemDataRole.UserRole + 2


def score(query: str, text: str) -> float | None:
    """A subsequence match score (higher is better), None when query's letters are not all in text in order. Word
    starts and consecutive letters count more."""
    q, t = query.lower().strip(), text.lower()
    if not q:
        return 0.0
    if q in t:
        return 100.0 - t.index(q)
    i, s, prev = 0, 0.0, -2
    for j, ch in enumerate(t):
        if i < len(q) and ch == q[i]:
            s += 3.0 if j == prev + 1 else (2.0 if j == 0 or t[j - 1] in ' -_/' else 1.0)
            prev = j
            i += 1
    return s if i == len(q) else None


class GroupDelegate(QStyledItemDelegate):
    """The first entry of each group carries the group's name in a band above it; the subtitle sits to the right,
    muted."""

    def sizeHint(self, option, index):
        sz = super().sizeHint(option, index)
        sz = QSize(sz.width(), max(sz.height(), 24))
        return QSize(sz.width(), sz.height() + HEADER_H) if index.data(GROUP_ROLE) else sz

    def paint(self, painter, option, index):
        group = index.data(GROUP_ROLE)
        r = QRect(option.rect)
        if group:
            band = QRect(r.left() + 8, r.top(), r.width() - 16, HEADER_H)
            painter.save()
            f = QFont(option.font)
            f.setPointSizeF(max(f.pointSizeF() - 1.5, 7.0))
            f.setWeight(QFont.Weight.DemiBold)
            painter.setFont(f)
            painter.setPen(THEME.color('muted'))
            painter.drawText(band, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, group.upper())
            painter.restore()
            r.setTop(r.top() + HEADER_H)
        opt = type(option)(option)
        opt.rect = r
        super().paint(painter, opt, index)
        sub = index.data(SUB_ROLE)
        if sub:
            painter.save()
            painter.setPen(THEME.color('muted') if not opt.state & QStyle.StateFlag.State_Selected else
                           opt.palette.highlightedText().color())
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(index.data(Qt.ItemDataRole.DisplayRole) or '') + 56
            box = QRect(r.left() + tw, r.top(), r.width() - tw - 10, r.height())
            if box.width() > 40:
                painter.drawText(box, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                 fm.elidedText(sub, Qt.TextElideMode.ElideMiddle, box.width()))
            painter.restore()


class CommandPalette(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle('Command palette')
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.resize(680, 480)
        v = QVBoxLayout(self)
        self.input = QLineEdit()
        self.input.setObjectName('paletteInput')
        self.input.setPlaceholderText('Type a command, or search runs, scenarios, worlds, models, settings, the '
                                      'manual…')
        self.list = QListWidget()
        self.list.setItemDelegate(GroupDelegate(self.list))
        self.list.setUniformItemSizes(False)
        self.hint = QLabel()
        self.hint.setProperty('muted', True)
        v.addWidget(self.input)
        v.addWidget(self.list, 1)
        v.addWidget(self.hint)
        self.commands = []
        for name, act in win.actions_.items():
            if name == 'palette' or not act.isEnabled():
                continue
            text = act.text().replace('&', '')
            self.commands.append({'kind': 'command', 'title': text, 'sub': act.shortcut().toString(),
                                  'ident': name, 'open': (lambda a: lambda _w: a.trigger())(act)})
        self.commands.sort(key=lambda c: c['title'])
        self._items = None                       # built on the first search (reading the workspace)
        self.build_s = 0.0
        self.input.textChanged.connect(self.filter)
        self.input.returnPressed.connect(self.run_current)
        self.list.itemActivated.connect(lambda it: self.run(it))
        self.filter('')
        self.input.setFocus()

    def items(self) -> list:
        if self._items is None:
            from . import search
            t0 = time.perf_counter()
            self._items = self.commands + search.entries(self.win)
            self.build_s = time.perf_counter() - t0
        return self._items

    def filter(self, text):
        from .search import KINDS, rank
        self.list.clear()
        groups = rank(text, self.commands if not text.strip() else self.items())
        n = 0
        for kind, rows in groups:
            title, icon, _ = KINDS.get(kind, (kind, 'more', 99))
            for k, e in enumerate(rows):
                label = e['title'] + (f'    {e["sub"]}' if kind == 'command' and e.get('sub') else '')
                it = QListWidgetItem(icons.icon(icon, 14), label)
                it.setData(Qt.ItemDataRole.UserRole, e)
                if k == 0 and len(groups) > 1:
                    it.setData(GROUP_ROLE, title)
                if kind != 'command' and e.get('sub'):
                    it.setData(SUB_ROLE, e['sub'])
                it.setToolTip(f'{title}: {e["title"]}' + (f' — {e["sub"]}' if e.get('sub') else ''))
                self.list.addItem(it)
                n += 1
        if text.strip() and getattr(self.win, 'assistant_panel', None) is not None and \
                self.win.assistant.cfg()['enabled']:                    # the request itself, to the assistant
            q = text.strip()
            e = {'kind': 'command', 'title': f'Ask Falcon: {q}', 'sub': '', 'ident': 'ask_text',
                 'open': lambda w, q=q: w.assistant.ask(q)}
            it = QListWidgetItem(icons.icon('sparkle', 14), e['title'])
            it.setData(Qt.ItemDataRole.UserRole, e)
            it.setData(GROUP_ROLE, 'Assistant')
            it.setToolTip('Send this to the assistant (Assistant panel)')
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)
        self.hint.setText('↑ ↓ to move · Enter to open · Esc to close' if n else
                          'Nothing matches — commands, runs, scenarios, worlds, rigs, models, evaluations, reports, '
                          'settings and the manual are searched.')

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_PageDown, Qt.Key.Key_PageUp):
            step = {Qt.Key.Key_Down: 1, Qt.Key.Key_Up: -1, Qt.Key.Key_PageDown: 8, Qt.Key.Key_PageUp: -8}[ev.key()]
            r = self.list.currentRow() + step
            self.list.setCurrentRow(max(0, min(r, self.list.count() - 1)))
            return
        super().keyPressEvent(ev)

    def run_current(self):
        it = self.list.currentItem()
        if it is not None:
            self.run(it)

    def run(self, it):
        e = it.data(Qt.ItemDataRole.UserRole)
        self.accept()
        e['open'](self.win)
