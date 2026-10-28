"""The command palette (Ctrl+K): every action and page by name, filtered as you type (subsequence match)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout


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


class CommandPalette(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle('Command palette')
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.resize(560, 420)
        v = QVBoxLayout(self)
        self.input = QLineEdit()
        self.input.setObjectName('paletteInput')
        self.input.setPlaceholderText('Type a command or a page…')
        self.list = QListWidget()
        v.addWidget(self.input)
        v.addWidget(self.list)
        self.entries = []
        for name, act in win.actions_.items():
            if name == 'palette' or not act.isEnabled():
                continue
            text = act.text().replace('&', '')
            sc = act.shortcut().toString()
            self.entries.append((text, sc, act))
        self.input.textChanged.connect(self.filter)
        self.input.returnPressed.connect(self.run_current)
        self.list.itemActivated.connect(lambda it: self.run(it))
        self.filter('')
        self.input.setFocus()

    def filter(self, text):
        self.list.clear()
        ranked = []
        for e in self.entries:
            s = score(text, e[0])
            if s is not None:
                ranked.append((-s, e[0], e))
        for _, _, (label, sc, act) in sorted(ranked, key=lambda r: (r[0], r[1])):
            it = QListWidgetItem(f'{label}{"    " + sc if sc else ""}')
            it.setData(Qt.ItemDataRole.UserRole, act)
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            r = self.list.currentRow() + (1 if ev.key() == Qt.Key.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(r, self.list.count() - 1)))
            return
        super().keyPressEvent(ev)

    def run_current(self):
        it = self.list.currentItem()
        if it is not None:
            self.run(it)

    def run(self, it):
        act = it.data(Qt.ItemDataRole.UserRole)
        self.accept()
        act.trigger()
