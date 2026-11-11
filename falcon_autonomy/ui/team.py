"""The team workspace in the pages (core.team): a comments panel for any object (a run in Data ▸ Browse, a triage item,
a failure cluster), and the conflict question when another member saved the project first."""
from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QMessageBox, QVBoxLayout, QWidget

from ..core import team as TM
from .widgets import PlaceholderList, button, label, set_enabled


def ident(win) -> dict:
    return TM.identity(win.settings)


class CommentsPanel(QWidget):
    """An object's comments, oldest first, and a line to add one (append-only: core.team.add_comment). set_ref(ref)
    shows another object; '' shows nothing."""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.ref = ''
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self.list = PlaceholderList('No comments yet. Comments are kept in the workspace\'s team folder, one file '
                                    'each, with your name (Settings ▸ Team).')
        self.list.setWordWrap(True)
        self.list.setMinimumHeight(90)
        self.list.setMaximumHeight(220)
        v.addWidget(self.list)
        row = QHBoxLayout()
        self.text = QLineEdit()
        self.text.setPlaceholderText('Add a comment…')
        self.text.returnPressed.connect(self.add)
        row.addWidget(self.text, 1)
        self.send = button('Comment', self.add)
        row.addWidget(self.send)
        v.addLayout(row)
        self.who = label('', muted=True, small=True)
        v.addWidget(self.who)
        self.set_ref('')

    def set_ref(self, ref: str):
        self.ref = ref or ''
        self.refresh()

    def refresh(self):
        self.list.clear()
        ok = bool(self.ref) and bool(self.win.project.path)
        set_enabled(self.text, ok, 'Save the project first: comments live in its workspace' if self.ref else
                    'Nothing chosen')
        set_enabled(self.send, ok, 'Save the project first: comments live in its workspace' if self.ref else
                    'Nothing chosen')
        if not self.ref:
            self.who.setText('')
            return
        me = ident(self.win)
        self.who.setText(f'as {me["user"]} on {me["host"]} · {self.ref}')
        try:
            cs = TM.comments(self.win.workspace(), self.ref)
        except (TM.TeamError, OSError):
            cs = []
        for c in cs:
            via = ' (via the assistant)' if c.get('via') else ''
            self.list.addItem(f'{c["user"]}{via} · {str(c.get("time", ""))[:16].replace("T", " ")}\n{c["text"]}')

    def add(self):
        t = self.text.text().strip()
        if not t or not self.ref:
            return None
        try:
            c = TM.add_comment(self.win.workspace(), self.ref, t, ident(self.win))
        except (TM.TeamError, OSError) as e:
            QMessageBox.warning(self, 'Comment', str(e))
            return None
        self.text.clear()
        self.refresh()
        return c


def save_conflict(win, err) -> bool:
    """Another member saved the project after this window read it: ask whether to save anyway (theirs is lost) —
    True when the user chose to."""
    r = QMessageBox.question(win, 'Saved by someone else', f'{err}\n\nSave anyway?',
                             QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                             QMessageBox.StandardButton.Cancel)
    return r == QMessageBox.StandardButton.Save
