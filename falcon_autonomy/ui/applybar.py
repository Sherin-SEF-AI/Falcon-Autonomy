"""The apply bar: when a page's fields differ from what was last applied, a bar rises at the page's foot — how many
changes are not applied yet, Discard (puts every field back) and Apply. It watches the page's inputs itself (line
edits, spin boxes, combo boxes, check boxes, text boxes); fields that take effect at once (the theme, say) are left
out. snapshot() takes the fields as they are now as the applied state (after a successful Apply)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                             QPlainTextEdit, QWidget)

from . import motion
from .widgets import button


def _value(w):
    if isinstance(w, QLineEdit):
        return w.text()
    if isinstance(w, QAbstractSpinBox):
        return w.value() if hasattr(w, 'value') else w.text()
    if isinstance(w, QComboBox):
        return (w.currentIndex(), w.currentText()) if w.isEditable() else w.currentIndex()
    if isinstance(w, QCheckBox):
        return w.checkState()
    if isinstance(w, QPlainTextEdit):
        return w.toPlainText()
    return None


def _restore(w, v):
    w.blockSignals(True)
    try:
        if isinstance(w, QLineEdit):
            w.setText(v)
        elif isinstance(w, QAbstractSpinBox) and hasattr(w, 'setValue'):
            w.setValue(v)
        elif isinstance(w, QComboBox):
            if isinstance(v, tuple):
                w.setCurrentIndex(v[0])
                w.setEditText(v[1])
            else:
                w.setCurrentIndex(v)
        elif isinstance(w, QCheckBox):
            w.setCheckState(v)
        elif isinstance(w, QPlainTextEdit):
            w.setPlainText(v)
    finally:
        w.blockSignals(False)


class ApplyBar(QFrame):
    def __init__(self, page: QWidget, apply, exclude=()):
        super().__init__(page)
        self.setObjectName('applyBar')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.page, self.apply_fn = page, apply
        self.exclude = {id(w) for w in exclude}
        h = QHBoxLayout(self)
        h.setContentsMargins(16, 8, 16, 8)
        h.setSpacing(10)
        self.dot = motion.Dot('', 6)
        h.addWidget(self.dot)
        self.text = QLabel()
        self.text.setProperty('mono', True)
        h.addWidget(self.text, 1)
        self.discard_btn = button('Discard', self.discard, role='ghost', tip='Put every changed field back')
        self.apply_btn = button('Apply', self.apply, role='primary')
        h.addWidget(self.discard_btn)
        h.addWidget(self.apply_btn)
        self.fields: list = []
        self.saved: dict = {}
        self._pending = False
        self.hide()
        QTimer.singleShot(0, self.watch)                      # after the page (and its late sections) is built

    def watch(self):
        """Take every input of the page (not excluded) and its current value as the applied state."""
        seen = {id(w) for w in self.fields}
        for w in self.page.findChildren(QWidget):
            if id(w) in seen or id(w) in self.exclude or _value(w) is None or self.isAncestorOf(w):
                continue
            if isinstance(w, QLineEdit) and isinstance(w.parent(), (QAbstractSpinBox, QComboBox)):
                continue                                      # a spin box's or combo box's own line edit
            self.fields.append(w)
            for sig in ('textChanged', 'valueChanged', 'currentIndexChanged', 'stateChanged'):
                s = getattr(w, sig, None)
                if s is not None and hasattr(s, 'connect'):
                    try:
                        s.connect(self._later)
                    except TypeError:
                        pass
        self.snapshot()

    def snapshot(self):
        self.saved = {id(w): _value(w) for w in self.fields}
        self.update_bar()

    def changed(self) -> list:
        out = []
        for w in self.fields:
            try:
                if _value(w) != self.saved.get(id(w)):
                    out.append(w)
            except RuntimeError:                              # a field deleted with its section
                continue
        return out

    def _later(self, *_a):
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self.update_bar)

    def update_bar(self):
        self._pending = False
        changed = self.changed()
        mark = {id(w) for w in changed}
        for w in self.fields:                               # a changed field wears an amber edge until applied
            try:
                on = id(w) in mark
                if bool(w.property('changed')) != on:
                    w.setProperty('changed', on)
                    w.style().unpolish(w)
                    w.style().polish(w)
            except RuntimeError:
                continue
        n = len(changed)
        from .theme import THEME
        if n:
            self.text.setText(f'{n} change{"s" if n != 1 else ""} not applied')
            self.dot.set_color(THEME.tok['accent'], pulse=False)
            if not self.isVisible():
                self.show()
                QTimer.singleShot(0, lambda: motion.slide_in(self, dy=self.height()))
        elif self.isVisible():
            self.hide()

    def discard(self):
        for w in self.changed():
            _restore(w, self.saved.get(id(w)))
        self.update_bar()

    def apply(self):
        self.apply_fn()
