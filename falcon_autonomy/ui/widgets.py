"""Small building blocks shared by the pages."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                             QSlider, QVBoxLayout, QWidget)

from ..core.paths import tilde


def label(text='', *, muted=False, small=False, title=False, mono=False, h=False, wrap=False, sel=False) -> QLabel:
    lb = QLabel(text)
    for k, v in (('muted', muted), ('small', small), ('title', title), ('mono', mono), ('h', h)):
        if v:
            lb.setProperty(k, True)
    lb.setWordWrap(wrap)
    if sel:
        lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lb


def path_label(path: str) -> QLabel:
    """A path as shown on screen: from ~, selectable, the full path in the tooltip."""
    lb = label(tilde(path), mono=True, sel=True)
    lb.setToolTip(tilde(path))
    return lb


def button(text, slot=None, role=None, tip='') -> QPushButton:
    b = QPushButton(text)
    if role:
        b.setProperty('role', role)
    if slot is not None:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


def hline() -> QFrame:
    f = QFrame()
    f.setProperty('hair', True)
    f.setFixedHeight(1)
    return f


class Section(QWidget):
    """A titled block of a page: a heading, an optional note, then content."""

    def __init__(self, title: str, note: str = '', parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 12)
        v.setSpacing(6)
        v.addWidget(label(title, title=True))
        if note:
            v.addWidget(label(note, muted=True, wrap=True))
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        v.addLayout(self.body)

    def add(self, w):
        if isinstance(w, QWidget):
            self.body.addWidget(w)
        else:
            self.body.addLayout(w)
        return w


class Form(QGridLayout):
    """Label / field rows with an optional note column."""

    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(12)
        self.setVerticalSpacing(6)
        self.setColumnStretch(1, 1)
        self.n = 0

    def row(self, text, field, note=''):
        self.addWidget(label(text, muted=True), self.n, 0)
        if isinstance(field, QWidget):
            self.addWidget(field, self.n, 1)
        else:
            self.addLayout(field, self.n, 1)
        if note:
            self.addWidget(label(note, muted=True, small=True, wrap=True), self.n, 2)
        self.n += 1
        return field


class KeyValues(QWidget):
    """A grid of key: value labels that can be updated (values selectable)."""

    def __init__(self, keys=(), parent=None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(12)
        self.grid.setColumnStretch(1, 1)
        self.vals: dict = {}
        for k in keys:
            self.set(k, '—')

    def set(self, key, value, tip=''):
        if key not in self.vals:
            r = len(self.vals)
            self.grid.addWidget(label(key, muted=True), r, 0, Qt.AlignmentFlag.AlignTop)
            v = label('', sel=True, wrap=True)
            self.grid.addWidget(v, r, 1)
            self.vals[key] = v
        self.vals[key].setText(str(value))
        self.vals[key].setToolTip(tip)

    def clear(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.vals.clear()


class SliderField(QWidget):
    """A slider with a numeric box; valueChanged(float) while dragging, committed(float) on release or entry."""
    valueChanged = pyqtSignal(float)
    committed = pyqtSignal(float)

    def __init__(self, lo, hi, unit='', decimals=0, step=1.0, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.k = float(lo), float(hi), 10 ** decimals
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(lo * self.k), int(hi * self.k))
        self.box = QDoubleSpinBox()
        self.box.setRange(lo, hi)
        self.box.setDecimals(decimals)
        self.box.setSingleStep(step)
        self.box.setSuffix(f' {unit}' if unit else '')
        self.box.setFixedWidth(84)
        h.addWidget(self.slider, 1)
        h.addWidget(self.box)
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderReleased.connect(lambda: self.committed.emit(self.value()))
        self.box.editingFinished.connect(self._from_box)

    def value(self) -> float:
        return float(self.box.value())

    def setValue(self, v):
        v = min(max(float(v), self.lo), self.hi)
        for w in (self.slider, self.box):
            w.blockSignals(True)
        self.slider.setValue(int(round(v * self.k)))
        self.box.setValue(v)
        for w in (self.slider, self.box):
            w.blockSignals(False)

    def _from_slider(self, i):
        v = i / self.k
        self.box.blockSignals(True)
        self.box.setValue(v)
        self.box.blockSignals(False)
        self.valueChanged.emit(v)
        if not self.slider.isSliderDown():
            self.committed.emit(v)

    def _from_box(self):
        v = self.value()
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self.k)))
        self.slider.blockSignals(False)
        self.committed.emit(v)


def scroll(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(widget)
    return sa


class Page(QWidget):
    """A page of the main window. key, title and icon name it in the navigator; refresh() redraws it from the
    project (called when the project changes or another one is opened)."""
    key = ''
    title = ''
    icon = 'file'
    about = ''

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win

    def refresh(self):
        pass

    def page_layout(self, title_text=None, note=None) -> QVBoxLayout:
        """A scrolling page body with its heading; returns the layout to fill."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        body.setObjectName('base')
        v = QVBoxLayout(body)
        v.setContentsMargins(24, 18, 24, 18)
        v.setSpacing(10)
        v.addWidget(label(title_text or self.title, title=True))
        if note or self.about:
            v.addWidget(label(note or self.about, muted=True, wrap=True))
        v.addWidget(hline())
        outer.addWidget(scroll(body))
        return v
