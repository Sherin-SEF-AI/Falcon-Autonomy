"""Small building blocks shared by the pages."""
from __future__ import annotations

import html
import re

from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QAbstractScrollArea, QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QLayout, QLineEdit, QListWidget, QPushButton,
                             QScrollArea, QSizePolicy, QSlider, QSplitter, QStyledItemDelegate, QTreeWidget,
                             QVBoxLayout, QWidget)

from ..core.paths import tilde


NOTE_AFTER = 200         # a muted, wrapped explanation longer than this opens as one sentence with *more*


def label(text='', *, muted=False, small=False, title=False, mono=False, h=False, wrap=False, sel=False,
          page=False, ink2=False) -> QLabel:
    if muted and wrap and not small and len(text) > NOTE_AFTER:
        return Note(text)
    lb = QLabel(text)
    for k, v in (('muted', muted), ('small', small), ('title', title), ('mono', mono), ('h', h), ('page', page),
                 ('ink2', ink2)):
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


_SENTENCE_END = re.compile(r'(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)(?<!\bcf)\.\s+(?=[A-Z(*"\'])')


def split_note(text: str, limit: int = 120) -> tuple[str, bool]:
    """The first sentence of a note (cut at a clause when the sentence itself is long), and whether more remains."""
    text = ' '.join(text.split())
    if len(text) <= limit:
        return text, False
    m = _SENTENCE_END.search(text)
    first = text[:m.start() + 1] if m else text
    if m and len(first) < 50:                        # a very short opener: take the next sentence too, if it fits
        m2 = _SENTENCE_END.search(text, m.end())
        two = text[:m2.start() + 1] if m2 else text
        if len(two) <= limit + 40:
            first = two
    if len(first) > limit + 40:
        cut = -1
        for sep in (' — ', '; ', ': ', ', '):
            k = first.find(sep, 40, limit + 20)
            if k > 0:
                cut = k
                break
        first = (first[:cut] if cut > 0 else first[:limit].rsplit(' ', 1)[0]).rstrip(' ,;:—') + ' …'
    return first, first != text


class Note(QLabel):
    """A page's or section's explanation: its first sentence, with *more* to read the rest in place — so a page
    opens with one line, and nothing it says (method, limits) is lost."""

    def __init__(self, text: str, limit: int = 120, parent=None):
        super().__init__(parent)
        self.limit = limit
        self.full = ' '.join(text.split())
        self.short, self.more = split_note(self.full, limit)
        self.expanded = False
        self.setProperty('muted', True)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                     Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.linkActivated.connect(lambda _href: self.toggle())
        from .theme import THEME
        THEME.changed.connect(self._themed)          # a bound slot: let go when the note is deleted (a lambda is not)
        self._render()

    def _themed(self, _name):
        self._render()

    def toggle(self):
        self.expanded = not self.expanded
        self._render()

    def setText(self, text):                         # noqa: N802 (Qt's name) — a later update is a note too
        self.full = ' '.join(str(text).split())
        self.short, self.more = split_note(self.full, self.limit)
        self._render()

    def plain(self) -> str:
        return self.full if self.expanded or not self.more else self.short

    def _render(self):
        if not self.more:
            self.setTextFormat(Qt.TextFormat.PlainText)
            QLabel.setText(self, self.full)
            return
        from .theme import THEME
        link = THEME.tok['ink2']
        body = html.escape(self.full if self.expanded else self.short)
        self.setTextFormat(Qt.TextFormat.RichText)
        QLabel.setText(self, f'{body} <a href="#more" style="color:{link}; text-decoration:none">'
                     f'{"less ▴" if self.expanded else "more ▾"}</a>')


class EmptyState(QWidget):
    """What to do when there is nothing to show yet: a message and, optionally, the action that fixes it."""

    def __init__(self, message: str, action: str = '', slot=None, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 24, 16, 24)
        v.addStretch(1)
        self.message = label(message, muted=True, wrap=True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.message)
        self.button = None
        if action:
            row = QHBoxLayout()
            row.addStretch(1)
            self.button = button(action, slot)
            row.addWidget(self.button)
            row.addStretch(1)
            v.addLayout(row)
        v.addStretch(1)

    def set_message(self, text: str):
        self.message.setText(text)


class _Placeholder:
    """Paints a muted line in an empty list's viewport (what goes here, and how to get some)."""
    placeholder = ''

    def setPlaceholderText(self, text):                 # noqa: N802 (Qt's naming)
        self.placeholder = text
        self.viewport().update()

    def paintEvent(self, ev):                           # noqa: N802
        super().paintEvent(ev)
        empty = (self.count() if isinstance(self, QListWidget) else self.topLevelItemCount()) == 0
        if empty and self.placeholder:
            from .theme import THEME
            p = QPainter(self.viewport())
            p.setPen(THEME.color('muted'))
            r = self.viewport().rect().adjusted(12, 8, -12, -8)
            p.drawText(r, int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self.placeholder)


class PlaceholderList(_Placeholder, QListWidget):
    def __init__(self, placeholder='', parent=None):
        QListWidget.__init__(self, parent)
        self.placeholder = placeholder


class PlaceholderTree(_Placeholder, QTreeWidget):
    def __init__(self, placeholder='', parent=None):
        QTreeWidget.__init__(self, parent)
        self.placeholder = placeholder


def set_enabled(w: QWidget, ok: bool, why: str = '', tip: str = ''):
    """Enable a control, or disable it saying why in its tooltip (a disabled control never leaves you guessing)."""
    w.setEnabled(bool(ok))
    if not hasattr(w, '_tip'):
        w._tip = tip or w.toolTip()
    w.setToolTip((tip or w._tip) if ok else why)


class Section(QFrame):
    """A titled card of a page: a heading, an optional note (one line, the rest behind *more*), then content. A
    section inside another is drawn as a divided block of its card, not a card of its own."""

    def __init__(self, title: str, note: str = '', parent=None):
        super().__init__(parent)
        self.title = title                   # show-me names a control by the section it is in (ui.highlight)
        self.setObjectName('section')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 13, 16, 15)
        v.setSpacing(8)
        v.addWidget(label(title, title=True))
        self.note = None
        if note:
            self.note = Note(note)
            v.addWidget(self.note)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        v.addLayout(self.body)

    def add(self, w):
        if isinstance(w, QWidget):
            self.body.addWidget(w)
        else:
            self.body.addLayout(w)
        return w


class Meter(QWidget):
    """A thin usage bar (used / total): accent while there is room, amber past 80 %, red past 95 %."""

    def __init__(self, width: int = 0, parent=None):
        super().__init__(parent)
        self.frac = 0.0
        self.setFixedHeight(6)
        if width:
            self.setFixedWidth(width)

    def set(self, frac):
        self.frac = 0.0 if frac is None else min(max(float(frac), 0.0), 1.0)
        self.update()

    def paintEvent(self, _ev):                                  # noqa: N802
        from .theme import STATUS, THEME
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        r = QRectF(self.rect()).adjusted(0, 1, 0, -1)
        p.setBrush(QColor(THEME.tok['raised'] if THEME.name == 'dark' else THEME.tok['hair']))
        p.drawRoundedRect(r, 2, 2)
        if self.frac > 0:
            c = STATUS['critical'] if self.frac > 0.95 else STATUS['warning'] if self.frac > 0.8 else THEME.tok['accent']
            p.setBrush(QColor(c))
            p.drawRoundedRect(QRectF(r.left(), r.top(), max(r.height(), r.width() * self.frac), r.height()), 2, 2)


class Stat(QFrame):
    """A headline number in a card: a small caption, the value in large mono with its unit, and a line under it
    (what it counts, or its interval). tone colours the line (a warning, say)."""

    def __init__(self, caption: str, value: str = '—', unit: str = '', sub: str = '', parent=None):
        super().__init__(parent)
        self.setObjectName('kpi')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 13)
        v.setSpacing(3)
        self.caption = label(caption.upper())
        self.caption.setProperty('cap', True)
        v.addWidget(self.caption)
        row = QHBoxLayout()
        row.setSpacing(5)
        self.value = label(value)
        self.value.setProperty('kpi', True)
        self.unit = label(unit)
        self.unit.setProperty('unit', True)
        row.addWidget(self.value)
        row.addWidget(self.unit, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        v.addLayout(row)
        self.sub = label(sub, muted=True, mono=True)
        self.sub.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # clips, never widens
        self.sub.setMinimumWidth(0)
        self.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.addWidget(self.sub)

    def set(self, value, unit: str = '', sub: str = '', tone: str = ''):
        from .theme import STATUS, THEME
        self.value.setText(str(value))
        self.unit.setText(unit)
        self.unit.setVisible(bool(unit))
        self.sub.setText(sub)
        self.sub.setToolTip(sub)
        self.sub.setStyleSheet(f'color: {THEME.status(tone)}' if tone in STATUS else '')


INTERVAL_ROLE = Qt.ItemDataRole.UserRole + 7     # (value, ci_low, ci_high) for IntervalDelegate


class IntervalDelegate(QStyledItemDelegate):
    """Draws a value and its interval on a shared axis (0 – 1 unless given): a band for the interval, a dot for the
    value and faint quarter ticks, so rows read against each other at a glance. Rows without data draw nothing."""

    def __init__(self, parent=None, lo: float = 0.0, hi: float = 1.0):
        super().__init__(parent)
        self.lo, self.hi = lo, hi

    def paint(self, p, opt, index):
        super().paint(p, opt, index)
        d = index.data(INTERVAL_ROLE)
        if not d or d[0] is None:
            return
        from .theme import THEME
        v, a, b = d
        a = v if a is None else a
        b = v if b is None else b
        r = QRectF(opt.rect).adjusted(10, 0, -10, 0)
        span = (self.hi - self.lo) or 1.0
        x = lambda t: r.left() + (min(max(t, self.lo), self.hi) - self.lo) / span * r.width()   # noqa: E731
        cy = r.center().y()
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(THEME.tok['hair']))
        for k in range(5):
            tx = r.left() + k / 4 * r.width()
            p.drawLine(QPointF(tx, cy - 6), QPointF(tx, cy + 6))
        p.setPen(Qt.PenStyle.NoPen)
        band = QColor(THEME.tok['accent'])
        band.setAlphaF(0.35)
        p.setBrush(band)
        p.drawRoundedRect(QRectF(x(a), cy - 4, max(2.0, x(b) - x(a)), 8), 3, 3)
        p.setBrush(QColor(THEME.tok['accent_hi']))
        p.drawEllipse(QPointF(x(v), cy), 4, 4)
        p.restore()


class FlowLayout(QLayout):
    """Children left to right, wrapping to the next line when the width runs out (chips in a narrow column, a row of buttons that wraps rather than widens its pane)."""

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


def flow_row(*widgets, spacing: int = 6) -> QWidget:
    """A row of buttons that wraps onto a second line in a narrow pane instead of widening it."""
    box = QWidget()
    f = FlowLayout(box, spacing)
    for wd in widgets:
        f.addWidget(wd)
    return box


def chip(text: str, tone: str = '') -> QLabel:
    """A small rounded tag (a state, a tag): tone good, warning, critical or accent colours it."""
    lb = QLabel(text)
    lb.setProperty('chip', True)
    if tone:
        lb.setProperty('tone', tone)
    return lb


def callout(text: str, tone: str = 'warning') -> QFrame:
    """A boxed one-line notice inside a card (a caveat that must not be missed)."""
    from .theme import STATUS, THEME
    f = QFrame()
    f.setObjectName('callout')
    f.setProperty('tone', tone)
    h = QHBoxLayout(f)
    h.setContentsMargins(12, 8, 12, 8)
    lb = label(text, wrap=True)
    if tone in STATUS:
        lb.setStyleSheet(f'color: {THEME.status(tone)}')
    h.addWidget(lb)
    f.label = lb
    return f


FIELD_MAX = 520          # a form's line edit, spin box or combo box grows no wider than this


class Form(QGridLayout):
    """Label / field rows with an optional note column; single-line fields stop at FIELD_MAX wide."""

    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(14)
        self.setVerticalSpacing(8)
        self.setColumnStretch(1, 1)
        self.n = 0
        self.rows: dict = {}             # id(field) -> the row's label and note widgets

    def row(self, text, field, note=''):
        lb = label(text, muted=True)
        self.addWidget(lb, self.n, 0)
        if isinstance(field, QWidget):
            if isinstance(field, (QLineEdit, QAbstractSpinBox, QComboBox)) and field.maximumWidth() > FIELD_MAX:
                field.setMaximumWidth(FIELD_MAX)
                box = QHBoxLayout()                      # left-aligned: it fills up to FIELD_MAX, the rest stays empty
                box.setContentsMargins(0, 0, 0, 0)
                box.addWidget(field, 100)
                box.addStretch(1)
                self.addLayout(box, self.n, 1)
            else:
                self.addWidget(field, self.n, 1)
        else:
            self.addLayout(field, self.n, 1)
        nb = None
        if note:
            nb = label(note, muted=True, small=True, wrap=True)
            self.addWidget(nb, self.n, 2)
        self.rows[id(field)] = (lb, nb)
        self.n += 1
        return field

    def set_row_visible(self, field, on: bool):
        """Show or hide a whole row (a field that does not apply to what is shown is not shown)."""
        lb, nb = self.rows.get(id(field), (None, None))
        for w in (lb, nb, field if isinstance(field, QWidget) else None):
            if w is not None:
                w.setVisible(on)



def clear_layout(layout):
    """Empty a layout, its nested layouts too, and take its widgets off screen at once: deleteLater alone waits for
    the event loop, and until then the stale widgets draw under the rows that replace them."""
    while layout.count():
        it = layout.takeAt(0)
        w = it.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif it.layout() is not None:
            clear_layout(it.layout())
            it.layout().deleteLater()

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
        clear_layout(self.grid)
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
        """A scrolling page body (the frame's page header above it names the page); returns the layout to fill."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        body.setObjectName('base')
        v = QVBoxLayout(body)
        v.setContentsMargins(8, 4, 8, 16)
        v.setSpacing(14)
        if title_text:                               # a sub-page's own heading (the frame's header names the page)
            v.addWidget(label(title_text, page=True))
        if note:
            self.header_note = note                  # the frame's header says it in place of the page's about
        outer.addWidget(scroll(body))
        return v


# ------------------------------------------------------------------------------------------------- layout fitting
LONG_CHECK = 44          # a check box label longer than this keeps its first words; the rest goes to its tooltip


def tidy(root: QWidget):
    """Make a page's controls fit narrow panes: combo boxes size to a short minimum (the full item in the tooltip)
    instead of their longest item, and a long check-box label keeps its words before the first parenthesis, the
    parenthetical moving to the tooltip. Called once per page when it is built."""
    for c in root.findChildren(QComboBox):
        if c.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow:
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(14)
            c.setProperty('tip0', c.toolTip())
            c.currentTextChanged.connect(lambda t, c=c: _combo_tip(c, t))
    for b in root.findChildren(QCheckBox):
        t = b.text()
        if len(t) > LONG_CHECK and ' (' in t:
            head, rest = t.split(' (', 1)
            if len(head) >= 8:
                b.setText(head)
                more = '(' + rest
                b.setToolTip((b.toolTip() + '\n' if b.toolTip() else '') + more)
    for s in root.findChildren(QSplitter):
        if s.orientation() == Qt.Orientation.Horizontal:
            for i in range(s.count()):
                w = s.widget(i)
                if not isinstance(w, (QAbstractScrollArea, QSplitter)) and w.minimumSizeHint().height() > TALL:
                    wrap_in_scroll(s, i)
        fit_splitter(s)


TALL = 440               # a side-by-side pane taller than this scrolls, so the window fits a 1280 × 720 screen


def wrap_in_scroll(split: QSplitter, i: int) -> QScrollArea:
    """Put a splitter's pane in a scroll area (keeping its place and size)."""
    w = split.widget(i)
    sizes = split.sizes()
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    split.replaceWidget(i, sa)
    sa.setWidget(w)
    w.show()
    split.setSizes(sizes)
    return sa


def _combo_tip(c: QComboBox, text: str):
    """A combo box too narrow for its item shows the item in its tooltip (a disabled one keeps saying why)."""
    if c.isEnabled():
        c.setToolTip(text if c.fontMetrics().horizontalAdvance(text) + 40 > c.width() else c.property('tip0') or '')


class _SplitFit(QObject):
    """On a splitter's first show (and when the window is resized), widen each side pane to what its scrolling
    content needs, taking the room from the widest pane, so a form is never cut off at a default size."""

    def __init__(self, split: QSplitter):
        super().__init__(split)
        self.split = split
        self._pending = False
        split.installEventFilter(self)

    def need(self, w: QWidget) -> int:
        sa = w if isinstance(w, QAbstractScrollArea) else next(
            (c for c in w.findChildren(QScrollArea) if c.parent() is w or c.parentWidget() is w), None)
        if isinstance(sa, QScrollArea) and sa.widget() is not None:
            inner = sa.widget()
            if not inner.property('fitWatched'):            # its content changes (a form swapped in): fit again
                inner.setProperty('fitWatched', True)
                inner.installEventFilter(self)
            return inner.minimumSizeHint().width() + sa.verticalScrollBar().sizeHint().width() + 4
        return w.minimumSizeHint().width()

    def fit(self):
        sp = self.split
        if sp.orientation() != Qt.Orientation.Horizontal or sp.count() < 2:
            return
        sizes = sp.sizes()
        total = sum(sizes)
        if total <= 0:
            return
        main = max(range(len(sizes)), key=lambda i: sizes[i])
        floor = int((0.4 if sp.count() == 2 else 0.33) * total)      # the main pane keeps at least this
        for i in range(len(sizes)):
            if i == main or not sp.widget(i).isVisible():
                continue
            want = min(self.need(sp.widget(i)), int(0.5 * total))
            if want > sizes[i]:
                give = min(want - sizes[i], max(0, sizes[main] - floor))
                sizes[i] += give
                sizes[main] -= give
        if sizes != sp.sizes():
            sp.setSizes(sizes)

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.Show, QEvent.Type.Resize) or (
                obj is not self.split and ev.type() == QEvent.Type.LayoutRequest):
            if not self._pending:
                self._pending = True
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, self._fit_safely)
        return False

    def _fit_safely(self):
        self._pending = False
        try:
            self.fit()
        except RuntimeError:                         # the splitter went with its page
            pass


def fit_splitter(split: QSplitter):
    if not getattr(split, '_fit', None):
        split._fit = _SplitFit(split)
    return split._fit


# ------------------------------------------------------------------------------------------- optional tabs (4.0)
def optional_tab(host, module: str, cls: str, title: str, owner=None):
    """A tab (or section) a later module adds to a page, if this build has it: falcon_autonomy.<module>.<cls>,
    imported only now and built as cls(owner) (cls() without an owner) — the page passes itself, so the tab reaches the
    window as owner.win. host is the page's QTabWidget (the widget becomes a tab titled `title`) or a page's column
    layout (it goes in a Section titled `title`, before the column's closing stretch). Returns the widget, or None —
    silently — when the module, the class or something they import is not in this build (ImportError,
    AttributeError): the page is then exactly as before."""
    import importlib

    from PyQt6.QtWidgets import QBoxLayout, QSpacerItem, QTabWidget
    try:
        kind = getattr(importlib.import_module(f'falcon_autonomy.{module}'), cls)
    except (ImportError, AttributeError):
        return None
    w = kind(owner) if owner is not None else kind()
    if isinstance(host, QTabWidget):
        host.addTab(w, title)
        return w
    sec = Section(title)
    sec.add(w)
    if isinstance(host, QBoxLayout):
        n = host.count()
        last = host.itemAt(n - 1) if n else None
        if last is not None and isinstance(last.spacerItem(), QSpacerItem):
            host.insertWidget(n - 1, sec)
        else:
            host.addWidget(sec)
    else:
        host.addWidget(sec)
    return w
