"""Show me: a pulsing outline over one control of a page for a few seconds, and the stable names controls go by.

The assistant's "show me" mode points at a control instead of operating it. A control is addressed by its page key and
a name: the attribute path the page keeps it under (e.g. data: record.world, record.go_btn; scenarios: seed), or a
visible label — 'button:Record', 'check:Then derive synthetic array audio', 'tab:Browse'. controls(page) lists every
name a page has; find() resolves one, bringing it into view first (its page, the tab holding it, the scroll area
around it); highlight() draws the outline on an overlay above the window that lets every click through and removes
itself when the time is up.
"""
from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit,
                             QPlainTextEdit, QScrollArea, QSlider, QStackedWidget, QTabWidget, QTextEdit, QWidget)

INPUTS = (QAbstractButton, QComboBox, QAbstractSpinBox, QLineEdit, QSlider, QAbstractItemView, QTabWidget,
          QPlainTextEdit, QTextEdit)
PULSE_MS = 40


def _attr_controls(obj, prefix: str, out: dict, depth: int):
    for k, v in list(vars(obj).items()):
        if k.startswith('_') or k in ('win', 'page', 'parent_page'):
            continue
        if isinstance(v, INPUTS):
            out.setdefault(prefix + k, v)
        elif isinstance(v, QWidget) and depth > 0:
            _attr_controls(v, prefix + k + '.', out, depth - 1)


def controls(page) -> dict:
    """{name: widget} of a page: attribute paths (two levels: the page's own and its sections' or tabs'), then the
    visible labels of its buttons, check boxes and tabs."""
    out: dict = {}
    _attr_controls(page, '', out, 1)
    for b in page.findChildren(QAbstractButton):
        t = b.text().replace('&', '').strip()
        if t:
            out.setdefault(f'{"check" if isinstance(b, QCheckBox) else "button"}:{t}', b)
    for tw in page.findChildren(QTabWidget):
        for i in range(tw.count()):
            out.setdefault(f'tab:{tw.tabText(i).replace("&", "")}', (tw, i))
    return out


def section_of(w) -> str:
    """The title of the page section a control is in ('' outside one)."""
    from .widgets import Section
    p = w.parentWidget() if isinstance(w, QWidget) else None
    while p is not None:
        if isinstance(p, Section):
            return p.title
        p = p.parentWidget()
    return ''


def describe(name: str, w) -> str:
    if isinstance(w, tuple):
        return 'tab'
    kind = type(w).__name__.replace('Q', '', 1).lower()
    sec = section_of(w)
    tip = w.toolTip() if isinstance(w, QWidget) else ''
    return kind + (f' in "{sec}"' if sec else '') + (f': {tip[:100]}' if tip else '')


def bring_into_view(w: QWidget):
    """Make the tabs holding w current and scroll it into view."""
    child, p = w, w.parentWidget()
    while p is not None:
        if isinstance(p, QStackedWidget) and isinstance(p.parentWidget(), QTabWidget):
            p.parentWidget().setCurrentWidget(child)
        if isinstance(p, QScrollArea) and p.widget() is not None:
            p.ensureWidgetVisible(w, 40, 40)
        child, p = p, p.parentWidget()


def find(win, page_key: str, name: str):
    """The widget named name on page page_key, in view (its page shown); None when there is no such control. A tab
    name selects the tab and returns its tab bar's rectangle owner."""
    page = win.page(page_key)
    if page is None:
        return None
    win.go(page_key)
    c = controls(page)
    w = c.get(name)
    if w is None:
        low = name.lower()
        w = next((v for k, v in c.items() if k.lower() == low or k.split(':', 1)[-1].lower() == low), None)
    if w is None:
        return None
    if isinstance(w, tuple):
        tw, i = w
        bring_into_view(tw)
        tw.setCurrentIndex(i)
        return tw.tabBar()
    bring_into_view(w)
    return w


class Outline(QWidget):
    """A pulsing rounded outline over target, drawn on an overlay child of its window (clicks pass through)."""

    def __init__(self, target: QWidget, seconds: float = 4.0):
        win = target.window()
        super().__init__(win)
        self.target = target
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setObjectName('assistantHighlight')
        self.t = 0
        self.ticks = max(int(seconds * 1000 / PULSE_MS), 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._place()
        self.show()
        self.raise_()
        self.timer.start(PULSE_MS)

    def _place(self):
        tl = self.target.mapTo(self.target.window(), QPoint(0, 0))
        r = QRect(tl, self.target.size()).adjusted(-6, -6, 6, 6)
        self.setGeometry(r)

    def _tick(self):
        self.t += 1
        if self.t >= self.ticks or not self.target.isVisible():
            self.timer.stop()
            self.hide()
            self.deleteLater()
            return
        self._place()
        self.update()

    def paintEvent(self, _ev):
        import math
        from .theme import CAMERA, THEME
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(CAMERA[THEME.name][0])                    # the theme's first accent (blue)
        c.setAlphaF(0.45 + 0.55 * (0.5 + 0.5 * math.sin(self.t * PULSE_MS / 1000.0 * 2 * math.pi)))
        p.setPen(QPen(c, 3))
        p.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 6, 6)
        p.end()


def highlight(win, page_key: str, name: str, seconds: float = 4.0) -> Outline | None:
    w = find(win, page_key, name)
    if w is None:
        return None
    return Outline(w, seconds)
