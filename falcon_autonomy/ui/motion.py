"""Motion: short animations that say where something went or what is alive — a page crossfade, the navigator's
selection sliding to the new page, a page's cards fading in the first time it opens, a pulsing dot while something
runs, a button's colour easing on hover. Each is 120–260 ms and never delays input: the new page is live while the old
one fades over it.

Off with Settings ▸ Appearance ▸ Reduce motion, or FALCON_REDUCE_MOTION=1 (the test suite sets it). Nothing here
puts an opacity effect on a widget holding an OpenGL view (a graphics effect on a QOpenGLWidget's parent breaks its
drawing): the crossfade fades a snapshot laid over the stack instead."""
from __future__ import annotations

import os

from PyQt6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, QRectF, Qt, QTimer, QVariantAnimation
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QLabel, QPushButton, QWidget

FAST, BASE, SLOW = 120, 200, 260
_enabled = os.environ.get('FALCON_REDUCE_MOTION', '').lower() not in ('1', 'true', 'yes')


def enabled() -> bool:
    return _enabled


def set_enabled(on: bool):
    global _enabled
    _enabled = bool(on)


def _has_gl(w: QWidget) -> bool:
    return w.inherits('QOpenGLWidget') or any(c.inherits('QOpenGLWidget') for c in w.findChildren(QWidget))


def crossfade(host: QWidget, old: QWidget | None, ms: int = BASE):
    """Lay a snapshot of the page being left over host and fade it out (call before switching)."""
    if not _enabled or old is None or not old.isVisible() or host.width() < 2:
        return None
    shot = QLabel(host)
    shot.setObjectName('motionSnapshot')
    shot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    shot.setPixmap(old.grab())
    shot.setGeometry(old.geometry())
    eff = QGraphicsOpacityEffect(shot)
    shot.setGraphicsEffect(eff)
    shot.show()
    shot.raise_()
    a = QPropertyAnimation(eff, b'opacity', shot)
    a.setDuration(ms)
    a.setStartValue(1.0)
    a.setEndValue(0.0)
    a.setEasingCurve(QEasingCurve.Type.OutCubic)
    a.finished.connect(shot.deleteLater)
    a.start()
    return a


def fade_in(w: QWidget, delay: int = 0, ms: int = SLOW):
    """Fade a widget in from transparent (skipped for one holding an OpenGL view); the effect is removed after."""
    if not _enabled or _has_gl(w) or w.graphicsEffect() is not None:
        return
    eff = QGraphicsOpacityEffect(w)
    eff.setOpacity(0.0)
    w.setGraphicsEffect(eff)
    a = QPropertyAnimation(eff, b'opacity', w)
    a.setDuration(ms)
    a.setStartValue(0.0)
    a.setEndValue(1.0)
    a.setEasingCurve(QEasingCurve.Type.OutCubic)

    def done():
        if w.graphicsEffect() is eff:
            w.setGraphicsEffect(None)
    a.finished.connect(done)
    QTimer.singleShot(delay, a.start)


def stagger(widgets, step: int = 40, ms: int = SLOW):
    """Fade widgets in one after another (a page's cards the first time it opens)."""
    for i, w in enumerate(widgets[:12]):
        fade_in(w, i * step, ms)


class Dot(QWidget):
    """A status dot; pulse() rings it while something is alive (a server starting, a job running)."""

    def __init__(self, color: str = '', d: int = 8, parent=None):
        super().__init__(parent)
        self.d = d
        self.c = QColor(color or '#717a89')
        self.t = 0.0
        self.setFixedSize(d * 2 + 2, d * 2 + 2)
        self.anim = QVariantAnimation(self)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setDuration(1400)
        self.anim.setLoopCount(-1)
        self.anim.valueChanged.connect(self._tick)

    def set_color(self, color: str, pulse: bool = False):
        self.c = QColor(color)
        self.pulse(pulse)
        self.update()

    def pulse(self, on: bool):
        if on and _enabled:
            if self.anim.state() != QVariantAnimation.State.Running:
                self.anim.start()
        else:
            self.anim.stop()
            self.t = 0.0
            self.update()

    def _tick(self, v):
        self.t = float(v)
        self.update()

    def paintEvent(self, _ev):                                  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        cx, cy, r = self.width() / 2, self.height() / 2, self.d / 2
        if self.t > 0:
            ring = QColor(self.c)
            ring.setAlphaF(0.45 * (1 - self.t))
            p.setBrush(ring)
            rr = r + self.t * r * 1.1
            p.drawEllipse(QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr))
        p.setBrush(self.c)
        p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))


class HoverEase(QObject):
    """Eases a push button's background between its resting and hover colours (installed on the application:
    a QSS :hover rule switches in one frame). Only plain and primary buttons; the colours come from the theme."""

    def __init__(self, app):
        super().__init__(app)
        self.anims: dict = {}
        app.installEventFilter(self)

    def colours(self, b: QPushButton):
        from .theme import THEME
        t = THEME.tok
        role = b.property('role')
        if role == 'primary':
            return t['accent'], t['accent_hi']
        if role in (None, ''):
            return t['raised'], t['sel']
        return None

    def eventFilter(self, obj, ev):                             # noqa: N802
        et = ev.type()
        if not isinstance(obj, QPushButton):
            return False
        if et == QEvent.Type.EnabledChange and id(obj) in self.anims:
            self.anims[id(obj)].stop()
            obj.setStyleSheet('')                               # a disabled button takes its QSS look at once
        elif et in (QEvent.Type.Enter, QEvent.Type.Leave) and _enabled and not obj.property('seg') \
                and not obj.isChecked() and (obj.isEnabled() or id(obj) in self.anims):
            cols = self.colours(obj)
            if cols:
                self._run(obj, *cols, et == QEvent.Type.Enter and obj.isEnabled())
        return False

    def _run(self, b: QPushButton, rest: str, hover: str, entering: bool):
        a = self.anims.get(id(b))
        if a is None:
            a = QVariantAnimation(b)
            a.setDuration(FAST)
            a.setEasingCurve(QEasingCurve.Type.OutCubic)
            a.valueChanged.connect(lambda c, b=b: self._paint(b, c))
            a.finished.connect(lambda b=b, a=a: b.setStyleSheet('') if a.endValue() == QColor(a.property('rest'))
                               else None)
            b.destroyed.connect(lambda *_x, k=id(b): self.anims.pop(k, None))
            self.anims[id(b)] = a
        start = a.currentValue() if a.state() == QVariantAnimation.State.Running else QColor(hover if not entering
                                                                                                else rest)
        a.stop()
        a.setProperty('rest', rest)
        a.setStartValue(QColor(start))
        a.setEndValue(QColor(hover if entering else rest))
        a.start()

    @staticmethod
    def _paint(b: QPushButton, c):
        if isinstance(c, QColor) and c.isValid():
            b.setStyleSheet(f'QPushButton:!pressed {{ background: {c.name()}; }}')   # pressed keeps its QSS


_hover = None


def reset_hover(b: QPushButton):
    """Drop a button's hover tint at once (its role changed under the pointer: Send became Stop)."""
    if _hover is not None and id(b) in _hover.anims:
        _hover.anims[id(b)].stop()
    b.setStyleSheet('')


def install(app):
    """Application-wide motion (the hover easing); idempotent."""
    global _hover
    if _hover is None:
        _hover = HoverEase(app)
    return _hover
