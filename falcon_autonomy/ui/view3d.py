"""A simple 3D view of CARLA's car and a rig's sensors: orbit (drag empty space), zoom (wheel), drag a sensor to move
it in its horizontal plane (Shift: vertically).

The scene is drawn with QPainter in perspective. On a QOpenGLWidget QPainter paints through OpenGL; when OpenGL is not
available (no context, a remote display) the same scene is drawn by a plain QWidget in software (make_view chooses).
Coordinates are CARLA's vehicle frame (x forward, y right, z up, left-handed), shown as seen from above-left-behind.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from ..core.rig import VEHICLES, carla_rotation
from .theme import THEME


class SceneMixin:
    """The camera, picking and painting shared by the OpenGL and software views (their classes declare the signals:
    poseDragged(sensor id, pose) while dragging, poseDropped(sensor id, pose) on release, selected(sensor id))."""

    def init_scene(self):
        self.rig = None
        self.sel = ''
        self.az, self.el, self.dist = -135.0, 30.0, 12.0
        self.drag = None
        self.setMinimumSize(360, 300)
        self.setMouseTracking(False)

    def set_rig(self, rig, sel=''):
        self.rig, self.sel = rig, sel
        self.update()

    # ---- camera: a look-at from spherical angles around the car's centre
    def view_matrix(self):
        az, el = math.radians(self.az), math.radians(self.el)
        # the eye in CARLA axes (y right): az measured from +x towards +y
        eye = np.array([self.dist * math.cos(el) * math.cos(az), self.dist * math.cos(el) * math.sin(az),
                        self.dist * math.sin(el) + 0.8])
        target = np.array([0.0, 0.0, 0.8])
        f = target - eye
        f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        # CARLA is left-handed: screen right = up × forward (a right vector consistent with y right)
        r = np.cross(up, f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        return eye, r, u, f

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        eye, r, u, f = self.view_matrix()
        d = P - eye
        z = d @ f
        fpx = 0.9 * min(self.width(), self.height())
        x = self.width() / 2 + fpx * (d @ r) / np.maximum(z, 1e-3)
        y = self.height() / 2 - fpx * (d @ u) / np.maximum(z, 1e-3)
        return np.c_[x, y], z > 0.1

    def ray(self, px, py):
        eye, r, u, f = self.view_matrix()
        fpx = 0.9 * min(self.width(), self.height())
        d = f + r * (px - self.width() / 2) / fpx - u * (py - self.height() / 2) / fpx
        return eye, d / np.linalg.norm(d)

    # ---- mouse
    def mousePressEvent(self, ev):
        p = ev.position()
        hit = self.pick(p.x(), p.y())
        if hit:
            self.sel = hit
            self.selected.emit(hit)
            s = self.sensor(hit)
            self.drag = ('sensor', hit, dict(s['pose']), p)
        else:
            self.drag = ('orbit', None, None, p)
        self.update()

    def mouseMoveEvent(self, ev):
        if not self.drag:
            return
        kind, sid, pose0, p0 = self.drag
        p = ev.position()
        if kind == 'orbit':
            self.az -= (p.x() - p0.x()) * 0.4
            self.el = max(min(self.el + (p.y() - p0.y()) * 0.3, 89.0), -10.0)
            self.drag = ('orbit', None, None, p)
            self.update()
            return
        new = dict(pose0)
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            new['z'] = round(max(pose0['z'] - (p.y() - p0.y()) * self.dist / 600.0, 0.0), 3)
        else:
            o, d = self.ray(p.x(), p.y())
            if abs(d[2]) < 1e-6:
                return
            t = (pose0['z'] - o[2]) / d[2]
            if t <= 0:
                return
            hit = o + t * d
            new['x'], new['y'] = round(float(hit[0]), 3), round(float(hit[1]), 3)
        self.drag = (kind, sid, pose0, p0)
        self._preview = (sid, new)
        self.sensor(sid)['pose'].update(new)            # shown at once; the page commits it as an undoable edit
        self.poseDragged.emit(sid, new)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self.drag and self.drag[0] == 'sensor' and getattr(self, '_preview', None):
            sid, new = self._preview
            pose0 = self.drag[2]
            self.sensor(sid)['pose'].update(pose0)      # back to the document's value: the edit applies it
            self.poseDropped.emit(sid, new)
        self._preview = None
        self.drag = None

    def wheelEvent(self, ev):
        self.dist = min(max(self.dist * (0.9 if ev.angleDelta().y() > 0 else 1.1), 3.0), 60.0)
        self.update()

    def sensor(self, sid):
        return next(s for s in self.rig['sensors'] if s['id'] == sid)

    def pick(self, x, y):
        if not self.rig:
            return ''
        best, bd = '', 12.0
        for s in self.rig['sensors']:
            p = s['pose']
            uv, ok = self.project([[p['x'], p['y'], p['z']]])
            if ok[0]:
                d = math.hypot(uv[0, 0] - x, uv[0, 1] - y)
                if d < bd:
                    best, bd = s['id'], d
        return best

    # ---- painting
    def paint_scene(self, p: QPainter):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['viewport']))
        grid = QPen(QColor(THEME.tok['grid2']), 1)
        p.setPen(grid)
        for k in range(-10, 11):
            self.line(p, (k, -10, 0), (k, 10, 0))
            self.line(p, (-10, k, 0), (10, k, 0))
        if not self.rig:
            return
        L, W, H = VEHICLES.get(self.rig.get('vehicle'), (4.6, 2.0, 1.5))
        body = QPen(QColor(THEME.tok['body']), 1.5)
        p.setPen(body)
        xs, ys = (-L / 2, L / 2), (-W / 2, W / 2)
        corners = [(x, y, z) for x in xs for y in ys for z in (0.0, H)]
        for a in range(8):
            for b in range(a + 1, 8):
                if sum(ca != cb for ca, cb in zip(corners[a], corners[b])) == 1:
                    self.line(p, corners[a], corners[b])
        self.line(p, (L / 2, 0, H * 0.5), (L / 2 + 0.8, 0, H * 0.5))    # the front
        for i, s in enumerate(self.rig['sensors']):
            self.paint_sensor(p, s, THEME.cam(i), s['id'] == self.sel)
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(8, self.height() - 8, 'drag: orbit · drag a sensor: move it (Shift: height) · wheel: zoom')

    def paint_sensor(self, p, s, col, sel):
        pose = s['pose']
        o = np.array([pose['x'], pose['y'], pose['z']])
        R = carla_rotation(pose['roll'], pose['pitch'], pose['yaw'])
        p.setPen(QPen(col, 3.0 if sel else 1.5))
        t = s['type']
        if t == 'camera.fisheye':
            # a disc at the edge of its field of view (half-angle θ), 0.6 m out, and the optical axis
            th = math.radians(min((s.get('lens') or {}).get('fov', 190.0), 260.0) / 2.0)
            rr, ax = 0.6 * math.sin(th), 0.6 * math.cos(th)
            ring = [o + R @ np.array([ax, rr * math.cos(a), rr * math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 33)]
            for a, b in zip(ring, ring[1:]):
                self.line(p, a, b)
            self.line(p, o, o + R @ np.array([0.8, 0.0, 0.0]))
        elif t.startswith('camera.'):
            fov = s.get('fov', 90.0)
            aspect = s.get('height', 9) / max(s.get('width', 16), 1)
            h = math.tan(math.radians(fov) / 2.0)
            n = 0.8
            pts = [o + R @ np.array([n, sx * n * h, sz * n * h * aspect]) for sx, sz in ((-1, -1), (1, -1), (1, 1),
                                                                                         (-1, 1))]
            for a in pts:
                self.line(p, o, a)
            for a, b in zip(pts, pts[1:] + pts[:1]):
                self.line(p, a, b)
        elif t == 'lidar':
            ring = [o + R @ np.array([0.35 * math.cos(a), 0.35 * math.sin(a), 0.0])
                    for a in np.linspace(0, 2 * math.pi, 25)]
            for a, b in zip(ring, ring[1:]):
                self.line(p, a, b)
        else:
            self.line(p, o, o + R @ np.array([0.4, 0.0, 0.0]))
        uv, ok = self.project([o])
        if ok[0]:
            r = 5.0 if sel else 3.5
            p.setBrush(col)
            p.drawEllipse(QPointF(uv[0, 0], uv[0, 1]), r, r)
            p.setPen(QColor(THEME.tok['ink']))
            p.drawText(QPointF(uv[0, 0] + 8, uv[0, 1] - 6), s['name'])

    def line(self, p, a, b):
        uv, ok = self.project([a, b])
        if ok.all():
            p.drawLine(QPointF(*uv[0]), QPointF(*uv[1]))

    def paintEvent(self, ev):
        p = QPainter(self)
        self.paint_scene(p)
        p.end()


class SoftView(SceneMixin, QWidget):
    poseDragged = pyqtSignal(str, dict)
    poseDropped = pyqtSignal(str, dict)
    selected = pyqtSignal(str)
    backend = 'software'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_scene()


def make_view(parent=None):
    """The OpenGL view when an OpenGL context can be made (QPainter then paints through OpenGL), else software.
    FALCON_SOFTWARE_3D=1 forces software."""
    if os.environ.get('FALCON_SOFTWARE_3D') == '1' or os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        return SoftView(parent)
    try:
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget
        from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext
        ctx = QOpenGLContext()
        surf = QOffscreenSurface()
        surf.create()
        if not ctx.create() or not ctx.makeCurrent(surf):
            return SoftView(parent)
        ctx.doneCurrent()
    except Exception:                            # noqa: BLE001 (no OpenGL: the software view)
        return SoftView(parent)

    class GLView(SceneMixin, QOpenGLWidget):
        poseDragged = pyqtSignal(str, dict)
        poseDropped = pyqtSignal(str, dict)
        selected = pyqtSignal(str)
        backend = 'opengl'

        def __init__(self, parent=None):
            super().__init__(parent)
            self.init_scene()

        def paintGL(self):
            p = QPainter(self)
            self.paint_scene(p)
            p.end()

        def paintEvent(self, ev):
            QOpenGLWidget.paintEvent(self, ev)
    return GLView(parent)
