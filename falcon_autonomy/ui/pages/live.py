"""Live: drive a world in real time — a chase, free or sensor view with ground-truth boxes, a top-down map, driving by
keyboard, game controller or wheel, by CARLA's autopilot or by a policy; live events; recording.

Every number shown is measured: simulation steps per second and the real-time factor by the worker, frames shown per
second and dropped frames (by the worker when the GUI had not taken a frame yet, and here when a newer frame arrived
before the last one was drawn) by the GUI."""
from __future__ import annotations

import math
import os
import time
import uuid

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtNetwork import QLocalServer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QMessageBox,
                             QPushButton, QSplitter, QVBoxLayout, QWidget)

from ...core import weather as W
from ...core.drive import run_dir_name
from ...core.jobs import FINAL, RUNNING
from ...core.live import live_config
from ...core.livelink import Reader
from ...core.paths import tilde
from ...core.project import ProjectError
from .. import input_devices
from ..theme import STATUS, THEME
from ..widgets import Form, KeyValues, Note, Page, Section, button, label, scroll, set_enabled

VIEW_SIZES = {'1280 × 720': (1280, 720), '960 × 540': (960, 540), '640 × 360': (640, 360)}
KIND_COLOR = {'v': '#3987e5', 'w': '#d95926', 'e': '#199e70'}


class LiveView(QWidget):
    """The view image with the box overlay and a HUD; takes the keyboard."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.img = None
        self.hdr = {}
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(480, 270)
        self.keys = set()
        self.drag = None

    def set_frame(self, img: QImage, hdr: dict):
        self.img, self.hdr = img, hdr
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['viewport']))
        if self.img is None:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       'Start a session to see the world.\nClick here to drive with the keyboard: W/S or ↑/↓ '
                       'throttle and brake, A/D or ←/→ steer, Space hand brake, Q reverse.\nFree view: I/K/J/L move, '
                       'U/O up and down, drag to look around.')
            return
        iw, ih = self.img.width(), self.img.height()
        k = min(self.width() / iw, self.height() / ih)
        r = QRectF((self.width() - iw * k) / 2, (self.height() - ih * k) / 2, iw * k, ih * k)
        p.drawImage(r, self.img)
        h = self.hdr
        if self.page.overlay.isChecked():
            p.setFont(QFont('IBM Plex Mono', 8))
            for aid, kind, x0, y0, x1, y1 in h.get('boxes') or []:
                c = QColor(KIND_COLOR.get(kind, '#ffffff'))
                p.setPen(QPen(c, 1.5))
                p.drawRect(QRectF(r.x() + x0 * k, r.y() + y0 * k, (x1 - x0) * k, (y1 - y0) * k))
        p.setFont(QFont('IBM Plex Mono', 9))
        lines = [f'{h.get("speed_kmh", 0):5.1f} km/h   {h.get("mode", "")}   view {h.get("view", "")}',
                 f'sim {h.get("ticks_per_s", 0):4.1f} steps/s   ×{h.get("rtf", 0):.2f} real time']
        if h.get('failures'):
            lines.append('sensor failure: ' + ', '.join(f'{k} {v}' for k, v in h['failures'].items()))
        if h.get('recording'):
            lines.append('● recording')
        odd = h.get('odd')
        if odd and not odd.get('in', True):                    # the runtime ODD monitor (workers.oddwatch)
            lines.append('▲ outside the ODD: ' + '; '.join(odd.get('why') or [])[:44])
        p.fillRect(QRectF(r.x() + 8, r.y() + 8, 330, 18 * len(lines) + 8), QColor(0, 0, 0, 140))
        p.setPen(QColor('#ffffff'))
        for i, t in enumerate(lines):
            if t.startswith('●'):
                p.setPen(QColor(STATUS['critical']))
            elif t.startswith('▲'):
                p.setPen(QColor(STATUS['warning']))
            p.drawText(QPointF(r.x() + 14, r.y() + 24 + 18 * i), t)

    def keyPressEvent(self, ev):
        if not ev.isAutoRepeat():
            self.keys.add(ev.key())
        if ev.key() == Qt.Key.Key_Q and not ev.isAutoRepeat():
            self.page.reverse = not self.page.reverse

    def keyReleaseEvent(self, ev):
        if not ev.isAutoRepeat():
            self.keys.discard(ev.key())

    def focusOutEvent(self, ev):
        self.keys.clear()                       # never keep "throttle held" when focus leaves
        super().focusOutEvent(ev)

    def mousePressEvent(self, ev):
        self.setFocus()
        self.drag = ev.position()

    def mouseMoveEvent(self, ev):
        if self.drag is not None:
            d = ev.position() - self.drag
            self.drag = ev.position()
            self.page.free_turn(d.x() * 0.2, -d.y() * 0.2)

    def mouseReleaseEvent(self, ev):
        self.drag = None


class TopDown(QWidget):
    """The road network (lane points) and the actors around the ego, north up, centred on the ego."""

    def __init__(self):
        super().__init__()
        self.map = None
        self.actors = []
        self.scale = 2.0                       # px per metre
        self.setMinimumSize(240, 240)

    def set_map(self, m):
        self.map = m
        self.update()

    def set_actors(self, actors):
        self.actors = actors or []
        self.update()

    def wheelEvent(self, ev):
        self.scale = min(max(self.scale * (1.15 if ev.angleDelta().y() > 0 else 1 / 1.15), 0.05), 20.0)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['viewport']))
        ego = next((a for a in self.actors if a[1] == 'e'), None)
        cx, cy = (ego[2], ego[3]) if ego else (0.0, 0.0)
        s = self.scale
        ox, oy = self.width() / 2, self.height() / 2

        def to(x, y):                          # CARLA: x east-ish, y south-ish (left-handed): draw y down
            return QPointF(ox + (x - cx) * s, oy + (y - cy) * s)
        if self.map:
            p.setPen(QPen(QColor(THEME.tok['strong']), max(1.0, 3.0 * s * 0.25)))
            half = self.map.get('step_m', 3.0) / 2
            w, h = self.width() / s / 2 + 5, self.height() / s / 2 + 5
            for x, y, yaw in zip(self.map['x'], self.map['y'], self.map['yaw']):
                if abs(x - cx) > w or abs(y - cy) > h:
                    continue
                a = math.radians(yaw)
                dx, dy = math.cos(a) * half, math.sin(a) * half
                p.drawLine(to(x - dx, y - dy), to(x + dx, y + dy))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for aid, kind, x, y, yaw, hl, hw in self.actors:
            a = math.radians(yaw)
            c, sn = math.cos(a), math.sin(a)
            pts = [to(x + c * u - sn * v, y + sn * u + c * v) for u, v in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            col = QColor(KIND_COLOR.get(kind, '#ffffff'))
            p.setPen(QPen(col, 1))
            p.setBrush(col)
            p.drawPolygon(QPolygonF(pts))
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(8, self.height() - 8, f'{10 / s:.0f} m = 10 px · wheel to zoom' if s < 1 else
                   f'{s:.1f} px/m · wheel to zoom')


class LivePage(Page):
    key, title, icon = 'live', 'Live', 'play'
    about = 'Drive a world in real time; trigger events; record what happens as a labelled dataset.'

    def __init__(self, win):
        super().__init__(win)
        self.job = None
        self.server = None
        self.conn = None
        self.reader = Reader()
        self.pending = None                    # the newest undrawn frame
        self.reverse = False
        self.joy = None
        self.shown = 0
        self.display_drops = 0
        self.shown_times = []
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QSplitter(Qt.Orientation.Vertical)
        self.view = LiveView(self)
        self.top = TopDown()
        left.addWidget(self.view)
        left.addWidget(self.top)
        left.setSizes([600, 260])
        split.addWidget(left)
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(12, 12, 12, 12)
        pv.addWidget(label(self.title, page=True))
        pv.addWidget(Note(self.about, limit=160))

        ses = Section('Session')
        f = Form()
        self.world = QComboBox()
        self.rig = QComboBox()
        self.size = QComboBox()
        self.size.addItems(list(VIEW_SIZES))
        self.fps = QDoubleSpinBox()
        self.fps.setRange(5, 60)
        self.fps.setValue(20)
        self.fps.setSuffix(' Hz')
        f.row('World', self.world)
        f.row('Rig', self.rig)
        f.row('View size', self.size)
        f.row('Step', self.fps, 'paced to the wall clock')
        ses.add(f)
        b = QHBoxLayout()
        self.start_btn = button('Start', self.start, role='primary')
        self.stop_btn = button('Stop', self.stop)
        self.rec_btn = button('● Record', self.toggle_record,
                              tip='Record the session as a run (Data ▸ Browse opens it)')
        self.rec_btn.setCheckable(True)
        b.addWidget(self.start_btn)
        b.addWidget(self.stop_btn)
        b.addWidget(self.rec_btn)
        b.addStretch(1)
        ses.add(b)
        pv.addWidget(ses)

        drv = Section('Driving and view')
        g = QGridLayout()
        self.mode_btns = {}
        for i, (m, text) in enumerate((('manual', 'Manual'), ('autopilot', 'Autopilot'), ('policy', 'Policy'))):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setProperty('seg', True)
            btn.setProperty('segpos', 'first' if i == 0 else 'last' if i == 2 else 'mid')
            btn.clicked.connect(lambda _=False, mm=m: self.set_mode(mm))
            g.addWidget(btn, 0, i)
            self.mode_btns[m] = btn
        self.mode_btns['autopilot'].setChecked(True)
        drv.add(g)
        drv.add(label('Policy: the built-in lane follower (pure pursuit) — a baseline that ignores lights and cross '
                      'traffic.', muted=True, small=True, wrap=True))
        vf = Form()
        self.view_kind = QComboBox()
        self.view_kind.addItem('Chase', ('chase', None))
        self.view_kind.addItem('Free', ('free', None))
        self.view_kind.activated.connect(lambda _: self.send_view())
        vf.row('View', self.view_kind)
        self.overlay = QCheckBox('Ground-truth boxes (full 3D boxes projected; occlusion not considered)')
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(lambda on: self.send({'cmd': 'overlay', 'boxes': on}))
        vf.row('', self.overlay)
        self.device = QComboBox()
        vf.row('Input', self.device)
        self.axes = {}
        ax = QHBoxLayout()
        ax.setContentsMargins(0, 0, 0, 0)
        for name, default in (('steer', 0), ('throttle', 1), ('brake', 2)):
            cb = QComboBox()
            cb.addItems([str(i) for i in range(8)])
            cb.setCurrentIndex(default)
            cb.setToolTip(f'The input device\'s axis for {name}')
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            self.axes[name] = cb
            ax.addWidget(label(name.title(), muted=True))
            ax.addWidget(cb, 1)
        self.pedals = QCheckBox('Pedals rest at −1 (wheels)')
        vf.row('Axes', ax)
        vf.row('', self.pedals)
        self.device.currentIndexChanged.connect(lambda _: self.open_device())
        drv.add(vf)
        pv.addWidget(drv)

        evs = Section('Live events')
        eg = QGridLayout()
        eg.addWidget(button('Pedestrian steps out', lambda: self.fire_event('pedestrian', distance_m=25.0, side='right')),
                     0, 0)
        eg.addWidget(button('Car cuts in', lambda: self.fire_event('cut_in', side='left')), 0, 1)
        self.weather = QComboBox()
        self.weather.addItems(W.PRESETS)
        eg.addWidget(self.weather, 1, 0)
        eg.addWidget(button('Change weather', lambda: self.fire_event('weather', fields=W.preset_values(
            self.weather.currentText()))), 1, 1)
        self.fail_sensor = QComboBox()
        self.fail_mode = QComboBox()
        self.fail_mode.addItems(['black', 'freeze', 'drop'])
        eg.addWidget(self.fail_sensor, 2, 0)
        eg.addWidget(self.fail_mode, 2, 1)
        eg.addWidget(button('Fail sensor for 3 s', lambda: self.fire_event('sensor_fail', sensor=self.fail_sensor.currentData(),
                                                                      mode=self.fail_mode.currentText(), seconds=3.0)),
                     3, 0, 1, 2)
        evs.add(eg)
        pv.addWidget(evs)

        rec = Section('Recording', 'Records the rig\'s cameras with their labels, the events and the controls as a '
                                   'run of its own (Data ▸ Browse opens it).')
        self.rec_list = label('', muted=True, wrap=True, small=True)
        rec.add(self.rec_list)
        pv.addWidget(rec)

        st = Section('Measured')
        self.stats = KeyValues(('Simulation', 'Real-time factor', 'Shown', 'Dropped', 'Session'))
        st.add(self.stats)
        pv.addWidget(st)
        pv.addStretch(1)
        split.addWidget(scroll(panel))
        split.setSizes([1000, 380])
        self.ctrl_timer = QTimer(self)
        self.ctrl_timer.timeout.connect(self.control_tick)
        self.ctrl_timer.start(33)
        win.bridge.changed.connect(self.on_job)
        win.bridge.finished.connect(self.on_job)
        self.refresh()
        self.update_buttons()

    # ------------------------------------------------------------------ project
    def choose_world(self, world_id):
        """Choose this world (Worlds' *Drive this world*); it applies when the session starts."""
        self.refresh()
        k = self.world.findData(world_id)
        if k >= 0 and self.world.isEnabled():
            self.world.setCurrentIndex(k)

    def refresh(self):
        doc = self.win.project.doc
        for combo, items in ((self.world, doc['worlds']), (self.rig, doc['rigs'])):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for it in items:
                combo.addItem(it['name'], it['id'])
            k = combo.findData(cur)
            combo.setCurrentIndex(k if k >= 0 else 0)
            combo.blockSignals(False)
        rig = self.win.project.item('rigs', self.rig.currentData() or '') or {'sensors': []}
        while self.view_kind.count() > 2:
            self.view_kind.removeItem(2)
        self.fail_sensor.clear()
        self.fail_sensor.addItem('the view', 'view')
        for s in rig['sensors']:
            if s['type'] == 'camera.pinhole':
                self.view_kind.addItem(f'Sensor: {s["name"]}', ('sensor', s['id']))
                self.fail_sensor.addItem(s['name'], s['id'])
        self.device.blockSignals(True)
        self.device.clear()
        self.device.addItem('Keyboard', '')
        for path, name in input_devices.devices():
            self.device.addItem(f'{name} ({path})', path)
        self.device.blockSignals(False)

    # ------------------------------------------------------------------ session
    @property
    def running(self) -> bool:
        return self.job is not None and self.job.state not in FINAL

    def start(self):
        if self.running:
            return
        w = self.win
        self.close_link()
        self.server = QLocalServer(self)
        path = os.path.join('/tmp', f'falcon-live-{os.getuid()}-{uuid.uuid4().hex[:8]}.sock')
        QLocalServer.removeServer(path)
        if not self.server.listen(path):
            QMessageBox.warning(self, 'Live', f'Cannot open the live link: {self.server.errorString()}')
            return
        self.server.newConnection.connect(self.on_connection)
        width, height = VIEW_SIZES[self.size.currentText()]
        kind, sensor = self.view_kind.currentData()
        mode = next(m for m, b in self.mode_btns.items() if b.isChecked())
        try:
            cfg = live_config(w.project, self.world.currentData(), self.rig.currentData(),
                              os.path.join(w.service.runs_dir(), run_dir_name('live')), w.service.port,
                              w.service.tm_port, path, w.service.runs_dir(), fps=self.fps.value(), mode=mode,
                              view={'kind': kind, 'sensor': sensor, 'width': width, 'height': height, 'fov': 90.0},
                              workspace=w.workspace(), trigger_rules='project')
            self.job = w.service.submit_live(cfg)
        except (ProjectError, RuntimeError, ValueError) as e:
            QMessageBox.warning(self, 'Live', str(e))
            self.close_link()
            return
        self.shown = self.display_drops = 0
        self.shown_times = []
        self.recordings = []
        self.reverse = False
        self.view.setFocus()
        self.update_buttons()

    def stop(self):
        if self.running:
            if not self.win.queue.send(self.job, {'cmd': 'stop'}):
                self.win.queue.cancel(self.job)

    def send(self, cmd: dict):
        if self.running and self.job.state == RUNNING:
            self.win.queue.send(self.job, cmd)

    def on_job(self, job):
        if job is not self.job:
            return
        for m in job.other:
            if m.get('t') == 'recording' and m['path'] not in getattr(self, 'recordings', []):
                self.recordings.append(m['path'])
                if self.win.service.catalog is not None:
                    self.win.service.catalog.add_run(m['path'])
                self.rec_list.setText('\n'.join(f'{tilde(p)}' for p in self.recordings))
        if job.state in FINAL:
            self.close_link()
            self.rec_btn.setChecked(False)
            self.stats.set('Session', f'{job.state}: {job.summary()}')
        elif job.state != RUNNING:
            self.stats.set('Session', job.summary())
        self.update_buttons()

    def update_buttons(self):
        r = self.running
        set_enabled(self.start_btn, not r, 'A session is running: stop it first')
        set_enabled(self.stop_btn, r, 'No session is running')
        set_enabled(self.rec_btn, r, 'Start a session first: recording captures a running session')
        for wd in (self.world, self.rig, self.size, self.fps):
            set_enabled(wd, not r, 'Fixed while a session runs: stop it to change')

    # ------------------------------------------------------------------ the link
    def on_connection(self):
        self.conn = self.server.nextPendingConnection()
        self.conn.readyRead.connect(self.on_ready)

    def on_ready(self):
        try:
            msgs = self.reader.feed(bytes(self.conn.readAll()))
        except ValueError as e:
            self.win.log.append_line(f'live link: {e}')
            self.close_link()
            return
        for hdr, payload in msgs:
            t = hdr.get('t')
            if t == 'map':
                self.top.set_map(hdr)
            elif t == 'frame':
                if self.pending is not None:
                    self.display_drops += 1
                self.pending = (hdr, payload)
            elif t == 'bye':
                self.win.log.append_line(f'live: {hdr.get("reason", "")}')
        if self.pending is not None:
            QTimer.singleShot(0, self.draw_pending)

    def draw_pending(self):
        if self.pending is None:
            return
        hdr, payload = self.pending
        self.pending = None
        img = QImage.fromData(payload, 'JPG') if payload else None
        if img is not None and not img.isNull():
            self.view.set_frame(img, hdr)
        else:
            self.view.hdr = hdr
            self.view.update()
        self.top.set_actors(hdr.get('actors'))
        now = time.monotonic()
        self.shown += 1
        self.shown_times.append(now)
        while self.shown_times and now - self.shown_times[0] > 2.0:
            self.shown_times.pop(0)
        fps = (len(self.shown_times) - 1) / max(self.shown_times[-1] - self.shown_times[0], 1e-6) \
            if len(self.shown_times) > 1 else 0.0
        self.stats.set('Simulation', f'{hdr.get("ticks_per_s", 0):.1f} steps/s (measured by the worker)')
        self.stats.set('Real-time factor', f'×{hdr.get("rtf", 0):.2f} (simulated s per wall s, last 2 s)')
        self.stats.set('Shown', f'{fps:.1f} frames/s here ({self.shown} frames)')
        self.stats.set('Dropped', f'{hdr.get("dropped", 0)} by the worker (not yet taken), {self.display_drops} '
                                  'here (newer frame arrived first); no simulation step is ever skipped')
        self.stats.set('Session', f'step frame {hdr.get("frame")}, {hdr.get("sim_t", 0):.1f} s simulated')
        odd = hdr.get('odd')
        if odd is not None:
            self.stats.set('ODD', 'inside' if odd.get('in') else 'OUTSIDE: ' + '; '.join(odd.get('why') or []),
                           'the runtime ODD monitor (docs/odd-monitoring.md)')
        if hdr.get('physics'):
            self.stats.set('Sensor physics', ', '.join(hdr['physics']), 'docs/sensor-physics.md')

    def close_link(self):
        if self.conn is not None:
            self.conn.abort()
            self.conn = None
        if self.server is not None:
            self.server.close()
            self.server.deleteLater()
            self.server = None
        self.reader = Reader()
        self.pending = None

    # ------------------------------------------------------------------ controls
    def set_mode(self, mode):
        for m, b in self.mode_btns.items():
            b.setChecked(m == mode)
        self.send({'cmd': 'mode', 'mode': mode})
        if mode == 'manual':
            self.view.setFocus()

    def send_view(self):
        kind, sensor = self.view_kind.currentData()
        self.send({'cmd': 'view', 'kind': kind, 'sensor': sensor})

    def fire_event(self, kind, **kw):
        self.send(dict({'cmd': 'event', 'kind': kind}, **kw))

    def toggle_record(self):
        self.send({'cmd': 'record', 'on': self.rec_btn.isChecked()})

    def open_device(self):
        if self.joy is not None:
            self.joy.close()
            self.joy = None
        path = self.device.currentData()
        if path:
            try:
                self.joy = input_devices.Joystick(path, self)
            except OSError as e:
                QMessageBox.warning(self, 'Input', f'Cannot open {path}: {e}')
                self.device.setCurrentIndex(0)

    def free_turn(self, dyaw, dpitch):
        if self.view_kind.currentData()[0] == 'free':
            self.send({'cmd': 'free_camera', 'turn': [dyaw, dpitch]})

    def control_tick(self):
        if not self.running or not self.mode_btns['manual'].isChecked():
            self._free_keys()
            return
        k = self.view.keys
        if self.joy is not None:
            steer = self.joy.axis(self.axes['steer'].currentIndex())
            pedal = self.pedals.isChecked()
            throttle = max(self.joy.axis(self.axes['throttle'].currentIndex(), invert=not pedal, pedal=pedal), 0.0)
            brake = max(self.joy.axis(self.axes['brake'].currentIndex(), invert=not pedal, pedal=pedal), 0.0)
        else:
            up = Qt.Key.Key_W in k or Qt.Key.Key_Up in k
            down = Qt.Key.Key_S in k or Qt.Key.Key_Down in k
            left = Qt.Key.Key_A in k or Qt.Key.Key_Left in k
            right = Qt.Key.Key_D in k or Qt.Key.Key_Right in k
            throttle, brake = (0.7 if up else 0.0), (1.0 if down else 0.0)
            target = (-0.5 if left else 0.0) + (0.5 if right else 0.0)
            cur = getattr(self, '_steer', 0.0)
            self._steer = cur + max(min(target - cur, 0.08), -0.08)     # a ramp, as a keyboard has no analogue
            steer = self._steer
        self.send({'cmd': 'control', 'throttle': throttle, 'steer': steer, 'brake': brake,
                   'hand_brake': Qt.Key.Key_Space in k, 'reverse': self.reverse})
        self._free_keys()

    def _free_keys(self):
        if not self.running or self.view_kind.currentData()[0] != 'free':
            return
        k = self.view.keys
        mv = [(Qt.Key.Key_I in k) - (Qt.Key.Key_K in k), (Qt.Key.Key_L in k) - (Qt.Key.Key_J in k),
              (Qt.Key.Key_U in k) - (Qt.Key.Key_O in k)]
        if any(mv):
            self.send({'cmd': 'free_camera', 'move': [0.5 * m for m in mv]})

    def shutdown(self):
        self.ctrl_timer.stop()
        if self.joy is not None:
            self.joy.close()
        self.close_link()
