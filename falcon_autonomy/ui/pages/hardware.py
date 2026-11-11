"""Hardware: a device in the loop — a Jetson on the network, or a simulated device on this machine. Set up the device
(its address, SSH user and key, kept in your settings only, never in a project; no password is stored: SSH keys
only), copy the device agent to it and check what it has, start a session (a job like any other) and watch the
latency waterfall and the deadlines as it runs; the report when it ends. docs/hardware.md is the guide.

Every number shown is measured by the session (core.hil): the waterfall is the median frame, stage by stage, over
the last two seconds while running and over the whole session at the end; one-way network stages are drawn hatched
when the clock-offset estimate is not trusted.

The 4.0 interfaces: the camera into a V4L2 device on this machine (hil.v4l2out), the CAN loop with the chassis model
(hil.canbridge, core.chassis: an interface Falcon may open — vcan, a simulated link, or a real adapter only once
confirmed as a bench network — its live counters while running) and the clock source (core.ptp: PTP, the NTP-style
estimate, or whichever is usable; both are recorded)."""
from __future__ import annotations

import os

from PyQt6.QtCore import QRectF, Qt, QTimer, QUrl
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                             QInputDialog, QLineEdit, QMessageBox, QPlainTextEdit, QRadioButton, QSpinBox, QSplitter,
                             QVBoxLayout, QWidget)

from ...core.drive import run_dir_name
from ...core.hil import DEVICE_DEFAULTS, device_settings, hil_config, submit_session
from ...core.jobs import FINAL, RUNNING
from ...core.paths import tilde
from ...core.project import ProjectError
from .. import tasks
from ..theme import THEME
from ..widgets import Form, KeyValues, Page, Section, button, label, scroll, set_enabled

SIZES = {'1280 × 720': (1280, 720), '960 × 540': (960, 540), '640 × 360': (640, 360), '320 × 180': (320, 180)}
PLACE_SLOT = {'host': 0, 'cross': 1, 'device': 2}           # the validated categorical palette's first three slots
PLACE_NAME = {'host': 'this machine', 'cross': 'network (one-way)', 'device': 'the device'}


class Waterfall(QWidget):
    """The latency waterfall: one row per stage, each bar starting where the previous stage's median ended, a whisker
    to its p95; the end-to-end row with its p95 and p99; dashed lines at whole steps."""

    def __init__(self):
        super().__init__()
        self.rows, self.e2e, self.period, self.trusted, self.note = [], {}, 50.0, True, ''
        self.setMinimumSize(560, 380)

    def set_data(self, rows, e2e, period_ms, trusted, note=''):
        self.rows, self.e2e, self.period, self.trusted, self.note = list(rows or []), dict(e2e or {}), \
            float(period_ms or 50.0), bool(trusted), note
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        tok = THEME.tok
        p.fillRect(self.rect(), QColor(tok['viewport']))
        if not self.rows:
            p.setPen(QColor(tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Start a session to see its latency, stage by stage.')
            return
        W, H = self.width(), self.height()
        lw, rw, top = 210, 150, 34
        n = len(self.rows) + 1
        rh = max(min((H - top - 40) / n, 30.0), 14.0)
        ends = [r['start_ms'] + max(r.get('p95_ms') or 0, r['median_ms']) for r in self.rows] + \
            [self.e2e.get('p99') or 0, 1.0]
        span = max(ends) * 1.05
        pw = W - lw - rw
        X = lambda ms: lw + pw * ms / span                    # noqa: E731
        small = QFont('IBM Plex Sans', 8)
        p.setFont(small)
        x = 12.0                                              # the key: where each stage runs
        for k in ('host', 'cross', 'device'):
            p.fillRect(QRectF(x, 10, 10, 10), THEME.cam(PLACE_SLOT[k]))
            p.setPen(QColor(tok['ink2']))
            p.drawText(QRectF(x + 14, 6, 140, 18), Qt.AlignmentFlag.AlignVCenter, PLACE_NAME[k])
            x += 30 + p.fontMetrics().horizontalAdvance(PLACE_NAME[k])
        pen = QPen(QColor(tok['muted']), 1, Qt.PenStyle.DashLine)
        k = 1
        while k * self.period <= span:
            xs = X(k * self.period)
            p.setPen(pen)
            p.drawLine(int(xs), top - 4, int(xs), int(top + rh * n))
            p.setPen(QColor(tok['muted']))
            p.drawText(QRectF(xs + 3, top - 16, 60, 14), Qt.AlignmentFlag.AlignLeft, f'step {k}')
            k += 1
        for i, r in enumerate(self.rows + [None]):
            y = top + i * rh
            if r is None:
                if self.e2e.get('median') is None:
                    break
                lab, x0, x1, col = 'End to end', X(0), X(self.e2e['median']), QColor(tok['ink2'])
                val = f'{self.e2e["median"]:.1f} · p95 {self.e2e.get("p95", 0):.1f} ms'
            else:
                lab, x0, x1 = r['label'], X(r['start_ms']), X(r['start_ms'] + r['median_ms'])
                col = THEME.cam(PLACE_SLOT.get(r['clock'], 0))
                val = f'{r["median_ms"]:.2f} · p95 {(r.get("p95_ms") or 0):.2f} ms'
            p.setPen(QColor(tok['ink']))
            p.setFont(QFont('IBM Plex Sans', 9, QFont.Weight.DemiBold if r is None else QFont.Weight.Normal))
            p.drawText(QRectF(0, y, lw - 10, rh), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, lab)
            p.setPen(Qt.PenStyle.NoPen)
            bar = QRectF(x0, y + rh * 0.22, max(x1 - x0, 2.0), rh * 0.56)
            if r is not None and r['clock'] == 'cross' and not self.trusted:
                p.setBrush(QBrush(col, Qt.BrushStyle.BDiagPattern))
            else:
                p.setBrush(col)
            p.drawRoundedRect(bar, 3, 3)
            if r is not None and (r.get('p95_ms') or 0) > r['median_ms']:
                xp = X(r['start_ms'] + r['p95_ms'])
                p.setPen(QPen(QColor(tok['ink2']), 1.2))
                p.drawLine(int(x1), int(y + rh / 2), int(xp), int(y + rh / 2))
                p.drawLine(int(xp), int(y + rh * 0.3), int(xp), int(y + rh * 0.7))
            if r is None:
                for key in ('p95', 'p99'):
                    if self.e2e.get(key) is not None:
                        xk = X(self.e2e[key])
                        p.setPen(QPen(QColor(tok['ink']), 2))
                        p.drawLine(int(xk), int(y + 2), int(xk), int(y + rh - 2))
            p.setPen(QColor(tok['ink2']))
            p.setFont(small)
            p.drawText(QRectF(W - rw + 8, y, rw - 8, rh), Qt.AlignmentFlag.AlignVCenter, val)
        p.setPen(QColor(tok['muted']))
        p.drawText(QRectF(lw, H - 22, pw, 18), Qt.AlignmentFlag.AlignCenter,
                   f'0 – {span:.0f} ms from the start of the step' + (f' · {self.note}' if self.note else ''))


class HardwarePage(Page):
    key, title, icon = 'hardware', 'Hardware', 'ecu'
    about = ('A device in the loop: stream a camera and a lidar to a Jetson (or a simulated device on this machine), '
             'take its control back, measure every frame\'s latency.')

    def __init__(self, win):
        super().__init__(win)
        self.job = None
        self.last_live = None
        self.report = ''
        self.agent_pid = None
        self.after_session = None
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.addWidget(label('Latency waterfall (the median frame)', title=True))
        self.fall = Waterfall()
        rv.addWidget(self.fall, 1)
        self.stats = KeyValues(('Session', 'Steps', 'End to end', 'Round trip', 'Deadlines', 'Clock offset',
                                'Device', 'CAN'))
        rv.addWidget(self.stats)
        rb = QHBoxLayout()
        self.report_lbl = label('', muted=True, small=True, wrap=True)
        self.open_btn = button('Open the report', self.open_report)
        rb.addWidget(self.report_lbl, 1)
        rb.addWidget(self.open_btn)
        rv.addLayout(rb)
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(12, 12, 12, 12)
        pv.addWidget(self._device_section())
        pv.addWidget(self._session_section())
        pv.addWidget(self._v4l2_section())
        pv.addWidget(self._can_section())
        pv.addWidget(self._clock_section())
        pv.addStretch(1)
        split.addWidget(scroll(panel))
        split.addWidget(right)
        split.setSizes([420, 900])
        win.bridge.changed.connect(self.on_job)
        win.bridge.finished.connect(self.on_job)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_live)
        self.timer.start(500)
        self.load_device()
        self.refresh()
        self.update_buttons()

    # ------------------------------------------------------------------ the device
    def _device_section(self) -> QWidget:
        sec = Section('Device', 'Kept in your settings only, never in a project. No password is stored: Falcon logs in '
                                'with your SSH key (install it once with ssh-copy-id).')
        mode = QHBoxLayout()
        self.sim_rb = QRadioButton('Simulated device (this machine)')
        self.remote_rb = QRadioButton('Jetson on the network')
        self.mode_grp = QButtonGroup(self)
        for b in (self.sim_rb, self.remote_rb):
            self.mode_grp.addButton(b)
            mode.addWidget(b)
            b.toggled.connect(lambda _=False: self.update_buttons())
        self.sim_rb.setChecked(True)
        sec.add(mode)
        f = Form()
        self.address = QLineEdit()
        self.address.setPlaceholderText('host name or IP address')
        self.port = QSpinBox()
        self.port.setRange(1024, 65532)
        self.ssh_user = QLineEdit()
        self.ssh_key = QLineEdit()
        self.ssh_key.setPlaceholderText('~/.ssh/id_ed25519')
        kb = QHBoxLayout()
        kb.addWidget(self.ssh_key, 1)
        kb.addWidget(button('Browse…', self.browse_key))
        self.ssh_port = QSpinBox()
        self.ssh_port.setRange(1, 65535)
        self.remote_dir = QLineEdit()
        self.dev_python = QLineEdit()
        f.row('Address', self.address)
        f.row('Agent port', self.port, 'TCP and UDP; the camera uses the next port, a UDP lidar the one after')
        f.row('SSH user', self.ssh_user)
        f.row('SSH key', kb)
        f.row('SSH port', self.ssh_port)
        f.row('Folder', self.remote_dir)
        f.row('Python', self.dev_python)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Save', self.save_device))
        self.check_btn = button('Copy the agent and check', self.check_device, tip='scp, then device_agent.py --check')
        self.start_agent_btn = button('Start the agent', self.start_agent)
        self.stop_agent_btn = button('Stop the agent', self.stop_agent)
        for w in (self.check_btn, self.start_agent_btn, self.stop_agent_btn):
            b.addWidget(w)
        b.addStretch(1)
        sec.add(b)
        self.dev_info = label('', muted=True, small=True, wrap=True, sel=True)
        sec.add(self.dev_info)
        return sec

    def load_device(self):
        d = device_settings(self.win.settings)
        self.address.setText(d['address'])
        self.port.setValue(int(d['port']))
        self.ssh_user.setText(d['ssh_user'])
        self.ssh_key.setText(d['ssh_key'])
        self.ssh_port.setValue(int(d['ssh_port']))
        self.remote_dir.setText(d['remote_dir'])
        self.dev_python.setText(d['python'])
        self.agent_pid = d.get('agent_pid')

    def device(self) -> dict:
        key = self.ssh_key.text().strip()
        return dict(DEVICE_DEFAULTS, address=self.address.text().strip(), port=self.port.value(),
                    ssh_user=self.ssh_user.text().strip(),
                    ssh_key=os.path.abspath(os.path.expanduser(key)) if key else '', ssh_port=self.ssh_port.value(),
                    remote_dir=self.remote_dir.text().strip() or DEVICE_DEFAULTS['remote_dir'],
                    python=self.dev_python.text().strip() or 'python3', agent_pid=self.agent_pid)

    def save_device(self):
        self.win.settings.set('hil_device', self.device())

    def browse_key(self):
        path, _ = QFileDialog.getOpenFileName(self, 'SSH key', os.path.expanduser('~/.ssh'))
        if path:
            self.ssh_key.setText(path)

    def _deploy_call(self, what: str, fn, done):
        from ...hil import deploy as D
        self.save_device()
        self.dev_info.setText(f'{what}…')

        def failed(e):
            self.dev_info.setText(f'{what} failed: {e}' if isinstance(e, D.DeployError) else f'{what} failed: '
                                  f'{type(e).__name__}: {e}')
        tasks.run(self, fn, done, failed)

    def check_device(self):
        from ...hil import deploy as D
        dev = self.device()
        self._deploy_call('Copying the agent and checking the device', lambda: D.deploy(dev), self.show_check)

    def show_check(self, out: dict):
        env = out['env']
        lines = [f'{env.get("model") or env.get("platform")} · L4T {env.get("l4t") or "—"} · Python '
                 f'{env.get("python")} · numpy {env.get("numpy") or "missing"} · '
                 f'{env.get("gstreamer") or "no GStreamer"} · H.264 decoder {env.get("decoder") or "none"} · ROS 2 {"yes" if env.get("rclpy") else "no"}']
        lines += [f'⚠ {p}' for p in out.get('problems') or []]
        self.dev_info.setText('\n'.join(lines))

    def start_agent(self):
        from ...hil import deploy as D
        dev = self.device()

        def done(pid):
            self.agent_pid = pid
            self.save_device()
            self.dev_info.setText(f'The agent is running on the device (pid {pid}).')
            self.update_buttons()
        self._deploy_call('Starting the agent', lambda: D.start(dev, once=False), done)

    def stop_agent(self):
        from ...hil import deploy as D
        dev, pid = self.device(), self.agent_pid
        if not pid:
            return

        def done(stopped):
            self.agent_pid = None
            self.save_device()
            self.dev_info.setText('The agent was stopped.' if stopped else 'The agent had already ended.')
            self.update_buttons()
        self._deploy_call('Stopping the agent', lambda: D.stop(dev, pid), done)

    # ------------------------------------------------------------------ the session
    def _session_section(self) -> QWidget:
        sec = Section('Session', 'One camera and one lidar of the rig, streamed at the step rate; the device\'s '
                                 'control applied at the next step. A rig without a lidar gets a 32-channel roof '
                                 'lidar.')
        f = Form()
        self.world = QComboBox()
        self.rig = QComboBox()
        self.size = QComboBox()
        self.size.addItems(list(SIZES))
        self.fps = QDoubleSpinBox()
        self.fps.setRange(5, 60)
        self.fps.setValue(20)
        self.fps.setSuffix(' Hz')
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0, 3600)
        self.seconds.setValue(60)
        self.seconds.setSuffix(' s')
        self.seconds.setSpecialValueText('until stopped')
        self.camera = QComboBox()
        self.camera.addItem('H.264 over RTP/UDP', 'h264-rtp')
        self.camera.addItem('Raw BGR over TCP', 'raw-tcp')
        self.camera.addItem('A V4L2 device on this machine', 'v4l2')
        self.camera.currentIndexChanged.connect(lambda _=0: self.update_buttons())
        self.encoder = QComboBox()
        for t, k in (('Auto (NVENC when present)', 'auto'), ('NVENC (nvh264enc)', 'nvh264enc'),
                     ('x264 (x264enc, zero latency)', 'x264enc')):
            self.encoder.addItem(t, k)
        self.lidar = QComboBox()
        self.lidar.addItem('Framed over TCP', 'tcp')
        self.lidar.addItem('Chunked over UDP', 'udp')
        self.cam_clock = QLineEdit('0, 0, 0')
        self.sim_clock = QLineEdit('0, 0')
        f.row('World', self.world)
        f.row('Rig', self.rig)
        f.row('Camera', self.size)
        f.row('Step', self.fps, 'paced to the wall clock')
        f.row('Length', self.seconds)
        f.row('Camera stream', self.camera)
        f.row('Encoder', self.encoder)
        f.row('Lidar stream', self.lidar)
        f.row('Sensor clocks', self.cam_clock, 'emulated hardware clock of both sensors: offset ms, drift ppm, '
                                               'jitter µs')
        f.row('Simulated clock', self.sim_clock, 'the simulated device\'s clock: offset ms, drift ppm (tests the '
                                                 'offset estimate against a known truth)')
        sec.add(f)
        b = QHBoxLayout()
        self.start_btn = button('Start', lambda: self.start(), role='primary')
        self.stop_btn = button('Stop', self.stop)
        self.accept_btn = button('Run the acceptance', self.acceptance,
                                 tip='Copy and start the agent on the Jetson, then 60 s at 1280 × 720, 20 Hz, H.264 '
                                     'and a 32-channel lidar, with the report')
        for w in (self.start_btn, self.stop_btn, self.accept_btn):
            b.addWidget(w)
        b.addStretch(1)
        sec.add(b)
        return sec

    # ------------------------------------------------------------------ the 4.0 interfaces
    def _v4l2_section(self) -> QWidget:
        from ...hil import v4l2out as V
        sec = Section('V4L2 output', 'With the camera stream "A V4L2 device on this machine": each frame is written '
                                     'into a V4L2 output device (a v4l2loopback node), which any program opens as '
                                     'a camera; the simulated device reads it back. Stamps go on a UDP side stream.')
        f = Form()
        self.v4l2_dev = QComboBox()
        self.v4l2_dev.setEditable(True)
        devs = [d['device'] for d in V.devices()] or ['/dev/video40']
        self.v4l2_dev.addItems(devs)
        self.v4l2_fmt = QComboBox()
        self.v4l2_fmt.addItems(list(V.WRITE_FORMATS))
        self.v4l2_method = QComboBox()
        self.v4l2_method.addItem('ioctl + write() (standard library)', 'ioctl')
        self.v4l2_method.addItem('GStreamer v4l2sink', 'gstreamer')
        self.v4l2_barcode = QCheckBox('Frame-number barcode in the top two rows')
        self.v4l2_barcode.setChecked(True)
        f.row('Device', self.v4l2_dev, 'falcon hil setup prints how to create /dev/video40 and 41')
        f.row('Format', self.v4l2_fmt)
        f.row('Writer', self.v4l2_method)
        f.row('', self.v4l2_barcode, 'lets a reader match frames and measure the latency; the stack sees it')
        sec.add(f)
        self.v4l2_info = label('' if V.devices() else 'No V4L2 devices here (v4l2loopback is not loaded).',
                               muted=True, small=True, wrap=True)
        sec.add(self.v4l2_info)
        return sec

    def _can_section(self) -> QWidget:
        from ...core.chassis import SURFACES
        from ...hil import canbridge as B
        sec = Section('CAN', 'The chassis model on a CAN bus: wheel ticks, speeds, yaw rate, steering and brake out '
                             'every 10 ms; the device\'s commands in, with a 100 ms watchdog that holds the vehicle. '
                             'Falcon opens only the interfaces your settings allow (vcan* by default).')
        self.can_on = QCheckBox('Run the CAN loop in the session')
        self.can_on.toggled.connect(lambda _=False: self.update_buttons())
        sec.add(self.can_on)
        f = Form()
        self.can_if = QComboBox()
        self.can_if.setEditable(True)
        self.can_if.addItem('sim:falcon')                    # a simulated link: no vcan, no privileges
        for i in B.interfaces():
            self.can_if.addItem(i['name'])
        self.can_if.currentTextChanged.connect(lambda _='': self.show_can_status())
        self.can_fd = QComboBox()
        for t, k in (('Auto (when the MTU is 72)', 'auto'), ('CAN FD', 'on'), ('Classical CAN', 'off')):
            self.can_fd.addItem(t, k)
        self.can_control = QComboBox()
        self.can_control.addItem('The CAN commands drive the vehicle', 'can')
        self.can_control.addItem('Recorded beside the session\'s control', 'record')
        self.surface = QComboBox()
        for k, s in SURFACES.items():
            self.surface.addItem(f'{k.replace("_", " ")} (μ {s["range"][0]}–{s["range"][1]})', k)
        self.mu = QDoubleSpinBox()
        self.mu.setRange(0.0, 1.2)
        self.mu.setSingleStep(0.05)
        self.mu.setSpecialValueText('the surface\'s')
        self.dbc = QLineEdit()
        self.dbc.setPlaceholderText('Falcon\'s message table (resources/can/falcon_chassis.dbc)')
        db = QHBoxLayout()
        db.addWidget(self.dbc, 1)
        db.addWidget(button('Browse…', self.browse_dbc))
        db.addWidget(button('Save Falcon\'s DBC…', self.save_dbc))
        self.dbc_map = QPlainTextEdit()
        self.dbc_map.setPlaceholderText('With a DBC of your own: one ROLE=MESSAGE.SIGNAL per line (BRAKE_REQ=…)')
        self.dbc_map.setMaximumHeight(70)
        f.row('Interface', self.can_if, 'sim:NAME: a simulated link between the processes here, no vcan needed')
        f.row('Frames', self.can_fd)
        f.row('Control', self.can_control)
        f.row('Surface', self.surface, 'waterlogged: hydroplaning from Horne\'s speed (about 94 km/h at 220 kPa)')
        f.row('Friction μ', self.mu)
        f.row('DBC', db)
        f.row('Signal map', self.dbc_map)
        sec.add(f)
        b = QHBoxLayout()
        self.confirm_btn = button('Confirm a bench adapter…', self.confirm_adapter,
                                  tip='Record that this adapter is on a bench network, not a vehicle')
        b.addWidget(self.confirm_btn)
        b.addStretch(1)
        sec.add(b)
        self.can_status = label('', muted=True, small=True, wrap=True, sel=True)
        sec.add(self.can_status)
        self.show_can_status()
        return sec

    def _clock_section(self) -> QWidget:
        from ...core.ptp import SOURCES
        sec = Section('Clock source', 'The offset between this machine\'s clock and the device\'s: from PTP when both '
                                      'follow one grandmaster (ptp4l), else from NTP-style pings. Both are computed '
                                      'and recorded with their disagreement; this chooses the one the waterfall uses.')
        f = Form()
        self.clock_src = QComboBox()
        for k, t in zip(SOURCES, ('Auto: PTP when usable, else the pings', 'PTP (falls back, saying why)',
                                  'NTP-style pings only')):
            self.clock_src.addItem(t, k)
        cur = (self.win.settings.get('hil_ptp') or {}).get('clock_source', 'auto')
        self.clock_src.setCurrentIndex(max(self.clock_src.findData(cur), 0))
        f.row('Clock source', self.clock_src)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Check PTP here', self.check_ptp, tip='ptp4l, pmc and ethtool -T on this machine'))
        b.addWidget(button('Save a ptp4l.conf…', self.save_ptp_conf, tip='software timestamping or the gPTP profile, '
                                                                          'for you to install'))
        b.addStretch(1)
        sec.add(b)
        self.ptp_info = label('', muted=True, small=True, wrap=True, sel=True)
        sec.add(self.ptp_info)
        return sec

    def show_can_status(self):
        from ...hil import canbridge as B
        if not hasattr(self, 'can_status'):
            return
        name = self.can_if.currentText().strip()
        if not name:
            self.can_status.setText('')
            return
        ok, why, _ = B.check_interface(name, self.win.settings)
        real = not name.startswith('sim:') and not B.is_virtual(name)
        self.can_status.setText(f'{name}: {"allowed" if ok else "refused"} — {why}')
        set_enabled(self.confirm_btn, real and not ok and not self.running,
                    'Only a real adapter that is not yet confirmed needs this')

    def confirm_adapter(self):
        from ...hil import canbridge as B
        name = self.can_if.currentText().strip()
        text, ok = QInputDialog.getText(
            self, 'A real CAN adapter', f'Falcon will send frames on {name} at 100 Hz and act on what comes back. '
            f'Type "{B.CONFIRM_TEXT}" to confirm that {name} is wired to a bench network, not a vehicle:')
        if not ok:
            return
        try:
            B.confirm_bench(self.win.settings, name, text)
        except ValueError as e:
            QMessageBox.warning(self, 'CAN', str(e))
            return
        self.show_can_status()

    def browse_dbc(self):
        path, _ = QFileDialog.getOpenFileName(self, 'A DBC file', os.path.expanduser('~'), 'DBC (*.dbc)')
        if path:
            self.dbc.setText(path)

    def save_dbc(self):
        from ...core.canmsgs import dbc_text
        path, _ = QFileDialog.getSaveFileName(self, 'Falcon\'s chassis DBC', 'falcon_chassis.dbc', 'DBC (*.dbc)')
        if path:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(dbc_text())

    def check_ptp(self):
        from ...core import ptp

        def work():
            rows = ptp.doctor_rows(self.win.settings)
            return rows, ptp.query_pmc()

        def done(out):
            rows, q = out
            s = q.get('summary') or {}
            lines = [f'{"✓" if ok else "–"} {n}: {d}' for n, ok, d in rows]
            lines.append(f'pmc: grandmaster {"present" if s.get("gm_present") else "absent"}, port '
                         f'{s.get("port_state")}, offset {s.get("offset_ns")} ns' if s else
                         f'pmc: {q.get("error") or "no answer"}')
            self.ptp_info.setText('\n'.join(lines))
        self.ptp_info.setText('Checking…')
        tasks.run(self, work, done, lambda e: self.ptp_info.setText(f'{type(e).__name__}: {e}'))

    def save_ptp_conf(self):
        from ...core import ptp
        prof, ok = QInputDialog.getItem(self, 'ptp4l.conf', 'Profile', ['software timestamping (this NIC)',
                                                                        'gPTP (802.1AS, hardware timestamping)'],
                                        0, False)
        if not ok:
            return
        iface = (self.win.settings.get('hil_ptp') or {}).get('interface') or ptp.default_interface() or 'eth0'
        profile = 'gptp' if prof.startswith('gPTP') else 'software'
        path, _ = QFileDialog.getSaveFileName(self, 'ptp4l.conf', f'falcon-{profile}.conf', 'conf (*.conf)')
        if path:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(ptp.ptp4l_conf(profile, iface))
            self.ptp_info.setText(f'Written for {iface}. Install it yourself: falcon hil setup prints the commands '
                                  '(Falcon runs none of them).')

    def can_config(self):
        """The session's CAN section (core.hil.can_section), or None when the loop is off."""
        if not self.can_on.isChecked():
            return None
        chassis = {'surface': self.surface.currentData()}
        if self.mu.value() > 0:
            chassis['mu'] = round(self.mu.value(), 3)
        can = {'interface': self.can_if.currentText().strip(), 'fd': self.can_fd.currentData(),
               'control': self.can_control.currentData(), 'chassis': chassis}
        if self.dbc.text().strip():
            mp = {}
            for line in self.dbc_map.toPlainText().splitlines():
                k, sep, v = line.partition('=')
                if sep:
                    mp[k.strip()] = v.strip()
            can.update(dbc=os.path.abspath(os.path.expanduser(self.dbc.text().strip())), dbc_map=mp)
        return can

    def v4l2_config(self):
        if self.camera.currentData() != 'v4l2':
            return None
        return {'device': self.v4l2_dev.currentText().strip(), 'format': self.v4l2_fmt.currentText(),
                'method': self.v4l2_method.currentData(), 'barcode': self.v4l2_barcode.isChecked()}

    def show_can_live(self, can: dict | None):
        if not can:
            return
        self.stats.set('CAN', f'{can.get("mode")}{" (FD)" if can.get("fd") else ""} · sent {can.get("tx_frames")} · '
                              f'commands {can.get("commands")} · watchdog {can.get("watchdog_trips")} · rejected '
                              f'{can.get("rejected_crc", 0) + can.get("rejected_counter", 0)} · error frames '
                              f'{can.get("error_frames")}' + (f' · ABORTED: {can["aborted"]}' if can.get('aborted')
                                                              else ''))

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

    @property
    def running(self) -> bool:
        return self.job is not None and self.job.state not in FINAL

    @staticmethod
    def _numbers(text: str, n: int) -> list:
        vals = [float(x) for x in text.replace(';', ',').split(',') if x.strip()]
        return (vals + [0.0] * n)[:n]

    def build_config(self, remote: bool, **over) -> dict:
        w = self.win
        width, height = SIZES[self.size.currentText()]
        off, drift, jit = self._numbers(self.cam_clock.text(), 3)
        soff, sdrift = self._numbers(self.sim_clock.text(), 2)
        clock = {'offset_ms': off, 'drift_ppm': drift, 'jitter_us': jit}
        kw = dict(seconds=self.seconds.value(), fps=self.fps.value(), width=width, height=height,
                  camera=self.camera.currentData(), encoder=self.encoder.currentData(), lidar=self.lidar.currentData(),
                  device='remote' if remote else 'local', device_port=self.port.value(), camera_clock=clock,
                  lidar_clock=clock, sim_device={'clock_offset_ms': soff, 'clock_drift_ppm': sdrift},
                  workspace=w.workspace(), can=self.can_config(), v4l2=self.v4l2_config(),
                  clock_source=self.clock_src.currentData(), settings=w.settings)
        kw.update(over)
        return hil_config(w.project, self.world.currentData(), self.rig.currentData(),
                          os.path.join(w.service.runs_dir(), run_dir_name('hil')), w.service.port, w.service.tm_port,
                          **kw)

    def start(self, **over):
        if self.running:
            return
        remote = self.remote_rb.isChecked()
        try:
            cfg = self.build_config(remote, **over)
            self.job = submit_session(self.win.service, cfg, self.address.text().strip() if remote else '')
        except (ProjectError, RuntimeError, ValueError) as e:
            QMessageBox.warning(self, 'Hardware', str(e))
            return
        self.last_live = None
        self.report = ''
        self.report_lbl.setText('')
        self.fall.set_data([], {}, cfg['fps'] and 1000.0 / cfg['fps'], True)
        self.update_buttons()

    def stop(self):
        if self.running and not self.win.queue.send(self.job, {'cmd': 'stop'}):
            self.win.queue.cancel(self.job)

    def acceptance(self):
        """The real-device acceptance: copy and check, start the agent for one session, then the session."""
        from ...hil import deploy as D
        if self.running:
            return
        self.remote_rb.setChecked(True)
        dev = self.device()

        def go(out):
            self.show_check(out)
            self.agent_pid = out['pid']
            self.save_device()
            if not out['env'].get('decoder') and self.camera.currentData() == 'h264-rtp':
                self.dev_info.setText(self.dev_info.text() + '\nThe device cannot decode H.264: choose the raw '
                                                             'camera stream.')
                return
            self.after_session = 'stop_agent'
            self.start(width=1280, height=720, fps=20.0, seconds=60.0,
                       lidar_overrides={'channels': 32})
        self._deploy_call('Copying, checking and starting the agent', lambda: D.deploy(dev, start_agent=True,
                                                                                         once=True), go)

    # ------------------------------------------------------------------ the job
    def on_job(self, job):
        if job is not self.job:
            return
        if job.state in FINAL:
            self.stats.set('Session', f'{job.state}: {job.summary()}')
            self.show_final()
            if self.after_session == 'stop_agent' and self.agent_pid:
                self.after_session = None
                self.stop_agent()
        elif job.state != RUNNING:
            self.stats.set('Session', job.summary())
        else:
            self.stats.set('Session', 'running')
        self.update_buttons()

    def poll_live(self):
        if not self.running:
            return
        live = next((m for m in reversed(self.job.other) if m.get('t') == 'hil'), None)
        if live is None or live is self.last_live:
            return
        self.last_live = live
        self.show_numbers(live['waterfall'], live.get('e2e') or {}, live.get('round_trip') or {},
                          live.get('deadline') or {}, live.get('offset'), live.get('one_way_trusted', True),
                          f'{live.get("steps")} steps, ×{live.get("realtime_factor")} real time',
                          'the last two seconds')
        self.show_can_live(live.get('can'))

    def show_numbers(self, rows, e2e, rt, dl, off, trusted, steps, note):
        period = 1000.0 / self.fps.value()
        self.fall.set_data(rows, e2e, period, trusted, note)
        f = lambda x: '—' if x is None else f'{x:.2f} ms'      # noqa: E731
        self.stats.set('Steps', steps)
        self.stats.set('End to end', f'median {f(e2e.get("median"))}, p95 {f(e2e.get("p95"))}, '
                                     f'p99 {f(e2e.get("p99"))}')
        self.stats.set('Round trip', f'median {f(rt.get("median"))}, p95 {f(rt.get("p95"))} (exact: this machine\'s '
                                     'clock only)')
        self.stats.set('Deadlines', f'on time {dl.get("on_time", 0)}, late {dl.get("late", 0)}, superseded '
                                    f'{dl.get("superseded", 0)}, no control {dl.get("missing", 0)}')
        if off:
            self.stats.set('Clock offset', f'{off["offset_ms"]:.3f} ms ± {off["uncertainty_ms"]:.3f} ms'
                                           f'{"" if trusted else " — one-way stages not trusted"}')

    def show_final(self):
        from ...core.provenance import read_json
        run_dir = self.job.spec.run_dir
        try:
            run = read_json(os.path.join(run_dir, 'run.json'))
        except (OSError, ValueError):
            return
        h = run.get('hil')
        if not h:
            if run.get('error'):
                self.report_lbl.setText(run['error'])
            return
        m = h['measures']
        est = h.get('clock_offset') or {}
        off = None if not est else {'offset_ms': est['offset_ns'] / 1e6, 'uncertainty_ms': est['uncertainty_ns'] / 1e6}
        perf = run.get('performance') or {}
        self.show_numbers(h['waterfall'], m['e2e'], m['round_trip'], h['deadline'], off, h['one_way_trusted'],
                          f'{perf.get("steps")} steps, ×{perf.get("realtime_factor")} real time', 'the whole session')
        dev = (h.get('session') or {}).get('device') or {}
        env = dev.get('env') or {}
        self.stats.set('Device', f'{"simulated, this machine" if dev.get("mode") == "local" else env.get("model")}'
                                 f' · decoder {dev.get("decoder") or "none (raw)"} · stack {dev.get("stack")}')
        can = h.get('can') or {}
        if can.get('stats'):
            self.show_can_live(dict(can['stats'], aborted=can.get('aborted')))
        cs = h.get('clock_sources') or {}
        if cs.get('used') and off:
            dis = cs.get('disagreement_ns')
            agree = 'consistent' if cs.get('consistent') else 'beyond their uncertainties'
            other = '' if dis is None else f'; the other source differs by {dis / 1e3:.1f} µs ({agree})'
            self.stats.set('Clock offset', f'{off["offset_ms"]:.3f} ms ± {off["uncertainty_ms"]:.3f} ms from '
                                           f'{cs["used"]}{other}', cs.get('why') or '')
        self.report = os.path.join(run_dir, 'report.html')
        if os.path.isfile(self.report):
            self.report_lbl.setText(tilde(self.report))
        self.update_buttons()

    def open_report(self):
        if self.report and os.path.isfile(self.report):
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.report))

    def update_buttons(self):
        if not hasattr(self, 'accept_btn'):       # a radio button toggles while the page is still being built
            return
        r = self.running
        remote = self.remote_rb.isChecked()
        busy = 'A run is in progress: stop it first'
        set_enabled(self.start_btn, not r, busy)
        set_enabled(self.stop_btn, r, 'Nothing is running')
        set_enabled(self.accept_btn, not r, busy)
        for w in (self.check_btn, self.start_agent_btn):
            set_enabled(w, remote and not r, busy if r else 'Only for a remote device: choose it above')
        set_enabled(self.stop_agent_btn, bool(self.agent_pid) and not r,
                    busy if r else 'No agent started from here is running')
        set_enabled(self.open_btn, bool(self.report) and os.path.isfile(self.report), 'No report written yet')
        for w in (self.world, self.rig, self.size, self.fps, self.seconds, self.camera, self.encoder, self.lidar):
            set_enabled(w, not r, 'Fixed while a run is in progress')
        if not hasattr(self, 'clock_src'):                 # the 4.0 sections are built after the session's
            return
        v4 = self.camera.currentData() == 'v4l2'
        for w in (self.v4l2_dev, self.v4l2_fmt, self.v4l2_method, self.v4l2_barcode):
            set_enabled(w, v4 and not r, busy if r else 'Only with the camera stream "A V4L2 device on this machine"')
        set_enabled(self.encoder, self.camera.currentData() == 'h264-rtp' and not r,
                    busy if r else 'Only for the H.264 camera stream')
        on = self.can_on.isChecked()
        set_enabled(self.can_on, not r, busy)
        for w in (self.can_if, self.can_fd, self.can_control, self.surface, self.mu, self.dbc, self.dbc_map):
            set_enabled(w, on and not r, busy if r else 'Turn the CAN loop on first')
        set_enabled(self.clock_src, not r, busy)
        self.show_can_status()
