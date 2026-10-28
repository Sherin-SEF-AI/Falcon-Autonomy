"""Hardware: a device in the loop — a Jetson on the network, or a simulated device on this machine. Set up the device
(its address, SSH user and key, kept in your settings only, never in a project; no password is stored: SSH keys
only), copy the device agent to it and check what it has, start a session (a job like any other) and watch the
latency waterfall and the deadlines as it runs; the report when it ends. docs/hardware.md is the guide.

Every number shown is measured by the session (core.hil): the waterfall is the median frame, stage by stage, over
the last two seconds while running and over the whole session at the end; one-way network stages are drawn hatched
when the clock-offset estimate is not trusted."""
from __future__ import annotations

import os

from PyQt6.QtCore import QRectF, Qt, QTimer, QUrl
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLineEdit,
                             QMessageBox, QRadioButton, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ...core.drive import run_dir_name
from ...core.hil import DEVICE_DEFAULTS, device_settings, hil_config, submit_session
from ...core.jobs import FINAL, RUNNING
from ...core.paths import tilde
from ...core.project import ProjectError
from .. import tasks
from ..theme import THEME
from ..widgets import Form, KeyValues, Page, Section, button, label, scroll

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
                                'Device'))
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
        pv.addWidget(label(self.title, title=True))
        pv.addWidget(label(self.about, muted=True, wrap=True))
        pv.addWidget(self._device_section())
        pv.addWidget(self._session_section())
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
                  workspace=w.workspace())
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
        self.report = os.path.join(run_dir, 'report.html')
        if os.path.isfile(self.report):
            self.report_lbl.setText(tilde(self.report))
        self.update_buttons()

    def open_report(self):
        if self.report and os.path.isfile(self.report):
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.report))

    def update_buttons(self):
        r = self.running
        remote = self.remote_rb.isChecked()
        self.start_btn.setEnabled(not r)
        self.stop_btn.setEnabled(r)
        self.accept_btn.setEnabled(not r)
        for w in (self.check_btn, self.start_agent_btn):
            w.setEnabled(remote and not r)
        self.stop_agent_btn.setEnabled(bool(self.agent_pid) and not r)
        self.open_btn.setEnabled(bool(self.report) and os.path.isfile(self.report))
        for w in (self.world, self.rig, self.size, self.fps, self.seconds, self.camera, self.encoder, self.lidar):
            w.setEnabled(not r)
