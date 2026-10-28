"""Settings: where CARLA is, Falcon's ports, the server's memory cap and quality, the render budget, the theme.

Machine settings, kept in the user's config folder ($XDG_CONFIG_HOME/FalconAutonomy), never in a project. Secrets
(a Jetson's credentials) are set where they are used and stored apart, never shown."""
from __future__ import annotations

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox,
                             QSpinBox)

from ...core.paths import tilde
from ...core.settings import RESERVED_PORTS
from ...sim import server as S
from .. import tasks
from ..widgets import Form, Page, Section, button, label


class SettingsPage(Page):
    key, title, icon = 'settings', 'Settings', 'gear'
    about = 'Machine settings: kept in your config folder, never in a project.'

    def __init__(self, win):
        super().__init__(win)
        st = win.settings
        v = self.page_layout(note=f'{self.about} ({tilde(st.path)})')

        carla = Section('CARLA')
        f = Form()
        self.root = QLineEdit(st.get('carla_root'))
        self.root.setPlaceholderText(tilde(win.service.carla_root) + ' (found)' if win.service.carla_root else
                                     'not found')
        rb = QHBoxLayout()
        rb.addWidget(self.root, 1)
        rb.addWidget(button('Browse…', self.browse_root))
        f.row('CARLA folder', rb, 'holds CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping')
        self.py = QLineEdit(st.get('client_python'))
        self.py.setPlaceholderText(tilde(S.client_python(win.service.carla_root)) or 'a venv beside the CARLA folder')
        pb = QHBoxLayout()
        pb.addWidget(self.py, 1)
        pb.addWidget(button('Check', self.check_python))
        f.row('CARLA\'s Python', pb, 'the venv with carla==0.9.16, numpy and OpenCV (workers run in it)')
        self.py_status = label('', muted=True, wrap=True)
        f.row('', self.py_status)
        carla.add(f)
        v.addWidget(carla)

        srv = Section('Falcon\'s server', 'Falcon starts its own headless server on its own ports, capped in memory with '
                                         'no swap. Ports 2000–2010 and 8100 (CARLA\'s defaults, where your own server '
                                         'usually listens) are used only if you choose them here — Falcon then '
                                         'connects to your server and never starts or stops it.')
        g = Form()
        self.port = QSpinBox()
        self.port.setRange(1024, 65533)
        self.port.setValue(int(st.get('rpc_port')))
        g.row('RPC port', self.port, 'CARLA also listens on the next two ports')
        self.tm_port = QSpinBox()
        self.tm_port.setRange(1024, 65535)
        self.tm_port.setValue(int(st.get('tm_port')))
        g.row('Traffic manager port', self.tm_port)
        self.mem = QDoubleSpinBox()
        self.mem.setRange(4, 64)
        self.mem.setSuffix(' GB')
        self.mem.setValue(float(st.get('server_memory_gb')))
        g.row('Memory cap', self.mem, 'a 0.9.16 server capped at 12 GB ran 77 Mpx per step and crashed at 306')
        self.quality = QComboBox()
        self.quality.addItems(['Epic', 'Low'])
        self.quality.setCurrentText(st.get('server_quality'))
        g.row('Quality', self.quality)
        self.budget = QDoubleSpinBox()
        self.budget.setRange(1, 2000)
        self.budget.setSuffix(' Mpx')
        self.budget.setValue(float(st.get('render_budget_mpx')))
        g.row('Render budget', self.budget, 'runs rendering more per step are refused before they start')
        srv.add(g)
        v.addWidget(srv)

        look = Section('Appearance and privacy')
        h = Form()
        self.theme = QComboBox()
        self.theme.addItems(['dark', 'light'])
        self.theme.setCurrentText(st.get('theme'))
        self.theme.activated.connect(lambda _: win.set_theme(self.theme.currentText()))
        h.row('Theme', self.theme)
        self.geo = QCheckBox('Allow place-name search over the network (OpenStreetMap Nominatim)')
        self.geo.setChecked(bool(st.get('geocoding_allowed')))
        h.row('Geocoding', self.geo, 'off: imports work from files and bounding boxes only')
        look.add(h)
        v.addWidget(look)
        b = QHBoxLayout()
        b.addWidget(button('Apply', self.apply, role='primary'))
        b.addStretch(1)
        v.addLayout(b)
        v.addStretch(1)

    def browse_root(self):
        d = QFileDialog.getExistingDirectory(self, 'CARLA folder', self.root.text() or self.win.service.carla_root)
        if d:
            self.root.setText(d)

    def check_python(self):
        py = self.py.text().strip() or S.client_python(self.root.text().strip() or self.win.service.carla_root)
        self.py_status.setText('checking…')
        tasks.run(self, lambda: S.check_client(py), lambda r: self.py_status.setText(
            ('✓ ' if r['ok'] else '⚠ ') + r['text']))

    def apply(self):
        st = self.win.settings
        root = self.root.text().strip()
        if root and not S.is_carla_root(root):
            QMessageBox.warning(self, 'Settings', f'{root} is not a CARLA folder ({S.BINARY} is missing).')
            return
        port, tm = self.port.value(), self.tm_port.value()
        allowed = set(st.get('allowed_ports') or [])
        for p in (port, tm):
            if p in RESERVED_PORTS and p not in allowed:
                r = QMessageBox.question(self, 'A default CARLA port', f'Port {p} is one of CARLA\'s defaults, where a '
                                         'server you started may listen. Use it? Falcon will then connect to the '
                                         'server there and never start or stop it.')
                if r != QMessageBox.StandardButton.Yes:
                    return
                allowed.add(p)
        allowed &= {port, tm}                  # a choice holds only while that port is chosen
        if self.win.service.server_state() in ('ready', 'starting') and port != self.win.service.port:
            QMessageBox.information(self, 'Settings', 'Stop Falcon\'s server before changing its port.')
            return
        st.set('carla_root', root, save=False)
        st.set('client_python', self.py.text().strip(), save=False)
        st.set('rpc_port', port, save=False)
        st.set('tm_port', tm, save=False)
        st.set('allowed_ports', sorted(allowed), save=False)
        st.set('server_memory_gb', self.mem.value(), save=False)
        st.set('server_quality', self.quality.currentText(), save=False)
        st.set('render_budget_mpx', self.budget.value(), save=False)
        st.set('geocoding_allowed', self.geo.isChecked(), save=False)
        st.save()
        self.win.log.append_line(f'settings saved ({tilde(st.path)})')
        for p in self.win.pages:
            if p is not self:
                p.refresh()
