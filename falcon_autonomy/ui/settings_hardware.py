"""Settings ▸ Hardware (4.0): which CAN interfaces Falcon may open (hil_can) and the PTP interface and clock source
(hil_ptp). A real CAN adapter is allowed only once confirmed as a bench network, not a vehicle (hil.canbridge
.confirm_bench: recorded with its time); patterns cannot allow one by themselves. A section of the Settings page
(ui.pages.settings SECTIONS): section(page) builds it, its apply(settings) stores it."""
from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QInputDialog, QLineEdit, QMessageBox

from .widgets import Form, Section, button, label


class HardwareSection(Section):
    def __init__(self, page):
        from ..core.ptp import SOURCES
        super().__init__('Hardware', 'Hardware in the loop: the CAN interfaces Falcon may open (vcan* by default; a '
                                     'real adapter also needs your confirmation that it is on a bench network, not a '
                                     'vehicle) and PTP time sync. Falcon never runs anything with privileges: falcon '
                                     'hil setup prints the commands.')
        self.page = page
        st = page.win.settings
        can = st.get('hil_can') or {}
        ptp = st.get('hil_ptp') or {}
        f = Form()
        self.patterns = QLineEdit(', '.join(can.get('interfaces_allowed') or ['vcan*']))
        f.row('CAN interfaces', self.patterns, 'shell patterns, comma-separated')
        self.ptp_if = QLineEdit(ptp.get('interface') or '')
        self.ptp_if.setPlaceholderText('the default route\'s interface')
        f.row('PTP interface', self.ptp_if)
        self.source = QComboBox()
        for k in SOURCES:
            self.source.addItem(k, k)
        self.source.setCurrentIndex(max(self.source.findData(ptp.get('clock_source', 'auto')), 0))
        f.row('Clock source', self.source, 'auto: PTP when both machines follow one grandmaster, else NTP-style')
        self.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Confirm a bench adapter…', self.confirm))
        b.addWidget(button('Withdraw a confirmation…', self.revoke))
        b.addStretch(1)
        self.add(b)
        self.confirmed = label('', muted=True, small=True, wrap=True, sel=True)
        self.add(self.confirmed)
        self.show_confirmed()

    def show_confirmed(self):
        c = (self.page.win.settings.get('hil_can') or {}).get('confirmed') or {}
        self.confirmed.setText('Confirmed bench adapters: ' + (', '.join(f'{k} ({v.get("at")})' for k, v in c.items())
                                                               or 'none'))

    def confirm(self):
        from ..hil import canbridge as B
        name, ok = QInputDialog.getText(self, 'A real CAN adapter',
                                        'Interface (can0, or python-can:INTERFACE:CHANNEL):')
        if not ok or not name.strip():
            return
        text, ok = QInputDialog.getText(self, 'A real CAN adapter', f'Type "{B.CONFIRM_TEXT}" to confirm that '
                                                                    f'{name.strip()} is wired to a bench network:')
        if not ok:
            return
        try:
            cfg = B.confirm_bench(self.page.win.settings, name.strip(), text)
        except ValueError as e:
            QMessageBox.warning(self, 'Hardware', str(e))
            return
        self.patterns.setText(', '.join(cfg['interfaces_allowed']))
        self.show_confirmed()

    def revoke(self):
        from ..hil import canbridge as B
        c = (self.page.win.settings.get('hil_can') or {}).get('confirmed') or {}
        if not c:
            return
        name, ok = QInputDialog.getItem(self, 'Withdraw', 'Adapter', sorted(c), 0, False)
        if ok:
            cfg = B.revoke_bench(self.page.win.settings, name)
            self.patterns.setText(', '.join(cfg['interfaces_allowed']))
            self.show_confirmed()

    def apply(self, settings) -> str:
        import re
        pats = [p.strip() for p in self.patterns.text().split(',') if p.strip()]
        bad = [p for p in pats if not re.match(r'^[A-Za-z0-9_.:*?\[\]-]{1,32}$', p)]
        if bad:
            return f'Not an interface pattern: {", ".join(bad)}'
        iface = self.ptp_if.text().strip()
        if iface and not re.match(r'^[A-Za-z0-9_.:-]{1,15}$', iface):
            return f'{iface!r} is not an interface name'
        can = dict(settings.get('hil_can') or {})
        can['interfaces_allowed'] = pats
        settings.set('hil_can', can, save=False)
        settings.set('hil_ptp', dict(settings.get('hil_ptp') or {}, interface=iface,
                                     clock_source=self.source.currentData()), save=False)
        return ''


def section(page) -> HardwareSection | None:
    """The Hardware section, or None when the page has no window (it reads and writes the window's settings)."""
    if getattr(getattr(page, 'win', None), 'settings', None) is None:
        return None
    return HardwareSection(page)
