"""The red banner under the page header while a cloud machine could not be terminated (cluster.cloud writes
cloud/alert.json when a terminate keeps failing; docs/cloud.md). It stays until the machine is terminated — by the
watchdog's timer, by Retry now here, or by `falcon cloud terminate` — and is re-read every 30 s. At Falcon's start
(on a desktop) the watchdog's pass and the orphan sweep run once in the background (cloud.startup_check)."""
from __future__ import annotations

import threading

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QFrame, QHBoxLayout

from .notify import HEADLESS
from .theme import STATUS
from .widgets import button, label

POLL_MS = 30000


class CloudBanner(QFrame):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setObjectName('cloudBanner')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        red = STATUS['critical']
        self.setStyleSheet(f'#cloudBanner {{ background: {red}; }} #cloudBanner QLabel {{ color: white; }}')
        h = QHBoxLayout(self)
        h.setContentsMargins(16, 6, 16, 6)
        self.text = label('', wrap=True)
        h.addWidget(self.text, 1)
        self.retry = button('Retry now', self.retry_now, tip='run the watchdog\'s pass now: terminate it again')
        h.addWidget(self.retry)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(POLL_MS)
        self.refresh()

    def refresh(self):
        from ..cluster.cloud import alert
        al = alert()
        if not al:
            self.hide()
            return
        ids = ', '.join(f'{b.get("provider")} {b.get("instance_id")}' for b in al.get('bursts') or ())
        self.text.setText(f'{al.get("message")} ({ids}). falcon cloud status says more.')
        self.show()

    def retry_now(self):
        self.retry.setEnabled(False)

        def run():
            from ..cluster.cloud import watchdog_tick
            try:
                watchdog_tick(self.win.settings)
            except Exception:                       # noqa: BLE001 (the banner stays; the timer tries again)
                pass
        t = threading.Thread(target=run, daemon=True)
        t.start()
        QTimer.singleShot(3000, lambda: (self.retry.setEnabled(True), self.refresh()))


def install(win, layout) -> CloudBanner:
    b = CloudBanner(win)
    layout.insertWidget(1, b)                       # under the page header, above the page
    win.cloud_banner = b
    if QGuiApplication.platformName() not in HEADLESS:
        from ..cluster.cloud import startup_check
        startup_check(win.settings)                 # in a background thread; no network without a burst or a key
    return b
