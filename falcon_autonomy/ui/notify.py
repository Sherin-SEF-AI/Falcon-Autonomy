"""Desktop notifications when a long job ends while Falcon's window is in the background.

A job that ran at least LONG_S seconds and finished, failed or was cancelled raises one notification — only when the
window is not the active one, the setting 'notify_jobs' is on and the platform has a desktop (never offscreen, as in
the tests). The message goes through the desktop's notification service (notify-send, freedesktop.org) when there
is one, else the system tray. Nothing is sent anywhere else.
"""
from __future__ import annotations

import shutil

from PyQt6.QtCore import QProcess
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QSystemTrayIcon

LONG_S = 20.0
HEADLESS = ('offscreen', 'minimal', 'vnc', 'eglfs-headless')


def should_notify(job, window_active: bool, enabled: bool, platform: str) -> tuple[bool, str]:
    """(notify?, why not) for a finished job."""
    if not enabled:
        return False, 'turned off in Settings'
    if platform in HEADLESS:
        return False, f'no desktop ({platform})'
    if window_active:
        return False, 'the window is active'
    if job.elapsed < LONG_S:
        return False, f'shorter than {LONG_S:g} s'
    return True, ''


def message(job) -> tuple[str, str]:
    from ..core import protocol as P
    verb = {'done': 'finished', 'failed': 'failed', 'cancelled': 'was cancelled'}.get(job.state, job.state)
    body = f'after {job.elapsed:.0f} s'
    if job.state == 'failed' and job.exit_code is not None:
        body += f' — {P.describe_exit(job.exit_code)}'
    return f'{job.spec.title} {verb}', body


class Notifier:
    def __init__(self, win):
        self.win = win
        self.tray = None
        self.sent: list = []                     # (title, body, backend) of what was sent (for the tests and the log)

    def on_finished(self, job):
        ok, _ = should_notify(job, self.win.isActiveWindow(), bool(self.win.settings.get('notify_jobs')),
                              QGuiApplication.platformName())
        if ok:
            self.send(*message(job), critical=job.state == 'failed')

    def send(self, title, body, critical=False):
        exe = shutil.which('notify-send')
        if exe:
            QProcess.startDetached(exe, ['--app-name=Falcon Autonomy', '--urgency=' + ('critical' if critical
                                                                                        else 'normal'), title, body])
            self.sent.append((title, body, 'notify-send'))
            return
        if QSystemTrayIcon.isSystemTrayAvailable():
            if self.tray is None:
                self.tray = QSystemTrayIcon(self.win.windowIcon(), self.win)
                self.tray.show()
            icon = QSystemTrayIcon.MessageIcon.Critical if critical else QSystemTrayIcon.MessageIcon.Information
            self.tray.showMessage(title, body, icon, 8000)
            self.sent.append((title, body, 'tray'))

    def stop(self):
        if self.tray is not None:
            self.tray.hide()
            self.tray = None
