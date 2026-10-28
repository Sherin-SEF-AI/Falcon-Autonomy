"""Game controllers and steering wheels through the Linux joystick API (/dev/input/js*), with no extra library.

Each event is 8 bytes: time (ms, uint32), value (int16), type (uint8: 1 button, 2 axis, | 0x80 initial state) and
number (uint8). The device's name comes from the JSIOCGNAME ioctl. Axes are scaled to −1..1. Which axis steers,
accelerates and brakes differs between devices, so it is chosen on the Live page (with inversion); pedals that rest
at −1 (most wheels) are mapped to 0..1 with `pedal=True`.
"""
from __future__ import annotations

import fcntl
import glob
import os
import struct

from PyQt6.QtCore import QObject, QSocketNotifier, pyqtSignal

EVENT = struct.Struct('IhBB')
JS_EVENT_BUTTON, JS_EVENT_AXIS, JS_EVENT_INIT = 0x01, 0x02, 0x80


def _jsiocgname(length=128):
    # _IOC(_IOC_READ, 'j', 0x13, len)
    return (2 << 30) | (length << 16) | (ord('j') << 8) | 0x13


def devices() -> list:
    """[(path, name)] of the joystick devices present."""
    out = []
    for path in sorted(glob.glob('/dev/input/js*')):
        name = os.path.basename(path)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                buf = bytearray(128)
                fcntl.ioctl(fd, _jsiocgname(), buf)
                name = buf.split(b'\0', 1)[0].decode(errors='replace') or name
            finally:
                os.close(fd)
        except OSError:
            continue
        out.append((path, name))
    return out


def decode(data: bytes) -> list:
    """[(kind 'axis'|'button', number, value)] from raw event bytes (axes −1..1, buttons 0/1)."""
    out = []
    for k in range(0, len(data) - EVENT.size + 1, EVENT.size):
        _t, value, typ, number = EVENT.unpack_from(data, k)
        typ &= ~JS_EVENT_INIT
        if typ == JS_EVENT_AXIS:
            out.append(('axis', number, max(value / 32767.0, -1.0)))
        elif typ == JS_EVENT_BUTTON:
            out.append(('button', number, 1 if value else 0))
    return out


class Joystick(QObject):
    changed = pyqtSignal()

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes: dict = {}
        self.buttons: dict = {}
        self.notifier = QSocketNotifier(self.fd, QSocketNotifier.Type.Read, self)
        self.notifier.activated.connect(self._read)

    def _read(self, *_):
        try:
            data = os.read(self.fd, EVENT.size * 64)
        except BlockingIOError:
            return
        except OSError:
            self.close()
            return
        for kind, n, v in decode(data):
            (self.axes if kind == 'axis' else self.buttons)[n] = v
        self.changed.emit()

    def axis(self, n: int, invert=False, pedal=False) -> float:
        v = self.axes.get(int(n), -1.0 if pedal else 0.0)
        v = -v if invert else v
        return (v + 1.0) / 2.0 if pedal else v

    def close(self):
        if self.fd is not None:
            self.notifier.setEnabled(False)
            os.close(self.fd)
            self.fd = None
