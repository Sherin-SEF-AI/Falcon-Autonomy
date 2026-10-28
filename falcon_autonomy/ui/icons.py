"""Single-colour SVG icons tinted per state at load."""
from __future__ import annotations

import os
import tempfile

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap, QColor, QImage, QGuiApplication
from PyQt6.QtSvg import QSvgRenderer

try:
    from .icon_data import ICONS
except Exception:  # icon set not generated yet
    ICONS = {}

FALLBACK = '<rect x="3.5" y="3.5" width="9" height="9"/>'
_cache: dict = {}
_files: dict = {}
_dir = os.path.join(tempfile.gettempdir(), f'vantage-icons-{os.getuid() if hasattr(os, "getuid") else 0}')


def clear_cache():
    _cache.clear()


def svg(name: str, color: str, stroke: float = 1.25) -> str:
    inner = ICONS.get(name, FALLBACK)
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none" '
         f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')
    return s.replace('currentColor', color)


def _dpr():
    app = QGuiApplication.instance()
    try:
        return max(1.0, app.devicePixelRatio()) if app else 1.0
    except Exception:
        return 1.0


def pixmap(name: str, color, size: int = 16) -> QPixmap:
    col = color.name() if isinstance(color, QColor) else str(color)
    dpr = _dpr()
    key = (name, col, size, dpr)
    if key in _cache:
        return _cache[key]
    # the 12 px set reuses the geometry with a thicker unit stroke so it still draws ~1.25 px
    stroke = 1.25 * 16 / size if size < 16 else 1.25
    r = QSvgRenderer(QByteArray(svg(name, col, stroke).encode()))
    px = int(round(size * dpr))
    img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    r.render(p, QRectF(0, 0, px, px))
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(dpr)
    _cache[key] = pm
    return pm


def icon(name: str, size: int = 16, color: str | None = None) -> QIcon:
    from .theme import THEME
    key = ('icon', name, size, color, THEME.name)
    if key in _cache:
        return _cache[key]
    ic = QIcon()
    normal = color or THEME.tok['ink2']
    ic.addPixmap(pixmap(name, normal, size), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(pixmap(name, color or THEME.tok['ink'], size), QIcon.Mode.Active, QIcon.State.Off)
    ic.addPixmap(pixmap(name, color or THEME.tok['ink'], size), QIcon.Mode.Selected, QIcon.State.Off)
    ic.addPixmap(pixmap(name, color or THEME.tok['ink'], size), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(pixmap(name, color or THEME.tok['ink'], size), QIcon.Mode.Active, QIcon.State.On)
    ic.addPixmap(pixmap(name, THEME.tok['muted'] if not color else color, size), QIcon.Mode.Disabled, QIcon.State.Off)
    _cache[key] = ic
    return ic


def status_icon(sev: str, size: int = 12) -> QIcon:
    from .theme import STATUS
    name = {'good': 'good', 'warning': 'warning', 'serious': 'serious', 'critical': 'critical'}.get(sev, 'warning')
    return icon(name, size, STATUS.get(sev, '#888888'))


def file_for(name: str, color: str, size: int = 12) -> str:
    """Path to a tinted SVG file (for QSS url())."""
    key = (name, color, size)
    if key in _files and os.path.exists(_files[key]):
        return _files[key]
    os.makedirs(_dir, exist_ok=True)
    path = os.path.join(_dir, f'{name}-{color.strip("#")}-{size}.svg')
    stroke = 1.25 * 16 / size if size < 16 else 1.25
    with open(path, 'w') as fh:
        fh.write(svg(name, color, min(stroke, 2.4)))
    _files[key] = path.replace('\\', '/')
    return _files[key]
