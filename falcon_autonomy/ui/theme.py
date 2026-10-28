"""Design tokens, QSS and fonts. Token names map 1:1 to @placeholders in the QSS template."""
from __future__ import annotations

import math
import os

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')

TOKENS = {
    'dark': {
        'base': '#121211', 'panel': '#1a1a19', 'raised': '#222220', 'sel': '#2c2c2a', 'hair': '#2c2c2a',
        'strong': '#383835', 'ink': '#f0efec', 'ink2': '#c3c2b7', 'muted': '#898781',
        'viewport': '#1a1a19', 'grid1': '#262624', 'grid2': '#2f2f2c', 'body': '#c8c7c2', 'glass': '#3b3c3b',
        'ground': '#7b7a74', 'sky': '#3f3f3b',
    },
    'light': {
        'base': '#ecebe7', 'panel': '#fcfcfb', 'raised': '#f3f2ef', 'sel': '#e1e0d9', 'hair': '#e1e0d9',
        'strong': '#c3c2b7', 'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#898781',
        'viewport': '#fcfcfb', 'grid1': '#eceae5', 'grid2': '#dcdad3', 'body': '#e2e1dc', 'glass': '#5b5c5a',
        'ground': '#9a9993', 'sky': '#5e5e59',
    },
}
CAMERA = {
    'dark': ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
    'light': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
}
STATUS = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
       '#1c5cab', '#184f95', '#104281', '#0d366b']
REDUNDANCY = ['#86b6ef', '#2a78d6', '#184f95']
DIVERGING = ['#b8312f', '#d9665e', '#ec9d93', '#383835', '#86b6ef', '#3987e5', '#1c5cab']
DIVERGING_LIGHT_MID = '#e1e0d9'

QSS = r'''
* { font-family: "IBM Plex Sans"; font-size: @fs12px; color: @ink; }
QMainWindow, QMainWindow::separator { background: @base; }
QMainWindow::separator { width: 2px; height: 2px; }
QDialog { background: @panel; }
QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; }
QDockWidget::title { background: @panel; height: 22px; padding-left: 8px; border-bottom: 1px solid @hair; text-align: left; }
QWidget[dockBody="true"], QWidget#dockBody { background: @panel; }
QWidget#panel { background: @panel; }
QWidget#base { background: @base; }
QMenuBar { background: @panel; min-height: 24px; border-bottom: 1px solid @hair; }
QMenuBar::item { padding: 0 8px; color: @ink2; background: transparent; }
QMenuBar::item:selected, QMenuBar::item:pressed { background: @sel; color: @ink; }
QMenu { background: @raised; border: 1px solid @strong; border-radius: 4px; padding: 4px 0; }
QMenu::item { min-height: 22px; padding: 0 28px 0 28px; color: @ink; }
QMenu::item:selected { background: @sel; }
QMenu::item:disabled { color: @muted; }
QMenu::separator { height: 1px; background: @hair; margin: 4px 0; }
QMenu::indicator { width: 12px; height: 12px; left: 9px; }
QMenu::right-arrow { image: url(@icon_chevr); width: 12px; height: 12px; right: 8px; }
QToolBar { background: @panel; border: none; border-bottom: 1px solid @hair; spacing: 2px; padding: 0 4px; }
QToolBar::separator { width: 1px; background: @hair; margin: 7px 5px; }
QToolButton { min-width: 22px; min-height: 22px; border: none; border-radius: 2px; background: transparent; padding: 0; }
QToolBar QToolButton { min-width: 26px; min-height: 26px; }
QToolButton:hover { background: @raised; }
QToolButton:checked, QToolButton:pressed { background: @sel; }
QToolButton:disabled { color: @muted; }
QToolButton::menu-indicator { image: none; width: 0; }
QToolButton[popupMode="1"] { padding-right: 10px; }
QToolButton::menu-button { border: none; width: 10px; }
QToolButton::menu-arrow { image: url(@icon_caret); width: 8px; height: 8px; }
QToolButton[text-button="true"] { padding: 0 6px; color: @ink2; }
QLineEdit, QAbstractSpinBox, QComboBox { min-height: 20px; max-height: 22px; padding: 0 6px; background: @raised;
    border: 1px solid @strong; border-radius: 2px; selection-background-color: @sel; selection-color: @ink; }
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover { border-color: @muted; }
QAbstractSpinBox { font-family: "IBM Plex Mono"; qproperty-buttonSymbols: NoButtons; qproperty-alignment: "AlignRight|AlignVCenter"; }
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus { border-color: @ink; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled { color: @muted; border-color: @hair; }
QLineEdit[readOnly="true"] { background: transparent; border-color: transparent; }
QAbstractSpinBox[mixed="true"] { color: @muted; }
QLineEdit[mono="true"] { font-family: "IBM Plex Mono"; }
QComboBox { padding-right: 18px; }
QComboBox::drop-down { border: none; width: 16px; }
QComboBox::down-arrow { image: url(@icon_chevd); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: @raised; border: 1px solid @strong; selection-background-color: @sel; outline: none; padding: 2px 0; }
QPushButton { min-height: 20px; padding: 0 10px; border: 1px solid @strong; border-radius: 2px; background: transparent; }
QPushButton:hover { background: @raised; }
QPushButton:pressed { background: @sel; }
QPushButton:focus { border-color: @ink; }
QPushButton[role="primary"] { background: @ink; border-color: @ink; color: @panel; }
QPushButton[role="primary"]:hover { background: @ink2; border-color: @ink2; }
QPushButton[role="primary"]:pressed { background: @muted; border-color: @muted; }
QPushButton[role="ghost"] { border-color: transparent; }
QPushButton[role="ghost"]:focus { border-color: @ink; }
QPushButton[role="destructive"] { border-color: @crit; }
QPushButton:disabled { color: @muted; border-color: @hair; }
QPushButton[role="primary"]:disabled { background: @sel; border-color: @sel; color: @muted; }
QPushButton[seg="true"] { border-radius: 0; border-right-width: 0; padding: 0 8px; color: @ink2; }
QPushButton[seg="true"][segpos="first"] { border-top-left-radius: 2px; border-bottom-left-radius: 2px; }
QPushButton[seg="true"][segpos="last"] { border-top-right-radius: 2px; border-bottom-right-radius: 2px; border-right-width: 1px; }
QPushButton[seg="true"][segpos="only"] { border-radius: 2px; border-right-width: 1px; }
QPushButton[seg="true"]:checked { background: @sel; color: @ink; }
QTabBar { background: @base; qproperty-drawBase: 0; }
QTabBar::tab { height: 22px; padding: 0 12px; background: @base; color: @ink2; border: none; border-top: 2px solid transparent; }
QTabBar::tab:selected { background: @panel; color: @ink; border-top-color: @ink; }
QTabBar::tab:hover:!selected { color: @ink; }
QTabWidget::pane { border: none; }
QHeaderView { background: @panel; }
QHeaderView::section { height: 22px; padding: 0 8px; background: @raised; color: @ink2; border: none; border-bottom: 1px solid @strong; }
QTreeView, QTableView, QListView { background: @panel; border: none; gridline-color: @hair; outline: none;
    selection-background-color: @sel; selection-color: @ink; alternate-background-color: @panel; }
QTreeView::item, QListView::item { min-height: 22px; }
QTreeView::item:hover, QTableView::item:hover, QListView::item:hover { background: @raised; }
QTreeView::item:selected, QTableView::item:selected, QListView::item:selected { background: @sel; color: @ink; }
QTreeView::branch { background: transparent; }
QTreeView::branch:has-children:!has-siblings:closed, QTreeView::branch:closed:has-children:has-siblings { image: url(@icon_chevr); }
QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings { image: url(@icon_chevd); }
QStatusBar { background: @panel; border-top: 1px solid @hair; font-size: @fs11px; color: @ink2; min-height: 22px; }
QStatusBar QLabel, QStatusBar QToolButton { font-size: @fs11px; color: @ink2; }
QStatusBar::item { border: none; }
QToolTip { background: @raised; color: @ink; border: 1px solid @strong; border-radius: 4px; padding: 4px 8px; }
QSlider { min-height: 16px; }
QSlider::groove:horizontal { height: 2px; background: @strong; }
QSlider::handle:horizontal { width: 10px; height: 10px; margin: -4px 0; background: @ink; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @ink2; }
QProgressBar { max-height: 2px; min-height: 2px; border: none; background: @hair; text-align: center; }
QProgressBar::chunk { background: @ink; }
QProgressBar[inline="true"] { max-height: 3px; min-height: 3px; }
QScrollBar:vertical { width: 8px; background: transparent; margin: 0; }
QScrollBar::handle:vertical { background: @strong; border-radius: 2px; min-height: 24px; }
QScrollBar:horizontal { height: 8px; background: transparent; margin: 0; }
QScrollBar::handle:horizontal { background: @strong; border-radius: 2px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QCheckBox, QRadioButton { spacing: 6px; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid @strong; border-radius: 2px; background: @raised; }
QCheckBox::indicator:hover { border-color: @muted; }
QCheckBox::indicator:checked { background: @ink; border-color: @ink; image: url(@icon_checkinv); }
QCheckBox::indicator:indeterminate { background: @ink; border-color: @ink; image: url(@icon_dashinv); }
QCheckBox::indicator:disabled { background: transparent; border-color: @hair; }
QCheckBox:disabled { color: @muted; }
QRadioButton::indicator { width: 12px; height: 12px; border: 1px solid @strong; border-radius: 6px; background: @raised; }
QRadioButton::indicator:checked { border-color: @ink; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 @ink, stop:0.42 @ink, stop:0.5 @raised, stop:1 @raised); }
QSplitter::handle { background: @base; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical { height: 2px; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QLabel { background: transparent; }
QLabel[muted="true"] { color: @muted; }
QLabel[ink2="true"] { color: @ink2; }
QLabel[mono="true"] { font-family: "IBM Plex Mono"; }
QLabel[small="true"] { font-size: @fs11px; }
QLabel[h="true"] { font-weight: 600; }
QLabel[title="true"] { font-weight: 600; font-size: @fs13px; }
QLabel[crit="true"] { color: @crit; }
QFrame[hair="true"] { background: @hair; border: none; }
QTextEdit, QPlainTextEdit { background: @panel; border: none; font-family: "IBM Plex Mono"; selection-background-color: @sel; }
QWidget#errorBox { border: 1px solid @crit; border-radius: 2px; }
QWidget#card { background: @raised; border: 1px solid @strong; border-radius: 4px; }
QListWidget#steps { background: @base; border: none; border-right: 1px solid @hair; }
QListWidget#steps::item { min-height: 40px; padding-left: 0; }
QListWidget#steps::item:selected { background: @sel; }
'''


class ThemeManager(QObject):
    changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.name = 'dark'
        self.scale = 1.0
        self.tok = dict(TOKENS['dark'])

    def color(self, key, alpha=None) -> QColor:
        if key in self.tok:
            c = QColor(self.tok[key])
        elif key in STATUS:
            c = QColor(STATUS[key])
        else:
            c = QColor(key)
        if alpha is not None:
            c.setAlphaF(alpha)
        return c

    def cam(self, slot, alpha=None) -> QColor:
        c = QColor(CAMERA[self.name][slot % 8])
        if alpha is not None:
            c.setAlphaF(alpha)
        return c

    def cam_hex(self, slot):
        return CAMERA[self.name][slot % 8]

    def px(self, v):
        return int(round(v * self.scale))

    def apply(self, app: QApplication, name: str | None = None, scale: float | None = None):
        if name == 'system':
            name = 'dark' if app.styleHints().colorScheme().name == 'Dark' else 'light'
        if name:
            self.name = name
        if scale:
            self.scale = scale
        self.tok = dict(TOKENS[self.name])
        from . import icons
        icons.clear_cache()
        qss = QSS
        tok = dict(self.tok)
        tok['crit'] = STATUS['critical']
        tok['fs12'] = str(self.px(12))
        tok['fs11'] = str(self.px(11))
        tok['fs13'] = str(self.px(13))
        tok['icon_chevr'] = icons.file_for('chev-r', self.tok['ink2'], 12)
        tok['icon_chevd'] = icons.file_for('chev-d', self.tok['ink2'], 12)
        tok['icon_caret'] = icons.file_for('caret', self.tok['ink2'], 8)
        tok['icon_checkinv'] = icons.file_for('check-inv', self.tok['panel'], 12)
        tok['icon_dashinv'] = icons.file_for('dash', self.tok['panel'], 12)
        for k, v in sorted(tok.items(), key=lambda kv: -len(kv[0])):
            qss = qss.replace('@' + k, v)
        pal = app.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(self.tok['panel']))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(self.tok['ink']))
        pal.setColor(QPalette.ColorRole.Base, QColor(self.tok['raised']))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(self.tok['panel']))
        pal.setColor(QPalette.ColorRole.Text, QColor(self.tok['ink']))
        pal.setColor(QPalette.ColorRole.Button, QColor(self.tok['panel']))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(self.tok['ink']))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(self.tok['sel']))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(self.tok['ink']))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(self.tok['raised']))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor(self.tok['ink']))
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(self.tok['muted']))
        pal.setColor(QPalette.ColorRole.Mid, QColor(self.tok['strong']))
        pal.setColor(QPalette.ColorRole.Dark, QColor(self.tok['base']))
        pal.setColor(QPalette.ColorRole.Light, QColor(self.tok['raised']))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(self.tok['muted']))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(self.tok['muted']))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(self.tok['muted']))
        app.setPalette(pal)
        f = QFont('IBM Plex Sans')
        f.setPixelSize(self.px(12))
        app.setFont(f)
        app.setStyleSheet(qss)
        self.changed.emit(self.name)


THEME = ThemeManager()


def load_fonts():
    fdir = os.path.join(RES, 'fonts')
    if os.path.isdir(fdir):
        for fn in sorted(os.listdir(fdir)):
            if fn.lower().endswith(('.ttf', '.otf')):
                QFontDatabase.addApplicationFont(os.path.join(fdir, fn))


def mono(px=12, weight=None) -> QFont:
    f = QFont('IBM Plex Mono')
    f.setPixelSize(THEME.px(px))
    f.setStyleHint(QFont.StyleHint.Monospace)
    if weight:
        f.setWeight(weight)
    return f


def sans(px=12, weight=None) -> QFont:
    f = QFont('IBM Plex Sans')
    f.setPixelSize(THEME.px(px))
    if weight:
        f.setWeight(weight)
    return f


def seq_color(v, lo=3.0, hi=16.0, alpha=None) -> QColor:
    """Pixel-density colour on a log scale (lo..hi px/cm)."""
    if v <= 0:
        return QColor(0, 0, 0, 0)
    t = (math.log(max(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    i = int(round(min(max(t, 0), 1) * (len(SEQ) - 1)))
    c = QColor(SEQ[i])
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


def seq_lut(lo=3.0, hi=16.0, n=256):
    """uint8 RGBA lookup for density values 0..hi*1.2 mapped through seq_color (index = v / (hi*1.2) * (n-1))."""
    import numpy as np
    lut = np.zeros((n, 4), np.uint8)
    for k in range(n):
        v = k / (n - 1) * hi * 1.2
        if v <= 0:
            continue
        c = seq_color(v, lo, hi)
        lut[k] = (c.red(), c.green(), c.blue(), 255)
    return lut


def fmt(v, dp=3, unit=''):
    from ..core.mathutil import fmt_num
    s = fmt_num(v, dp)
    return f'{s} {unit}'.rstrip() if unit else s
