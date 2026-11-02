"""Design tokens, QSS and fonts: the lab-console look (graphite panels in bordered cards, one blue accent for the
primary action and the selection, status colours only for status, monospaced numbers). Token names map 1:1 to
@placeholders in the QSS template."""
from __future__ import annotations

import math
import os

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')

TOKENS = {
    'dark': {
        'base': '#0d0f12', 'panel': '#14171c', 'raised': '#1a1e24', 'sunken': '#101317', 'sel': '#1c2a3e',
        'hair': '#232830', 'strong': '#2e3540', 'ink': '#e7eaee', 'ink2': '#a9b1bd', 'muted': '#717a89',
        'accent': '#3d8bfd', 'accent_hi': '#64a2ff', 'accent_soft': '#16263d', 'accent_line': '#24405f',
        'on_accent': '#ffffff', 'alt': '#171a20',
        'viewport': '#101317', 'grid1': '#1b1f26', 'grid2': '#252b34', 'body': '#c8ccd2', 'glass': '#3b3f46',
        'ground': '#7b8088', 'sky': '#3f444b',
    },
    'light': {
        'base': '#f2f3f5', 'panel': '#ffffff', 'raised': '#f6f7f9', 'sunken': '#f0f2f4', 'sel': '#e1ebfd',
        'hair': '#e1e4e9', 'strong': '#cbd1d9', 'ink': '#0f1720', 'ink2': '#475263', 'muted': '#768091',
        'accent': '#2563eb', 'accent_hi': '#1d4ed8', 'accent_soft': '#e8f0fe', 'accent_line': '#bcd2fb',
        'on_accent': '#ffffff', 'alt': '#f8f9fb',
        'viewport': '#fbfbfc', 'grid1': '#eceef1', 'grid2': '#dde1e6', 'body': '#e2e4e8', 'glass': '#5b5f66',
        'ground': '#9a9ea6', 'sky': '#5e636a',
    },
}
CAMERA = {
    'dark': ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
    'light': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
}
STATUS = {'good': '#2fbf71', 'warning': '#f2a93b', 'serious': '#ec835a', 'critical': '#ef5350'}
STATUS_LIGHT = {'good': '#16884a', 'warning': '#b36b00', 'serious': '#c4552b', 'critical': '#d32f2f'}   # text on white
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
       '#1c5cab', '#184f95', '#104281', '#0d366b']
REDUNDANCY = ['#86b6ef', '#2a78d6', '#184f95']
DIVERGING = ['#b8312f', '#d9665e', '#ec9d93', '#383835', '#86b6ef', '#3987e5', '#1c5cab']
DIVERGING_LIGHT_MID = '#e1e0d9'

QSS = r'''
* { font-family: "IBM Plex Sans"; font-size: @fs13px; color: @ink; }
QMainWindow, QMainWindow::separator { background: @base; }
QMainWindow::separator { width: 1px; height: 1px; background: @hair; }
QDialog { background: @panel; }
QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; }
QDockWidget::title { background: @panel; height: 26px; padding-left: 10px; border-bottom: 1px solid @hair; text-align: left; }
QWidget[dockBody="true"], QWidget#dockBody { background: @panel; }
QWidget#panel { background: @panel; }
QWidget#base { background: @base; }
QWidget#topBar { background: @panel; border-bottom: 1px solid @hair; }
QLabel#brand { font-family: "IBM Plex Mono"; font-weight: 600; font-size: @fs13px; color: @ink; }
QLabel#crumbSep { color: @strong; }
QLabel#projectName { font-weight: 600; }
QLabel#projectPath { color: @muted; font-family: "IBM Plex Mono"; font-size: @fs12px; }
QToolButton#searchBox { background: @sunken; border: 1px solid @hair; border-radius: 6px; color: @muted; text-align: left;
    padding: 0 12px; min-height: 28px; }
QToolButton#searchBox:hover { border-color: @strong; color: @ink2; }
QFrame#topChip { background: @raised; border: 1px solid @hair; border-radius: 14px; }
QFrame#topChip:hover { border-color: @strong; }
QLabel#chipText { color: @ink2; font-family: "IBM Plex Mono"; font-size: @fs12px; }
QWidget#pageHeader { background: @base; }
QLabel#pageTitle { font-size: @fs22px; font-weight: 600; }
QMenuBar { background: transparent; min-height: 26px; border: none; }
QMenuBar::item { padding: 3px 9px; color: @ink2; background: transparent; border-radius: 4px; }
QMenuBar::item:selected, QMenuBar::item:pressed { background: @raised; color: @ink; }
QMenu { background: @raised; border: 1px solid @strong; border-radius: 8px; padding: 5px 0; }
QMenu::item { min-height: 26px; padding: 0 28px 0 28px; color: @ink; border-radius: 4px; margin: 0 5px; }
QMenu::item:selected { background: @accent_soft; }
QMenu::item:disabled { color: @muted; }
QMenu::separator { height: 1px; background: @hair; margin: 5px 0; }
QMenu::indicator { width: 12px; height: 12px; left: 9px; }
QMenu::right-arrow { image: url(@icon_chevr); width: 12px; height: 12px; right: 8px; }
QToolBar { background: @panel; border: none; border-bottom: 1px solid @hair; spacing: 2px; padding: 0 4px; }
QToolBar::separator { width: 1px; background: @hair; margin: 7px 5px; }
QToolButton { min-width: 22px; min-height: 22px; border: none; border-radius: 4px; background: transparent; padding: 0; }
QToolBar QToolButton { min-width: 28px; min-height: 28px; }
QToolButton:hover { background: @raised; }
QToolButton:checked, QToolButton:pressed { background: @accent_soft; }
QToolButton:disabled { color: @muted; }
QToolButton::menu-indicator { image: none; width: 0; }
QToolButton[popupMode="1"] { padding-right: 10px; }
QToolButton::menu-button { border: none; width: 10px; }
QToolButton::menu-arrow { image: url(@icon_caret); width: 8px; height: 8px; }
QToolButton[text-button="true"] { padding: 0 8px; color: @ink2; }
QToolButton[text-button="true"]:hover { color: @ink; }
QLineEdit, QAbstractSpinBox, QComboBox { min-height: 24px; max-height: 28px; padding: 0 8px; background: @sunken;
    border: 1px solid @strong; border-radius: 5px; selection-background-color: @accent_soft; selection-color: @ink; }
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover { border-color: @muted; }
QAbstractSpinBox { qproperty-buttonSymbols: NoButtons; qproperty-alignment: "AlignLeft|AlignVCenter"; }
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus { border-color: @accent; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled { color: @muted; border-color: @hair; background: transparent; }
QLineEdit[readOnly="true"] { background: transparent; border-color: transparent; }
QAbstractSpinBox[mixed="true"] { color: @muted; }
QLineEdit[mono="true"] { font-family: "IBM Plex Mono"; }
QComboBox { padding-right: 20px; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox::down-arrow { image: url(@icon_chevd); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: @raised; border: 1px solid @strong; selection-background-color: @accent_soft;
    outline: none; padding: 3px; }
QPushButton { min-height: 26px; padding: 0 14px; border: 1px solid @strong; border-radius: 5px; background: @raised; }
QPushButton:hover { background: @sel; border-color: @muted; }
QPushButton:pressed { background: @accent_soft; }
QPushButton:focus { border-color: @accent; }
QPushButton:checked { background: @accent_soft; border-color: @accent_line; color: @ink; }
QPushButton[role="primary"] { background: @accent; border-color: @accent; color: @on_accent; font-weight: 600; }
QPushButton[role="primary"]:hover { background: @accent_hi; border-color: @accent_hi; }
QPushButton[role="primary"]:pressed { background: @accent_line; border-color: @accent_line; }
QPushButton[role="primary"]:focus { border-color: @ink; }
QPushButton[role="ghost"] { border-color: transparent; background: transparent; color: @ink2; }
QPushButton[role="ghost"]:hover { background: @raised; color: @ink; }
QPushButton[role="link"] { border-color: transparent; background: transparent; text-align: left; padding: 0 4px; color: @accent_hi; }
QPushButton[role="link"]:hover { color: @ink; }
QPushButton[role="ghost"]:focus { border-color: @accent; }
QPushButton[role="destructive"] { border-color: @crit; color: @crit; }
QPushButton[role="destructive"]:hover { background: @crit; color: @on_accent; }
QPushButton:disabled { color: @muted; border-color: @hair; background: transparent; }
QPushButton[role="primary"]:disabled { background: @raised; border-color: @hair; color: @muted; }
QPushButton[seg="true"] { border-radius: 0; border-right-width: 0; padding: 0 10px; color: @ink2; background: @sunken; }
QPushButton[seg="true"][segpos="first"] { border-top-left-radius: 5px; border-bottom-left-radius: 5px; }
QPushButton[seg="true"][segpos="last"] { border-top-right-radius: 5px; border-bottom-right-radius: 5px; border-right-width: 1px; }
QPushButton[seg="true"][segpos="only"] { border-radius: 5px; border-right-width: 1px; }
QPushButton[seg="true"]:checked { background: @accent_soft; color: @ink; border-color: @accent_line; }
QTabBar { background: transparent; qproperty-drawBase: 0; }
QTabBar::tab { height: 32px; padding: 0 14px; background: transparent; color: @ink2; border: none;
    border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: @ink; border-bottom-color: @accent; font-weight: 600; }
QTabBar::tab:hover:!selected { color: @ink; border-bottom-color: @strong; }
QTabWidget::pane { border: none; border-top: 1px solid @hair; top: -1px; }
QHeaderView { background: @panel; }
QHeaderView::section { height: 28px; padding: 0 10px; background: @panel; color: @muted; border: none;
    border-bottom: 1px solid @hair; font-size: @fs11px; font-weight: 600; }
QHeaderView::section:hover { color: @ink2; }
QTreeView, QTableView, QListView { background: @panel; border: none; gridline-color: @hair; outline: none;
    selection-background-color: @accent_soft; selection-color: @ink; alternate-background-color: @alt; }
QTreeView::item, QListView::item { min-height: 26px; }
QTreeView::item:hover, QTableView::item:hover, QListView::item:hover { background: @raised; }
QTreeView::item:selected, QTableView::item:selected, QListView::item:selected { background: @accent_soft; color: @ink; }
QTreeView::branch { background: transparent; }
QTreeView::branch:selected { background: @accent_soft; }
QTreeView::branch:has-children:!has-siblings:closed, QTreeView::branch:closed:has-children:has-siblings { image: url(@icon_chevr); }
QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings { image: url(@icon_chevd); }
QTableCornerButton::section { background: @panel; border: none; }
QStatusBar { background: @panel; border-top: 1px solid @hair; font-size: @fs12px; color: @ink2; min-height: 26px; }
QStatusBar QLabel, QStatusBar QToolButton { font-size: @fs12px; color: @ink2; }
QStatusBar QLabel[mono="true"] { font-family: "IBM Plex Mono"; }
QStatusBar::item { border: none; }
QToolTip { background: @raised; color: @ink; border: 1px solid @strong; border-radius: 6px; padding: 5px 9px; }
QSlider { min-height: 18px; }
QSlider::groove:horizontal { height: 4px; background: @strong; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; background: @ink; border: 2px solid @accent; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: @accent_hi; }
QSlider::sub-page:horizontal { background: @accent; border-radius: 2px; }
QSlider::groove:vertical { width: 4px; background: @strong; border-radius: 2px; }
QSlider::handle:vertical { width: 14px; height: 14px; margin: 0 -5px; background: @ink; border: 2px solid @accent; border-radius: 7px; }
QProgressBar { max-height: 4px; min-height: 4px; border: none; background: @raised; border-radius: 2px; text-align: center; }
QProgressBar::chunk { background: @accent; border-radius: 2px; }
QProgressBar[inline="true"] { max-height: 4px; min-height: 4px; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { background: @strong; border-radius: 3px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: @muted; }
QScrollBar:horizontal { height: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:horizontal { background: @strong; border-radius: 3px; min-width: 28px; }
QScrollBar::handle:horizontal:hover { background: @muted; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QCheckBox, QRadioButton { spacing: 8px; }
QCheckBox::indicator, QAbstractItemView::indicator { width: 14px; height: 14px; border: 1px solid @strong; border-radius: 4px; background: @sunken; }
QCheckBox::indicator:hover, QAbstractItemView::indicator:hover { border-color: @accent; }
QCheckBox::indicator:checked, QAbstractItemView::indicator:checked { background: @accent; border-color: @accent; image: url(@icon_checkinv); }
QCheckBox::indicator:indeterminate, QAbstractItemView::indicator:indeterminate { background: @accent; border-color: @accent; image: url(@icon_dashinv); }
QCheckBox::indicator:disabled { background: transparent; border-color: @hair; }
QCheckBox:disabled { color: @muted; }
QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid @strong; border-radius: 7px; background: @sunken; }
QRadioButton::indicator:hover { border-color: @accent; }
QRadioButton::indicator:checked { border-color: @accent; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 @accent, stop:0.45 @accent, stop:0.55 @sunken, stop:1 @sunken); }
QSplitter::handle { background: @base; }
QSplitter::handle:hover { background: @accent_line; }
QSplitter::handle:horizontal { width: 6px; }
QSplitter::handle:vertical { height: 6px; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QLabel { background: transparent; }
QLabel[muted="true"] { color: @muted; }
QLabel[ink2="true"] { color: @ink2; }
QLabel[mono="true"] { font-family: "IBM Plex Mono"; }
QLabel[small="true"] { font-size: @fs12px; }
QLabel[h="true"] { font-weight: 600; }
QLabel[title="true"] { font-weight: 600; font-size: @fs14px; }
QLabel[page="true"] { font-weight: 600; font-size: @fs22px; }
QLabel[crit="true"] { color: @crit; }
QLabel[cap="true"] { color: @muted; font-size: @fs11px; font-weight: 600; }
QLabel[kpi="true"] { font-family: "IBM Plex Mono"; font-size: @fs26px; font-weight: 600; }
QLabel[unit="true"] { color: @muted; font-family: "IBM Plex Mono"; font-size: @fs12px; }
QLabel[chip="true"] { background: @raised; border: 1px solid @strong; border-radius: 9px; padding: 0 8px; font-size: @fs11px;
    color: @ink2; min-height: 18px; max-height: 18px; font-family: "IBM Plex Mono"; }
QLabel[chip="true"][tone="good"] { color: @good; border-color: @good; }
QLabel[chip="true"][tone="warning"] { color: @warn; border-color: @warn; }
QLabel[chip="true"][tone="critical"] { color: @crit; border-color: @crit; }
QLabel[chip="true"][tone="accent"] { color: @accent_hi; border-color: @accent_line; background: @accent_soft; }
QFrame[hair="true"] { background: @hair; border: none; }
QTextEdit, QPlainTextEdit { background: @panel; border: none; font-family: "IBM Plex Mono"; selection-background-color: @accent_soft; }
QTextBrowser { font-family: "IBM Plex Sans"; }
QWidget#errorBox { border: 1px solid @crit; border-radius: 6px; }
QWidget#card { background: @raised; border: 1px solid @strong; border-radius: 8px; }
QFrame#section { background: @panel; border: 1px solid @hair; border-radius: 8px; }
QFrame#section QFrame#section { background: transparent; border: none; border-top: 1px solid @hair; border-radius: 0; }
QFrame#kpi { background: @panel; border: 1px solid @hair; border-radius: 8px; }
QFrame#banner { background: @accent_soft; border: 1px solid @accent_line; border-radius: 8px; }
QFrame#callout { background: @raised; border: 1px solid @strong; border-radius: 6px; }
QFrame#callout[tone="warning"] { border-color: @warn; }
QFrame#callout[tone="critical"] { border-color: @crit; }
QListWidget#cardList { background: @panel; border: 1px solid @hair; border-radius: 8px; padding: 6px; }
QListWidget#cardList::item { border-radius: 6px; padding-left: 6px; }
QListWidget#cardList::item:selected { background: @accent_soft; color: @ink; }
QListWidget#steps { background: @base; border: none; border-right: 1px solid @hair; }
QListWidget#steps::item { min-height: 40px; padding-left: 0; }
QListWidget#steps::item:selected { background: @accent_soft; }
QWidget#side { background: @panel; border-right: 1px solid @hair; }
QListWidget#nav { background: transparent; border: none; padding: 6px 8px; }
QListWidget#nav::item { min-height: 32px; padding-left: 8px; color: @ink2; border-radius: 6px; }
QListWidget#nav::item:selected { background: transparent; color: @ink; }
QListWidget#nav::item:hover:!selected { background: @raised; }
QToolButton#navFoot { color: @muted; font-size: @fs11px; min-height: 30px; padding: 0 16px; text-align: left; background: transparent; border: none; border-top: 1px solid @hair; border-radius: 0; font-family: "IBM Plex Mono"; }
QToolButton#navFoot:hover { color: @ink; }
QToolButton#jobsBadge { padding: 0 8px; color: @ink2; }
QToolButton#jobsBadge[busy="true"] { color: @accent_hi; font-weight: 600; }
QWidget#homeCard { background: @panel; border: 1px solid @hair; border-radius: 8px; }
QFrame#suggestChip { background: @accent_soft; border: 1px solid @accent_line; border-radius: 13px; }
QFrame#suggestChip:hover { border-color: @accent; }
QToolButton#suggestAct { border: none; background: transparent; color: @ink; padding: 2px 4px; min-height: 22px; }
QToolButton#suggestDismiss { border: none; background: transparent; color: @muted; padding: 0 6px; }
QToolButton#suggestDismiss:hover { color: @ink; }
QLabel[stat="true"] { font-family: "IBM Plex Mono"; font-size: @fs14px; font-weight: 600; }
QLineEdit#paletteInput { min-height: 34px; max-height: 36px; font-size: @fs14px; }
QLabel#bubbleUser { background: @accent_soft; border: 1px solid @accent_line; border-radius: 8px; }
QFrame#toolCard { background: @panel; border: 1px solid @hair; border-radius: 8px; }
QFrame#toolCard QToolButton { text-align: left; color: @ink2; font-size: @fs12px; }
QPlainTextEdit#composer { background: @sunken; border: 1px solid @strong; border-radius: 6px; padding: 4px 6px; font-family: "IBM Plex Sans"; }
QPlainTextEdit#composer:focus { border-color: @accent; }
QPlainTextEdit#composer:disabled { color: @muted; border-color: @hair; }
QFrame#composerBox { background: @sunken; border: 1px solid @strong; border-radius: 10px; }
QFrame#composerBox:hover { border-color: @muted; }
QFrame#composerBox[focus="true"] { border-color: @accent; }
QFrame#composerBox QPlainTextEdit#composer { background: transparent; border: none; padding: 6px 6px 2px 6px; }
QFrame#composerBox QToolButton { min-width: 26px; min-height: 26px; border-radius: 6px; }
QLabel#composerHint { color: @muted; font-size: @fs11px; }
QPushButton[role="stop"] { background: @raised; border-color: @crit; color: @ink; font-weight: 600; }
QPushButton[role="stop"]:hover { background: @sel; }
QToolButton#assistantBadge { padding: 0 8px; color: @ink2; }
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
            c = QColor(self.status(key))
        else:
            c = QColor(key)
        if alpha is not None:
            c.setAlphaF(alpha)
        return c

    def status(self, tone: str) -> str:
        """A status colour for text and marks on this theme's panels (darker on the light theme)."""
        return (STATUS_LIGHT if self.name == 'light' else STATUS).get(tone, STATUS.get(tone, self.tok['muted']))

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
        tok['crit'] = self.status('critical')
        tok['warn'] = self.status('warning')
        tok['good'] = self.status('good')
        for n in (11, 12, 13, 14, 18, 22, 26):
            tok[f'fs{n}'] = str(self.px(n))
        tok['icon_chevr'] = icons.file_for('chev-r', self.tok['ink2'], 12)
        tok['icon_chevd'] = icons.file_for('chev-d', self.tok['ink2'], 12)
        tok['icon_caret'] = icons.file_for('caret', self.tok['ink2'], 8)
        tok['icon_checkinv'] = icons.file_for('check-inv', self.tok['on_accent'], 12)
        tok['icon_dashinv'] = icons.file_for('dash', self.tok['on_accent'], 12)
        for k, v in sorted(tok.items(), key=lambda kv: -len(kv[0])):
            qss = qss.replace('@' + k, v)
        pal = app.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(self.tok['panel']))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(self.tok['ink']))
        pal.setColor(QPalette.ColorRole.Base, QColor(self.tok['raised']))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(self.tok['alt']))
        pal.setColor(QPalette.ColorRole.Text, QColor(self.tok['ink']))
        pal.setColor(QPalette.ColorRole.Button, QColor(self.tok['panel']))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(self.tok['ink']))
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
        pal.setColor(QPalette.ColorRole.Highlight, QColor(self.tok['accent_soft']))
        pal.setColor(QPalette.ColorRole.Link, QColor(self.tok['accent_hi']))
        pal.setColor(QPalette.ColorRole.Accent, QColor(self.tok['accent']))
        app.setPalette(pal)
        f = QFont('IBM Plex Sans')
        f.setPixelSize(self.px(13))
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
    if v is None:
        return '—'
    s = f'{v:.{dp}f}'.rstrip('0').rstrip('.') if isinstance(v, float) else str(v)
    return f'{s} {unit}'.rstrip() if unit else s
