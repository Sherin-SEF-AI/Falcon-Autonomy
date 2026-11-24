"""Design tokens, QSS and fonts: the test-bench look (a dark instrument: warm graphite, thin hairlines instead of
boxed cards, small capitals for section names, monospaced numbers, one amber accent for the main action and for
what needs you, blue for met and finished, orange for failed and not met, and a word beside every status colour).
Token names map 1:1 to @placeholders in the QSS template."""
from __future__ import annotations

import math
import os

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')

TOKENS = {
    'dark': {
        'base': '#121311', 'chrome': '#0e0f0d', 'panel': '#121311', 'raised': '#1a1b17', 'sunken': '#161714',
        'sel': '#24231c', 'nav_sel': '#1f211c', 'tab_sel': '#26281f',
        'hair': '#262823', 'soft': '#1f211c', 'strong': '#34372f', 'ink': '#eceae3', 'ink2': '#bdbab0',
        'muted': '#8f8c82', 'off': '#5e6158',
        'accent': '#e8b04a', 'accent_hover': '#f3cb7a', 'accent_hi': '#e8b04a', 'accent_soft': '#2a2416',
        'accent_line': '#5c4720', 'on_accent': '#1a1405', 'alt': '#151613', 'dialog': '#1a1b17',
        'viewport': '#161714', 'grid1': '#1f211c', 'grid2': '#2a2c26', 'body': '#c9c6bc', 'glass': '#3d3f38',
        'ground': '#7d7b72', 'sky': '#43453e',
    },
    'light': {
        'base': '#f4f2ec', 'chrome': '#ebe8e0', 'panel': '#f4f2ec', 'raised': '#fbfaf6', 'sunken': '#ffffff',
        'sel': '#ebe3cd', 'nav_sel': '#e1ddd2', 'tab_sel': '#e6e2d7',
        'hair': '#dcd8cc', 'soft': '#e6e2d7', 'strong': '#c6c1b3', 'ink': '#1b1c18', 'ink2': '#46463f',
        'muted': '#6f6c63', 'off': '#a3a097',
        'accent': '#e8b04a', 'accent_hover': '#f3cb7a', 'accent_hi': '#8a5d0a', 'accent_soft': '#f3e7c8',
        'accent_line': '#d9b871', 'on_accent': '#1a1405', 'alt': '#efede6', 'dialog': '#fbfaf6',
        'viewport': '#fbfaf6', 'grid1': '#ebe8e0', 'grid2': '#dcd8cc', 'body': '#e2e0d9', 'glass': '#5b5d56',
        'ground': '#9c998f', 'sky': '#5f615a',
    },
}
CAMERA = {
    'dark': ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
    'light': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
}
# Status: blue = met / finished / healthy, amber = needs you, orange = failed / not met. Blue and orange also differ in
# lightness, and every status mark carries a word, so none of them is read by hue alone.
STATUS = {'good': '#86b4f0', 'warning': '#e8b04a', 'serious': '#f0a06e', 'critical': '#f08a4e'}
STATUS_LIGHT = {'good': '#2c5fa8', 'warning': '#8a5d0a', 'serious': '#a95a26', 'critical': '#b4481c'}   # text on paper
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
       '#1c5cab', '#184f95', '#104281', '#0d366b']
HEAT = ['#1a1b17', '#5c4720', '#a87b2e', '#e8b04a']           # counts (failures in a region): surface -> amber
HEAT_LIGHT = ['#fbfaf6', '#f1dfb4', '#dfb565', '#b9862a']
REDUNDANCY = ['#86b6ef', '#2a78d6', '#184f95']
DIVERGING = ['#c4551f', '#e07f4c', '#eeb08e', '#3a3b36', '#86b6ef', '#3987e5', '#1c5cab']
DIVERGING_LIGHT_MID = '#e1e0d9'

QSS = r'''
* { font-family: "IBM Plex Sans"; font-size: @fs13px; color: @ink; }
QMainWindow, QMainWindow::separator { background: @base; }
QMainWindow::separator { width: 1px; height: 1px; background: @hair; }
QDialog { background: @dialog; }
QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; }
QDockWidget::title { background: @chrome; height: 26px; padding-left: 10px; border-bottom: 1px solid @hair; text-align: left; }
QWidget[dockBody="true"], QWidget#dockBody { background: @chrome; }
QWidget#panel { background: @panel; }
QWidget#base { background: @base; }
QWidget#topBar { background: @chrome; border-bottom: 1px solid @hair; }
QLabel#brand { font-family: "IBM Plex Mono"; font-weight: 600; font-size: @fs13px; color: @ink; letter-spacing: 2px; }
QLabel#crumbSep { color: @off; }
QLabel#projectName { font-weight: 500; color: @ink2; }
QLabel#projectPath { color: @muted; font-family: "IBM Plex Mono"; font-size: @fs11px; }
QToolButton#searchBox { background: @sunken; border: 1px solid @hair; border-radius: 4px; color: @muted; text-align: left;
    padding: 0 10px; min-height: 28px; font-size: @fs12px; }
QToolButton#searchBox:hover { border-color: @strong; color: @ink2; }
QFrame#topChip { background: transparent; border: 1px solid @hair; border-radius: 4px; }
QFrame#topChip:hover { border-color: @strong; }
QLabel#chipText { color: @ink2; font-family: "IBM Plex Mono"; font-size: @fs11px; }
QWidget#pageHeader { background: @base; border-bottom: 1px solid @soft; }
QLabel#pageTitle { font-size: @fs18px; font-weight: 600; }
QMenuBar { background: transparent; min-height: 26px; border: none; }
QMenuBar::item { padding: 3px 8px; color: @muted; background: transparent; border-radius: 4px; font-size: @fs12px; }
QMenuBar::item:selected, QMenuBar::item:pressed { background: @raised; color: @ink; }
QMenu { background: @raised; border: 1px solid @strong; border-radius: 4px; padding: 4px 0; }
QMenu::item { min-height: 26px; padding: 0 28px 0 28px; color: @ink; border-radius: 3px; margin: 0 4px; }
QMenu::item:selected { background: @sel; }
QMenu::item:disabled { color: @off; }
QMenu::separator { height: 1px; background: @hair; margin: 4px 0; }
QMenu::indicator { width: 12px; height: 12px; left: 9px; }
QMenu::right-arrow { image: url(@icon_chevr); width: 12px; height: 12px; right: 8px; }
QToolBar { background: @base; border: none; border-bottom: 1px solid @hair; spacing: 2px; padding: 0 4px; }
QToolBar::separator { width: 1px; background: @hair; margin: 7px 5px; }
QToolButton { min-width: 22px; min-height: 22px; border: none; border-radius: 4px; background: transparent; padding: 0; }
QToolBar QToolButton { min-width: 28px; min-height: 28px; }
QToolButton:hover { background: @raised; }
QToolButton:checked, QToolButton:pressed { background: @tab_sel; }
QToolButton:disabled { color: @off; }
QToolButton::menu-indicator { image: none; width: 0; }
QToolButton[popupMode="1"] { padding-right: 10px; }
QToolButton::menu-button { border: none; width: 10px; }
QToolButton::menu-arrow { image: url(@icon_caret); width: 8px; height: 8px; }
QToolButton[text-button="true"] { padding: 0 8px; color: @ink2; }
QToolButton[text-button="true"]:hover { color: @ink; }
QLineEdit, QAbstractSpinBox, QComboBox { min-height: 24px; max-height: 28px; padding: 0 8px; background: @sunken;
    border: 1px solid @hair; border-radius: 4px; selection-background-color: @accent_soft; selection-color: @ink; }
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover { border-color: @strong; }
QAbstractSpinBox { qproperty-buttonSymbols: NoButtons; qproperty-alignment: "AlignLeft|AlignVCenter";
    font-family: "IBM Plex Mono"; font-size: @fs12px; }
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus { border-color: @accent; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled { color: @muted; border-color: @soft; background: transparent; }
QLineEdit[readOnly="true"] { background: transparent; border-color: transparent; }
QAbstractSpinBox[mixed="true"] { color: @muted; }
QLineEdit[mono="true"] { font-family: "IBM Plex Mono"; font-size: @fs12px; }
QComboBox { padding-right: 20px; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox::down-arrow { image: url(@icon_chevd); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: @raised; border: 1px solid @strong; selection-background-color: @sel;
    outline: none; padding: 3px; }
QPushButton { min-height: 26px; padding: 0 12px; border: 1px solid @strong; border-radius: 4px; background: transparent;
    color: @ink; }
QPushButton:hover { background: @raised; border-color: @muted; }
QPushButton:pressed { background: @tab_sel; }
QPushButton:focus { border-color: @accent; }
QPushButton:checked { background: @tab_sel; border-color: @strong; color: @ink; }
QPushButton[role="primary"] { background: @accent; border-color: @accent; color: @on_accent; font-weight: 600; }
QPushButton[role="primary"]:hover { background: @accent_hover; border-color: @accent_hover; }
QPushButton[role="primary"]:pressed { background: @accent_line; border-color: @accent_line; color: @ink; }
QPushButton[role="primary"]:focus { border-color: @ink; }
QPushButton[role="ghost"] { border-color: transparent; background: transparent; color: @ink2; }
QPushButton[role="ghost"]:hover { background: @raised; color: @ink; }
QPushButton[role="link"] { border-color: transparent; background: transparent; text-align: left; padding: 0 4px; color: @accent_hi; }
QPushButton[role="link"]:hover { color: @ink; }
QPushButton[role="ghost"]:focus { border-color: @accent; }
QPushButton[role="destructive"] { border-color: @crit; color: @crit; }
QPushButton[role="destructive"]:hover { background: @crit; color: @on_accent; }
QPushButton:disabled { color: @off; border-color: @soft; background: transparent; }
QPushButton[role="primary"]:disabled { background: @raised; border-color: @hair; color: @off; }
QPushButton[seg="true"] { border-radius: 0; border-right-width: 0; padding: 0 10px; color: @muted; background: @sunken;
    border-color: @hair; font-size: @fs12px; }
QPushButton[seg="true"][segpos="first"] { border-top-left-radius: 4px; border-bottom-left-radius: 4px; }
QPushButton[seg="true"][segpos="last"] { border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-right-width: 1px; }
QPushButton[seg="true"][segpos="only"] { border-radius: 4px; border-right-width: 1px; }
QPushButton[seg="true"]:checked { background: @tab_sel; color: @ink; border-color: @strong; font-weight: 600; }
QTabBar { background: transparent; qproperty-drawBase: 0; }
QTabBar::tab { height: 30px; padding: 0 12px; background: transparent; color: @muted; border: none;
    border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: @ink; border-bottom-color: @accent; font-weight: 600; }
QTabBar::tab:hover:!selected { color: @ink2; border-bottom-color: @strong; }
QTabWidget::pane { border: none; border-top: 1px solid @hair; top: -1px; }
QHeaderView { background: @base; }
QHeaderView::section { height: 26px; padding: 0 10px; background: @base; color: @muted; border: none;
    border-bottom: 1px solid @hair; font-family: "IBM Plex Mono"; font-size: @fs11px; font-weight: 500; }
QHeaderView::section:hover { color: @ink2; }
QTreeView, QTableView, QListView { background: @base; border: none; gridline-color: @soft; outline: none;
    selection-background-color: @sel; selection-color: @ink; alternate-background-color: @alt; }
QTreeView::item, QListView::item { min-height: 26px; border-bottom: 1px solid @soft; }
QTreeView::item:hover, QTableView::item:hover, QListView::item:hover { background: @raised; }
QTreeView::item:selected, QTableView::item:selected, QListView::item:selected { background: @sel; color: @ink; }
QTreeView::branch { background: transparent; border-bottom: 1px solid @soft; }
QTreeView::branch:selected { background: @sel; }
QTreeView::branch:has-children:!has-siblings:closed, QTreeView::branch:closed:has-children:has-siblings { image: url(@icon_chevr); }
QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings { image: url(@icon_chevd); }
QTableCornerButton::section { background: @base; border: none; }
QStatusBar { background: @chrome; border-top: 1px solid @hair; font-family: "IBM Plex Mono"; font-size: @fs11px; color: @muted;
    min-height: 26px; }
QStatusBar QLabel, QStatusBar QToolButton { font-family: "IBM Plex Mono"; font-size: @fs11px; color: @muted; }
QStatusBar QToolButton:hover { color: @ink; }
QStatusBar QLabel[mono="true"] { font-family: "IBM Plex Mono"; }
QStatusBar::item { border: none; }
QToolTip { background: @raised; color: @ink; border: 1px solid @strong; border-radius: 4px; padding: 5px 8px; }
QSlider { min-height: 18px; }
QSlider::groove:horizontal { height: 2px; background: @strong; border-radius: 1px; }
QSlider::handle:horizontal { width: 12px; height: 12px; margin: -5px 0; background: @ink; border: 2px solid @base; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: @accent_hover; }
QSlider::sub-page:horizontal { background: @accent; border-radius: 1px; }
QSlider::groove:vertical { width: 2px; background: @strong; border-radius: 1px; }
QSlider::handle:vertical { width: 12px; height: 12px; margin: 0 -5px; background: @ink; border: 2px solid @base; border-radius: 6px; }
QProgressBar { max-height: 3px; min-height: 3px; border: none; background: @soft; border-radius: 1px; text-align: center; }
QProgressBar::chunk { background: @accent; border-radius: 1px; }
QProgressBar[inline="true"] { max-height: 3px; min-height: 3px; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { background: @strong; border-radius: 3px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: @muted; }
QScrollBar:horizontal { height: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:horizontal { background: @strong; border-radius: 3px; min-width: 28px; }
QScrollBar::handle:horizontal:hover { background: @muted; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QCheckBox, QRadioButton { spacing: 8px; }
QCheckBox::indicator, QAbstractItemView::indicator { width: 14px; height: 14px; border: 1px solid @strong; border-radius: 3px; background: @sunken; }
QCheckBox::indicator:hover, QAbstractItemView::indicator:hover { border-color: @accent; }
QCheckBox::indicator:checked, QAbstractItemView::indicator:checked { background: @accent; border-color: @accent; image: url(@icon_checkinv); }
QCheckBox::indicator:indeterminate, QAbstractItemView::indicator:indeterminate { background: @accent; border-color: @accent; image: url(@icon_dashinv); }
QCheckBox::indicator:disabled { background: transparent; border-color: @soft; }
QCheckBox:disabled { color: @off; }
QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid @strong; border-radius: 7px; background: @sunken; }
QRadioButton::indicator:hover { border-color: @accent; }
QRadioButton::indicator:checked { border-color: @accent; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 @accent, stop:0.45 @accent, stop:0.55 @sunken, stop:1 @sunken); }
QSplitter::handle { background: @base; }
QSplitter::handle:hover { background: @accent_line; }
QSplitter::handle:horizontal { width: 9px; image: none; border-left: 4px solid @base; border-right: 4px solid @base; background: @soft; }
QSplitter::handle:vertical { height: 9px; border-top: 4px solid @base; border-bottom: 4px solid @base; background: @soft; }
QSplitter::handle:horizontal:hover, QSplitter::handle:vertical:hover { background: @accent_line; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QLabel { background: transparent; }
QLabel[muted="true"] { color: @muted; }
QLabel[ink2="true"] { color: @ink2; }
QLabel[mono="true"] { font-family: "IBM Plex Mono"; font-size: @fs12px; }
QLabel[small="true"] { font-size: @fs12px; }
QLabel[h="true"] { font-family: "IBM Plex Mono"; font-size: @fs11px; font-weight: 600; color: @muted; }
QLabel[h="true"][small="true"] { font-family: "IBM Plex Sans"; font-size: @fs12px; color: @ink; }
QLabel[title="true"] { font-weight: 600; font-size: @fs14px; }
QLabel[page="true"] { font-weight: 600; font-size: @fs18px; }
QLabel[crit="true"] { color: @crit; }
QLabel[cap="true"] { color: @muted; font-family: "IBM Plex Mono"; font-size: @fs11px; font-weight: 500; }
QLabel[sect="true"] { color: @muted; font-family: "IBM Plex Mono"; font-size: @fs11px; font-weight: 600; }
QLabel[kpi="true"] { font-family: "IBM Plex Mono"; font-size: @fs22px; font-weight: 500; }
QLabel[unit="true"] { color: @muted; font-family: "IBM Plex Mono"; font-size: @fs11px; }
QLabel[chip="true"] { background: transparent; border: 1px solid @strong; border-radius: 3px; padding: 0 6px; font-size: @fs11px;
    color: @ink2; min-height: 18px; max-height: 18px; font-family: "IBM Plex Mono"; }
QLabel[chip="true"][tone="good"] { color: @good; border-color: @good_line; }
QLabel[chip="true"][tone="warning"] { color: @warn; border-color: @accent_line; }
QLabel[chip="true"][tone="critical"] { color: @crit; border-color: @crit_line; }
QLabel[chip="true"][tone="accent"] { color: @accent_hi; border-color: @accent_line; background: @accent_soft; }
QFrame[hair="true"] { background: @hair; border: none; }
QTextEdit, QPlainTextEdit { background: @base; border: none; font-family: "IBM Plex Mono"; font-size: @fs12px;
    selection-background-color: @accent_soft; }
QTextBrowser { font-family: "IBM Plex Sans"; font-size: @fs13px; }
QWidget#errorBox { border: 1px solid @crit; border-radius: 4px; }
QWidget#card { background: @raised; border: 1px solid @hair; border-radius: 4px; }
QFrame#section { background: transparent; border: none; border-top: 1px solid @hair; border-radius: 0; }
QFrame#section QFrame#section { background: transparent; border: none; border-top: 1px solid @soft; border-radius: 0; }
QFrame#kpi { background: transparent; border: none; border-left: 1px solid @hair; border-radius: 0; }
QFrame#banner { background: @accent_soft; border: none; border-left: 2px solid @accent; border-radius: 0; }
QFrame#callout { background: @raised; border: none; border-left: 2px solid @strong; border-radius: 0; }
QFrame#callout[tone="warning"] { border-left-color: @warn; }
QFrame#callout[tone="critical"] { border-left-color: @crit; }
QFrame#callout[tone="good"] { border-left-color: @good; }
QListWidget#cardList { background: transparent; border: none; border-top: 1px solid @hair; padding: 0; }
QListWidget#cardList::item { border-radius: 0; padding-left: 6px; border-bottom: 1px solid @soft; }
QListWidget#cardList::item:selected { background: @sel; color: @ink; }
QListWidget#steps { background: @chrome; border: none; border-right: 1px solid @hair; }
QListWidget#steps::item { min-height: 40px; padding-left: 0; }
QListWidget#steps::item:selected { background: @nav_sel; }
QWidget#side { background: @chrome; border-right: 1px solid @hair; }
QListWidget#nav { background: transparent; border: none; padding: 4px 8px; }
QListWidget#nav::item { min-height: 30px; padding-left: 8px; color: @ink2; border-radius: 4px; border: none; }
QListWidget#nav::item:selected { background: transparent; color: @ink; }
QListWidget#nav::item:hover:!selected { background: @soft; }
QToolButton#navFoot { color: @off; font-size: @fs11px; min-height: 28px; padding: 0 16px; text-align: left; background: transparent; border: none; border-top: 1px solid @hair; border-radius: 0; font-family: "IBM Plex Mono"; }
QToolButton#navFoot:hover { color: @ink2; }
QToolButton#jobsBadge { padding: 0 8px; color: @muted; }
QToolButton#jobsBadge[busy="true"] { color: @accent_hi; font-weight: 600; }
QWidget#homeCard { background: transparent; border: none; border-top: 1px solid @hair; border-radius: 0; }
QFrame#suggestChip { background: transparent; border: 1px solid @accent_line; border-radius: 4px; }
QFrame#suggestChip:hover { border-color: @accent; background: @accent_soft; }
QToolButton#suggestAct { border: none; background: transparent; color: @ink; padding: 2px 4px; min-height: 22px; }
QToolButton#suggestDismiss { border: none; background: transparent; color: @muted; padding: 0 6px; }
QToolButton#suggestDismiss:hover { color: @ink; }
QLabel[stat="true"] { font-family: "IBM Plex Mono"; font-size: @fs14px; font-weight: 500; }
QLineEdit#paletteInput { min-height: 34px; max-height: 36px; font-size: @fs14px; }
QLabel#bubbleUser { background: @raised; border: 1px solid @hair; border-radius: 4px; }
QFrame#toolCard { background: transparent; border: none; border-left: 2px solid @hair; border-radius: 0; }
QFrame#toolCard QToolButton { text-align: left; color: @muted; font-family: "IBM Plex Mono"; font-size: @fs11px; }
QPlainTextEdit#composer { background: @sunken; border: 1px solid @hair; border-radius: 4px; padding: 4px 6px; font-family: "IBM Plex Sans"; font-size: @fs13px; }
QPlainTextEdit#composer:focus { border-color: @accent; }
QPlainTextEdit#composer:disabled { color: @muted; border-color: @soft; }
QFrame#composerBox { background: @sunken; border: 1px solid @hair; border-radius: 4px; }
QFrame#composerBox:hover { border-color: @strong; }
QFrame#composerBox[focus="true"] { border-color: @accent; }
QFrame#composerBox QPlainTextEdit#composer { background: transparent; border: none; padding: 6px 6px 2px 6px; }
QFrame#composerBox QToolButton { min-width: 26px; min-height: 26px; border-radius: 4px; }
QLabel#composerHint { color: @muted; font-size: @fs11px; }
QPushButton[role="stop"] { background: transparent; border-color: @crit; color: @ink; font-weight: 600; }
QPushButton[role="stop"]:hover { background: @raised; }
QToolButton#assistantBadge { padding: 0 8px; color: @muted; }
QLineEdit[changed="true"], QAbstractSpinBox[changed="true"], QComboBox[changed="true"], QPlainTextEdit[changed="true"] {
    border-color: @accent_line; }
QCheckBox[changed="true"] { color: @accent_hi; }
QFrame#applyBar { background: @raised; border: none; border-top: 1px solid @accent_line; }
QFrame#attention { background: transparent; border: none; border-left: 2px solid @accent; }
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
        dark = self.name == 'dark'
        tok['good_line'] = '#3a5677' if dark else '#9db6d9'
        tok['crit_line'] = '#7a4a2e' if dark else '#dea88c'
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
        pal.setColor(QPalette.ColorRole.Window, QColor(self.tok['base']))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(self.tok['ink']))
        pal.setColor(QPalette.ColorRole.Base, QColor(self.tok['base']))
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
        pal.setColor(QPalette.ColorRole.Highlight, QColor(self.tok['sel']))
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


def caps(lb, px: int = 11):
    """Set a label in small capitals (mono, tracked): a section's name or a readout's caption. The text itself is
    unchanged, so code and tests still find it by its words."""
    f = lb.font()
    f.setCapitalization(QFont.Capitalization.AllUppercase)
    f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
    lb.setFont(f)
    return lb


def heat_color(n: int) -> tuple[QColor, QColor]:
    """(fill, text) for a count on the amber heat ramp: 0 is the surface, 1, 2 and 3 or more step up to amber. Text is
    dark on the two brightest steps (contrast stays above 4.5:1 on each)."""
    dark = THEME.name == 'dark'
    ramp = HEAT if dark else HEAT_LIGHT
    i = max(0, min(int(n), len(ramp) - 1))
    if dark:
        return QColor(ramp[i]), QColor('#1a1405' if i >= 2 else THEME.tok['ink'])
    return QColor(ramp[i]), QColor(THEME.tok['ink'])


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
