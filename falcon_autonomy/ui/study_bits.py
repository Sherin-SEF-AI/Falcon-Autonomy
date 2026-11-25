"""Small pieces the study sections share (4.2): a fidelity chip that can change (FidelitySlot), a row of compact
key-value badges (Badges) and the run hand-offs of a run table (link_runs: selecting a row sets the window's active
run, its context menu offers *Send to ▸* and the table's own actions)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QWidget

from .widgets import FlowLayout, chip, fidelity_chip, label


class FidelitySlot(QWidget):
    """A fidelity chip (ui.widgets.fidelity_chip) beside a result, swapped when the result changes; an optional
    caption before it ("Runs", "Screen") and an optional sentence after it (a mixed-fidelity warning)."""

    def __init__(self, caption: str = '', parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.caption = label(caption, muted=True, small=True) if caption else None
        if self.caption is not None:
            h.addWidget(self.caption)
        self.chip: QLabel | None = None
        self.level = ''
        self.note = label('', small=True, wrap=True)
        self.note.hide()
        h.addWidget(self.note, 1)
        self.set('unknown')

    def set(self, level: str, note: str = '', tone: str = ''):
        from .theme import STATUS, THEME
        level = level or 'unknown'
        if self.chip is None or level != self.level:
            if self.chip is not None:
                self.layout().removeWidget(self.chip)
                self.chip.hide()                         # it paints until deleted otherwise
                self.chip.deleteLater()
            self.chip = fidelity_chip(level)
            self.layout().insertWidget(1 if self.caption is not None else 0, self.chip, 0,
                                       Qt.AlignmentFlag.AlignVCenter)
            self.level = level
        self.note.setText(note)
        self.note.setStyleSheet(f'color: {THEME.status(tone)}' if tone in STATUS else '')
        self.note.setVisible(bool(note))
        self.setToolTip(note)


class Badges(QWidget):
    """Compact key-value badges in a wrapping row (what a paragraph of prose said, as facts): set([(key, value,
    tooltip, tone), …]). The tooltip keeps the long form."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self, 6)
        self.items: list = []

    def set(self, items):
        while self.flow.count():
            it = self.flow.takeAt(0)
            if it is not None and it.widget() is not None:
                it.widget().deleteLater()
        self.items = []
        for x in items:
            key, value = x[0], x[1]
            tip = x[2] if len(x) > 2 else ''
            tone = x[3] if len(x) > 3 else ''
            c = chip(f'{key}  {value}' if key else str(value), tone)
            c.setToolTip(tip)
            self.flow.addWidget(c)
            self.items.append(c)
        self.updateGeometry()

    def texts(self) -> list:
        return [c.text() for c in self.items]


def link_runs(view, win, run_of, exclude: str = '', actions=None):
    """A run table's hand-offs (ui.context): selecting a row sets the window's active run (and seed, scenario when
    run_of's row says them); the context menu lists `actions(item)` → [(text, slot)] then *Send to ▸* (every
    registered target but `exclude`) and, when the view had ui.assistant_dock.add_ask_menu's menu, its *Ask about this*
    (one menu, not two). run_of(item) -> run path or '' (or a dict {'run', 'seed', 'scenario'})."""
    from .context import context_of, send_to_menu

    def info(it):
        x = run_of(it) if it is not None else None
        return x if isinstance(x, dict) else {'run': x or ''}

    def selected():
        x = info(view.currentItem())
        if x.get('run') and win is not None and hasattr(win, 'context'):
            context_of(win).set(**{k: x[k] for k in ('run', 'seed', 'scenario') if x.get(k) not in (None, '')})

    def menu(pos):
        it = view.itemAt(pos)
        if it is None:
            return
        view.setCurrentItem(it)
        m = build_menu(it)
        if m is not None:
            m.exec(view.viewport().mapToGlobal(pos))

    def build_menu(it):
        m = QMenu(view)
        for text, slot in (actions(it) if actions else ()):
            m.addAction(text).triggered.connect(lambda _=False, s=slot: s())
        run = info(it).get('run')
        if run and win is not None:
            if m.actions():
                m.addSeparator()
            sub = send_to_menu(win, run, exclude=exclude, parent=m)
            sub.setTitle('Send to')
            m.addMenu(sub)
        ask = getattr(view, '_ask_target', None)           # ui.assistant_dock.add_ask_menu's "Ask about this"
        t = ask(it) if ask is not None else None
        if t and win is not None and hasattr(win, 'ask_about'):
            if m.actions():
                m.addSeparator()
            kind, ids = t
            m.addAction('Ask about this').triggered.connect(lambda _=False: win.ask_about(kind, **ids))
        return m if m.actions() else None

    view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    if getattr(view, '_ask_target', None) is not None:  # one menu, not two: the ask action moves into this one
        try:
            view.customContextMenuRequested.disconnect()
        except TypeError:
            pass
    view.customContextMenuRequested.connect(menu)
    view.currentItemChanged.connect(lambda *_: selected())
    view.run_menu = build_menu                        # tests read the menu without showing it
    return build_menu
