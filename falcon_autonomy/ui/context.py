"""What the user is looking at across pages, and the one-click hand-offs between pages (4.2).

RunContext (the window's `win.context`) holds the active run, seed, rig, scenario and world. A page sets what it shows
(`win.context.set(run=path, seed=3)`); the top bar shows each as a pill, and a pill's × clears it. Nothing is
remembered across sessions.

Send to…: a page that can take a run registers a target once (`register_target(win, 'replay', 'Replay', handler)`,
handler(run_path) -> None); send_to_menu(win, run_path) builds the menu of every registered target but the page it is
opened from, so a run row's context menu or a *Send to…* button reaches Replay, Rare cases, Fault tolerance, Moments
or Reports in one click. A target's handler goes to its page itself (win.go(key)) and loads the run."""
from __future__ import annotations

import os

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMenu

KEYS = ('run', 'seed', 'rig', 'scenario', 'world')
TITLES = {'run': 'Run', 'seed': 'Seed', 'rig': 'Rig', 'scenario': 'Scenario', 'world': 'World'}


class RunContext(QObject):
    changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: dict = {}

    def set(self, **kw):
        """Set (value not None) or clear (value '' or None after being set) any of KEYS; emits changed once."""
        changed = False
        for k, v in kw.items():
            if k not in KEYS:
                raise KeyError(k)
            v = '' if v is None else str(v)
            if self.values.get(k, '') != v:
                if v:
                    self.values[k] = v
                else:
                    self.values.pop(k, None)
                changed = True
        if changed:
            self.changed.emit(dict(self.values))

    def clear(self, key: str = ''):
        if key:
            self.set(**{key: ''})
        elif self.values:
            self.values = {}
            self.changed.emit({})

    def get(self, key: str, default: str = '') -> str:
        return self.values.get(key, default)

    def pill_text(self, key: str) -> str:
        v = self.values.get(key, '')
        if key == 'run':
            v = os.path.basename(v.rstrip('/')) or v
        return f'{TITLES[key]} {v}'


def context_of(win) -> RunContext:
    ctx = getattr(win, 'context', None)
    if not isinstance(ctx, RunContext):
        ctx = RunContext(win)
        win.context = ctx
    return ctx


def register_target(win, key: str, title: str, handler, tip: str = ''):
    """A page that can take a run: key is its page key (win.go(key)), title the menu text, handler(run_path)."""
    targets = getattr(win, 'send_targets', None)
    if targets is None:
        targets = win.send_targets = {}
    targets[key] = (title, handler, tip)


def targets(win) -> dict:
    return dict(getattr(win, 'send_targets', {}) or {})


def send_to_menu(win, run_path: str, exclude: str = '', parent=None) -> QMenu:
    """A *Send to* menu for one run: an action per registered target (but `exclude`), each calling its handler with
    the run and setting the run as the active one."""
    m = QMenu('Send to', parent)
    for key, (title, handler, tip) in sorted(targets(win).items(), key=lambda kv: kv[1][0]):
        if key == exclude:
            continue
        a = m.addAction(title)
        if tip:
            a.setToolTip(tip)
        a.triggered.connect(lambda _=False, h=handler: _send(win, h, run_path))
    m.setEnabled(bool(m.actions()))
    return m


def _send(win, handler, run_path: str):
    context_of(win).set(run=run_path)
    handler(run_path)
