"""Run-level pieces the Data, Home and Replay pages share (4.2): a run's fidelity read once per run.json version, a
chip slot that can change its level, the like-for-like notice of two results, and the run context a selected run
sets (win.context: the run, and its seed, rig, scenario and world when run.json records them)."""
from __future__ import annotations

import json
import os

from PyQt6.QtWidgets import QHBoxLayout, QWidget

_CACHE: dict = {}                     # run.json path -> ((mtime, size), level, run.json's context fields)


def _read(path: str):
    p = os.path.join(path, 'run.json') if os.path.isdir(path) else path
    try:
        st = os.stat(p)
    except OSError:
        return 'unknown', {}
    key = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(p)
    if hit and hit[0] == key:
        return hit[1], hit[2]
    from ..core import fidelity as F
    try:
        with open(p, encoding='utf-8') as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        doc = None
    level = F.of_doc(doc)

    def d(x):                                   # a block a worker may also write as a plain id (rig: "rig-1")
        return x if isinstance(x, dict) else {}

    def text(x):
        return x if isinstance(x, (str, int, float)) and not isinstance(x, bool) else ''
    prov = d(d(doc).get('provenance'))
    st_ = d(prov.get('settings'))
    proj = d(st_.get('project'))
    rig, scn = st_.get('rig'), st_.get('scenario')
    seed = st_.get('seed', prov.get('seed'))
    ctx = {'seed': str(text(seed)),
           'rig': str(text(proj.get('rig')) or text(d(rig).get('id')) or text(rig)),
           'scenario': str(text(d(scn).get('id')) or text(d(scn).get('name')) or text(scn)),
           'world': str(text(proj.get('world')) or text(st_.get('town')))}
    _CACHE[p] = (key, level, ctx)
    return level, ctx


def fidelity_of(path: str) -> str:
    """core.fidelity.of_run, cached on run.json's modification time and size (a run list re-reads nothing)."""
    return _read(path)[0] if path else 'unknown'


def set_run_context(win, path: str):
    """The run as the window's active run, with the seed, rig, scenario and world its run.json records (each cleared
    when it records none, so a pill never shows another run's)."""
    from .context import context_of
    if not path:
        return
    ctx = dict(_read(path)[1])
    context_of(win).set(run=path, **ctx)


class FidelitySlot(QWidget):
    """A fidelity chip whose level can change (ui.widgets.fidelity_chip, rebuilt on set)."""

    def __init__(self, level: str = 'unknown', parent=None):
        super().__init__(parent)
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(0, 0, 0, 0)
        self.chip = None
        self.level = None
        self.set(level)

    def set(self, level: str):
        from .widgets import fidelity_chip
        level = level or 'unknown'
        if level == self.level and self.chip is not None:
            return
        if self.chip is not None:
            self.h.removeWidget(self.chip)
            self.chip.hide()                       # gone at once, not when the event loop deletes it
            self.chip.setParent(None)
            self.chip.deleteLater()
        self.level = level
        self.chip = fidelity_chip(level)
        self.h.addWidget(self.chip)

    def text(self) -> str:
        return self.chip.text() if self.chip is not None else ''


def likeness(levels) -> tuple[str, str]:
    """How results of these levels compare: ('critical', why) when two are not like for like, ('muted', why) when
    one does not record its fidelity, ('', '') when they agree."""
    from ..core import fidelity as F
    levels = [lv or 'unknown' for lv in levels]
    worst = ('', '')
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            ok, why = F.comparable(levels[i], levels[j])
            if not ok:                              # said once: "Not like for like: A vs B (different fidelity)"
                return 'critical', why.replace(': different fidelity, not like for like', ' (different fidelity)')
            if why and not worst[0]:
                worst = ('muted', why)
    return worst


def fit_rows(tree, n: int):
    """Size a list to show exactly its n rows: no half row cut off at the bottom, no inner scroll bar."""
    rh = tree.sizeHintForRow(0) if n else 0
    h = tree.header().sizeHint().height() + n * max(rh, 1) + 2 * tree.frameWidth() + 2
    tree.setFixedHeight(max(h, 60))
