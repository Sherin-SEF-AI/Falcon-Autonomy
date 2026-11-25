"""Shared pieces of the 4.2 robustness and compliance sections (Fault tolerance, Regulation packs, Latency budget):
the world and rig picker — with a warning when the fake simulator would take minutes per run on the chosen rig —, the
hand-off of a result to the safety case ("Attach evidence"), and the run rows' context menu (Send to ▸).

Why the warning. The fake `carla` module the tests and demos use (tests/fake_carla) ray-casts every camera pixel in
numpy: measured on the development machine, one 1280 × 720 frame of a scene with six boxes takes about 1.2 s
(320 × 180: 0.04 s; 64 × 36: 0.002 s), and a semantic label image is a second cast of the same size. A project's
default rig (one 1280 × 720 camera with semantic labels) therefore spends minutes per run on the fake, and a fault
tolerance campaign or a latency budget (dozens of runs) looks stalled. The cost is the test double's, not Falcon's
and not a real CARLA's: the picker says so and suggests a small rig when the fake is in use."""
from __future__ import annotations

import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from .widgets import callout, label

FAKE_S_PER_MPX = 1.3          # the fake's ray-cast render, seconds per megapixel-frame (measured, module doc)
WARN_S_PER_STEP = 0.1         # above this the warning shows (a 20 fps, 15 s scenario: 30 s per run)


def camera_mpx(rig: dict | None, labels: bool = True) -> float:
    """Megapixels the rig's cameras render per step (each label image counted as another frame when labels)."""
    total = 0.0
    for s in (rig or {}).get('sensors') or []:
        if str(s.get('type', '')).startswith('camera'):
            n = 1 + (len(s.get('labels') or []) if labels else 0)
            total += float(s.get('width') or 0) * float(s.get('height') or 0) * n / 1e6
    return total


def fake_cost_note(rig: dict | None, labels: bool = True) -> str:
    """'' when the rig is cheap on the fake simulator, else a short warning with the estimate."""
    mpx = camera_mpx(rig, labels)
    secs = mpx * FAKE_S_PER_MPX
    if secs < WARN_S_PER_STEP:
        return ''
    return (f'The fake simulator renders this rig\'s {mpx:.2f} Mpx per step in Python: about {secs:.1f} s per step, '
            f'minutes per run. Choose a small rig (320 × 180, no labels) for runs on the fake.')


def fake_in_use(win) -> str:
    """The fake carla folder the window runs with ('' on a real simulator)."""
    return getattr(win, 'fake_carla', '') or getattr(getattr(win, 'service', None), 'fake_carla', '') or ''


class RigWorldPicker(QWidget):
    """World and rig of the project for a section's runs: the active ones (win.context) preselected; choosing one
    sets it as active. labels: whether the section's runs render the rig's label images (for the fake's cost)."""

    changed = pyqtSignal()

    def __init__(self, win, labels: bool = True, what: str = '', parent=None):
        super().__init__(parent)
        self.win, self.labels, self.what = win, labels, what
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.world_box = QComboBox()
        self.rig_box = QComboBox()
        self.world_box.setToolTip('The world the runs drive in (the project\'s worlds)')
        self.rig_box.setToolTip('The rig the runs carry (the project\'s rigs)')
        for lab, box in (('World', self.world_box), ('Rig', self.rig_box)):
            row.addWidget(label(lab, muted=True))
            row.addWidget(box, 1)
            box.currentIndexChanged.connect(self._picked)
        v.addLayout(row)
        self.warning = callout('', 'warning')
        self.warning.setVisible(False)
        v.addWidget(self.warning)
        self.refresh()

    def _doc(self) -> dict:
        try:
            return self.win.project.doc
        except AttributeError:
            return {}

    def refresh(self):
        doc = self._doc()
        ctx = getattr(self.win, 'context', None)
        for box, items, key in ((self.world_box, doc.get('worlds') or [], 'world'),
                                (self.rig_box, doc.get('rigs') or [], 'rig')):
            cur = box.currentData() or (ctx.get(key) if ctx is not None else '')
            box.blockSignals(True)
            box.clear()
            for it in items:
                box.addItem(it.get('name') or it['id'], it['id'])
                box.setItemData(box.count() - 1, it['id'], 3)          # the id as the tooltip
            k = box.findData(cur)
            box.setCurrentIndex(k if k >= 0 else 0)
            box.blockSignals(False)
        self.check()

    def world(self) -> str:
        return self.world_box.currentData() or ''

    def rig(self) -> str:
        return self.rig_box.currentData() or ''

    def rig_doc(self) -> dict | None:
        return next((r for r in self._doc().get('rigs') or [] if r['id'] == self.rig()), None)

    def check(self) -> str:
        """Show the fake simulator's cost warning when it applies; returns its text."""
        text = fake_cost_note(self.rig_doc(), self.labels) if fake_in_use(self.win) else ''
        if text and self.what:
            text += f' ({self.what})'
        self.warning.label.setText(text)
        self.warning.setVisible(bool(text))
        self.warning.setToolTip('The fake carla module ray-casts every pixel in numpy (about 1.2 s per 1280 × 720 '
                                'frame, measured); a real CARLA renders on the GPU. Nothing is wrong with the run: '
                                'it is slow on the test double.' if text else '')
        return text

    def _picked(self, _i=None):
        ctx = getattr(self.win, 'context', None)
        if ctx is not None:
            ctx.set(world=self.world(), rig=self.rig())
        self.check()
        self.changed.emit()


def attach_evidence(win, kind: str, ref: str, claim: dict | None, text: str) -> str:
    """Add a result to the safety case as a new Solution (the GSN section's undoable attach) and show it there;
    returns the new node's id ('' when the build has no safety case section or it refused, with the reason shown
    there)."""
    page = win.page('reports') if hasattr(win, 'page') else None
    gsn = getattr(page, 'gsn', None)
    if gsn is None:
        return ''
    if hasattr(win, 'go'):
        win.go('reports')
    return gsn.attach(kind, ref, claim, text) or ''


def run_menu(win, view, run_of, exclude: str = ''):
    """A right-click Send to ▸ menu on a tree's rows that carry a run (run_of(item) -> path or '')."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QCursor
    from .context import send_to_menu
    view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def show(pos):
        it = view.itemAt(pos)
        path = run_of(it) if it is not None else ''
        if not path:
            return
        m = send_to_menu(win, path, exclude=exclude, parent=view)
        m.setTitle('Send to ▸')
        menu = m if not m.isEmpty() else None
        if menu is not None:
            menu.exec(QCursor.pos())
    view.customContextMenuRequested.connect(show)


def set_context_run(win, path: str):
    """Make a run the active one (the top bar's pill)."""
    ctx = getattr(win, 'context', None)
    if ctx is not None and path and os.path.isdir(path):
        ctx.set(run=path)


def flag(text: str, tone: str, actions) -> 'QWidget':
    """ui.widgets.action_callout with the text taking the room and the buttons at their own width (a flag's next
    steps beside it, not stretched across the row)."""
    from PyQt6.QtWidgets import QSizePolicy
    from .widgets import action_callout
    f = action_callout(text, tone, actions)
    f.layout().setStretch(0, 1)
    for b in f.buttons:
        b.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return f
