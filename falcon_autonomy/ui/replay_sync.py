"""Replay ▸ Side by side: a baseline (A) and a candidate (B) run seen from above next to each other, driven by ONE time
slider, with the steps where they part (core.logreplay.sync) marked as ticks on it and buttons to the first, next and
previous divergence. Everything is computed once when two runs are loaded; moving the slider is an index lookup and a
repaint."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QHBoxLayout, QSizePolicy, QSlider, QStyle, QStyleOptionSlider, QVBoxLayout, QWidget)

from . import tasks
from .runfid import FidelitySlot, likeness
from .theme import THEME
from .widgets import Details, IdLabel, button, callout, label

SLOTS = {'A': 'baseline', 'B': 'candidate'}


def _slot_of(aid: str) -> int:
    return sum(map(ord, str(aid))) % 6 + 1          # a stable colour per actor id (slot 0 is kept for nothing)


class RunTopDown(QWidget):
    """One run on the shared grid, from above (CARLA's x right, y down on screen): every track's trail faint, the
    positions at the current step solid, the ego thick; the other run's ego at the same moment as a hollow ring; the
    frame's edge in the critical colour while the runs diverge there."""

    def __init__(self, slot: str):
        super().__init__()
        self.slot = slot
        self.side = None           # the run (sync's 'a' or 'b')
        self.other = None
        self.bounds = None
        self.k = 0
        self.diverging = False
        self._paths = None         # (size, [(QPainterPath, QColor, is_ego)])
        self.setMinimumSize(220, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_runs(self, side, other, bounds):
        self.side, self.other, self.bounds = side, other, bounds
        self._paths = None
        self.update()

    def set_step(self, k: int, diverging: bool):
        self.k, self.diverging = int(k), bool(diverging)
        self.update()

    def _xf(self):
        x0, x1, y0, y1 = self.bounds
        m = 18
        top = 22
        s = min((self.width() - 2 * m) / max(x1 - x0, 1.0), (self.height() - top - m - 16) / max(y1 - y0, 1.0))
        ox = m + (self.width() - 2 * m - (x1 - x0) * s) / 2
        oy = top + (self.height() - top - m - 16 - (y1 - y0) * s) / 2
        return lambda x, y: (ox + (x - x0) * s, oy + (y - y0) * s)

    def _trails(self, P):
        size = (self.width(), self.height())
        if self._paths is not None and self._paths[0] == size:
            return self._paths[1]
        out = []
        on = self.side['on']
        tracks = [('ego', on['ego'], True)] + [(aid, M, False) for aid, M in on['actors'].items()]
        for aid, M, is_ego in tracks:
            ok = ~np.isnan(M[:, 1])
            if not ok.any():
                continue
            X, Y = P(M[:, 1], M[:, 2])
            path = QPainterPath()
            pen_down = False
            for i in np.flatnonzero(ok if len(ok) < 4000 else ok & (np.arange(len(ok)) % 2 == 0)):
                if pen_down and ok[i - 1]:
                    path.lineTo(float(X[i]), float(Y[i]))
                else:
                    path.moveTo(float(X[i]), float(Y[i]))
                pen_down = True
            col = QColor(THEME.tok['ink']) if is_ego else THEME.cam(_slot_of(aid))
            out.append((path, col, is_ego))
        self._paths = (size, out)
        return out

    def paintEvent(self, ev):                                       # noqa: N802
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), QColor(THEME.tok['panel']))
        if self.diverging:
            qp.setPen(QPen(QColor(THEME.status('critical')), 2))
            qp.drawRect(QRectF(1, 1, self.width() - 2, self.height() - 2))
        qp.setPen(QColor(THEME.tok['muted']))
        head = f'{self.slot} · {SLOTS.get(self.slot, "")}'
        qp.drawText(QPointF(10, 15), head)
        if self.side is None:
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Load two runs to compare')
            return
        P = self._xf()
        for path, col, is_ego in self._trails(P):
            c = QColor(col)
            c.setAlphaF(0.35)
            qp.setPen(QPen(c, 2.4 if is_ego else 1.1))
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.drawPath(path)
        on = self.side['on']
        k = min(self.k, len(on['ego']) - 1)
        qp.setPen(Qt.PenStyle.NoPen)
        for aid, M in on['actors'].items():
            if not np.isnan(M[k, 1]):
                x, y = P(M[k, 1], M[k, 2])
                qp.setBrush(THEME.cam(_slot_of(aid)))
                qp.drawEllipse(QPointF(x, y), 3.5, 3.5)
        oe = self.other['on']['ego'] if self.other is not None else None
        if oe is not None and k < len(oe) and not np.isnan(oe[k, 1]):
            x, y = P(oe[k, 1], oe[k, 2])
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.setPen(QPen(QColor(THEME.tok['muted']), 1.4, Qt.PenStyle.DashLine))
            qp.drawEllipse(QPointF(x, y), 7.0, 7.0)
        e = on['ego']
        if not np.isnan(e[k, 1]):
            x, y = P(e[k, 1], e[k, 2])
            qp.setPen(Qt.PenStyle.NoPen)
            qp.setBrush(QColor(THEME.status('critical')) if on['collision'][k] else QColor(THEME.tok['ink']))
            qp.drawEllipse(QPointF(x, y), 5.5, 5.5)
            state = 'collision' if on['collision'][k] else ''
        else:
            state = 'not running at this time'
        ttc = on['ttc'][k]
        qp.setPen(QColor(THEME.tok['muted']))
        foot = (f'min TTC {ttc:.2f} s' if np.isfinite(ttc) else 'min TTC > 10 s') + (f' · {state}' if state else '')
        qp.drawText(QPointF(10, self.height() - 6), foot)


class DivergenceSlider(QSlider):
    """The shared time slider, with a tick at the start of every divergence (and its stretch shaded)."""

    def __init__(self):
        super().__init__(Qt.Orientation.Horizontal)
        self.spans: list = []                    # [(start, end)] grid indices
        self.setMinimumHeight(26)

    def set_spans(self, spans):
        self.spans = list(spans)
        self.update()

    def _x(self, v):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                             QStyle.SubControl.SC_SliderGroove, self)
        handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                             QStyle.SubControl.SC_SliderHandle, self)
        span = groove.width() - handle.width()
        pos = QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), int(v), max(span, 1))
        return groove.x() + handle.width() / 2 + pos

    def paintEvent(self, ev):                                        # noqa: N802
        if self.spans and self.maximum() > self.minimum():
            qp = QPainter(self)
            col = QColor(THEME.status('critical'))
            shade = QColor(col)
            shade.setAlphaF(0.22)
            h = self.height()
            for s, e in self.spans:
                x0, x1 = self._x(s), self._x(e)
                qp.fillRect(QRectF(x0, h / 2 - 5, max(x1 - x0, 2.0), 10), shade)
                qp.fillRect(QRectF(x0 - 1, 1, 2, h - 2), col)
            qp.end()
        super().paintEvent(ev)


class SyncCompare(QWidget):
    """Two runs side by side on one slider (module doc). load(a, b) computes in the background (loaded() when done);
    load_now(a, b) in the calling thread."""
    loaded = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.data = None
        self.error = ''
        self.paths = ('', '')
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self.likeness = callout('', 'critical')
        self.likeness.hide()
        v.addWidget(self.likeness)
        self.likeness_note = label('', muted=True, small=True, wrap=True)
        self.likeness_note.hide()
        v.addWidget(self.likeness_note)
        views = QHBoxLayout()                       # each run: its folder (copyable) and fidelity over its view
        views.setSpacing(8)
        self.view_a, self.view_b = RunTopDown('A'), RunTopDown('B')
        self.ids, self.fids = {}, {}
        for s, view in (('A', self.view_a), ('B', self.view_b)):
            col = QVBoxLayout()
            col.setSpacing(4)
            head = QHBoxLayout()
            head.setSpacing(6)
            self.ids[s] = IdLabel('', keep=7)
            self.ids[s].setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.fids[s] = FidelitySlot('unknown')
            head.addWidget(self.fids[s])
            head.addWidget(self.ids[s], 1)
            col.addLayout(head)
            col.addWidget(view, 1)
            views.addLayout(col, 1)
        v.addLayout(views, 1)
        row = QHBoxLayout()
        self.prev_btn = button('◀', self.prev_divergence, tip='The previous divergence')
        self.slider = DivergenceSlider()
        self.slider.valueChanged.connect(self.seek_index)
        self.next_btn = button('▶', self.next_divergence, tip='The next divergence')
        self.time_lbl = label('', mono=True)
        self.time_lbl.setMinimumWidth(80)
        row.addWidget(self.prev_btn)
        row.addWidget(self.slider, 1)
        row.addWidget(self.next_btn)
        row.addWidget(self.time_lbl)
        v.addLayout(row)
        row = QHBoxLayout()
        self.first_btn = button('Jump to first divergence', self.jump_first,
                                tip='The first step where the runs part (ticks on the slider)')
        row.addWidget(self.first_btn)
        self.here = label('', muted=True)
        self.here.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.here, 1)
        v.addLayout(row)
        self.summary = label('', ink2=True, wrap=True)       # divergences, the first, the largest gap, the rule
        v.addWidget(self.summary)
        self.details = Details(
            'A step diverges when, with both runs present: the egos are more than the gap threshold apart (box '
            'centres); one ego\'s box overlaps an actor\'s and the other\'s does not; the smallest TTC to any actor '
            '(constant velocity, 10 s horizon) differs by more than the TTC threshold with one of them within the '
            'horizon; or an actor is present in one run only (matched by id; by how many are present when the runs '
            'share fewer than half their ids). Thresholds are choices, not a standard. Both runs are put on one clock '
            'of the finer of their steps, aligned by time from each run\'s first sample (never by step index); a run '
            'shows nothing where it has no sample within half its step. Poses come from trajectories.json, else '
            'actors.jsonl, else frames.jsonl (the ego alone, a typical box: heuristic).')
        v.addWidget(self.details)
        self._set_enabled(False)

    # ------------------------------------------------------------------ loading
    def load(self, run_a: str, run_b: str, gap_m: float | None = None, ttc_s: float | None = None):
        from ..core.logreplay import sync as S
        self.paths = (run_a, run_b)
        self.data, self.error = None, ''
        self.here.setText('loading the two runs…')
        g, t = gap_m or S.GAP_M, ttc_s or S.TTC_S
        tasks.run(self, lambda: (run_a, run_b, S.synchronize(run_a, run_b, g, t)), self._done, self._failed)

    def load_now(self, run_a: str, run_b: str, gap_m: float | None = None, ttc_s: float | None = None):
        from ..core.logreplay import sync as S
        self.paths = (run_a, run_b)
        try:
            self._done((run_a, run_b, S.synchronize(run_a, run_b, gap_m or S.GAP_M, ttc_s or S.TTC_S)))
        except (ValueError, OSError) as e:
            self._failed(e)

    def _failed(self, e):
        self.error = str(e)
        self.data = None
        self.here.setText(f'cannot compare: {e}')
        self._set_enabled(False)
        self.loaded.emit(None)

    def _done(self, res):
        from ..core.logreplay.sync import reason_text
        a, b, d = res
        if (a, b) != self.paths:
            return                                  # a newer pair was asked for meanwhile
        self.data = d
        self.error = ''
        A, B = d['a'], d['b']
        for s, side in (('A', A), ('B', B)):
            self.ids[s].setText(side['dir'])
            self.ids[s].setToolTip(f'{side["name"]}\n{side["dir"]}\nposes from {side["source"]}\nClick to copy')
            self.fids[s].set(side['fidelity'])
        tone, why = likeness([A['fidelity'], B['fidelity']])
        self.likeness.setVisible(tone == 'critical')
        self.likeness.label.setText(f'Not like for like: {why}. The divergences below may come from the '
                                    'simulators, not the change.' if tone == 'critical' else '')
        self.likeness_note.setVisible(tone == 'muted')
        self.likeness_note.setText(f'Fidelity: {why}.' if tone == 'muted' else '')
        self.view_a.set_runs(A, B, d['bounds'])
        self.view_b.set_runs(B, A, d['bounds'])
        n = len(d['grid'])
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.slider.set_spans([(e['start'], e['end']) for e in d['events']])
        sm = d['summary']
        ev = d['events']
        th = sm['thresholds']
        parts = [f'{len(ev)} divergence{"s" * (len(ev) != 1)}' if ev else 'No divergence within the thresholds']
        if ev:
            parts.append(f'first at {ev[0]["t"]:.2f} s ({reason_text(ev[0]["reasons"])})')
        if sm['max_gap_m'] is not None:
            parts.append(f'largest ego gap {sm["max_gap_m"]:.2f} m at {sm["max_gap_t"]:.2f} s')
        self.summary.setText(' · '.join(parts))
        self.summary.setToolTip(f'Rule: ego gap > {th["gap_m"]:g} m, min TTC change > {th["ttc_s"]:g} s, a collision '
                                'or an actor in one run only (Details)' + ''.join(f'\n{n}' for n in d['notes']))
        self._set_enabled(True)
        self.seek_index(0)
        self.loaded.emit(d)

    def _set_enabled(self, on: bool):
        has = on and bool(self.data and self.data['events'])
        for w in (self.slider,):
            w.setEnabled(on)
        for w in (self.prev_btn, self.next_btn, self.first_btn):
            w.setEnabled(has)

    # ------------------------------------------------------------------ the clock
    @property
    def events(self) -> list:
        return list((self.data or {}).get('events') or ())

    def seek_index(self, k: int):
        from ..core.logreplay.sync import reason_text
        d = self.data
        if d is None:
            return
        k = int(min(max(k, 0), len(d['grid']) - 1))
        if self.slider.value() != k:
            self.slider.blockSignals(True)
            self.slider.setValue(k)
            self.slider.blockSignals(False)
        bits = int(d['reasons'][k])
        div = bits > 0
        self.view_a.set_step(k, div)
        self.view_b.set_step(k, div)
        self.time_lbl.setText(f't {d["grid"][k]:6.2f} s')
        gap = d['gap'][k]
        from ..core.logreplay.sync import REASONS
        names = [REASONS[i][0] for i in range(len(REASONS)) if bits >> i & 1]
        txt = 'ego gap ' + ('—' if np.isnan(gap) else f'{gap:.2f} m')
        self.here.setText(f'{txt} · diverging here: {reason_text(names)}' if div else f'{txt} · together here')

    def jump_first(self):
        if self.events:
            self.slider.setValue(self.events[0]['start'])

    def next_divergence(self):
        k = self.slider.value()
        nxt = next((e['start'] for e in self.events if e['start'] > k), None)
        if nxt is not None:
            self.slider.setValue(nxt)

    def prev_divergence(self):
        k = self.slider.value()
        prv = [e['start'] for e in self.events if e['start'] < k]
        if prv:
            self.slider.setValue(prv[-1])
