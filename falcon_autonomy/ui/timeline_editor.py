"""The scenario timeline editor (Scenarios ▸ Timeline): the timeline's events on a time axis, an event's trigger and
actions in a form, and a top-down plan of where everyone goes.

The time axis has one row per event. A time trigger is a diamond at its time; a numeric one can be dragged (snapped
to the chosen step), one bound to a parameter ("$brake_at") sits at the sampled value with the parameter's range as a
band and is changed through the parameter. A distance or "ahead" trigger has no time of its own: it sits, hollow, where
the plan (core.scenario_plan) has it fire — or at the right edge, "not reached in the plan". Each action is a bar from
its firing: a lane change's numeric duration can be dragged by its right end; a brake or speed change lasts as long as
the plan's ramp. The wheel zooms around the pointer; a click on the axis moves the playhead, which the plan view
follows. Every change is one undoable project edit (win.edit through the page); nothing is written until something
changes, so an untouched scenario saves byte for byte as it was.
"""
from __future__ import annotations

import copy
import json

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QSplitter, QTableWidget,
                             QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from ..core import scenario_plan as PL
from ..core import scenarios as SC
from .theme import THEME
from .widgets import Form, Note, button, label, set_enabled

ROW_H = 30
AXIS_H = 24
LEFT = 150
SNAPS = (('0.05 s', 0.05), ('0.1 s', 0.1), ('0.5 s', 0.5), ('1 s', 1.0), ('off', 0.0))
KIND_COL = {'pedestrian': '#d95926', 'child': '#d95926', 'bicycle': '#c98500', 'motorcycle': '#c98500',
            'bus': '#9085e9', 'truck': '#9085e9', 'van': '#9085e9'}
ACTION_COL = {'brake': '#d95926', 'stop': '#d95926', 'speed': '#3987e5', 'go': '#3987e5', 'lane_change': '#9085e9',
              'walk': '#c98500', 'open_door': '#c98500'}


def num(text):
    """A typed value: a number when it reads as one, else the text (an expression such as "$brake_at * 2")."""
    t = (text or '').strip()
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def describe(trig: dict) -> str:
    t = trig or {}
    return {'time': f'at {t.get("t_s")} s', 'distance': f'ego within {t.get("lt_m")} m of {t.get("actor")}',
            'ahead': f'{t.get("actor")} > {t.get("gt_m")} m ahead'}.get(t.get('type'), json.dumps(t))


def params_text(a: dict) -> str:
    return ', '.join(f'{k}={v}' for k, v in a.items() if k not in ('actor', 'do'))


def parse_params(text: str) -> tuple[dict, str]:
    """'kmh=30, accel=$a' -> ({'kmh': 30, 'accel': '$a'}, '') or ({}, the problem)."""
    out = {}
    for part in (text or '').split(','):
        if not part.strip():
            continue
        if '=' not in part:
            return {}, f'"{part.strip()}" is not name=value'
        k, v = part.split('=', 1)
        k = k.strip()
        if not k.isidentifier() or k in ('actor', 'do'):
            return {}, f'"{k}" cannot be a field name here'
        out[k] = num(v)
    return out, ''


class TimeAxis(QWidget):
    """Events on a time axis (see the module's docstring)."""
    moved = pyqtSignal(float)                     # the playhead's time

    def __init__(self, editor):
        super().__init__()
        self.ed = editor
        self.t0 = 0.0                             # the time at the left edge
        self.pps = 0.0                            # pixels per second (0: fit)
        self.playhead = 0.0
        self.drag = None                          # (kind 'trigger'|'duration', event index, action index)
        self.drag_val = None
        self.setMouseTracking(True)
        self.setMinimumHeight(AXIS_H + 2 * ROW_H + 8)

    # ------------------------------------------------------------------ geometry
    def duration(self):
        return max(PL._f((self.ed.conc or {}).get('duration_s'), 12.0), 1.0)

    def scale(self):
        fit = (self.width() - LEFT - 16) / self.duration()
        return self.pps or fit

    def x_of(self, t):
        return LEFT + (t - self.t0) * self.scale()

    def t_of(self, x):
        return self.t0 + (x - LEFT) / self.scale()

    def row_y(self, i):
        return AXIS_H + 4 + i * ROW_H

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(600, AXIS_H + 8 + ROW_H * max(len(self.ed.timeline()), 2))

    def zoom(self, f, about_x=None):
        s0 = self.scale()
        about_x = LEFT if about_x is None else about_x
        t_at = self.t_of(about_x)
        self.pps = min(max(s0 * f, 4.0), 2000.0)
        self.t0 = max(t_at - (about_x - LEFT) / self.pps, 0.0)
        self.update()

    def fit(self):
        self.pps, self.t0 = 0.0, 0.0
        self.update()

    # ------------------------------------------------------------------ what is where
    def trigger_time(self, i):
        """(time, how): how is 'fixed' (numeric, draggable), 'param' (bound to a parameter), 'plan' (when the plan
        fires it) or 'never' (not reached in the plan)."""
        ev = self.ed.timeline()[i]
        trig = ev.get('trigger') or {}
        if trig.get('type') == 'time':
            if self.drag and self.drag[0] == 'trigger' and self.drag[1] == i and self.drag_val is not None:
                return self.drag_val, 'fixed'
            if is_num(trig.get('t_s')):
                return float(trig['t_s']), 'fixed'
        fired = dict((k, t) for k, t in (self.ed.plan or {}).get('fired', []))
        if i in fired:
            return fired[i], 'param' if trig.get('type') == 'time' else 'plan'
        return self.duration(), 'never'

    def action_span(self, i, j):
        """(start, end or None, draggable end?) of action j of event i, from the plan."""
        for (ei, aj, t, end) in (self.ed.plan or {}).get('actions', []):
            if ei == i and aj == j:
                a = self.ed.timeline()[i]['actions'][j]
                if a.get('do') == 'lane_change' and is_num(a.get('duration_s')):
                    d = self.drag_val if (self.drag and self.drag[0] == 'duration' and self.drag[1:] == (i, j)
                                          and self.drag_val is not None) else float(a['duration_s'])
                    return t, t + d, True
                return t, end, False
        return None

    def hit(self, pos):
        """What is under the pointer: ('trigger', i, None) | ('duration', i, j) | None."""
        tl = self.ed.timeline()
        for i in range(len(tl)):
            y = self.row_y(i)
            if not (y <= pos.y() <= y + ROW_H):
                continue
            t, how = self.trigger_time(i)
            if how == 'fixed' and abs(pos.x() - self.x_of(t)) <= 7 and pos.y() <= y + ROW_H / 2 + 4:
                return ('trigger', i, None)
            for j in range(len(tl[i].get('actions') or [])):
                sp = self.action_span(i, j)
                if sp and sp[2] and abs(pos.x() - self.x_of(sp[1])) <= 6:
                    return ('duration', i, j)
        return None

    # ------------------------------------------------------------------ painting
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['viewport']))
        T = self.duration()
        sc = self.scale()
        step = next(s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 60) if s * sc >= 48)
        p.setFont(self.font())
        k = int(self.t0 / step)
        while True:
            t = k * step
            x = self.x_of(t)
            if x > self.width():
                break
            if x >= LEFT - 1:
                p.setPen(QColor(THEME.tok['grid2']))
                p.drawLine(QPointF(x, AXIS_H), QPointF(x, self.height()))
                p.setPen(QColor(THEME.tok['muted']))
                p.drawText(QPointF(x + 3, AXIS_H - 8), f'{t:g} s')
            k += 1
        xe = self.x_of(T)
        if xe < self.width():                                     # past the scenario's end
            c = QColor(THEME.tok['panel'])
            c.setAlphaF(0.6)
            p.fillRect(QRectF(xe, AXIS_H, self.width() - xe, self.height() - AXIS_H), c)
        tl = self.ed.timeline()
        bad = self.ed.bad_events
        for i, evt in enumerate(tl):
            y = self.row_y(i)
            if i == self.ed.current:
                c = QColor(THEME.tok['sel'])
                c.setAlphaF(0.35)
                p.fillRect(QRectF(0, y, self.width(), ROW_H), c)
            p.setPen(QColor('#d95926') if i in bad else QColor(THEME.tok['ink']))
            p.drawText(QRectF(8, y, LEFT - 12, ROW_H), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(f'{i + 1}. {describe(evt.get("trigger"))}',
                                                  Qt.TextElideMode.ElideRight, LEFT - 14))
            t, how = self.trigger_time(i)
            trig = evt.get('trigger') or {}
            if how == 'param':                                       # the parameter's range as a band
                pn = str(trig.get('t_s', '')).strip()
                prm = (self.ed.scn.get('params') or {}).get(pn.lstrip('$')) if pn.startswith('$') else None
                if prm and prm.get('dist') == 'uniform':
                    c = QColor('#3987e5')
                    c.setAlphaF(0.18)
                    x0, x1 = self.x_of(float(prm['lo'])), self.x_of(float(prm['hi']))
                    p.fillRect(QRectF(x0, y + 4, x1 - x0, ROW_H / 2 - 2), c)
            # actions: bars from the firing
            acts = evt.get('actions') or []
            for j, a in enumerate(acts):
                sp = self.action_span(i, j)
                if sp is None:
                    continue
                s0, s1, dr = sp
                c = QColor(ACTION_COL.get(a.get('do'), '#3987e5'))
                yb = y + ROW_H / 2 + 2 + j * min(10, (ROW_H / 2 - 4) / max(len(acts), 1))
                hb = max(min(8, (ROW_H / 2 - 4) / max(len(acts), 1) - 1), 3)
                if s1 is None or s1 <= s0:
                    p.setBrush(c)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.drawEllipse(QPointF(self.x_of(s0), yb + hb / 2), 3, 3)
                else:
                    c.setAlphaF(0.75)
                    p.setBrush(c)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.drawRoundedRect(QRectF(self.x_of(s0), yb, max(self.x_of(s1) - self.x_of(s0), 2), hb), 2, 2)
                    if dr:
                        p.setPen(QPen(QColor(THEME.tok['ink']), 2))
                        p.drawLine(QPointF(self.x_of(s1), yb - 2), QPointF(self.x_of(s1), yb + hb + 2))
            # the trigger
            x = self.x_of(t)
            yc = y + ROW_H / 2 - 4
            d = 6
            poly = QPolygonF([QPointF(x, yc - d), QPointF(x + d, yc), QPointF(x, yc + d), QPointF(x - d, yc)])
            col = QColor('#d95926') if i in bad else QColor(THEME.tok['ink'])
            if how == 'fixed':
                p.setBrush(col)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(poly)
            elif how == 'param':
                p.setBrush(QColor('#3987e5'))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(poly)
            elif how == 'plan':
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(col, 1.5))
                p.drawEllipse(QPointF(x, yc), d - 1, d - 1)
            else:
                p.setPen(QPen(QColor(THEME.tok['muted']), 1.5, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x - 10, yc), QPointF(x, yc))
                p.drawText(QPointF(max(x - 150, LEFT), yc - 6), 'not reached in the plan')
        # the playhead
        xp = self.x_of(self.playhead)
        p.setPen(QPen(QColor('#d95926'), 1.5))
        p.drawLine(QPointF(xp, 2), QPointF(xp, self.height()))
        path = QPainterPath()
        path.addPolygon(QPolygonF([QPointF(xp - 5, 2), QPointF(xp + 5, 2), QPointF(xp, 9)]))
        p.fillPath(path, QColor('#d95926'))
        if not tl:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(QRectF(LEFT, AXIS_H, self.width() - LEFT, ROW_H * 2), Qt.AlignmentFlag.AlignCenter,
                       'No events yet' + (' — Add an event' if self.ed.editable else ''))

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, ev):
        pos = ev.position()
        h = self.hit(pos) if self.ed.editable else None
        if h is not None:
            self.drag, self.drag_val = h, None
            self.ed.select(h[1])
            return
        if pos.y() < AXIS_H or pos.x() >= LEFT:
            if pos.y() >= AXIS_H:
                i = int((pos.y() - AXIS_H - 4) // ROW_H)
                if 0 <= i < len(self.ed.timeline()):
                    self.ed.select(i)
            self.set_playhead(self.t_of(pos.x()))
        else:
            i = int((pos.y() - AXIS_H - 4) // ROW_H)
            if 0 <= i < len(self.ed.timeline()):
                self.ed.select(i)

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self.drag is None:
            h = self.hit(pos) if self.ed.editable else None
            self.setCursor(Qt.CursorShape.SizeHorCursor if h else Qt.CursorShape.ArrowCursor)
            return
        kind, i, j = self.drag
        if kind == 'trigger':
            self.drag_val = max(self.ed.snap(self.t_of(pos.x())), 0.0)
        else:
            t0 = self.action_span(i, j)[0]
            self.drag_val = max(self.ed.snap(self.t_of(pos.x()) - t0), self.ed.snap_step() or 0.05)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self.drag is None:
            return
        kind, i, j = self.drag
        v, self.drag, self.drag_val = self.drag_val, None, None
        if v is None:
            return
        v = round(v, 3)
        if kind == 'trigger':
            self.ed.set_trigger_field(i, 't_s', v)
        else:
            self.ed.set_action_field(i, j, 'duration_s', v)

    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:          # shift + wheel: pan
            self.t0 = max(self.t0 - dy / 120.0 * 40.0 / self.scale(), 0.0)
            self.update()
            return
        self.zoom(1.25 if dy > 0 else 0.8, ev.position().x())

    def set_playhead(self, t):
        self.playhead = min(max(t, 0.0), self.duration())
        self.update()
        self.moved.emit(self.playhead)


class PlanView(QWidget):
    """The plan from above: the schematic lanes, everyone's path, and where they are at the playhead (across the road
    drawn up to 4× the scale along it, so the lanes stay readable on a wide view)."""

    def __init__(self, editor):
        super().__init__()
        self.ed = editor
        self.t = 0.0
        self.setMinimumHeight(120)

    def set_time(self, t):
        self.t = t
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['viewport']))
        pl = self.ed.plan
        if not pl or not pl['t']:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No plan: the scenario cannot be sampled')
            return
        lht = self.ed.lht
        xs = [x for tr in pl['tracks'].values() for x, _ in tr]
        ys = [y for tr in pl['tracks'].values() for _, y in tr] + [-PL.LANE_W * 2.5, PL.LANE_W * 1.5]
        x0, x1 = min(xs + [0.0]) - 8, max(xs + [40.0]) + 8
        y0, y1 = min(ys) - 2, max(ys) + 2
        W, H = self.width() - 20, self.height() - 24
        k = W / (x1 - x0)
        ky = min(H / (y1 - y0), 4.0 * k)     # across the road drawn up to 4× larger, so lanes stay readable
        sgn = -1.0 if lht else 1.0           # the kerb side is drawn at the bottom in right-hand traffic

        def P(x, y):
            return QPointF(10 + (x - x0) * k, 10 + H / 2 + (sgn * y - (y0 + y1) / 2 * sgn) * ky)
        # lanes: the ego lane, the adjacent / opposite lanes, the sidewalks (schematic)
        for yc, col in ((0.0, 'panel'), (-PL.LANE_W, 'panel'), (-2 * PL.LANE_W, 'panel'),
                        (PL.LANE_W / 2 + 1.0, 'raised'), (-2.5 * PL.LANE_W - 1.0, 'raised')):
            a, b = P(x0, yc - (PL.LANE_W / 2 if col == 'panel' else 1.0)), P(x1, yc + (PL.LANE_W / 2 if col == 'panel'
                                                                                      else 1.0))
            p.fillRect(QRectF(a, b).normalized(), QColor(THEME.tok[col]))
        p.setPen(QPen(QColor(THEME.tok['ink2']), 1, Qt.PenStyle.DashLine))
        for yl in (-PL.LANE_W / 2, -1.5 * PL.LANE_W):
            p.drawLine(P(x0, yl), P(x1, yl))
        p.setPen(QColor(THEME.tok['muted']))
        for m in range(int(x0 // 20) * 20, int(x1) + 1, 20):
            if m >= x0:
                q = P(m, -2.5 * PL.LANE_W - 2.0)
                p.drawText(q + QPointF(2, -2), f'{m} m')
        idx = min(max(int(round(self.t / max(pl['t'][1] - pl['t'][0], 1e-6))) if len(pl['t']) > 1 else 0, 0),
                  len(pl['t']) - 1)
        kinds = {a['id']: a.get('kind') for a in (self.ed.conc or {}).get('actors', [])}
        for aid, tr in pl['tracks'].items():
            col = QColor('#199e70') if aid == 'ego' else QColor(KIND_COL.get(kinds.get(aid), '#3987e5'))
            faint = QColor(col)
            faint.setAlphaF(0.45)
            p.setPen(QPen(faint, 1.2))
            path = QPainterPath(P(*tr[0]))
            for pt in tr[1::4]:
                path.lineTo(P(*pt))
            p.drawPath(path)
            x, y = tr[idx]
            L, Wd = pl['sizes'].get(aid, (4.6, 2.0))
            c = P(x, y)
            p.setBrush(col)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(c.x() - L * k / 2, c.y() - Wd * ky / 2, max(L * k, 3), max(Wd * ky, 3)))
            p.setPen(QColor(THEME.tok['ink']))
            p.drawText(QPointF(c.x() - L * k / 2, c.y() - Wd * ky / 2 - 3), aid)


class TimelineEditor(QWidget):
    """The Timeline tab: the axis, the selected event's form and the plan (see the module's docstring)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.scn: dict = {}
        self.conc: dict = {}
        self.plan: dict | None = None
        self.editable = False
        self.lht = False
        self.current = -1
        self.bad_events: set = set()
        self._filling = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.add_btn = button('Add an event', self.add_event)
        self.dup_btn = button('Duplicate', self.duplicate_event)
        self.rm_btn = button('Remove', self.remove_event)
        for b in (self.add_btn, self.dup_btn, self.rm_btn):
            top.addWidget(b)
        top.addStretch(1)
        self.json_btn = button('Edit as JSON…', page.edit_timeline)
        top.addWidget(self.json_btn)
        v.addLayout(top)
        tb = QHBoxLayout()
        tb.addWidget(label('Snap', muted=True))
        self.snap_box = QComboBox()
        for name, val in SNAPS:
            self.snap_box.addItem(name, val)
        self.snap_box.setCurrentIndex(1)
        tb.addWidget(self.snap_box)
        for text, tip, slot in (('−', 'Zoom out', lambda: self.axis.zoom(0.8)),
                                ('+', 'Zoom in (or the wheel; shift + wheel pans)', lambda: self.axis.zoom(1.25)),
                                ('Fit', 'The whole scenario', lambda: self.axis.fit())):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            tb.addWidget(b)
        self.play_btn = QToolButton()
        self.play_btn.setText('▶ Play the plan')
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self.play)
        tb.addWidget(self.play_btn)
        tb.addStretch(1)
        v.addLayout(tb)
        split = QSplitter(Qt.Orientation.Vertical)
        self.axis = TimeAxis(self)
        split.addWidget(self.axis)
        form = QWidget()
        fv = QVBoxLayout(form)
        fv.setContentsMargins(0, 4, 0, 0)
        f = Form()
        self.trig_type = QComboBox()
        self.trig_type.addItems(list(SC.TRIGGERS))
        self.trig_type.activated.connect(lambda _: self.change_trigger_type())
        f.row('When', self.trig_type)
        self.trig_actor = QComboBox()
        self.trig_actor.activated.connect(lambda _: self.set_trigger_field(self.current, 'actor',
                                                                           self.trig_actor.currentText()))
        f.row('Actor', self.trig_actor)
        self.trig_value = QLineEdit()
        self.trig_value.setPlaceholderText('a number or an expression such as $brake_at')
        self.trig_value.editingFinished.connect(self.commit_trigger_value)
        f.row('Value', self.trig_value)
        self.form = f
        fv.addLayout(f)
        self.actions = QTableWidget(0, 3)
        self.actions.setHorizontalHeaderLabels(['Actor', 'Action', 'Fields (name=value, …)'])
        self.actions.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.actions.verticalHeader().setVisible(False)
        self.actions.cellChanged.connect(self.action_cell)
        self.actions.itemSelectionChanged.connect(lambda: set_enabled(
            self.rm_act_btn, self.editable and self.actions.currentRow() >= 0, 'Choose an action first'))
        fv.addWidget(self.actions, 1)
        ab = QHBoxLayout()
        self.add_act_btn = button('Add an action', self.add_action)
        self.rm_act_btn = button('Remove the action', self.remove_action)
        ab.addWidget(self.add_act_btn)
        ab.addWidget(self.rm_act_btn)
        ab.addStretch(1)
        fv.addLayout(ab)
        self.problems = QLabel()
        self.problems.setWordWrap(True)
        self.problems.setProperty('crit', True)
        fv.addWidget(self.problems)
        split.addWidget(form)
        split.setSizes([120, 320])
        # the plan from above: the page shows it in place of the scene sketch while this tab is open
        self.plan_panel = QWidget()
        pvl = QVBoxLayout(self.plan_panel)
        pvl.setContentsMargins(0, 0, 0, 0)
        self.planview = PlanView(self)
        pvl.addWidget(self.planview, 1)
        self.plan_status = label('', muted=True, small=True)
        pvl.addWidget(self.plan_status)
        self.plan_note = Note(PL.NOTE)
        pvl.addWidget(self.plan_note)
        v.addWidget(split, 1)
        self.axis.moved.connect(self.planview.set_time)
        self.axis.moved.connect(lambda _t: self.show_status())
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ model
    def timeline(self) -> list:
        return self.scn.get('timeline') or []

    def snap_step(self) -> float:
        return float(self.snap_box.currentData() or 0.0)

    def snap(self, t):
        s = self.snap_step()
        return round(round(t / s) * s, 6) if s > 0 else t

    def set_scenario(self, scn: dict | None, conc: dict | None, editable: bool, lht: bool = False):
        self.scn = copy.deepcopy(scn or {})
        self.conc = conc or {}
        self.editable, self.lht = editable, lht
        try:
            self.plan = PL.plan(self.conc) if self.conc.get('actors') is not None else None
        except (TypeError, ValueError, KeyError):
            self.plan = None
        self.bad_events = set()
        probs = SC.validate(self.scn) if self.scn else []
        for pr in probs:
            if pr.startswith('timeline '):
                try:
                    self.bad_events.add(int(pr.split()[1].rstrip(':')))
                except ValueError:
                    pass
        self.problems.setText('\n'.join('⚠ ' + p for p in probs if p.startswith('timeline')))
        n = len(self.timeline())
        if self.current >= n:
            self.current = n - 1
        if self.current < 0 and n:
            self.current = 0
        self.axis.setMinimumHeight(AXIS_H + 8 + ROW_H * max(n, 2))
        self.axis.update()
        self.planview.update()
        self.show_status()
        self.fill_form()

    def show_status(self):
        pl = self.plan
        if not pl:
            self.plan_status.setText('')
            return
        fired = ', '.join(f'{i + 1} at {t:.2f} s' for i, t in pl['fired']) or 'none'
        seed = self.conc.get('seed')
        self.plan_status.setText(f't = {self.axis.playhead:.2f} s · events fired in the plan: {fired}'
                                 + (f' · sample of seed {seed}' if seed is not None else ''))

    def select(self, i):
        self.current = i
        self.fill_form()
        self.axis.update()

    def fill_form(self):
        self._filling = True
        tl = self.timeline()
        ev = tl[self.current] if 0 <= self.current < len(tl) else None
        ok = self.editable and ev is not None
        why = 'Copy the built-in to the project to edit it' if not self.editable else 'Choose an event first'
        for w in (self.trig_type, self.trig_actor, self.trig_value, self.dup_btn, self.rm_btn, self.add_act_btn,
                  self.actions):
            set_enabled(w, ok, why)
        set_enabled(self.add_btn, self.editable, 'Copy the built-in to the project to edit it')
        set_enabled(self.json_btn, self.editable, 'Copy the built-in to the project to edit it')
        set_enabled(self.rm_act_btn, ok and self.actions.currentRow() >= 0, why if not ok else
                    'Choose an action first')
        ids = [a['id'] for a in self.scn.get('actors', [])]
        self.trig_actor.clear()
        self.trig_actor.addItems(ids)
        self.actions.setRowCount(0)
        if ev is None:
            self.trig_value.setText('')
            self._filling = False
            return
        trig = ev.get('trigger') or {}
        tt = trig.get('type', 'time')
        self.trig_type.setCurrentText(tt)
        field = {'time': 't_s', 'distance': 'lt_m', 'ahead': 'gt_m'}.get(tt, 't_s')
        self.trig_value.setText('' if trig.get(field) is None else str(trig.get(field)))
        lb, _ = self.form.rows.get(id(self.trig_value), (None, None))
        if lb is not None:
            lb.setText({'t_s': 'At (s)', 'lt_m': 'Within (m)', 'gt_m': 'Ahead by (m)'}[field])
        self.form.set_row_visible(self.trig_actor, tt != 'time')
        if tt != 'time':
            k = self.trig_actor.findText(str(trig.get('actor', '')))
            if k < 0 and trig.get('actor'):
                self.trig_actor.addItem(str(trig['actor']))
                k = self.trig_actor.count() - 1
            self.trig_actor.setCurrentIndex(k)
        for j, a in enumerate(ev.get('actions') or []):
            self.actions.insertRow(j)
            actor = QComboBox()
            actor.addItems(ids if a.get('actor') in ids else ids + [str(a.get('actor'))])
            actor.setCurrentText(str(a.get('actor')))
            actor.activated.connect(lambda _k, jj=j, c=actor: self.set_action_field(self.current, jj, 'actor',
                                                                                    c.currentText()))
            do = QComboBox()
            do.addItems(list(SC.ACTIONS) if a.get('do') in SC.ACTIONS else list(SC.ACTIONS) + [str(a.get('do'))])
            do.setCurrentText(str(a.get('do')))
            do.activated.connect(lambda _k, jj=j, c=do: self.change_action_kind(jj, c.currentText()))
            for w in (actor, do):
                w.setEnabled(self.editable)
            self.actions.setCellWidget(j, 0, actor)
            self.actions.setCellWidget(j, 1, do)
            it = QTableWidgetItem(params_text(a))
            if not self.editable:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.actions.setItem(j, 2, it)
        self._filling = False

    # ------------------------------------------------------------------ edits (each one undoable)
    def commit(self, tl, text):
        """Write the whole timeline as one project edit after checking it; refused with the problems shown."""
        probs = [p for p in SC.validate(dict(self.scn, timeline=tl)) if p.startswith(('timeline', 'cannot'))]
        hard = [p for p in probs if 'unknown trigger' in p or 'needs' in p or 'an entry is' in p]
        if hard:
            self.problems.setText('\n'.join('⚠ ' + p + ' (not applied)' for p in hard))
            return False
        self.page.set(['timeline'], tl, text)
        return True

    def _tl(self):
        return copy.deepcopy(self.timeline())

    def set_trigger_field(self, i, field, value):
        if not self.editable or not (0 <= i < len(self.timeline())):
            return
        tl = self._tl()
        if tl[i]['trigger'].get(field) == value:
            self.axis.update()
            return
        tl[i]['trigger'][field] = value
        self.commit(tl, f'Timeline event {i + 1}: {field}')

    def commit_trigger_value(self):
        if self._filling or not self.editable or not (0 <= self.current < len(self.timeline())):
            return
        tt = (self.timeline()[self.current].get('trigger') or {}).get('type', 'time')
        field = {'time': 't_s', 'distance': 'lt_m', 'ahead': 'gt_m'}.get(tt, 't_s')
        text = self.trig_value.text().strip()
        if not text:
            self.problems.setText(f'⚠ the trigger needs a value ({field})')
            return
        self.set_trigger_field(self.current, field, num(text))

    def change_trigger_type(self):
        if self._filling or not self.editable or not (0 <= self.current < len(self.timeline())):
            return
        tt = self.trig_type.currentText()
        tl = self._tl()
        old = tl[self.current].get('trigger') or {}
        if old.get('type') == tt:
            return
        ids = [a['id'] for a in self.scn.get('actors', [])]
        actor = old.get('actor') or (ids[0] if ids else '')
        tl[self.current]['trigger'] = {'time': {'type': 'time', 't_s': 1.0},
                                       'distance': {'type': 'distance', 'actor': actor, 'lt_m': 20.0},
                                       'ahead': {'type': 'ahead', 'actor': actor, 'gt_m': 5.0}}[tt]
        self.commit(tl, f'Timeline event {self.current + 1}: {tt} trigger')

    def set_action_field(self, i, j, field, value):
        if not self.editable:
            return
        tl = self._tl()
        if tl[i]['actions'][j].get(field) == value:
            self.axis.update()
            return
        tl[i]['actions'][j][field] = value
        self.commit(tl, f'Timeline event {i + 1}: action {j + 1} {field}')

    def change_action_kind(self, j, do):
        if self._filling or not self.editable:
            return
        tl = self._tl()
        a = tl[self.current]['actions'][j]
        if a.get('do') == do:
            return
        tl[self.current]['actions'][j] = dict({'actor': a.get('actor'), 'do': do}, **SC.ACTIONS.get(do, {}))
        self.commit(tl, f'Timeline event {self.current + 1}: action {j + 1} → {do}')

    def action_cell(self, r, c):
        if self._filling or c != 2 or not self.editable:
            return
        vals, err = parse_params(self.actions.item(r, 2).text())
        if err:
            self.problems.setText('⚠ ' + err + ' (not applied)')
            return
        tl = self._tl()
        a = tl[self.current]['actions'][r]
        new = {'actor': a.get('actor'), 'do': a.get('do'), **vals}
        if new == a:
            return
        tl[self.current]['actions'][r] = new
        self.commit(tl, f'Timeline event {self.current + 1}: action {r + 1} fields')

    def add_event(self):
        if not self.editable:
            return
        ids = [a['id'] for a in self.scn.get('actors', [])]
        tl = self._tl()
        t = self.snap(self.axis.playhead) if self.axis.playhead > 0 else 1.0
        tl.append({'trigger': {'type': 'time', 't_s': t},
                   'actions': [dict({'actor': ids[0], 'do': 'brake'}, **SC.ACTIONS['brake'])] if ids else []})
        if self.commit(tl, 'Add a timeline event'):
            self.select(len(tl) - 1)

    def duplicate_event(self):
        if not self.editable or not (0 <= self.current < len(self.timeline())):
            return
        tl = self._tl()
        tl.insert(self.current + 1, copy.deepcopy(tl[self.current]))
        if self.commit(tl, f'Duplicate timeline event {self.current + 1}'):
            self.select(self.current + 1)

    def remove_event(self):
        if not self.editable or not (0 <= self.current < len(self.timeline())):
            return
        tl = self._tl()
        del tl[self.current]
        self.commit(tl, f'Remove timeline event {self.current + 1}')

    def add_action(self):
        if not self.editable or not (0 <= self.current < len(self.timeline())):
            return
        ids = [a['id'] for a in self.scn.get('actors', [])]
        if not ids:
            self.problems.setText('⚠ add an actor first')
            return
        tl = self._tl()
        tl[self.current].setdefault('actions', []).append(dict({'actor': ids[0], 'do': 'speed'},
                                                               **SC.ACTIONS['speed']))
        self.commit(tl, f'Timeline event {self.current + 1}: add an action')

    def remove_action(self):
        j = self.actions.currentRow()
        if not self.editable or not (0 <= self.current < len(self.timeline())) or j < 0:
            return
        tl = self._tl()
        del tl[self.current]['actions'][j]
        self.commit(tl, f'Timeline event {self.current + 1}: remove action {j + 1}')

    # ------------------------------------------------------------------ playing the plan
    def play(self, on):
        if on:
            if self.axis.playhead >= self.axis.duration() - 1e-6:
                self.axis.set_playhead(0.0)
            self.timer.start()
        else:
            self.timer.stop()
        self.play_btn.setText('❚❚ Pause' if on else '▶ Play the plan')

    def _tick(self):
        t = self.axis.playhead + self.timer.interval() / 1000.0
        if t >= self.axis.duration():
            self.axis.set_playhead(self.axis.duration())
            self.play_btn.setChecked(False)
            return
        self.axis.set_playhead(t)

    def stop(self):
        self.timer.stop()
