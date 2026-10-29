"""Data ▸ Compare: two to four recorded runs side by side (core.runcompare): their cameras on one time scrubber,
aligned by simulation time from each run's first frame; their events on a shared timeline; and a table of what each
recorded, with the difference from the first run. Rule violations and lane margins come from a traffic-rule check
of each run, computed in the background."""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QSlider, QSplitter, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import runcompare as RC
from .. import tasks
from ..drivesets import index_entries
from ..widgets import EmptyState, Section, button, label
from .data import ImageView

SLOTS = 'ABCD'
KIND_COLOUR = {'collision': '#d03b3b', 'rule': '#fab219', 'teleport': '#ec835a', 'odd_exit': '#3987e5'}


class EventTimeline(QWidget):
    """One row per run over the longest run's time: its length as a bar, its events as marks (intervals as spans),
    the scrubber's time as a line; a click seeks."""
    seek = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.rows: list = []                 # [(slot, duration, events)]
        self.t = 0.0
        self.setMouseTracking(True)
        self.setMinimumHeight(40)

    def set_rows(self, rows):
        self.rows = rows
        self.setMinimumHeight(24 + 20 * max(len(rows), 1))
        self.update()

    def set_time(self, t):
        self.t = t
        self.update()

    def _geom(self):
        T = max([d for _, d, _ in self.rows] + [1e-6])
        return 40, self.width() - 10, T

    def mousePressEvent(self, ev):
        x0, x1, T = self._geom()
        if self.rows and x0 <= ev.position().x() <= x1:
            self.seek.emit(T * (ev.position().x() - x0) / (x1 - x0))

    def mouseMoveEvent(self, ev):
        x0, x1, T = self._geom()
        tip = ''
        if self.rows and x0 <= ev.position().x() <= x1:
            t = T * (ev.position().x() - x0) / (x1 - x0)
            k = int((ev.position().y() - 4) // 20)
            if 0 <= k < len(self.rows):
                near = [e for e in self.rows[k][2] if (e['t1'] is not None and e['t'] <= t <= e['t1'])
                        or abs(e['t'] - t) <= T / 100]
                tip = '\n'.join(f'{e["t"]:.2f} s: {e["text"]}' for e in near[:8])
        self.setToolTip(tip)

    def paintEvent(self, ev):
        from ..theme import THEME
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        if not self.rows:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Choose runs to compare')
            return
        x0, x1, T = self._geom()

        def X(t):
            return x0 + (x1 - x0) * min(max(t / T, 0.0), 1.0)
        for k, (slot, dur, evs) in enumerate(self.rows):
            y = 4 + 20 * k
            p.setPen(QColor(THEME.tok['ink2']))
            p.drawText(QRectF(0, y, x0 - 8, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, slot)
            p.fillRect(QRectF(x0, y + 7, X(dur) - x0, 4), QColor(THEME.tok['strong']))
            for e in evs:
                col = QColor(KIND_COLOUR.get(e['kind'], THEME.tok['muted']))
                if e['t1'] is not None:
                    p.fillRect(QRectF(X(e['t']), y + 3, max(X(e['t1']) - X(e['t']), 2.0), 12), col)
                else:
                    p.fillRect(QRectF(X(e['t']) - 1.5, y + 1, 3, 16), col)
        yb = 4 + 20 * len(self.rows)
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QRectF(x0, yb, 80, 14), Qt.AlignmentFlag.AlignLeft, '0 s')
        p.drawText(QRectF(x1 - 80, yb, 80, 14), Qt.AlignmentFlag.AlignRight, f'{T:.1f} s')
        p.drawText(QRectF(x0 + 80, yb, x1 - x0 - 160, 14), Qt.AlignmentFlag.AlignCenter,
                   'red: collision · amber: rule violation · orange: put back on the road · blue: outside the ODD · grey: '
                   'triggers, min TTC')
        p.setPen(QPen(QColor(THEME.tok['ink']), 1.5))
        p.drawLine(QPointF(X(self.t), 2), QPointF(X(self.t), yb))


class CompareTab(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.runs: list = []                 # [{'slot', 'path', 'entry', 'frames', 'ts', 'camera', 'events', 'rules'}]
        self.entries: list = []
        self._t = 0.0
        v = QVBoxLayout(self)
        pick = QHBoxLayout()
        self.combos = []
        for s in SLOTS:
            c = QComboBox()
            c.setMinimumContentsLength(18)
            c.activated.connect(lambda _i: self.load())
            pick.addWidget(label(s, muted=True))
            pick.addWidget(c, 1)
            self.combos.append(c)
        pick.addWidget(button('Browse\'s selection', self.from_browse,
                              tip='Compare the runs selected in Data ▸ Browse (up to four)'))
        v.addLayout(pick)
        self.empty = EmptyState('Choose two to four runs above (or select them in Browse and press Compare): their '
                                'cameras play side by side on one clock, with their events and a table of what each '
                                'recorded.')
        self.body = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        grid = QGridLayout()
        self.views, self.captions = [], []
        for k in range(4):
            cell = QVBoxLayout()
            view = ImageView()
            view.setMinimumSize(192, 108)
            cap = label('', muted=True, mono=True)
            cell.addWidget(view, 1)
            cell.addWidget(cap)
            grid.addLayout(cell, k // 2, k % 2)
            self.views.append(view)
            self.captions.append(cap)
        tv.addLayout(grid, 1)
        sc = QHBoxLayout()
        sc.addWidget(button('◀', lambda: self.nudge(-1), tip='One step back (the smallest step of the runs)'))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(lambda ms: self.seek(ms / 1000.0, from_slider=True))
        sc.addWidget(self.slider, 1)
        sc.addWidget(button('▶', lambda: self.nudge(1), tip='One step forward'))
        self.time_lbl = label('', mono=True)
        sc.addWidget(self.time_lbl)
        tv.addLayout(sc)
        self.timeline = EventTimeline()
        self.timeline.seek.connect(self.seek)
        tv.addWidget(self.timeline)
        self.body.addWidget(top)
        sec = Section('What each run recorded', 'The difference is against A. "not recorded" means the run did not '
                                                'record it (a plain drive has no scenario, so no collision check).')
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setMinimumHeight(90)
        sec.add(self.table)
        self.body.addWidget(sec)
        self.body.setSizes([520, 300])
        v.addWidget(self.empty, 1)
        v.addWidget(self.body, 1)
        self.body.hide()

    # ------------------------------------------------------------------ choosing
    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def refresh(self):
        self.entries = [e for e in index_entries(self.win) if e.get('frames')]
        for k, c in enumerate(self.combos):
            cur = c.currentData()
            c.blockSignals(True)
            c.clear()
            c.addItem('—', None)
            for e in self.entries:
                c.addItem(f'{e["name"]}  ·  {(e.get("started") or "")[:16].replace("T", " ")}', e['path'])
                c.setItemData(c.count() - 1, e['rel'], Qt.ItemDataRole.ToolTipRole)
            i = c.findData(cur)
            c.setCurrentIndex(i if i >= 0 else 0)
            c.blockSignals(False)

    def from_browse(self):
        self.set_runs(self.page.browse.selected_runs())

    def set_runs(self, paths):
        if not self.entries:
            self.refresh()
        paths = list(dict.fromkeys(paths))[:4]
        for k, c in enumerate(self.combos):
            i = c.findData(paths[k]) if k < len(paths) else 0
            c.setCurrentIndex(i if i >= 0 else 0)
        self.load()

    def chosen(self) -> list:
        return [c.currentData() for c in self.combos if c.currentData()]

    # ------------------------------------------------------------------ loading
    def load(self):
        by = {e['path']: e for e in self.entries}
        self.runs = []
        for k, c in enumerate(self.combos):
            p = c.currentData()
            if not p or p not in by:
                continue
            frames = RC.load_frames(p)
            cams = RC.cameras(frames)
            self.runs.append({'slot': SLOTS[k], 'path': p, 'entry': by[p], 'frames': frames,
                              'ts': RC.times(frames), 'camera': cams[0] if cams else None,
                              'events': RC.events(p, by[p]), 'rules': None})
        ok = len(self.runs) >= 2
        self.empty.setVisible(not ok)
        self.body.setVisible(ok)
        if not ok:
            self.empty.set_message('Choose at least two runs above (or select them in Browse and press Compare).'
                                   if self.runs else self.empty.message.text())
            return
        T = max(float(r['ts'][-1]) if len(r['ts']) else 0.0 for r in self.runs)
        steps = [float(np.median(np.diff(r['ts']))) for r in self.runs if len(r['ts']) > 1]
        self.step = min(steps) if steps else 0.05
        self.slider.blockSignals(True)
        self.slider.setRange(0, int(round(T * 1000)))
        self.slider.setSingleStep(max(int(round(self.step * 1000)), 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        for k in range(4):
            self.views[k].setVisible(k < len(self.runs))
            self.captions[k].setVisible(k < len(self.runs))
        self.show_table()
        self.show_timeline()
        self.seek(0.0)
        for r in self.runs:
            path = r['path']
            tasks.run(self, lambda path=path: (path, RC.rule_metrics(path)), self._rules_done,
                      lambda e, path=path: self._rules_done((path, {'error': str(e)})))

    def _rules_done(self, res):
        path, rm = res
        for r in self.runs:
            if r['path'] == path:
                r['rules'] = rm
        self.show_table()
        self.show_timeline()

    def show_table(self):
        rows = RC.table([r['entry'] for r in self.runs], [r['rules'] for r in self.runs])
        self.table.clear()
        self.table.setHeaderLabels([''] + [f'{r["slot"]}: {r["entry"]["name"]}' for r in self.runs])
        for row in rows:
            cells = [c['text'] + (f'  ({c["delta"]})' if c['delta'] else '') for c in row['cells']]
            it = QTreeWidgetItem([row['label']] + cells)
            for k, c in enumerate(row['cells']):
                if c['value'] is None:
                    it.setForeground(k + 1, QColor(self.palette().placeholderText().color()))
            self.table.addTopLevelItem(it)
        rs = next((r['rules'] for r in self.runs if r['rules'] and not r['rules'].get('error')), None)
        if rs:
            it = QTreeWidgetItem(['Rule set'] + [(r['rules'] or {}).get('ruleset') or '' for r in self.runs])
            self.table.addTopLevelItem(it)
        hdr = self.table.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for k in range(1, len(self.runs) + 1):
            hdr.setSectionResizeMode(k, QHeaderView.ResizeMode.Stretch)

    def show_timeline(self):
        rows = []
        for r in self.runs:
            evs = list(r['events']) + list(((r['rules'] or {}).get('events') or ()))
            rows.append((r['slot'], float(r['ts'][-1]) if len(r['ts']) else 0.0, evs))
        self.timeline.set_rows(rows)

    # ------------------------------------------------------------------ the clock
    def nudge(self, k):
        self.seek(max(self._t + k * self.step, 0.0))

    def seek(self, t: float, from_slider: bool = False):
        self._t = float(t)
        if not from_slider:
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(t * 1000)))
            self.slider.blockSignals(False)
        self.time_lbl.setText(f't {t:6.2f} s')
        self.timeline.set_time(t)
        want = []
        for k, r in enumerate(self.runs):
            i = RC.frame_at(r['ts'], t)
            if i is None:
                end = float(r['ts'][-1]) if len(r['ts']) else 0.0
                self.captions[k].setText(f'{r["slot"]}: ended at {end:.2f} s')
                want.append(None)
                continue
            f = r['frames'][i]
            sp = f.get('speed_mps')
            self.captions[k].setText(f'{r["slot"]}: t {r["ts"][i]:.2f} s · step {f.get("step")}'
                                     + (f' · {sp * 3.6:.0f} km/h' if sp is not None else ''))
            rel = (f.get('files') or {}).get(f'{r["camera"]}/rgb') if r['camera'] else None
            want.append(os.path.join(r['path'], rel) if rel else None)

        def load():
            import cv2
            out = []
            for p in want:
                img = cv2.imread(p) if p else None
                out.append(img[:, :, ::-1].copy() if img is not None else None)
            return t, out
        tasks.run(self, load, self._shown)

    def _shown(self, res):
        t, imgs = res
        if abs(t - self._t) > 1e-9:
            return                            # a newer time was asked for meanwhile
        for k, img in enumerate(imgs):
            self.views[k].set_rgb(img)
