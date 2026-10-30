"""The map editor for an imported world, and re-importing a world with the current converter.

The editor draws the converted lanes (graph.json) under the OpenStreetMap ways they came from; selecting a road or a
junction shows what can be changed. Every change is a project edit to the world's osm.edits list (core.mapedit), so it
is undoable; *Apply* converts the area again with the list applied (a background job), into a new folder of the
world, and points the world at it — one more project edit, so undo returns to the previous files, which are kept."""
from __future__ import annotations

import copy
import json
import math
import os

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox,
                             QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ..core import mapedit as ME
from ..core import osm as O
from ..core.jobs import FINAL
from ..core.project import Edit, new_id
from ..core.worlds import world_dir as world_root
from .osmimport import RoadGraphView
from .theme import STATUS, THEME
from .widgets import Form, KeyValues, Note, Section, button, label, scroll, set_enabled

LIMITS = ('What CARLA 0.9.16 does with an edit (measured, docs/worlds.md): lanes, direction, road type and removed '
          'roads are geometry, built by the converter and driven as built. A speed limit is written as the lanes\' '
          'speed records; CARLA\'s traffic manager ignores them (30 km/h: a generated world has no speed-limit signs), '
          'so Falcon gives each vehicle its road\'s limit itself. Traffic signals become traffic lights that control '
          'their lanes. Speed breakers are elevation profiles the vehicle physics drives over. Applying '
          'also converts the area with the current converter.')
TYPES = O.CARLA_HIGHWAYS
DIRS = (('no', 'Two-way'), ('yes', 'One-way, along the way'), ('-1', 'One-way, against the way'))


def _world(win, world_id):
    for i, w in enumerate(win.project.doc['worlds']):
        if w['id'] == world_id:
            return i, w
    return None, None


def world_files(win, w) -> str:
    return os.path.join(win.workspace(), os.path.dirname(w.get('xodr', '')))


def base_data(folder: str) -> O.OsmData:
    """The area as imported, before any edits: area.source.osm when edits were applied, else area.osm."""
    for name in ('area.source.osm', 'area.osm'):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return O.read_osm(p)
    raise FileNotFoundError(f'no area.osm in {folder}')


def read_report(folder: str) -> dict:
    try:
        with open(os.path.join(folder, 'import.json'), encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def applied_hash(rep: dict) -> str:
    return (rep.get('map_edits') or {}).get('sha256') or ME.edits_hash([])


class MapCanvas(RoadGraphView):
    """The converted lanes with the OSM ways over them; wheel zooms about the cursor, drag pans, a click picks the
    nearest junction or road (or, in breaker mode, places a speed breaker on the nearest road)."""
    picked = pyqtSignal(str, int)             # 'way' | 'node' | '' , id
    placed = pyqtSignal(int, float, float)    # way id, x, y (metres)

    def __init__(self):
        super().__init__()
        self.bottom_pad = 22
        self.geo = {'ways': [], 'junctions': [], 'signals': [], 'calming': []}
        self.sel = ('', 0)
        self.edited, self.removed = set(), set()
        self.mode = 'select'
        self.zoom, self.centre = 1.0, None
        self._press = None
        self.setMouseTracking(False)
        self.setMinimumSize(420, 360)

    # ------------------------------------------------------------------ view
    def set_overlay(self, geo: dict, edited=(), removed=(), removed_xy=()):
        self.geo = geo
        self.edited, self.removed = set(edited), set(removed)
        self.removed_xy = list(removed_xy)
        if self.bounds is None and geo['ways']:
            xs = [x for w in geo['ways'] for x, _ in w['xy']]
            ys = [y for w in geo['ways'] for _, y in w['xy']]
            self.bounds = (min(xs), min(ys), max(xs), max(ys))
        self.update()

    def base_k(self):
        x0, y0, x1, y1 = self.bounds
        return min((self.width() - 20) / max(x1 - x0, 1e-6),
                   (self.height() - 20 - self.bottom_pad) / max(y1 - y0, 1e-6))

    def fit(self):
        if self.bounds is None:
            return 1.0, 0.0, 0.0
        x0, y0, x1, y1 = self.bounds
        k = self.base_k() * self.zoom
        cx, cy = self.centre or ((x0 + x1) / 2, (y0 + y1) / 2)
        return k, cx - (self.width() / 2 - 10) / k, cy + ((self.height() - self.bottom_pad) / 2 - 10) / k

    def reset_view(self):
        self.zoom, self.centre = 1.0, None
        self.update()

    def zoom_at(self, factor, p: QPointF | None = None):
        if self.bounds is None:
            return
        p = p or QPointF(self.width() / 2, (self.height() - self.bottom_pad) / 2)
        mx, my = self.to_m(p)
        self.zoom = max(0.5, min(200.0, self.zoom * factor))
        k, x0, y1 = self.fit()
        nx, ny = self.to_m(p)
        cx, cy = self.centre or ((self.bounds[0] + self.bounds[2]) / 2, (self.bounds[1] + self.bounds[3]) / 2)
        self.centre = (cx + mx - nx, cy + my - ny)
        self.update()

    def focus(self, x, y):
        self.centre = (x, y)
        self.update()

    def wheelEvent(self, ev):
        self.zoom_at(1.25 if ev.angleDelta().y() > 0 else 0.8, ev.position())

    def mousePressEvent(self, ev):
        self._press = (ev.position(), self.centre or ((self.bounds[0] + self.bounds[2]) / 2,
                                                      (self.bounds[1] + self.bounds[3]) / 2) if self.bounds else None)

    def mouseMoveEvent(self, ev):
        if self._press is None or self._press[1] is None:
            return
        p0, c0 = self._press
        k = self.fit()[0]
        d = ev.position() - p0
        if abs(d.x()) + abs(d.y()) > 3:
            self.centre = (c0[0] - d.x() / k, c0[1] + d.y() / k)
            self._dragged = True
            self.update()

    def mouseReleaseEvent(self, ev):
        dragged = getattr(self, '_dragged', False)
        self._dragged = False
        self._press = None
        if not dragged:
            self.click(ev.position())

    def click(self, p: QPointF):
        x, y = self.to_m(p)
        if self.mode == 'breaker':
            w, d = self.nearest_way(p)
            if w is not None and d <= 12:
                self.placed.emit(w['id'], x, y)
            return
        kind, ident = self.pick(p)
        self.picked.emit(kind, ident)

    # ------------------------------------------------------------------ picking
    def nearest_way(self, p: QPointF):
        best, bd = None, 1e9
        for w in self.geo['ways']:
            pts = [self.to_px(x, y) for x, y in w['xy']]
            for a, b in zip(pts, pts[1:]):
                dx, dy = b.x() - a.x(), b.y() - a.y()
                L2 = dx * dx + dy * dy
                t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / L2))
                d = math.hypot(a.x() + t * dx - p.x(), a.y() + t * dy - p.y())
                if d < bd:
                    best, bd = w, d
        return best, bd

    def pick(self, p: QPointF):
        best, bd = ('', 0), 9.0
        for group in ('signals', 'calming', 'junctions'):
            for n in self.geo[group]:
                q = self.to_px(*n['xy'])
                d = math.hypot(q.x() - p.x(), q.y() - p.y())
                if d < bd:
                    best, bd = ('node', n['id']), d
        if best[0]:
            return best
        w, d = self.nearest_way(p)
        return ('way', w['id']) if w is not None and d <= 8 else ('', 0)

    # ------------------------------------------------------------------ drawing
    def paint_extra(self, p):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        muted, ink = QColor(THEME.tok['muted']), QColor(THEME.tok['ink'])
        edited = THEME.cam(1)
        for w in self.geo['ways']:
            pts = [self.to_px(x, y) for x, y in w['xy']]
            sel = self.sel == ('way', w['id'])
            if sel:
                pen = QPen(ink, 4)
            elif w['id'] in self.edited:
                pen = QPen(edited, 2.5)
            else:
                c = QColor(muted)
                c.setAlphaF(0.55)
                pen = QPen(c, 1)
            p.setPen(pen)
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        dash = QPen(QColor(STATUS['critical']), 2, Qt.PenStyle.DashLine)
        p.setPen(dash)
        for xy in getattr(self, 'removed_xy', []):
            pts = [self.to_px(x, y) for x, y in xy]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        p.setPen(QPen(QColor(THEME.tok['strong']), 1))
        p.setBrush(QColor(THEME.tok['panel']))
        for n in self.geo['junctions']:
            p.drawEllipse(self.to_px(*n['xy']), 3, 3)
        p.setBrush(QColor(STATUS['critical']))
        p.setPen(QPen(QColor(THEME.tok['ink']), 1))
        for n in self.geo['signals']:
            p.drawEllipse(self.to_px(*n['xy']), 5, 5)
        p.setBrush(QColor(STATUS['warning']))
        for n in self.geo['calming']:
            q = self.to_px(*n['xy'])
            p.drawRect(int(q.x()) - 5, int(q.y()) - 2, 10, 4)
        if self.sel[0] == 'node':
            for group in ('signals', 'calming', 'junctions'):
                for n in self.geo[group]:
                    if n['id'] == self.sel[1]:
                        p.setBrush(Qt.BrushStyle.NoBrush)
                        p.setPen(QPen(ink, 2))
                        p.drawEllipse(self.to_px(*n['xy']), 9, 9)
        p.fillRect(0, self.height() - self.bottom_pad, self.width(), self.bottom_pad, QColor(THEME.tok['viewport']))
        p.setPen(muted)
        p.drawText(10, self.height() - 8, 'lanes: colour and width by lanes per direction · thin grey: OSM ways · '
                                          'orange: edited · dashed red: removed · red dot: signals · amber: breakers')


class MapEditorDialog(QDialog):
    """Edit an imported world's roads (non-modal): select, change, apply."""

    def __init__(self, win, world_id: str):
        super().__init__(win)
        self.win, self.world_id = win, world_id
        self.job = None
        self.setWindowTitle('Edit the map')
        self.resize(1280, 820)
        i, w = _world(win, world_id)
        self.folder = world_files(win, w)
        self.base = base_data(self.folder)
        self.types = tuple((w['osm'].get('settings') or {}).get('way_types') or TYPES)
        rep = read_report(self.folder)
        self.proj = O.local_projection(*(rep.get('centre') or w['osm'].get('centre') or O.summary(self.base)['centre']))
        self.base_geo = ME.geometry(self.base, self.proj, self.types)

        v = QVBoxLayout(self)
        split = QSplitter()
        v.addWidget(split, 1)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        tb = QHBoxLayout()
        self.select_btn = button('Select', lambda: self.set_mode('select'), tip='Click a road or a junction (Esc)')
        self.breaker_btn = button('Add a speed breaker', lambda: self.set_mode('breaker'),
                                  tip='Click on a road to place one there')
        self.breaker_kind = QComboBox()
        for k, t in (('hump', 'hump (IRC speed breaker)'), ('bump', 'bump (short)'), ('table', 'table')):
            self.breaker_kind.addItem(t, k)
        for b in (self.select_btn, self.breaker_btn):
            b.setCheckable(True)
        tb.addWidget(self.select_btn)
        tb.addWidget(self.breaker_btn)
        tb.addWidget(self.breaker_kind)
        tb.addStretch(1)
        tb.addWidget(button('Zoom in', lambda: self.canvas.zoom_at(1.5)))
        tb.addWidget(button('Zoom out', lambda: self.canvas.zoom_at(1 / 1.5)))
        tb.addWidget(button('Fit', lambda: self.canvas.reset_view(), tip='Show the whole area (F)'))
        lv.addLayout(tb)
        self.canvas = MapCanvas()
        self.canvas.picked.connect(self.select)
        self.canvas.placed.connect(self.place_breaker)
        lv.addWidget(self.canvas, 1)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        self.insp = Section('Selection')
        self.sel_title = label('Click a road or a junction on the map.', muted=True, wrap=True)
        self.insp.add(self.sel_title)
        self.form_w = QWidget()
        f = Form()
        self.form_w.setLayout(f)
        self.f = f
        self.road_type = QComboBox()
        for t in self.types:
            self.road_type.addItem(t, t)
        self.road_type.activated.connect(lambda _: self.edit_way('highway', value=self.road_type.currentData()))
        f.row('Road type', self.road_type, 'The converter\'s default lanes and speed follow the type')
        self.direction = QComboBox()
        for k, t in DIRS:
            self.direction.addItem(t, k)
        self.direction.activated.connect(lambda _: self.edit_way('oneway', value=self.direction.currentData()))
        f.row('Direction', self.direction)
        self.fwd = QSpinBox()
        self.bwd = QSpinBox()
        for sb in (self.fwd, self.bwd):
            sb.setRange(0, ME.MAX_LANES)
            sb.editingFinished.connect(self.edit_lanes)
        f.row('Lanes forward', self.fwd, 'Along the way\'s node order')
        f.row('Lanes backward', self.bwd)
        self.speed = QSpinBox()
        self.speed.setRange(0, ME.SPEED_KMH[1])
        self.speed.setSuffix(' km/h')
        self.speed.setSpecialValueText('the type\'s default')
        self.speed.editingFinished.connect(self.edit_speed)
        f.row('Speed limit', self.speed)
        self.insp.add(self.form_w)
        self.signal_cb = QCheckBox('Traffic signals here')
        self.signal_cb.toggled.connect(self.edit_signal)
        self.insp.add(self.signal_cb)
        rb = QHBoxLayout()
        self.remove_btn = button('Remove this road', self.remove_way)
        self.unbreak_btn = button('Remove this speed breaker', self.remove_breaker)
        rb.addWidget(self.remove_btn)
        rb.addWidget(self.unbreak_btn)
        rb.addStretch(1)
        self.insp.add(rb)
        self.honour = label('', muted=True, wrap=True, small=True)
        self.insp.add(self.honour)
        rv.addWidget(self.insp)

        es = Section('Edits', 'Applied in this order on top of the imported area; kept by a re-import.')
        self.edits_list = QListWidget()
        self.edits_list.setWordWrap(True)
        self.edits_list.setMinimumHeight(90)
        self.edits_list.setMaximumHeight(170)
        self.edits_list.currentRowChanged.connect(self.show_edit)
        es.add(self.edits_list)
        eb = QHBoxLayout()
        self.drop_btn = button('Remove the selected edit', self.drop_edit)
        eb.addWidget(self.drop_btn)
        eb.addStretch(1)
        es.add(eb)
        self.warn = label('', wrap=True)
        es.add(self.warn)
        rv.addWidget(es)

        ap = Section('Apply', LIMITS)
        self.state = label('', wrap=True)
        ap.add(self.state)
        ab = QHBoxLayout()
        self.apply_btn = button('Apply the edits', self.apply, role='primary',
                                tip='Convert the area again with the edits (a background job; the current files are '
                                    'kept, undo returns to them)')
        ab.addWidget(self.apply_btn)
        ab.addStretch(1)
        ap.add(ab)
        self.results = KeyValues()
        ap.add(self.results)
        rv.addWidget(ap)
        rv.addStretch(1)
        split.addWidget(scroll(right))
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 460])
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(button('Close', self.close))
        v.addLayout(bottom)

        win.projectChanged.connect(self.on_project)
        win.bridge.finished.connect(self.on_job)
        self.set_mode('select')
        self.refresh()
        self.show_results(read_report(self.folder), None)

    # ------------------------------------------------------------------ state
    def world(self):
        return _world(self.win, self.world_id)

    def edits(self) -> list:
        _, w = self.world()
        return list(((w or {}).get('osm') or {}).get('edits') or [])

    def on_project(self, path):
        i, w = self.world()
        if w is None:
            self.close()
            return
        folder = world_files(self.win, w)
        if folder != self.folder:               # applied, or undone back to other files
            self.folder = folder
            self.show_results(read_report(folder), None)
        self.refresh()

    def refresh(self):
        edits = self.edits()
        self.data, self.rep = ME.apply_edits(self.base, edits, self.types)
        geo = ME.geometry(self.data, self.proj, self.types)
        removed = set(self.rep['removed_ways'])
        removed_xy = [w['xy'] for w in self.base_geo['ways'] if w['id'] in removed]
        try:
            with open(os.path.join(self.folder, 'graph.json'), encoding='utf-8') as fh:
                graph = json.load(fh)
        except (OSError, ValueError):
            graph = None
        if graph is not None:
            bounds = self.canvas.bounds              # keep the view where it is
            self.canvas.set_graph(graph)
            self.canvas.bounds = bounds or self.canvas.bounds
        self.canvas.set_overlay(geo, self.rep['touched_ways'], removed, removed_xy)
        self.edits_list.clear()
        skipped = {s['index']: s for s in self.rep['skipped']}
        for k, e in enumerate(edits):
            it = QListWidgetItem(ME.describe(e, self.base))
            if k in skipped:
                it.setText(f'⚠ {it.text()} — not applied: {"; ".join(skipped[k]["reasons"])}')
            it.setToolTip(it.text())
            self.edits_list.addItem(it)
        set_enabled(self.drop_btn, bool(edits), 'No edits yet')
        cut = self.rep['connectivity']['cut_off']
        warns = []
        if cut:
            names = sorted({(ME.way_by_id(self.base, c) or {}).get('tags', {}).get('name') or f'way {c}' for c in cut})
            warns.append(f'⚠ {len(cut)} road(s) no longer connect to the rest of the network: {", ".join(names[:6])}'
                         + ('…' if len(names) > 6 else '') + ' — vehicles there cannot leave')
        if self.rep['skipped']:
            warns.append(f'⚠ {len(self.rep["skipped"])} edit(s) cannot be applied (marked above)')
        self.warn.setText('\n'.join(warns))
        self.warn.setStyleSheet(f'color: {STATUS["warning"]};' if warns else '')
        rep = read_report(self.folder)
        up = applied_hash(rep) == ME.edits_hash(edits)
        if up:
            self.state.setText('The world is up to date with these edits.' if edits else
                               'No edits yet: select a road or a junction to change it.')
        else:
            self.state.setText(f'{len(edits)} edit(s); the world\'s files do not have them yet — apply to convert '
                               'again.' if edits else 'The world\'s files still have edits that were removed — apply '
                                                     'to convert without them.')
        busy = self.job is not None and self.job.state not in FINAL
        set_enabled(self.apply_btn, not busy and not up,
                    'Converting…' if busy else ('The world already has these edits' if edits else 'No edits to apply'))
        self.select(*self.canvas.sel)

    # ------------------------------------------------------------------ selection
    def set_mode(self, mode):
        self.canvas.mode = mode
        self.select_btn.setChecked(mode == 'select')
        self.breaker_btn.setChecked(mode == 'breaker')
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if mode == 'breaker' else Qt.CursorShape.ArrowCursor)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape:
            self.set_mode('select')
            self.select('', 0)
            return
        if ev.key() == Qt.Key.Key_F:
            self.canvas.reset_view()
            return
        if ev.key() == Qt.Key.Key_Delete and self.canvas.sel[0] == 'way':
            self.remove_way()
            return
        super().keyPressEvent(ev)

    def select(self, kind, ident):
        self.canvas.sel = (kind, ident)
        self.canvas.update()
        way = ME.way_by_id(self.data, ident) if kind == 'way' else None
        node = ident if kind == 'node' and ident in self.data.nodes else None
        self.form_w.setVisible(way is not None)
        self.remove_btn.setVisible(way is not None)
        junc = ME.junction_nodes(self.data, self.types)
        tags = self.data.node_tags.get(node, {}) if node is not None else {}
        has_sig = tags.get('highway') == 'traffic_signals'
        self.signal_cb.setVisible(node is not None and (node in junc or has_sig))
        self.unbreak_btn.setVisible(node is not None and 'traffic_calming' in tags)
        if way is None and node is None:
            self.sel_title.setText('Click a road or a junction on the map.' if kind != 'way' or ident not in
                                   set(self.rep['removed_ways']) else f'Way {ident} is removed (remove the edit to '
                                                                      'bring it back).')
            self.honour.setText('')
            return
        if way is not None:
            t = way['tags']
            self.sel_title.setText(f'{t.get("name") or "an unnamed road"} — OSM way {way["id"]}, '
                                   f'{O.way_length_m(self.data, way):.0f} m')
            for wd in (self.road_type, self.direction, self.fwd, self.bwd, self.speed):
                wd.blockSignals(True)
            k = self.road_type.findData(t.get('highway'))
            self.road_type.setCurrentIndex(max(k, 0))
            ow = O.oneway(t)
            self.direction.setCurrentIndex({0: 0, 1: 1, -1: 2}.get(ow, 0))
            fb = O.expected_lanes(way)
            self.fwd.setValue(fb[0] if fb else 0)
            self.bwd.setValue(fb[1] if fb else 0)
            for sb in (self.fwd, self.bwd):
                sb.setSpecialValueText('untagged' if not fb else '')
            sp = t.get('maxspeed', '')
            try:
                self.speed.setValue(int(float(sp.split()[0])) if sp else 0)
            except (ValueError, IndexError):
                self.speed.setValue(0)
            for wd in (self.road_type, self.direction, self.fwd, self.bwd, self.speed):
                wd.blockSignals(False)
            self.honour.setText('Lanes, direction and type: ' + ME.HONOURED['lanes'] + '. Speed limit: '
                                + ME.HONOURED['maxspeed'] + '.')
        else:
            ways = junc.get(node, [])
            what = []
            if tags.get('highway') == 'traffic_signals':
                what.append('traffic signals')
            if tags.get('traffic_calming'):
                what.append(f'traffic_calming={tags["traffic_calming"]}')
            self.sel_title.setText(f'Node {node}' + (f' — a junction of {len(ways)} roads' if ways else '')
                                   + (f' ({", ".join(what)})' if what else ''))
            self.signal_cb.blockSignals(True)
            self.signal_cb.setChecked(tags.get('highway') == 'traffic_signals')
            self.signal_cb.blockSignals(False)
            self.honour.setText(' '.join(t for t in (
                'Signals: ' + ME.HONOURED['signal'] + '.' if node in junc or has_sig else '',
                'Speed breakers: ' + ME.HONOURED['speed_breaker'] + '.' if tags.get('traffic_calming') else '') if t))

    # ------------------------------------------------------------------ edits
    def push(self, e: dict, text: str):
        errs = ME.check_edit(self.data, e, self.types)
        if errs:
            QMessageBox.warning(self, 'Edit the map', '\n'.join(errs))
            self.select(*self.canvas.sel)
            return False
        i, w = self.world()
        new = ME.upsert(self.edits(), e)
        self.win.edit(Edit('set', ['worlds', i, 'osm', 'edits'], new, text=f'{w["name"]}: {text}'))
        return True

    def edit_way(self, op, **kw):
        kind, ident = self.canvas.sel
        if kind != 'way':
            return
        e = dict(op=op, way=ident, **kw)
        self.push(e, ME.describe(e, self.base))

    def edit_lanes(self):
        kind, ident = self.canvas.sel
        way = ME.way_by_id(self.data, ident) if kind == 'way' else None
        if way is None:
            return
        fb = O.expected_lanes(way)
        f, b = self.fwd.value(), self.bwd.value()
        if fb == (f, b) or (fb is None and f == 0 and b == 0):
            return
        self.edit_way('lanes', forward=f, backward=b)

    def edit_speed(self):
        kind, ident = self.canvas.sel
        way = ME.way_by_id(self.data, ident) if kind == 'way' else None
        if way is None:
            return
        v = self.speed.value() or None
        cur = way['tags'].get('maxspeed')
        if (v is None and not cur) or (v is not None and cur == f'{float(v):g}'):
            return
        self.edit_way('maxspeed', kmh=v)

    def remove_way(self):
        kind, ident = self.canvas.sel
        if kind == 'way' and ME.way_by_id(self.data, ident) is not None:
            self.edit_way('remove')

    def edit_signal(self, on):
        kind, ident = self.canvas.sel
        if kind == 'node':
            e = {'op': 'signal', 'node': ident, 'on': bool(on)}
            self.push(e, ME.describe(e))

    def remove_breaker(self):
        kind, ident = self.canvas.sel
        if kind != 'node':
            return
        edits = self.edits()
        if ident not in self.base.nodes:         # placed by an edit: take that edit out
            k = self._breaker_edit(ident, edits)
            if k is not None:
                self.drop(k)
            return
        e = {'op': 'remove_feature', 'node': ident}
        self.push(e, ME.describe(e))

    def _breaker_edit(self, node, edits):
        """The speed_breaker edit that made a new node (nodes are numbered in the list's order)."""
        first = max(int(n) for n in self.base.nodes) + 1
        made = [k for k, e in enumerate(edits) if e['op'] == 'speed_breaker' and
                k not in {s['index'] for s in self.rep['skipped']}]
        j = node - first
        return made[j] if 0 <= j < len(made) else None

    def place_breaker(self, way_id, x, y):
        lat, lon = self.proj.to_geo(x, y)
        e = {'op': 'speed_breaker', 'way': way_id, 'lat': round(float(lat), 7), 'lon': round(float(lon), 7),
             'value': self.breaker_kind.currentData()}
        self.push(e, ME.describe(e, self.base))

    def show_edit(self, k):
        edits = self.edits()
        if not 0 <= k < len(edits):
            return
        e = edits[k]
        if 'node' in e and e['node'] in self.data.nodes:
            self.select('node', int(e['node']))
            xy = self.proj.to_local(*self.data.nodes[int(e['node'])])
        elif 'way' in e:
            self.select('way', int(e['way']))
            way = next((w for w in self.base_geo['ways'] if w['id'] == int(e['way'])), None)
            xy = way['xy'][len(way['xy']) // 2] if way else None
        else:
            return
        if xy is not None:
            self.canvas.focus(float(xy[0]), float(xy[1]))

    def drop_edit(self):
        k = self.edits_list.currentRow()
        if k >= 0:
            self.drop(k)

    def drop(self, k):
        i, w = self.world()
        edits = self.edits()
        if not 0 <= k < len(edits):
            return
        text = ME.describe(edits[k], self.base)
        del edits[k]
        self.win.edit(Edit('set', ['worlds', i, 'osm', 'edits'], edits, text=f'{w["name"]}: undo "{text}"'))

    # ------------------------------------------------------------------ apply
    def apply(self):
        i, w = self.world()
        edits = self.edits()
        root = world_root(self.win.workspace(), w['id'])
        out = os.path.join(root, f'edit-{ME.edits_hash(edits)[:10]}')
        try:
            cfg = ME.import_config_for(w, self.folder, out, edits)
        except FileNotFoundError as e:
            QMessageBox.warning(self, 'Apply the edits', str(e))
            return
        self.cfg = cfg
        self.prev = read_report(self.folder)
        try:
            self.job = self.win.service.submit_osm_import(cfg, title=f'Apply map edits: {w["name"]}')
        except RuntimeError as e:
            QMessageBox.warning(self, 'Apply the edits', str(e))
            return
        self.state.setText('Converting with the edits…')
        set_enabled(self.apply_btn, False, 'Converting…')

    def on_job(self, job):
        if job is not self.job:
            return
        if job.state != 'done':
            tail = '; '.join(t for lvl, t in job.log[-5:] if lvl in ('stderr', 'error', 'warning'))
            self.state.setText(f'⚠ {job.state}: {tail}')
            self.refresh()
            return
        from ..core.worlds import edited_world
        rep = read_report(self.cfg['out'])
        i, w = self.world()
        if w is None:
            return
        nw = edited_world(w, self.cfg, rep, self.win.workspace())
        self.win.edit(Edit('set', ['worlds', i], nw, text=f'Apply map edits to {w["name"]}'))
        self.show_results(rep, self.prev)

    def show_results(self, rep: dict, prev: dict | None):
        self.results.clear()
        if not rep:
            return
        show_checks(self.results, rep, prev, lht=bool(((self.world()[1] or {}).get('osm') or {}).get('lht')))


def _pos(rep):
    return ((rep or {}).get('lane_check') or {}).get('position') or {}


def show_checks(kv: KeyValues, rep: dict, prev: dict | None = None, lht: bool | None = None):
    """The checks of a conversion (and, with prev, against the one before): edited roads' lanes, the lane check's
    counts and positions, lane links, handedness, signals."""
    me = rep.get('map_edits') or {}
    if me:
        kv.set('Edits', f'{len(me.get("applied", []))} of {me.get("count", 0)} applied'
                        + (f', {len(me["skipped"])} skipped' if me.get('skipped') else '')
                        + f' (list {me.get("sha256", "")[:12]})')
        lc = me.get('lane_check') or {}
        if lc.get('rows'):
            bad = [r for r in lc['rows'] if r.get('matched') is False]
            kv.set('Lanes of the edited roads', f'{lc["matched"]} of {lc["comparable"]} as edited'
                   + (f'; differ: ' + ', '.join(f'way {r["way"]} expected {tuple(r["expected"])} found '
                                                 f'{tuple(r["found"])}' for r in bad[:4]) if bad else '')
                   + (f'; {len(lc["rows"]) - lc["comparable"]} not found in the conversion'
                      if len(lc['rows']) > lc['comparable'] else ''))
        conn = me.get('connectivity') or {}
        if conn.get('cut_off'):
            kv.set('Disconnected', f'{len(conn["cut_off"])} road(s) cut off from the network by the edits')
    lc = rep.get('lane_check') or {}
    if lc.get('comparable'):
        kv.set('Lanes against OSM', f'{lc["matched"]} of {lc["comparable"]} sampled ways agree')
    p, q = _pos(rep), _pos(prev)
    if p.get('checked'):
        now = f'median {p["median_m"]} m, 95th percentile {p["p95_m"]} m; {p["misplaced"]} of {p["checked"]} misplaced'
        if q.get('checked'):
            now = (f'before: median {q["median_m"]} m, 95th percentile {q["p95_m"]} m, {q["misplaced"]} of '
                   f'{q["checked"]} misplaced → now: {now}')
        kv.set('Lane positions', now)
    ln, lp = rep.get('lane_links') or {}, (prev or {}).get('lane_links') or {}
    if ln.get('links'):
        now = (f'{ln["over_0_5_m"]} of {ln["links"]} lane links apart by more than 0.5 m '
               f'(95th percentile {ln["p95_m"]} m)')
        if lp.get('links'):
            now = f'before {lp["over_0_5_m"]} of {lp["links"]} → now {now}'
        kv.set('Lane links', now)
    sl, sp = rep.get('signal_lanes') or {}, (prev or {}).get('signal_lanes') or {}
    if sl.get('oriented') or sp.get('oriented'):
        def lights(x):
            return (f'{x["no_lane"]} of {x["oriented"]} signal records name a side with no lane (CARLA\'s lights there '
                    'control nothing)' if x.get('no_lane') else f'all {x.get("oriented", 0)} signal records control '
                                                                 'lanes')
        kv.set('Traffic lights', (f'before: {lights(sp)} → now: ' if sp and prev is not None else '') + lights(sl),
               sl.get('method', ''))
    h = rep.get('handedness')
    if h:
        ok = lht is None or (h == 'LHT') == bool(lht)
        kv.set('Driving side', f'{h} on every road' + ('' if ok else ' ⚠ not what the world asks for'))
    t = rep.get('graph_totals') or {}
    tp = (prev or {}).get('graph_totals') or {}
    if t:
        kv.set('Network', f'{t.get("roads")} roads, {t.get("road_km", 0):.2f} km, {t.get("junctions")} junctions, '
                          f'{t.get("signals")} signals' + (f' (before: {tp.get("roads")} roads, {tp.get("signals")} '
                                                          'signals)' if tp else ''))
    rf = rep.get('road_features') or {}
    if rf:
        kv.set('Speed breakers', f'{rf.get("placed", 0)} placed, {rf.get("skipped", 0)} skipped')


def open_map_editor(win, world_id: str) -> MapEditorDialog:
    dlg = MapEditorDialog(win, world_id)
    dlg.show()
    return dlg


class ReimportDialog(QDialog):
    """Convert an imported world again with the current converter into a NEW world (the old one stays), and compare
    the lane checks before choosing to add it."""

    def __init__(self, win, world_id: str):
        super().__init__(win)
        self.win, self.world_id = win, world_id
        self.job = self.cfg = None
        _, w = _world(win, world_id)
        self.setWindowTitle(f'Re-import {w["name"]}')
        self.resize(760, 520)
        v = QVBoxLayout(self)
        v.addWidget(label(f'Re-import {w["name"]}', h=True))
        v.addWidget(Note('Converts the same area again with the current converter (same centre, handedness, '
                         'settings and map edits) into a new world; the old world and its runs stay as they are. '
                         'Compare the checks, then add the new world or discard it.'))
        self.kv = KeyValues()
        v.addWidget(self.kv)
        v.addStretch(1)
        row = QHBoxLayout()
        self.go = button('Convert', self.convert, role='primary')
        self.add_btn = button('Add as a new world', self.add_world)
        set_enabled(self.add_btn, False, 'Convert first')
        row.addWidget(self.go)
        row.addWidget(self.add_btn)
        row.addStretch(1)
        row.addWidget(button('Close', self.reject))
        v.addLayout(row)
        self.folder = world_files(win, w)
        self.prev = read_report(self.folder)
        self.new_id = new_id('world')
        try:
            src, _, what = ME.source_for(w, self.folder)
            self.kv.set('Source', f'{what}: {os.path.basename(src)}')
        except FileNotFoundError as e:
            self.kv.set('Source', f'⚠ {e}')
            set_enabled(self.go, False, str(e))
        if not _pos(self.prev).get('checked') or not self.prev.get('lane_links') or not self.prev.get('signal_lanes'):
            self.kv.set('Before', 'measuring the lane positions of the current files…')
            from . import tasks
            tasks.run(self, lambda: measure_checks(self.folder, w), self._measured,
                      lambda e: self.kv.set('Before', f'could not measure: {e}'))
        else:
            self.kv.set('Before', describe(self.prev))
        win.bridge.finished.connect(self.on_job)

    def _measured(self, checks):
        self.prev = dict(self.prev, **checks)
        self.kv.set('Before', describe(self.prev))

    def convert(self):
        _, w = _world(self.win, self.world_id)
        out = world_root(self.win.workspace(), self.new_id)
        try:
            self.cfg = ME.import_config_for(w, self.folder, out)
            self.job = self.win.service.submit_osm_import(self.cfg, title=f'Re-import {w["name"]}')
        except (FileNotFoundError, RuntimeError) as e:
            QMessageBox.warning(self, 'Re-import', str(e))
            return
        set_enabled(self.go, False, 'Converting…')
        self.kv.set('State', 'converting…')

    def on_job(self, job):
        if job is not self.job:
            return
        if job.state != 'done':
            tail = '; '.join(t for lvl, t in job.log[-5:] if lvl in ('stderr', 'error', 'warning'))
            self.kv.set('State', f'⚠ {job.state}: {tail}')
            set_enabled(self.go, True)
            return
        self.rep = read_report(self.cfg['out'])
        self.kv.set('State', 'converted')
        self.kv.set('After', describe(self.rep))
        show_checks(self.kv, self.rep, self.prev, lht=self.cfg['lht'])
        set_enabled(self.add_btn, True)

    def add_world(self):
        from ..core.worlds import world_entry
        _, w = _world(self.win, self.world_id)
        osm = w.get('osm') or {}
        nw = world_entry(self.new_id, f'{w["name"]} (re-imported)', self.cfg, self.rep, osm.get('mesh'),
                         w.get('traffic_profile', 'tp-default'))
        nw['weather'] = copy.deepcopy(w.get('weather') or nw['weather'])
        for k in ('sensor_physics',):
            if k in w:
                nw[k] = copy.deepcopy(w[k])
        if self.cfg.get('edits'):
            nw['osm']['edits'] = list(self.cfg['edits'])
            nw['osm']['applied_edits'] = ME.edits_hash(self.cfg['edits'])
        if osm.get('road_features'):
            nw['osm']['road_features'] = dict(osm['road_features'])
        nw['osm']['reimported_from'] = w['id']
        self.win.edit(Edit('insert', ['worlds'], nw, text=f'Re-import {w["name"]}'))
        self.accept()


def describe_pos(p: dict) -> str:
    if not p.get('checked'):
        return 'no lane position check (no comparable way)'
    return (f'lane positions against OSM: median {p["median_m"]} m, 95th percentile {p["p95_m"]} m; {p["misplaced"]} '
            f'of {p["checked"]} checked ways misplaced')


def describe(rep: dict) -> str:
    """A conversion's checks in a few lines: lane positions, lane links, traffic lights."""
    out = [describe_pos(_pos(rep))]
    ln = rep.get('lane_links') or {}
    if ln.get('links'):
        out.append(f'{ln["over_0_5_m"]} of {ln["links"]} lane links apart by more than 0.5 m')
    sl = rep.get('signal_lanes') or {}
    if sl.get('oriented'):
        out.append(f'{sl["no_lane"]} of {sl["oriented"]} signal records control no lane in CARLA' if sl['no_lane']
                   else f'all {sl["oriented"]} signal records control lanes')
    return '\n'.join(out)


def measure_checks(folder: str, w: dict) -> dict:
    """The lane check (counts and positions) and the lane-link gaps of a world's current files, for a world imported
    before the import measured them: {'lane_check', 'lane_links'} as import.json has them."""
    import numpy as np
    from ..core import xodr as X
    rep = read_report(folder)
    with open(os.path.join(folder, 'world.xodr'), encoding='utf-8') as fh:
        text = fh.read()
    od = X.parse(text)
    data = O.read_osm(os.path.join(folder, 'area.osm'))
    centre = rep.get('centre') or (w.get('osm') or {}).get('centre') or O.summary(data)['centre']
    gaps = X.lane_link_gaps(od)
    return {'lane_check': X.lane_check(od, data, O.local_projection(*centre), sample=50, seed=0),
            'signal_lanes': X.signal_lanes(text),
            'lane_links': {'links': int(len(gaps)), 'over_0_5_m': int((gaps > 0.5).sum()) if len(gaps) else 0,
                           'p95_m': round(float(np.percentile(gaps, 95)), 3) if len(gaps) else None}}


def open_reimport(win, world_id: str) -> ReimportDialog:
    dlg = ReimportDialog(win, world_id)
    dlg.show()
    return dlg
