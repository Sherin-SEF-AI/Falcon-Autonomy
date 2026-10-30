"""Importing a real place from OpenStreetMap: the area picker, the road-graph preview and the import dialog."""
from __future__ import annotations

import json
import math
import os

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QGridLayout,
                             QHBoxLayout, QLineEdit, QListWidget, QMessageBox, QSplitter, QVBoxLayout, QWidget)

from ..core import osm as O
from ..core.jobs import FINAL
from ..core.paths import tilde
from ..core.project import Edit, new_id
from ..core.worlds import MESH_DEFAULTS, import_config, imported_world, world_dir
from . import tasks
from .theme import SEQ, THEME
from .widgets import Form, KeyValues, Section, button, label, set_enabled


class _Canvas(QWidget):
    """Polylines in local metres (x east, y north), fitted to the widget, north up."""

    def __init__(self):
        super().__init__()
        self.lines = []                         # [(xy list, colour, width)]
        self.bounds = None                      # (x0, y0, x1, y1)
        self.bottom_pad = 0                     # room kept under the drawing for a caption
        self.setMinimumSize(320, 260)

    def fit(self):
        if self.bounds is None:
            return 1.0, 0.0, 0.0
        x0, y0, x1, y1 = self.bounds
        k = min((self.width() - 20) / max(x1 - x0, 1e-6),
                (self.height() - 20 - self.bottom_pad) / max(y1 - y0, 1e-6))
        return k, x0, y1

    def to_px(self, x, y):
        k, x0, y1 = self.fit()
        return QPointF(10 + (x - x0) * k, 10 + (y1 - y) * k)

    def to_m(self, p):
        k, x0, y1 = self.fit()
        return x0 + (p.x() - 10) / k, y1 - (p.y() - 10) / k

    def paint_lines(self, p):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for xy, col, w in self.lines:
            p.setPen(QPen(QColor(col), w))
            pts = [self.to_px(x, y) for x, y in xy]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['viewport']))
        self.paint_lines(p)
        self.paint_extra(p)

    def paint_extra(self, p):
        pass


class AreaPicker(_Canvas):
    """The source's roads; drag a rectangle to choose the area (a bbox in degrees)."""
    picked = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.proj = None
        self.rect_m = None
        self.drag = None

    def set_data(self, data: O.OsmData):
        s = O.summary(data)
        lat0, lon0 = s['centre']
        self.proj = O.local_projection(lat0, lon0)
        self.lines = []
        xs, ys = [], []
        col = THEME.tok['ink2']
        for w in data.ways:
            if w.get('kind') != 'highway':
                continue
            pts = [data.nodes[n] for n in w['nodes'] if n in data.nodes]
            xy = [self.proj.to_local(lat, lon) for lat, lon in pts]
            if len(xy) >= 2:
                self.lines.append((xy, col, 1.0))
                xs += [p[0] for p in xy]
                ys += [p[1] for p in xy]
        self.bounds = (min(xs), min(ys), max(xs), max(ys)) if xs else None
        self.update()

    def set_bbox(self, bbox):
        if self.proj is None or bbox is None:
            self.rect_m = None
        else:
            s, w, n, e = bbox
            x0, y0 = self.proj.to_local(s, w)
            x1, y1 = self.proj.to_local(n, e)
            self.rect_m = (x0, y0, x1, y1)
        self.update()

    def mousePressEvent(self, ev):
        if self.proj is not None:
            self.drag = self.to_m(ev.position())

    def mouseMoveEvent(self, ev):
        if self.drag is not None:
            x, y = self.to_m(ev.position())
            self.rect_m = (min(self.drag[0], x), min(self.drag[1], y), max(self.drag[0], x), max(self.drag[1], y))
            self.update()

    def mouseReleaseEvent(self, ev):
        if self.drag is not None and self.rect_m is not None:
            x0, y0, x1, y1 = self.rect_m
            if x1 - x0 > 20 and y1 - y0 > 20:
                s, w = self.proj.to_geo(x0, y0)
                n, e = self.proj.to_geo(x1, y1)
                self.picked.emit((float(s), float(w), float(n), float(e)))
        self.drag = None

    def paint_extra(self, p):
        if self.rect_m is None:
            return
        x0, y0, x1, y1 = self.rect_m
        a, b = self.to_px(x0, y1), self.to_px(x1, y0)
        p.setPen(QPen(QColor(SEQ[6]), 2))
        p.setBrush(QColor(57, 135, 229, 40))
        p.drawRect(QRectF(a, b))
        p.setPen(QColor(THEME.tok['ink']))
        p.drawText(a + QPointF(4, 14), f'{x1 - x0:.0f} × {y1 - y0:.0f} m')


class RoadGraphView(_Canvas):
    """A converted road network: each road coloured by its driving lanes per direction; junction connections thin."""

    def __init__(self):
        super().__init__()
        self.bottom_pad = 22                    # the caption sits under the roads, not on them

    def set_graph(self, g: dict):
        self.lines = []
        conn_col = THEME.tok['strong']
        for c in g.get('connections', []):
            xy = list(zip(c['x'], c['y'])) if 'x' in c else c.get('xy', [])
            if len(xy) >= 2:
                self.lines.append((xy, conn_col, 1.0))
        for r in g.get('roads', []):
            xy = list(zip(r['x'], r['y'])) if 'x' in r else r.get('xy', [])
            n = max(r.get('left', 0), r.get('right', 0))
            col = SEQ[min(3 + 3 * n, len(SEQ) - 1)] if n else THEME.tok['muted']
            if len(xy) >= 2:
                self.lines.append((xy, col, 1.0 + 0.8 * n))
        b = (g.get('totals') or {}).get('bounds')
        self.bounds = tuple(b) if b else None
        self.update()

    def paint_extra(self, p):
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(10, self.height() - 8, 'colour and width: driving lanes per direction (1 – 4+); thin: junctions')


class ImportDialog(QDialog):
    """Choose the source and the area, convert (a job), look at the result, add it as a world."""

    def __init__(self, win, source: str = ''):
        super().__init__(win)
        self.win = win
        self.setWindowTitle('Import a place from OpenStreetMap')
        self.resize(1180, 760)
        self.data = None
        self.job = None
        self.world_id = new_id('world')
        v = QVBoxLayout(self)
        split = QSplitter()
        v.addWidget(split, 1)
        left = QWidget()
        lv = QVBoxLayout(left)
        src = QHBoxLayout()
        self.source = QLineEdit(source)
        self.source.setPlaceholderText('an .osm or .osm.pbf file (offline)')
        src.addWidget(self.source, 1)
        src.addWidget(button('Browse…', self.browse))
        src.addWidget(button('Load', self.load))
        lv.addLayout(src)
        geo = QHBoxLayout()
        self.place = QLineEdit()
        self.place.setPlaceholderText('a place name (network search: OpenStreetMap Nominatim)')
        geo.addWidget(self.place, 1)
        self.geo_btn = button('Search', self.search)
        geo.addWidget(self.geo_btn)
        allowed = bool(win.settings.get('geocoding_allowed'))
        self.place.setEnabled(allowed)
        set_enabled(self.geo_btn, allowed, 'Place-name search is off: allow it in Settings ▸ Geocoding')
        if not allowed:
            self.place.setPlaceholderText('place-name search is off (Settings ▸ Geocoding): work from files')
        lv.addLayout(geo)
        self.results = QListWidget()
        self.results.setMaximumHeight(80)
        self.results.itemActivated.connect(self.use_result)
        self.results.setVisible(False)
        lv.addWidget(self.results)
        self.picker = AreaPicker()
        self.picker.picked.connect(self.set_bbox)
        lv.addWidget(self.picker, 1)
        lv.addWidget(label('Drag a rectangle over the roads to choose the area, or set its centre and size.',
                           muted=True, small=True))
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        area = Section('Area')
        f = Form()
        self.lat, self.lon, self.size = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for sb, lo, hi, dec in ((self.lat, -85, 85, 6), (self.lon, -180, 180, 6), (self.size, 100, 20000, 0)):
            sb.setRange(lo, hi)
            sb.setDecimals(dec)
        self.size.setValue(1000)
        self.size.setSuffix(' m')
        f.row('Centre latitude', self.lat)
        f.row('Centre longitude', self.lon)
        f.row('Side', self.size, '1000 m: 1 km²')
        apply_btn = button('Use centre and side', self.bbox_from_centre)
        f.row('', apply_btn)
        self.bbox_lbl = label('', mono=True, small=True, wrap=True)
        f.row('Box', self.bbox_lbl)
        area.add(f)
        rv.addWidget(area)
        conv = Section('Converter (carla.Osm2Odr)')
        g = Form()
        self.lht = QCheckBox('Left-hand traffic (India, UK, Japan, …)')
        self.lht.setChecked(True)
        g.row('Traffic', self.lht, 'converted after the converter, which only writes right-hand roads')
        self.lane_w = QDoubleSpinBox()
        self.lane_w.setRange(2.5, 5.0)
        self.lane_w.setValue(3.35)
        self.lane_w.setSuffix(' m')
        g.row('Default lane width', self.lane_w)
        self.lights = QCheckBox('Traffic lights where OSM has them')
        self.lights.setChecked(True)
        g.row('', self.lights)
        self.all_lights = QCheckBox('Traffic lights at every junction')
        g.row('', self.all_lights)
        types = QGridLayout()
        self.types = {}
        for i, t in enumerate(O.CARLA_HIGHWAYS):
            cb = QCheckBox(t)
            cb.setChecked(True)
            self.types[t] = cb
            types.addWidget(cb, i // 3, i % 3)
        g.row('Road types', types)
        conv.add(g)
        rv.addWidget(conv)
        mesh = Section('World mesh (generate_opendrive_world)')
        m = Form()
        self.mesh = {}
        for k, text, lo, hi in (('vertex_distance', 'Vertex distance', 0.5, 10.0), ('wall_height', 'Wall height',
                                                                                    0.0, 5.0),
                                ('additional_width', 'Extra width', 0.0, 3.0)):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(MESH_DEFAULTS[k])
            sb.setSuffix(' m')
            self.mesh[k] = sb
            m.row(text, sb)
        self.smooth = QCheckBox('Smooth junctions')
        self.smooth.setChecked(True)
        m.row('', self.smooth)
        mesh.add(m)
        rv.addWidget(mesh)
        b = QHBoxLayout()
        self.go = button('Convert', self.convert, role='primary')
        b.addWidget(self.go)
        b.addStretch(1)
        rv.addLayout(b)
        rv.addStretch(1)
        split.addWidget(right)

        res = QWidget()
        resv = QVBoxLayout(res)
        self.graph = RoadGraphView()
        resv.addWidget(self.graph, 1)
        self.report = KeyValues(('State',))
        resv.addWidget(self.report)
        split.addWidget(res)
        split.setSizes([420, 360, 420])
        bb = QDialogButtonBox()
        self.add_btn = bb.addButton('Add as a world', QDialogButtonBox.ButtonRole.AcceptRole)
        self.drive_btn = bb.addButton('Add and drive it', QDialogButtonBox.ButtonRole.ApplyRole)
        self.drive_btn.clicked.connect(lambda: self.add_world(drive=True))
        bb.addButton(QDialogButtonBox.StandardButton.Close)
        for b in (self.add_btn, self.drive_btn):
            set_enabled(b, False, 'Convert an area first')
        bb.accepted.connect(self.add_world)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.bbox = None
        win.bridge.changed.connect(self.on_job)
        win.bridge.finished.connect(self.on_job)
        if source:
            self.load()

    # ------------------------------------------------------------------ the source and the area
    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, 'OpenStreetMap file', os.path.expanduser('~'),
                                           'OpenStreetMap (*.osm *.osm.pbf *.osm.gz *.osm.bz2)')
        if p:
            self.source.setText(p)
            self.load()

    def load(self):
        path = self.source.text().strip()
        if not os.path.isfile(path):
            QMessageBox.warning(self, 'Import', f'{path} is not a file.')
            return
        size = os.path.getsize(path)
        self.report.set('State', f'reading {tilde(path)} ({size / 1e6:.1f} MB)…')
        if size > 200e6:                           # a large regional extract: no full preview, work from centre and size
            self.report.set('State', f'{tilde(path)} is {size / 1e6:.0f} MB: too large to preview whole; set the '
                                     'centre and side, and the import reads only that area')
            return
        tasks.run(self, lambda: O.read_osm(path), self.loaded,
                  lambda e: self.report.set('State', f'⚠ cannot read it: {e}'))

    def loaded(self, data):
        self.data = data
        self.picker.set_data(data)
        s = O.summary(data)
        lat, lon = s['centre']
        self.lat.setValue(lat)
        self.lon.setValue(lon)
        self.report.set('State', f'{s["highways"]} roads, {s["highway_length_m"] / 1000:.1f} km of highway in the '
                                 'file; choose the area')

    def bbox_from_centre(self):
        self.set_bbox(O.bbox_around(self.lat.value(), self.lon.value(), self.size.value()))

    def set_bbox(self, bbox):
        self.bbox = tuple(bbox)
        self.picker.set_bbox(self.bbox)
        s, w, n, e = self.bbox
        self.bbox_lbl.setText(f'S {s:.5f}  W {w:.5f}\nN {n:.5f}  E {e:.5f}')

    def search(self):
        q = self.place.text().strip()
        if not q:
            return
        allowed = bool(self.win.settings.get('geocoding_allowed'))
        tasks.run(self, lambda: O.geocode(q, allowed), self.show_results,
                  lambda e: QMessageBox.warning(self, 'Search', str(e)))

    def show_results(self, res):
        self.results.clear()
        self.results.setVisible(True)
        self._geo = res
        for r in res:
            self.results.addItem(r[0])

    def use_result(self, it):
        name, lat, lon, _bbox = self._geo[self.results.row(it)]
        self.lat.setValue(lat)
        self.lon.setValue(lon)
        self.bbox_from_centre()

    # ------------------------------------------------------------------ converting
    def convert(self):
        src = self.source.text().strip()
        if not os.path.isfile(src):
            QMessageBox.warning(self, 'Import', 'Choose an OpenStreetMap file first (offline-first: a place-name '
                                                'search only finds the area; the roads come from a file).')
            return
        if self.bbox is None:
            self.bbox_from_centre()
        ws = self.win.workspace()
        out = world_dir(ws, self.world_id)
        settings = {'lane_width': self.lane_w.value(), 'traffic_lights': self.lights.isChecked(),
                    'all_junction_lights': self.all_lights.isChecked(),
                    'way_types': [t for t, cb in self.types.items() if cb.isChecked()]}
        self.cfg = import_config(src, out, self.bbox, self.lht.isChecked(), settings)
        try:
            self.job = self.win.service.submit_osm_import(self.cfg)
        except RuntimeError as e:
            QMessageBox.warning(self, 'Import', str(e))
            return
        self.go.setEnabled(False)
        for b in (self.add_btn, self.drive_btn):
            set_enabled(b, False, 'Converting…')

    def on_job(self, job):
        if job is not self.job:
            return
        self.report.set('State', job.summary())
        if job.state not in FINAL:
            return
        self.go.setEnabled(True)
        if job.state != 'done':
            tail = '; '.join(t for lvl, t in job.log[-5:] if lvl in ('stderr', 'error', 'warning'))
            self.report.set('State', f'⚠ {job.state}: {tail}')
            return
        out = self.cfg['out']
        with open(os.path.join(out, 'graph.json'), encoding='utf-8') as fh:
            self.graph.set_graph(json.load(fh))
        with open(os.path.join(out, 'import.json'), encoding='utf-8') as fh:
            self.rep = json.load(fh)
        show_report(self.report, self.rep)
        for b in (self.add_btn, self.drive_btn):
            set_enabled(b, True)

    def add_world(self, drive=False):
        mesh = {k: sb.value() for k, sb in self.mesh.items()}
        mesh['smooth_junctions'] = self.smooth.isChecked()
        w = imported_world(self.win.project.doc, self.world_id, self.cfg, self.rep, mesh)
        name = w['name']
        self.win.edit(Edit('insert', ['worlds'], w, text=f'Import {name}'))
        self.accept()
        if drive:                                 # straight to Live with the new world chosen
            live = self.win.go('live')
            if live is not None:
                live.choose_world(w['id'])


def show_report(kv: KeyValues, rep: dict):
    t = rep.get('graph_totals') or {}
    lc = rep.get('lane_check') or {}
    kv.set('State', 'converted')
    kv.set('Road network', f'{t.get("roads", "?")} roads, {t.get("road_km", 0):.2f} km (one road per travel '
                           f'direction), {t.get("junctions", 0)} junctions, {t.get("signals", 0)} signals')
    kv.set('Lanes per direction', ', '.join(f'{k} lane{"s" if int(k) > 1 else ""}: {v} roads' for k, v in sorted(
        (t.get('driving_lanes_per_direction') or {}).items(), key=lambda kv: int(kv[0]))))
    kv.set('Traffic', {'LHT': 'left-hand (rule="LHT" on every road)', 'RHT': 'right-hand'}.get(
        rep.get('handedness'), rep.get('handedness')))
    if lc.get('comparable'):
        kv.set('Lanes against OSM', f'{lc["matched"]} of {lc["comparable"]} comparable ways agree '
                                    f'({100 * lc["agreement"]:.0f} %) — {lc.get("method", "")}')
    else:
        kv.set('Lanes against OSM', 'no comparable way in the sample (the area has no named, lane-tagged roads)')
    nr = rep.get('not_represented') or {}
    kv.set('Not in the world', f'{nr.get("buildings", 0)} buildings in the extract — {nr.get("text", "")}')
    tm = rep.get('timings') or {}
    kv.set('Timings', f'read {tm.get("read_s")} s, convert {tm.get("convert_s")} s, total {tm.get("total_s")} s')
    for n in (rep.get('notes') or [])[:6]:
        kv.set('Note', n)
