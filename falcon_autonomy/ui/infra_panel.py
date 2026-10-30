"""Worlds ▸ Roadside units: the world's V2X roadside units (core.v2x) — placed at a junction of an imported world or at a
position, their pole, heading and height, their sensors (rig definitions), the CPM they send — on a top-down map of
the world's roads with each sensor's field of view. Every change is a project edit (undoable). docs/v2x.md."""
from __future__ import annotations

import copy
import json
import math
import os

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
                             QMessageBox, QSpinBox, QVBoxLayout, QWidget)

from ..core import v2x as V
from ..core.project import Edit
from .osmimport import RoadGraphView
from .theme import SEQ, THEME
from . import tasks
from .widgets import Form, KeyValues, Section, button, label, set_enabled

ABOUT = ('Roadside units stand in the world, not on the ego: smart-pole cameras and intersection lidars (rig sensor '
         'definitions) recorded with the same labels as the ego\'s, and a Collective Perception Message (the shape '
         'of ETSI TS 103 324, UPER-encoded) per CPM period listing the road users their sensors actually saw — '
         'occluded ones are left out — with each sensor\'s field of view and the free space it saw. In every drive '
         'of this world; published over ROS 2 by the stack bridge. docs/v2x.md.')


class RsuMap(RoadGraphView):
    """The world's roads (graph.json, OpenDRIVE frame: x east, y north) with the RSUs and their sensors' fields of view;
    a click picks the nearest RSU."""
    picked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.rsus, self.sel = [], -1

    def set_rsus(self, rsus: list, sel: int = -1):
        self.rsus, self.sel = rsus, sel
        if self.bounds is None and rsus:
            xs, ys = [r['x'] for r in rsus], [-r['y'] for r in rsus]
            self.bounds = (min(xs) - 80, min(ys) - 80, max(xs) + 80, max(ys) + 80)
        self.update()

    def paint_extra(self, p):
        super().paint_extra(p)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        k = self.fit()[0]
        for i, r in enumerate(self.rsus):
            c = self.to_px(r['x'], -r['y'])
            for s in r.get('sensors') or []:
                col = QColor(SEQ[5] if s['type'] == 'lidar' else SEQ[8])
                col.setAlpha(60 if i != self.sel else 110)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col)
                rng = float(s.get('range', 100.0)) if s['type'] == 'lidar' else \
                    float(V.cpm_settings(r)['camera_range_m'])
                rad = min(rng, 60.0) * k                 # drawn to 60 m at most (the label says the real range)
                fov = float(s.get('horizontal_fov', 360.0)) if s['type'] == 'lidar' else \
                    float(s.get('fov', (s.get('lens') or {}).get('fov', 90.0)))
                hdg = float(r.get('yaw', 0.0)) + float((s.get('pose') or {}).get('yaw', 0.0))
                path = QPainterPath(c)
                if fov >= 359.9:
                    path.addEllipse(c, rad, rad)
                else:
                    # screen y grows down: a CARLA yaw (towards +y, South) is clockwise on screen, as Qt's angles
                    # are counter-clockwise, the start is −(hdg + fov / 2)
                    path.arcTo(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad, -(hdg + fov / 2.0), fov)
                    path.closeSubpath()
                p.drawPath(path)
            p.setPen(QPen(QColor(THEME.tok['ink']), 2 if i == self.sel else 1))
            p.setBrush(QColor(SEQ[9] if i == self.sel else SEQ[6]))
            p.drawEllipse(c, 5, 5)
            a = math.radians(float(r.get('yaw', 0.0)))
            p.drawLine(c, c + QPointF(12 * math.cos(a), 12 * math.sin(a)))
            p.setPen(QColor(THEME.tok['ink']))
            p.drawText(c + QPointF(8, -8), r.get('name', r['id'])[:32])

    def mousePressEvent(self, ev):
        if not self.rsus:
            return
        x, y = self.to_m(ev.position())
        d = [math.hypot(r['x'] - x, -r['y'] - y) for r in self.rsus]
        k = min(range(len(d)), key=d.__getitem__)
        if d[k] * self.fit()[0] <= 20:
            self.picked.emit(k)


class InfraSection(Section):
    def __init__(self, page):
        super().__init__('Roadside units (V2X)', ABOUT)
        self.page = page
        self._junctions = {}                     # xodr path -> (mtime, junctions)
        row = QHBoxLayout()
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.setFixedWidth(230)
        self.list.setMinimumHeight(120)
        self.list.currentRowChanged.connect(lambda _: self.show_rsu())
        left.addWidget(self.list, 1)
        self.remove_btn = button('Remove', self.remove)
        left.addWidget(self.remove_btn)
        row.addLayout(left)
        self.map = RsuMap()
        self.map.setMinimumHeight(260)
        self.map.picked.connect(lambda k: self.list.setCurrentRow(k))
        row.addWidget(self.map, 1)
        holder = QWidget()
        holder.setLayout(row)
        self.add(holder)
        add = QHBoxLayout()
        self.junction = QComboBox()
        self.junction.setMinimumWidth(280)
        add.addWidget(self.junction, 1)
        self.preset = QComboBox()
        for k, t in V.PRESETS.items():
            self.preset.addItem(k, k)
            self.preset.setItemData(self.preset.count() - 1, t, Qt.ItemDataRole.ToolTipRole)
        self.preset.setCurrentIndex(self.preset.findData('camera-lidar'))
        add.addWidget(self.preset)
        self.add_j_btn = button('Add at the junction', self.add_at_junction, role='primary',
                                tip='A pole at a corner of the junction clear of the driving lanes, facing its centre')
        add.addWidget(self.add_j_btn)
        self.add_p_btn = button('Add at a position', self.add_at_position,
                                tip='A pole at the map\'s centre (or the origin of a town): set its x, y and heading')
        add.addWidget(self.add_p_btn)
        self.add(add)
        f = Form()
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self.set_field('name', self.name.text().strip() or 'Roadside unit'))
        f.row('Name', self.name)
        self.spins = {}
        for key, text, lo, hi, dec, unit, note in (
                ('x', 'x', -1e5, 1e5, 2, ' m', 'CARLA\'s world: the pole\'s foot'),
                ('y', 'y', -1e5, 1e5, 2, ' m', 'CARLA\'s world (+y is South)'),
                ('yaw', 'Heading', -360, 360, 1, '°', 'CARLA\'s yaw: 0 = +x (East), positive towards +y'),
                ('height_m', 'Mast height', 1, 30, 1, ' m', 'the sensors\' poses are from the mast head')):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(dec)
            sb.setSuffix(unit)
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda v, k=key: self.set_field(k, round(float(v), 3)))
            self.spins[key] = sb
            f.row(text, sb, note)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(0.1, 100.0)
        self.rate.setSuffix(' Hz')
        self.rate.setKeyboardTracking(False)
        self.rate.valueChanged.connect(lambda v: self.set_cpm('rate_hz', round(float(v), 3)))
        f.row('CPM rate', self.rate, 'every message lists every perceived object (TS 103 324\'s inclusion rules are '
                                     'not applied)')
        vis = QHBoxLayout()
        self.min_px, self.min_pts = QSpinBox(), QSpinBox()
        for sb, key, unit in ((self.min_px, 'min_pixels', ' px'), (self.min_pts, 'min_points', ' points')):
            sb.setRange(1, 100000)
            sb.setSuffix(unit)
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda v, k=key: self.set_cpm(k, int(v)))
            vis.addWidget(sb)
        vis.addStretch(1)
        f.row('Seen when', vis, 'a camera\'s instance pixels, a lidar sweep\'s points of the object (heuristic '
                                'thresholds; the measured benefit applies the same to the ego)')
        noise = QHBoxLayout()
        self.n_pos, self.n_vel = QDoubleSpinBox(), QDoubleSpinBox()
        for sb, key, unit in ((self.n_pos, 'position_m', ' m'), (self.n_vel, 'velocity_mps', ' m/s')):
            sb.setRange(0.0, 10.0)
            sb.setDecimals(2)
            sb.setSuffix(unit)
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda v, k=key: self.set_noise(k, round(float(v), 3)))
            noise.addWidget(sb)
        noise.addStretch(1)
        f.row('Noise (1 σ)', noise, 'Gaussian, on the reported position and velocity; 0: the ground truth')
        self.form = f
        self.editor = QWidget()
        self.editor.setLayout(f)
        self.add(self.editor)
        self.kv = KeyValues()
        self.add(self.kv)
        self.add(label('Sensors are rig definitions posed from the mast head (x along the heading); a camera always '
                       'records its ground truth. The map draws fields of view to 60 m.', muted=True, small=True,
                       wrap=True))
        brow = QHBoxLayout()
        self.benefit_btn = button('Measure the benefit on recorded drives…', self.measure_benefit,
                                  tip='Choose a drive recorded with roadside units (or a folder of them): how many of '
                                      'the road users near the ego its own sensors saw, against its sensors and the '
                                      'CPMs together (core.v2x.benefit)')
        brow.addWidget(self.benefit_btn)
        brow.addStretch(1)
        self.add(brow)
        self.benefit_kv = KeyValues(('Drives', 'Road users seen within 50 m', 'While near a pole',
                                     'Walkers and two-wheelers', 'Out of occlusion'))
        self.benefit_kv.setVisible(False)
        self.add(self.benefit_kv)

    # ------------------------------------------------------------------ the benefit, measured
    @staticmethod
    def drives_with_cpms(folder: str) -> list:
        """The folder itself when it is a drive with CPMs, else its drives (one level down) that have them."""
        has = lambda d: os.path.isfile(os.path.join(d, 'cpm.jsonl')) and os.path.isfile(os.path.join(d, 'run.json'))
        if has(folder):
            return [folder]
        try:
            subs = sorted(os.path.join(folder, x) for x in os.listdir(folder))
        except OSError:
            return []
        return [d for d in subs if os.path.isdir(d) and has(d)]

    def measure_benefit(self, folder: str = ''):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        if not folder:
            win = self.page.win
            start = win.service.runs_dir() if win.workspace() else ''
            folder = QFileDialog.getExistingDirectory(self, 'A drive with roadside units, or a folder of them', start)
            if not folder:
                return None
        runs = self.drives_with_cpms(folder)
        if not runs:
            QMessageBox.information(self, 'V2X benefit', 'No drive there was recorded with roadside units (no '
                                    'cpm.jsonl): record a drive in a world that has them first.')
            return None
        set_enabled(self.benefit_btn, False, 'Measuring…')
        self.benefit_kv.setVisible(True)
        self.benefit_kv.set('Drives', f'measuring {len(runs)}…')

        def work():
            res = [V.benefit(r) for r in runs]
            return res, V.pooled(res)

        def done(out):
            res, pl = out
            set_enabled(self.benefit_btn, True)
            pc = lambda x: '—' if x is None else f'{100 * x:.1f} %'
            row = lambda k: (f'{pc(pl[k]["ego_share"])} by the ego alone → {pc(pl[k]["ego_or_rsu_share"])} with the '
                             f'CPMs ({pl[k]["road_user_steps"]} road-user steps)') if pl.get(k) else '—'
            t = pl.get('time_to_first_detection') or {}
            self.benefit_kv.set('Drives', f'{len(res)}: ' + ', '.join(r['run'] for r in res))
            self.benefit_kv.set('Road users seen within 50 m', row('all'))
            self.benefit_kv.set('While near a pole', row('near_station'))
            self.benefit_kv.set('Walkers and two-wheelers', row('vru'))
            self.benefit_kv.set('Out of occlusion', f'{t.get("cpm_first", 0)} of {t.get("episodes", 0)} episodes '
                                f'named by a CPM first' + (f', {t["gain_s_mean"]} s earlier on average'
                                                           if t.get('gain_s_mean') is not None else ''))
            self.benefit_kv.setToolTip('Ground truth behind real occlusion: an upper bound (a perfect classifier, no '
                                       'radio). docs/v2x.md')

        def failed(err):
            set_enabled(self.benefit_btn, True)
            self.benefit_kv.set('Drives', f'could not measure: {err}')

        tasks.run(self, work, done, failed)
        return runs

    # ------------------------------------------------------------------ view
    def world(self):
        return self.page.current()[1]

    def doc(self):
        return self.page.win.project.doc

    def rsus(self):
        w = self.world()
        return [] if w is None else V.rsus(self.doc(), w['id'])

    def index_in_doc(self, k: int) -> int:
        """The position in infrastructure.rsus of the world's k-th RSU."""
        target = self.rsus()[k]
        return next(i for i, r in enumerate(V.rsus(self.doc())) if r is target)

    def junctions_of(self, w) -> list:
        if w is None or w.get('kind') == 'town' or not w.get('xodr'):
            return []
        path = os.path.join(self.page.win.workspace(), w['xodr'])
        try:
            mt = os.path.getmtime(path)
        except OSError:
            return []
        hit = self._junctions.get(path)
        if hit is None or hit[0] != mt:
            try:
                with open(path, encoding='utf-8') as fh:
                    text = fh.read()
                hit = (mt, V.junctions(text, os.path.dirname(path), (w.get('osm') or {}).get('centre')), text)
            except (OSError, ValueError) as e:
                hit = (mt, [], '')
                self.kv.set('Junctions', f'⚠ {e}')
            self._junctions[path] = hit
        return hit[1]

    def show_world(self, w):
        i = max(self.list.currentRow(), 0)
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.rsus():
            self.list.addItem(QListWidgetItem(f'{r["name"]}\n{len(r.get("sensors") or [])} sensor'
                                              f'{"s" if len(r.get("sensors") or []) != 1 else ""}, '
                                              f'{float(r.get("height_m", V.HEIGHT_M)):g} m'))
        self.list.setCurrentRow(min(i, self.list.count() - 1))
        self.list.blockSignals(False)
        js = self.junctions_of(w)
        cur = self.junction.currentData()
        self.junction.blockSignals(True)
        self.junction.clear()
        for j in js:
            self.junction.addItem(f'{j["id"]}: {j["name"] or "junction"} — {len(j["approaches"])} approaches'
                                  f'{", signals" if j["signalised"] else ""}', j['id'])
        k = self.junction.findData(cur)
        self.junction.setCurrentIndex(max(k, 0))
        self.junction.blockSignals(False)
        set_enabled(self.add_j_btn, bool(js), 'A town\'s junctions are not read offline: add at a position'
                    if w is not None and w.get('kind') == 'town' else 'No junction found in the world\'s roads')
        self.map.bounds, self.map.lines = None, []
        if w is not None and w.get('xodr'):
            try:
                with open(os.path.join(self.page.win.workspace(), os.path.dirname(w['xodr']), 'graph.json'),
                          encoding='utf-8') as fh:
                    self.map.set_graph(json.load(fh))
            except (OSError, ValueError):
                pass
        self.show_rsu()

    def show_rsu(self):
        rs = self.rsus()
        k = self.list.currentRow()
        self.map.set_rsus(rs, k)
        r = rs[k] if 0 <= k < len(rs) else None
        self.editor.setEnabled(r is not None)
        set_enabled(self.remove_btn, r is not None, 'Choose a roadside unit first')
        self.kv.clear()
        if r is None:
            self.kv.set('Roadside units', 'none in this world yet' if not rs else 'choose one')
            return
        widgets = [self.name, self.rate, self.min_px, self.min_pts, self.n_pos, self.n_vel] + list(self.spins.values())
        for wd in widgets:
            wd.blockSignals(True)
        self.name.setText(r['name'])
        for key, sb in self.spins.items():
            sb.setValue(float(r.get(key, V.HEIGHT_M if key == 'height_m' else 0.0)))
        c = V.cpm_settings(r)
        self.rate.setValue(float(c['rate_hz']))
        self.min_px.setValue(int(c['min_pixels']))
        self.min_pts.setValue(int(c['min_points']))
        self.n_pos.setValue(float(c['noise']['position_m']))
        self.n_vel.setValue(float(c['noise']['velocity_mps']))
        for wd in widgets:
            wd.blockSignals(False)
        from ..core.rig import sensor_cost
        self.kv.set('Sensors', '; '.join(f'{s["name"]} ({s["type"]}: {sensor_cost(V.sensor_with_defaults(s))["note"]})'
                                         for s in r.get('sensors') or []) or 'none')
        self.kv.set('Station', f'ETSI station id {V.station_id(r)}; CPM fields of view and objects in its '
                               'East-North-Up frame from the pole\'s foot')
        if r.get('junction'):
            j = r['junction']
            self.kv.set('Placed', f'junction {j.get("id")} {j.get("name") or ""}: corner {j.get("corner")}, '
                                  f'{j.get("distance_m")} m from its centre, {j.get("clearance_m")} m from the nearest '
                                  'driving lane (heuristic)')
        bad = V.validate_rsu(r, {w['id'] for w in self.doc()['worlds']})
        if bad:
            self.kv.set('Problems', '⚠ ' + '; '.join(bad))

    # ------------------------------------------------------------------ edits
    def _insert(self, r: dict):
        if V.KEY not in self.doc():
            self.page.win.edit(Edit('set', [V.KEY], {'rsus': [r]}, text=f'Add {r["name"]}'))
        else:
            self.page.win.edit(Edit('insert', [V.KEY, 'rsus'], r, text=f'Add {r["name"]}'))
        self.list.setCurrentRow(len(self.rsus()) - 1)
        self.show_rsu()

    def add_at_junction(self):
        w = self.world()
        jid = self.junction.currentData()
        js = {j['id']: j for j in self.junctions_of(w)}
        if w is None or jid not in js:
            return None
        text = self._junctions[os.path.join(self.page.win.workspace(), w['xodr'])][2]
        used = sum(1 for r in self.rsus() if (r.get('junction') or {}).get('id') == jid)
        try:
            r = V.place_at_junction(w['id'], js[jid], text, self.preset.currentData(), corner=used)
        except ValueError as e:
            QMessageBox.warning(self, 'Roadside units', str(e))
            return None
        self._insert(r)
        return r

    def add_at_position(self):
        w = self.world()
        if w is None:
            return None
        b = self.map.bounds
        x, y = ((b[0] + b[2]) / 2.0, -(b[1] + b[3]) / 2.0) if b else (0.0, 0.0)
        r = V.new_rsu(w['id'], x, y, 0.0, self.preset.currentData())
        self._insert(r)
        return r

    def remove(self):
        k = self.list.currentRow()
        rs = self.rsus()
        if not 0 <= k < len(rs):
            return
        self.page.win.edit(Edit('remove', [V.KEY, 'rsus'], index=self.index_in_doc(k), text=f'Remove {rs[k]["name"]}'))

    def set_field(self, key, value):
        k = self.list.currentRow()
        rs = self.rsus()
        if not 0 <= k < len(rs) or rs[k].get(key) == value:
            return
        self.page.win.edit(Edit('set', [V.KEY, 'rsus', self.index_in_doc(k), key], value,
                                text=f'Set {key} of {rs[k]["name"]}'), merge_key=f'rsu-{rs[k]["id"]}-{key}')

    def set_cpm(self, key, value):
        k = self.list.currentRow()
        rs = self.rsus()
        if not 0 <= k < len(rs):
            return
        c = copy.deepcopy(rs[k].get('cpm') or {})
        if c.get(key) == value:
            return
        c[key] = value
        self.set_field('cpm', c)

    def set_noise(self, key, value):
        k = self.list.currentRow()
        rs = self.rsus()
        if not 0 <= k < len(rs):
            return
        c = copy.deepcopy(rs[k].get('cpm') or {})
        c['noise'] = dict(c.get('noise') or {}, **{key: value})
        self.set_field('cpm', c)
