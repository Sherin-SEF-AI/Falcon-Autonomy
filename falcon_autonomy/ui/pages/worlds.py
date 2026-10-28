"""Worlds: the project's worlds — a CARLA town with its weather and time of day, and its traffic profile.

Every change is a project edit (undoable); a slider drag is one undo step."""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget)

from ...core import weather as W
from ...core.project import Edit, new_id
from .. import icons
from ..widgets import Form, KeyValues, Page, Section, SliderField, button, label

# the towns of CARLA 0.9.16's base package (the "_Opt" variants have layers that can be unloaded; Town10HD_Opt is the
# default map). Town06, 07 and 11–15 come with CARLA's additional-maps package: once a probe has listed the server's
# maps, those are offered instead.
TOWNS = ('Town01', 'Town01_Opt', 'Town02', 'Town02_Opt', 'Town03', 'Town03_Opt', 'Town04', 'Town04_Opt', 'Town05',
         'Town05_Opt', 'Town10HD', 'Town10HD_Opt')
LIMITS = ('Limits of CARLA\'s weather: its lidar and radar ignore it (no loss of range in rain or fog) — switch on '
          'Falcon\'s sensor weather physics below for modelled losses and clutter; rain on a camera '
          'is water on the lens and wet surfaces; fog visibility ≈ 7000 m / fog density (measured on 0.9.16, a 5 % '
          'contrast threshold — a human observer\'s; for a detector it is an upper bound). Puddles (waterlogging) '
          'cover the whole world, not chosen places, and are drawn on CARLA\'s towns but not on an imported '
          '(OpenDRIVE) world (measured: docs/indian-assets.md).')
FEATURES_ABOUT = ('Speed breakers written into the road\'s elevation, so the vehicle physics drives over them: from '
                  'OpenStreetMap\'s traffic_calming tags, and optionally every N metres (IRC:99-1988 spaces repeated '
                  'humps 100–120 m apart). They span the full road width (an elevation profile). Potholes are not '
                  'expressible as such (a full-width dip is the nearest). docs/indian-assets.md.')


class WorldsPage(Page):
    key, title, icon = 'worlds', 'Worlds', 'ground'
    about = 'CARLA towns with their weather and time of day, and the traffic each world drives with.'

    def __init__(self, win):
        super().__init__(win)
        v = self.page_layout()
        row = QHBoxLayout()
        row.setSpacing(18)
        v.addLayout(row, 1)
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.setFixedWidth(240)
        self.list.currentRowChanged.connect(lambda _: self.show_world())
        left.addWidget(self.list, 1)
        bl = QHBoxLayout()
        bl.addWidget(button('Add town', self.add_world))
        bl.addWidget(button('Import a place…', self.import_place))
        self.remove_btn = button('Remove', self.remove_world)
        bl.addWidget(self.remove_btn)
        left.addLayout(bl)
        row.addLayout(left)

        self.editor = QWidget()
        ev = QVBoxLayout(self.editor)
        ev.setContentsMargins(0, 0, 0, 0)
        sec = Section('World')
        f = Form()
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self.set_field(['name'], self.name.text().strip() or 'World'))
        f.row('Name', self.name)
        self.town = QComboBox()
        self.town.addItems(TOWNS)
        self.town.setEditable(True)
        self.town.activated.connect(lambda _: self.set_field(['town'], self.town.currentText()))
        f.row('Town', self.town, 'The base package\'s towns until Tools ▸ Probe lists the server\'s own')
        self.traffic = QComboBox()
        self.traffic.activated.connect(lambda _: self.set_field(['traffic_profile'],
                                                                self.traffic.currentData()))
        f.row('Traffic profile', self.traffic)
        sec.add(f)
        ev.addWidget(sec)

        self.place = Section('Imported place', 'An OpenDRIVE world: roads, lanes, sidewalks and junctions only — no '
                                               'buildings, vegetation or textures (docs/worlds.md: the full-fidelity '
                                               'path).')
        from ..osmimport import RoadGraphView
        self.graph = RoadGraphView()
        self.graph.setMinimumHeight(280)
        self.place.add(self.graph)
        self.place_kv = KeyValues()
        self.place.add(self.place_kv)
        pb = QHBoxLayout()
        self.verify_btn = button('Verify traffic handedness', self.verify, tip='Load the world, run traffic and '
                                                                               'measure which side it keeps')
        pb.addWidget(self.verify_btn)
        pb.addStretch(1)
        self.place.add(pb)
        self.verify_kv = KeyValues()
        self.place.add(self.verify_kv)
        ev.addWidget(self.place)
        self.features = RoadFeatures(self)
        ev.addWidget(self.features)
        self.verify_job = None
        win.bridge.finished.connect(self.on_verify)

        ws = Section('Weather and time of day', LIMITS)
        wf = Form()
        self.preset = QComboBox()
        self.preset.addItems(W.PRESETS)
        self.preset.activated.connect(lambda _: self.set_preset(self.preset.currentText()))
        wf.row('Preset', self.preset, 'Sliders override the preset field by field')
        self.sliders = {}
        for field, (lo, hi, unit, text) in W.SLIDERS.items():
            s = SliderField(lo, hi, unit)
            s.committed.connect(lambda val, fld=field: self.set_weather(fld, val, merge=False))
            s.valueChanged.connect(lambda val, fld=field: self.set_weather(fld, val, merge=True))
            self.sliders[field] = s
            wf.row(text, s)
        self.summary = label('', muted=True, wrap=True)
        wf.row('Reads as', self.summary)
        from PyQt6.QtWidgets import QCheckBox
        self.physics = {k: QCheckBox(t) for k, t in (('enabled', 'On'), ('lidar', 'Lidar'), ('radar', 'Radar'),
                                                     ('camera', 'Camera'))}
        pw = QWidget()
        pl = QHBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        for k, cb in self.physics.items():
            cb.toggled.connect(lambda on, key=k: self.set_physics(key, on))
            pl.addWidget(cb)
        pl.addStretch(1)
        wf.row('Sensor weather physics', pw, 'Modelled losses and clutter CARLA leaves out: lidar and radar in rain '
                                             'and fog, droplets, glare (docs/sensor-physics.md). Recorded in every '
                                             'run; never applied unless on.')
        ws.add(wf)
        ev.addWidget(ws)
        self.traffic_editor = TrafficEditor(self)
        ev.addWidget(self.traffic_editor)
        ev.addStretch(1)
        row.addWidget(self.editor, 1)
        self.refresh()

    # ------------------------------------------------------------------ view
    def worlds(self):
        return self.win.project.doc['worlds']

    def current(self):
        i = self.list.currentRow()
        ws = self.worlds()
        return (i, ws[i]) if 0 <= i < len(ws) else (None, None)

    def refresh(self):
        i = max(self.list.currentRow(), 0)
        self.list.blockSignals(True)
        self.list.clear()
        for w in self.worlds():
            it = QListWidgetItem(icons.icon('ground', 16), f'{w["name"]}\n{w.get("town") or w["kind"]}')
            self.list.addItem(it)
        self.list.setCurrentRow(min(i, self.list.count() - 1))
        self.list.blockSignals(False)
        probe = next((j for j in reversed(self.win.queue.jobs) if j.spec.kind == 'probe' and j.result), None)
        maps = [m for m in ((probe.result.get('summary') or {}).get('maps') or []) if m.startswith('Town')] \
            if probe else []
        self.town.blockSignals(True)
        self.town.clear()
        self.town.addItems(maps or TOWNS)
        self.town.blockSignals(False)
        self.traffic.clear()
        for tp in self.win.project.doc['traffic_profiles']:
            self.traffic.addItem(tp['name'], tp['id'])
        self.show_world()

    def show_world(self):
        _, w = self.current()
        self.editor.setEnabled(w is not None)
        self.remove_btn.setEnabled(w is not None and len(self.worlds()) > 1)
        if w is None:
            return
        for wd in (self.name, self.town, self.preset, self.traffic):
            wd.blockSignals(True)
        self.name.setText(w['name'])
        self.town.setCurrentText(w.get('town', ''))
        self.town.setEnabled(w['kind'] == 'town')
        self.show_place(w)
        k = self.traffic.findData(w.get('traffic_profile'))
        self.traffic.setCurrentIndex(max(k, 0))
        wt = w.get('weather') or {}
        self.preset.setCurrentText(wt.get('preset', W.DEFAULT_PRESET))
        vals = W.resolve(wt)
        for field, s in self.sliders.items():
            s.setValue(vals[field])
        self.summary.setText(W.describe(wt))
        from ...core.sensorphysics import DEFAULTS as PHYS
        ph = dict(PHYS, **(w.get('sensor_physics') or {}))
        for k, cb in self.physics.items():
            cb.blockSignals(True)
            cb.setChecked(bool(ph.get(k)))
            cb.setEnabled(k == 'enabled' or bool(ph.get('enabled')))
            cb.blockSignals(False)
        self.traffic_editor.show_profile(w.get('traffic_profile'))
        for wd in (self.name, self.town, self.preset, self.traffic):
            wd.blockSignals(False)

    def show_place(self, w):
        import json
        import os
        self.place.setVisible(w['kind'] != 'town')
        self.features.setVisible(w['kind'] != 'town')
        if w['kind'] == 'town':
            return
        self.features.show_world(w)
        d = os.path.join(self.win.workspace(), os.path.dirname(w.get('xodr', '')))
        self.place_kv.clear()
        self.verify_kv.clear()
        try:
            with open(os.path.join(d, 'graph.json'), encoding='utf-8') as fh:
                self.graph.set_graph(json.load(fh))
            with open(os.path.join(d, 'import.json'), encoding='utf-8') as fh:
                rep = json.load(fh)
        except (OSError, ValueError):
            self.place_kv.set('State', f'the import\'s files are missing from the workspace ({d})')
            return
        from ..osmimport import show_report
        show_report(self.place_kv, rep)
        self.place_kv.set('State', f'imported from {os.path.basename(rep["source"])}')
        try:
            with open(os.path.join(d, 'verify', 'run.json'), encoding='utf-8') as fh:
                self.show_verify(json.load(fh))
        except (OSError, ValueError):
            self.verify_kv.set('Handedness', 'not verified yet')

    def show_verify(self, run):
        h = run.get('handedness')
        if not h:
            self.verify_kv.set('Handedness', f'{run.get("status")}: {run.get("error") or ""}')
            return
        self.verify_kv.set('Handedness', f'{h["verdict"]}: {100 * h["keep_left_share"]:.1f} % of {h["samples"]} '
                                         f'samples of {h["vehicles"]} vehicles keep left, '
                                         f'{100 * h["with_lane_share"]:.1f} % move with their lane (measured)')
        self.verify_kv.set('Method', h.get('method', ''))

    def verify(self):
        import os
        from PyQt6.QtWidgets import QMessageBox
        from ...core.drive import drive_config
        from ...core.project import ProjectError
        i, w = self.current()
        if w is None:
            return
        d = os.path.join(self.win.workspace(), os.path.dirname(w['xodr']), 'verify')
        try:
            cfg = drive_config(self.win.project, w['id'], self.win.project.doc['rigs'][0]['id'], d,
                               self.win.service.port, self.win.service.tm_port, seed=0,
                               traffic_id='tp-default', workspace=self.win.workspace())
        except ProjectError as e:
            QMessageBox.warning(self, 'Verify', str(e))
            return
        cfg.update(seconds=30.0, vehicles=30)
        self.verify_job = self.win.service.submit_handedness(cfg)
        self.verify_kv.set('Handedness', 'queued…')

    def on_verify(self, job):
        if job is self.verify_job:
            import json
            import os
            try:
                with open(os.path.join(job.spec.run_dir, 'run.json'), encoding='utf-8') as fh:
                    self.show_verify(json.load(fh))
            except (OSError, ValueError):
                self.verify_kv.set('Handedness', job.summary())

    def import_place(self):
        from ..osmimport import ImportDialog
        ImportDialog(self.win).exec()

    # ------------------------------------------------------------------ edits
    def set_field(self, path, value, merge_key=''):
        i, w = self.current()
        if w is None or w.get(path[-1]) == value:
            return
        self.win.edit(Edit('set', ['worlds', i] + path, value, text=f'Set {path[-1]} of {w["name"]}'), merge_key)

    def set_preset(self, name):
        i, w = self.current()
        if w is None:
            return
        self.win.edit(Edit('set', ['worlds', i, 'weather'], {'preset': name},
                           text=f'Weather of {w["name"]}: {name}'))

    def set_physics(self, key, on):
        i, w = self.current()
        if w is None:
            return
        from ...core.sensorphysics import DEFAULTS as PHYS
        block = {k: PHYS[k] for k in ('enabled', 'lidar', 'radar', 'camera')}
        block.update(w.get('sensor_physics') or {})
        block[key] = bool(on)
        self.win.edit(Edit('set', ['worlds', i, 'sensor_physics'], block,
                           text=f'Sensor weather physics of {w["name"]}: {key} {"on" if on else "off"}'))
        self.show_world()

    def set_weather(self, field, value, merge):
        i, w = self.current()
        if w is None:
            return
        wt = dict(w.get('weather') or {'preset': W.DEFAULT_PRESET})
        if abs(W.resolve(wt)[field] - value) < 1e-9:
            return
        wt[field] = round(float(value), 3)
        # one undo step per drag: the drag's moves merge, the release (committed) merges into them too
        self.win.edit(Edit('set', ['worlds', i, 'weather'], wt, text=f'{W.SLIDERS[field][3]} of {w["name"]}'),
                      merge_key=f'weather/{w["id"]}/{field}')

    def add_world(self):
        doc = self.win.project.doc
        w = {'id': new_id('world'), 'name': f'Town 10 ({len(doc["worlds"]) + 1})', 'kind': 'town',
             'town': 'Town10HD_Opt', 'weather': {'preset': W.DEFAULT_PRESET},
             'traffic_profile': doc['traffic_profiles'][0]['id'] if doc['traffic_profiles'] else 'tp-default'}
        self.win.edit(Edit('insert', ['worlds'], w, text='Add a world'))
        self.list.setCurrentRow(self.list.count() - 1)

    def remove_world(self):
        i, w = self.current()
        if w is None or len(self.worlds()) <= 1:
            return
        self.win.edit(Edit('remove', ['worlds'], index=i, text=f'Remove {w["name"]}'))


class RoadFeatures(Section):
    """An imported world's speed breakers (core.roadfeatures): what is placed, and the choices to place them again.
    Applying rewrites the world's OpenDRIVE file (world.base.xodr keeps it without features) and sets its mesh's vertex
    distance (a project edit); undo restores the settings, not the file — apply again to change it."""

    def __init__(self, page):
        super().__init__('Road features', FEATURES_ABOUT)
        self.page = page
        from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox
        f = Form()
        self.use_osm = QCheckBox('From OpenStreetMap (traffic_calming = bump, hump, table)')
        self.use_osm.setChecked(True)
        f.row('Tags', self.use_osm)
        self.bump = QComboBox()
        self.bump.addItem('a short bump (0.30 m long, 0.075 m high)', 'osm_bump')
        self.bump.addItem('an IRC speed breaker (3.7 m, 0.10 m)', 'irc_hump')
        f.row('"bump" becomes', self.bump, 'OSM\'s bump is ≤ 30 cm long; many Indian breakers tagged so are humps')
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.0, 1000.0)
        self.spacing.setDecimals(0)
        self.spacing.setSuffix(' m')
        self.spacing.setSpecialValueText('none')
        f.row('Also every', self.spacing, 'an IRC hump along every road longer than 60 m (0: none; IRC: 100–120 m)')
        self.add(f)
        row = QHBoxLayout()
        self.apply_btn = button('Apply road features', self.apply, tip='Rewrite the world\'s OpenDRIVE file')
        self.remove_btn = button('Remove', self.remove)
        row.addWidget(self.apply_btn)
        row.addWidget(self.remove_btn)
        row.addStretch(1)
        self.add(row)
        self.kv = KeyValues()
        self.add(self.kv)

    def world_dir(self, w):
        import os
        return os.path.join(self.page.win.workspace(), os.path.dirname(w.get('xodr', '')))

    def show_world(self, w):
        import json
        import os
        self.kv.clear()
        opts = (w.get('osm') or {}).get('road_features') or {}
        self.use_osm.setChecked(bool(opts.get('osm', True)))
        k = self.bump.findData(opts.get('osm_bump', 'osm_bump'))
        self.bump.setCurrentIndex(max(k, 0))
        self.spacing.setValue(float(opts.get('spacing_m', 0.0) or 0.0))
        p = os.path.join(self.world_dir(w), 'features.json')
        try:
            with open(p, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            self.kv.set('Placed', 'none (no features.json)')
            return
        counts = ', '.join(f'{v} {k.replace("_", " ")}' for k, v in sorted(doc.get('counts', {}).items())) or 'none'
        self.kv.set('Placed', counts)
        if doc.get('skipped'):
            self.kv.set('Skipped', f'{len(doc["skipped"])} (in junctions, near road ends or overlapping: '
                                   'features.json says which)')
        if doc.get('not_represented'):
            self.kv.set('Not represented', ', '.join(f'{v} {k}' for k, v in sorted(doc['not_represented'].items())))
        mesh = (w.get('osm') or {}).get('mesh') or {}
        self.kv.set('Mesh', f'vertex distance {mesh.get("vertex_distance", 2.0)} m '
                            f'({doc.get("mesh_note", "")})')

    def apply(self):
        from PyQt6.QtWidgets import QMessageBox
        from ...core import roadfeatures as RF
        i, w = self.page.current()
        if w is None or w['kind'] == 'town':
            return
        opts = {'osm': self.use_osm.isChecked(), 'osm_bump': self.bump.currentData(),
                'spacing_m': float(self.spacing.value()), 'spaced_kind': 'irc_hump'}
        try:
            doc = RF.apply_to_world(self.world_dir(w), opts)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, 'Road features', str(e))
            return
        self.set_world(i, w, opts, bool(doc['placed']))

    def remove(self):
        from ...core import roadfeatures as RF
        i, w = self.page.current()
        if w is None or w['kind'] == 'town':
            return
        RF.remove_from_world(self.world_dir(w))
        self.set_world(i, w, {'osm': False, 'spacing_m': 0.0}, False)

    def set_world(self, i, w, opts, has):
        import copy
        from ...core import roadfeatures as RF
        from ...core.worlds import MESH_DEFAULTS
        osm = copy.deepcopy(w.get('osm') or {})
        mesh = dict(osm.get('mesh') or MESH_DEFAULTS)
        if not has:
            mesh['vertex_distance'] = MESH_DEFAULTS['vertex_distance']
        osm['mesh'] = RF.mesh_for(mesh, has)
        osm['road_features'] = opts
        self.page.win.edit(Edit('set', ['worlds', i, 'osm'], osm, text=f'Road features of {w["name"]}'))
        self.show_world(self.page.worlds()[i])


class TrafficEditor(Section):
    """The world's traffic profile: counts, fleet shares, behaviour. Edits go to the profile (shared by every world
    that uses it)."""
    BEHAVIOUR = (('distance_to_leading_m', 'Gap to the vehicle ahead', 0.0, 20.0, 'm'),
                 ('speed_spread_pct', 'Speed spread (±, per vehicle)', 0.0, 80.0, '%'),
                 ('speed_difference_pct', 'Speed below the limit', -50.0, 80.0, '%'),
                 ('ignore_lights_pct', 'Ignore traffic lights', 0.0, 100.0, '%'),
                 ('ignore_signs_pct', 'Ignore signs', 0.0, 100.0, '%'),
                 ('random_left_lanechange_pct', 'Random lane changes (left)', 0.0, 100.0, '%'),
                 ('random_right_lanechange_pct', 'Random lane changes (right)', 0.0, 100.0, '%'),
                 ('walker_crossing_factor', 'Walkers crossing anywhere', 0.0, 1.0, 'share'))

    def __init__(self, page):
        super().__init__('Traffic profile', 'Shared by every world that uses it. Fleet shares are weights per CARLA '
                                            'base type, normalised at run time.')
        self.page = page
        self.pid = None
        top = QHBoxLayout()
        self.add_india = button('Add the Indian mixed-traffic profile', self.add_preset)
        top.addWidget(self.add_india)
        top.addStretch(1)
        self.add(top)
        self.form_host = QWidget()
        self.form = Form()
        self.form_host.setLayout(self.form)
        self.add(self.form_host)
        self.notes = label('', muted=True, wrap=True, small=True)
        self.add(self.notes)

    def profile(self):
        doc = self.page.win.project.doc
        for i, tp in enumerate(doc['traffic_profiles']):
            if tp['id'] == self.pid:
                return i, tp
        return None, None

    def show_profile(self, pid):
        from ...core.traffic import INDIA_MIXED, fleet_label
        self.pid = pid
        self.add_india.setVisible(self.page.win.project.item('traffic_profiles', INDIA_MIXED['id']) is None)
        while self.form.count():
            it = self.form.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.form.n = 0
        i, tp = self.profile()
        if tp is None:
            return
        from PyQt6.QtWidgets import QDoubleSpinBox, QSpinBox
        for key, text, hi in (('vehicles', 'Vehicles', 500), ('walkers', 'Walkers', 500)):
            sb = QSpinBox()
            sb.setRange(0, hi)
            sb.setValue(int(tp.get(key, 0)))
            sb.editingFinished.connect(lambda k=key, w=sb: self.set(['traffic_profiles', i, k], w.value()))
            self.form.row(text, sb)
        hand = QComboBox()
        hand.addItems(['map', 'left', 'right'])
        hand.setCurrentText(tp.get('handedness', 'map'))
        hand.activated.connect(lambda _: self.set(['traffic_profiles', i, 'handedness'], hand.currentText()))
        self.form.row('Handedness', hand, 'left or right is checked against the map before every run')
        for k, v in sorted(tp.get('fleet', {}).items()):
            sb = QDoubleSpinBox()
            sb.setRange(0, 1)
            sb.setDecimals(2)
            sb.setSingleStep(0.01)
            sb.setValue(float(v))
            sb.editingFinished.connect(lambda k=k, w=sb: self.set(['traffic_profiles', i, 'fleet', k], w.value()))
            self.form.row(f'Fleet: {fleet_label(k, tp)}', sb)
        beh = tp.get('behaviour', {})
        for k, text, lo, hi, unit in self.BEHAVIOUR:
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(2)
            sb.setSuffix(f' {unit}')
            sb.setValue(float(beh.get(k, 0.0)))
            sb.editingFinished.connect(lambda k=k, w=sb: self.set(['traffic_profiles', i, 'behaviour', k],
                                                                   w.value()))
            self.form.row(text, sb)
        notes = []
        if tp.get('fleet_source'):
            notes.append(f'Fleet shares: {tp["fleet_source"]}.')
        if tp.get('fleet_labels'):
            notes.append('CARLA 0.9.16 has no autorickshaw: the Microlino, the nearest vehicle in measured size, '
                         'stands in with a three-wheeler\'s top speed and acceleration, labelled "autorickshaw '
                         '(stand-in)" (docs/indian-assets.md). Lane-splitting by two-wheelers is not reproduced by the '
                         'traffic manager.')
        self.notes.setText(' '.join(notes))

    def set(self, path, value):
        cur = self.page.win.project.doc
        for k in path[:-1]:
            cur = cur[k]
        if cur.get(path[-1]) == value:
            return
        self.page.win.edit(Edit('set', path, value, text=f'Traffic: {path[-1]}'))

    def add_preset(self):
        from ...core.traffic import preset
        tp = preset('tp-india-mixed')
        self.page.win.edit(Edit('insert', ['traffic_profiles'], tp, text='Add the Indian mixed-traffic profile'))
