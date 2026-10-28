"""Worlds: the project's worlds — a CARLA town with its weather and time of day, and its traffic profile.

Every change is a project edit (undoable); a slider drag is one undo step."""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget)

from ...core import weather as W
from ...core.project import Edit, new_id
from .. import icons
from ..widgets import Form, Page, Section, SliderField, button, label

# the towns of CARLA 0.9.16's base package (the "_Opt" variants have layers that can be unloaded; Town10HD_Opt is the
# default map). Town06, 07 and 11–15 come with CARLA's additional-maps package: once a probe has listed the server's
# maps, those are offered instead.
TOWNS = ('Town01', 'Town01_Opt', 'Town02', 'Town02_Opt', 'Town03', 'Town03_Opt', 'Town04', 'Town04_Opt', 'Town05',
         'Town05_Opt', 'Town10HD', 'Town10HD_Opt')
LIMITS = ('Limits of CARLA\'s weather: lidar and radar ignore it (no loss of range in rain or fog); rain on a camera '
          'is water on the lens and wet surfaces; fog visibility ≈ 7000 m / fog density (measured on 0.9.16, a 5 % '
          'contrast threshold — a human observer\'s; for a detector it is an upper bound).')


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
        k = self.traffic.findData(w.get('traffic_profile'))
        self.traffic.setCurrentIndex(max(k, 0))
        wt = w.get('weather') or {}
        self.preset.setCurrentText(wt.get('preset', W.DEFAULT_PRESET))
        vals = W.resolve(wt)
        for field, s in self.sliders.items():
            s.setValue(vals[field])
        self.summary.setText(W.describe(wt))
        self.traffic_editor.show_profile(w.get('traffic_profile'))
        for wd in (self.name, self.town, self.preset, self.traffic):
            wd.blockSignals(False)

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
            notes.append('CARLA 0.9.16 has no autorickshaw: the Microlino (a small three-door car) stands in and is '
                         'labelled as a substitute. Lane-splitting by two-wheelers is not reproduced by the traffic '
                         'manager.')
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
