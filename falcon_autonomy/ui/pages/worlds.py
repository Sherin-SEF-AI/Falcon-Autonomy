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
