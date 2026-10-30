"""Rigs: a vehicle and its sensors — poses by numbers or by dragging in the 3D view, each sensor's settings and its
cost against the render budget, import and export (Falcon JSON, ROS URDF and TF, nuScenes calibration). Every change
is an undoable project edit."""
from __future__ import annotations

import copy
import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLineEdit, QListWidget,
                             QListWidgetItem, QMenu, QMessageBox, QProgressBar, QSpinBox, QSplitter, QToolButton,
                             QVBoxLayout, QWidget)

from ...core import budget
from ...core import rig as R
from ...core.project import Edit, new_id
from .. import icons
from ..view3d import make_view
from ..widgets import Form, KeyValues, Page, Section, button, label, scroll, set_enabled

LABELS = ('semantic', 'instance', 'depth', 'boxes2d', 'boxes3d', 'flow')
TYPE_NAMES = {'camera.pinhole': 'Pinhole camera', 'camera.fisheye': 'Fisheye camera', 'lidar': 'Lidar',
              'radar': 'Radar', 'gnss': 'GNSS', 'imu': 'IMU', 'audio_array': 'Microphone array (synthetic)'}
FIELDS = {     # type: [(key, label, kind, lo, hi, decimals/None, unit)]
    'camera.pinhole': [('width', 'Width', 'int', 16, 8192, None, 'px'), ('height', 'Height', 'int', 16, 8192, None, 'px'),
                       ('fov', 'Field of view (horiz.)', 'float', 1, 179, 1, '°'),
                       ('rate_hz', 'Rate', 'float', 1, 100, 1, 'Hz')],
    'camera.fisheye': [('width', 'Width', 'int', 16, 8192, None, 'px'), ('height', 'Height', 'int', 16, 8192, None, 'px'),
                       ('lens.fov', 'Field of view', 'float', 60, 270, 1, '°'),
                       ('lens.f_px', 'Focal length (0: fit)', 'float', 0, 5000, 1, 'px/rad'),
                       ('rate_hz', 'Rate', 'float', 1, 100, 1, 'Hz')],
    'lidar': [('channels', 'Channels', 'int', 1, 256, None, ''), ('range', 'Range', 'float', 1, 500, 1, 'm'),
              ('points_per_second', 'Points per second', 'int', 1000, 5000000, None, ''),
              ('rotation_frequency', 'Rotation', 'float', 1, 50, 1, 'Hz'),
              ('upper_fov', 'Upper field of view', 'float', -90, 90, 1, '°'),
              ('lower_fov', 'Lower field of view', 'float', -90, 90, 1, '°'),
              ('horizontal_fov', 'Horizontal field of view', 'float', 1, 360, 1, '°'),
              ('noise_stddev', 'Range noise (1 σ)', 'float', 0, 1, 3, 'm')],
    'radar': [('horizontal_fov', 'Horizontal field of view', 'float', 1, 180, 1, '°'),
              ('vertical_fov', 'Vertical field of view', 'float', 1, 90, 1, '°'),
              ('range', 'Range', 'float', 1, 300, 1, 'm'),
              ('points_per_second', 'Points per second', 'int', 100, 100000, None, ''),
              ('rate_hz', 'Rate', 'float', 1, 100, 1, 'Hz')],
    'gnss': [('rate_hz', 'Rate', 'float', 0.1, 100, 1, 'Hz'),
             ('noise_lat_stddev', 'Latitude noise (1 σ)', 'float', 0, 0.001, 7, '°'),
             ('noise_lon_stddev', 'Longitude noise (1 σ)', 'float', 0, 0.001, 7, '°'),
             ('noise_alt_stddev', 'Altitude noise (1 σ)', 'float', 0, 10, 2, 'm')],
    'imu': [('rate_hz', 'Rate', 'float', 1, 1000, 0, 'Hz'),
            ('noise_accel_stddev', 'Accelerometer noise (1 σ)', 'float', 0, 1, 4, 'm/s²'),
            ('noise_gyro_stddev', 'Gyroscope noise (1 σ)', 'float', 0, 1, 5, 'rad/s')],
    'audio_array': [('mics', 'Microphones (circle)', 'int', 2, 64, None, ''),
                    ('radius', 'Circle radius', 'float', 0.01, 2, 3, 'm')],
}


def _get(d, dotted, default=None):
    for k in dotted.split('.'):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


class RigsPage(Page):
    key, title, icon = 'rigs', 'Rigs', 'car-top'
    about = 'A vehicle with its sensors: where they sit, what they record, what they cost to render.'

    def __init__(self, win):
        super().__init__(win)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 6, 12)
        lv.addWidget(label(self.title, page=True))
        self.rigs = QListWidget()
        self.rigs.setMaximumHeight(120)
        self.rigs.currentRowChanged.connect(lambda _: self.show_rig())
        lv.addWidget(self.rigs)
        rb = QHBoxLayout()
        rb.addWidget(button('New', self.add_rig))
        rb.addWidget(button('Duplicate', self.dup_rig))
        self.rm_rig = button('Remove', self.remove_rig)
        rb.addWidget(self.rm_rig)
        io = QToolButton()
        io.setText('Import / export')
        io.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(io)
        m.addAction('Import a Falcon rig (JSON)…', lambda: self.import_file('json'))
        m.addAction('Import from ROS URDF…', lambda: self.import_file('urdf'))
        m.addAction('Import nuScenes calibration (folder)…', lambda: self.import_file('nuscenes'))
        m.addSeparator()
        m.addAction('Export as Falcon JSON…', lambda: self.export_file('json'))
        m.addAction('Export as ROS URDF…', lambda: self.export_file('urdf'))
        m.addAction('Export ROS TF (static transforms, JSON)…', lambda: self.export_file('tf'))
        m.addAction('Export nuScenes calibration…', lambda: self.export_file('nuscenes'))
        io.setMenu(m)
        rb.addWidget(io)
        lv.addLayout(rb)
        f = Form()
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self.set_rig_field('name', self.name.text().strip() or 'Rig'))
        f.row('Name', self.name)
        self.vehicle = QComboBox()
        self.vehicle.setEditable(True)
        self.vehicle.addItems(list(R.VEHICLES))
        self.vehicle.activated.connect(lambda _: self.set_rig_field('vehicle', self.vehicle.currentText()))
        f.row('Vehicle', self.vehicle)
        lv.addWidget(label('CARLA\'s car carries the rig in the simulation, not your own vehicle body.', muted=True,
                           small=True, wrap=True))
        lv.addLayout(f)
        lv.addWidget(label('Sensors', h=True))
        self.sensors = QListWidget()
        self.sensors.currentRowChanged.connect(lambda _: self.show_sensor())
        lv.addWidget(self.sensors, 1)
        sb = QHBoxLayout()
        add = QToolButton()
        add.setText('Add')
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        am = QMenu(add)
        for t, n in TYPE_NAMES.items():
            am.addAction(n, lambda tt=t: self.add_sensor(tt))
        add.setMenu(am)
        sb.addWidget(add)
        sb.addWidget(button('Duplicate', self.dup_sensor))
        sb.addWidget(button('Remove', self.remove_sensor))
        lv.addLayout(sb)
        self.budget_bar = QProgressBar()
        self.budget_bar.setRange(0, 1000)
        self.budget_bar.setProperty('inline', True)
        lv.addWidget(self.budget_bar)
        self.budget_lbl = label('', muted=True, small=True, wrap=True)
        lv.addWidget(self.budget_lbl)
        self.problems = label('', wrap=True, small=True)
        self.problems.setProperty('crit', True)
        lv.addWidget(self.problems)
        split.addWidget(left)

        self.view = make_view()
        self.view.selected.connect(self.select_sensor)
        self.view.poseDropped.connect(self.drop_pose)
        split.addWidget(self.view)

        right = QWidget()
        self.rv = QVBoxLayout(right)
        self.rv.setContentsMargins(6, 12, 12, 12)
        self.sensor_sec = Section('Sensor')
        self.sensor_box = QWidget()                     # holds one form at a time
        self.sensor_box_layout = QVBoxLayout(self.sensor_box)
        self.sensor_box_layout.setContentsMargins(0, 0, 0, 0)
        self.sensor_form_host = None
        self.sensor_sec.add(self.sensor_box)
        self.cost = KeyValues(('Cost',))
        right.setMinimumWidth(420)
        self.sensor_sec.add(self.cost)
        self.rv.addWidget(self.sensor_sec)
        self.rv.addStretch(1)
        split.addWidget(scroll(right))
        split.setSizes([300, 620, 480])
        self.refresh()

    # ------------------------------------------------------------------ model
    def rig_index(self):
        i = self.rigs.currentRow()
        return i if 0 <= i < len(self.win.project.doc['rigs']) else None

    def rig(self):
        i = self.rig_index()
        return None if i is None else self.win.project.doc['rigs'][i]

    def sensor_index(self):
        r = self.rig()
        j = self.sensors.currentRow()
        return j if r is not None and 0 <= j < len(r['sensors']) else None

    def edit(self, path_tail, value, text, merge=''):
        i = self.rig_index()
        if i is None:
            return
        self.win.edit(Edit('set', ['rigs', i] + path_tail, value, text=text), merge)

    # ------------------------------------------------------------------ view
    def refresh(self):
        i = max(self.rigs.currentRow(), 0)
        self.rigs.blockSignals(True)
        self.rigs.clear()
        for r in self.win.project.doc['rigs']:
            self.rigs.addItem(QListWidgetItem(icons.icon('car-top', 16), r['name']))
        self.rigs.setCurrentRow(min(i, self.rigs.count() - 1))
        self.rigs.blockSignals(False)
        self.show_rig()

    def show_rig(self):
        r = self.rig()
        set_enabled(self.rm_rig, r is not None and len(self.win.project.doc['rigs']) > 1,
                    'A project keeps at least one rig' if r is not None else 'Choose a rig first')
        if r is None:
            self.view.set_rig(None)
            return
        for w in (self.name, self.vehicle):
            w.blockSignals(True)
        self.name.setText(r['name'])
        self.vehicle.setCurrentText(r['vehicle'])
        for w in (self.name, self.vehicle):
            w.blockSignals(False)
        j = max(self.sensors.currentRow(), 0)
        self.sensors.blockSignals(True)
        self.sensors.clear()
        for k, s in enumerate(r['sensors']):
            c = R.sensor_cost(s)
            it = QListWidgetItem(f'{s["name"]}  ·  {TYPE_NAMES.get(s["type"], s["type"])}'
                                 + (f'  ·  {c["mpx"]:.1f} Mpx' if c['mpx'] else ''))
            self.sensors.addItem(it)
        self.sensors.setCurrentRow(min(j, self.sensors.count() - 1))
        self.sensors.blockSignals(False)
        lim = float(self.win.settings.get('render_budget_mpx'))
        mpx = budget.rig_mpx(r)
        self.budget_bar.setValue(int(1000 * min(mpx / lim, 1.0)))
        self.budget_lbl.setText(f'{mpx:.1f} of {lim:g} Mpx rendered per step (images and label cameras); a '
                                '0.9.16 server capped at 12 GB crashed at 306')
        probs = R.validate_rig(r) + ([] if mpx <= lim else ['over the render budget: runs will be refused'])
        self.problems.setText('\n'.join('⚠ ' + p for p in probs))
        self.show_sensor()

    def show_sensor(self):
        r, j = self.rig(), self.sensor_index()
        # a fresh form each time: the old one is detached at once (hidden now, deleted later)
        if self.sensor_form_host is not None:
            self.sensor_box_layout.removeWidget(self.sensor_form_host)
            self.sensor_form_host.setParent(None)
            self.sensor_form_host.deleteLater()
        self.sensor_form_host = QWidget()
        self.sensor_box_layout.addWidget(self.sensor_form_host)
        self.view.set_rig(r, r['sensors'][j]['id'] if r is not None and j is not None else '')
        if j is None:
            self.sensor_sec.setVisible(False)
            return
        self.sensor_sec.setVisible(True)
        s = r['sensors'][j]
        f = Form()
        f.setContentsMargins(0, 0, 0, 0)
        self.sensor_form_host.setLayout(f)
        nm = QLineEdit(s['name'])
        nm.editingFinished.connect(lambda: self.edit(['sensors', j, 'name'], nm.text().strip() or s['id'],
                                                     'Rename a sensor'))
        f.row('Name', nm)
        f.row('Type', label(TYPE_NAMES.get(s['type'], s['type'])))
        for k, text, unit, lo, hi in (('x', 'x (forward)', 'm', -10, 10), ('y', 'y (right)', 'm', -10, 10),
                                      ('z', 'z (up)', 'm', -2, 10), ('roll', 'Roll', '°', -180, 180),
                                      ('pitch', 'Pitch (up)', '°', -90, 90), ('yaw', 'Yaw (right)', '°', -180, 180)):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(3 if unit == 'm' else 2)
            sp.setSingleStep(0.05 if unit == 'm' else 1.0)
            sp.setSuffix(f' {unit}')
            sp.setValue(float(s['pose'][k]))
            sp.editingFinished.connect(lambda k=k, sp=sp: self.set_pose(j, {k: sp.value()}))
            f.row(text, sp)
        for key, text, kind, lo, hi, dec, unit in FIELDS.get(s['type'], []):
            val = _get(s, key, 0)
            if kind == 'int':
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setValue(int(val or 0))
            else:
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setDecimals(dec)
                w.setValue(float(val or 0))
            if unit:
                w.setSuffix(f' {unit}')
            w.editingFinished.connect(lambda key=key, w=w, kind=kind: self.set_field(
                j, key, (int(w.value()) if kind == 'int' else float(w.value())) if not (
                    key == 'lens.f_px' and w.value() == 0) else None))
            f.row(text, w)
        if s['type'] == 'camera.fisheye':
            model = QComboBox()
            model.addItems(['equidistant', 'equisolid', 'stereographic', 'orthographic', 'kannala_brandt'])
            model.setCurrentText(_get(s, 'lens.model', 'equidistant'))
            model.activated.connect(lambda _: self.set_field(j, 'lens.model', model.currentText()))
            f.row('Lens model', model)
        if s['type'].startswith('camera.') or s['type'] == 'lidar':
            row = QHBoxLayout()
            avail = LABELS if s['type'].startswith('camera.') else ('semantic', 'instance')
            for lb in avail:
                cb = QCheckBox(lb)
                cb.setChecked(lb in (s.get('labels') or []))
                cb.toggled.connect(lambda on, lb=lb: self.toggle_label(j, lb, on))
                row.addWidget(cb)
            f.row('Labels', row)
        if s['type'].startswith('camera.') or s['type'] in ('lidar', 'radar'):
            wp = QCheckBox('when the world enables it (docs/sensor-physics.md)')
            wp.setChecked(s.get('weather_physics', True) is not False)
            wp.setToolTip('Falcon\'s weather models for this sensor (lidar and radar in rain and fog, lens droplets, '
                          'glare); off: this sensor records CARLA\'s data as it is')
            wp.toggled.connect(lambda on: self.set_field(j, 'weather_physics', bool(on)))
            f.row('Weather physics', wp)
        self.fault_editor(f, j, s)
        c = R.sensor_cost(s)
        self.cost.set('Cost', f'{c["mpx"]:.2f} Mpx per step — {c["note"]}' if c['mpx'] else c['note'])

    FAULT_TEMPLATES = {'mount_rotation': {'deg': [0.0, 0.0, 1.0]}, 'mount_translation': {'m': [0.0, 0.0, 0.02]},
                       'clock_offset': {'ms': 20}, 'drop': {'rate': 0.05}, 'freeze': {'from_s': 2.0, 'to_s': 3.0},
                       'noise': {'sigma': 3.0}, 'dirt': {'cover': 0.1}, 'rain': {'cover': 0.1},
                       'glare': {'strength': 0.5, 'u': 0.5, 'v': 0.25}, 'rolling_shutter': {'readout_ms': 30},
                       'compression': {'quality': 40}, 'motion_blur': {'intensity': 0.45, 'max_distortion': 0.35}}

    def fault_editor(self, f, j, s):
        """The sensor's faults as JSON (one object per fault), added from templates, validated before applying."""
        from PyQt6.QtWidgets import QPlainTextEdit
        from ...core import faults as FL
        box = QVBoxLayout()
        ed = QPlainTextEdit(json.dumps(s.get('faults') or [], indent=1))
        ed.setFixedHeight(110)
        box.addWidget(ed)
        row = QHBoxLayout()
        add = QToolButton()
        add.setText('Add fault')
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(add)
        kinds = FL.FAULTS if s['type'].startswith('camera.') else ('mount_rotation', 'mount_translation',
                                                                     'clock_offset', 'drop', 'freeze')
        for k in kinds:
            m.addAction(k.replace('_', ' '), lambda k=k: self.set_faults(j, (s.get('faults') or []) +
                                                                      [dict({'kind': k}, **self.FAULT_TEMPLATES[k])]))
        add.setMenu(m)
        row.addWidget(add)
        msg = label('', small=True, wrap=True)

        def apply():
            try:
                fl = json.loads(ed.toPlainText() or '[]')
                probs = FL.validate(fl) if isinstance(fl, list) else ['a list of faults is needed']
            except ValueError as e:
                probs = [f'not JSON: {e}']
            if probs:
                msg.setText('⚠ ' + '; '.join(probs))
                return
            self.set_faults(j, fl)
        row.addWidget(button('Apply', apply))
        row.addStretch(1)
        box.addLayout(row)
        box.addWidget(msg)
        f.row('Faults', box, 'outputs keep the planned calibration')

    def set_faults(self, j, fl):
        s = self.rig()['sensors'][j]
        if (s.get('faults') or []) != fl:
            self.edit(['sensors', j, 'faults'], fl, f'{s["name"]}: faults')

    # ------------------------------------------------------------------ edits
    def set_rig_field(self, key, value):
        r = self.rig()
        if r is not None and r.get(key) != value:
            self.edit([key], value, f'Rig {key}')

    def set_pose(self, j, upd: dict, merge=''):
        r = self.rig()
        pose = dict(r['sensors'][j]['pose'], **{k: float(v) for k, v in upd.items()})
        if pose != r['sensors'][j]['pose']:
            self.edit(['sensors', j, 'pose'], pose, f'Move {r["sensors"][j]["name"]}', merge)

    def set_field(self, j, dotted, value):
        r = self.rig()
        s = copy.deepcopy(r['sensors'][j])
        cur = s
        keys = dotted.split('.')
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        if cur.get(keys[-1]) == value:
            return
        cur[keys[-1]] = value
        self.edit(['sensors', j], s, f'{s["name"]}: {keys[-1]}')

    def toggle_label(self, j, lb, on):
        s = self.rig()['sensors'][j]
        cur = list(s.get('labels') or [])
        new = sorted(set(cur) | {lb}) if on else [x for x in cur if x != lb]
        if new != cur:
            self.edit(['sensors', j, 'labels'], new, f'{s["name"]}: labels')

    def drop_pose(self, sid, pose):
        r = self.rig()
        j = next(k for k, s in enumerate(r['sensors']) if s['id'] == sid)
        self.set_pose(j, pose)

    def select_sensor(self, sid):
        r = self.rig()
        j = next((k for k, s in enumerate(r['sensors']) if s['id'] == sid), None)
        if j is not None:
            self.sensors.setCurrentRow(j)

    def add_sensor(self, stype):
        i = self.rig_index()
        if i is None:
            return
        n = sum(1 for s in self.rig()['sensors'] if s['type'] == stype) + 1
        base = stype.split('.')[-1]
        s = R.new_sensor(stype, new_id(base), f'{base}_{n}', {'z': 1.6, 'x': 1.0 if stype.startswith('camera') else 0.0})
        self.win.edit(Edit('insert', ['rigs', i, 'sensors'], s, text=f'Add {TYPE_NAMES[stype].lower()}'))
        self.sensors.setCurrentRow(self.sensors.count() - 1)

    def dup_sensor(self):
        i, j = self.rig_index(), self.sensor_index()
        if j is None:
            return
        s = copy.deepcopy(self.rig()['sensors'][j])
        s['id'], s['name'] = new_id(s['type'].split('.')[-1]), s['name'] + '_copy'
        self.win.edit(Edit('insert', ['rigs', i, 'sensors'], s, text=f'Duplicate {s["name"]}'))

    def remove_sensor(self):
        i, j = self.rig_index(), self.sensor_index()
        if j is not None:
            self.win.edit(Edit('remove', ['rigs', i, 'sensors'], index=j, text='Remove a sensor'))

    def add_rig(self):
        from ...core.project import default_rig
        r = default_rig()
        r['id'], r['name'] = new_id('rig'), f'Rig {len(self.win.project.doc["rigs"]) + 1}'
        self.win.edit(Edit('insert', ['rigs'], r, text='New rig'))
        self.rigs.setCurrentRow(self.rigs.count() - 1)

    def dup_rig(self):
        r = self.rig()
        if r is None:
            return
        r = copy.deepcopy(r)
        r['id'], r['name'] = new_id('rig'), r['name'] + ' (copy)'
        self.win.edit(Edit('insert', ['rigs'], r, text='Duplicate a rig'))
        self.rigs.setCurrentRow(self.rigs.count() - 1)

    def remove_rig(self):
        i = self.rig_index()
        if i is not None and len(self.win.project.doc['rigs']) > 1:
            self.win.edit(Edit('remove', ['rigs'], index=i, text='Remove a rig'))

    # ------------------------------------------------------------------ files
    def import_file(self, kind):
        try:
            if kind == 'nuscenes':
                d = QFileDialog.getExistingDirectory(self, 'nuScenes table folder (sensor.json, calibrated_sensor.json)')
                if not d:
                    return
                with open(os.path.join(d, 'sensor.json'), encoding='utf-8') as fh:
                    srows = json.load(fh)
                with open(os.path.join(d, 'calibrated_sensor.json'), encoding='utf-8') as fh:
                    crows = json.load(fh)
                r = R.import_nuscenes(srows, crows)
            else:
                path, _ = QFileDialog.getOpenFileName(self, 'Import a rig', os.path.expanduser('~'),
                                                      'Falcon rig (*.json)' if kind == 'json' else 'URDF (*.urdf *.xml)')
                if not path:
                    return
                with open(path, encoding='utf-8') as fh:
                    text = fh.read()
                r = R.import_json(text) if kind == 'json' else R.import_urdf(text)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, 'Import', f'Cannot import it: {e}')
            return
        r = copy.deepcopy(r)
        r['id'] = new_id('rig')
        self.win.edit(Edit('insert', ['rigs'], r, text=f'Import {r["name"]}'))
        self.rigs.setCurrentRow(self.rigs.count() - 1)

    def export_file(self, kind):
        r = self.rig()
        if r is None:
            return
        ext = {'json': 'falcon-rig.json', 'urdf': 'urdf', 'tf': 'tf.json', 'nuscenes': ''}[kind]
        if kind == 'nuscenes':
            d = QFileDialog.getExistingDirectory(self, 'Folder for sensor.json and calibrated_sensor.json')
            if not d:
                return
            doc = R.export_nuscenes(r)
            for k in ('sensor', 'calibrated_sensor'):
                with open(os.path.join(d, f'{k}.json'), 'w', encoding='utf-8') as fh:
                    json.dump(doc[k], fh, indent=2)
            self.win.log.append_line(f'exported nuScenes calibration to {d} ({doc["ego_frame"]})')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export the rig', os.path.join(os.path.expanduser('~'),
                                                                                   f'{r["name"]}.{ext}'))
        if not path:
            return
        text = {'json': lambda: R.export_json(r), 'urdf': lambda: R.export_urdf(r),
                'tf': lambda: json.dumps(R.tf_list(r), indent=2) + '\n'}[kind]()
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)
        self.win.log.append_line(f'exported {r["name"]} to {path}')

