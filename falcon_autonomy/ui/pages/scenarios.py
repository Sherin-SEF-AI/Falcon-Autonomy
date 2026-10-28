"""Scenarios: the scenario studio — the library (built-ins and the project's own), a sketch of the scene with the
actors where a sample puts them and their parameter ranges as bands, the actors, the parameters as distributions and
the timeline of triggers and actions, running one sample or a sweep, OpenSCENARIO import and export."""
from __future__ import annotations

import copy
import json
import os
import sys

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QListWidget,
                             QListWidgetItem, QMessageBox, QPlainTextEdit, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import scenarios as SC
from ...core.jobs import JobSpec
from ...core.project import Edit, new_id
from ..theme import THEME
from ..widgets import Form, Page, Section, button, label

ROWS = ('far_sidewalk', 'opposite', 'adjacent', 'line', 'ego', 'kerb', 'sidewalk')   # schematic, right-hand traffic
KIND_COL = {'pedestrian': '#d95926', 'child': '#d95926', 'bicycle': '#c98500', 'motorcycle': '#c98500',
            'bus': '#9085e9', 'truck': '#9085e9', 'van': '#9085e9'}
ACTOR_COLS = ('id', 'kind', 'lane', 'ahead_m', 'lateral_m', 'heading', 'speed_kmh', 'start')
PARAM_COLS = ('name', 'dist', 'lo', 'hi', 'choices', 'unit', 'about')


def _num(text):
    try:
        return float(text) if '.' in text or 'e' in text.lower() else int(text)
    except (ValueError, AttributeError):
        return text


class SceneSketch(QWidget):
    """The road seen from above, ego at the left driving right: lanes as rows (right-hand traffic; mirrored for
    left-hand), each actor at a sample's position, a parameter's range as a translucent band, distance triggers as
    dashed lines. A numeric ahead_m can be dragged."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.scn = None
        self.sample = None
        self.lht = False
        self.drag = None
        self.setMinimumHeight(240)

    def set_scenario(self, scn, sample, lht):
        self.scn, self.sample, self.lht = scn, sample, lht
        self.update()

    def geom(self):
        rows = list(ROWS[::-1]) if self.lht else list(ROWS)
        h = self.height() - 30
        rh = h / len(rows)
        span = 140.0
        x0 = 60.0
        k = (self.width() - x0 - 20) / span
        return rows, rh, x0, k, span

    def y_of(self, lane, rows, rh):
        return 10 + rh * (rows.index(lane) + 0.5) if lane in rows else 10 + rh * (rows.index('ego') + 0.5)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['viewport']))
        if self.scn is None:
            return
        rows, rh, x0, k, span = self.geom()
        for i, r in enumerate(rows):
            y = 10 + rh * i
            col = THEME.tok['raised'] if r in ('sidewalk', 'far_sidewalk') else THEME.tok['panel']
            if r == 'line':
                continue
            p.fillRect(QRectF(0, y, self.width(), rh), QColor(col))
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(QRectF(4, y, 56, rh), Qt.AlignmentFlag.AlignVCenter, r.replace('_', ' '))
        ly = self.y_of('line', rows, rh)
        p.setPen(QPen(QColor(THEME.tok['ink2']), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x0, ly), QPointF(self.width(), ly))
        for m in range(0, int(span) + 1, 20):
            x = x0 + m * k
            p.setPen(QColor(THEME.tok['grid2']))
            p.drawLine(QPointF(x, 10), QPointF(x, self.height() - 20))
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(QPointF(x - 8, self.height() - 6), f'{m} m')
        ey = self.y_of('ego', rows, rh)
        p.setBrush(QColor('#199e70'))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(x0 - 4.6 * k / 2, ey - 2.0 * k / 2, 4.6 * k, 2.0 * k))
        params = self.scn.get('params') or {}
        conc = self.sample
        for a, ca in zip(self.scn.get('actors', []), conc.get('actors', [])):
            lane = ca['place'].get('lane', 'ego')
            y = self.y_of(lane, rows, rh) + float(ca['place'].get('lateral_m', 0.0)) * k * (-1 if self.lht else 1)
            expr = str(a['place'].get('ahead_m', 0))
            pname = expr.strip().lstrip('-$') if expr.strip().startswith(('$', '-$')) else None
            if pname in params and params[pname].get('dist') == 'uniform':
                sgn = -1 if expr.strip().startswith('-') else 1
                lo, hi = sorted((sgn * params[pname]['lo'], sgn * params[pname]['hi']))
                c = QColor(KIND_COL.get(ca['kind'], '#3987e5'))
                c.setAlphaF(0.18)
                p.fillRect(QRectF(x0 + lo * k, y - rh * 0.35, (hi - lo) * k, rh * 0.7), c)
            x = x0 + float(ca['place'].get('ahead_m', 0.0)) * k
            L, W = {'bus': (10.3, 3.9), 'truck': (6, 2.6), 'van': (5.5, 2), 'motorcycle': (2.2, 0.9),
                    'bicycle': (1.7, 0.6), 'pedestrian': (0.6, 0.6), 'child': (0.5, 0.5),
                    'autorickshaw': (2.6, 1.3), 'cycle_rickshaw': (2.2, 1.0), 'cattle': (1.6, 0.5),
                    'handcart': (2.4, 1.2), 'vendor': (2.3, 1.5)}.get(ca['kind'], (4.6, 2.0))
            p.setBrush(QColor(KIND_COL.get(ca['kind'], '#3987e5')))
            p.drawRect(QRectF(x - L * k / 2, y - W * k / 2, L * k, W * k))
            arrow = {'lane': 1, 'against': -1}.get(ca.get('heading'), 0)
            p.setPen(QPen(QColor(THEME.tok['ink']), 1.5))
            if arrow:
                p.drawLine(QPointF(x, y), QPointF(x + arrow * 4 * k, y))
            elif ca.get('heading') == 'across':
                p.drawLine(QPointF(x, y), QPointF(x, y + (-1 if not self.lht else 1) * rh * 1.5))
            p.drawText(QPointF(x - L * k / 2, y - W * k / 2 - 3), ca['id'])
            p.setPen(Qt.PenStyle.NoPen)
        p.setPen(QPen(QColor(THEME.tok['muted']), 1, Qt.PenStyle.DotLine))
        for ev_ in conc.get('timeline', []):
            t = ev_.get('trigger', {})
            if t.get('type') == 'distance':
                a = next((x for x in conc['actors'] if x['id'] == t['actor']), None)
                if a is not None:
                    xt = x0 + (float(a['place'].get('ahead_m', 0)) - float(t['lt_m'])) * k
                    p.drawLine(QPointF(xt, 10), QPointF(xt, self.height() - 20))
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QPointF(x0, self.height() - 22), 'dotted: where the ego triggers an event · bands: a parameter\'s '
                                                    'range · drag an actor with a fixed position')

    def mousePressEvent(self, ev):
        if self.scn is None or not self.page.editable():
            return
        rows, rh, x0, k, _ = self.geom()
        for i, (a, ca) in enumerate(zip(self.scn.get('actors', []), self.sample.get('actors', []))):
            x = x0 + float(ca['place'].get('ahead_m', 0.0)) * k
            y = self.y_of(ca['place'].get('lane', 'ego'), rows, rh)
            if abs(ev.position().x() - x) < 18 and abs(ev.position().y() - y) < rh / 2 and \
                    isinstance(a['place'].get('ahead_m'), (int, float)):
                self.drag = i
                return

    def mouseMoveEvent(self, ev):
        if self.drag is None:
            return
        rows, rh, x0, k, _ = self.geom()
        self.scn['actors'][self.drag]['place']['ahead_m'] = round((ev.position().x() - x0) / k, 1)
        self.sample['actors'][self.drag]['place']['ahead_m'] = self.scn['actors'][self.drag]['place']['ahead_m']
        self.update()

    def mouseReleaseEvent(self, ev):
        if self.drag is not None:
            v = self.scn['actors'][self.drag]['place']['ahead_m']
            self.page.set_actor_field(self.drag, 'ahead_m', v, from_sketch=True)
        self.drag = None


class JsonDialog(QDialog):
    def __init__(self, parent, title, value, check):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 520)
        v = QVBoxLayout(self)
        self.ed = QPlainTextEdit(json.dumps(value, indent=1))
        v.addWidget(self.ed)
        self.msg = label('', small=True, wrap=True)
        v.addWidget(self.msg)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.check, self.value = check, None

    def ok(self):
        try:
            val = json.loads(self.ed.toPlainText())
        except ValueError as e:
            self.msg.setText(f'⚠ not JSON: {e}')
            return
        probs = self.check(val)
        if probs:
            self.msg.setText('⚠ ' + '; '.join(probs))
            return
        self.value = val
        self.accept()


class ScenariosPage(Page):
    key, title, icon = 'scenarios', 'Scenarios', 'timeline'
    about = 'Script scenarios: actors around the ego, triggers and actions, parameters as distributions; run them.'

    def __init__(self, win):
        super().__init__(win)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(label(self.title, title=True))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(lambda _: self.show_scenario())
        lv.addWidget(self.list, 1)
        lb = QHBoxLayout()
        self.copy_btn = button('Copy to project', self.copy_builtin)
        lb.addWidget(self.copy_btn)
        self.rm_btn = button('Remove', self.remove)
        lb.addWidget(self.rm_btn)
        lv.addLayout(lb)
        io = QHBoxLayout()
        io.addWidget(button('Import .xosc…', self.import_xosc))
        io.addWidget(button('Export .xosc…', self.export_xosc))
        lv.addLayout(io)
        split.addWidget(left)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        self.title_lbl = label('', title=True)
        self.about_lbl = label('', muted=True, wrap=True)
        mv.addWidget(self.title_lbl)
        mv.addWidget(self.about_lbl)
        self.sketch = SceneSketch(self)
        mv.addWidget(self.sketch, 1)
        self.tabs = QTabWidget()
        self.actors = QTableWidget(0, len(ACTOR_COLS))
        self.actors.setHorizontalHeaderLabels(list(ACTOR_COLS))
        self.actors.cellChanged.connect(self.actor_cell)
        at = QWidget()
        atl = QVBoxLayout(at)
        atl.addWidget(self.actors)
        atl.addWidget(button('Add an actor', self.add_actor))
        self.tabs.addTab(at, 'Actors')
        self.params = QTableWidget(0, len(PARAM_COLS))
        self.params.setHorizontalHeaderLabels(list(PARAM_COLS))
        self.params.cellChanged.connect(self.param_cell)
        pt = QWidget()
        ptl = QVBoxLayout(pt)
        ptl.addWidget(self.params)
        ptl.addWidget(button('Add a parameter', self.add_param))
        self.tabs.addTab(pt, 'Parameters')
        self.timeline = QTreeWidget()
        self.timeline.setHeaderLabels(['Trigger / action', 'Details'])
        tt = QWidget()
        ttl = QVBoxLayout(tt)
        ttl.addWidget(self.timeline)
        ttl.addWidget(button('Edit the timeline…', self.edit_timeline))
        self.tabs.addTab(tt, 'Timeline')
        mv.addWidget(self.tabs, 1)
        self.problems = label('', small=True, wrap=True)
        self.problems.setProperty('crit', True)
        mv.addWidget(self.problems)
        split.addWidget(mid)

        right = QWidget()
        rv = QVBoxLayout(right)
        run = Section('Run', 'A sample is drawn with the seed; the same seed on the same server reproduces it. A '
                             'sweep runs unattended and resumes where it stopped.')
        f = Form()
        self.world = QComboBox()
        self.rig = QComboBox()
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.valueChanged.connect(lambda _: self.show_sketch())
        self.samples = QSpinBox()
        self.samples.setRange(2, 10000)
        self.samples.setValue(20)
        f.row('World', self.world)
        f.row('Rig', self.rig)
        f.row('Seed', self.seed)
        f.row('Sweep samples', self.samples)
        run.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Run this sample', self.run_once, role='primary'))
        b.addWidget(button('Sweep', self.sweep))
        run.add(b)
        rv.addWidget(run)
        res = Section('Results')
        self.results = QTreeWidget()
        self.results.setHeaderLabels(['Run', 'Passed', 'Criticality', 'Min TTC', 'Values'])
        self.results.setRootIsDecorated(False)
        self.results.itemDoubleClicked.connect(lambda it, _c: self.win.go('data').show_run(it.data(0, 256)))
        res.add(self.results)
        res.add(button('Refresh', self.load_results))
        rv.addWidget(res, 1)
        split.addWidget(right)
        split.setSizes([240, 760, 420])
        win.bridge.finished.connect(lambda j: self.load_results() if j.spec.kind in ('drive', 'study') else None)
        self.refresh()

    # ------------------------------------------------------------------ model
    def entries(self):
        out = [('builtin', k, SC.builtin(k)) for k in SC.LIBRARY]
        out += [('project', s['id'], s) for s in self.win.project.doc['scenarios']]
        return out

    def current(self):
        i = self.list.currentRow()
        e = self.entries()
        return e[i] if 0 <= i < len(e) else (None, None, None)

    def editable(self):
        return self.current()[0] == 'project'

    def index(self):
        src, ident, _ = self.current()
        return next(k for k, s in enumerate(self.win.project.doc['scenarios']) if s['id'] == ident)

    def set(self, path, value, text):
        if not self.editable():
            return
        self.win.edit(Edit('set', ['scenarios', self.index()] + path, value, text=text))

    # ------------------------------------------------------------------ view
    def refresh(self):
        i = max(self.list.currentRow(), 0)
        self.list.blockSignals(True)
        self.list.clear()
        for src, ident, s in self.entries():
            it = QListWidgetItem(('★ ' if src == 'builtin' else '') + s['name'] +
                                 ('  (Indian roads)' if ident in SC.INDIAN else ''))
            it.setToolTip(s.get('about', ''))
            self.list.addItem(it)
        self.list.setCurrentRow(min(i, self.list.count() - 1))
        self.list.blockSignals(False)
        doc = self.win.project.doc
        for combo, items in ((self.world, doc['worlds']), (self.rig, doc['rigs'])):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for it in items:
                combo.addItem(it['name'], it['id'])
            k = combo.findData(cur)
            combo.setCurrentIndex(k if k >= 0 else 0)
            combo.blockSignals(False)
        self.show_scenario()

    def show_scenario(self):
        src, ident, s = self.current()
        self.copy_btn.setEnabled(src == 'builtin')
        self.rm_btn.setEnabled(src == 'project')
        if s is None:
            return
        self.title_lbl.setText(s['name'] + ('  — built-in (copy it to edit)' if src == 'builtin' else ''))
        self.about_lbl.setText(s.get('about', ''))
        ed = src == 'project'
        for t, cols, rows in ((self.actors, ACTOR_COLS, s.get('actors', [])),
                              (self.params, PARAM_COLS, sorted((s.get('params') or {}).items()))):
            t.blockSignals(True)
            t.setRowCount(0)
            for r in rows:
                k = t.rowCount()
                t.insertRow(k)
                for c, col in enumerate(cols):
                    if t is self.actors:
                        v = r['place'].get(col) if col in ('lane', 'ahead_m', 'lateral_m') else r.get(col)
                    else:
                        name, p = r
                        v = name if col == 'name' else ('|'.join(map(str, p['choices'])) if col == 'choices' and
                                                        'choices' in p else p.get(col))
                    it = QTableWidgetItem('' if v is None else str(v))
                    if not ed:
                        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    t.setItem(k, c, it)
            t.blockSignals(False)
        self.timeline.clear()
        for ev_ in s.get('timeline', []):
            t = ev_.get('trigger', {})
            desc = {'time': f'at {t.get("t_s")} s', 'distance': f'ego within {t.get("lt_m")} m of {t.get("actor")}',
                    'ahead': f'{t.get("actor")} more than {t.get("gt_m")} m ahead'}.get(t.get('type'), str(t))
            top = QTreeWidgetItem(['when', desc])
            for a in ev_.get('actions', []):
                QTreeWidgetItem(top, [f'{a["actor"]}: {a["do"]}', ', '.join(f'{k}={v}' for k, v in a.items()
                                                                            if k not in ('actor', 'do'))])
            self.timeline.addTopLevelItem(top)
            top.setExpanded(True)
        probs = SC.validate(s)
        self.problems.setText('\n'.join('⚠ ' + p for p in probs))
        self.show_sketch()
        self.load_results()

    def show_sketch(self):
        src, ident, s = self.current()
        if s is None:
            return
        try:
            sample = SC.sample(s, self.seed.value())
        except (KeyError, ValueError):
            sample = {'actors': [], 'timeline': []}
        world = self.win.project.item('worlds', self.world.currentData() or '') or {}
        lht = (world.get('osm') or {}).get('handedness') == 'LHT'
        self.sketch.set_scenario(copy.deepcopy(s), sample, lht)

    # ------------------------------------------------------------------ edits
    def actor_cell(self, r, c):
        v = _num(self.actors.item(r, c).text())
        self.set_actor_field(r, ACTOR_COLS[c], v)

    def set_actor_field(self, r, col, v, from_sketch=False):
        if not self.editable():
            return
        s = self.current()[2]
        a = copy.deepcopy(s['actors'][r])
        if col in ('lane', 'ahead_m', 'lateral_m'):
            a['place'][col] = v
        else:
            a[col] = v
        self.set(['actors', r], a, f'Actor {a["id"]}: {col}')

    def param_cell(self, r, c):
        s = self.current()[2]
        name = self.params.item(r, 0).text()
        old = sorted((s.get('params') or {}).items())[r][0] if r < len(s.get('params') or {}) else name
        p = copy.deepcopy((s.get('params') or {}).get(old, {}))
        col = PARAM_COLS[c]
        text = self.params.item(r, c).text()
        if col == 'choices':
            p['choices'] = [x for x in text.split('|') if x]
        elif col != 'name':
            p[col] = _num(text)
        params = {k: v for k, v in (s.get('params') or {}).items() if k != old}
        params[name] = p
        self.set(['params'], params, f'Parameter {name}')

    def add_actor(self):
        if not self.editable():
            return
        s = self.current()[2]
        a = {'id': f'actor{len(s.get("actors", [])) + 1}', 'kind': 'car', 'place': {'lane': 'ego', 'ahead_m': 30.0},
             'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}
        self.win.edit(Edit('insert', ['scenarios', self.index(), 'actors'], a, text='Add an actor'))

    def add_param(self):
        if not self.editable():
            return
        s = self.current()[2]
        params = dict(s.get('params') or {})
        params[f'p{len(params) + 1}'] = SC.U(0.0, 1.0)
        self.set(['params'], params, 'Add a parameter')

    def edit_timeline(self):
        if not self.editable():
            QMessageBox.information(self, 'Timeline', 'Copy the built-in to the project to edit it.')
            return
        s = self.current()[2]

        def check(val):
            if not isinstance(val, list):
                return ['a list of {trigger, actions} entries is needed']
            return SC.validate(dict(s, timeline=val))
        d = JsonDialog(self, 'Timeline', s.get('timeline', []), check)
        if d.exec() and d.value is not None:
            self.set(['timeline'], d.value, 'Edit the timeline')

    def copy_builtin(self):
        src, ident, s = self.current()
        if src != 'builtin':
            return
        c = copy.deepcopy(s)
        c.update(id=new_id('scn'), name=s['name'] + ' (copy)', builtin=False)
        self.win.edit(Edit('insert', ['scenarios'], c, text=f'Copy {s["name"]}'))
        self.list.setCurrentRow(self.list.count() - 1)

    def remove(self):
        if self.editable():
            self.win.edit(Edit('remove', ['scenarios'], index=self.index(), text='Remove a scenario'))

    # ------------------------------------------------------------------ running
    def run_once(self):
        from ...core.drive import run_dir_name, scenario_config
        from ...core.project import ProjectError
        src, ident, s = self.current()
        if s is None:
            return
        w = self.win
        try:
            cfg = scenario_config(w.project, s, self.world.currentData(), self.rig.currentData(),
                                  os.path.join(w.service.runs_dir(), run_dir_name(f'scenario-{ident}')),
                                  w.service.port, w.service.tm_port, seed=self.seed.value(),
                                  render_budget_mpx=float(w.settings.get('render_budget_mpx')),
                                  workspace=w.workspace())
            w.service.submit_drive(cfg, title=f'{s["name"]} (seed {self.seed.value()})')
        except (ProjectError, RuntimeError, ValueError) as e:
            QMessageBox.warning(self, 'Run', str(e))
            return
        w.jobs_dock.raise_()

    def study_name(self):
        src, ident, _ = self.current()
        return f'sweep-{ident}'

    def sweep(self):
        src, ident, s = self.current()
        w = self.win
        if s is None:
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        argv = [sys.executable, '-m', 'falcon_autonomy', 'sweep', w.project.path, '--scenario', ident, '--name',
                self.study_name(), '--samples', str(self.samples.value()), '--seed', str(self.seed.value()),
                '--world', self.world.currentData(), '--rig', self.rig.currentData()]
        if w.fake_carla:
            argv += ['--fake-carla', w.fake_carla]
        w.queue.submit(JobSpec('study', f'Sweep {s["name"]} ({self.samples.value()})', argv, memory_gb=0.5))
        w.jobs_dock.raise_()

    def load_results(self):
        self.results.clear()
        src, ident, s = self.current()
        if s is None or not self.win.project.path:
            return
        p = os.path.join(self.win.workspace(), 'studies', self.study_name(), 'results.jsonl')
        if not os.path.isfile(p):
            return
        with open(p, encoding='utf-8') as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
        for r in sorted(rows, key=lambda r: -r.get('objective', 0)):
            sc = r.get('scenario') or {}
            c = sc.get('criticality') or {}
            ttc = c.get('min_ttc_s')
            it = QTreeWidgetItem([os.path.basename(r['run']), 'yes' if sc.get('passed') else 'NO',
                                  f'{r.get("objective", 0):.3f}', '—' if ttc is None else f'{ttc:.2f} s',
                                  ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                            for k, v in sorted(r['values'].items()))])
            it.setData(0, 256, r['run'])
            self.results.addTopLevelItem(it)

    # ------------------------------------------------------------------ OpenSCENARIO
    def export_xosc(self):
        from ...core import openscenario as X
        src, ident, s = self.current()
        if s is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export OpenSCENARIO', os.path.join(
            os.path.expanduser('~'), f'{ident}-seed{self.seed.value()}.xosc'), 'OpenSCENARIO (*.xosc)')
        if path:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(X.export_xosc(SC.sample(s, self.seed.value())))
            self.win.log.append_line(f'exported {s["name"]} (seed {self.seed.value()}) to {path}')

    def import_xosc(self):
        from ...core import openscenario as X
        path, _ = QFileDialog.getOpenFileName(self, 'Import OpenSCENARIO', os.path.expanduser('~'),
                                              'OpenSCENARIO (*.xosc)')
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as fh:
                scn = X.import_xosc(fh.read())
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, 'Import', f'Cannot import it: {e}')
            return
        scn.update(id=new_id('scn'), schema=SC.SCENARIO_SCHEMA, params={})
        self.win.edit(Edit('insert', ['scenarios'], scn, text=f'Import {scn["name"]}'))
        self.list.setCurrentRow(self.list.count() - 1)
