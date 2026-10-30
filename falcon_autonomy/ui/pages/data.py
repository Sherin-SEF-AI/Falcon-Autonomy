"""Data: record a drive (its cost shown before it starts), and browse recorded runs frame by frame with their label
layers and their traffic-rule check (core.rules: each rule's violations on a timeline under the frames, with the
robustness and counts per km). The dataset engine, exports and coverage dashboards are the Datasets tab."""
from __future__ import annotations

import json
import os

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView,
                             QLabel, QLineEdit, QMessageBox, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import budget, labels
from ...core.drive import drive_config, run_dir_name
from ...core.paths import tilde
from ...core.project import ProjectError
from .. import icons, tasks
from ..widgets import (EmptyState, Form, KeyValues, Note, Page, PlaceholderTree, Section, button, label, scroll,
                       set_enabled)


class ImageView(QLabel):
    """An image scaled to fit, aspect kept."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self._pm = None

    def set_rgb(self, rgb: np.ndarray | None):
        if rgb is None:
            self._pm = None
            self.clear()
            return
        h, w, _ = rgb.shape
        img = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pm = QPixmap.fromImage(img)
        self._fit()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()

    def _fit(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))


class RuleTimeline(QWidget):
    """One row per rule over the run's time: violations as red bars, unknown stretches as grey ones, the current
    frame as a line; a click asks for that time (seeked to by the browser)."""
    seek = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.result = None
        self.t = None
        self.setMinimumHeight(60)
        self.setMouseTracking(True)

    def set_result(self, res):
        self.result = res
        self.setMinimumHeight(30 + 18 * len((res or {}).get('rules') or ()))
        self.update()

    def set_time(self, t):
        self.t = t
        self.update()

    def _geom(self):
        return 150, self.width() - 12, max(float((self.result or {}).get('duration_s') or 0.0), 1e-6)

    def mousePressEvent(self, ev):
        x0, x1, T = self._geom()
        if self.result and x0 <= ev.position().x() <= x1:
            self.seek.emit(T * (ev.position().x() - x0) / (x1 - x0))

    def paintEvent(self, ev):
        from ..theme import STATUS, THEME
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        res = self.result
        if not res:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Not checked yet')
            return
        x0, x1, T = self._geom()
        rh = 18

        def X(t):
            return x0 + (x1 - x0) * min(max(t / T, 0.0), 1.0)
        for k, r in enumerate(res['rules']):
            y = 4 + k * rh
            p.setPen(QColor(THEME.tok['ink2'] if r['state'] != 'not evaluable' else THEME.tok['muted']))
            p.drawText(QRectF(0, y, x0 - 8, rh), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, r['id'])
            p.setPen(QColor(THEME.tok['grid2']))
            p.drawLine(QPointF(x0, y + rh - 2), QPointF(x1, y + rh - 2))
            for a, b in r.get('unknown') or ():
                p.fillRect(QRectF(X(a), y + 4, max(X(b) - X(a), 1.0), rh - 8), QColor(THEME.tok['strong']))
            for v in r['violations']:
                p.fillRect(QRectF(X(v['start']), y + 2, max(X(v['end']) - X(v['start']), 2.0), rh - 4),
                           QColor(STATUS['critical']))
        yb = 4 + rh * len(res['rules'])
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QRectF(x0, yb + 2, 80, 14), Qt.AlignmentFlag.AlignLeft, '0 s')
        p.drawText(QRectF(x1 - 80, yb + 2, 80, 14), Qt.AlignmentFlag.AlignRight, f'{T:.1f} s')
        p.drawText(QRectF(x0 + 80, yb + 2, x1 - x0 - 160, 14), Qt.AlignmentFlag.AlignCenter,
                   'red: violated · grey: unknown (not recorded) · click to go there')
        if self.t is not None:
            p.setPen(QPen(QColor(THEME.tok['ink']), 1.5))
            p.drawLine(QPointF(X(self.t), 2), QPointF(X(self.t), yb))


class RulesPanel(QWidget):
    """The traffic-rule check of the run being browsed (core.rules, in the background): the rule set (by the run's
    handedness, India, generic or a file), the timeline, each rule's state, robustness, violations and counts per km;
    Save writes it to the workspace's rules/ folder for a report (Reports ▸ Traffic-rule check)."""

    def __init__(self, browse):
        super().__init__()
        self.browse = browse
        self.result = None
        self.path = ''
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.set_box = QComboBox()
        self.set_box.addItem('By the run\'s traffic side', '')
        self.set_box.addItem('India (left-hand)', 'india')
        self.set_box.addItem('Generic (right-hand)', 'generic')
        self.set_box.addItem('A rule file…', 'file')
        self.set_box.currentIndexChanged.connect(lambda _: self.check())
        row.addWidget(label('Rules', muted=True))
        row.addWidget(self.set_box)
        self.status = label('', muted=True)
        row.addWidget(self.status, 1)
        self.save_btn = button('Save for a report', self.save)
        row.addWidget(self.save_btn)
        v.addLayout(row)
        self.timeline = RuleTimeline()
        self.timeline.seek.connect(self.browse.seek_time)
        v.addWidget(self.timeline)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Rule', 'State', 'Robustness', 'Violations', 'Per km', 'Evaluated', 'First at'])
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(120)
        self.tree.itemDoubleClicked.connect(self._go)
        v.addWidget(self.tree)
        self.rule_file = ''

    def show_run(self, path):
        self.path = path or ''
        self.result = None
        self.tree.clear()
        self.timeline.set_result(None)
        if self.path:
            self.check()

    def check(self):
        from ...core import rules as RU
        if not self.path:
            return
        spec = self.set_box.currentData()
        if spec == 'file':
            from PyQt6.QtWidgets import QFileDialog
            f, _ = QFileDialog.getOpenFileName(self, 'A rule file (falcon.rules/1)', '',
                                               'Rules (*.rules *.txt);;All (*)')
            if not f:
                self.set_box.setCurrentIndex(0)
                return
            self.rule_file = f
            spec = f
        path = self.path
        self.status.setText('checking…')

        def work():
            return path, RU.check_run(path, spec or None)

        def fail(e):
            if path == self.path:
                self.status.setText(f'not checked: {e}')
        tasks.run(self, work, self._done, fail)

    def _done(self, res):
        path, r = res
        if path != self.path:
            return
        self.result = r
        self.timeline.set_result(r)
        self.tree.clear()
        from ..theme import STATUS
        for x in r['rules']:
            rob = '∞' if x['vacuous'] else ('—' if x['robustness'] is None else f'{x["robustness"]:.3f}')
            it = QTreeWidgetItem([x['title'], x['state'], rob, str(x['count']),
                                  '—' if x['per_km'] is None else f'{x["per_km"]:.2f}',
                                  f'{100 * (x["evaluated_share"] or 0):.0f} %',
                                  f'{x["violations"][0]["start"]:.2f} s' if x['violations'] else ''])
            it.setData(0, 256, x['violations'][0]['start'] if x['violations'] else None)
            it.setToolTip(0, f'{x["id"]}: {x["formula"]}' + (f'\nunknown: {", ".join(x["missing_signals"])}'
                                                              if x['missing_signals'] else ''))
            sev = {'violated': 'critical', 'satisfied': 'good'}.get(x['state'])
            if sev and (x['state'] == 'violated' or (x['evaluated_share'] or 0) >= 0.5):
                it.setForeground(1, QColor(STATUS[sev]))       # satisfied on under half the run: not coloured good
            self.tree.addTopLevelItem(it)
        for c in range(7):
            self.tree.resizeColumnToContents(c)
        rs = r['ruleset']
        warn = ' ⚠ ' + ' '.join(r['warnings']) if r.get('warnings') else ''
        self.status.setText(f'{rs.get("name")}: {r["totals"]["violations"]} violation(s) in {r["duration_s"]} s, '
                            f'{r["distance_km"]} km{warn}')

    def _go(self, it, _c):
        t = it.data(0, 256)
        if t is not None:
            self.browse.seek_time(float(t))

    def save(self):
        from ...core.provenance import block, write_json
        if not self.result:
            return
        ws = self.browse.win.workspace()
        name = f'{os.path.basename(self.path.rstrip("/"))}-{self.result["ruleset"]["id"]}'
        d = os.path.join(ws, 'rules', name)
        os.makedirs(d, exist_ok=True)
        write_json(os.path.join(d, 'rules.json'), {
            'schema': 'falcon.rules_results/1', 'results': [self.result], 'aggregates': {},
            'provenance': block({'run': os.path.join(self.path, 'frames.jsonl')},
                                settings={'ruleset': self.result['ruleset']})})
        self.status.setText(f'saved to {tilde(d)} (Reports ▸ Traffic-rule check)')


BROWSABLE = (None, 'drive', 'translate', 'replay')            # workers whose runs have frames to browse
RUN_COLUMNS = ['Run', 'Date', 'World', 'Weather', 'Duration', 'Status', 'Tags']
SEARCH_HELP = ('Search the runs (docs/runs.md). Terms are ANDed:\n'
               '  world:w-blr  map:Town10  weather:rain (a preset or clear/cloudy/wet/rain)  tod:night\n'
               '  status:ok  kind:scenario|search|route|drive  scenario:lead  rig:…  tag:mine\n'
               '  events:collision|teleport|odd_exit|rule_violation|failed_check\n'
               '  dur>20  frames>=100  speed>50 (max km/h)  ttc<1.5  collisions>0  seed=3\n'
               '  after:2026-10-28  before:2026-11-01   -term negates   "quoted text"   other words: free text')


class RunItem(QTreeWidgetItem):
    """A run's row: sorts dates and durations as values, not text."""

    def __init__(self, e):
        dur = e.get('duration_s')
        super().__init__([e['name'], (e.get('started') or '')[:16].replace('T', ' '), e.get('world') or e.get('map')
                          or '—', e.get('weather') or '—', f'{dur:.1f} s' if dur is not None else '—',
                          e.get('status') or '?', ', '.join(e.get('tags') or ())])
        self.entry = e
        self.setData(0, 256, e['path'])
        self.setToolTip(0, f'{e["name"]}\n{tilde(e["path"])}\n{(e.get("started") or "").replace("T", " ").rstrip("Z")}'
                           + (f'\nevents: {", ".join(e["events"])}' if e.get('events') else ''))

    def __lt__(self, other):
        c = self.treeWidget().sortColumn() if self.treeWidget() else 0
        key = {1: 'started', 4: 'duration_s'}.get(c)
        if key is not None:
            a, b = self.entry.get(key), other.entry.get(key)
            if a is None or b is None:
                return a is None and b is not None
            return a < b
        return self.text(c).lower() < other.text(c).lower()


class RecordTab(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        v = QVBoxLayout(self)
        sec = Section('Record a drive', 'Synchronous mode at a fixed step; the ego and the traffic on CARLA\'s traffic '
                                        'manager; every camera\'s label cameras at its exact pose and tick. Seeded: '
                                        'the same seed on the same server reproduces the drive (falcon reproduce).')
        f = Form()
        self.world = QComboBox()
        self.rig = QComboBox()
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0.5, 3600)
        self.seconds.setValue(10)
        self.seconds.setSuffix(' s')
        self.fps = QDoubleSpinBox()
        self.fps.setRange(1, 100)
        self.fps.setValue(20)
        self.fps.setSuffix(' Hz')
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.weather = QComboBox()
        self.weather.addItem('The world\'s weather', '')
        from ...core import weather as W
        for name in W.PRESETS:
            self.weather.addItem(name, name)
        f.row('World', self.world)
        f.row('Weather', self.weather, 'a CARLA preset for this drive only; the world keeps its own')
        f.row('Rig', self.rig)
        f.row('Duration', self.seconds, 'simulated time')
        f.row('Step', self.fps, 'fixed_delta_seconds = 1 / rate')
        f.row('Seed', self.seed, 'spawn choices, traffic manager, pedestrians')
        sec.add(f)
        self.note = label('', muted=True, wrap=True)
        sec.add(self.note)
        self.cost = KeyValues(('Rendering', 'Budget', 'Estimate'))
        sec.add(self.cost)
        b = QHBoxLayout()
        self.go_btn = button('Record', self.record, role='primary')
        b.addWidget(self.go_btn)
        b.addStretch(1)
        sec.add(b)
        v.addWidget(sec)
        v.addStretch(1)
        for w in (self.world, self.rig):
            w.currentIndexChanged.connect(self.update_cost)
        for w in (self.seconds, self.fps):
            w.valueChanged.connect(self.update_cost)

    def refresh(self):
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
        self.update_cost()

    def update_cost(self):
        rig = self.win.project.item('rigs', self.rig.currentData() or '')
        if rig is None:
            return
        mpx = budget.rig_mpx(rig)
        lim = float(self.win.settings.get('render_budget_mpx'))
        ok, why = budget.check_budget(rig, lim)
        steps = int(round(self.seconds.value() * self.fps.value()))
        t = self.win.service.throughput()
        est = t.estimate(steps, mpx)
        self.cost.set('Rendering', f'{mpx:.2f} Mpx per step ({len(rig["sensors"])} sensors, images and label '
                                   'cameras)')
        self.cost.set('Budget', ('' if ok else '⚠ ') + why)
        self.cost.set('Estimate', f'{steps} steps: about {budget.fmt_duration(est["seconds"])} '
                                  f'({est["per_step_s"] * 1000:.0f} ms per step, {est["label"]})', t.source)
        set_enabled(self.go_btn, ok, f'Over the render budget: {why}')

    def record(self):
        w = self.win
        try:
            out = os.path.join(w.service.runs_dir(), run_dir_name(f'drive-{self.world.currentText()}'))
            cfg = drive_config(w.project, self.world.currentData(), self.rig.currentData(), out, w.service.port,
                               w.service.tm_port, self.seconds.value(), self.fps.value(), self.seed.value(),
                               render_budget_mpx=float(w.settings.get('render_budget_mpx')))
            preset = self.weather.currentData()
            if preset:                        # this drive's weather (Health's "record more of"), the world unchanged
                from ...core import weather as W
                cfg['weather'] = W.resolve({'preset': preset})
                cfg['weather_source'] = {'preset': preset, 'override': 'Data ▸ Record'}
            from ...core.triggers import attach           # the project's trigger rules for drives
            attach(cfg, w.project, 'drives')
            w.service.submit_drive(cfg)
        except (ProjectError, RuntimeError) as e:
            QMessageBox.warning(self, 'Record a drive', str(e))
            return
        w.jobs_dock.raise_()


class BrowseTab(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.run_dir = ''
        self.frames: list = []
        self.cache = None                     # (index, sensor, rgb, label colours, tags)
        h = QHBoxLayout(self)
        split = QSplitter()
        h.addWidget(split)
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText('Search: world:w-blr weather:rain status:ok dur>20 tag:night …')
        self.filter.setToolTip(SEARCH_HELP)
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(lambda _t: self.apply_filter())
        sv.addWidget(self.filter)
        self.query_state = label('', muted=True, small=True)
        sv.addWidget(self.query_state)
        self.runs = PlaceholderTree('No recorded runs yet: Record makes one.')
        self.runs.setHeaderLabels(RUN_COLUMNS)
        self.runs.setRootIsDecorated(False)
        self.runs.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.runs.setSortingEnabled(True)
        self.runs.sortByColumn(1, Qt.SortOrder.DescendingOrder)
        self.runs.header().setStretchLastSection(False)
        self.runs.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.runs.setColumnWidth(0, 170)
        for c in range(1, len(RUN_COLUMNS)):
            self.runs.header().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.runs.setMinimumWidth(420)         # name, date and world without scrolling; the rest scrolls
        self.runs.currentItemChanged.connect(lambda it, _p: self.show_run(it.data(0, 256)) if it else None)
        self.runs.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.runs.customContextMenuRequested.connect(self._menu)
        sv.addWidget(self.runs, 1)
        hb = QHBoxLayout()                    # hand-offs: use the selected drives elsewhere
        self.eval_btn = button('Evaluate on these', self.evaluate_selected,
                               tip='Open Models ▸ Evaluate with the selected drives checked (Ctrl/Shift-click to '
                                   'select several)')
        self.train_btn = button('Train on these', self.train_selected,
                                tip='Open Training ▸ Detectors with the selected drives checked')
        self.compare_btn = button('Compare', self.compare_selected,
                                  tip='Open Data ▸ Compare with the selected runs (two to four) side by side')
        hb.addWidget(self.eval_btn)
        hb.addWidget(self.train_btn)
        hb.addWidget(self.compare_btn)
        hb.addStretch(1)
        sv.addLayout(hb)
        sb = QHBoxLayout()                    # the user's own organisation of runs: tags and drive sets
        sb.addWidget(button('Tags…', self.edit_tags, tip='Tag the selected runs (kept in the project; the runs are '
                                                         'not changed)'))
        sb.addWidget(button('Save as a set…', self.save_selection_set,
                            tip='Save the selected runs as a named drive set (Training, Models and Health use it)'))
        self.save_query_btn = button('Save search…', self.save_query_set,
                                     tip='Save the search as a drive set that is re-evaluated when used')
        sb.addWidget(self.save_query_btn)
        sb.addStretch(1)
        sv.addLayout(sb)
        split.addWidget(side)
        self.empty = EmptyState('Choose a run on the left to see its frames, labels, objects and traffic rules.')
        right = QWidget()
        rv = QVBoxLayout(right)
        top = QHBoxLayout()
        self.sensor = QComboBox()
        self.layer = QComboBox()
        self.layer.addItems(['semantic', 'instance', 'range', 'boxes', 'none'])
        self.layer.currentIndexChanged.connect(lambda _: self.load_frame(self.slider.value()))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.setFixedWidth(140)
        top.addWidget(label('Camera', muted=True))
        top.addWidget(self.sensor)
        top.addWidget(label('Layer', muted=True))
        top.addWidget(self.layer)
        top.addWidget(label('Opacity', muted=True))
        top.addWidget(self.opacity)
        top.addStretch(1)
        rv.addLayout(top)
        self.open_btn = button('Open folder', self.open_folder)
        self.export_fmt = QComboBox()
        self.export_fmt.addItems(['coco', 'yolo', 'kitti', 'nuscenes', 'waymo', 'bdd_lanes', 'mcap'])
        for v in ('occ3d', 'openocc', 'surroundocc'):          # semantic occupancy (core.occexport): from the lidar
            self.export_fmt.addItem(v, ('occ3d', v))
        self.export_fmt.setToolTip('waymo: a documented Waymo-style JSON layout, not TFRecords; occ3d, openocc, '
                                   'surroundocc: semantic occupancy from a run with a labelled lidar')
        xb = QHBoxLayout()                    # the chosen run's files: under the run list, with the hand-offs
        xb.addWidget(self.export_fmt)
        xb.addWidget(button('Export…', self.export))
        xb.addWidget(self.open_btn)
        xb.addStretch(1)
        sv.addLayout(xb)
        self.view = ImageView()
        rv.addWidget(self.view, 1)
        fr = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_lbl = label('', mono=True)
        fr.addWidget(self.slider, 1)
        fr.addWidget(self.frame_lbl)
        rv.addLayout(fr)
        lower = QHBoxLayout()
        self.legend = QWidget()
        self.legend_grid = QGridLayout(self.legend)
        self.legend_grid.setContentsMargins(0, 0, 0, 0)
        leg = Section('Classes in this frame')
        leg.add(self.legend)
        lower.addWidget(leg, 1)
        info = Section('Run')
        self.info = KeyValues()
        info.add(self.info)
        lower.addWidget(info, 1)
        objs = Section('Objects in this frame')
        self.objects = QTreeWidget()
        self.objects.setHeaderLabels(['Track', 'Class', 'Box', 'Occluded', 'Truncated', 'Range'])
        self.objects.setRootIsDecorated(False)
        self.objects.setMinimumHeight(140)
        objs.add(self.objects)
        below = QWidget()                     # the frame's details and the run's rules, under a splitter
        bv = QVBoxLayout(below)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addWidget(objs)
        bv.addLayout(lower)
        rules = Section('Traffic rules', 'Each rule as a temporal-logic formula over what the run recorded '
                                         '(docs/traffic-rules.md): red where it was violated, grey where a fact it '
                                         'needs was not recorded.')
        self.rules = RulesPanel(self)
        rules.add(self.rules)
        bv.addWidget(rules)
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(right)
        vsplit.addWidget(scroll(below))
        vsplit.setSizes([620, 520])
        self.body = QStackedWidget()          # the empty state until a run is chosen
        self.body.addWidget(self.empty)
        self.body.addWidget(vsplit)
        split.addWidget(self.body)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 900])
        self.slider.valueChanged.connect(self.load_frame)
        self.sensor.currentIndexChanged.connect(lambda _: self.load_frame(self.slider.value()))
        self.opacity.valueChanged.connect(lambda _: self.compose())

    def refresh(self):
        from ..drivesets import index_entries
        cur = self.run_dir
        self.entries = [e for e in index_entries(self.win) if e.get('worker') in BROWSABLE]
        self.runs.blockSignals(True)
        self.runs.setSortingEnabled(False)
        self.runs.clear()
        for e in self.entries:
            it = RunItem(e)
            it.setIcon(5, icons.status_icon({'ok': 'good', 'cancelled': 'warning'}.get(e['status'], 'critical')))
            self.runs.addTopLevelItem(it)
            if e['path'] == cur:
                self.runs.setCurrentItem(it)
        self.runs.setSortingEnabled(True)
        self.runs.blockSignals(False)
        self.apply_filter()
        self.empty.set_message('Choose a run on the left to see its frames, labels, objects and traffic rules.'
                               if self.runs.topLevelItemCount() else
                               'No recorded runs yet. Record a drive (the Record tab), or run a scenario, and it '
                               'appears here.')

    def apply_filter(self):
        from ...core import runindex as RI
        q = self.filter.text().strip()
        try:
            terms = RI.parse(q)
        except RI.QueryError as e:
            self.query_state.setText(f'⚠ {e}')
            set_enabled(self.save_query_btn, False, f'The search is not valid: {e}')
            return
        shown = 0
        for i in range(self.runs.topLevelItemCount()):
            it = self.runs.topLevelItem(i)
            ok = RI.match(it.entry, terms)
            it.setHidden(not ok)
            shown += ok
        n = self.runs.topLevelItemCount()
        self.query_state.setText(f'{shown} of {n} runs' if q else (f'{n} runs' if n else ''))
        set_enabled(self.save_query_btn, bool(q), 'Type a search first')

    def visible_runs(self) -> list:
        return [self.runs.topLevelItem(i).data(0, 256) for i in range(self.runs.topLevelItemCount())
                if not self.runs.topLevelItem(i).isHidden()]

    def _menu(self, pos):
        from PyQt6.QtWidgets import QMenu
        if not self.selected_runs():
            return
        m = QMenu(self)
        m.addAction('Tags…', self.edit_tags)
        m.addAction('Save as a drive set…', self.save_selection_set)
        m.addSeparator()
        m.addAction('Compare', self.compare_selected)
        m.addAction('Evaluate on these', self.evaluate_selected)
        m.addAction('Train on these', self.train_selected)
        m.exec(self.runs.viewport().mapToGlobal(pos))

    def edit_tags(self):
        from PyQt6.QtWidgets import QInputDialog
        from ..drivesets import set_tags
        runs = self.selected_runs()
        if not runs:
            return
        have = sorted({t for e in self.entries if e['path'] in runs for t in e.get('tags') or ()})
        text, ok = QInputDialog.getText(self, 'Tags', f'Tags for {len(runs)} run(s), separated by commas (empty: '
                                                      'none):', text=', '.join(have))
        if ok:
            set_tags(self.win, runs, text.split(','))

    def save_selection_set(self):
        from PyQt6.QtWidgets import QInputDialog
        from ..drivesets import add_set
        runs = self.selected_runs()
        if not runs:
            return
        name, ok = QInputDialog.getText(self, 'Save a drive set', f'A name for these {len(runs)} runs:')
        if ok and name.strip():
            add_set(self.win, name.strip(), 'list', runs)
            self.query_state.setText(f'saved the drive set {name.strip()!r} ({len(runs)} runs)')

    def save_query_set(self):
        from PyQt6.QtWidgets import QInputDialog
        from ..drivesets import add_set
        q = self.filter.text().strip()
        if not q:
            return
        name, ok = QInputDialog.getText(self, 'Save a search', f'A name for the search “{q}” (re-evaluated when '
                                                               'used):')
        if ok and name.strip():
            add_set(self.win, name.strip(), 'query', query=q)
            self.query_state.setText(f'saved the search as {name.strip()!r}')

    def compare_selected(self):
        self.page.compare_runs(self.selected_runs())

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self.run_dir and self.runs.topLevelItemCount():      # the newest run, rather than a blank viewer
            self.runs.setCurrentItem(self.runs.topLevelItem(0))

    def selected_runs(self) -> list:
        sel = [it.data(0, 256) for it in self.runs.selectedItems()]
        return sel or ([self.run_dir] if self.run_dir else [])

    def evaluate_selected(self):
        models = self.win.go('models')
        if models is not None:
            models.refresh()
            models.check_runs(self.selected_runs())

    def train_selected(self):
        tr = self.win.go('training')
        if tr is not None:
            tr.refresh()
            tr.check_runs(self.selected_runs())

    def seek_step(self, step: int):
        """Go to the frame recorded at this simulation step (Models ▸ Failures opens a failing frame here)."""
        k = next((i for i, f in enumerate(self.frames) if f.get('step') == step), None)
        if k is not None:
            self.slider.setValue(k)

    def seek_time(self, t: float):
        """Go to the frame nearest t seconds from the run's first."""
        if not self.frames:
            return
        t0 = float(self.frames[0].get('sim_t') or 0.0)
        ts = np.array([float(f.get('sim_t') or 0.0) - t0 for f in self.frames])
        self.slider.setValue(int(np.argmin(np.abs(ts - t))))

    def show_run(self, path):
        self.run_dir = path or ''
        self.body.setCurrentIndex(1 if self.run_dir else 0)
        self.frames, self.cache = [], None
        self.info.clear()
        self.rules.show_run(path)
        self.sensor.blockSignals(True)
        self.sensor.clear()
        self.sensor.blockSignals(False)
        if not path:
            return
        try:
            with open(os.path.join(path, 'run.json'), encoding='utf-8') as fh:
                run = json.load(fh)
        except (OSError, ValueError):
            run = {}
        perf = run.get('performance') or {}
        prov = run.get('provenance') or {}
        self.info.set('Status', f'{run.get("status", "?")}' + (f' — {run["error"]}' if run.get('error') else ''))
        self.info.set('Folder', tilde(path))
        if perf.get('steps'):
            parts = [f'{perf["steps"]} steps in {perf.get("wall_s", 0):.1f} s']
            if 'realtime_factor' in perf:
                parts.append(f'{perf["realtime_factor"]:.2f}× real time')
            if 'mpx_per_s' in perf:
                parts.append(f'{perf["mpx_per_s"]:.0f} Mpx/s')
            self.info.set('Measured', ', '.join(parts))
        host = prov.get('host') or {}
        self.info.set('Ran with', f'seed {prov.get("seed")}, CARLA {host.get("carla_server", "?")}, Falcon '
                                  f'{prov.get("app_version", "?")}')
        if run.get('config_sha256'):
            self.info.set('Config', run['config_sha256'][:16] + '…', 'SHA-256 of drive.json (canonical JSON)')
        for n in run.get('notes') or []:
            if not n.startswith('semantic labels:'):
                self.info.set('Note', n)
        try:
            with open(os.path.join(path, 'dataset.json'), encoding='utf-8') as fh:
                for wtext in json.load(fh).get('label_warnings') or []:
                    self.info.set('⚠ Labels', wtext)
        except (OSError, ValueError):
            pass
        idx = os.path.join(path, 'frames.jsonl')
        if not os.path.isfile(idx):
            self.view.set_rgb(None)
            self.slider.setRange(0, 0)
            self.frame_lbl.setText('no frames (only run.json)')
            return
        with open(idx, encoding='utf-8') as fh:
            self.frames = [json.loads(x) for x in fh if x.strip()]
        sensors = sorted({k.split('/')[0] for f in self.frames[:1] for k in f['files']})
        self.sensor.blockSignals(True)
        self.sensor.addItems(sensors)
        self.sensor.blockSignals(False)
        self.slider.setRange(0, max(len(self.frames) - 1, 0))
        self.slider.setValue(0)
        self.load_frame(0)

    def load_frame(self, i):
        if not self.frames or not 0 <= i < len(self.frames):
            return
        f = self.frames[i]
        s = self.sensor.currentText()
        self.rules.timeline.set_time(float(f.get('sim_t') or 0.0) - float(self.frames[0].get('sim_t') or 0.0))
        self.frame_lbl.setText(f'step {f["step"]}  frame {f["frame"]}  t {f["sim_t"]:.2f} s  '
                               f'{f.get("speed_mps", 0) * 3.6:.0f} km/h')
        files = {k.split('/')[1]: os.path.join(self.run_dir, v) for k, v in f['files'].items()
                 if k.split('/')[0] == s}

        layer = self.layer.currentText()

        def load():
            import cv2
            rgb = cv2.imread(files['rgb'])[:, :, ::-1] if 'rgb' in files else None
            tags = cv2.imread(files['semantic'], cv2.IMREAD_UNCHANGED) if 'semantic' in files else None
            col = None
            if layer == 'semantic' and tags is not None:
                col = labels.colorize(np.minimum(tags, 28))
            elif layer == 'instance' and 'instance' in files:
                inst = cv2.imread(files['instance'], cv2.IMREAD_UNCHANGED).astype(np.int64)
                rnd = np.random.default_rng(7).integers(40, 255, (65536, 3), dtype=np.uint8)
                rnd[0] = 0
                col = rnd[inst]
            elif layer == 'range' and 'range' in files:
                r = cv2.imread(files['range'], cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
                v = np.clip(np.log1p(r) / np.log1p(100.0), 0, 1)
                col = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
                col[r <= 0] = 0
            objs = []
            if 'objects' in files:
                with open(files['objects'], encoding='utf-8') as fh:
                    objs = json.load(fh).get('objects', [])
            if layer == 'boxes' and rgb is not None:
                col = rgb.copy()
                for o in objs:
                    x0, y0, x1, y1 = (int(round(v)) for v in o['bbox'])
                    cv2.rectangle(col, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 90), 2)
                    if o.get('bbox_full'):
                        a0, b0, a1, b1 = (int(round(v)) for v in o['bbox_full'])
                        cv2.rectangle(col, (a0, b0), (a1, b1), (255, 150, 0), 1)
            return i, s, rgb, tags, col, objs
        tasks.run(self, load, self._loaded)

    def _loaded(self, res):
        i, s, rgb, tags, col, objs = res
        if i != self.slider.value() or s != self.sensor.currentText():
            return                                 # a newer frame was asked for meanwhile
        self.cache = (rgb, col, tags)
        self.compose()
        self.show_legend(tags)
        self.objects.clear()
        for o in objs:
            def f(v):
                return '—' if v is None else f'{v:.2f}'
            it = QTreeWidgetItem([str(o.get('track_id')), o.get('class', ''),
                                  ' '.join(f'{v:.0f}' for v in o['bbox']), f(o.get('occluded')),
                                  f(o.get('truncated')), f'{o.get("range_m", 0):.1f} m'])
            self.objects.addTopLevelItem(it)

    def compose(self):
        if not self.cache or self.cache[0] is None:
            return
        rgb, col, _ = self.cache
        a = self.opacity.value() / 100.0
        if self.layer.currentText() == 'boxes' and col is not None:
            a = 1.0                                       # boxes are drawn on the image itself
        out = rgb if col is None or a <= 0 else (rgb * (1 - a) + col * a).astype(np.uint8)
        self.view.set_rgb(out)

    def show_legend(self, tags):
        while self.legend_grid.count():
            it = self.legend_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if tags is None:
            return
        tags = np.minimum(tags, 28)
        vals, counts = np.unique(tags, return_counts=True)
        order = np.argsort(-counts)
        total = float(tags.size)
        for r, k in enumerate(order[:24]):
            t = int(vals[k])
            sw = QLabel()
            sw.setFixedSize(12, 12)
            c = labels.color(t)
            sw.setStyleSheet(f'background: {QColor(*c).name()}; border-radius: 2px;')
            self.legend_grid.addWidget(sw, r // 2, (r % 2) * 3)
            self.legend_grid.addWidget(label(labels.name(t)), r // 2, (r % 2) * 3 + 1)
            self.legend_grid.addWidget(label(f'{100 * counts[k] / total:.1f} %', muted=True, mono=True), r // 2,
                                       (r % 2) * 3 + 2)

    def export(self):
        """Export the run in the chosen format: a background job running `falcon export` (the CLI's code)."""
        import sys
        from PyQt6.QtWidgets import QFileDialog
        from ...core.jobs import JobSpec
        if not self.run_dir:
            return
        fmt = self.export_fmt.currentText()
        out = QFileDialog.getExistingDirectory(self, f'Folder for the {fmt} export')
        if not out:
            return
        variant = self.export_fmt.currentData()
        argv = [sys.executable, '-m', 'falcon_autonomy', 'export', self.run_dir, '--format',
                variant[0] if variant else fmt, '--out', out] + (['--option', f'variant={variant[1]}'] if variant else [])
        self.win.queue.submit(JobSpec('export', f'Export {os.path.basename(self.run_dir)} as {fmt}', argv,
                                      memory_gb=1.0))
        self.win.jobs_dock.raise_()

    def open_folder(self):
        if self.run_dir:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.run_dir))


class DataPage(Page):
    key, title, icon = 'data', 'Data', 'dataset'
    about = 'Record drives with perfectly labelled frames, and browse what was recorded.'

    def __init__(self, win):
        super().__init__(win)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.addWidget(label(self.title, page=True))
        v.addWidget(Note(self.about, limit=160))
        self.tabs = QTabWidget()
        from .data_health import HealthTab
        from .datasets import DatasetsTab
        from .runs_compare import CompareTab
        self.record = RecordTab(self)
        self.browse = BrowseTab(self)
        self.datasets = DatasetsTab(self)
        self.compare = CompareTab(self)
        self.health = HealthTab(self)
        self.tabs.addTab(scroll(self.record), 'Record')
        self.tabs.addTab(self.browse, 'Browse')
        self.tabs.addTab(self.compare, 'Compare')
        self.tabs.addTab(scroll(self.health), 'Health')
        self.tabs.addTab(scroll(self.datasets), 'Datasets')
        v.addWidget(self.tabs, 1)
        self.refresh()

    def refresh(self):
        self.record.refresh()
        self.browse.refresh()
        self.datasets.refresh()
        self.health.refresh()                 # (Compare reads the run index when it is shown)

    def record_in(self, world_id, preset: str | None = None):
        """Open Record with this world (and weather preset) chosen (Worlds' *Record a dataset here*, Health's
        *Record this*)."""
        self.tabs.setCurrentIndex(0)
        self.record.refresh()
        k = self.record.world.findData(world_id)
        if k >= 0:
            self.record.world.setCurrentIndex(k)
        k = self.record.weather.findData(preset or '')
        self.record.weather.setCurrentIndex(k if k >= 0 else 0)
        self.record.note.setText('')

    def show_drive_set(self, ident) -> bool:
        """Open Browse on a drive set (the command palette's drive sets): a saved query becomes the search; a list's
        runs are selected."""
        ds = self.win.project.item('drive_sets', ident)
        if ds is None:
            return False
        self.tabs.setCurrentWidget(self.browse)
        self.browse.refresh()
        if ds.get('kind') == 'query':
            self.browse.filter.setText(ds.get('query') or '')
            return True
        from ..drivesets import resolve
        want = set(resolve(self.win, ds, self.browse.entries))
        self.browse.filter.setText('')
        self.browse.runs.clearSelection()
        first = None
        for i in range(self.browse.runs.topLevelItemCount()):
            it = self.browse.runs.topLevelItem(i)
            if it.data(0, 256) in want:
                it.setSelected(True)
                first = first or it
        if first is not None:
            self.browse.runs.scrollToItem(first)
        self.browse.query_state.setText(f'drive set {ds["name"]!r}: {len(want)} run(s) selected')
        return True

    select_drive_set = show_drive_set

    def compare_runs(self, paths):
        """Open Compare with these runs (Browse's *Compare*)."""
        self.tabs.setCurrentWidget(self.compare)
        self.compare.refresh()
        self.compare.set_runs(paths)

    def show_run(self, path):
        self.tabs.setCurrentWidget(self.browse)
        self.browse.refresh()
        self.browse.show_run(path)

    def on_run_finished(self, job):
        self.browse.refresh()
        self.record.update_cost()
        self.datasets.on_run_finished(job)
