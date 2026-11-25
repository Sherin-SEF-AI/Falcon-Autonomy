"""Replay: import a real log (nuScenes, Waymo v2, a ROS 2 bag, KITTI raw, a Falcon run), see where it goes in a world
and how well its road matches, see its trajectories top-down over time, edit what-if changes, replay the original and
the changed log in CARLA as jobs, and compare them (core.logreplay; docs/log-replay.md); and reconstruct the log's own
appearance to look at it from poses it never took (recon_panel; core.recon; docs/reconstruction.md). Side by side
(ui.replay_sync, core.logreplay.sync): any two runs on one time slider with their divergences marked —
compare_runs(run_a, run_b) is the way in for other pages (Release gate, Send to)."""
from __future__ import annotations

import json
import os
import re

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLineEdit, QSlider,
                             QSpinBox, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core.jobs import FINAL
from ..theme import STATUS, THEME
from ..runfid import FidelitySlot, fidelity_of, fit_rows, likeness, set_run_context
from ..widgets import Form, KeyValues, Page, Section, button, label, optional_tab, scroll

KINDS = (('', 'Detect from the path'), ('nuscenes', 'nuScenes (JSON tables)'), ('waymo', 'Waymo Open Dataset v2'),
         ('ros2', 'ROS 2 bag (MCAP or .db3)'), ('kitti_raw', 'KITTI raw (OXTS)'), ('falcon', 'Falcon run'))
OPS = (('add_pedestrian', 'Add a pedestrian crossing the ego\'s path'), ('time_shift', 'Shift an actor in time'),
       ('speed_scale', 'Scale an actor\'s speed'), ('remove', 'Remove an actor'), ('weather', 'Change the weather'))
CLASS_SLOT = {'car': 0, 'truck': 3, 'bus': 3, 'motorcycle': 4, 'bicycle': 2, 'pedestrian': 1}
SHOWN_METRICS = (('collision', 'Collision (0/1)'), ('min_ttc_s', 'Minimum TTC, s'),
                 ('min_distance_m', 'Minimum distance, m'), ('max_drac_mps2', 'Largest DRAC, m/s²'),
                 ('criticality', 'Criticality (heuristic)'), ('ego_path_m', 'Ego path, m'),
                 ('ego_min_speed_mps', 'Ego minimum speed, m/s'), ('ego_stopped_s', 'Ego stopped, s'))


class TopDown(QWidget):
    """The replay's trajectories seen from above (north up is not implied: CARLA's x right, y down on screen, as a
    CARLA map is drawn), the ego thick, every actor coloured by class, positions at the chosen step."""

    def __init__(self):
        super().__init__()
        self.plan = None
        self.i = 0
        self.setMinimumHeight(260)

    def set_plan(self, plan):
        self.plan = plan
        self.i = min(self.i, (plan or {}).get('steps', 1) - 1)
        self.update()

    def set_step(self, i):
        self.i = int(i)
        self.update()

    def _tracks(self):
        p = self.plan
        out = [('ego', 'ego', 0, p['ego']['x'], p['ego']['y'])]
        for a in p['actors']:
            out.append((a['id'], a['class'], a['i0'], a['x'], a['y']))
        return out

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), QColor(THEME.tok['panel']))
        if not self.plan:
            qp.setPen(QColor(THEME.tok['muted']))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Import or choose a log')
            return
        tracks = self._tracks()
        xs = [v for _, _, _, x, _ in tracks for v in x if v is not None]
        ys = [v for _, _, _, _, y in tracks for v in y if v is not None]
        if not xs:
            return
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        m = 24
        s = min((self.width() - 2 * m) / max(x1 - x0, 1.0), (self.height() - 2 * m - 18) / max(y1 - y0, 1.0))

        def P(x, y):
            return QPointF(m + (x - x0) * s, m + (y - y0) * s)
        added = {c.get('id') for c in self.plan.get('changes') or [] if c.get('op') == 'add_pedestrian'}
        for ident, cls, i0, x, y in tracks:
            col = QColor(STATUS['critical']) if ident in added else \
                (QColor(THEME.tok['ink']) if ident == 'ego' else THEME.cam(CLASS_SLOT.get(cls, 6)))
            col.setAlphaF(0.45)
            qp.setPen(QPen(col, 2.5 if ident == 'ego' else 1.2))
            prev = None
            for k in range(len(x)):
                if x[k] is None:
                    prev = None
                    continue
                q = P(x[k], y[k])
                if prev is not None:
                    qp.drawLine(prev, q)
                prev = q
            k = self.i - i0
            if 0 <= k < len(x) and x[k] is not None:
                col.setAlphaF(1.0)
                qp.setPen(Qt.PenStyle.NoPen)
                qp.setBrush(col)
                r = 5.0 if ident == 'ego' else 3.5
                qp.drawEllipse(P(x[k], y[k]), r, r)
        qp.setPen(QColor(THEME.tok['muted']))
        t = self.i * self.plan['dt']
        qp.drawText(QPointF(m, self.height() - 6),
                    f't = {t:.2f} s of {(self.plan["steps"] - 1) * self.plan["dt"]:.2f} s · {x1 - x0:.0f} × '
                    f'{y1 - y0:.0f} m · white: the ego; red: added; others by class · '
                    f'{self.plan["placement"]["mode"]} placement')


class ReplayPage(Page):
    key, title, icon = 'replay', 'Replay', 'history'
    about = ('Real logs replayed in CARLA: rebuild a scene, change it (timing, speeds, a pedestrian, the weather), and '
             'compare how the stack or a model behaves.')

    def __init__(self, win):
        super().__init__(win)
        self.log = None
        self.log_path = ''
        self.changes: list = []
        self.plan = None
        self.jobs = {}                    # 'original' | 'changed' -> (job, run dir)
        self.comparison = None
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)

        sec = Section('Log', 'Imported logs live in the workspace (logs/<name>/log.json): the trajectories only, never '
                             'the dataset\'s images or points. A dataset\'s log is Recorded; a Falcon run keeps its '
                             'own run\'s fidelity.')
        f = Form()
        self.kind = QComboBox()
        for k, text in KINDS:
            self.kind.addItem(text, k)
        self.path = QLineEdit()
        self.path.setPlaceholderText('a dataset folder, a bag folder, an .mcap or .db3, a run folder')
        pb = QHBoxLayout()
        pb.addWidget(self.path, 1)
        pb.addWidget(button('Folder…', lambda: self.browse(True)))
        pb.addWidget(button('File…', lambda: self.browse(False)))
        self.scene = QLineEdit()
        self.scene.setPlaceholderText('nuScenes scene / Waymo segment (default: the first)')
        self.town = QLineEdit()
        self.town.setPlaceholderText('ROS bags: the CARLA town of their map frame, when known')
        f.row('Format', self.kind)
        f.row('Source', pb)
        f.row('Scene', self.scene)
        f.row('Town', self.town)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Import', self.do_import))
        self.logs = QComboBox()
        self.logs.currentIndexChanged.connect(lambda _: self.load_log(self.logs.currentData()))
        b.addWidget(self.logs, 1)
        sec.add(b)
        self.info = KeyValues(('Log', 'Source', 'Duration', 'Actors', 'Map', 'Left out'))
        sec.add(self.info)
        fr = QWidget()
        frh = QHBoxLayout(fr)
        frh.setContentsMargins(0, 0, 0, 0)
        frh.addWidget(label('Fidelity', muted=True))
        self.log_fid = FidelitySlot('unknown')          # recorded (a dataset's log), else the source run's
        frh.addWidget(self.log_fid)
        frh.addStretch(1)
        self.log_fid_row = fr
        fr.hide()                                       # until a log is loaded
        sec.add(fr)
        lv.addWidget(sec)

        sec = Section('World and placement', 'A log recorded in a world replays on its own roads; a geo-located log in '
                                             'an imported OpenStreetMap world of its place; anything else on an '
                                             'anchor of the map, marked "no matching road". The replay measures the '
                                             'match again on the server.')
        f = Form()
        self.world = QComboBox()
        self.world.currentIndexChanged.connect(lambda _: self.replan())
        self.rig = QComboBox()
        self.driver = QComboBox()
        for k, text in (('log', 'Replays its log'), ('follower', 'Driven by the lane follower along its path')):
            self.driver.addItem(text, k)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.valueChanged.connect(lambda _: self.replan())
        self.fps = QDoubleSpinBox()
        self.fps.setRange(5.0, 100.0)
        self.fps.setValue(20.0)
        self.fps.valueChanged.connect(lambda _: self.replan())
        for text, w in (('World', self.world), ('Rig', self.rig), ('The ego', self.driver), ('Seed', self.seed),
                        ('Steps per second', self.fps)):
            f.row(text, w)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Check the map', self.check_map))
        b.addStretch(1)
        sec.add(b)
        self.match = KeyValues(('Placement', 'Why', 'Road match'))
        sec.add(self.match)
        lv.addWidget(sec)

        sec = Section('Changes', 'Applied in order to a copy of the log; seeded. A pedestrian crosses the ego\'s '
                                 'logged path at right angles where the ego was at the time given, reaching it then.')
        f = Form()
        self.op = QComboBox()
        for k, text in OPS:
            self.op.addItem(text, k)
        self.actor = QComboBox()
        self.value = QDoubleSpinBox()
        self.value.setRange(-600.0, 600.0)
        self.value.setDecimals(2)
        self.side = QComboBox()
        self.side.addItems(['right', 'left'])
        self.preset = QComboBox()
        from ...core.weather import PRESETS
        self.preset.addItems(list(PRESETS))
        for text, w in (('Change', self.op), ('Actor', self.actor), ('Value (s, factor or time)', self.value),
                        ('Pedestrian from', self.side), ('Weather preset', self.preset)):
            f.row(text, w)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Add', self.add_from_form))
        b.addWidget(button('Remove selected', self.remove_selected))
        b.addWidget(button('Load…', self.load_changes))
        b.addWidget(button('Save…', self.save_changes))
        b.addStretch(1)
        sec.add(b)
        self.change_list = QTreeWidget()
        self.change_list.setHeaderLabels(['#', 'Change'])
        self.change_list.setRootIsDecorated(False)
        self.change_list.setMinimumHeight(110)
        sec.add(self.change_list)
        lv.addWidget(sec)
        from .recon_panel import ReconPanel
        self.recon = ReconPanel(self)          # the log's own appearance, reconstructed (core.recon)
        lv.addWidget(self.recon)
        self.nearmiss = optional_tab(lv, 'ui.dashcam_ingest', 'NearMissTab', 'Near-misses', self)   # 4.0, in progress
        self.moments = optional_tab(lv, 'ui.moments_tab', 'MomentsTab', 'Moments', self)   # 4.1
        lv.addStretch(1)
        split.addWidget(scroll(left))

        self.right = right = QTabWidget()
        plan = QWidget()
        rv = QVBoxLayout(plan)
        rv.addWidget(label('Trajectories (the replay\'s, with the changes)', h=True))
        self.view = TopDown()
        rv.addWidget(self.view, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.view.set_step)
        rv.addWidget(self.slider)
        b = QHBoxLayout()
        b.addWidget(button('Replay the original', lambda: self.run('original')))
        b.addWidget(button('Replay with the changes', lambda: self.run('changed'), role='primary'))
        b.addWidget(button('Compare', self.do_compare,
                           tip='The original and the changed replay side by side on one slider, with the numbers'))
        b.addStretch(1)
        rv.addLayout(b)
        self.state = KeyValues(('Original', 'Changed', 'State'))
        rv.addWidget(self.state)
        right.addTab(plan, 'Plan')
        # ---- side by side: two runs on one slider, divergences marked (ui.replay_sync), and the numbers
        from ..replay_sync import SyncCompare
        cmp_ = QWidget()
        cv = QVBoxLayout(cmp_)
        self.sync = SyncCompare()
        self.sync.loaded.connect(self._sync_loaded)
        self._want_summary = False                      # fill the numbers from the sync when it is computed
        cv.addWidget(self.sync, 3)
        hr = QHBoxLayout()
        hr.addWidget(label('Numbers (paired by step; B − A)', h=True))
        self.cmp_fid = FidelitySlot('unknown')          # the comparison's fidelity: combine() of the two runs
        hr.addWidget(self.cmp_fid)
        hr.addStretch(1)
        cv.addLayout(hr)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(['Measure', 'A (original)', 'B (changed)', 'Difference'])
        self.table.setRootIsDecorated(False)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)   # sized to its rows
        hd = self.table.header()
        hd.setStretchLastSection(False)
        hd.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hd.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(c, 100 if c < 3 else 132)
            self.table.headerItem().setTextAlignment(c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cv.addWidget(self.table)
        self.note = label('', muted=True, wrap=True)
        cv.addWidget(self.note)
        right.addTab(scroll(cmp_), 'Side by side')
        split.addWidget(right)
        split.setSizes([460, 820])
        win.bridge.finished.connect(self.on_finished)
        from ..context import register_target
        register_target(win, 'replay', 'Replay', self.open_run,
                        'Replay ▸ the run imported as a log (its trajectories), to change and replay it')
        self.refresh()

    # ---------------------------------------------------------------- logs
    def logs_dir(self) -> str:
        return os.path.join(self.win.workspace(), 'logs')

    def refresh(self):
        doc = self.win.project.doc
        for combo, items in ((self.world, doc['worlds']), (self.rig, doc['rigs'])):
            c = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for it in items:
                combo.addItem(it['name'], it['id'])
            k = combo.findData(c)
            combo.setCurrentIndex(k if k >= 0 else 0)
            combo.blockSignals(False)
        cur = self.logs.currentData()
        self.logs.blockSignals(True)
        self.logs.clear()
        d = self.logs_dir()
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            p = os.path.join(d, name, 'log.json')
            if os.path.isfile(p):
                self.logs.addItem(name, p)
        k = self.logs.findData(cur or self.log_path)
        self.logs.setCurrentIndex(k if k >= 0 else (0 if self.logs.count() else -1))
        self.logs.blockSignals(False)
        if self.logs.currentData() and self.logs.currentData() != self.log_path:
            self.load_log(self.logs.currentData())
        else:
            self.replan()

    def browse(self, folder: bool):
        if folder:
            p = QFileDialog.getExistingDirectory(self, 'Log folder', self.path.text())
        else:
            p, _ = QFileDialog.getOpenFileName(self, 'Log file', self.path.text(), 'Bags (*.mcap *.db3);;All (*)')
        if p:
            self.path.setText(p)

    def do_import(self):
        from .. import tasks
        src = self.path.text().strip()
        if not src:
            self.info.set('Log', 'choose a source first')
            return
        kind, scene, town = self.kind.currentData() or None, self.scene.text().strip() or None, \
            self.town.text().strip() or None
        out_root = self.logs_dir()
        self.info.set('Log', 'importing…')

        def work():
            from ...core.logreplay import import_log
            log = import_log(kind, src, scene=scene, segment=scene, town=town)
            name = re.sub(r'[^A-Za-z0-9_.-]+', '-', log.meta.get('name') or 'log').strip('-').lower()[:60] or 'log'
            return log.save(os.path.join(out_root, name))
        tasks.run(self, work, self.imported, lambda e: self.info.set('Log', f'import failed: {e}'))

    def imported(self, path):
        self.log_path = ''
        self.refresh()
        k = self.logs.findData(path)
        if k >= 0:
            self.logs.setCurrentIndex(k)
        self.load_log(path)

    def load_log(self, path):
        from ...core.logreplay import Log, LogError
        if not path:
            return
        try:
            self.log = Log.load(path)
        except (OSError, ValueError, LogError) as e:
            self.info.set('Log', f'cannot read it: {e}')
            return
        self.log_path = path
        c = self.log.counts()
        src = self.log.meta.get('source') or {}
        self.info.set('Log', self.log.meta.get('name') or '')
        self.info.set('Source', f'{src.get("format")} · {src.get("scene") or src.get("segment") or ""}')
        self.info.set('Duration', f'{c["duration_s"]:.1f} s, ego path {c["ego_path_m"]:.0f} m')
        self.info.set('Actors', f'{c["actors"]}: ' + ', '.join(f'{n} {k}' for k, n in c['by_class'].items()))
        mp = self.log.map
        self.info.set('Map', f'{mp.get("kind")} {mp.get("name") or ""}' + (' (geo-located)' if mp.get('geo') else ''))
        self.info.set('Left out', '; '.join(x['what'] for x in self.log.meta.get('left_out') or []) or 'nothing',
                      tip='\n'.join(f'{x["what"]}: {x["why"]}' for x in self.log.meta.get('left_out') or []))
        self.actor.clear()
        self.actor.addItem('ego', 'ego')
        for a in self.log.actors:
            self.actor.addItem(f'{a.id} ({a.cls})', a.id)
        self.log_fid.set(log_fidelity(self.log.meta))
        self.log_fid_row.show()
        self.changes = []
        self.show_changes()
        self.replan()
        self.recon.set_log(self.log, path)

    # ---------------------------------------------------------------- placement and the preview
    def world_info(self):
        from ...core.logreplay import world_info
        return world_info(self.win.project, self.world.currentData(), self.win.workspace())

    def make_plan(self, changes=None):
        from ...core.logreplay import make_plan
        return make_plan(self.log, self.world_info(), fps=self.fps.value(),
                         changes=self.changes if changes is None else changes, seed=self.seed.value(),
                         driver=self.driver.currentData(), log_path=self.log_path)

    def replan(self):
        from ...core.logreplay import LogError
        if self.log is None or self.world.currentData() is None:
            self.view.set_plan(None)
            return
        try:
            self.plan = self.make_plan()
        except (LogError, ValueError, KeyError) as e:
            self.plan = None
            self.match.set('Placement', f'cannot plan: {e}')
            self.view.set_plan(None)
            return
        pl = self.plan['placement']
        self.match.set('Placement', pl['mode'])
        self.match.set('Why', pl['why'])
        self.view.set_plan(self.plan)
        self.slider.setRange(0, self.plan['steps'] - 1)

    def check_map(self):
        from .. import tasks
        if self.log is None:
            return
        log, world, ws = self.log, self.world_info(), self.win.workspace()
        self.match.set('Road match', 'measuring…')

        def work():
            from ...core.logreplay.commands import offline_match
            return offline_match(log, world, ws)

        def done(res):
            mm = res.get('match')
            self.match.set('Placement', res['placement']['mode'])
            self.match.set('Why', res['placement']['why'])
            if mm is None:
                self.match.set('Road match', res.get('match_note', ''))
            else:
                self.match.set('Road match', f'{mm["verdict"]}: {mm["why"]}', tip=mm.get('rule', ''))
        tasks.run(self, work, done, lambda e: self.match.set('Road match', f'failed: {e}'))

    # ---------------------------------------------------------------- changes
    def add_change(self, c: dict):
        from ...core.logreplay import LogError
        from ...core.logreplay.changes import validate
        try:
            validate(self.changes + [c])
            if self.log is not None:
                self.make_plan(self.changes + [c])
        except (LogError, ValueError, KeyError) as e:
            self.note.setText(f'change refused: {e}')
            return False
        self.changes.append(c)
        self.show_changes()
        self.replan()
        return True

    def add_from_form(self):
        op, v, actor = self.op.currentData(), self.value.value(), self.actor.currentData()
        k = sum(1 for c in self.changes if c['op'] == 'add_pedestrian')
        c = {'add_pedestrian': {'op': op, 'id': f'pedestrian_{k + 1}', 'at_ego_s': v, 'from': self.side.currentText()},
             'time_shift': {'op': op, 'actor': actor, 'dt_s': v},
             'speed_scale': {'op': op, 'actor': actor, 'factor': max(v, 0.0)},
             'remove': {'op': op, 'actor': actor},
             'weather': {'op': op, 'preset': self.preset.currentText()}}[op]
        self.add_change(c)

    def remove_selected(self):
        it = self.change_list.currentItem()
        if it is None:
            return
        del self.changes[int(it.text(0)) - 1]
        self.show_changes()
        self.replan()

    def show_changes(self):
        self.change_list.clear()
        for k, c in enumerate(self.changes, 1):
            self.change_list.addTopLevelItem(QTreeWidgetItem([str(k), json.dumps(c)]))

    def load_changes(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Changes', '', 'JSON (*.json)')
        if p:
            with open(p, encoding='utf-8') as fh:
                doc = json.load(fh)
            self.changes = []
            for c in doc if isinstance(doc, list) else doc.get('changes', []):
                self.add_change(c)

    def save_changes(self):
        p, _ = QFileDialog.getSaveFileName(self, 'Changes', 'changes.json', 'JSON (*.json)')
        if p:
            with open(p, 'w', encoding='utf-8') as fh:
                json.dump(self.changes, fh, indent=1)

    # ---------------------------------------------------------------- runs and the comparison
    def run(self, which: str):
        from ...core.drive import run_dir_name
        from ...core.logreplay import LogError
        from ...core.logreplay.commands import replay_config, submit
        w = self.win
        if self.log is None:
            return
        busy = self.jobs.get(which)
        if busy is not None and busy[0].state not in FINAL:
            return
        try:
            plan = self.make_plan([] if which == 'original' else self.changes)
            out = os.path.join(w.service.runs_dir() or os.path.join(w.workspace(), 'runs'),
                               run_dir_name(f'replay-{which}'))
            cfg = replay_config(w.project, plan, self.world.currentData(), self.rig.currentData(), out,
                                w.service.port, w.service.tm_port, float(w.settings.get('render_budget_mpx')),
                                w.workspace())
            job = submit(w.service, cfg, f'Replay ({which}): {self.log.meta.get("name")}')
        except (LogError, ValueError, KeyError, RuntimeError) as e:
            self.state.set('State', f'cannot run: {e}')
            return
        self.jobs[which] = (job, out)
        self.state.set(which.capitalize(), f'queued: {os.path.basename(out)}')
        w.jobs_dock.raise_()

    def on_finished(self, job):
        for which, (j, out) in self.jobs.items():
            if j is job:
                self.state.set(which.capitalize(), f'{job.state}: {os.path.basename(out)}')
        self.recon.on_finished(job)

    def done_runs(self):
        out = {}
        for which, (j, d) in self.jobs.items():
            if j.state in FINAL and os.path.isfile(os.path.join(d, 'replay.json')):
                out[which] = d
        return out

    def do_compare(self, original: str | None = None, changed: str | None = None):
        from .. import tasks
        runs = self.done_runs()
        a, b = original or runs.get('original'), changed or runs.get('changed')
        if not a or not b:
            self.state.set('State', 'replay the original and the changed log first')
            return
        self.state.set('State', 'comparing…')
        self.show_pair(a, b)

        def work():
            from ...core.logreplay.compare import compare
            return compare(a, [b], out_dir=os.path.join(os.path.dirname(b), os.path.basename(b) + '-comparison'))
        tasks.run(self, work, self.show_comparison, lambda e: self.state.set('State', f'compare failed: {e}'))

    def compare_runs(self, run_a: str, run_b: str):
        """Two Falcon run folders side by side: A the baseline, B the candidate, on one time slider with their
        divergences marked (ui.replay_sync; computed in the background — `self.sync.loaded` fires when done). The
        numbers below are the replay comparison when both are replays of one log (core.logreplay.compare), else
        what both recorded (core.logreplay.sync's summary). Other pages: `win.go('replay').compare_runs(a, b)`."""
        from .. import tasks
        a, b = os.path.abspath(run_a), os.path.abspath(run_b)
        if self.win.page('replay') is self and self.win.stack.currentWidget() is not self:
            self.win.go('replay')
        self.show_pair(a, b)
        both_replays = all(os.path.isfile(os.path.join(d, 'replay.json')) for d in (a, b))
        if not both_replays:
            self._want_summary = True
            return

        def work():
            from ...core.logreplay.compare import compare
            return compare(a, [b])

        def fail(e):
            if self.sync.paths != (a, b):
                return
            self._want_summary = f'not a paired replay comparison ({e}).'
            if self.sync.data is not None:
                self._sync_loaded(self.sync.data)
        tasks.run(self, work, self.show_comparison, fail)

    def show_pair(self, a: str, b: str):
        """Open Side by side on runs a and b (their fidelity and like-for-like notice at once, the views when the
        synchronisation is computed)."""
        self.comparison = None
        self._want_summary = False
        self.table.clear()
        self.note.setText('')
        self.right.setCurrentIndex(1)
        set_run_context(self.win, b)
        self._show_fidelity([fidelity_of(a), fidelity_of(b)])
        self.sync.load(a, b)

    def _show_fidelity(self, levels):
        from ...core import fidelity as F
        self.cmp_fid.set(F.combine(levels))
        tone, why = likeness(levels)
        self.cmp_fid.setToolTip(f'Not like for like: {why}' if tone == 'critical' else why)

    def _sync_loaded(self, d):
        want, self._want_summary = self._want_summary, False
        if not want or d is None or self.comparison is not None:
            return
        sa, sb = d['summary']['a'], d['summary']['b']

        def fmt(v):
            return '—' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))

        def diff(x, y):
            if x is None or y is None:
                return '—'
            return f'{float(y) - float(x):+.3f}'
        rows = [('Collision (boxes overlap)', 'collision'), ('Minimum TTC, s', 'min_ttc_s'),
                ('Ego path, m', 'ego_path_m'), ('Duration, s', 'duration_s'), ('Actors', 'actors')]
        for text, key in rows:
            x, y = sa.get(key), sb.get(key)
            if isinstance(x, bool):
                x, y = float(x), float(y)
            self._row(text, fmt(x), fmt(y), diff(x, y))
        sm = d['summary']
        self._row('Ego position, largest difference (m)', '', '',
                  '—' if sm['max_gap_m'] is None else f'{sm["max_gap_m"]:.3f} at {sm["max_gap_t"]:.2f} s')
        why = want if isinstance(want, str) else 'not a paired replay comparison.'
        self.note.setText(f'What both runs recorded, on one clock: {why} One pair, no interval. ' + ' '.join(d['notes']))

    def _row(self, text, a, b, diff):
        it = QTreeWidgetItem([text, a, b, diff])
        from ..theme import mono
        for c in (1, 2, 3):
            it.setTextAlignment(c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            it.setFont(c, mono(12))
            it.setToolTip(c, it.text(c))
        self.table.addTopLevelItem(it)
        fit_rows(self.table, self.table.topLevelItemCount())
        return it

    def show_comparison(self, doc):
        self.comparison = doc
        self.table.clear()
        p = doc['pairs'][0]

        def fmt(v):
            return '—' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
        for key, text in SHOWN_METRICS:
            self._row(text, fmt(doc['original_metrics'].get(key)), fmt(p['metrics'].get(key)),
                      fmt(p['difference'].get(key)))
        st = p['ego_step_by_step']
        self._row('Ego position, largest difference (m)', '', '',
                  f'{fmt(st["max_position_difference_m"])} at {fmt(st["at_t"])} s')
        fid = doc.get('fidelity_runs') or {}
        if fid:
            self._show_fidelity([fid.get('original')] + list(fid.get('changed') or []))
        self.note.setText(doc.get('interval_note') or '')
        self.state.set('State', 'compared')

    def open_run(self, run_path: str):
        """Send to ▸ Replay: the Falcon run imported as a log (kind 'falcon') and chosen, ready to change and replay."""
        self.win.go('replay')
        k = self.kind.findData('falcon')
        self.kind.setCurrentIndex(k if k >= 0 else 0)
        self.path.setText(run_path)
        self.right.setCurrentIndex(0)
        self.do_import()


def log_fidelity(meta: dict) -> str:
    """A log's fidelity: what its import recorded; for a log imported before 4.2, recorded when it comes from a
    dataset or a bag (a recording by definition) and the source run's when it is a Falcon run (unknown when that
    run is gone)."""
    from ...core import fidelity as F
    lv = (meta or {}).get('fidelity')
    if lv in F.LEVELS:
        return lv
    src = (meta or {}).get('source') or {}
    if src.get('format') and src.get('format') != 'falcon':
        return 'recorded'
    return F.of_run(src['path']) if src.get('path') else 'unknown'
