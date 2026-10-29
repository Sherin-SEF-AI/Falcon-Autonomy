"""The dashcam-route dialog: pick clips and a window, import that stretch's roads as a world, drive it in CARLA and
pair the simulated frames with the footage (core.dashcam_route). Entry point: open_route_dialog(win)."""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                             QHBoxLayout, QLineEdit, QListWidget, QMessageBox, QVBoxLayout)

from ..core import dashcam_route as R
from ..core.jobs import FINAL
from ..core.paths import tilde
from ..core.project import Edit
from ..core.worlds import world_dir
from . import tasks
from .widgets import Form, KeyValues, Section, button, label


class RouteDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle('Drive a dashcam route')
        self.resize(820, 760)
        self.win_doc = None             # window.json
        self.route_dir = ''
        self.cfg = None                 # the import config
        self.world_id = ''
        self.job = None
        self.drive_job = None
        self.drive_out = ''
        v = QVBoxLayout(self)

        s1 = Section('1 · Footage', 'Clips from your dashcam and the stretch of them to drive. Consecutive clips join '
                                    'on their GPS time. Nothing leaves this computer.')
        self.clips = QListWidget()
        self.clips.setMaximumHeight(90)
        s1.add(self.clips)
        row = QHBoxLayout()
        row.addWidget(button('Add clips…', self.add_clips))
        row.addWidget(button('Remove', lambda: [self.clips.takeItem(self.clips.row(i))
                                                for i in self.clips.selectedItems()]))
        row.addStretch(1)
        s1.add(row)
        f = Form()
        self.start = f.row('Start', QDoubleSpinBox(), 'seconds from the first clip\'s start')
        self.start.setRange(0, 86400)
        self.seconds = f.row('Length', QDoubleSpinBox(), 'seconds (a longer stretch makes a bigger world)')
        self.seconds.setRange(1, 3600)
        self.seconds.setValue(60)
        s1.add(f)
        s1.add(button('Plan the route', self.plan, role='primary'))
        self.plan_kv = KeyValues(('Route',))
        s1.add(self.plan_kv)
        v.addWidget(s1)

        s2 = Section('2 · Roads', 'The stretch\'s roads from an OpenStreetMap file you have (.osm or .osm.pbf), cut to '
                                  'a corridor along the track and imported as a world. Downloading them is optional '
                                  'and asks first.')
        f = Form()
        src = QHBoxLayout()
        self.source = QLineEdit()
        self.source.setPlaceholderText('an .osm or .osm.pbf file covering the area')
        src.addWidget(self.source, 1)
        src.addWidget(button('Browse…', self.browse_osm))
        f.row('OpenStreetMap', src)
        self.lht = f.row('Traffic', QCheckBox('left-hand (India, UK, …)'))
        self.lht.setChecked(True)
        s2.add(f)
        row = QHBoxLayout()
        self.import_btn = button('Import the roads', self.import_roads, role='primary')
        row.addWidget(self.import_btn)
        self.dl_btn = button('Download the area\'s roads…', self.download,
                             tip=f'sends the area to {R.OVERPASS_URL}; asks first')
        row.addWidget(self.dl_btn)
        row.addStretch(1)
        s2.add(row)
        self.import_kv = KeyValues(('Import', 'Fit'))
        s2.add(self.import_kv)
        v.addWidget(s2)

        s3 = Section('3 · Drive', 'A replay run on the imported world: the lane follower drives the route at the '
                                  'recorded speeds, or the car is placed on it every step. Simulated and dashcam '
                                  'frames are then paired by time.')
        f = Form()
        self.driver = f.row('Driver', QComboBox())
        self.driver.addItem('Lane follower at the recorded speeds', 'follower')
        self.driver.addItem('Placed on the route every step', 'log')
        self.rig = f.row('Rig', QComboBox())
        for r in self.win.project.doc['rigs']:
            self.rig.addItem(r.get('name') or r['id'], r['id'])
        self.frames = f.row('Pairs', QCheckBox('also write the dashcam frames (2 a second)'))
        s3.add(f)
        self.drive_btn = button('Drive', self.drive, role='primary')
        self.drive_btn.setEnabled(False)
        self.drive_btn.setToolTip('Plan the route and import its roads first')
        s3.add(self.drive_btn)
        self.drive_kv = KeyValues(('Run', 'Pairs'))
        s3.add(self.drive_kv)
        v.addWidget(s3)
        v.addWidget(label('Limits: the GPS has metres of error; the simulated camera is the rig\'s; an imported world '
                          'has roads only (no buildings or textures); the footage\'s traffic is not reproduced.',
                          muted=True, small=True, wrap=True))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        win.bridge.changed.connect(self.on_job)
        win.bridge.finished.connect(self.on_job)

    # ------------------------------------------------------------------ 1. footage
    def add_clips(self):
        root = R.DC.DEFAULT_ROOT if os.path.isdir(R.DC.DEFAULT_ROOT) else os.path.expanduser('~')
        paths, _ = QFileDialog.getOpenFileNames(self, 'Dashcam clips', root, 'Video (*.mp4 *.MP4)')
        for p in paths:
            self.clips.addItem(p)

    def _clips(self) -> list:
        return [self.clips.item(i).text() for i in range(self.clips.count())]

    def plan(self):
        clips = self._clips()
        if not clips:
            QMessageBox.information(self, 'Route', 'Add one or more clips first.')
            return
        start, secs = self.start.value(), self.seconds.value()
        self.plan_kv.set('Route', 'reading the GPS…')
        tasks.run(self, lambda: R.window(clips, start, secs), self.planned,
                  lambda e: self.plan_kv.set('Route', f'⚠ {e}'))

    def planned(self, w):
        self.win_doc = w
        self.route_dir = R.default_route_dir(w)
        R.save_window(w, self.route_dir)
        self.plan_kv.set('Route', f'{len(w["clips"])} clip(s), {w["start_s"]:g}–{w["end_s"]:g} s: '
                                  f'{len(w["fixes"])} GPS fixes, {w["path_m"]:.0f} m; area '
                                  f'{w["size_m"][0]:.0f} × {w["size_m"][1]:.0f} m'
                                  + (f'; {len(w["holes"])} hole(s) in the GPS' if w['holes'] else '')
                                  + f' → {tilde(self.route_dir)}')

    # ------------------------------------------------------------------ 2. roads
    def browse_osm(self):
        p, _ = QFileDialog.getOpenFileName(self, 'OpenStreetMap file', os.path.expanduser('~'),
                                           'OpenStreetMap (*.osm *.pbf *.osm.gz *.osm.bz2)')
        if p:
            self.source.setText(p)

    def _ensure_world_dir(self) -> str:
        if not self.world_id:
            from ..core.project import new_id
            self.world_id = new_id('w-route')
        return world_dir(self.win.workspace(), self.world_id)

    def download(self):
        if self.win_doc is None:
            QMessageBox.information(self, 'Route', 'Plan the route first: the download is for its area.')
            return
        s, w, n, e = self.win_doc['bbox']
        ok = QMessageBox.question(
            self, 'Download from OpenStreetMap',
            f'This sends the area ({s:.4f}, {w:.4f}) – ({n:.4f}, {e:.4f}) to {R.OVERPASS_URL} and downloads its roads.'
            f'\n\nThe data is {R.OSM_LICENCE}. Download?')
        if ok != QMessageBox.StandardButton.Yes:
            return
        out = os.path.join(self._ensure_world_dir(), 'overpass.osm')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        bbox = self.win_doc['bbox']
        self.import_kv.set('Import', 'downloading…')
        tasks.run(self, lambda: R.fetch_overpass(bbox, out, True), self.downloaded,
                  lambda e: self.import_kv.set('Import', f'⚠ {e}'))

    def downloaded(self, got):
        from ..core.provenance import write_json
        write_json(os.path.join(os.path.dirname(got['out']), 'download.json'), got)
        self.source.setText(got['out'])
        self.import_kv.set('Import', f'downloaded {got["bytes"] / 1e6:.1f} MB in {got["seconds"]:.0f} s '
                                     f'({R.OSM_LICENCE})')

    def import_roads(self):
        if self.win_doc is None:
            QMessageBox.information(self, 'Route', 'Plan the route first.')
            return
        src = self.source.text().strip()
        if not os.path.isfile(src):
            QMessageBox.information(self, 'Route', 'Choose an OpenStreetMap file covering the area (or download it).')
            return
        out = self._ensure_world_dir()
        w = self.win_doc
        self.import_kv.set('Import', 'cutting the corridor (reading a large .pbf takes minutes)…')
        tasks.run(self, lambda: R.corridor_osm(src, w, os.path.join(out, 'corridor.osm')), self.corridor_done,
                  lambda e: self.import_kv.set('Import', f'⚠ {e}'))

    def corridor_done(self, corr):
        from ..core.provenance import write_json
        out = world_dir(self.win.workspace(), self.world_id)
        write_json(os.path.join(out, 'corridor.json'), corr)
        self.cfg = R.import_config(self.win_doc, corr['out'], out, lht=self.lht.isChecked())
        try:
            self.job = self.win.service.submit_osm_import(self.cfg, title='Import the dashcam route\'s roads')
        except RuntimeError as e:
            self.import_kv.set('Import', f'⚠ {e}')
            return
        self.import_kv.set('Import', f'{corr["highways"]} roads in the corridor; converting…')

    # ------------------------------------------------------------------ 3. drive
    def drive(self):
        svc = self.win.service
        self.drive_out = R.new_run_dir(self.win.workspace())
        try:
            cfg = R.drive_config(self.win.project, self.route_dir, self.world_id, self.rig.currentData(),
                                 self.drive_out, svc.port, svc.tm_port, self.driver.currentData(), 20.0, 0,
                                 float(svc.settings.get('render_budget_mpx')))
            from ..core.logreplay.commands import submit
            self.drive_job = submit(svc, cfg, cfg['name'])
        except (ValueError, OSError, RuntimeError, R.DashcamError) as e:
            self.drive_kv.set('Run', f'⚠ {e}')
            return
        self.drive_btn.setEnabled(False)
        self.drive_kv.set('Run', 'queued')

    # ------------------------------------------------------------------ jobs
    def on_job(self, job):
        if job is self.job:
            self._import_job(job)
        elif job is self.drive_job:
            self._drive_job(job)

    def _import_job(self, job):
        self.import_kv.set('Import', job.summary())
        if job.state not in FINAL:
            return
        if job.state != 'done':
            tail = '; '.join(t for lvl, t in job.log[-5:] if lvl in ('stderr', 'error', 'warning'))
            self.import_kv.set('Import', f'⚠ {job.state}: {tail}')
            return
        import json
        with open(os.path.join(self.cfg['out'], 'import.json'), encoding='utf-8') as fh:
            rep = json.load(fh)
        from ..core.worlds import world_entry
        doc = self.win.project.doc
        left = next((tp['id'] for tp in doc['traffic_profiles'] if tp.get('handedness') == 'left'), None)
        tp = left if rep.get('handedness') == 'LHT' and left else 'tp-default'
        w = self.win_doc
        name = f'Dashcam route {w["start_s"]:g}–{w["end_s"]:g} s'
        self.win.edit(Edit('insert', ['worlds'], world_entry(self.world_id, name, self.cfg, rep, None, tp),
                           text=f'Import {name}'))
        lc = rep.get('lane_check') or {}
        self.import_kv.set('Import', f'{(rep.get("graph_totals") or {}).get("roads", "?")} roads, '
                                     f'{rep.get("handedness")}, lane check {lc.get("matched")}/{lc.get("comparable")}; '
                                     f'added as the world "{name}"')
        project, wid, ws, rd = self.win.project, self.world_id, self.win.workspace(), self.route_dir

        def build():
            route = R.build_route(R.load(rd), R.world_for(project, wid, ws))
            from ..core.provenance import write_json
            write_json(os.path.join(rd, 'route.json'), route)
            return route
        self.import_kv.set('Fit', 'matching the route to the lanes…')
        tasks.run(self, build, self.route_built, lambda e: self.import_kv.set('Fit', f'⚠ {e}'))

    def route_built(self, route):
        self.import_kv.set('Fit', R.route_text(route))
        self.drive_btn.setEnabled(True)
        self.drive_btn.setToolTip('')

    def _drive_job(self, job):
        self.drive_kv.set('Run', job.summary())
        if job.state not in FINAL:
            return
        self.drive_btn.setEnabled(True)
        if job.state != 'done':
            self.drive_kv.set('Run', f'⚠ {job.state}: see {tilde(os.path.join(self.drive_out, "run.json"))}')
            return
        self.drive_kv.set('Run', tilde(self.drive_out))
        rd, out, frames = self.route_dir, self.drive_out, self.frames.isChecked()
        self.drive_kv.set('Pairs', 'pairing frames…')
        tasks.run(self, lambda: R.pairs(rd, out, 2.0, frames), self.paired,
                  lambda e: self.drive_kv.set('Pairs', f'⚠ {e}'))

    def paired(self, doc):
        d = doc['sim_along_m']
        a = doc['sim_across_m']
        self.drive_kv.set('Pairs', f'{doc["pairs"]} pairs → {tilde(os.path.join(self.route_dir, "pairs"))}; the '
                                   f'simulated car vs the route: along median {d["median"]} m, across median '
                                   f'{a["median"]} m')


def open_route_dialog(win):
    """Open the dashcam-route dialog over the main window (non-modal; it follows its own jobs)."""
    dlg = RouteDialog(win)
    dlg.show()
    return dlg
