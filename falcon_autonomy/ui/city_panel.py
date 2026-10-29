"""Worlds ▸ City around the roads: an imported world dressed with CARLA's own buildings, trees, lamps and street
furniture from an OpenStreetMap file (core.cityprops) — switched on or off per world, built again from another file
(a regional extract has the buildings a roads-only download lacks), and what was placed, skipped and why."""
from __future__ import annotations

import json
import os

from PyQt6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QMessageBox, QSpinBox

from ..core.paths import tilde
from ..core.project import Edit
from .widgets import Form, KeyValues, Section, button, label, set_enabled

ABOUT = ('CARLA cannot import a new mesh at runtime, but it can place the meshes cooked into its package: its towns\' '
         'buildings, trees and lamps. Each OpenStreetMap building footprint gets the town building nearest in size '
         '(turned to the footprint, uniformly scaled 0.6–1.6×, kept off the lanes and other buildings) — a stand-in '
         'of about the right size, not the real building: façade, roof and style are the town mesh\'s. Trees are '
         'OSM\'s where mapped and spread over parks and woods; lamps OSM\'s and every 30 m along sidewalks; bus '
         'stops, benches and bins at OSM\'s points. Beside the roads the ground stays CARLA\'s empty grey (no '
         'plausible ground mesh in the package). Labels stay right: buildings 3, vegetation 9, poles 6 (measured). '
         'docs/worlds.md.')
LAMP_MODES = (('osm+procedural', 'OSM\'s and every 30 m along sidewalks'), ('osm', 'OSM\'s only'), ('off', 'none'))


class CitySection(Section):
    def __init__(self, page):
        super().__init__('City around the roads', ABOUT)
        self.page = page
        self.job = None
        self.pending_source = {}             # world id → a file chosen but not built yet
        self.on = QCheckBox('Place the city in every run of this world')
        self.on.toggled.connect(self.set_on)
        f = Form()
        f.row('In runs', self.on, 'Recorded in run.json (the plan\'s SHA-256); off: the roads alone')
        src_row = QHBoxLayout()
        self.source = label('', muted=True, wrap=True)
        src_row.addWidget(self.source, 1)
        src_row.addWidget(button('Choose a file…', self.choose, tip='An .osm or .osm.pbf file with buildings (a '
                                                                    'regional extract): read, never changed'))
        f.row('OpenStreetMap file', src_row)
        self.trees = QCheckBox('Trees')
        self.trees.setChecked(True)
        opts = QHBoxLayout()
        opts.addWidget(self.trees)
        opts.addStretch(1)
        f.row('Place', opts)
        self.lamps = QComboBox()
        for k, t in LAMP_MODES:
            self.lamps.addItem(t, k)
        f.row('Street lamps', self.lamps)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        f.row('Seed', self.seed, 'Which stand-in among equally good ones, tree positions and turns')
        self.add(f)
        row = QHBoxLayout()
        self.build_btn = button('Build the city', self.build, role='primary',
                                tip='Read the file around the world\'s box and plan every placement (a regional '
                                    '.pbf takes minutes)')
        row.addWidget(self.build_btn)
        row.addStretch(1)
        self.add(row)
        self.kv = KeyValues()
        self.add(self.kv)
        page.win.bridge.finished.connect(self.on_job)

    # ------------------------------------------------------------------ view
    def world_dir(self, w):
        return os.path.join(self.page.win.workspace(), os.path.dirname(w.get('xodr', '')))

    def source_for(self, w) -> str:
        c = w.get('city') or {}
        return self.pending_source.get(w['id']) or (c.get('source') or {}).get('path') or \
            (w.get('osm') or {}).get('source', '')

    def show_world(self, w):
        self.kv.clear()
        c = w.get('city') or {}
        self.on.blockSignals(True)
        self.on.setChecked(bool(c.get('enabled')))
        self.on.blockSignals(False)
        set_enabled(self.on, bool(c.get('plan')), 'Build the city first')
        src = self.source_for(w)
        self.source.setText(tilde(src) if src else 'none')
        o = c.get('options') or {}
        self.trees.setChecked(bool(o.get('trees', True)))
        k = self.lamps.findData(o.get('lamps', 'osm+procedural'))
        self.lamps.setCurrentIndex(max(k, 0))
        self.seed.setValue(int(c.get('seed', 0)))
        ok = bool(src) and os.path.isfile(src)
        busy = self.job is not None and self.job.state not in ('done', 'failed', 'cancelled', 'crashed')
        set_enabled(self.build_btn, ok and not busy, 'Building…' if busy else
                    'Choose an OpenStreetMap file first' if not src else f'{tilde(src)} is not on this machine')
        if c.get('error'):
            self.kv.set('State', f'⚠ not built: {c["error"]}')
        if not c.get('plan'):
            self.kv.set('Placed', 'nothing yet (Build the city)')
            return
        self.show_summary(c)

    def show_summary(self, c):
        s = c.get('summary') or {}
        by = s.get('by_kind') or {}
        parts = []
        for kind in ('building', 'tree', 'lamp', 'bus_stop', 'bench', 'bin'):
            if kind in by:
                v = by[kind]
                n = sum(v.values())
                extra = ', '.join(f'{k} {x}' for k, x in sorted(v.items())) if len(v) > 1 else next(iter(v))
                parts.append(f'{n} {kind.replace("_", " ")}{"s" if n != 1 else ""} ({extra})')
        other = sum(sum(v.values()) for k, v in by.items() if k not in ('building', 'tree', 'lamp', 'bus_stop',
                                                                         'bench', 'bin', 'ground'))
        if other:
            parts.append(f'{other} other street furniture')
        if 'ground' in by:
            parts.append('the ground')
        self.kv.set('Placed', '; '.join(parts) or 'nothing')
        fb = (c.get('features') or {}).get('buildings')
        if fb is not None:
            hf = s.get('height_from') or {}
            self.kv.set('Buildings', f'{sum((by.get("building") or {}).values())} of {fb} OSM footprints; size '
                                     f'closeness (RMS of relative L, W, H differences) median {s.get("fit_score_median")}; '
                                     f'heights from tags {hf.get("height", 0)}, from levels {hf.get("levels", 0)}, '
                                     f'typical for the type {hf.get("typical", 0)}')
        sk = s.get('skipped') or {}
        if sk:
            self.kv.set('Skipped', '; '.join(f'{k}: ' + ', '.join(f'{n} {why}' for why, n in sorted(v.items()))
                                             for k, v in sorted(sk.items())))
        src = c.get('source') or {}
        t = c.get('timings') or {}
        self.kv.set('From', f'{tilde(src.get("path", ""))} (read {t.get("read_s", "?")} s, planned '
                            f'{t.get("plan_s", "?")} s); seed {c.get("seed", 0)}')

    # ------------------------------------------------------------------ actions
    def choose(self):
        _, w = self.page.current()
        if w is None:
            return
        start = os.path.dirname(self.source_for(w)) or os.path.expanduser('~')
        path, _ = QFileDialog.getOpenFileName(self, 'An OpenStreetMap file with buildings', start,
                                              'OpenStreetMap (*.osm *.osm.pbf *.pbf *.osm.gz *.osm.bz2)')
        if path:
            self.pending_source[w['id']] = path
            self.show_world(w)

    def set_on(self, on):
        i, w = self.page.current()
        if w is None or not (w.get('city') or {}).get('plan'):
            return
        c = dict(w['city'], enabled=bool(on))
        self.page.win.edit(Edit('set', ['worlds', i, 'city'], c,
                                text=f'City around {w["name"]} {"on" if on else "off"}'))

    def config(self, w) -> dict:
        from ..workers.osm_import import IMPORT_SCHEMA
        d = self.world_dir(w)
        return {'schema': IMPORT_SCHEMA, 'city_only': True, 'world_dir': d, 'out': os.path.join(d, 'city'),
                'source': self.source_for(w), 'seed': int(self.seed.value()),
                'options': {'trees': self.trees.isChecked(),
                            'lamps': self.lamps.currentData(), 'buildings': True, 'props': True}}

    def build(self):
        _, w = self.page.current()
        if w is None or w['kind'] == 'town':
            return None
        cfg = self.config(w)
        try:
            self.job = self.page.win.service.submit_osm_import(cfg, title=f'City around {w["name"]}')
        except (OSError, RuntimeError) as e:
            QMessageBox.warning(self, 'City around the roads', str(e))
            return None
        self.job_world = w['id']
        self.kv.set('State', 'building… (reading the file, then planning)')
        set_enabled(self.build_btn, False, 'Building…')
        return self.job

    def on_job(self, job):
        if job is not self.job:
            return
        proj = self.page.win.project
        i = next((k for k, x in enumerate(proj.doc['worlds']) if x['id'] == self.job_world), None)
        if i is None:
            return
        w = proj.doc['worlds'][i]
        if job.state != 'done':
            tail = '; '.join(t for lvl, t in job.log[-4:] if lvl in ('stderr', 'error', 'warning'))
            self.kv.set('State', f'⚠ {job.state}: {tail}')
            set_enabled(self.build_btn, True)
            return
        try:
            with open(os.path.join(job.spec.run_dir, 'summary.json'), encoding='utf-8') as fh:
                summ = json.load(fh)
        except (OSError, ValueError) as e:
            self.kv.set('State', f'⚠ the build wrote no summary: {e}')
            set_enabled(self.build_btn, True)
            return
        self.pending_source.pop(w['id'], None)
        self.page.win.edit(Edit('set', ['worlds', i, 'city'], summ, text=f'City around {w["name"]}'))
        cur = self.page.current()[1]
        if cur is not None and cur['id'] == w['id']:
            self.show_world(proj.doc['worlds'][i])
