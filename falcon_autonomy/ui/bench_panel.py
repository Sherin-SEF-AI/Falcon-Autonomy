"""Scenarios ▸ Planner benchmark: drive planners over a fixed route set and compare them by the nuPlan/NAVSIM-style PDM
score with bootstrap intervals (core.planbench, sim.benchrun; docs/planner-benchmark.md)."""
from __future__ import annotations

import json
import os
import sys

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem

from ..core import planbench as PB
from ..core.jobs import JobSpec
from .widgets import Form, Section, button, label

COLS = ('pdms', 'nc', 'dac', 'ddc', 'ttc', 'ep', 'comfort')


class BenchSection(Section):
    def __init__(self, page):
        super().__init__('Planner benchmark',
                         'Each planner drives the same fixed routes (a world, a scenario, a pinned seed; kinematic '
                         'actors, no background traffic) and is scored as nuPlan and NAVSIM score planners: no '
                         'at-fault collisions, drivable area, driving direction, time to collision, progress against '
                         'the lane follower, comfort, and their PDM score — means over routes with 95 % bootstrap '
                         'intervals. Routes whose world the project lacks are skipped.')
        self.page, self.win = page, page.win
        f = Form()
        self.route_set = QComboBox()
        for k in sorted(PB.ROUTE_SETS):
            self.route_set.addItem(f'{k} ({len(PB.ROUTE_SETS[k])} routes)', k)
        self.route_set.setToolTip('\n'.join(f'{k}: {v}' for k, v in PB.ROUTE_SET_ABOUT.items()))
        self.planners = QLineEdit('follower, none')
        self.planners.setToolTip('follower (the reference), none, autopilot, policy:<folder>, stack:<suite file>; '
                                 'the follower is always driven as the reference for progress')
        self.name = QLineEdit('bench')
        f.row('Routes', self.route_set)
        f.row('Planners', self.planners)
        f.row('Benchmark', self.name)
        self.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Run the benchmark', self.run, tip='falcon bench run: one scenario run per planner and '
                                                              'route, resumable'))
        b.addWidget(button('Show results', self.show))
        b.addStretch(1)
        self.add(b)
        self.status = label('', muted=True, wrap=True, small=True)
        self.add(self.status)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(['Planner', 'Routes'] + [PB.TERM_NAMES.get(c, c).split(' (')[0] if c != 'pdms'
                                                            else 'PDMS' for c in COLS])
        self.table.setRootIsDecorated(False)
        self.add(self.table)
        self.win.bridge.finished.connect(lambda j: self.show() if j.spec.kind == 'bench' else None)

    def planner_list(self) -> list:
        return [p.strip() for p in self.planners.text().split(',') if p.strip()]

    def argv(self) -> list:
        from ..sim.benchrun import run_argv
        return run_argv(self.win.project.path, self.name.text().strip() or 'bench', self.planner_list(),
                        self.route_set.currentData(), None, self.win.fake_carla)

    def run(self):
        w = self.win
        if not self.planner_list():
            self.status.setText('name at least one planner')
            return None
        try:
            PB.bench_dir(w.workspace(), self.name.text().strip() or 'bench')
        except PB.BenchError as e:
            self.status.setText(str(e))
            return None
        if not w.project.path or w.project.dirty:
            if not w.save():
                return None
        skipped = [r['id'] for r in PB.routes(self.route_set.currentData())
                   if PB.resolve_world(w.project, r['world'])[0] is None]
        job = w.queue.submit(JobSpec('bench', f'Planner benchmark {self.name.text().strip() or "bench"}',
                                     [sys.executable, '-m', 'falcon_autonomy'] + self.argv(), memory_gb=0.5))
        self.status.setText(f'queued; {len(skipped)} routes will be skipped (no world for them)' if skipped
                            else 'queued')
        w.jobs_dock.raise_()
        return job

    def show(self):
        self.table.clear()
        if not self.win.project.path:
            return None
        try:
            d = PB.bench_dir(self.win.workspace(), self.name.text().strip() or 'bench')
        except PB.BenchError as e:
            self.status.setText(str(e))
            return None
        p = os.path.join(d, 'results.json')
        if not os.path.isfile(p):
            self.status.setText('no results yet')
            return None
        with open(p, encoding='utf-8') as fh:
            res = json.load(fh)
        for name, v in sorted(res['planners'].items(), key=lambda kv: -((kv[1]['metrics'].get('pdms') or {})
                                                                          .get('value') or 0)):
            cells = []
            for c in COLS:
                m = v['metrics'].get(c) or {}
                cells.append('—' if m.get('value') is None else
                             f'{m["value"]:.3f} [{m["ci_low"]:.2f}, {m["ci_high"]:.2f}]')
            self.table.addTopLevelItem(QTreeWidgetItem([name, str(v['n_routes'])] + cells))
        comps = []
        for c in res.get('comparisons') or ():
            m = c['metrics'].get('pdms') or {}
            if m.get('diff') is not None:
                comps.append(f'{c["b"]} − {c["a"]}: PDMS {m["diff"]:+.3f} [{m["ci_low"]:+.3f}, {m["ci_high"]:+.3f}] over '
                             f'{c["routes"]} routes')
        self.status.setText('; '.join(comps) + (f'; {len(res["failed"])} runs failed' if res.get('failed') else ''))
        return res
