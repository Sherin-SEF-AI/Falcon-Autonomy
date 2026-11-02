"""The Rare cases page's sections for how often a scenario fails and how much of the operating domain has been tested
(docs/failure-rate.md, docs/odd-coverage.md):

* Failure rate — adaptive importance sampling over the page's scenario (its objective, world, rig, threshold, weather
  searched, rate and seed), a job (`falcon failrate run`, sim.estimate); the probability per encounter with its 95 %
  interval, the effective failures, the rate per km or hour when an encounter rate is given, and why an estimate was
  refused or carries a warning;
* Operating-domain coverage — the ODD grid (core.oddcoverage) as two dimensions at a time, each entry coloured by the
  stated rule on its runs, with how many of its cells are untested; a chosen entry's cells listed with their runs,
  bounds and the failure-rate estimates there; and an action that adds a scenario per empty or under-sampled cell to
  the project (undoable) and queues a rare-case search (or a sweep) of each.
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLineEdit, QSpinBox,
                             QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem)

from ..core import failrate as FR
from ..core import oddcoverage as OC
from ..core.jobs import FINAL, JobSpec
from .widgets import Form, KeyValues, Section, Stat, button, callout, label

JUDGES = (('objective', 'The search\'s objective ≥ the threshold'), ('checks', 'The scenario\'s pass criteria'),
          ('rules', 'A traffic-rule violation'))
EXPOSURE = (('encounter', 'Per encounter'), ('km', 'Per km'), ('hour', 'Per hour'))
DIM_NAMES = {'weather': 'Weather', 'road': 'Road type', 'actor': 'Actor kind', 'speed': 'Ego speed (km/h)',
             'light': 'Light'}
TONE = {'untested': '', 'too_few': 'warning', 'enough': 'good', 'above_target': 'critical'}
STATUS_TEXT = {'untested': 'untested', 'too_few': 'too few runs', 'enough': 'enough runs',
               'above_target': 'above the target'}


def _fmt(p) -> str:
    return '—' if p is None else f'{p:.3g}'


class FailureRateSection(Section):
    """The estimate of the page's scenario (module doc)."""

    def __init__(self, page):
        super().__init__('Failure rate',
                         'How often the scenario fails as its parameters occur, not where: adaptive importance '
                         'sampling (the cross-entropy method) over the parameters as the scenario declares them — '
                         'their distributions are the assumption — with a 95 % interval, refused when too few '
                         'effective failures carry it. It uses the scenario, objective, world, rig, threshold, '
                         'weather, rate and seed chosen above. docs/failure-rate.md.')
        self.page = page
        self.win = page.win
        self.job = None
        f = Form()
        self.stage = QSpinBox()
        self.stage.setRange(10, 5000)
        self.stage.setValue(FR.DEFAULTS['stage_size'])
        self.final = QSpinBox()
        self.final.setRange(10, 20000)
        self.final.setValue(FR.DEFAULTS['final_size'])
        self.stages = QSpinBox()
        self.stages.setRange(1, 20)
        self.stages.setValue(FR.DEFAULTS['max_stages'])
        self.judge = QComboBox()
        for k, t in JUDGES:
            self.judge.addItem(t, k)
        self.judge.currentIndexChanged.connect(lambda _: self.load())
        self.per = QComboBox()
        for k, t in EXPOSURE:
            self.per.addItem(t, k)
        self.per.currentIndexChanged.connect(lambda _: self._exposure_changed())
        self.encounters = QDoubleSpinBox()
        self.encounters.setDecimals(6)
        self.encounters.setRange(0.0, 1e6)
        self.encounters.setValue(0.0)
        self.encounters.setToolTip('How many encounters of this kind happen per km (or hour) in the operating domain: '
                                   'your figure, with its source — simulation cannot measure it')
        self.source = QLineEdit()
        self.source.setPlaceholderText('where the encounter rate comes from')
        for text, w, note in (('Runs per stage', self.stage, 'adaptation: each stage moves the sampling towards the '
                                                             'failures'),
                              ('Final runs', self.final, 'the estimate is made from these'),
                              ('Stages at most', self.stages, ''), ('A failure is', self.judge, ''),
                              ('Rate', self.per, ''), ('Encounters per unit', self.encounters, ''),
                              ('Source of the rate', self.source, '')):
            f.row(text, w, note)
        self.form = f
        self.add(f)
        b = QHBoxLayout()
        self.start_btn = button('Estimate the failure rate', self.start, role='primary',
                                tip='falcon failrate run (a job; resumable)')
        b.addWidget(self.start_btn)
        b.addWidget(button('Refresh', self.load))
        b.addStretch(1)
        self.add(b)
        tiles = QGridLayout()
        tiles.setSpacing(10)
        self.tiles = {}
        for i, (k, cap) in enumerate((('p', 'P(fail) per encounter'), ('rate', 'Per exposure'),
                                      ('ess', 'Effective failures'), ('runs', 'Runs'))):
            t = Stat(cap)
            self.tiles[k] = t
            tiles.addWidget(t, i // 2, i % 2)
        self.add(tiles)
        self.notice = callout('', 'warning')
        self.notice.hide()
        self.add(self.notice)
        self.kv = KeyValues(('State',))
        self.add(self.kv)
        self._exposure_changed()

    def _exposure_changed(self):
        on = self.per.currentData() != 'encounter'
        for w in (self.encounters, self.source):
            self.form.set_row_visible(w, on)
        self.load()

    def study(self) -> str:
        from ..sim.estimate import estimate_name
        p = self.page
        return estimate_name(p.scenario.currentData() or '', p.objective.currentData() or 'criticality',
                             p.model.currentData() or '', p.weather_dims(), p.fps.value(),
                             self.judge.currentData() or 'objective')

    def result_path(self) -> str:
        if not self.win.project.path:
            return ''
        return os.path.join(self.win.workspace(), 'studies', self.study(), 'failure-rate.json')

    def start(self):
        from ..sim.estimate import estimate_argv
        w, p = self.win, self.page
        if self.job is not None and self.job.state not in FINAL:
            return
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        obj = p.objective.currentData()
        if obj != 'criticality' and not p.model.currentData():
            self.kv.set('State', 'the perception objective needs a model: register one on the Models page')
            return
        per = self.per.currentData()
        if per != 'encounter' and not self.encounters.value() > 0:
            self.kv.set('State', f'give how many encounters happen per {per}, or choose per encounter')
            return
        argv = [sys.executable, '-m', 'falcon_autonomy'] + estimate_argv(
            w.project.path, p.scenario.currentData(), self.study(), p.world.currentData(), p.rig.currentData(),
            self.stage.value(), self.final.value(), max_stages=self.stages.value(), batch=p.batch.value(),
            threshold=p.threshold.value(), seed=p.seed.value(), judge=self.judge.currentData(), objective=obj,
            count=p.count.currentData(), feasibility=p.feasible.isChecked(), model=p.model.currentData() or '',
            weather=p.weather_dims(), fps=p.fps.value(), exposure_per=per, encounters=self.encounters.value(),
            exposure_source=self.source.text().strip(), fake_carla=w.fake_carla)
        self.job = w.queue.submit(JobSpec('study', f'Failure rate of {p.scenario.currentText()} (at most '
                                                   f'{self.stages.value() * self.stage.value() + self.final.value()} '
                                                   'runs)', argv, memory_gb=0.5))
        w.jobs_dock.raise_()
        self.kv.set('State', self.job.summary())

    def load(self):
        from ..core.provenance import read_json
        path = self.result_path()
        res = None
        if path and os.path.isfile(path):
            try:
                res = read_json(path)
            except (OSError, ValueError):
                res = None
        self.result = res
        busy = self.job is not None and self.job.state not in FINAL
        for t in self.tiles.values():
            t.set('—')
        self.notice.hide()
        if res is None:
            if not busy:
                self.kv.set('State', 'no estimate for this scenario and objective yet')
            return
        e = res.get('estimate') or {}
        refused = e.get('status') == 'refused'
        if refused:
            self.tiles['p'].set('refused' if e.get('upper95') is None else f'≤ {e["upper95"]:.3g}',
                                sub='a 95 % bound only' if e.get('upper95') else 'no answer', tone='critical')
        else:
            ci = e.get('ci95') or [None, None]
            self.tiles['p'].set(_fmt(e.get('p')), sub=f'95 %: {_fmt(ci[0])} – {_fmt(ci[1])}',
                                tone='warning' if e.get('status') == 'warning' else '')
        try:
            rate = FR.exposure_rate(e, {'per': self.per.currentData(), 'encounters': self.encounters.value(),
                                        'source': self.source.text().strip()}
                                    if self.per.currentData() != 'encounter' else None)
        except FR.FailRateError as ex:
            rate = {'text': str(ex), 'rate': None}
        if rate.get('rate') is not None and rate.get('per') != 'encounter':
            unit = f'/ 10^{rate["power"]} {rate["per"]}' if rate['power'] else f'/ {rate["per"]}'
            ci = rate.get('ci95') or [None, None]
            self.tiles['rate'].set(f'{rate["rate"]:.3g}', unit, sub=f'95 %: {_fmt(ci[0])} – {_fmt(ci[1])}')
        else:
            self.tiles['rate'].set('—', sub='per encounter' if self.per.currentData() == 'encounter' and not refused
                                   else rate.get('text', ''))
        self.tiles['ess'].set(_fmt(e.get('ess_failures')) if e.get('ess_failures') is not None else '—',
                              sub=f'{e.get("failures", 0)} failing of {e.get("n", 0)} final runs')
        runs = res.get('runs') or {}
        self.tiles['runs'].set(str(runs.get('total', 0)), sub=f'{len(res.get("stages") or [])} adaptation stages'
                               + (f', {runs["lost"]} lost' if runs.get('lost') else ''))
        if e.get('reasons'):
            self.notice.label.setText(('Refused: ' if refused else 'Warning: ') + '; '.join(e['reasons']) + '.')
            self.notice.show()
        if not busy:
            mc = res.get('plain_mc') or {}
            need = res.get('mc_runs_for_same_rel_se')
            self.kv.set('State', ('done: ' if res.get('done') else 'stopped: ') + (res.get('stop_reason') or ''))
            self.kv.set('Plain Monte Carlo', f'stage 0: {mc.get("failures")} failures in {mc.get("runs")} runs'
                        + (f'; the same precision would need about {need:,.0f} runs' if need and not refused else ''))
            self.kv.set('Failure', (res.get('judge') or {}).get('about', ''))

    def poll(self):
        if self.job is not None and self.job.state not in FINAL:
            self.kv.set('State', self.job.summary())

    def finished(self, job):
        if job is self.job:
            self.load()


class CoverageGrid(QTableWidget):
    """The pivot of the ODD grid: each entry its runs and untested cells, coloured by the rule on its runs."""

    def __init__(self):
        super().__init__()
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.setMinimumHeight(260)
        self.pv = None

    def show_pivot(self, pv: dict):
        from .theme import THEME
        self.pv = pv
        self.clear()
        self.setRowCount(len(pv['row_labels']))
        self.setColumnCount(len(pv['col_labels']))
        self.setVerticalHeaderLabels([OC.LABELS.get(pv['rows'], {}).get(x, x) for x in pv['row_labels']])
        self.setHorizontalHeaderLabels([OC.LABELS.get(pv['cols'], {}).get(x, x) for x in pv['col_labels']])
        for i, line in enumerate(pv['grid']):
            for j, x in enumerate(line):
                cs = x['cell_status']
                text = f'{x["runs"]} run{"s" if x["runs"] != 1 else ""}'
                if x['failures']:
                    text += f' · {x["failures"]} failed'
                text += f'\n{cs["untested"]}/{x["cells"]} untested'
                if x['estimates']:
                    text += f'\n{len(x["estimates"])} estimate{"s" if len(x["estimates"]) > 1 else ""}'
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                tone = TONE[x['status']]
                if tone:
                    c = QColor(THEME.status(tone))
                    c.setAlphaF(0.28)
                    it.setBackground(c)
                tip = (f'{STATUS_TEXT[x["status"]]}: {x["runs"]} distinct runs, {x["failures"]} failed'
                       + (f'; 95 % upper bound {x["upper"]:.3g}' if x.get('upper') is not None else '')
                       + (f'; needs {x["needed"]} more without a failure' if x['status'] == 'too_few' else '')
                       + f'\n{x["cells"]} cells: ' + ', '.join(f'{STATUS_TEXT[k]} {n}' for k, n in cs.items() if n))
                it.setToolTip(tip)
                it.setData(256, (pv['row_labels'][i], pv['col_labels'][j]))
                self.setItem(i, j, it)
        self.resizeRowsToContents()
        self.resizeColumnsToContents()


class CoverageSection(Section):
    """The operating domain's grid and the action that fills its gaps (module doc)."""

    def __init__(self, page):
        super().__init__('Operating-domain coverage',
                         'Every scenario run of the workspace placed in the ODD: weather × road type × actor kind × '
                         'ego speed × light. A cell has enough runs when the Clopper–Pearson upper bound on its '
                         'failure rate is at most the target (59 runs without a failure for 5 % at 95 %); a '
                         'search\'s runs count, though they are not random draws. Failure-rate estimates appear in '
                         'their cells. '
                         'docs/odd-coverage.md.')
        self.page = page
        self.win = page.win
        self.jobs: list = []
        self.cov = None
        f = Form()
        self.rows = QComboBox()
        self.cols = QComboBox()
        for c, default in ((self.rows, 'weather'), (self.cols, 'light')):
            for k in OC.DIMENSIONS:
                c.addItem(DIM_NAMES[k], k)
            c.setCurrentIndex(c.findData(default))
            c.currentIndexChanged.connect(lambda _: self.draw())
        self.target = QDoubleSpinBox()
        self.target.setDecimals(4)
        self.target.setRange(0.0001, 0.5)
        self.target.setSingleStep(0.01)
        self.target.setValue(OC.TARGET)
        self.conf = QComboBox()
        for v in (0.9, 0.95, 0.99):
            self.conf.addItem(f'{v:.0%}', v)
        self.conf.setCurrentIndex(1)
        for text, w in (('Rows', self.rows), ('Columns', self.cols), ('Target failure rate', self.target),
                        ('Confidence', self.conf)):
            f.row(text, w)
        self.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Refresh', lambda: self.refresh(force=True)))
        b.addStretch(1)
        self.add(b)
        tiles = QGridLayout()
        tiles.setSpacing(10)
        self.tiles = {}
        for i, (k, cap) in enumerate((('cells', 'Cells in the domain'), ('enough', 'Enough runs'),
                                      ('too_few', 'Too few runs'), ('untested', 'Untested'))):
            t = Stat(cap)
            self.tiles[k] = t
            tiles.addWidget(t, i // 2, i % 2)
        self.add(tiles)
        self.summary = label('', muted=True, wrap=True)
        self.add(self.summary)
        self.grid = CoverageGrid()
        self.grid.currentCellChanged.connect(lambda *_: self.show_entry())
        self.add(self.grid)
        self.add(label('The chosen entry\'s cells', h=True))
        self.cells = QTreeWidget()
        self.cells.setHeaderLabels(['Cell', 'Runs', 'Failed', 'Status', 'Upper 95 %', 'Needs', 'Estimate'])
        self.cells.setRootIsDecorated(False)
        self.cells.setMinimumHeight(140)
        self.add(self.cells)
        ff = Form()
        self.n_fill = QSpinBox()
        self.n_fill.setRange(1, 100)
        self.n_fill.setValue(5)
        self.fill_kind = QComboBox()
        self.fill_kind.addItem('A rare-case search (maps failures)', 'search')
        self.fill_kind.addItem('A sweep of random runs (the draws the rule assumes)', 'sweep')
        ff.row('Cells to fill', self.n_fill)
        ff.row('With', self.fill_kind)
        self.add(ff)
        fb = QHBoxLayout()
        self.fill_btn = button('Fill the gaps', self.fill, role='primary',
                               tip='Add a scenario per untested or under-sampled cell to the project (undoable) and '
                                   'queue a search of each')
        fb.addWidget(self.fill_btn)
        fb.addStretch(1)
        self.add(fb)
        self.fill_state = KeyValues(('Filling',))
        self.add(self.fill_state)
        self.target.valueChanged.connect(lambda _: self.refresh())
        self.conf.currentIndexChanged.connect(lambda _: self.refresh())

    def signature(self) -> tuple:
        """What the grid depends on, cheap to read (directory listings and file times, not the runs' configs): the
        page refreshes on every project edit, and the grid is recomputed only when this changes."""
        import json
        ws = self.win.workspace()
        sig = [self.target.value(), self.conf.currentData(),
               json.dumps((self.win.project.doc.get('loop') or {}).get('odd'), sort_keys=True, default=str)]
        for sub in ('studies', 'suites'):
            d = os.path.join(ws, sub)
            for n in sorted(os.listdir(d)) if os.path.isdir(d) else ():
                for f in ('results.jsonl', 'failure-rate.json', 'study.json'):
                    try:
                        st = os.stat(os.path.join(d, n, f))
                    except OSError:
                        continue
                    sig.append((sub, n, f, st.st_mtime_ns, st.st_size))
        for f in ('runs', 'odd.json'):
            try:
                sig.append((f, os.stat(os.path.join(ws, f)).st_mtime_ns))
            except OSError:
                pass
        return tuple(sig)

    def refresh(self, force: bool = False):
        if not self.win.project.path:
            self.cov = None
            self._sig = None
            self.summary.setText('Save the project: coverage is read from its workspace.')
            self.grid.setRowCount(0)
            return
        sig = self.signature()
        if not force and self.cov is not None and sig == getattr(self, '_sig', None):
            return
        self._sig = sig
        try:
            self.cov = OC.coverage(self.win.workspace(), self.win.project, self.target.value(),
                                   self.conf.currentData())
        except ValueError as e:
            self.cov = None
            self.summary.setText(str(e))
            return
        cov = self.cov
        n, cnt = cov['domain']['cells'], cov['counts']
        self.tiles['cells'].set(f'{n:,}', sub=cov['domain']['source'])
        self.tiles['enough'].set(str(cnt['enough']), sub=f'{cnt["above_target"]} above the target',
                                 tone='critical' if cnt['above_target'] else '')
        self.tiles['too_few'].set(str(cnt['too_few']), sub=f'{cov["rule"]["runs_if_no_failure"]} runs each, without '
                                                          'a failure')
        self.tiles['untested'].set(f'{cnt["untested"]:,}', sub=f'{cnt["untested"] / n:.1%} of the domain' if n else '')
        skipped = sum(cov['skipped'].values())
        self.summary.setText(f'{cov["runs"]} scenario runs placed' + (f'; {cov["outside_domain"]} outside the ODD'
                                                                       if cov['outside_domain'] else '')
                             + (f'; {skipped} not placed ({", ".join(sorted(cov["skipped"]))})' if skipped else '')
                             + f'; {len(cov["estimates"])} failure-rate estimate'
                             + ('' if len(cov['estimates']) == 1 else 's')
                             + '. '
                             + cov['rule']['about'][:1].upper() + cov['rule']['about'][1:] + '.')
        self.draw()

    def draw(self):
        if self.cov is None:
            return
        r, c = self.rows.currentData(), self.cols.currentData()
        if r == c:
            self.grid.setRowCount(0)
            return
        self.grid.show_pivot(OC.pivot(self.cov, r, c))
        self.show_entry()

    def show_entry(self):
        self.cells.clear()
        it = self.grid.currentItem()
        if it is None or self.cov is None or self.grid.pv is None:
            return
        rl, cl = it.data(256)
        pv = self.grid.pv
        for cid, e in sorted(self.cov['cells'].items()):
            lab = e['labels']
            if lab[pv['rows']] != rl or lab[pv['cols']] != cl:
                continue
            est = '; '.join(f'{x["p"]:.3g} ({x["study"]})' if x.get('status') in ('ok', 'warning') and x.get('p')
                            else f'refused ({x["study"]})' for x in e.get('estimates') or ())
            row = QTreeWidgetItem([cid, str(e['runs']), str(e['failures']), STATUS_TEXT[e['status']],
                                   _fmt(e.get('upper')), str(e['needed']), est or '—'])
            row.setToolTip(0, OC.describe_cell(cid))
            row.setData(0, 256, cid)
            self.cells.addTopLevelItem(row)

    def fill(self):
        from ..core import scenarios as SC
        w, p = self.win, self.page
        if not w.project.path or w.project.dirty:
            if not w.save():
                return
        self.refresh(force=True)
        if self.cov is None:
            return
        scns = [SC.builtin(k) for k in SC.LIBRARY] + list(w.project.doc['scenarios'])
        plan = OC.fill_plan(self.cov, scns, self.n_fill.value(), self.fill_kind.currentData())
        if not plan['plan']:
            self.fill_state.set('Filling', 'nothing to fill that a scenario of the library or the project can '
                                           'produce')
            return
        from ..core.project import Edit
        for e in plan['plan']:
            if w.project.item('scenarios', e['scenario']['id']) is None:
                w.edit(Edit('insert', ['scenarios'], e['scenario'],
                            text=f'Add the coverage scenario {e["scenario"]["name"]}'))
        if not w.save():
            return
        for e in plan['plan']:
            argv = [sys.executable, '-m', 'falcon_autonomy'] + OC.plan_argv(
                w.project.path, e, p.world.currentData(), p.rig.currentData(), p.seed.value(),
                p.threshold.value(), w.fake_carla)
            self.jobs.append(w.queue.submit(JobSpec('study', f'Fill {e["cell"]} ({e["kind"]} of {e["base"]}, '
                                                             f'{e["runs"]} runs)', argv, memory_gb=0.5)))
        w.jobs_dock.raise_()
        what = {'search': 'searches', 'sweep': 'sweeps'}[plan['plan'][0]['kind']]
        self.fill_state.set('Filling', f'{len(plan["plan"])} scenarios added and {len(plan["plan"])} {what} queued; '
                                       f'{plan["candidates"]} cells need runs, '
                                       f'{plan["no_scenario_cells"]:,} no scenario can produce')
        p.refresh()

    def finished(self, job):
        if job in self.jobs:
            self.refresh(force=True)


def attach(page) -> tuple:
    """The two sections, created for a Rare cases page (connected to the bridge's finished signal)."""
    rate, cov = FailureRateSection(page), CoverageSection(page)
    page.win.bridge.finished.connect(rate.finished)
    page.win.bridge.finished.connect(cov.finished)
    return rate, cov

