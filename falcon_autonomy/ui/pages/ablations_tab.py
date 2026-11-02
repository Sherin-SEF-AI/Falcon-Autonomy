"""Training ▸ Ablations: state a hypothesis and its factors, plan the grid with a power analysis, run it, read the
verdict with paired intervals, re-run it and check the re-run, export it for a paper (core.ablation, core.paper);
and the registry of every experiment in the workspace (hypothesis, factors, status, verdict). Running, re-running and
exporting are jobs (`falcon experiment ablation run|rerun`, `falcon paper experiment`); planning and registering are
immediate."""
from __future__ import annotations

import json
import os

from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QMessageBox, QPlainTextEdit,
                             QSplitter, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import ablation as AB
from ...core.paths import tilde
from ...core.provenance import read_json
from ..widgets import (INTERVAL_ROLE, Form, IntervalDelegate, PlaceholderTree, Section, button, label, scroll)

METRICS = ('AR100', 'AP', 'AP50', 'AP75')


def project_example(project) -> dict:
    """The manual's example adapted to the project: its first two models (or placeholders), two weathers, day and
    night; the user edits the factors."""
    s = AB.example_spec()
    models = [m['id'] for m in (getattr(project, 'doc', None) or {}).get('models') or ()]
    if len(models) >= 2:
        s['factors'][0]['levels'] = models[:2]
        s['contrasts'][0].update(a=models[0], b=models[1])
    return s


class AblationsTab(QWidget):
    """The tab; `page` is the Training page (its workspace, project, job submission)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        sec = Section('New ablation', 'State the hypothesis and its factors. Falcon plans every combination with '
                                      'seeded replicates — as many as a power analysis says the comparison needs — '
                                      'records the drives, evaluates each model on the same drive, and analyses the '
                                      'pairs with bootstrap intervals and Holm\'s correction.')
        f = Form()
        self.exp_id = QLineEdit('night-recall')
        self.hyp = QLineEdit('the fine-tuned detector raises night recall')
        self.direction = QComboBox()
        for d in AB.DIRECTIONS:
            self.direction.addItem(d, d)
        self.metric = QComboBox()
        self.metric.setEditable(True)
        for m in METRICS:
            self.metric.addItem(m)
        self.effect = QDoubleSpinBox()
        self.effect.setDecimals(3)
        self.effect.setRange(0.001, 100.0)
        self.effect.setValue(0.05)
        self.sd = QDoubleSpinBox()
        self.sd.setDecimals(3)
        self.sd.setRange(0.001, 100.0)
        self.sd.setValue(0.06)
        self.alpha = QDoubleSpinBox()
        self.alpha.setDecimals(3)
        self.alpha.setRange(0.001, 0.5)
        self.alpha.setValue(0.05)
        self.power = QDoubleSpinBox()
        self.power.setDecimals(2)
        self.power.setRange(0.5, 0.99)
        self.power.setValue(0.8)
        self.executor = QComboBox()
        self.executor.addItem('drives and evaluations (CARLA)', 'detection')
        self.executor.addItem('simulated values (a check of the design)', 'simulated')
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(1.0, 600.0)
        self.seconds.setValue(10.0)
        self.direction.setToolTip('Of the metric, B against A')
        self.metric.setToolTip('A number of the evaluation: AP, AP50, AP75, AR100, AP.<class>, …')
        self.effect.setToolTip('The smallest difference worth detecting, in the metric\'s units')
        self.sd.setToolTip('The standard deviation of the paired differences B − A (from a pilot, or a guess)')
        f.row('Experiment', self.exp_id)
        f.row('Hypothesis', self.hyp)
        f.row('Direction', self.direction)
        f.row('Metric', self.metric)
        f.row('Effect to detect', self.effect)
        f.row('Spread of the differences', self.sd)
        f.row('Alpha', self.alpha)
        f.row('Power', self.power)
        f.row('Runs', self.executor)
        f.row('Seconds per drive', self.seconds)
        sec.add(f)
        sec.add(label('Factors and contrasts (JSON)', h=True))
        self.factors = QPlainTextEdit()
        self.factors.setObjectName('mono')
        self.factors.setMinimumHeight(200)
        sec.add(self.factors)
        b = QHBoxLayout()
        b.addWidget(button('Plan', self.plan, tip='The grid, the seeds and the sample size; nothing is created'))
        b.addWidget(button('Register', self.register, role='primary', tip='Write it into experiments/ (its '
                                                                          'definition never changes after)'))
        b.addWidget(button('Example', self.fill_example))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rv = QVBoxLayout(right)
        sec = Section('Registry', 'Every experiment in the workspace: its kind, status and, for an ablation, the '
                                  'hypothesis, factors and verdict. Re-runs say which experiment they re-run.')
        self.reg = PlaceholderTree('No experiment yet: register an ablation on the left.')
        self.reg.setHeaderLabels(['Experiment', 'Kind', 'Status', 'Verdict', 'Hypothesis', 'Factors'])
        self.reg.setRootIsDecorated(False)
        self.reg.setMinimumHeight(160)
        self.reg.currentItemChanged.connect(lambda it, _p: self.show(it))
        sec.add(self.reg)
        b = QHBoxLayout()
        b.addWidget(button('Run or resume', self.run, role='primary'))
        b.addWidget(button('Analyse', self.analyse))
        b.addWidget(button('Re-run and check', self.rerun, tip='The same definition and seeds into a new '
                                                               'experiment, compared with this one'))
        b.addWidget(button('Export for a paper', self.export, tip='LaTeX and CSV tables, PDF/PNG/SVG figures into '
                                                                  'paper/<id>/'))
        b.addStretch(1)
        sec.add(b)
        rv.addWidget(sec)
        sec = Section('Result', 'Each contrast\'s difference B − A with its 95 % interval (the bar: the interval on '
                                'the axis of all contrasts), its Holm-adjusted p and the verdict.')
        self.contrasts = PlaceholderTree('Not analysed yet.')
        self.contrasts.setHeaderLabels(['Contrast', 'n', 'Δ [95 % CI]', 'Interval', 'p adj', 'Verdict'])
        self.contrasts.setRootIsDecorated(False)
        self.contrasts.setMinimumHeight(110)
        self.bars = IntervalDelegate(self.contrasts, -1.0, 1.0)
        self.contrasts.setItemDelegateForColumn(3, self.bars)
        for k, w in enumerate((250, 36, 200, 120, 60)):
            self.contrasts.setColumnWidth(k, w)
        sec.add(self.contrasts)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setObjectName('mono')
        self.text.setMinimumHeight(150)
        sec.add(self.text)
        rv.addWidget(sec, 1)
        split.addWidget(right)
        split.setSizes([460, 820])
        self.fill_example()

    # ------------------------------------------------------------------ the spec
    def ws(self, *parts):
        return self.page.ws(*parts)

    def fill_example(self):
        s = project_example(self.page.win.project)
        self.factors.setPlainText(json.dumps({'factors': s['factors'], 'contrasts': s['contrasts']}, indent=2))

    def spec(self) -> dict:
        try:
            fc = json.loads(self.factors.toPlainText() or '{}')
        except ValueError as e:
            raise AB.AblationError(f'the factors are not JSON: {e}') from None
        if not isinstance(fc, dict):
            raise AB.AblationError('the factors box holds a JSON object {"factors": [...], "contrasts": [...]}')
        s = {'id': self.exp_id.text().strip(), 'hypothesis': {'text': self.hyp.text().strip(),
                                                              'direction': self.direction.currentData()},
             'metric': self.metric.currentText().strip(), 'factors': fc.get('factors') or [],
             'contrasts': fc.get('contrasts') or [],
             'design': {'effect': self.effect.value(), 'sd': self.sd.value(), 'alpha': self.alpha.value(),
                        'power': self.power.value()},
             'run': {'executor': self.executor.currentData(), 'seconds': self.seconds.value()}}
        if self.executor.currentData() == 'simulated':
            s['run']['simulate'] = {'base': 0.5, 'noise_sd': self.sd.value() / 2 ** 0.5}
        return s

    def plan(self):
        try:
            pl = AB.plan(AB.normalise(self.spec(), self.page.win.project))
        except AB.AblationError as e:
            QMessageBox.warning(self, 'Ablation', str(e))
            return None
        self.text.setPlainText(AB.describe_plan(pl))
        return pl

    def register(self):
        try:
            c = AB.create(self.page.win.workspace(), self.spec(), self.page.win.project)
        except AB.AblationError as e:
            QMessageBox.warning(self, 'Ablation', str(e))
            return None
        self.text.setPlainText(f'registered {c["id"]} in {tilde(c["path"])}\n\n' + AB.describe_plan(c['plan']))
        self.refresh(select=c['id'])
        return c

    # ------------------------------------------------------------------ the registry
    def refresh(self, select: str | None = None):
        cur = select or self.current()
        self.reg.clear()
        for r in AB.registry(self.page.win.workspace()):
            fac = ' × '.join(f'{k} ({len(v)})' for k, v in (r['factors'] or {}).items())
            it = QTreeWidgetItem([r['id'], r['kind'] or '', r['status'], r['verdict'] or '—',
                                  r['hypothesis'] or '', fac])
            it.setData(0, 256, r)
            it.setToolTip(4, r['hypothesis'] or '')
            it.setToolTip(0, tilde(r['path']) + (f'\nre-run of {r["rerun_of"]}' if r.get('rerun_of') else ''))
            self.reg.addTopLevelItem(it)
            if r['id'] == cur:
                self.reg.setCurrentItem(it)

    def current(self) -> str | None:
        it = self.reg.currentItem()
        return it.data(0, 256)['id'] if it is not None else None

    def _ablation(self) -> dict | None:
        it = self.reg.currentItem()
        r = it.data(0, 256) if it is not None else None
        if not r or r['kind'] != AB.KIND:
            QMessageBox.information(self, 'Ablation', 'Select an ablation in the registry.')
            return None
        return r

    def show(self, it):
        r = it.data(0, 256) if it is not None else None
        self.contrasts.clear()
        if not r:
            return
        p = os.path.join(r['path'], 'results.json')
        if not os.path.isfile(p):
            st = r['runs'] or {}
            self.text.setPlainText(f'{r["id"]}: {r["status"]}; runs ' + (', '.join(f'{k} {v}' for k, v in st.items())
                                                                       or 'none yet')
                                   + (f' of {r["planned"]} planned' if r.get('planned') else ''))
            return
        try:
            res = read_json(p)
        except (OSError, ValueError) as e:
            self.text.setPlainText(f'results.json could not be read: {e}')
            return
        if res.get('schema') != AB.RESULTS_SCHEMA:
            from ...core.experiments import table
            try:
                self.text.setPlainText(table(res))
            except (KeyError, TypeError):
                self.text.setPlainText(json.dumps(res, indent=1)[:4000])
            return
        span = max([abs(x) for c in res['contrasts'] for x in (c['ci_low'], c['ci_high'], c['diff'])
                    if x is not None] or [1.0])
        self.bars.lo, self.bars.hi = -span * 1.05, span * 1.05
        for c in res['contrasts']:
            ci = (f'{c["diff"]:+.4f} [{c["ci_low"]:+.4f}, {c["ci_high"]:+.4f}]' if c['diff'] is not None
                  and c['ci_low'] is not None else '—')
            it2 = QTreeWidgetItem([c['label'], str(c['n_pairs']), ci, '',
                                   '' if c['p_adjusted'] is None else f'{c["p_adjusted"]:.4f}', c['verdict']])
            it2.setData(3, INTERVAL_ROLE, (c['diff'], c['ci_low'], c['ci_high']))
            it2.setToolTip(5, c.get('note') or '')
            self.contrasts.addTopLevelItem(it2)
        self.text.setPlainText(AB.table(res))

    # ------------------------------------------------------------------ actions (jobs)
    def _job(self, action: str):
        r = self._ablation()
        if r is None or not self.page._ready():
            return None
        argv = ['experiment', 'ablation', action, r['id'], '--project', self.page.win.project.path]
        argv += self.page._carla_args()
        return self.page._cli('experiment', f'Ablation {r["id"]}: {action}', argv, memory_gb=1.5)

    def run(self):
        return self._job('run')

    def rerun(self):
        return self._job('rerun')

    def analyse(self):
        r = self._ablation()
        if r is None:
            return None
        try:
            res = AB.analyse(self.page.win.workspace(), r['id'])
        except (AB.AblationError, OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, 'Ablation', str(e))
            return None
        self.refresh(select=r['id'])
        return res

    def export(self):
        it = self.reg.currentItem()
        r = it.data(0, 256) if it is not None else None
        if not r:
            QMessageBox.information(self, 'Export', 'Select an experiment in the registry.')
            return None
        out = self.ws('paper', r['id'])
        return self.page._cli('export', f'Paper outputs: {r["id"]}', ['paper', 'experiment', r['path'], '--out', out],
                              memory_gb=0.5)
