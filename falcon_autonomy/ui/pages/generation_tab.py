"""Training ▸ Generation: active learning that generates its data (core.genround) — rank the class × condition cells
by expected value, generate drives and scenario runs for the best within a budget, retrain, compare per cell on a
held-out set fixed before the first round; the rounds with their cells, intervals and verdicts, and why the chain
stopped. Running is a job (`falcon generate run`); planning and registering are immediate."""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLineEdit, QMessageBox, QPlainTextEdit, QSpinBox, QSplitter,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import genround as G
from ...core.paths import tilde
from ...core.provenance import read_json
from ..widgets import INTERVAL_ROLE, Form, IntervalDelegate, PlaceholderTree, Section, button, scroll


def _spin(lo, hi, v):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(v)
    return s


class GenerationTab(QWidget):
    """The tab; `page` is the Training page (its window, workspace and job submission)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        h.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        sec = Section('New generation experiment',
                      'Each round ranks the class × condition cells by the expected value of more data there, '
                      'records drives and scenario runs for the best cells (labelled from the ground truth), retrains '
                      'the detector from its own settings and compares each cell before and after on a held-out set '
                      'fixed before the first round. It stops when no chosen cell improves.')
        f = Form()
        self.exp_id = QLineEdit('gen-1')
        self.model = QComboBox()
        self.world = QComboBox()
        self.rig = QComboBox()
        self.conditions = QLineEdit(','.join(G.DEFAULTS['conditions']))
        self.conditions.setToolTip('weather/light, comma-separated: weather clear, cloudy, wet, rain, fog, dust; '
                                   'light day, low_sun, twilight, night')
        self.drives = _spin(1, 1000, G.DEFAULTS['budget']['drives'])
        self.frames = _spin(1, 1_000_000, G.DEFAULTS['budget']['frames'])
        self.rounds = _spin(1, 50, G.DEFAULTS['rounds'])
        self.cells = _spin(1, 20, G.DEFAULTS['acquisition']['max_cells'])
        self.reference = QComboBox()
        self.reference.addItem('the overall AP', 'overall')
        self.reference.addItem('a perfect AP (for a model weak everywhere)', 'perfect')
        f.row('Experiment', self.exp_id)
        f.row('Detector', self.model)
        f.row('World', self.world)
        f.row('Rig', self.rig)
        f.row('Conditions', self.conditions)
        f.row('Drives per round', self.drives)
        f.row('Frames per round', self.frames)
        f.row('Rounds at most', self.rounds)
        f.row('Cells per round', self.cells)
        f.row('Shortfall against', self.reference)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Plan', self.plan, tip='The definition, the budget and the rule; nothing is created'))
        b.addWidget(button('Register', self.register, tip='Write it into experiments/ (its definition never changes '
                                                          'after)'))
        b.addWidget(button('Run or resume', self.run, role='primary', tip='A job: fixes the held-out set, then runs '
                                                                          'rounds until the stop rule'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rv = QVBoxLayout(right)
        sec = Section('Rounds', 'Per round the chosen cells with their expected value, the AP before and after on the '
                                'held-out set with 95 % intervals over drives, and the paired difference with its '
                                'interval (the bar) and verdict.')
        self.exps = QComboBox()
        self.exps.currentIndexChanged.connect(lambda _i: self.show())
        sec.add(self.exps)
        self.tree = PlaceholderTree('No round yet.')
        self.tree.setHeaderLabels(['Round · cell', 'EV', 'AP before', 'AP after', 'Δ [95 % CI]', 'Interval',
                                   'Verdict'])
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(160)
        self.bars = IntervalDelegate(self.tree, -1.0, 1.0)
        self.tree.setItemDelegateForColumn(5, self.bars)
        for k, w in enumerate((220, 60, 90, 90, 190, 120)):
            self.tree.setColumnWidth(k, w)
        sec.add(self.tree)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setObjectName('mono')
        self.text.setMinimumHeight(140)
        sec.add(self.text)
        rv.addWidget(sec, 1)
        split.addWidget(right)
        split.setSizes([460, 820])

    # ------------------------------------------------------------------ the definition
    def _fill(self, combo, items):
        cur = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for i, n in items:
            combo.addItem(n, i)
        k = combo.findData(cur)
        combo.setCurrentIndex(k if k >= 0 else 0)
        combo.blockSignals(False)

    def spec(self) -> dict:
        return {'id': self.exp_id.text().strip(), 'model': self.model.currentData() or '',
                'world': self.world.currentData() or '', 'rig': self.rig.currentData() or '',
                'conditions': [c.strip() for c in self.conditions.text().split(',') if c.strip()],
                'budget': {'drives': self.drives.value(), 'frames': self.frames.value()},
                'rounds': self.rounds.value(),
                'acquisition': dict(G.DEFAULTS['acquisition'], max_cells=self.cells.value(),
                                    reference=self.reference.currentData())}

    def plan(self):
        try:
            s = G.normalise(self.spec(), self.page.win.project)
        except G.GenError as e:
            QMessageBox.warning(self, 'Generation', str(e))
            return None
        self.text.setPlainText(G.describe_plan(s))
        return s

    def register(self):
        try:
            c = G.create(self.page.win.workspace(), self.spec(), self.page.win.project)
        except G.GenError as e:
            QMessageBox.warning(self, 'Generation', str(e))
            return None
        self.refresh(select=c['id'])
        self.text.setPlainText(f'registered {c["id"]} in {tilde(c["path"])}\n\n' + G.describe_plan(c['spec']))
        return c

    def run(self):
        exp = self.exps.currentData() or self.exp_id.text().strip()
        if not exp or not os.path.isfile(os.path.join(G.exp_dir(self.page.win.workspace(), exp), 'experiment.json')):
            if self.register() is None:
                return None
            exp = self.exp_id.text().strip()
        if not self.page._ready():
            return None
        argv = ['generate', 'run', self.page.win.project.path, '--id', exp] + self.page._carla_args()
        return self.page._cli('experiment', f'Generation {exp}', argv, memory_gb=1.5)

    # ------------------------------------------------------------------ the results
    def refresh(self, select: str | None = None):
        proj = self.page.win.project
        doc = getattr(proj, 'doc', None) or {}
        self._fill(self.model, [(m['id'], m.get('name') or m['id']) for m in doc.get('models') or ()
                                if m.get('kind') == 'detector' and (m.get('source') or {}).get('training_run')])
        self._fill(self.world, [(w['id'], w.get('name') or w['id']) for w in doc.get('worlds') or ()])
        self._fill(self.rig, [(r['id'], r.get('name') or r['id']) for r in doc.get('rigs') or ()])
        from ...core.ablation import registry
        cur = select or self.exps.currentData()
        rows = [r for r in registry(self.page.win.workspace()) if r['kind'] == G.KIND]
        self._fill(self.exps, [(r['id'], f'{r["id"]} — {r["status"]}') for r in rows])
        k = self.exps.findData(cur)
        if k >= 0:
            self.exps.setCurrentIndex(k)
        self.show()

    def show(self):
        self.tree.clear()
        exp = self.exps.currentData()
        if not exp:
            self.text.setPlainText('No generation experiment yet: plan and register one on the left.')
            return
        p = os.path.join(G.exp_dir(self.page.win.workspace(), exp), 'results.json')
        if not os.path.isfile(p):
            self.text.setPlainText(f'{exp}: registered; no round has run yet (Run or resume).')
            return
        try:
            res = read_json(p)
        except (OSError, ValueError) as e:
            self.text.setPlainText(f'results.json could not be read: {e}')
            return
        span = max([abs(x) for r in res['rounds'] for c in r['chosen_cells']
                    for x in (c['diff']['ci_low'], c['diff']['ci_high']) if x is not None] or [1.0])
        self.bars.lo, self.bars.hi = -span * 1.05, span * 1.05

        def f(v):
            return '—' if v is None else f'{v:.3f}'
        for r in res['rounds']:
            for c in r['chosen_cells']:
                b, a, d = c['before'], c['after'], c['diff']
                it = QTreeWidgetItem([f'{r["round"]} · {c["cell"]}', f'{c["ev"]:.4f}',
                                      f'{f(b["value"])} [{f(b["ci_low"])}, {f(b["ci_high"])}]',
                                      f'{f(a["value"])} [{f(a["ci_low"])}, {f(a["ci_high"])}]',
                                      '—' if d['value'] is None else
                                      f'{d["value"]:+.3f} [{d["ci_low"]:+.3f}, {d["ci_high"]:+.3f}]', '',
                                      c['verdict']])
                it.setData(5, INTERVAL_ROLE, (d['value'], d['ci_low'], d['ci_high']))
                self.tree.addTopLevelItem(it)
        self.text.setPlainText(G.table(res))
