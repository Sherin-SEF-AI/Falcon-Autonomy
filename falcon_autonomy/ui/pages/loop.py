"""Loop, the home page: the closed loop of core.loop — choose a model and a target ODD, see the eight steps with their
state, the cost of the next step before it runs, run one step or up to a step as a background job (`falcon loop`,
cancellable and resumable), the weak cells, the before and after with intervals, and the iteration history. Its Status
tab holds what the first page always showed: the server, the machine, the measured throughput and the latest runs."""
from __future__ import annotations

import json
import os
import sys

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QPlainTextEdit, QSpinBox, QTabWidget,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import loop as L
from ...core.jobs import FINAL, JobSpec
from ...core.project import Edit
from .. import icons, tasks
from ..theme import STATUS, THEME
from ..widgets import KeyValues, Page, Section, button, label, scroll
from .home import StatusPanel


def _f(v, signed=False):
    if v is None:
        return '—'
    return f'{v:+.3f}' if signed else f'{v:.3f}'


def _ci(v):
    v = v or {}
    if v.get('value') is None:
        return '—'
    if v.get('ci_low') is None:
        return _f(v['value'])
    return f'{_f(v["value"])} [{_f(v["ci_low"])}, {_f(v["ci_high"])}]'


class BeforeAfterChart(QWidget):
    """Each row's before (blue circle) and after (orange square) with its 95 % interval as a line, on 0–1; the
    legend names both, so colour is never the only cue."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.setMinimumHeight(120)

    def set_rows(self, rows):
        self.rows = list(rows or [])
        self.setMinimumHeight(60 + 34 * max(len(self.rows), 1))
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        if not self.rows:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No before and after yet: it is step 8')
            return
        lw = min(300, self.width() // 3)
        x0, x1 = lw + 10, self.width() - 20
        top, rh = 12, 34

        def X(v):
            return x0 + (x1 - x0) * min(max(float(v), 0.0), 1.0)
        bottom = top + rh * len(self.rows)
        for k in range(5):
            x = X(k / 4)
            p.setPen(QColor(THEME.tok['grid2']))
            p.drawLine(QPointF(x, top - 4), QPointF(x, bottom))
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(QRectF(x - 20, bottom + 2, 40, 14), Qt.AlignmentFlag.AlignCenter, f'{k / 4:g}')
        blue, orange = THEME.cam(0), THEME.cam(1)
        for i, r in enumerate(self.rows):
            y = top + rh * i + rh / 2
            p.setPen(QColor(THEME.tok['ink2']))
            p.drawText(QRectF(0, y - 9, lw, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(r['label'], Qt.TextElideMode.ElideMiddle, lw - 4))
            for key, col, dy in (('before', blue, -6), ('after', orange, 6)):
                v = r.get(key) or {}
                if v.get('value') is None:
                    continue
                if v.get('ci_low') is not None and v.get('ci_high') is not None:
                    p.setPen(QPen(col, 2))
                    p.drawLine(QPointF(X(v['ci_low']), y + dy), QPointF(X(v['ci_high']), y + dy))
                p.setPen(QPen(QColor(THEME.tok['panel']), 1.5))
                p.setBrush(col)
                if key == 'before':
                    p.drawEllipse(QPointF(X(v['value']), y + dy), 5, 5)
                else:
                    p.drawRect(QRectF(X(v['value']) - 5, y + dy - 5, 10, 10))
            verdict = r.get('verdict') or ''
            sev = {'improvement': 'good', 'regression': 'critical'}.get(verdict)
            if sev:
                p.setPen(QColor(STATUS[sev]))
                p.drawText(QRectF(x1 - 120, y - 20, 120, 12), Qt.AlignmentFlag.AlignRight, verdict)
        p.setBrush(Qt.BrushStyle.NoBrush)
        ly = bottom + 30
        p.setPen(QPen(QColor(THEME.tok['panel']), 1.5))
        p.setBrush(blue)
        p.drawEllipse(QPointF(x0 + 5, ly - 4), 5, 5)
        p.setBrush(orange)
        p.drawRect(QRectF(x0 + 95, ly - 9, 10, 10))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QColor(THEME.tok['muted']))
        p.drawText(QPointF(x0 + 14, ly), 'before')
        p.drawText(QPointF(x0 + 110, ly), 'after')
        p.drawText(QPointF(x0 + 160, ly), 'lines: 95 % intervals (bootstrap over drives); same evaluation set')


class OddDialog(QDialog):
    """The ODD as JSON, checked against the project before it is accepted."""

    def __init__(self, parent, project, odd):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle('Operational design domain')
        self.resize(720, 620)
        v = QVBoxLayout(self)
        v.addWidget(label('Worlds, weathers (each with a name; "ranges" draw a field per drive, seeded), the traffic '
                          'profile, the classes of interest and range bins, and the settings of each step. '
                          'docs/loop.md documents every field.', muted=True, wrap=True))
        self.text = QPlainTextEdit(json.dumps(odd or {}, indent=2, ensure_ascii=False))
        self.text.setProperty('mono', True)
        v.addWidget(self.text, 1)
        self.problems = label('', muted=True, wrap=True)
        v.addWidget(self.problems)
        row = QHBoxLayout()
        row.addWidget(button('Bengaluru monsoon-dusk example', self.example))
        row.addWidget(button('Check', self.check))
        row.addStretch(1)
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept_if_valid)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.odd = None

    def example(self):
        odd, missing = L.example_odd(self.project)
        self.text.setPlainText(json.dumps(odd, indent=2, ensure_ascii=False))
        self.problems.setText('The example needs: ' + '; '.join(missing) if missing else
                              'The project has what the example needs.')

    def parse(self):
        try:
            return json.loads(self.text.toPlainText()), None
        except ValueError as e:
            return None, f'not JSON: {e}'

    def check(self) -> bool:
        odd, err = self.parse()
        if err:
            self.problems.setText(err)
            return False
        _, errs = L.normalise_odd(odd, self.project)
        self.problems.setText('; '.join(errs) if errs else 'Valid.')
        return not errs

    def accept_if_valid(self):
        odd, err = self.parse()
        if err:
            self.problems.setText(err)
            return
        self.check()                       # an ODD with problems is kept (the page shows them) but said so
        self.odd = odd
        self.accept()


class LoopPage(Page):
    key, title, icon = 'loop', 'Loop', 'refresh'
    about = ('Close the loop: evaluate a model on a target ODD, find its weak cells, search for rare cases there, '
             'generate targeted data, retrain and evaluate again — with the before and after, with intervals, and '
             'the cost of each step shown before it runs.')

    def __init__(self, win):
        super().__init__(win)
        self.job = None
        self.it = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        v.addWidget(self.tabs)
        self.tabs.addTab(self._loop_tab(), 'Loop')
        self.status = StatusPanel(win)
        self.tabs.addTab(self.status, 'Status')
        win.bridge.finished.connect(self._job_finished)
        self.refresh()

    # ------------------------------------------------------------------ building
    def _loop_tab(self):
        body = QWidget()
        body.setObjectName('base')
        v = QVBoxLayout(body)
        v.setContentsMargins(24, 18, 24, 18)
        v.setSpacing(10)
        v.addWidget(label(self.title, title=True))
        v.addWidget(label(self.about, muted=True, wrap=True))

        sec = Section('Model and ODD', 'The model each new iteration starts with (after a finished iteration, the '
                                       'model it retrained) and the operational design domain it is tested on.')
        row = QHBoxLayout()
        self.model = QComboBox()
        self.model.setMinimumWidth(260)
        self.model.activated.connect(self.set_model)
        row.addWidget(label('Model', muted=True))
        row.addWidget(self.model)
        row.addWidget(button('Edit the ODD…', self.edit_odd))
        row.addWidget(button('Use the Bengaluru monsoon-dusk example', self.use_example))
        row.addStretch(1)
        sec.add(row)
        self.odd_kv = KeyValues(('ODD', 'Cells', 'Classes', 'Evaluation set', 'Problems'))
        sec.add(self.odd_kv)
        v.addWidget(sec)

        sec = Section('Iteration', 'Each step runs as a job (the Jobs dock shows its progress and ETA; cancel it '
                                   'there or here). A finished step whose outputs are unchanged is never run again.')
        self.head = label('', wrap=True)
        sec.add(self.head)
        self.steps = QTreeWidget()
        self.steps.setHeaderLabels(['#', 'Step', 'State', 'Estimated before', 'Measured', 'Decisions'])
        self.steps.setRootIsDecorated(False)
        self.steps.setMinimumHeight(230)
        self.steps.setColumnWidth(0, 30)
        self.steps.setColumnWidth(1, 180)
        self.steps.setColumnWidth(3, 200)
        sec.add(self.steps)
        self.next_cost = label('', wrap=True)
        sec.add(self.next_cost)
        row = QHBoxLayout()
        self.run_next = button('Run the next step', self.next_step, role='primary')
        row.addWidget(self.run_next)
        self.until = QSpinBox()
        self.until.setRange(1, len(L.STEPS))
        self.until.setValue(len(L.STEPS))
        self.until.setPrefix('to step ')
        row.addWidget(self.until)
        self.run_to = button('Run', self.run_until)
        row.addWidget(self.run_to)
        self.cancel_b = button('Cancel', self.cancel)
        row.addWidget(self.cancel_b)
        row.addWidget(button('New iteration', self.new_iteration,
                             tip='Leave the unfinished iteration and start a new one with the project\'s model and '
                                 'ODD'))
        row.addWidget(button('Report…', self.report, tip='An HTML report of this iteration (Reports page)'))
        row.addStretch(1)
        sec.add(row)
        v.addWidget(sec)

        sec = Section('Weak cells', L.WEAK_RULE[0].upper() + L.WEAK_RULE[1:] + '.')
        self.cells = QTreeWidget()
        self.cells.setHeaderLabels(['Cell', 'AP', '95 % interval', 'Objects', 'Weak', 'Chosen'])
        self.cells.setRootIsDecorated(False)
        self.cells.setMinimumHeight(150)
        self.cells.setColumnWidth(0, 340)
        sec.add(self.cells)
        v.addWidget(sec)

        sec = Section('Before and after', 'The same evaluation set before and after retraining; the difference has '
                                          'a paired bootstrap interval over drives. A verdict needs the whole '
                                          'interval on one side of zero.')
        self.chart = BeforeAfterChart()
        sec.add(self.chart)
        self.diff = QTreeWidget()
        self.diff.setHeaderLabels(['', 'Before', 'After', 'Difference (paired)', 'Verdict'])
        self.diff.setRootIsDecorated(False)
        self.diff.setMinimumHeight(110)
        for k, wd in enumerate((300, 190, 190, 240)):
            self.diff.setColumnWidth(k, wd)
        sec.add(self.diff)
        v.addWidget(sec)

        sec = Section('History', 'Every iteration is a versioned record: workspace/loop/NNN/iteration.json.')
        self.hist = QTreeWidget()
        self.hist.setHeaderLabels(['#', 'Status', 'Model', 'ODD', 'AP before', 'AP after', 'Difference',
                                   'Created'])
        self.hist.setRootIsDecorated(False)
        self.hist.setMinimumHeight(120)
        for k, wd in enumerate((40, 90, 220, 200, 170, 170, 170)):
            self.hist.setColumnWidth(k, wd)
        self.hist.itemDoubleClicked.connect(lambda it, _c: self.show_iteration(it.data(0, 256)))
        sec.add(self.hist)
        v.addWidget(sec)
        v.addStretch(1)
        return scroll(body)

    # ------------------------------------------------------------------ the project's settings
    def ws(self):
        return self.win.workspace()

    def set_model(self, _i=None):
        mid = self.model.currentData()
        if mid and mid != (self.win.project.doc.get('loop') or {}).get('model'):
            self.win.edit(Edit('set', ['loop', 'model'], mid, text='Choose the loop\'s model'))

    def set_odd(self, odd):
        self.win.edit(Edit('set', ['loop', 'odd'], odd, text='Set the loop\'s ODD'))

    def edit_odd(self):
        d = OddDialog(self, self.win.project, (self.win.project.doc.get('loop') or {}).get('odd'))
        if d.exec() and d.odd is not None:
            self.set_odd(d.odd)

    def use_example(self):
        odd, missing = L.example_odd(self.win.project)
        self.set_odd(odd)
        if missing:
            self.odd_kv.set('Problems', 'The example needs: ' + '; '.join(missing))

    # ------------------------------------------------------------------ jobs
    def _ready(self) -> bool:
        w = self.win
        if not w.project.path or w.project.dirty:
            return w.save()
        return True

    def _submit(self, args, title):
        if self.job is not None and self.job.state not in FINAL:
            return None
        if not self._ready():
            return None
        w = self.win
        argv = [sys.executable, '-m', 'falcon_autonomy', 'loop', w.project.path] + list(args) + ['--no-save']
        if w.fake_carla:
            argv += ['--fake-carla', w.fake_carla]
        self.job = w.queue.submit(JobSpec('loop', title, argv, memory_gb=1.0, resumable=True))
        w.jobs_dock.raise_()
        self._buttons()
        return self.job

    def next_step(self):
        return self._submit(['next'], 'Loop: the next step')

    def run_until(self, n=None):
        n = int(n or self.until.value())
        return self._submit(['run', '--until', str(n)], f'Loop: run to step {n} ({L.STEP_TITLES[L.STEPS[n - 1]]})')

    def new_iteration(self):
        return self._submit(['new'], 'Loop: a new iteration')

    def cancel(self):
        if self.job is not None and self.job.state not in FINAL:
            self.win.queue.cancel(self.job)

    def _job_finished(self, job):
        if job is not self.job:
            return
        for e in L.sync_edits(self.win.project):         # the iterations and what they made, as undoable edits
            self.win.edit(e)
        self.refresh()
        reports = self.win.page('reports')
        if reports is not None:
            reports.refresh()

    def _buttons(self):
        busy = self.job is not None and self.job.state not in FINAL
        for b in (self.run_next, self.run_to):
            b.setEnabled(not busy)
        self.cancel_b.setEnabled(busy)

    def report(self):
        if self.it is None:
            return
        page = self.win.page('reports')
        if page is not None:
            page.make('iteration', L.iteration_dir(self.ws(), self.it['n']))
            self.win.go('reports')

    # ------------------------------------------------------------------ showing
    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def refresh(self):
        if not hasattr(self, 'steps'):
            return
        proj = self.win.project
        loop = proj.doc.get('loop') or {}
        self.model.clear()
        for m in proj.doc.get('models', []):
            if m.get('kind') == 'detector':
                self.model.addItem(m['name'], m['id'])
        i = self.model.findData(loop.get('model'))
        self.model.setCurrentIndex(i)
        odd, errs = L.normalise_odd(loop.get('odd'), proj)
        if odd:
            cells = L.odd_cells(odd)
            ev = odd['evaluation']
            traffic = odd['traffic_profile'] or 'each world\'s own'
            self.odd_kv.set('ODD', f'{odd["name"]} ({odd["id"]}); traffic profile {traffic}')
            self.odd_kv.set('Cells', ', '.join(c['label'] for c in cells))
            self.odd_kv.set('Classes', f'{", ".join(odd["classes"])}; range bins '
                                       f'{", ".join(f"{b:g}" for b in odd["range_bins"])} m and beyond')
            self.odd_kv.set('Evaluation set', f'{len(cells) * int(ev["drives_per_cell"])} drives of '
                                              f'{ev["drive_s"]:g} s ({ev["drives_per_cell"]} per cell), fixed seeds; '
                                              'never trained on')
        else:
            for k in ('ODD', 'Cells', 'Classes', 'Evaluation set'):
                self.odd_kv.set(k, '—')
        self.odd_kv.set('Problems', '; '.join(errs) if errs else 'none')
        its = L.iterations(self.ws()) if proj.path or os.path.isdir(self.ws()) else []
        self.it = its[-1] if its else None
        self._show_iteration(self.it)
        self.hist.clear()
        for it in reversed(its):
            rows = {r['label']: r for r in (it.get('summary') or {}).get('rows', [])}
            ap = rows.get('AP, all') or {}
            before = ap.get('before') or (it['steps']['evaluate'].get('result') or {}).get('AP')
            t = QTreeWidgetItem([str(it['n']), it['status'], str(it['model'].get('name')), it['odd']['name'],
                                 _ci(before), _ci(ap.get('after')),
                                 _ci(ap.get('diff')) if ap else '—', it['created'].replace('T', ' ').rstrip('Z')])
            t.setData(0, 256, it['n'])
            if it.get('stopped'):
                t.setToolTip(1, it['stopped'])
            self.hist.addTopLevelItem(t)
        self._buttons()
        self._estimate()

    def show_iteration(self, n):
        try:
            self._show_iteration(L.load_iteration(self.ws(), int(n)))
        except (OSError, ValueError):
            return

    def _show_iteration(self, it):
        ws = self.ws()
        self.steps.clear()
        self.cells.clear()
        self.diff.clear()
        if it is None:
            self.head.setText('No iteration yet. Choose a model and an ODD, then run: step 1 freezes them.')
            for k, s in enumerate(L.STEPS, 1):
                self.steps.addTopLevelItem(QTreeWidgetItem([str(k), L.STEP_TITLES[s], 'pending', '', '', '']))
            self.chart.set_rows([])
            return
        self.head.setText(f'Iteration {it["n"]} — {it["status"]}: {it["model"].get("name")} ({it.get("model_why")}) '
                          f'on {it["odd"]["name"]}' + (f'. Stopped: {it["stopped"]}' if it.get('stopped') else ''))
        for k, s in enumerate(L.STEPS, 1):
            rec = it['steps'][s]
            state = rec.get('state', 'pending')
            if state == 'done' and not L.step_done(ws, it, s):
                state = 'to run again (an input changed)'
            est = rec.get('estimate')
            meas = (rec.get('measured') or {}).get('wall_s')
            dec = '; '.join(rec.get('decisions') or []) or (rec.get('error') or '')
            t = QTreeWidgetItem([str(k), L.STEP_TITLES[s], state, L.describe_estimate(est).split(' — ')[0]
                                 if est else '', f'{meas:.1f} s' if meas is not None else '', dec])
            t.setToolTip(3, L.describe_estimate(est) if est else '')
            t.setToolTip(5, dec)
            sev = {'done': 'good', 'failed': 'critical', 'stopped': 'serious', 'cancelled': 'warning',
                   'running': 'warning'}.get(state)
            if sev:
                t.setIcon(2, icons.status_icon(sev))
            self.steps.addTopLevelItem(t)
        wcp = os.path.join(L.iteration_dir(ws, it['n']), 'weak_cells.json')
        if os.path.isfile(wcp):
            try:
                wc = json.load(open(wcp, encoding='utf-8'))
            except (OSError, ValueError):
                wc = None
            if wc:
                rows = sorted([r for r in wc['rows'] if r['ci_high'] is not None and r['n_objects']],
                              key=lambda r: (r.get('rank') or 99, r['ci_high'], -r['n_objects']))[:40]
                for r in rows:
                    t = QTreeWidgetItem([L._cell_name(r), _f(r['AP']), f'[{_f(r["ci_low"])}, {_f(r["ci_high"])}]',
                                         str(r['n_objects']), 'yes' if r.get('weak') else '',
                                         (f'#{r["rank"]}' + (' (fallback)' if r.get('fallback') else ''))
                                         if r.get('rank') else ''])
                    self.cells.addTopLevelItem(t)
        rows = (it.get('summary') or {}).get('rows') or []
        self.chart.set_rows(rows)
        for r in rows:
            d = r['diff']
            t = QTreeWidgetItem([r['label'], _ci(r['before']), _ci(r['after']),
                                 f'{_f(d["value"], True)} [{_f(d["ci_low"], True)}, {_f(d["ci_high"], True)}]',
                                 r['verdict']])
            self.diff.addTopLevelItem(t)

    def _estimate(self):
        proj, it = self.win.project, self.it
        thr = self.win.service.throughput()
        budget = float(self.win.settings.get('render_budget_mpx'))
        ws = self.ws()

        def work():
            if it is not None and it['status'] in ('running', 'failed', 'cancelled'):
                nxt = L.next_step(ws, it)
                return nxt, (L.estimate(proj, it, nxt, thr, budget_mpx=budget) if nxt else None)
            return 'pick', L.estimate(proj, None, 'pick', thr)

        def done(res):
            nxt, est = res
            if nxt is None:
                self.next_cost.setText('Every step of this iteration is done; the next run starts a new iteration.')
                return
            what = 'a new iteration\'s ' if it is None or it['status'] not in ('running', 'failed', 'cancelled') \
                else ''
            self.next_cost.setText(f'Next: {what}step {L.STEPS.index(nxt) + 1}, {L.STEP_TITLES[nxt]} — '
                                   f'{L.describe_estimate(est)}')

        def failed(e):
            self.next_cost.setText(f'Next step: its cost could not be estimated ({e})')
        self.next_cost.setText('Estimating the next step…')
        tasks.run(self, work, done, failed)
