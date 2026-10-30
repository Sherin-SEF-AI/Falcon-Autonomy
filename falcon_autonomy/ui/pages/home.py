"""Home, the start page: where the project stands (worlds, rigs, scenarios, models; the recent runs, evaluations and
jobs; the server and the machine) and the next step it suggests — every item a link to where it is done. The card on
top is the "Next best action" when the assistant's suggestions have one (assistant.suggest: a tool call, reviewed in
the Assistant panel), else the next step of the work's order (next_step). Also the Status panel, the Loop page's
Status tab: the server, the machine, the measured throughput and the latest runs."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QHeaderView, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ...core import budget, sysinfo
from ...core.jobs import FINAL
from ...core.paths import tilde
from ...sim import server as S
from .. import icons, tasks
from ..widgets import EmptyState, KeyValues, Page, Section, button, label


def next_step(doc: dict, runs: int, evaluations: int) -> tuple[str, str, str, str]:
    """(headline, why, button text, page key) for the first thing this project lacks, in the order the work goes:
    a world, a rig, recorded drives, a model, an evaluation; then the search for what the model gets wrong."""
    if not doc.get('worlds'):
        return ('Add a world', 'Every drive happens in a world: a CARLA town, or a place imported from OpenStreetMap.',
                'Go to Worlds', 'worlds')
    if not doc.get('rigs'):
        return ('Add a sensor rig', 'A rig says which cameras, lidars and radars a drive records, and their labels.',
                'Go to Rigs', 'rigs')
    if runs == 0:
        return ('Record a first dataset', 'Nothing is recorded in this project yet: a drive gives images, labels and '
                'the ground truth to train and evaluate on.', 'Go to Data', 'data')
    if not doc.get('models'):
        return ('Register a model', 'Register your perception model (or add the synthetic detector to try the '
                'harness) to score it on the recorded drives.', 'Go to Models', 'models')
    if evaluations == 0:
        return ('Evaluate a model', 'Score a registered model on the recorded drives, with intervals over whole '
                'drives.', 'Go to Models', 'models')
    return ('Find where it fails', 'Search a scenario\'s parameters for the cases where the model or the stack '
            'fails, then ask why.', 'Go to Rare cases', 'rarecases')


class HomePage(Page):
    key, title, icon = 'home', 'Home', 'app'
    about = 'Where this project stands, and what to do next.'

    def __init__(self, win):
        super().__init__(win)
        v = self.page_layout()
        # ---- the next step
        self.next_card = QWidget()
        self.next_card.setObjectName('homeCard')
        nc = QHBoxLayout(self.next_card)
        nc.setContentsMargins(16, 12, 16, 12)
        col = QVBoxLayout()
        self.next_head = label('', title=True)
        self.next_why = label('', muted=True, wrap=True)
        self.next_kind = label('Next step', muted=True, small=True)
        col.addWidget(self.next_kind)
        col.addWidget(self.next_head)
        col.addWidget(self.next_why)
        nc.addLayout(col, 1)
        self.next_btn = button('', self._go_next, role='primary')
        self.next_dismiss = button('Not now', self._dismiss, role='ghost',
                                   tip='Dismiss this suggestion for this project (Settings ▸ Assistant turns '
                                       'suggestions off)')
        self.next_dismiss.hide()
        nc.addWidget(self.next_dismiss, 0, Qt.AlignmentFlag.AlignVCenter)
        nc.addWidget(self.next_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self._next_key = ''
        self.suggestion = None                       # the "Next best action" shown (assistant.suggest.Suggestion)
        self._step = ('', '', '', '')
        v.addWidget(self.next_card)
        # ---- project, server, machine
        row = QHBoxLayout()
        row.setSpacing(16)
        proj = Section('Project')
        self.counts = QGridLayout()
        self.counts.setHorizontalSpacing(16)
        self.count_btns = {}
        for i, (k, text, page) in enumerate((('worlds', 'Worlds', 'worlds'), ('rigs', 'Rigs', 'rigs'),
                                            ('scenarios', 'Scenarios', 'scenarios'), ('models', 'Models', 'models'))):
            b = button('', lambda _=False, p=page: win.go(p), role='link', tip=f'Go to {text}')
            self.count_btns[k] = (b, text)
            self.counts.addWidget(b, i // 2, i % 2)
        proj.add(self.counts)
        self.where = label('', muted=True, small=True, wrap=True, sel=True)
        proj.add(self.where)
        row.addWidget(proj, 1, Qt.AlignmentFlag.AlignTop)
        srv = Section('CARLA server')
        self.server_kv = KeyValues(('State', 'Folder'))
        srv.add(self.server_kv)
        sb = QHBoxLayout()
        self.server_btn = button('Start', lambda: win.status.toggle_server())
        sb.addWidget(self.server_btn)
        sb.addWidget(button('Probe', win.probe, tip='Ask the server for its version, map and maps (changes nothing)'))
        sb.addStretch(1)
        srv.add(sb)
        row.addWidget(srv, 1, Qt.AlignmentFlag.AlignTop)
        mach = Section('This machine')
        self.machine_kv = KeyValues(('GPU', 'Memory'))
        mach.add(self.machine_kv)
        row.addWidget(mach, 1, Qt.AlignmentFlag.AlignTop)
        v.addLayout(row)
        # ---- recent work
        lists = QHBoxLayout()
        lists.setSpacing(16)
        runs = Section('Recent runs', 'Double-click to browse a run in Data.')
        self.runs = QTreeWidget()
        self.runs.setHeaderLabels(['When', 'Run', 'Status'])
        self.runs.setRootIsDecorated(False)
        self.runs.header().setStretchLastSection(False)
        self.runs.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (0, 2):
            self.runs.header().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.runs.setMinimumHeight(200)
        self.runs.itemDoubleClicked.connect(lambda it, _c: self.open_run(it))
        from ..assistant_dock import add_ask_menu
        add_ask_menu(self.runs, win, lambda it: ('run', {'run': it.data(0, 256)}) if it.data(0, 256) else None)
        self.runs_empty = EmptyState('No runs yet.', 'Record a dataset', lambda: win.go('data'))
        runs.add(self.runs)
        runs.add(self.runs_empty)
        lists.addWidget(runs, 3)
        evs = Section('Recent evaluations', 'Double-click to open one in Models.')
        self.evals = QTreeWidget()
        self.evals.setHeaderLabels(['Evaluation'])
        self.evals.setRootIsDecorated(False)
        self.evals.setMinimumHeight(200)
        self.evals.itemDoubleClicked.connect(lambda it, _c: self.open_eval(it))
        self.evals_empty = EmptyState('No evaluations yet.', 'Evaluate a model', lambda: win.go('models'))
        evs.add(self.evals)
        evs.add(self.evals_empty)
        lists.addWidget(evs, 2)
        jobs = Section('Jobs', 'Running, waiting and finished in this session. Double-click a finished one to open '
                               'its result.')
        self.jobs = QTreeWidget()
        self.jobs.setHeaderLabels(['Job', 'State'])
        self.jobs.setRootIsDecorated(False)
        self.jobs.setMinimumHeight(200)
        self.jobs.itemDoubleClicked.connect(lambda it, _c: win.open_job(it.data(0, 256)))
        self.jobs_empty = EmptyState('No jobs in this session.')
        jobs.add(self.jobs)
        jobs.add(self.jobs_empty)
        lists.addWidget(jobs, 2)
        v.addLayout(lists)
        v.addStretch(1)
        for sig in (win.bridge.added, win.bridge.finished):
            sig.connect(lambda _job: self.refresh_jobs())
        win.bridge.finished.connect(lambda _job: self.refresh())
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_machine)
        self.timer.start(2000)
        self.refresh()

    # ------------------------------------------------------------------ state
    def evaluation_dirs(self, n=8) -> list:
        root = os.path.join(self.win.workspace(), 'evaluations')
        if not os.path.isdir(root):
            return []
        ds = [d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, 'evaluation.json'))]
        return [os.path.join(root, d) for d in sorted(ds, reverse=True)[:n]]

    def refresh(self):
        w = self.win
        doc = w.project.doc
        for k, (b, text) in self.count_btns.items():
            n = len(doc.get(k) or [])
            b.setText(f'{n}  {text.lower() if n != 1 else text.lower().rstrip("s")}')
        self.where.setText(f'{doc.get("name", "Untitled")} — workspace {tilde(w.workspace())}'
                           + ('' if w.project.path else ' (not saved yet: runs go to a temporary workspace)'))
        self.runs.clear()
        cat = w.service.catalog
        rows = cat.list('run', 10) if cat is not None else []
        for r in rows:
            it = QTreeWidgetItem([r['created'].replace('T', ' ').rstrip('Z')[:16], r['name'], r['status']])
            it.setData(0, 256, r['path'])
            it.setToolTip(1, tilde(r['path']))
            sev = {'ok': 'good', 'failed': 'critical', 'crashed': 'critical', 'cancelled': 'warning'}.get(
                r['status'], 'warning')
            it.setIcon(2, icons.status_icon(sev))
            self.runs.addTopLevelItem(it)
        self.runs.setVisible(bool(rows))
        self.runs_empty.setVisible(not rows)
        self.evals.clear()
        evs = self.evaluation_dirs()
        for d in evs:
            it = QTreeWidgetItem([os.path.basename(d)])
            it.setData(0, 256, d)
            it.setToolTip(0, tilde(d))
            self.evals.addTopLevelItem(it)
        self.evals.setVisible(bool(evs))
        self.evals_empty.setVisible(not evs)
        n_runs = sum(r['status'] == 'ok' for r in rows)           # a cancelled or failed run gives nothing to use
        self._step = next_step(doc, n_runs, len(evs))
        self._show_next()
        self.refresh_jobs()
        self.refresh_machine()

    def refresh_jobs(self):
        self.jobs.clear()
        jobs = list(reversed(self.win.queue.jobs))[:12]
        for j in jobs:
            it = QTreeWidgetItem([j.spec.title, j.state])
            it.setData(0, 256, j.id)
            sev = {'failed': 'critical', 'cancelled': 'warning', 'done': 'good'}.get(j.state)
            it.setIcon(0, icons.status_icon(sev) if sev else icons.icon('history', 12))
            it.setToolTip(0, 'Double-click to open its result' if j.state in FINAL else j.summary())
            self.jobs.addTopLevelItem(it)
        self.jobs.setVisible(bool(jobs))
        self.jobs_empty.setVisible(not jobs)

    def refresh_machine(self):
        w = self.win
        svc = w.service
        st = svc.server_state()
        self.server_kv.set('State', w.status.server.text() or st)
        root = svc.carla_root
        self.server_kv.set('Folder', f'{tilde(root)} ({S.carla_version(root) or "?"})' if root else
                           'not found: choose it in Settings')
        self.server_btn.setText('Stop' if st in ('ready', 'starting') else 'Start')
        self.server_btn.setVisible(st not in ('external', 'fake', 'shared'))
        self.machine_kv.set('GPU', w.status.gpu.text() or '—')
        self.machine_kv.set('Memory', w.status.mem.text() or '—')

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def show_suggestion(self, s):
        """The suggestions' best (or None: the work's next step again)."""
        self.suggestion = s
        self._show_next()

    def _show_next(self):
        s = self.suggestion
        if s is not None:
            self.next_kind.setText('Next best action' + (' · worded by the assistant' if s.phrased else ''))
            self.next_head.setText(s.title)
            self.next_why.setText(s.shown())
            self.next_btn.setText('Review…')
            self.next_btn.setToolTip(f'Open the Assistant panel with {s.tool} — an approval card when it changes '
                                     'something and your permission mode asks')
            self.next_dismiss.show()
            return
        head, why, text, key = self._step
        self.next_kind.setText('Next step')
        self.next_head.setText(head)
        self.next_why.setText(why)
        self.next_btn.setText(text)
        self.next_btn.setToolTip('')
        self.next_dismiss.hide()
        self._next_key = key

    # ------------------------------------------------------------------ links
    def _go_next(self):
        if self.suggestion is not None and getattr(self.win, 'suggestions', None) is not None:
            self.win.suggestions.act(self.suggestion)
        elif self._next_key:
            self.win.go(self._next_key)

    def _dismiss(self):
        if self.suggestion is not None and getattr(self.win, 'suggestions', None) is not None:
            self.win.suggestions.dismiss(self.suggestion)

    def open_run(self, it):
        data = self.win.go('data')
        if data is not None:
            data.show_run(it.data(0, 256))

    def open_eval(self, it):
        models = self.win.go('models')
        if models is not None:
            models.open_evaluation(it.data(0, 256))

    def on_run_finished(self, job):
        self.refresh()


class StatusPanel(Page):
    """Not a page of its own: the Loop page shows it as its Status tab."""
    key, title, icon = 'status', 'Status', 'app'
    about = 'The state of the server and the machine, what this machine measured, and the latest runs.'

    def __init__(self, win):
        super().__init__(win)
        v = self.page_layout()
        row = QHBoxLayout()
        row.setSpacing(24)
        v.addLayout(row)

        srv = Section('CARLA server', 'Falcon runs its own headless server on its own port, capped in memory; it never '
                                      'touches a server you started unless you choose its port in Settings.')
        self.server_kv = KeyValues(('State', 'Folder', 'Client Python', 'Last probe'))
        srv.add(self.server_kv)
        b = QHBoxLayout()
        b.addWidget(button('Start / stop', lambda: win.status.toggle_server()))
        b.addWidget(button('Probe', win.probe, tip='Ask the server for its version, map and maps (changes nothing)'))
        b.addWidget(button('Tidy', win.tidy, tip='Remove what a killed worker left and restore asynchronous mode'))
        b.addStretch(1)
        srv.add(b)
        row.addWidget(srv, 1, Qt.AlignmentFlag.AlignTop)

        mach = Section('This machine')
        self.machine_kv = KeyValues()
        mach.add(self.machine_kv)
        row.addWidget(mach, 1, Qt.AlignmentFlag.AlignTop)

        tp = Section('Throughput', 'Cost estimates use what this machine measured; until the smoke drive has run they '
                                   'use the reference machine\'s figure, labelled typical.')
        self.tp_kv = KeyValues(('Source', 'Rendering', 'Estimate'))
        tp.add(self.tp_kv)
        tb = QHBoxLayout()
        tb.addWidget(button('Run the 30 s smoke drive', self.smoke, role='primary'))
        tb.addWidget(button('First-run wizard…', win.show_wizard))
        tb.addStretch(1)
        tp.add(tb)
        v.addWidget(tp)

        runs = Section('Latest runs')
        self.runs = QTreeWidget()
        self.runs.setHeaderLabels(['When', 'Run', 'Status', 'Steps', 'Steps/s', 'Folder'])
        self.runs.setRootIsDecorated(False)
        self.runs.setMinimumHeight(180)
        self.runs.itemDoubleClicked.connect(lambda it, _c: self.open_run(it))
        runs.add(self.runs)
        v.addWidget(runs)
        v.addStretch(1)
        win.bridge.finished.connect(lambda job: self.refresh())
        self.refresh()
        self.load_machine()

    def load_machine(self):
        tasks.run(self, sysinfo.checks, self._show_machine)

    def _show_machine(self, rows):
        self.machine_kv.clear()
        for name, ok, detail in rows:
            self.machine_kv.set(name, ('' if ok else '⚠ ') + detail)

    def refresh(self):
        w = self.win
        svc = w.service
        st = svc.server_state()
        self.server_kv.set('State', f'{st} — port {svc.port}, traffic manager {svc.tm_port}')
        root = svc.carla_root
        self.server_kv.set('Folder', f'{tilde(root)} ({S.carla_version(root) or "?"})' if root else
                           'not found: choose it in Settings')
        self.server_kv.set('Client Python', tilde(svc.python) or 'not found')
        probe = next((j for j in reversed(w.queue.jobs) if j.spec.kind == 'probe' and j.result), None)
        if probe is not None:
            s = probe.result.get('summary') or {}
            self.server_kv.set('Last probe', f'CARLA {s.get("server_version")}, map {s.get("map")}, '
                                             f'{len(s.get("maps") or [])} maps, '
                                             f'{"synchronous" if s.get("synchronous_mode") else "asynchronous"}')
        t = svc.throughput()
        self.tp_kv.set('Source', t.source)
        self.tp_kv.set('Rendering', f'{t.mpx_per_s:.0f} Mpx/s, {1000 * t.overhead_s:.0f} ms per step overhead '
                                    f'({"measured" if t.measured else "typical"})')
        rig = (w.project.doc.get('rigs') or [None])[0]
        if rig:
            mpx = budget.rig_mpx(rig)
            est = t.estimate(int(60 * 20), mpx)
            self.tp_kv.set('Estimate', f'60 s at 20 fps with rig "{rig["name"]}" ({mpx:.1f} Mpx per step): about '
                                       f'{budget.fmt_duration(est["seconds"])} ({est["label"]})')
        self.runs.clear()
        if svc.catalog is not None:
            for r in svc.catalog.list('run', 12):
                s = r['summary']
                it = QTreeWidgetItem([r['created'].replace('T', ' ').rstrip('Z'), r['name'], r['status'],
                                      str(s.get('steps', '')), str(s.get('steps_per_s', '')), tilde(r['path'])])
                it.setData(0, 256, r['path'])
                sev = {'ok': 'good', 'failed': 'critical', 'crashed': 'critical', 'cancelled': 'warning'}.get(
                    r['status'], 'warning')
                it.setIcon(2, icons.status_icon(sev))
                self.runs.addTopLevelItem(it)

    def smoke(self):
        from ...core.drive import run_dir_name, smoke_config
        from ...core.project import ProjectError
        w = self.win
        try:
            cfg = smoke_config(w.project, os.path.join(w.service.runs_dir(), run_dir_name('smoke')), w.service.port,
                               w.service.tm_port)
            w.service.submit_drive(cfg, title='Smoke drive (30 s)', kind='smoke')
        except (ProjectError, RuntimeError) as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Smoke drive', str(e))
            return
        w.jobs_dock.raise_()

    def open_run(self, it):
        data = self.win.go('data')
        if data is not None:
            data.show_run(it.data(0, 256))

    def on_run_finished(self, job):
        self.refresh()
