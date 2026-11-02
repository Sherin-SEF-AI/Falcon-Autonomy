"""Home, the start page: where the project stands (worlds, rigs, scenarios, models; the recent runs, evaluations and
jobs; the server and the machine) and the next step it suggests — every item a link to where it is done. The card on
top is the "Next best action" when the assistant's suggestions have one (assistant.suggest: a tool call, reviewed in
the Assistant panel), else the next step of the work's order (next_step). Also the Status panel, the Loop page's
Status tab: the server, the machine, the measured throughput and the latest runs."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QHeaderView, QLabel, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout)

from ...core import budget, sysinfo
from ...core.jobs import FINAL
from ...core.paths import tilde
from ...sim import server as S
from .. import icons, motion, tasks
from ..theme import THEME, mono
from ..widgets import EmptyState, KeyValues, Meter, Page, Section, Stat, button, label


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


def _run_numbers(path: str):
    """(steps, real-time factor) from a run's run.json summary, or None where it does not say."""
    import json
    try:
        with open(os.path.join(path, 'run.json'), encoding='utf-8') as fh:
            s = json.load(fh).get('summary') or {}
    except (OSError, ValueError):
        return None, None
    steps, rtf = s.get('steps'), s.get('realtime_factor')
    return (steps if isinstance(steps, int) else None), (float(rtf) if isinstance(rtf, (int, float)) else None)


def _eval_numbers(path: str):
    """(model name, (AP, CI low, CI high)) of an evaluation, the AP part None when it has no AP."""
    import json
    try:
        with open(os.path.join(path, 'evaluation.json'), encoding='utf-8') as fh:
            e = json.load(fh)
    except (OSError, ValueError):
        return None, None
    name = (e.get('model') or {}).get('name')
    row = next((r for r in ((e.get('scores') or {}).get('table') or []) if r.get('metric') == 'AP'), None)
    if not row or not isinstance(row.get('value'), (int, float)):
        return name, None
    return name, (row['value'], row.get('ci_low', row['value']), row.get('ci_high', row['value']))


class StatLink(Stat):
    """A headline tile that goes somewhere when clicked (text() says what it counts, for the keyboard and tests)."""

    def __init__(self, caption, on_click, tip=''):
        super().__init__(caption)
        self.on_click = on_click
        self.link_text = caption
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tip)
        self.setAccessibleName(caption)

    def text(self) -> str:
        return self.link_text

    def click(self):
        self.on_click()

    def mouseReleaseEvent(self, ev):                            # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton and self.rect().contains(ev.position().toPoint()):
            self.click()


class HomePage(Page):
    key, title, icon = 'home', 'Home', 'app'
    about = 'Where this project stands, and what to do next.'

    def __init__(self, win):
        super().__init__(win)
        v = self.page_layout()
        # ---- the next step (the assistant's next best action, or the work's next step)
        self.next_card = QFrame()
        self.next_card.setObjectName('banner')
        nc = QHBoxLayout(self.next_card)
        nc.setContentsMargins(18, 13, 16, 13)
        nc.setSpacing(14)
        self.next_icon = QLabel()
        self.next_icon.setPixmap(icons.icon('sparkle', 18, THEME.tok['accent_hi']).pixmap(18, 18))
        nc.addWidget(self.next_icon, 0, Qt.AlignmentFlag.AlignVCenter)
        col = QVBoxLayout()
        col.setSpacing(3)
        self.next_head = label('', title=True)
        self.next_why = label('', ink2=True, wrap=True)
        self.next_kind = label('Next step')
        self.next_kind.setProperty('cap', True)
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
        # ---- the project in numbers: each tile goes to its page
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.count_btns = {}
        for k, text, page in (('worlds', 'Worlds', 'worlds'), ('rigs', 'Rigs', 'rigs'),
                              ('scenarios', 'Scenarios', 'scenarios'), ('runs', 'Recorded runs', 'data'),
                              ('models', 'Models', 'models')):
            t = StatLink(text, lambda p=page: win.go(p), tip=f'Go to {win.page(page).title if win.page(page) else text}')
            self.count_btns[k] = (t, text)
            tiles.addWidget(t, 1)
        self.ap_tile = StatLink('Latest AP', lambda: self.open_eval_path(self._latest_eval),
                                tip='The newest evaluation\'s AP with its bootstrap 95 % interval (Models opens it)')
        tiles.addWidget(self.ap_tile, 1)
        v.addLayout(tiles)
        # ---- recent runs | the server and the machine
        mid = QHBoxLayout()
        mid.setSpacing(12)
        runs = Section('Recent runs', 'Double-click a run to browse its frames in Data.')
        self.runs = QTreeWidget()
        self.runs.setHeaderLabels(['When', 'World', 'Rig', 'Steps', 'Real time', 'Status'])
        self.runs.setRootIsDecorated(False)
        self.runs.setAlternatingRowColors(True)
        hd = self.runs.header()
        hd.setStretchLastSection(False)
        for c, mode in ((0, 'fit'), (1, 'stretch'), (2, 'stretch'), (3, 'fit'), (4, 'fit'), (5, 'fit')):
            hd.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if mode == 'stretch'
                                    else QHeaderView.ResizeMode.ResizeToContents)
        for c in (3, 4):
            self.runs.headerItem().setTextAlignment(c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.runs.headerItem().setToolTip(3, 'Simulation steps recorded')
        self.runs.headerItem().setToolTip(4, 'Simulated time over wall time while recording (1.00 × = real time)')
        self.runs.setMinimumHeight(260)
        self.runs.itemDoubleClicked.connect(lambda it, _c: self.open_run(it))
        from ..assistant_dock import add_ask_menu
        add_ask_menu(self.runs, win, lambda it: ('run', {'run': it.data(0, 256)}) if it.data(0, 256) else None)
        self.runs_empty = EmptyState('No runs yet.', 'Record a dataset', lambda: win.go('data'))
        runs.add(self.runs)
        runs.add(self.runs_empty)
        mid.addWidget(runs, 3)
        side = QVBoxLayout()
        side.setSpacing(12)
        srv = Section('CARLA server')
        st = QHBoxLayout()
        st.setSpacing(8)
        self.server_dot = motion.Dot('', 8)
        self.server_state = label('', sel=True)
        st.addWidget(self.server_dot)
        st.addWidget(self.server_state, 1)
        srv.add(st)
        self.server_kv = KeyValues(('Folder',))
        srv.add(self.server_kv)
        sb = QHBoxLayout()
        self.server_btn = button('Start', lambda: win.status.toggle_server())
        sb.addWidget(self.server_btn)
        sb.addWidget(button('Probe', win.probe, role='ghost',
                            tip='Ask the server for its version, map and maps (changes nothing)'))
        sb.addStretch(1)
        srv.add(sb)
        side.addWidget(srv)
        mach = Section('This machine')
        self.meters = {}
        for k in ('GPU memory', 'GPU load', 'RAM', 'Disk (workspace)'):
            row = QHBoxLayout()
            row.addWidget(label(k, muted=True))
            row.addStretch(1)
            val = label('—', mono=True)
            row.addWidget(val)
            mach.add(row)
            m = Meter()
            mach.add(m)
            self.meters[k] = (val, m)
        self.machine_note = label('', muted=True, mono=True, small=True)
        mach.add(self.machine_note)
        side.addWidget(mach)
        side.addStretch(1)
        mid.addLayout(side, 2)
        v.addLayout(mid)
        # ---- evaluations | jobs
        lists = QHBoxLayout()
        lists.setSpacing(12)
        evs = Section('Recent evaluations', 'Double-click one to open it in Models.')
        self.evals = QTreeWidget()
        self.evals.setHeaderLabels(['Evaluation', 'Model', 'AP', '95 % CI'])
        self.evals.setRootIsDecorated(False)
        self.evals.setAlternatingRowColors(True)
        eh = self.evals.header()
        eh.setStretchLastSection(False)
        eh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            eh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        for c in (2, 3):
            self.evals.headerItem().setTextAlignment(c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.evals.setMinimumHeight(150)
        self.evals.itemDoubleClicked.connect(lambda it, _c: self.open_eval(it))
        self.evals_empty = EmptyState('No evaluations yet.', 'Evaluate a model', lambda: win.go('models'))
        evs.add(self.evals)
        evs.add(self.evals_empty)
        lists.addWidget(evs, 3)
        jobs = Section('Jobs', 'This session\'s jobs. Double-click a finished one to open its result.')
        self.jobs = QTreeWidget()
        self.jobs.setHeaderLabels(['Job', 'State'])
        self.jobs.setRootIsDecorated(False)
        self.jobs.setMinimumHeight(150)
        self.jobs.header().setStretchLastSection(False)
        self.jobs.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.jobs.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.jobs.itemDoubleClicked.connect(lambda it, _c: win.open_job(it.data(0, 256)))
        self.jobs_empty = EmptyState('No jobs in this session. Record a dataset, train or evaluate: its progress '
                                     'shows here.')
        jobs.add(self.jobs)
        jobs.add(self.jobs_empty)
        lists.addWidget(jobs, 2)
        v.addLayout(lists)
        v.addStretch(1)
        self._latest_eval = ''
        self._shown_once = False
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
        worlds = doc.get('worlds') or []
        towns = sum(x.get('kind') == 'town' for x in worlds)
        rigs = doc.get('rigs') or []
        cat = w.service.catalog
        every = cat.list('run', 100000) if cat is not None else []
        ok = sum(r['status'] == 'ok' for r in every)
        from ...core.scenarios import LIBRARY
        models = doc.get('models') or []
        tiles = {
            'worlds': (len(worlds), f'{len(worlds) - towns} imported · {towns} CARLA town{"s" * (towns != 1)}'
                       if worlds else 'none yet'),
            'rigs': (len(rigs), f'{sum(len(r.get("sensors") or []) for r in rigs)} sensors in all' if rigs
                     else 'none yet'),
            'scenarios': (len(doc.get('scenarios') or []), f'yours · {len(LIBRARY)} built-in'),
            'runs': (len(every), f'{ok} ok' + (f' · {len(every) - ok} not' if len(every) > ok else '') if every
                     else 'none yet'),
            'models': (len(models), ', '.join(sorted({m.get('kind', '') for m in models} - {''})) or 'none yet'),
        }
        for k, (t, text) in self.count_btns.items():
            n, sub = tiles[k]
            t.set(n, sub=sub)
            t.link_text = f'{n} {text.lower() if n != 1 else text.lower().rstrip("s")}'
        self.runs.clear()
        rows = every[:10]
        for r in rows:
            world, _, rig = r['name'].partition(' / ')
            steps, rtf = _run_numbers(r['path'])
            it = QTreeWidgetItem([r['created'].replace('T', ' ').rstrip('Z')[:16], world, rig or '—',
                                  '—' if steps is None else f'{steps:,}'.replace(',', ' '),
                                  '—' if rtf is None else f'{rtf:.2f} ×', r['status']])
            it.setData(0, 256, r['path'])
            it.setToolTip(1, tilde(r['path']))
            for c in (0, 3, 4):
                it.setFont(c, mono(12))
            for c in (3, 4):
                it.setTextAlignment(c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for c in (2, 3, 4):
                it.setForeground(c, THEME.color('ink2'))
            sev = {'ok': 'good', 'failed': 'critical', 'crashed': 'critical', 'cancelled': 'warning'}.get(
                r['status'], 'warning')
            it.setIcon(5, icons.status_icon(sev))
            self.runs.addTopLevelItem(it)
        self.runs.setVisible(bool(rows))
        self.runs_empty.setVisible(not rows)
        self.evals.clear()
        evs = self.evaluation_dirs()
        self._latest_eval = ''
        latest = None
        for d in evs:
            name, ap = _eval_numbers(d)
            it = QTreeWidgetItem([os.path.basename(d), name or '—',
                                  '—' if ap is None else f'{ap[0]:.3f}',
                                  '—' if ap is None else f'{ap[1]:.3f} – {ap[2]:.3f}'])
            it.setData(0, 256, d)
            it.setToolTip(0, tilde(d))
            for c in (0, 2, 3):
                it.setFont(c, mono(12))
            for c in (2, 3):
                it.setTextAlignment(c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            it.setForeground(1, THEME.color('ink2'))
            it.setForeground(3, THEME.color('ink2'))
            self.evals.addTopLevelItem(it)
            if latest is None and ap is not None:
                latest, self._latest_eval = (name, ap), d
        self.evals.setVisible(bool(evs))
        self.evals_empty.setVisible(not evs)
        if latest is None:
            self.ap_tile.set('—', sub='no evaluation yet')
        else:
            name, (v, lo, hi) = latest
            synthetic = 'synthetic' in (name or '').lower() or 'testing only' in (name or '').lower()
            self.ap_tile.set(f'{v:.3f}', sub=('synthetic · ' if synthetic else '') + f'CI {lo:.3f}–{hi:.3f}',
                             tone='warning' if synthetic else '')
            self.ap_tile.sub.setToolTip(f'{name}: bootstrap 95 % interval {lo:.3f} – {hi:.3f}'
                                        + (' (a synthetic test model: says nothing about a real detector)'
                                           if synthetic else ''))
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
        from ..frame import SERVER_LOOK
        tone, pulse = SERVER_LOOK.get(st, ('muted', False))
        self.server_dot.set_color(THEME.tok.get(tone) or THEME.color(tone).name(), pulse)
        self.server_state.setText(w.status.server.text() or st)
        root = svc.carla_root
        self.server_kv.set('Folder', f'{tilde(root)} ({S.carla_version(root) or "?"})' if root else
                           'not found: choose it in Settings')
        self.server_btn.setText('Stop' if st in ('ready', 'starting') else 'Start')
        self.server_btn.setVisible(st not in ('external', 'fake', 'shared'))
        g = getattr(w.status, 'gpu_vals', None)                  # (name, used MiB, total MiB, load %)
        if g:
            self._meter('GPU memory', f'{g[1] / 1024:.1f} / {g[2] / 1024:.0f} GB', g[1] / g[2] if g[2] else None)
            self._meter('GPU load', f'{g[3]:.0f} %', g[3] / 100)
            self.machine_note.setText(g[0])
        else:
            self._meter('GPU memory', w.status.gpu.text() or '—', None)
            self._meter('GPU load', '—', None)
        m = sysinfo.meminfo()
        if m.get('total_gb'):
            used = m['total_gb'] - m['available_gb']
            self._meter('RAM', f'{used:.1f} / {m["total_gb"]:.0f} GB', used / m['total_gb'])
        try:
            import shutil
            du = shutil.disk_usage(w.workspace() if os.path.isdir(w.workspace()) else os.path.expanduser('~'))
            self._meter('Disk (workspace)', f'{du.used / 1e9:.0f} GB / {du.total / 1e12:.1f} TB' if du.total > 1e12
                        else f'{du.used / 1e9:.0f} / {du.total / 1e9:.0f} GB', du.used / du.total)
        except OSError:
            self._meter('Disk (workspace)', '—', None)

    def _meter(self, key, text, frac):
        val, m = self.meters[key]
        val.setText(text)
        m.set(frac or 0.0)
        m.setVisible(frac is not None)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()
        if not self._shown_once:
            self._shown_once = True
            motion.stagger([self.next_card] + [t for t, _ in self.count_btns.values()] + [self.ap_tile])

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
        self.open_eval_path(it.data(0, 256))

    def open_eval_path(self, path):
        models = self.win.go('models')
        if models is not None and path:
            models.open_evaluation(path)

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
