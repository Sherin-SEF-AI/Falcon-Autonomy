"""The main window: a left navigator of pages in groups (Home first), the Jobs and Log docks (shown while there is
work, behind a status-bar badge otherwise), the status bar (jobs, server, GPU, memory), undo and redo for every project
edit, the command palette (Ctrl+K), dark and light themes."""
from __future__ import annotations

import os

from PyQt6.QtCore import QProcess, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QIcon, QKeySequence, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                             QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QSizePolicy,
                             QStackedWidget, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..core import protocol as P
from ..core import sysinfo
from ..core.jobs import FINAL, RUNNING, WAITING, JobQueue
from ..core.paths import PROJECT_SUFFIX, tilde
from ..core.project import Edit, Project, ProjectError
from ..core.settings import MEMORY_RESERVE_GB
from ..sim.service import CarlaService
from . import icons, tasks
from .qtjobs import JobsBridge, QProcessLauncher
from .theme import THEME, sans
from .widgets import tidy


class EditCommand(QUndoCommand):
    """One project Edit on the undo stack. Consecutive edits of the same value with the same merge key (a slider
    drag) merge into one step."""

    def __init__(self, win, edit: Edit, merge_key: str = ''):
        super().__init__(edit.text)
        self.win, self.edit, self.inv = win, edit, None
        self.merge_key = merge_key

    def redo(self):
        self.inv = self.edit.apply(self.win.project.doc)
        self.win.project.dirty = True
        self.win.projectChanged.emit(self.edit.path)

    def undo(self):
        self.inv.apply(self.win.project.doc)
        self.win.project.dirty = True
        self.win.projectChanged.emit(self.edit.path)

    def id(self):
        return 1 if self.merge_key else -1

    def mergeWith(self, other):
        if not isinstance(other, EditCommand) or other.merge_key != self.merge_key or not self.merge_key \
                or other.edit.kind != 'set' or self.edit.kind != 'set':
            return False
        self.edit = Edit('set', self.edit.path, other.edit.value, text=self.edit.text)   # keep the first inverse
        return True


class JobsDock(QWidget):
    COLS = ('Job', 'State', 'Progress', 'Time', '')

    def __init__(self, win):
        super().__init__()
        self.win = win
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(self.COLS)
        self.tree.setRootIsDecorated(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.tree.setColumnWidth(1, 260)
        self.tree.setColumnWidth(2, 120)
        self.tree.itemDoubleClicked.connect(lambda it, _c: self.win.open_job(it.data(0, 256)))
        v.addWidget(self.tree)
        self.items: dict = {}
        b = win.bridge
        b.added.connect(self.update_job)
        b.changed.connect(self.update_job)
        b.finished.connect(self.update_job)

    def update_job(self, job):
        it = self.items.get(job.id)
        if it is None:
            it = QTreeWidgetItem([job.spec.title, '', '', '', ''])
            it.setData(0, 256, job.id)
            self.tree.insertTopLevelItem(0, it)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setProperty('inline', True)
            self.tree.setItemWidget(it, 2, bar)
            btn = QToolButton()
            btn.setIcon(icons.icon('close', 12))
            btn.setToolTip('Cancel this job (it tidies up first)')
            btn.clicked.connect(lambda _=False, j=job: self.win.queue.cancel(j) if j.state not in FINAL
                                else self.win.open_job(j.id))
            self.tree.setItemWidget(it, 4, btn)
            self.items[job.id] = it
        it.setText(1, job.summary() if job.state in (RUNNING, WAITING) else
                   (job.state if job.exit_code in (None, 0) else f'{job.state}: {P.describe_exit(job.exit_code)}'))
        it.setToolTip(1, job.summary())
        bar = self.tree.itemWidget(it, 2)
        if bar is not None:
            bar.setValue(int(1000 * job.progress))
        el = job.elapsed
        it.setText(3, f'{el:.0f} s' if el else '')
        btn = self.tree.itemWidget(it, 4)
        if btn is not None and job.state in FINAL and not btn.property('open'):
            btn.setProperty('open', True)
            btn.setIcon(QIcon())
            btn.setText('Open')
            btn.setProperty('text-button', True)
            btn.setToolTip('Go to where this job\'s result is shown')
        sev = {'failed': 'critical', 'cancelled': 'warning', 'done': 'good'}.get(job.state)
        it.setIcon(0, icons.status_icon(sev) if sev else icons.icon('history', 12))


class LogDock(QPlainTextEdit):
    def __init__(self, win):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.seen: dict = {}
        win.bridge.changed.connect(self.on_job)
        win.bridge.finished.connect(self.on_job)

    def append_line(self, text):
        self.appendPlainText(text)

    def on_job(self, job):
        n = self.seen.get(job.id, 0)
        for lvl, text in job.log[n:]:
            if text:
                self.appendPlainText(f'[{job.spec.title}] {lvl}: {text}' if lvl != 'info' else
                                     f'[{job.spec.title}] {text}')
        self.seen[job.id] = len(job.log)
        if job.state in FINAL and not getattr(job, '_logged_end', False):
            job._logged_end = True
            self.appendPlainText(f'[{job.spec.title}] {job.state} ({P.describe_exit(job.exit_code)}) after '
                                 f'{job.elapsed:.0f} s')


class StatusBarWidgets:
    """Server, GPU and memory in the status bar, refreshed every 2 s; the GPU query runs as a QProcess."""

    def __init__(self, win):
        self.win = win
        sb = win.statusBar()
        self.jobs = QToolButton()
        self.jobs.setObjectName('jobsBadge')
        self.jobs.setToolTip('Show or hide the Jobs and Log panel')
        self.jobs.clicked.connect(win.toggle_jobs)
        sb.addWidget(self.jobs)
        self.server = QLabel()
        self.server_btn = QToolButton()
        self.server_btn.setProperty('text-button', True)
        self.server_btn.clicked.connect(self.toggle_server)
        self.gpu = QLabel()
        self.mem = QLabel()
        for w in (self.server, self.server_btn):
            sb.addWidget(w)
        sb.addPermanentWidget(self.gpu)
        sb.addPermanentWidget(self.mem)
        self.smi = None
        self.timer = QTimer(win)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        self.refresh()

    def refresh_jobs(self):
        q = self.win.queue
        running = sum(j.state == RUNNING for j in q.jobs)
        waiting = sum(j.state not in FINAL and j.state != RUNNING for j in q.jobs)
        parts = [f'{running} running'] * bool(running) + [f'{waiting} waiting'] * bool(waiting)
        self.jobs.setText('Jobs' + (': ' + ', '.join(parts) if parts else ''))
        self.jobs.setProperty('busy', bool(parts))
        self.jobs.style().unpolish(self.jobs)
        self.jobs.style().polish(self.jobs)

    def refresh(self):
        self.refresh_jobs()
        svc = self.win.service
        st = svc.server_state()
        text = {'ready': f'CARLA ready · port {svc.port}', 'starting': f'CARLA starting · port {svc.port}',
                'stopped': f'CARLA stopped · port {svc.port}', 'crashed': f'CARLA crashed · port {svc.port}',
                'external': f'CARLA (yours) · port {svc.port}', 'fake': 'fake CARLA (tests)',
                'shared': f'CARLA ready (another Falcon process\'s) · port {svc.port}'}.get(st, st)
        self.server.setText(text)
        self.server.setToolTip(svc.server_error or (svc.server.log_tail(6) if svc.server and st == 'crashed' else
                                                    'Falcon\'s own CARLA server (headless, memory-capped)'))
        own = st in ('ready', 'starting')
        self.server_btn.setText('Stop' if own else 'Start')
        self.server_btn.setVisible(st not in ('external', 'fake', 'shared'))
        m = sysinfo.meminfo()
        self.mem.setText(f'RAM {m["available_gb"]:.1f} GB free of {m["total_gb"]:.0f}')
        self.mem.setToolTip(f'MemAvailable. Falcon keeps {MEMORY_RESERVE_GB:g} GB free and queues jobs rather than '
                            'overcommit.')
        self.query_gpu()

    def query_gpu(self):
        if self.smi is not None:
            return
        import shutil
        smi = shutil.which('nvidia-smi')
        if not smi:
            self.gpu.setText('no NVIDIA GPU')
            return
        p = QProcess(self.win)
        self.smi = p

        def done(*_):
            out = bytes(p.readAllStandardOutput()).decode(errors='replace').strip().splitlines()
            self.smi = None
            p.deleteLater()
            if not out:
                return
            f = [x.strip() for x in out[0].split(',')]
            try:
                self.gpu.setText(f'{f[0].replace("NVIDIA GeForce ", "")} · {float(f[2]) / 1024:.1f}/'
                                 f'{float(f[1]) / 1024:.0f} GB · {f[3]} %')
            except (IndexError, ValueError):
                pass
        p.finished.connect(done)
        p.start(smi, ['--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])

    def toggle_server(self):
        svc = self.win.service
        if svc.server_state() in ('ready', 'starting'):
            if any(j.spec.needs_carla and j.state == RUNNING for j in self.win.queue.jobs):
                QMessageBox.information(self.win, 'Server busy', 'A CARLA job is running: cancel it first.')
                return
            svc.stop_server()
        else:
            try:
                svc.start_server()
            except RuntimeError as e:
                QMessageBox.warning(self.win, 'CARLA server', str(e))
        self.refresh()

    def stop(self):
        self.timer.stop()
        if self.smi is not None:
            self.smi.kill()
            self.smi.waitForFinished(2000)


class MainWindow(QMainWindow):
    projectChanged = pyqtSignal(object)      # the edited path (list), or None: the whole project was replaced

    def __init__(self, settings, project: Project | None = None, fake_carla: str = ''):
        super().__init__()
        self.settings = settings
        self.fake_carla = fake_carla
        self.project = project or Project.new('Untitled')
        self.undo = QUndoStack(self)
        self.queue = JobQueue(QProcessLauncher(self), memory_reserve_gb=MEMORY_RESERVE_GB)
        self.bridge = JobsBridge(self.queue, self)
        self.service = CarlaService(settings, self.queue, self.workspace(), fake_carla=fake_carla)
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        self.actions_: dict = {}
        self.pages: list = []
        self._closing = False

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        side = QWidget()
        side.setFixedWidth(176)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName('nav')
        self.nav.setIconSize(QSize(16, 16))
        sv.addWidget(self.nav, 1)
        self.palette_hint = QToolButton()
        self.palette_hint.setObjectName('navFoot')
        self.palette_hint.setText('Ctrl+K  Search commands')
        self.palette_hint.setToolTip('The command palette: every page and action by name')
        self.palette_hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.palette_hint.clicked.connect(lambda: self.show_palette())
        sv.addWidget(self.palette_hint)
        self.stack = QStackedWidget()
        h.addWidget(side)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.nav.currentRowChanged.connect(self._nav_changed)

        self.jobs_dock = self._dock('Jobs', JobsDock(self), Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log = LogDock(self)
        self.log_dock = self._dock('Log', self.log, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.tabifyDockWidget(self.jobs_dock, self.log_dock)
        self.jobs_dock.raise_()
        self.resizeDocks([self.jobs_dock], [170], Qt.Orientation.Vertical)
        self.status = StatusBarWidgets(self)
        self.bridge.added.connect(lambda job: (self.show_jobs(), self.status.refresh_jobs()))
        self.bridge.changed.connect(lambda job: self.status.refresh_jobs())
        self.bridge.finished.connect(lambda job: self.status.refresh_jobs())
        self._job_origin: dict = {}                      # job id -> the page it was started from
        self.bridge.added.connect(lambda job: self._job_origin.setdefault(job.id, self.current_key()))
        self.set_jobs_visible(False)                     # nothing to show until a job starts
        self._build_pages()
        self._build_actions()
        self.projectChanged.connect(self._on_project_changed)
        self.undo.cleanChanged.connect(lambda _: self._update_title())
        self.service.subscribe(self._on_service)
        self._update_title()
        self.go('home')
        self.log.append_line(f'{APP_NAME} {__version__} — project {self.project.doc["name"]}'
                             + (f' ({tilde(self.project.path)})' if self.project.path else ' (not saved yet)'))
        for n in self.project.notes:
            self.log.append_line(f'project: {n}')

    # ------------------------------------------------------------------ building
    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                      QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                      QDockWidget.DockWidgetFeature.DockWidgetClosable)
        d.setTitleBarWidget(QWidget())                   # the tab under the panel names it; no second title
        self.addDockWidget(area, d)
        return d

    def _build_pages(self):
        from .pages import GROUPS
        for group, classes in GROUPS:
            if group:
                hd = QListWidgetItem(group.upper())
                hd.setFlags(Qt.ItemFlag.NoItemFlags)
                hd.setFont(sans(10, 600))
                hd.setSizeHint(QSize(0, 26))
                hd.setForeground(QBrush(THEME.color('muted')))
                self.nav.addItem(hd)
            elif self.nav.count():
                sp = QListWidgetItem('')
                sp.setFlags(Qt.ItemFlag.NoItemFlags)
                sp.setSizeHint(QSize(0, 12))
                self.nav.addItem(sp)
            for cls in classes:
                page = cls(self)
                it = QListWidgetItem(icons.icon(cls.icon, 16), cls.title)
                it.setToolTip(cls.about)
                it.setData(Qt.ItemDataRole.UserRole, len(self.pages))
                self.pages.append(page)
                self.nav.addItem(it)
                self.stack.addWidget(page)
                tidy(page)

    def _nav_rows(self):
        """(row, page index) of the navigator's page items (group headers are not pages)."""
        for r in range(self.nav.count()):
            i = self.nav.item(r).data(Qt.ItemDataRole.UserRole)
            if i is not None:
                yield r, i

    def _nav_changed(self, row):
        it = self.nav.item(row)
        i = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        if i is not None:
            self.stack.setCurrentIndex(i)

    def page(self, key):
        return next((p for p in self.pages if p.key == key), None)

    def current_key(self) -> str:
        w = self.stack.currentWidget()
        return getattr(w, 'key', '')

    def go(self, key):
        for r, i in self._nav_rows():
            if self.pages[i].key == key:
                self.nav.setCurrentRow(r)
                return self.pages[i]
        return None

    # ------------------------------------------------------------------ jobs panel
    def set_jobs_visible(self, on: bool):
        for d in (self.jobs_dock, self.log_dock):
            d.setVisible(on)
        if on:
            self.jobs_dock.raise_()

    def show_jobs(self):
        if not self.jobs_dock.isVisible():
            self.set_jobs_visible(True)

    def toggle_jobs(self):
        self.set_jobs_visible(not (self.jobs_dock.isVisible() or self.log_dock.isVisible()))

    def open_job(self, job_id):
        """Go to where a job's result is shown: a recorded drive in Data, anything else on the page it was started
        from."""
        job = next((j for j in self.queue.jobs if j.id == job_id), None)
        if job is None:
            return None
        if job.spec.kind in ('drive', 'smoke') and job.spec.run_dir and os.path.isdir(job.spec.run_dir):
            data = self.go('data')
            data.show_run(job.spec.run_dir)
            return data
        key = self._job_origin.get(job.id) or ''
        return self.go(key) if key and key != 'home' else None

    def _act(self, name, text, slot, shortcut=None, menu=None, tip=''):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        a.setToolTip(tip or text)
        self.addAction(a)
        self.actions_[name] = a
        if menu is not None:
            menu.addAction(a)
        return a

    def _build_actions(self):
        mb = self.menuBar()
        f = mb.addMenu('&File')
        self._act('new', 'New project…', self.new_project, QKeySequence.StandardKey.New, f)
        self._act('open', 'Open project…', self.open_project_dialog, QKeySequence.StandardKey.Open, f)
        self._act('save', 'Save', self.save, QKeySequence.StandardKey.Save, f)
        self._act('save_as', 'Save as…', self.save_as, QKeySequence.StandardKey.SaveAs, f)
        self.recent_menu = f.addMenu('Recent projects')
        self._fill_recent()
        f.addSeparator()
        self._act('quit', 'Quit', self.close, QKeySequence.StandardKey.Quit, f)
        e = mb.addMenu('&Edit')
        ua = self.undo.createUndoAction(self, 'Undo')
        ua.setShortcut(QKeySequence.StandardKey.Undo)
        ra = self.undo.createRedoAction(self, 'Redo')
        ra.setShortcut(QKeySequence.StandardKey.Redo)
        e.addAction(ua)
        e.addAction(ra)
        self.actions_['undo'], self.actions_['redo'] = ua, ra
        e.addSeparator()
        self._act('palette', 'Command palette…', self.show_palette, 'Ctrl+K', e)
        v = mb.addMenu('&View')
        for name in ('dark', 'light'):
            self._act(f'theme_{name}', f'{name.title()} theme', lambda _=False, n=name: self.set_theme(n), menu=v)
        v.addSeparator()
        self._act('jobs_panel', 'Jobs and log panel', self.toggle_jobs, 'Ctrl+J', v)
        for i, p in enumerate(self.pages):            # Ctrl+0 Home, Ctrl+1 … Ctrl+9 the next nine pages
            self._act(f'page_{p.key}', f'Go to {p.title}', lambda _=False, k=p.key: self.go(k),
                      f'Ctrl+{i}' if i <= 9 else None, v if i == 0 else None)
        t = mb.addMenu('&Tools')
        self._act('wizard', 'First-run wizard…', self.show_wizard, menu=t)
        self._act('probe', 'Probe the CARLA server', self.probe, menu=t)
        self._act('tidy', 'Tidy the CARLA server', self.tidy, menu=t,
                  tip='Remove what a killed Falcon worker left on Falcon\'s server and restore asynchronous mode')
        self._act('server_start', 'Start the CARLA server', lambda: self.status.toggle_server()
                  if self.service.server_state() not in ('ready', 'starting') else None, menu=t)
        self._act('server_stop', 'Stop the CARLA server', lambda: self.status.toggle_server()
                  if self.service.server_state() in ('ready', 'starting') else None, menu=t)
        hm = mb.addMenu('&Help')
        self._act('manual', 'Manual', lambda: self.go('help'), 'F1', hm)
        self._act('about', f'About {APP_NAME}', self.about, menu=hm)

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in self.settings.get('recent_projects') or []:
            if os.path.isfile(p):
                a = self.recent_menu.addAction(tilde(p))
                a.triggered.connect(lambda _=False, path=p: self.open_project(path))
        self.recent_menu.setEnabled(bool(self.recent_menu.actions()))

    # ------------------------------------------------------------------ project
    def workspace(self) -> str:
        if self.project.path:
            return self.project.workspace
        from ..core.paths import cache_dir
        return os.path.join(cache_dir(), 'untitled.falcon')     # an unsaved project's runs go here until it is saved

    def edit(self, edit: Edit, merge_key: str = ''):
        """Apply an edit through the undo stack (every project change goes here)."""
        self.undo.push(EditCommand(self, edit, merge_key))

    def _on_project_changed(self, path):
        self._update_title()
        for p in self.pages:
            p.refresh()

    def _update_title(self):
        name = self.project.doc.get('name', 'Untitled')
        where = f' — {tilde(self.project.path)}' if self.project.path else ''
        self.setWindowTitle(f'{"● " if self.project.dirty else ""}{name}{where} · {APP_NAME}')

    def set_project(self, project: Project):
        self.project = project
        self.undo.clear()
        self.service.set_workspace(self.workspace())
        if project.path:
            self.settings.add_recent(project.path)
            self._fill_recent()
        self.projectChanged.emit(None)
        for n in project.notes:
            self.log.append_line(f'project: {n}')

    def maybe_save(self) -> bool:
        if not self.project.dirty:
            return True
        r = QMessageBox.question(self, 'Unsaved changes', f'Save the changes to {self.project.doc["name"]}?',
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Cancel:
            return False
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return True

    def new_project(self):
        if not self.maybe_save():
            return
        self.set_project(Project.new('Untitled'))

    def open_project_dialog(self):
        if not self.maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Open project', os.path.expanduser('~'),
                                              f'Falcon projects (*{PROJECT_SUFFIX})')
        if path:
            self.open_project(path)

    def open_project(self, path):
        try:
            p = Project.load(path)
        except (OSError, ProjectError) as e:
            QMessageBox.warning(self, 'Cannot open project', str(e))
            return False
        self.set_project(p)
        return True

    def save(self) -> bool:
        if not self.project.path:
            return self.save_as()
        try:
            self.project.save()
        except (OSError, ProjectError) as e:
            QMessageBox.warning(self, 'Cannot save', str(e))
            return False
        self.undo.setClean()
        self._update_title()
        self.log.append_line(f'saved {tilde(self.project.path)}')
        return True

    def save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, 'Save project', os.path.join(
            os.path.expanduser('~'), self.project.doc['name'] + PROJECT_SUFFIX), f'Falcon projects (*{PROJECT_SUFFIX})')
        if not path:
            return False
        old_ws = self.workspace()
        self.project.path = path if path.endswith(PROJECT_SUFFIX) else path + PROJECT_SUFFIX
        if not self.save():
            return False
        if old_ws != self.workspace():
            self.service.set_workspace(self.workspace())
            self.log.append_line(f'workspace: {tilde(self.workspace())}')
        self.settings.add_recent(self.project.path)
        self._fill_recent()
        return True

    # ------------------------------------------------------------------ commands
    def show_palette(self):
        from .palette import CommandPalette
        CommandPalette(self).exec()

    def set_theme(self, name):
        THEME.apply(QApplication.instance(), name)
        self.settings.set('theme', name)
        for r in range(self.nav.count()):             # icons are tinted per theme
            it = self.nav.item(r)
            i = it.data(Qt.ItemDataRole.UserRole)
            if i is None:
                it.setForeground(QBrush(THEME.color('muted')))
            else:
                it.setIcon(icons.icon(self.pages[i].icon, 16))
        for p in self.pages:
            p.refresh()

    def show_wizard(self):
        from .wizard import FirstRunWizard
        FirstRunWizard(self).exec()

    def probe(self):
        try:
            job = self.service.submit_probe(wait_s=5.0)
        except RuntimeError as e:
            QMessageBox.warning(self, 'Probe', str(e))
            return
        self.show_jobs()
        return job

    def tidy(self):
        try:
            self.service.submit_tidy()
        except RuntimeError as e:
            QMessageBox.warning(self, 'Tidy', str(e))

    def about(self):
        QMessageBox.about(self, f'About {APP_NAME}', f'<b>{APP_NAME}</b> {__version__}<br>Autonomous-vehicle '
                          'development and AI training on the CARLA simulator.<br>GPL-3.0-or-later.')

    def _on_service(self, ev, payload):
        if ev == 'job' and payload.spec.kind in ('drive', 'smoke'):
            for p in self.pages:
                if hasattr(p, 'on_run_finished'):
                    p.on_run_finished(payload)

    # ------------------------------------------------------------------ closing
    def closeEvent(self, ev):
        if not self._closing:
            if not self.maybe_save():
                ev.ignore()
                return
            active = self.queue.active()
            if active:
                r = QMessageBox.question(self, 'Jobs running', f'{len(active)} job(s) are queued or running. Cancel '
                                         'them and quit? (Each worker tidies the server first.)')
                if r != QMessageBox.StandardButton.Yes:
                    ev.ignore()
                    return
        self.shutdown()
        ev.accept()

    def shutdown(self):
        """Cancel the jobs (waiting for them to tidy up), stop Falcon's own server, wait for background tasks."""
        if self._closing:
            return
        self._closing = True
        self.status.stop()
        self.bridge.timer.stop()
        for p in self.pages:
            if hasattr(p, 'shutdown'):
                p.shutdown()
        for j in self.queue.active():
            self.queue.cancel(j)
        launcher = self.queue.launcher
        for proc in list(getattr(launcher, 'procs', {}).values()):
            if not proc.waitForFinished(20000):
                proc.kill()
                proc.waitForFinished(5000)
        QApplication.processEvents()
        self.service.shutdown(grace_s=0.0)
        tasks.wait_all(30.0)
        QApplication.processEvents()
