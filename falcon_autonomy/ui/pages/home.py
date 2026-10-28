"""Status: the server, the machine, the measured throughput and the latest runs at a glance — the Status tab of the
Loop page (the home page), where the first page used to show them."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QTreeWidget, QTreeWidgetItem

from ...core import budget, sysinfo
from ...core.paths import tilde
from ...sim import server as S
from .. import icons, tasks
from ..widgets import KeyValues, Page, Section, button


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
