"""The first-run wizard: find CARLA and its Python, check the GPU, the Vulkan driver and memory, then run the 30 s
smoke drive and show what this machine measured."""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWizard,
                             QWizardPage)

from ..core import sysinfo
from ..core.jobs import FINAL
from ..core.paths import tilde
from ..sim import server as S
from . import tasks
from .widgets import KeyValues, button, label


class FindCarlaPage(QWizardPage):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setTitle('Find CARLA')
        self.setSubTitle('Falcon runs CARLA 0.9.16 as its own headless server and talks to it from CARLA\'s own '
                         'Python (a venv with the carla wheel). Neither is bundled.')
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        self.root = QComboBox()
        self.root.setEditable(True)
        roots = S.candidate_roots()
        cur = win.settings.get('carla_root')
        self.root.addItems(([cur] if cur and cur not in roots else []) + roots)
        h.addWidget(self.root, 1)
        h.addWidget(button('Browse…', self.browse))
        v.addWidget(label('CARLA folder', muted=True))
        v.addLayout(h)
        self.kv = KeyValues(('Folder', 'Version', 'CARLA\'s Python', 'Client'))
        v.addWidget(self.kv)
        v.addStretch(1)
        self.ok = False
        self.root.currentTextChanged.connect(lambda _: self.check())

    def initializePage(self):
        self.check()

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, 'CARLA folder', os.path.expanduser('~'))
        if d:
            self.root.setCurrentText(d)

    def check(self):
        root = self.root.currentText().strip()
        is_root = S.is_carla_root(root)
        self.kv.set('Folder', tilde(root) + ('' if is_root else ' — not a CARLA folder'))
        self.kv.set('Version', S.carla_version(root) or '?')
        py = S.client_python(root) if is_root else ''
        self.kv.set('CARLA\'s Python', tilde(py) or 'no venv beside the folder (set it in Settings)')
        self.ok = False
        self.completeChanged.emit()
        if not py:
            self.kv.set('Client', '—')
            return
        self.kv.set('Client', 'checking…')

        def done(r):
            self.kv.set('Client', ('✓ ' if r['ok'] else '⚠ ') + r['text'])
            self.ok = r['ok']
            if r['ok']:
                self.win.settings.set('carla_root', root, save=False)
                self.win.settings.set('client_python', py)
            self.completeChanged.emit()
        tasks.run(self, lambda: S.check_client(py), done)

    def isComplete(self):
        return self.ok


class MachinePage(QWizardPage):
    def __init__(self, win):
        super().__init__()
        self.setTitle('Check the machine')
        self.setSubTitle('The GPU CARLA renders on, its Vulkan driver (pinned so Unreal never picks the desktop\'s '
                         'integrated GPU), memory, and the memory caps Falcon runs everything under.')
        v = QVBoxLayout(self)
        self.kv = KeyValues()
        v.addWidget(self.kv)
        v.addStretch(1)

    def initializePage(self):
        def show(rows):
            for name, ok, detail in rows:
                self.kv.set(name, ('✓ ' if ok else '⚠ ') + detail)
        tasks.run(self, sysinfo.checks, show)


class SmokePage(QWizardPage):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setTitle('Smoke drive')
        self.setSubTitle('30 s of driving with the project\'s first rig (one 1280 × 720 camera with semantic labels by '
                         'default) in Town 10, recorded as a normal run. What it measures becomes this machine\'s '
                         'throughput for every cost estimate.')
        v = QVBoxLayout(self)
        self.btn = button('Run the smoke drive', self.run, role='primary')
        v.addWidget(self.btn)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        v.addWidget(self.bar)
        self.state = QLabel('')
        self.state.setWordWrap(True)
        v.addWidget(self.state)
        self.kv = KeyValues()
        v.addWidget(self.kv)
        v.addStretch(1)
        self.job = None
        win.bridge.changed.connect(self.on_job)
        win.bridge.finished.connect(self.on_job)

    def run(self):
        from ..core.drive import run_dir_name, smoke_config
        w = self.win
        try:
            cfg = smoke_config(w.project, os.path.join(w.service.runs_dir(), run_dir_name('smoke')), w.service.port,
                               w.service.tm_port)
            self.job = w.service.submit_drive(cfg, title='Smoke drive (30 s)', kind='smoke')
        except Exception as e:                       # noqa: BLE001 (shown)
            self.state.setText(f'⚠ {e}')
            return
        self.btn.setEnabled(False)

    def on_job(self, job):
        if job is not self.job:
            return
        self.bar.setValue(int(1000 * job.progress))
        self.state.setText(job.summary())
        if job.state in FINAL:
            self.btn.setEnabled(True)
            perf = {}
            if job.state == 'done':
                from ..core.provenance import read_json
                try:
                    perf = read_json(os.path.join(job.spec.run_dir, 'run.json')).get('performance') or {}
                except (OSError, ValueError):
                    pass
            if perf:
                self.kv.set('Steps', f'{perf["steps"]} in {perf["wall_s"]:.1f} s ({perf["steps_per_s"]:.1f} steps/s, '
                                     f'{perf["realtime_factor"]:.2f}× real time)')
                self.kv.set('Rendering', f'{perf["mpx_per_step"]:.2f} Mpx per step, {perf["mpx_per_s"]:.0f} Mpx/s '
                                         '(measured)')
                self.kv.set('Saved as', self.win.service.throughput().source)
                self.win.settings.set('first_run_done', True)
            else:
                tail = '; '.join(t for lvl, t in job.log[-4:] if lvl in ('stderr', 'error', 'warning'))
                self.kv.set('Result', f'{job.state}: {tail or job.summary()}')


class FirstRunWizard(QWizard):
    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle('First run')
        self.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
        self.resize(760, 520)
        for p in (FindCarlaPage(win), MachinePage(win), SmokePage(win)):
            self.addPage(p)

    def done(self, r):
        tasks.wait_all(10.0)
        super().done(r)
