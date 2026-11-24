"""Hardware ▸ Runtime monitors: the rules Falcon checks offline, generated as an online monitor for the vehicle
computer (core.monitorgen; docs/runtime-monitors.md).

Pick a rule set and Generate: a package is written to <workspace>/monitors/<name> (a standard-library evaluator, an
rclpy node, a topic mapping to edit, a systemd user unit, a manifest). The table lists the package's rules with the
equivalence of the generated monitor and the offline checker, in a word and a colour — equal (the latest check agreed,
sample by sample), differs, not checked, formula changed (the rule was edited since: generate again) — its verdict
delay and formula hash. "Check against a run…" replays a recorded run through the generated module. The deploy
commands are shown as text: Falcon copies files over SSH only when you run `falcon monitor deploy` yourself, and never
installs or starts anything on the device."""
from __future__ import annotations

import os

from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QPlainTextEdit, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import monitorgen as MG
from . import motion
from .theme import THEME, mono
from .widgets import button, flow_row, label

COLUMNS = ('Rule', 'Equivalence', 'Delay', 'Formula hash')
WORD = {'agreed': ('equal', 'good'), 'disagreed': ('differs', 'critical'), 'unchecked': ('not checked', ''),
        'stale': ('formula changed', 'warning')}


class MonitorTab(QWidget):
    """The Runtime monitors section (module doc). owner: the Hardware page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.status = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(label('The rule that failed a run in simulation, watching the car: a rule set compiled into an '
                          'online monitor (same formulas, same robustness). A verdict that looks ahead is delayed by '
                          'its horizon. Deployment and the ROS 2 node are tested with fakes only.', muted=True,
                          wrap=True))
        self.ruleset = QComboBox()
        for k in ('india', 'generic'):
            self.ruleset.addItem(k, k)
        self.name = QLineEdit()
        self.name.setPlaceholderText('name (default: the rule set\'s)')
        self.gen_btn = button('Generate', self.generate, role='primary',
                              tip='falcon monitor gen: write the package in the workspace\'s monitors folder')
        v.addWidget(flow_row(self.ruleset, self.name, self.gen_btn))
        self.package = QComboBox()
        self.package.currentIndexChanged.connect(lambda _i: self.show_package())
        self.check_btn = button('Check against a run…', self.pick_run,
                                tip='falcon monitor check: replay a recorded run through the generated monitor and '
                                    'compare with the offline checker')
        v.addWidget(flow_row(self.package, self.check_btn, button('Refresh', self.refresh)))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(170)
        v.addWidget(self.tree)
        self.note = label('', muted=True, wrap=True)
        v.addWidget(self.note)
        v.addWidget(label('Deploy (you run these; nothing is run as root)', h=True))
        self.commands = QPlainTextEdit()
        self.commands.setReadOnly(True)
        self.commands.setFont(mono(11))
        self.commands.setFixedHeight(150)
        v.addWidget(self.commands)
        THEME.changed.connect(self._themed)          # a bound slot: let go when the tab is deleted
        self.refresh()

    def _themed(self, _name):
        self.show_package()

    def workspace(self) -> str:
        try:
            return self.win.workspace() or ''
        except AttributeError:
            return ''

    def current(self) -> str:
        return self.package.currentData() or ''

    def refresh(self, select: str = ''):
        ws = self.workspace()
        keep = select or self.current()
        self.package.blockSignals(True)
        self.package.clear()
        for p in (MG.list_packages(ws) if ws else []):
            self.package.addItem(os.path.basename(p), p)
        k = self.package.findData(keep)
        self.package.setCurrentIndex(max(k, 0))
        self.package.blockSignals(False)
        self.show_package()

    def show_package(self, fresh=()):
        self.tree.clear()
        pkg = self.current()
        self.check_btn.setEnabled(bool(pkg))
        if not pkg:
            self.status = None
            self.note.setText('No monitor package in this workspace yet: choose a rule set and Generate.'
                              if self.workspace() else 'Open a project to generate a monitor.')
            self.commands.setPlainText('')
            return
        try:
            st = self.status = MG.status(pkg)
        except MG.MonitorGenError as e:
            self.status = None
            self.note.setText(str(e))
            return
        for r in st['rules']:
            word, tone = WORD[r['status']]
            it = QTreeWidgetItem([r['id'], word.upper(), f'{r["delay_s"]:g} s', r['sha']])
            it.setToolTip(0, r['title'])
            it.setToolTip(1, r['why'])
            it.setForeground(1, QBrush(QColor(THEME.status(tone) if tone else THEME.tok['muted'])))
            for c in (1, 2, 3):
                it.setFont(c, mono(11))
            self.tree.addTopLevelItem(it)
            if r['id'] in fresh and motion.enabled():
                motion.flash_item(it, len(COLUMNS))
        for c in range(len(COLUMNS)):
            self.tree.resizeColumnToContents(c)
        word = WORD[st['status']][0]
        when = f', checked {st["checked_at"]}' if st['checked_at'] else ''
        self.note.setText(f'{st["name"]}: {word}{when}. Manifest {st["hash"][:16]}. '
                          + ('Files changed since it was generated: ' + ', '.join(st['changed_files']) + '. '
                             if st['changed_files'] else '')
                          + 'Edit mapping.json in the package to point the signals at your stack\'s topics.')
        self.commands.setPlainText('\n'.join(
            ['# on this machine (copies the package over SSH with the Device settings above; starts nothing):',
             f'falcon monitor deploy {pkg}',
             '# then on the device:'] + MG.install_commands(st['name'])))

    def generate(self):
        ws = self.workspace()
        if not ws:
            self.note.setText('Open a project first: the package is written to its workspace.')
            return None
        rs = self.ruleset.currentData()
        name = self.name.text().strip() or rs
        try:
            rules, src = MG.compile_rules(rs)
            man = MG.generate(os.path.join(ws, 'monitors', name), rules, src, name)
        except (MG.MonitorGenError, OSError) as e:
            self.note.setText(f'Not generated: {e}')
            return None
        self.refresh(os.path.join(ws, 'monitors', name))
        return man

    def pick_run(self):
        d = QFileDialog.getExistingDirectory(self, 'A recorded run', os.path.join(self.workspace(), 'runs'))
        if d:
            self.check_run(d)

    def check_run(self, run_dir: str):
        pkg = self.current()
        if not pkg:
            return None
        try:
            doc = MG.check(pkg, [run_dir])
        except (MG.MonitorGenError, OSError) as e:
            self.note.setText(f'Not checked: {e}')
            return None
        self.show_package(fresh=set(doc['rules']))
        err = next((x['error'] for x in doc['runs'] if x.get('error')), None)
        if err:
            self.note.setText(f'The run could not be read: {err}')
        return doc
