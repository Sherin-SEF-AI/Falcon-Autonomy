"""Hardware ▸ Runtime monitors: the rules Falcon checks offline, generated as an online monitor for the vehicle
computer (core.monitorgen; docs/runtime-monitors.md).

Pick a rule set and Generate: a package is written to <workspace>/monitors/<name> (a standard-library evaluator, an
rclpy node, a topic mapping to edit, a systemd user unit, a manifest). The table lists the package's rules with the
equivalence of the generated monitor and the offline checker, in a word and a colour — equal (the latest check agreed,
sample by sample), differs, not checked, formula changed (the rule was edited since: generate again) — its verdict
delay and formula hash. "Check against a run…" replays a recorded run through the generated module. The deploy
commands are shown as text: Falcon copies files over SSH only when you run `falcon monitor deploy` yourself, and never
installs or starts anything on the device.

4.2: three commands, each with a Copy button (ui.widgets.CommandBlock; Falcon runs none of them): the dry run
(`--dry-run`: the package's files, unit and manifest validated here, what would be copied and enabled printed; no SSH),
the deploy (it keeps the version it replaces on the device) and the rollback (`falcon monitor rollback`: the kept
version restored, its user unit restarted only if it is running). The device-side install commands follow, also with
Copy. The package's state is compact rows (status, check, manifest id — click to copy); the equivalence check wears
the fidelity of the runs it replayed (core.fidelity: `recorded`, of their level)."""
from __future__ import annotations

import os
import shlex

from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ..core import monitorgen as MG
from . import motion
from .theme import THEME, mono
from .widgets import CommandBlock, Details, IdLabel, KeyValues, Note, button, fidelity_chip, flow_row, label

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
        v.addWidget(Note('The rule that failed a run in simulation, watching the car: a rule set compiled into an '
                         'online monitor (same formulas, same robustness). A verdict that looks ahead is delayed by '
                         'its horizon. Deployment and the ROS 2 node are tested with fakes only.'))
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
        self.note = label('', muted=True, wrap=True, small=True)
        v.addWidget(self.note)
        self.facts = KeyValues()                             # compact: status, check, manifest, changed files
        v.addWidget(self.facts)
        row = QHBoxLayout()
        row.addWidget(label('Manifest', muted=True))
        self.manifest_id = IdLabel('', keep=8)
        row.addWidget(self.manifest_id)
        self.chip = fidelity_chip('unknown')
        self.chip.setVisible(False)
        row.addWidget(self.chip)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(label('Deploy (you run these; nothing is run as root)', h=True))
        self.dry_cmd = CommandBlock('', 'Validates the package here (files, unit, manifest) and prints what would '
                                        'be copied and enabled. No SSH, no network.')
        self.deploy_cmd = CommandBlock('', 'Copies the package to the device over SSH with the Device settings; '
                                           'keeps the version it replaces; installs and starts nothing.')
        self.rollback_cmd = CommandBlock('', 'Restores the version the last deploy replaced and restarts the user '
                                             'unit only if it is running (systemctl --user try-restart). Over SSH; '
                                             'no root.')
        for title, blk in (('Dry run (here, no SSH)', self.dry_cmd), ('Deploy', self.deploy_cmd),
                           ('Roll back', self.rollback_cmd)):
            v.addWidget(label(title, muted=True, small=True))
            v.addWidget(blk)
        self.device_cmd = CommandBlock('', 'Run on the device: installs and starts the user unit. The last line '
                                           'needs root: run it yourself.')
        self.device = Details(self.device_cmd, 'On the device (install and start the unit)')
        v.addWidget(self.device)
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
            self.set_commands('')
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
        self.note.setText(f'{st["name"]}: {word}{when}.')
        self.note.setToolTip('Edit mapping.json in the package to point the signals at your stack\'s topics.')
        self.facts.clear()
        if st['changed_files']:
            self.facts.set('Changed', ', '.join(st['changed_files']), tip='Files changed since it was generated: '
                                                                         'generate it again')
        self.facts.set('Signals', 'mapping.json (edit it for your stack\'s topics)',
                       tip='Not pinned by the manifest: it is yours to edit')
        self.manifest_id.setText(st['hash'])
        lv = st.get('check_fidelity') or ''
        self.chip.setVisible(bool(lv))
        if lv:
            from ..core import fidelity as F
            self.chip.setText(f'{F.label("recorded")} · of {F.label(lv)} runs' if lv != 'unknown'
                              else F.label('recorded'))
            self.chip.setToolTip('The check replays recorded runs offline. ' + F.sentence(lv))
            self.chip.setProperty('tone', F.tone(lv))
            self.chip.style().unpolish(self.chip)
            self.chip.style().polish(self.chip)
        self.set_commands(pkg, st['name'])

    def set_commands(self, pkg: str, name: str = ''):
        """The dry-run, deploy and rollback commands of a package and the device's install commands ('' clears)."""
        q = shlex.quote(pkg) if pkg else 'PKG'
        self.dry_cmd.set_command(f'falcon monitor deploy {q} --dry-run', self.dry_cmd.toolTip())
        self.deploy_cmd.set_command(f'falcon monitor deploy {q}', self.deploy_cmd.toolTip())
        self.rollback_cmd.set_command(f'falcon monitor rollback {q}', self.rollback_cmd.toolTip())
        for blk in (self.dry_cmd, self.deploy_cmd, self.rollback_cmd, self.device_cmd):
            blk.copy_btn.setEnabled(bool(pkg))
        self.device_cmd.set_command('\n'.join(MG.install_commands(name)) if name else '', self.device_cmd.toolTip())

    def command_text(self) -> str:
        """Every command shown, one per line (what the Copy buttons copy)."""
        return '\n'.join(b.command for b in (self.dry_cmd, self.deploy_cmd, self.rollback_cmd, self.device_cmd))

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
