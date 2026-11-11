"""Settings ▸ Compute: the cluster's nodes and pools, and cloud bursts' provider, caps and keys (docs/cluster.md,
docs/cloud.md). Built only when the Settings page is (ui/pages/settings.py SECTIONS); nothing here touches the
network until the user presses Test.

Nodes are added in a dialog (an SSH key is chosen as a file: Falcon keeps its path, never the key), removed, and
tested (each slot's daemon answers, or the batch system does) on a background thread. Pools take their limits in the
table. The providers' API keys go to the keyring (assistant.keystore names runpod_api_key, lambda_api_key) as soon
as they are set: the section shows where a key is kept, never the key. apply(settings) writes the cloud settings
(provider, image, the caps, the TTL factor) and the pools' limits with the page's Apply.
"""
from __future__ import annotations

import os
import threading

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                             QHeaderView, QLineEdit, QMessageBox, QSpinBox, QTableWidget, QTableWidgetItem,
                             QVBoxLayout)

from ..cluster import nodes as N
from .widgets import Form, Section, button, callout, label

NODE_COLS = ('Node', 'Kind', 'Pool', 'Slots', 'Where', 'Test')


class _Signals(QObject):
    tested = pyqtSignal(str, object)


class NodeDialog(QDialog):
    """A new node: its kind decides the fields shown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add a compute node')
        v = QVBoxLayout(self)
        f = self.form = Form()
        self.kind = f.row('Kind', QComboBox(), 'ssh: a workstation running falcon-worker-daemon@N; local: daemons on '
                                               'this machine; slurm: sbatch arrays; k8s: kubectl Jobs')
        self.kind.addItems(N.KINDS)
        self.name = f.row('Name', QLineEdit())
        self.pool = f.row('Pool', QLineEdit('default'))
        self.slots = f.row('Slots', QLineEdit('0'), 'daemon slot numbers, comma-separated (slot N: port 2350+N)')
        self.host = f.row('Host', QLineEdit(), 'the node, or the Slurm login host (empty: sbatch runs here)')
        self.user = f.row('SSH user', QLineEdit())
        kb = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setPlaceholderText('~/.ssh/id_ed25519')
        kb.addWidget(self.key, 1)
        kb.addWidget(button('Choose…', self._choose_key))
        f.row('SSH key file', kb, 'Falcon keeps the path to your key, never the key (ssh-copy-id installs it)')
        self.workdir = f.row('Shared folder', QLineEdit(), 'slurm: a folder every compute node sees')
        self.image = f.row('Image', QLineEdit(), 'k8s: the Falcon worker image (packaging/cluster/Dockerfile)')
        self.context = f.row('kubectl context', QLineEdit())
        self.namespace = f.row('Namespace', QLineEdit())
        self.max_jobs = f.row('Jobs at once', QSpinBox(), 'slurm, k8s')
        self.max_jobs.setRange(1, 1000)
        self.max_jobs.setValue(4)
        v.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.kind.currentTextChanged.connect(self._kind)
        self._kind(self.kind.currentText())
        self.node = None

    def _choose_key(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Your SSH private key', os.path.expanduser('~/.ssh'))
        if p:
            self.key.setText(p)

    def _kind(self, kind):
        ssh = kind in ('ssh', 'slurm')
        for w, on in ((self.slots, kind in ('ssh', 'local')), (self.host, ssh), (self.user, ssh),
                      (self.key, ssh), (self.workdir, kind == 'slurm'), (self.image, kind == 'k8s'),
                      (self.context, kind == 'k8s'), (self.namespace, kind == 'k8s'),
                      (self.max_jobs, kind in ('slurm', 'k8s'))):
            self.form.set_row_visible(w, on)
            w.setVisible(on)

    def values(self) -> dict:
        kind = self.kind.currentText()
        n = {'name': self.name.text().strip(), 'kind': kind, 'pool': self.pool.text().strip() or 'default'}
        if kind in ('ssh', 'local'):
            try:
                n['slots'] = [int(x) for x in self.slots.text().split(',') if x.strip()]
            except ValueError:
                n['slots'] = 'bad'
        if kind in ('ssh', 'slurm'):
            for k, w in (('host', self.host), ('ssh_user', self.user), ('ssh_key', self.key)):
                if w.text().strip():
                    n[k] = w.text().strip()
        if kind == 'slurm':
            n['workdir'] = self.workdir.text().strip()
        if kind == 'k8s':
            for k, w in (('image', self.image), ('context', self.context), ('namespace', self.namespace)):
                if w.text().strip():
                    n[k] = w.text().strip()
        if kind in ('slurm', 'k8s'):
            n['max_jobs'] = self.max_jobs.value()
        return n

    def _ok(self):
        try:
            self.node = N.validate(self.values())
        except (N.NodeError, ValueError) as e:
            QMessageBox.warning(self, 'Compute node', str(e))
            return
        self.accept()


class ComputeSection(Section):
    def __init__(self, page):
        super().__init__('Compute', 'Where jobs run besides this machine: workstations, Slurm, Kubernetes; and cloud '
                                    'bursts, priced and approved first. docs/cluster.md and docs/cloud.md.')
        self.page = page
        self.st = page.win.settings
        self.sig = _Signals()
        self.sig.tested.connect(self._tested)
        self.add(label('Nodes', h=True))
        self.table = QTableWidget(0, len(NODE_COLS))
        self.table.setHorizontalHeaderLabels(NODE_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(len(NODE_COLS) - 1, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(110)
        self.add(self.table)
        row = QHBoxLayout()
        self.add_btn = row.addWidget(button('Add node…', self.add_node)) or None
        row.addWidget(button('Remove', self.remove_node))
        row.addWidget(button('Test', self.test_node, tip='reach the selected node now (its slots, or its batch '
                                                         'system)'))
        row.addStretch(1)
        self.add(row)
        self.add(label('Pools', h=True))
        self.pools = QTableWidget(0, 4)
        self.pools.setHorizontalHeaderLabels(('Pool', 'Slots', 'Jobs at once (0: no limit)',
                                              'Jobs a day (0: no limit)'))
        self.pools.verticalHeader().setVisible(False)
        self.pools.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.pools.setMinimumHeight(80)
        self.add(self.pools)
        self._cloud()
        self.refresh()

    # ---------------------------------------------------------------- cloud
    def _cloud(self):
        from ..cluster.cloud import alert, cloud_settings
        cs = cloud_settings(self.st)
        self.add(label('Cloud bursts', h=True))
        al = alert()
        if al:
            self.add(callout(al['message'], 'critical'))
        f = Form()
        self.provider = f.row('Provider', QComboBox(), 'on demand on both (neither offers spot through its current '
                                                       'API)')
        self.provider.addItem('none', '')
        self.provider.addItem('RunPod', 'runpod')
        self.provider.addItem('Lambda', 'lambda')
        self.provider.setCurrentIndex(max(self.provider.findData(cs.get('provider') or ''), 0))
        self.image = f.row('Worker image', QLineEdit(str(cs.get('image') or '')),
                           'packaging/cluster/Dockerfile, pushed where the provider can pull it')
        self.per_burst = f.row('Cap per burst', QDoubleSpinBox(), 'a burst whose worst case is over this is refused, '
                                                                 'approved or not')
        self.per_month = f.row('Cap per month', QDoubleSpinBox(), 'finished bursts plus running ones\' worst cases')
        for w, v in ((self.per_burst, cs['max_usd_per_burst']), (self.per_month, cs['max_usd_month'])):
            w.setRange(0.0, 100000.0)
            w.setDecimals(2)
            w.setPrefix('$ ')
            w.setValue(float(v))
        self.ttl = f.row('TTL factor', QDoubleSpinBox(), 'the machine is terminated at this many times the '
                                                         'expected hours, whatever happens')
        self.ttl.setRange(1.0, 10.0)
        self.ttl.setSingleStep(0.1)
        self.ttl.setValue(float(cs.get('ttl_factor') or 1.5))
        self.keys = {}
        for name, title in (('runpod', 'RunPod API key'), ('lambda', 'Lambda API key')):
            box = QHBoxLayout()
            e = QLineEdit()
            e.setEchoMode(QLineEdit.EchoMode.Password)
            e.setPlaceholderText('paste a key to store it')
            state = label('', muted=True, small=True)
            box.addWidget(e, 1)
            box.addWidget(button('Set', lambda _=False, n=name: self.set_key(n)))
            box.addWidget(button('Remove', lambda _=False, n=name: self.remove_key(n)))
            box.addWidget(state)
            f.row(title, box)
            self.keys[name] = (e, state)
        self.add(f)
        self.month = label('', muted=True, small=True)
        self.add(self.month)
        self._key_states()

    def _store(self, name):
        from ..assistant.keystore import KeyStore
        from ..cluster.cloud import KEY_NAMES
        return KeyStore(name=KEY_NAMES[name])

    def _key_states(self):
        from ..cluster.cloud import Ledger
        for name, (_e, state) in self.keys.items():
            ks = self._store(name)
            state.setText(f'stored in {ks.where()}' if ks.has() else 'not set')
        m = Ledger().month_spend()
        self.month.setText(f'This month: ${m["spent"]:.2f} spent, ${m["reserved"]:.2f} reserved by running bursts '
                           '(falcon cloud status, falcon cloud ledger).')

    def set_key(self, name):
        e, _s = self.keys[name]
        try:
            self._store(name).set(e.text())
        except ValueError as err:
            QMessageBox.warning(self, 'API key', str(err))
            return
        e.clear()
        self._key_states()

    def remove_key(self, name):
        self._store(name).clear() if hasattr(self._store(name), 'clear') else self._store(name).remove()
        self._key_states()

    # ---------------------------------------------------------------- nodes and pools
    def refresh(self):
        cfg = N.load(self.st)
        self.table.setRowCount(0)
        for n in cfg['nodes']:
            r = self.table.rowCount()
            self.table.insertRow(r)
            where = {'ssh': n.get('host', ''), 'local': 'this machine', 'slurm': n.get('host') or 'sbatch here',
                     'k8s': n.get('context') or 'kubectl\'s context'}[n['kind']]
            for c, v in enumerate((n['name'], n['kind'], n['pool'], str(N.capacity(n)), where, '')):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.pools.setRowCount(0)
        for name, p in sorted(N.pools(cfg).items()):
            r = self.pools.rowCount()
            self.pools.insertRow(r)
            it = QTableWidgetItem(name)
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.pools.setItem(r, 0, it)
            it = QTableWidgetItem(str(p['slots']))
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.pools.setItem(r, 1, it)
            for c, k in ((2, 'max_running'), (3, 'max_per_day')):
                sp = QSpinBox()
                sp.setRange(0, 100000)
                sp.setValue(int(p[k]))
                self.pools.setCellWidget(r, c, sp)

    def add_node(self, node: dict | None = None):
        """Add a node (from the dialog; tests pass one). True when it was added."""
        if node is None:
            d = NodeDialog(self)
            if d.exec() != QDialog.DialogCode.Accepted or d.node is None:
                return False
            node = d.node
        try:
            N.add(self.st, node)
        except N.NodeError as e:
            QMessageBox.warning(self, 'Compute node', str(e))
            return False
        self.refresh()
        return True

    def _selected(self) -> str:
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        return it.text() if it else ''

    def remove_node(self):
        name = self._selected()
        if name and N.remove(self.st, name):
            self.refresh()

    def test_node(self, name: str = ''):
        name = name or self._selected()
        node = next((n for n in N.load(self.st)['nodes'] if n['name'] == name), None)
        if node is None:
            return
        self._mark(name, 'testing…')

        def run():
            from ..cluster.cli import test_node
            try:
                r = test_node(node, timeout=15)
            except Exception as e:                  # noqa: BLE001 (shown in the table)
                r = {'ok': False, 'error': str(e)}
            self.sig.tested.emit(name, r)
        threading.Thread(target=run, daemon=True).start()

    def _tested(self, name, r):
        if r.get('slots'):
            ok = sum(1 for s in r['slots'] if s.get('ok'))
            text = f'{ok} of {len(r["slots"])} slot(s) answer' + ('' if r['ok'] else
                                                                   f': {next(s.get("error") for s in r["slots"] if not s.get("ok"))}')
        else:
            text = 'reachable' if r.get('ok') else f'not reachable: {r.get("error") or r.get("answer")}'
        self._mark(name, text)

    def _mark(self, name, text):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() == name:
                self.table.setItem(r, len(NODE_COLS) - 1, QTableWidgetItem(text))
                self.table.item(r, len(NODE_COLS) - 1).setToolTip(text)

    # ---------------------------------------------------------------- Apply
    def apply(self, st) -> str:
        """Write the pools' limits and the cloud settings ('' when saved, else why not)."""
        if self.per_month.value() < self.per_burst.value():
            return 'Compute: the monthly cap is below the cap per burst'
        cfg = N.load(st)
        for r in range(self.pools.rowCount()):
            name = self.pools.item(r, 0).text()
            run, day = self.pools.cellWidget(r, 2).value(), self.pools.cellWidget(r, 3).value()
            if run or day or name in cfg['pools']:
                cfg['pools'][name] = {'max_running': run, 'max_per_day': day}
        from ..cluster.cloud import cloud_settings
        cs = dict(cloud_settings(st), provider=self.provider.currentData() or '', image=self.image.text().strip(),
                  max_usd_per_burst=round(self.per_burst.value(), 2), max_usd_month=round(self.per_month.value(), 2),
                  ttl_factor=round(self.ttl.value(), 2))
        st.set('cluster', {'nodes': cfg['nodes'], 'pools': cfg['pools']}, save=False)
        st.set('cloud', cs, save=False)
        return ''


def section(page) -> ComputeSection | None:
    """The section for the Settings page; None without a window's settings (extra_sections skips it)."""
    if getattr(getattr(page, 'win', None), 'settings', None) is None:
        return None
    return ComputeSection(page)
