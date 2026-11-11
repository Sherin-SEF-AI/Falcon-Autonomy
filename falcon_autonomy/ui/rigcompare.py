"""Rigs ▸ Vantage variants and comparing rigs by perception (docs/rig-compare.md): the import dialog (a .vantage
file's variants as rigs, read only), the dialog that queues `falcon rigs compare`, and a study's results with the
Vantage design-log note. The numbers come from core.rigcompare through sim.rigstudy; nothing is computed here."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                             QListWidget, QListWidgetItem, QMessageBox, QSpinBox, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ..core import rigcompare as RC
from ..core import vantagerig as VR
from ..core.jobs import JobSpec
from ..core.project import Edit
from .widgets import Form, Section, Stat, button, callout, label, scroll

ID_ROLE = Qt.ItemDataRole.UserRole


def _checklist(rows, checked=lambda r: True) -> QListWidget:
    """A list of (text, data) rows with check boxes."""
    w = QListWidget()
    for text, data in rows:
        it = QListWidgetItem(text)
        it.setData(ID_ROLE, data)
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Checked if checked(data) else Qt.CheckState.Unchecked)
        w.addItem(it)
    return w


def checked(w: QListWidget) -> list:
    return [w.item(i).data(ID_ROLE) for i in range(w.count()) if w.item(i).checkState() == Qt.CheckState.Checked]


# ============================================================================================ the import
class VantageImportDialog(QDialog):
    """A .vantage file's variants: pick some, the CARLA vehicle and the image scale; rigs() gives the Falcon rigs."""

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Import variants from a Vantage project')
        self.doc = VR.read(path)
        self.summary = VR.summary(self.doc)
        v = QVBoxLayout(self)
        sec = Section(self.doc['name'], f'{path} — read only; Falcon keeps its path and SHA-256, never writes it.')
        rows = []
        for r in self.summary['variants']:
            seams = ', '.join(f'{s["a"]}|{s["b"]} {s["angle_deg"]:+.0f}°' + ('' if s['source'] == 'vantage' else '*')
                              for s in r.get('seams') or [])
            rows.append((f'{r["name"]} — {len(r["cameras"])} cameras'
                         + (f', {len(r["sensors"])} other sensors' if r['sensors'] else '')
                         + (f'   seams {seams}' if seams else ''), r['uid']))
        self.variants = _checklist(rows)
        self.variants.setMinimumWidth(640)
        sec.add(self.variants)
        sec.add(label('* a seam Vantage computed itself (not in its file): estimated by its rule on a box-shaped '
                      'body. A seam you set in Vantage is used exactly.', muted=True, small=True, wrap=True))
        f = Form()
        self.vehicle = QComboBox()
        veh = self.summary['vehicle']
        self.vehicle.addItem(f'auto: {veh["blueprint"]} ({veh["how"]}'
                             + (f', {veh["size_error"] * 100:.0f} % off' if veh.get('size_error') is not None else '')
                             + ')', 'auto')
        for bp in sorted(VR.CARLA_VEHICLES):
            self.vehicle.addItem(bp, bp)
        f.row('CARLA vehicle', self.vehicle, 'every variant goes on the same car')
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.05, 1.0)
        self.scale.setSingleStep(0.05)
        self.scale.setDecimals(2)
        self.scale.setValue(VR.IMAGE_SCALE)
        f.row('Image scale', self.scale, 'of the camera module\'s resolution')
        sec.add(f)
        for n in self.doc['notes']:
            sec.add(callout(n))
        v.addWidget(sec)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText('Add as rigs')
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def rigs(self) -> list:
        chosen = checked(self.variants)
        if not chosen:
            return []
        return VR.rigs_for(self.doc, chosen, self.vehicle.currentData(), self.scale.value())


def import_vantage(page, path: str | None = None) -> list:
    """Rigs ▸ Import / export ▸ Vantage variants: the file, the dialog, one undoable edit adding the rigs."""
    win = page.win
    if path is None:
        path, _ = QFileDialog.getOpenFileName(page, 'A Vantage project', os.path.expanduser('~'),
                                              'Vantage project (*.vantage)')
        if not path:
            return []
    try:
        dlg = VantageImportDialog(path, page)
    except VR.VantageError as e:
        QMessageBox.warning(page, 'Vantage', f'Cannot read it: {e}')
        return []
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return []
    rigs = dlg.rigs()
    if not rigs:
        return []
    new = VR.merge_into(win.project.doc['rigs'], rigs)
    win.edit(Edit('set', ['rigs'], new, text=f'Import {len(rigs)} rig(s) from {os.path.basename(path)}'))
    page.refresh()
    page.rigs.setCurrentRow(page.rigs.count() - 1)
    return rigs


# ============================================================================================ the comparison
class RigCompareDialog(QDialog):
    """Which rigs, which detector, scenarios and seeds: argv() is `falcon rigs compare`'s."""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.setWindowTitle('Compare rigs by perception')
        doc = win.project.doc
        v = QVBoxLayout(self)
        sec = Section('Compare rigs by perception', 'Every rig records the same scenarios with the same seeds '
                                                    '(kinematic motion, no background traffic); the detector runs on '
                                                    'every camera; each rig is compared with the first, paired.')
        self.rigs = _checklist([(f'{r["name"]} ({r["id"]})' + ('  · from Vantage' if r.get('vantage') else ''),
                                 r['id']) for r in doc['rigs']], lambda rid: bool(
            (win.project.item('rigs', rid) or {}).get('vantage')))
        self.rigs.setMaximumHeight(150)
        sec.add(label('Rigs (the first checked is the baseline)', muted=True, small=True))
        sec.add(self.rigs)
        f = Form()
        self.model = QComboBox()
        for m in doc.get('models') or []:
            if m.get('kind') == 'detector':
                self.model.addItem(m.get('name') or m['id'], m['id'])
        f.row('Detector', self.model)
        self.world = QComboBox()
        for w in doc['worlds']:
            self.world.addItem(w['name'], w['id'])
        f.row('World', self.world)
        self.seeds = QSpinBox()
        self.seeds.setRange(1, 20)
        self.seeds.setValue(3)
        f.row('Seeds per scenario', self.seeds, 'five runs or more give an interval over whole runs')
        sec.add(f)
        from ..core import scenarios as SC
        rows = [(f'{s["name"]} (surround view)', k) for k, s in RC.SVM_SCENARIOS.items()]
        rows += [(s['name'], s['id']) for s in doc.get('scenarios') or []]
        rows += [(SC.LIBRARY[k]['name'], k) for k in SC.LIBRARY]
        self.scenarios = _checklist(rows, lambda k: k in RC.DEFAULT_SCENARIOS)
        self.scenarios.setMaximumHeight(140)
        sec.add(label('Scenarios', muted=True, small=True))
        sec.add(self.scenarios)
        self.why = label('', small=True, wrap=True)
        sec.add(self.why)
        v.addWidget(sec)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok = bb.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setText('Queue the study')
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def problem(self) -> str:
        rigs = checked(self.rigs)
        if len(rigs) < 2:
            return 'Check two rigs or more.'
        if self.model.currentData() is None:
            return 'Register a detector on the Models page first (the synthetic detector tests the method).'
        if not checked(self.scenarios):
            return 'Check a scenario.'
        for rid in rigs:
            r = self.win.project.item('rigs', rid)
            cams = [s for s in r['sensors'] if s.get('type', '').startswith('camera.')]
            if not cams or any(not {'instance', 'depth'} <= set(s.get('labels') or []) for s in cams):
                return f'{r["name"]}: every camera must record the instance and depth labels.'
        return ''

    def argv(self) -> list:
        from ..sim.rigstudy import compare_argv, study_name
        rigs = [self.win.project.item('rigs', r) for r in checked(self.rigs)]
        return compare_argv(self.win.project.path, [r['id'] for r in rigs], self.model.currentData(),
                            checked(self.scenarios), self.seeds.value(), self.world.currentData(), study_name(rigs),
                            self.win.fake_carla)

    def _accept(self):
        why = self.problem()
        self.why.setText(why)
        if not why:
            self.accept()


def compare_rigs(page):
    """Rigs ▸ Compare by perception…: the dialog, then `falcon rigs compare` queued as a job."""
    win = page.win
    dlg = RigCompareDialog(win, page)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    if not win.project.path or win.project.dirty:
        if not win.save():
            return None
    argv = dlg.argv()
    job = win.queue.submit(JobSpec('rigstudy', f'Compare rigs by perception ({argv[argv.index("--name") + 1]})',
                                   [sys.executable, '-m', 'falcon_autonomy'] + argv, memory_gb=0.5))
    if getattr(win, 'jobs_dock', None) is not None:
        win.jobs_dock.raise_()
    return job


# ============================================================================================ the results
def _fmt(r) -> str:
    if r is None or r.get('value') is None:
        return '—'
    return f'{r["value"]:.3f}' + ('' if r.get('ci_low') is None else f'  [{r["ci_low"]:.3f}, {r["ci_high"]:.3f}]')


class RigStudyDialog(QDialog):
    """A rig study's report: the headline per rig, every number per rig with its interval, the paired verdicts, the
    seams; and the Vantage design-log note, written to a file the user chooses."""

    def __init__(self, study: str, parent=None):
        super().__init__(parent)
        from ..core.provenance import read_json
        self.study = study
        self.rep = read_json(os.path.join(study, 'report.json'))
        self.setWindowTitle(f'Rig study {os.path.basename(study)}')
        self.resize(980, 720)
        body = QWidget()
        v = QVBoxLayout(body)
        rep = self.rep
        if rep.get('synthetic_model'):
            v.addWidget(callout('The synthetic detector is not a model: these numbers test the method, not a '
                                'real detector.'))
        for n in rep.get('notes') or []:
            v.addWidget(callout(n, 'serious'))
        tiles = QHBoxLayout()
        for k, var in rep['variants'].items():
            t = {r['metric']: r for r in var['table']}
            st = Stat(var['name'][-34:])
            rr = t.get('recall') or {}
            sr = t.get('stitched_recall')
            st.set('—' if rr.get('value') is None else f'{rr["value"]:.2f}', 'recall',
                   (f'[{rr["ci_low"]:.2f}, {rr["ci_high"]:.2f}]' if rr.get('ci_low') is not None else '')
                   + (f' · stitched {sr["value"]:.2f}' if sr and sr.get('value') is not None else ''))
            tiles.addWidget(st)
        v.addLayout(tiles)
        sec = Section('Per rig', f'{len(rep["labels"])} scenario runs; objects within '
                                 f'{rep["settings"]["radius_m"]:g} m of the car; seam bands '
                                 f'±{rep["settings"]["band_deg"]:g}°; 95 % intervals over {rep["n_units"]} '
                                 f'{rep["unit"]}s.')
        ids = list(rep['variants'])
        self.table = QTreeWidget()
        self.table.setHeaderLabels(['number'] + [rep['variants'][i]['name'] for i in ids])
        self.table.setRootIsDecorated(False)
        metrics = []
        for i in ids:
            for r in rep['variants'][i]['table']:
                if r['metric'] not in metrics:
                    metrics.append(r['metric'])
        for m in metrics:
            row = [m]
            for i in ids:
                r = next((x for x in rep['variants'][i]['table'] if x['metric'] == m), None)
                row.append(_fmt(r))
            self.table.addTopLevelItem(QTreeWidgetItem(row))
        self.table.setMinimumHeight(260)
        sec.add(self.table)
        seams = [f'{rep["variants"][i]["name"]}: ' + ', '.join(f'{s["a"]}|{s["b"]} {s["angle_deg"]:+.1f}° '
                                                                f'({s["source"]})' for s in rep['variants'][i]['seams'])
                 for i in ids if rep['variants'][i].get('seams')]
        if seams:
            sec.add(label('Seams — ' + '; '.join(seams), muted=True, small=True, wrap=True))
        v.addWidget(sec)
        from .theme import THEME
        self.pairs = []
        for p in rep['pairs']:
            ps = Section(f'{rep["variants"][p["b"]]["name"]} against {rep["variants"][p["a"]]["name"]}',
                         f'paired over {p["n_objects"]} shared object-steps: worse or better where a cluster t-test '
                         'over the runs, Holm-adjusted over every number, gives p < 0.05; else no detectable change.')
            tw = QTreeWidget()
            tw.setHeaderLabels(['number', 'A', 'B', 'B − A [95 % CI]', 'p (Holm)', 'verdict'])
            tw.setRootIsDecorated(False)
            for r in p['rows']:
                d = '—' if r['diff'] is None else f'{r["diff"]:+.3f}' + (
                    '' if r['ci_low'] is None else f'  [{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]')
                it = QTreeWidgetItem([r['metric'], '—' if r['a'] is None else f'{r["a"]:.3f}',
                                      '—' if r['b'] is None else f'{r["b"]:.3f}', d,
                                      '—' if r.get('p_holm') is None else f'{r["p_holm"]:.3f}', r['verdict']])
                if r['verdict'] != 'no detectable change':
                    it.setForeground(5, QColor(THEME.status('critical' if r['verdict'] == 'worse' else 'good')))
                tw.addTopLevelItem(it)
            tw.setMinimumHeight(220)
            ps.add(tw)
            self.pairs.append(tw)
            v.addWidget(ps)
        lim = Section('What this does not show')
        for x in rep.get('limits') or []:
            lim.add(label('• ' + x, muted=True, small=True, wrap=True))
        v.addWidget(lim)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll(body))
        row = QHBoxLayout()
        row.addWidget(label(os.path.join(study, 'report.md'), muted=True, small=True))
        row.addStretch(1)
        row.addWidget(button('Vantage design-log note…', self.export_note,
                             tip='a note for Vantage\'s design log, written to a file of your choosing (never into '
                                 'the .vantage project)'))
        row.addWidget(button('Close', self.accept))
        outer.addLayout(row)

    def export_note(self, path: str | None = None) -> str:
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Vantage design-log note', os.path.join(
                os.path.expanduser('~'), f'{os.path.basename(self.study)}-design-log.json'), 'JSON (*.json)')
            if not path:
                return ''
        if path.lower().endswith('.vantage'):
            QMessageBox.warning(self, 'Note', 'The note goes into its own file, never into a .vantage project.')
            return ''
        from ..core.provenance import write_json
        write_json(path, RC.design_log(self.rep))
        return path


def open_studies(page):
    """Rigs ▸ Rig studies…: the workspace's studies; the chosen one's results."""
    from PyQt6.QtWidgets import QInputDialog
    from ..sim.rigstudy import list_studies
    rows = [s for s in list_studies(page.win.workspace()) if s['reported']]
    if not rows:
        QMessageBox.information(page, 'Rig studies', 'No rig study has a report yet: Compare rigs by perception… '
                                                     'queues one.')
        return None
    names = [f'{s["id"]} — {", ".join(s["rigs"])}' for s in rows]
    pick, ok = QInputDialog.getItem(page, 'Rig studies', 'Study', names, 0, False)
    if not ok:
        return None
    dlg = RigStudyDialog(rows[names.index(pick)]['dir'], page)
    dlg.exec()
    return dlg
