"""Reports: self-contained HTML reports of a loop iteration, a model evaluation, a dataset, a rare-case study or a
Leaderboard-style result (core.reports), made as jobs (`falcon report`), listed from the workspace's reports folder,
opened in the browser and exported to a folder; and outputs for a paper (core.paper: LaTeX and CSV tables, figures,
data cards) as `falcon paper` jobs."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QMessageBox, QTreeWidget,
                             QTreeWidgetItem)

from ...core import reports as R
from ...core.jobs import FINAL, JobSpec
from ...core.paths import tilde
from ..widgets import KeyValues, Page, PlaceholderTree, Section, button, label, optional_tab

KIND_TITLES = {'iteration': 'Loop iteration', 'evaluation': 'Model evaluation', 'dataset': 'Dataset coverage',
               'study': 'Rare-case study', 'leaderboard': 'Leaderboard result', 'rules': 'Traffic-rule check',
               'rootcause': 'Root cause', 'vlm': 'VLM explanations', 'deploy': 'Deployment (TensorRT)',
               'ci': 'CI run', 'gsn': 'Safety case (GSN)'}
PAPER_TITLES = {'experiment': 'Experiment (ablation or sim + real)', 'evaluation': 'Model evaluation',
                'comparison': 'Comparison of two evaluations', 'datacard': 'Data card of an export'}


class ReportsPage(Page):
    key, title, icon = 'reports', 'Reports', 'report'
    about = ('Self-contained HTML reports — no network assets, light and dark — of loop iterations, evaluations, '
             'datasets, rare-case studies, Leaderboard results and traffic-rule checks, each with the provenance of every '
             'number.')

    def __init__(self, win):
        super().__init__(win)
        self.jobs = {}
        v = self.page_layout()
        sec = Section('New report', 'A report shows what the source\'s files say and computes nothing new; it lists '
                                    'the SHA-256 of every file it read.')
        row = QHBoxLayout()
        self.kind = QComboBox()
        for k in R.KINDS:
            self.kind.addItem(KIND_TITLES[k], k)
        self.kind.currentIndexChanged.connect(self.fill_sources)
        self.src = QComboBox()
        self.src.setPlaceholderText('Nothing of this kind yet: make one first (its page says how)')
        self.src.setMinimumWidth(360)
        row.addWidget(label('Kind', muted=True))
        row.addWidget(self.kind)
        row.addWidget(label('Source', muted=True))
        row.addWidget(self.src, 1)
        row.addWidget(button('Make the report', self.make_selected, role='primary'))
        sec.add(row)
        v.addWidget(sec)
        v.addWidget(self._paper_section())
        sec = Section('Reports', 'In the workspace\'s reports folder. Double-click one to open it in the browser.')
        self.list = PlaceholderTree('No reports yet: choose a kind and a source above, then Make the report.')
        self.list.setHeaderLabels(['Created', 'Kind', 'Title', 'File'])
        self.list.setRootIsDecorated(False)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setMinimumHeight(220)
        self.list.setColumnWidth(0, 150)
        self.list.setColumnWidth(2, 380)
        self.list.itemDoubleClicked.connect(lambda it, _c: self.open(it.data(0, 256)))
        self.list.currentItemChanged.connect(lambda it, _p: self.show_meta(it))
        sec.add(self.list)
        row = QHBoxLayout()
        row.addWidget(button('Open in the browser', lambda: self.open(self._current())))
        row.addWidget(button('Export to a folder…', self.export_dialog))
        row.addWidget(button('Show the folder', lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(R.reports_dir(self.win.workspace())))))
        row.addStretch(1)
        sec.add(row)
        self.info = KeyValues(('Title', 'Kind', 'Source', 'Created', 'Files read', 'File'))
        sec.add(self.info)
        v.addWidget(sec)
        self.gsn = optional_tab(v, 'ui.gsn_tab', 'GsnTab', 'Safety case (GSN)', self)   # 4.0, in progress
        v.addStretch(1)
        win.bridge.finished.connect(self._finished)
        self.refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    # ------------------------------------------------------------------ outputs for papers (core.paper)
    def _paper_section(self):
        sec = Section('Outputs for a paper', 'LaTeX (booktabs) and CSV tables and PDF, PNG and SVG figures with their '
                                             'intervals — byte-identical for the same source, so they can be diffed '
                                             '— or an export\'s data card. Written into the workspace\'s paper '
                                             'folder; Data ▸ Compare exports a run comparison.')
        row = QHBoxLayout()
        self.pkind = QComboBox()
        for k, t in PAPER_TITLES.items():
            self.pkind.addItem(t, k)
        self.pkind.currentIndexChanged.connect(self.fill_paper_sources)
        self.psrc = QComboBox()
        self.psrc.setPlaceholderText('Nothing of this kind yet')
        self.psrc.setMinimumWidth(360)
        row.addWidget(label('Kind', muted=True))
        row.addWidget(self.pkind)
        row.addWidget(label('Source', muted=True))
        row.addWidget(self.psrc, 1)
        row.addWidget(button('Write the outputs', self.make_paper))
        row.addWidget(button('Show the folder', lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(os.path.join(self.win.workspace(), 'paper')))))
        sec.add(row)
        return sec

    def fill_paper_sources(self, _i=None):
        from ...core import paper as PP
        cur = self.psrc.currentData()
        self.psrc.clear()
        for name, path in PP.sources(self.win.workspace()).get(self.pkind.currentData(), []):
            self.psrc.addItem(name, path)
            self.psrc.setItemData(self.psrc.count() - 1, tilde(path), Qt.ItemDataRole.ToolTipRole)
        i = self.psrc.findData(cur)
        self.psrc.setCurrentIndex(i if i >= 0 else 0)

    def make_paper(self):
        """Queue `falcon paper KIND SRC --out <ws>/paper/<kind>-<name>` (a data card goes into the export itself)."""
        src, kind = self.psrc.currentData(), self.pkind.currentData()
        if not src:
            QMessageBox.information(self, 'Outputs for a paper', 'Nothing of this kind in the workspace yet.')
            return None
        out = src if kind == 'datacard' else os.path.join(self.win.workspace(), 'paper',
                                                          f'{kind}-{os.path.basename(os.path.normpath(src))}')
        job = self.win.queue.submit(JobSpec('export', f'Paper outputs: {PAPER_TITLES[kind]} '
                                                      f'({os.path.basename(src)})',
                                            [sys.executable, '-m', 'falcon_autonomy', 'paper', kind, src, '--out',
                                             out], memory_gb=0.5))
        self.jobs[job.id] = job
        self.win.jobs_dock.raise_()
        return job

    def folder(self):
        return R.reports_dir(self.win.workspace())

    def refresh(self):
        self.fill_sources()
        self.fill_paper_sources()
        sel = self._current()
        self.list.clear()
        for m in R.list_reports(self.folder()):
            t = QTreeWidgetItem([(m.get('created') or '').replace('T', ' ').rstrip('Z'),
                                 KIND_TITLES.get(m['kind'], m['kind']), m['title'], os.path.basename(m['path'])])
            t.setData(0, 256, m['path'])
            t.setData(1, 256, m)
            t.setToolTip(3, tilde(m['path']))
            self.list.addTopLevelItem(t)
            if m['path'] == sel:
                self.list.setCurrentItem(t)

    def fill_sources(self, _i=None):
        kind = self.kind.currentData()
        cur = self.src.currentData()
        self.src.clear()
        for name, path in R.sources(self.win.workspace()).get(kind, []):
            self.src.addItem(name, path)
            self.src.setItemData(self.src.count() - 1, tilde(path), Qt.ItemDataRole.ToolTipRole)
        i = self.src.findData(cur)
        self.src.setCurrentIndex(i if i >= 0 else 0)       # a placeholder keeps Qt from choosing the first itself

    def _current(self):
        it = self.list.currentItem()
        return it.data(0, 256) if it is not None else None

    def selected(self) -> list:
        return [it.data(0, 256) for it in self.list.selectedItems()]

    def show_meta(self, it):
        m = it.data(1, 256) if it is not None else None
        if not m:
            return
        self.info.set('Title', m['title'])
        self.info.set('Kind', KIND_TITLES.get(m['kind'], m['kind']))
        self.info.set('Source', tilde(m['source']))
        self.info.set('Created', m.get('created'))
        self.info.set('Files read', f'{len(m.get("inputs") or {})} (their SHA-256 are in the report)')
        self.info.set('File', tilde(m['path']))

    # ------------------------------------------------------------------ actions
    def make_selected(self):
        src = self.src.currentData()
        if not src:
            QMessageBox.information(self, 'Report', 'Nothing of this kind in the workspace yet.')
            return None
        return self.make(self.kind.currentData(), src)

    def make(self, kind: str, src: str):
        """Queue `falcon report KIND SRC --out <reports folder>/<name>.html`."""
        out = os.path.join(self.folder(), R.default_name(kind, src))
        os.makedirs(self.folder(), exist_ok=True)
        job = self.win.queue.submit(JobSpec('report', f'Report: {KIND_TITLES[kind]} ({os.path.basename(src)})',
                                            [sys.executable, '-m', 'falcon_autonomy', 'report', kind, src, '--out',
                                             out], memory_gb=0.5))
        self.jobs[job.id] = job
        self.win.jobs_dock.raise_()
        return job

    def _finished(self, job):
        if self.jobs.pop(job.id, None) is not None:
            self.refresh()
            if job.state == 'done' and job.result and job.result.get('path'):
                for i in range(self.list.topLevelItemCount()):
                    if self.list.topLevelItem(i).data(0, 256) == job.result['path']:
                        self.list.setCurrentItem(self.list.topLevelItem(i))

    def open(self, path):
        if path and os.path.isfile(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def export_dialog(self):
        paths = self.selected()
        if not paths:
            QMessageBox.information(self, 'Export', 'Select one or more reports.')
            return
        d = QFileDialog.getExistingDirectory(self, 'Export the reports to', os.path.expanduser('~'))
        if d:
            self.export_to(d, paths)

    def export_to(self, folder: str, paths: list | None = None) -> list:
        out = R.export(paths if paths is not None else self.selected(), folder)
        self.win.log.append_line(f'exported {len(out)} report(s) to {tilde(folder)}')
        return out

    def busy(self) -> bool:
        return any(j.state not in FINAL for j in self.jobs.values())
