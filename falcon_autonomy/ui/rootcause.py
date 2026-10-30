"""The answer to "Why did this fail?" (core.rootcause): a view of a rootcause.json — a failing case's factors with
whether removing each fixes it and whether it alone fails, an evaluation's correlates of misses with intervals, or a
policy failure by its inputs — and the window the Rare cases and Models pages open it in when the job ends."""
from __future__ import annotations

import os
import sys

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..core.jobs import JobSpec
from ..core.provenance import read_json
from ..core.rootcause import out_dir  # noqa: F401 (the folder of a why-job: core.rootcause.out_dir)
from .theme import STATUS
from .widgets import KeyValues, button, label

YN = {True: 'yes', False: 'no', None: '—'}


def submit(win, argv: list, title: str):
    """Queue `falcon why …` as a job (its drives queue behind the CARLA lock in its own process)."""
    job = win.queue.submit(JobSpec('why', title, [sys.executable, '-m', 'falcon_autonomy', 'why'] + argv,
                                   memory_gb=1.0))
    win.jobs_dock.raise_()
    return job


class RootCauseView(QWidget):
    """A rootcause.json shown as its summary, its honest wording and its table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.summary = label('', wrap=True, sel=True)
        self.summary.setProperty('h', True)
        v.addWidget(self.summary)
        self.wording = label('', muted=True, wrap=True)
        v.addWidget(self.wording)
        self.info = KeyValues(('Case', 'Test', 'Control re-run', 'Re-runs', 'Note'))
        v.addWidget(self.info)
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setMinimumHeight(180)
        v.addWidget(self.table, 1)
        self.doc = None

    def show_doc(self, doc: dict | None):
        self.doc = doc
        self.table.clear()
        self.info.clear()
        if not doc:
            self.summary.setText('')
            self.wording.setText('')
            return
        self.summary.setText(doc.get('summary') or '')
        self.wording.setText(doc.get('wording') or '')
        kind = doc.get('kind')
        if kind == 'counterfactual':
            c = doc['case']
            self.info.set('Case', f'{c.get("scenario")} · {c.get("actor") or c.get("track_id") or "the run"} · '
                                  f'{os.path.basename(c["run"])}')
            self.info.set('Test', f'{c["failure"]} (threshold {c["threshold"]:g}) — the case: '
                                  f'{doc["base"]["detail"]}')
            ctl = doc['control']
            self.info.set('Control re-run', f'{"fails again" if ctl["fails"] else "does not fail"}; '
                                            f'{ctl.get("repeat") or ""}')
            self.info.set('Re-runs', str(doc['n_runs']))
            self.table.setHeaderLabels(['Factor', 'In this case', 'Removed by', 'Removing it alone fixes it',
                                        'It alone still fails', 'Verdict'])
            for f in doc['factors']:
                fixes = YN[f.get('removing_fixes')] + (f' ({f["fixed_in"]})' if f.get('fixed_in') and
                                                       f['factor'] == 'appearance' else '')
                it = QTreeWidgetItem([f['title'], f['in_case'] or '', f['intervention'] if f.get('applicable') else
                                      (f.get('why_not') or 'not present'), fixes, YN[f.get('alone_fails')],
                                      f.get('verdict') or ('' if f['present'] else 'not present in this case')])
                it.setToolTip(5, f.get('caveat') or '')
                if f.get('removing_fixes'):
                    for k in range(6):
                        it.setForeground(k, QColor(STATUS['critical']))
                self.table.addTopLevelItem(it)
        elif kind == 'static':
            m = doc.get('model') or {}
            self.info.set('Case', f'evaluation of {m.get("name")}: {doc["n_objects"]} objects, {doc["n_missed"]} '
                                  'missed')
            self.info.set('Test', f'missed at IoU ≥ {doc["matching"]["iou"]}, score ≥ {doc["matching"]["min_score"]}')
            self.info.set('Re-runs', 'none (correlational)')
            self.info.set('Note', doc.get('note') or f'95 % bootstrap over {doc["n_units"]} {doc["unit"]}s')
            self.table.setHeaderLabels(['Attribute', 'Spearman ρ', '95 % interval', 'Objects', 'Miss rate by bin'])
            for a in doc['attributes']:
                if a.get('spearman') is not None:
                    bins = '; '.join(f'{b["lo"]:.3g}–{b["hi"]:.3g}: {b["miss_rate"]:.2f}' for b in a['bins'])
                    it = QTreeWidgetItem([a['attribute'], f'{a["spearman"]:+.2f}',
                                          f'[{a["ci_low"]:+.2f}, {a["ci_high"]:+.2f}]' if a['ci_low'] is not None
                                          else '—', str(a['n']), bins])
                    if a.get('clear'):
                        for k in range(5):
                            it.setForeground(k, QColor(STATUS['critical']))
                elif a.get('categories'):
                    it = QTreeWidgetItem([a['attribute'], '', '', str(a['n']), '; '.join(
                        f'{c["value"]}: {c["miss_rate"]:.2f}' for c in a['categories'])])
                else:
                    it = QTreeWidgetItem([a['attribute'], '', '', str(a.get('n', '')), a.get('note') or ''])
                self.table.addTopLevelItem(it)
        else:
            e = doc['episode']
            self.info.set('Case', f'episode {e.get("episode")} (seed {e.get("seed")}), '
                                  f'{(e.get("infraction") or {}).get("kind")}')
            again = 'again' if doc['control']['infraction_recurs'] else 'not met'
            self.info.set('Control re-run', f'infraction {again}')
            self.info.set('Re-runs', str(doc['n_runs']))
            self.table.setHeaderLabels(['Input group', 'Hidden as', 'Removing it prevents the infraction'])
            for g in doc['groups']:
                it = QTreeWidgetItem([g['title'], g.get('intervention') or g.get('why_not') or '',
                                      YN[g.get('removing_fixes')]])
                self.table.addTopLevelItem(it)
        for k in range(self.table.columnCount()):
            self.table.resizeColumnToContents(k)

    def load(self, folder: str) -> bool:
        try:
            self.show_doc(read_json(os.path.join(folder, 'rootcause.json')))
            return True
        except (OSError, ValueError):
            return False


class RootCauseDialog(QDialog):
    """A non-modal window with the attribution of a finished job (or why there is none)."""

    def __init__(self, parent, folder: str, job=None):
        super().__init__(parent)
        self.setWindowTitle('Why did this fail?')
        self.resize(1000, 520)
        v = QVBoxLayout(self)
        self.view = RootCauseView()
        v.addWidget(self.view, 1)
        self.folder = folder
        if not self.view.load(folder):
            run = {}
            try:
                run = read_json(os.path.join(folder, 'run.json'))
            except (OSError, ValueError):
                pass
            why = run.get('error') or (job.summary() if job is not None else 'no attribution was written')
            self.view.summary.setText(f'No attribution: {why}')
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(button('Report…', self.report))
        h.addWidget(button('Close', self.close))
        v.addLayout(h)

    def report(self):
        if self.view.doc is not None:
            self.parent().win.go('reports').make('rootcause', self.folder)
