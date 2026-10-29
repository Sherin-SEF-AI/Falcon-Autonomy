"""Data ▸ Health: what a set of drives holds and lacks (core.datahealth): class balance, condition coverage, scenario
values, near-duplicate frames and label sanity, with "record more of" suggestions that open Record set up for them."""
from __future__ import annotations

import os

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QHeaderView, QSpinBox, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ...core import datahealth as DH
from .. import tasks
from ..drivesets import drive_sets, index_entries, resolve
from ..widgets import EmptyState, Form, KeyValues, Section, button, label, set_enabled

BARS = ' ▁▂▃▄▅▆▇█'


def spark(counts) -> str:
    m = max(counts or [0])
    return ''.join(BARS[0] if not c else BARS[max(1, round(8 * c / m))] for c in counts) if m else '·' * len(counts)


def _tree(headers, minh=120):
    t = QTreeWidget()
    t.setHeaderLabels(headers)
    t.setRootIsDecorated(False)
    t.setMinimumHeight(minh)
    t.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    return t


class HealthTab(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.report = None
        self._busy = False
        v = QVBoxLayout(self)
        sec = Section('Dataset health', 'Measured from the drives\' own files: what they hold, what they lack, and '
                                        'what to record next (docs/runs.md).')
        f = Form()
        self.source = QComboBox()
        self.max_frames = QSpinBox()
        self.max_frames.setRange(100, 100000)
        self.max_frames.setValue(3000)
        self.max_frames.setSingleStep(500)
        f.row('Drives', self.source)
        f.row('Frames read', self.max_frames, 'at most, evenly over the set')
        sec.add(f)
        row = QHBoxLayout()
        self.go_btn = button('Analyse', self.analyse, role='primary')
        row.addWidget(self.go_btn)
        self.state = label('', muted=True)
        row.addWidget(self.state, 1)
        sec.add(row)
        v.addWidget(sec)
        self.empty = EmptyState('Choose the drives (all of them, Browse\'s selection or a drive set) and Analyse.')
        v.addWidget(self.empty, 1)
        self.body = QWidget()
        bv = QVBoxLayout(self.body)
        bv.setContentsMargins(0, 0, 0, 0)
        self.summary = KeyValues()
        s = Section('Summary')
        s.add(self.summary)
        bv.addWidget(s)
        lr = QHBoxLayout()
        s = Section('Class balance', 'Objects per class in the labelled frames read.')
        self.classes = _tree(['Class', 'Objects', 'Share', ''])
        s.add(self.classes)
        lr.addWidget(s, 1)
        s = Section('Record more of', 'The empty and thinnest cells of world × time of day × weather. Record opens '
                                      'with that world and the CARLA preset that gives the cell.')
        self.sugg = _tree(['Cell', 'Frames', 'Preset'])
        self.sugg.itemDoubleClicked.connect(lambda it, _c: self.record_cell(it))
        s.add(self.sugg)
        rb = QHBoxLayout()
        self.rec_btn = button('Record this…', lambda: self.record_cell(self.sugg.currentItem()),
                              tip='Open Data ▸ Record with the chosen cell\'s world and weather')
        rb.addWidget(self.rec_btn)
        rb.addStretch(1)
        s.add(rb)
        lr.addWidget(s, 1)
        bv.addLayout(lr)
        s = Section('Condition coverage', 'Frames per world, time of day (sun altitude) and weather kind; grey: none.')
        self.cover = _tree(['World', 'Time of day', 'Weather', 'Frames'], 160)
        s.add(self.cover)
        bv.addWidget(s)
        s = Section('Scenario values', 'Each parameter\'s recorded values over its range, in ten bins.')
        self.values = _tree(['Scenario', 'Parameter', 'Range', 'Runs', 'Distribution', 'Empty bins'], 80)
        s.add(self.values)
        bv.addWidget(s)
        self.body.hide()
        v.addWidget(self.body, 1)

    def refresh(self):
        cur = self.source.currentData()
        self.source.blockSignals(True)
        self.source.clear()
        self.source.addItem('All recorded drives', ('all', None))
        self.source.addItem('The runs selected in Browse', ('browse', None))
        for ds in drive_sets(self.win):
            self.source.addItem(f'Drive set: {ds["name"]}', ('set', ds['id']))
        k = self.source.findData(cur)
        self.source.setCurrentIndex(k if k >= 0 else 0)
        self.source.blockSignals(False)

    def runs(self) -> list:
        kind, ident = self.source.currentData() or ('all', None)
        entries = index_entries(self.win)
        if kind == 'browse':
            return self.page.browse.selected_runs()
        if kind == 'set':
            ds = self.win.project.item('drive_sets', ident)
            return resolve(self.win, ds, entries) if ds else []
        return [e['path'] for e in entries if e.get('frames') and e.get('status') == 'ok']

    def analyse(self):
        if self._busy:
            return
        runs = self.runs()
        if not runs:
            self.state.setText('No drives to analyse: record one, or choose another set.')
            return
        ws, doc, mx = self.win.workspace(), self.win.project.doc, self.max_frames.value()
        self._busy = True
        set_enabled(self.go_btn, False, 'An analysis is running')
        self.state.setText(f'reading {len(runs)} run(s)…')
        tasks.run(self, lambda: DH.analyse(runs, ws, doc, max_frames=mx), self._done, self._failed)

    def _failed(self, e):
        self._busy = False
        set_enabled(self.go_btn, True)
        self.state.setText(f'failed: {e}')

    def _done(self, r):
        self._busy = False
        set_enabled(self.go_btn, True)
        self.report = r
        self.empty.hide()
        self.body.show()
        self.state.setText(f'{len(r["runs"])} run(s), {r["frames_read"]} of {r["frames"]} frames read')
        dup = r['duplicates']
        lab = r['labels']
        self.summary.clear()
        every = 'every frame' if r['every'] == 1 else f'every {r["every"]}th frame'
        self.summary.set('Drives', f'{len(r["runs"])} runs, {r["frames"]} frames ({every} read: {r["frames_read"]})')
        self.summary.set('Labels', f'{r["objects"]} objects in {r["labelled_frames"]} labelled camera frames'
                         if r['labelled_frames'] else 'no box labels recorded (the rig\'s cameras need the boxes2d '
                                                      'layer)')
        if r['labelled_frames']:
            self.summary.set('Empty frames', f'{r["empty_frames"]} ({100 * r["empty_frames"] / r["labelled_frames"]:.1f} %)'
                             ' with no labelled object')
            self.summary.set('Box sanity', f'{lab["tiny"]} tiny ({lab["tiny_rule"]}), {lab["degenerate"]} without area, '
                                           f'{lab["outside"]} outside the image')
        if dup['hashed']:
            self.summary.set('Near duplicates', f'{dup["near_duplicates"]} of {dup["hashed"]} frames '
                                                f'({100 * dup["share"]:.1f} %) within {dup["bits"]} bits of an '
                                                'earlier frame (dHash)',
                             'Mostly a stopped car\'s identical frames; examples: ' + '; '.join(
                                 f'{os.path.basename(x["run"])} step {x["step"]} ≈ step {x["like"]["step"]}'
                                 for x in dup['examples'][:4]))
        self.summary.set('Coverage', f'{r["coverage"]["empty"]} of {r["coverage"]["of"]} cells empty '
                                     '(world × day, low sun, night × clear, cloudy, wet, rain)')
        self.classes.clear()
        tot = max(r['objects'], 1)
        for c, n in r['classes'].items():
            self.classes.addTopLevelItem(QTreeWidgetItem([c, str(n), f'{100 * n / tot:.1f} %',
                                                          '█' * max(1, round(30 * n / tot))]))
        names = {w['id']: w['name'] for w in self.win.project.doc['worlds']}
        self.cover.clear()
        grey = QColor(self.palette().placeholderText().color())
        for c in sorted(r['coverage']['cells'], key=lambda c: (c['world'], c['tod'], c['weather'])):
            it = QTreeWidgetItem([names.get(c['world'], c['world']), c['tod'], c['weather'], str(c['frames'])])
            if not c['frames']:
                for k in range(4):
                    it.setForeground(k, grey)
            self.cover.addTopLevelItem(it)
        self.sugg.clear()
        for s in r['suggestions']:
            it = QTreeWidgetItem([s['text'], str(s['frames']), s['preset'] or '—'])
            it.setData(0, 256, s)
            self.sugg.addTopLevelItem(it)
        if self.sugg.topLevelItemCount():
            self.sugg.setCurrentItem(self.sugg.topLevelItem(0))
        set_enabled(self.rec_btn, bool(r['suggestions']), 'Nothing to suggest: every cell has frames')
        self.values.clear()
        for h in r['values']:
            self.values.addTopLevelItem(QTreeWidgetItem([h['scenario'], h['param'], f'{h["lo"]:g} – {h["hi"]:g}',
                                                         str(h['runs']), spark(h['counts']),
                                                         f'{h["empty_bins"]} of {len(h["counts"])}']))
        for t in (self.classes, self.cover, self.sugg, self.values):
            for k in range(t.columnCount()):
                t.resizeColumnToContents(k)

    def record_cell(self, it):
        s = it.data(0, 256) if it is not None else None
        if s:
            self.page.record_in(s['world'], s.get('preset'))
            self.page.record.note.setText(f'For the health check\'s gap: {s["text"]}. The rig, length and seed are '
                                          'yours to set.')

