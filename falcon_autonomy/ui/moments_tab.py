"""The Replay page's Moments section: every recorded run searched for what happened in it, in plain words
(core.moments, docs/moments.md).

A query box ("two-wheeler cuts in from the left at night in rain"), the filter as it was understood in mono — and the
words that were not, in the warning colour with the words "not understood" — and the hits: type, actors, side, min
TTC, run, time and a severity meter beside its number. A partial hit says what it misses. Double-click (or Open)
shows the run on the Data page at that time; an imported log's moment loads the log here. Make a scenario… writes the
selected hit as a scenario and OpenSCENARIO through core.incident's path; Drive set… adds the hits' runs to a new
drive set (one undoable project edit). Index reads new or changed runs in the background (the same code as `falcon
moments index`). Everything is local: no model, no network. Runs derived from dashcam footage are listed here — this
is the user's own screen — marked private.

4.2: each hit shows its run's fidelity (core.fidelity: carla, fake, recorded …) and the search can require one (the
Fidelity box, or words such as "on the fake simulator"); *Open in Replay* shows a hit's run on the Replay page when this
build's Replay page can load a run (else it says so and Open shows it on the Data page); a hit row selects its run as
the window's active one and has *Send to ▸*. The section registers the Send-to target *Moments*: a run sent here is
indexed when it is not yet, then its moments are listed (*Show all* goes back to the search)."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QInputDialog, QLineEdit, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ..core import fidelity as FD
from ..core import moments as M
from . import motion, tasks
from .study_bits import link_runs
from .theme import THEME, mono
from .widgets import Meter, button, fidelity_item, label, short_id

COLUMNS = ('Type', 'Actors', 'Side', 'Min TTC', 'Run', 'Time', 'Severity', 'Fidelity')
REPLAY_LOADERS = ('load_run', 'open_run', 'show_run')     # what a Replay page may offer to load one run
EXAMPLE = 'two-wheeler cuts in from the left at night in rain'


def severity_word(s: float) -> str:
    """The severity as a word beside its number (the colour is never the only cue)."""
    return 'collision' if s >= 1.0 else 'high' if s >= 0.6 else 'medium' if s >= 0.35 else 'low'


class MomentsTab(QWidget):
    """Built by the Replay page (ui.widgets.optional_tab) as MomentsTab(page)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.win = page.win
        self.hits: list = []
        self.shown: set = set()
        self.busy = False
        self.run_filter = ''                                  # a run sent here: its moments only
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText(EXAMPLE)
        self.query.setToolTip('Plain words, understood by a fixed vocabulary on this machine (no model): a moment '
                              'type, actor classes (auto, two wheeler, bike, cow …), a side, light, weather, and '
                              'numbers such '
                              'as "TTC under 1.5 s" or "above 40 km/h". Empty: every moment, most severe first.')
        self.query.returnPressed.connect(self.search)
        row.addWidget(self.query, 1)
        self.fidelity = QComboBox()
        self.fidelity.addItem('Any fidelity', '')
        for lv in FD.LEVELS:
            if lv != 'mixed':
                self.fidelity.addItem(FD.label(lv), lv)
        self.fidelity.setToolTip('Only moments of runs of this fidelity (core.fidelity); words such as "on the fake '
                                 'simulator" or "in CARLA" in the query do the same')
        self.fidelity.currentIndexChanged.connect(lambda _: self.search())
        row.addWidget(self.fidelity)
        self.search_btn = button('Search', self.search, tip='falcon moments search')
        self.index_btn = button('Index', self.index, tip='falcon moments index: read new or changed runs, imported '
                                                         'logs and registered near-miss rankings into the store')
        row.addWidget(self.search_btn)
        row.addWidget(self.index_btn)
        v.addLayout(row)
        self.understood = label('understood as: —', mono=True, sel=True, wrap=True)
        v.addWidget(self.understood)
        self.unknown = label('', mono=True, wrap=True)
        self.unknown.hide()
        v.addWidget(self.unknown)
        rr = QHBoxLayout()
        self.run_label = label('', small=True, mono=True)
        self.all_btn = button('Show all', self.show_all, tip='leave the one run and search every indexed run again')
        rr.addWidget(self.run_label)
        rr.addWidget(self.all_btn)
        rr.addStretch(1)
        self.run_row = QWidget()
        self.run_row.setLayout(rr)
        rr.setContentsMargins(0, 0, 0, 0)
        self.run_row.hide()
        v.addWidget(self.run_row)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(180)
        self.tree.itemDoubleClicked.connect(lambda it, _c: self.open(it))
        link_runs(self.tree, self.win, self._run_of, exclude='moments',
                  actions=lambda it: [('Open on the Data page', lambda: self.open(it)),
                                      ('Open in Replay', lambda: self.open_in_replay(it))])
        v.addWidget(self.tree, 1)
        b = QHBoxLayout()
        self.open_btn = button('Open', lambda: self.open(), tip='show the run on the Data page at the moment\'s time')
        self.replay_btn = button('Open in Replay', lambda: self.open_in_replay(),
                                 tip='the hit\'s run on the Replay page (an imported log loads there too)')
        self.scn_btn = button('Make a scenario…', self.ask_scenario,
                              tip='falcon moments show ID --scenario NAME: the moment as a scenario, its variations '
                                  'and OpenSCENARIO (moments/scenarios/NAME/)')
        self.set_btn = button('Drive set…', self.ask_drive_set,
                              tip='the runs of the hits matching every term, as a new drive set')
        for x in (self.open_btn, self.replay_btn, self.scn_btn, self.set_btn):
            b.addWidget(x)
        b.addStretch(1)
        v.addLayout(b)
        self.note = label('', muted=True, wrap=True)
        v.addWidget(self.note)
        try:
            from .context import register_target
            register_target(self.win, 'moments', 'Moments', self.take_run,
                            'Index the run if it is not yet, then list its moments (Replay ▸ Moments)')
        except (ImportError, AttributeError):
            pass

    # ------------------------------------------------------------------ data
    def workspace(self) -> str:
        try:
            return self.win.workspace() or ''
        except AttributeError:
            return ''

    def index(self):
        """Read new or changed sources in the background; search again when it is done."""
        ws = self.workspace()
        if not ws or self.busy:
            return
        os.makedirs(ws, exist_ok=True)
        self.busy = True
        self.index_btn.setEnabled(False)
        self.note.setText('Indexing the workspace\'s runs…')

        def done(res):
            self.busy = False
            self.index_btn.setEnabled(True)
            self.note.setText(f'{res["sources"]} source(s): {res["indexed"]} read, {res["unchanged"]} unchanged, '
                              f'{res["removed"]} removed; {res["moments"]} moment(s) in the store'
                              + (f'; {len(res["failed"])} without a recorded track' if res['failed'] else ''))
            self.search()

        def failed(e):
            self.busy = False
            self.index_btn.setEnabled(True)
            self.note.setText(f'Indexing failed: {e}')
        tasks.run(self, lambda: M.index(ws), done, failed)

    def search(self):
        ws = self.workspace()
        self.run_filter = ''
        self.run_row.hide()
        pq = M.parse_query(self.query.text())
        if self.fidelity.currentData():                     # the box over the words
            pq['filter']['fidelity'] = self.fidelity.currentData()
            pq['understood'] = M.understood(pq['filter'])
        self.understood.setText(f'understood as: {pq["understood"]}')
        if pq['not_understood']:
            self.unknown.setText('not understood: ' + ', '.join(pq['not_understood']))
            self.unknown.setStyleSheet(f'color: {THEME.status("warning")}')
        self.unknown.setVisible(bool(pq['not_understood']))
        if not ws or not os.path.isfile(M.db_path(ws)):
            self.hits = []
            self.tree.clear()
            self.note.setText('No moment index in this workspace yet: Index reads the recorded runs.')
            return []
        try:
            res = M.search(ws, pq['filter'], limit=200)
        except M.MomentsError as e:
            self.note.setText(str(e))
            return []
        self.show_hits(res)
        return res['hits']

    def show_hits(self, res: dict):
        old = self.shown
        self.hits = res['hits']
        self.tree.clear()
        fresh = []
        for m in self.hits:
            ttc = '—' if m.get('min_ttc_s') is None else f'{m["min_ttc_s"]:.2f} s'
            sev = float(m.get('severity') or 0.0)
            run = m['rel'] + ('  · private' if m['private'] else '')
            if m['match'] < 1.0:
                run += f'  · partial: not {", ".join(m["missed"])}'
            it = QTreeWidgetItem([M.NAMES.get(m['type'], m['type']), ', '.join(m['actors']) or '—',
                                  m.get('side') or '—',
                                  ttc, run, f'{m["t0"]:.1f}–{m["t1"]:.1f} s', f'{sev:.2f} {severity_word(sev)}', ''])
            fidelity_item(it, 7, m.get('fidelity') or 'unknown')
            for k in (3, 5, 6):
                it.setFont(k, mono(12))
            it.setToolTip(4, m['run'])
            it.setToolTip(6, M.SEVERITY_NOTE)
            it.setData(0, Qt.ItemDataRole.UserRole, m['id'])
            self.tree.addTopLevelItem(it)
            meter = Meter(70)
            meter.set(sev)
            cell = QWidget()
            h = QHBoxLayout(cell)
            h.setContentsMargins(96, 0, 6, 0)
            h.addWidget(meter, 0, Qt.AlignmentFlag.AlignVCenter)
            h.addStretch(1)
            self.tree.setItemWidget(it, 6, cell)
            if m['id'] not in old:
                fresh.append(it)
        for k in range(len(COLUMNS) - 2):
            self.tree.resizeColumnToContents(k)
        self.tree.setColumnWidth(6, 190)
        self.tree.resizeColumnToContents(7)
        if old:                                         # a first listing is not "new"; later arrivals ease in
            for it in fresh[:40]:
                motion.flash_item(it, len(COLUMNS))
        self.shown = {m['id'] for m in self.hits}
        n = res['total']
        self.note.setText(f'{n} moment(s), {res["full"]} matching every term'
                          + (f' (the first {len(self.hits)} shown)' if n > len(self.hits) else '')
                          + ('. Nothing matches: try fewer words, or Index if runs are new.' if not n else ''))

    def selected(self, it=None) -> dict | None:
        it = it or self.tree.currentItem()
        if it is None:
            return None
        mid = it.data(0, Qt.ItemDataRole.UserRole)
        return next((m for m in self.hits if m['id'] == mid), None)

    def _run_of(self, it):
        m = self.selected(it)
        if m is None or m['source'] == 'nearmiss':
            return ''
        return {'run': m['run']}

    # ------------------------------------------------------------------ a run sent here (ui.context)
    def take_run(self, run: str):
        """The Send-to target: go to the Replay page, index the run when it is not yet, then list its moments."""
        if hasattr(self.win, 'go'):
            self.win.go('replay')
        ws = self.workspace()
        run = os.path.abspath(run)
        self.run_filter = run
        if not ws:
            self.note.setText('No workspace: save the project first.')
            return None
        if M.indexed(ws, run):
            return self.show_run(run)
        if self.busy:
            self.note.setText('Indexing is running: send the run again when it ends.')
            return None
        os.makedirs(ws, exist_ok=True)
        self.busy = True
        self.index_btn.setEnabled(False)
        self.note.setText(f'Indexing {os.path.basename(run)}…')

        def done(res):
            self.busy = False
            self.index_btn.setEnabled(True)
            if res['failed']:
                self.note.setText(f'{os.path.basename(run)} was not indexed: {res["failed"][0]["why"]}')
            self.show_run(run)

        def failed(e):
            self.busy = False
            self.index_btn.setEnabled(True)
            self.note.setText(f'Indexing failed: {e}')
        tasks.run(self, lambda: M.index(ws, only=[run]), done, failed)
        return None

    def show_run(self, run: str) -> list:
        """The moments of one indexed run, in time order (a run sent here)."""
        ws = self.workspace()
        hits = M.run_moments(ws, run) if ws else []
        self.run_filter = run
        self.run_label.setText(f'Run {short_id(os.path.basename(run.rstrip(os.sep)), 12)}: its moments')
        self.run_label.setToolTip(run)
        self.run_row.show()
        self.understood.setText('understood as: the moments of one run')
        self.unknown.hide()
        self.show_hits({'hits': hits, 'total': len(hits), 'full': len(hits)})
        if not hits:
            self.note.setText('No moment in this run (nothing the detectors look for happened, or it has no '
                              'recorded track).')
        return hits

    def show_all(self):
        self.run_filter = ''
        self.run_row.hide()
        return self.search()

    # ------------------------------------------------------------------ actions on a hit
    def open_in_replay(self, it=None):
        """The hit's run on the Replay page: an imported log loads there; a run through the page's run loader when
        this build has one (else the note says so and Open shows it on the Data page)."""
        m = self.selected(it)
        if m is None:
            self.note.setText('Select a moment first.')
            return None
        if m['source'] == 'nearmiss':
            self.note.setText('A near-miss clip\'s event opens from the Near-misses section (its clip is not in the '
                              'index).')
            return None
        page = self.win.go('replay') if hasattr(self.win, 'go') else self.page
        page = page or self.page
        if m['source'] == 'log' and hasattr(page, 'load_log'):
            page.load_log(m['run'])
            return m
        for name in REPLAY_LOADERS:
            fn = getattr(page, name, None)
            if callable(fn):
                fn(m['run'])
                if hasattr(page, 'seek_time'):
                    page.seek_time(m['t0'])
                return m
        self.note.setText('This build\'s Replay page cannot load a run yet: Open shows it on the Data page at the '
                          'moment\'s time.')
        return None

    def open(self, it=None):
        """The run on the Data page at the moment's time (a log's moment: the log, on this page)."""
        m = self.selected(it)
        if m is None:
            return None
        if m['source'] == 'nearmiss':
            self.note.setText('A near-miss clip\'s event opens from the Near-misses section (its clip is not in the '
                              'index).')
            return None
        if m['source'] == 'log':
            self.page.load_log(m['run'])
            return m
        data = self.win.go('data')
        data.show_run(m['run'])
        data.browse.seek_time(m['t0'])
        return m

    def ask_scenario(self):
        m = self.selected()
        if m is None:
            self.note.setText('Select a moment first.')
            return
        name, ok = QInputDialog.getText(self, 'Make a scenario', 'Name (letters, digits, _ . -):',
                                        text=f'{m["type"].replace("_", "-")}-{m["id"]}')
        if ok and name:
            self.make_scenario(name.strip())

    def make_scenario(self, name: str) -> dict | None:
        m = self.selected()
        if m is None:
            return None
        try:
            res = M.make_scenario(self.workspace(), m, name)
        except M.MomentsError as e:
            self.note.setText(str(e))
            return None
        self.note.setText(f'Scenario written: {res["dir"]} ({res["actors"]} actor(s), {res["parameters"]} search '
                          f'parameter(s){"; private" if res["private"] else ""}). Open variations.json from the '
                          'Scenarios page, or search it: falcon search … --scenario variations.json')
        return res

    def ask_drive_set(self):
        name, ok = QInputDialog.getText(self, 'Drive set', 'Name of the new drive set:')
        if ok and name.strip():
            self.drive_set(name.strip())

    def drive_set(self, name: str):
        """One undoable project edit: a drive set of the runs of the hits that match every term."""
        hits = [m for m in self.hits if m['source'] == 'run' and m['match'] == 1.0]
        if not hits:
            self.note.setText('No run among the hits to put in a drive set.')
            return None
        edit, ds = M.drive_set_edit(self.win.project.doc, self.workspace(), name, hits)
        self.win.edit(edit)
        self.note.setText(f'Drive set {name}: {len(ds["runs"])} run(s) (Data ▸ Browse lists it; Edit ▸ Undo removes '
                          'it).')
        return ds
