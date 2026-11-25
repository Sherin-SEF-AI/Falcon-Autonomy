"""Worlds ▸ Imported place ▸ Known import gaps: what an OpenStreetMap import cannot represent, each with its count, what
was done about it, and a tooltip saying what it means (core.importgaps). From the world's import.json when the import
recorded it (4.2 on); for an older world counted from its stored area.osm in the background (marked so); else "not
recorded for this world". The city's and the speed breakers' left-outs come from their own reports, so they follow a
rebuild."""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from ..core import importgaps as G
from .widgets import clear_layout, label

_cache: dict = {}            # area.osm path -> (mtime, gaps) computed for worlds imported before 4.2


def gaps_from_area(area: str, rep: dict) -> dict:
    """Gaps of an older world, from its area.osm and its import.json's settings and centre (runs off the GUI thread)."""
    from ..core import osm as O
    data = O.read_osm(area)
    s = rep.get('settings') or {}
    return G.compute(data, tuple(s.get('way_types') or O.CARLA_HIGHWAYS), rep.get('centre'),
                     float(s.get('lane_width', 3.35)))


class ImportGaps(QWidget):
    """The compact list (title · count · handled), one row per gap."""

    def __init__(self, owner=None):
        super().__init__()
        self.owner = owner
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        self.state = label('', muted=True, small=True, wrap=True)
        v.addWidget(self.state)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(2)
        self.grid.setColumnStretch(2, 1)
        v.addLayout(self.grid)
        self.rows: dict = {}                  # key -> (title, count, handled) labels
        self.gaps: dict | None = None
        self._world = None

    # ------------------------------------------------------------------ view
    def show_gaps(self, gaps: dict | None, how: str = ''):
        clear_layout(self.grid)
        self.rows.clear()
        self.gaps = gaps
        if not gaps or not gaps.get('items'):
            self.state.setText(how or 'Not recorded for this world.')
            self.state.show()
            return
        self.state.setText(how)
        self.state.setVisible(bool(how))
        for r, it in enumerate(gaps['items']):
            tip = it['tip'] + (f'\n\n{G.detail_text(it)}' if G.detail_text(it) else '')
            t = label(it['title'], muted=True)
            n = '—' if it['count'] is None else str(it['count'])
            c = label(n, mono=True)
            c.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            h = label(it['handled'], small=True, wrap=True)
            h.setProperty('muted', True)
            for w in (t, c, h):
                w.setToolTip(tip)
                w.setAccessibleDescription(tip)
            self.grid.addWidget(t, r, 0, Qt.AlignmentFlag.AlignTop)
            self.grid.addWidget(c, r, 1, Qt.AlignmentFlag.AlignTop)
            self.grid.addWidget(h, r, 2)
            self.rows[it['key']] = (t, c, h)

    def show_world(self, w: dict, world_dir: str, rep: dict | None):
        """The gaps of an imported world: recorded, computed now from area.osm (in the background), or not recorded."""
        self._world = w.get('id')
        rep = rep or {}
        feats = None
        try:
            with open(os.path.join(world_dir, 'features.json'), encoding='utf-8') as fh:
                feats = json.load(fh)
        except (OSError, ValueError):
            pass

        def full(gaps):
            g = G.with_city(gaps, w.get('city'), (rep.get('osm') or {}).get('buildings'))
            return G.with_features(g, feats)

        rec = rep.get('import_gaps')
        if isinstance(rec, dict) and rec.get('items'):
            self.show_gaps(full(rec))
            return
        area = os.path.join(world_dir, 'area.osm')
        try:
            m = os.stat(area).st_mtime_ns
        except OSError:
            self.show_gaps(None, 'Not recorded for this world (imported before 4.2, and its area.osm is not in the '
                                 'workspace).')
            return
        hit = _cache.get(area)
        if hit is not None and hit[0] == m:
            self.show_gaps(full(hit[1]), 'Counted from the stored area.osm (imported before 4.2).')
            return
        self.show_gaps(None, 'Counting from the stored area.osm (imported before 4.2)…')
        from . import tasks
        wid = w.get('id')

        def done(gaps):
            _cache[area] = (m, gaps)
            if self._world == wid:
                self.show_gaps(full(gaps), 'Counted from the stored area.osm (imported before 4.2).')

        def failed(e):
            if self._world == wid:
                self.show_gaps(None, f'Not recorded for this world: area.osm could not be read ({e}).')
        tasks.run(self, lambda: gaps_from_area(area, rep), done, failed)

    def count(self, key: str):
        t = self.rows.get(key)
        return None if t is None else t[1].text()


def row_label(gaps_view: ImportGaps, key: str) -> QLabel | None:
    t = gaps_view.rows.get(key)
    return t[0] if t else None
