"""Reports ▸ Safety case (GSN): the workspace's living safety case (core.gsn; docs/gsn.md) drawn in Goal Structuring
Notation, each node coloured by its status as verified now — green (its evidence holds), amber (missing or changed
evidence: re-bind to accept it), red (a claim violated: a bound broken), grey (undeveloped) — with the status
propagated up as AND. Clicking a node shows its text, status and why, and a Solution's evidence: the result it is
bound to, the claim, the pinned SHA-256 and when it was bound. Re-bind pins the result's current hash (undoable here:
Undo puts the previous evidence back); Verify now re-verifies and logs it (safety/gsn-events.jsonl); a new safety case
starts from the ISO 26262 or ISO 21448 SOTIF template; Export writes the self-contained HTML report (core.reports) or
an SVG."""
from __future__ import annotations

import copy
import json
import os

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QGraphicsItem, QGraphicsPathItem, QGraphicsScene,
                             QGraphicsTextItem, QGraphicsView, QHBoxLayout, QVBoxLayout, QWidget)

from ..core import gsn as G
from .theme import STATUS, THEME
from .widgets import KeyValues, button, label

COLOUR = {'green': STATUS['good'], 'amber': STATUS['warning'], 'red': STATUS['critical'], 'grey': '#9a9993'}
TEMPLATE_TITLES = (('sotif', 'ISO 21448 SOTIF'), ('iso26262', 'ISO 26262 functional safety'), ('blank', 'Blank'))


def _shape(t: str, r: QRectF) -> QPainterPath:
    p = QPainterPath()
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    if t == 'Strategy':
        p.addPolygon(QPolygonF([QPointF(x + 14, y), QPointF(x + w, y), QPointF(x + w - 14, y + h),
                                QPointF(x, y + h), QPointF(x + 14, y)]))
    elif t in ('Solution', 'Assumption', 'Justification'):
        p.addEllipse(r)
    elif t == 'Context':
        p.addRoundedRect(r, 22, 22)
    else:
        p.addRect(r)
    return p


class NodeItem(QGraphicsPathItem):
    def __init__(self, tab, n: dict, rect: QRectF, status: str | None):
        super().__init__(_shape(n['type'], rect))
        self.tab, self.node_id = tab, n['id']
        col = QColor(COLOUR.get(status, THEME.tok['muted'])) if status else QColor(THEME.tok['muted'])
        fill = QColor(col)
        fill.setAlpha(70 if status else 25)
        self.setBrush(QBrush(fill))
        self.setPen(QPen(col, 2.0 if status else 1.2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f'{n["id"]} ({n["type"]}): {status or "no status"}')
        txt = QGraphicsTextItem(self)
        txt.setDefaultTextColor(QColor(THEME.tok['ink']))
        f = QFont()
        f.setPointSizeF(8.0)
        txt.setFont(f)
        txt.setTextWidth(rect.width() - 16)
        head = f'<b>{n["id"]}</b>' + (f' · {status}' if status else '')
        body = (n.get('text') or '')[:150]
        txt.setHtml(f'{head}<br>{body.replace("<", "&lt;")}')
        txt.setPos(rect.x() + 8, rect.y() + 4)
        if n['type'] in ('Assumption', 'Justification'):
            tag = QGraphicsTextItem(n['type'][0], self)
            tag.setDefaultTextColor(QColor(THEME.tok['ink']))
            tag.setPos(rect.right() - 16, rect.bottom() - 20)

    def mousePressEvent(self, ev):                                # noqa: N802
        super().mousePressEvent(ev)
        self.tab.select(self.node_id)


class RebindCommand(QUndoCommand):
    """Re-bind a Solution's evidence (pin the result's current hash); undo restores the evidence it had."""

    def __init__(self, tab, nid: str, before: dict | None, after: dict):
        super().__init__(f'Re-bind {nid}')
        self.tab, self.nid, self.before, self.after = tab, nid, copy.deepcopy(before), copy.deepcopy(after)

    def _set(self, ev, what):
        ws = self.tab.workspace()
        doc = G.load(ws)
        n = G.node(doc, self.nid)
        if ev is None:
            n.pop('evidence', None)
        else:
            n['evidence'] = copy.deepcopy(ev)
        G.save(ws, doc)
        G.log_event(ws, what, 'gui', node=self.nid, evidence=ev, gsn_sha256=G.doc_sha(doc))
        self.tab.reload()

    def redo(self):
        self._set(self.after, 'bind')

    def undo(self):
        self._set(self.before, 'bind' if self.before else 'unbind')


class GsnTab(QWidget):
    """The safety-case view (module doc). owner: the Reports page (owner.win the window)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.page = owner
        self.win = getattr(owner, 'win', owner)
        self.doc, self.ver, self.selected = None, None, None
        self.undo = QUndoStack(self)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(label('The argument in GSN, each node coloured by its evidence as verified now: green holds, '
                          'amber missing or changed (re-bind to accept), red a claim violated, grey undeveloped; '
                          'a goal is as good as its worst support. Falcon is not a qualified tool (ISO 26262-8 §11): '
                          'this is evidence to assess, not a certification argument.', muted=True, wrap=True))
        row = QHBoxLayout()
        self.template = QComboBox()
        for k, t in TEMPLATE_TITLES:
            self.template.addItem(t, k)
        row.addWidget(self.template)
        self.new_btn = button('New from the template', self.new)
        row.addWidget(self.new_btn)
        row.addWidget(button('Verify now', self.verify_now, role='primary'))
        self.rebind_btn = button('Re-bind', self.rebind, tip='pin the selected Solution\'s result as it is now')
        row.addWidget(self.rebind_btn)
        self.undo_btn = button('Undo', self.undo.undo)
        row.addWidget(self.undo_btn)
        row.addWidget(button('Export HTML…', lambda: self.export('html')))
        row.addWidget(button('Export SVG…', lambda: self.export('svg')))
        row.addStretch(1)
        v.addLayout(row)
        self.state = label('', small=True, wrap=True)
        v.addWidget(self.state)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumHeight(320)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        v.addWidget(self.view)
        self.info = KeyValues(('Node', 'Type', 'Text', 'Status', 'Why', 'Evidence', 'Claim', 'Pinned', 'Ran',
                               'Bound'))
        v.addWidget(self.info)
        self.undo.canUndoChanged.connect(self.undo_btn.setEnabled)
        self.undo_btn.setEnabled(False)
        self.reload()

    def workspace(self) -> str:
        try:
            return self.win.workspace() if self.win.project.path else ''
        except AttributeError:
            return ''

    def refresh(self):
        self.reload()

    def showEvent(self, ev):                                      # noqa: N802 (a CI run may have changed it)
        super().showEvent(ev)
        self.reload()

    def reload(self, verify: bool = True):
        ws = self.workspace()
        self.doc, self.ver = None, None
        self.scene.clear()
        if not ws or not os.path.isfile(G.gsn_path(ws)):
            self.state.setText('No safety case in this workspace yet: start one from a template.' if ws else
                               'Save the project first: the safety case lives in its workspace.')
            self.new_btn.setEnabled(bool(ws))
            self.rebind_btn.setEnabled(False)
            return
        try:
            self.doc = G.load(ws)
            self.ver = G.verify(ws, self.doc) if verify else None
        except G.GSNError as e:
            self.state.setText(f'The safety case cannot be read: {e}')
            return
        self.new_btn.setEnabled(False)
        self.draw()
        tops = ', '.join(f'{g} {s}' for g, s in self.ver['top'].items())
        self.state.setText(f'{self.doc.get("title")}: top goals {tops}; ' + ', '.join(
            f'{k} {n}' for k, n in self.ver['counts'].items() if n) + f' — {G.gsn_path(ws)}')
        if self.selected:
            self.select(self.selected)
        else:
            self.rebind_btn.setEnabled(False)

    def draw(self):
        lay = G.layout(self.doc)
        st = self.ver['nodes'] if self.ver else {}
        by = {n['id']: n for n in self.doc['nodes']}
        self.items = {}
        for ln in self.doc['links']:
            if ln['from'] not in lay or ln['to'] not in lay:
                continue
            x0, y0, w0, h0 = lay[ln['from']]
            x1, y1, w1, _ = lay[ln['to']]
            pen = QPen(QColor(THEME.tok['ink2']), 1.2)
            if ln['kind'] == 'InContextOf':
                pen.setStyle(Qt.PenStyle.DashLine)
            a, b = QPointF(x0 + w0 / 2, y0 + h0), QPointF(x1 + w1 / 2, y1)
            self.scene.addLine(a.x(), a.y(), b.x(), b.y(), pen)
            head = QPolygonF([b, QPointF(b.x() - 5, b.y() - 9), QPointF(b.x() + 5, b.y() - 9), b])
            self.scene.addPolygon(head, pen, QBrush(QColor(THEME.tok['ink2'])) if ln['kind'] == 'SupportedBy'
                                  else QBrush(QColor(THEME.tok['panel'])))
        for nid, (x, y, w, h) in lay.items():
            it = NodeItem(self, by[nid], QRectF(x, y, w, h), (st.get(nid) or {}).get('status'))
            self.scene.addItem(it)
            self.items[nid] = it
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-10, -10, 10, 10))

    def select(self, nid: str):
        if self.doc is None:
            return
        try:
            n = G.node(self.doc, nid)
        except G.GSNError:
            self.selected = None
            return
        self.selected = nid
        r = (self.ver or {}).get('nodes', {}).get(nid) or {}
        ev = n.get('evidence') or {}
        self.info.set('Node', nid)
        self.info.set('Type', n['type'])
        self.info.set('Text', n.get('text') or '')
        self.info.set('Status', r.get('status') or '—')
        self.info.set('Why', r.get('why') or '')
        self.info.set('Evidence', f'{ev.get("kind")}: {ev.get("ref")} → {r.get("path") or ev.get("path") or "—"}'
                      if ev else '—')
        self.info.set('Claim', json.dumps(ev.get('claim')) if ev else '—')
        self.info.set('Pinned', ev.get('sha256') or '—', tip='\n'.join(f'{k}: {v}' for k, v in (
            (ev.get('data') or {}).get('sha256') or {}).items()))
        w = ev.get('where')
        self.info.set('Ran', G.where_text(w) if w else '—', tip=', '.join((w or {}).get('gids') or ())[:2000])
        self.info.set('Bound', ev.get('bound_at') or '—')
        self.rebind_btn.setEnabled(n['type'] == 'Solution' and bool(ev.get('ref')))

    # ------------------------------------------------------------------ actions
    def new(self):
        ws = self.workspace()
        if not ws or os.path.isfile(G.gsn_path(ws)):
            return
        doc = G.template(self.template.currentData(), ws)
        G.save(ws, doc)
        G.log_event(ws, 'new', 'gui', template=self.template.currentData(), nodes=len(doc['nodes']),
                    gsn_sha256=G.doc_sha(doc))
        self.reload()

    def verify_now(self):
        ws = self.workspace()
        if not ws or not os.path.isfile(G.gsn_path(ws)):
            return
        try:
            G.verify_and_log(ws, 'gui')
        except G.GSNError as e:
            self.state.setText(str(e))
            return
        self.reload()

    def rebind(self):
        if self.doc is None or not self.selected:
            return
        n = G.node(self.doc, self.selected)
        before = copy.deepcopy(n.get('evidence'))
        if not before:
            return
        doc = copy.deepcopy(self.doc)
        try:
            after = G.bind(self.workspace(), doc, self.selected, before['kind'], before['ref'], keep_claim=True)
        except G.GSNError as e:
            self.state.setText(f'Not re-bound: {e}')
            return
        self.undo.push(RebindCommand(self, self.selected, before, after))

    def export(self, kind: str):
        ws = self.workspace()
        if not ws or self.doc is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export the safety case', os.path.join(
            ws, 'reports', f'safety-case.{kind}'), 'HTML (*.html)' if kind == 'html' else 'SVG (*.svg)')
        if path:
            self.export_to(path)

    def export_to(self, path: str) -> str:
        out = G.export(self.workspace(), path, 'gui')
        self.state.setText(f'Exported {out}')
        return out
