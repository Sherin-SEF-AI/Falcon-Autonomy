"""Reports ▸ Safety case (GSN): the workspace's living safety case (core.gsn; docs/gsn.md) drawn in Goal Structuring
Notation, each node marked by its status as verified now, in words and colour — met (blue: its evidence holds), evidence
changed (amber: missing or changed evidence, re-bind to accept it), not met (orange: a claim violated, a bound broken),
undeveloped (grey) — with the status propagated up as AND. Verify now draws the argument in level by level; the
selected node wears a slow amber ring. Clicking a node shows its text, status and why, and a Solution's evidence: the result it is
bound to, the claim, the pinned SHA-256 and when it was bound. Re-bind pins the result's current hash (undoable here:
Undo puts the previous evidence back); Verify now re-verifies and logs it (safety/gsn-events.jsonl); a new safety case
starts from the ISO 26262 or ISO 21448 SOTIF template; Export writes the self-contained HTML report (core.reports) or
an SVG.

4.2: an evidence node names its result's fidelity (core.fidelity) — evidence from the fake simulator, the kinematic
screen or mixed results verifies amber at best. attach() adds a result as a new Solution (the other sections' "Attach
evidence", undoable here like a re-bind); offer_run() is Send to ▸ Reports: the run waits in a bar, preselected, and is
attached only on the user's click."""
from __future__ import annotations

import copy
import json
import os

from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGraphicsItem, QGraphicsPathItem, QGraphicsScene,
                             QGraphicsTextItem, QGraphicsView, QHBoxLayout, QVBoxLayout, QWidget)

from ..core import gsn as G
from . import motion
from .theme import THEME, mono
from .widgets import IdLabel, KeyValues, Note, button, fidelity_chip, label, short_id

TONE = {'green': 'good', 'amber': 'warning', 'red': 'critical'}
WORD = {'green': 'met', 'amber': 'evidence changed', 'red': 'not met', 'grey': 'undeveloped'}


def colour(status: str | None) -> QColor:
    """A status's mark colour on this theme (grey and none: the muted ink)."""
    return QColor(THEME.status(TONE[status]) if status in TONE else THEME.tok['muted'])
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
    """A node: its shape on the surface colour, outlined in its status colour; its id and status word in mono (the
    word, not the colour alone, carries the status), then its text."""

    def __init__(self, tab, n: dict, rect: QRectF, status: str | None, fidelity: str = '', source: str = ''):
        super().__init__(_shape(n['type'], rect))
        self.tab, self.node_id = tab, n['id']
        col = colour(status)
        self.setBrush(QBrush(QColor(THEME.tok['raised'])))
        self.setPen(QPen(col, 1.6 if status in TONE else 1.0))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        word = WORD.get(status, 'no status') if status else 'no status'
        self.setToolTip(f'{n["id"]} ({n["type"]}): {word}')
        txt = QGraphicsTextItem(self)
        txt.setDefaultTextColor(QColor(THEME.tok['ink']))
        f = QFont('IBM Plex Sans')
        f.setPixelSize(11)
        txt.setFont(f)
        pad = 18 if n['type'] in ('Solution', 'Assumption', 'Justification', 'Strategy') else 10
        txt.setTextWidth(rect.width() - 2 * pad)
        mf = mono(10).family()
        head = (f'<span style="font-family:\'{mf}\'; font-weight:600">{n["id"]}</span>'
                + (f' <span style="font-family:\'{mf}\'; color:{col.name()}">{word.upper()}</span>' if status else ''))
        if fidelity and fidelity != 'unknown':                  # the evidence's fidelity, as a chip's word (4.2)
            from ..core import fidelity as F
            t = F.tone(source) if fidelity == 'recorded' and source else F.tone(fidelity)
            fc = THEME.status(t) if t in TONE.values() else THEME.tok['accent' if t == 'accent' else 'muted']
            what = F.label(fidelity) + (f' · of {F.label(source)} runs' if source and source != 'unknown' else '')
            head += f'<br><span style="font-family:\'{mf}\'; font-size:9px; color:{fc}">◆ {what.upper()}</span>'
            self.setToolTip(f'{n["id"]} ({n["type"]}): {word} — {what}: {F.sentence(fidelity)}')
        body = (n.get('text') or '')[:150]
        txt.setHtml(f'{head}<br>{body.replace("<", "&lt;")}')
        txt.setPos(rect.x() + pad, rect.y() + 4)
        if n['type'] in ('Assumption', 'Justification'):
            tag = QGraphicsTextItem(n['type'][0], self)
            tag.setDefaultTextColor(QColor(THEME.tok['muted']))
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


class AttachCommand(QUndoCommand):
    """Attach a result as a new Solution (core.gsn.attach); undo removes the node and its link (core.gsn.detach)."""

    def __init__(self, tab, kind: str, ref: str, claim: dict | None, text: str, parent: str = ''):
        super().__init__(f'Attach {os.path.basename(ref.rstrip("/"))}')
        self.tab, self.kind, self.ref, self.claim, self.text, self.parent = tab, kind, ref, claim, text, parent
        self.nid, self.error = '', ''

    def redo(self):
        ws = self.tab.workspace()
        doc = G.load(ws)
        try:
            self.nid = G.attach(ws, doc, self.kind, self.ref, copy.deepcopy(self.claim), self.text, self.parent)
        except G.GSNError as e:
            self.error = str(e)
            return
        G.save(ws, doc)
        G.log_event(ws, 'attach', 'gui', node=self.nid, evidence=G.node(doc, self.nid).get('evidence'),
                    gsn_sha256=G.doc_sha(doc))
        self.tab.selected = self.nid
        self.tab.reload()

    def undo(self):
        if not self.nid:
            return
        ws = self.tab.workspace()
        doc = G.load(ws)
        G.detach(doc, self.nid)
        G.save(ws, doc)
        G.log_event(ws, 'detach', 'gui', node=self.nid, gsn_sha256=G.doc_sha(doc))
        if self.tab.selected == self.nid:
            self.tab.selected = None
        self.tab.reload()


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
        v.addWidget(Note('The argument in GSN, each node marked by its evidence as verified now: met (it holds), '
                         'evidence changed (missing or changed: re-bind to accept; or evidence from the fake '
                         'simulator, the kinematic screen or mixed results), not met (a claim violated), undeveloped; '
                         'a goal is as good as its worst support. Falcon is not a qualified tool (ISO 26262-8 §11): '
                         'this is evidence to assess, not a certification argument.'))
        self.offer = QFrame()                                     # Send to ▸ Reports: a run waiting to be attached
        self.offer.setProperty('card', True)
        oh = QHBoxLayout(self.offer)
        oh.setContentsMargins(10, 6, 6, 6)
        oh.addWidget(label('Attach run', muted=True))
        self.offer_id = IdLabel('', keep=10)
        oh.addWidget(self.offer_id)
        self.offer_chip = fidelity_chip('unknown')
        oh.addWidget(self.offer_chip)
        oh.addWidget(label('as evidence under', muted=True))
        self.offer_parent = QComboBox()
        oh.addWidget(self.offer_parent)
        oh.addStretch(1)
        self.offer_btn = button('Attach', self.attach_offered, tip='Add the run as a new Solution (undoable)')
        oh.addWidget(self.offer_btn)
        oh.addWidget(button('Dismiss', lambda: self.offer_run('')))
        self.offer.setVisible(False)
        self.offered = ''
        v.addWidget(self.offer)
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
        self.info = KeyValues(('Node', 'Type', 'Text', 'Status', 'Why', 'Evidence', 'Fidelity', 'Claim', 'Pinned',
                               'Ran', 'Bound'))
        v.addWidget(self.info)
        self.undo.canUndoChanged.connect(self.undo_btn.setEnabled)
        self.undo_btn.setEnabled(False)
        self.ring = None                                          # the selected node's pulsing outline
        self.ring_anim = QVariantAnimation(self)
        self.ring_anim.setStartValue(0.0)
        self.ring_anim.setEndValue(1.0)
        self.ring_anim.setDuration(1800)
        self.ring_anim.setLoopCount(-1)
        self.ring_anim.valueChanged.connect(self._ring_tick)
        self.reveal = motion.Driver(self.view.viewport(), 900, QEasingCurve.Type.Linear)
        self.reveal.anim.valueChanged.connect(self._reveal_tick)
        THEME.changed.connect(self._themed)          # a bound slot: the drawing follows the theme
        self.reload()

    def _themed(self, _name):
        if self.doc is not None:
            self.draw()
            if self.selected:
                self.select(self.selected)

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
        tops = ', '.join(f'{g} {WORD.get(s, s)}' for g, s in self.ver['top'].items())
        self.state.setText(f'{self.doc.get("title")}: top goals {tops}; ' + ', '.join(
            f'{n} {WORD.get(k, k)}' for k, n in self.ver['counts'].items() if n) + f' — {G.gsn_path(ws)}')
        if self.selected:
            self.select(self.selected)
        else:
            self.rebind_btn.setEnabled(False)

    def draw(self):
        lay = G.layout(self.doc)
        st = self.ver['nodes'] if self.ver else {}
        by = {n['id']: n for n in self.doc['nodes']}
        self.items = {}
        self.ring = None
        self.ring_anim.stop()
        for ln in self.doc['links']:
            if ln['from'] not in lay or ln['to'] not in lay:
                continue
            x0, y0, w0, h0 = lay[ln['from']]
            x1, y1, w1, _ = lay[ln['to']]
            pen = QPen(QColor(THEME.tok['strong']), 1.2)
            if ln['kind'] == 'InContextOf':
                pen.setStyle(Qt.PenStyle.DashLine)
            a, b = QPointF(x0 + w0 / 2, y0 + h0), QPointF(x1 + w1 / 2, y1)
            self.scene.addLine(a.x(), a.y(), b.x(), b.y(), pen)
            head = QPolygonF([b, QPointF(b.x() - 5, b.y() - 9), QPointF(b.x() + 5, b.y() - 9), b])
            self.scene.addPolygon(head, pen, QBrush(QColor(THEME.tok['strong'])) if ln['kind'] == 'SupportedBy'
                                  else QBrush(QColor(THEME.tok['base'])))
        for nid, (x, y, w, h) in lay.items():
            r = st.get(nid) or {}
            ev = bool(by[nid].get('evidence'))
            it = NodeItem(self, by[nid], QRectF(x, y, w, h), r.get('status'), r.get('fidelity', '') if ev else '',
                          r.get('source_fidelity', '') if ev else '')
            self.scene.addItem(it)
            self.items[nid] = it
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-10, -10, 10, 10))
        self.view.setBackgroundBrush(QBrush(QColor(THEME.tok['base'])))

    def _reveal_tick(self, v):
        """Level by level from the top goal down: each row of nodes (and the lines into it) fades in in turn."""
        rows = sorted({round(it.pos().y() + it.boundingRect().top()) for it in self.scene.items()
                       if isinstance(it, NodeItem)})
        n = max(len(rows), 1)
        for it in self.scene.items():
            if it.parentItem() is not None:
                continue
            y = it.boundingRect().top() + it.pos().y()
            k = sum(1 for r in rows if r <= y + 1) - 1
            k = max(k, 0)
            it.setOpacity(min(max((float(v) * (n + 1) - k), 0.0), 1.0))

    def _ring_tick(self, v):
        if self.ring is not None:
            c = QColor(THEME.tok['accent'])
            c.setAlphaF(0.2 + 0.6 * (1 - abs(float(v) - 0.5) * 2))
            self.ring.setBrush(QBrush(c))

    def select(self, nid: str):
        if self.doc is None:
            return
        try:
            n = G.node(self.doc, nid)
        except G.GSNError:
            self.selected = None
            return
        self.selected = nid
        it = getattr(self, 'items', {}).get(nid)
        if it is not None:
            if self.ring is None or self.ring.scene() is not self.scene:
                self.ring = QGraphicsPathItem()             # a band around the node, under it: only its outer
                self.ring.setPen(QPen(Qt.PenStyle.NoPen))   # half shows
                self.ring.setZValue(-1)
                self.scene.addItem(self.ring)
            from PyQt6.QtGui import QPainterPathStroker
            st = QPainterPathStroker()
            st.setWidth(8)
            self.ring.setPath(st.createStroke(it.path()))
            self.ring.setBrush(QBrush(QColor(THEME.tok['accent'])))
            if motion.enabled():
                self.ring_anim.start()
        r = (self.ver or {}).get('nodes', {}).get(nid) or {}
        ev = n.get('evidence') or {}
        self.info.set('Node', nid)
        self.info.set('Type', n['type'])
        self.info.set('Text', n.get('text') or '')
        self.info.set('Status', WORD.get(r.get('status'), r.get('status')) if r.get('status') else '—')
        self.info.set('Why', r.get('why') or '')
        self.info.set('Evidence', f'{ev.get("kind")}: {ev.get("ref")} → {r.get("path") or ev.get("path") or "—"}'
                      if ev else '—')
        if ev and r.get('fidelity'):
            from ..core import fidelity as F
            src = r.get('source_fidelity')
            self.info.set('Fidelity', F.label(r['fidelity']) + (f' (of {F.label(src)} runs)' if src else ''),
                          tip=F.sentence(r['fidelity']))
        else:
            self.info.set('Fidelity', '—')
        self.info.set('Claim', json.dumps(ev.get('claim')) if ev else '—')
        self.info.set('Pinned', short_id(ev.get('sha256') or '—', 12), tip='\n'.join(
            [ev.get('sha256') or ''] + [f'{k}: {v}' for k, v in ((ev.get('data') or {}).get('sha256') or {}).items()]))
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
        if motion.enabled() and self.isVisible():                 # the verified argument draws in from the top goal
            self._reveal_tick(0.0)
            self.reveal.start()

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

    # ------------------------------------------------------------------ attach (4.2)
    def ensure_case(self) -> bool:
        """A safety case to attach to: the workspace's, or a new blank one (said in the state line)."""
        ws = self.workspace()
        if not ws:
            self.state.setText('Save the project first: the safety case lives in its workspace.')
            return False
        if not os.path.isfile(G.gsn_path(ws)):
            doc = G.template('blank', ws)
            G.save(ws, doc)
            G.log_event(ws, 'new', 'gui', template='blank', nodes=len(doc['nodes']), gsn_sha256=G.doc_sha(doc))
            self.reload()
        return True

    def attach(self, kind: str, ref: str, claim: dict | None = None, text: str = '', parent: str = '') -> str:
        """Add a result as a new Solution under `parent` (default: the selected Goal or Strategy, else the first top
        goal) through the undo stack; returns its id ('' when refused — the reason is in the state line)."""
        if not self.ensure_case():
            return ''
        if not parent and self.doc is not None and self.selected:
            try:
                if G.node(self.doc, self.selected)['type'] in G.SUPPORTERS:
                    parent = self.selected
            except G.GSNError:
                pass
        cmd = AttachCommand(self, kind, ref, claim, text, parent)
        self.undo.push(cmd)
        if cmd.error:
            self.undo.undo()
            self.state.setText(f'Not attached: {cmd.error}')
            return ''
        self.state.setText(f'Attached {ref} as {cmd.nid} (Undo removes it). ' + self.state.text())
        it = getattr(self, 'items', {}).get(cmd.nid)
        if it is not None:
            self.view.ensureVisible(it)
        return cmd.nid

    def offer_run(self, run_path: str):
        """Send to ▸ Reports: show the run, preselected, ready to attach on a click ('' hides the bar)."""
        from ..core import fidelity as F
        self.offered = run_path or ''
        self.offer.setVisible(bool(self.offered))
        if not self.offered:
            return
        self.offer_id.setText(self.offered)
        lv = F.of_run(self.offered)
        self.offer_chip.setText(F.label(lv))
        self.offer_chip.setToolTip(F.sentence(lv))
        self.offer_chip.setProperty('tone', F.tone(lv))
        self.offer_chip.style().unpolish(self.offer_chip)
        self.offer_chip.style().polish(self.offer_chip)
        self.offer_parent.clear()
        goals = [n for n in (self.doc or {}).get('nodes') or [] if n['type'] in G.SUPPORTERS]
        for n in goals:
            self.offer_parent.addItem(f'{n["id"]} · {(n.get("text") or "")[:40]}', n['id'])
        if not goals:
            self.offer_parent.addItem('G1 · a new blank safety case', '')
        self.offer_btn.setFocus()

    def attach_offered(self) -> str:
        if not self.offered:
            return ''
        ws = self.workspace()
        run = self.offered
        ref = os.path.relpath(run, ws) if ws and os.path.abspath(run).startswith(os.path.abspath(ws) + os.sep) \
            else os.path.abspath(run)
        claim = None
        try:
            with open(os.path.join(run, 'run.json'), encoding='utf-8') as fh:
                if (json.load(fh).get('scenario') or {}).get('passed') is not None:
                    claim = {'measure': 'field', 'path': 'scenario.passed', 'equals': True}
        except (OSError, ValueError):
            pass
        nid = self.attach('run', ref, claim, f'Run {os.path.basename(run.rstrip("/"))} passes its scenario'
                          if claim else f'Run {os.path.basename(run.rstrip("/"))}', self.offer_parent.currentData())
        if nid:
            self.offer_run('')
        return nid

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
