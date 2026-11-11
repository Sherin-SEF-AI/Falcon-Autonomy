"""The tools an MCP client gets, and running one on a session — the open Falcon window (WindowSession, driven by the
window's bridge, ui.mcp_bridge) or a project file (session.FileSession). Ported from Vantage mcp/service.py (GPL).

The tools are the assistant's own registry (assistant.tool_api.REGISTRY, docs/assistant-tools.md), with the same
validation, privacy guard and redaction (tool_api.call) the in-app assistant has:
- read tools run as they are; act tools change the project as one undoable step named "Claude · <tool>" (in the window
  a single entry of Edit ▸ Undo however many edits the tool makes; in a file session a step of its own undo stack,
  saved with a backup of the file as it was), queue jobs (on the window's queue, or run in this process), or move the
  window's view;
- never listed and refused when called: trash tools (moving runs to the trash asks the user in Falcon's window every
  time) and every name on the never-list (tool_api.NEVER: deleting, settings, servers, shell, publishing, downloads,
  images); a tool whose own gate asks the user on a card in every mode (Tool.approval — the research autopilot's
  compute steps) is refused too, with where to approve it; a window tool without a window says it needs one;
- the MCP server's own: falcon_status (where calls go, the steps made, jobs), undo_last_change and redo_last_change
  (only the client's own steps: a step of the user's is left alone), open_project (a project file, when no window is
  open).
Results are the tool's JSON, whole, as text and as structuredContent.
"""
from __future__ import annotations

import json
import os

from ..assistant import privacy
from ..assistant import tool_api as T
from .session import PREFIX

PROTOCOL_NOTE = ('Falcon Autonomy develops self-driving AI on the CARLA simulator: worlds, rigs, scenarios, recorded '
                 'runs, datasets, models, evaluations, rare-case search, training, stack testing, reports. These tools '
                 'are the ones Falcon\'s own assistant uses, on the same code paths as its buttons. Start with '
                 'project_overview and runs_search; read before you change anything. Cite only numbers a tool '
                 'returned, with their run, job or evaluation id. An act tool makes one undoable step '
                 '("Claude · <tool>"): in Falcon\'s window the user sees it at once and can undo it; on a project file '
                 'each step is saved, with a backup of the file as it was first. Jobs run in the window\'s queue '
                 '(jobs_* tools; jobs_wait or jobs_get read their outcome). Never possible here: moving runs to the '
                 'trash, deleting anything, changing Falcon\'s settings, starting or stopping a server outside the '
                 'job queue, shell commands, downloads, and anything from the user\'s dashcam footage.')

OWN = ('falcon_status', 'undo_last_change', 'redo_last_change', 'open_project')


class NeedsWindow(Exception):
    pass


# =========================================================================== results
def text(doc, error: bool = False) -> dict:
    """An MCP tool result: the JSON as text (whole) and, for an object, as structuredContent."""
    body = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False, default=str, separators=(',', ':'))
    out = {'content': [{'type': 'text', 'text': body}], 'isError': bool(error)}
    if isinstance(doc, dict):
        out['structuredContent'] = json.loads(json.dumps(doc, default=str))
    return out


def error(why: str) -> dict:
    return text({'error': why}, True)


# =========================================================================== the listing
def _registry():
    from ..assistant import tools  # noqa: F401 (registers Falcon's tools)
    return T.REGISTRY


def _full(t) -> str:
    try:
        from ..assistant.tools._compact import DETAILS
    except ImportError:                       # pragma: no cover
        DETAILS = {}
    return (DETAILS.get(t.name) or {}).get('description') or t.description or ''


def offered(t) -> bool:
    return t.tier in ('read', 'act') and t.name not in T.NEVER


def refusal(name: str) -> str | None:
    """Why a tool is never run over MCP (None: it may run)."""
    if name in T.NEVER:
        return f'{name} is never allowed (docs/assistant.md, "What the assistant never does")'
    t = _registry().get(name)
    if t is None:
        return f'no tool named {name!r} (tools/list lists them)'
    if t.tier == 'trash':
        return (f'{name} moves runs into the workspace trash: that is done only from Falcon\'s window, where the user '
                'is asked every time')
    if not offered(t):
        return f'{name} is not offered here'
    return None


OWN_TOOLS = [
    {'name': 'falcon_status', 'title': 'Where calls go',
     'description': 'Where the tool calls go now — Falcon\'s window (its project, the page open) or a project file — '
                    'the steps this client made there, unsaved changes, and the jobs queued from here.',
     'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
     'annotations': {'readOnlyHint': True, 'destructiveHint': False, 'idempotentHint': True, 'openWorldHint': False}},
    {'name': 'undo_last_change', 'title': 'Undo the last change',
     'description': 'Undo this client\'s last step ("Claude · <tool>"): in the window only while it is the newest '
                    'step on the undo stack (a step of the user\'s is left as it is); on a file, saved.',
     'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
     'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': False,
                     'openWorldHint': False}},
    {'name': 'redo_last_change', 'title': 'Redo the last undone change',
     'description': 'Redo this client\'s last undone step (in the window: only when the next step to redo is one).',
     'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
     'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': False,
                     'openWorldHint': False}},
    {'name': 'open_project', 'title': 'Open a project file',
     'description': 'Work on a project file (*.falcon.json) when no Falcon window is open (a file open in a window '
                    'is worked on there). Without a path: back to the open window.',
     'inputSchema': {'type': 'object', 'properties': {'path': {'type': 'string',
                                                               'description': 'the project file'}},
                     'additionalProperties': False},
     'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': True, 'openWorldHint': False}},
]


def listing() -> list:
    """tools/list: every read and act tool (the full description; the schema Falcon validates against), annotated,
    then the server's own."""
    out = []
    for t in _registry().values():
        if not offered(t):
            continue
        desc = _full(t)
        if t.tier == 'act':
            desc += ' [Changes something: one undoable step.]'
        if t.gui:
            desc += ' [Needs Falcon\'s window open.]'
        out.append({'name': t.name, 'title': t.name.replace('_', ' ').capitalize(), 'description': desc,
                    'inputSchema': t.params,
                    'annotations': {'title': t.name.replace('_', ' ').capitalize(),
                                    'readOnlyHint': t.tier == 'read', 'destructiveHint': False,
                                    'idempotentHint': t.tier == 'read', 'openWorldHint': False}})
    return out + [dict(x) for x in OWN_TOOLS]


def gate_refusal(t, ctx, args) -> str | None:
    """A tool whose own gate (Tool.approval) asks on a card in every mode cannot be answered from here."""
    fn = getattr(t, 'approval', None)
    if fn is None:
        return None
    try:
        g = fn(ctx, **(args or {}))
    except Exception:                         # noqa: BLE001 (when in doubt, the user decides — in the window)
        g = {'ask': True}
    if isinstance(g, dict) and g.get('ask'):
        why = g.get('why') or 'it asks the user every time'
        return (f'{t.name} is approved by the user on a card in Falcon\'s Assistant panel ({why}); an MCP client '
                'cannot answer that card. Ask the user to run it from the panel.')
    return None


def call(session, name: str, args) -> dict:
    """One of Falcon's tools on a session (FileSession or WindowSession) as an MCP result."""
    args = args if isinstance(args, dict) else {}
    why = refusal(name)
    if why:
        return error(why)
    t = _registry()[name]
    if t.gui and getattr(session, 'mode', '') != 'window':
        return error(f'{name} needs Falcon\'s window: open Falcon (with the project) and call it again')
    try:
        res = session.call_tool(name, args, t.tier)
    except NeedsWindow as e:
        return error(str(e))
    except RuntimeError as e:
        return error(str(e))
    if not isinstance(res, dict):
        res = {'result': res}
    return text(res, 'error' in res)


def dispatch(session, name: str, args) -> dict:
    """A call on a session (window or file): the server's own tools there, else one of Falcon's (call)."""
    try:
        if name == 'falcon_status':
            return text(session.status())
        if name in ('undo_last_change', 'redo_last_change'):
            out = session.redo() if name.startswith('redo') else session.undo()
            return text({'result': out}, out.startswith(('nothing', 'the last step is', 'the next step is')))
    except RuntimeError as e:
        return error(str(e))
    return call(session, name, args)


# =========================================================================== the window
class WindowSession:
    """The window's side (Qt-free): Falcon's tools on the window's project, called from the bridge's worker threads.
    ui(fn) runs fn on the GUI thread and returns its result. Read tools run off the GUI thread, as the in-app
    assistant runs them; act and window tools on it (the undo stack, the job queue's processes, the view)."""
    mode = 'window'

    def __init__(self, win, ui, log=None):
        self.win = win
        self.ui = ui
        self.log = log or (lambda text: None)

    def _ctx(self):
        w = self.win
        return T.ToolContext(project=w.project, settings=w.settings, queue=w.queue, service=w.service, win=w,
                             workspace=w.workspace(), redact=privacy.redact, approval={'how': 'mcp'})

    def call_tool(self, name: str, args: dict, tier: str) -> dict:
        t = _registry()[name]
        ctx = self.ui(self._ctx)
        if tier == 'read' and not t.gui:
            g = gate_refusal(t, ctx, args)
            return {'error': g} if g else T.call(ctx, name, args)
        return self.ui(lambda: self._act(t, ctx, name, args))

    def _act(self, t, ctx, name, args) -> dict:
        """On the GUI thread: the tool's edits as one undo step "Claude · <tool>" (a macro opened at its first edit,
        so a call that edits nothing leaves no empty step)."""
        g = gate_refusal(t, ctx, args)
        if g:
            return {'error': g}
        w = self.win
        u = w.undo
        opened = [False]
        own = w.edit                                   # MainWindow.edit: every project change goes through it

        def edit(e, merge_key=''):
            if not opened[0]:
                u.beginMacro(f'{PREFIX} {name}')
                opened[0] = True
            own(e, '')                                 # never merged into an earlier step of the user's
        w.edit = edit
        try:
            res = T.call(ctx, name, args)
        finally:
            if w.__dict__.get('edit') is edit:
                del w.edit
            if opened[0]:
                u.endMacro()
        self.log(f'MCP: {name}' + (' (one undo step)' if opened[0] else ''))
        if opened[0] and isinstance(res, dict):
            res = dict(res, undo='Edit ▸ Undo in Falcon\'s window, or undo_last_change')
        return res

    def _step(self, redo: bool) -> str:
        u = self.win.undo
        i = u.index()
        j = i if redo else i - 1
        if not 0 <= j < u.count():
            return 'nothing to ' + ('redo' if redo else 'undo')
        name = u.text(j)
        if not name.startswith(PREFIX):
            return f'the {"next" if redo else "last"} step is the user\'s ({name}): left as it is'
        (u.redo if redo else u.undo)()
        return f'{"redid" if redo else "undid"}: {name}'

    def undo(self) -> str:
        return self.ui(lambda: self._step(False))

    def redo(self) -> str:
        return self.ui(lambda: self._step(True))

    def status(self) -> dict:
        def read():
            w = self.win
            u = w.undo
            steps = [{'text': u.text(j), 'state': 'done' if j < u.index() else 'undone'}
                     for j in range(max(0, u.count() - 20), u.count()) if u.text(j).startswith(PREFIX)]
            page = w.current_key() if hasattr(w, 'current_key') else ''
            return {'where': f'Falcon\'s window (pid {os.getpid()})', 'mode': 'window',
                    'project': w.project.doc.get('name', ''), 'path': w.project.path,
                    'workspace': w.workspace(), 'page': page, 'unsaved_changes': bool(w.project.dirty),
                    'steps': steps, 'jobs_active': len(w.queue.active())}
        return self.ui(read)

    def revision(self):
        return self.ui(lambda: (self.win.project.path, int(self.win.undo.index())))
