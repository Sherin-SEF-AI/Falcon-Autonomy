"""What every tool module shares: resolving the ids the model names to the workspace's own files (never a path it
invents), applying project edits the way the GUI and the CLI do, queuing jobs on the app's queue, and keeping results
compact.

Paths. A run is named by its folder relative to the workspace (runs/<name>, studies/<id>/runs/<name>) or by its folder
name alone when that is unique; other outputs (studies, evaluations, suites, reports …) by their folder name under
their own workspace folder. resolve_* turn them into real paths inside the workspace — a name with '..', an absolute
path or a symbolic link leading out is refused — and every path is checked by privacy.guard before anything is read
from it.

Edits. apply_edit() pushes a core.project.Edit through MainWindow.edit (the undo stack) when the window is there, else
applies it to the Project and saves it (the CLI). One tool call is one Edit wherever possible (an item replaced whole),
so one undo takes a whole tool call back.

Jobs. cli_job() queues `python -m falcon_autonomy <argv>` exactly as the pages do (their JobSpec: kind, memory, the
GPU lock), so the job runs the CLI's code path; jobs submitted by the service (drives) go through the service.
"""
from __future__ import annotations

import json
import os
import shlex
import sys

from ..privacy import guard
from ..tool_api import ToolError

LIST_CAP = 25              # rows a list returns at most (the rest counted, with how to narrow)
TEXT_CAP = 600             # characters of a free-text field
LOG_LINES = 12             # lines of a job's log tail


# ------------------------------------------------------------------------------------------------ the project
def project(ctx):
    p = ctx.project if ctx.win is None else ctx.win.project
    if p is None:
        raise ToolError('no project is open')
    return p


def doc(ctx) -> dict:
    return project(ctx).doc


def workspace(ctx) -> str:
    """The project's workspace (as MainWindow.workspace() says it, worked out here without calling into the window:
    read tools run off its thread)."""
    if ctx.win is not None:
        p = ctx.win.project
        if p.path:
            ws = p.workspace
        else:
            from ...core.paths import cache_dir
            ws = os.path.join(cache_dir(), 'untitled.falcon')
    else:
        ws = ctx.workspace or getattr(ctx.project, 'workspace', '')
    if not ws:
        raise ToolError('the project has no workspace yet (save it first)')
    return os.path.realpath(ws)


def find(ctx, section: str, ident: str, what: str = '') -> tuple:
    """(index, item) of a project item by id (or, when unique, by its name); ToolError naming the ids otherwise."""
    items = doc(ctx).get(section) or []
    for i, it in enumerate(items):
        if it.get('id') == ident:
            return i, it
    named = [(i, it) for i, it in enumerate(items) if str(it.get('name', '')).lower() == str(ident).lower()]
    if len(named) == 1:
        return named[0]
    ids = ', '.join(str(it.get('id')) for it in items[:30]) or 'none'
    raise ToolError(f'no {what or section.rstrip("s")} {ident!r} in the project (ids: {ids})')


def apply_edit(ctx, edit, merge_key: str = '') -> dict:
    """Apply one Edit: through the window's undo stack, or to the Project (saved, when it has a file) from the CLI.
    Returns {'edit': its text, 'undo': how to take it back}."""
    p = project(ctx)
    if ctx.win is not None:
        ctx.win.edit(edit, merge_key)
        how = 'Edit ▸ Undo (Ctrl+Z) in the window'
    else:
        p.apply(edit)
        if p.path:
            p.save()
        how = 'not undoable from the command line once saved: make the opposite edit'
    ctx.emit('project_edit', {'text': edit.text, 'path': list(edit.path)})
    return {'edit': edit.text, 'undo': how}


def _short(v, n=160) -> str:
    s = json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
    return s if len(s) <= n else s[:n - 1] + '…'


def edit_preview(ctx, edit) -> str:
    """An Edit as a short diff against the project as it is now."""
    from ...core.project import get_at
    d = doc(ctx)
    where = '/'.join(map(str, edit.path))
    if edit.kind == 'set':
        try:
            old = get_at(d, edit.path)
        except (KeyError, IndexError, TypeError):
            old = '(absent)'
        if isinstance(old, dict) and isinstance(edit.value, dict):
            keys = sorted(set(old) | set(edit.value))
            ch = [f'  {k}: {_short(old.get(k, "(absent)"), 70)} → {_short(edit.value.get(k, "(absent)"), 70)}'
                  for k in keys if old.get(k, '(absent)') != edit.value.get(k, '(absent)')]
            return f'{edit.text}\n{where}:\n' + ('\n'.join(ch[:20]) or '  (no change)')
        return f'{edit.text}\n{where}: {_short(old)} → {_short(edit.value)}'
    if edit.kind == 'insert':
        return f'{edit.text}\n+ {where}[{"end" if edit.index is None else edit.index}]: {_short(edit.value, 400)}'
    try:
        old = get_at(d, edit.path)[edit.index]
    except (KeyError, IndexError, TypeError):
        old = '?'
    return f'{edit.text}\n- {where}[{edit.index}]: {_short(old, 400)}'


# ------------------------------------------------------------------------------------------------ paths
def inside(path: str, root: str) -> bool:
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def checked(ctx, path: str) -> str:
    """A path inside the workspace that is not private (ToolError otherwise); the real path."""
    ws = workspace(ctx)
    p = os.path.realpath(path)
    if not inside(p, ws):
        raise ToolError('refused: that is outside the project\'s workspace (the assistant reads only Falcon\'s own '
                        'outputs there)')
    guard(p)
    return p


def _bad_name(name: str) -> bool:
    return (not isinstance(name, str) or not name or name.startswith(('/', '~')) or '\\' in name
            or any(part in ('..', '.') for part in name.split('/')) or '\x00' in name)


def resolve_run(ctx, run: str) -> str:
    """A run id (runs/<name>, studies/<id>/runs/<name>, or a unique folder name) → its folder, checked."""
    if _bad_name(run):
        raise ToolError(f'{run!r} is not a run id: use the ids runs_search returns (e.g. runs/<folder>)')
    ws = workspace(ctx)
    run = run[4:] if run.startswith('run:') else run
    cands = []
    if '/' in run:
        cands.append(os.path.join(ws, run))
    else:
        from ...core import runindex as RI
        cands = [d for d in RI.run_dirs(ws) if os.path.basename(d) == run]
        if len(cands) > 1:
            raise ToolError(f'{run!r} names {len(cands)} runs: use the id with its folder (' +
                            ', '.join(os.path.relpath(d, ws) for d in cands[:5]) + ')')
    if not cands or not os.path.isfile(os.path.join(cands[0], 'run.json')):
        raise ToolError(f'no run {run!r} in the workspace (runs_search lists them)')
    return checked(ctx, cands[0])


def run_id(ctx, path: str) -> str:
    return os.path.relpath(os.path.realpath(path), workspace(ctx))


def resolve_in(ctx, folder: str, name: str, what: str, must=('',)) -> str:
    """<workspace>/<folder>/<name>, checked (and containing one of must when given)."""
    if _bad_name(name) or '/' in name:
        raise ToolError(f'{name!r} is not a {what} id')
    p = os.path.join(workspace(ctx), folder, name)
    if not os.path.exists(p) or (must != ('',) and not any(os.path.exists(os.path.join(p, m)) for m in must)):
        raise ToolError(f'no {what} {name!r} in the workspace ({folder}/)')
    return checked(ctx, p)


def out_dir(ctx, folder: str, name: str) -> str:
    """A new output folder <workspace>/<folder>/<name> for a job (refused when it exists)."""
    if _bad_name(name) or '/' in name:
        raise ToolError(f'{name!r} is not a usable folder name (letters, digits, - and _)')
    p = os.path.join(workspace(ctx), folder, name)
    if os.path.exists(p):
        raise ToolError(f'{folder}/{name} exists already: choose another name')
    return p


def read_json(path: str, default=None):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def read_jsonl(path: str, limit: int | None = None) -> list:
    out = []
    try:
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
                if limit is not None and len(out) >= limit:
                    break
    except OSError:
        pass
    return out


# ------------------------------------------------------------------------------------------------ compact results
def capped(rows: list, cap: int = LIST_CAP, hint: str = '') -> dict:
    """{'items': the first cap rows, 'total': n, 'more': n - cap (when cut), 'hint': how to narrow}."""
    rows = list(rows)
    out = {'items': rows[:cap], 'total': len(rows)}
    if len(rows) > cap:
        out['more'] = len(rows) - cap
        if hint:
            out['hint'] = hint
    return out


def clip(text, n: int = TEXT_CAP) -> str:
    s = '' if text is None else str(text)
    return s if len(s) <= n else s[:n - 1] + '…'


def rounded(v, nd: int = 4):
    """Floats rounded (recursively) so results stay short; everything else unchanged."""
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, dict):
        return {k: rounded(x, nd) for k, x in v.items()}
    if isinstance(v, list):
        return [rounded(x, nd) for x in v]
    return v


def no_media(v):
    """Drop what must never reach the model from a result: image, audio and point-cloud arrays, and base64 blobs (a
    long string without spaces). Paths and statistics stay."""
    if isinstance(v, dict):
        return {k: no_media(x) for k, x in v.items()
                if k not in ('image', 'images', 'pixels', 'png', 'jpeg', 'jpg', 'thumbnail', 'audio', 'samples',
                             'points', 'pointcloud', 'b64', 'base64')}
    if isinstance(v, list):
        if len(v) > 64 and all(isinstance(x, (int, float)) for x in v[:64]):
            return f'[{len(v)} numbers left out]'
        return [no_media(x) for x in v]
    if isinstance(v, str) and len(v) > 2000 and ' ' not in v[:2000]:
        return '[binary-looking text left out]'
    return v


def on_gui(ctx, fn, timeout: float = 10.0):
    """fn() on the window's thread (read tools run off it, and widgets may only be touched there; act tools already
    run on it): ui.guicall.call. Without a window, fn() here."""
    if ctx.win is None:
        return fn()
    from ...ui.guicall import call
    try:
        return call(fn, timeout)
    except TimeoutError as e:
        raise ToolError(str(e)) from e


# ------------------------------------------------------------------------------------------------ jobs
def need_queue(ctx):
    q = ctx.queue if ctx.win is None else ctx.win.queue
    if q is None:
        raise ToolError('no job queue here')
    return q


def need_service(ctx):
    s = ctx.service if ctx.win is None else ctx.win.service
    if s is None:
        raise ToolError('no CARLA service here')
    return s


def cli_argv(argv: list) -> list:
    return [sys.executable, '-m', 'falcon_autonomy'] + [str(a) for a in argv]


def command_line(argv: list) -> str:
    """The command a job runs, as the user would type it (falcon …)."""
    return 'falcon ' + ' '.join(shlex.quote(str(a)) for a in argv)


def cli_job(ctx, kind: str, title: str, argv: list, memory_gb: float = 1.0, needs_gpu: bool = False,
            needs_carla: bool = False, env: dict | None = None, run_dir: str = '', resumable: bool = False,
            spec=None) -> dict:
    """Queue `falcon <argv>` (or a ready JobSpec from a core job_spec helper) and say what was queued."""
    from ...core.jobs import JobSpec
    if spec is None:
        spec = JobSpec(kind, title, cli_argv(argv), env=env, needs_gpu=needs_gpu, memory_gb=memory_gb,
                       run_dir=run_dir, resumable=resumable)
    job = need_queue(ctx).submit(spec)
    ctx.emit('job_queued', {'job': job.id, 'title': job.spec.title})
    return {'job': job.id, 'title': job.spec.title, 'state': job.state, 'command': command_line(argv),
            'follow': 'jobs_wait or jobs_get with this job id'}


def job_brief(ctx, job, log: int = 0) -> dict:
    from ...core import protocol as P
    from ...core.jobs import FINAL
    d = {'job': job.id, 'kind': job.spec.kind, 'title': job.spec.title, 'state': job.state,
         'progress': round(float(job.progress), 3)}
    if job.text:
        d['step'] = clip(job.text, 200)
    if job.wait_reason:
        d['waiting_for'] = clip(job.wait_reason, 200)
    if job.eta_s is not None and job.state not in FINAL:
        d['eta_s'] = round(float(job.eta_s), 1)
    if job.started:
        d['elapsed_s'] = round(job.elapsed, 1)
    if job.state in FINAL:
        d['exit'] = P.describe_exit(job.exit_code) if job.exit_code is not None else ''
        errs = [t for lvl, t in job.log if lvl == 'error' and t]
        if job.state == 'failed' and errs:
            d['error'] = clip(errs[-1], 300)
    rd = job.spec.run_dir
    if rd:
        try:
            ws = workspace(ctx)
            if inside(rd, ws):
                d['run'] = os.path.relpath(os.path.realpath(rd), ws)
        except ToolError:
            pass
    if job.metrics:
        d['metrics'] = {k: rounded(m.get('value')) for k, m in list(job.metrics.items())[:12]}
    if log:
        d['log_tail'] = [f'{lvl}: {clip(t, 240)}' for lvl, t in job.log[-log:]]
    return d


# ------------------------------------------------------------------------------------------------ schema helpers
def obj(props: dict | None = None, required=(), **kw) -> dict:
    d = {'type': 'object', 'properties': props or {}, 'additionalProperties': False}
    if required:
        d['required'] = list(required)
    d.update(kw)
    return d


def s(desc: str, **kw) -> dict:
    return dict({'type': 'string', 'description': desc}, **kw)


def n(desc: str, **kw) -> dict:
    return dict({'type': 'number', 'description': desc}, **kw)


def i(desc: str, **kw) -> dict:
    return dict({'type': 'integer', 'description': desc}, **kw)


def b(desc: str, **kw) -> dict:
    return dict({'type': 'boolean', 'description': desc}, **kw)


def arr(items: dict, desc: str, **kw) -> dict:
    return dict({'type': 'array', 'items': items, 'description': desc}, **kw)
