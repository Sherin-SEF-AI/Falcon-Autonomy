"""Window tools (gui): what page is open, go to a page or tab, show a run (at a frame) in Data ▸ Browse, compare runs,
open a job's result, and "show me" — a pulsing outline on one control (ui.highlight) so the user does it themselves.
They move the view only; nothing in the project or the workspace changes. Each needs Falcon's window. Act tools run
on the window's thread; the read ones (ui_current, ui_controls) run off it and look at the widgets through
_common.on_gui."""
from __future__ import annotations

import os

from ..tool_api import ToolError, tool
from . import _common as C

PAGE_KEYS = ('home', 'worlds', 'rigs', 'scenarios', 'live', 'data', 'replay', 'loop', 'rarecases', 'training',
             'models', 'stack', 'hardware', 'reports', 'settings', 'help')


def _tabs(page) -> list:
    tw = getattr(page, 'tabs', None)
    return [tw.tabText(i) for i in range(tw.count())] if tw is not None else []


@tool('ui_current', 'ui', 'read',
      'What the window shows: the open page (key and title) and its tab, every page key with its tabs, the run open '
      'in Data ▸ Browse (its id) and the frame shown, and whether the project has unsaved changes.', gui=True)
def ui_current(ctx):
    return C.on_gui(ctx, lambda: _current(ctx))


def _current(ctx):
    w = ctx.win
    key = w.current_key()
    page = w.page(key)
    out = {'page': key, 'title': getattr(page, 'title', ''),
           'tab': page.tabs.tabText(page.tabs.currentIndex()) if getattr(page, 'tabs', None) is not None else None,
           'pages': {p.key: _tabs(p) for p in w.pages}, 'unsaved_changes': bool(w.project.dirty)}
    data = w.page('data')
    rd = getattr(getattr(data, 'browse', None), 'run_dir', '') if data is not None else ''
    if rd:
        try:
            out['browse_run'] = C.run_id(ctx, C.checked(ctx, rd))
            out['browse_frame'] = data.browse.slider.value()
        except ToolError:
            out['browse_run'] = '(not shown: outside the workspace or private)'
    return out


@tool('ui_open_page', 'ui', 'act',
      'Go to a page (and optionally one of its tabs by its label, e.g. data + Browse, models + Failures, training + '
      'World model). ui_current lists the pages and tabs.',
      C.obj({'page': C.s('page key', enum=list(PAGE_KEYS)), 'tab': C.s('a tab label of that page')}, ['page']),
      preview=lambda ctx, **a: f'go to {a["page"]}' + (f' ▸ {a["tab"]}' if a.get('tab') else ''), gui=True)
def ui_open_page(ctx, page, tab=''):
    p = ctx.win.go(page)
    if p is None:
        raise ToolError(f'no page {page!r} in this build')
    if tab:
        tw = getattr(p, 'tabs', None)
        names = _tabs(p)
        k = next((i for i, n in enumerate(names) if n.lower() == tab.lower()), None)
        if k is None:
            raise ToolError(f'{page} has no tab {tab!r} (tabs: {", ".join(names) or "none"})')
        tw.setCurrentIndex(k)
    return {'page': page, 'tab': tab or None}


@tool('ui_show_run', 'ui', 'act',
      'Open a run in Data ▸ Browse, optionally at a simulation step or a time in seconds from its start, so the user '
      'sees its frames, labels and objects. The assistant itself never sees the images.',
      C.obj({'run': C.s('run id'), 'step': C.i('simulation step', minimum=0), 't_s': C.n('seconds from the start',
                                                                                        minimum=0)}, ['run']),
      preview=lambda ctx, **a: f'show {a["run"]} in Data ▸ Browse', gui=True)
def ui_show_run(ctx, run, step=None, t_s=None):
    d = C.resolve_run(ctx, run)
    data = ctx.win.page('data')
    ctx.win.go('data')
    data.show_run(d)
    if step is not None:
        data.browse.seek_step(int(step))
    elif t_s is not None:
        data.browse.seek_time(float(t_s))
    return {'shown': C.run_id(ctx, d), 'frame_index': data.browse.slider.value()}


@tool('ui_show_frame', 'ui', 'act',
      'Show one frame of the run open in Data ▸ Browse (or of the given run) by its frame index (runs_frames lists '
      'them).', C.obj({'index': C.i('frame index', minimum=0), 'run': C.s('run id (default the one shown)')},
                      ['index']),
      preview=lambda ctx, **a: f'show frame {a["index"]}', gui=True)
def ui_show_frame(ctx, index, run=''):
    data = ctx.win.page('data')
    if run:
        ui_show_run(ctx, run)
    elif not data.browse.run_dir:
        raise ToolError('no run is open in Data ▸ Browse: give run')
    else:
        C.checked(ctx, data.browse.run_dir)
        ctx.win.go('data')
        data.tabs.setCurrentWidget(data.browse)
    n = data.browse.slider.maximum()
    if index > n:
        raise ToolError(f'the run has frames 0…{n}')
    data.browse.slider.setValue(int(index))
    return {'frame_index': int(index)}


@tool('ui_compare_runs', 'ui', 'act',
      'Open Data ▸ Compare with 2–4 runs on one clock (their numbers: runs_compare).',
      C.obj({'runs': C.arr(C.s('run id'), 'run ids', minItems=2, maxItems=4)}, ['runs']),
      preview=lambda ctx, **a: f'compare {", ".join(a["runs"])} in Data ▸ Compare', gui=True)
def ui_compare_runs(ctx, runs):
    dirs = [C.resolve_run(ctx, r) for r in runs]
    ctx.win.go('data')
    ctx.win.page('data').compare_runs(dirs)
    return {'compared': [C.run_id(ctx, d) for d in dirs]}


@tool('ui_open_job', 'ui', 'act',
      'Go to where a job\'s result is shown (a drive in Data ▸ Browse, anything else on the page it was started from) '
      '— the Jobs panel\'s Open.', C.obj({'job': C.i('job id')}, ['job']),
      preview=lambda ctx, job: f'open job {job}\'s result', gui=True)
def ui_open_job(ctx, job):
    j = ctx.win.queue.get(int(job))
    if j is None:
        raise ToolError(f'no job {job} in this session')
    if j.spec.run_dir and os.path.isdir(j.spec.run_dir):
        C.checked(ctx, j.spec.run_dir)
    p = ctx.win.open_job(j.id)
    return {'page': getattr(p, 'key', None) or ctx.win.current_key()}


@tool('ui_controls', 'ui', 'read',
      'The named controls of a page, for ui_highlight: attribute names (e.g. record.world, record.go_btn) and visible '
      'labels (button:Record, check:…, tab:Browse) with each control\'s kind and tooltip.',
      C.obj({'page': C.s('page key', enum=list(PAGE_KEYS)), 'filter': C.s('only names containing this')}, ['page']),
      gui=True)
def ui_controls(ctx, page, filter=''):
    from ...ui import highlight as H

    def rows():
        p = ctx.win.page(page)
        if p is None:
            raise ToolError(f'no page {page!r}')
        return [{'name': k, 'kind': H.describe(k, w)} for k, w in H.controls(p).items()
                if filter.lower() in k.lower()]
    return C.capped(C.on_gui(ctx, rows), 60, 'use filter')


@tool('ui_highlight', 'ui', 'act',
      '"Show me": bring a control into view (its page, tab and scroll position) and draw a pulsing outline around it '
      'for a few seconds, so the user can do the step themselves. Nothing is clicked or changed.',
      C.obj({'page': C.s('page key', enum=list(PAGE_KEYS)), 'control': C.s('a name from ui_controls'),
             'seconds': C.n('how long (default 4)', minimum=1, maximum=15)}, ['page', 'control']),
      preview=lambda ctx, **a: f'outline {a["control"]} on {a["page"]}', gui=True)
def ui_highlight(ctx, page, control, seconds=4.0):
    from ...ui import highlight as H
    o = H.highlight(ctx.win, page, control, float(seconds))
    if o is None:
        raise ToolError(f'no control {control!r} on {page} (ui_controls lists them)')
    return {'highlighted': control, 'page': page, 'seconds': float(seconds)}
