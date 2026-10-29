"""What the command palette (Ctrl+K) searches besides commands: the project's worlds, rigs, scenarios, models and
drive sets, the workspace's runs, evaluations and reports, the settings and the manual's headings.

Each entry is a dict {kind, title, sub, ident, open: callable(win)}. Sources are read when the palette opens: the
project's from memory; runs from the workspace's catalogue (SQLite, newest 1000); evaluations and reports by listing
their folders, cached until the folder changes (its mtime); the manual's headings once per process. Opening an entry
goes to its page and selects it there where the page can (else it just goes to the page).
"""
from __future__ import annotations

import os
import re

from ..core.paths import tilde

KINDS = {          # kind: (group title, icon, priority when scores tie: lower first)
    'command': ('Commands', 'more', 0),
    'run': ('Runs', 'play', 1),
    'scenario': ('Scenarios', 'timeline', 2),
    'world': ('Worlds', 'ground', 3),
    'rig': ('Rigs', 'car-top', 4),
    'model': ('Models', 'cube', 5),
    'evaluation': ('Evaluations', 'checklist', 6),
    'report': ('Reports', 'report', 7),
    'driveset': ('Drive sets', 'dataset', 8),
    'setting': ('Settings', 'gear', 9),
    'doc': ('Manual', 'help', 10),
    'job': ('Job history', 'history', 11),
}

_cache: dict = {}            # (what, folder) -> (mtime, entries)


def _mtime(p):
    try:
        return os.stat(p).st_mtime_ns
    except OSError:
        return None


def _cached(what, folder, build):
    m = _mtime(folder)
    hit = _cache.get((what, folder))
    if hit is not None and hit[0] == m:
        return hit[1]
    out = build() if m is not None else []
    _cache[(what, folder)] = (m, out)
    return out


# ---------------------------------------------------------------------------------------------------- openers
def _select_list(lst, i):
    if lst is not None and 0 <= i < lst.count():
        lst.setCurrentRow(i)


def _select_tree(tree, value, col=0):
    if tree is None:
        return
    for k in range(tree.topLevelItemCount()):
        it = tree.topLevelItem(k)
        if it.data(col, 256) == value:
            tree.setCurrentItem(it)
            return


def _open_world(i):
    def go(win):
        page = win.go('worlds')
        _select_list(getattr(page, 'list', None), i)
    return go


def _open_rig(i):
    def go(win):
        page = win.go('rigs')
        _select_list(getattr(page, 'rigs', None), i)
    return go


def _open_model(mid):
    def go(win):
        page = win.go('models')
        _select_tree(getattr(page, 'models', None), mid)
    return go


def _open_report(path):
    def go(win):
        page = win.go('reports')
        if hasattr(page, 'refresh'):
            page.refresh()
        _select_tree(getattr(page, 'list', None), path)
    return go


def _open_driveset(ident):
    def go(win):
        page = win.go('data')
        for name in ('select_drive_set', 'show_drive_set'):
            if hasattr(page, name):
                getattr(page, name)(ident)
                return
    return go


# ---------------------------------------------------------------------------------------------------- sources
def project_entries(win) -> list:
    from ..core import scenarios as SC
    doc = win.project.doc
    out = []
    for i, w in enumerate(doc.get('worlds') or []):
        out.append({'kind': 'world', 'title': w.get('name', w.get('id', '')),
                    'sub': w.get('town') or w.get('kind', ''), 'ident': w.get('id'), 'open': _open_world(i)})
    for i, r in enumerate(doc.get('rigs') or []):
        out.append({'kind': 'rig', 'title': r.get('name', ''), 'sub': f'{len(r.get("sensors") or [])} sensors',
                    'ident': r.get('id'), 'open': _open_rig(i)})
    scns = [(k, SC.builtin(k), 'built-in') for k in SC.LIBRARY]
    scns += [(s['id'], s, 'project') for s in doc.get('scenarios') or []]
    for ident, s, src in scns:
        out.append({'kind': 'scenario', 'title': s.get('name', ident), 'sub': src, 'ident': ident,
                    'open': (lambda i: lambda w: w.go('scenarios').select(i))(ident)})
    for m in doc.get('models') or []:
        out.append({'kind': 'model', 'title': m.get('name', ''), 'sub': f'{m.get("kind", "")} {m.get("runtime", "")}'
                    .strip(), 'ident': m.get('id'), 'open': _open_model(m.get('id'))})
    for ds in doc.get('drive_sets') or []:
        if isinstance(ds, dict):
            sub = f'query: {ds["query"]}' if ds.get('kind') == 'query' else f'{len(ds.get("runs") or [])} drives'
            out.append({'kind': 'driveset', 'title': ds.get('name', ds.get('id', '')), 'sub': sub,
                        'ident': ds.get('id'), 'open': _open_driveset(ds.get('id'))})
    return out


def run_entries(win, limit=1000) -> list:
    cat = win.service.catalog
    out = []
    for r in (cat.list('run', limit) if cat is not None else []):
        sm = r.get('summary') or {}
        when = (r.get('created') or '')[:16].replace('T', ' ')
        sub = ' · '.join(x for x in (when, r.get('status') or '', sm.get('worker') or '') if x)
        out.append({'kind': 'run', 'title': r['name'], 'sub': sub, 'ident': r['path'],
                    'open': (lambda p: lambda w: w.go('data').show_run(p))(r['path'])})
    return out


def evaluation_entries(win) -> list:
    root = os.path.join(win.workspace(), 'evaluations')

    def build():
        out = []
        for d in sorted(os.listdir(root), reverse=True):
            p = os.path.join(root, d)
            if os.path.isfile(os.path.join(p, 'evaluation.json')):
                out.append({'kind': 'evaluation', 'title': d, 'sub': tilde(p), 'ident': p,
                            'open': (lambda q: lambda w: w.go('models').open_evaluation(q))(p)})
        return out
    return _cached('evaluations', root, build)


def report_entries(win) -> list:
    from ..core import reports as R
    folder = R.reports_dir(win.workspace())

    def build():
        return [{'kind': 'report', 'title': m.get('title') or os.path.basename(m['path']),
                 'sub': f'{m.get("kind", "")} · {(m.get("created") or "")[:16].replace("T", " ")}',
                 'ident': m['path'], 'open': _open_report(m['path'])} for m in R.list_reports(folder)]
    return _cached('reports', folder, build)


def settings_entries(win) -> list:
    page = win.page('settings')
    out = []
    for text, note, _field in getattr(page, 'fields', []) if page is not None else []:
        if text:
            out.append({'kind': 'setting', 'title': text, 'sub': note, 'ident': text,
                        'open': (lambda t: lambda w: w.go('settings').focus_field(t))(text)})
    return out


def doc_entries() -> list:
    from .pages.help import docs_dir, load_docs
    d = docs_dir()

    def build():
        out = []
        for title, fn, md in load_docs():
            out.append({'kind': 'doc', 'title': title, 'sub': fn, 'ident': (fn, None),
                        'open': (lambda f: lambda w: w.go('help').open_doc(f))(fn)})
            fence = False
            for line in md.splitlines():
                if line.startswith('```'):
                    fence = not fence
                m = None if fence else re.match(r'^(#{2,3})\s+(.+?)\s*$', line)
                if m:
                    h = re.sub(r'[`*_]', '', m.group(2))
                    out.append({'kind': 'doc', 'title': h, 'sub': title, 'ident': (fn, h),
                                'open': (lambda f, hh: lambda w: w.go('help').open_doc(f, hh))(fn, h)})
        return out
    return _cached('docs', d, build) if d else []


def job_entries(win, limit=200) -> list:
    hist = getattr(win, 'history', None)
    out = []
    for rec in (hist.read(limit) if hist is not None else []):
        when = (rec.get('finished') or '')[:16].replace('T', ' ')
        out.append({'kind': 'job', 'title': rec.get('title', ''), 'sub': f'{rec.get("status")} · {when} UTC',
                    'ident': rec.get('uid'), 'open': (lambda r: lambda w: w.open_history(r))(rec)})
    return out


def entries(win) -> list:
    """Everything searchable except the commands (the palette adds those from the window's actions)."""
    out = project_entries(win)
    for fn in (run_entries, evaluation_entries, report_entries, settings_entries, job_entries):
        try:
            out += fn(win)
        except OSError:
            pass
    try:
        out += doc_entries()
    except OSError:
        pass
    return out


def rank(query: str, items: list, per_group: int = 8, empty_groups=('command',)) -> list:
    """[(kind, [items best first])] groups, the best-scoring group first. A title matches the query in order (a
    scattered match must score at least 2 a letter: word starts and runs of letters); the subtitle only when it
    contains the whole query, at a third of the score. An empty query lists only `empty_groups`, every item, in
    their order."""
    from .palette import score
    q = query.strip()
    groups: dict = {}
    for it in items:
        if not q:
            if it['kind'] in empty_groups:
                groups.setdefault(it['kind'], []).append((0.0, len(groups.get(it['kind'], [])), it))
            continue
        s = score(q, it['title'])
        if s is not None and s < 100.0 - len(it['title']) and s < 2.0 * len(q):
            s = None                                  # a scattered match of a few letters is noise, not a hit
        if s is None and it.get('sub') and q.lower() in it['sub'].lower():
            s = score(q, it['sub']) / 3.0             # the subtitle counts only for the whole query
        if s is not None:
            groups.setdefault(it['kind'], []).append((-s, len(groups.get(it['kind'], [])), it))
    out = []
    for kind, rows in groups.items():
        rows.sort(key=lambda r: (r[0], r[1]))
        out.append((rows[0][0], KINDS.get(kind, ('', '', 99))[2], kind,
                    [r[2] for r in (rows if not q else rows[:per_group])]))
    out.sort(key=lambda g: (g[0], g[1]))
    return [(kind, rows) for _, _, kind, rows in out]

