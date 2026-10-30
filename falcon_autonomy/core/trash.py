"""The workspace's trash: recorded runs moved aside, never deleted, and put back on request (standard library only).

Falcon never deletes a run outright (assistant.tool_api.NEVER). Trashing a run moves its folder, unchanged, into

    <workspace>/trash/<UTC stamp>-<run folder>/<run folder>/
    <workspace>/trash/<UTC stamp>-<run folder>/manifest.json

The manifest says where it came from (its path relative to the workspace), when it was trashed, its catalogue row and
links and the project's tags for it at that moment, so restore() can put back exactly what was there. The catalogue
row is removed on trash and re-read from the run's own run.json on restore (with its links); the run index's cache
(<workspace>/runindex.json) is refreshed on both. The project's tags and drive sets refer to a run by its path, which
restore gives back, so they are left alone. Emptying the trash is the user's own act in their file manager.

Only a workspace's own recorded runs (<workspace>/runs/<name>) are trashed: a study's runs belong to its results
(studies/<id>/runs) and stay with the study.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil

from .paths import atomic_write, ensure_dir
from .provenance import utc_now

SCHEMA = 'falcon.trash/1'
MANIFEST = 'manifest.json'


class TrashError(ValueError):
    pass


def trash_dir(workspace: str) -> str:
    return os.path.join(workspace, 'trash')


def _inside(path: str, root: str) -> bool:
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def _catalog(workspace: str):
    from .catalog import Catalog
    p = os.path.join(workspace, 'catalog.sqlite')
    return Catalog(p) if os.path.isfile(p) else None


def _refresh_index(workspace: str):
    from . import runindex as RI
    try:
        RI.build(workspace)
    except OSError:
        pass


def check_run(workspace: str, run_dir: str) -> str:
    """The run's folder, checked: a folder with a run.json directly in <workspace>/runs (else TrashError)."""
    runs = os.path.realpath(os.path.join(workspace, 'runs'))
    d = os.path.realpath(run_dir)
    if os.path.dirname(d) != runs:
        raise TrashError(f'{os.path.basename(d)} is not one of the workspace\'s recorded runs (runs/<name>); a '
                         'study\'s runs stay with the study')
    if not os.path.isfile(os.path.join(d, 'run.json')):
        raise TrashError(f'{os.path.basename(d)} has no run.json: not a run')
    return d


def trash(workspace: str, run_dirs, project_doc: dict | None = None, catalog=None) -> list:
    """Move each run folder into the trash with its manifest. Returns the manifests (each with 'item', the trash
    folder's name, to restore() it). Checks every run before moving any. catalog: an open Catalog to update (the
    GUI's); else the workspace's catalogue file is opened, updated and closed."""
    ws = os.path.realpath(workspace)
    dirs = [check_run(ws, d) for d in run_dirs]
    if len(set(dirs)) != len(dirs):
        raise TrashError('a run is named twice')
    tags = (project_doc or {}).get('run_tags') or {}
    cat = catalog if catalog is not None else _catalog(ws)
    out = []
    try:
        for d in dirs:
            name = os.path.basename(d)
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            item = f'{stamp}-{name}'
            k = 1
            while os.path.exists(os.path.join(trash_dir(ws), item)):
                k += 1
                item = f'{stamp}-{name}-{k}'
            dest = ensure_dir(os.path.join(trash_dir(ws), item))
            rel = os.path.relpath(d, ws)
            ident = 'run:' + name
            row = cat.get(ident) if cat is not None else None
            man = {'schema': SCHEMA, 'item': item, 'name': name, 'original': rel, 'trashed': utc_now(),
                   'catalog': row, 'links': cat.links(ident) if cat is not None else [],
                   'tags': list(tags.get(rel) or ())}
            atomic_write(os.path.join(dest, MANIFEST), json.dumps(man, indent=1, default=str) + '\n')
            shutil.move(d, os.path.join(dest, name))
            if cat is not None:
                cat.remove(ident)
            out.append(man)
    finally:
        if cat is not None and catalog is None:
            cat.close()
    _refresh_index(ws)
    return out


def read_manifest(workspace: str, item: str) -> dict:
    if not item or os.sep in item or item.startswith('.'):
        raise TrashError(f'{item!r} is not a trash item\'s name')
    p = os.path.join(trash_dir(workspace), item, MANIFEST)
    try:
        with open(p, encoding='utf-8') as fh:
            man = json.load(fh)
    except (OSError, ValueError) as e:
        raise TrashError(f'no trash item {item!r}') from e
    if man.get('schema') != SCHEMA:
        raise TrashError(f'{item!r} has no Falcon trash manifest')
    return man


def restore(workspace: str, item: str, catalog=None) -> str:
    """Put a trashed run back where it was; returns its folder. Refused when something is there now."""
    ws = os.path.realpath(workspace)
    man = read_manifest(ws, item)
    src = os.path.join(trash_dir(ws), item, man['name'])
    dest = os.path.join(ws, man['original'])
    if not _inside(dest, os.path.join(ws, 'runs')):
        raise TrashError(f'the manifest of {item!r} points outside the workspace\'s runs')
    if not os.path.isdir(src):
        raise TrashError(f'the run folder of {item!r} is missing from the trash')
    if os.path.exists(dest):
        raise TrashError(f'{man["original"]} exists again: rename or trash it first')
    ensure_dir(os.path.dirname(dest))
    shutil.move(src, dest)
    shutil.rmtree(os.path.join(trash_dir(ws), item))         # only the manifest is left in it
    cat = catalog if catalog is not None else _catalog(ws)
    if cat is not None:
        try:
            cat.add_run(dest)
            for ln in man.get('links') or ():
                cat.link(ln['src'], ln['dst'], ln['role'])
        finally:
            if catalog is None:
                cat.close()
    _refresh_index(ws)
    return dest


def list_items(workspace: str) -> list:
    """The trash's items, newest first: their manifests (with 'size_bytes' of the run folder)."""
    root = trash_dir(workspace)
    out = []
    if not os.path.isdir(root):
        return out
    for item in sorted(os.listdir(root), reverse=True):
        try:
            man = read_manifest(workspace, item)
        except TrashError:
            continue
        size = 0
        for dp, _dn, fn in os.walk(os.path.join(root, item, man['name'])):
            for f in fn:
                try:
                    size += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        out.append(dict(man, size_bytes=size))
    return out
