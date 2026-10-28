"""Where Falcon keeps things, and how paths are shown.

* settings and secrets: $XDG_CONFIG_HOME/FalconAutonomy (tests point XDG_CONFIG_HOME at a throw-away folder);
* a project: <name>.falcon.json, with its workspace (runs, datasets, models) in <name>.falcon/ beside it;
* the catalogue: <workspace>/catalog.sqlite.

On screen every path is shown from ~ (tilde()).
"""
from __future__ import annotations

import os

from .. import APP_ID

PROJECT_SUFFIX = '.falcon.json'
WORKSPACE_SUFFIX = '.falcon'


def config_dir() -> str:
    base = os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('~'), '.config')
    return os.path.join(base, APP_ID)


def cache_dir() -> str:
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, APP_ID)


def tilde(path) -> str:
    """path shown from the home folder: /home/jo/x -> ~/x (other paths unchanged)."""
    if not path:
        return ''
    p = os.path.abspath(os.fspath(path))
    home = os.path.expanduser('~')
    if p == home:
        return '~'
    if p.startswith(home + os.sep):
        return '~' + p[len(home):]
    return p


def workspace_for(project_path: str) -> str:
    """The workspace folder beside a project file: a/b.falcon.json -> a/b.falcon/."""
    p = os.path.abspath(project_path)
    base = p[:-len(PROJECT_SUFFIX)] if p.endswith(PROJECT_SUFFIX) else os.path.splitext(p)[0]
    return base + WORKSPACE_SUFFIX


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write(path: str, data: bytes | str):
    """Write path through a temporary file in the same folder and rename (never a half-written file)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'.{os.path.basename(path)}.{os.getpid()}.tmp')
    mode = 'wb' if isinstance(data, bytes) else 'w'
    kw = {} if isinstance(data, bytes) else {'encoding': 'utf-8'}
    with open(tmp, mode, **kw) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
