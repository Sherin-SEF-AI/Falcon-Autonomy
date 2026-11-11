"""Cleaning up after the cluster: the caches Falcon keeps here and on its nodes, collected (standard library;
docs/cluster.md, "Cleaning up"). `falcon cluster gc`, the assistant's cluster_cache_gc.

What may go, by where it is:

    this machine   the workspace's content store (core.cas, core.lineage's mark and sweep): an object no dataset
                   version's splits file, run index (the workspace's runs, a study's runs, a dataset's drives, a run in
                   the trash) or training's data.json names, linked into the store more than max(1 h, the age) ago.
                   Only files under <workspace>/cas are touched: an object is the store's own name for bytes (a hard
                   link to a run's file, a reflink or a copy), so removing one never changes a run.
    a daemon node  (ssh, local) each slot's scratch workspace (cluster.daemon's Daemon.gc): runs/<gid>.<attempt>
                   folders of jobs the daemon no longer runs — left by an earlier life of the daemon, or finished and
                   never collected — and its CARLA logs, unchanged for the age. Never a job running there or
                   finished within the hour (it waits to be collected), never one whose global id a scheduler of this
                   process still has (running, or lost and re-queueable).
    a Slurm node   falcon-<id> array folders under its workdir (slurm.Backend.gc), unchanged for the age, of no array
                   in Slurm's queue (nothing at all when squeue does not answer) and none this process runs.
    Kubernetes     nothing: a run's Job and ConfigMap are deleted when its run comes back (ttlSecondsAfterFinished
                   for one Falcon could not delete).
    the cloud      nothing persistent: a burst's machine is terminated at its end and keeps nothing. Its ledger
                   (cloud/ledger.jsonl) and its records (cloud/bursts/) are never removed; a burst still running or
                   not confirmed terminated is listed, never touched.

Never, anywhere: a run folder, the trash, a project, anything private (dashcam-derived: a node never has any, and a
scratch folder whose records say so is kept), a link, a name not of Falcon's own pattern, anything referenced.

plan() is a dry run: every item with its size (and for the store what removing it frees: an object hard-linked from a
run frees nothing until the run goes too). collect() removes; given `only` (the item ids of an approved plan) it
removes just those that are still removable — what became referenced or busy since stays, what appeared since is not
touched. save_plan()/load_plan() keep a plan for an hour (the assistant's tool removes only an approved dry run).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from . import nodes as N

DEFAULT_H = 24.0                     # the age, in hours, below which nothing is collected (--older-than)
PLAN_TTL_S = 3600.0
_PLAN_ID = 16


class GCError(ValueError):
    pass


def _now() -> float:
    """The time the store's ages are measured against (a test moves it on)."""
    return time.time()


class _Clock:
    """What a Slot or a batch backend asks of a scheduler when it is used on its own."""

    def __init__(self):
        self.lock = threading.RLock()
        self.jobs: dict = {}

    @staticmethod
    def clock():
        return time.monotonic()


def human(n: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024 or unit == 'TB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024.0
    return f'{n:.1f} TB'


def _busy() -> tuple:
    """(global ids of the jobs this process's schedulers still have, Slurm array folders they run)."""
    from . import launcher as LA
    gids, dirs = set(), set()
    with LA._LOCK:
        scheds = list(LA._SCHEDULERS.values())
    for sch in scheds:
        with sch.lock:
            for rid in sch.jobs:
                gids.add(rid.split('.', 1)[0])
        for b in sch.batch:
            for bt in list(getattr(b, 'batches', {}).values()):
                dirs.add(bt['dir'])
    return gids, dirs


# ================================================================================================ this machine
def _store(workspace: str, older_than_s: float, dry_run: bool, only) -> dict:
    from ..core import cas as C
    from ..core import lineage as L
    sec = {'where': 'this machine', 'kind': 'store', 'items': [], 'kept': 0, 'kept_why': {}, 'notes': [],
           'error': '', 'removed': 0, 'bytes_freed': 0}
    if not workspace:
        sec['notes'].append('no workspace given: its content store was not looked at')
        return sec
    root = C.store_dir(workspace)
    if not os.path.isdir(os.path.join(root, 'objects')):
        sec['notes'].append('the workspace has no content store (cas/)')
        return sec
    grace = max(C.DEFAULT_GRACE_S, float(older_than_s))
    now = _now()
    try:
        store = C.Store(root)
        roots = set(L.roots(workspace))             # a root that cannot be read stops everything (nothing removed)
        refs = set(roots)
        if only is not None:                         # an approved plan: every other object counts as referenced
            refs |= {f'{store.alg}:{hx}' for hx in store.objects() if f'cas/{hx}' not in only}
        res = store.gc(refs, dry_run=True, grace_s=grace, now=now)
    except (L.LineageError, C.CASError, OSError, ValueError) as e:
        sec['error'] = f'the content store was not collected: {e}'
        return sec
    for hx in res['removed']:
        try:
            st = os.stat(os.path.join(root, 'objects', hx[:2], hx))
        except OSError:
            continue
        sec['items'].append({'id': f'cas/{hx}', 'name': f'cas/objects/{hx[:2]}/{hx[:12]}…', 'what': 'store object',
                             'bytes': st.st_size, 'frees': st.st_size if st.st_nlink <= 1 else 0,
                             'age_h': round((now - st.st_ctime) / 3600.0, 2)})
    sec['kept'] = res['kept'] + res['young']
    sec['kept_why'] = {k: v for k, v in (('referenced', res['kept']),
                                          (f'linked in the last {grace / 3600.0:g} h', res['young'])) if v}
    if res['tmp_removed']:
        sec['notes'].append(f'{res["tmp_removed"]} unfinished put(s) in cas/tmp older than {grace / 3600.0:g} h go '
                            'too')
    sec['notes'].append(f'{len(roots)} addresses referenced (dataset versions, run indexes, trainings)')
    if not dry_run and sec['items']:              # the roots again, now; only what was just listed can go
        listed = {it['id'][4:] for it in sec['items']}
        try:
            refs = set(L.roots(workspace)) | {f'{store.alg}:{hx}' for hx in store.objects() if hx not in listed}
        except (L.LineageError, C.CASError, OSError, ValueError) as e:
            sec['error'] = f'the content store was not collected: {e}'
            return sec
        done = store.gc(refs, dry_run=False, grace_s=grace, now=now)
        gone = set(done['removed'])
        sec['removed'] = len(gone)
        sec['bytes_freed'] = sum(it['frees'] for it in sec['items'] if it['id'][4:] in gone)
    return sec


# ================================================================================================ nodes
def _daemon_node(node: dict, older_than_s: float, dry_run: bool, only, keep: set) -> list:
    from .scheduler import Slot
    out = []
    clock = _Clock()
    for slot, port in N.slots(node):
        s = Slot(clock, node, slot, port)
        sec = {'where': f'node {s.key}', 'kind': node['kind'], 'items': [], 'kept': 0, 'kept_why': {}, 'notes': [],
               'error': '', 'removed': 0, 'bytes_freed': 0}
        names = None if only is None else [i.split(':', 1)[1] for i in only if i.startswith(f'{s.key}:')]
        if names == []:
            out.append(sec)
            continue
        try:
            r = s.ask({'t': 'gc', 'older_than_s': float(older_than_s), 'dry_run': bool(dry_run),
                       'keep': sorted(keep), 'only': names}, 'gc', timeout=300)
        except (OSError, ValueError) as e:
            sec['error'] = f'not reached: {e}'
            out.append(sec)
            continue
        finally:
            if s.conn is not None:
                s.conn.close()
        if r.get('error'):
            sec['error'] = str(r['error'])
        for it in r.get('items') or ():
            sec['items'].append(dict(it, id=f'{s.key}:{it["name"]}', frees=it.get('bytes', 0)))
        _kept(sec, r)
        sec['removed'], sec['bytes_freed'] = int(r.get('removed') or 0), int(r.get('bytes_freed') or 0)
        if r.get('items_total'):
            sec['notes'].append(f'{r["items_total"]} items in all (the first 2000 listed)')
        out.append(sec)
    return out


def _kept(sec: dict, r: dict):
    kept = r.get('kept') or []
    sec['kept'] = int(r.get('kept_total') or len(kept))
    for k in kept:
        why = str(k.get('why') or '')
        why = 'changed recently' if why.startswith('changed ') else why
        sec['kept_why'][why] = sec['kept_why'].get(why, 0) + 1


def _slurm_node(node: dict, older_than_s: float, dry_run: bool, only, busy: set) -> dict:
    from .slurm import Backend
    b = Backend(_Clock(), node)
    sec = {'where': f'node {b.key}', 'kind': 'slurm', 'items': [], 'kept': 0, 'kept_why': {}, 'notes': [],
           'error': '', 'removed': 0, 'bytes_freed': 0}
    names = None if only is None else {i.split(':', 1)[1] for i in only if i.startswith(f'{b.key}:')}
    try:
        if names == set():
            return sec
        r = b.gc(older_than_s, dry_run=dry_run, only=names, busy=busy)
    except OSError as e:
        sec['error'] = f'not reached: {e}'
        return sec
    finally:
        if b.conn is not None:
            b.conn.close()
    for it in r['items']:
        sec['items'].append(dict(it, id=f'{b.key}:{it["name"]}', frees=it['bytes']))
    _kept(sec, r)
    sec['removed'], sec['bytes_freed'] = r['removed'], r['bytes_freed']
    return sec


def _cloud() -> dict:
    from . import cloud as CL
    sec = {'where': 'the cloud', 'kind': 'cloud', 'items': [], 'kept': 0, 'kept_why': {}, 'error': '', 'removed': 0,
           'bytes_freed': 0, 'notes': ['nothing persistent: a burst\'s machine is terminated at its end and keeps '
                                       'nothing; the ledger and the bursts\' records are never removed']}
    try:
        active = [s for s in CL.states() if s.get('state') not in CL.FINAL]
    except (OSError, ValueError):
        active = []
    if active:
        sec['kept'] = len(active)
        sec['kept_why'] = {'a burst still running or not confirmed terminated': len(active)}
    return sec


# ================================================================================================ the whole
def collect(settings=None, workspace: str = '', node: str = '', older_than_h: float = DEFAULT_H,
            dry_run: bool = True, only=None, keep=()) -> dict:
    """Collect (or, dry_run, list) the caches (module doc). node: one node's only (its name), else this machine's
    store (when a workspace is given), every node and the cloud. only: item ids of an approved plan. keep: global
    ids of jobs to spare besides those this process's schedulers have. GCError for a bad node name or age."""
    if settings is None:
        from ..core.settings import Settings
        settings = Settings()
    h = float(older_than_h)
    if not 0.0 <= h <= 24 * 3650:
        raise GCError('--older-than is hours, 0 or more')
    cfg = N.load(settings)
    nodes = cfg['nodes']
    if node:
        nodes = [n for n in nodes if n['name'] == node]
        if not nodes:
            raise GCError(f'no node {node!r} (falcon cluster nodes list)')
    only = None if only is None else set(only)
    gids, dirs = _busy()
    gids |= {str(g) for g in keep or ()}
    secs = []
    if not node:
        secs.append(_store(workspace, h * 3600.0, dry_run, only))
    for n in nodes:
        if n['kind'] in ('ssh', 'local'):
            secs += _daemon_node(n, h * 3600.0, dry_run, only, gids)
        elif n['kind'] == 'slurm':
            secs.append(_slurm_node(n, h * 3600.0, dry_run, only, dirs))
        else:
            secs.append({'where': f'node {n["name"]}/k8s', 'kind': n['kind'], 'items': [], 'kept': 0,
                         'kept_why': {}, 'error': '', 'removed': 0, 'bytes_freed': 0,
                         'notes': ['nothing kept between runs: each run\'s Job and ConfigMap are deleted when it '
                                   'comes back (ttlSecondsAfterFinished for one that could not be)']})
    if not node:
        secs.append(_cloud())
    for sec in secs:
        sec['count'] = len(sec['items'])
        sec['bytes'] = sum(int(it.get('bytes') or 0) for it in sec['items'])
        sec['frees'] = sum(int(it.get('frees') or 0) for it in sec['items'])
    return {'older_than_h': h, 'dry_run': bool(dry_run), 'workspace': workspace, 'node': node, 'sections': secs,
            'count': sum(s['count'] for s in secs), 'bytes': sum(s['bytes'] for s in secs),
            'frees': sum(s['frees'] for s in secs), 'removed': sum(s['removed'] for s in secs),
            'bytes_freed': sum(s['bytes_freed'] for s in secs), 'errors': [f'{s["where"]}: {s["error"]}'
                                                                           for s in secs if s['error']]}


def plan(settings=None, workspace: str = '', node: str = '', older_than_h: float = DEFAULT_H, keep=()) -> dict:
    """A dry run (collect with dry_run)."""
    return collect(settings, workspace, node, older_than_h, dry_run=True, keep=keep)


# ================================================================================================ plans
def _plans_dir() -> str:
    from ..core.paths import config_dir
    return os.path.join(config_dir(), 'cluster', 'gc-plans')


def save_plan(res: dict) -> str:
    """Keep a dry run's items for an hour; its id (16 hex digits) names exactly what it listed."""
    from ..core.paths import atomic_write
    ids = sorted(it['id'] for s in res['sections'] for it in s['items'])
    args = {k: res[k] for k in ('workspace', 'node', 'older_than_h')}
    pid = hashlib.sha256(json.dumps([args, ids], sort_keys=True).encode()).hexdigest()[:_PLAN_ID]
    d = _plans_dir()
    os.makedirs(d, exist_ok=True)
    now = time.time()
    for f in os.listdir(d):                          # stale plans go
        p = os.path.join(d, f)
        try:
            if f.endswith('.json') and now - os.path.getmtime(p) > PLAN_TTL_S:
                os.unlink(p)
        except OSError:
            pass
    atomic_write(os.path.join(d, f'{pid}.json'), json.dumps({'id': pid, 'created': now, 'args': args,
                                                              'items': ids, 'bytes': res['bytes']}))
    return pid


def load_plan(pid: str) -> dict:
    """A saved plan, GCError when it is unknown or more than an hour old."""
    if not (len(pid) == _PLAN_ID and all(c in '0123456789abcdef' for c in pid)):
        raise GCError('a plan id is 16 hex digits (from a dry run)')
    p = os.path.join(_plans_dir(), f'{pid}.json')
    try:
        with open(p, encoding='utf-8') as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        raise GCError(f'no dry run {pid}: run a dry run first') from None
    if time.time() - float(doc.get('created') or 0) > PLAN_TTL_S:
        raise GCError(f'the dry run {pid} is more than an hour old: run a new dry run')
    return doc


# ================================================================================================ text
def lines(res: dict, limit: int = 8) -> list:
    """The result as the command prints it."""
    out = []
    verb = 'would remove' if res['dry_run'] else 'removed'
    for s in res['sections']:
        n = s['count'] if res['dry_run'] else s['removed']
        size = s['bytes'] if res['dry_run'] else s['bytes_freed']
        head = f'{s["where"]}: '
        if s['error']:
            head += f'FAILED — {s["error"]}'
        elif n:
            head += f'{verb} {n} item{"s" * (n != 1)}, {human(size)}'
            if s['kind'] == 'store' and s['frees'] != s['bytes']:
                head += f' (frees {human(s["frees"])}: the rest are hard links of files still in runs)'
        else:
            head += 'nothing to remove'
        if s['kept']:
            head += '; kept ' + ', '.join(f'{v} {k}' for k, v in s['kept_why'].items())
        out.append(head)
        for it in sorted(s['items'], key=lambda x: -int(x.get('bytes') or 0))[:limit]:
            out.append(f'  {human(it.get("bytes") or 0):>9}  {it["name"]}  ({it["what"]}, '
                       f'{it.get("age_h", 0):.1f} h old)')
        if len(s['items']) > limit:
            out.append(f'  … and {len(s["items"]) - limit} more')
        out += [f'  note: {x}' for x in s['notes']]
    total = res['bytes'] if res['dry_run'] else res['bytes_freed']
    out.append(f'{"would remove" if res["dry_run"] else "removed"} {res["count"] if res["dry_run"] else res["removed"]}'
               f' item(s), {human(total)}' + (' — a dry run: nothing was removed' if res['dry_run'] else ''))
    return out
