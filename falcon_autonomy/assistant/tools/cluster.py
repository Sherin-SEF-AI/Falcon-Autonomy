"""Cluster tools (the cluster group): the compute nodes jobs may be placed on, their pools' quotas, and a study whose
runs go to the cluster (falcon_autonomy.cluster; docs/cluster.md). The cloud-burst tools of the same group are in
tools/cloud.py.

    cluster_nodes     read   this machine and the configured nodes: name, kind, slots, pool, backends, video memory —
                             never an address, SSH user, key path or Kubernetes context (the `cluster` setting is
                             private)
    cluster_quotas    read   every pool: its nodes and slots, its limits (at once, per day), jobs started today and
                             this session's jobs on it now
    cluster_dispatch  act    a sweep, rare-case search or failure-rate estimate whose drives run on a pool of the
                             cluster (`falcon sweep|search|failrate run … --on cluster[:POOL]`): the study itself runs
                             here; its preview is the command line, its estimate the runs over the pool's slots
    cluster_cache_gc  act    the caches nothing refers to (cluster.gc; `falcon cluster gc`): a dry run (the default)
                             lists each item with its size and keeps the list as a plan for an hour; removing takes
                             that plan's id, asks the user on a card in every mode (no "approve all": it cannot be
                             undone) whose preview lists what goes now with sizes, and removes only what the plan
                             listed that is still removable. Never a run folder, the trash, private data, the cloud's
                             ledger or anything referenced

A module adds its tools' one-line descriptions to _compact.SHORT itself (MINE below; tools/__init__ explains why)."""
from __future__ import annotations

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact

MINE = {
    'cluster_nodes': 'Compute nodes jobs may run on: this machine with its GPUs and the configured nodes.',
    'cluster_quotas': 'The cluster\'s pools: nodes, slots, limits, jobs started today and running now.',
    'cluster_dispatch': 'Queue a sweep, search or failure-rate study whose runs go to a cluster pool.',
    'cluster_cache_gc': 'Clear caches nothing refers to, here and on cluster nodes: dry run first, then remove its '
                        'plan.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)


def _settings(ctx):
    return ctx.settings if ctx.win is None else ctx.win.settings


def _cfg(ctx) -> dict:
    from ...cluster import nodes as N
    return N.load(_settings(ctx))


@tool('cluster_nodes', 'cluster', 'read',
      'The compute nodes Falcon may place jobs on: this machine as the local node (its NVIDIA GPUs from nvidia-smi: '
      'name, memory total and used) and the nodes configured in Settings ▸ Compute, each by name, kind (ssh '
      'workstation, local, slurm, k8s), slots, pool, backends and video memory — never its address, SSH user, key '
      'path or Kubernetes context. A study placed on the cluster (cluster_dispatch) sends its drives to them.')
def cluster_nodes(ctx):
    from ...cluster import nodes as N
    from ...core import sysinfo
    cfg = _cfg(ctx)
    gpus = [{k: g.get(k) for k in ('index', 'name', 'memory_total_mb', 'memory_used_mb')} for g in sysinfo.gpus()]
    return {'local': {'name': 'local', 'gpus': gpus, 'placement': 'local'},
            'nodes': [N.public(n) for n in cfg['nodes']], 'pools': sorted(N.pools(cfg)),
            'placements': ['local'] + (['cluster'] + [f'cluster:{p}' for p in sorted(N.pools(cfg))]
                                       if cfg['nodes'] else [])}


@tool('cluster_quotas', 'cluster', 'read',
      'The cluster\'s pools: each with its nodes and slots, its limits (max_running: jobs at once, which wait '
      'beyond it; max_per_day: jobs started per day, refused beyond it; 0 is no limit), the jobs started on it '
      'today and this session\'s jobs on it now (running or waiting). Empty when no node is configured.')
def cluster_quotas(ctx):
    from ...cluster import nodes as N
    from ...cluster.scheduler import Usage
    from ...core.jobs import FINAL
    cfg = _cfg(ctx)
    u = Usage()
    rows = N.pools(cfg)
    q = ctx.queue if ctx.win is None else ctx.win.queue
    jobs = [j for j in (q.jobs if q is not None else ()) if j.spec.placement != 'local' and j.state not in FINAL]
    for name, p in rows.items():
        p['started_today'] = u.today(name)
        p['now'] = sum(1 for j in jobs if N.parse_placement(j.spec.placement)[1] in ('', name))
    return {'pools': rows, 'session_remote_jobs': len(jobs)}


def _built(ctx, kind, scenario, pool='', samples=20, budget=60, seed=0, world='', rig='', method='bayes',
           batch=4, threshold=0.5, dry=False):
    from .jobs import _search, _sweep
    if kind == 'sweep':
        k, title, argv, kw = _sweep(ctx, scenario, samples=samples, seed=seed, world=world, rig=rig, dry=dry)
    else:
        k, title, argv, kw = _search(ctx, scenario, method='estimate' if kind == 'failrate' else method,
                                     budget=budget, batch=batch, threshold=threshold, seed=seed, world=world, rig=rig,
                                     dry=dry)
    on = 'cluster' + (f':{pool}' if pool else '')
    return k, f'{title} on {on}', argv + ['--on', on], kw


def _pool_slots(ctx, pool: str) -> int:
    from ...cluster import nodes as N
    cfg = _cfg(ctx)
    if not cfg['nodes']:
        raise ToolError('no cluster is set up: the user adds nodes in Settings ▸ Compute (falcon cluster nodes add)')
    p = N.pools(cfg)
    if pool and pool not in p:
        raise ToolError(f'no pool {pool!r} (cluster_quotas lists them: {", ".join(sorted(p))})')
    return max(sum(v['slots'] for k, v in p.items() if not pool or k == pool), 1)


def _estimate(ctx, **a):
    from .jobs import _drive_estimate, _scenario, _world_rig
    slots = _pool_slots(ctx, a.get('pool', ''))
    n = int(a.get('samples', 20)) if a['kind'] == 'sweep' else int(a.get('budget', 60))
    steps = int(float(_scenario(ctx, a['scenario']).get('duration_s', 12)) * 20)
    est = _drive_estimate(ctx, _world_rig(ctx, a.get('world'), a.get('rig'))[1], steps, n)
    est['seconds'] = round(est['seconds'] / slots, 1)
    est['what'] = f'{est["what"]}, over {slots} slot{"s" * (slots > 1)} (this machine\'s throughput as the guide)'
    return est


def _preview(ctx, **a):
    _pool_slots(ctx, a.get('pool', ''))
    return C.command_line(_built(ctx, dry=True, **a)[2])


@tool('cluster_dispatch', 'cluster', 'act',
      'Queue a study whose drives run on the cluster: a sweep (samples), a rare-case search (method, budget) or a '
      'failure-rate estimate (failrate: adaptive importance sampling, at most budget runs), on a pool of the '
      'cluster (pool; every pool when empty). The study runs here and sends each drive to a free slot; a node lost '
      'mid-run has its drives queued again elsewhere. The study\'s folder is studies/<name> as for jobs_sweep and '
      'jobs_search (analysis_study, analysis_outputs read it). Refused: no cluster, an unknown pool; a scenario made '
      'from the dashcam footage never leaves this machine.',
      C.obj({'kind': C.s('the study', enum=['sweep', 'search', 'failrate']),
             'scenario': C.s('scenario key or id'), 'pool': C.s('a pool of the cluster (default: every pool)'),
             'samples': C.i('sweep: runs (default 20)', minimum=2, maximum=10000),
             'budget': C.i('search, failrate: runs at most (default 60)', minimum=4, maximum=5000),
             'method': C.s('search method', enum=['bayes', 'bayes-pof', 'cmaes', 'adaptive', 'random']),
             'batch': C.i('runs per batch (default 4)', minimum=1, maximum=256),
             'threshold': C.n('failure threshold (default 0.5)', minimum=0, maximum=1),
             'seed': C.i('seed'), 'world': C.s('world id'), 'rig': C.s('rig id')}, ['kind', 'scenario']),
      estimate=_estimate, preview=_preview)
def cluster_dispatch(ctx, **a):
    _pool_slots(ctx, a.get('pool', ''))
    from ...cluster.bundle import _private
    from .jobs import _scenario
    why = _private({'scenario': _scenario(ctx, a['scenario'])}, [])
    if why:
        raise ToolError(f'private: {why} — dashcam-derived data never leaves this machine')
    kind, title, argv, kw = _built(ctx, **a)
    return C.cli_job(ctx, kind, title, argv, **kw)


# ------------------------------------------------------------------------------------------------ cache gc
def _gc_ws(ctx) -> str:
    try:
        return C.workspace(ctx)
    except ToolError:
        return ''                                    # no project: the nodes only


def _gc_keep(ctx) -> set:
    """Global ids of this session's cluster jobs that are not finished (running, waiting or re-queueable)."""
    from ...core.jobs import FINAL
    q = ctx.queue if ctx.win is None else ctx.win.queue
    return {j.spec.gid for j in (q.jobs if q is not None else ()) if j.spec.placement != 'local' and j.spec.gid
            and j.state not in FINAL}


def _gc_plan(ctx, node, older_than_h, plan):
    from ...cluster import gc as G
    if not plan:
        raise ToolError('removing needs the plan id of a dry run (call with dry_run true first)')
    try:
        doc = G.load_plan(plan)
    except G.GCError as e:
        raise ToolError(str(e)) from None
    want = {'workspace': _gc_ws(ctx), 'node': node, 'older_than_h': float(older_than_h)}
    if doc['args'] != want:
        raise ToolError(f'the dry run {plan} was made with other arguments: run a new dry run with these (node '
                        f'{doc["args"].get("node") or "all"}, older_than_h {doc["args"].get("older_than_h")})')
    return doc


def _gc_brief(res: dict, limit: int = 5) -> list:
    from ...cluster.gc import human
    out = []
    for sec in res['sections']:
        row = {'where': sec['where'], 'items': sec['count'], 'size': human(sec['bytes'])}
        if sec['kind'] == 'store' and sec['frees'] != sec['bytes']:
            row['frees'] = human(sec['frees'])
        if not res['dry_run']:
            row.update(removed=sec['removed'], freed=human(sec['bytes_freed']))
        if sec['kept_why']:
            row['kept'] = sec['kept_why']
        top = sorted(sec['items'], key=lambda x: -int(x.get('bytes') or 0))[:limit]
        if top:
            row['largest'] = [f'{it["name"]} {human(it.get("bytes") or 0)}' for it in top]
        if sec['notes']:
            row['notes'] = sec['notes']
        if sec['error']:
            row['error'] = sec['error']
        out.append(row)
    return out


def _gc_preview(ctx, node='', older_than_h=24.0, dry_run=True, plan=''):
    from ...cluster import gc as G
    if dry_run:
        return 'a dry run: lists what would be removed, with sizes, and removes nothing'
    doc = _gc_plan(ctx, node, older_than_h, plan)
    res = G.collect(_settings(ctx), _gc_ws(ctx), node, older_than_h, dry_run=True, only=doc['items'],
                    keep=_gc_keep(ctx))
    return '\n'.join([f'remove for good what dry run {plan} listed and is still removable:']
                     + G.lines(res, 12)[:-1] + [f'{res["count"]} item(s), {G.human(res["bytes"])} — this cannot be '
                                                'undone'])


def _gc_gate(ctx, dry_run=True, **_a):
    """Tool.approval: a dry run is the mode's to decide; removing asks in every mode, never "approve all"."""
    if dry_run:
        return None
    return {'ask': True, 'allow_all': False,
            'why': 'it deletes cache files for good (they do not go to the trash): every clean-up asks you, in every '
                   'mode'}


@tool('cluster_cache_gc', 'cluster', 'act',
      'Clear the caches nothing refers to (docs/cluster.md, "Cleaning up"): this machine\'s content store (objects no '
      'dataset version, run index or training names), and on the cluster\'s nodes the scratch run folders and CARLA '
      'logs of jobs no longer there and Slurm array folders, each unchanged for older_than_h hours (default 24) and '
      'not of a running or re-queueable job. Never a run folder, the trash, private (dashcam) data, the cloud\'s '
      'ledger or anything referenced. dry_run (the default) lists what would go with sizes and returns a plan id '
      '(valid an hour); then dry_run false with that plan removes only what it listed and is still removable — the '
      'user approves it on a card in every mode. node: one node only.',
      C.obj({'node': C.s('one node\'s caches only (default: this machine, every node, the cloud)', maxLength=40),
             'older_than_h': C.n('hours unchanged (default 24)', minimum=0, maximum=87600),
             'dry_run': C.b('list only (default true)'),
             'plan': C.s('the dry run\'s plan id, to remove it', pattern='^[0-9a-f]{16}$')}),
      preview=_gc_preview, approval=_gc_gate)
def cluster_cache_gc(ctx, node='', older_than_h=24.0, dry_run=True, plan=''):
    from ...cluster import gc as G
    if node and node not in {n['name'] for n in _cfg(ctx)['nodes']}:
        raise ToolError(f'no node {node!r} (cluster_nodes lists them)')
    try:
        if dry_run:
            res = G.plan(_settings(ctx), _gc_ws(ctx), node, older_than_h, keep=_gc_keep(ctx))
            pid = G.save_plan(res)
            return {'dry_run': True, 'plan': pid, 'would_remove': res['count'], 'size': G.human(res['bytes']),
                    'sections': _gc_brief(res), 'errors': res['errors'],
                    'next': f'to remove exactly these: dry_run false, plan {pid} (the user approves it)'}
        doc = _gc_plan(ctx, node, older_than_h, plan)
        res = G.collect(_settings(ctx), _gc_ws(ctx), node, older_than_h, dry_run=False, only=doc['items'],
                        keep=_gc_keep(ctx))
    except G.GCError as e:
        raise ToolError(str(e)) from None
    return {'dry_run': False, 'plan': plan, 'removed': res['removed'], 'freed': G.human(res['bytes_freed']),
            'sections': _gc_brief(res), 'errors': res['errors']}
