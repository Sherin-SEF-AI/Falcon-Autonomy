"""`falcon cluster …`: the compute nodes jobs may be placed on (docs/cluster.md).

    falcon cluster nodes list [--json]
    falcon cluster nodes add NAME --kind ssh|local|slurm|k8s [--host H --user U --key PATH [--ssh-port P]] [--slots 0,1]
                             [--pool P] [--backend B]... [--vram-gb G] [slurm: --workdir --partition --account --gres
                             --time-limit --falcon --max-jobs] [k8s: --context --namespace --image --gpu --deadline-s
                             --max-jobs] [--token-dir D] [--replace]
    falcon cluster nodes test NAME [--json]      reach each slot (SSH, the daemon's token and hello) or the batch system
    falcon cluster nodes remove NAME
    falcon cluster pools [--json]                every pool: its nodes, slots, limits and use today
    falcon cluster pools set NAME [--max-running N] [--max-per-day N]   (0: no limit)
    falcon cluster pools remove NAME
    falcon cluster dispatch sweep|search|failrate PROJECT [--pool P] ARGS…   a study whose runs go to the cluster
                                                                            (`falcon sweep … --on cluster:P`)
    falcon cluster status [--json]               nodes reachable now, jobs running on them, pools used today
    falcon cluster gc [--dry-run] [--node N] [--older-than H] [--workspace W] [--json]
                                                 remove the caches nothing refers to: the workspace's content store,
                                                 the nodes' scratch folders and Slurm array folders (cluster.gc;
                                                 never a run folder, the trash, private data or the cloud's ledger)

Nodes live in the private `cluster` setting: an SSH key is a path to the user's key file, never the key. Exit codes:
0 ok, 2 failed (the message says why); dispatch exits with the study's code.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import nodes as N


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='falcon cluster', description='compute nodes, placement and quotas '
                                                                    '(docs/cluster.md)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    nd = sub.add_parser('nodes', help='list, add, test or remove nodes')
    ns = nd.add_subparsers(dest='action', required=True)
    ls = ns.add_parser('list', help='the configured nodes')
    ls.add_argument('--json', action='store_true')
    ad = ns.add_parser('add', help='add a node (the key is a path to your key file)')
    ad.add_argument('name')
    ad.add_argument('--kind', default='ssh', choices=N.KINDS)
    ad.add_argument('--host', default='')
    ad.add_argument('--user', default='', dest='ssh_user')
    ad.add_argument('--key', default='', dest='ssh_key', help='the path to your private key file (never the key)')
    ad.add_argument('--ssh-port', type=int, default=None)
    ad.add_argument('--slots', default='', help='slot numbers, comma-separated (default 0)')
    ad.add_argument('--pool', default='default')
    ad.add_argument('--backend', action='append', default=[], dest='backends')
    ad.add_argument('--vram-gb', type=float, default=None)
    ad.add_argument('--token-dir', default='')
    for opt in ('workdir', 'partition', 'account', 'gres', 'time-limit', 'falcon', 'context', 'namespace', 'image'):
        ad.add_argument('--' + opt, default='')
    ad.add_argument('--max-jobs', type=int, default=None)
    ad.add_argument('--gpu', type=int, default=None)
    ad.add_argument('--deadline-s', type=int, default=None)
    ad.add_argument('--replace', action='store_true')
    te = ns.add_parser('test', help='reach the node now')
    te.add_argument('name')
    te.add_argument('--json', action='store_true')
    rm = ns.add_parser('remove', help='remove a node')
    rm.add_argument('name')
    po = sub.add_parser('pools', help='pools and their quotas')
    po.add_argument('action', nargs='?', default='list', choices=('list', 'set', 'remove'))
    po.add_argument('pool', nargs='?', default='')
    po.add_argument('--max-running', type=int, default=None, help='jobs at once (0: no limit)')
    po.add_argument('--max-per-day', type=int, default=None, help='jobs started per day (0: no limit)')
    po.add_argument('--json', action='store_true')
    di = sub.add_parser('dispatch', help='a study whose runs go to the cluster')
    di.add_argument('kind', choices=('sweep', 'search', 'failrate'))
    di.add_argument('project')
    di.add_argument('--pool', default='')
    st = sub.add_parser('status', help='nodes reachable now and pools used today')
    st.add_argument('--json', action='store_true')
    from .gc import DEFAULT_H
    gc = sub.add_parser('gc', help='remove the caches nothing refers to: the workspace\'s content store and the '
                                   'nodes\' scratch (docs/cluster.md, "Cleaning up")')
    gc.add_argument('--dry-run', action='store_true', help='list what would be removed, with sizes; remove nothing')
    gc.add_argument('--node', default='', help='only this node (default: this machine\'s store, every node, the '
                                               'cloud)')
    gc.add_argument('--older-than', type=float, default=DEFAULT_H, metavar='H',
                    help=f'only what nothing changed in for H hours (default {DEFAULT_H:g}; the store keeps at least '
                         'the last hour\'s)')
    gc.add_argument('--workspace', default='', help='the workspace (or a project file) whose content store is '
                                                    'collected')
    gc.add_argument('--json', action='store_true')
    return ap


def node_from_args(a) -> dict:
    n = {'name': a.name, 'kind': a.kind, 'pool': a.pool}
    for k in ('host', 'ssh_user', 'ssh_key', 'ssh_port', 'vram_gb', 'token_dir', 'workdir', 'partition', 'account',
              'gres', 'falcon', 'context', 'namespace', 'image', 'max_jobs', 'gpu', 'deadline_s'):
        v = getattr(a, k, None)
        if v not in (None, ''):
            n[k] = v
    if a.time_limit:
        n['time_limit'] = a.time_limit
    if a.backends:
        n['backends'] = a.backends
    if a.slots and a.kind in ('ssh', 'local'):
        n['slots'] = [int(x) for x in a.slots.split(',') if x.strip()]
    return n


def test_node(node: dict, timeout: float = 20.0) -> dict:
    """Reach a node now: each slot's daemon (hello) or the batch system's own command."""
    out = {'name': node['name'], 'kind': node['kind'], 'ok': True, 'slots': []}
    if node['kind'] in ('ssh', 'local'):
        from .scheduler import Scheduler
        sch = Scheduler({'nodes': [node], 'pools': {}}, tick_s=3600)
        try:
            for s in sch.slots:
                row = {'slot': s.slot}
                try:
                    s.connect()
                    row.update(ok=True, node=s.info.get('node'), app_version=s.info.get('app_version'),
                               jobs=s.info.get('jobs'), codecs=s.info.get('codecs'),
                               backends=s.info.get('backends') or [])
                except Exception as e:           # noqa: BLE001 (reported per slot)
                    row.update(ok=False, error=str(e)[:300])
                    out['ok'] = False
                out['slots'].append(row)
        finally:
            sch.close()
        return out
    try:
        if node['kind'] == 'slurm':
            from .slurm import Backend
            b = Backend(_Stub(), node)
            code, so, se = b._run(['sinfo', '--version'], timeout=timeout)
        else:
            from .k8s import Backend
            b = Backend(_Stub(), node)
            code, so, se = b.kubectl('auth', 'can-i', 'create', 'jobs', timeout=timeout)
        out['ok'] = code == 0
        out['answer'] = (so or se).decode('utf-8', 'replace').strip()[-200:]
    except OSError as e:
        out.update(ok=False, error=str(e)[:300])
    return out


class _Stub:
    """What a batch backend asks of a scheduler when it is only tested (nodes test)."""

    def __init__(self):
        import threading
        self.lock = threading.RLock()
        self.jobs = {}


def main_cluster(argv) -> int:
    from ..core.settings import Settings
    ap = parser()
    a, extra = ap.parse_known_args(argv)
    if extra and a.cmd != 'dispatch':
        ap.error(f'unrecognized arguments: {" ".join(extra)}')
    st = Settings()
    try:
        cfg = N.load(st)
        if a.cmd == 'nodes':
            if a.action == 'list':
                if a.json:
                    print(json.dumps(cfg['nodes'], indent=1))
                    return 0
                if not cfg['nodes']:
                    print('no nodes: falcon cluster nodes add NAME … (docs/cluster.md)')
                for n in cfg['nodes']:
                    where = n.get('host') or n.get('context') or n.get('workdir') or 'this machine'
                    print(f'{n["name"]:20s} {n["kind"]:6s} pool {n["pool"]:10s} {N.capacity(n)} slot(s)  {where}')
                return 0
            if a.action == 'add':
                n = N.add(st, node_from_args(a), replace=a.replace)
                print(f'added {n["name"]} ({n["kind"]}, pool {n["pool"]}, {N.capacity(n)} slot(s)); try it: falcon '
                      f'cluster nodes test {n["name"]}')
                return 0
            if a.action == 'remove':
                if not N.remove(st, a.name):
                    print(f'failed: no node {a.name!r}', file=sys.stderr)
                    return 2
                print(f'removed {a.name}')
                return 0
            if a.action == 'test':
                node = next((n for n in cfg['nodes'] if n['name'] == a.name), None)
                if node is None:
                    print(f'failed: no node {a.name!r}', file=sys.stderr)
                    return 2
                r = test_node(node)
                if a.json:
                    print(json.dumps(r, indent=1))
                else:
                    for s in r['slots']:
                        print(f'slot {s["slot"]}: ' + (f'ok — {s.get("node")}, Falcon {s.get("app_version")}, '
                                                       f'{", ".join(s.get("backends") or []) or "no CARLA"}, '
                                                       f'{s.get("jobs")} job(s)' if s.get('ok') else
                                                       f'FAILED: {s.get("error")}'))
                    if not r['slots']:
                        print(('ok: ' + r.get('answer', '')) if r['ok'] else f'FAILED: {r.get("error") or r.get("answer")}')
                return 0 if r['ok'] else 2
        if a.cmd == 'pools':
            if a.action in ('set', 'remove'):
                if not a.pool:
                    print('failed: pools set|remove NAME', file=sys.stderr)
                    return 2
                N.set_pool(st, a.pool, a.max_running, a.max_per_day, remove_it=a.action == 'remove')
                cfg = N.load(st)
            return _pools(cfg, a.json)
        if a.cmd == 'dispatch':
            return _dispatch(a, extra)
        if a.cmd == 'gc':
            return _gc(a, st)
        if a.cmd == 'status':
            rows = [test_node(n, timeout=10) for n in cfg['nodes']]
            if a.json:
                print(json.dumps({'nodes': rows}, indent=1))
            else:
                for r in rows:
                    ok = sum(1 for s in r['slots'] if s.get('ok'))
                    print(f'{r["name"]:20s} ' + (f'{ok} of {len(r["slots"])} slot(s) answer' if r['slots'] else
                                                 ('reachable' if r['ok'] else f'NOT reachable: {r.get("error")}')))
                _pools(cfg, False)
            return 0
    except (N.NodeError, ValueError, OSError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
    return 2


def _pools(cfg: dict, as_json: bool) -> int:
    from .scheduler import Usage
    u = Usage()
    rows = N.pools(cfg)
    for name, p in rows.items():
        p['used_today'] = u.today(name)
    if as_json:
        print(json.dumps(rows, indent=1))
        return 0
    for name, p in sorted(rows.items()):
        lim = ', '.join(x for x in (f'{p["max_running"]} at once' if p['max_running'] else '',
                                    f'{p["max_per_day"]} a day' if p['max_per_day'] else '') if x) or 'no limits'
        print(f'{name:12s} {len(p["nodes"])} node(s), {p["slots"]} slot(s); {lim}; {p["used_today"]} started today')
    return 0


def _workspace(path: str) -> str:
    """A workspace folder, or a project file's workspace."""
    import os
    if not path:
        return ''
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        from ..core.project import Project
        return Project.load(path).workspace
    if not os.path.isdir(path):
        raise ValueError(f'{path}: no such workspace or project')
    return path


def _gc(a, st) -> int:
    from . import gc as G
    try:
        res = G.collect(st, _workspace(a.workspace), a.node, a.older_than, dry_run=a.dry_run)
    except G.GCError as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print('\n'.join(G.lines(res)))
    for e in res['errors']:
        print(f'failed: {e}', file=sys.stderr)
    return 2 if res['errors'] else 0


def _dispatch(a, rest: list) -> int:
    """The study command with --on cluster[:POOL], the rest of the arguments as given."""
    if '--on' in rest:
        print('failed: dispatch places the runs itself (--pool P, not --on)', file=sys.stderr)
        return 2
    on = 'cluster' + (f':{a.pool}' if a.pool else '')
    if a.kind == 'failrate':
        from ..sim.estimate import main_failrate
        return main_failrate(['run', a.project] + rest + ['--on', on])
    from ..sim.studio import main_study
    return main_study([a.kind, a.project] + rest + ['--on', on])
