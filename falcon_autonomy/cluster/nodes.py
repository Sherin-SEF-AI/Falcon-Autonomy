"""The user's cluster: its nodes and pools, kept in the private `cluster` setting (standard library; docs/cluster.md).

    {"nodes": [node, …], "pools": {"<pool>": {"max_running": n, "max_per_day": n}}}

A node is one of
    ssh     a workstation running the worker daemon's slots (`falcon-worker-daemon@N` units), reached through an SSH
            tunnel with the user's key: host, ssh_user, ssh_key (a path), ssh_port
    local   daemons on this machine (a second GPU here; the tests): token_dir where their tokens are (default: this
            machine's config folder)
    slurm   a Slurm cluster: jobs go as sbatch arrays running `falcon-worker-daemon oneshot` on its shared folder
            (workdir); its login host over SSH as above, or none when Falcon runs on the login node
    k8s     a Kubernetes cluster through kubectl (context, namespace, image, gpu, deadline_s)
and has a name, a pool ('default'), slots (numbers N — slot N's daemon listens on 127.0.0.1:2350+N of its node — or
{"slot": N, "port": P}), max_jobs (slurm/k8s: at once), what it can run (backends, vram_gb) and nothing secret: the
key is a path to the user's private key file, never its content, and no password is ever kept.
"""
from __future__ import annotations

import copy
import os
import re

from .protocol import PORT_BASE

KINDS = ('ssh', 'local', 'slurm', 'k8s')
DEFAULT_BACKEND = 'carla-0.9.16'
_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$')
_HOST = re.compile(r'^[A-Za-z0-9][A-Za-z0-9.:_-]*$')
_USER = re.compile(r'^[A-Za-z_][A-Za-z0-9._-]*$')
FIELDS = {
    'ssh': ('host', 'ssh_user', 'ssh_key', 'ssh_port', 'known_hosts'),
    'local': ('token_dir',),
    'slurm': ('host', 'ssh_user', 'ssh_key', 'ssh_port', 'known_hosts', 'workdir', 'partition', 'account', 'gres',
              'time_limit', 'falcon', 'max_jobs', 'sbatch_extra', 'poll_s'),
    'k8s': ('context', 'namespace', 'image', 'gpu', 'deadline_s', 'kubectl', 'max_jobs', 'hold_s', 'poll_s'),
}
COMMON = ('name', 'kind', 'pool', 'slots', 'backends', 'vram_gb', 'memory_gb')


class NodeError(ValueError):
    pass


def load(settings) -> dict:
    cfg = copy.deepcopy(settings.get('cluster') or {}) if settings is not None else {}
    cfg.setdefault('nodes', [])
    cfg.setdefault('pools', {})
    cfg['nodes'] = [n for n in cfg['nodes'] if isinstance(n, dict) and n.get('name')]
    for n in cfg['nodes']:
        n.setdefault('kind', 'ssh')
        n.setdefault('pool', 'default')
    return cfg


def save(settings, cfg: dict):
    settings.set('cluster', {'nodes': cfg.get('nodes') or [], 'pools': cfg.get('pools') or {}})


def slots(node: dict) -> list:
    """[(slot, port)] of a daemon node (ssh, local); [] for a batch node."""
    if node.get('kind') not in ('ssh', 'local'):
        return []
    raw = node.get('slots')
    if raw in (None, ''):
        raw = [0]
    if isinstance(raw, int):
        raw = list(range(raw))
    out = []
    for s in raw:
        if isinstance(s, dict):
            out.append((int(s['slot']), int(s.get('port') or PORT_BASE + int(s['slot']))))
        else:
            out.append((int(s), PORT_BASE + int(s)))
    return out


def capacity(node: dict) -> int:
    if node.get('kind') in ('slurm', 'k8s'):
        return max(int(node.get('max_jobs') or 4), 1)
    return len(slots(node))


def validate(node: dict) -> dict:
    """The node checked and normalised (NodeError says what is wrong). A key is a path to a file, never a key."""
    n = {k: v for k, v in node.items() if v not in (None, '')}
    kind = n.get('kind', 'ssh')
    if kind not in KINDS:
        raise NodeError(f'kind must be one of {", ".join(KINDS)}')
    if not _NAME.match(str(n.get('name', ''))):
        raise NodeError('a node needs a name: letters, digits, ".", "_" or "-" (at most 40)')
    allowed = set(COMMON) | set(FIELDS[kind])
    extra = sorted(set(n) - allowed)
    if extra:
        raise NodeError(f'a {kind} node has no {", ".join(extra)}')
    n['kind'] = kind
    n['pool'] = str(n.get('pool') or 'default')
    if not _NAME.match(n['pool']):
        raise NodeError('a pool name is letters, digits, ".", "_" or "-"')
    needs_ssh = kind == 'ssh' or (kind == 'slurm' and n.get('host'))
    if needs_ssh:
        if not _HOST.match(str(n.get('host', ''))):
            raise NodeError('the node\'s host is not set (or not a host name or IP address)')
        if not _USER.match(str(n.get('ssh_user', ''))):
            raise NodeError('the node\'s SSH user is not set (or not a valid user name)')
        key = str(n.get('ssh_key', ''))
        if 'PRIVATE KEY' in key or '\n' in key:
            raise NodeError('ssh_key is the path to your key file, never the key itself')
        if not key or not os.path.isfile(os.path.expanduser(key)):
            raise NodeError('the SSH key file is not set or does not exist (Falcon uses keys only: ssh-keygen, then '
                            'ssh-copy-id)')
        n['ssh_port'] = int(n.get('ssh_port') or 22)
    if kind == 'slurm' and not str(n.get('workdir', '')).startswith('/'):
        raise NodeError('a Slurm node needs workdir: an absolute folder shared by its compute nodes')
    if kind == 'k8s' and not n.get('image'):
        raise NodeError('a Kubernetes node needs image: the Falcon worker image (packaging/cluster/Dockerfile)')
    if kind in ('ssh', 'local'):
        n['slots'] = n.get('slots', [0])
        try:
            sl = slots(n)
        except (KeyError, TypeError, ValueError):
            raise NodeError('slots: numbers (or {"slot": N, "port": P})') from None
        if not sl or len({s for s, _ in sl}) != len(sl) or any(not 0 <= s < 50 for s, _ in sl):
            raise NodeError('slots: distinct numbers from 0 to 49')
        if any(not 1024 <= p <= 65535 for _, p in sl):
            raise NodeError('a slot\'s port must be 1024–65535')
    for k in ('vram_gb', 'memory_gb'):
        if k in n:
            n[k] = float(n[k])
    if 'backends' in n:
        n['backends'] = [str(b) for b in n['backends']]
    return n


def add(settings, node: dict, replace: bool = False) -> dict:
    cfg = load(settings)
    n = validate(node)
    have = [i for i, x in enumerate(cfg['nodes']) if x['name'] == n['name']]
    if have and not replace:
        raise NodeError(f'there is already a node named {n["name"]!r} (remove it first, or --replace)')
    if have:
        cfg['nodes'][have[0]] = n
    else:
        cfg['nodes'].append(n)
    save(settings, cfg)
    return n


def remove(settings, name: str) -> bool:
    cfg = load(settings)
    keep = [n for n in cfg['nodes'] if n['name'] != name]
    if len(keep) == len(cfg['nodes']):
        return False
    cfg['nodes'] = keep
    save(settings, cfg)
    return True


def set_pool(settings, name: str, max_running: int | None = None, max_per_day: int | None = None,
             remove_it: bool = False) -> dict:
    if not _NAME.match(name):
        raise NodeError('a pool name is letters, digits, ".", "_" or "-"')
    cfg = load(settings)
    if remove_it:
        cfg['pools'].pop(name, None)
    else:
        p = dict(cfg['pools'].get(name) or {})
        for k, v in (('max_running', max_running), ('max_per_day', max_per_day)):
            if v is not None:
                if int(v) < 0:
                    raise NodeError(f'{k} must be 0 (no limit) or more')
                p[k] = int(v)
        cfg['pools'][name] = p
    save(settings, cfg)
    return cfg['pools'].get(name, {})


def public(node: dict) -> dict:
    """What may be shown to the assistant (never host, user, key path, context)."""
    out = {k: node[k] for k in ('name', 'kind', 'pool', 'backends', 'vram_gb') if k in node}
    out['slots'] = capacity(node)
    return out


def pools(cfg: dict) -> dict:
    """Every pool: its nodes, slots and limits (a pool named by a node but not configured has no limits)."""
    out = {}
    for n in cfg['nodes']:
        p = out.setdefault(n.get('pool') or 'default', {'nodes': [], 'slots': 0, 'max_running': 0, 'max_per_day': 0})
        p['nodes'].append(n['name'])
        p['slots'] += capacity(n)
    for name, lim in (cfg.get('pools') or {}).items():
        p = out.setdefault(name, {'nodes': [], 'slots': 0, 'max_running': 0, 'max_per_day': 0})
        p['max_running'] = int(lim.get('max_running') or 0)
        p['max_per_day'] = int(lim.get('max_per_day') or 0)
    return out


def parse_placement(placement: str) -> tuple:
    """('local', '') | ('cluster', pool or '') from 'local', 'cluster', 'cluster:<pool>'; ValueError otherwise."""
    p = (placement or 'local').strip()
    if p == 'local':
        return 'local', ''
    if p == 'cluster':
        return 'cluster', ''
    if p.startswith('cluster:') and _NAME.match(p[8:]):
        return 'cluster', p[8:]
    raise ValueError(f'placement {placement!r}: local, cluster or cluster:<pool>')
