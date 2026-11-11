"""Cluster tools (the cluster group; 4.0, in progress): the compute nodes jobs may be placed on. In this build only
cluster_nodes (read): the nodes configured in Settings ▸ Compute (the `cluster` setting) and this machine as the local
node with its GPUs — no address, user or key path is ever returned (the setting is private). Placement on other nodes,
quotas and cloud bursts come with falcon_autonomy.cluster (cluster_dispatch, cluster_quotas, cluster_spot_estimate,
cluster_spot_burst; tools/cloud.py).

A module adds its tools' one-line descriptions to _compact.SHORT itself (MINE below; tools/__init__ explains why)."""
from __future__ import annotations

from ..tool_api import tool
from . import _compact

MINE = {
    'cluster_nodes': 'Compute nodes jobs may run on: this machine with its GPUs and the configured nodes.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)

NODE_FIELDS = ('name', 'slots', 'pool', 'backends')        # what the model may see of a node (never host, user, key)


@tool('cluster_nodes', 'cluster', 'read',
      'The compute nodes Falcon may place jobs on: this machine as the local node (its NVIDIA GPUs from nvidia-smi: '
      'name, memory total and used) and the nodes configured in Settings ▸ Compute, each by name, slots, pool and '
      'backends — never its address, SSH user or key path. In this build every job runs on the local node; placement '
      'on other nodes is in progress (4.0).')
def cluster_nodes(ctx):
    from ...core import sysinfo
    st = ctx.settings if ctx.win is None else ctx.win.settings
    cfg = (st.get('cluster') if st is not None else None) or {}
    nodes = [{k: n[k] for k in NODE_FIELDS if k in n} for n in (cfg.get('nodes') or []) if isinstance(n, dict)]
    gpus = [{k: g.get(k) for k in ('index', 'name', 'memory_total_mb', 'memory_used_mb')} for g in sysinfo.gpus()]
    return {'local': {'name': 'local', 'gpus': gpus, 'placement': 'local'}, 'nodes': nodes,
            'note': 'every job runs on the local node in this build'}
