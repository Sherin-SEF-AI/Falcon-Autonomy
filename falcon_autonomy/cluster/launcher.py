"""RemoteLauncher: the JobQueue's door to the cluster's scheduler (standard library; docs/cluster.md).

JobQueue.poll() asks can_start(job) for a job placed on the cluster — '' when a slot is free, else why it waits;
core.jobs.Refused when it can never run there (no node fits, a bundle that cannot go: private, too large, a worker
not on the allow-list) — and then start(job, queue), which hands it to the scheduler's threads and returns at once.

One scheduler per process and cluster configuration (for_settings): the GUI's queue and every study in the process
share the node connections.
"""
from __future__ import annotations

import json
import threading

from ..core.jobs import Launcher, Refused
from . import nodes as N

_LOCK = threading.Lock()
_SCHEDULERS: dict = {}


class RemoteLauncher(Launcher):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    @classmethod
    def for_settings(cls, settings=None, **kw) -> 'RemoteLauncher':
        """The launcher of the cluster in these settings (Refused when it has no node)."""
        from ..core.settings import Settings
        cfg = N.load(settings if settings is not None else Settings())
        if not cfg['nodes']:
            raise Refused('no cluster is set up: add a node in Settings ▸ Compute (or falcon cluster nodes add)')
        key = json.dumps(cfg, sort_keys=True) + json.dumps(kw, sort_keys=True, default=str)
        with _LOCK:
            sch = _SCHEDULERS.get(key)
            if sch is None or sch.stop.is_set():
                from .scheduler import Scheduler
                sch = _SCHEDULERS[key] = Scheduler(cfg, **kw)
        return cls(sch)

    def can_start(self, job) -> str:
        return self.scheduler.can_start(job)

    def start(self, job, queue):
        return self.scheduler.start(job, queue)


def default_launcher() -> RemoteLauncher:
    """For a queue that meets a cluster job without a launcher (the GUI's): the user's cluster."""
    return RemoteLauncher.for_settings()


def close_all():
    """Stop every scheduler of this process (tests; the GUI at exit)."""
    with _LOCK:
        for s in _SCHEDULERS.values():
            s.close()
        _SCHEDULERS.clear()
