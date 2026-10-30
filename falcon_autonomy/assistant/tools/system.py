"""System tools (read): the CARLA server's state, the machine's GPU and memory, the job queue's locks, versions. What the
status bar and `falcon doctor` show, from the same calls; nothing here starts, stops or changes anything."""
from __future__ import annotations

import os

from ..tool_api import ToolError, tool
from . import _common as C


@tool('system_server', 'system', 'read',
      'Falcon\'s CARLA server as the status bar shows it: its state (ready, starting, stopped, crashed, external = a '
      'server the user started and chose, shared = another Falcon process\'s, fake = tests), its port, the last '
      'start error, and the tail of its log when it crashed. Never starts or stops it (only the job queue does, '
      'for a job).')
def system_server(ctx):
    svc = C.need_service(ctx)
    st = svc.server_state()
    out = {'state': st, 'port': svc.port, 'traffic_manager_port': svc.tm_port, 'external': svc.external,
           'memory_cap_gb': float(svc.settings.get('server_memory_gb')),
           'quality': svc.settings.get('server_quality'), 'stops_for_gpu_jobs': bool(svc.settings.get(
               'free_gpu_for_jobs'))}
    if svc.server_error:
        out['error'] = C.clip(svc.server_error, 300)
    if st == 'crashed' and svc.server is not None:
        out['log_tail'] = C.clip(svc.server.log_tail(8), 1200)
    return out


@tool('system_machine', 'system', 'read',
      'The machine: memory (total, available, swap) and the memory Falcon keeps free, the NVIDIA GPUs from '
      'nvidia-smi (name, memory used/total, utilisation, driver), the Vulkan driver check and systemd-run — the '
      'checks `falcon doctor` runs without CARLA. Measured now; nothing is estimated.')
def system_machine(ctx):
    from ...core import sysinfo
    from ...core.settings import MEMORY_RESERVE_GB
    m = sysinfo.meminfo()
    return {'memory_gb': {k: round(v, 1) for k, v in m.items()}, 'reserve_gb': MEMORY_RESERVE_GB,
            'gpus': sysinfo.gpus(), 'checks': [{'name': n, 'ok': bool(ok), 'detail': C.clip(d, 240)}
                                               for n, ok, d in sysinfo.checks()]}


@tool('system_queue', 'system', 'read',
      'The job queue\'s state: how many jobs run, wait (and why) and are queued; who holds the machine-wide CARLA '
      'lock and the GPU lock (a process id; this process or another Falcon process), and the extra memory the next '
      'CARLA job\'s server start needs. jobs_list lists the jobs themselves.')
def system_queue(ctx):
    from ...core.jobs import FINAL, QUEUED, RUNNING, WAITING
    q = C.need_queue(ctx)
    jobs = list(q.jobs)

    def lock(lk):
        held_here = lk.held
        holder = lk.holder()
        return {'held_by_this_process': held_here, 'holder_pid': holder or None,
                'this_process_pid': str(os.getpid())}
    return {'running': sum(j.state == RUNNING for j in jobs), 'waiting': sum(j.state == WAITING for j in jobs),
            'queued': sum(j.state == QUEUED for j in jobs), 'finished': sum(j.state in FINAL for j in jobs),
            'waiting_reasons': sorted({C.clip(j.wait_reason, 160) for j in jobs if j.state == WAITING})[:8],
            'carla_lock': lock(q.carla_lock), 'gpu_lock': lock(q.gpu_lock),
            'memory_reserve_gb': q.memory_reserve_gb}


@tool('system_versions', 'system', 'read',
      'Versions: Falcon\'s, the project format it reads, Python\'s, CARLA\'s (from the CARLA folder in Settings, when '
      'found) and the open project\'s name, file and the app version that last saved it.')
def system_versions(ctx):
    import platform
    from ... import __version__
    from ...core.project import VERSION
    from ...sim import server as S
    svc = C.need_service(ctx)
    root = svc.carla_root
    out = {'falcon': __version__, 'project_format': VERSION, 'python': platform.python_version(),
           'carla': (S.carla_version(root) or 'unknown') if root and S.is_carla_root(root) else 'CARLA folder not '
                                                                                              'found',
           'fake_carla': bool(svc.fake_carla)}
    try:
        p = C.project(ctx)
        out['project'] = {'name': p.doc.get('name'), 'file': p.path or '(not saved yet)',
                          'saved_by': p.doc.get('app_version'), 'unsaved_changes': bool(p.dirty)}
    except ToolError:                      # no project: the rest still answers
        pass
    return out
