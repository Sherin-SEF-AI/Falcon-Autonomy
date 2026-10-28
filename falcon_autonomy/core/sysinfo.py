"""The machine: memory, the NVIDIA GPU, Vulkan drivers (standard library only; Linux first).

Memory policy: Falcon checks MemAvailable before it starts anything heavy and refuses to overcommit — the job waits in
the queue instead (jobs.JobQueue). A CARLA server is capped by its systemd scope (settings.server_memory_gb, no swap),
so starting one needs that cap free plus MEMORY_RESERVE_GB; a heavy GUI-side job states its own need.
"""
from __future__ import annotations

import os
import shutil
import subprocess

ICD_DIRS = ('/etc/vulkan/icd.d', '/usr/share/vulkan/icd.d', '/usr/local/share/vulkan/icd.d')


def meminfo(path='/proc/meminfo') -> dict:
    """{'total_gb', 'available_gb', 'swap_total_gb', 'swap_free_gb'} from /proc/meminfo (0 where unknown)."""
    out = {'total_gb': 0.0, 'available_gb': 0.0, 'swap_total_gb': 0.0, 'swap_free_gb': 0.0}
    keys = {'MemTotal': 'total_gb', 'MemAvailable': 'available_gb', 'SwapTotal': 'swap_total_gb',
            'SwapFree': 'swap_free_gb'}
    try:
        with open(path, encoding='ascii') as fh:
            for line in fh:
                k, _, rest = line.partition(':')
                if k in keys:
                    out[keys[k]] = int(rest.split()[0]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return out


def memory_ok(need_gb: float, reserve_gb: float = 2.0, info: dict | None = None) -> tuple[bool, str]:
    """Whether need_gb can start now with reserve_gb left over: (ok, reason)."""
    m = info or meminfo()
    avail = m['available_gb']
    if avail <= 0:
        return True, 'memory unknown (no /proc/meminfo): not checked'
    if avail >= need_gb + reserve_gb:
        return True, f'{avail:.1f} GB available, {need_gb:.1f} GB needed + {reserve_gb:.1f} GB reserve'
    return False, (f'{avail:.1f} GB available but {need_gb:.1f} GB needed + {reserve_gb:.1f} GB reserve: waiting for '
                   'memory (close other programs, or lower the server\'s memory cap in Settings)')


def nvidia_driver_loaded(proc='/proc/driver/nvidia/version') -> bool:
    return os.path.exists(proc)


def nvidia_icd(dirs=ICD_DIRS) -> str:
    """NVIDIA's Vulkan driver manifest when its kernel driver is loaded, else ''."""
    if not nvidia_driver_loaded():
        return ''
    for d in dirs:
        f = os.path.join(d, 'nvidia_icd.json')
        if os.path.isfile(f):
            return f
    return ''


def vulkan_icds(dirs=ICD_DIRS) -> list:
    out = []
    for d in dirs:
        try:
            out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.json')]
        except OSError:
            pass
    return out


def gpus(timeout=5.0) -> list:
    """NVIDIA GPUs from nvidia-smi: [{'index', 'name', 'memory_total_mb', 'memory_used_mb', 'utilization_pct',
    'driver'}] ([] without nvidia-smi)."""
    smi = shutil.which('nvidia-smi')
    if not smi:
        return []
    q = 'index,name,memory.total,memory.used,utilization.gpu,driver_version'
    try:
        r = subprocess.run([smi, f'--query-gpu={q}', '--format=csv,noheader,nounits'], capture_output=True,
                           text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(',')]
        if len(f) < 6:
            continue
        try:
            out.append({'index': int(f[0]), 'name': f[1], 'memory_total_mb': float(f[2]),
                        'memory_used_mb': float(f[3]), 'utilization_pct': float(f[4]), 'driver': f[5]})
        except ValueError:
            continue
    return out


def has_systemd_run() -> bool:
    return bool(shutil.which('systemd-run'))


def checks() -> list:
    """The first-run checks that need no CARLA: [(name, ok, detail)]."""
    out = []
    g = gpus()
    out.append(('NVIDIA GPU', bool(g), ', '.join(f'{x["name"]} ({x["memory_total_mb"] / 1024:.0f} GB, driver '
                                                  f'{x["driver"]})' for x in g) or 'nvidia-smi found no GPU'))
    icd = nvidia_icd()
    out.append(('NVIDIA Vulkan driver', bool(icd),
                f'{icd} (CARLA is started with VK_ICD_FILENAMES pinned to it, never the desktop\'s iGPU)' if icd else
                'no nvidia_icd.json: Unreal would render on whatever Vulkan driver it finds first'))
    m = meminfo()
    out.append(('Memory', m['total_gb'] >= 16, f'{m["total_gb"]:.1f} GB total, {m["available_gb"]:.1f} GB available '
                                                 'now'))
    out.append(('systemd-run', has_systemd_run(), 'servers and workers run in memory-capped scopes' if
                has_systemd_run() else 'missing: Falcon only starts CARLA under a memory cap'))
    return out
