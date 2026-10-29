"""Finding CARLA, starting and stopping Falcon's own server, and building worker commands (standard library only).

The server runs headless in its own systemd user scope with a memory cap and no swap:

    systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0 --unit falcon-carla-<port>-<pid> \\
        CarlaUE4-Linux-Shipping CarlaUE4 -RenderOffScreen -carla-rpc-port=<port> -nosound [-quality-level=Low] [--ros2]

with VK_ICD_FILENAMES (and VK_DRIVER_FILES) pinned to NVIDIA's Vulkan driver, else Unreal can pick the integrated
GPU that drives the desktop. Without systemd-run it is not started at all. Falcon starts servers only on its own
ports (settings.port_allowed) and never stops a server it did not start; a server the user started is "external":
Falcon connects to it only on a port the user chose in the app, and never stops it.

--ros2 (the setting server_ros2, off by default) turns on CARLA 0.9.16's native ROS 2 interface: its embedded Fast DDS
then publishes the sensors a client enabled with Sensor.enable_for_ros() and takes the vehicles' control topics
(docs/carla-lessons.md, "Native ROS 2", says what was measured).

Workers (falcon_autonomy.workers) run in CARLA's own Python with Falcon's source folder on PYTHONPATH, each in a
capped scope of its own.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

from ..core import sysinfo
from ..core.settings import port_allowed, rpc_ports

BINARY = os.path.join('CarlaUE4', 'Binaries', 'Linux', 'CarlaUE4-Linux-Shipping')
CLIENT_MODULES = ('carla', 'numpy', 'cv2')
WORKER_MEMORY_GB = 4.0           # a worker's scope cap
WORKER_NEED_GB = 1.0             # what a worker is counted as needing when the queue checks memory
START_TIMEOUT_S = 180.0


def is_carla_root(path) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, BINARY))


def carla_version(root) -> str:
    try:
        with open(os.path.join(root, 'VERSION'), encoding='utf-8') as fh:
            return fh.read().strip()
    except OSError:
        pass
    m = os.path.basename(os.path.normpath(root or '')).rpartition('_')[2]
    return m if m[:1].isdigit() else ''


def candidate_roots(home: str | None = None) -> list:
    """CARLA folders in the usual places (a folder holding CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping)."""
    home = home or os.path.expanduser('~')
    pats = [os.path.join(home, p) for p in ('carla', 'CARLA*', 'carla/CARLA*', 'Documents/carla/CARLA*',
                                            'Documents/*/carla/CARLA*', 'Documents/*/carla', 'opt/carla*')]
    pats += ['/opt/carla*', '/opt/carla-simulator']
    out = []
    for p in pats:
        for d in sorted(glob.glob(p)):
            d = os.path.abspath(d)
            if is_carla_root(d) and d not in out:
                out.append(d)
    return out


def client_python(root) -> str:
    """The Python of CARLA's client environment beside the CARLA folder (<root>/../venv or <root>/venv), or ''."""
    if not root:
        return ''
    for d in (os.path.join(os.path.dirname(root), 'venv'), os.path.join(root, 'venv')):
        py = os.path.join(d, 'bin', 'python')
        if os.path.isfile(py) and os.access(py, os.X_OK):
            return py
    return ''


def clean_env(extra: dict | None = None) -> dict:
    """The environment for a child in another Python: none of this Python's or Qt's variables."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'QT_', 'VIRTUAL_ENV'))}
    env.update(extra or {})
    return env


def check_client(python, timeout=60.0) -> dict:
    """Whether CARLA's Python imports what the workers need: {'ok', 'carla_version', 'python', 'missing', 'text'}."""
    if not python or not os.path.isfile(python):
        return {'ok': False, 'missing': list(CLIENT_MODULES), 'text': 'No Python for CARLA\'s client chosen.'}
    code = ('import importlib, json, sys\nout = {"python": sys.version.split()[0], "missing": []}\n'
            f'for m in {CLIENT_MODULES!r}:\n'
            '    try:\n        importlib.import_module(m)\n'
            '    except Exception:\n        out["missing"].append(m)\n'
            'try:\n    from importlib.metadata import version\n    out["carla_version"] = version("carla")\n'
            'except Exception:\n    out["carla_version"] = ""\nprint(json.dumps(out))\n')
    try:
        r = subprocess.run([python, '-I', '-c', code], capture_output=True, text=True, timeout=timeout,
                           env=clean_env())
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'ok': False, 'missing': list(CLIENT_MODULES), 'text': f'{python} did not run ({e}).'}
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {'ok': False, 'missing': list(CLIENT_MODULES), 'text': (r.stderr or r.stdout).strip()[-400:]}
    out['ok'] = not out['missing']
    out['text'] = (f'Python {out["python"]}, carla {out.get("carla_version") or "?"}' if out['ok'] else
                   f'Missing in CARLA\'s Python: {", ".join(out["missing"])} (pip install carla==0.9.16 numpy '
                   'opencv-python-headless)')
    return out


def port_open(host='127.0.0.1', port=2000, timeout=0.5) -> bool:
    try:
        socket.create_connection((host, int(port)), timeout).close()
        return True
    except OSError:
        return False


def falcon_source() -> str:
    """The folder holding the falcon_autonomy package as source files (the workers' PYTHONPATH), or ''."""
    roots = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
    if getattr(sys, 'frozen', False) and getattr(sys, '_MEIPASS', ''):
        roots.insert(0, sys._MEIPASS)                   # a bundle carries the workers' sources as data files
    for root in roots:
        pkg = os.path.join(root, 'falcon_autonomy')
        if all(os.path.isfile(os.path.join(pkg, *f)) for f in (('__init__.py',), ('core', 'protocol.py'),
                                                                ('workers', '__main__.py'))):
            return root
    return ''


def scoped(cmd: list, memory_gb: float, unit: str = '') -> list:
    """cmd in a systemd user scope capped at memory_gb (no swap). Raises RuntimeError without systemd-run."""
    sr = shutil.which('systemd-run')
    if not sr:
        raise RuntimeError('systemd-run is not available: Falcon only starts CARLA and its workers under a memory '
                           'cap.')
    pre = [sr, '--user', '--scope', '-q', '-p', f'MemoryMax={max(int(memory_gb * 1024), 256)}M',
           '-p', 'MemorySwapMax=0']
    if unit:
        pre += ['--unit', unit]
    return pre + list(cmd)


def registry_path(port: int) -> str:
    from ..core.paths import config_dir
    return os.path.join(config_dir(), 'servers', f'{int(port)}.json')


def registered(port: int) -> dict | None:
    """The Falcon-started server on this port, when one is recorded and its process is alive (another Falcon process
    — the GUI, a study, the CLI — may use it, never stop it)."""
    try:
        with open(registry_path(port), encoding='utf-8') as fh:
            d = json.load(fh)
        os.kill(int(d['pid']), 0)             # alive (the scope's process: systemd-run execs into the server)
        return d
    except (OSError, ValueError, KeyError):
        return None


def stop_orphan(port: int, timeout=20.0) -> bool:
    """Stop a Falcon-started server on this port whose owner process has exited (a server left running because
    another process's job was using it when its owner shut down). True when one was stopped."""
    d = registered(port)
    if not d:
        return False
    try:
        os.kill(int(d.get('owner_pid', 0)), 0)
        return False                          # its owner is alive: it stops its own server
    except (OSError, ValueError):
        pass
    pid = int(d['pid'])
    try:
        os.killpg(pid, signal.SIGTERM)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            os.kill(pid, 0)
            time.sleep(0.2)
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        os.remove(registry_path(port))
    except OSError:
        pass
    return True


class Server:
    """A CARLA server this process started (its own scope, capped) — or, with external=True, a server the user
    started on a port they chose: then start() and stop() do nothing and only health is watched."""

    STOPPED, STARTING, READY, CRASHED, EXTERNAL = 'stopped', 'starting', 'ready', 'crashed', 'external'

    def __init__(self, root='', port=2310, memory_gb=12.0, quality='Epic', no_render=False, log_path='',
                 external=False, settings=None, host='127.0.0.1', ros2=False):
        self.root, self.port, self.memory_gb = root, int(port), float(memory_gb)
        self.ros2 = bool(ros2)
        self.quality, self.no_render, self.log_path = quality, bool(no_render), log_path
        self.external, self.settings, self.host = bool(external), settings, host
        self.proc = None
        self.env = None
        self.started = 0.0
        self.exit_code = None
        self.unit = f'falcon-carla-{self.port}-{os.getpid()}'
        self.starts = 0
        self._log = None

    def command(self) -> list:
        args = [os.path.join(self.root, BINARY), 'CarlaUE4', f'-carla-rpc-port={self.port}', '-nosound']
        if self.no_render:
            args.append('-nullrhi')                       # (-quality-level=Low crashes UE4 with -nullrhi)
        else:
            args.append('-RenderOffScreen')
            if self.quality == 'Low':
                args.append('-quality-level=Low')
        if self.ros2:
            args.append('--ros2')                         # CARLA 0.9.16's native ROS 2 (its own Fast DDS)
        return scoped(args, self.memory_gb, self.unit)

    def environment(self) -> dict:
        env = clean_env()
        env.pop('DISPLAY', None)
        env.pop('WAYLAND_DISPLAY', None)
        if self.no_render:                                # no Vulkan driver at all: it cannot reach a GPU
            env['VK_DRIVER_FILES'] = env['VK_ICD_FILENAMES'] = '/nonexistent.json'
        else:
            icd = sysinfo.nvidia_icd()                    # the discrete GPU, never the desktop's integrated one
            if icd:
                env['VK_DRIVER_FILES'] = env['VK_ICD_FILENAMES'] = icd
        return env

    def start(self):
        if self.external or self.running:
            return
        if not is_carla_root(self.root):
            raise RuntimeError(f'{self.root or "(none)"} is not a CARLA folder ({BINARY} is missing).')
        if not port_allowed(self.port, self.settings):
            raise RuntimeError(f'Port {self.port} is one of CARLA\'s defaults (2000–2010, 8100), where a server you '
                               'started may listen: Falcon starts its own server elsewhere unless you choose this '
                               'port in Settings.')
        for p in rpc_ports(self.port):
            if port_open(self.host, p):
                raise RuntimeError(f'Port {p} is already in use (another server?). Falcon does not touch a server it '
                                   'did not start: choose another port in Settings.')
        self.env = self.environment()
        self.starts += 1
        if self.starts > 1:
            # a crashed (e.g. OOM-killed) scope stays loaded as failed under its name: systemd-run refuses the same
            # unit name again ("was already loaded"), so a restart in this process takes a new one (measured)
            self.unit = f'falcon-carla-{self.port}-{os.getpid()}-{self.starts}'
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            self._log = open(self.log_path, 'w', encoding='utf-8', errors='replace')
        self.proc = subprocess.Popen(self.command(), stdout=self._log or subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self.env,
                                     start_new_session=True, cwd=self.root)
        self.started = time.time()
        self.exit_code = None
        os.makedirs(os.path.dirname(registry_path(self.port)), exist_ok=True)
        with open(registry_path(self.port), 'w', encoding='utf-8') as fh:
            json.dump({'pid': self.proc.pid, 'owner_pid': os.getpid(), 'port': self.port, 'started': self.started}, fh)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def state(self) -> str:
        if self.external:
            return self.EXTERNAL if port_open(self.host, self.port) else self.STOPPED
        if self.proc is None:
            return self.CRASHED if self.exit_code not in (None, 0, -15) else self.STOPPED
        code = self.proc.poll()
        if code is not None:
            self.exit_code = code
            self.proc = None
            return self.CRASHED
        return self.READY if port_open(self.host, self.port) else self.STARTING

    def wait_ready(self, timeout=START_TIMEOUT_S, cancel=None) -> bool:
        """Block until the RPC port answers (True), the process ends or timeout (False)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            st = self.state()
            if st in (self.READY, self.EXTERNAL):
                return True          # the RPC port answers; the probe worker then waits for the world itself
            if st in (self.CRASHED, self.STOPPED):
                return False
            if cancel and cancel():
                return False
            time.sleep(0.5)
        return False

    def log_tail(self, n=20) -> str:
        if not self.log_path:
            return ''
        try:
            with open(self.log_path, encoding='utf-8', errors='replace') as fh:
                return ''.join(fh.readlines()[-n:])
        except OSError:
            return ''

    def stop(self, timeout=20.0):
        """SIGTERM to the server's process group, SIGKILL after timeout; the scope goes with it. External: nothing."""
        p = self.proc
        if self.external or p is None:
            return
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    pass
        self.exit_code = p.returncode
        self.proc = None
        try:
            os.remove(registry_path(self.port))
        except OSError:
            pass
        if self._log:
            self._log.close()
            self._log = None


def worker_command(python: str, worker: str, args=(), memory_gb=WORKER_MEMORY_GB, scope=True,
                   extra_path=(), unit: str = '') -> tuple[list, dict]:
    """(argv, env) running `python -m falcon_autonomy.workers <worker> args` in CARLA's Python with Falcon's source on
    PYTHONPATH (extra_path first: tests put the fake carla there), in a capped scope unless scope=False."""
    src = falcon_source()
    if not src:
        raise RuntimeError('The workers need Falcon\'s source folder (a frozen build carries it as data): none found.')
    cmd = [python, '-m', 'falcon_autonomy.workers', worker] + [str(a) for a in args]
    if scope:
        cmd = scoped(cmd, memory_gb, unit)
    path = os.pathsep.join(list(extra_path) + [src])
    return cmd, clean_env({'PYTHONPATH': path, 'PYTHONUNBUFFERED': '1'})
