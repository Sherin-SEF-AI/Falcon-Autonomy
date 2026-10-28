"""CarlaService: the one place that turns "run this on CARLA" into a queued job (standard library only).

Shared by the GUI and the CLI so both take the same path: it finds CARLA and its Python from the settings, owns
Falcon's server (started on demand before the first CARLA job, under its memory cap, on Falcon's own port — or a
server the user chose in Settings, which it never starts or stops), writes each job's inputs into its run folder,
submits the job to the JobQueue, and after a job: indexes the run in the catalogue, updates the measured throughput,
and tidies the server when a worker died without tidying up itself.
"""
from __future__ import annotations

import os
import sys

from ..core import protocol as P
from ..core.budget import Throughput
from ..core.catalog import Catalog
from ..core.jobs import JobQueue, JobSpec
from ..core.paths import ensure_dir, tilde
from ..core.provenance import read_json, write_json
from ..core.settings import MEMORY_RESERVE_GB, Settings
from . import server as S


class CarlaService:
    def __init__(self, settings: Settings, queue: JobQueue, workspace: str = '', fake_carla: str = '',
                 scope: bool | None = None):
        """fake_carla: a folder holding a fake `carla` package (tests): workers then run in this Python, unscoped,
        against it, and no server is started."""
        self.settings, self.queue = settings, queue
        self.workspace = workspace
        self.fake_carla = fake_carla
        self.scope = (not fake_carla) if scope is None else scope
        self.server: S.Server | None = None
        self.catalog: Catalog | None = None
        self.listeners: list = []
        self.server_error = ''
        queue.before_start = self._before_start
        queue.subscribe(self._on_job)
        if workspace:
            self.set_workspace(workspace)

    # ---------------------------------------------------------------- configuration
    def set_workspace(self, workspace: str):
        if self.catalog is not None:
            self.catalog.close()
        self.workspace = workspace
        self.catalog = Catalog(os.path.join(ensure_dir(workspace), 'catalog.sqlite')) if workspace else None

    @property
    def carla_root(self) -> str:
        r = self.settings.get('carla_root')
        if not r:
            found = S.candidate_roots()
            r = found[0] if found else ''
        return r

    @property
    def python(self) -> str:
        if self.fake_carla:
            return sys.executable
        return self.settings.get('client_python') or S.client_python(self.carla_root)

    @property
    def port(self) -> int:
        return int(self.settings.get('rpc_port'))

    @property
    def tm_port(self) -> int:
        return int(self.settings.get('tm_port'))

    @property
    def external(self) -> bool:
        """A server the user started and chose in Settings (a reserved port they allowed): never started or stopped."""
        return self.port in set(self.settings.get('allowed_ports') or [])

    def throughput(self) -> Throughput:
        return Throughput.from_settings(self.settings.get('measured'))

    # ---------------------------------------------------------------- the server
    def ensure_server_object(self) -> S.Server:
        if self.server is None or self.server.port != self.port:
            log = os.path.join(self.workspace, 'logs', f'carla-{self.port}.log') if self.workspace else ''
            self.server = S.Server(self.carla_root, self.port, float(self.settings.get('server_memory_gb')),
                                   self.settings.get('server_quality'), log_path=log, external=self.external,
                                   settings=self.settings)
        return self.server

    def server_state(self) -> str:
        if self.fake_carla:
            return 'fake'
        if self.server is None:
            return 'external' if self.external and S.port_open('127.0.0.1', self.port) else 'stopped'
        return self.server.state()

    def start_server(self):
        srv = self.ensure_server_object()
        self.server_error = ''
        srv.start()
        self._emit('server', srv.state())

    def stop_server(self):
        if self.server is not None:
            self.server.stop()
            self._emit('server', self.server.state())

    def _before_start(self, job) -> str:
        """Before a CARLA job starts: Falcon's server must be up (started here, its memory checked by the queue)."""
        if not job.spec.needs_carla or self.fake_carla or job.spec.meta.get('no_server'):
            self.queue.extra_memory_gb = 0.0
            return ''
        srv = self.ensure_server_object()
        st = srv.state()
        if st in ('ready', 'external'):
            self.queue.extra_memory_gb = 0.0
            return ''
        if st == 'starting':
            return f'waiting for the CARLA server on port {self.port} to start'
        if self.external:
            return f'waiting for your CARLA server on port {self.port} (Falcon does not start a server you chose)'
        from ..core.sysinfo import memory_ok
        ok, why = memory_ok(float(self.settings.get('server_memory_gb')) + job.spec.memory_gb, MEMORY_RESERVE_GB)
        if not ok:
            return f'not starting the CARLA server: {why}'
        try:
            srv.start()
        except RuntimeError as e:
            self.server_error = str(e)
            return f'cannot start the CARLA server: {e}'
        self._emit('server', srv.state())
        return f'starting the CARLA server on port {self.port}'

    # ---------------------------------------------------------------- jobs
    def worker_spec(self, worker: str, args: list, title: str, kind: str, run_dir: str = '', needs_carla=True,
                    memory_gb=S.WORKER_NEED_GB, meta: dict | None = None) -> JobSpec:
        py = self.python
        if not py:
            raise RuntimeError('No Python for CARLA\'s client: choose CARLA\'s folder (and its venv) in Settings.')
        argv, env = S.worker_command(py, worker, args, S.WORKER_MEMORY_GB, scope=self.scope,
                                     extra_path=[self.fake_carla] if self.fake_carla else ())
        return JobSpec(kind, title, argv, env, needs_carla=needs_carla, memory_gb=memory_gb, run_dir=run_dir,
                       meta=meta)

    def submit_drive(self, cfg: dict, title: str = '', kind='drive'):
        """Write cfg into its run folder (drive.json) and queue the drive."""
        run_dir = ensure_dir(cfg['out'])
        path = os.path.join(run_dir, 'drive.json')
        write_json(path, cfg)
        spec = self.worker_spec('drive', [path], title or cfg.get('name', 'drive'), kind, run_dir,
                                meta={'config': path})
        return self.queue.submit(spec)

    def submit_live(self, cfg: dict, title: str = 'Live session'):
        """Write a live config into its run folder (live.json) and queue the session (it runs until stopped)."""
        run_dir = ensure_dir(cfg['out'])
        path = os.path.join(run_dir, 'live.json')
        write_json(path, cfg)
        spec = self.worker_spec('live', [path], title, 'live', run_dir, meta={'config': path})
        return self.queue.submit(spec)

    def submit_probe(self, wait_s: float = 10.0):
        spec = self.worker_spec('probe', ['--port', self.port, '--wait', wait_s], f'Probe port {self.port}', 'probe',
                                memory_gb=0.3)
        return self.queue.submit(spec)

    def submit_tidy(self):
        spec = self.worker_spec('tidy', ['--port', self.port],
                                f'Tidy the server on port {self.port}', 'tidy', memory_gb=0.3,
                                meta={'no_server': True})
        return self.queue.submit(spec)

    def runs_dir(self) -> str:
        return ensure_dir(os.path.join(self.workspace, 'runs')) if self.workspace else ''

    def _on_job(self, ev, job):
        if ev != 'finished':
            return
        s = job.spec
        if s.run_dir and self.catalog is not None:
            self.catalog.add_run(s.run_dir)
            for m in job.other:                       # a live session's recordings are runs of their own
                if m.get('t') == 'recording' and m.get('path'):
                    self.catalog.add_run(m['path'])
                    self.catalog.link('run:' + os.path.basename(s.run_dir), 'run:' + os.path.basename(m['path']),
                                      'recorded')
        code = job.exit_code
        # a worker that died without tidying (killed, crashed) may leave the server synchronous with its actors
        if s.needs_carla and s.kind != 'tidy' and (code is None or code < 0 or code == P.CRASHED) and \
                self.server_state() in ('ready', 'external'):
            job.log.append(('warning', 'the worker ended without tidying up: tidying the server'))
            try:
                self.submit_tidy()
            except RuntimeError:
                pass
        if code == P.OK and s.kind in ('drive', 'smoke') and s.run_dir:
            self._update_throughput(s.run_dir)
        self._emit('job', job)

    def _update_throughput(self, run_dir):
        try:
            perf = read_json(os.path.join(run_dir, 'run.json')).get('performance') or {}
        except (OSError, ValueError):
            return
        if not perf.get('steps'):
            return
        if self.fake_carla:
            return                         # a fake server says nothing about this machine
        t = self.throughput().add(perf['mpx_per_step'], perf['wall_s'] / perf['steps'],
                                  f'{perf["steps"]} steps at {perf["mpx_per_step"]:.2f} Mpx in {perf["wall_s"]:.1f} s, '
                                  f'{tilde(run_dir)}')
        self.settings.set('measured', t.to_dict())

    def subscribe(self, fn):
        self.listeners.append(fn)

    def _emit(self, ev, payload):
        for fn in list(self.listeners):
            fn(ev, payload)

    def shutdown(self, grace_s: float = 20.0):
        """Cancel what is still queued or running (each worker tidies up; killed after grace_s), then stop Falcon's
        own server (never an external one) and close the catalogue."""
        import time
        from ..core.jobs import FINAL
        active = self.queue.active()
        for j in active:
            self.queue.cancel(j)
        t0 = time.monotonic()
        while any(j.state not in FINAL for j in active) and time.monotonic() - t0 < grace_s:
            time.sleep(0.1)
        for j in active:
            if j.state not in FINAL and j.handle is not None:
                j.handle.kill()
        if self.server is not None:
            self.server.stop()
        if self.catalog is not None:
            self.catalog.close()
            self.catalog = None
