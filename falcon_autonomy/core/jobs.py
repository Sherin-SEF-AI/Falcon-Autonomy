"""Background jobs: a queue, the app-wide CARLA lock and the memory guard (standard library only).

Every long action is a Job: a worker command (falcon_autonomy.sim.workers builds them) with what it needs. The
JobQueue starts a job when what it needs is free:

* needs_carla: the CARLA lock — one CARLA job at a time per GPU, across every Falcon process on the machine (a file
  lock in the config folder, so the CLI and the GUI queue behind each other);
* needs_gpu: the GPU lock — a training or evaluation job never runs while a CARLA job holds the GPU (and vice versa:
  a CARLA job waits for it), so the two never run at full load together;
* memory_gb: MemAvailable must cover it plus the reserve; else the job waits (it is never started to overcommit).

The queue is driven by poll(), called from the GUI's timer or the CLI's loop; a Launcher actually starts the
process (a QProcess in the GUI, subprocess in the CLI) and reports protocol messages and the exit back. The same job
specs, worker commands and protocol run in both, so the GUI and the CLI share every code path below the window.
"""
from __future__ import annotations

import errno
import itertools
import os
import subprocess
import threading
import time

from . import protocol as P
from .paths import config_dir
from .sysinfo import meminfo, memory_ok

QUEUED, WAITING, RUNNING, DONE, FAILED, CANCELLED = 'queued', 'waiting', 'running', 'done', 'failed', 'cancelled'
FINAL = (DONE, FAILED, CANCELLED)


class JobSpec:
    """What to run: argv and env (a worker command), and what it needs. run_dir holds its run.json."""

    def __init__(self, kind: str, title: str, argv: list, env: dict | None = None, needs_carla=False,
                 needs_gpu=False, memory_gb=1.0, run_dir: str = '', cwd: str | None = None, resumable=False,
                 meta: dict | None = None):
        self.kind, self.title, self.argv, self.env = kind, title, list(argv), env
        self.needs_carla, self.needs_gpu, self.memory_gb = bool(needs_carla), bool(needs_gpu), float(memory_gb)
        self.run_dir, self.cwd, self.resumable = run_dir, cwd, bool(resumable)
        self.meta = dict(meta or {})


class Job:
    _ids = itertools.count(1)

    def __init__(self, spec: JobSpec):
        self.id = next(Job._ids)
        self.spec = spec
        self.state = QUEUED
        self.progress = 0.0
        self.text = ''
        self.eta_s = None
        self.wait_reason = ''
        self.log: list = []              # (level, text)
        self.metrics: dict = {}
        self.result: dict | None = None
        self.exit_code = None
        self.submitted = time.time()
        self.started = None
        self.finished = None
        self.handle = None
        self.cancel_requested = False
        self.other: list = []           # protocol messages of other types (a worker's own: server, recording, …)

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def summary(self) -> str:
        if self.state == RUNNING:
            eta = f', ETA {self.eta_s:.0f} s' if self.eta_s is not None else ''
            return f'{100 * self.progress:.0f} %{eta} — {self.text}'
        if self.state == WAITING:
            return self.wait_reason
        if self.state in FINAL:
            return f'{self.state} ({P.describe_exit(self.exit_code)})' if self.exit_code else self.state
        return 'queued'


class FileLock:
    """An exclusive, non-blocking lock on a file (fcntl.flock); released when the process dies."""

    def __init__(self, name: str, directory: str | None = None):
        self.path = os.path.join(directory or config_dir(), name)
        self.fd = None

    def acquire(self) -> bool:
        if self.fd is not None:
            return True
        import fcntl
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                return False
            raise
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self.fd = fd
        return True

    def holder(self) -> str:
        try:
            with open(self.path, encoding='ascii') as fh:
                return fh.read().strip()
        except OSError:
            return ''

    def release(self):
        if self.fd is None:
            return
        import fcntl
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None

    @property
    def held(self) -> bool:
        return self.fd is not None


class Launcher:
    """Starts a job's process. start() returns a handle with cancel() (polite: stdin cancel, then SIGTERM) and
    kill(); the launcher calls queue.on_message(job, msg) per protocol line and queue.on_exit(job, code) once."""

    def start(self, job: Job, queue: 'JobQueue'):
        raise NotImplementedError


class SubprocessLauncher(Launcher):
    """For the CLI and tests: subprocess with a reader thread per pipe. Callbacks arrive on those threads; the
    queue's methods are thread-safe."""

    def start(self, job, queue):
        s = job.spec
        p = subprocess.Popen(s.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=s.env, cwd=s.cwd, text=True, bufsize=1, start_new_session=True)
        h = _ProcHandle(p)

        def out():
            for line in p.stdout:
                queue.on_message(job, P.parse_line(line))

        def err():
            for line in p.stderr:
                queue.on_message(job, {'t': 'log', 'level': 'stderr', 'text': line.rstrip('\n')})

        def wait():
            to, te = threading.Thread(target=out, daemon=True), threading.Thread(target=err, daemon=True)
            to.start()
            te.start()
            code = p.wait()
            to.join(5)
            te.join(5)
            queue.on_exit(job, code)
        threading.Thread(target=wait, daemon=True).start()
        return h


class _ProcHandle:
    def __init__(self, p):
        self.p = p

    def send(self, cmd: dict) -> bool:
        import json
        try:
            self.p.stdin.write(json.dumps(cmd) + '\n')
            self.p.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def cancel(self):
        try:
            self.p.stdin.write('{"cmd": "cancel"}\n')
            self.p.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            self.p.terminate()
        except OSError:
            pass

    def kill(self):
        try:
            os.killpg(self.p.pid, 9)
        except OSError:
            pass


class JobQueue:
    """Jobs in submission order; poll() starts what can start. Listeners get (event, job): 'added', 'changed',
    'finished'."""

    def __init__(self, launcher: Launcher, lock_dir: str | None = None, memory_reserve_gb=2.0, meminfo_fn=meminfo,
                 gpu_index=0):
        self.launcher = launcher
        self.jobs: list = []
        self.listeners: list = []
        self.carla_lock = FileLock(f'carla-gpu{gpu_index}.lock', lock_dir)
        self.gpu_lock = FileLock(f'compute-gpu{gpu_index}.lock', lock_dir)
        self.memory_reserve_gb = float(memory_reserve_gb)
        self.meminfo_fn = meminfo_fn
        self.extra_memory_gb = 0.0      # memory the server about to start for a job needs (set by the GUI/CLI)
        self._lock = threading.RLock()
        self.before_start = None        # callable(job) -> str | None: '' to start, or a reason to wait (server start)

    def subscribe(self, fn):
        self.listeners.append(fn)

    def _emit(self, ev, job):
        for fn in list(self.listeners):
            try:
                fn(ev, job)
            except Exception:           # noqa: BLE001 (a listener's bug must not stop the queue)
                import traceback
                traceback.print_exc()

    def submit(self, spec: JobSpec) -> Job:
        job = Job(spec)
        with self._lock:
            self.jobs.append(job)
        self._emit('added', job)
        self.poll()
        return job

    def get(self, job_id: int) -> Job | None:
        return next((j for j in self.jobs if j.id == job_id), None)

    def active(self) -> list:
        return [j for j in self.jobs if j.state not in FINAL]

    def cancel(self, job: Job):
        with self._lock:
            if job.state in (QUEUED, WAITING):
                job.state = CANCELLED
                job.finished = time.time()
                job.exit_code = P.CANCELLED
                self._emit('finished', job)
                return
            if job.state == RUNNING and job.handle is not None:
                job.cancel_requested = True
                job.handle.cancel()
                self._emit('changed', job)

    def send(self, job: Job, cmd: dict) -> bool:
        """A command to a running job's worker (its stdin). False when it is not running."""
        if job.state != RUNNING or job.handle is None:
            return False
        return job.handle.send(cmd)

    def poll(self):
        """Start every queued job whose needs are free, in order (a CARLA job waits behind an earlier CARLA job)."""
        with self._lock:
            carla_busy = any(j.state == RUNNING and j.spec.needs_carla for j in self.jobs)
            gpu_busy = any(j.state == RUNNING and j.spec.needs_gpu for j in self.jobs)
            for job in self.jobs:
                if job.state not in (QUEUED, WAITING):
                    continue
                s = job.spec
                why = ''
                if s.needs_carla and (carla_busy or gpu_busy):
                    why = 'waiting for the running CARLA job' if carla_busy else 'waiting for the GPU job to finish'
                elif s.needs_gpu and (carla_busy or gpu_busy):
                    why = 'waiting for the GPU (a CARLA or GPU job is running)'
                if not why and s.needs_carla and not self.carla_lock.acquire():
                    why = f'waiting for the CARLA lock (held by process {self.carla_lock.holder() or "?"})'
                if not why and s.needs_gpu and not self.gpu_lock.acquire():
                    why = f'waiting for the GPU lock (held by process {self.gpu_lock.holder() or "?"})'
                if not why:
                    ok, reason = memory_ok(s.memory_gb + (self.extra_memory_gb if s.needs_carla else 0.0),
                                           self.memory_reserve_gb, self.meminfo_fn())
                    if not ok:
                        why = reason
                if not why and self.before_start is not None:
                    why = self.before_start(job) or ''
                if why:
                    self._release_unused()
                    if job.state != WAITING or job.wait_reason != why:
                        job.state, job.wait_reason = WAITING, why
                        self._emit('changed', job)
                    if s.needs_carla:
                        carla_busy = True        # later CARLA jobs keep their order behind this one
                    continue
                job.state, job.started, job.wait_reason = RUNNING, time.time(), ''
                try:
                    job.handle = self.launcher.start(job, self)
                except OSError as e:
                    job.log.append(('error', f'could not start: {e}'))
                    self._finish(job, P.FAILED)
                    continue
                carla_busy = carla_busy or s.needs_carla
                gpu_busy = gpu_busy or s.needs_gpu
                self._emit('changed', job)

    def _release_unused(self):
        if not any(j.state == RUNNING and j.spec.needs_carla for j in self.jobs):
            self.carla_lock.release()
        if not any(j.state == RUNNING and j.spec.needs_gpu for j in self.jobs):
            self.gpu_lock.release()

    def on_message(self, job: Job, msg: dict):
        with self._lock:
            t = msg.get('t')
            if t == 'progress':
                job.progress = float(msg.get('f', job.progress))
                job.text = msg.get('text', job.text)
                job.eta_s = msg.get('eta_s')
            elif t == 'log':
                job.log.append((msg.get('level', 'info'), msg.get('text', '')))
            elif t == 'metric':
                job.metrics[msg.get('name', '?')] = msg
            elif t == 'result':
                job.result = msg
            elif t == 'hello':
                job.log.append(('info', f'worker {msg.get("worker")} started (pid {msg.get("pid")})'))
            else:
                job.other.append(msg)
        if t == 'log':
            self._emit('message', job)
        self._emit('changed', job)

    def on_exit(self, job: Job, code: int):
        with self._lock:
            self._finish(job, code)
        self.poll()

    def _finish(self, job, code):
        job.exit_code = code
        job.finished = time.time()
        if code == P.OK:
            job.state, job.progress = DONE, 1.0
        elif code == P.CANCELLED or (job.cancel_requested and code is not None and code < 0):
            job.state = CANCELLED
        else:
            job.state = FAILED
        job.handle = None
        self._release_unused()
        self._emit('finished', job)

    def wait(self, job: Job, timeout: float | None = None, interval=0.2) -> bool:
        """Block until job is final (the CLI's loop): polls the queue meanwhile. False on timeout."""
        t0 = time.monotonic()
        while job.state not in FINAL:
            self.poll()
            if timeout is not None and time.monotonic() - t0 > timeout:
                return False
            time.sleep(interval)
        return True
