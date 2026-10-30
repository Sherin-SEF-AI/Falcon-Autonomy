"""The job queue (CARLA lock, GPU lock, memory guard, order, cancel), the service and the CLI."""
import json
import os
import subprocess
import sys

import pytest

from conftest import FAKE, ROOT, worker_env
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.jobs import (CANCELLED, DONE, FAILED, QUEUED, RUNNING, WAITING, FileLock, JobQueue,
                                       JobSpec, Launcher, SubprocessLauncher)


class FakeLauncher(Launcher):
    """Starts nothing: the test finishes jobs by hand."""

    def __init__(self):
        self.started = []

    def start(self, job, queue):
        self.started.append(job)

        class H:
            def cancel(self_):
                queue.on_exit(job, P.CANCELLED)

            def kill(self_):
                queue.on_exit(job, -9)
        return H()


def _mem(gb):
    return lambda: {'total_gb': 30.0, 'available_gb': gb, 'swap_total_gb': 0, 'swap_free_gb': 0}


def test_one_carla_job_at_a_time_in_order(tmp_path):
    L = FakeLauncher()
    q = JobQueue(L, str(tmp_path), meminfo_fn=_mem(28))
    a = q.submit(JobSpec('drive', 'a', ['x'], needs_carla=True))
    b = q.submit(JobSpec('drive', 'b', ['x'], needs_carla=True))
    c = q.submit(JobSpec('export', 'c', ['x']))                  # no CARLA: runs alongside
    assert a.state == RUNNING and b.state == WAITING and c.state == RUNNING
    assert 'CARLA job' in b.wait_reason
    q.on_exit(a, 0)
    assert a.state == DONE and b.state == RUNNING


def test_gpu_jobs_never_overlap_carla_jobs(tmp_path):
    q = JobQueue(FakeLauncher(), str(tmp_path), meminfo_fn=_mem(28))
    a = q.submit(JobSpec('train', 'train', ['x'], needs_gpu=True))
    b = q.submit(JobSpec('drive', 'drive', ['x'], needs_carla=True))
    assert a.state == RUNNING and b.state == WAITING and 'GPU' in b.wait_reason
    q.on_exit(a, 0)
    assert b.state == RUNNING


def test_the_carla_lock_is_machine_wide(tmp_path):
    other = FileLock('carla-gpu0.lock', str(tmp_path))            # another Falcon process holds it
    assert other.acquire()
    q = JobQueue(FakeLauncher(), str(tmp_path), meminfo_fn=_mem(28))
    j = q.submit(JobSpec('drive', 'd', ['x'], needs_carla=True))
    assert j.state == WAITING and 'CARLA lock' in j.wait_reason and str(os.getpid()) in j.wait_reason
    other.release()
    q.poll()
    assert j.state == RUNNING


def test_memory_guard_waits_instead_of_overcommitting(tmp_path):
    mem = {'gb': 9.0}
    q = JobQueue(FakeLauncher(), str(tmp_path), meminfo_fn=lambda: _mem(mem['gb'])())
    j = q.submit(JobSpec('train', 't', ['x'], memory_gb=8.0))
    assert j.state == WAITING and 'waiting for memory' in j.wait_reason
    mem['gb'] = 12.0
    q.poll()
    assert j.state == RUNNING


def test_cancel_queued_and_running(tmp_path):
    q = JobQueue(FakeLauncher(), str(tmp_path), meminfo_fn=_mem(28))
    a = q.submit(JobSpec('drive', 'a', ['x'], needs_carla=True))
    b = q.submit(JobSpec('drive', 'b', ['x'], needs_carla=True))
    q.cancel(b)
    assert b.state == CANCELLED and b.exit_code == P.CANCELLED
    q.cancel(a)
    assert a.state == CANCELLED
    assert not q.carla_lock.held


def test_subprocess_launcher_reads_the_protocol(tmp_path):
    code = ('import json,sys\n'
            'print(json.dumps({"t":"progress","f":0.5,"text":"half"}), flush=True)\n'
            'print("free text", flush=True)\n'
            'print("oops", file=sys.stderr)\n'
            'print(json.dumps({"t":"result","status":"failed"}), flush=True)\n'
            'sys.exit(2)\n')
    q = JobQueue(SubprocessLauncher(), str(tmp_path), meminfo_fn=_mem(28))
    j = q.submit(JobSpec('x', 'x', [sys.executable, '-c', code]))
    assert q.wait(j, 30)
    assert j.state == FAILED and j.exit_code == 2 and j.result['status'] == 'failed'
    assert ('info', 'free text') in j.log and ('stderr', 'oops') in j.log
    assert 'failed' in j.summary()


def test_service_runs_a_drive_and_catalogues_it(project, tmp_path, settings):
    from falcon_autonomy.core.drive import drive_config
    from falcon_autonomy.sim.service import CarlaService
    q = JobQueue(SubprocessLauncher(), str(tmp_path / 'locks'))
    svc = CarlaService(settings, q, project.workspace, fake_carla=FAKE)
    cfg = drive_config(project, 'world-town10', 'rig-default', os.path.join(svc.runs_dir(), 'r1'), svc.port,
                       svc.tm_port, seconds=0.5)
    job = svc.submit_drive(cfg)
    assert q.wait(job, 120) and job.state == DONE, job.log
    assert os.path.isfile(os.path.join(job.spec.run_dir, 'drive.json'))
    rows = svc.catalog.list('run')
    assert len(rows) == 1 and rows[0]['status'] == 'ok' and rows[0]['summary']['steps'] == 10
    assert svc.server_state() == 'fake' and settings.get('measured') == {}    # a fake measures nothing
    svc.shutdown()


def test_server_command_is_capped_scoped_and_on_the_nvidia_driver(settings, monkeypatch, tmp_path):
    from falcon_autonomy.core import sysinfo
    from falcon_autonomy.sim import server as S
    monkeypatch.setattr(sysinfo, 'nvidia_icd', lambda: '/usr/share/vulkan/icd.d/nvidia_icd.json')
    monkeypatch.setattr(S.shutil, 'which', lambda n: '/usr/bin/' + n)
    srv = S.Server('/opt/carla', 2310, 12.0, settings=settings)
    cmd = srv.command()
    assert cmd[:4] == ['/usr/bin/systemd-run', '--user', '--scope', '-q']
    assert 'MemoryMax=12288M' in cmd and 'MemorySwapMax=0' in cmd
    assert '-RenderOffScreen' in cmd and '-carla-rpc-port=2310' in cmd
    env = srv.environment()
    assert env['VK_ICD_FILENAMES'] == '/usr/share/vulkan/icd.d/nvidia_icd.json' and 'DISPLAY' not in env


def test_server_refuses_reserved_and_busy_ports(settings, tmp_path, monkeypatch):
    from falcon_autonomy.sim import server as S
    root = tmp_path / 'carla'
    (root / os.path.dirname(S.BINARY)).mkdir(parents=True)
    (root / S.BINARY).write_text('')
    with pytest.raises(RuntimeError, match='defaults'):
        S.Server(str(root), 2000, settings=settings).start()
    monkeypatch.setattr(S, 'port_open', lambda host, p, timeout=0.5: p == 2311)
    with pytest.raises(RuntimeError, match='2311 is already in use'):
        S.Server(str(root), 2310, settings=settings).start()


def test_external_server_is_never_started_or_stopped(settings, monkeypatch):
    from falcon_autonomy.sim import server as S
    srv = S.Server('', 2000, external=True, settings=settings)
    srv.start()
    srv.stop()
    assert srv.proc is None


def _fake_server(S, port, owner_pid):
    """A stand-in server process (its own session, as Server.start makes it), registered on the port."""
    p = subprocess.Popen(['sleep', '60'], start_new_session=True)
    os.makedirs(os.path.dirname(S.registry_path(port)), exist_ok=True)
    with open(S.registry_path(port), 'w', encoding='utf-8') as fh:
        json.dump({'pid': p.pid, 'owner_pid': owner_pid, 'port': port, 'started': 0}, fh)
    return p


def test_a_server_is_not_stopped_under_another_process_job(settings, tmp_path, monkeypatch):
    """Owner A stops its server only while holding the CARLA lock: a job of process B that took the lock (and uses
    A's server as shared) is never cut off. Measured before the fix: a 10-minute drive failed with "no CARLA server
    answered" when another process stopped the server it had found running."""
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.sim import server as S
    from falcon_autonomy.sim.service import CarlaService
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    locks = str(tmp_path / 'locks')
    svc = CarlaService(settings, JobQueue(SubprocessLauncher(), locks), str(tmp_path / 'ws'))
    srv = svc.ensure_server_object()
    srv.proc = _fake_server(S, srv.port, os.getpid())
    os.makedirs(locks, exist_ok=True)
    holder = subprocess.Popen([sys.executable, '-c', 'import fcntl, os, sys, time\n'
                               'fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n'
                               'fcntl.flock(fd, fcntl.LOCK_EX)\nprint("held", flush=True)\ntime.sleep(60)',
                               os.path.join(locks, 'carla-gpu0.lock')], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'held'
        assert svc.stop_server(wait_s=0.5) is False and srv.running and 'not stopped' in svc.server_error
        holder.kill()
        holder.wait()
        assert svc.stop_server(wait_s=5) is True and not srv.running
        assert S.registered(srv.port) is None and svc.queue.carla_lock.fd is None
    finally:
        holder.kill()
        if srv.proc is not None:
            srv.proc.kill()


def test_an_orphaned_server_is_stopped_by_the_next_process(tmp_path, monkeypatch):
    from falcon_autonomy.sim import server as S
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    dead = subprocess.Popen(['true'])
    dead.wait()
    live = _fake_server(S, 2390, os.getpid())
    assert S.stop_orphan(2390) is False and live.poll() is None          # its owner is alive
    live.kill()
    live.wait()
    orphan = _fake_server(S, 2390, dead.pid)
    try:
        assert S.stop_orphan(2390, timeout=5) is True
        assert orphan.wait(5) is not None and S.registered(2390) is None
    finally:
        orphan.kill()


def _cli(*args, cwd=ROOT):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *args], capture_output=True, text=True,
                          cwd=cwd, env=env, timeout=300)


def test_cli_project_drive_reproduce_runs(tmp_path):
    path = str(tmp_path / 'p.falcon.json')
    assert _cli('project', 'new', 'CLI', path).returncode == 0
    d = json.load(open(path))
    d['rigs'][0]['sensors'][0].update(width=96, height=64)
    d['traffic_profiles'][0]['vehicles'] = 2
    json.dump(d, open(path, 'w'))
    assert _cli('project', 'validate', path).stdout.strip() == 'valid'
    r = _cli('--fake-carla', FAKE, 'drive', path, '--seconds', '0.5', '--seed', '2')
    assert r.returncode == 0, r.stderr
    run = r.stdout.strip().splitlines()[-1]
    rr = _cli('--fake-carla', FAKE, 'reproduce', run)
    assert rr.returncode == 0 and 'reproduced within tolerance' in rr.stdout, rr.stdout + rr.stderr
    assert json.load(open(run + '-reproduced/reproduce.json'))['ok']
    lst = _cli('runs', path, '--rebuild')
    assert 'indexed 2 runs' in lst.stdout


def test_cli_rejects_an_invalid_project(tmp_path):
    path = tmp_path / 'bad.falcon.json'
    path.write_text(json.dumps({'format': 'falcon.project', 'version': 1, 'name': ''}))
    r = _cli('project', 'validate', str(path))
    assert r.returncode == P.FAILED and 'name' in r.stderr


def test_a_registered_falcon_server_is_shared_not_restarted(settings, tmp_path):
    """A study's own process uses the server the GUI started (recorded in the registry), never starts another."""
    import socket
    from falcon_autonomy.sim import server as S
    from falcon_autonomy.sim.service import CarlaService
    lst = socket.socket()
    lst.bind(('127.0.0.1', 0))
    lst.listen(1)
    port = lst.getsockname()[1]
    settings.set('rpc_port', port)
    svc = CarlaService(settings, JobQueue(SubprocessLauncher(), str(tmp_path)), str(tmp_path / 'ws'))
    assert svc.server_state() == 'stopped'
    os.makedirs(os.path.dirname(S.registry_path(port)), exist_ok=True)
    with open(S.registry_path(port), 'w') as fh:
        json.dump({'pid': os.getpid(), 'owner_pid': os.getpid(), 'port': port}, fh)
    assert svc.server_state() == 'shared'
    job = type('J', (), {'spec': JobSpec('drive', 'd', ['x'], needs_carla=True)})()
    assert svc._before_start(job) == ''
    os.remove(S.registry_path(port))
    lst.close()


def test_a_gpu_job_stops_the_idle_server_and_only_falcons_own(settings, tmp_path, monkeypatch):
    """An idle 0.9.16 server holds 5.9 of 16 GB of video memory (measured): a GPU job stops Falcon's own server
    first. Not with the setting off, not for a CARLA job, never a server the user chose, and not one whose CARLA
    lock another process holds (its job runs on it)."""
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.sim import server as S
    from falcon_autonomy.sim.service import CarlaService
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    locks = str(tmp_path / 'locks')
    svc = CarlaService(settings, JobQueue(SubprocessLauncher(), locks), str(tmp_path / 'ws'))
    srv = svc.ensure_server_object()
    gpu = lambda: type('J', (), {'spec': JobSpec('train', 't', ['x'], needs_gpu=True), 'log': []})()
    procs = []
    try:
        srv.proc = _fake_server(S, srv.port, os.getpid())
        procs.append(srv.proc)
        settings.set('free_gpu_for_jobs', False)
        job = gpu()
        assert svc._before_start(job) == '' and srv.running and job.log == []
        settings.set('free_gpu_for_jobs', True)
        carla = type('J', (), {'spec': JobSpec('drive', 'd', ['x'], needs_carla=True), 'log': []})()
        monkeypatch.setattr(srv, 'state', lambda: 'ready')
        assert svc._before_start(carla) == '' and srv.running
        monkeypatch.undo()
        monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
        os.makedirs(locks, exist_ok=True)
        holder = subprocess.Popen([sys.executable, '-c', 'import fcntl, os, sys, time\n'
                                   'fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n'
                                   'fcntl.flock(fd, fcntl.LOCK_EX)\nprint("held", flush=True)\ntime.sleep(60)',
                                   os.path.join(locks, 'carla-gpu0.lock')], stdout=subprocess.PIPE, text=True)
        procs.append(holder)
        assert holder.stdout.readline().strip() == 'held'
        job = gpu()
        assert svc._before_start(job) == '' and srv.running
        assert 'left running' in job.log[-1][1] and svc.server_error == ''
        holder.kill()
        holder.wait()
        job = gpu()
        assert svc._before_start(job) == '' and not srv.running and S.registered(srv.port) is None
        assert 'to free the GPU' in job.log[-1][1]
        srv.external = True                      # a server the user started: never stopped
        srv.proc = _fake_server(S, srv.port, os.getpid())
        procs.append(srv.proc)
        job = gpu()
        assert svc._before_start(job) == '' and srv.proc.poll() is None and job.log == []
    finally:
        for p in procs:
            p.kill()
