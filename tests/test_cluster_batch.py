"""Batch nodes against fakes on PATH: Slurm (sbatch arrays, squeue, sacct, scancel, a lost compute node) and
Kubernetes (kubectl: a Job per run with its bundle in a ConfigMap, logs followed, the run copied out, an evicted
pod). The fakes run `falcon-worker-daemon oneshot` here, on the fake carla."""
import json
import os
import signal
import stat
import sys

import pytest

from cluster_helpers import wait_for
from conftest import FAKE, ROOT
from falcon_autonomy.cluster import k8s as K
from falcon_autonomy.cluster.launcher import RemoteLauncher, close_all
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core.jobs import CANCELLED, DONE, JobQueue, SubprocessLauncher
from falcon_autonomy.core.project import Project
from falcon_autonomy.core.settings import Settings
from falcon_autonomy.sim.service import CarlaService, study_service
from falcon_autonomy.sim.studio import Study, sweep_samples

HERE = os.path.join(ROOT, 'tests', 'fake_slurm')


def _wrapper(tmp_path) -> str:
    """`falcon-worker-daemon` as the batch system runs it: this source, the fake carla."""
    w = tmp_path / 'bin' / 'falcon-test'
    w.parent.mkdir(exist_ok=True)
    w.write_text(f'#!/bin/sh\nexec env PYTHONPATH={ROOT} {sys.executable} -m falcon_autonomy.cluster.daemon "$@" '
                 f'--fake-carla {FAKE}\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC)
    return str(w)


def _kill_recorded(d):
    try:
        pids = [int(x) for x in open(os.path.join(d, 'pids')) if x.strip()]
    except OSError:
        return
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    b = tmp_path / 'slurm-bin'
    b.mkdir()
    for n in ('sbatch', 'squeue', 'sacct', 'scancel'):
        os.symlink(os.path.join(HERE, 'fakeslurm.py'), b / n)
    d = tmp_path / 'slurm-state'
    monkeypatch.setenv('PATH', f'{b}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_SLURM_DIR', str(d))
    node = {'name': 'hpc', 'kind': 'slurm', 'workdir': str(tmp_path / 'shared'), 'falcon': _wrapper(tmp_path),
            'max_jobs': 4, 'poll_s': 0.3, 'gres': 'gpu:1', 'partition': 'gpu'}
    yield node, d
    close_all()
    _kill_recorded(str(d))


@pytest.fixture
def kube(tmp_path, monkeypatch):
    b = tmp_path / 'kube-bin'
    b.mkdir()
    os.symlink(os.path.join(HERE, 'fakekubectl.py'), b / 'kubectl')
    d = tmp_path / 'kube-state'
    monkeypatch.setenv('FAKE_KUBE_DIR', str(d))
    monkeypatch.setenv('FAKE_KUBE_FALCON', _wrapper(tmp_path))
    node = {'name': 'kube', 'kind': 'k8s', 'image': 'falcon-worker:test', 'kubectl': str(b / 'kubectl'),
            'namespace': 'falcon', 'poll_s': 0.3, 'hold_s': 120, 'deadline_s': 1800, 'max_jobs': 3}
    yield node, d
    close_all()
    _kill_recorded(str(d))


def _project(path):
    p = Project.new('T')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(path))
    return p


def _settings(tmp_path, node):
    st = Settings(str(tmp_path / 'settings.json'))
    st.set('cluster', {'nodes': [node], 'pools': {}})
    return st


def _drive(p, out, seconds, seed=0):
    from falcon_autonomy.core.drive import drive_config
    cfg = drive_config(p, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=seconds, seed=seed)
    cfg['traffic'] = dict(cfg['traffic'], vehicles=0, walkers=0)
    return cfg


def _remote(st, ws):
    q = JobQueue(SubprocessLauncher())
    svc = CarlaService(st, q, ws, fake_carla=FAKE)
    q.remote = RemoteLauncher.for_settings(st)
    svc.placement = 'cluster'
    return svc, q


# ------------------------------------------------------------------------------------------------ Slurm
def test_a_sweep_as_slurm_arrays(tmp_path, slurm):
    node, d = slurm
    st = _settings(tmp_path, node)
    p = _project(tmp_path / 'p.falcon.json')
    scn = SC.builtin('lead_brake')
    svc = study_service(st, p.workspace, FAKE, placement='cluster')
    try:
        s = Study(svc, p, scn, 'hpc4', 'world-town10', 'rig-default', seed=1)
        rows = s.sweep(sweep_samples(s.space(), scn, 4, 1))
        assert all(j.state == DONE and j.node == 'hpc/slurm' for j in svc.queue.jobs), [j.log for j in svc.queue.jobs]
    finally:
        svc.shutdown()
    assert len(rows) == 4 and all(r['status'] == 'ok' for r in rows)
    subs = [json.loads(x) for x in open(d / 'submitted')]
    assert sum(x['tasks'] for x in subs) == 4                      # every run went as a task of an sbatch array
    script = subs[0]['text']
    assert '#SBATCH --gres=gpu:1' in script and '#SBATCH --partition=gpu' in script and 'oneshot --bundle' in script
    for r in rows:
        run = json.load(open(os.path.join(r['run'], 'run.json')))
        assert run['node']['kind'] == 'slurm' and run['node']['name'] == 'hpc' and run['status'] == 'ok'
        assert run['provenance']['settings']['out'] == r['run']           # this machine's paths in the record
    assert not [x for x in os.listdir(node['workdir']) if x.startswith('falcon-')]      # its folders are cleaned


def test_a_lost_slurm_task_is_queued_again(tmp_path, slurm, monkeypatch):
    node, d = slurm
    monkeypatch.setenv('FAKE_SLURM_NODE_FAIL', '0')
    st = _settings(tmp_path, node)
    p = _project(tmp_path / 'p.falcon.json')
    svc, q = _remote(st, p.workspace)
    try:
        job = svc.submit_drive(_drive(p, tmp_path / 'r1', 30.0), title='lost once')
        wait_for(lambda: job.state in ('done', 'failed', 'cancelled'), 180, q.poll)
        assert job.state == DONE and job.requeues == 1, job.log
        assert any('NODE_FAIL' in t and 'queued again' in t for _l, t in job.log)
    finally:
        svc.shutdown()
    assert json.load(open(tmp_path / 'r1' / 'run.json'))['status'] == 'ok'


def test_cancel_a_slurm_task(tmp_path, slurm):
    node, d = slurm
    st = _settings(tmp_path, node)
    p = _project(tmp_path / 'p.falcon.json')
    svc, q = _remote(st, p.workspace)
    try:
        job = svc.submit_drive(_drive(p, tmp_path / 'r1', 600.0), title='long')
        wait_for(lambda: job.progress > 0, 120, q.poll)
        q.cancel(job)
        wait_for(lambda: job.state in ('done', 'failed', 'cancelled'), 60, q.poll)
        assert job.state == CANCELLED and job.exit_code == P.CANCELLED
        assert [f for f in os.listdir(d) if f.endswith('.cancelled')]          # scancel was called for its task
    finally:
        svc.shutdown()


# ------------------------------------------------------------------------------------------------ Kubernetes
def test_a_run_as_a_kubernetes_job(tmp_path, kube):
    node, d = kube
    st = _settings(tmp_path, node)
    p = _project(tmp_path / 'p.falcon.json')
    svc, q = _remote(st, p.workspace)
    try:
        job = svc.submit_drive(_drive(p, tmp_path / 'r1', 5.0), title='k8s')
        wait_for(lambda: job.state in ('done', 'failed', 'cancelled'), 180, q.poll)
        assert job.state == DONE, job.log
        assert job.progress == 1.0 and any('Kubernetes Job falcon-' in t for _l, t in job.log)
    finally:
        svc.shutdown()
    run = json.load(open(tmp_path / 'r1' / 'run.json'))
    assert run['status'] == 'ok' and run['node']['kind'] == 'k8s'
    calls = [json.loads(x) for x in open(d / 'calls.jsonl')]
    name = next(c[2] for c in calls if c[:2] == ['create', 'configmap'])
    rec = json.load(open(d / f'{name}.json'))['job']
    spec = rec['spec']
    assert spec['backoffLimit'] == 0 and spec['activeDeadlineSeconds'] == 1800
    c = spec['template']['spec']['containers'][0]
    assert c['resources']['limits'] == {'nvidia.com/gpu': 1} and c['image'] == 'falcon-worker:test'
    assert c['args'][:2] == ['falcon-worker-daemon', 'oneshot'] and 'command' not in c
    assert ['delete', 'job', name, '--ignore-not-found', '--wait=false'] in calls
    assert ['delete', 'configmap', name, '--ignore-not-found'] in calls
    assert any(x[0] == 'exec' and 'touch' in x for x in calls)                # the pod was told it may end


def test_an_evicted_pod_is_queued_again_and_cancel_deletes_the_job(tmp_path, kube, monkeypatch):
    node, d = kube
    monkeypatch.setenv('FAKE_KUBE_EVICT', '1')
    st = _settings(tmp_path, node)
    p = _project(tmp_path / 'p.falcon.json')
    svc, q = _remote(st, p.workspace)
    try:
        job = svc.submit_drive(_drive(p, tmp_path / 'r1', 5.0), title='evicted once')
        wait_for(lambda: job.state in ('done', 'failed', 'cancelled'), 180, q.poll)
        assert job.state == DONE and job.requeues == 1, job.log
        assert any('Evicted' in t for _l, t in job.log)
        job2 = svc.submit_drive(_drive(p, tmp_path / 'r2', 600.0, seed=3), title='long')
        wait_for(lambda: job2.progress > 0, 120, q.poll)
        q.cancel(job2)
        wait_for(lambda: job2.state in ('done', 'failed', 'cancelled'), 60, q.poll)
        assert job2.state == CANCELLED
    finally:
        svc.shutdown()
    calls = [json.loads(x) for x in open(d / 'calls.jsonl')]
    assert len([c for c in calls if c[:2] == ['delete', 'job']]) >= 3        # evicted, finished, cancelled


def test_the_manifest_and_a_bundle_too_large_for_a_configmap(tmp_path):
    m = K.manifest({'image': 'img', 'gpu': 0}, 'falcon-x', 'abc.1')
    assert m['spec']['template']['spec']['containers'][0]['resources']['limits'] == {}
    assert K.job_name('ABC_def.2') == 'falcon-abc-def-2'

    class S:
        import threading
        lock = threading.RLock()
        jobs = {}
    b = K.Backend(S(), {'name': 'k', 'image': 'img', 'kubectl': '/nonexistent/kubectl'})
    with pytest.raises(OSError, match='over what a Kubernetes ConfigMap holds'):
        b.submit('r.1', None, b'x' * (1 << 20))
    with pytest.raises(OSError, match='is not installed'):
        b.submit('r.1', None, b'x')


def test_the_container_stand_in_for_systemd_run(tmp_path):
    import subprocess
    shim = os.path.join(ROOT, 'packaging', 'cluster', 'systemd-run-shim.sh')
    out = subprocess.run(['sh', shim, '--user', '--scope', '-q', '-p', 'MemoryMax=4096M', '-p', 'MemorySwapMax=0',
                          '--unit', 'falcon-carla-2310-1', 'echo', 'ran', '-p', 'x'], capture_output=True, text=True)
    assert out.stdout.strip() == 'ran -p x'
