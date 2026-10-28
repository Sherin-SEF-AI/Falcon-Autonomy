"""Against a real CARLA 0.9.16 server (FALCON_REAL_CARLA=1): Falcon starts its own server on its own port under a
memory cap (FALCON_TEST_SERVER_GB, default 8), probes it, records a short drive, checks the tidy-up, and stops it."""
import json
import os

import pytest

pytestmark = pytest.mark.real_server


@pytest.fixture(scope='module')
def svc(tmp_path_factory):
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    d = tmp_path_factory.mktemp('real')
    st = Settings(str(d / 'settings.json'))
    st.set('server_memory_gb', float(os.environ.get('FALCON_TEST_SERVER_GB', '8')))
    s = CarlaService(st, JobQueue(SubprocessLauncher(), str(d / 'locks')), str(d / 'ws'))
    yield s
    s.shutdown()
    assert s.server is None or not s.server.running


def test_probe_starts_falcons_own_server(svc):
    job = svc.submit_probe(wait_s=120.0)
    assert svc.queue.wait(job, 400) and job.state == 'done', job.log
    info = job.result['summary']
    assert info['server_version'].startswith('0.9.16') and info['map'] and not info['synchronous_mode']
    assert svc.server is not None and svc.server.port == svc.port and svc.port not in range(2000, 2011)
    assert svc.server.env['VK_ICD_FILENAMES'].endswith('nvidia_icd.json')


def test_drive_records_and_tidies(svc, tmp_path):
    from falcon_autonomy.core.drive import drive_config
    from falcon_autonomy.core.project import Project
    p = Project.new('real')
    cfg = drive_config(p, 'world-town10', 'rig-default', str(tmp_path / 'run'), svc.port, svc.tm_port, seconds=2.0,
                       seed=3)
    job = svc.submit_drive(cfg)
    assert svc.queue.wait(job, 600) and job.state == 'done', job.log
    run = json.load(open(tmp_path / 'run' / 'run.json'))
    assert run['performance']['steps'] == 40 and run['provenance']['host']['carla_server'].startswith('0.9.16')
    assert len(os.listdir(tmp_path / 'run' / 'semantic' / 'cam-front')) == 40
    assert svc.throughput().measured
    probe = svc.submit_probe(wait_s=10.0)
    assert svc.queue.wait(probe, 120) and not probe.result['summary']['synchronous_mode']
