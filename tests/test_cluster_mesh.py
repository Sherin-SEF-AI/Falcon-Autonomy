"""The worker mesh against two local daemons and the fake carla: a sweep on the cluster gives the local run's results,
a lost daemon's job is queued again, cancel reaches the node, and what must never go is refused."""
import copy
import json
import os
import socket

import pytest

from cluster_helpers import Daemons, wait_for
from conftest import FAKE
from falcon_autonomy.cluster import bundle as B
from falcon_autonomy.cluster import nodes as N
from falcon_autonomy.cluster import protocol as W
from falcon_autonomy.cluster.launcher import RemoteLauncher, close_all
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import provenance as PV
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core.jobs import CANCELLED, DONE, FAILED, JobQueue, JobSpec, SubprocessLauncher
from falcon_autonomy.core.project import Project
from falcon_autonomy.core.settings import Settings
from falcon_autonomy.sim.service import CarlaService, study_service
from falcon_autonomy.sim.studio import Study, sweep_samples


@pytest.fixture
def daemons(tmp_path):
    d = Daemons(tmp_path / 'nodes')
    yield d
    close_all()
    d.stop()


def _project(path):
    p = Project.new('T')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(path))
    return p


def _settings(tmp_path, nodes, pools=None):
    st = Settings(str(tmp_path / 'settings.json'))
    st.set('cluster', {'nodes': nodes, 'pools': pools or {}})
    return st


VOLATILE = ('started', 'finished', 'performance', 'node')


def _norm(run: dict) -> dict:
    """run.json without what differs between any two runs (times, speeds) or between two folders (hashes of a
    config that holds its own path: checked against this machine's drive.json instead)."""
    r = {k: copy.deepcopy(v) for k, v in run.items() if k not in VOLATILE + ('config_sha256',)}
    r.get('provenance', {}).pop('created', None)
    r.get('provenance', {}).get('inputs', {}).get('config', {}).pop('sha256', None)
    for k in ('realtime_factor', 'steps_per_s'):
        r.get('summary', {}).pop(k, None)
    return r


def _diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in _diff(a.get(k), b.get(k), f'{path}.{k}')]
    return [] if a == b else [(path, a, b)]


def test_a_sweep_on_two_daemons_matches_the_local_run(tmp_path, daemons):
    a, b = daemons.start('node-a'), daemons.start('node-b')
    st = _settings(tmp_path, [a, b])
    scn = SC.builtin('lead_brake')
    results = {}
    for where in ('local', 'cluster'):
        p = _project(tmp_path / where / 'p.falcon.json')
        svc = study_service(st, p.workspace, FAKE, placement=where)
        try:
            s = Study(svc, p, scn, 'sweep12', 'world-town10', 'rig-default', seed=1)
            rows = s.sweep(sweep_samples(s.space(), scn, 12, 1))
            if where == 'cluster':
                jobs = svc.queue.jobs
                assert len(jobs) == 12 and all(j.state == DONE for j in jobs)
                assert {j.node.split('/')[0] for j in jobs} == {'node-a', 'node-b'}     # both nodes took drives
                assert all(j.spec.placement == 'cluster' and j.spec.gid for j in jobs)
        finally:
            svc.shutdown()
        assert len(rows) == 12 and all(r['status'] == 'ok' for r in rows)
        results[where] = {r['key']: r for r in rows}
    assert set(results['local']) == set(results['cluster'])                  # the same cache keys: the same drives
    for key, loc in results['local'].items():
        rem = results['cluster'][key]
        assert rem['criticality'] == loc['criticality'] and rem['values'] == loc['values']
        lr = json.load(open(os.path.join(loc['run'], 'run.json')))
        rr = json.load(open(os.path.join(rem['run'], 'run.json')))
        assert 'node' not in lr and rr['node']['name'] in ('node-a', 'node-b') and rr['node']['dispatch']
        assert rr['node']['kind'] == 'local' and rr['node']['slot'] == 0
        lw, rw = os.path.dirname(os.path.dirname(loc['run'])), os.path.dirname(os.path.dirname(rem['run']))
        ln = json.dumps(_norm(lr)).replace(lw, '<study>')
        rn = json.dumps(_norm(rr)).replace(rw, '<study>')
        d = _diff(json.loads(ln), json.loads(rn))
        assert not d, json.dumps(d)                                           # identical but for the node field
        assert sorted(os.listdir(loc['run'])) == sorted(os.listdir(rem['run']))
        drive = json.load(open(os.path.join(rem['run'], 'drive.json')))      # this machine's config, as written here
        assert drive['out'] == rem['run'] and drive['port'] == lr['provenance']['settings']['port']
        assert rr['config_sha256'] == PV.sha256_json(drive) and rr['node']['config_sha256']
        assert rr['provenance']['inputs']['config']['sha256'] == PV.sha256_file(os.path.join(rem['run'],
                                                                                              'drive.json'))


def _long_drive(p, out, seconds=120.0, seed=0):
    from falcon_autonomy.core.drive import drive_config
    cfg = drive_config(p, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=seconds, seed=seed)
    cfg['traffic'] = dict(cfg['traffic'], vehicles=0, walkers=0)
    return cfg


def _remote_service(st, ws, loss_s=60.0):
    q = JobQueue(SubprocessLauncher())
    svc = CarlaService(st, q, ws, fake_carla=FAKE)
    q.remote = RemoteLauncher.for_settings(st, loss_s=loss_s)
    svc.placement = 'cluster'
    return svc, q


def test_a_lost_daemon_s_job_is_queued_again_on_another(tmp_path, daemons):
    a, b = daemons.start('node-a'), daemons.start('node-b')
    st = _settings(tmp_path, [a, b])
    p = _project(tmp_path / 'p.falcon.json')
    svc, q = _remote_service(st, p.workspace, loss_s=2.0)
    try:
        wait_for(lambda: all(s.state == 'up' for s in q.remote.scheduler.slots), 30)
        job = svc.submit_drive(_long_drive(p, tmp_path / 'run1'), title='long')
        wait_for(lambda: job.state == 'running' and job.progress > 0, 60, q.poll)
        assert job.node == 'node-a/0'                      # both idle: the first by name
        daemons.kill('node-a')                             # SIGKILL: no goodbye, its connection just ends
        wait_for(lambda: job.state in ('done', 'failed', 'cancelled'), 120, q.poll)
        assert job.state == DONE, job.log
        assert job.requeues == 1 and job.node == 'node-b/0'
        assert any('node node-a/0 lost' in t and 'queued again (attempt 2 of 3)' in t for _lvl, t in job.log)
        run = json.load(open(tmp_path / 'run1' / 'run.json'))
        assert run['status'] == 'ok' and run['node']['name'] == 'node-b' and run['node']['dispatch'].endswith('.2')
    finally:
        svc.shutdown()


def test_cancel_reaches_the_node(tmp_path, daemons):
    a = daemons.start('node-a')
    st = _settings(tmp_path, [a])
    p = _project(tmp_path / 'p.falcon.json')
    svc, q = _remote_service(st, p.workspace)
    try:
        job = svc.submit_drive(_long_drive(p, tmp_path / 'run1', seconds=600.0), title='long')
        wait_for(lambda: job.state == 'running' and job.progress > 0, 60, q.poll)
        q.cancel(job)
        wait_for(lambda: job.state in ('done', 'failed', 'cancelled'), 60, q.poll)
        assert job.state == CANCELLED and job.exit_code == P.CANCELLED
        assert os.path.isfile(tmp_path / "run1" / "run.json"), (job.log, os.listdir(tmp_path / "run1"))
        run = json.load(open(tmp_path / 'run1' / 'run.json'))      # the node's worker tidied up and said so
        assert run['status'] == 'cancelled' and run['node']['name'] == 'node-a', (run.get('error'), run.get('notes'), job.log)
    finally:
        svc.shutdown()


def test_pools_and_quotas(tmp_path, daemons):
    a, b = daemons.start('node-a', pool='gpu'), daemons.start('node-b', pool='gpu')
    st = _settings(tmp_path, [a, b], {'gpu': {'max_running': 1}})
    p = _project(tmp_path / 'p.falcon.json')
    svc, q = _remote_service(st, p.workspace)
    svc.placement = 'cluster:gpu'
    try:
        wait_for(lambda: all(s.state == 'up' for s in q.remote.scheduler.slots), 30)
        j1 = svc.submit_drive(_long_drive(p, tmp_path / 'r1', seconds=30.0), title='one')
        j2 = svc.submit_drive(_long_drive(p, tmp_path / 'r2', seconds=30.0, seed=1), title='two')
        q.poll()
        assert j1.state == 'running' and j2.state == 'waiting'
        assert 'pool gpu runs its quota of 1 jobs at once' in j2.wait_reason
        wait_for(lambda: j1.state == DONE and j2.state == DONE, 120, q.poll)
        svc.placement = 'cluster:cpu'                                  # no such pool: refused, never started
        j3 = svc.submit_drive(_long_drive(p, tmp_path / 'r3', seconds=1.0), title='three')
        assert j3.state == FAILED and j3.exit_code == P.REFUSED and "no node of pool 'cpu'" in j3.log[-1][1]
    finally:
        svc.shutdown()
    from falcon_autonomy.cluster.scheduler import Usage
    assert Usage().today('gpu') >= 2
    N.set_pool(st, 'gpu', max_per_day=1)
    close_all()
    svc, q = _remote_service(st, p.workspace)
    svc.placement = 'cluster:gpu'
    try:
        j4 = svc.submit_drive(_long_drive(p, tmp_path / 'r4', seconds=1.0), title='four')
        q.poll()
        assert j4.state == FAILED and 'used its quota of 1 jobs today' in j4.log[-1][1]
    finally:
        svc.shutdown()


# ------------------------------------------------------------------------------------------------ refusals
def test_what_never_goes_to_the_cluster(tmp_path, daemons):
    a = daemons.start('node-a')
    st = _settings(tmp_path, [a])
    p = _project(tmp_path / 'p.falcon.json')
    svc, q = _remote_service(st, p.workspace)
    try:
        live = q.submit(JobSpec('live', 'a live session', ['x'], placement='cluster', meta={'interactive': True}))
        assert live.state == FAILED and live.exit_code == P.REFUSED and 'interactive' in live.log[-1][1]
        osm = svc.worker_spec('osm_import', ['x.json'], 'import', 'osm_import', needs_carla=False)
        assert osm.placement == 'local' and 'does not run on the cluster' in osm.meta['placement_note']
        rogue = q.submit(JobSpec('stack', 'a stack run', ['x'], placement='cluster', meta={'worker': 'stack'}))
        assert rogue.state == FAILED and 'the stack worker does not run on the cluster' in rogue.log[-1][1]
        cfg = _long_drive(p, tmp_path / 'dash', seconds=1.0)            # a drive of a scenario from the dashcam
        cfg['scenario'] = {'id': 'incident-1', 'source': {'format': 'dashcam'}}
        j = svc.submit_drive(cfg, title='private')
        assert j.state == FAILED and j.log[-1][1].startswith('refused: private:') and 'dashcam' in j.log[-1][1]
        big = tmp_path / 'big.bin'
        big.write_bytes(os.urandom(6 << 20))
        cfg = _long_drive(p, tmp_path / 'big', seconds=1.0)
        cfg['extra_file'] = str(big)
        j = svc.submit_drive(cfg, title='big')
        assert j.state == FAILED and 'over its cap of 5120 KiB' in j.log[-1][1]
        clips = tmp_path / 'clips'                                      # a folder ingested for near-miss mining
        os.makedirs(clips / 'falcon-ingest' / 'events' / 'e1')
        (clips / 'falcon-ingest' / 'ingest.json').write_text('{}')
        (clips / 'falcon-ingest' / 'events' / 'e1' / 'track.json').write_text('{}')
        cfg = _long_drive(p, tmp_path / 'ingest', seconds=1.0)
        cfg['tracks'] = str(clips / 'falcon-ingest' / 'events' / 'e1' / 'track.json')
        j = svc.submit_drive(cfg, title='an ingest event')
        assert j.state == FAILED and 'private:' in j.log[-1][1] and 'near-miss' in j.log[-1][1]
        cfg = _long_drive(p, tmp_path / 'dir', seconds=1.0)
        cfg['folder'] = str(tmp_path / 'p.falcon')
        j = svc.submit_drive(cfg, title='folder')
        assert j.state == FAILED and 'names a folder on this machine' in j.log[-1][1]
    finally:
        svc.shutdown()
    q2 = JobQueue(SubprocessLauncher())
    j = q2.submit(JobSpec('drive', 'd', ['x'], placement='cluster', meta={'worker': 'drive'}))
    assert j.state == FAILED and 'no cluster is set up' in j.log[-1][1]
    with pytest.raises(ValueError, match='no cluster is set up'):
        study_service(Settings(str(tmp_path / 'empty.json')), str(tmp_path), FAKE, placement='cluster')
    with pytest.raises(ValueError, match='local, cluster or cluster:<pool>'):
        study_service(st, str(tmp_path), FAKE, placement='cloud')


def _hello(port, token):
    s = socket.create_connection(('127.0.0.1', port), timeout=10)
    ch = W.Channel(s)
    ch.send({'t': 'hello', 'protocol': W.PROTOCOL, 'token': token, 'codecs': W.codecs()})
    return ch, ch.recv()[0]


def test_the_daemon_checks_its_token_and_its_allow_list(tmp_path, daemons):
    a = daemons.start('node-a')
    port = a['slots'][0]['port']
    ch, msg = _hello(port, 'not-the-token')
    assert msg['t'] == 'error' and 'wrong token' in msg['why']
    ch.close()
    tok = W.read_token(0, a['token_dir'])
    ch, msg = _hello(port, tok)
    assert msg['t'] == 'hello' and msg['slot'] == 0 and msg['protocol'] == W.PROTOCOL
    try:
        evil = {'schema': B.SCHEMA, 'kind': 'job', 'worker': 'stack', 'config': {'commands': ['rm -rf ~']},
                'inputs': {}, 'notes': []}
        ch.send({'t': 'submit', 'rid': 'abc.1'}, B.encode(evil))
        m, _ = ch.recv()
        while m['t'] == 'heartbeat':
            m, _ = ch.recv()
        assert m == {'t': 'refused', 'rid': 'abc.1', 'why': "the stack worker is not on this node's allow-list"}
        bad = dict(evil, worker='drive', config={'x': '@input:nope'})
        ch.send({'t': 'submit', 'rid': 'abc.2'}, B.encode(bad))
        m, _ = ch.recv()
        while m['t'] == 'heartbeat':
            m, _ = ch.recv()
        assert m['t'] == 'refused' and 'does not carry' in m['why']
        ch.send({'t': 'submit', 'rid': '../x'}, B.encode(evil))
        m, _ = ch.recv()
        while m['t'] == 'heartbeat':
            m, _ = ch.recv()
        assert m['t'] == 'refused' and m['why'] == 'a bad job id'
    finally:
        ch.close()


def test_artifacts_unpack_only_inside_their_folder(tmp_path):
    import io
    import tarfile
    for name, kind in (('../escape', tarfile.REGTYPE), ('/abs', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w') as tf:
            ti = tarfile.TarInfo(name)
            ti.type = kind
            ti.linkname = '/etc/passwd' if kind == tarfile.SYMTYPE else ''
            tf.addfile(ti, io.BytesIO(b''))
        with pytest.raises(W.ProtocolError):
            W.unpack_tar(buf.getvalue(), str(tmp_path / 'out'))
    data = W.pack_files({'a/b.json': b'{"p": "/node/x/run"}', 'c.bin': b'/node/x/run'})
    names = W.unpack_tar(data, str(tmp_path / 'ok'))
    W.translate_paths(str(tmp_path / 'ok'), names, '/node/x/run', '/here/run')
    assert open(tmp_path / 'ok' / 'a' / 'b.json').read() == '{"p": "/here/run"}'
    assert open(tmp_path / 'ok' / 'c.bin', 'rb').read() == b'/node/x/run'          # binary files untouched
    for codec in W.codecs():
        assert W.decompress(W.compress(b'x' * 1000, codec), codec) == b'x' * 1000
    with pytest.raises(W.ProtocolError):
        W.decompress(W.compress(b'x' * 1000, 'zlib'), 'zlib', max_size=100)


def test_a_bundle_carries_inputs_not_paths(tmp_path):
    xodr = tmp_path / 'map.xodr'
    xodr.write_text('<OpenDRIVE/>')
    cfg = {'out': str(tmp_path / 'run'), 'host': '10.0.0.5', 'port': 2310, 'tm_port': 8310, 'xodr': str(xodr),
           'nested': {'again': str(xodr)}, 'odd_workspace': str(tmp_path), 'seed': 3}
    os.makedirs(tmp_path / 'run')
    json.dump(cfg, open(tmp_path / 'run' / 'drive.json', 'w'))
    spec = JobSpec('drive', 'd', ['python', '-m', 'falcon_autonomy.workers', 'drive', 'x'], run_dir=str(tmp_path / 'run'),
                   meta={'config': str(tmp_path / 'run' / 'drive.json')})
    b = B.make(spec)
    text = json.dumps(b)
    assert str(tmp_path) not in text and '10.0.0.5' not in text           # no path, no address of this machine
    assert b['config']['xodr'] == b['config']['nested']['again'] == '@input:0-map.xodr'
    assert 'odd_workspace' not in b['config'] and 'feature monitor' in b['notes'][0]
    node = tmp_path / 'node' / 'rid'
    os.makedirs(node)
    path = B.materialise(B.decode(B.encode(b)), str(node), '127.0.0.1', 2320, 8311)
    got = json.load(open(path))
    assert got['out'] == str(node) and got['port'] == 2320 and got['tm_port'] == 8311 and got['host'] == '127.0.0.1'
    assert open(got['xodr']).read() == '<OpenDRIVE/>' and got['xodr'].startswith(str(node))


def test_nodes_hold_key_paths_never_keys(tmp_path):
    st = Settings(str(tmp_path / 's.json'))
    key = tmp_path / 'id_ed25519'
    key.write_text('k')
    n = N.add(st, {'name': 'ws1', 'kind': 'ssh', 'host': 'ws1.lan', 'ssh_user': 'jo', 'ssh_key': str(key),
                   'slots': [0, 1], 'pool': 'gpu', 'vram_gb': 16})
    assert n['ssh_port'] == 22 and N.slots(n) == [(0, 2350), (1, 2351)]
    for bad, why in (({'ssh_key': '-----BEGIN OPENSSH PRIVATE KEY-----\nabc'}, 'never the key'),
                     ({'ssh_key': str(tmp_path / 'missing')}, 'does not exist'),
                     ({'host': '-oProxyCommand=x'}, 'host'), ({'password': 'x'}, 'has no password'),
                     ({'slots': [0, 0]}, 'distinct')):
        with pytest.raises(N.NodeError, match=why):
            N.add(st, dict({'name': 'ws2', 'kind': 'ssh', 'host': 'ws2', 'ssh_user': 'jo', 'ssh_key': str(key)}, **bad))
    with pytest.raises(N.NodeError, match='already a node'):
        N.add(st, {'name': 'ws1', 'kind': 'local'})
    assert N.public(N.load(st)['nodes'][0]) == {'name': 'ws1', 'kind': 'ssh', 'pool': 'gpu', 'vram_gb': 16.0,
                                                'slots': 2}
    assert N.parse_placement('cluster:gpu') == ('cluster', 'gpu') and N.parse_placement('local') == ('local', '')
    assert N.remove(st, 'ws1') and not N.load(st)['nodes']


def test_the_cluster_command(tmp_path, daemons, capsys, monkeypatch):
    from falcon_autonomy.cluster.cli import main_cluster
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    a = daemons.start('node-a')
    assert main_cluster(['nodes', 'add', 'node-a', '--kind', 'local', '--slots', '0', '--token-dir',
                         a['token_dir'], '--pool', 'gpu']) == 0
    st = Settings()
    cfg = N.load(st)
    cfg['nodes'][0]['slots'] = a['slots']                          # the test daemon's own port
    N.save(st, cfg)
    capsys.readouterr()
    assert main_cluster(['nodes', 'test', 'node-a']) == 0
    assert 'slot 0: ok' in capsys.readouterr().out
    assert main_cluster(['pools', 'set', 'gpu', '--max-running', '2']) == 0
    assert 'gpu' in capsys.readouterr().out
    assert main_cluster(['pools', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['gpu']['max_running'] == 2
    assert main_cluster(['nodes', 'add', 'x', '--kind', 'ssh', '--host', 'h', '--user', 'u', '--key', '/nope']) == 2
    assert 'does not exist' in capsys.readouterr().err
    p = _project(tmp_path / 'p.falcon.json')
    rc = main_cluster(['dispatch', 'sweep', p.path, '--pool', 'gpu', '--scenario', 'lead_brake', '--name', 'cl',
                       '--samples', '2', '--fake-carla', FAKE])
    assert rc == 0
    rows = [json.loads(x) for x in open(os.path.join(p.workspace, 'studies', 'cl', 'results.jsonl'))]
    assert len(rows) == 2
    assert all(json.load(open(os.path.join(r['run'], 'run.json')))['node']['pool'] == 'gpu' for r in rows)
    assert main_cluster(['nodes', 'remove', 'node-a']) == 0
