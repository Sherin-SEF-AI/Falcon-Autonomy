"""The cluster's backends and its caches, against local daemons and the fakes: a node's hello names the CARLA backends
it has (the UE5 fake on one daemon, the 0.9.16 fake on another) and the scheduler sends each job to a node that has
its backend and map, refusing one no node can run; `falcon cluster gc` and the cluster_cache_gc tool remove only what
nothing refers to — orphaned scratch folders and old CARLA logs on a daemon, Slurm array folders of no queued array,
unreferenced objects of the workspace's content store — after a dry run, never a run folder, the trash, a folder of
another name, a link, a job still held, nor what changed since the plan."""
import hashlib
import json
import os
import time

import pytest

from cluster_helpers import Daemons, wait_for
from conftest import FAKE, ROOT
from falcon_autonomy.cluster import gc as G
from falcon_autonomy.cluster.launcher import RemoteLauncher, close_all
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.jobs import DONE, FAILED, JobQueue, SubprocessLauncher
from falcon_autonomy.core.project import Project
from falcon_autonomy.core.settings import Settings
from falcon_autonomy.sim.service import CarlaService

FAKE_UE5 = os.path.join(ROOT, 'tests', 'fake_carla_ue5')
DAY = 24 * 3600.0


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


def _settings(tmp_path, nodes, name='settings.json'):
    st = Settings(str(tmp_path / name))
    st.set('cluster', {'nodes': nodes, 'pools': {}})
    return st


def _drive(p, out, seconds=0.5, **kw):
    from falcon_autonomy.core.drive import drive_config
    cfg = drive_config(p, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=seconds, seed=1)
    cfg['traffic'] = dict(cfg['traffic'], vehicles=0, walkers=0)
    cfg.update(kw)
    return cfg


def _remote(st, ws, fake):
    q = JobQueue(SubprocessLauncher())
    svc = CarlaService(st, q, ws, fake_carla=fake)
    q.remote = RemoteLauncher.for_settings(st)
    svc.placement = 'cluster'
    return svc, q


def _ended(job, q, timeout=90):
    wait_for(lambda: job.state in ('done', 'failed', 'cancelled'), timeout, q.poll)
    return job


# ============================================================================================ backends
def test_nodes_advertise_backends_and_jobs_go_where_their_backend_is(tmp_path, daemons):
    from falcon_autonomy.cluster.cli import test_node
    old, ue5 = daemons.start('node-old'), daemons.start('node-ue5', fake=FAKE_UE5)
    t_old, t_ue5 = test_node(old), test_node(ue5)
    assert t_old['slots'][0]['backends'] == ['carla-0.9.16'] and t_ue5['slots'][0]['backends'] == ['carla-ue5']
    st = _settings(tmp_path, [old, ue5])
    p = _project(tmp_path / 'p.falcon.json')
    svc5, q5 = _remote(st, p.workspace, FAKE_UE5)
    svc, q = _remote(st, p.workspace, FAKE)
    try:
        sch = q5.remote.scheduler
        wait_for(lambda: all(s.state == 'up' for s in sch.slots), 30)
        caps = {s.key: s.info['capabilities'] for s in sch.slots}
        assert caps['node-ue5/0']['carla-ue5']['maps'] == ['Town10HD_Opt', 'Mine_01']
        assert caps['node-ue5/0']['carla-ue5']['caps']['ros2'] is False and not caps['node-ue5/0']['carla-ue5'][
            'measured']
        assert caps['node-old/0']['carla-0.9.16']['caps']['osm2odr'] is True
        assert {r['slot']: r['backends'] for r in sch.status()} == {'node-old/0': ['carla-0.9.16'],
                                                                   'node-ue5/0': ['carla-ue5']}
        j5 = svc5.submit_drive(_drive(p, tmp_path / 'r5'), title='ue5')
        j9 = svc.submit_drive(_drive(p, tmp_path / 'r9'), title='0.9.16')
        assert j5.spec.meta['backend'] == 'carla-ue5' and j9.spec.meta['backend'] == 'carla-0.9.16'
        for job, qq in ((j5, q5), (j9, q)):
            assert _ended(job, qq).state == DONE, job.log
        assert j5.node == 'node-ue5/0' and j9.node == 'node-old/0'
        r5 = json.load(open(tmp_path / 'r5' / 'run.json'))
        r9 = json.load(open(tmp_path / 'r9' / 'run.json'))
        assert r5['provenance']['backend']['id'] == 'carla-ue5' and r5['node']['name'] == 'node-ue5'
        assert r9['provenance']['backend']['id'] == 'carla-0.9.16' and r9['node']['name'] == 'node-old'
        # a map the UE5 node does not have: refused before it goes anywhere, saying which node lacks what
        jt = svc5.submit_drive(_drive(p, tmp_path / 'r5b', town='Town05'), title='ue5 Town05')
        assert _ended(jt, q5).state == FAILED and jt.exit_code == P.REFUSED
        why = jt.log[-1][1]
        assert 'no node with carla-ue5 can run it' in why and 'node-ue5/0' in why and 'no map Town05' in why
        assert not os.path.exists(tmp_path / 'r5b' / 'run.json')
    finally:
        svc5.shutdown()
        svc.shutdown()
    # a cluster of 0.9.16 nodes only: a UE5 job is refused once the node said what it has
    close_all()
    only_old = _settings(tmp_path, [old], 'only-old.json')
    svc5, q5 = _remote(only_old, p.workspace, FAKE_UE5)
    try:
        job = svc5.submit_drive(_drive(p, tmp_path / 'r5c'), title='ue5 nowhere')
        assert _ended(job, q5).state == FAILED and job.exit_code == P.REFUSED
        assert 'no node has carla-ue5 (node-old/0 has carla-0.9.16)' in job.log[-1][1], job.log
    finally:
        svc5.shutdown()


def test_a_slot_not_yet_heard_from_and_a_configured_list():
    from falcon_autonomy.cluster.scheduler import Scheduler
    from falcon_autonomy.core.jobs import Job, JobSpec, Refused
    nodes = [{'name': 'a', 'kind': 'local', 'pool': 'default', 'slots': [{'slot': 0, 'port': 1}]},
             {'name': 'b', 'kind': 'local', 'pool': 'default', 'slots': [{'slot': 0, 'port': 2}],
              'backends': ['carla-0.9.16']},
             {'name': 'h', 'kind': 'slurm', 'pool': 'batch', 'workdir': '/x', 'backends': ['carla-ue5']}]
    sch = Scheduler({'nodes': nodes, 'pools': {}}, tick_s=3600)
    try:
        job = Job(JobSpec('drive', 'd', ['x'], placement='cluster', meta={'backend': 'carla-ue5'}))
        pool, slots, batch = sch._candidates(job)
        assert [s.key for s in slots] == ['a/0'] and [b.key for b in batch] == ['h/slurm']   # a: not yet heard from
        sch.slots[0].info = {'backends': ['carla-0.9.16'], 'capabilities': {}}
        assert [b.key for b in sch._candidates(job)[2]] == ['h/slurm']
        job.spec.placement = 'cluster:default'
        with pytest.raises(Refused, match=r"no node of pool 'default' has carla-ue5 \(a/0 has carla-0.9.16; b/0 has "
                                          r"carla-0.9.16\)"):
            sch._candidates(job)
        job.spec.meta['backend'] = 'carla-0.9.16'
        assert [s.key for s in sch._candidates(job)[1]] == ['a/0', 'b/0']
    finally:
        sch.close()


# ============================================================================================ gc
def _age(path, seconds):
    """Make a file or a folder's tree look unchanged for this long."""
    t = time.time() - seconds
    for dp, dns, fns in os.walk(path):
        for n in fns:
            os.utime(os.path.join(dp, n), (t, t))
    if os.path.isdir(path):
        for dp, dns, _fns in os.walk(path, topdown=False):
            os.utime(dp, (t, t))
    else:
        os.utime(path, (t, t))


def _scratch(base, name, size=1000, age=2 * DAY):
    d = os.path.join(base, name)
    os.makedirs(os.path.join(d, 'rgb'), exist_ok=True)
    with open(os.path.join(d, 'rgb', '000000.jpg'), 'wb') as fh:
        fh.write(b'x' * size)
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok'}, fh)
    _age(d, age)
    return d


def _tree_digest(root) -> str:
    h = hashlib.sha256()
    for dp, dns, fns in sorted(os.walk(root)):
        dns.sort()
        for f in sorted(fns):
            p = os.path.join(dp, f)
            h.update(os.path.relpath(p, root).encode())
            if not os.path.islink(p):
                h.update(open(p, 'rb').read())
    return h.hexdigest()


def _workspace(tmp_path):
    """A workspace with a content store: objects referenced by a study run's index and by a trashed run's, an
    unreferenced object of its own, and one hard-linked from a run that no index names."""
    pytest.importorskip('pyarrow')
    from test_lineage import make_run
    from falcon_autonomy.core import cas as C
    from falcon_autonomy.core import lineage as L
    ws = str(tmp_path / 'ws')
    study = make_run(os.path.join(ws, 'studies', 's1'), 'r1')
    trashed = make_run(os.path.join(ws, 'trash', '20261101T000000Z-r2'), 'r2', seed=1)
    os.rename(trashed, os.path.join(ws, 'trash', '20261101T000000Z-r2', 'r2'))
    trashed = os.path.join(ws, 'trash', '20261101T000000Z-r2', 'r2')
    loose = make_run(ws, 'loose', steps=1, seed=2)
    store = C.Store(C.store_dir(ws))
    for run in (study, trashed):
        L.index_run(run)
        for r in L.read_index(run)[0]:
            store.put(os.path.join(run, r['frame_path']))
    orphan = store.put_bytes(b'nobody refers to me' * 50)
    linked = store.put(os.path.join(loose, 'rgb', 'cam', '000000.jpg'))     # a hard link of a run's file
    return ws, store, C.parse(orphan)[1], C.parse(linked)[1]


def test_gc_on_a_daemon_node_and_the_workspace_store(tmp_path, daemons, monkeypatch, capsys):
    from falcon_autonomy.cluster import cli
    from falcon_autonomy.core import cas as C
    a = daemons.start('node-a')
    home = daemons.homes['node-a']
    scratch = os.path.join(home, 'cache', 'FalconAutonomy', 'worker-0')
    runs = os.path.join(scratch, 'runs')
    old = _scratch(runs, 'a' * 20 + '.1', size=4000)                     # an orphan of an earlier daemon life
    young = _scratch(runs, 'b' * 20 + '.1', age=600)                      # changed ten minutes ago
    held = _scratch(runs, 'c' * 20 + '.2', age=3 * DAY)                    # a job the client still has
    other = _scratch(runs, 'my-run', age=9 * DAY)                         # not a dispatch id: never Falcon's to remove
    target = _scratch(str(tmp_path), 'elsewhere', age=9 * DAY)
    os.symlink(target, os.path.join(runs, 'd' * 20 + '.1'))              # a link: never followed, never removed
    os.makedirs(os.path.join(scratch, 'logs'), exist_ok=True)
    log = os.path.join(scratch, 'logs', 'carla-2310.log')
    with open(log, 'w') as fh:
        fh.write('server log\n' * 100)
    _age(log, 3 * DAY)
    ws, store, orphan, linked = _workspace(tmp_path)
    monkeypatch.setattr(G, '_now', lambda: time.time() + 2 * DAY)        # the store's objects were put "two days ago"
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))       # the command's own settings and plans
    st = Settings()
    st.set('cluster', {'nodes': [a], 'pools': {}})
    before_ws = {d: _tree_digest(os.path.join(ws, d)) for d in ('studies', 'trash', 'runs')}

    res = G.plan(st, ws, keep={'c' * 20})
    by = {s['where']: s for s in res['sections']}
    node = by['node node-a/0']
    assert {i['name'] for i in node['items']} == {f'runs/{"a" * 20}.1', 'logs/carla-2310.log'}, node
    it = next(i for i in node['items'] if i['name'].startswith('runs/'))
    assert it['bytes'] == 4000 + len(json.dumps({'status': 'ok'})) and it['age_h'] >= 47.9
    assert node['kept_why'] == {'changed recently': 1, 'the client still has its job': 1}
    here = by['this machine']
    assert {i['id'] for i in here['items']} == {f'cas/{orphan}', f'cas/{linked}'}
    frees = {i['id']: i['frees'] for i in here['items']}
    assert frees[f'cas/{orphan}'] == 950 and frees[f'cas/{linked}'] == 0   # a hard link frees nothing yet
    assert here['kept_why']['referenced'] == 16                            # the study's and the trashed run's frames
    assert by['the cloud']['items'] == [] and 'ledger' in by['the cloud']['notes'][0]
    for p in (old, young, held, other, log, target):                       # a dry run removes nothing
        assert os.path.exists(p)
    assert os.path.isfile(store.object_path(f's256:{orphan}'))

    pid = G.save_plan(res)
    late = _scratch(runs, 'e' * 20 + '.1', age=5 * DAY)                    # appeared after the plan: not in it
    done = G.collect(st, ws, '', 24.0, dry_run=False, only=G.load_plan(pid)['items'], keep={'c' * 20})
    assert done['removed'] == 4 and not done['errors']
    assert not os.path.exists(old) and not os.path.exists(log)
    for p in (young, held, other, late, target, os.path.join(runs, 'd' * 20 + '.1')):
        assert os.path.lexists(p), p
    assert not os.path.isfile(store.object_path(f's256:{orphan}'))
    assert os.path.isfile(os.path.join(ws, 'runs', 'loose', 'rgb', 'cam', '000000.jpg'))   # the run keeps its file
    assert {d: _tree_digest(os.path.join(ws, d)) for d in ('studies', 'trash', 'runs')} == before_ws
    assert C.Store(C.store_dir(ws)).stats()['objects'] == 16

    # the command: a dry run with sizes, then one node only; a node that is not there
    assert cli.main_cluster(['gc', '--dry-run', '--workspace', ws]) == 0
    out = capsys.readouterr().out
    assert 'node node-a/0: would remove 2 items, ' in out and f'runs/{"e" * 20}.1' in out   # this command holds no
    assert f'runs/{"c" * 20}.2' in out                                                     # job: c… is not kept
    assert 'this machine: nothing to remove; kept 16 referenced' in out
    assert 'a dry run: nothing was removed' in out and os.path.exists(late)
    assert cli.main_cluster(['gc', '--node', 'node-a', '--older-than', '200']) == 0      # 5 days is 120 h: kept
    assert os.path.exists(late) and 'nothing to remove' in capsys.readouterr().out
    assert cli.main_cluster(['gc', '--node', 'node-a', '--older-than', '100', '--json']) == 0
    got = json.loads(capsys.readouterr().out)
    assert got['removed'] == 1 and not os.path.exists(late) and os.path.exists(other) and os.path.exists(held)
    assert cli.main_cluster(['gc', '--node', 'nosuch']) == 2
    assert "no node 'nosuch'" in capsys.readouterr().err


def test_gc_of_a_slurm_node(tmp_path, monkeypatch):
    b = tmp_path / 'slurm-bin'
    b.mkdir()
    here = os.path.join(ROOT, 'tests', 'fake_slurm', 'fakeslurm.py')
    for n in ('sbatch', 'squeue', 'sacct', 'scancel'):
        os.symlink(here, b / n)
    state = tmp_path / 'slurm-state'
    state.mkdir()
    monkeypatch.setenv('PATH', f'{b}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_SLURM_DIR', str(state))
    wd = tmp_path / 'shared'
    old = _scratch(str(wd), 'falcon-0123456789ab', size=3000)
    young = _scratch(str(wd), 'falcon-ba9876543210', age=60)
    queued = _scratch(str(wd), 'falcon-aaaaaaaaaaaa', age=4 * DAY)         # an array still waiting in Slurm's queue
    mine = _scratch(str(wd), 'falcon-bbbbbbbbbbbb', age=4 * DAY)           # one this process runs
    other = _scratch(str(wd), 'falcon-results', age=9 * DAY)              # not Falcon's name for an array
    (state / 'submitted').write_text(json.dumps({'job': 1000, 'text': '#SBATCH --job-name=falcon-aaaaaaaaaaaa\n',
                                                 'tasks': 1}) + '\n')
    (state / '1000_0.json').write_text('{"pid": 0}')
    st = _settings(tmp_path, [{'name': 'hpc', 'kind': 'slurm', 'workdir': str(wd), 'pool': 'batch'}])
    from falcon_autonomy.cluster.slurm import Backend
    be = Backend(G._Clock(), {'name': 'hpc', 'kind': 'slurm', 'workdir': str(wd)})
    r = be.gc(DAY, dry_run=True, busy=[mine])
    assert [i['name'] for i in r['items']] == ['falcon-0123456789ab'] and r['items'][0]['bytes'] >= 3000
    assert {k['name']: k['why'] for k in r['kept']} == {
        'falcon-aaaaaaaaaaaa': 'its array is in Slurm\'s queue', 'falcon-bbbbbbbbbbbb': 'an array this Falcon runs',
        'falcon-ba9876543210': 'changed 0.0 h ago'}
    monkeypatch.setenv('FAKE_SLURM_SQUEUE_FAIL', '1')                     # no controller: nothing at all goes
    res = G.collect(st, '', 'hpc', 24.0, dry_run=False)
    assert res['removed'] == 0 and os.path.exists(old)
    assert res['sections'][0]['kept_why'] == {'squeue did not answer: what runs is unknown, so nothing is removed': 4}
    monkeypatch.delenv('FAKE_SLURM_SQUEUE_FAIL')
    res = G.collect(st, '', 'hpc', 24.0, dry_run=False)                   # no scheduler here runs "mine" now
    assert res['removed'] == 2 and not os.path.exists(old) and not os.path.exists(mine)
    for p in (young, queued, other):
        assert os.path.exists(p), p


def test_the_tool_removes_only_an_approved_dry_run(tmp_path, daemons, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    from falcon_autonomy.assistant import tool_api as T
    from falcon_autonomy.assistant import tools  # noqa: F401 (registers them)
    from falcon_autonomy.assistant.cli import schema_tokens
    from falcon_autonomy.assistant.tools import _compact
    t = T.REGISTRY['cluster_cache_gc']
    assert t.group == 'cluster' and t.tier == 'act' and t.description == _compact.SHORT['cluster_cache_gc']
    assert schema_tokens()['cluster'] < 3500
    a = daemons.start('node-a')
    runs = os.path.join(daemons.homes['node-a'], 'cache', 'FalconAutonomy', 'worker-0', 'runs')
    old = _scratch(runs, 'f' * 20 + '.1', size=2048)
    st = _settings(tmp_path, [a])
    p = _project(tmp_path / 'p.falcon.json')
    q = JobQueue(SubprocessLauncher(), lock_dir=str(tmp_path / 'locks'))
    ctx = T.ToolContext(project=p, settings=st, queue=q, service=CarlaService(st, q, p.workspace),
                        workspace=p.workspace)
    assert t.approval(ctx, dry_run=True) is None                         # the mode decides a dry run
    gate = t.approval(ctx, dry_run=False, plan='0' * 16)
    assert gate['ask'] is True and gate['allow_all'] is False                # removing: a card in every mode
    assert 'removes nothing' in t.preview(ctx, dry_run=True)
    assert 'plan id of a dry run' in T.call(ctx, 'cluster_cache_gc', {'dry_run': False})['error']
    dry = T.call(ctx, 'cluster_cache_gc', {})
    assert dry['dry_run'] and dry['would_remove'] == 1 and dry['size'] == '2.0 KB', dry
    node = next(s for s in dry['sections'] if s['where'] == 'node node-a/0')
    assert node['largest'][0].startswith(f'runs/{"f" * 20}.1 2.0 KB') and os.path.exists(old)
    pv = t.preview(ctx, dry_run=False, plan=dry['plan'])
    assert f'runs/{"f" * 20}.1' in pv and '2.0 KB' in pv and 'cannot be undone' in pv
    wrong = T.call(ctx, 'cluster_cache_gc', {'dry_run': False, 'plan': dry['plan'], 'older_than_h': 1})
    assert 'other arguments' in wrong['error'] and os.path.exists(old)
    assert 'no dry run' in T.call(ctx, 'cluster_cache_gc', {'dry_run': False, 'plan': '0' * 16})['error']
    assert "no node 'x'" in T.call(ctx, 'cluster_cache_gc', {'node': 'x'})['error']
    gone = T.call(ctx, 'cluster_cache_gc', {'dry_run': False, 'plan': dry['plan']})
    assert gone['removed'] == 1 and gone['freed'] == '2.0 KB' and not os.path.exists(old), gone
    ctx.service.shutdown()


def test_a_put_of_bytes_already_stored_renews_their_grace(tmp_path):
    """A version written while gc runs re-uses objects: their ctime is renewed, so the sweep's grace spares them."""
    from falcon_autonomy.core import cas as C
    store = C.Store(str(tmp_path / 'cas'))
    a = store.put_bytes(b'a label set')
    f = tmp_path / 'frame.jpg'
    f.write_bytes(b'jpeg bytes')
    b = store.put(str(f), how='copy')
    old = {x: os.stat(store.object_path(x)).st_ctime_ns for x in (a, b)}
    time.sleep(0.02)
    assert store.put_bytes(b'a label set') == a and store.put(str(f), how='copy') == b
    assert all(os.stat(store.object_path(x)).st_ctime_ns > old[x] for x in (a, b))
    assert store.gc(set(), dry_run=True, grace_s=60.0)['young'] == 2


@pytest.mark.parametrize('mode', ['ask', 'auto', 'autopilot'])
@pytest.mark.parametrize('decide', ['approve', 'reject'])
def test_removing_asks_in_every_mode(tmp_path, monkeypatch, mode, decide):
    from fake_groq import FAKE_KEY, FakeGroq, text, tools
    from falcon_autonomy.assistant import agent as A
    from falcon_autonomy.assistant import tool_api as T
    from falcon_autonomy.assistant import tools as _tools  # noqa: F401 (registers them)
    from falcon_autonomy.assistant.keystore import KeyStore
    from falcon_autonomy.assistant.service import AssistantService
    from falcon_autonomy.assistant.session import Session
    from falcon_autonomy.core import cas as C
    from falcon_autonomy.core.settings import Secrets
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setattr(G, '_now', lambda: time.time() + 2 * DAY)
    ws = str(tmp_path / 'proj.falcon')
    store = C.Store(C.store_dir(ws))
    orphan = store.object_path(store.put_bytes(b'nobody refers to me'))
    ctx = T.ToolContext(queue=JobQueue(SubprocessLauncher(), lock_dir=str(tmp_path / 'locks')), workspace=ws)
    dry = T.call(ctx, 'cluster_cache_gc', {})
    assert dry['would_remove'] == 1 and os.path.isfile(orphan)

    class Host(A.Host):
        def __init__(self, c):
            super().__init__(c)
            self.cards = []

        def event(self, kind, payload):
            if kind == 'approval':
                self.cards.append(payload)
                payload.decide(decide, {})
    srv = FakeGroq().start()
    try:
        st = Settings(str(tmp_path / 'config' / 'settings.json'))
        ks = KeyStore(Secrets(str(tmp_path / 'config' / 'secrets.json')), use_keyring=False)
        svc = AssistantService(st, ks, workspace_fn=lambda: ws)
        svc.set_cfg(base_url=srv.base_url, enabled=True, tpm_limit=0)
        svc.accept_notice()
        ks.set(FAKE_KEY)
        host = Host(ctx)
        ag = A.Agent(svc, Session(ws), host, mode)
        ag.session.allowed.append('cluster_cache_gc')            # even "approved all of this kind" before: asks
        srv.push(tools(('cluster_cache_gc', {'dry_run': False, 'plan': dry['plan']})), text('Done.'), text('Done.'))
        assert ag.start('Clear the caches of that dry run.')
        assert ag.wait(60)
    finally:
        srv.stop()
    assert [c.tool for c in host.cards] == ['cluster_cache_gc']
    card = host.cards[0]
    assert card.allow_all is False and 'every mode' in card.why and 'cannot be undone' in card.preview
    assert 'cas/objects/' in card.preview and '19 B' in card.preview
    assert os.path.isfile(orphan) == (decide == 'reject')
