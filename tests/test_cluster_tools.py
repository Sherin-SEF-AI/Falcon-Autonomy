"""The cluster group's assistant tools: what the model sees of the nodes (never an address, user or key), the pools'
quotas, a study placed on a pool (its preview and estimate), the cloud estimate against a fake provider, and the
burst's card — asked in every permission mode, never "approve all", nothing queued when declined."""
import json
import os

import pytest

from fake_groq import FAKE_KEY, FakeGroq, text, tools
from falcon_autonomy.assistant import agent as A
from falcon_autonomy.assistant import tool_api as T
from falcon_autonomy.assistant.keystore import KeyStore
from falcon_autonomy.assistant.service import AssistantService
from falcon_autonomy.assistant.session import Session
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.jobs import JobQueue, Launcher
from falcon_autonomy.core.settings import Secrets, Settings

NAMES = ('cluster_nodes', 'cluster_quotas', 'cluster_dispatch', 'cluster_spot_estimate', 'cluster_spot_burst')


class Hold(Launcher):
    def start(self, job, queue):
        class H:
            def cancel(self_):
                queue.on_exit(job, P.CANCELLED)

            def kill(self_):
                queue.on_exit(job, P.CANCELLED)

            def send(self_, cmd):
                return True
        return H()


class Quick(Hold):
    """Ends each job a moment after it starts (Autopilot waits for the jobs it queued)."""

    def start(self, job, queue):
        import threading
        threading.Timer(0.2, lambda: queue.on_exit(job, P.OK)).start()
        return super().start(job, queue)


def _queue(tmp_path, launcher=None):
    return JobQueue(launcher or Hold(), lock_dir=str(tmp_path / 'locks'),
                    meminfo_fn=lambda: {'available_gb': 64.0, 'total_gb': 64.0})


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    from falcon_autonomy.assistant import tools  # noqa: F401 (registers them)
    from falcon_autonomy.core.project import Project
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    key = tmp_path / 'id_ed25519'
    key.write_text('k')
    st = Settings()
    st.set('cluster', {'nodes': [
        {'name': 'ws1', 'kind': 'ssh', 'host': 'ws1.secret-lab.lan', 'ssh_user': 'jo-private', 'ssh_key': str(key),
         'slots': [0, 1], 'pool': 'gpu', 'vram_gb': 16.0},
        {'name': 'hpc', 'kind': 'slurm', 'host': 'login.hpc.example', 'ssh_user': 'jo-private', 'ssh_key': str(key),
         'workdir': '/scratch/jo', 'max_jobs': 6, 'pool': 'batch'}],
        'pools': {'gpu': {'max_running': 2, 'max_per_day': 100}}})
    from falcon_autonomy.sim.service import CarlaService
    p = Project.new('Tools')
    p.save(str(tmp_path / 'tools.falcon.json'))
    q = _queue(tmp_path)
    return T.ToolContext(project=p, settings=st, queue=q, service=CarlaService(st, q, p.workspace),
                         workspace=p.workspace)


def test_the_group_s_names_size_and_descriptions():
    from falcon_autonomy.assistant.cli import schema_tokens
    from falcon_autonomy.assistant.tools import _compact
    for n in NAMES:
        t = T.REGISTRY[n]
        assert t.group == 'cluster' and n.startswith('cluster_') and t.description == _compact.SHORT[n]
    assert T.REGISTRY['cluster_dispatch'].tier == 'act' and T.REGISTRY['cluster_spot_burst'].tier == 'act'
    assert T.REGISTRY['cluster_spot_burst'].approval is not None
    toks = schema_tokens()
    assert toks['cluster'] < 3500 and toks['all'] < 18000, toks


def test_nodes_and_quotas_never_show_an_address_user_or_key(ctx):
    nodes = T.call(ctx, 'cluster_nodes', {})
    text_ = json.dumps(nodes)
    for secret in ('secret-lab', 'jo-private', 'id_ed25519', 'login.hpc', '/scratch'):
        assert secret not in text_
    assert {n['name']: n['slots'] for n in nodes['nodes']} == {'ws1': 2, 'hpc': 6}
    assert nodes['placements'] == ['local', 'cluster', 'cluster:batch', 'cluster:gpu']
    q = T.call(ctx, 'cluster_quotas', {})
    assert q['pools']['gpu']['max_running'] == 2 and q['pools']['gpu']['slots'] == 2
    assert q['pools']['batch']['slots'] == 6 and q['session_remote_jobs'] == 0


def test_dispatch_previews_estimates_and_queues_the_study_on_a_pool(ctx):
    t = T.REGISTRY['cluster_dispatch']
    args = {'kind': 'sweep', 'scenario': 'lead_brake', 'pool': 'gpu', 'samples': 8}
    pv = t.preview(ctx, **args)
    assert pv.startswith('falcon sweep ') and pv.endswith('--on cluster:gpu')
    est = t.estimate(ctx, **args)
    one = t.estimate(ctx, **dict(args, pool='batch'))
    assert est['seconds'] > 0 and 'over 2 slots' in est['what'] and 'over 6 slots' in one['what']
    out = T.call(ctx, 'cluster_dispatch', args)
    job = ctx.queue.get(out['job'])
    assert job.spec.argv[-2:] == ['--on', 'cluster:gpu'] and 'on cluster:gpu' in job.spec.title
    fr = T.call(ctx, 'cluster_dispatch', {'kind': 'failrate', 'scenario': 'lead_brake', 'budget': 60})
    argv = ctx.queue.get(fr['job']).spec.argv
    assert 'failrate' in argv and argv[-2:] == ['--on', 'cluster']
    assert 'no pool' in T.call(ctx, 'cluster_dispatch', dict(args, pool='cpu'))['error']
    ctx.settings.set('cluster', {'nodes': []})
    assert 'no cluster is set up' in T.call(ctx, 'cluster_dispatch', args)['error']


def test_spot_estimate_against_the_fake_provider(ctx, monkeypatch):
    from falcon_autonomy.assistant import keystore
    from test_cloud import RP_KEY, Fake
    monkeypatch.setattr(keystore, '_keyring', lambda: None)
    keystore.KeyStore(name='runpod_api_key').set(RP_KEY)
    rp = Fake('runpod', RP_KEY)
    try:
        ctx.settings.set('cloud', {'provider': 'runpod', 'max_usd_per_burst': 10.0, 'max_usd_month': 50.0,
                                   'ttl_factor': 1.5, 'image': 'img:test', 'runpod_base_url': rp.url})
        out = T.call(ctx, 'cluster_spot_estimate', {'kind': 'sweep', 'scenario': 'lead_brake',
                                                    'gpu': 'NVIDIA RTX A5000', 'samples': 4})
        assert len(out['estimate_id']) == 16 and out['usd_per_hour'] == 0.44 and out['on_demand']
        assert out['evaluations']['expected'] == 4 and out['no_render'] and out['caps_refuse'] is None
        assert RP_KEY not in json.dumps(out) and rp.launches == 0
        bad = T.call(ctx, 'cluster_spot_estimate', {'kind': 'sweep', 'scenario': 'lead_brake', 'gpu': 'H100'})
        assert "has no GPU 'H100'" in bad['error']
        t = T.REGISTRY['cluster_spot_burst']
        pv = t.preview(ctx, estimate_id=out['estimate_id'])
        assert 'Rent a NVIDIA RTX A5000 on runpod' in pv and 'at most' in pv and 'changed price launches nothing' in pv
        assert t.estimate(ctx, estimate_id=out['estimate_id'])['seconds'] > 0
        assert t.approval(ctx, estimate_id=out['estimate_id'])['ask'] is True
        assert t.approval(ctx, estimate_id=out['estimate_id'])['allow_all'] is False
        assert 'invalid arguments' in T.call(ctx, 'cluster_spot_burst', {'estimate_id': 'xyz'})['error']
    finally:
        rp.close()


# ------------------------------------------------------------------------------------------------ the burst's card
def _fake_estimate(eid='0123456789abcdef', cap=''):
    from falcon_autonomy.cluster import cloud as CL
    est = {'estimate_id': eid, 'provider': 'runpod', 'gpu': 'NVIDIA RTX A5000', 'usd_per_hour': 0.44,
           'usd': {'low': 0.2, 'expected': 0.3, 'high': 0.4}, 'hours': {'low': 0.4, 'expected': 0.6, 'high': 0.8},
           'ttl_h': 0.9, 'max_usd': 0.4, 'caps': {'max_usd_per_burst': 10.0, 'max_usd_month': 50.0},
           'cap_check': cap, 'plan': {'command': 'sweep', 'evaluations': {'low': 3, 'expected': 3, 'high': 3},
                                      'seconds_per_eval': 60.0, 'source': 'a prior'},
           'bundle': {'name': 'cloud-sweep-lead_brake', 'no_render': True}, 'created': 0}
    os.makedirs(os.path.join(CL.cloud_dir(), 'estimates'), exist_ok=True)
    json.dump(est, open(os.path.join(CL.cloud_dir(), 'estimates', f'{eid}.json'), 'w'))
    return est


class Host(A.Host):
    def __init__(self, ctx, decide):
        super().__init__(ctx)
        self.cards, self.decide = [], decide

    def event(self, kind, payload):
        if kind == 'approval':
            self.cards.append(payload)
            payload.decide(self.decide, {})


@pytest.fixture
def srv():
    s = FakeGroq().start()
    yield s
    s.stop()


@pytest.mark.parametrize('mode', ['ask', 'auto', 'autopilot'])
@pytest.mark.parametrize('decide', ['approve', 'reject'])
def test_a_burst_asks_in_every_mode(tmp_path, srv, monkeypatch, mode, decide):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    _fake_estimate()
    st = Settings(str(tmp_path / 'config' / 'settings.json'))
    ks = KeyStore(Secrets(str(tmp_path / 'config' / 'secrets.json')), use_keyring=False)
    ws = str(tmp_path / 'proj.falcon')
    os.makedirs(ws, exist_ok=True)
    svc = AssistantService(st, ks, workspace_fn=lambda: ws)
    svc.set_cfg(base_url=srv.base_url, enabled=True, tpm_limit=0)
    svc.accept_notice()
    ks.set(FAKE_KEY)
    q = _queue(tmp_path, Quick())
    host = Host(T.ToolContext(queue=q, workspace=ws), decide)
    ag = A.Agent(svc, Session(ws), host, mode)
    ag.session.allowed.append('cluster_spot_burst')                 # even "approved all of this kind" before: asks
    srv.push(tools(('cluster_spot_burst', {'estimate_id': '0123456789abcdef'})), text('Done.'), text('Done.'))
    assert ag.start('Run the cloud estimate 0123456789abcdef.')
    assert ag.wait(60)
    assert [c.tool for c in host.cards] == ['cluster_spot_burst']
    card = host.cards[0]
    assert card.allow_all is False and 'every mode' in card.why and 'at most' in card.preview
    jobs = [j for j in q.jobs if j.spec.kind == 'cloud_burst']
    if decide == 'approve':
        assert len(jobs) == 1 and [str(a) for a in jobs[0].spec.argv][-4:] == ['cloud', 'burst', '--approve',
                                                                               '0123456789abcdef']
    else:
        assert jobs == []


def test_a_burst_the_caps_refuse_is_never_queued(ctx):
    _fake_estimate('fedcba9876543210', cap='its worst case is over the cap')
    out = T.call(ctx, 'cluster_spot_burst', {'estimate_id': 'fedcba9876543210'})
    assert 'refused by the caps' in out['error'] and not ctx.queue.jobs
