"""Cloud bursts against fake RunPod and Lambda servers on 127.0.0.1 (field names pinned here: docs/cloud.md says
which were read from the providers' documentation and which await the real acceptance run). Nothing is launched
without an approved estimate under the caps; termination is retried and confirmed; the watchdog (on a fake clock)
terminates what is past its TTL, silent, done or over a cap; the keys never reach a log."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import FAKE, ROOT
from falcon_autonomy.cluster import bundle as B
from falcon_autonomy.cluster import cloud as C
from falcon_autonomy.cluster import protocol as W
from falcon_autonomy.core.project import Project
from falcon_autonomy.core.settings import Settings

RP_KEY = 'rpa_FAKEKEY0123456789abcdefSECRET'
LB_KEY = 'secret_lambda_FAKEKEY_0123456789abcdef'


# ================================================================================================ fake providers
class Fake:
    def __init__(self, kind: str, key: str):
        self.kind, self.key = kind, key
        self.price = 0.44 if kind == 'runpod' else 1.29
        self.calls: list = []
        self.launches = 0
        self.terminate_fail = 0            # answer this many terminate calls with HTTP 500 first
        self.instances: dict = {}
        self.keys: dict = {}
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, doc=None):
                body = b'' if doc is None else json.dumps(doc).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _do(self, method):
                n = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                with fake.lock:
                    fake.calls.append((method, self.path, body))
                    if self.headers.get('Authorization') != f'Bearer {fake.key}':
                        return self._send(401, {'title': 'Unauthorized', 'status': 401} if fake.kind == 'runpod' else
                                          {'error': {'code': 'global/invalid-api-key', 'message': 'bad key'}})
                    code, doc = (fake.runpod if fake.kind == 'runpod' else fake.lambda_)(method, self.path, body)
                return self._send(code, doc)

            def do_GET(self):
                self._do('GET')

            def do_POST(self):
                self._do('POST')

            def do_DELETE(self):
                self._do('DELETE')
        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.srv.server_address[1]}'
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    # RunPod REST v2 (docs/cloud.md)
    def runpod(self, method, path, body):
        if method == 'GET' and path.startswith('/catalog/gpus'):
            return 200, {'gpus': [{'id': 'NVIDIA RTX A5000', 'name': 'RTX A5000', 'memory': 24, 'secure': True,
                                   'community': True, 'price': {'secure': self.price, 'community': 0.16,
                                                                'serverless': None}}]}
        if method == 'POST' and path == '/pods':
            self.launches += 1
            pid = f'pod{self.launches}'
            self.instances[pid] = {'id': pid, 'name': body['name'], 'status': 'PROVISIONING', 'gets': 0}
            return 201, {'id': pid, 'name': body['name'], 'status': 'PROVISIONING', 'cost': self.price}
        if path.startswith('/pods/'):
            pid = path.split('/')[2]
            p = self.instances.get(pid)
            if method == 'DELETE':
                if self.terminate_fail > 0:
                    self.terminate_fail -= 1
                    return 500, {'title': 'Internal Server Error', 'status': 500, 'detail': 'try again'}
                if p is not None:
                    p['status'] = 'TERMINATED'
                return 204, None
            if p is None:
                return 404, {'title': 'Not Found', 'status': 404}
            p['gets'] += 1
            if p['status'] == 'PROVISIONING' and p['gets'] > 1:
                p['status'] = 'RUNNING'
            ssh = {'direct': {'host': '127.0.0.1', 'port': 2222}} if p['status'] == 'RUNNING' else {'direct': None}
            return 200, {'id': pid, 'name': p['name'], 'status': p['status'], 'ssh': ssh, 'cost': self.price}
        if method == 'GET' and path.startswith('/pods'):
            return 200, {'pods': [{'id': i, 'name': p['name'], 'status': p['status']} for i, p in self.instances.items()],
                         'pagination': {'nextCursor': None, 'hasNextPage': False}}
        return 404, {'title': 'Not Found', 'status': 404}

    # Lambda Cloud API v1 (docs/cloud.md)
    def lambda_(self, method, path, body):
        if method == 'GET' and path == '/instance-types':
            return 200, {'data': {'gpu_1x_a10': {'instance_type': {
                'name': 'gpu_1x_a10', 'description': '1x A10 (24 GB PCIe)', 'gpu_description': 'A10 (24 GB PCIe)',
                'price_cents_per_hour': int(round(self.price * 100)), 'specs': {'vcpus': 30, 'memory_gib': 200,
                                                                                  'storage_gib': 1400, 'gpus': 1}},
                'regions_with_capacity_available': [{'name': 'us-east-1', 'description': 'Virginia'}]}}}
        if method == 'POST' and path == '/ssh-keys':
            kid = f'key{len(self.keys) + 1}'
            self.keys[kid] = body
            return 200, {'data': {'id': kid, 'name': body['name'], 'public_key': body['public_key']}}
        if method == 'DELETE' and path.startswith('/ssh-keys/'):
            self.keys.pop(path.split('/')[2], None)
            return 200, {'data': {}}
        if method == 'POST' and path == '/instance-operations/launch':
            self.launches += 1
            iid = f'i-{self.launches}'
            self.instances[iid] = {'id': iid, 'name': body['name'], 'status': 'booting', 'gets': 0}
            return 200, {'data': {'instance_ids': [iid]}}
        if method == 'POST' and path == '/instance-operations/terminate':
            if self.terminate_fail > 0:
                self.terminate_fail -= 1
                return 500, {'error': {'code': 'global/unknown', 'message': 'try again'}}
            for i in body['instance_ids']:
                if i in self.instances:
                    self.instances[i]['status'] = 'terminated'
            return 200, {'data': {'terminated_instances': [self.instances.get(i, {'id': i})
                                                           for i in body['instance_ids']]}}
        if method == 'GET' and path.startswith('/instances/'):
            p = self.instances.get(path.split('/')[2])
            if p is None:
                return 404, {'error': {'code': 'global/not-found', 'message': 'no such instance'}}
            p['gets'] += 1
            if p['status'] == 'booting' and p['gets'] > 1:
                p['status'] = 'active'
            return 200, {'data': {'id': p['id'], 'name': p['name'], 'status': p['status'], 'ip': '127.0.0.1'}}
        if method == 'GET' and path.startswith('/instances'):
            return 200, {'data': [{'id': i, 'name': p['name'], 'status': p['status']}
                                  for i, p in self.instances.items()], 'page_token': None}
        return 404, {'error': {'code': 'global/not-found', 'message': path}}

    def terminate_calls(self) -> int:
        return sum(1 for m, p, _b in self.calls if (m == 'DELETE' and p.startswith('/pods/')) or
                   p == '/instance-operations/terminate')

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    """A throw-away config folder with the two keys in its secrets file (never the real keyring), both fakes."""
    from falcon_autonomy.assistant import keystore
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setattr(keystore, '_keyring', lambda: None)
    keystore.KeyStore(name='runpod_api_key').set(RP_KEY)
    keystore.KeyStore(name='lambda_api_key').set(LB_KEY)
    rp, lb = Fake('runpod', RP_KEY), Fake('lambda', LB_KEY)
    st = Settings()
    st.set('cloud', {'provider': 'runpod', 'max_usd_per_burst': 10.0, 'max_usd_month': 50.0, 'ttl_factor': 1.5,
                     'image': 'registry.example/falcon-worker:test', 'runpod_base_url': rp.url,
                     'lambda_base_url': lb.url})
    p = Project.new('Cloud')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(tmp_path / 'cloud.falcon.json'))
    timer_calls = []
    yield {'st': st, 'rp': rp, 'lb': lb, 'project': p, 'timer': lambda a: timer_calls.append(a) or True,
           'timer_calls': timer_calls}
    rp.close()
    lb.close()
    C._EXTRA_SECRETS.clear()


def _estimate(cloud, provider='runpod', gpu='NVIDIA RTX A5000', kind='sweep', **opts):
    import argparse
    from falcon_autonomy.cluster.cloudcli import build_estimate
    base = dict(project=cloud['project'].path, kind=kind, scenario='lead_brake', name='', world='', rig='',
                provider=provider, gpu=gpu, image='', render=False, list_gpus=False, json=False, stage_size=None,
                final=None, rho=None, defensive=None, max_stages=None, batch=None, threshold=None, seed=None,
                search_seed=None, grid=None, samples=3, budget=None, fps=None, judge=None, method=None, count=None,
                no_feasibility=False)
    base.update(opts)
    return build_estimate(argparse.Namespace(**base), cloud['st'])


def local_runner(fake_carla=FAKE):
    """The study run as the machine would run it (`falcon-worker-daemon oneshot` on the bundle), here."""
    def run(st, bundle_bytes, on_line, key_path):
        d = os.path.join(C.burst_dir(st['burst']), 'machine')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'bundle'), 'wb') as fh:
            fh.write(bundle_bytes)
        env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
        env.update(PYTHONPATH=ROOT, XDG_CONFIG_HOME=os.path.join(d, 'config'))
        p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.cluster.daemon', 'oneshot', '--bundle',
                              os.path.join(d, 'bundle'), '--out', os.path.join(d, 'out'), '--fake-carla', fake_carla],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            on_line(line.rstrip('\n'))
        code = p.wait()
        return code, W.pack_dir(os.path.join(d, 'out', 'summary'), skip=())
    return run


# ================================================================================================ estimates
def test_an_estimate_is_priced_from_the_provider_and_names_its_id(cloud):
    est = _estimate(cloud)
    assert est['provider'] == 'runpod' and est['usd_per_hour'] == 0.44 and est['spot'] is False
    assert est['plan']['evaluations'] == {'low': 3, 'expected': 3, 'high': 3}
    assert est['usd']['low'] <= est['usd']['expected'] <= est['usd']['high']
    assert est['ttl_h'] >= est['hours']['expected'] and est['max_usd'] == round(est['ttl_h'] * 0.44, 2)
    assert est['bundle']['no_render'] and est['bundle']['project']['rigs'][0]['sensors'] == []
    assert len(B.encode(est['bundle'])) < B.CLOUD_MAX and est['cap_check'] == ''
    assert est == C.load_estimate(est['estimate_id'])
    assert est['estimate_id'] == _estimate(cloud)['estimate_id']           # the same plan and price: the same id
    cloud['rp'].price = 0.45
    assert _estimate(cloud)['estimate_id'] != est['estimate_id']           # a new price: another id
    fr = _estimate(cloud, kind='failrate', stage_size=20, final=40, max_stages=4, samples=None)
    assert fr['plan']['evaluations'] == {'low': 60, 'expected': 100, 'high': 120}
    lb = _estimate(cloud, provider='lambda', gpu='gpu_1x_a10')
    assert lb['usd_per_hour'] == 1.29 and lb['provider'] == 'lambda'
    assert cloud['rp'].launches == cloud['lb'].launches == 0                # pricing never launches anything


def test_what_a_cloud_bundle_refuses(cloud, tmp_path):
    p = cloud['project']
    scn = {'id': 'incident-7', 'name': 'From the dashcam', 'source': {'format': 'dashcam'}, 'duration_s': 5,
           'params': {}}
    with pytest.raises(B.BundleError, match='private'):
        B.make_study(p, 'sweep', scn, 'world-town10', 'rig-default', 'x', {'samples': 2})
    with pytest.raises(B.BundleError, match='does not take'):
        B.make_study(p, 'sweep', {'id': 'lead_brake'}, 'world-town10', 'rig-default', 'x', {'model': 'm'})
    with pytest.raises(B.BundleError, match='does not run as a cloud study'):
        B.make_study(p, 'train', {'id': 'lead_brake'}, 'world-town10', 'rig-default', 'x', {})
    p.doc['worlds'].append({'id': 'w-osm', 'name': 'Bengaluru', 'kind': 'osm', 'xodr': 'maps/b.xodr'})
    with pytest.raises(B.BundleError, match='imported map'):
        B.make_study(p, 'sweep', {'id': 'lead_brake'}, 'w-osm', 'rig-default', 'x', {})


# ================================================================================================ approval and caps
def test_nothing_launches_without_an_approved_estimate_under_the_caps(cloud):
    rp = cloud['rp']
    est = _estimate(cloud)
    with pytest.raises(C.CloudError, match='an estimate id is 16 hex'):
        C.burst('not-an-id', cloud['project'].workspace, cloud['st'], timer=cloud['timer'])
    with pytest.raises(C.CloudError, match='no estimate 0123456789abcdef'):
        C.burst('0123456789abcdef', cloud['project'].workspace, cloud['st'], timer=cloud['timer'])
    rp.price = 0.52                                                          # the price moved since the approval
    with pytest.raises(C.CloudError, match='the price or the plan changed .* nothing was launched'):
        C.burst(est['estimate_id'], cloud['project'].workspace, cloud['st'], timer=cloud['timer'])
    rp.price = 0.44
    st = cloud['st']
    cs = dict(st.get('cloud'), max_usd_per_burst=0.01)
    st.set('cloud', cs)
    with pytest.raises(C.CloudError, match=r'refused by the caps: .* over the cap of \$0.01 per burst'):
        C.burst(est['estimate_id'], cloud['project'].workspace, st, timer=cloud['timer'])
    st.set('cloud', dict(cs, max_usd_per_burst=10.0, max_usd_month=50.0))
    led = C.Ledger()
    for i in range(5):                                                      # bursts earlier this month: $49.90 spent
        led.append('intent', f'old{i}', max_usd=10.0)
        led.append('terminated', f'old{i}', cost_usd=9.98)
    with pytest.raises(C.CloudError, match='over the monthly cap of \\$50.00'):
        C.burst(est['estimate_id'], cloud['project'].workspace, st, timer=cloud['timer'])
    old = json.load(open(os.path.join(C.cloud_dir(), 'estimates', f'{est["estimate_id"]}.json')))
    old['created'] -= 2 * 86400
    json.dump(old, open(os.path.join(C.cloud_dir(), 'estimates', f'{est["estimate_id"]}.json'), 'w'))
    with pytest.raises(C.CloudError, match='more than a day old'):
        C.burst(est['estimate_id'], cloud['project'].workspace, st, timer=cloud['timer'])
    assert rp.launches == 0 and not [c for c in rp.calls if c[0] == 'POST']     # zero launch calls throughout
    assert not cloud['timer_calls']


# ================================================================================================ a burst
def test_a_burst_from_launch_to_confirmed_termination(cloud, capsys):
    rp = cloud['rp']
    rp.terminate_fail = 2                                  # terminate answers 500 twice, then succeeds
    est = _estimate(cloud)
    sleeps = []
    out = C.burst(est['estimate_id'], cloud['project'].workspace, cloud['st'], runner=local_runner(),
                  timer=cloud['timer'], sleep=sleeps.append, poll_s=0.0)
    assert out['state'] == 'terminated' and out['study_exit'] == 0 and out['files'] > 0, out
    assert rp.launches == 1 and rp.instances['pod1']['status'] == 'TERMINATED'
    assert rp.terminate_calls() == 3 and sleeps[-2:] == [2.0, 4.0]          # retried with backoff, then confirmed
    launch = next(b for m, p, b in rp.calls if m == 'POST' and p == '/pods')
    assert launch['env']['PUBLIC_KEY'].startswith('ssh-ed25519 ') and launch['ports'] == ['22/tcp']
    assert launch['name'].startswith(f'falcon-burst-{out["burst"]}-') and launch['gpu'] == {'id': 'NVIDIA RTX A5000',
                                                                                           'count': 1}
    key = os.path.join(C.burst_dir(out['burst']), 'id_ed25519')
    assert oct(os.stat(key).st_mode & 0o777) == '0o600'
    events = [r['event'] for r in C.Ledger().records() if r['burst'] == out['burst']]
    assert events == ['intent', 'launched', 'terminate_failed', 'terminate_failed', 'terminated']
    rows = [json.loads(x) for x in open(os.path.join(cloud['project'].workspace, 'studies', est['bundle']['name'],
                                                     'results.jsonl'))]
    assert len(rows) == 3 and all(r['status'] == 'ok' for r in rows)
    assert all(r['run'].startswith(cloud['project'].workspace) for r in rows)   # the machine's paths made this one's
    assert os.path.isfile(os.path.join(rows[0]['run'], 'run.json'))
    assert cloud['timer_calls'] == ['start', 'stop']
    assert C.alert() is None


def test_lambda_burst_and_a_terminate_that_keeps_failing(cloud):
    lb = cloud['lb']
    est = _estimate(cloud, provider='lambda', gpu='gpu_1x_a10')
    lb.terminate_fail = 100
    notes = []
    out = C.burst(est['estimate_id'], cloud['project'].workspace, cloud['st'], runner=local_runner(),
                  timer=cloud['timer'], sleep=lambda s: None, poll_s=0.0,
                  notify=lambda t, b: notes.append((t, b)))
    assert out['state'] == 'terminate_failed' and lb.terminate_calls() == C.TERMINATE_TRIES
    al = C.alert()
    assert al and al['bursts'][0]['burst'] == out['burst'] and 'could not be terminated' in al['message']
    assert notes and 'NOT terminated' in notes[0][0]
    assert cloud['timer_calls'] == ['start']                                # the watchdog stays on for it
    assert lb.keys and next(iter(lb.keys.values()))['public_key'].startswith('ssh-ed25519 ')
    lb.terminate_fail = 0                                                   # the provider recovers: the next tick
    done = C.watchdog_tick(cloud['st'], providers={'lambda': C.provider('lambda', settings=cloud['st'])},
                           sleep=lambda s: None, timer=cloud['timer'])
    assert done == [(out['burst'], ['its terminate failed before'], True)]
    assert C.load_state(out['burst'])['state'] == 'terminated' and C.alert() is None
    assert not lb.keys                                                      # the burst's key is removed after
    assert cloud['timer_calls'][-1] == 'stop'


# ================================================================================================ the watchdog
def _active(cloud, **kw):
    """A burst in progress as its driver left it (launched through the fake, so terminating it is real)."""
    prov = C.provider('runpod', settings=cloud['st'])
    now = 1_800_000_000.0
    bid = f'b{len(C.states())}'
    launched = prov.launch(C.instance_name(bid, now + 3600), 'NVIDIA RTX A5000', 'ssh-ed25519 AAAA x', 'img')
    st = dict({'burst': bid, 'provider': 'runpod', 'instance_id': launched['id'], 'state': 'running',
               'launched_at': now, 'last_seen': now, 'deadline': now + 3600, 'usd_per_hour': 0.44, 'max_usd': 0.66},
              **kw)
    C.save_state(st)
    C.Ledger().append('intent', bid, now, max_usd=st['max_usd'])
    C.Ledger().append('launched', bid, now, instance_id=launched['id'])
    return st, now, prov


def test_the_watchdog_on_a_fake_clock(cloud):
    a, now, prov = _active(cloud, last_seen=1_800_000_000.0 + 3500)       # heard from: only its TTL ends it
    provs = {'runpod': prov}
    tick = lambda t: C.watchdog_tick(cloud['st'], now=t, providers=provs, sleep=lambda s: None,  # noqa: E731
                                     timer=cloud['timer'])
    assert tick(now + 60) == [] and C.load_state(a['burst'])['state'] == 'running'
    assert tick(now + 3601)[0][1] == ['past its TTL'] and C.load_state(a['burst'])['state'] == 'terminated'
    b, now, _p = _active(cloud, deadline=now + 7200, max_usd=10.0)
    assert tick(now + 601)[0][1] == ['silent for 10 min']
    c, now, _p = _active(cloud, deadline=now + 7200, max_usd=10.0, last_seen=now + 1e6, state='done')
    assert tick(now + 30)[0][1] == ['its study is done']
    d, now, _p = _active(cloud, deadline=now + 7200, max_usd=0.05, last_seen=now + 1e6)
    assert 'over its approved cost' in tick(now + 600)[0][1][0]
    e, now, _p = _active(cloud, deadline=now + 1e6, max_usd=1e6, last_seen=now + 1e6)
    C.Ledger().append('terminated', 'spent', now, cost_usd=49.9)
    assert tick(now + 3600)[0][1] == ['over the monthly cap']
    for s in (a, b, c, d, e):
        st = C.load_state(s['burst'])
        assert st['state'] == 'terminated' and st['terminated_by'] == 'watchdog'
        assert cloud['rp'].instances[st['instance_id']]['status'] == 'TERMINATED'
    assert cloud['timer_calls'][-1] == 'stop'                    # nothing left to watch: its timer stops


def test_the_watchdog_timer_is_a_systemd_user_timer(cloud, monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0 if 'is-active' not in cmd else 3, stdout='inactive\n', stderr='')
    monkeypatch.setattr(C.subprocess, 'run', run)
    monkeypatch.setattr(C.shutil, 'which', lambda n: f'/usr/bin/{n}')
    assert C.systemd_timer('start')
    cmd = calls[-1]
    assert cmd[:3] == ['/usr/bin/systemd-run', '--user', f'--unit={C.TIMER_UNIT}']
    assert '--on-unit-active=60' in cmd and cmd[-2:] == ['falcon_autonomy.cluster.cloudcli', 'watchdog']
    assert any(x.startswith('--setenv=XDG_CONFIG_HOME=') for x in cmd)
    assert C.systemd_timer('stop') and calls[-1] == ['systemctl', '--user', 'stop', f'{C.TIMER_UNIT}.timer']


def test_orphans_never_cut_short_another_machine_s_burst(cloud):
    a, now, prov = _active(cloud)
    C.save_state(dict(C.load_state(a['burst']), state='terminated'))         # finished here, still running there
    other = prov.launch(C.instance_name('elsewhere1', now + 3600), 'NVIDIA RTX A5000', 'k', 'img')
    stale = prov.launch(C.instance_name('elsewhere2', now - 7200), 'NVIDIA RTX A5000', 'k', 'img')
    rows = {r['instance_id']: r for r in C.orphans(cloud['st'], {'runpod': prov}, now=now + 60)}
    assert rows[a['instance_id']]['orphan'] and rows[stale['id']]['orphan'] and not rows[other['id']]['orphan']
    C.orphans(cloud['st'], {'runpod': prov}, now=now + 60, terminate_them=True, sleep=lambda s: None)
    status = {i: p['status'] for i, p in cloud['rp'].instances.items()}
    assert status[a['instance_id']] == status[stale['id']] == 'TERMINATED' and status[other['id']] == 'PROVISIONING'
    assert any(r['event'] == 'orphan' and r['burst'] == 'elsewhere2' for r in C.Ledger().records())


# ================================================================================================ keys
def test_the_keys_never_reach_a_log(cloud, capsys, tmp_path):
    cloud['rp'].terminate_fail = 1
    est = _estimate(cloud)
    out = C.burst(est['estimate_id'], cloud['project'].workspace, cloud['st'], runner=local_runner(),
                  timer=cloud['timer'], sleep=lambda s: None, poll_s=0.0)
    said = C.scrub(f'failed with {RP_KEY} and {LB_KEY}')        # a message that would carry the keys
    assert RP_KEY not in said and LB_KEY not in said and said.count('[secret]') == 2
    printed = capsys.readouterr()
    assert out['state'] == 'terminated'
    for text in (printed.out, printed.err):
        assert RP_KEY not in text and LB_KEY not in text
    roots = [os.path.dirname(C.cloud_dir()), cloud['project'].workspace]
    for root in roots:
        for d, _dirs, files in os.walk(root):
            for f in files:
                p = os.path.join(d, f)
                if f == 'secrets.json':
                    continue                                       # where the key is kept (0600), the only place
                data = open(p, 'rb').read()
                assert RP_KEY.encode() not in data and LB_KEY.encode() not in data, p
    secrets = os.path.join(os.path.dirname(C.cloud_dir()), 'secrets.json')
    assert oct(os.stat(secrets).st_mode & 0o777) == '0o600'


def test_the_cloud_command(cloud, capsys):
    from falcon_autonomy.cluster.cloudcli import main_cloud
    assert main_cloud(['estimate', '--list-gpus', '--provider', 'runpod']) == 0
    assert 'NVIDIA RTX A5000' in capsys.readouterr().out
    assert main_cloud(['estimate', cloud['project'].path, '--kind', 'sweep', '--scenario', 'lead_brake', '--samples',
                       '3', '--provider', 'runpod', '--gpu', 'NVIDIA RTX A5000', '--json']) == 0
    est = json.loads(capsys.readouterr().out)
    assert est['estimate_id'] and 'bundle' not in est
    assert main_cloud(['estimate', cloud['project'].path, '--scenario', 'lead_brake', '--provider', 'runpod',
                       '--gpu', 'NVIDIA RTX A5000']) == 0
    text = capsys.readouterr().out
    assert 'to run it: falcon cloud burst --approve' in text and 'at most' in text
    cloud['rp'].price = 9.99
    assert main_cloud(['burst', '--approve', est['estimate_id']]) == 2
    assert 'nothing was launched' in capsys.readouterr().err and cloud['rp'].launches == 0
    assert main_cloud(['status']) == 0 and 'spent' in capsys.readouterr().out
    assert main_cloud(['ledger', '--json']) == 0 and json.loads(capsys.readouterr().out) == []
    assert main_cloud(['orphans', '--json']) == 0
    assert main_cloud(['estimate', cloud['project'].path, '--scenario', 'lead_brake', '--provider', 'nope',
                       '--gpu', 'x']) == 2
    assert "no cloud provider 'nope'" in capsys.readouterr().err
