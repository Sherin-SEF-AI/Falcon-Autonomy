"""The shared team workspace (core.team): identity, the lock on a shared folder (exclusive creation by link and by
O_EXCL, the holder named, stale locks taken over by heartbeat age and by a dead holder on this host, a lock lost to a
takeover, several processes counting under it), comments, run owners, the triage queue (numbers claimed by mkdir under
contention, transitions, concurrent events folded in time order), the project saved under the lock and never over
another member's save, the catalogue kept on this machine, and the commands."""
import json
import multiprocessing
import os
import socket
import threading
import time

import pytest

from falcon_autonomy.core import team as TM

ME = {'user': 'asha', 'host': 'ws-1'}
RAVI = {'user': 'ravi', 'host': 'ws-2'}


def test_identity_from_settings_env_and_login(monkeypatch):
    class St(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)
    monkeypatch.setenv('FALCON_USER', 'env-user')
    assert TM.identity(St(user_name='  Asha K ')) == {'user': 'Asha K', 'host': socket.gethostname()}
    assert TM.identity(St())['user'] == 'env-user'
    monkeypatch.delenv('FALCON_USER')
    assert TM.identity(None)['user']                     # the login name


def test_write_new_is_exclusive_and_falls_back_without_hard_links(tmp_path, monkeypatch):
    p = str(tmp_path / 'a' / 'x.json')
    assert TM.write_new(p, {'n': 1}) is True
    assert TM.write_new(p, {'n': 2}) is False and TM.read_json(p) == {'n': 1}
    assert not [f for f in os.listdir(tmp_path / 'a') if f.endswith('.tmp')]       # no temporary file left
    # an SMB share refuses hard links: O_CREAT|O_EXCL instead, still exclusive
    monkeypatch.setattr(os, 'link', lambda a, b: (_ for _ in ()).throw(PermissionError(1, 'no links here')))
    q = str(tmp_path / 'b.json')
    assert TM.write_new(q, {'n': 3}) is True and TM.read_json(q) == {'n': 3}
    assert TM.write_new(q, {'n': 4}) is False and TM.read_json(q) == {'n': 3}


def test_lock_is_exclusive_names_its_holder_and_is_released(tmp_path):
    path = str(tmp_path / 'proj.falcon.json.lock')
    a = TM.FileLock(path, ME, heartbeat=False)
    b = TM.FileLock(path, RAVI, heartbeat=False)
    a.acquire(timeout=0)
    rec = TM.read_json(path)
    assert rec['user'] == 'asha' and rec['pid'] == os.getpid() and rec['token'] == a.token
    with pytest.raises(TM.LockBusy) as e:
        b.acquire(timeout=0.3)
    assert 'asha' in str(e.value) and 'ws-1' in str(e.value)
    h = b.holder()
    assert h['stale'] is False and h['age_s'] < 5
    a.release()
    assert not os.path.exists(path)
    with b:
        assert b.held()
    assert not os.path.exists(path)


def test_stale_locks_are_taken_over(tmp_path):
    path = str(tmp_path / 'x.lock')
    # a holder on another machine whose heartbeat stopped long ago (by the folder's own clock)
    TM.write_new(path, {'user': 'ravi', 'host': 'elsewhere', 'pid': 1, 'token': 'old'})
    old = time.time() - 600
    os.utime(path, (old, old))
    lk = TM.FileLock(path, ME, stale_s=120, heartbeat=False)
    assert lk.holder()['stale'] is True
    lk.acquire(timeout=0)
    assert TM.read_json(path)['token'] == lk.token
    lk.release()
    # a fresh holder on another machine is not stale
    TM.write_new(path, {'user': 'ravi', 'host': 'elsewhere', 'pid': 1, 'token': 'fresh'})
    with pytest.raises(TM.LockBusy):
        TM.FileLock(path, ME, heartbeat=False).acquire(timeout=0)
    os.unlink(path)
    # a holder on this host whose process is gone: stale at once
    dead = 4_000_000
    while os.path.exists(f'/proc/{dead}'):
        dead += 1
    TM.write_new(path, {'user': 'ravi', 'host': socket.gethostname(), 'pid': dead, 'token': 'gone'})
    lk = TM.FileLock(path, ME, heartbeat=False)
    assert lk.holder()['stale'] is True
    lk.acquire(timeout=0)
    lk.release()


def test_heartbeat_keeps_a_lock_fresh_and_a_taken_lock_is_lost(tmp_path):
    path = str(tmp_path / 'h.lock')
    lk = TM.FileLock(path, ME, stale_s=0.6, heartbeat_s=0.05)
    lk.acquire(timeout=0)
    time.sleep(0.9)                                     # longer than stale_s: the heartbeat kept it fresh
    other = TM.FileLock(path, RAVI, stale_s=0.6, heartbeat=False)
    assert other.holder()['stale'] is False
    with pytest.raises(TM.LockBusy):
        other.acquire(timeout=0)
    lk._stop.set()                                      # the holder stalls (a suspended laptop) …
    lk._thread.join()
    time.sleep(0.8)
    other.acquire(timeout=0)                            # … and its lock is taken over
    assert lk.beat() is False and lk.lost is True
    lk.release()                                        # leaves the new holder's lock alone
    assert other.held()
    other.release()


def _count(args):
    path, n = args
    from falcon_autonomy.core import team
    for _ in range(n):
        with team.FileLock(path + '.lock', {'user': f'p{os.getpid()}', 'host': 'h'}, heartbeat=False).acquire(20):
            with open(path, encoding='utf-8') as fh:
                v = int(fh.read() or 0)
            time.sleep(0.001)
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(str(v + 1))
    return n


def test_processes_counting_under_the_lock_lose_no_update(tmp_path):
    path = str(tmp_path / 'counter')
    with open(path, 'w') as fh:
        fh.write('0')
    with multiprocessing.get_context('fork').Pool(4) as pool:
        assert sum(pool.map(_count, [(path, 15)] * 4)) == 60
    assert open(path).read() == '60'
    assert not os.path.exists(path + '.lock')


def test_comments_owners_and_refs(tmp_path):
    ws = str(tmp_path / 'p.falcon')
    with pytest.raises(TM.TeamError):
        TM.add_comment(ws, 'lunch:today', 'x', ME)
    with pytest.raises(TM.TeamError):
        TM.add_comment(ws, 'run:../../etc', 'x', ME)
    with pytest.raises(TM.TeamError):
        TM.add_comment(ws, 'run:runs/a', '   ', ME)
    a = TM.add_comment(ws, 'run:runs/a', 'brakes 0.4 s late', ME)
    b = TM.add_comment(ws, 'run:runs/a', 'same on seed 3', dict(RAVI, via='assistant'), reply_to=a['id'])
    cs = TM.comments(ws, 'run:runs/a')
    assert [c['text'] for c in cs] == ['brakes 0.4 s late', 'same on seed 3']
    assert cs[1]['via'] == 'assistant' and cs[1]['reply_to'] == a['id'] and b['user'] == 'ravi'
    assert TM.commented(ws) == {'run:runs/a': 2}
    # owners: only in a team workspace, first writer stands
    run = os.path.join(ws, 'runs', 'r1')
    os.makedirs(run)
    assert TM.record_owner(ws, run, ME) is False
    TM.init(ws, 'blr', ME)
    assert TM.is_team(ws) and TM.info(ws)['members'][0]['user'] == 'asha'
    assert TM.record_owner(ws, run, ME, 'drive') is True
    assert TM.record_owner(ws, run, RAVI) is False
    assert TM.owner_of(ws, run)['user'] == 'asha' and TM.owners(ws)['runs/r1']['host'] == 'ws-1'


def test_triage_numbers_transitions_and_concurrent_events(tmp_path):
    ws = str(tmp_path / 'p.falcon')
    got = []

    def make(k):
        got.append(TM.create_item(ws, f'item {k}', ME)['id'])
    ts = [threading.Thread(target=make, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(got, key=lambda s: int(s[2:])) == [f'T-{k}' for k in range(1, 9)]       # 8 distinct numbers
    it = TM.create_item(ws, 'Right turns at night', ME, ['scenario:right_turn', 'run:runs/x'])
    iid = it['id']
    assert it['state'] == 'open' and TM.norm_item('9') == 'T-9' and TM.norm_item('triage:t-9') == 'T-9'
    with pytest.raises(TM.TeamError, match='needs an assignee'):
        TM.set_state(ws, iid, 'investigating', ME)
    with pytest.raises(TM.TeamError, match='not a step'):
        TM.set_state(ws, iid, 'verified', ME)
    it = TM.assign(ws, iid, 'ravi', ME)
    assert it['state'] == 'assigned' and it['assignee'] == 'ravi'
    it = TM.set_state(ws, iid, 'investigating', RAVI)
    # two members mark it fixed at the same moment on two machines: both events are written (checked against the
    # state each saw), and the fold keeps one and the other as ignored
    same = TM.utc_stamp()
    for who in (ME, RAVI):
        TM.write_new(os.path.join(ws, 'team', 'triage', iid, 'events', f'z-{who["user"]}.json'),
                     {'type': 'state', 'to': 'fixed', 'user': who['user'], 'time': same, 'id': f'z-{who["user"]}'})
    it = TM.get_item(ws, iid)
    assert it['state'] == 'fixed' and it['fixed_by'] == 'asha' and it['conflicts'] == 1
    assert not it['history'][-1]['ok'] and 'ignored' in it['history'][-1]['what']
    it = TM.link(ws, iid, ['cluster:C-1234abcd'], ['run:runs/x'], ME)
    assert it['links'] == ['scenario:right_turn', 'cluster:C-1234abcd']
    it = TM.set_state(ws, iid, 'investigating', RAVI, 'the fix does not hold on seed 4')
    assert it['state'] == 'investigating' and it['history'][-1]['note'] == 'the fix does not hold on seed 4'
    assert [i['id'] for i in TM.items(ws, state='investigating')] == [iid]
    assert [i['id'] for i in TM.items(ws, assignee='ravi')] == [iid]
    assert TM.items(ws, link='cluster:C-1234abcd')[0]['id'] == iid
    with pytest.raises(TM.TeamError):
        TM.get_item(ws, 'T-99')


def test_project_saved_under_the_lock_never_over_another_save(tmp_path):
    from falcon_autonomy.core.project import Project
    path = str(tmp_path / 'p.falcon.json')
    Project.new('P').save(path)
    solo = Project.load(path)
    TM.save_project(solo, ME)                        # a solo workspace: a plain save
    TM.init(solo.workspace, 'p', ME)
    a, b = Project.load(path), Project.load(path)
    TM.remember(a)
    TM.remember(b)
    a.doc['name'] = 'by asha'
    TM.save_project(a, ME)
    b.doc['name'] = 'by ravi'
    with pytest.raises(TM.SaveConflict) as e:
        TM.save_project(b, RAVI)
    assert 'asha' in str(e.value) and 'ws-1' in str(e.value)
    assert Project.load(path).doc['name'] == 'by asha'
    TM.save_project(b, RAVI, force=True)
    assert Project.load(path).doc['name'] == 'by ravi'
    TM.save_project(b, RAVI)                          # its own last save is its base now
    with TM.FileLock(path + '.lock', RAVI):           # another member is writing it
        with pytest.raises(TM.LockBusy):
            TM.save_project(a, ME, timeout=0)
    assert not os.path.exists(path + '.lock')


def test_the_catalogue_of_a_team_workspace_stays_on_this_machine(tmp_path):
    from falcon_autonomy.core.catalog import catalog_path
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.paths import cache_dir
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    ws = str(tmp_path / 'p.falcon')
    assert catalog_path(ws) == os.path.join(ws, 'catalog.sqlite')
    run = os.path.join(ws, 'runs', 'r1')
    os.makedirs(run)
    with open(os.path.join(run, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok', 'name': 'r1', 'started': '2026-11-01T00:00:00Z'}, fh)
    TM.init(ws, 'p', ME)
    local = catalog_path(ws)
    assert local.startswith(cache_dir()) and not local.startswith(ws)
    svc = CarlaService(Settings(str(tmp_path / 's.json')), JobQueue(SubprocessLauncher()), ws, fake_carla='x')
    try:
        assert [r['name'] for r in svc.catalog.list('run')] == ['r1']          # rebuilt from the run folders
        assert not os.path.exists(os.path.join(ws, 'catalog.sqlite'))
        svc._owner(run, 'drive')
        assert TM.owner_of(ws, run)['user']
    finally:
        svc.catalog.close()


def test_commands(tmp_path, capsys, monkeypatch):
    from falcon_autonomy.cli import main
    from falcon_autonomy.core.project import Project
    monkeypatch.setenv('FALCON_USER', 'asha')
    path = str(tmp_path / 'c.falcon.json')
    Project.new('C').save(path)
    assert main(['team', 'init', path, '--name', 'blr']) == 0
    assert main(['triage', 'add', path, 'Late braking', '--link', 'scenario:lead_brake', '--assign', 'ravi']) == 0
    assert main(['triage', 'state', path, '1', 'investigating']) == 0
    assert main(['triage', 'state', path, '1', 'verified']) == 2
    assert main(['triage', 'link', path, 'T-1', '--add', 'commit:abc123']) == 0
    assert main(['comment', path, 'triage:T-1', 'reproduced on seed 2']) == 0
    capsys.readouterr()
    assert main(['triage', 'show', path, '1', '--json']) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['state'] == 'investigating' and doc['links'] == ['scenario:lead_brake', 'commit:abc123']
    assert doc['comments'][0]['text'] == 'reproduced on seed 2' and doc['comments'][0]['user'] == 'asha'
    assert main(['triage', 'list', path, '--mine']) == 0 and 'no triage items' in capsys.readouterr().out
    assert main(['team', 'info', path, '--json']) == 0
    assert json.loads(capsys.readouterr().out)['items'] == 1
    assert main(['team', 'lock', path]) == 0 and capsys.readouterr().out.strip() == 'free'
    assert main(['comment', path, 'bogus']) == 2
