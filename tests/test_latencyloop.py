"""Latency in the loop (core.latencyloop; docs/latency-loop.md): latency profiles (measured sources read from their
files, synthetic ones seeded), the delay line by hand (quantisation up to whole steps, hold, coasting, overtaken
results, serial mode, common random numbers), the workers' opt-in — a run without the block and a run with zero
latency are byte-identical — and the latency budget on the fake carla (bisection against every grid point, the
profile's scale, the cases that fail only when late), the command, the tools and the Hardware tab."""
from __future__ import annotations

import json
import math
import os
import time

import pytest

from falcon_autonomy.core import latencyloop as LL
from falcon_autonomy.core import protocol as P

from conftest import FAKE


# ================================================================================================ profiles
def test_profiles_say_where_they_came_from(tmp_path):
    c = LL.profile_synthetic('constant:40')
    assert c['samples_ms'] == [40.0] and not c['measured'] and c['source']['kind'] == 'synthetic'
    a = LL.profile_synthetic('lognormal:median=30,sigma=0.4,spike_p=0.02,spike_ms=250,n=5000', seed=3)
    b = LL.profile_synthetic('lognormal:median=30,sigma=0.4,spike_p=0.02,spike_ms=250,n=5000', seed=3)
    assert a['samples_ms'] == b['samples_ms'] and a['sha256'] == b['sha256'] and len(a['samples_ms']) == LL.MAX_SAMPLES
    assert a['stats']['n'] == 5000 and 27 < a['stats']['median'] < 33 and a['stats']['max'] >= 250
    assert a['stats']['p99'] == pytest.approx(250, abs=1) and a['samples_ms'][-1] == a['stats']['max']
    assert LL.profile_synthetic('lognormal:median=30', seed=4)['sha256'] != LL.profile_synthetic(
        'lognormal:median=30', seed=5)['sha256']
    for bad in ('gauss:3', 'constant:x', 'lognormal:sigma=1', 'lognormal:median=30,what=1', 'constant:-4'):
        with pytest.raises(LL.LatencyError):
            LL.profile_synthetic(bad)
    # sampling: the inverse CDF, interpolated, monotone in u
    assert LL.quantile([10.0, 20.0, 40.0], 0.0) == 10 and LL.quantile([10.0, 20.0, 40.0], 0.75) == 30
    assert LL.quantile([10.0, 20.0, 40.0], 1.0) == 40 and LL.quantile([7.0], 0.3) == 7
    h = LL.histogram(a, 20)
    assert sum(h['counts']) == len(a['samples_ms']) and len(h['edges']) == 21 and LL.histogram(c)['counts'] == [1]
    # a file
    f = tmp_path / 'lat.txt'
    f.write_text('12.5\n14\n13.25\n90\n')
    d = LL.profile_from_file(str(f))
    assert d['samples_ms'] == [12.5, 13.25, 14.0, 90.0] and d['source']['kind'] == 'file' and not d['measured']
    (tmp_path / 'lat.json').write_text(json.dumps({'samples_ms': [5, 6, 7]}))
    assert LL.profile_from_file(str(tmp_path / 'lat.json'))['stats']['median'] == 6
    (tmp_path / 'bad.txt').write_text('fast\n')
    with pytest.raises(LL.LatencyError):
        LL.profile_from_file(str(tmp_path / 'bad.txt'))
    with pytest.raises(LL.LatencyError, match='not negative'):
        LL.make_profile([3, -1], {'kind': 'file'})
    # a HIL session: the device's own camera-received → control-sent interval, per frame
    run = tmp_path / 'hil'
    run.mkdir()
    (run / 'hil.json').write_text(json.dumps({'fps': 20, 'device': {'mode': 'local'}}))
    with open(run / 'latency.jsonl', 'w') as fh:
        for k in range(50):
            rec = {'frame': k, 'tick_start': k * 50_000_000, 'applied': k * 50_000_000 + 31_000_000 + k * 100_000,
                   'dev_cam_rx': 7_000_000_000 + k * 50_000_000}
            if k != 10:                                      # one frame whose control never came back
                rec['dev_ctrl_sent'] = rec['dev_cam_rx'] + 18_000_000 + k * 200_000
            fh.write(json.dumps(rec) + '\n')
    d = LL.profile_from_hil(str(run))
    assert d['measured'] and d['source']['measure'] == 'on_device' and d['source']['frames_used'] == 49
    assert 'simulated device' in d['source']['device'] and d['stats']['min'] == 18.0 and d['stats']['max'] == 27.8
    e = LL.profile_from_hil(str(run), 'e2e')
    assert e['stats']['n'] == 50 and e['stats']['min'] == 31.0
    with pytest.raises(LL.LatencyError, match='no HIL session'):
        LL.profile_from_hil(str(tmp_path))
    # a deployment's on-device profiling: the timed runs; a simulated device is said to be one
    dep = tmp_path / 'dep' / 'engines' / 'fp16'
    dep.mkdir(parents=True)
    (dep / 'profile.json').write_text(json.dumps({'latency_ms': [8.1, 8.4, 9.0, 8.2], 'simulated': True,
                                                  'latency_definition': 'session.run()'}))
    d = LL.profile_from_deploy(str(tmp_path / 'dep'))
    assert d['measured'] and d['source']['simulated_device'] and 'not the Orin' in d['source']['device']
    assert d['stats']['median'] == pytest.approx(8.3) and d['name'] == 'deploy-fp16'
    with pytest.raises(LL.LatencyError, match='no profiled engine'):
        LL.profile_from_deploy(str(tmp_path / 'dep'), 'int8')
    p = tmp_path / 'p.json'
    p.write_text(json.dumps(d))
    assert LL.load_profile(str(p))['sha256'] == d['sha256'] and 'MEASURED' in LL.profile_text(d)
    with pytest.raises(LL.LatencyError, match='not a latency profile'):
        LL.load_profile(str(run / 'hil.json'))


# ================================================================================================ the delay line
def _u(k):
    return {'throttle': 0.1 * k, 'steer': 0.0, 'brake': 0.0, 'k': k}


def _applied(line, n):
    return [line.step(_u(k)).get('k') for k in range(n)]


def test_the_delay_line_by_hand():
    dt = 0.05
    # no latency: the control of the step, the same values
    z = LL.DelayLine(LL.latency_block(constant_ms=0.0), dt)
    assert [z.step(_u(k)) for k in range(5)] == [_u(k) for k in range(5)]
    assert z.report()['delay_steps'] == {'0': 5} and z.report()['held_steps'] == 0
    # 120 ms at 50 ms steps: rounded UP to 3 steps; coasting until the first arrives
    first = LL.DelayLine(LL.latency_block(constant_ms=120.0), dt).step(_u(0))
    assert first == LL.COAST and 'k' not in first
    d = LL.DelayLine(LL.latency_block(constant_ms=120.0), dt)
    assert _applied(d, 8) == [None, None, None, 0, 1, 2, 3, 4] and d.report()['delay_steps'] == {'3': 8}
    assert d.report()['mean_ms'] == 120.0 and d.report()['held_steps'] == 3
    # exactly one step is one step; a hair more is two
    assert _applied(LL.DelayLine(LL.latency_block(constant_ms=50.0), dt), 4) == [None, 0, 1, 2]
    assert _applied(LL.DelayLine(LL.latency_block(constant_ms=50.5), dt), 4) == [None, None, 0, 1]
    assert _applied(LL.DelayLine(LL.latency_block(constant_ms=0.4), dt), 3) == [None, 0, 1]
    # the scale multiplies the latency
    assert _applied(LL.DelayLine(LL.latency_block(constant_ms=50.0, scale=2.0), dt), 4) == [None, None, 0, 1]
    # serial: one computation at a time — the observations in between are skipped, the control held
    s = LL.DelayLine(LL.latency_block(constant_ms=150.0, mode='serial'), dt)
    assert _applied(s, 10) == [None, None, None, 0, 0, 0, 3, 3, 3, 6]
    assert s.report()['skipped_observations'] == 6 and s.report()['computed'] == 4
    # a spike: the results computed during it arrive first and are applied; the spike's own, overtaken, is dropped
    prof = LL.make_profile([0.0, 400.0], {'kind': 'file'})

    class Spike(LL.DelayLine):
        def draw_ms(self):
            self.rnd.random()
            return 400.0 if self.k == 3 else 0.0             # (k is already the next step's index)
    sp = Spike(LL.latency_block(prof), dt)
    out = _applied(sp, 12)
    assert out[:2] == [0, 1] and out[2] == 1 and out[3:] == list(range(3, 12))     # step 2 held step 1's control
    assert sp.report()['overtaken'] == 1 and sp.report()['delay_steps'] == {'0': 11, '8': 1}
    assert sp.report()['held_steps'] == 1
    # common random numbers: the same uniform at every step whatever the scale, so the latency scales exactly
    prof = LL.profile_synthetic('lognormal:median=60,sigma=0.6,spike_p=0.05,spike_ms=400', seed=1)
    a = LL.DelayLine(LL.latency_block(prof, scale=1.0, seed=9), dt)
    b = LL.DelayLine(LL.latency_block(prof, scale=2.0, seed=9), dt)
    c = LL.DelayLine(LL.latency_block(prof, scale=1.0, seed=10), dt)
    la, lb, lc = ([x.draw_ms() for _ in range(200)] for x in (a, b, c))
    assert lb == [2 * v for v in la] and la != lc
    # the delay never brings back an older control than the one applied
    line = LL.DelayLine(LL.latency_block(prof, scale=3.0, seed=2), dt)
    seen = [k for k in _applied(line, 400) if k is not None]
    assert seen == sorted(seen) and len(set(seen)) > 50 and line.report()['overtaken'] > 0
    for bad in ({'scale': 1.0}, {'constant_ms': 5.0, 'mode': 'batch'}):
        with pytest.raises(LL.LatencyError):
            LL.DelayLine(bad, dt)
    with pytest.raises(LL.LatencyError):
        LL.latency_block(prof, constant_ms=3.0)
    with pytest.raises(LL.LatencyError):
        LL.latency_block(constant_ms=3.0, scale=-1)


# ================================================================================================ on the fake carla
@pytest.fixture(scope='module')
def sim(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.sim.service import study_service
    base = tmp_path_factory.mktemp('lat')
    p = Project.new('Latency')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(base / 'lat.falcon.json'))
    svc = study_service(workspace=p.workspace, fake_carla=FAKE, lock_dir=str(base / 'locks'))
    yield {'project': p, 'svc': svc, 'base': base}
    svc.shutdown()


def _budget(sim, keys=('lead_brake',), **kw):
    from falcon_autonomy.core import scenarios as SC
    args = dict(max_ms=1500.0, fps=10, seed=2)
    args.update(kw)
    return LL.budget(sim['svc'], sim['project'], [SC.builtin(k) for k in keys], 'b', 'world-town10', 'rig-default',
                     **args)


def _drive(sim, cfg):
    from falcon_autonomy.core.jobs import FINAL
    job = sim['svc'].submit_drive(cfg, title='t')
    t0 = time.time()
    while job.state not in FINAL and time.time() - t0 < 120:
        sim['svc'].queue.poll()
        time.sleep(0.1)
    assert job.state == 'done', job.state
    return json.load(open(os.path.join(cfg['out'], 'run.json')))


def test_off_is_the_run_it_always_was_and_zero_latency_changes_nothing(sim):
    from falcon_autonomy.core import datasets as D
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.sim.studio import Study
    p, svc = sim['project'], sim['svc']
    scn = SC.builtin('lead_brake')
    vals = SC.sample_params(scn, 2 * 1000003)
    plain = Study(svc, p, scn, 'off', 'world-town10', 'rig-default', seed=2, fps=10)
    lat = LL.make_study(svc, p, scn, 'off', 'world-town10', 'rig-default', 2, 10, None)
    a, b = plain.config(vals, 0), lat.config(vals, 0)
    assert a == b and 'latency' not in a                       # no block: the config (and its cache key) as before
    assert lat.config(dict(vals, **{LL.LAT_KEY: 0.0}), 0) == a
    late = lat.config(dict(vals, **{LL.LAT_KEY: 300.0}), 0)
    assert late['latency'] == {'scale': 1.0, 'mode': 'pipeline', 'seed': 2, 'constant_ms': 300.0}
    assert late['cache_key'] != a['cache_key'] and D.drive_key({k: v for k, v in late.items() if k != 'latency'
                                                                 and k != 'cache_key'}) == a['cache_key']
    base = _drive(sim, dict(a, out=str(sim['base'] / 'r-off')))
    again = _drive(sim, dict(a, out=str(sim['base'] / 'r-off2')))
    zero = _drive(sim, dict(a, out=str(sim['base'] / 'r-zero'), latency=LL.latency_block(constant_ms=0.0, seed=2)))
    slow = _drive(sim, dict(late, out=str(sim['base'] / 'r-late')))

    def frames(name):
        return open(sim['base'] / name / 'frames.jsonl', 'rb').read()
    assert 'latency' not in base['scenario'] and frames('r-off') == frames('r-off2')
    assert frames('r-zero') == frames('r-off')                  # a delay of 0 steps: byte for byte the same drive
    assert zero['scenario']['latency']['delay_steps'] == {'0': 120}
    assert zero['scenario']['criticality'] == base['scenario']['criticality'] == again['scenario']['criticality']
    assert frames('r-late') != frames('r-off')
    rep = slow['scenario']['latency']
    assert rep['delay_steps'] == {'3': 120} and rep['step_ms'] == 100.0 and rep['held_steps'] == 3
    assert slow['scenario']['criticality']['criticality'] > base['scenario']['criticality']['criticality']
    # the applied control is the one computed three steps earlier (and the vehicle coasts until the first arrives)
    f0 = [json.loads(x) for x in frames('r-off').splitlines()]
    f3 = [json.loads(x) for x in frames('r-late').splitlines()]
    def acts(f):
        return any(abs(float(f['control'].get(k, 0.0))) > 0 for k in ('throttle', 'steer', 'brake'))
    j = next(i for i, f in enumerate(f0) if acts(f))               # the first step the driver does anything
    assert not any(acts(f) for f in f3[:j + 3]) and f3[j + 3]['control'] == f0[j]['control']


def test_the_budget_is_bisected_and_agrees_with_every_grid_point(sim):
    doc = _budget(sim)
    assert doc['schema'] == LL.BUDGET_SCHEMA and doc['unit'] == 'ms' and doc['grid_step'] == 100.0
    assert doc['grid_max'] == 1500.0 and not doc['exhaustive'] and 'monotone' in doc['assumes']
    c = doc['cases'][0]
    assert c['status'] == 'fails only when late' and (c['budget'], c['first_failure']) == (800.0, 900.0)
    assert c['runs'] <= 7 and doc['fails_only_when_late'] == ['lead_brake#0']       # 16 grid points, 6 runs
    assert {t['x'] for t in c['tested']} >= {0.0, 800.0, 900.0, 1500.0}
    at = {t['x']: t for t in c['tested']}
    assert at[0.0]['latency'] is None and at[900.0]['latency']['delay_steps'] == {'9': 120}
    assert at[0.0]['passed'] and not at[900.0]['passed'] and at[900.0]['criticality'] > at[0.0]['criticality']
    full = _budget(sim, exhaustive=True)
    f = full['cases'][0]
    assert f['runs'] == 16 and (f['budget'], f['first_failure']) == (800.0, 900.0) and not f.get('non_monotone')
    crit = [t['criticality'] for t in f['tested']]
    assert crit[-1] >= crit[8] > crit[0]                        # the later the control, the more critical the run
    # the criticality criterion instead of the scenario's own checks; a scenario that fails with no latency at all
    d2 = _budget(sim, ('lead_brake', 'stalled_vehicle'), criterion='criticality', threshold=0.55)
    by = {c['scenario']: c for c in d2['cases']}
    assert by['stalled_vehicle']['status'] == 'fails without latency' and by['stalled_vehicle']['budget'] is None
    assert by['stalled_vehicle']['runs'] == 2
    assert by['lead_brake']['status'] == 'fails only when late' and by['lead_brake']['budget'] == 100.0
    assert d2['fails_only_when_late'] == ['lead_brake#0']
    # nothing fails below the swept maximum: the budget is at least that
    d3 = _budget(sim, max_ms=300.0)
    assert d3['cases'][0]['status'] == 'no failure up to the maximum' and d3['cases'][0]['budget'] == 300.0
    assert d3['cases'][0]['first_failure'] is None and d3['fails_only_when_late'] == []
    text = LL.budget_table(doc)
    assert 'lead_brake#0' in text and '800 ms' in text and 'fail only when the computer is late: lead_brake#0' in text
    # read against a profile: the share of its latencies over the budget
    prof = LL.make_profile([100.0] * 95 + [850.0] * 5, {'kind': 'file'})
    r = LL.with_profile(doc, prof)['cases'][0]
    assert r['over_budget_share'] == pytest.approx(0.05) and '5.0 %' in LL.budget_table(doc)


def test_a_profile_is_swept_by_its_scale(sim):
    prof = LL.profile_synthetic('constant:300')
    doc = _budget(sim, profile=prof, max_scale=4.0, step=0.5)
    c = doc['cases'][0]
    assert doc['unit'] == 'scale' and doc['profile']['measured'] is False and doc['grid_max'] == 4.0
    assert (c['budget'], c['first_failure']) == (2.5, 3.0)          # 750 ms → 8 steps passes, 900 ms → 9 fails
    assert c['budget_ms'] == {'median': 750.0, 'p99': 750.0, 'max': 750.0} and c['runs'] <= 6
    at = {t['x']: t for t in c['tested']}
    assert at[3.0]['latency']['delay_steps'] == {'9': 120} and at[3.0]['latency']['scale'] == 3.0
    # a distribution with spikes: seeded, so the sweep repeats exactly
    spiky = LL.profile_synthetic('lognormal:median=250,sigma=0.3,spike_p=0.05,spike_ms=1200', seed=1)
    a = _budget(sim, profile=spiky, max_scale=2.0, step=0.5)
    b = _budget(sim, profile=spiky, max_scale=2.0, step=0.5)
    ca, cb = a['cases'][0], b['cases'][0]
    assert (ca['status'], ca['budget']) == (cb['status'], cb['budget'])
    assert [t['criticality'] for t in ca['tested']] == [t['criticality'] for t in cb['tested']]
    lat = ca['tested'][-1]['latency']
    assert len(lat['delay_steps']) > 2 and lat['max_ms'] > 1000 and 'median' in LL.budget_table(a)
    with pytest.raises(LL.LatencyError):
        _budget(sim, max_ms=0.0)
    with pytest.raises(LL.LatencyError):
        _budget(sim, mode='batch')


def test_the_commands(sim, capsys, tmp_path):
    p = sim['project']
    assert LL.main_latency(['profile', '--synthetic', 'lognormal:median=40,sigma=0.3', '--project', p.path, '--name',
                            'orin-guess']) == P.OK
    out = capsys.readouterr().out
    pf = os.path.join(p.workspace, 'latency', 'profiles', 'orin-guess.json')
    assert 'NOT measured' in out and os.path.isfile(pf)
    assert LL.main_latency(['show', pf]) == P.OK and 'median' in capsys.readouterr().out
    assert LL.main_latency(['profile', '--synthetic', 'constant:5', '--from-file', 'x', '--out', 'y']) == P.FAILED
    assert LL.main_latency(['profile', '--synthetic', 'constant:5']) == P.FAILED
    capsys.readouterr()
    common = [p.path, '--scenario', 'lead_brake', '--fps', '10', '--seed', '2', '--fake-carla', FAKE]
    assert LL.main_latency(['run'] + common + ['--constant', '900', '--name', 'b-lead_brake', '--json']) == P.OK
    d = json.loads(capsys.readouterr().out)
    assert d['without']['passed'] is True and d['with']['passed'] is False and d['with']['latency']['max_ms'] == 900
    assert LL.main_latency(['run'] + common) == P.REFUSED
    assert 'one of them' in capsys.readouterr().err
    assert LL.main_latency(['budget'] + common + ['--constant', '--max-ms', '1500', '--name', 'b', '--profile',
                                                  pf]) == P.OK
    out = capsys.readouterr().out
    assert 'fails only when late' in out and '800 ms' in out and '"t":"result"' in out
    bj = os.path.join(LL.budget_dir(p.workspace, 'b'), 'latency-budget.json')
    doc = json.load(open(bj))
    assert doc['cases'][0]['budget'] == 800.0 and doc['reference_profile']['name'] == 'orin-guess'
    assert doc['cases'][0]['over_budget_share'] == 0.0
    assert LL.main_latency(['show', bj]) == P.OK and 'lead_brake#0' in capsys.readouterr().out
    assert LL.main_latency(['budget'] + common) == P.REFUSED
    assert LL.main_latency(['budget', p.path, '--scenario', 'nope', '--constant']) == P.REFUSED
    assert LL.main_latency(['show', p.path]) == P.FAILED
    capsys.readouterr()
    from falcon_autonomy import cli
    assert cli.LAZY['latency'][:2] == ('core.latencyloop', 'main_latency')
    assert LL.list_budgets(p.workspace) == [bj] and LL.list_profiles(p.workspace) == [pf]


def test_the_learning_environment_takes_the_delay_too(project, fake_carla, tmp_path):
    """workers/gym.py: with a `latency` block the action acts late; without one the episode is as before."""
    from falcon_autonomy.core import drivefeatures as DF
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service
    from falcon_autonomy.core.settings import Settings
    svc = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                       str(tmp_path / 'locks'))

    def speeds(name, block):
        cfg = gym_config(project, 'world-town10', str(tmp_path / name), svc.port, svc.tm_port)
        if block is not None:
            cfg['latency'] = block
        env = FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0)
        try:
            env.reset(seed=1)
            return [float(env.step([0.0, 1.0])[0]['state'][DF.FI['speed_mps']]) for _ in range(8)]
        finally:
            env.close()
    try:
        off = speeds('off', None)
        zero = speeds('zero', LL.latency_block(constant_ms=0.0))
        late = speeds('late', LL.latency_block(constant_ms=150.0))       # 3 steps at 20 Hz
    finally:
        svc.shutdown()
    assert off == zero and off[0] > 0 and off == sorted(off)
    assert late[:3] == [0.0, 0.0, 0.0] and late[3:] == pytest.approx(off[:5], abs=1e-6)


def test_the_latency_tools(sim):
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    p = sim['project']
    if not LL.list_budgets(p.workspace):                         # run alone: the budget the command test writes
        from falcon_autonomy.core.provenance import write_json
        os.makedirs(LL.budget_dir(p.workspace, 'b'), exist_ok=True)
        write_json(os.path.join(LL.budget_dir(p.workspace, 'b'), 'latency-budget.json'), _budget(sim))
    assert REGISTRY['analysis_latency_budget'].tier == 'read' and REGISTRY['analysis_latency_run'].tier == 'act'
    ctx = TOOLS.context(p.path, FAKE)
    try:
        r = call(ctx, 'analysis_latency_budget', {})
        assert r['budgets'][0] == {'name': 'b', 'unit': 'ms', 'cases': 1, 'fails_only_when_late': 1}
        r = call(ctx, 'analysis_latency_budget', {'name': 'b'})
        assert r['fails_only_when_late'] == ['lead_brake#0'] and r['step_ms'] == 100.0
        row = next(iter(v for v in r.values() if isinstance(v, list) and v and isinstance(v[0], dict)))[0]
        assert row['budget'] == 800.0 and row['first_failure'] == 900.0 and row['status'] == 'fails only when late'
        assert 'no latency budget' in call(ctx, 'analysis_latency_budget', {'name': 'nope'})['error']
        pv = REGISTRY['analysis_latency_run'].preview(ctx, scenarios=['lead_brake', 'cut_in'], max_ms=400)
        assert pv.startswith('falcon latency budget ') and '--scenario lead_brake --scenario cut_in' in pv
        assert '--constant --max-ms 400.0' in pv and '--fake-carla' in pv
        est = REGISTRY['analysis_latency_run'].estimate(ctx, scenarios=['lead_brake'], max_ms=400)
        assert est['seconds'] is not None and '5 ×' in est['what']
        assert 'no latency profile' in call(ctx, 'analysis_latency_run', {'scenarios': ['lead_brake'],
                                                                           'profile': 'nope'})['error']
    finally:
        ctx.service.shutdown()


# ================================================================================================ the tab
_APP = []


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    from falcon_autonomy.ui.app import make_app
    from falcon_autonomy.ui.theme import THEME
    app = QApplication.instance() or make_app()
    _APP.append(app)
    THEME.apply(app, 'dark')
    return app


@pytest.mark.gui
def test_the_latency_budget_tab_offscreen(qapp, tmp_path):
    from types import SimpleNamespace
    from falcon_autonomy.core.provenance import write_json
    from falcon_autonomy.ui.latency_tab import LatencyTab, word
    from falcon_autonomy.ui.theme import THEME
    ws = str(tmp_path / 'ws')
    os.makedirs(os.path.join(ws, 'latency', 'profiles'))
    submitted = []

    class Q:
        def submit(self, spec):
            submitted.append(spec)
            return SimpleNamespace(state='queued', summary=lambda: 'queued')
    win = SimpleNamespace(project=SimpleNamespace(path=str(tmp_path / 'p.falcon.json'), dirty=False, doc={}),
                          workspace=lambda: ws, queue=Q(), fake_carla=FAKE, save=lambda: True)
    tab = LatencyTab(SimpleNamespace(win=win))
    assert tab.tree.topLevelItemCount() == 0 and tab.hist.hist is None and 'No latency budget' in tab.note.text()
    assert tab.scenario.findData('lead_brake') >= 0 and tab.run_btn.property('role') == 'primary'
    prof = LL.profile_synthetic('lognormal:median=30,sigma=0.4,spike_p=0.02,spike_ms=250', seed=3, name='guess')
    write_json(os.path.join(ws, 'latency', 'profiles', 'guess.json'), prof)
    case = {'case': 'lead_brake#0', 'scenario': 'lead_brake', 'name': 'Lead vehicle brakes', 'values': {}, 'runs': 6,
            'baseline_criticality': 0.4, 'tested': [], 'status': 'fails only when late', 'budget': 150.0,
            'first_failure': 200.0, 'why': 'passes up to 150 ms, fails from 200 ms', 'over_budget_share': 0.02}
    doc = {'schema': LL.BUDGET_SCHEMA, 'name': 'b', 'unit': 'ms', 'grid_step': 50.0, 'grid_max': 500.0,
           'step_ms': 50.0, 'sweep': 'constant latency', 'quantisation': 'latencies are rounded up to whole steps',
           'assumes': 'common random numbers', 'fails_only_when_late': ['lead_brake#0'],
           'cases': [case,
                     dict(case, case='cut_in#0', status='no failure up to the maximum', budget=500.0,
                          first_failure=None, over_budget_share=None),
                     dict(case, case='stalled#0', status='fails without latency', budget=None, first_failure=0.0,
                          over_budget_share=None)]}
    os.makedirs(os.path.join(ws, 'latency', 'b'))
    write_json(os.path.join(ws, 'latency', 'b', 'latency-budget.json'), doc)
    tab.refresh()
    tab.profile.setCurrentIndex(tab.profile.findText('guess'))
    assert tab.hist.hist is not None and sum(tab.hist.hist['counts']) == len(prof['samples_ms'])
    assert 'NOT MEASURED' in tab.profile_note.text() and 'median' in tab.profile_note.text()
    tab.resize(420, 700)
    tab.hist.resize(400, 140)
    k = tab.hist.bar_at(12.0)
    assert k == 0 and tab.hist.tip(k).endswith('samples') and ' ms: ' in tab.hist.tip(k)
    assert tab.hist.bar_at(-5.0) is None
    rows = [tab.tree.topLevelItem(i) for i in range(tab.tree.topLevelItemCount())]
    assert [(r.text(0), r.text(1), r.text(2), r.text(3)) for r in rows] == [
        ('lead_brake#0', 'LATE ONLY', '150 ms', '200 ms'), ('cut_in#0', 'HOLDS', '≥ 500 ms', '—'),
        ('stalled#0', 'FAILS ANYWAY', '—', '0 ms')]
    assert [r.foreground(1).color().name() for r in rows] == [THEME.status(t) for t in ('warning', 'good', 'critical')]
    assert '2.0 %' in rows[0].toolTip(1) and tab.late.text() == 'lead_brake#0'
    assert word('not evaluated') == ('not evaluated', '')
    tab.grab()
    tab.hist.grab()
    tab.scenario.setCurrentIndex(tab.scenario.findData('lead_brake'))
    tab.start()
    argv = submitted[-1].argv
    assert argv[3:6] == ['latency', 'budget', win.project.path] and argv[argv.index('--scenario') + 1] == 'lead_brake'
    assert '--constant' in argv and argv[argv.index('--profile') + 1].endswith('guess.json')
    assert argv[argv.index('--fake-carla') + 1] == FAKE and 'Queued' in tab.note.text()
    src = open(os.path.join(os.path.dirname(LL.__file__), '..', 'ui', 'pages', 'hardware.py')).read()
    assert "optional_tab(pv, 'ui.latency_tab', 'LatencyTab', 'Latency budget', self)" in src
    assert not math.isnan(prof['stats']['p99'])
