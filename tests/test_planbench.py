"""The closed-loop planner benchmark (core.planbench, sim.benchrun): every term by hand on synthetic records, the
aggregate and the comparison, the route sets, a benchmark driven on the fake through the CLI, a policy's episode."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from falcon_autonomy.core import planbench as PB

LANES = {'columns': ['x', 'y', 'z', 'yaw', 'width', 'road', 'section', 'lane', 's'], 'step_m': 2.0,
         'points': [[float(x), 0.0, 0.0, 0.0, 3.5, 1, 0, -1, float(x) + 100] for x in range(-100, 301, 2)]
         + [[float(x), -3.5, 0.0, 180.0, 3.5, 1, 0, 1, float(x) + 100] for x in range(-100, 301, 2)]}


def traj(t, x, y=0.0, yaw=0.0, v=None, L=4.6, W=2.0):
    """Rows (t, x, y, yaw, vx, vy, L, W); v defaults to the positions' central difference along x."""
    t = np.asarray(t, float)
    x = np.broadcast_to(np.asarray(x, float), t.shape).astype(float)
    y = np.broadcast_to(np.asarray(y, float), t.shape).astype(float)
    if v is None:
        vx = np.gradient(x, t) if len(t) > 1 else np.zeros_like(t)
        vy = np.gradient(y, t) if len(t) > 1 else np.zeros_like(t)
    else:
        r = math.radians(yaw)
        vx, vy = np.full(t.shape, v * math.cos(r)), np.full(t.shape, v * math.sin(r))
    return np.column_stack([t, x, y, np.full(t.shape, yaw), vx, vy, np.full(t.shape, L), np.full(t.shape, W)])


T = np.round(np.arange(0, 6.0001, 0.05), 4)


def cruise(v=10.0, y=0.0):
    return traj(T, v * T, y)


# ------------------------------------------------------------------------------------------------- terms
def test_comfort_bounds():
    assert PB.comfort(PB.record(cruise()))['score'] == 1.0
    v = np.maximum(10.0 - 6.0 * np.maximum(T - 1.0, 0.0), 0.0)            # braking at 6 m/s²
    x = np.concatenate([[0.0], np.cumsum(v[:-1] * 0.05)])
    rec = PB.record(np.column_stack([T, x, 0 * T, 0 * T, v, 0 * T, 4.6 + 0 * T, 2 + 0 * T]))
    c = PB.comfort(rec)
    assert c['score'] == 0.0 and 'lon_accel_min' in c['failed']
    assert c['worst']['lon_accel_min'] == pytest.approx(-6.0, abs=0.6)
    # a steady turn: lateral acceleration v²/R, yaw rate v/R
    R, v = 20.0, 8.0
    th = v / R * T
    ego = np.column_stack([T, R * np.sin(th), R * (1 - np.cos(th)), np.degrees(th), v * np.cos(th), v * np.sin(th),
                           4.6 + 0 * T, 2 + 0 * T])
    k = PB.kinematics(ego)
    assert np.median(k['lat_accel']) == pytest.approx(v * v / R, rel=0.02)
    assert np.median(k['yaw_rate']) == pytest.approx(v / R, rel=0.02)


def test_collision_classes_and_nc():
    # the ego drives into a car standing in its lane: at fault (stopped track), NC 0
    stand = {'kind': 'car', 'traj': traj(T, 40.0, v=0.0)}
    nc = PB.collisions(PB.record(cruise(), {'c': stand}))
    assert nc['score'] == 0.0 and nc['collisions'][0]['class'] == 'stopped_track'
    assert nc['collisions'][0]['t'] == pytest.approx(3.5, abs=0.06)
    # the ego stands; a car runs into its back: not at fault
    ego = traj(T, 0.0, v=0.0)
    back = {'kind': 'car', 'traj': traj(T, -20.0 + 8.0 * T)}
    nc = PB.collisions(PB.record(ego, {'b': back}))
    assert nc['score'] == 1.0 and nc['collisions'][0]['class'] == 'stopped_ego'
    # both moving, a faster car from behind: rear, not at fault
    fast = {'kind': 'car', 'traj': traj(T, -20.0 + 16.0 * T)}
    nc = PB.collisions(PB.record(cruise(8.0), {'f': fast}))
    assert nc['collisions'][0]['class'] == 'active_rear' and nc['score'] == 1.0
    # a handcart standing: a static object, 0.5
    cart = {'kind': 'handcart', 'traj': traj(T, 40.0, v=0.0, L=2.4, W=1.2)}
    assert PB.collisions(PB.record(cruise(), {'h': cart}))['score'] == 0.5
    # a sensor contact with something that is not a scenario actor, the ego moving: static, at fault
    nc = PB.collisions(PB.record(cruise(), {}, events=[(2.0, None)]))
    assert nc['score'] == 0.5 and nc['collisions'][0]['kind'] == 'object'


def test_lateral_contact_at_fault_only_across_lanes():
    side = {'kind': 'car', 'traj': traj(T, 10.0 * T, 3.5)}           # beside in the next lane, same speed
    # the ego drifts into it within its lane's margin? no: the other drifts into the ego staying in its lane
    other_in = {'kind': 'car', 'traj': traj(T, 10.0 * T, np.maximum(3.5 - 1.0 * T, 1.9))}
    rec = PB.record(cruise(), {'o': other_in}, LANES)
    facts = PB.lane_facts(rec['map'], rec['ego'])
    nc = PB.collisions(rec, facts)
    assert nc['collisions'][0]['class'] == 'active_lateral' and nc['score'] == 1.0
    # the ego drifts over the line into the car beside it: at fault
    ego = traj(T, 10.0 * T, np.minimum(0.8 * T, 1.7))
    rec = PB.record(ego, {'s': side}, LANES)
    nc = PB.collisions(rec, PB.lane_facts(rec['map'], rec['ego']))
    assert nc['collisions'][0]['class'] == 'active_lateral' and nc['score'] == 0.0


def test_drivable_area_and_direction():
    rec = PB.record(cruise(), None, LANES)
    dac, _ = PB.drivable_area(rec)
    assert dac['score'] == 1.0
    rec = PB.record(cruise(y=2.6), None, LANES)             # the body 0.6 m past the right edge (no lane there)
    dac, _ = PB.drivable_area(rec)
    assert dac['score'] == 0.0 and dac['max_outside_m'] == pytest.approx(2.6 + 1.0 - 1.75, abs=0.05)
    rec = PB.record(cruise(y=1.9), None, LANES)                # 0.3 m tolerance: 1.9 + 1 − 1.75 = 1.15 m out
    assert PB.drivable_area(rec)[0]['score'] == 0.0
    rec = PB.record(cruise(y=0.9), None, LANES)                # 0.15 m out: within the tolerance
    assert PB.drivable_area(rec)[0]['score'] == 1.0
    # against the traffic on the oncoming lane (y = −3.5, heading 180°) at 4 and 8 m/s: 4 m and 8 m in 1 s
    for v, want in ((1.5, 1.0), (4.0, 0.5), (8.0, 0.0)):
        ego = traj(T, v * T, -3.5)
        rec = PB.record(ego, None, LANES)
        d = PB.driving_direction(rec, PB.lane_facts(rec['map'], rec['ego']))
        assert d['score'] == want and d['max_against_m'] == pytest.approx(v * 1.0, abs=v * 0.06)
    assert PB.driving_direction(PB.record(cruise()), None)['score'] is None


def test_ttc_bound():
    # closing on a car standing 20 m ahead at 10 m/s, stopping before it: the TTC falls below 0.95 s on the way
    stop_at = 16.0
    v = np.where(T < 1.0, 10.0, np.maximum(10.0 - 10.0 * (T - 1.0), 0.0))
    x = np.minimum(np.concatenate([[0.0], np.cumsum(v[:-1] * 0.05)]), stop_at)
    ego = np.column_stack([T, x, 0 * T, 0 * T, v, 0 * T, 4.6 + 0 * T, 2 + 0 * T])
    car = {'kind': 'car', 'traj': traj(T, 20.0, v=0.0)}
    r = PB.ttc_bound(PB.record(ego, {'c': car}))
    assert r['score'] == 0.0 and r['min_ttc_s'] < 0.95
    far = {'kind': 'car', 'traj': traj(T, 200.0, v=0.0)}
    assert PB.ttc_bound(PB.record(cruise(), {'c': far}))['score'] == 1.0
    behind = {'kind': 'car', 'traj': traj(T, -6.0 + 10.0 * T, v=12.0)}       # closing from behind: not counted
    assert PB.ttc_bound(PB.record(cruise(), {'b': behind}))['score'] == 1.0


def test_progress_pdms_and_partial():
    ref = PB.record(cruise(10.0), None, LANES)
    half = PB.record(traj(T, 5.0 * T), None, LANES)
    row = PB.score_record(half, ref)
    assert row['terms']['ep']['score'] == pytest.approx(0.5, abs=0.01)
    s = row['scores']
    assert s['nc'] == s['dac'] == s['ddc'] == s['ttc'] == s['comfort'] == 1.0
    assert row['pdms'] == pytest.approx((5 + 5 * 0.5 + 2) / 12, abs=0.005)
    assert row['pdms_navsim_v1'] == pytest.approx(row['pdms']) and row['partial'] == []
    # EP is 1 when the reference hardly moved
    assert PB.ep_score(0.0, 3.0) == 1.0 and PB.ep_score(12.0, 10.0) == 1.0
    # driving direction is a multiplier in Falcon's PDMS, not in NAVSIM v1's
    wrong = PB.record(traj(T, 4.0 * T, -3.5), None, LANES)
    r = PB.score_record(wrong, 24.0)
    assert r['scores']['ddc'] == 0.5 and r['pdms'] == pytest.approx(0.5 * r['pdms_navsim_v1'])
    # without a map, DAC and DDC are left out and said
    r = PB.score_record(PB.record(cruise()), 60.0)
    assert r['partial'] == ['dac', 'ddc'] and r['pdms'] == pytest.approx(1.0)


def _row(nc=1.0, ttc=1.0, comfort=1.0, progress=50.0):
    return {'terms': {'nc': {'score': nc}, 'dac': {'score': 1.0}, 'ddc': {'score': 1.0}, 'ttc': {'score': ttc},
                      'comfort': {'score': comfort}}, 'progress_m': progress}


def test_aggregate_intervals_and_comparison():
    scored = {'follower': {f'r{k}': _row(progress=50.0) for k in range(6)},
              'worse': {f'r{k}': _row(nc=0.0 if k % 2 else 1.0, progress=40.0) for k in range(6)}}
    res = PB.aggregate(scored, 'follower', n_boot=300)
    f, w = res['planners']['follower'], res['planners']['worse']
    assert f['metrics']['pdms']['value'] == pytest.approx(1.0)
    assert w['metrics']['ep']['value'] == pytest.approx(0.8)
    want = 0.5 * (5 + 5 * 0.8 + 2) / 12
    assert w['metrics']['pdms']['value'] == pytest.approx(want)
    lo, hi = w['metrics']['pdms']['ci_low'], w['metrics']['pdms']['ci_high']
    assert lo < want < hi
    c = res['comparisons'][0]
    assert c['a'] == 'follower' and c['b'] == 'worse' and c['routes'] == 6
    assert c['metrics']['pdms']['diff'] == pytest.approx(want - 1.0) and c['metrics']['pdms']['ci_high'] < 0
    assert res['reference']['from']['r0'] == 'follower'
    # no reference planner: the best progress on each route
    res = PB.aggregate({'a': {'r0': _row(progress=30.0)}, 'b': {'r0': _row(progress=60.0)}}, None, n_boot=50)
    assert res['planners']['a']['metrics']['ep']['value'] == pytest.approx(0.5)
    assert 'best' in res['reference']['from']['r0']
    assert 'pdms' in PB.table(res)


def test_route_sets_and_worlds(project):
    rs = PB.routes('falcon-pdm-v1')
    ids = [r['id'] for r in rs]
    assert len(ids) == len(set(ids)) == 23
    assert all(r['motion'] == 'kinematic' and r['traffic'] == 0 for r in rs)
    assert sum(r['world'] == 'osm:bengaluru' for r in rs) == 11
    assert PB.resolve_world(project, 'town:Town10HD_Opt') == ('world-town10', None)
    wid, why = PB.resolve_world(project, 'osm:bengaluru')
    assert wid is None and 'Import a place' in why
    project.doc['worlds'].append({'id': 'w-blr', 'name': 'Central Bengaluru', 'kind': 'osm'})
    assert PB.resolve_world(project, 'osm:bengaluru')[0] == 'w-blr'
    with pytest.raises(PB.BenchError):
        PB.routes('nope')


# ------------------------------------------------------------------------------------------------- on the fake
def test_bench_run_on_the_fake_through_the_cli(project, tmp_path, fake_carla):
    path = str(tmp_path / 'b.falcon.json')
    project.save(path)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *a], capture_output=True, text=True, env=env,
                              timeout=600)
    r = cli('bench', 'run', path, '--name', 'mini', '--set', 'falcon-pdm-mini', '--planner', 'none', '--route',
            't10-lead_brake-s1', '--route', 't10-stalled_vehicle-s1', '--boot', '100', '--fake-carla', fake_carla)
    assert r.returncode == 0, r.stdout + r.stderr
    bdir = os.path.join(project.workspace, 'benchmarks', 'mini')
    res = json.loads(open(os.path.join(bdir, 'results.json')).read())
    assert set(res['planners']) == {'follower', 'none'}
    nf, nn = res['planners']['follower'], res['planners']['none']
    assert nf['n_routes'] == nn['n_routes'] == 2 and nf['partial_routes'] == 0      # lanes.json was written
    assert nf['metrics']['nc']['value'] == 1.0 and nn['metrics']['nc']['value'] == 0.0   # 'none' never brakes
    assert nn['metrics']['ttc']['value'] == 0.0
    assert res['comparisons'][0]['metrics']['pdms']['diff'] < 0
    run = os.path.join(bdir, 'runs', 'none', 't10-lead_brake-s1')
    assert os.path.isfile(os.path.join(run, 'lanes.json'))
    assert json.load(open(os.path.join(run, 'drive.json')))['rig']['sensors'][0]['width'] == 64
    # resume: nothing is driven again
    n = len(open(os.path.join(bdir, 'index.jsonl')).read().splitlines())
    r = cli('bench', 'run', path, '--name', 'mini', '--set', 'falcon-pdm-mini', '--planner', 'none', '--route',
            't10-lead_brake-s1', '--route', 't10-stalled_vehicle-s1', '--boot', '50', '--fake-carla', fake_carla)
    assert r.returncode == 0 and len(open(os.path.join(bdir, 'index.jsonl')).read().splitlines()) == n
    r = cli('bench', 'score', run, '--json')
    assert r.returncode == 0 and json.loads(r.stdout)[run]['scores']['nc'] == 0.0
    r = cli('bench', 'results', bdir, '--boot', '50')
    assert r.returncode == 0 and 'none − follower' in r.stdout
    r = cli('bench', 'routes', '--set', 'falcon-pdm-mini')
    assert r.returncode == 0 and 't10-cut_in-s1' in r.stdout
    assert PB.list_benchmarks(project.workspace)[0]['runs'] == 4


def test_a_policy_episode_carries_its_trajectories(project, tmp_path, fake_carla):
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service
    from falcon_autonomy.core.settings import Settings
    svc = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                       str(tmp_path / 'locks'))
    try:
        cfg = gym_config(project, 'world-town10', str(tmp_path / 'env'), svc.port, svc.tm_port,
                         scenario='lead_brake', values={'speed': 40.0, 'gap': 10.0, 'decel': 9.0, 'brake_at': 0.5},
                         max_episode_s=6.0)
        cfg['write_lanes'] = True
        with FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0) as env:
            env.reset(seed=1)
            done = False
            while not done:
                _o, _r, term, trunc, info = env.step(np.array([0.0, 1.0], np.float32))
                done = term or trunc
            assert 'trajectories' not in info['episode_summary']          # kept out of the step's reply
            run_dir = env.run_dir
    finally:
        svc.shutdown()
    e = json.loads(open(os.path.join(run_dir, 'episodes.jsonl')).read().splitlines()[0])
    tj = e['trajectories']
    assert tj['kinds'] == {'lead': 'car'} and len(tj['actors']['ego']) == len(tj['actors']['lead']) > 10
    lanes = json.load(open(os.path.join(run_dir, 'lanes.json')))
    row = PB.score_record(PB.record_from_episode(e, lanes))
    assert row['scores']['nc'] == 0.0 and row['scores']['dac'] == 1.0 and row['partial'] == ['ep']
    entry = {'planner': 'policy:x', 'route': 'r', 'episodes': os.path.join(run_dir, 'episodes.jsonl'), 'episode': 0,
             'env_dir': run_dir}
    assert PB.score_entry(entry)['scores']['nc'] == 0.0
    with pytest.raises(PB.BenchError, match='no trajectories'):
        PB.record_from_episode({'trace': {}})
