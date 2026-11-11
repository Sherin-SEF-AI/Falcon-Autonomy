"""Level-k game-theoretic traffic (core.levelk, numpy) and differentiable traffic (core.trafficgraph, torch): an
analytic two-agent gap game, quantal-response probabilities, emergent inching, realism, numpy without torch, the
torch utility equal to the numpy one, gradcheck of the bicycle rollout, recovery of planted parameters, the gradient
adversary search; and the CARLA-side controllers on the fake: a level-k scenario actor, level-k adversaries and
level-k background traffic."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT
from falcon_autonomy.core import levelk as L


# ------------------------------------------------------------------------------------------------ the model
def test_quantal_response_probabilities():
    U = np.array([1.0, 0.0, -1.0])
    p = L.quantal(U, 2.0)
    e = np.exp(2.0 * U)
    assert np.allclose(p, e / e.sum()) and abs(p.sum() - 1) < 1e-12
    assert np.allclose(L.quantal(U, 0.0), 1 / 3)                         # λ = 0: uniform
    assert L.quantal(U, 200.0)[0] > 0.999                                # λ → ∞: the best response
    assert np.isfinite(L.quantal(np.array([1e6, 0.0]), 50.0)).all()      # stable for large utilities


def test_two_agent_gap_game_analytic():
    """A lone car with a free road: the best plan reaches v_des (progress capped there, the speed term zero at it,
    comfort favouring small accelerations); a stopped car 20 m ahead makes every accelerating plan a collision."""
    road = L.RefPath.line(0, 0, 0, 300, 0, (-1.75, 1.75))
    a = L.make_agent('a', 'car', road, 0.0, 0.0, 11.0, {'level': 1, 'lam': 50.0})
    sc = L.Scene([a])
    b = L.Solver(sc, mode='best').action(0)
    assert b['accel'] == 0.0 and b['lateral'] == 0.0                    # at v_des: hold
    # a blocking agent: plans that overlap it within the horizon pay the collision weight
    obst = L.make_agent('b', 'car', road, 20.0, 0.0, 0.0, {'level': 0, 'v_des': 0.3}, controlled=False)
    a2 = L.make_agent('a', 'car', road, 0.0, 0.0, 8.0, {'level': 1, 'lam': 50.0})
    sol = L.Solver(L.Scene([a2, obst]), mode='best')
    b2 = sol.action(0)
    F = b2['features']
    assert (F['coll'][b2['A'] >= 0] > 0).all()                          # every non-braking plan collides
    assert b2['accel'] <= -2.0 and F['coll'][b2['index']] == 0           # it brakes hard enough not to
    # analytic: with a constant deceleration from 8 m/s the stopping distance is v² / 2a (+ the jerk ramp)
    R = L.roll(a2, [b2['accel']], [0.0])
    assert R['s'][0, -1] < 20.0 - (4.5 + 4.5) / 2.0


def test_level0_idm_and_gap_acceptance():
    road = L.RefPath.line(0, 0, 0, 300, 0)
    lead = L.make_agent('lead', 'car', road, 30.0, 0.0, 5.0, {'level': 0}, controlled=False)
    f = L.make_agent('f', 'car', road, 0.0, 0.0, 10.0, {'level': 0})
    a, lat = L.level0(L.Scene([f, lead]), 0)
    gap = 30.0 - 4.5
    assert a == pytest.approx(float(np.clip(L.idm(10.0, 11.0, gap, 5.0, 2.0), L.IDM_MIN, 2.0)))
    assert lat == 0.0
    # gap acceptance: a stopped rider before a crossing goes for a car arriving later than t_gap (free IDM), and for
    # one arriving sooner only up to its stop line (IDM on a stopped virtual leader there)
    side = L.RefPath([(30.0, 12.0), (30.0, -40.0)])
    stop = 12.0 - 1.9 / 2.0 - L.CORRIDOR_MARGIN_M - (6.5 + 1.0)            # conflict − half the car − margin − front
    for car_s, expect in ((15.0, L.idm(0.0, 11.0, stop, 0.0, 2.0)), (-40.0, L.idm(0.0, 11.0, 1e6, 0.0, 2.0))):
        car = L.make_agent('car', 'car', road, car_s, 0.0, 9.0, {'level': 0}, controlled=False, priority=True)
        rider = L.make_agent('rider', 'two_wheeler', side, 6.5, 0.0, 0.0, {'level': 0, 't_gap': 3.0})
        acc, _ = L.level0(L.Scene([car, rider]), 1)
        assert acc == pytest.approx(float(expect)), (car_s, acc)
    assert L.idm(0.0, 11.0, stop, 0.0, 2.0) < 1.5                          # it creeps, then stops at the line


def test_inching_emerges_at_level_1_not_level_0():
    """The crossing scene: a stream of level-0 cars with the right of way; an assertive rider (α 0.6) never enters at
    level 0 (no gap of t_gap) but creeps into the stream at level 1, making a car yield within the realism bounds."""
    out = {}
    for lvl in (0, 1, 2):
        sim = L.simulate(L.scene('crossing', {'rider': lvl}), 10.0, seed=0, mode='best')
        out[lvl] = (L.inching(sim), L.realism(sim))
    i0, r0 = out[0]
    assert not i0['entered'] and r0['realistic']
    for lvl in (1, 2):
        i1, r1 = out[lvl]
        assert i1['inched'] and i1['entered'], i1
        assert i1['vehicle_time_to_reach_s'] < 3.0 and i1['mean_approach_speed_mps'] < 2.5
        assert 0.5 < i1['vehicle_max_decel_after_mps2'] <= 4.0 + 1e-6      # it yields, within REALISM_LIMITS
        assert r1['realistic'] and not r1['collisions']
    # a timid rider (α 0.3) waits at level 1 too: assertiveness is what makes it inch
    sim = L.simulate(L.scene('crossing', {'rider': 1}, {'rider': {'alpha': 0.3}}), 10.0, seed=0, mode='best')
    assert not L.inching(sim)['entered']
    # sampled quantal responses (λ 8) inch as well, reproducibly per seed
    a = L.simulate(L.scene('crossing'), 10.0, seed=3)
    b = L.simulate(L.scene('crossing'), 10.0, seed=3)
    assert a['trajectories'] == b['trajectories'] and L.inching(a)['inched']


def test_scenes_realistic_and_bounded():
    for name in L.SCENES:
        sim = L.simulate(L.scene(name), 6.0, seed=1)
        r = L.realism(sim)
        assert not r['collisions'], (name, r['collisions'])
        for aid, v in r['agents'].items():
            assert v['peaks']['a_long'] <= 4.0 + 1e-6, (name, aid, v)
        info = {a['id']: a for a in sim['agents']}
        for aid, tr in sim['trajectories'].items():                       # lateral targets stay on the road
            b = info[aid]['path'].get('bounds')
            if b and info[aid]['type'] != 'pedestrian':
                w = L.TYPES[info[aid]['type']]['width'] / 2.0
                assert min(tr['d']) >= b[0] + w - 0.1 and max(tr['d']) <= b[1] - w + 0.1, (name, aid)


def test_params_and_profile():
    from falcon_autonomy.core.project import Project
    p = L.params_of({'k': 2, 'lambda': 3.0}, 'two_wheeler')
    assert p['level'] == 2 and p['lam'] == 3.0 and p['v_des'] == L.TYPES['two_wheeler']['v_des']
    with pytest.raises(ValueError, match='outside'):
        L.params_of({'alpha': 2.0}, 'car')
    with pytest.raises(ValueError, match='unknown'):
        L.params_of({'speed': 3}, 'car')
    tp = Project.new('T').doc['traffic_profiles'][0]
    tp['behaviour'].update(model='levelk', levelk={'motorcycle': {'k': 2, 'alpha': 0.7}, '*': {'lambda': 4.0}})
    t = L.profile_params(tp)
    assert L.params_for(t, 'two_wheeler') == dict(L.params_of({}, 'two_wheeler'), level=2, alpha=0.7, lam=4.0)
    tp['behaviour']['levelk'] = {'tram': {}}
    with pytest.raises(ValueError, match='unknown kind'):
        L.profile_params(tp)


def test_numpy_only_without_torch():
    """core.levelk runs in CARLA's Python, which has no torch: import and simulate with torch made unimportable."""
    code = ('import sys; sys.modules["torch"] = None\n'
            'from falcon_autonomy.core import levelk as L\n'
            'sim = L.simulate(L.scene("crossing"), 2.0, seed=0)\n'
            'assert "torch" not in [m for m in sys.modules if sys.modules[m] is not None]\n'
            'print(len(sim["trajectories"]))')
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == '5', r.stderr


def test_controller_interface_and_brake_lights():
    """The per-tick interface a neural simulator calls (core.levelk.Controller.step)."""
    road = L.RefPath.line(0, 0, 0, 300, 0, (-1.75, 1.75))                 # one lane: no swerving around it
    ctl = L.Controller(seed=0, dt=0.05)
    agents = [{'id': 'a', 'type': 'car', 'path': road, 's': 0.0, 'd': 0.0, 'v': 10.0, 'controlled': True},
              {'id': 'b', 'type': 'car', 'path': road, 's': 25.0, 'd': 0.0, 'v': 0.0, 'controlled': False}]
    seen = []
    for k in range(20):
        acts = ctl.step(k * 0.05, agents)
        assert set(acts) == {'a'} and set(acts['a']) >= {'accel', 'lateral', 'plan_accel', 'level', 'brake_light'}
        seen.append(acts['a'])
        ag = dict(agents[0], a=acts['a']['accel'], type='car')
        L.execute(ag, acts['a']['plan_accel'], acts['a']['lateral'], 0.05)
        agents[0].update(s=ag['s'], d=ag['d'], v=ag['v'])
    assert ctl.decisions == 4                                              # every DT_PLAN of 0.25 s
    acc = [a['accel'] for a in seen]
    assert all(abs(b - a) <= L.JERK_MAX * 0.05 + 1e-9 for a, b in zip(acc, acc[1:]))     # jerk-limited
    assert seen[-1]['brake_light'] and acc[-1] < -L.BRAKE_LIGHT_DECEL


# ------------------------------------------------------------------------------------------------ torch
def test_torch_utility_equals_numpy():
    torch = pytest.importorskip('torch')
    from falcon_autonomy.core import trafficgraph as TG
    sim = L.simulate(L.scene('platoon'), 5.0, seed=0)
    samples = L.decisions_from_sim(sim)
    assert samples
    th = {'alpha': 0.4, 'lam': 5.0, 'v_des': 9.0, 't_gap': 2.0}
    F, _choice = TG._batches(samples[:5], torch.float64)[0]
    Ut = TG.batched_utility({k: torch.tensor(v, dtype=torch.float64) for k, v in th.items()}, F)
    for b in range(len(Ut)):
        Fn = {k: (v[b].numpy() if hasattr(v, 'numpy') else v) for k, v in F.items()}
        Fn['gaps'] = Fn['gaps'][..., :samples[b]['features']['gaps'].shape[-1]]
        assert np.allclose(Ut[b].numpy(), L.utility(L.NP, th, Fn), atol=1e-9)


def test_gradcheck_bicycle_rollout():
    torch = pytest.importorskip('torch')
    from falcon_autonomy.core import trafficgraph as TG
    s0 = torch.tensor([[0.0, 0.0, 0.1, 5.0], [1.0, -2.0, -0.3, 8.0]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor(np.random.default_rng(0).normal(0, 0.3, (2, 6, 2)), dtype=torch.float64, requires_grad=True)
    wb = torch.tensor([2.7, 3.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, c: TG.bicycle_rollout(a, b, 0.1, c), (s0, u, wb))
    out = TG.bicycle_rollout(s0.detach(), torch.zeros(2, 3, 2, dtype=torch.float64), 0.1, 2.7)
    assert torch.allclose(out[0, -1, 0], torch.tensor(1.5 * math.cos(0.1), dtype=torch.float64))   # straight line
    e = TG.proximity_graph(np.array([[0, 0], [10, 0], [200, 0]], float), 50.0)
    assert sorted(map(tuple, e.T.tolist())) == [(0, 1), (1, 0)]


def test_recovers_planted_parameters():
    """Quantal choices sampled from known θ on random situations; maximum likelihood finds θ (Wald intervals)."""
    pytest.importorskip('torch')
    from falcon_autonomy.core import trafficgraph as TG
    true = {'alpha': 0.6, 'lam': 6.0, 'v_des': 9.0, 't_gap': 2.5}
    rng = np.random.default_rng(0)
    path = L.RefPath.line(0, 0, 0, 500, 0, (-5.25, 3.1))
    samples = []
    for _ in range(300):
        ag = L.make_agent('a', 'car', path, 0.0, 0.0, rng.uniform(1, 14), dict(true, level=1))
        A, D = L.plans(ag)
        R = L.roll(ag, A, D)
        others = []
        for _j in range(rng.integers(0, 3)):
            x0, y0, v = rng.uniform(5, 40), rng.choice([0.0, -3.5]), rng.uniform(0, 12)
            t = np.arange(1, L.N_STEPS + 1) * L.DT_PLAN
            others.append(((x0 + v * t)[None], np.full((1, L.N_STEPS), y0), 4.5, 1.9))
        F = L.plan_features(ag, R, others, A, D)
        P = L.quantal(L.utility(L.NP, ag['params'], F), true['lam'])
        samples.append({'features': F, 'choice': int(rng.choice(len(P), p=P))})
    res = TG.calibrate(samples, steps=150)
    est, se = res['estimates'], res['se']
    for k in ('alpha', 'lam', 'v_des'):
        assert abs(est[k] - true[k]) <= 3.0 * se[k] + 1e-6, (k, est[k], se[k])
    assert abs(est['alpha'] - 0.6) < 0.1 and abs(est['lam'] - 6.0) < 1.5 and abs(est['v_des'] - 9.0) < 0.6
    lo, hi = res['ci95']['t_gap']
    assert lo < true['t_gap'] < hi                                          # weakly identified, but covered
    assert res['n_decisions'] == 300 and 'observed' in res['assumption']


def test_calibrate_cli_on_a_simulation(tmp_path):
    pytest.importorskip('torch')
    assert L.main_levelk(['simulate', '--scene', 'platoon', '--seconds', '4', '--out', str(tmp_path / 'sim')]) == 0
    sim = json.load(open(tmp_path / 'sim' / 'trajectories.json'))
    assert sim['realism']['realistic'] and set(sim['trajectories']) == {a['id'] for a in sim['agents']}
    assert L.main_levelk(['calibrate', '--log', str(tmp_path / 'sim' / 'trajectories.json'), '--steps', '30',
                          '--out', str(tmp_path / 'cal')]) == 0
    cal = json.load(open(tmp_path / 'cal' / 'calibration.json'))
    assert cal['n_decisions'] > 10 and 0.0 <= cal['estimates']['alpha'] <= 1.0 and cal['provenance']
    assert L.main_levelk(['check', '--scene', 'merge', '--seeds', '2', '--seconds', '3',
                          '--out', str(tmp_path / 'chk')]) == 0
    chk = json.load(open(tmp_path / 'chk' / 'check.json'))
    assert len(chk['runs']) == 2 and chk['limits']['jerk_mps3'] == 8.0
    assert L.main_levelk(['nope']) == 2 and L.main_levelk(['--help']) == 0


def test_adversarial_search_lowers_the_predicted_gap():
    pytest.importorskip('torch')
    from falcon_autonomy.core import trafficgraph as TG
    from falcon_autonomy.core.advfeatures import BRAKE_MAX
    rng = np.random.default_rng(1)
    v, g, c = rng.uniform(0, 17, 1500), rng.uniform(0, 60, 1500), rng.uniform(-5, 10, 1500)
    S, rep = TG.fit_surrogate(np.c_[v, g, c], np.clip(L.idm(v, 12.0, np.maximum(g, 0.1), c, 2.0), -8, 2), epochs=250)
    assert rep['mae'] < 0.2 * rep['baseline_mae']                           # the IDM planner is cloned
    init = {k: 0.5 * (lo + hi) for k, (lo, hi) in TG.LEAD_BRAKE_BOUNDS.items()}
    res = TG.adversarial_search(S, init, steps=25)
    assert res['best']['min_gap'] < res['start']['min_gap']
    for k, (lo, hi) in res['bounds'].items():
        assert lo - 1e-9 <= res['best']['values'][k] <= hi + 1e-9
    assert res['bounds']['decel'][1] == BRAKE_MAX and 'not differentiable' in res['note']
    sv = TG.scenario_values(res['best']['values'])
    assert sv['speed'] == pytest.approx(res['best']['values']['speed'] * 3.6)


# ------------------------------------------------------------------------------------------------ in CARLA (fake)
@pytest.fixture
def svc(tmp_path, fake_carla):
    from falcon_autonomy.core.gymenv import make_service
    from falcon_autonomy.core.settings import Settings
    s = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                     str(tmp_path / 'locks'))
    yield s
    s.shutdown()


def _lines(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def test_levelk_scenario_actor_in_the_director(project, svc, tmp_path):
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config
    doc = SC.builtin('lead_brake')
    doc['actors'][0].update(driver='levelk', levelk={'k': 1, 'alpha': 0.5})
    cfg = gym_config(project, 'world-town10', str(tmp_path / 'env'), svc.port, svc.tm_port, scenario=doc,
                     values={'speed': 30.0, 'gap': 20.0})
    env = FalconDriveEnv(cfg, svc, start_timeout_s=60, step_timeout_s=30)
    try:
        env.reset(seed=0)
        for _ in range(60):
            *_x, term, trunc, _i = env.step([0.0, 0.0])
            if term or trunc:
                break
    finally:
        env.close()
    e = _lines(tmp_path / 'env' / 'episodes.jsonl')[-1]
    lead = np.asarray(e['trajectories']['actors']['lead'], float)
    v = np.hypot(lead[:, 4], lead[:, 5])
    # its desired speed is the scenario's: the brake action is ignored (a level-k driver chooses its motion)
    assert abs(v[-1] - 30.0 / 3.6) < 1.0 and v.min() > 30.0 / 3.6 - 1.0
    acc = np.diff(v) / 0.05
    assert np.abs(acc).max() <= 4.0 + 1e-6 and np.abs(np.diff(acc) / 0.05).max() <= 8.0 + 1e-3
    assert 'levelk' not in e.get('scenario', {})


def test_levelk_adversaries_realistic(project, svc, tmp_path):
    from falcon_autonomy.core.adversary import open_adversary_env, run_episode
    env, _ = open_adversary_env(project, 'world-town10', str(tmp_path / 'adv'), svc, 'follower', log_samples=True)
    try:
        rows = [run_episode(env, lambda a, o: None, sd, {'roles': ['cut_in_car', 'filtering_bike'],
                                                         'controller': 'levelk'}) for sd in range(3)]
        with pytest.raises(Exception, match='controller'):
            env.reset(seed=0, options={'roles': ['lead_car'], 'controller': 'brain'})
    finally:
        env.close()
    for e in rows:
        adv = e['adversary']
        assert adv['controller'] == 'levelk' and adv['order'] == []          # driven in the worker, not agents
        for aid, a in adv['actors'].items():
            assert a['levelk'] and not a['learned']
            assert a['realism']['realistic'], (aid, a['realism'], a['rules'])
        assert all(len(v) == e['steps'] for v in e['adversary_actions'].values())
        s = e['adversary_samples']
        assert max(abs(x) for v in s.values() for x in v['jerk']) <= 8.0 + 1e-3


def test_levelk_as_an_evaluation_baseline(project, svc, tmp_path):
    from falcon_autonomy.core import adversary as A
    assert 'levelk' in A.METHODS and 'levelk' not in A.DEFAULT_METHODS
    out = str(tmp_path / 'eval')
    doc = A.evaluate(project, svc, 'world-town10', '', ['follower'], out, episodes=2, methods=('levelk',),
                     rosters=[['lead_car']])
    r = doc['results']['follower']['levelk']
    assert r['episodes'] == 2 and r['realism']['adversaries'] == 2 and r['realism']['unrealistic_share'] == 0.0
    lines = _lines(os.path.join(out, 'episodes.jsonl'))
    assert [e['eval']['method'] for e in lines] == ['levelk'] * 2
    assert all(e['adversary']['controller'] == 'levelk' and e['options']['controller'] == 'levelk' for e in lines)


def test_levelk_background_traffic(project, svc, tmp_path):
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config
    from falcon_autonomy.core.project import Project
    p = Project.load(project.path)
    p.doc['traffic_profiles'][0]['behaviour'].update(model='levelk', levelk={'car': {'k': 1}})
    cfg = gym_config(p, 'world-town10', str(tmp_path / 'env'), svc.port, svc.tm_port, traffic_vehicles=3,
                     tokens=True)
    env = FalconDriveEnv(cfg, svc, start_timeout_s=60, step_timeout_s=30)
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(30):
            obs, *_r = env.step([0.0, 0.3])
    finally:
        env.close()
    run = json.load(open(tmp_path / 'env' / 'run.json'))
    lk = run['traffic']['levelk']
    assert lk['model'] == 'levelk' and lk['vehicles'] == 3 and lk['steps'] >= 30 and lk['decisions'] >= 6
    assert lk['parameters']['car'] == {'level': 1}
    moving = obs['objects'][obs['object_ids'] >= 0]
    assert len(moving) and (np.hypot(moving[:, 13], moving[:, 14]) > 0.5).any()    # their velocity, not CARLA's 0


def test_cli_levelk_entry():
    import falcon_autonomy.cli as cli
    assert cli.main(['levelk', '--help']) == 0
