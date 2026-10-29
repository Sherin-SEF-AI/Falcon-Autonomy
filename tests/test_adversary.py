"""Adversarial multi-agent traffic against the fake carla: the shared definitions as units (observations, actions and
their actuator, fault attribution, rewards, realism, placement, the curriculum, the distribution distances); the
multi-agent environment (its API, determinism, failures with fault, scripted episodes, an ego policy driven from the
client); parameter-shared PPO (logs, checkpoints, resume, seeding); the evaluation with its baselines, cases and
replay; the CLI; and the Rare cases page's section."""
import json
import math
import os

import numpy as np
import pytest

from falcon_autonomy.core import adversary as A
from falcon_autonomy.core import advfeatures as AF
from falcon_autonomy.core.gymenv import EnvError, make_service
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401 (fixtures)


# ------------------------------------------------------------------------------------------------- units
def _pose(x, y, yaw, speed=10.0, length=4.6, width=2.0, accel=0.0):
    return {'x': x, 'y': y, 'yaw': yaw, 'speed': speed, 'length': length, 'width': width, 'accel': accel}


def test_features_are_in_the_adversarys_frame_and_finite():
    lane = {'on_lane': True, 'lat': 0.4, 'lane_yaw': 90.0, 'width': 3.5, 'wrong_way': False, 'lanes_left': 1,
            'lanes_right': 7}
    f = AF.features(_pose(0, 0, 90.0, 8.0), _pose(0, 10, 90.0, 12.0), lane, {'s_rel': -10, 'lat': 3.5, 'ego_lat': 0},
                    (-5.0, 0.0), 0.5, [0.2, -0.3], math.inf, 5.2, False, 'autorickshaw')
    assert f.shape == (AF.N_ADV_FEATURES,) and f.dtype == np.float32 and np.all(np.isfinite(f))
    assert f[AF.AI['ego_dx_m']] == pytest.approx(10.0, abs=1e-5)          # yaw 90°: +y is ahead
    assert abs(f[AF.AI['ego_dy_m']]) < 1e-5 and f[AF.AI['ego_rel_vx_mps']] == pytest.approx(4.0, abs=1e-5)
    assert f[AF.AI['other_dy_m']] == pytest.approx(5.0, abs=1e-5)         # −x is to the right when heading +y
    assert f[AF.AI['ttc_s']] == AF.TTC_CAP_S and f[AF.AI['lanes_right']] == 3.0
    assert f[AF.AI['kind_autorickshaw']] == 1.0 and f[AF.AI['kind_car']] == 0.0
    off = AF.features(_pose(0, 0, 0), _pose(100, 0, 0), {'on_lane': False}, {'s_rel': 200, 'lat': 0, 'ego_lat': 0},
                      None, 2.0, [0, 0], 0.0, 100.0, True, 'truck')
    assert off[AF.AI['on_lane']] == 0 and off[AF.AI['ego_dx_m']] == AF.REL_CAP_M and off[AF.AI['kind_car']] == 1
    assert off[AF.AI['time_frac']] == 1.0 and off[AF.AI['ww_allowed']] == 1.0 and off[AF.AI['other_present']] == 0


def test_actions_map_to_bounded_commands():
    lat, acc, ap = AF.command([1.0, -1.0], 'car', 'lateral')
    assert (lat, acc, ap) == (AF.LAT_RATE_MAX, -AF.BRAKE_MAX, [1.0, -1.0])
    assert AF.command([0.0, 1.0], 'autorickshaw', 'lateral')[1] == 2.0               # its class's acceleration
    assert AF.command([5.0, 0.5], 'car', 'lateral')[2] == [1.0, 0.5]                 # clipped
    k = AF.MANOEUVRES.index((-1.0, 'brake'))
    assert AF.command(k, 'car', 'discrete')[:2] == (-AF.LAT_RATE_DISCRETE, AF.BRAKE_DISCRETE)
    assert AF.command([0.3, -0.2], 'car', 'continuous')[:2] == (0.3, -0.2)
    for bad, kind in (([float('nan'), 0], 'lateral'), (9, 'discrete'), ([1, 2, 3], 'lateral'), ([0, 0], 'other')):
        with pytest.raises(ValueError):
            AF.command(bad, 'car', kind)
    act = AF.Actuator(0.05, 0.0, 2.0)
    accs, lats, vl = [], [], []
    for k in range(80):
        lat, a = act.update(1.5 if k < 60 else -1.5, -6.0 if k < 30 else 3.0)
        accs.append(a)
        lats.append(lat)
        vl.append(act.vlat)
    assert max(abs(b - a) for a, b in zip([0.0] + accs, accs)) <= AF.JERK_ACT * 0.05 + 1e-9
    assert max(abs(b - a) for a, b in zip([0.0] + vl, vl)) <= AF.LAT_ACC_ACT * 0.05 + 0.02    # it slows to the limit
    assert 1.9 < max(lats) <= 2.0 and min(accs) == pytest.approx(-6.0)
    u = AF.accel_to_control(10.0, 10.1, 0.05, 0.1)
    assert u['throttle'] == pytest.approx(2.0 / 3.0)
    assert AF.accel_to_control(10.0, 9.8, 0.05, 0)['brake'] == pytest.approx(0.5)


def test_dynamics_count_seconds_over_the_limits():
    d = AF.Dynamics(0.05)
    out = [d.update(v, 0.0) for v in (10.0, 10.1, 10.2, 9.8, 9.8)]
    assert out[0][3] == [] and 'a_long_mps2' in out[3][3] and 'jerk_mps3' in out[3][3]
    assert d.over_s['a_long_mps2'] == pytest.approx(0.05) and len(d.samples['a_long']) == 4
    d2 = AF.Dynamics(0.05)
    d2.update(10.0, 0.0)
    assert d2.update(10.0, 2.0)[1] == pytest.approx(10.0 * math.radians(2.0) / 0.05)    # speed × yaw rate


def _hist(xs, ys, yaw, speed, lane_yaw=0.0, ww=False, on=True, length=4.6, width=2.0):
    n = len(xs)
    yaws = yaw if isinstance(yaw, list) else [yaw] * n
    return {'x': list(xs), 'y': list(ys), 'yaw': yaws, 'speed': [speed] * n, 'lane_yaw': [lane_yaw] * n,
            'wrong_way': [ww] * n, 'on_lane': [on] * n, 'length': length, 'width': width}


def test_fault_attribution_rules():
    dt = 0.05
    t = np.arange(12) * dt
    # F3 rear-end: A follows B in the same lane and runs into it: the follower is at fault
    a = _hist(10 * t, [0.0] * 12, 0.0, 10.0)
    b = _hist(7.85 + 4 * t, [0.0] * 12, 0.0, 4.0)
    f = AF.attribute_fault(a, b, dt)
    assert f['at_fault'] == 'a' and f['rule'] == 'F3' and f['detail']['a_zone'] == 'front'
    # F3 with the cut-in exemption: B came from the next lane within CUT_IN_S, then A hit its rear
    n = 41
    tt = np.arange(n) * dt
    by = list(np.clip(3.5 - 3.5 * tt / 1.0, 0.0, 3.5))
    a2 = _hist(30 + 10 * tt, [0.0] * n, 0.0, 10.0)
    b2 = _hist(30 + 10 * tt + 4.7 - np.linspace(0, 0.2, n), by, 0.0, 10.0)     # just ahead, closing, now in lane
    f = AF.attribute_fault(a2, b2, dt)
    assert f['at_fault'] == 'b' and f['rule'] == 'F3' and f['detail']['struck_was_clear_within_s'] == AF.CUT_IN_S
    # F2 lateral encroachment: B swerves into A from the left while A keeps its lane
    m = 12
    a3 = _hist(10 * t, [0.0] * m, 0.0, 10.0)
    b3 = _hist(10 * t, list(-2.3 + 0.5 * t), [8.0] * m, 10.0)          # heading 8° right of its lane: drifting right
    f = AF.attribute_fault(a3, b3, dt)
    assert f['at_fault'] == 'b' and f['rule'] == 'F2' and f['detail']['b_lateral_toward_mps'] > 1.0
    # F1: the wrong-way one is at fault whatever the geometry
    a4 = _hist(10 * t, [0.0] * m, 0.0, 10.0)
    b4 = _hist(10 - 5 * t, [0.0] * m, 180.0, 5.0, lane_yaw=0.0, ww=True)
    assert AF.attribute_fault(a4, b4, dt)['at_fault'] == 'b' and AF.attribute_fault(b4, a4, dt)['at_fault'] == 'a'
    # F4: side by side, neither moving towards the other, no rear-end: shared
    a5 = _hist(10 * t, [0.0] * m, 0.0, 10.0)
    b5 = _hist(10 * t, [1.99] * m, 0.0, 10.0)
    assert AF.attribute_fault(a5, b5, dt)['at_fault'] == 'shared'


def test_reward_terms_team_mix_and_ttc_records():
    w = AF.DEFAULT_ADV_REWARD
    t = AF.adversary_terms({'ego_collision': 1, 'adv_fault': 1, 'shared_fault': 1, 'off_road': True,
                            'off_road_onset': True, 'wrong_way_over': True, 'dyn_over': 2, 'rules_over': 3}, dt=0.05)
    assert t['ego_collision'] == w['ego_collision'] and t['adv_fault'] == -1.5 * w['adv_fault']
    assert t['off_road'] == pytest.approx(-(w['off_road'] * 0.05 + w['off_road_onset']))
    assert t['dynamics'] == pytest.approx(-2 * w['dynamics'] * 0.05) and t['rules'] == -3 * w['rules']
    assert set(t) == set(AF.ADV_REWARD_TERMS) and t['diversity'] == 0.0
    best, total = 0.0, 0.0
    for ttc in (9.0, 5.0, 6.0, 2.0, 3.0, 2.5):                           # records only: the sum is the minimum's score
        s = AF.ttc_score(ttc)
        total += AF.adversary_terms({'ttc_gain': max(s - best, 0.0)})['ttc']
        best = max(best, s)
    assert total == pytest.approx(w['ttc'] * math.exp(-2.0 / AF.TTC_SCALE_S))
    zero = AF.adversary_terms({})
    r = AF.mix_team({'a': dict(zero, ego_collision=10.0, adv_contact=-10.0), 'b': dict(zero)})
    assert r['a'] == pytest.approx(0.5 * 10 + 0.5 * 5 - 10) and r['b'] == pytest.approx(0.5 * 5)


def test_realism_and_descriptor():
    ok = {'fault_collisions': 0, 'contacts': 0, 'off_road_s': 0.0, 'wrong_way_over_s': 0.0, 'dynamics_over_s': 0.4,
          'rules_over': 0}
    assert AF.realism_ok(ok)
    for k, v in (('fault_collisions', 1), ('contacts', 1), ('off_road_s', 0.05), ('wrong_way_over_s', 0.05),
                 ('dynamics_over_s', AF.DYN_BUDGET_S + 0.05), ('rules_over', 1)):
        assert not AF.realism_ok(dict(ok, **{k: v}))
    assert AF.sector(5, 1) == 'front' and AF.sector(-5, 1) == 'rear' and AF.sector(1, -3) == 'left'
    assert AF.descriptor({'role': 'lead_car', 'kind': 'collision', 'sector': 'front', 'ego': {'speed_mps': 7.0}}) == \
        'lead_car|collision|front|20'


def test_setup_scenarios_are_seeded_valid_and_clear():
    from falcon_autonomy.core import scenarios as SC
    roster = ['cut_in_car', 'filtering_bike', 'wrong_way_scooter']
    a, b = AF.setup_scenario(roster, 7), AF.setup_scenario(roster, 7)
    assert a == b and AF.setup_scenario(roster, 8) != a
    assert not SC.validate(dict(a, params={})) and a['needs'] == ['adjacent_same']
    assert [x['id'] for x in a['actors']] == ['cut_in_car_0', 'filtering_bike_1', 'wrong_way_scooter_2']
    assert 15.0 <= a['ego']['speed_kmh'] <= 25.0 and all(x['speed_kmh'] <= AF.ROLE_SPEED_CAP_KMH for x in a['actors'])
    for seed in range(200):
        acts = AF.setup_scenario(['lead_car', 'autorickshaw_cut_in', 'filtering_bike', 'cut_in_car'], seed)['actors']
        assert not any(AF._overlaps(x, y) for i, x in enumerate(acts) for y in acts[i + 1:] + [AF.EGO_PLACE])
    with pytest.raises(ValueError, match='unknown adversary roles'):
        AF.setup_scenario(['tank'], 0)


def test_curriculum_promotes_by_success_or_episodes():
    c = A.AdvCurriculum({'window': 2, 'threshold': 0.5, 'max_episodes': 4, 'stages': [
        {'name': 'one', 'rosters': [['lead_car']]}, {'name': 'two', 'rosters': [['lead_car', 'cut_in_car']]},
        {'name': 'three', 'rosters': [['lead_car', 'cut_in_car', 'filtering_bike']]}]})
    rng = np.random.default_rng(0)
    assert c.options(rng) == {'roles': ['lead_car']}
    assert not c.record(False) and c.record(True) and c.stage == 1            # 1 of the last 2
    assert c.promotions[-1]['by'] == 'success rate'
    assert [c.record(False) for _ in range(4)] == [False, False, False, True] and c.stage == 2
    assert c.promotions[-1]['by'] == 'max_episodes' and c.promotions[-1]['after_episodes'] == 4
    assert not any(c.record(False) for _ in range(10))                        # the last stage is never left
    c2 = A.AdvCurriculum(c.doc).load(json.loads(json.dumps(c.state())))
    assert c2.stage == 2 and c2.promotions == c.promotions
    with pytest.raises(ValueError, match='unknown roles'):
        A.AdvCurriculum({'stages': [{'name': 'x', 'rosters': [['tank']]}]})
    assert A.load_adv_curriculum('default').doc is A.DEFAULT_ADV_CURRICULUM


def test_distribution_distances():
    assert A.ks_statistic([0, 1, 2], [0, 1, 2]) == 0.0 and A.ks_statistic([0, 0], [1, 1]) == 1.0
    assert A.wasserstein1([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)
    assert A.wasserstein1([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(0.0)
    assert A.wasserstein1(np.zeros(10), np.full(5, 2.5)) == pytest.approx(2.5)
    assert A.ks_statistic([], [1]) is None and A.distribution([])['n'] == 0
    d = A.distribution([-4, 1, 2, 3])
    assert d['max'] == 4 and d['n'] == 4


# ------------------------------------------------------------------------------------------------- the environment
@pytest.fixture
def svc(tmp_path, fake_carla):
    from falcon_autonomy.core.settings import Settings
    s = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                     str(tmp_path / 'locks'))
    yield s
    s.shutdown()


def _env(project, svc, path, ego='follower', **kw):
    kw.setdefault('max_episode_s', 4.0)
    pol = kw.pop('ego_policy', None)
    cfg = A.adversary_config(project, 'world-town10', str(path), svc.port, svc.tm_port, ego=ego, **kw)
    return A.AdversaryEnv(cfg, svc, ego_policy=pol, start_timeout_s=60.0, step_timeout_s=30.0)


def _lines(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def test_api_and_episode_summary(project, svc, tmp_path):
    env = _env(project, svc, tmp_path / 'env', roles=['cut_in_car', 'filtering_bike'], log_samples=True)
    try:
        assert env.adv_feature_names == AF.ADV_FEATURE_NAMES and env.adv_action_kind == 'lateral'
        assert env.adv_action_space.shape == (2,) and env.ready['adversary']['ego'] == 'follower'
        obs, info = env.reset(seed=2)
        assert env.agents == ['cut_in_car_0', 'filtering_bike_1'] and set(obs) == set(env.agents)
        assert all(o.shape == (AF.N_ADV_FEATURES,) and np.all(np.isfinite(o)) for o in obs.values())
        assert info['roles'] == {'cut_in_car_0': 'cut_in_car', 'filtering_bike_1': 'filtering_bike'}
        assert env.ego_obs['state'].shape[0] > 10 and info['mode'] == 'learned'
        with pytest.raises(EnvError, match='no action for adversary'):
            env.step({'cut_in_car_0': [0, 0]})
        with pytest.raises(EnvError, match='finite'):
            env.step({'cut_in_car_0': [0, 0], 'filtering_bike_1': [float('nan'), 0]})
        steps = 0
        while True:
            obs, rew, term, trunc, infos = env.step({a: [0.0, 0.0] for a in env.agents})
            steps += 1
            assert set(rew) == set(env.agents) and all(isinstance(r, float) for r in rew.values())
            assert set(infos['cut_in_car_0']['terms']) == set(AF.ADV_REWARD_TERMS)
            if env._needs_reset:
                break
        assert term['cut_in_car_0'] == term['filtering_bike_1'] and trunc['cut_in_car_0']
        s = env.last_info['episode_summary']
        assert s['termination'] == 'time_limit' and steps == 80 and 'adversary_actions' not in s
        adv = s['adversary']
        assert adv['mode'] == 'learned' and set(adv['actors']) == set(env.agents) and adv['ego_driver'] == 'follower'
        r = adv['actors']['cut_in_car_0']
        assert r['realism']['distance_m'] > 10 and r['rules']['distance_km'] > 0 and r['min_gap_m'] > 0
        with pytest.raises(EnvError, match='unknown reset options'):
            env.reset(seed=1, options={'speed': 3})
        with pytest.raises(EnvError, match='unknown adversary roles'):
            env.reset(seed=1, options={'roles': ['tank']})
    finally:
        env.close()
    line = _lines(env.episodes_path)[0]
    assert len(line['adversary_actions']['cut_in_car_0']) == 80 and line['adversary_actions']['cut_in_car_0'][0] == \
        [0.0, 0.0]
    assert len(line['adversary_samples']['filtering_bike_1']['jerk']) == 80
    assert json.load(open(tmp_path / 'env' / 'run.json'))['status'] == 'ok'


def test_same_seed_and_actions_repeat_and_a_brake_check_fails_a_blind_ego(project, svc, tmp_path):
    env = _env(project, svc, tmp_path / 'env', ego='none', roles=['lead_car'], max_episode_s=8.0)
    acts = [[0.0, -0.6]] * 200                         # 3.6 m/s²: within the realism limit

    def episode(seed):
        obs, info = env.reset(seed=seed)
        out = [obs['lead_car_0'].copy()]
        for a in acts:
            obs, r, te, tr, _ = env.step({'lead_car_0': a})
            out.append((obs['lead_car_0'].copy(), r['lead_car_0']))
            if env._needs_reset:
                break
        return out, env.last_info['episode_summary']
    try:
        a, sa = episode(4)
        b, sb = episode(4)
    finally:
        env.close()
    assert len(a) == len(b) and all(np.array_equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a[1:], b[1:]))
    assert sa['termination'] == 'collision' and sa['adversary']['failed']
    f = sa['adversary']['failures'][0]
    assert f['kind'] == 'collision' and f['fault'] == 'ego' and f['rule'] == 'F3' and f['sector'] == 'front'
    assert f['failure'] and f['realistic'] and sa['adversary']['realistic_failure']
    assert sa['adversary']['actors']['lead_car_0']['reward_terms']['ego_collision'] > 0
    assert sa['adversary']['failures'] == sb['adversary']['failures']


def test_ramming_is_the_adversarys_fault_and_unrealistic(project, svc, tmp_path):
    env = _env(project, svc, tmp_path / 'env', ego='none', roles=['wrong_way_scooter'], max_episode_s=10.0,
               action='discrete')
    try:
        obs, info = env.reset(seed=1)
        hold = AF.MANOEUVRES.index((0.0, 'accel'))
        while not env._needs_reset:
            env.step({'wrong_way_scooter_0': hold})
        s = env.last_info['episode_summary']['adversary']
        r = s['actors']['wrong_way_scooter_0']
        if s['failures']:                             # it rode on in the ego's lane: a head-on, its fault (F1)
            f = s['failures'][0]
            assert f['fault'] == 'wrong_way_scooter_0' and f['rule'] == 'F1' and not f['failure']
            assert r['realism']['fault_collisions'] == 1 and not r['realism']['realistic']
            assert r['reward_terms']['adv_fault'] < 0
        assert r['realism']['wrong_way_s'] > 0 and r['realism']['wrong_way_over_s'] == 0      # its role allows it
        env.reset(seed=1, options={'roles': ['cut_in_car']})
        k = 0
        while not env._needs_reset:                   # straight across the road: off it
            env.step({'cut_in_car_0': AF.MANOEUVRES.index((1.0, 'hold'))})
            k += 1
        s = env.last_info['episode_summary']
        r = s['adversary']['actors']['cut_in_car_0']
        assert r['realism']['off_road_s'] > 0 or r['realism']['fault_collisions'] or r['realism']['rules_over']
        assert not s['adversary']['realistic']
    finally:
        env.close()


def test_scripted_scenarios_are_judged_alike(project, svc, tmp_path):
    env = _env(project, svc, tmp_path / 'env', ego='none', max_episode_s=8.0)
    try:
        obs, info = env.reset(seed=3, options={'scenario': 'lead_brake', 'values': {'gap': 10.0, 'decel': 8.0,
                                                                                    'brake_at': 1.0}})
        assert env.agents == [] and obs == {} and info['tracked'] == ['lead'] and info['roles'] == {'lead': 'lead_car'}
        while not env._needs_reset:
            env.step({})
        s = env.last_info['episode_summary']['adversary']
        assert s['mode'] == 'scripted' and s['failed'] and s['failures'][0]['fault'] == 'ego'
        assert s['actors']['lead']['realism']['dynamics_over_s'] > AF.DYN_BUDGET_S      # 8 m/s²: beyond the bound
        assert not s['realistic'] and not s['realistic_failure']
        assert A.failure_score(env.last_info['episode_summary']) < 1.0
    finally:
        env.close()


def test_an_ego_policy_drives_from_the_client(project, svc, tmp_path):
    import torch
    from falcon_autonomy.core import drivefeatures as DF
    from falcon_autonomy.core.policy import build_net, load_policy, net_spec, save_policy
    torch.manual_seed(0)
    spec = net_spec('mlp', [8], 'tanh', 'tanh')
    net = build_net(spec, DF.N_FEATURES, 2)
    pdir = save_policy(str(tmp_path / 'egopol'), net, {
        'name': 'tiny', 'trained_by': 'bc', 'features': list(DF.FEATURE_NAMES), 'bev': None,
        'normaliser': {'mean': [0.0] * DF.N_FEATURES, 'std': [100.0] * DF.N_FEATURES, 'clip': 5.0},
        'action': {'kind': 'continuous', 'keys': ['steer', 'accel']}, 'net': spec})
    assert A.ego_spec(pdir) == ('agent', os.path.abspath(pdir)) and A.ego_spec('none') == ('none', None)
    with pytest.raises(ValueError):
        A.ego_spec(str(tmp_path / 'nothing'))
    with pytest.raises(ValueError, match='ego_policy'):
        _env(project, svc, tmp_path / 'x', ego='agent')
    env = _env(project, svc, tmp_path / 'env', ego='agent', roles=['lead_car'], ego_policy=load_policy(pdir),
               max_episode_s=1.0)
    try:
        env.reset(seed=0)
        while not env._needs_reset:
            env.step({'lead_car_0': [0.0, 0.0]})
    finally:
        env.close()
    line = _lines(env.episodes_path)[0]
    assert len(line['actions']) == 20 and all(len(a) == 2 for a in line['actions'])
    assert line['adversary']['ego_driver'] == 'agent'


# ------------------------------------------------------------------------------------------------- training
SMALL = {'n_steps': 64, 'n_epochs': 2, 'minibatch': 32, 'checkpoint_every': 1, 'keep_checkpoints': 2}
CUR = {'window': 2, 'threshold': 1.0, 'max_episodes': 1, 'stages': [
    {'name': 'one', 'rosters': [['lead_car']], 'options': {'max_episode_s': 1.0}},
    {'name': 'two', 'rosters': [['lead_car', 'cut_in_car']], 'options': {'max_episode_s': 1.0}}]}


def test_ippo_logs_checkpoints_resumes_and_grows_the_roster(project, svc, tmp_path):
    env = _env(project, svc, tmp_path / 'env', ego='none', log_actions=False)
    out = str(tmp_path / 'train')
    try:
        s = A.train_adversaries(env, out, dict(SMALL, total_updates=2, seed=1), CUR)
        assert s['updates'] == 2 and s['global_step'] == 128 and s['stage'] == 1
        assert s['promotions'][0]['by'] == 'max_episodes'
        ups = _lines(os.path.join(out, 'updates.jsonl'))
        assert [u['update'] for u in ups] == [1, 2] and ups[1]['transitions'] > 64       # two adversaries a step
        for u in ups:
            assert all(math.isfinite(u[k]) for k in ('policy_loss', 'value_loss', 'entropy', 'approx_kl'))
        eps = [e for e in _lines(os.path.join(out, 'episodes.jsonl')) if 'episode' in e and 'returns' in e]
        assert eps[0]['options']['roles'] == ['lead_car'] and eps[-1]['options']['roles'] == ['lead_car', 'cut_in_car']
        assert sorted(os.listdir(os.path.join(out, 'checkpoints'))) == ['ckpt-000001.pt', 'ckpt-000002.pt',
                                                                       'latest.json']
        s2 = A.train_adversaries(env, out, dict(SMALL, total_updates=3, seed=1), CUR, resume=True)
        assert s2['updates'] == 3 and s2['global_step'] == 192 and s2['stage'] == 1
        run = json.load(open(os.path.join(out, 'run.json')))
        assert run['status'] == 'ok' and run['resumed_at_update'] == 2 and run['provenance']['seed'] == 1
        pol = A.load_adversary_policy(os.path.join(out, 'policy'))
        assert pol.doc['training']['updates'] == 3 and pol.rosters() == [['lead_car', 'cut_in_car']]
        a = pol.act(np.zeros(AF.N_ADV_FEATURES, np.float32))
        assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)
        with pytest.raises(ValueError, match='nothing to resume'):
            A.train_adversaries(env, str(tmp_path / 'other'), SMALL, CUR, resume=True)
    finally:
        env.close()


def test_ippo_is_seeded_and_runs_discrete(project, svc, tmp_path):
    rows = []
    for k in range(2):
        env = _env(project, svc, tmp_path / f'env{k}', ego='follower', action='discrete', log_actions=False)
        try:
            A.train_adversaries(env, str(tmp_path / f't{k}'), dict(SMALL, total_updates=1, seed=5), CUR)
        finally:
            env.close()
        rows.append(_lines(os.path.join(tmp_path / f't{k}', 'updates.jsonl'))[0])
    keys = ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction', 'transitions')
    assert [rows[0][k] for k in keys] == [rows[1][k] for k in keys]
    pol = A.load_adversary_policy(str(tmp_path / 't0' / 'policy'))
    assert pol.action['kind'] == 'discrete' and isinstance(pol.act(np.zeros(AF.N_ADV_FEATURES)), int)


# ------------------------------------------------------------------------------------------------- evaluation
def test_evaluation_baselines_cases_and_replay(project, svc, tmp_path):
    env = _env(project, svc, tmp_path / 'env', ego='none', log_actions=False)
    try:
        A.train_adversaries(env, str(tmp_path / 'train'), dict(SMALL, total_updates=1, seed=0), CUR)
    finally:
        env.close()
    out = os.path.join(project.workspace, 'adversary', 'eval1')
    doc = A.evaluate(project, svc, 'world-town10', str(tmp_path / 'train' / 'policy'), ['none', 'follower'], out,
                     episodes=2, seed=3, tm_reference=2, families=('lead_brake', 'cut_in'))
    assert set(doc['results']) == {'none', 'follower'} and not [f for f in doc['failed'] if 'ego' in f]
    for ego in ('none', 'follower'):
        assert set(doc['results'][ego]) == set(A.METHODS)
        for m, r in doc['results'][ego].items():
            assert r['episodes'] == 2 and 0.0 <= r['failure_rate'] <= 1.0
            lo, hi = r['failure_rate_ci95']
            assert lo <= r['failure_rate'] <= hi
            assert r['realism']['adversaries'] >= 2 and 'a_long' in r['realism']['dynamics']
    assert doc['traffic_manager_reference']['a_long']['n'] > 0
    assert 'vs_traffic_manager' in doc['results']['none']['learned']['realism']
    eps = _lines(os.path.join(out, 'episodes.jsonl'))
    assert len(eps) == 2 * 5 * 2 and all('eval' in e for e in eps)
    assert {e['eval']['method'] for e in eps} == set(A.METHODS)
    assert json.load(open(os.path.join(out, 'adversary-eval.json')))['provenance']['seed'] == 3
    fails = _lines(os.path.join(out, 'failures.jsonl'))
    cases = A.list_cases(project.workspace)
    assert len(cases) == sum(1 for f in fails if f['realistic']) and all(c['eval_dir'] == out for c in cases)
    # replay: a learned episode's failure if there is one, else its whole episode
    k = next((f['line'] for f in fails if f['method'] == 'learned' and f['replayable']), None)
    e = eps[k] if k is not None else next(x for x in eps if x['eval']['method'] == 'learned')
    j = next((i for i, f in enumerate(e['adversary']['failures']) if f['failure']), None)
    from falcon_autonomy.core.leaderboard import CHASE_CAMERA
    renv, _ = A.open_adversary_env(project, 'world-town10', str(tmp_path / 'replay-env'), svc, e['eval']['ego'],
                                   action=e['eval']['action'], camera=dict(CHASE_CAMERA, width=64, height=36),
                                   max_episode_s=e['eval']['max_episode_s'])
    try:
        rep = A.replay(e, renv, j, out_dir=str(tmp_path / 'replay'))
    finally:
        renv.close()
    assert rep['frames'] and (rep['match'] is True if j is not None else rep['match'] is None)
    assert A.replayable(dict(e, motion='physics')) and A.replayable(dict(e, eval={'traffic_vehicles': 3}))


def test_cli_train_eval_replay(project, tmp_path, fake_carla):
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    cur = tmp_path / 'cur.json'
    cur.write_text(json.dumps(CUR))
    out = str(tmp_path / 'tr')
    base = [sys.executable, '-m', 'falcon_autonomy', 'adversary']
    r = subprocess.run(base + ['train', project.path, '--out', out, '--ego', 'none', '--curriculum', str(cur),
                               '--updates', '1', '--steps', '32', '--option', 'n_epochs=1', '--fake-carla',
                               fake_carla], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"updates": 1' in r.stdout and os.path.isfile(os.path.join(out, 'policy', 'policy.json'))
    ev = str(tmp_path / 'ev')
    r = subprocess.run(base + ['eval', project.path, '--policy', os.path.join(out, 'policy'), '--out', ev, '--ego',
                               'none', '--episodes', '1', '--baselines', 'idle', '--fake-carla', fake_carla],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert os.path.isfile(os.path.join(ev, 'adversary-eval.json'))
    assert len(_lines(os.path.join(ev, 'episodes.jsonl'))) == 2
    r = subprocess.run(base + ['replay', project.path, '--episodes', os.path.join(ev, 'episodes.jsonl'), '--episode',
                               '0', '--out', str(tmp_path / 'rp'), '--fake-carla', fake_carla],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.load(open(tmp_path / 'rp' / 'replay.json'))['frames']
    r = subprocess.run(base + ['bogus'], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 2 and 'usage' in r.stdout


# ------------------------------------------------------------------------------------------------- the GUI
@pytest.mark.gui
def test_rare_cases_page_lists_adversarial_cases(qapp, window, slot_exceptions):
    ws = window.workspace()
    ev = os.path.join(ws, 'adversary', 'eval-x')
    os.makedirs(ev)
    with open(os.path.join(ev, 'adversary-eval.json'), 'w') as fh:
        json.dump({'schema': A.EVAL_SCHEMA}, fh)
    A.write_cases(ev, [{'ego': 'none', 'method': 'learned', 'line': 0, 'failure': 0, 'seed': 4, 'kind': 'collision',
                        'fault': 'ego', 'actor': 'lead_car_0', 'role': 'lead_car', 'step': 61, 'realistic': True,
                        'replayable': True},
                       {'ego': 'none', 'method': 'random', 'line': 1, 'failure': 0, 'kind': 'collision',
                        'realistic': False}])
    os.makedirs(os.path.join(ws, 'adversary', 'train-1', 'policy'))
    with open(os.path.join(ws, 'adversary', 'train-1', 'policy', 'policy.json'), 'w') as fh:
        json.dump({'schema': A.ADV_POLICY_SCHEMA}, fh)
    page = window.go('rarecases')
    sec = page.adversary
    sec.refresh()
    assert sec.cases.topLevelItemCount() == 1                       # the realistic one only
    it = sec.cases.topLevelItem(0)
    assert [it.text(k) for k in (1, 2, 3, 4, 5, 6)] == ['none', 'learned', 'collision', 'ego', 'lead_car', '61']
    assert sec.policy.count() == 1 and sec.ego.count() >= 3 and 'realistic failures' in sec.state.vals['Cases'].text()
    sec.cases.setCurrentItem(it)
    window.project.path or window.save()
    sec.replay()
    jobs = list(sec.jobs.values())
    assert len(jobs) == 1 and jobs[0].spec.argv[3:5] == ['adversary', 'replay'] and '--failure' in jobs[0].spec.argv
    for j in jobs:
        window.queue.cancel(j)
    assert pump(qapp, lambda: all(j.state in ('cancelled', 'failed', 'done') for j in jobs), timeout=60)
