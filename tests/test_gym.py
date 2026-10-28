"""The learning environment against the fake carla: the Gymnasium API contract, determinism, reward terms that add
up, termination and infractions, the worker's tidy-up; and the shared definitions (core.drivefeatures) as units."""
import json
import math

import numpy as np
import pytest

from falcon_autonomy.core import drivefeatures as DF
from falcon_autonomy.core.gymenv import EnvError, FalconDriveEnv, gym_config, make_service


@pytest.fixture
def svc(tmp_path, fake_carla):
    from falcon_autonomy.core.settings import Settings
    s = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                     str(tmp_path / 'locks'))
    yield s
    s.shutdown()


def _env(project, svc, tmp_path, name='env', **kw):
    cfg = gym_config(project, 'world-town10', str(tmp_path / name), svc.port, svc.tm_port, **kw)
    return FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0)


def _episodes(env):
    with open(env.episodes_path) as fh:
        return [json.loads(x) for x in fh]


# ------------------------------------------------------------------------------------------------- the API
def test_api_contract_spaces_reset_step_close(project, svc, tmp_path):
    env = _env(project, svc, tmp_path, scenario='lead_brake', values={'gap': 15.0}, bev=True, camera=True)
    try:
        sp = env.observation_space
        assert set(sp.keys()) == {'state', 'bev', 'camera'}
        assert sp['state'].shape == (DF.N_FEATURES,) and sp['state'].dtype == np.float32
        assert sp['bev'].shape == (8, 80, 64) and sp['camera'].shape == (72, 128, 3)
        assert env.action_space.shape == (2,) and np.all(env.action_space.low == -1) and \
            np.all(env.action_space.high == 1)
        assert env.feature_names == DF.FEATURE_NAMES and env.render() is None
        with pytest.raises(EnvError, match='reset'):
            env.step([0.0, 0.0])
        obs, info = env.reset(seed=1)
        assert sp.contains(obs) and isinstance(info, dict) and info['seed'] == 1
        assert info['scenario']['id'] == 'lead_brake' and info['motion'] == 'kinematic'
        assert obs['state'][DF.FI['lead_present']] == 1.0 and obs['bev'].max() == 1
        assert obs['bev'][7].any() and obs['bev'][3].any()            # the ego and the lead (vehicle) drawn
        a = env.action_space.sample()
        assert env.action_space.contains(a)
        obs, r, term, trunc, info = env.step(a)
        assert sp.contains(obs) and isinstance(r, float) and term is False and trunc is False
        assert info['step'] == 1 and set(info['reward_terms']) == set(DF.REWARD_TERMS)
        obs, *_ = env.step(np.array([5.0, -7.0]))                       # out of range: clipped
        assert obs['state'][DF.FI['prev_steer']] == 1.0 and obs['state'][DF.FI['prev_accel']] == -1.0
        with pytest.raises(EnvError, match='finite'):
            env.step([float('nan'), 0.0])
        with pytest.raises(EnvError, match='unknown reset options'):
            env.reset(seed=1, options={'speed': 3})
        with pytest.raises(EnvError, match='no start on this map'):
            env.reset(seed=1, options={'scenario': 'unprotected_turn'})    # the fake's road has no junction
        obs, info = env.reset(seed=1)                                  # still usable after refused commands
        assert info['episode'] == 1
    finally:
        env.close()
    env.close()                                                        # idempotent
    assert env.exit_status == 'done'
    run = json.load(open(tmp_path / 'env' / 'run.json'))
    assert run['status'] == 'ok' and run['performance']['steps'] == 2


def test_discrete_actions(project, svc, tmp_path):
    env = _env(project, svc, tmp_path, action='discrete', max_episode_s=0.5)
    try:
        assert env.action_space.n == len(DF.DISCRETE_ACTIONS) == 25
        env.reset(seed=0)
        k = DF.DISCRETE_ACTIONS.index((0.2, 0.7))
        obs, r, term, trunc, info = env.step(k)
        assert info['action_applied'] == [0.2, 0.7]
        with pytest.raises(EnvError, match='not in 0'):
            env.step(25)
        for _ in range(9):
            obs, r, term, trunc, info = env.step(0)
        assert trunc and not term and info['termination'] == 'time_limit'
        with pytest.raises(EnvError, match='reset'):
            env.step(0)
    finally:
        env.close()
    ep = _episodes(env)[0]
    assert ep['actions'][0] == k and ep['truncated'] and ep['steps'] == 10


# ------------------------------------------------------------------------------------------------- determinism
def test_same_seed_and_actions_give_the_same_episode(project, svc, tmp_path):
    env = _env(project, svc, tmp_path, scenario='cut_in', bev=True)
    acts = np.random.default_rng(0).uniform([-0.05, -0.2], [0.05, 0.6], size=(60, 2))

    def episode(seed):
        obs, info = env.reset(seed=seed)
        out = [(obs['state'].copy(), obs['bev'].copy(), 0.0)]
        for a in acts:
            obs, r, term, trunc, info = env.step(a)
            out.append((obs['state'].copy(), obs['bev'].copy(), r))
            if term or trunc:
                break
        return out, info
    try:
        a, ia = episode(5)
        b, ib = episode(5)
        c, _ = episode(6)
        s1 = env.reset()[1]['seed']                       # unseeded resets draw from the generator the seed set
        env.reset(seed=6)
        s2 = env.reset()[1]['seed']
    finally:
        env.close()
    assert len(a) == len(b) > 30
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) and x[2] == y[2] for x, y in zip(a, b))
    assert ia['ego'] == ib['ego']
    assert not all(np.array_equal(x[0], y[0]) for x, y in zip(a, c))       # another seed: another cut-in
    assert s1 == s2
    eps = _episodes(env)
    assert eps[0]['actions'] == eps[1]['actions'] and eps[0]['scenario']['values'] == eps[1]['scenario']['values']


# ------------------------------------------------------------------------------------------------- rewards
def test_reward_terms_add_up_to_the_reward_and_the_return(project, svc, tmp_path):
    env = _env(project, svc, tmp_path, scenario='lead_brake', values={'speed': 30.0}, max_episode_s=3.0)
    rewards = []
    try:
        env.reset(seed=2)
        rng = np.random.default_rng(1)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(rng.uniform([-0.05, -0.2], [0.05, 1.0]))
            assert list(info['reward_terms']) == list(DF.REWARD_TERMS)
            assert sum(info['reward_terms'][k] for k in DF.REWARD_TERMS) == r
            rewards.append(r)
            done = term or trunc
    finally:
        env.close()
    summ = info['episode_summary']
    assert summ['return'] == pytest.approx(sum(rewards), abs=1e-5)
    assert sum(summ['reward_terms'].values()) == pytest.approx(summ['return'], abs=1e-4)
    assert summ['reward_terms']['progress'] > 0 and summ['progress_m'] > 5.0


def test_reward_terms_unit():
    w = DF.DEFAULT_REWARD
    q = {'ds': 0.5, 'speed': 8.0, 'target': 10.0, 'lateral': 0.875, 'lane_width': 3.5, 'heading_err': 0.1,
         'jerk': 1.0, 'steer_rate': 0.5, 'off_road': True, 'wrong_way': False, 'outside_lanes': True}
    t = DF.reward_terms(q, [{'kind': 'collision_pedestrian'}, {'kind': 'off_road'}], dt=0.05)
    assert t['progress'] == pytest.approx(0.05)
    assert t['speed'] == pytest.approx(-w['speed'] * 0.2 * 0.05)
    assert DF.reward_terms(dict(q, speed=12.0), [], dt=0.05)['speed'] == pytest.approx(
        -w['speed'] * DF.OVERSPEED_FACTOR * 0.2 * 0.05)                 # overspeed costs more
    assert t['lane_offset'] == pytest.approx(-w['lane_offset'] * 0.25 * 0.05)
    assert t['heading'] == pytest.approx(-w['heading'] * 0.1 * 0.05)
    assert t['jerk'] == pytest.approx(-w['jerk'] * 0.5 * 0.05)
    assert t['collision'] == -w['collision_pedestrian']
    assert t['off_road'] == pytest.approx(-w['off_road'] * 0.05 - w['off_road_onset'])
    assert t['outside_lanes'] == pytest.approx(-w['outside_lanes'] * 0.05)


# ------------------------------------------------------------------------------------------------- termination
def test_collision_terminates_and_is_recorded(project, svc, tmp_path):
    env = _env(project, svc, tmp_path, scenario='lead_brake',
               values={'speed': 40.0, 'gap': 10.0, 'decel': 9.0, 'brake_at': 0.5})
    try:
        env.reset(seed=1)
        for k in range(200):
            obs, r, term, trunc, info = env.step([0.0, 1.0])
            if term or trunc:
                break
    finally:
        env.close()
    assert term and info['termination'] == 'collision'
    ev = [e for e in info['infractions'] if e['kind'] == 'collision_vehicle']
    assert len(ev) == 1 and ev[0]['step'] == info['step'] and ev[0]['detail']['source'] in ('box', 'sensor')
    assert isinstance(ev[0]['actors'][0], int) and set(ev[0]['ego']) == {'x', 'y', 'z', 'yaw', 'speed_mps'}
    assert info['reward_terms']['collision'] == -DF.DEFAULT_REWARD['collision']
    summ = _episodes(env)[0]
    assert summ['termination'] == 'collision' and summ['terminated'] and not summ['success']
    assert summ['infraction_counts']['collision_vehicle'] == 1 and summ['infractions'][0]['kind'] == 'collision_vehicle'


def test_off_road_and_lane_infractions(project, svc, tmp_path):
    env = _env(project, svc, tmp_path, start_speed_kmh=20.0)
    try:
        obs, info = env.reset(seed=0)
        start = info['start']
        for k in range(300):
            obs, r, term, trunc, info = env.step([0.25, 0.3])
            if term or trunc:
                break
    finally:
        env.close()
    assert term and info['termination'] == 'off_road'
    summ = _episodes(env)[0]
    kinds = [e['kind'] for e in summ['infractions']]
    assert 'off_road' in kinds
    off = next(e for e in summ['infractions'] if e['kind'] == 'off_road')
    assert off['detail']['max_offroad_m'] > DF.DEFAULT_THRESHOLDS['offroad_terminate_m'] and off['end_step'] >= \
        off['step']
    if start['y'] == 0.0:                                      # from the inner lane: through the next lane first
        assert 'outside_route_lanes' in kinds


def test_red_light_run_is_recorded_and_observed(project, svc, tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_CARLA_LIGHT_STATE', 'Red')
    env = _env(project, svc, tmp_path, start_speed_kmh=30.0, target_speed_kmh=30.0)
    try:
        for seed in range(40):                                 # a start on a +x lane before the stop line (x = 8)
            obs, info = env.reset(seed=seed)
            if info['start']['yaw'] == 0.0 and info['start']['x'] < 0.0:
                break
        assert info['traffic_lights_on_route'] == 1
        assert obs['state'][DF.FI['tl_known']] == 1.0
        seen = []
        for k in range(200):
            st = obs['state']
            if st[DF.FI['tl_red']] == 1.0:
                seen.append(float(st[DF.FI['tl_distance_m']]))
            obs, r, term, trunc, info = env.step([0.0, 0.0])
            if any(e['kind'] == 'red_light' for e in info['infractions']):
                break
    finally:
        env.close()
    ev = [e for e in info['infractions'] if e['kind'] == 'red_light']
    assert ev and ev[0]['actors'] == [50] and info['reward_terms']['red_light'] == -DF.DEFAULT_REWARD['red_light']
    assert seen and all(b < a for a, b in zip(seen, seen[1:]))          # the red light comes closer, seen all along


def test_blocked_agent(project, svc, tmp_path):
    env = _env(project, svc, tmp_path, thresholds={'blocked_s': 1.0})
    try:
        env.reset(seed=3)
        for k in range(100):
            obs, r, term, trunc, info = env.step([0.0, -1.0])
            if term or trunc:
                break
    finally:
        env.close()
    assert term and info['termination'] == 'agent_blocked' and k + 1 == 21       # slow since step 1 (t = 0.05)
    assert info['infractions'][0]['kind'] == 'agent_blocked'


# ------------------------------------------------------------------------------------------------- tidy-up
def test_worker_removes_each_episode_and_tidies(project, svc, tmp_path):
    env = _env(project, svc, tmp_path, scenario='jaywalker_behind_bus', camera=True)
    try:
        for seed in (1, 2, 3):
            obs, info = env.reset(seed=seed)
            assert info['stale_actors'] == 0 and info['background_actors'] == 0
            for _ in range(3):
                env.step([0.0, 0.2])
    finally:
        env.close()
    run = json.load(open(tmp_path / 'env' / 'run.json'))
    assert run['status'] == 'ok' and run['tidy_check'] == {'falcon_actors_left': 0}
    eps = _episodes(env)
    assert [e['termination'] for e in eps] == ['interrupted'] * 3 and all(e['steps'] == 3 for e in eps)


def test_cancel_and_lost_client_end_the_worker(project, svc, tmp_path):
    env = _env(project, svc, tmp_path)
    env.reset(seed=0)
    env.conn.close()                                               # the client goes away without closing
    assert svc.queue.wait(env.job, 30)
    run = json.load(open(tmp_path / 'env' / 'run.json'))
    assert run['status'] == 'failed' and 'closed its socket' in run['error']
    assert run['tidy_check'] == {'falcon_actors_left': 0}
    env.close()


# ------------------------------------------------------------------------------------------------- units
def test_route_projection_and_curvature():
    r = 50.0                                                   # a right-hand arc (CARLA: yaw grows to the right)
    a = np.radians(np.arange(0.0, 90.0, 0.5))
    pts = np.c_[r * np.sin(a), r - r * np.cos(a)]
    R = DF.Route(pts, np.full(len(pts), 3.5))
    assert R.length == pytest.approx(r * math.radians(89.5), rel=1e-4)
    s, lat, h = R.project(pts[40, 0] + math.sin(a[40]), pts[40, 1] - math.cos(a[40]))     # 1 m to its left
    assert lat == pytest.approx(-1.0, abs=0.01) and s == pytest.approx(r * a[40], rel=1e-3)
    s, lat, h = R.project(0.0, 1.0)                            # 1 m to the right of the start
    assert s == pytest.approx(0.0, abs=0.01) and lat == pytest.approx(1.0, abs=1e-4) and h == pytest.approx(0.25,
                                                                                                    abs=0.01)
    assert R.curvature(30.0) == pytest.approx(1.0 / r, rel=0.02)
    assert R.truncated(20.0).length == pytest.approx(20.0)


def test_featuriser_lead_vru_light_and_progress():
    R = DF.Route([[0.0, 0.0], [200.0, 0.0]], [3.5, 3.5])
    f = DF.Featuriser(R, 10.0, 0.05)
    ego = {'x': 10.0, 'y': 0.3, 'yaw': 5.0, 'speed_mps': 8.0, 'length': 4.0, 'width': 2.0}
    car = {'id': 7, 'x': 40.0, 'y': 0.5, 'yaw': 0.0, 'vx': 3.0, 'vy': 0.0, 'extent': [2.0, 1.0, 0.8],
           'category': 'vehicle.car'}
    side = dict(car, id=8, x=30.0, y=4.0)                       # the next lane: not a lead
    ped = {'id': 9, 'x': 25.0, 'y': -3.0, 'yaw': 90.0, 'vx': 0.0, 'vy': 1.5, 'extent': [0.2, 0.2, 0.9],
           'category': 'human.pedestrian.adult'}
    x, facts = f(ego, [car, side, ped], [(60.0, 'Red')], (0.1, -0.2))
    F = dict(zip(DF.FEATURE_NAMES, x.tolist()))
    assert F['lateral_offset_m'] == pytest.approx(0.3, abs=1e-6)
    assert F['heading_error_rad'] == pytest.approx(math.radians(5.0), abs=1e-6)
    assert F['lead_present'] == 1 and F['lead_gap_m'] == pytest.approx(30.0 - 2.0 - 2.0, abs=1e-4)
    v_along = 8.0 * math.cos(math.radians(5.0))
    assert F['lead_closing_mps'] == pytest.approx(v_along - 3.0, abs=1e-5)
    assert F['lead_ttc_s'] == pytest.approx(26.0 / (v_along - 3.0), abs=1e-4)
    assert F['vru_present'] == 1 and F['vru_lateral_m'] == pytest.approx(-3.0) and F['vru_lat_speed_mps'] == 1.5
    assert F['tl_known'] == 1 and F['tl_red'] == 1 and F['tl_distance_m'] == pytest.approx(48.0)
    assert F['route_progress'] == pytest.approx(0.05) and F['prev_accel'] == pytest.approx(-0.2)
    x2, _ = f(dict(ego, speed_mps=8.5), [], None)               # no actors, lights unknown
    F2 = dict(zip(DF.FEATURE_NAMES, x2.tolist()))
    assert F2['lead_present'] == 0 and F2['lead_gap_m'] == DF.LEAD_RANGE_M and F2['tl_known'] == 0
    assert F2['accel_mps2'] == pytest.approx(10.0)


def test_actions_lines_boxes_and_bev_bits():
    c, ap = DF.to_control([0.3, -0.4])
    assert c['brake'] == pytest.approx(0.4) and c['throttle'] == 0 and ap == [0.3, -0.4]
    assert DF.control_to_action({'throttle': 0.0, 'steer': 0.3, 'brake': 0.4}) == pytest.approx(ap)
    with pytest.raises(ValueError):
        DF.to_control([1.0], 'continuous')
    assert DF.crossed_line((7.0, 0.5), (8.2, 0.5), (8.0, 0.0), 0.0, 1.75)
    assert not DF.crossed_line((8.2, 0.5), (7.0, 0.5), (8.0, 0.0), 0.0, 1.75)          # backwards
    assert not DF.crossed_line((7.0, 3.0), (8.2, 3.0), (8.0, 0.0), 0.0, 1.75)          # another lane
    assert DF.in_box(1.5, 0.5, 0.0, 0.0, 90.0, 1.0, 2.0) and not DF.in_box(0.5, 1.5, 0.0, 0.0, 90.0, 1.0, 2.0)
    bits = np.random.default_rng(0).integers(0, 256, (2, 5, 6), dtype=np.uint8)
    assert np.array_equal(DF.pack_bev(DF.unpack_bev(bits)), bits)


def test_infraction_monitor_conditions_rearm_stops_and_route_end():
    m = DF.InfractionMonitor(0.05, {'blocked_s': 0.5})
    ego = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'speed_mps': 5.0}
    base = {'speed': 5.0, 's': 1.0, 'route_length': 100.0, 'travel_m': 0.25}
    new, why = m.update(1, 0.05, ego, dict(base, contacts=[(5, 'vehicle', 'box')]))
    assert [e['kind'] for e in new] == ['collision_vehicle'] and why == 'collision'
    new, _ = m.update(2, 0.10, ego, dict(base, contacts=[(5, 'vehicle', 'sensor')]))
    assert new == []                                            # the same contact: once
    new, _ = m.update(40, 2.0, ego, dict(base, contacts=[(5, 'vehicle', 'box')]))
    assert [e['kind'] for e in new] == ['collision_vehicle']    # out of contact > 1 s: a new collision
    m.update(41, 2.05, ego, dict(base, wrong_way=True, outside_lanes=True))
    m.update(42, 2.10, ego, dict(base, wrong_way=True, outside_lanes=True))
    m.update(43, 2.15, ego, dict(base, outside_lanes=True))
    ww = [e for e in m.events if e['kind'] == 'wrong_way'][0]
    ol = [e for e in m.events if e['kind'] == 'outside_route_lanes'][0]
    assert ww['step'] == 41 and ww['end_step'] == 43 and ww['distance_m'] == pytest.approx(0.25)
    assert ol['step'] == 43                                     # wrong way is not also outside the route's lanes
    m.update(44, 2.2, ego, dict(base, stop_inside={77: True}))
    new, _ = m.update(45, 2.25, ego, dict(base))
    assert [e['kind'] for e in new] == ['stop_sign'] and new[0]['actors'] == [77]
    m.update(46, 2.3, ego, dict(base, stop_inside={78: True}, speed=0.05))
    new, _ = m.update(47, 2.35, ego, dict(base))
    assert new == []                                            # it stopped inside
    for k in range(12):
        new, why = m.update(50 + k, 2.5 + 0.05 * k, ego, dict(base, speed=0.0))
    assert why == 'agent_blocked' or any(e['kind'] == 'agent_blocked' for e in m.events)
    new, why = m.update(70, 3.5, ego, dict(base, s=99.0))
    assert why == 'route_complete'
    m.finish(70, 3.5)
    assert not m.open and m.counts()['collision_vehicle'] == 2
    assert DF.episode_success({'infraction_counts': {}, 'termination': 'time_limit', 'progress_m': 30.0},
                              min_progress_m=20)
    assert not DF.episode_success({'infraction_counts': {'off_road': 1}, 'termination': 'time_limit',
                                   'progress_m': 30.0})


def test_shim_spaces():
    from falcon_autonomy.core import gymenv as G
    b = G.Box(-1.0, 1.0, (3,), np.float32, seed=4)
    s = b.sample()
    assert b.contains(s) and s.dtype == np.float32 and not b.contains(np.array([2.0, 0, 0], np.float32))
    assert np.array_equal(G.Box(-1.0, 1.0, (3,), np.float32, seed=4).sample(), s)
    u = G.Box(-np.inf, np.inf, (2,), np.float32, seed=0)
    assert u.contains(u.sample())
    d = G.Discrete(5, seed=1)
    assert d.contains(d.sample()) and not d.contains(5) and not d.contains(1.5)
    D = G.Dict({'a': b, 'b': d}, seed=2)
    assert D.contains(D.sample()) and set(D.keys()) == {'a', 'b'}
