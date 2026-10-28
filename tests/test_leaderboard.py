"""Leaderboard-style scoring (core.leaderboard) with hand-worked answers, and end to end against the fake carla: a
policy that floors the throttle into a braking lead car is scored over a small route set, and its collision is
replayed — the replay meets the same infraction at the same step, with the chase camera's frames around it."""
import json
import os

import numpy as np
import pytest

from falcon_autonomy.core import leaderboard as LB
from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service


def _ep(rc, counts=None, infractions=(), length=100.0, dist=None, term='time_limit', **kw):
    return dict({'episode': 0, 'seed': 0, 'route_length_m': length, 'route_completion': rc,
                 'distance_m': rc * length if dist is None else dist, 'infraction_counts': counts or {},
                 'infractions': list(infractions), 'termination': term, 'motion': 'kinematic'}, **kw)


def test_scores_by_hand():
    e = _ep(0.9, {'collision_vehicle': 2, 'red_light': 1})
    s = LB.score_episode(e)
    assert s['infraction_penalty'] == pytest.approx(0.6 ** 2 * 0.7)
    assert s['driving_score'] == pytest.approx(0.9 * 0.6 ** 2 * 0.7)
    off = _ep(0.8, {'off_road': 1, 'outside_route_lanes': 1},
              [{'kind': 'off_road', 'distance_m': 5.0}, {'kind': 'outside_route_lanes', 'distance_m': 10.0}])
    s = LB.score_episode(off)
    assert s['route_completion'] == pytest.approx(0.8 - 15.0 / 100.0) and s['infraction_penalty'] == 1.0
    assert s['not_counted_m'] == 15.0 and s['route_completion_raw'] == 0.8
    assert LB.score_episode(_ep(0.1, {'off_road': 1}, [{'kind': 'off_road', 'distance_m': 50.0}]))[
        'route_completion'] == 0.0
    blocked = LB.score_episode(_ep(0.4, {'agent_blocked': 1}, term='agent_blocked'))
    assert blocked['shutdown'] == 'agent_blocked' and blocked['driving_score'] == pytest.approx(0.4)
    eps = [_ep(1.0, dist=200.0), _ep(0.5, {'collision_pedestrian': 1}, dist=300.0, scenario={'id': 'ped'})]
    doc = LB.score(eps, n_boot=200)
    m = doc['metrics']
    assert m['driving_score']['value'] == pytest.approx((1.0 + 0.25) / 2)
    assert m['route_completion']['value'] == pytest.approx(0.75)
    assert m['driving_score']['ci_low'] <= m['driving_score']['value'] <= m['driving_score']['ci_high']
    assert doc['infractions_per_km'] == {'collision_pedestrian': pytest.approx(1 / 0.5)}
    assert doc['by_scenario'] == {'free': 1.0, 'ped': 0.25} and doc['distance_km'] == pytest.approx(0.5)
    assert 'minimum-speed' in doc['not_scored']
    with pytest.raises(LB.LeaderboardError):
        LB.score([])


def test_replayable_refusals():
    ok = _ep(1.0, actions=[[0.0, 0.0]])
    assert LB.replayable(ok) is None
    assert 'physics' in LB.replayable(dict(ok, motion='physics'))
    assert 'traffic' in LB.replayable(dict(ok, traffic_vehicles=10))
    assert 'actions' in LB.replayable(_ep(1.0))


def test_route_set():
    rs = LB.route_set(['lead_brake', 'cut_in'], seeds_per=2, free=1, seed=3)
    assert [r['scenario'] for r in rs] == ['lead_brake', 'lead_brake', 'cut_in', 'cut_in', None]
    assert len({r['seed'] for r in rs}) == 5


@pytest.fixture
def svc(tmp_path, fake_carla):
    from falcon_autonomy.core.settings import Settings
    s = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                     str(tmp_path / 'locks'))
    yield s
    s.shutdown()


def test_evaluate_and_replay_a_collision(project, svc, tmp_path, fake_carla):
    def make(scn, camera=None, out='env'):
        cfg = gym_config(project, 'world-town10', str(tmp_path / out / (scn or 'free')), svc.port, svc.tm_port,
                         scenario=scn, values={'speed': 40.0, 'gap': 10.0, 'decel': 9.0, 'brake_at': 0.5}
                         if scn == 'lead_brake' else None, camera=camera, max_episode_s=8.0)
        return FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0)

    def floor_it(obs):
        return np.array([0.0, 1.0], np.float32)
    routes = [{'scenario': 'lead_brake', 'seed': 1}, {'scenario': 'lead_brake', 'seed': 2}]
    doc = LB.evaluate(floor_it, make, routes, str(tmp_path / 'lb'), n_boot=100)
    assert doc['routes'] == 2 and doc['failed'] == []
    for r in doc['per_route']:
        assert r['infraction_counts'].get('collision_vehicle') == 1
        assert r['infraction_penalty'] == pytest.approx(0.6) and r['driving_score'] < 0.6
    saved = json.loads((tmp_path / 'lb' / 'leaderboard.json').read_text())
    assert saved['schema'] == LB.SCHEMA and 'provenance' in saved
    eps = [json.loads(x) for x in (tmp_path / 'lb' / 'episodes.jsonl').read_text().splitlines()]
    assert len(eps) == 2 and eps[0]['actions'] and eps[0]['traffic_vehicles'] == 0
    e = eps[0]
    k = next(i for i, x in enumerate(e['infractions']) if x['kind'] == 'collision_vehicle')
    env = make('lead_brake', camera=dict(LB.CHASE_CAMERA, width=64, height=36), out='replay-env')
    try:
        rep = LB.replay(e, env, k, before_s=1.0, after_s=0.5, out_dir=str(tmp_path / 'replay'))
    finally:
        env.close()
    step = e['infractions'][k]['step']
    assert rep['match'] is True and rep['steps'][0] == max(0, step - 20)
    assert rep['frames'] and all(os.path.isfile(p) for p in rep['frames'])
    assert os.path.basename(rep['frames'][-1]) == f'{step:06d}.jpg'      # the collision ends the episode there
    # and through the command
    from falcon_autonomy.cli import main
    out = tmp_path / 'replay-cli'
    assert main(['policy', 'replay', project.path, '--episodes', str(tmp_path / 'lb' / 'episodes.jsonl'),
                 '--episode', '1', '--infraction', '0', '--out', str(out), '--fake-carla', fake_carla]) == 0
    assert json.loads((out / 'replay.json').read_text())['match'] is True
    assert main(['policy', 'replay', project.path, '--episodes', str(tmp_path / 'lb' / 'episodes.jsonl'),
                 '--episode', '9', '--out', str(tmp_path / 'x'), '--fake-carla', fake_carla]) == 2
