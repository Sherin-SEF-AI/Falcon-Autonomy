"""PPO and the curriculum: promotion logic and running statistics as units; a few PPO updates against the fake carla
with sane loss and return logs, checkpoints, a resume, a seeded repeat, and the exported policy running in the
environment."""
import json
import math
import os

import numpy as np
import pytest

from falcon_autonomy.core import rl
from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service
from falcon_autonomy.core.policy import RunningMeanStd, load_policy, rollout


# ------------------------------------------------------------------------------------------------- units
def test_curriculum_promotes_on_the_success_rate_over_the_window():
    c = rl.Curriculum({'window': 3, 'threshold': 2 / 3, 'stages': [
        {'name': 'a', 'scenario': None, 'options': {'target_speed_kmh': 20.0}},
        {'name': 'b', 'scenario': 'lead_brake', 'params': {'gap': [20.0, 30.0], 'decel': {'choices': [3.0, 6.0]},
                                                           'speed': 25.0}},
        {'name': 'c', 'scenario': 'cut_in'}]})
    assert not c.record(True, 0) and not c.record(False, 1)              # fewer than `window` episodes: never
    assert c.record(True, 2) and c.stage == 1                            # 2 of the last 3
    assert c.promotions[0] == {'from': 0, 'to': 1, 'episode': 2, 'after_episodes': 3, 'success_rate': 2 / 3}
    for ok in (False, False, True, False):
        assert not c.record(ok)                                          # the window restarts in the new stage
    assert c.record(True) and c.stage == 2                               # the last 3: T F T, 2/3
    for _ in range(10):
        assert not c.record(True)                                        # the last stage is never left
    rng = np.random.default_rng(0)
    c2 = rl.Curriculum(c.doc)
    c2.stage = 1
    o = [c2.options(rng) for _ in range(50)]
    assert all(x['scenario'] == 'lead_brake' and 20.0 <= x['values']['gap'] <= 30.0 for x in o)
    assert {x['values']['decel'] for x in o} == {3.0, 6.0} and all(x['values']['speed'] == 25.0 for x in o)
    rng2 = np.random.default_rng(0)
    assert [c2.options(rng2) for _ in range(50)] == o                   # seeded draws repeat
    st = json.loads(json.dumps(c.state()))
    c3 = rl.Curriculum(c.doc).load(st)
    assert c3.stage == 2 and c3.results == c.results and c3.promotions == c.promotions
    assert rl.Curriculum.single('cut_in').options(rng) == {'scenario': 'cut_in'}
    with pytest.raises(ValueError):
        rl.Curriculum({'stages': []})
    assert rl.load_curriculum('default').stages[0]['scenario'] is None


def test_curriculum_success_criteria():
    c = rl.Curriculum(rl.DEFAULT_CURRICULUM)
    ok = {'infraction_counts': {}, 'termination': 'time_limit', 'progress_m': 80.0}
    assert c.success(ok) and not c.success(dict(ok, progress_m=10.0))
    assert not c.success(dict(ok, termination='off_road'))
    assert not c.success(dict(ok, infraction_counts={'collision_vehicle': 1}))


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    xs = [rng.normal(3.0, 2.0, (k, 4)) for k in (1, 7, 30, 2)]
    r = RunningMeanStd((4,), epsilon=1e-12)
    for x in xs:
        r.update(x)
    allx = np.concatenate(xs)
    assert np.allclose(r.mean, allx.mean(0)) and np.allclose(r.var, allx.var(0))


# ------------------------------------------------------------------------------------------------- PPO on the fake
@pytest.fixture
def svc(tmp_path, fake_carla):
    from falcon_autonomy.core.settings import Settings
    s = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                     str(tmp_path / 'locks'))
    yield s
    s.shutdown()


def _env(project, svc, path, **kw):
    cfg = gym_config(project, 'world-town10', str(path), svc.port, svc.tm_port, log_actions=False, **kw)
    return FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0)


SHORT = rl.Curriculum.single(None, {'max_episode_s': 2.0, 'route_m': 60.0, 'start_speed_kmh': 10.0})
CFG = {'n_steps': 128, 'n_epochs': 3, 'minibatch': 32, 'checkpoint_every': 1, 'keep_checkpoints': 2}


def _jsonl(p):
    with open(p) as fh:
        return [json.loads(x) for x in fh]


def test_ppo_logs_checkpoints_and_resumes(project, svc, tmp_path):
    out = str(tmp_path / 'ppo')
    env = _env(project, svc, tmp_path / 'env')
    try:
        s = rl.train_ppo(env, out, dict(CFG, total_updates=3, seed=1), rl.Curriculum(SHORT.doc))
        assert s['updates'] == 3 and s['global_step'] == 384 and s['episodes'] >= 9
        ups = _jsonl(os.path.join(out, 'updates.jsonl'))
        assert [u['update'] for u in ups] == [1, 2, 3]
        for u in ups:
            assert all(math.isfinite(u[k]) for k in ('policy_loss', 'value_loss', 'entropy', 'approx_kl'))
            assert u['value_loss'] >= 0 and 0 <= u['clip_fraction'] <= 1 and u['approx_kl'] >= -1e-6
            assert u['env_steps_per_s'] > 0 and len(u['action_std']) == 2
        eps = [e for e in _jsonl(os.path.join(out, 'episodes.jsonl')) if 'episode' in e]
        assert [e['episode'] for e in eps] == list(range(len(eps))) and all(e['length'] == 40 for e in eps)
        assert all(e['termination'] == 'time_limit' and math.isfinite(e['return']) for e in eps)
        assert all(abs(sum(e['reward_terms'].values()) - e['return']) < 1e-3 for e in eps)
        assert len({e['seed'] for e in eps}) == len(eps)
        ck = os.listdir(os.path.join(out, 'checkpoints'))
        assert sorted(ck) == ['ckpt-000002.pt', 'ckpt-000003.pt', 'latest.json']       # keep_checkpoints = 2
        n_eps = len(eps)
        s2 = rl.train_ppo(env, out, dict(CFG, total_updates=5, seed=1), rl.Curriculum(SHORT.doc), resume=True)
        assert s2['updates'] == 5 and s2['global_step'] == 640
        ups = _jsonl(os.path.join(out, 'updates.jsonl'))
        assert [u['update'] for u in ups] == [1, 2, 3, 4, 5] and ups[3]['global_step'] == 512
        eps = [e for e in _jsonl(os.path.join(out, 'episodes.jsonl')) if 'episode' in e]
        assert [e['episode'] for e in eps] == list(range(len(eps))) and len(eps) > n_eps
        run = json.load(open(os.path.join(out, 'run.json')))
        assert run['status'] == 'ok' and run['resumed_at_update'] == 3 and run['provenance']['seed'] == 1
        pol = load_policy(os.path.join(out, 'policy'))
        assert pol.doc['trained_by'] == 'ppo' and pol.doc['training']['updates'] == 5 and pol.doc['training']['final']
        doc = rollout(pol, env, 2, seed=7, options=SHORT.current['options'], out_dir=str(tmp_path / 'roll'))
        assert doc['aggregate']['episodes'] == 2 and doc['seeds'] == [7, 8]
        again = rollout(pol, env, 2, seed=7, options=SHORT.current['options'])
        assert [e['return'] for e in again['episodes']] == [e['return'] for e in doc['episodes']]   # deterministic
    finally:
        env.close()


def test_ppo_is_seeded_and_runs_discrete(project, svc, tmp_path):
    runs = []
    for k in range(2):
        env = _env(project, svc, tmp_path / f'env{k}', action='discrete')
        try:
            rl.train_ppo(env, str(tmp_path / f'ppo{k}'), dict(CFG, total_updates=1, seed=3), rl.Curriculum(SHORT.doc))
        finally:
            env.close()
        runs.append(_jsonl(os.path.join(tmp_path / f'ppo{k}', 'updates.jsonl'))[0])
    keys = ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction', 'mean_return_last20')
    assert [runs[0][k] for k in keys] == [runs[1][k] for k in keys]
    pol = load_policy(str(tmp_path / 'ppo0' / 'policy'))
    assert pol.action['kind'] == 'discrete' and pol.action['n'] == 25
    from falcon_autonomy.core.drivefeatures import N_FEATURES
    assert isinstance(pol.act({'state': np.zeros(N_FEATURES, np.float32)}), int)


def test_cli_ppo(project, tmp_path, fake_carla):
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    cur = tmp_path / 'cur.json'
    cur.write_text(json.dumps(SHORT.doc))
    out = str(tmp_path / 'ppo')
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'policy', 'ppo', project.path, '--out', out,
                        '--curriculum', str(cur), '--updates', '2', '--steps', '64', '--option', 'n_epochs=2',
                        '--fake-carla', fake_carla], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"updates": 2' in r.stdout and os.path.isfile(os.path.join(out, 'policy', 'policy.json'))
    assert json.load(open(os.path.join(out, 'env', 'run.json')))['status'] == 'ok'
