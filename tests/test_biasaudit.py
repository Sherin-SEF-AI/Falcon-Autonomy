"""Closed-loop bias audits (core.biasaudit) on the fake carla: the environment's token observations, brake lights
and perturbations (workers/gym.py, workers/director.py), a planted shortcut policy (it brakes on the lead's brake
light only) flagged and an honest one (it brakes on the lead's closing speed) not, the input-side arms, black-box
egos limited to the simulation side, the reliance statistics as units, and the CLI."""
import json

import numpy as np
import pytest

from falcon_autonomy.core import biasaudit as BA
from falcon_autonomy.core import drivefeatures as DF
from falcon_autonomy.core import tokenfeatures as TF
from falcon_autonomy.core.gymenv import EnvError, FalconDriveEnv, gym_config, make_service
from falcon_autonomy.core.tokenpolicy import StateExpert


@pytest.fixture
def svc(tmp_path, fake_carla):
    from falcon_autonomy.core.settings import Settings
    s = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                     str(tmp_path / 'locks'))
    yield s
    s.shutdown()


def _lead_row(obs):
    objs = obs['objects']
    m = (objs[:, TF.OI['present']] > 0.5) & (objs[:, TF.OI['dx_m']] > 0) & (np.abs(objs[:, TF.OI['dy_m']]) < 1.5)
    return objs[m][0] if m.any() else None


class Shortcut:
    """Planted shortcut: brakes fully while the vehicle ahead shows its brake light, otherwise holds the target."""
    reads_tokens = True
    name = 'planted shortcut'

    def __init__(self):
        self.x = StateExpert()

    def act(self, obs, names=None):
        steer = float(self.x.act(obs)[0])
        r = _lead_row(obs)
        if r is not None and r[TF.OI['brake_light']] > 0.5:
            return np.array([steer, -1.0], np.float32)
        s = obs['state']
        return np.array([steer, 0.3 if s[DF.FI['speed_mps']] < s[DF.FI['target_speed_mps']] else 0.0], np.float32)


class Honest:
    """Honest: the expert's IDM on the lead's gap and closing speed (no brake light)."""
    reads_tokens = True
    name = 'honest'

    def __init__(self):
        self.x = StateExpert()

    def act(self, obs, names=None):
        return self.x.act(obs)


def test_token_observation_lights_and_perturbations(project, svc, tmp_path):
    cfg = gym_config(project, 'world-town10', str(tmp_path / 'env'), svc.port, svc.tm_port, scenario='lead_brake',
                     values={'gap': 15.0, 'speed': 30.0, 'brake_at': 1.0, 'decel': 6.0},
                     camera={'width': 64, 'height': 48, 'fov': 90.0, 'depth': True}, tokens={'history': 5})
    env = FalconDriveEnv(cfg, svc, start_timeout_s=60, step_timeout_s=30)
    try:
        sp = env.observation_space
        assert sp['objects'].shape == (TF.MAX_OBJECTS, TF.N_OBJ_FEATURES) and sp['ego_history'].shape == (5,)
        assert sp['depth'].shape == (48, 64) and env.ready['tokens']['history'] == 5
        seen = {}
        for pert in (None, {'lead': {'decel': False, 'lights': 'on'}}, {'lead': {'lights': 'off'}}):
            obs, info = env.reset(seed=1, options={'perturb': pert} if pert else None)
            assert sp.contains(obs)
            r = _lead_row(obs)
            assert r is not None and r[TF.OI['class_car']] == 1.0 and 10.0 < r[TF.OI['dx_m']] < 25.0
            lights, speeds = [], []
            for _ in range(50):
                obs, *_x = env.step([0.0, 0.0])
                r = _lead_row(obs)
                lights.append(r[TF.OI['brake_light']])
                speeds.append(r[TF.OI['vx_mps']])
            seen[json.dumps(pert)] = (lights, speeds)
            assert obs['ego_history'].shape == (5,) and np.allclose(obs['ego_history'], 30.0 / 3.6, atol=0.01)
            assert 0.0 < obs['depth'].min() and obs['depth'].max() <= 1000.0
        base, cue, nocue = seen['null'], seen[json.dumps({'lead': {'decel': False, 'lights': 'on'}})], \
            seen[json.dumps({'lead': {'lights': 'off'}})]
        assert base[0][:15] == [0.0] * 15 and base[0][-1] == 1.0 and base[1][-1] < 2.0     # brakes, lights on
        assert cue[0][-1] == 1.0 and min(cue[1]) > 30.0 / 3.6 - 0.01                     # lit, no deceleration
        assert max(nocue[0]) == 0.0 and nocue[1][-1] < 2.0                              # decelerates, never lit
        with pytest.raises(EnvError, match='perturb names'):
            env.reset(seed=1, options={'perturb': {'ghost': {'decel': False}}})
        with pytest.raises(EnvError, match='lights must be'):
            env.reset(seed=1, options={'perturb': {'lead': {'lights': 'dim'}}})
    finally:
        env.close()
    lines = [json.loads(x) for x in open(tmp_path / 'env' / 'episodes.jsonl')]
    assert lines[0]['brake_lights'][0][1:] == ['lead', True] and 'perturb' not in lines[0]
    assert lines[1]['perturb'] == {'lead': {'decel': False, 'lights': 'on'}} and 'brake_lights' not in lines[2]


def test_reliance_statistics():
    rng = np.random.default_rng(0)
    r = BA.reliance(rng.normal(-3.0, 0.2, 8), rng.normal(0.0, 0.05, 8), 500)
    assert r['flagged'] and r['R'] > 10 and r['cue_effect']['ci95'][1] < 0
    r = BA.reliance(np.zeros(8), rng.normal(-3.0, 0.2, 8), 500)
    assert not r['flagged'] and r['R'] == 0.0 and r['cue_effect']['ci95'] == [0.0, 0.0]
    r = BA.reliance(np.zeros(5), np.zeros(5), 100)
    assert r['R'] is None and not r['flagged']
    a, b = BA.reliance(rng.normal(1, 1, 6), rng.normal(2, 1, 6), 300, seed=4), None
    b = BA.reliance(rng.normal(1, 1, 6), rng.normal(2, 1, 6), 300, seed=4)
    assert a != b                                                          # different data …
    c = BA.reliance(np.arange(6.0), np.arange(6.0) + 1, 300, seed=4)
    assert c == BA.reliance(np.arange(6.0), np.arange(6.0) + 1, 300, seed=4)   # … the same data: the same answer


def test_planted_shortcut_flagged_honest_not(project, svc, tmp_path):
    opts = {'seeds': 4, 'max_episode_s': 6.0, 'boot': 500}
    short = BA.audit(project, svc, BA.Ego(Shortcut()), str(tmp_path / 'short'), ['brake_light'], opts)
    honest = BA.audit(project, svc, BA.Ego(Honest()), str(tmp_path / 'honest'), ['brake_light'], opts)
    s, h = short['arms']['brake_light'], honest['arms']['brake_light']
    assert s['flagged'] and short['flagged'] == ['brake_light']
    on = s['outcomes']['brake_onset_s']
    assert on['cue_effect']['ci95'][1] < 0 and on['R'] >= BA.R_FLAG          # the light, not the deceleration
    assert s['outcomes']['collision']['cue_effect']['mean'] < 0             # lights off: it hits the lead
    assert not h['flagged'] and honest['flagged'] == []
    for m, v in h['outcomes'].items():
        assert v['cue_effect']['mean'] == pytest.approx(0.0, abs=1e-9), m      # the light changes nothing
    assert h['outcomes']['brake_onset_s']['cause_effect']['mean'] < -1.0     # the deceleration does
    assert len(short['episodes']) == 16 and {e['seed'] for e in short['episodes']} == set(short['seeds'])
    doc = json.load(open(tmp_path / 'short' / 'audit.json'))
    assert doc['arms']['brake_light']['flagged'] and doc['provenance'] and 'common random numbers' in doc['note']


def test_input_side_arms_and_black_box(project, svc, tmp_path):
    opts = {'seeds': 2, 'max_episode_s': 5.0, 'boot': 200}
    doc = BA.audit(project, svc, BA.Ego(Honest()), str(tmp_path / 'h'), ['velocity', 'autorickshaw'], opts)
    v, a = doc['arms']['velocity'], doc['arms']['autorickshaw']
    assert v['available'] and v['cells'] == ['base', 'history_shift', 'scene_harder']
    assert v['outcomes']['brake_onset_s']['cue_effect']['mean'] == pytest.approx(0.0, abs=1e-9)  # no history use
    assert a['available'] and a['cells'] == ['base', 'removed', 'token_masked']
    hist = [e for e in doc['episodes'] if e['cell'] == 'history_shift']
    assert len(hist) == 2
    bb = BA.audit(project, svc, BA.Ego('follower'), str(tmp_path / 'f'), list(BA.ARMS), opts)
    assert bb['arms']['velocity'] == {'available': False, 'why': bb['arms']['velocity']['why']}
    assert 'black box' in bb['arms']['autorickshaw']['why'] and bb['arms']['brake_light']['available']
    assert not bb['arms']['brake_light']['flagged']                         # the lane follower reads no lights
    with pytest.raises(ValueError, match='unknown arms'):
        BA.audit(project, svc, BA.Ego(Honest()), str(tmp_path / 'x'), ['colour'], opts)


def test_audit_cli_and_introspect_dispatch(project, tmp_path, fake_carla, capsys):
    from falcon_autonomy.core.introspect import main_introspect
    assert main_introspect(['audit', '--help']) == 0
    rc = main_introspect(['audit', project.path, '--policy', 'none', '--out', str(tmp_path / 'a'), '--arms',
                          'brake_light,velocity', '--seeds', '2', '--max-episode-s', '4', '--boot', '100',
                          '--fake-carla', fake_carla])
    out = capsys.readouterr().out
    assert rc == 0, out
    doc = json.load(open(tmp_path / 'a' / 'audit.json'))
    assert doc['ego'] == 'none' and doc['arms']['velocity']['available'] is False
    assert main_introspect(['audit', project.path, '--policy', 'none', '--out', str(tmp_path / 'b'), '--arms',
                            'nope', '--fake-carla', fake_carla]) == 2
