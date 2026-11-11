"""The token transformer policy (core.tokenpolicy): its architecture and policy folder (core.policy net kind
"token"), the shared token definitions (core.tokenfeatures), the privileged expert, behaviour cloning in the
environment on the fake carla and a rollout of the cloned policy through `falcon policy`."""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from falcon_autonomy.core import drivefeatures as DF  # noqa: E402
from falcon_autonomy.core import tokenfeatures as TF  # noqa: E402
from falcon_autonomy.core import tokenpolicy as TP  # noqa: E402

SMALL = {'d_model': 64, 'heads': 8, 'layers': 4, 'ff': 128}


def make_obs(seed=0, cam=(72, 128)):
    """A synthetic token observation: a car ahead with its brake light on, an autorickshaw beside, a pedestrian."""
    rng = np.random.default_rng(seed)
    state = np.zeros(DF.N_FEATURES, np.float32)
    state[DF.FI['speed_mps']], state[DF.FI['target_speed_mps']] = 8.0, 8.3
    state[DF.FI['lane_width_m']], state[DF.FI['tl_distance_m']] = 3.5, DF.TL_RANGE_M
    ego = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
    rows = [{'id': 11, 'x': 15.0, 'y': 0.2, 'yaw': 0.0, 'vx': 5.0, 'vy': 0.0, 'extent': [2.3, 1.0, 0.8],
             'category': 'vehicle.car', 'brake_light': True},
            {'id': 12, 'x': 6.0, 'y': -3.4, 'yaw': 5.0, 'vx': 9.0, 'vy': 0.5, 'extent': [1.3, 0.65, 0.85],
             'category': 'vehicle.car', 'subclass': 'autorickshaw (stand-in)'},
            {'id': 13, 'x': 25.0, 'y': 4.0, 'yaw': 90.0, 'vx': 0.0, 'vy': 1.2, 'extent': [0.3, 0.3, 0.9],
             'category': 'human.pedestrian.adult'}]
    objs, ids = TF.object_tokens(ego, rows, TF.spec(True))
    return {'state': state, 'objects': objs, 'object_ids': ids,
            'ego_history': np.linspace(7.5, 8.0, TF.HISTORY_STEPS).astype(np.float32),
            'camera': rng.integers(0, 255, (cam[0], cam[1], 3)).astype(np.uint8)}


def make_policy(tmp_path, spec=None, seed=0, name='pol'):
    from falcon_autonomy.core.policy import load_policy, save_policy
    torch.manual_seed(seed)
    sp = TP.token_spec(dict(SMALL, **(spec or {})))
    net = TP.build(sp)
    feats = TP.state_features()
    doc = {'name': 'test token policy', 'trained_by': 'token-bc', 'features': feats,
           'normaliser': {'mean': [0.0] * len(feats), 'std': [1.0] * len(feats), 'clip': 10.0}, 'bev': None,
           'camera': {'width': 128, 'height': 72, 'fov': 90.0, 'pose': {'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0,
                                                                         'pitch': 0.0, 'yaw': 0.0}}
           if sp['image'] else None,
           'tokens': sp['tokens'], 'action': {'kind': 'continuous', 'keys': list(DF.ACTION_KEYS)}, 'net': sp}
    save_policy(str(tmp_path / name), net, doc)
    return load_policy(str(tmp_path / name))


def test_object_tokens_definition():
    obs = make_obs()
    o, ids = obs['objects'], obs['object_ids']
    assert o.shape == (TF.MAX_OBJECTS, TF.N_OBJ_FEATURES) and o.dtype == np.float32
    assert list(ids[:4]) == [12, 11, 13, -1]                                # nearest first, then empty rows
    auto, car, ped = o[0], o[1], o[2]
    assert auto[TF.OI['class_autorickshaw']] == 1 and car[TF.OI['class_car']] == 1
    assert ped[TF.OI['class_pedestrian']] == 1 and ped[TF.OI['brake_light']] == 0
    assert car[TF.OI['brake_light']] == 1 and car[TF.OI['dx_m']] == pytest.approx(15.0)
    assert car[TF.OI['length_m']] == pytest.approx(4.6) and o[3:].sum() == 0
    o2, _ = TF.object_tokens({'x': 0, 'y': 0, 'yaw': 90.0}, [{'id': 1, 'x': 0.0, 'y': 10.0, 'yaw': 90.0, 'vx': 0.0,
                                                               'vy': 3.0, 'category': 'vehicle.motorcycle'}],
                             TF.spec({'max_objects': 2}))
    assert o2[0, TF.OI['dx_m']] == pytest.approx(10.0) and o2[0, TF.OI['vx_mps']] == pytest.approx(3.0)
    assert o2[0, TF.OI['class_two_wheeler']] == 1 and abs(o2[0, TF.OI['dy_m']]) < 1e-9
    h = TF.SpeedHistory(3)
    assert h.push(2.0).tolist() == [2.0] * 3 and h.push(3.0).tolist() == [2.0, 2.0, 3.0]
    assert TF.with_brake(1, True) == 9 and TF.with_brake(9, False) == 1 and TF.brake_light_on(8)
    with pytest.raises(ValueError, match='unknown'):
        TF.spec({'objects': 3})


def test_architecture_and_policy_folder(tmp_path):
    pol = make_policy(tmp_path)
    net = pol.net
    assert pol.uses_tokens and pol.uses_camera and not pol.uses_bev
    assert len(net.blocks) == 4 and all(b.attn.h == 8 for b in net.blocks) and net.plan.h == 8
    assert isinstance(net.trunk[-1], type(__import__('torchvision').models.resnet18().layer3))
    obs = make_obs()
    a1, a2 = pol.act(obs), pol.act(obs)
    assert a1.shape == (2,) and np.all(np.abs(a1) <= 1) and np.array_equal(a1, a2)
    lay = net.layout(TF.MAX_OBJECTS)
    assert lay['grid'] == (5, 8) and lay['patch'] == (0, 40) and lay['object'] == (40, 56) and lay['n'] == 58
    # an empty object row is hidden: changing its (zero) features changes nothing
    obs2 = dict(obs, objects=obs['objects'].copy())
    obs2['objects'][10, TF.OI['dx_m']] = 30.0
    assert np.array_equal(pol.act(obs2), a1)
    with pytest.raises(ValueError, match='whole observations'):
        pol.act_batch(obs['state'][None])
    with pytest.raises(ValueError, match='object tokens'):
        pol.act({'state': obs['state']})
    # no image: objects, route and ego only
    p2 = make_policy(tmp_path, {'image': False}, name='noimg')
    assert not p2.uses_camera and p2.net.layout(TF.MAX_OBJECTS)['patch'] == (0, 0)
    assert p2.act({k: v for k, v in obs.items() if k != 'camera'}).shape == (2,)
    with pytest.raises(ValueError, match='multiple'):
        TP.token_spec({'d_model': 30, 'heads': 8})
    with pytest.raises(ValueError, match='not pinned'):
        TP.build(TP.token_spec(dict(SMALL, trunk_weights='resnet18-imagenet1k-v1')))


def test_expert_brakes_for_the_lead_and_keeps_the_lane():
    x = StateExpert = TP.StateExpert()
    s = np.zeros(DF.N_FEATURES, np.float32)
    s[DF.FI['speed_mps']], s[DF.FI['target_speed_mps']], s[DF.FI['tl_distance_m']] = 8.0, 8.0, DF.TL_RANGE_M
    free = StateExpert.act({'state': s})
    s2 = s.copy()
    s2[DF.FI['lead_present']], s2[DF.FI['lead_gap_m']], s2[DF.FI['lead_closing_mps']] = 1.0, 6.0, 4.0
    s3 = s.copy()
    s3[DF.FI['lateral_offset_m']] = 0.8                                   # right of the centre: steer left
    assert abs(free[1]) < 0.05 and x.act({'state': s2})[1] == -1.0 and x.act({'state': s3})[0] < 0


def test_bc_in_the_environment_and_rollout(project, tmp_path, fake_carla):
    """Clone the expert on the fake (two short episodes, two epochs), then roll the policy out with the CLI."""
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service
    from falcon_autonomy.core.policy import load_policy, main_policy
    from falcon_autonomy.core.settings import Settings
    svc = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                       str(tmp_path / 'locks'))
    try:
        cfg = gym_config(project, 'world-town10', str(tmp_path / 'env'), svc.port, svc.tm_port,
                         camera={'width': 128, 'height': 72, 'fov': 90.0}, tokens=True)
        env = FalconDriveEnv(cfg, svc, start_timeout_s=60, step_timeout_s=30)
        try:
            m = TP.train_bc(env, str(tmp_path / 'pol'), {'episodes': 2, 'epochs': 2, 'max_episode_s': 1.5,
                                                         'scenarios': ['lead_brake', 'free'], 'dagger': 1},
                            SMALL)
        finally:
            env.close()
    finally:
        svc.shutdown()
    assert m['samples']['train'] > 0 and m['samples']['val'] > 0 and len(m['history']) == 2
    assert m['dagger'][0]['round'] == 1 and m['dagger'][0]['beta'] == 0.5
    pol = load_policy(str(tmp_path / 'pol'))
    assert pol.uses_tokens and pol.doc['trained_by'] == 'token-bc' and pol.doc['tokens']['max_objects'] == 16
    assert pol.features == TP.state_features() and 'lead_gap_m' not in pol.features
    run = json.load(open(tmp_path / 'pol' / 'run.json'))
    assert run['status'] == 'ok'
    rc = main_policy(['rollout', project.path, '--policy', str(tmp_path / 'pol'), '--out', str(tmp_path / 'ro'),
                      '--episodes', '1', '--scenario', 'lead_brake', '--fake-carla', fake_carla])
    assert rc == 0
    ro = json.load(open(tmp_path / 'ro' / 'rollout.json'))
    assert ro['aggregate']['episodes'] == 1 and ro['policy'] == pol.path


def test_bc_token_cli_usage(capsys):
    from falcon_autonomy.core.policy import main_policy
    assert main_policy(['bc', '--arch', 'token', '--help']) == 0
    assert 'behaviour cloning of the token transformer' in capsys.readouterr().out
    assert main_policy(['bc', '--arch', 'token', 'p.json', '--out', 'x', '--net', 'heads=5']) == 2
