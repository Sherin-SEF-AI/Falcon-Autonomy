"""The learned world model, imagination and end-to-end driving (milestone 12, feature 9), on the CPU with tiny networks
and drives recorded on the fake carla: the derived-action inverse of the kinematic model, sequences built with the
environment's featuriser and BEV, the constant-velocity baseline exact where motion is constant, training seeded and
resumable bit for bit, the evaluation's curves and verdicts, collection from the environment, a policy learnt in
imagination and an end-to-end camera policy both running in the environment, the CLI, and the Training ▸ World model
tab."""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import worker_env
from falcon_autonomy.core import drivefeatures as DF
from falcon_autonomy.core import imitation as IM
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import wmdata as D
from falcon_autonomy.core import wmtrain as W
from falcon_autonomy.core.drive import scenario_config
from test_gui import pump, qapp, slot_exceptions, window          # noqa: F401  (fixtures)

TINY = {'deter': 16, 'stoch': 4, 'classes': 4, 'hidden': 16, 'depth': 4, 'batch': 2, 'seq_len': 8,
        'device': 'cpu', 'log_every': 5, 'eval_every': 10, 'val_batches': 1}


@pytest.fixture(scope='module')
def drives(tmp_path_factory):
    """Three lead_brake drives on the fake (lane follower, kinematic, recorded control), the camera 128 × 72 at the
    environment camera's pose."""
    from falcon_autonomy.core.project import Project
    d = tmp_path_factory.mktemp('wmdrives')
    p = Project.new('WM')
    p.doc['rigs'][0]['sensors'][0].update(width=128, height=72)
    out = []
    for seed in (1, 2, 3):
        run = str(d / f'run{seed}')
        cfg = scenario_config(p, SC.builtin('lead_brake'), 'world-town10', 'rig-default', run, 2310, 8310, seed=seed)
        os.makedirs(run)
        with open(os.path.join(run, 'drive.json'), 'w') as fh:
            json.dump(cfg, fh)
        r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', os.path.join(run, 'drive.json')],
                           capture_output=True, text=True, env=worker_env(), timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]
        out.append(run)
    return out


@pytest.fixture(scope='module')
def tiny_model(drives, tmp_path_factory):
    out = str(tmp_path_factory.mktemp('wm') / 'model')
    W.train(dict(TINY, train=drives[:2], val=drives[2:], iterations=20), out)
    return out


# ------------------------------------------------------------------------------------------------- data
def test_derived_actions_invert_the_kinematic_bicycle_model():
    rng = np.random.default_rng(0)
    wb, dt = 2.9, 0.05
    for _ in range(200):
        v0 = rng.uniform(1.0, 15.0)
        steer = rng.uniform(-0.5, 0.5)
        thr, brk = (rng.uniform(0, 1), 0.0) if rng.random() < 0.5 else (0.0, rng.uniform(0, 0.3))
        v1 = max(v0 + (3.0 * thr - 8.0 * brk) * dt, 0.0)                     # workers/director.py kin_step
        dyaw = math.degrees(v1 * math.tan(steer * math.radians(70.0)) / wb * dt)
        a = D.derive_action(v0, v1, 10.0, 10.0 + dyaw, dt, wb)
        assert a == pytest.approx([steer, thr - brk], abs=1e-9)
    assert D.derive_action(0.2, 0.2, 0.0, 5.0, dt, wb)[0] == 0.0             # too slow to state a steer


def test_drive_sequences_use_the_environments_featuriser_and_bev(drives):
    from falcon_autonomy.core import worldmodel as WM
    (s,) = D.drive_sequence(drives[0])
    T = len(s['state'])
    assert s['bev'].shape == (T, 80, 64) and s['state'].shape == (T, DF.N_FEATURES) and s['kind'] == 'drive'
    assert s['report']['action_source'] == 'recorded' and not s['terminal'].any()
    fr = [json.loads(x) for x in open(os.path.join(drives[0], 'frames.jsonl'))]
    assert np.allclose(s['action'][1:], [DF.control_to_action(f['control']) for f in fr[1:]])
    assert np.allclose(s['reward'], s['terms'].sum(1), atol=1e-5) and s['reward'][0] == 0.0
    assert s['terms'][1:, DF.REWARD_TERMS.index('progress')].sum() > 1.0      # it drove along the route
    data = WM.load_run(drives[0])
    lm = WM.lane_map_for(data['lanes'], [f['ego'] for f in fr], DF.bev_spec())
    assert np.array_equal(s['bev'][37], DF.bev_bits(fr[37]['ego'], data['actors'][37], DF.bev_spec(), lm,
                                                    data['ego_id']))
    bc = IM.drive_samples(drives[0])                      # behaviour cloning's pairs: the same feature vectors
    assert np.allclose(s['state'][bc['steps']], bc['X'], atol=1e-5)
    with pytest.raises(ValueError):
        D.drive_sequence(os.path.dirname(drives[0]))


def test_constant_velocity_baseline_is_exact_under_constant_velocity(drives, tmp_path):
    """A drive rewritten so the ego and the lead move at exactly constant velocities: the baseline's actor and ego
    planes are the truth's at every horizon, and at k = 0 its frame is the frame."""
    run = str(tmp_path / 'cv')
    os.makedirs(run)
    for f in ('lanes.json', 'dataset.json', 'drive.json'):
        shutil.copy(os.path.join(drives[0], f), run)
    fr = [json.loads(x) for x in open(os.path.join(drives[0], 'frames.jsonl'))]
    ac = [json.loads(x) for x in open(os.path.join(drives[0], 'actors.jsonl'))]
    dt = 0.05
    with open(os.path.join(run, 'frames.jsonl'), 'w') as ff, open(os.path.join(run, 'actors.jsonl'), 'w') as fa:
        for t, (f, a) in enumerate(zip(fr, ac)):
            rows = []
            for r in a['actors']:
                v = 6.0 if r['role'] == 'hero' else 8.5
                x0 = ac[0]['actors'][[q['id'] for q in ac[0]['actors']].index(r['id'])]['x']
                rows.append(dict(r, x=x0 + v * t * dt, vx=v, vy=0.0, yaw=0.0))
            hero = next(r for r in rows if r['role'] == 'hero')
            f = dict(f, ego=dict(f['ego'], x=hero['x'], y=hero['y'], yaw=0.0), speed_mps=6.0)
            ff.write(json.dumps(f) + '\n')
            fa.write(json.dumps(dict(a, actors=rows)) + '\n')
    (s,) = D.drive_sequence(run)
    cv = W.ConstantVelocity(run, DF.bev_spec())
    actor_bits = sum(1 << k for k in range(3, 8))
    assert np.array_equal(cv.predict(10, s['bev'][10], [0])[0], s['bev'][10])
    ks = [1, 5, 10, 20]
    for t0 in (10, 60):
        for k, p in zip(ks, cv.predict(t0, s['bev'][t0], ks)):
            assert np.array_equal(p & actor_bits, s['bev'][t0 + k] & actor_bits), (t0, k)


def test_scripted_behaviour_and_ou_noise_are_seeded():
    n1, n2 = D.OUNoise(np.random.default_rng(3)), D.OUNoise(np.random.default_rng(3))
    a, b = [n1() for _ in range(20)], [n2() for _ in range(20)]
    assert np.array_equal(a, b) and np.std(a) > 0.05
    x = np.zeros(DF.N_FEATURES, np.float32)
    x[DF.FI['lateral_offset_m']], x[DF.FI['target_speed_mps']] = 1.0, 8.0
    steer, accel = D.scripted_action(x)
    assert steer < 0 and accel > 0                                 # right of the centre: steer left; slow: speed up
    x[DF.FI['lead_present']], x[DF.FI['lead_ttc_s']], x[DF.FI['lead_gap_m']] = 1.0, 1.5, 20.0
    assert D.scripted_action(x)[1] == -1.0


# ------------------------------------------------------------------------------------------------- training
def test_training_is_seeded_and_resumes_bit_for_bit(drives, tmp_path):
    import torch
    from falcon_autonomy.core.training import Cancelled
    s = dict(TINY, train=drives[:2], iterations=12, ckpt_every=4)
    W.train(s, str(tmp_path / 'a'))
    W.train(s, str(tmp_path / 'b'))
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        return calls['n'] > 7
    with pytest.raises(Cancelled):
        W.train(s, str(tmp_path / 'c'), cancel=cancel)
    assert os.path.isfile(tmp_path / 'c' / 'checkpoints' / 'last.pt')
    assert json.load(open(tmp_path / 'c' / 'run.json'))['status'] == 'cancelled'
    W.train(s, str(tmp_path / 'c'))                                   # resumes from iteration 7
    wa, wb, wc = (torch.load(tmp_path / x / 'weights.pt') for x in 'abc')
    for k in wa:
        assert torch.equal(wa[k], wb[k]) and torch.equal(wa[k], wc[k]), k
    doc = json.load(open(tmp_path / 'a' / 'model.json'))
    assert doc['schema'] == W.MODEL_SCHEMA and doc['cost']['iterations'] == 12
    assert 'tl_known' not in doc['features'] and doc['data']['val'] and doc['provenance']['inputs']
    rows = [json.loads(x) for x in open(tmp_path / 'a' / 'metrics.jsonl')]
    assert [r['it'] for r in rows] == [5, 10, 12] and 'val' in rows[1]
    rows_c = [json.loads(x) for x in open(tmp_path / 'c' / 'metrics.jsonl')]
    assert [r['it'] for r in rows_c] == [5, 10, 12]
    calls['n'] = 0
    with pytest.raises(Cancelled):
        W.train(s, str(tmp_path / 'd'), cancel=cancel)
    with pytest.raises(W.WMError):                                    # a checkpoint of other settings is refused
        W.train(dict(s, seed=1), str(tmp_path / 'd'))
    with pytest.raises(W.WMError):
        W.normalise_settings({'nonsense': 1})


def test_evaluation_curves_baselines_and_verdict(tiny_model, drives):
    ev = W.evaluate(tiny_model, horizon=10, stride=20, context=5)
    assert ev['horizons'] == list(range(1, 11)) and ev['report_horizons'] == [1, 5, 10]
    assert set(ev['iou']) == {'model', 'copy_last', 'constant_velocity'}
    for m in ev['iou']:
        assert all(len(v) == 10 for v in ev['iou'][m].values())
    assert ev['iou']['copy_last']['drivable'][0] == pytest.approx(1.0)       # the fake's straight road
    cl, cv = ev['iou']['copy_last']['actors'], ev['iou']['constant_velocity']['actors']
    assert cl[9] < cl[0] and cv[0] > 0.9                                      # actors move away from where they were
    assert all(p > 0 for p in ev['pairs']) and set(ev['verdict']) == {'1', '5', '10'}
    assert ev['continue']['episodes'] == 0 and 'note' in ev['continue']
    assert len(ev['reward_mae']['model']) == 10 and ev['reward_mae']['train_mean'][0] is not None
    z = np.load(os.path.join(tiny_model, 'eval', 'samples.npz'))
    assert z['truth'].shape[1:] == (11, 80, 64) and z['model'].shape == z['truth'].shape and z['has_cv'].all()
    assert np.array_equal(z['truth'][:, 0], z['model'][:, 0])
    assert 'IoU, model' in W.table(ev)


def test_collect_imagine_and_drive_in_the_environment(tiny_model, drives, project, settings, fake_carla, tmp_path):
    """Environment episodes with a scripted behaviour become sequences with terminals; a model trained with them
    measures the continue head; a policy learnt in imagination is a policy folder that runs in the environment."""
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service
    from falcon_autonomy.core.policy import load_policy, rollout
    svc = make_service(settings, fake_carla, str(tmp_path / 'ws'), str(tmp_path / 'locks'))
    try:
        cfg = gym_config(project, 'world-town10', str(tmp_path / 'env'), svc.port, svc.tm_port, scenario='cut_in',
                         bev=True, log_actions=False)
        with FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0) as env:
            doc = D.collect(env, 'scripted', 3, str(tmp_path / 'exp'), seed=5, sigma=0.3)
        seqs = D.experience_sequences(str(tmp_path / 'exp'))
        assert len(seqs) == 3 and doc['steps'] == sum(len(q['state']) for q in seqs)
        for q, row in zip(seqs, doc['episodes']):
            assert q['terminal'].sum() == (row['termination'] not in ('time_limit',)) and not q['terminal'][:-1].any()
            assert q['reward'].sum() == pytest.approx(row['return'], abs=1e-3) and q['reward'][0] == 0.0
        out = str(tmp_path / 'wm2')
        W.train(dict(TINY, train=drives[:1] + [str(tmp_path / 'exp')], val=[str(tmp_path / 'exp')],
                     iterations=6), out)
        ev = W.evaluate(out, horizon=5, stride=10)
        assert ev['continue']['episodes'] == 3 and len(ev['continue']['brier']) == 5
        assert 'constant_velocity' not in ev['iou']                          # no recorded drive among the sources
        m = W.imagine(tiny_model, str(tmp_path / 'pol'), {'iterations': 6, 'batch': 2, 'seq_len': 4, 'horizon': 4,
                                                           'device': 'cpu', 'log_every': 3})
        assert len(m['history']) == 2 and math.isfinite(m['final']['imagined_return'])
        pol = load_policy(str(tmp_path / 'pol'))
        assert pol.doc['trained_by'] == 'dreamer' and pol.features == json.load(
            open(os.path.join(tiny_model, 'model.json')))['features']
        cfg = gym_config(project, 'world-town10', str(tmp_path / 'env2'), svc.port, svc.tm_port, log_actions=False)
        with FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0) as env:
            r = rollout(pol, env, 1, seed=2, options={'max_episode_s': 3.0, 'start_speed_kmh': 30.0})
        assert r['aggregate']['episodes'] == 1
    finally:
        svc.shutdown()


# ------------------------------------------------------------------------------------------------- end to end
def test_end_to_end_policy_offline_and_on_the_leaderboard(drives, project, settings, fake_carla, tmp_path):
    from falcon_autonomy.core import e2e as E
    from falcon_autonomy.core import leaderboard as LB
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service
    from falcon_autonomy.core.policy import load_policy
    m = E.train_e2e(drives, str(tmp_path / 'e2e'), {'epochs': 2, 'batch': 32, 'device': 'cpu', 'every': 4})
    assert m['samples']['train'] > 0 and len(m['per_drive_val']) == 1 and len(m['history']) == 2
    assert set(m['val']) >= {'steer_mae', 'accel_mae', 'mae'} and 'baseline_val' in m
    pol = load_policy(str(tmp_path / 'e2e'))
    assert pol.uses_camera and not pol.uses_bev and pol.features == ['speed_mps', 'target_speed_mps']
    assert pol.doc['camera']['width'] == 128 and pol.doc['camera']['pose']['x'] == 1.5
    off = E.offline_eval(str(tmp_path / 'e2e'), drives[2:], str(tmp_path / 'off'))
    assert off['overall']['n'] > 0 and os.path.isfile(tmp_path / 'off' / 'offline.json')
    with pytest.raises(ValueError):
        pol.act({'state': np.zeros(DF.N_FEATURES, np.float32)})               # no camera in the observation
    svc = make_service(settings, fake_carla, str(tmp_path / 'ws'), str(tmp_path / 'locks'))
    try:
        def env_factory(scn):
            cfg = gym_config(project, 'world-town10', str(tmp_path / 'lb' / (scn or 'free')), svc.port,
                             svc.tm_port, scenario=scn, camera=pol.doc['camera'], log_actions=True,
                             max_episode_s=2.0)
            return FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0)
        doc = LB.evaluate(pol, env_factory, [{'scenario': 'lead_brake', 'seed': 1}, {'scenario': None, 'seed': 2}],
                          str(tmp_path / 'lbout'), n_boot=50)
        assert doc['routes'] == 2 and not doc['failed'] and 0.0 <= doc['metrics']['driving_score']['value'] <= 1.0
    finally:
        svc.shutdown()
    with pytest.raises(ValueError):
        E.train_e2e(drives[:1], str(tmp_path / 'one'), {'epochs': 1, 'device': 'cpu'})


# ------------------------------------------------------------------------------------------------- the command
def test_cli_train_show_and_eval(drives, tmp_path, capsys):
    out = str(tmp_path / 'cli')
    argv = ['train', '--out', out, '--run', drives[0], '--val', drives[1], '--iterations', '4', '--device', 'cpu']
    for k, v in TINY.items():
        if k != 'device':
            argv += ['--option', f'{k}={v}']
    assert W.main_worldmodel(argv) == 0
    assert W.main_worldmodel(['eval', '--model', out, '--horizon', '5', '--stride', '40', '--device', 'cpu']) == 0
    assert W.main_worldmodel(['show', out]) == 0
    txt = capsys.readouterr().out
    assert 'IoU, constant_velocity' in txt and 'reward mean absolute error' in txt
    assert W.main_worldmodel(['train', '--out', str(tmp_path / 'x'), '--device', 'cpu']) == 2   # no source
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'worldmodel', '--help'], capture_output=True,
                       text=True, env=worker_env(), timeout=60)
    assert r.returncode == 0 and 'imagine' in r.stdout
    spec = W.job_spec(['train', '--out', out], 'x')
    assert spec.needs_gpu and spec.env['FALCON_GPU_LOCK_HELD'] == '1' and spec.resumable


# ------------------------------------------------------------------------------------------------- the GUI
@pytest.mark.gui
def test_world_model_tab(window, qapp, tiny_model, drives):
    from PyQt6.QtCore import Qt
    page = window.go('training')
    tab = page.worldmodel
    assert page.tabs.tabText(page.tabs.indexOf(tab)) == 'World model'
    W.evaluate(tiny_model, horizon=8, stride=30)
    dst = page.ws('worldmodels', 'tiny')
    shutil.copytree(tiny_model, dst)
    tab.refresh()
    k = tab.models.findData(dst)
    assert k >= 0
    tab.models.setCurrentIndex(k)
    pump(qapp, timeout=0.2)
    tab.show_model()
    assert tab.ev is not None and tab.channel.count() == 9
    assert tab.info.vals['Verdict'].text().startswith('channels where the model beats')
    tab.channel.setCurrentIndex(tab.channel.findData('vehicle'))
    assert tab.chart.channel == 'vehicle'
    v = tab.viewer
    assert v.sample.count() > 0 and v.slider.maximum() == 8
    v.slider.setValue(8)
    v.base.setCurrentIndex(1)
    pump(qapp, timeout=0.2)
    assert not v.views[1].pixmap().isNull() and 'k = 8' in v.caption.text()
    tab.chart.grab()
    sent = []
    page._submit = lambda spec: sent.append(spec)
    for r in drives[:2]:
        window.service.catalog.add_run(r)
    tab.refresh()
    for i in range(tab.sources.count()):
        if tab.sources.item(i).data(256) in drives[:2]:
            tab.sources.item(i).setCheckState(Qt.CheckState.Checked)
    assert sorted(page._checked(tab.sources)) == sorted(drives[:2])
    tab.train()
    assert sent[-1].argv[3:5] == ['worldmodel', 'train'] and sent[-1].argv.count('--run') == 2
    tab.e2e()
    assert sent[-1].argv[4] == 'e2e' and os.path.isdir(sent[-1].argv[sent[-1].argv.index('--out') + 1])
    tab.evaluate()
    assert sent[-1].argv[4] == 'eval' and '--model' in sent[-1].argv
