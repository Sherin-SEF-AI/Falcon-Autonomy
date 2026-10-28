"""Behaviour cloning against the fake carla: drives record the ego's control, pairs come from the environment's own
featuriser, a seeded MLP beats the constant baseline on held-out drives, the policy folder loads and runs in the
environment, and the CLI does the same."""
import json
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
from falcon_autonomy.core.drive import scenario_config


@pytest.fixture(scope='module')
def drives(tmp_path_factory):
    """Three lead_brake drives (the lane follower brakes behind a braking lead; kinematic motion), 12 s each."""
    from falcon_autonomy.core.project import Project
    d = tmp_path_factory.mktemp('bc')
    p = Project.new('BC')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32)
    out = []
    for seed in (1, 2, 3):
        run = str(d / f'run{seed}')
        cfg = scenario_config(p, SC.builtin('lead_brake'), 'world-town10', 'rig-default', run, 2310, 8310, seed=seed)
        os.makedirs(run)
        cp = os.path.join(run, 'drive.json')                # as the service writes it: the run's own input
        with open(cp, 'w') as fh:
            json.dump(cfg, fh)
        r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', cp], capture_output=True,
                           text=True, env=worker_env(), timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]
        out.append(run)
    return out


def _frames(run):
    with open(os.path.join(run, 'frames.jsonl')) as fh:
        return [json.loads(x) for x in fh]


def test_drives_record_the_control_and_the_kinematic_speed(drives):
    fr = _frames(drives[0])
    assert all(set(f['control']) == {'throttle', 'steer', 'brake', 'hand_brake', 'reverse'} for f in fr)
    assert any(f['control']['brake'] > 0 for f in fr) and any(f['control']['throttle'] > 0 for f in fr)
    assert fr[5]['speed_mps'] > 5.0                       # the director's speed (CARLA reports a kinematic ego still)
    with open(os.path.join(drives[0], 'actors.jsonl')) as fh:
        a5 = [json.loads(x) for x in fh][5]
    hero = next(a for a in a5['actors'] if a['role'] == 'hero')
    assert fr[5]['speed_mps'] == pytest.approx((hero['vx'] ** 2 + hero['vy'] ** 2) ** 0.5, abs=1e-3)


def test_pairs_come_from_the_environments_featuriser(drives):
    s = IM.drive_samples(drives[0])
    X, Y = s['X'], s['Y']
    assert X.shape[1] == DF.N_FEATURES and len(X) == len(Y) == len(_frames(drives[0])) - 1
    assert np.all(X[:, DF.FI['tl_known']] == 0)                         # drives do not record the lights
    # the action of step t is the next step's control; prev_* of step t + 1 is that same control
    assert np.allclose(X[1:, DF.FI['prev_accel']], Y[:-1, 1]) and np.allclose(X[1:, DF.FI['prev_steer']], Y[:-1, 0])
    drive = json.load(open(os.path.join(drives[0], 'drive.json')))
    gap0 = drive['scenario']['values']['gap']
    assert X[0, DF.FI['lead_present']] == 1 and X[0, DF.FI['lead_gap_m']] == pytest.approx(gap0 - 4.4, abs=0.5)
    assert X[0, DF.FI['target_speed_mps']] == pytest.approx(drive['scenario']['ego']['speed_kmh'] / 3.6, rel=1e-4)
    assert s['report']['target_speed_source'] == 'the scenario\'s ego speed'
    assert abs(float(X[:, DF.FI['lateral_offset_m']].max())) < 0.05     # the lane follower stays on its lane centre


def test_bc_beats_the_constant_baseline_seeded_and_loads(drives, tmp_path):
    from falcon_autonomy.core.policy import load_policy
    m = IM.train_bc(drives, str(tmp_path / 'pol'), {'epochs': 30, 'seed': 0})
    assert m['samples']['val'] > 200 and len(m['drives']['val']) == 1 and len(m['drives']['train']) == 2
    assert m['val']['mae'] < 0.95 * m['baseline_val']['mae']
    assert m['train']['mae'] < 0.9 * m['baseline_train']['mae']
    files = set(os.listdir(tmp_path / 'pol'))
    assert {'policy.json', 'weights.pt', 'metrics.json', 'run.json'} <= files
    run = json.load(open(tmp_path / 'pol' / 'run.json'))
    assert run['status'] == 'ok' and run['provenance']['seed'] == 0 and len(run['provenance']['inputs']) == 3
    pol = load_policy(str(tmp_path / 'pol'))
    assert not set(pol.features) & {'tl_known', 'tl_red'} and pol.doc['trained_by'] == 'bc'
    a = pol.act({'state': np.zeros(DF.N_FEATURES, np.float32)})
    assert a.shape == (2,) and np.all(np.abs(a) <= 1)
    m2 = IM.train_bc(drives, str(tmp_path / 'pol2'), {'epochs': 30, 'seed': 0})
    assert m2['val'] == m['val']                                         # seeded: the same run twice
    off = IM.offline_eval(str(tmp_path / 'pol'), drives[:1], str(tmp_path / 'off'))
    assert off['overall']['n'] == len(_frames(drives[0])) - 1 and os.path.isfile(tmp_path / 'off' / 'offline.json')


def test_bc_refuses_what_it_cannot_learn_from(drives, tmp_path):
    with pytest.raises(ValueError, match='at least two drives'):
        IM.train_bc(drives[:1], str(tmp_path / 'p'))
    old = str(tmp_path / 'old')
    shutil.copytree(drives[0], old)
    fr = _frames(old)
    with open(os.path.join(old, 'frames.jsonl'), 'w') as fh:
        for f in fr:
            f.pop('control')
            fh.write(json.dumps(f) + '\n')
    with pytest.raises(ValueError, match='records no control'):
        IM.drive_samples(old)


def test_cli_bc_then_rollout_in_the_environment(drives, project, tmp_path, fake_carla):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'policy', *a], capture_output=True,
                              text=True, env=env, timeout=600)
    pol = str(tmp_path / 'pol')
    r = cli('bc', *sum((['--run', d] for d in drives), []), '--out', pol, '--epochs', '5')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'constant baseline' in r.stdout
    out = str(tmp_path / 'roll')
    r = cli('rollout', project.path, '--policy', pol, '--out', out, '--episodes', '2', '--scenario', 'lead_brake',
            '--fake-carla', fake_carla)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.load(open(os.path.join(out, 'rollout.json')))
    assert doc['aggregate']['episodes'] == 2 and doc['seeds'] == [0, 1] and doc['policy'] == pol
    assert all(e['scenario']['id'] == 'lead_brake' for e in doc['episodes'])
    assert set(doc['provenance']['inputs']) == {'environment', 'policy'}
    r = cli('offline', '--policy', pol, '--run', drives[0])
    assert r.returncode == 0 and '"baseline_mean_action"' in r.stdout
    r = cli('nonsense')
    assert r.returncode == 2
