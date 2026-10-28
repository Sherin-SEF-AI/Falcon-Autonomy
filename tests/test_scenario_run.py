"""Scenarios run by the director inside a drive, against the fake: placement, the timeline, trajectories,
criticality and pass/fail, determinism, refusal when the map lacks what a scenario needs."""
import json
import os
import subprocess
import sys

import pytest

from conftest import worker_env
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core.drive import scenario_config


def _run(project, tmp_path, key, seed=1, values=None, name='run'):
    project.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=[])
    cfg = scenario_config(project, SC.builtin(key), 'world-town10', 'rig-default', str(tmp_path / name), 2310, 8310,
                          seed=seed, values=values)
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=180)
    run = json.loads((tmp_path / name / 'run.json').read_text())
    return r, run


def test_lead_brake_runs_fires_and_measures(project, tmp_path):
    r, run = _run(project, tmp_path, 'lead_brake', values={'speed': 40.0, 'gap': 12.0, 'decel': 8.0,
                                                           'brake_at': 2.0})
    assert r.returncode == P.OK, r.stderr
    sc = run['scenario']
    assert sc['fired'] and sc['fired'][0][1] == pytest.approx(2.0, abs=0.06)
    assert sc['values']['gap'] == 12.0 and 'criticality' in sc and sc['checks']
    traj = json.loads((tmp_path / 'run' / 'trajectories.json').read_text())
    assert len(traj['actors']['ego']) == len(traj['actors']['lead']) == 240
    lead = traj['actors']['lead']
    assert lead[30][4] ** 2 + lead[30][5] ** 2 > 50 and lead[-1][4] ** 2 + lead[-1][5] ** 2 < 1   # moving, stopped
    # actors.jsonl: kinematic actors (physics off, so CARLA reports them still) carry the director's velocity
    rows = [json.loads(x) for x in (tmp_path / 'run' / 'actors.jsonl').read_text().splitlines()]
    ego = next(a for a in rows[30]['actors'] if a['role'] == 'hero')
    v_traj = (traj['actors']['ego'][30][4] ** 2 + traj['actors']['ego'][30][5] ** 2) ** 0.5
    assert v_traj > 5 and (ego['vx'] ** 2 + ego['vy'] ** 2) ** 0.5 == pytest.approx(v_traj, abs=0.5)


def test_same_seed_same_scenario_bit_for_bit(project, tmp_path):
    _, a = _run(project, tmp_path, 'cut_in', seed=5, name='a')
    _, b = _run(project, tmp_path, 'cut_in', seed=5, name='b')
    ta = json.loads((tmp_path / 'a' / 'trajectories.json').read_text())
    tb = json.loads((tmp_path / 'b' / 'trajectories.json').read_text())
    assert ta == tb and a['scenario']['values'] == b['scenario']['values']
    assert a['scenario']['criticality'] == b['scenario']['criticality']


@pytest.mark.parametrize('key', ['pedestrian_crossing', 'stalled_vehicle', 'jaywalker_behind_bus',
                                 'wrong_way_scooter', 'bus_pulling_out', 'door_opening', 'two_wheeler_filtering',
                                 'cyclist_crossing'])
def test_builtins_run_on_the_fake(project, tmp_path, key):
    r, run = _run(project, tmp_path, key, seed=2)
    assert r.returncode == P.OK, r.stderr
    assert run['scenario']['passed'] in (True, False) and run['scenario']['checks']


def test_a_scenario_the_map_cannot_host_is_refused(project, tmp_path):
    r, run = _run(project, tmp_path, 'unprotected_turn')          # the fake's road has no junction
    assert r.returncode == P.FAILED and 'no start on this map' in run['error'] and 'junction' in run['error']


def test_sweep_then_search_resumes_through_the_cli(project, tmp_path, fake_carla):
    import os
    project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *a], capture_output=True, text=True,
                              env=env, timeout=600)
    r = cli('sweep', path, '--scenario', 'lead_brake', '--name', 'lb', '--samples', '3', '--fake-carla', fake_carla)
    assert r.returncode == 0, r.stdout + r.stderr
    ws = project.workspace
    res = [json.loads(x) for x in open(os.path.join(ws, 'studies', 'lb', 'results.jsonl'))]
    assert len(res) == 3 and all(x['status'] == 'ok' for x in res)
    # the same sweep again: every evaluation cached, nothing re-run
    r = cli('sweep', path, '--scenario', 'lead_brake', '--name', 'lb', '--samples', '3', '--fake-carla', fake_carla)
    assert r.returncode == 0 and len(open(os.path.join(ws, 'studies', 'lb', 'results.jsonl')).readlines()) == 3
    # a search in the same study starts from the sweep's three and adds three more
    r = cli('search', path, '--scenario', 'lead_brake', '--name', 'lb', '--method', 'bayes', '--budget', '6',
            '--batch', '3', '--fake-carla', fake_carla)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.load(open(os.path.join(ws, 'studies', 'lb', 'search-bayes.json')))
    assert len(out['history']) == 6 and all(0 <= h['value'] <= 1 for h in out['history'])
