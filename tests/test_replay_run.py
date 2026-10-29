"""Log replays in CARLA against the fake (workers.replay through the drive loop), end to end: a Falcon scenario drive
exported to MCAP and imported back gives the same trajectories; its replay rebuilds it pose for pose; replays repeat
bit for bit; changes (an added pedestrian, a slower lead, weather) change what they should; the comparison of driving
behaviour and of a model; logs without a matching road; the command line; `falcon reproduce` of a replay."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKE, worker_env
from falcon_autonomy.core import exporters as E
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core.drive import scenario_config
from falcon_autonomy.core.logreplay import Log, Track, import_log, make_plan, world_info
from falcon_autonomy.core.logreplay.commands import replay_config
from falcon_autonomy.core.logreplay.compare import compare
from falcon_autonomy.core.reproduce import compare_drives

cv2 = pytest.importorskip('cv2')


def _project(path):
    from falcon_autonomy.core.project import Project
    p = Project.new('Replay')
    p.doc['rigs'][0]['sensors'][0].update(width=96, height=54, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.save(str(path))
    return p


def _rows(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    """A kinematic lead_brake scenario drive (the lead brakes hard at 2 s), its MCAP export with the actors' boxes,
    and the project."""
    base = tmp_path_factory.mktemp('replay')
    proj = _project(base / 'p.falcon.json')
    cfg = scenario_config(proj, SC.builtin('lead_brake'), 'world-town10', 'rig-default', str(base / 'drive'), 2310,
                          8310, seed=1, values={'speed': 36.0, 'gap': 15.0, 'decel': 6.0, 'brake_at': 2.0})
    cfg['scenario']['duration_s'] = 8.0
    os.makedirs(base / 'drive')
    (base / 'drive' / 'drive.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(base / 'drive' / 'drive.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=300)
    assert r.returncode == P.OK, r.stderr[-3000:]
    E.export([str(base / 'drive')], 'mcap', str(base / 'mcap'), {'objects': True})
    return {'base': base, 'project': proj, 'run': str(base / 'drive'), 'bag': str(base / 'mcap' / 'drive')}


def _replay(drive, name, log, changes=(), seed=0, driver='log', world='world-town10', env=None, **kw):
    base = drive['base']
    log_path = str(base / 'logs' / name / 'log.json')
    log.save(log_path)
    plan = make_plan(log, world_info(drive['project'], world), changes=list(changes), seed=seed, driver=driver,
                     log_path=log_path, **kw)
    cfg = replay_config(drive['project'], plan, world, 'rig-default', str(base / name), 2310, 8310)
    os.makedirs(base / name, exist_ok=True)
    (base / name / 'drive.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'replay', str(base / name / 'drive.json')],
                       capture_output=True, text=True, env=worker_env(env), timeout=300)
    run = json.loads((base / name / 'run.json').read_text())
    return r, run, str(base / name)


def test_falcon_mcap_round_trip_gives_the_same_trajectories(drive):
    """The key end-to-end check: drive → MCAP (Falcon's exporter, /tf and /falcon/objects) → import: the same ego
    and actor trajectories as the run's own frames.jsonl and actors.jsonl, the town from the export's manifest."""
    log = import_log(None, drive['bag'])
    assert log.meta['source']['format'] == 'ros2' and log.meta['source']['ego'] == '/tf: map -> base_link'
    assert log.map['kind'] == 'carla' and log.map['name'] == 'Town10HD_Opt'
    frames = _rows(os.path.join(drive['run'], 'frames.jsonl'))
    e = log.ego
    assert len(e) == len(frames) == 160
    assert e.t == pytest.approx([f['sim_t'] for f in frames], abs=1e-9)
    assert e.x == pytest.approx([f['ego']['x'] for f in frames], abs=1e-9)
    assert e.y == pytest.approx([f['ego']['y'] for f in frames], abs=1e-9)
    assert e.yaw == pytest.approx([f['ego']['yaw'] for f in frames], abs=1e-9)
    direct = import_log('falcon', drive['run'])
    assert direct.map['name'] == 'Town10HD_Opt'
    assert [a.id for a in log.actors] == [a.id for a in direct.actors] and log.actors
    for a, b in zip(log.actors, direct.actors):
        assert a.cls == b.cls and a.size == pytest.approx(b.size)
        for k in ('t', 'x', 'y', 'z', 'yaw'):
            assert getattr(a, k) == pytest.approx(getattr(b, k), abs=1e-9), k
    assert log.ego.size == pytest.approx(direct.ego.size)                 # the hero's box, written as id 'ego'


def test_replay_rebuilds_the_drive_pose_for_pose(drive):
    log = import_log('falcon', drive['run'])
    r, run, out = _replay(drive, 'rebuild', log)
    assert r.returncode == P.OK, r.stderr[-3000:]
    assert run['worker'] == 'replay' and run['status'] == 'ok'
    rp = json.loads(open(os.path.join(out, 'replay.json')).read())
    assert rp['placement']['mode'] == 'matched' and rp['map_match']['verdict'] == 'match'
    assert rp['map_match']['road_matches'] is True and rp['map_match']['share_on_lane_in_direction'] == 1.0
    # the same blueprint (the log names it): the replayed box is the logged one
    assert all(a['blueprint_choice'] == 'the log\'s' and a['replay_size'] == pytest.approx(a['log_size'], abs=1e-3)
               for a in rp['actors'])
    fa, fb = _rows(os.path.join(drive['run'], 'frames.jsonl')), _rows(os.path.join(out, 'frames.jsonl'))
    assert len(fa) == len(fb)
    for a, b in zip(fa, fb):
        assert b['sim_t'] == pytest.approx(a['sim_t'], abs=1e-9)
        for k in ('x', 'y', 'z', 'yaw'):
            assert b['ego'][k] == pytest.approx(a['ego'][k], abs=1e-4)
        assert b['control'] is None                        # a logged ego's control is not known
    aa, ab = _rows(os.path.join(drive['run'], 'actors.jsonl')), _rows(os.path.join(out, 'actors.jsonl'))
    for ra, rb in zip(aa, ab):
        orig = [x for x in ra['actors'] if x['role'] != 'hero']
        rep = [x for x in rb['actors'] if x['role'] != 'hero']
        assert len(orig) == len(rep) == 1
        for k in ('x', 'y', 'z', 'yaw'):
            assert rep[0][k] == pytest.approx(orig[0][k], abs=1e-4), k
        assert rep[0]['type_id'] == orig[0]['type_id'] and rep[0]['role'] == 'falcon_replay'
    assert os.path.isfile(os.path.join(out, 'lanes.json')) and os.path.isfile(os.path.join(out, 'dataset.json'))
    assert any(k.endswith('/objects') for k in fb[40]['files'])            # labelled like any drive


def test_moves_are_in_place_before_the_tick(drive):
    """set_transform is asynchronous in CARLA's client and can reach the server after the next tick (measured on 0.9.16:
    a log replay's ego a step behind in 22 of 1181 frames). The fake delivers every 3rd set_transform a tick late
    here: a replay and a kinematic scenario drive must still show every pose in its own step."""
    late = {'FAKE_CARLA_LATE_SET_TRANSFORM': '3'}
    log = import_log('falcon', drive['run'])
    r, run, out = _replay(drive, 'late', log, env=late)
    assert r.returncode == P.OK, r.stderr[-3000:]
    fa, fb = _rows(os.path.join(drive['run'], 'frames.jsonl')), _rows(os.path.join(out, 'frames.jsonl'))
    assert len(fa) == len(fb) == 160
    assert [b['ego']['x'] for b in fb] == pytest.approx([a['ego']['x'] for a in fa], abs=1e-4)
    for ra, rb in zip(_rows(os.path.join(drive['run'], 'actors.jsonl')), _rows(os.path.join(out, 'actors.jsonl'))):
        orig = [x for x in ra['actors'] if x['role'] != 'hero']
        rep = [x for x in rb['actors'] if x['role'] != 'hero']
        assert rep[0]['x'] == pytest.approx(orig[0]['x'], abs=1e-4)
    base = drive['base']
    cfg = json.loads(open(os.path.join(drive['run'], 'drive.json')).read())
    cfg['out'] = str(base / 'late_drive')
    os.makedirs(base / 'late_drive')
    (base / 'late_drive' / 'drive.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive',
                        str(base / 'late_drive' / 'drive.json')], capture_output=True, text=True,
                       env=worker_env(late), timeout=300)
    assert r.returncode == P.OK, r.stderr[-3000:]
    assert _rows(str(base / 'late_drive' / 'actors.jsonl')) == _rows(os.path.join(drive['run'], 'actors.jsonl'))
    assert [f['ego'] for f in _rows(str(base / 'late_drive' / 'frames.jsonl'))] == \
        [f['ego'] for f in fa]


def test_same_log_changes_and_seed_repeat_bit_for_bit(drive):
    log = import_log(None, drive['bag'])
    ch = [{'op': 'add_pedestrian', 'at_ego_s': 4.0, 'speed_mps': [1.0, 2.0]},
          {'op': 'speed_scale', 'actor': '101', 'factor': 0.8}]
    lead = [a.id for a in log.actors][0]
    ch[1]['actor'] = lead
    r1, run1, a = _replay(drive, 'rep_a', log, ch, seed=5, driver='follower')
    r2, run2, b = _replay(drive, 'rep_b', log, ch, seed=5, driver='follower')
    assert r1.returncode == r2.returncode == P.OK, r1.stderr[-2000:]
    ta = json.loads(open(os.path.join(a, 'trajectories.json')).read())
    tb = json.loads(open(os.path.join(b, 'trajectories.json')).read())
    assert ta == tb and len(ta['actors']) == 3
    assert _rows(os.path.join(a, 'actors.jsonl')) == _rows(os.path.join(b, 'actors.jsonl'))
    rep = compare_drives(a, b, {'pose_tol': 0.0, 'rgb_tol': 0.0, 'label_agree': 1.0})
    assert rep['ok'], rep['checks']
    assert run1['scenario']['criticality'] == run2['scenario']['criticality']


def test_changes_and_the_comparison(drive):
    log = import_log('falcon', drive['run'])
    lead = log.actors[0].id
    ped = {'op': 'add_pedestrian', 'id': 'walker', 'at_ego_s': 5.0, 'from': 'right', 'start_m': 6.0, 'speed_mps': 1.5}
    weather = {'op': 'weather', 'preset': 'WetCloudyNoon', 'params': {'fog_density': 30.0}}
    runs = {}
    for name, ch, drv in (('orig_log', [], 'log'), ('ped_log', [ped, weather], 'log'),
                          ('orig_follow', [], 'follower'), ('ped_follow', [ped], 'follower'),
                          ('slow_follow', [{'op': 'speed_scale', 'actor': lead, 'factor': 0.5, 'from_s': 0.0}],
                           'follower')):
        r, run, out = _replay(drive, name, log, ch, seed=1, driver=drv)
        assert r.returncode == P.OK, (name, r.stderr[-2000:])
        runs[name] = out
    cfg = json.loads(open(os.path.join(runs['ped_log'], 'drive.json')).read())
    assert cfg['weather']['fog_density'] == 30.0 and cfg['weather_source']['preset'] == 'WetCloudyNoon'
    rows = _rows(os.path.join(runs['ped_log'], 'actors.jsonl'))
    assert all(any(a['category'].startswith('human.pedestrian') for a in r['actors']) for r in rows)
    # the logged ego does not react: it drives into the pedestrian crossing its path when it was there
    d = compare(runs['orig_log'], [runs['ped_log']], out_dir=os.path.join(drive['base'], 'cmp_log'))
    p0 = d['pairs'][0]
    assert d['original_metrics']['collision'] == 0.0 and p0['metrics']['collision'] == 1.0
    assert p0['difference']['collision'] == 1.0 and 'walker' in p0['metrics']['collided_with']
    assert p0['ego_step_by_step']['max_position_difference_m'] == pytest.approx(0.0)
    assert d['intervals'] is None and 'deterministic' in d['interval_note']
    assert os.path.isfile(os.path.join(drive['base'], 'cmp_log', 'comparison.json'))
    # the follower brakes for it (it sees it coming into its corridor): no collision, but it falls behind
    f = compare(runs['orig_follow'], [runs['ped_follow'], runs['slow_follow']])
    ped_pair, slow_pair = f['pairs']
    assert ped_pair['metrics']['collision'] == 0.0 and ped_pair['metrics']['min_ttc_actor'] == 'walker'
    assert ped_pair['metrics']['min_distance_m'] > 1.0 and ped_pair['difference']['ego_path_m'] < -3.0
    assert ped_pair['ego_step_by_step']['max_position_difference_m'] > 3.0
    assert slow_pair['difference']['min_distance_m'] is not None
    iv = f['intervals']
    assert iv['n_pairs'] == 2 and 'not reliable' in iv['note']
    assert iv['mean_difference']['ego_path_m']['ci95'] is not None
    with pytest.raises(ValueError, match='timeline'):
        other = _replay(drive, 'short', log, [], driver='log', t_end=4.0)[2]
        compare(runs['orig_log'], [other])


def test_model_comparison_on_paired_frames(drive):
    log = import_log('falcon', drive['run'])
    _, _, a = _replay(drive, 'm_orig', log, [], driver='log')
    _, _, b = _replay(drive, 'm_ped', log, [{'op': 'add_pedestrian', 'id': 'walker', 'at_ego_s': 5.0,
                                              'from': 'right', 'start_m': 3.0, 'speed_mps': 0.5}], driver='log')
    syn = {'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
           'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector', 'licence': 'GPL-3.0-or-later'}
    d = compare(a, [b], model=syn, every=4, n_boot=200, out_dir=os.path.join(drive['base'], 'cmp_model'))
    m = d['model']
    assert m['synthetic_model'] and m['pairs'][0]['frames'] == 40
    assert m['comparison']['unit'] == 'image' and 'too narrow' in m['note']
    assert m['pairs'][0]['gt_objects']['changed'] >= m['pairs'][0]['gt_objects']['original']
    assert {r['metric'] for r in m['comparison']['rows']} >= {'AP', 'AP50'}
    assert os.path.isfile(os.path.join(drive['base'], 'cmp_model', 'predictions-changed0.jsonl'))


def test_no_matching_road_is_said_and_a_false_match_refused(drive):
    n = 81
    t = np.arange(n) * 0.1
    # somewhere else entirely (an unknown map, 5 km away): replayed on the map's first spawn point, flagged
    far = Log(Track('ego', 'car', t, 5000 + 8 * t, np.full(n, 5000.0), np.zeros(n), np.zeros(n)),
              [Track('c', 'car', t, 5020 + 8 * t, np.full(n, 5000.0), np.full(n, 0.75), np.zeros(n))],
              {'name': 'elsewhere', 'map': {'kind': 'unknown'}})
    r, run, out = _replay(drive, 'anchored', far)
    assert r.returncode == P.OK, r.stderr[-2000:]
    rp = json.loads(open(os.path.join(out, 'replay.json')).read())
    assert rp['placement']['mode'] == 'anchored' and rp['map_match']['road_matches'] is False
    assert rp['anchor']['to'][:2] != [5000.0, 5000.0] and any('no matching road' in x for x in run['notes'])
    assert run['scenario']['replay']['road_matches'] is False
    # a log that says it is in this town but runs 40 m beside its road: refused with the numbers, unless allowed
    off = Log(Track('ego', 'car', t, -40 + 8 * t, np.full(n, 40.0), np.zeros(n), np.zeros(n)), [],
              {'name': 'beside', 'map': {'kind': 'carla', 'name': 'Town10HD_Opt'}})
    r, run, _ = _replay(drive, 'refused', off)
    assert r.returncode == P.FAILED and 'does not match' in run['error'] and 'positions' in run['error']
    r, run, out = _replay(drive, 'allowed', off, allow_mismatch=True)
    assert r.returncode == P.OK and run['scenario']['replay']['road_matches'] is False


def test_command_line_import_run_compare_and_reproduce(drive, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    proj_path = str(drive['base'] / 'p.falcon.json')

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *a], capture_output=True, text=True,
                              env=env, timeout=600)
    r = cli('replay', 'import', 'ros2:' + drive['bag'], '--out', str(tmp_path / 'log'))
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])['summary']['map_name'] == 'Town10HD_Opt'
    r = cli('replay', 'match', str(tmp_path / 'log'), proj_path)
    assert r.returncode == 0 and json.loads(r.stdout.strip().splitlines()[-1])['summary']['placement'] == 'matched'
    (tmp_path / 'changes.json').write_text(json.dumps([{'op': 'add_pedestrian', 'at_ego_s': 5.0}]))
    for name, extra in (('a', []), ('b', ['--changes', str(tmp_path / 'changes.json')])):
        r = cli('replay', 'run', proj_path, '--log', str(tmp_path / 'log'), '--out', str(tmp_path / name),
                '--driver', 'follower', '--fake-carla', FAKE, *extra)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = cli('replay', 'compare', '--original', str(tmp_path / 'a'), '--changed', str(tmp_path / 'b'), '--out',
            str(tmp_path / 'cmp'))
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads((tmp_path / 'cmp' / 'comparison.json').read_text())
    assert doc['schema'] == 'falcon.replay_comparison/1' and doc['provenance']['inputs']
    r = cli('--fake-carla', FAKE, 'reproduce', str(tmp_path / 'a'), '--pose-tol', '0')
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = cli('replay', 'run', proj_path, '--log', str(tmp_path / 'nope'), '--out', str(tmp_path / 'c'))
    assert r.returncode == P.FAILED
