"""Root-cause attribution (core.rootcause): the factors a case has and how each is removed, on a hand-made run folder;
what cannot be re-run exactly is refused; the static attribution's correlations with intervals on hand-made
evaluations with a planted cause; a driving failure attributed to the policy's inputs in a small closed loop with a
planted bug; and against the fake carla, a counterfactual attribution with a known answer (a test model blind in fog:
fog is necessary and sufficient, range and appearance are neither), a failure that does not repeat (refused to
attribute), the report, the command and the pages' "Why did this fail?"."""
import json
import os

import numpy as np
import pytest

from conftest import FAKE
from falcon_autonomy.core import milsearch as MS
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import rootcause as RC
from falcon_autonomy.core import scenarios as SC

from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (fixtures)


class FogBlind:
    """A planted test model: sees every object (score 1) unless the fog density is 50 or more, then nothing."""
    synthetic = True

    def predict(self, rgb, meta):
        if float((meta.get('weather') or {}).get('fog_density', 0.0)) >= 50.0:
            return []
        return [{'bbox': o['bbox'], 'class': o['class'], 'score': 1.0} for o in meta['objects'] if o.get('bbox')]


SEEN = {}


class Flaky:
    """Misses everything the first time it sees a frame (by run name and step), then sees it: a failure that does
    not repeat."""
    synthetic = True

    def predict(self, rgb, meta):
        k = (meta.get('run'), meta.get('step'))
        SEEN[k] = SEEN.get(k, 0) + 1
        if SEEN[k] == 1:
            return []
        return [{'bbox': o['bbox'], 'class': o['class'], 'score': 1.0} for o in meta['objects'] if o.get('bbox')]


def _model(entry, mid):
    return {'id': mid, 'name': mid, 'kind': 'detector', 'runtime': 'python', 'entry': entry,
            'licence': 'GPL-3.0-or-later'}


FOGBLIND = _model('test_rootcause:FogBlind', 'fogblind')
FLAKY = _model('test_rootcause:Flaky', 'flaky')


# ------------------------------------------------------------------------------------------------ hand-made case
def _case_dir(tmp_path, *, trigger_on_bus=False, occluder='bus', faults=True, sun=5.0):
    from falcon_autonomy.core.project import Project
    p = Project.new('T')
    rig = p.doc['rigs'][0]
    rig['sensors'][0].update(width=320, height=180, labels=['boxes2d'])
    if faults:
        rig['sensors'][0]['faults'] = {'drop': {'p': 0.1}}
    scn = SC.sample(SC.builtin('jaywalker_behind_bus'), 1, {'distance': 40.0, 'speed': 30.0, 'walk': 1.5,
                                                             'trigger': 20.0})
    if trigger_on_bus:
        scn['timeline'][0]['trigger']['actor'] = 'bus'
    from falcon_autonomy.core.weather import resolve
    w = resolve({'preset': 'ClearNoon', 'sun_altitude_angle': sun, 'fog_density': 60.0, 'precipitation': 30.0})
    cfg = {'schema': 'falcon.drive_config/1', 'scenario': scn, 'rig': rig, 'weather': w,
           'weather_source': {'preset': 'ClearNoon', 'sun_altitude_angle': sun, 'fog_density': 60.0,
                              'precipitation': 30.0},
           'traffic': {'vehicles': 0, 'walkers': 0}, 'fps': 10.0, 'seed': 1}
    d = tmp_path / 'case'
    (d / 'objects' / 'cam-front').mkdir(parents=True)
    run = {'status': 'ok', 'scenario': {'id': 'jaywalker_behind_bus', 'values': scn['values'],
                                        'actor_ids': {'bus': 201, 'ped': 202},
                                        'blueprints': {'bus': 'vehicle.mitsubishi.fusorosa',
                                                       'ped': 'walker.pedestrian.0002'},
                                        'criticality': {'criticality': 0.7, 'overall': {'critical_actor': 'ped'},
                                                        'actors': {'ped': {'collision': False, 'min_ttc_s': 1.0,
                                                                           'min_ttc_t': 2.0, 'min_distance_t': 2.5}}}}}
    (d / 'run.json').write_text(json.dumps(run))
    (d / 'drive.json').write_text(json.dumps(cfg))
    occ_track = 201 if occluder == 'bus' else 'level:3'
    for s in range(0, 25):                                   # t = 0.1 … 2.5 s; the window [-1, 2] holds steps 0–19
        objs = [{'track_id': 202, 'class': 'pedestrian', 'bbox': [100, 50, 110, 80], 'bbox_full': [100, 40, 110, 80],
                 'occluded': 0.6, 'range_m': 20.0, 'pixels': 200},
                {'track_id': occ_track, 'class': 'bus', 'bbox': [80, 20, 140, 90], 'occluded': 0.0, 'range_m': 15.0,
                 'pixels': 4000}]
        (d / 'objects' / 'cam-front' / f'{s:06d}.json').write_text(json.dumps({'objects': objs}))
    return str(d), run, cfg


def test_the_factors_of_a_case_and_how_each_is_removed(tmp_path):
    d, run, cfg = _case_dir(tmp_path)
    case = {'failure': 'perception', 'actor': 'ped', 'window_s': 3.0}
    assert RC.failure_frames(case, run, cfg) == list(range(0, 20))
    rows = {r['factor']: r for r in RC.plan_factors(case, run, cfg, d)}
    assert list(rows) == list(RC.FACTORS)
    assert all(rows[f]['present'] and rows[f]['applicable'] for f in RC.FACTORS)
    assert 'by bus' in rows['occlusion']['in_case'] and '60%' in rows['occlusion']['in_case']
    import copy
    out = rows['occlusion']['transforms'][0][1](copy.deepcopy(cfg))
    assert [a['id'] for a in out['scenario']['actors']] == ['ped'] and out['scenario']['timeline']
    ahead = next(a for a in cfg['scenario']['actors'] if a['id'] == 'ped')['place']['ahead_m']      # 40 − 7
    out = rows['range']['transforms'][0][1](copy.deepcopy(cfg))
    assert next(a for a in out['scenario']['actors'] if a['id'] == 'ped')['place']['ahead_m'] == ahead - 10.0
    assert rows['range']['in_case'].startswith('20.0 m')
    out = rows['lighting']['transforms'][0][1](copy.deepcopy(cfg))
    assert out['weather']['sun_altitude_angle'] == RC.DAY_SUN_DEG and out['weather']['fog_density'] == 60.0
    assert out['weather_source']['sun_altitude_angle'] == RC.DAY_SUN_DEG
    out = rows['fog']['transforms'][0][1](copy.deepcopy(cfg))
    assert out['weather']['fog_density'] == 0.0 and out['weather']['precipitation'] == 30.0
    out = rows['rain']['transforms'][0][1](copy.deepcopy(cfg))
    assert out['weather']['precipitation'] == 0.0 and out['weather']['wetness'] == 0.0
    vs = [t(copy.deepcopy(cfg))['scenario']['actors'] for _, t in rows['appearance']['transforms']]
    assert [next(a for a in v if a['id'] == 'ped')['variant'] for v in vs] == [1, 2]
    assert 'walker.pedestrian.0002' in rows['appearance']['in_case']
    assert 'faults' not in rows['sensor_faults']['transforms'][0][1](copy.deepcopy(cfg))['rig']['sensors'][0]
    # what cannot be moved, and what is not there
    d2, run2, cfg2 = _case_dir(tmp_path / 'b', trigger_on_bus=True, faults=False, sun=60.0)
    rows = {r['factor']: r for r in RC.plan_factors(case, run2, cfg2, d2)}
    assert rows['occlusion']['present'] and not rows['occlusion']['applicable'] and 'trigger' in \
        rows['occlusion']['why_not']
    assert not rows['lighting']['present'] and not rows['sensor_faults']['present']
    d3, run3, cfg3 = _case_dir(tmp_path / 'c', occluder='level')
    rows = {r['factor']: r for r in RC.plan_factors(case, run3, cfg3, d3)}
    assert not rows['occlusion']['applicable'] and 'level:3' in rows['occlusion']['in_case']
    assert 'cannot be moved' in rows['occlusion']['why_not']


def test_what_cannot_be_re_run_exactly_is_refused():
    base = {'scenario': {'motion': 'kinematic', 'ego': {'driver': 'follower'}}, 'traffic': {'vehicles': 0}}
    assert RC.repeatable(base) is None
    assert 'not a scenario' in RC.repeatable({'traffic': {}})
    assert 'physics' in RC.repeatable({'scenario': {'motion': 'physics'}})
    assert 'traffic' in RC.repeatable(dict(base, traffic={'vehicles': 5}))
    assert 'autopilot' in RC.repeatable({'scenario': {'ego': {'driver': 'autopilot'}}})
    assert 'replay' in RC.repeatable(dict(base, replay={'plan': {}}))
    assert RC.verdict(True, True).startswith('necessary and sufficient')
    assert RC.verdict(True, False).startswith('necessary here') and RC.verdict(False, True).startswith('sufficient')
    assert RC.verdict(False, False).startswith('neither') and RC.verdict(None, None).startswith('not decided')


# ------------------------------------------------------------------------------------------------ static
def test_spearman_with_ties():
    assert RC.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert RC.spearman([1, 2, 3, 4], [1, 1, 1, 1]) is None
    assert RC._rankdata(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]
    # a point-biserial case worked by hand: x = 1..6, missed for the top three → ρ = 0.878
    assert RC.spearman([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1]) == pytest.approx(0.8783, abs=1e-4)


def _evaluation(tmp_path, n_drives=6):
    """Hand-made predictions: per frame a car at 10–70 m (missed beyond 40 m) and a pedestrian at 15 m (always
    found); the fog is the same everywhere; sun altitude varies by drive but has nothing to do with the misses."""
    rng = np.random.default_rng(3)
    frames = []
    for d in range(n_drives):
        for s in range(20):
            r = float(rng.uniform(10, 70))
            gt = [{'bbox': [10, 10, 30, 20], 'class': 'car', 'ignore': 0, 'range_m': r, 'occluded': 0.0,
                   'truncated': 0.0},
                  {'bbox': [100, 10, 10, 30], 'class': 'pedestrian', 'ignore': 0, 'range_m': 15.0, 'occluded': 0.2,
                   'truncated': 0.0},
                  {'bbox': [200, 10, 3, 3], 'class': 'car', 'ignore': 1, 'range_m': 90.0}]
            det = [{'bbox': [100, 10, 10, 30], 'class': 'pedestrian', 'score': 0.9}]
            if r <= 40:
                det.append({'bbox': [11, 10, 30, 20], 'class': 'car', 'score': 0.8})
            frames.append({'image_id': f'd{d}/{s}', 'drive': f'd{d}', 'step': s, 'gt': gt, 'det': det,
                           'conditions': {'fog_density': 20.0, 'sun_altitude_angle': 10.0 * d, 'weather': 'x'}})
    ev = tmp_path / 'ev'
    ev.mkdir()
    (ev / 'predictions.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
    (ev / 'evaluation.json').write_text(json.dumps({'model': {'id': 'm', 'name': 'M', 'kind': 'detector'},
                                                    'synthetic_model': False}))
    return str(ev), frames


def test_static_attribution_finds_the_planted_correlate(tmp_path):
    ev, frames = _evaluation(tmp_path)
    doc = RC.static(ev, str(tmp_path / 'why'), n_boot=300)
    assert doc['kind'] == 'static' and 'Correlational' in doc['wording'] and doc['unit'] == 'drive'
    assert doc['n_objects'] == 240 and doc['n_missed'] == sum(1 for f in frames if f['gt'][0]['range_m'] > 40)
    a = {x['attribute']: x for x in doc['attributes']}
    assert a['range_m']['clear'] and a['range_m']['ci_low'] > 0.5 and doc['ranking'][0] == 'range_m'
    assert not a['sun_altitude_angle']['clear']                           # no association: its interval spans 0
    assert a['fog_density']['note'].startswith('no variation')
    far = a['range_m']['bins'][-1]
    assert far['miss_rate'] == 1.0 and far['ci_low'] == 1.0
    cls = {c['value']: c for c in a['class']['categories']}
    assert cls['pedestrian']['miss_rate'] == 0.0 and 0 < cls['car']['miss_rate'] < 1
    assert 'range_m' in doc['summary'] and doc['note'] is None
    saved = json.load(open(tmp_path / 'why' / 'rootcause.json'))
    assert saved['provenance']['inputs']['predictions.jsonl']['sha256']
    assert json.load(open(tmp_path / 'why' / 'run.json'))['status'] == 'ok'
    # one drive: the intervals are over frames, and said to be too narrow
    (tmp_path / 'one').mkdir()
    ev1, _ = _evaluation(tmp_path / 'one', n_drives=1)
    one = RC.static(ev1, None, n_boot=100)
    assert one['unit'] == 'frame' and 'too narrow' in one['note']
    # through the command
    from falcon_autonomy.cli import main
    assert main(['why', 'evaluation', ev, '--out', str(tmp_path / 'cli'), '--boot', '100']) == P.OK
    assert json.load(open(tmp_path / 'cli' / 'rootcause.json'))['ranking'][0] == 'range_m'


# ------------------------------------------------------------------------------------------------ episodes
class LoopEnv:
    """A one-dimensional closed loop: the ego 30 m behind a lead car at a steady 8 m/s; a collision when the gap
    closes. Observations are the drivefeatures vector."""

    def __init__(self):
        from falcon_autonomy.core import drivefeatures as DF
        self.DF = DF
        self.feature_names = DF.FEATURE_NAMES
        self.ready = {'fps': 10.0}

    def _obs(self):
        st = np.zeros(len(self.feature_names), np.float32)
        fi = self.DF.FI
        st[fi['speed_mps']] = self.v
        st[fi['lead_present']] = 1.0
        st[fi['lead_gap_m']] = self.gap
        st[fi['lead_ttc_s']] = min(self.gap / max(self.v, 0.1), self.DF.TTC_CAP_S)
        st[fi['vru_ahead_m']] = self.DF.VRU_RANGE_M
        st[fi['tl_distance_m']] = self.DF.TL_RANGE_M
        return {'state': st}

    def reset(self, seed=None, options=None):
        self.v, self.gap, self.k = 8.0, 30.0, 0
        return self._obs(), {}

    def step(self, a):
        self.v = max(self.v + float(a[1]) * 0.5, 0.0)
        self.gap += (8.0 - self.v) * 0.1
        self.k += 1
        infs = [{'kind': 'collision_vehicle', 'step': self.k}] if self.gap <= 0 else []
        done = self.gap <= 0 or self.k >= 100
        info = {'infractions_new': infs}
        if done:
            info['episode_summary'] = {'infractions': [{'kind': 'collision_vehicle', 'step': self.k}]
                                       if self.gap <= 0 else []}
        return self._obs(), 0.0, done, False, info


def planted_policy(obs, names=None):
    """Holds 8 m/s — but floors it towards a lead car within 40 m (the planted bug: it tailgates)."""
    from falcon_autonomy.core import drivefeatures as DF
    st = obs['state']
    if st[DF.FI['lead_present']] and st[DF.FI['lead_gap_m']] < 40.0:
        return np.array([0.0, 1.0], np.float32)
    return np.array([0.0, float(np.clip(8.0 - st[DF.FI['speed_mps']], -1.0, 1.0))], np.float32)


def test_a_driving_failure_is_attributed_to_the_policys_inputs():
    env = LoopEnv()
    # the episode as the Leaderboard logs it
    obs, _ = env.reset()
    actions, infs = [], []
    while True:
        a = planted_policy(obs)
        actions.append(a.tolist())
        obs, _, term, _, info = env.step(a)
        if term:
            infs = info['episode_summary']['infractions']
            break
    e = {'episode': 0, 'seed': 3, 'motion': 'kinematic', 'actions': actions, 'infractions': infs,
         'traffic_vehicles': 0, 'traffic_walkers': 0}
    doc = RC.episode(e, env, planted_policy)
    assert doc['status'] == 'attributed' and doc['control']['infraction_recurs'] and doc['control']['same_actions']
    g = {r['group']: r for r in doc['groups']}
    assert g['lead']['removing_fixes'] and 'nothing there' in g['lead']['intervention']
    assert not any(r['removing_fixes'] for k, r in g.items() if k != 'lead')
    assert 'lead vehicle' in doc['summary'] and 'single-case' in doc['wording']
    assert doc['n_runs'] == 1 + len(RC.FEATURE_GROUPS)
    with pytest.raises(RC.Refused, match='traffic'):
        RC.episode(dict(e, traffic_vehicles=3), env, planted_policy)
    with pytest.raises(RC.Refused, match='no infraction'):
        RC.episode(dict(e, infractions=[]), env, planted_policy)


def tailgater(obs, names=None):
    """Floors it towards a lead car within 40 m, else stands still (a planted bug for the environment)."""
    from falcon_autonomy.core import drivefeatures as DF
    st = obs['state']
    near = st[DF.FI['lead_present']] and st[DF.FI['lead_gap_m']] < 40.0
    return np.array([0.0, 1.0 if near else -1.0], np.float32)


def test_a_leaderboard_infraction_through_the_command(tmp_path, project):
    """The episode attribution end to end: the environment worker against the fake, a Leaderboard evaluation of the
    planted policy, then `falcon why episode` on its collision."""
    from falcon_autonomy.cli import main
    from falcon_autonomy.core import leaderboard as LB
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service
    from falcon_autonomy.core.settings import Settings
    svc = make_service(Settings(str(tmp_path / 'settings.json')), FAKE, str(tmp_path / 'ws'))
    try:
        def make(scn):
            cfg = gym_config(project, 'world-town10', str(tmp_path / 'env'), svc.port, svc.tm_port, scenario=scn,
                             values={'speed': 40.0, 'gap': 10.0, 'decel': 9.0, 'brake_at': 0.5}, max_episode_s=6.0)
            return FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0)
        doc = LB.evaluate(tailgater, make, [{'scenario': 'lead_brake', 'seed': 1}], str(tmp_path / 'lb'), n_boot=50)
    finally:
        svc.shutdown()
    assert doc['per_route'][0]['infraction_counts'].get('collision_vehicle') == 1, doc
    out = tmp_path / 'why'
    assert main(['why', 'episode', project.path, '--episodes', str(tmp_path / 'lb' / 'episodes.jsonl'),
                 '--episode', '0', '--policy', 'test_rootcause:tailgater', '--out', str(out),
                 '--fake-carla', FAKE]) == P.OK
    d = json.load(open(out / 'rootcause.json'))
    assert d['kind'] == 'episode' and d['status'] == 'attributed' and d['control']['infraction_recurs']
    g = {r['group']: r for r in d['groups']}
    assert g['lead']['removing_fixes'] and not g['curvature']['removing_fixes'] and not g['route']['removing_fixes']
    assert d['provenance']['inputs']['episodes.jsonl']['sha256']
    assert json.load(open(out / 'run.json'))['status'] == 'ok'


# ------------------------------------------------------------------------------------------------ against the fake
@pytest.fixture(scope='module')
def fake(tmp_path_factory):
    from falcon_autonomy.core.gymenv import make_service
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.studio import Study
    base = tmp_path_factory.mktemp('why')
    p = Project.new('Why')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=90, labels=['boxes2d'])
    p.doc['models'] = [FOGBLIND, FLAKY]
    p.save(str(base / 'why.falcon.json'))
    svc = make_service(Settings(str(base / 'settings.json')), FAKE, p.workspace)
    st = Study(svc, p, SC.builtin('lead_brake'), 'fog', 'world-town10', 'rig-default', seed=2,
               objective={'objective': 'perception', 'model': FOGBLIND}, extra_space=MS.weather_space(['fog_density']),
               fps=10)
    vals = dict(SC.sample_params(SC.builtin('lead_brake'), 5), gap=30.0, speed=40.0, **{'weather.fog_density': 80.0})
    (row,) = st.evaluate([vals])
    yield {'project': p, 'svc': svc, 'base': base, 'row': row, 'study': st}
    svc.shutdown()


def test_a_counterfactual_attribution_with_a_known_answer(fake, tmp_path):
    from falcon_autonomy.core import reports as R
    row = fake['row']
    assert row['status'] == 'ok' and row['objective'] == 1.0                 # blind in fog: never sees the lead
    case = {'run': row['run'], 'failure': 'perception', 'model': FOGBLIND, 'threshold': 0.5}
    out = tmp_path / 'why'
    doc = RC.counterfactual(case, fake['svc'], str(out), fake['project'].workspace)
    assert doc['status'] == 'attributed' and doc['case']['actor'] == 'lead'
    assert doc['control']['fails'] is True and doc['control']['bit_for_bit'] is True
    f = {r['factor']: r for r in doc['factors']}
    assert f['fog']['removing_fixes'] is True and f['fog']['alone_fails'] is True
    assert f['fog']['verdict'].startswith('necessary and sufficient')
    assert f['range']['present'] and f['range']['removing_fixes'] is False and f['range']['alone_fails'] is False
    assert f['appearance']['fixed_in'] == '0 of 2' and f['appearance']['alone_fails'] is False
    assert not f['lighting']['present'] and not f['rain']['present'] and not f['sensor_faults']['present']
    assert doc['all_removed']['fails'] is False and 'fog' in doc['summary'] and 'single-case' in doc['wording']
    # runs: control, fog, range, two swaps, all removed, and fog, range, appearance each alone
    assert doc['n_runs'] == 9 and len(os.listdir(out / 'runs')) == 9
    swapped = [json.load(open(out / r['run'] / 'run.json'))['scenario']['blueprints']['lead']
               for r in doc['runs'] if r['label'].startswith('appearance: variant')]
    base_bp = json.load(open(os.path.join(row['run'], 'run.json')))['scenario']['blueprints']['lead']
    assert len(set(swapped + [base_bp])) == 3                             # two other cars than the case's
    run = json.load(open(out / 'run.json'))
    assert run['status'] == 'ok' and run['outputs']['rootcause.json']
    saved = json.load(open(out / 'rootcause.json'))
    assert saved['provenance']['inputs']['case/drive.json']['sha256'] and saved['case']['model']['id'] == 'fogblind'
    # again: every re-run is read back from its cache key, none driven
    t0 = os.path.getmtime(out / doc['runs'][0]['run'] / 'run.json')
    doc2 = RC.counterfactual(case, fake['svc'], str(out), fake['project'].workspace)
    assert os.path.getmtime(out / doc['runs'][0]['run'] / 'run.json') == t0 and doc2['summary'] == doc['summary']
    # the report
    meta = R.write_report('rootcause', str(out), str(tmp_path / 'r.html'))
    html = open(tmp_path / 'r.html', encoding='utf-8').read()
    assert meta['kind'] == 'rootcause' and 'necessary and sufficient' in html and 'single-case' in html


def test_a_failure_that_does_not_repeat_is_not_attributed(fake, tmp_path):
    SEEN.clear()
    case = {'run': fake['row']['run'], 'failure': 'perception', 'model': FLAKY, 'actor': 'lead', 'threshold': 0.5}
    doc = RC.counterfactual(case, fake['svc'], str(tmp_path / 'flaky'), fake['project'].workspace,
                            sufficiency=False)
    assert doc['status'] == 'not_reproduced' and doc['control']['fails'] is False
    assert doc['control']['bit_for_bit'] is True and 'cannot be attributed' in doc['summary']
    assert all('removing_fixes' not in f for f in doc['factors'])
    # a case that does not fail is refused; so is a drive that is not a scenario run
    with pytest.raises(RC.Refused, match='does not fail'):
        RC.counterfactual(dict(case, failure='criticality', threshold=0.99), fake['svc'], str(tmp_path / 'x'),
                          fake['project'].workspace)
    assert json.load(open(tmp_path / 'x' / 'run.json'))['status'] == 'refused'


def test_the_command_refuses_a_traffic_drive(fake, tmp_path):
    from falcon_autonomy.cli import main
    d = tmp_path / 'drive'
    d.mkdir()
    (d / 'run.json').write_text(json.dumps({'status': 'ok'}))
    (d / 'drive.json').write_text(json.dumps({'rig': {}, 'traffic': {'vehicles': 10}}))
    code = main(['why', 'case', fake['project'].path, '--run', str(d), '--out', str(tmp_path / 'o'),
                 '--fake-carla', FAKE])
    assert code == P.REFUSED
    assert 'not a scenario run' in json.load(open(tmp_path / 'o' / 'run.json'))['error']


# ------------------------------------------------------------------------------------------------ the pages
def test_why_from_the_rare_cases_and_models_pages(window, qapp, tmp_path, fake):  # noqa: F811
    from falcon_autonomy.core import harness as H
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.project.doc['rigs'][0]['sensors'][0].update(width=160, height=90, labels=['boxes2d'])
    window.project.doc['models'] = [dict(_model('falcon_autonomy.plugins.synthetic_detector:SyntheticDetector', 'syn'),
                                         params={'h50': 6.0, 'h_scale': 1.5})]
    window.save()
    window.service.set_workspace(window.workspace())
    rc = window.go('rarecases')
    rc.refresh()
    rc.scenario.setCurrentIndex(rc.scenario.findData('lead_brake'))
    rc.objective.setCurrentIndex(rc.objective.findData('perception'))
    assert rc.model.isEnabled() and rc.model.currentData() == 'syn'
    rc.weather_fields['fog_density'].setChecked(True)
    rc.fps.setValue(10)
    rc.budget.setValue(4)                          # its minimum
    rc.batch.setValue(4)
    rc.threshold.setValue(0.05)
    rc.start()
    argv = rc.job.spec.argv
    assert argv[argv.index('--objective') + 1] == 'perception' and argv[argv.index('--model') + 1] == 'syn'
    assert argv[argv.index('--weather') + 1] == 'fog_density' and rc.study().startswith('mil-lead_brake-syn')
    assert pump(qapp, lambda: rc.job.state in ('done', 'failed'), 300), rc.job.log
    assert rc.job.state == 'done', rc.job.log
    rc.load()
    assert len(rc.curve.values) == 4 and rc.failures.topLevelItemCount() >= 1
    assert 'weather.fog_density' in [rc.px.itemText(i) for i in range(rc.px.count())]
    rc.failures.setCurrentItem(rc.failures.topLevelItem(0))
    rc.why()
    job = rc.why_job
    assert job.spec.argv[job.spec.argv.index('why') + 1] == 'case'
    assert pump(qapp, lambda: job.state in ('done', 'failed'), 600), job.log
    assert job.state == 'done', job.log
    assert pump(qapp, lambda: rc.why_view is not None and rc.why_view.table.topLevelItemCount() == len(RC.FACTORS),
                10)
    assert rc.why_view.doc['status'] == 'attributed' and rc.why_view.summary.text()
    from falcon_autonomy.core import reports as R
    (label_, src), = R.sources(window.workspace())['rootcause']        # the Reports page lists it
    assert os.path.isfile(os.path.join(src, 'rootcause.json'))
    # the Models page: an evaluation of a scenario run's frames, then "Why did this fail?" on its worst frame
    run = fake['row']['run']
    ev = os.path.join(window.workspace(), 'evaluations', 'e1')
    H.evaluate([run], window.project.doc['models'][0], ev, window.workspace(), n_boot=50)
    page = window.go('models')
    page.refresh()
    page.ev.setCurrentIndex(page.ev.findData(ev))
    assert page.gal.topLevelItemCount() >= 1
    page.gal.setCurrentItem(page.gal.topLevelItem(0))
    page.why()
    wj = page.why_job
    assert wj.spec.argv[wj.spec.argv.index('why') + 1] == 'case' and '--step' in wj.spec.argv
    window.queue.cancel(wj)
    assert pump(qapp, lambda: wj.state in ('done', 'failed', 'cancelled'), 60)
    page.why_correlates()
    sj = page.why_job
    assert sj.spec.argv[sj.spec.argv.index('why') + 1] == 'evaluation'
    assert pump(qapp, lambda: sj.state in ('done', 'failed'), 120), sj.log
    assert sj.state == 'done', sj.log
    assert pump(qapp, lambda: page.why_view is not None and page.why_view.table.topLevelItemCount() > 0, 10)
