"""Level-k calibrated on dashcam near-miss tracks (core.levelk_nearmiss): synthetic ingests whose road users were
driven by level-k with planted parameters — a two-wheeler behind a slower car, your car ahead in the next lane —
written in core.nearmiss's own form (per clip nearmiss.json events with (N, 8) trajectories and the ego's GPS track;
for the smoothed variant also perception.json rows of the tracked boxes' near faces, the trajectories then
core.nearmiss.actor_traj's), calibrated per kind: the planted parameters within their intervals, the depth scale's
uncertainty propagated (exactly through the rows, approximately without them), the result a traffic profile's level-k
block, and every output private (the assistant, cluster and cloud bundles refuse it). No video, no models, no real
footage."""
import json
import math
import os

import numpy as np
import pytest

pytest.importorskip('torch')

from falcon_autonomy.core import levelk as LK  # noqa: E402
from falcon_autonomy.core import nearmiss as NM  # noqa: E402

TRUE = {'alpha': 0.6, 'lam': 6.0, 'v_des': 9.0, 't_gap': 2.0}
RATE = 20.0                      # frames a second of the synthetic ingest (falcon dashcam ingest --rate 20)
RIDER = (2.2, 0.9)               # core.scenario_plan.SIZES['motorcycle']
CAR = (4.6, 2.0)                 # core.scenario_plan.EGO_SIZE (D's size of a car)
LEAD_V = (4.0, 8.0)


def episode(seed: int, seconds: float = 8.0) -> dict:
    """A level-k simulation: a two-wheeler with the planted parameters behind a slower car in its lane, your car (the
    ego, level 0) ahead in the next lane; trajectories in core.levelk's frame. Everyone keeps to their lane (the
    carriageway bounds leave no lateral option): the calibration conditions on the lateral course a road user took,
    which matches a held lateral plan only when it does not re-plan its lateral target within the horizon."""
    rng = np.random.default_rng(seed)
    main = LK.RefPath.line(0.0, 0.0, 0.0, 600.0, 0.0, bounds=(-0.5, 0.5))     # lanes kept (module doc)
    lane2 = LK.RefPath.line(0.0, -LK.LANE_W, 0.0, 600.0, 0.0, bounds=(-0.5, 0.5))
    v_r = float(rng.uniform(4.0, 9.0))
    lead_v = float(rng.uniform(LEAD_V[0], LEAD_V[1]))
    rider = LK.make_agent('rider', 'two_wheeler', main, 20.0, 0.0, v_r, dict(TRUE, level=1))
    lead = LK.make_agent('lead', 'car', main, 20.0 + float(rng.uniform(10.0, 35.0)), 0.0, lead_v,
                         {'level': 0, 'v_des': lead_v}, controlled=False)
    ego_v = float(rng.uniform(4.0, 8.0))
    ego = LK.make_agent('ego', 'car', lane2, 20.0 + float(rng.uniform(5.0, 25.0)), 0.0, ego_v,
                        {'level': 0, 'v_des': ego_v}, controlled=False)
    rider['length'], rider['width'] = RIDER
    for a in (lead, ego):
        a['length'], a['width'] = CAR
    return LK.simulate(LK.Scene([rider, lead, ego]), seconds, dt=1.0 / RATE, seed=seed, mode='sample')


def _rh(tr: dict) -> np.ndarray:
    """(N, 5) t, x, y, yaw°, v of a core.levelk trajectory in the right-handed east-north frame."""
    return np.c_[tr['t'], tr['x'], -np.asarray(tr['y']), -np.asarray(tr['yaw']), tr['v']]


def _n8(tr: dict, size) -> np.ndarray:
    """A core.levelk trajectory as core.nearmiss's (N, 8) (right-handed east-north; velocity along the heading)."""
    R = _rh(tr)
    y = np.radians(R[:, 3])
    return np.c_[R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4] * np.cos(y), R[:, 4] * np.sin(y),
                 np.full(len(R), size[0]), np.full(len(R), size[1])]


def write_clip(od: str, key: str, sim: dict, t_offset: float = 0.0, smoothed: bool = False):
    """clips/<key>/nearmiss.json of one simulated episode (one window, two events: the rider and the car ahead) with
    level-k's own trajectories; smoothed: perception.json rows of the boxes' near faces too, and the trajectories
    core.nearmiss.actor_traj makes of them (as an ingest does)."""
    T = sim['trajectories']
    E = _rh(T['ego'])
    n = len(E)
    yaw = np.radians(E[:, 3])
    ego = {'t': E[:, 0] + t_offset, 'x': E[:, 1], 'y': E[:, 2], 'yaw': E[:, 3], 'vx': E[:, 4] * np.cos(yaw),
           'vy': E[:, 4] * np.sin(yaw), 'speed': E[:, 4], 'accel': np.r_[0.0, np.diff(E[:, 4]) * RATE],
           'ok': np.ones(n, bool), 'lat0': 12.97, 'lon0': 77.59, 'proj': 'synthetic'}
    frames = [{'n': k, 't': float(ego['t'][k]), 'boxes': []} for k in range(n)]
    kinds = {'rider': ('motorcycle', 1, RIDER), 'lead': ('car', 2, CAR)}
    for aid, (kind, tid, (L, _W)) in kinds.items():
        A = _rh(T[aid])
        for k in range(n):
            a = math.radians(ego['yaw'][k])
            dx, dy = A[k, 1] - ego['x'][k], A[k, 2] - ego['y'][k]
            f, lft = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
            r = math.hypot(f, lft)
            near = (r - L / 2.0) / r                     # the near face along the ray (core.nearmiss stage 4)
            frames[k]['boxes'].append({'track': tid, 'forward': f * near, 'left': lft * near, 'sigma': 0.05,
                                       'kind': kind, 'class_source': 'model', 'source': 'geometric',
                                       'box': [0, 0, 1, 1]})
    perc = {'schema': 'falcon.dashcam_perception/1', 'frames': frames}
    tracks = NM._tracks(perc)
    ks = np.arange(n)
    events = []
    for aid, (kind, tid, size) in kinds.items():
        if smoothed:
            a = NM.actor_traj(tracks[tid], ego, kind)
        else:
            a = _n8(T[aid], size)
            a[:, 0] = ego['t']
        events.append({'id': f'{key}-{tid}', 'track': tid, 'kind': kind, 'class_source': 'model',
                       'window_s': [float(ego['t'][0]), float(ego['t'][-1])], 'frames': list(range(n)),
                       'score': 0.5, 'score_median': 0.5,
                       'trajectory': {'actor': np.round(a, 6).tolist(), 'ego': np.round(NM.ego_traj(ego, ks),
                                                                                         6).tolist(),
                                      'columns': ['t', 'x', 'y', 'yaw_deg', 'vx', 'vy', 'length', 'width']}})
    d = os.path.join(od, 'clips', key)
    os.makedirs(d, exist_ok=True)
    if smoothed:
        with open(os.path.join(d, 'perception.json'), 'w') as fh:
            json.dump(perc, fh)
    with open(os.path.join(d, 'nearmiss.json'), 'w') as fh:
        json.dump({'schema': NM.SCHEMA, 'events': events, 'windows': [],
                   'ego_track': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ego.items()}}, fh)


def write_ingest(folder: str, seeds, smoothed: bool = False) -> str:
    od = os.path.join(folder, 'falcon-ingest')
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, 'ingest.json'), 'w') as fh:
        json.dump({'schema': 'falcon.dashcam_ingest/1', 'worker': 'dashcam-ingest', 'source': {'format': 'dashcam'},
                   'status': 'ok'}, fh)
    for s in seeds:
        write_clip(od, f'clip{s:03d}', episode(s), 100.0 * s, smoothed)
    return folder


@pytest.fixture(scope='module')
def ingest(tmp_path_factory):
    return write_ingest(str(tmp_path_factory.mktemp('nm') / 'footage'), range(12))


@pytest.fixture(scope='module')
def smoothed_ingest(tmp_path_factory):
    return write_ingest(str(tmp_path_factory.mktemp('nms') / 'footage'), range(2), smoothed=True)


@pytest.fixture(scope='module')
def calibrated(ingest):
    from falcon_autonomy.core.levelk_nearmiss import calibrate_nearmiss
    return calibrate_nearmiss(ingest, kinds=['two-wheeler'], depth_draws=4, seed=1)


# ================================================================================================ reading
def test_windows_read_the_ingest_and_rebuild_trajectories_at_a_depth_scale(smoothed_ingest, ingest):
    from falcon_autonomy.core import levelk_nearmiss as LN
    assert LN.DEPTH_SCALE == NM.DEPTH_SCALE
    root = LN.ingest_root(smoothed_ingest)
    assert root == LN.ingest_root(root) and root.endswith('falcon-ingest')
    wins = LN.windows(root)
    assert len(wins) == 2 and all(len(w['actors']) == 2 and w['actors'][0]['rows'] for w in wins)
    assert all(a['rows'] is None for w in LN.windows(LN.ingest_root(ingest)) for a in w['actors'])
    w = wins[0]
    a = w['actors'][0]
    T1, how = LN.scaled(w, a, 1.0)
    assert how == 'exact' and np.array_equal(T1, a['traj'])
    T2, how = LN.scaled(w, a, 1.1)                        # core.nearmiss.actor_traj on the rows, ranges × 1.1
    assert how == 'exact' and T2.shape == T1.shape
    r1 = np.hypot(T1[:, 1] - w['ego'][:, 1], T1[:, 2] - w['ego'][:, 2])
    r2 = np.hypot(T2[:, 1] - w['ego'][:, 1], T2[:, 2] - w['ego'][:, 2])
    assert np.median(r2 / r1) == pytest.approx(1.1, abs=0.02)
    a2 = dict(a, rows=None)
    T3, how = LN.scaled(w, a2, 1.1)
    assert how == 'approximate' and np.median(np.hypot(T3[:, 1] - w['ego'][:, 1], T3[:, 2] - w['ego'][:, 2])
                                              / r1) == pytest.approx(1.1, abs=1e-6)
    s, info = LN.decisions(w, ('two_wheeler',))
    assert s and all(x['type'] == 'two_wheeler' for x in s) and info['how'] == 'exact'
    assert all(len(x['features']['ds']) == len(LK.ACCELS) for x in s)      # acceleration choices, given its course
    assert all(np.allclose(x['features']['latmove'], 0.0) for x in s)
    assert any(len(x.get('choices') or ()) > 1 for x in s)                 # plans the track cannot tell apart
    assert LN.decisions(w, ('two_wheeler',), scale=1.1)[1]['how'] == 'exact'
    w2 = dict(w, actors=[dict(a, rows=None) for a in w['actors']])
    assert LN.decisions(w2, ('two_wheeler',), scale=1.1)[1]['how'] == 'approximate'
    # the smoother's ends are left out, and decisions are DT_PLAN apart
    ts = [x['t'] for x in s]
    assert min(ts) >= w['actors'][0]['traj'][0, 0] + LN.EDGE_S - 1e-6 and np.all(np.diff(ts) >= LK.DT_PLAN - 1e-6)
    assert LN.kind_type('two-wheeler') == LN.kind_type('motorcycle') == 'two_wheeler'
    assert LN.kind_type('truck') == 'bus' and LN.kind_type('cattle') is None
    assert LN._at_bound('t_gap', 0.2) and LN._at_bound('alpha', 1.0) and not LN._at_bound('lam', 6.0)
    with pytest.raises(ValueError):
        LN.ingest_root(os.path.dirname(ingest))


# ================================================================================================ calibration
def test_planted_parameters_are_recovered_within_their_intervals(calibrated):
    res = calibrated['kinds']['two_wheeler']
    assert res['decisions'] >= 200 and res['road_users'] == 12, res
    est, wald, comb = res['estimates'], res['wald95'], res['combined95']
    for k in ('alpha', 'lam', 'v_des'):
        lo, hi = comb[k]
        assert lo <= TRUE[k] <= hi, (k, est[k], comb[k], wald[k])
    assert abs(est['v_des'] - TRUE['v_des']) < 1.0 and abs(est['alpha'] - TRUE['alpha']) < 0.25
    lo, hi = comb['t_gap']
    assert lo <= TRUE['t_gap'] <= hi, (est['t_gap'], comb['t_gap'])
    ds = res['depth_scale']
    assert ds['draws'] == 4 and ds['how'] == 'approximate' and set(ds['sd']) == {'alpha', 'lam', 'v_des', 't_gap'}
    for k in ('alpha', 'lam', 'v_des', 't_gap'):              # the depth scale widens, never narrows
        assert comb[k][1] - comb[k][0] >= wald[k][1] - wald[k][0] - 1e-9
    assert any('optimistic' in a for a in calibrated['assumptions'])
    assert any('independent' in a for a in calibrated['assumptions'])


def test_the_result_is_a_traffic_profile_block_and_private(calibrated, ingest):
    from falcon_autonomy.assistant.privacy import _says_dashcam, levelk_reason, private_reason
    beh = calibrated['behaviour']
    assert beh['model'] == 'levelk' and set(beh['levelk']) == {'two_wheeler'}
    p = beh['levelk']['two_wheeler']
    assert p['k'] == 1 and set(p) == {'k', 'lambda', 'alpha', 'v_des', 't_gap'}
    tp = {'behaviour': beh}
    from falcon_autonomy.core.project import levelk_params, traffic_model
    assert traffic_model(tp) == 'levelk' and LK.profile_params(tp)['two_wheeler']['lam'] == p['lambda']
    out = calibrated['out']
    assert out == os.path.join(ingest, 'falcon-ingest', 'levelk')
    for f in ('calibration.json', 'run.json'):
        assert private_reason(os.path.join(out, f)), f
    with open(os.path.join(out, 'calibration.json')) as fh:
        doc = json.load(fh)
    assert _says_dashcam(doc) and levelk_reason(doc)
    # a profile it was loaded into, a drive with that profile, a run that recorded it, a project: all private
    assert levelk_reason(tp) and _says_dashcam({'traffic': tp})
    assert _says_dashcam({'provenance': {'settings': {'traffic': tp}}})
    assert _says_dashcam({'traffic_profiles': [{'behaviour': {'model': 'tm'}}, tp]})
    assert not _says_dashcam({'traffic': {'behaviour': {'model': 'levelk', 'levelk': levelk_params(tp)}}})


def test_written_elsewhere_it_carries_its_own_marker(ingest, tmp_path):
    from falcon_autonomy.assistant.privacy import private_reason
    from falcon_autonomy.core.levelk_nearmiss import calibrate_nearmiss
    out = str(tmp_path / 'elsewhere')
    doc = calibrate_nearmiss(ingest, out, kinds=['car'], depth_draws=0, steps=60)
    # the car ahead was scripted at a constant speed: calibrated, but not a planted level-k agent
    assert doc['kinds']['car']['decisions'] > 0 and 'not propagated' in doc['kinds']['car']['depth_scale']['how']
    assert private_reason(os.path.join(out, 'calibration.json')) and private_reason(out)


def test_cluster_and_cloud_bundles_refuse_it(calibrated, tmp_path):
    from falcon_autonomy.cluster import bundle as B
    tp = {'id': 'tp-cal', 'name': 'calibrated', 'vehicles': 10, 'walkers': 0, 'fleet': {},
          'behaviour': calibrated['behaviour']}
    # a drive job whose config carries the profile
    cfg = str(tmp_path / 'drive.json')
    with open(cfg, 'w') as fh:
        json.dump({'traffic': tp, 'town': 'Town10HD_Opt'}, fh)

    class Spec:
        meta = {'worker': 'drive', 'config': cfg}
        inputs = ()
        run_dir = str(tmp_path / 'run')
        argv, kind, title, memory_gb = [], 'drive', 'a drive', 1.0
    with pytest.raises(B.BundleError, match='private'):
        B.make(Spec())
    # a job reading the calibration file itself
    Spec.inputs = (os.path.join(calibrated['out'], 'calibration.json'),)
    with open(cfg, 'w') as fh:
        json.dump({'town': 'Town10HD_Opt'}, fh)
    with pytest.raises(B.BundleError, match='private'):
        B.make(Spec())
    # a cloud study whose world uses the profile
    from falcon_autonomy.core.project import Project
    pr = Project.new('Study')
    doc = pr.doc
    w = doc['worlds'][0]
    B.make_study(pr, 'sweep', {'id': 'lead_brake'}, w['id'], doc['rigs'][0]['id'], 'n', {})   # without: it goes
    doc['traffic_profiles'].append(tp)
    w['traffic_profile'] = 'tp-cal'
    with pytest.raises(B.BundleError, match='private'):
        B.make_study(pr, 'sweep', {'id': 'lead_brake'}, w['id'], doc['rigs'][0]['id'], 'n', {})


def test_cli(ingest, tmp_path, capsys):
    out = str(tmp_path / 'cli')
    assert LK.main_levelk(['calibrate', '--from-nearmiss', ingest, '--kind', 'two-wheeler', '--depth-draws', '0',
                           '--steps', '80', '--out', out]) == 0
    printed = capsys.readouterr().out
    assert '"two_wheeler"' in printed and 'levelk_source' in printed
    assert os.path.isfile(os.path.join(out, 'calibration.json'))
    assert LK.main_levelk(['calibrate', '--from-nearmiss', ingest, '--kind', 'cattle', '--out', out]) == 2
    assert LK.main_levelk(['calibrate', '--from-nearmiss', str(tmp_path), '--out', out]) == 2     # not an ingest
    assert LK.main_levelk(['calibrate', '--out', out]) == 2                     # --log or --from-nearmiss
    assert LK.main_levelk(['calibrate', '--help']) == 0


def test_the_assistant_never_sees_calibrated_parameters(calibrated, project):
    """project_get of a traffic profile carrying dashcam-calibrated level-k parameters withholds them; the
    calibration's own files are refused by the guard every tool calls."""
    from conftest import FAKE
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.privacy import guard
    from falcon_autonomy.assistant.tool_api import ToolError, call
    tp = dict(project.doc['traffic_profiles'][0], behaviour=calibrated['behaviour'])
    project.doc['traffic_profiles'][0] = tp
    project.save(project.path)
    ctx = TOOLS.context(project.path, FAKE, emit=lambda kind, payload: None)
    try:
        got = call(ctx, 'project_get', {'kind': 'traffic_profile', 'id': tp['id']})['traffic_profile']
        assert str(got['behaviour']['levelk']).startswith('private:')
        lam = calibrated['behaviour']['levelk']['two_wheeler']['lambda']
        assert str(lam) not in json.dumps(got)
    finally:
        ctx.service.shutdown(grace_s=10.0)
    with pytest.raises(ToolError, match='private'):
        guard(os.path.join(calibrated['out'], 'calibration.json'))


def test_z_of_survives_an_estimate_past_its_upper_bound():
    """A refit started from an estimate the softplus let run far past its upper bound (seen on a synthetic CARLA
    clip: λ or v_des more than ~709 above its floor overflowed expm1) starts at the bound instead of raising."""
    import torch
    from falcon_autonomy.core import levelk as LK
    from falcon_autonomy.core import trafficgraph as TG
    z = TG.z_of({'alpha': 0.7, 'lam': 5.0e3, 'v_des': 1.0e4, 't_gap': 3.0})
    th = {k: float(v) for k, v in TG.theta_of(torch.tensor(z, dtype=torch.float64)).items()}
    assert abs(th['lam'] - LK.PARAM_BOUNDS['lam'][1]) < 1e-6 and abs(th['v_des'] - LK.PARAM_BOUNDS['v_des'][1]) < 1e-6
    assert abs(th['t_gap'] - 3.0) < 1e-9 and abs(th['alpha'] - 0.7) < 1e-9
