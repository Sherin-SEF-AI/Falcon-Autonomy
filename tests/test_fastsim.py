"""Two-speed failure search (core.fastsim): the fast simulator against the real controllers (LaneFollower.act and
director.follow stepped one scenario at a time), its criticality against core.criticality.summarise on the same
trajectories, which built-ins it covers and why not the rest, the torch back end, the statistics (Latin hypercube,
ranks, Spearman, isotonic regression), the two-speed loop on a planted landscape (more seeds with a failure than plain
search for the same budget; a decorrelated cheap model is distrusted and the search falls back; the controls and the
false-negative estimate), and end to end on the fake carla: an ordinary study with fidelity fields that the result
readers read, resumed without new runs; the commands."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from conftest import FAKE

from falcon_autonomy.core import criticality as CR
from falcon_autonomy.core import fastsim as F
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import search as S


# ================================================================================================ coverage
def test_coverage_names_what_is_refused_and_why():
    cov = F.coverage()
    assert set(cov) == set(SC.LIBRARY)
    refused = {k: v for k, v in cov.items() if v}
    assert set(refused) == {'unprotected_turn'} and 'junction' in refused['unprotected_turn']
    scn = SC.builtin('lead_brake')
    assert 'opposite' in F.unsupported(dict(scn, actors=[dict(scn['actors'][0], place={'lane': 'opposite'})]))
    assert 'level-k' in F.unsupported(dict(scn, actors=[dict(scn['actors'][0], driver='levelk')]))
    assert 'track' in F.unsupported(dict(scn, actors=[dict(scn['actors'][0], track={'t': [0]})]))
    assert 'autopilot' in F.unsupported(dict(scn, ego=dict(scn['ego'], driver='autopilot')))
    assert 'speed_breaker' in F.unsupported(dict(scn, needs=['speed_breaker']))
    assert 'no map' in F.unsupported(dict(scn, start={'x': 0, 'y': 0, 'yaw': 0}))
    with pytest.raises(F.Unsupported, match='straight road'):
        F.LowFidelity(SC.builtin('unprotected_turn'))
    with pytest.raises(F.Unsupported):
        F.rollout(SC.builtin('unprotected_turn'), {}, 1)


# ================================================================================================ the controllers
def _kin(state, u, half):
    """workers.director.Director.kin_step's arithmetic (it needs a CARLA actor; the constants are the class's)."""
    from falcon_autonomy.workers.director import Director as D
    x, y, yaw, v = state
    if u.get('hand_brake'):
        v = 0.0
    else:
        v = max(v + (D.ACCEL * u.get('throttle', 0.0) - D.BRAKE * u.get('brake', 0.0)) * 0.05, 0.0)
    wb = max(2.0 * half * 0.6, 0.8)
    yaw2 = yaw + math.degrees(v * math.tan(u.get('steer', 0.0) * math.radians(D.STEER_DEG)) / wb * 0.05)
    mid = math.radians((yaw + yaw2) / 2.0)
    return [x + math.cos(mid) * v * 0.05, y + math.sin(mid) * v * 0.05, yaw2, v]


def _reference(scn, values, actor_y, steps=240):
    """One scenario stepped with the REAL controllers — plugins.lane_follower.LaneFollower.act for the ego and
    workers.director.follow for its one vehicle actor — in the order the director uses (timeline, actor, ego, all
    from the frame just ticked). Returns (ego s, ego v, actor x, actor y, actor v) after each step."""
    from falcon_autonomy.plugins.lane_follower import LaneFollower
    from falcon_autonomy.workers.director import Path, follow
    conc = SC.concrete(scn, values)
    a = conc['actors'][0]
    v0 = float(conc['ego']['speed_kmh']) / 3.6
    lf = LaneFollower(cruise_kmh=v0 * 3.6 + 1e-6)
    ahead = float(a['place']['ahead_m'])
    pe = Path([(float(k), 0.0) for k in range(401)], [0.0] * 401, [3.5] * 401, 0.0)
    pa = Path([(ahead + k, actor_y) for k in range(301)], [0.0] * 301, [3.5] * 301, 0.0)
    half = 2.3
    box = NS(bounding_box=NS(extent=NS(x=half)))
    ego, act = [0.0, 0.0, 0.0, v0], [ahead, actor_y, 0.0, float(a['speed_kmh']) / 3.6]
    st = {'speed': act[3], 'target': act[3], 'accel': 3.0, 'lateral': 0.0, 'lat_from': 0.0, 'lat_to': 0.0,
          'lat_t0': 0.0, 'lat_dur': 0.0}
    fired, out = False, []
    ev = conc['timeline'][0]
    for k in range(steps):
        t = k * 0.05
        trig = ev['trigger']
        go = t >= float(trig['t_s']) if trig['type'] == 'time' else \
            (act[0] - ego[0]) - half - half > float(trig['gt_m'])
        if go and not fired:
            fired = True
            for do in ev['actions']:
                if do['do'] == 'brake':
                    st['target'], st['accel'] = 0.0, float(do['decel'])
                else:                                            # a lane change into the ego's lane
                    st.update(lat_from=st['lateral'], lat_to=0.0 - actor_y, lat_t0=t,
                              lat_dur=float(do['duration_s']))
        s = pe.project(ego[0], ego[1])
        route = [list(pe.at(s + 2.0 * j, 0.0)[:2]) + [None] for j in range(1, 26)]
        lead = None
        for j, p in enumerate(route):
            if math.hypot(act[0] - p[0], act[1] - p[1]) < 1.8:
                lead = {'distance_m': max(2.0 * (j + 1) - half - half, 0.0), 'speed_mps': act[3]}
                break
        u = lf.act({'t': t, 'dt': 0.05, 'ego': {'x': ego[0], 'y': ego[1], 'yaw': ego[2], 'speed_mps': ego[3]},
                    'route': route, 'lead': lead})
        if st['speed'] < st['target']:
            st['speed'] = min(st['target'], st['speed'] + st['accel'] * 0.05)
        elif st['speed'] > st['target']:
            st['speed'] = max(st['target'], st['speed'] - st['accel'] * 0.05)
        if st['lat_dur'] > 0:
            f = min(max((t - st['lat_t0']) / st['lat_dur'], 0.0), 1.0)
            st['lateral'] = st['lat_from'] + (st['lat_to'] - st['lat_from']) * f
        r = math.radians(act[2])
        ua = follow(box, pa, st['lateral'], st['speed'], 0.05, st,
                    NS(location=NS(x=act[0], y=act[1]), rotation=NS(yaw=act[2])),
                    NS(x=act[3] * math.cos(r), y=act[3] * math.sin(r)))
        act = _kin(act, ua, half)
        ego = _kin(ego, u, half)
        out.append((ego[0], ego[3], act[0], act[1], act[3]))
    return np.array(out)


@pytest.mark.parametrize('key,actor,actor_y,values', [
    ('lead_brake', 'lead', 0.0, {'speed': 50.0, 'gap': 14.0, 'decel': 8.0, 'brake_at': 2.5}),
    ('lead_brake', 'lead', 0.0, {'speed': 28.0, 'gap': 35.0, 'decel': 4.0, 'brake_at': 4.0}),
    ('lead_brake', 'lead', 0.0, {'speed': 58.0, 'gap': 9.0, 'decel': 9.0, 'brake_at': 2.0}),
    ('cut_in', 'cutter', -3.5, {'speed': 40.0, 'cutter_speed': 60.0, 'start_offset': -8.0, 'cut_gap': 3.0,
                                 'cut_time': 1.5}),
    ('cut_in', 'cutter', -3.5, {'speed': 55.0, 'cutter_speed': 62.0, 'start_offset': 2.0, 'cut_gap': 6.0,
                                 'cut_time': 2.5}),
])
def test_the_batched_rollout_is_the_real_controllers(key, actor, actor_y, values):
    scn = SC.builtin(key)
    ref = _reference(scn, values, actor_y)
    # in a batch with other parameter sets: the rows do not mix
    other = SC.sample_params(scn, 5)
    tr = F.rollout(scn, {k: np.array([other[k], values[k]]) for k in values}, 2)
    e, a = tr['ego'][1], tr['actors'][actor][1]
    assert tr['t'][0] == pytest.approx(0.05) and e.shape == (240, 8)
    assert np.allclose(e[:, CR.X], ref[:, 0], atol=1e-6) and np.allclose(e[:, CR.VX], ref[:, 1], atol=1e-6)
    assert np.allclose(a[:, CR.X], ref[:, 2], atol=1e-6) and np.allclose(a[:, CR.Y], ref[:, 3], atol=1e-6)
    assert np.allclose(np.hypot(a[:, CR.VX], a[:, CR.VY]), ref[:, 4], atol=1e-6)
    if key == 'cut_in':                                           # it did change lane, and fired when the gap opened
        assert abs(a[-1, CR.Y]) < 0.2 and a[0, CR.Y] == pytest.approx(-3.5, abs=1e-3)
        assert np.isfinite(tr['fired'][0][1])


def test_placements_walkers_props_and_the_constants_come_from_the_classes():
    from falcon_autonomy.plugins.lane_follower import LaneFollower
    from falcon_autonomy.workers.director import Director
    K = F._controller_constants()
    assert K['accel'] == Director.ACCEL and K['brake'] == Director.BRAKE and K['gap_s'] == LaneFollower().gap_s
    # a pedestrian waits on the sidewalk (the kerb side, 2 m beyond the lane edge) and walks across once triggered
    scn = SC.builtin('pedestrian_crossing')
    tr = F.rollout(scn, {'speed': 30.0, 'distance': 40.0, 'walk': 1.5, 'trigger': 20.0}, 1)
    ped = tr['actors']['ped'][0]
    assert ped[0, CR.Y] == pytest.approx(3.5 / 2 + 2.0) and ped[0, CR.X] == pytest.approx(40.0)
    t_fire = float(tr['fired'][0][0])
    assert 1.0 < t_fire < 4.0
    moving = ped[:, CR.T] > t_fire + 0.06
    assert np.allclose(ped[moving, CR.VY], -1.5) and np.all(ped[~moving][:-1, CR.VY] == 0.0)
    assert ped[0, CR.YAW] == pytest.approx(-90.0) and (ped[0, CR.LEN], ped[0, CR.WID]) == (0.5, 0.5)
    # a wrong-way scooter drives against the lane; a stalled truck has the truck's box; a bus stands at the kerb
    tr = F.rollout(SC.builtin('wrong_way_scooter'), {'speed': 30.0, 'scooter_speed': 20.0, 'distance': 80.0,
                                                     'lateral': 1.0}, 1)
    sc = tr['actors']['scooter'][0]
    assert sc[10, CR.VX] == pytest.approx(-20 / 3.6, abs=0.2) and sc[0, CR.Y] == pytest.approx(1.0, abs=0.01)
    tr = F.rollout(SC.builtin('stalled_vehicle'), {'speed': 40.0, 'distance': 80.0, 'kind': 'truck'}, 1)
    assert tuple(tr['actors']['stalled'][0, 0, [CR.LEN, CR.WID]]) == (6.0, 2.6)
    assert tr['ego'][0, -1, CR.VX] < 0.5 and tr['ego'][0, -1, CR.X] < 80.0        # the follower stopped behind it
    tr = F.rollout(SC.builtin('bus_pulling_out'), {'speed': 30.0, 'distance': 50.0, 'trigger': 30.0,
                                                   'pull_time': 3.0, 'bus_speed': 20.0}, 1)
    bus = tr['actors']['bus'][0]
    assert bus[0, CR.Y] == pytest.approx(3.5 / 2 + 3.9 / 2 + 0.2) and abs(bus[-1, CR.Y]) < 0.5
    # a proxy (the handcart) moves along its path at walking pace; an autorickshaw is capped at its class's top speed
    tr = F.rollout(SC.builtin('handcart_kerb_lane'), {'speed': 30.0, 'distance': 50.0, 'cart_speed': 4.0,
                                                      'lateral': 0.5}, 1)
    cart = tr['actors']['cart'][0]
    assert cart[-1, CR.X] - cart[0, CR.X] == pytest.approx(4.0 / 3.6 * (16.0 - 0.05), rel=1e-6)
    tr = F.rollout(SC.builtin('autorickshaw_cut_in'), {'speed': 40.0, 'faster': 20.0, 'start_offset': 4.0,
                                                       'cut_gap': 4.0, 'cut_time': 1.5}, 1)
    assert tr['actors']['auto'][0, :, CR.VX].max() <= 65.0 / 3.6 + 1e-6
    # the stated driver model is another driver: it runs, and is not the follower
    a = F.rollout(SC.builtin('lead_brake'), {'speed': 50.0, 'gap': 14.0, 'decel': 8.0, 'brake_at': 2.5}, 1)
    b = F.rollout(SC.builtin('lead_brake'), {'speed': 50.0, 'gap': 14.0, 'decel': 8.0, 'brake_at': 2.5}, 1,
                  driver='idm')
    assert b['driver'] == 'idm' and not np.allclose(a['ego'], b['ego'])
    with pytest.raises(ValueError, match='driver'):
        F.rollout(SC.builtin('lead_brake'), {'speed': 50.0, 'gap': 14.0, 'decel': 8.0, 'brake_at': 2.5}, 1,
                  driver='human')


# ================================================================================================ the criticality
@pytest.mark.parametrize('key', ['lead_brake', 'cut_in', 'pedestrian_crossing', 'two_wheeler_filtering',
                                 'jaywalker_behind_bus', 'handcart_kerb_lane'])
def test_the_criticality_is_core_criticalitys(key):
    scn = SC.builtin(key)
    space = SC.search_space(scn)
    Pp = [S.to_params(space, x) for x in np.random.default_rng(3).random((6, len(space)))]
    low = F.LowFidelity(scn, space)
    got = low(Pp)
    assert low.throughput()['evaluations'] == 6 and low.throughput()['per_s'] > 0
    for i, p in enumerate(Pp):
        tr = F.rollout(scn, p, 1)
        ref = CR.summarise(tr['ego'][0], {k: v[0] for k, v in tr['actors'].items()})
        pets = [a['pet_s'] for a in ref['actors'].values() if a['pet_s'] is not None]
        tol = 0.9 * math.exp(-F.PET_MAX_S) + 1e-9 if any(x > F.PET_MAX_S for x in pets) else 1e-9
        assert got[i] == pytest.approx(ref['criticality'], abs=tol), (key, p)
    # the measures themselves, one batch against summarise
    vals = {k: np.array([p[k] for p in Pp]) for k in scn['params']}
    tr = F.rollout(scn, vals, len(Pp))
    c = F.criticality(tr)
    for i in range(len(Pp)):
        ref = CR.summarise(tr['ego'][i], {k: v[i] for k, v in tr['actors'].items()})
        for aid, a in ref['actors'].items():
            m = c['actors'][aid]
            if a['min_ttc_s'] is None:
                assert math.isinf(m['min_ttc_s'][i])
            else:
                assert m['min_ttc_s'][i] == pytest.approx(a['min_ttc_s'], abs=1e-9)
            assert m['min_distance_m'][i] == pytest.approx(a['min_distance_m'], abs=1e-9)
            assert m['max_drac_mps2'][i] == pytest.approx(a['max_drac_mps2'], abs=1e-9)
            assert bool(m['collision'][i]) == a['collision']
            if a['pet_s'] is None or a['pet_s'] > F.PET_MAX_S:
                assert math.isinf(m['pet_s'][i])
            else:
                assert m['pet_s'][i] == pytest.approx(a['pet_s'], abs=1e-9)
        assert bool(c['collision'][i]) == ref['overall']['collision']


def test_a_choice_parameter_groups_the_batch_and_weather_is_ignored():
    scn = SC.builtin('stalled_vehicle')
    space = SC.search_space(scn) + [{'name': 'weather.fog_density', 'lo': 0.0, 'hi': 100.0}]
    low = F.LowFidelity(scn, space)
    assert low.ignored == ['weather.fog_density']
    Pp = [{'speed': 60.0, 'distance': 45.0, 'kind': k, 'weather.fog_density': 50.0} for k in ('car', 'truck', 'car')]
    v = low(Pp)
    assert v[0] == v[2] and v[0] != v[1]
    with pytest.raises(ValueError, match='missing: distance'):
        low([{'speed': 60.0, 'kind': 'car'}])


def test_the_torch_back_end_gives_the_same_rollout():
    pytest.importorskip('torch')
    scn = SC.builtin('cut_in')
    space = SC.search_space(scn)
    Pp = [S.to_params(space, x) for x in np.random.default_rng(0).random((5, len(space)))]
    vals = {k: np.array([p[k] for p in Pp]) for k in scn['params']}
    a, b = F.rollout(scn, vals, 5), F.rollout(scn, vals, 5, device='torch-cpu')
    assert b['device'] == 'cpu' and np.allclose(a['ego'], b['ego'], atol=1e-9)
    assert np.allclose(a['actors']['cutter'], b['actors']['cutter'], atol=1e-9)
    assert np.allclose(a['fired'][0], b['fired'][0], equal_nan=True)
    with pytest.raises(ValueError):
        F.backend('quantum')


# ================================================================================================ the statistics
def test_lhs_ranks_spearman_and_isotonic():
    X = F.lhs(50, 3, np.random.default_rng(0))
    for j in range(3):                                            # one point per stratum in every dimension
        assert sorted(np.floor(X[:, j] * 50).astype(int)) == list(range(50))
    assert list(F.ranks([10, 20, 20, 5])) == [2.0, 3.5, 3.5, 1.0]
    assert F.spearman([1, 2, 3, 4], [10, 20, 25, 100]) == pytest.approx(1.0)
    assert F.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # by hand: ranks (1, 2, 3, 4, 5) and (2, 1, 4, 3, 5): Σd² = 4, ρ = 1 − 6·4 / (5·24) = 0.8
    assert F.spearman([1, 2, 3, 4, 5], [20, 10, 40, 30, 50]) == pytest.approx(0.8)
    assert F.spearman([1, 2], [1, 2]) is None and F.spearman([1, 2, 3], [5, 5, 5]) is None
    iso = F.Isotonic([1, 2, 3, 4, 5], [1, 3, 2, 4, 6])           # pool adjacent violators: (3, 2) -> 2.5, 2.5
    assert np.allclose(iso.predict([1, 2, 3, 4, 5]), [1, 2.5, 2.5, 4, 6])
    assert np.all(np.diff(iso.predict(np.linspace(0, 6, 50))) >= -1e-12)
    assert iso.predict([-5])[0] == 1 and iso.predict([50])[0] == 6
    rng = np.random.default_rng(1)
    x = rng.random(200)
    y = x ** 2 + 0.05 * rng.standard_normal(200)
    fit = F.Isotonic(x, y).predict(x)
    assert np.all(np.diff(fit[np.argsort(x)]) >= -1e-12) and np.mean((fit - x ** 2) ** 2) < 0.002


# ================================================================================================ the loop, planted
REGION = {'center': [0.78, 0.32, 0.25, 0.60], 'radii': [0.12, 0.18, 0.15, 0.24]}     # 0.4 % of the cube fails
BUDGET = 24


def _planted(seed):
    """The 'CARLA' landscape and a cheap model of it: the region a little displaced and wider, a monotone
    distortion (0.8·v^1.5) and noise."""
    high = S.PlantedObjective(REGION, seed=seed)
    cheap = S.PlantedObjective({'center': [0.76, 0.34, 0.27, 0.58], 'radii': [0.14, 0.2, 0.17, 0.26]},
                               seed=seed + 100, noise=0.03)
    return high, lambda Pp: np.array([0.8 * cheap(p) ** 1.5 for p in Pp])


def _two_speed(seed, low=None, **kw):
    high, lo = _planted(seed)
    calls = []

    def run(ps):
        calls.append(len(ps))
        return [high(p) for p in ps]
    ts = F.TwoSpeed(high.space, low or lo, 0.5, seed, screen=4000, refine=400, **kw)
    return F.run_two_speed(ts, run, BUDGET, 6), ts, calls


def _plain(seed, method):
    high, _ = _planted(seed)
    s = S.make(method, high.space, seed, threshold=0.5)
    n = 0
    while n < BUDGET:
        ps = s.ask(min(6, BUDGET - n))
        s.tell(ps, [high(p) for p in ps])
        n += len(ps)
    return len(S.failures(s.history, 0.5))


def test_two_speed_finds_failures_in_clearly_more_seeds_than_plain_search():
    seeds = range(10)
    two = [_two_speed(s) for s in seeds]
    found = [len(r['failures']) for r, _ts, _c in two]
    pof = [_plain(s, 'bayes_pof') for s in seeds]
    rnd = [_plain(s, 'random') for s in seeds]
    assert sum(f > 0 for f in found) >= 9, found                 # measured: 10 of 10 (13–20 failures each)
    assert sum(f > 0 for f in pof) <= 5 and sum(f > 0 for f in rnd) <= 5, (pof, rnd)     # measured: 3 and 3
    assert sum(f > 0 for f in found) >= sum(f > 0 for f in pof) + 4
    for r, ts, calls in two:
        assert sum(calls) == BUDGET == len(ts.pairs) and r['fallback'] is None
        f = r['fidelity']
        assert f['trust'] == 'trusted' and f['spearman'] >= 0.5 and f['pairs'] == BUDGET
        # a stated share of every round is random controls: 1 of each batch of 6 (20 % rounded, at least one)
        roles = [p['role'] for p in ts.pairs]
        assert roles.count('control') == 4 == f['controls'] and roles.count('top') == 20
        fn = f['false_negative']
        assert fn['dropped_controls'] <= 4 and fn['failed'] <= fn['dropped_controls']
        if fn['dropped_controls']:
            from falcon_autonomy.core.failrate import binom_upper
            assert fn['upper95'] == pytest.approx(binom_upper(fn['failed'], fn['dropped_controls']))
            assert fn['rate'] == fn['failed'] / fn['dropped_controls']
        assert f['selected']['runs'] == 20 and f['selected']['failures'] == len(r['failures']) - \
            sum(1 for p in ts.pairs if p['role'] == 'control' and p['high'] >= 0.5)
        assert f['isotonic'] and f['residual_rmse'] is not None
        # no two confirmed points closer than the diversity distance
        X = np.array([p['x'] for p in ts.pairs if p['role'] == 'top'])
        d = np.linalg.norm(X[:, None] - X[None], axis=-1) + np.eye(len(X))
        assert d.min() >= F.DIVERSITY * 2.0 - 1e-9
        assert [h['i'] for h in r['history']] == list(range(BUDGET))


def test_controls_are_predicted_before_they_run_and_do_not_depend_on_the_fit():
    high, lo = _planted(0)
    ts = F.TwoSpeed(high.space, lo, 0.5, 0, screen=1000, refine=0)
    a = ts.ask(6)
    assert [c['role'] for c in a] == ['control'] + ['top'] * 5 and a[0]['round'] == 0
    assert a[0]['predicted'] == a[0]['low'] and a[1]['p_fail'] is None          # before any pair: the cheap value
    assert a[1]['low'] == pytest.approx(float(ts.pool_low.max()))
    ts.tell(a, [high(c['params']) for c in a])
    b = ts.ask(6)
    assert b[0]['round'] == 1 and b[1]['p_fail'] is not None                     # now the fitted prediction
    # the controls are the seed's, whatever was told: another history draws the same control for round 1
    ts2 = F.TwoSpeed(high.space, lo, 0.5, 0, screen=1000, refine=0)
    a2 = ts2.ask(6)
    ts2.tell(a2, [0.0] * 6)
    assert ts2.ask(6)[0]['params'] == b[0]['params']
    assert F.TwoSpeed(high.space, lo, 0.5, 0, screen=1000, refine=0, control_fraction=0.0).ask(4)[0]['role'] == 'top'
    assert [c['role'] for c in F.TwoSpeed(high.space, lo, 0.5, 0, screen=200, refine=0).ask(1)] == ['top']
    with pytest.raises(ValueError, match='control share'):
        F.TwoSpeed(high.space, lo, control_fraction=1.0)
    # the false-negative estimate counts a control the screen predicted below the threshold and CARLA failed
    ts3 = F.TwoSpeed(high.space, lo, 0.5, 0, screen=200, refine=0)
    c = ts3.ask(2)
    ts3.tell(c, [0.9, 0.9])
    fn = ts3.fidelity()['false_negative']
    assert c[0]['predicted'] < 0.5 and fn == dict(fn, dropped_controls=1, failed=1, rate=1.0, upper95=1.0,
                                                 control_failures=1, control_failures_kept=0)


def test_a_decorrelated_cheap_model_is_distrusted_and_the_search_falls_back():
    for seed in range(4):
        bad = S.PlantedObjective({'center': [0.2, 0.8, 0.8, 0.15], 'radii': [0.2, 0.2, 0.2, 0.2]}, seed=seed + 7,
                                 noise=0.05)
        r, ts, calls = _two_speed(seed, low=lambda Pp: np.array([bad(p) for p in Pp]))
        f = r['fidelity']
        assert f['trust'] == 'not trusted' and f['spearman'] < 0.5
        fb = r['fallback']
        assert fb['method'] == 'bayes-pof' and 'Spearman' in fb['why'] and fb['from_run'] == 12 and fb['runs'] == 12
        assert sum(calls) == BUDGET == len(ts.pairs)              # the budget is CARLA runs, spent in full
        assert [p['role'] for p in ts.pairs[12:]] == ['fallback'] * 12
        assert ts.trust() == 'not trusted'                        # final: it does not flip back
    # too few pairs: not judged; a cheap model that says the same everywhere: not judged (no rank correlation)
    high, lo = _planted(0)
    ts = F.TwoSpeed(high.space, lo, 0.5, 0, screen=300, refine=0)
    c = ts.ask(4)
    ts.tell(c, [high(x['params']) for x in c])
    assert ts.trust() == 'not judged yet'
    flat = F.TwoSpeed(high.space, lambda Pp: np.zeros(len(Pp)), 0.5, 0, screen=300, refine=0)
    r = F.run_two_speed(flat, lambda ps: [high(p) for p in ps], 12, 6)
    assert r['fidelity']['spearman'] is None and r['fidelity']['trust'] == 'not judged yet' and r['fallback'] is None


def test_prior_evaluations_are_pairs_and_a_failed_run_counts_to_the_budget():
    high, lo = _planted(1)
    ts = F.TwoSpeed(high.space, lo, 0.5, 1, screen=500, refine=0)
    for x in np.random.default_rng(0).random((5, 4)):
        p = S.to_params(high.space, x)
        ts.add_prior(p, high(p))
    assert [p['role'] for p in ts.pairs] == ['prior'] * 5 and ts.fidelity()['controls'] == 0
    r = F.run_two_speed(ts, lambda ps: [None] + [high(p) for p in ps[1:]], 11, 6)
    assert len(ts.pairs) == 11 and len(r['history']) == 10       # one run failed: spent, not in the history
    with pytest.raises(RuntimeError, match='every evaluation'):
        F.run_two_speed(F.TwoSpeed(high.space, lo, 0.5, 1, screen=300, refine=0), lambda ps: [None] * len(ps), 30, 6)


# ================================================================================================ on the fake carla
@pytest.fixture(scope='module')
def ws(tmp_path_factory):
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    base = tmp_path_factory.mktemp('fastsim')
    p = Project.new('Two-speed')
    rig = p.doc['rigs'][0]
    rig['sensors'] = rig['sensors'][:1]
    rig['sensors'][0].update(width=64, height=36, labels=[])
    p.save(str(base / 'p.falcon.json'))
    svc = CarlaService(Settings(str(base / 'settings.json')), JobQueue(SubprocessLauncher()), p.workspace,
                       fake_carla=FAKE)
    yield {'project': p, 'svc': svc, 'base': base}
    svc.shutdown()


def _study(ws, name='search-lead_brake', objective=None, key='lead_brake'):
    from falcon_autonomy.sim.studio import Study
    return Study(ws['svc'], ws['project'], SC.builtin(key), name, 'world-town10', 'rig-default', seed=11,
                 objective=objective or {'objective': 'criticality', 'count': 'all'})


def test_a_two_speed_study_is_an_ordinary_study_with_fidelity_fields(ws):
    st = _study(ws)
    logs = []
    d = F.run_study(st, 8, batch=4, threshold=0.7, screen=600, refine=100, min_pairs=6, log=logs.append)
    rows = st.results()
    runs = sorted(os.listdir(os.path.join(st.dir, 'runs')))
    assert len(rows) == 8 == len(runs) == d['spent'] and d['state'] == 'finished' and d['screened'] >= 700
    assert d['schema'] == F.PRESCREEN_SCHEMA and d == F.read_prescreen(st.dir)
    # rows as any search's, plus the fidelity record; the fake runs the director, which the fast simulator mirrors
    for r in rows:
        assert r['status'] == 'ok' and r['objective'] == r['criticality'] and r['objective_key'] == st.okey
        f = r['fidelity']
        assert f['role'] in ('top', 'control') and f['driver'] == 'follower' and 0.0 <= f['low'] <= 1.0
        assert abs(f['low'] - r['criticality']) < 0.1, r
    assert [r['fidelity']['role'] for r in rows] == ['control', 'top', 'top', 'top'] * 2
    assert d['trust'] == 'trusted' and d['fidelity']['spearman'] > 0.6 and d['fallback'] is None
    assert d['throughput']['per_s'] > 50 and d['throughput']['device'] == 'cpu'
    assert {p['run'] for p in d['pairs']} == {r['run'] for r in rows}
    assert all(f['value'] >= 0.7 and os.path.isdir(f['run']) for f in d['failures'])
    assert any('Spearman' in t for t in logs) and any('screened' in t for t in logs)
    s = json.load(open(os.path.join(st.dir, 'search-prescreen.json')))
    assert s['method'] == 'prescreen' and len(s['history']) == 8 and s['threshold'] == 0.7
    assert s['fidelity']['trust'] == 'trusted' and [h['i'] for h in s['history']] == list(range(8))
    assert s['failures'] == sorted(s['failures'], key=lambda h: -h['value'])
    # the other result readers read it as a search: the report, the ODD coverage's kind, the assistant's study tool
    from falcon_autonomy.core import oddcoverage as OC
    from falcon_autonomy.core import reports as R
    assert OC.study_kind(st.dir) == 'search' and OC.study_threshold(st.dir) == 0.7
    assert R.report_study(st.dir) is not None                   # builds: it reads search-prescreen.json and the rows
    # resumed with the same budget: nothing is run; with a larger one: only the difference, rounds continuing
    d2 = F.run_study(_study(ws), 8, batch=4, threshold=0.7, screen=600, refine=100, min_pairs=6)
    assert len(os.listdir(os.path.join(st.dir, 'runs'))) == 8 and d2['spent'] == 8
    assert d2['fidelity']['spearman'] == pytest.approx(d['fidelity']['spearman'])
    d3 = F.run_study(_study(ws), 10, batch=4, threshold=0.7, screen=600, refine=100, min_pairs=6)
    rows = _study(ws).results()
    assert len(rows) == 10 and d3['spent'] == 10 and [r['fidelity']['round'] for r in rows[8:]] == [2, 2]
    assert 'fidelity' in F.headline(d3) and 'trusted' in F.headline(d3)
    # an evaluation another search made is a pair (a prior), never a control
    plain = _study(ws)
    extra = S.to_params(plain.space(), [0.5] * len(plain.space()))
    plain.evaluate([extra])
    d4 = F.run_study(_study(ws), 11, batch=4, threshold=0.7, screen=600, refine=100, min_pairs=6)
    assert d4['spent'] == 11 and [p['role'] for p in d4['pairs']].count('prior') == 1
    with pytest.raises(F.Unsupported, match='perception'):
        st.spec = dict(st.spec, objective='perception')
        F.run_study(st, 4)


def test_the_commands(ws):
    p = ws['project']
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'prescreen', *a], capture_output=True,
                              text=True, env=env, timeout=300)
    r = cli('scenarios')
    assert r.returncode == 0 and 'lead_brake' in r.stdout and 'refused: the ego\'s route' in r.stdout
    r = cli('screen', '--scenario', 'cut_in', '--n', '300', '--top', '3', '--json')
    d = json.loads(r.stdout)
    assert r.returncode == 0 and len(d['top']) == 3 and d['top'][0]['low'] >= d['top'][2]['low']
    assert d['throughput']['evaluations'] == 300
    r = cli('bench', '--n', '64')
    assert r.returncode == 0 and json.loads(r.stdout)['per_s'] > 0
    r = cli('run', p.path, '--scenario', 'unprotected_turn', '--fake-carla', FAKE)
    assert r.returncode == P.REFUSED and 'cannot run unprotected_turn' in r.stdout and 'falcon search' in r.stdout
    r = cli('run', p.path, '--scenario', 'lead_brake', '--device', 'nope', '--fake-carla', FAKE)
    assert r.returncode == P.REFUSED
    r = cli('show', p.path, '--scenario', 'cut_in')
    assert r.returncode == P.FAILED and 'no two-speed result' in r.stdout
    r = cli('run', p.path, '--scenario', 'cut_in', '--budget', '4', '--batch', '4', '--screen', '300', '--refine',
            '0', '--count', 'all', '--fake-carla', FAKE)
    assert r.returncode == 0, r.stdout + r.stderr
    d = F.read_prescreen(os.path.join(p.workspace, 'studies', F.default_name('cut_in')))
    assert d['spent'] == 4 and d['study'] == 'search-cut_in' and d['trust'] == 'not judged yet'
    r = cli('show', p.path, '--scenario', 'cut_in')
    assert r.returncode == 0 and '4 of 4 CARLA runs' in r.stdout and 'not judged yet' in r.stdout
    argv = F.prescreen_argv(p.path, 'cut_in', 'search-cut_in', 4, 4, 0.5, 0, 'world-town10', 'rig-default',
                            fake_carla=FAKE)
    assert argv[:3] == ['prescreen', 'run', p.path] and '--on' not in argv and argv[-2:] == ['--fake-carla', FAKE]


def test_the_assistant_tools(ws):
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    assert REGISTRY['research_prescreen_run'].tier == 'act' and REGISTRY['research_prescreen_show'].tier == 'read'
    c = TOOLS.context(ws['project'].path, FAKE)
    try:
        r = call(c, 'research_prescreen_show', {})
        assert 'lead_brake' in r['covered'] and 'junction' in r['refused']['unprotected_turn']
        r = call(c, 'research_prescreen_show', {'scenario': 'lead_brake', 'failures': 2})    # the study made above
        assert r['study'] == 'search-lead_brake' and r['trust'] == 'trusted' and r['carla_runs'] == 11
        assert r['pairs'] == 11 and r['spearman'] > 0.6 and r['false_negative']['dropped_controls'] >= 0
        assert len(r['failures']) <= 2 and all(f['run'].startswith('studies/search-lead_brake/runs/')
                                               for f in r['failures'])
        assert 'screen, never a result' in r['limits'] and len(json.dumps(r)) < 4000
        assert 'no two-speed result' in call(c, 'research_prescreen_show', {'scenario': 'stray_cattle'})['error']
        assert 'cannot run unprotected_turn' in call(c, 'research_prescreen_run',
                                                     {'scenario': 'unprotected_turn'})['error']
        t = REGISTRY['research_prescreen_run']
        args = {'scenario': 'stray_cattle', 'budget': 4, 'batch': 4, 'screen': 300, 'count': 'all'}
        assert 'prescreen run' in t.preview(c, **args) and '--screen 300' in t.preview(c, **args)
        assert '4 ×' in t.estimate(c, **args)['what'] and 'cheap candidates' in t.estimate(c, **args)['what']
        r = call(c, 'research_prescreen_run', args)
        assert r['study'] == 'search-stray_cattle' and 'prescreen run' in r['command']
        w = call(c, 'jobs_wait', {'jobs': [r['job']], 'timeout_s': 200})
        assert w['finished'] == [r['job']] and w['jobs'][0]['state'] == 'done', w
        r = call(c, 'research_prescreen_show', {'scenario': 'stray_cattle'})
        assert r['carla_runs'] == 4 and r['state'] == 'finished' and r['trust'] == 'not judged yet'
    finally:
        c.service.shutdown(grace_s=10.0)
