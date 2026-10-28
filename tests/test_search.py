"""Rare-case search: the parameter space, ask/tell, determinism, resume, the failure map, and the acceptance test —
on a planted weakness every searcher finds failures sooner than random sampling with the same budget."""
import json
import math

import numpy as np
import pytest

from falcon_autonomy.core import search as S

SPACE = [{'name': 'speed', 'lo': 5.0, 'hi': 25.0}, {'name': 'gap_m', 'lo': 2.0, 'hi': 40.0},
         {'name': 'weather', 'choices': ['ClearNoon', 'WetNoon', 'HardRainNight']}]


def toy(p):
    """Smooth, with a peak at speed 20, gap 8, worst in rain at night."""
    w = {'ClearNoon': 0.0, 'WetNoon': 0.1, 'HardRainNight': 0.3}[p['weather']]
    return math.exp(-((p['speed'] - 20) / 5) ** 2 - ((p['gap_m'] - 8) / 8) ** 2) * 0.7 + w


def test_space_maps_to_the_unit_cube_and_back():
    sp = S.normalise_space(SPACE)
    p = S.to_params(sp, [0.25, 1.0, 0.5])
    assert p == {'speed': 10.0, 'gap_m': 40.0, 'weather': 'WetNoon'}
    assert np.allclose(S.to_unit(sp, p), [0.25, 1.0, 0.5])
    assert [S.to_params(sp, [0, 0, u])['weather'] for u in (0.0, 0.33, 0.34, 0.99, 1.0)] == \
        ['ClearNoon', 'ClearNoon', 'WetNoon', 'HardRainNight', 'HardRainNight']
    assert S.to_unit(sp, {'speed': 99, 'gap_m': 2, 'weather': 'HardRainNight'}).tolist() == [1.0, 0.0, 5 / 6]
    with pytest.raises(ValueError):
        S.to_unit(sp, {'speed': 5, 'gap_m': 2, 'weather': 'Snow'})
    with pytest.raises(ValueError):
        S.normalise_space([{'name': 'a', 'lo': 1, 'hi': 1}])


@pytest.mark.parametrize('method', list(S.METHODS))
def test_every_method_is_deterministic_by_seed_and_handles_choices(method):
    def go(seed):
        s = S.make(method, SPACE, seed, threshold=0.8)
        S.run(s, toy, 30)
        return s

    a, b, c = go(3), go(3), go(4)
    assert [h['params'] for h in a.history] == [h['params'] for h in b.history]
    assert [h['value'] for h in a.history] == [h['value'] for h in b.history]
    assert [h['params'] for h in a.history] != [h['params'] for h in c.history]
    assert len(a.history) == 30 and [h['i'] for h in a.history] == list(range(30))
    assert all(h['params']['weather'] in SPACE[2]['choices'] and 5 <= h['params']['speed'] <= 25
               for h in a.history)
    best = a.best(3)
    assert [h['value'] for h in best] == sorted((h['value'] for h in a.history), reverse=True)[:3]


@pytest.mark.parametrize('method', list(S.METHODS))
def test_ask_tell_batches_pending_and_failed_runs(method):
    s = S.make(method, SPACE, 0, threshold=0.8)
    asked = s.ask(5)
    assert len(asked) == 5 and len(s._pending) == 5
    more = s.ask(3)                                   # asked before any tell: pending grows, points differ
    assert len(s._pending) == 8 and all(p not in asked for p in more)
    s.tell(asked[:2], [toy(p) for p in asked[:2]])
    s.tell(asked[2:], [toy(asked[2]), None, toy(asked[4])])   # one run failed: not recorded, no longer pending
    assert len(s.history) == 4 and len(s._pending) == 3
    s.tell(more, [toy(p) for p in more])
    assert len(s.history) == 7 and not s._pending
    # an evaluation that was never asked (an earlier campaign) is accepted
    s.tell([{'speed': 20.0, 'gap_m': 8.0, 'weather': 'HardRainNight'}], [1.0])
    assert s.best(1)[0]['value'] == 1.0
    with pytest.raises(ValueError):
        s.tell([more[0]], [float('nan')])


def test_cmaes_generation_does_not_depend_on_how_results_are_told():
    one, many = S.CMAES(SPACE, seed=5), S.CMAES(SPACE, seed=5)
    assert one.lam == 4 + int(3 * math.log(3)) == 7
    ps = one.ask(one.lam)
    assert many.ask(many.lam) == ps
    one.tell(ps, [toy(p) for p in ps])
    for p in ps:
        many.tell([p], [toy(p)])
    assert one.gen == many.gen == 1
    assert np.allclose(one.mean, many.mean) and np.allclose(one.C, many.C) and one.sigma == many.sigma
    assert one.ask(4) == many.ask(4)


@pytest.mark.parametrize('method', list(S.METHODS))
def test_state_round_trip_resumes_the_same_continuation(method):
    f = S.planted_objective(seed=1)

    def step(s):
        ps = s.ask(4)
        s.tell(ps, [f(p) for p in ps])

    straight = S.make(method, f.space, 2, threshold=0.5)
    for _ in range(12):
        step(straight)
    resumed = S.make(method, f.space, 2, threshold=0.5)
    for _ in range(5):
        step(resumed)
    ps = resumed.ask(4)                               # snapshot with asks still pending
    state = json.loads(json.dumps(resumed.state()))
    assert state['method'] == method and len(state['pending']) == 4
    resumed = S.from_state(state)
    resumed.tell(ps, [f(p) for p in ps])
    for _ in range(6):
        step(resumed)
    assert [h['params'] for h in resumed.history] == [h['params'] for h in straight.history]
    assert [h['value'] for h in resumed.history] == [h['value'] for h in straight.history]


def test_a_search_can_start_from_past_evaluations():
    past = S.RandomSearch(SPACE, seed=9)
    S.run(past, toy, 20)
    warm = S.BayesOpt(SPACE, seed=0, history=past.history)
    assert len(warm.history) == 20 and warm.best(1)[0]['value'] == past.best(1)[0]['value']
    warm2 = S.CMAES(SPACE, seed=0, history=[(h['params'], h['value']) for h in past.history])
    assert warm2.gen == 20 // warm2.lam and len(warm2._buffer) == 20 % warm2.lam


def test_halton_sequence_is_low_discrepancy():
    s = S.RandomSearch([{'name': f'p{k}', 'lo': 0, 'hi': 1} for k in range(3)], seed=0, sequence='halton')
    X = np.asarray([[p[f'p{k}'] for k in range(3)] for p in s.ask(64)])
    for d in range(3):
        counts = np.histogram(X[:, d], bins=8, range=(0, 1))[0]
        assert counts.min() >= 7 and counts.max() <= 9          # uniform draws would spread far wider


def test_failures_and_failure_map():
    hist = [{'params': {'speed': 6.0, 'gap_m': 3.0, 'weather': 'ClearNoon'}, 'value': 0.2, 'i': 0},
            {'params': {'speed': 7.0, 'gap_m': 3.5, 'weather': 'ClearNoon'}, 'value': 0.9, 'i': 1},
            {'params': {'speed': 24.0, 'gap_m': 39.0, 'weather': 'HardRainNight'}, 'value': 0.6, 'i': 2},
            {'params': {'speed': 24.5, 'gap_m': 21.5, 'weather': 'WetNoon'}, 'value': 0.95, 'i': 3}]
    assert [h['i'] for h in S.failures(hist, 0.6)] == [3, 1, 2]
    m = S.failure_map(hist, SPACE, 'speed', 'gap_m', bins=2, threshold=0.6)
    assert m['x_edges'] == [5.0, 15.0, 25.0] and m['y_edges'] == [2.0, 21.0, 40.0]
    assert m['count'] == [[2, 0], [0, 2]]                      # rows: gap bins; columns: speed bins
    assert m['max'] == [[0.9, None], [None, 0.95]]
    assert m['mean'][0][0] == pytest.approx(0.55) and m['mean'][1][1] == pytest.approx(0.775)
    assert m['failure_share'] == [[0.5, None], [None, 1.0]]
    w = S.failure_map(hist, SPACE, 'weather', 'speed', bins=(4, 5), threshold=0.6)
    assert len(w['count']) == 5 and len(w['count'][0]) == 3     # one column per choice
    assert w['x_labels'] == SPACE[2]['choices'] and sum(map(sum, w['count'])) == 4
    assert S.failure_map(hist, SPACE, 'speed', 'gap_m', bins=3)['failure_share'] is None


def test_planted_objective_is_what_it_says():
    f = S.planted_objective()
    n = len(f.space)
    assert f.volume == pytest.approx(math.pi ** 2 / 2 * 0.2 * 0.3 * 0.25 * 0.4)      # 2.96 % of the cube
    centre = S.to_params(f.space, f.center)
    assert f.clean(centre) == pytest.approx(0.98) and f.inside(centre)
    edge = S.to_params(f.space, f.center + np.eye(n)[1] * f.radii[1])
    assert f.clean(edge) == pytest.approx(0.5)
    assert f(centre) == f(centre) != f.clean(centre)                                 # noise, repeatable
    U = np.random.default_rng(0).random((4000, n))
    vals = np.asarray([f(S.to_params(f.space, u)) for u in U])
    assert abs(np.mean(vals >= 0.5) - f.volume) < 0.012
    assert np.median(vals) < 0.1


@pytest.fixture(scope='module')
def comparison():
    f = S.planted_objective()
    return S.compare(f, f.space, budget=120, seeds=range(5), threshold=0.5, n_failures=5)


def test_searchers_find_the_planted_failure_region_sooner_than_random(comparison):
    r = comparison['methods']
    print('\nmethod     first failure   5th failure   failures at 120 (mean of 5 seeds)')
    for name, m in r.items():
        print(f'{name:10s} {m["first_failure_mean"]:13.1f} {m["to_n_failures_mean"]:13.1f} '
              f'{m["failures_at_budget"]:17.1f}   per seed {m["failures_per_seed"]}')
    rnd = r['random']
    for name in ('bayesopt', 'cmaes', 'adaptive'):
        m = r[name]
        assert len(m['failures']) == len(m['best']) == 120
        assert all(v is not None for v in m['to_n_failures'])            # every seed found ≥ 5 failures
        assert m['to_n_failures_mean'] < 0.6 * rnd['to_n_failures_mean']
        assert m['failures_at_budget'] >= 4 * rnd['failures_at_budget']
        assert m['failures'][59] > rnd['failures'][-1]                   # with half the budget, more than random
    assert np.all(np.diff(rnd['failures']) >= 0) and np.all(np.diff(rnd['best']) >= 0)
