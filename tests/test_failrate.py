"""The failure-rate estimator (core.failrate): the nominal's transforms, the binomial bounds, the trust rules, the
sampler's exact resume, the per-cell ratio estimator, the exposure units — and the estimator validated on synthetic
problems with a known probability (1e-4 and 1e-6): its 95 % interval covers the truth at about the nominal rate over
seeds, where plain Monte Carlo with the same runs sees no failure at all. Then an estimate end to end on the fake
CARLA through `falcon failrate`, resumed, and read by the safety case."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from falcon_autonomy.core import failrate as F
from falcon_autonomy.core import scenarios as SC


# ------------------------------------------------------------------------------------------------ the nominal
def test_nominal_transforms_each_declared_distribution():
    nom = F.Nominal([{'name': 'u', 'dist': 'uniform', 'lo': 10.0, 'hi': 20.0},
                     {'name': 'n', 'dist': 'normal', 'mean': 5.0, 'sd': 2.0},
                     {'name': 't', 'dist': 'normal', 'mean': 0.0, 'sd': 1.0, 'lo': 0.0, 'hi': 1.0},
                     {'name': 'c', 'dist': 'choice', 'choices': ['car', 'truck', 'van']}])
    z = np.random.default_rng(0).standard_normal((20000, 4))
    xs = [nom.to_params(r) for r in z]
    u = np.array([x['u'] for x in xs])
    n = np.array([x['n'] for x in xs])
    t = np.array([x['t'] for x in xs])
    assert 10 <= u.min() and u.max() <= 20 and abs(u.mean() - 15) < 0.1          # uniform on its range
    assert abs(n.mean() - 5) < 0.05 and abs(n.std() - 2) < 0.05                   # the normal itself
    assert 0 <= t.min() and t.max() <= 1 and abs(t.mean() - 0.4599) < 0.01       # truncated to [0, 1]
    share = {k: sum(x['c'] == k for x in xs) / len(xs) for k in ('car', 'truck', 'van')}
    assert all(abs(v - 1 / 3) < 0.015 for v in share.values())                    # a choice: uniform over its cells
    assert nom.to_params([0, 0, 0, 0])['u'] == pytest.approx(15.0)


def test_nominal_of_a_study_is_the_declared_distributions_narrowed_to_the_feasible_part():
    from falcon_autonomy.core.rss import feasible_space
    scn = SC.builtin('lead_brake')
    scn['params']['decel'] = {'dist': 'uniform', 'lo': 3.0, 'hi': 12.0, 'unit': 'm/s²'}
    space, notes = feasible_space(scn, SC.search_space(scn))
    nom = F.Nominal.from_scenario(scn, space + [{'name': 'weather.fog_density', 'lo': 0.0, 'hi': 100.0}])
    d = {x['name']: x for x in nom.doc()}
    assert d['decel']['hi'] == 9.0 and d['decel']['mass'] == pytest.approx(6.0 / 9.0)
    assert 'narrowed' in d['decel']['source']
    assert d['weather.fog_density']['dist'] == 'uniform' and 'weather' in d['weather.fog_density']['source']
    assert nom.mass == pytest.approx(6.0 / 9.0) and notes
    with pytest.raises(F.FailRateError):
        F.Nominal([])


# ------------------------------------------------------------------------------------------------ bounds and trust
def test_binomial_bounds_and_the_rule_of_three():
    assert F.binom_upper(0, 100) == pytest.approx(1 - 0.05 ** 0.01)              # exact zero-failure bound
    assert F.binom_upper(0, 100) == pytest.approx(3 / 100, rel=0.02)               # ≈ the rule of three
    assert F.binom_upper(1, 10) == pytest.approx(0.3942, abs=2e-4)                 # Clopper–Pearson, one-sided
    assert F.binom_lower(1, 10) == pytest.approx(0.00512, abs=2e-5)
    assert F.runs_needed(0.05) == 59 and F.runs_needed(0.01) == 299
    assert F.binom_upper(2, F.runs_needed(0.05, failures=2)) <= 0.05 < F.binom_upper(
        2, F.runs_needed(0.05, failures=2) - 1)


def test_trust_rules_refuse_too_few_effective_failures_and_bound_zero():
    z = F.summary(np.ones(300), np.zeros(300), defensive=0.1)
    assert z['status'] == 'refused' and z['upper95'] == pytest.approx(0.1) and 'rule of three' in z['reasons'][1]
    few = F.summary(np.full(100, 0.01), np.arange(100) < 5)
    assert few['status'] == 'refused' and 'only 5 failing runs' in few['reasons'][0]
    w = np.full(200, 1e-3)
    w[0] = 0.5                                                   # one run carries the failures' weight
    heavy = F.summary(w, np.arange(200) < 40)
    assert heavy['status'] == 'refused' and 'effective sample size' in heavy['reasons'][0]
    ok = F.summary(np.full(400, 1e-3), np.arange(400) < 200)
    assert ok['status'] == 'ok' and ok['p'] == pytest.approx(5e-4) and ok['ci95'][0] < 5e-4 < ok['ci95'][1]


# ------------------------------------------------------------------------------------------------ the sampler
def test_sampler_resumes_exactly_and_replaces_lost_runs():
    nom, f, c = F.tail_problem(1e-3, 3)
    a = F.Sampler(nom, threshold=c, seed=4, stage_size=60, final_size=80)
    full = F.run(a, f, batch=7)
    b = F.Sampler(nom, threshold=c, seed=4, stage_size=60, final_size=80)
    for _ in range(9):                                           # part of the way, then a snapshot through JSON
        asked = b.ask(7)
        b.tell([x['id'] for x in asked], f([x['params'] for x in asked]))
    pending = b.ask(5)                                           # asked, never told: re-asked after the resume
    b2 = F.Sampler.from_state(json.loads(json.dumps(b.state())))
    again = b2.ask(5)
    assert [x['id'] for x in again] == [x['id'] for x in pending] and again[0]['params'] == pending[0]['params']
    b2.tell([x['id'] for x in again], f([x['params'] for x in again]))
    res = F.run(b2, f, batch=7)
    assert res['estimate']['p'] == pytest.approx(full['estimate']['p'], rel=1e-12)
    lost = F.Sampler(nom, threshold=c, seed=1, stage_size=10, final_size=10, max_stages=1)
    asked = lost.ask(10)
    lost.tell([x['id'] for x in asked], [None] + f([x['params'] for x in asked[1:]]))
    assert lost.remaining() == 1 and lost.stage == 0             # the lost run is replaced by a new draw


def test_the_full_covariance_family_is_kept_only_for_its_measurement():
    nom, f, c = F.tail_problem(1e-4, 2)
    assert F.Sampler(nom, threshold=c).family == 'mean_shift'
    with pytest.raises(F.FailRateError):
        F.Sampler(nom, threshold=c, family='student')


@pytest.mark.parametrize('kind,p,dim', [('tail', 1e-4, 2), ('tail', 1e-6, 5), ('corner', 1e-5, 2)])
def test_interval_covers_the_known_probability_and_plain_monte_carlo_cannot(kind, p, dim):
    """The validation of the estimator itself: 100 seeds on a problem whose failure probability is known exactly."""
    rep = F.validate([p], [dim], range(100), stage_size=300, final_size=600, kinds=(kind,))
    row = rep['rows'][0]
    assert row['refused'] <= 2
    assert row['coverage'] >= 0.88, row                          # nominal 0.95; measured 0.92–0.98 (docs)
    assert row['mean_abs_rel_error'] < 0.25
    assert row['mean_runs'] <= 3000
    assert row['plain_mc_zero_failure_share'] >= 0.75            # plain MC with the same runs: mostly no failure
    assert row['plain_mc_runs_for_same_rel_se'] > 100 * row['mean_runs']


def test_estimate_per_cell_is_the_ratio_estimator():
    """P(fail | cell) for a linear limit state split by the sign of z2, against the probability by quadrature."""
    nom, f, c = F.tail_problem(1e-3, 2)
    sampler = F.Sampler(nom, threshold=c, seed=3, stage_size=300, final_size=2000)
    F.run(sampler, f)
    fin = sampler.final_samples()
    cells = F.by_cell(fin, lambda s: 'up' if s['z'][1] > 0 else 'down')
    z = np.linspace(0, 12, 20001)
    phi = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    tail = np.array([0.5 * math.erfc((c * math.sqrt(2) - v) / math.sqrt(2)) for v in z])
    truth_up = float(np.trapezoid(phi * tail, z)) / 0.5
    up = cells['up']
    assert up['status'] in ('ok', 'warning') and abs(up['p'] - truth_up) < 4 * up['se']
    assert up['ci95'][0] < up['p'] < up['ci95'][1]
    assert cells['down']['runs'] + up['runs'] == len(fin)


def test_exposure_units_scale_the_rate_and_need_a_figure():
    est = {'p': 2e-6, 'ci95': [1e-6, 4e-6], 'status': 'ok'}
    r = F.exposure_rate(est, {'per': 'km', 'encounters': 0.5, 'source': 'a survey'})
    assert r['power'] == 6 and r['rate'] == pytest.approx(1.0) and r['ci95'] == pytest.approx([0.5, 2.0])
    assert 'per 10^6 km ' in r['text'] and r['source'] == 'a survey'
    assert F.exposure_rate(est, None)['text'].startswith('2 failures per 10^6 encounters')
    with pytest.raises(F.FailRateError):
        F.exposure_rate(est, {'per': 'km'})
    refused = F.exposure_rate({'status': 'refused', 'p': 0.0, 'upper95': 0.01}, {'per': 'hour', 'encounters': 10})
    assert refused['rate'] is None and refused['text'].startswith('at most 1 failures per 10^1 hours')


def test_check_command_prints_the_validation_table():
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'failrate', 'check', '--p', '1e-4', '--dims', '2',
                        '--seeds', '3', '--stage-size', '200', '--final', '300', '--problems', 'tail'],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert 'tail' in r.stdout and 'covered' in r.stdout and '1e-04' in r.stdout


# ------------------------------------------------------------------------------------------------ end to end
def test_an_estimate_on_the_fake_carla_resumes_and_is_evidence(project, tmp_path, fake_carla):
    project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *a], capture_output=True, text=True,
                              env=env, timeout=600)
    args = ('failrate', 'run', path, '--scenario', 'lead_brake', '--stage-size', '4', '--final', '4',
            '--max-stages', '1', '--batch', '4', '--exposure-per', 'km', '--encounters', '0.2',
            '--exposure-source', 'test figure', '--fake-carla', fake_carla)
    r = cli(*args)
    assert r.returncode == 0, r.stdout + r.stderr
    d = os.path.join(project.workspace, 'studies', 'rate-lead_brake')
    res = json.load(open(os.path.join(d, 'failure-rate.json')))
    assert res['schema'] == F.SCHEMA and res['done'] and res['runs']['total'] == 8
    assert res['estimate']['status'] == 'refused'                # 4 final runs: too few to trust, and it says so
    assert res['judge']['kind'] == 'objective' and res['nominal'][0]['dist'] == 'uniform'
    assert all(x['run'] and os.path.isfile(os.path.join(x['run'], 'run.json')) for x in res['final_runs'])
    assert res['by_cell'] and all(k.count('/') == 4 for k in res['by_cell'])
    rows = [json.loads(x) for x in open(os.path.join(d, 'results.jsonl'))]
    assert len(rows) == 8
    r = cli(*args)                                              # resumed: nothing driven again
    assert r.returncode == 0 and len(open(os.path.join(d, 'results.jsonl')).readlines()) == 8
    r = cli('failrate', 'show', d)
    assert r.returncode == 0 and r.stdout.startswith('refused after 8 runs')
    from falcon_autonomy.core import safetycase as SCS
    rates = SCS.failure_rates([d])
    assert rates[0]['study'] == 'rate-lead_brake' and rates[0]['risk']['kind'] == 'insufficient_evidence'
    html = SCS._failure_rates_html(rates)
    assert 'Failure rates' in html and 'rate-lead_brake' in html and 'refused' in html


def test_safety_case_reads_a_given_estimate(tmp_path):
    from falcon_autonomy.core import safetycase as SCS
    d = tmp_path / 'studies' / 'rate-x'
    d.mkdir(parents=True)
    est = F.summary(np.full(400, 1e-3), np.arange(400) < 200)
    (d / 'failure-rate.json').write_text(json.dumps({
        'schema': F.SCHEMA, 'scenario': 'cut_in', 'scenario_name': 'Cut-in', 'estimate': est,
        'runs': {'total': 700}, 'judge': {'about': 'x', 'threshold': 0.5},
        'rate': F.exposure_rate(est, {'per': 'km', 'encounters': 0.1})}))
    r = SCS.failure_rates([str(d)], max_failure_rate=0.05)[0]
    assert r['risk']['kind'] == 'failure_rate' and r['risk']['severity'] == 'context'
    assert 'fails in 0.0005 of encounters' in r['risk']['text'] and 'per 10^5 km' in r['risk']['text']
