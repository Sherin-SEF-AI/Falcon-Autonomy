"""The release gate (core.releasegate): the inequality the confidence sequence rests on, checked on a grid; the
empirical-Bernstein sequence's coverage under optional stopping; the mixture test of the discordant pairs against its
exact formula; the type-I error of the whole gate by simulation (no difference, a look after every trial: the false
"better / worse" rate stays within α); its power and what equivalence costs; the gate with planted versions (early
stop, the verdicts, newly failing scenarios with their run pairs, resume, a changed design refused); versions and
trials (common random numbers); CI mode's gate and the safety case's evidence; end to end on the fake carla."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKE
from test_gui import qapp  # noqa: F401 (the GUI tests' application)

from falcon_autonomy.core import releasegate as G
from falcon_autonomy.core import scenarios as SC


# ================================================================================================ the statistics
def test_the_inequality_behind_the_confidence_sequence_holds_on_a_grid():
    # Fan, Grama & Liu (2015): exp(λξ − ψ(λ)ξ²) ≤ 1 + λξ for ξ ≥ −1, 0 ≤ λ < 1 — the module's proof starts here
    lam = np.linspace(0.0, 0.999, 400)[:, None]
    xi = np.linspace(-1.0, 6.0, 1401)[None, :]
    lhs = np.exp(lam * xi - G.psi_e(lam) * xi ** 2)
    assert np.all(lhs <= 1.0 + lam * xi + 1e-12)
    assert np.allclose(np.exp(-lam - G.psi_e(lam)), 1.0 - lam)          # equality at ξ = −1
    assert G.psi_e(0.5) == pytest.approx(math.log(2) - 0.5)


def test_the_empirical_bernstein_sequence_covers_at_every_time():
    rng = np.random.default_rng(0)
    reps, n, alpha = 3000, 300, 0.1
    for draw, mean in ((lambda: rng.choice([-1.0, 0.0, 1.0], p=[0.1, 0.7, 0.2], size=(reps, n)), 0.1),
                       (lambda: 2.0 * rng.beta(2.0, 5.0, size=(reps, n)) - 1.0, 2.0 * 2.0 / 7.0 - 1.0),
                       (lambda: np.zeros((reps, n)), 0.0)):
        x = draw()
        lo, hi = G.eb_sequence(x, alpha)
        assert lo.shape == x.shape and np.all(np.diff(lo, axis=1) >= -1e-12) and np.all(np.diff(hi, axis=1) <= 1e-12)
        miss = np.any((lo > mean + 1e-12) | (hi < mean - 1e-12), axis=1)     # at ANY time: optional stopping
        assert miss.mean() <= alpha, miss.mean()
        assert np.all(lo >= -1.0) and np.all(hi <= 1.0)
    # it narrows: constant zeros reach ±17.6/t at α = 0.025 (the docs' "17.5 / margin trials")
    lo, hi = G.eb_sequence(np.zeros(400), 0.025)
    assert hi[99] == pytest.approx(4.0 * math.log(80.0) / 100.0, rel=0.02) and hi[349] < 0.0505 < hi[339]
    assert np.allclose(lo, -hi)
    # one sequence, another range
    lo, hi = G.eb_sequence(np.full(200, 3.0), 0.05, lo=0.0, hi=4.0)
    assert lo[-1] < 3.0 < hi[-1] and hi[-1] - lo[-1] < 0.6
    with pytest.raises(ValueError, match='outside'):
        G.eb_sequence([0.0, 2.0])
    assert G.eb_sequence([])[0].shape == (0,)


def test_the_mixture_test_of_the_discordant_pairs_is_its_formula():
    for n in (1, 5, 12, 40):
        for k in range(n + 1):
            exact = math.lgamma(k + 1) + math.lgamma(n - k + 1) - math.lgamma(n + 2)       # log B(k+1, n−k+1)
            assert float(G.log_mixture(k, n)) == pytest.approx(exact, abs=1e-9)
            assert float(G.discordant_log_e(k, n)) == pytest.approx(exact + n * math.log(2), abs=1e-9)
    # E = 2^N / ((N + 1) C(N, K)): nine pairs one way reach 1/0.025, eight do not
    assert math.exp(float(G.discordant_log_e(9, 9))) == pytest.approx(512 / 10)
    assert math.exp(float(G.discordant_log_e(8, 8))) == pytest.approx(256 / 9) and 256 / 9 < 40 < 512 / 10
    # the interval is the test inverted: 1/2 is outside exactly when E ≥ 1/α
    for k, n in ((9, 9), (8, 8), (0, 9), (12, 20), (16, 20), (30, 40), (3, 3)):
        lo, hi = G.discordant_interval(k, n, 0.025)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
        assert (lo > 0.5 or hi < 0.5) == (float(G.discordant_log_e(k, n)) >= math.log(40.0)), (k, n)
    assert G.discordant_interval(0, 0) == (0.0, 1.0)
    a, b = G.discordant_interval(50, 100, 0.05), G.discordant_interval(500, 1000, 0.05)
    assert b[1] - b[0] < a[1] - a[0] < 0.5
    # under θ = 1/2 with a look after every pair, E reaches 1/α in at most α of the sequences (Ville)
    rng = np.random.default_rng(1)
    k = np.cumsum(rng.random((4000, 400)) < 0.5, axis=1)
    n = np.arange(1, 401)[None, :]
    assert np.mean(np.any(G.discordant_log_e(k, n) >= math.log(20.0), axis=1)) <= 0.05


@pytest.mark.parametrize('p,rho', [(0.1, 0.0), (0.5, 0.0), (0.3, 0.6), (0.02, 0.0)])
def test_type_one_error_under_optional_stopping(p, rho):
    """No difference between the versions, a look after every trial for 400 trials, 2000 gates: the share that says
    better or worse stays within α (plus three standard errors of the Monte Carlo)."""
    alpha, reps = 0.05, 2000
    r = G.simulate_verdicts(p, p, 400, reps, alpha, margin=0.02, seed=11, rho=rho)
    wrong = r['better'] + r['worse']
    assert wrong <= alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / reps), r
    assert r['better'] + r['worse'] + r['equivalent'] + r['cant_tell'] == pytest.approx(1.0)


def test_type_one_error_of_the_value_outcome():
    rng = np.random.default_rng(5)
    reps, n, alpha = 2000, 400, 0.05
    common = rng.random((reps, n))
    a = np.clip(0.6 * common + 0.4 * rng.random((reps, n)), 0, 1)
    b = np.clip(0.6 * common + 0.4 * rng.random((reps, n)), 0, 1)
    lo, hi = G.eb_sequence(b - a, alpha)
    assert np.mean(np.any((lo > 0) | (hi < 0), axis=1)) <= alpha


def test_power_equivalence_and_the_simulation_is_the_gates_own_rule():
    r = G.simulate_verdicts(0.1, 0.5, 200, 500, seed=2)
    assert r['worse'] >= 0.95 and r['better'] == 0.0 and r['median_trials'] <= 60
    r = G.simulate_verdicts(0.5, 0.1, 200, 500, seed=2)
    assert r['better'] >= 0.95 and r['worse'] == 0.0
    r = G.simulate_verdicts(0.0, 0.0, 120, 50, margin=0.2)
    assert r['equivalent'] == 1.0 and r['median_trials'] == 88          # ≈ 17.5 / margin
    r = G.simulate_verdicts(0.1, 0.4, 300, 1000, margin=0.2, seed=3)   # a real difference beyond the margin
    assert r['equivalent'] <= 0.05
    r = G.simulate_verdicts(0.2, 0.2, 60, 300, margin=0.05, seed=4)
    assert r['cant_tell'] > 0.9                                         # too few trials: it says so
    # analyse() — what a gate runs — gives the simulation's verdict, sequence by sequence
    rng = np.random.default_rng(9)
    for p_a, p_b in ((0.1, 0.5), (0.3, 0.3), (0.6, 0.2), (0.0, 0.0)):
        fa, fb = rng.random(150) < p_a, rng.random(150) < p_b
        an = G.analyse(fa, fb, fa * 1.0, fb * 1.0, margin=0.2)
        d = fb.astype(float) - fa
        lo, hi = G.eb_sequence(d, 0.025)
        b_only, nd = np.cumsum(fb & ~fa), np.cumsum(fb ^ fa)
        rej = G.discordant_log_e(b_only, nd) >= math.log(40.0)
        worse = (lo > 0) | (rej & (2 * b_only > nd))
        better = (hi < 0) | (rej & (2 * b_only < nd))
        eq = (-0.2 < lo) & (hi < 0.2)
        any_ = worse | better | eq
        if not any_.any():
            assert an['verdict'] == 'cant_tell' and an['decided_at'] is None
            continue
        k = int(any_.argmax())
        assert an['decided_at'] == k + 1
        assert an['verdict'] == ('worse' if worse[k] else 'better' if better[k] else 'equivalent')


def test_analyse_reports_the_bounds_the_margins_and_the_secondary_outcome():
    n = 120
    fa = np.zeros(n, bool)
    fb = np.zeros(n, bool)
    fb[::4] = True                                                  # B fails every fourth trial, A never
    an = G.analyse(fa, fb, np.full(n, 0.2), np.where(fb, 1.0, 0.2))
    assert an['verdict'] == 'worse' and an['word'] == 'worse' and an['decided_at'] == 33    # the 9th discordant pair
    assert an['decided_by'] == 'discordant'
    assert an['discordant'] == dict(an['discordant'], b_only=9, a_only=0, n=9, theta=1.0)
    assert an['discordant']['interval'][0] > 0.5 and an['failures'] == {'a': 0, 'b': 9}
    assert an['risk']['lower'] < an['risk']['estimate'] < an['risk']['upper'] and an['risk']['alpha'] == 0.025
    assert len(an['path']['trial']) == n == len(an['path']['risk_upper']) and an['value']['alpha'] == 0.05
    assert an['path']['theta_lower'][31] <= 0.5 < an['path']['theta_lower'][32]
    assert G.analyse(fa, fb, np.zeros(n), np.zeros(n), min_trials=50)['decided_at'] == 50
    # the value as the primary outcome: B is more critical by 0.3 in every pair
    an = G.analyse(fa, fa, np.full(n, 0.2), np.full(n, 0.5), primary='value')
    assert an['verdict'] == 'worse' and an['value']['lower'] > 0 and an['risk']['alpha'] == 0.05
    assert an['decided_by'] == 'value'
    an = G.analyse(fa, fa, np.full(n, 0.5), np.full(n, 0.2), primary='value')
    assert an['verdict'] == 'better'
    # no difference at all: equivalent once the interval is inside the margin; not before; non-inferiority sooner
    an = G.analyse(fa, fa, np.zeros(n), np.zeros(n), margin=0.2)
    assert an['verdict'] == 'equivalent' and an['decided_at'] == 88 and an['within_margin']
    assert G.analyse(fa, fa, np.zeros(n), np.zeros(n), margin=0.05)['verdict'] == 'cant_tell'
    an = G.analyse(fa, fa, np.zeros(n), np.zeros(n), margin=0.05, non_inferiority=0.25)
    assert an['verdict'] == 'non_inferior' and an['word'] == 'not worse within the margin' and an['decided_at'] < 88
    assert G.analyse([], [], [], [])['verdict'] == 'cant_tell'
    with pytest.raises(G.GateError):
        G.analyse(fa, fb, fa, fb, primary='speed')


# ================================================================================================ planted versions
class Side:
    """A version that fails a trial when its planted rule says so; counts what it was asked to run."""

    def __init__(self, tag, fails, value=None, broken=()):
        self.tag, self.fails, self.value, self.broken, self.asked = tag, fails, value, set(broken), []

    def run(self, trials):
        out = []
        for t in trials:
            self.asked.append(t['i'])
            if t['i'] in self.broken:
                out.append({'ok': False, 'run': None, 'error': 'no run.json'})
                continue
            f = bool(self.fails(t))
            out.append({'ok': True, 'failed': f, 'run': f'runs/{self.tag}/{t["i"]:04d}',
                        'value': float(self.value(t)) if self.value else (1.0 if f else 0.1)})
        return out


SCN = [SC.builtin('lead_brake'), SC.builtin('cut_in')]


def _gate(tmp_path, a, b, **kw):
    return G.Gate(str(tmp_path / 'g'), a, b, SCN, **dict({'budget': 200, 'batch': 8}, **kw))


def test_a_gate_stops_as_soon_as_the_answer_is_clear_and_lists_what_newly_fails(tmp_path):
    a = Side('a', lambda t: False)
    b = Side('b', lambda t: t['values'].get('gap', 99) < 22 or t['i'] % 5 == 0)    # the new version brakes late
    logs = []
    d = _gate(tmp_path, a, b, meta={'versions': {'a': {'label': 'v1'}, 'b': {'label': 'v2'}}}).run(log=logs.append)
    assert d['verdict'] == 'worse' and d['state'] == 'finished' and d['schema'] == G.GATE_SCHEMA
    assert d['trials_run'] < 200 and d['trials_run'] % 8 == 0 and d['trials_used'] == d['decided_at'] <= d['trials_run']
    assert a.asked == b.asked == list(range(d['trials_run']))                # the same trials, in the same order
    nf = d['newly_failing']
    assert len(nf) == d['discordant']['b_only'] == 9 and d['newly_passing'] == []
    for x in nf:                                                             # each links its pair of runs
        assert x['a'] == f'runs/a/{x["i"]:04d}' and x['b'] == f'runs/b/{x["i"]:04d}' and x['value_b'] == 1.0
        assert x['scenario'] == SCN[x['i'] % 2]['id'] and x['seed'] == x['i']
        assert x['values'] == SC.sample_params(SCN[x['i'] % 2], x['seed'])
    assert 'tested scenario distribution only' in d['scope'] and 'lead_brake, cut_in' in d['scope']
    assert d == G.read_gate(str(tmp_path / 'g')) and len(G.read_trials(str(tmp_path / 'g'))) == d['trials_run']
    md = open(tmp_path / 'g' / 'gate.md', encoding='utf-8').read()
    assert '## Release gate: worse' in md and 'Newly failing under B (9)' in md and '`runs/b/' in md
    assert 'B (v2) against A (v1): WORSE after' in G.headline(d) and any('discordant' in t for t in logs)
    # resumed: the verdict stands, nothing runs
    a2, b2 = Side('a', lambda t: False), Side('b', lambda t: True)
    d2 = _gate(tmp_path, a2, b2, meta={'versions': {'a': {'label': 'v1'}, 'b': {'label': 'v2'}}}).run()
    assert a2.asked == [] and b2.asked == [] and d2['verdict'] == 'worse' and d2['trials_run'] == d['trials_run']
    # another design in the same folder is refused, not mixed in
    with pytest.raises(G.GateError, match='another design'):
        _gate(tmp_path, a2, b2, margin=0.1).run()


def test_the_other_verdicts_the_budget_and_resuming_with_more(tmp_path):
    d = _gate(tmp_path / '1', Side('a', lambda t: t['i'] % 3 == 0), Side('b', lambda t: False)).run()
    assert d['verdict'] == 'better' and len(d['newly_passing']) == 9 and d['newly_failing'] == []
    same = lambda t: t['i'] % 7 == 0                                         # noqa: E731 (both fail the same trials)
    d = _gate(tmp_path / '2', Side('a', same), Side('b', same), margin=0.2).run()
    assert d['verdict'] == 'equivalent' and d['word'] == 'no difference within the margin'
    assert d['decided_at'] == 88 and d['trials_run'] == 88 and d['discordant']['n'] == 0
    a, b = Side('a', same), Side('b', same)
    d = _gate(tmp_path / '3', a, b, budget=24).run()
    assert d['verdict'] == 'cant_tell' and d['word'] == 'can\'t tell yet' and d['trials_run'] == 24
    assert d['decided_at'] is None and d['trials_used'] == 24 and 'at the budget of 24' in G.headline(d)
    a, b = Side('a', same), Side('b', same)                                 # more budget: only the new trials run
    d = _gate(tmp_path / '3', a, b, budget=40).run()
    assert a.asked == list(range(24, 40)) and d['trials_run'] == 40 and d['verdict'] == 'cant_tell'
    d = _gate(tmp_path / '4', Side('a', same), Side('b', same), margin=0.05, non_inferiority=0.3).run()
    assert d['verdict'] == 'non_inferior' and d['trials_run'] < 88
    # the value as the primary outcome: no failures at all, B's criticality higher
    d = _gate(tmp_path / '5', Side('a', lambda t: False, lambda t: 0.2), Side('b', lambda t: False, lambda t: 0.45),
              primary='value').run()
    assert d['verdict'] == 'worse' and d['value']['lower'] > 0 and d['failures'] == {'a': 0, 'b': 0}
    # a pair with a side that did not finish is not a trial; it is listed
    d = _gate(tmp_path / '6', Side('a', same, broken={3}), Side('b', same), budget=16).run()
    assert [s['i'] for s in d['skipped']] == [3] and 'A: no run.json' in d['skipped'][0]['why']
    assert d['trials_run'] == 16 and 3 not in [t['i'] for t in G.read_trials(str(tmp_path / '6' / 'g'))]
    with pytest.raises(G.GateError, match='no pair finished'):
        _gate(tmp_path / '7', Side('a', same, broken=set(range(99))), Side('b', same)).run()
    for bad in ({'alpha': 0.0}, {'margin': 0.0}, {'primary': 'x'}):
        with pytest.raises(G.GateError):
            _gate(tmp_path / '8', a, b, **bad)
    with pytest.raises(G.GateError, match='at least one scenario'):
        G.Gate(str(tmp_path / '9'), a, b, [])


def test_versions_and_trials(tmp_path):
    assert G.parse_version('follower') == {'kind': 'stack', 'stack': {'driver': 'follower'}, 'label': 'follower'}
    assert G.parse_version('model:det-v2') == {'kind': 'model', 'model': 'det-v2', 'label': 'model:det-v2'}
    suite = {'schema': 'falcon.suite/1', 'name': 'stack 1.4', 'stack': {'driver': 'ros2', 'launch': [['run', '{port}']],
                                                                       'domain_id': 3}}
    (tmp_path / 's.json').write_text(json.dumps(suite))
    v = G.parse_version('stack:s.json', str(tmp_path))
    assert v['label'] == 'stack 1.4' and v['stack']['launch'] == [['run', '{port}']] and v['suite'].endswith('s.json')
    assert G.parse_version(str(tmp_path / 's.json'))['stack']['domain_id'] == 3
    (tmp_path / 'bad.json').write_text(json.dumps({'stack': {'driver': 'ros2'}}))
    for spec, why in (('', 'a version is'), ('model:', 'id'), ('nope.json', 'no suite file'),
                      ('bad.json', 'launch commands')):
        with pytest.raises(G.GateError, match=why):
            G.parse_version(spec, str(tmp_path))
    # common random numbers: a trial is a function of (scenarios, seed, i) alone
    t = G.trial_plan(SCN, 3, 5)
    assert t['scenario']['id'] == 'cut_in' and t['seed'] == 3 * 1000003 + 5
    assert t['values'] == SC.sample_params(SCN[1], t['seed']) == G.trial_plan(SCN, 3, 5)['values']
    assert G.trial_plan(SCN, 3, 6)['scenario']['id'] == 'lead_brake'
    argv = G.gate_argv('/p.json', 'g1', 'follower', 'autopilot', ['lead_brake', 'cut_in'], budget=40,
                       non_inferiority=0.1, fake_carla='/fake', on='cluster')
    assert argv[:8] == ['gate', 'run', '/p.json', '--name', 'g1', '--a', 'follower', '--b']
    assert argv.count('--scenario') == 2 and argv[argv.index('--non-inferiority') + 1] == '0.1'
    assert argv[argv.index('--on') + 1] == 'cluster' and argv[-2:] == ['--fake-carla', '/fake']


# ============================================================================================ CI and the safety case
def _finished_gate(tmp_path, name, b_fails):
    d = G.Gate(str(tmp_path / name), Side('a', lambda t: False), Side('b', b_fails), SCN, budget=96, margin=0.2).run()
    return str(tmp_path / name), d


def test_ci_mode_fails_on_worse(tmp_path):
    from falcon_autonomy.core import ci as CI
    worse, _ = _finished_gate(tmp_path, 'worse', lambda t: True)
    fine, _ = _finished_gate(tmp_path, 'fine', lambda t: False)
    g = G.ci_gate({'path': worse})
    assert g['status'] == 'failed' and g['measured'] == 'worse' and 'newly failing' in g['detail']
    assert G.ci_gate({'path': fine})['status'] == 'passed'
    assert G.ci_gate({'path': fine, 'fail_on': ['worse', 'equivalent']})['status'] == 'failed'
    assert G.ci_gate({'path': 'nope'}, str(tmp_path))['status'] == 'error'
    assert G.ci_gate({'path': 'worse/gate.json'}, str(tmp_path))['status'] == 'failed'     # relative to the config
    cfg = {'schema': CI.CONFIG_SCHEMA, 'subject': {'kind': 'stack', 'suite': {'x': 1}},
           'gates': {'release': {'path': worse}}}
    assert CI.validate_config(cfg) == []
    assert CI.validate_config(dict(cfg, gates={'release': {'path': worse, 'fail_on': ['bad']}})) != []
    assert CI.validate_config(dict(cfg, gates={'release': {}})) != []
    subj = {'kind': 'stack', 'records': []}
    gates = CI.evaluate_gates(dict(cfg, config_dir=str(tmp_path)), subj, {'overall': {}}, None, None)
    (row,) = [x for x in gates if x['id'] == 'release']
    assert row['status'] == 'failed' and row['measured'] == 'worse' and row['name'] == 'release gate (B against A)'
    gates = CI.evaluate_gates(dict(cfg, gates={'release': {'path': 'fine'}}, config_dir=str(tmp_path)), subj,
                              {'overall': {}}, None, None)
    assert [x['status'] for x in gates if x['id'] == 'release'] == ['passed']


def test_a_gate_verdict_is_evidence_of_the_safety_case(tmp_path):
    from falcon_autonomy.core import gsn as GS
    assert 'gate' in GS.KINDS
    ws = tmp_path / 'ws'
    for name, fails in (('rel-fine', lambda t: False), ('rel-worse', lambda t: True)):
        G.Gate(G.gate_dir(str(ws), name), Side('a', lambda t: False), Side('b', fails), SCN, budget=96,
               margin=0.2).run()
    assert GS.evidence_candidates(str(ws))['gate'] == ['gates/rel-fine', 'gates/rel-worse']
    for name, ok in (('rel-fine', True), ('rel-worse', False)):
        ev = {'kind': 'gate', 'ref': f'gates/{name}'}
        p, _d, _alias = GS.resolve(str(ws), ev)
        assert p.endswith(os.path.join(name, 'gate.json'))
        r = GS.evaluate_claim('gate', {'measure': 'verdict', 'equals': 'equivalent'}, json.load(open(p)))
        assert r['ok'] is ok
        r = GS.evaluate_claim('gate', {'measure': 'field', 'path': 'risk.upper', 'op': '<', 'limit': 0.2},
                              json.load(open(p)))
        assert r['ok'] is ok


# ================================================================================================ on the fake carla
@pytest.fixture(scope='module')
def proj(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    base = tmp_path_factory.mktemp('gate')
    p = Project.new('Gate')
    rig = p.doc['rigs'][0]
    rig['sensors'] = rig['sensors'][:1]
    rig['sensors'][0].update(width=64, height=36, labels=[])
    p.save(str(base / 'p.falcon.json'))
    return p


def _cli(*a):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'gate', *a], capture_output=True, text=True,
                          env=env, timeout=400)


def test_the_command_on_the_fake_carla(proj):
    r = _cli('run', proj.path, '--name', 'x', '--a', 'follower', '--b', 'model:m', '--scenario', 'lead_brake')
    assert r.returncode == 6 and 'both versions must be' in r.stdout
    r = _cli('run', proj.path, '--name', 'x', '--a', 'follower', '--b', 'follower', '--scenario', 'nope')
    assert r.returncode == 2 and 'no scenario' in r.stdout
    r = _cli('show', proj.path, '--name', 'x')
    assert r.returncode == 2 and 'no gate result' in r.stdout
    # the same version twice: every pair is identical (kinematic motion repeats bit for bit), so no verdict in 4
    r = _cli('run', proj.path, '--name', 'same', '--a', 'follower', '--b', 'follower', '--scenario', 'lead_brake',
             '--scenario', 'cut_in', '--budget', '4', '--batch', '4', '--seed', '2', '--fake-carla', FAKE,
             '--gate', '--fail-on', 'worse')
    assert r.returncode == 0, r.stdout + r.stderr
    out = G.gate_dir(proj.workspace, 'same')
    d = G.read_gate(out)
    assert d['verdict'] == 'cant_tell' and d['trials_run'] == 4 and d['discordant']['n'] == 0
    assert d['versions']['a']['label'] == 'follower' and d['failure'] == 'criteria' and d['where'] == 'local'
    trials = G.read_trials(out)
    assert [t['scenario'] for t in trials] == ['lead_brake', 'cut_in'] * 2
    for t in trials:
        assert t['a']['value'] == t['b']['value'] and t['a']['failed'] == t['b']['failed']
        assert t['a']['run'] != t['b']['run'] and os.sep + os.path.join('runs', 'a') + os.sep in t['a']['run']
        ca = json.load(open(os.path.join(t['a']['run'], 'stack.json')))
        cb = json.load(open(os.path.join(t['b']['run'], 'stack.json')))
        assert ca['seed'] == cb['seed'] == t['seed'] and ca['scenario']['values'] == cb['scenario']['values']
        assert ca['scenario']['values'] == pytest.approx(t['values']) and ca['driver'] == 'follower'
    assert os.path.isfile(os.path.join(out, 'gate.md'))
    r = _cli('show', proj.path, '--name', 'same', '--gate', '--fail-on', 'worse,cant_tell')
    assert r.returncode == 1 and 'CAN\'T TELL YET at the budget of 4' in r.stdout
    assert _cli('show', proj.path, '--name', 'same', '--json').returncode == 0
    # two drivers that differ: on the fake, CARLA's autopilot stops well short of a stalled vehicle where the lane
    # follower comes close (criticality ≥ 0.5) — the gate stops early with its verdict, well inside the budget
    r = _cli('run', proj.path, '--name', 'drivers', '--a', 'follower', '--b', 'autopilot', '--scenario',
             'stalled_vehicle', '--failure', 'value', '--threshold', '0.5', '--budget', '40', '--batch', '8',
             '--fake-carla', FAKE, '--gate')
    assert r.returncode == 0, r.stdout + r.stderr
    d = G.read_gate(G.gate_dir(proj.workspace, 'drivers'))
    assert d['verdict'] == 'better' and d['state'] == 'finished' and d['trials_run'] == 16 < d['budget']
    assert d['decided_by'] == 'discordant' and d['discordant']['a_only'] >= 9 and d['discordant']['b_only'] == 0
    assert len(d['newly_passing']) == d['discordant']['a_only'] and d['newly_failing'] == []
    for x in d['newly_passing']:
        assert x['value_a'] >= 0.5 > x['value_b'] and os.path.isfile(os.path.join(x['a'], 'run.json'))
        assert json.load(open(os.path.join(x['b'], 'stack.json')))['driver'] == 'autopilot'
    assert d['value']['estimate'] < -0.3 and 'BETTER after' in r.stdout     # the value's mean agrees (its band is
    assert d['value']['upper'] > 0                                          # still wide after 9 trials, and says so)
    r = _cli('plan', '--p-a', '0.1', '--p-b', '0.5', '--trials', '100', '--reps', '200')
    assert r.returncode == 0 and json.loads(r.stdout)['worse'] > 0.8


def test_two_detectors_score_the_same_recorded_run(tmp_path):
    """Models as versions: each trial's scenario is driven once (with the rig's labels) and both detectors score
    that run — here the synthetic detector and a weaker registration of it."""
    from falcon_autonomy.core.project import Project
    p = Project.new('Gate models')
    rig = p.doc['rigs'][0]
    rig['sensors'] = rig['sensors'][:1]
    rig['sensors'][0].update(width=160, height=90, labels=['boxes2d'])
    syn = {'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python', 'licence': 'GPL-3.0-or-later',
           'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector'}
    p.doc['models'] = [dict(syn, id='det-old', params={'h50': 6.0, 'h_scale': 1.5}),
                       dict(syn, id='det-new', params={'h50': 60.0, 'h_scale': 1.5},      # needs ten times the height
                            options={'version': 2}),
                       dict(syn, id='det-same', params={'h50': 9.0, 'h_scale': 1.5})]
    p.save(str(tmp_path / 'm.falcon.json'))
    # two plug-in registrations that differ only in their params share a fingerprint: refused, not compared blindly
    r = _cli('run', p.path, '--name', 'det', '--a', 'model:det-old', '--b', 'model:det-same', '--scenario',
             'lead_brake', '--fake-carla', FAKE)
    assert r.returncode == 2 and 'same fingerprint' in r.stdout
    r = _cli('run', p.path, '--name', 'det', '--a', 'model:det-old', '--b', 'model:nope', '--scenario', 'lead_brake',
             '--fake-carla', FAKE)
    assert r.returncode == 2 and 'no model' in r.stdout
    r = _cli('run', p.path, '--name', 'det', '--a', 'model:det-old', '--b', 'model:det-new', '--scenario',
             'lead_brake', '--budget', '2', '--batch', '2', '--fps', '10', '--threshold', '0.8', '--fake-carla', FAKE)
    assert r.returncode == 0, r.stdout + r.stderr
    out = G.gate_dir(p.workspace, 'det')
    d = G.read_gate(out)
    assert d['trials_run'] == 2 and d['versions']['b'] == {'kind': 'model', 'label': 'model:det-new',
                                                           'model': 'det-new', 'suite': None}
    assert d['failure'] == 'value' and d['verdict'] == 'cant_tell'
    trials = G.read_trials(out)
    for t in trials:
        assert t['a']['run'] == t['b']['run'] and os.path.isfile(os.path.join(t['a']['run'], 'run.json'))
        assert 0.0 <= t['a']['value'] <= t['b']['value'] <= 1.0             # the weaker detector misses no less
        assert t['a']['failed'] == (t['a']['value'] >= 0.8) and t['b']['failed'] == (t['b']['value'] >= 0.8)
    assert any(t['b']['value'] > t['a']['value'] for t in trials)
    runs = os.path.join(p.workspace, 'studies', 'gate-det-lead_brake', 'runs')
    assert len(os.listdir(runs)) == 2                                        # driven once per trial, scored twice


def test_the_assistant_tools(proj):
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    assert REGISTRY['formal_gate_run'].tier == 'act' and REGISTRY['formal_gate_show'].tier == 'read'
    c = TOOLS.context(proj.path, FAKE)
    try:
        r = call(c, 'formal_gate_show', {})
        assert [(g['name'], g['verdict']) for g in r['gates']] == [('drivers', 'better'), ('same', 'cant_tell')]
        r = call(c, 'formal_gate_show', {'name': 'same'})
        assert r['word'] == 'can\'t tell yet' and r['trials_run'] == 4 and r['versions'] == {'a': 'follower',
                                                                                              'b': 'follower'}
        assert r['risk_difference']['lower'] < 0 < r['risk_difference']['upper'] and 'only' in r['scope']
        assert 'no gate' in call(c, 'formal_gate_show', {'name': 'nope'})['error']
        assert 'by path' in call(c, 'formal_gate_show', {'name': '../x'})['error']
        bad = {'name': 'g', 'a': 'follower', 'b': 'model:m', 'scenarios': ['lead_brake']}
        assert 'both versions must be' in call(c, 'formal_gate_run', bad)['error']
        assert 'no suite file' in call(c, 'formal_gate_run', dict(bad, b='new-stack.json'))['error']
        t = REGISTRY['formal_gate_run']
        args = {'name': 'tool', 'a': 'follower', 'b': 'follower', 'scenarios': ['stalled_vehicle'], 'budget': 2,
                'batch': 2, 'margin': 0.2}
        assert 'gate run' in t.preview(c, **args) and '--margin 0.2' in t.preview(c, **args)
        assert 'at most 2 paired trials (2 run(s) each' in t.estimate(c, **args)['what']
        r = call(c, 'formal_gate_run', args)
        assert r['gate'] == 'tool' and 'gate run' in r['command']
        w = call(c, 'jobs_wait', {'jobs': [r['job']], 'timeout_s': 200})
        assert w['finished'] == [r['job']] and w['jobs'][0]['state'] == 'done', w
        r = call(c, 'formal_gate_show', {'name': 'tool'})
        assert r['state'] == 'finished' and r['trials_run'] == 2 and r['verdict'] == 'cant_tell'
        # a newly failing row names its pair of runs; a run outside the workspace only by its folder name
        d = G.Gate(G.gate_dir(proj.workspace, 'planted'), Side('a', lambda t: False), Side('b', lambda t: True),
                   SCN).run()
        r = call(c, 'formal_gate_show', {'name': 'planted', 'rows': 3})
        assert r['verdict'] == 'worse' and len(r['newly_failing']) == 3 and r['newly_failing_total'] == 9
        assert r['newly_failing'][0]['run_b'] == os.path.basename(d['newly_failing'][0]['b'])
    finally:
        c.service.shutdown(grace_s=10.0)


@pytest.mark.gui
def test_the_release_gate_section_offscreen(qapp, tmp_path):
    from types import SimpleNamespace as NS
    from PyQt6.QtCore import Qt
    from falcon_autonomy.ui.gate_tab import Band, GateTab
    from falcon_autonomy.ui.theme import THEME
    ws = str(tmp_path / 'ws')
    os.makedirs(ws)
    submitted, shown = [], []

    class Q:
        def submit(self, spec):
            submitted.append(spec)
            return NS(state='queued', summary=lambda: 'queued')
    doc = {'models': [{'id': 'det-1', 'kind': 'detector'}, {'id': 'det-2', 'kind': 'detector'}], 'scenarios': [],
           'worlds': [{'id': 'world-town10', 'name': 'Town10'}], 'rigs': [{'id': 'rig-default', 'name': 'Rig'}]}
    data = NS(show_run=lambda p: shown.append(('run', p)), compare_runs=lambda ps: shown.append(('pair', ps)))
    win = NS(project=NS(path=str(tmp_path / 'p.falcon.json'), dirty=False, doc=doc), workspace=lambda: ws,
             queue=Q(), fake_carla=FAKE, save=lambda: True, go=lambda key: data)
    tab = GateTab(NS(win=win))
    assert tab.a.currentText() == 'follower' and tab.b.currentText() == 'autopilot'
    assert [tab.a.itemText(i) for i in range(tab.a.count())] == ['follower', 'autopilot', 'model:det-1',
                                                                 'model:det-2']
    assert tab.selected() == ['lead_brake', 'cut_in'] and tab.start_btn.isEnabled()
    assert tab.start_btn.property('role') == 'primary'
    # what cannot be queued says why
    tab.b.setEditText('model:det-2')
    assert 'Both versions must be' in tab.why.text() and not tab.start_btn.isEnabled()
    tab.a.setEditText('model:det-9')
    assert 'No model' in tab.why.text()
    tab.a.setEditText('stack:missing.json')
    assert 'no suite file' in tab.why.text()
    tab.a.setEditText('follower')
    tab.b.setEditText('autopilot')
    for i in range(tab.scenarios.count()):
        tab.scenarios.item(i).setCheckState(Qt.CheckState.Unchecked)
    assert 'at least one scenario' in tab.why.text()
    tab.start()
    assert submitted == [] and 'Not queued' in tab.state.text()
    tab.scenarios.item(0).setCheckState(Qt.CheckState.Checked)
    tab.name.setText('rel 1')
    tab.check()
    assert 'name' in tab.why.text()
    tab.name.setText('rel-1')
    tab.budget.setValue(96)
    tab.margin.setValue(0.2)
    tab.check()
    assert tab.start_btn.isEnabled()
    tab.start()
    argv = submitted[-1].argv
    assert argv[3:6] == ['gate', 'run', win.project.path] and argv[argv.index('--name') + 1] == 'rel-1'
    assert argv[argv.index('--a') + 1] == 'follower' and argv[argv.index('--b') + 1] == 'autopilot'
    assert argv[argv.index('--scenario') + 1] == 'lead_brake' and argv.count('--scenario') == 1
    assert argv[argv.index('--budget') + 1] == '96' and argv[argv.index('--margin') + 1] == '0.2'
    assert argv[argv.index('--world') + 1] == 'world-town10' and argv[argv.index('--fake-carla') + 1] == FAKE
    assert '--on' not in argv and submitted[-1].resumable and tab.timer.isActive() and tab.sweep.isVisibleTo(tab)
    # a running gate on disk, then its verdict: the word with its colour, the band, the newly failing list
    a = Side('a', lambda t: False)
    b = Side('b', lambda t: t['i'] % 4 == 0)
    g = G.Gate(tab.gate_dir(), a, b, [SC.builtin('lead_brake')], budget=96, margin=0.2,
               meta={'versions': {'a': {'label': 'follower'}, 'b': {'label': 'autopilot'}}})
    part = g.result(G.read_trials(tab.gate_dir()), 'running')
    g.write(part)
    tab.poll()
    assert tab.s_verdict.value.text() == 'can\'t tell yet' and 'still running' in tab.s_verdict.sub.text()
    d = g.run()
    tab.poll()
    assert d['verdict'] == 'worse' and tab.s_verdict.value.text() == 'worse'
    assert THEME.status('critical') in tab.s_verdict.value.styleSheet()
    assert tab.s_trials.value.text() == str(d['trials_used']) and '9 discordant' in tab.s_trials.sub.text()
    assert tab.band.lo == d['path']['risk_lower'] and tab.band.decided == d['decided_at'] and tab.band.margin == 0.2
    assert tab.band.by == 'discordant' and 'θ in [' in tab.s_diff.sub.text()
    assert tab.changes.topLevelItemCount() == 9 and tab.changes.topLevelItem(0).text(0) == 'newly fails'
    assert 'tested scenario distribution only' in tab.scope.text()
    assert tab.sweep.frac == pytest.approx(d['trials_run'] / 96)
    tab.changes.setCurrentItem(tab.changes.topLevelItem(0))
    tab.browse('b')
    tab.compare()
    x = d['newly_failing'][0]
    assert shown == [('run', x['b']), ('pair', [x['a'], x['b']])]
    tab.resize(760, 1200)
    tab.grab()                                                   # the band paints: bounds, zero, margin, the mark
    Band().grab()                                                # and its empty state
    tab.job.state = 'done'
    tab.poll()
    assert not tab.timer.isActive() and not tab.sweep.isVisibleTo(tab)
    # an equivalent gate reads in the good tone; the value as the primary outcome draws its band
    G.Gate(G.gate_dir(ws, 'same'), Side('a', lambda t: False, lambda t: 0.3), Side('b', lambda t: False,
                                                                                  lambda t: 0.3),
           [SC.builtin('lead_brake')], budget=96, margin=0.2, primary='value').run()
    tab.name.setText('same')
    tab.load()
    assert tab.s_verdict.value.text() == 'no difference within the margin' and tab.changes.topLevelItemCount() == 0
    assert THEME.status('good') in tab.s_verdict.value.styleSheet() and tab.band.what == 'value'
    assert 'inside the margin' in tab.s_diff.sub.text()
    tab.grab()
    tab.shutdown()
    import inspect
    from falcon_autonomy.ui.pages import reports
    assert "optional_tab(v, 'ui.gate_tab', 'GateTab', 'Release gate', self)" in inspect.getsource(reports)
