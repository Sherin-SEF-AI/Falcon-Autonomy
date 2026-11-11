"""Regression bisect (core.regbisect): Fisher's exact test and the smallest p a run count allows, the endpoint rule in
both modes, the pure bisection with a fake evaluator (several scenarios at once, skipped commits like git bisect
skip, an ambiguous answer, a step limit, a non-monotonic history noted), a noisy stack's bisect over a real git
history with a fake evaluator, and end to end: a throw-away git repository whose stack stops braking at one commit,
with a commit that does not build, bisected through CI mode against the fake carla."""
import json
import os
import subprocess
import sys

import pytest

from conftest import FAKE, ROOT
from falcon_autonomy.core import regbisect as B

STAND_IN = os.path.join(ROOT, 'tests', 'data', 'ci_stack.py')


# --------------------------------------------------------------------------------------------- statistics
def test_fisher_and_the_smallest_p():
    assert B.fisher_less(3, 3, 0, 3) == pytest.approx(0.05)            # 1 / C(6, 3)
    assert B.min_p(4, 4) == pytest.approx(1 / 70)
    assert B.min_p(1, 1) == pytest.approx(0.5) and B.min_p(2, 2) == pytest.approx(1 / 6)
    assert B.fisher_less(5, 5, 5, 5) == 1.0
    # 9/10 → 3/10: P(X ≤ 3) with 12 passes in 20, 10 drawn
    from math import comb
    want = sum(comb(12, x) * comb(8, 10 - x) for x in range(2, 4)) / comb(20, 10)
    assert B.fisher_less(9, 10, 3, 10) == pytest.approx(want)


def _recs(spec):
    """{scenario: [(pair, passed), …]} → CI run records."""
    return [{'group': s, 'pair': p, 'name': f'{s} {p}', 'status': 'ok', 'passed': ok}
            for s, rows in spec.items() for p, ok in rows]


def test_endpoint_rules_deterministic_and_noisy():
    good = B.tally(_recs({'right_turn': [('s0', True), ('s1', True)], 'lead_brake': [('s0', True), ('s1', False)],
                          'only_good': [('s0', True)]}))
    bad = B.tally(_recs({'right_turn': [('s0', True), ('s1', False)], 'lead_brake': [('s0', True), ('s1', True)]}))
    rows = {r['scenario']: r for r in B.find_regressions(good, bad, deterministic=True)}
    assert rows['right_turn']['regressed'] and rows['right_turn']['flipped'] == ['s1']
    assert not rows['lead_brake']['regressed'] and rows['lead_brake']['improved']
    assert rows['only_good']['why'] == 'not run at both ends'
    # noisy: 8 of 8 → 2 of 8 is shown at 0.05; 2 of 2 → 0 of 2 cannot be (the smallest p is 1/6)
    good = B.tally(_recs({'a': [(f'r{i}', True) for i in range(8)], 'b': [('r0', True), ('r1', True)]}))
    bad = B.tally(_recs({'a': [(f'r{i}', i < 2) for i in range(8)], 'b': [('r0', False), ('r1', False)]}))
    rows = {r['scenario']: r for r in B.find_regressions(good, bad, deterministic=False)}
    assert rows['a']['regressed'] and rows['a']['p'] < 0.05 and rows['a']['test'] == 'Fisher exact, one-sided'
    assert not rows['b']['regressed'] and 'cannot show a change' in rows['b']['why']
    # a commit's verdict: the regressed pairs (deterministic), the midpoint of the two ends' rates (noisy)
    reg = {'flipped': ['s1'], 'good': {'k': 8, 'n': 8}, 'bad': {'k': 2, 'n': 8}}
    assert B.classify(B.tally(_recs({'x': [('s1', False)]}))['x'], reg, True) == B.BAD
    assert B.classify(B.tally(_recs({'x': [('s0', False), ('s1', True)]}))['x'], reg, True) == B.GOOD
    assert B.classify(B.tally(_recs({'x': [('s9', False)]}))['x'], reg, True) == B.SKIP
    assert B.classify(None, reg, True) == B.SKIP
    mk = lambda k, n: B.tally(_recs({'x': [(f'r{i}', i < k) for i in range(n)]}))['x']   # noqa: E731
    assert B.classify(mk(7, 8), reg, False) == B.GOOD and B.classify(mk(3, 8), reg, False) == B.BAD
    assert B.classify(mk(5, 8), reg, False) == B.SKIP                 # 5/8 is the midpoint of 8/8 and 2/8


# --------------------------------------------------------------------------------------------- the pure bisection
def fake(first_bad: dict, skip=(), flaky=None):
    """evaluate(i): bad for a scenario from its first bad index on; skip: indices that do not build."""
    calls = []

    def ev(i):
        calls.append(i)
        if i in skip:
            return {s: B.SKIP for s in first_bad}
        out = {s: (B.BAD if i >= k else B.GOOD) for s, k in first_bad.items()}
        if flaky and i in flaky:
            out.update(flaky[i])
        return out
    return ev, calls


@pytest.mark.parametrize('n,k', [(1, 0), (2, 0), (2, 1), (7, 3), (64, 0), (64, 63), (100, 37), (257, 200)])
def test_bisect_finds_the_first_bad_commit_in_log2_steps(n, k):
    import math
    ev, calls = fake({'s': k})
    r = B.run_bisect(n, ['s'], ev)
    assert r['result']['s']['status'] == 'found' and r['result']['s']['first_bad'] == k
    assert len(calls) <= math.ceil(math.log2(max(n, 2))) and len(set(calls)) == len(calls)


def test_several_scenarios_share_the_commits_tested():
    ev, calls = fake({'right_turn_night': 11, 'lead_brake': 40, 'jaywalker': 39})
    r = B.run_bisect(64, ['right_turn_night', 'lead_brake', 'jaywalker'], ev)
    assert {s: x['first_bad'] for s, x in r['result'].items()} == {'right_turn_night': 11, 'lead_brake': 40,
                                                                 'jaywalker': 39}
    assert len(calls) < 3 * 6                                          # fewer than three separate bisects


def test_skipped_commits_like_git_bisect_skip():
    ev, calls = fake({'s': 5}, skip={2, 3})
    r = B.run_bisect(10, ['s'], ev)['result']['s']
    assert r['status'] == 'found' and r['first_bad'] == 5              # the skips were not next to the change
    ev, calls = fake({'s': 5}, skip={4})
    r = B.run_bisect(10, ['s'], ev)['result']['s']
    assert r['status'] == 'ambiguous' and r['candidates'] == [4, 5] and r['first_bad'] is None
    assert 'skipped' in r['why']
    ev, calls = fake({'s': 0}, skip=set(range(9)))
    r = B.run_bisect(10, ['s'], ev)['result']['s']
    assert r['status'] == 'ambiguous' and r['candidates'] == list(range(10))


def test_step_limit_and_a_non_monotonic_history():
    ev, calls = fake({'s': 50})
    r = B.run_bisect(100, ['s'], ev, max_steps=2)['result']['s']
    assert r['status'] == 'stopped' and len(calls) == 2 and 50 in r['candidates']
    # a scenario good again after its first bad commit (a fix, then a break): noted
    b = B.Bisector(8, ['a', 'b'])
    b.record(3, {'a': B.BAD, 'b': B.GOOD})
    b.record(5, {'a': B.GOOD, 'b': B.BAD})                      # a is good again after its first bad commit
    b.record(1, {'a': B.GOOD, 'b': B.BAD})                      # b is bad before its last good one
    assert any(n.startswith('a: commit 5 is good after the bad commit 3') for n in b.notes)
    assert any(n.startswith('b: commit 1 is bad before the good commit 3') for n in b.notes)
    with pytest.raises(B.BisectError):
        B.Bisector(0, ['s'])


# --------------------------------------------------------------------------------------------- git
def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=t@example.org',
                           '-c', 'commit.gpgsign=false', *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(repo, msg, files):
    for name, text in files.items():
        p = repo / name
        if text is None:
            p.unlink()
        else:
            p.write_text(text)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', msg)
    return git(repo, 'rev-parse', 'HEAD')


def new_repo(tmp_path):
    repo = tmp_path / 'stack'
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'main')
    return repo


class NoisyEvaluator:
    """A stand-in for CIEvaluator: a non-deterministic stack whose right turn passes 9 of 10 times before the
    breaking commit and 2 of 10 after it; lead_brake is unaffected."""

    def __init__(self, shas, broken_from):
        self.shas, self.broken_from, self.repeats, self.seen = shas, broken_from, 1, []

    def records(self, sha):
        self.seen.append(sha)
        i = self.shas.index(sha)
        k = 2 if i >= self.broken_from else 9
        recs = []
        for rep in range(max(self.repeats, 1)):
            for s in range(2):
                n = rep * 2 + s
                recs.append({'group': 'right_turn_night', 'pair': f's{s}', 'name': f'rt {s}', 'status': 'ok',
                             'passed': (n % 10) < k if self.repeats > 1 else True})
                recs.append({'group': 'lead_brake', 'pair': f's{s}', 'name': f'lb {s}', 'status': 'ok',
                             'passed': True})
        return {'status': 'ok', 'records': recs, 'deterministic': False}


def test_a_noisy_regression_bisected_over_a_real_history(tmp_path):
    repo = new_repo(tmp_path)
    shas = [commit(repo, f'change {i}', {'f.txt': str(i), 'falcon-ci.json': '{}'}) for i in range(12)]
    ev = NoisyEvaluator(shas, broken_from=7)
    doc = B.bisect_stack(str(repo), 'falcon-ci.json', shas[0], 'main', str(tmp_path / 'out'), repeats=5,
                         evaluator=ev)
    rows = {r['scenario']: r for r in doc['endpoints']}
    assert doc['deterministic'] is False and doc['repeats'] == 5 and 'Fisher' in doc['rule']
    assert rows['right_turn_night']['regressed'] and rows['right_turn_night']['good'] == {'k': 9, 'n': 10}
    assert rows['right_turn_night']['bad'] == {'k': 2, 'n': 10} and not rows['lead_brake']['regressed']
    assert doc['regressed'] == ['right_turn_night']
    r = doc['bisect']['right_turn_night']
    assert r['status'] == 'found' and r['first_bad']['sha'] == shas[7] and r['first_bad']['subject'] == 'change 7'
    assert r['last_good']['sha'] == shas[6] and r['first_bad']['author'] == 'Test'
    assert len(doc['commits']) == 11 and doc['bad']['sha'] == shas[11] and doc['good']['sha'] == shas[0]
    assert os.path.isfile(tmp_path / 'out' / 'bisect.json') and 'first bad commit `' in \
        (tmp_path / 'out' / 'summary.md').read_text()
    with pytest.raises(B.BisectError, match='not an ancestor'):
        B.bisect_stack(str(repo), 'falcon-ci.json', shas[5], shas[2], str(tmp_path / 'o2'), evaluator=ev)
    with pytest.raises(B.BisectError, match='same commit'):
        B.bisect_stack(str(repo), 'falcon-ci.json', shas[5], shas[5], str(tmp_path / 'o3'), evaluator=ev)
    with pytest.raises(B.BisectError, match='inside the repository'):
        B.bisect_stack(str(repo), '/etc/x.json', shas[0], shas[5], str(tmp_path / 'o4'), evaluator=ev)


# --------------------------------------------------------------------------------------------- end to end
LAUNCH = '''import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
speed = json.load(open(os.path.join(here, "stack.json")))["speed"]
os.execv(sys.executable, [sys.executable, {stand_in!r}, "--connect", sys.argv[1], "--speed", str(speed)])
'''


def test_bisect_through_ci_mode_end_to_end(tmp_path, monkeypatch):
    """A stack that stops for a stalled car until the commit "raise the cruise speed" (index 3 after the good ref),
    with a commit that does not build (index 1, skipped): the stalled_vehicle scenario regressed, and its first bad
    commit is the speed change."""
    from falcon_autonomy.cli import main
    from falcon_autonomy.core.project import Project
    monkeypatch.setenv('FALCON_CI_NO_POST', '1')
    for k in ('GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_URL', 'BUILD_URL', 'GITHUB_STEP_SUMMARY'):
        monkeypatch.delenv(k, raising=False)
    p = Project.new('Bisect')
    p.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=[])
    proj = str(tmp_path / 'p.falcon.json')
    p.save(proj)
    repo = new_repo(tmp_path)
    suite = {'schema': 'falcon.suite/1', 'name': 'aeb', 'stack': {'driver': 'ros2', 'bridge': {}, 'launch': [],
                                                                  'domain_id': 0},
             'world': 'world-town10', 'rig': 'rig-default', 'fps': 20.0, 'motion': 'kinematic',
             'traffic_vehicles': 0, 'criteria': {'no_collision': True, 'min_ttc_s': 1.0, 'max_missed_deadlines': 0},
             'entries': [{'scenario': 'stalled_vehicle', 'seeds': [0],
                          'values': {'speed': 30.0, 'distance': 60.0, 'kind': 'car'}}]}
    cfg = {'schema': 'falcon.ci/1', 'name': 'aeb', 'project': proj,
           'subject': {'kind': 'stack', 'suite': 'suite.json', 'stack': {
               'bridge_node': 'none', 'bridge': {'control': 'autoware', 'deadline_s': 5.0},
               'command': [[sys.executable, '{repo}/launch.py', '127.0.0.1:{port}']]}},
           'carla': {'mode': 'fake', 'fake_carla': FAKE}, 'gates': {'pass_rate': {'min': 0.5, 'rule': 'estimate'}}}
    good = commit(repo, 'the stack stops', {'launch.py': LAUNCH.format(stand_in=STAND_IN),
                                            'stack.json': json.dumps({'speed': 0}),
                                            'suite.json': json.dumps(suite), 'falcon-ci.json': json.dumps(cfg),
                                            'README': 'a'})
    shas = [commit(repo, 'docs', {'README': 'b'}),
            commit(repo, 'half-done refactor', {'BROKEN': 'x'}),
            commit(repo, 'refactor finished', {'BROKEN': None}),
            commit(repo, 'raise the cruise speed', {'stack.json': json.dumps({'speed': 14})}),
            commit(repo, 'docs again', {'README': 'c'})]
    out = tmp_path / 'bisect'
    code = main(['bisect', 'run', str(repo), '--config', 'falcon-ci.json', '--good', good, '--bad', 'main',
                 '--out', str(out), '--build', 'test ! -f BROKEN', '--fake-carla', FAKE])
    assert code == 0
    doc = json.loads((out / 'bisect.json').read_text())
    assert doc['deterministic'] is True and doc['repeats'] == 1, doc['notes']
    row = doc['endpoints'][0]
    assert row['scenario'] == 'stalled_vehicle' and row['regressed'] and row['good'] == {'k': 1, 'n': 1}
    r = doc['bisect']['stalled_vehicle']
    assert r['status'] == 'found' and r['first_bad']['sha'] == shas[3], json.dumps(doc, indent=1)[:3000]
    assert r['first_bad']['subject'] == 'raise the cruise speed'
    assert [s['sha'] for s in doc['skipped']] == [shas[1]] and 'build failed' in doc['skipped'][0]['why']
    tested = [s['sha'] for s in doc['steps']]
    assert shas[1] in tested and len(tested) <= 4
    # the checkouts are gone, the repository is as it was, and a second run reads every commit back
    assert not os.listdir(out / 'checkouts') and git(repo, 'status', '--porcelain') == ''
    assert git(repo, 'worktree', 'list').count('\n') == 0
    mt = os.path.getmtime(out / 'commits' / shas[3][:12] / 'records.json')
    assert main(['bisect', 'run', str(repo), '--config', 'falcon-ci.json', '--good', good, '--bad', 'main',
                 '--out', str(out), '--build', 'test ! -f BROKEN', '--fake-carla', FAKE]) == 0
    assert os.path.getmtime(out / 'commits' / shas[3][:12] / 'records.json') == mt
    assert main(['bisect', 'show', str(out)]) == 0
