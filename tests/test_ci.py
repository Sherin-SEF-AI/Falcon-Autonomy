"""CI mode (core.ci, core.ci_github) against the fake carla: the config and its schema, the commit under test, the
interval rules, a follower suite end to end (every output, a failing gate's exit code, a deterministic re-run against
it as the baseline), a ROS 2 stack started from a command (tests/data/ci_stack.py plays the bridge node) with a
regression and the latency gate, the flaky-run policy, a policy and a model as subjects with the results store, a
cancel by SIGTERM, the workflow template and its checks, and posting — through a recording fake transport only:
FALCON_CI_NO_POST is set for every test and nothing here ever reaches GitHub."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from conftest import ROOT
from falcon_autonomy.core import ci as CI
from falcon_autonomy.core import ci_github as G
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (fixtures)

STAND_IN = os.path.join(ROOT, 'tests', 'data', 'ci_stack.py')
GH_ENV = ('GITHUB_ACTIONS', 'GITHUB_STEP_SUMMARY', 'GITHUB_REF', 'GITHUB_SHA', 'GITHUB_REPOSITORY', 'GITLAB_CI',
          'JENKINS_URL', 'BUILD_URL', 'GITHUB_EVENT_PATH', 'GITHUB_TOKEN', 'GH_TOKEN')


@pytest.fixture(autouse=True)
def no_posting(monkeypatch):
    """No test may post anything, and none sees a real CI system's variables."""
    for k in GH_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('FALCON_CI_NO_POST', '1')


class Oracle:
    """The ground truth as a detector (score 1): the baseline model of the model tests."""
    synthetic = True

    def predict(self, rgb, meta):
        return [{'bbox': o['bbox'], 'class': o['class'], 'score': 1.0} for o in meta['objects'] if o.get('bbox')]


def small_project(tmp_path, name='p.falcon.json', **cam):
    from falcon_autonomy.core.project import Project
    p = Project.new('CI test')
    p.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=[])
    p.doc['rigs'][0]['sensors'][0].update(cam)
    path = str(tmp_path / name)
    p.save(path)
    return p, path


def write(path, doc):
    path.write_text(json.dumps(doc, indent=1))
    return str(path)


def suite(name, driver='follower', entries=None, **kw):
    doc = {'schema': 'falcon.suite/1', 'name': name, 'stack': {'driver': driver, 'bridge': {}, 'launch': [],
                                                                 'domain_id': 0},
           'world': 'world-town10', 'rig': 'rig-default', 'fps': 20.0, 'motion': 'kinematic', 'traffic_vehicles': 0,
           'criteria': {'no_collision': True, 'min_ttc_s': 1.0, 'max_missed_deadlines': 0},
           'entries': entries or [{'scenario': 'lead_brake', 'seeds': [0, 1]}]}
    doc.update(kw)
    return doc


def stand_in(speed, delay_ms=0.0, **extra):
    argv = [sys.executable, STAND_IN, '--connect', '127.0.0.1:{port}', '--speed', str(speed), '--delay-ms',
            str(delay_ms)]
    for k, v in extra.items():
        argv += [f'--{k.replace("_", "-")}', str(v)]
    return argv


def run(cfg, env=None, **o):
    return CI.run_ci(str(cfg), o, log=lambda t: None, env=env or {})


STALLED = [{'scenario': 'stalled_vehicle', 'seeds': [0, 1, 2], 'values': {'speed': 30.0, 'distance': 60.0,
                                                                          'kind': 'car'}}]


# --------------------------------------------------------------------------------------------- config and rules
def test_config_schema_and_semantics(tmp_path):
    ok = {'schema': 'falcon.ci/1', 'subject': {'kind': 'stack', 'suite': suite('s')}}
    assert CI.validate_config(ok) == []
    errs = CI.validate_config({'schema': 'falcon.ci/1', 'subject': {'kind': 'robot'}, 'bogus': 1})
    assert any('bogus' in e for e in errs) and any('robot' in e for e in errs)
    errs = CI.validate_config({'schema': 'falcon.ci/1', 'subject': {'kind': 'model', 'model': 'm', 'drives': ['x']},
                               'gates': {'pass_rate': {'min': 0.5}, 'latency': {'p95_ms': 10}}})
    assert any('project' in e for e in errs) and any('pass_rate' in e for e in errs) and any('p95_ms' in e
                                                                                         for e in errs)
    errs = CI.validate_config(dict(ok, gates={'metrics': [{'metric': 'pass_rate', 'min': 0.1, 'max': 0.9}],
                                              'regressions': {'metrics': ['nope']}}))
    assert any('min or max' in e for e in errs) and any('nope' in e for e in errs)
    errs = CI.validate_config({'schema': 'falcon.ci/1', 'subject': {'kind': 'stack', 'suite': 'x.json', 'stack': {
        'driver': 'follower', 'image': 'me/stack:1'}}, 'flaky': {'retries': 9}})
    assert any('retries' in e for e in errs)
    # the resolved config: paths from the config's folder, an unset variable means "not configured"
    c = CI.resolve({'schema': 'falcon.ci/1', 'subject': {'kind': 'stack', 'suite': 's.json'},
                    'baseline': {'kind': 'store', 'store': '${FALCON_NOT_SET_ANYWHERE}', 'branch': 'main'}},
                   str(tmp_path), {'fake_carla': str(tmp_path)})
    assert c['subject']['suite'] == str(tmp_path / 's.json') and c['baseline']['store'] == ''
    assert c['carla'] == {'mode': 'fake', 'keep_server': False, 'fake_carla': str(tmp_path)}
    assert c['out'] == str(tmp_path / 'falcon-ci-out') and c['flaky'] == {'retries': 0, 'count_as': 'pass'}
    assert any('no file' in e for e in CI.validate_config(c))
    from falcon_autonomy.cli import main
    assert main(['ci', 'schema']) == 0
    p = tmp_path / 'bad.json'
    p.write_text('{"schema": "falcon.ci/1"}')
    assert main(['ci', 'validate', str(p)]) == 2


def test_interval_rules_and_runs_needed():
    assert CI.judge(0.9, 0.6, 1.0, 0.8, 'min', 'demonstrate') == (False, 0.6)
    assert CI.judge(0.9, 0.6, 1.0, 0.8, 'min', 'refute') == (True, 1.0)
    assert CI.judge(0.9, 0.6, 1.0, 0.8, 'min', 'estimate') == (True, 0.9)
    assert CI.judge(3.0, 1.0, 6.0, 5.0, 'max', 'demonstrate') == (False, 6.0)
    assert CI.judge(3.0, 1.0, 6.0, 5.0, 'max', 'refute') == (True, 1.0)
    assert CI.judge(3.0, None, None, 5.0, 'max', 'demonstrate') == (None, None)
    from falcon_autonomy.core.stack import wilson
    n = CI.runs_to_demonstrate(0.9)
    assert n == 35 and wilson(35, 35)[0] >= 0.9 > wilson(34, 34)[0]
    assert CI.runs_to_demonstrate(0.5) == 4 and wilson(6, 6)[0] == pytest.approx(0.61, abs=0.01)


def test_commit_info(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    g = ['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@example.invalid']
    subprocess.run(g + ['init', '-q', '-b', 'feature'], check=True)
    (repo / 'a.txt').write_text('a')
    subprocess.run(g + ['add', 'a.txt'], check=True)
    subprocess.run(g + ['commit', '-q', '-m', 'one'], check=True)
    subprocess.run(g + ['remote', 'add', 'origin', 'https://user:s3cret@example.invalid/o/r.git'], check=True)
    c = CI.commit_info(str(repo), {})
    sha = subprocess.run(g + ['rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    assert c['sha'] == sha and c['branch'] == 'feature' and c['dirty'] is False and c['source'].startswith('git')
    assert c['remote'] == 'https://example.invalid/o/r.git'                          # the credential is gone
    (repo / 'a.txt').write_text('b')
    assert CI.commit_info(str(repo), {})['dirty'] is True
    env = {'GITHUB_ACTIONS': 'true', 'GITHUB_SHA': 'f' * 40, 'GITHUB_REF': 'refs/pull/17/merge',
           'GITHUB_HEAD_REF': 'fix-brakes', 'GITHUB_REPOSITORY': 'o/r', 'GITHUB_RUN_ID': '99'}
    c = CI.commit_info(str(tmp_path / 'not-a-repo'), env)
    assert (c['sha'], c['ci'], c['pr'], c['branch']) == ('f' * 40, 'github', 17, 'fix-brakes')
    assert c['run_url'] == 'https://github.com/o/r/actions/runs/99' and c['source'] == 'github environment'
    c = CI.commit_info('', {'GITLAB_CI': 'true', 'CI_COMMIT_SHA': 'a' * 40, 'CI_COMMIT_REF_NAME': 'main'})
    assert (c['ci'], c['branch'], c['sha']) == ('gitlab', 'main', 'a' * 40)
    assert CI.commit_info('', {})['sha'] is None


def test_paired_comparison_of_runs():
    def rec(i, passed, ttc):
        return {'pair': f'p{i}', 'name': f'r{i}', 'status': 'ok', 'passed': passed,
                'metrics': {'passed': passed, 'collision': not passed, 'criticality': 0.5, 'min_ttc_s': ttc,
                            'missed_deadlines': 0, 'rule_violations': None, 'distance_km': None}}
    base = [rec(i, True, 3.0) for i in range(6)]
    same = CI.compare_records(base, [dict(r) for r in base], CI.STACK_METRICS, list(CI.STACK_METRICS))
    assert same['pairs'] == 6 and same['regressions'] == [] and same['note'] is None
    row = next(r for r in same['rows'] if r['metric'] == 'pass_rate')
    assert row['diff'] == 0 and row['ci_low'] == row['ci_high'] == 0 and row['verdict'].startswith('no detectable')
    worse = [rec(i, i == 0, 1.0) for i in range(6)]
    cmp_ = CI.compare_records(base, worse, CI.STACK_METRICS, list(CI.STACK_METRICS))
    assert set(cmp_['regressions']) == {'pass_rate', 'collision_rate', 'min_ttc_s'}
    assert len(cmp_['new_failures']) == 5
    rv = next(r for r in cmp_['rows'] if r['metric'] == 'rule_violations_per_km')
    assert rv['verdict'] == 'not measured'
    few = CI.compare_records(base[:2], worse[:2], CI.STACK_METRICS, ['pass_rate'])
    assert 'not reliable' in few['note'] and few['only_baseline'] == [] and few['pairs'] == 2


def test_gates_that_cannot_be_judged():
    recs = [{'status': 'ok', 'passed': True, 'group': 'x', 'metrics': {'passed': True}}]
    s = {'overall': CI.summarise_records(recs, CI.STACK_METRICS), 'groups': {}}
    subj = {'kind': 'stack', 'records': recs, 'latency': None, 'rules': None}
    cfg = {'gates': {'latency': {'p95_ms': 10}, 'regressions': {'on_missing_baseline': 'error'},
                     'metrics': [{'metric': 'AP', 'min': 0.1}, {'metric': 'criticality', 'max': 1.0,
                                                                'rule': 'demonstrate'}],
                     'rule_violations': {'max_per_km': 1.0}}}
    g = {x['id']: x for x in CI.evaluate_gates(cfg, subj, s, None, 'no baseline configured')}
    assert g['latency']['status'] == 'error' and 'ROS 2' in g['latency']['detail']
    assert g['regressions']['status'] == 'error' and g['regressions']['detail'] == 'no baseline configured'
    assert g['metric.AP']['status'] == 'error' and g['metric.criticality']['status'] == 'error'
    assert g['rule_violations']['status'] == 'error'
    cfg['gates'] = {'regressions': {}}
    assert CI.evaluate_gates(cfg, subj, s, None, 'none')[0]['status'] == 'skipped'


def test_output_folder_is_never_reused_silently(tmp_path):
    d = tmp_path / 'out'
    d.mkdir()
    (d / 'mine.txt').write_text('x')
    with pytest.raises(CI.CIError, match='not empty'):
        CI.prepare_out(str(d), force=True)                       # not a CI output: never cleared
    shutil.rmtree(d)
    d.mkdir()
    (d / 'ci-results.json').write_text('{}')
    (d / 'suite').mkdir()
    with pytest.raises(CI.CIError, match='--force'):
        CI.prepare_out(str(d))
    CI.prepare_out(str(d), force=True)
    assert os.listdir(d) == []


# --------------------------------------------------------------------------------------------- stacks end to end
def test_follower_suite_end_to_end(tmp_path, fake_carla):
    """A follower suite: every output; a pass-rate gate that cannot be demonstrated with 4 runs fails (exit 1) and
    says how many runs it needs; the same config re-run against that output as its baseline finds no change (the
    runs repeat bit for bit), through the command line."""
    _, proj = small_project(tmp_path)
    sp = write(tmp_path / 'suite.json', suite('baseline', entries=[{'scenario': 'lead_brake', 'seeds': [0, 1]},
                                                                    {'scenario': 'stalled_vehicle', 'seeds': [0, 1]}]))
    cfg = write(tmp_path / 'falcon-ci.json', {
        'schema': 'falcon.ci/1', 'name': 'follower-smoke', 'project': 'p.falcon.json',
        'subject': {'kind': 'stack', 'suite': os.path.basename(sp)}, 'carla': {'mode': 'fake'},
        'gates': {'pass_rate': {'min': 0.9, 'rule': 'demonstrate', 'per_scenario': True},
                  'metrics': [{'metric': 'collision_rate', 'max': 0.5, 'rule': 'estimate'}],
                  'regressions': {}, 'rule_violations': {'max_per_km': 1000.0}, 'max_missed_deadlines': 0}})
    summary_file = tmp_path / 'step-summary.md'
    env = {'GITHUB_ACTIONS': 'true', 'GITHUB_STEP_SUMMARY': str(summary_file), 'GITHUB_SHA': 'c' * 40,
           'GITHUB_REF_NAME': 'main', 'GITHUB_REPOSITORY': 'o/r', 'GITHUB_RUN_ID': '7'}
    res, code = run(cfg, env, fake_carla=fake_carla)
    out = tmp_path / 'falcon-ci-out'
    assert res['error'] is None, res.get('traceback') or res['error']
    assert len(res['runs']) == 4 and all(r['status'] == 'ok' for r in res['runs'])
    gates = {g['id']: g for g in res['gates']}
    passed = res['summary']['overall']['passed']
    if passed == 4:                                       # 4 of 4: Wilson's lower bound 0.51 cannot reach 0.9
        assert gates['pass_rate.overall']['status'] == 'failed' and '35 runs' in gates['pass_rate.overall']['detail']
    assert code == 1 and res['verdict'] == 'failed'
    assert gates['regressions']['status'] == 'skipped' and gates['missed_deadlines']['status'] == 'passed'
    assert {'pass_rate.lead_brake', 'pass_rate.stalled_vehicle'} <= set(gates)
    assert res['deterministic'] is True and res['commit']['sha'] == 'c' * 40
    for f in CI.OUT_FILES:
        assert (out / f).is_file(), f
    # JUnit: a testcase per run and per gate, counts consistent
    root = ET.parse(out / 'junit.xml').getroot()
    cases = root.findall('.//testcase')
    assert len(cases) == 4 + len(res['gates']) and int(root.get('tests')) == len(cases)
    assert int(root.get('failures')) == sum(1 for c in cases if c.find('failure') is not None)
    run_cases = root.find('testsuite').findall('testcase')
    assert {c.get('name') for c in run_cases} == {'lead_brake seed 0', 'lead_brake seed 1', 'stalled_vehicle seed 0',
                                                   'stalled_vehicle seed 1'}
    assert root.find('.//property[@name="commit"]').get('value') == 'c' * 40
    md = (out / 'summary.md').read_text()
    assert md.startswith('## Falcon CI: follower-smoke — FAILED') and '| pass rate (all runs) ≥ 0.9 |' in md
    assert 'fake CARLA' in md and summary_file.read_text().strip() == md.strip()      # $GITHUB_STEP_SUMMARY
    from falcon_autonomy.core.reports import read_meta
    meta = read_meta(str(out / 'report.html'))
    assert meta['kind'] == 'ci' and str(out / 'ci-results.json') in meta['inputs']
    html_ = (out / 'report.html').read_text()
    assert '<svg' in html_ and 'Wilson' in html_
    lines = (out / 'ci-runs.jsonl').read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])['schema'] == CI.RUN_SCHEMA
    runj = json.loads((out / 'run.json').read_text())
    assert runj['worker'] == 'ci' and runj['exit_code'] == 1 and runj['provenance']['inputs']['config']['sha256']
    assert res['provenance']['inputs']['suite']['sha256']
    # the same suite again, against the first run: a fresh folder, nothing resumed, no change
    from falcon_autonomy.cli import main
    out2 = tmp_path / 'again'
    args = ['ci', '--fake-carla', fake_carla, 'run', cfg, '--out', str(out2), '--baseline', str(out), '--quiet']
    assert main(args) == 1                              # the pass-rate gate still cannot be demonstrated
    res2 = json.loads((out2 / 'ci-results.json').read_text())
    assert res2['comparison']['pairs'] == 4 and res2['comparison']['regressions'] == []
    assert {r['verdict'] for r in res2['comparison']['rows']} <= {'no detectable change at 95 %', 'not measured'}
    def same(r):
        return {k: v for k, v in r['metrics'].items() if k != 'steps_per_s'}       # all but the wall-clock speed
    assert [same(r) for r in res2['runs']] == [same(r) for r in res['runs']]       # bit for bit
    g2 = {g['id']: g for g in res2['gates']}
    assert g2['regressions']['status'] == 'passed'
    assert main(args) == 2                              # the folder holds a run: refused without --force
    assert main(args + ['--force']) == 1


def test_ros2_command_stack_regression_and_latency(tmp_path, fake_carla):
    """A ROS 2 stack started from a command (the stand-in plays the bridge node): a stopping stack, then a commit
    whose stack does not stop — a regression of the pass and collision rates against the first run; the control
    wait's latency gate from every step's wait."""
    _, proj = small_project(tmp_path)
    sp = write(tmp_path / 'suite.json', suite('aeb', 'ros2', STALLED))

    def cfg(speed, name, gates):
        return write(tmp_path / f'{name}.json', {
            'schema': 'falcon.ci/1', 'name': name, 'project': 'p.falcon.json',
            'subject': {'kind': 'stack', 'suite': 'suite.json', 'stack': {
                'bridge_node': 'none', 'bridge': {'control': 'autoware', 'deadline_s': 5.0},
                'command': [stand_in(speed, 3.0)]}},
            'carla': {'mode': 'fake', 'fake_carla': fake_carla}, 'gates': gates})
    a, code = run(cfg(0, 'stops', {'pass_rate': {'min': 0.2, 'rule': 'refute'}, 'max_missed_deadlines': 0,
                                   'latency': {'p95_ms': 2000}}), out=str(tmp_path / 'a'))
    assert a['error'] is None, a.get('traceback') or a['error']
    assert code == 0 and all(r['passed'] for r in a['runs']) and a['latency']['n'] > 100
    assert a['latency']['p50_ms'] >= 3.0 and 'control wait' in a['latency']['source']
    assert a['subject']['describe']['launch'][0][:2] == [sys.executable, STAND_IN]
    b, code = run(cfg(14, 'does-not-stop', {'regressions': {'metrics': ['pass_rate', 'collision_rate', 'criticality']},
                                            'latency': {'p50_ms': 1.0}}),
                  out=str(tmp_path / 'b'), baseline=str(tmp_path / 'a'))
    assert b['error'] is None, b.get('traceback') or b['error']
    assert code == 1 and all(r['passed'] is False and r['metrics']['collision'] for r in b['runs'])
    cmp_ = b['comparison']
    assert cmp_['pairs'] == 3 and {'pass_rate', 'collision_rate'} <= set(cmp_['regressions'])
    assert sorted(cmp_['new_failures']) == ['stalled_vehicle seed 0', 'stalled_vehicle seed 1',
                                            'stalled_vehicle seed 2']
    g = {x['id']: x for x in b['gates']}
    assert g['regressions']['status'] == 'failed' and g['latency.p50_ms']['status'] == 'failed'
    md = (tmp_path / 'b' / 'summary.md').read_text()
    assert '| regression |' in md and 'Runs that passed in the baseline and fail now' in md
    root = ET.parse(tmp_path / 'b' / 'junit.xml').getroot()
    assert all(c.find('failure') is not None for c in root.find('testsuite').findall('testcase'))


def test_flaky_runs_are_retried_only_when_not_deterministic(tmp_path, fake_carla):
    _, proj = small_project(tmp_path)
    write(tmp_path / 'suite.json', suite('flaky', 'ros2', [dict(STALLED[0], seeds=[0])]))
    state = tmp_path / 'invocations'
    doc = {'schema': 'falcon.ci/1', 'name': 'flaky', 'project': 'p.falcon.json',
           'subject': {'kind': 'stack', 'suite': 'suite.json', 'stack': {
               'bridge_node': 'none', 'bridge': {'control': 'autoware', 'deadline_s': 5.0},
               'command': [stand_in(14, state=state, then_speed=0)]}},
           'carla': {'mode': 'fake', 'fake_carla': fake_carla}, 'flaky': {'retries': 2},
           'gates': {'pass_rate': {'min': 0.01, 'rule': 'estimate'}}}
    res, code = run(write(tmp_path / 'det.json', doc), out=str(tmp_path / 'det'))
    assert code == 2 and 'deterministic' in res['error'] and not state.exists()       # refused before any run
    doc['subject']['stack']['deterministic'] = False
    res, code = run(write(tmp_path / 'nondet.json', doc), out=str(tmp_path / 'nondet'))
    assert res['error'] is None, res.get('traceback') or res['error']
    r = res['runs'][0]
    assert code == 0 and r['flaky'] and r['passed'] and len(r['attempts']) == 2 and state.read_text() == '2'
    assert r['attempts'][0]['passed'] is False and r['attempts'][1]['passed'] is True
    assert res['not_deterministic_because'] == ['the stack is declared non-deterministic']
    root = ET.parse(tmp_path / 'nondet' / 'junit.xml').getroot()
    assert 'FLAKY' in root.find('.//testcase/system-out').text
    assert 'Flaky (failed, then passed on a retry; counted as pass)' in (tmp_path / 'nondet' / 'summary.md').read_text()


def test_cancel_by_sigterm(tmp_path, fake_carla):
    """SIGTERM (a CI system cancelling the job) while a run waits for its stack: exit 3, the outputs written, the
    stack's process stopped."""
    _, proj = small_project(tmp_path)
    write(tmp_path / 'suite.json', suite('cancel', 'ros2', [dict(STALLED[0], seeds=[0])]))
    pidfile = tmp_path / 'sleeper.pid'
    sleeper = [sys.executable, '-c', f'import os, time; open({str(pidfile)!r}, "w").write(str(os.getpid())); '
                                     'time.sleep(120)']
    cfg = write(tmp_path / 'c.json', {
        'schema': 'falcon.ci/1', 'name': 'cancel', 'project': 'p.falcon.json',
        'subject': {'kind': 'stack', 'suite': 'suite.json', 'stack': {
            'bridge_node': 'none', 'bridge': {'control': 'autoware', 'connect_wait_s': 100.0}, 'command': [sleeper]}},
        'carla': {'mode': 'fake', 'fake_carla': fake_carla}})
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy', 'ci', 'run', cfg, '--out', str(tmp_path / 'o')],
                         cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.monotonic()
    while not pidfile.exists() and time.monotonic() - t0 < 60 and p.poll() is None:
        time.sleep(0.1)
    assert pidfile.exists(), p.communicate(timeout=10)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    out, err = p.communicate(timeout=90)
    assert p.returncode == 3, out + err
    res = json.loads((tmp_path / 'o' / 'ci-results.json').read_text())
    assert res['verdict'] == 'cancelled' and res['exit_code'] == 3
    root = ET.parse(tmp_path / 'o' / 'junit.xml').getroot()
    assert root.find('.//testcase/error') is not None
    pid = int(pidfile.read_text())
    for _ in range(50):
        if not os.path.exists(f'/proc/{pid}') or 'zombie' in open(f'/proc/{pid}/status').read().lower():
            break
        time.sleep(0.1)
    assert not os.path.exists(f'/proc/{pid}') or 'zombie' in open(f'/proc/{pid}/status').read().lower()


# -------------------------------------------------------------------------------------------- policy and model
def test_policy_subject_with_the_results_store(tmp_path, fake_carla):
    """A policy over two routes; a run on main passes and is stored as main's baseline; the next run compares
    against the store (no change: the episodes repeat)."""
    torch = pytest.importorskip('torch')
    from falcon_autonomy.core import drivefeatures as DF
    from falcon_autonomy.core.policy import build_net, net_spec, save_policy
    torch.manual_seed(0)
    spec = net_spec('mlp', [8], 'tanh', 'tanh')
    save_policy(str(tmp_path / 'pol'), build_net(spec, DF.N_FEATURES, 2), {
        'name': 'tiny', 'trained_by': 'bc', 'features': list(DF.FEATURE_NAMES), 'bev': None,
        'normaliser': {'mean': [0.0] * DF.N_FEATURES, 'std': [100.0] * DF.N_FEATURES, 'clip': 5.0},
        'action': {'kind': 'continuous', 'keys': ['steer', 'accel']}, 'net': spec})
    small_project(tmp_path)
    store = tmp_path / 'store'
    cfg = write(tmp_path / 'pol.json', {
        'schema': 'falcon.ci/1', 'name': 'policy', 'project': 'p.falcon.json',
        'subject': {'kind': 'policy', 'policy': 'pol', 'routes': {'scenarios': ['lead_brake'], 'seeds_per': 2},
                    'route_pass': {'min_route_completion': 0.0, 'max_penalised_infractions': 99},
                    'max_episode_s': 2.0},
        'carla': {'mode': 'fake', 'fake_carla': fake_carla},
        'baseline': {'kind': 'store', 'store': 'store', 'branch': 'main'},
        'store': {'dir': 'store', 'branches': ['main']},
        'gates': {'pass_rate': {'min': 0.01, 'rule': 'estimate'}, 'latency': {'p99_ms': 5000},
                  'metrics': [{'metric': 'route_completion', 'min': 0.0}], 'regressions': {}}})
    env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REF_NAME': 'main', 'GITHUB_SHA': 'd' * 40}
    a, code = run(cfg, env, out=str(tmp_path / 'a'))
    assert a['error'] is None, a.get('traceback') or a['error']
    assert code == 0 and len(a['runs']) == 2 and a['latency']['n'] > 0 and 'decision' in a['latency']['source']
    assert {g['id']: g['status'] for g in a['gates']}['regressions'] == 'skipped'
    assert a['stored'] == str(store / 'main') and (store / 'main' / 'ci-runs.jsonl').is_file()
    assert (tmp_path / 'a' / 'leaderboard' / 'leaderboard.json').is_file()
    b, code = run(cfg, dict(env, GITHUB_REF_NAME='feature', GITHUB_SHA='e' * 40), out=str(tmp_path / 'b'))
    assert code == 0 and b['comparison']['pairs'] == 2 and b['comparison']['regressions'] == []
    assert b['baseline']['commit']['sha'] == 'd' * 40 and b['stored'] is None          # a feature branch: not stored
    assert json.loads((store / 'main' / 'ci-results.json').read_text())['commit']['sha'] == 'd' * 40


def test_model_subject_against_another_model_and_pinned_drives(tmp_path, fake_carla):
    pytest.importorskip('cv2')
    p, path = small_project(tmp_path, width=320, height=180, labels=['semantic', 'instance', 'boxes2d'])
    p.doc['models'] = [{'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
                        'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector',
                        'licence': 'GPL-3.0-or-later'},
                       {'id': 'oracle', 'name': 'Oracle', 'kind': 'detector', 'runtime': 'python',
                        'entry': 'test_ci:Oracle', 'licence': 'GPL-3.0-or-later'}]
    p.save(path)
    doc = {'schema': 'falcon.ci/1', 'name': 'detector', 'project': 'p.falcon.json',
           'subject': {'kind': 'model', 'model': 'syn', 'n_boot': 200,
                       'record': {'seeds': [3, 4], 'seconds': 1.0, 'traffic_vehicles': 8}},
           'carla': {'mode': 'fake', 'fake_carla': fake_carla}, 'baseline': {'kind': 'model', 'model': 'oracle'},
           'gates': {'metrics': [{'metric': 'AP', 'min': 0.0, 'rule': 'estimate'}], 'latency': {'p90_ms': 10000},
                     'regressions': {'n_boot': 200}}}
    a, code = run(write(tmp_path / 'm.json', doc), out=str(tmp_path / 'a'))
    assert a['error'] is None, a.get('traceback') or a['error']
    names = [r['name'] for r in a['runs']]
    assert names == ['drive-world-town10-seed3', 'drive-world-town10-seed4'] and all(r['informational']
                                                                                     for r in a['runs'])
    g = {x['id']: x for x in a['gates']}
    assert g['metric.AP']['status'] == 'passed' and g['latency.p90_ms']['status'] == 'passed'
    assert g['regressions']['status'] == 'failed' and 'AP' in a['comparison']['regressions']        # syn < oracle
    assert code == 1 and a['evaluation']['synthetic_model'] and 'SYNTHETIC' in (tmp_path / 'a' / 'summary.md') \
        .read_text()
    # the same drives pinned, the first run as the baseline: the same frames, no change
    drives = [r['dir'] for r in a['runs']]
    pinned = dict(doc, subject={'kind': 'model', 'model': 'syn', 'n_boot': 200, 'drives': drives},
                  baseline={'kind': 'dir', 'dir': str(tmp_path / 'a')})
    b, code = run(write(tmp_path / 'm2.json', pinned), out=str(tmp_path / 'b'))
    assert code == 0 and b['comparison']['regressions'] == [] and b['comparison']['pairs'] > 0
    # frames whose ground truth differs are not the same frames: refused
    ev = tmp_path / 'b' / 'evaluation'
    rows = [json.loads(x) for x in (ev / 'predictions.jsonl').read_text().splitlines()]
    rows[0]['gt'] = rows[0]['gt'][1:] if rows[0]['gt'] else [{'bbox': [0, 0, 5, 5], 'class': 'car', 'ignore': 0}]
    tam = tmp_path / 'tampered'
    tam.mkdir()
    (tam / 'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    with pytest.raises(CI.CIError, match='pin the test drives'):
        CI.compare_models(str(tam), str(ev), 'detector', 100)


# ------------------------------------------------------------------------------------------- GitHub templates
def test_init_writes_a_config_suite_and_a_workflow_that_passes_its_checks(tmp_path):
    yaml = pytest.importorskip('yaml')
    target = tmp_path / 'my-stack'
    target.mkdir()
    from falcon_autonomy.cli import main
    assert main(['ci', 'init', str(target), '--github', '--falcon-ref', 'v1.4.0']) == 0
    cfg, wf = target / 'falcon-ci.yml', target / '.github' / 'workflows' / 'falcon.yml'
    assert cfg.is_file() and (target / 'falcon-suite.json').is_file() and wf.is_file()
    assert main(['ci', 'validate', str(cfg)]) == 0
    assert G.lint_workflow_file(str(wf)) == []
    assert main(['ci', 'lint', str(wf)]) == 0
    w = yaml.safe_load(wf.read_text())
    on = w.get('on', w.get(True))
    assert set(on) == {'push', 'pull_request', 'workflow_dispatch'} and w['permissions'] == {'contents': 'read'}
    smoke, carla = w['jobs']['smoke'], w['jobs']['carla']
    assert smoke['runs-on'] == 'ubuntu-24.04' and carla['runs-on'] == list(G.RUNNER_LABELS)
    assert carla['needs'] == 'smoke' and 'head.repo.full_name == github.repository' in carla['if']
    steps = {s.get('name'): s for s in smoke['steps']}
    assert steps['Check out Falcon Autonomy']['with'] == {'repository': G.FALCON_REPO, 'ref': '${{ env.FALCON_REF }}',
                                                         'path': '.falcon'}
    assert '--fake-carla .falcon/tests/fake_carla' in steps['Falcon CI against the fake carla']['run']
    assert w['env']['FALCON_REF'] == 'v1.4.0'
    post = next(s for s in carla['steps'] if 'ci post' in s.get('run', ''))
    assert "vars.FALCON_CI_POST == 'true'" in post['if'] and '--post' in post['run']
    assert carla['permissions'] == {'contents': 'read', 'pull-requests': 'write'}
    assert all(s.get('if') == 'always()' for s in smoke['steps'] + carla['steps'] if 'upload-artifact' in
               s.get('uses', ''))
    # files are kept unless --force; a policy or model config passes the schema
    assert main(['ci', 'init', str(target), '--github']) == 2
    assert main(['ci', 'init', str(target), '--github', '--force']) == 0
    for kind in ('policy', 'model'):
        assert CI.validate_config(yaml.safe_load(G.config_template(kind))) == []
        assert G.lint_workflow(G.workflow(kind=kind)) == []
    if shutil.which('actionlint'):                   # the full linter when this machine has it
        assert not [x for x in G.lint_workflow_file(str(wf)) if x.startswith('actionlint')]


def test_lint_finds_broken_workflows():
    pytest.importorskip('yaml')
    bad = ('name: x\non:\n  push:\njobs:\n  a:\n    steps:\n      - uses: actions/checkout\n        run: echo\n'
           '  b:\n    runs-on: ubuntu-24.04\n    needs: nope\n    permissions:\n      contents: maybe\n    steps:\n'
           '      - run: echo ${{ secretz.TOKEN }}\n')
    probs = G.lint_workflow(bad)
    for want in ('jobs.a: no runs-on', 'exactly one of uses and run', 'not owner/repo@version', "needs 'nope'",
                 'permission contents', "unknown name 'secretz'"):
        assert any(want in p for p in probs), (want, probs)
    assert G.lint_workflow('jobs: {}\n') and G.lint_workflow('a: [\n')


# ------------------------------------------------------------------------------------------------------ posting
class FakeTransport:
    def __init__(self, comments=()):
        self.calls, self.comments = [], list(comments)

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method == 'GET':
            return self.comments if path.endswith('page=1') else []
        return {'html_url': f'https://example.invalid/{method.lower()}', 'id': 1}


class Refuse:
    def request(self, *a):
        raise AssertionError('a dry run must send nothing')


def _results(tmp_path, verdict='failed'):
    d = tmp_path / 'res'
    d.mkdir(exist_ok=True)
    res = {'schema': CI.CI_SCHEMA, 'name': 'aeb', 'verdict': verdict,
           'commit': {'sha': 'a' * 40, 'run_url': 'https://example.invalid/run/1', 'pr': 3},
           'gates': [{'name': 'pass rate (all runs) ≥ 0.9', 'status': 'failed'}]}
    (d / 'ci-results.json').write_text(json.dumps(res))
    (d / 'summary.md').write_text('## Falcon CI: aeb — FAILED\n')
    return d


def _args(d, **kw):
    a = dict(results=str(d), repo='o/r', pr=0, sha='', kind='comment', via='gh', api_url='', post=False)
    a.update(kw)
    return SimpleNamespace(**a)


def test_post_is_a_dry_run_unless_asked_and_refused_here(tmp_path, capsys):
    d = _results(tmp_path)
    assert G.main_post(_args(d), transport=Refuse(), env={}) == 0
    out = capsys.readouterr().out
    assert 'dry run: nothing was sent' in out and 'PUBLIC' in out and 'POST|PATCH /repos/o/r/issues/3/comments' in out
    assert G.main_post(_args(d, post=True), transport=Refuse(), env={'FALCON_CI_NO_POST': '1'}) == 2
    from falcon_autonomy.cli import main
    assert main(['ci', 'post', str(d), '--repo', 'o/r', '--pr', '3']) == 0             # no --post: nothing sent
    assert main(['ci', 'post', str(d), '--repo', 'o/r', '--pr', '3', '--post']) == 2   # FALCON_CI_NO_POST is set
    with pytest.raises(CI.CIError, match='OWNER/NAME'):
        G.plan({}, '', 'comment', '', 3, None)


def test_post_through_a_fake_transport(tmp_path, capsys):
    d = _results(tmp_path)
    t = FakeTransport()
    assert G.main_post(_args(d, post=True), transport=t, env={}) == 0
    assert [c[0] for c in t.calls] == ['GET', 'POST'] and t.calls[1][1] == '/repos/o/r/issues/3/comments'
    assert t.calls[1][2]['body'].startswith('<!-- falcon-ci:aeb -->\n## Falcon CI: aeb')
    t = FakeTransport([{'id': 41, 'body': 'other'}, {'id': 42, 'body': '<!-- falcon-ci:aeb -->\nold'}])
    assert G.main_post(_args(d, post=True), transport=t, env={}) == 0
    assert t.calls[-1][:2] == ('PATCH', '/repos/o/r/issues/comments/42')            # the earlier comment, edited
    t = FakeTransport()
    assert G.main_post(_args(d, post=True, kind='check'), transport=t, env={}) == 0
    m, path, body = t.calls[0]
    assert (m, path, body['head_sha'], body['conclusion']) == ('POST', '/repos/o/r/check-runs', 'a' * 40, 'failure')
    t = FakeTransport()
    assert G.main_post(_args(d, post=True, kind='status', sha='b' * 40), transport=t, env={}) == 0
    m, path, body = t.calls[0]
    assert path == '/repos/o/r/statuses/' + 'b' * 40 and body['state'] == 'failure' and len(body['description']) <= 140
    assert body['context'] == 'falcon-ci/aeb' and body['target_url'] == 'https://example.invalid/run/1'
    # the transports build the requests they would send (nothing leaves this process)
    seen = {}

    def fake_run(argv, input=None, **kw):
        seen.update(argv=argv, input=input)
        return SimpleNamespace(returncode=0, stdout='{"html_url": "u"}', stderr='')
    assert G.GhTransport(run=fake_run).request('POST', '/repos/o/r/issues/3/comments', {'body': 'x'}) == \
        {'html_url': 'u'}
    assert seen['argv'][:5] == ['gh', 'api', '--method', 'POST', '/repos/o/r/issues/3/comments']
    assert json.loads(seen['input']) == {'body': 'x'}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"id": 5}'

    def fake_open(req, timeout=0):
        seen.update(url=req.full_url, auth=req.get_header('Authorization'), data=req.data, method=req.get_method())
        return Resp()
    assert G.HttpTransport('t0ken', 'https://api.example.invalid', fake_open).request('PATCH', '/x', {'a': 1}) == \
        {'id': 5}
    assert (seen['url'], seen['auth'], seen['method']) == ('https://api.example.invalid/x', 'Bearer t0ken', 'PATCH')
    with pytest.raises(G.PostError, match='no token'):
        G.HttpTransport('')


# ----------------------------------------------------------------------------------------------------------- GUI
@pytest.mark.gui
def test_stack_page_lists_ci_results_and_writes_templates(window, qapp, tmp_path, fake_carla):
    """The Stack testing page's CI section: a CI run written into the workspace's ci/ folder is listed with its
    verdict and the gates it did not pass; the config and workflow are written into a chosen folder. Nothing posts."""
    from falcon_autonomy.ui.pages.stack import StackPage
    ws = window.workspace()
    sp = write(tmp_path / 'suite.json', suite('gui-ci'))
    proj = window.project
    proj.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    proj.save(proj.path)
    cfg = write(tmp_path / 'gui-ci.json', {'schema': 'falcon.ci/1', 'name': 'gui-ci', 'project': proj.path,
                                           'subject': {'kind': 'stack', 'suite': sp},
                                           'gates': {'pass_rate': {'min': 0.99, 'rule': 'demonstrate'}}})
    res, code = run(cfg, fake_carla=fake_carla, out=os.path.join(ws, 'ci', 'first'))
    assert code == 1, res.get('traceback') or res['error']
    page = StackPage(window)
    try:
        assert page.ci_runs.topLevelItemCount() == 1
        it = page.ci_runs.topLevelItem(0)
        assert (it.text(1), it.text(2)) == ('gui-ci', 'failed') and 'pass rate' in it.text(4)
        target = tmp_path / 'repo'
        target.mkdir()
        written = page.write_ci(True, str(target))
        assert {os.path.basename(w) for w in written} == {'falcon-ci.yml', 'falcon-suite.json', 'falcon.yml'}
        assert page.state.vals['State'].text().startswith('wrote ')
        assert page.write_ci(False, str(target)) == []                     # files that exist are kept
        pump(qapp, timeout=0.2)
    finally:
        page.timer.stop()
        page.deleteLater()
