"""CI mode: run a driving stack's (or a Falcon policy's, or a perception model's) scenario suite headless on every
commit, judge it with gates, and report in the forms CI systems read (standard library + numpy at use; the runners,
which are Qt-free, are imported inside the functions). `falcon ci run CONFIG`; docs/ci.md is the reference.

A CI config (CONFIG_SCHEMA, falcon.ci/1; YAML or JSON; CONFIG_JSON_SCHEMA is its JSON Schema) names:

  subject   what is tested — a stack: a falcon.suite/1 suite (core.stack) whose stack is a ROS 2 stack started from
            its Docker image or from commands beside Falcon's bridge node (or a baseline driver); a policy: a Falcon
            policy folder driven over Leaderboard-style routes (core.leaderboard); a model: a registered model scored on
            pinned or freshly recorded drives (core.harness) — and the repository whose commit is under test
  carla     Falcon's own server ("falcon": CarlaService starts it, the runs use it, it is stopped at the end) or the
            fake carla ("fake": smoke runs on any machine; its numbers test the integration, not the stack)
  baseline  none; a previous run's folder ("dir": a `falcon ci run` output, or a suite, Leaderboard or evaluation
            folder); a results store's branch ("store", e.g. main, written by earlier runs through `store`); or, for a
            model, another registered model evaluated on the same drives now ("model")
  gates     pass_rate (a minimum judged with Wilson's 95 % interval), metrics (a minimum or maximum of any reported
            number, judged with its interval), regressions (none against the baseline by a paired bootstrap),
            rule_violations (a budget per km, core.rules), latency (percentile budgets: a ROS 2 stack's control wait,
            a policy's decision time, a model's inference time), max_missed_deadlines (ROS 2 stacks)
  flaky     retries, allowed only when the mode is not deterministic, and recorded

How an interval judges a limit (`rule`): "demonstrate" passes only when the whole 95 % interval is on the good side
of the limit (it needs enough runs: e.g. 35 passing runs before a pass rate of 0.9 can be demonstrated); "refute"
fails only when the whole interval is on the bad side (a shortfall shown at 95 %); "estimate" compares the point
estimate. Regressions: a per-run metric whose paired bootstrap interval of now − baseline (pairs: the same scenario,
seed and parameter values) lies wholly on its bad side.

Outputs in OUT: ci-results.json (CI_SCHEMA: the verdict, gates, runs, comparison, the commit under test, provenance),
ci-runs.jsonl (one RUN_SCHEMA line per run: what a later run pairs against), junit.xml (one testcase per run and per
gate), summary.md (for $GITHUB_STEP_SUMMARY and pull-request comments), report.html (core.reports kind "ci"),
ci-config.json (the resolved config) and run.json.

Exit codes (a test runner's convention; docs/ci.md): 0 every gate passed, 1 a gate failed, 2 an error (a config that
does not validate, no server, a run that did not finish, a gate that could not be judged), 3 cancelled (SIGINT or
SIGTERM).

Determinism: kinematic motion with no background traffic repeats bit for bit (docs/stack-testing.md); physics motion,
traffic and CARLA's autopilot do not. The config defaults to kinematic with no traffic, and a retry is refused in a
deterministic mode (it would repeat the failure exactly). A ROS 2 stack is taken as deterministic given its inputs
unless the config says otherwise (stack.deterministic: false).
"""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time

CONFIG_SCHEMA = 'falcon.ci/1'
CI_SCHEMA = 'falcon.ci_results/1'
RUN_SCHEMA = 'falcon.ci_run/1'
KINDS = ('stack', 'policy', 'model')
RULES = ('demonstrate', 'refute', 'estimate')
VERDICT_CODE = {'passed': 0, 'failed': 1, 'error': 2, 'cancelled': 3}
OUT_FILES = ('ci-results.json', 'ci-runs.jsonl', 'junit.xml', 'summary.md', 'report.html', 'run.json',
             'ci-config.json')
OUT_DIRS = ('suite', 'leaderboard', 'evaluation', 'baseline-evaluation', 'drives', 'project', 'retries', '.locks')
SMALL_N = 5                    # fewer pairs: the paired bootstrap's interval is flagged as not reliable
LATENCY_KEYS = ('p50_ms', 'p90_ms', 'p95_ms', 'p99_ms', 'max_ms', 'mean_ms')
MODEL_LATENCY_KEYS = ('p50_ms', 'p90_ms', 'p99_ms', 'mean_ms')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Per-run metrics compared with a baseline: name -> (the run record's field, direction (+1: higher is better), how
# the runs combine: the mean over runs, or violations per km as a ratio of sums).
STACK_METRICS = {'pass_rate': ('passed', +1, 'mean'), 'collision_rate': ('collision', -1, 'mean'),
                 'criticality': ('criticality', -1, 'mean'), 'min_ttc_s': ('min_ttc_s', +1, 'mean'),
                 'missed_deadlines': ('missed_deadlines', -1, 'mean'),
                 'rule_violations_per_km': ('rule_violations', -1, 'per_km')}
POLICY_METRICS = {'pass_rate': ('passed', +1, 'mean'), 'driving_score': ('driving_score', +1, 'mean'),
                  'route_completion': ('route_completion', +1, 'mean'),
                  'infraction_penalty': ('infraction_penalty', +1, 'mean'),
                  'rule_violations_per_km': ('rule_violations', -1, 'per_km')}
MODEL_REGRESSION_DEFAULT = ('AP', 'AP50', 'AP75', 'AR100', 'mIoU')

_RULE = {'enum': list(RULES)}
_PATHS = {'type': 'string'}
CONFIG_JSON_SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema', 'title': CONFIG_SCHEMA, 'type': 'object',
    'required': ['schema', 'subject'], 'additionalProperties': False,
    'properties': {
        'schema': {'const': CONFIG_SCHEMA}, 'name': {'type': 'string', 'minLength': 1}, 'about': {'type': 'string'},
        'out': _PATHS, 'project': _PATHS, 'seed': {'type': 'integer'},
        'subject': {'type': 'object', 'required': ['kind'], 'additionalProperties': False, 'properties': {
            'kind': {'enum': list(KINDS)}, 'repo': _PATHS,
            'suite': {'type': ['string', 'object']},
            'stack': {'type': 'object', 'additionalProperties': False, 'properties': {
                'driver': {'enum': ['ros2', 'follower', 'autopilot']}, 'image': {'type': 'string'},
                'image_args': {'type': 'array', 'items': {'type': ['string', 'number']}},
                'image_env': {'type': 'object', 'additionalProperties': {'type': ['string', 'number']}},
                'command': {'type': 'array', 'items': {'type': 'array', 'minItems': 1,
                                                       'items': {'type': ['string', 'number']}}},
                'bridge_node': {'enum': ['image', 'native', 'none']}, 'bridge': {'type': 'object'},
                'test_stack': {'type': 'boolean'}, 'speed': {'type': 'number'},
                'domain_id': {'type': 'integer', 'minimum': 0, 'maximum': 232},
                'deterministic': {'type': 'boolean'}}},
            'world': {'type': 'string'}, 'rig': {'type': 'string'}, 'fps': {'type': 'number', 'exclusiveMinimum': 0},
            'motion': {'enum': ['kinematic', 'physics']}, 'traffic_vehicles': {'type': 'integer', 'minimum': 0},
            'criteria': {'type': 'object'}, 'rules': {'type': 'string'},
            'policy': _PATHS, 'device': {'type': 'string'},
            'routes': {'type': 'object', 'additionalProperties': False, 'properties': {
                'scenarios': {'type': 'array', 'items': {'type': 'string'}},
                'seeds_per': {'type': 'integer', 'minimum': 1}, 'free': {'type': 'integer', 'minimum': 0},
                'seed': {'type': 'integer'}}},
            'route_pass': {'type': 'object', 'additionalProperties': False, 'properties': {
                'min_route_completion': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'max_penalised_infractions': {'type': 'integer', 'minimum': 0}}},
            'max_episode_s': {'type': 'number', 'exclusiveMinimum': 0},
            'model': {'type': 'string'}, 'drives': {'type': 'array', 'items': {'type': 'string'}},
            'record': {'type': 'object', 'additionalProperties': False, 'required': ['seeds'], 'properties': {
                'world': {'type': 'string'}, 'rig': {'type': 'string'},
                'seconds': {'type': 'number', 'exclusiveMinimum': 0}, 'fps': {'type': 'number', 'exclusiveMinimum': 0},
                'seeds': {'type': 'array', 'minItems': 1, 'items': {'type': 'integer'}},
                'traffic_vehicles': {'type': 'integer', 'minimum': 0}}},
            'camera': {'type': 'string'}, 'every': {'type': 'integer', 'minimum': 1},
            'n_boot': {'type': 'integer', 'minimum': 50}}},
        'carla': {'type': 'object', 'additionalProperties': False, 'properties': {
            'mode': {'enum': ['falcon', 'fake']}, 'fake_carla': _PATHS, 'keep_server': {'type': 'boolean'}}},
        'baseline': {'type': 'object', 'additionalProperties': False, 'properties': {
            'kind': {'enum': ['none', 'dir', 'store', 'model']}, 'dir': _PATHS, 'store': _PATHS,
            'branch': {'type': 'string'}, 'model': {'type': 'string'}}},
        'store': {'type': ['object', 'null'], 'additionalProperties': False, 'properties': {
            'dir': _PATHS, 'branches': {'type': 'array', 'items': {'type': 'string'}},
            'only_if_passed': {'type': 'boolean'}}},
        'gates': {'type': 'object', 'additionalProperties': False, 'properties': {
            'pass_rate': {'type': 'object', 'required': ['min'], 'additionalProperties': False, 'properties': {
                'min': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'rule': _RULE,
                'per_scenario': {'type': 'boolean'}}},
            'metrics': {'type': 'array', 'items': {'type': 'object', 'required': ['metric'],
                                                   'additionalProperties': False, 'properties': {
                'metric': {'type': 'string'}, 'min': {'type': 'number'}, 'max': {'type': 'number'}, 'rule': _RULE}}},
            'regressions': {'type': 'object', 'additionalProperties': False, 'properties': {
                'enabled': {'type': 'boolean'}, 'metrics': {'type': ['array', 'string']},
                'allow': {'type': 'array', 'items': {'type': 'string'}}, 'n_boot': {'type': 'integer', 'minimum': 100},
                'on_missing_baseline': {'enum': ['skip', 'error']}}},
            'rule_violations': {'type': 'object', 'additionalProperties': False, 'properties': {
                'max_per_km': {'type': 'number', 'minimum': 0}, 'rule': _RULE,
                'rules': {'type': 'object', 'additionalProperties': {'type': 'number', 'minimum': 0}}}},
            'latency': {'type': 'object', 'additionalProperties': False, 'minProperties': 1,
                        'properties': {k: {'type': 'number', 'exclusiveMinimum': 0} for k in LATENCY_KEYS}},
            'max_missed_deadlines': {'type': 'integer', 'minimum': 0}}},
        'flaky': {'type': 'object', 'additionalProperties': False, 'properties': {
            'retries': {'type': 'integer', 'minimum': 0, 'maximum': 5}, 'count_as': {'enum': ['pass', 'fail']}}},
        'outputs': {'type': 'object', 'additionalProperties': False, 'properties': {
            'html': {'type': 'boolean'}, 'github_step_summary': {'enum': ['auto', True, False]}}}}}


class CIError(Exception):
    pass


# =========================================================================================================== config
def read_doc(path: str) -> dict:
    """A YAML or JSON document (YAML needs PyYAML; JSON does not)."""
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as e:
        raise CIError(f'cannot read {path}: {e}') from None
    if path.endswith(('.yml', '.yaml')) or not text.lstrip().startswith('{'):
        try:
            import yaml
        except ImportError:
            raise CIError(f'{path}: a YAML config needs PyYAML (pip install PyYAML); or write it as JSON') from None
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise CIError(f'{path}: not valid YAML: {e}') from None
    else:
        try:
            doc = json.loads(text)
        except ValueError as e:
            raise CIError(f'{path}: not valid JSON: {e}') from None
    if not isinstance(doc, dict):
        raise CIError(f'{path}: not a mapping')
    return doc


def _path(base: str, p) -> str:
    """A config path: ${VAR} and ~ expanded, relative to the config's folder; '' when unset or a variable is not."""
    if not p:
        return ''
    s = os.path.expanduser(os.path.expandvars(str(p)))
    if '$' in s:                       # an unset environment variable: the path is not configured here
        return ''
    return os.path.abspath(os.path.join(base, s))


def resolve(raw: dict, base: str, overrides: dict | None = None) -> dict:
    """The config with its defaults filled in and every path absolute (overrides: the command line's)."""
    o = overrides or {}
    c = copy.deepcopy(raw)
    c.setdefault('name', 'falcon-ci')
    c.setdefault('seed', 0)
    sub = c.setdefault('subject', {})
    sub['repo'] = _path(base, sub.get('repo', '.'))
    for k in ('policy',):
        if sub.get(k):
            sub[k] = _path(base, sub[k])
    if isinstance(sub.get('suite'), str):
        sub['suite'] = _path(base, sub['suite'])
    if sub.get('drives'):
        found = []
        for pat in sub['drives']:
            p = _path(base, pat)
            hits = sorted(glob.glob(p)) if any(ch in p for ch in '*?[') else [p]
            found += [h for h in hits if h not in found]
        sub['drives'] = found
    st = sub.get('stack') or {}
    if st.get('command'):
        st['command'] = [[str(a) for a in cmd] for cmd in st['command']]
    c['project'] = _path(base, c.get('project'))
    car = c.setdefault('carla', {})
    car.setdefault('mode', 'falcon')
    car.setdefault('keep_server', False)
    if o.get('mode'):
        car['mode'] = o['mode']
    if o.get('fake_carla'):
        car['fake_carla'] = os.path.abspath(o['fake_carla'])
        car['mode'] = o.get('mode') or 'fake'
    elif car.get('fake_carla'):
        car['fake_carla'] = _path(base, car['fake_carla'])
    if car['mode'] == 'fake' and not car.get('fake_carla'):
        car['fake_carla'] = os.path.join(ROOT, 'tests', 'fake_carla')
    if o.get('keep_server'):
        car['keep_server'] = True
    b = c.setdefault('baseline', {'kind': 'none'})
    b.setdefault('kind', 'none')
    if o.get('no_baseline'):
        c['baseline'] = b = {'kind': 'none'}
    elif o.get('baseline'):
        c['baseline'] = b = {'kind': 'dir', 'dir': os.path.abspath(o['baseline'])}
    for k in ('dir', 'store'):
        if b.get(k):
            b[k] = _path(base, b[k])
    if c.get('store'):
        c['store'] = dict(c['store'], dir=_path(base, c['store'].get('dir')))
        c['store'].setdefault('branches', ['main'])
        c['store'].setdefault('only_if_passed', True)
    c.setdefault('gates', {})
    fl = c.setdefault('flaky', {})
    fl.setdefault('retries', 0)
    fl.setdefault('count_as', 'pass')
    out = c.setdefault('outputs', {})
    out.setdefault('html', True)
    out.setdefault('github_step_summary', 'auto')
    if o.get('out'):
        c['out'] = os.path.abspath(o['out'])
    elif c.get('out'):
        c['out'] = _path(base, c['out'])
    else:
        c['out'] = os.path.abspath(os.path.join(base, 'falcon-ci-out'))
    c['config_dir'] = base
    return c


def validate_config(c: dict) -> list:
    """Problems with a config (raw or resolved): its JSON Schema, then what the schema cannot say."""
    out = []
    try:
        import jsonschema
        v = jsonschema.Draft202012Validator(CONFIG_JSON_SCHEMA)
        for e in sorted(v.iter_errors({k: x for k, x in c.items() if k != 'config_dir'}), key=lambda e: list(e.path)):
            where = '.'.join(str(p) for p in e.absolute_path) or '(top)'
            out.append(f'{where}: {e.message}')
    except ImportError:                                         # a dependency, but say so rather than skip silently
        out.append('jsonschema is not installed: the config cannot be validated')
    if out:
        return out
    sub, gates, kind = c['subject'], c.get('gates') or {}, c['subject']['kind']
    if kind == 'stack':
        if not sub.get('suite'):
            out.append('subject.suite: a stack is tested with a suite (a falcon.suite/1 file or the suite itself)')
        elif isinstance(sub['suite'], str) and not os.path.isfile(sub['suite']) and 'config_dir' in c:
            out.append(f'subject.suite: no file {sub["suite"]}')
        st = sub.get('stack') or {}
        if st.get('test_stack') and (st.get('image') or st.get('command')):
            out.append('subject.stack: test_stack is Falcon\'s own test follower; give it, or an image / command')
        if st.get('driver') and st['driver'] != 'ros2' and (st.get('image') or st.get('command')):
            out.append(f'subject.stack: an image or command runs a ROS 2 stack; driver is {st["driver"]}')
    elif kind == 'policy':
        if not sub.get('policy'):
            out.append('subject.policy: the policy folder to test')
        elif 'config_dir' in c and not os.path.isfile(os.path.join(sub['policy'], 'policy.json')):
            out.append(f'subject.policy: {sub["policy"]} has no policy.json')
    else:
        if not sub.get('model'):
            out.append('subject.model: the id of a model registered in the project')
        if not c.get('project'):
            out.append('project: a model is tested from a project (where it is registered)')
        if bool(sub.get('drives')) == bool(sub.get('record')):
            out.append('subject: give drives (recorded drives to score on) or record (drives to record), not both')
        if 'config_dir' in c and sub.get('drives'):
            out += [f'subject.drives: no run folder {d}' for d in sub['drives'] if not os.path.isdir(d)]
        if 'pass_rate' in gates:
            out.append('gates.pass_rate: a model has no runs to pass; judge its numbers with gates.metrics')
        if 'max_missed_deadlines' in gates:
            out.append('gates.max_missed_deadlines: only a ROS 2 stack has deadlines')
        bad = sorted(set(gates.get('latency') or {}) - set(MODEL_LATENCY_KEYS))
        if bad:
            out.append(f'gates.latency: a model\'s latency has {", ".join(MODEL_LATENCY_KEYS)}, not {", ".join(bad)}')
        if (c.get('flaky') or {}).get('retries'):
            out.append('flaky.retries: a model\'s evaluation is deterministic; nothing to retry')
    if kind == 'policy' and 'max_missed_deadlines' in gates:
        out.append('gates.max_missed_deadlines: only a ROS 2 stack has deadlines')
    if kind != 'stack' and (gates.get('rule_violations') or {}).get('rules'):
        out.append('gates.rule_violations.rules: budgets per rule are for stacks; give max_per_km')
    if kind == 'model' and 'rule_violations' in gates:
        out.append('gates.rule_violations: a model does not drive')
    for k, g in enumerate(gates.get('metrics') or ()):
        if ('min' in g) == ('max' in g):
            out.append(f'gates.metrics[{k}]: give min or max')
    reg = gates.get('regressions') or {}
    if isinstance(reg.get('metrics'), str) and reg['metrics'] != 'all':
        out.append('gates.regressions.metrics: a list of metric names, or "all"')
    elif isinstance(reg.get('metrics'), list) and kind in ('stack', 'policy'):
        known = STACK_METRICS if kind == 'stack' else POLICY_METRICS
        bad = sorted(set(reg['metrics']) - set(known))
        if bad:
            out.append(f'gates.regressions.metrics: {", ".join(bad)} not among {", ".join(known)}')
    b = c.get('baseline') or {}
    if b.get('kind') == 'dir' and not b.get('dir'):
        out.append('baseline.dir: the folder of a previous run')
    if b.get('kind') == 'store' and not b.get('branch'):
        out.append('baseline.branch: the stored branch to compare against (e.g. main)')
    if b.get('kind') == 'model' and (kind != 'model' or not b.get('model')):
        out.append('baseline.model: another registered model, for a model subject')
    if (c.get('carla') or {}).get('mode') == 'fake' and 'config_dir' in c and \
            not os.path.isdir(c['carla'].get('fake_carla') or ''):
        out.append(f'carla.fake_carla: no fake carla package at {c["carla"].get("fake_carla")!r} (give --fake-carla)')
    return out


# ============================================================================================ the commit under test
_CRED = re.compile(r'(?<=://)[^/@]+@')


def _git(repo: str, *args) -> str | None:
    try:
        p = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def commit_info(repo: str | None = None, env=None) -> dict:
    """The commit under test: read from the repository (git) when it is one, and from the CI system's variables
    (GitHub Actions, GitLab CI, Jenkins). Credentials in a remote URL are removed."""
    env = os.environ if env is None else env
    out = {'sha': None, 'branch': None, 'dirty': None, 'remote': None, 'source': None, 'ci': None, 'ci_sha': None,
           'repository': None, 'pr': None, 'run_url': None}
    if env.get('GITHUB_ACTIONS') == 'true':
        ref = env.get('GITHUB_REF', '')
        m = re.match(r'refs/pull/(\d+)/', ref)
        out.update(ci='github', ci_sha=env.get('GITHUB_SHA'), repository=env.get('GITHUB_REPOSITORY'),
                   branch=env.get('GITHUB_HEAD_REF') or env.get('GITHUB_REF_NAME'),
                   pr=int(m.group(1)) if m else None,
                   run_url=(f'{env.get("GITHUB_SERVER_URL", "https://github.com")}/{env["GITHUB_REPOSITORY"]}/actions/'
                            f'runs/{env["GITHUB_RUN_ID"]}') if env.get('GITHUB_REPOSITORY') and env.get('GITHUB_RUN_ID')
                   else None)
    elif env.get('GITLAB_CI'):
        mr = env.get('CI_MERGE_REQUEST_IID')
        out.update(ci='gitlab', ci_sha=env.get('CI_COMMIT_SHA'), repository=env.get('CI_PROJECT_PATH'),
                   branch=env.get('CI_MERGE_REQUEST_SOURCE_BRANCH_NAME') or env.get('CI_COMMIT_REF_NAME'),
                   pr=int(mr) if mr and mr.isdigit() else None, run_url=env.get('CI_JOB_URL'))
    elif env.get('JENKINS_URL') or env.get('BUILD_URL'):
        ch = env.get('CHANGE_ID')
        out.update(ci='jenkins', ci_sha=env.get('GIT_COMMIT'), branch=env.get('BRANCH_NAME') or env.get('GIT_BRANCH'),
                   pr=int(ch) if ch and ch.isdigit() else None, run_url=env.get('BUILD_URL'))
    if repo and os.path.isdir(repo):
        sha = _git(repo, 'rev-parse', 'HEAD')
        if sha:
            out['sha'], out['source'] = sha, f'git ({repo})'
            st = _git(repo, 'status', '--porcelain', '--untracked-files=no')
            out['dirty'] = bool(st) if st is not None else None
            br = _git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')
            if not out['branch'] and br and br != 'HEAD':
                out['branch'] = br
            rem = _git(repo, 'remote', 'get-url', 'origin')
            out['remote'] = _CRED.sub('', rem) if rem else None
    if not out['sha'] and out['ci_sha']:
        out['sha'], out['source'] = out['ci_sha'], f'{out["ci"]} environment'
    return out


def branch_dir(branch: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', branch).strip('.') or '_'


# =================================================================================================== determinism
def determinism(motion: str, traffic: int, driver: str = '', declared=None) -> tuple:
    """(deterministic, reasons it is not)."""
    why = []
    if motion != 'kinematic':
        why.append('physics motion')
    if traffic:
        why.append(f'{traffic} background vehicles (CARLA\'s traffic manager)')
    if driver == 'autopilot':
        why.append('CARLA\'s autopilot (physics)')
    if declared is False:
        why.append('the stack is declared non-deterministic')
    return not why, why


# ============================================================================================== intervals, numbers
def _f(v):
    """A finite float, or None (booleans count as 0 / 1)."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def runs_to_demonstrate(min_rate: float, cap: int = 100000) -> int | None:
    """The fewest runs, all passing, whose Wilson 95 % lower bound reaches min_rate."""
    from .stack import wilson
    if min_rate <= 0:
        return 1
    n = 1
    while n <= cap:
        if wilson(n, n)[0] >= min_rate:
            return n
        n = n + 1 if n < 64 else int(n * 1.05) + 1
    return None


def judge(est, lo, hi, limit: float, side: str, rule: str):
    """(ok or None when it cannot be judged, the number judged). side 'min': the value must reach the limit; 'max':
    it must not exceed it. rule: RULES (module doc)."""
    if rule == 'estimate':
        v = est
    elif rule == 'demonstrate':
        v = lo if side == 'min' else hi
    else:
        v = hi if side == 'min' else lo
    if v is None:
        return None, None
    return (v >= limit if side == 'min' else v <= limit), v


def summarise_records(records: list, metrics: dict, seed: int = 0, n_boot: int = 1000) -> dict:
    """Over the finished runs: each metric with its 95 % interval (pass and collision rates: Wilson; means: a
    percentile bootstrap over runs; per km: a bootstrap of the ratio of sums), the runs, passes and failures."""
    from .metrics.uncertainty import bootstrap
    from .stack import wilson
    ok = [r for r in records if r['status'] == 'ok']
    out = {'runs': len(ok), 'not_finished': len(records) - len(ok),
           'passed': sum(1 for r in ok if r.get('passed')), 'failed': sum(1 for r in ok if r.get('passed') is False),
           'flaky': sum(1 for r in ok if r.get('flaky')), 'metrics': {}}
    for name, (field, direction, how) in metrics.items():
        vals = [_f(r['metrics'].get(field)) for r in ok]
        vals = [v for v in vals if v is not None]
        row = {'value': None, 'ci_low': None, 'ci_high': None, 'n': len(vals), 'direction': direction}
        if how == 'mean' and vals:
            row['value'] = sum(vals) / len(vals)
            if field in ('passed', 'collision'):
                lo, hi = wilson(int(round(sum(vals))), len(vals))
                row.update(ci_low=lo, ci_high=hi, interval='Wilson score, 95 %')
            elif len(vals) > 1:
                b = bootstrap(lambda xs: sum(xs) / len(xs), vals, n=n_boot, seed=seed, unit='image')
                row.update(ci_low=b['ci_low'], ci_high=b['ci_high'], interval='percentile bootstrap over runs')
            else:
                row['interval'] = 'one run: no interval'
        elif how == 'per_km':
            items = [(_f(r['metrics'].get('rule_violations')), _f(r['metrics'].get('distance_km'))) for r in ok]
            items = [(v, k) for v, k in items if v is not None and k is not None]
            km = sum(k for _, k in items)
            row['n'] = len(items)
            if items and km > 0:
                row['value'] = sum(v for v, _ in items) / km
                if len(items) > 1:
                    b = bootstrap(lambda xs: (sum(v for v, _ in xs) / sum(k for _, k in xs))
                                  if sum(k for _, k in xs) > 0 else float('nan'), items, n=n_boot, seed=seed,
                                  unit='image')
                    row.update(ci_low=b['ci_low'], ci_high=b['ci_high'],
                               interval='percentile bootstrap over runs (ratio of sums)')
                row['distance_km'] = km
                row['violations'] = sum(v for v, _ in items)
        if field == 'min_ttc_s' and vals:
            row['min'] = min(vals)
        out['metrics'][name] = row
    return out


def pooled_latency(values_ms: list, source: str) -> dict | None:
    import numpy as np
    if not values_ms:
        return None
    a = np.asarray(values_ms, float)
    return {'source': source, 'n': int(a.size), 'p50_ms': float(np.percentile(a, 50)),
            'p90_ms': float(np.percentile(a, 90)), 'p95_ms': float(np.percentile(a, 95)),
            'p99_ms': float(np.percentile(a, 99)), 'max_ms': float(a.max()), 'mean_ms': float(a.mean())}


# ============================================================================================ subjects: the stack
def pair_key(group, seed, values) -> str:
    from .provenance import sha256_json
    return sha256_json({'scenario': group, 'seed': seed, 'values': values or {}})[:24]


def load_suite(sub: dict) -> dict:
    """The suite of a stack subject, with the subject's overrides applied (stack, motion, traffic, world, …)."""
    s = sub['suite']
    doc = copy.deepcopy(s) if isinstance(s, dict) else read_doc(s)
    st = dict(doc.get('stack') or {})
    ov = sub.get('stack') or {}
    if ov.get('driver'):
        st['driver'] = ov['driver']
    if ov.get('bridge'):
        st['bridge'] = dict(st.get('bridge') or {}, **ov['bridge'])
    if 'domain_id' in ov:
        st['domain_id'] = int(ov['domain_id'])
    if st.get('driver') == 'ros2' and (ov.get('image') or ov.get('command') or ov.get('test_stack')):
        st['launch'] = stack_launch(ov, st)
    doc['stack'] = st
    for k in ('world', 'rig', 'fps', 'motion', 'traffic_vehicles', 'criteria', 'rules'):
        if k in sub:
            doc[k] = sub[k]
    doc.setdefault('motion', 'kinematic')
    doc.setdefault('traffic_vehicles', 0)
    return doc


def stack_launch(ov: dict, st: dict) -> list:
    """A ROS 2 stack's launch commands (core.stack "launch"): Falcon's bridge node (in the bridge image, natively, or
    not at all when your command starts it) and the stack — its Docker image on the host network, and/or commands."""
    from . import stackbridge as SB
    control = (st.get('bridge') or {}).get('control', 'autoware')
    if ov.get('test_stack'):
        from ..sim.stacksuite import test_stack_launch
        return test_stack_launch(control, float(ov.get('speed', 8.0)))
    cmds = []
    node = ov.get('bridge_node', 'image')
    if node == 'image':
        b = SB.bridge_command('{port}', name='{name}-bridge', domain_id=0)
        b[b.index('ROS_DOMAIN_ID=0')] = 'ROS_DOMAIN_ID={domain}'
        cmds.append(b)
    elif node == 'native':
        cmds.append(['env', 'ROS_DOMAIN_ID={domain}'] + SB.bridge_command('{port}', docker=False))
    if ov.get('image'):
        image = os.path.expandvars(str(ov['image']))
        if '$' in image:
            raise CIError(f'subject.stack.image {ov["image"]!r}: an environment variable in it is not set')
        argv = ['docker', 'run', '--rm', '--name', '{name}-stack', '--network', 'host', '--ipc', 'host',
                '-e', 'ROS_DOMAIN_ID={domain}']
        for k, v in sorted((ov.get('image_env') or {}).items()):
            argv += ['-e', f'{k}={v}']
        cmds.append(argv + [image] + [str(a) for a in ov.get('image_args') or ()])
    cmds += [[str(a) for a in c] for c in ov.get('command') or ()]
    return cmds


def _read(path):
    from .provenance import read_json
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def stack_record(row: dict) -> dict:
    """A run record (RUN_SCHEMA) from a suite's results row, its run.json and rules.json."""
    run = _read(os.path.join(row['run'], 'run.json')) or {} if row.get('run') else {}
    rules = _read(os.path.join(row['run'], 'rules.json')) if row.get('run') else None
    br = run.get('bridge') or {}
    ok = row.get('status') == 'ok' and row.get('passed') is not None
    m = {'passed': row.get('passed'), 'collision': bool(row.get('collision')) if ok else None,
         'criticality': row.get('criticality'), 'min_ttc_s': _f(row.get('min_ttc_s')),
         'min_distance_m': _f(row.get('min_distance_m')),
         'missed_deadlines': row.get('missed_deadlines') if br else (0 if ok else None),
         'rule_violations': ((rules or {}).get('totals') or {}).get('violations'),
         'distance_km': (rules or {}).get('distance_km'),
         'control_wait_p95_ms': (br.get('wait_ms') or {}).get('p95'),
         'steps_per_s': row.get('steps_per_s')}
    return {'schema': RUN_SCHEMA, 'id': str(row['key'])[:16], 'key': row['key'], 'entry': row.get('entry'),
            'pair': pair_key(row['scenario'], row['seed'], row.get('values')), 'group': row['scenario'],
            'group_name': row.get('scenario_name', row['scenario']), 'seed': row['seed'], 'values': row.get('values'),
            'name': f'{row["scenario"]} seed {row["seed"]}', 'status': 'ok' if ok else 'not finished',
            'passed': row.get('passed') if ok else None, 'error': row.get('error'),
            'failed_checks': [f'{c.get("name")}: {c.get("detail")}' for c in row.get('checks') or ()
                              if not c.get('ok')],
            'metrics': m, 'time_s': (run.get('performance') or {}).get('wall_s'), 'dir': row.get('run'),
            'attempts': [{'dir': row.get('run'), 'status': row.get('status'), 'passed': row.get('passed')}],
            'flaky': False}


def _wait_ms(run_dir: str) -> list:
    """Every step's control wait (ms) of a ROS 2 run, from its frames.jsonl."""
    out = []
    p = os.path.join(run_dir or '', 'frames.jsonl')
    if not os.path.isfile(p):
        return out
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            if '"bridge"' in line:
                w = (json.loads(line).get('bridge') or {}).get('wait_ms')
                if w is not None:
                    out.append(float(w))
    return out


def run_stack(ctx) -> dict:
    """The suite into OUT/suite (sim.stacksuite.SuiteRunner: fresh, so nothing is resumed from another commit),
    retries where allowed, the traffic-rule checks of every counted run, the pooled control wait."""
    from . import rules as RU
    from . import stack as ST
    from ..sim.stacksuite import SuiteRunner, run_one
    doc, cfg = ctx.suite, ctx.cfg
    suite_dir = os.path.join(ctx.out, 'suite')
    SuiteRunner(ctx.svc, ctx.project, doc, suite_dir, lambda f, t: ctx.log(f'suite: {t}')).run()
    plan = (_read(os.path.join(suite_dir, 'run.json')) or {}).get('runs') or []
    rows = {r['key']: r for r in ST.read_results(suite_dir)}
    records = [stack_record(rows[k]) for k in plan if k in rows]
    st = doc['stack']
    launch = (st.get('launch') or []) if st['driver'] == 'ros2' else []
    retries, count_as = int(cfg['flaky']['retries']), cfg['flaky']['count_as']
    for rec in records:
        if not retries or (rec['status'] == 'ok' and rec['passed']):
            continue
        run_cfg = _read(os.path.join(rec['dir'], 'stack.json'))
        if not run_cfg:
            continue
        for k in range(1, retries + 1):
            out = os.path.join(ctx.out, 'retries', f'{rec["id"]}-{k}')
            ctx.log(f'retry {k} of {rec["name"]} (non-deterministic mode)')
            run = run_one(ctx.svc, dict(run_cfg, out=out), launch, int(st.get('domain_id', 0)),
                          f'{run_cfg.get("name", "run")} (retry {k})')
            if run.get('_job_state') == 'cancelled':
                raise KeyboardInterrupt
            spec = {'key': rec['key'], 'entry': rec.get('entry'), 'scenario': rec['group'],
                    'scenario_name': rec['group_name'], 'seed': rec['seed'], 'dir': out}
            att = stack_record(ST.result_row(spec, run))
            rec['attempts'].append(att['attempts'][0])
            if att['status'] == 'ok' and att['passed']:
                rec['flaky'] = True
                if count_as == 'pass':
                    keep = {x: rec[x] for x in ('attempts', 'flaky', 'id', 'pair')}
                    rec.update(att, **keep)
                break
    ruleset = doc.get('rules') or None
    for rec in records:                          # every counted run checked against the rules (retries too)
        if rec['status'] != 'ok' or rec['metrics'].get('rule_violations') is not None:
            continue
        try:
            res = RU.check_run(rec['dir'], ruleset)
        except (RU.RulesError, OSError, ValueError, KeyError):
            continue
        from .provenance import write_json
        write_json(os.path.join(rec['dir'], 'rules.json'), res)
        rec['metrics'].update(rule_violations=res['totals']['violations'], distance_km=res['distance_km'])
    rule_results = [r for r in (_read(os.path.join(x['dir'], 'rules.json')) for x in records if x['status'] == 'ok')
                    if r]
    try:
        rules = RU.aggregate(rule_results, 1000, int(cfg['seed'])) if rule_results else None
    except RU.RulesError as e:
        rules = {'error': str(e)}
    waits = []
    for rec in records:
        if rec['status'] == 'ok':
            waits += _wait_ms(rec['dir'])
    det, why = determinism(doc.get('motion', 'kinematic'), int(doc.get('traffic_vehicles', 0)), st['driver'],
                           (cfg['subject'].get('stack') or {}).get('deterministic'))
    return {'kind': 'stack', 'records': records, 'metrics': STACK_METRICS, 'rules': rules,
            'latency': pooled_latency(waits, 'the stack\'s control wait through Falcon\'s bridge, per step, pooled '
                                             'over the runs (frames.jsonl)'),
            'artifacts': {'suite': suite_dir}, 'deterministic': det, 'not_deterministic_because': why,
            'describe': {'suite': doc.get('name'), 'driver': st['driver'], 'launch': launch,
                         'motion': doc.get('motion'), 'traffic_vehicles': doc.get('traffic_vehicles'),
                         'world': doc.get('world'), 'rig': doc.get('rig'),
                         'image': (cfg['subject'].get('stack') or {}).get('image')}}


# ========================================================================================== subjects: the policy
class _TimedPolicy:
    """A policy whose every decision is timed (perf_counter, ms) — this machine's decision latency."""

    def __init__(self, pol):
        self.pol, self.path, self.times = pol, getattr(pol, 'path', None), []

    def act(self, obs, names=None):
        t = time.perf_counter()
        a = self.pol.act(obs, names) if names is not None else self.pol.act(obs)
        self.times.append((time.perf_counter() - t) * 1000.0)
        return a


def route_record(r: dict, rp: dict) -> dict:
    """A run record from a Leaderboard route row (core.leaderboard.score's per_route)."""
    from .leaderboard import PENALTY
    pen = sum(int(v) for k, v in (r.get('infraction_counts') or {}).items() if k in PENALTY)
    min_rc = float(rp.get('min_route_completion', 0.9))
    max_inf = int(rp.get('max_penalised_infractions', 0))
    passed = r['route_completion'] >= min_rc and pen <= max_inf
    why = []
    if r['route_completion'] < min_rc:
        why.append(f'route completion {r["route_completion"]:.3f} < {min_rc:g}')
    if pen > max_inf:
        why.append(f'{pen} penalised infractions > {max_inf}: ' + ', '.join(
            f'{k} {v}' for k, v in sorted((r.get('infraction_counts') or {}).items()) if k in PENALTY and v))
    km = (r.get('distance_m') or 0.0) / 1000.0
    return {'schema': RUN_SCHEMA, 'id': f'{r["scenario"]}-{r["seed"]}',
            'pair': pair_key(r['scenario'], r['seed'], None),
            'group': r['scenario'], 'group_name': r['scenario'], 'seed': r['seed'], 'values': None,
            'name': f'{r["scenario"]} seed {r["seed"]}', 'status': 'ok', 'passed': passed, 'error': None,
            'failed_checks': why,
            'metrics': {'passed': passed, 'driving_score': r['driving_score'],
                        'route_completion': r['route_completion'],
                        'infraction_penalty': r['infraction_penalty'], 'penalised_infractions': pen,
                        'rule_violations': r.get('rule_violations'),
                        'distance_km': km if r.get('rule_violations') is not None else None},
            'time_s': None, 'dir': None, 'attempts': [{'status': 'ok', 'passed': passed}], 'flaky': False}


def run_policy(ctx) -> dict:
    """The policy over its routes (core.leaderboard.evaluate) into OUT/leaderboard; a route passes when its route
    completion reaches route_pass.min_route_completion with no more penalised infractions than allowed."""
    from . import leaderboard as LB
    from .gymenv import FalconDriveEnv, gym_config
    from .policy import load_policy
    cfg, proj, svc = ctx.cfg, ctx.project, ctx.svc
    sub = cfg['subject']
    pol = load_policy(sub['policy'], sub.get('device', 'cpu'))
    timed = _TimedPolicy(pol)
    rt = sub.get('routes') or {}
    routes = LB.route_set(rt.get('scenarios') or ['lead_brake'], int(rt.get('seeds_per', 3)), int(rt.get('free', 0)),
                          int(rt.get('seed', 0)))
    world = sub.get('world') or proj.doc['worlds'][0]['id']
    motion, traffic = sub.get('motion', 'kinematic'), int(sub.get('traffic_vehicles', 0))
    bev = pol.doc.get('bev') if pol.uses_bev else None
    cam = pol.doc.get('camera') if getattr(pol, 'uses_camera', False) else None

    def factory(base):
        def make(scn):
            kw = {'max_episode_s': float(sub['max_episode_s'])} if sub.get('max_episode_s') else {}
            c = gym_config(proj, world, os.path.join(base, 'env', scn or 'free'), svc.port, svc.tm_port,
                           rig_id=sub.get('rig') or None, seed=int(cfg['seed']), scenario=scn, motion=motion,
                           action=pol.action['kind'], bev=bev, camera=cam, fps=float(sub.get('fps', 20.0)),
                           traffic_vehicles=traffic, log_actions=True, workspace=proj.workspace, **kw)
            return FalconDriveEnv(c, svc)
        return make
    lb_dir = os.path.join(ctx.out, 'leaderboard')
    doc = LB.evaluate(timed, factory(lb_dir), routes, lb_dir, 1000, lambda f, t: ctx.log(f'routes: {t}'))
    rp = sub.get('route_pass') or {}
    by = {(r['scenario'], r['seed']): route_record(r, rp) for r in doc['per_route']}
    records = []
    for r in routes:
        k = (r['scenario'] or 'free', r['seed'])
        if k in by:
            records.append(by[k])
        else:
            err = next((f['error'] for f in doc.get('failed') or () if (f['scenario'] or 'free') == k[0]), 'not run')
            records.append({'schema': RUN_SCHEMA, 'id': f'{k[0]}-{k[1]}', 'pair': pair_key(k[0], k[1], None),
                            'group': k[0], 'group_name': k[0], 'seed': k[1], 'values': None,
                            'name': f'{k[0]} seed {k[1]}', 'status': 'not finished', 'passed': None, 'error': err,
                            'failed_checks': [], 'metrics': {}, 'time_s': None, 'dir': None,
                            'attempts': [{'status': 'not finished', 'error': err}], 'flaky': False})
    retries, count_as = int(cfg['flaky']['retries']), cfg['flaky']['count_as']
    for k in range(1, retries + 1):
        todo = [r for r in records if not (r['status'] == 'ok' and r['passed']) and not r.get('_settled')]
        if not todo:
            break
        rdir = os.path.join(ctx.out, 'retries', f'leaderboard-{k}')
        ctx.log(f'retry {k}: {len(todo)} routes (non-deterministic mode)')
        again = LB.evaluate(timed, factory(rdir), [{'scenario': None if r['group'] == 'free' else r['group'],
                                                    'seed': r['seed']} for r in todo], rdir, 200)
        got = {(x['scenario'], x['seed']): route_record(x, rp) for x in again['per_route']}
        for r in todo:
            att = got.get((r['group'], r['seed']))
            r['attempts'].append({'status': 'ok' if att else 'not finished', 'passed': att and att['passed'],
                                  'dir': rdir})
            if att and att['passed']:
                r['flaky'] = r['_settled'] = True
                if count_as == 'pass':
                    keep = {x: r[x] for x in ('attempts', 'flaky', 'id', 'pair', '_settled')}
                    r.update(att, **keep)
    for r in records:
        r.pop('_settled', None)
    det, why = determinism(motion, traffic)
    return {'kind': 'policy', 'records': records, 'metrics': POLICY_METRICS, 'rules': doc.get('rules'),
            'latency': pooled_latency(timed.times, 'the policy\'s decision time per step on this machine (act(), '
                                                   'pooled over the routes)'),
            'artifacts': {'leaderboard': lb_dir}, 'deterministic': det, 'not_deterministic_because': why,
            'leaderboard_metrics': doc.get('metrics'),
            'describe': {'policy': sub['policy'], 'routes': len(routes), 'world': world, 'motion': motion,
                         'traffic_vehicles': traffic, 'not_scored': doc.get('not_scored')}}


# =========================================================================================== subjects: the model
def _wait_job(svc, job):
    from .jobs import FINAL
    while job.state not in FINAL:
        svc.queue.poll()
        time.sleep(0.1)
    if job.state == 'cancelled':
        raise KeyboardInterrupt


def record_drives(ctx, rec: dict) -> list:
    """Drives for a model subject, recorded now into OUT/drives/drive-<world>-seed<N> (stable names, so a later run's
    frames carry the same ids); background traffic off unless asked (then not repeatable)."""
    from .drive import drive_config
    proj, svc = ctx.project, ctx.svc
    world = rec.get('world') or proj.doc['worlds'][0]['id']
    rig = rec.get('rig') or proj.doc['rigs'][0]['id']
    out = []
    for seed in rec['seeds']:
        d = os.path.join(ctx.out, 'drives', f'drive-{world}-seed{int(seed)}')
        c = drive_config(proj, world, rig, d, svc.port, svc.tm_port, float(rec.get('seconds', 5.0)),
                         float(rec.get('fps', 20.0)), int(seed),
                         render_budget_mpx=float(svc.settings.get('render_budget_mpx')), workspace=proj.workspace)
        c['traffic'] = dict(c.get('traffic') or {}, vehicles=int(rec.get('traffic_vehicles', 0)))
        ctx.log(f'recording {os.path.basename(d)}')
        job = svc.submit_drive(c, title=f'CI drive {os.path.basename(d)}')
        _wait_job(svc, job)
        run = _read(os.path.join(d, 'run.json')) or {}
        if run.get('status') != 'ok':
            raise CIError(f'the drive {d} did not finish: {run.get("error") or job.state}')
        out.append(d)
    return out


def _drive_numbers(frames: list, kind: str) -> dict:
    from . import harness as H
    from . import metrics as M
    if kind == 'segmenter':
        return {'mIoU': H._seg_numbers(frames)['mIoU']}
    e = [H._frame_errors(f) for f in frames]
    return {'AP': M.coco_evaluate(frames)['AP'], 'missed': sum(len(x['missed']) for x in e),
            'false_positives': sum(len(x['false_positives']) for x in e)}


def run_model(ctx) -> dict:
    """The model on its drives (core.harness.evaluate) into OUT/evaluation; per drive its frames and numbers."""
    from . import harness as H
    cfg, proj = ctx.cfg, ctx.project
    sub = cfg['subject']
    m = proj.item('models', sub['model'])
    if m is None:
        raise CIError(f'no model {sub["model"]!r} in {proj.path}')
    drives = sub.get('drives') or record_drives(ctx, sub['record'])
    ev_dir = os.path.join(ctx.out, 'evaluation')
    n_boot = int(sub.get('n_boot', 1000))
    ctx.log(f'evaluating {m["id"]} on {len(drives)} drives')
    doc = H.evaluate(drives, m, ev_dir, proj.workspace, sub.get('camera'), int(sub.get('every', 1)), n_boot,
                     int(cfg['seed']), None, lambda f, t: None)
    frames = H.read_predictions(os.path.join(ev_dir, 'predictions.jsonl'))
    by = {}
    for f in frames:
        by.setdefault(f['drive'], []).append(f)
    records = []
    for d in drives:
        key = os.path.basename(os.path.normpath(d))
        fr = by.get(key, [])
        nums = _drive_numbers(fr, m['kind']) if fr else {}
        records.append({'schema': RUN_SCHEMA, 'id': key, 'pair': key, 'group': 'drives', 'group_name': 'drives',
                        'seed': None, 'values': None, 'name': key, 'status': 'ok', 'passed': None, 'error': None,
                        'failed_checks': [], 'metrics': dict(nums, frames=len(fr)), 'time_s': None, 'dir': d,
                        'attempts': [], 'flaky': False, 'informational': True})
    lat = doc.get('latency_ms') or {}
    latency = {'source': f'the model\'s inference per frame on this machine ({", ".join(doc.get("providers") or [])}:'
                         f' pre-processing, inference, decoding)', 'n': lat.get('n'), 'p50_ms': lat.get('median'),
               'p90_ms': lat.get('p90'), 'p99_ms': lat.get('p99'), 'mean_ms': lat.get('mean')} if lat else None
    return {'kind': 'model', 'records': records, 'metrics': {}, 'rules': None, 'latency': latency,
            'artifacts': {'evaluation': ev_dir}, 'deterministic': True, 'not_deterministic_because': [],
            'evaluation': {'table': doc['scores']['table'], 'unit': doc['scores']['unit'],
                           'n_units': doc['scores']['n_units'], 'note': doc['scores'].get('note'),
                           'n_frames': doc['n_frames'], 'synthetic_model': doc.get('synthetic_model')},
            'describe': {'model': {k: m.get(k) for k in ('id', 'name', 'kind', 'runtime', 'licence')},
                         'drives': drives, 'model_kind': m['kind']}}


# ======================================================================================================= baselines
def load_baseline(cfg: dict, kind: str):
    """(baseline or None, why there is none). A baseline: {'source', 'records' (stack, policy) or 'evaluation' (an
    evaluation folder: model), 'commit'}."""
    from . import stack as ST
    b = cfg.get('baseline') or {}
    k = b.get('kind', 'none')
    if k == 'none':
        return None, 'no baseline configured'
    if k == 'model':
        return {'source': f'model {b["model"]} on the same drives, now', 'kind': 'model', 'model': b['model'],
                'commit': None}, None
    if k == 'dir':
        d = b.get('dir')
    else:
        if not b.get('store'):
            return None, 'baseline.store is not set (an unset environment variable?)'
        d = os.path.join(b['store'], branch_dir(b['branch']))
    if not d or not os.path.isdir(d):
        return None, f'no baseline at {d}'
    res = _read(os.path.join(d, 'ci-results.json'))
    if res is not None:
        if (res.get('subject') or {}).get('kind') != kind:
            raise CIError(f'the baseline {d} tested a {(res.get("subject") or {}).get("kind")}, not a {kind}')
        out = {'source': d, 'kind': k, 'commit': res.get('commit'), 'verdict': res.get('verdict'),
               'created': (res.get('provenance') or {}).get('created')}
        if kind == 'model':
            ev = os.path.join(d, 'evaluation')
            if not os.path.isfile(os.path.join(ev, 'predictions.jsonl')):
                return None, f'the baseline {d} has no evaluation/predictions.jsonl'
            out['evaluation'] = ev
        else:
            p = os.path.join(d, 'ci-runs.jsonl')
            if not os.path.isfile(p):
                return None, f'the baseline {d} has no ci-runs.jsonl'
            with open(p, encoding='utf-8') as fh:
                out['records'] = [json.loads(x) for x in fh if x.strip()]
        return out, None
    if kind == 'stack' and os.path.isfile(os.path.join(d, 'results.jsonl')):
        run = _read(os.path.join(d, 'run.json')) or {}
        rows = ST.read_results(d)
        keys = run.get('runs') or [r['key'] for r in rows]
        by = {r['key']: r for r in rows}
        return {'source': d, 'kind': k, 'commit': None,
                'records': [stack_record(by[x]) for x in keys if x in by]}, None
    if kind == 'policy' and os.path.isfile(os.path.join(d, 'leaderboard.json')):
        lb = _read(os.path.join(d, 'leaderboard.json'))
        rp = cfg['subject'].get('route_pass') or {}
        return {'source': d, 'kind': k, 'commit': None, 'records': [route_record(r, rp) for r in lb['per_route']]}, None
    if kind == 'model' and os.path.isfile(os.path.join(d, 'predictions.jsonl')):
        return {'source': d, 'kind': k, 'commit': None, 'evaluation': d}, None
    what = {'stack': 'suite', 'policy': 'Leaderboard', 'model': 'evaluation'}[kind]
    raise CIError(f'{d} is not a Falcon CI output, nor a {what} folder')


def compare_records(base: list, cur: list, metrics: dict, names, n_boot: int = 1000, seed: int = 0) -> dict:
    """A paired bootstrap (core.metrics.paired_bootstrap; each pair of runs — the same scenario, seed and values —
    a unit) of every metric's change now − baseline; a regression is a change whose 95 % interval lies wholly on the
    metric's bad side, an improvement wholly on its good side."""
    from .metrics.uncertainty import paired_bootstrap
    bmap = {r['pair']: r for r in base if r.get('status') == 'ok'}
    cmap = {r['pair']: r for r in cur if r.get('status') == 'ok'}
    common = [p for p in cmap if p in bmap]
    pairs = [{'a': bmap[p]['metrics'], 'b': cmap[p]['metrics'], 'name': cmap[p]['name']} for p in common]
    out = {'pairs': len(pairs), 'only_baseline': sorted(bmap[p]['name'] for p in bmap if p not in cmap),
           'only_now': sorted(cmap[p]['name'] for p in cmap if p not in bmap), 'rows': [], 'regressions': [],
           'improvements': [], 'n_boot': n_boot, 'seed': seed, 'unit': 'a pair of runs (same scenario, seed, values)',
           'new_failures': sorted(cmap[p]['name'] for p in common if bmap[p].get('passed') and
                                  cmap[p].get('passed') is False),
           'fixed': sorted(cmap[p]['name'] for p in common if bmap[p].get('passed') is False and
                           cmap[p].get('passed')),
           'note': None}
    if not pairs:
        return out
    if len(pairs) < SMALL_N:
        out['note'] = (f'only {len(pairs)} pairs: a bootstrap over so few units has few distinct resamples, so its '
                       f'interval is not reliable; run {SMALL_N} or more')

    def fn(items, which):
        res = {}
        for name in names:
            field, _, how = metrics[name]
            if how == 'mean':
                v = [_f(it[which].get(field)) for it in items
                     if _f(it['a'].get(field)) is not None and _f(it['b'].get(field)) is not None]
                res[name] = sum(v) / len(v) if v else None
            else:
                sel = [it for it in items if all(_f(it[s].get('rule_violations')) is not None
                                                 and _f(it[s].get('distance_km')) for s in ('a', 'b'))]
                km = sum(float(it[which]['distance_km']) for it in sel)
                res[name] = sum(float(it[which]['rule_violations']) for it in sel) / km if km > 0 else None
        return res
    r = paired_bootstrap(fn, pairs, 'a', 'b', n=n_boot, seed=seed, unit='image')
    for name in names:
        x = (r.get('metrics') or {}).get(name)
        if not x:
            continue
        d = metrics[name][1]
        lo, hi = x.get('ci_low'), x.get('ci_high')
        verdict = 'no detectable change at 95 %'
        if lo is not None and hi is not None:
            if (d > 0 and hi < 0) or (d < 0 and lo > 0):
                verdict = 'regression'
            elif (d > 0 and lo > 0) or (d < 0 and hi < 0):
                verdict = 'improvement'
        elif x.get('diff') is None:
            verdict = 'not measured'
        out['rows'].append({'metric': name, 'baseline': x.get('a'), 'now': x.get('b'), 'diff': x.get('diff'),
                            'ci_low': lo, 'ci_high': hi, 'better': 'higher' if d > 0 else 'lower',
                            'verdict': verdict})
        if verdict == 'regression':
            out['regressions'].append(name)
        elif verdict == 'improvement':
            out['improvements'].append(name)
    return out


def compare_models(base_dir: str, cur_dir: str, kind: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """core.harness.compare (a paired bootstrap over drives) of two evaluations on the same frames — refused when the
    frames' ground truth differs (drives recorded again are not the same drives)."""
    from . import harness as H
    from .provenance import sha256_json
    fa = H.read_predictions(os.path.join(base_dir, 'predictions.jsonl'))
    fb = H.read_predictions(os.path.join(cur_dir, 'predictions.jsonl'))
    ga = {f['image_id']: sha256_json(f.get('gt', f.get('confusion'))) for f in fa} if kind != 'segmenter' else {}
    diff = [f['image_id'] for f in fb if f['image_id'] in ga and sha256_json(f.get('gt')) != ga[f['image_id']]]
    if diff:
        raise CIError(f'the baseline\'s frames are not these drives\' frames ({len(diff)} frames with different '
                      'ground truth, e.g. ' + diff[0] + '): pin the test drives (subject.drives) so both evaluations '
                      'score the same frames')
    res = H.compare(fa, fb, kind, n_boot, seed)
    rows = [dict(r, baseline=r['a'], now=r['b'], better='higher',
                 verdict='regression' if r['metric'] in res['regressions'] else
                 'improvement' if r['metric'] in res['improvements'] else 'no detectable change at 95 %')
            for r in res['rows']]
    note = None
    if res['unit'] == 'drive' and (res.get('n_units') or 0) < H.MIN_DRIVES:
        note = (f'only {res.get("n_units")} drives: the paired interval over so few drives is not reliable; pin '
                f'{H.MIN_DRIVES} or more')
    elif res['unit'] == 'image':
        note = 'one drive: the interval is over frames, which are not independent — it is too narrow'
    return {'pairs': res['n_frames'], 'unit': res['unit'], 'n_units': res.get('n_units'), 'rows': rows,
            'regressions': res['regressions'], 'improvements': res['improvements'], 'n_boot': n_boot, 'seed': seed,
            'note': note, 'only_baseline': [], 'only_now': [], 'new_failures': [], 'fixed': []}


# =========================================================================================================== gates
def _gate(gid, name, status, measured='', limit='', rule='', detail='', **extra) -> dict:
    return dict({'id': gid, 'name': name, 'status': status, 'measured': measured, 'limit': limit, 'rule': rule,
                 'detail': detail}, **extra)


def _fmt(v, dp=3):
    return '—' if v is None else (f'{v:.{dp}f}' if isinstance(v, float) else str(v))


def _ci(row):
    if row.get('value') is None:
        return '—'
    if row.get('ci_low') is None:
        return _fmt(row['value'])
    return f'{_fmt(row["value"])} [{_fmt(row["ci_low"])}, {_fmt(row["ci_high"])}]'


def evaluate_gates(cfg: dict, subj: dict, summary: dict, comparison: dict | None, baseline_why: str | None) -> list:
    from .stack import wilson
    gates, kind = cfg.get('gates') or {}, subj['kind']
    out = []
    pr = gates.get('pass_rate')
    if pr:
        rule = pr.get('rule', 'demonstrate')
        groups = [('overall', 'all runs', summary['overall'])]
        if pr.get('per_scenario'):
            groups += [(g, s.get('name', g), s) for g, s in summary['groups'].items()]
        for gid, label, s in groups:
            n, k = s['runs'], s['passed']
            lo, hi = wilson(k, n) if n else (None, None)
            ok, v = judge(k / n if n else None, lo, hi, float(pr['min']), 'min', rule)
            meas = f'{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}] (Wilson 95 %)' if n else 'no finished run'
            detail = ''
            if ok is False and rule == 'demonstrate':
                need = runs_to_demonstrate(float(pr['min']))
                detail = (f'the interval\'s lower bound {lo:.3f} is below {pr["min"]:g}; with every run passing, '
                          f'{need} runs are needed to demonstrate it')
            elif ok is False:
                detail = f'{"upper bound" if rule == "refute" else "estimate"} {v:.3f} < {pr["min"]:g}'
            out.append(_gate(f'pass_rate.{gid}', f'pass rate ({label}) ≥ {pr["min"]:g}',
                             'error' if ok is None else 'passed' if ok else 'failed', meas, f'≥ {pr["min"]:g}', rule,
                             detail or ('cannot be judged without finished runs' if ok is None else '')))
    for g in gates.get('metrics') or ():
        name, rule = g['metric'], g.get('rule', 'estimate')
        side, limit = ('min', g['min']) if 'min' in g else ('max', g['max'])
        if kind == 'model':
            row = next((r for r in subj['evaluation']['table'] if r['metric'] == name), None)
        else:
            row = (summary['overall']['metrics'].get(name) or None)
        if row is None:
            out.append(_gate(f'metric.{name}', f'{name} {"≥" if side == "min" else "≤"} {limit:g}', 'error', '',
                             f'{limit:g}', rule, f'{name} is not a number this run reports'))
            continue
        ok, v = judge(_f(row.get('value')), _f(row.get('ci_low')), _f(row.get('ci_high')), float(limit), side, rule)
        out.append(_gate(f'metric.{name}', f'{name} {"≥" if side == "min" else "≤"} {limit:g}',
                         'error' if ok is None else 'passed' if ok else 'failed', _ci(row),
                         f'{"≥" if side == "min" else "≤"} {limit:g}', rule,
                         'no interval to judge with (too few runs?); use rule estimate or more runs' if ok is None
                         else ''))
    reg = gates.get('regressions')
    if reg is not None and reg.get('enabled', True):
        miss = reg.get('on_missing_baseline', 'skip')
        allow = set(reg.get('allow') or ())
        if comparison is None or not comparison.get('pairs'):
            why = baseline_why or (comparison or {}).get('error') or 'no runs in common with the baseline'
            out.append(_gate('regressions', 'no regressions against the baseline',
                             'error' if miss == 'error' else 'skipped', '', '', 'paired bootstrap, 95 %', why))
        else:
            found = [m for m in comparison['regressions'] if m not in allow]
            out.append(_gate('regressions', 'no regressions against the baseline', 'failed' if found else 'passed',
                             f'{len(found)} regressions' + (f': {", ".join(found)}' if found else '')
                             + (f' over {comparison["pairs"]} shared frames ({comparison.get("n_units")} '
                                f'{comparison.get("unit")}s)' if kind == 'model' else
                                f' over {comparison["pairs"]} pairs of runs'), 'none', 'paired bootstrap, 95 %',
                             '; '.join(x for x in (comparison.get('note'),
                                                   f'allowed: {", ".join(sorted(allow))}' if allow else '') if x)))
    rv = gates.get('rule_violations')
    if rv:
        rule = rv.get('rule', 'estimate')
        row = summary['overall']['metrics'].get('rule_violations_per_km') or {}
        if row.get('value') is None:
            out.append(_gate('rule_violations', 'traffic-rule violations per km', 'error', '', '', rule,
                             'no run could be checked against the traffic rules'))
        else:
            if 'max_per_km' in rv:
                ok, v = judge(row['value'], _f(row.get('ci_low')), _f(row.get('ci_high')), float(rv['max_per_km']),
                              'max', rule)
                out.append(_gate('rule_violations', f'traffic-rule violations ≤ {rv["max_per_km"]:g} per km',
                                 'error' if ok is None else 'passed' if ok else 'failed',
                                 f'{_ci(row)} per km ({row.get("violations"):g} over {row.get("distance_km"):.3f} km)',
                                 f'≤ {rv["max_per_km"]:g}', rule,
                                 'no interval to judge with; use rule estimate or more runs' if ok is None else ''))
            agg = subj.get('rules') or {}
            for rid, lim in sorted((rv.get('rules') or {}).items()):
                r = next((x for x in agg.get('rules') or () if x['id'] == rid), None)
                if r is None or r.get('per_km') is None:
                    out.append(_gate(f'rule.{rid}', f'{rid} ≤ {lim:g} per km', 'error', '', f'≤ {lim:g}', rule,
                                     f'no run could evaluate rule {rid}'))
                    continue
                c = r.get('per_km_ci95') or [None, None]
                ok, v = judge(r['per_km'], c[0], c[1], float(lim), 'max', rule)
                out.append(_gate(f'rule.{rid}', f'{r.get("title", rid)} ≤ {lim:g} per km',
                                 'error' if ok is None else 'passed' if ok else 'failed',
                                 _ci({'value': r['per_km'], 'ci_low': c[0], 'ci_high': c[1]}), f'≤ {lim:g}', rule,
                                 'no interval to judge with' if ok is None else ''))
    lat = gates.get('latency')
    if lat:
        L = subj.get('latency')
        if not L:
            why = ('no latency measured: only a ROS 2 stack (driver ros2) has a control wait' if kind == 'stack'
                   else 'no latency measured')
            out.append(_gate('latency', 'latency budget', 'error', '', '', 'pooled percentiles', why))
        else:
            for k in LATENCY_KEYS:
                if k not in lat:
                    continue
                v = L.get(k)
                if v is None:
                    out.append(_gate(f'latency.{k}', f'latency {k[:-3]} ≤ {lat[k]:g} ms', 'error', '',
                                     f'≤ {lat[k]:g} ms', 'pooled percentiles', f'{k} is not measured here'))
                    continue
                out.append(_gate(f'latency.{k}', f'latency {k[:-3]} ≤ {lat[k]:g} ms',
                                 'passed' if v <= lat[k] else 'failed', f'{v:.2f} ms over {L.get("n")} samples',
                                 f'≤ {lat[k]:g} ms', 'pooled percentile (this machine)', L.get('source', '')))
    if 'max_missed_deadlines' in gates:
        recs = [r for r in subj['records'] if r['status'] == 'ok']
        n = sum(int(r['metrics'].get('missed_deadlines') or 0) for r in recs)
        lim = int(gates['max_missed_deadlines'])
        out.append(_gate('missed_deadlines', f'missed control deadlines ≤ {lim}', 'passed' if n <= lim else 'failed',
                         f'{n} over {len(recs)} runs', f'≤ {lim}', 'total'))
    return out


def group_summaries(records: list, metrics: dict, seed: int) -> dict:
    by = {}
    for r in records:
        by.setdefault(r['group'], []).append(r)
    return {g: dict(summarise_records(rs, metrics, seed), name=rs[0].get('group_name', g)) for g, rs in by.items()}


# ========================================================================================================== outputs
def _xml_name(s) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s)).strip('_') or 'x'


def junit_xml(res: dict) -> str:
    """JUnit XML: a testsuite of the runs (a testcase per scenario run; a run that did not finish is an error, one that
    failed its criteria a failure, a model's drives informational) and one of the gates (failed: a failure; could not
    be judged: an error; skipped: skipped)."""
    import xml.etree.ElementTree as ET
    name = res.get('name', 'falcon-ci')
    recs, gates = res.get('runs') or [], res.get('gates') or []
    root = ET.Element('testsuites', name=f'Falcon CI: {name}')

    def suite(title, cases):
        ts = ET.SubElement(root, 'testsuite', name=title, timestamp=(res.get('started') or '').rstrip('Z'),
                           hostname=_xml_name(os.uname().nodename) if hasattr(os, 'uname') else 'host')
        props = ET.SubElement(ts, 'properties')
        cm = res.get('commit') or {}
        for k, v in (('commit', cm.get('sha')), ('branch', cm.get('branch')), ('subject', res.get('subject', {})
                                                                                  .get('kind')),
                     ('carla', (res.get('carla') or {}).get('mode')), ('deterministic', res.get('deterministic')),
                     ('falcon', res.get('app_version'))):
            if v is not None:
                ET.SubElement(props, 'property', name=k, value=str(v))
        n = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        total = 0.0
        for cls, case_name, t, state, msg, text, out in cases:
            tc = ET.SubElement(ts, 'testcase', classname=cls, name=case_name, time=f'{t or 0.0:.3f}')
            total += t or 0.0
            n['tests'] += 1
            if state == 'failure':
                n['failures'] += 1
                ET.SubElement(tc, 'failure', message=msg[:1000], type='GateFailed' if cls.endswith('.gates')
                              else 'CriteriaFailed').text = text
            elif state == 'error':
                n['errors'] += 1
                ET.SubElement(tc, 'error', message=msg[:1000], type='NotJudged' if cls.endswith('.gates')
                              else 'NotFinished').text = text
            elif state == 'skipped':
                n['skipped'] += 1
                ET.SubElement(tc, 'skipped', message=msg[:1000])
            if out:
                ET.SubElement(tc, 'system-out').text = out
        for k, v in n.items():
            ts.set(k, str(v))
        ts.set('time', f'{total:.3f}')
        return n, total
    base = f'falcon.{_xml_name(name)}'
    cases = []
    for r in recs:
        m = ', '.join(f'{k} {_fmt(v if isinstance(v, int) else _f(v))}' for k, v in (r.get('metrics') or {}).items()
                      if _f(v) is not None and not isinstance(v, bool))
        out = f'{r.get("dir") or ""}\n{m}'.strip()
        if r.get('flaky'):
            out += (f'\nFLAKY: failed at first, passed on attempt {len(r["attempts"])} (non-deterministic mode); '
                    f'attempts: ' + json.dumps(r['attempts']))
        if r.get('informational'):
            state, msg = 'passed', ''
            out += '\ninformational: a drive is not pass/fail; the gates judge the model'
        elif r['status'] != 'ok':
            state, msg = 'error', f'did not finish: {r.get("error") or "no result"}'
        elif r.get('passed') is False:
            state, msg = 'failure', '; '.join(r.get('failed_checks') or ()) or 'failed its criteria'
        else:
            state, msg = 'passed', ''
        cases.append((f'{base}.{_xml_name(r["group"])}', r['name'], r.get('time_s'), state, msg, msg, out))
    if res.get('error'):
        cases.append((f'{base}.run', 'falcon ci run', 0.0, 'error', res['error'], res.get('traceback') or res['error'],
                      ''))
    n1, t1 = suite(f'{name}: runs', cases)
    gcases = []
    for g in gates:
        state = {'failed': 'failure', 'error': 'error', 'skipped': 'skipped'}.get(g['status'], 'passed')
        text = f'measured {g.get("measured")}; limit {g.get("limit")}; rule {g.get("rule")}. {g.get("detail")}'.strip()
        gcases.append((f'{base}.gates', g['name'], 0.0, state, g.get('detail') or g.get('measured') or g['status'],
                       text, text))
    n2, t2 = suite(f'{name}: gates', gcases)
    for k in n1:
        root.set(k, str(n1[k] + n2[k]))
    root.set('time', f'{t1 + t2:.3f}')
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding='unicode') + '\n'


def _md_cell(s) -> str:
    return str('' if s is None else s).replace('|', '\\|').replace('\n', ' ')


def markdown(res: dict, max_rows: int = 30) -> str:
    """The summary for $GITHUB_STEP_SUMMARY and a pull-request comment (GitHub-flavoured Markdown, no images)."""
    v = res.get('verdict', 'error')
    cm = res.get('commit') or {}
    subj = res.get('subject') or {}
    car = res.get('carla') or {}
    head = {'passed': 'PASSED', 'failed': 'FAILED (a gate failed)', 'error': 'ERROR (could not be judged)',
            'cancelled': 'CANCELLED'}.get(v, v.upper())
    L = [f'## Falcon CI: {_md_cell(res.get("name"))} — {head}', '']
    bits = []
    if cm.get('sha'):
        bits.append(f'commit `{cm["sha"][:12]}`' + (f' on `{cm["branch"]}`' if cm.get('branch') else '')
                    + (' (uncommitted changes)' if cm.get('dirty') else ''))
    bits.append(f'{subj.get("kind")}: {_md_cell(subj.get("label"))}')
    bits.append('fake CARLA (a smoke test of the integration: its numbers are not evidence about the stack)'
                if car.get('mode') == 'fake' else 'Falcon\'s CARLA 0.9.16 server')
    det = res.get('deterministic')
    bits.append('deterministic mode (kinematic, no traffic: runs repeat bit for bit)' if det else
                f'NOT deterministic ({", ".join(res.get("not_deterministic_because") or [])})')
    L += ['- ' + b for b in bits]
    if res.get('error'):
        L += ['', f'**Error:** {_md_cell(res["error"])}']
    gates = res.get('gates') or []
    if gates:
        L += ['', '### Gates', '', '| gate | result | measured | limit | rule |', '|---|---|---|---|---|']
        for g in gates:
            L.append(f'| {_md_cell(g["name"])} | **{g["status"].upper()}** | {_md_cell(g.get("measured"))} | '
                     f'{_md_cell(g.get("limit"))} | {_md_cell(g.get("rule"))} |')
        notes = [f'- {_md_cell(g["name"])}: {_md_cell(g["detail"])}' for g in gates
                 if g.get('detail') and g['status'] != 'passed']
        if notes:
            L += [''] + notes
    s = res.get('summary') or {}
    groups = s.get('groups') or {}
    if groups and subj.get('kind') != 'model':
        L += ['', '### Runs', '', '| scenario | runs | passed | pass rate [95 % Wilson] | flaky | not finished |',
              '|---|---|---|---|---|---|']
        for g, x in list(groups.items()) + [('overall', dict(s['overall'], name='**all**'))]:
            mr = (x.get('metrics') or {}).get('pass_rate') or {}
            L.append(f'| {_md_cell(x.get("name", g))} | {x["runs"]} | {x["passed"]} | {_ci(mr)} | {x.get("flaky", 0)} '
                     f'| {x.get("not_finished", 0)} |')
    if subj.get('kind') == 'model' and res.get('evaluation'):
        ev = res['evaluation']
        L += ['', f'### Scores ({ev["n_frames"]} frames, intervals over {ev["unit"]}s)', '',
              '| metric | value [95 %] |', '|---|---|']
        for r in ev['table']:
            if r['metric'] in MODEL_REGRESSION_DEFAULT:
                L.append(f'| {r["metric"]} | {_ci(r)} |')
        if ev.get('note'):
            L += ['', f'_{_md_cell(ev["note"])}_']
        if ev.get('synthetic_model'):
            L += ['', '_SYNTHETIC model (reads the ground truth): its numbers say nothing about a real detector._']
    cmp_ = res.get('comparison')
    base = res.get('baseline') or {}
    if cmp_ and cmp_.get('rows'):
        bc = (base.get('commit') or {}).get('sha')
        what = 'shared frames' if subj.get('kind') == 'model' else 'pairs of runs'
        L += ['', f'### Against the baseline ({_md_cell(base.get("label"))}'
              + (f', commit `{bc[:12]}`' if bc else '') + f'; {cmp_["pairs"]} {what})', '',
              '| metric | baseline | now | change [95 %] | verdict |', '|---|---|---|---|---|']
        for r in cmp_['rows']:
            ch = '—' if r.get('diff') is None else (f'{r["diff"]:+.3f}' + (
                f' [{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]' if r.get('ci_low') is not None else ''))
            L.append(f'| {_md_cell(r["metric"])} ({r["better"]} is better) | {_fmt(_f(r.get("baseline")))} | '
                     f'{_fmt(_f(r.get("now")))} | {ch} | {r["verdict"]} |')
        if cmp_.get('new_failures'):
            L += ['', 'Runs that passed in the baseline and fail now: ' + ', '.join(
                f'`{x}`' for x in cmp_['new_failures'][:max_rows])]
        if cmp_.get('note'):
            L += ['', f'_{_md_cell(cmp_["note"])}_']
    elif base.get('why'):
        L += ['', f'_No comparison: {_md_cell(base["why"])}._']
    fails = [r for r in res.get('runs') or () if r['status'] != 'ok' or r.get('passed') is False]
    if fails:
        L += ['', f'### Failing runs ({len(fails)})', '', '| run | result | why |', '|---|---|---|']
        for r in fails[:max_rows]:
            why = r.get('error') if r['status'] != 'ok' else '; '.join(r.get('failed_checks') or ())
            L.append(f'| {_md_cell(r["name"])} | {"did not finish" if r["status"] != "ok" else "failed"} | '
                     f'{_md_cell(why)[:300]} |')
        if len(fails) > max_rows:
            L.append(f'| … {len(fails) - max_rows} more in ci-results.json | | |')
    flaky = [r['name'] for r in res.get('runs') or () if r.get('flaky')]
    if flaky:
        L += ['', f'Flaky (failed, then passed on a retry; counted as {res.get("flaky_count_as")}): '
              + ', '.join(f'`{x}`' for x in flaky[:max_rows])]
    lat = res.get('latency')
    if lat:
        L += ['', f'Latency ({_md_cell(lat.get("source"))}): ' + ', '.join(
            f'{k[:-3]} {lat[k]:.2f} ms' for k in LATENCY_KEYS if lat.get(k) is not None) + f'; n = {lat.get("n")}.']
    L += ['', '<details><summary>Provenance</summary>', '',
          f'- Falcon Autonomy {res.get("app_version")}, config sha256 `{(res.get("config_sha256") or "")[:16]}`',
          f'- started {res.get("started")}, finished {res.get("finished")}' + (
              f' ({res["wall_s"]:.0f} s)' if res.get('wall_s') is not None else ''),
          f'- commit source: {_md_cell(cm.get("source") or "not known (not a git repository; no CI variables)")}'
          + (f'; CI run: {cm["run_url"]}' if cm.get('run_url') else ''),
          '- Not a certification claim: simulation evidence within the scenarios run (docs/ci.md, limits).',
          '', '</details>', '']
    return '\n'.join(L)


def report(src: str):
    """core.reports' "ci" page: the verdict, gates, runs, comparison and provenance of a CI output folder."""
    from . import reports as R
    p = src if src.endswith('.json') else os.path.join(src, 'ci-results.json')
    pg = R.Page('ci', 'Falcon CI', os.path.dirname(os.path.abspath(p)))
    res = pg.json(p)
    if res.get('schema') != CI_SCHEMA:
        raise R.ReportError(f'{p} is not a CI result ({CI_SCHEMA})')
    pg.title = f'Falcon CI: {res.get("name")} — {res.get("verdict", "?").upper()}'
    cm = res.get('commit') or {}
    subj = res.get('subject') or {}
    pg.add(R.kv([('verdict', f'<b class="{"good" if res.get("verdict") == "passed" else "crit"}">'
                             f'{R.esc(res.get("verdict"))}</b> (exit code {R.esc(res.get("exit_code"))})'),
                 ('subject', R.esc(f'{subj.get("kind")}: {subj.get("label")}')),
                 ('commit', f'<span class="mono">{R.esc(cm.get("sha"))}</span> {R.esc(cm.get("branch") or "")}'
                            f'{" (uncommitted changes)" if cm.get("dirty") else ""} — {R.esc(cm.get("source"))}'),
                 ('CARLA', R.esc('fake carla (smoke: not evidence about the stack)' if (res.get('carla') or {})
                                 .get('mode') == 'fake' else 'Falcon\'s own CARLA 0.9.16 server')),
                 ('deterministic', R.esc('yes (kinematic, no traffic)' if res.get('deterministic') else
                                         'no: ' + ', '.join(res.get('not_deterministic_because') or []))),
                 ('started', R.esc(res.get('started'))), ('wall time', R.secs(res.get('wall_s')))]))
    if res.get('error'):
        pg.note(f'Error: {res["error"]}')
    if (res.get('carla') or {}).get('mode') == 'fake':
        pg.note('These runs used the fake carla: they test that the stack, the bridge and the gates work together, '
                'not how the stack drives.')
    pg.h2('Gates')
    cls = {'passed': 'good', 'failed': 'crit', 'error': 'crit', 'skipped': 'muted'}
    pg.add(R.table(['gate', 'result', 'measured', 'limit', 'rule', 'detail'], [
        [R.esc(g['name']), f'<span class="{cls.get(g["status"], "")}">{R.esc(g["status"])}</span>',
         R.esc(g.get('measured')), R.esc(g.get('limit')), R.esc(g.get('rule')), R.esc(g.get('detail'))]
        for g in res.get('gates') or ()]))
    s = res.get('summary') or {}
    if s.get('groups') and subj.get('kind') != 'model':
        pg.h2('Pass rate by scenario')
        rows = [{'label': x.get('name', g), 'before': (x.get('metrics') or {}).get('pass_rate') or {}}
                for g, x in s['groups'].items()]
        rows.append({'label': 'all runs', 'before': (s['overall'].get('metrics') or {}).get('pass_rate') or {}})
        pg.add(R.legend('pass rate (Wilson 95 %)', None), R.interval_chart(rows, title='pass rate',
                                                                           names=('pass rate', None)))
        pg.add(R.table(['metric', 'value [95 %]', 'runs', 'interval'], [
            [R.esc(k), R.esc(_ci(v)), R.esc(v.get('n')), R.esc(v.get('interval', ''))]
            for k, v in (s['overall'].get('metrics') or {}).items()], numeric=(2,)))
    if res.get('evaluation'):
        ev = res['evaluation']
        pg.h2('Scores')
        pg.note(ev.get('note'))
        pg.add(R.table(['metric', 'value', '95 % interval'], [
            [R.esc(r['metric']), R.num(r['value']), f'[{R.num(r["ci_low"])}, {R.num(r["ci_high"])}]']
            for r in ev['table'] if r['metric'] in MODEL_REGRESSION_DEFAULT], numeric=(1, 2)))
    cmp_ = res.get('comparison')
    if cmp_ and cmp_.get('rows'):
        pg.h2(f'Against the baseline ({(res.get("baseline") or {}).get("label")})')
        pg.note(cmp_.get('note'))
        pg.add(R.table(['metric', 'better', 'baseline', 'now', 'change', '95 % interval', 'verdict'], [
            [R.esc(r['metric']), R.esc(r['better']), R.num(_f(r.get('baseline'))), R.num(_f(r.get('now'))),
             R.num(_f(r.get('diff')), signed=True), f'[{R.num(_f(r.get("ci_low")), signed=True)}, '
                                                   f'{R.num(_f(r.get("ci_high")), signed=True)}]',
             f'<span class="{VERDICT_CLASS.get(r["verdict"], "muted")}">{R.esc(r["verdict"])}</span>']
            for r in cmp_['rows']], numeric=(2, 3, 4, 5)))
    elif (res.get('baseline') or {}).get('why'):
        pg.note(f'No comparison: {res["baseline"]["why"]}')
    runs = res.get('runs') or []
    if runs:
        pg.h2(f'Runs ({len(runs)})')
        pg.add(R.table(['run', 'result', 'why', 'folder'], [
            [R.esc(r['name']), R.esc('did not finish' if r['status'] != 'ok' else 'informational'
                                     if r.get('informational') else 'passed' + (' (flaky)' if r.get('flaky') else '')
                                     if r.get('passed') else 'failed'),
             R.esc(r.get('error') if r['status'] != 'ok' else '; '.join(r.get('failed_checks') or ())),
             f'<span class="mono">{R.esc(R.tilde(r.get("dir") or ""))}</span>'] for r in runs]))
    lat = res.get('latency')
    if lat:
        pg.h2('Latency')
        pg.add(R.kv([('source', R.esc(lat.get('source'))), ('samples', R.esc(lat.get('n')))] +
                    [(k[:-3], R.esc(f'{lat[k]:.2f} ms')) for k in LATENCY_KEYS if lat.get(k) is not None]))
    pg.h2('Limits')
    pg.add('<ul>' + ''.join(f'<li>{R.esc(x)}</li>' for x in LIMITS) + '</ul>')
    pg.provenance(res.get('provenance'), 'the CI run')
    return pg


VERDICT_CLASS = {'regression': 'crit', 'improvement': 'good'}
LIMITS = ('Simulation evidence within the scenarios, seeds and parameter values run — not a certification claim, and '
          'not a substitute for the safety case\'s own limits (docs/stack-testing.md).',
          'Intervals are 95 %: with several gates and metrics, one of them can fail by chance about once in twenty '
          'per metric when nothing changed (the refute rule and the paired bootstrap keep that to the bad side).',
          'Latency is this machine\'s, under its load at the time; budgets on a shared runner are noisy.',
          'The fake carla tests the integration (the bridge, the topics, the controls, the gates), not driving.')


# ======================================================================================================= the run
class Ctx:
    def __init__(self, cfg, out, svc=None, project=None, suite=None, log=None):
        self.cfg, self.out, self.svc, self.project, self.suite = cfg, out, svc, project, suite
        self.log = log or (lambda text: None)


def prepare_out(out: str, force: bool = False) -> str:
    """A fresh output folder: one holding a previous CI run's outputs is cleared of exactly those (with force), never
    reused (a suite there would resume runs of another commit); a folder with anything else is refused."""
    if os.path.isdir(out) and os.listdir(out):
        ours = set(OUT_FILES) | set(OUT_DIRS)
        prev = (_read(os.path.join(out, 'run.json')) or {}).get('worker') == 'ci' or \
            os.path.isfile(os.path.join(out, 'ci-results.json'))
        others = [x for x in os.listdir(out) if x not in ours]
        if not prev or others:
            raise CIError(f'{out} is not empty' + (f' (it holds {", ".join(sorted(others)[:5])})' if others else '')
                          + ': give an empty or new folder')
        if not force:
            raise CIError(f'{out} holds a previous CI run: give --force to replace it, or a new folder')
        for x in os.listdir(out):
            p = os.path.join(out, x)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
    os.makedirs(out, exist_ok=True)
    return out


def _project(cfg, out):
    from .project import Project, ProjectError
    if cfg.get('project'):
        try:
            return Project.load(cfg['project'])
        except (OSError, ProjectError) as e:
            raise CIError(f'project {cfg["project"]}: {e}') from None
    p = Project.new(f'{cfg["name"]} (CI)')
    p.save(os.path.join(out, 'project', 'ci.falcon.json'))
    return p


def _service(cfg, out, workspace):
    from .jobs import JobQueue, SubprocessLauncher
    from .settings import Settings
    from ..sim.service import CarlaService
    fake = cfg['carla'].get('fake_carla', '') if cfg['carla']['mode'] == 'fake' else ''
    lock_dir = os.path.join(out, '.locks') if fake else None      # no server: never wait on the machine's CARLA lock
    return CarlaService(Settings(), JobQueue(SubprocessLauncher(), lock_dir), workspace, fake_carla=fake)


def subject_label(cfg: dict, suite: dict | None) -> str:
    sub = cfg['subject']
    if sub['kind'] == 'stack':
        st = (suite or {}).get('stack') or {}
        ov = sub.get('stack') or {}
        how = (f'image {ov["image"]}' if ov.get('image') else 'Falcon\'s test stack' if ov.get('test_stack') else
               'launch commands' if st.get('launch') else st.get('driver', ''))
        return f'suite {(suite or {}).get("name")} ({st.get("driver")}: {how})'
    if sub['kind'] == 'policy':
        return os.path.basename(os.path.normpath(sub['policy']))
    return str(sub.get('model'))


def store_results(cfg: dict, out: str, res: dict) -> str | None:
    """Copy this run's results to the store's folder for its branch (ci-results.json, ci-runs.jsonl, summary.md, and
    a model's evaluation), replacing the previous ones atomically; None when not configured or not this branch."""
    st = cfg.get('store') or {}
    branch = (res.get('commit') or {}).get('branch')
    if not st.get('dir') or not branch or branch not in (st.get('branches') or ()):
        return None
    if st.get('only_if_passed', True) and res.get('verdict') != 'passed':
        return None
    dest = os.path.join(st['dir'], branch_dir(branch))
    tmp = f'{dest}.tmp-{os.getpid()}'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in ('ci-results.json', 'ci-runs.jsonl', 'summary.md', 'ci-config.json'):
        if os.path.isfile(os.path.join(out, f)):
            shutil.copy2(os.path.join(out, f), os.path.join(tmp, f))
    ev = os.path.join(out, 'evaluation')
    if os.path.isdir(ev):
        os.makedirs(os.path.join(tmp, 'evaluation'))
        for f in ('evaluation.json', 'predictions.jsonl'):
            if os.path.isfile(os.path.join(ev, f)):
                shutil.copy2(os.path.join(ev, f), os.path.join(tmp, 'evaluation', f))
    old = None
    if os.path.isdir(dest):
        if not os.path.isfile(os.path.join(dest, 'ci-results.json')):
            shutil.rmtree(tmp, ignore_errors=True)
            raise CIError(f'{dest} exists and is not a Falcon CI store folder: not replaced')
        old = f'{dest}.old-{os.getpid()}'
        os.replace(dest, old)
    os.replace(tmp, dest)
    if old:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def run_ci(config_path: str, overrides: dict | None = None, log=None, env=None) -> tuple:
    """`falcon ci run`: (the results document, the exit code). Every output is written whatever the outcome (an
    error before the output folder is known raises CIError)."""
    from .. import __version__
    from . import provenance as PV
    env = os.environ if env is None else env
    log = log or (lambda text: print(text, file=sys.stderr, flush=True))
    o = overrides or {}
    base = os.path.dirname(os.path.abspath(config_path))
    raw = read_doc(config_path)
    probs = validate_config(raw)
    cfg = resolve(raw, base, o) if not probs else None
    if cfg is not None:
        probs = validate_config(cfg)
    if probs:
        raise CIError('the config does not validate:\n  ' + '\n  '.join(probs))
    out = prepare_out(cfg['out'], bool(o.get('force')))
    PV.write_json(os.path.join(out, 'ci-config.json'), cfg)
    t0, tm = PV.utc_now(), time.monotonic()
    kind = cfg['subject']['kind']
    res = {'schema': CI_SCHEMA, 'name': cfg['name'], 'verdict': None, 'exit_code': None, 'started': t0,
           'finished': None, 'wall_s': None, 'app_version': __version__, 'config': os.path.abspath(config_path),
           'config_sha256': PV.sha256_json({k: v for k, v in raw.items()}), 'out': out,
           'subject': {'kind': kind, 'label': None}, 'carla': {'mode': cfg['carla']['mode'],
                                                               'fake_carla': cfg['carla'].get('fake_carla')},
           'commit': commit_info(cfg['subject'].get('repo'), env), 'deterministic': None,
           'not_deterministic_because': [], 'flaky_count_as': cfg['flaky']['count_as'],
           'runs': [], 'summary': None, 'gates': [], 'comparison': None, 'baseline': None, 'latency': None,
           'rules': None, 'evaluation': None, 'artifacts': {}, 'stored': None, 'error': None, 'notes': []}
    svc = None
    old = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGINT)}

    def on_term(*_):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, on_term)
    except ValueError:                           # not the main thread (a GUI's worker): no handler
        old = None
    try:
        proj = _project(cfg, out)
        suite = None
        if kind == 'stack':
            suite = load_suite(cfg['subject'])
            from .stack import validate_suite
            bad = validate_suite(suite, proj)
            if bad:
                raise CIError('the suite does not validate: ' + '; '.join(bad))
            st = suite['stack']
            det, why = determinism(suite.get('motion', 'kinematic'), int(suite.get('traffic_vehicles', 0)),
                                   st['driver'], (cfg['subject'].get('stack') or {}).get('deterministic'))
            if st['driver'] == 'ros2' and not st.get('launch'):
                raise CIError('a ROS 2 stack needs launch commands: give subject.stack.image, .command or '
                              '.test_stack (or "launch" in the suite)')
        elif kind == 'policy':
            det, why = determinism(cfg['subject'].get('motion', 'kinematic'),
                                   int(cfg['subject'].get('traffic_vehicles', 0)))
        elif cfg['subject'].get('drives'):
            det, why = True, []                  # pinned drives: the same frames every time
        else:
            det, why = False, ['drives recorded now: a drive is not repeatable on a real server (CARLA\'s physics '
                               'and traffic manager); pin subject.drives to compare across commits']
        res['deterministic'], res['not_deterministic_because'] = det, why
        res['subject']['label'] = subject_label(cfg, suite)
        if cfg['flaky']['retries'] and det and kind != 'model':
            raise CIError('flaky.retries: this mode is deterministic (kinematic, no traffic) — a retry repeats the '
                          'run bit for bit, failure included. Retries are for physics, traffic or a stack declared '
                          'non-deterministic (subject.stack.deterministic: false)')
        svc = _service(cfg, out, proj.workspace)
        ctx = Ctx(cfg, out, svc, proj, suite, log)
        log(f'falcon ci: {res["subject"]["label"]}, {cfg["carla"]["mode"]} CARLA, output {out}')
        subj = {'stack': run_stack, 'policy': run_policy, 'model': run_model}[kind](ctx)
        records = subj['records']
        res['runs'] = records
        res['artifacts'] = subj['artifacts']
        res['latency'] = subj.get('latency')
        res['rules'] = subj.get('rules')
        res['subject']['describe'] = subj.get('describe')
        res['evaluation'] = subj.get('evaluation')
        if subj.get('leaderboard_metrics'):
            res['leaderboard_metrics'] = subj['leaderboard_metrics']
        seed = int(cfg['seed'])
        res['summary'] = {'overall': summarise_records(records, subj['metrics'], seed),
                          'groups': group_summaries(records, subj['metrics'], seed)}
        with open(os.path.join(out, 'ci-runs.jsonl'), 'w', encoding='utf-8') as fh:
            for r in records:
                fh.write(json.dumps(r, default=str) + '\n')
        baseline, why_none = load_baseline(cfg, kind)
        comparison = None
        reg = (cfg['gates'].get('regressions') or {})
        n_boot = int(reg.get('n_boot', 1000))
        if baseline is not None:
            if kind == 'model':
                kd = subj['describe']['model_kind']
                bdir = baseline.get('evaluation')
                if baseline['kind'] == 'model':
                    from . import harness as H
                    bm = proj.item('models', baseline['model'])
                    if bm is None:
                        raise CIError(f'baseline.model: no model {baseline["model"]!r} in the project')
                    bdir = os.path.join(out, 'baseline-evaluation')
                    log(f'evaluating the baseline model {bm["id"]} on the same drives')
                    H.evaluate(subj['describe']['drives'], bm, bdir, proj.workspace, cfg['subject'].get('camera'),
                               int(cfg['subject'].get('every', 1)), int(cfg['subject'].get('n_boot', 1000)), seed)
                comparison = compare_models(bdir, subj['artifacts']['evaluation'], kd, n_boot, seed)
                want = reg.get('metrics') or list(MODEL_REGRESSION_DEFAULT)
                if want != 'all':
                    comparison['rows'] = [r for r in comparison['rows'] if r['metric'] in want]
                    comparison['regressions'] = [m for m in comparison['regressions'] if m in want]
                    comparison['improvements'] = [m for m in comparison['improvements'] if m in want]
            else:
                names = reg.get('metrics') or list(subj['metrics'])
                names = list(subj['metrics']) if names == 'all' else names
                comparison = compare_records(baseline['records'], records, subj['metrics'], names, n_boot, seed)
                if not comparison['pairs']:
                    why_none = (f'no run in common with the baseline {baseline["source"]} (the suite, seeds or '
                                f'parameter values changed?)')
        res['comparison'] = comparison
        res['baseline'] = {'label': (baseline or {}).get('source'), 'kind': (baseline or {}).get('kind'),
                           'commit': (baseline or {}).get('commit'), 'why': why_none if why_none else None}
        res['gates'] = evaluate_gates(cfg, subj, res['summary'], comparison, why_none)
        unfinished = [r['name'] for r in records if r['status'] != 'ok']
        if unfinished:
            res['error'] = (f'{len(unfinished)} of {len(records)} runs did not finish ({", ".join(unfinished[:5])}'
                            f'{" …" if len(unfinished) > 5 else ""}): the result is incomplete')
        if not records:
            res['error'] = 'no runs'
        st = [g['status'] for g in res['gates']]
        res['verdict'] = ('error' if res['error'] or 'error' in st else 'failed' if 'failed' in st else 'passed')
    except KeyboardInterrupt:
        res['verdict'], res['error'] = 'cancelled', 'cancelled (SIGINT or SIGTERM)'
    except CIError as e:
        res['verdict'], res['error'] = 'error', str(e)
    except Exception as e:                       # a crash still leaves JUnit and the results for the CI to show
        import traceback
        res['verdict'], res['error'] = 'error', f'{type(e).__name__}: {e}'
        res['traceback'] = traceback.format_exc()
        log(res['traceback'])
    finally:
        if old is not None:                      # a second signal must not cut the tidying and the outputs short
            for n in old:
                signal.signal(n, signal.SIG_IGN)
        if svc is not None and not cfg['carla'].get('keep_server'):
            try:
                svc.shutdown()
            except Exception as e:               # noqa: BLE001 — tidying must not hide the result
                res['notes'].append(f'shutting the service down: {e}')
    try:
        return _finish(cfg, out, res, raw, config_path, tm, env)
    finally:
        if old is not None:
            for n, h in old.items():
                signal.signal(n, h)


def _finish(cfg, out, res, raw, config_path, tm, env) -> tuple:
    from . import provenance as PV
    res['exit_code'] = VERDICT_CODE[res['verdict']]
    res['finished'] = PV.utc_now()
    res['wall_s'] = round(time.monotonic() - tm, 3)
    res['provenance'] = PV.block({'config': os.path.abspath(config_path)} |
                                 ({'suite': cfg['subject']['suite']} if isinstance(cfg['subject'].get('suite'), str)
                                  else {}) |
                                 ({'policy': os.path.join(cfg['subject']['policy'], 'policy.json')}
                                  if cfg['subject'].get('policy') else {}) |
                                 ({'project': cfg['project']} if cfg.get('project') else {}),
                                 seed=int(cfg['seed']), settings={'carla': cfg['carla'], 'flaky': cfg['flaky']})
    write_outputs(cfg, out, res, env)
    return res, res['exit_code']


def write_outputs(cfg: dict, out: str, res: dict, env=None):
    """ci-results.json, junit.xml, summary.md, run.json, report.html; the summary appended to $GITHUB_STEP_SUMMARY
    when it is set (outputs.github_step_summary auto) or asked for; the store's branch folder when configured."""
    from . import provenance as PV
    from .paths import atomic_write
    env = os.environ if env is None else env
    PV.write_json(os.path.join(out, 'ci-results.json'), res)
    atomic_write(os.path.join(out, 'junit.xml'), junit_xml(res))
    md = markdown(res)
    atomic_write(os.path.join(out, 'summary.md'), md)
    PV.write_json(os.path.join(out, 'run.json'), {
        'schema': PV.RUN_SCHEMA, 'worker': 'ci', 'status': 'ok' if res['verdict'] in ('passed', 'failed') else
        res['verdict'], 'exit_code': res['exit_code'], 'started': res['started'], 'finished': res['finished'],
        'error': res.get('error'), 'name': res['name'], 'verdict': res['verdict'],
        'provenance': res['provenance'], 'outputs': {f: PV.sha256_file(os.path.join(out, f)) for f in
                                                     ('ci-results.json', 'junit.xml', 'summary.md')}})
    if (cfg.get('outputs') or {}).get('html', True):
        try:
            from .reports import write_report
            write_report('ci', out, os.path.join(out, 'report.html'))
        except Exception as e:                   # noqa: BLE001 — the other outputs stand without the page
            res['notes'].append(f'report.html not written: {e}')
            PV.write_json(os.path.join(out, 'ci-results.json'), res)
    if res.get('verdict') in ('passed', 'failed'):
        try:
            res['stored'] = store_results(cfg, out, res)
        except (CIError, OSError) as e:
            res['notes'].append(f'not stored: {e}')
        if res.get('stored') or res['notes']:
            PV.write_json(os.path.join(out, 'ci-results.json'), res)
    want = (cfg.get('outputs') or {}).get('github_step_summary', 'auto')
    target = env.get('GITHUB_STEP_SUMMARY')
    if target and want in ('auto', True):
        with open(target, 'a', encoding='utf-8') as fh:       # a file the runner gives the job: not a network call
            fh.write(md + '\n')


# ========================================================================================================== the CLI
def main_ci(argv) -> int:
    """`falcon ci run|validate|init|summary|post|lint|schema …` (docs/ci.md). run: 0 passed, 1 a gate failed, 2 error,
    3 cancelled; the others: 0 ok, 2 failed or a usage error."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon ci', description='CI mode: a scenario suite headless, gated, reported '
                                                               'for CI systems (docs/ci.md)')
    ap.add_argument('--fake-carla', default='')
    sub = ap.add_subparsers(dest='what', required=True)
    r = sub.add_parser('run', help='run a CI config: exit 0 passed, 1 a gate failed, 2 error, 3 cancelled')
    r.add_argument('config')
    r.add_argument('--out', default='', help='the output folder (default: the config\'s "out", else falcon-ci-out '
                                             'beside it)')
    r.add_argument('--mode', choices=('falcon', 'fake'), default='', help='CARLA: Falcon\'s own server, or the fake '
                                                                         'carla (smoke)')
    r.add_argument('--fake-carla', dest='fake_carla_run', default='', help='the fake carla package (implies --mode '
                                                                           'fake)')
    r.add_argument('--baseline', default='', help='compare against this folder (overrides the config)')
    r.add_argument('--no-baseline', action='store_true')
    r.add_argument('--force', action='store_true', help='replace a previous CI run\'s outputs in the folder')
    r.add_argument('--keep-server', action='store_true')
    r.add_argument('--quiet', action='store_true')
    v = sub.add_parser('validate', help='check a CI config')
    v.add_argument('config')
    i = sub.add_parser('init', help='write a CI config (and a suite) into a repository; --github adds a workflow')
    i.add_argument('target', nargs='?', default='.')
    i.add_argument('--kind', choices=KINDS, default='stack')
    i.add_argument('--github', action='store_true', help='also write .github/workflows/falcon.yml')
    i.add_argument('--config', default='falcon-ci.yml', help='the config\'s file name in TARGET')
    i.add_argument('--falcon-ref', default='', help='the Falcon Autonomy git ref the smoke job installs')
    i.add_argument('--force', action='store_true', help='overwrite files that exist')
    s = sub.add_parser('summary', help='print a CI output\'s Markdown summary')
    s.add_argument('results')
    p = sub.add_parser('post', help='post the summary to a pull request (comment), a check run or a commit status; '
                                    'a dry run unless --post')
    p.add_argument('results')
    p.add_argument('--repo', default='', help='OWNER/NAME (default: $GITHUB_REPOSITORY)')
    p.add_argument('--pr', type=int, default=0, help='the pull request (default: from $GITHUB_REF / the event)')
    p.add_argument('--sha', default='', help='the commit for a check or status (default: the PR head / the result\'s)')
    p.add_argument('--as', dest='kind', choices=('comment', 'check', 'status'), default='comment')
    p.add_argument('--via', choices=('gh', 'api'), default='gh', help='the gh CLI (its login or GH_TOKEN) or the REST '
                                                                       'API with $GITHUB_TOKEN')
    p.add_argument('--api-url', default='', help='the API root (default: $GITHUB_API_URL or https://api.github.com)')
    p.add_argument('--post', action='store_true', help='actually post (PUBLIC: visible to everyone who can see the '
                                                       'repository); without it nothing is sent')
    lt = sub.add_parser('lint', help='structural checks of a GitHub Actions workflow file (and actionlint if '
                                     'installed)')
    lt.add_argument('workflow')
    sub.add_parser('schema', help='print the falcon.ci/1 JSON Schema')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        return _dispatch(a)
    except CIError as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print('cancelled', file=sys.stderr, flush=True)
        return 3


def _dispatch(a) -> int:
    from . import ci_github as G
    if a.what == 'schema':
        print(json.dumps(CONFIG_JSON_SCHEMA, indent=2))
        return 0
    if a.what == 'validate':
        raw = read_doc(a.config)
        probs = validate_config(raw)
        if not probs:
            probs = validate_config(resolve(raw, os.path.dirname(os.path.abspath(a.config))))
        for x in probs:
            print(x, file=sys.stderr)
        print('valid' if not probs else f'{len(probs)} problems', flush=True)
        return 0 if not probs else 2
    if a.what == 'run':
        fake = a.fake_carla_run or a.fake_carla
        o = {'out': a.out, 'mode': a.mode, 'fake_carla': fake, 'baseline': a.baseline, 'no_baseline': a.no_baseline,
             'force': a.force, 'keep_server': a.keep_server}
        res, code = run_ci(a.config, o, log=(lambda t: None) if a.quiet else None)
        for g in res['gates']:
            print(f'  {g["status"]:8s} {g["name"]}: {g.get("measured") or ""}', flush=True)
        if res.get('error'):
            print(f'  error: {res["error"]}', flush=True)
        print(f'{res["verdict"].upper()} (exit {code}); outputs in {res["out"]}', flush=True)
        return code
    if a.what == 'summary':
        p = a.results if os.path.isfile(a.results) else os.path.join(a.results, 'summary.md')
        with open(p, encoding='utf-8') as fh:
            print(fh.read())
        return 0
    if a.what == 'init':
        written = G.init(a.target, a.kind, a.github, a.config, a.falcon_ref, a.force)
        for w in written:
            print(w)
        return 0
    if a.what == 'lint':
        probs = G.lint_workflow_file(a.workflow)
        for x in probs:
            print(x, file=sys.stderr)
        print('no problems found' if not probs else f'{len(probs)} problems', flush=True)
        return 0 if not probs else 2
    if a.what == 'post':
        return G.main_post(a)
    raise CIError(f'unknown command {a.what}')
