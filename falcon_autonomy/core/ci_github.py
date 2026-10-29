"""CI mode's integration with GitHub (standard library; PyYAML for the workflow checks): `falcon ci init` (a CI config,
a suite and a GitHub Actions workflow written into a repository the user names), `falcon ci lint` (structural checks
of a workflow file, and actionlint when it is installed) and `falcon ci post` (the Markdown summary as a pull-request
comment, a check run or a commit status). docs/ci.md is the reference.

Posting is a PUBLIC action: it writes to a GitHub repository, visible to everyone who can see it. `falcon ci post`
sends nothing without --post (it prints what it would send), refuses when FALCON_CI_NO_POST is set (the tests set
it), and goes through a transport — GhTransport (the gh CLI, with its own login or GH_TOKEN) or HttpTransport (the
REST API with GITHUB_TOKEN) — that the tests replace with a recording fake. A comment carries a hidden marker
(<!-- falcon-ci:NAME -->) so a later run edits it instead of adding another.

The workflow (workflow()) has two jobs: "smoke" on a GitHub-hosted runner (ubuntu-24.04) runs the config against the
fake carla — Falcon checked out from its public repository at a pinned ref, the fake carla from its tests — and
"carla" on a self-hosted runner with the labels [self-hosted, linux, x64, gpu, carla] (an NVIDIA GPU, CARLA 0.9.16,
Falcon installed, Docker for ROS 2 stacks: docs/ci.md, "The self-hosted runner") runs it on Falcon's own server,
never for pull requests from forks. Both upload the output folder as an artifact; the carla job can post the summary
when the repository variable FALCON_CI_POST is 'true' (off unless the repository's owner turns it on).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

FALCON_REPO = 'Sherin-SEF-AI/Falcon-Autonomy'
RUNNER_LABELS = ('self-hosted', 'linux', 'x64', 'gpu', 'carla')
MARKER = '<!-- falcon-ci:{name} -->'
COMMENT_MAX = 65000             # GitHub's limit for a comment or a check run's summary is 65 536 characters
EXPR_NAMES = {'github', 'env', 'vars', 'secrets', 'needs', 'steps', 'runner', 'matrix', 'inputs', 'job', 'jobs',
              'strategy', 'always', 'success', 'failure', 'cancelled', 'contains', 'startsWith', 'endsWith', 'format',
              'join', 'toJSON', 'fromJSON', 'hashFiles', 'true', 'false', 'null'}


class PostError(Exception):
    pass


# ======================================================================================================= templates
def config_template(kind: str = 'stack', name: str = 'my-stack', suite: str = 'falcon-suite.json') -> str:
    """A CI config (falcon.ci/1) to edit, as commented YAML."""
    head = ('# Falcon Autonomy CI config (falcon.ci/1). Reference: docs/ci.md of Falcon Autonomy.\n'
            '# Runs headless: `falcon ci run falcon-ci.yml`; exit 0 passed, 1 a gate failed, 2 error, 3 cancelled.\n'
            'schema: falcon.ci/1\n'
            f'name: {name}\n')
    common_tail = ('carla:\n'
                   '  mode: falcon              # Falcon\'s own CARLA 0.9.16 server; `--mode fake` (or '
                   '--fake-carla DIR) for smoke runs\n'
                   'baseline:                   # what regressions are judged against\n'
                   '  kind: store               # the results stored for a branch by earlier runs (see store)\n'
                   '  store: ${FALCON_CI_STORE} # a folder on the self-hosted runner; unset: no baseline, no store\n'
                   '  branch: main\n'
                   'store:                      # keep this run\'s results as the baseline of its branch\n'
                   '  dir: ${FALCON_CI_STORE}\n'
                   '  branches: [main]\n'
                   '  only_if_passed: true\n'
                   'flaky:\n'
                   '  retries: 0                # only for non-deterministic modes (physics, traffic); recorded\n'
                   'outputs:\n'
                   '  html: true\n'
                   '  github_step_summary: auto # append summary.md to $GITHUB_STEP_SUMMARY when it is set\n')
    if kind == 'stack':
        body = ('subject:\n'
                '  kind: stack\n'
                '  repo: .                   # the git repository whose commit is under test (relative to this file)\n'
                f'  suite: {suite}   # a falcon.suite/1 suite: scenarios x seeds with pass/fail criteria\n'
                '  stack:\n'
                '    driver: ros2\n'
                '    image: ${STACK_IMAGE}   # your stack\'s Docker image, run on the host network beside Falcon\'s '
                'bridge node\n'
                '    bridge:\n'
                '      control: autoware     # autoware | ackermann | carla\n'
                '      deadline_s: 0.5\n'
                '  motion: kinematic         # kinematic repeats bit for bit; physics and traffic do not\n'
                '  traffic_vehicles: 0\n'
                'gates:\n'
                '  pass_rate:\n'
                '    min: 0.5                # demonstrate: the Wilson 95 % lower bound must reach it (6 of 6 passing: '
                '0.61)\n'
                '    rule: demonstrate\n'
                '  regressions:\n'
                '    metrics: [pass_rate, collision_rate, criticality, min_ttc_s]\n'
                '    on_missing_baseline: skip\n'
                '  rule_violations:\n'
                '    max_per_km: 5.0\n'
                '    rule: estimate\n'
                '  latency:\n'
                '    p95_ms: 200             # the stack\'s control wait through the bridge, pooled over steps\n'
                '  max_missed_deadlines: 0\n')
    elif kind == 'policy':
        body = ('subject:\n'
                '  kind: policy\n'
                '  repo: .\n'
                '  policy: policies/my-policy   # a Falcon policy folder (policy.json, weights.pt)\n'
                '  routes:\n'
                '    scenarios: [lead_brake, stalled_vehicle]\n'
                '    seeds_per: 3\n'
                '    free: 0\n'
                '  route_pass:\n'
                '    min_route_completion: 0.9\n'
                '    max_penalised_infractions: 0\n'
                '  motion: kinematic\n'
                '  traffic_vehicles: 0\n'
                'gates:\n'
                '  pass_rate: {min: 0.5, rule: demonstrate}\n'
                '  metrics:\n'
                '    - {metric: driving_score, min: 0.5, rule: estimate}\n'
                '  regressions:\n'
                '    metrics: [pass_rate, driving_score, route_completion, infraction_penalty]\n'
                '  latency:\n'
                '    p99_ms: 20              # the policy\'s decision time on the runner\n')
    else:
        body = ('project: sim/project.falcon.json   # where the model is registered\n'
                'subject:\n'
                '  kind: model\n'
                '  repo: .\n'
                '  model: my-detector\n'
                '  drives: [/data/falcon/test-drives/*]   # pinned test drives (or record: {seeds: [1, 2, 3]})\n'
                '  every: 2\n'
                'gates:\n'
                '  metrics:\n'
                '    - {metric: AP, min: 0.3, rule: refute}\n'
                '  regressions:\n'
                '    metrics: [AP, AP50, AR100]\n'
                '  latency:\n'
                '    p90_ms: 50              # inference on the runner, not on the target device\n')
    return head + body + common_tail


def workflow(config: str = 'falcon-ci.yml', kind: str = 'stack', ros2_image: bool = True,
             falcon_ref: str = '', stack_image: bool = True) -> str:
    """The GitHub Actions workflow (module doc) for a config in the repository's root."""
    from .. import __version__
    ref = falcon_ref or f'v{__version__}'
    extras = {'policy': ' torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url '
                        'https://pypi.org/simple', 'model': ' onnxruntime'}.get(kind, '')
    build = ''
    if kind == 'stack' and stack_image:
        build = ('      - name: Build the stack\'s image (adapt to how your stack is built)\n'
                 '        run: docker build -t "$STACK_IMAGE" .\n')
    bridge = ''
    if kind == 'stack' and ros2_image:
        bridge = '      - name: Build Falcon\'s ROS 2 bridge image\n        run: {falcon} stack image\n'
    env_stack = '  STACK_IMAGE: falcon-ci-stack:${{ github.sha }}\n' if kind == 'stack' and stack_image else ''
    labels = ', '.join(RUNNER_LABELS)
    return (
        '# Falcon Autonomy CI (written by `falcon ci init --github`; docs/ci.md of Falcon Autonomy).\n'
        '# smoke: a GitHub-hosted runner, the fake carla — a test of the integration, not of driving.\n'
        f'# carla: a self-hosted runner labelled [{labels}] with an NVIDIA GPU, CARLA 0.9.16 and\n'
        '#        Falcon installed (docs/ci.md, "The self-hosted runner"); never for pull requests from forks.\n'
        'name: Falcon CI\n'
        '\n'
        'on:\n'
        '  push:\n'
        '    branches: [main]\n'
        '  pull_request:\n'
        '  workflow_dispatch:\n'
        '\n'
        'permissions:\n'
        '  contents: read\n'
        '\n'
        'concurrency:\n'
        '  group: falcon-ci-${{ github.ref }}\n'
        '  cancel-in-progress: true\n'
        '\n'
        'env:\n'
        f'  FALCON_REF: {ref}     # the Falcon Autonomy release the smoke job installs\n'
        + env_stack +
        '\n'
        'jobs:\n'
        '  smoke:\n'
        '    name: Falcon smoke (fake CARLA)\n'
        '    runs-on: ubuntu-24.04\n'
        '    timeout-minutes: 45\n'
        '    steps:\n'
        '      - name: Check out the code under test\n'
        '        uses: actions/checkout@v4\n'
        '      - name: Check out Falcon Autonomy\n'
        '        uses: actions/checkout@v4\n'
        '        with:\n'
        f'          repository: {FALCON_REPO}\n'
        '          ref: ${{ env.FALCON_REF }}\n'
        '          path: .falcon\n'
        '      - name: Set up Python\n'
        '        uses: actions/setup-python@v5\n'
        '        with:\n'
        "          python-version: '3.12'\n"
        '      - name: Install Falcon Autonomy\n'
        '        run: |\n'
        '          sudo apt-get update -q\n'
        '          sudo apt-get install -y -q libgl1 libegl1 libxkbcommon0 libglib2.0-0 libdbus-1-3\n'
        '          python -m pip install --upgrade pip\n'
        f'          python -m pip install ./.falcon PyYAML{extras}\n'
        + build + bridge.format(falcon='python -m falcon_autonomy') +
        '      - name: Falcon CI against the fake carla\n'
        f'        run: python -m falcon_autonomy ci run {config} --fake-carla .falcon/tests/fake_carla --no-baseline '
        '--out falcon-ci-smoke --force\n'
        '      - name: Keep the results\n'
        '        if: always()\n'
        '        uses: actions/upload-artifact@v4\n'
        '        with:\n'
        '          name: falcon-ci-smoke\n'
        '          path: falcon-ci-smoke\n'
        '          if-no-files-found: ignore\n'
        '\n'
        '  carla:\n'
        '    name: Falcon suite on CARLA (self-hosted GPU runner)\n'
        '    needs: smoke\n'
        '    if: github.event_name != \'pull_request\' || github.event.pull_request.head.repo.full_name == '
        'github.repository\n'
        f'    runs-on: [{labels}]\n'
        '    timeout-minutes: 240\n'
        '    permissions:\n'
        '      contents: read\n'
        '      pull-requests: write\n'
        '    env:\n'
        '      FALCON_CI_STORE: ${{ vars.FALCON_CI_STORE }}\n'
        '    steps:\n'
        '      - name: Check out the code under test\n'
        '        uses: actions/checkout@v4\n'
        '      - name: Check the runner\n'
        '        run: falcon doctor || echo "::warning::falcon doctor reported a problem (see its output)"\n'
        + build + bridge.format(falcon='falcon') +
        '      - name: Falcon CI on Falcon\'s CARLA server\n'
        f'        run: falcon ci run {config} --out falcon-ci-out --force\n'
        '      - name: Keep the results\n'
        '        if: always()\n'
        '        uses: actions/upload-artifact@v4\n'
        '        with:\n'
        '          name: falcon-ci\n'
        '          path: falcon-ci-out\n'
        '          if-no-files-found: ignore\n'
        '      - name: Comment on the pull request (only when the repository variable FALCON_CI_POST is true)\n'
        '        if: always() && github.event_name == \'pull_request\' && vars.FALCON_CI_POST == \'true\'\n'
        '        env:\n'
        '          GH_TOKEN: ${{ github.token }}\n'
        '        run: falcon ci post falcon-ci-out --as comment --via gh --post\n')


def init(target: str, kind: str = 'stack', github: bool = False, config: str = 'falcon-ci.yml',
         falcon_ref: str = '', force: bool = False) -> list:
    """Write a CI config (and, for a stack, a suite) into TARGET and, with github, .github/workflows/falcon.yml;
    returns the paths written. Files that exist are kept unless force. An existing config decides the workflow's
    kind."""
    from . import stack as ST
    from .ci import CIError, read_doc
    from .paths import atomic_write
    from .provenance import write_json
    if not os.path.isdir(target):
        raise CIError(f'{target} is not a folder')
    target = os.path.abspath(target)
    name = re.sub(r'[^A-Za-z0-9_.-]+', '-', os.path.basename(target)) or 'falcon-ci'
    written = []
    cpath = os.path.join(target, config)
    ros2 = stack_image = True
    if os.path.isfile(cpath) and not force:
        doc = read_doc(cpath)
        kind = (doc.get('subject') or {}).get('kind', kind)
        st = (doc.get('subject') or {}).get('stack') or {}
        stack_image = bool(st.get('image'))
        ros2 = st.get('bridge_node', 'image') == 'image' and (st.get('driver', 'ros2') == 'ros2')
    else:
        atomic_write(cpath, config_template(kind, name))
        written.append(cpath)
        if kind == 'stack':
            spath = os.path.join(target, 'falcon-suite.json')
            if force or not os.path.isfile(spath):
                doc = ST.new_suite(name, 'ros2', ['lead_brake', 'stalled_vehicle'], [0, 1, 2])
                doc['stack']['bridge'] = {'control': 'autoware', 'deadline_s': 0.5}
                doc['about'] = 'written by falcon ci init: edit the scenarios, seeds and criteria'
                write_json(spath, doc)
                written.append(spath)
    if github:
        wpath = os.path.join(target, '.github', 'workflows', 'falcon.yml')
        if os.path.isfile(wpath) and not force:
            raise CIError(f'{wpath} exists: give --force to replace it')
        text = workflow(config, kind, ros2 and kind == 'stack', falcon_ref, stack_image and kind == 'stack')
        probs = lint_workflow(text)
        if probs:                                            # never write a template that fails its own checks
            raise CIError('the workflow template fails its checks: ' + '; '.join(probs))
        atomic_write(wpath, text)
        written.append(wpath)
    return written


# ====================================================================================================== the checks
def _expr_problems(expr: str, where: str) -> list:
    body = re.sub(r"'(?:[^']|'')*'", "''", expr)
    out = []
    for m in re.finditer(r"(?<![\w.\-'])([A-Za-z_][\w-]*)", body):
        if m.group(1) not in EXPR_NAMES and not m.group(1).isdigit():
            out.append(f'{where}: unknown name {m.group(1)!r} in the expression {expr.strip()!r}')
    return out


def lint_workflow(text: str) -> list:
    """Structural checks of a GitHub Actions workflow: YAML, triggers, jobs with runs-on and steps, each step one of
    uses / run, actions pinned to a version, needs naming jobs that exist, permissions and expressions (known
    contexts and functions only, balanced). Not GitHub's full schema: actionlint (lint_workflow_file) goes further."""
    out = []
    if '\t' in text:
        out.append('tabs in the file (YAML indents with spaces)')
    if text.count('${{') != text.count('}}'):
        out.append('unbalanced ${{ … }}')
    try:
        import yaml
    except ImportError:
        return out + ['PyYAML is needed to check a workflow']
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return out + [f'not valid YAML: {e}']
    if not isinstance(doc, dict):
        return out + ['not a mapping']
    on = doc.get('on', doc.get(True))           # YAML 1.1 reads a bare `on` key as true
    if not on:
        out.append('no triggers (on:)')
    elif isinstance(on, dict):
        bad = sorted(set(on) - {'push', 'pull_request', 'pull_request_target', 'workflow_dispatch', 'schedule',
                                'workflow_call', 'release', 'merge_group', 'workflow_run'})
        if bad:
            out.append(f'on: unknown events {bad}')
    perms_ok = {'read', 'write', 'none'}

    def perms(p, where):
        if isinstance(p, str):
            if p not in ('read-all', 'write-all'):
                out.append(f'{where}: permissions {p!r}')
        elif isinstance(p, dict):
            out.extend(f'{where}: permission {k}: {v!r}' for k, v in p.items() if v not in perms_ok)
    if 'permissions' in doc:
        perms(doc['permissions'], 'workflow')
    jobs = doc.get('jobs')
    if not isinstance(jobs, dict) or not jobs:
        return out + ['no jobs']
    for jid, j in jobs.items():
        w = f'jobs.{jid}'
        if not re.match(r'^[A-Za-z_][\w-]*$', str(jid)):
            out.append(f'{w}: a job id is letters, digits, - and _')
        if not isinstance(j, dict):
            out.append(f'{w}: not a mapping')
            continue
        if 'runs-on' not in j and 'uses' not in j:
            out.append(f'{w}: no runs-on')
        needs = j.get('needs') or []
        for n in [needs] if isinstance(needs, str) else needs:
            if n not in jobs:
                out.append(f'{w}: needs {n!r}, which is not a job')
        if 'timeout-minutes' in j and not isinstance(j['timeout-minutes'], int):
            out.append(f'{w}: timeout-minutes is a number')
        if 'permissions' in j:
            perms(j['permissions'], w)
        if isinstance(j.get('if'), str) and '${{' not in j['if']:
            out += _expr_problems(j['if'], f'{w}.if')
        steps = j.get('steps')
        if 'uses' not in j and (not isinstance(steps, list) or not steps):
            out.append(f'{w}: no steps')
            continue
        for k, s in enumerate(steps or ()):
            ws = f'{w}.steps[{k}]'
            if not isinstance(s, dict):
                out.append(f'{ws}: not a mapping')
                continue
            if ('uses' in s) == ('run' in s):
                out.append(f'{ws}: give exactly one of uses and run')
            if 'uses' in s and not re.match(r'^[\w.-]+/[\w./-]+@[\w.-]+$', str(s['uses'])) and \
                    not str(s['uses']).startswith(('./', 'docker://')):
                out.append(f'{ws}: uses {s["uses"]!r} is not owner/repo@version')
            if 'with' in s and not isinstance(s['with'], dict):
                out.append(f'{ws}: with is a mapping')
            if isinstance(s.get('if'), str) and '${{' not in s['if']:
                out += _expr_problems(s['if'], f'{ws}.if')
            unknown = sorted(set(s) - {'name', 'id', 'uses', 'run', 'with', 'env', 'if', 'shell', 'working-directory',
                                       'continue-on-error', 'timeout-minutes'})
            if unknown:
                out.append(f'{ws}: unknown keys {unknown}')
    for m in re.finditer(r'\$\{\{(.*?)\}\}', text, re.S):
        out += _expr_problems(m.group(1), 'expression')
    return out


def lint_workflow_file(path: str) -> list:
    """lint_workflow on a file, and actionlint's findings when actionlint is on PATH."""
    from .ci import CIError
    try:
        with open(path, encoding='utf-8') as fh:
            out = lint_workflow(fh.read())
    except OSError as e:
        raise CIError(f'cannot read {path}: {e}') from None
    exe = shutil.which('actionlint')
    if exe:
        p = subprocess.run([exe, '-no-color', path], capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            out += [f'actionlint: {x}' for x in (p.stdout + p.stderr).splitlines() if x.strip()]
    return out


# ========================================================================================================= posting
class GhTransport:
    """GitHub's REST API through the gh CLI (`gh api`, authenticated by `gh auth login` or GH_TOKEN)."""

    def __init__(self, run=subprocess.run, gh: str = 'gh'):
        self.run, self.gh = run, gh

    def request(self, method: str, path: str, body=None):
        argv = [self.gh, 'api', '--method', method, path, '-H', 'Accept: application/vnd.github+json']
        if body is not None:
            argv += ['--input', '-']
        p = self.run(argv, input=json.dumps(body) if body is not None else None, capture_output=True, text=True,
                     timeout=60)
        if p.returncode != 0:
            raise PostError(f'gh api {method} {path}: {(p.stderr or p.stdout).strip()[:500]}')
        return json.loads(p.stdout) if (p.stdout or '').strip() else None


class HttpTransport:
    """GitHub's REST API over HTTPS with a token (GITHUB_TOKEN in Actions)."""

    def __init__(self, token: str, api: str = 'https://api.github.com', urlopen=None):
        if not token:
            raise PostError('no token: set GITHUB_TOKEN (or GH_TOKEN), or post --via gh')
        import urllib.request
        self.token, self.api, self.urlopen = token, api.rstrip('/'), urlopen or urllib.request.urlopen

    def request(self, method: str, path: str, body=None):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.api + path, data=None if body is None else json.dumps(body).encode(),
                                     method=method, headers={'Authorization': f'Bearer {self.token}',
                                                             'Accept': 'application/vnd.github+json',
                                                             'X-GitHub-Api-Version': '2022-11-28',
                                                             'User-Agent': 'falcon-autonomy-ci',
                                                             'Content-Type': 'application/json'})
        try:
            with self.urlopen(req, timeout=60) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            raise PostError(f'{method} {path}: HTTP {e.code} {e.read()[:300]!r}') from None
        except urllib.error.URLError as e:
            raise PostError(f'{method} {path}: {e.reason}') from None
        return json.loads(data) if data.strip() else None


def comment_body(res: dict, md: str) -> str:
    body = MARKER.format(name=res.get('name', 'falcon-ci')) + '\n' + md
    if len(body) > COMMENT_MAX:
        body = (body[:COMMENT_MAX - 200] + '\n\n… (truncated: the whole summary is summary.md in the CI run\'s '
                'artifact)\n')
    return body


def plan(res: dict, md: str, kind: str, repo: str, pr: int | None, sha: str | None) -> list:
    """The requests a post makes, in order, as (method, path, body); a comment's first request looks for an earlier
    Falcon comment (by its marker) to edit."""
    from .ci import CIError
    if not repo or '/' not in repo:
        raise CIError('give --repo OWNER/NAME (or run in GitHub Actions: $GITHUB_REPOSITORY)')
    v = res.get('verdict', 'error')
    name = res.get('name', 'falcon-ci')
    if kind == 'comment':
        if not pr:
            raise CIError('a comment needs a pull request: give --pr N (or run on a pull_request event)')
        return [('GET', f'/repos/{repo}/issues/{int(pr)}/comments?per_page=100', None),
                ('POST|PATCH', f'/repos/{repo}/issues/{int(pr)}/comments', {'body': comment_body(res, md)})]
    if not sha:
        raise CIError(f'a {kind} needs a commit: give --sha')
    if kind == 'check':
        concl = {'passed': 'success', 'failed': 'failure', 'error': 'failure', 'cancelled': 'cancelled'}[v]
        summ = md if len(md) <= COMMENT_MAX else md[:COMMENT_MAX - 100] + '\n\n… (truncated)\n'
        return [('POST', f'/repos/{repo}/check-runs', {
            'name': f'Falcon CI: {name}', 'head_sha': sha, 'status': 'completed', 'conclusion': concl,
            'details_url': (res.get('commit') or {}).get('run_url') or None,
            'output': {'title': f'Falcon CI {v}', 'summary': summ}})]
    state = {'passed': 'success', 'failed': 'failure', 'error': 'error', 'cancelled': 'error'}[v]
    failed = [g['name'] for g in res.get('gates') or () if g['status'] in ('failed', 'error')]
    desc = (f'{v}: ' + (', '.join(failed) if failed else 'every gate passed'))[:140]
    body = {'state': state, 'context': f'falcon-ci/{name}', 'description': desc}
    if (res.get('commit') or {}).get('run_url'):
        body['target_url'] = res['commit']['run_url']
    return [('POST', f'/repos/{repo}/statuses/{sha}', body)]


def post(res: dict, md: str, kind: str, repo: str, pr: int | None, sha: str | None, transport) -> dict:
    """Send what plan() lists through the transport: {'action', 'url'}."""
    steps = plan(res, md, kind, repo, pr, sha)
    if kind != 'comment':
        method, path, body = steps[0]
        r = transport.request(method, path, body) or {}
        return {'action': f'{kind} created', 'url': r.get('html_url') or r.get('url')}
    marker = MARKER.format(name=res.get('name', 'falcon-ci'))
    body = steps[1][2]
    found = None
    for page in range(1, 11):                   # an earlier Falcon comment, in the first 1000 comments
        rows = transport.request('GET', f'{steps[0][1]}&page={page}') or []
        found = next((c for c in rows if marker in (c.get('body') or '')), None)
        if found or len(rows) < 100:
            break
    if found:
        r = transport.request('PATCH', f'/repos/{repo}/issues/comments/{found["id"]}', body) or {}
        return {'action': 'comment updated', 'url': r.get('html_url')}
    r = transport.request('POST', steps[1][1], body) or {}
    return {'action': 'comment created', 'url': r.get('html_url')}


def _event(env) -> dict:
    p = env.get('GITHUB_EVENT_PATH')
    if p and os.path.isfile(p):
        try:
            with open(p, encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}
    return {}


def main_post(a, transport=None, env=None) -> int:
    """`falcon ci post RESULTS …`: a dry run (what would be sent) unless --post. 0 ok, 2 failed or refused."""
    from .ci import CIError
    from .provenance import read_json
    env = os.environ if env is None else env
    d = a.results
    try:
        res = read_json(os.path.join(d, 'ci-results.json'))
        with open(os.path.join(d, 'summary.md'), encoding='utf-8') as fh:
            md = fh.read()
    except (OSError, ValueError) as e:
        raise CIError(f'{d}: not a CI output ({e})') from None
    ev = _event(env)
    repo = a.repo or env.get('GITHUB_REPOSITORY') or (res.get('commit') or {}).get('repository') or ''
    pr = a.pr or (ev.get('pull_request') or {}).get('number') or (res.get('commit') or {}).get('pr')
    sha = a.sha or ((ev.get('pull_request') or {}).get('head') or {}).get('sha') or (res.get('commit') or {}).get('sha')
    steps = plan(res, md, a.kind, repo, pr, sha)
    if not a.post:
        print(f'dry run: nothing was sent. With --post this {a.kind} would be sent to {repo} (PUBLIC: visible to '
              f'everyone who can see the repository) via {a.via}:', flush=True)
        for method, path, body in steps:
            short = '' if body is None else ' ' + json.dumps({k: (v[:120] + '…' if isinstance(v, str) and len(v) > 120
                                                                  else v) for k, v in body.items()})
            print(f'  {method} {path}{short}', flush=True)
        return 0
    if env.get('FALCON_CI_NO_POST'):
        print('refused: FALCON_CI_NO_POST is set (posting is disabled here)', file=sys.stderr, flush=True)
        return 2
    if transport is None:
        if a.via == 'gh':
            if not shutil.which('gh'):
                raise CIError('the gh CLI is not installed: install it, or post --via api with GITHUB_TOKEN')
            transport = GhTransport()
        else:
            transport = HttpTransport(env.get('GITHUB_TOKEN') or env.get('GH_TOKEN') or '',
                                      a.api_url or env.get('GITHUB_API_URL') or 'https://api.github.com')
    try:
        r = post(res, md, a.kind, repo, pr, sha, transport)
    except PostError as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return 2
    print(f'{r["action"]}: {r.get("url") or "(no url returned)"}', flush=True)
    return 0
