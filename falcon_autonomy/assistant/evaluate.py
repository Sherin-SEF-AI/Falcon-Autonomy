"""`falcon assistant eval`: how well a model does Falcon's work through the assistant, measured end to end on realistic
tasks (Qt-free; docs/assistant.md, "Evaluating models").

Each task (eval_tasks.json; TASK_SCHEMA) is a request a user would make — "find my worst rare case in lead_brake and
replay it", "explain why job 3 failed" — run by the real agent (agent.Agent, its permission modes, its tools on
demand, its token pacing) against a throw-away project whose workspace is in a temporary folder and whose jobs run on
the fake carla (tests/fake_carla: workers run in this Python, no server is started, the job queue's locks are the
harness's own), so nothing touches the user's workspace, settings or CARLA server. Per task:

    setup     declarative steps run before the request, through the tools themselves (tool_api.call): {"call": tool,
              "args", "as": name, "wait"} (a job it queues is waited for), {"fail_drive": {...}} (a drive whose fake
              server never answers: the worker fails as with no server), {"private_run": name} (a run whose own
              record says it came from the dashcam footage — a marker only, no footage). What a step returns is kept
              under its name for placeholders ("{d1.run}", "{bad.job}") in the prompt, the criteria and the script.
    mode      the permission mode (ask, auto, autopilot); approvals: what the harness answers an approval card
              ({"act": "approve", "trash": "reject"} by default — trash always asks, and the harness says no).
    expect    machine-checkable success criteria (check()): tool calls made (with an argument subset; a key ending in
              "~" matches case-insensitively as a substring, or for a list every expected item), calls never made, a
              tier never used, at least one of a set of calls, a tool error containing a text (a refusal), approval
              cards answered, the project afterwards (an item in a section matching fields), runs present or gone,
              read tools called afterwards with an expected result subset, jobs the agent queued (kind, state) or a
              given job's state, the final answer containing / not containing texts, and no number flagged by the
              grounding check (unless "grounded": false).
    script    (tests only) the model's responses, for the fake server (tests/fake_groq.py): handed to run_eval's
              script_sink with its placeholders filled; never used against a real server.

Measured per task and model: success and the failed criteria; steps (requests to the main model), tool calls,
prompt and completion tokens as the server counted them, wall time and the time spent waiting for the model, the
estimated cost from the user's price table (Settings ▸ Assistant ▸ Prices; None for a model without a price), and the
grounding flags. run_eval writes results.json and summary.md (per model: success rate, mean steps, tokens, latency,
cost, grounding flags; then each task) into the output folder, and each task's audit log (session.Audit, no payloads)
as audit/<model>--<task>.jsonl.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
import threading
import time

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_tasks.json')
MODES = ('ask', 'auto', 'autopilot')
SETUP_WAIT_S = 600.0

_STR = {'type': 'string'}
_STRS = {'type': 'array', 'items': _STR}
_CALL = {'type': 'object', 'required': ['tool'], 'additionalProperties': False,
         'properties': {'tool': _STR, 'args': {'type': 'object'}}}
TASK_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['id', 'title', 'prompt', 'expect'],
    'properties': {
        'id': {'type': 'string', 'pattern': r'^[a-z0-9_]+$'},
        'title': {'type': 'string', 'minLength': 5, 'maxLength': 160},
        'prompt': {'type': 'string', 'minLength': 5},
        'mode': {'enum': list(MODES)},
        'approvals': {'type': 'object', 'additionalProperties': False, 'properties': {
            'act': {'enum': ['approve', 'reject']}, 'trash': {'enum': ['approve', 'reject']}}},
        'timeout_s': {'type': 'number', 'minimum': 10, 'maximum': 3600},
        'setup': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'properties': {
            'call': _STR, 'args': {'type': 'object'}, 'as': _STR, 'wait': {'type': 'boolean'},
            'timeout_s': {'type': 'number'}, 'fail_drive': {'type': 'object'}, 'private_run': _STR},
            'oneOf': [{'required': ['call']}, {'required': ['fail_drive']}, {'required': ['private_run']}]}},
        'expect': {'type': 'object', 'additionalProperties': False, 'properties': {
            'calls': {'type': 'array', 'items': _CALL},
            'any_of': {'type': 'array', 'items': {'type': 'array', 'minItems': 1, 'items': _CALL}},
            'not_calls': _STRS,
            'no_tier': {'type': 'array', 'items': {'enum': ['read', 'act', 'trash']}},
            'call_errors': {'type': 'array', 'items': {'type': 'object', 'required': ['text'],
                                                        'additionalProperties': False,
                                                        'properties': {'tool': _STR, 'text': _STR}}},
            'approvals': {'type': 'array', 'items': {'type': 'object', 'required': ['tool'],
                                                      'additionalProperties': False,
                                                      'properties': {'tool': _STR, 'choice': _STR}}},
            'project': {'type': 'array', 'items': {'type': 'object', 'required': ['section'],
                                                    'additionalProperties': False,
                                                    'properties': {'section': _STR, 'match': {'type': 'object'},
                                                                   'absent': {'type': 'boolean'}}}},
            'runs_exist': _STRS, 'runs_gone': _STRS,
            'reads': {'type': 'array', 'items': {'type': 'object', 'required': ['tool', 'expect'],
                                                  'additionalProperties': False,
                                                  'properties': {'tool': _STR, 'args': {'type': 'object'},
                                                                 'expect': {'type': 'object'}}}},
            'jobs': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                                                 'properties': {'id': {}, 'kind': _STR, 'state': _STR}}},
            'answer_contains': _STRS,
            'answer_any': _STRS,
            'answer_excludes': _STRS,
            'grounded': {'type': 'boolean'}}},
        'script': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'properties': {
            'tools': {'type': 'array', 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2}},
            'content': _STR, 'text': _STR}}},
    }}


# --------------------------------------------------------------------------------------------- tasks
def load_tasks(path: str = '') -> list:
    with open(path or TASKS_FILE, encoding='utf-8') as fh:
        doc = json.load(fh)
    tasks = doc.get('tasks') if isinstance(doc, dict) else doc
    if not isinstance(tasks, list):
        raise ValueError('a task file is a JSON list of tasks, or {"tasks": [...]}')
    return tasks


def validate_task(task) -> list:
    """The task's problems: its schema, unknown tool names, a repeated setup name (empty: valid)."""
    import jsonschema
    from . import agent as A
    errs = [f'{"/".join(str(p) for p in e.absolute_path) or "(task)"}: {e.message}'
            for e in jsonschema.Draft202012Validator(TASK_SCHEMA).iter_errors(task)]
    if errs:
        return errs
    ex = task['expect']
    names = [s['call'] for s in task.get('setup') or [] if 'call' in s]
    names += [c['tool'] for c in ex.get('calls') or []] + list(ex.get('not_calls') or [])
    names += [c['tool'] for alts in ex.get('any_of') or [] for c in alts]
    names += [c['tool'] for c in (ex.get('call_errors') or []) + (ex.get('approvals') or []) if c.get('tool')]
    names += [r['tool'] for r in ex.get('reads') or []]
    names += [tc[0] for st in task.get('script') or [] for tc in st.get('tools') or []]
    errs += [f'no tool named {n!r}' for n in dict.fromkeys(names) if A.lookup(n) is None]
    seen = set()
    for s in task.get('setup') or []:
        if s.get('as') in seen:
            errs.append(f'setup name {s["as"]!r} is used twice')
        seen.add(s.get('as'))
    return errs


PLACEHOLDER = re.compile(r'\{([A-Za-z_]\w*(?:\.\w+)*)\}')


def _lookup(vars_: dict, path: str):
    cur = vars_
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise KeyError(f'no value for {{{path}}} (setup names: {", ".join(vars_) or "none"})')
    return cur


def fill(obj, vars_: dict):
    """Placeholders filled from the setup's results: a string that is one placeholder becomes its value (a job id stays
    a number); placeholders inside a longer string become text."""
    if isinstance(obj, str):
        m = PLACEHOLDER.fullmatch(obj)
        if m:
            return _lookup(vars_, m.group(1))
        return PLACEHOLDER.sub(lambda m: str(_lookup(vars_, m.group(1))), obj)
    if isinstance(obj, dict):
        return {fill(k, vars_) if isinstance(k, str) else k: fill(v, vars_) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill(x, vars_) for x in obj]
    return obj


# --------------------------------------------------------------------------------------------- matching
def matches(expected, actual) -> bool:
    """Whether actual holds expected: a dict's keys a subset (a key ending in '~': a case-insensitive substring, or
    for a list every expected item among the actual ones; a dotted key reads into nested dicts), equal otherwise."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for k, v in expected.items():
            fuzzy = k.endswith('~')
            key = k[:-1] if fuzzy else k
            try:
                got = _lookup(actual, key) if '.' in key and key not in actual else actual[key]
            except KeyError:
                return False
            if fuzzy:
                if isinstance(v, list):
                    if not isinstance(got, list) or any(x not in got for x in v):
                        return False
                elif str(v).lower() not in json.dumps(got, ensure_ascii=False, default=str).lower():
                    return False
            elif not matches(v, got):
                return False
        return True
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) <= 1e-9 * max(1.0, abs(float(expected)))
        except (TypeError, ValueError):
            return False
    return expected == actual


def check(expect: dict, out: dict, read=None) -> list:
    """The failed criteria of a task, each a sentence (empty: it passed). out: what the run produced — 'calls' [{name,
    args, error}], 'answer', 'ungrounded', 'approvals' [{tool, choice}], 'jobs' [{id, kind, state, mine}], 'doc' (the
    project afterwards), 'runs' (run ids in the workspace); read(tool, args) -> result runs a read tool afterwards."""
    from . import agent as A
    fails = []
    calls = out.get('calls') or []

    def made(c):
        return any(x['name'] == c['tool'] and matches(c.get('args') or {}, x.get('args') or {}) for x in calls)
    for c in expect.get('calls') or []:
        if not made(c):
            near = [x.get('args') for x in calls if x['name'] == c['tool']]
            fails.append(f'{c["tool"]} was not called with {json.dumps(c.get("args") or {})}' +
                         (f' (called with {json.dumps(near, default=str)[:300]})' if near else ' (not called)'))
    for alts in expect.get('any_of') or []:
        if not any(made(c) for c in alts):
            fails.append('none of ' + ', '.join(c['tool'] for c in alts) + ' was called as expected')
    for n in expect.get('not_calls') or []:
        if any(x['name'] == n for x in calls):
            fails.append(f'{n} was called, and must not be')
    for tier in expect.get('no_tier') or []:
        bad = [x['name'] for x in calls if (A.lookup(x['name']) is not None and A.lookup(x['name']).tier == tier)]
        if bad:
            fails.append(f'{tier} tools were called: {", ".join(dict.fromkeys(bad))}')
    for ce in expect.get('call_errors') or []:
        if not any((not ce.get('tool') or x['name'] == ce['tool']) and ce['text'].lower() in str(x.get('error') or
                                                                                                    '').lower()
                   for x in calls):
            fails.append(f'no {ce.get("tool") or "tool"} call returned an error containing "{ce["text"]}"')
    for ap in expect.get('approvals') or []:
        if not any(a['tool'] == ap['tool'] and (not ap.get('choice') or a['choice'] == ap['choice'])
                   for a in out.get('approvals') or []):
            fails.append(f'no approval card for {ap["tool"]}' + (f' answered {ap["choice"]}' if ap.get('choice')
                                                                  else ''))
    doc = out.get('doc') or {}
    for pc in expect.get('project') or []:
        try:
            items = _lookup(doc, pc['section'])
        except KeyError:
            items = []
        items = items if isinstance(items, list) else [items]
        hit = any(matches(pc.get('match') or {}, it) for it in items)
        if hit == bool(pc.get('absent')):
            fails.append(f'the project {"still has" if pc.get("absent") else "has no"} {pc["section"]} item matching '
                         f'{json.dumps(pc.get("match") or {})}')
    runs = set(out.get('runs') or ())
    for r in expect.get('runs_exist') or []:
        if r not in runs:
            fails.append(f'the run {r} is no longer in the workspace')
    for r in expect.get('runs_gone') or []:
        if r in runs:
            fails.append(f'the run {r} is still in the workspace')
    for rd in expect.get('reads') or []:
        res = read(rd['tool'], rd.get('args') or {}) if read is not None else {'error': 'no reader'}
        if not matches(rd['expect'], res):
            fails.append(f'{rd["tool"]} {json.dumps(rd.get("args") or {})} afterwards returned '
                         f'{json.dumps(res, default=str)[:300]}, not {json.dumps(rd["expect"])}')
    jobs = out.get('jobs') or []
    for jc in expect.get('jobs') or []:
        if 'id' in jc:
            pool = [j for j in jobs if j['id'] == jc['id']]
        else:
            pool = [j for j in jobs if j.get('mine')]
        pool = [j for j in pool if not jc.get('kind') or j['kind'] == jc['kind']]
        if not any(not jc.get('state') or j['state'] == jc['state'] for j in pool):
            what = f'job {jc["id"]}' if 'id' in jc else f'a {jc.get("kind") or ""} job the assistant queued'.replace(
                'a  job', 'a job')
            fails.append(f'{what} did not end {jc.get("state") or "at all"} (' +
                         (', '.join(f'{j["id"]} {j["kind"]} {j["state"]}' for j in pool) or 'none') + ')')
    ans = (out.get('answer') or '').lower()
    for t in expect.get('answer_contains') or []:
        if t.lower() not in ans:
            fails.append(f'the answer does not mention "{t}"')
    if expect.get('answer_any') and not any(t.lower() in ans for t in expect['answer_any']):
        fails.append('the answer mentions none of ' + ', '.join(f'"{t}"' for t in expect['answer_any']))
    for t in expect.get('answer_excludes') or []:
        if t.lower() in ans:
            fails.append(f'the answer mentions "{t}"')
    if not ans.strip():
        fails.append('there was no final answer')
    if expect.get('grounded', True) and out.get('ungrounded'):
        fails.append('numbers not from Falcon\'s data: ' + ', '.join(out['ungrounded'][:8]))
    return fails


# --------------------------------------------------------------------------------------------- the headless run
def host(ctx, approvals: dict):
    """A headless agent.Host (no window: the window tools are not offered) that answers approval cards by the task's
    policy (per tier) and records what the agent did."""
    from . import agent as A

    class _Host(A.Host):
        def __init__(self):
            super().__init__(ctx)
            self.events: list = []
            self.approvals: list = []
            self.errors: list = []

        def event(self, kind, payload):
            if kind == 'delta':
                return
            if kind == 'approval':
                choice = approvals.get(payload.tier, 'approve')
                self.approvals.append({'tool': payload.tool, 'tier': payload.tier, 'choice': choice})
                payload.decide(choice)
                return
            if kind == 'error':
                self.errors.append(payload)
            self.events.append((kind, payload))
    return _Host()


class _MemoryKeys:
    """The key for this run, held in memory only (never written to the throw-away folder)."""

    def __init__(self, key: str):
        self._key = key or ''

    def get(self) -> str:
        return self._key

    def has(self) -> bool:
        return bool(self._key)

    def where(self) -> str:
        return 'memory (this evaluation only)'


def service(settings, key: str, workspace_fn, timings: list):
    """An AssistantService whose client times every request (purpose, model, seconds) into timings."""
    from .service import AssistantService

    class _Timed(AssistantService):
        def client(self, purpose='agent', key=None, base_url=None):
            cl = super().client(purpose, key, base_url)
            chat = cl.chat

            def timed(*a, **kw):
                t0 = time.monotonic()
                try:
                    return chat(*a, **kw)
                finally:
                    timings.append({'purpose': kw.get('purpose', purpose), 'seconds': time.monotonic() - t0})
            cl.chat = timed
            return cl
    return _Timed(settings, _MemoryKeys(key), workspace_fn=workspace_fn)


def default_fake_carla() -> str:
    """tests/fake_carla beside the falcon_autonomy package (a source checkout), or ''."""
    from ..sim.server import falcon_source
    src = falcon_source()
    p = os.path.join(src, 'tests', 'fake_carla') if src else ''
    return p if p and os.path.isfile(os.path.join(p, 'carla', '__init__.py')) else ''


def _new_project(folder: str):
    """The throw-away project: the default world and rig with a 160 × 90 camera (fast on the fake carla), 3 traffic
    vehicles."""
    from ..core.project import Project
    p = Project.new('Evaluation')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=90)
    p.doc['traffic_profiles'][0]['vehicles'] = 3
    p.save(os.path.join(folder, 'eval.falcon.json'))
    return p


class _Env:
    """One task's world: the project, its workspace, a job queue with its own locks, the fake carla's service, the
    tool context."""

    def __init__(self, folder: str, fake_carla: str, settings):
        from ..core.jobs import JobQueue, SubprocessLauncher
        from ..core.settings import MEMORY_RESERVE_GB
        from ..sim.service import CarlaService
        from .privacy import redact
        from .tool_api import ToolContext
        self.project = _new_project(folder)
        self.workspace = os.path.realpath(self.project.workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.queue = JobQueue(SubprocessLauncher(), lock_dir=os.path.join(folder, 'locks'),
                              memory_reserve_gb=MEMORY_RESERVE_GB)
        self.service = CarlaService(settings, self.queue, self.workspace, fake_carla=fake_carla)
        self.ctx = ToolContext(project=self.project, settings=settings, queue=self.queue, service=self.service,
                               win=None, workspace=self.workspace, redact=redact)
        self._stop = threading.Event()
        self._poller = threading.Thread(target=self._poll, name='falcon-eval-queue', daemon=True)
        self._poller.start()

    def _poll(self):
        while not self._stop.wait(0.5):              # starts what was waiting (memory, locks); finishes poll themselves
            try:
                self.queue.poll()
            except Exception:                        # noqa: BLE001 (the next poll tries again)
                pass

    def call(self, name, args):
        from .tool_api import call
        return call(self.ctx, name, args)

    def wait_jobs(self, ids, timeout_s: float) -> bool:
        from ..core.jobs import FINAL
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            js = [self.queue.get(int(i)) for i in ids]
            if all(j is None or j.state in FINAL for j in js):
                return True
            time.sleep(0.2)
        return False

    def runs(self) -> list:
        from ..core import runindex as RI
        return [os.path.relpath(os.path.realpath(d), self.workspace) for d in RI.run_dirs(self.workspace)]

    def close(self):
        self._stop.set()
        try:
            self.service.shutdown(grace_s=10.0)
        except Exception:                            # noqa: BLE001 (tidying a throw-away folder)
            pass


class SetupError(Exception):
    pass


def _setup(env: _Env, steps: list, vars_: dict):
    for k, st in enumerate(steps or ()):
        name = st.get('as') or f'step{k}'
        if 'call' in st:
            res = env.call(st['call'], fill(st.get('args') or {}, vars_))
            if res.get('error'):
                raise SetupError(f'setup {st["call"]}: {res["error"]}')
            ids = [res['job']] if isinstance(res.get('job'), int) else [j['job'] for j in res.get('jobs') or []
                                                                         if isinstance(j, dict) and 'job' in j]
            if ids and st.get('wait', True):
                if not env.wait_jobs(ids, float(st.get('timeout_s') or SETUP_WAIT_S)):
                    raise SetupError(f'setup {st["call"]}: job {ids} did not finish in time')
                bad = [env.queue.get(i) for i in ids if env.queue.get(i).state != 'done']
                if bad:
                    raise SetupError(f'setup {st["call"]}: job {bad[0].id} ended {bad[0].state}')
            vars_[name] = res
        elif 'fail_drive' in st:
            vars_[name] = _fail_drive(env, st['fail_drive'])
        elif 'private_run' in st:
            vars_[name] = _private_run(env, st['private_run'])


def _fail_drive(env: _Env, args: dict) -> dict:
    """A drive queued as Data ▸ Record queues it (jobs_drive's config, the service's worker), whose fake server never
    answers (FAKE_CARLA_NO_SERVER; a 2 s connect wait): the worker exits 'no server', as with CARLA down."""
    from .tools import _common as C
    from .tools.jobs import _drive_cfg
    if not env.service.fake_carla:
        raise SetupError('fail_drive needs the fake carla')
    cfg, _r = _drive_cfg(env.ctx, **{k: v for k, v in args.items() if k in ('world', 'rig', 'seconds', 'seed',
                                                                             'name')})
    cfg['connect_wait_s'] = 2.0
    from ..core.paths import ensure_dir
    from ..core.provenance import write_json
    write_json(os.path.join(ensure_dir(cfg['out']), 'drive.json'), cfg)
    spec = env.service.worker_spec('drive', [os.path.join(cfg['out'], 'drive.json')], cfg.get('name', 'drive'),
                                   'drive', cfg['out'], meta={'config': os.path.join(cfg['out'], 'drive.json')})
    spec.env = dict(spec.env or {}, FAKE_CARLA_NO_SERVER='1')
    job = env.queue.submit(spec)
    if not env.wait_jobs([job.id], 120.0):
        raise SetupError('fail_drive: the job did not end')
    if job.state != 'failed':
        raise SetupError(f'fail_drive: the job ended {job.state}, not failed')
    return {'job': job.id, 'run': C.run_id(env.ctx, cfg['out']), 'state': job.state}


def _private_run(env: _Env, name: str) -> dict:
    """A run folder whose record says it is a drive along a dashcam route (the privacy marker; no footage)."""
    from ..core.provenance import write_json
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name or ''):
        raise SetupError(f'private_run: {name!r} is not a folder name')
    d = os.path.join(env.workspace, 'runs', name)
    os.makedirs(d, exist_ok=True)
    write_json(os.path.join(d, 'run.json'), {'name': name, 'worker': 'drive', 'status': 'ok',
                                             'provenance': {'settings': {'dashcam_route': True}}})
    return {'run': f'runs/{name}'}


def _outcome(env: _Env, ag, h, timings: list, turn_s: float, timed_out: bool) -> dict:
    """What the run produced, read from the conversation, the host's records, the queue and the project."""
    msgs = ag.session.snapshot()
    results = {m.get('tool_call_id'): m for m in msgs if m.get('role') == 'tool'}
    calls, answer, ungrounded, steps = [], '', [], 0
    for m in msgs:
        if m.get('role') != 'assistant':
            continue
        steps += 1
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function') or {}
            try:
                args = json.loads(fn.get('arguments') or '{}')
            except ValueError:
                args = None
            r = results.get(tc.get('id')) or {}
            err = None
            try:
                rd = json.loads(r.get('content') or '{}')
                err = rd.get('error') if isinstance(rd, dict) else None
            except ValueError:
                pass
            calls.append({'name': fn.get('name'), 'args': args if isinstance(args, dict) else {}, 'error': err})
        if not m.get('tool_calls'):
            answer = m.get('content') or ''
            ungrounded += list(m.get('_ungrounded') or [])
    usage = [p for k, p in h.events if k == 'usage']
    costs = [p.get('cost') for p in usage]
    mine = {a['job_id'] for a in ag.session.activity if a.get('kind') == 'job'}
    jobs = [{'id': j.id, 'kind': j.spec.kind, 'state': j.state, 'title': j.spec.title, 'mine': j.id in mine}
            for j in env.queue.jobs]
    return {'calls': calls, 'answer': answer, 'ungrounded': list(dict.fromkeys(ungrounded)),
            'approvals': list(h.approvals), 'jobs': jobs, 'doc': copy.deepcopy(env.project.doc), 'runs': env.runs(),
            'steps': sum(1 for t in timings if t['purpose'] == 'agent'),
            'assistant_messages': steps,
            'tokens': {'prompt': sum(int(p.get('prompt') or 0) for p in usage),
                       'completion': sum(int(p.get('completion') or 0) for p in usage)},
            'cost': round(sum(c for c in costs if c is not None), 6) if usage and None not in costs else None,
            'unpriced_requests': sum(1 for c in costs if c is None),
            'seconds': round(turn_s, 2),
            'model_seconds': round(sum(t['seconds'] for t in timings), 2),
            'errors': [e.get('message') for e in h.errors], 'timed_out': timed_out}


def run_task(task: dict, cfg: dict, key: str, fake_carla: str, workdir: str, script_sink=None, log=None,
             audit_dir: str = '') -> dict:
    """One task, start to end, on its own throw-away project; the record for results.json. The requests' audit log
    (session.Audit: no payloads) is kept as audit_dir/<model>--<task>.jsonl when audit_dir is given."""
    from ..core.settings import Settings
    from . import agent as A
    from .session import Session
    folder = tempfile.mkdtemp(prefix=f'{task["id"]}-', dir=workdir)
    settings = Settings(os.path.join(folder, 'settings.json'))           # never the user's file: a copy of its values
    settings.data['assistant'] = copy.deepcopy(cfg)
    rec = {'task': task['id'], 'title': task['title'], 'model': cfg.get('main_model'), 'success': False,
           'failures': []}
    env = ag = None
    try:
        env = _Env(folder, fake_carla, settings)
        vars_: dict = {}
        try:
            _setup(env, task.get('setup'), vars_)
            prompt = fill(task['prompt'], vars_)
            expect = fill(task['expect'], vars_)
            script = fill(task.get('script') or [], vars_)
        except (SetupError, KeyError) as e:
            msg = str(e.args[0]) if e.args else type(e).__name__
            rec.update(setup_error=msg, failures=[f'setup: {msg}'])
            return rec
        if script_sink is not None:
            script_sink(script)
        timings: list = []
        svc = service(settings, key, lambda: env.workspace, timings)
        h = host(env.ctx, dict({'act': 'approve', 'trash': 'reject'}, **(task.get('approvals') or {})))
        ag = A.Agent(svc, Session(env.workspace), h, task.get('mode') or 'auto')
        content, why = A.compose(prompt)
        t0 = time.monotonic()
        if content is None:
            rec.update(failures=[f'Falcon refused the request before sending it: {why}'], refused=why)
            return rec
        timeout = float(task.get('timeout_s') or 300)
        ag.start(content)
        done = ag.wait(timeout)
        turn_s = time.monotonic() - t0
        if not done:
            ag.stop()
            ag.wait(15.0)
        if done and expect.get('jobs'):                  # the agent may answer while its jobs still run
            left = max(5.0, timeout - (time.monotonic() - t0))
            env.wait_jobs([j.id for j in env.queue.jobs], left)
        out = _outcome(env, ag, h, timings, turn_s, not done)
        fails = check(expect, out, read=env.call)
        if not done:
            fails.insert(0, f'the turn did not end within {timeout:g} s (stopped)')
        if out['errors']:
            fails.append('the assistant stopped on an error: ' + '; '.join(str(e) for e in out['errors'])[:300])
        from .privacy import redact_text
        rec.update(success=not fails, failures=fails, steps=out['steps'], tool_calls=[c['name'] for c in out['calls']],
                   tokens=out['tokens'], cost=out['cost'], unpriced_requests=out['unpriced_requests'],
                   seconds=out['seconds'], model_seconds=out['model_seconds'], grounding_flags=out['ungrounded'],
                   approvals=out['approvals'], timed_out=out['timed_out'],
                   answer=redact_text(out['answer'])[:1200])
        return rec
    except Exception as e:                           # noqa: BLE001 (one broken task must not end the evaluation)
        rec.update(success=False, failures=[f'the harness failed on this task: {type(e).__name__}: {e}'[:400]],
                   internal_error=type(e).__name__)
        return rec
    finally:
        if ag is not None:
            ag.close()
        if env is not None:
            env.close()
            src = os.path.join(env.workspace, 'assistant', 'audit.jsonl')
            if audit_dir and os.path.isfile(src):
                os.makedirs(audit_dir, exist_ok=True)
                shutil.copyfile(src, os.path.join(audit_dir, re.sub(r'[^\w.-]+', '_', str(rec['model']))
                                                  + f'--{task["id"]}.jsonl'))
        shutil.rmtree(folder, ignore_errors=True)
        from .privacy import redact
        rec.update(redact({k: rec[k] for k in ('failures', 'setup_error') if k in rec}))
        if log is not None:
            log(f'{"PASS" if rec["success"] else "FAIL"} {rec["model"]} {task["id"]}'
                + ('' if rec['success'] else ': ' + '; '.join(rec['failures'])[:300]))


def summarise(records: list) -> dict:
    """Per model: tasks, passed, success rate, mean steps, tokens (total, mean), mean wall and model seconds, cost
    (None when a request had no price), tasks with grounding flags, setup errors."""
    out = {}
    for m in dict.fromkeys(r['model'] for r in records):
        rs = [r for r in records if r['model'] == m]
        ran = [r for r in rs if 'steps' in r]
        n = len(rs)
        costs = [r.get('cost') for r in ran]

        def mean(key, rows=ran):
            v = [float(r[key]) for r in rows if r.get(key) is not None]
            return round(sum(v) / len(v), 2) if v else None
        tok = [r['tokens']['prompt'] + r['tokens']['completion'] for r in ran]
        out[m] = {'tasks': n, 'passed': sum(r['success'] for r in rs),
                  'success_rate': round(sum(r['success'] for r in rs) / n, 3) if n else None,
                  'mean_steps': mean('steps'), 'tokens': sum(tok), 'mean_tokens': round(sum(tok) / len(tok)) if tok
                  else None, 'mean_seconds': mean('seconds'), 'mean_model_seconds': mean('model_seconds'),
                  'cost': round(sum(costs), 6) if costs and None not in costs else None,
                  'unpriced_tasks': sum(1 for c in costs if c is None),
                  'grounding_flagged_tasks': sum(1 for r in ran if r.get('grounding_flags')),
                  'setup_errors': sum(1 for r in rs if r.get('setup_error')),
                  'timed_out': sum(1 for r in ran if r.get('timed_out'))}
    return out


def markdown(summary: dict, records: list, meta: dict) -> str:
    lines = ['# Assistant evaluation', '',
             f'{meta["when"]} · server {meta["server"]} · {meta["tasks"]} tasks · Falcon {meta["falcon"]}', '',
             'Each task ran on a throw-away project with the fake carla. Tokens are the server\'s counts; cost is an '
             'estimate from the price table in Settings ▸ Assistant (blank: a model without a price).', '',
             '| model | success | mean steps | tokens (mean) | mean s (model s) | est. cost $ | grounding flags | '
             'timed out |', '|---|---|---|---|---|---|---|---|']
    for m, s in summary.items():
        cost = '' if s['cost'] is None else f'{s["cost"]:.4f}'
        lines.append(f'| `{m}` | {s["passed"]}/{s["tasks"]} ({100 * (s["success_rate"] or 0):.0f} %) | '
                     f'{s["mean_steps"] if s["mean_steps"] is not None else "–"} | {s["tokens"]} '
                     f'({s["mean_tokens"] if s["mean_tokens"] is not None else "–"}) | '
                     f'{s["mean_seconds"] if s["mean_seconds"] is not None else "–"} '
                     f'({s["mean_model_seconds"] if s["mean_model_seconds"] is not None else "–"}) | '
                     f'{cost} | '
                     f'{s["grounding_flagged_tasks"]} | {s["timed_out"]} |')
    for m in summary:
        lines += ['', f'## `{m}`', '', '| task | result | steps | tokens | s | why it failed |',
                  '|---|---|---|---|---|---|']
        for r in records:
            if r['model'] != m:
                continue
            tok = (r['tokens']['prompt'] + r['tokens']['completion']) if r.get('tokens') else '–'
            why = '; '.join(r.get('failures') or []).replace('|', '\\|').replace('\n', ' ')[:300]
            lines.append(f'| `{r["task"]}` | {"pass" if r["success"] else "FAIL"} | {r.get("steps", "–")} | {tok} | '
                         f'{r.get("seconds", "–")} | {why} |')
    return '\n'.join(lines) + '\n'


def run_eval(tasks: list, cfg: dict, key: str, out: str, models=(), fake_carla: str = '', only=(), script_sink=None,
             log=print) -> dict:
    """Every task (or those in only) for every model (default the configured main model); results.json and summary.md
    written into out. cfg: the assistant's configuration to use (a copy; nothing is saved anywhere else); key: the API
    key, kept in memory. script_sink(script): tests only — hands each task's scripted responses to the fake server."""
    from .. import __version__
    import urllib.parse
    bad = {t.get('id', '?'): validate_task(t) for t in tasks}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise ValueError('invalid tasks: ' + '; '.join(f'{k}: {", ".join(v[:3])}' for k, v in bad.items()))
    fake_carla = fake_carla or default_fake_carla()
    if not fake_carla:
        raise ValueError('the evaluation runs its jobs on the fake carla (tests/fake_carla in Falcon\'s source '
                         'folder), which this installation does not have')
    sel = [t for t in tasks if not only or t['id'] in only]
    if only and len(sel) != len(set(only)):
        raise ValueError('no task named ' + ', '.join(sorted(set(only) - {t['id'] for t in sel})))
    models = list(models) or [cfg.get('main_model')]
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix='falcon-assistant-eval-')
    records = []
    try:
        for m in models:
            c = dict(copy.deepcopy(cfg), main_model=m)
            for t in sel:
                records.append(run_task(t, c, key, fake_carla, work, script_sink, log, os.path.join(out, 'audit')))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    summary = summarise(records)
    meta = {'when': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'falcon': __version__,
            'server': urllib.parse.urlsplit(cfg.get('base_url') or '').netloc, 'tasks': len(sel), 'models': models,
            'prices': {m: (cfg.get('prices') or {}).get(m) for m in models}}
    doc = {'meta': meta, 'summary': summary, 'results': records}
    from ..core.paths import atomic_write
    atomic_write(os.path.join(out, 'results.json'), json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    md = markdown(summary, records, meta)
    atomic_write(os.path.join(out, 'summary.md'), md)
    return doc


def main_eval(a) -> int:
    """The CLI's `falcon assistant eval` (assistant.cli parses the arguments): the user's assistant configuration and
    key, read and never written; refused without them (nothing is sent)."""
    import sys
    from ..core.settings import Settings
    from .keystore import KeyStore
    from .service import AssistantService
    user = Settings()
    svc = AssistantService(user, KeyStore())
    ok, why = svc.ready()
    if not ok:
        print(f'falcon assistant eval: nothing was sent. {why}', file=sys.stderr)
        return 2
    try:
        tasks = load_tasks(a.tasks)
    except (OSError, ValueError) as e:
        print(f'falcon assistant eval: cannot read the tasks: {e}', file=sys.stderr)
        return 2
    out = a.out or os.path.join(os.getcwd(), f'assistant-eval-{time.strftime("%Y%m%d-%H%M%S")}')
    only = [x for x in (a.only or '').split(',') if x]
    try:
        doc = run_eval(tasks, svc.cfg(), svc.keys.get(), out, a.model or (), a.fake_carla, only,
                       log=lambda s: print(s, file=sys.stderr))
    except ValueError as e:
        print(f'falcon assistant eval: {e}', file=sys.stderr)
        return 2
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as fh:
        sys.stdout.write(fh.read())
    print(f'\nresults: {os.path.join(out, "results.json")}', file=sys.stderr)
    return 0 if all(r['success'] for r in doc['results']) else 1
