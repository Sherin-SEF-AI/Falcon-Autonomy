"""Real-time suggestions (Qt-free): the next useful actions in this project, found from its own state and files, each
carrying the exact tool call that would act on it (docs/assistant.md, "Suggestions").

gather(state) runs every trigger (TRIGGERS) over a snapshot of the app (State: the workspace, the project, this
session's jobs, the server, the GPU, the last dataset-health report) and the workspace's files, and returns candidates
(Suggestion): a stable key (what is dismissed), the trigger, a short title for the chip, a one-line `why` whose numbers
are read from the files (the facts it was built from are kept beside it), the tool and arguments that act on it, a
priority and the pages it belongs on. No model is needed: the deterministic wording always works, offline.

phrase(service, candidates) asks the FAST model (when the assistant is on and its notice accepted) to rank the top few
and word each in one short sentence, keeping every number as given. Each wording goes through the grounding check
against that candidate's own facts; a wording with a number the facts do not hold is dropped for the deterministic one.

Pacer keeps it quiet: a gather at most every MIN_GATHER_S (an event — a job finishing, the project changing — may
bring it forward to MIN_FORCED_S), a phrasing request at most every PHRASE_EVERY_S and only for candidates not worded
yet, at most MAX_PER_PAGE chips on a page. Store remembers what the user dismissed, per project
(WORKSPACE/assistant/suggestions.json).

Privacy: runs, studies and worlds the privacy rules call private (anything from the dashcam footage) never become a
suggestion, so their names never reach the chips or the model.

Triggers (key: what it looks at → the action):
    server_crashed   Falcon's CARLA server crashed → system_server (its state and log tail)
    job_failed       a job of this session failed, with its error → jobs_get with the log tail
    avoidable        a rare-case study holds failures RSS labels avoidable → jobs_replay_failure of the worst
    mrm_failed       a stack suite's runs failed the minimal-risk-manoeuvre check → jobs_why on the worst one
    gpu_idle         CARLA jobs wait while none runs and the GPU is idle → system_queue (who holds the lock)
    world_lights     an imported world's traffic lights control no lane, or a left-hand world was converted by an
                     earlier version → jobs_apply_map_edits (a re-import with the current converter)
    coverage_gap     the recorded drives leave condition cells (world × time of day × weather) empty → jobs_drive
                     with the preset that fills the first one
    duplicates       the last Data ▸ Health report found many near-duplicate frames → runs_tag them
    drift_refused    a translated run kept raw frames its drift check refused → ui_show_run
    job_done         a job of this session finished → ui_open_job
    first_drive      worlds and rigs but nothing recorded → jobs_drive
    no_evaluation    models and drives but no evaluation → jobs_evaluate
    no_scenarios     no scenario of the project's own → project_scenario_create (a copy of lead_brake)
    no_rules         no trigger rules → project_trigger_add (an example rule, parsed when added)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from . import grounding
from .privacy import private_reason

MIN_GATHER_S = 30.0           # between two gathers (a timer)
MIN_FORCED_S = 3.0            # between two gathers brought forward by an event
PHRASE_EVERY_S = 120.0        # between two requests to the fast model
PHRASE_TOP = 4                # candidates the model ranks and words
MAX_PER_PAGE = 3              # chips on one page
TEXT_CAP = 160                # characters of a wording
RECENT_DONE_S = 15 * 60.0     # a finished job is suggested for this long
GPU_IDLE_PCT = 10.0
STUDIES_READ = 12             # newest studies, suites and translated runs read per gather
EXAMPLE_RULE = 'when ego.speed > 40 and visibility < 30 -> spawn jaywalker'

P_SERVER, P_FAILED, P_AVOIDABLE, P_MRM, P_GPU, P_WORLD = 95, 90, 72, 70, 62, 56
P_COVERAGE, P_DUPS, P_DRIFT, P_DONE, P_FIRST, P_EVAL, P_SCN, P_RULES = 50, 46, 40, 38, 34, 30, 24, 18


@dataclass
class Suggestion:
    key: str                  # what it is about, stable across gathers (dismissed by it): '<trigger>:<object>'
    trigger: str
    title: str                # the chip's label (≤ 48 characters)
    why: str                  # one line, its numbers from the files
    tool: str
    args: dict
    priority: int
    pages: tuple = ()         # the pages it shows on (Home shows the best of all)
    facts: dict = field(default_factory=dict)
    text: str = ''            # what is shown: the model's grounded wording, else why
    phrased: bool = False
    rank: int = 1000          # the model's ranking (lower first), else unranked

    def shown(self) -> str:
        return self.text or self.why

    def action(self) -> dict:
        """What AssistantService.propose takes."""
        return {'tool': self.tool, 'args': dict(self.args), 'why': self.shown()}

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ('key', 'trigger', 'title', 'why', 'tool', 'args', 'priority', 'pages',
                                              'facts', 'text', 'phrased')}


@dataclass
class State:
    """A snapshot of the app for gather(): plain data (job_facts makes a job's), taken on the GUI thread."""
    workspace: str = ''
    doc: dict = field(default_factory=dict)
    jobs: list = field(default_factory=list)
    server: dict = field(default_factory=dict)          # {'state', 'error'}
    gpu_util: float | None = None                       # percent, measured now (None: no GPU or not measured)
    health: dict | None = None                          # the last Data ▸ Health report (core.datahealth.analyse)
    now: float = 0.0


def job_facts(job, page: str = '') -> dict:
    """A core.jobs.Job as plain data: id, kind, title, state, exit (what its exit code means), the last error line,
    whether it needs CARLA or the GPU, why it waits, its run folder, when it finished, the page it was started from."""
    from ..core import protocol as P
    errs = [t for lvl, t in job.log if lvl == 'error' and t] or \
        [t for lvl, t in job.log if lvl == 'stderr' and str(t).strip()]
    return {'id': job.id, 'kind': job.spec.kind, 'title': job.spec.title, 'state': job.state,
            'exit_code': job.exit_code, 'exit': P.describe_exit(job.exit_code) if job.exit_code is not None else '',
            'error': str(errs[-1]).strip()[:300] if errs else '', 'needs_carla': bool(job.spec.needs_carla),
            'needs_gpu': bool(job.spec.needs_gpu), 'wait_reason': job.wait_reason or '',
            'run_dir': job.spec.run_dir or '', 'finished': job.finished, 'elapsed_s': round(job.elapsed, 1),
            'page': page}


# --------------------------------------------------------------------------------------------- reading files
def _json(path: str, default=None):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _jsonl(path: str) -> list:
    out = []
    try:
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _inside(p: str, root: str) -> bool:
    p, r = os.path.realpath(p), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def _rel(p: str, ws: str) -> str:
    return os.path.relpath(os.path.realpath(p), os.path.realpath(ws))


def _newest(root: str, must: str, n: int = STUDIES_READ) -> list:
    """The newest folders under root that hold `must`, not private."""
    if not root or not os.path.isdir(root):
        return []
    ds = [os.path.join(root, d) for d in os.listdir(root) if not d.startswith('.')
          and os.path.isfile(os.path.join(root, d, must))]
    ds.sort(key=lambda d: os.path.getmtime(os.path.join(d, must)), reverse=True)
    return [d for d in ds[:n] if not private_reason(d)]


def _entries(state: State) -> list:
    """The run index's entries inside the workspace that are not private (cached in the state for one gather)."""
    if '_entries' in state.__dict__:
        return state.__dict__['_entries']
    out = []
    if state.workspace and os.path.isdir(state.workspace):
        from ..core import runindex as RI
        try:
            es = RI.build(state.workspace)
        except OSError:
            es = []
        out = [e for e in es if _inside(e['path'], state.workspace) and not e.get('dashcam_route')
               and not private_reason(e['path'])]
    state.__dict__['_entries'] = out
    return out


def _fmt(v, nd: int = 2) -> str:
    return f'{v:.{nd}f}'.rstrip('0').rstrip('.') if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------------------------- the triggers
def t_server_crashed(state: State) -> list:
    if (state.server or {}).get('state') != 'crashed':
        return []
    err = str((state.server or {}).get('error') or '').strip().splitlines()
    why = 'Falcon\'s CARLA server crashed' + (f': {err[0][:140]}' if err else '') + \
        ' — its log tail says what happened; the next CARLA job starts it again.'
    return [Suggestion('server_crashed:server', 'server_crashed', 'See why the CARLA server crashed', why,
                       'system_server', {}, P_SERVER, ('home', 'live', 'data'), {'state': 'crashed'})]


def t_job_failed(state: State) -> list:
    out = []
    failed = [j for j in state.jobs if j.get('state') == 'failed']
    for j in sorted(failed, key=lambda j: -(j.get('finished') or 0))[:2]:
        what = j.get('error') or j.get('exit') or 'no error line logged'
        why = f'Job {j["id"]} "{j["title"]}" failed — {j.get("exit") or "exit unknown"}' + \
            (f' Last error: {what[:140]}' if j.get('error') else '')
        out.append(Suggestion(f'job_failed:{j["id"]}', 'job_failed', f'Look into failed job {j["id"]}', why,
                              'jobs_get', {'job': int(j['id']), 'log_lines': 30}, P_FAILED,
                              tuple(dict.fromkeys(('home', j.get('page') or 'home'))),
                              {'job': j['id'], 'exit_code': j.get('exit_code')}))
    return out


def t_job_done(state: State) -> list:
    now = state.now or time.time()
    done = [j for j in state.jobs if j.get('state') == 'done' and j.get('finished')
            and now - j['finished'] <= RECENT_DONE_S and j.get('kind') not in ('probe', 'tidy')]
    out = []
    for j in sorted(done, key=lambda j: -j['finished'])[:1]:
        why = f'Job {j["id"]} "{j["title"]}" finished after {_fmt(float(j.get("elapsed_s") or 0.0), 0)} s.'
        out.append(Suggestion(f'job_done:{j["id"]}', 'job_done', f'Open the result of job {j["id"]}', why,
                              'ui_open_job', {'job': int(j['id'])}, P_DONE,
                              tuple(dict.fromkeys(('home', j.get('page') or 'home'))),
                              {'job': j['id'], 'elapsed_s': j.get('elapsed_s')}))
    return out


def t_avoidable(state: State) -> list:
    """The newest study holding failures the ego could have avoided (core.rss label 'avoidable', criticality at the
    study's threshold or above): replay the worst."""
    ws = state.workspace
    for d in _newest(os.path.join(ws, 'studies') if ws else '', 'results.jsonl'):
        st = _json(os.path.join(d, 'study.json'), {}) or {}
        thr = 0.5
        for f in sorted(os.listdir(d)):
            if f.startswith('search-') and f.endswith('.json'):
                thr = float((_json(os.path.join(d, f), {}) or {}).get('threshold') or thr)
        rows = _jsonl(os.path.join(d, 'results.jsonl'))
        av = []
        for i, r in enumerate(rows):
            crit = r.get('criticality', r.get('objective'))
            if (r.get('rss') or {}).get('label') == 'avoidable' and isinstance(crit, (int, float)) and crit >= thr:
                if r.get('run') and private_reason(r['run']):
                    continue
                av.append((float(crit), i))
        if not av:
            continue
        crit, i = max(av, key=lambda c: (c[0], -c[1]))
        sid = os.path.basename(d)
        scn = st.get('scenario')
        scn = scn.get('id') if isinstance(scn, dict) else scn
        why = (f'{len(av)} of {len(rows)} runs of study {sid}' + (f' ({scn})' if scn else '') +
               f' are failures the ego could have avoided (RSS); the worst, row {i}, reached criticality '
               f'{_fmt(round(crit, 3), 3)}.')
        return [Suggestion(f'avoidable:{sid}', 'avoidable', 'Replay the worst avoidable failure', why,
                           'jobs_replay_failure', {'study': sid, 'index': i}, P_AVOIDABLE, ('rarecases', 'home'),
                           {'avoidable': len(av), 'runs': len(rows), 'index': i, 'criticality': round(crit, 3),
                            'threshold': thr})]
    return []


def t_mrm_failed(state: State) -> list:
    ws = state.workspace
    from ..core import stack as ST
    for d in _newest(os.path.join(ws, 'suites') if ws else '', 'results.jsonl'):
        try:
            rows = ST.read_results(d)
        except (OSError, ValueError, KeyError):
            continue
        mrm = [r for r in rows if r.get('status') == 'ok' and r.get('mrm')]
        bad = [r for r in mrm if not (r['mrm'] or {}).get('passed')]
        if not bad:
            continue
        checks: dict = {}
        for r in bad:
            for c in r['mrm'].get('checks') or ():
                if not c.get('ok'):
                    checks[c.get('name', '?')] = checks.get(c.get('name', '?'), 0) + 1
        top = max(checks.items(), key=lambda kv: (kv[1], kv[0])) if checks else None
        sid = os.path.basename(d)
        why = f'{len(bad)} of {len(mrm)} runs of suite {sid} failed the minimal-risk-manoeuvre check' + \
            (f' (most often: {top[0]}, {top[1]} times).' if top else '.')
        worst = max(bad, key=lambda r: (r.get('criticality') or 0.0))
        run = worst.get('run') or ''
        facts = {'failed': len(bad), 'runs': len(mrm), 'checks': checks}
        if run and os.path.isfile(os.path.join(run, 'run.json')) and _inside(run, ws) and not private_reason(run):
            why += f' "Why did it fail?" re-runs the worst ({worst.get("scenario")}, seed {worst.get("seed")}) ' \
                   'removing one factor at a time.'
            facts.update(seed=worst.get('seed'))
            return [Suggestion(f'mrm_failed:{sid}', 'mrm_failed', 'Ask why the MRM check failed', why, 'jobs_why',
                               {'run': _rel(run, ws)}, P_MRM, ('stack', 'home'), facts)]
        return [Suggestion(f'mrm_failed:{sid}', 'mrm_failed', 'Read the failed MRM checks', why, 'analysis_suite',
                           {'id': sid}, P_MRM, ('stack', 'home'), facts)]
    return []


def t_gpu_idle(state: State) -> list:
    waiting = [j for j in state.jobs if j.get('needs_carla') and j.get('state') in ('queued', 'waiting')]
    running = [j for j in state.jobs if j.get('needs_carla') and j.get('state') == 'running']
    if not waiting or running or state.gpu_util is None or state.gpu_util >= GPU_IDLE_PCT:
        return []
    reason = next((j['wait_reason'] for j in waiting if j.get('wait_reason')), '')
    why = f'{len(waiting)} CARLA job(s) wait while the GPU is at {_fmt(float(state.gpu_util), 0)} % and none runs' + \
        (f': {reason[:120]}' if reason else '') + ' — the queue says who holds the CARLA lock.'
    return [Suggestion('gpu_idle:queue', 'gpu_idle', 'See why CARLA jobs wait', why, 'system_queue', {}, P_GPU,
                       ('home', 'data', 'rarecases', 'stack'), {'waiting': len(waiting),
                                                                'gpu_util_pct': state.gpu_util})]


def t_world_lights(state: State) -> list:
    ws, out = state.workspace, []
    for w in (state.doc or {}).get('worlds') or ():
        if w.get('kind') != 'osm' or not w.get('xodr') or not ws:
            continue
        d = os.path.join(ws, os.path.dirname(w['xodr']))
        if private_reason(d):
            continue
        rep = _json(os.path.join(d, 'import.json'))
        if not isinstance(rep, dict):
            continue
        sl = rep.get('signal_lanes')
        name = w.get('name') or w['id']
        if isinstance(sl, dict) and sl.get('no_lane'):
            why = (f'{name}: {sl["no_lane"]} of {sl.get("oriented")} traffic-light records control no lane in CARLA, '
                   'so traffic drives through them; a re-import with the current converter corrects it (a new world).')
            facts = {'no_lane': sl['no_lane'], 'oriented': sl.get('oriented')}
        elif sl is None and str(rep.get('handedness', '')).upper() == 'LHT':
            why = (f'{name} is a left-hand world converted by an earlier version: its traffic lights may control no '
                   'lane; a re-import with the current converter makes a corrected copy.')
            facts = {'handedness': 'LHT'}
        else:
            continue
        out.append(Suggestion(f'world_lights:{w["id"]}', 'world_lights', f'Re-import {name[:28]}', why,
                              'jobs_apply_map_edits', {'world': w['id']}, P_WORLD, ('worlds', 'home'), facts))
    return out[:2]


def t_coverage_gap(state: State) -> list:
    """Condition cells the recorded drives leave empty, from the run index alone (frames per world, time of day and
    weather kind; core.datahealth.coverage) — no image is read."""
    from ..core import datahealth as DH
    es = [e for e in _entries(state) if e.get('status') == 'ok' and (e.get('frames') or 0) > 0
          and e.get('kind') not in ('scenario', 'search', 'route')]
    if not es or not (state.doc or {}).get('worlds'):
        return []
    grid: dict = {}
    for e in es:
        cell = (e.get('world') or e.get('map') or '?', e.get('tod') or '?', e.get('weather_kind') or '?')
        grid[cell] = grid.get(cell, 0) + int(e.get('frames') or 0)
    cov = DH.coverage(grid, state.doc)
    first = next((c for c in cov['suggestions'] if c['frames'] == 0 and c.get('preset')), None)
    if first is None:
        return []
    frames = sum(grid.values())
    why = (f'{first["world_name"]} has no frames at {first["tod"]}, {first["weather"]}: {cov["empty"]} of '
           f'{cov["of"]} condition cells are empty over {frames} frames in {len(es)} drives. A drive with '
           f'{first["preset"]} fills it.')
    return [Suggestion(f'coverage_gap:{first["world"]}:{first["tod"]}:{first["weather"]}', 'coverage_gap',
                       f'Record {first["tod"]}, {first["weather"]} in {first["world_name"][:20]}', why, 'jobs_drive',
                       {'world': first['world'], 'weather_preset': first['preset'], 'seconds': 30.0}, P_COVERAGE,
                       ('data', 'home'), {'empty': cov['empty'], 'of': cov['of'], 'frames': frames,
                                          'drives': len(es)})]


def t_duplicates(state: State) -> list:
    rep = state.health or {}
    dup = rep.get('duplicates') or {}
    share, n = dup.get('share'), dup.get('near_duplicates') or 0
    if not share or share < 0.2 or n < 10 or not state.workspace:
        return []
    runs = []
    for ex in dup.get('examples') or ():
        p = ex.get('run')
        if p and _inside(p, state.workspace) and not private_reason(p) and _rel(p, state.workspace) not in runs:
            runs.append(_rel(p, state.workspace))
    if not runs:
        return []
    why = (f'{n} of {dup.get("hashed")} sampled frames ({_fmt(round(100 * share, 1), 1)} %) are near-duplicates '
           f'(within {dup.get("bits")} bits, dHash), mostly in {len(runs)} run(s); a tag lets training sets leave '
           'them out (-tag:near-duplicates).')
    return [Suggestion('duplicates:' + ','.join(sorted(runs)), 'duplicates', 'Tag the runs with near-duplicates', why,
                       'runs_tag', {'runs': runs, 'tags': ['near-duplicates']}, P_DUPS, ('data',),
                       {'near_duplicates': n, 'hashed': dup.get('hashed'), 'share_pct': round(100 * share, 1),
                        'bits': dup.get('bits'), 'runs': len(runs)})]


def t_drift_refused(state: State) -> list:
    ws = state.workspace
    root = os.path.join(ws, 'runs') if ws else ''
    if not root or not os.path.isdir(root):
        return []
    cands = [os.path.join(root, d) for d in os.listdir(root) if '-tr-' in d
             and os.path.isfile(os.path.join(root, d, 'run.json'))]
    cands.sort(key=lambda d: os.path.getmtime(os.path.join(d, 'run.json')), reverse=True)
    for d in cands[:STUDIES_READ]:
        if private_reason(d):
            continue
        dr = ((_json(os.path.join(d, 'run.json'), {}) or {}).get('translation') or {}).get('drift') or {}
        if not dr.get('refused'):
            continue
        rid = _rel(d, ws)
        why = (f'{dr["refused"]} of {dr.get("frames")} frames of {rid} were refused by the drift check (drift over '
               f'{_fmt(dr.get("threshold"), 3)}) and stay raw.')
        return [Suggestion(f'drift_refused:{rid}', 'drift_refused', 'Look at the refused translated frames', why,
                           'ui_show_run', {'run': rid}, P_DRIFT, ('training', 'data'),
                           {'refused': dr['refused'], 'frames': dr.get('frames'), 'threshold': dr.get('threshold')})]
    return []


def t_first_drive(state: State) -> list:
    d = state.doc or {}
    if not d.get('worlds') or not d.get('rigs') or any(e.get('status') == 'ok' for e in _entries(state)):
        return []
    w, r = d['worlds'][0], d['rigs'][0]
    why = (f'Nothing is recorded in this project yet: a 10 s drive in {w.get("name")} with the rig {r.get("name")} '
           'gives images, labels and ground truth to work with.')
    return [Suggestion('first_drive:project', 'first_drive', 'Record a first 10 s drive', why, 'jobs_drive',
                       {'world': w['id'], 'rig': r['id'], 'seconds': 10.0}, P_FIRST, ('data', 'home'),
                       {'seconds': 10})]


def t_no_evaluation(state: State) -> list:
    d, ws = state.doc or {}, state.workspace
    models = d.get('models') or []
    root = os.path.join(ws, 'evaluations') if ws else ''
    if not models or (root and os.path.isdir(root) and any(
            os.path.isfile(os.path.join(root, x, 'evaluation.json')) for x in os.listdir(root))):
        return []
    drives = [e for e in _entries(state) if e.get('status') == 'ok' and e.get('kind') == 'drive'][:3]
    if not drives:
        return []
    m = models[0]
    runs = [e['rel'] for e in drives]
    why = f'No evaluation yet: score {m.get("name") or m["id"]} on the {len(runs)} newest drive(s), with intervals ' \
          'over whole drives.'
    return [Suggestion(f'no_evaluation:{m["id"]}', 'no_evaluation', f'Evaluate {str(m.get("name") or m["id"])[:28]}',
                       why, 'jobs_evaluate', {'model': m['id'], 'runs': runs}, P_EVAL, ('home',),
                       {'drives': len(runs)})]


def t_no_scenarios(state: State) -> list:
    d = state.doc or {}
    if d.get('scenarios') or not d.get('worlds'):
        return []
    why = 'The project has no scenario of its own: an editable copy of the built-in lead_brake can be changed, swept ' \
          'and searched for failures.'
    return [Suggestion('no_scenarios:project', 'no_scenarios', 'Make an editable scenario', why,
                       'project_scenario_create', {'copy_of': 'lead_brake', 'name': 'Lead brake (editable)'}, P_SCN,
                       ('scenarios', 'home'), {})]


def t_no_rules(state: State) -> list:
    d = state.doc or {}
    if d.get('trigger_rules') or not d.get('worlds'):
        return []
    why = f'No trigger rules yet: a rule makes drives react to what happens — e.g. "{EXAMPLE_RULE}" (checked when ' \
          'added; edit it after).'
    return [Suggestion('no_rules:project', 'no_rules', 'Add a trigger rule', why, 'project_trigger_add',
                       {'rule': EXAMPLE_RULE, 'name': 'Jaywalker in poor visibility'}, P_RULES, ('scenarios',),
                       {'rule': EXAMPLE_RULE})]


TRIGGERS = (t_server_crashed, t_job_failed, t_avoidable, t_mrm_failed, t_gpu_idle, t_world_lights, t_coverage_gap,
            t_duplicates, t_drift_refused, t_job_done, t_first_drive, t_no_evaluation, t_no_scenarios, t_no_rules)


def gather(state: State, triggers=TRIGGERS) -> list:
    """Every trigger's candidates, best first (priority, then key). A trigger that fails on a file it cannot read gives
    nothing; the others still answer."""
    if not state.now:
        state.now = time.time()
    out = []
    for t in triggers:
        try:
            out.extend(t(state))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    seen, uniq = set(), []
    for s in sorted(out, key=lambda s: (-s.priority, s.key)):
        if s.key not in seen:
            seen.add(s.key)
            uniq.append(s)
    return uniq


def for_page(cands: list, page: str, dismissed=(), n: int = MAX_PER_PAGE) -> list:
    """The chips a page shows: its own suggestions not dismissed, model-ranked first, then by priority; at most n.
    'home' gets every page's."""
    rows = [s for s in cands if s.key not in dismissed and (page == 'home' or page in s.pages)]
    return sorted(rows, key=lambda s: (s.rank, -s.priority, s.key))[:n]


# --------------------------------------------------------------------------------------------- the model's wording
PHRASE_SYSTEM = (
    'You word next-step suggestions for the user of Falcon Autonomy (self-driving AI development on CARLA). You get '
    'candidates as JSON, each with a key, a title and why (the facts). Rank the most useful first for this user now '
    'and write each as ONE short sentence (at most 110 characters) saying what to do and why. Rules: keep every '
    'number exactly as written in its own why; add no number, name or claim that its why does not hold; no '
    'greetings. Answer with JSON only: {"suggestions": [{"key": "...", "text": "..."}]}.')


def _parse(text: str) -> list:
    t = (text or '').strip()
    m = re.search(r'\{.*\}', t, re.S)
    if not m:
        return []
    try:
        doc = json.loads(m.group(0))
    except ValueError:
        return []
    rows = doc.get('suggestions') if isinstance(doc, dict) else None
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get('key'), str) and isinstance(r.get('text'),
                                                                                                     str)] \
        if isinstance(rows, list) else []


def grounded(text: str, s: Suggestion) -> bool:
    """Whether every number of a wording is one the suggestion's own why, facts, title or arguments hold."""
    src = [{'role': 'tool', 'content': json.dumps({'why': s.why, 'title': s.title, 'facts': s.facts,
                                                   'args': s.args}, default=str)}]
    return not grounding.check(text, src)


def phrase(service, cands: list, context: dict | None = None, top: int = PHRASE_TOP, cancel=None) -> dict:
    """Ask the fast model to rank and word the top candidates (blocking: call it off the GUI thread). Returns
    {'ranked': [keys in the model's order], 'texts': {key: grounded wording}, 'rejected': {key: wording whose numbers
    were not in its facts}}. Raises provider errors (the caller keeps the deterministic wording)."""
    pick = cands[:top]
    if not pick:
        return {'ranked': [], 'texts': {}, 'rejected': {}}
    ctx = dict(context or {})
    ctx['candidates'] = [{'key': s.key, 'title': s.title, 'why': s.why} for s in pick]
    raw = service.quick('Rank and word these suggestions.', ctx, system=PHRASE_SYSTEM, max_tokens=500, cancel=cancel)
    by = {s.key: s for s in pick}
    ranked, texts, rejected = [], {}, {}
    for r in _parse(raw):
        s = by.get(r['key'])
        if s is None or r['key'] in ranked:
            continue
        ranked.append(r['key'])
        t = ' '.join(r['text'].split())[:TEXT_CAP]
        if t and grounded(t, s):
            texts[r['key']] = t
        else:
            rejected[r['key']] = t
    return {'ranked': ranked, 'texts': texts, 'rejected': rejected}


def apply_phrasing(cands: list, res: dict):
    """The model's ranking and grounded wordings onto the candidates (the rest keep their deterministic why)."""
    rank = {k: i for i, k in enumerate(res.get('ranked') or [])}
    for s in cands:
        if s.key in rank:
            s.rank = rank[s.key]
        t = (res.get('texts') or {}).get(s.key)
        if t:
            s.text, s.phrased = t, True


# --------------------------------------------------------------------------------------------- pacing and dismissal
class Store:
    """What the user dismissed in this project: WORKSPACE/assistant/suggestions.json {'dismissed': {key: utc time}}."""

    def __init__(self, workspace: str):
        from .session import folder
        self.workspace = workspace
        self.path = os.path.join(folder(workspace), 'suggestions.json') if workspace else ''
        doc = _json(self.path, {}) if self.path else {}
        self.dismissed: dict = dict((doc or {}).get('dismissed') or {})

    def dismiss(self, key: str):
        from ..core.paths import atomic_write
        from .session import utc
        self.dismissed[key] = utc()
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            atomic_write(self.path, json.dumps({'dismissed': self.dismissed}, indent=1))

    def is_dismissed(self, key: str) -> bool:
        return key in self.dismissed


class Pacer:
    """When a gather and a phrasing may happen (module doc), and the wordings already received (by key and why: a why
    whose numbers changed is worded again)."""

    def __init__(self, min_gather_s: float = MIN_GATHER_S, min_forced_s: float = MIN_FORCED_S,
                 phrase_every_s: float = PHRASE_EVERY_S):
        self.min_gather_s, self.min_forced_s, self.phrase_every_s = min_gather_s, min_forced_s, phrase_every_s
        self.last_gather = None
        self.last_phrase = None
        self.worded: dict = {}              # (key, why) -> (rank, text or '')
        self._lock = threading.Lock()

    def gather_due(self, now: float, forced: bool = False) -> bool:
        if self.last_gather is None:
            return True
        return now - self.last_gather >= (self.min_forced_s if forced else self.min_gather_s)

    def gathered(self, now: float):
        self.last_gather = now

    def recall(self, cands: list):
        """Put wordings already received back on candidates whose why is unchanged."""
        for s in cands:
            got = self.worded.get((s.key, s.why))
            if got:
                s.rank = got[0]
                if got[1]:
                    s.text, s.phrased = got[1], True

    def phrase_due(self, cands: list, now: float, top: int = PHRASE_TOP) -> bool:
        """Whether to ask the model now: some of the top candidates are not worded yet, and the last request is
        PHRASE_EVERY_S old."""
        fresh = [s for s in cands[:top] if (s.key, s.why) not in self.worded]
        if not fresh:
            return False
        return self.last_phrase is None or now - self.last_phrase >= self.phrase_every_s

    def phrased(self, cands: list, res: dict, now: float, top: int = PHRASE_TOP):
        self.last_phrase = now
        rank = {k: i for i, k in enumerate(res.get('ranked') or [])}
        for s in cands[:top]:
            self.worded[(s.key, s.why)] = (rank.get(s.key, 1000), (res.get('texts') or {}).get(s.key, ''))
