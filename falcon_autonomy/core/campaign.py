"""An overnight campaign: a goal and a budget in, a one-page brief out (standard library; the studies it runs are
Falcon's own commands). docs/campaign.md is the manual.

A campaign is a layer on top of what exists — the research autopilot (core.autoresearch: ablations with a power
analysis), failure modes (core.failclusters), root cause (core.rootcause), moments (core.moments), failure rate,
search, sweep and falsification — not a second autopilot: it decides WHICH of those studies to run next within a
budget, runs each as the command the pages and the CLI already run, reads its result, and writes down every decision.

    new(workspace, goal, budget, kinds, …) -> record     a draft with its plan and costs (nothing runs)
    approval_text(record)                                 the whole scope and budget, as the one approval shows it
    approve(workspace, id, approval)                      once, up front (core.autoresearch.APPROVED_HOW)
    step(workspace, id, io) / run(workspace, id, io)      the loop; io is the job queue (QueueIO) or a test's fake
    stop(workspace, id) ; status(workspace, id) ; brief(workspace, id)
    falcon campaign new|approve|run|status|stop|brief|list

State on disk, <workspace>/campaigns/<id>/: campaign.json (the record: goal, budget, scope, approval, hypotheses,
studies, needs-you items), log.jsonl (append-only: every decision with its reason), stop.json (a stop request),
brief.md and brief.html. The loop reads the record at every step, so a campaign resumes after a restart: a study
whose job is gone is marked interrupted and may run once more if the budget still allows.

Scope and approval. The user approves a campaign once, with the goal, the budgets, the allowed study kinds (KINDS)
and the planned studies with their estimated costs shown (approval_text). The approval is the same object the
research autopilot takes ({'how': 'approved' | 'preapproved' | 'command line', …}) and is handed to it unchanged
for an ablation; nothing new is invented. The approved scope is hashed (scope_sha256): a record edited after
approval no longer runs. Inside the scope the campaign runs unattended. What is outside it is not done and is
written to the brief as "needs you": a study of a kind that was not approved, a study whose estimate alone exceeds
a budget, and anything a planner asks for that is never done unattended (OUTSIDE: cloud spend, a model download,
moving anything to the trash, training, and every name on the assistant's never-list, assistant.tool_api.NEVER).

The accountant (Accountant) enforces the budgets and does not depend on any model: before a job starts it is
refused if the campaign is stopped, if the job count is at its limit, if the estimate would take the GPU-hours used
past the budget or end after the deadline, or if the disk budget is used up; while a job runs it is cancelled at
the deadline or when the GPU-hours are used up (the queue's cancel path). GPU-hours are the wall time of the
studies of a kind that holds the GPU (KINDS[kind]['gpu']): an upper bound on GPU time, not a measurement of
utilisation. Estimates are the median duration of finished jobs of the same kind in the workspace's job history
(core.jobhistory); without one, the kind's TYPICAL value, labelled as not measured.

Planning. Deterministic (planner 'playbook'): candidates() builds, from the workspace only, a fixed playbook of
hypotheses, each with the study that would test it — failure modes, the goal's conditions as moments, the root
cause of the newest evaluations, the ablations core.autoresearch.suggestions offers (ranked up when their
conditions are the goal's), a failure-rate study of a project scenario the goal names — ordered by a prior that is
a stated heuristic (PRIORS). With an assistant provider (planner 'assistant') the model is shown the candidates
(public facts only) and returns their order; it cannot add a study: anything else it asks for becomes a "needs you"
item. An unreadable reply falls back to the playbook's order, and the log says so.

The brief (brief()): the goal, the answer in two sentences with its confidence in plain words (CONFIDENCE), what
was tried (study, cost, result), what was ruled out, the evidence (paths), budget used against given, the "needs
you" items and the recommended next step. Every sentence is assembled from the studies' own result files; nothing
from the user's dashcam footage enters it (the studies read are the public ones: core.autoresearch.suggestions and
core.moments with public_only). A conclusion is only as good as the studies that ran.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time

SCHEMA = 'falcon.campaign/1'
FOLDER = 'campaigns'
FILE = 'campaign.json'
LOG = 'log.jsonl'
STOP = 'stop.json'
BRIEF_MD, BRIEF_HTML = 'brief.md', 'brief.html'
STATES = ('draft', 'approved', 'running', 'stopped', 'done')
PLANNERS = ('playbook', 'assistant')

# study kinds a campaign may be allowed to run: what it is, whether it holds the GPU (counted in GPU-hours), a
# typical duration used only until the job history has one (heuristic, labelled "not measured"), and the job kinds
# of the history whose durations estimate it
KINDS = {
    'clusters': {'about': 'failure modes of the failures recorded so far (falcon clusters build)', 'gpu': False,
                 'typical_s': 30.0, 'history': ('clusters',)},
    'moments': {'about': 'how often the goal\'s conditions occur in recorded driving (falcon moments index)',
                'gpu': False, 'typical_s': 120.0, 'history': ('moments-index',)},
    'rootcause': {'about': 'which attributes an evaluation\'s misses go with (falcon why evaluation)', 'gpu': False,
                  'typical_s': 120.0, 'history': ('why', 'rootcause')},
    'ablation': {'about': 'a controlled comparison with a power analysis (the research autopilot)', 'gpu': True,
                 'typical_s': 1800.0, 'history': ('experiment',)},
    'failrate': {'about': 'a scenario\'s failure rate with an interval (falcon failrate run)', 'gpu': True,
                 'typical_s': 3600.0, 'history': ('failrate',)},
    'search': {'about': 'a rare-case search over a scenario\'s parameters (falcon search)', 'gpu': True,
               'typical_s': 2400.0, 'history': ('search', 'study')},
    'sweep': {'about': 'a parameter sweep of a scenario (falcon sweep)', 'gpu': True, 'typical_s': 2400.0,
              'history': ('sweep', 'study')},
    'falsify': {'about': 'a search for a run violating a requirement (falcon falsify run)', 'gpu': True,
                'typical_s': 2400.0, 'history': ('falsify',)},
    'evaluate': {'about': 'a model evaluated on a dataset (falcon evaluate)', 'gpu': True, 'typical_s': 900.0,
                 'history': ('evaluate', 'evaluation')},
}
DEFAULT_KINDS = ('clusters', 'moments', 'rootcause', 'ablation')
# never done unattended, whatever was approved: queued as "needs you"
OUTSIDE = {'cloud': 'any cloud spend needs its own priced approval (falcon cloud)',
           'download': 'a model download is never started unattended',
           'trash': 'nothing is moved to the trash unattended',
           'train': 'training a model is outside a campaign: it is a decision, not a study',
           'settings': 'settings are the user\'s', 'push': 'nothing is pushed anywhere'}
# the playbook's order (heuristic): what is cheap and descriptive first, then the controlled comparison, then the
# long simulator studies
PRIORS = {'clusters': 0.9, 'moments': 0.8, 'rootcause': 0.7, 'ablation': 0.5, 'failrate': 0.4}
GOAL_BONUS = 0.15              # an ablation whose conditions are the goal's ranks this much higher (heuristic)
DEFAULT_BUDGET = {'gpu_hours': 2.0, 'deadline': None, 'max_jobs': 8, 'disk_gb': 20.0}
MIN_FREE_GB = 1.0              # no job starts with less free space than this on the workspace's disk (heuristic)
MAX_ATTEMPTS = 2               # a study whose job was lost in a restart runs at most once more
CONFIDENCE = {
    'high': 'a controlled comparison with enough power supported it',
    'moderate': 'the recorded evidence points one way, but no controlled comparison confirmed it',
    'low': 'only an association in the data recorded so far; nothing was tested',
    'none': 'no study produced evidence for or against',
}
PERCEPTION_WORDS = ('recall', 'precision', 'detect', 'miss', 'perception', 'map', 'ap', 'false')


class CampaignError(Exception):
    pass


def utc(t: float | None = None) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))


def epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    import calendar
    try:
        return float(calendar.timegm(time.strptime(iso[:19], '%Y-%m-%dT%H:%M:%S')))
    except ValueError:
        raise CampaignError(f'{iso!r} is not a UTC time (YYYY-MM-DDTHH:MM:SSZ)') from None


def parse_until(text: str, now: float | None = None) -> str:
    """'07:00' (the next time the local clock shows it), '+90m' / '+6h' (from now) or an ISO UTC time -> ISO UTC."""
    now = time.time() if now is None else now
    text = str(text).strip()
    m = re.fullmatch(r'\+(\d+(?:\.\d+)?)\s*([mh])', text)
    if m:
        return utc(now + float(m.group(1)) * (60.0 if m.group(2) == 'm' else 3600.0))
    m = re.fullmatch(r'(\d{1,2}):(\d{2})', text)
    if m:
        lt = time.localtime(now)
        t = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, int(m.group(1)), int(m.group(2)), 0, 0, 0, -1))
        return utc(t if t > now else t + 86400.0)
    epoch(text)
    return text[:19] + 'Z'


# ================================================================================================ the record
def folder(workspace: str, cid: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,47}', cid or ''):
        raise CampaignError(f'a campaign id is letters, digits, _ . - (not {cid!r})')
    return os.path.join(workspace, FOLDER, cid)


def load(workspace: str, cid: str) -> dict:
    p = os.path.join(folder(workspace, cid), FILE)
    try:
        with open(p, encoding='utf-8') as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        raise CampaignError(f'no campaign {cid!r} in the workspace ({FOLDER}/{cid}/{FILE})') from None
    if not isinstance(rec, dict) or rec.get('schema') != SCHEMA:
        raise CampaignError(f'{FOLDER}/{cid}/{FILE} is not a campaign')
    return rec


def save(workspace: str, rec: dict):
    from .paths import atomic_write
    d = folder(workspace, rec['id'])
    os.makedirs(d, exist_ok=True)
    atomic_write(os.path.join(d, FILE), json.dumps(rec, indent=1, default=str))


def log(workspace: str, cid: str, decision: str, reason: str, **extra) -> dict:
    """One line of the append-only decision log."""
    row = dict({'time': utc(), 'decision': decision, 'reason': reason}, **extra)
    d = folder(workspace, cid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, LOG), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, default=str) + '\n')
    return row


def read_log(workspace: str, cid: str, limit: int | None = None) -> list:
    p = os.path.join(folder(workspace, cid), LOG)
    out = []
    if os.path.isfile(p):
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out[-limit:] if limit else out


def campaigns(workspace: str) -> list:
    d = os.path.join(workspace, FOLDER) if workspace else ''
    out = []
    for n in sorted(os.listdir(d)) if d and os.path.isdir(d) else ():
        try:
            out.append(load(workspace, n))
        except CampaignError:
            continue
    return out


def scope_hash(rec: dict) -> str:
    """What the approval covers: the goal, the budgets, the allowed kinds, the project, the planner and the planned
    studies' commands."""
    doc = {'goal': rec['goal'], 'budget': rec['budget'], 'kinds': sorted(rec['scope']['kinds']),
           'project': rec['scope'].get('project') or '', 'planner': rec['planner'],
           'studies': [[s['id'], s['kind'], (s.get('ablation') or {}).get('study') or s.get('argv')]
                       for s in rec['studies'] if s.get('planned_at_approval', True)]}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def state_of(rec: dict) -> str:
    if rec.get('finished'):
        return 'stopped' if (rec.get('stopped') or {}).get('by') == 'user' else 'done'
    if not rec.get('approval'):
        return 'draft'
    return 'running' if any(s['state'] == 'running' for s in rec['studies']) or rec.get('started') else 'approved'


# ================================================================================================ the accountant
def _public(path) -> bool:
    try:
        from ..assistant.privacy import private_reason
        return not private_reason(path)
    except Exception:              # noqa: BLE001 (a check that fails must not make a private thing public)
        return False


class Accountant:
    """The budgets, enforced without any model (module doc). history: the workspace's finished jobs
    (core.jobhistory records); free_gb(): the free space of the workspace's disk; size_gb(path): what a study wrote."""

    def __init__(self, budget: dict, history=(), free_gb=None, size_gb=None):
        self.b = dict(DEFAULT_BUDGET, **(budget or {}))
        self.history = list(history or ())
        self.free_gb = free_gb or (lambda: float('inf'))
        self.size_gb = size_gb or (lambda path: 0.0)

    def estimate(self, kind: str, given_s: float | None = None) -> dict:
        """{'seconds', 'gpu_seconds', 'source', 'n'}: the median of the kind's finished jobs in the history; else a
        study's own estimate (an ablation's drives at the measured throughput); else the kind's typical value."""
        k = KINDS[kind]
        names = (f'campaign-{kind}',) + tuple(k['history'])
        xs = sorted(float(r['elapsed_s']) for r in self.history
                    if r.get('status') == 'done' and r.get('kind') in names and r.get('elapsed_s'))
        if xs:
            s = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
            src = f'measured: the median of {len(xs)} finished job(s) of this kind in the job history'
        elif given_s:
            s, src = float(given_s), 'the study\'s own estimate (its drives at this machine\'s measured or typical rate)'
        else:
            s, src = float(k['typical_s']), 'typical, not measured: no finished job of this kind in the job history yet'
        return {'seconds': round(s, 1), 'gpu_seconds': round(s, 1) if k['gpu'] else 0.0, 'source': src, 'n': len(xs)}

    def used(self, rec: dict, now: float | None = None) -> dict:
        gpu = jobs = 0.0
        disk = 0.0
        for s in rec['studies']:
            if not s.get('launches'):
                continue
            jobs += int(s['launches'])
            el = float(s.get('elapsed_s') or 0.0)
            if s['state'] == 'running' and now is not None and s.get('started_t'):
                el = max(el, now - float(s['started_t']))
            el += sum(float(x) for x in s.get('lost_s') or ())
            if KINDS[s['kind']]['gpu']:
                gpu += el
            if s.get('out'):
                disk += float(self.size_gb(s['out']))
        dl = epoch(self.b.get('deadline'))
        return {'gpu_hours': round(gpu / 3600.0, 4), 'jobs': int(jobs), 'disk_gb': round(disk, 3),
                'seconds_left': None if dl is None or now is None else round(dl - now, 1)}

    def may_start(self, rec: dict, study: dict, now: float) -> tuple:
        """(ok, reason). The reason names the budget that refuses and the numbers."""
        u = self.used(rec, now)
        est = study.get('est') or self.estimate(study['kind'])
        if u['jobs'] >= int(self.b['max_jobs']):
            return False, f'the job limit is reached ({u["jobs"]} of {int(self.b["max_jobs"])})'
        need = est['gpu_seconds'] / 3600.0
        if need and u['gpu_hours'] + need > float(self.b['gpu_hours']) + 1e-9:
            return False, (f'its estimate of {need:.2f} GPU-hours would exceed what is left '
                           f'({max(float(self.b["gpu_hours"]) - u["gpu_hours"], 0.0):.2f} of '
                           f'{float(self.b["gpu_hours"]):g})')
        dl = epoch(self.b.get('deadline'))
        if dl is not None and now + est['seconds'] > dl:
            return False, (f'its estimate of {est["seconds"] / 60.0:.0f} min would end after the deadline '
                           f'({self.b["deadline"]}; {max(dl - now, 0.0) / 60.0:.0f} min left)')
        if u['disk_gb'] >= float(self.b['disk_gb']):
            return False, f'the disk budget is used up ({u["disk_gb"]:.2f} of {float(self.b["disk_gb"]):g} GB)'
        if self.free_gb() < MIN_FREE_GB:
            return False, f'less than {MIN_FREE_GB:g} GB is free on the workspace\'s disk'
        return True, (f'within budget: about {est["seconds"] / 60.0:.0f} min'
                      + (f', {need:.2f} GPU-hours' if need else ', no GPU') + f' ({est["source"]})')

    def must_cancel(self, rec: dict, study: dict, now: float) -> str | None:
        dl = epoch(self.b.get('deadline'))
        if dl is not None and now >= dl:
            return f'the deadline ({self.b["deadline"]}) is reached'
        u = self.used(rec, now)
        if KINDS[study['kind']]['gpu'] and u['gpu_hours'] > float(self.b['gpu_hours']):
            return f'the GPU-hours are used up ({u["gpu_hours"]:.2f} of {float(self.b["gpu_hours"]):g})'
        return None


def _dir_gb(path: str) -> float:
    total = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total / 1e9


def accountant(workspace: str, rec: dict) -> Accountant:
    from .jobhistory import JobHistory
    return Accountant(rec['budget'], JobHistory(workspace).read(2000),
                      free_gb=lambda: shutil.disk_usage(workspace).free / 1e9, size_gb=_dir_gb)


# ================================================================================================ planning
def goal_conditions(goal: str) -> dict:
    """What the goal names, by core.scenario_check's phrase set and core.moments' query words (local, rule-based):
    {'light', 'weather', 'kinds', 'types', 'perception' (it is about detection quality), 'filter'}."""
    from . import moments as M
    from . import scenario_check as CK
    ck = CK.extract(goal)
    pq = M.parse_query(goal)
    words = set(re.findall(r'[a-z]+', goal.lower()))
    return {'light': next((c['class'] for c in ck['claims'] if c['type'] == 'light'), None),
            'weather': next((c['class'] for c in ck['claims'] if c['type'] == 'weather'), None),
            'kinds': [c['kind'] for c in ck['claims'] if c['type'] == 'actor'],
            'manoeuvres': [c['what'] for c in ck['claims'] if c['type'] == 'manoeuvre'],
            'types': list(pq['filter']['types']),
            'perception': any(any(w.startswith(p) for p in PERCEPTION_WORDS if len(p) > 3) or w in ('map', 'ap')
                              for w in words),
            'filter': pq['filter']}


def _about_matches(text: str, cond: dict) -> bool:
    t = str(text or '').lower().replace('_', ' ')
    want = [cond.get('light'), cond.get('weather')] + list(cond.get('kinds') or ())
    return any(w and str(w).replace('_', ' ') in t for w in want)


def candidates(workspace: str, rec: dict, project=None) -> list:
    """The playbook (module doc): [{'key', 'kind', 'hypothesis', 'title', 'prior', 'why', 'argv' | 'ablation', 'out'}],
    best first. Built from public facts of the workspace only; deterministic."""
    from . import autoresearch as R
    from . import reports as RP
    cid = rec['id']
    cond = goal_conditions(rec['goal'])
    proj_path = rec['scope'].get('project') or ''
    target = proj_path or workspace
    base = os.path.join(folder(workspace, cid), 'studies')
    out = [
        {'key': 'clusters', 'kind': 'clusters', 'prior': PRIORS['clusters'],
         'hypothesis': 'The failures already recorded fall into a mode whose conditions are the goal\'s.',
         'title': 'Failure modes of the recorded failures', 'argv': ['clusters', 'build', target, '--json'],
         'why': 'cheap, and it tells which conditions the failures share before anything is run'},
        {'key': 'moments', 'kind': 'moments', 'prior': PRIORS['moments'],
         'hypothesis': 'The goal\'s conditions occur in the recorded driving often enough to be looked at there.',
         'title': 'Moments matching the goal in recorded runs', 'argv': ['moments', 'index', workspace],
         'why': 'cheap: it finds the recorded moments the goal describes and how often they occur'},
    ]
    if cond['perception']:
        evs = [(lb, p) for lb, p in (RP.sources(workspace).get('evaluation') or ()) if _public(p)][:2]
        for k, (lb, p) in enumerate(evs):
            od = os.path.join(base, f'why-{k + 1}')
            out.append({'key': f'rootcause:{lb}', 'kind': 'rootcause', 'prior': PRIORS['rootcause'] - 0.05 * k,
                        'hypothesis': f'The misses of evaluation {lb} go with an attribute its ground truth records '
                                      '(range, occlusion, size, light).',
                        'title': f'Root cause of evaluation {lb}', 'out': od,
                        'argv': ['why', 'evaluation', p, '--out', od],
                        'why': 'the goal is about detection quality and this evaluation is the newest evidence of it'})
    try:
        sugg = R.suggestions(workspace, project)
    except Exception:              # noqa: BLE001 (a workspace the suggestions cannot read offers none)
        sugg = []
    k = 0
    for s in sugg:
        if not s.get('template'):
            continue
        k += 1
        hit = _about_matches(s.get('about'), cond) or _about_matches(json.dumps(s.get('facts'), default=str), cond)
        out.append({'key': f'ablation:{s["id"]}', 'kind': 'ablation',
                    'prior': PRIORS['ablation'] - 0.05 * (k - 1) + (GOAL_BONUS if hit else 0.0),
                    'hypothesis': s['question'], 'title': f'Ablation: {s.get("about")}',
                    'ablation': {'study': f'{cid}-a{k}', 'spec': s['template'], 'source': s['id']},
                    'why': ('its conditions are the goal\'s; ' if hit else '') + 'a controlled comparison is the only '
                           'study here that can support a cause'})
    doc = getattr(project, 'doc', None) or {}
    if proj_path and cond['manoeuvres']:
        from .scenario_text import BASES
        seen = set()
        for man in cond['manoeuvres']:
            by = BASES.get(man) or {}
            want = [by.get(kd) for kd in cond['kinds'] if by.get(kd)] or [by.get(None)]
            for sid in want:
                have = any(s.get('id') == sid for s in doc.get('scenarios') or ())
                if not sid or sid in seen or not have:
                    continue
                seen.add(sid)
                out.append({'key': f'failrate:{sid}', 'kind': 'failrate', 'prior': PRIORS['failrate'],
                            'hypothesis': f'The scenario {sid} fails more often than the goal assumes.',
                            'title': f'Failure rate of {sid}',
                            'argv': ['failrate', 'run', proj_path, '--scenario', sid, '--name', f'{cid}-fr-{sid}'],
                            'why': 'the goal names this manoeuvre and the project has its scenario'})
    out.sort(key=lambda c: (-c['prior'], c['key']))
    return out


PLANNER_PROMPT = ('You order studies for an overnight research campaign. Reply with JSON only: {"order": [candidate '
                  'keys, most worth running first], "extra": [{"action": "...", "why": "..."}]}. You cannot add a '
                  'study: "extra" is for anything else you would want done, which a person will decide.')


def _model_order(cands: list, rec: dict, complete) -> tuple:
    """(candidates reordered by the model, extra requests, note). The model sees the goal, the budgets and the
    candidates' public texts; a reply that cannot be read keeps the playbook's order."""
    msg = [{'role': 'system', 'content': PLANNER_PROMPT},
           {'role': 'user', 'content': json.dumps({
               'goal': rec['goal'], 'budget': rec['budget'],
               'candidates': [{'key': c['key'], 'kind': c['kind'], 'hypothesis': c['hypothesis'],
                               'estimate_min': round((c.get('est') or {}).get('seconds', 0.0) / 60.0, 1)}
                              for c in cands]})}]
    try:
        reply = complete(msg)
        m = re.search(r'\{.*\}', str(reply), re.S)
        doc = json.loads(m.group(0)) if m else None
        order = [str(k) for k in doc['order']]
        extra = [e for e in doc.get('extra') or () if isinstance(e, dict)]
    except Exception as e:         # noqa: BLE001 (any provider or parsing failure: the deterministic order stands)
        return cands, [], f'the planner\'s reply could not be used ({type(e).__name__}): the playbook\'s order stands'
    by = {c['key']: c for c in cands}
    ranked = [by[k] for k in dict.fromkeys(order) if k in by]
    ranked += [c for c in cands if c not in ranked]
    extra += [{'action': k, 'why': 'named in the planner\'s order but not a candidate'} for k in order if k not in by]
    return ranked, extra, f'ordered by the assistant ({len(order)} key(s) in its reply)'


def plan(workspace: str, rec: dict, project=None, complete=None, at_approval: bool = True) -> list:
    """Add the candidates not yet in the record: allowed kinds become planned studies (with estimates), the others
    "needs you" items. Returns the log rows it wrote."""
    acc = accountant(workspace, rec)
    rows = []
    cands = candidates(workspace, rec, project)
    known = {s['key'] for s in rec['studies']} | {n.get('key') for n in rec['needs_you']}
    cands = [c for c in cands if c['key'] not in known]
    for c in cands:
        given = None
        if c['kind'] == 'ablation':
            try:
                from . import autoresearch as R
                d = R.draft(workspace, c['ablation']['study'], c['hypothesis'], c['ablation']['spec'], project)
                given = d['cost'].get('seconds')
                c['cost_note'] = d['cost'].get('what')
            except Exception as e:           # noqa: BLE001 (a template that cannot run is not a candidate)
                c['broken'] = str(e)[:200]
        c['est'] = acc.estimate(c['kind'], given)
        if c['kind'] == 'ablation' and given is None and not c.get('broken') and not c['est']['n']:
            c['est'] = dict(c['est'], seconds=1.0, gpu_seconds=0.0, source='simulated executor: no simulator time')
    cands = [c for c in cands if not c.get('broken')]
    note = 'the playbook\'s order (planner: playbook)'
    extra = []
    if complete is not None and rec['planner'] == 'assistant' and cands:
        cands, extra, note = _model_order(cands, rec, complete)
    for e in extra:
        act = str(e.get('action') or '')[:200]
        why = next((w for k, w in OUTSIDE.items() if k in act.lower()), 'outside the approved scope')
        rec['needs_you'].append({'key': f'extra:{act}', 'what': act, 'why': f'{why} (asked for by the planner: '
                                 f'{str(e.get("why") or "")[:200]})', 'time': utc()})
        rows.append(log(workspace, rec['id'], 'needs_you', why, what=act))
    n0 = len(rec['hypotheses'])
    for c in cands:
        if c['kind'] not in rec['scope']['kinds']:
            rec['needs_you'].append({'key': c['key'], 'what': f'{c["title"]} ({c["kind"]})',
                                     'why': f'the study kind {c["kind"]!r} is not in the approved scope', 'time': utc()})
            rows.append(log(workspace, rec['id'], 'needs_you', f'{c["kind"]} is not an approved study kind',
                            what=c['title']))
            continue
        b = rec['budget']
        if c['est']['gpu_seconds'] / 3600.0 > float(b['gpu_hours']):
            rec['needs_you'].append({'key': c['key'], 'what': c['title'],
                                     'why': f'its estimate ({c["est"]["gpu_seconds"] / 3600.0:.2f} GPU-hours) alone '
                                            f'exceeds the campaign\'s {float(b["gpu_hours"]):g} GPU-hours', 'time': utc()})
            rows.append(log(workspace, rec['id'], 'needs_you', 'the estimate alone exceeds the GPU budget',
                            what=c['title']))
            continue
        hid, sid = f'h{len(rec["hypotheses"]) + 1}', f's{len(rec["studies"]) + 1}'
        rec['hypotheses'].append({'id': hid, 'text': c['hypothesis'], 'prior': round(c['prior'], 3), 'study': sid,
                                  'status': 'open', 'why_ranked': c['why']})
        st = {'id': sid, 'key': c['key'], 'hypothesis': hid, 'kind': c['kind'], 'title': c['title'],
              'argv': c.get('argv'), 'out': c.get('out'), 'est': c['est'], 'state': 'planned',
              'planned_at_approval': bool(at_approval)}
        if c.get('ablation'):
            st['ablation'] = c['ablation']
            st['cost_note'] = c.get('cost_note')
        rec['studies'].append(st)
    if len(rec['hypotheses']) > n0 or not rec['studies']:
        rows.append(log(workspace, rec['id'], 'plan', note, hypotheses=[h['id'] for h in rec['hypotheses'][n0:]],
                        studies=[s['id'] for s in rec['studies'] if s['state'] == 'planned']))
    return rows


def new(workspace: str, goal: str, budget: dict | None = None, kinds=DEFAULT_KINDS, cid: str = '',
        project_path: str = '', project=None, planner: str = 'playbook', complete=None, now: float | None = None) -> dict:
    """A draft campaign with its plan (nothing runs): the record, written."""
    goal = str(goal or '').strip()
    if len(goal) < 8:
        raise CampaignError('state the goal in a sentence (what should be found out)')
    if not workspace or not os.path.isdir(workspace):
        raise CampaignError(f'{workspace!r} is not a workspace folder')
    if planner not in PLANNERS:
        raise CampaignError(f'planner {planner!r}: one of {", ".join(PLANNERS)}')
    kinds = list(dict.fromkeys(kinds or ()))
    bad = [k for k in kinds if k not in KINDS]
    if bad or not kinds:
        why = '; '.join(f'{k}: {OUTSIDE[k]}' for k in bad if k in OUTSIDE)
        raise CampaignError(f'study kinds: choose from {", ".join(KINDS)}' + (f' ({why})' if why else ''))
    b = dict(DEFAULT_BUDGET, **{k: v for k, v in (budget or {}).items() if v is not None})
    unknown = set(b) - set(DEFAULT_BUDGET)
    if unknown:
        raise CampaignError(f'unknown budget field(s) {", ".join(sorted(unknown))}')
    try:
        b['gpu_hours'], b['max_jobs'], b['disk_gb'] = float(b['gpu_hours']), int(b['max_jobs']), float(b['disk_gb'])
    except (TypeError, ValueError):
        raise CampaignError('budget: gpu_hours and disk_gb are numbers, max_jobs a whole number') from None
    if b['gpu_hours'] < 0 or b['max_jobs'] < 1 or b['disk_gb'] <= 0:
        raise CampaignError('budget: gpu_hours ≥ 0, max_jobs ≥ 1, disk_gb > 0')
    t = time.time() if now is None else now
    if b['deadline']:
        if epoch(b['deadline']) <= t:
            raise CampaignError(f'the deadline {b["deadline"]} is in the past')
    cid = cid or 'c-' + hashlib.sha1(f'{goal}|{t}'.encode()).hexdigest()[:8]
    if os.path.exists(folder(workspace, cid)):
        raise CampaignError(f'{FOLDER}/{cid} exists already: choose another id')
    rec = {'schema': SCHEMA, 'id': cid, 'goal': goal, 'created': utc(t), 'budget': b, 'planner': planner,
           'scope': {'kinds': kinds, 'project': os.path.abspath(project_path) if project_path else ''},
           'approval': None, 'scope_sha256': None, 'hypotheses': [], 'studies': [], 'needs_you': [],
           'started': None, 'finished': None, 'stopped': None, 'answer': None}
    log(workspace, cid, 'new', 'the campaign was drafted', goal=goal, budget=b, kinds=kinds, planner=planner)
    plan(workspace, rec, project, complete, at_approval=True)
    save(workspace, rec)
    return rec


def _fmt_s(s) -> str:
    from .budget import fmt_duration
    return fmt_duration(float(s or 0.0))


def approval_text(rec: dict) -> str:
    """The whole scope and budget, as the one approval shows it."""
    b = rec['budget']
    lines = [f'Campaign {rec["id"]}: {rec["goal"]}', '',
             f'Budget: {b["gpu_hours"]:g} GPU-hours · deadline {b["deadline"] or "none"} · at most {b["max_jobs"]} jobs '
             f'· {b["disk_gb"]:g} GB of disk',
             f'Allowed study kinds: {", ".join(rec["scope"]["kinds"])}',
             f'Planner: {rec["planner"]}' + (' (the model orders the candidates below; it cannot add a study)'
                                             if rec['planner'] == 'assistant' else ' (rule-based, no model)'),
             '', 'Planned studies, in order (each is refused at its turn if it no longer fits the budget):']
    for s in rec['studies']:
        h = next((h for h in rec['hypotheses'] if h['id'] == s['hypothesis']), {})
        gpu = f', {s["est"]["gpu_seconds"] / 3600.0:.2f} GPU-h' if s['est']['gpu_seconds'] else ', no GPU'
        lines.append(f'  {s["id"]} [{s["kind"]}] {s["title"]} — about {_fmt_s(s["est"]["seconds"])}{gpu} '
                     f'({s["est"]["source"].split(":")[0]})')
        lines.append(f'      tests: {h.get("text", "")}')
    if not rec['studies']:
        lines.append('  (none: nothing in the workspace suggests a study of an allowed kind)')
    lines += ['', 'Studies found later (after a result) run only if they are of an allowed kind and fit what is left.',
              'Never done unattended, whatever is approved — written to the brief as "needs you": '
              + '; '.join(OUTSIDE) + '; anything on the assistant\'s never-list.',
              'Approving once lets these run unattended until the budget, the deadline, Stop or an answer.']
    if rec['needs_you']:
        lines += ['', 'Already outside the scope (needs you):'] + [f'  {n["what"]} — {n["why"]}'
                                                                   for n in rec['needs_you']]
    return '\n'.join(lines)


def approve(workspace: str, cid: str, approval: dict) -> dict:
    """The user's one approval (core.autoresearch.APPROVED_HOW): the scope is hashed and the campaign may run."""
    from . import autoresearch as R
    rec = load(workspace, cid)
    if rec.get('finished'):
        raise CampaignError(f'campaign {cid} is finished')
    if rec.get('approval'):
        raise CampaignError(f'campaign {cid} is approved already')
    try:
        ap = R._approved(approval)
    except R.StudyError:
        raise CampaignError('a campaign starts only after the user approves its scope and budget (its card, or '
                            '`falcon campaign approve`)') from None
    rec['approval'] = ap
    rec['scope_sha256'] = scope_hash(rec)
    save(workspace, rec)
    log(workspace, cid, 'approved', f'approved once, with the whole scope and budget shown ({ap["how"]})',
        scope_sha256=rec['scope_sha256'])
    return rec


def detached_command(workspace: str, cid: str) -> str:
    """What to run to leave the campaign running after the terminal closes (printed, never run by Falcon)."""
    import shlex
    cmd = ['systemd-run', '--user', '--collect', f'--unit=falcon-campaign-{cid}', '-p', 'MemoryMax=6G', sys.executable,
           '-m', 'falcon_autonomy', 'campaign', 'run', workspace, cid]
    return ' '.join(shlex.quote(c) for c in cmd)


# ================================================================================================ results
def _read(path, default=None):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def triage(workspace: str, rec: dict, study: dict, project=None) -> dict:
    """A finished study's result, read from its own files: {'summary', 'verdict' ('supported' | 'not supported' |
    'ruled out' | 'inconclusive' | 'evidence'), 'strength' (a key of CONFIDENCE), 'numbers', 'links'}."""
    from . import autoresearch as R
    cond = goal_conditions(rec['goal'])
    kind = study['kind']
    if kind == 'clusters':
        try:
            modes = [s for s in R.suggestions(workspace, project, limit=50) if s['kind'] == 'failure_mode']
        except Exception:          # noqa: BLE001
            modes = []
        hit = [m for m in modes if _about_matches(m.get('about'), cond)
               or _about_matches(' '.join(m['facts'].get('features') or ()), cond)]
        if not modes:
            return {'summary': 'no failure mode with two or more public failures was found', 'verdict': 'inconclusive',
                    'strength': 'none', 'numbers': {'modes': 0}, 'links': [os.path.join(workspace, 'clusters')]}
        top = (hit or modes)[0]
        return {'summary': (f'{len(modes)} failure mode(s); "{top["about"]}" ({top["facts"]["size"]} failures) '
                            + ('shares the goal\'s conditions' if hit else 'is the largest; none shares the goal\'s '
                                                                           'conditions')),
                'verdict': 'evidence' if hit else 'not supported', 'strength': 'low' if hit else 'none',
                'numbers': {'modes': len(modes), 'matching': len(hit), 'size': top['facts']['size']},
                'links': [os.path.join(workspace, 'clusters')]}
    if kind == 'moments':
        from . import moments as M
        f = cond['filter']
        res = M.search(workspace, f, limit=5, strict=True, public_only=True)
        st = M.stats(workspace, public_only=True)
        n, h = res['total'], st['hours']
        constrained = M.understood(f) != '(everything)'
        return {'summary': (f'{n} recorded moment(s) match "{res["understood"]}" in {h:.2f} h of indexed driving'
                            + (f' ({res["private_left_out"]} more in private runs, not read)'
                               if res['private_left_out'] else '')) if constrained else
                           f'the goal names no moment the index knows; {h:.2f} h of driving are indexed',
                'verdict': 'evidence' if n and constrained else 'inconclusive',
                'strength': 'low' if n and constrained else 'none',
                'numbers': {'matching_moments': n, 'hours': h, 'private_left_out': res['private_left_out']},
                'links': [m['run'] for m in res['hits']] or [M.db_path(workspace)]}
    if kind == 'rootcause':
        doc = _read(os.path.join(study.get('out') or '', 'rootcause.json'), {}) or {}
        if not doc:
            return {'summary': 'no root-cause result was written', 'verdict': 'inconclusive', 'strength': 'none',
                    'numbers': {}, 'links': [study.get('out')]}
        clear = [a for a in doc.get('attributes') or () if a.get('clear')]
        names = [a['attribute'] for a in clear]
        hit = any(_about_matches(a, cond) or (cond.get('light') and a in ('light', 'brightness', 'sun'))
                  for a in names)
        return {'summary': f'{doc.get("summary")} (miss rate {doc.get("miss_rate")}; an association, not a cause'
                           + (f'; {doc["note"]}' if doc.get('note') else '') + ')',
                'verdict': 'evidence' if clear else 'not supported',
                'strength': 'moderate' if hit else ('low' if clear else 'none'),
                'numbers': {'miss_rate': doc.get('miss_rate'), 'clear_attributes': names[:6],
                            'objects': doc.get('n_objects')},
                'links': [os.path.join(study['out'], 'rootcause.json')]}
    if kind == 'ablation':
        sid = study['ablation']['study']
        try:
            st = R.status(workspace, sid, job_state=lambda j: 'done')
            r = st['rounds'][-1]
        except Exception as e:     # noqa: BLE001 (StudyError, a missing analysis)
            return {'summary': f'the ablation\'s result could not be read ({e})', 'verdict': 'inconclusive',
                    'strength': 'none', 'numbers': {}, 'links': [R.study_dir(workspace, sid)]}
        v = r.get('verdict')
        cs = r.get('contrasts') or []
        c0 = cs[0] if cs else {}
        power = [c.get('power_at_planned') for c in cs if c.get('power_at_planned') is not None]
        powered = bool(power) and min(power) >= 0.8
        verdict = {'supported': 'supported', 'contradicted': 'ruled out'}.get(v) or \
            ('ruled out' if v == 'not supported' and powered else 'inconclusive')
        return {'summary': (f'the ablation\'s verdict is "{v}"' + (
            f': {c0.get("label") or c0.get("id")} differs by {c0.get("diff")} (95 % [{c0.get("ci_low")}, '
            f'{c0.get("ci_high")}], adjusted p {c0.get("p_adjusted")})' if c0 else '')
            + ('' if verdict != 'inconclusive' else '; it did not have the power to rule the effect out')),
            'verdict': verdict, 'strength': 'high' if verdict == 'supported' else 'none',
            'numbers': {'verdict': v, 'diff': c0.get('diff'), 'ci': [c0.get('ci_low'), c0.get('ci_high')],
                        'p_adjusted': c0.get('p_adjusted'), 'runs': r.get('runs')},
            'links': [R.study_dir(workspace, sid)]}
    # failrate, search, sweep, falsify, evaluate: the study's own folder, read generically
    name = next((study['argv'][i + 1] for i, a in enumerate(study.get('argv') or ()) if a in ('--name', '--study')), '')
    d = os.path.join(workspace, 'studies', name) if name else ''
    doc = next((x for x in (_read(os.path.join(d, f)) for f in ('failrate.json', 'estimate.json', 'study.json'))
                if isinstance(x, dict)), {}) if d else {}
    est = doc.get('estimate') if isinstance(doc.get('estimate'), dict) else doc
    p = est.get('p') if isinstance(est, dict) else None
    return {'summary': (f'failure rate {p} (95 % [{est.get("ci_low", est.get("lower"))}, '
                        f'{est.get("ci_high", est.get("upper"))}])' if p is not None else
                        'the study finished; its folder holds the result (not summarised here)'),
            'verdict': 'evidence' if p is not None else 'inconclusive', 'strength': 'low' if p is not None else 'none',
            'numbers': {'p': p}, 'links': [d or workspace]}


# ================================================================================================ the loop
class QueueIO:
    """The campaign's jobs through core.jobs.JobQueue (subprocesses, as the CLI runs them): submit, state, cancel,
    and every finished job written to the workspace's job history (so later estimates are measured)."""

    def __init__(self, workspace: str, queue=None, fake_carla: str = ''):
        from .jobhistory import JobHistory
        from .jobs import JobQueue, SubprocessLauncher
        from .settings import MEMORY_RESERVE_GB
        self.workspace = workspace
        self.queue = queue or JobQueue(SubprocessLauncher(), memory_reserve_gb=MEMORY_RESERVE_GB)
        self.history = JobHistory(workspace)
        self.fake_carla = fake_carla
        self.queue.subscribe(self._on)

    def _on(self, ev, job):
        if ev == 'finished' and job.spec.meta.get('campaign'):
            try:
                self.history.record(job, page='campaign')
            except OSError:
                pass

    def now(self) -> float:
        return time.time()

    def submit(self, cid: str, study: dict) -> dict:
        from .jobs import JobSpec
        argv = list(study['argv'])
        if self.fake_carla and study['kind'] in ('failrate', 'search', 'sweep', 'falsify'):
            argv += ['--fake-carla', self.fake_carla]
        spec = JobSpec(f'campaign-{study["kind"]}', f'Campaign {cid} · {study["id"]} {study["title"]}',
                       [sys.executable, '-m', 'falcon_autonomy'] + argv, memory_gb=1.5,
                       meta={'campaign': cid, 'study': study['id']})
        job = self.queue.submit(spec)
        return {'id': job.id, 'submitted': job.submitted}

    def _job(self, ref):
        job = self.queue.get(int(ref.get('id') or 0)) if ref else None
        if job is None or abs(float(job.submitted) - float(ref.get('submitted') or 0.0)) > 1e-3:
            return None
        return job

    def state(self, ref) -> tuple:
        self.queue.poll()
        job = self._job(ref)
        return (None, 0.0) if job is None else (job.state, float(job.elapsed))

    def cancel(self, ref):
        job = self._job(ref)
        if job is not None:
            self.queue.cancel(job)

    def cancel_all(self):
        for job in list(self.queue.active()):
            if job.spec.meta.get('campaign'):
                self.queue.cancel(job)
        self.queue.poll()

    def notify(self, title: str, body: str):
        from ..cluster.cloud import desktop_notify
        desktop_notify(title, body)


def _check_runnable(workspace: str, rec: dict):
    if not rec.get('approval'):
        raise CampaignError(f'campaign {rec["id"]} is not approved: `falcon campaign approve` shows its scope and budget')
    from . import autoresearch as R
    if rec['approval'].get('how') not in R.APPROVED_HOW:
        raise CampaignError('the stored approval is not one the user gave')
    if rec.get('scope_sha256') != scope_hash(rec):
        raise CampaignError('the campaign\'s goal, budget or scope was changed after it was approved: it does not run '
                            '(make a new campaign)')


def stop_requested(workspace: str, cid: str) -> dict | None:
    return _read(os.path.join(folder(workspace, cid), STOP))


def _launch(workspace: str, rec: dict, study: dict, io, project=None, settings=None) -> dict:
    """Queue a study's job: the command itself, or — an ablation — through the research autopilot, with the
    campaign's approval."""
    cid = rec['id']
    if study['kind'] != 'ablation':
        return io.submit(cid, study)
    from . import ablation as AB
    from . import autoresearch as R
    ab = study['ablation']
    ref = {}

    def submit(exp_id):
        _e, s, _p = AB.load(workspace, exp_id)
        if s['run']['executor'] == 'detection':
            if not rec['scope'].get('project'):
                raise R.StudyError('this ablation drives CARLA and the campaign has no project file')
            argv = ['experiment', 'ablation', 'run', exp_id, '--project', rec['scope']['project']]
        else:
            argv = ['experiment', 'ablation', 'run', exp_id, '--workspace', workspace]
        ref.update(io.submit(cid, dict(study, argv=argv)))
        study['command'] = argv
        return dict(ref)
    if os.path.exists(R.study_dir(workspace, ab['study'])):         # a restart: the study is there, run it on
        R.resume(workspace, ab['study'], rec['approval'], submit)
    else:
        prop = R.draft(workspace, ab['study'], study['title'], ab['spec'], project, settings,
                       {'kind': 'campaign', 'ref': cid, 'about': ab.get('source')})
        R.start(workspace, prop, dict(rec['approval'], options={}), submit, project)
    return ref


def _answer(rec: dict) -> dict:
    """The answer, from the studies' verdicts (module doc): two sentences and a confidence of CONFIDENCE."""
    done = [s for s in rec['studies'] if s['state'] == 'done' and s.get('result')]
    by = {h['id']: h for h in rec['hypotheses']}
    sup = [s for s in done if s['result']['verdict'] == 'supported']
    ev = sorted((s for s in done if s['result']['verdict'] == 'evidence'),
                key=lambda s: -('none', 'low', 'moderate', 'high').index(s['result']['strength']))
    out = [s for s in done if s['result']['verdict'] in ('ruled out', 'not supported')]
    if sup:
        s = sup[0]
        return {'confidence': 'high', 'study': s['id'],
                'text': f'{by[s["hypothesis"]]["text"]} Yes: {s["result"]["summary"]}.',
                'second': 'This is the one result here that tests a cause; the other studies only describe what was '
                          'recorded.'}
    if ev and ev[0]['result']['strength'] != 'none':
        s = ev[0]
        return {'confidence': s['result']['strength'], 'study': s['id'],
                'text': f'The best lead: {s["result"]["summary"]}.',
                'second': 'No controlled comparison confirmed it within the budget, so it is an association in the '
                          'recorded data, not an established cause.'}
    return {'confidence': 'none', 'study': None,
            'text': 'The campaign did not find an answer.' + (
                f' {len(out)} hypothesis(es) were not supported by what ran.' if out else ''),
            'second': 'Either nothing in the workspace bears on the goal yet, or the budget ended before a study that '
                      'could answer it ran.'}


def _finish(workspace: str, rec: dict, io, reason: str, by: str = 'falcon') -> dict:
    rec['finished'] = utc(io.now())
    rec['stopped'] = {'by': by, 'why': reason, 'time': rec['finished']}
    for s in rec['studies']:
        if s['state'] == 'planned':
            s['state'], s['why'] = 'not run', reason
    rec['answer'] = _answer(rec)
    save(workspace, rec)
    log(workspace, rec['id'], 'stop', reason, by=by)
    b = brief(workspace, rec['id'])
    try:
        io.notify(f'Falcon campaign {rec["id"]} finished',
                  f'{reason}. Confidence: {rec["answer"]["confidence"]}. The brief is in the campaign\'s folder.')
    except Exception:              # noqa: BLE001 (a missing notifier never fails the campaign)
        pass
    try:
        os.remove(os.path.join(folder(workspace, rec['id']), STOP))
    except OSError:
        pass
    return b


def step(workspace: str, cid: str, io, project=None, settings=None, complete=None) -> str:
    """One decision of the loop (module doc): 'waiting' (a job runs), 'progress' (something was decided; call again)
    or 'done' (finished: the brief is written). Every decision is logged with its reason."""
    rec = load(workspace, cid)
    if rec.get('finished'):
        return 'done'
    _check_runnable(workspace, rec)
    now = io.now()
    acc = accountant(workspace, rec)
    if not rec.get('started'):
        rec['started'] = utc(now)
        save(workspace, rec)
        log(workspace, cid, 'start', 'the approved campaign started', budget=rec['budget'])
    running = next((s for s in rec['studies'] if s['state'] == 'running'), None)
    sr = stop_requested(workspace, cid)
    if sr:
        if running is not None:
            io.cancel(running['job'])
            running['state'], running['why'] = 'cancelled', 'the campaign was stopped'
            running['elapsed_s'] = round(max(now - float(running.get('started_t') or now), 0.0), 1)
            log(workspace, cid, 'cancel', 'the campaign was stopped: its running job was cancelled',
                study=running['id'])
        save(workspace, rec)
        _finish(workspace, rec, io, f'stopped: {sr.get("why") or "the user asked"}', by=sr.get('by') or 'user')
        return 'done'
    if running is not None:
        st, el = io.state(running['job'])
        if st is None:                               # the job is gone: Falcon or the machine was restarted
            # when it died is not known: the time since its start, at most its estimate, is counted as used
            lost = max(now - float(running.get('started_t') or now), 0.0)
            again = int(running.get('launches') or 1) < MAX_ATTEMPTS
            running['state'] = 'planned' if again else 'failed'
            running['why'] = 'its job was lost (a restart)'
            running.setdefault('lost_s', []).append(round(min(lost, running['est']['seconds']), 1))
            running['job'] = None
            save(workspace, rec)
            log(workspace, cid, 'interrupted', 'the study\'s job is gone (a restart): '
                + ('it may run once more if the budget allows' if again else 'it already ran twice, not again'),
                study=running['id'])
            return 'progress'
        if st in ('queued', 'waiting', 'running'):
            why = None if running.get('cancel_why') else acc.must_cancel(rec, running, now)
            if why:
                io.cancel(running['job'])
                log(workspace, cid, 'cancel', f'{why}: the running job was cancelled', study=running['id'])
                running['cancel_why'] = why
                save(workspace, rec)
            return 'waiting'
        running['elapsed_s'], running['finished'] = round(float(el), 1), utc(now)
        running['state'] = {'done': 'done', 'failed': 'failed', 'cancelled': 'cancelled'}[st]
        h = next(h for h in rec['hypotheses'] if h['id'] == running['hypothesis'])
        if st == 'done':
            try:
                running['result'] = triage(workspace, rec, running, project)
            except Exception as e:                   # noqa: BLE001 (a result that cannot be read is said, not raised)
                running['result'] = {'summary': f'its result could not be read ({type(e).__name__}: {e})',
                                     'verdict': 'inconclusive', 'strength': 'none', 'numbers': {}, 'links': []}
            r = running['result']
            h['status'] = {'evidence': 'some evidence'}.get(r['verdict'], r['verdict'])
            log(workspace, cid, 'result', r['summary'], study=running['id'], verdict=r['verdict'],
                elapsed_s=running['elapsed_s'])
            plan(workspace, rec, project, complete, at_approval=False)       # what the result newly suggests
        else:
            running['why'] = running.get('cancel_why') or f'its job {st}'
            h['status'] = 'not tested'
            log(workspace, cid, 'result', f'the study\'s job {st}' + (f' ({running["why"]})' if st == 'cancelled'
                                                                       else ''), study=running['id'])
        save(workspace, rec)
        return 'progress'
    # nothing runs: stop, or start the next study
    if any(s['state'] == 'done' and (s.get('result') or {}).get('verdict') == 'supported' for s in rec['studies']):
        _finish(workspace, rec, io, 'the goal is answered: a controlled comparison supported a hypothesis')
        return 'done'
    dl = epoch(rec['budget'].get('deadline'))
    if dl is not None and now >= dl:
        _finish(workspace, rec, io, f'the deadline ({rec["budget"]["deadline"]}) is reached')
        return 'done'
    planned = [s for s in rec['studies'] if s['state'] == 'planned']
    order = {h['id']: k for k, h in enumerate(rec['hypotheses'])}
    planned.sort(key=lambda s: order.get(s['hypothesis'], 0))
    refused = False
    for s in planned:
        if s['kind'] not in rec['scope']['kinds'] or s['kind'] not in KINDS:
            s['state'], s['why'] = 'refused', 'its kind is not in the approved scope'
            log(workspace, cid, 'refuse', s['why'], study=s['id'])
            continue
        ok, why = acc.may_start(rec, s, now)
        if not ok:
            s['state'], s['why'] = 'refused', why
            refused = True
            log(workspace, cid, 'refuse', f'not started: {why}', study=s['id'])
            continue
        try:
            ref = _launch(workspace, rec, s, io, project, settings)
        except Exception as e:                       # noqa: BLE001 (StudyError, AblationError, a queue that refuses)
            s['state'], s['why'] = 'failed', f'it could not be started: {e}'
            log(workspace, cid, 'result', s['why'], study=s['id'])
            continue
        s['state'], s['job'], s['started'], s['started_t'] = 'running', ref, utc(now), now
        s['launches'] = int(s.get('launches') or 0) + 1
        s.pop('why', None)
        save(workspace, rec)
        log(workspace, cid, 'run', f'{s["title"]}: the highest-ranked hypothesis still open; {why}', study=s['id'],
            hypothesis=s['hypothesis'], estimate_s=s['est']['seconds'])
        return 'progress'
    save(workspace, rec)
    left = [s for s in rec['studies'] if s['state'] == 'refused']
    _finish(workspace, rec, io, 'the budget is used up: ' + '; '.join(dict.fromkeys(s['why'] for s in left))
            if refused else 'nothing is left that is worth its cost: every planned study ran or could not run')
    return 'done'


def run(workspace: str, cid: str, io, project=None, settings=None, complete=None, poll_s: float = 2.0,
        sleep=time.sleep, max_steps: int | None = None) -> dict:
    """The loop until the campaign is finished. One runner at a time (a lock in the campaign's folder). Whatever
    ends it — the end, Stop, an error, an interrupt — no job of the campaign is left running (io.cancel_all)."""
    from .jobs import FileLock
    lock = FileLock('run.lock', folder(workspace, cid))
    if not lock.acquire():
        raise CampaignError(f'campaign {cid} is being run already (another `falcon campaign run`)')
    n = 0
    try:
        while True:
            r = step(workspace, cid, io, project, settings, complete)
            n += 1
            if r == 'done' or (max_steps is not None and n >= max_steps):
                break
            if r == 'waiting':
                sleep(poll_s)
    except BaseException as e:
        try:
            log(workspace, cid, 'interrupted', f'the runner ended ({type(e).__name__}): its jobs are cancelled; '
                                               '`falcon campaign run` resumes')
        except OSError:
            pass
        raise
    finally:
        cancel_all = getattr(io, 'cancel_all', None)
        if cancel_all is not None:
            cancel_all()
        lock.release()
    return status(workspace, cid)


def stop(workspace: str, cid: str, why: str = '', by: str = 'user', io=None) -> dict:
    """Ask a campaign to stop: stop.json is written (the runner sees it at its next step, cancels its job and writes
    the brief). When no runner holds the campaign's lock, it is finished here."""
    from .jobs import FileLock
    from .paths import atomic_write
    rec = load(workspace, cid)
    if rec.get('finished'):
        return status(workspace, cid)
    atomic_write(os.path.join(folder(workspace, cid), STOP),
                 json.dumps({'by': by, 'why': why or 'the user asked', 'time': utc()}))
    log(workspace, cid, 'stop_requested', why or 'the user asked', by=by)
    lock = FileLock('run.lock', folder(workspace, cid))
    if lock.acquire():                               # no runner: nothing of it is running; finish the record now
        try:
            if rec.get('approval'):
                step(workspace, cid, io or _NoIO())
            else:
                rec['finished'] = utc()
                rec['stopped'] = {'by': by, 'why': 'stopped before it was approved', 'time': rec['finished']}
                rec['answer'] = _answer(rec)
                save(workspace, rec)
                brief(workspace, cid)
        finally:
            lock.release()
    return status(workspace, cid)


class _NoIO:
    """No queue: what a stop without a runner needs (nothing of the campaign runs)."""

    def now(self):
        return time.time()

    def state(self, ref):
        return None, 0.0

    def cancel(self, ref):
        pass

    def notify(self, title, body):
        pass

    def submit(self, cid, study):
        raise CampaignError('no job queue here')


def status(workspace: str, cid: str, now: float | None = None) -> dict:
    """A campaign at a glance (nothing written)."""
    rec = load(workspace, cid)
    now = time.time() if now is None else now
    u = accountant(workspace, rec).used(rec, now if not rec.get('finished') else None)
    b = rec['budget']
    return {'id': rec['id'], 'goal': rec['goal'], 'state': state_of(rec), 'planner': rec['planner'],
            'approved': bool(rec.get('approval')), 'scope': rec['scope']['kinds'], 'budget': b,
            'used': {'gpu_hours': u['gpu_hours'], 'jobs': u['jobs'], 'disk_gb': u['disk_gb']},
            'left': {'gpu_hours': round(max(float(b['gpu_hours']) - u['gpu_hours'], 0.0), 4),
                     'jobs': max(int(b['max_jobs']) - u['jobs'], 0),
                     'seconds': None if not b.get('deadline') else round(epoch(b['deadline']) - now, 1)},
            'studies': [{k: s.get(k) for k in ('id', 'kind', 'title', 'state', 'why', 'elapsed_s')}
                        | {'estimate_s': s['est']['seconds'], 'result': (s.get('result') or {}).get('summary'),
                           'verdict': (s.get('result') or {}).get('verdict')} for s in rec['studies']],
            'hypotheses': [{k: h.get(k) for k in ('id', 'text', 'status', 'study')} for h in rec['hypotheses']],
            'needs_you': [{k: n.get(k) for k in ('what', 'why')} for n in rec['needs_you']],
            'answer': rec.get('answer'), 'stopped': rec.get('stopped'),
            'brief': os.path.join(folder(workspace, cid), BRIEF_MD)
            if os.path.isfile(os.path.join(folder(workspace, cid), BRIEF_MD)) else None,
            'log': read_log(workspace, cid, 12)}


# ================================================================================================ the brief
def next_step(rec: dict) -> str:
    """The recommended next step, by rule."""
    ans = rec.get('answer') or {}
    refused = [s for s in rec['studies'] if s['state'] in ('refused', 'not run')]
    if ans.get('confidence') == 'high':
        return 'Act on the supported hypothesis: reproduce it once (falcon reproduce), then fix or add data for it.'
    abl = [s for s in refused if s['kind'] == 'ablation']
    if abl:
        return (f'Run the controlled comparison that did not fit the budget: {abl[0]["title"]} (about '
                f'{_fmt_s(abl[0]["est"]["seconds"])}) — it is the study that can turn the lead into a cause.')
    if rec['needs_you']:
        return f'Decide the first "needs you" item: {rec["needs_you"][0]["what"]}.'
    if refused:
        return f'Give the next campaign more budget: {len(refused)} planned study(ies) did not run.'
    if ans.get('confidence') in ('moderate', 'low'):
        return 'State the lead as a hypothesis and run an ablation on it (the Research view, or research_autopilot).'
    return 'Record more of the conditions the goal names (drives or a search), then run the campaign again.'


def brief_markdown(workspace: str, rec: dict) -> str:
    ans = rec.get('answer') or _answer(rec)
    u = accountant(workspace, rec).used(rec)
    b = rec['budget']
    by = {h['id']: h for h in rec['hypotheses']}
    L = [f'# Morning brief — campaign {rec["id"]}', '', f'**Goal.** {rec["goal"]}', '',
         f'**Answer.** {ans["text"]} {ans["second"]}', '',
         f'**Confidence: {ans["confidence"]}** — {CONFIDENCE[ans["confidence"]]}.', '',
         '## What was tried', '', '| Study | Kind | Cost | Result |', '|---|---|---|---|']
    for s in rec['studies']:
        if s['state'] in ('done', 'failed', 'cancelled'):
            cost = _fmt_s(s.get('elapsed_s')) + (' GPU' if KINDS[s['kind']]['gpu'] else ' CPU')
            res = (s.get('result') or {}).get('summary') or s.get('why') or s['state']
            L.append(f'| {s["id"]} {s["title"]} | {s["kind"]} | {cost} | {str(res).replace("|", "/")} |')
    if not any(s['state'] in ('done', 'failed', 'cancelled') for s in rec['studies']):
        L.append('| (nothing ran) | | | |')
    ruled = [s for s in rec['studies'] if (s.get('result') or {}).get('verdict') in ('ruled out', 'not supported')]
    L += ['', '## Ruled out', '']
    L += [f'- {by[s["hypothesis"]]["text"]} — {s["result"]["verdict"]}: {s["result"]["summary"]}' for s in ruled] or \
        ['- Nothing was ruled out.']
    notrun = [s for s in rec['studies'] if s['state'] in ('refused', 'not run')]
    if notrun:
        L += ['', '## Planned but not run', ''] + [f'- {s["id"]} {s["title"]} — {s.get("why")}' for s in notrun]
    L += ['', '## Evidence', '']
    links = [(s['id'], p) for s in rec['studies'] for p in ((s.get('result') or {}).get('links') or ()) if p]
    L += [f'- {sid}: `{os.path.relpath(p, workspace) if os.path.isabs(p) and p.startswith(workspace) else p}`'
          for sid, p in links] or ['- No study left a result.']
    L += [f'- the decision log: `{os.path.join(FOLDER, rec["id"], LOG)}`']
    L += ['', '## Budget', '', '| | Used | Given |', '|---|---|---|',
          f'| GPU-hours | {u["gpu_hours"]:.2f} | {float(b["gpu_hours"]):g} |',
          f'| Jobs | {u["jobs"]} | {b["max_jobs"]} |', f'| Disk (GB) | {u["disk_gb"]:.2f} | {float(b["disk_gb"]):g} |',
          f'| Time | {rec.get("started") or "—"} → {rec.get("finished") or "—"} | until {b["deadline"] or "no deadline"} |',
          '', f'Stopped because: {(rec.get("stopped") or {}).get("why") or "it is still running"}.',
          '', '## Needs you', '']
    L += [f'- {n["what"]} — {n["why"]}' for n in rec['needs_you']] or ['- Nothing.']
    L += ['', '## Recommended next step', '', next_step(rec), '',
          '---', 'Limits: the answer is assembled by rule from the result files of the studies that ran; it is only as '
          'good as those studies. Cost estimates come from this workspace\'s job history. GPU-hours are the wall time '
          'of GPU-holding studies. Runs derived from dashcam footage are not read.']
    return '\n'.join(L) + '\n'


def brief_html(md: str, title: str) -> str:
    """The brief as a page in core.reports' style (its CSS; the Markdown's headings, paragraphs, lists and tables)."""
    from .reports import CSS, esc

    def inline(t):
        t = esc(t)
        t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        return re.sub(r'`(.+?)`', r'<span class="mono">\1</span>', t)
    parts, rows, items = [], [], []

    def flush():
        if rows:
            head, body = rows[0], rows[2:] if len(rows) > 1 else []
            parts.append('<table><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in head) + '</tr></thead><tbody>'
                         + ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in body)
                         + '</tbody></table>')
            rows.clear()
        if items:
            parts.append('<ul>' + ''.join(f'<li>{inline(i)}</li>' for i in items) + '</ul>')
            items.clear()
    for line in md.splitlines():
        if line.startswith('|'):
            if items:
                flush()
            rows.append([c.strip() for c in line.strip().strip('|').split('|')])
            continue
        if line.startswith('- '):
            if rows:
                flush()
            items.append(line[2:])
            continue
        flush()
        if line.startswith('# '):
            parts.append(f'<h1>{inline(line[2:])}</h1>')
        elif line.startswith('## '):
            parts.append(f'<h2>{inline(line[3:])}</h2>')
        elif line.strip() == '---':
            parts.append('<hr>')
        elif line.strip():
            parts.append(f'<p>{inline(line)}</p>')
    flush()
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body><main>' + ''.join(parts)
            + '</main></body></html>\n')


def brief(workspace: str, cid: str) -> dict:
    """Write brief.md and brief.html from the record as it stands; {'markdown', 'md', 'html', 'final'}."""
    from .paths import atomic_write
    rec = load(workspace, cid)
    if rec.get('answer') is None:
        rec = dict(rec, answer=_answer(rec))
    md = brief_markdown(workspace, rec)
    d = folder(workspace, cid)
    atomic_write(os.path.join(d, BRIEF_MD), md)
    atomic_write(os.path.join(d, BRIEF_HTML), brief_html(md, f'Morning brief — campaign {cid}'))
    return {'markdown': md, 'md': os.path.join(d, BRIEF_MD), 'html': os.path.join(d, BRIEF_HTML),
            'final': bool(rec.get('finished'))}


# ================================================================================================ the command line
def _ws_project(arg: str) -> tuple:
    """(workspace, project path or '', Project or None) of a project file or a workspace folder."""
    from .paths import PROJECT_SUFFIX, workspace_for
    if arg.endswith(PROJECT_SUFFIX):
        from .project import Project
        proj = Project.load(arg)
        return proj.workspace or workspace_for(arg), os.path.abspath(arg), proj
    return arg, '', None


def _completer(goal: str):
    from ..assistant.service import AssistantService
    from .settings import Settings
    return AssistantService(Settings()).scenario_writer(goal)       # refuses a goal naming a private location


def _print_status(st: dict):
    b, u, left = st['budget'], st['used'], st['left']
    print(f'campaign {st["id"]}: {st["state"]} — {st["goal"]}')
    print(f'  GPU-hours {u["gpu_hours"]:.2f} of {float(b["gpu_hours"]):g} · jobs {u["jobs"]} of {b["max_jobs"]} · '
          f'disk {u["disk_gb"]:.2f} of {float(b["disk_gb"]):g} GB · deadline {b["deadline"] or "none"}'
          + (f' ({left["seconds"] / 60.0:.0f} min left)' if left['seconds'] is not None and st['state'] != 'done'
             else ''))
    for s in st['studies']:
        print(f'  {s["id"]:4} {s["state"]:9} [{s["kind"]}] {s["title"]}'
              + (f' — {s["result"]}' if s.get('result') else f' — {s["why"]}' if s.get('why') else ''))
    for n in st['needs_you']:
        print(f'  needs you: {n["what"]} — {n["why"]}')
    if st.get('answer'):
        print(f'  answer ({st["answer"]["confidence"]} confidence): {st["answer"]["text"]}')
    if st.get('brief'):
        print(f'  brief: {st["brief"]}')


def main_campaign(argv) -> int:
    """falcon campaign new|approve|run|status|stop|brief|list (docs/campaign.md)."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon campaign', description='An overnight campaign: a goal and a budget in, '
                                                                     'a one-page brief out (docs/campaign.md).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    wh = 'a project file (*.falcon.json) or its workspace folder'
    nw = sub.add_parser('new', help='draft a campaign: its plan and costs are shown; nothing runs')
    nw.add_argument('workspace', help=wh)
    nw.add_argument('goal', help='what should be found out, in a sentence')
    nw.add_argument('--id', default='')
    nw.add_argument('--gpu-hours', type=float, default=DEFAULT_BUDGET['gpu_hours'])
    nw.add_argument('--until', default='', help='the deadline: 07:00 (the next time the clock shows it), +6h, +90m or '
                                                'a UTC time YYYY-MM-DDTHH:MM:SSZ')
    nw.add_argument('--max-jobs', type=int, default=DEFAULT_BUDGET['max_jobs'])
    nw.add_argument('--disk-gb', type=float, default=DEFAULT_BUDGET['disk_gb'])
    nw.add_argument('--kinds', default=','.join(DEFAULT_KINDS), help='allowed study kinds, comma-separated: '
                                                                     + ', '.join(KINDS))
    nw.add_argument('--planner', choices=PLANNERS, default='playbook',
                    help='playbook: rule-based, no model; assistant: the configured provider orders the candidates')
    nw.add_argument('--json', action='store_true')
    for name, text in (('approve', 'show the whole scope and budget and approve it (once)'),
                       ('run', 'run an approved campaign until it is finished (resumes after a restart)'),
                       ('status', 'budgets used, the studies and the last decisions'),
                       ('stop', 'stop it: its running job is cancelled and the brief written'),
                       ('brief', 'write and print the brief (final once the campaign is finished)')):
        p = sub.add_parser(name, help=text)
        p.add_argument('workspace', help=wh)
        p.add_argument('id')
        if name == 'approve':
            p.add_argument('--yes', action='store_true', help='approve without the question (the scope is still '
                                                              'printed)')
        if name == 'run':
            p.add_argument('--fake-carla', default='', help='for tests: the fake CARLA\'s folder')
            p.add_argument('--poll', type=float, default=5.0, help='seconds between looks at a running job')
            p.add_argument('--detached', action='store_true', help='print the systemd-run --user command that leaves '
                                                                   'it running, and exit (Falcon does not run it)')
        if name == 'stop':
            p.add_argument('--why', default='')
        if name in ('status', 'brief'):
            p.add_argument('--json', action='store_true')
    ls = sub.add_parser('list', help='the workspace\'s campaigns')
    ls.add_argument('workspace', help=wh)
    a = ap.parse_args(argv)
    try:
        ws, proj_path, proj = _ws_project(a.workspace)
        if a.cmd == 'list':
            for rec in campaigns(ws):
                print(f'{rec["id"]:20} {state_of(rec):9} {rec["goal"][:80]}')
            return 0
        if a.cmd == 'new':
            complete = _completer(a.goal) if a.planner == 'assistant' else None
            rec = new(ws, a.goal, {'gpu_hours': a.gpu_hours, 'max_jobs': a.max_jobs, 'disk_gb': a.disk_gb,
                                   'deadline': parse_until(a.until) if a.until else None},
                      [k.strip() for k in a.kinds.split(',') if k.strip()], a.id, proj_path, proj, a.planner, complete)
            if a.json:
                print(json.dumps(rec, indent=1, default=str))
            else:
                print(approval_text(rec))
                print(f'\nNothing runs yet. Approve it: falcon campaign approve {a.workspace} {rec["id"]}')
            return 0
        if a.cmd == 'approve':
            rec = load(ws, a.id)
            print(approval_text(rec))
            if not a.yes:
                try:
                    ans = input('\nApprove this scope and budget? [y/N] ')
                except EOFError:
                    ans = ''
                if ans.strip().lower() not in ('y', 'yes'):
                    print('not approved')
                    return 1
            approve(ws, a.id, {'how': 'command line'})
            print(f'\napproved. Run it: falcon campaign run {a.workspace} {a.id}\n'
                  f'or leave it running after this terminal closes (Falcon prints this, it does not run it):\n  '
                  + detached_command(a.workspace, a.id))
            return 0
        if a.cmd == 'run':
            if a.detached:
                print(detached_command(a.workspace, a.id))
                return 0
            rec = load(ws, a.id)
            if not proj and rec['scope'].get('project'):
                from .project import Project
                proj = Project.load(rec['scope']['project'])
            from .settings import Settings
            complete = _completer(rec['goal']) if rec['planner'] == 'assistant' else None
            import signal

            def _term(_sig, _frm):
                raise KeyboardInterrupt
            old = signal.signal(signal.SIGTERM, _term)
            try:
                st = run(ws, a.id, QueueIO(ws, fake_carla=a.fake_carla), proj, Settings(), complete, poll_s=a.poll)
            except KeyboardInterrupt:
                print(f'interrupted: the campaign\'s jobs are cancelled; falcon campaign run {a.workspace} {a.id} '
                      f'resumes it')
                return 130
            finally:
                signal.signal(signal.SIGTERM, old)
            _print_status(st)
            return 0
        if a.cmd == 'stop':
            _print_status(stop(ws, a.id, a.why))
            return 0
        if a.cmd == 'status':
            st = status(ws, a.id)
            if a.json:
                print(json.dumps(st, indent=1, default=str))
            else:
                _print_status(st)
                for row in st['log']:
                    print(f'  {row["time"]} {row["decision"]:12} {row["reason"]}')
            return 0
        b = brief(ws, a.id)
        print(json.dumps({k: b[k] for k in ('md', 'html', 'final')}, indent=1) if a.json else b['markdown'])
        return 0
    except CampaignError as e:
        print(f'falcon campaign {a.cmd}: {e}')
        return 2
