"""The research autopilot: a question or hypothesis taken through the whole research loop — an ablation designed with a
power analysis, approved, run through the job queue, analysed, followed up when it was inconclusive, and written up
as a report whose every number is checked — as a deterministic, resumable state machine (standard library + numpy
through core.ablation; the job queue only through callables the caller passes). docs/research-autopilot.md is the
manual.

The assistant's model fills in the ablation spec and the prose of the report; this module owns everything else: the
sequencing, which step needs the user's approval, the waiting (through the caller's job queue), the follow-up rule,
the report's numbers and its grounding check, and resumption after a restart. A study is the file
experiments/<id>/autopilot.json (SCHEMA) beside its first experiment (core.ablation); each round is an experiment of
the registry (<id>, then <id>-f1, <id>-f2 … for the follow-ups).

States (STATES). What is stored is the user's and the model's decisions (the rounds and their approvals, a stop, the
report); the state is derived from them and from the files and jobs (derive()), so a study resumes by construction:

    proposed     the model drafted a plan (draft()); its approval card is waiting — nothing is written yet
    approved     approved and registered; its job is not queued yet (queued next, under the same approval)
    running      its job is queued or running in this session
    interrupted  its job is gone (Falcon was restarted, or the job ended without finishing every run): resume
    failed       its job failed (a run's error is in the experiment): resume runs it again from the failed run
    analysed     every run finished and the analysis is written; conclusive, or inconclusive with no follow-up
    follow_up    analysed and inconclusive for want of power: a follow-up is proposed (follow_up_proposal())
    reported     the report covers every round (final when every number in it passed the grounding check)
    stopped      the user stopped it (Stop, or a job cancelled — the activity log's Undo); resume continues it

Approvals (COMPUTE: the actions that queue compute — start, follow_up, resume). Each needs the user's approval card
in every permission mode, Autopilot included, unless the user ticked "approve the follow-ups of this study" on the
study's first card: then follow-ups and resumes of this study run without a card — bounded to the approved plan's
factors and levels, its metric, hypothesis and executor (within_bounds; a follow-up can only narrow) — until Stop.
start(), follow_up() and resume() check the approval they are handed (APPROVED_HOW) and refuse without one: the gate
does not rest on the caller. There is no budget: nothing caps the rounds or the replicates beyond the spec's own
max_replicates; Stop ends a study.

The follow-up rule (follow_up_proposal), deterministic. A round is inconclusive when its hypothesis is "not
supported" and none of its contrasts contradicts it. A follow-up is proposed only when an inconclusive contrast was
underpowered — the power the experiment had for the planned effect at the spread it observed is below the target, or
every bootstrap resample went one way (core.ablation's p_at_floor) — never when it had the power and found nothing
(that is an answer: no detectable effect of the planned size). It is a fresh experiment with new seeds, planned for
the planned effect at the largest observed spread of the paired differences; when some contrasts of a "by" split were
supported and others not, it narrows that factor to the inconclusive levels. Rounds are not pooled. Each look halves
alpha (round k at alpha / 2^(k-1)), so the chance of a false "supported" over all the looks of a study stays below
twice the first round's alpha (a Bonferroni bound over the looks), and the report says which look reached the verdict.

The report (report()): the hypothesis, the design and its approvals, the pins, every round's contrasts with their
differences, intervals, effect sizes and adjusted p-values, the verdict, the limits, the paper outputs (core.paper)
and the model's discussion; checked by assistant.grounding against the study's own facts (facts(): the record, each
round's plan summary, spec and results — the per-run list and the seeds' list are left out, so that a run index or a
seed cannot ground an invented number). A number no fact holds is marked in the report and listed; a report with one
is a draft, never final.

Suggestions (suggestions()): questions found in the workspace — failure modes (core.failclusters), operating-domain
cells above the failure target or with too few runs (core.oddcoverage), failure-rate estimates' worst cells
(core.failrate studies), regressions a bisect found (core.regbisect) — each with its facts and, when its conditions
map to the ablation executors' factors (weather and light), a template spec. Anything from the user's dashcam footage
(assistant.privacy) is left out before anything is computed from it: a private failure never shapes a cluster, a cell
or a suggestion.
"""
from __future__ import annotations

import copy
import json
import os
import time

SCHEMA = 'falcon.research_autopilot/1'
FILE = 'autopilot.json'
REPORT = 'report.md'
REPORT_JSON = 'report.json'
STATES = ('proposed', 'approved', 'running', 'interrupted', 'failed', 'analysed', 'follow_up', 'reported',
          'stopped')
COMPUTE = ('start', 'follow_up', 'resume')
APPROVED_HOW = ('approved', 'preapproved', 'command line')
ACTIVE = ('queued', 'waiting', 'running')
FOLLOW_UPS = ('Approve the follow-ups of this study: more replicates or a narrower factor, within this plan\'s '
              'factors and levels, without asking again (Stop ends it)')
SEED_STEP = 1000                 # a follow-up's seeds start this far after the round before's: fresh replicates
MAX_SUGGESTIONS = 8
ASSUMED_SD = 0.06                # a template's spread when no analysed ablation measured one (the manual's example)
ASSUMED_EFFECT = 0.05


class StudyError(Exception):
    pass


def utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ============================================================================================== the record
def _ab():
    from . import ablation as AB
    return AB


def study_dir(workspace: str, sid: str) -> str:
    return _ab().path_of(workspace, sid)


def path_of(workspace: str, sid: str) -> str:
    return os.path.join(study_dir(workspace, sid), FILE)


def load(workspace: str, sid: str) -> dict:
    from .provenance import read_json
    p = path_of(workspace, sid)
    try:
        rec = read_json(p)
    except (OSError, ValueError):
        raise StudyError(f'no research study {sid!r} in the workspace (experiments/{sid}/{FILE})') from None
    if not isinstance(rec, dict) or rec.get('schema') != SCHEMA:
        raise StudyError(f'experiments/{sid}/{FILE} is not a research study')
    return rec


def save(workspace: str, rec: dict):
    from .provenance import write_json
    write_json(path_of(workspace, rec['id']), rec)


def studies(workspace: str) -> list:
    """Every study of the workspace (its id and record), newest first."""
    out = []
    root = _ab().root(workspace)
    for n in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        if os.path.isfile(os.path.join(root, n, FILE)):
            try:
                out.append(load(workspace, n))
            except StudyError:
                continue
    out.sort(key=lambda r: (r.get('created') or '', r['id']), reverse=True)
    return out


def _note(rec: dict, state: str, why: str, by: str):
    h = rec.setdefault('history', [])
    if not h or h[-1].get('state') != state or h[-1].get('why') != why:
        h.append({'time': utc(), 'state': state, 'why': why, 'by': by})
    rec['state'] = state


# ============================================================================================== a round's facts
def _spec(workspace: str, exp_id: str) -> tuple:
    AB = _ab()
    try:
        exp, s, pl = AB.load(workspace, exp_id)
    except AB.AblationError as e:
        raise StudyError(str(e)) from None
    return exp, s, pl


def _results(workspace: str, exp_id: str) -> dict | None:
    from .provenance import read_json
    p = os.path.join(study_dir(workspace, exp_id), 'results.json')
    try:
        return read_json(p) if os.path.isfile(p) else None
    except (OSError, ValueError):
        return None


def plan_summary(pl: dict) -> dict:
    pw, c = pl['power'], pl['counts']
    return {'cells': c['cells'], 'replicates': c['replicates'], 'runs': c['runs'], 'drives': c.get('drives', 0),
            'evaluations': c.get('evaluations', 0), 'dz': pw.get('dz'), 'alpha': pw.get('alpha'),
            'alpha_per_contrast': pw.get('alpha_per_contrast'), 'target_power': pw.get('target_power'),
            'capped': pw.get('capped'), 'contrasts': [k['label'] for k in pl['contrasts']],
            'power': [r.get('power') for r in pw.get('contrasts') or ()], 'seeds': [pl['seeds'][0], pl['seeds'][-1]],
            'notes': list(pw.get('notes') or ())}


def cost(spec: dict, pl: dict, project=None, settings=None) -> dict:
    """{'seconds', 'what', 'label'} of a plan: for the detection executor its drives at the throughput measured on
    this machine (core.budget.Throughput, 'typical' until a drive measured it); evaluations are counted, not timed."""
    c = pl['counts']
    if spec['run']['executor'] != 'detection':
        return {'seconds': None, 'label': 'simulated',
                'what': f'{c["runs"]} runs drawn from the spec\'s stated model (SIMULATED: no simulator time), one job'}
    from .budget import Throughput, fmt_duration, rig_mpx
    doc = getattr(project, 'doc', None) or {}
    rigs = {r.get('id'): r for r in doc.get('rigs') or ()}
    ids = [str(f['values'][lv]) for f in spec['factors'] if f['kind'] == 'rig' for lv in f['levels']]
    ids = ids or [spec['run'].get('rig') or next(iter(rigs), None)]
    mpx = max((rig_mpx(rigs[i]) for i in ids if i in rigs), default=0.0)
    measured = settings.get('measured') if settings is not None and hasattr(settings, 'get') else None
    thr = Throughput.from_settings(measured)
    steps = int(round(float(spec['run']['seconds']) * float(spec['run']['fps'])))
    est = thr.estimate(steps, mpx)
    total = c['drives'] * est['seconds']
    return {'seconds': total, 'label': est['label'],
            'what': (f'{c["drives"]} drives of {float(spec["run"]["seconds"]):g} s '
                     f'(about {fmt_duration(est["seconds"])} '
                     f'each at the {est["label"]} rate) and {c["evaluations"]} evaluations (not timed: an '
                     'evaluation\'s time depends on the model); one job, the drives under the CARLA lock')}


def _brief_contrast(r: dict) -> dict:
    return {k: r.get(k) for k in ('id', 'label', 'n_pairs', 'mean_a', 'mean_b', 'diff', 'ci_low', 'ci_high',
                                  'sim_ci_low', 'sim_ci_high', 'sd_diff', 'dz', 'dz_ci_low', 'dz_ci_high',
                                  'p_adjusted', 'verdict', 'power_at_planned', 'note')}


# ============================================================================================== drafting
def _private_world(project, workspace: str, wid: str) -> str | None:
    from ..assistant.privacy import private_reason
    w = project.item('worlds', wid) if project is not None else None
    x = (w or {}).get('xodr')
    if not x:
        return None
    p = x if os.path.isabs(x) else os.path.join(workspace, x)
    return private_reason(os.path.dirname(p))


def _check_private(s: dict, project, workspace: str):
    worlds = [str(f['values'][lv]) for f in s['factors'] if f['kind'] == 'world' for lv in f['levels']]
    if s['run'].get('world'):
        worlds.append(s['run']['world'])
    for w in worlds:
        why = _private_world(project, workspace, w)
        if why:
            raise StudyError(f'world {w!r} is private ({why}): a study never runs on the dashcam footage')


def draft(workspace: str, sid: str, question: str, spec: dict, project=None, settings=None,
          source: dict | None = None) -> dict:
    """A proposal (nothing written): the spec normalised under the study's id, its plan, the cost and the card's text.
    Raises StudyError (a used id, a spec that cannot be run, a private world)."""
    AB = _ab()
    if not str(question or '').strip():
        raise StudyError('state the question or hypothesis in the user\'s words (question)')
    if not isinstance(spec, dict):
        raise StudyError('give the ablation spec (spec: hypothesis, metric, factors, contrasts, design, run)')
    sid = AB._safe_id(sid or spec.get('id') or '')
    if os.path.exists(study_dir(workspace, sid)):
        raise StudyError(f'experiments/{sid} exists already: choose another study id')
    try:
        s = AB.normalise(dict(spec, id=sid), project)
        pl = AB.plan(s)
    except AB.AblationError as e:
        raise StudyError(str(e)) from None
    _check_private(s, project, workspace)
    c = cost(s, pl, project, settings)
    return {'study': sid, 'question': str(question).strip(), 'source': dict(source or {'kind': 'user'}), 'spec': s,
            'plan': plan_summary(pl), 'cost': c, 'text': card_text(question, s, pl, c, 'start')}


def card_text(question: str, s: dict, pl: dict, c: dict, kind: str, why: str = '') -> str:
    """What an approval card shows for a round: the question, the hypothesis, the design, the power analysis, the
    counts and the cost, and what happens after."""
    h = s['hypothesis']
    lines = []
    if kind == 'start':
        lines.append(f'Research study {s["id"]}: "{question}"')
    else:
        lines.append(f'Follow-up {s["id"]} ({kind.replace("_", " ")}): {why}')
    lines.append(f'hypothesis: {h["text"]} ({h["direction"]}, rule {h["rule"]}); metric {s["metric"]}')
    for f in s['factors']:
        lines.append(f'factor {f["name"]} ({f["kind"]}): {", ".join(f["levels"])}')
    lines.append(_ab().describe_plan(pl))
    lines.append(f'cost: {c["what"]}')
    if s['run']['executor'] == 'simulated':
        lines.append('SIMULATED: the values come from the spec\'s stated model; they test the design, not the '
                     'hypothesis')
    lines.append('after: Falcon waits for the job, analyses it and, when it was inconclusive for want of power, '
                 'proposes a follow-up on another card (unless you approve the follow-ups now)')
    return '\n'.join(lines)


# ============================================================================================== approvals
def _approved(approval) -> dict:
    a = dict(approval or {})
    if a.get('how') not in APPROVED_HOW:
        raise StudyError('this step queues compute: it runs only after the user approves its card (the research '
                         'autopilot asks at each step that runs anything, in every permission mode)')
    a.setdefault('time', utc())
    return {k: a.get(k) for k in ('how', 'time', 'options', 'by') if a.get(k) is not None}


def within_bounds(rec: dict, spec: dict) -> tuple:
    """(True, '') when a follow-up's (normalised) spec stays within the study's first, approved plan: the same
    factors, each level one of the approved levels with the same value, the same metric, hypothesis, executor and run
    settings. Else (False, why)."""
    first = rec['rounds'][0]['spec']
    if spec['metric'] != first['metric'] or spec['hypothesis']['text'] != first['hypothesis']['text'] or \
            spec['hypothesis']['direction'] != first['hypothesis']['direction']:
        return False, 'a follow-up keeps the study\'s metric and hypothesis'
    keep = ('executor', 'world', 'rig', 'traffic', 'seconds', 'fps', 'every', 'camera', 'model', 'simulate')
    if any(spec['run'].get(k) != first['run'].get(k) for k in keep):
        return False, 'a follow-up keeps the study\'s executor and run settings'
    fa = {f['name']: f for f in first['factors']}
    if [f['name'] for f in spec['factors']] != list(fa):
        return False, f'a follow-up keeps the study\'s factors ({", ".join(fa)}); narrower levels only'
    for f in spec['factors']:
        g = fa[f['name']]
        if f['kind'] != g['kind']:
            return False, f'factor {f["name"]} keeps its kind'
        for lv in f['levels']:
            if lv not in g['levels'] or json.dumps(f['values'][lv], sort_keys=True) != \
                    json.dumps(g['values'][lv], sort_keys=True):
                return False, f'{f["name"]}={lv} is not one of the approved levels'
    return True, ''


def preapproved(rec: dict, action: str, spec: dict | None = None) -> tuple:
    """(True, why) when the user approved this study's follow-ups ahead (the first card's box) and the step stays
    within the approved plan; else (False, '')."""
    fu = rec.get('follow_ups')
    if not fu or not fu.get('approved') or rec.get('stopped'):
        return False, ''
    if action == 'follow_up' and spec is not None:
        ok, _why = within_bounds(rec, spec)
        if not ok:
            return False, ''
    if action not in ('follow_up', 'resume'):
        return False, ''
    return True, f'the user approved the follow-ups of this study on its first card ({fu.get("time")})'


# ============================================================================================== the rounds
def _round(rec: dict, exp_id: str, kind: str, why: str, s: dict, pl: dict, approval: dict) -> dict:
    r = {'round': len(rec['rounds']) + 1, 'experiment': exp_id, 'kind': kind, 'why': why,
         'alpha': float(s['design']['alpha']), 'spec': s, 'plan': plan_summary(pl), 'approval': approval, 'jobs': []}
    rec['rounds'].append(r)
    return r


def _queue(workspace: str, rec: dict, r: dict, submit) -> dict:
    """Queue a round's run through the caller's submit(exp_id) -> {'id', 'submitted'}; recorded on the round."""
    j = dict(submit(r['experiment']))
    j['time'] = utc()
    r['jobs'].append({k: j.get(k) for k in ('id', 'submitted', 'time')})
    rec['stopped'] = None
    _note(rec, 'running', f'job {j["id"]} queued for {r["experiment"]}', 'falcon')
    save(workspace, rec)
    return j


def start(workspace: str, proposal: dict, approval: dict, submit, project=None) -> dict:
    """The approved proposal: the first experiment registered, the study written, its job queued. Returns the
    record. approval: {'how', 'options': {'follow_ups': bool}} as the card was answered (APPROVED_HOW)."""
    AB = _ab()
    ap = _approved(approval)
    sid = proposal['study']
    if os.path.exists(study_dir(workspace, sid)):
        raise StudyError(f'experiments/{sid} exists already')
    try:
        c = AB.create(workspace, proposal['spec'], project)
    except AB.AblationError as e:
        raise StudyError(str(e)) from None
    _e, s, pl = _spec(workspace, c['id'])
    rec = {'schema': SCHEMA, 'id': c['id'], 'question': proposal['question'], 'source': proposal.get('source'),
           'created': utc(), 'state': 'proposed', 'rounds': [], 'follow_ups': None, 'stopped': None, 'report': None,
           'history': [{'time': proposal.get('time') or ap['time'], 'state': 'proposed', 'why': 'the plan was drafted',
                        'by': 'assistant'}]}
    if (ap.get('options') or {}).get('follow_ups'):
        rec['follow_ups'] = {'approved': True, 'time': ap['time'],
                             'bound': {f['name']: list(f['levels']) for f in s['factors']}}
    _round(rec, c['id'], 'initial', 'the stated hypothesis', s, pl, ap)
    _note(rec, 'approved', f'approved ({ap["how"]})' + (' with its follow-ups' if rec['follow_ups'] else ''), 'user')
    save(workspace, rec)
    _queue(workspace, rec, rec['rounds'][-1], submit)
    return rec


def _complete(workspace: str, exp_id: str) -> tuple:
    """(complete, analysed, results): every planned run ok, and results.json analysing all of them."""
    AB = _ab()
    st = AB.status(study_dir(workspace, exp_id))
    res = _results(workspace, exp_id)
    planned = st.get('planned') or 0
    ok = (st.get('runs') or {}).get('ok', 0)
    complete = planned > 0 and ok >= planned
    analysed = complete and res is not None and (res.get('runs') or {}).get('ok', 0) >= planned
    return complete, analysed, res


def derive(workspace: str, rec: dict, job_state=None) -> tuple:
    """(state, why) of a study from its record, its experiments' files and its jobs. job_state(job record) -> the
    job's state in this session's queue, or None when it is not there (another session: a restart)."""
    if rec.get('stopped'):
        return 'stopped', rec['stopped'].get('why') or 'stopped by the user'
    if not rec.get('rounds'):
        return 'proposed', 'not approved yet'
    r = rec['rounds'][-1]
    js = job_state(r['jobs'][-1]) if r['jobs'] and job_state is not None else None
    if js in ACTIVE:
        return 'running', f'job {r["jobs"][-1]["id"]} is {js}'
    complete, analysed, res = _complete(workspace, r['experiment'])
    if analysed:
        rep = rec.get('report') or {}
        if rep.get('rounds') == len(rec['rounds']):
            return 'reported', ('final' if rep.get('final') else 'a draft: it has numbers not from Falcon\'s data')
        fu = follow_up_proposal(workspace, rec)
        if fu.get('spec') is not None:
            return 'follow_up', fu['why']
        return 'analysed', fu['why']
    if complete:
        return 'interrupted', 'every run finished; the analysis is not written yet (resume writes it)'
    if js == 'cancelled':
        return 'stopped', f'job {r["jobs"][-1]["id"]} was cancelled'
    st = _ab().status(study_dir(workspace, r['experiment']))
    if js == 'failed' or st['status'] == 'failed':
        return 'failed', 'a run failed (the experiment\'s run says why); resume runs it again from there'
    if not r['jobs']:
        return 'approved', 'approved; its job is not queued yet'
    if js == 'done':
        return 'interrupted', 'its job ended without finishing every run'
    return 'interrupted', 'its job is not in this session\'s queue (Falcon was restarted): resume continues it'


def sync(workspace: str, rec: dict, job_state=None, by: str = 'falcon') -> tuple:
    """derive(), and the transition written into the record's history when the state changed."""
    state, why = derive(workspace, rec, job_state)
    if state != rec.get('state'):
        if state in ('analysed', 'follow_up', 'reported'):
            res = _results(workspace, rec['rounds'][-1]['experiment']) or {}
            rec['rounds'][-1]['verdict'] = res.get('verdict')
        _note(rec, state, why, by)
        save(workspace, rec)
    return state, why


def needs_compute(workspace: str, rec: dict) -> bool:
    """Whether resuming queues runs (else it only writes the analysis)."""
    r = rec['rounds'][-1]
    complete, _a, _r = _complete(workspace, r['experiment'])
    return not complete


def resume(workspace: str, sid: str, approval: dict | None, submit, job_state=None) -> dict:
    """Continue a stopped, failed or interrupted study: the analysis written when every run finished (no compute,
    no approval needed), else its last round queued again (resumable: finished runs are not repeated) — which needs
    the user's approval, or the follow-ups approved ahead."""
    rec = load(workspace, sid)
    state, _why = derive(workspace, rec, job_state)
    if state == 'running':
        raise StudyError(f'{sid} is running already (job {rec["rounds"][-1]["jobs"][-1]["id"]})')
    if state in ('analysed', 'follow_up', 'reported'):
        raise StudyError(f'{sid} is {state}: nothing to resume')
    r = rec['rounds'][-1]
    if not needs_compute(workspace, rec):
        _ab().analyse(workspace, r['experiment'])
        rec['stopped'] = None
        save(workspace, rec)
        sync(workspace, rec, job_state)
        return rec
    ok, why = preapproved(rec, 'resume')                 # never after a Stop: Stop revokes the approval ahead
    ap = _approved(dict(approval or {}, how='preapproved') if ok and (approval or {}).get('how') not in
                   APPROVED_HOW else approval)
    if ap['how'] == 'preapproved' and not ok:
        raise StudyError('the follow-ups of this study were not approved ahead: this step needs its own card')
    r.setdefault('resumes', []).append(dict(ap, why=why or 'approved on its card'))
    rec['stopped'] = None
    _note(rec, 'approved', f'resume approved ({ap["how"]})', 'user')
    save(workspace, rec)
    _queue(workspace, rec, r, submit)
    return rec


def stop(workspace: str, sid: str, cancel=None, job_state=None, by: str = 'user', why: str = '') -> dict:
    """Stop a study: its running job cancelled (cancel(job record)), the stop recorded. Resume continues it."""
    rec = load(workspace, sid)
    r = rec['rounds'][-1] if rec.get('rounds') else None
    cancelled = []
    if r and r['jobs'] and job_state is not None and job_state(r['jobs'][-1]) in ACTIVE and cancel is not None:
        cancel(r['jobs'][-1])
        cancelled.append(r['jobs'][-1]['id'])
    rec['stopped'] = {'time': utc(), 'by': by, 'why': why or f'stopped by the {by}', 'cancelled': cancelled}
    if (rec.get('follow_ups') or {}).get('approved'):    # the approval ahead holds until Stop
        rec['follow_ups'] = dict(rec['follow_ups'], approved=False, revoked=rec['stopped']['time'])
    _note(rec, 'stopped', rec['stopped']['why'], by)
    save(workspace, rec)
    return rec


# ============================================================================================== the follow-up rule
def _sequential_alpha(rec: dict, k: int) -> float:
    return float(rec['rounds'][0]['alpha']) / (2 ** (k - 1))


def follow_up_proposal(workspace: str, rec: dict, project=None, settings=None) -> dict:
    """The deterministic follow-up of the last round (module doc): {'kind', 'why', 'spec', 'plan', 'cost', 'text'},
    or {'spec': None, 'why'} when there is none."""
    AB = _ab()
    r = rec['rounds'][-1]
    res = _results(workspace, r['experiment'])
    if res is None:
        return {'spec': None, 'why': 'not analysed yet'}
    v = res.get('verdict')
    if v in ('supported', 'contradicted'):
        return {'spec': None, 'why': f'conclusive: the hypothesis is {v} (look {r["round"]})'}
    if v == 'not measurable':
        return {'spec': None, 'why': 'nothing measurable: no contrast had two complete units'}
    rows = res.get('contrasts') or []
    inc = [c for c in rows if c.get('verdict') == 'no detectable effect']
    if not inc:
        return {'spec': None, 'why': 'no inconclusive contrast'}
    s = r['spec']
    target = float(s['design']['power'])
    under = [c for c in inc if (c.get('power_at_planned') is not None and c['power_at_planned'] < target)
             or c.get('p_at_floor')]
    if s['design'].get('effect') is None and s['design'].get('dz') is None:
        return {'spec': None, 'why': 'inconclusive, but no effect was planned: the power cannot be judged (state '
                                     'an effect to plan a follow-up)'}
    if not under:
        pw = min((c['power_at_planned'] for c in inc if c.get('power_at_planned') is not None), default=None)
        return {'spec': None, 'why': 'no detectable effect, and the experiment had the target power for the planned '
                                     'effect at the spread it observed' + (f' (at least {100 * pw:.1f} %)'
                                                                           if pw is not None else '')}
    k = len(rec['rounds']) + 1
    ns = copy.deepcopy(s)
    ns['id'] = f'{rec["id"]}-f{k - 1}'
    ns['rerun_of'] = None
    d = ns['design']
    sds = [c['sd_diff'] for c in under if c.get('sd_diff')]
    if d.get('effect') is not None and sds:
        d['sd'], d['dz'] = max(sds), None
    prev = int(r['plan']['replicates'])
    d['replicates'] = None
    d['alpha'] = _sequential_alpha(rec, k)
    d['seed'] = int(s['design']['seed']) + SEED_STEP
    d['min_replicates'] = min(max(int(d['min_replicates']), prev + 1), int(d['max_replicates']))
    kind = 'more_replicates'
    sup = [c for c in rows if c.get('verdict') == 'supported']
    if sup and len(ns['contrasts']) == 1 and ns['contrasts'][0]['by']:
        by = ns['contrasts'][0]['by']
        keep = {b: sorted({c['where'][b] for c in under if b in (c.get('where') or {})}) for b in by}
        for f in ns['factors']:
            if f['name'] in keep and keep[f['name']] and set(keep[f['name']]) != set(f['levels']):
                f['levels'] = [lv for lv in f['levels'] if lv in keep[f['name']]]
                f['values'] = {lv: f['values'][lv] for lv in f['levels']}
                kind = 'narrower'
    try:
        ns = AB.normalise(ns, project)
        pl = AB.plan(ns)
    except AB.AblationError as e:
        return {'spec': None, 'why': f'a follow-up could not be planned: {e}'}
    if pl['power']['replicates'] <= prev and not under[0].get('p_at_floor'):
        return {'spec': None, 'why': f'inconclusive, but the plan at the observed spread asks for no more than the '
                                     f'{prev} replicates already run'}
    worst = min(under, key=lambda c: c.get('power_at_planned') if c.get('power_at_planned') is not None else 0.0)
    why = (f'inconclusive: contrast {worst["label"]} had '
           + (f'{100 * worst["power_at_planned"]:.1f} % power' if worst.get('power_at_planned') is not None
              else 'too few units for its bootstrap')
           + f' for the planned effect at the observed spread {max(sds) if sds else 0:.4g} (target '
           f'{100 * target:.0f} %); a fresh experiment of {pl["power"]["replicates"]} replicates per cell at alpha '
           f'{d["alpha"]:.4g} (look {k}: alpha halves with each look)')
    if kind == 'narrower':
        why += '; narrowed to the inconclusive levels (' + ', '.join(
            f'{f["name"]}: {", ".join(f["levels"])}' for f in ns['factors'] if f['name'] in ns['contrasts'][0]['by']) \
            + ')'
    c = cost(ns, pl, project, settings)
    return {'kind': kind, 'why': why, 'spec': ns, 'plan': plan_summary(pl), 'cost': c,
            'text': card_text(rec['question'], ns, pl, c, kind, why)}


def follow_up(workspace: str, sid: str, approval: dict | None, submit, project=None, settings=None,
              spec: dict | None = None, job_state=None) -> dict:
    """The next round: the proposal (or the model's own spec, which must stay within the approved plan), registered
    and queued. Needs the user's approval, or the follow-ups approved ahead."""
    AB = _ab()
    rec = load(workspace, sid)
    state, why = sync(workspace, rec, job_state)
    if state not in ('follow_up', 'analysed', 'reported'):
        raise StudyError(f'{sid} is {state}: a follow-up comes after an analysed round ({why})')
    prop = follow_up_proposal(workspace, rec, project, settings)
    if spec is not None:
        k = len(rec['rounds']) + 1
        try:
            ns = AB.normalise(dict(spec, id=f'{sid}-f{k - 1}'), project)
        except AB.AblationError as e:
            raise StudyError(str(e)) from None
        ns['design']['alpha'] = _sequential_alpha(rec, k)
        ns['design']['seed'] = int(rec['rounds'][-1]['spec']['design']['seed']) + SEED_STEP     # fresh replicates
        ns = AB.normalise(ns, project)
        first = {f['name']: len(f['levels']) for f in rec['rounds'][0]['spec']['factors']}
        kind = 'narrower' if any(len(f['levels']) < first.get(f['name'], 0) for f in ns['factors']) else \
            'more_replicates'
        why = 'the assistant\'s follow-up within the approved plan'
    elif prop.get('spec') is None:
        raise StudyError(f'no follow-up: {prop["why"]}')
    else:
        ns, kind, why = prop['spec'], prop['kind'], prop['why']
    ok, bwhy = within_bounds(rec, ns)
    if not ok:
        raise StudyError(f'{bwhy}: a broader question is a new study (action start)')
    pre, pwhy = preapproved(rec, 'follow_up', ns)
    if pre and (approval or {}).get('how') not in APPROVED_HOW:
        approval = dict(approval or {}, how='preapproved')
    ap = _approved(approval)
    if ap['how'] == 'preapproved' and not pre:
        raise StudyError('the follow-ups of this study were not approved ahead: this step needs its own card')
    if pre:
        ap['why'] = pwhy
    if os.path.exists(study_dir(workspace, ns['id'])):
        raise StudyError(f'experiments/{ns["id"]} exists already')
    try:
        c = AB.create(workspace, ns, project)
    except AB.AblationError as e:
        raise StudyError(str(e)) from None
    _e, s, pl = _spec(workspace, c['id'])
    _round(rec, c['id'], kind, why, s, pl, ap)
    _note(rec, 'approved', f'follow-up {c["id"]} approved ({ap["how"]})', 'user')
    save(workspace, rec)
    _queue(workspace, rec, rec['rounds'][-1], submit)
    return rec


# ============================================================================================== status
def status(workspace: str, sid: str, job_state=None, project=None, settings=None) -> dict:
    """A study at a glance, for the tools and the Research view (nothing written): its state, its rounds with their
    approvals, jobs, verdicts and contrasts, the follow-up proposed, the report, the history and the next step."""
    rec = load(workspace, sid)
    state, why = derive(workspace, rec, job_state)
    rounds = []
    for r in rec['rounds']:
        res = _results(workspace, r['experiment'])
        st = _ab().status(study_dir(workspace, r['experiment']))
        rounds.append({'round': r['round'], 'experiment': r['experiment'], 'kind': r['kind'], 'alpha': r['alpha'],
                       'plan': {k: r['plan'][k] for k in ('cells', 'replicates', 'runs', 'drives', 'dz',
                                                          'alpha_per_contrast', 'target_power', 'capped')},
                       'approval': {k: r['approval'].get(k) for k in ('how', 'time')},
                       'jobs': [j['id'] for j in r['jobs']], 'runs': st['runs'], 'status': st['status'],
                       'verdict': (res or {}).get('verdict'),
                       'contrasts': [_brief_contrast(c) for c in (res or {}).get('contrasts') or ()][:12]})
    out = {'study': rec['id'], 'question': rec['question'], 'source': rec.get('source'), 'state': state, 'why': why,
           'follow_ups_approved': bool((rec.get('follow_ups') or {}).get('approved')), 'rounds': rounds,
           'report': rec.get('report'), 'history': rec.get('history', [])[-12:]}
    if state == 'follow_up':
        fu = follow_up_proposal(workspace, rec, project, settings)
        out['follow_up'] = {k: fu.get(k) for k in ('kind', 'why', 'plan', 'cost')}
    out['next'] = next_step(state, rec)
    return out


def next_step(state: str, rec: dict) -> str:
    pre = bool((rec.get('follow_ups') or {}).get('approved'))
    card = 'approved ahead: no card' if pre else 'the user approves it on a card'
    return {
        'proposed': 'research_autopilot action start (the user approves the plan on a card)',
        'approved': f'research_autopilot action resume queues its job ({card})',
        'running': 'wait for its job (plan_wait_jobs), then read this study again',
        'interrupted': f'research_autopilot action resume ({card} when runs are left; none when only the analysis '
                       'is)',
        'failed': f'say why it failed (the experiment\'s failed run); research_autopilot action resume runs it again '
                  f'({card})',
        'analysed': 'write the report: research_autopilot action report with your discussion (cite only the numbers '
                    'of this study)',
        'follow_up': f'research_autopilot action follow_up ({card}), or write the report now (action report)',
        'reported': 'done: the report is written' + ('' if (rec.get('report') or {}).get('final') else
                                                     ' as a draft (numbers not from Falcon\'s data are marked): '
                                                     'write it again with only the study\'s numbers'),
        'stopped': 'stopped by the user: do not resume unless the user asks',
    }[state]


def summary_rows(workspace: str, job_state=None) -> list:
    rows = []
    for rec in studies(workspace):
        state, why = derive(workspace, rec, job_state)
        last = rec['rounds'][-1] if rec.get('rounds') else {}
        res = _results(workspace, last.get('experiment', '')) if last else None
        rows.append({'study': rec['id'], 'question': rec['question'], 'state': state, 'why': why,
                     'rounds': len(rec.get('rounds') or ()), 'verdict': (res or {}).get('verdict'),
                     'report': (rec.get('report') or {}).get('final'), 'created': rec.get('created')})
    return rows


# ============================================================================================== the report
def facts(workspace: str, rec: dict) -> dict:
    """What the report's numbers may come from (module doc): the record, and per round its plan summary, its spec's
    design, analysis, run settings and factor values, and its results (contrasts, cells, design, runs, warnings)."""
    out = {'record': {k: v for k, v in rec.items() if k not in ('history',)}, 'rounds': [],
           'looks': {'count': len(rec['rounds']), 'error_bound': 2 * float(rec['rounds'][0]['alpha'])}}
    fu = follow_up_proposal(workspace, rec)
    out['follow_up'] = {k: fu.get(k) for k in ('why', 'plan')}
    for r in rec['rounds']:
        res = _results(workspace, r['experiment']) or {}
        out['rounds'].append({
            'experiment': r['experiment'], 'plan': r['plan'],
            'spec': {k: r['spec'].get(k) for k in ('design', 'analysis', 'run', 'factors', 'contrasts')},
            'results': {k: res.get(k) for k in ('verdict', 'contrasts', 'cells', 'design', 'replicates', 'runs',
                                                'warnings')}})
    return out


def _messages(workspace: str, rec: dict) -> list:
    return [{'role': 'tool', 'content': json.dumps(facts(workspace, rec), default=str)},
            {'role': 'user', 'content': rec.get('question') or ''}]


def ground(workspace: str, rec: dict, text: str) -> list:
    """The numbers of text that none of the study's facts holds (assistant.grounding's rules)."""
    from ..assistant import grounding
    return grounding.check(text or '', _messages(workspace, rec))


def _f(v, nd=4, signed=False):
    if v is None:
        return '—'
    return f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}'


def _pct(v):
    return '—' if v is None else f'{100 * v:.1f} %'


def _pins_lines(workspace: str, exp_id: str) -> list:
    from .provenance import read_json
    try:
        pins = (read_json(os.path.join(study_dir(workspace, exp_id), 'plan.json')) or {}).get('pins') or {}
    except (OSError, ValueError):
        return ['- (no pins recorded)']
    out = []
    for k, v in sorted(pins.items()):
        if isinstance(v, dict):
            v = ', '.join(f'{kk} {str(vv)[:16]}' for kk, vv in sorted(v.items()))
        elif isinstance(v, list):
            v = ', '.join(str(x) for x in v[:3]) + (' …' if len(v) > 3 else '')
        out.append(f'- {k}: `{str(v)[:160]}`')
    return out or ['- (no pins recorded)']


def report_markdown(workspace: str, rec: dict, prose: str, papers: dict, state_why: str) -> str:
    """The report (module doc), before the grounding marks."""
    first = rec['rounds'][0]['spec']
    h = first['hypothesis']
    last = rec['rounds'][-1]
    res_last = _results(workspace, last['experiment']) or {}
    L = [f'# Research study {rec["id"]}', '', f'**Question.** {rec["question"]}', '']
    src = rec.get('source') or {}
    if src.get('kind') and src['kind'] != 'user':
        L.append(f'**Source.** {src["kind"]} `{src.get("ref", "")}`' + (f' — {src.get("about")}' if src.get('about')
                                                                        else ''))
        L.append('')
    L += ['## Hypothesis', '', f'{h["text"]} — the metric {first["metric"]}, B against A: {h["direction"]}; rule '
          f'{h["rule"]} over the contrasts.', '', '## Design', '']
    for f in first['factors']:
        L.append(f'- factor {f["name"]} ({f["kind"]}): ' + ', '.join(
            f'{lv}' + (f' `{json.dumps(f["values"][lv], sort_keys=True)}`' if f['values'][lv] != lv else '')
            for lv in f['levels']))
    L.append(f'- executor: {first["run"]["executor"]}' + (
        f'; drives of {float(first["run"]["seconds"]):g} s at {float(first["run"]["fps"]):g} steps per second'
        if first['run']['executor'] == 'detection' else ''))
    for r in rec['rounds']:
        p = r['plan']
        ap = r['approval']
        L.append(f'- round {r["round"]} — `{r["experiment"]}` ({r["kind"].replace("_", " ")}): {p["cells"]} cells × '
                 f'{p["replicates"]} replicates = {p["runs"]} runs; ' + (
                     f'power analysis: dz {_f(p["dz"], 3)}, alpha {r["alpha"]:g} ({_f(p["alpha_per_contrast"], 4)} '
                     f'per contrast), target power {_pct(p["target_power"])}' + (' (capped)' if p['capped'] else '')
                     if p.get('dz') is not None else 'replicates given, no power analysis') +
                 f'; approved: {ap.get("how")} ({ap.get("time")})')
        if r['kind'] != 'initial':
            L.append(f'  - why: {r["why"]}')
    L += ['', '## Pins', '']
    for r in rec['rounds']:
        L.append(f'Round {r["round"]} (`{r["experiment"]}`):')
        L += _pins_lines(workspace, r['experiment'])
    L += ['', '## Results', '']
    for r in rec['rounds']:
        res = _results(workspace, r['experiment']) or {}
        L.append(f'Round {r["round"]} (`{r["experiment"]}`), {(res.get("runs") or {}).get("ok", 0)} of '
                 f'{(res.get("runs") or {}).get("planned", 0)} runs: hypothesis **{res.get("verdict")}**.')
        L.append('')
        L.append('| contrast | pairs | A | B | Δ [95 % CI] | simultaneous CI | dz [CI] | p adj | verdict |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for c in res.get('contrasts') or ():
            L.append(f'| {c["label"].replace(" | ", "; ")} | {c["n_pairs"]} | {_f(c["mean_a"])} | {_f(c["mean_b"])} | '
                     f'{_f(c["diff"], signed=True)} [{_f(c["ci_low"], signed=True)}, {_f(c["ci_high"], signed=True)}] '
                     f'| [{_f(c["sim_ci_low"], signed=True)}, {_f(c["sim_ci_high"], signed=True)}] | '
                     f'{_f(c["dz"], 2)} [{_f(c["dz_ci_low"], 2)}, {_f(c["dz_ci_high"], 2)}] | '
                     f'{_f(c["p_adjusted"])} | {c["verdict"]} |')
        notes = [c['note'] for c in res.get('contrasts') or () if c.get('note')]
        for n in notes:
            L.append(f'- {n}')
        L.append('')
    L += ['## Verdict', '']
    v = res_last.get('verdict')
    k = len(rec['rounds'])
    L.append(f'**{v}** at look {k} (round `{last["experiment"]}`, alpha {last["alpha"]:g}). {state_why}.')
    if k > 1:
        L.append(f'The study looked {k} times; alpha halved with each look, so the chance of a false "supported" over '
                 f'all of them stays below {2 * rec["rounds"][0]["alpha"]:g}. Rounds are separate experiments with '
                 'their own seeds and are not pooled.')
    L += ['', '## Discussion', '', (prose or '').strip() or '(none written)', '', '## Limits', '']
    lims = []
    for r in rec['rounds']:
        res = _results(workspace, r['experiment']) or {}
        lims += [f'round {r["round"]}: {w}' for w in res.get('warnings') or ()]
        lims += [f'round {r["round"]}: {n}' for n in r['plan'].get('notes') or ()]
    d0 = first['design']
    if d0.get('sd') is not None and d0.get('effect') is not None:
        lims.append(f'the first plan assumed a spread of the paired differences of {d0["sd"]:g} for an effect of '
                    f'{d0["effect"]:g}; a follow-up is planned at the spread observed')
    lims.append('a verdict of "no detectable effect" is not "no effect": it is an effect smaller than the experiment '
                'could see at its power')
    if first['run']['executor'] == 'detection':
        lims.append('the detection executor varies models, weather, worlds, rigs and traffic; the scenario, its '
                    'parameters and the sensors\' settings are fixed (core.ablation)')
    L += [f'- {x}' for x in dict.fromkeys(lims)]
    L += ['', '## Paper outputs', '']
    for exp_id, p in papers.items():
        if p.get('error'):
            L.append(f'- `{exp_id}`: not written ({p["error"]})')
        else:
            L.append(f'- `{exp_id}`: `{p["folder"]}/` — ' + ', '.join(f'`{f}`' for f in p['files']))
    return '\n'.join(L) + '\n'


def mark(text: str, ungrounded: list) -> str:
    """The report with each ungrounded number marked in place."""
    out = text
    for n in sorted(ungrounded, key=lambda n: -n['start']):
        out = out[:n['end']] + ' [not from Falcon\'s data]' + out[n['end']:]
    return out


def report(workspace: str, sid: str, prose: str, job_state=None, paper: bool = True) -> dict:
    """Write experiments/<id>/report.md and report.json (and each round's paper outputs into paper/<experiment>/).
    final only when every number passed the grounding check. Returns report.json's document."""
    from . import paper as PP
    from .paths import atomic_write
    from .provenance import write_json
    rec = load(workspace, sid)
    state, why = sync(workspace, rec, job_state)
    if state not in ('analysed', 'follow_up', 'reported'):
        raise StudyError(f'{sid} is {state}: the report comes after the analysis ({why})')
    papers = {}
    for r in rec['rounds']:
        out = os.path.join(workspace, 'paper', r['experiment'])
        if not paper:
            continue
        try:
            doc = PP.export('experiment', study_dir(workspace, r['experiment']), out)
            papers[r['experiment']] = {'folder': os.path.relpath(out, workspace), 'files': sorted(doc['files'])}
        except (PP.PaperError, OSError, ValueError, KeyError) as e:
            papers[r['experiment']] = {'error': str(e)[:200]}
    fu = follow_up_proposal(workspace, rec)
    text = report_markdown(workspace, rec, prose, papers, fu['why'])
    bad = ground(workspace, rec, text)
    final = not bad
    body = mark(text, bad)
    head = ('**Status: final** — every number in this report is one of the study\'s own (the grounding check).\n\n'
            if final else
            f'**Status: draft** — {len(bad)} number(s) are not from Falcon\'s data (marked); a report is final only '
            'when every number is the study\'s own.\n\n')
    lines = body.split('\n', 2)
    body = lines[0] + '\n\n' + head + (lines[2] if len(lines) > 2 else '')
    if bad:
        body += '\n## Not from Falcon\'s data\n\n' + '\n'.join(f'- {n["text"]}' for n in bad) + '\n'
    d = study_dir(workspace, sid)
    atomic_write(os.path.join(d, REPORT), body)
    doc = {'schema': SCHEMA.replace('autopilot', 'report'), 'study': sid, 'final': final, 'rounds': len(rec['rounds']),
           'verdict': (_results(workspace, rec['rounds'][-1]['experiment']) or {}).get('verdict'),
           'ungrounded': [n['text'] for n in bad], 'papers': papers, 'written': utc(),
           'path': os.path.relpath(os.path.join(d, REPORT), workspace)}
    write_json(os.path.join(d, REPORT_JSON), doc)
    rec['report'] = {k: doc[k] for k in ('final', 'rounds', 'ungrounded', 'written', 'path', 'verdict')}
    rec['stopped'] = None
    _note(rec, 'reported', 'final' if final else f'a draft: {len(bad)} number(s) not from Falcon\'s data', 'assistant')
    save(workspace, rec)
    return doc


# ============================================================================================== suggestions
def _public(path) -> bool:
    from ..assistant.privacy import private_reason
    return not path or not private_reason(path)


def _pilot_sd(workspace: str, metric: str) -> tuple:
    """(sd, where) of the paired differences an analysed ablation of the workspace measured for this metric, else
    (ASSUMED_SD, None)."""
    AB = _ab()
    for row in AB.registry(workspace):
        if row.get('kind') != AB.KIND or row.get('metric') != metric or row.get('status') != 'analysed':
            continue
        res = _results(workspace, row['id']) or {}
        sds = [c['sd_diff'] for c in res.get('contrasts') or () if c.get('sd_diff')]
        if sds:
            return max(sds), row['id']
    return ASSUMED_SD, None


def template(workspace: str, project, conditions: dict, about: str, metric: str = 'AR100') -> tuple:
    """(spec, note): an ablation of the detection executor that puts a condition (weather fields) against a clear
    day — two registered models compared in it, or one model's metric in it against the clear day. (None, why) when
    the project has no model."""
    from .oddcoverage import CELL_SUN, CELL_WEATHER
    models = [m['id'] for m in (getattr(project, 'doc', None) or {}).get('models') or () if m.get('id')]
    if not models:
        return None, 'register a detector (Models page) to measure this: the ablation evaluates registered models'
    ref = dict(CELL_WEATHER['clear'], sun_altitude_angle=CELL_SUN['day'])
    cond = {'name': 'conditions', 'kind': 'weather', 'levels': {'clear_day': ref, 'mode': dict(conditions)}}
    sd, pilot = _pilot_sd(workspace, metric)
    design = {'effect': ASSUMED_EFFECT, 'sd': sd, 'alpha': 0.05, 'power': 0.8}
    note = (f'the spread of the paired differences {sd:g} was measured by {pilot}' if pilot else
            f'effect {ASSUMED_EFFECT:g} and spread {ASSUMED_SD:g} are the manual\'s example, not measured: check '
            'them (a pilot measures the spread)')
    if len(models) >= 2:
        a, b = models[0], models[1]
        spec = {'hypothesis': {'text': f'{b} scores higher {metric} than {a} in {about}', 'direction': 'increase'},
                'metric': metric, 'factors': [{'name': 'model', 'kind': 'model', 'levels': [a, b]}, cond],
                'contrasts': [{'factor': 'model', 'a': a, 'b': b, 'where': {'conditions': 'mode'}}],
                'design': design, 'run': {'executor': 'detection', 'seconds': 10.0, 'fps': 10.0}}
    else:
        spec = {'hypothesis': {'text': f'{about} lowers {models[0]}\'s {metric}', 'direction': 'decrease'},
                'metric': metric, 'factors': [cond],
                'contrasts': [{'factor': 'conditions', 'a': 'clear_day', 'b': 'mode'}], 'design': design,
                'run': {'executor': 'detection', 'seconds': 10.0, 'fps': 10.0, 'model': models[0]}}
    return spec, note


def _cluster_conditions(c: dict) -> tuple:
    from .oddcoverage import CELL_SUN, CELL_WEATHER
    w = dict(CELL_WEATHER['clear'], sun_altitude_angle=CELL_SUN['day'])
    used = []
    for f in c.get('features') or ():
        name = f.get('feature')
        if name == 'tod' and f.get('value') in ('night', 'twilight', 'low sun', 'day'):
            w['sun_altitude_angle'] = {'night': CELL_SUN['night'], 'twilight': CELL_SUN['twilight'],
                                       'low sun': CELL_SUN['low_sun'], 'day': CELL_SUN['day']}[f['value']]
            used.append(f['text'])
        elif name in ('fog', 'rain', 'sun') and f.get('median_in') is not None:
            key = {'fog': 'fog_density', 'rain': 'precipitation', 'sun': 'sun_altitude_angle'}[name]
            w[key] = float(f['median_in'])
            if name == 'fog':
                w['fog_distance'] = 0.0
            used.append(f['text'])
    return (w, used) if used else (None, [])


def _cell_conditions(cid: str) -> dict:
    from .oddcoverage import CELL_SUN, CELL_WEATHER, parse_cell
    lab = parse_cell(cid)
    return dict(CELL_WEATHER[lab['weather']], sun_altitude_angle=CELL_SUN[lab['light']])


def suggestions(workspace: str, project=None, limit: int = MAX_SUGGESTIONS) -> list:
    """Questions worth a study, found in the workspace (module doc), private sources left out: [{'id', 'kind',
    'question', 'facts', 'template', 'note'}]."""
    from . import failclusters as FC
    from . import oddcoverage as OC
    out = []
    # failure modes: clusters of the public failures (a private failure shapes no cluster)
    try:
        fails = FC.collect(workspace, keep=lambda f: _public(f.get('run')))
        cl = [c for c in FC.cluster(fails) if c['size'] >= 2] if fails else []
    except (OSError, ValueError, KeyError):
        cl = []
    for c in cl[:3]:
        w, used = _cluster_conditions(c)
        about = ', '.join(used) or c['name']
        spec, note = template(workspace, project, w, about) if w else (
            None, 'its distinguishing features are not weather or light — the scenario, its parameters or the '
                  'outcome — which the ablation executors do not vary')
        out.append({'id': f'cluster:{c["id"]}', 'kind': 'failure_mode', 'about': c['name'],
                    'question': f'Does perception degrade in the conditions of the failure mode "{c["name"]}"?',
                    'facts': {'cluster': c['id'], 'size': c['size'], 'scenarios': c['scenarios'],
                              'features': [f['text'] for f in c['features'][:5]]},
                    'template': spec, 'note': note})
    # operating-domain cells: counted from public runs only
    cells = {}
    try:
        cov = OC.coverage(workspace, project)
        for cid, e in cov['cells'].items():
            pub = [i for i in e['samples'] if _public(cov['samples'][i]['run'])]
            if not pub:
                continue
            n, k = len(pub), sum(1 for i in pub if cov['samples'][i]['fail'])
            j = OC.judge(n, k, cov['rule']['target'], cov['rule']['confidence'])
            cells[cid] = dict(j, runs=n, failures=k)
    except (OSError, ValueError, KeyError):
        cov = None
    ranked = sorted(cells.items(), key=lambda kv: ({'above_target': 0, 'too_few': 1}.get(kv[1]['status'], 9),
                                                   -kv[1]['failures'], -kv[1]['needed'], kv[0]))
    for cid, e in [kv for kv in ranked if kv[1]['status'] in ('above_target', 'too_few')][:3]:
        lab = OC.describe_cell(cid)
        spec, note = template(workspace, project, _cell_conditions(cid), f'{OC.parse_cell(cid)["weather"]} '
                                                                         f'{OC.parse_cell(cid)["light"]}')
        why = 'failures above the target' if e['status'] == 'above_target' else 'too few runs'
        out.append({'id': f'cell:{cid}', 'kind': 'odd_cell', 'about': lab,
                    'question': f'Is perception worse in the operating-domain cell {lab} ({why})?',
                    'facts': {'cell': cid, 'status': e['status'], 'runs': e['runs'], 'failures': e['failures'],
                              'upper95': e['upper'], 'lower95': e['lower'], 'needed': e['needed']},
                    'template': spec,
                    'note': note + '; the cell\'s road, actor and speed are not ablation factors: the template '
                                   'varies its weather and light'})
    # failure-rate estimates: the worst cell of each public study
    for e in OC.estimates(workspace):
        d = os.path.join(workspace, 'studies', e['study'])
        if not _public(d):
            continue
        bc = {c: x for c, x in (e.get('by_cell') or {}).items() if x.get('status') != 'refused' and x.get('p')}
        if not bc:
            continue
        cid, x = max(bc.items(), key=lambda kv: (kv[1]['p'], kv[0]))
        try:
            cond = _cell_conditions(cid)
        except (ValueError, KeyError):
            continue
        spec, note = template(workspace, project, cond, f'{OC.parse_cell(cid)["weather"]} '
                                                        f'{OC.parse_cell(cid)["light"]}')
        out.append({'id': f'failure_rate:{e["study"]}', 'kind': 'failure_rate', 'about': OC.describe_cell(cid),
                    'question': f'Does perception explain the failure rate of {e.get("scenario")} in '
                                f'{OC.describe_cell(cid)}?',
                    'facts': {'study': e['study'], 'scenario': e.get('scenario'), 'cell': cid, 'p': x.get('p'),
                              'runs': x.get('runs'), 'failures': x.get('failures')},
                    'template': spec, 'note': note})
    # regressions a bisect found
    bd = os.path.join(workspace, 'bisect')
    for name in sorted(os.listdir(bd)) if os.path.isdir(bd) else ():
        d = os.path.join(bd, name)
        p = os.path.join(d, 'bisect.json')
        if not os.path.isfile(p) or not _public(d):
            continue
        try:
            with open(p, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        for scn, r in sorted((doc.get('bisect') or {}).items()):
            fb = r.get('first_bad') or {}
            if not fb:
                continue
            out.append({'id': f'bisect:{name}:{scn}', 'kind': 'regression', 'about': scn,
                        'question': f'What did commit {fb.get("short")} ("{fb.get("subject")}") change that made '
                                    f'{scn} fail?',
                        'facts': {'bisect': name, 'scenario': scn, 'first_bad': fb.get('short'),
                                  'subject': fb.get('subject')},
                        'template': None,
                        'note': 'commits are not a factor the ablation executors run: register the model of each '
                                'commit (two model levels), or re-run the suite at both commits in CI mode'})
    return out[:int(limit)]
