"""The research autopilot's tools (core.autoresearch; docs/research-autopilot.md), in the research group.

research_autopilot (act) drives a study through the deterministic orchestrator in core.autoresearch: start (the
model's ablation spec for the user's question — its approval card shows the plan, the power analysis's replicates
and the cost, and offers to approve the study's follow-ups ahead), follow_up (the orchestrator's follow-up of an
inconclusive round, or the model's own within the approved plan), resume (after Stop, a failure or a restart), report
(the model's discussion inside a report whose numbers are checked) and stop. research_studies (read) lists the
studies with their state and next step, reads one study's rounds, and lists the questions the app suggests from its
own state (failure modes, operating-domain cells, failure rates, bisect regressions).

Approvals. Every action that queues compute (core.autoresearch.COMPUTE) asks the user on a card in every permission
mode, Autopilot included (Tool.approval: ask), with no "approve all of this kind" — unless the user approved the
study's follow-ups on its first card, which then hold within the approved plan until Stop (ask False: no card). The
orchestrator itself refuses those actions without the approval the agent hands it (ToolContext.approval), so the gate
does not rest on the permission modes. report and stop ask as the permission mode says.

Jobs go through the app's queue as `falcon experiment ablation run <id>` (the page's and research_ablation's code
path); the agent's queue subscription resumes Autopilot when one ends (plan_wait_jobs in the other modes). A job is
known to a study by its id and its submission time, so a job of an earlier session (before a restart) is never taken
for one of this session's.

Privacy: a suggestion never comes from the dashcam footage (core.autoresearch.suggestions leaves private failures,
runs and studies out before computing anything); start refuses a source that is not one of those suggestions and a
spec whose world is private; a study's folder is read through the privacy guard.
"""
from __future__ import annotations

from ..tool_api import ToolError, tool
from . import _common as C
from .jobs import _fake, _saved

ACTIONS = ['start', 'follow_up', 'resume', 'report', 'stop']


def _R():
    from ...core import autoresearch as R
    return R


def _proj(ctx):
    try:
        return C.project(ctx)
    except ToolError:
        return None


def _settings(ctx):
    return ctx.settings if ctx.win is None else ctx.win.settings


def _queue(ctx):
    return ctx.queue if ctx.win is None else ctx.win.queue


def job_state(ctx):
    """job record -> its state in this session's queue, or None (not there: an earlier session's job)."""
    q = _queue(ctx)

    def f(j):
        if not j or q is None or j.get('submitted') is None:
            return None
        job = q.get(int(j.get('id') or 0))
        if job is None or abs(float(job.submitted) - float(j['submitted'])) > 1e-3:
            return None
        return job.state
    return f


def _cancel(ctx):
    q = C.need_queue(ctx)

    def f(j):
        job = q.get(int(j['id']))
        if job is not None:
            q.cancel(job)
    return f


def _submit(ctx, sid):
    """exp_id -> queue `falcon experiment ablation run <id>` -> {'id', 'submitted'}."""
    def f(exp_id):
        from ...core import ablation as AB
        from ...core.jobs import FINAL, JobSpec
        ws = C.workspace(ctx)
        _e, s, _p = AB.load(ws, exp_id)
        q = C.need_queue(ctx)
        if any(j.spec.title.startswith(f'Ablation {exp_id}:') and j.state not in FINAL for j in q.jobs):
            raise ToolError(f'a job for {exp_id} is running already')
        if s['run']['executor'] == 'detection':
            argv = ['experiment', 'ablation', 'run', exp_id, '--project', _saved(ctx)]
            if _fake(ctx):
                argv += ['--fake-carla', _fake(ctx)]
        else:                                    # the simulated executor needs the workspace only
            argv = ['experiment', 'ablation', 'run', exp_id, '--workspace', ws]
        title = f'Ablation {exp_id}: run (research study {sid})'
        spec = JobSpec('experiment', title, C.cli_argv(argv), memory_gb=1.5, resumable=True,
                       meta={'study': sid, 'experiment': exp_id})
        out = C.cli_job(ctx, 'experiment', title, argv, spec=spec)
        job = q.get(out['job'])
        return {'id': job.id, 'submitted': job.submitted}
    return f


def _source(ctx, source: str) -> dict:
    """A suggestion id (research_studies suggest) as the study's source; private ones are never among them."""
    if not source:
        return {'kind': 'user'}
    R = _R()
    for s in R.suggestions(C.workspace(ctx), _proj(ctx), limit=50):
        if s['id'] == source:
            return {'kind': s['kind'], 'ref': s['id'], 'about': s.get('about'), 'facts': s.get('facts')}
    raise ToolError(f'no suggestion {source!r} (research_studies with suggest lists them)')


def _approval(ctx) -> dict:
    return dict(getattr(ctx, 'approval', None) or {})


def _err(fn, *a, **kw):
    R = _R()
    from ...core.ablation import AblationError
    try:
        return fn(*a, **kw)
    except (R.StudyError, AblationError) as e:
        raise ToolError(str(e)) from None


def _brief(ctx, rec: dict, extra: dict | None = None) -> dict:
    R = _R()
    state, why = R.derive(C.workspace(ctx), rec, job_state(ctx))
    r = rec['rounds'][-1] if rec.get('rounds') else {}
    out = {'study': rec['id'], 'state': state, 'why': why, 'round': r.get('round'), 'experiment': r.get('experiment'),
           'approved': (r.get('approval') or {}).get('how'),
           'follow_ups_approved': bool((rec.get('follow_ups') or {}).get('approved'))}
    if r:
        out['plan'] = {k: r['plan'][k] for k in ('cells', 'replicates', 'runs', 'drives', 'dz', 'alpha_per_contrast',
                                                 'capped')}
    if state == 'running' and r.get('jobs'):
        out['job'] = r['jobs'][-1]['id']
        out['next'] = (f'plan_wait_jobs with [{out["job"]}]; when it has ended, research_studies with study '
                       f'{rec["id"]} (never describe results before)')
    else:
        out['next'] = R.next_step(state, rec)
    out.update(extra or {})
    return C.rounded(out, 4)


# ================================================================================================ approval and preview
def _proposal(ctx, action, study, spec):
    """The plan an approval card shows: start's draft or follow_up's proposal (or the model's own spec)."""
    R = _R()
    ws = C.workspace(ctx)
    if action == 'start':
        return R.draft(ws, study, 'preview', spec, _proj(ctx), _settings(ctx))
    rec = R.load(ws, study)
    if spec is None:
        return R.follow_up_proposal(ws, rec, _proj(ctx), _settings(ctx))
    from ...core import ablation as AB
    k = len(rec['rounds']) + 1
    ns = AB.normalise(dict(spec, id=f'{study}-f{k - 1}'), _proj(ctx))
    pl = AB.plan(ns)
    c = R.cost(ns, pl, _proj(ctx), _settings(ctx))
    return {'spec': ns, 'cost': c, 'text': R.card_text(rec['question'], ns, pl, c, 'follow_up',
                                                        'the assistant\'s follow-up within the approved plan')}


def _gate(ctx, action='start', study='', spec=None, **_a):
    """Tool.approval (module doc): a card in every mode for what queues compute, none for what was approved ahead."""
    R = _R()
    if action not in R.COMPUTE:
        return None                                      # report, stop: the permission mode decides
    ws = C.workspace(ctx)
    ask = {'ask': True, 'allow_all': False,
           'why': 'it queues compute: every step of a research study that runs anything asks you, in every mode'}
    try:
        if action == 'start':
            _proposal(ctx, action, study, spec)          # a spec that cannot run fails before any card
            return dict(ask, options={'follow_ups': R.FOLLOW_UPS})
        rec = R.load(ws, study)
        if action == 'resume':
            state, _why = R.derive(ws, rec, job_state(ctx))
            if state == 'approved':
                return {'ask': False, 'why': 'approved on its card; its job was not queued yet'}
            if not R.needs_compute(ws, rec):
                return None                              # only the analysis is written: no compute
            ok, why = R.preapproved(rec, 'resume')
        else:
            prop = _proposal(ctx, action, study, spec)
            ok, why = R.preapproved(rec, 'follow_up', prop['spec']) if prop.get('spec') else (False, '')
    except Exception:                                    # noqa: BLE001 (the tool says what is wrong; nothing runs)
        return None
    return {'ask': False, 'why': why} if ok else ask


def _preview(ctx, action='start', study='', question='', spec=None, source='', text='', **_a):
    R = _R()
    ws = C.workspace(ctx)
    try:
        if action == 'start':
            p = R.draft(ws, study, question or '(the question)', spec, _proj(ctx), _settings(ctx))
            return p['text']
        if action == 'follow_up':
            p = _proposal(ctx, action, study, spec)
            return p.get('text') or f'no follow-up: {p.get("why")}'
        rec = R.load(ws, study)
        if action == 'resume':
            r = rec['rounds'][-1]
            if not R.needs_compute(ws, rec):
                return f'resume {study}: every run of {r["experiment"]} finished; write its analysis (no compute)'
            from ...core import ablation as AB
            st = AB.status(R.study_dir(ws, r['experiment']))
            ok = (st['runs'] or {}).get('ok', 0)
            return (f'resume {study}: queue `falcon experiment ablation run {r["experiment"]}` — {st["planned"] - ok} '
                    f'of {st["planned"]} runs left (finished runs are not repeated)')
        if action == 'report':
            return (f'write experiments/{study}/report.md and report.json, and the paper outputs of its '
                    f'{len(rec["rounds"])} round(s) into paper/<experiment>/; final only when every number in it is '
                    'the study\'s own')
        js = job_state(ctx)
        r = rec['rounds'][-1] if rec.get('rounds') else None
        j = r['jobs'][-1] if r and r['jobs'] else None
        return f'stop {study}' + (f': cancel job {j["id"]}' if j and js(j) in R.ACTIVE else '') + \
            ' (resume continues it)'
    except Exception as e:                               # noqa: BLE001 (the card says why; the call will fail)
        return f'cannot: {e}'


def _estimate(ctx, action='start', study='', spec=None, **_a):
    try:
        if action in ('start', 'follow_up'):
            c = _proposal(ctx, action, study, spec).get('cost')
            if c:
                return {'seconds': c.get('seconds'), 'what': c.get('what')}
    except Exception:                                    # noqa: BLE001
        pass
    return {'seconds': None, 'what': {'resume': 'the runs left of the approved plan', 'report': 'a few seconds',
                                      'stop': 'immediate'}.get(action, '')}


# ================================================================================================ the tools
@tool('research_autopilot', 'research', 'act',
      'The research autopilot (docs/research-autopilot.md): a question or hypothesis taken through the research loop '
      'by a deterministic orchestrator (core.autoresearch) — a study stored with its experiment '
      '(experiments/<study>/autopilot.json) and resumable after a restart. start: the user\'s question and your '
      'ablation spec (hypothesis with direction, metric, factors with levels, contrasts, design with the effect and '
      'the sd of paired differences, run with the executor; docs/experiments.md), optionally the suggestion it came '
      'from (source, from research_studies suggest) — the user\'s approval card shows the plan, the replicates the '
      'power analysis asks for and the cost, then its job is queued. Wait for the job (plan_wait_jobs), then read the '
      'study (research_studies). follow_up: the orchestrator\'s follow-up of an inconclusive, underpowered round (more '
      'replicates at the observed spread, or a narrower factor; alpha halves with each look), or your own spec within '
      'the approved plan. resume: continue after Stop, a failure or a restart (finished runs are not repeated). '
      'report: your discussion (text) inside the report Falcon writes — hypothesis, design, pins, every contrast '
      'with its interval and effect size, verdict, limits, paper outputs; every number is checked against the '
      'study\'s own and the report is final only when all pass. stop: cancel its job. start, follow_up and resume '
      'queue compute: each asks the user on a card in every mode (unless the user approved the study\'s follow-ups '
      'ahead); never ask in text.',
      C.obj({'action': C.s('what to do', enum=ACTIONS),
             'study': C.s('the study id (start: a new folder name in experiments/)'),
             'question': C.s('start: the question or hypothesis in the user\'s words'),
             'spec': {'type': 'object', 'description': 'start: the ablation spec; follow_up: your own spec within '
                                                       'the approved plan (default the orchestrator\'s)'},
             'source': C.s('start: the suggestion id it came from (research_studies suggest)'),
             'text': C.s('report: your discussion, citing only numbers of this study (Markdown)')},
            ['action', 'study']),
      estimate=_estimate, preview=_preview, approval=_gate)
def research_autopilot(ctx, action, study, question='', spec=None, source='', text=''):
    R = _R()
    ws = C.workspace(ctx)
    if action == 'start':
        src = _source(ctx, source)
        prop = _err(R.draft, ws, study, question, spec, _proj(ctx), _settings(ctx), src)
        rec = _err(R.start, ws, prop, _approval(ctx), _submit(ctx, prop['study']), _proj(ctx))
        return _brief(ctx, rec, {'cost': prop['cost'].get('what')})
    C.resolve_in(ctx, 'experiments', study, 'research study', (R.FILE,))      # inside the workspace, not private
    if action == 'follow_up':
        rec = _err(R.follow_up, ws, study, _approval(ctx), _submit(ctx, study), _proj(ctx), _settings(ctx), spec,
                   job_state(ctx))
        r = rec['rounds'][-1]
        return _brief(ctx, rec, {'kind': r['kind'], 'why_follow_up': r['why']})
    if action == 'resume':
        rec = _err(R.resume, ws, study, _approval(ctx), _submit(ctx, study), job_state(ctx))
        return _brief(ctx, rec)
    if action == 'report':
        doc = _err(R.report, ws, study, text, job_state(ctx))
        out = {'study': study, 'final': doc['final'], 'verdict': doc['verdict'], 'report': doc['path'],
               'ungrounded': doc['ungrounded'], 'rounds': doc['rounds'],
               'paper': {k: v.get('folder') or v.get('error') for k, v in doc['papers'].items()}}
        out['next'] = ('done: tell the user the verdict and where the report is' if doc['final'] else
                       'a draft: the numbers listed are not the study\'s own — write the discussion again citing '
                       'only the numbers research_studies returned, then report again')
        return out
    rec = _err(R.stop, ws, study, _cancel(ctx), job_state(ctx), 'user', 'stopped at the user\'s request (via the '
                                                                          'assistant)')
    return _brief(ctx, rec)


@tool('research_studies', 'research', 'read',
      'Research studies (core.autoresearch): every study of the workspace with its state (proposed, approved, '
      'running, interrupted, failed, analysed, follow_up, reported, stopped), rounds, verdict and question; with '
      'study, its rounds (experiment, plan, how it was approved, jobs, runs, verdict, each contrast\'s difference with '
      'its interval, dz, adjusted p, power at the planned effect), the follow-up the orchestrator proposes, the '
      'report, the history and the next step. With suggest: questions worth a study found in the app\'s own state — '
      'failure modes, operating-domain cells above the failure target or with too few runs, failure-rate '
      'estimates\' worst cells, regressions a bisect found — each with its facts and a template spec to adjust (start '
      'with source its id). Nothing from the dashcam footage is ever among them.',
      C.obj({'study': C.s('a study id: its rounds and next step'),
             'suggest': C.b('questions the app suggests from its own state')}))
def research_studies(ctx, study='', suggest=False):
    R = _R()
    ws = C.workspace(ctx)
    if suggest:
        rows = R.suggestions(ws, _proj(ctx))
        return C.rounded({'suggestions': [{k: s.get(k) for k in ('id', 'kind', 'question', 'facts', 'template',
                                                                  'note')} for s in rows],
                          'next': 'research_autopilot action start with the user\'s question, the template adjusted '
                                  'to it and source the suggestion\'s id' if rows else
                                  'nothing found: the user states a question'}, 4)
    if study:
        C.resolve_in(ctx, 'experiments', study, 'research study', (R.FILE,))
        return C.rounded(_err(R.status, ws, study, job_state(ctx), _proj(ctx), _settings(ctx)), 4)
    rows = R.summary_rows(ws, job_state(ctx))
    return C.capped([C.rounded(r, 4) for r in rows], 20, 'read one with study')
