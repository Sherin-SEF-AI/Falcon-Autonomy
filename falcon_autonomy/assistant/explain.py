"""Explain this error and Ask about this (Qt-free; docs/assistant.md).

Explain. job_context(job) is what the fast model gets about a failed job: its title, kind, what its exit code means,
the tail of its log (redacted: the home folder shortened, credentials replaced) — nothing from a job whose run folder
is private (the dashcam footage: then the title and exit alone). known_fixes() matches the log against failures with an
obvious next step and gives that step as a tool call (a CTA): the server not answering → system_server; refused for
memory → system_machine; the rig without a microphone array → project_sensor_add of one; CARLA's Python not found →
knowledge_setting client_python; weights not fetched → the manual's section on fetching them (the user accepts a licence
themselves); the GPU taken → system_queue. manual_for() finds the manual's sections about the error (core.manual, offline).
explain() asks the fast model for a short cause and fix from these (and which CTA, by key, if any) and checks its
numbers against what it was given; offline() words the same from the exit description, the last error and the known
fix — the answer when the assistant is off or not set up, or when the request fails.

Ask about this. about(kind, ...) builds the request and its context for AssistantService.ask: the object named by its id
(a run by its id relative to the workspace, a job by number, a study row by study and index, a suite result, an
evaluation, a chart's page and source) — never its images, never a path outside the workspace, and refused for anything
private (privacy.private_reason).
"""
from __future__ import annotations

import json
import os
import re

from . import grounding, privacy

LOG_TAIL = 30
LINE_CAP = 300
EXPLAIN_SYSTEM = (
    'You are Falcon\'s assistant (Falcon Autonomy: self-driving AI development on CARLA). A job failed. From the job, '
    'its log tail and the manual sections given, write at most four short lines: "Cause: …" (the most likely cause, '
    'quoting the log where it says so) and "Fix: …" (what the user should do in Falcon). Then a last line "Action: '
    '<key>" naming one of the offered actions if it is the obvious next step, else "Action: none". Never invent '
    'numbers, paths or settings that the context does not hold; say when the log does not show the cause.')

# (pattern over the exit description and the log tail, key, what the CTA does, tool, args (rig: the project's first))
FIXES = (
    (r'no carla server answered|server stopped answering|connection refused|time-?out.*server|server_lost|no_server',
     'server', 'Look at the CARLA server\'s state and log', 'system_server', {}),
    (r'not enough memory|memory guard|out of memory|oom|cannot allocate memory', 'memory',
     'Check the machine\'s free memory', 'system_machine', {}),
    (r'cuda out of memory|gpu lock|gpu is busy|cuda error', 'gpu', 'See who holds the GPU', 'system_queue', {}),
    (r'no (microphone|audio)[ _-]?array|needs? an? audio[ _]array|no audio_array', 'audio_array',
     'Add a microphone array to the rig', 'project_sensor_add', {'type': 'audio_array'}),
    (r'carla.s python|import carla|no module named .carla.|carla module|client python', 'client_python',
     'Read what the CARLA Python setting does', 'knowledge_setting', {'key': 'client_python'}),
    (r'render budget|mpx per step', 'budget', 'Read what the render budget setting does', 'knowledge_setting',
     {'key': 'render_budget_mpx'}),
    (r'weights? (are )?not (fetched|found|downloaded)|fetch (the )?weights|licen[cs]e', 'weights',
     'Read how model weights are fetched', 'knowledge_manual', {'query': 'fetch model weights licence'}),
)


def job_context(job, workspace: str = '') -> dict:
    """What the model is told about a job (a core.jobs.Job, or a job-history record): title, kind, state, exit, the
    last error and the log tail (redacted); without the log when its run folder is private."""
    from ..core import protocol as P
    if isinstance(job, dict):                 # a job-history record (core.jobhistory) with its log_text added
        log = [ln for ln in str(job.get('log_text') or '').splitlines() if ln.strip()]
        ctx = {'title': job.get('title'), 'kind': job.get('kind'), 'state': job.get('status'),
               'exit': job.get('exit') or '', 'error': job.get('error') or '', 'run_dir': job.get('result') or ''}
    else:
        log = [t if lvl == 'info' else f'{lvl}: {t}' for lvl, t in job.log if t]
        errs = [t for lvl, t in job.log if lvl == 'error' and t] or \
            [t for lvl, t in job.log if lvl == 'stderr' and str(t).strip()]    # what the job itself said last
        ctx = {'job': job.id, 'title': job.spec.title, 'kind': job.spec.kind, 'state': job.state,
               'exit': P.describe_exit(job.exit_code) if job.exit_code is not None else '',
               'error': errs[-1] if errs else '', 'run_dir': job.spec.run_dir or '',
               'elapsed_s': round(job.elapsed, 1)}
    rd = ctx.pop('run_dir')
    why = privacy.private_reason(rd) if rd else None
    if why:
        ctx['private'] = f'{why}: its log is not read'
        ctx['error'] = ''
        return privacy.redact(ctx)
    if rd and workspace:
        try:
            if os.path.commonpath([os.path.realpath(rd), os.path.realpath(workspace)]) == os.path.realpath(workspace):
                ctx['run'] = os.path.relpath(os.path.realpath(rd), os.path.realpath(workspace))
        except ValueError:
            pass
    ctx['error'] = str(ctx.get('error') or '')[:LINE_CAP]
    ctx['log_tail'] = [str(ln)[:LINE_CAP] for ln in log[-LOG_TAIL:]]
    return privacy.redact(ctx)


def known_fixes(ctx: dict, doc: dict | None = None) -> list:
    """The CTAs whose pattern the job's exit, error or log tail matches: [{'key', 'label', 'tool', 'args'}]."""
    hay = ' '.join([str(ctx.get('exit') or ''), str(ctx.get('error') or '')] +
                   [str(x) for x in ctx.get('log_tail') or ()]).lower()
    out = []
    for rx, key, label, tool, args in FIXES:
        if not re.search(rx, hay):
            continue
        a = dict(args)
        if tool == 'project_sensor_add':
            rigs = (doc or {}).get('rigs') or []
            if not rigs:
                continue
            a['rig'] = rigs[0]['id']
        out.append({'key': key, 'label': label, 'tool': tool, 'args': a})
    return out


def manual_for(ctx: dict, limit: int = 2, chars: int = 700) -> list:
    """The manual's sections about the error (offline search over its words): [{'link', 'heading', 'text'}]."""
    from ..core import manual as M
    if not M.docs_dir():
        return []
    words = ' '.join([str(ctx.get('kind') or ''), str(ctx.get('exit') or ''), str(ctx.get('error') or '')])
    words = re.sub(r'[/~][^\s]*', ' ', words)                # paths say nothing the manual has
    out = []
    for _sc, s in M.search(words, limit):
        out.append({'link': f'docs/{s["file"]}' + (f'#{s["anchor"]}' if s['anchor'] else ''),
                    'heading': ' ▸ '.join(s['trail']), 'text': s['text'][:chars]})
    return out


def offline(ctx: dict, fixes: list, manual: list) -> dict:
    """The explanation without a model: what the exit means, the last error, the obvious fix, where the manual says
    more."""
    lines = []
    if ctx.get('private'):
        lines.append(f'This job worked on private data ({ctx["private"]}).')
    lines.append(f'Cause: {ctx.get("exit") or "the job ended without an exit description"}'
                 + (f' The last error: {ctx["error"]}' if ctx.get('error') else
                    ' (no error line was logged — the log has the details).'))
    if fixes:
        lines.append(f'Fix: {fixes[0]["label"]}.')
    if manual:
        lines.append(f'The manual: {manual[0]["heading"]} ({manual[0]["link"]}).')
    return {'text': '\n\n'.join(lines), 'cta': fixes[0] if fixes else None, 'model': None, 'ungrounded': []}


def explain(service, ctx: dict, fixes: list, manual: list, cancel=None) -> dict:
    """The fast model's cause and fix (blocking: call it off the GUI thread): {'text', 'cta' (one of fixes or None),
    'model', 'ungrounded' (numbers in the text the context does not hold)}. Provider errors are raised (the caller
    shows offline())."""
    offered = [{'key': f['key'], 'does': f['label']} for f in fixes]
    raw = service.quick('Explain why this job failed and how to fix it.',
                        {'job': ctx, 'manual': manual, 'actions': offered}, system=EXPLAIN_SYSTEM, max_tokens=450,
                        cancel=cancel)
    text, cta = raw.strip(), (fixes[0] if fixes else None)
    m = re.search(r'^\s*Action:\s*([\w-]+)\s*$', text, re.M | re.I)
    if m:
        key = m.group(1).lower()
        cta = next((f for f in fixes if f['key'] == key), None)
        text = (text[:m.start()] + text[m.end():]).strip()
    src = [{'role': 'tool', 'content': json.dumps({'job': ctx, 'manual': manual}, default=str)}]
    return {'text': text, 'cta': cta, 'model': service.fast_model(), 'ungrounded': [n['text'] for n in
                                                                                   grounding.check(text, src)]}


# --------------------------------------------------------------------------------------------- ask about this
KINDS = ('run', 'job', 'case', 'suite_result', 'evaluation', 'chart', 'log')


class Refused(Exception):
    """The object is private, or outside the workspace: nothing is asked."""


def _ws_rel(path: str, workspace: str) -> str:
    p, w = os.path.realpath(path), os.path.realpath(workspace)
    if not workspace or not (p == w or p.startswith(w + os.sep)):
        raise Refused('that is outside the project\'s workspace: the assistant reads only Falcon\'s own outputs')
    why = privacy.private_reason(p)
    if why:
        raise Refused(f'private: {why} — the assistant never reads it')
    return os.path.relpath(p, w)


def about(kind: str, workspace: str, **ids) -> tuple:
    """(the request text, its context) for "Ask about this". ids by kind: run (run=path); job (job=id, title=, state=,
    run_dir=); case (study=id or path, index=row, run=path or ''); suite_result (suite=path, scenario=, seed=, run=);
    evaluation (evaluation=path); chart (page=, what=, source=path or id); log (job=id, title=, run_dir=).
    Raises Refused for a private or outside path."""
    if kind not in KINDS:
        raise ValueError(f'kind is one of {KINDS}')
    ctx = {'about': kind}
    if kind == 'run':
        rid = _ws_rel(ids['run'], workspace)
        ctx['run'] = rid
        return f'Tell me about run `{rid}`: what it recorded, how it went, and anything unusual in its numbers.', ctx
    if kind in ('job', 'log'):
        if ids.get('run_dir'):
            _ws_rel(ids['run_dir'], workspace)
        ctx['job'] = int(ids['job'])
        if kind == 'log':
            return f'Read the log of job {ctx["job"]} (jobs_get with its log tail) and tell me what happened.', ctx
        return f'What happened in job {ctx["job"]}, and what should I do next with its result?', ctx
    if kind == 'case':
        study = ids['study']
        sid = os.path.basename(os.path.normpath(_ws_rel(study, workspace))) if os.sep in str(study) else str(study)
        ctx.update(study=sid, index=int(ids['index']))
        if ids.get('run'):
            ctx['run'] = _ws_rel(ids['run'], workspace)
        return (f'Explain failing case row {ctx["index"]} of study `{sid}`: what made it critical, its RSS label, '
                'and what to try next.'), ctx
    if kind == 'suite_result':
        sid = os.path.basename(os.path.normpath(_ws_rel(ids['suite'], workspace)))
        ctx.update(suite=sid, scenario=ids.get('scenario'), seed=ids.get('seed'))
        if ids.get('run'):
            ctx['run'] = _ws_rel(ids['run'], workspace)
        return (f'Explain the result of {ids.get("scenario")} seed {ids.get("seed")} in suite `{sid}`: why it passed '
                'or failed, and what it says about the stack.'), ctx
    if kind == 'evaluation':
        eid = os.path.basename(os.path.normpath(_ws_rel(ids['evaluation'], workspace)))
        ctx['evaluation'] = eid
        return f'Summarise evaluation `{eid}`: its scores with their intervals, and where the model is weakest.', ctx
    src = ids.get('source') or ''
    if src and (os.sep in src):
        src = _ws_rel(src, workspace)
    ctx.update(page=ids.get('page'), chart=ids.get('what'), source=src or None)
    return (f'Explain the {ids.get("what") or "chart"} on {ids.get("page") or "this page"}' +
            (f' (from `{src}`)' if src else '') + ': what it shows and what stands out.'), ctx
