"""The agent: the main model works through a request with Falcon's tools until it answers or the user stops it
(Qt-free; the dock drives it through a Host, tests through a headless one).

A turn: the conversation and the tool schemas go to the model (streamed); the tool calls it makes are checked against
the permission mode and run — read-tier calls side by side, act and trash calls one at a time — and their results go
back; until the model answers without a tool call. There is no step limit (the user chose none): Stop always works —
it closes the HTTP connection in flight, answers a pending approval card with "stopped" and ends a wait for a job.

Permission modes (MODES):
    ask        every act call waits for the user's approval card ("approve all of this kind" holds for the rest of the
               conversation); trash always asks
    auto       act calls run without asking; trash still asks
    autopilot  as auto, and the agent keeps going across turns: while the plan has open items it is told to go on
               (until it answers without calling a tool after being told — it is done or blocked), and when jobs it
               queued are still running it waits for the queue to say one finished (a queue subscription, no polling)
               and continues with what happened
In every mode the model can ask to be resumed when given jobs finish (plan_wait_jobs).

Meta-tools the agent registers itself (tool_api tier read; they change nothing but the panel): plan_update — the short
checklist the system prompt asks the model to keep, shown live; plan_wait_jobs — resume when a job finishes.

After each answer the grounding check (assistant.grounding) marks the numbers no tool returned. Read calls run off the
GUI thread (a small pool, side by side); Tool.gui calls, and with a window every act and trash call (project edits
through the undo stack, jobs whose QProcess must be created there), run on the GUI thread through the host. What an
act or trash call changed goes in the activity log with how to undo it: a project edit by the window's undo stack, a
queued job by cancelling it, runs moved to the trash by moving them back.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import threading
import time

from . import grounding, privacy
from . import tool_api as T
from .provider import Cancelled, CancelToken, ProviderError, model_params
from .session import Session, wire

MODES = ('ask', 'auto', 'autopilot')
MODE_TEXT = {'ask': 'Ask', 'auto': 'Auto', 'autopilot': 'Autopilot'}
RESULT_CAP = 16000                 # characters of one tool result fed back (the card shows the same)
ATTACH_CAP = 256 * 1024            # bytes of one attached text file
BINARY_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp', '.exr', '.heic', '.mp4', '.mov',
              '.mkv', '.avi', '.webm', '.ts', '.h264', '.h265', '.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac',
              '.pcd', '.ply', '.las', '.laz', '.bin', '.npy', '.npz', '.pt', '.pth', '.onnx', '.engine', '.mcap',
              '.db3', '.bag', '.zip', '.gz', '.tar', '.7z', '.pdf', '.sqlite', '.parquet', '.h5', '.pkl'}

SYSTEM = """You are Falcon's assistant, inside Falcon Autonomy — a desktop application for developing self-driving AI \
on the CARLA simulator (worlds, rigs, scenarios, recorded drives, datasets, models, evaluations, rare-case search, \
training, stack testing, reports). You act only through the tools you are given; they call the same code as Falcon's \
own buttons. You never see images, video, lidar, audio or the user's dashcam footage, and you never ask for them.

Rules:
- Numbers: cite only measured numbers that a tool returned in this conversation, with the run, job, evaluation or \
scenario id they came from. Never invent, estimate or recall a metric. If a number is not available, say so and say \
which tool or job would measure it. Falcon marks any number in your answer that no tool returned.
- Label heuristics, rules of thumb and your own reasoning as such ("a heuristic: …"); keep them apart from measurements.
- How-to questions about Falcon: search Falcon's documentation with the docs tool first and answer from it.
- Prefer the user's project: its worlds, rigs, scenarios, runs and models, before built-ins or generic advice.
- Plan: for any request needing more than one step, call plan_update first with a short checklist (a few items, \
imperative), and update it as items are done. Skip it for a single question.
- Long jobs: when you queue a job you need the result of, call plan_wait_jobs with its id and end your turn; Falcon \
resumes you when it finishes. Do not poll.
- Refusals from tools are final: a private source, an action that is never allowed, or the user declining. Explain \
and offer an alternative; never try to work around one.
- Be brief and precise. Use Markdown sparingly: short paragraphs, lists, `code` for ids and paths."""

MODE_RULES = {
    'ask': 'Permission mode: Ask. Before you call a tool that changes anything, say in one or two sentences what you '
           'are about to do and why; the user approves each change on a card.',
    'auto': 'Permission mode: Auto. Changes run without a card (moving runs to the trash still asks). Say what you '
            'changed when you answer.',
    'autopilot': 'Permission mode: Autopilot. Work through the goal on your own across many steps and long jobs; keep '
                 'the plan current; moving runs to the trash still asks. Stop when the goal is met or you are blocked, '
                 'and say which.',
}


# --------------------------------------------------------------------------------------------- meta-tools
PLAN_PARAMS = {'type': 'object', 'additionalProperties': False, 'required': ['items'], 'properties': {
    'items': {'type': 'array', 'maxItems': 12, 'items': {
        'type': 'object', 'additionalProperties': False, 'required': ['text'],
        'properties': {'text': {'type': 'string', 'maxLength': 160}, 'done': {'type': 'boolean'}}}}}}
WAIT_PARAMS = {'type': 'object', 'additionalProperties': False, 'required': ['job_ids'], 'properties': {
    'job_ids': {'type': 'array', 'minItems': 1, 'items': {'type': 'integer'}},
    'note': {'type': 'string', 'maxLength': 200}}}


def _plan_update(ctx, items):
    items = [{'text': str(i['text']).strip(), 'done': bool(i.get('done'))} for i in items if str(i['text']).strip()]
    ctx.emit('plan', items)
    return {'ok': True, 'open': sum(not i['done'] for i in items), 'done': sum(i['done'] for i in items)}


def _plan_wait_jobs(ctx, job_ids, note=''):
    q = ctx.queue
    known = [j for j in job_ids if q is None or q.get(int(j)) is not None]
    missing = [j for j in job_ids if j not in known]
    if not known:
        raise T.ToolError(f'no job with id {", ".join(str(j) for j in missing)} in this session')
    ctx.emit('wait_jobs', known)
    out = {'scheduled': known, 'next': 'end your turn now: Falcon resumes this conversation when one of them finishes'}
    if missing:
        out['unknown'] = missing
    return out


META = (
    T.Tool('plan_update', 'ui', 'read', 'Show the user your plan as a short checklist (replaces the previous one). '
           'Call it first for any multi-step request and again as items are done. Changes nothing in Falcon.',
           PLAN_PARAMS, _plan_update),
    T.Tool('plan_wait_jobs', 'jobs', 'read', 'Resume this conversation when one of these queued jobs finishes (you get '
           'its outcome as a message). Call it after queueing a job whose result you need, then end your turn — do not '
           'poll. Returns the ids it will wait for.', WAIT_PARAMS, _plan_wait_jobs),
)


def ensure_meta():
    for t in META:
        if T.REGISTRY.get(t.name) is not t:
            T.REGISTRY.pop(t.name, None)
            T.register(t)


# --------------------------------------------------------------------------------------------- the user's message
PATH_IN_TEXT = re.compile(r'(?:(?<=\s)|^)(~?/[^\s\'"`<>|;,]+)')


def compose(text: str, context: dict | None = None, attachments=()) -> tuple:
    """The user's message as it will be sent: (content, None), or (None, why it is refused). Refused: a path in the
    text, or an attached file, that is private (privacy.private_reason: the dashcam footage), and an attachment that is
    not plain text (images, video, audio, lidar, archives never go to the model) or larger than ATTACH_CAP."""
    text = (text or '').strip()
    for m in PATH_IN_TEXT.finditer(text):
        why = privacy.private_reason(os.path.expanduser(m.group(1).rstrip('.')))
        if why:
            return None, f'Not sent: your message names a private location ({why}). The assistant never reads it.'
    parts = [text] if text else []
    for p in attachments or ():
        p = os.path.expanduser(str(p))
        name = os.path.basename(p)
        why = privacy.private_reason(p)
        if why:
            return None, f'Not sent: {name} is private ({why}).'
        if os.path.splitext(name)[1].lower() in BINARY_EXT:
            return None, f'Not sent: {name} is not a text file — only text is ever sent (no images, video, audio, ' \
                         'lidar or archives).'
        try:
            size = os.path.getsize(p)
            if size > ATTACH_CAP:
                return None, f'Not sent: {name} is {size / 1024:.0f} KB; attachments are text up to ' \
                             f'{ATTACH_CAP // 1024} KB.'
            with open(p, 'rb') as fh:
                raw = fh.read()
        except OSError as e:
            return None, f'Not sent: cannot read {name} ({e.strerror or e}).'
        if b'\x00' in raw[:8192]:
            return None, f'Not sent: {name} is not a text file.'
        try:
            body = raw.decode('utf-8')
        except UnicodeDecodeError:
            return None, f'Not sent: {name} is not UTF-8 text.'
        parts.append(f'Attached file `{name}`:\n```\n{body}\n```')
    if context:
        parts.append('[Falcon context] ' + json.dumps(context, ensure_ascii=False, default=str))
    if not parts:
        return None, 'Nothing to send.'
    return '\n\n'.join(parts), None


# --------------------------------------------------------------------------------------------- approvals, host
class Approval:
    """One approval card: what the tool will do, decided on the GUI thread, waited for on the agent's."""

    def __init__(self, call_id, tool, tier, args, preview='', estimate=None, description=''):
        self.call_id, self.tool, self.tier, self.args = call_id, tool, tier, args
        self.preview, self.estimate, self.description = preview, estimate, description
        self.allow_all = tier == 'act'          # trash always asks: no "approve all" for it
        self.choice = None
        self._ev = threading.Event()

    def decide(self, choice: str):
        if choice not in ('approve', 'approve_all', 'reject', 'stopped'):
            raise ValueError(choice)
        if choice == 'approve_all' and not self.allow_all:
            choice = 'approve'
        if self.choice is None:
            self.choice = choice
        self._ev.set()

    def wait(self, cancel: CancelToken) -> str:
        cancel.on_cancel(lambda: self.decide('stopped'))
        self._ev.wait()
        return self.choice


class Host:
    """What the agent runs against. This one has no window (tests, the command line); ui.assistant_dock.QtHost runs
    gui tools on the GUI thread and reads the window's undo stack."""
    has_window = False

    def __init__(self, ctx: T.ToolContext | None = None):
        self.ctx = ctx or T.ToolContext()

    @property
    def queue(self):
        return self.ctx.queue

    def tool_context(self) -> T.ToolContext:
        c = self.ctx
        return T.ToolContext(project=c.project, settings=c.settings, queue=c.queue, service=c.service, win=c.win,
                             workspace=c.workspace)

    def event(self, kind: str, payload):
        pass

    def run_gui(self, fn):
        return fn()

    def undo_index(self):
        return None

    def context_summary(self) -> dict:
        p = self.ctx.project
        doc = getattr(p, 'doc', None) or {}
        return {'project': doc.get('name')} if doc else {}


def summarize(result: dict) -> str:
    """One line for a tool card: its error, or what it returned at a glance."""
    if not isinstance(result, dict):
        return ''
    if result.get('error'):
        return str(result['error'])[:160]
    for k, v in result.items():
        if isinstance(v, list):
            return f'{len(v)} {k}'
    keys = [k for k in result if not k.startswith('_')]
    return ', '.join(keys[:4]) + (' …' if len(keys) > 4 else '')


# --------------------------------------------------------------------------------------------- the agent
class Agent:
    def __init__(self, service, session: Session, host: Host | None = None, mode: str = 'ask'):
        ensure_meta()
        self.service = service
        self.session = session
        self.host = host or Host()
        self.mode = mode if mode in MODES else 'ask'
        self.cancel = CancelToken()
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._wait_ids: set = set()             # jobs plan_wait_jobs asked to resume after
        self.my_jobs: set = set()               # jobs this conversation's tools queued
        self._finished: list = []               # watched jobs that finished, not yet reported
        self._job_ev = threading.Event()
        self._subscribed = None
        self.pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix='falcon-tool')
        self._subscribe()

    # ------------------------------------------------------------------ control
    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, content: str | None, event_note: bool = False, display: str = '') -> bool:
        """Send the user's message (composed) and run the turn on the agent's thread. False when a turn is running.
        display: what the panel shows for it (the typed text and the names of attached files)."""
        if self.busy:
            return False
        self.cancel = CancelToken()
        self.cancel.on_cancel(self._job_ev.set)
        if content:
            m = {'role': 'user', 'content': content}
            if event_note:
                m['_event'] = True
            if display and display != content:
                m['_display'] = display
            self.session.add(m)
            self.session.save()
        self.thread = threading.Thread(target=self._run, name='falcon-assistant', daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        t = self.thread
        if t is not None:
            t.join(timeout)
        return not self.busy

    def close(self):
        self.stop()
        self.wait(5.0)
        q = self.host.queue
        if q is not None and self._subscribed in getattr(q, 'listeners', []):
            q.listeners.remove(self._subscribed)
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ the queue
    def _subscribe(self):
        q = self.host.queue
        if q is None or not hasattr(q, 'subscribe'):
            return

        def on(ev, job):
            if ev == 'finished' and (job.id in self._wait_ids or job.id in self.my_jobs):
                with self._lock:
                    if job.id not in self._finished:
                        self._finished.append(job.id)
                self._job_ev.set()
        self._subscribed = on
        q.subscribe(on)

    def _job_final(self, jid) -> bool:
        q = self.host.queue
        j = q.get(jid) if q is not None else None
        return j is None or j.state in ('done', 'failed', 'cancelled')

    def _pending_jobs(self) -> set:
        ids = set(self._wait_ids)
        if self.mode == 'autopilot':
            ids |= self.my_jobs
        return {j for j in ids if not self._job_final(j)} | {j for j in ids if j in self._finished}

    def _wait_for_jobs(self, ids: set) -> list:
        """Block until one of ids finishes (or Stop); the jobs that finished, each reported once."""
        self.host.event('waiting', sorted(ids))
        while True:
            self._job_ev.clear()                # cleared before looking: a finish after the look still wakes us
            with self._lock:
                done = [j for j in self._finished if j in ids]
                for j in ids:
                    if j not in done and self._job_final(j):
                        done.append(j)
                if done:
                    self._finished = [j for j in self._finished if j not in done]
            if done:
                for j in done:
                    self._wait_ids.discard(j)
                    self.my_jobs.discard(j)
                self.host.event('waiting', [])
                return done
            if self.cancel.is_set():
                raise Cancelled()
            self._job_ev.wait()

    def job_event_text(self, jid) -> str:
        q = self.host.queue
        j = q.get(jid) if q is not None else None
        if j is None:
            return f'[Falcon] Job {jid} is no longer in this session\'s queue.'
        from ..core import protocol as P
        parts = [f'[Falcon] Job {j.id} "{j.spec.title}" finished: {j.state}']
        if j.started:
            parts.append(f'after {j.elapsed:.0f} s')
        if j.exit_code not in (None, 0):
            parts.append(f'({P.describe_exit(j.exit_code)})')
        text = ' '.join(parts) + '.'
        err = [t for lvl, t in j.log if lvl == 'error' and t]
        if j.state == 'failed' and err:
            text += f' Last error: {err[-1][:300]}'
        if j.spec.run_dir:
            text += f' Result folder: {j.spec.run_dir}'
        if isinstance(j.result, dict) and j.result:
            text += ' Result: ' + json.dumps(j.result, default=str)[:1500]
        return privacy.redact_text(text)

    # ------------------------------------------------------------------ the turn
    def _run(self):
        try:
            close_dangling(self.session)
            self.host.event('turn', {'state': 'started'})
            self._loop()
            self.host.event('turn', {'state': 'done'})
        except Cancelled:
            self.host.event('turn', {'state': 'stopped'})
        except ProviderError as e:
            self.host.event('error', {'kind': e.kind, 'message': e.message})
            self.host.event('turn', {'state': 'failed'})
        except Exception as e:                  # noqa: BLE001 (a bug must not take the window with it)
            self.host.event('error', {'kind': 'internal', 'message': f'The assistant stopped on an internal error '
                                      f'({type(e).__name__}).'})
            self.host.event('turn', {'state': 'failed'})
        finally:
            try:
                self.session.save()
            except OSError as e:
                self.host.event('error', {'kind': 'save', 'message': f'Could not save the conversation: {e}'})

    def system_message(self) -> dict:
        ctx = self.host.context_summary()
        extra = ('\n\nNow: ' + json.dumps(ctx, ensure_ascii=False, default=str)) if ctx else ''
        return {'role': 'system', 'content': SYSTEM + '\n\n' + MODE_RULES[self.mode] + extra}

    def _loop(self):
        nudged_at = None                        # tool calls made when the last Autopilot nudge went out
        calls_made = 0
        while True:
            if self.cancel.is_set():
                raise Cancelled()
            res = self._request()
            msg = res.message()
            if res.reasoning:
                msg['_reasoning_chars'] = len(res.reasoning)
            idx = self.session.add(msg)
            if res.tool_calls:
                self.host.event('assistant', {'index': idx, 'content': res.content, 'tool_calls': len(res.tool_calls),
                                              'final': False})
                self._run_tools(res.tool_calls)
                calls_made += len(res.tool_calls)
                self.session.save()
                continue
            ungrounded = grounding.check(res.content or '', self.session.messages[:idx])
            msg['_ungrounded'] = [n['text'] for n in ungrounded]
            if res.finish_reason == 'length':
                msg['_cut'] = True
            self.session.save()
            self.host.event('assistant', {'index': idx, 'content': res.content, 'final': True,
                                          'ungrounded': msg['_ungrounded'], 'cut': res.finish_reason == 'length'})
            pending = self._pending_jobs()
            if pending:
                for jid in self._wait_for_jobs(pending):
                    self._add_event(self.job_event_text(jid))
                continue
            if self.mode == 'autopilot':
                open_items = [p['text'] for p in self.session.plan if not p.get('done')]
                if open_items and nudged_at != calls_made:
                    nudged_at = calls_made
                    self._add_event(f'[Falcon] Autopilot: continue with the plan. Next open item: "{open_items[0]}". '
                                    'If the goal is met or you are blocked, update the plan and say so.')
                    continue
            return

    def _add_event(self, text: str):
        idx = self.session.add({'role': 'user', 'content': text, '_event': True})
        self.session.save()
        self.host.event('event', {'index': idx, 'text': text})

    def _request(self):
        cfg = self.service.cfg()
        model = cfg['main_model']
        client = self.service.client(purpose='agent')
        client.on_retry = lambda status, wait, attempt: self.host.event(
            'retry', {'status': status, 'wait_s': wait, 'attempt': attempt})
        tools = T.schemas(gui=self.host.has_window)
        squeeze, failed_tool = 0, False
        while True:
            msgs = self.fit([self.system_message()] + wire(self.session.snapshot()), tools, model, squeeze)
            try:
                res = client.chat(msgs, model, tools=tools, stream=True, temperature=cfg.get('temperature'),
                                  extra=model_params(model, cfg.get('reasoning_effort', 'medium')),
                                  on_delta=lambda d: self.host.event('delta', d), cancel=self.cancel,
                                  purpose='agent')
            except ProviderError as e:
                if e.kind == 'context' and squeeze < 2:
                    squeeze += 1                # elide older tool results harder and try again
                    self.host.event('notice', 'The conversation is long: older tool results are left out.')
                    continue
                if e.kind == 'tool_use_failed' and not failed_tool:
                    failed_tool = True          # the model wrote an unparseable call: once more
                    self.host.event('notice', 'The model wrote a tool call the server could not read; asking again.')
                    continue
                raise
            break
        cost = self.service.add_usage(model, res.usage)
        with self.session.lock:
            u = self.session.usage
            u['prompt'] += res.usage.prompt
            u['completion'] += res.usage.completion
            u['requests'] += 1
            u['cost'] = round(u['cost'] + (cost or 0.0), 6)
        self.host.event('usage', {'model': model, 'prompt': res.usage.prompt, 'completion': res.usage.completion,
                                  'cost': cost, 'conversation': dict(self.session.usage)})
        return res

    def context_window(self, model: str) -> int:
        cw = getattr(self.service, 'context_window', None)
        n = cw(model) if callable(cw) else None
        return int(n) if n else 131072

    def fit(self, msgs: list, tools: list, model: str, squeeze: int = 0) -> list:
        """The messages trimmed to fit the model's context: the oldest tool results are elided first (their call
        stays; the model can call again). squeeze > 0 (the server said it is too long) elides all but the last
        four results whatever the estimate says."""
        budget = int(self.context_window(model) * 3.0 * 0.7) - (len(json.dumps(tools)) if tools else 0)
        size = len(json.dumps(msgs, ensure_ascii=False))
        tool_idx = [i for i, m in enumerate(msgs) if m.get('role') == 'tool']
        out = list(msgs)
        for i in (tool_idx[:-4] if squeeze else tool_idx):
            if not squeeze and size <= budget:
                break
            c = out[i].get('content') or ''
            if len(c) > 200:
                stub = json.dumps({'elided': 'this older result was left out to fit the context; call the tool '
                                             'again if you need it'})
                size -= len(c) - len(stub)
                out[i] = dict(out[i], content=stub)
        return out

    # ------------------------------------------------------------------ tools
    def _ctx(self, sink: list) -> T.ToolContext:
        ctx = self.host.tool_context()
        ctx.redact = privacy.redact

        def emit(kind, payload):
            sink.append((kind, payload))
            if kind == 'plan':
                with self.session.lock:
                    self.session.plan = list(payload)
                self.host.event('plan', list(payload))
            elif kind == 'wait_jobs':
                with self._lock:
                    self._wait_ids |= {int(j) for j in payload}
            else:
                self.host.event('tool_event', {'kind': kind, 'payload': payload})
        ctx.emit = emit
        return ctx

    def _parse(self, tc):
        a = (tc.arguments or '').strip()
        if not a:
            return {}, None
        try:
            v = json.loads(a)
        except ValueError as e:
            return None, f'the arguments were not valid JSON ({e.msg} at character {e.pos}); call {tc.name} again ' \
                         'with a JSON object'
        if not isinstance(v, dict):
            return None, f'the arguments must be a JSON object; call {tc.name} again with one'
        return v, None

    def _run_tools(self, calls):
        plan = []                                # (tc, tool, args, parse error)
        for tc in calls:
            args, err = self._parse(tc)
            plan.append((tc, T.REGISTRY.get(tc.name), args, err))
        i = 0
        while i < len(plan):
            if self.cancel.is_set():
                raise Cancelled()
            tc, tool, args, err = plan[i]
            if err is None and tool is not None and tool.tier == 'read' and not tool.gui:
                j = i
                while j < len(plan) and plan[j][3] is None and plan[j][1] is not None and plan[j][1].tier == 'read' \
                        and not plan[j][1].gui:
                    j += 1
                self._run_reads(plan[i:j])
                i = j
                continue
            self._run_one(tc, tool, args, err)
            i += 1

    def _finish(self, tc, result: dict, ms: float, approval: str = '', tier: str = 'read'):
        content = json.dumps(result, ensure_ascii=False, default=str)
        if len(content) > RESULT_CAP:
            content = content[:RESULT_CAP] + f'… [cut: the result was {len(content)} characters]'
        m = {'role': 'tool', 'tool_call_id': tc.id, 'content': content, '_name': tc.name, '_ms': round(ms, 1),
             '_summary': summarize(result), '_tier': tier}
        if approval:
            m['_approval'] = approval
        if isinstance(result, dict) and result.get('error'):
            m['_error'] = True
        idx = self.session.add(m)
        self.host.event('tool_done', {'index': idx, 'id': tc.id, 'name': tc.name, 'ms': m['_ms'],
                                      'summary': m['_summary'], 'error': bool(m.get('_error')), 'approval': approval,
                                      'result': content, 'arguments': tc.arguments})

    def _run_reads(self, items):
        futs = {}
        for tc, tool, args, _ in items:
            self.host.event('tool_start', {'id': tc.id, 'name': tc.name, 'arguments': tc.arguments, 'tier': 'read'})
            sink: list = []
            ctx = self._ctx(sink)
            futs[self.pool.submit(self._timed, ctx, tc.name, args)] = tc
        pending = set(futs)
        while pending:
            done, pending = cf.wait(pending, timeout=0.1, return_when=cf.FIRST_COMPLETED)
            if self.cancel.is_set():
                raise Cancelled()
        for f, tc in futs.items():
            res, ms = f.result()
            self._finish(tc, res, ms)

    @staticmethod
    def _timed(ctx, name, args):
        t0 = time.perf_counter()
        res = T.call(ctx, name, args)
        return res, (time.perf_counter() - t0) * 1000

    def on_gui(self, tool) -> bool:
        """Whether a call runs on the GUI thread: gui tools, and — with a window — every act and trash call: they edit
        the project through the window's undo stack and queue jobs whose QProcess must be created on that thread (one
        at a time anyway, and quick: a long wait is a read tool's, off it)."""
        return tool is not None and (tool.gui or (self.host.has_window and tool.tier in ('act', 'trash')))

    def _call_tool(self, tool, ctx, name, args):
        """Run one call where it must run (on_gui), waiting with Stop honoured."""
        if self.on_gui(tool):
            return self.host.run_gui(lambda: self._timed(ctx, name, args))
        f = self.pool.submit(self._timed, ctx, name, args)
        while True:
            try:
                return f.result(timeout=0.1)
            except cf.TimeoutError:
                if self.cancel.is_set():
                    raise Cancelled() from None

    def _run_one(self, tc, tool, args, err):
        tier = tool.tier if tool is not None else 'read'
        self.host.event('tool_start', {'id': tc.id, 'name': tc.name, 'arguments': tc.arguments, 'tier': tier})
        if err is not None:
            self._finish(tc, {'error': err}, 0.0, tier=tier)
            return
        approval = ''
        if tool is not None and tier in ('act', 'trash'):
            approval = self._permit(tc, tool, args)
            if approval in ('rejected', 'stopped'):
                self._finish(tc, {'error': 'the user declined this action' if approval == 'rejected' else
                                  'stopped by the user before it ran', 'declined': True}, 0.0, approval, tier)
                if approval == 'stopped':
                    raise Cancelled()
                return
        sink: list = []
        ctx = self._ctx(sink)
        before = self._snapshot() if tier in ('act', 'trash') else None
        res, ms = self._call_tool(tool, ctx, tc.name, args)
        self._finish(tc, res, ms, approval, tier)
        if before is not None and not (isinstance(res, dict) and res.get('error')):
            self._record_changes(tc.name, args, res, before, sink)

    def _permit(self, tc, tool, args) -> str:
        if tool.tier == 'act':
            if self.mode in ('auto', 'autopilot'):
                return 'auto'
            if tool.name in self.session.allowed:
                return 'allowed'
        ctx = self._ctx([])
        preview = estimate = None
        for attr in ('preview', 'estimate'):
            fn = getattr(tool, attr)
            if fn is None:
                continue
            try:
                v = self.host.run_gui(lambda: fn(ctx, **args)) if self.on_gui(tool) else fn(ctx, **args)
                v = privacy.redact(v)
            except Exception as e:              # noqa: BLE001 (a card without a preview still asks)
                v = f'(no {attr}: {type(e).__name__})' if attr == 'preview' else None
            if attr == 'preview':
                preview = v
            else:
                estimate = v
        req = Approval(tc.id, tool.name, tool.tier, args, preview or '', estimate, tool.description)
        self.host.event('approval', req)
        choice = req.wait(self.cancel)
        self.host.event('approval_done', {'id': tc.id, 'choice': choice})
        if choice == 'approve_all':
            with self.session.lock:
                if tool.name not in self.session.allowed:
                    self.session.allowed.append(tool.name)
            return 'approved_all'
        return {'approve': 'approved', 'reject': 'rejected'}.get(choice, 'stopped')

    # ------------------------------------------------------------------ what changed, and its undo
    def _snapshot(self) -> dict:
        q = self.host.queue
        return {'undo': self.host.undo_index(), 'jobs': {j.id for j in getattr(q, 'jobs', [])} if q is not None
                else set()}

    def _record_changes(self, name, args, res, before, sink):
        after = self._snapshot()
        entries = []
        if before['undo'] is not None and after['undo'] is not None and after['undo'] > before['undo']:
            entries.append({'kind': 'edit', 'undo_from': before['undo'], 'undo_to': after['undo'],
                            'text': f'{name}: {after["undo"] - before["undo"]} project edit(s)'})
        q = self.host.queue
        for jid in sorted(after['jobs'] - before['jobs']):
            j = q.get(jid) if q is not None else None
            entries.append({'kind': 'job', 'job_id': jid, 'text': f'{name}: queued job {jid}'
                            + (f' ({j.spec.title})' if j is not None else '')})
            with self._lock:
                self.my_jobs.add(jid)
            self.host.event('job', jid)
        moved = [p for k, p in sink if k == 'trashed']
        if isinstance(res, dict) and isinstance(res.get('trashed'), list) and not moved:
            moved = res['trashed']
        for m in moved:
            if isinstance(m, dict) and m.get('from') and m.get('to'):
                entries.append({'kind': 'trash', 'from': os.path.expanduser(m['from']),
                                'to': os.path.expanduser(m['to']),
                                'text': f'{name}: moved {os.path.basename(m["from"])} to the workspace trash'})
        if not entries:
            entries.append({'kind': 'other', 'text': f'{name}: {summarize(res) or "done"}'})
        for e in entries:
            e.update(tool=name, args=args)
            self.host.event('activity', self.session.add_activity(e))

    def run_proposed(self, tool_name: str, args: dict, approval: str = 'approved') -> dict:
        """Run a suggested action the user approved on a card outside a turn (AssistantService.propose): the same
        checks, the same activity log, and a note in the conversation so the model knows. Blocking: call it off the
        GUI thread (gui tools are run on it by the host)."""
        tool = T.REGISTRY.get(tool_name)
        sink: list = []
        ctx = self._ctx(sink)
        before = self._snapshot() if tool is not None and tool.tier in ('act', 'trash') else None
        res, ms = (self.host.run_gui(lambda: self._timed(ctx, tool_name, args)) if self.on_gui(tool)
                   else self._timed(ctx, tool_name, args))
        if before is not None and not res.get('error'):
            self._record_changes(tool_name, args, res, before, sink)
        self._add_event(f'[Falcon] The user ran {tool_name} from a suggestion ({approval}): '
                        + json.dumps(res, ensure_ascii=False, default=str)[:2000])
        return res


def close_dangling(session: Session) -> int:
    """Give every tool call that has no result one (a turn stopped between the call and its result): the server
    refuses a conversation with an unanswered call. Returns how many were closed."""
    n = 0
    with session.lock:
        msgs = session.messages
        i = 0
        while i < len(msgs):
            m = msgs[i]
            ids = [tc.get('id') for tc in m.get('tool_calls') or []] if m.get('role') == 'assistant' else []
            if ids:
                j = i + 1
                seen = set()
                while j < len(msgs) and msgs[j].get('role') == 'tool':
                    seen.add(msgs[j].get('tool_call_id'))
                    j += 1
                names = {tc.get('id'): (tc.get('function') or {}).get('name', '') for tc in m['tool_calls']}
                missing = [x for x in ids if x not in seen]
                for k, x in enumerate(missing):
                    msgs.insert(j + k, {'role': 'tool', 'tool_call_id': x, '_name': names.get(x, ''),
                                        'content': json.dumps({'error': 'not run: the user stopped the turn'}),
                                        '_summary': 'not run (stopped)', '_error': True, '_ms': 0.0})
                n += len(missing)
                i = j + len(missing)
                continue
            i += 1
    return n


def undo_activity(entry: dict, win=None, queue=None, workspace: str = '') -> str:
    """Undo one activity entry; returns what happened (a sentence). Project edits through the window's undo stack (only
    while they are the newest steps on it), jobs by cancelling them, runs moved to the trash by moving them back."""
    kind = entry.get('kind')
    if entry.get('undone'):
        return 'Already undone.'
    if kind == 'edit':
        if win is None:
            return 'Project edits are undone in Falcon\'s window (Edit ▸ Undo).'
        st = win.undo
        if st.index() == entry['undo_to']:
            st.setIndex(entry['undo_from'])
            return f'Undid {entry["undo_to"] - entry["undo_from"]} project edit(s).'
        if st.index() < entry['undo_to']:
            return 'Those edits are already undone.'
        return 'Later edits sit on top of these: undo them first (Edit ▸ Undo).'
    if kind == 'job':
        j = queue.get(entry['job_id']) if queue is not None else None
        if j is None:
            return 'That job is no longer in this session\'s queue.'
        if j.state in ('done', 'failed', 'cancelled'):
            return f'Job {j.id} already {j.state}; its result stays (Jobs ▸ History).'
        queue.cancel(j)
        return f'Cancelled job {j.id}.'
    if kind == 'trash':
        src, dst = entry.get('to', ''), entry.get('from', '')
        trash = os.path.realpath(os.path.join(workspace, 'trash')) if workspace else ''
        rs = os.path.realpath(src)
        if not trash or not (rs == trash or rs.startswith(trash + os.sep)):
            return 'Not restored: that folder is not in this workspace\'s trash.'
        if not os.path.exists(src):
            return 'Not restored: it is no longer in the trash.'
        if os.path.exists(dst):
            return f'Not restored: {os.path.basename(dst)} exists again at its old place.'
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        return f'Restored {os.path.basename(dst)}.'
    return 'This change cannot be undone from here.'
