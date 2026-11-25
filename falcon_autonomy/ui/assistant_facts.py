"""What the Assistant panel marks in a reply, as pure functions (no Qt): the run references and the times inside a run
that become links, the numbers a tool of that turn returned (verified, "from <tool>"), and whether the model behind
the panel is live at all.

annotate(text, …) cuts one piece of a reply's rich text (HTML-escaped, outside tags) into spans:
  run      a known run id (the workspace's catalogue) or any `runs/<id>` path → falcon://run/<id>
  time     "t = 12.4 s", "at 12.4 s" or "12.4–14.0 s" after a run was named in the reply → falcon://run/<id>?t=12.4
           (the time of the last run named before it; Data opens the run at that time)
  scenario a known scenario id → falcon://scenario/<id>;  job "job 12" of this session → falcon://job/12
  verified a number a tool of this turn returned (assistant.grounding's matching: rounding to the precision shown,
           fractions as percentages, k/M/B) → falcon://fact/<tool>: underlined, "from <tool>" on hover
  ungrounded a number no tool of the conversation returned (the agent's grounding check) → marked as before
Everything else is the model's commentary and stays plain. HTML entities are never cut.

verified_numbers(text, tool_results) -> {number text: tool name}: the numbers of the reply (assistant.grounding
.numbers: not inside code, not part of a name, not a date, version or list marker, not a lone 0, 1 or 2) that a tool
result of the turn holds, each credited to the first tool (in call order) that holds it.

liveness(...) -> None for a live model, else (kind, banner text): 'scripted' (a test double or a scripted demo: the
service's `scripted` flag or FALCON_ASSISTANT_SCRIPTED=1), 'no_key' (the provider needs a key and none is stored),
'refused' (the server refused the key on the last request: 401 or 403).
"""
from __future__ import annotations

import json
import os
import re

RUN_PATH = re.compile(r'(?<![\w])(?:[\w.~\-]*/)*runs/([A-Za-z0-9][\w.\-]*[A-Za-z0-9_])')
WORD = re.compile(r'[A-Za-z0-9][\w.\-]{2,}')
JOB = re.compile(r'\b([Jj]ob)\s+#?(\d+)\b')
TIME = re.compile(r'(?:\b[tT]\s*=\s*|\bat\s+)(\d+(?:\.\d+)?)(?:\s?[–-]\s?\d+(?:\.\d+)?)?\s?s\b'
                  r'|\b(\d+(?:\.\d+)?)\s?[–-]\s?\d+(?:\.\d+)?\s?s\b')
ENTITY = re.compile(r'&(?:#\d+|#x[0-9a-fA-F]+|\w+);')
SCRIPTED_ENV = 'FALCON_ASSISTANT_SCRIPTED'

PRIORITY = {'entity': 0, 'run': 1, 'scenario': 2, 'job': 3, 'time': 4, 'verified': 5, 'ungrounded': 6}


def _values(result) -> list:
    from ..assistant import grounding
    vals: list = []
    try:
        grounding._walk(json.loads(result) if isinstance(result, str) else result, vals)
    except ValueError:
        grounding._walk(result, vals)
    return vals


def verified_numbers(text: str, tool_results) -> dict:
    """{number text: tool} for the reply's numbers that a tool result of the turn holds (module doc).
    tool_results: [(tool name, result text or object)] in call order."""
    from ..assistant import grounding
    pools = [(name, _values(res)) for name, res in tool_results or ()]
    out: dict = {}
    for n in grounding.numbers(text or ''):
        if n['text'] in out:
            continue
        for name, vals in pools:
            if vals and grounding.matches(n, vals):
                out[n['text']] = name
                break
    return out


def _num_key(t: str) -> str:
    return t.strip()


def annotate(text: str, runs=(), scenarios=(), jobs=(), verified=None, ungrounded=(), last_run: str = '') -> tuple:
    """(spans, last_run): spans [(start, end, kind, value)] of one piece of rich text, without overlaps, in order;
    last_run is the last run named up to the end of the piece (pass it on to the next piece)."""
    verified = verified or {}
    cands = []
    for m in ENTITY.finditer(text):
        cands.append((m.start(), m.end(), 'entity', ''))
    for m in RUN_PATH.finditer(text):
        cands.append((m.start(), m.end(), 'run', m.group(1)))
    runs = set(runs or ())
    scenarios = set(scenarios or ())
    if runs or scenarios:
        for m in WORD.finditer(text):
            w = m.group(0)
            if w in runs:
                cands.append((m.start(), m.end(), 'run', w))
            elif w in scenarios:
                cands.append((m.start(), m.end(), 'scenario', w))
    jobs = set(jobs or ())
    for m in JOB.finditer(text):
        if int(m.group(2)) in jobs:
            cands.append((m.start(), m.end(), 'job', m.group(2)))
    for m in TIME.finditer(text):
        cands.append((m.start(), m.end(), 'time', m.group(1) or m.group(2)))
    nums = {_num_key(k): ('verified', v) for k, v in verified.items()}
    for u in ungrounded or ():
        nums.setdefault(_num_key(re.sub(r'\s?%$|[kKMB]$', '', u)), ('ungrounded', ''))
        nums.setdefault(_num_key(u), ('ungrounded', ''))
    if nums:
        alts = sorted(nums, key=len, reverse=True)
        rx = re.compile(r'(?<![\w.#/])(' + '|'.join(re.escape(n) for n in alts) + r')(?![\w]|\.\d)')
        for m in rx.finditer(text):
            kind, val = nums[m.group(1)]
            cands.append((m.start(), m.end(), kind, val))
    cands.sort(key=lambda c: (c[0], PRIORITY[c[2]], -(c[1] - c[0])))
    out, end = [], -1
    for c in cands:
        if c[0] < end:
            continue
        out.append(c)
        end = c[1]
    spans = []
    for c in out:
        if c[2] == 'entity':
            continue
        if c[2] == 'run':
            last_run = c[3]
        if c[2] == 'time':
            if not last_run:
                continue
            c = (c[0], c[1], 'time', (last_run, c[3]))
        spans.append(c)
    return spans, last_run


def liveness(enabled: bool, needs_key: bool, has_key: bool, scripted: bool = False,
             last_error: str = '', provider_title: str = 'the model server') -> tuple | None:
    """None when the replies come from a live model, else (kind, banner text) — module doc."""
    if scripted or os.environ.get(SCRIPTED_ENV, '').strip().lower() in ('1', 'true', 'yes'):
        return ('scripted', 'Not a live model: replies are scripted (a test double or a recorded demo).')
    if not enabled:
        return None
    if last_error in ('auth', 'forbidden'):
        return ('refused', f'No model: {provider_title} refused the API key — nothing here is a live answer until '
                           'it is fixed in Settings ▸ Assistant.')
    if needs_key and not has_key:
        return ('no_key', f'No model: the API key for {provider_title} is missing (Settings ▸ Assistant).')
    return None


def turn_tools(messages: list, upto: int) -> list:
    """[(tool name, result)] of the turn that message `upto` belongs to: the tool results since the last message of
    the user (not an event) before it, in call order."""
    names = {}
    out = []
    start = 0
    for i in range(min(upto, len(messages)) - 1, -1, -1):
        m = messages[i]
        if m.get('role') == 'user' and not m.get('_event'):
            start = i + 1
            break
    for m in messages[start:upto]:
        if m.get('role') == 'assistant':
            for tc in m.get('tool_calls') or []:
                names[tc.get('id')] = (tc.get('function') or {}).get('name', '')
        elif m.get('role') == 'tool' and not m.get('_error'):
            out.append((names.get(m.get('tool_call_id'), m.get('name', '') or 'a tool'), m.get('content') or ''))
    return out
