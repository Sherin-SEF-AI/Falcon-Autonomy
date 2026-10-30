"""The grounding check: every number in an answer must come from Falcon's data in that conversation (standard library).

check(answer, messages) -> the numbers of the answer that no source holds, each {'text', 'start', 'end', 'value'}.
Sources: every tool result of the conversation (its numbers, the numbers written inside its strings, and the length of
every list in it — "5 runs" counts a list of five), the arguments of the tool calls Falcon ran (what it did: "a 30 s
drive"), the user's own messages and Falcon's job events. Not sources: the model's earlier answers and the system
prompt.

A number matches a source when it is the source rounded to the precision the answer shows: 0.734 → "0.73", "73.4 %"
(a fraction as a percentage, or a percentage as written), "1.2k" (k, M, B suffixes); a thousands separator is read.
Not checked: numbers inside code spans and code blocks (ids, paths, commands), numbers that are part of a name (a run
id, gpt-oss-120b, ResNet50, 3D), versions (0.9.16), dates and times (2026-10-30, 14:05), markdown list markers, and
the integers 0, 1 and 2 standing alone (counting words: "two options", "step 1"). These are documented limits
(docs/assistant.md, "Grounding"): the check finds invented measurements, it does not prove an answer right.
"""
from __future__ import annotations

import json
import re

SEP = ',\u202f\u2009\u00a0'                 # thousands separators: a comma, the narrow, thin and no-break spaces
NUM = re.compile(r'(?P<sign>[-−])?(?P<int>\d{1,3}(?:[' + SEP + r']\d{3})+(?!\d)|\d+)(?P<frac>\.\d+)?'
                 r'(?P<suffix>\s?%|[kKMB](?![A-Za-z]))?')
CODE_BLOCK = re.compile(r'```.*?```', re.S)
CODE_SPAN = re.compile(r'`[^`\n]*`')
DATE_TIME = re.compile(r'\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?Z?\b|\b\d{1,2}:\d{2}(?::\d{2})?\b')
VERSION = re.compile(r'\b\d+(?:\.\d+){2,}\b')
LIST_MARK = re.compile(r'(?m)^\s*\d+[.)]\s')
FREE_INTS = {0, 1, 2}
ANY_NUM = re.compile(r'[-−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def _blank(text: str, pattern) -> str:
    return pattern.sub(lambda m: ' ' * (m.end() - m.start()), text)


def numbers(answer: str) -> list:
    """The checkable numbers of an answer: [{'text', 'start', 'end', 'value', 'decimals', 'percent', 'scale'}]."""
    t = answer or ''
    for p in (CODE_BLOCK, CODE_SPAN, DATE_TIME, VERSION, LIST_MARK):
        t = _blank(t, p)
    out = []
    for m in NUM.finditer(t):
        start, end = m.start(), m.end()
        if m.group('sign') and start > 0 and (t[start - 1].isalnum() or t[start - 1] == '-'):
            start += 1                                     # 5-10: a range, not a negative ten
        prev = t[start - 1] if start > 0 else ''
        if prev.isalpha() or (prev and prev in '_#/.\\') or (prev == '-' and start > 1 and t[start - 2].isalpha()):
            continue                                       # part of a name: ResNet50, gpt-oss-120b, job #12, a path
        nxt = t[end:end + 1]
        suffix = (m.group('suffix') or '').strip()
        if not suffix and (nxt == '_' or nxt.isdigit() or (nxt == 'D' and not t[end + 1:end + 2].isalpha())):
            continue                                       # 3D, 2D, run_12_x
        whole = re.sub('[' + SEP + ']', '', m.group('int'))
        frac = m.group('frac') or ''
        v = float(whole + frac)
        if m.group('sign') and start == m.start():
            v = -v
        dec = len(frac) - 1 if frac else 0
        if not frac and not suffix and abs(v) in FREE_INTS:
            continue
        scale = {'k': 1e3, 'K': 1e3, 'M': 1e6, 'B': 1e9}.get(suffix, 1.0)
        out.append({'text': answer[start:end], 'start': start, 'end': end, 'value': v, 'decimals': dec,
                    'percent': suffix == '%', 'scale': scale})
    return out


def _walk(obj, vals: list):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        vals.append(float(obj))
    elif isinstance(obj, str):
        for m in ANY_NUM.finditer(obj.replace(',', '')):
            try:
                vals.append(float(m.group().replace('−', '-')))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk(k, vals)
            _walk(v, vals)
    elif isinstance(obj, (list, tuple)):
        vals.append(float(len(obj)))
        for x in obj:
            _walk(x, vals)


def sources(messages: list) -> list:
    """Every number Falcon's data holds in the conversation (see the module doc for what counts)."""
    vals: list = []
    for m in messages or []:
        role = m.get('role')
        if role == 'tool':
            c = m.get('content')
            try:
                _walk(json.loads(c) if isinstance(c, str) else c, vals)
            except ValueError:
                _walk(c, vals)
        elif role == 'user':
            _walk(m.get('content') or '', vals)
        elif role == 'assistant':
            for tc in m.get('tool_calls') or []:
                a = (tc.get('function') or {}).get('arguments') or ''
                try:
                    _walk(json.loads(a), vals)
                except ValueError:
                    _walk(a, vals)
    return vals


def matches(n: dict, vals) -> bool:
    v, dec, scale = n['value'], n['decimals'], n['scale']
    tol = 0.5 * 10 ** -dec * scale + 1e-9
    cands = [(v * scale, tol)]
    if scale == 1.0:                                     # a fraction shown as a percentage (with the sign or not)
        cands.append((v / 100.0, 0.5 * 10 ** -dec / 100.0 + 1e-12))
    for s in vals:
        for c, t in cands:
            if abs(s - c) <= t + 1e-9 * max(1.0, abs(c)):
                return True
    return False


def check(answer: str, messages: list) -> list:
    """The answer's numbers no source holds (empty: every number is Falcon's)."""
    vals = sources(messages)
    return [n for n in numbers(answer) if not matches(n, vals)]
