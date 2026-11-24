"""The runtime monitor: an online (causal, bounded-memory) evaluator of Falcon's rule formulas, in the standard library
only. This file imports nothing of Falcon's: `falcon monitor gen` (core.monitorgen) copies it verbatim into a monitor
package as falcon_monitor.py, with the compiled rules beside it (rules.json), and the vehicle computer runs that copy.
core.monitorgen checks the copy against the offline checker (core.stl.evaluate, core.rules.check) sample by sample.

A compiled rule is data (JSON): {'id', 'title', 'formula', 'tree', 'signals', 'horizon_s': {'past', 'future'},
'sha'}; the tree's nodes are {'op', 'args', 'name', 'value', 'lo', 'hi'} as core.stl's (inf is the string "inf").
The rule must hold AT EVERY SAMPLE (as core.rules judges a rule: G φ), so the monitor evaluates φ at each sample:

    m = Monitor(rule)
    r = m.step(t, {'speed_mps': 4.2, 'lead_gap_m': 11.0, ...})    one sample: time in seconds (increasing), signals
    r = {'verdict': 'ok' | 'violated' | 'unknown' | 'pending', 'robustness': float | None,
         'decided': [(index, time, robustness), ...], 'pending': n, 'worst': float | None, 'violations': n}
    m.finish()                                                      the end of a trace: decide what is still pending

Semantics: exactly core.stl's quantitative semantics (robustness ρ, ρ ≥ 0 satisfied; and = min, or = max; windows
[t+a, t+b] / [t−b, t−a] with core.stl's 1e-9 s tolerance; Kleene three-valued for unknown samples, NaN; a missing
signal is unknown). What differs online is only WHEN a sample's value is known:

  - a formula with no future operator is decided at its own sample (delay 0);
  - a bounded future operator (eventually[a,b], always[a,b], until[a,b]) delays the verdict of sample i until a
    sample later than t_i + (the formula's future horizon) has arrived: the verdict is `pending` until then, and
    `decided` then carries sample i's value — the value core.stl gives on the whole trace. A step's verdict is
    `violated` if any sample decided at this step is violated, else `ok` if any is known, else `unknown`;
  - finish() decides the samples still pending with their windows cut at the end, which is core.stl's finite-trace
    rule (G over an empty window +inf, F −inf). A monitor that never ends never calls it: its last horizon stays
    pending.

Refused at construction (MonitorError): an unbounded FUTURE operator inside the formula (eventually[a,inf],
always[a,inf] other than the rule's own outer "at every sample", until[a,inf]) — its value is never known online —
and since[a,inf] (this evaluator keeps since's window, which would be unbounded). once and historically may be
unbounded (a running maximum / minimum: constant memory).

Memory: per temporal operator, the samples inside its window (window length / sample period) plus the samples waiting
on its future horizon; `max_buffer` (default 200 000 samples per operator) is a hard cap — exceeding it raises
MonitorError rather than growing. Sample times must increase by more than MIN_STEP_S (1e-6 s).

SignalMapper turns the latest values of mapped inputs into one sample (stale inputs become unknown); it is what the
rclpy node (falcon_monitor_node.py in the package) feeds from its subscriptions. It knows nothing of ROS.
"""
from __future__ import annotations

import json
import math
from collections import deque

VERSION = 'falcon.monitor_runtime/1'
EPS = 1e-9                      # core.stl's window tolerance
MIN_STEP_S = 1e-6
NAN = float('nan')
INF = float('inf')
MAX_BUFFER = 200000


class MonitorError(ValueError):
    """A formula this evaluator cannot monitor, or samples it cannot take; the message says why."""


def kmin(a: float, b: float) -> float:
    """Kleene conjunction of robustness values (core.stl.kmin)."""
    if a != a:
        return b if b < 0 else NAN          # (NaN < 0 is False: both unknown stays unknown)
    if b != b:
        return a if a < 0 else NAN
    return a if a <= b else b


def kmax(a: float, b: float) -> float:
    """Kleene disjunction (core.stl.kmax)."""
    if a != a:
        return b if b >= 0 else NAN
    if b != b:
        return a if a >= 0 else NAN
    return a if a >= b else b


def _num(v) -> float:
    if isinstance(v, str):
        return {'inf': INF, '-inf': -INF, 'nan': NAN}[v]
    return float(v)


def _div(a: float, b: float) -> float:
    """IEEE division as numpy's (Python raises on a zero divisor)."""
    if b == 0.0:
        if a != a or a == 0.0:
            return NAN
        return math.copysign(INF, a) * math.copysign(1.0, b)
    return a / b


# ================================================================================================ expressions
def _expr(n: dict):
    """A node without temporal operators → f(sample dict) -> float (an expression's value or a formula's ρ)."""
    o = n['op']
    args = [_expr(a) for a in n.get('args') or ()]
    if o == 'num':
        c = _num(n['value'])
        return lambda s: c
    if o == 'name':
        k = n['name']
        return lambda s: s.get(k, NAN)
    if o == 'neg' or o == 'not':
        f = args[0]
        return lambda s: -f(s)
    if o in ('+', '-', '*', '/'):
        f, g = args
        if o == '+':
            return lambda s: f(s) + g(s)
        if o == '-':
            return lambda s: f(s) - g(s)
        if o == '*':
            return lambda s: f(s) * g(s)
        return lambda s: _div(f(s), g(s))
    if o == 'fn':
        name = n['name']
        if name == 'abs':
            f = args[0]
            return lambda s: abs(f(s))
        if name == 'default':
            f, g = args

            def default(s):
                v = f(s)
                return g(s) if v != v else v
            return default
        pick = min if name == 'min' else max

        def many(s):
            vs = [f(s) for f in args]
            for v in vs:
                if v != v:
                    return NAN
            return pick(vs)
        return many
    if o == 'cmp':
        f, g = args
        if n['name'] in ('>=', '>'):
            return lambda s: f(s) - g(s)        # NaN operands give NaN, as core.stl's
        return lambda s: g(s) - f(s)
    if o == 'bool':
        k = n['name']
        return lambda s: s.get(k, NAN) - 0.5
    if o == 'true':
        return lambda s: INF
    if o == 'false':
        return lambda s: -INF
    if o in ('and', 'or', '->'):
        f, g = args
        if o == 'and':
            return lambda s: kmin(f(s), g(s))
        if o == 'or':
            return lambda s: kmax(f(s), g(s))
        return lambda s: kmax(-f(s), g(s))
    raise MonitorError(f'unknown node {o!r}')


TEMPORAL = ('G', 'F', 'H', 'O', 'U', 'S')


def is_temporal(n: dict) -> bool:
    return n['op'] in TEMPORAL or any(is_temporal(a) for a in n.get('args') or ())


def horizon(n: dict) -> tuple:
    """(past, future) seconds a tree looks at around t (core.stl.horizon on the compiled tree)."""
    sub = [horizon(a) for a in n.get('args') or ()] or [(0.0, 0.0)]
    p, f = max(s[0] for s in sub), max(s[1] for s in sub)
    if n['op'] in ('G', 'F', 'U'):
        return p, _num(n['hi']) + f
    if n['op'] in ('H', 'O', 'S'):
        return _num(n['hi']) + p, f
    return p, f


# ================================================================================================ streams
# A stream node's step(t, sample, final) takes the new sample (t None: no new sample, only `final`) and returns the
# values it can now decide, in sample order. Every node of a tree sees every sample exactly once.
class _Atom:
    def __init__(self, n):
        self.f = _expr(n)

    def step(self, t, sample, final):
        return [] if t is None else [self.f(sample)]


class _Map:
    def __init__(self, child, fn):
        self.child, self.fn = child, fn

    def step(self, t, sample, final):
        return [self.fn(v) for v in self.child.step(t, sample, final)]


class _Zip:
    """A pointwise binary operator over two streams that decide at different delays."""

    def __init__(self, a, b, fn, cap):
        self.a, self.b, self.fn, self.cap = a, b, fn, cap
        self.qa, self.qb = deque(), deque()

    def step(self, t, sample, final):
        self.qa.extend(self.a.step(t, sample, final))
        self.qb.extend(self.b.step(t, sample, final))
        out = []
        while self.qa and self.qb:
            out.append(self.fn(self.qa.popleft(), self.qb.popleft()))
        if len(self.qa) > self.cap or len(self.qb) > self.cap:
            raise MonitorError('the monitor\'s buffer limit was reached (samples arrive faster than the formula\'s '
                               'horizon allows: max_buffer)')
        return out


class _Timed:
    """Sample times by index for a temporal node, trimmed from the front."""

    def __init__(self):
        self.t, self.tb = deque(), 0

    def n(self) -> int:
        return self.tb + len(self.t)

    def at(self, j: int) -> float:
        return self.t[j - self.tb]

    def trim(self, keep_from: int):
        while self.tb < keep_from and self.t:
            self.t.popleft()
            self.tb += 1


class _Window(_Timed):
    """G / F / H / O over [lo, hi]: a sliding Kleene minimum (maximum) of the child's values (core.stl._window_min /
    _window_max): the known values' minimum by a monotonic queue, the unknown ones counted."""

    def __init__(self, child, op, lo, hi, cap):
        super().__init__()
        self.child, self.past, self.is_min = child, op in ('H', 'O'), op in ('G', 'H')
        self.lo, self.hi, self.cap = lo, hi, cap
        self.forever = self.past and math.isinf(hi)
        self.v = deque()               # the child's values from index self.a on (not yet in the window)
        self.i = 0                     # the next sample to decide
        self.f = 0                     # the window's first index
        self.a = 0                     # the next index to enter the window
        self.l = 0                     # the number of samples with times ≤ the window's end (last index + 1)
        self.mono, self.unk = deque(), deque()
        self.m, self.nunk = INF, 0     # forever: the running minimum of the keys and the unknown count

    def _add(self, j, v):
        if v != v:
            if self.forever:
                self.nunk += 1
            else:
                self.unk.append(j)
            return
        key = v if self.is_min else -v
        if self.forever:
            if key < self.m:
                self.m = key
            return
        mono = self.mono
        while mono and mono[-1][1] >= key:
            mono.pop()
        mono.append((j, key))

    def step(self, t, sample, final):
        if t is not None:
            self.t.append(t)
        self.v.extend(self.child.step(t, sample, final))
        out = []
        n = self.n()
        while self.i < n:
            ti = self.at(self.i)
            if self.past:
                start, end = (ti - self.hi) - EPS, (ti - self.lo) + EPS
            else:
                start, end = (ti + self.lo) - EPS, (ti + self.hi) + EPS
            while self.l < n and self.at(self.l) <= end:
                self.l += 1
            if not (self.past or final or self.l < n):
                break                                  # a later sample may still fall in the window
            last = self.l - 1
            if last >= self.a + len(self.v):
                break                                  # the child has not decided the window's last sample yet
            while self.a <= last:
                self._add(self.a, self.v.popleft())
                self.a += 1
            if self.forever:
                empty, m, unk = last < 0, self.m, self.nunk
            else:
                while self.f < n and self.at(self.f) < start:
                    self.f += 1
                first = self.f
                while self.mono and self.mono[0][0] < first:
                    self.mono.popleft()
                while self.unk and self.unk[0] < first:
                    self.unk.popleft()
                empty = last < first
                m = self.mono[0][1] if self.mono else INF
                unk = len(self.unk)
            if self.is_min:
                r = INF if empty else (NAN if unk > 0 and not m < 0 else m)
            else:
                r = -INF if empty else (NAN if unk > 0 and not m <= 0 else -m)
            out.append(r)
            self.i += 1
        self.trim(min(self.i, self.f, self.l) if not self.forever else min(self.i, self.l))
        if max(len(self.t), len(self.v), len(self.mono), len(self.unk)) > self.cap:
            raise MonitorError('the monitor\'s buffer limit was reached (the window holds more samples than '
                               'max_buffer)')
        return out


class _Until(_Timed):
    """φ until[lo,hi] ψ / φ since[lo,hi] ψ with finite bounds: core.stl._until's loop on the buffered window."""

    def __init__(self, a, b, op, lo, hi, cap):
        super().__init__()
        self.ca, self.cb, self.past, self.lo, self.hi, self.cap = a, b, op == 'S', lo, hi, cap
        self.va, self.vb, self.base = deque(), deque(), 0      # both children's values from index base on
        self.i = self.f = self.l = 0

    def step(self, t, sample, final):
        if t is not None:
            self.t.append(t)
        self.va.extend(self.ca.step(t, sample, final))
        self.vb.extend(self.cb.step(t, sample, final))
        out = []
        n = self.n()
        while self.i < n:
            i = self.i
            ti = self.at(i)
            if self.past:
                start, end = (ti - self.hi) - EPS, (ti - self.lo) + EPS
            else:
                start, end = (ti + self.lo) - EPS, (ti + self.hi) + EPS
            while self.l < n and self.at(self.l) <= end:
                self.l += 1
            if not (self.past or final or self.l < n):
                break
            last = self.l - 1
            have = self.base + min(len(self.va), len(self.vb))
            if max(last, i if self.past else -1) >= have:
                break
            while self.f < n and self.at(self.f) < start:
                self.f += 1
            first = self.f
            best = -INF
            if last >= first:
                a, b, o = self.va, self.vb, self.base
                run = INF
                rng = range(i, first - 1, -1) if self.past else range(i, last + 1)
                for j in rng:
                    if (j <= last) if self.past else (j >= first):
                        best = kmax(best, kmin(b[j - o], run))
                    run = kmin(run, a[j - o])
                    if run < 0 and run <= best and best == best:
                        break
            out.append(best)
            self.i += 1
            keep = min(self.i, self.f) if self.past else self.i
            while self.base < keep:
                self.va.popleft()
                self.vb.popleft()
                self.base += 1
        self.trim(min(self.i, self.f, self.l))
        if max(len(self.t), len(self.va), len(self.vb)) > self.cap:
            raise MonitorError('the monitor\'s buffer limit was reached (the window holds more samples than '
                               'max_buffer)')
        return out


def _stream(n: dict, cap: int):
    if not is_temporal(n):
        return _Atom(n)
    o = n['op']
    kids = [_stream(a, cap) for a in n.get('args') or ()]
    if o == 'not':
        return _Map(kids[0], lambda v: -v)
    if o == 'and':
        return _Zip(kids[0], kids[1], kmin, cap)
    if o == 'or':
        return _Zip(kids[0], kids[1], kmax, cap)
    if o == '->':
        return _Zip(kids[0], kids[1], lambda a, b: kmax(-a, b), cap)
    lo, hi = _num(n.get('lo', 0.0)), _num(n.get('hi', 'inf'))
    if o in ('G', 'F') and math.isinf(hi):
        raise MonitorError(f'an unbounded future operator ({"always" if o == "G" else "eventually"}[{lo:g},inf]) '
                           'cannot be monitored online: its value is never known; give it a bound')
    if o in ('U', 'S') and math.isinf(hi):
        raise MonitorError(f'{"until" if o == "U" else "since"}[{lo:g},inf] cannot be monitored: give it a bound '
                           '(once and historically may be unbounded)')
    if o in ('G', 'F', 'H', 'O'):
        return _Window(kids[0], o, lo, hi, cap)
    if o in ('U', 'S'):
        return _Until(kids[0], kids[1], o, lo, hi, cap)
    raise MonitorError(f'a temporal operator under {o!r} is not a formula')


# ================================================================================================ the monitor
class Monitor:
    """One rule's online monitor (module doc). rule: the compiled rule (a dict, or its JSON text)."""

    def __init__(self, rule, max_buffer: int = MAX_BUFFER):
        self.rule = json.loads(rule) if isinstance(rule, str) else rule
        self.id = self.rule.get('id', 'rule')
        self.max_buffer = int(max_buffer)
        self.delay_s = horizon(self.rule['tree'])[1]
        self.reset()

    def reset(self):
        self.root = _stream(self.rule['tree'], self.max_buffer)
        self.times = deque()            # the times of the samples not decided yet
        self.n = self.decided_n = 0
        self.last_t = None
        self.worst = None               # the lowest known robustness so far
        self.worst_t = None
        self.violations = 0             # violated samples so far
        self.unknown = 0                # samples decided unknown so far

    def _take(self, vals) -> dict:
        decided = []
        for v in vals:
            t = self.times.popleft()
            decided.append((self.decided_n, t, v))
            self.decided_n += 1
            if v != v:
                self.unknown += 1
                continue
            if v < 0:
                self.violations += 1
            if self.worst is None or v < self.worst:
                self.worst, self.worst_t = v, t
        known = [v for _i, _t, v in decided if v == v]
        if not decided:
            verdict, rob = 'pending', None
        elif not known:
            verdict, rob = 'unknown', None
        else:
            rob = min(known)
            verdict = 'violated' if rob < 0 else 'ok'
        return {'verdict': verdict, 'robustness': rob, 'decided': decided, 'pending': len(self.times),
                'worst': self.worst, 'violations': self.violations}

    def step(self, t: float, signals: dict) -> dict:
        t = float(t)
        if self.last_t is not None and not t - self.last_t > MIN_STEP_S:
            raise MonitorError(f'sample times must increase (by more than {MIN_STEP_S:g} s): {t!r} after '
                               f'{self.last_t!r}')
        self.last_t = t
        if 't' not in signals:
            signals = dict(signals, t=t)
        self.times.append(t)
        self.n += 1
        return self._take(self.root.step(t, signals, False))

    def finish(self) -> dict:
        """The end of the trace: the pending samples decided with their windows cut at the end."""
        return self._take(self.root.step(None, None, True))

    def state(self) -> str:
        """The run so far, as core.rules.check words it: violated | satisfied | not evaluable."""
        if self.worst is None:
            return 'not evaluable'
        return 'violated' if self.worst < 0 else 'satisfied'


class MonitorSet:
    """Several rules stepped together: step() → {'alarm': a rule is violated at this step, 'rules': {id: result}}."""

    def __init__(self, rules, max_buffer: int = MAX_BUFFER):
        rules = json.loads(rules) if isinstance(rules, str) else rules
        self.rules = rules['rules'] if isinstance(rules, dict) else list(rules)
        self.monitors = [Monitor(r, max_buffer) for r in self.rules]
        self.signals = sorted({s for r in self.rules for s in r.get('signals') or ()})

    def step(self, t: float, signals: dict) -> dict:
        if 't' not in signals:
            signals = dict(signals, t=float(t))
        res = {m.id: m.step(t, signals) for m in self.monitors}
        return {'t': float(t), 'alarm': any(r['verdict'] == 'violated' for r in res.values()), 'rules': res}

    def finish(self) -> dict:
        res = {m.id: m.finish() for m in self.monitors}
        return {'t': None, 'alarm': any(r['verdict'] == 'violated' for r in res.values()), 'rules': res}

    def summary(self) -> dict:
        return {m.id: {'state': m.state(), 'worst': m.worst, 'worst_t': m.worst_t, 'violations': m.violations,
                       'unknown': m.unknown, 'samples': m.decided_n, 'pending': len(m.times)} for m in self.monitors}


# ================================================================================================ inputs → samples
def get_field(msg, path: str):
    """A message's field by a dotted path ("twist.twist.linear.x"; "" the message itself; a list index as a number:
    "ranges.0"); None when it is not there."""
    cur = msg
    for part in [p for p in path.split('.') if p]:
        try:
            cur = cur[int(part)] if part.lstrip('-').isdigit() else getattr(cur, part)
        except (AttributeError, IndexError, KeyError, TypeError):
            return None
    return cur


class SignalMapper:
    """The mapping file's `signals` → samples. An entry is {'topic', 'type', 'field', 'scale', 'offset'} (an input: a
    message field, value·scale + offset; a boolean becomes 0/1) or {'from': another signal, 'scale', 'offset'}
    (derived) or {'const': number}. update(name, value, now) stores an input's latest value; sample(now) gives every
    mapped signal: unknown (NaN) when never received, older than stale_s (or stamped later than now by more than
    stale_s: the clock jumped back), when its source is unknown, or when the value is not a number."""

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.stale_s = float(mapping.get('stale_s', 0.5))
        self.entries = dict(mapping.get('signals') or {})
        self.latest = {}                       # name -> (value, time received)
        for k, e in self.entries.items():
            if not isinstance(e, dict) or not ('topic' in e or 'from' in e or 'const' in e):
                raise MonitorError(f'mapping: signal {k!r} needs a topic, a from or a const')
            if 'from' in e and e['from'] not in self.entries:
                raise MonitorError(f'mapping: signal {k!r} is derived from {e["from"]!r}, which is not mapped')

    def inputs(self) -> list:
        """[(signal, topic, type, field)] of the mapped inputs."""
        return [(k, e['topic'], e.get('type', 'std_msgs/msg/Float64'), e.get('field', 'data'))
                for k, e in self.entries.items() if 'topic' in e]

    def update(self, name: str, value, now: float):
        e = self.entries[name]
        try:
            v = float(value) * float(e.get('scale', 1.0)) + float(e.get('offset', 0.0))
        except (TypeError, ValueError):
            v = NAN
        self.latest[name] = (v, float(now))

    def on_message(self, name: str, msg, now: float):
        self.update(name, get_field(msg, self.entries[name].get('field', 'data')), now)

    def _value(self, name: str, now: float, depth: int = 0) -> float:
        e = self.entries[name]
        if 'const' in e:
            return float(e['const'])
        if 'from' in e:
            if depth > 8:
                return NAN
            v = self._value(e['from'], now, depth + 1)
            return v * float(e.get('scale', 1.0)) + float(e.get('offset', 0.0))
        got = self.latest.get(name)
        if got is None or abs(now - got[1]) > self.stale_s:        # too old, or from before the clock jumped back
            return NAN
        return got[0]

    def sample(self, now: float) -> dict:
        return {k: self._value(k, now) for k in self.entries}


def load_rules(path: str) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
