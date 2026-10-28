"""Signal temporal logic: a small formula language, its parser (no eval of Python) and a quantitative evaluator with
bounded temporal operators (numpy only). Falcon's traffic rules (core.rules) are written in it.

Signals. A trace is a set of named signals sampled at the same increasing times t (seconds; the steps need not be
uniform). A value is a float; NaN means *unknown* at that sample (not recorded, not measurable there); ±inf is allowed
(a gap to a lead vehicle when there is none is +inf).

Formulas (precedence from loose to tight; every binary operator but -> is left-associative):

    φ -> ψ                      implication (right-associative)
    φ or ψ                      disjunction
    φ and ψ                     conjunction
    φ until[a,b] ψ              ψ holds at some t' in [t+a, t+b] and φ at every sample in [t, t')
    φ since[a,b] ψ              ψ held at some t' in [t-b, t-a] and φ at every sample in (t', t]
    not φ                       negation
    always[a,b] φ    G[a,b] φ   φ at every sample in [t+a, t+b]
    eventually[a,b] φ F[a,b] φ  φ at some sample in [t+a, t+b]
    historically[a,b] φ H[a,b] φ  φ at every sample in [t-b, t-a]
    once[a,b] φ      O[a,b] φ   φ at some sample in [t-b, t-a]
    e1 <= e2, e1 < e2, e1 >= e2, e1 > e2       a predicate over expressions
    NAME                        a boolean signal: NAME >= 0.5
    true, false, ( φ )

The interval is optional (G φ is G[0, inf]); bounds are numbers in seconds, constants, or inf, with 0 <= a <= b.
Expressions: numbers (1, 2.5, 1e-3), signal and constant names, + - * /, unary -, parentheses, and the functions
abs(e), min(e, e, …), max(e, e, …), default(e, d) (d where e is unknown). Keywords: and or not until since always
eventually historically once G F H O U S true false inf. A comment starts with # (in rule files).

Quantitative semantics (robustness ρ, Fainekos & Pappas 2009; Donzé & Maler 2010): ρ(e1 >= e2, t) = e1(t) − e2(t)
and ρ(e1 <= e2, t) = e2(t) − e1(t) (> and < have the same robustness: strictness is not represented; a robustness of
exactly 0 counts as satisfied); ρ(NAME) = NAME(t) − 0.5; ρ(not φ) = −ρ(φ); and = min, or = max, φ -> ψ = max(−ρφ, ρψ);
G/H = min over the window, F/O = max; until: max over t' of min(ρψ(t'), min over [t, t') of ρφ); since: max over t' of
min(ρψ(t'), min over (t', t] of ρφ). ρ ≥ 0 means satisfied, ρ < 0 violated, and |ρ| is how far the signals are from
the other verdict, in the units of the predicate (metres, m/s, …: a conjunction of predicates in different units
compares numbers in different units — the usual STL caveat).

Finite traces. A window is cut at the trace's ends: G over an empty window is +inf (nothing to violate), F over an
empty window is −inf (nothing happened). So G[0,5] φ near the end of a trace checks only what remains; F[0,5] φ in the
last 5 s can fail only for lack of trace. Rules that look ahead state it.

Unknown samples (three-valued, Kleene): and(ρ, unknown) is ρ when ρ < 0 (violated whatever the unknown is), else
unknown; or(ρ, unknown) is ρ when ρ ≥ 0, else unknown; a window with an unknown sample is unknown unless its known
samples already decide it (G: a known violation; F: a known satisfaction), and then the robustness is the known
samples' bound. So an unknown never makes a violation: it can only leave a sample unknown. Arithmetic and predicates
over an unknown are unknown; default(e, d) is how a rule supplies a value.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

KEYWORDS = {'and', 'or', 'not', 'until', 'since', 'always', 'eventually', 'historically', 'once', 'G', 'F', 'H', 'O',
            'U', 'S', 'true', 'false', 'inf'}
TEMPORAL = {'always': 'G', 'G': 'G', 'eventually': 'F', 'F': 'F', 'historically': 'H', 'H': 'H', 'once': 'O',
            'O': 'O'}
BINARY_TEMPORAL = {'until': 'U', 'U': 'U', 'since': 'S', 'S': 'S'}
FUNCTIONS = {'abs': (1, 1), 'min': (2, 64), 'max': (2, 64), 'default': (2, 2)}
_TOKEN = re.compile(r'\s*(?:(?P<num>\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)|'
                    r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>->|<=|>=|<|>|[()\[\],+\-*/]))')


class STLError(ValueError):
    """A formula that does not parse or cannot be evaluated; the message says where."""


@dataclass
class Node:
    """A formula or expression node. op: 'num' (value), 'name' (name), 'neg', '+', '-', '*', '/', 'fn' (name, args),
    'cmp' (name: <= < >= >, args [l, r]), 'bool' (name: a boolean signal), 'true', 'false', 'not', 'and', 'or', '->',
    'G' 'F' 'H' 'O' (lo, hi, args [φ]), 'U' 'S' (lo, hi, args [φ, ψ])."""
    op: str
    args: list = field(default_factory=list)
    value: float | None = None
    name: str | None = None
    lo: float = 0.0
    hi: float = math.inf

    def text(self) -> str:
        """The node back as formula text (canonical spacing, full parentheses)."""
        o = self.op
        if o == 'num':
            return _fmt(self.value)
        if o in ('name', 'bool'):
            return self.name
        if o == 'neg':
            return f'-{self.args[0].text()}'
        if o in ('+', '-', '*', '/'):
            return f'({self.args[0].text()} {o} {self.args[1].text()})'
        if o == 'fn':
            return f'{self.name}({", ".join(a.text() for a in self.args)})'
        if o == 'cmp':
            return f'{self.args[0].text()} {self.name} {self.args[1].text()}'
        if o in ('true', 'false'):
            return o
        if o == 'not':
            return f'not ({self.args[0].text()})'
        if o in ('and', 'or', '->'):
            return f'({self.args[0].text()} {o} {self.args[1].text()})'
        b = f'[{_fmt(self.lo)},{_fmt(self.hi)}]'
        if o in ('G', 'F', 'H', 'O'):
            return f'{o}{b} ({self.args[0].text()})'
        return f'({self.args[0].text()} {o}{b} {self.args[1].text()})'


def _fmt(v) -> str:
    if v is None:
        return 'nan'
    if math.isinf(v):
        return 'inf' if v > 0 else '-inf'
    return repr(float(v)) if v != int(v) else str(int(v))


# ================================================================================================ parsing
def tokenize(text: str) -> list:
    """[(kind, value, position)] with kind num | name | op; raises STLError at an unknown character."""
    out, i = [], 0
    text = text.split('#', 1)[0] if '#' in text else text
    while i < len(text):
        if text[i:].strip() == '':
            break
        m = _TOKEN.match(text, i)
        if not m or m.end() == i:
            j = len(text) - len(text[i:].lstrip())
            raise STLError(f'unexpected character {text[j]!r} at column {j + 1}')
        kind = m.lastgroup
        out.append((kind, m.group(kind), m.start(kind)))
        i = m.end()
    return out


class _Parser:
    def __init__(self, text: str, consts: dict | None):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0
        self.consts = dict(consts or {})

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else ('end', '', len(self.text))

    def error(self, what: str):
        kind, val, pos = self.peek()
        got = 'the end' if kind == 'end' else repr(val)
        raise STLError(f'{what} at column {pos + 1}, found {got}')

    def take(self, value=None, kind=None):
        k, v, _ = self.peek()
        if (value is not None and v != value) or (kind is not None and k != kind) or k == 'end':
            self.error(f'expected {value or kind}')
        self.i += 1
        return v

    def at(self, *values):
        k, v, _ = self.peek()
        return k != 'end' and v in values

    # formulas
    def formula(self) -> Node:
        left = self.disj()
        if self.at('->'):
            self.take('->')
            return Node('->', [left, self.formula()])
        return left

    def disj(self) -> Node:
        n = self.conj()
        while self.at('or'):
            self.take('or')
            n = Node('or', [n, self.conj()])
        return n

    def conj(self) -> Node:
        n = self.until()
        while self.at('and'):
            self.take('and')
            n = Node('and', [n, self.until()])
        return n

    def until(self) -> Node:
        n = self.unary()
        while self.peek()[0] == 'name' and self.peek()[1] in BINARY_TEMPORAL:
            op = BINARY_TEMPORAL[self.take()]
            lo, hi = self.interval()
            n = Node(op, [n, self.unary()], lo=lo, hi=hi)
        return n

    def interval(self) -> tuple:
        if not self.at('['):
            return 0.0, math.inf
        self.take('[')
        lo = self.bound()
        self.take(',')
        hi = self.bound()
        self.take(']')
        if not 0.0 <= lo <= hi:
            raise STLError(f'an interval [a, b] needs 0 <= a <= b; got [{_fmt(lo)}, {_fmt(hi)}]')
        return lo, hi

    def bound(self) -> float:
        k, v, _ = self.peek()
        if k == 'num':
            self.i += 1
            return float(v)
        if k == 'name' and v == 'inf':
            self.i += 1
            return math.inf
        if k == 'name' and v in self.consts:
            self.i += 1
            return float(self.consts[v])
        self.error('expected a number, a constant or inf as an interval bound')

    def unary(self) -> Node:
        k, v, _ = self.peek()
        if k == 'name' and v == 'not':
            self.take()
            return Node('not', [self.unary()])
        if k == 'name' and v in TEMPORAL:
            self.take()
            lo, hi = self.interval()
            return Node(TEMPORAL[v], [self.unary()], lo=lo, hi=hi)
        if k == 'name' and v in ('true', 'false'):
            self.take()
            return Node(v)
        save = self.i
        if self.at('('):                  # a parenthesised formula, or an expression that starts a predicate
            try:
                return self.predicate()
            except STLError:
                self.i = save
            self.take('(')
            n = self.formula()
            self.take(')')
            return n
        return self.predicate()

    def predicate(self) -> Node:
        left = self.expr()
        if self.at('<=', '<', '>=', '>'):
            op = self.take()
            return Node('cmp', [left, self.expr()], name=op)
        if left.op == 'name' and left.name not in self.consts:
            return Node('bool', name=left.name)
        self.error('expected a comparison (<=, <, >=, >)')

    # expressions
    def expr(self) -> Node:
        n = self.term()
        while self.at('+', '-'):
            op = self.take()
            n = Node(op, [n, self.term()])
        return n

    def term(self) -> Node:
        n = self.factor()
        while self.at('*', '/'):
            op = self.take()
            n = Node(op, [n, self.factor()])
        return n

    def factor(self) -> Node:
        k, v, _ = self.peek()
        if self.at('-'):
            self.take()
            return Node('neg', [self.factor()])
        if k == 'num':
            self.i += 1
            return Node('num', value=float(v))
        if k == 'name' and v == 'inf':
            self.i += 1
            return Node('num', value=math.inf)
        if k == 'name' and v in FUNCTIONS and self.peek(1)[1] == '(':
            self.take()
            self.take('(')
            args = [self.expr()]
            while self.at(','):
                self.take(',')
                args.append(self.expr())
            self.take(')')
            lo, hi = FUNCTIONS[v]
            if not lo <= len(args) <= hi:
                raise STLError(f'{v}() takes {lo if lo == hi else f"{lo} or more"} argument(s), got {len(args)}')
            return Node('fn', args, name=v)
        if k == 'name' and v not in KEYWORDS:
            self.i += 1
            if v in self.consts:
                return Node('num', value=float(self.consts[v]), name=v)
            return Node('name', name=v)
        if self.at('('):
            self.take('(')
            n = self.expr()
            self.take(')')
            return n
        self.error('expected a number, a name or (')


def parse(text: str, consts: dict | None = None) -> Node:
    """A formula's tree; constants (name → number) are substituted. Raises STLError with the column."""
    p = _Parser(text, consts)
    if not p.toks:
        raise STLError('an empty formula')
    n = p.formula()
    if p.peek()[0] != 'end':
        p.error('unexpected text after the formula')
    return n


def parse_expr(text: str, consts: dict | None = None) -> Node:
    p = _Parser(text, consts)
    n = p.expr()
    if p.peek()[0] != 'end':
        p.error('unexpected text after the expression')
    return n


def names(node: Node) -> set:
    """The signal names a formula reads."""
    out = set()
    if node.op in ('name', 'bool'):
        out.add(node.name)
    for a in node.args:
        out |= names(a)
    return out


def horizon(node: Node) -> tuple:
    """(past, future) seconds the formula looks at around t (inf when unbounded)."""
    if node.op in ('G', 'F', 'U'):
        sub = [horizon(a) for a in node.args]
        return max(s[0] for s in sub), node.hi + max(s[1] for s in sub)
    if node.op in ('H', 'O', 'S'):
        sub = [horizon(a) for a in node.args]
        return node.hi + max(s[0] for s in sub), max(s[1] for s in sub)
    sub = [horizon(a) for a in node.args] or [(0.0, 0.0)]
    return max(s[0] for s in sub), max(s[1] for s in sub)


# ================================================================================================ evaluation
def kmin(a, b):
    """Kleene conjunction of robustness arrays (NaN unknown): min where both are known; where one is unknown, the
    known one if it is negative (decided), else unknown."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.minimum(a, b)
    ua, ub = np.isnan(a), np.isnan(b)
    out = np.where(ua & ~ub, np.where(b < 0, b, np.nan), out)
    out = np.where(ub & ~ua, np.where(a < 0, a, np.nan), out)
    return out


def kmax(a, b):
    """Kleene disjunction: max where both are known; with one unknown, the known one if it is ≥ 0 (satisfied whatever
    the unknown is: its robustness is a lower bound), else unknown."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.maximum(a, b)
    ua, ub = np.isnan(a), np.isnan(b)
    out = np.where(ua & ~ub, np.where(b >= 0, b, np.nan), out)
    out = np.where(ub & ~ua, np.where(a >= 0, a, np.nan), out)
    return out


def _windows(t: np.ndarray, lo: float, hi: float, past: bool) -> tuple:
    """Per sample i the inclusive index range [first, last] of the samples with times in [t_i + lo, t_i + hi] (or
    [t_i − hi, t_i − lo] for the past); first > last when empty. A tolerance of 1e-9 s absorbs float steps."""
    eps = 1e-9
    if past:
        first = np.searchsorted(t, t - hi - eps, 'left') if math.isfinite(hi) else np.zeros(len(t), int)
        last = np.searchsorted(t, t - lo + eps, 'right') - 1
    else:
        first = np.searchsorted(t, t + lo - eps, 'left')
        last = (np.searchsorted(t, t + hi + eps, 'right') - 1) if math.isfinite(hi) else np.full(len(t), len(t) - 1)
    return first.astype(int), last.astype(int)


class _RangeMin:
    """Range minima of the known values and counts of unknown ones, O(1) per query after an O(n log n) table."""

    def __init__(self, v: np.ndarray):
        v = np.asarray(v, float)
        self.n = len(v)
        unk = np.isnan(v)
        self.cum = np.concatenate([[0], np.cumsum(unk)])
        base = np.where(unk, np.inf, v)
        self.table = [base]
        k = 1
        while (1 << k) <= self.n:
            prev = self.table[-1]
            h = 1 << (k - 1)
            self.table.append(np.minimum(prev[:-h], prev[h:]))
            k += 1

    def query(self, first: np.ndarray, last: np.ndarray) -> tuple:
        """(min of the known values, +inf when none; number of unknown values; empty mask)."""
        empty = last < first
        f = np.where(empty, 0, first)
        la = np.where(empty, 0, last)
        length = la - f + 1
        k = np.floor(np.log2(np.maximum(length, 1))).astype(int)
        mins = np.full(len(f), np.inf)
        for lev in np.unique(k):
            sel = k == lev
            tab = self.table[lev]
            mins[sel] = np.minimum(tab[f[sel]], tab[la[sel] - (1 << lev) + 1])
        unknown = self.cum[la + 1] - self.cum[f]
        return np.where(empty, np.inf, mins), np.where(empty, 0, unknown), empty


def _window_min(v: np.ndarray, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Kleene window minimum: +inf for an empty window; unknown when a window has unknown samples and its known
    minimum is not negative (or it has no known sample)."""
    m, unk, empty = _RangeMin(v).query(first, last)
    out = np.where((unk > 0) & ~(m < 0), np.nan, m)
    return np.where(empty, np.inf, out)


def _window_max(v: np.ndarray, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Kleene window maximum: −inf for an empty window; unknown when a window has unknown samples and its known
    maximum is negative (or it has no known sample)."""
    m, unk, empty = _RangeMin(-np.asarray(v, float)).query(first, last)
    out = np.where((unk > 0) & ~(m <= 0), np.nan, -m)
    return np.where(empty, -np.inf, out)


def _until(a: np.ndarray, b: np.ndarray, t: np.ndarray, lo: float, hi: float, past: bool) -> np.ndarray:
    """φ until/since ψ (module doc), Kleene-lifted; O(n · window)."""
    n = len(t)
    first, last = _windows(t, lo, hi, past)
    out = np.full(n, -np.inf)
    for i in range(n):
        if last[i] < first[i]:
            continue
        best = -np.inf
        if not past:
            run = np.inf                       # min of φ over [i, j)
            for j in range(i, last[i] + 1):
                if j >= first[i]:
                    best = float(kmax(best, kmin(b[j], run)))
                run = float(kmin(run, a[j]))
                if run < 0 and run <= best and not math.isnan(best):
                    break
        else:
            run = np.inf                       # min of φ over (j, i]
            for j in range(i, first[i] - 1, -1):
                if j <= last[i]:
                    best = float(kmax(best, kmin(b[j], run)))
                run = float(kmin(run, a[j]))
                if run < 0 and run <= best and not math.isnan(best):
                    break
        out[i] = best
    return out


def evaluate(node: Node, signals: dict, t) -> np.ndarray:
    """The robustness of a formula (or the value of an expression) at every sample: float array, NaN unknown.
    signals: name → array (or a number for a constant signal) of the samples' length; t: the sample times."""
    t = np.asarray(t, float)
    n = len(t)
    if n and np.any(np.diff(t) <= 0):
        raise STLError('sample times must increase')

    def sig(name):
        if name not in signals:
            raise STLError(f'unknown signal {name!r}')
        v = signals[name]
        a = np.full(n, float(v)) if np.isscalar(v) or v is None else np.asarray(v, float)
        if a.shape != (n,):
            raise STLError(f'signal {name!r} has {a.shape} samples, the trace {n}')
        return a

    def ev(x: Node) -> np.ndarray:
        o = x.op
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            if o == 'num':
                return np.full(n, float(x.value))
            if o == 'name':
                return sig(x.name)
            if o == 'neg':
                return -ev(x.args[0])
            if o in ('+', '-', '*', '/'):
                a, b = ev(x.args[0]), ev(x.args[1])
                return {'+': a + b, '-': a - b, '*': a * b, '/': a / b}[o]
            if o == 'fn':
                vs = [ev(a) for a in x.args]
                if x.name == 'abs':
                    return np.abs(vs[0])
                if x.name == 'default':
                    return np.where(np.isnan(vs[0]), vs[1], vs[0])
                st = np.vstack(vs)
                bad = np.isnan(st).any(0)
                r = st.min(0) if x.name == 'min' else st.max(0)
                return np.where(bad, np.nan, r)
            if o == 'cmp':
                a, b = ev(x.args[0]), ev(x.args[1])
                r = a - b if x.name in ('>=', '>') else b - a
                return np.where(np.isnan(a) | np.isnan(b), np.nan, r)
            if o == 'bool':
                return sig(x.name) - 0.5
            if o == 'true':
                return np.full(n, np.inf)
            if o == 'false':
                return np.full(n, -np.inf)
            if o == 'not':
                return -ev(x.args[0])
            if o == 'and':
                return kmin(ev(x.args[0]), ev(x.args[1]))
            if o == 'or':
                return kmax(ev(x.args[0]), ev(x.args[1]))
            if o == '->':
                return kmax(-ev(x.args[0]), ev(x.args[1]))
            if o in ('G', 'F', 'H', 'O'):
                v = ev(x.args[0])
                first, last = _windows(t, x.lo, x.hi, past=o in ('H', 'O'))
                return _window_min(v, first, last) if o in ('G', 'H') else _window_max(v, first, last)
            if o in ('U', 'S'):
                return _until(ev(x.args[0]), ev(x.args[1]), t, x.lo, x.hi, past=o == 'S')
        raise STLError(f'unknown node {o!r}')
    return ev(node)


def intervals(rho, t, end_dt: float | None = None) -> list:
    """The maximal runs of samples with ρ < 0 as [{'start', 'end', 'i0', 'i1', 'min_robustness', 'at'}] (times in
    seconds; end is the last violating sample's time plus end_dt, the trace's step, when given)."""
    rho = np.asarray(rho, float)
    t = np.asarray(t, float)
    bad = rho < 0                              # NaN compares False: an unknown sample never violates
    out = []
    i = 0
    n = len(rho)
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and bad[j + 1]:
            j += 1
        seg = rho[i:j + 1]
        k = i + int(np.argmin(seg))
        out.append({'start': float(t[i]), 'end': float(t[j] + (end_dt or 0.0)), 'i0': i, 'i1': j,
                    'min_robustness': float(rho[k]), 'at': float(t[k])})
        i = j + 1
    return out
