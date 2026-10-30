"""Live trigger rules: ``when <condition> -> <action>`` evaluated at every tick of a recorded drive, a scenario run or
a Live session (standard library; the runner is falcon_autonomy.workers.triggers).

    when ego.speed > 40 and visibility < 30 -> spawn jaywalker
    when actor.ttc < 2 or ego.light:red and ego.light_m < 25 -> marker "close call"; weather fog_density=40
    when time >= 8 -> fault cam-front dirt cover=0.3 for=4

A rule document (the project's "trigger_rules", docs/scenarios.md "Trigger rules"):

    {"id", "name", "rule": "when … -> …", "repeat": "once" | "every" | "cooldown", "cooldown_s": number ≥ 0,
     "enabled": bool, "applies": ["drives", "scenarios", "live"], "about"}

Repeat. "once" (the default): the first tick the condition holds, never again in the run. "every": each tick the
condition becomes true (false at the tick before, or the first tick), skipped while within cooldown_s of the rule's
last firing. "cooldown": every tick the condition holds, at most once every cooldown_s (it fires again after
cooldown_s while the condition stays true; cooldown_s must be > 0).

The condition language reuses the run query language's comparisons (core.runindex: ``key>n``, ``key<n``, ``key>=n``,
``key<=n``, ``key=n`` on numbers and ``key:value`` on text, a value's unit optional) joined by ``and``, ``or``, ``not``
and parentheses (``and`` binds tighter than ``or``); ``!=`` negates ``=``. Signals (SIGNALS; every one from the world
snapshot of the frame just ticked, or the scenario director's kinematic state where it drives an actor):

    time            s     simulation time since the run's first recorded frame (alias t)
    ego.speed       km/h  the ego's speed
    ego.accel       m/s²  its longitudinal acceleration over the last step (braking: negative); 0 at the first tick
    ego.junction_m  m     from the ego's front along its lane to the next junction (look-ahead 150 m; none: inf)
    ego.light_m     m     from its front along its lane to the stop line of the next traffic light on it (150 m)
    ego.light       text  that light's state: red, yellow, green, off, unknown or none
    visibility      m     the fog's meteorological visibility, 7000 m / fog_density (core.weather; no fog: inf)
    fog rain wetness puddles cloud   %   CARLA's fog_density, precipitation, wetness, precipitation_deposits,
                                         cloudiness (the weather in force, changed by a weather action)
    sun             °     the sun's altitude
    tod             text  day, low sun, twilight or night (core.runindex.tod_of: the run index's own names)
    actor.nearest_m m     box to box to the nearest vehicle, walker or scenario prop (core.criticality.box_distance;
                          none within 80 m: inf); vehicle.nearest_m, walker.nearest_m by kind
    actor.ttc       s     the smallest time to collision with any of them within 80 m (core.criticality.ttc, boxes at
                          constant velocity, horizon 10 s; none on a collision course: inf)
    actor.count           vehicles, walkers and props within 50 m (centre to centre)
    fired.<rule id>       how many times that rule has fired so far in the run

A signal that cannot be read (no traffic light state, no snapshot) makes its comparison false.

Actions (ACTIONS), several separated by ";":

    spawn <what> [key=value …]      a scenario actor of a kind the director already runs (SPAWNS): jaywalker, child,
                                    cut_in, autorickshaw_cut_in, cattle, handcart, stalled, wrong_way, or any scenario
                                    kind (core.scenarios.KINDS) with lane= ahead= lateral= heading= speed= start=
    weather [Preset] [field=value …]   change the weather (a CARLA preset, then fields; core.weather.FIELDS)
    fault <sensor id> <kind> [key=value …] [for=s]   start a sensor fault now (core.faults.TIMED: a timed fault of
                                    known size); for= its length (default: to the end of the run)
    marker [text]                   a marker in the run (frames.jsonl and run.json), nothing else

Determinism. Rules read only simulation state and simulation time, and every random choice an action makes (a
walker's blueprint) draws from a generator seeded by the run's seed and the rule's id, separate from the run's own:
a run with rules repeats as its run without rules repeats (scenario runs bit for bit; `falcon reproduce` re-runs
drive.json, which carries the rules). An actor a rule spawns is driven by the scenario director (kinematic by
default) — in a drive without a scenario the runner attaches one that drives only those actors.

Records. A firing is written on the frame whose state made its condition true (frames.jsonl "triggers"); its
actions act from the next tick. run.json "trigger_rules" lists the rules, every firing with its step, time, the
signals its condition read and what each action did (an action that could not be carried out — no adjacent lane
for a cut-in, a sensor the rig lacks — says why; the run goes on).
"""
from __future__ import annotations

import math
import re
import shlex

RULES_SCHEMA = 'falcon.trigger_rules/1'
REPEATS = ('once', 'every', 'cooldown')
CONTEXTS = ('drives', 'scenarios', 'live')
LOOKAHEAD_M = 150.0
ACTOR_RANGE_M = 80.0
COUNT_RANGE_M = 50.0
TTC_HORIZON_S = 10.0

# name: (unit, kind, about); unit '' for counts, 'text' signals compare with ':' / '=' / '!='
SIGNALS = {
    'time': ('s', 'num', 'simulation time since the run\'s first recorded frame'),
    'ego.speed': ('km/h', 'num', 'the ego\'s speed'),
    'ego.accel': ('m/s²', 'num', 'the ego\'s longitudinal acceleration over the last step (braking: negative)'),
    'ego.junction_m': ('m', 'num', f'from the ego\'s front along its lane to the next junction (look-ahead '
                                   f'{LOOKAHEAD_M:g} m)'),
    'ego.light_m': ('m', 'num', 'from the ego\'s front along its lane to the stop line of the next traffic light'),
    'ego.light': ('text', 'text', 'that light\'s state: red, yellow, green, off, unknown or none'),
    'visibility': ('m', 'num', 'the fog\'s meteorological visibility, 7000 m / fog_density (no fog: inf)'),
    'fog': ('%', 'num', 'CARLA\'s fog_density'),
    'rain': ('%', 'num', 'CARLA\'s precipitation'),
    'wetness': ('%', 'num', 'CARLA\'s wetness'),
    'puddles': ('%', 'num', 'CARLA\'s precipitation_deposits'),
    'cloud': ('%', 'num', 'CARLA\'s cloudiness'),
    'sun': ('°', 'num', 'the sun\'s altitude'),
    'tod': ('text', 'text', 'time of day from the sun: day, low sun, twilight or night'),
    'actor.nearest_m': ('m', 'num', f'box to box to the nearest vehicle, walker or prop within {ACTOR_RANGE_M:g} m'),
    'vehicle.nearest_m': ('m', 'num', 'box to box to the nearest vehicle'),
    'walker.nearest_m': ('m', 'num', 'box to box to the nearest walker'),
    'actor.ttc': ('s', 'num', 'the smallest time to collision with any actor (constant velocity, horizon 10 s)'),
    'actor.count': ('', 'num', f'vehicles, walkers and props within {COUNT_RANGE_M:g} m'),
}
ALIASES = {'t': 'time', 'world.visibility': 'visibility', 'world.fog': 'fog', 'world.rain': 'rain',
           'world.wetness': 'wetness', 'world.sun': 'sun', 'world.tod': 'tod', 'ttc': 'actor.ttc',
           'speed': 'ego.speed', 'accel': 'ego.accel'}
UNITS = {'km/h': ('km/h', 'kmh', 'kph'), 'm': ('m',), 's': ('s',), 'm/s²': ('m/s²', 'm/s2', 'mps2'),
         '%': ('%',), '°': ('°', 'deg'), '': ()}
TEXT_VALUES = {'ego.light': ('red', 'yellow', 'green', 'off', 'unknown', 'none'),
               'tod': ('day', 'low sun', 'twilight', 'night')}
ALL_UNITS = {u for us in UNITS.values() for u in us}
OPS = ('>=', '<=', '==', '!=', '>', '<', '=', ':')

# spawn presets: the scenario actor they add (core.scenarios' actor document) and their arguments
# {arg: (default, lo, hi, about)}; '$ego_kmh' in a speed is the ego's speed at the firing
SPAWNS = {
    'jaywalker': {'kind': 'pedestrian', 'about': 'a pedestrian steps off the kerb-side sidewalk ahead and walks across',
                  'args': {'distance': (25.0, 3.0, 120.0, 'm ahead of the ego'),
                           'walk': (1.4, 0.3, 4.0, 'walking speed, m/s')}},
    'child': {'kind': 'child', 'about': 'a child runs across from the kerb-side sidewalk ahead',
              'args': {'distance': (20.0, 3.0, 120.0, 'm ahead of the ego'),
                       'walk': (2.0, 0.3, 4.0, 'running speed, m/s')}},
    'cut_in': {'kind': 'car', 'about': 'a faster car in the lane beside (overtaking side) cuts in ahead of the ego',
               'args': {'behind': (8.0, -20.0, 40.0, 'm its start behind the ego along the road'),
                        'faster': (15.0, 0.0, 60.0, 'km/h faster than the ego at the firing'),
                        'gap': (5.0, 0.5, 30.0, 'm: it cuts in when its rear is this far ahead of the ego\'s front'),
                        'duration': (1.5, 0.5, 6.0, 'lane-change duration, s')}},
    'autorickshaw_cut_in': {'kind': 'autorickshaw', 'about': 'an autorickshaw stand-in cuts in (within its class\'s '
                                                             'top speed and acceleration, docs/indian-assets.md)',
                            'args': {'behind': (4.0, -20.0, 40.0, 'm its start behind the ego'),
                                     'faster': (10.0, 0.0, 40.0, 'km/h faster than the ego'),
                                     'gap': (3.0, 0.5, 30.0, 'm ahead of the ego\'s front when it cuts in'),
                                     'duration': (1.2, 0.5, 6.0, 'lane-change duration, s')}},
    'cattle': {'kind': 'cattle', 'about': 'a cow (a proxy of its size) standing in the ego\'s lane ahead',
               'args': {'distance': (40.0, 5.0, 150.0, 'm ahead of the ego'),
                        'lateral': (0.0, -1.5, 1.5, 'm off the lane centre (positive: towards the kerb)')}},
    'handcart': {'kind': 'handcart', 'about': 'a handcart (a proxy) pushed at walking pace near the kerb ahead',
                 'args': {'distance': (40.0, 5.0, 150.0, 'm ahead of the ego'),
                          'speed': (4.0, 0.0, 8.0, 'km/h'),
                          'lateral': (0.6, -1.5, 1.5, 'm from the lane centre towards the kerb')}},
    'stalled': {'kind': 'car', 'about': 'a vehicle standing still in the ego\'s lane ahead',
                'args': {'distance': (60.0, 5.0, 150.0, 'm ahead of the ego')},
                'choices': {'kind': ('car', 'truck', 'van', 'bus', 'autorickshaw')}},
    'wrong_way': {'kind': 'motorcycle', 'about': 'a scooter comes against the traffic in the ego\'s lane, near the kerb',
                  'args': {'distance': (60.0, 10.0, 150.0, 'm ahead of the ego'),
                           'speed': (25.0, 5.0, 60.0, 'km/h'),
                           'lateral': (1.0, -1.5, 1.5, 'm from the lane centre towards the kerb')}},
}
GENERIC_ARGS = {'ahead': (30.0, -40.0, 150.0, 'm ahead of the ego along the lane (negative: behind)'),
                'lateral': (0.0, -3.0, 3.0, 'm off the lane centre'),
                'speed': (0.0, 0.0, 120.0, 'km/h')}
GENERIC_CHOICES = {'lane': ('ego', 'adjacent', 'line', 'opposite', 'kerb', 'sidewalk', 'far_sidewalk'),
                   'heading': ('lane', 'against', 'across'), 'start': ('moving', 'stopped')}
ACTIONS = {'spawn': 'add a scenario actor', 'weather': 'change the weather', 'fault': 'start a sensor fault',
           'marker': 'a marker in the run'}


class RuleError(ValueError):
    """A rule that cannot be parsed or is not valid: the message says where (the column, 1-based) and why."""


# ================================================================================================ the tokenizer
_TOK = re.compile(r'''\s*(?:
    (?P<arrow>->)
  | (?P<op>>=|<=|==|!=|>|<|=|:)
  | (?P<lp>\() | (?P<rp>\)) | (?P<semi>;)
  | (?P<str>"[^"]*"|'[^']*')
  | (?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?:km/h|kmh|kph|m/s²|m/s2|mps2|deg|°|%|m|s)?(?![A-Za-z0-9_]))
  | (?P<unit>km/h|m/s²|m/s2)
  | (?P<word>[A-Za-z_][A-Za-z0-9_.\-]*|°|%)
)''', re.VERBOSE)


def tokenize(text: str) -> list:
    """[(kind, text, column)]; kind: arrow, op, lp, rp, semi, str, num, word."""
    out, i = [], 0
    s = text or ''
    while i < len(s):
        if s[i:].strip() == '':
            break
        m = _TOK.match(s, i)
        if not m or m.end() == i:
            col = i + len(s[i:]) - len(s[i:].lstrip()) + 1
            raise RuleError(f'column {col}: cannot read {s[col - 1:col + 9]!r}')
        kind = m.lastgroup
        col = m.start(kind) + 1
        out.append((kind, m.group(kind), col))
        i = m.end()
    return out


def _split_num(text: str):
    """'40km/h' -> (40.0, 'km/h'); '30' -> (30.0, '')."""
    m = re.match(r'^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(.*)$', text)
    return float(m.group(1)), m.group(2)


# ================================================================================================ the condition
class _Parser:
    def __init__(self, toks):
        self.toks, self.k = toks, 0

    def peek(self):
        return self.toks[self.k] if self.k < len(self.toks) else (None, None, None)

    def take(self):
        t = self.peek()
        self.k += 1
        return t

    def col(self):
        t = self.peek()
        return t[2] if t[2] is not None else (self.toks[-1][2] + len(self.toks[-1][1]) if self.toks else 1)

    def expr(self):
        node = self.conj()
        while self.peek()[0] == 'word' and self.peek()[1].lower() == 'or':
            self.take()
            node = ['or', node, self.conj()]
        return node

    def conj(self):
        node = self.neg()
        while self.peek()[0] == 'word' and self.peek()[1].lower() == 'and':
            self.take()
            node = ['and', node, self.neg()]
        return node

    def neg(self):
        t = self.peek()
        if t[0] == 'word' and t[1].lower() == 'not':
            self.take()
            return ['not', self.neg()]
        return self.primary()

    def primary(self):
        t = self.peek()
        if t[0] == 'lp':
            self.take()
            node = self.expr()
            if self.peek()[0] != 'rp':
                raise RuleError(f'column {self.col()}: a ")" is missing (the "(" is at column {t[2]})')
            self.take()
            return node
        if t[0] != 'word':
            raise RuleError(f'column {self.col()}: expected a signal (e.g. ego.speed), found '
                            f'{t[1]!r}' if t[0] else f'column {self.col()}: the condition ends too soon')
        return self.comparison()

    def comparison(self):
        kind, name, col = self.take()
        sig = signal_name(name)
        if sig is None:
            raise RuleError(f'column {col}: unknown signal {name!r} (signals: {", ".join(sorted(SIGNALS))}, '
                            'fired.<rule id>)')
        op = self.peek()
        if op[0] != 'op':
            raise RuleError(f'column {self.col()}: a comparison (> < >= <= = != :) must follow {name!r}')
        self.take()
        o = '=' if op[1] == '==' else op[1]
        unit, skind = signal_unit(sig)
        v = self.peek()
        if v[0] is None:
            raise RuleError(f'column {self.col()}: a value must follow {name} {op[1]}')
        if skind == 'text':
            if o not in ('=', ':', '!='):
                raise RuleError(f'column {op[2]}: {name} is text; compare it with ":" or "=" or "!="')
            self.take()
            val = v[1].strip('"\'').lower()
            # 'low sun' may be written unquoted: tod:low sun
            if sig == 'tod' and val == 'low' and self.peek()[0] == 'word' and self.peek()[1].lower() == 'sun':
                self.take()
                val = 'low sun'
            allowed = TEXT_VALUES.get(sig)
            if allowed and val not in allowed:
                raise RuleError(f'column {v[2]}: {name} is one of {", ".join(allowed)}, not {val!r}')
            return ['cmp', sig, '!=' if o == '!=' else '=', val]
        if v[0] == 'word' and v[1].lower() in ('inf', 'infinity'):
            self.take()
            return ['cmp', sig, o, math.inf]
        if v[0] != 'num':
            raise RuleError(f'column {v[2]}: {name} is a number{f" ({unit})" if unit else ""}, not {v[1]!r}')
        self.take()
        x, u = _split_num(v[1])
        nxt = self.peek()                         # a unit after a space: 40 km/h
        if not u and nxt[0] in ('unit', 'word') and nxt[1] in ALL_UNITS:
            u = nxt[1]
            self.take()
        if u and u not in UNITS.get(unit, ()):
            raise RuleError(f'column {v[2]}: {name} is in {unit or "a count (no unit)"}, not {u}')
        return ['cmp', sig, '=' if o == ':' else o, x]


def signal_name(name: str) -> str | None:
    """The canonical signal of a name (aliases resolved; fired.<id> kept), or None."""
    n = ALIASES.get(name, name)
    if n in SIGNALS:
        return n
    if n.startswith('fired.') and re.match(r'^fired\.[A-Za-z0-9][A-Za-z0-9_.-]*$', n):
        return n
    return None


def signal_unit(sig: str) -> tuple:
    if sig.startswith('fired.'):
        return '', 'num'
    return SIGNALS[sig][0], SIGNALS[sig][1]


def parse_condition(text: str) -> list:
    """A condition as a JSON-able tree: ['and'|'or', a, b] | ['not', a] | ['cmp', signal, op, value]."""
    toks = tokenize(text)
    if not toks:
        raise RuleError('column 1: the condition is empty')
    p = _Parser(toks)
    node = p.expr()
    if p.peek()[0] is not None:
        t = p.peek()
        raise RuleError(f'column {t[2]}: unexpected {t[1]!r} (join comparisons with "and" or "or")')
    return node


def signals_of(node) -> list:
    """The signals a condition reads, sorted."""
    out = set()

    def walk(n):
        if n[0] == 'cmp':
            out.add(n[1])
        else:
            for c in n[1:]:
                walk(c)
    walk(node)
    return sorted(out)


def _cmp(v, op, x) -> bool:
    if v is None:
        return False
    if isinstance(x, str):
        eq = str(v).lower() == x
        return not eq if op == '!=' else eq
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    if op == '>':
        return v > x
    if op == '<':
        return v < x
    if op == '>=':
        return v >= x
    if op == '<=':
        return v <= x
    if op == '!=':
        return abs(v - x) > 1e-9 if math.isfinite(v) or math.isfinite(x) else v != x
    return abs(v - x) <= 1e-9 if math.isfinite(v) and math.isfinite(x) else v == x


def evaluate(node, get) -> bool:
    """Evaluate a condition tree; get(signal) returns its value (None: not readable — the comparison is false)."""
    k = node[0]
    if k == 'and':
        return evaluate(node[1], get) and evaluate(node[2], get)
    if k == 'or':
        return evaluate(node[1], get) or evaluate(node[2], get)
    if k == 'not':
        return not evaluate(node[1], get)
    return _cmp(get(node[1]), node[2], node[3])


def describe_condition(node) -> str:
    k = node[0]
    if k in ('and', 'or'):
        def side(n):
            s = describe_condition(n)
            return f'({s})' if n[0] in ('and', 'or') and n[0] != k else s
        return f'{side(node[1])} {k} {side(node[2])}'
    if k == 'not':
        return f'not ({describe_condition(node[1])})'
    v = node[3]
    if isinstance(v, str):
        return f'{node[1]}:{v}' if node[2] == '=' else f'{node[1]} != {v}'
    vs = 'inf' if v == math.inf else f'{v:g}'
    return f'{node[1]} {node[2]} {vs}'


# ================================================================================================ the actions
def _args(words: list, col: int):
    """Positional words and key=value pairs of an action (values: numbers or words)."""
    pos, kv = [], {}
    for w in words:
        if '=' in w and not w.startswith('='):
            k, v = w.split('=', 1)
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
                raise RuleError(f'column {col}: {w!r} is not key=value')
            if k in kv:
                raise RuleError(f'column {col}: {k} given twice')
            try:
                kv[k] = float(v)
            except ValueError:
                kv[k] = v
        else:
            if kv:
                raise RuleError(f'column {col}: {w!r} after key=value arguments (positional ones come first)')
            pos.append(w)
    return pos, kv


def _num_args(kv: dict, spec: dict, col: int, what: str) -> dict:
    out = {}
    for k, v in kv.items():
        if k not in spec:
            raise RuleError(f'column {col}: {what} takes {", ".join(f"{a}=" for a in spec) or "no arguments"}, '
                            f'not {k}=')
        d, lo, hi, about = spec[k]
        if not isinstance(v, float):
            raise RuleError(f'column {col}: {what} {k}= is a number ({about}), not {v!r}')
        if not lo <= v <= hi:
            raise RuleError(f'column {col}: {what} {k}={v:g} is outside {lo:g}…{hi:g} ({about})')
        out[k] = v
    return out


def parse_action(text: str, col: int = 1) -> dict:
    """One action as a dict: {'do': 'spawn', 'what', 'args'} | {'do': 'weather', 'preset', 'fields'} |
    {'do': 'fault', 'sensor', 'fault': {kind, …}, 'for_s'} | {'do': 'marker', 'text'}."""
    try:
        words = shlex.split(text)
    except ValueError as e:
        raise RuleError(f'column {col}: {e}') from None
    if not words:
        raise RuleError(f'column {col}: an action is empty (after "->" or ";")')
    verb, rest = words[0].lower(), words[1:]
    if verb == 'marker' or verb == 'log':
        return {'do': 'marker', 'text': ' '.join(rest)}
    if verb not in ACTIONS:
        raise RuleError(f'column {col}: unknown action {words[0]!r} (actions: {", ".join(ACTIONS)})')
    pos, kv = _args(rest, col)
    if verb == 'spawn':
        return _parse_spawn(pos, kv, col)
    if verb == 'weather':
        return _parse_weather(pos, kv, col)
    return _parse_fault(pos, kv, col)


def _parse_spawn(pos, kv, col) -> dict:
    from . import scenarios as SC
    if len(pos) != 1:
        raise RuleError(f'column {col}: spawn takes one thing to spawn ({", ".join(SPAWNS)} or a scenario kind: '
                        f'{", ".join(SC.KINDS)})')
    what = pos[0].lower().replace('-', '_')
    if what in SPAWNS:
        spec = SPAWNS[what]
        choices = spec.get('choices') or {}
        num = {k: v for k, v in kv.items() if k not in choices}
        args = _num_args(num, spec['args'], col, f'spawn {what}')
        for k, allowed in choices.items():
            if k in kv:
                if str(kv[k]) not in allowed:
                    raise RuleError(f'column {col}: spawn {what} {k}= is one of {", ".join(allowed)}')
                args[k] = str(kv[k])
        return {'do': 'spawn', 'what': what, 'args': args}
    if what in SC.KINDS:
        num = {k: v for k, v in kv.items() if k not in GENERIC_CHOICES}
        args = _num_args(num, GENERIC_ARGS, col, f'spawn {what}')
        for k, allowed in GENERIC_CHOICES.items():
            if k in kv:
                if str(kv[k]) not in allowed:
                    raise RuleError(f'column {col}: spawn {what} {k}= is one of {", ".join(allowed)}')
                args[k] = str(kv[k])
        return {'do': 'spawn', 'what': what, 'args': args}
    raise RuleError(f'column {col}: cannot spawn {pos[0]!r} (presets: {", ".join(SPAWNS)}; kinds: '
                    f'{", ".join(SC.KINDS)})')


def _parse_weather(pos, kv, col) -> dict:
    from . import weather as W
    preset = None
    if len(pos) > 1:
        raise RuleError(f'column {col}: weather takes at most one preset, then field=value')
    if pos:
        preset = next((p for p in W.PRESETS if p.lower() == pos[0].lower()), None)
        if preset is None:
            raise RuleError(f'column {col}: unknown weather preset {pos[0]!r} (e.g. ClearNoon, HardRainNoon, '
                            'WetCloudySunset)')
    fields = {}
    for k, v in kv.items():
        if k not in W.FIELDS:
            raise RuleError(f'column {col}: weather has no field {k!r} (fields: {", ".join(W.FIELDS)})')
        if not isinstance(v, float):
            raise RuleError(f'column {col}: weather {k}= is a number')
        lo, hi = (-90.0, 90.0) if k == 'sun_altitude_angle' else (0.0, 360.0) if k == 'sun_azimuth_angle' else \
            (0.0, 100.0) if k in ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
                                  'fog_density', 'wetness') else (-1e6, 1e6)
        if not lo <= v <= hi:
            raise RuleError(f'column {col}: weather {k}={v:g} is outside {lo:g}…{hi:g}')
        fields[k] = v
    if preset is None and not fields:
        raise RuleError(f'column {col}: weather needs a preset or field=value (e.g. weather fog_density=60)')
    return {'do': 'weather', 'preset': preset, 'fields': fields}


def _parse_fault(pos, kv, col) -> dict:
    from . import faults as F
    if len(pos) != 2:
        raise RuleError(f'column {col}: fault takes a sensor id and a fault kind (e.g. fault cam-front dirt '
                        'cover=0.3 for=5)')
    sensor, kind = pos
    if kind not in F.TIMED:
        raise RuleError(f'column {col}: {kind!r} cannot start mid-run (faults that can: {", ".join(F.TIMED)}; '
                        'mount, clock and motion-blur faults are set when the sensor is spawned)')
    for_s = kv.pop('for', None)
    if for_s is not None and (not isinstance(for_s, float) or not 0.05 <= for_s <= 3600):
        raise RuleError(f'column {col}: fault for= is a length in seconds, 0.05…3600')
    f = {'kind': kind}
    for k, v in kv.items():
        f[k] = v
    if kind == 'drop' and 'rate' not in f:
        f['rate'] = 1.0                                  # a sensor that drops out: every frame lost
    probs = F.validate([dict(f, onset_s=0.0)])
    if probs:
        raise RuleError(f'column {col}: fault {"; ".join(probs)}')
    return {'do': 'fault', 'sensor': sensor, 'fault': f, 'for_s': for_s}


def describe_action(a: dict) -> str:
    if a['do'] == 'spawn':
        return 'spawn ' + ' '.join([a['what']] + [f'{k}={v:g}' if isinstance(v, float) else f'{k}={v}'
                                                  for k, v in sorted(a['args'].items())])
    if a['do'] == 'weather':
        return 'weather ' + ' '.join(([a['preset']] if a['preset'] else []) +
                                     [f'{k}={v:g}' for k, v in sorted(a['fields'].items())])
    if a['do'] == 'fault':
        extra = [f'{k}={v:g}' if isinstance(v, float) else f'{k}={v}' for k, v in sorted(a['fault'].items())
                 if k != 'kind']
        return ' '.join(['fault', a['sensor'], a['fault']['kind']] + extra +
                        ([f'for={a["for_s"]:g}'] if a.get('for_s') else []))
    return f'marker {a["text"]}'.rstrip()


# ================================================================================================ rules
def _split_rule(text: str):
    """(condition text, its column, [(action text, column)]): 'when' … '->' …; …, quotes respected."""
    s = text or ''
    m = re.match(r'^\s*when\b', s, re.IGNORECASE)
    if not m:
        raise RuleError('column 1: a rule starts with "when" (when <condition> -> <action>)')
    i = m.end()
    arrow, q = -1, None
    k = i
    while k < len(s):
        c = s[k]
        if q:
            q = None if c == q else q
        elif c in '"\'':
            q = c
        elif s.startswith('->', k):
            arrow = k
            break
        k += 1
    if arrow < 0:
        raise RuleError(f'column {len(s) + 1}: "->" and an action are missing (when <condition> -> <action>)')
    cond, acts = s[i:arrow], []
    start, q = arrow + 2, None
    for k in range(arrow + 2, len(s) + 1):
        c = s[k] if k < len(s) else ';'
        if q:
            q = None if c == q else q
        elif c in '"\'':
            q = c
        elif c == ';':
            acts.append((s[start:k], start + 1 + len(s[start:k]) - len(s[start:k].lstrip())))
            start = k + 1
    return cond, i + 1, acts


def _shift(e: RuleError, by: int) -> RuleError:
    """The error with its columns counted from the rule's start instead of the condition's."""
    return RuleError(re.sub(r'column (\d+)', lambda m: f'column {int(m.group(1)) + by}', str(e)))


def parse_rule(text: str) -> dict:
    """{'when': condition tree, 'then': [actions], 'signals': [...]}; RuleError with the column on a problem."""
    cond, ccol, acts = _split_rule(text)
    try:
        node = parse_condition(cond)
    except RuleError as e:
        raise _shift(e, ccol - 1) from None
    then = []
    for a, col in acts:
        then.append(parse_action(a, col))
    if not then:
        raise RuleError(f'column {len(text) + 1}: no action after "->"')
    return {'when': node, 'then': then, 'signals': signals_of(node)}


def describe(rule: dict) -> str:
    p = parse_rule(rule['rule'])
    return f'when {describe_condition(p["when"])} -> ' + '; '.join(describe_action(a) for a in p['then'])


def new_rule(rule: str, ident: str = '', name: str = '', repeat: str = 'once', cooldown_s: float = 0.0,
             applies=CONTEXTS, enabled: bool = True) -> dict:
    from .project import new_id
    return {'id': ident or new_id('rule'), 'name': name or rule[:60], 'rule': rule, 'repeat': repeat,
            'cooldown_s': float(cooldown_s), 'enabled': bool(enabled), 'applies': list(applies)}


def validate_rule(r: dict, ids=(), sensors=None) -> list:
    """Problems with one rule document ([] when valid). ids: every rule id of the project (fired.<id> must name one);
    sensors: the rig's sensor ids, when known (a fault must name one)."""
    out = []
    rid = r.get('id', '?')
    try:
        p = parse_rule(r.get('rule', ''))
    except RuleError as e:
        return [f'rule {rid}: {e}']
    rep = r.get('repeat', 'once')
    if rep not in REPEATS:
        out.append(f'rule {rid}: repeat is one of {", ".join(REPEATS)}, not {rep!r}')
    cd = r.get('cooldown_s', 0.0)
    if not isinstance(cd, (int, float)) or cd < 0:
        out.append(f'rule {rid}: cooldown_s must be a number ≥ 0')
    elif rep == 'cooldown' and cd <= 0:
        out.append(f'rule {rid}: repeat "cooldown" needs cooldown_s > 0 (it fires at most once per cooldown_s '
                   'while the condition holds)')
    for c in r.get('applies', CONTEXTS):
        if c not in CONTEXTS:
            out.append(f'rule {rid}: applies to {", ".join(CONTEXTS)}, not {c!r}')
    for s in p['signals']:
        if s.startswith('fired.') and ids and s[6:] not in ids:
            out.append(f'rule {rid}: {s} names no rule of the project')
    if sensors is not None:
        for a in p['then']:
            if a['do'] == 'fault' and a['sensor'] not in sensors:
                out.append(f'rule {rid}: fault on {a["sensor"]!r}, which the rig does not have (sensors: '
                           f'{", ".join(sensors) or "none"})')
            elif a['do'] == 'fault':
                kind = a['fault']['kind']
                st = sensors[a['sensor']]
                from . import faults as F
                if st and not F.applies_to(kind, st):
                    out.append(f'rule {rid}: a {kind} fault cannot act on {a["sensor"]} (a {st})')
    return out


def validate_rules(rules: list, sensors=None) -> list:
    out = []
    ids = [r.get('id') for r in rules]
    for d in sorted({i for i in ids if ids.count(i) > 1}):
        out.append(f'rule id {d!r} used twice')
    for r in rules:
        out += validate_rule(r, ids, sensors)
    return out


def for_context(project_doc: dict, context: str, only=None) -> list:
    """The project's enabled rules that apply to a context (drives, scenarios, live); only: ids to keep (None: all)."""
    if context not in CONTEXTS:
        raise ValueError(f'context {context!r} is not one of {CONTEXTS}')
    out = []
    for r in project_doc.get('trigger_rules') or []:
        if not r.get('enabled', True) or context not in (r.get('applies') or CONTEXTS):
            continue
        if only is not None and r.get('id') not in only:
            continue
        out.append(r)
    return out


def attach(cfg: dict, project, context: str, only=None) -> dict:
    """Put the project's rules for this context into a run config (cfg['trigger_rules']), validated against the
    config's rig: a ProjectError says what is wrong. No rule: the config is left as it was (its hash unchanged)."""
    from .project import ProjectError
    rules = for_context(project.doc, context, only)
    if only is not None:
        missing = sorted(set(only) - {r.get('id') for r in project.doc.get('trigger_rules') or []})
        if missing:
            raise ProjectError(f'no trigger rule {", ".join(missing)} in the project')
    if not rules:
        return cfg
    sensors = {s['id']: s['type'] for s in (cfg.get('rig') or {}).get('sensors') or []}
    all_ids = [r.get('id') for r in project.doc.get('trigger_rules') or []]
    probs = []
    for r in rules:
        probs += validate_rule(r, all_ids, sensors)
    if probs:
        raise ProjectError('trigger rules: ' + '; '.join(probs))
    cfg['trigger_rules'] = {'schema': RULES_SCHEMA, 'context': context,
                            'rules': [{'id': r['id'], 'name': r.get('name', r['id']), 'rule': r['rule'],
                                       'repeat': r.get('repeat', 'once'),
                                       'cooldown_s': float(r.get('cooldown_s', 0.0))} for r in rules]}
    return cfg


# ================================================================================================ firing logic
class Engine:
    """The rules of a run and their firing state (process-neutral: the worker supplies the signals, core decides
    what fires). step(t, get) -> [(rule, parsed actions, {signal: value read})] for the rules that fire at time t."""

    def __init__(self, rules: list):
        self.rules = []
        for r in rules:
            p = parse_rule(r['rule'])
            self.rules.append({'doc': r, 'when': p['when'], 'then': p['then'], 'signals': p['signals'],
                               'repeat': r.get('repeat', 'once'), 'cooldown_s': float(r.get('cooldown_s', 0.0)),
                               'last': None, 'prev': False, 'count': 0})
        self.counts = {r['doc']['id']: 0 for r in self.rules}

    @property
    def signals(self) -> list:
        return sorted({s for r in self.rules for s in r['signals']})

    def step(self, t: float, get) -> list:
        def look(sig):
            if sig.startswith('fired.'):
                return self.counts.get(sig[6:], 0)
            return get(sig)
        fire = []
        for r in self.rules:
            seen = {}

            def rec(sig):
                v = look(sig)
                seen[sig] = v
                return v
            now = evaluate(r['when'], rec)
            go = False
            since = math.inf if r['last'] is None else t - r['last']
            if r['repeat'] == 'once':
                go = now and r['count'] == 0
            elif r['repeat'] == 'every':
                go = now and not r['prev'] and since >= r['cooldown_s'] - 1e-9
            else:
                go = now and since >= r['cooldown_s'] - 1e-9
            r['prev'] = now
            if go:
                r['last'] = t
                r['count'] += 1
                fire.append((r, seen))
        for r, _ in fire:                        # fired.<id> counts change after every rule of the tick was evaluated
            self.counts[r['doc']['id']] = r['count']
        return [(r['doc'], r['then'], seen) for r, seen in fire]


def jsonable(v):
    """A signal value for JSON: inf as the string 'inf' (JSON has no infinity), floats rounded."""
    if isinstance(v, float):
        if math.isinf(v):
            return 'inf' if v > 0 else '-inf'
        return round(v, 4)
    return v


def help_text() -> str:
    """The signals and actions, for the GUI and `falcon triggers signals`."""
    lines = ['Signals:']
    for k, (u, kind, about) in SIGNALS.items():
        lines.append(f'  {k:18s} {u or ("text" if kind == "text" else ""):5s} {about}')
    lines.append('  fired.<rule id>          how many times that rule has fired')
    lines.append('Actions:')
    for k, s in SPAWNS.items():
        args = ' '.join(f'{a}={d[0]:g}' for a, d in s['args'].items())
        lines.append(f'  spawn {k} {args}  — {s["about"]}')
    lines.append('  spawn <kind> lane= ahead= lateral= heading= speed= start=  — any scenario kind')
    lines.append('  weather [Preset] field=value …  — e.g. weather fog_density=60 wetness=80')
    lines.append('  fault <sensor> <kind> key=value … [for=s]  — e.g. fault cam-front dirt cover=0.3 for=5')
    lines.append('  marker [text]')
    return '\n'.join(lines)


# ================================================================================================ the CLI
def main_triggers(argv) -> int:
    """falcon triggers list|check|add|remove|signals PROJECT …"""
    import argparse
    import json
    import sys
    from .project import Edit, Project, ProjectError
    ap = argparse.ArgumentParser(prog='falcon triggers', description='live trigger rules (docs/scenarios.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('signals', help='the signals and actions a rule can use')
    for name in ('list', 'check'):
        p = sub.add_parser(name)
        p.add_argument('project')
        p.add_argument('--json', action='store_true')
    p = sub.add_parser('add')
    p.add_argument('project')
    p.add_argument('rule', help='"when <condition> -> <action>[; <action>]"')
    p.add_argument('--id', default='')
    p.add_argument('--name', default='')
    p.add_argument('--repeat', choices=REPEATS, default='once')
    p.add_argument('--cooldown', type=float, default=0.0)
    p.add_argument('--applies', default=','.join(CONTEXTS), help='drives,scenarios,live')
    p.add_argument('--disabled', action='store_true')
    p = sub.add_parser('remove')
    p.add_argument('project')
    p.add_argument('id')
    a = ap.parse_args(argv)
    if a.action == 'signals':
        print(help_text())
        return 0
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
    rules = proj.doc.get('trigger_rules') or []
    if a.action == 'list':
        if a.json:
            print(json.dumps(rules, indent=1))
        for r in rules:
            print(f'{r["id"]:16s} {"on " if r.get("enabled", True) else "off"} {r.get("repeat", "once"):8s} '
                  f'{",".join(r.get("applies") or CONTEXTS):22s} {r["rule"]}')
        return 0
    if a.action == 'check':
        rigs = proj.doc.get('rigs') or []
        sensors = {s['id']: s['type'] for rig in rigs for s in rig.get('sensors') or []}
        probs = validate_rules(rules, sensors)
        if a.json:
            print(json.dumps({'rules': len(rules), 'problems': probs}, indent=1))
        else:
            for p in probs:
                print(p)
            print(f'{len(rules)} rule(s), {len(probs)} problem(s)')
        return 0 if not probs else 2
    if a.action == 'add':
        r = new_rule(a.rule, a.id, a.name, a.repeat, a.cooldown, [x for x in a.applies.split(',') if x],
                     not a.disabled)
        probs = validate_rules(rules + [r])
        if probs:
            for p in probs:
                print(p, file=sys.stderr)
            return 2
        if 'trigger_rules' not in proj.doc:
            proj.apply(Edit('set', ['trigger_rules'], [], text='trigger rules'))
        proj.apply(Edit('insert', ['trigger_rules'], r, text=f'add trigger rule {r["id"]}'))
        proj.save()
        print(r['id'])
        return 0
    k = next((i for i, r in enumerate(rules) if r.get('id') == a.id), None)
    if k is None:
        print(f'failed: no trigger rule {a.id!r}', file=sys.stderr)
        return 2
    proj.apply(Edit('remove', ['trigger_rules'], index=k, text=f'remove trigger rule {a.id}'))
    proj.save()
    return 0
