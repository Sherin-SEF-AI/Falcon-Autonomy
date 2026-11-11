"""Formal tools (the formal group; 4.0, in progress): signal temporal logic and the safety case. In this build only
formal_stl_check (read): a formula parsed by core.stl against the traffic rules' signals (core.rules.SIGNALS) — its
signals, look-back and look-ahead horizon and normalised text, or the parse error with its column. The falsifier, the
robustness of runs and the living GSN safety case add formal_stl_compile, formal_stl_robustness, formal_falsify_stl,
formal_gsn_tree and formal_gsn_verify.

A module adds its tools' one-line descriptions to _compact.SHORT itself (MINE below; tools/__init__ explains why)."""
from __future__ import annotations

import math

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact

MINE = {
    'formal_stl_check': 'Parse an STL formula over the rule signals: its signals, horizon and normal form, or the '
                        'error.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)


def _secs(v: float):
    return None if math.isinf(v) else round(float(v), 3)


@tool('formal_stl_check', 'formal', 'read',
      'Check a signal temporal logic formula (core.stl\'s language: always[a,b], eventually[a,b], until, since, '
      'and/or/not/->, comparisons of expressions over signals) against the signals the traffic rules can read '
      '(core.rules.SIGNALS). Returns ok, the formula in normal form (full parentheses), the signals it reads with '
      'their units, and its horizon — how many seconds it looks back and ahead (null: unbounded); or ok false with '
      'the parse error and its column, or the unknown signal names. Reads nothing from runs; changes nothing.',
      C.obj({'formula': C.s('the formula, e.g. "always[0,10] (lead_gap_m >= 5)"', minLength=1, maxLength=2000),
             'consts': C.obj(description='named constants substituted into the formula (name → number)',
                             additionalProperties={'type': 'number'})}, required=['formula']))
def formal_stl_check(ctx, formula, consts=None):
    from ...core import stl as S
    from ...core.rules import SIGNALS
    try:
        node = S.parse(formula, consts or {})
    except S.STLError as e:
        return {'ok': False, 'error': C.clip(str(e), 300)}
    except (ValueError, TypeError) as e:          # a constant that is not a number
        raise ToolError(f'consts: {e}') from None
    used = sorted(S.names(node))
    unknown = [n for n in used if n not in SIGNALS]
    if unknown:
        return {'ok': False, 'error': f'unknown signal(s) {", ".join(unknown)}', 'unknown': unknown,
                'known': sorted(SIGNALS)}
    past, future = S.horizon(node)
    return {'ok': True, 'normalised': C.clip(node.text(), 2000),
            'signals': [{'name': n, 'unit': SIGNALS[n][0]} for n in used],
            'horizon_s': {'past': _secs(past), 'future': _secs(future)}}
