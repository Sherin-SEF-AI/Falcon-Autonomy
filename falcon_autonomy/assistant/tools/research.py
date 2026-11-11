"""Research tools: questions answered by recording variants of a design and measuring them — today, rig variants
(from Vantage, or any project rigs) recorded on the same scenarios and seeds and compared by what a detector finds
(`falcon rigs compare`, sim.rigstudy, docs/rig-compare.md). The study's numbers are read with analysis_outputs kind
rigstudy and an id (lab.result).
"""
from __future__ import annotations

from ..tool_api import ToolError, tool
from . import _common as C
from .jobs import _cli_preview, _fake, _model, _queue, _saved


def _rigs(ctx, rigs, model, seeds=3, dry=False):
    """`falcon rigs compare` as Rigs ▸ Compare by perception queues it (sim.rigstudy.compare_argv)."""
    from ...core import milsearch as MS
    from ...sim.rigstudy import compare_argv, study_name
    rs = [C.find(ctx, 'rigs', r, 'rig')[1] for r in rigs]
    if len({r['id'] for r in rs}) < 2:
        raise ToolError('two different rigs or more are needed')
    for r in rs:
        cams = [s for s in r['sensors'] if s.get('type', '').startswith('camera.')]
        bad = [s['name'] for s in cams if not {'instance', 'depth'} <= set(s.get('labels') or [])]
        if bad or not cams:
            raise ToolError(f'rig {r["name"]!r}: every camera must record the instance and depth labels '
                            f'({", ".join(bad) or "no camera"})')
    m = _model(ctx, model)
    why = MS.evaluable(m, C.workspace(ctx))
    if why:
        raise ToolError(f'the model cannot be evaluated: {why}')
    path = (C.project(ctx).path or '<the project file>') if dry else _saved(ctx)
    argv = compare_argv(path, [r['id'] for r in rs], m['id'], seeds=int(seeds), name=study_name(rs),
                        fake_carla=_fake(ctx))
    return 'rigstudy', f'Compare rigs by perception: {", ".join(r["name"] for r in rs)}', argv, {'memory_gb': 0.5}


@tool('research_rig_compare', 'research', 'act',
      'Record two or more rigs (Rigs ▸ Import / export ▸ Vantage variants makes one per variant) on the same '
      'surround-view scenarios and seeds in kinematic motion, run a registered detector on every camera, and compare: '
      'precision and recall per rig with 95 % intervals, recall by sector round the car, by clearance and near each '
      'Vantage stitch seam, and a paired test of each rig against the first (`falcon rigs compare`). Output '
      'rigstudies/<id>; read with analysis_outputs kind rigstudy and the id.',
      C.obj({'rigs': C.arr(C.s('rig id'), 'two or more rigs, the first the baseline', minItems=2, maxItems=8),
             'model': C.s('detector id'),
             'seeds': C.i('seeds per scenario (default 3)', minimum=1, maximum=20)}, ['rigs', 'model']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': f'{len(a.get("rigs") or [])} rigs × 2 scenarios × '
                                                          f'{int(a.get("seeds", 3))} seeds of 10 s each, then the '
                                                          'model on every camera frame'},
      preview=_cli_preview(_rigs))
def research_rig_compare(ctx, **a):
    out = _queue(ctx, _rigs(ctx, **a))
    out['rigstudy'] = out['command'].split('--name ', 1)[1].split(' ', 1)[0]
    return out
