"""Research tools: the closed-loop planner benchmark (core.planbench, sim.benchrun; Scenarios ▸ Planner benchmark) —
queue it, read its scores — and a real incident made a scenario (core.incident; Replay ▸ Make a scenario from this
event).

Privacy. A dashcam incident is derived from the user's footage: research_incident runs the extraction on this machine
on a reconstruction id (the reconstruction itself is never read into a result) and returns counts, durations and
shares only — never positions, tracks, times of day, places, file names or the clip's name. The incident's folder is
private (assistant.privacy: its run.json's worker "dashcam-incident", the scenarios' source "dashcam", inherited by
every drive and study of them), its default name a hash; the search over its variations is queued as a job whose
study is private too.
"""
from __future__ import annotations

import os

from ..tool_api import ToolError, tool
from . import _common as C
from .jobs import _fake, _saved

SETS = ['falcon-pdm-v1', 'falcon-pdm-mini']


# ================================================================================================= the benchmark
def _planner(ctx, p):
    """follower, none, autopilot, policy:<policies/ id or folder in the workspace>, stack:<suite file>."""
    from ...core import planbench as PB
    if p in PB.PLANNERS_DRIVE:
        return p
    kind, _, arg = p.partition(':')
    if kind not in ('policy', 'stack') or not arg:
        raise ToolError(f'unknown planner {p!r}: follower, none, autopilot, policy:<id>, stack:<suite file>')
    if kind == 'policy' and '/' not in arg:
        return 'policy:' + C.resolve_in(ctx, 'policies', arg, 'policy', ('policy.json',))
    path = C.checked(ctx, arg if os.path.isabs(arg) else os.path.join(C.workspace(ctx), arg))
    if not os.path.exists(path):
        raise ToolError(f'{p}: not in the workspace')
    return f'{kind}:{path}'


def _bench(ctx, planners=('follower',), routes='falcon-pdm-v1', name='', dry=False):
    from ...core import planbench as PB
    from ...sim.benchrun import run_argv
    ps = [_planner(ctx, p) for p in (planners or ['follower'])]
    nm = name or f'bench-{routes}'
    try:
        PB.bench_dir(C.workspace(ctx), nm)
    except PB.BenchError as e:
        raise ToolError(str(e)) from None
    path = C.project(ctx).path if dry else _saved(ctx)
    argv = run_argv(path or '<the project file>', nm, ps, routes, None, _fake(ctx))
    return argv, nm, ps


def _missing(ctx, routes) -> list:
    from ...core import planbench as PB
    return [r['id'] for r in PB.routes(routes) if PB.resolve_world(C.project(ctx), r['world'])[0] is None]


def _bench_preview(ctx, **a):
    argv, _nm, _ps = _bench(ctx, dry=True, **a)
    miss = _missing(ctx, a.get('routes', 'falcon-pdm-v1'))
    return C.command_line(argv) + (f'\n{len(miss)} routes skipped: the project has no world for them' if miss else '')


@tool('research_bench_run', 'research', 'act',
      'The closed-loop planner benchmark (Scenarios ▸ Planner benchmark; docs/planner-benchmark.md): each planner — '
      'follower (Falcon\'s lane follower, also driven as the reference for progress), none (constant speed), autopilot, '
      'policy:<policy id>, stack:<a stack suite file> — drives a fixed route set (falcon-pdm-v1: 23 routes on central '
      'Bengaluru, Town10HD_Opt and Town04, pinned seeds, kinematic actors, no background traffic; falcon-pdm-mini: 4), '
      'scored with nuPlan/NAVSIM-style terms (no at-fault collisions, drivable area, driving direction, TTC within '
      'bound, ego progress, comfort) and the PDM score. Resumable: routes already driven in the named benchmark are not '
      'driven again, so a new planner joins its comparison. Read with research_bench_results.',
      C.obj({'planners': C.arr(C.s('planner'), 'follower, none, autopilot, policy:<id>, stack:<suite file>'),
             'routes': C.s('route set (default falcon-pdm-v1)', enum=SETS),
             'name': C.s('benchmark name (default bench-<set>)')}),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'one scenario run (12–16 s simulated) per planner and '
                                                          'route, the reference follower included'},
      preview=_bench_preview)
def research_bench_run(ctx, planners=('follower',), routes='falcon-pdm-v1', name=''):
    from ...core import planbench as PB
    argv, nm, ps = _bench(ctx, planners, routes, name)
    out = C.cli_job(ctx, 'bench', f'Planner benchmark {nm} ({", ".join(PB.planner_slug(p) for p in ps)})', argv,
                    memory_gb=0.5)
    out['benchmark'] = nm
    miss = _missing(ctx, routes)
    if miss:
        out['skipped_routes'] = len(miss)
        out['skipped_why'] = 'the project has no world for them (Worlds page: Town04, an imported Bengaluru)'
    return out


@tool('research_bench_results', 'research', 'read',
      'A planner benchmark\'s scores (benchmarks/<name>): per planner the mean over routes of every term and of the '
      'PDM score with 95 % bootstrap intervals, the routes it drove and how many were partial; the paired comparisons '
      'against the reference (B − A with intervals); failed runs. Without a name, the benchmarks in the workspace.',
      C.obj({'name': C.s('benchmark name (default: list them)')}))
def research_bench_results(ctx, name=''):
    from ...core import planbench as PB
    if not name:
        rows = [{'name': b['name'], 'runs': b['runs'], 'scored': b['results'] is not None}
                for b in PB.list_benchmarks(C.workspace(ctx))]
        return C.capped(rows, 20)
    d = C.resolve_in(ctx, 'benchmarks', name, 'benchmark', ('benchmark.json',))
    res = C.read_json(os.path.join(d, 'results.json'))
    if not res:
        try:
            res = PB.results(d, 500)
        except PB.BenchError as e:
            raise ToolError(str(e)) from None
    planners = {p: {'routes': v['n_routes'], 'partial_routes': v.get('partial_routes'),
                    **{k: [m.get('value'), m.get('ci_low'), m.get('ci_high')] for k, m in v['metrics'].items()}}
                for p, v in res['planners'].items()}
    comps = [{'b_minus_a': f'{c["b"]} − {c["a"]}', 'routes': c['routes'],
              **{k: [m.get('diff'), m.get('ci_low'), m.get('ci_high')] for k, m in c['metrics'].items()}}
             for c in res.get('comparisons') or ()]
    return C.rounded({'benchmark': name, 'route_set': res.get('route_set'), 'planners': planners,
                      'comparisons': comps, 'failed_runs': len(res.get('failed') or ()),
                      'format': '[value, ci_low, ci_high] (95 % bootstrap over routes)',
                      'terms': res.get('terms')}, 3)


# ================================================================================================= incidents
@tool('research_incident', 'research', 'act',
      'A recorded dashcam event made a scenario (docs/dashcam.md), on this machine: the reconstruction (falcon recon '
      'from-dashcam; its id in recon/) placed on the imported world\'s matched road, written as a replay scenario, a '
      'parametric scenario whose ranges are centred on the extracted values, and OpenSCENARIO. The footage is private: '
      'only counts, durations and shares come back, never positions, names or images. search > 0 also queues the '
      'rare-case search over the variations (its study is private too).',
      C.obj({'recon': C.s('reconstruction id (recon/<id>)'), 'world': C.s('the imported world the event is on'),
             'start_s': C.n('window start, s from the first posed frame (default all)', minimum=0),
             'end_s': C.n('window end, s (default all)', minimum=0),
             'search': C.i('rare-case search runs to queue (default 0: none)', minimum=0, maximum=2000)},
            ['recon', 'world']),
      estimate=lambda ctx, **a: {'seconds': 5.0, 'what': 'the extraction (seconds), and the search\'s runs if asked'},
      preview=lambda ctx, **a: (f'Make a scenario from the reconstructed event {a.get("recon")!r} on world '
                                f'{a.get("world")!r}, on this machine (private: counts come back, nothing else)'
                                + (f'; then queue a {int(a["search"])}-run rare-case search over its variations'
                                   if a.get('search') else '')))
def research_incident(ctx, recon, world, start_s=None, end_s=None, search=0):
    from ...core import incident as INC
    from ...core.dashcam import DashcamError
    ws = C.workspace(ctx)
    if C._bad_name(recon) or '/' in recon:
        raise ToolError(f'{recon!r} is not a reconstruction id')
    d = os.path.join(ws, 'recon', recon)
    if not os.path.isfile(os.path.join(d, 'posed', 'posed.json')):         # never read into a result: only checked
        raise ToolError(f'no reconstruction {recon!r} with posed frames in the workspace (recon/)')
    proj = C.project(ctx)
    if proj.item('worlds', world) is None:
        raise ToolError(f'no world {world!r} in the project: give the imported world the event is on')
    name = INC.default_name(d, start_s, end_s)
    try:
        inc = INC.extract(d, proj, world, start_s, end_s, workspace=ws)
        summ = INC.write(inc, INC.incident_dir(ws, name), name)
    except (INC.IncidentError, DashcamError, OSError, ValueError, KeyError) as e:
        raise ToolError(f'the event could not be made a scenario ({type(e).__name__}); the Replay page says why on '
                        f'this machine') from None
    out = {'incident': name, 'private': True, **{k: summ[k] for k in ('duration_s', 'actors', 'kinds',
                                                                       'matched_share', 'search_parameters')}}
    if not summ['actors']:
        out['note'] = ('other road users come from a detections file (falcon dashcam incident extract --detections); '
                       'without one the scenario has the ego only')
    if search:
        argv = INC.search_argv(_saved(ctx), ws, name, int(search), fake_carla=_fake(ctx))
        job = C.cli_job(ctx, 'study', f'Search the variations of {name} ({int(search)})', argv, memory_gb=0.5)
        out['search_job'] = job['job']
    return C.rounded(out, 3)
