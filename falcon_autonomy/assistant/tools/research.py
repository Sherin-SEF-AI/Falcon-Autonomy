"""Research tools: the closed-loop planner benchmark (core.planbench, sim.benchrun; Scenarios ▸ Planner benchmark) —
queue it, read its scores — and a real incident made a scenario (core.incident; Replay ▸ Make a scenario from this
event).

Privacy. A dashcam incident is derived from the user's footage: research_incident runs the extraction on this machine
on a reconstruction id (the reconstruction itself is never read into a result) and returns counts, durations and
shares only — never positions, tracks, times of day, places, file names or the clip's name. The incident's folder is
private (assistant.privacy: its run.json's worker "dashcam-incident", the scenarios' source "dashcam", inherited by
every drive and study of them), its default name a hash; the search over its variations is queued as a job whose
study is private too.

And hypothesis-driven experiments and what a paper needs (core.ablation, core.paper; Training ▸
Ablations, Reports ▸ Outputs for a paper; docs/experiments.md).

research_ablation registers an ablation from a spec (the plan — grid, seeds, the replicates a power analysis asks
for — is the approval card's preview), queues its run or re-run as `falcon experiment ablation run|rerun` (the drives
go through the job queue under the CARLA lock, as the page's), re-analyses it, or writes its paper outputs.
research_registry lists every experiment of the workspace with its hypothesis, factors, status and verdict, and with
an id reads one experiment's plan or results. research_paper writes LaTeX and CSV tables and PDF, PNG and SVG figures
of an experiment, an evaluation or a comparison into paper/, or an export's data card. Every path is resolved inside
the workspace and refused when private (assistant.privacy); a data card never names a run from the dashcam footage.
"""
from __future__ import annotations

import os

from ..tool_api import ToolError, tool
from . import _common as C
from .analysis import compact
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


# ================================================================================ ablations, the registry and papers
ACTIONS = ['create', 'run', 'analyse', 'rerun', 'export']
PAPER_FOLDERS = {'experiment': ('experiments', ('results.json',)), 'evaluation': ('evaluations', ('evaluation.json',)),
                 'comparison': ('comparisons', ('comparison.json',)), 'datacard': ('exports', ('falcon_export.json',))}


def _ablation(ctx, name, action='create', spec=None, dry=False):
    """(preview text, result) of one research_ablation call."""
    from ...core import ablation as AB
    from ...core import paper as PP
    if C._bad_name(name) or '/' in name:
        raise ToolError(f'{name!r} is not an experiment name')
    ws = C.workspace(ctx)
    try:
        if action == 'create':
            if not spec:
                raise ToolError('create needs spec: hypothesis, metric, factors, design (docs/experiments.md)')
            s = AB.normalise(dict(spec, id=name), C.project(ctx))
            text = f'register ablation {s["id"]} ("{s["hypothesis"]["text"]}"): ' + AB.describe_plan(AB.plan(s))
            if dry:
                return text, None
            c = AB.create(ws, dict(spec, id=name), C.project(ctx))
            return text, {'experiment': c['id'], 'counts': c['plan']['counts'], 'power': C.rounded(
                {k: c['plan']['power'][k] for k in ('dz', 'alpha_per_contrast', 'replicates', 'capped')}, 4),
                'contrasts': [k['label'] for k in c['plan']['contrasts']],
                'follow': 'research_ablation action run, then jobs_wait, then research_registry with the id'}
        exp, s, pl = AB.load(ws, name)
        if action in ('run', 'rerun'):
            from ...core.jobs import FINAL
            path = C.project(ctx).path if dry else _saved(ctx)
            argv = ['experiment', 'ablation', action, exp.id, '--project', path or '<the project file>']
            if _fake(ctx) and s['run']['executor'] == 'detection':
                argv += ['--fake-carla', _fake(ctx)]
            c = pl['counts']
            text = C.command_line(argv) + f'  ({c["runs"]} runs' + (
                f', {c["drives"]} drives of {s["run"]["seconds"]:g} s' if c.get('drives') else '') + ')'
            if dry:
                return text, None
            if any(j.spec.kind == 'experiment' and j.spec.title.startswith(f'Ablation {exp.id}:')
                   and j.state not in FINAL for j in C.need_queue(ctx).jobs):
                raise ToolError(f'a job for {exp.id} is running already')
            out = C.cli_job(ctx, 'experiment', f'Ablation {exp.id}: {action}', argv, memory_gb=1.5, resumable=True)
            out['experiment'] = exp.id
            return text, out
        if action == 'analyse':
            if dry:
                return f'analyse {exp.id}: paired bootstrap of each contrast, Holm\'s correction → results.json', None
            res = AB.analyse(ws, exp.id)
            return '', {'experiment': exp.id, 'verdict': res['verdict'], 'warnings': res['warnings'],
                        'contrasts': [C.rounded({k: r[k] for k in ('label', 'n_pairs', 'diff', 'ci_low', 'ci_high',
                                                                    'dz', 'p_adjusted', 'verdict')}, 4)
                                      for r in res['contrasts'][:12]]}
        if dry:
            return f'write LaTeX and CSV tables and PDF, PNG and SVG figures of {exp.id} into paper/{exp.id}/', None
        doc = PP.export('experiment', exp.path, os.path.join(ws, 'paper', exp.id))
        return '', {'experiment': exp.id, 'folder': f'paper/{exp.id}', 'files': sorted(doc['files'])}
    except AB.AblationError as e:
        raise ToolError(str(e)) from e
    except PP.PaperError as e:
        raise ToolError(str(e)) from e


@tool('research_ablation', 'research', 'act',
      'An ablation (Training ▸ Ablations, `falcon experiment ablation`): a stated hypothesis over factors (model, '
      'weather, world, rig, traffic). create registers it from spec (hypothesis with direction, metric, factors with '
      'levels, contrasts, design with the effect and the sd of paired differences, run with the executor) and plans '
      'the grid, the seeds and the replicates by a power analysis — the approval card shows the plan; run and rerun '
      'queue a job (resumable; rerun re-runs it with the same seeds and checks it against the original, listing pins '
      'that changed); analyse recomputes the paired bootstrap contrasts with Holm\'s correction and the verdict; '
      'export writes LaTeX, CSV and figures into paper/<id>/. Read results with research_registry.',
      C.obj({'action': C.s('create (default), run, analyse, rerun or export', enum=ACTIONS),
             'name': C.s('the experiment id (folder in experiments/)'),
             'spec': {'type': 'object', 'description': 'create: hypothesis, metric, factors, contrasts, design, run '
                                                       '(docs/experiments.md)'}}, ['name']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'run and rerun: the plan\'s drives and evaluations (its '
                                                          'counts); the job reports its ETA once running'},
      preview=lambda ctx, **a: _ablation(ctx, dry=True, **a)[0])
def research_ablation(ctx, **a):
    return _ablation(ctx, **a)[1]


@tool('research_registry', 'research', 'read',
      'The experiment registry (core.ablation.registry): every experiment of the workspace — ablations, mixed '
      'sim/real, active learning, translation — newest first with its kind, status (planned, running, failed, '
      'cancelled, complete, analysed) and, for an ablation, its hypothesis, factors, metric, verdict and the '
      'experiment it re-runs. With id: an ablation\'s results (verdict, each contrast\'s difference with its interval, dz, '
      'adjusted p, verdict and notes, the cells, the design, warnings) or, before its analysis, its plan and run '
      'counts; another kind\'s results.json, compacted.',
      C.obj({'id': C.s('an experiment id: its results'),
             'limit': C.i('rows (default 20)', minimum=1, maximum=100)}))
def research_registry(ctx, id='', limit=20):
    from ...core import ablation as AB
    ws = C.workspace(ctx)
    if not id:
        rows = [{k: r.get(k) for k in ('id', 'kind', 'status', 'hypothesis', 'factors', 'metric', 'verdict',
                                         'rerun_of', 'created') if r.get(k) is not None} for r in AB.registry(ws)]
        return C.capped(rows, int(limit))
    d = C.resolve_in(ctx, 'experiments', id, 'experiment', ('experiment.json',))
    res = C.read_json(os.path.join(d, 'results.json')) or {}
    if not res:
        row = AB.summary(d)
        out = {k: row.get(k) for k in ('id', 'kind', 'status', 'runs', 'planned', 'hypothesis', 'factors')}
        pl = C.read_json(os.path.join(d, 'plan.json')) or {}
        if pl:
            out['power'] = C.rounded({k: (pl.get('power') or {}).get(k) for k in ('dz', 'alpha_per_contrast',
                                                                                  'replicates', 'capped')}, 4)
            out['contrasts'] = [k['label'] for k in pl.get('contrasts') or ()]
        out['note'] = 'not analysed yet: the analysis is written when every run has finished'
        return out
    res = {k: v for k, v in res.items() if k not in ('provenance', 'schema', 'key', 'method')}
    return compact(res, cap=12)


def _paper(ctx, kind, source, dry=False):
    from ...core import paper as PP
    folder, must = PAPER_FOLDERS[kind]
    src = C.resolve_in(ctx, folder, source, kind, must)
    out = src if kind == 'datacard' else os.path.join(C.workspace(ctx), 'paper', f'{kind}-{os.path.basename(src)}')
    rel = os.path.relpath(out, C.workspace(ctx))
    text = (f'write the data card of exports/{source} (DATACARD.md, datacard.json)' if kind == 'datacard' else
            f'write LaTeX and CSV tables and PDF, PNG and SVG figures of {folder}/{source} into {rel}/')
    if dry:
        return text, None
    try:
        doc = PP.export(kind, src, out)
    except PP.PaperError as e:
        raise ToolError(str(e)) from e
    return text, {'folder': rel, 'files': sorted(doc['files']), 'captions': {k: C.clip(v, 200) for k, v in
                                                                             (doc.get('captions') or {}).items()}}


@tool('research_paper', 'research', 'act',
      'Outputs ready for a paper (core.paper, `falcon paper`): an experiment (an ablation or a mixed sim/real one), '
      'a model evaluation or a comparison of two evaluations as LaTeX (booktabs) and CSV tables and PDF, PNG and SVG '
      'figures with their intervals, byte-identical for the same source, into paper/<kind>-<id>/; or an export\'s '
      'data card (composition, collection, labels, conditions, known gaps, licence, provenance) into the export. '
      'Returns the files and the captions.',
      C.obj({'kind': C.s('what the source is', enum=sorted(PAPER_FOLDERS)),
             'source': C.s('its id (folder in experiments/, evaluations/, comparisons/ or exports/)')},
            ['kind', 'source']),
      estimate=lambda ctx, **a: {'seconds': 5.0, 'what': 'tables and figures, a few seconds'},
      preview=lambda ctx, **a: _paper(ctx, dry=True, **a)[0])
def research_paper(ctx, **a):
    return _paper(ctx, **a)[1]
