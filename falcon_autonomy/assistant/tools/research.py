"""Research tools: the closed-loop planner benchmark (core.planbench, sim.benchrun; Scenarios ▸ Planner benchmark) —
queue it, read its scores — and a real incident made a scenario (core.incident; Replay ▸ Make a scenario from this
event).

Privacy. A dashcam incident is derived from the user's footage: research_incident runs the extraction on this machine
on a reconstruction id (the reconstruction itself is never read into a result) and returns counts, durations and
shares only — never positions, tracks, times of day, places, file names or the clip's name. The incident's folder is
private (assistant.privacy: its run.json's worker "dashcam-incident", the scenarios' source "dashcam", inherited by
every drive and study of them), its default name a hash; the search over its variations is queued as a job whose
study is private too.

A scenario from a description (core.scenario_text, core.scenario_check; Scenarios ▸ Describe a scenario): the agent
is the model that writes it — research_scenario_check gives the description's claims and Falcon's draft, validates the
agent's JSON against the scenario schema and the description (the repair rounds, at most five invalid attempts per
description in a session), and checks a run against the description; research_scenario_text adds a valid one and runs
it. Nothing recorded goes to the model but the run's claims table (statuses, measured numbers).

And hypothesis-driven experiments and what a paper needs (core.ablation, core.paper; Training ▸
Ablations, Reports ▸ Outputs for a paper; docs/experiments.md).

research_ablation registers an ablation from a spec (the plan — grid, seeds, the replicates a power analysis asks
for — is the approval card's preview), queues its run or re-run as `falcon experiment ablation run|rerun` (the drives
go through the job queue under the CARLA lock, as the page's), re-analyses it, or writes its paper outputs.
research_registry lists every experiment of the workspace with its hypothesis, factors, status and verdict, and with
an id reads one experiment's plan or results. research_paper writes LaTeX and CSV tables and PDF, PNG and SVG figures
of an experiment, an evaluation or a comparison into paper/, or an export's data card. Every path is resolved inside
the workspace and refused when private (assistant.privacy); a data card never names a run from the dashcam footage.

And questions answered by recording variants of a design and measuring them — today, rig variants (from Vantage, or
any project rigs) recorded on the same scenarios and seeds and compared by what a detector finds (`falcon rigs
compare`, sim.rigstudy, docs/rig-compare.md). The study's numbers are read with analysis_outputs kind rigstudy and
an id (lab.result).
"""
from __future__ import annotations

import os

from ..tool_api import ToolError, tool
from . import _common as C
from .analysis import compact
from .jobs import _cli_preview, _fake, _model, _queue, _saved

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


# ================================================================================ a scenario from a description
def _text_rounds(ctx, description: str, add: bool = False) -> int:
    """Invalid scenarios already checked for this description in this session (the repair rounds' bound)."""
    seen = ctx.__dict__.setdefault('_scenario_text_rounds', {})
    k = ' '.join(str(description).split()).lower()
    if add:
        seen[k] = seen.get(k, 0) + 1
    return seen.get(k, 0)


def _text_scenario(ctx, description, scenario, world=''):
    """(the scenario as the project keeps it, its validation) — ToolError when it cannot be used."""
    from ...core import scenario_check as CK
    from ...core import scenario_text as T
    from .jobs import _world_rig
    w, _r = _world_rig(ctx, world, '')
    hand = T.world_handedness(C.project(ctx), w['id'])
    if not isinstance(scenario, dict):
        raise ToolError('scenario: one JSON object (schemas/scenario.schema.json)')
    scn = T._strip(scenario)
    errs = T.problems(scn)
    claims = CK.extract(description)['claims']
    mm = [] if errs else T.mismatches(scn, claims, hand)
    return scn, errs, mm, claims, hand


@tool('research_scenario_check', 'research', 'read',
      'A scenario from a description in words (Scenarios ▸ Describe a scenario; docs/scenario-text.md). Give the '
      'description and: nothing — its checkable claims (actor kinds, manoeuvres such as a cut-in from the left, a '
      'speed breaker or junction within N m, light, weather, the ego\'s speed, the outcome), Falcon\'s draft scenario '
      'from them and the words it did not recognise; scenario — your JSON scenario validated against the scenario '
      'schema and the description (errors, mismatches): repair it and check again (at most 5 invalid attempts); run '
      '— a recorded run checked against the description: each claim met, not met or not checkable with the measured '
      'value.',
      C.obj({'description': C.s('the scenario in words'),
             'scenario': {'type': 'object', 'description': 'a scenario to validate (falcon.scenario/1)'},
             'run': C.s('a run id to check against the description'),
             'world': C.s('world id (its traffic handedness; default the first)')}, ['description']))
def research_scenario_check(ctx, description, scenario=None, run='', world=''):
    from ...core import scenario_check as CK
    from ...core import scenario_text as T
    if run:
        d = C.resolve_run(ctx, run)
        try:
            rep = CK.check_run(d, description)
        except CK.CheckError as e:
            raise ToolError(str(e)) from None
        return C.rounded({'run': C.run_id(ctx, d), 'verdict': rep['verdict'], 'counts': rep['counts'],
                          'claims': [{'claim': r['claim'], 'status': r['status'], 'value': C.clip(r['value'], 240),
                                      **({'why': r['why']} if r.get('why') else {})} for r in rep['claims']],
                          'unrecognised': rep['unrecognised']}, 3)
    if scenario is not None:
        scn, errs, mm, _claims, _h = _text_scenario(ctx, description, scenario, world)
        if errs:
            n = _text_rounds(ctx, description, add=True)
            if n > T.MAX_ROUNDS:
                return {'valid': False, 'errors': errs[:12], 'stop': f'{n} invalid attempts: stop and tell the user '
                                                                     'what the errors are'}
            return {'valid': False, 'errors': errs[:20], 'attempt': n, 'repair': 'fix these and check again'}
        return {'valid': True, 'mismatches': mm, 'plan': T.plan_notes(scn),
                'next': 'research_scenario_text with this scenario adds it and runs it' +
                        ('; or repair the mismatches first' if mm else '')}
    from .jobs import _world_rig
    w, _r = _world_rig(ctx, world, '')
    hand = T.world_handedness(C.project(ctx), w['id'])
    ex = CK.extract(description)
    dr, notes = T.draft(ex['claims'], hand, description)
    return {'claims': [{'claim': CK.claim_text(c), 'phrase': c['phrase']} for c in ex['claims']],
            'unrecognised': ex['unrecognised'], 'handedness': hand, 'draft': T.example_for(dr) if dr else None,
            'draft_notes': notes, 'format': 'schemas/scenario.schema.json; knowledge_manual "scenario format"'}


def _text_preview(ctx, description='', scenario=None, world='', rig='', seed=0, run_once=True):
    scn, errs, mm, _c, _h = _text_scenario(ctx, description, scenario, world)
    if errs:
        return 'not valid — nothing will be added:\n' + '\n'.join(errs[:8])
    acts = ', '.join(f'{a.get("kind")} ({(a.get("place") or {}).get("lane")})' for a in scn.get('actors') or ())
    then = f', then run it once (seed {int(seed)}) and check the run against the description' if run_once else ''
    return (f'add the scenario {scn.get("name")!r} to the project ({acts}; needs: '
            f'{", ".join(scn.get("needs") or ()) or "none"}){then}'
            + (f'\nmismatches: {"; ".join(mm[:4])}' if mm else ''))


@tool('research_scenario_text', 'research', 'act',
      'Add a scenario written from a description to the project (an undoable edit; its source keeps the description '
      'and its claims) and, with run_once (default), run it once on Falcon\'s server (Scenarios ▸ Run: the world '
      'reloaded, background traffic off). Refused when the scenario is not valid (validate it with '
      'research_scenario_check first). Then jobs_wait, and research_scenario_check with the run checks it against the description.',
      C.obj({'description': C.s('the scenario in words'),
             'scenario': {'type': 'object', 'description': 'the scenario (falcon.scenario/1), validated'},
             'world': C.s('world id'), 'rig': C.s('rig id'), 'seed': C.i('seed (default 0)', minimum=0),
             'run_once': C.b('run it once after adding it (default true)')}, ['description', 'scenario']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'one scenario run (its duration_s of simulated time)'},
      preview=_text_preview)
def research_scenario_text(ctx, description, scenario, world='', rig='', seed=0, run_once=True):
    from ...core import scenario_text as T
    from ...core.drive import run_dir_name, scenario_config
    from ...core.project import Edit, ProjectError
    from .jobs import _world_rig
    scn, errs, mm, claims, _h = _text_scenario(ctx, description, scenario, world)
    if errs:
        raise ToolError('not valid: ' + '; '.join(errs[:8]))
    scn = T.finalise(scn, description, claims, 'assistant', _text_rounds(ctx, description))
    out = {'scenario': scn['id'], 'name': scn['name'], 'mismatches': mm}
    if C.project(ctx).item('scenarios', scn['id']) is None:
        out.update(C.apply_edit(ctx, Edit('insert', ['scenarios'], scn,
                                          text=f'Add the scenario {scn["name"]} (from a description)')))
    if not run_once:
        return out
    w, r = _world_rig(ctx, world, rig)
    svc = C.need_service(ctx)
    d = os.path.join(C.workspace(ctx), 'runs', run_dir_name(f'scenario-{scn["id"]}'))
    try:
        cfg = scenario_config(C.project(ctx), scn, w['id'], r['id'], d, svc.port, svc.tm_port, seed=int(seed),
                              render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                              workspace=C.workspace(ctx), trigger_rules='project')
        job = svc.submit_drive(cfg, title=f'{scn["name"]} (seed {int(seed)})')
    except (ProjectError, RuntimeError, ValueError, OSError) as e:
        raise ToolError(f'added, but not run: {e}') from None
    ctx.emit('job_queued', {'job': job.id, 'title': job.spec.title})
    out.update(job=job.id, run=C.run_id(ctx, d), follow='jobs_wait, then research_scenario_check with run')
    return out


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


# ================================================================================ rig variants compared by perception
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


# ================================================================================================ generation rounds
def _generate(ctx, name, spec=None, rounds=0, dry=False):
    """(preview text, result) of one research_generate call (core.genround)."""
    from ...core import genround as GR
    if C._bad_name(name) or '/' in name:
        raise ToolError(f'{name!r} is not an experiment name')
    ws, proj = C.workspace(ctx), C.project(ctx)
    try:
        if os.path.isfile(os.path.join(GR.exp_dir(ws, name), 'experiment.json')):
            _d, s, st = GR.load(ws, name)
            if spec:
                GR.create(ws, dict(spec, id=name), proj)          # refused unless it is the same definition
        elif spec:
            s, st = GR.normalise(dict(spec, id=name), proj), None
        else:
            raise ToolError(f'no generation experiment {name!r}: give spec to register it (docs/generation.md)')
        path = proj.path if dry else _saved(ctx)
        argv = ['generate', 'run', path or '<the project file>', '--id', name]
        if rounds:
            argv += ['--rounds', str(int(rounds))]
        if _fake(ctx):
            argv += ['--fake-carla', _fake(ctx)]
        text = ('' if st else 'register, then ') + C.command_line(argv) + '\n' + GR.describe_plan(s)
        if dry:
            return text, None
        if st is None:
            GR.create(ws, dict(spec, id=name), proj)
        out = C.cli_job(ctx, 'experiment', f'Generation {name}', argv, memory_gb=1.5, resumable=True)
        out['experiment'] = name
        out['follow'] = 'jobs_wait, then research_registry with the id (results: rounds, cells, verdicts)'
        return text, out
    except GR.GenError as e:
        raise ToolError(str(e)) from e


@tool('research_generate', 'research', 'act',
      'Generation rounds (Training ▸ Generation, `falcon generate`; docs/generation.md): active learning that '
      'generates data. Each round ranks class × condition cells by expected value — EV = exposure share × the AP '
      'shortfall on a held-out set × sqrt(n0/(n0 + training objects)) × (1 + the coverage grid\'s failure rate) — '
      'records drives and scenario runs for the best cells within a budget (labels from the ground truth), adds '
      'them to a dataset version, retrains the detector from its own settings and compares per cell before and '
      'after with paired bootstrap intervals on the held-out set fixed before round 1 (never trained on); stops '
      'when no chosen cell improves. spec registers it (id = name; model, world, rig, conditions weather/light, '
      'budget {drives, frames}, rounds); then the job runs or resumes rounds. Read with research_registry.',
      C.obj({'name': C.s('experiment id (experiments/<id>)'),
             'spec': {'type': 'object', 'description': 'to register: model, world, rig, conditions, budget, rounds, '
                                                       'acquisition (docs/generation.md)'},
             'rounds': C.i('at most this many more rounds (default: to the stop rule)', minimum=1, maximum=50)},
            ['name']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'per round: its drives (the budget), a training and an '
                                                          'evaluation; the job reports its ETA once running'},
      preview=lambda ctx, **a: _generate(ctx, dry=True, **a)[0])
def research_generate(ctx, **a):
    return _generate(ctx, **a)[1]
