"""Research tools: hypothesis-driven experiments and what a paper needs (core.ablation, core.paper; Training ▸
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
