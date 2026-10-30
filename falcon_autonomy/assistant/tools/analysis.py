"""Analysis tools (read): what the analyses wrote — studies (failures ranked, RSS labels, the failure-region map),
criticality, root-cause answers, evaluations with their intervals, comparisons, the sim-to-real gap, stack suites and
safety cases, V2X benefit, traffic rules — read from their own files with the same functions the pages use
(core.search.failure_map, core.rss, core.runcompare.rule_metrics, core.stack.summarise, core.reports.sources). Numbers
are the files' own, rounded; nothing is recomputed differently from the page that shows it.
"""
from __future__ import annotations

import os

from ..tool_api import ToolError, tool
from . import _common as C

OUTPUTS = {'study': ('studies', ('study.json',)), 'evaluation': ('evaluations', ('evaluation.json',)),
           'comparison': ('comparisons', ('comparison.json',)), 'gap': ('gaps', ('gap.json',)),
           'suite': ('suites', ('results.jsonl', 'suite-request.json')), 'rootcause': ('rootcause',
                                                                                       ('rootcause.json',)),
           'translator': ('translators', ('',)), 'reconstruction': ('recon', ('',)),
           'training': ('training', ('',)), 'worldmodel': ('worldmodels', ('',)), 'policy': ('policies', ('',)),
           'export': ('exports', ('',)), 'report': ('reports', ('',)), 'v2x_benefit': ('v2x', ('',)),
           'derived': ('derived', ('',)), 'leaderboard': ('leaderboard', ('leaderboard.json',)),
           # the lab's jobs (tools/lab.py), whose numbers analysis_outputs reads with an id (lab.result)
           'ci': ('ci', ('ci-results.json',)), 'deploy': ('deploy', ('deploy.json',)),
           'adversary': ('adversary', ('adversary-eval.json', os.path.join('policy', 'policy.json'))),
           'experiment': ('experiments', ('experiment.json',)),
           'vlm': (os.path.join('vlm', 'evaluations'), ('scores.json', 'run.json')), 'log': ('logs', ('log.json',)),
           'hil': ('runs', ('hil.json',)), 'replay_comparison': ('runs', ('comparison.json',))}
LAB = ('ci', 'deploy', 'adversary', 'experiment', 'vlm', 'log', 'hil', 'replay_comparison')
READER = {'study': 'analysis_study', 'evaluation': 'analysis_evaluation', 'comparison': 'analysis_evaluation',
          'gap': 'analysis_gap', 'suite': 'analysis_suite', 'rootcause': 'analysis_rootcause',
          'v2x_benefit': 'analysis_v2x_benefit'}


def compact(v, depth=0, cap=10):
    """A result document cut down for the model: lists capped (with their length), deep blocks summarised, floats
    rounded, media left out."""
    v = C.no_media(v)
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        if depth >= 4:
            return f'{{{len(v)} keys}}'
        return {k: compact(x, depth + 1, cap) for k, x in list(v.items())[:40]}
    if isinstance(v, list):
        out = [compact(x, depth + 1, cap) for x in v[:cap]]
        if len(v) > cap:
            out.append(f'… {len(v) - cap} more')
        return out
    if isinstance(v, str):
        return C.clip(v, 400)
    return v


@tool('analysis_outputs', 'analysis', 'read',
      'The workspace\'s analysis outputs of one kind, newest first, by id (folder or file name): studies, '
      'evaluations, comparisons, gaps, suites, root-cause answers, translators, reconstructions, trainings, world '
      'models, policies, exports, reports, V2X benefit files, derived data, leaderboards, and the lab jobs\' outputs '
      '(CI runs, deployments, adversary trainings and evaluations, experiments, VLM evaluations, imported logs, HIL '
      'sessions, replay comparisons). Use the ids in the analysis and jobs tools. With id, a lab output\'s numbers: '
      'a CI run\'s verdict and gates, a HIL session\'s latency per stage and deadlines, a deployment\'s engines '
      '(latency, power, AP ratio, gate verdict), failure rates, an experiment\'s arms with intervals, VLM scores '
      '(text only), a log\'s actors and duration, a replay comparison\'s differences.',
      C.obj({'kind': C.s('what to list', enum=sorted(OUTPUTS)),
             'id': C.s('a lab output\'s id: its numbers (a run id for hil)'),
             'limit': C.i('rows (default 20)', minimum=1, maximum=50)}, ['kind']))
def analysis_outputs(ctx, kind, id='', limit=20):
    from ..privacy import private_reason
    if id:
        if kind not in LAB:
            raise ToolError(f'read a {kind} with {READER[kind]}' if kind in READER else
                            f'a {kind} id is used by the jobs tools; nothing more to read here')
        from .lab import result
        return result(ctx, kind, id)
    folder, must = OUTPUTS[kind]
    root = os.path.join(C.workspace(ctx), folder)
    rows, hidden = [], 0
    if os.path.isdir(root):
        names = sorted(os.listdir(root), key=lambda n: os.path.getmtime(os.path.join(root, n)), reverse=True)
        for n in names:
            p = os.path.join(root, n)
            if n.startswith('.') or (must != ('',) and not any(os.path.exists(os.path.join(p, m)) for m in must)):
                continue
            if private_reason(p):
                hidden += 1
                continue
            import time
            rows.append({'id': n, 'modified': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(os.path.getmtime(p))),
                         'kind': 'folder' if os.path.isdir(p) else 'file'})
    out = C.capped(rows, int(limit))
    if hidden:
        out['private_left_out'] = hidden
    return out


def _study_rows(ctx, study):
    d = C.resolve_in(ctx, 'studies', study, 'study', ('study.json',))
    return d, C.read_json(os.path.join(d, 'study.json'), {}), C.read_jsonl(os.path.join(d, 'results.jsonl'))


def _hist(rows):
    return [{'params': r.get('values') or {}, 'value': r.get('objective', 0.0) or 0.0, 'i': k, 'run': r.get('run'),
             'criticality': r.get('criticality', r.get('objective')),
             'perception': (r.get('perception') or {}).get('value'), 'rss': r.get('rss')} for k, r in enumerate(rows)]


@tool('analysis_study', 'analysis', 'read',
      'A sweep or rare-case search (studies/<id>), as Rare cases shows it: its scenario, objective, search method '
      'and space; how many runs; the failures counted by the objective (≥ threshold) ranked, then those reaching the '
      'threshold in criticality that the objective does not count (their RSS label is not avoidable), each with its '
      'row index (for jobs_replay_failure), objective, criticality, perception score, RSS label and reason, '
      'parameter values and run id; RSS label counts.',
      C.obj({'study': C.s('study id'), 'threshold': C.n('failure threshold (default the study\'s or 0.5)', minimum=0,
                                                         maximum=1),
             'limit': C.i('failures listed (default 15)', minimum=1, maximum=50)}, ['study']))
def analysis_study(ctx, study, threshold=None, limit=15):
    from ...core import search as S
    d, st, rows = _study_rows(ctx, study)
    srch = {}
    for f in sorted(os.listdir(d)):
        if f.startswith('search-') and f.endswith('.json'):
            srch = C.read_json(os.path.join(d, f), {}) or {}
    thr = float(threshold if threshold is not None else srch.get('threshold', 0.5) or 0.5)
    hist = _hist(rows)
    counted = S.failures(hist, thr)
    seen = {h['i'] for h in counted}
    uncounted = sorted((h for h in hist if h['i'] not in seen and (h['criticality'] or 0.0) >= thr),
                       key=lambda h: -(h['criticality'] or 0.0))

    def row(h, is_counted):
        rss = h.get('rss') or {}
        out = {'index': h['i'], 'objective': h['value'], 'criticality': h['criticality'],
               'perception': h['perception'], 'counted': is_counted, 'rss': rss.get('label'),
               'rss_reason': C.clip(rss.get('reason'), 160) if rss.get('reason') else None,
               'values': h['params'], 'run': C.run_id(ctx, h['run']) if h.get('run') and C.inside(
                   h['run'], C.workspace(ctx)) else None}
        return C.rounded({k: v for k, v in out.items() if v is not None}, 3)
    labels = {}
    for h in hist:
        lb = (h.get('rss') or {}).get('label') or 'unlabelled'
        labels[lb] = labels.get(lb, 0) + 1
    fails = [row(h, True) for h in counted] + [row(h, False) for h in uncounted]
    scn = st.get('scenario')
    return {'study': study, 'scenario': scn.get('id') if isinstance(scn, dict) else scn, 'world': st.get('world'),
            'rig': st.get('rig'),
            'objective': compact(st.get('objective')), 'feasibility': compact(st.get('feasibility')),
            'method': srch.get('method') or ('sweep' if study.startswith('sweep') else None),
            'budget': srch.get('budget'), 'runs': len(rows),
            'errors': sum(1 for r in rows if r.get('status') not in (None, 'ok')),
            'threshold': thr, 'failures_counted': len(counted), 'failures_not_counted': len(uncounted),
            'rss_labels': labels, 'failures': C.capped(fails, int(limit)),
            'space': [{'name': p.get('name'), 'lo': p.get('lo'), 'hi': p.get('hi'), 'choices': p.get('choices')}
                      for p in (st.get('space') or [])][:20]}


@tool('analysis_failure_map', 'analysis', 'read',
      'The failure-region map of a study over two of its parameters (Rare cases ▸ map, core.search.failure_map): '
      'bin edges and per cell the number of runs, the share failing (objective ≥ threshold) and the mean objective '
      '(None where no run fell). Parameter names: analysis_study\'s space.',
      C.obj({'study': C.s('study id'), 'x': C.s('parameter on the x axis'), 'y': C.s('parameter on the y axis'),
             'bins': C.i('bins per axis (default 8)', minimum=2, maximum=20),
             'threshold': C.n('failure threshold (default 0.5)', minimum=0, maximum=1)}, ['study', 'x', 'y']))
def analysis_failure_map(ctx, study, x, y, bins=8, threshold=0.5):
    from ...core import milsearch as MS
    from ...core import scenarios as SC
    from ...core import search as S
    from ...sim.studio import WEATHER_TAGS
    _d, st, rows = _study_rows(ctx, study)
    if not rows:
        raise ToolError('the study has no results yet')
    sk = st.get('scenario')
    scn = sk if isinstance(sk, dict) else SC.builtin(sk) if sk in SC.LIBRARY else \
        C.find(ctx, 'scenarios', sk, 'scenario')[1]
    keys = set().union(*[set(r.get('values') or {}) for r in rows])
    space = SC.search_space(scn) + MS.weather_space([k for k in WEATHER_TAGS if k in keys])
    names = [p['name'] for p in space]
    for a in (x, y):
        if a not in names:
            raise ToolError(f'{a!r} is not a searched parameter ({", ".join(names)})')
    if x == y:
        raise ToolError('choose two different parameters')
    g = S.failure_map(_hist(rows), space, x, y, bins=int(bins), threshold=float(threshold))
    keep = ('x_name', 'y_name', 'x_edges', 'y_edges', 'x_labels', 'y_labels', 'count', 'failure_share', 'mean', 'max',
            'threshold')
    return C.rounded({k: g.get(k) for k in keep if k in g}, 3)


@tool('analysis_criticality', 'analysis', 'read',
      'A scenario run\'s criticality as recorded (core.criticality): per actor the minimum TTC and when, PET, minimum '
      'distance, maximum required deceleration, collision and when; overall the same and the critical actor; the '
      'scalar criticality and its formula; and the run\'s RSS label when it can be computed (core.rss: avoidable, '
      'other, unavoidable, safe — straight-road axis only).',
      C.obj({'run': C.s('a scenario run id')}, ['run']))
def analysis_criticality(ctx, run):
    from ...core import rss as RSS
    d = C.resolve_run(ctx, run)
    r = C.read_json(os.path.join(d, 'run.json'), {})
    crit = (r.get('scenario') or {}).get('criticality')
    if not crit:
        raise ToolError('the run recorded no criticality (it is not a scenario run)')
    out = {'run': C.run_id(ctx, d), 'criticality': compact(crit)}
    try:
        lab = RSS.label_run_dir(d)
    except (OSError, ValueError, KeyError, TypeError):
        lab = None
    if lab:
        out['rss'] = compact({k: lab.get(k) for k in ('label', 'reason', 'actor', 'event', 't_event', 'response',
                                                      'relation', 'ego_violations', 'actor_violations',
                                                      'stopping_envelope_m', 'counted')})
    return out


@tool('analysis_rootcause', 'analysis', 'read',
      'A "why did it fail?" answer (rootcause/<id>, core.rootcause): for a scenario case its status (attributed, '
      'no_factor, not_reproduced), summary and wording, and per factor whether it was present, whether removing it '
      'fixed the failure and whether it alone fails, with caveats; for an evaluation the correlates of misses with '
      'intervals; for an episode the input groups whose removal fixes it.',
      C.obj({'id': C.s('root-cause id (folder in rootcause/)')}, ['id']))
def analysis_rootcause(ctx, id):
    d = C.resolve_in(ctx, 'rootcause', id, 'root-cause answer')
    doc = C.read_json(os.path.join(d, 'rootcause.json'))
    if not doc:
        raise ToolError('no answer yet (the job may still be running)')
    keep = ('kind', 'status', 'summary', 'wording', 'case', 'control', 'factors', 'all_removed', 'n_runs',
            'evaluation', 'model', 'n_objects', 'n_missed', 'miss_rate', 'unit', 'n_units', 'note', 'ranking',
            'attributes', 'episode', 'policy', 'groups')
    out = {k: doc[k] for k in keep if k in doc}
    if isinstance(out.get('case'), dict):
        out['case'] = {k: v for k, v in out['case'].items() if k in ('scenario', 'values', 'failure', 'threshold',
                                                                     'actor', 'step', 'model')}
    return compact(out, cap=12)


@tool('analysis_evaluation', 'analysis', 'read',
      'A model evaluation (evaluations/<id>, core.harness): the model, drives, frames scored, method (bootstrap unit '
      'and resamples), and every score with its 95 % interval (AP, AP50, AP75, AR100, per class and per attribute '
      'bin; mIoU and per class for a segmenter), with the note on few drives; or a comparison (comparisons/<id>): '
      'each metric A, B, the change with its interval, regressions and improvements.',
      C.obj({'id': C.s('evaluation or comparison id'), 'kind': C.s('evaluation (default) or comparison',
                                                                    enum=['evaluation', 'comparison']),
             'metric': C.s('only metrics starting with this (e.g. AP.)')}, ['id']))
def analysis_evaluation(ctx, id, kind='evaluation', metric=''):
    if kind == 'comparison':
        d = C.resolve_in(ctx, 'comparisons', id, 'comparison', ('comparison.json',))
        doc = C.read_json(os.path.join(d, 'comparison.json'), {})
        rows = [r for r in doc.get('rows') or [] if str(r.get('metric', '')).startswith(metric)]
        return compact({'rows': C.capped(rows, 30), 'regressions': doc.get('regressions'),
                        'improvements': doc.get('improvements')}, cap=30)
    d = C.resolve_in(ctx, 'evaluations', id, 'evaluation', ('evaluation.json',))
    doc = C.read_json(os.path.join(d, 'evaluation.json'), {})
    sc = doc.get('scores') or {}
    table = [r for r in sc.get('table') or [] if str(r.get('metric', '')).startswith(metric)]
    m = doc.get('model') or {}
    return C.rounded({'id': id, 'model': {k: m.get(k) for k in ('id', 'name', 'kind', 'runtime')},
                      'runs': [C.run_id(ctx, r) if C.inside(r, C.workspace(ctx)) else os.path.basename(r)
                               for r in (doc.get('runs') or [])][:20],
                      'n_frames': doc.get('n_frames'), 'classes': doc.get('classes'), 'method': doc.get('method'),
                      'unit': sc.get('unit'), 'n_units': sc.get('n_units'), 'n_boot': sc.get('n_boot'),
                      'note': sc.get('note'), 'latency_ms': doc.get('latency_ms'),
                      'scores': C.capped(table, 40, 'filter with metric')}, 4)


@tool('analysis_gap', 'analysis', 'read',
      'A sim-to-real gap result (gaps/<id>, core.realgap): per appearance statistic the simulated and real medians, '
      'Wasserstein distance and shift; night statistics; object class shares; and the feature distances (FD, KD) '
      'with their real-vs-real floor (compare with the floor, not with published FID).',
      C.obj({'id': C.s('gap id (folder in gaps/)')}, ['id']))
def analysis_gap(ctx, id):
    d = C.resolve_in(ctx, 'gaps', id, 'gap result', ('gap.json',))
    doc = C.read_json(os.path.join(d, 'gap.json'), {})
    return compact({'gap': doc.get('gap'), 'features': doc.get('features'), 'matched': doc.get('matched'),
                    'real': {k: (doc.get('real') or {}).get(k) for k in ('kind', 'n_total', 'notes')},
                    'method': doc.get('method')}, cap=15)


@tool('analysis_suite', 'analysis', 'read',
      'A stack suite (suites/<id>, core.stack): overall and per-scenario pass rates with Wilson 95 % intervals, '
      'collisions, missed deadlines, minimum TTC, criticality with its interval, the failures (scenario, seed, run, '
      'why), traffic-rule and MRM summaries; and its safety case\'s residual risks when built.',
      C.obj({'id': C.s('suite id')}, ['id']))
def analysis_suite(ctx, id):
    from ...core import stack as ST
    d = C.resolve_in(ctx, 'suites', id, 'suite')
    rows = ST.read_results(d) if os.path.isfile(os.path.join(d, 'results.jsonl')) else []
    if not rows:
        raise ToolError('the suite has no results yet')
    summ = ST.summarise(rows)
    res = C.read_json(os.path.join(d, 'results.json'), {}) or {}
    out = {'id': id, 'name': res.get('name'), 'driver': res.get('driver'), 'planned': res.get('planned'),
           'finished': len(rows), 'overall': summ.get('overall'), 'scenarios': summ.get('scenarios'),
           'failures': [{k: (C.run_id(ctx, f[k]) if k == 'run' and f.get(k) and C.inside(f[k], C.workspace(ctx))
                             else f.get(k)) for k in ('scenario', 'seed', 'run', 'status', 'error', 'collision',
                                                       'min_ttc_s', 'checks') if k in f}
                        for f in summ.get('failures') or []],
           'rules': res.get('rules'), 'mrm': res.get('mrm')}
    sc = C.read_json(os.path.join(d, 'safety-case', 'safety-case.json'))
    if sc:
        out['safety_case'] = {'residual_risks': sc.get('residual_risks'), 'not_a_claim': sc.get('not_a_claim'),
                              'coverage': sc.get('coverage')}
    return compact(out, cap=12)


@tool('analysis_v2x_benefit', 'analysis', 'read',
      'A V2X benefit result (v2x/<file>, from jobs_v2x_benefit): the share of road users near the ego seen by its '
      'own sensors against with the roadside units\' CPMs (overall and by class), and the time gained out of '
      'occlusion — an upper bound with perfect detection.',
      C.obj({'id': C.s('file name in v2x/')}, ['id']))
def analysis_v2x_benefit(ctx, id):
    p = C.resolve_in(ctx, 'v2x', id, 'V2X result')
    doc = C.read_json(p)
    if doc is None:
        raise ToolError('not a readable result (the job may still be running)')
    return compact(doc, cap=12)


@tool('analysis_rules', 'analysis', 'read',
      'A run checked against the Indian traffic rules written in temporal logic (core.runcompare.rule_metrics, as '
      'Data ▸ Browse ▸ Rules): violations, per km, rules not evaluable, lane margins, the violation events with '
      'times. Takes a few seconds per run.',
      C.obj({'run': C.s('run id')}, ['run']))
def analysis_rules(ctx, run):
    from ...core import runcompare as RC
    d = C.resolve_run(ctx, run)
    rm = RC.rule_metrics(d)
    if rm.get('error'):
        raise ToolError(f'rules: {rm["error"]}')
    return compact(dict(rm, run=C.run_id(ctx, d)), cap=15)


@tool('analysis_report_sources', 'analysis', 'read',
      'What can be reported on in the workspace (core.reports.sources): per report kind the sources\' labels (use '
      'one with jobs_report), and the reports already written (title, kind, created).')
def analysis_report_sources(ctx):
    from ..privacy import private_reason
    from ...core import reports as R
    ws = C.workspace(ctx)
    srcs = {k: [lb for lb, p in v if not private_reason(p)][:15] for k, v in R.sources(ws).items() if v}
    done = []
    for m in R.list_reports(R.reports_dir(ws))[:15]:
        done.append({'file': os.path.basename(m.get('path') or ''), 'kind': m.get('kind'),
                     'title': C.clip(m.get('title'), 120), 'created': m.get('created')})
    return {'sources': srcs, 'reports': done}
