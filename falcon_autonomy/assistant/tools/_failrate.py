"""The failure rate (core.failrate, sim.estimate) and the coverage of the operating domain (core.oddcoverage) for the
assistant, inside two existing tools so the schemas the model is sent stay within the token budget:

* analysis_outputs kind failure_rate — the studies with an estimate; with id, one estimate (the probability per
  encounter with its 95 % interval, effective failures, trust, stages, the plain-Monte-Carlo comparison, the rate per
  exposure unit, the estimate per ODD cell);
* analysis_outputs kind coverage — the grid: the domain, the cells per status under the stated rule, weather × light
  with runs and untested cells, the cells with too few runs, the next cells to fill with the scenario that would
  produce each; with id (a cell, weather/road/actor/speed/light), that cell;
* jobs_search method estimate — queue `falcon failrate run` (budget: the runs at most; a fifth per adaptation stage,
  three stages, the rest for the final stage);
* jobs_search scenario cell:<cell> — a search (or estimate) of the scenario generated for an ODD cell (added to the
  project as an undoable edit when the job is queued).
"""
from __future__ import annotations

import os

from ..tool_api import ToolError
from . import _common as C


def _r(v, nd=4):
    """Floats to nd significant figures (probabilities of 1e-6 stay readable), lists element by element."""
    if isinstance(v, float):
        return float(f'{v:.{max(int(nd), 1)}g}')
    if isinstance(v, list):
        return [_r(x, nd) for x in v]
    return v


def _scenarios(ctx) -> list:
    from ...core import scenarios as SC
    return [SC.builtin(k) for k in SC.LIBRARY] + list(C.doc(ctx).get('scenarios') or [])


def cell_scenario(ctx, scenario: str) -> dict:
    """The scenario generated for 'cell:<weather/road/actor/speed/light>' (the project's copy when it is there)."""
    from ...core import oddcoverage as OC
    cid = scenario[len('cell:'):]
    try:
        OC.parse_cell(cid)
    except ValueError as e:
        raise ToolError(str(e)) from None
    base = OC.scenario_for_cell(cid, _scenarios(ctx))
    if base is None:
        raise ToolError(f'no scenario of the library or the project can produce the cell {cid}')
    scn = OC.cell_scenario(base, cid)
    have = C.project(ctx).item('scenarios', scn['id'])
    return have or scn


def base_scenario(ctx, scenario: str) -> dict:
    """The scenario a jobs_search call names (a cell's generated one for cell:…)."""
    from .jobs import _scenario
    return cell_scenario(ctx, scenario) if str(scenario).startswith('cell:') else _scenario(ctx, scenario)


def search_or_estimate(ctx, scenario, method='bayes', budget=60, batch=4, threshold=0.5, seed=0, world='', rig='',
                       objective='criticality', count='avoidable', feasibility=True, model='', weather=(), fps=20,
                       dry=False):
    """jobs_search's (kind, title, argv, options) for an estimate or a cell's scenario (module doc)."""
    from ...core.project import Edit
    from ...sim.estimate import estimate_argv, estimate_name
    from ...sim.studio import search_argv, search_name
    from .jobs import _fake, _model, _saved, _scenario, _world_rig
    if str(scenario).startswith('cell:'):
        s = cell_scenario(ctx, scenario)
        if not dry and C.project(ctx).item('scenarios', s['id']) is None:
            C.apply_edit(ctx, Edit('insert', ['scenarios'], s, text=f'Add the coverage scenario {s["name"]}'))
    else:
        s = _scenario(ctx, scenario)
    w, r = _world_rig(ctx, world, rig)
    if objective != 'criticality':
        if not model:
            raise ToolError('the perception objective needs a registered model (project_overview lists them)')
        _model(ctx, model)
    path = (C.project(ctx).path if dry else _saved(ctx)) or '<the project file>'
    if method == 'estimate':
        stage = max(10, int(budget) // 5)
        final = max(10, int(budget) - 3 * stage)
        nm = estimate_name(s['id'], objective, model, list(weather), int(fps))
        argv = estimate_argv(path, s['id'], nm, w['id'], r['id'], stage, final, max_stages=3, batch=int(batch),
                             threshold=float(threshold), seed=int(seed), objective=objective, count=count,
                             feasibility=bool(feasibility), model=model, weather=list(weather), fps=int(fps),
                             fake_carla=_fake(ctx))
        return 'study', f'Failure rate of {s["name"]} (at most {3 * stage + final} runs)', argv, {'memory_gb': 0.5}
    nm = search_name(s['id'], objective, model, list(weather), int(fps))
    argv = search_argv(path, s['id'], nm, method, int(budget), int(batch), float(threshold), int(seed), w['id'],
                       r['id'], objective, count, bool(feasibility), model, list(weather), int(fps), _fake(ctx))
    return 'study', f'Search {s["name"]} ({method}, {budget})', argv, {'memory_gb': 0.5}


# ================================================================================================ reading
def _estimate(fr: dict, full: bool) -> dict:
    from ...sim.estimate import headline
    e = fr.get('estimate') or {}
    given = e.get('status') in ('ok', 'warning')          # a refused estimate's numbers are not an answer
    out = {'headline': headline(fr), 'scenario': fr.get('scenario'), 'status': e.get('status'),
           'reasons': e.get('reasons'), 'p_per_encounter': _r(e.get('p')) if given else None,
           'ci95': _r(e['ci95']) if given and e.get('ci95') else None, 'upper95': _r(e.get('upper95')),
           'failures': e.get('failures'), 'final_runs': e.get('n'), 'ess_failures': _r(e.get('ess_failures')),
           'rel_se': _r(e.get('rel_se')), 'runs': (fr.get('runs') or {}).get('total'),
           'rate': (fr.get('rate') or {}).get('text')}
    if full:
        out.update(judge=(fr.get('judge') or {}).get('about'), threshold=(fr.get('judge') or {}).get('threshold'),
                   nominal=[f'{d["name"]}: {d["dist"]}' + (f' {d["lo"]:g}–{d["hi"]:g}' if d.get('lo') is not None
                                                           and d.get('hi') is not None else '')
                            for d in fr.get('nominal') or ()],
                   nominal_mass=_r(fr.get('nominal_mass')), stop_reason=fr.get('stop_reason'),
                   stages=[{'level': _r(s['level']), 'failures': s['failures'], 'runs': s['runs']}
                           for s in fr.get('stages') or ()],
                   plain_mc_stage0={k: _r(v) for k, v in (fr.get('plain_mc') or {}).items() if k != 'about'},
                   plain_mc_runs_for_same_rel_se=_r(fr.get('mc_runs_for_same_rel_se')) if given else None,
                   by_cell=C.capped([{'cell': c, 'runs': x['runs'], 'failures': x['failures'],
                                      'p': _r(x['p']) if x['status'] != 'refused' else None, 'status': x['status']}
                                     for c, x in sorted((fr.get('by_cell') or {}).items())], 12),
                   estimator=fr.get('estimator'), proposal=fr.get('proposal_family'))
    return out


def read_output(ctx, kind: str, id: str = '', limit: int = 20) -> dict:
    """analysis_outputs for kinds failure_rate and coverage (module doc)."""
    from ...core import oddcoverage as OC
    ws = C.workspace(ctx)
    if kind == 'failure_rate':
        if id:
            d = C.resolve_in(ctx, 'studies', id, 'study', ('failure-rate.json',))
            return _estimate(C.read_json(os.path.join(d, 'failure-rate.json'), {}), True)
        rows = []
        for e in OC.estimates(ws):
            rows.append(dict(_estimate(C.read_json(os.path.join(ws, 'studies', e['study'], 'failure-rate.json'), {}),
                                       False), id=e['study']))
        return C.capped(rows, int(limit), 'read one with its id')
    try:
        cov = OC.coverage(ws, C.project(ctx))
    except ValueError as e:
        raise ToolError(str(e)) from None
    if id:
        try:
            OC.parse_cell(id)
        except ValueError as e:
            raise ToolError(str(e)) from None
        e = cov['cells'].get(id)
        base = OC.scenario_for_cell(id, _scenarios(ctx))
        out = {'cell': id, 'about': OC.describe_cell(id),
               'scenario_that_produces_it': base.get('id') if base else None}
        if e is None:
            out.update(status='untested', runs=0, needed=cov['rule']['runs_if_no_failure'])
        else:
            out.update(status=e['status'], runs=e['runs'], failures=e['failures'], sources=e['sources'],
                       upper95=_r(e['upper']), lower95=_r(e['lower']), needed=e['needed'],
                       estimates=[{k: _r(x.get(k)) for k in ('study', 'p', 'ci95', 'status', 'runs')}
                                  for x in e.get('estimates') or ()])
        if base is not None:
            out['fill'] = f'jobs_search with scenario "cell:{id}"'
        return out
    pv = OC.pivot(cov, 'weather', 'light')
    plan = OC.fill_plan(cov, _scenarios(ctx), int(limit))
    few = sorted(((c, e) for c, e in cov['cells'].items() if e['status'] == 'too_few'), key=lambda t: -t[1]['needed'])
    return {'rule': cov['rule']['about'], 'domain': {'source': cov['domain']['source'], 'cells': cov['domain']['cells'],
                                                     'dims': cov['domain']['dims']},
            'counts': cov['counts'], 'runs': cov['runs'], 'outside_domain': cov['outside_domain'],
            'weather_x_light': {r: {c: f'{x["runs"]} runs, {x["cell_status"]["untested"]} of {x["cells"]} untested'
                                    for c, x in zip(pv['col_labels'], line)}
                                for r, line in zip(pv['row_labels'], pv['grid'])},
            'too_few': [{'cell': c, 'runs': e['runs'], 'needed': e['needed']} for c, e in few[:int(limit)]],
            'above_target': [c for c, e in cov['cells'].items() if e['status'] == 'above_target'][:int(limit)],
            'estimates': [{'study': e['study'], 'status': (e['estimate'] or {}).get('status'),
                           'p': _r((e['estimate'] or {}).get('p'))} for e in cov['estimates']],
            'next_to_fill': [{'cell': p['cell'], 'status': p['status'], 'scenario': p['base'], 'needed': p['needed']}
                             for p in plan['plan']],
            'cells_no_scenario_can_produce': plan['no_scenario_cells'],
            'fill': 'jobs_search with scenario "cell:<cell>" (method bayes-pof to map failures, estimate for a rate)'}
