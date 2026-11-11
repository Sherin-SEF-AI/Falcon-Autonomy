"""The failure-rate estimate of a scenario, run unattended through the job queue (shared by the GUI and the CLI), and
the commands `falcon failrate` and `falcon coverage` (docs/failure-rate.md, docs/odd-coverage.md).

An estimate is a study (sim.studio.Study: the same scenario runs, cache, objective and RSS labels as a rare-case
search) whose parameters are drawn by core.failrate.Sampler — adaptive importance sampling — instead of a searcher.
Its folder studies/<name>/ holds study.json, results.jsonl (every run, as a search's), failrate-state.json (the
sampler's state after every batch: a stopped estimate resumes exactly) and, when it ends, failure-rate.json: the
estimate with its interval, effective sample size and trust, the stages, the plain-Monte-Carlo comparison, the rate per
exposure unit, the estimate per ODD cell (core.oddcoverage cells of the final stage's runs) and the final runs with
their weights. core.safetycase reads failure-rate.json as evidence; core.oddcoverage shows it in its grid.
"""
from __future__ import annotations

import os

from ..core import failrate as FR
from ..core import protocol as P
from ..core.provenance import read_json, write_json

STATE_FILE = 'failrate-state.json'
RESULT_FILE = 'failure-rate.json'
JUDGE_ABOUT = {
    'objective': 'the study\'s objective ≥ the threshold — the rare-case search\'s own failure (for the criticality '
                 'objective by default counting only failures the ego could have avoided, core.rss)',
    'checks': 'the scenario\'s pass criteria failed (no collision, minimum time to collision: run.json, as a stack '
              'suite judges a run, core.stack)',
    'rules': 'a traffic rule was violated (core.rules: a signal-temporal-logic formula with negative robustness, '
             'core.stl; the rule set for the run\'s handedness)'}


def estimate_name(scenario: str, objective: str = 'criticality', model: str = '', weather=(), fps: float = 20,
                  judge: str = 'objective') -> str:
    """An estimate's study name (Rare cases ▸ Failure rate): one per scenario, objective (and model), weather searched,
    rate and judge."""
    from .studio import WEATHER_TAGS
    parts = ['rate', scenario] + ([] if objective == 'criticality' else [model or 'none', objective])
    parts += [WEATHER_TAGS.get(k, k) for k in weather]
    if fps != 20:
        parts.append(f'{fps:g}hz')
    return '-'.join(parts + ([] if judge == 'objective' else [judge]))


def estimate_argv(project_path: str, scenario: str, name: str, world: str, rig: str, stage_size: int = 100,
                  final_size: int = 200, rho: float = 0.1, defensive: float = 0.1, max_stages: int = 6,
                  batch: int = 4, threshold: float = 0.5, seed: int = 0, judge: str = 'objective',
                  objective: str = 'criticality', count: str = 'avoidable', feasibility: bool = True, model: str = '',
                  weather=(), fps: float = 20, exposure_per: str = 'encounter', encounters: float = 0.0,
                  exposure_source: str = '', fake_carla: str = '') -> list:
    """`falcon failrate run …` as Rare cases ▸ Failure rate queues it (the arguments after `python -m
    falcon_autonomy`)."""
    argv = ['failrate', 'run', project_path, '--scenario', scenario, '--name', name, '--world', world, '--rig', rig,
            '--stage-size', str(stage_size), '--final', str(final_size), '--rho', str(rho), '--defensive',
            str(defensive), '--max-stages', str(max_stages), '--batch', str(batch), '--threshold', str(threshold),
            '--seed', str(seed), '--judge', judge, '--objective', objective, '--count', count]
    if not feasibility:
        argv.append('--no-feasibility')
    if objective != 'criticality':
        argv += ['--model', model]
    for k in weather:
        argv += ['--weather', k]
    if fps != 20:
        argv += ['--fps', str(fps)]
    if exposure_per != 'encounter':
        argv += ['--exposure-per', exposure_per, '--encounters', str(encounters)]
        if exposure_source:
            argv += ['--exposure-source', exposure_source]
    return argv + (['--fake-carla', fake_carla] if fake_carla else [])


def _judge(kind: str, row: dict) -> bool | None:
    if kind == 'checks':
        p = (row.get('scenario') or {}).get('passed')
        return None if p is None else p is False
    if kind == 'rules':
        from ..core.rules import check_run
        try:
            res = check_run(row['run'])
        except Exception:                     # noqa: BLE001 (a run the rule checker cannot read: not judged)
            return None
        return res['totals']['violated_rules'] > 0
    return None                               # the objective: the sampler's own (value ≥ threshold)


def run_estimate(study, judge: str = 'objective', threshold: float = 0.5, opts: dict | None = None,
                 batch: int = 4, exposure: dict | None = None, progress=None, log=None) -> dict:
    """Drive a core.failrate.Sampler over the study's scenario (module doc); returns and writes failure-rate.json."""
    from ..core.oddcoverage import run_cells
    if judge not in FR.JUDGES:
        raise ValueError(f'judge: one of {", ".join(FR.JUDGES)}')
    opts = dict(FR.DEFAULTS, **(opts or {}))
    space = study.space()
    nom = FR.Nominal.from_scenario(study.scn, space)
    sp = os.path.join(study.dir, STATE_FILE)
    key = {'objective_key': study.okey, 'judge': judge, 'threshold': float(threshold), 'nominal': nom.doc(),
           'options': {k: opts[k] for k in sorted(opts)}, 'search_seed': study.search_seed}
    sampler, runs = None, {}
    try:
        d = read_json(sp)
        if {k: d.get(k) for k in key} == key:
            sampler = FR.Sampler.from_state(d['state'])
            runs = {int(k): v for k, v in (d.get('runs') or {}).items()}
    except (OSError, ValueError, KeyError, FR.FailRateError):
        sampler = None
    if sampler is None:
        sampler = FR.Sampler(nom, threshold=threshold, seed=study.search_seed, **opts)
    elif log:
        log(f'resuming the estimate: {sampler.total_runs()} runs already made')
    verdicts = {}
    while not sampler.done:
        asked = sampler.ask(max(1, min(int(batch), sampler.remaining() or int(batch))))
        if not asked:
            break
        res = study.evaluate([a['params'] for a in asked], progress)
        vals, fails = [], []
        for a, r in zip(asked, res):
            ok = bool(r) and r.get('status') == 'ok'
            vals.append(float(r['objective']) if ok else None)
            f = _judge(judge, r) if ok and judge != 'objective' else None
            if ok and judge != 'objective' and f is None:
                vals[-1] = None                     # the judge cannot read it: a lost run, replaced
            fails.append(f)
            if ok:
                runs[a['id']] = r['run']
                verdicts[a['id']] = f
        sampler.tell([a['id'] for a in asked], vals, fails)
        write_json(sp, dict(key, state=sampler.state(), runs={str(k): v for k, v in runs.items()}))
        if log and sampler.stages and len(sampler.stages) != getattr(study, '_logged_stages', 0):
            st = sampler.stages[-1]
            study._logged_stages = len(sampler.stages)
            log(f'stage {st["stage"]}: level {st["level"]:.3g} (threshold {threshold:g}), {st["failures"]} '
                f'failures in {st["runs"]} runs' + (f'; {sampler.stop_reason}' if sampler.stop_reason else ''))
    out = sampler.result()
    fin = sampler.final_samples()
    cfgs = {}

    def cells(s):
        run = runs.get(s['id'])
        if not run:
            return []
        if run not in cfgs:
            try:
                cfgs[run] = run_cells(read_json(os.path.join(run, 'drive.json')))[0]
            except (OSError, ValueError):
                cfgs[run] = []
        return cfgs[run]
    out['by_cell'] = FR.by_cell(fin, cells)
    est = out['estimate']
    try:
        out['rate'] = FR.exposure_rate(est, exposure)
    except FR.FailRateError as e:
        out['rate'] = {'per': 'encounter', 'text': str(e)}
    from ..core.provenance import block
    out.update(study=os.path.basename(study.dir), scenario=study.scn.get('id'), scenario_name=study.scn.get('name'),
               world=study.world, rig=study.rig, seed=study.seed, objective=study.objective_doc(),
               judge={'kind': judge, 'threshold': float(threshold), 'about': JUDGE_ABOUT[judge]},
               feasibility=study.feasibility_notes(),
               encounter=('one run of the scenario: its actors and timeline, the world, rig, traffic and scenario seed '
                          'of the study, the parameters drawn from the nominal'),
               final_runs=[{'run': runs.get(s['id']), 'w': s['w'], 'fail': s['fail'], 'value': s['value']}
                           for s in fin],
               provenance=block({'results': os.path.join(study.dir, 'results.jsonl'), 'state': sp},
                                seed=study.search_seed))
    write_json(os.path.join(study.dir, RESULT_FILE), out)
    return out


def headline(res: dict) -> str:
    """The estimate in one line (the CLI, the job's summary, the assistant)."""
    e = res.get('estimate') or {}
    if e.get('status') == 'refused':
        why = '; '.join(e.get('reasons') or [])
        return f'refused after {res["runs"]["total"]} runs: {why}'
    ci = e.get('ci95') or [None, None]
    s = (f'P(fail per encounter) = {e["p"]:.3g} (95 %: {ci[0]:.3g}–{ci[1]:.3g}), {e["failures"]} failing of '
         f'{e["n"]} final runs, ESS of the failures {e["ess_failures"]:.0f}, {res["runs"]["total"]} runs in all')
    if (res.get('rate') or {}).get('per') not in (None, 'encounter'):
        s += f'; {res["rate"]["text"]}'
    if e.get('status') == 'warning':
        s += ' — WARNING: ' + '; '.join(e['reasons'])
    return s


# ============================================================================================== falcon failrate
def _exposure(a) -> dict | None:
    if a.exposure_per == 'encounter':
        return None
    return {'per': a.exposure_per, 'encounters': a.encounters, 'source': a.exposure_source}


def main_failrate(argv) -> int:
    """falcon failrate run PROJECT --scenario S … | show STUDY|PROJECT [--name N] | check […]  (docs/cli.md)."""
    import argparse
    import json
    from .studio import objective_args
    ap = argparse.ArgumentParser(prog='falcon failrate', description='the failure rate of a scenario with a 95 % '
                                 'interval, by adaptive importance sampling (docs/failure-rate.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run', help='estimate a scenario\'s failure rate (a resumable study)')
    r.add_argument('project')
    r.add_argument('--scenario', required=True)
    r.add_argument('--name', default='', help='the study (default rate-<scenario>…)')
    r.add_argument('--world', default='')
    r.add_argument('--rig', default='')
    r.add_argument('--seed', type=int, default=0, help='the scenario seed (world reload)')
    r.add_argument('--search-seed', type=int, default=None, help='the sampler\'s seed (default: --seed)')
    r.add_argument('--stage-size', type=int, default=FR.DEFAULTS['stage_size'])
    r.add_argument('--final', type=int, default=FR.DEFAULTS['final_size'])
    r.add_argument('--rho', type=float, default=FR.DEFAULTS['rho'])
    r.add_argument('--defensive', type=float, default=FR.DEFAULTS['defensive'])
    r.add_argument('--max-stages', type=int, default=FR.DEFAULTS['max_stages'])
    r.add_argument('--batch', type=int, default=4)
    r.add_argument('--threshold', type=float, default=0.5)
    r.add_argument('--judge', default='objective', choices=FR.JUDGES)
    r.add_argument('--exposure-per', default='encounter', choices=FR.EXPOSURE_UNITS)
    r.add_argument('--encounters', type=float, default=0.0, help='encounters of this kind per km (or hour)')
    r.add_argument('--exposure-source', default='', help='where the encounter rate comes from')
    r.add_argument('--fps', type=float, default=0.0)
    r.add_argument('--fake-carla', default='')
    objective_args(r)
    s = sub.add_parser('show', help='an estimate: falcon failrate show STUDY_DIR | PROJECT --name N')
    s.add_argument('target')
    s.add_argument('--name', default='')
    s.add_argument('--json', action='store_true')
    s.add_argument('--exposure-per', default='', choices=('',) + FR.EXPOSURE_UNITS)
    s.add_argument('--encounters', type=float, default=0.0)
    s.add_argument('--exposure-source', default='')
    c = sub.add_parser('check', help='the estimator on problems with a known probability (no CARLA)')
    c.add_argument('--p', default='1e-4,1e-6', help='true probabilities, comma-separated')
    c.add_argument('--dims', default='2,5')
    c.add_argument('--seeds', type=int, default=50)
    c.add_argument('--stage-size', type=int, default=500)
    c.add_argument('--final', type=int, default=1000)
    c.add_argument('--max-stages', type=int, default=10)
    c.add_argument('--family', default='mean_shift', choices=FR.FAMILIES)
    c.add_argument('--problems', default='tail,corner', help='tail (standard normals), corner (uniform parameters)')
    c.add_argument('--json', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if a.action == 'check':
        rep = FR.validate([float(x) for x in a.p.split(',')], [int(x) for x in a.dims.split(',')], range(a.seeds),
                          a.stage_size, a.final, max_stages=a.max_stages, family=a.family,
                          kinds=[k for k in a.problems.split(',') if k])
        if a.json:
            print(json.dumps(rep, indent=2))
        else:
            print(f'{"problem":<7} {"P true":>7} {"d":>2} {"given":>7} {"warned":>6} {"covered":>7} {"(Wald)":>6} '
                  f'{"ok-cov":>6} {"rel.err":>7} {"rel.SE":>6} {"runs":>5} {"MC 0-fail":>9} {"MC runs, same SE":>16}')

            def f3(v):
                return '—' if v is None else f'{v:.3f}'
            for x in rep['rows']:
                mc = '—' if x['plain_mc_runs_for_same_rel_se'] is None else f'{x["plain_mc_runs_for_same_rel_se"]:.3g}'
                print(f'{x["problem"]:<7} {x["p_true"]:>7.0e} {x["dim"]:>2} {x["given"]:>3}/{x["seeds"]:<3} '
                      f'{x["warned"]:>6} {f3(x["coverage"]):>7} {f3(x["coverage_wald"]):>6} {f3(x["coverage_ok"]):>6} '
                      f'{(x["mean_abs_rel_error"] or 0):>7.3f} {(x["median_rel_se"] or 0):>6.3f} '
                      f'{x["mean_runs"]:>5.0f} {x["plain_mc_zero_failure_share"]:>9.2f} {mc:>16}')
        return 0
    if a.action == 'show':
        d = a.target
        if a.target.endswith('.falcon.json') or os.path.isfile(a.target):
            from ..core.project import Project, ProjectError
            try:
                proj = Project.load(a.target)
            except (OSError, ProjectError) as e:
                print(f'failed: {e}', flush=True)
                return P.FAILED
            d = os.path.join(proj.workspace, 'studies', a.name)
        try:
            res = read_json(os.path.join(d, RESULT_FILE))
        except (OSError, ValueError):
            print(f'failed: no estimate in {d}', flush=True)
            return P.FAILED
        if a.exposure_per:
            try:
                res['rate'] = FR.exposure_rate(res['estimate'], {'per': a.exposure_per, 'encounters': a.encounters,
                                                                 'source': a.exposure_source})
            except FR.FailRateError as e:
                print(f'failed: {e}', flush=True)
                return P.REFUSED
        print(json.dumps(res, indent=2) if a.json else headline(res), flush=True)
        return 0
    return _run(a)


def _run(a) -> int:
    from ..core.project import Project, ProjectError
    from ..core import scenarios as SC
    from .service import study_service
    from .studio import Study, objective_from_args
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    scn = proj.item('scenarios', a.scenario) or (SC.builtin(a.scenario) if a.scenario in SC.LIBRARY else None)
    if scn is None:
        print(f'failed: no scenario {a.scenario!r}', flush=True)
        return P.FAILED
    try:
        spec, extra = objective_from_args(a, proj)
        exposure = _exposure(a)
        if exposure:
            FR.exposure_rate({'p': 1e-6, 'status': 'ok', 'ci95': [1e-6, 1e-6]}, exposure)    # checks the figure
        opts = {'stage_size': a.stage_size, 'final_size': a.final, 'rho': a.rho, 'defensive': a.defensive,
                'max_stages': a.max_stages}
        FR.Sampler(FR.Nominal.standard(1), **opts)                                             # checks the options
    except (ValueError, FR.FailRateError) as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    svc = study_service(None, proj.workspace, fake_carla=a.fake_carla)
    em = P.Emitter()
    em.hello('failrate')
    name = a.name or estimate_name(scn['id'], spec['objective'], a.model, [w.split(':')[0] for w in a.weather],
                                   a.fps or 20, a.judge)
    try:
        st = Study(svc, proj, scn, name, a.world or proj.doc['worlds'][0]['id'], a.rig or proj.doc['rigs'][0]['id'],
                   a.seed, objective=spec, extra_space=extra, fps=a.fps or None, search_seed=a.search_seed)
        FR.Nominal.from_scenario(scn, st.space())
    except (ValueError, FR.FailRateError) as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.REFUSED
    st.emit = em
    em.log(f'objective: {st.objective_doc()["about"]}; failure: {JUDGE_ABOUT[a.judge]}')
    stages = a.max_stages * a.stage_size + a.final

    def prog(n):
        em.progress(min(n / max(stages, 1), 0.99), f'{n} runs (at most {stages})', force=True)
    try:
        res = run_estimate(st, a.judge, a.threshold, opts, a.batch, exposure, prog, em.log)
    except (RuntimeError, ValueError, FR.FailRateError) as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.FAILED
    em.log(headline(res))
    e = res['estimate']
    em.result('ok', summary={'status': e['status'], 'p': e['p'], 'ci95': e['ci95'], 'failures': e['failures'],
                             'ess_failures': e['ess_failures'], 'runs': res['runs']['total'],
                             'rate': (res.get('rate') or {}).get('text')})
    svc.shutdown()
    return P.OK


# ============================================================================================== falcon coverage
def main_coverage(argv) -> int:
    """falcon coverage show PROJECT [--rows D --cols D] [--json] | fill PROJECT [--max N] [--kind search|sweep]
    [--apply] [--run]  (docs/odd-coverage.md)."""
    import argparse
    import json
    from ..core import oddcoverage as OC
    from ..core import scenarios as SC
    from ..core.project import Project, ProjectError
    ap = argparse.ArgumentParser(prog='falcon coverage', description='coverage of the operating domain: tested, '
                                 'under-sampled and untested cells (docs/odd-coverage.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    for name in ('show', 'fill'):
        p = sub.add_parser(name)
        p.add_argument('project')
        p.add_argument('--target', type=float, default=OC.TARGET, help='failure rate per encounter to bound')
        p.add_argument('--confidence', type=float, default=OC.CONFIDENCE)
        p.add_argument('--json', action='store_true')
        if name == 'show':
            p.add_argument('--rows', default='weather', choices=OC.DIMENSIONS)
            p.add_argument('--cols', default='light', choices=OC.DIMENSIONS)
            p.add_argument('--cells', action='store_true', help='list every tested cell')
        else:
            p.add_argument('--max', type=int, default=10, help='cells to plan for')
            p.add_argument('--kind', default='search', choices=('search', 'sweep'))
            p.add_argument('--budget', type=int, default=0, help='runs per cell (default: the runs it still needs)')
            p.add_argument('--apply', action='store_true', help='add the scenarios to the project (saved)')
            p.add_argument('--run', action='store_true', help='then run each study now (implies --apply)')
            p.add_argument('--world', default='')
            p.add_argument('--rig', default='')
            p.add_argument('--seed', type=int, default=0)
            p.add_argument('--threshold', type=float, default=0.5)
            p.add_argument('--fake-carla', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        proj = Project.load(a.project)
        cov = OC.coverage(proj.workspace, proj, a.target, a.confidence)
    except (OSError, ProjectError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    if a.action == 'show':
        if a.json:
            print(json.dumps({k: v for k, v in cov.items() if k != 'samples'}, indent=2, default=str))
            return 0
        n, cnt = cov['domain']['cells'], cov['counts']
        print(f'{cov["runs"]} scenario runs; domain: {cov["domain"]["source"]}, {n} cells — '
              + ', '.join(f'{k.replace("_", " ")} {cnt[k]}' for k in OC.STATUSES))
        print(f'rule: {cov["rule"]["about"]}')
        pv = OC.pivot(cov, a.rows, a.cols)
        w = max(len(x) for x in pv['row_labels'] + [a.rows]) + 2
        print(f'\n{a.rows + " \\ " + a.cols:<{w}}' + ''.join(f'{c:>16}' for c in pv['col_labels']))
        for lab, line in zip(pv['row_labels'], pv['grid']):
            print(f'{lab:<{w}}' + ''.join(f'{x["runs"]:>5} run{"s" if x["runs"] != 1 else " "} '
                                          f'{x["cell_status"]["untested"]:>4}∅ ' for x in line))
        print(f'\n(each entry: distinct runs, and how many of its {pv["cells_per_entry"]} cells are untested)')
        if a.cells:
            for cid, e in sorted(cov['cells'].items()):
                est = '; '.join(f'{x["study"]}: {x["p"]:.3g}' if x.get('p') else f'{x["study"]}: refused'
                                for x in e.get('estimates') or ())
                print(f'{cid:<52} {e["runs"]:>4} runs {e["failures"]:>3} failed  {e["status"]:<12} '
                      f'needs {e["needed"]:>3}' + (f'  estimate {est}' if est else ''))
        return 0
    scns = [SC.builtin(k) for k in SC.LIBRARY] + list(proj.doc['scenarios'])
    plan = OC.fill_plan(cov, scns, a.max, a.kind, a.budget)
    world = a.world or proj.doc['worlds'][0]['id']
    rig = a.rig or proj.doc['rigs'][0]['id']
    for e in plan['plan']:
        e['argv'] = OC.plan_argv(proj.path, e, world, rig, a.seed, a.threshold, a.fake_carla)
    if a.json:
        print(json.dumps(plan, indent=2, default=str))
    else:
        for e in plan['plan']:
            print(f'{e["cell"]:<52} {e["status"]:<9} {e["kind"]} of {e["base"]}, {e["runs"]} runs → '
                  f'studies/{e["study"]}')
        print(f'{plan["candidates"]} cells need runs; {plan["no_scenario_cells"]} cells of the domain no scenario of '
              'the library or the project can produce' + (' (--json lists them)' if plan['no_scenario'] else ''))
    if a.apply or a.run:
        OC.apply_plan(proj, plan)
        proj.save()
        print(f'added {len(plan["plan"])} scenarios to {proj.path}', flush=True)
    if a.run:
        from .studio import main_study
        for e in plan['plan']:
            code = main_study(e['argv'])
            if code != P.OK:
                return code
    return 0
