"""Hypothesis-driven experiments and the registry (core.ablation), and the statistics they rest on
(core.metrics.uncertainty: the t distribution, the paired t-test's power and sample size, Holm's correction, the
expanded paired bootstrap).

The statistics are validated against known values and by simulation: the t quantiles and the sample sizes of Cohen's
tables (dz 0.5 → 34 pairs at power 0.808, dz 0.8 → 15 at 0.821, dz 0.2 → 199); the exact power against simulated
t-tests; the expanded paired bootstrap's interval covering a known effect at about its nominal rate where the plain
percentile interval under-covers; its test's rejection rate matching the planned power, and alpha under no effect.
Then the ablation itself on the simulated executor (a known effect is found, a contrary one contradicts, none is "no
detectable effect" with its power stated, an interaction is recovered), resuming, the registry and catalogue, the
re-run checked against the original, the command, the assistant's tool — and on the fake carla, the detection
executor end to end: drives recorded through the job queue, each model evaluated on the same drive, pinned, re-run
and checked."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKE, ROOT
from falcon_autonomy.core import ablation as A
from falcon_autonomy.core import metrics as M
from falcon_autonomy.core import provenance as PV
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (GUI fixtures; Qt slot exceptions fail tests)

SYN = 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector'


# ================================================================================================ the statistics
def test_t_distribution_and_sample_sizes_match_known_values():
    assert M.t_ppf(0.975, 10) == pytest.approx(2.228139, abs=1e-5)          # Student's table
    assert M.t_ppf(0.975, 1) == pytest.approx(12.7062, abs=1e-3)
    assert M.t_ppf(0.975, 100) == pytest.approx(1.983972, abs=1e-5)
    assert M.t_ppf(0.025, 10) == pytest.approx(-2.228139, abs=1e-5)
    # the noncentral integral with ncp → 0 is the central t
    assert M.t_cdf(1.5, 7, 1e-12) == pytest.approx(M.t_cdf(1.5, 7), abs=1e-6)
    # Cohen (1988) / G*Power: two-sided paired t-test, alpha 0.05, power 0.8
    for dz, n, pw in ((0.5, 34, 0.8078), (0.8, 15, 0.8213), (0.2, 199, 0.8017)):
        r = M.paired_sample_size(dz, 0.05, 0.8)
        assert r['n'] == n and r['power'] == pytest.approx(pw, abs=1e-3) and not r['capped']
        assert M.paired_power(n - 1, dz) < 0.8
    with pytest.raises(ValueError):
        M.paired_sample_size(0.0)
    # Holm's step-down adjustment (as statsmodels' multipletests(method='holm') gives it; the order kept, None left)
    adj = M.holm([0.01, 0.04, 0.03, None])
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06]) and adj[3] is None
    assert M.holm([0.2, 0.001, 0.5]) == pytest.approx([0.4, 0.003, 0.5])


def test_the_exact_power_matches_simulated_t_tests():
    rng = np.random.default_rng(3)
    for n, dz in ((34, 0.5), (10, 0.8), (6, 1.2)):
        d = rng.normal(dz, 1.0, (20000, n))
        t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(n))
        sim = float(np.mean(np.abs(t) > M.t_ppf(0.975, n - 1)))
        assert sim == pytest.approx(M.paired_power(n, dz), abs=0.012), (n, dz)       # ≈ 3.5 standard errors


def _paired_sims(n, delta, sd, reps, method, seed):
    """reps datasets of n units with a known mean difference: (coverage of delta, rejection rate at 0.05)."""
    rng = np.random.default_rng(seed)
    cover = rej = 0
    for s in range(reps):
        base = rng.normal(0.6, 0.1, n)
        d = rng.normal(delta, sd, n)
        items = [{'a': x, 'b': x + y} for x, y in zip(base, d)]
        r = M.paired_bootstrap(lambda its, k: float(np.mean([it[k] for it in its])), items, 'a', 'b', n=1000,
                               seed=s, unit='image', method=method)
        cover += r['ci_low'] <= delta <= r['ci_high']
        rej += r['p_value'] < 0.05
    return cover / reps, rej / reps


def test_the_expanded_paired_bootstrap_covers_a_known_effect_at_the_nominal_rate():
    exp, _ = _paired_sims(10, 0.1, 0.2, 300, 'expanded', 11)
    pct, _ = _paired_sims(10, 0.1, 0.2, 300, 'percentile', 11)
    # measured while writing (400 datasets each): n = 8: 0.943 expanded against 0.880 percentile; n = 12: 0.945
    # against 0.890; n = 20: 0.953 against 0.923
    assert 0.91 <= exp <= 0.985, exp
    assert pct < exp, (pct, exp)


def test_the_bootstrap_test_has_the_planned_power_and_its_alpha():
    n = M.paired_sample_size(0.8, 0.05, 0.8)['n']
    _, power = _paired_sims(n, 0.16, 0.2, 300, 'expanded', 5)           # dz = 0.16 / 0.2 = 0.8
    assert power == pytest.approx(M.paired_power(n, 0.8), abs=0.08), power      # measured 0.818 against 0.821
    _, alpha = _paired_sims(n, 0.0, 0.2, 300, 'expanded', 7)
    assert alpha <= 0.1, alpha                                                   # measured 0.07 (400 datasets)


def test_expanded_bootstrap_and_p_value_shapes():
    items = [{'a': 0.5, 'b': 0.5 + 0.1 * k} for k in range(1, 9)]
    r = M.paired_bootstrap(lambda its, k: float(np.mean([it[k] for it in its])), items, 'a', 'b', n=500, seed=0,
                           unit='image', method='expanded')
    assert r['method'] == 'expanded' and r['ci_low'] > 0 and r['p_b_gt_a'] == 1.0
    # no resample at or below 0: the p-value is its floor, computed at half a resample (an upper bound)
    from statistics import NormalDist
    floor = 2 * (1 - M.t_cdf(NormalDist().inv_cdf(1 - 0.5 / 500) / (8 / 7) ** 0.5, 7))
    assert r['p_value'] == pytest.approx(floor) and r['p_value'] < 0.05
    b = M.bootstrap(lambda its: float(np.mean([it['b'] for it in its])), items, n=500, unit='image',
                    method='expanded')
    p = M.bootstrap(lambda its: float(np.mean([it['b'] for it in its])), items, n=500, unit='image')
    assert b['ci_high'] - b['ci_low'] > p['ci_high'] - p['ci_low']              # wider than the percentile's
    with pytest.raises(ValueError):
        M.paired_bootstrap(lambda its, k: 0.0, items, 'a', 'b', method='bca')


# ================================================================================================ the spec and plan
def _sim_spec(eid='sim', add=0.05, effect=0.05, sd=0.06, **kw):
    s = {'id': eid, 'hypothesis': {'text': 'the fine-tuned detector raises night recall', 'direction': 'increase'},
         'metric': 'AR100',
         'factors': [{'name': 'model', 'kind': 'value', 'levels': ['base', 'ft']},
                     {'name': 'weather', 'kind': 'value', 'levels': ['clear', 'rain']},
                     {'name': 'light', 'kind': 'value', 'levels': ['day', 'night']}],
         'contrasts': [{'factor': 'model', 'a': 'base', 'b': 'ft', 'where': {'light': 'night'}, 'by': ['weather']}],
         'design': {'effect': effect, 'sd': sd}, 'analysis': {'n_boot': 1000},
         'run': {'executor': 'simulated', 'simulate': {'base': 0.6, 'noise_sd': sd / 2 ** 0.5, 'unit_sd': 0.1,
                                                        'effects': [{'when': {'model': 'ft', 'light': 'night'},
                                                                     'add': add}]}}}
    s.update(kw)
    return s


def test_normalise_refuses_what_cannot_be_run():
    ok = A.normalise(_sim_spec())
    assert ok['factors'][0]['levels'] == ['base', 'ft'] and ok['analysis']['correction'] == 'holm'
    assert ok['factors'][0]['values'] == {'base': 'base', 'ft': 'ft'} and ok['run']['simulate']['base'] == 0.6
    ex = A.normalise(A.example_spec())                       # idempotent (a re-run normalises a normalised spec)
    assert A.normalise(ex) == ex and ex['factors'][2]['values']['night'] == {'sun_altitude_angle': -90.0}
    for change, why in (({'hypothesis': {'text': ''}}, 'state the hypothesis'), ({'metric': ''}, 'metric'),
                        ({'factors': []}, 'at least one factor'),
                        ({'contrasts': [{'factor': 'model', 'a': 'base', 'b': 'nope'}]}, 'two different levels'),
                        ({'contrasts': [{'factor': 'model', 'a': 'base', 'b': 'ft', 'where': {'light': 'dusk'}}]},
                         'where'),
                        ({'contrasts': [{'factor': 'colour', 'a': 'x', 'b': 'y'}]}, 'no factor'),
                        ({'design': {'effect': None, 'sd': None}}, 'state the effect'),
                        ({'run': {'executor': 'detection'}}, 'detection executor'),
                        ({'analysis': {'correction': 'fdr'}}, 'correction')):
        s = _sim_spec()
        s.update(change)
        with pytest.raises(A.AblationError, match=why):
            A.normalise(s)


def test_normalise_checks_against_the_project(project):
    s = A.example_spec()
    with pytest.raises(A.AblationError, match='no model'):
        A.normalise(s, project)
    project.doc['models'] = [{'id': 'base', 'name': 'B', 'kind': 'detector', 'runtime': 'python', 'entry': SYN},
                             {'id': 'finetuned', 'name': 'F', 'kind': 'detector', 'runtime': 'python', 'entry': SYN}]
    assert A.normalise(s, project)['run']['executor'] == 'detection'
    s['factors'][1]['levels'] = ['ClearNoon', 'Monsoon']
    with pytest.raises(A.AblationError, match='not a CARLA weather preset'):
        A.normalise(s, project)


def test_the_plan_grid_seeds_and_power_analysis():
    s = A.normalise(_sim_spec())
    pl = A.plan(s)
    # dz = 0.05 / 0.06; two contrasts (by weather) each at 0.025: 17 pairs, one unit per replicate
    assert pl['power']['dz'] == pytest.approx(0.05 / 0.06) and pl['power']['alpha_per_contrast'] == 0.025
    assert pl['power']['replicates'] == 17 == M.paired_sample_size(0.05 / 0.06, 0.025, 0.8)['n']
    assert [c['units_per_replicate'] for c in pl['power']['contrasts']] == [1, 1]
    assert pl['counts'] == {'cells': 8, 'replicates': 17, 'runs': 136, 'drives': 0, 'evaluations': 0}
    assert [k['label'] for k in pl['contrasts']] == ['model: ft − base | light=night, weather=clear',
                                                     'model: ft − base | light=night, weather=rain']
    # every cell of a replicate shares its seed (the pairing); names are stable
    r0 = [r for r in pl['runs'] if r['replicate'] == 0]
    assert {r['seed'] for r in r0} == {0} and pl['runs'][0]['name'] == 'c000-r000' and pl['seeds'][-1] == 16
    # a pooled contrast spans the other factors: four units per replicate, a quarter of the replicates
    p2 = A.plan(A.normalise(_sim_spec(contrasts=[{'factor': 'model', 'a': 'base', 'b': 'ft'}])))
    n = M.paired_sample_size(0.05 / 0.06, 0.05, 0.8)['n']
    assert p2['power']['contrasts'][0]['units_per_replicate'] == 4 and p2['power']['replicates'] == -(-n // 4)
    # the drives of the detection executor: one per non-model cell and replicate, shared by the model levels
    s3 = _sim_spec(design={'replicates': 3}, run={'executor': 'detection'})
    for f in s3['factors']:
        f['kind'] = 'model' if f['name'] == 'model' else 'weather'
    s3['factors'][1]['levels'] = ['ClearNoon', 'HardRainNoon']
    s3['factors'][2]['levels'] = {'day': {'sun_altitude_angle': 45.0}, 'night': {'sun_altitude_angle': -90.0}}
    p3 = A.plan(A.normalise(s3))
    assert p3['counts']['drives'] == 12 and p3['counts']['evaluations'] == 24
    by_drive = {}
    for r in p3['runs']:
        by_drive.setdefault(r['drive'], set()).add(r['cell']['model'])
    assert all(v == {'base', 'ft'} for v in by_drive.values()) and p3['power']['dz'] is None
    # capped: said so, with the power reached
    p4 = A.plan(A.normalise(_sim_spec(design={'effect': 0.05, 'sd': 0.06, 'max_replicates': 5})))
    assert p4['power']['capped'] and p4['power']['replicates'] == 5 and 'capped' in p4['power']['notes'][0]
    assert p4['power']['contrasts'][0]['power'] < 0.8
    assert 'power analysis' in A.describe_plan(pl) and 'noncentral t' in pl['power']['method']


# ================================================================================================ running (simulated)
def test_a_known_effect_is_found_and_the_registry_lists_it(tmp_path):
    from falcon_autonomy.core.catalog import Catalog
    ws = str(tmp_path / 'ws')
    c = A.create(ws, _sim_spec())
    assert os.path.isfile(os.path.join(c['path'], 'plan.json')) and c['plan']['pins']['app'] == 'Falcon Autonomy'
    assert A.registry(ws)[0]['status'] == 'planned'
    res = A.run(ws, 'sim')
    assert res['verdict'] == 'supported' and res['simulated'] and 'SIMULATED' in res['warnings'][0]
    for k in res['contrasts']:
        assert k['verdict'] == 'supported' and k['n_pairs'] == 17 and k['ci_low'] <= 0.05 <= k['ci_high']
        assert k['p_adjusted'] >= k['p_value'] and k['sim_ci_low'] <= k['ci_low'] and k['dz'] > 0
        assert k['dz_ci_low'] <= k['dz'] <= k['dz_ci_high'] and k['power_at_planned'] is not None
    assert len(res['cells']) == 8 and all(c_['n'] == 17 for c_ in res['cells'])
    row = A.registry(ws)[0]
    assert row['status'] == 'analysed' and row['verdict'] == 'supported' and row['hypothesis'].startswith('the fine')
    assert row['factors'] == {'model': ['base', 'ft'], 'weather': ['clear', 'rain'], 'light': ['day', 'night']}
    cat = Catalog(os.path.join(ws, 'catalog.sqlite'))
    e = cat.get('experiment:sim')
    assert e['kind'] == 'experiment' and e['status'] == 'analysed' and e['summary']['verdict'] == 'supported'
    cat._exec("DELETE FROM entries WHERE kind = 'experiment'", commit=True)
    cat.rebuild(ws)                                                     # the files are the truth
    assert cat.get('experiment:sim')['summary']['hypothesis'].startswith('the fine')
    cat.close()
    assert 'SUPPORTED' in A.table(res) and 'model: ft − base | light=night, weather=rain' in A.table(res)
    # an unchanged analysis is not recomputed; the same id with another definition is refused
    assert A.analyse(ws, 'sim')['key'] == res['key']
    with pytest.raises(A.AblationError, match='another definition'):
        A.create(ws, _sim_spec(add=0.0))


def test_contrary_null_and_interaction_effects(tmp_path):
    ws = str(tmp_path / 'ws')
    r = A.create(ws, _sim_spec('contra', add=-0.08)) and A.run(ws, 'contra')
    assert r['verdict'] == 'contradicted' and all(k['verdict'] == 'contradicted' for k in r['contrasts'])
    r = A.create(ws, _sim_spec('null', add=0.0)) and A.run(ws, 'null')
    assert r['verdict'] == 'not supported'
    assert all(k['verdict'] == 'no detectable effect' or k['verdict'] == 'supported' for k in r['contrasts'])
    assert sum(k['verdict'] == 'supported' for k in r['contrasts']) == 0
    # an interaction: (ft − base at night) − (ft − base by day) = 0.05, pooled over weather
    s = _sim_spec('inter', contrasts=[{'factor': 'model', 'a': 'base', 'b': 'ft',
                                       'interaction': {'factor': 'light', 'a': 'day', 'b': 'night'}}])
    r = A.create(ws, s) and A.run(ws, 'inter')
    k = r['contrasts'][0]
    assert k['label'].startswith('(model: ft − base) at light=night − at light=day')
    assert k['ci_low'] <= 0.05 <= k['ci_high'] and k['verdict'] == 'supported' and k['n_pairs'] == 2 * \
        r['replicates']
    # rule any: one of two contrasts is enough; direction change: either way
    s = _sim_spec('anyrule', add=0.08, hypothesis={'text': 'h', 'direction': 'change', 'rule': 'any'},
                  contrasts=[{'factor': 'model', 'a': 'base', 'b': 'ft', 'by': ['light']}])
    r = A.create(ws, s) and A.run(ws, 'anyrule')
    assert [k['verdict'] for k in r['contrasts']] == ['no detectable effect', 'supported'] and \
        r['verdict'] == 'supported'


def test_resume_failure_and_a_python_executor(tmp_path):
    ws = str(tmp_path / 'ws')
    calls = []

    class Exe:
        def __init__(self, fail_at=None):
            self.fail_at = fail_at

        def __call__(self, r, ctx):
            calls.append(r['name'])
            if r['name'] == self.fail_at:
                raise RuntimeError('the simulator went away')
            return {'metrics': {'AR100': 0.5 + (0.1 if r['cell']['model'] == 'ft' else 0.0) + 0.01 * r['seed']}}
    A.create(ws, _sim_spec('py', design={'replicates': 3}, contrasts=[{'factor': 'model', 'a': 'base', 'b': 'ft'}]))
    with pytest.raises(A.AblationError, match='went away'):
        A.run(ws, 'py', executor=Exe(fail_at='c005-r001'))
    assert A.status(A.path_of(ws, 'py'))['status'] == 'failed' and len(calls) == 14
    calls.clear()
    res = A.run(ws, 'py', executor=Exe())
    assert calls == [f'c{k:03d}-r{r:03d}' for r in (1, 2) for k in range(8)
                     if (r, k) >= (1, 5)]                                # only what had not finished
    assert res['contrasts'][0]['diff'] == pytest.approx(0.1) and not res['simulated']


def test_the_rerun_is_checked_against_the_original(tmp_path):
    from falcon_autonomy.core.reproduce import compare_experiments
    ws = str(tmp_path / 'ws')
    A.create(ws, _sim_spec('orig', design={'replicates': 4}))
    A.run(ws, 'orig')
    doc = A.rerun(ws, 'orig')
    assert doc['rerun'] == 'orig-rerun1' and doc['check']['ok'] and doc['check']['pins_changed'] == []
    assert [c['name'] for c in doc['check']['checks']] == ['planned runs', 'runs finished in both',
                                                          'each run\'s AR100', 'effects and intervals', 'verdicts']
    assert A.registry(ws)[0]['rerun_of'] == 'orig' or A.registry(ws)[1]['rerun_of'] == 'orig'
    assert os.path.isfile(os.path.join(A.path_of(ws, 'orig-rerun1'), 'rerun.json'))
    # a run whose value differs, and a pin that changed, are both reported
    from falcon_autonomy.core.experiments import Experiment
    tr = Experiment(A.path_of(ws, 'orig-rerun1')).run('c003-r001')
    tr.log_metric('AR100', 0.99)
    info = tr.info()
    tr.finish('ok', pins=dict(info['pins'], carla_server='0.9.15'))
    rep = compare_experiments(A.path_of(ws, 'orig'), A.path_of(ws, 'orig-rerun1'))
    assert not rep['ok'] and rep['differing'][0]['run'] == 'c003-r001'
    assert {d['key'] for d in rep['pins_changed']} == {'runs.carla_server'}
    assert PV.diff_pins({'a': 1, 'w': {'x': 1}}, {'a': 1, 'w': {'x': 2}}) == [{'key': 'w.x', 'a': 1, 'b': 2}]


def test_pins_hash_what_the_runs_depend_on(project):
    project.doc['models'] = [{'id': 'm', 'name': 'M', 'kind': 'detector', 'runtime': 'python', 'entry': SYN,
                              'params': {'h50': 20.0}}]
    a = PV.pins(project, project.workspace, ['world-town10'], ['rig-default'], ['m'], seeds=[3, 1, 3])
    assert a['seeds'] == [1, 3] and a['weights']['m'].startswith('registration:') and a['app_version']
    project.doc['models'][0]['params']['h50'] = 21.0                     # other weights: another pin
    project.doc['rigs'][0]['sensors'][0]['width'] = 320
    b = PV.pins(project, project.workspace, ['world-town10'], ['rig-default'], ['m'], seeds=[1, 3])
    assert {d['key'] for d in PV.diff_pins(a, b)} == {'weights.m', 'rigs.rig-default'}


# ================================================================================================ the command and tool
def _cli(*args):
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', *args], capture_output=True, text=True, cwd=ROOT,
                       timeout=300)
    return r.returncode, r.stdout + r.stderr


def test_the_command(tmp_path):
    from falcon_autonomy.cli import main
    ws = str(tmp_path / 'ws')
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(_sim_spec('cmd', design={'replicates': 3},
                                         contrasts=[{'factor': 'model', 'a': 'base', 'b': 'ft'}])))
    assert main(['experiment', 'ablation', 'plan', str(spec)]) == 0
    assert main(['experiment', 'ablation', 'new', str(spec), '--workspace', ws]) == 0
    assert main(['experiment', 'ablation', 'new', str(tmp_path / 'missing.json'), '--workspace', ws]) == 2
    code, out = _cli('experiment', 'ablation', 'run', 'cmd', '--workspace', ws)
    assert code == 0 and 'verdict:' in out, out
    code, out = _cli('experiment', 'results', A.path_of(ws, 'cmd'))
    assert code == 0 and 'hypothesis: the fine-tuned detector raises night recall' in out
    code, out = _cli('experiment', 'ablation', 'rerun', 'cmd', '--workspace', ws)
    assert code == 0 and 're-run agrees with the original' in out, out
    code, out = _cli('experiment', 'ablation', 'list', '--workspace', ws)
    assert code == 0 and 'cmd-rerun1' in out and 'analysed' in out
    code, out = _cli('experiment', 'ablation', 'example')
    assert code == 0 and json.loads(out)['metric'] == 'AR100'


def test_the_research_tools(tmp_path, project):
    from falcon_autonomy.assistant.agent import GROUP_ABOUT, guess_groups
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    from falcon_autonomy.assistant.tools import context
    assert {n for n, t in REGISTRY.items() if t.group == 'research'} == {'research_ablation', 'research_registry',
                                                                        'research_paper'}
    assert 'research' in GROUP_ABOUT and guess_groups('plan an ablation of this hypothesis')[0] == 'research'
    path = str(tmp_path / 'test.falcon.json')
    ctx = context(path, FAKE)
    try:
        spec = _sim_spec(design={'replicates': 3}, contrasts=[{'factor': 'model', 'a': 'base', 'b': 'ft'}])
        spec.pop('id')
        pv = REGISTRY['research_ablation'].preview(ctx, name='tool', spec=spec)
        assert pv.startswith('register ablation tool') and '8 cells × 3 replicates = 24 runs' in pv
        assert 'create needs spec' in call(ctx, 'research_ablation', {'name': 'tool'})['error']
        r = call(ctx, 'research_ablation', {'name': 'tool', 'spec': spec})
        assert r['experiment'] == 'tool' and r['counts']['runs'] == 24 and r['power']['replicates'] == 3
        r = call(ctx, 'research_registry', {'id': 'tool'})
        assert r['status'] == 'planned' and r['power']['replicates'] == 3 and 'not analysed' in r['note']
        pv = REGISTRY['research_ablation'].preview(ctx, name='tool', action='run')
        assert pv.startswith(f'falcon experiment ablation run tool --project {path}') and '24 runs' in pv
        q = call(ctx, 'research_ablation', {'name': 'tool', 'action': 'run'})
        assert q['experiment'] == 'tool' and ctx.queue.jobs[-1].spec.kind == 'experiment'
        assert 'running already' in call(ctx, 'research_ablation', {'name': 'tool', 'action': 'rerun'})['error']
        assert len(ctx.queue.jobs) == 1                                 # one job per experiment at a time
        ctx.queue.cancel(ctx.queue.jobs[0])
        A.run(ctx.workspace, 'tool')                                   # what the job runs
        r = call(ctx, 'research_ablation', {'name': 'tool', 'action': 'analyse'})
        assert r['verdict'] in ('supported', 'not supported') and r['contrasts'][0]['n_pairs'] == 12
        r = call(ctx, 'research_ablation', {'name': 'tool', 'action': 'export'})
        assert r['folder'] == 'paper/tool' and 'contrasts.tex' in r['files'] and 'cells.pdf' in r['files']
        rows = call(ctx, 'research_registry', {})['items']
        row = next(x for x in rows if x['id'] == 'tool')
        assert row['status'] == 'analysed' and row['hypothesis'].startswith('the fine') and row['kind'] == 'ablation'
        assert row['factors']['model'] == ['base', 'ft'] and row['verdict'] in ('supported', 'not supported')
        got = call(ctx, 'research_registry', {'id': 'tool'})
        assert got['verdict'] == row['verdict'] and got['contrasts'][0]['n_pairs'] == 12
        listed = next(x for x in call(ctx, 'analysis_outputs', {'kind': 'experiment'})['items'] if x['id'] == 'tool')
        assert listed['verdict'] == row['verdict']                       # the analysis listing carries the row too
        assert REGISTRY['research_paper'].preview(ctx, kind='experiment', source='tool').endswith(
            'into paper/experiment-tool/')
        r = call(ctx, 'research_paper', {'kind': 'experiment', 'source': 'tool'})
        assert r['folder'] == 'paper/experiment-tool' and 'cells.svg' in r['files'] and 'contrasts' in r['captions']
        assert 'no experiment' in call(ctx, 'research_paper', {'kind': 'experiment', 'source': 'nope'})['error']
        assert 'not an experiment' in call(ctx, 'research_ablation', {'name': 'nope', 'action': 'run'})['error']
        assert 'invalid arguments' in call(ctx, 'research_ablation', {'name': 'tool', 'action': 'delete'})['error']
    finally:
        ctx.service.shutdown(grace_s=10.0)


# ================================================================================================ on the fake carla
def _project(base):
    from falcon_autonomy.core.project import Project
    p = Project.new('Ablation')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=96, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.doc['traffic_profiles'][0].update(vehicles=10, walkers=6)
    p.doc['models'] = [{'id': 'base', 'name': 'Base', 'kind': 'detector', 'runtime': 'python', 'entry': SYN,
                        'licence': 'GPL-3.0-or-later', 'params': {'h50': 30.0}},
                       {'id': 'ft', 'name': 'Fine-tuned', 'kind': 'detector', 'runtime': 'python', 'entry': SYN,
                        'licence': 'GPL-3.0-or-later', 'params': {'h50': 12.0}}]
    p.save(str(base / 'abl.falcon.json'))
    return p


def test_the_detection_executor_records_evaluates_pins_and_reruns(tmp_path):
    from falcon_autonomy.core.experiments import Experiment
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    p = _project(tmp_path)
    spec = {'id': 'det', 'hypothesis': {'text': 'the fine-tuned detector finds more at night'}, 'metric': 'AR100',
            'factors': [{'name': 'model', 'kind': 'model', 'levels': ['base', 'ft']},
                        {'name': 'light', 'kind': 'weather', 'levels': {'day': {'sun_altitude_angle': 45.0},
                                                                         'night': {'sun_altitude_angle': -90.0}}}],
            'contrasts': [{'factor': 'model', 'a': 'base', 'b': 'ft', 'by': ['light']}],
            'design': {'replicates': 2}, 'analysis': {'n_boot': 200},
            'run': {'executor': 'detection', 'seconds': 1.0, 'fps': 5.0}}
    svc = CarlaService(Settings(str(tmp_path / 'settings.json')), JobQueue(SubprocessLauncher()), p.workspace,
                       fake_carla=FAKE)
    try:
        A.create(p.workspace, spec, p)
        res = A.run(p.workspace, 'det', p, svc)
        assert [j.spec.kind for j in svc.queue.jobs] == ['drive'] * 4          # one drive per light × replicate
        exp = Experiment(A.path_of(p.workspace, 'det'))
        d = PV.read_json(os.path.join(exp.path, 'drives', 'd001-r000', 'drive.json'))
        assert d['weather']['sun_altitude_angle'] == -90.0 and d['seed'] == 0
        info = exp.run('c001-r000').info()                                     # base, night, replicate 0
        assert info['status'] == 'ok' and info['pins']['carla_server'] == '0.9.16'
        assert info['pins']['model'] == 'base' and info['pins']['model_weights'].startswith('registration:')
        assert info['pins']['model_weights'] != exp.run('c003-r000').info()['pins']['model_weights']
        assert info['synthetic_model'] is True
        assert os.path.isfile(os.path.join(exp.path, 'runs', 'c001-r000', 'evaluation', 'evaluation.json'))
        # the same drive serves both models (paired by construction): the runs share its key
        assert info['pins']['drive_key'] == exp.run('c003-r000').info()['pins']['drive_key']
        assert res['runs'] == {'planned': 8, 'ok': 8, 'states': {'ok': 8}} and len(res['contrasts']) == 2
        assert all(k['diff'] is not None and k['diff'] >= 0 for k in res['contrasts'])   # the lower h50 finds more
        # the whole experiment again with one call, checked against the original (the fake carla is exact)
        doc = A.rerun(p.workspace, 'det', p, svc)
        assert doc['check']['ok'], doc['check']
        assert doc['check']['pins_changed'] == []
    finally:
        svc.shutdown()


# ================================================================================================ the page
@pytest.mark.gui
def test_the_ablations_tab(qapp, window):
    from falcon_autonomy.ui.widgets import INTERVAL_ROLE
    page = window.page('training')
    tab = page.ablations
    assert page.tabs.tabText(page.tabs.indexOf(tab)) == 'Ablations'
    tab.exp_id.setText('gui')
    tab.executor.setCurrentIndex(tab.executor.findData('simulated'))
    fc = _sim_spec()
    tab.factors.setPlainText(json.dumps({'factors': fc['factors'], 'contrasts': fc['contrasts']}))
    assert tab.plan() is not None and '136 runs' in tab.text.toPlainText()
    c = tab.register()
    assert c['id'] == 'gui' and tab.current() == 'gui'
    item = tab.reg.currentItem()
    assert item.text(2) == 'planned' and item.text(1) == 'ablation'
    A.run(window.workspace(), 'gui')                                       # what the job would do
    tab.refresh(select='gui')
    pump(qapp, timeout=0.2)
    assert tab.reg.currentItem().text(2) == 'analysed' and tab.contrasts.topLevelItemCount() == 2
    v, lo, hi = tab.contrasts.topLevelItem(0).data(3, INTERVAL_ROLE)
    assert lo <= v <= hi and tab.bars.lo < lo and hi < tab.bars.hi
    assert 'verdict:' in tab.text.toPlainText()
    tab.contrasts.grab()                                                   # the interval bars paint
    job = tab.run()
    assert job.spec.argv[3:7] == ['experiment', 'ablation', 'run', 'gui'] and '--project' in job.spec.argv
    window.queue.cancel(job)
    job = tab.export()
    assert job.spec.argv[3:6] == ['paper', 'experiment', A.path_of(window.workspace(), 'gui')]
    assert job.spec.argv[-1] == os.path.join(window.workspace(), 'paper', 'gui')
    window.queue.cancel(job)
    rep = window.page('reports')
    rep.pkind.setCurrentIndex(rep.pkind.findData('experiment'))
    rep.fill_paper_sources()
    assert rep.psrc.findText('gui') >= 0
    rep.psrc.setCurrentIndex(rep.psrc.findText('gui'))
    job = rep.make_paper()
    assert job.spec.argv[3:6] == ['paper', 'experiment', A.path_of(window.workspace(), 'gui')]
    window.queue.cancel(job)
