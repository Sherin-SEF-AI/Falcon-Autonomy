"""The STL falsifier (core.stlobjective; the surrogate-safety signals and rules.robustness in core.rules; the soft-min
and the witness in core.stl; the cross-entropy optimiser in core.search): the new signals against core.criticality by
hand, the per-step robustness and the time that decides it, the value's sign and threshold consistency, the study key,
the planted robustness landscape (searchers on ρ against a binary objective — the measured numbers are in
docs/falsification.md), end to end on the fake carla with the cut-in scenario (scored, re-scored without driving, the
command, the stl judge, the loop's objective), the Falsify tab offscreen and the formal tools."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from conftest import FAKE

from falcon_autonomy.core import criticality as CR
from falcon_autonomy.core import milsearch as MS
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import rules as R
from falcon_autonomy.core import search as S
from falcon_autonomy.core import stl
from falcon_autonomy.core import stlobjective as SO


def _trace():
    """The ego at 10 m/s along x; a stopped car whose box spans x 38–42 in its lane; a car beside it (3 m to its
    right) at the ego's speed; a pedestrian that crosses the ego's path at x = 20 at t = 3 s."""
    t = np.round(np.arange(0.0, 5.0, 0.1), 6)
    n = len(t)
    tr = R.Trace(t, 10.0 * t, np.zeros(n), np.zeros(n), np.full(n, 10.0))
    acts = []
    for ti in t:
        acts.append([{'id': 'lead', 'x': 40.0, 'y': 0.0, 'yaw': 0.0, 'vx': 0.0, 'vy': 0.0, 'l': 4.0, 'w': 1.8,
                      'kind': 'vehicle'},
                     {'id': 'side', 'x': 10.0 * ti, 'y': 3.0, 'yaw': 0.0, 'vx': 10.0, 'vy': 0.0, 'l': 4.0, 'w': 1.8,
                      'kind': 'vehicle'},
                     {'id': 'ped', 'x': 20.0, 'y': -10.0 + 2.0 * ti if ti < 3.0 else -4.0 + 4.0 * (ti - 3.0),
                      'yaw': 90.0, 'vx': 0.0, 'vy': 2.0, 'l': 0.5, 'w': 0.5, 'kind': 'pedestrian'}])
    tr.actors = acts
    return tr


def _state(tr, i):
    return [tr.t[i], tr.x[i], tr.y[i], tr.yaw[i], tr.speed[i], 0.0, tr.ego_size[0], tr.ego_size[1]]


def _act(a, t):
    return [t, a['x'], a['y'], a['yaw'], a['vx'], a['vy'], a['l'], a['w']]


# ================================================================================================ signals
def test_surrogate_signals_against_criticality_by_hand():
    tr = _trace()
    sig = R.signals(tr)
    hl = tr.ego_size[0] / 2.0
    for i in (0, 5, 20):
        lead = tr.actors[i][0]
        gap = 38.0 - (tr.x[i] + hl)                              # the lead's rear minus the ego's front
        assert sig['lead_gap_m'][i] == pytest.approx(gap)
        assert sig['ttc_lead_s'][i] == pytest.approx(gap / 10.0)     # closing at 10 m/s on a stopped car
        assert sig['ttc_lead_s'][i] == pytest.approx(CR.ttc(_state(tr, i), _act(lead, tr.t[i])))
        assert sig['drac_mps2'][i] >= 100.0 / (2.0 * gap) - 1e-9   # at least the lead's v²/2d (the pedestrian's
        #                                                            demand is higher while it is in the path)
        assert sig['lat_clear_m'][i] == pytest.approx(3.0 - 0.92 - 0.9)   # the car beside: 1.18 m of clearance
        assert sig['lon_gap_m'][i] == pytest.approx(gap)
    ttc_all = [min(CR.ttc(_state(tr, i), _act(a, tr.t[i])) for a in tr.actors[i]) for i in range(len(tr.t))]
    assert np.allclose(sig['ttc_min_s'], ttc_all)
    drac_all = [max(CR.required_deceleration(_state(tr, i), _act(a, tr.t[i])) for a in tr.actors[i])
                for i in range(len(tr.t))]
    assert np.allclose(sig['drac_mps2'], drac_all)
    # the run's PET is pet()'s at the trajectories' sampling: the minimum of the pointwise series
    e = np.array([_state(tr, i) for i in range(len(tr.t))])
    for k, aid in enumerate(('lead', 'ped')):
        a = np.array([_act(tr.actors[i][[0, 2][k]], tr.t[i]) for i in range(len(tr.t))])
        assert float(np.min(CR.pet_rows(e, a))) == pytest.approx(CR.pet(e, a))
    assert np.nanmin(sig['pet_s']) == pytest.approx(min(CR.pet(e, np.array([_act(tr.actors[i][j], tr.t[i])
                                                                            for i in range(len(tr.t))]))
                                                       for j in (0, 2) if CR.pet(e, np.array(
                                                           [_act(tr.actors[i][j], tr.t[i])
                                                            for i in range(len(tr.t))])) is not None))
    # overlapping the stopped car (t ≥ 3.6): a collision — TTC 0, DRAC infinite
    k = int(np.flatnonzero(tr.t >= 3.6)[0])
    assert sig['ttc_min_s'][k] == 0.0 and math.isinf(sig['drac_mps2'][k])
    # no actors recorded: every surrogate signal is unknown, never a violation
    tr.actors = None
    sig = R.signals(tr)
    assert all(np.isnan(sig[k]).all() for k in R.SURROGATE)
    assert set(R.SURROGATE) <= set(R.SIGNALS)


def test_axis_gaps_by_hand():
    ego = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0]
    g_lon, g_lat, ahead = CR.axis_gaps(ego, [0.0, 10.0, 0.5, 0.0, 0, 0, 4.0, 2.0])
    assert g_lon[0] == pytest.approx(6.0) and g_lat[0] == 0.0 and ahead[0]
    g_lon, g_lat, ahead = CR.axis_gaps(ego, [0.0, -1.0, -5.0, 90.0, 0, 0, 4.0, 2.0])   # turned across, on the left
    assert g_lon[0] == 0.0 and g_lat[0] == pytest.approx(5.0 - 2.0 - 1.0) and not ahead[0]


# ================================================================================================ robustness
def test_robustness_series_the_deciding_time_and_rules():
    tr = _trace()
    r = R.robustness(tr, 'G (ttc_min_s >= 2.5)')
    assert r['kind'] == 'formula' and r['rho_t0'] == pytest.approx(-2.5) and r['state'] == 'violated'
    assert r['series'].shape == tr.t.shape and r['series'][0] == r['rho_t0']
    sig = R.signals(tr)
    k = int(np.argmin(sig['ttc_min_s']))
    assert r['worst_t'] == pytest.approx(tr.t[k])                 # the first sample where ttc_min is 0
    r = R.robustness(tr, 'F[0,1] (lon_gap_m <= 30)')
    assert r['rho_t0'] == pytest.approx(30.0 - (38.0 - 2.45 - 10.0))   # best within 1 s: at t = 1 s
    assert r['worst_t'] == pytest.approx(1.0)
    # rules: the lowest robustness over the known samples, as check() reports it
    rs = R.load_ruleset('india')
    res = R.check(tr, rs)
    fd = next(x for x in res['rules'] if x['id'] == 'following_distance')
    r = R.robustness(tr, ['following_distance'], rs)
    assert r['kind'] == 'rules' and r['rho_t0'] == pytest.approx(fd['robustness'], abs=1e-4)
    assert r['per_rule']['following_distance'] == pytest.approx(r['rho_t0'])
    with pytest.raises(R.RulesError, match='no rule'):
        R.robustness(tr, ['nope'], rs)
    with pytest.raises(R.RulesError, match='unknown signal'):
        R.robustness(tr, 'G (warp_factor < 9)')
    with pytest.raises(stl.STLError, match='column'):
        R.robustness(tr, 'G (ttc_min_s >= )')


def test_soft_min_is_a_labelled_lower_approximation():
    v = np.array([3.0, 1.0, 2.0, np.nan])
    for beta in (1.0, 5.0, 50.0):
        s = stl.softmin(v, beta)
        assert s <= 1.0 and 1.0 - s <= math.log(3) / beta + 1e-12
    assert stl.softmin(np.array([1.0, -np.inf]), 2.0) == -np.inf and stl.softmin(np.array([np.inf]), 2.0) == np.inf
    tr = _trace()
    node = R.parse_formula('G (lon_gap_m >= 2)')
    sig = R.signals(tr)
    exact = stl.evaluate(node, sig, tr.t)
    smooth = stl.evaluate_smooth(node, sig, tr.t, 20.0)
    assert np.all(smooth[:5] <= exact[:5] + 1e-9)
    r = R.robustness(tr, node, smooth=20.0)
    assert r['rho_smooth'] <= r['rho_t0'] and r['rho_t0'] == exact[0]
    with pytest.raises(stl.STLError):
        stl.evaluate_smooth(node, sig, tr.t, 0.0)


def test_the_value_keeps_the_verdict_and_the_threshold():
    tiny = 1e-300
    for rho in (-1e3, -1.0, -1e-9, -tiny, float('-inf'), '-inf'):
        v = SO.value(rho)
        assert v > 0.5 and SO.falsified(rho), rho
    for rho in (0.0, tiny, 1e-9, 1.0, 1e3, float('inf'), 'inf'):
        v = SO.value(rho)
        assert v < 0.5 and not SO.falsified(rho), rho
    assert SO.value(None) is None and SO.value(float('nan')) is None
    assert SO.value(-1.0) == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert SO.value(-1.0, scale=2.0) == pytest.approx(1 / (1 + math.exp(-0.5)))
    # a smoothed ρ on the other side of 0 never moves the value across 0.5: the exact verdict decides
    assert SO.value(0.2, rho_smooth=-0.5) < 0.5 and SO.value(-0.2, rho_smooth=0.5) > 0.5
    assert SO.value(-0.2, rho_smooth=-0.5) == pytest.approx(1 / (1 + math.exp(-0.5)))
    # the failure threshold every reader uses (value ≥ 0.5) reads "falsified" exactly
    hist = [{'params': {}, 'value': SO.value(r), 'i': i} for i, r in enumerate((-0.1, 0.0, 0.1, -5.0))]
    assert [h['i'] for h in S.failures(hist, 0.5)] == [3, 0]
    assert SO.jnum(float('inf')) == 'inf' and SO.jnum(float('nan')) is None and SO.fnum('-inf') == float('-inf')


def test_the_study_key_follows_the_requirement():
    k = MS.spec_key
    a = k({'objective': 'robustness', 'stl': 'G (ttc_min_s >= 1.5)'})
    assert a.startswith('stl-')
    assert a == k({'objective': 'robustness', 'stl': 'G(ttc_min_s>=1.5)  '})          # the same normal form
    assert a != k({'objective': 'robustness', 'stl': 'G (ttc_min_s >= 1.6)'})
    assert a != k({'objective': 'robustness', 'stl': 'G (ttc_min_s >= 1.5)', 'scale': 2.0})
    assert a != k({'objective': 'robustness', 'stl': 'G (ttc_min_s >= 1.5)', 'smooth': 5.0})
    assert k({'objective': 'robustness', 'rule': 'india:following_distance'}) != \
        k({'objective': 'robustness', 'rule': 'generic:following_distance'})          # another set's SHA-256
    # a criticality or perception study's key does not change with a judge's requirement
    assert k({'objective': 'criticality', 'stl': 'G (ttc_min_s >= 1.5)'}) == k({'objective': 'criticality'})
    assert not MS.needs_model({'objective': 'robustness'}) and MS.needs_model({'objective': 'both'})
    with pytest.raises(MS.MilError, match='requirement'):
        MS.normalise_spec({'objective': 'robustness'})
    with pytest.raises(MS.MilError, match='column'):
        MS.normalise_spec({'objective': 'robustness', 'stl': 'G (ttc_min_s >='})
    s = MS.normalise_spec({'objective': 'robustness', 'rule': 'india:following_distance'})
    assert 'model' not in s and MS.objective_value(s, 0.3, None, None, {'value': 0.7}) == 0.7
    assert 'falsification' in MS.describe(s)


# ================================================================================================ the searchers
def test_cem_is_resumable_and_registered():
    from falcon_autonomy.sim.studio import METHOD_NAMES, method_class
    assert 'cem' in METHOD_NAMES and method_class('cem') is S.CrossEntropy and S.METHODS['cem'] is S.CrossEntropy
    f = SO.PlantedRobustness()
    a = S.make('cem', f.space, 3)
    assert a.n_pop == 20 and a.n_elite == 2
    for _ in range(3):
        ps = a.ask(7)
        a.tell(ps, [f.value(p) for p in ps])
    b = S.from_state(json.loads(json.dumps(a.state())))
    pa, pb = a.ask(5), b.ask(5)
    assert pa == pb
    with pytest.raises(ValueError):
        S.CrossEntropy(f.space, elite=0.9)


def test_the_planted_landscape_robustness_against_a_binary_objective():
    """On the planted landscape (ρ < 0 in about 0.1 % of the cube) with 100 evaluations and seeds 0–9: CMA-ES, the
    cross-entropy method and Bayesian optimisation on the robustness reach ρ < 0 in at least 8 of 10 seeds; on the
    binary verdict (what a search sees without robustness) in at most 2."""
    rep = SO.planted_check(('cmaes', 'cem', 'bayesopt'), range(10), 100)
    assert 0.0005 < rep['volume'] < 0.002
    for m, d in rep['methods'].items():
        assert d['robustness']['found'] >= 8, (m, d)
        assert d['binary']['found'] <= 2, (m, d)
    assert rep['methods']['cmaes']['robustness']['found'] == 10


# ================================================================================================ end to end
@pytest.fixture(scope='module')
def falsify(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.sim.service import study_service
    base = tmp_path_factory.mktemp('fz')
    p = Project.new('Falsify')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(base / 'fz.falcon.json'))
    svc = study_service(workspace=p.workspace, fake_carla=FAKE, lock_dir=str(base / 'locks'))
    yield {'project': p, 'svc': svc, 'base': base}
    svc.shutdown()


PET = 'G (pet_s >= 3.5)'


_APP = []


@pytest.fixture(scope='module')
def qapp():
    """The application, kept alive for the session (another test file's may already exist)."""
    from PyQt6.QtWidgets import QApplication
    from falcon_autonomy.ui.app import make_app
    from falcon_autonomy.ui.theme import THEME
    app = QApplication.instance() or make_app()
    _APP.append(app)
    THEME.apply(app, 'dark')
    return app


def _a_run(falsify) -> str:
    """A recorded cut-in run of the module's project (one driven when no earlier test left one)."""
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.sim.studio import Study
    d = os.path.join(falsify['project'].workspace, 'studies', 'fz')
    rows = SO.study_rows(d)
    if rows:
        return rows[0]['run']
    st = Study(falsify['svc'], falsify['project'], SC.builtin('cut_in'), 'fz', 'world-town10', 'rig-default', seed=2,
               fps=10, objective={'objective': 'robustness', 'stl': PET})
    return st.evaluate([SC.sample_params(SC.builtin('cut_in'), 1)])[0]['run']


def test_a_cut_in_falsified_on_the_fake_carla(falsify):
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.sim.estimate import _judge
    from falcon_autonomy.sim.studio import Study
    p, svc = falsify['project'], falsify['svc']
    scn = SC.builtin('cut_in')

    def study(req, name='fz'):
        return Study(svc, p, scn, name, 'world-town10', 'rig-default', seed=2, fps=10,
                     objective=dict(req, objective='robustness'))
    st = study({'stl': PET})
    out = st.search('cmaes', 4, 4, 0.5)
    rows = st.results()
    assert len(rows) == 4 and len(out['history']) == 4
    doc = json.load(open(os.path.join(st.dir, 'study.json')))
    assert doc['objective']['kind'] == 'robustness' and doc['objective']['stl']['formula'] == 'G[0,inf] (pet_s >= 3.5)'
    for r in rows:
        stl_ = r['stl']
        assert r['objective_kind'] == 'robustness' and r['objective_key'] == st.okey
        assert stl_['state'] in ('violated', 'satisfied') and stl_['missing_signals'] == []
        rho = SO.fnum(stl_['rho'])
        assert (r['objective'] > 0.5) == (rho < 0) == stl_['falsified']
        sc = json.load(open(SO.series_path(st.dir, r['run'], st.okey)))
        assert len(sc['t']) == len(sc['rho_series']) and sc['rho'] == stl_['rho']
        # the run's own pointwise PET agrees with its criticality summary (the cutter's PET)
        tr = R.trace_from_run(r['run'], '')
        pet = r['scenario']['criticality']['overall']['min_pet_s']
        if pet is not None:
            assert float(np.min(R.signals(tr)['pet_s'])) == pytest.approx(pet, abs=0.11)
            assert rho == pytest.approx(min(pet, np.inf) - 3.5, abs=0.11)
        assert _judge('stl', r, st.spec) is stl_['falsified']
    runs = sorted(os.listdir(os.path.join(st.dir, 'runs')))
    # another requirement on the same study: re-scored from the recorded runs, nothing driven again
    st2 = study({'stl': 'G (lon_gap_m >= 1)'})
    rows2 = st2.current_results()
    assert st2.okey != st.okey and len(rows2) == 4 and sorted(os.listdir(os.path.join(st.dir, 'runs'))) == runs
    assert all(r['objective_key'] == st2.okey and r['stl']['rho'] is not None for r in rows2)
    # the stl judge of a criticality study reads the requirement from its spec
    from falcon_autonomy.core.milsearch import normalise_spec
    crit = normalise_spec({'objective': 'criticality', 'stl': PET})
    row = {k: v for k, v in rows[0].items() if k != 'stl'}
    assert _judge('stl', row, crit) is rows[0]['stl']['falsified']
    # an unknown requirement (light states are not recorded in drives) stops the search instead of wasting runs
    st3 = study({'stl': 'G (not tl_red)'}, 'fz-unknown')
    with pytest.raises(RuntimeError, match='unknown in every run'):
        st3.search('random', 2, 2, 0.5)
    r3 = st3.results()[0]
    assert r3['stl']['state'] == 'not evaluable' and r3['objective'] == 0.0 and 'tl_red' in r3['stl']['missing_signals']
    assert st3.told(r3) is None
    # falsify.json
    d = SO.write_result(st, 'cmaes', 4, margin=1.0)
    s = d['summary']
    assert s['evaluated'] == 4 and len(s['falsified']) == sum(r['stl']['falsified'] for r in rows)
    assert s['best_rho'] == min((r['stl']['rho'] for r in rows), key=SO.fnum)
    assert all(0 <= SO.fnum(x['rho']) < 1.0 for x in s['near_misses'])
    assert 'runs' in SO.headline(d)


def test_the_commands(falsify, capsys):
    p = falsify['project']
    code = SO.main_falsify(['run', p.path, '--scenario', 'cut_in', '--stl', PET, '--method', 'cem', '--budget', '3',
                            '--batch', '3', '--seed', '2', '--fps', '10', '--fake-carla', FAKE])
    assert code == 0
    name = SO.study_name('cut_in', {'stl': PET}, (), 10)
    d = os.path.join(p.workspace, 'studies', name)
    assert os.path.isfile(os.path.join(d, 'falsify.json'))
    capsys.readouterr()
    assert SO.main_falsify(['show', d]) == 0 and '3 runs' in capsys.readouterr().out
    assert SO.main_falsify(['run', p.path, '--scenario', 'cut_in', '--stl', 'G (pet_s >= ', '--fake-carla',
                            FAKE]) == P.REFUSED                              # it does not parse
    capsys.readouterr()
    run = _a_run(falsify)
    assert SO.main_stl(['robustness', run, '--stl', PET, '--json', '--series']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[run]['rho'] is not None and len(out[run]['rho_series']) == len(out[run]['t'])
    assert SO.main_stl(['compile', 'G (ttc_min_s >= 1.5)', '--save', 'ttc', '--project', p.path]) == 0
    assert 'ttc' in SO.load_formulas(p.workspace)
    assert SO.main_stl(['robustness', run, '--name', 'ttc', '--project', p.path]) == 0
    assert SO.main_stl(['compile', '--rule', 'india:nope']) == 2
    capsys.readouterr()
    assert SO.main_falsify(['check', '--methods', 'cmaes', '--seeds', '2', '--budget', '100']) == 0
    assert 'cmaes' in capsys.readouterr().out


def test_the_loop_and_the_estimate_take_the_requirement():
    from falcon_autonomy.core import loop as L
    odd = {'search': {'objective': 'robustness', 'stl': PET}}
    assert L.search_objective(odd, None, '') == ('robustness', None)
    kind, why = L.search_objective({'search': {'objective': 'robustness'}}, None, '')
    assert kind == 'criticality' and 'requirement' in why
    cond = {'weather': {'name': 'fog', 'ranges': {'fog_density': [20, 80]}}}
    spec, extra = L.cell_objective('robustness', {'class': 'car', 'bin': [0.0, 20.0]}, cond, odd, None)
    assert spec['objective'] == 'robustness' and spec['stl'] == 'G[0,inf] (pet_s >= 3.5)'
    assert extra == [{'name': 'weather.fog_density', 'lo': 20.0, 'hi': 80.0}]
    from falcon_autonomy.core.failrate import JUDGES
    from falcon_autonomy.sim.estimate import JUDGE_ABOUT, estimate_argv
    from falcon_autonomy.sim.studio import search_argv
    assert 'stl' in JUDGES and 'stl' in JUDGE_ABOUT
    argv = estimate_argv('p', 'cut_in', 'n', 'w', 'r', judge='stl', requirement={'stl': PET, 'smooth': 2.0})
    assert argv[argv.index('--stl') + 1] == PET and '--smooth' in argv and '--model' not in argv
    argv = search_argv('p', 'cut_in', 'n', 'cmaes', 4, 4, 0.5, 0, 'w', 'r', 'robustness',
                       requirement={'rule': 'india:following_distance'})
    assert '--rule' in argv and '--model' not in argv


# ================================================================================================ the tab
@pytest.mark.gui
def test_the_falsify_tab_offscreen(qapp, tmp_path):
    from types import SimpleNamespace
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.ui.falsify_tab import FalsifyTab
    ws = str(tmp_path / 'ws')
    os.makedirs(ws)
    submitted = []

    class Q:
        def submit(self, spec):
            submitted.append(spec)
            return SimpleNamespace(state='queued', summary=lambda: 'queued')
    win = SimpleNamespace(project=SimpleNamespace(path=str(tmp_path / 'p.falcon.json'), dirty=False),
                          workspace=lambda: ws, queue=Q(), fake_carla=FAKE, save=lambda: True)
    page = SimpleNamespace(win=win, scenarios=lambda: [(k, SC.builtin(k)) for k in ('cut_in', 'lead_brake')])
    tab = FalsifyTab(page)
    tab.formula.setPlainText('G (ttc_min_s >= ')
    tab.check()
    assert 'column' in tab.parse.text() and not tab.start_btn.isEnabled()
    tab.formula.setPlainText('G (warp >= 1)')
    tab.check()
    assert 'unknown signal' in tab.parse.text()
    tab.formula.setPlainText(PET)
    tab.check()
    assert tab.parse.text().startswith('✓') and tab.start_btn.isEnabled()
    tab.start()
    argv = submitted[-1].argv
    assert argv[3:6] == ['falsify', 'run', win.project.path] and argv[argv.index('--stl') + 1] == PET
    assert argv[argv.index('--method') + 1] == 'cmaes' and argv[argv.index('--fake-carla') + 1] == FAKE
    # results: a study with a falsified run, a near miss and a satisfied one
    d = tab.study_dir()
    os.makedirs(os.path.join(d, 'stl'))
    okey = SO.spec_key(tab.spec)
    rows = []
    for k, rho in enumerate((-0.4, 0.3, 2.0)):
        run = os.path.join(d, 'runs', f'r{k}')
        rows.append({'key': f'k{k}', 'run': run, 'values': {'cut_gap': 3.0 + k}, 'status': 'ok', 'objective_key': okey,
                     'stl': {'rho': rho, 'value': SO.value(rho), 'state': 'violated' if rho < 0 else 'satisfied',
                             'falsified': rho < 0, 'worst_t': 2.0, 'evaluated_share': 1.0, 'missing_signals': []}})
        with open(SO.series_path(d, run, okey), 'w') as fh:
            json.dump({'t': [0.0, 1.0, 2.0], 'rho_series': [rho + 2, rho + 1, rho], 'rho': rho, 'worst_t': 2.0}, fh)
    with open(os.path.join(d, 'results.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(r) + '\n' for r in rows))
    tab.load()
    assert tab.kv.vals['Best ρ'].text().startswith('-0.4') and tab.kv.vals['Falsified'].text() == '1'
    assert tab.kv.vals['Near misses'].text() == '1' and tab.runs.topLevelItemCount() == 3
    assert tab.runs.topLevelItem(0).text(1) == 'falsified' and tab.runs.topLevelItem(1).text(1) == 'near miss'
    assert tab.trace.rho == [1.6, 0.6, -0.4] and tab.trace.worst == 2.0
    # a rule as the requirement
    tab.source.setCurrentIndex(tab.source.findData('rule:india:following_distance'))
    assert tab.formula.isReadOnly() and 'lead_gap_m' in tab.formula.toPlainText() and tab.spec['rule']
    tab.grab()
    tab.shutdown()


# ================================================================================================ the tools
def test_the_formal_tools(falsify):
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    from falcon_autonomy.core import gsn as G
    p = falsify['project']
    names = {n for n, t in REGISTRY.items() if t.group == 'formal'}
    assert {'formal_stl_check', 'formal_stl_compile', 'formal_stl_robustness', 'formal_falsify_stl',
            'formal_gsn_tree', 'formal_gsn_verify'} <= names
    ctx = TOOLS.context(p.path, FAKE)
    try:
        r = call(ctx, 'formal_stl_compile', {'name': 'pet', 'formula': PET, 'about': 'PET at least 3.5 s'})
        assert r['saved'] == 'pet' and r['signals'] == ['pet_s'] and r['key'].startswith('stl-')
        assert 'column' in call(ctx, 'formal_stl_compile', {'name': 'bad', 'formula': 'G (pet_s >='})['error']
        assert 'one of formula' in call(ctx, 'formal_stl_compile', {'name': 'x'})['error']
        assert 'pet_s' in REGISTRY['formal_stl_compile'].preview(ctx, name='pet', formula=PET)
        run = os.path.relpath(_a_run(falsify), p.workspace)
        r = call(ctx, 'formal_stl_robustness', {'runs': [run], 'name': 'pet', 'series': True})
        assert r['runs'][0]['state'] in ('violated', 'satisfied') and len(r['runs'][0]['t']) <= 61
        pv = REGISTRY['formal_falsify_stl'].preview(ctx, scenario='cut_in', rule='india:following_distance',
                                                    budget=8)
        assert pv.startswith('falcon falsify run ') and '--rule india:following_distance' in pv
        est = REGISTRY['formal_falsify_stl'].estimate(ctx, scenario='cut_in', budget=8)
        assert est['seconds'] is not None and '8 ×' in est['what']
        assert 'error' in call(ctx, 'formal_gsn_tree', {})                    # no safety case yet
        doc = G.template('sotif', p.workspace)
        G.save(p.workspace, doc)
        t = call(ctx, 'formal_gsn_tree', {})
        assert t['top'] == {'G1': 'grey'} and any(n['id'] == 'J1' for n in t['nodes'])
        v = call(ctx, 'formal_gsn_verify', {})
        assert v['top'] == {'G1': 'grey'} and G.events(p.workspace)[-1]['by'] == 'assistant'
        sn = next((n['id'] for n in doc['nodes'] if n.get('evidence')), None)
        if sn:                                                                 # a falsification study is evidence
            v = call(ctx, 'formal_gsn_verify', {'rebind': sn})
            assert v['rebound'] == sn and G.node(G.load(p.workspace), sn)['evidence']['sha256']
        assert 'error' in call(ctx, 'formal_gsn_verify', {'rebind': 'G1'})
    finally:
        ctx.service.shutdown(grace_s=10.0)
