"""Generation rounds (core.genround): the acquisition rule on known inputs (each term, the ranking, the budget's
allocation), the stop rule, the definition's checks; then a tiny round end to end on the fake carla with a tiny
detector Falcon trained on the CPU — the held-out set fixed before round 1 and refused when it changes or when a
training source holds one of its drives, drives and scenario runs generated for the chosen cells within the budget,
labelled from the ground truth, a dataset version, the retrain from the model's own settings, the before and after per
cell with intervals, the stop rule, resume, the registry; the command and the assistant tool."""
import copy
import json
import math
import os
import subprocess
import sys

import pytest

from falcon_autonomy.core import genround as G
from falcon_autonomy.core import loop as L
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import provenance as PV

from conftest import FAKE, ROOT
from test_gui import qapp, slot_exceptions  # noqa: F401  (GUI fixtures; Qt slot exceptions fail tests)

TINY = {'arch': 'fasterrcnn_mobilenet_v3_small_fpn', 'init': 'scratch', 'size': [128, 64], 'batch': 4, 'epochs': 1,
        'workers': 0, 'device': 'cpu', 'box_head_size': 64, 'verify_images': 2, 'export': True}


def _s(**kw):
    s = copy.deepcopy(G.DEFAULTS)
    s.update(id='g', conditions=['clear/day', 'rain/night'])
    s.update(kw)
    return s


# ------------------------------------------------------------------------------------------------ the rule
def test_the_acquisition_rule_on_known_inputs():
    s = _s(acquisition=dict(G.DEFAULTS['acquisition'], max_cells=2, min_objects=10, n0=100.0))
    rows = [{'class': 'car', 'condition': 'clear/day', 'AP': 0.6, 'n_objects': 50},
            {'class': 'car', 'condition': 'rain/night', 'AP': 0.2, 'n_objects': 50},
            {'class': 'pedestrian', 'condition': 'rain/night', 'AP': 0.1, 'n_objects': 5},      # too few objects
            {'class': 'pedestrian', 'condition': 'clear/day', 'AP': 0.3, 'n_objects': 40}]
    counts = {('car', 'rain/night'): 300, ('pedestrian', 'clear/day'): 0}
    fails = {('pedestrian', 'clear/day'): (3, 8)}
    out = G.rank_cells(rows, 0.5, s, counts, fails)
    by = {(r['class'], r['condition']): r for r in out}
    car = by[('car', 'rain/night')]
    # π 0.5 (two conditions), gap 0.3, s = sqrt(100 / 400) = 0.5, f = 1 + 1/2 (no scenario run: 0.5)
    assert car['terms']['pi'] == 0.5 and car['terms']['gap'] == pytest.approx(0.3)
    assert car['terms']['s'] == pytest.approx(0.5) and car['terms']['f'] == pytest.approx(1.5)
    assert car['ev'] == pytest.approx(0.5 * 0.3 * 0.5 * 1.5)
    ped = by[('pedestrian', 'clear/day')]
    assert ped['terms']['f'] == pytest.approx(1 + 4 / 10) and ped['ev'] == pytest.approx(0.5 * 0.2 * 1.0 * 1.4)
    assert by[('car', 'clear/day')]['ev'] == 0.0 and not by[('car', 'clear/day')]['eligible']    # above the overall
    assert not by[('pedestrian', 'rain/night')]['eligible']                                     # 5 objects < 10
    assert [(r['class'], r['condition'], r['rank']) for r in out if r.get('rank')] == [
        ('pedestrian', 'clear/day', 1), ('car', 'rain/night', 2)]
    # exposure weights the conditions
    s2 = _s(exposure={'clear/day': 1.0, 'rain/night': 3.0}, acquisition=s['acquisition'])
    out2 = G.rank_cells(rows, 0.5, s2, counts, fails)
    assert [r['condition'] for r in out2 if r.get('rank')][0] == 'rain/night'
    # a model weak everywhere has no gap to its overall AP; the reference "perfect" ranks the shortfall from 1
    flat = [dict(r, AP=0.0) for r in rows]
    assert not any(r.get('rank') for r in G.rank_cells(flat, 0.0, s, counts, fails))
    sp = _s(acquisition=dict(s['acquisition'], reference='perfect'))
    assert [r['rank'] for r in G.rank_cells(flat, 0.0, sp, counts, fails) if r.get('rank')] == [1, 2]
    # the budget shared in proportion to EV, one drive each first
    assert G.allocate([0.3, 0.1], 6) == [4, 2]
    assert G.allocate([0.1, 0.3, 0.2], 2) == [0, 1, 1]
    assert G.allocate([], 5) == [] and G.allocate([0.2], 0) == [0]
    assert G.round_drives(_s(budget={'drives': 8, 'frames': 200})) == 2          # 80 frames per free drive
    assert G.frames_per_drive(G.DEFAULTS['generate']) == 80


def test_the_stop_rule():
    up = {'diff': {'ci_low': 0.01, 'ci_high': 0.2}}
    flat = {'diff': {'ci_low': -0.05, 'ci_high': 0.2}}
    assert G.stop_reason([up, flat]) is None
    assert 'no chosen cell improved' in G.stop_reason([flat, {'diff': {'ci_low': -0.3, 'ci_high': -0.1}}])
    assert G.stop_reason([]) == 'no cell was chosen'


def test_conditions_weather_and_scenarios(project):
    from falcon_autonomy.core import oddcoverage as OC
    w = G.condition_weather('rain/night')
    assert OC.weather_class(w) == 'rain' and OC.light_class(w['sun_altitude_angle']) == 'night'
    with pytest.raises(G.GenError, match='weather/light'):
        G.condition_labels('snow/day')
    scn = G.scenario_for('pedestrian', 'fog/twilight', G.all_scenarios(project))
    assert scn['coverage_cell'].startswith('fog/') and scn['coverage_cell'].endswith('/twilight')
    assert any(a['kind'] == 'pedestrian' for a in scn['actors'])
    assert OC.weather_class(scn['weather']) == 'fog'
    assert G.scenario_for('car', 'clear/day', []) is None


def test_the_definition_is_checked(project):
    with pytest.raises(G.GenError) as e:
        G.normalise({'id': 'bad id', 'model': 'nope', 'world': 'nope', 'rig': 'rig-default',
                     'conditions': ['snow/day'], 'budget': {'drives': 0, 'frames': 1}}, project)
    msg = str(e.value)
    for want in ('id:', 'no model', 'no world', 'boxes2d', 'snow/day', 'drives: at least 1', 'budget.frames'):
        assert want in msg, (want, msg)
    syn = {'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
           'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector'}
    project.doc['models'] = [syn]
    with pytest.raises(G.GenError, match='cannot retrain'):
        G.normalise({'id': 'g', 'model': 'syn', 'world': 'world-town10', 'rig': 'rig-default'}, project)


# ------------------------------------------------------------------------------------------------ a whole round
@pytest.fixture(scope='module')
def genws(tmp_path_factory):
    pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    from falcon_autonomy.core import training as T
    from falcon_autonomy.core.project import Project
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('gen')
    p = Project.new('Gen')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=96, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.doc['traffic_profiles'][0].update(vehicles=10, walkers=10)
    p.save(str(base / 'gen.falcon.json'))
    ws = p.workspace
    srcs = []
    for name, seed in (('train-a', 11), ('train-b', 12)):
        out = os.path.join(ws, 'runs', name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        code, _, err = _run(_cfg(p, out, seconds=1.0, seed=seed))
        assert code == P.OK, err
        srcs.append({'kind': 'falcon', 'path': out})
    tdir = os.path.join(ws, 'training', 'tiny')
    T.train(dict(TINY, train=srcs[:1], val=srcs[1:]), tdir)
    reg = PV.read_json(os.path.join(tdir, 'model.json'))
    reg['id'] = 'tiny'
    p.doc['models'] = [reg]
    p.save()
    spec = {'id': 'g1', 'model': 'tiny', 'world': 'world-town10', 'rig': 'rig-default',
            'conditions': ['clear/day', 'fog/day'],
            'held_out': {'drives_per_condition': 2, 'drive_s': 1.0, 'fps': 20.0, 'record_every': 1, 'seed': 3,
                         'n_boot': 100},
            'budget': {'drives': 3, 'frames': 120},
            'generate': {'drive_s': 1.0, 'fps': 20.0, 'record_every': 4, 'scenario_share': 0.5,
                         'traffic_vehicles': 4, 'seed': 2},
            'acquisition': {'max_cells': 2, 'min_objects': 5, 'reference': 'perfect'}, 'rounds': 2}
    return {'path': p.path, 'ws': ws, 'spec': spec, 'srcs': srcs, 'base': base}


def _runner(path, settings_path, exp_id):
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    proj = Project.load(path)
    svc = CarlaService(Settings(str(settings_path)), JobQueue(SubprocessLauncher()), proj.workspace, fake_carla=FAKE)
    return proj, svc, G.Runner(proj, svc, exp_id)


def test_a_round_end_to_end(genws, tmp_path):
    from falcon_autonomy.core import ablation as AB
    from falcon_autonomy.core.project import Project
    ws = genws['ws']
    proj = Project.load(genws['path'])
    made = G.create(ws, genws['spec'], proj)
    assert made['id'] == 'g1' and os.path.isfile(os.path.join(made['path'], 'plan.json'))
    with pytest.raises(G.GenError, match='another definition'):         # fixed once registered
        G.create(ws, dict(genws['spec'], rounds=5), proj)
    proj, svc, r = _runner(genws['path'], tmp_path / 's.json', 'g1')
    try:
        st = r.run(1)
    finally:
        svc.shutdown()
    (rnd,) = st['rounds']
    assert rnd['status'] == 'done', st
    # the held-out set: fixed before round 1, hashed, 2 conditions × 2 drives
    ho = st['held_out']
    assert ho['drives'] == 4 and set(ho['conditions'].values()) == {'clear/day', 'fog/day'}
    assert PV.sha256_file(os.path.join(ws, ho['manifest'])) == ho['sha256']
    ev = PV.read_json(os.path.join(ws, ho['manifest']))
    held = {os.path.realpath(os.path.join(ws, d['run'])) for d in ev['drives']}
    held_keys = {d['key'] for d in ev['drives']}
    rdir = G.round_dir(ws, 'g1', 1)
    rank = PV.read_json(os.path.join(rdir, 'rank.json'))
    assert rank['rule'] == G.ACQUISITION and rank['chosen'] and len(rank['chosen']) <= 2
    assert sum(c['drives'] for c in rank['chosen']) == G.round_drives(G.normalise(genws['spec'], proj)) == 3
    assert all(c['n_objects'] >= 5 and c['ev'] > 0 for c in rank['chosen'])
    # n(c): the model's own training drive (clear/day) counts in its condition, not in fog
    by = {(c['class'], c['condition']): c['terms'] for c in rank['rows']}
    assert by[('car', 'clear/day')]['train_objects'] > 0 and by[('car', 'fog/day')]['train_objects'] == 0
    assert by[('car', 'clear/day')]['s'] < 1.0 == by[('car', 'fog/day')]['s']
    # generated within the budget, labelled from the ground truth, never a held-out drive
    gen = PV.read_json(os.path.join(rdir, 'generate.json'))
    assert gen['drives'] and gen['planned_frames'] <= 120 and gen['frames'] > 0
    assert {d['kind'] for d in gen['drives']} <= {'drive', 'scenario'}
    for d in gen['drives']:
        p = os.path.realpath(os.path.join(ws, d['run']))
        assert p not in held and d['key'] not in held_keys
        assert 'boxes2d' in PV.read_json(os.path.join(p, 'drive.json'))['rig']['sensors'][0]['labels']
    assert any(d['kind'] == 'scenario' for d in gen['drives']) and any(d['kind'] == 'drive' for d in gen['drives'])
    sd = next(d for d in gen['drives'] if d['kind'] == 'scenario')          # a scenario of the coverage fill
    cfg = PV.read_json(os.path.join(ws, sd['run'], 'drive.json'))
    assert cfg['scenario']['coverage_cell'] and cfg['traffic']['vehicles'] == 4
    # the dataset version, the retrain from the model's own settings plus the version
    ver = PV.read_json(os.path.join(rdir, 'version.json'))
    assert ver['runs'] == len(gen['drives'])
    rd = PV.read_json(os.path.join(rdir, 'round.json'))
    tr = PV.read_json(os.path.join(ws, rd['steps']['retrain']['result']['training'], 'run.json'))
    s = tr['provenance']['settings']
    assert s['arch'] == TINY['arch'] and s['init'] == 'scratch' and s['epochs'] == 1
    assert s['train'][0]['path'] == genws['srcs'][0]['path'] and s['train'][1]['kind'] == 'version'
    # before and after per cell, with intervals and the paired difference; the stop rule recorded
    cells = PV.read_json(os.path.join(rdir, 'cells.json'))
    assert cells['chosen'] and all(set(c) >= {'before', 'after', 'diff', 'verdict', 'ev'} for c in cells['chosen'])
    for c in cells['chosen']:
        assert c['before']['ci_low'] is not None and c['diff']['value'] is not None
        assert c['diff']['value'] == pytest.approx(c['after']['value'] - c['before']['value'], abs=1e-9)
    assert cells['headline']['before'] is not None and cells['headline']['unit'] == 'drive'
    assert (st['status'] == 'stopped') == (not cells['improved'])
    # the registry and the results
    res = PV.read_json(os.path.join(made['path'], 'results.json'))
    assert res['schema'] == G.RESULTS_SCHEMA and res['rounds'][0]['round'] == 1 and res['warnings']
    row = next(x for x in AB.registry(ws) if x['id'] == 'g1')
    assert row['kind'] == 'generation' and row['hypothesis'] == G.HYPOTHESIS and row['verdict'] == res['verdict']
    assert row['runs'] == {'ok': 1} and 'round-001' in os.listdir(os.path.join(made['path'], 'runs'))
    assert 'round 1' in G.table(res) and 'EV' in G.table(res)


def test_resume_and_the_guards(genws, tmp_path):
    ws = genws['ws']
    proj, svc, r = _runner(genws['path'], tmp_path / 's.json', 'g1')
    try:
        if r.st.get('stopped'):
            with pytest.raises(G.Stopped, match='stopped'):          # a stopped chain is not run again
                r.run()
            r.st['stopped'], r.st['status'] = None, 'running'        # (for the rest of this test only)
        # resume: round 1's steps are all done — nothing is queued again
        rd = PV.read_json(os.path.join(G.round_dir(ws, 'g1', 1), 'round.json'))
        _b, bout, _m = r.before(1)
        assert all(G.step_done(ws, rd, s, bout) for s in G.STEPS)
        r.run_round(1)
        assert len(svc.queue.jobs) == 0
        # the held-out set is checked against the hash taken when it was fixed
        man = os.path.join(ws, r.st['held_out']['manifest'])
        doc = PV.read_json(man)
        PV.write_json(man, dict(doc, note='edited'))
        with pytest.raises(G.GenError, match='held-out set changed'):
            r.evalset()
        PV.write_json(man, doc)
        r.st['held_out']['sha256'] = PV.sha256_file(man)
        r.evalset()
        # a training source that holds a held-out drive is refused
        from falcon_autonomy.core.project import Project
        p2 = Project.load(genws['path'])
        m = p2.item('models', 'tiny')
        trun = os.path.join(m['source']['training_run'], 'run.json')
        orig = PV.read_json(trun)
        bad = copy.deepcopy(orig)
        bad['provenance']['settings']['train'].append({'kind': 'falcon', 'path': os.path.join(
            ws, doc['drives'][0]['run'])})
        PV.write_json(trun, bad)
        try:
            with pytest.raises(L.LoopError, match='never trained on'):
                r.fix()
        finally:
            PV.write_json(trun, orig)
    finally:
        svc.shutdown()


def test_the_command_and_the_tool(genws, tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)

    def cli(*args):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'generate', *args], capture_output=True,
                              text=True, env=env, cwd=ROOT, timeout=300)
    path = genws['path']
    r = cli('plan', path, '--id', 'g2', '--model', 'tiny', '--world', 'world-town10', '--conditions',
            'clear/day,rain/night', '--budget-drives', '2', '--budget-frames', '200')
    assert r.returncode == 0 and 'EV(c)' in r.stdout and 'never trained on' in r.stdout, r.stdout + r.stderr
    r = cli('plan', path, '--id', 'g2', '--conditions', 'snow/day')
    assert r.returncode == P.FAILED and 'snow/day' in r.stdout
    r = cli('list', path)
    assert r.returncode == 0 and 'g1:' in r.stdout
    r = cli('show', path, '--id', 'g1', '--json')
    assert r.returncode == 0 and json.loads(r.stdout)['schema'] == G.RESULTS_SCHEMA
    r = cli('show', path, '--id', 'nope')
    assert r.returncode == P.FAILED and 'no experiment' in r.stdout
    # the assistant's tool: registers (the plan is the preview) and queues the run as a job
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    ctx = TOOLS.context(path, FAKE)
    try:
        spec = {'model': 'tiny', 'world': 'world-town10', 'rig': 'rig-default', 'conditions': ['clear/day'],
                'budget': {'drives': 2, 'frames': 400}}
        pv = REGISTRY['research_generate'].preview(ctx, name='g3', spec=spec)
        assert 'never trained on' in pv and 'EV(c)' in pv and 'falcon generate run' in pv
        out = call(ctx, 'research_generate', {'name': 'g3', 'spec': spec, 'rounds': 1})
        assert out['experiment'] == 'g3' and out['job'] and 'generate run' in out['command'], out
        q = ctx.service.queue
        q.cancel(next(j for j in q.jobs if j.id == out['job']))     # its rounds are the test above's
        again = call(ctx, 'research_generate', {'name': 'g1'})         # registered already: queued as it is
        assert again.get('experiment') == 'g1' or 'error' in again
        for j in q.jobs:
            q.cancel(j)
        assert 'snow/day' in call(ctx, 'research_generate', {'name': 'g4', 'spec': dict(
            spec, conditions=['snow/day'])})['error']
        assert 'error' in call(ctx, 'research_generate', {'name': '../x'})
        assert 'error' in call(ctx, 'research_generate', {'name': 'nope'})            # no spec, not registered
    finally:
        ctx.service.shutdown(grace_s=10.0)


# ------------------------------------------------------------------------------------------------ the page
@pytest.mark.gui
def test_the_generation_tab(genws, qapp, tmp_path):
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.ui import tasks
    from falcon_autonomy.ui.mainwindow import MainWindow
    from falcon_autonomy.ui.widgets import INTERVAL_ROLE
    from test_gui import pump
    win = MainWindow(Settings(str(tmp_path / 'settings.json')), Project.load(genws['path']), fake_carla=FAKE)
    win.show()
    pump(qapp, timeout=0.2)
    try:
        page = win.page('training')
        tab = page.generation
        assert page.tabs.tabText(page.tabs.indexOf(tab)) == 'Generation'
        tab.refresh(select='g1')
        assert tab.model.currentData() == 'tiny' and tab.exps.currentData() == 'g1'
        # the rounds of g1 (test_a_round_end_to_end): each chosen cell with its interval bar and verdict
        assert tab.tree.topLevelItemCount() >= 1 and 'round 1' in tab.text.toPlainText()
        v, lo, hi = tab.tree.topLevelItem(0).data(5, INTERVAL_ROLE)
        assert lo <= v <= hi and tab.bars.lo <= lo and hi <= tab.bars.hi
        tab.tree.grab()
        tab.exp_id.setText('gui-gen')
        tab.conditions.setText('clear/day,rain/night')
        assert tab.plan() is not None and 'EV(c)' in tab.text.toPlainText()
        c = tab.register()
        assert c['id'] == 'gui-gen' and tab.exps.currentData() == 'gui-gen'
        assert tab.text.toPlainText().startswith('registered gui-gen')
        tab.show()
        assert 'no round has run yet' in tab.text.toPlainText() and tab.tree.topLevelItemCount() == 0
        job = tab.run()
        assert job.spec.argv[3:5] == ['generate', 'run'] and job.spec.argv[6:8] == ['--id', 'gui-gen']
        assert '--fake-carla' in job.spec.argv
        win.queue.cancel(job)
    finally:
        win.shutdown()
        assert tasks.wait_all(30)
        pump(qapp, timeout=0.2)
        win._closing = True
        win.close()
        win.deleteLater()
        pump(qapp, timeout=0.2)
