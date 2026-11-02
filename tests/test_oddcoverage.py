"""Coverage of the operating domain (core.oddcoverage): the classes a run's config is placed in, the grid over a
workspace's studies, suites and runs, the stated rule (Clopper–Pearson, rule of three), the domain an ODD restricts
it to, failure-rate estimates in their cells, the plan that fills the gaps through the scenario and search code paths
(`falcon coverage`), and a gap filled with a sweep on the fake CARLA."""
import json
import os
import subprocess
import sys

import pytest

from falcon_autonomy.core import oddcoverage as OC
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import weather as W
from falcon_autonomy.core.drive import scenario_config


def _cfg(project, key, values=None, weather=None, seed=0):
    scn = SC.builtin(key)
    if weather:
        scn['weather'] = weather
    return scenario_config(project, scn, 'world-town10', 'rig-default', '/pending', 2000, 8000, seed=seed,
                           values=values)


def _run(ws, folder, cfg, passed=True):
    d = os.path.join(ws, folder)
    os.makedirs(d, exist_ok=True)
    cfg = dict(cfg, out=d)
    with open(os.path.join(d, 'drive.json'), 'w') as fh:
        json.dump(cfg, fh)
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok', 'scenario': {'id': cfg['scenario']['id'], 'passed': passed}}, fh)
    return d


def _study(ws, name, rows, threshold=0.5, kind='search'):
    d = os.path.join(ws, 'studies', name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'study.json'), 'w') as fh:
        json.dump({'name': name}, fh)
    if kind == 'search':
        with open(os.path.join(d, 'search-bayes.json'), 'w') as fh:
            json.dump({'threshold': threshold}, fh)
    with open(os.path.join(d, 'results.jsonl'), 'w') as fh:
        for run, obj in rows:
            fh.write(json.dumps({'status': 'ok', 'run': run, 'objective': obj}) + '\n')
    return d


# ------------------------------------------------------------------------------------------------ classes
@pytest.mark.parametrize('preset,weather,light', [
    ('ClearNoon', 'clear', 'day'), ('CloudyNoon', 'cloudy', 'day'), ('WetNoon', 'wet', 'day'),
    ('SoftRainNoon', 'rain', 'day'), ('HardRainNight', 'rain', 'night'), ('ClearSunset', 'clear', 'low_sun'),
    ('ClearNight', 'clear', 'night'), ('CloudyNight', 'fog', 'night'), ('DustStorm', 'dust', 'day')])
def test_weather_and_light_classes_of_carlas_presets(preset, weather, light):
    w = W.preset_values(preset)
    assert OC.weather_class(w) == weather
    assert OC.light_class(w['sun_altitude_angle']) == light


def test_each_cells_representative_weather_lands_in_its_cell():
    for k, fields in OC.CELL_WEATHER.items():
        assert OC.weather_class(fields) == k
    for k, sun in OC.CELL_SUN.items():
        assert OC.light_class(sun) == k
    assert OC.road_class(['straight', 'multi_lane', 'two_way']) == 'two_way'
    assert OC.speed_class(40.0) == '40-60' and OC.speed_class(130.0) == '80-130' and OC.speed_class(-1) is None


def test_a_runs_cells_from_its_config(project):
    cells, why = OC.run_cells(_cfg(project, 'jaywalker_behind_bus', {'speed': 30.0, 'distance': 40.0, 'walk': 1.5,
                                                                      'trigger': 20.0}))
    assert why == '' and sorted(cells) == ['clear/straight/bus/20-40/day', 'clear/straight/pedestrian/20-40/day']
    cells, _ = OC.run_cells(_cfg(project, 'cut_in', {'speed': 50.0, 'cutter_speed': 60.0, 'start_offset': 0.0,
                                                     'cut_gap': 5.0, 'cut_time': 2.0},
                                 weather={'preset': 'MidRainyNight'}))
    assert cells == ['rain/multi_lane/car/40-60/night']
    assert OC.run_cells({'weather': {}})[1] == 'no scenario'


# ------------------------------------------------------------------------------------------------ the grid
@pytest.fixture
def ws(project, tmp_path):
    """A workspace with a search study (lead_brake at 30 km/h: 70 runs, one failing), a sweep of cut-in at 50 km/h
    in rain at night (5 runs) and a single pedestrian-crossing run that failed."""
    w = project.workspace
    lb = [(_run(w, f'studies/s1/runs/r{i}', _cfg(project, 'lead_brake', {'speed': 30.0, 'gap': 10 + i * 0.1,
                                                                         'decel': 5.0, 'brake_at': 3.0})),
           0.9 if i == 0 else 0.1) for i in range(70)]
    _study(w, 's1', lb)
    ci = [(_run(w, f'studies/s2/runs/r{i}', _cfg(project, 'cut_in', {'speed': 50.0, 'cutter_speed': 60.0,
                                                                     'start_offset': 0.0, 'cut_gap': 5.0,
                                                                     'cut_time': 2.0 + i * 0.1},
                                                  weather={'preset': 'MidRainyNight'})), 0.2) for i in range(5)]
    _study(w, 's2', ci, kind='sweep')
    _run(w, 'runs/ped', _cfg(project, 'pedestrian_crossing', {'speed': 25.0, 'distance': 30.0, 'walk': 1.2,
                                                              'trigger': 20.0}), passed=False)
    os.makedirs(os.path.join(w, 'runs', 'free-drive'))
    with open(os.path.join(w, 'runs', 'free-drive', 'run.json'), 'w') as fh:
        json.dump({'status': 'ok'}, fh)                          # a drive without a scenario: not a test
    return w


def test_the_grid_counts_runs_failures_and_applies_the_rule(project, ws):
    cov = OC.coverage(ws, project)
    assert cov['runs'] == 76 and cov['domain']['cells'] == 6 * 4 * 13 * 5 * 4
    lb = cov['cells']['clear/straight/car/20-40/day']
    assert (lb['runs'], lb['failures'], lb['sources']) == (70, 1, {'search': 70})
    assert lb['status'] == 'too_few' and lb['upper'] > 0.05 and lb['needed'] == OC.FR.runs_needed(0.05, 0.95, 1) - 70
    rain = cov['cells']['rain/multi_lane/car/40-60/night']
    assert rain['status'] == 'too_few' and rain['needed'] == 59 - 5 and rain['sources'] == {'sweep': 5}
    ped = cov['cells']['clear/straight/pedestrian/20-40/day']
    assert ped['status'] == 'above_target' and ped['failures'] == 1           # one run, failed: lower bound 0.05+
    assert cov['counts']['untested'] == cov['domain']['cells'] - 3
    loose = OC.coverage(ws, project, target=0.1)
    assert loose['cells']['clear/straight/car/20-40/day']['status'] == 'enough'
    pv = OC.pivot(cov, 'weather', 'light')
    clear_day = pv['grid'][pv['row_labels'].index('clear')][pv['col_labels'].index('day')]
    assert clear_day['runs'] == 71 and clear_day['failures'] == 2 and clear_day['cell_status']['untested'] == 258
    by_actor = OC.pivot(cov, 'actor', 'speed', where={'weather': 'clear'})
    car20 = by_actor['grid'][by_actor['row_labels'].index('car')][by_actor['col_labels'].index('20-40')]
    assert car20['runs'] == 70 and car20['cells'] == 4 * 4
    with pytest.raises(ValueError):
        OC.pivot(cov, 'weather', 'weather')


def test_an_odd_restricts_the_domain_and_estimates_appear_in_their_cells(project, ws):
    from falcon_autonomy.core.safetycase import example_odd
    odd = example_odd()
    with open(os.path.join(ws, 'odd.json'), 'w') as fh:
        json.dump(odd, fh)
    d = os.path.join(ws, 'studies', 'rate-lead_brake')
    os.makedirs(d)
    cell = 'clear/straight/car/20-40/day'
    with open(os.path.join(d, 'failure-rate.json'), 'w') as fh:
        json.dump({'schema': OC.FR.SCHEMA, 'scenario': 'lead_brake', 'estimate': {'status': 'ok', 'p': 1e-3},
                   'by_cell': {cell: {'runs': 40, 'failures': 12, 'p': 2e-3, 'ci95': [1e-3, 4e-3], 'status': 'ok'},
                               'fog/straight/car/20-40/day': {'runs': 3, 'failures': 0, 'p': 0.0,
                                                              'status': 'refused'}}}, fh)
    cov = OC.coverage(ws, project)
    dims = cov['domain']['dims']
    assert dims['actor'] == odd['dimensions']['actors']['values'] and 'night' not in dims['light']
    assert dims['speed'] == ['0-20', '20-40', '40-60'] and 'fog' not in dims['weather']
    assert cov['outside_domain'] == 5                       # the rain-at-night sweep lies outside this ODD
    assert cov['cells'][cell]['estimates'][0]['p'] == 2e-3
    assert 'fog/straight/car/20-40/day' not in cov['cells']    # an estimate outside the domain is left out
    assert cov['estimates'][0]['study'] == 'rate-lead_brake'


# ------------------------------------------------------------------------------------------------ filling the gaps
def test_fill_plan_generates_a_scenario_per_cell_through_the_scenario_code(project, ws):
    cov = OC.coverage(ws, project)
    scns = [SC.builtin(k) for k in SC.LIBRARY]
    plan = OC.fill_plan(cov, scns, 4)
    first = plan['plan'][0]
    assert first['status'] == 'too_few' and first['cell'] == 'rain/multi_lane/car/40-60/night'   # largest shortfall
    assert plan['plan'][1]['cell'] == 'clear/straight/car/20-40/day'
    assert plan['plan'][2]['status'] == 'untested'
    assert plan['no_scenario_cells'] > 0 and plan['candidates'] > 4
    for e in plan['plan']:
        scn = e['scenario']
        assert SC.validate(scn) == [] and scn['coverage_cell'] == e['cell']
        cfg = scenario_config(project, scn, 'world-town10', 'rig-default', '/x', 2000, 8000, seed=0)
        assert e['cell'] in OC.run_cells(cfg)[0]               # a run of it lands in its cell
        lo, hi = (float(x) for x in e['cell'].split('/')[3].split('-'))
        sp = scn['params']['speed']
        assert lo <= sp['lo'] < sp['hi'] <= hi
    untested = plan['plan'][2]['cell'].split('/')
    tested = [c.split('/') for c, x in cov['cells'].items() if x['runs']]
    assert min(sum(a != b for a, b in zip(untested, t)) for t in tested) == 1      # grows out from what was run
    n0 = len(project.doc['scenarios'])
    OC.apply_plan(project, plan)
    OC.apply_plan(project, plan)                               # idempotent
    assert len(project.doc['scenarios']) == n0 + 4
    argv = OC.plan_argv(project.path, first, 'world-town10', 'rig-default')
    assert argv[:2] == ['search', project.path] and 'bayes-pof' in argv and first['scenario']['id'] in argv
    sweep = OC.fill_plan(cov, scns, 1, 'sweep', budget=3)['plan'][0]
    assert OC.plan_argv(project.path, sweep, 'w', 'r')[:1] == ['sweep'] and sweep['runs'] == 3
    # a generated scenario is never the base of another
    assert OC.scenario_for_cell(first['cell'], [first['scenario']]) is None


def test_coverage_commands(project, ws):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'coverage', 'show', project.path, '--cells'],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert '76 scenario runs' in r.stdout and 'clear/straight/car/20-40/day' in r.stdout and 'too_few' in r.stdout
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'coverage', 'fill', project.path, '--max', '2',
                        '--apply', '--json'], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout[:r.stdout.rindex('}') + 1])
    assert len(plan['plan']) == 2 and plan['plan'][0]['argv'][0] == 'search'
    from falcon_autonomy.core.project import Project
    assert sum(1 for s in Project.load(project.path).doc['scenarios'] if s.get('coverage_cell')) == 2


def test_a_gap_filled_with_a_sweep_on_the_fake_carla(project, tmp_path, fake_carla):
    project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    project.save()
    odd = {'schema': 'falcon.odd/1', 'name': 'pedestrians on straight roads',
           'dimensions': {'road_type': {'kind': 'categorical', 'values': ['straight']},
                          'actors': {'kind': 'set', 'values': ['pedestrian']},
                          'sun_altitude_deg': {'kind': 'range', 'lo': 30, 'hi': 90},
                          'precipitation': {'kind': 'range', 'lo': 0, 'hi': 5},
                          'cloudiness': {'kind': 'range', 'lo': 0, 'hi': 10}}}
    os.makedirs(project.workspace, exist_ok=True)
    with open(os.path.join(project.workspace, 'odd.json'), 'w') as fh:
        json.dump(odd, fh)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'coverage', 'fill', project.path, '--max', '1',
                        '--kind', 'sweep', '--budget', '2', '--run', '--fake-carla', fake_carla],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    from falcon_autonomy.core.project import Project
    cov = OC.coverage(project.workspace, Project.load(project.path))
    assert cov['domain']['cells'] == 1 * 1 * 1 * 5 * 1 and cov['runs'] == 2
    (cid, cell), = cov['cells'].items()
    assert cid.startswith('clear/straight/pedestrian/20-40/day') and cell['runs'] == 2
    assert cell['sources'] == {'sweep': 2} and cell['status'] in ('too_few', 'above_target')
