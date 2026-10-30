"""Live trigger rules (core.triggers, workers.triggers): the language and its errors, once / every / cooldown, the
project format, rules firing in drives, scenario runs and Live against the fake — spawns, weather, faults, markers —
recorded in frames.jsonl and run.json, and repeating with the seed."""
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time

import pytest
from PyQt6.QtCore import Qt

from conftest import worker_env
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import triggers as T
from falcon_autonomy.core.drive import drive_config, scenario_config
from falcon_autonomy.core.project import Project, ProjectError, validate


# ------------------------------------------------------------------------------------------------- the language
def test_the_example_rule_parses_into_its_condition_and_action():
    p = T.parse_rule('when ego.speed > 40 and visibility < 30 -> spawn jaywalker')
    assert p['when'] == ['and', ['cmp', 'ego.speed', '>', 40.0], ['cmp', 'visibility', '<', 30.0]]
    assert p['then'] == [{'do': 'spawn', 'what': 'jaywalker', 'args': {}}]
    assert p['signals'] == ['ego.speed', 'visibility']


def test_and_binds_tighter_than_or_and_units_are_checked():
    p = T.parse_rule('when actor.ttc < 2 s or ego.light:red and ego.light_m < 25m -> marker "close call"; '
                     'weather fog_density=40')
    assert p['when'][0] == 'or' and p['when'][2][0] == 'and'
    assert p['then'][0] == {'do': 'marker', 'text': 'close call'}
    assert p['then'][1] == {'do': 'weather', 'preset': None, 'fields': {'fog_density': 40.0}}
    assert T.parse_rule('when ego.speed > 40 km/h -> marker')['when'] == ['cmp', 'ego.speed', '>', 40.0]
    assert T.parse_rule('when tod:low sun and not (rain > 10) -> marker')['when'][1] == ['cmp', 'tod', '=',
                                                                                         'low sun']


@pytest.mark.parametrize('rule, where, what', [
    ('ego.speed > 40 -> marker', 'column 1', 'starts with "when"'),
    ('when ego.sped > 40 -> marker', 'column 6', 'unknown signal'),
    ('when ego.speed 40 -> marker', 'column 16', 'comparison'),
    ('when ego.speed > 40s -> marker', 'column 18', 'km/h, not s'),
    ('when (ego.speed > 40 -> marker', 'column 21', '"(" is at column 6'),
    ('when ego.speed > 40 and -> marker', 'column 24', 'ends too soon'),
    ('when ego.speed > 40', 'column 20', '"->" and an action are missing'),
    ('when tod > 3 -> marker', 'column 10', 'is text'),
    ('when ego.light:blue -> marker', 'column 16', 'red, yellow'),
    ('when time > 1 -> spawn unicorn', 'column 18', 'cannot spawn'),
    ('when time > 1 -> spawn jaywalker distance=500', 'column 18', 'outside 3…120'),
    ('when time > 1 -> spawn jaywalker speed=3', 'column 18', 'takes distance=, walk='),
    ('when time > 1 -> fault cam-front mount_rotation', 'column 18', 'cannot start mid-run'),
    ('when time > 1 -> fault cam-front dirt cover=0.9', 'column 18', 'outside 0–0.5'),
    ('when time > 1 -> weather Sunny', 'column 18', 'unknown weather preset'),
    ('when time > 1 -> weather fog_density=150', 'column 18', 'outside 0…100'),
    ('when time > 1 -> fly away', 'column 18', 'unknown action'),
])
def test_errors_say_where_and_why(rule, where, what):
    with pytest.raises(T.RuleError) as e:
        T.parse_rule(rule)
    assert str(e.value).startswith(where + ':') and what in str(e.value), str(e.value)


def test_validation_of_rule_documents():
    rules = [T.new_rule('when time > 1 -> marker', 'r1'),
             T.new_rule('when fired.r1 >= 1 -> fault cam-front dirt cover=0.2', 'r2', repeat='cooldown'),
             T.new_rule('when fired.nope > 0 -> fault lidar-top noise sigma=3', 'r3'),
             T.new_rule('when time > 1 -> marker', 'r1')]
    probs = T.validate_rules(rules, {'cam-front': 'camera.pinhole', 'lidar-top': 'lidar'})
    assert "rule id 'r1' used twice" in probs
    assert any('r2' in p and 'cooldown_s > 0' in p for p in probs)
    assert any('fired.nope names no rule' in p for p in probs)
    assert any('noise fault cannot act on lidar-top' in p for p in probs)
    assert T.validate_rules(rules[:1], {}) == []


# ------------------------------------------------------------------------------------------------- repeat semantics
def _fire_times(repeat, cooldown, series, dt=0.5):
    e = T.Engine([{'id': 'r', 'rule': 'when ego.speed > 10 -> marker', 'repeat': repeat, 'cooldown_s': cooldown}])
    out = []
    for k, v in enumerate(series):
        if e.step(k * dt, lambda s: v):
            out.append(k * dt)
    return out


SERIES = [0, 20, 20, 0, 20, 0, 20, 20, 20, 20, 20, 20]       # true at 0.5–1.0, 2.0, 3.0–5.5 s


def test_once_every_and_cooldown():
    assert _fire_times('once', 0, SERIES) == [0.5]
    assert _fire_times('every', 0, SERIES) == [0.5, 2.0, 3.0]              # each rising edge
    assert _fire_times('every', 2.0, SERIES) == [0.5, 3.0]                 # the edge at 2.0 s is within 2 s of 0.5
    assert _fire_times('cooldown', 1.0, SERIES) == [0.5, 2.0, 3.0, 4.0, 5.0]   # while true, once per second


def test_fired_counts_and_unreadable_signals():
    e = T.Engine([{'id': 'a', 'rule': 'when time >= 1 -> marker', 'repeat': 'every'},
                  {'id': 'b', 'rule': 'when fired.a >= 1 and ego.light:red -> marker'},
                  {'id': 'c', 'rule': 'when not (actor.ttc < 2) -> marker'}])
    vals = {'time': 0.0, 'ego.light': None, 'actor.ttc': math.inf}
    assert [r['id'] for r, _, _ in e.step(0.0, vals.get)] == ['c']        # inf: no collision course
    vals.update(time=1.0)
    assert [r['id'] for r, _, _ in e.step(1.0, vals.get)] == ['a']        # b sees fired.a after the tick
    vals.update(time=1.5, **{'ego.light': 'red'})
    fired = e.step(1.5, vals.get)
    assert [r['id'] for r, _, _ in fired] == ['b'] and fired[0][2] == {'fired.a': 1, 'ego.light': 'red'}


# ------------------------------------------------------------------------------------------------- the project
def test_rules_in_the_project_schema_and_attached_to_configs(project, tmp_path):
    project.doc['trigger_rules'] = [
        T.new_rule('when time > 1 -> marker one', 'r1', applies=['drives']),
        T.new_rule('when time > 2 -> marker two', 'r2', applies=['scenarios', 'live']),
        dict(T.new_rule('when time > 3 -> marker off', 'r3'), enabled=False)]
    assert validate(project.doc) == []
    project.save(str(tmp_path / 'p.falcon.json'))
    back = Project.load(str(tmp_path / 'p.falcon.json'))
    assert [r['id'] for r in back.doc['trigger_rules']] == ['r1', 'r2', 'r3']
    bad = dict(project.doc, trigger_rules=[{'id': 'x', 'rule': 'when t > 1 -> marker', 'repeat': 'sometimes'}])
    assert any('repeat' in e for e in validate(bad))
    cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'd'), 2310, 8310)
    plain = json.dumps(cfg, sort_keys=True)
    T.attach(cfg, project, 'drives')
    assert [r['id'] for r in cfg['trigger_rules']['rules']] == ['r1'] and cfg['trigger_rules']['context'] == 'drives'
    scn = scenario_config(project, SC.builtin('lead_brake'), 'world-town10', 'rig-default', str(tmp_path / 's'),
                          2310, 8310, trigger_rules='project')
    assert [r['id'] for r in scn['trigger_rules']['rules']] == ['r2']
    assert 'trigger_rules' not in scenario_config(project, SC.builtin('lead_brake'), 'world-town10', 'rig-default',
                                                  str(tmp_path / 's'), 2310, 8310)
    project.doc['trigger_rules'] = []                      # no rule: the config (and its hash) as before
    cfg2 = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'd'), 2310, 8310)
    assert json.dumps(T.attach(cfg2, project, 'drives'), sort_keys=True) == plain
    project.doc['trigger_rules'] = [T.new_rule('when time > 1 -> fault cam-back dirt cover=0.2', 'r9')]
    with pytest.raises(ProjectError, match="fault on 'cam-back', which the rig does not have"):
        T.attach(drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'd'), 2310, 8310), project,
                 'drives')


def test_the_cli_adds_checks_lists_and_removes(project, tmp_path):
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'triggers', *a], capture_output=True,
                              text=True, timeout=60)
    r = cli('add', path, 'when ego.speed > 40 and visibility < 30 -> spawn jaywalker', '--id', 'jw')
    assert r.returncode == 0 and r.stdout.strip() == 'jw', r.stderr
    r = cli('add', path, 'when ego.speed > 40 -> spawn unicorn')
    assert r.returncode == 2 and 'cannot spawn' in r.stderr
    assert cli('check', path).returncode == 0
    r = cli('list', path)
    assert 'jw' in r.stdout and 'spawn jaywalker' in r.stdout
    assert 'ego.junction_m' in cli('signals').stdout
    assert cli('remove', path, 'jw').returncode == 0
    assert Project.load(path).doc['trigger_rules'] == []


# ------------------------------------------------------------------------------------------------- in the worker
def _drive(project, tmp_path, rules, name='run', seconds=4.0, seed=3, scenario=None, values=None, faults=None):
    project.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=['semantic'])
    if faults is not None:
        project.doc['rigs'][0]['sensors'][0]['faults'] = faults
    project.doc['trigger_rules'] = rules
    if scenario:
        cfg = scenario_config(project, SC.builtin(scenario), 'world-town10', 'rig-default', str(tmp_path / name),
                              2310, 8310, seed=seed, values=values, trigger_rules='project')
    else:
        cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / name), 2310, 8310,
                           seconds=seconds, seed=seed)
        T.attach(cfg, project, 'drives')
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=240)
    run = json.loads((tmp_path / name / 'run.json').read_text())
    frames = [json.loads(x) for x in (tmp_path / name / 'frames.jsonl').read_text().splitlines()] \
        if (tmp_path / name / 'frames.jsonl').exists() else []
    return r, run, frames


def test_a_drive_fires_spawns_changes_weather_and_starts_a_fault(project, tmp_path):
    rules = [T.new_rule('when time >= 1 -> spawn jaywalker distance=15 walk=1.5; marker "walker out"', 'jw'),
             T.new_rule('when time >= 1.5 and fog < 10 -> weather fog_density=70 wetness=60', 'fog'),
             T.new_rule('when visibility < 200 -> fault cam-front dirt cover=0.3 for=1', 'mud'),
             T.new_rule('when ego.speed >= 0 -> marker tick', 'tick', repeat='cooldown', cooldown_s=1.0),
             T.new_rule('when time >= 2 -> spawn cut_in behind=6 faster=20 gap=3', 'cut')]
    r, run, frames = _drive(project, tmp_path, rules)
    assert r.returncode == P.OK, r.stderr[-3000:]
    tr = run['trigger_rules']
    assert tr['fired'] == {'jw': 1, 'fog': 1, 'mud': 1, 'tick': 4, 'cut': 1}
    by = {f['rule']: f for f in tr['firings']}
    assert by['jw']['t'] == pytest.approx(1.0, abs=0.06) and by['jw']['actions'][1]['did'] == 'marker: walker out'
    assert by['jw']['actions'][0]['actor'] == 'jw-jaywalker-1' and tr['spawned']['jw-jaywalker-1'] > 0
    assert by['mud']['t'] > by['fog']['t'] and by['mud']['signals']['visibility'] == pytest.approx(100.0)
    assert 'error' not in by['cut']['actions'][0], by['cut']
    # on the frame whose state made the condition true; the fault from the next frame, for 1 s
    fi = next(f for f in frames if f.get('triggers') and f['triggers'][0]['rule'] == 'mud')
    onset = run['faults_applied']['cam-front']['started'][0]['onset_s']
    dirty = [f['sim_t'] for f in frames if f['sim_t'] >= onset - 1e-6 and f['sim_t'] < onset + 1.0 - 1e-6]
    assert onset == pytest.approx(fi['sim_t'] + 0.05, abs=1e-6) and run['faults_applied']['cam-front']['frames'][
        'dirt'] == len(dirty)
    # the walker walks (kinematic, from the rule's own generator), recorded with criticality
    traj = json.loads((tmp_path / 'run' / 'trajectories.json').read_text())['actors']
    walker = traj['jw-jaywalker-1']
    assert len(walker) > 10 and abs(walker[-1][2] - walker[0][2]) > 1.0
    assert tr['criticality'] is not None and 'jw-jaywalker-1' in tr['criticality']['actors']
    assert run['provenance']['settings']['trigger_rules']['rules'][0]['id'] == 'jw'


def test_scenario_runs_with_rules_repeat_bit_for_bit(project, tmp_path):
    rules = [T.new_rule('when actor.nearest_m < 25 -> spawn cattle distance=30', 'cow'),
             T.new_rule('when ego.speed > 20 -> marker fast', 'fast', repeat='every'),
             T.new_rule('when actor.ttc < 4 -> marker ttc', 'ttc', repeat='cooldown', cooldown_s=0.5)]
    vals = {'speed': 40.0, 'gap': 20.0, 'decel': 6.0, 'brake_at': 2.0}
    ra, a, fa = _drive(project, tmp_path, rules, 'a', scenario='lead_brake', values=vals, seed=4)
    rb, b, fb = _drive(project, tmp_path, rules, 'b', scenario='lead_brake', values=vals, seed=4)
    assert ra.returncode == rb.returncode == P.OK, ra.stderr[-3000:]
    ta = json.loads((tmp_path / 'a' / 'trajectories.json').read_text())
    tb = json.loads((tmp_path / 'b' / 'trajectories.json').read_text())
    assert ta == tb and 'cow-cattle-1' in ta['actors']               # the rule's cow is driven by the director
    strip = lambda fs: [{k: v for k, v in f['triggers'][0].items() if k != 'actions'} for f in fs
                        if f.get('triggers')]
    assert strip(fa) == strip(fb) and a['trigger_rules']['fired'] == b['trigger_rules']['fired']
    assert a['trigger_rules']['fired']['cow'] == 1 and a['trigger_rules']['fired']['ttc'] >= 1
    assert a['scenario']['added'][0]['id'] == 'cow-cattle-1' and 'cow-cattle-1' in a['scenario']['criticality'][
        'actors']


def test_an_action_that_cannot_be_done_says_why_and_the_run_goes_on(project, tmp_path):
    with pytest.raises(T.RuleError):                       # out of range: refused before the run
        T.parse_rule('when time >= 0.5 -> spawn bus lane=opposite ahead=500')
    # seed 0: the ego drives the fake's lane −1, whose overtaking side is the oncoming lane — no lane to cut in from
    rules = [T.new_rule('when time >= 0.5 -> spawn cut_in; marker after', 'cut'),
             T.new_rule('when ego.junction_m < 200 -> marker junction', 'junction')]
    r, run, frames = _drive(project, tmp_path, rules, seconds=1.5, seed=0)
    assert r.returncode == P.OK and run['status'] == 'ok', r.stderr[-3000:]
    acts = run['trigger_rules']['firings'][0]['actions']
    assert 'no adjacent lane' in acts[0]['error'] and acts[1]['did'] == 'marker: after'
    assert run['trigger_rules']['spawned'] == {}
    assert run['trigger_rules']['fired']['junction'] == 0           # the fake's straight road has no junction


def test_light_distance_and_state_from_the_stop_line(project, tmp_path):
    rules = [T.new_rule('when ego.light:green and ego.light_m < 150 -> marker green light', 'g')]
    r, run, frames = _drive(project, tmp_path, rules, seconds=1.0, seed=0)     # seed 0: in a +x lane, x ≈ −32
    assert r.returncode == P.OK, r.stderr[-3000:]
    f = run['trigger_rules']['firings'][0]
    ego = frames[f['step']]['ego']
    assert f['signals']['ego.light'] == 'green' and abs(ego['yaw']) < 30 and ego['x'] < 8.0
    # the fake's stop line is at x = 8; measured from the ego's front (half its length ahead of its centre)
    assert 8.0 - ego['x'] - 3.0 < f['signals']['ego.light_m'] < 8.0 - ego['x'] - 1.5


# ------------------------------------------------------------------------------------------------- Live
@pytest.fixture
def gui_socket(tmp_path):
    from falcon_autonomy.core.livelink import Reader
    path = str(tmp_path / 'live.sock')
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    got = {'msgs': []}

    def serve():
        conn, _ = srv.accept()
        r = Reader()
        while True:
            try:
                b = conn.recv(1 << 20)
            except OSError:
                break
            if not b:
                break
            got['msgs'] += r.feed(b)
    threading.Thread(target=serve, daemon=True).start()
    yield path, got
    srv.close()


def test_rules_fire_in_live_as_its_events(project, tmp_path, gui_socket):
    from falcon_autonomy.core.live import live_config
    path, got = gui_socket
    project.doc['trigger_rules'] = [T.new_rule('when time >= 0.5 -> spawn jaywalker distance=20', 'jw'),
                                    T.new_rule('when time >= 0.8 -> fault cam-front freeze for=0.5', 'fz'),
                                    T.new_rule('when time >= 0.6 -> weather fog_density=50', 'fog')]
    cfg = live_config(project, 'world-town10', 'rig-default', str(tmp_path / 'live'), 2310, 8310, path,
                      str(tmp_path / 'recs'), mode='autopilot', view={'kind': 'chase', 'width': 64, 'height': 48,
                                                                      'fov': 90.0}, trigger_rules='project')
    cp = tmp_path / 'live.json'
    cp.write_text(json.dumps(cfg))
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'live', str(cp)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60 and sum(1 for h, _ in got['msgs'] if h['t'] == 'frame') < 40:
            time.sleep(0.05)
            if p.poll() is not None:
                break
        p.stdin.write(json.dumps({'cmd': 'stop'}) + '\n')
        p.stdin.flush()
        p.wait(60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == P.OK, p.stderr.read()[-3000:]
    run = json.load(open(tmp_path / 'live' / 'run.json'))
    kinds = [e['kind'] for e in run['events']]
    assert kinds.count('trigger') == 3 and 'sensor_fail' in kinds
    assert run['trigger_rules']['fired'] == {'jw': 1, 'fz': 1, 'fog': 1}
    sf = next(e for e in run['events'] if e['kind'] == 'sensor_fail')
    assert sf['detail']['mode'] == 'freeze' and sf['detail']['by'] == 'trigger rule'


# ------------------------------------------------------------------------------------------------- the GUI
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)


@pytest.mark.gui
def test_the_scenarios_page_edits_rules(window, qapp):
    page = window.go('scenarios')
    rt = page.rules_tab
    page.tabs.setCurrentWidget(rt)
    rt.add_rule()
    assert len(window.project.doc['trigger_rules']) == 1
    col = rt.COLS.index
    rt.table.item(0, col('rule')).setText('when ego.speed > 40 and visibility < 30 -> spawn jaywalker')
    assert window.project.doc['trigger_rules'][0]['rule'].endswith('spawn jaywalker')
    assert rt.problems.text() == ''
    rt.table.item(0, col('rule')).setText('when ego.sped > 40 -> spawn jaywalker')
    assert 'unknown signal' in rt.problems.text()
    window.undo.undo()
    assert rt.problems.text() == '' and 'ego.speed' in window.project.doc['trigger_rules'][0]['rule']
    rt.table.item(0, col('repeat')).setText('cooldown')
    assert 'cooldown_s > 0' in rt.problems.text()
    rt.table.item(0, col('cooldown_s')).setText('2')
    assert window.project.doc['trigger_rules'][0]['cooldown_s'] == 2.0 and rt.problems.text() == ''
    rt.table.item(0, col('applies')).setText('drives, live')
    assert window.project.doc['trigger_rules'][0]['applies'] == ['drives', 'live']
    rt.table.item(0, col('on')).setCheckState(Qt.CheckState.Unchecked)
    assert window.project.doc['trigger_rules'][0]['enabled'] is False
    rt.table.setCurrentCell(0, col('rule'))
    rt.remove_rule()
    assert window.project.doc['trigger_rules'] == []
