"""Formal traffic-rule checking (core.stl, core.rules): the STL evaluator against robustness worked out by hand on
synthetic signals (every operator, non-uniform sampling, finite-trace windows, unknown samples); the formula and the
rule-file languages (precedence, errors with their column or line, nothing evaluated as Python); the signals of a
synthetic drive on a synthetic map (an OpenDRIVE road and its lanes.json) — lanes beside, margins, solid lines, the
lead, a pass on the left, a pedestrian on a crosswalk, the map's speed limits and junction roads — and the India and
generic sets on it, with the violation intervals' times; drives recorded against the fake carla; a gym episode's trace
with a red light; the Leaderboard, stack suites, the safety case and the HTML report with rules; the command; the Data
page's timeline and the Leaderboard tab."""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (GUI fixtures; Qt slot exceptions fail tests)

from falcon_autonomy.core import rules as RU
from falcon_autonomy.core import stl as S

T10 = np.round(np.arange(0, 1.01, 0.1), 10)
X = np.array([1, 2, 3, -1, 0, 2, np.nan, 3, 4, 5, 6.0])


def ev(text, sig=None, t=None, consts=None):
    return S.evaluate(S.parse(text, consts), sig if sig is not None else {'x': X}, T10 if t is None else t)


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and np.all((np.isnan(a) & np.isnan(b)) | np.isclose(a, b)), (a, b)


# ================================================================================================ STL by hand
def test_predicates_boolean_and_arithmetic_by_hand():
    same(ev('x >= 0'), X)
    same(ev('x <= 2'), 2 - X)
    same(ev('x > 1'), X - 1)                                       # strictness is not represented
    same(ev('not x >= 0'), -X)
    same(ev('2 * x - 1 >= abs(x - 3)'), 2 * X - 1 - np.abs(X - 3))
    same(ev('max(x, 2) <= min(x + 5, 4)'), [2, 2, 1, 2, 2, 2, np.nan, 1, 0, -1, -2])
    same(ev('default(x, 9) <= 5'), [4, 3, 2, 6, 5, 3, -4, 2, 1, 0, -1])
    same(ev('flag', {'flag': [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0.0]}), [-.5, .5, .5, -.5, .5, -.5, -.5, .5, .5, .5, -.5])
    a, b = X - 1, 2 - X                                          # x >= 1 and x <= 2
    same(ev('x >= 1 and x <= 2'), [0, 0, -1, -2, -1, 0, np.nan, -1, -2, -3, -4])
    same(ev('x >= 1 or x <= 2'), np.where(np.isnan(X), np.nan, np.maximum(a, b)))
    same(ev('x >= 1 -> x <= 2'), np.where(np.isnan(X), np.nan, np.maximum(-a, b)))
    same(ev('true'), np.full(11, np.inf))
    same(ev('false or x >= 0'), X)


def test_bounded_temporal_operators_by_hand():
    # G[0,0.2]: the minimum over the sample and the next two; the window is cut at the end
    same(ev('G[0,0.2] x >= 0'), [1, -1, -1, -1, np.nan, np.nan, np.nan, 3, 4, 5, 6])
    same(ev('F[0,0.2] x >= 1'), [2, 2, 2, 1, 1, 2, 3, 4, 5, 5, 5])
    same(ev('historically[0,0.2] x >= 0'), [1, 1, 1, -1, -1, -1, np.nan, np.nan, np.nan, 3, 4])
    same(ev('once[0.1,0.2] x >= 0'), [-np.inf, 1, 2, 3, 3, 0, 2, 2, 3, 4, 5])
    same(ev('always x >= 2'), [-3, -3, -3, -3, -2, np.nan, np.nan, 1, 2, 3, 4])       # G = G[0, inf]
    # G over an empty window (past the end) is +inf, F is −inf
    same(ev('G[2,3] x >= 0'), np.full(11, np.inf))
    same(ev('F[2,3] x >= 0'), np.full(11, -np.inf))
    # until: ψ (x ≥ 4) within [0, 0.3] and φ (x ≥ 0) at every sample before it
    same(ev('x >= 0 until[0,0.3] x >= 4'), [-1, -1, -1, -1, np.nan, np.nan, np.nan, 2, 2, 2, 2])
    # since: ψ (x ≥ 3) within [0, 0.2] back and φ (x ≥ 0) at every sample after it
    same(ev('x >= 0 since[0,0.2] x >= 3'), [-2, -1, 0, -1, -1, -1, np.nan, 0, 1, 2, 3])


def test_non_uniform_sampling_and_constants():
    t = np.array([0.0, 0.5, 0.6, 2.0, 2.05, 4.0])
    v = np.array([5.0, 1.0, 3.0, -2.0, 4.0, 0.0])
    # windows are in seconds, not samples: [t, t + 1.0]
    same(S.evaluate(S.parse('G[0,w] v >= 0', {'w': 1.0}), {'v': v}, t), [1, 1, 3, -2, 4, 0])
    same(S.evaluate(S.parse('F[1,2] v >= 3'), {'v': v}, t), [-5, 1, 1, -3, -3, -np.inf])
    with pytest.raises(S.STLError, match='increase'):
        S.evaluate(S.parse('v >= 0'), {'v': v}, t[::-1])


def test_unknown_samples_are_three_valued():
    u = np.array([np.nan, np.nan, np.nan])
    t = np.array([0.0, 1.0, 2.0])
    k = {'u': u, 'a': np.array([-1.0, 2.0, np.nan])}
    same(S.evaluate(S.parse('u >= 0 and a >= 0'), k, t), [-1, np.nan, np.nan])     # false ∧ unknown = false
    same(S.evaluate(S.parse('u >= 0 or a >= 0'), k, t), [np.nan, 2, np.nan])       # true ∨ unknown = true
    same(S.evaluate(S.parse('G[0,1] a >= 0'), k, t), [-1, np.nan, np.nan])
    same(S.evaluate(S.parse('F[0,1] a >= 0'), k, t), [2, 2, np.nan])
    assert S.intervals(S.evaluate(S.parse('a >= 0'), k, t), t) == [
        {'start': 0.0, 'end': 0.0, 'i0': 0, 'i1': 0, 'min_robustness': -1.0, 'at': 0.0}]


def test_violation_intervals():
    rho = np.array([1, -1, -3, 2, 0, -0.5, -0.1, np.nan, -2])
    t = np.arange(9) * 0.5
    iv = S.intervals(rho, t, 0.5)
    assert [(x['start'], x['end'], x['min_robustness'], x['at']) for x in iv] == [
        (0.5, 1.5, -3.0, 1.0), (2.5, 3.5, -0.5, 2.5), (4.0, 4.5, -2.0, 4.0)]


def test_parser_precedence_errors_and_no_python():
    n = S.parse('a >= 1 or b >= 1 and not c >= 1 -> G[0,2] d < e')
    assert n.op == '->' and n.args[0].op == 'or' and n.args[0].args[1].op == 'and'
    assert S.parse('(a + b) * 2 >= 1').args[0].op == '*'
    assert S.parse('(a >= 1)').op == 'cmp' and S.parse('not (a >= 1 and b)').args[0].op == 'and'
    assert S.names(S.parse('a until[0,1] once b >= c')) == {'a', 'b', 'c'}
    assert S.horizon(S.parse('G[0,2] (x >= 0 and H[0,3] y >= 0)')) == (3.0, 2.0)
    assert S.parse(S.parse('x >= 1 -> F[0,1] y').text()).text() == S.parse('x >= 1 -> F[0,1] y').text()
    for bad, where in (('x >= ', 'column 6'), ('x >= 1 )', 'column 8'), ('G[2,1] x >= 0', '0 <= a <= b'),
                       ('x $ 1', "'$'"), ('min(x) >= 0', 'min() takes'), ('', 'empty')):
        with pytest.raises(S.STLError) as e:
            S.parse(bad)
        assert where in str(e.value), (bad, str(e.value))
    # nothing is evaluated as Python: these are unknown signals, a syntax error, never a call
    with pytest.raises(S.STLError):
        S.parse("__import__('os').system('true') >= 0")
    n = S.parse('__class__ >= 0')
    with pytest.raises(S.STLError, match='unknown signal'):
        S.evaluate(n, {'x': X}, T10)



# ================================================================================================ rule files
def test_rule_file_language():
    text = '''# a test set
ruleset test "Test rules"
handedness LHT
about "A set for the tests."
let gap_s = 2   # heuristic: the two-second rule
rule follow "Keep a gap" heuristic:
    speed_mps < 1 or
      lead_gap_m >= gap_s * speed_mps
about follow "Two seconds."
rule fast "Not too fast" law: speed_kmh <= 50
'''
    rs = RU.parse_ruleset(text, 'mine.rules')
    assert rs['id'] == 'test' and rs['name'] == 'Test rules' and rs['handedness'] == 'LHT'
    assert rs['consts'] == {'gap_s': 2.0} and rs['notes'] == {'gap_s': 'heuristic: the two-second rule'}
    f = rs['rules'][0]
    assert f['id'] == 'follow' and f['tags'] == ['heuristic'] and f['about'] == 'Two seconds.'
    assert f['formula'] == 'speed_mps < 1 or lead_gap_m >= gap_s * speed_mps'
    assert f['signals'] == ['lead_gap_m', 'speed_mps'] and rs['rules'][1]['tags'] == ['law']
    for bad, msg in (('rule a "A": speed_kmh <= 50\nrule a "B": speed_kmh <= 40', 'line 2: rule a is defined twice'),
                     ('rule a "A": nonsense_signal >= 1', 'unknown signal(s) nonsense_signal'),
                     ('rule a "A": speed_kmh <=', 'line 1: rule a: expected'),
                     ('let speed_mps = 3\nrule a "A": t >= 0', 'line 1: speed_mps is a signal'),
                     ('let x = abc', 'line 1: let NAME = NUMBER'), ('handedness up', 'LHT or RHT'),
                     ('  speed_kmh <= 1', 'indented line'), ('import os', "unknown statement 'import'"),
                     ('about b "x"\nrule a "A": t >= 0', 'about for unknown rule'), ('# nothing', 'no rules')):
        with pytest.raises(RU.RulesError, match=re.escape(msg)):
            RU.parse_ruleset(bad, 'bad.rules')
    for k in RU.BUILTIN:
        s = RU.load_ruleset(k)
        assert s['rules'] and all(r['tags'] for r in s['rules']), k
    assert RU.load_ruleset(None, 'LHT')['id'] == 'india' and RU.load_ruleset(None, 'RHT')['id'] == 'generic'
    ids = {r['id'] for r in RU.load_ruleset('india')['rules']}
    assert {'keep_left', 'overtake_right', 'speed_limit', 'stop_at_red', 'yield_crossing', 'no_wrong_way',
            'following_distance', 'junction_clear', 'lane_discipline'} <= ids
    assert {'keep_right', 'overtake_left'} <= {r['id'] for r in RU.load_ruleset('generic')['rules']}
    about = {r['id']: r['about'] for r in RU.load_ruleset('india')['rules']}['overtake_right']   # continued lines
    assert 'turning right and signalling; slow queues' in about


# ================================================================================================ a synthetic world
W = 3.5
LANES = ((-1, W / 2, 0.0), (-2, 1.5 * W, 0.0), (1, -W / 2, 180.0))     # CARLA y of each lane's centre, heading


def synth_xodr(cross=True):
    """Road 1 along x from −100 (s 0) to 100: right lanes −1, −2 (+s), left lane 1, 3.5 m; a town speed of 50 km/h
    from s 0 and 30 from s 120; a crosswalk at s 110 across the road. Road 2 (x 100 to 120) is a junction road."""
    lane = '<lane id="{i}" type="driving" level="false"><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>'
    sec = ('<lanes><laneSection s="0"><left>' + lane.format(i=1) + '</left><center><lane id="0" type="none"/>'
           '</center><right>' + lane.format(i=-1) + lane.format(i=-2) + '</right></laneSection></lanes>')
    cw = ('<objects><object id="9" type="crosswalk" s="110" t="0" hdg="1.5707963267948966" length="10" width="3">'
          '<outline><cornerLocal u="-5" v="-1.5" z="0"/><cornerLocal u="5" v="-1.5" z="0"/><cornerLocal u="5" v="1.5" '
          'z="0"/><cornerLocal u="-5" v="1.5" z="0"/></outline></object></objects>') if cross else ''
    return ('<?xml version="1.0"?><OpenDRIVE><header revMajor="1" revMinor="4"/>'
            '<road id="1" junction="-1" length="200"><type s="0" type="town"><speed max="50" unit="km/h"/></type>'
            '<type s="120" type="town"><speed max="30" unit="km/h"/></type>'
            '<planView><geometry s="0" x="-100" y="0" hdg="0" length="200"><line/></geometry></planView>'
            + sec + cw + '</road>'
            '<road id="2" junction="7" length="20"><planView><geometry s="0" x="100" y="0" hdg="0" length="20"><line/>'
            '</geometry></planView>' + sec + '</road></OpenDRIVE>')


def synth_lanes():
    """lanes.json of the same roads, as CARLA writes it (markings: the centre line solid double, between the two +x
    lanes broken, the kerbs solid)."""
    marks = {-1: ('SolidSolid', 'Broken'), -2: ('Broken', 'Solid'), 1: ('SolidSolid', 'Solid')}
    pts = []
    for road, x0, n in ((1, -100.0, 100), (2, 100.0, 10)):
        for k in range(n):
            x = x0 + 2.0 * k
            for lid, y, yaw in LANES:
                pts.append([x, y, 0.0, yaw, W, road, 0, lid, x - x0] + [marks[lid][0], 'White', marks[lid][1],
                                                                          'White'])
    return {'schema': 'falcon.lanes/1', 'step_m': 2.0, 'graph': [],
            'columns': ['x', 'y', 'z', 'yaw', 'width', 'road', 'section', 'lane', 's', 'left_type', 'left_color',
                        'right_type', 'right_color'], 'points': pts}


def synth_trace(tmp_path=None):
    """The ego in lane −1 (y 1.75) at 10 m/s from x −50 for 10 s; a lead 30 m ahead at the same speed; a slower car in
    lane −2 that the ego passes at 3.95 s (on its left: the car is on the ego's right); a pedestrian standing on the
    crosswalk (x 10) in the ego's lane."""
    t = np.round(np.arange(0, 10.001, 0.1), 6)
    x = -50.0 + 10.0 * t
    tr = RU.Trace(t, x, np.full_like(t, W / 2), np.zeros_like(t), np.full_like(t, 10.0))
    acts = []
    for i, ti in enumerate(t):
        acts.append([{'id': 'lead', 'x': x[i] + 30.0, 'y': W / 2, 'yaw': 0.0, 'vx': 10.0, 'vy': 0.0, 'l': 4.0, 'w': 1.8,
                      'kind': 'vehicle'},
                     {'id': 'slow', 'x': -30.25 + 5.0 * ti, 'y': 1.5 * W, 'yaw': 0.0, 'vx': 5.0, 'vy': 0.0, 'l': 4.0,
                      'w': 1.8, 'kind': 'vehicle'},
                     {'id': 'ped', 'x': 10.0, 'y': W / 2, 'yaw': 90.0, 'vx': 0.0, 'vy': 0.0, 'l': 0.5, 'w': 0.5,
                      'kind': 'pedestrian'}])
    tr.actors = acts
    tr.map.add_lanes_json(synth_lanes())
    tr.map.add_xodr(synth_xodr(), 'synthetic')
    return tr


def test_map_from_opendrive_and_lanes_json():
    od = RU.RoadMap()
    od.add_xodr(synth_xodr(), 'synthetic', make_lanes=True)
    lj = RU.RoadMap()
    lj.add_lanes_json(synth_lanes())
    assert RU._xodr_agrees(synth_xodr(), lj) == 1.0                    # the lanes made from OpenDRIVE are CARLA's
    k = {key: i for i, key in enumerate(od.key) if od.s[i] == 10.0}
    assert od.P[k[(1, 0, -1)]].tolist() == pytest.approx([-90.0, 1.75]) and od.yaw[k[(1, 0, -1)]] == 0.0
    assert od.P[k[(1, 0, 1)]].tolist() == pytest.approx([-90.0, -1.75]) and abs(od.yaw[k[(1, 0, 1)]]) == 180.0
    lj.add_xodr(synth_xodr(), 'synthetic')
    junction = {lj.key[i][0]: lj.junction[i] for i in range(len(lj.key))}
    assert junction == {1: 0.0, 2: 1.0}
    sp = {(lj.key[i][0], lj.s[i]): lj.speed[i] for i in range(len(lj.key))}
    assert sp[(1, 118.0)] == 50.0 and sp[(1, 120.0)] == 30.0 and np.isnan(sp[(2, 0.0)])   # a junction road: none
    cw = lj.crosswalks[0]
    assert len(lj.crosswalks) == 1 and cw[:, 0].min() == pytest.approx(8.5) and cw[:, 0].max() == pytest.approx(11.5)
    assert cw[:, 1].min() == pytest.approx(-5.0) and cw[:, 1].max() == pytest.approx(5.0)
    assert RU._point_in(cw, 10.0, 1.75) and not RU._point_in(cw, 12.0, 1.75)
    other = RU.RoadMap()
    other.add_lanes_json({**synth_lanes(), 'points': [[p[0] + 40.0, p[1] + 9.0] + p[2:] for p in
                                                      synth_lanes()['points']]})
    assert RU._xodr_agrees(synth_xodr(), other) < RU.XODR_AGREE_SHARE   # another map: refused
    # the heuristic without OpenDRIVE: a road crossing two others is a junction road
    h = RU.RoadMap()
    pts = [[x, 0.0, 0.0, 0.0, 3.5, 1, 0, -1, 0.0] for x in np.arange(-20, 20, 2.0)]
    pts += [[0.0, y, 0.0, 90.0, 3.5, 5, 0, -1, 0.0] for y in np.arange(-20, 20, 2.0)]
    pts += [[2.0, y, 0.0, 90.0, 3.5, 6, 0, -1, 0.0] for y in np.arange(-20, 20, 2.0)]
    h.add_lanes_json({'columns': ['x', 'y', 'z', 'yaw', 'width', 'road', 'section', 'lane', 's'], 'points': pts})
    h.junction_heuristic()
    assert {h.key[i][0]: h.junction[i] for i in range(len(h.key))} == {1: 1.0, 5: 0.0, 6: 0.0}


def test_signals_of_a_synthetic_drive_by_hand():
    tr = synth_trace()
    sig = RU.signals(tr)
    t = tr.t
    hw, hl = RU.EGO_DEFAULT[1] / 2, RU.EGO_DEFAULT[0] / 2
    assert np.all(sig['on_lane'] == 1) and np.all(sig['in_junction'] == 0)
    assert np.all(sig['lanes_left'] == 0) and np.all(sig['lanes_right'] == 1)
    assert np.allclose(sig['lane_margin_m'], W / 2 - hw) and np.allclose(sig['solid_margin_m'], W / 2 - hw)
    assert np.allclose(sig['lane_heading_err_deg'], 0.0)
    assert np.allclose(sig['lead_gap_m'], 30.0 - hl - 2.0)
    i = int(np.argmax(sig['pass_on_left']))
    assert t[i] == 4.0 and sig['pass_on_left'].sum() == 1 and sig['pass_on_right'].sum() == 0
    assert sig['passing'][i] == 1
    ahead = 10.0 - tr.x > 0
    g = np.maximum(10.0 - tr.x - hl - 0.25, 0.0)
    gap = np.where(ahead & (g <= RU.PED_RANGE_M), g, np.inf)              # up to 50 m ahead
    seen = np.isfinite(gap)
    assert np.allclose(sig['ped_path_gap_m'], gap) and np.all(sig['ped_on_crossing'][seen] == 1)
    assert np.all(sig['ped_on_crossing'][~seen] == 0)
    lim = sig['speed_limit_kmh']
    assert np.all(lim[t < 6.95] == 50.0) and np.all(lim[t >= 7.0] == 30.0)
    assert tr.sources['speed_limit'].startswith('synthetic')


def test_rule_sets_on_the_synthetic_drive():
    gen = RU.check(synth_trace(), RU.load_ruleset('generic'))
    by = {r['id']: r for r in gen['rules']}
    assert gen['distance_km'] == pytest.approx(0.1) and gen['samples'] == 101 and gen['step_s'] == pytest.approx(0.1)
    sl = by['speed_limit']
    assert sl['state'] == 'violated' and sl['count'] == 1 and sl['robustness'] == pytest.approx(-6.0)
    assert sl['violations'][0]['start'] == 7.0 and sl['violations'][0]['end'] == pytest.approx(10.1)
    assert sl['per_km'] == pytest.approx(10.0)
    yc = by['yield_crossing']        # stopping takes v²/8 = 12.5 m + 2 m of margin: the gap 7.3 − x < 14.5 from x −7.2
    assert yc['count'] == 1 and yc['violations'][0]['start'] == pytest.approx(4.3)        # (x = −50 + 10 t)
    assert yc['violations'][0]['end'] == pytest.approx(6.0)        # past the pedestrian (x = 10) it is not ahead
    assert by['following_distance']['state'] == 'satisfied'
    assert by['following_distance']['robustness'] == pytest.approx(30.0 - 2.45 - 2.0 - 20.0)
    assert by['keep_right']['state'] == 'satisfied'           # out of the kerb lane, but within 10 s of a pass
    assert by['overtake_left']['state'] == 'satisfied'
    assert by['stop_at_red']['state'] == 'not evaluable' and by['stop_at_red']['missing_signals'] == [
        'tl_distance_m', 'tl_red']
    assert by['no_wrong_way']['robustness'] == pytest.approx(135.0)
    ind = RU.check(synth_trace(), RU.load_ruleset('india'))
    ob = {r['id']: r for r in ind['rules']}
    o = ob['overtake_right']                                  # India: the pass on the left is a violation, at 4.0 s
    assert o['state'] == 'violated' and o['count'] == 1 and o['violations'][0]['start'] == 4.0
    assert o['robustness'] == -0.5 and ob['keep_left']['state'] == 'satisfied'
    assert ob['pedestrian_clearance']['count'] == 1
    # without the pass, keeping out of the kerb lane is a violation for the whole drive
    tr = synth_trace()
    tr.actors = [[a for a in acts if a['id'] != 'slow'] for acts in tr.actors]
    kr = {r['id']: r for r in RU.check(tr, RU.load_ruleset('generic'))['rules']}['keep_right']
    assert kr['count'] == 1 and kr['violations'][0]['start'] == 0.0 and kr['violated_s'] == pytest.approx(10.1)
    # a handedness mismatch is said
    tr = synth_trace()
    tr.handedness = 'RHT'
    assert any('LHT' in w for w in RU.check(tr, RU.load_ruleset('india'))['warnings'])


# ================================================================================================ recorded runs
@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    from falcon_autonomy.core import protocol as P
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.weather import resolve
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('rules-drive')
    p = Project.new('Rules')
    p.doc['rigs'][0]['sensors'][0].update(width=96, height=64, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.doc['traffic_profiles'][0]['vehicles'] = 8
    code, _, err = _run(_cfg(p, str(base / 'd'), seconds=2.0, seed=3, weather=resolve({'preset': 'ClearNoon'})))
    assert code == P.OK, err
    return str(base / 'd')


def test_check_a_recorded_drive(drive):
    res = RU.check_run(drive, 'generic', xodr='')
    assert res['schema'] == RU.SCHEMA and res['samples'] >= 30 and res['distance_km'] > 0
    by = {r['id']: r for r in res['rules']}
    assert by['no_wrong_way']['evaluated_share'] == 1.0 and by['no_wrong_way']['state'] == 'satisfied'
    assert by['lane_discipline']['evaluated_share'] == 1.0
    assert res['sources']['actors'] == 'actors.jsonl' and res['sources']['lanes'].startswith('lanes.json')
    assert 'heuristic' in res['sources']['junction']                # no OpenDRIVE: the junction heuristic, said so
    assert by['speed_limit']['evaluated_share'] == 1.0                # the set's default limit, the map has none
    assert any('OpenDRIVE' in n for n in res['notes'])
    # the town's own OpenDRIVE file is checked against the run's lanes and refused when it is another map
    xp = os.path.join(os.path.dirname(drive), 'other.xodr')
    with open(xp, 'w') as fh:
        fh.write(synth_xodr())
    tr = RU.trace_from_run(drive, carla_root='/nonexistent')
    assert 'xodr' not in tr.sources or tr.sources['xodr'] != xp
    cfg = RU._run_config(drive)                                        # run.json's settings: the drive's config
    assert RU.find_xodr({'xodr': xp})[0] == xp and RU.find_xodr({'town': cfg['town']}, '/nonexistent')[0] is None
    tr = RU.trace_from_run(drive, xodr=xp)
    assert any('does not match' in n for n in tr.notes) and 'xodr' not in tr.sources
    # timeline rows for the UI
    rows = RU.timeline(res)
    assert len(rows) == len(res['rules']) and all(len(r) == 4 for r in rows)


def test_speed_from_positions_when_the_recorded_speed_is_wrong(tmp_path):
    d = tmp_path / 'old'
    d.mkdir()
    with open(d / 'frames.jsonl', 'w') as fh:
        for k in range(40):
            fh.write(json.dumps({'step': k, 'sim_t': k * 0.05, 'speed_mps': 0.0,
                                 'ego': {'x': 0.5 * k, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}}) + '\n')
    tr = RU.trace_from_run(str(d), xodr='')
    assert np.allclose(tr.speed, 10.0) and any('positions' in n for n in tr.notes)
    res = RU.check(tr, RU.load_ruleset('generic'))
    assert {r['id']: r for r in res['rules']}['no_wrong_way']['state'] == 'not evaluable'     # no lanes


def test_cli(drive, tmp_path):
    from conftest import worker_env
    rf = tmp_path / 'mine.rules'
    rf.write_text('ruleset mine "Mine"\nrule slow "Slow" heuristic: speed_kmh <= 1\n')

    def cli(*args):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'rules'] + list(args), capture_output=True,
                              text=True, env=worker_env(), timeout=120)
    r = cli('check', drive, drive, '--set', str(rf), '--xodr', 'none', '--out', str(tmp_path / 'out'))
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'slow' in r.stdout and 'violated' in r.stdout and 'mine: 2 runs' in r.stdout
    doc = json.loads((tmp_path / 'out' / 'rules.json').read_text())
    assert doc['schema'] == 'falcon.rules_results/1' and len(doc['results']) == 2
    agg = doc['aggregates']['mine']
    assert agg['runs'] == 2 and agg['rules'][0]['runs_violating'] == 2
    assert agg['rules'][0]['share_ci95'][1] == pytest.approx(1.0)
    assert cli('list').stdout.count('keep_left') == 1 and 'overtake_left' in cli('list').stdout
    assert cli('show', 'india').stdout == RU.INDIA and 'lead_gap_m' in cli('signals').stdout
    r = cli('check', drive, '--set', 'nonsense')
    assert r.returncode == 2 and 'no rule set' in r.stdout
    with pytest.raises(RU.RulesError, match='different rule sets'):
        RU.aggregate([RU.check_run(drive, 'india', ''), RU.check_run(drive, 'generic', '')])
    # the HTML report of the check
    from falcon_autonomy.core import reports as R
    out = str(tmp_path / 'rules.html')
    meta = R.write_report('rules', str(tmp_path / 'out'), out)
    html = open(out).read()
    assert meta['kind'] == 'rules' and 'Traffic rules: Mine on 2 run(s)' in html and '<svg' in html
    assert 'speed_kmh &lt;= 1' in html and 'violated' in html and '://' not in html
    assert R.read_meta(out)['inputs'] and 'turn indicators' in html


# ============================================================================================ gym, Leaderboard, stack
@pytest.fixture
def svc(tmp_path, fake_carla):
    from falcon_autonomy.core.gymenv import make_service
    from falcon_autonomy.core.settings import Settings
    s = make_service(Settings(str(tmp_path / 'settings.json')), fake_carla, str(tmp_path / 'ws'),
                     str(tmp_path / 'locks'))
    yield s
    s.shutdown()


def test_gym_trace_red_light_and_the_leaderboard(project, svc, tmp_path, monkeypatch):
    from falcon_autonomy.core import leaderboard as LB
    from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config
    monkeypatch.setenv('FAKE_CARLA_LIGHT_STATE', 'Red')
    cfg = gym_config(project, 'world-town10', str(tmp_path / 'env'), svc.port, svc.tm_port, start_speed_kmh=30.0,
                     target_speed_kmh=30.0, max_episode_s=10.0)
    env = FalconDriveEnv(cfg, svc, start_timeout_s=60.0, step_timeout_s=30.0)
    try:
        for seed in range(40):
            obs, info = env.reset(seed=seed)
            if info['start']['yaw'] == 0.0 and info['start']['x'] < 0.0:
                break
        term = trunc = False
        while not (term or trunc):
            obs, r, term, trunc, info = env.step([0.0, 0.0])
        path = env.episodes_path
    finally:
        env.close()
    eps = [json.loads(x) for x in open(path)]
    e = next(x for x in eps if x['seed'] == seed)
    assert e['trace']['columns'] == list(RU.TRACE_COLUMNS) and len(e['trace']['rows']) == e['steps']
    red = next(i for i in e['infractions'] if i['kind'] == 'red_light')
    res = RU.check_episode(e, 'generic')
    by = {x['id']: x for x in res['rules']}
    s = by['stop_at_red']
    assert s['state'] == 'violated' and s['evaluated_share'] > 0.9
    assert abs(s['violations'][0]['end'] - red['sim_t']) <= 0.2       # it ends where the gym counted the red light
    assert by['no_wrong_way']['state'] == 'satisfied' and by['lane_discipline']['evaluated_share'] == 1.0
    assert by['overtake_left']['state'] == 'not evaluable'           # no per-actor poses in a gym trace
    assert res['sources']['junction'] == 'CARLA Waypoint.is_junction'
    doc = LB.score([e, dict(e, trace=None)], n_boot=50)
    assert doc['rules']['routes_checked'] == 1 and doc['rules']['violations'] >= 1
    assert doc['per_route'][0]['rule_violations'] >= 1 and doc['per_route'][1]['rule_violations'] is None
    assert LB.score([dict(e, trace=None)], n_boot=20)['rules'] is None
    # the Leaderboard report shows them beside the score
    from falcon_autonomy.core import reports as R
    from falcon_autonomy.core.provenance import write_json
    lb = tmp_path / 'lb'
    lb.mkdir()
    write_json(str(lb / 'leaderboard.json'), dict(doc, failed=[], policy=None, provenance={}))
    R.write_report('leaderboard', str(lb), str(tmp_path / 'lb.html'))
    assert 'Traffic rules (beside the score, not in it)' in open(tmp_path / 'lb.html').read()


def test_stack_suite_rules_and_the_safety_case(drive, tmp_path):
    import shutil
    from falcon_autonomy.core import safetycase as SCase
    from falcon_autonomy.core import stack as ST
    from falcon_autonomy.core.provenance import write_json
    suite = tmp_path / 'suite'
    runs = []
    for k in range(3):
        d = suite / 'runs' / f'r{k}'
        shutil.copytree(drive, d)
        write_json(str(d / 'stack.json'), RU._run_config(drive))      # a stack run keeps its config as stack.json
        runs.append({'key': f'k{k}', 'entry': 0, 'scenario': 'lead_brake', 'scenario_name': 'Lead brakes', 'seed': k,
                     'run': str(d), 'status': 'ok', 'passed': True, 'checks': [], 'values': {}, 'criticality': 0.1,
                     'min_ttc_s': 5.0, 'collision': False, 'missed_deadlines': 0})
    with open(suite / 'results.jsonl', 'w') as fh:
        fh.writelines(json.dumps(r) + '\n' for r in runs)
    doc = ST.new_suite('rules', scenarios=('lead_brake',))
    doc['rules'] = str(tmp_path / 'slow.rules')
    (tmp_path / 'slow.rules').write_text('ruleset slow "Slow"\nrule slow "Slow" heuristic: speed_kmh <= 1\n')
    assert ST.validate_suite(doc) == [] and ST.validate_suite(dict(doc, rules='mars')) != []
    write_json(str(suite / 'suite.json'), doc)
    summ = ST.rule_summary(runs, doc['rules'])
    assert summ['runs'] == 3 and summ['rules'][0]['runs_violating'] == 3 and summ['scenarios']['lead_brake']['runs'] \
        == 3 and summ['rules'][0]['per_km_ci95'] is not None
    assert (suite / 'runs' / 'r0' / 'rules.json').is_file()
    rep = SCase.build_report(str(suite), SCase.example_odd(), str(tmp_path / 'case'))
    assert rep['traffic_rules']['runs'] == 3
    assert any(r['kind'] == 'traffic_rule' for r in rep['residual_risks'])
    html = (tmp_path / 'case' / 'safety-case.html').read_text()
    assert '<h2>Traffic rules</h2>' in html and 'Slow' in html


# ================================================================================================ the GUI
def synthetic_episode(n=40, limit=30.0):
    """A gym episode with a hand-made trace (TRACE_COLUMNS): 10 m/s in a lane whose CARLA limit is `limit` km/h."""
    rows = [[0.05 * (k + 1), 0.5 * (k + 1), 1.75, 0.0, 10.0, 1, 0, 0.0, 0.0, 3.5, 0, 1, 'SolidSolid', 'Broken', None,
             None, 0, None, limit] for k in range(n)]
    return {'episode': 0, 'seed': 1, 'route_length_m': 100.0, 'route_completion': 0.2, 'distance_m': 0.5 * n,
            'infraction_counts': {}, 'infractions': [], 'termination': 'time_limit', 'motion': 'kinematic',
            'trace': {'columns': list(RU.TRACE_COLUMNS), 'rows': rows, 'ego': {'length': 4.9, 'width': 1.84},
                      'handedness': 'RHT', 'town': 'Town10HD_Opt'}}


def test_episode_trace_by_hand():
    e = synthetic_episode()
    tr = RU.trace_from_episode(e)
    assert tr.t[0] == pytest.approx(0.05) and tr.kind == 'episode' and tr.handedness == 'RHT'
    sig = RU.signals(tr)
    assert np.allclose(sig['lane_margin_m'], 1.75 - 0.92) and np.allclose(sig['solid_margin_m'], 1.75 - 0.92)
    assert np.all(np.isinf(sig['lead_gap_m'])) and np.all(sig['speed_limit_kmh'] == 30.0)
    assert np.all(sig['tl_red'] == 0) and np.isnan(sig['pass_on_left']).all()
    r = {x['id']: x for x in RU.check_episode(e, 'generic')['rules']}
    assert r['speed_limit']['robustness'] == pytest.approx(-6.0) and r['speed_limit']['violations'][0]['start'] == \
        pytest.approx(0.05)
    assert r['keep_right']['state'] == 'not evaluable'                  # out of the kerb lane, passing unknown
    assert r['stop_at_red']['state'] == 'satisfied'
    with pytest.raises(RU.RulesError, match='no trace'):
        RU.check_episode(dict(e, trace=None))


def test_data_page_timeline_and_leaderboard_tab(window, qapp, drive):
    from falcon_autonomy.core import leaderboard as LB
    from falcon_autonomy.core import reports as R
    from falcon_autonomy.core.provenance import write_json
    page = window.go('data')
    page.show_run(drive)
    panel = page.browse.rules
    assert pump(qapp, lambda: panel.result is not None, 30)
    assert panel.tree.topLevelItemCount() == len(panel.result['rules'])
    assert panel.timeline.result is panel.result and 'violation' in panel.status.text()
    page.browse.seek_time(1.0)
    pump(qapp, timeout=0.3)
    assert abs(panel.timeline.t - 1.0) <= 0.06
    panel.timeline.grab()                                        # paints
    panel.set_box.setCurrentIndex(panel.set_box.findData('india'))
    assert pump(qapp, lambda: panel.result is not None and panel.result['ruleset']['id'] == 'india', 30)
    panel.save()
    saved = os.path.join(window.workspace(), 'rules', 'd-india', 'rules.json')
    assert os.path.isfile(saved)
    assert any(p == os.path.dirname(saved) for _, p in R.sources(window.workspace())['rules'])
    # the Leaderboard tab: the rule violations beside the score, per route
    e = synthetic_episode()
    lb = os.path.join(window.workspace(), 'leaderboard', '20261029-120000-test')
    os.makedirs(lb)
    write_json(os.path.join(lb, 'leaderboard.json'), dict(LB.score([e], n_boot=20), failed=[], policy=None,
                                                          provenance={}))
    with open(os.path.join(lb, 'episodes.jsonl'), 'w') as fh:
        fh.write(json.dumps(e) + '\n')
    tp = window.go('training')
    tp.load_leaderboards()
    tp.lbs.setCurrentIndex(tp.lbs.findData(lb))
    pump(qapp, timeout=0.2)
    text = tp.lb_info.vals['Traffic rules'].text()
    assert '1 violations' in text and 'speed_limit 1' in text
    assert tp.infr.topLevelItem(0).text(6) == '1'
