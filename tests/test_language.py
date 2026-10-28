"""Language labels (core.language): captions and QA from hand-made states with known answers, grounding spans,
seeded determinism, the paraphrase hook's checks, and end to end on runs the drive worker records against the fake
carla — a traffic drive (test_worldmodel's run fixture) and a lead_brake scenario. Every actors.jsonl is the drive
worker's own; nothing is synthesised.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import worker_env
from falcon_autonomy.core import language as L
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import worldmodel as WM
from test_worldmodel import run_dir  # noqa: F401  (the module-scoped recorded drive)

COLS = ['x', 'y', 'z', 'yaw', 'width', 'road', 'section', 'lane', 's', 'left_type', 'left_color', 'right_type',
        'right_color']


def _lanes():
    """A straight road along world x: lanes −1 (y 0) and −2 (y 3.5) heading +x, lane 1 (y −3.5) heading −x."""
    pts = []
    for lane, y, yaw in ((-1, 0.0, 0.0), (-2, 3.5, 0.0), (1, -3.5, 180.0)):
        for x in range(-100, 101, 2):
            pts.append([float(x), y, 0.0, yaw, 3.5, 1, 0, lane, float(x), 'Broken', 'White', 'Solid', 'White'])
    return {'columns': COLS, 'points': pts, 'step_m': 2.0}


def _row(aid, x, y, yaw=0.0, vx=0.0, vy=0.0, ext=(2.0, 0.9, 0.75), cat='vehicle.car', role=''):
    return {'id': aid, 'x': x, 'y': y, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw, 'vx': vx, 'vy': vy, 'vz': 0.0,
            'type_id': 'x', 'category': cat, 'role': role, 'extent': list(ext), 'bbox_center': [0.0, 0.0, ext[2]]}


EGO = _row(1, 0.0, 0.0, vx=10.0, ext=(2.4, 1.0, 0.75), role='hero')
PED = ('human.pedestrian.adult', (0.2, 0.2, 0.9))


def _scene():
    lead = _row(10, 25.0, 0.0, vx=5.0)                                  # in the ego's lane, 5 m/s slower
    ped = _row(11, 12.0, 3.0, ext=PED[1], cat=PED[0])                   # in the lane to the right, standing
    onc = _row(12, -15.0, -3.5, yaw=180.0, vx=-8.0)                     # behind on the left, oncoming lane
    return L.build_state(5, 0.25, EGO, [lead, ped, onc], _lanes(), frame=105)


def _qa(d, kind, ids=None):
    return [q for q in d['qa'] if q['type'] == kind and (ids is None or q['ids'] == ids)]


# --------------------------------------------------------------------------------------------------------- answers
def test_known_answers_counts_distances_lead_ttc():
    st = _scene()
    g = {a['id']: a for a in st['agents']}
    assert g[10]['distance_m'] == pytest.approx(25.0 - 2.4 - 2.0)        # box to box
    assert g[10]['ttc_s'] == pytest.approx(20.6 / 5.0, abs=1e-6)
    assert g[10]['lane_rel'] == 'ego' and g[11]['lane_rel'] == 'right' and g[12]['lane_rel'] == 'oncoming'
    assert st['lead_id'] == 10 and g[10]['range_rate_mps'] == pytest.approx(-5.0)
    d = L.describe(st, seed=0)
    assert sorted(q['value'] for q in _qa(d, 'count')) == [0, 1, 2]
    veh = next(q for q in _qa(d, 'count') if 'vehicles' in q['question']['text'])
    assert veh['value'] == 2 and veh['ids'] == [10, 12]
    dist = _qa(d, 'distance', [10])
    assert dist and all(q['value'] == pytest.approx(20.6) for q in dist)
    assert '20.6 m' in dist[0]['answer']['text']
    ttc = _qa(d, 'risk_ttc')[0]
    assert ttc['value'] == pytest.approx(4.12) and '4.1 s' in ttc['answer']['text']
    lead = _qa(d, 'lead')[0]
    assert lead['value'] == 10 and lead['answer']['text'].startswith('Yes: a car 21 m ahead')
    assert _qa(d, 'risk_brake')[0]['value'] is False                       # TTC 4.1 s > 3 s, nobody in the path
    texts = ' '.join(c['text'] for c in d['captions'])
    assert 'a car 21 m ahead' in texts.lower() and 'closing at 5.0 m/s' in texts
    assert 'in the lane to your right' in texts and 'oncoming lane' in texts


def test_left_and_right_in_the_left_handed_frame():
    """+y in the ego frame is the RIGHT (CARLA is left-handed); heading world +y, the ego's right is world −x."""
    for ego_yaw, right_xy, left_xy in ((0.0, (10.0, 3.0), (10.0, -3.0)), (90.0, (-3.0, 10.0), (3.0, 10.0))):
        ego = dict(EGO, yaw=ego_yaw, vx=0.0)
        a = _row(20, *right_xy, ext=PED[1], cat=PED[0])
        b = _row(21, *left_xy, ext=PED[1], cat=PED[0])
        st = L.build_state(0, 0.0, ego, [a, b])
        g = {x['id']: x for x in st['agents']}
        assert g[20]['y'] == pytest.approx(3.0) and g[20]['direction'] == 'ahead on your right'
        assert g[21]['y'] == pytest.approx(-3.0) and g[21]['direction'] == 'ahead on your left'
        d = L.describe(st, seed=3)
        side = {q['ids'][0]: q for q in _qa(d, 'relative_position')}
        assert side[20]['value'] == 'right' and 'on your right' in side[20]['answer']['text']
        assert side[21]['value'] == 'left' and 'on your left' in side[21]['answer']['text']


def test_why_brake_ttc_and_a_pedestrian_entering_the_path():
    close = _row(30, 9.0, 0.0, vx=2.0)                                  # 4.6 m gap, closing at 8 m/s: 0.58 s
    st = L.build_state(0, 0.0, EGO, [close], _lanes())
    d = L.describe(st)
    brake = [c for c in d['captions'] if c['kind'] == 'brake']
    assert brake and brake[0]['ids'] == [30] and '0.6 s' in brake[0]['text']
    assert _qa(d, 'risk_brake')[0]['value'] is True
    # a pedestrian on the kerb walking into the road ahead: not in the path yet, entering it within 3 s
    walker = _row(31, 10.0, 4.0, vy=-1.5, ext=PED[1], cat=PED[0])
    st = L.build_state(0, 0.0, dict(EGO, vx=2.0), [walker], None)
    g = st['agents'][0]
    assert not g['in_path'] and g['entering_path'] and g['ttc_s'] is None
    d = L.describe(st)
    b = next(c for c in d['captions'] if c['kind'] == 'brake')
    assert 'entering your path' in b['text'] and b['ids'] == [31]
    # walking away from the road: no reason to brake
    st = L.build_state(0, 0.0, dict(EGO, vx=2.0), [dict(walker, vy=1.5)], None)
    assert not st['agents'][0]['entering_path']
    assert _qa(L.describe(st), 'risk_brake')[0]['value'] is False


def test_a_car_behind_closing_is_no_reason_to_brake():
    chaser = _row(40, -9.0, 0.0, vx=18.0)
    st = L.build_state(0, 0.0, EGO, [chaser], _lanes())
    assert st['agents'][0]['ttc_s'] is not None and st['agents'][0]['ttc_s'] < 1.0
    assert _qa(L.describe(st), 'risk_brake')[0]['value'] is False


# --------------------------------------------------------------------------------------------------------- grounding
def _check_grounding(sentence, valid_ids):
    for r in sentence['refs']:
        a, b = r['span']
        assert sentence['text'][a:b] == r['phrase'] and r['ids'] and set(r['ids']) <= set(valid_ids)
    spans = sorted(r['span'] for r in sentence['refs'])
    assert all(s1[0] >= s0[1] for s0, s1 in zip(spans, spans[1:]))


def test_grounding_spans_point_at_the_right_text():
    st = _scene()
    d = L.describe(st, seed=1)
    ids = [a['id'] for a in st['agents']]
    for c in d['captions']:
        _check_grounding(c, ids)
        assert c['step'] == 5 and c['frame'] == 105
    for q in d['qa']:
        _check_grounding(q['question'], ids)
        _check_grounding(q['answer'], ids)
    lead = next(c for c in d['captions'] if c['kind'] == 'lead')
    (r,) = lead['refs']
    assert r['ids'] == [10] and r['phrase'].lower() == 'a car 21 m ahead'
    ped = [c for c in d['captions'] if c['ids'] == [11]]
    assert ped and ped[0]['refs'][0]['phrase'].lower() == 'a pedestrian 9.6 m ahead'
    count = next(c for c in d['captions'] if c['kind'] == 'count')
    assert {frozenset(r['ids']): r['phrase'] for r in count['refs']} == {frozenset((10, 12)): '2 vehicles',
                                                                        frozenset((11,)): '1 pedestrian'}


def test_two_cars_with_the_same_direction_get_ordinals():
    a, b = _row(50, 20.0, 0.0, vx=10.0), _row(51, 40.0, 0.0, vx=10.0)
    d = L.describe(L.build_state(0, 0.0, EGO, [a, b], _lanes()))
    qs = {q['ids'][0]: q['question']['text'] for q in _qa(d, 'distance')}
    assert 'the nearest car ahead' in qs[50] and 'the second-nearest car ahead' in qs[51]


def test_seeded_variation_is_deterministic():
    st = _scene()
    assert L.describe(st, seed=7) == L.describe(st, seed=7)
    texts = {json.dumps([c['text'] for c in L.describe(st, seed=s)['captions']]) for s in range(8)}
    assert len(texts) > 1                                                # the seed chooses among phrasings


# --------------------------------------------------------------------------------------------------------- paraphrase
def test_paraphrase_hook_keeps_or_rejects_grounding():
    st = _scene()
    lead = next(c for c in L.describe(st)['captions'] if c['kind'] == 'lead')
    refs_before = json.loads(json.dumps(lead['refs']))
    phrase = lead['refs'][0]['phrase']
    prompts = []

    def good(prompt):
        prompts.append(prompt)
        return f'Up ahead in your lane you have {phrase.lower()}, closing at 5.0 m/s.'
    p = L.paraphrase(lead['text'], lead['refs'], good)
    (r,) = p['refs']
    assert p['text'][r['span'][0]:r['span'][1]] == r['phrase'] and r['ids'] == [10]
    assert phrase in prompts[0] and lead['refs'] == refs_before          # the input is not changed

    class Client:                                                       # an object with .complete works too
        def complete(self, prompt):
            return 'A vehicle up ahead is getting closer at 5.0 m/s.'
    with pytest.raises(L.ParaphraseRejected, match='drops the reference'):
        L.paraphrase(lead['text'], lead['refs'], Client())
    with pytest.raises(L.ParaphraseRejected, match='numbers'):
        L.paraphrase(lead['text'], lead['refs'], lambda p: f'{phrase}, closing at 6 m/s.')
    with pytest.raises(ValueError, match='no paraphrase client'):
        L.paraphrase(lead['text'], lead['refs'], None)


# --------------------------------------------------------------------------------------------------------- end to end
def test_derive_language_on_a_recorded_drive(run_dir, tmp_path):  # noqa: F811
    out = str(tmp_path / 'lang')
    m = L.derive_language(run_dir, out, {'seed': 2})
    data = WM.load_run(run_dir)
    assert m['schema'] == L.SCHEMA and m['records'] == 30 and m['captions'] > 30 and m['qa_pairs'] > 60
    assert m['right_of_way'].startswith('not stated') and m['lanes'] == 'lanes.json' and m['scenario'] is None
    assert m['provenance']['inputs']['actors']['sha256'] and m['paraphrase'] == {'client': False, 'accepted': 0,
                                                                                 'rejected': 0}
    recs = [json.loads(x) for x in open(os.path.join(out, 'language.jsonl'))]
    assert [r['step'] for r in recs] == list(range(30))
    for r in recs:
        f = data['frames'][r['step']]
        line = data['actors'][r['step']]
        ids = [a['id'] for a in line if a['id'] != data['ego_id']]
        for c in r['captions']:
            _check_grounding(c, ids)
            assert c['step'] == r['step'] and c['frame'] == f['frame']
        for q in r['qa']:
            _check_grounding(q['question'], ids)
            _check_grounding(q['answer'], ids)
        # the facts are the state: positions in the ego frame from actors.jsonl, sides from the sign of y
        Ti = np.linalg.inv(WM.pose_matrix(f['ego']))
        facts = {a['id']: a for a in r['facts']['agents']}
        for a in line:
            if a['id'] in facts:
                c, _R, _e = WM.actor_box(a)
                p = Ti[:3, :3] @ c + Ti[:3, 3]
                assert facts[a['id']]['x'] == pytest.approx(p[0], abs=2e-3)
                assert facts[a['id']]['y'] == pytest.approx(p[1], abs=2e-3)
        for q in r['qa']:
            if q['type'] == 'relative_position' and q['value'] in ('left', 'right'):
                assert (facts[q['ids'][0]]['y'] > 0) == (q['value'] == 'right')
    again = str(tmp_path / 'lang2')
    L.derive_language(run_dir, again, {'seed': 2})
    assert open(os.path.join(out, 'language.jsonl')).read() == open(os.path.join(again, 'language.jsonl')).read()


def test_derive_language_on_a_scenario_run(tmp_path):
    """lead_brake on the fake: the brake event is captioned at its step naming the lead's CARLA id, and the run's
    most critical moment is the step where the state's own TTC equals the scenario's minimum."""
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.core.drive import scenario_config
    from falcon_autonomy.core.project import Project
    p = Project.new('Test')
    p.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=[])
    p.save(str(tmp_path / 'p.falcon.json'))
    cfg = scenario_config(p, SC.builtin('lead_brake'), 'world-town10', 'rig-default', str(tmp_path / 'run'), 2310,
                          8310, seed=1, values={'speed': 40.0, 'gap': 12.0, 'decel': 8.0, 'brake_at': 2.0})
    (tmp_path / 'run.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(tmp_path / 'run.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=180)
    assert r.returncode == P.OK, r.stderr
    m = L.derive_language(str(tmp_path / 'run'), str(tmp_path / 'lang'))
    sc = m['scenario']
    lead_id = sc['actor_ids']['lead']
    assert sc['name'] == 'Lead vehicle brakes' and sc['timeline_found'] and isinstance(lead_id, int)
    (ev,) = sc['events']
    assert ev['do'] == 'brake' and ev['actor_id'] == lead_id and ev['step'] == 40
    recs = {x['step']: x for x in map(json.loads, open(tmp_path / 'lang' / 'language.jsonl'))}
    caps = [c for c in recs[40]['captions'] if c['kind'] == 'scenario']
    assert caps and caps[0]['ids'] == [lead_id] and 'starts braking now' in caps[0]['text']
    crit = sc['critical']
    rec = recs[crit['step']]
    assert any(c['kind'] == 'criticality' and c['ids'] == [lead_id] for c in rec['captions'])
    fact = next(a for a in rec['facts']['agents'] if a['id'] == lead_id)
    assert fact['ttc_s'] == pytest.approx(crit['min_ttc_s'], abs=0.01)
    assert math.isfinite(sc['criticality'])
