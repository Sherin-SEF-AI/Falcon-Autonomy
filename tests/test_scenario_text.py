"""A scenario from a description and a run checked against it (core.scenario_text, core.scenario_check,
sim.scenariotext, the research tools, Scenarios ▸ Describe a scenario): the phrase set, the schema, Falcon's draft,
the repair rounds against a scripted model and against the fake Groq server (only text sent), every claim type on
synthetic runs with known answers, and the whole path on the fake carla — a world with a speed breaker, the director's
new need, the run checked."""
import copy
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKE, worker_env
from fake_groq import FAKE_KEY, FakeGroq

from falcon_autonomy.core import roadfeatures as RF
from falcon_autonomy.core import scenario_check as CK
from falcon_autonomy.core import scenario_text as T
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import weather as W

DESC = 'an autorickshaw cuts in from the left at a speed breaker at dusk'


# ================================================================================================ the phrase set
def test_extract_the_documented_phrases():
    ex = CK.extract(DESC)
    got = [(c['type'], c.get('kind'), c.get('what'), c.get('side'), c.get('feature'), c.get('class'))
           for c in ex['claims']]
    assert got == [('actor', 'autorickshaw', None, None, None, None),
                   ('manoeuvre', 'autorickshaw', 'cut_in', 'left', None, None),
                   ('location', None, None, None, 'speed_breaker', None),
                   ('light', None, None, None, None, 'twilight')]
    assert ex['claims'][2]['within_m'] == CK.THRESHOLDS['location_m'] and ex['unrecognised'] == []
    assert CK.extract(DESC) == ex                                       # deterministic
    ex = CK.extract('A cow stands in the lane in heavy rain at night while we drive at 40 km/h; the ego stops in '
                    'time. Then a parked bus within 50 m of a junction, and a jaywalker steps out.')
    c = {(x['type'], x.get('what') or x.get('class') or x.get('feature') or x.get('kind')): x for x in ex['claims']}
    assert [x['kind'] for x in ex['claims'] if x.get('what') == 'stopped'] == ['cattle', 'bus']   # "parked bus"
    assert c[('ego_speed', None)]['kmh'] == 40.0 and c[('outcome', None)]['collision'] is False
    assert c[('location', 'junction')]['within_m'] == 50.0 and ('weather', 'rain') in c and ('light', 'night') in c
    assert c[('manoeuvre', 'cross')]['kind'] == 'pedestrian'
    assert 'heavy' in ex['unrecognised']
    assert CK.extract('a cycle rickshaw overtakes on the right')['claims'][0]['kind'] == 'cycle_rickshaw'
    assert CK.extract('a scooter comes the wrong way')['claims'][1]['what'] == 'wrong_way'
    assert CK.extract('nothing to see')['claims'] == []


def test_model_claims_are_validated_and_merged():
    good, bad = CK.parse_model_claims('```json\n{"claims": [{"type": "weather", "class": "fog", "phrase": "murky"},'
                                      '{"type": "light", "class": "rain"}, {"type": "teleport"}]}\n```')
    assert [c['class'] for c in good] == ['fog'] and good[0]['source'] == 'model' and len(bad) == 2
    merged = CK.merge(CK.extract('a car brakes at night')['claims'], good + [{'type': 'light', 'class': 'day',
                                                                              'source': 'model'}])
    assert [c['id'] for c in merged] == ['c1', 'c2', 'c3', 'c4']         # the model's day light is not added
    assert [c['type'] for c in merged] == ['actor', 'manoeuvre', 'light', 'weather']
    with pytest.raises(ValueError, match='no JSON'):
        CK.parse_json_reply('I cannot do that')
    msgs = CK.claims_prompt('a murky morning')
    assert all(isinstance(m['content'], str) for m in msgs) and msgs[-1]['content'] == 'a murky morning'


# ================================================================================================ the schema, the draft
def test_schema_accepts_the_library_and_names_errors():
    for k in SC.LIBRARY:
        assert T.problems(SC.builtin(k)) == [], k
    bad = SC.builtin('cut_in')
    bad['actors'][0]['kind'] = 'rickshaw'
    bad['timeline'][0]['actions'][0]['actor'] = 'ghost'
    errs = T.problems(bad)
    assert any(e.startswith('at actors/0/kind') for e in errs)
    bad['actors'][0]['kind'] = 'car'
    assert T.problems(bad) == ["timeline 0: unknown actor 'ghost'"]          # then core.scenarios.validate
    assert T.problems([1]) == ['the scenario is not a JSON object']
    x = SC.builtin('lead_brake')
    x['params']['speed'] = {'dist': 'uniform', 'lo': 150, 'hi': 160}
    assert any('outside 0–130' in e for e in T.problems(x))


@pytest.mark.parametrize('hand,lane', [('RHT', 'adjacent'), ('LHT', 'kerb')])
def test_draft_puts_the_cut_in_on_the_named_side(hand, lane):
    claims = CK.extract(DESC)['claims']
    scn, notes = T.draft(claims, hand, DESC)
    assert T.problems(scn) == [] and T.mismatches(scn, claims, hand) == []
    a = scn['actors'][0]
    assert a['kind'] == 'autorickshaw' and a['place']['lane'] == lane
    assert 'speed_breaker' in scn['needs'] and scn['weather']['sun_altitude_angle'] == 0.0
    assert ('adjacent_same' in scn['needs']) == (lane == 'adjacent')
    # what the description says and a scenario does not
    other = copy.deepcopy(scn)
    other['needs'] = []
    other['weather'] = {'sun_altitude_angle': 45}
    other['actors'][0]['place']['lane'] = 'kerb' if lane == 'adjacent' else 'adjacent'
    mm = T.mismatches(other, claims, hand)
    assert any('speed_breaker' in m for m in mm) and any('is day' in m for m in mm)
    assert any('from the left' in m for m in mm)


def test_draft_needs_a_manoeuvre_and_maps_the_rest():
    assert T.draft(CK.extract('a bus at dusk')['claims'])[0] is None
    assert T.draft(CK.extract('a pedestrian brakes hard')['claims'])[0] is None
    scn, _ = T.draft(CK.extract('a stalled lorry in fog while we drive at 50 km/h')['claims'])
    assert scn['actors'][0]['kind'] == 'truck' and 'kind' not in scn['params']
    assert scn['params']['speed'] == {'dist': 'fixed', 'value': 50.0, 'unit': 'km/h',
                                      'about': 'the ego\'s speed, from the description'}
    assert W.resolve(scn['weather'])['fog_density'] == 50.0 and T.problems(scn) == []
    rec = T.write('a child runs across from the left in the rain at night')
    assert rec['status'] == 'valid' and rec['mode'] == 'phrases' and rec['sent']['requests'] == 0
    s = rec['scenario']
    assert s['id'].startswith('txt-') and s['source']['format'] == 'text' and s['actors'][0]['place']['lane'] == \
        'far_sidewalk'


# ================================================================================================ the repair rounds
class Scripted:
    """A model: replies in order; records the messages it was sent."""
    model = 'scripted/model'

    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, messages):
        self.sent.append(copy.deepcopy(messages))
        return self.replies.pop(0)


def _valid(hand='RHT'):
    scn, _ = T.draft(CK.extract(DESC)['claims'], hand, DESC)
    return T.example_for(scn)


def test_an_invalid_reply_is_repaired_within_the_rounds():
    broken = _valid()
    broken['actors'][0]['kind'] = 'tuk-tuk'                    # not a kind
    del broken['timeline'][0]['trigger']['gt_m']
    m = Scripted('Here it is: ' + json.dumps(broken), json.dumps(_valid()))
    rec = T.write(DESC, m, 'RHT', rounds=2)
    assert rec['status'] == 'valid' and len(rec['attempts']) == 2 and rec['scenario']['source']['rounds'] == 1
    assert rec['attempts'][0]['errors'] and any('actors/0/kind' in e for e in rec['attempts'][0]['errors'])
    first, second = m.sent
    # only text: the schema, the rules, the draft and the description; then the errors
    assert all(isinstance(x['content'], str) for x in second)
    assert '"$schema"' in first[0]['content'] and first[-1]['content'].endswith(DESC)
    assert 'actors/0/kind' in second[-1]['content'] and second[-2]['role'] == 'assistant'
    assert rec['sent']['requests'] == 2 and rec['scenario']['source']['model'] == 'scripted/model'
    # a valid scenario that misses what the text says is sent back while rounds remain, then kept with its mismatch
    mism = _valid()
    mism['needs'] = ['adjacent_same']
    m = Scripted(json.dumps(mism), json.dumps(mism))
    rec = T.write(DESC, m, 'RHT', rounds=1)
    assert rec['status'] == 'valid_with_mismatches' and any('speed_breaker' in x for x in rec['mismatches'])
    assert 'does not say what the description says' in m.sent[1][-1]['content']
    # never valid: none kept, bounded
    m = Scripted('no', '{"name": 1}', '{')
    rec = T.write(DESC, m, 'RHT', rounds=2)
    assert rec['status'] == 'invalid' and rec['scenario'] is None and len(rec['attempts']) == 3 and m.replies == []
    with pytest.raises(T.TextError):
        T.write('   ', m)


def answer(content, model='openai/gpt-oss-120b'):
    """A non-streamed chat completion of the fake Groq (the scenario writer does not stream)."""
    return {'json': {'model': model, 'choices': [{'message': {'role': 'assistant', 'content': content},
                                                  'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': 900, 'completion_tokens': 400}}}


@pytest.fixture
def groq_service(tmp_path, monkeypatch):
    from falcon_autonomy.assistant import keystore
    from falcon_autonomy.assistant.keystore import KeyStore
    from falcon_autonomy.assistant.service import NOTICE_VERSION, AssistantService
    from falcon_autonomy.core.settings import ASSISTANT_DEFAULTS, Settings
    monkeypatch.setattr(keystore, '_keyring', lambda: None)
    srv = FakeGroq().start()
    st = Settings(str(tmp_path / 'settings.json'))
    st.set('assistant', dict(copy.deepcopy(ASSISTANT_DEFAULTS), base_url=srv.base_url, enabled=True,
                             consent=NOTICE_VERSION, tpm_limit=0))
    from falcon_autonomy.core.settings import Secrets
    ks = KeyStore(Secrets(str(tmp_path / 'secrets.json')))
    ks.set(FAKE_KEY)
    yield AssistantService(st, keystore=ks), srv
    srv.stop()


def test_through_the_fake_groq_only_text_is_sent(groq_service):
    from falcon_autonomy.assistant.provider import ProviderError
    svc, srv = groq_service
    broken = _valid()
    broken['timeline'] = [{'trigger': {'type': 'time', 't_s': 1}, 'actions': [{'actor': 'ego', 'do': 'brake'}]}]
    srv.push(answer(json.dumps(broken)), answer('```json\n' + json.dumps(_valid()) + '\n```'))
    rec = T.write(DESC, svc.scenario_writer(DESC), 'RHT')
    assert rec['status'] == 'valid' and rec['model'] == 'openai/gpt-oss-120b'
    assert rec['attempts'][0]['errors'] == ["timeline 0: unknown actor 'ego'"]
    chats = [r for r in srv.requests if r['path'].endswith('/chat/completions')]
    assert len(chats) == 2
    for r in chats:
        body = r['body']
        assert all(isinstance(m['content'], str) for m in body['messages'])
        blob = json.dumps(body)
        assert DESC in blob and 'frames.jsonl' not in blob and 'image' not in blob.lower()
    assert "unknown actor 'ego'" in chats[1]['body']['messages'][-1]['content']
    # a description naming a private place is refused before anything is sent
    n = len(srv.requests)
    with pytest.raises(ProviderError, match='private'):
        svc.scenario_writer('replay ~/Videos/dashcam/clip.mp4 as a cut-in')
    assert len(srv.requests) == n


# ================================================================================================ checking runs
COLS = ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'length', 'width']


def _rows(xs, ys, yaw=0.0, v=None, L=4.6, Wd=2.0, dt=0.05):
    out = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        s = v[i] if v is not None else 8.0
        out.append([round((i + 1) * dt, 4), float(x), float(y), yaw, s * math.cos(math.radians(yaw)),
                    s * math.sin(math.radians(yaw)), L, Wd])
    return out


def synthetic_run(root, name, actors: dict, kinds: dict, weather=None, xodr='', collisions=0, subclasses=None):
    """A run folder with what the checker reads: run.json, drive.json and trajectories.json (the ego along +x at
    8 m/s for 8 s)."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    n = 160
    t = np.arange(n) * 0.05
    ego = _rows(t * 8.0, np.zeros(n))
    acts = {'ego': ego}
    for k, (xs, ys, yaw, v) in actors.items():
        acts[k] = _rows(xs, ys, yaw, v)
    with open(os.path.join(d, 'trajectories.json'), 'w') as fh:
        json.dump({'columns': COLS, 'actors': acts}, fh)
    with open(os.path.join(d, 'frames.jsonl'), 'w') as fh:
        for i in range(n):
            fh.write(json.dumps({'step': i, 'sim_t': float(t[i]), 'ego': {'x': ego[i][1], 'y': 0.0, 'yaw': 0.0}})
                     + '\n')
    sc = {'id': 's', 'name': 'synthetic', 'actor_ids': {k: 10 + i for i, k in enumerate(actors)},
          'blueprints': {}, 'subclasses': subclasses or {}, 'collisions': [{'t': 1.0}] * collisions, 'fired': []}
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok', 'scenario': sc}, fh)
    cfg = {'scenario': {'id': 's', 'actors': [{'id': k, 'kind': v} for k, v in kinds.items()]},
           'weather': W.resolve(weather or {}), 'xodr': xodr, 'town': ''}
    with open(os.path.join(d, 'drive.json'), 'w') as fh:
        json.dump(cfg, fh)
    return d


def _breaker_xodr(tmp_path, s=60.0):
    p = os.path.join(str(tmp_path), f'breaker-{s:g}.xodr')
    with open(p, 'w') as fh:
        fh.write(RF.apply(RF.test_road(300.0), [{'road': '1', 's': s, 'kind': 'irc_hump', 'source': 'test'}])[0])
    return p


def _by(rep):
    return {r['claim']: r for r in rep['claims']}


def test_each_claim_type_on_runs_with_known_answers(tmp_path):
    n = 160
    t = np.arange(n) * 0.05
    ex = t * 8.0
    # the auto: 3.5 m to the left (y −3.5), level with the ego + 12 m, moves into the lane between 2 and 3 s
    lat = np.where(t < 2.0, -3.5, np.where(t > 3.0, 0.0, -3.5 + 3.5 * (t - 2.0)))
    auto = (ex + 12.0, lat, 0.0, np.full(n, 8.0))
    # the lead: ahead in the lane, braking at 6 m/s² from 4 s
    v = np.where(t < 4.0, 8.0, np.maximum(0.0, 8.0 - 6.0 * (t - 4.0)))
    lead = (40.0 + np.concatenate([[0.0], np.cumsum(v[:-1] * 0.05)]), np.zeros(n), 0.0, v)
    xodr = _breaker_xodr(tmp_path, 40.0)
    d = synthetic_run(str(tmp_path), 'r1', {'auto': auto, 'lead': lead}, {'auto': 'autorickshaw', 'lead': 'car'},
                      weather={'sun_altitude_angle': 2.0, 'precipitation': 60.0}, xodr=xodr,
                      subclasses={'auto': 'autorickshaw (stand-in)'})
    rep = CK.check_run(d, 'an autorickshaw cuts in from the left at a speed breaker at dusk in the rain; a car '
                          'brakes hard ahead; no collision; we drive at 29 km/h')
    r = _by(rep)
    cut = r['autorickshaw: cut-in from the left']
    assert cut['status'] == 'met' and cut['measured']['from'] == 'left'
    assert cut['measured']['start_lateral_m'] == -3.5
    assert cut['measured']['entered_at_s'] == pytest.approx(2.75, abs=0.06)
    assert cut['measured']['gap_m'] == pytest.approx(12.0 - 4.6, abs=0.01)
    # the ego is at 8 m/s × 2.8 s ≈ 22 m when the auto enters; the breaker at 40 m: 18 m away
    loc = r['within 30 m of a speed breaker']
    assert loc['status'] == 'met' and loc['measured']['distance_m'] == pytest.approx(40 - 8 * 2.8, abs=0.5)
    assert r['an actor of kind autorickshaw']['status'] == 'met'
    assert r['light: twilight']['status'] == 'met' and r['light: twilight']['measured']['sun_altitude_deg'] == 2.0
    assert r['weather: rain']['status'] == 'met'
    br = r['car: hard braking']
    assert br['status'] == 'met' and br['measured']['peak_decel_mps2'] == pytest.approx(6.0, abs=0.3)
    assert r['no collision']['status'] == 'met' and r['the ego at 29 km/h']['status'] == 'met'
    assert rep['counts'] == {'met': 9, 'not_met': 0, 'not_checkable': 0}
    assert json.load(open(os.path.join(d, CK.REPORT)))['counts']['met'] == 9
    # the same run against other words: each not met, with what was measured
    rep = CK.check_run(d, 'a bus cuts in from the right within 5 m of a speed breaker at night in fog; the '
                          'autorickshaw cuts in from the right; a collision; we drive at 60 km/h', write=False)
    r = _by(rep)
    bus = r['an actor of kind bus']
    assert bus['status'] == 'not_met' and 'autorickshaw, car' in bus['value']
    assert r['autorickshaw: cut-in from the right']['status'] == 'not_met'
    assert 'came from the left' in r['autorickshaw: cut-in from the right']['value']
    loc = r['within 5 m of a speed breaker']                   # no manoeuvre met: the closest approach counts
    assert loc['status'] == 'met' and 'at its closest' in loc['value'] and loc['measured']['distance_m'] < 0.5
    assert r['light: night']['status'] == 'not_met' and r['weather: fog']['status'] == 'not_met'
    assert r['a collision']['status'] == 'not_met' and r['the ego at 60 km/h']['status'] == 'not_met'
    loc = _by(CK.check_run(d, 'the autorickshaw cuts in within 5 m of a speed breaker', write=False))[
        'within 5 m of a speed breaker']
    assert loc['status'] == 'not_met' and 'at the cut-in' in loc['value'] and loc['measured']['distance_m'] > 15


def test_crossing_standing_wrong_way_overtaking_and_limits(tmp_path):
    n = 160
    t = np.arange(n) * 0.05
    ex = t * 8.0
    ped = (np.full(n, 30.0), np.clip(-6.0 + 1.5 * t, -6.0, 6.0), 90.0, np.full(n, 1.5))      # left to right
    cow = (np.full(n, 70.0), np.full(n, 0.3), 0.0, np.zeros(n))
    scooter = (90.0 - 5.0 * t, np.full(n, 0.5), 180.0, np.full(n, 5.0))
    bike = (ex - 10.0 + 4.0 * t, np.full(n, -1.8), 0.0, np.full(n, 12.0))                   # passes on the left
    far = (ex + 40.0, np.where(t < 2, -3.5, 0.0), 0.0, np.full(n, 8.0))                        # moves in 35 m ahead
    d = synthetic_run(str(tmp_path), 'r2', {'ped': ped, 'cow': cow, 'scooter': scooter, 'bike': bike, 'far': far},
                      {'ped': 'pedestrian', 'cow': 'cattle', 'scooter': 'motorcycle', 'bike': 'bicycle', 'far': 'van'},
                      subclasses={'cow': 'vendor (proxy)'})
    rep = CK.check_run(d, 'a pedestrian crosses from the left; a cow stands in the lane; a scooter comes the '
                          'wrong way; a cyclist overtakes on the left; a van cuts in; at a junction', write=False)
    r = _by(rep)
    assert r['pedestrian: crossing from the left']['status'] == 'met'
    assert r['cattle: standing still']['status'] == 'met'
    assert r['cattle: standing still']['measured']['longest_still_s'] >= 1.0
    assert r['motorcycle: wrong-way']['status'] == 'met'
    assert r['bicycle: overtaking on the left']['status'] == 'met'
    assert r['an actor of kind cattle']['status'] == 'not_met'           # a proxy recorded as another subclass
    assert 'recorded as vendor (proxy)' in r['an actor of kind cattle']['value']
    van = r['van: cut-in']
    assert van['status'] == 'not_met' and 'a cut-in is within 25 m' in van['value']
    assert r['within 30 m of a junction']['status'] == 'not_checkable'   # no lanes, no OpenDRIVE file
    assert r['an actor of kind pedestrian']['status'] == 'met'
    # a free drive: actors and manoeuvres are not checkable, the light still is
    os.remove(os.path.join(d, 'trajectories.json'))
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok'}, fh)
    rep = CK.check_run(d, 'a car brakes at noon', write=False)
    assert [x['status'] for x in rep['claims']] == ['not_checkable', 'not_checkable', 'met']
    with pytest.raises(CK.CheckError, match='no description'):
        CK.check_run(d)
    with pytest.raises(CK.CheckError, match='not a recorded run'):
        CK.check_run(str(tmp_path))


# ================================================================================================ on the fake carla
@pytest.fixture
def breaker_project(tmp_path):
    from falcon_autonomy.core.project import Project
    p = Project.new('words')
    p.save(str(tmp_path / 'words.falcon.json'))

    def world(wid, s):
        wdir = os.path.join(p.workspace, 'worlds', wid)
        os.makedirs(wdir, exist_ok=True)
        with open(os.path.join(wdir, 'world.xodr'), 'w') as fh:
            fh.write(RF.apply(RF.test_road(300.0), [{'road': '1', 's': s, 'kind': 'irc_hump', 'source': 't'}])[0])
        p.doc['worlds'].append({'id': wid, 'name': f'Breaker at {s:g} m', 'kind': 'osm',
                                'xodr': f'worlds/{wid}/world.xodr',
                                'osm': {'bbox': [0, 0, 0.01, 0.01], 'centre': [12.97, 77.59], 'handedness': 'RHT'}})
    world('w-blr', 60.0)
    world('w-far', 250.0)
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save()
    return p


def _drive(cfg, out):
    with open(out + '.json', 'w') as fh:
        json.dump(cfg, fh)
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', out + '.json'], capture_output=True,
                       text=True, env=worker_env(), timeout=300)
    return r, json.load(open(os.path.join(out, 'run.json')))


def test_written_run_and_checked_on_the_fake(breaker_project, tmp_path):
    from falcon_autonomy.core import protocol as P
    from falcon_autonomy.core.drive import scenario_config
    from falcon_autonomy.core.project import ProjectError
    p = breaker_project
    broken = _valid()
    broken['actors'][0]['place']['lane'] = 'shoulder'
    m = Scripted(json.dumps(broken), json.dumps(_valid()))
    rec = T.write(DESC, m, T.world_handedness(p, 'w-blr'))
    assert rec['status'] == 'valid' and len(rec['attempts']) == 2
    scn = rec['scenario']
    out = str(tmp_path / 'run')
    cfg = scenario_config(p, scn, 'w-blr', 'rig-default', out, 2310, 8310, seed=0)
    assert cfg['breakers'] == [{'x': 60.0, 'y': -0.0, 'kind': 'irc_hump'}]
    r, run = _drive(cfg, out)
    assert r.returncode == P.OK, r.stderr
    rep = CK.check_run(out)                                  # the description from the scenario's source
    by = _by(rep)
    assert rep['description'] == DESC and rep['counts']['not_checkable'] == 0
    assert by['an actor of kind autorickshaw']['status'] == 'met'
    cut = by['autorickshaw: cut-in from the left']
    assert cut['status'] == 'met' and cut['measured']['from'] == 'left'
    assert cut['measured']['language']['first']['words'].startswith('an autorickshaw ahead on your left')
    assert by['light: twilight']['status'] == 'met'
    loc = by['within 30 m of a speed breaker']
    assert loc['measured']['feature'] == 'irc_hump' and loc['status'] == ('met' if loc['measured']['distance_m'] <= 30
                                                                          else 'not_met')
    # the director's need: no breaker 10–60 m ahead of any start, refused; a town has none at all
    r2, run2 = _drive(scenario_config(p, scn, 'w-far', 'rig-default', str(tmp_path / 'far'), 2310, 8310), str(
        tmp_path / 'far'))
    assert r2.returncode == P.FAILED and 'no speed breaker 10–60 m ahead' in run2['error']
    with pytest.raises(ProjectError, match='a CARLA town has none'):
        scenario_config(p, scn, p.doc['worlds'][0]['id'], 'rig-default', str(tmp_path / 't'), 2310, 8310)


def test_cli_from_text_offline_run_and_check(breaker_project, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'scenario', *a], capture_output=True,
                              text=True, env=env, timeout=600)
    path = breaker_project.path
    r = cli('from-text', path, DESC)                             # the assistant is off in the tests' settings
    assert r.returncode == 2 and 'nothing was sent' in r.stderr
    r = cli('from-text', path, DESC, '--offline', '--world', 'w-blr', '--run', '--fake-carla', FAKE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'valid (phrases, 1 attempt)' in r.stdout and 'autorickshaw: cut-in from the left' in r.stdout
    from falcon_autonomy.core.project import Project
    proj = Project.load(path)
    s = [x for x in proj.doc['scenarios'] if x['id'].startswith('txt-')]
    assert len(s) == 1 and s[0]['source']['description'] == DESC
    run = next(ln for ln in r.stdout.splitlines() if os.path.isdir(ln.strip()))
    assert os.path.isfile(os.path.join(run, CK.REPORT))
    r = cli('check', run, 'an autorickshaw cuts in from the left at dusk')
    assert r.returncode == 0 and 'every claim met' in r.stdout
    r = cli('check', run, 'a bus brakes at night', '--json')
    assert r.returncode == 1 and json.loads(r.stdout)['counts']['not_met'] == 3
    assert cli('check', str(tmp_path / 'nope')).returncode == 2


# ================================================================================================ the assistant's tools
def test_research_tools(breaker_project):
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.cli import schema_tokens
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    assert REGISTRY['research_scenario_check'].tier == 'read' and REGISTRY['research_scenario_text'].tier == 'act'
    assert schema_tokens()['research'] < 3500
    ctx = TOOLS.context(breaker_project.path, FAKE)
    try:
        d = call(ctx, 'research_scenario_check', {'description': DESC, 'world': 'w-blr'})
        assert d['draft']['actors'][0]['kind'] == 'autorickshaw' and len(d['claims']) == 4
        bad = _valid()
        bad['actors'][0]['kind'] = 'tuk-tuk'
        for k in range(1, T.MAX_ROUNDS + 1):
            v = call(ctx, 'research_scenario_check', {'description': DESC, 'scenario': bad})
            assert v['valid'] is False and v['attempt'] == k
        assert 'stop' in call(ctx, 'research_scenario_check', {'description': DESC, 'scenario': bad})
        ok = call(ctx, 'research_scenario_check', {'description': DESC, 'scenario': _valid()})
        assert ok == {'valid': True, 'mismatches': [], 'plan': [], 'next': ok['next']}
        assert 'not valid' in REGISTRY['research_scenario_text'].preview(ctx, description=DESC, scenario=bad)
        pv = REGISTRY['research_scenario_text'].preview(ctx, description=DESC, scenario=_valid(), world='w-blr')
        assert pv.startswith('add the scenario') and 'speed_breaker' in pv
        assert 'not valid' in call(ctx, 'research_scenario_text', {'description': DESC, 'scenario': bad})['error']
        q = call(ctx, 'research_scenario_text', {'description': DESC, 'scenario': _valid(), 'world': 'w-blr'})
        assert q['scenario'].startswith('txt-') and q['run'].startswith('runs/') and 'job' in q
        assert ctx.project.item('scenarios', q['scenario'])['source']['mode'] == 'assistant'
        from test_assistant_tools import _wait
        _wait(ctx, q['job'])
        c = call(ctx, 'research_scenario_check', {'description': DESC, 'run': q['run']})
        assert c['counts']['not_checkable'] == 0 and {x['claim'] for x in c['claims']} >= {
            'autorickshaw: cut-in from the left', 'light: twilight'}
    finally:
        ctx.service.shutdown(grace_s=10.0)
