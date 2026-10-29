"""Vision-language explanations scored against ground truth (core.vlm, core.vlm_eval, core.vlm_models, core.vlm_lora).

The scorer is calibrated against hand-written answers whose scores are known, on hand-made states (core.language's
build_state) — counts, yes/no, lanes, sides, motion, tolerance bands, grounding, hallucination and the faithfulness of
brake explanations. End to end on two drives the drive worker records against the fake carla: the resolver reproduces
every phrase core.language grounded; the oracle scores 1.0 everywhere; the noisy baseline scores exactly what its
corruption log predicts; a cancelled evaluation resumes to the same answers. The Hugging Face runner and LoRA run on a
tiny randomly initialised Idefics3 (SmolVLM's architecture) built here offline — no weights are downloaded, and
fetch() is tested against a local file:// repository.
"""
import hashlib
import json
import os

import numpy as np
import pytest

from falcon_autonomy.core import language as L
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import vlm as V
from falcon_autonomy.core import vlm_eval as E
from falcon_autonomy.core import vlm_models as VM
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (the GUI fixtures)
from test_language import EGO, PED, _lanes, _row

MAN = {'lanes': 'lanes.json', 'options': dict(L.DEFAULTS)}


def _record(st, seed=0):
    """A language.jsonl record of a state, exactly as core.language.derive_language writes it."""
    d = L.describe(st, seed)
    facts = {'ego': st['ego'], 'lead_id': st['lead_id'], 'traffic_light': st['traffic_light'],
             'agents': [{k: g[k] for k in ('id', 'category', 'x', 'y', 'distance_m', 'range_rate_mps', 'speed_mps',
                                           'ttc_s', 'lane_rel', 'lanes_over', 'in_path', 'entering_path')}
                        for g in st['agents'] if g['range_m'] <= 50.0]}
    return {'step': st['step'], 'frame': st['frame'], 'sim_t': st['sim_t'], 'captions': d['captions'], 'qa': d['qa'],
            'facts': facts}


def _calm():
    """test_language's scene: a lead car 20.6 m ahead in the ego's lane closing at 5 m/s (TTC 4.1 s), a pedestrian
    standing in the lane to the right, an oncoming car behind on the left. No reason to brake."""
    lead = _row(10, 25.0, 0.0, vx=5.0)
    ped = _row(11, 12.0, 3.0, ext=PED[1], cat=PED[0])
    onc = _row(12, -15.0, -3.5, yaw=180.0, vx=-8.0)
    st = L.build_state(5, 0.25, EGO, [lead, ped, onc], _lanes(), frame=105)
    rec = _record(st)
    return rec, V.context_of(rec, MAN, 'd0')


def _danger():
    """A slower car 9.6 m ahead in the ego's lane (closing at 6 m/s: TTC 1.6 s) and, beyond it, a pedestrian walking
    along the lane 27 m ahead (TTC 3.2 s: in the path, not a TTC reason)."""
    car = _row(20, 14.0, 0.0, vx=4.0)
    ped = _row(21, 30.0, 0.0, vx=1.5, ext=PED[1], cat=PED[0])
    st = L.build_state(7, 0.35, EGO, [car, ped], _lanes(), frame=107)
    rec = _record(st)
    return rec, V.context_of(rec, MAN, 'd0')


def _q(qs, kind, target=None):
    return next(q for q in qs if q['type'] == kind and (target is None or q['target'] == target))


# --------------------------------------------------------------------------------------------------------- questions
def test_questions_carry_the_truth_of_the_state():
    rec, ctx = _calm()
    qs = V.questions(rec, ctx)
    assert {q['type'] for q in qs} == set(V.QTYPES)
    assert _q(qs, 'count_pedestrians')['truth']['value'] == 1 and _q(qs, 'count_vehicles')['truth']['value'] == 2
    assert _q(qs, 'count_cyclists')['truth']['value'] == 0
    assert _q(qs, 'lead')['truth']['value'] is True and _q(qs, 'lead')['target'] == 10
    assert _q(qs, 'ttc')['truth']['value'] == pytest.approx(4.12) and _q(qs, 'ttc')['target'] == 10
    assert _q(qs, 'lane', 10)['truth'] == {'value': 'ego', 'n': 0}
    assert _q(qs, 'lane', 11)['truth'] == {'value': 'right', 'n': 1}
    assert _q(qs, 'distance', 11)['text'].startswith('How far is the pedestrian') or \
        'pedestrian' in _q(qs, 'distance', 11)['text']
    assert _q(qs, 'motion', 10)['truth']['value'] == 'closing' and _q(qs, 'motion', 11)['truth']['value'] == 'closing'
    assert _q(qs, 'brake')['truth']['value'] is False and _q(qs, 'brake_reason')['truth']['reasons'] == []
    rec, ctx = _danger()
    kinds = [(r['kind'], r['id']) for r in ctx['reasons']]
    assert ('ttc', 20) in kinds and ('path', 21) in kinds
    assert set(_q(V.questions(rec, ctx), 'brake')['target']) == {20, 21}


def test_the_template_answers_score_perfectly_and_ground_to_their_targets():
    for rec, ctx in (_calm(), _danger()):
        for q in V.questions(rec, ctx):
            s = V.score_answer(q, q['template'], ctx)
            assert s['correct'], (q, s)
            assert s['hallucinated'] == 0 and s['grounded'] in (True, None), (q, s)


# --------------------------------------------------------------------------------------------------------- calibration
CALM = [  # (type, target, answer, correct, grounded, hallucinated)
    ('count_pedestrians', None, 'There is one pedestrian.', True, None, 0),
    ('count_pedestrians', None, 'I can see 1 pedestrian within 50 m.', True, None, 0),
    ('count_pedestrians', None, 'Two pedestrians.', False, None, 0),
    ('count_pedestrians', None, 'There are no pedestrians.', False, None, 0),
    ('count_cyclists', None, 'None.', True, None, 0),
    ('count_cyclists', None, 'There are 2 cyclists.', False, None, 1),          # a group claim with none present
    ('count_vehicles', None, 'I count two vehicles.', True, None, 0),
    ('lead', 10, 'Yes, a car about 20 m ahead.', True, True, 0),
    ('lead', 10, 'Yes: the car ahead of you.', True, True, 0),
    ('lead', 10, 'No, the lane is empty.', False, None, 0),
    ('lead', 10, 'Yes, the cyclist ahead.', True, False, 1),                    # right yes, an absent object
    ('distance', 10, 'About 22 m.', True, None, 0),                             # band max(2, 15 % of 20.6) = 3.09
    ('distance', 10, 'It is roughly 30 metres away.', False, None, 0),
    ('distance', 10, '0.021 km.', True, None, 0),
    ('speed', 10, 'It moves at 18 km/h.', True, None, 0),                       # 5.0 m/s
    ('speed', 10, '10 m/s.', False, None, 0),
    ('speed', 11, 'The pedestrian is standing still.', True, True, 0),
    ('ttc', 10, 'About 4 seconds.', True, None, 0),                             # band max(0.5, 20 % of 4.12)
    ('ttc', 10, '2 s.', False, None, 0),
    ('ttc', 10, 'There is no collision course.', False, None, 0),
    ('lane', 10, 'The car ahead is in your lane.', True, True, 0),
    ('lane', 11, 'It is in the lane to your right.', True, None, 0),
    ('lane', 11, 'The pedestrian is in your lane.', False, True, 0),
    ('lane', 11, 'It is two lanes to your right.', False, None, 0),
    ('relative_position', 11, 'On the right.', True, None, 0),
    ('relative_position', 11, 'The pedestrian is on your left.', False, True, 0),
    ('motion', 10, 'The car is approaching.', True, True, 0),
    ('motion', 10, 'It is pulling away.', False, None, 0),
    ('motion', 11, 'The car behind you is getting closer.', True, False, 0),    # right motion, wrong object
    ('motion', 11, 'The car on your left is getting closer.', True, False, 1),  # "on your left" is beside: nothing
    ('brake', None, 'No.', True, None, 0),
    ('brake', None, 'Yes, you should brake now.', False, None, 0),
    ('brake_reason', None, 'No need to brake: the road ahead is clear.', True, None, 0),
    ('brake_reason', None, 'No, the car ahead is far enough away.', True, None, 0),
    ('brake_reason', None, 'Yes, because the pedestrian on the right is crossing.', False, None, 0),
    ('brake_reason', None, 'Yes: the traffic light is red.', False, None, 0),
    ('brake_reason', None, 'No, the light is green.', True, None, 0),
]
DANGER = [
    ('brake', None, 'Yes, brake: the car ahead is much slower.', True, True, 0),
    ('brake', None, 'Yes, the pedestrian ahead is walking in your lane.', True, True, 0),
    ('brake', None, 'Yes, because of the cyclist on your left.', True, False, 1),
    ('brake_reason', None, 'Yes: the car ahead is slower and you would hit it within two seconds.', True, True, 0),
    ('brake_reason', None, 'Yes, there is a pedestrian walking in the road ahead.', True, True, 0),
    ('brake_reason', None, 'Brake: the light is red.', False, None, 0),
    ('brake_reason', None, 'No, nothing is in the way.', False, None, 0),
    ('brake_reason', None, 'Yes, because of the speed bump.', False, None, 0),
    ('brake_reason', None, 'Yes, the cyclist on the left is swerving.', False, False, 1),
    ('brake_reason', None, 'Yes.', False, None, 0),                             # no reason given: unparsed
]


@pytest.mark.parametrize('scene,case', [('calm', c) for c in CALM] + [('danger', c) for c in DANGER])
def test_hand_made_answers_get_their_known_scores(scene, case):
    rec, ctx = _calm() if scene == 'calm' else _danger()
    kind, target, answer, correct, grounded, halluc = case
    q = _q(V.questions(rec, ctx, {'max_objects': 3}), kind, target)
    s = V.score_answer(q, answer, ctx)
    assert s['correct'] is correct, s
    assert s['grounded'] is grounded, s
    assert s['hallucinated'] == halluc, s


def test_explanation_precision_and_recall():
    rec, ctx = _danger()
    q = _q(V.questions(rec, ctx), 'brake_reason')
    one = V.score_answer(q, 'Yes: the car ahead is closing in.', ctx)
    assert one['faithful'] and one['precision'] == 1.0 and one['recall'] == 0.5        # one of the two reasons
    both = V.score_answer(q, 'Yes: the car ahead is closing in, and a pedestrian is walking in your lane.', ctx)
    assert both['recall'] == 1.0 and both['precision'] == 1.0
    mixed = V.score_answer(q, 'Yes: the car ahead is closing in and the light is red.', ctx)
    assert not mixed['faithful'] and mixed['precision'] == 0.5 and mixed['decision_correct'] is True


def test_mentions_and_resolution_rules():
    rec, ctx = _calm()
    ms = V.mentions('The gap to the pedestrian 9.6 m away on your right is 9.6 m, and a car 21 m ahead closes in.')
    assert [m['phrase'] for m in ms] == ['the pedestrian 9.6 m away', 'a car 21 m ahead']
    assert ms[0]['distance'] == pytest.approx(9.6) and ms[1]['direction'] == 'ahead'
    r = {m['phrase']: V.candidates(m, ctx) for m in V.mentions('the car with track id 12; the car with track id 99')}
    assert r['the car with track id 12']['resolved'] == 12 and r['the car with track id 99']['status'] == 'hallucinated'
    far = V.candidates(V.mentions('a truck 80 m ahead')[0], ctx)
    assert far['status'] == 'out_of_range'                                   # beyond the radius: not scored
    near = V.candidates(V.mentions('a truck 30 m ahead')[0], ctx)
    assert near['status'] == 'hallucinated'                                  # no vehicle 30 m ahead (the lead is 20.6)
    behind = V.candidates(V.mentions('the car behind on your left')[0], ctx)
    assert behind['resolved'] == 12
    assert V.candidates(V.mentions('the car behind on your right')[0], ctx)['status'] == 'hallucinated'


def test_parsers():
    assert V.parse_count('twelve cars') == 12 and V.parse_count('none at all') == 0 and V.parse_count('?') is None
    assert V.parse_yesno('Yes.') is True and V.parse_yesno('There is no car.') is False
    assert V.parse_yesno('maybe') is None
    assert V.parse_number('about 36 km/h', 'speed') == (pytest.approx(10.0), 'unit')
    assert V.parse_number('5 m/s', 'distance')[1] == 'bare'                  # m/s is not metres
    assert V.parse_number('never', 'ttc')[1] == 'no collision'
    assert V.parse_lane('it is in the oncoming lane') == ('oncoming', None)
    assert V.parse_lane('three lanes to your left') == ('left', 3)
    assert V.parse_lane('on the sidewalk') == ('off_road', None)
    assert V.parse_motion('it is stationary') == 'still' and V.parse_motion('coming towards you') == 'closing'


# --------------------------------------------------------------------------------------------------------- aggregation
ACC = (0.5, 0.6, 0.65, 0.75, 0.8, 0.9)


def _rows(drives, per, acc_of):
    out = []
    for d in range(drives):
        for k in range(per):
            ok = k < acc_of(d) * per
            out.append({'type': 'lead', 'kind': 'yesno', 'correct': ok, 'parsed': True, 'grounded': None,
                        'n_mentions': 1, 'hallucinated': 0 if ok else 1, 'drive': f'd{d}', 'sim_t': 0.5 * k})
    return out


def test_aggregate_intervals_over_drives_or_blocks():
    many = V.aggregate(_rows(6, 20, lambda d: ACC[d]), n_boot=300, seed=1)
    m = many['metrics']['lead.accuracy']
    assert many['unit'] == 'drive' and many['reliable'] and many['n_units'] == 6
    assert m['ci_low'] < m['value'] < m['ci_high'] and m['value'] == pytest.approx(0.7)
    assert many['metrics']['hallucination_rate']['value'] == pytest.approx(0.3)
    few = V.aggregate(_rows(2, 20, lambda d: 0.5), n_boot=300, seed=1)
    assert few['unit'] == '2 s block' and not few['reliable'] and 'too narrow' in few['note']
    assert few['n_units'] == 2 * 5                                            # 10 s of frames per drive, 2 s blocks
    again = V.aggregate(_rows(6, 20, lambda d: ACC[d]), n_boot=300, seed=1)
    assert again['metrics'] == many['metrics']                                # seeded


# --------------------------------------------------------------------------------------------------------- end to end
def _project(path):
    from falcon_autonomy.core.project import Project
    p = Project.new('VLM')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=90, labels=['semantic'])
    p.doc['traffic_profiles'][0].update(vehicles=10, walkers=4)
    p.save(str(path))
    return p


@pytest.fixture(scope='module')
def drives(tmp_path_factory):
    """Two drives recorded by the drive worker against the fake carla, and their language data."""
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('vlm')
    project = _project(base / 'p.falcon.json')
    runs, langs = [], []
    for seed in (3, 4):
        run = base / f'drive{seed}'
        code, _, err = _run(_cfg(project, run, seconds=1.5, seed=seed))
        assert code == P.OK, err
        L.derive_language(str(run), str(base / f'lang{seed}'))
        runs.append(str(run))
        langs.append(str(base / f'lang{seed}'))
    return {'runs': runs, 'langs': langs, 'base': base}


def test_the_resolver_reproduces_every_grounded_phrase(drives):
    n = 0
    for d in drives['langs']:
        Lg = E.load_language(d)
        for rec in Lg['records']:
            ctx = V.context_of(rec, Lg['manifest'], Lg['name'])
            for s in list(rec['captions']) + [q['question'] for q in rec['qa']] + [q['answer'] for q in rec['qa']]:
                ms = V.ground(s['text'], ctx)
                for ref in s['refs']:
                    m = next(m for m in ms if m['span'][0] <= ref['span'][0] < m['span'][1])
                    if m['count'] is not None:
                        assert set(ref['ids']) <= set(m['ids'])
                    else:
                        assert [m['resolved']] == ref['ids'], (ref, m, s['text'])
                    n += 1
    assert n > 50


def test_oracle_end_to_end_through_the_command(drives, tmp_path):
    out = str(tmp_path / 'oracle')
    argv = ['eval', '--runner', 'oracle', '--out', out, '--every', '3', '--boot', '200']
    for r in drives['runs']:
        argv += ['--run', r]
    assert E.main_vlm(argv) == P.OK
    sc = json.load(open(os.path.join(out, 'scores.json')))
    m = sc['scores']['metrics']
    accs = {k: v['value'] for k, v in m.items() if k.endswith('.accuracy')}
    assert accs and all(v == 1.0 for v in accs.values()), accs
    assert m['hallucination_rate']['value'] == 0.0 and m['grounding']['value'] == 1.0
    assert sc['not_a_model'] and sc['runner']['licence'].startswith('GPL') and not sc['scores']['reliable']
    run = json.load(open(os.path.join(out, 'run.json')))
    assert run['status'] == 'ok' and run['runner']['name'] == 'oracle' and 'answers.jsonl' in run['outputs']
    assert any(k.startswith('lang:') for k in run['provenance']['inputs'])
    rows = [json.loads(x) for x in open(os.path.join(out, 'answers.jsonl'))]
    assert len(rows) == sc['n_questions'] and all(os.path.isfile(r['images'][-1]) for r in rows)
    assert os.path.isdir(os.path.join(out, 'language', os.path.basename(drives['runs'][0])))
    pg = E.report(out)
    html, meta = pg.render()
    assert 'NOT A MODEL' in html and meta['kind'] == 'vlm'


def test_noisy_baseline_scores_exactly_what_its_corruption_predicts(drives, tmp_path):
    s = {'runner': 'noisy:p=0.4,h=0.15,seed=2', 'lang': drives['langs'], 'every': 2, 'n_boot': 100}
    E.evaluate(s, str(tmp_path / 'a'))
    rows = [json.loads(x) for x in open(tmp_path / 'a' / 'answers.jsonl')]
    corrupted = hall = 0
    for r in rows:
        note = r['note'] or []
        expect = not ('corrupted' in note or (r['type'] == 'brake_reason' and 'hallucinated' in note))
        assert r['score']['correct'] is expect, r
        if 'hallucinated' in note:
            assert r['score']['hallucinated'] >= 1
            hall += 1
        corrupted += 'corrupted' in note
    assert 0.25 < corrupted / len(rows) < 0.55 and hall > 0
    sc = json.load(open(tmp_path / 'a' / 'scores.json'))
    acc = sc['scores']['metrics']['count_vehicles.accuracy']['value']
    cv = [r for r in rows if r['type'] == 'count_vehicles']
    assert acc == pytest.approx(1 - sum('corrupted' in (r['note'] or []) for r in cv) / len(cv))
    E.evaluate(s, str(tmp_path / 'b'))                                        # seeded: the same answers
    assert [json.loads(x)['answer'] for x in open(tmp_path / 'b' / 'answers.jsonl')] == [r['answer'] for r in rows]


def test_a_cancelled_evaluation_resumes_to_the_same_answers(drives, tmp_path):
    from falcon_autonomy.core.training import Cancelled
    s = {'runner': 'noisy:p=0.3,h=0.1,seed=0', 'lang': drives['langs'], 'every': 4, 'n_boot': 50}
    E.evaluate(s, str(tmp_path / 'whole'))
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        return calls['n'] > 25
    with pytest.raises(Cancelled):
        E.evaluate(s, str(tmp_path / 'part'), cancel=cancel)
    assert json.load(open(tmp_path / 'part' / 'run.json'))['status'] == 'cancelled'
    E.evaluate(s, str(tmp_path / 'part'))
    a = [json.loads(x)['answer'] for x in open(tmp_path / 'whole' / 'answers.jsonl')]
    b = [json.loads(x)['answer'] for x in open(tmp_path / 'part' / 'answers.jsonl')]
    assert a == b
    with pytest.raises(E.EvalError, match='other settings'):
        E.evaluate(dict(s, every=5), str(tmp_path / 'part'))


class EchoVLM:
    """A plug-in: answers yes, and records what it was given."""
    licence = 'MIT'
    seen = []

    def answer(self, images, prompt, meta):
        EchoVLM.seen.append((list(images), prompt, dict(meta)))
        return 'Yes.'


class FakeJudge:
    """A judge client written by the user: says FAITHFUL when the explanation has no 'red light' in it."""

    def complete(self, prompt):
        return 'UNFAITHFUL' if 'red' in prompt.split('The explanation:')[1] else 'FAITHFUL, because ...'


def test_plugin_runner_never_sees_the_truth_and_the_judge_needs_approval(drives, tmp_path):
    s = {'runner': 'plugin:test_vlm:EchoVLM', 'lang': drives['langs'][:1], 'every': 10, 'n_boot': 50, 'clip': 3,
         'clip_step': 2}
    E.evaluate(s, str(tmp_path / 'p'))
    images, prompt, meta = EchoVLM.seen[-1]
    assert 'question' not in meta and 'context' not in meta and meta['type'] in V.QTYPES
    assert prompt.endswith(V.PROMPT_SUFFIX) and 1 <= len(images) <= 3
    sc = json.load(open(tmp_path / 'p' / 'scores.json'))
    assert sc['runner']['licence'] == 'MIT' and sc['scores']['metrics']['lead.accuracy']['value'] is not None
    with pytest.raises(E.EvalError, match='approve'):
        E.evaluate(dict(s, runner='oracle', judge='test_vlm:FakeJudge'), str(tmp_path / 'j0'))
    E.evaluate(dict(s, runner='oracle', judge='test_vlm:FakeJudge', judge_approved=True), str(tmp_path / 'j'))
    j = json.load(open(tmp_path / 'j' / 'scores.json'))['judge']
    assert j['approved'] and j['judged'] > 0 and j['agreement_with_rules'] == 1.0


# --------------------------------------------------------------------------------------------------------- candidates
def test_candidates_are_permissive_and_pinned_to_revisions():
    for n, c in VM.CANDIDATES.items():
        assert c['licence'] == 'Apache-2.0' and not VM.check_licence(c)
        assert len(c['revision']) == 40 and c['sha256_pin'] == 'to be pinned on first approved download'
        assert 2e8 < c['params'] < 4e9
        weights = [f for f in c['files'] if f['name'].endswith('.safetensors')]
        assert weights and all(len(f['lfs_sha256']) == 64 for f in weights)
        assert all(len(f['git_oid']) == 40 for f in c['files'])
    assert {'qwen2.5-vl-3b', 'fastvlm-1.5b', 'paligemma2-3b'} <= set(VM.REFUSED)
    assert VM.check_licence({'licence': 'AGPL-3.0'}) and VM.check_licence({'licence': 'CC-BY-NC-4.0'})
    with pytest.raises(VM.VLMError, match='refused'):
        VM.fetch('qwen2.5-vl-3b', '/nonexistent')
    text = VM.describe_candidate('smolvlm-256m')
    assert 'Apache-2.0' in text and '74dea590' in text and '7e3e67ed' in text


def _fake_repo(root, files):
    """A local repository laid out like the Hub's resolve URLs, and a candidate describing it."""
    rev = 'a' * 40
    d = root / 'hub' / 'org' / 'tiny' / 'resolve' / rev
    d.mkdir(parents=True)
    out = []
    for name, data, lfs in files:
        (d / name).write_bytes(data)
        oid = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        out.append({'name': name, 'size': len(data), 'lfs_sha256': hashlib.sha256(data).hexdigest() if lfs else None,
                    'git_oid': oid})
    return {'repo': 'org/tiny', 'revision': rev, 'licence': 'Apache-2.0', 'files': out}, f'file://{root / "hub"}'


def test_fetch_checks_publisher_checksums_and_pins_sha256(tmp_path, monkeypatch):
    c, base = _fake_repo(tmp_path, [('config.json', b'{"a": 1}', False), ('model.safetensors', b'\x00' * 4096, True)])
    ws = str(tmp_path / 'ws')
    monkeypatch.setitem(VM.CANDIDATES, 'tiny', c)
    folder = VM.fetch('tiny', ws, base_url=base, candidates={'tiny': c})
    pins = json.load(open(os.path.join(folder, 'pins.json')))
    assert pins['files']['model.safetensors']['sha256'] == hashlib.sha256(b'\x00' * 4096).hexdigest()
    assert pins['files']['config.json']['checked'] == 'publisher git blob id'
    assert VM.verify('tiny', folder, c)['ok']
    with open(os.path.join(folder, 'config.json'), 'wb') as fh:       # changed on disk after the download
        fh.write(b'{"a": 2}')
    v = VM.verify('tiny', folder, c)
    assert not v['ok'] and any('pinned' in p for p in v['problems'])
    bad = dict(c, files=[dict(c['files'][1], lfs_sha256='0' * 64)])
    with pytest.raises(VM.VLMError, match='publisher'):
        VM.fetch('tiny', str(tmp_path / 'ws2'), base_url=base, candidates={'tiny': bad})
    assert not os.path.exists(os.path.join(VM.candidate_dir(str(tmp_path / 'ws2'), 'tiny'), 'model.safetensors'))


def test_fetch_command_asks_first_and_never_downloads_unconfirmed(tmp_path, monkeypatch, capsys):
    import urllib.request

    def no_network(*a, **k):
        raise AssertionError('no download may happen unconfirmed')
    monkeypatch.setattr(urllib.request, 'urlopen', no_network)
    monkeypatch.setattr('sys.stdin.isatty', lambda: False, raising=False)
    assert E.main_vlm(['fetch', 'smolvlm-256m', '--workspace', str(tmp_path)]) == P.REFUSED
    out = capsys.readouterr().out
    assert 'Apache-2.0' in out and 'not confirmed' in out
    assert E.main_vlm(['fetch', 'qwen2.5-vl-3b', '--workspace', str(tmp_path)]) == P.REFUSED
    assert E.main_vlm(['list']) == P.OK
    assert 'refused: non-commercial' in capsys.readouterr().out


# --------------------------------------------------------------------------------------------------------- HF + LoRA
def build_tiny_vlm(out):
    """A randomly initialised Idefics3 (SmolVLM's architecture: SigLIP-style vision encoder, Llama text model) with a
    byte-level BPE tokenizer trained here — about 70 000 parameters, nothing downloaded."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (Idefics3Config, Idefics3ForConditionalGeneration, Idefics3ImageProcessor,
                              Idefics3Processor, PreTrainedTokenizerFast)
    specials = ['<|endoftext|>', '<unk>', '<image>', '<fake_token_around_image>', '<global-img>', '<end_of_utterance>']
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ['How many pedestrians are within 50 m of you? The car ahead is 12.5 m away, closing at 3.0 m/s.',
              'User: Assistant: Yes. No. None within 50 m. Brake: a car is 2 s from a collision.'] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=specials,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token='<|endoftext|>', eos_token='<end_of_utterance>',
                                 unk_token='<unk>', pad_token='<|endoftext|>',
                                 additional_special_tokens=specials[2:])
    chat = ("{% for message in messages %}{{message['role'] | capitalize}}: {% for line in message['content'] %}"
            "{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}"
            "{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}"
            "{% endif %}")
    ip = Idefics3ImageProcessor(size={'longest_edge': 64}, max_image_size={'longest_edge': 32},
                                do_image_splitting=False)
    proc = Idefics3Processor(image_processor=ip, tokenizer=ft, image_seq_len=4, chat_template=chat)
    cfg = Idefics3Config(
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=32, patch_size=8),
        text_config=dict(model_type='llama', hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, vocab_size=len(ft), max_position_embeddings=512,
                         pad_token_id=ft.pad_token_id, bos_token_id=ft.bos_token_id, eos_token_id=ft.eos_token_id),
        image_token_id=ft.convert_tokens_to_ids('<image>'), scale_factor=2)
    torch.manual_seed(0)
    Idefics3ForConditionalGeneration(cfg).save_pretrained(out)
    proc.save_pretrained(out)
    with open(os.path.join(out, 'falcon_vlm.json'), 'w') as fh:
        json.dump({'licence': 'Apache-2.0', 'note': 'random weights, for tests'}, fh)
    return out


@pytest.fixture(scope='module')
def tiny_vlm(tmp_path_factory):
    pytest.importorskip('transformers')
    return build_tiny_vlm(str(tmp_path_factory.mktemp('tinyvlm')))


def test_hf_runner_runs_a_local_model_offline_and_deterministically(tiny_vlm, drives):
    Lg = E.load_language(drives['langs'][0])
    imgs = E.images_for(Lg, 10, clip=2, clip_step=3)
    assert len(imgs) == 2
    r = VM.make_runner(f'hf:{tiny_vlm},device=cpu,max_new_tokens=5,max_side=64')
    assert os.environ.get('HF_HUB_OFFLINE') == '1' and r.licence == 'Apache-2.0'
    a1 = r.answer(imgs, 'How many pedestrians are within 50 m of you?', {})
    a2 = r.answer(imgs, 'How many pedestrians are within 50 m of you?', {})
    assert isinstance(a1, str) and a1 == a2
    with pytest.raises(VM.VLMError, match='not downloaded'):
        VM.make_runner('hf:smolvlm-256m', str(drives['base'] / 'empty-ws'))


def _tiny_attention():
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(16, 16) for _ in range(4))
            self.vision_q_proj = nn.Linear(16, 16)

        def forward(self, x):
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            a = torch.softmax(q @ k.transpose(-1, -2) / 4.0, -1)
            return self.o_proj(a @ v) + self.vision_q_proj(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])
            self.vision = Block()

        def forward(self, x):
            for blk in self.layers:
                x = blk(x)
            return x + self.vision(x)
    torch.manual_seed(0)
    return Net()


def test_lora_starts_at_the_base_model_trains_only_itself_and_merges_exactly(tmp_path):
    import torch
    from falcon_autonomy.core import vlm_lora as LO
    net = _tiny_attention()
    x = torch.randn(3, 5, 16)
    y0 = net(x).detach()
    names = LO.apply_lora(net, r=4, alpha=8, dropout=0.0)
    assert sorted(names) == sorted(f'layers.{i}.{p}' for i in range(2) for p in ('q_proj', 'k_proj', 'v_proj',
                                                                                  'o_proj'))
    assert torch.equal(net(x), y0)                                           # B = 0: the base model exactly
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n in trainable)
    opt = torch.optim.SGD(LO.lora_parameters(net), lr=0.1)
    base_before = net.layers[0].q_proj.base.weight.clone()
    for _ in range(3):
        opt.zero_grad()
        ((net(x) - 1.0) ** 2).mean().backward()
        opt.step()
    assert torch.equal(net.layers[0].q_proj.base.weight, base_before)       # the base is frozen
    y1 = net(x).detach()
    assert not torch.allclose(y1, y0)
    LO.save_adapter(net, str(tmp_path / 'ad'), {'targets': list(LO.DEFAULT_TARGETS), 'scope': 'text'})
    other = _tiny_attention()
    LO.load_adapter(other, str(tmp_path / 'ad'))
    assert torch.allclose(other(x), y1, atol=1e-6)
    assert LO.merge_lora(net) == 8 and not LO.lora_modules(net)
    assert torch.allclose(net(x), y1, atol=1e-5)                             # merged: the same function
    with pytest.raises(LO.LoRAError, match='no linear layer'):
        LO.apply_lora(_tiny_attention(), targets=('nothing',))
    allm = LO.apply_lora(_tiny_attention(), scope='all')
    assert any('vision' in n for n in allm)


def test_finetune_splits_by_drive_and_resumes_to_the_same_adapter(tiny_vlm, drives, tmp_path):
    import torch
    from falcon_autonomy.core import vlm_lora as LO
    from falcon_autonomy.core.training import Cancelled
    s = {'model': tiny_vlm, 'lang': drives['langs'], 'epochs': 2, 'every': 6, 'max_examples': 16, 'device': 'cpu',
         'val_generate': 3, 'max_new_tokens': 4, 'max_side': 64, 'accumulate': 4, 'lr': 5e-3, 'seed': 1}
    res = LO.finetune(s, str(tmp_path / 'whole'))
    split = json.load(open(tmp_path / 'whole' / 'split.json'))
    assert len(split['train']) == 1 and len(split['val']) == 1 and not set(split['train']) & set(split['val'])
    assert res['n_train'] == 16 and res['trainable_parameters'] == 2 * 4 * 8 * (32 + 32)
    rows = [json.loads(x) for x in open(tmp_path / 'whole' / 'metrics.jsonl')]
    assert len(rows) == 2 and all(np.isfinite(r['val_loss']) for r in rows) and rows[0]['val_generated'] == 3
    cfg = json.load(open(tmp_path / 'whole' / 'adapter' / 'adapter.json'))
    assert cfg['licence'] == 'Apache-2.0' and cfg['scope'] == 'text'
    run = json.load(open(tmp_path / 'whole' / 'run.json'))
    assert run['status'] == 'ok' and run['model']['licence'] == 'Apache-2.0'
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        return calls['n'] == 22                                              # mid-epoch 2, mid-accumulation
    with pytest.raises(Cancelled):
        LO.finetune(s, str(tmp_path / 'part'), cancel=cancel)
    LO.finetune(s, str(tmp_path / 'part'))
    a = torch.load(tmp_path / 'whole' / 'adapter' / 'adapter.pt')
    b = torch.load(tmp_path / 'part' / 'adapter' / 'adapter.pt')
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(LO.LoRAError, match='at least two'):
        LO.split_drives(['only'], 0.25, 0)
    out = str(tmp_path / 'eval-adapter')
    E.evaluate({'runner': f'hf:{tiny_vlm},adapter={tmp_path / "whole" / "adapter"},device=cpu,max_new_tokens=4,'
                          f'max_side=64', 'lang': drives['langs'][:1], 'every': 20, 'max_objects': 1,
                'n_boot': 50}, out)
    sc = json.load(open(os.path.join(out, 'scores.json')))
    assert '+lora' in sc['runner']['name'] and not sc['not_a_model']


# --------------------------------------------------------------------------------------------------------- the GUI
def test_language_tab_evaluates_and_browses(window, qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox
    win = window
    win.project.path = str(tmp_path / 'proj.falcon.json')
    win.project.doc['rigs'][0]['sensors'][0].update(width=160, height=90, labels=['semantic'])
    win.save()
    win.service.set_workspace(win.workspace())
    data = win.go('data')
    data.record.seconds.setValue(1.0)
    data.record.record()
    job = win.queue.jobs[-1]
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120), job.log
    assert job.state == 'done', job.log
    page = win.go('models')
    tab = page.language
    page.tabs.setCurrentWidget(tab)
    pump(qapp, timeout=0.2)
    assert tab.cands.topLevelItemCount() == len(VM.CANDIDATES) + len(VM.REFUSED)
    asked = []
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.No)
    tab.cands.setCurrentItem(tab.cands.topLevelItem(0))
    n_jobs = len(win.queue.jobs)
    tab.download()
    assert asked and 'Apache-2.0' in asked[0][2] and len(win.queue.jobs) == n_jobs      # declined: no job
    assert tab.runs.count() == 1
    tab.runs.item(0).setCheckState(Qt.CheckState.Checked)
    tab.runner.setCurrentIndex(tab.runner.findData('noisy'))
    tab.boot.setValue(50)
    tab.every.setValue(4)
    tab.evaluate()
    j = win.queue.jobs[-1]
    assert not j.spec.needs_gpu
    assert pump(qapp, lambda: j.state in ('done', 'failed', 'cancelled'), 120), j.log
    assert j.state == 'done', j.log
    assert pump(qapp, lambda: tab.ev.count() == 1, 5)
    assert tab.table.topLevelItemCount() > 10 and 'NOT A MODEL' in tab.info.vals['Note'].text()
    assert tab.frames.topLevelItemCount() > 0
    tab.frames.setCurrentItem(tab.frames.topLevelItem(0))
    assert tab.qa.topLevelItemCount() > 5 and tab.image.pixmap() is not None and not tab.image.pixmap().isNull()
