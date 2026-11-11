"""Does a run match its description? A description in words ("an autorickshaw cuts in from the left at a speed breaker
at dusk") decomposed into checkable claims, each verified against what the run recorded — met, not met or not
checkable, with the measured value (standard library + numpy). docs/scenario-text.md is the manual.

    extract(text) -> {'claims', 'recognised', 'unrecognised'}     the documented phrase set (deterministic, offline)
    check_run(run_dir, text | claims) -> report                    falcon.scenario_check/1, also RUN/scenario-check.json
    claims_prompt(text) / parse_model_claims(reply)                 the optional model parse (only the text is sent)

Claims (CLAIM_TYPES; thresholds in THRESHOLDS, every one stated in the report):
* actor {kind} — a scenario actor of that kind (core.scenarios.KINDS) was spawned and recorded: in the drive config's
  concrete scenario, in run.json's scenario.actor_ids (its CARLA id) and with at least two rows in trajectories.json;
  for the Indian stand-ins and proxies also run.json's subclass. Background traffic is not counted.
* manoeuvre {what, side, kind} — from trajectories.json (the director's own record of every scenario actor and the
  ego, the same rows the criticality uses), each actor in the ego's frame (x forward, lateral + to the right, CARLA's
  left-handed frame; "left" is a negative lateral):
  - cut_in: first seen at least side_m (2.0 m) to one side, later within in_lane_m (1.0 m) of the ego's centre line
    while its rear is ahead of the ego's front by 0 to cut_gap_max_m (25 m); the side is the side it came from;
  - brake: ahead within lane_m (1.75 m) laterally and decelerating at least brake_decel_mps2 (3.0 m/s², the speed's
    derivative averaged over 0.25 s);
  - cross: moving across the ego's heading (45°–135°) and entering the ego's path (within in_lane_m) from at least
    side_m to one side, ahead of the ego;
  - stopped: ahead within lane_m laterally and within ahead_max_m (100 m), slower than stopped_mps (0.3 m/s) for at
    least stopped_s (1 s);
  - wrong_way: ahead within lane_m laterally, moving (> 0.3 m/s), heading at least wrong_way_deg (135°) from the ego's;
  - overtake: from behind the ego to ahead of it, within overtake_lateral_m (3.5 m) laterally as it passes.
  When a manoeuvre is found, the grounded language labels of core.language (build_state on the recorded actors and
  lanes: the actor's direction words and lane relation) at its first sample and at the moment found are reported
  beside the numbers.
* location {feature, within_m} — speed_breaker (the map's breakers: core.roadfeatures' falcon:road_feature userData in
  the world's OpenDRIVE file, kinds with a crossing speed), dip (the pothole proxy) or junction (core.rules: the ego in
  a junction road): the ego within within_m (30 m by default; "within N m of" in the text sets it) of one — at the
  moment of the first manoeuvre found when there is one, otherwise at its closest. Distances are centre to the
  feature's point on the road's reference line (heuristic: up to a lane or two of lateral offset is included).
* light {class} — the sun's altitude of the run's weather (drive.json) in core.oddcoverage's classes: night below
  −6°, twilight −6° to 6°, low sun 6° to 20°, day above. Dusk and dawn are both twilight (the azimuth is not read).
* weather {class} — the run's CARLA weather (drive.json) by core.oddcoverage.weather_class; "wet" is met by wetness
  or puddles ≥ 40 whatever the class, "cloudy" by cloudiness ≥ 50.
* ego_speed {kmh} — the ego's median speed over its first second (trajectories.json) within ±max(5 km/h, 15 %).
* outcome {collision} — run.json's scenario collisions (the director's collision events): none, or at least one.

Not checkable — and said why: a run without a scenario (no trajectories.json) for actors and manoeuvres, a map
without its OpenDRIVE file for breakers and dips, a run without lanes or OpenDRIVE for junctions, a weather not
recorded.

The phrase set (extract; PHRASES): kind words (autorickshaw/auto/tuk-tuk/three-wheeler, cycle rickshaw, scooter/
motorbike/bike/two-wheeler, cyclist/bicycle, car/taxi/sedan/SUV, truck/lorry, bus, van/tempo, pedestrian/person/man/
woman/jaywalker, child/kid, cow/cattle/bull/buffalo, handcart/pushcart/cart, vendor/hawker); manoeuvres (cuts in /
cut-in / swerves in / merges in front / pulls out; brakes / slams on the brakes / stops suddenly; crosses / walks
across / steps out / jaywalks; stalled / broken down / stopped / parked / stationary; wrong way / against the traffic;
overtakes / filters / weaves) with "from the left/right"; places (speed breaker/bump/hump/table, hump, bump; pothole,
dip; junction, intersection, crossroads, roundabout, traffic light) with "within N m"; light (dusk, dawn, twilight,
sunset, sunrise, evening; night, dark; low sun, glare; noon, midday, daytime, daylight, afternoon); weather (rain,
monsoon, downpour, drizzle; fog, mist, haze, smog; dust storm, dusty; wet road, waterlogged, puddles; cloudy,
overcast; sunny, clear sky); the ego's speed ("we drive at 40 km/h", "the ego at 30 kmph"); the outcome ("without a
collision", "avoids it", "collides with", "hits"). A manoeuvre takes the kind named nearest before it in its sentence
(else after it). Words of four letters or more that no phrase covered are listed as unrecognised: the model, when the
assistant is on, is asked to turn them into claims of the same types — only the text of the description is sent.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

SCHEMA = 'falcon.scenario_check/1'
REPORT = 'scenario-check.json'
CLAIM_TYPES = ('actor', 'manoeuvre', 'location', 'light', 'weather', 'ego_speed', 'outcome')
MANOEUVRES = ('cut_in', 'brake', 'cross', 'stopped', 'wrong_way', 'overtake')
FEATURES = ('speed_breaker', 'dip', 'junction')
LIGHTS = ('day', 'low_sun', 'twilight', 'night')
WEATHERS = ('clear', 'cloudy', 'wet', 'rain', 'fog', 'dust')
STATUSES = ('met', 'not_met', 'not_checkable')
THRESHOLDS = {'side_m': 2.0, 'in_lane_m': 1.0, 'cut_gap_max_m': 25.0, 'lane_m': 1.75, 'brake_decel_mps2': 3.0,
              'brake_window_s': 0.25, 'cross_heading_deg': [45.0, 135.0], 'stopped_mps': 0.3, 'stopped_s': 1.0,
              'ahead_max_m': 100.0, 'range_m': 60.0, 'moving_mps': 0.3, 'wrong_way_deg': 135.0,
              'overtake_lateral_m': 3.5, 'location_m': 30.0, 'ego_speed_tol_kmh': 5.0, 'ego_speed_tol_rel': 0.15,
              'ego_speed_window_s': 1.0, 'wet_level': 40.0, 'cloudy_level': 50.0}
NAMES = {'cut_in': 'cut-in', 'brake': 'hard braking', 'cross': 'crossing', 'stopped': 'standing still',
         'wrong_way': 'wrong-way', 'overtake': 'overtaking', 'speed_breaker': 'speed breaker',
         'dip': 'dip (pothole proxy)', 'junction': 'junction', 'low_sun': 'low sun'}


class CheckError(ValueError):
    pass


# ============================================================================================ the phrase set
def _alts(*words) -> str:
    return '|'.join(sorted((re.escape(w).replace(r'\ ', r'[\s-]+') for w in words), key=len, reverse=True))


KIND_WORDS = {
    'cycle_rickshaw': ('cycle rickshaw', 'pedal rickshaw'),
    'autorickshaw': ('autorickshaw', 'auto rickshaw', 'auto', 'tuk tuk', 'three wheeler', 'rickshaw', 'autos'),
    'motorcycle': ('motorcycle', 'motorbike', 'motor bike', 'scooter', 'two wheeler', 'bike', 'moped', 'scooty',
                   'motorcyclist', 'biker'),
    'bicycle': ('bicycle', 'cyclist', 'cycle', 'push bike'),
    'truck': ('truck', 'lorry', 'tanker'),
    'bus': ('bus', 'buses'),
    'van': ('van', 'minivan', 'tempo'),
    'car': ('car', 'sedan', 'hatchback', 'suv', 'taxi', 'cab', 'jeep'),
    'child': ('child', 'children', 'kid', 'boy', 'girl', 'schoolchild'),
    'pedestrian': ('pedestrian', 'person', 'man', 'woman', 'jaywalker', 'walker', 'people', 'men', 'women'),
    'cattle': ('cow', 'cattle', 'bull', 'buffalo', 'ox', 'bullock', 'calf'),
    'handcart': ('handcart', 'hand cart', 'pushcart', 'push cart', 'cart'),
    'vendor': ('vendor', 'hawker', 'street vendor'),
}
PHRASES = {
    'kind': [(k, re.compile(rf'\b(?:{_alts(*ws)})(?:s|es)?\b', re.I)) for k, ws in KIND_WORDS.items()],
    'manoeuvre': [
        ('cut_in', re.compile(r'\b(?:cut(?:s|ting)?[\s-]+in(?:to)?|cut[\s-]?in|swerv(?:es|ing|e)[\s-]+in(?:to)?|'
                              r'merg(?:es|ing|e)[\s-]+in[\s-]+front|pull(?:s|ing)?[\s-]+(?:out|in)(?:[\s-]+in[\s-]+'
                              r'front)?)\b', re.I)),
        ('brake', re.compile(r'\b(?:brak(?:es|ing|e)(?:[\s-]+hard)?|slams?[\s-]+on[\s-]+(?:the[\s-]+|its[\s-]+|his'
                             r'[\s-]+|her[\s-]+)?brakes?|stops?[\s-]+suddenly|emergency[\s-]+stop)\b', re.I)),
        ('cross', re.compile(r'\b(?:cross(?:es|ing)?(?![\s-]*roads?)|walks?[\s-]+across|runs?[\s-]+across|'
                             r'steps?[\s-]+out|darts?[\s-]+out|jaywalk(?:s|ing)?)\b', re.I)),
        ('stopped', re.compile(r'\b(?:stalled|broken[\s-]+down|breaks?[\s-]+down|stopped|parked|stationary|'
                               r'stand(?:s|ing)?[\s-]+(?:still|in)|stuck)\b', re.I)),
        ('wrong_way', re.compile(r'\b(?:wrong[\s-]+(?:way|side)|against[\s-]+(?:the[\s-]+)?(?:traffic|flow)|'
                                 r'contraflow)\b', re.I)),
        ('overtake', re.compile(r'\b(?:overtak(?:es|ing|e)|filter(?:s|ing)?|weav(?:es|ing|e)|pass(?:es|ing)?'
                                r'[\s-]+(?:us|the[\s-]+ego|you|me))\b', re.I))],
    'side': re.compile(r'\b(?:from|on)[\s-]+(?:the[\s-]+|your[\s-]+|our[\s-]+|its[\s-]+)?(left|right)\b|'
                       r'\b(left|right)[\s-]+(?:hand[\s-]+)?side\b', re.I),
    'location': [
        ('speed_breaker', re.compile(r'\b(?:speed[\s-]*(?:breakers?|bumps?|humps?|tables?)|humps?|bumps?|'
                                     r'breakers?)\b', re.I)),
        ('dip', re.compile(r'\b(?:pot[\s-]*holes?|dips?)\b', re.I)),
        ('junction', re.compile(r'\b(?:junctions?|intersections?|cross[\s-]*roads|roundabouts?|traffic[\s-]+'
                                r'(?:lights?|signals?))\b', re.I))],
    'within': re.compile(r'\bwithin\s+(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)\b', re.I),
    'light': [
        ('low_sun', re.compile(r'\b(?:low[\s-]+sun|setting[\s-]+sun|rising[\s-]+sun|sun[\s-]+glare|glare|'
                               r'golden[\s-]+hour)\b', re.I)),
        ('twilight', re.compile(r'\b(?:dusk|dawn|twilight|sunset|sunrise|evening|nightfall|daybreak)\b', re.I)),
        ('night', re.compile(r'\b(?:night(?:[\s-]*time)?|dark(?:ness)?|midnight)\b', re.I)),
        ('day', re.compile(r'\b(?:noon|midday|day[\s-]*time|daylight|afternoon|broad[\s-]+day)\b', re.I))],
    'weather': [
        ('dust', re.compile(r'\b(?:dust[\s-]*storm|sand[\s-]*storm|dusty)\b', re.I)),
        ('rain', re.compile(r'\b(?:rain(?:s|ing|y)?|monsoon|downpour|drizzl(?:e|ing)|showers?)\b', re.I)),
        ('fog', re.compile(r'\b(?:fog(?:gy)?|mist(?:y)?|haz(?:e|y)|smog)\b', re.I)),
        ('wet', re.compile(r'\b(?:wet[\s-]+roads?|wet|water[\s-]*logged|puddles?|flooded)\b', re.I)),
        ('cloudy', re.compile(r'\b(?:cloudy|overcast|clouds)\b', re.I)),
        ('clear', re.compile(r'\b(?:sunny|clear[\s-]+(?:sky|skies|day|weather)|fine[\s-]+weather)\b', re.I))],
    'ego_speed': re.compile(r'\b(?:(?:the[\s-]+)?ego(?:[\s-]+vehicle)?|we|i|our[\s-]+car)\b(?:[\s-]+[a-z]+){0,3}?'
                            r'[\s-]+(?:at|doing)[\s-]+(\d+(?:\.\d+)?)\s*(?:km/?h|kmph|kph)\b|\bdriving[\s-]+at[\s-]+'
                            r'(\d+(?:\.\d+)?)\s*(?:km/?h|kmph|kph)\b', re.I),
    'no_collision': re.compile(r'\b(?:without[\s-]+(?:a[\s-]+|any[\s-]+)?(?:collision|crash|contact)|no[\s-]+'
                               r'(?:collision|crash)|avoids?[\s-]+(?:a[\s-]+)?(?:collision|crash|it|them|hitting)|'
                               r'stops?[\s-]+in[\s-]+time)\b', re.I),
    'collision': re.compile(r'\b(?:collides?|collisions?|crash(?:es|ed)?|hits?)\b', re.I),
}
STOP = set('''a an the and or but with from into onto over under near at in on of to for by as is are was were be
been being it its this that these those there their they them then than while when where which who whom what into
just very quite some any each every other another one two three four five ahead behind front left right side lane
lanes road roads ego vehicle vehicles suddenly slowly quickly fast slow close closely past across along around
about after before during within metres meters km/h kmph kph our your you we i me us has have had does doing done
also only still even much many more most less least very really make makes made making goes going come comes
coming here away down onto upon same next first last while'''.split())


def _sentences(text: str) -> list:
    out, start = [], 0
    for m in re.finditer(r'[.;!?\n]+', text):
        out.append((start, m.end()))
        start = m.end()
    if start < len(text):
        out.append((start, len(text)))
    return out


def _claim(ctype: str, phrase: str, span, **fields) -> dict:
    return dict({'type': ctype, 'phrase': phrase, 'span': list(span), 'source': 'phrases'}, **fields)


def extract(text: str) -> dict:
    """The description's claims by the documented phrase set (module doc). Deterministic: the same text gives the
    same claims. {'claims': [...], 'recognised': [{'span', 'phrase', 'as'}], 'unrecognised': [words]}."""
    text = str(text or '')
    claims, covered, recog = [], [], []

    def note(m, what):
        covered.append((m.start(), m.end()))
        recog.append({'span': [m.start(), m.end()], 'phrase': m.group(0), 'as': what})

    kinds = []
    for k, rx in PHRASES['kind']:
        for m in rx.finditer(text):
            if any(a <= m.start() < b for a, b, *_ in kinds):
                continue                           # "cycle rickshaw" already, not also "cycle"
            kinds.append((m.start(), m.end(), k, m.group(0)))
    kinds.sort()
    seen_kinds = set()
    for a, b, k, ph in kinds:
        covered.append((a, b))
        recog.append({'span': [a, b], 'phrase': ph, 'as': f'actor {k}'})
        if k not in seen_kinds:
            seen_kinds.add(k)
            claims.append(_claim('actor', ph, (a, b), kind=k))
    sents = _sentences(text)
    events = []
    for what, rx in PHRASES['manoeuvre']:
        for m in rx.finditer(text):
            if what == 'cross' and any(m.start() >= a and m.end() <= b for a, b in covered):
                continue
            s0, s1 = next(((a, b) for a, b in sents if a <= m.start() < b), (0, len(text)))
            before = [x for x in kinds if s0 <= x[0] < m.start()]
            after = [x for x in kinds if m.end() <= x[0] < s1]
            kind = (before[-1] if before else after[0] if after else None)
            if what == 'stopped' and after and after[0][0] - m.end() <= 12:
                kind = after[0]                    # "a parked bus", "a stalled white car": the noun it describes
            side = None
            sm = PHRASES['side'].search(text, m.end(), s1) or PHRASES['side'].search(text, s0, m.start())
            if sm and what in ('cut_in', 'cross', 'overtake'):
                side = (sm.group(1) or sm.group(2)).lower()
                note(sm, f'side {side}')
            note(m, f'manoeuvre {what}')
            events.append((m.start(), _claim('manoeuvre', m.group(0), (m.start(), m.end()), what=what, side=side,
                                             kind=kind[2] if kind else None)))
    seen = set()
    for _s, c in sorted(events, key=lambda e: e[0]):
        key = (c['what'], c['side'], c['kind'])
        if key not in seen:
            seen.add(key)
            claims.append(c)
    wm = PHRASES['within'].search(text)
    if wm:
        note(wm, 'distance')
    for feat, rx in PHRASES['location']:
        ms = list(rx.finditer(text))
        m = ms[0] if ms else None
        for x in ms[1:]:
            note(x, f'location {feat}')
        if m:
            note(m, f'location {feat}')
            claims.append(_claim('location', m.group(0), (m.start(), m.end()), feature=feat,
                                 within_m=float(wm.group(1)) if wm else THRESHOLDS['location_m']))
    for group in ('light', 'weather'):
        for cls, rx in PHRASES[group]:
            ms = [x for x in rx.finditer(text) if not any(a <= x.start() < b for a, b in covered)]
            m = ms[0] if ms else None
            for x in ms[1:]:
                note(x, f'{group} {cls}')
            if m:
                note(m, f'{group} {cls}')
                claims.append(_claim(group, m.group(0), (m.start(), m.end()), **{'class': cls}))
    m = PHRASES['ego_speed'].search(text)
    if m:
        note(m, 'ego speed')
        claims.append(_claim('ego_speed', m.group(0), (m.start(), m.end()), kmh=float(m.group(1) or m.group(2))))
    m = PHRASES['no_collision'].search(text)
    if m:
        note(m, 'outcome: no collision')
        claims.append(_claim('outcome', m.group(0), (m.start(), m.end()), collision=False))
    else:
        m = PHRASES['collision'].search(text)
        if m:
            note(m, 'outcome: a collision')
            claims.append(_claim('outcome', m.group(0), (m.start(), m.end()), collision=True))
    for i, c in enumerate(claims):
        c['id'] = f'c{i + 1}'
    un = []
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]{3,}", text):
        if m.group(0).lower() in STOP or any(a <= m.start() < b for a, b in covered):
            continue
        if m.group(0).lower() not in un:
            un.append(m.group(0).lower())
    recog.sort(key=lambda r: r['span'])
    return {'claims': claims, 'recognised': recog, 'unrecognised': un}


# ============================================================================================ claims as data
CLAIM_SCHEMA = {
    'type': 'object', 'required': ['claims'], 'additionalProperties': False,
    'properties': {'claims': {'type': 'array', 'maxItems': 16, 'items': {
        'type': 'object', 'required': ['type'],
        'properties': {
            'type': {'enum': list(CLAIM_TYPES)}, 'phrase': {'type': 'string', 'maxLength': 120},
            'kind': {'enum': list(KIND_WORDS) + [None]}, 'what': {'enum': list(MANOEUVRES)},
            'side': {'enum': ['left', 'right', None]}, 'feature': {'enum': list(FEATURES)},
            'within_m': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 500},
            'class': {'enum': list(LIGHTS) + list(WEATHERS)}, 'kmh': {'type': 'number', 'minimum': 0,
                                                                      'maximum': 200},
            'collision': {'type': 'boolean'}},
        'additionalProperties': False}}}}
REQUIRED = {'actor': ('kind',), 'manoeuvre': ('what',), 'location': ('feature',), 'light': ('class',),
            'weather': ('class',), 'ego_speed': ('kmh',), 'outcome': ('collision',)}


def claim_problems(c: dict) -> list:
    """Why a claim (from the model or a file) is not one Falcon can check, or []."""
    import jsonschema
    v = jsonschema.Draft202012Validator(CLAIM_SCHEMA['properties']['claims']['items'])
    out = [e.message for e in v.iter_errors(c)]
    if not out:
        out += [f'a {c["type"]} claim needs {f}' for f in REQUIRED[c['type']] if c.get(f) is None]
        if c['type'] == 'light' and c.get('class') not in LIGHTS:
            out.append(f'light is one of {", ".join(LIGHTS)}')
        if c['type'] == 'weather' and c.get('class') not in WEATHERS:
            out.append(f'weather is one of {", ".join(WEATHERS)}')
        if c['type'] == 'actor' and c.get('kind') is None:
            out.append('an actor claim needs a kind')
    return out


def claims_prompt(text: str) -> list:
    """The messages that ask a model to turn a description into claims: the description and the claim schema — text
    only, nothing recorded."""
    return [{'role': 'system', 'content': (
        'You turn a driving-scenario description into checkable claims. Reply with one JSON object only, valid '
        'against this JSON Schema:\n' + json.dumps(CLAIM_SCHEMA, separators=(',', ':')) + '\nTypes: actor {kind}; '
        'manoeuvre {what, side, kind}: cut_in, brake, cross, stopped, wrong_way, overtake; location {feature, '
        'within_m}: speed_breaker, dip (a pothole), junction; light {class}: day, low_sun, twilight (dusk, dawn), '
        'night; weather {class}: clear, cloudy, wet, rain, fog, dust; ego_speed {kmh}; outcome {collision}. '
        '"phrase" is the words of the description the claim comes from. Only claims the description makes.')},
        {'role': 'user', 'content': str(text)}]


def parse_json_reply(reply: str):
    """The JSON object in a model's reply (a fenced block or the outermost braces), or raise ValueError."""
    s = str(reply or '').strip()
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', s, re.S)
    if m:
        s = m.group(1)
    elif not s.startswith('{'):
        a, b = s.find('{'), s.rfind('}')
        if a < 0 or b <= a:
            raise ValueError('the reply holds no JSON object')
        s = s[a:b + 1]
    try:
        doc = json.loads(s)
    except ValueError as e:
        raise ValueError(f'the reply is not valid JSON ({e.msg} at line {e.lineno} column {e.colno})') from None
    if not isinstance(doc, dict):
        raise ValueError('the reply is JSON but not an object')
    return doc


def parse_model_claims(reply: str) -> tuple:
    """(claims the model gave that Falcon can check, [rejected with why])."""
    doc = parse_json_reply(reply)
    good, bad = [], []
    for c in doc.get('claims') or []:
        if not isinstance(c, dict):
            bad.append({'claim': c, 'why': 'not an object'})
            continue
        why = claim_problems(c)
        if why:
            bad.append({'claim': c, 'why': '; '.join(why)})
            continue
        good.append(dict(c, source='model'))
    return good, bad


def _key(c: dict) -> tuple:
    """What a claim is about: one light, ego speed and outcome per description; the rest by their subject."""
    t = c['type']
    return {'actor': (t, c.get('kind')), 'manoeuvre': (t, c.get('what'), c.get('kind')),
            'location': (t, c.get('feature')), 'weather': (t, c.get('class'))}.get(t, (t,))


def merge(phrases: list, model: list) -> list:
    """The phrase claims, then the model's claims of a kind not already there; ids renumbered."""
    have = {_key(c) for c in phrases}
    out = [dict(c) for c in phrases]
    for c in model:
        if _key(c) not in have:
            have.add(_key(c))
            out.append(dict(c))
    for i, c in enumerate(out):
        c['id'] = f'c{i + 1}'
    return out


# ============================================================================================ the run
def _read(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class RunFacts:
    """What a run recorded, loaded once: run.json, the drive config, trajectories.json in the ego's frame."""

    def __init__(self, run_dir: str):
        if not os.path.isfile(os.path.join(run_dir, 'run.json')):
            raise CheckError(f'{run_dir}: no run.json (not a recorded run)')
        self.dir = run_dir
        self.run = _read(os.path.join(run_dir, 'run.json')) or {}
        self.cfg = None
        for f in ('drive.json', 'stack.json'):
            d = _read(os.path.join(run_dir, f))
            if isinstance(d, dict):
                self.cfg = d
                break
        if self.cfg is None:
            self.cfg = ((self.run.get('provenance') or {}).get('settings')) or {}
        self.sc = self.run.get('scenario') if isinstance(self.run.get('scenario'), dict) else None
        self.scn = (self.cfg.get('scenario') or {}) if isinstance(self.cfg.get('scenario'), dict) else {}
        self.kinds = {a.get('id'): a.get('kind') for a in self.scn.get('actors') or () if isinstance(a, dict)}
        tj = _read(os.path.join(run_dir, 'trajectories.json'))
        self.traj = None
        if isinstance(tj, dict) and (tj.get('actors') or {}).get('ego'):
            cols = tj.get('columns') or ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'length', 'width']
            ci = {c: i for i, c in enumerate(cols)}
            self.traj = {}
            for name, rows in tj['actors'].items():
                if rows:
                    A = np.asarray(rows, float)
                    self.traj[name] = {k: A[:, ci[k]] for k in ('t', 'x', 'y', 'yaw', 'vx', 'vy', 'length', 'width')
                                       if k in ci}
        self._lang = None
        self._map = None

    # ---------------------------------------------------------------- the ego's frame
    def relative(self, name: str) -> dict | None:
        """An actor's series in the ego's frame: t, lon (its centre ahead of the ego's), lat (+ right), gap (its rear
        ahead of the ego's front; negative behind or overlapping), relative heading (deg), its speed (m/s)."""
        if not self.traj or name not in self.traj or name == 'ego':
            return None
        e, a = self.traj['ego'], self.traj[name]
        n = min(len(e['t']), len(a['t']))
        if n < 2:
            return None
        ex, ey, ey_ = e['x'][:n], e['y'][:n], np.radians(e['yaw'][:n])
        dx, dy = a['x'][:n] - ex, a['y'][:n] - ey
        c, s = np.cos(ey_), np.sin(ey_)
        lon, lat = dx * c + dy * s, -dx * s + dy * c
        le = float(e['length'][0]) if 'length' in e else 4.6
        la = a['length'][:n] if 'length' in a else np.full(n, 4.6)
        gap = lon - le / 2.0 - la / 2.0
        rel = (a['yaw'][:n] - e['yaw'][:n] + 180.0) % 360.0 - 180.0
        v = np.hypot(a['vx'][:n], a['vy'][:n])
        return {'t': a['t'][:n], 'lon': lon, 'lat': lat, 'gap': gap, 'rel': rel, 'v': v, 'ego_x': ex, 'ego_y': ey}

    def actors_of(self, kind: str | None) -> list:
        names = [n for n in (self.traj or {}) if n != 'ego']
        if kind is None:
            return names
        return [n for n in names if self.kinds.get(n) == kind]

    def step_of(self, j: int):
        """The recorded step of trajectories.json row j (row j is the j-th recorded frame)."""
        fr = self._frames()
        return fr[j]['step'] if j < len(fr) else None

    def _frames(self):
        if not hasattr(self, '_fr'):
            from .worldmodel import _jsonl
            p = os.path.join(self.dir, 'frames.jsonl')
            self._fr = _jsonl(p) if os.path.isfile(p) else []
        return self._fr

    # ---------------------------------------------------------------- the grounded words
    def language_at(self, j: int, name: str) -> dict | None:
        """core.language's facts for the actor at trajectories row j (its direction words, lane relation, gap), or
        None when the run has no actors.jsonl."""
        try:
            from . import language as L
            from . import worldmodel as WM
            if self._lang is None:
                data = WM.load_run(self.dir)
                if data['actors'] is None:
                    self._lang = False
                else:
                    ctx = L._scenario_context(data)
                    vel, _ = L._velocities(data, ctx)
                    li = L.LaneIndex(data['lanes']) if data['lanes'] and data['lanes'].get('points') else None
                    self._lang = (data, ctx, vel, li)
            if not self._lang:
                return None
            data, ctx, vel, li = self._lang
            aid = ((ctx or {}).get('actor_ids') or {}).get(name)
            step = self.step_of(j)
            if aid is None or step is None or step not in data['actors']:
                return None
            line = [dict(a, vx=vel[(step, a['id'])][0], vy=vel[(step, a['id'])][1]) if (step, a['id']) in vel
                    else a for a in data['actors'][step]]
            ego = next((a for a in line if a['id'] == data['ego_id']), None)
            if ego is None:
                return None
            st = L.build_state(step, 0.0, ego, [a for a in line if a['id'] != data['ego_id']], li)
            g = next((g for g in st['agents'] if g['id'] == aid), None)
            if g is None:
                return None
            lane = L.lane_phrase(g['lane_rel'], g['lanes_over']) if g.get('lane_rel') else None
            noun = str(self.kinds.get(name) or g['noun']).replace('_', ' ')     # the scenario's kind, not CARLA's
            return {'words': f'{L._article(noun)} {noun} {g["direction"]}' + (f', {lane}' if lane else ''),
                    'direction': g['direction'], 'lane': lane, 'distance_m': g['distance_m'], 'carla_id': aid,
                    'step': step}
        except Exception:                 # noqa: BLE001 (the words are extra: the numbers stand without them)
            return None

    # ---------------------------------------------------------------- the map
    def features(self) -> tuple:
        """([{'kind', 'x', 'y'} in CARLA's frame], source) of the map's OpenDRIVE road features, or (None, why)."""
        if self._map is None:
            from . import roadfeatures as RF
            from .rules import find_xodr
            xp, why = find_xodr(self.cfg)
            if not xp:
                self._map = (None, why)
            else:
                try:
                    with open(xp, encoding='utf-8') as fh:
                        feats = RF.read(fh.read())
                    self._map = ([{'kind': f.get('kind'), 'x': float(f['x']), 'y': -float(f['y']),
                                   'crossing_kmh': f.get('crossing_kmh')} for f in feats
                                  if f.get('x') is not None and f.get('y') is not None], os.path.basename(xp))
                except (OSError, ValueError, KeyError) as e:
                    self._map = (None, f'{os.path.basename(xp)} could not be read ({e})')
        return self._map


# ============================================================================================ the checks
def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def _res(status, measured, text, why=''):
    return {'status': status, 'measured': measured, 'value': text, **({'why': why} if why else {})}


def _side(lat):
    return 'left' if lat < 0 else 'right'


def _smooth(x, n):
    if n <= 1 or len(x) < n:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode='same')


def _find_cut_in(r, T, side):
    near = np.where(np.abs(r['lon']) <= T['range_m'])[0]
    if not len(near):
        return None, {'note': 'never within range'}
    i0 = int(near[0])
    lat0 = float(r['lat'][i0])
    best = {'start_lateral_m': _r(lat0), 'from': _side(lat0), 'min_abs_lateral_m': _r(np.min(np.abs(r['lat'][i0:])))}
    if abs(lat0) < T['side_m']:
        return None, dict(best, note=f'it started {abs(lat0):.2f} m from your centre line (a cut-in starts at least '
                                     f'{T["side_m"]:g} m to the side)')
    inside = np.where((np.arange(len(r['t'])) > i0) & (np.abs(r['lat']) <= T['in_lane_m']) & (r['gap'] >= 0))[0]
    if not len(inside):
        return None, dict(best, note=f'it never came within {T["in_lane_m"]:g} m of your centre line ahead of you')
    j = int(inside[0])
    m = dict(best, entered_at_s=_r(r['t'][j]), gap_m=_r(r['gap'][j]), lateral_at_entry_m=_r(r['lat'][j]))
    if r['gap'][j] > T['cut_gap_max_m']:
        return None, dict(m, note=f'it moved in {r["gap"][j]:.1f} m ahead (a cut-in is within '
                                  f'{T["cut_gap_max_m"]:g} m)')
    if side and _side(lat0) != side:
        return None, dict(m, note=f'it came from the {_side(lat0)}, not the {side}')
    return j, m


def _find_brake(r, T, side):
    t = r['t']
    if len(t) < 3:
        return None, {'note': 'too few samples'}
    dt = float(np.median(np.diff(t)))
    dec = -_smooth(np.gradient(r['v'], t), max(1, int(round(T['brake_window_s'] / max(dt, 1e-6)))))
    ok = (r['gap'] >= 0) & (np.abs(r['lat']) <= T['lane_m'])
    if not ok.any():
        return None, {'note': 'never ahead in your lane'}
    k = int(np.argmax(np.where(ok, dec, -np.inf)))
    m = {'peak_decel_mps2': _r(dec[k]), 'at_s': _r(t[k]), 'gap_m': _r(r['gap'][k])}
    if dec[k] < T['brake_decel_mps2']:
        return None, dict(m, note=f'its strongest braking ahead of you was {dec[k]:.2f} m/s²')
    j = int(np.where(ok & (dec >= T['brake_decel_mps2']))[0][0])
    return j, dict(m, from_s=_r(t[j]))


def _find_cross(r, T, side):
    lo, hi = T['cross_heading_deg']
    across = (np.abs(r['rel']) >= lo) & (np.abs(r['rel']) <= hi) & (r['v'] > T['moving_mps'])
    if not across.any():
        return None, {'note': 'it never moved across your heading'}
    inside = np.where((np.abs(r['lat']) <= T['in_lane_m']) & (r['lon'] > 0))[0]
    out = np.where(np.abs(r['lat']) >= T['side_m'])[0]
    m = {'min_abs_lateral_m': _r(np.min(np.abs(r['lat'])))}
    for j in inside:
        before = out[out < j]
        if len(before) and across[:j + 1].any():
            i = int(before[-1])
            m.update(entered_at_s=_r(r['t'][j]), ahead_m=_r(r['lon'][j]), from_=_side(r['lat'][i]))
            m['from'] = m.pop('from_')
            if side and m['from'] != side:
                return None, dict(m, note=f'it came from the {m["from"]}, not the {side}')
            return int(j), m
    return None, dict(m, note=f'it never entered your path (within {T["in_lane_m"]:g} m of your centre line) from '
                              f'{T["side_m"]:g} m or more to the side')


def _find_stopped(r, T, side):
    t = r['t']
    ok = (r['lon'] > 0) & (r['lon'] <= T['ahead_max_m']) & (np.abs(r['lat']) <= T['lane_m']) & \
        (r['v'] <= T['stopped_mps'])
    best, cur, start = 0.0, 0.0, None
    first = None
    for k in range(len(t)):
        if ok[k]:
            start = k if start is None else start
            cur = t[k] - t[start]
            if cur >= T['stopped_s'] and first is None:
                first = start
            best = max(best, cur)
        else:
            start = None
    m = {'longest_still_s': _r(best)}
    if first is None:
        return None, dict(m, note=f'it never stood still ahead in your lane for {T["stopped_s"]:g} s')
    return first, dict(m, ahead_m=_r(r['lon'][first]), at_s=_r(t[first]))


def _find_wrong_way(r, T, side):
    ok = (np.abs(r['rel']) >= T['wrong_way_deg']) & (np.abs(r['lat']) <= T['lane_m']) & (r['lon'] > 0) & \
        (r['v'] > T['moving_mps'])
    if not ok.any():
        return None, {'max_heading_diff_deg': _r(np.max(np.abs(r['rel']))),
                      'note': 'it never moved towards you in your lane'}
    j = int(np.where(ok)[0][0])
    return j, {'heading_diff_deg': _r(abs(r['rel'][j])), 'ahead_m': _r(r['lon'][j]), 'at_s': _r(r['t'][j])}


def _find_overtake(r, T, side):
    lon = r['lon']
    cross = np.where((lon[:-1] < 0) & (lon[1:] >= 0))[0]
    if not len(cross):
        return None, {'note': 'it never passed from behind you to ahead of you'}
    j = int(cross[0]) + 1
    m = {'lateral_at_pass_m': _r(r['lat'][j]), 'side': _side(r['lat'][j]), 'at_s': _r(r['t'][j])}
    if abs(r['lat'][j]) > T['overtake_lateral_m']:
        return None, dict(m, note=f'it passed {abs(r["lat"][j]):.2f} m to the side (an overtake here is within '
                                  f'{T["overtake_lateral_m"]:g} m)')
    if side and m['side'] != side:
        return None, dict(m, note=f'it passed on your {m["side"]}, not your {side}')
    return j, m


FINDERS = {'cut_in': _find_cut_in, 'brake': _find_brake, 'cross': _find_cross, 'stopped': _find_stopped,
           'wrong_way': _find_wrong_way, 'overtake': _find_overtake}


def check_actor(f: RunFacts, c: dict, T: dict, ctx: dict) -> dict:
    if not f.scn or f.sc is None:
        return _res('not_checkable', None, '', 'not a scenario run: background traffic is not identified by kind')
    names = [a for a, k in f.kinds.items() if k == c['kind']]
    if not names:
        have = sorted({k for k in f.kinds.values() if k})
        return _res('not_met', {'kinds': have},
                    f'no {c["kind"]}: the scenario\'s actors are {", ".join(have) or "none"}')
    ids = f.sc.get('actor_ids') or {}
    subs = f.sc.get('subclasses') or {}
    rows = []
    for n in names:
        tr = (f.traj or {}).get(n)
        rec = float(tr['t'][-1] - tr['t'][0]) if tr is not None and len(tr['t']) >= 2 else 0.0
        rows.append({'id': n, 'carla_id': ids.get(n), 'blueprint': (f.sc.get('blueprints') or {}).get(n),
                     'subclass': subs.get(n), 'recorded_s': _r(rec)})
    ok = [r for r in rows if r['carla_id'] is not None and r['recorded_s'] > 0]
    stand_in = c['kind'] in ('autorickshaw', 'cycle_rickshaw', 'cattle', 'handcart', 'vendor')
    if stand_in:
        wrong = [r for r in ok if r['subclass'] is not None and str(r['subclass']).split(' (')[0] != c['kind']]
        ok = [r for r in ok if r not in wrong]
        if wrong and not ok:
            return _res('not_met', {'actors': rows}, f'{wrong[0]["id"]} is a {c["kind"]} in the scenario but was '
                                                      f'recorded as {wrong[0]["subclass"]}')
    if not ok:
        return _res('not_met', {'actors': rows}, f'{c["kind"]} in the scenario but not spawned and recorded')
    r = ok[0]
    bp = f', blueprint {r["blueprint"]}' if r['blueprint'] else ''
    sub = f', subclass {r["subclass"]}' if r['subclass'] else (' (no subclass recorded)' if stand_in else '')
    return _res('met', {'actors': rows}, f'{r["id"]}: CARLA id {r["carla_id"]}{bp}{sub}, recorded '
                                         f'{r["recorded_s"]:.1f} s')


def check_manoeuvre(f: RunFacts, c: dict, T: dict, ctx: dict) -> dict:
    if f.traj is None:
        return _res('not_checkable', None, '', 'no trajectories.json: not a scenario run (the director records them)')
    names = f.actors_of(c.get('kind'))
    if not names:
        return _res('not_met', None, f'no {c["kind"]} among the scenario\'s actors' if c.get('kind')
                    else 'no scenario actor besides the ego')
    tried = []
    for n in names:
        r = f.relative(n)
        if r is None:
            continue
        j, m = FINDERS[c['what']](r, T, c.get('side'))
        m = {'actor': n, **{k: v for k, v in m.items()}}
        if j is not None:
            words = {k: f.language_at(i, n) for k, i in (('first', 0), ('event', j))}
            if any(words.values()):
                m['language'] = words
            ctx.setdefault('events', []).append({'claim': c['id'], 'row': j, 't': float(r['t'][j]),
                                                 'ego_x': float(r['ego_x'][j]), 'ego_y': float(r['ego_y'][j])})
            return _res('met', m, _describe(c['what'], m))
        tried.append(m)
    return _res('not_met', {'actors': tried}, '; '.join(f'{m["actor"]}: {m.get("note", "not found")}' for m in tried))


def _describe(what, m) -> str:
    if what == 'cut_in':
        return (f'{m["actor"]} from the {m["from"]} ({abs(m["start_lateral_m"]):.2f} m to the side) into your lane '
                f'at {m["entered_at_s"]:.2f} s, {m["gap_m"]:.1f} m ahead')
    if what == 'brake':
        return f'{m["actor"]} braked at {m["peak_decel_mps2"]:.1f} m/s² (peak, at {m["at_s"]:.2f} s), ' \
               f'{m["gap_m"]:.1f} m ahead'
    if what == 'cross':
        return f'{m["actor"]} entered your path from the {m["from"]} at {m["entered_at_s"]:.2f} s, ' \
               f'{m["ahead_m"]:.1f} m ahead'
    if what == 'stopped':
        return f'{m["actor"]} stood still {m["ahead_m"]:.1f} m ahead in your lane for {m["longest_still_s"]:.1f} s'
    if what == 'wrong_way':
        return f'{m["actor"]} came towards you in your lane ({m["heading_diff_deg"]:.0f}° from your heading), ' \
               f'{m["ahead_m"]:.1f} m ahead at {m["at_s"]:.2f} s'
    return f'{m["actor"]} passed on your {m["side"]} {abs(m["lateral_at_pass_m"]):.2f} m to the side at ' \
           f'{m["at_s"]:.2f} s'


def check_location(f: RunFacts, c: dict, T: dict, ctx: dict) -> dict:
    within = float(c.get('within_m') or T['location_m'])
    if f.traj is not None:
        ego = f.traj['ego']
        ex, ey, et = ego['x'], ego['y'], ego['t']
    else:
        fr = [x for x in f._frames() if x.get('ego')]
        if not fr:
            return _res('not_checkable', None, '', 'no recorded ego positions')
        ex = np.array([x['ego']['x'] for x in fr])
        ey = np.array([x['ego']['y'] for x in fr])
        et = np.array([float(x.get('sim_t', 0.0)) for x in fr])
    ev = (ctx.get('events') or [None])[0]
    at = f'at the {NAMES.get(ev["what"], ev["what"])} ({ev["t"]:.2f} s)' if ev else 'at its closest'
    if c['feature'] == 'junction':
        from . import rules as RU
        try:
            tr = RU.trace_from_run(f.dir)
            sig = RU.signals(tr)
        except Exception as e:            # noqa: BLE001 (a run the rules cannot read: said)
            return _res('not_checkable', None, '', f'the map could not be read ({e})')
        jn = sig['in_junction']
        if np.isnan(jn).all():
            return _res('not_checkable', None, '', 'no lanes.json and no OpenDRIVE file: junctions unknown')
        pts = np.column_stack([tr.x, tr.y])[jn > 0.5]
        if not len(pts):
            return _res('not_met', {'junction_samples': 0}, 'the ego never entered a junction')
        px, py = (ev['ego_x'], ev['ego_y']) if ev else (tr.x[0], tr.y[0])
        d = float(np.min(np.hypot(pts[:, 0] - px, pts[:, 1] - py)))
        if not ev:
            d = 0.0
        m = {'distance_m': _r(d), 'within_m': within, 'map': tr.sources.get('xodr') or 'lanes.json'}
        return _res('met' if d <= within else 'not_met', m, f'{d:.1f} m from a junction {at}')
    feats, src = f.features()
    if feats is None:
        return _res('not_checkable', None, '', f'the map\'s road features are unknown: {src}')
    want = [x for x in feats if (x['kind'] == 'dip') == (c['feature'] == 'dip')
            and (c['feature'] == 'dip' or x.get('crossing_kmh') is not None)]
    if not want:
        return _res('not_met', {'features': len(feats), 'map': src},
                    f'the map ({src}) carries no {NAMES[c["feature"]]}')
    F = np.array([[x['x'], x['y']] for x in want])
    if ev:
        d = np.hypot(F[:, 0] - ev['ego_x'], F[:, 1] - ev['ego_y'])
        k = int(np.argmin(d))
        dist, t_at = float(d[k]), ev['t']
    else:
        D = np.hypot(F[:, 0][None, :] - np.asarray(ex)[:, None], F[:, 1][None, :] - np.asarray(ey)[:, None])
        i, k = np.unravel_index(int(np.argmin(D)), D.shape)
        dist, t_at = float(D[i, k]), float(et[i] - et[0]) if len(et) else 0.0
    m = {'distance_m': _r(dist), 'within_m': within, 'feature': want[k]['kind'], 'at_s': _r(t_at), 'map': src,
         'features_on_map': len(want)}
    return _res('met' if dist <= within else 'not_met', m,
                f'the nearest {NAMES[c["feature"]]} ({want[k]["kind"]}) {dist:.1f} m from the ego {at}')


def _weather(f: RunFacts):
    w = f.cfg.get('weather')
    return w if isinstance(w, dict) and w else None


def check_light(f: RunFacts, c: dict, T: dict, ctx: dict) -> dict:
    from . import oddcoverage as OC
    from . import weather as W
    w = _weather(f)
    if w is None:
        return _res('not_checkable', None, '', 'the run\'s weather is not recorded')
    sun = W.resolve({k: v for k, v in w.items() if k in W.FIELDS or k == 'preset'})['sun_altitude_angle']
    cls = OC.light_class(sun)
    m = {'sun_altitude_deg': _r(sun, 1), 'class': cls,
         'classes': 'night < −6°, twilight −6° to 6°, low sun 6° to 20°, day ≥ 20°'}
    return _res('met' if cls == c['class'] else 'not_met', m,
                f'sun {sun:.1f}° above the horizon: {NAMES.get(cls, cls)}')


def check_weather(f: RunFacts, c: dict, T: dict, ctx: dict) -> dict:
    from . import oddcoverage as OC
    from . import weather as W
    w = _weather(f)
    if w is None:
        return _res('not_checkable', None, '', 'the run\'s weather is not recorded')
    d = W.resolve({k: v for k, v in w.items() if k in W.FIELDS or k == 'preset'})
    cls = OC.weather_class(w)
    m = {'class': cls, **{k: _r(d[k], 1) for k in ('precipitation', 'wetness', 'precipitation_deposits',
                                                    'cloudiness', 'fog_density', 'fog_distance', 'dust_storm')}}
    want = c['class']
    if want == 'wet':
        ok = max(d['wetness'], d['precipitation_deposits']) >= T['wet_level']
        txt = f'wetness {d["wetness"]:.0f}, puddles {d["precipitation_deposits"]:.0f} (wet at ≥ {T["wet_level"]:g})'
    elif want == 'cloudy':
        ok = d['cloudiness'] >= T['cloudy_level']
        txt = f'cloudiness {d["cloudiness"]:.0f} (cloudy at ≥ {T["cloudy_level"]:g})'
    else:
        ok = cls == want
        txt = f'the run\'s weather class is {cls} (rain {d["precipitation"]:.0f}, fog {d["fog_density"]:.0f}, ' \
              f'dust {d.get("dust_storm", 0.0):.0f}, cloud {d["cloudiness"]:.0f})'
    return _res('met' if ok else 'not_met', m, txt)


def check_ego_speed(f: RunFacts, c: dict, T: dict, ctx: dict) -> dict:
    if f.traj is None:
        return _res('not_checkable', None, '', 'no trajectories.json')
    e = f.traj['ego']
    sel = e['t'] - e['t'][0] <= T['ego_speed_window_s']
    v = float(np.median(np.hypot(e['vx'][sel], e['vy'][sel]))) * 3.6
    tol = max(T['ego_speed_tol_kmh'], T['ego_speed_tol_rel'] * float(c['kmh']))
    m = {'kmh': _r(v, 1), 'claimed_kmh': c['kmh'], 'tolerance_kmh': _r(tol, 1)}
    return _res('met' if abs(v - float(c['kmh'])) <= tol else 'not_met', m,
                f'the ego at {v:.1f} km/h over its first {T["ego_speed_window_s"]:g} s (±{tol:.1f})')


def check_outcome(f: RunFacts, c: dict, T: dict, ctx: dict) -> dict:
    if f.sc is None:
        return _res('not_checkable', None, '', 'not a scenario run: collisions are recorded by the director')
    n = len(f.sc.get('collisions') or [])
    m = {'collisions': n}
    ok = (n > 0) == bool(c['collision'])
    return _res('met' if ok else 'not_met', m, f'{n} collision event{"s" if n != 1 else ""} recorded')


CHECKS = {'actor': check_actor, 'manoeuvre': check_manoeuvre, 'location': check_location, 'light': check_light,
          'weather': check_weather, 'ego_speed': check_ego_speed, 'outcome': check_outcome}


def claim_text(c: dict) -> str:
    """A claim in words."""
    t = c['type']
    if t == 'actor':
        return f'an actor of kind {c["kind"]}'
    if t == 'manoeuvre':
        who = c.get('kind') or 'an actor'
        side = f' from the {c["side"]}' if c.get('side') and c['what'] in ('cut_in', 'cross') else \
            (f' on the {c["side"]}' if c.get('side') else '')
        return f'{who}: {NAMES.get(c["what"], c["what"])}{side}'
    if t == 'location':
        return f'within {float(c.get("within_m") or THRESHOLDS["location_m"]):g} m of a {NAMES[c["feature"]]}'
    if t in ('light', 'weather'):
        return f'{t}: {NAMES.get(c["class"], c["class"])}'
    if t == 'ego_speed':
        return f'the ego at {c["kmh"]:g} km/h'
    return 'a collision' if c['collision'] else 'no collision'


def check_run(run_dir: str, description: str | None = None, claims: list | None = None,
              thresholds: dict | None = None, write: bool = True, model_claims: list | None = None) -> dict:
    """The report (module doc): every claim with its status, measured values and the thresholds used. The claims
    come from the description by the phrase set (with the model's additions, model_claims, when given) or are given.
    Written to RUN/scenario-check.json unless write is False."""
    T = dict(THRESHOLDS, **(thresholds or {}))
    f = RunFacts(run_dir)
    if description is None and claims is None:
        src = (f.scn.get('source') or {}) if isinstance(f.scn.get('source'), dict) else {}
        description = src.get('description')
        if not description:
            raise CheckError('no description given and the run\'s scenario was not written from one')
    ex = extract(description) if claims is None else {'claims': [dict(c) for c in claims], 'recognised': [],
                                                       'unrecognised': []}
    cl = merge(ex['claims'], model_claims or [])
    for c in cl:
        why = claim_problems({k: v for k, v in c.items() if k not in ('id', 'span', 'source')})
        if why:
            raise CheckError(f'claim {c.get("id")}: {"; ".join(why)}')
    ctx: dict = {}
    order = sorted(cl, key=lambda c: 0 if c['type'] in ('manoeuvre', 'actor') else 1)    # events before places
    out = {}
    for c in order:
        r = CHECKS[c['type']](f, c, T, ctx)
        if c['type'] == 'manoeuvre' and r['status'] == 'met':
            ctx['events'][-1]['what'] = c['what']
        out[c['id']] = dict(c, claim=claim_text(c), **r)
    rows = [out[c['id']] for c in cl]
    counts = {s: sum(r['status'] == s for r in rows) for s in STATUSES}
    rep = {'schema': SCHEMA, 'run': os.path.abspath(run_dir), 'description': description,
           'scenario': {k: (f.scn or {}).get(k) for k in ('id', 'name')} if f.scn else None,
           'claims': rows, 'counts': counts,
           'verdict': ('every claim met' if rows and counts['met'] == len(rows) else
                       f'{counts["not_met"]} not met' + (f', {counts["not_checkable"]} not checkable'
                                                          if counts['not_checkable'] else '')
                       if rows else 'no claim recognised'),
           'recognised': ex['recognised'], 'unrecognised': ex['unrecognised'], 'thresholds': T,
           'sources': {'trajectories': 'trajectories.json (the director\'s record)' if f.traj else None,
                       'weather': 'drive.json weather' if _weather(f) else None,
                       'language': 'core.language.build_state on actors.jsonl' if f._lang else None},
           'about': ('each claim checked against the recorded ground truth; thresholds are Falcon\'s heuristics, '
                     'stated (docs/scenario-text.md)')}
    if write:
        from .provenance import write_json
        write_json(os.path.join(run_dir, REPORT), rep)
    return rep


def table(rep: dict) -> str:
    """The report as text: one line per claim."""
    mark = {'met': 'met', 'not_met': 'NOT MET', 'not_checkable': 'not checkable'}
    lines = [f'{rep["description"]!r}', f'run {os.path.basename(rep["run"])}: {rep["verdict"]}']
    for r in rep['claims']:
        lines.append(f'  {mark[r["status"]]:<14} {r["claim"]:<44} {r["value"] or r.get("why", "")}')
    if rep.get('unrecognised'):
        lines.append(f'  not recognised: {", ".join(rep["unrecognised"])}')
    return '\n'.join(lines)
