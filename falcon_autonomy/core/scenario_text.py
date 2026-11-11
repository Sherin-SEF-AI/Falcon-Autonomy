"""A scenario written from a description in words: the model writes a Falcon scenario as one JSON object, Falcon
validates it against the scenario schema (schemas/scenario.schema.json) and core.scenarios.validate, feeds the errors
back for a bounded number of repair rounds, and checks the result against the description's claims
(core.scenario_check) before anything runs (standard library + jsonschema). docs/scenario-text.md is the manual.

    write(description, complete, handedness, rounds) -> record     falcon.scenario_text/1
    draft(claims, handedness) -> (scenario | None, notes)          Falcon's own scenario from the phrase claims
    problems(scenario) -> [errors];  mismatches(scenario, claims, handedness) -> [what the description says and the
    scenario does not]

complete(messages) -> str is the caller's: the assistant's main model through assistant.service (Groq), or a test's
fake. This module opens no connection. What is sent (build_messages, repair_message), text only: the description, the
scenario schema, the rules of the format in a paragraph, one scenario as an example (Falcon's draft from the
description's phrases, else the library's autorickshaw cut-in), and in a repair round the validation errors and the
mismatches. Never a run, an image, lidar, audio or anything from the dashcam footage.

Rounds. The first reply, then up to `rounds` repairs (default 2, at most 4): a reply that is not a JSON object, that the
schema refuses, or that core.scenarios.validate refuses is INVALID and is sent back with its errors; a valid scenario
whose content does not say what the description says (mismatches: an actor kind missing, a cut-in without a lane change
into the ego's lane or from the other side, no speed-breaker need, a sun altitude outside the light class, …) is sent
back too while rounds remain. The record keeps every attempt's errors. The result is the last valid scenario (with
its mismatches listed) or none.

Without a model (offline, or the assistant off), draft() maps the claims onto a library scenario: the manoeuvre picks
the base (cut-in: the autorickshaw cut-in for an autorickshaw, else the cut-in; braking: the lead brake; crossing: the
pedestrian or cyclist crossing; standing still: stray cattle for cattle, else the stalled vehicle; wrong way: the
wrong-way scooter; overtaking: two-wheeler filtering), the claimed kind replaces its actor's, the side moves the actor
to the kerb (or the far sidewalk) when it is not the overtaking side of the world's traffic, the light and weather
classes set core.oddcoverage's representative weather (CELL_SUN, CELL_WEATHER), a speed breaker adds the need
"speed_breaker" (a breaker 10–60 m ahead of the start, on an imported world with road features), the ego's speed
fixes its parameter. A description without a manoeuvre has no draft.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

from . import scenario_check as CK
from . import scenarios as SC

SCHEMA = 'falcon.scenario_text/1'
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'schemas',
                           'scenario.schema.json')
MAX_ROUNDS = 4
ROUNDS = 2
FOLDER = 'scenario-text'
BASES = {'cut_in': {'autorickshaw': 'autorickshaw_cut_in', None: 'cut_in'},
         'brake': {None: 'lead_brake'},
         'cross': {'pedestrian': 'pedestrian_crossing', 'child': 'pedestrian_crossing', None: 'cyclist_crossing'},
         'stopped': {'cattle': 'stray_cattle', None: 'stalled_vehicle'},
         'wrong_way': {None: 'wrong_way_scooter'},
         'overtake': {None: 'two_wheeler_filtering'}}
WALKERS = ('pedestrian', 'child')
RULES = ('Lanes are relative to the ego and the traffic\'s handedness: "adjacent" is the next lane in the ego\'s '
         'direction on the overtaking side ({over} in this world), "kerb" the near edge of the road ({kerb} here), '
         '"sidewalk" beyond the kerb, "far_sidewalk" beyond the oncoming lanes, "opposite" the nearest oncoming lane, '
         '"line" the lane line between the ego\'s lane and the adjacent one. ahead_m is along the road from the '
         'ego (negative: behind). A "lane_change" action moves an actor to "ego", "adjacent" or "kerb" over '
         'duration_s. Triggers: time (t_s), distance (the ego within lt_m of the actor), ahead (the actor\'s rear '
         'more than gt_m ahead of the ego\'s front). Pedestrians and children "walk"; vehicles change "speed", '
         '"brake", "lane_change", "go", "stop"; a parked car can "open_door". Use parameters ("$name") with '
         'distributions for what the description leaves open, and keep their ranges physically plausible. Set '
         '"weather" for the light and weather the description names: sun_altitude_angle -30 at night, 0 at dusk or '
         'dawn (twilight is -6 to 6), 12 for a low sun, 45 by day; precipitation 60 for rain, fog_density 50 with '
         'fog_distance 0 for fog, wetness 60 and precipitation_deposits 50 for a wet road. Add "speed_breaker" to '
         '"needs" when the description places the event at a speed breaker (the start then has one 10-60 m ahead), '
         '"adjacent_same" when an actor starts in the adjacent lane. Actor kinds: ' + ', '.join(SC.KINDS) +
         ' (autorickshaw and cycle_rickshaw are stand-in vehicles; cattle, handcart and vendor are props).')


NAMES = CK.NAMES


class TextError(ValueError):
    pass


# ============================================================================================ validation
_SCHEMA = None


def scenario_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        with open(SCHEMA_PATH, encoding='utf-8') as fh:
            _SCHEMA = json.load(fh)
    return _SCHEMA


def problems(scn) -> list:
    """Why a scenario cannot be used, or []: the schema's errors (at their path), then core.scenarios.validate's and
    Falcon's own checks (a "$param" kind names a choice of known kinds; the ego's set speed within 0–130 km/h)."""
    import jsonschema
    if not isinstance(scn, dict):
        return ['the scenario is not a JSON object']
    v = jsonschema.Draft202012Validator(scenario_schema())
    out = []
    for e in sorted(v.iter_errors(scn), key=lambda e: list(e.absolute_path)):
        where = '/'.join(str(p) for p in e.absolute_path) or '(the top level)'
        msg = e.message if len(e.message) < 300 else e.message[:300] + '…'
        if e.validator in ('oneOf', 'anyOf') and e.context:
            msg += ' — ' + '; '.join(sorted({c.message[:120] for c in e.context}))[:400]
        out.append(f'at {where}: {msg}')
    if out:
        return out
    out += SC.validate(scn)
    for a in scn.get('actors') or ():
        k = a.get('kind')
        if isinstance(k, str) and k.startswith('$'):
            p = (scn.get('params') or {}).get(k[1:]) or {}
            bad = [c for c in p.get('choices') or [p.get('value')] if c not in SC.KINDS]
            if p.get('dist') not in ('choice', 'fixed') or bad:
                out.append(f'actor {a.get("id")}: kind {k} must be a choice (or fixed) of known kinds'
                           + (f', not {bad}' if bad else ''))
    if not out:
        for seed in (0, 1):
            conc = SC.sample(scn, seed)
            v = conc.get('ego', {}).get('speed_kmh')
            if isinstance(v, (int, float)) and not 0 <= v <= 130:
                out.append(f'the ego\'s speed {v:g} km/h (seed {seed}) is outside 0–130 km/h')
                break
    return out


def _sides(handedness: str) -> dict:
    lht = str(handedness).upper() == 'LHT'
    return {'over': 'right' if lht else 'left', 'kerb': 'left' if lht else 'right'}


def _start_side(a: dict, handedness: str) -> str | None:
    lane = (a.get('place') or {}).get('lane')
    s = _sides(handedness)
    return {'adjacent': s['over'], 'line': s['over'], 'opposite': s['over'], 'far_sidewalk': s['over'],
            'kerb': s['kerb'], 'sidewalk': s['kerb']}.get(lane)


def _kind_of(a: dict, scn: dict) -> list:
    k = a.get('kind')
    if isinstance(k, str) and k.startswith('$'):
        p = (scn.get('params') or {}).get(k[1:]) or {}
        return list(p.get('choices') or [p.get('value')])
    return [k]


def _actions(scn: dict, actor: str) -> list:
    return [x for ev in scn.get('timeline') or () for x in ev.get('actions') or () if x.get('actor') == actor]


def _behind(v) -> bool:
    return (isinstance(v, (int, float)) and v < 0) or (isinstance(v, str) and v.strip().startswith('-'))


def mismatches(scn: dict, claims: list, handedness: str = 'RHT') -> list:
    """What the description's claims say that the scenario as written does not (module doc). Read from the scenario
    itself, before any run; the run's check (core.scenario_check) is the real test."""
    from . import oddcoverage as OC
    from . import weather as W
    out = []
    acts = [a for a in scn.get('actors') or () if isinstance(a, dict)]
    for c in claims:
        t = c['type']
        if t == 'actor' and not any(c['kind'] in _kind_of(a, scn) for a in acts):
            out.append(f'the description has {c["phrase"]!r} ({c["kind"]}) and no actor is of kind {c["kind"]}')
        elif t == 'manoeuvre':
            who = [a for a in acts if c.get('kind') is None or c['kind'] in _kind_of(a, scn)]
            if not who:
                continue                           # said by the actor claim
            if c['what'] == 'cut_in':
                ok = [a for a in who if any(x.get('do') == 'lane_change' and x.get('to') == 'ego'
                                            for x in _actions(scn, a['id']))]
                if not ok:
                    out.append(f'{c["phrase"]!r}: no {c.get("kind") or "actor"} has a lane_change to "ego"')
                elif c.get('side') and not any(_start_side(a, handedness) == c['side'] for a in ok):
                    s = _sides(handedness)
                    out.append(f'{c["phrase"]!r} from the {c["side"]}: in {handedness} traffic the '
                               f'{"adjacent" if c["side"] == s["over"] else "kerb"} lane is on the {c["side"]}; '
                               f'the cutting actor starts on the other side')
            elif c['what'] == 'brake' and not any(x.get('do') == 'brake' for a in who for x in _actions(scn, a['id'])):
                out.append(f'{c["phrase"]!r}: no {c.get("kind") or "actor"} has a "brake" action')
            elif c['what'] == 'cross' and not any(a.get('heading') == 'across' for a in who):
                out.append(f'{c["phrase"]!r}: no {c.get("kind") or "actor"} has heading "across"')
            elif c['what'] == 'stopped' and not any(a.get('start') == 'stopped' and not any(
                    x.get('do') in ('go', 'speed', 'walk') for x in _actions(scn, a['id'])) for a in who):
                out.append(f'{c["phrase"]!r}: no {c.get("kind") or "actor"} starts "stopped" and stays so')
            elif c['what'] == 'wrong_way' and not any(a.get('heading') == 'against' for a in who):
                out.append(f'{c["phrase"]!r}: no {c.get("kind") or "actor"} has heading "against"')
            elif c['what'] == 'overtake' and not any(_behind((a.get('place') or {}).get('ahead_m')) for a in who):
                out.append(f'{c["phrase"]!r}: no {c.get("kind") or "actor"} starts behind the ego (negative ahead_m)')
        elif t == 'location':
            needs = set(scn.get('needs') or ())
            if c['feature'] == 'speed_breaker' and 'speed_breaker' not in needs:
                out.append(f'{c["phrase"]!r}: "needs" lacks "speed_breaker"')
            elif c['feature'] == 'junction' and 'junction_turn' not in needs:
                out.append(f'{c["phrase"]!r}: only a turn across traffic can ask for a junction ("junction_turn" in '
                           f'"needs", ego route "turn_across"); the run will say whether a junction was near')
        elif t == 'light':
            sun = W.resolve(scn.get('weather'))['sun_altitude_angle'] if scn.get('weather') else None
            if sun is None:
                out.append(f'{c["phrase"]!r}: no "weather" — the world\'s own light will be used; set '
                           f'weather.sun_altitude_angle for {NAMES.get(c["class"], c["class"])}')
            elif OC.light_class(sun) != c['class']:
                out.append(f'{c["phrase"]!r}: sun_altitude_angle {sun:g} is {OC.light_class(sun)}, not '
                           f'{NAMES.get(c["class"], c["class"])}')
        elif t == 'weather':
            if not scn.get('weather'):
                out.append(f'{c["phrase"]!r}: no "weather" — the world\'s own will be used')
                continue
            d = W.resolve(scn['weather'])
            cls = OC.weather_class(scn['weather'])
            ok = (max(d['wetness'], d['precipitation_deposits']) >= 40 if c['class'] == 'wet' else
                  d['cloudiness'] >= 50 if c['class'] == 'cloudy' else cls == c['class'])
            if not ok:
                out.append(f'{c["phrase"]!r}: the weather as written is {cls}')
        elif t == 'ego_speed':
            v = (scn.get('ego') or {}).get('speed_kmh')
            lo = hi = None
            if isinstance(v, (int, float)):
                lo = hi = float(v)
            elif isinstance(v, str) and v.startswith('$') and v[1:].isidentifier():
                p = (scn.get('params') or {}).get(v[1:]) or {}
                if p.get('dist') == 'fixed':
                    lo = hi = float(p['value'])
                elif p.get('dist') == 'uniform':
                    lo, hi = float(p['lo']), float(p['hi'])
            tol = max(5.0, 0.15 * c['kmh'])
            if lo is not None and (c['kmh'] < lo - tol or c['kmh'] > hi + tol):
                out.append(f'{c["phrase"]!r}: the ego\'s speed is {lo:g}–{hi:g} km/h')
    return out


# ============================================================================================ Falcon's own draft
def draft(claims: list, handedness: str = 'RHT', description: str = '') -> tuple:
    """(a scenario from the claims, notes) by the library mapping (module doc), or (None, why not)."""
    from . import oddcoverage as OC
    notes = []
    man = next((c for c in claims if c['type'] == 'manoeuvre'), None)
    if man is None:
        return None, ['no manoeuvre recognised (cut-in, braking, crossing, standing still, wrong way, overtaking): '
                      'Falcon cannot draft a scenario from the phrases alone']
    kind = man.get('kind') or next((c['kind'] for c in claims if c['type'] == 'actor'), None)
    if man['what'] in ('brake', 'overtake', 'wrong_way', 'cut_in') and kind in WALKERS:
        return None, [f'a {kind} does not {NAMES[man["what"]]} in Falcon\'s scenarios (pedestrians walk)']
    bases = BASES[man['what']]
    key = bases.get(kind, bases[None])
    scn = SC.builtin(key)
    notes.append(f'based on the library scenario {scn["name"]!r} ({key})')
    actor = scn['actors'][0]
    if man['what'] == 'stopped' and actor['kind'] == '$kind':
        scn['params'].pop('kind', None)
        actor['kind'] = kind or 'car'
    elif kind and actor['kind'] != kind:
        notes.append(f'its {actor["kind"]} made a {kind}')
        actor['kind'] = kind
    side, s = man.get('side'), _sides(handedness)
    if man['what'] == 'cut_in' and side and side != s['over']:
        actor['place'] = {'lane': 'kerb', 'ahead_m': actor['place'].get('ahead_m', 5.0)}
        scn['needs'] = [n for n in scn.get('needs') or () if n != 'adjacent_same']
        notes.append(f'from the {side}: it starts at the kerb (the {side} in {handedness} traffic) and moves into '
                     f'your lane')
    elif man['what'] == 'cross' and side and side != s['kerb']:
        actor['place'] = dict(actor['place'], lane='far_sidewalk')
        notes.append(f'from the {side}: it starts on the far sidewalk')
    elif man['what'] == 'overtake' and side and side != s['over']:
        actor['place'] = dict(actor['place'], lane='kerb')
        scn['needs'] = [n for n in scn.get('needs') or () if n != 'adjacent_same']
        notes.append(f'on the {side}: it passes along the kerb')
    light = next((c['class'] for c in claims if c['type'] == 'light'), None)
    wcls = next((c['class'] for c in claims if c['type'] == 'weather'), None)
    if light or wcls:
        w = copy.deepcopy(OC.CELL_WEATHER[wcls]) if wcls else dict(scn.get('weather') or {})
        w['sun_altitude_angle'] = OC.CELL_SUN[light] if light else w.get('sun_altitude_angle', OC.CELL_SUN['day'])
        scn['weather'] = w
        notes.append('weather: ' + ', '.join(x for x in (light and f'light {light} (sun {w["sun_altitude_angle"]:g}°)',
                                                         wcls and f'{wcls} (core.oddcoverage)') if x))
    for c in claims:
        if c['type'] == 'location' and c['feature'] == 'speed_breaker':
            scn['needs'] = sorted(set(scn.get('needs') or ()) | {'speed_breaker'})
            notes.append('needs a speed breaker 10–60 m ahead of the start (an imported world with road features)')
        elif c['type'] == 'location':
            notes.append(f'{NAMES[c["feature"]]}: no scenario need places it; the run\'s check measures the distance')
        elif c['type'] == 'ego_speed':
            v = scn['ego']['speed_kmh']
            if isinstance(v, str) and v.startswith('$') and v[1:] in scn.get('params', {}):
                scn['params'][v[1:]] = {'dist': 'fixed', 'value': float(c['kmh']), 'unit': 'km/h',
                                        'about': 'the ego\'s speed, from the description'}
            else:
                scn['ego']['speed_kmh'] = float(c['kmh'])
    scn.update(name=_name(description) or f'{scn["name"]} (from text)', builtin=False,
               about=(description or scn.get('about') or '').strip()[:800])
    scn.pop('id', None)
    return scn, notes


def _name(description: str) -> str:
    d = ' '.join(str(description or '').split())
    return (d[:1].upper() + d[1:])[:80].rstrip(' .,;') if d else ''


# ============================================================================================ talking to the model
def example_for(draft_scn: dict | None) -> dict:
    ex = copy.deepcopy(draft_scn) if draft_scn else SC.builtin('autorickshaw_cut_in')
    for k in ('builtin', 'schema', 'id', 'source'):
        ex.pop(k, None)
    return ex


def build_messages(description: str, handedness: str = 'RHT', draft_scn: dict | None = None) -> list:
    """The first request (module doc: text only)."""
    s = _sides(handedness)
    sys_ = ('You write driving scenarios for Falcon, a simulator on CARLA. Reply with ONE JSON object only — no prose, '
            'no code fence — that is a scenario valid against this JSON Schema:\n'
            + json.dumps(scenario_schema(), separators=(',', ':')) + '\n\n' + RULES.format(**s) +
            f'\nThe world drives on the {"left" if handedness == "LHT" else "right"} ({handedness}).')
    ex = ('Falcon\'s draft from the phrases it recognised in the description (improve it; keep what fits):\n'
          if draft_scn else 'An example scenario (another situation):\n')
    return [{'role': 'system', 'content': sys_},
            {'role': 'user', 'content': ex + json.dumps(example_for(draft_scn), separators=(',', ':')) +
             '\n\nWrite the scenario for this description:\n' + str(description)}]


def repair_message(errors: list, mism: list) -> dict:
    parts = []
    if errors:
        parts.append('Your scenario is not valid:\n' + '\n'.join(f'- {e}' for e in errors[:20]))
    if mism:
        parts.append('It does not say what the description says:\n' + '\n'.join(f'- {e}' for e in mism[:12]))
    parts.append('Reply with the corrected scenario: one JSON object only.')
    return {'role': 'user', 'content': '\n\n'.join(parts)}


def _strip(scn: dict) -> dict:
    out = {k: v for k, v in scn.items() if k not in ('builtin', 'source', 'values', 'seed')}
    return out


def finalise(scn: dict, description: str, claims: list, mode: str, rounds: int = 0, model: str = '') -> dict:
    """The scenario as the project keeps it: a stable id from the description and the scenario, the schema, its
    source (the description, the claims, how it was written)."""
    d = copy.deepcopy(scn)
    d.pop('id', None)
    h = hashlib.sha256((str(description) + json.dumps(d, sort_keys=True)).encode()).hexdigest()[:8]
    d.update(id=f'txt-{h}', schema=SC.SCENARIO_SCHEMA, builtin=False)
    d.setdefault('name', _name(description) or 'From a description')
    d.setdefault('about', str(description)[:800])
    d['source'] = {'format': 'text', 'description': str(description), 'mode': mode, 'rounds': int(rounds),
                   **({'model': model} if model else {}),
                   'claims': [{k: v for k, v in c.items() if k not in ('span',)} for c in claims]}
    return d


def write(description: str, complete=None, handedness: str = 'RHT', rounds: int = ROUNDS,
          model_claims: bool = True) -> dict:
    """The scenario for a description (module doc): with complete, the model's (repaired within rounds); without,
    Falcon's draft from the phrases. The record says how every attempt fared and what was sent."""
    description = ' '.join(str(description or '').split())
    if not description:
        raise TextError('describe the scenario in words')
    if len(description) > 2000:
        raise TextError('a description is at most 2,000 characters')
    rounds = max(0, min(int(rounds), MAX_ROUNDS))
    ex = CK.extract(description)
    claims = ex['claims']
    rec = {'schema': SCHEMA, 'description': description, 'handedness': handedness, 'recognised': ex['recognised'],
           'unrecognised': ex['unrecognised'], 'attempts': [], 'sent': {'requests': 0, 'characters': 0,
                                                                        'what': []}}
    model = getattr(complete, 'model', '') if complete else ''
    if complete is not None and model_claims and ex['unrecognised']:
        msgs = CK.claims_prompt(description)
        try:
            reply = complete(msgs)
            good, bad = CK.parse_model_claims(reply)
            claims = CK.merge(claims, good)
            rec['model_claims'] = {'accepted': len(good), 'rejected': bad}
        except ValueError as e:
            rec['model_claims'] = {'accepted': 0, 'error': str(e)}
        _sent(rec, msgs, 'the description and the claim schema')
    rec['claims'] = claims
    dr, notes = draft(claims, handedness, description)
    rec['draft_notes'] = notes
    if complete is None:
        rec['mode'] = 'phrases'
        if dr is None:
            rec.update(status='none', scenario=None, problems=notes, mismatches=[])
            return rec
        errs = problems(dr)
        mm = mismatches(dr, claims, handedness)
        rec['attempts'].append({'round': 0, 'errors': errs, 'mismatches': mm})
        rec.update(status='invalid' if errs else 'valid', problems=errs, mismatches=mm,
                   scenario=None if errs else finalise(dr, description, claims, 'phrases'))
        return rec
    rec['mode'] = 'model'
    rec['model'] = model
    msgs = build_messages(description, handedness, dr)
    _sent(rec, msgs, 'the description, the scenario schema and its rules, ' +
          ('Falcon\'s draft from the description\'s phrases' if dr else 'a library scenario as an example'))
    best = None
    for k in range(rounds + 1):
        reply = complete(msgs)
        errs, mm, scn = [], [], None
        try:
            scn = CK.parse_json_reply(reply)
        except ValueError as e:
            errs = [str(e)]
        if scn is not None:
            scn = _strip(scn)
            errs = problems(scn)
            if not errs:
                mm = mismatches(scn, claims, handedness)
                best = (scn, mm, k)
        rec['attempts'].append({'round': k, 'reply_characters': len(reply or ''), 'errors': errs, 'mismatches': mm})
        if not errs and not mm:
            break
        if k < rounds:
            msgs = msgs + [{'role': 'assistant', 'content': str(reply or '')[:12000]}, repair_message(errs, mm)]
            _sent(rec, msgs[-1:], 'the validation errors' + (' and mismatches' if mm else ''))
    if best is None:
        last = rec['attempts'][-1]
        rec.update(status='invalid', scenario=None, problems=last['errors'], mismatches=[])
        return rec
    scn, mm, k = best
    rec.update(status='valid' if not mm else 'valid_with_mismatches', problems=[], mismatches=mm,
               scenario=finalise(scn, description, claims, 'model', k, model))
    return rec


def _sent(rec, msgs, what):
    rec['sent']['requests'] += 1
    rec['sent']['characters'] += sum(len(m['content']) for m in msgs)
    if what not in rec['sent']['what']:
        rec['sent']['what'].append(what)


def plan_notes(scn: dict, seed: int = 0) -> list:
    """The schematic plan's view (core.scenario_plan) of the scenario's timeline: events it never fires."""
    from .scenario_plan import plan
    try:
        p = plan(SC.sample(scn, seed))
    except Exception as e:                       # noqa: BLE001 (the plan is a preview: said, not raised)
        return [f'the plan could not be made ({e})']
    fired = {i for i, _t in p['fired']}
    return [f'timeline event {i} does not fire in the plan (a plan, not a simulation)'
            for i in range(len(scn.get('timeline') or ())) if i not in fired]


def save_record(workspace: str, rec: dict) -> str:
    """The record into workspace/scenario-text/<scenario id or a hash>.json; its path."""
    from .provenance import write_json
    ident = (rec.get('scenario') or {}).get('id') or \
        'none-' + hashlib.sha256(rec['description'].encode()).hexdigest()[:8]
    p = os.path.join(workspace, FOLDER, f'{ident}.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    write_json(p, rec)
    return p


def world_handedness(project, world_id: str) -> str:
    w = project.item('worlds', world_id) if project is not None and world_id else None
    if not w or w.get('kind') == 'town':
        return 'RHT'
    return (w.get('osm') or {}).get('handedness', 'RHT')
