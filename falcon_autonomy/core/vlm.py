"""Vision-language driving explanations scored against the simulator's truth (numpy and the standard library only).

Falcon asks a vision-language model (VLM) grounded questions about a recorded frame — or a short clip ending at it —
and scores each free-text answer against the exact state the language data (core.language, falcon.language/1) was
computed from. Nothing here needs a model: questions() builds the questions and their truth from a language.jsonl
record, score_answer() scores one answer, aggregate() makes the per-type numbers with bootstrap intervals.

    questions(record, context) -> [question]       the question set of one frame, with its truth and target
    template_answer(question) -> str               core.language's own answer (perfect by construction)
    score_answer(question, answer, context) -> s   one answer's score (the rules below)
    aggregate(scored, n_boot, seed) -> scores      per question type, with intervals over drives

The truth is about what the model can see (TRUTH_MODES; core.vlm_eval's default 'visible'). An actor is VISIBLE in
a frame when its track id is among the frame's camera objects (objects/<camera>/<n>.json of the camera whose image
the model gets; a clip: its last frame) with at least core.harness.MIN_PIXELS visible pixels and at most
core.harness.MAX_OCCLUDED of its box hidden — the harness's rule for what a detector can be asked to find
(visible_ids()). context_of(..., visible=ids) then keeps only the visible actors as the frame's objects: counts count
the visible actors of the group within the radius ("How many vehicles can you see within 50 m?"); objects are asked
about only when visible, named by core.language's definite phrases recomputed among the visible actors (so "the
nearest car ahead" means the nearest visible one); the lead question ("Can you see a vehicle ahead of you in your
lane within 50 m?") and the lead's ttc and lane questions are SKIPPED when the lead exists but is not visible; brake
and brake_reason are SKIPPED when there is a reason to brake but none of the reasons' actors is visible (a recorded
red or yellow light always counts as seen), and otherwise their truth holds only the visible reasons. skipped() counts
the questions dropped per type. A mention that resolves to no visible actor but to a non-visible one is NOT_VISIBLE,
reported apart from a HALLUCINATION (a mention that resolves to no actor at all). Without visible ids (the
'all-actors' truth) every actor within the radius counts, as before the rule: the old wording and numbers exactly.

Question types (QTYPES) and how an answer is scored. The answer is free text; parsing rules are deterministic and
documented here, so a score can be checked by hand. Before a number, class or side is read, the spans of the
answer's object mentions (below) are blanked out, so "the car 12 m away is 14.4 m away" reads 14.4.
* count_vehicles, count_pedestrians, count_cyclists — "How many pedestrians can you see within 50 m?" (all-actors:
  "… are within 50 m of you?"): EXACT. The
  first integer or number word (zero … twenty; "no", "none" → 0; "a", "an", "one" → 1). Groups as
  core.worldmodel's BEV classes (a motorcyclist is a cyclist).
* lead — "Is there a vehicle ahead of you in your lane?": EXACT yes/no (the first word yes/no; otherwise a negation
  word before any affirmation → no; "there is", "I see" … → yes; else unparsed).
* brake — "Should you brake now?": EXACT yes/no, against core.language's heuristic brake rule.
* lane — "Which lane is the vehicle ahead of you in?": EXACT class: your lane, n lanes to the left / right (n counted;
  "the lane to your left" is 1), oncoming, crossing, off the roadway. Only asked when the drive has lanes.json.
* relative_position — "Is the car 12 m away on your left or on your right?": EXACT side (the last left/right word),
  or ahead/behind for an object within 1 m of the ego's axis.
* motion — "Is the car … getting closer or moving away?": EXACT class: closing, pulling away, standing still,
  keeping its distance — core.language's thresholds (still_mps, closing_mps) applied to the facts.
* distance (m, box to box), speed (m/s; km/h and mph are converted), ttc (s, the lead's time to collision) —
  TOLERANCE BANDS, stated in every scores.json: correct when |answer − truth| ≤ max(absolute, relative × truth):
  distance max(2 m, 15 %), speed max(1 m/s, 15 %), ttc max(0.5 s, 20 %). These bands are heuristic choices — wide
  enough for a monocular estimate to count, narrow enough that a guess does not — and can be changed (options).
  A truth of "no collision course within the horizon" is matched by an answer saying so (no collision, not on a
  collision course, never …) or by a number at or beyond the horizon. The absolute error is kept too.
* brake_reason — "Should you brake now? Say why.": FAITHFULNESS of the stated reason to the facts, by RULES (no LLM
  judge by default). The answer is cut into clauses (at . ; : ! ? and the words and, but, because, since, so). A
  clause is an ABSENCE claim when a negation word (no, not, nothing, none, nobody, no one, never, without, n't,
  clear) comes before its first hazard cue, else a PRESENCE claim. Kinds: "ttc" (a vehicle mention, or a collision
  cue: collision, collide, crash, too close, closing, rear-end, tailgating, cutting in, obstacle), "path" (a
  pedestrian or cyclist mention), "light" (a traffic light or signal, or a red / yellow / amber light; a green light
  is an absence claim), "other" (stop sign, speed bump, junction, turning, weather, animal … — things the facts
  cannot confirm: never supported). Actions alone (brake, slow down, stop) are decisions, not reasons. The truth
  reasons are core.language's brake rule recomputed from the facts (a TTC ahead below ttc_brake_s; a pedestrian or
  cyclist in or entering the path; a red or yellow light). A presence claim is SUPPORTED when a truth reason of its
  kind exists and, if it names an object, one of the objects the mention could denote (its candidates, below) is that
  reason's actor; an absence claim when no truth reason of its kinds exists (no kind named: none at all).
  precision = supported claims / claims; recall = truth reasons covered by a supported presence claim of their kind
  and actor (with no truth reason: 1 when a supported absence claim is made, else 0). FAITHFUL = some claim made,
  precision 1 and recall > 0. The decision the explanation implies (any presence claim → brake) is scored too.

Grounding: resolving a mention to a track id. A MENTION is a determiner (the, a, an, that, this, or a count) with an
optional ordinal (nearest, second-nearest …), up to two other words, and an object noun (car, truck, pedestrian,
person, cyclist, motorcyclist, … — NOUNS), followed by its qualifiers: the words after the noun as long as they are
qualifier words (_QUALIFIER: a distance "12 m", directions — Falcon's ahead, ahead on your left, on your right,
behind on your left, behind, or free left, right, in front, behind —, lanes "in your lane", "in the oncoming lane",
"track id N", "ahead of you"), ending at "N m away". Its candidates and its resolution, in order:
1. "track id N" names that actor (none present: unresolved).
2. Candidates: the frame's actors within the language data's radius whose group the noun fits (vehicle nouns fit
   vehicles — "vehicle" also fits cyclists —, person nouns pedestrians, bicycle nouns cyclists; other nouns such as
   cone or animal fit no actor); those whose core.language noun is the mention's own noun first, when there are any
   ("the police car" prefers police cars, "the car" plain cars).
3. A bare definite mention ("the car", no distance, direction, lane or ordinal) refers back to the object the question
   named when that object is a candidate (anaphora: the answer's "the car" is the question's car). Then "the
   <vehicle> ahead (of you | in your lane)" with no distance or ordinal is the lead vehicle when there is one
   (core.language's lead: the nearest vehicle or cyclist ahead in the ego's lane, even on a curve).
4. A stated direction keeps the candidates in its sector: Falcon's sectors exactly (core.language.direction_of),
   else widened by 15° each side, else for free words the half-plane (left y < 0, right y > 0, ahead x > 0, behind
   x < 0; CARLA's frame, +y right). A stated lane keeps the candidates in that lane relation. A filter that leaves
   none makes the mention resolve to nothing.
5. A stated distance d keeps candidates whose box-to-box gap or centre range is within the default distance band
   of d (max(2 m, 15 %), whatever band the scoring uses) and resolves to the nearest in |gap − d| (Falcon's
   distances are gaps); an ordinal resolves to the k-th nearest; otherwise the nearest candidate (flagged ambiguous
   when there were several).
A mention that resolves to nothing is a HALLUCINATION — unless its stated distance lies beyond the radius (beyond the
truth: not scored), or, in the visible truth, the same rules over every actor in range resolve it (NOT_VISIBLE: the
object exists but is not in the image's visible set; counted apart, not a hallucination). A count mention
("2 pedestrians") is a group claim: hallucinated only when no actor of the group is present (visible truth: none at
all; not_visible when only hidden ones are). GROUNDING of an answer to a question about an object (lead, lane,
distance, speed, motion, relative_position, ttc, and brake / brake_reason when the truth is to brake): its first
mention resolves to the target (or one of the reason actors). An answer naming no object is not scored for
grounding. Note that an answer repeating the question's own phrase is grounded by construction; grounding is
informative for lead, brake and brake_reason, where the answer has to name the object itself.

Negated mentions and the ego (scorer 2). A mention is NEGATED when a negation word (no, not, nothing, none, nobody,
no one, never, without, cannot, n't — NEGATION without "clear") stands before it in its clause (the faithfulness
splitter: . ; : ! ? and, but, because, since, so, as, while) with no comma in between: "No, I cannot see a vehicle
ahead of you in your lane" — not "No, the car ahead is far away", whose "No," is the answer word. A negated mention
is a CLAIM OF ABSENCE, not a reference: it is resolved as if present only to check the claim (absence_correct when it
would name no actor of the truth — in the visible truth a hidden actor is truly not seen; false_absence_rate = wrong
absence claims / absence claims), and it is never a hallucination, never grounded and outside the mention counts. A
count answer reads 0 for "not a" / "not … any". A mention of the EGO — the ego vehicle, the ego car, the ego, your /
my / our car — names the ego: never another actor, never a hallucination or not_visible, outside the mention counts.
SCORER (in scores.json and every answer row) is 2 since these two rules; folders without it were scored by 1.

Hallucination rate: hallucinated mentions / mentions (micro, over all answers), and the share of answers with one.
The not-visible rate: not_visible mentions / mentions (0 in the all-actors truth).

Intervals (aggregate): every number is a mean of per-answer scores with a 95 % percentile bootstrap interval
(core.metrics.bootstrap, Efron & Tibshirani 1993) over whole DRIVES (a cluster bootstrap: frames of one drive are not
independent). With fewer than MIN_DRIVES drives the unit becomes a BLOCK_S-second block of consecutive frames of a
drive, and the scores say the interval is optimistic (blocks of one drive are still correlated) — the same caution
as the model harness (core.harness), which flags drive intervals under five drives.

Limits (also in docs/vlm.md): visibility is the labels' pixel count and occlusion share, not whether a model could
recognise the object (a 25-pixel car 48 m away counts as visible); only actors are objects — CARLA's parked cars in
Town10 are map meshes, not actors, so a VLM that names one is counted as hallucinating; the brake rule and lane
relations are core.language's heuristics; the parsing rules are English-only and keyword-based, so a correct answer
phrased unusually can be scored wrong (unparsed answers are counted and shown); a class noun is checked at group level
only (a "truck" that is a car still grounds).
"""
from __future__ import annotations

import hashlib
import math
import random
import re

from . import language as L

SCHEMA_SCORES = 'falcon.vlm.scores/1'
SCORER = 2      # the scoring rules' version, in scores.json and every answer row: 1 until negated mentions became
                # absence claims and mentions of the ego the ego (folders without the field were scored by 1)
QTYPES = ('count_vehicles', 'count_pedestrians', 'count_cyclists', 'lead', 'brake', 'lane', 'relative_position',
          'motion', 'distance', 'speed', 'ttc', 'brake_reason')
KIND_OF = {'count_vehicles': 'count', 'count_pedestrians': 'count', 'count_cyclists': 'count', 'lead': 'yesno',
           'brake': 'yesno', 'lane': 'class', 'relative_position': 'class', 'motion': 'class', 'distance': 'number',
           'speed': 'number', 'ttc': 'number', 'brake_reason': 'explanation'}
GROUNDED_TYPES = ('lead', 'lane', 'relative_position', 'motion', 'distance', 'speed', 'ttc', 'brake', 'brake_reason')
TOLERANCE = {'distance': (2.0, 0.15), 'speed': (1.0, 0.15), 'ttc': (0.5, 0.20)}      # (absolute, relative)
SECTOR_MARGIN_DEG = 15.0
MIN_DRIVES = 5
BLOCK_S = 2.0
PROMPT_SUFFIX = ('Answer in one short sentence. Give distances in metres, speeds in m/s and times in seconds. "You" '
                 'are the ego vehicle whose camera took the image.')
TRUTH_MODES = ('visible', 'all-actors')
VISIBILITY_RULE = ('an actor is visible when its track id is in the frame\'s objects/<camera>/<n>.json (the camera '
                   'whose image the model gets; a clip: its last frame) with pixels >= {min_pixels} and occluded <= '
                   '{max_occluded} (core.harness\'s ignore rule)')
DEFAULTS = {'max_objects': 2, 'types': list(QTYPES), 'tolerance': {k: list(v) for k, v in TOLERANCE.items()}}

# ================================================================================================ nouns and words
NOUNS = {
    'vehicle': ('car', 'cars', 'truck', 'trucks', 'bus', 'buses', 'van', 'vans', 'suv', 'lorry', 'lorries', 'taxi',
                'taxis', 'ambulance', 'ambulances', 'jeep', 'pickup', 'trailer', 'sedan', 'hatchback', 'coupe',
                'police car', 'police cars', 'construction vehicle', 'firetruck', 'fire truck'),
    'any_vehicle': ('vehicle', 'vehicles'),
    'cyclist': ('cyclist', 'cyclists', 'bicycle', 'bicycles', 'bike', 'bikes', 'biker', 'bikers', 'motorcyclist',
                'motorcyclists', 'motorcycle', 'motorcycles', 'motorbike', 'motorbikes', 'scooter', 'scooters',
                'rider', 'riders'),
    'pedestrian': ('pedestrian', 'pedestrians', 'person', 'people', 'persons', 'man', 'men', 'woman', 'women',
                   'child', 'children', 'kid', 'kids', 'boy', 'girl', 'walker', 'walkers', 'jogger', 'police officer',
                   'construction worker'),
    'ego': ('ego vehicle', 'ego car', 'ego-vehicle', 'ego'),
    'other': ('traffic cone', 'traffic cones', 'cone', 'cones', 'barrier', 'barriers', 'animal', 'animals', 'dog',
              'dogs', 'cow', 'cows', 'cattle', 'deer', 'horse'),
}
NOUN_GROUP = {n: g for g, ns in NOUNS.items() for n in ns}
GROUP_FITS = {'vehicle': ('vehicle',), 'any_vehicle': ('vehicle', 'cyclist'), 'cyclist': ('cyclist',),
              'pedestrian': ('pedestrian',), 'other': (), 'ego': ()}
EGO_DETS = ('your', 'my', 'our')
PLURAL_GROUP = {'count_vehicles': 'vehicle', 'count_pedestrians': 'pedestrian', 'count_cyclists': 'cyclist'}
NUMBER_WORDS = {w: i for i, w in enumerate(('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                                            'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
                                            'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'))}
_ORD = {o: k for k, o in enumerate(L.ORDINALS)}
_ORD.update({'closest': 0, 'second-closest': 1, 'second closest': 1, 'second nearest': 1, 'first': 0})
_noun_alt = '|'.join(sorted((re.escape(n) for n in NOUN_GROUP), key=len, reverse=True))
_det = r'(?:the|a|an|that|this|one|your|my|our|\d+|' + '|'.join(NUMBER_WORDS) + ')'
_ord_alt = '|'.join(sorted((re.escape(o) for o in _ORD), key=len, reverse=True))
MENTION = re.compile(r'\b(?P<det>' + _det + r')\s+(?:(?P<ord>' + _ord_alt + r')\s+)?(?P<adj>(?:[a-z-]+\s+){0,2}?)'
                     r'(?P<noun>' + _noun_alt + r')\b(?P<tail>(?:[^.,;:!?()]|(?<=\d)[.,](?=\d))*)', re.I)
_QUALIFIER = set('m metre metres meter meters away ahead behind left right on your the to in of you me us my our front '
                 'lane lanes oncoming crossing a an with track id off roadway road street next far side directly just '
                 'slightly about around roughly approximately nearly over there same this current ego'.split()) | \
    set(NUMBER_WORDS)
_CONNECTORS = ('in', 'on', 'the', 'of', 'to', 'a', 'an', 'with', 'about', 'around', 'roughly', 'approximately',
               'nearly', 'over', 'there', 'just', 'slightly', 'directly')
_TOKEN = re.compile(r'\S+')
_UNIT_M = re.compile(r'^(?:m|metres?|meters?)$', re.I)
_DIST = re.compile(r'(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)\b', re.I)
_TRACK = re.compile(r'\b(?:track\s+id|id)\s*#?\s*(\d+)', re.I)
_NUM = re.compile(r'-?\d+(?:\.\d+)?')
FALCON_DIRS = ('ahead on your left', 'ahead on your right', 'behind on your left', 'behind on your right',
               'on your left', 'on your right', 'ahead', 'behind')
_SECTORS = {'ahead': (-15.0, 15.0), 'ahead on your right': (15.0, 60.0), 'on your right': (60.0, 120.0),
            'behind on your right': (120.0, 165.0), 'behind': (165.0, 195.0), 'behind on your left': (-165.0, -120.0),
            'on your left': (-120.0, -60.0), 'ahead on your left': (-60.0, -15.0)}
NEGATION = re.compile(r"\b(?:no|not|nothing|none|nobody|no one|never|without|cannot|clear)\b|n't\b", re.I)
MENTION_NEGATION = re.compile(r"\b(?:no|not|nothing|none|nobody|no one|never|without|cannot)\b|n't\b", re.I)
_COMMA = re.compile(r'(?<!\d),|,(?!\d)')


# ================================================================================================ the facts
def thresholds_of(manifest: dict | None) -> dict:
    """core.language's options that the truth depends on (its manifest's options, else its DEFAULTS)."""
    o = dict(L.DEFAULTS, **((manifest or {}).get('options') or {}))
    return {k: float(o[k]) for k in ('radius_m', 'ttc_brake_s', 'ttc_horizon_s', 'still_mps', 'closing_mps')}


def agents_of(record: dict, radius: float) -> list:
    """The record's actors (facts) within the radius, with their group, noun, direction and range."""
    out = []
    for a in (record.get('facts') or {}).get('agents') or ():
        rng = math.hypot(a['x'], a['y'])
        if rng > radius:
            continue
        out.append(dict(a, group=_group_of(a.get('category')), noun=L.noun_of(a.get('category')), range_m=rng,
                        direction=L.direction_of(a['x'], a['y']), bearing=math.degrees(math.atan2(a['y'], a['x']))))
    return sorted(out, key=lambda g: (g['distance_m'], str(g['id'])))


def _group_of(cat):
    from .worldmodel import bev_class_of
    return bev_class_of(cat)


def motion_class(a: dict, th: dict) -> str:
    """core.language's motion words for an actor (its _motion rule) as a class."""
    if a['speed_mps'] < th['still_mps'] and abs(a['range_rate_mps']) < th['closing_mps']:
        return 'still'
    if a['range_rate_mps'] <= -th['closing_mps']:
        return 'closing'
    if a['range_rate_mps'] >= th['closing_mps']:
        return 'away'
    return 'keeping'


def brake_reasons(record: dict, agents: list, th: dict) -> list:
    """core.language's brake rule recomputed from the facts: [{'kind': 'ttc'|'path'|'light', 'id', 'group',
    'state'}], in its order."""
    out = []
    for g in agents:
        if g['x'] > 0 and g.get('ttc_s') is not None and g['ttc_s'] < th['ttc_brake_s']:
            out.append({'kind': 'ttc', 'id': g['id'], 'group': g['group']})
    for g in agents:
        if g['group'] in ('pedestrian', 'cyclist') and (g.get('in_path') or g.get('entering_path')) and \
                not any(r['id'] == g['id'] for r in out):
            out.append({'kind': 'path', 'id': g['id'], 'group': g['group'],
                        'how': 'in' if g.get('in_path') else 'entering'})
    tl = (record.get('facts') or {}).get('traffic_light')
    if tl and str(tl.get('state', '')).lower() in ('red', 'yellow'):
        out.append({'kind': 'light', 'id': None, 'state': str(tl['state']).lower()})
    return out


def visible_ids(objects) -> set:
    """The track ids a camera frame shows (module doc): its objects (objects/<camera>/<n>.json "objects") with at least
    core.harness.MIN_PIXELS pixels and at most core.harness.MAX_OCCLUDED of the box hidden."""
    from .harness import MAX_OCCLUDED, MIN_PIXELS
    return {o['track_id'] for o in objects or () if o.get('track_id') is not None and
            (o.get('pixels') or 0) >= MIN_PIXELS and (o.get('occluded') or 0.0) <= MAX_OCCLUDED}


def context_of(record: dict, manifest: dict | None, drive: str = '', lanes: bool | None = None,
               visible=None) -> dict:
    """What scoring a record's answers needs: thresholds, the actors in range, the lead, the brake reasons. visible:
    the track ids the model's camera shows (visible_ids()) — the visible truth: 'agents' and 'by_id' are then the
    visible actors only, 'all_agents' and 'all_by_id' every actor in range; None: the all-actors truth."""
    th = thresholds_of(manifest)
    every = agents_of(record, th['radius_m'])
    if lanes is None:
        lanes = str((manifest or {}).get('lanes', '')).startswith('lanes.json')
    ag = every if visible is None else [a for a in every if a['id'] in set(visible)]
    ctx = {'drive': drive, 'step': record.get('step'), 'frame': record.get('frame'), 'sim_t': record.get('sim_t'),
           'th': th, 'agents': ag, 'by_id': {g['id']: g for g in ag},
           'lead_id': (record.get('facts') or {}).get('lead_id'), 'lanes': bool(lanes),
           'reasons': brake_reasons(record, every, th), 'truth': 'all-actors' if visible is None else 'visible'}
    if visible is not None:
        ctx.update(all_agents=every, all_by_id={g['id']: g for g in every})
    return ctx


def all_actors_view(ctx: dict) -> dict:
    """The all-actors truth of a context (itself when it is one)."""
    if ctx.get('all_agents') is None:
        return ctx
    out = {k: v for k, v in ctx.items() if k not in ('all_agents', 'all_by_id')}
    out.update(agents=ctx['all_agents'], by_id=ctx['all_by_id'], truth='all-actors')
    return out


# ================================================================================================ questions
def _qid(drive, step, k):
    return f'{drive}:{step}:{k}'


def questions(record: dict, ctx: dict, options: dict | None = None) -> list:
    """Falcon's grounded questions about one frame, each {'qid', 'type', 'text', 'truth', 'target', 'template'}.
    Objects asked about (max_objects) are the lead first, then the nearest described objects, named by core.language's
    own definite phrases (the record's qa). template is core.language's answer. In the visible truth (ctx from
    context_of(..., visible=…)) only visible actors are counted and asked about, their phrases recomputed among the
    visible actors, and the lead and brake questions are skipped when their object is not visible (module doc)."""
    o = dict(DEFAULTS, **(options or {}))
    want = set(o['types'])
    th, R = ctx['th'], f'{ctx["th"]["radius_m"]:g}'
    by_id, lead_id = ctx['by_id'], ctx['lead_id']
    vis = ctx.get('all_agents') is not None
    qa = record.get('qa') or []
    out = []

    def add(kind, text, truth, target=None, template=''):
        if kind in want:
            out.append({'qid': _qid(ctx['drive'], ctx['step'], len(out)), 'type': kind, 'text': text,
                        'truth': truth, 'target': target, 'template': template})

    counts = {g: [a['id'] for a in ctx['agents'] if a['group'] == g] for g in ('vehicle', 'pedestrian', 'cyclist')}
    for kind, (grp, one, many) in zip(('count_vehicles', 'count_pedestrians', 'count_cyclists'), L.GROUPS):
        ids = counts[grp]
        tmpl = f'{len(ids)} {one if len(ids) == 1 else many}.' if ids else f'None within {R} m.'
        text = f'How many {many} can you see within {R} m?' if vis else f'How many {many} are within {R} m of you?'
        add(kind, text, {'value': len(ids), 'ids': ids}, None, tmpl)
    lead = by_id.get(lead_id)
    lead_hidden = vis and lead is None and lead_id in ctx['all_by_id']        # there is one, not in the image
    lead_q = next((q for q in qa if q['type'] == 'lead'), None)
    if not lead_hidden:
        if vis:
            text = f'Can you see a vehicle ahead of you in your {"lane" if ctx["lanes"] else "path"} within {R} m?'
        else:
            text = lead_q['question']['text'] if lead_q else 'Is there a vehicle ahead of you in your lane?'
        add('lead', text, {'value': lead is not None}, lead_id if lead is not None else None,
            lead_q['answer']['text'] if lead_q else ('Yes.' if lead else 'No.'))
    if lead is not None:
        tq = next((q for q in qa if q['type'] == 'risk_ttc'), None)
        if tq is not None:
            add('ttc', 'What is the time to collision with the vehicle ahead of you?',
                {'value': lead.get('ttc_s'), 'horizon_s': th['ttc_horizon_s']}, lead_id, tq['answer']['text'])
        if ctx['lanes'] and lead.get('lane_rel'):
            add('lane', 'Which lane is the vehicle ahead of you in?',
                {'value': lead['lane_rel'], 'n': lead.get('lanes_over')}, lead_id,
                f'The vehicle ahead of you is {L.lane_phrase(lead["lane_rel"], lead.get("lanes_over"))}.')
    # objects: core.language's definite phrases (the distance question names each described object)
    seen = []
    for q in qa:
        if q['type'] == 'distance' and q['ids'] and q['ids'][0] in by_id and q['ids'][0] not in seen:
            seen.append(q['ids'][0])
    if lead_id in seen:                       # the lead first, then the nearest described objects
        seen = [lead_id] + [i for i in seen if i != lead_id]
    for aid in seen[:int(o['max_objects'])]:
        g = by_id[aid]
        mine = [q for q in qa if q['ids'] == [aid]]
        # the visible truth names the object among the visible actors (core.language's rules): by direction for the
        # distance, motion and speed questions, by distance for the side question
        dph = L._def_by_direction(g, ctx['agents']) if vis else None
        nph = L._def_by_distance(g, ctx['agents']) if vis else None
        for q in mine:
            t = q['type']
            qt, at = q['question']['text'], q['answer']['text']
            if vis and t in ('distance', 'relative_position', 'motion', 'speed'):
                ph = nph if t == 'relative_position' else dph
                qt, at = _with_phrase(q['question'], aid, ph), _with_phrase(q['answer'], aid, ph)
            if t == 'distance':
                add('distance', qt, {'value': g['distance_m']}, aid, at)
                phrase = dph if vis else q['question']['refs'][0]['phrase']
                if ctx['lanes'] and g.get('lane_rel') and aid != lead_id:
                    lp = L.lane_phrase(g['lane_rel'], g.get('lanes_over'))
                    add('lane', f'Which lane is {phrase} in?', {'value': g['lane_rel'], 'n': g.get('lanes_over')},
                        aid, f'{phrase[:1].upper()}{phrase[1:]} is {lp}.')
            elif t == 'relative_position':
                add('relative_position', qt, {'value': q['value']}, aid, at)
            elif t == 'motion':
                add('motion', qt, {'value': motion_class(g, th)}, aid, at)
            elif t == 'speed':
                add('speed', qt, {'value': g['speed_mps']}, aid, at)
    bq = next((q for q in qa if q['type'] == 'risk_brake'), None)
    reasons = ctx['reasons']
    brake_t = bq['answer']['text'] if bq else ('Yes.' if reasons else 'No.')
    caps = [c for c in record.get('captions') or () if c.get('kind') == 'brake']
    if vis:
        shown = [r for r in reasons if r['id'] is None or r['id'] in by_id]
        if reasons and not shown:             # a reason to brake, none of it in the image: not asked
            return out
        if shown and shown[0] is not reasons[0]:
            brake_t = _brake_answer(shown[0], ctx)
        reasons = shown
        caps = [c for c in caps if all(i in by_id for i in c.get('ids') or ())]
    targets = [r['id'] for r in reasons if r['id'] is not None]
    add('brake', bq['question']['text'] if bq else 'Should you brake now?', {'value': bool(reasons)},
        targets or None, brake_t)
    add('brake_reason', 'Should you brake now? Say why.', {'value': bool(reasons), 'reasons': reasons},
        targets or None, ' '.join(c['text'] for c in caps))
    return out


def _with_phrase(s: dict, aid, phrase: str) -> str:
    """A core.language sentence ({'text', 'refs'}) with each phrase referring to aid alone replaced by phrase (its
    capital kept at a sentence start)."""
    text, out, pos = s['text'], '', 0
    for r in sorted(s['refs'], key=lambda r: r['span'][0]):
        if r['ids'] != [aid]:
            continue
        a, b = r['span']
        out += text[pos:a] + (phrase[:1].upper() + phrase[1:] if text[a:a + 1].isupper() else phrase)
        pos = b
    return out + text[pos:]


def _brake_answer(r: dict, ctx: dict) -> str:
    """core.language's answer to "Should you brake now?" for one reason (its risk_brake templates)."""
    if r['kind'] == 'light':
        return f'Yes: the traffic light ahead is {r["state"]}.'
    g = ctx['by_id'][r['id']]
    if r['kind'] == 'ttc':
        return f'Yes: {L._indef(g)} is {g["ttc_s"]:.1f} s from a collision.'
    return f'Yes: {L._indef(g)} is {r["how"]} your path.'


def skipped(record: dict, ctx: dict, options: dict | None = None) -> dict:
    """{type: n} — how many fewer questions of each type the visible truth asks about this frame than the all-actors
    truth would, because their object or reason is not in the image (a hidden object's slot goes to the next visible
    described object when there is one, so this is the net drop); {} in the all-actors truth."""
    if ctx.get('all_agents') is None:
        return {}
    n = {}
    for q in questions(record, all_actors_view(ctx), options):
        n[q['type']] = n.get(q['type'], 0) + 1
    for q in questions(record, ctx, options):
        n[q['type']] = n.get(q['type'], 0) - 1
    return {t: k for t, k in n.items() if k > 0}


def template_answer(q: dict) -> str:
    return q['template']


# ================================================================================================ mentions
def mentions(text: str) -> list:
    """The object mentions of a text (module doc): [{'span', 'phrase', 'det', 'ord', 'noun', 'group', 'count',
    'distance', 'direction', 'dir_kind', 'lane', 'track', 'ahead_of_you', 'ego', 'negated'}]. ego: the mention names
    the ego (the ego vehicle, the ego, your / my / our car); negated: it stands in a negated clause (negated())."""
    out = []
    pos = 0
    while True:
        m = MENTION.search(text, pos)
        if m is None:
            break
        pos = m.end('noun')                   # resumed after the trimmed mention below (its raw tail may hold more)
        noun = m.group('noun').lower()
        adj = (m.group('adj') or '').strip().lower()
        if adj and any(w in NOUN_GROUP for w in adj.split()):       # "a car park"-like run-ons: take the first noun
            continue
        tail = _qualifiers(m.group('tail') or '')
        det = m.group('det').lower()
        count = None
        if det.isdigit():
            count = int(det)
        elif det in NUMBER_WORDS and det != 'one':
            count = NUMBER_WORDS[det]
        end = m.start('tail') + len(tail)
        pos = max(pos, end)
        phrase = text[m.start():end]
        low = (adj + ' ' + tail).lower()
        d = _DIST.search(tail)
        tr = _TRACK.search(tail)
        direction, kind = _direction(tail.lower())
        ego = NOUN_GROUP[noun] == 'ego' or det in EGO_DETS or 'ego' in adj.split()
        out.append({'span': [m.start(), end], 'phrase': phrase, 'det': det, 'ord': (m.group('ord') or '').lower()
                    or None, 'noun': noun, 'group': 'ego' if ego else NOUN_GROUP[noun], 'count': count,
                    'ego': ego, 'negated': not ego and negated(text, m.start()),
                    'distance': float(d.group(1)) if d else None, 'direction': direction, 'dir_kind': kind,
                    'lane': parse_lane(low) if 'lane' in low or 'oncoming' in low or 'off the road' in low else None,
                    'track': int(tr.group(1)) if tr else None,
                    'ahead_of_you': bool(re.search(r'\b(?:ahead of (?:you|me|us)|in front of (?:you|me|us)|ahead in '
                                                   r'(?:your|my) lane)\b', low)) or tail.strip().lower() == 'ahead'})
    return out


def negated(text: str, start: int) -> bool:
    """Whether the mention starting at start stands in a negated clause: a negation word (MENTION_NEGATION:
    NEGATION's words without "clear") before it in its clause (the faithfulness splitter, _CLAUSES), with no comma in
    between. "No, I cannot see a car ahead" and "There is not a pedestrian" negate the mention; "No, the car ahead is
    far away" (an answer word, then a comma) and "The lane is not free: a car is ahead" do not."""
    head = text[:start]
    cut = 0
    for c in _CLAUSES.finditer(head):
        cut = c.end()
    seg = head[cut:]
    commas = list(_COMMA.finditer(seg))
    if commas:
        seg = seg[commas[-1].end():]
    return bool(MENTION_NEGATION.search(seg))


def _qualifiers(tail: str) -> str:
    """The leading part of a mention's tail made of qualifier words only (_QUALIFIER: distances, directions, lanes,
    track ids), without dangling connectors; it ends at "N m away" (core.language's distance phrase)."""
    toks = list(_TOKEN.finditer(tail))
    end = 0
    for k, t in enumerate(toks):
        w = t.group(0).lower()
        num = re.fullmatch(r'#?\d+(?:[.,]\d+)?', w)
        if num:
            nxt = toks[k + 1].group(0).lower() if k + 1 < len(toks) else ''
            prev = toks[k - 1].group(0).lower() if k else ''
            if not (_UNIT_M.match(nxt) or prev in ('id', 'track')):
                break
        elif not re.fullmatch(r'\d+(?:\.\d+)?m', w) and w not in _QUALIFIER:
            break
        end = t.end()
        if w == 'away' and k and _UNIT_M.match(toks[k - 1].group(0)):
            break
    kept = tail[:end].split()
    while kept and kept[-1].lower() in _CONNECTORS:
        kept.pop()
    return tail[:_end_of(tail, kept)] if kept else ''


def _end_of(tail: str, kept: list) -> int:
    """The end offset in tail of the kept words (tail's own whitespace preserved)."""
    pos = 0
    for w in kept:
        pos = tail.index(w, pos) + len(w)
    return pos


def _direction(t: str):
    """(direction, kind) of a mention's qualifiers: Falcon's sector words ('falcon') or free words ('free')."""
    for d in FALCON_DIRS:
        if re.search(r'\b' + re.escape(d) + r'\b', t):
            return d, 'falcon'
    side = 'left' if re.search(r'\bleft\b', t) else 'right' if re.search(r'\bright\b', t) else None
    fb = 'ahead' if re.search(r'\b(?:ahead|in front|front)\b', t) else 'behind' if re.search(
        r'\b(?:behind|back|rear)\b', t) else None
    if side and fb:
        return f'{fb} {side}', 'free'
    if side or fb:
        return side or fb, 'free'
    return None, None


def _in_sector(a, direction, kind, margin=0.0) -> bool:
    if kind == 'falcon':
        lo, hi = _SECTORS[direction]
        b = a['bearing']
        if direction == 'behind':
            return abs(b) >= 165.0 - margin
        return lo - margin <= b <= hi + margin
    ok = True
    for w in direction.split():
        ok &= {'left': a['y'] < 0, 'right': a['y'] > 0, 'ahead': a['x'] > 0, 'behind': a['x'] < 0}[w]
    return ok


def tolerance(kind: str, truth: float, tol: dict | None = None) -> float:
    ab, rel = (tol or TOLERANCE)[kind]
    return max(float(ab), float(rel) * abs(float(truth)))


def candidates(m: dict, ctx: dict, prefer=()) -> dict:
    """{'ids': candidates after the filters, 'resolved': id or None, 'status': 'resolved' | 'hallucinated' |
    'not_visible' | 'out_of_range' | 'group' | 'absent' | 'ego', 'ambiguous', 'why'} of a mention (module doc's
    resolution rule). In the visible truth a mention that resolves to no visible actor but to a non-visible one is
    'not_visible' ('hidden': the actor or group it resolves to then). A mention of the ego is 'ego'. A negated
    mention is 'absent', a claim of absence: 'absence_correct' when, resolved as if present, it would name no actor
    of the truth (in the visible truth a hidden actor is truly not seen)."""
    if m.get('ego'):
        return {'ids': [], 'resolved': 'ego', 'status': 'ego', 'ambiguous': False, 'why': 'the ego vehicle'}
    if m.get('negated'):                      # a claim of absence: right when it denies nothing the truth holds
        r = candidates(dict(m, negated=False), ctx, prefer)
        seen = r['status'] == 'resolved' or (r['status'] == 'group' and m['count'] != 0 and bool(r['ids']))
        return dict(r, status='absent', absence_correct=not seen, denies=r['status'],
                    why=f'negated (a claim of absence): {r["why"]}')
    r = _resolve(m, ctx, prefer)
    if r['status'] == 'hallucinated' and ctx.get('all_agents') is not None:
        h = _resolve(m, all_actors_view(ctx), prefer)
        if h['status'] in ('resolved', 'group'):
            r = dict(r, status='not_visible', hidden=h['resolved'] if h['resolved'] is not None else h['ids'],
                     why=f'{r["why"]}; not visible: {h["why"]}')
    return r


def _resolve(m: dict, ctx: dict, prefer=()) -> dict:
    """candidates() over the context's actors."""
    agents, th = ctx['agents'], ctx['th']
    tol = None                                 # resolution always uses the default distance band, not the scoring's
    if m['track'] is not None:
        hit = m['track'] in ctx['by_id']
        return {'ids': [m['track']] if hit else [], 'resolved': m['track'] if hit else None,
                'status': 'resolved' if hit else 'hallucinated', 'ambiguous': False,
                'why': 'track id' if hit else f'no actor {m["track"]} in range'}
    fits = GROUP_FITS[m['group']]
    C = [a for a in agents if a['group'] in fits]
    if m['count'] is None:                                       # the exact noun first (core.language's nouns)
        C = [a for a in C if a['noun'] == m['noun']] or C
    if m['count'] is not None:                                   # a group claim: "2 pedestrians"
        ok = m['count'] == 0 or bool(C)
        return {'ids': [a['id'] for a in C], 'resolved': None, 'status': 'group' if ok else 'hallucinated',
                'ambiguous': False, 'why': 'count'}

    def none(why):
        far = m['distance'] is not None and m['distance'] - tolerance('distance', m['distance'], tol) > th['radius_m']
        return {'ids': [], 'resolved': None, 'status': 'out_of_range' if far else 'hallucinated', 'ambiguous': False,
                'why': why}
    if not C:
        return none(f'no {m["group"]} in range')
    bare = m['det'] in ('the', 'that', 'this') and not (m['distance'] is not None or m['direction'] or m['lane'] or
                                                       m['ord'])
    hit = [a['id'] for a in C if a['id'] in set(prefer or ())]
    if bare and hit:                          # "the car" in an answer: the car the question named
        return {'ids': [a['id'] for a in C], 'resolved': hit[0], 'status': 'resolved', 'ambiguous': False,
                'why': 'the question\'s object'}
    lead = ctx['by_id'].get(ctx['lead_id'])
    if m['ahead_of_you'] and m['group'] in ('vehicle', 'any_vehicle') and m['distance'] is None and not m['ord'] \
            and lead is not None and lead in C:
        return {'ids': [lead['id']], 'resolved': lead['id'], 'status': 'resolved', 'ambiguous': False,
                'why': 'the lead'}
    if m['direction']:
        D = [a for a in C if _in_sector(a, m['direction'], m['dir_kind'])]
        if not D and m['dir_kind'] == 'falcon':
            D = [a for a in C if _in_sector(a, m['direction'], m['dir_kind'], SECTOR_MARGIN_DEG)]
        if not D:
            return none(f'no {m["group"]} {m["direction"]}')
        C = D
    if m['lane'] and ctx['lanes']:
        rel, n = m['lane']
        D = [a for a in C if a.get('lane_rel') == rel and (rel not in ('left', 'right') or a.get('lanes_over') == n)]
        if not D:
            return none(f'no {m["group"]} {rel}')
        C = D
    ids = [a['id'] for a in C]
    if m['distance'] is not None:
        d = m['distance']
        t = tolerance('distance', d, tol)
        D = [a for a in C if min(abs(a['distance_m'] - d), abs(a['range_m'] - d)) <= t]
        if not D:
            return none(f'no {m["group"]} at {d:g} m')
        best = min(D, key=lambda a: (abs(a['distance_m'] - d), a['distance_m']))
        return {'ids': [a['id'] for a in D], 'resolved': best['id'], 'status': 'resolved',
                'ambiguous': len(D) > 1, 'why': 'distance'}
    if m['ord']:
        k = _ORD[m['ord']]
        exact = sorted(C, key=lambda a: (a['distance_m'], str(a['id'])))
        if k >= len(exact):
            return none(f'no {m["ord"]} {m["noun"]}')
        return {'ids': ids, 'resolved': exact[k]['id'], 'status': 'resolved', 'ambiguous': False, 'why': 'ordinal'}
    return {'ids': ids, 'resolved': C[0]['id'], 'status': 'resolved', 'ambiguous': len(C) > 1, 'why': 'nearest'}


def ground(text: str, ctx: dict, prefer=()) -> list:
    """Every mention of text with its resolution (candidates(); prefer: the question's object ids)."""
    return [dict(m, **candidates(m, ctx, prefer)) for m in mentions(text)]


def _blank(text: str, ms: list) -> str:
    """text with the mentions' spans replaced by spaces (so a phrase's own number or side is not read)."""
    t = list(text)
    for m in ms:
        a, b = m['span']
        t[a:b] = ' ' * (b - a)
    return ''.join(t)


# ================================================================================================ parsing answers
def parse_yesno(text: str):
    """True / False / None: the first word yes or no; else, in the first clause, a negation word → no, an
    affirmation (there is, I see, you should, brake, slow down, stop) → yes."""
    t = text.strip().lower()
    w = re.match(r"[a-z']+", t)
    w = w.group(0) if w else ''
    if w in ('yes', 'yeah', 'yep', 'correct', 'affirmative', 'definitely', 'absolutely'):
        return True
    if w in ('no', 'nope', 'not', 'negative', 'nothing', 'none'):
        return False
    first = _CLAUSES.split(t, maxsplit=1)[0]
    if NEGATION.search(first):
        return False
    if re.search(r'\b(?:there is|there are|there\'s|i see|i can see|you should|i should|should brake|must brake|need '
                 r'to brake|brake|braking|slow down|stop)\b', first):
        return True
    return None


def parse_count(text: str):
    """The first integer or number word (no / none → 0; a / an → 1). A negated existence claim counts 0: an "a" or
    "an" after a negation in its clause ("there is not a single cyclist"), or no number at all and a negation before
    "any" in the first clause ("I cannot see any cyclists")."""
    t = text.lower()
    m = re.search(r'\b(\d+|' + '|'.join(NUMBER_WORDS) + r'|no|none|a|an)\b', t)
    if not m:
        first = _CLAUSES.split(t, maxsplit=1)[0]
        neg = MENTION_NEGATION.search(first)
        return 0 if neg and re.search(r'\bany\b', first[neg.end():]) else None
    w = m.group(1)
    if w.isdigit():
        return int(w)
    if w in ('no', 'none'):
        return 0
    if w in ('a', 'an'):
        return 0 if negated(t, m.start()) else 1
    return NUMBER_WORDS[w]


_UNITS = {'distance': ((r'(?:m|metres?|meters?)\b(?!/)', 1.0), (r'(?:km|kilometres?|kilometers?)\b', 1000.0),
                       (r'(?:ft|feet|foot)\b', 0.3048)),
          'speed': ((r'(?:m/s|mps|metres? per second|meters? per second)', 1.0),
                    (r'(?:km/h|kmh|kph|km per hour|kilometres? per hour|kilometers? per hour)', 1 / 3.6),
                    (r'(?:mph|miles per hour)', 0.44704)),
          'ttc': ((r'(?:s|sec|secs|seconds?)\b', 1.0), (r'(?:ms|milliseconds?)\b', 0.001),
                  (r'(?:min|minutes?)\b', 60.0))}
NO_COLLISION = re.compile(r"\b(?:not on a collision course|no collision|not (?:going to )?collid\w*|won't collide|will "
                          r"not collide|never|no risk|infinite|infinity|none|n/a|no time to collision|not approaching|"
                          r"moving away|pulling away)\b", re.I)
STILL = re.compile(r'\b(?:standing still|stationary|stopped|not moving|parked|at rest|zero)\b', re.I)


def parse_number(text: str, kind: str):
    """(value in the type's unit, how) or (None, why): the first number with a unit of its kind (converted), else
    the first bare number. ttc: (inf, 'no collision') for an answer saying there is none; speed: 0 for still words."""
    t = text.lower()
    if kind == 'ttc' and NO_COLLISION.search(t):
        return math.inf, 'no collision'
    for m in _NUM.finditer(t):
        rest = t[m.end():].lstrip()
        for pat, k in _UNITS[kind]:
            if re.match(pat, rest):
                return float(m.group(0)) * k, 'unit'
    if kind == 'speed' and STILL.search(t):
        return 0.0, 'still'
    m = _NUM.search(t)
    if m:
        return float(m.group(0)), 'bare'
    return None, 'no number'


def parse_lane(text: str):
    """(relation, lanes over) of a lane answer, or None."""
    t = text.lower()
    m = re.search(r'\b(\d+|' + '|'.join(NUMBER_WORDS) + r')\s+lanes?\s+(?:to|on|over to)\s+(?:your|the|my)\s+'
                  r'(left|right)\b', t)
    if m:
        w = m.group(1)
        return m.group(2), int(w) if w.isdigit() else NUMBER_WORDS[w]
    if re.search(r'\boncoming\b|\bopposite (?:lane|direction)\b|\bother side of the road\b|\bon-coming\b', t):
        return 'oncoming', None
    if re.search(r'\bcrossing\b|\bcross(?:-| )street\b|\bperpendicular\b|\bintersecting\b', t):
        return 'crossing', None
    if re.search(r'\boff the road(?:way)?\b|\bsidewalk\b|\bpavement\b|\bfootpath\b|\bverge\b|\bkerb\b|\bcurb\b|'
                 r'\bshoulder\b|\bnot (?:in|on) (?:a|any) lane\b', t):
        return 'off_road', None
    m = re.search(r'\b(?:the )?(?:lane|one) (?:to|on) (?:your|the|my) (left|right)\b|\b(left|right)(?:-hand)? lane\b|'
                  r'\bnext lane (?:to|on) (?:your|the|my) (left|right)\b|\badjacent lane (?:to|on) (?:your|the|my) '
                  r'(left|right)\b', t)
    if m:
        return next(g for g in m.groups() if g), 1
    if re.search(r'\b(?:your|my|our|the same|same|ego|this|current|the ego)\s+lane\b', t):
        return 'ego', 0
    return None


def parse_side(text: str):
    """The last direction word of an answer: 'left' | 'right' | 'ahead' | 'behind', or None."""
    t = text.lower()
    found = [(m.start(), {'front': 'ahead', 'in front': 'ahead', 'back': 'behind', 'rear': 'behind'}.get(
        m.group(0), m.group(0))) for m in re.finditer(r'\b(?:left|right|ahead|in front|front|behind|back|rear)\b', t)]
    return found[-1][1] if found else None


def _side_words(value) -> set:
    """The words a relative_position truth accepts: its side, or the words of its direction ('ahead on your right'
    accepts ahead and right)."""
    v = str(value)
    return {v} if v in ('left', 'right') else set(v.replace('on your', '').split())


def parse_motion(text: str):
    t = text.lower()
    for cls, pat in (('still', r'\b(?:standing still|stationary|stopped|not moving|parked|at rest|motionless)\b'),
                     ('away', r'\b(?:moving away|pulling away|going away|receding|getting farther|getting further|'
                              r'increasing (?:its )?distance|away from (?:you|me|us))\b'),
                     ('closing', r'\b(?:closing|getting closer|approaching|coming closer|moving towards|moving toward|'
                                 r'coming towards|coming toward|nearer|decreasing (?:its )?distance)\b'),
                     ('keeping', r'\b(?:keeping (?:its|the same|a constant) distance|same distance|constant '
                                 r'distance|steady distance|neither)\b')):
        if re.search(pat, t):
            return cls
    return None


# ================================================================================================ explanations
_TTC_CUE = re.compile(r'\b(?:collision|collide|colliding|crash|hit|too close|closing|cut(?:s|ting)? in|rear-end|'
                      r'tailgat\w*|time to collision|ttc|obstacle)\b', re.I)
_COLOUR = re.compile(r'\b(red|yellow|amber|green)\b', re.I)
_LIGHT = re.compile(r'\b(?:traffic )?(?:light|signal)s?\b(?! (?:traffic|rain|snow|fog))|'
                    r'\b(?:red|amber|yellow) (?:is|was) (?:on|showing)\b', re.I)
_OTHER = re.compile(r'\b(?:stop sign|give way|yield sign|speed (?:bump|breaker|hump|limit)|junction|intersection|'
                    r'roundabout|turn|turning|curve|bend|weather|rain|fog|wet|slippery|school|animal|dog|cow|cattle|'
                    r'pothole|construction|roadworks?|road works|cone|barrier|crosswalk|zebra crossing)\b', re.I)
_CLAUSES = re.compile(r'(?<!\d)\.(?!\d)|[;:!?]+|\b(?:and|but|because|since|so|as|while|whereas)\b', re.I)


def claims(text: str, ctx: dict) -> list:
    """The stated reasons of an explanation (module doc): [{'polarity': 'present'|'absent', 'kinds', 'mentions',
    'clause'}]."""
    out = []
    pos = 0
    pieces = []
    for m in _CLAUSES.finditer(text):
        pieces.append((pos, text[pos:m.start()]))
        pos = m.end()
    pieces.append((pos, text[pos:]))
    for start, clause in pieces:
        if not clause.strip():
            continue
        ms = ground(clause, ctx)
        cues = []                         # (position, kind)
        for m in ms:
            if m['group'] == 'ego':
                continue
            k = {'vehicle': 'ttc', 'any_vehicle': 'ttc', 'pedestrian': 'path', 'cyclist': 'path',
                 'other': 'other'}[m['group']]
            cues.append((m['span'][0], k))
        blank = _blank(clause, ms)
        for rx, k in ((_TTC_CUE, 'ttc'), (_LIGHT, 'light'), (_OTHER, 'other')):
            for m in rx.finditer(blank):
                cues.append((m.start(), k))
        neg = NEGATION.search(blank)
        if not cues and not neg:
            continue
        first = min(c[0] for c in cues) if cues else len(clause)
        polarity = 'absent' if neg and neg.start() < first else 'present'
        kinds = sorted({k for _, k in cues})
        colour = None
        if 'light' in kinds:
            cm = _COLOUR.search(blank)
            colour = {'amber': 'yellow'}.get(cm.group(1).lower(), cm.group(1).lower()) if cm else None
            if colour == 'green' and polarity == 'present':      # a green light is a reason not to stop
                polarity, kinds = 'absent', ['light']
        if polarity == 'absent':
            kinds = [k for k in kinds if k != 'other'] or ['ttc', 'path', 'light']
        out.append({'polarity': polarity, 'kinds': kinds, 'mentions': ms, 'clause': clause.strip(),
                    'colour': colour})
    return out


def score_explanation(answer: str, truth: dict, ctx: dict) -> dict:
    reasons = truth['reasons']
    cs = claims(answer, ctx)
    supported, covered = 0, set()
    detail = []
    n_claims = 0
    for c in cs:
        if c['polarity'] == 'absent':
            ok = not any(r['kind'] in c['kinds'] for r in reasons)
            n_claims += 1
            supported += ok
            detail.append({'claim': f'no {"/".join(c["kinds"])}', 'supported': ok, 'clause': c['clause']})
            continue
        for k in c['kinds']:
            n_claims += 1
            ok = False
            if k == 'light':
                hits = [i for i, r in enumerate(reasons) if r['kind'] == 'light' and
                        (c['colour'] is None or c['colour'] == r['state'])]
            elif k in ('ttc', 'path'):
                groups = {'ttc': ('vehicle', 'any_vehicle'), 'path': ('pedestrian', 'cyclist')}[k]
                ms = [m for m in c['mentions'] if m['group'] in groups and m['count'] is None and
                      m['status'] != 'absent']
                hits = []
                for i, r in enumerate(reasons):
                    if r['kind'] != k:
                        continue
                    if not ms or any(r['id'] in m['ids'] for m in ms):
                        hits.append(i)
                if ms and all(m['status'] in ('hallucinated', 'not_visible') for m in ms):
                    hits = []
            else:
                hits = []
            ok = bool(hits)
            covered.update(hits)
            supported += ok
            detail.append({'claim': k, 'supported': ok, 'clause': c['clause']})
    if reasons:
        recall = len(covered) / len(reasons)
    else:
        recall = 1.0 if any(d['supported'] and d['claim'].startswith('no ') for d in detail) else 0.0
    precision = supported / n_claims if n_claims else None
    decision = None
    if cs:
        decision = any(c['polarity'] == 'present' for c in cs)
    faithful = bool(n_claims and precision == 1.0 and recall > 0)
    return {'correct': faithful, 'faithful': faithful, 'precision': precision, 'recall': recall,
            'decision': decision, 'decision_correct': None if decision is None else decision == truth['value'],
            'claims': detail, 'parsed': bool(n_claims)}


# ================================================================================================ scoring
def score_answer(q: dict, answer: str, ctx: dict) -> dict:
    """One answer's score: {'correct': bool, 'parsed', 'pred', ...type details, 'mentions', 'hallucinated',
    'grounded' (None when not scored)}."""
    answer = answer if isinstance(answer, str) else ''
    t, truth = q['type'], q['truth']
    target = q.get('target')
    targets = set(target if isinstance(target, list) else [target]) - {None}
    ms = ground(answer, ctx, targets)
    rest = _blank(answer, ms)
    out = {'type': t, 'kind': KIND_OF[t]}
    if KIND_OF[t] == 'count':
        pred = parse_count(answer)
        out.update(pred=pred, correct=pred == truth['value'], parsed=pred is not None)
    elif t in ('lead', 'brake'):
        pred = parse_yesno(answer)
        out.update(pred=pred, correct=pred is not None and pred == truth['value'], parsed=pred is not None)
    elif t == 'lane':
        pred = parse_lane(rest)
        ok = pred is not None and pred[0] == truth['value'] and (
            truth['value'] not in ('left', 'right') or pred[1] == truth['n'])
        out.update(pred=list(pred) if pred else None, correct=bool(ok), parsed=pred is not None)
    elif t == 'relative_position':
        pred = parse_side(rest)
        out.update(pred=pred, correct=pred in _side_words(truth['value']), parsed=pred is not None)
    elif t == 'motion':
        pred = parse_motion(rest)
        out.update(pred=pred, correct=pred == truth['value'], parsed=pred is not None)
    elif KIND_OF[t] == 'number':
        pred, how = parse_number(rest, t)
        tv = truth['value']
        tol = ctx.get('tolerance')
        if t == 'ttc' and tv is None:                       # no collision course within the horizon
            ok = pred is not None and pred >= truth['horizon_s']
            err = None
        elif pred is None:
            ok, err = False, None
        elif not math.isfinite(pred):
            ok, err = False, None
        else:
            err = abs(pred - tv)
            ok = err <= tolerance(t, tv, tol)
        out.update(pred=None if pred is None else (pred if math.isfinite(pred) else 'no collision'), how=how,
                   correct=bool(ok), parsed=pred is not None, abs_error=err,
                   band=None if tv is None else tolerance(t, tv, tol))
    else:
        out.update(score_explanation(answer, truth, ctx))
        out['pred'] = out.get('decision')
    out['mentions'] = [dict({'phrase': m['phrase'], 'status': m['status'], 'resolved': m['resolved'],
                             'candidates': m['ids'], 'why': m['why']},
                            **({'absence_correct': m['absence_correct']} if m['status'] == 'absent' else {}))
                       for m in ms]
    out['n_mentions'] = sum(m['status'] in ('resolved', 'hallucinated', 'group', 'not_visible') for m in ms)
    out['hallucinated'] = sum(m['status'] == 'hallucinated' for m in ms)
    out['not_visible'] = sum(m['status'] == 'not_visible' for m in ms)
    out['absent'] = sum(m['status'] == 'absent' for m in ms)
    out['absent_wrong'] = sum(m['status'] == 'absent' and not m['absence_correct'] for m in ms)
    out['ego'] = sum(m['status'] == 'ego' for m in ms)
    out['scorer'] = SCORER
    single = [m for m in ms if m['count'] is None and m['status'] in ('resolved', 'hallucinated', 'not_visible')]
    if t in GROUNDED_TYPES and targets and single:
        out['grounded'] = single[0]['resolved'] in targets
    else:
        out['grounded'] = None
    return out


# ================================================================================================ noisy answers
def _rng(*parts) -> random.Random:
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(h[:8], 'big'))


def absent_object_phrase(ctx: dict) -> str:
    """A definite phrase naming an object that is not there: a group with no actor in range (visible or not), else a
    traffic cone (actors are only vehicles and walkers)."""
    present = {a['group'] for a in all_actors_view(ctx)['agents']}
    for grp, noun in (('cyclist', 'cyclist'), ('pedestrian', 'pedestrian'), ('vehicle', 'truck')):
        if grp not in present:
            return f'the {noun} on your left'
    return 'the traffic cone ahead'


def corrupt(q: dict, ctx: dict, rng: random.Random) -> str:
    """A wrong answer to q, wrong by construction under the scoring rules (the noisy baseline)."""
    t, tv = q['type'], q['truth'].get('value')
    if KIND_OF[t] == 'count':
        n = tv + 1 if tv == 0 or rng.random() < 0.5 else tv - 1
        return f'{n}.'
    if t == 'lead':
        return 'No, there is not.' if tv else f'Yes: {absent_object_phrase(ctx)}.'
    if t == 'brake':
        return 'No, nothing calls for braking.' if tv else 'Yes, you should brake now.'
    if t == 'lane':
        return 'It is in the lane to your left.' if tv == 'ego' else 'It is in your lane.'
    if t == 'relative_position':
        sv = str(tv)
        if sv in ('left', 'right'):
            return f'It is on your {"right" if sv == "left" else "left"}.'
        return 'It is behind.' if 'ahead' in _side_words(sv) else 'It is ahead.'
    if t == 'motion':
        return {'closing': 'It is pulling away.', 'away': 'It is getting closer.', 'still': 'It is getting closer.',
                'keeping': 'It is pulling away.'}[tv]
    if t in ('distance', 'speed', 'ttc'):
        unit = {'distance': 'm', 'speed': 'm/s', 'ttc': 's'}[t]
        if tv is None:                                       # no collision course: a short TTC is wrong
            return f'About {0.5 * q["truth"]["horizon_s"]:.1f} {unit}.'
        band = tolerance(t, tv, ctx.get('tolerance'))
        v = tv + (3.0 * band if rng.random() < 0.5 or tv - 3.0 * band < 0 else -3.0 * band)
        return f'About {v:.1f} {unit}.'
    if t == 'brake_reason':
        if q['truth']['reasons']:
            return 'No need to brake: the road ahead is clear.'
        return 'Brake: the traffic light ahead is red.'
    raise ValueError(t)


def hallucinate(answer: str, ctx: dict) -> str:
    """The answer with its first object phrase replaced by one naming an absent object — or, when it names none or
    its first is a count ("1 pedestrian", whose number is the answer), with a sentence naming one added. In an
    explanation the added object is a claim the facts do not support, so a faithful explanation becomes unfaithful."""
    ms = [m for m in mentions(answer) if not (m['ego'] or m['negated'])]
    ph = absent_object_phrase(ctx)
    if ms and ms[0]['count'] is None:
        a, b = ms[0]['span']
        cap = answer[a:a + 1].isupper()
        return answer[:a] + (ph[:1].upper() + ph[1:] if cap else ph) + answer[b:]
    return answer.rstrip() + f' I also see {ph}.'


# ================================================================================================ aggregation
def unit_of(row: dict, mode: str) -> str:
    if mode == 'drive':
        return str(row['drive'])
    return f'{row["drive"]}#{int(math.floor(float(row.get("sim_t") or 0.0) / BLOCK_S))}'


def _metrics(rows: list) -> dict:
    """The per-type means of a set of scored answers (the bootstrap's statistic)."""
    out = {}
    by = {}
    for r in rows:
        by.setdefault(r['type'], []).append(r)
    allc = []
    for t, rs in by.items():
        c = [float(r['correct']) for r in rs]
        out[f'{t}.accuracy'] = sum(c) / len(c)
        allc.append(out[f'{t}.accuracy'])
        g = [float(r['grounded']) for r in rs if r.get('grounded') is not None]
        if g:
            out[f'{t}.grounding'] = sum(g) / len(g)
        if KIND_OF[t] == 'number':
            e = sorted(r['abs_error'] for r in rs if r.get('abs_error') is not None)
            if e:
                out[f'{t}.median_abs_error'] = e[len(e) // 2] if len(e) % 2 else 0.5 * (e[len(e) // 2 - 1] +
                                                                                          e[len(e) // 2])
        if t == 'brake_reason':
            p = [r['precision'] for r in rs if r.get('precision') is not None]
            out['brake_reason.precision'] = sum(p) / len(p) if p else None
            out['brake_reason.recall'] = sum(r['recall'] for r in rs) / len(rs)
            d = [float(r['decision_correct']) for r in rs if r.get('decision_correct') is not None]
            out['brake_reason.decision'] = sum(d) / len(d) if d else None
        out[f'{t}.parsed'] = sum(float(r['parsed']) for r in rs) / len(rs)
    if allc:
        out['macro_accuracy'] = sum(allc) / len(allc)
    nm = sum(r['n_mentions'] for r in rows)
    out['hallucination_rate'] = sum(r['hallucinated'] for r in rows) / nm if nm else 0.0
    out['answers_with_hallucination'] = sum(r['hallucinated'] > 0 for r in rows) / len(rows) if rows else None
    out['not_visible_rate'] = sum(r.get('not_visible', 0) for r in rows) / nm if nm else 0.0
    na = sum(r.get('absent', 0) for r in rows)
    out['false_absence_rate'] = sum(r.get('absent_wrong', 0) for r in rows) / na if na else None
    g = [float(r['grounded']) for r in rows if r.get('grounded') is not None]
    out['grounding'] = sum(g) / len(g) if g else None
    return out


def aggregate(rows: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """Per-type accuracy (and grounding, errors, faithfulness parts), the hallucination rate and the macro accuracy,
    each with a 95 % percentile bootstrap interval over drives (or blocks, module doc). rows: score_answer() results
    with 'drive' and 'sim_t'."""
    from .metrics import bootstrap
    if not rows:
        raise ValueError('nothing to score')
    drives = sorted({r['drive'] for r in rows})
    mode = 'drive' if len(drives) >= MIN_DRIVES else 'block'
    items = [dict(r, _unit=unit_of(r, mode)) for r in rows]
    b = bootstrap(_metrics, items, n=n_boot, seed=seed, unit='drive', unit_key='_unit')
    counts = {}
    for r in rows:
        c = counts.setdefault(r['type'], {'n': 0, 'unparsed': 0})
        c['n'] += 1
        c['unparsed'] += not r['parsed']
    mentions_n = {'mentions': sum(r['n_mentions'] for r in rows), 'hallucinated': sum(r['hallucinated'] for r in rows),
                  'not_visible': sum(r.get('not_visible', 0) for r in rows),
                  'absence_claims': sum(r.get('absent', 0) for r in rows),
                  'false_absence': sum(r.get('absent_wrong', 0) for r in rows),
                  'ego': sum(r.get('ego', 0) for r in rows)}
    note = None
    if mode == 'block':
        note = (f'{len(drives)} drive(s), fewer than {MIN_DRIVES}: the intervals are over {BLOCK_S:g} s blocks of '
                'consecutive frames, which are correlated within a drive — they are too narrow; record more drives '
                'for intervals over drives')
    return {'metrics': b['metrics'], 'counts': counts, 'mentions': mentions_n,
            'unit': 'drive' if mode == 'drive' else f'{BLOCK_S:g} s block',
            'n_units': b['n_units'], 'n_answers': len(rows), 'drives': drives, 'n_boot': n_boot, 'seed': seed,
            'ci': 0.95, 'reliable': mode == 'drive', 'note': note}


def truth_doc(mode: str) -> dict:
    """What the truth of an evaluation is about (scores.json 'truth')."""
    from .harness import MAX_OCCLUDED, MIN_PIXELS
    if mode == 'all-actors':
        return {'mode': mode, 'rule': 'every actor within the radius (behind the ego and hidden ones too), whether or '
                                      'not the camera shows it'}
    return {'mode': mode, 'rule': VISIBILITY_RULE.format(min_pixels=MIN_PIXELS, max_occluded=MAX_OCCLUDED),
            'min_pixels': MIN_PIXELS, 'max_occluded': MAX_OCCLUDED,
            'applied': 'counts count visible actors; objects asked about only when visible (phrases among the visible '
                       'actors); lead, ttc and lead lane skipped when the lead is not visible; brake and brake_reason '
                       'skipped when no reason\'s actor is visible, else scored on the visible reasons; a mention of a '
                       'non-visible actor is not_visible, not a hallucination'}


def method(options: dict | None = None) -> dict:
    """The scoring rules as numbers and words, for scores.json (options' 'truth': the truth mode)."""
    tol = dict(TOLERANCE, **{k: tuple(v) for k, v in ((options or {}).get('tolerance') or {}).items()})
    truth = (options or {}).get('truth') or 'all-actors'
    return {'schema': SCHEMA_SCORES, 'scorer': SCORER, 'truth': truth_doc(truth),
            'question_types': {t: KIND_OF[t] for t in QTYPES},
            'tolerance': {k: {'absolute': a, 'relative': r, 'rule': f'|answer - truth| <= max({a:g}, {r:g} x truth)'}
                          for k, (a, r) in tol.items()},
            'grounding': 'first object mention resolved to the target track id (module doc of core.vlm)',
            'hallucination': 'mentions resolving to no actor in range / mentions',
            'not_visible': 'mentions resolving to no visible actor but to a non-visible one / mentions (visible truth)',
            'negated': 'a mention after a negation word in its clause (no comma between) is a claim of absence, not a '
                       'reference: never a hallucination, outside the mention counts; false_absence_rate = absence '
                       'claims denying an actor of the truth / absence claims',
            'ego': 'the ego vehicle, the ego, your / my / our car name the ego: never resolved to another actor, '
                   'outside the mention counts',
            'explanation': 'rule-based faithfulness: every claim supported by the facts and at least one true reason '
                           'stated (no LLM judge unless one was configured and approved)',
            'sector_margin_deg': SECTOR_MARGIN_DEG, 'interval': f'95 % percentile bootstrap over drives; blocks of '
                                                                f'{BLOCK_S:g} s with fewer than {MIN_DRIVES} drives',
            'prompt_suffix': PROMPT_SUFFIX, 'heuristic': ['brake rule (core.language)', 'lane relations '
                                                          '(core.language)', 'tolerance bands', 'parsing keywords']}
