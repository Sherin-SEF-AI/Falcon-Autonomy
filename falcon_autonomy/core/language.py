"""Language labels derived from a recorded run: template captions and question–answer pairs computed from the exact
simulation state, every phrase grounded in the objects it names (schema falcon.language/1). numpy only.

    derive_language(run_dir, out_dir, options) -> manifest      OUT/language.jsonl + OUT/language.json
    build_state(...) -> state;  describe(state, seed, options) -> {'captions', 'qa'}
    paraphrase(text, refs, client) -> {'text', 'refs', ...}     (optional; only through a client passed in)

State (per recorded step; build_state): the ego's pose, speed and box (actors.jsonl, role 'hero'; frames.jsonl
without it), every other vehicle and walker of actors.jsonl with its nuScenes category, its box centre in the ego
frame (CARLA vehicle frame: x forward, y RIGHT, z up — left-handed — so +y is "on your right"), its velocity, the
box-to-box distance (the footprints' gap, core.criticality.box_distance; 0 when touching), the range rate (centres),
its time to collision with the ego (core.criticality.ttc: boxes at constant velocity, exact; None beyond
ttc_horizon_s), and its lane relation from lanes.json; the traffic light, the scenario and its criticality when the
run has them. Units in the text: metres, m/s (the ego's speed also km/h), seconds.

What CARLA's recorded state supports, and what is said:
* Positions, distances, directions, speeds, closing rates, counts, TTC — exact from the state. Velocities are
  actors.jsonl's (the world snapshot's), except in a kinematic scenario run (run.json scenario.motion): its director
  places the vehicles with set_transform, physics off, and the snapshot reports them standing still — there the
  director's own velocities (trajectories.json) are used for its actors and the ego, and the central difference of
  the positions for any other actor (the manifest's "velocities" says which).
* Lanes (lanes.json: driving-lane waypoints with width and heading): an object's lane is the driving lane whose
  waypoint stretch contains it (|lateral| ≤ width/2, |along| ≤ step·¾). Relative to the ego's lane: "your lane"
  (the same road, section and lane), same direction left/right with the lane count (lateral offset of the lane
  centres ÷ the ego lane's width, rounded), "oncoming" (heading differs by more than 135°), "crossing" (45°–135°),
  or off the roadway (no driving lane: a sidewalk, a verge). Heuristic on curves (the offset is measured along the
  ego lane's right at its waypoint). Without lanes.json no lane is named; the ego's path is then a corridor of the
  ego's width + corridor_margin_m on each side.
* Right of way: CARLA's state gives it only through traffic lights. lanes.json carries no priority rules, and
  traffic-light states are not recorded by default: a sentence about who must stop is made only when actors.jsonl
  lines carry "traffic_light" {"id", "state"} for the ego (CARLA's Vehicle.get_traffic_light /
  get_traffic_light_state); otherwise none is made and the manifest says so.
* Why to brake (heuristic, thresholds in the manifest): (1) an object ahead of the ego (centre x > 0) whose TTC is
  below ttc_brake_s (default 3 s); (2) a pedestrian or cyclist in the ego's path (its lane, or the corridor) within
  the look-ahead (max(10 m, speed · path_lookahead_s) beyond the ego's front), or entering it: its position, moved at
  its velocity relative to the ego, inside the path at some time within path_lookahead_s (sampled every 0.1 s);
  (3) a red or yellow light, when recorded.
* Scenario runs (run.json "scenario"): its name, the timeline events as they fire (their actions read from the
  drive config when it is found: OUT/drive.json or run.json's provenance input "config"), the run's criticality
  (core.criticality.summarise) and its most critical moment (the critical actor's minimum TTC). A scenario actor is
  matched to its CARLA id by position: its trajectories.json rows against actors.jsonl at the same step (≤ 0.1 m).

Grounding. Every caption, question and answer is {'text', 'refs': [{'ids', 'span', 'phrase'}]}: span = [start, end)
character offsets into text with text[start:end] == phrase, ids the CARLA actor ids (track ids, the same in every
frame, as objects/…json "track_id") the phrase refers to; each record names its step, frame and sim_t. Referring
phrases: indefinite in captions ("a car 23 m ahead"), definite in questions — by direction ("the car ahead on your
right"; "the nearest / second-nearest …" when two share it) or by distance ("the car 12 m away").

Determinism. Phrasings are chosen among several per template with a seeded generator (seed, step, slot): the same
seed gives the same text, byte for byte.

Paraphrase hook. paraphrase(text, refs, client) asks the client (an object passed in by the caller: a callable or
an object with .complete(prompt) -> str; this module opens no connection of its own and never paraphrases unless a
client is given) to rephrase while keeping every object phrase verbatim; the result is accepted only if every
phrase is found (spans re-located, in order, not overlapping) and its numbers are exactly the original's —
otherwise ParaphraseRejected.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re

import numpy as np

from . import criticality as C
from . import worldmodel as WM

SCHEMA = 'falcon.language/1'
MANIFEST = 'language.json'
DEFAULTS = {'seed': 0, 'every': 1, 'radius_m': 50.0, 'max_described': 6, 'ttc_brake_s': 3.0, 'ttc_horizon_s': 10.0,
            'path_lookahead_s': 3.0, 'corridor_margin_m': 0.5, 'still_mps': 0.3, 'closing_mps': 0.5}
EGO_FALLBACK_EXTENT = (2.45, 1.05, 0.75)       # half sizes (m) when the ego's row is missing: a Lincoln MKZ's
ORDINALS = ('nearest', 'second-nearest', 'third-nearest', 'fourth-nearest', 'fifth-nearest')
NUMBER_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}
GROUPS = (('vehicle', 'vehicle', 'vehicles'), ('pedestrian', 'pedestrian', 'pedestrians'),
          ('cyclist', 'cyclist', 'cyclists'))
ACTION_PHRASES = {'brake': 'starts braking', 'lane_change': 'starts changing lanes', 'walk': 'starts walking',
                  'speed': 'changes speed', 'open_door': 'opens a door', 'stop': 'stops'}


# ============================================================================================ nouns
def noun_of(category: str) -> str:
    c = str(category or '')
    table = (('human.pedestrian.child', 'child'), ('human.pedestrian.police_officer', 'police officer'),
             ('human.pedestrian.construction_worker', 'construction worker'), ('human.', 'pedestrian'),
             ('vehicle.bicycle', 'cyclist'), ('vehicle.motorcycle', 'motorcyclist'), ('vehicle.bus', 'bus'),
             ('vehicle.truck', 'truck'), ('vehicle.trailer', 'trailer'), ('vehicle.emergency.ambulance', 'ambulance'),
             ('vehicle.emergency.police', 'police car'), ('vehicle.construction', 'construction vehicle'),
             ('vehicle.', 'car'), ('movable_object.trafficcone', 'traffic cone'), ('movable_object.barrier', 'barrier'))
    for prefix, noun in table:
        if c.startswith(prefix):
            return noun
    return 'object'


def _article(noun):
    return 'an' if noun[:1] in 'aeiou' else 'a'


def _m(d):
    """A distance in the text: whole metres from 10 m, one decimal below."""
    return f'{d:.0f}' if d >= 10.0 else f'{d:.1f}'


def direction_of(x: float, y: float) -> str:
    """Bearing words in the ego frame (+y is the RIGHT in CARLA's left-handed frame)."""
    b = math.degrees(math.atan2(y, x))
    side = 'right' if b > 0 else 'left'
    a = abs(b)
    if a <= 15.0:
        return 'ahead'
    if a <= 60.0:
        return f'ahead on your {side}'
    if a <= 120.0:
        return f'on your {side}'
    if a < 165.0:
        return f'behind on your {side}'
    return 'behind'


# ============================================================================================ lanes
class LaneIndex:
    """lanes.json for lookups: which driving lane contains a world point (module doc)."""

    def __init__(self, lanes: dict):
        cols = lanes['columns']
        A = lanes['points']
        g = {k: cols.index(k) for k in ('x', 'y', 'yaw', 'width', 'road', 'section', 'lane')}
        self.P = np.array([[r[g['x']], r[g['y']]] for r in A], float).reshape(-1, 2)
        self.yaw = np.array([r[g['yaw']] for r in A], float)
        self.width = np.array([r[g['width']] for r in A], float)
        self.key = [(r[g['road']], r[g['section']], r[g['lane']]) for r in A]
        self.step = float(lanes.get('step_m', 2.0))
        r = np.radians(self.yaw)
        self.fwd = np.stack([np.cos(r), np.sin(r)], 1)
        self.right = np.stack([-np.sin(r), np.cos(r)], 1)

    def locate(self, x: float, y: float):
        """Index of the waypoint whose lane stretch contains (x, y), or None (off the driving lanes)."""
        if not len(self.P):
            return None
        d = np.array([x, y]) - self.P
        lon = np.sum(d * self.fwd, 1)
        lat = np.sum(d * self.right, 1)
        ok = (np.abs(lat) <= self.width / 2.0) & (np.abs(lon) <= 0.75 * self.step)
        if not ok.any():
            return None
        k = np.flatnonzero(ok)
        return int(k[np.argmin(np.hypot(lon[k], lat[k]))])

    def relation(self, e, a) -> tuple:
        """(relation, lanes over) of lane waypoint a to the ego's lane waypoint e: ('ego', 0), ('left'|'right', n),
        ('oncoming', n), ('crossing', 0), ('off_road', None); (None, None) when the ego is on no lane."""
        if e is None:
            return None, None
        if a is None:
            return 'off_road', None
        if self.key[a] == self.key[e]:
            return 'ego', 0
        dyaw = abs((self.yaw[a] - self.yaw[e] + 180.0) % 360.0 - 180.0)
        off = float((self.P[a] - self.P[e]) @ self.right[e])
        n = max(1, int(round(abs(off) / max(self.width[e], 0.5))))
        if dyaw > 135.0:
            return 'oncoming', n
        if dyaw > 45.0:
            return 'crossing', 0
        return ('right' if off > 0 else 'left'), n


def lane_phrase(rel, n) -> str | None:
    if rel is None:
        return None
    if rel == 'ego':
        return 'in your lane'
    if rel in ('left', 'right'):
        return f'in the lane to your {rel}' if n == 1 else f'{NUMBER_WORDS.get(n, str(n))} lanes to your {rel}'
    if rel == 'oncoming':
        return 'in the oncoming lane' if n == 1 else 'in an oncoming lane'
    if rel == 'crossing':
        return 'in a crossing lane'
    return 'off the roadway'


# ============================================================================================ state
def _row8(c, yaw, v, ext, t=0.0):
    return [t, c[0], c[1], yaw, v[0], v[1], 2.0 * ext[0], 2.0 * ext[1]]


def build_state(step: int, sim_t: float, ego: dict, actors, lanes=None, frame=None, traffic_light=None,
                scenario=None, options: dict | None = None) -> dict:
    """The facts of one step (module doc). ego: an actors.jsonl-style row of the ego (x y z roll pitch yaw, vx vy vz,
    extent, bbox_center, id); actors: the other rows; lanes: a lanes.json document or a LaneIndex."""
    o = dict(DEFAULTS, **(options or {}))
    li = LaneIndex(lanes) if isinstance(lanes, dict) else lanes
    T = WM.pose_matrix(ego)
    Ti = np.linalg.inv(T)
    ext_e = np.asarray(ego.get('extent') or EGO_FALLBACK_EXTENT, float)
    ce, _Re, _ = WM.actor_box(dict(ego, extent=list(ext_e)))
    ve = np.array([ego.get('vx', 0.0), ego.get('vy', 0.0), ego.get('vz', 0.0)], float)
    ego8 = _row8(ce, ego['yaw'], ve, ext_e)
    foot_e = C.box_corners(ce[0], ce[1], ego['yaw'], 2 * ext_e[0], 2 * ext_e[1])
    e_lane = li.locate(*ce[:2]) if li is not None else None
    speed = float(np.hypot(ve[0], ve[1]))
    half_corr = ext_e[1] + o['corridor_margin_m']
    look = max(10.0, speed * o['path_lookahead_s']) + ext_e[0]
    agents = []
    for a in actors or ():
        c, R, ext = WM.actor_box(a)
        p = Ti[:3, :3] @ c + Ti[:3, 3]
        va = np.array([a.get('vx', 0.0), a.get('vy', 0.0), a.get('vz', 0.0)], float)
        vrel = Ti[:3, :3] @ (va - ve)
        rng = float(np.hypot(p[0], p[1]))
        rr = float((p[0] * vrel[0] + p[1] * vrel[1]) / max(rng, 1e-6))
        foot = C.box_corners(c[0], c[1], a['yaw'], 2 * ext[0], 2 * ext[1])
        dist = C.box_distance(foot_e, foot)
        ttc = C.ttc(ego8, _row8(c, a['yaw'], va, ext), o['ttc_horizon_s'])
        a_lane = li.locate(*c[:2]) if li is not None else None
        rel, n = li.relation(e_lane, a_lane) if li is not None else (None, None)
        group = WM.bev_class_of(a.get('category'))
        if li is not None and rel is not None:
            in_path_now = rel == 'ego'
        else:
            in_path_now = abs(p[1]) <= half_corr + ext[1]
        ahead = p[0] > 0
        in_path = bool(in_path_now and ahead and p[0] - ext[0] <= look)
        entering = False
        if group in ('pedestrian', 'cyclist') and not in_path:
            ts = np.arange(0.0, o['path_lookahead_s'] + 1e-9, 0.1)
            px, py = p[0] + vrel[0] * ts, p[1] + vrel[1] * ts
            entering = bool(np.any((px > 0) & (px - ext[0] <= look) & (np.abs(py) <= half_corr + ext[1])))
        agents.append({'id': a['id'], 'category': a.get('category'), 'noun': noun_of(a.get('category')),
                       'group': group, 'x': round(float(p[0]), 3), 'y': round(float(p[1]), 3),
                       'z': round(float(p[2]), 3), 'distance_m': round(float(dist), 3), 'range_m': round(rng, 3),
                       'range_rate_mps': round(rr, 3), 'speed_mps': round(float(np.hypot(va[0], va[1])), 3),
                       'vrel': [round(float(vrel[0]), 3), round(float(vrel[1]), 3)],
                       'length': round(2 * float(ext[0]), 3), 'width': round(2 * float(ext[1]), 3),
                       'ttc_s': None if not math.isfinite(ttc) else round(ttc, 3),
                       'lane_rel': rel, 'lanes_over': n, 'in_path': in_path, 'entering_path': entering,
                       'direction': direction_of(float(p[0]), float(p[1]))})
    agents.sort(key=lambda g: (g['distance_m'], g['id']))
    lead = next((g for g in sorted(agents, key=lambda g: g['x']) if g['in_path'] and g['group'] in
                 ('vehicle', 'cyclist') and g['range_m'] <= o['radius_m']), None)
    return {'step': step, 'frame': frame, 'sim_t': sim_t, 'ego_id': ego.get('id'),
            'ego': {'speed_mps': round(speed, 3), 'yaw': ego['yaw'], 'length': round(2 * float(ext_e[0]), 3),
                    'width': round(2 * float(ext_e[1]), 3), 'on_lane': e_lane is not None if li is not None else None},
            'agents': agents, 'lead_id': lead['id'] if lead else None, 'lanes': li is not None,
            'traffic_light': traffic_light, 'scenario': scenario,
            'thresholds': {k: o[k] for k in ('ttc_brake_s', 'ttc_horizon_s', 'path_lookahead_s', 'corridor_margin_m',
                                             'radius_m')}}


# ============================================================================================ grounded text
def _rng(seed, step, slot) -> random.Random:
    h = hashlib.sha256(f'{seed}|{step}|{slot}'.encode()).digest()
    return random.Random(int.from_bytes(h[:8], 'big'))


_FIELD = re.compile(r'\{(\w+)(:cap)?\}')


def render(template: str, objs: dict | None = None, **vals) -> dict:
    """{'text', 'refs'} of a template: {name} is an object (objs[name] = (phrase, ids)) or a value; {name:cap}
    capitalises its first letter. Object phrases get a ref with their span."""
    objs = objs or {}
    text, refs, pos = '', [], 0
    for m in _FIELD.finditer(template):
        text += template[pos:m.start()]
        pos = m.end()
        name, cap = m.group(1), bool(m.group(2))
        if name in objs:
            phrase, ids = objs[name]
            s = phrase[:1].upper() + phrase[1:] if cap else phrase
            refs.append({'ids': list(ids), 'span': [len(text), len(text) + len(s)], 'phrase': s})
            text += s
        else:
            s = str(vals[name])
            text += s[:1].upper() + s[1:] if cap else s
    text += template[pos:]
    return {'text': text, 'refs': refs}


class _Builder:
    """A sentence built piece by piece (lists of several objects)."""

    def __init__(self):
        self.text, self.refs = '', []

    def add(self, s):
        self.text += s
        return self

    def obj(self, phrase, ids):
        self.refs.append({'ids': list(ids), 'span': [len(self.text), len(self.text) + len(phrase)], 'phrase': phrase})
        self.text += phrase
        return self

    def done(self):
        return {'text': self.text, 'refs': self.refs}


def _indef(g) -> str:
    return f'{_article(g["noun"])} {g["noun"]} {_m(g["distance_m"])} m {g["direction"]}'


def _def_by_direction(g, agents) -> str:
    same = [h for h in agents if h['noun'] == g['noun'] and h['direction'] == g['direction']]
    if len(same) == 1:
        return f'the {g["noun"]} {g["direction"]}'
    k = sorted(same, key=lambda h: (h['distance_m'], h['id'])).index(g)
    if k < len(ORDINALS):
        return f'the {ORDINALS[k]} {g["noun"]} {g["direction"]}'
    return f'the {g["noun"]} with track id {g["id"]}'


def _def_by_distance(g, agents) -> str:
    same = [h for h in agents if h['noun'] == g['noun'] and _m(h['distance_m']) == _m(g['distance_m'])]
    if len(same) == 1:
        return f'the {g["noun"]} {_m(g["distance_m"])} m away'
    return f'the {g["noun"]} with track id {g["id"]}'


def _motion(g, o) -> str:
    if g['speed_mps'] < o['still_mps'] and abs(g['range_rate_mps']) < o['closing_mps']:
        return 'standing still'
    if g['range_rate_mps'] <= -o['closing_mps']:
        return f'closing at {-g["range_rate_mps"]:.1f} m/s'
    if g['range_rate_mps'] >= o['closing_mps']:
        return f'pulling away at {g["range_rate_mps"]:.1f} m/s'
    return 'keeping its distance'


def _brake_reasons(st) -> list:
    th = st['thresholds']
    out = []
    for g in st['agents']:
        if g['x'] > 0 and g['ttc_s'] is not None and g['ttc_s'] < th['ttc_brake_s']:
            out.append(('ttc', g))
    for g in st['agents']:
        if g['group'] in ('pedestrian', 'cyclist') and (g['in_path'] or g['entering_path']) and \
                not any(r[1] is g for r in out):
            out.append(('in_path' if g['in_path'] else 'entering', g))
    tl = st.get('traffic_light')
    if tl and str(tl.get('state', '')).lower() in ('red', 'yellow'):
        out.append(('light', tl))
    return out


T_EGO = ['You are driving at {kmh} km/h ({mps} m/s).', 'The ego vehicle is travelling at {mps} m/s ({kmh} km/h).',
         'Your speed is {kmh} km/h.']
T_EGO_STOP = ['You are standing still.', 'The ego vehicle is stopped.']
T_AGENT_LANE = ['{A:cap} {lane}, {motion}.', 'There is {A} {lane}; it is {motion}.', '{A:cap}, {lane}, is {motion}.']
T_AGENT = ['{A:cap}, {motion}.', 'There is {A}; it is {motion}.', '{A:cap} is {motion}.']
T_LEAD = ['The vehicle ahead in your lane is {A}, {motion}.', 'Leading you in your lane: {A}, {motion}.']
T_LEAD_CORR = ['The vehicle ahead in your path is {A}, {motion}.', 'Leading you: {A}, {motion}.']
T_LEAD_TTC = ['At the current speeds you would collide with {A} in {ttc} s.',
              'Time to collision with {A}: {ttc} s.']
T_NO_LEAD = ['No vehicle is ahead of you in your lane within {R} m.', 'There is no lead vehicle in your lane within '
                                                                      '{R} m.']
T_NO_LEAD_CORR = ['No vehicle is ahead in your path within {R} m.', 'There is no lead vehicle in your path within '
                                                                    '{R} m.']
T_BRAKE_TTC = ['Brake: {A} is {ttc} s from a collision at the current speeds.',
               'You should brake: at the current speeds the collision with {A} is {ttc} s away.']
T_BRAKE_PATH = ['Slow down: {A} is {how} your path.', 'Be ready to stop: {A} is {how} your path.']
T_BRAKE_LIGHT = ['Stop: the traffic light ahead of you is {state}.', 'The light for your lane is {state}: stop.']
T_NO_BRAKE = ['Nothing calls for braking: no time to collision ahead is below {thr} s and no one is in or entering '
              'your path.', 'No need to brake now: nothing ahead is within {thr} s of a collision and your path is '
                            'clear of pedestrians and cyclists.']
T_LIGHT_GO = ['The traffic light ahead of you is green: you have the right of way through it.']

Q_COUNT = ['How many {plural} are within {R} m of you?', 'Count the {plural} within {R} m.']
Q_DIST = ['How far is {A} from you?', 'What is the distance to {A}?']
A_DIST = ['{A:cap} is {d} m away.', 'The gap to {A} is {d} m.']
Q_SIDE = ['Is {A} on your left or on your right?', 'On which side of you is {A}?']
A_SIDE = ['{A:cap} is on your {side}.', 'On your {side}: {A}.']
Q_WHERE = ['Where is {A}?', 'In which direction is {A}?']
A_WHERE = ['{A:cap} is {dir}.']
Q_MOTION = ['Is {A} getting closer or moving away?', 'Is {A} moving towards you or away from you?']
Q_SPEED = ['How fast is {A} moving?', 'What is the speed of {A}?']
Q_LEAD = ['Is there a vehicle ahead of you in your lane?', 'Is anything leading you in your lane?']
Q_LEAD_CORR = ['Is there a vehicle ahead of you in your path?']
Q_TTC = ['What is the time to collision with {A}?', 'How long until a collision with {A} at the current speeds?']
Q_BRAKE = ['Should you brake now?', 'Do you need to brake?']
Q_LIGHT = ['What colour is the traffic light ahead of you?']


def describe(state: dict, seed: int = 0, options: dict | None = None) -> dict:
    """{'captions': [sentence], 'qa': [pair]} of a state (build_state). A sentence: {'kind', 'text', 'refs', 'ids'};
    a pair: {'type', 'question', 'answer', 'value', 'unit', 'ids'} (question and answer: {'text', 'refs'})."""
    o = dict(DEFAULTS, **(options or {}))
    step = state['step']
    slot = [0]

    def pick(templates):
        slot[0] += 1
        return _rng(seed, step, slot[0]).choice(templates)

    agents = [g for g in state['agents'] if g['range_m'] <= o['radius_m']]
    by_id = {g['id']: g for g in agents}
    caps, qa = [], []
    R = f'{o["radius_m"]:g}'

    def cap(kind, s):
        s['kind'] = kind
        s['ids'] = sorted({i for r in s['refs'] for i in r['ids']}, key=str)
        caps.append(s)

    def pair(kind, q, a, value=None, unit=None):
        ids = sorted({i for r in q['refs'] + a['refs'] for i in r['ids']}, key=str)
        qa.append({'type': kind, 'question': q, 'answer': a, 'value': value, 'unit': unit, 'ids': ids})

    # the ego
    sp = state['ego']['speed_mps']
    if sp < o['still_mps']:
        cap('ego', render(pick(T_EGO_STOP)))
    else:
        cap('ego', render(pick(T_EGO), kmh=f'{sp * 3.6:.0f}', mps=f'{sp:.1f}'))
    # counts
    b = _Builder().add(pick([f'Within {R} m of you: ', f'In a {R} m radius: ']))
    parts = []
    for grp, one, many in GROUPS:
        ids = [g['id'] for g in agents if g['group'] == grp]
        parts.append((f'{len(ids)} {one if len(ids) == 1 else many}' if ids else f'no {many}', ids))
    for k, (ph, ids) in enumerate(parts):
        b.add('' if k == 0 else (' and ' if k == len(parts) - 1 else ', '))
        if ids:
            b.obj(ph, ids)
        else:
            b.add(ph)
    cap('count', b.add('.').done())
    # the lead
    lanes = state.get('lanes')
    lead = by_id.get(state.get('lead_id'))
    if lead is not None:
        cap('lead', render(pick(T_LEAD if lanes else T_LEAD_CORR), {'A': (_indef(lead), [lead['id']])},
                           motion=_motion(lead, o)))
        if lead['ttc_s'] is not None:
            cap('lead_ttc', render(pick(T_LEAD_TTC), {'A': ('the vehicle ahead', [lead['id']])},
                                   ttc=f'{lead["ttc_s"]:.1f}'))
    else:
        cap('lead', render(pick(T_NO_LEAD if lanes else T_NO_LEAD_CORR), R=R))
    # the others, nearest first
    described = [g for g in agents if g is not lead][:o['max_described']]
    for g in described:
        lp = lane_phrase(g['lane_rel'], g['lanes_over'])
        objs = {'A': (_indef(g), [g['id']])}
        if lp:
            cap('object', render(pick(T_AGENT_LANE), objs, lane=lp, motion=_motion(g, o)))
        else:
            cap('object', render(pick(T_AGENT), objs, motion=_motion(g, o)))
    # traffic light (right of way) and braking
    tl = state.get('traffic_light')
    reasons = _brake_reasons(dict(state, agents=agents))
    for why, g in reasons:
        if why == 'ttc':
            cap('brake', render(pick(T_BRAKE_TTC), {'A': (_indef(g), [g['id']])}, ttc=f'{g["ttc_s"]:.1f}'))
        elif why in ('in_path', 'entering'):
            cap('brake', render(pick(T_BRAKE_PATH), {'A': (_indef(g), [g['id']])},
                                how='in' if why == 'in_path' else 'entering'))
        else:
            cap('brake', render(pick(T_BRAKE_LIGHT), state=str(g['state']).lower()))
    if not reasons:
        cap('brake', render(pick(T_NO_BRAKE), thr=f'{state["thresholds"]["ttc_brake_s"]:g}'))
    if tl and str(tl.get('state', '')).lower() == 'green':
        cap('right_of_way', render(pick(T_LIGHT_GO)))
    # the scenario
    sc = state.get('scenario')
    if sc:
        for ev in sc.get('events_now', ()):
            act = ACTION_PHRASES.get(ev.get('do'), 'acts') if ev.get('do') else 'triggers a scenario event'
            g = by_id.get(ev.get('actor_id'))
            if g is not None:
                cap('scenario', render(f'{sc["name"]}: {{A:cap}} {act} now.', {'A': (_indef(g), [g['id']])}))
            else:
                cap('scenario', render(f'{sc["name"]}: the scenario\'s {ev.get("actor") or "event"} {act} now.'))
        cm = sc.get('critical_now')
        if cm:
            g = by_id.get(cm.get('actor_id'))
            if g is not None:
                cap('criticality', render('This is the run\'s most critical moment: the time to collision with {A} '
                                          'is at its minimum, {ttc} s.', {'A': (_indef(g), [g['id']])},
                                          ttc=f'{cm["min_ttc_s"]:.2f}'))

    # ---- questions and answers
    for grp, one, many in GROUPS:
        ids = [g['id'] for g in agents if g['group'] == grp]
        q = render(pick(Q_COUNT), plural=many, R=R)
        a = _Builder().obj(f'{len(ids)} {one if len(ids) == 1 else many}', ids).add('.').done() if ids else \
            render(f'None within {R} m.')
        pair('count', q, a, len(ids), 'objects')
    for g in described + ([lead] if lead is not None else []):
        dref = (_def_by_direction(g, agents), [g['id']])
        nref = (_def_by_distance(g, agents), [g['id']])
        pair('distance', render(pick(Q_DIST), {'A': dref}),
             render(pick(A_DIST), {'A': dref}, d=f'{g["distance_m"]:.1f}'), g['distance_m'], 'm')
        if abs(g['y']) >= 1.0:
            side = 'right' if g['y'] > 0 else 'left'
            pair('relative_position', render(pick(Q_SIDE), {'A': nref}), render(pick(A_SIDE), {'A': nref}, side=side),
                 side, None)
        else:
            pair('relative_position', render(pick(Q_WHERE), {'A': nref}),
                 render(pick(A_WHERE), {'A': nref}, dir=g['direction']), g['direction'], None)
        pair('motion', render(pick(Q_MOTION), {'A': dref}), render('{A:cap} is {m}.', {'A': dref}, m=_motion(g, o)),
             -g['range_rate_mps'], 'm/s closing')
        sa = render('{A:cap} is standing still.', {'A': dref}) if g['speed_mps'] < o['still_mps'] else \
            render('{A:cap} moves at {v} m/s ({k} km/h).', {'A': dref}, v=f'{g["speed_mps"]:.1f}',
                   k=f'{g["speed_mps"] * 3.6:.0f}')
        pair('speed', render(pick(Q_SPEED), {'A': dref}), sa, g['speed_mps'], 'm/s')
    if lead is not None:
        a = render('Yes: {A}.', {'A': (_indef(lead), [lead['id']])})
    else:
        a = render(f'No, not within {R} m.')
    pair('lead', render(pick(Q_LEAD if lanes else Q_LEAD_CORR)), a, lead['id'] if lead else None, 'track id')
    if lead is not None:
        dref = ('the vehicle ahead of you', [lead['id']])
        if lead['ttc_s'] is not None:
            a = render('At the current speeds, {A} is {t} s from a collision.', {'A': dref}, t=f'{lead["ttc_s"]:.1f}')
        else:
            a = render(f'{{A:cap}} is not on a collision course within {state["thresholds"]["ttc_horizon_s"]:g} s.',
                       {'A': dref})
        pair('risk_ttc', render(pick(Q_TTC), {'A': dref}), a, lead['ttc_s'], 's')
    if reasons:
        why, g = reasons[0]
        if why == 'ttc':
            a = render('Yes: {A} is {t} s from a collision.', {'A': (_indef(g), [g['id']])}, t=f'{g["ttc_s"]:.1f}')
        elif why in ('in_path', 'entering'):
            a = render('Yes: {A} is {how} your path.', {'A': (_indef(g), [g['id']])},
                       how='in' if why == 'in_path' else 'entering')
        else:
            a = render('Yes: the traffic light ahead is {s}.', s=str(g['state']).lower())
    else:
        a = render(f'No: nothing ahead is within {state["thresholds"]["ttc_brake_s"]:g} s of a collision and no one '
                   'is in or entering your path.')
    pair('risk_brake', render(pick(Q_BRAKE)), a, bool(reasons), 'yes/no')
    if tl and tl.get('state'):
        pair('traffic_light', render(pick(Q_LIGHT)), render(f'{str(tl["state"]).capitalize()}.'),
             str(tl['state']).lower(), None)
    for s in caps:
        s.update(step=step, frame=state.get('frame'))
    for p in qa:
        p.update(step=step, frame=state.get('frame'))
    return {'captions': caps, 'qa': qa}


# ============================================================================================ paraphrase hook
class ParaphraseRejected(ValueError):
    """A paraphrase that loses an object phrase, overlaps phrases, or changes the numbers."""


PARAPHRASE_PROMPT = ('Rephrase the sentence below in natural English, keeping its meaning. Copy each of these '
                     'phrases exactly as written, character for character: {phrases}. Do not add, drop or change any '
                     'number. Answer with the rephrased sentence only.\n\nSentence: {text}')
_NUM = re.compile(r'\d+(?:\.\d+)?')


def paraphrase(text: str, refs, client) -> dict:
    """Ask client to rephrase text (module doc) and re-ground it. refs: the sentence's refs (or the sentence dict).
    Returns {'text', 'refs', 'source_text'}; raises ParaphraseRejected when the result would lose grounding."""
    if isinstance(refs, dict):
        refs = refs.get('refs', [])
    if client is None:
        raise ValueError('no paraphrase client: paraphrasing runs only through a client the caller passes in')
    prompt = PARAPHRASE_PROMPT.format(phrases='; '.join(f'"{r["phrase"]}"' for r in refs) or '(none)', text=text)
    out = client.complete(prompt) if hasattr(client, 'complete') else client(prompt)
    if not isinstance(out, str) or not out.strip():
        raise ParaphraseRejected('the client returned no text')
    out = out.strip()
    taken, new_refs = [], []
    for r in sorted(refs, key=lambda r: r['span'][0]):
        k, ph = -1, r['phrase']
        for cand in (ph, ph[:1].swapcase() + ph[1:]):     # a sentence start may change its capital
            k = out.find(cand)
            while k >= 0 and any(k < b and k + len(cand) > a for a, b in taken):
                k = out.find(cand, k + 1)
            if k >= 0:
                ph = cand
                break
        if k < 0:
            raise ParaphraseRejected(f'the paraphrase drops the reference to {r["ids"]}: "{r["phrase"]}"')
        taken.append((k, k + len(ph)))
        new_refs.append(dict(r, span=[k, k + len(ph)], phrase=ph))
    if sorted(_NUM.findall(out)) != sorted(_NUM.findall(text)):
        raise ParaphraseRejected(f'the paraphrase changes the numbers ({_NUM.findall(text)} -> {_NUM.findall(out)})')
    return {'text': out, 'refs': new_refs, 'source_text': text}


# ============================================================================================ a run
def _scenario_context(data: dict) -> dict | None:
    """The scenario of a run (module doc): name, events by step with CARLA ids, the most critical moment."""
    sc = (data['run'] or {}).get('scenario')
    if not sc:
        return None
    fps = data['fps']
    frames = data['frames']
    steps = [f['step'] for f in frames]
    acts = data.get('actors') or {}
    traj_p = os.path.join(data['dir'], 'trajectories.json')
    ids = {}
    traj = None
    if os.path.isfile(traj_p):
        with open(traj_p, encoding='utf-8') as fh:
            traj = json.load(fh).get('actors', {})
        for name, rows in traj.items():
            if name == 'ego' or not rows:
                continue
            votes = {}
            for j in sorted({0, len(rows) // 2, len(rows) - 1}):
                if j >= len(steps):
                    continue
                cand = [(math.hypot(a['x'] - rows[j][1], a['y'] - rows[j][2]), a['id']) for a in acts.get(steps[j], [])]
                if cand:
                    d, aid = min(cand)
                    if d <= 0.1:
                        votes[aid] = votes.get(aid, 0) + 1
            if votes:
                ids[name] = max(votes, key=votes.get)
    timeline = None
    for p in (os.path.join(data['dir'], 'drive.json'),
              os.path.expanduser(((data['run'].get('provenance') or {}).get('inputs') or {}).get('config', {})
                                 .get('path', '') or '')):
        if p and os.path.isfile(p):
            try:
                with open(p, encoding='utf-8') as fh:
                    timeline = (json.load(fh).get('scenario') or {}).get('timeline')
            except (OSError, ValueError):
                timeline = None
            if timeline is not None:
                break
    events = []
    for k, t in sc.get('fired') or ():
        actions = (timeline[k].get('actions') if timeline and k < len(timeline) else None) or [{}]
        for a in actions:
            events.append({'t': t, 'step': int(round(float(t) * fps)), 'actor': a.get('actor'),
                           'actor_id': ids.get(a.get('actor')), 'do': a.get('do')})
    crit = (sc.get('criticality') or {})
    ov = crit.get('overall') or {}
    critical = None
    ca = ov.get('min_ttc_actor')
    if ca is not None and ov.get('min_ttc_s') is not None:
        t = (crit.get('actors') or {}).get(ca, {}).get('min_ttc_t')
        rows = (traj or {}).get(ca) or []
        if t is not None and rows:
            j = int(np.argmin([abs(r[0] - t) for r in rows]))            # trajectories.json row j = step j
            step = steps[j] if j < len(steps) else None
        else:
            step = int(round(float(t) * fps)) - 1 if t is not None else None   # rows are (i + 1) / fps
        critical = {'actor': ca, 'actor_id': ids.get(ca), 'min_ttc_s': ov['min_ttc_s'], 'step': step}
    return {'name': sc.get('name') or sc.get('id') or 'scenario', 'id': sc.get('id'), 'events': events,
            'actor_ids': ids, 'criticality': crit.get('criticality'), 'passed': sc.get('passed'),
            'critical': critical, 'timeline_found': timeline is not None, 'motion': sc.get('motion'),
            '_traj': traj}


def _velocities(data: dict, ctx: dict | None) -> tuple[dict, str]:
    """({(step, actor id): (vx, vy)} replacing actors.jsonl's velocities, what they are). A kinematic scenario places
    its vehicles with set_transform, physics off: the snapshot (actors.jsonl) reports them standing still. Their
    velocities are then the director's own (trajectories.json, row j = step j, matched to CARLA ids by position) and
    every other actor's the central difference of its positions (np.gradient over consecutive steps)."""
    if not ctx or ctx.get('motion') != 'kinematic':
        return {}, 'recorded (actors.jsonl, the world snapshot)'
    out = {}
    tracks = {}
    for step in sorted(data['actors']):
        for a in data['actors'][step]:
            tracks.setdefault(a['id'], []).append((step, data['frames'][0]['sim_t'] + step / data['fps'], a['x'],
                                                   a['y']))
    for aid, rows in tracks.items():
        A = np.asarray(rows, float)
        if len(A) < 2:
            continue
        for k in range(len(A)):                    # central inside, one-sided at the ends (consecutive steps only)
            lo, hi = max(k - 1, 0), min(k + 1, len(A) - 1)
            if A[hi, 0] - A[lo, 0] != hi - lo:
                continue
            dt = A[hi, 1] - A[lo, 1]
            out[(int(A[k, 0]), aid)] = ((A[hi, 2] - A[lo, 2]) / dt, (A[hi, 3] - A[lo, 3]) / dt)
    steps = [f['step'] for f in data['frames']]
    names = dict(ctx.get('actor_ids') or {}, ego=data.get('ego_id'))
    for name, rows in (ctx.get('_traj') or {}).items():
        aid = names.get(name)
        if aid is None:
            continue
        for j, r in enumerate(rows[:len(steps)]):
            out[(steps[j], aid)] = (float(r[4]), float(r[5]))
    return out, ('kinematic scenario: the director\'s velocities (trajectories.json) for its actors and the ego, '
                 'central differences of the positions for the others; actors.jsonl reports set_transform-placed '
                 'actors as standing still')


def derive_language(run_dir: str, out_dir: str, options: dict | None = None, client=None) -> dict:
    """language.jsonl (one record per described step: step, frame, sim_t, captions, qa, facts) and language.json
    (the manifest, with provenance) in out_dir. options: DEFAULTS' keys. client: a paraphrase client (never by
    default): each caption is paraphrased through it, a rejected paraphrase keeps the template text (counted)."""
    from .provenance import block, write_json
    o = dict(DEFAULTS, **(options or {}))
    unknown = sorted(set(o) - set(DEFAULTS))
    if unknown:
        raise ValueError(f'unknown options: {", ".join(unknown)}')
    data = WM.load_run(run_dir)
    if data['actors'] is None:
        raise ValueError(f'{run_dir}: no actors.jsonl — the language labels need every actor\'s state per step')
    li = LaneIndex(data['lanes']) if data['lanes'] and data['lanes'].get('points') else None
    sc = _scenario_context(data)
    vel, vel_source = _velocities(data, sc)
    ego_id = data['ego_id']
    os.makedirs(out_dir, exist_ok=True)
    n = n_caps = n_qa = n_para = n_rej = 0
    lights = False
    every = max(int(o['every']), 1)
    with open(os.path.join(out_dir, 'language.jsonl'), 'w', encoding='utf-8') as fh:
        for k, f in enumerate(data['frames']):
            if k % every or f.get('recorded') is False:
                continue
            line = data['actors'].get(f['step'])
            if line is None:
                continue
            line = [dict(a, vx=vel[(f['step'], a['id'])][0], vy=vel[(f['step'], a['id'])][1])
                    if (f['step'], a['id']) in vel else a for a in line]
            ego = next((a for a in line if a['id'] == ego_id), None)
            if ego is None:                          # no row for the ego: its pose and speed from frames.jsonl
                yaw = math.radians(f['ego']['yaw'])
                v = f.get('speed_mps', 0.0)
                ego = dict(f['ego'], id=ego_id, vx=v * math.cos(yaw), vy=v * math.sin(yaw), vz=0.0,
                           extent=list(EGO_FALLBACK_EXTENT), bbox_center=[0.0, 0.0, EGO_FALLBACK_EXTENT[2]])
            others = [a for a in line if a['id'] != ego_id]
            tl = _light_of(data, f['step'])
            lights = lights or tl is not None
            scs = None
            if sc is not None:
                scs = {'name': sc['name'], 'events_now': [e for e in sc['events'] if e['step'] == f['step']],
                       'critical_now': sc['critical'] if sc['critical'] and sc['critical']['step'] == f['step']
                       else None}
            st = build_state(f['step'], f['sim_t'], ego, others, li, f.get('frame'), tl, scs, o)
            d = describe(st, int(o['seed']), o)
            if client is not None:
                for s in d['captions']:
                    try:
                        p = paraphrase(s['text'], s['refs'], client)
                        s.update(template_text=s['text'], text=p['text'], refs=p['refs'], paraphrased=True)
                        n_para += 1
                    except ParaphraseRejected as e:
                        s.update(paraphrased=False, paraphrase_rejected=str(e))
                        n_rej += 1
            facts = {'ego': st['ego'], 'lead_id': st['lead_id'], 'traffic_light': tl,
                     'agents': [{k2: g[k2] for k2 in ('id', 'category', 'x', 'y', 'distance_m', 'range_rate_mps',
                                                      'speed_mps', 'ttc_s', 'lane_rel', 'lanes_over', 'in_path',
                                                      'entering_path')} for g in st['agents']
                                if g['range_m'] <= o['radius_m']]}
            fh.write(json.dumps({'step': f['step'], 'frame': f.get('frame'), 'sim_t': f['sim_t'],
                                 'captions': d['captions'], 'qa': d['qa'], 'facts': facts},
                                separators=(',', ':'), ensure_ascii=False) + '\n')
            n += 1
            n_caps += len(d['captions'])
            n_qa += len(d['qa'])
    ins = {k: os.path.join(run_dir, nm) for k, nm in (('run', 'run.json'), ('frames', 'frames.jsonl'),
                                                     ('lanes', 'lanes.json'), ('actors', 'actors.jsonl'))}
    manifest = {'schema': SCHEMA, 'run': os.path.abspath(run_dir), 'file': 'language.jsonl', 'records': n,
                'captions': n_caps, 'qa_pairs': n_qa, 'seed': int(o['seed']), 'ego_id': ego_id,
                'grounding': 'refs: ids = CARLA actor ids (track ids), span = [start, end) into the text',
                'coordinates': 'ego frame: CARLA vehicle frame, x forward, y right (left-handed): +y is "your right"',
                'distance': 'box-to-box gap of the footprints (0 when touching), metres',
                'velocities': vel_source,
                'lanes': 'lanes.json' if li is not None else 'none: lane relations not stated; the path is a corridor',
                'right_of_way': 'from the recorded traffic-light state' if lights else
                'not stated: the run records no traffic-light state and lanes.json carries no priority rules',
                'scenario': None if sc is None else {k: sc[k] for k in ('name', 'id', 'events', 'actor_ids',
                                                                         'criticality', 'passed', 'critical',
                                                                         'timeline_found')},
                'heuristics': {'brake': f'TTC ahead < {o["ttc_brake_s"]:g} s; a pedestrian or cyclist in or entering '
                                        f'the path within {o["path_lookahead_s"]:g} s; a red or yellow light',
                               'lane_relation': 'nearest driving-lane waypoint stretch; lanes over = centre offset / '
                                                'lane width'},
                'paraphrase': {'client': client is not None, 'accepted': n_para, 'rejected': n_rej},
                'options': o, 'provenance': block(ins, o['seed'], o)}
    write_json(os.path.join(out_dir, MANIFEST), manifest)
    return manifest


def _light_of(data, step):
    """The ego's traffic light at a step when actors.jsonl records it ({'id', 'state'}), else None."""
    path_rows = data.get('_lights')
    if path_rows is None:
        path_rows = data['_lights'] = {}
        p = os.path.join(data['dir'], 'actors.jsonl')
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as fh:
                for x in fh:
                    if '"traffic_light"' in x:
                        r = json.loads(x)
                        if r.get('traffic_light'):
                            path_rows[int(r['step'])] = r['traffic_light']
    return path_rows.get(step)
