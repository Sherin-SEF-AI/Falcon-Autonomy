"""Formal traffic-rule checking: traffic rules written as signal-temporal-logic formulas (core.stl) over the recorded
state of a run, checked with robustness values, violation intervals and counts per km (numpy at import).

    check_run(run_dir, ruleset) -> result        a recorded drive, a stack run, a live recording or a replay
    check_episode(episode, ruleset) -> result    a gym / Leaderboard episode that carries its per-step trace
    aggregate(results) -> summary                over runs or routes: counts, per km with intervals, runs violating
    falcon rules check RUN... [--set india|generic|FILE] [--out DIR] ;  falcon rules list ;  falcon rules show SET

Rule files (falcon.rules/1), a small line-oriented language parsed here (nothing is evaluated as Python):

    # a comment
    ruleset ID "Name"                    the set's id and name (once)
    handedness LHT | RHT                 the traffic side the set is written for
    about "text"                         the set's description (repeated lines continue it)
    let NAME = NUMBER                    a constant (numbers only); "# heuristic" after it is kept as its note
    rule ID "Title" [TAG ...]: FORMULA   a rule; the formula may continue on the following indented lines
    about ID "text"                      a rule's explanation (repeated lines continue it)

A rule's formula must hold at every sample: the run's robustness for it is ρ(G φ) = the minimum of φ's robustness
over the known samples (core.stl: units of the predicate; ≥ 0 satisfied); a violation is a maximal interval of
samples where φ's robustness is negative, with its start and end time and its most negative value; the counts per km
divide the violations by the distance the ego drove (its positions). Tags: "law" (the rule restates a traffic law),
"heuristic" (a threshold or a reading of the law that is Falcon's own choice). A rule some of whose signals are unknown
throughout the run is "not evaluable" and says which signals; a rule known on only part of the run gives the share of
time it was evaluated (core.stl's three-valued semantics: an unknown never makes a violation).

Signals (SIGNALS; all per sample, NaN where unknown):
    t                      s: a run's from its first sample; an episode's simulation time (its infractions' sim_t)
    speed_mps, speed_kmh   the ego's speed (frames.jsonl speed_mps; a gym trace's)
    accel_mps2             its central difference
    on_lane                1 when the ego's centre is on a driving lane of the map (lanes), else 0
    in_junction            1 in a junction: the map's OpenDRIVE road is a junction road; without the OpenDRIVE file, a
                           heuristic from lanes.json (a road whose lane points overlap two other roads' at headings
                           20°–160° apart); a gym trace: CARLA's Waypoint.is_junction
    lane_heading_err_deg   |ego yaw − lane heading| (0–180°) of the lane the ego is on (the lane nearest its heading
                           where lanes overlap)
    lanes_left, lanes_right  same-direction driving lanes beside the ego's lane on its left / right (0: the kerb-side
                           or the median-side lane); geometric from the lane points (heading within 30°, along ±step)
    lane_margin_m          the ego's body's distance inside its lane: width/2 − |lateral offset| − half the ego's
                           width (negative: the body is over a lane line)
    solid_margin_m         the distance from the body to the nearest solid (Solid or SolidSolid) line of its lane,
                           negative when over it; +inf when neither line is solid; unknown without markings (lanes
                           made from OpenDRIVE, gym traces without markings)
    lead_gap_m             bumper to bumper to the nearest vehicle ahead in the ego's corridor (a straight corridor in
                           the ego's frame: |lateral| < half widths + 0.3 m, heading within 60°, up to 80 m); +inf when
                           none; a gym trace: the environment's own lead feature
    ped_path_gap_m         to the nearest pedestrian in the ego's path (|lateral| < half widths + 1 m, ahead, up to
                           50 m); +inf when none; a gym trace: its vulnerable road user (pedestrian or cyclist)
    ped_on_crossing        1 when that pedestrian stands on a crosswalk of the map (OpenDRIVE crosswalk objects); 0
                           when none does; unknown when the map's crosswalks are not known
    pass_on_left, pass_on_right  1 at the sample where the ego passes a same-direction vehicle (its centre goes from
                           ahead to behind the ego's while beside it, the ego moving faster) with the ego on that
                           vehicle's left / right; passing = either
    tl_red, tl_distance_m  the light on the ego's route ahead is red; the distance from the ego's front to its stop
                           line — recorded by gym traces; drives do not record light states (unknown)
    speed_limit_kmh        CARLA's Vehicle.get_speed_limit() when the trace records it; else the map's OpenDRIVE record
                           for the ego's road (a lane <speed> or the road type's <speed>); in a junction road the last
                           limit before it (a limit holds until the next); unknown where neither exists

Map. lanes.json (driving-lane points with width, ids and markings: drives that record labels) and the map's OpenDRIVE
file — the drive's own (an imported world) or, for a CARLA town, CarlaUE4/Content/Carla/Maps/OpenDrive/<town>.xodr
under the CARLA folder in Settings — for junction roads, speed records and crosswalks; without lanes.json, the lane
points are made from the OpenDRIVE file (every driving lane's centre every 2 m: reference line, laneOffset, widths;
markings unknown). OpenDRIVE (x, y, heading) → CARLA (x, −y, −heading); the header's offset is not applied.

What is not recorded, so not checked: turn indicators (lane changes and turns without signalling); horn use;
right of way at unsignalled junctions (OpenDRIVE junction priorities are not read); traffic-light states in drives;
the legal speed limit where the map records none (the set's urban default is used and labelled heuristic). Gym traces
carry no other actors' poses (no passing events). These are said in each result.

Rule sets: INDIA (left-hand traffic; the Motor Vehicles (Driving) Regulations, 2017 as the rules restate them) and
GENERIC (right-hand traffic). `falcon rules show india > my.rules` writes a set to edit.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from . import stl as S

SCHEMA = 'falcon.rules_check/1'
SET_SCHEMA = 'falcon.rules/1'
SIGNALS = {
    't': ('s', 'time (a run: from its first sample; an episode: its simulation time)'),
    'speed_mps': ('m/s', 'the ego\'s speed'), 'speed_kmh': ('km/h', 'the ego\'s speed'),
    'accel_mps2': ('m/s²', 'the speed\'s central difference'),
    'on_lane': ('0/1', 'the ego\'s centre is on a driving lane'),
    'in_junction': ('0/1', 'the ego is in a junction'),
    'lane_heading_err_deg': ('°', '|ego yaw − its lane\'s heading|'),
    'lanes_left': ('lanes', 'same-direction lanes on the ego\'s left'),
    'lanes_right': ('lanes', 'same-direction lanes on the ego\'s right'),
    'lane_margin_m': ('m', 'the body\'s distance inside its lane (negative: over a line)'),
    'solid_margin_m': ('m', 'the body\'s distance to a solid line of its lane (negative: over it)'),
    'lead_gap_m': ('m', 'gap to the vehicle ahead in the corridor (+inf none)'),
    'ped_path_gap_m': ('m', 'gap to a pedestrian in the path (+inf none)'),
    'ped_on_crossing': ('0/1', 'that pedestrian is on a crosswalk'),
    'pass_on_left': ('0/1', 'the ego passes a vehicle with the ego on its left'),
    'pass_on_right': ('0/1', 'the ego passes a vehicle with the ego on its right'),
    'passing': ('0/1', 'either'),
    'tl_red': ('0/1', 'the light ahead is red'), 'tl_distance_m': ('m', 'front to the stop line ahead'),
    'speed_limit_kmh': ('km/h', 'the speed limit'),
}
EGO_DEFAULT = (4.9, 1.84)                         # length, width: CARLA's mkz_2020 (actors.jsonl extents × 2)
LEAD_RANGE_M, PED_RANGE_M = 80.0, 50.0
LEAD_MARGIN_M, PED_MARGIN_M = 0.3, 1.0
LEAD_HEADING_DEG, PASS_HEADING_DEG, PASS_BESIDE_M = 60.0, 45.0, 8.0
SAME_DIR_DEG = 30.0
SOLID = ('Solid', 'SolidSolid')
STATES = ('violated', 'satisfied', 'not evaluable')
NOT_RECORDED = ('turn indicators are not recorded: lane changes and turns without signalling are not checked',
                'right of way at unsignalled junctions is not checked (OpenDRIVE junction priorities are not read)',
                'horn use and vehicle condition are not recorded')

INDIA = '''# Falcon's traffic rules for India (falcon.rules/1). Left-hand traffic.
ruleset india "India (left-hand traffic)"
handedness LHT
about "The Motor Vehicles (Driving) Regulations, 2017 as these rules restate them, for Falcon's recorded state."
about "Values marked heuristic are Falcon's choices, not the law's: edit them here."

let urban_limit_kmh = 50        # heuristic: where the map records no limit (limits are notified per state and city)
let speed_tolerance_kmh = 0     # heuristic: no tolerance
let follow_time_s = 2           # heuristic: the two-second rule (the regulations ask for a sufficient distance)
let comfort_decel_mps2 = 4      # heuristic: a firm but comfortable stop
let stop_margin_m = 2           # heuristic: room left before a pedestrian
let red_line_m = 1.5            # heuristic: the stop line is reached within this distance (one step at 108 km/h, 20 Hz)
let lane_change_s = 3           # heuristic: a lane change takes up to 3 s
let junction_stand_s = 5        # heuristic: standing longer than this in a junction blocks it
let queue_gap_m = 10            # heuristic: queued behind a vehicle this close is not blocking by choice
let overtake_window_s = 10      # heuristic: time out of the kerb lane allowed around an overtake

rule speed_limit "Keep within the speed limit" law heuristic:
    speed_kmh <= default(speed_limit_kmh, urban_limit_kmh) + speed_tolerance_kmh
about speed_limit "The map's or CARLA's limit where known; elsewhere urban_limit_kmh (heuristic)."
rule stop_at_red "Stop at a red light" law:
    not (tl_red and tl_distance_m <= red_line_m) or speed_mps <= 0.5
about stop_at_red "Checked where light states are recorded (gym and Leaderboard traces); drives do not record them."
rule no_wrong_way "Do not drive against the lane's direction" law:
    in_junction or not on_lane or lane_heading_err_deg <= 135
rule keep_left "Keep to the left-most lane except to overtake" law heuristic:
    in_junction or not on_lane or lanes_left < 0.5
        or once[0,overtake_window_s] passing or eventually[0,overtake_window_s] passing
about keep_left "Regulation 2 (keep left). Out of the kerb lane only within overtake_window_s of passing a vehicle"
about keep_left "(heuristic); a lane blocked ahead is not an excuse here."
rule overtake_right "Overtake on the right" law:
    not pass_on_left
about overtake_right "Regulation 6 (overtaking on the right). The exceptions (a vehicle ahead turning right and"
about overtake_right "signalling; slow queues) need indicators, which are not recorded: such a pass is still counted."
rule lane_discipline "Do not straddle lane lines beyond a lane change" law heuristic:
    in_junction or not on_lane or t < lane_change_s or once[0,lane_change_s] (lane_margin_m >= 0)
rule solid_line "Do not cross a solid line" law:
    in_junction or solid_margin_m >= 0
rule following_distance "Keep a safe distance from the vehicle ahead" heuristic:
    speed_mps < 1 or lead_gap_m >= follow_time_s * speed_mps
about following_distance "The two-second rule (heuristic): the regulations ask for a distance sufficient to stop."
rule yield_crossing "Yield to pedestrians on a crossing" law:
    not ped_on_crossing or speed_mps <= 0.5
        or ped_path_gap_m - speed_mps * speed_mps / (2 * comfort_decel_mps2) >= stop_margin_m
about yield_crossing "Checked where the map's crosswalks are known (OpenDRIVE crosswalk objects)."
rule pedestrian_clearance "Be able to stop for a pedestrian in the path" heuristic:
    speed_mps <= 0.5 or ped_path_gap_m - speed_mps * speed_mps / (2 * comfort_decel_mps2) >= stop_margin_m
about pedestrian_clearance "Anywhere: on Indian roads pedestrians cross away from crossings."
rule junction_clear "Do not stand in a junction" law heuristic:
    not in_junction or t < junction_stand_s or once[0,junction_stand_s] (speed_mps >= 0.5) or lead_gap_m <= queue_gap_m
'''

GENERIC = '''# Falcon's generic traffic rules (falcon.rules/1). Right-hand traffic.
ruleset generic "Generic (right-hand traffic)"
handedness RHT
about "Common rules of right-hand traffic for Falcon's recorded state; not any one country's law."
about "Values marked heuristic are Falcon's choices."

let urban_limit_kmh = 50        # heuristic: a common urban default where the map records no limit
let speed_tolerance_kmh = 0     # heuristic
let follow_time_s = 2           # heuristic: the two-second rule
let comfort_decel_mps2 = 4      # heuristic
let stop_margin_m = 2           # heuristic
let red_line_m = 1.5            # heuristic
let lane_change_s = 3           # heuristic
let junction_stand_s = 5        # heuristic
let queue_gap_m = 10            # heuristic
let overtake_window_s = 10      # heuristic

rule speed_limit "Keep within the speed limit" law heuristic:
    speed_kmh <= default(speed_limit_kmh, urban_limit_kmh) + speed_tolerance_kmh
rule stop_at_red "Stop at a red light" law:
    not (tl_red and tl_distance_m <= red_line_m) or speed_mps <= 0.5
rule no_wrong_way "Do not drive against the lane's direction" law:
    in_junction or not on_lane or lane_heading_err_deg <= 135
rule keep_right "Keep to the right-most lane except to overtake" law heuristic:
    in_junction or not on_lane or lanes_right < 0.5
        or once[0,overtake_window_s] passing or eventually[0,overtake_window_s] passing
rule overtake_left "Overtake on the left" law:
    not pass_on_right
rule lane_discipline "Do not straddle lane lines beyond a lane change" law heuristic:
    in_junction or not on_lane or t < lane_change_s or once[0,lane_change_s] (lane_margin_m >= 0)
rule solid_line "Do not cross a solid line" law:
    in_junction or solid_margin_m >= 0
rule following_distance "Keep a safe distance from the vehicle ahead" heuristic:
    speed_mps < 1 or lead_gap_m >= follow_time_s * speed_mps
rule yield_crossing "Yield to pedestrians on a crossing" law:
    not ped_on_crossing or speed_mps <= 0.5
        or ped_path_gap_m - speed_mps * speed_mps / (2 * comfort_decel_mps2) >= stop_margin_m
rule junction_clear "Do not stand in a junction" law heuristic:
    not in_junction or t < junction_stand_s or once[0,junction_stand_s] (speed_mps >= 0.5) or lead_gap_m <= queue_gap_m
'''
BUILTIN = {'india': INDIA, 'generic': GENERIC}


class RulesError(ValueError):
    pass


# ================================================================================================ rule sets
def parse_ruleset(text: str, source: str = '') -> dict:
    """A rule file's text → {'id', 'name', 'handedness', 'about', 'consts', 'notes', 'rules': [{'id', 'title',
    'tags', 'formula', 'node', 'about', 'signals'}], 'sha256', 'source'}. Raises RulesError with the line."""
    from .provenance import sha256_bytes
    rs = {'schema': SET_SCHEMA, 'id': '', 'name': '', 'handedness': None, 'about': '', 'consts': {}, 'notes': {},
          'rules': [], 'source': source, 'sha256': sha256_bytes(text.encode('utf-8'))}
    lines = text.splitlines()
    pending = None                                       # (line number, id, title, tags, formula text)
    abouts = {}

    def close():
        nonlocal pending
        if pending is None:
            return
        ln, rid, title, tags, ftext = pending
        try:
            node = S.parse(ftext, rs['consts'])
        except S.STLError as e:
            raise RulesError(f'{source or "rules"}: line {ln}: rule {rid}: {e}') from None
        unknown = sorted(S.names(node) - set(SIGNALS))
        if unknown:
            raise RulesError(f'{source or "rules"}: line {ln}: rule {rid}: unknown signal(s) {", ".join(unknown)} '
                             f'(the signals are: {", ".join(SIGNALS)})')
        rs['rules'].append({'id': rid, 'title': title, 'tags': tags, 'formula': ' '.join(ftext.split()),
                            'node': node, 'about': '', 'signals': sorted(S.names(node))})
        pending = None

    for i, raw in enumerate(lines, 1):
        line = raw.rstrip()
        body = line.split('#', 1)[0].rstrip() if not line.lstrip().startswith(('about', 'ruleset')) else line
        if not body.strip():
            continue
        if raw[:1] in (' ', '\t'):
            if pending is None:
                raise RulesError(f'{source or "rules"}: line {i}: an indented line belongs to a rule\'s formula')
            pending = pending[:4] + (pending[4] + ' ' + line.split('#', 1)[0].strip(),)
            continue
        close()
        word = body.split(None, 1)[0]
        rest = body[len(word):].strip()
        if word == 'ruleset':
            m = re.fullmatch(r'([A-Za-z_][\w-]*)\s+"([^"]*)"', rest)
            if not m:
                raise RulesError(f'{source or "rules"}: line {i}: ruleset ID "Name"')
            rs['id'], rs['name'] = m.group(1), m.group(2)
        elif word == 'handedness':
            if rest not in ('LHT', 'RHT'):
                raise RulesError(f'{source or "rules"}: line {i}: handedness LHT or RHT')
            rs['handedness'] = rest
        elif word == 'let':
            m = re.fullmatch(r'([A-Za-z_]\w*)\s*=\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)', rest)
            if not m:
                raise RulesError(f'{source or "rules"}: line {i}: let NAME = NUMBER')
            if m.group(1) in SIGNALS or m.group(1) in S.KEYWORDS or m.group(1) in S.FUNCTIONS:
                raise RulesError(f'{source or "rules"}: line {i}: {m.group(1)} is a signal or a keyword')
            rs['consts'][m.group(1)] = float(m.group(2))
            note = line.split('#', 1)[1].strip() if '#' in line else ''
            if note:
                rs['notes'][m.group(1)] = note
        elif word == 'rule':
            m = re.fullmatch(r'([A-Za-z_][\w-]*)\s+"([^"]*)"\s*([A-Za-z_ ]*):(.*)', rest)
            if not m:
                raise RulesError(f'{source or "rules"}: line {i}: rule ID "Title" [tags]: FORMULA')
            rid = m.group(1)
            if any(r['id'] == rid for r in rs['rules']):
                raise RulesError(f'{source or "rules"}: line {i}: rule {rid} is defined twice')
            pending = (i, rid, m.group(2), m.group(3).split(), m.group(4).strip())
        elif word == 'about':
            m = re.fullmatch(r'(?:([A-Za-z_][\w-]*)\s+)?"([^"]*)"\s*(?:#.*)?', rest)
            if not m:
                raise RulesError(f'{source or "rules"}: line {i}: about "text" or about RULE "text"')
            if m.group(1):                           # repeated about lines continue the text
                abouts[m.group(1)] = (abouts.get(m.group(1), '') + ' ' + m.group(2)).strip()
            else:
                rs['about'] = (rs['about'] + ' ' + m.group(2)).strip()
        else:
            raise RulesError(f'{source or "rules"}: line {i}: unknown statement {word!r} (ruleset, handedness, '
                             f'about, let, rule)')
    close()
    for r in rs['rules']:
        r['about'] = abouts.pop(r['id'], '')
    if abouts:
        raise RulesError(f'{source or "rules"}: about for unknown rule(s): {", ".join(abouts)}')
    if not rs['rules']:
        raise RulesError(f'{source or "rules"}: no rules')
    rs['id'] = rs['id'] or os.path.splitext(os.path.basename(source))[0] or 'rules'
    rs['name'] = rs['name'] or rs['id']
    return rs


def load_ruleset(spec: str | dict | None, handedness: str | None = None) -> dict:
    """'india', 'generic', a rule file's path, a parsed set (returned as is), or None: the set for the handedness
    (LHT → india, else generic)."""
    if isinstance(spec, dict):
        return spec
    if not spec:
        spec = 'india' if (handedness or '').upper() == 'LHT' else 'generic'
    if spec in BUILTIN:
        return parse_ruleset(BUILTIN[spec], f'builtin:{spec}')
    if not os.path.isfile(spec):
        raise RulesError(f'no rule set {spec!r}: india, generic or a rule file')
    with open(spec, encoding='utf-8') as fh:
        return parse_ruleset(fh.read(), os.path.abspath(spec))


def ruleset_doc(rs: dict) -> dict:
    """A rule set without its parsed trees (for JSON)."""
    return {**{k: rs[k] for k in ('id', 'name', 'handedness', 'about', 'consts', 'notes', 'source', 'sha256')},
            'rules': [{k: r[k] for k in ('id', 'title', 'tags', 'formula', 'about', 'signals')} for r in rs['rules']]}


# ================================================================================================ the map
def _xodr_speed(el) -> float | None:
    """A <speed> element's max in km/h (OpenDRIVE's unit attribute; m/s when absent)."""
    try:
        v = float(el.get('max'))
    except (TypeError, ValueError):
        return None
    unit = (el.get('unit') or 'm/s').strip().lower()
    return v * {'m/s': 3.6, 'km/h': 1.0, 'mph': 1.609344}.get(unit, 3.6)


def _cubic(entries, s):
    best = None
    for e in entries:
        if e[0] <= s + 1e-9:
            best = e
    if best is None:
        return 0.0
    u = s - best[0]
    return best[1] + best[2] * u + best[3] * u * u + best[4] * u ** 3


def _road_pose(road, s):
    """(x, y, heading) of a parsed road's reference line at s (OpenDRIVE's frame)."""
    from . import xodr as X
    geoms = [g for g in road.geometries if g.length > 0]
    if not geoms:
        return None
    g = geoms[0]
    for c in geoms:
        if c.s <= s + 1e-9:
            g = c
    x, y, h = X.eval_geometry(g, np.array([s - g.s]))
    return float(x[0]), float(y[0]), float(h[0])


class RoadMap:
    """What the checker reads from the map: driving-lane points (x, y, yaw°, width, road, section, lane, s, left and
    right marking, junction 0/1/NaN, speed km/h or NaN, CARLA's frame) and crosswalk polygons; with `sources` saying
    where each came from."""

    def __init__(self):
        self.P = np.zeros((0, 2))
        self.yaw = self.width = self.s = np.zeros(0)
        self.key = []
        self.left = self.right = []
        self.junction = self.speed = np.zeros(0)
        self.crosswalks = None                     # None: unknown; [] none in the map
        self.step = 2.0
        self.sources = {}
        self.road_junction = {}                    # road id → bool (OpenDRIVE)
        self.road_speed = {}                       # road id → [(s, km/h)] (OpenDRIVE road type)
        self.lane_speed = {}                       # (road, section, lane) → [(sOffset, km/h)]
        self.road_rule = {}

    @property
    def has_lanes(self) -> bool:
        return len(self.P) > 0

    # -- sources
    def add_lanes_json(self, doc: dict):
        cols = doc['columns']
        g = {k: cols.index(k) for k in cols}
        A = doc.get('points') or []
        self.P = np.array([[r[g['x']], r[g['y']]] for r in A], float).reshape(-1, 2)
        self.yaw = np.array([r[g['yaw']] for r in A], float)
        self.width = np.array([r[g['width']] for r in A], float)
        self.s = np.array([r[g['s']] for r in A], float) if 's' in g else np.zeros(len(A))
        self.key = [(int(r[g['road']]), int(r[g['section']]), int(r[g['lane']])) for r in A]
        self.left = [str(r[g['left_type']]) if 'left_type' in g else '' for r in A]
        self.right = [str(r[g['right_type']]) if 'right_type' in g else '' for r in A]
        self.step = float(doc.get('step_m', 2.0))
        self.junction = np.full(len(A), np.nan)
        self.speed = np.full(len(A), np.nan)
        self.sources['lanes'] = 'lanes.json (CARLA\'s waypoints)'
        self.sources['markings'] = 'lanes.json (CARLA\'s lane markings)'

    def add_xodr(self, text: str, name: str = 'the OpenDRIVE file', make_lanes: bool = False, step: float = 2.0):
        """Junction roads, speed records and crosswalks from an OpenDRIVE document; lane points too when
        make_lanes (no lanes.json)."""
        import xml.etree.ElementTree as ET
        from . import xodr as X
        root = ET.fromstring(text)
        od = X.parse(text)
        by_id = {r.id: r for r in od.roads}
        cws = []
        for rel in root.findall('road'):
            rid = rel.get('id', '')
            try:
                ri = int(rid)
            except ValueError:
                continue
            self.road_junction[ri] = rel.get('junction', '-1') not in ('-1', '', None)
            self.road_rule[ri] = (rel.get('rule') or 'RHT').upper()
            sp = []
            for ty in rel.findall('type'):
                spd = ty.find('speed')
                v = _xodr_speed(spd) if spd is not None else None
                if v is not None:
                    sp.append((float(ty.get('s', 0.0)), v))
            if sp:
                self.road_speed[ri] = sorted(sp)
            lanes = rel.find('lanes')
            for k, sec in enumerate(lanes.findall('laneSection') if lanes is not None else ()):
                for side in ('left', 'right'):
                    se = sec.find(side)
                    for ln in se.findall('lane') if se is not None else ():
                        ls = [(float(e.get('sOffset', 0.0)), _xodr_speed(e)) for e in ln.findall('speed')]
                        ls = [x for x in ls if x[1] is not None]
                        if ls:
                            self.lane_speed[(ri, k, int(ln.get('id')))] = sorted(ls)
            objs = rel.find('objects')
            road = by_id.get(rid)
            for ob in objs.findall('object') if objs is not None and road is not None else ():
                if (ob.get('type') or '').lower() != 'crosswalk':
                    continue
                pose = _road_pose(road, float(ob.get('s', 0.0)))
                if pose is None:
                    continue
                x0, y0, h = pose
                t = float(ob.get('t', 0.0))
                cx, cy = x0 - math.sin(h) * t, y0 + math.cos(h) * t
                th = h + float(ob.get('hdg', 0.0))
                corners = [(float(c.get('u', 0.0)), float(c.get('v', 0.0))) for c in ob.iter('cornerLocal')]
                if len(corners) < 3:
                    L, W = float(ob.get('length', 0.0) or 0.0), float(ob.get('width', 0.0) or 0.0)
                    corners = [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]
                poly = [(cx + u * math.cos(th) - v * math.sin(th), -(cy + u * math.sin(th) + v * math.cos(th)))
                        for u, v in corners]
                if len(poly) >= 3:
                    cws.append(np.array(poly, float))
        self.crosswalks = cws
        self.sources['junction'] = f'{name} (junction roads)'
        self.sources['speed'] = f'{name} (lane and road-type <speed> records)'
        self.sources['crosswalks'] = f'{name} ({len(cws)} crosswalk objects)'
        if make_lanes:
            self._lanes_from_od(od, step)
            self.sources['lanes'] = f'{name} (lane centres from the reference line, laneOffset and widths)'
            self.sources['markings'] = 'unknown (lanes made from OpenDRIVE: road marks not read)'
        self._apply_road_info()

    def _lanes_from_od(self, od, step: float):
        from . import xodr as X
        rows = []
        for road in od.roads:
            try:
                ri = int(road.id)
            except ValueError:
                continue
            if not road.sections:
                continue
            for g in road.geometries:
                if g.length <= 0:
                    continue
                n = max(1, math.ceil(g.length / step))
                ds = np.linspace(0.0, g.length, n + 1)[:-1] if g.length > step / 2 else np.array([0.0])
                xs, ys, hs = X.eval_geometry(g, ds)
                for x, y, h, s in zip(xs, ys, hs, g.s + ds):
                    k = 0
                    for j, c in enumerate(road.sections):
                        if c.s <= s + 1e-9:
                            k = j
                    sec = road.sections[k]
                    ls = s - sec.s
                    base = _cubic(road.lane_offsets, s)
                    nx, ny = -math.sin(h), math.cos(h)
                    for side, lanes, sign in (('right', sec.right, -1.0),
                                              ('left', sorted(sec.left, key=lambda q: q.id), 1.0)):
                        t = base
                        for ln in lanes:
                            w = _cubic(ln.widths, ls)
                            c = t + sign * w / 2.0
                            t += sign * w
                            if ln.type != 'driving' or w <= 0.05:
                                continue
                            along = (side == 'right') != road.lht
                            hd = h if along else h + math.pi
                            rows.append((x + nx * c, -(y + ny * c), (-math.degrees(hd) + 180.0) % 360.0 - 180.0, w,
                                         ri, k, ln.id, s))
        a = np.array([r[:4] for r in rows], float).reshape(-1, 4)
        self.P, self.yaw, self.width = a[:, :2], a[:, 2], a[:, 3]
        self.key = [(r[4], r[5], r[6]) for r in rows]
        self.s = np.array([r[7] for r in rows], float)
        self.left = self.right = [''] * len(rows)
        self.step = step
        self.junction = np.full(len(rows), np.nan)
        self.speed = np.full(len(rows), np.nan)

    def _apply_road_info(self):
        if not self.has_lanes:
            return
        J, V = self.junction.copy(), self.speed.copy()
        for i, (road, sec, lane) in enumerate(self.key):
            if road in self.road_junction:
                J[i] = 1.0 if self.road_junction[road] else 0.0
            if self.road_junction.get(road):
                continue                                # a junction road's speed is not a limit (SUMO: curvature)
            ls = self.lane_speed.get((road, sec, lane))
            v = None
            if ls:
                v = ls[0][1]
                for so, x in ls:
                    v = x if so <= self.s[i] + 1e-9 else v
            elif road in self.road_speed:
                for so, x in self.road_speed[road]:
                    if so <= self.s[i] + 1e-9 or v is None:
                        v = x
            if v is not None:
                V[i] = v
        self.junction, self.speed = J, V

    def junction_heuristic(self):
        """Without OpenDRIVE: a road is a junction road when its lane points overlap lane points of two or more other
        roads at headings 20°–160° apart (crossing or turning lanes)."""
        n = len(self.P)
        if not n:
            return
        roads = np.array([k[0] for k in self.key])
        other = {}
        cell = 4.0
        grid = {}
        for i, (x, y) in enumerate(self.P):
            grid.setdefault((int(x // cell), int(y // cell)), []).append(i)
        for i, (x, y) in enumerate(self.P):
            gx, gy = int(x // cell), int(y // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((gx + dx, gy + dy), ()):
                        if roads[j] == roads[i]:
                            continue
                        if math.hypot(self.P[j, 0] - x, self.P[j, 1] - y) > 0.5 * max(self.width[i], 1.0):
                            continue
                        dyaw = abs((self.yaw[j] - self.yaw[i] + 180.0) % 360.0 - 180.0)
                        if 20.0 <= dyaw <= 160.0:
                            other.setdefault(int(roads[i]), set()).add(int(roads[j]))
        jr = {r for r, s in other.items() if len(s) >= 2}
        self.junction = np.array([1.0 if r in jr else 0.0 for r in roads])
        self.sources['junction'] = ('heuristic from lanes.json: a road whose lanes overlap two other roads\' at '
                                    'headings 20°–160° apart (no OpenDRIVE file)')

    def crop(self, xy: np.ndarray, margin: float = 60.0):
        """Keep the lane points within margin of the trajectory's bounding box (speed)."""
        if not self.has_lanes or not len(xy):
            return
        lo, hi = xy.min(0) - margin, xy.max(0) + margin
        k = np.flatnonzero((self.P[:, 0] >= lo[0]) & (self.P[:, 0] <= hi[0]) & (self.P[:, 1] >= lo[1])
                           & (self.P[:, 1] <= hi[1]))
        self.P, self.yaw, self.width, self.s = self.P[k], self.yaw[k], self.width[k], self.s[k]
        self.key = [self.key[i] for i in k]
        self.left = [self.left[i] for i in k] if self.left else self.left
        self.right = [self.right[i] for i in k] if self.right else self.right
        self.junction, self.speed = self.junction[k], self.speed[k]
        r = np.radians(self.yaw)
        self.fwd = np.stack([np.cos(r), np.sin(r)], 1)
        self.rgt = np.stack([-np.sin(r), np.cos(r)], 1)

    def locate(self, x: float, y: float, yaw: float):
        """(index of the lane point whose stretch holds (x, y) — the one nearest the heading where lanes overlap —,
        lateral offset + right) or (None, None)."""
        if not self.has_lanes:
            return None, None
        d = np.array([x, y]) - self.P
        lon = np.einsum('ij,ij->i', d, self.fwd)
        lat = np.einsum('ij,ij->i', d, self.rgt)
        ok = (np.abs(lat) <= self.width / 2.0) & (np.abs(lon) <= 0.75 * self.step)
        if not ok.any():
            return None, None
        k = np.flatnonzero(ok)
        dyaw = np.abs((self.yaw[k] - yaw + 180.0) % 360.0 - 180.0)
        cost = np.hypot(lon[k], lat[k]) + 1000.0 * (dyaw > 45.0) + dyaw / 90.0
        j = int(k[np.argmin(cost)])
        return j, float(lat[j])

    def beside(self, j: int) -> tuple:
        """(same-direction lanes on the left, on the right) of lane point j's cross-section."""
        d = self.P - self.P[j]
        lon = np.einsum('ij,j->i', d, self.fwd[j])
        off = np.einsum('ij,j->i', d, self.rgt[j])
        dyaw = np.abs((self.yaw - self.yaw[j] + 180.0) % 360.0 - 180.0)
        w = max(float(self.width[j]), 1.0)
        ok = (np.abs(lon) <= self.step) & (dyaw <= SAME_DIR_DEG) & (np.abs(off) <= 6.5 * w)
        idx = {int(round(o / w)) for o in off[ok]}
        idx.discard(0)
        return sum(1 for i in idx if i < 0), sum(1 for i in idx if i > 0)


def _point_in(poly: np.ndarray, x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
    return inside


def find_xodr(cfg: dict, carla_root: str = '') -> tuple:
    """(path, why) of the map's OpenDRIVE file for a run's config: its own (an imported world), or the CARLA town's
    under the CARLA folder (Settings' carla_root, or the usual places); (None, why not)."""
    p = (cfg or {}).get('xodr')
    if p and os.path.isfile(p):
        return p, 'the world\'s OpenDRIVE file'
    town = (cfg or {}).get('town') or ''
    if not town:
        return None, 'no town and no OpenDRIVE file in the run\'s config'
    roots = [carla_root] if carla_root else []
    if not roots:
        try:
            from .settings import Settings
            r = Settings().get('carla_root')
            if r:
                roots.append(r)
        except Exception:            # noqa: BLE001 (no settings: look in the usual places)
            pass
        try:
            from ..sim.server import candidate_roots
            roots += candidate_roots()
        except Exception:            # noqa: BLE001
            pass
    name = os.path.basename(town)
    for r in roots:
        f = os.path.join(r, 'CarlaUE4', 'Content', 'Carla', 'Maps', 'OpenDrive', f'{name}.xodr')
        if os.path.isfile(f):
            return f, f'CARLA\'s {name}.xodr'
    return None, f'{name}.xodr not found under a CARLA folder (set it in Settings, or pass --xodr)'


# ================================================================================================ traces
def _jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _run_config(run_dir: str) -> dict:
    for f in ('drive.json', 'stack.json', 'live.json', 'replay.json', 'config.json'):
        p = os.path.join(run_dir, f)
        if os.path.isfile(p):
            try:
                with open(p, encoding='utf-8') as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
    p = os.path.join(run_dir, 'run.json')
    if os.path.isfile(p):
        with open(p, encoding='utf-8') as fh:
            return ((json.load(fh).get('provenance') or {}).get('settings')) or {}
    return {}


def _box_center(a: dict):
    r = math.radians(a.get('yaw', 0.0))
    c = a.get('bbox_center') or [0.0, 0.0, 0.0]
    return (a['x'] + c[0] * math.cos(r) - c[1] * math.sin(r), a['y'] + c[0] * math.sin(r) + c[1] * math.cos(r))


def _kind(a: dict) -> str:
    c = str(a.get('category') or a.get('type_id') or '')
    if c.startswith(('human', 'walker', 'pedestrian', 'child')):
        return 'pedestrian'
    return 'vehicle'


class Trace:
    """The checker's input: sample times, the ego's poses and speeds, the other actors per sample (or None), the map,
    precomputed signals (gym traces), the ego's size, sources and notes."""

    def __init__(self, t, x, y, yaw, speed, ego_size=EGO_DEFAULT):
        self.t = np.asarray(t, float)
        self.x, self.y, self.yaw = np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float)
        self.speed = np.asarray(speed, float)
        self.ego_size = (float(ego_size[0]), float(ego_size[1]))
        self.actors = None                       # per sample [{'id', 'x', 'y', 'yaw', 'vx', 'vy', 'l', 'w', 'kind'}]
        self.map = RoadMap()
        self.given = {}                          # signals the trace carries (gym)
        self.sources = {}
        self.notes = []
        self.handedness = None
        self.kind = 'run'

    @property
    def distance_m(self) -> float:
        if len(self.x) < 2:
            return 0.0
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))


XODR_AGREE_M, XODR_AGREE_SHARE = 0.25, 0.9


def _xodr_agrees(text: str, lanes: RoadMap, n: int = 300) -> float:
    """The share of (up to n, evenly spaced) lanes.json points that have a lane centre made from the OpenDRIVE file
    with the same heading (±10°) within XODR_AGREE_M laterally and a step along: whether the file is the run's map
    (measured on Town10HD_Opt: lateral differences median 0.4 mm, 95th percentile 3 cm)."""
    m = RoadMap()
    try:
        m.add_xodr(text, make_lanes=True)
    except Exception:              # noqa: BLE001 (an unreadable file does not match)
        return 0.0
    if not m.has_lanes or not lanes.has_lanes:
        return 0.0
    r = np.radians(m.yaw)
    fwd, rgt = np.stack([np.cos(r), np.sin(r)], 1), np.stack([-np.sin(r), np.cos(r)], 1)
    idx = np.linspace(0, len(lanes.P) - 1, min(n, len(lanes.P))).astype(int)
    ok = 0
    for i in idx:
        d = lanes.P[i] - m.P
        lon, lat = np.einsum('ij,ij->i', d, fwd), np.einsum('ij,ij->i', d, rgt)
        dyaw = np.abs((m.yaw - lanes.yaw[i] + 180.0) % 360.0 - 180.0)
        ok += bool(np.any((np.abs(lat) <= XODR_AGREE_M) & (np.abs(lon) <= m.step) & (dyaw <= 10.0)))
    return ok / len(idx)


def _positional_speed(tr: Trace) -> np.ndarray:
    """The ego's speed from its positions (central differences; one-sided at the ends)."""
    if len(tr.t) < 2:
        return np.zeros(len(tr.t))
    return np.hypot(np.gradient(tr.x, tr.t), np.gradient(tr.y, tr.t))


def trace_from_run(run_dir: str, xodr: str | None = None, carla_root: str = '', use_map: bool = True) -> Trace:
    """A recorded run's trace: frames.jsonl, actors.jsonl (or trajectories.json's scenario actors), lanes.json and the
    map's OpenDRIVE file (module doc). xodr: None finds the map's file (find_xodr), '' uses none, a path that file."""
    fp = os.path.join(run_dir, 'frames.jsonl')
    if not os.path.isfile(fp):
        raise RulesError(f'{run_dir}: no frames.jsonl (not a recorded run)')
    frames = [f for f in _jsonl(fp) if f.get('ego')]
    if len(frames) < 2:
        raise RulesError(f'{run_dir}: fewer than two frames with the ego\'s pose')
    cfg = _run_config(run_dir)
    ap = os.path.join(run_dir, 'actors.jsonl')
    rows_by_step = None
    ego_size = None
    ego_id = None
    src_actors = 'none recorded'
    if os.path.isfile(ap):
        rows_by_step = {}
        for r in _jsonl(ap):
            rows_by_step[r['step']] = r['actors']
            for a in r['actors']:
                if a.get('role') == 'hero':
                    ego_id = a['id']
                    if ego_size is None and a.get('extent'):
                        ego_size = (2.0 * a['extent'][0], 2.0 * a['extent'][1])
        frames = [f for f in frames if f['step'] in rows_by_step] or frames
        src_actors = 'actors.jsonl'
    t0 = float(frames[0].get('sim_t') or 0.0)
    t = np.array([float(f.get('sim_t', f['step'] * 0.05)) - t0 for f in frames])
    keep = np.concatenate([[True], np.diff(t) > 0])
    frames = [f for f, k in zip(frames, keep) if k]
    t = t[keep]
    tr = Trace(t, [f['ego']['x'] for f in frames], [f['ego']['y'] for f in frames],
               [f['ego'].get('yaw', 0.0) for f in frames], [f.get('speed_mps', np.nan) for f in frames],
               ego_size or EGO_DEFAULT)
    tr.kind = 'run'
    tr.handedness = (cfg.get('handedness') or ((cfg.get('scenario') or {}).get('handedness'))) or None
    tr.sources['ego'] = 'frames.jsonl'
    vpos = _positional_speed(tr)
    moving = vpos > 1.0
    if np.isnan(tr.speed).all() or (moving.sum() >= 3 and np.nanmedian(np.abs(tr.speed[moving] - vpos[moving])) > 1.0):
        tr.speed = vpos
        tr.sources['ego'] = 'frames.jsonl poses; speed from the positions'
        tr.notes.append('frames.jsonl speed_mps disagrees with the ego\'s positions (a kinematic scenario recorded '
                        'before Falcon 0.6 logged 0): the speed is the positions\' central difference')
    if ego_size is None:
        tr.notes.append(f'the ego\'s size is not recorded: {EGO_DEFAULT[0]} × {EGO_DEFAULT[1]} m assumed (CARLA\'s '
                        f'mkz_2020)')
    if rows_by_step is not None:
        acts = []
        for f in frames:
            out = []
            for a in rows_by_step.get(f['step'], ()):
                if a['id'] == ego_id or a.get('role') == 'hero':
                    continue
                cx, cy = _box_center(a)
                ext = a.get('extent') or [0.5, 0.5, 0.5]
                out.append({'id': a['id'], 'x': cx, 'y': cy, 'yaw': a.get('yaw', 0.0), 'vx': a.get('vx', 0.0),
                            'vy': a.get('vy', 0.0), 'l': 2.0 * ext[0], 'w': 2.0 * ext[1], 'kind': _kind(a)})
            acts.append(out)
        tr.actors = acts
    else:
        tj = os.path.join(run_dir, 'trajectories.json')
        if os.path.isfile(tj):
            with open(tj, encoding='utf-8') as fh:
                doc = json.load(fh)
            kinds = {a.get('id'): ('pedestrian' if a.get('kind') in ('pedestrian', 'child') else 'vehicle')
                     for a in ((cfg.get('scenario') or {}).get('actors') or ())}
            cols = doc.get('columns') or ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'length', 'width']
            ci = {c: i for i, c in enumerate(cols)}
            series = {}
            for name, rows in (doc.get('actors') or {}).items():
                if name == 'ego':
                    if rows:
                        tr.ego_size = (float(rows[0][ci['length']]), float(rows[0][ci['width']]))
                    continue
                series[name] = {round(float(r[ci['t']]), 4): r for r in rows}
            acts = []
            for f in frames:
                ts = round(float(f.get('sim_t') or 0.0), 4)
                out = []
                for name, by in series.items():
                    r = by.get(ts)
                    if r is None:
                        continue
                    out.append({'id': name, 'x': r[ci['x']], 'y': r[ci['y']], 'yaw': r[ci['yaw']], 'vx': r[ci['vx']],
                                'vy': r[ci['vy']], 'l': r[ci['length']], 'w': r[ci['width']],
                                'kind': kinds.get(name, 'vehicle')})
                acts.append(out)
            tr.actors = acts
            src_actors = 'trajectories.json (the scenario\'s actors only; no background traffic)'
    tr.sources['actors'] = src_actors
    if tr.actors is None:
        tr.notes.append('no other actors recorded: following distance, pedestrians and passing are not evaluable')
    # recorded light states and limits, when a run carries them
    if any('traffic_light' in f for f in frames):
        tr.given['tl_red'] = np.array([(1.0 if str((f.get('traffic_light') or {}).get('state')) == 'Red' else 0.0)
                                       if 'traffic_light' in f else np.nan for f in frames])
        tr.given['tl_distance_m'] = np.array([float((f.get('traffic_light') or {}).get('distance_m', np.inf))
                                              if 'traffic_light' in f else np.nan for f in frames])
        tr.sources['traffic_lights'] = 'frames.jsonl traffic_light'
    else:
        tr.sources['traffic_lights'] = 'not recorded in this run'
    if any(f.get('speed_limit_kmh') is not None for f in frames):
        tr.given['carla_speed_limit_kmh'] = np.array([f.get('speed_limit_kmh') if f.get('speed_limit_kmh')
                                                      is not None else np.nan for f in frames], float)
    # the map
    if use_map:
        lp = os.path.join(run_dir, 'lanes.json')
        if os.path.isfile(lp):
            with open(lp, encoding='utf-8') as fh:
                tr.map.add_lanes_json(json.load(fh))
        if xodr is None:
            xp, why = find_xodr(cfg, carla_root)
        else:
            xp, why = (xodr or None), ('given' if xodr else 'not used (asked)')
        if xp:
            with open(xp, encoding='utf-8') as fh:
                text = fh.read()
            agree = _xodr_agrees(text, tr.map) if tr.map.has_lanes else None
            if agree is not None and agree < XODR_AGREE_SHARE:
                tr.notes.append(f'{why} ({xp}) does not match the run\'s lanes.json ({100 * agree:.0f} % of its lane '
                                f'points within {XODR_AGREE_M} m laterally; another map or version?): not used')
                xp = None
        if xp:
            name = why if os.path.basename(xp) in why else f'{why} ({os.path.basename(xp)})'
            tr.map.add_xodr(text, name, make_lanes=not tr.map.has_lanes)
            tr.sources['xodr'] = xp
        else:
            tr.notes.append(f'map OpenDRIVE file: {why}')
            if tr.map.has_lanes:
                tr.map.junction_heuristic()
        if not tr.map.has_lanes:
            tr.notes.append('no lanes (no lanes.json and no OpenDRIVE file): lane rules are not evaluable')
        tr.sources.update(tr.map.sources)
    return tr


TRACE_COLUMNS = ('t', 'x', 'y', 'yaw', 'speed_mps', 'on_lane', 'in_junction', 'lane_yaw', 'lane_lat_m', 'lane_width_m',
                 'lanes_left', 'lanes_right', 'left_marking', 'right_marking', 'lead_gap_m', 'vru_gap_m', 'tl_red',
                 'tl_distance_m', 'speed_limit_kmh')


def trace_from_episode(e: dict) -> Trace:
    """A gym or Leaderboard episode's trace (the gym worker's per-step 'trace': TRACE_COLUMNS)."""
    tr_doc = e.get('trace')
    if not tr_doc or not tr_doc.get('rows'):
        raise RulesError('the episode carries no trace (recorded before Falcon 1.1, or log_trace off)')
    cols = tr_doc['columns']
    ci = {c: i for i, c in enumerate(cols)}
    R = tr_doc['rows']

    def col(name, conv=float):
        if name not in ci:
            return np.full(len(R), np.nan)
        return np.array([np.nan if r[ci[name]] is None else conv(r[ci[name]]) for r in R], float)
    size = tr_doc.get('ego') or {}
    tr = Trace(col('t'), col('x'), col('y'), col('yaw'), col('speed_mps'),
               (size.get('length', EGO_DEFAULT[0]), size.get('width', EGO_DEFAULT[1])))
    tr.kind = 'episode'
    tr.handedness = tr_doc.get('handedness')
    on = col('on_lane')
    junction = col('in_junction')
    lane_yaw = col('lane_yaw')
    lat, width = col('lane_lat_m'), col('lane_width_m')
    hw = tr.ego_size[1] / 2.0
    err = np.abs((tr.yaw - lane_yaw + 180.0) % 360.0 - 180.0)
    g = tr.given
    g['on_lane'] = on
    g['in_junction'] = junction
    g['lane_heading_err_deg'] = np.where(on > 0.5, err, np.nan)
    g['lanes_left'] = np.where(on > 0.5, col('lanes_left'), np.nan)
    g['lanes_right'] = np.where(on > 0.5, col('lanes_right'), np.nan)
    g['lane_margin_m'] = np.where(on > 0.5, width / 2.0 - np.abs(lat) - hw, np.nan)
    if 'left_marking' in ci:
        lm = [r[ci['left_marking']] for r in R]
        rm = [r[ci['right_marking']] for r in R]
        g['solid_margin_m'] = np.array([_solid_margin(a, b, la, w, hw) if o > 0.5 else np.nan
                                        for a, b, la, w, o in zip(lm, rm, lat, width, on)], float)
    lead = col('lead_gap_m')
    g['lead_gap_m'] = np.where(np.isnan(lead), np.inf, lead)
    vru = col('vru_gap_m')
    g['ped_path_gap_m'] = np.where(np.isnan(vru), np.inf, vru)
    g['tl_red'] = col('tl_red')
    g['tl_distance_m'] = col('tl_distance_m')
    g['carla_speed_limit_kmh'] = col('speed_limit_kmh')
    tr.sources.update({'ego': 'the gym trace', 'lanes': 'CARLA\'s map at each step (the gym worker)',
                       'junction': 'CARLA Waypoint.is_junction', 'markings': 'CARLA\'s lane markings',
                       'actors': 'the environment\'s lead and vulnerable-road-user features (no per-actor poses)',
                       'traffic_lights': 'the environment\'s light on the route ahead',
                       'pedestrians': 'the vulnerable road user in the path (pedestrian or cyclist)'})
    tr.notes.append('a gym trace carries no other actors\' poses: passing events and crosswalks are not evaluable')
    return tr


def _solid_margin(left, right, lat, width, hw) -> float:
    if left in (None, '') and right in (None, ''):
        return np.nan
    m = np.inf
    if str(left) in SOLID:
        m = min(m, width / 2.0 + lat - hw)              # the left line is at lateral −width/2
    if str(right) in SOLID:
        m = min(m, width / 2.0 - lat - hw)
    return m


# ================================================================================================ signals
def signals(tr: Trace) -> dict:
    """Every signal of SIGNALS on the trace's samples (module doc)."""
    n = len(tr.t)
    sig = {'t': tr.t.copy(), 'speed_mps': tr.speed.copy(), 'speed_kmh': tr.speed * 3.6}
    if n > 2:
        sig['accel_mps2'] = np.gradient(tr.speed, tr.t)
    else:
        sig['accel_mps2'] = np.full(n, np.nan)
    hl, hw = tr.ego_size[0] / 2.0, tr.ego_size[1] / 2.0
    nan = lambda: np.full(n, np.nan)          # noqa: E731
    for k in ('on_lane', 'in_junction', 'lane_heading_err_deg', 'lanes_left', 'lanes_right', 'lane_margin_m',
              'solid_margin_m', 'lead_gap_m', 'ped_path_gap_m', 'ped_on_crossing', 'pass_on_left', 'pass_on_right',
              'tl_red', 'tl_distance_m', 'speed_limit_kmh'):
        sig[k] = nan()
    m = tr.map
    map_limit = nan()
    if m.has_lanes:
        m.crop(np.column_stack([tr.x, tr.y]))
    if m.has_lanes and 'on_lane' not in tr.given:
        for i in range(n):
            j, lat = m.locate(tr.x[i], tr.y[i], tr.yaw[i])
            if j is None:
                sig['on_lane'][i] = 0.0
                continue
            sig['on_lane'][i] = 1.0
            sig['in_junction'][i] = m.junction[j]
            sig['lane_heading_err_deg'][i] = abs((tr.yaw[i] - m.yaw[j] + 180.0) % 360.0 - 180.0)
            ll, lr = m.beside(j)
            sig['lanes_left'][i], sig['lanes_right'][i] = ll, lr
            sig['lane_margin_m'][i] = m.width[j] / 2.0 - abs(lat) - hw
            if m.left[j] or m.right[j]:
                sig['solid_margin_m'][i] = _solid_margin(m.left[j], m.right[j], lat, m.width[j], hw)
            map_limit[i] = m.speed[j]
    elif m.has_lanes:
        for i in range(n):
            j, _ = m.locate(tr.x[i], tr.y[i], tr.yaw[i])
            if j is not None:
                map_limit[i] = m.speed[j]
    for k, v in tr.given.items():
        if k in sig:
            sig[k] = np.asarray(v, float)
    # the speed limit: CARLA's where recorded, else the map's; in a junction road the last limit before it
    limit = np.asarray(tr.given.get('carla_speed_limit_kmh', nan()), float).copy()
    src = 'CARLA (recorded)' if np.isfinite(limit).any() else None
    fill = np.isnan(limit)
    limit[fill] = map_limit[fill]
    if np.isfinite(map_limit).any():
        src = (src + ' and ' if src else '') + (m.sources.get('speed') or 'the map')
    last = np.nan
    for i in range(n):
        if np.isnan(limit[i]) and sig['in_junction'][i] == 1.0 and not np.isnan(last):
            limit[i] = last
        elif not np.isnan(limit[i]):
            last = limit[i]
    sig['speed_limit_kmh'] = limit
    tr.sources['speed_limit'] = src or 'unknown (neither recorded nor in the map)'
    # actors: lead, pedestrians, crosswalks, passing
    if tr.actors is not None:
        cws = m.crosswalks
        lead, ped, onx = np.full(n, np.inf), np.full(n, np.inf), np.zeros(n) if cws is not None else nan()
        pl, pr = np.zeros(n), np.zeros(n)
        prev = {}
        for i in range(n):
            r = math.radians(tr.yaw[i])
            fx, fy = math.cos(r), math.sin(r)
            rx, ry = -math.sin(r), math.cos(r)
            v_ego = tr.speed[i]
            near_ped = None
            seen = {}
            for a in tr.actors[i]:
                dx, dy = a['x'] - tr.x[i], a['y'] - tr.y[i]
                xr, yr = dx * fx + dy * fy, dx * rx + dy * ry
                dyaw = math.radians(a['yaw'] - tr.yaw[i])
                along = abs(a['l'] / 2.0 * math.cos(dyaw)) + abs(a['w'] / 2.0 * math.sin(dyaw))
                across = abs(a['l'] / 2.0 * math.sin(dyaw)) + abs(a['w'] / 2.0 * math.cos(dyaw))
                hdiff = abs((a['yaw'] - tr.yaw[i] + 180.0) % 360.0 - 180.0)
                if a['kind'] == 'vehicle':
                    if xr > 0 and abs(yr) < hw + across + LEAD_MARGIN_M and hdiff <= LEAD_HEADING_DEG:
                        gap = max(xr - hl - along, 0.0)
                        if gap <= LEAD_RANGE_M:
                            lead[i] = min(lead[i], gap)
                    if hdiff <= PASS_HEADING_DEG:
                        seen[a['id']] = (xr, yr, a['vx'] * fx + a['vy'] * fy, across)
                else:
                    if xr > 0 and abs(yr) < hw + across + PED_MARGIN_M:
                        gap = max(xr - hl - along, 0.0)
                        if gap <= PED_RANGE_M and gap < ped[i]:
                            ped[i], near_ped = gap, a
            if cws is not None and near_ped is not None:
                onx[i] = 1.0 if any(_point_in(p, near_ped['x'], near_ped['y']) for p in cws) else 0.0
            for aid, (xr, yr, va, across) in seen.items():
                p = prev.get(aid)
                if p is not None and p[0] > 0.0 >= xr and hw + across * 0.5 < abs(yr) < PASS_BESIDE_M \
                        and v_ego > va and v_ego > 1.0:
                    if yr < 0:
                        pr[i] = 1.0                   # the vehicle is on the ego's left: the ego is on its right
                    else:
                        pl[i] = 1.0
            prev = seen
        sig['lead_gap_m'], sig['ped_path_gap_m'], sig['ped_on_crossing'] = lead, ped, onx
        sig['pass_on_left'], sig['pass_on_right'] = pl, pr
        if cws is None:
            tr.sources['crosswalks'] = 'unknown (no OpenDRIVE file)'
    for k in ('lead_gap_m', 'ped_path_gap_m'):
        if k in tr.given:
            sig[k] = np.asarray(tr.given[k], float)
    sig['passing'] = np.where(np.isnan(sig['pass_on_left']) | np.isnan(sig['pass_on_right']), np.nan,
                              np.maximum(sig['pass_on_left'], sig['pass_on_right']))
    return sig


# ================================================================================================ checking
def _round(v, nd=4):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), nd) + 0.0


def _runs(mask: np.ndarray, t: np.ndarray, dt: float) -> list:
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append([_round(t[i], 3), _round(t[j] + dt, 3)])
        i = j + 1
    return out


def check(tr: Trace, ruleset: dict) -> dict:
    """Every rule of the set on the trace: robustness, violation intervals, counts per km, evaluated share."""
    sig = signals(tr)
    t = tr.t
    n = len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    km = tr.distance_m / 1000.0
    rows = []
    for r in ruleset['rules']:
        rho = S.evaluate(r['node'], sig, t)
        known = ~np.isnan(rho)
        missing = sorted(k for k in r['signals'] if np.isnan(np.asarray(sig[k], float)).all())
        vio = S.intervals(rho, t, dt)
        kr = rho[known]
        if not known.any():
            state, rob = 'not evaluable', None
        else:
            rob = float(kr.min())
            state = 'violated' if rob < 0 else 'satisfied'
        rows.append({'id': r['id'], 'title': r['title'], 'tags': r['tags'], 'formula': r['formula'],
                     'state': state, 'robustness': _round(rob), 'vacuous': bool(rob is not None and math.isinf(rob)
                                                                             and rob > 0),
                     'evaluated_share': _round(known.mean() if n else 0.0, 4),
                     'violations': [{k: (_round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}
                                    for x in vio],
                     'count': len(vio), 'per_km': (len(vio) / km) if km > 0 else None,
                     'violated_s': _round(sum(x['end'] - x['start'] for x in vio), 3),
                     'unknown': _runs(~known, t, dt), 'missing_signals': missing})
    hand = (tr.handedness or '').upper() or None
    warn = []
    if ruleset.get('handedness') and hand and hand in ('LHT', 'RHT') and hand != ruleset['handedness']:
        warn.append(f'the rule set is for {ruleset["handedness"]} and the run drove on a {hand} map: '
                    f'keep-side and overtaking-side rules read the wrong side')
    return {'schema': SCHEMA, 'kind': tr.kind, 'ruleset': {k: ruleset.get(k) for k in ('id', 'name', 'handedness',
                                                                                        'source', 'sha256')},
            'handedness': hand, 'samples': n, 'duration_s': _round(t[-1] - t[0] + dt if n else 0.0, 3),
            'step_s': _round(dt, 4), 'distance_km': _round(km, 5), 'rules': rows,
            'totals': {'violations': sum(r['count'] for r in rows),
                       'per_km': (sum(r['count'] for r in rows) / km) if km > 0 else None,
                       'violated_rules': sum(1 for r in rows if r['state'] == 'violated'),
                       'not_evaluable': [r['id'] for r in rows if r['state'] == 'not evaluable']},
            'sources': dict(tr.sources), 'notes': list(tr.notes), 'not_recorded': list(NOT_RECORDED),
            'warnings': warn, 'method': 'core.stl quantitative semantics; a rule holds at every sample (G φ); '
                                        'violations are maximal intervals of negative robustness'}


def check_run(run_dir: str, ruleset=None, xodr: str | None = None, carla_root: str = '') -> dict:
    tr = trace_from_run(run_dir, xodr, carla_root)
    rs = load_ruleset(ruleset, tr.handedness)
    out = check(tr, rs)
    out['run'] = os.path.abspath(run_dir)
    return out


def check_episode(e: dict, ruleset=None) -> dict:
    tr = trace_from_episode(e)
    rs = load_ruleset(ruleset, tr.handedness)
    out = check(tr, rs)
    out['episode'], out['seed'] = e.get('episode'), e.get('seed')
    return out


def timeline(result: dict) -> list:
    """[(rule id, title, state, [(start, end, kind)])] with kind 'violation' or 'unknown', for a timeline chart."""
    out = []
    for r in result['rules']:
        segs = [(v['start'], v['end'], 'violation') for v in r['violations']]
        segs += [(a, b, 'unknown') for a, b in r['unknown']]
        out.append((r['id'], r['title'], r['state'], sorted(segs)))
    return out


def aggregate(results: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """Over runs or routes: per rule the violations, per km (a ratio, with a 95 % percentile bootstrap interval over
    the runs), the runs with at least one (with Wilson's interval), the runs where it could be evaluated."""
    from .metrics.uncertainty import bootstrap
    from .stack import wilson
    res = [r for r in results if r]
    if not res:
        return {'runs': 0, 'rules': [], 'distance_km': 0.0, 'violations': 0, 'per_km': None, 'ruleset': None}
    if len({(r.get('ruleset') or {}).get('sha256') for r in res}) > 1:
        raise RulesError('the results were checked against different rule sets: aggregate each set apart')
    km = [float(r.get('distance_km') or 0.0) for r in res]
    ids = []
    for r in res:
        for x in r['rules']:
            if x['id'] not in ids:
                ids.append(x['id'])
    rows = []
    for rid in ids:
        per = [next((x for x in r['rules'] if x['id'] == rid), None) for r in res]
        items = [(p['count'] if p and p['state'] != 'not evaluable' else None, k) for p, k in zip(per, km)]
        ev = [(c, k) for c, k in items if c is not None]
        n_ev = len(ev)
        cnt = sum(c for c, _ in ev)
        kmv = sum(k for _, k in ev)
        runs_v = sum(1 for c, _ in ev if c > 0)
        lo, hi = wilson(runs_v, n_ev)
        row = {'id': rid, 'title': next(p['title'] for p in per if p), 'runs_evaluated': n_ev, 'violations': cnt,
               'distance_km': kmv, 'per_km': cnt / kmv if kmv > 0 else None, 'per_km_ci95': None,
               'runs_violating': runs_v, 'share_violating': runs_v / n_ev if n_ev else None,
               'share_ci95': [lo, hi] if n_ev else None,
               'min_robustness': min((p['robustness'] for p in per if p and p['robustness'] is not None),
                                     default=None)}
        if n_ev > 1 and kmv > 0:
            b = bootstrap(lambda xs: (sum(c for c, _ in xs) / sum(k for _, k in xs)) if sum(k for _, k in xs) > 0
                          else float('nan'), ev, n=n_boot, seed=seed, unit='image')
            row['per_km_ci95'] = [b['ci_low'], b['ci_high']]
        rows.append(row)
    tot = sum(r['totals']['violations'] for r in res)
    return {'runs': len(res), 'distance_km': sum(km), 'violations': tot,
            'per_km': tot / sum(km) if sum(km) > 0 else None, 'rules': rows,
            'ruleset': res[0].get('ruleset'),
            'interval': 'per km: percentile bootstrap over runs (ratio of sums); runs violating: Wilson 95 %'}


def table(result: dict) -> str:
    """A check's result as text."""
    lines = [f'{result.get("run") or result.get("episode") or ""}  rules {result["ruleset"]["id"]}, '
             f'{result["samples"]} samples, {result["duration_s"]} s, {result["distance_km"]} km',
             f'{"rule":22s} {"state":14s} {"robustness":>11s} {"count":>6s} {"per km":>8s} {"evaluated":>9s}']
    for r in result['rules']:
        rob = '∞' if r['vacuous'] else ('—' if r['robustness'] is None else f'{r["robustness"]:.3f}')
        pk = '—' if r['per_km'] is None else f'{r["per_km"]:.2f}'
        lines.append(f'{r["id"][:22]:22s} {r["state"]:14s} {rob:>11s} {r["count"]:6d} {pk:>8s} '
                     f'{100 * (r["evaluated_share"] or 0):8.0f}%')
        for v in r['violations'][:5]:
            lines.append(f'{"":24s}{v["start"]:.2f}–{v["end"]:.2f} s, worst {v["min_robustness"]:.3f} at '
                         f'{v["at"]:.2f} s')
        if r['missing_signals']:
            lines.append(f'{"":24s}unknown signals: {", ".join(r["missing_signals"])}')
    lines += [f'warning: {w}' for w in result.get('warnings') or ()]
    lines += [f'note: {n}' for n in result.get('notes') or ()]
    return '\n'.join(lines)


# ================================================================================================ the command
def main_rules(argv) -> int:
    """`falcon rules check RUN... [--set india|generic|FILE] [--xodr FILE] [--out DIR] [--json]`, `falcon rules list`,
    `falcon rules show SET`: 0 ok (violations are results, not failures), 2 failed."""
    import argparse
    from . import provenance as PV
    ap = argparse.ArgumentParser(prog='falcon rules', description='traffic rules as temporal logic: check runs')
    sub = ap.add_subparsers(dest='action', required=True)
    c = sub.add_parser('check', help='check recorded runs (or an episodes.jsonl) against a rule set')
    c.add_argument('runs', nargs='+', help='run folders, or an episodes.jsonl of gym / Leaderboard episodes')
    c.add_argument('--set', dest='ruleset', default='', help='india, generic or a rule file (default: by the run\'s '
                                                             'handedness: LHT india, else generic)')
    c.add_argument('--xodr', default=None, help='the map\'s OpenDRIVE file (default: the run\'s or CARLA\'s; '
                                                '"none": no file)')
    c.add_argument('--out', default='', help='write rules.json (and per run rules-N.json) here')
    c.add_argument('--json', action='store_true', help='print the result as JSON')
    sub.add_parser('list', help='the built-in rule sets and their rules')
    s = sub.add_parser('show', help='print a rule set\'s file (to copy and edit)')
    s.add_argument('set')
    sub.add_parser('signals', help='the signals a rule can read')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        if a.action == 'list':
            for k in BUILTIN:
                rs = load_ruleset(k)
                print(f'{k}: {rs["name"]} [{rs["handedness"]}]')
                for r in rs['rules']:
                    print(f'  {r["id"]:22s} {r["title"]} ({", ".join(r["tags"])})')
            return 0
        if a.action == 'signals':
            for k, (u, d) in SIGNALS.items():
                print(f'{k:22s} {u:6s} {d}')
            return 0
        if a.action == 'show':
            if a.set in BUILTIN:
                print(BUILTIN[a.set], end='')
            else:
                with open(a.set, encoding='utf-8') as fh:
                    parse_ruleset(fh.read(), a.set)
                    fh.seek(0)
                    print(fh.read(), end='')
            return 0
        results = []
        for p in a.runs:
            if os.path.isfile(p) and p.endswith('.jsonl'):
                for e in _jsonl(p):
                    r = check_episode(e, a.ruleset or None)
                    r['source'] = os.path.abspath(p)
                    results.append(r)
            else:
                results.append(check_run(p, a.ruleset or None, '' if a.xodr == 'none' else a.xodr))
        groups = {}
        for r in results:
            groups.setdefault(r['ruleset']['id'], []).append(r)
        aggs = {k: aggregate(v) for k, v in groups.items() if len(v) > 1}
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            doc = {'schema': 'falcon.rules_results/1', 'results': results, 'aggregates': aggs,
                   'provenance': PV.block({f'run{i}': os.path.join(r['run'], 'frames.jsonl')
                                           for i, r in enumerate(results) if r.get('run')},
                                          settings={'ruleset': a.ruleset or 'by handedness', 'xodr': a.xodr})}
            PV.write_json(os.path.join(a.out, 'rules.json'), doc)
        if a.json:
            print(json.dumps({'results': results, 'aggregates': aggs}, indent=1))
        else:
            for r in results:
                print(table(r))
                print()
            for k, agg in aggs.items():
                print(f'{k}: {agg["runs"]} runs, {agg["distance_km"]:.3f} km: {agg["violations"]} violations'
                      + (f' ({agg["per_km"]:.2f} per km)' if agg['per_km'] is not None else ''))
        return 0
    except (RulesError, S.STLError, OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return 2
