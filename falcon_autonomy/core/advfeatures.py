"""Adversarial multi-agent traffic: the definitions both sides share (numpy only; importable from CARLA's Python and
the app's) — the adversary roles, their observation vector, their actions and actuator limits, fault attribution
(who hit whom), the realism constraints and the adversaries' reward terms. The environment worker is
workers/adversary.py (the gym worker with learned traffic agents), the client and the multi-agent PPO are
core.adversary; docs/adversarial-traffic.md is the guide.

Frames and units as core.drivefeatures: CARLA's (x forward, y right, z up, left-handed), metres, seconds, yaw in
degrees positive turning right; a heading (cos yaw, sin yaw) has its right at (−sin yaw, cos yaw).

Roles (ROLES)
-------------
An adversary is placed like a scenario actor (the director's placement, relative to the ego and the traffic's
handedness) from a role: its kind, where it starts, its speed, what the map must offer (needs) and the traffic rules
its role legitimately bends (allow: e.g. a filtering two-wheeler rides on the lane line, a wrong-way scooter rides
against the traffic — both everyday on Bengaluru roads). A role's numbers are drawn with the episode seed from the
ranges below (heuristic ranges, near the built-in Indian scenarios of core.scenarios); the ego starts at 15–25 km/h
and every adversary at no more than ROLE_SPEED_CAP_KMH, so that no episode starts above a 30 km/h limit (an urban
limit; Vehicle.get_speed_limit as measured is in docs/adversarial-traffic.md) — going faster is the adversary's own
choice, and charged by the speed_limit rule.

The observation vector (ADV_FEATURES, float32; one per adversary, in its own frame)
----------------------------------------------------------------------------------
Its own motion (speed, longitudinal acceleration), where it is on CARLA's map (the lateral offset from its lane's
centre, + right; heading error to the lane; lane width; on a driving lane; wrong way: the lane's direction differs from
its heading by more than drivefeatures.WRONG_WAY_DEG outside junctions; same-direction lanes beside it, capped at 3),
the ego relative to it (position, heading, speed, relative velocity in its frame, the ego's acceleration, the
box-to-box gap and the time to collision of the two boxes at constant velocity — core.criticality.ttc, capped at
TTC_CAP_S), both on the ego's route (its arc length ahead of the ego's and its lateral offset; the ego's own offset),
the nearest other road user (present, position in its frame), the share of the episode elapsed, its previous action,
whether its role may drive the wrong way, and its kind (one-hot over ADV_KINDS). Absent quantities take their cap and
a 0 presence flag, never NaN.

Actions (ACTION_KINDS)
----------------------
"lateral" (default): [lateral, accel] ∈ [−1, 1]²: a lateral velocity command (× LAT_RATE_MAX m/s, + right) and an
acceleration command (× the kind's top acceleration when ≥ 0, × BRAKE_MAX when < 0). "discrete": an index into
MANOEUVRES (lateral −/0/+ LAT_RATE_DISCRETE m/s × accel BRAKE_DISCRETE / 0 / ACCEL_DISCRETE m/s²). Both go through
an actuator with bounded rates (Actuator: the acceleration moves towards its command at no more than JERK_ACT m/s³,
the lateral velocity at no more than LAT_ACC_ACT m/s²; the lateral offset from the adversary's own lane path stays
within ±LAT_LIMIT_LANES lane widths, the lateral velocity slowing in time to stop there) and are followed along the
path by pure pursuit (the director's follow()), so the motion is smooth by construction. "continuous": [steer, accel] with the ego's own semantics (drivefeatures:
the kinematic bicycle model; the throttle scaled to the kind's top acceleration) — nothing bounds its jerk, so the
realism penalties do.

Fault attribution (attribute_fault)
-----------------------------------
At the onset of a box contact between two vehicles A and B, from their recent states (pose, speed, the heading of the
lane each is on, wrong-way and on-lane flags), the first rule that applies decides:
  F1 wrong way / off road: exactly one of them is driving the wrong way or is off the driving lanes at the contact →
     that one (driving against the lane's direction or off the carriageway is the violation that put it there).
  F2 lateral encroachment: each one's mean lateral velocity in its own lane's frame (speed · sin(yaw − lane yaw))
     towards the other over the last LAT_WINDOW_S; when the larger is at least LAT_FAULT_MPS and exceeds the other's
     by at least LAT_FAULT_MPS → that one (it moved into a space the other occupied: a cut-in, a swerve, a side-swipe).
  F3 rear-end: one's front zone meets the other's rear zone (zones from the closest points of the two boxes the step
     before the contact: the side of the box, normalised by its half extents, the point lies beyond most) → the one
     whose front struck (boxes already overlapping then: the zone of the other's centre), unless the struck vehicle
     was laterally clear of the striker's path (no lateral overlap of the boxes in the striker's frame) at some sample
     within the last CUT_IN_S — then the struck vehicle cut in at an unsafe distance and is at fault (after
     Responsibility-Sensitive Safety's longitudinal rule: the follower keeps a safe distance, and a vehicle cutting in
     must leave one; Shalev-Shwartz, Shammah & Shashua 2017, arXiv:1708.06374 — RSS's safe-distance formula itself
     is not evaluated here).
  F4 otherwise: shared.
The thresholds are heuristic. The ego's collision "caused by the ego" is one where the ego is at fault.

Realism (REALISM_LIMITS, realism_ok)
------------------------------------
An adversary is realistic in an episode when: it caused no collision (fault attribution, with the ego or any other
road user) and touched no other adversary; it never left the driving lanes; it drove the wrong way no longer than its
role's budget (0 s unless the role allows it); its dynamics stayed within REALISM_LIMITS (|longitudinal acceleration|,
|lateral acceleration| = speed × yaw rate, |jerk|; heuristic bounds) but for DYN_BUDGET_S seconds; and its traffic-rule
violations (core.rules on its own trace: the Indian set on a left-hand-traffic world, the generic set on CARLA's
right-hand towns, or a configured set) on the rules its role does not allow are at most RULE_BUDGET per episode. A
failure of the ego counts as realistic when every adversary of its episode is realistic.

Rewards (adversary_terms; weights DEFAULT_ADV_REWARD, heuristic, not tuned; every term logged)
------------------------------------------------------------------------------------------------
For adversary i at each step (its individual terms; the reward is (1 − team_share) × its own + team_share × the mean
over the episode's adversaries of theirs, for the "ego failure" terms only — the penalties stay individual):
    ego_collision   + w per new collision of the ego with i that the ego caused (fault attribution)
    ego_infraction  + w per new ego failure infraction (EGO_FAILURE_INFRACTIONS) while i is within CREDIT_RADIUS_M
    ttc             + w · (s(TTC) − its best s so far)⁺, s = exp(−TTC / 1.5 s) (core.criticality's TTC scale): pays
                    only new records, so over an episode it sums to w · exp(−min TTC_i / 1.5 s)
    adv_fault       − w per collision i caused (with the ego); − w/2 for a shared one
    adv_contact     − w per new contact of i with another adversary or a scripted actor
    off_road        − w_onset per off-road onset − w · dt while off the driving lanes
    wrong_way       − w · dt while driving the wrong way beyond its role's budget
    dynamics        − w · dt per realism limit exceeded this step
    rules           − w per traffic-rule violation beyond the budget (at the episode's end, core.rules)
    diversity       + w / √N per realistic ego failure, N the failures with the same descriptor so far in training
                    (count-based novelty: Strehl & Littman 2008; Bellemare et al. 2016) — added by the trainer
"agent_blocked" (an adversary standing in front of the ego for 20 s) and outside_route_lanes (an evasive lane change)
are not rewarded: neither is a failure of the ego's driving. Nor are red_light and stop_sign: an adversary cannot
plausibly make the ego run a light, and the lane follower ignores lights (measured on Town10HD: it ran a red light in
an adversarial episode), so crediting them would reward the adversaries for the ego's own standing faults. They stay in
the ego's infractions.

Limits: lanes and fault use the boxes' centres and headings at 20 Hz; F2 needs a lane heading (off the lanes the
vehicle's own heading at the window's start stands in); junction right of way is not modelled (a T-bone in a junction
falls to F2 or F4); the adversaries have no collision sensor, so their contacts with map geometry are not detected.
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA = 'falcon.advfeatures/1'
TTC_CAP_S = 10.0
REL_CAP_M = 60.0
LAT_CAP_M = 15.0
TTC_SCALE_S = 1.5                       # core.criticality's TTC scale (van der Horst 1990)
CREDIT_RADIUS_M = 25.0                  # heuristic: an ego infraction is credited to adversaries this close
ADV_OFFROAD_END_M = 2.0                 # an adversary this far beyond a driving lane's edge ends the episode

ADV_KINDS = ('car', 'motorcycle', 'autorickshaw')
KIND_TOP_ACCEL = {'car': 3.0, 'motorcycle': 3.0, 'autorickshaw': 2.0}      # m/s²; autorickshaw: core.assets
KIND_TOP_KMH = {'car': 80.0, 'motorcycle': 70.0, 'autorickshaw': 65.0}     # heuristic urban caps; autorickshaw: assets

# ---------------------------------------------------------------------------------------------------- actions
ACTION_KINDS = ('lateral', 'discrete', 'continuous')
LAT_RATE_MAX = 1.5                      # m/s: lateral velocity command at |lateral| = 1 (heuristic)
BRAKE_MAX = 6.0                         # m/s²: the hardest braking a command can ask (heuristic: emergency braking)
LAT_RATE_DISCRETE, ACCEL_DISCRETE, BRAKE_DISCRETE = 1.0, 2.0, -4.0
MANOEUVRES = tuple((lat, acc) for lat in (-1.0, 0.0, 1.0) for acc in ('brake', 'hold', 'accel'))
JERK_ACT = 8.0                          # m/s³: the actuator's jerk limit (heuristic)
LAT_ACC_ACT = 2.0                       # m/s²: the actuator's lateral acceleration limit (heuristic)
LAT_LIMIT_LANES = 1.6                   # the lateral offset from its lane path stays within this many lane widths

# ---------------------------------------------------------------------------------------------------- realism
REALISM_LIMITS = {'a_long_mps2': 4.0, 'a_lat_mps2': 4.0, 'jerk_mps3': 8.0}     # heuristic bounds
DYN_BUDGET_S = 0.5                      # heuristic: seconds over a dynamics limit an episode tolerates
RULE_BUDGET = 1                         # heuristic: violations of not-allowed rules tolerated per episode
LAT_WINDOW_S, LAT_FAULT_MPS, CUT_IN_S = 0.5, 0.3, 2.0                          # fault attribution (heuristic)
EGO_FAILURE_INFRACTIONS = ('off_road', 'wrong_way', 'route_deviation', 'collision_static', 'collision_pedestrian')
DEFAULT_ADV_REWARD = {
    'ego_collision': 10.0, 'ego_infraction': 5.0, 'ttc': 3.0, 'team_share': 0.5,
    'adv_fault': 10.0, 'adv_contact': 10.0, 'off_road': 2.0, 'off_road_onset': 5.0, 'wrong_way': 1.0,
    'dynamics': 1.0, 'rules': 2.0, 'diversity': 2.0,
}
ADV_REWARD_TERMS = ('ego_collision', 'ego_infraction', 'ttc', 'adv_fault', 'adv_contact', 'off_road', 'wrong_way',
                    'dynamics', 'rules', 'diversity')
TEAM_TERMS = ('ego_collision', 'ego_infraction', 'ttc')


def _u(lo, hi):
    return {'lo': float(lo), 'hi': float(hi)}


# allow: rule ids of core.rules (india and generic) the role legitimately bends; budget_ww_s: wrong-way seconds
ROLES = {
    'cut_in_car': {
        'name': 'Car cutting in', 'kind': 'car', 'needs': ['adjacent_same'], 'lane': 'adjacent', 'heading': 'lane',
        'ahead_m': _u(-15, 5), 'lateral_m': _u(0, 0), 'speed': ('ego_plus', _u(0, 10)),
        'allow': ['following_distance'], 'budget_ww_s': 0.0,
        'about': 'a car in the next lane, beside or just behind the ego, faster than it'},
    'lead_car': {
        'name': 'Lead car', 'kind': 'car', 'needs': [], 'lane': 'ego', 'heading': 'lane',
        'ahead_m': _u(15, 40), 'lateral_m': _u(0, 0), 'speed': ('ego_plus', _u(-5, 5)),
        'allow': [], 'budget_ww_s': 0.0, 'about': 'a car ahead in the ego\'s lane'},
    'filtering_bike': {
        'name': 'Two-wheeler filtering', 'kind': 'motorcycle', 'needs': ['adjacent_same'], 'lane': 'line',
        'heading': 'lane', 'ahead_m': _u(-30, -10), 'lateral_m': _u(-0.4, 0.4), 'speed': ('ego_plus', _u(5, 20)),
        'allow': ['lane_discipline', 'keep_left', 'keep_right', 'overtake_right', 'overtake_left',
                  'following_distance'], 'budget_ww_s': 0.0,
        'about': 'a motorcycle behind the ego on the lane line, overtaking between the lanes (Bengaluru-style)'},
    'wrong_way_scooter': {
        'name': 'Wrong-way scooter', 'kind': 'motorcycle', 'needs': [], 'lane': 'ego', 'heading': 'against',
        'ahead_m': _u(40, 100), 'lateral_m': _u(0.5, 1.4), 'speed': ('abs', _u(15, 35)),
        'allow': ['no_wrong_way', 'keep_left', 'keep_right', 'lane_discipline'], 'budget_ww_s': math.inf,
        'about': 'a scooter riding against the traffic in the ego\'s lane near the kerb (everyday in Bengaluru)'},
    'autorickshaw_cut_in': {
        'name': 'Autorickshaw cutting in', 'kind': 'autorickshaw', 'needs': ['adjacent_same'], 'lane': 'adjacent',
        'heading': 'lane', 'ahead_m': _u(0, 8), 'lateral_m': _u(0, 0), 'speed': ('ego_plus', _u(3, 12)),
        'allow': ['following_distance', 'lane_discipline'], 'budget_ww_s': 0.0,
        'about': 'an autorickshaw in the next lane, faster than the ego'},
}
# the scripted scenarios (core.scenarios) whose actors play a role: the same allowances apply to them
SCRIPTED_ROLES = {('cut_in', 'cutter'): 'cut_in_car', ('lead_brake', 'lead'): 'lead_car',
                  ('two_wheeler_filtering', 'bike'): 'filtering_bike',
                  ('wrong_way_scooter', 'scooter'): 'wrong_way_scooter',
                  ('autorickshaw_cut_in', 'auto'): 'autorickshaw_cut_in'}
EGO_SPEED_KMH = _u(15, 25)              # heuristic: urban speeds of the ego in adversarial episodes
ROLE_SPEED_CAP_KMH = 30.0               # heuristic: start speeds within a 30 km/h urban limit


PLACE_TRIES = 50
LANE_POS = {'ego': 0.0, 'adjacent': 1.0, 'line': 0.5}      # lanes towards the overtaking side
KIND_SIZE = {'car': (4.8, 2.0), 'motorcycle': (2.2, 0.9), 'autorickshaw': (2.7, 1.3)}   # heuristic, for placement
EGO_PLACE = {'place': {'lane': 'ego', 'ahead_m': 0.0, 'lateral_m': 0.0}, 'kind': 'car', 'heading': 'lane'}


def _overlaps(a: dict, b: dict, lane_w: float = 3.5, margin: float = 1.5) -> bool:
    """Whether two placements (lane, ahead_m, lateral_m) would overlap at the start, with a margin (a lane width of
    3.5 m assumed; the director refuses an overlapping spawn, so overlaps are redrawn before it)."""
    def band(x):                                  # metres towards the overtaking side (lateral_m: towards the kerb)
        p = x['place']
        return LANE_POS.get(p['lane'], 0.0) * lane_w - float(p.get('lateral_m', 0.0))
    (la, wa), (lb, wb) = KIND_SIZE.get(a['kind'], KIND_SIZE['car']), KIND_SIZE.get(b['kind'], KIND_SIZE['car'])
    if abs(band(a) - band(b)) >= (wa + wb) / 2.0 + 0.3:
        return False
    return abs(float(a['place']['ahead_m']) - float(b['place']['ahead_m'])) < (la + lb) / 2.0 + margin


def kind_of(kind: str) -> str:
    return kind if kind in ADV_KINDS else 'car'


def role_actor(role_key: str, aid: str, ego_kmh: float, rng) -> dict:
    """A scenario actor (core.scenarios' actor document) for a role, its numbers drawn with rng."""
    r = ROLES[role_key]

    def draw(d):
        return float(d['lo']) if d['hi'] == d['lo'] else float(rng.uniform(d['lo'], d['hi']))
    mode, rng_d = r['speed']
    sp = draw(rng_d) + (ego_kmh if mode == 'ego_plus' else 0.0)
    sp = max(0.0, min(sp, KIND_TOP_KMH[kind_of(r['kind'])], ROLE_SPEED_CAP_KMH))
    return {'id': aid, 'kind': r['kind'], 'place': {'lane': r['lane'], 'ahead_m': draw(r['ahead_m']),
                                                    'lateral_m': draw(r['lateral_m'])},
            'heading': r['heading'], 'speed_kmh': sp, 'start': 'moving', 'role': role_key}


def setup_scenario(roles, seed: int, ego_kmh: float | None = None, duration_s: float = 15.0,
                   ego_driver: str = 'follower') -> dict:
    """The concrete scenario document (core.scenarios) of an adversarial episode: the roles' actors, no timeline (the
    adversaries are driven by their policy), the ego at ego_kmh (drawn from EGO_SPEED_KMH when None)."""
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 7919])
    if ego_kmh is None:
        ego_kmh = float(rng.uniform(EGO_SPEED_KMH['lo'], EGO_SPEED_KMH['hi']))
    unknown = [r for r in roles if r not in ROLES]
    if unknown:
        raise ValueError(f'unknown adversary roles {unknown} (roles: {sorted(ROLES)})')
    actors, needs = [], []
    for k, role in enumerate(roles):
        for _ in range(PLACE_TRIES):                 # redraw a start that would overlap another road user's
            a = role_actor(role, f'{role}_{k}', ego_kmh, rng)
            if not any(_overlaps(a, b) for b in actors + [EGO_PLACE]):
                break
        else:
            raise ValueError(f'role {role}: no start clear of the others in {PLACE_TRIES} draws')
        actors.append(a)
        needs += [n for n in ROLES[role]['needs'] if n not in needs]
    return {'id': 'adversarial', 'name': 'adversarial traffic: ' + ', '.join(roles) if roles else 'no adversaries',
            'needs': needs, 'actors': actors, 'timeline': [], 'duration_s': float(duration_s),
            'ego': {'speed_kmh': float(ego_kmh), 'driver': ego_driver, 'route': 'straight'},
            'values': {'ego_kmh': float(ego_kmh)}, 'seed': int(seed), 'roles': list(roles)}


# ---------------------------------------------------------------------------------------------------- observations
ADV_FEATURES = (
    ('speed_mps', 'm/s', 'its speed'),
    ('accel_mps2', 'm/s²', 'its longitudinal acceleration'),
    ('lane_lat_m', 'm', 'offset from its lane\'s centre (+ right; 0 off the lanes)'),
    ('lane_heading_err_rad', 'rad', 'its yaw − its lane\'s heading'),
    ('lane_width_m', 'm', 'its lane\'s width'),
    ('on_lane', '', '1: its centre is on a driving lane'),
    ('wrong_way', '', '1: against its lane\'s direction'),
    ('lanes_left', '', 'same-direction lanes on its left (cap 3)'),
    ('lanes_right', '', 'same-direction lanes on its right (cap 3)'),
    ('ego_dx_m', 'm', 'the ego ahead of it (its frame)'),
    ('ego_dy_m', 'm', 'the ego to its right (its frame)'),
    ('ego_rel_yaw_rad', 'rad', 'ego yaw − its yaw'),
    ('ego_speed_mps', 'm/s', 'the ego\'s speed'),
    ('ego_rel_vx_mps', 'm/s', 'ego velocity − its velocity, forward (its frame)'),
    ('ego_rel_vy_mps', 'm/s', 'ego velocity − its velocity, right (its frame)'),
    ('ego_accel_mps2', 'm/s²', 'the ego\'s longitudinal acceleration'),
    ('box_gap_m', 'm', 'box-to-box distance to the ego (cap)'),
    ('ttc_s', 's', 'time to collision with the ego, boxes at constant velocity (cap)'),
    ('route_s_rel_m', 'm', 'its arc length on the ego\'s route − the ego\'s (cap ±)'),
    ('route_lat_m', 'm', 'its lateral offset from the ego\'s route (+ right)'),
    ('ego_route_lat_m', 'm', 'the ego\'s lateral offset from its route'),
    ('other_present', '', '1: another road user within range'),
    ('other_dx_m', 'm', 'the nearest other road user ahead of it'),
    ('other_dy_m', 'm', 'the nearest other road user to its right'),
    ('time_frac', '', 'share of the episode elapsed'),
    ('prev_a0', '', 'its previous action, first component (discrete: lateral)'),
    ('prev_a1', '', 'its previous action, second component (discrete: accel)'),
    ('ww_allowed', '', '1: its role may drive the wrong way'),
) + tuple((f'kind_{k}', '', f'1: a {k}') for k in ADV_KINDS)
ADV_FEATURE_NAMES = tuple(f[0] for f in ADV_FEATURES)
N_ADV_FEATURES = len(ADV_FEATURES)
AI = {n: k for k, n in enumerate(ADV_FEATURE_NAMES)}


def to_frame(dx, dy, yaw_deg):
    """A world displacement (dx, dy) in a frame heading yaw_deg: (forward, right)."""
    r = math.radians(float(yaw_deg))
    c, s = math.cos(r), math.sin(r)
    return dx * c + dy * s, -dx * s + dy * c


def wrap_rad(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def state_row(t, p) -> np.ndarray:
    """core.criticality's state (t, x, y, yaw, vx, vy, length, width) of a pose dict {'x', 'y', 'yaw', 'speed',
    'length', 'width'} (velocity along the heading)."""
    r = math.radians(p['yaw'])
    return np.array([t, p['x'], p['y'], p['yaw'], p['speed'] * math.cos(r), p['speed'] * math.sin(r),
                     p['length'], p['width']], float)


def features(adv: dict, ego: dict, lane: dict, route_rel: dict, other, t_frac: float, prev_action, ttc_s: float,
             gap_m: float, ww_allowed: bool, kind: str) -> np.ndarray:
    """The observation vector (module doc). adv / ego: {'x', 'y', 'yaw', 'speed', 'accel', 'length', 'width'};
    lane: adversary lane facts {'on_lane', 'lat', 'lane_yaw', 'width', 'wrong_way', 'lanes_left', 'lanes_right'};
    route_rel: {'s_rel', 'lat', 'ego_lat'}; other: (dx, dy) world displacement of the nearest other road user or
    None."""
    f = np.zeros(N_ADV_FEATURES, np.float64)
    f[AI['speed_mps']] = adv['speed']
    f[AI['accel_mps2']] = adv.get('accel', 0.0)
    if lane.get('on_lane'):
        f[AI['on_lane']] = 1.0
        f[AI['lane_lat_m']] = float(np.clip(lane['lat'], -LAT_CAP_M, LAT_CAP_M))
        f[AI['lane_heading_err_rad']] = wrap_rad(math.radians(adv['yaw'] - lane['lane_yaw']))
        f[AI['lane_width_m']] = lane['width']
        f[AI['wrong_way']] = float(bool(lane.get('wrong_way')))
        f[AI['lanes_left']] = min(float(lane.get('lanes_left') or 0), 3.0)
        f[AI['lanes_right']] = min(float(lane.get('lanes_right') or 0), 3.0)
    else:
        f[AI['lane_width_m']] = 3.5
    dx, dy = to_frame(ego['x'] - adv['x'], ego['y'] - adv['y'], adv['yaw'])
    f[AI['ego_dx_m']] = float(np.clip(dx, -REL_CAP_M, REL_CAP_M))
    f[AI['ego_dy_m']] = float(np.clip(dy, -REL_CAP_M, REL_CAP_M))
    f[AI['ego_rel_yaw_rad']] = wrap_rad(math.radians(ego['yaw'] - adv['yaw']))
    f[AI['ego_speed_mps']] = ego['speed']
    re, ra = math.radians(ego['yaw']), math.radians(adv['yaw'])
    rvx = ego['speed'] * math.cos(re) - adv['speed'] * math.cos(ra)
    rvy = ego['speed'] * math.sin(re) - adv['speed'] * math.sin(ra)
    f[AI['ego_rel_vx_mps']], f[AI['ego_rel_vy_mps']] = to_frame(rvx, rvy, adv['yaw'])
    f[AI['ego_accel_mps2']] = ego.get('accel', 0.0)
    f[AI['box_gap_m']] = min(float(gap_m), REL_CAP_M)
    f[AI['ttc_s']] = min(float(ttc_s), TTC_CAP_S)
    f[AI['route_s_rel_m']] = float(np.clip(route_rel['s_rel'], -REL_CAP_M, REL_CAP_M))
    f[AI['route_lat_m']] = float(np.clip(route_rel['lat'], -LAT_CAP_M, LAT_CAP_M))
    f[AI['ego_route_lat_m']] = float(np.clip(route_rel['ego_lat'], -LAT_CAP_M, LAT_CAP_M))
    f[AI['other_dx_m']], f[AI['other_dy_m']] = REL_CAP_M, 0.0
    if other is not None:
        ox, oy = to_frame(other[0], other[1], adv['yaw'])
        if math.hypot(ox, oy) < REL_CAP_M:
            f[AI['other_present']] = 1.0
            f[AI['other_dx_m']], f[AI['other_dy_m']] = ox, oy
    f[AI['time_frac']] = min(max(float(t_frac), 0.0), 1.0)
    f[AI['prev_a0']], f[AI['prev_a1']] = float(prev_action[0]), float(prev_action[1])
    f[AI['ww_allowed']] = float(bool(ww_allowed))
    f[AI[f'kind_{kind_of(kind)}']] = 1.0
    return f.astype(np.float32)


# ---------------------------------------------------------------------------------------------------- actions
def n_discrete() -> int:
    return len(MANOEUVRES)


def command(action, kind: str, action_kind: str):
    """(lateral velocity command m/s, acceleration command m/s², applied [a0, a1] as logged and observed) of an
    action; continuous: (steer, accel) of the ego's semantics in place of the two commands. ValueError when
    malformed."""
    amax = KIND_TOP_ACCEL[kind_of(kind)]
    if action_kind == 'discrete':
        try:
            k = int(np.asarray(action).reshape(-1)[0]) if np.ndim(action) else int(action)
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f'a discrete adversary action is an index, not {action!r}') from e
        if not 0 <= k < len(MANOEUVRES):
            raise ValueError(f'discrete adversary action {k} is not in 0…{len(MANOEUVRES) - 1}')
        lat, lon = MANOEUVRES[k]
        acc = {'brake': BRAKE_DISCRETE, 'hold': 0.0, 'accel': min(ACCEL_DISCRETE, amax)}[lon]
        return lat * LAT_RATE_DISCRETE, acc, [lat, {'brake': -1.0, 'hold': 0.0, 'accel': 1.0}[lon]]
    a = np.asarray(action, float).reshape(-1)
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        raise ValueError(f'an adversary action is two finite numbers, not {action!r}')
    a0, a1 = float(np.clip(a[0], -1.0, 1.0)), float(np.clip(a[1], -1.0, 1.0))
    if action_kind == 'continuous':
        return a0, a1, [a0, a1]
    if action_kind != 'lateral':
        raise ValueError(f'adversary action kind must be one of {ACTION_KINDS}, not {action_kind!r}')
    return a0 * LAT_RATE_MAX, (a1 * amax if a1 >= 0 else a1 * BRAKE_MAX), [a0, a1]


class Actuator:
    """Rate-limited actuation of the lateral and discrete actions (module doc): the acceleration follows its
    command at ≤ JERK_ACT, the lateral velocity at ≤ LAT_ACC_ACT, the lateral offset (from the adversary's lane
    path) within ±limit_m."""

    def __init__(self, dt: float, lateral0: float, limit_m: float, accel0: float = 0.0):
        self.dt = float(dt)
        self.lat, self.vlat, self.a = float(lateral0), 0.0, float(accel0)
        self.lat0, self.limit = float(lateral0), float(limit_m)

    def update(self, vlat_cmd: float, a_cmd: float):
        """Advance one step; returns (lateral offset, acceleration)."""
        dj = JERK_ACT * self.dt
        self.a += float(np.clip(a_cmd - self.a, -dj, dj))
        dl = LAT_ACC_ACT * self.dt
        lo, hi = self.lat0 - self.limit, self.lat0 + self.limit
        room = (hi - self.lat) if vlat_cmd > 0 else (self.lat - lo)     # slow down to stop at the limit
        vmax = math.sqrt(2.0 * LAT_ACC_ACT * max(room - abs(self.vlat) * self.dt, 0.0))
        vlat_cmd = math.copysign(min(abs(vlat_cmd), vmax), vlat_cmd)
        self.vlat += float(np.clip(vlat_cmd - self.vlat, -dl, dl))
        lat = self.lat + self.vlat * self.dt
        if lat < lo or lat > hi:                                        # a last guard (a residual of under one step)
            lat, self.vlat = min(max(lat, lo), hi), 0.0
        self.lat = lat
        return self.lat, self.a


def accel_to_control(v: float, v_next: float, dt: float, steer: float) -> dict:
    """The kinematic bicycle control (director.kin_step: 3 m/s² full throttle, 8 m/s² full brake) that takes the
    speed from v to v_next in one step, with this steer."""
    a = (float(v_next) - float(v)) / float(dt)
    if a >= 0:
        return {'throttle': min(a / 3.0, 1.0), 'steer': float(steer), 'brake': 0.0}
    return {'throttle': 0.0, 'steer': float(steer), 'brake': min(-a / 8.0, 1.0)}


# ---------------------------------------------------------------------------------------------------- dynamics
class Dynamics:
    """Longitudinal acceleration, jerk (from the speed) and lateral acceleration (speed × yaw rate) per step, and the
    seconds over each REALISM_LIMITS bound."""

    def __init__(self, dt: float, limits: dict | None = None):
        self.dt = float(dt)
        self.lim = dict(REALISM_LIMITS, **(limits or {}))
        self.v = self.a = self.yaw = None
        self.over_s = {k: 0.0 for k in self.lim}
        self.samples = {'a_long': [], 'a_lat': [], 'jerk': []}

    def update(self, speed: float, yaw_deg: float):
        """(a_long, a_lat, jerk, [limits exceeded this step])."""
        a = 0.0 if self.v is None else (speed - self.v) / self.dt
        j = 0.0 if self.a is None else (a - self.a) / self.dt
        yr = 0.0 if self.yaw is None else math.radians(((yaw_deg - self.yaw) + 180.0) % 360.0 - 180.0) / self.dt
        alat = speed * yr
        first = self.v is None
        self.v, self.a, self.yaw = speed, a, yaw_deg
        over = []
        if not first:
            for k, val in (('a_long_mps2', a), ('a_lat_mps2', alat), ('jerk_mps3', j)):
                if abs(val) > self.lim[k] + 1e-9:
                    over.append(k)
                    self.over_s[k] += self.dt
            self.samples['a_long'].append(round(a, 3))
            self.samples['a_lat'].append(round(alat, 3))
            self.samples['jerk'].append(round(j, 3))
        return a, alat, j, over


# ---------------------------------------------------------------------------------------------------- fault
def _box(x, y, yaw, length, width):
    r = math.radians(yaw)
    c, s = math.cos(r), math.sin(r)
    hx, hy, rx, ry = c * length / 2, s * length / 2, -s * width / 2, c * width / 2
    return np.array([[x + hx + rx, y + hy + ry], [x + hx - rx, y + hy - ry], [x - hx - rx, y - hy - ry],
                     [x - hx + rx, y - hy + ry]])


def _closest_on_segment(p, a, b):
    e = b - a
    t = float(np.clip(np.dot(p - a, e) / max(float(np.dot(e, e)), 1e-18), 0.0, 1.0))
    return a + t * e


def _disjoint(A: np.ndarray, B: np.ndarray) -> bool:
    """Whether two convex polygons are separated (separating axis test)."""
    for P in (A, B):
        for k in range(len(P)):
            e = P[(k + 1) % len(P)] - P[k]
            n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
                return True
    return False


def closest_points(A: np.ndarray, B: np.ndarray):
    """The closest pair of points (on A, on B) of two disjoint convex polygons (vertex-to-edge both ways; for
    overlapping ones, the deepest such pair is not computed: the pair found is still on the boundaries)."""
    best = (math.inf, None, None)
    for P, Q, flip in ((A, B, False), (B, A, True)):
        for p in P:
            for k in range(len(Q)):
                q = _closest_on_segment(p, Q[k], Q[(k + 1) % len(Q)])
                d = float(np.hypot(*(p - q)))
                if d < best[0]:
                    best = (d, q, p) if flip else (d, p, q)
    return best[1], best[2]


def zone(pt, x, y, yaw, length, width) -> str:
    """The side of a box ('front', 'rear', 'left', 'right') a point lies beyond most (normalised by the half
    extents)."""
    fx, fy = to_frame(pt[0] - x, pt[1] - y, yaw)
    u, v = fx / max(length / 2.0, 1e-6), fy / max(width / 2.0, 1e-6)
    if abs(u) >= abs(v):
        return 'front' if u >= 0 else 'rear'
    return 'right' if v > 0 else 'left'


def attribute_fault(A: dict, B: dict, dt: float) -> dict:
    """Who caused a contact (module doc: F1–F4). A, B: histories up to and including the contact step, each
    {'x', 'y', 'yaw', 'speed', 'lane_yaw' (NaN off the lanes), 'wrong_way', 'on_lane': sequences of equal length,
    'length', 'width'}. Returns {'at_fault': 'a' | 'b' | 'shared', 'rule': 'F1'…'F4', 'detail': {…}}."""
    n = min(len(A['x']), len(B['x']))
    if n < 1:
        raise ValueError('fault attribution needs at least the contact step')

    def last(H, k):
        return H[k][-1]
    bad_a = bool(last(A, 'wrong_way')) or not bool(last(A, 'on_lane'))
    bad_b = bool(last(B, 'wrong_way')) or not bool(last(B, 'on_lane'))
    if bad_a != bad_b:
        return {'at_fault': 'a' if bad_a else 'b', 'rule': 'F1',
                'detail': {'a_wrong_way': bool(last(A, 'wrong_way')), 'a_on_lane': bool(last(A, 'on_lane')),
                           'b_wrong_way': bool(last(B, 'wrong_way')), 'b_on_lane': bool(last(B, 'on_lane'))}}
    w = max(1, min(n, int(round(LAT_WINDOW_S / dt))))

    def toward(H, O):
        """Mean lateral velocity of H in its lane's frame towards O's current position, over the window."""
        k0 = len(H['x']) - w
        ref = H['lane_yaw'][k0] if np.isfinite(H['lane_yaw'][k0]) else H['yaw'][k0]
        vs = []
        for k in range(k0, len(H['x'])):
            ly = H['lane_yaw'][k] if np.isfinite(H['lane_yaw'][k]) else ref
            vs.append(H['speed'][k] * math.sin(math.radians(H['yaw'][k] - ly)))
        ly = H['lane_yaw'][-1] if np.isfinite(H['lane_yaw'][-1]) else ref
        side = to_frame(O['x'][-1] - H['x'][-1], O['y'][-1] - H['y'][-1], ly)[1]
        return float(np.mean(vs)) * (1.0 if side >= 0 else -1.0)
    ta, tb = toward(A, B), toward(B, A)
    det = {'a_lateral_toward_mps': round(ta, 3), 'b_lateral_toward_mps': round(tb, 3)}
    if max(ta, tb) >= LAT_FAULT_MPS and abs(ta - tb) >= LAT_FAULT_MPS:
        return {'at_fault': 'a' if ta > tb else 'b', 'rule': 'F2', 'detail': det}
    k = len(A['x']) - 2 if n >= 2 else len(A['x']) - 1
    kb = len(B['x']) - 2 if n >= 2 else len(B['x']) - 1
    ba = _box(A['x'][k], A['y'][k], A['yaw'][k], A['length'], A['width'])
    bb = _box(B['x'][kb], B['y'][kb], B['yaw'][kb], B['length'], B['width'])
    if _disjoint(ba, bb):
        pa, pb = closest_points(ba, bb)
        mid = (pa + pb) / 2.0
        za = zone(mid, A['x'][k], A['y'][k], A['yaw'][k], A['length'], A['width'])
        zb = zone(mid, B['x'][kb], B['y'][kb], B['yaw'][kb], B['length'], B['width'])
    else:                                          # already overlapping the step before: the other's centre
        za = zone((B['x'][kb], B['y'][kb]), A['x'][k], A['y'][k], A['yaw'][k], A['length'], A['width'])
        zb = zone((A['x'][k], A['y'][k]), B['x'][kb], B['y'][kb], B['yaw'][kb], B['length'], B['width'])
    det.update(a_zone=za, b_zone=zb)
    pair = None
    if za == 'front' and zb == 'rear':
        pair = ('a', A, B)
    elif zb == 'front' and za == 'rear':
        pair = ('b', B, A)
    if pair is not None:
        striker, S, V = pair
        m = max(1, min(n, int(round(CUT_IN_S / dt)) + 1))
        clear = False
        for j in range(1, m + 1):
            sx, sy, syaw = S['x'][-j], S['y'][-j], S['yaw'][-j]
            lat = to_frame(V['x'][-j] - sx, V['y'][-j] - sy, syaw)[1]
            rel = math.radians(V['yaw'][-j] - syaw)
            across = abs(V['length'] / 2 * math.sin(rel)) + abs(V['width'] / 2 * math.cos(rel))
            if abs(lat) >= S['width'] / 2 + across:
                clear = True
                break
        det['struck_was_clear_within_s'] = CUT_IN_S if clear else None
        other = 'b' if striker == 'a' else 'a'
        return {'at_fault': other if clear else striker, 'rule': 'F3', 'detail': det}
    return {'at_fault': 'shared', 'rule': 'F4', 'detail': det}


# ---------------------------------------------------------------------------------------------------- rewards
def ttc_score(ttc_s: float) -> float:
    return math.exp(-max(float(ttc_s), 0.0) / TTC_SCALE_S) if math.isfinite(ttc_s) else 0.0


def adversary_terms(q: dict, weights: dict | None = None, dt: float = 0.05) -> dict:
    """One adversary's individual reward terms for a step (module doc). q: 'ego_collision' (new ego-caused
    collisions with it), 'ego_infractions' (new ego failure infractions credited to it), 'ttc_gain' (s(TTC) − its
    best so far, ≥ 0), 'adv_fault' (collisions it caused), 'shared_fault', 'adv_contact', 'off_road' (bool),
    'off_road_onset' (bool), 'wrong_way_over' (bool: beyond its budget), 'dyn_over' (limits exceeded), 'rules_over'
    (violations beyond the budget, at the episode's end)."""
    w = dict(DEFAULT_ADV_REWARD, **(weights or {}))
    t = {
        'ego_collision': w['ego_collision'] * float(q.get('ego_collision', 0)),
        'ego_infraction': w['ego_infraction'] * float(q.get('ego_infractions', 0)),
        'ttc': w['ttc'] * max(float(q.get('ttc_gain', 0.0)), 0.0),
        'adv_fault': -w['adv_fault'] * (float(q.get('adv_fault', 0)) + 0.5 * float(q.get('shared_fault', 0))),
        'adv_contact': -w['adv_contact'] * float(q.get('adv_contact', 0)),
        'off_road': -(w['off_road'] * dt if q.get('off_road') else 0.0) -
        (w['off_road_onset'] if q.get('off_road_onset') else 0.0),
        'wrong_way': -w['wrong_way'] * dt if q.get('wrong_way_over') else 0.0,
        'dynamics': -w['dynamics'] * dt * float(q.get('dyn_over', 0)),
        'rules': -w['rules'] * float(q.get('rules_over', 0)),
        'diversity': 0.0,
    }
    return {k: float(t[k]) for k in ADV_REWARD_TERMS}


def mix_team(terms: dict, weights: dict | None = None) -> dict:
    """{agent: terms} → {agent: reward}: (1 − s) × own + s × the mean of the team's for TEAM_TERMS; the rest own."""
    w = dict(DEFAULT_ADV_REWARD, **(weights or {}))
    s = float(w['team_share'])
    if not terms:
        return {}
    team = {k: float(np.mean([t[k] for t in terms.values()])) for k in TEAM_TERMS}
    out = {}
    for a, t in terms.items():
        r = sum(t[k] for k in ADV_REWARD_TERMS if k not in TEAM_TERMS)
        r += sum((1.0 - s) * t[k] + s * team[k] for k in TEAM_TERMS)
        out[a] = float(r)
    return out


def realism_ok(summary: dict) -> bool:
    """Whether one adversary's episode summary (workers/adversary.py) meets the realism constraints (module doc)."""
    return (summary.get('fault_collisions', 0) == 0 and summary.get('contacts', 0) == 0 and
            summary.get('off_road_s', 0.0) <= 1e-9 and summary.get('wrong_way_over_s', 0.0) <= 1e-9 and
            summary.get('dynamics_over_s', 0.0) <= DYN_BUDGET_S + 1e-9 and summary.get('rules_over', 0) == 0)


def sector(dx: float, dy: float) -> str:
    """Where a road user is relative to the ego (dx forward, dy right in the ego's frame): front, rear, left, right."""
    if abs(dx) >= abs(dy):
        return 'front' if dx >= 0 else 'rear'
    return 'right' if dy > 0 else 'left'


def descriptor(failure: dict) -> str:
    """A failure's diversity descriptor: the adversary's role, the failure kind, its sector relative to the ego and
    the ego's speed in 10 km/h bins."""
    return '|'.join([str(failure.get('role') or '-'), str(failure.get('kind')), str(failure.get('sector') or '-'),
                     str(int(float((failure.get('ego') or {}).get('speed_mps', 0.0)) * 3.6 // 10) * 10)])
