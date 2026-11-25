"""Regulation test packs: ready-made scenario suites scored after published test procedures (standard library +
numpy at use; docs/regulation-packs.md is the manual and lists the sources).

THESE PACKS ARE AN ENGINEERING PRE-CHECK IN SIMULATION. They are not a type-approval, not an official Euro NCAP
rating and not a conformity assessment; Falcon Autonomy is not a qualified tool (DISCLAIMER is printed in every
result and report).

A pack is data: its sources (document title, version, date, where it was read), test cases — each with the clause it
comes from, the Falcon scenario it runs (core.scenarios; exportable as OpenSCENARIO through core.openscenario) and
its criteria — and how the cases are scored. Every number that comes from a document is a value record

    {'value', 'unit', 'source': <a key of SOURCES>, 'clause', 'verified': True | False, 'basis', 'note'}

basis 'text': read in the document's text (verified True); 'figure': read from a chart or figure of the document by
measurement, not printed there as a number (verified False, and listed as such); 'recalled': recalled, not checked
against the document (verified False). A number that is Falcon's own choice — where the document leaves the test
parameters to the Technical Service, or says nothing — carries source 'falcon', verified None and the reason: it is
never presented as the regulation's. unverified(pack) lists every value with verified False; a report prints them.

The packs (PACKS):

  r157          UN Regulation No. 157 (ALKS), the provisions up to 60 km/h that the original series and the 01 series
                share: lane keeping (Annex 5 4.1), a stationary obstacle and a crossing pedestrian (4.2), following
                with the minimum following distance of 5.2.3.3 and a lead vehicle braking at 6 m/s² (4.3), cut-in
                (4.4) and cut-out (4.5) grids classified by the regulation's performance model 1 (the competent and
                careful human driver, implemented in pm1_* from its parameter table: a case the model avoids must be
                avoided; one it cannot is run and reported as informative), and the minimum risk manoeuvre's
                deceleration demand (5.5) through core.mrm. Annex 5 leaves the specific test parameters to the
                Technical Service: the speeds and gaps chosen here are Falcon's and say so.
  ncap_aeb_c2c  Euro NCAP AEB Car-to-Car, the 2023 protocols (test protocol v4.3, assessment protocol v10.4): the
                CCRs, CCRm and CCRb grids with their speed and overlap matrices, each grid point scored by the colour
                band of its (relative) impact speed, the overlap weighting, the points per speed and the scenario
                points. The FCW rows need the protocol's braking-robot driver and are refused by name.
  ncap_lss      Euro NCAP Lane Support Systems (test protocol v4.3): the LKA and ELK grids with their lateral
                velocities and limits. Falcon's scenario language cannot impose the lane-departure path on the ego
                (no steering disturbance): every case is refused by name; the pack is the data and the scoring rule.
  iso34502      ISO 34502:2022, as a coverage CHECKLIST, not a test: the critical-scenario categories (traffic,
                perception, vehicle control) mapped to Falcon scenarios and features. The standard is paywalled: only
                its public title, scope and annex titles are from ISO; the breakdown inside each category is the
                public JAMA framework's (Ver 3.0) and is labelled as that.

Running a pack (run_pack): each runnable case is one stack run (core.stack / workers.stack) through a CarlaService
(sim.service.study_service) with the chosen driver — the system under test: Falcon's built-in follower, CARLA's
autopilot, or a ROS 2 stack — in <workspace>/regpacks/<name>/runs/<key>/; a case the world or backend cannot run is
refused by name with the worker's reason. Scoring reads the run's own results: collision and its time
(core.criticality), the relative speed at the first contact and the gap to the lead (trajectories.json), the lateral
offset from the lane centre (frames.jsonl), the MRM verdict (core.mrm). Simulated vehicles, sensors and brakes: a
pass here says the stack handled Falcon's rendering of the case, nothing more.

    falcon regpack list
    falcon regpack show PACK [--json]
    falcon regpack run PROJECT PACK --name N [--case ID …] [--group G …] [--driver follower|autopilot] …
    falcon regpack report DIR [--out FILE.html]
    falcon regpack export PACK --out DIR        scenarios as Falcon JSON and OpenSCENARIO, and a falcon.suite/1 suite

Output: regpack.json (SCHEMA) and report.html in the result folder. core.gsn binds a result as evidence kind
'regpack' (the verdict, or a field such as score.share).
"""
from __future__ import annotations

import copy
import json
import math
import os

SCHEMA = 'falcon.regpack/1'
DISCLAIMER = ('An engineering pre-check in simulation: not a type-approval, not an official Euro NCAP rating and not '
              'a conformity assessment. Falcon Autonomy is not a qualified tool; the vehicles, sensors and brakes are '
              'simulated, and the scenarios are Falcon\'s rendering of the published procedures.')
BASIS_TEXT = {'figure': 'read from a chart or figure of the document by measurement, not printed there as a number',
              'recalled': 'recalled, not checked against the document'}
G = 9.80665                        # m/s² per g (standard gravity; the regulation gives decelerations in G)
EGO = (4.9, 1.84)                  # the ego's length and width (core.rules.EGO_DEFAULT: CARLA's mkz_2020)
TARGET = (4.7, 1.8)                # Falcon's stand-in car target (not the Global Vehicle Target)
LANE_M = 3.5
ROUTE_M = 400.0                    # the scenario runner's ego route (workers.director: lane_path(wp, 400))

SOURCES = {
    'r157': {'title': 'UN Regulation No. 157 — Uniform provisions concerning the approval of vehicles with regard '
                      'to Automated Lane Keeping Systems', 'version': 'E/ECE/TRANS/505/Rev.3/Add.156 (original '
             'series)', 'date': '2021-03-04',
             'read_from': 'an Internet Archive copy of unece.org/sites/default/files/2023-12/R157e.pdf'},
    'r157_01': {'title': 'UN Regulation No. 157, 01 series of amendments (consolidated)',
                'version': 'E/ECE/TRANS/505/Rev.3/Add.156/Amend.4', 'date': '2023-03-03',
                'read_from': 'a mirror of the UNECE PDF (R157am4e.pdf), cross-checked against '
                             'ECE/TRANS/WP.29/2022/59/Rev.1 where noted'},
    'ncap_aeb_tp': {'title': 'Euro NCAP Test Protocol — AEB Car-to-Car systems', 'version': '4.3',
                    'date': '2023-12', 'read_from': 'an Internet Archive copy of cdn.euroncap.com/media/79864/'
                                                    'euro-ncap-aeb-c2c-test-protocol-v43.pdf'},
    'ncap_sa_ap': {'title': 'Euro NCAP Assessment Protocol — Safety Assist — Collision Avoidance',
                   'version': '10.4', 'date': '2023-12',
                   'read_from': 'an Internet Archive copy of cdn.euroncap.com/media/79866/'
                                'euro-ncap-assessment-protocol-sa-collision-avoidance-v104.pdf'},
    'ncap_lss_tp': {'title': 'Euro NCAP Test Protocol — Lane Support Systems', 'version': '4.3', 'date': '2023-12',
                    'read_from': 'an Internet Archive copy of cdn.euroncap.com/media/79865/'
                                 'euro-ncap-lss-test-protocol-v43.pdf'},
    'iso34502': {'title': 'ISO 34502:2022 Road vehicles — Test scenarios for automated driving systems — Scenario '
                          'based safety evaluation framework', 'version': 'First edition', 'date': '2022-11',
                 'read_from': 'ISO\'s catalogue page and the public sample (cover, contents, introduction); the '
                              'body is paywalled and was not read'},
    'jama_v3': {'title': 'JAMA Automated Driving Safety Evaluation Framework', 'version': 'Ver 3.0',
                'date': '2022-12', 'read_from': 'an Internet Archive copy of jama.or.jp/english/reports/docs/'
                                                'Automated_Driving_Safety_Evaluation_Framework_Ver3.0.pdf'},
    'falcon': {'title': 'Falcon\'s own choice (not a value of any document)', 'version': '', 'date': '',
               'read_from': ''},
}


class RegPackError(Exception):
    """A pack, case or result that cannot be used as asked."""


def V(value, unit, source, clause, basis='text', note='') -> dict:
    return {'value': value, 'unit': unit, 'source': source, 'clause': clause, 'verified': basis == 'text',
            'basis': basis, 'note': note}


def C(value, unit, why) -> dict:
    """Falcon's own choice: not a document's value (verified None)."""
    return {'value': value, 'unit': unit, 'source': 'falcon', 'clause': '', 'verified': None, 'basis': 'choice',
            'note': why}


# ============================================================================================ UN R157: the numbers
R157 = {
    'max_speed': V(60.0, 'km/h', 'r157', '5.2.3.1'),
    'max_speed_01': V(130.0, 'km/h', 'r157_01', '5.2.3.1', note='only with an MRM lane change capability; this pack '
                      'tests up to 60 km/h'),
    'following_table': V([[7.2, 1.0], [10.0, 1.1], [20.0, 1.2], [30.0, 1.3], [40.0, 1.4], [50.0, 1.5],
                          [60.0, 1.6]], '[km/h, s]', 'r157', '5.2.3.3',
                         note='M1: time gap t_front by speed; d_min = v · t_front, linear interpolation between rows'),
    'following_floor': V(2.0, 'm', 'r157', '5.2.3.3'),
    'cut_in_visible': V(0.72, 's', 'r157', '5.2.5.2(b)'),
    'cut_in_decel': V(6.0, 'm/s²', 'r157', '5.2.5.2(c)', note='TTC_LaneIntrusion > v_rel / (2 · 6 m/s²) + 0.35 s'),
    'cut_in_margin': V(0.35, 's', 'r157', '5.2.5.2(c)'),
    'lane_intrusion': V(0.3, 'm', 'r157', '5.2.5.2', note='front wheel 0.3 m beyond the outside edge of the lane '
                        'marking'),
    'ped_speed': V(5.0, 'km/h', 'r157', '5.2.5.3'),
    'em_decel': V(5.0, 'm/s²', 'r157', '5.3.1.1', note='a deceleration demand above this is an emergency manoeuvre'),
    'transition_s': V(10.0, 's', 'r157', '5.4.4.1', note='MRM earliest 10 s after the transition demand; '
                      'immediately on a severe failure (5.4.4.1.1)'),
    'mrm_decel': V(4.0, 'm/s²', 'r157', '5.5.1', note='5.5.2 in the 01 series'),
    'lane_keep_s': V(300.0, 's', 'r157', 'Annex 5, 4.1.2(a)', note='minimum test duration of 5 minutes'),
    'lead_decel': V(6.0, 'm/s²', 'r157', 'Annex 5, 4.3.2(f)', note='at least 6 m/s² mean fully developed '
                    'deceleration until standstill'),
    'detect_fwd': V(46.0, 'm', 'r157', '7.1.1'),
    # performance model 1 (original series Annex 4 Appendix 3; 01 series Annex 3 para 3.3), Table 1
    'pm_wander': V(0.375, 'm', 'r157', 'Annex 4 Appendix 3, Table 1'),
    'pm_perception': V(0.4, 's', 'r157', 'Annex 4 Appendix 3, Table 1'),
    'pm_delay': V(0.75, 's', 'r157', 'Annex 4 Appendix 3, Table 1'),
    'pm_jerk_s': V(0.6, 's', 'r157', 'Annex 4 Appendix 3, Table 1', note='0.6 s to 0.774 G'),
    'pm_decel_g': V(0.774, 'G', 'r157', 'Annex 4 Appendix 3, Table 1'),
    'pm_decel_trigger': V(5.0, 'm/s²', 'r157', 'Annex 4 Appendix 3, 3.4.3', note='perception begins when the '
                          'leading vehicle exceeds this deceleration'),
    'pm_thw': V(2.0, 's', 'r157', 'Annex 4 Appendix 3, 3.4.2', note='the time headway of the cut-out and '
                'deceleration data sheets'),
    'pm_dy0': V(1.6, 'm', 'r157', 'Annex 4 Appendix 3, Figure 6', 'figure', 'the example of the figure: lane 3.5 m, '
                'vehicle 1.9 m, driving in the centre'),
    'pm_vy_max': V(3.0, 'm/s', 'r157', 'Annex 4 Appendix 3, section 5', 'figure', 'the upper end of the data '
                   'sheets\' lateral-velocity axis'),
}


def r157_following_m(v_mps: float) -> float:
    """The minimum following distance of 5.2.3.3 (M1) at a speed: v · t_front, t_front interpolated, never < 2 m."""
    tab = R157['following_table']['value']
    kmh = v_mps * 3.6
    if kmh <= tab[0][0]:
        return R157['following_floor']['value']
    t = tab[-1][1]
    for (a, ta), (b, tb) in zip(tab, tab[1:]):
        if kmh <= b:
            t = ta + (tb - ta) * (kmh - a) / (b - a)
            break
    return max(v_mps * t, R157['following_floor']['value'])


def _pm1_stop(ve: float, gap_of_t, t_brake: float, vo_of_t, horizon: float = 30.0, dt: float = 0.005,
              t_from: float = 0.0) -> bool:
    """True when performance model 1's ego (braking from t_brake: the deceleration ramps over pm_jerk_s to
    pm_decel_g and holds) collides with the object whose position ahead of the ego's start is gap_of_t(t) (m, its
    rear against the ego's front) — checked from t_from on."""
    a_max, ramp = R157['pm_decel_g']['value'] * G, R157['pm_jerk_s']['value']
    x, v, t = 0.0, ve, 0.0
    while t < horizon:
        a = 0.0 if t < t_brake else -a_max * min((t - t_brake) / ramp, 1.0)
        v = max(v + a * dt, 0.0)
        x += v * dt
        t += dt
        if t >= t_from and gap_of_t(t) - x <= 0.0:
            return True
        if t >= max(t_brake, t_from) and v <= vo_of_t(t) + 1e-9:
            return False
    return False


def pm1_cut_in_avoidable(ve_kmh: float, vo_kmh: float, dx0: float, vy: float, dy0: float | None = None) -> bool:
    """Whether the regulation's performance model 1 avoids a cut-in: the other vehicle (constant speed vo < ve, its
    rear dx0 ahead of the ego's front) moves sideways at vy from t = 0; the risk is perceived when its centre has
    left the wander zone (pm_wander / vy), then the perception time and the delay pass before braking. It is in
    the ego's path from dy0 / vy. Falcon's implementation from the model's parameter table: the regulation's data
    sheets were not reproduced point by point."""
    ve, vo = ve_kmh / 3.6, vo_kmh / 3.6
    dy0 = R157['pm_dy0']['value'] if dy0 is None else dy0
    t_b = R157['pm_wander']['value'] / vy + R157['pm_perception']['value'] + R157['pm_delay']['value']
    return not _pm1_stop(ve, lambda t: dx0 + vo * t, t_b, lambda t: vo, t_from=dy0 / vy)


def pm1_cut_out_avoidable(ve_kmh: float, dx0_f: float, vy: float) -> bool:
    """Performance model 1 on a cut-out: the lead (at ve, pm_thw ahead) leaves the lane at vy; a stationary vehicle
    stands dx0_f ahead of the lead. Perception begins when the lead's centre leaves the wander zone."""
    ve = ve_kmh / 3.6
    t_b = R157['pm_wander']['value'] / vy + R157['pm_perception']['value'] + R157['pm_delay']['value']
    dist = ve * R157['pm_thw']['value'] + TARGET[0] + dx0_f
    return not _pm1_stop(ve, lambda t: dist, t_b, lambda t: 0.0)


def pm1_decel_avoidable(ve_kmh: float, decel: float) -> bool:
    """Performance model 1 on a lead braking at `decel` (a step) from pm_thw: perception begins at the brake onset
    (the model starts it when the lead exceeds pm_decel_trigger; a milder brake is taken the same way here)."""
    ve = ve_kmh / 3.6
    t_b = R157['pm_perception']['value'] + R157['pm_delay']['value']
    d0 = ve * R157['pm_thw']['value']

    def lead(t):
        ts = min(t, ve / decel)
        return d0 + ve * ts - 0.5 * decel * ts * ts
    return not _pm1_stop(ve, lead, t_b, lambda t: max(ve - decel * t, 0.0))


def r157_cut_in_required(ve_kmh: float, vo_kmh: float, dx0: float, vy: float) -> bool:
    """Whether 5.2.5.2 itself requires the collision to be avoided: the lateral movement visible for at least
    0.72 s before the lane intrusion and TTC there > v_rel / (2 · 6) + 0.35 s (the other vehicle slower, at constant
    speed). The intrusion is taken at the moment the other vehicle's side reaches the ego's lane line."""
    v_rel = (ve_kmh - vo_kmh) / 3.6
    if v_rel <= 0:
        return True
    t_int = ((LANE_M - TARGET[1]) / 2.0) / vy             # its side from its lane centre to the line between the lanes
    ttc = (dx0 - v_rel * t_int) / v_rel
    return t_int >= R157['cut_in_visible']['value'] and \
        ttc > v_rel / (2.0 * R157['cut_in_decel']['value']) + R157['cut_in_margin']['value']


# ============================================================================================== scenario builders
def _scn(sid, name, about, speed, actors, timeline, duration, needs=()):
    return {'id': sid, 'name': name, 'about': about, 'needs': list(needs), 'duration_s': float(duration),
            'params': {}, 'ego': {'speed_kmh': float(speed), 'driver': 'follower', 'route': 'straight'},
            'actors': actors, 'timeline': timeline, 'pass': {'no_collision': True}}


def _car(aid, lane, ahead, speed, lateral=0.0, kind='car'):
    return {'id': aid, 'kind': kind, 'place': {'lane': lane, 'ahead_m': round(float(ahead), 3),
                                              'lateral_m': round(float(lateral), 3)},
            'heading': 'lane', 'speed_kmh': float(speed), 'start': 'moving' if speed > 0 else 'stopped'}


def _stop_time(v_kmh, decel=4.0):
    return v_kmh / 3.6 / decel


def scn_stationary(sid, v, dist, lateral=0.0, name='Stationary vehicle in the lane'):
    return _scn(sid, name, f'A car stands in the ego\'s lane {dist:g} m ahead; the ego approaches at {v:g} km/h.', v,
                [_car('target', 'ego', dist, 0.0, lateral)], [], dist / max(v / 3.6, 1.0) + _stop_time(v) + 4.0)


def scn_moving(sid, v, vt, gap, lateral=0.0):
    ttc = gap / max((v - vt) / 3.6, 0.1)
    return _scn(sid, 'Slower vehicle ahead', f'A car drives at {vt:g} km/h in the ego\'s lane; the ego closes at '
                f'{v:g} km/h.', v, [_car('target', 'ego', gap, vt, lateral)], [], ttc + _stop_time(v - vt) + 4.0)


def scn_braking(sid, v, gap, decel, brake_at=2.0):
    return _scn(sid, 'Lead vehicle brakes', f'Both at {v:g} km/h, {gap:g} m apart; the lead brakes at {decel:g} m/s² '
                'to standstill.', v, [_car('target', 'ego', gap, v)],
                [{'trigger': {'type': 'time', 't_s': brake_at}, 'actions': [{'actor': 'target', 'do': 'brake',
                                                                            'decel': float(decel)}]}],
                brake_at + v / 3.6 / min(decel, 4.0) + 5.0)


def scn_follow(sid, v, duration, vary=True):
    gap = max(r157_following_m(v / 3.6) * 1.5, 10.0)
    tl = []
    if vary:
        tl = [{'trigger': {'type': 'time', 't_s': duration * 0.25},
               'actions': [{'actor': 'lead', 'do': 'speed', 'kmh': v * 0.5, 'accel': 2.0}]},
              {'trigger': {'type': 'time', 't_s': duration * 0.6},
               'actions': [{'actor': 'lead', 'do': 'speed', 'kmh': float(v), 'accel': 2.0}]}]
    return _scn(sid, 'Following a lead vehicle', f'A lead car at {v:g} km/h' + (', slowing to half and back' if vary
                else '') + '; the ego follows.', v, [_car('lead', 'ego', gap, v)], tl, duration)


def scn_cut_in(sid, ve, vo, dx0, vy):
    start = dx0 + (ve - vo) / 3.6 * 3.0                    # 3 s of approach before the lane change starts
    lt = math.hypot(dx0 + (EGO[0] + TARGET[0]) / 2.0, LANE_M)
    return _scn(sid, 'Cut-in', f'A car at {vo:g} km/h in the next lane changes into the ego\'s lane {dx0:g} m ahead '
                f'at {vy:g} m/s sideways; the ego drives {ve:g} km/h.', ve, [_car('cutter', 'adjacent', start, vo)],
                [{'trigger': {'type': 'distance', 'actor': 'cutter', 'lt_m': round(lt, 3)},
                  'actions': [{'actor': 'cutter', 'do': 'lane_change', 'to': 'ego',
                               'duration_s': round(LANE_M / vy, 3)}]}],
                3.0 + LANE_M / vy + _stop_time(ve) + 6.0, needs=('adjacent_same',))


def scn_cut_out(sid, ve, dx0_f, vy):
    gap = ve / 3.6 * R157['pm_thw']['value']
    far = gap + TARGET[0] + dx0_f + 2.0 * ve / 3.6          # 2 s of following before the lead leaves
    lt = gap + TARGET[0] + dx0_f + (EGO[0] + TARGET[0]) / 2.0
    return _scn(sid, 'Cut-out', f'The lead ({ve:g} km/h, {gap:.1f} m ahead) leaves the lane at {vy:g} m/s sideways '
                f'and reveals a stationary car {dx0_f:g} m ahead of it.', ve,
                [_car('lead', 'ego', gap, ve), _car('target', 'ego', far, 0.0)],
                [{'trigger': {'type': 'distance', 'actor': 'target', 'lt_m': round(lt, 3)},
                  'actions': [{'actor': 'lead', 'do': 'lane_change', 'to': 'adjacent',
                               'duration_s': round(LANE_M / vy, 3)}]}],
                2.0 + (gap + dx0_f) / max(ve / 3.6, 1.0) + _stop_time(ve) + 4.0, needs=('adjacent_same',))


def scn_pedestrian(sid, v, walk_kmh):
    dist = max(v / 3.6 * 4.0, 25.0)
    return _scn(sid, 'Pedestrian crossing', f'A pedestrian crosses at {walk_kmh:g} km/h; the ego drives {v:g} km/h.',
                v, [{'id': 'ped', 'kind': 'pedestrian', 'place': {'lane': 'sidewalk', 'ahead_m': round(dist, 2)},
                     'heading': 'across', 'speed_kmh': 0, 'start': 'stopped'}],
                [{'trigger': {'type': 'distance', 'actor': 'ped', 'lt_m': round(dist * 0.9, 2)},
                  'actions': [{'actor': 'ped', 'do': 'walk', 'speed_mps': round(walk_kmh / 3.6, 4)}]}],
                dist / max(v / 3.6, 1.0) + _stop_time(v) + 5.0)


# ====================================================================================================== the packs
def _case(cid, group, title, clause, source, scenario=None, criteria=(), values=None, mandatory=True, refuse='',
          **extra):
    return dict({'id': cid, 'group': group, 'title': title, 'clause': clause, 'source': source,
                 'scenario': scenario, 'criteria': list(criteria), 'values': values or {}, 'mandatory': mandatory,
                 'refuse': refuse}, **extra)


TS = 'Annex 5 leaves the specific test parameters to the Technical Service (para 1): this value is Falcon\'s choice'
NO_COLL = {'kind': 'no_collision'}


def _pack_r157() -> dict:
    cases = []
    vmax = R157['max_speed']['value']
    cases.append(_case('lane-keeping', 'lane keeping', 'Lane keeping behind a lead vehicle, 5 minutes',
                       'Annex 5, 4.1', 'r157', scn_follow('regpack-r157-lane-keeping', vmax,
                                                          R157['lane_keep_s']['value'], vary=False),
                       [{'kind': 'stay_in_lane', 'limit': C(0.0, 'm', 'the body does not cross its lane\'s edge '
                                                            '("does not leave its lane", 4.1.1): a limit of 0 m '
                                                            'beyond the lane width')}, NO_COLL],
                       {'duration': R157['lane_keep_s'], 'speed': C(vmax, 'km/h', TS)}, scalable=True))
    for v in (20.0, 40.0, 60.0):
        cases.append(_case(f'stationary-{v:g}', 'obstacle', f'Stationary passenger car, {v:g} km/h',
                           'Annex 5, 4.2.2(a); 5.2.4', 'r157',
                           scn_stationary(f'regpack-r157-stationary-{v:g}', v, max(v / 3.6 * 6.0, 40.0)), [NO_COLL],
                           {'speed': C(v, 'km/h', TS), 'max_speed': R157['max_speed']}))
    for v in (30.0, 60.0):
        cases.append(_case(f'pedestrian-{v:g}', 'obstacle', f'Pedestrian crossing at 5 km/h, {v:g} km/h',
                           'Annex 5, 4.2.2(d); 5.2.5.3', 'r157',
                           scn_pedestrian(f'regpack-r157-pedestrian-{v:g}', v, R157['ped_speed']['value']),
                           [NO_COLL], {'ped_speed': R157['ped_speed'], 'speed': C(v, 'km/h', TS)}))
    for v in (20.0, 40.0, 60.0):
        cases.append(_case(f'following-{v:g}', 'following', f'Following distance, lead at {v:g} km/h slowing and '
                           'recovering', 'Annex 5, 4.3; 5.2.3.3', 'r157',
                           scn_follow(f'regpack-r157-following-{v:g}', v, min(40.0, 0.9 * ROUTE_M / (v / 3.6))),
                           [{'kind': 'following_distance', 'limit': R157['following_table'],
                             'steady': C(0.5, 'm/s²', 'judged only while the lead has been steady (|a| below this) '
                                         'for 2 s: 5.2.3.3 lets the distance be restored after a disturbance')},
                            NO_COLL],
                           {'table': R157['following_table'], 'floor': R157['following_floor'],
                            'speed': C(v, 'km/h', TS)}))
        avoid = pm1_decel_avoidable(v, R157['lead_decel']['value'])
        cases.append(_case(f'lead-brake-{v:g}', 'following', f'Lead brakes at 6 m/s² to standstill from {v:g} km/h',
                           'Annex 5, 4.3.2(f); 5.2.5.1', 'r157',
                           scn_braking(f'regpack-r157-lead-brake-{v:g}', v, v / 3.6 * R157['pm_thw']['value'],
                                       R157['lead_decel']['value']), [NO_COLL],
                           {'decel': R157['lead_decel'], 'headway': R157['pm_thw'], 'speed': C(v, 'km/h', TS)},
                           pm1_avoidable=avoid))
    for ve, vo, dx0, vy in ((40.0, 20.0, 30.0, 1.0), (40.0, 20.0, 12.0, 1.0), (60.0, 40.0, 30.0, 1.0),
                            (60.0, 40.0, 15.0, 2.0), (60.0, 20.0, 40.0, 1.0), (60.0, 20.0, 10.0, 2.0)):
        avoid = pm1_cut_in_avoidable(ve, vo, dx0, vy)
        req = r157_cut_in_required(ve, vo, dx0, vy)
        must = avoid or req
        cid = f'cut-in-{ve:g}-{vo:g}-{dx0:g}-{vy:g}'
        cases.append(_case(cid, 'cut-in', f'Cut-in: ego {ve:g}, other {vo:g} km/h, {dx0:g} m ahead, {vy:g} m/s',
                           'Annex 5, 4.4; 5.2.5.2; Annex 4 Appendix 3', 'r157',
                           scn_cut_in(f'regpack-r157-{cid}', ve, vo, dx0, vy), [NO_COLL],
                           {'ego_speed': C(ve, 'km/h', TS), 'other_speed': C(vo, 'km/h', TS),
                            'dx0': C(dx0, 'm', TS), 'vy': C(vy, 'm/s', TS), 'wander': R157['pm_wander'],
                            'perception': R157['pm_perception'], 'delay': R157['pm_delay'],
                            'jerk_time': R157['pm_jerk_s'], 'decel': R157['pm_decel_g'], 'dy0': R157['pm_dy0'],
                            'visible': R157['cut_in_visible'], 'ttc_decel': R157['cut_in_decel'],
                            'ttc_margin': R157['cut_in_margin']},
                           mandatory=must, pm1_avoidable=avoid, required_5252=req,
                           note='' if must else 'performance model 1 does not avoid this cut-in and 5.2.5.2 does '
                           'not require it: run and reported, a collision is not a failure'))
    for ve, dx0_f, vy in ((40.0, 40.0, 1.0), (60.0, 60.0, 1.0), (60.0, 30.0, 2.0)):
        avoid = pm1_cut_out_avoidable(ve, dx0_f, vy)
        cid = f'cut-out-{ve:g}-{dx0_f:g}-{vy:g}'
        cases.append(_case(cid, 'cut-out', f'Cut-out: {ve:g} km/h, stationary car {dx0_f:g} m ahead of the lead, '
                           f'{vy:g} m/s', 'Annex 5, 4.5; Annex 4 Appendix 3', 'r157',
                           scn_cut_out(f'regpack-r157-{cid}', ve, dx0_f, vy), [NO_COLL],
                           {'speed': C(ve, 'km/h', TS), 'dx0_f': C(dx0_f, 'm', TS), 'vy': C(vy, 'm/s', TS),
                            'headway': R157['pm_thw'], 'wander': R157['pm_wander']},
                           mandatory=avoid, pm1_avoidable=avoid,
                           note='' if avoid else 'performance model 1 does not avoid this cut-out: informative'))
    cases.append(_case('mrm', 'minimum risk manoeuvre', 'Minimum risk manoeuvre after a severe sensor failure',
                       '5.5; 5.4.4.1.1', 'r157', scn_follow('regpack-r157-mrm', 40.0, 30.0, vary=False),
                       [{'kind': 'mrm', 'limit': R157['mrm_decel']}],
                       {'decel': R157['mrm_decel'], 'transition': R157['transition_s'],
                        'speed': C(40.0, 'km/h', TS),
                        'other_checks': C(None, '', 'reaction, standstill, jerk and lane limits are core.mrm\'s '
                                          'heuristics, not the regulation\'s; the 10 s transition phase is not '
                                          'simulated (a severe failure: the MRM may start immediately)')},
                       fault={'kind': 'dropout', 'onset_s': 5.0}))
    return {'id': 'r157', 'title': 'UN Regulation No. 157 (ALKS) — pre-check up to 60 km/h', 'kind': 'test',
            'sources': ['r157', 'r157_01'], 'cases': cases, 'numbers': R157,
            'about': 'Lane keeping, obstacles, following, cut-in and cut-out against performance model 1, and the '
                     'minimum risk manoeuvre. The verdict needs every mandatory case run and passed.',
            'limits': ['Annex 5 has no numeric test parameters: speeds, gaps and lateral velocities are Falcon\'s '
                       'selection inside the regulation\'s ranges',
                       'performance model 1 is implemented from its parameter table; the regulation\'s data sheets '
                       'were not reproduced point by point; performance model 2 (01 series, fuzzy) is not '
                       'implemented',
                       'no transition demand, driver monitoring, HMI, field-of-view, PTW or lane-change tests',
                       'the 5-minute lane keeping test cannot be run at its duration: the scenario runner\'s ego '
                       'route is 400 m of one lane, so the pack\'s verdict is never better than incomplete; '
                       'curves are not tested (scenarios are placed on straight lanes)',
                       'amendments 1–3 to the original series and supplements after the 01 series were not read']}


# Euro NCAP AEB C2C 2023: bands = the upper bound (km/h) of green, yellow, orange, brown for each test speed; None: no
# such band. The 50 km/h row is a text table of the assessment protocol; the others were measured from its bar chart.
BAND_NAMES = ('green', 'yellow', 'orange', 'brown', 'red')
BAND_SCORE = V({'green': 1.0, 'yellow': 0.75, 'orange': 0.5, 'brown': 0.25, 'red': 0.0}, 'factor', 'ncap_sa_ap',
               '3.3.2')
CCRS_BANDS = {80: (5, 25, 50, 70), 75: (5, 25, 45, 65), 70: (5, 20, 40, 60), 65: (5, 20, 40, 55),
              60: (5, 20, 35, 50), 55: (5, 15, 30, 45), 50: (5, 15, 30, 40), 45: (5, 15, 25, 35),
              40: (5, 15, 25, 35), 35: (5, 15, 25, None), 30: (5, 15, 25, None), 25: (5, None, 15, None),
              20: (None, None, None, None), 15: (None, None, None, None), 10: (None, None, None, None)}
CCRM_BANDS = {80: (5, 20, 35, 50), 75: (5, 15, 30, 45), 70: (5, 15, 30, 40), 65: (5, 15, 25, 35),
              60: (5, 15, 25, 35), 55: (5, 15, 25, None), 50: (5, 15, 25, None), 45: (5, None, 15, None),
              40: (5, None, 15, None), 35: (5, None, None, None), 30: (5, None, None, None)}
CCRS_POINTS = V({10: 1, 15: 2, 20: 2, 25: 2, 30: 2, 35: 2, 40: 1, 45: 1, 50: 1}, 'points', 'ncap_sa_ap', '3.3.2',
                note='AEB CCRs, total 14')
CCRM_POINTS = V({30: 1, 35: 1, 40: 1, 45: 1, 50: 1, 55: 1, 60: 1, 65: 2, 70: 2, 75: 2, 80: 2}, 'points',
                'ncap_sa_ap', '3.3.2', note='AEB CCRm, total 15')
OVERLAPS = V([-50, -75, 100, 75, 50], '%', 'ncap_sa_ap', '3.3.2',
             note='per test speed: (−50 + −75 + 2 × 100 + 75 + 50) / 6; the 100 % overlap is double counted')
NCAP = {
    'ccrs_speeds': V([10, 50, 5], 'km/h [from, to, step]', 'ncap_aeb_tp', '8.2.2.1', note='the AEB range of an AEB + '
                     'FCW system; 55–80 km/h are FCW tests'),
    'ccrm_speeds': V([30, 80, 5], 'km/h [from, to, step]', 'ncap_aeb_tp', '8.2.2.2'),
    'ccrm_target': V(20.0, 'km/h', 'ncap_aeb_tp', '8.2.2.2'),
    'ccrb_speed': V(50.0, 'km/h', 'ncap_aeb_tp', '8.2.2.3'),
    'ccrb_headways': V([12.0, 40.0], 'm', 'ncap_aeb_tp', '8.2.2.3'),
    'ccrb_decels': V([2.0, 6.0], 'm/s²', 'ncap_aeb_tp', '8.2.2.3'),
    't0_ttc': V(4.0, 's', 'ncap_aeb_tp', '4.1.1', note='T0 = TTC 4 s'),
    'overlaps': OVERLAPS, 'band_score': BAND_SCORE, 'ccrs_points': CCRS_POINTS, 'ccrm_points': CCRM_POINTS,
    'bands_50': V([5, 15, 30, 40], 'km/h', 'ncap_sa_ap', '3.3.2', note='CCRs and CCRb at 50 km/h: green < 5, yellow '
                  '< 15, orange < 30, brown < 40, red from 40'),
    'bands_ccrs': V({k: list(v) for k, v in CCRS_BANDS.items()}, 'km/h', 'ncap_sa_ap', '3.3.2', 'figure',
                    'CCRs/CCRb impact-speed bands per test speed; only the 50 km/h row is printed as numbers'),
    'bands_ccrm': V({k: list(v) for k, v in CCRM_BANDS.items()}, 'km/h', 'ncap_sa_ap', '3.3.2', 'figure',
                    'CCRm relative-impact-speed bands per test speed'),
    'scenario_points': V({'CCRs': 1.0, 'CCRm': 1.0, 'CCRb': 1.0}, 'points', 'ncap_sa_ap', '3.3.2',
                         note='AEB; FCW CCRs 0.5 is not assessed here'),
    'low_speed_avoid': V(20.0, 'km/h', 'ncap_sa_ap', '3.3', note='full avoidance needed up to and including this '
                         'test speed'),
    'overlap_sign': V('negative overlap towards the driver\'s side', '', 'ncap_aeb_tp', 'Figure 3-3', 'recalled',
                      'which side a negative overlap is on; the grid here is run mirror-symmetrically'),
}


def band(impact_kmh: float | None, bounds) -> str:
    """The colour band of an impact speed (None or 0: avoided, green) for a test speed's bounds."""
    if impact_kmh is None or impact_kmh <= 0.0:
        return 'green'
    for name, b in zip(BAND_NAMES, bounds):
        if b is not None and impact_kmh < b:
            return name
    return 'red'


def overlap_offset(overlap_pct: float) -> float:
    """The target's lateral offset (m) for an overlap: the share of the ego's width overlapping the target
    (test protocol 3.4.1), with Falcon's stand-in widths; the sign follows the overlap's."""
    o = abs(overlap_pct) / 100.0
    off = (EGO[1] + TARGET[1]) / 2.0 - o * EGO[1]
    return math.copysign(max(off, 0.0), -1.0 if overlap_pct < 0 else 1.0) if o < 1.0 else 0.0


def _pack_aeb() -> dict:
    cases = []
    ttc0 = NCAP['t0_ttc']['value']
    a, b, st = NCAP['ccrs_speeds']['value']
    for v in range(a, b + 1, st):
        for o in OVERLAPS['value']:
            cid = f'ccrs-{v}-{o}'
            cases.append(_case(cid, 'CCRs', f'CCRs {v} km/h, overlap {o} %', '8.2.2.1', 'ncap_aeb_tp',
                               scn_stationary(f'regpack-ncap-{cid}', float(v), v / 3.6 * (ttc0 + 2.0),
                                              overlap_offset(o), 'CCRs: stationary target'),
                               [{'kind': 'impact_band', 'bands': list(CCRS_BANDS[v]), 'relative': False}],
                               {'speed': NCAP['ccrs_speeds'], 'overlap': OVERLAPS,
                                'bands': NCAP['bands_50'] if v == 50 else NCAP['bands_ccrs']},
                               speed=v, overlap=o))
    for v in range(55, 81, 5):
        cases.append(_case(f'ccrs-fcw-{v}', 'CCRs FCW', f'CCRs FCW {v} km/h', '8.2.2.1; 8.4.6', 'ncap_aeb_tp',
                           refuse='FCW tests need the protocol\'s braking-robot driver (reacts 1.2 s after the '
                                  'warning with −4 m/s²): Falcon simulates no warned driver', mandatory=False))
    a, b, st = NCAP['ccrm_speeds']['value']
    vt = NCAP['ccrm_target']['value']
    for v in range(a, b + 1, st):
        for o in OVERLAPS['value']:
            cid = f'ccrm-{v}-{o}'
            cases.append(_case(cid, 'CCRm', f'CCRm {v} km/h, target 20 km/h, overlap {o} %', '8.2.2.2',
                               'ncap_aeb_tp', scn_moving(f'regpack-ncap-{cid}', float(v), vt,
                                                         (v - vt) / 3.6 * (ttc0 + 2.0), overlap_offset(o)),
                               [{'kind': 'impact_band', 'bands': list(CCRM_BANDS[v]), 'relative': True}],
                               {'speed': NCAP['ccrm_speeds'], 'target_speed': NCAP['ccrm_target'],
                                'overlap': OVERLAPS, 'bands': NCAP['bands_ccrm']}, speed=v, overlap=o))
    for hw in NCAP['ccrb_headways']['value']:
        for d in NCAP['ccrb_decels']['value']:
            cid = f'ccrb-{hw:g}-{d:g}'
            cases.append(_case(cid, 'CCRb', f'CCRb 50 km/h, headway {hw:g} m, target −{d:g} m/s²', '8.2.2.3',
                               'ncap_aeb_tp', scn_braking(f'regpack-ncap-{cid}', NCAP['ccrb_speed']['value'], hw, d),
                               [{'kind': 'impact_band', 'bands': list(CCRS_BANDS[50]), 'relative': False}],
                               {'speed': NCAP['ccrb_speed'], 'headway': NCAP['ccrb_headways'],
                                'decel': NCAP['ccrb_decels'], 'bands': NCAP['bands_50']}, speed=50, overlap=100))
    return {'id': 'ncap_aeb_c2c', 'title': 'Euro NCAP AEB Car-to-Car 2023 (CCRs, CCRm, CCRb) — pre-check',
            'kind': 'scored', 'sources': ['ncap_aeb_tp', 'ncap_sa_ap'], 'cases': cases, 'numbers': NCAP,
            'about': 'Each grid point is scored by the colour band of its impact speed; per speed the overlaps are '
                     'weighted, the speeds carry points, and each scenario is worth 1 point (3 in all here).',
            'limits': ['the target is Falcon\'s stand-in car, not the Global Vehicle Target; its width and the '
                       'ego\'s set the overlap offset',
                       'no manufacturer prediction, so no correction factor; no verification tolerance (2 km/h)',
                       'FCW, CCFtap, CCCscp, head-on and HMI points are not assessed: at most 3 of the 9 points of '
                       'AEB Car-to-Car are covered',
                       'the protocol\'s tolerances (speed +1 km/h, lateral 0 ± 0.05 m) are not checked on the runs',
                       'the eligibility conditions (default on, whiplash rating, 130 km/h evidence) do not apply '
                       'to a simulation',
                       'the 2026 protocols (Crash Avoidance — Frontal Collisions) are not what is implemented']}


LSS = {
    'speed': V(72.0, 'km/h', 'ncap_lss_tp', '7.4.1'),
    'vlat': V([0.2, 0.3, 0.4, 0.5, 0.6], 'm/s', 'ncap_lss_tp', '7.2.5.1', note='0.1 m/s steps, both sides'),
    'lka_dtle': V(-0.3, 'm', 'ncap_sa_ap', '4.3.2', note='must not cross the inner edge of the lane marking by '
                  'more than 0.3 m'),
    'elk_edge_dtle': V(-0.1, 'm', 'ncap_sa_ap', '4.3.3.3'),
    'elk_solid_dtle': V(-0.3, 'm', 'ncap_sa_ap', '4.3.3.3'),
    'elk_target_speed': V(72.0, 'km/h', 'ncap_lss_tp', '7.4.3', note='oncoming target'),
    'elk_overtake': V([72.0, 80.0], 'km/h', 'ncap_lss_tp', '7.2.4.4', note='target at the ego\'s speed, or '
                      'overtaking at 80 km/h'),
    'elk_overtake_vlat': V([0.5, 0.6, 0.7], 'm/s', 'ncap_lss_tp', '7.2.4.4', note='intentional lane change'),
    'lane_width': V([3.5, 3.7], 'm', 'ncap_lss_tp', '6.1.3.1'),
    'yaw_table': V({0.2: 0.57, 0.3: 0.86, 0.4: 1.15, 0.5: 1.43, 0.6: 1.72}, '° by m/s', 'ncap_lss_tp', '7.2.3',
                   note='the approach yaw angle per lateral velocity'),
    'points': V({'HMI': 0.5, 'LKA': 0.5, 'ELK': 2.0}, 'points', 'ncap_sa_ap', '4.3',
                note='LKA dashed 0.25 + solid 0.25; ELK road edge 0.25 + 0.25, solid 0.5, oncoming 0.5, overtaking '
                     '0.5; all tests of a scenario and marking must pass'),
}
LSS_REFUSE = ('the test steers the ego onto a lane-departure path (a yaw angle per lateral velocity, test protocol '
              '7.2.3): Falcon\'s scenario language has no action that imposes a path or a steering disturbance on '
              'the ego')


def _pack_lss() -> dict:
    cases = []
    for name, group, clause, lim in (('lka-dashed', 'LKA', '7.2.5.1', 'lka_dtle'),
                                     ('lka-solid', 'LKA', '7.2.5.2', 'lka_dtle'),
                                     ('elk-road-edge', 'ELK', '7.2.4.1', 'elk_edge_dtle'),
                                     ('elk-solid', 'ELK', '7.2.4.2', 'elk_solid_dtle'),
                                     ('elk-oncoming', 'ELK', '7.2.4.3', None),
                                     ('elk-overtaking', 'ELK', '7.2.4.4', None)):
        for vl in LSS['vlat']['value']:
            vals = {'speed': LSS['speed'], 'vlat': LSS['vlat'], 'yaw': LSS['yaw_table']}
            if lim:
                vals['limit'] = LSS[lim]
            cases.append(_case(f'{name}-{vl:g}', group, f'{name.replace("-", " ")}, {vl:g} m/s', clause,
                               'ncap_lss_tp', values=vals, refuse=LSS_REFUSE))
    return {'id': 'ncap_lss', 'title': 'Euro NCAP Lane Support Systems 2023 (LKA, ELK) — data only', 'kind': 'scored',
            'sources': ['ncap_lss_tp', 'ncap_sa_ap'], 'cases': cases, 'numbers': LSS,
            'about': 'The LKA and ELK grids with their limits. Not runnable in this build: every case is refused '
                     'by name.',
            'limits': ['no case runs: the ego cannot be put on the protocol\'s departure path',
                       'the 2026 protocols (Lane Departure Collisions) are not what is listed']}


ISO_NOTE = ('the breakdown inside each category is the public JAMA framework\'s (Ver 3.0), not read from ISO 34502, '
            'whose annex bodies are paywalled')
ISO = {
    'annex_b': V('Annex B (informative): Traffic-related critical scenarios', '', 'iso34502', 'contents'),
    'annex_c': V('Annex C (informative): Perception-related critical scenarios', '', 'iso34502', 'contents'),
    'annex_d': V('Annex D (informative): Vehicle control related critical scenarios', '', 'iso34502', 'contents'),
    'scope': V('applicable for limited access highways', '', 'iso34502', 'scope (catalogue abstract)'),
    'traffic_matrix': V(24, 'combinations', 'jama_v3', '4.1.1.4', note='motorways: 3 road geometries × 2 ego '
                        'behaviours × 4 surrounding-vehicle motions; ' + ISO_NOTE),
    'same_in_iso': V('ISO 34502 Annex B holds the same 24-combination matrix', '', 'iso34502', 'Annex B', 'recalled',
                     'the annex body was not read'),
}


def _pack_iso() -> dict:
    items = []
    cover = {('straight', 'going straight', 'decelerating ahead'): (['lead_brake', 'stalled_vehicle'], 'covered'),
             ('straight', 'going straight', 'cut-in'): (['cut_in', 'autorickshaw_cut_in'], 'covered'),
             ('straight', 'going straight', 'cut-out'): (['regpack r157 cut-out cases'], 'covered'),
             ('straight', 'going straight', 'accelerating behind'): ([], 'gap')}
    for geo in ('straight', 'merging', 'branching'):
        for ego in ('going straight', 'lane change'):
            for other in ('accelerating behind', 'decelerating ahead', 'cut-in', 'cut-out'):
                scn, st = cover.get((geo, ego, other), ([], 'gap'))
                why = '' if st == 'covered' else (
                    'no scenario places a vehicle closing from behind' if geo == 'straight' and ego ==
                    'going straight' else 'the scenario ego does not change lanes' if ego == 'lane change' else
                    f'no scenario is placed on a {geo} section')
                items.append({'id': f'traffic-{geo}-{ego}-{other}'.replace(' ', '-'), 'category': 'traffic',
                              'item': f'{geo} road, ego {ego}, other vehicle {other}', 'scenarios': scn,
                              'status': st, 'note': why, 'source': 'jama_v3', 'clause': '4.1.1.4'})
    for iid, item, clause, scn, st, note in (
            ('perception-sensor', 'vehicle / sensor: the sensor itself and what is in front of it', '4.2.1.1',
             ['falcon faulttol (misalignment, clock, dirty lens, noise)', 'core.faults (rain, glare, occlusion)'],
             'covered', 'as fault campaigns, not scenarios'),
            ('perception-environment', 'surrounding environment: structures, space (weather), moving objects',
             '4.2.1.1', ['monsoon_waterlogged', 'weather sweeps (fog, rain; core.sensorphysics)'], 'partial',
             'weather yes; reflective structures and radar multipath are not modelled'),
            ('perception-target', 'perception target: route, traffic information, obstacles, moving objects',
             '4.2.1.1', ['stray_cattle', 'handcart_kerb_lane', 'pedestrian_crossing'], 'partial',
             'unusual obstacles yes; lane-marking and sign degradation are not scenarios'),
            ('perception-blind-spot', 'blind spot: peripheral vehicles, road structure, road shape', '4.2.2',
             ['jaywalker_behind_bus', 'bus_pulling_out'], 'partial', 'occlusion by vehicles only'),
            ('perception-connectivity', 'connectivity disturbance: sensors, environment, transmitter', '4.2.3',
             ['core.v2x (V2X messages)'], 'partial', 'V2X messages only; no digital-map error'),
            ('control-body-road', 'vehicle body input: road shape (slopes, curvature)', '4.3.1', [], 'gap',
             'no scenario varies slope or curvature'),
            ('control-body-wind', 'vehicle body input: natural phenomena (cross, tail and head wind)', '4.3.1', [],
             'gap', 'wind forces are not simulated'),
            ('control-tyre-surface', 'tyre input: road surface (friction, external force)', '4.3.2',
             ['monsoon_waterlogged', 'speed breakers (core.roadfeatures)'], 'partial',
             'wetness and breakers; friction is CARLA\'s, not measured'),
            ('control-tyre-condition', 'tyre input: tyre condition (puncture, burst, wear)', '4.3.2', [], 'gap',
             'not simulated')):
        items.append({'id': iid, 'category': iid.split('-')[0] if not iid.startswith('control') else
                      'vehicle control', 'item': item, 'scenarios': scn, 'status': st, 'note': note,
                      'source': 'jama_v3', 'clause': clause})
    return {'id': 'iso34502', 'title': 'ISO 34502:2022 critical-scenario categories — coverage checklist',
            'kind': 'checklist', 'sources': ['iso34502', 'jama_v3'], 'cases': [], 'items': items, 'numbers': ISO,
            'about': 'Which of the critical-scenario categories Falcon\'s scenarios and features touch. A '
                     'checklist of coverage, not a test: nothing is run and nothing passes.',
            'limits': [ISO_NOTE, 'ISO 34502 applies to limited access highways; Falcon\'s built-in scenarios are '
                       'mostly urban', '"covered" means a Falcon scenario of that kind exists, not that the '
                       'category\'s parameter space was explored']}


_BUILDERS = {'r157': _pack_r157, 'ncap_aeb_c2c': _pack_aeb, 'ncap_lss': _pack_lss, 'iso34502': _pack_iso}
PACKS = tuple(_BUILDERS)
_CACHE: dict = {}


def pack(pid: str) -> dict:
    if pid not in _BUILDERS:
        raise RegPackError(f'no pack {pid!r}: one of {", ".join(PACKS)}')
    if pid not in _CACHE:
        _CACHE[pid] = _BUILDERS[pid]()
    return copy.deepcopy(_CACHE[pid])


def _is_value(v) -> bool:
    return isinstance(v, dict) and {'value', 'source', 'verified', 'basis'} <= set(v)


def values_of(p: dict) -> list:
    """Every value record of a pack, once: [(name, record)]."""
    seen, out = set(), []
    for k, v in (p.get('numbers') or {}).items():
        if _is_value(v):
            out.append((k, v))
            seen.add(json.dumps(v, sort_keys=True, default=str))
    for c in p.get('cases') or []:
        recs = list((c.get('values') or {}).items()) + [(f'{cr["kind"]}.{k}', x) for cr in c.get('criteria') or []
                                                         for k, x in cr.items()]
        for k, v in recs:
            if _is_value(v):
                key = json.dumps(v, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    out.append((f'{c["id"]}.{k}', v))
    return out


def unverified(p: dict) -> list:
    """The pack's values that were not read as numbers in the document's text, each with why."""
    out = []
    for name, v in values_of(p):
        if v['verified'] is False:
            val = json.dumps(v['value'], default=str)
            out.append({'name': name, 'value': val if len(val) <= 80 else val[:77] + '…', 'unit': v['unit'],
                        'source': v['source'], 'clause': v['clause'], 'basis': v['basis'],
                        'why': BASIS_TEXT.get(v['basis'], v['basis']), 'note': v.get('note', '')})
    return out


def choices(p: dict) -> list:
    """The pack's values that are Falcon's own (not any document's), once per distinct reason."""
    seen, out = set(), []
    for name, v in values_of(p):
        if v['verified'] is None and v['note'] not in seen:
            seen.add(v['note'])
            out.append({'name': name, 'note': v['note']})
    return out


def runnable(c: dict) -> bool:
    return bool(c.get('scenario')) and not c.get('refuse')


# ================================================================================ how much an unverified value matters
SENS_BAND = 0.10        # the band tried around a value not verified in its document when it gives no range (4.2)


def value_band(v: dict, rel: float = SENS_BAND) -> tuple:
    """(low, high, how it was chosen) of a value record: the range the record gives (`range`), else ±rel of a number
    (a dict or list of numbers: each scaled), else None for a value that is not a number."""
    if v.get('range'):
        lo, hi = v['range']
        return lo, hi, 'the range the document gives'

    def scale(x, f):
        if isinstance(x, bool) or x is None:
            return x
        if isinstance(x, (int, float)):
            return round(x * f, 6)
        if isinstance(x, list):
            return [scale(y, f) for y in x]
        if isinstance(x, dict):
            return {k: scale(y, f) for k, y in x.items()}
        raise TypeError
    try:
        return (scale(v['value'], 1.0 - rel), scale(v['value'], 1.0 + rel),
                f'±{100 * rel:g} % of the value (the document gives no range)')
    except TypeError:
        return None, None, 'not a number: no band to try'


def cases_using(p: dict, name: str) -> list:
    """The ids of the pack's cases whose values include the value record `name` (a name of values_of)."""
    rec = dict(values_of(p)).get(name)
    if rec is None:
        return []
    if '.' in name and name.split('.', 1)[0] in {c['id'] for c in p['cases']}:
        return [name.split('.', 1)[0]]
    return [c['id'] for c in p['cases'] if any(x == rec for x in (c.get('values') or {}).values())]


def _case_result(must: bool, row: dict | None):
    """A cut-in row's result word with its classification `must`, from its measured collision (None: not run)."""
    if not row or row.get('result') not in ('pass', 'fail', 'informative'):
        return None
    if not (row.get('measured') or {}).get('collision'):
        return 'pass'
    return 'fail' if must else 'informative'


def sensitivity(pid: str, name: str, doc: dict | None = None, rel: float = SENS_BAND) -> dict:
    """Re-evaluate what an unverified value of a pack decides at the low and high ends of its band (value_band) and
    say whether the verdict flips (4.2, a pure function: no simulation is run — a run case is re-judged from what
    was measured). doc: a result of the pack (None: the pack's classification only).

    What is re-evaluated: R157 pm_dy0 — performance model 1's classification of each cut-in (mandatory or
    informative) and with it the case's result and the pack's verdict; Euro NCAP bands_ccrs / bands_ccrm — each run
    case's colour band from its measured impact speed with the band bounds scaled, and the AEB score; overlap_sign —
    both signs (the grid is run mirror-symmetrically, so a sign swaps ±overlap cases). A value no check reads, or a
    statement that is not a number, says so and cannot flip anything."""
    p = pack(pid)
    rec = dict(values_of(p)).get(name)
    if rec is None:
        raise RegPackError(f'pack {pid} has no value {name!r}')
    lo, hi, how = value_band(rec, rel)
    using = cases_using(p, name)
    out = {'pack': pid, 'name': name, 'value': rec['value'], 'unit': rec['unit'], 'source': rec['source'],
           'document': SOURCES.get(rec['source'], {}).get('title', rec['source']), 'clause': rec['clause'],
           'basis': rec['basis'], 'verified': rec['verified'], 'low': lo, 'high': hi, 'band': how, 'cases': [],
           'verdict': None, 'flips': False, 'applicable': True, 'checked': '', 'run': doc is not None}
    rows = {r['id']: r for r in (doc or {}).get('cases') or []}
    whole = bool(doc) and (doc.get('inputs') or {}).get('cases_selected') == len(p['cases'])
    if name == 'pm_dy0':
        out['checked'] = 'performance model 1\'s cut-in classification (mandatory / informative)'
        alt = {}
        for end, dy in (('nominal', rec['value']), ('low', lo), ('high', hi)):
            alt[end] = {}
            for c in p['cases']:
                if c['id'] not in using:
                    continue
                v = c['values']
                avoid = pm1_cut_in_avoidable(v['ego_speed']['value'], v['other_speed']['value'], v['dx0']['value'],
                                             v['vy']['value'], dy)
                alt[end][c['id']] = avoid or c.get('required_5252', False)
        for cid in using:
            words = {e: ('mandatory' if alt[e][cid] else 'informative') for e in alt}
            res = {e: _case_result(alt[e][cid], rows.get(cid)) for e in alt}
            row = {'id': cid, **{e: words[e] + (f', {res[e]}' if res[e] else '') for e in alt}}
            row['flips'] = len({row[e] for e in alt}) > 1
            out['cases'].append(row)
        if doc is not None:
            ver = {}
            for end in alt:
                d2 = copy.deepcopy(doc)
                for r in d2['cases']:
                    if r['id'] in alt[end]:
                        r['mandatory'] = alt[end][r['id']]
                        got = _case_result(r['mandatory'], r)
                        if got:
                            r['result'] = got
                ver[end] = verdict_of(p, d2, whole)['verdict']
            out['verdict'] = ver
    elif name in ('bands_ccrs', 'bands_ccrm') or name.endswith('.bands'):
        out['checked'] = 'the colour band of each run case\'s impact speed, and the AEB score'
        if doc is not None:
            alt = {'nominal': copy.deepcopy(doc['cases']), 'low': copy.deepcopy(doc['cases']),
                   'high': copy.deepcopy(doc['cases'])}
            by_case = {c['id']: c for c in p['cases']}
            for cid in using:
                r, c = rows.get(cid), by_case[cid]
                if not r or r.get('factor') is None:
                    continue
                cr = next(x for x in c['criteria'] if x['kind'] == 'impact_band')
                m = r.get('measured') or {}
                v_imp = m.get('relative_impact_kmh') if cr.get('relative') else m.get('impact_speed_kmh')
                words = {}
                for end, f in (('nominal', 1.0), ('low', 1.0 - rel), ('high', 1.0 + rel)):
                    words[end] = band(v_imp, [None if b is None else b * f for b in cr['bands']])
                    for x in alt[end]:
                        if x['id'] == cid:
                            x['factor'] = BAND_SCORE['value'][words[end]]
                out['cases'].append({'id': cid, **words, 'flips': len(set(words.values())) > 1})
            sc = {e: aeb_score(v)['points'] for e, v in alt.items()}
            out['verdict'] = {e: ('—' if v is None else f'{v:g} of 3 points') for e, v in sc.items()}
    elif name == 'overlap_sign':
        out['low'], out['high'] = 'as recalled', 'the other side'
        out['band'] = 'a direction: both signs tried (no ±10 % band for a sign)'
        out['checked'] = 'the AEB score with every ±overlap pair of results swapped'
        if doc is not None:
            sw = copy.deepcopy(doc['cases'])
            fac = {r['id']: r.get('factor') for r in doc['cases']}
            for r in sw:
                if r.get('overlap') not in (None, 100) and r.get('speed') is not None:
                    mirror = f'{r["id"].rsplit("-", 1)[0]}-{-int(r["overlap"])}'
                    r['factor'] = fac.get(mirror)
            a, b = aeb_score(doc['cases'])['points'], aeb_score(sw)['points']
            out['verdict'] = {'nominal': '—' if a is None else f'{a:g} of 3 points', 'low': '—' if a is None else
                              f'{a:g} of 3 points', 'high': '—' if b is None else f'{b:g} of 3 points'}
    elif lo is None:
        out.update(applicable=False, checked='a statement, not a number: there is no band to try and no verdict '
                                             'reads it as a number')
    elif not using:
        out['checked'] = 'no check of this pack reads it: its band cannot change a case or the verdict'
    else:
        out.update(applicable=False, checked='no re-evaluation is implemented for this value')
    if isinstance(out['verdict'], dict):
        out['flips'] = len(set(out['verdict'].values())) > 1
    out['flips'] = out['flips'] or any(c['flips'] for c in out['cases'])
    return out


def sensitivity_text(s: dict) -> str:
    """One line per finding of sensitivity(), for the command line and a tooltip."""
    def fmt(x):
        t = json.dumps(x, default=str) if not isinstance(x, str) else x
        return t if len(t) <= 60 else t[:57] + '…'
    tried = f'tried {fmt(s["low"])} and {fmt(s["high"])}' if s['low'] is not None else 'nothing to try'
    head = f'{s["name"]} = {fmt(s["value"])} {s["unit"]}'.rstrip() + \
        f' ({s["document"]}, {s["clause"]}): {tried} — {s["band"]}'
    lines = [head, f're-evaluated: {s["checked"]}' if s['checked'] else '']
    for c in s['cases']:
        lines.append(f'  {c["id"]}: {c["nominal"]} → low {c["low"]}, high {c["high"]}' +
                     (' FLIPS' if c['flips'] else ''))
    if s['verdict']:
        v = s['verdict']
        lines.append(f'verdict: {v["nominal"]} → low {v["low"]}, high {v["high"]}')
    lines.append(('the verdict FLIPS inside the band' if s['flips'] else 'nothing flips inside the band')
                 if s['applicable'] else 'not applicable')
    return '\n'.join(x for x in lines if x)


# ========================================================================================================= scoring
def _traj(run_dir: str) -> dict:
    import numpy as np
    with open(os.path.join(run_dir, 'trajectories.json'), encoding='utf-8') as fh:
        d = json.load(fh)
    return {k: np.asarray(v, float) for k, v in (d.get('actors') or {}).items() if len(v)}


def _frames(run_dir: str) -> list:
    with open(os.path.join(run_dir, 'frames.jsonl'), encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def impact_speed_kmh(run: dict, run_dir: str) -> tuple:
    """(the ego's speed, the relative speed) in km/h at the first contact, or (None, None) without one."""
    import numpy as np
    ov = ((run.get('scenario') or {}).get('criticality') or {}).get('overall') or {}
    if not ov.get('collision'):
        return None, None
    tr = _traj(run_dir)
    ego = tr['ego']
    t = ov.get('collision_t')
    k = int(np.argmin(np.abs(ego[:, 0] - float(t if t is not None else ego[-1, 0]))))
    k = max(k - 1, 0)                                           # the step before the boxes overlap
    ve = ego[k, 4:6]
    who = next((w for w in ov.get('collided_with') or [] if w in tr), None)     # the ids and the scenario names
    other = tr.get(who)
    vo = other[min(k, len(other) - 1), 4:6] if other is not None else np.zeros(2)
    return float(np.hypot(*ve) * 3.6), float(np.hypot(*(ve - vo)) * 3.6)


def gap_series(run_dir: str, actor: str = 'lead'):
    """(t, the bumper gap along the ego's heading to the actor, the ego's speed, the actor's speed)."""
    import numpy as np
    tr = _traj(run_dir)
    e, a = tr['ego'], tr[actor]
    n = min(len(e), len(a))
    e, a = e[:n], a[:n]
    yaw = np.radians(e[:, 3])
    along = (a[:, 1] - e[:, 1]) * np.cos(yaw) + (a[:, 2] - e[:, 2]) * np.sin(yaw)
    gap = along - (e[:, 6] + a[:, 6]) / 2.0
    return e[:, 0], gap, np.hypot(e[:, 4], e[:, 5]), np.hypot(a[:, 4], a[:, 5])


def lane_excursion_m(run_dir: str) -> float | None:
    """How far the ego's body went beyond its lane's width at most (≤ 0: inside), from the recorded lateral offsets;
    None when the run has none."""
    worst = None
    for f in _frames(run_dir):
        if f.get('lane_offset_m') is None:
            continue
        x = abs(float(f['lane_offset_m'])) + float(f.get('ego_width_m') or EGO[1]) / 2.0 - \
            float(f.get('lane_width_m') or LANE_M) / 2.0
        worst = x if worst is None else max(worst, x)
    return worst


def score_case(case: dict, run: dict, run_dir: str) -> dict:
    """One finished run against its case's criteria: {'result', 'word', 'tone', 'checks', 'measured'}."""
    import numpy as np
    sc = run.get('scenario') or {}
    ov = (sc.get('criticality') or {}).get('overall') or {}
    v_imp, v_rel = impact_speed_kmh(run, run_dir)
    measured = {'collision': bool(ov.get('collision')), 'impact_speed_kmh': v_imp, 'relative_impact_kmh': v_rel,
                'min_ttc_s': ov.get('min_ttc_s'), 'min_distance_m': ov.get('min_distance_m')}
    checks, word, factor = [], None, None
    for cr in case['criteria']:
        k = cr['kind']
        if k == 'no_collision':
            checks.append({'name': 'no collision', 'ok': not measured['collision'],
                           'detail': 'none' if not measured['collision'] else
                           f'contact at {ov.get("collision_t")} s, {v_rel:.1f} km/h relative'})
        elif k == 'stay_in_lane':
            x = lane_excursion_m(run_dir)
            measured['lane_excursion_m'] = x
            lim = float(cr['limit']['value'])
            checks.append({'name': 'stays in its lane', 'ok': None if x is None else x <= lim + 1e-9,
                           'detail': 'no lane offsets recorded' if x is None else
                           f'body at most {x:+.2f} m beyond the lane width (limit {lim:g} m)'})
        elif k == 'following_distance':
            t, gap, ve, va = gap_series(run_dir, 'lead')
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
            acc = np.abs(np.gradient(va, dt)) if len(va) > 2 else np.zeros(len(va))
            n = max(int(round(2.0 / dt)), 1)
            steady = np.array([bool(np.all(acc[max(i - n, 0):i + 1] < float(cr['steady']['value'])))
                               for i in range(len(t))]) & (t >= 2.0)
            need = np.array([r157_following_m(v) for v in ve])
            short = need - gap
            worst = float(short[steady].max()) if steady.any() else None
            measured['following_shortfall_m'] = worst
            checks.append({'name': 'keeps the minimum following distance', 'ok': None if worst is None else
                           worst <= 1e-6, 'detail': 'the lead was never steady' if worst is None else
                           (f'at least {-worst:.2f} m above d_min' if worst <= 0 else
                            f'{worst:.2f} m below d_min at the worst')})
        elif k == 'mrm':
            m = next((c for c in sc.get('checks') or [] if c.get('name', '').startswith('minimal risk')), None)
            mm = (run.get('mrm') or sc.get('mrm') or {})
            measured['mrm_max_decel_mps2'] = (mm.get('measured') or {}).get('max_decel_mps2')
            checks.append({'name': f'minimum risk manoeuvre (deceleration ≤ {cr["limit"]["value"]:g} m/s², '
                                   'standstill, in lane)', 'ok': None if m is None else bool(m['ok']),
                           'detail': 'no MRM verdict in the run' if m is None else m.get('detail', '')})
        elif k == 'impact_band':
            v = v_rel if cr.get('relative') else v_imp
            word = band(v, cr['bands'])
            factor = BAND_SCORE['value'][word]
            measured['band'] = word
            checks.append({'name': 'impact speed band', 'ok': word == 'green',
                           'detail': 'avoided' if v is None else f'{v:.1f} km/h at contact: {word}'})
    oks = [c['ok'] for c in checks]
    if factor is not None:
        tone = 'good' if word == 'green' else 'warning' if word in ('yellow', 'orange') else 'critical'
        return {'result': word, 'word': word, 'tone': tone, 'factor': factor, 'checks': checks, 'measured': measured}
    if any(o is None for o in oks):
        res = 'not judged'
    else:
        res = 'pass' if all(oks) else 'fail'
    if not case.get('mandatory', True) and res == 'fail':
        return {'result': 'informative', 'word': 'not avoided (informative)', 'tone': 'warning', 'checks': checks,
                'measured': measured}
    return {'result': res, 'word': {'pass': 'passed', 'fail': 'failed', 'not judged': 'not judged'}[res],
            'tone': {'pass': 'good', 'fail': 'critical', 'not judged': 'warning'}[res], 'checks': checks,
            'measured': measured}


def aeb_score(rows: list) -> dict:
    """The Euro NCAP AEB Car-to-Car points of the cases run (assessment protocol 3.3.2): per speed the weighted
    overlaps, × the speed's points, per scenario a share × its scenario points. A scenario whose grid was not run
    completely has no points ('partial')."""
    by = {r['id']: r for r in rows}
    out, total, complete = {}, 0.0, True
    for grp, pts, prefix in (('CCRs', CCRS_POINTS['value'], 'ccrs'), ('CCRm', CCRM_POINTS['value'], 'ccrm')):
        got = have = 0.0
        full = True
        for v, p in pts.items():
            f = [by.get(f'{prefix}-{v}-{o}', {}).get('factor') for o in OVERLAPS['value']]
            if any(x is None for x in f):
                full = False
                continue
            got += p * (f[0] + f[1] + 2 * f[2] + f[3] + f[4]) / 6.0
            have += p
        tot = float(sum(pts.values()))
        out[grp] = {'points': round(got / tot, 4) if full else None, 'of': NCAP['scenario_points']['value'][grp],
                    'grid_points_run': sum(1 for r in rows if r['id'].startswith(prefix + '-') and
                                           r.get('factor') is not None), 'complete': full,
                    'weighted': round(got, 4), 'weight_run': have, 'weight_total': tot}
        complete &= full
        total += got / tot if full else 0.0
    f = [r.get('factor') for r in rows if r['id'].startswith('ccrb-')]
    full = len(f) == 4 and all(x is not None for x in f)
    out['CCRb'] = {'points': round(sum(f) / 4.0, 4) if full else None, 'of': 1.0, 'grid_points_run': len(
        [x for x in f if x is not None]), 'complete': full}
    complete &= full
    total += sum(f) / 4.0 if full else 0.0
    return {'scenarios': out, 'points': round(total, 4) if complete else None, 'of': 3.0,
            'share': round(total / 3.0, 4) if complete else None, 'complete': complete,
            'note': 'of the 3 AEB points of CCRs, CCRm and CCRb (9 in all for AEB Car-to-Car); no correction factor'
                    + ('' if complete else '; partial: not every grid point was run, so there is no score')}


# ===================================================================================================== running it
def result_dir(workspace: str, name: str) -> str:
    return os.path.join(workspace, 'regpacks', name)


def results(workspace: str) -> list:
    d = os.path.join(workspace, 'regpacks')
    return sorted(n for n in (os.listdir(d) if os.path.isdir(d) else [])
                  if os.path.isfile(os.path.join(d, n, 'regpack.json')))


def select(p: dict, cases=None, groups=None) -> list:
    ids = {c['id'] for c in p['cases']}
    for c in cases or ():
        if c not in ids:
            raise RegPackError(f'pack {p["id"]} has no case {c!r}')
    grp = {c['group'] for c in p['cases']}
    for g in groups or ():
        if g not in grp:
            raise RegPackError(f'pack {p["id"]} has no group {g!r}: {", ".join(sorted(grp))}')
    return [c for c in p['cases'] if (not cases or c['id'] in cases) and (not groups or c['group'] in groups)]


def _row(c: dict, **kw) -> dict:
    return dict({'id': c['id'], 'group': c['group'], 'title': c['title'], 'clause': c['clause'],
                 'source': c['source'], 'mandatory': c.get('mandatory', True),
                 'unverified': sorted(k for k, v in (c.get('values') or {}).items()
                                      if _is_value(v) and v['verified'] is False),
                 'note': c.get('note', ''), 'checks': [], 'measured': {}, 'run': None}, **kw)


def case_config(svc, project, c: dict, world: str, rig_id: str, driver: str, seed: int, fps: float,
                duration_scale: float, degraded_mode: bool) -> dict:
    """The stack run of a case (ValueError / RegPackError when it cannot be made)."""
    from . import stack as ST
    scn = copy.deepcopy(c['scenario'])
    if c.get('scalable') and duration_scale != 1.0:
        scn['duration_s'] = max(scn['duration_s'] * duration_scale, 5.0)
    need = float(scn['ego']['speed_kmh']) / 3.6 * float(scn['duration_s'])
    if c.get('scalable') and need > ROUTE_M:
        raise RegPackError(f'the case drives {need:.0f} m at its speed and duration; the scenario runner\'s ego route '
                           f'is {ROUTE_M:g} m long (--duration-scale shortens it, and the result then says so)')
    faults = mrm = None
    if c.get('fault'):
        rig = project.item('rigs', rig_id) or {}
        lid = next((s for s in rig.get('sensors') or [] if s.get('type') == 'lidar'), None)
        if lid is None:
            raise RegPackError('the injected failure is a lidar dropout: the rig has no lidar')
        if driver != 'follower' and driver != 'ros2':
            raise RegPackError(f'the {driver} driver has no failure reaction to judge')
        faults = [dict(c['fault'], sensor=lid['id'])]
        mrm = {'max_decel_mps2': R157['mrm_decel']['value']}
    cfg = ST.stack_config(project, world, rig_id, '/pending', svc.port, svc.tm_port, driver, scn, seed=seed,
                          values={}, fps=fps, render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                          workspace=project.workspace, criteria={'no_collision': True}, faults=faults, mrm=mrm,
                          degraded_mode=bool(degraded_mode or faults))
    if any(cr['kind'] == 'stay_in_lane' for cr in c['criteria']):
        cfg['record_lane_offset'] = True
    return cfg


def run_pack(svc, project, pid: str, out_dir: str, world: str, rig_id: str, cases=None, groups=None,
             driver: str = 'follower', seed: int = 0, fps: float = 20.0, duration_scale: float = 1.0,
             degraded_mode: bool = False, progress=None, name: str = '') -> dict:
    """Run a pack's cases (cached per run key) and return the result document (written by write())."""
    from . import fidelity as F
    from . import stack as ST
    from .provenance import utc_now
    from ..sim.stacksuite import run_one
    p = pack(pid)
    if p['kind'] == 'checklist':
        return checklist_result(p, project.workspace, name)
    chosen = select(p, cases, groups)
    rows = []
    for n, c in enumerate(chosen):
        if not runnable(c):
            rows.append(_row(c, result='refused', word='refused', tone='', why=c.get('refuse') or 'no scenario'))
        else:
            try:
                cfg = case_config(svc, project, c, world, rig_id, driver, seed, fps, duration_scale, degraded_mode)
                key = ST.run_key(cfg)
                cfg['out'] = os.path.join(out_dir, 'runs', key[:16])
                cfg['cache_key'] = key
                cfg['name'] = f'{pid} {c["id"]}'
                rp = os.path.join(cfg['out'], 'run.json')
                run = None
                if os.path.isfile(rp):
                    with open(rp, encoding='utf-8') as fh:
                        run = json.load(fh)
                    run = run if run.get('status') == 'ok' else None
                if run is None:
                    run = run_one(svc, cfg, title=cfg['name'])
                if run.get('status') != 'ok':
                    rows.append(_row(c, result='refused', word='refused', tone='',
                                     why=f'the run did not finish: {run.get("error") or run.get("status")}'))
                else:
                    s = score_case(c, run, cfg['out'])
                    r = _row(c, run=os.path.relpath(cfg['out'], out_dir), fidelity=F.of_doc(run), **s)
                    if c.get('scalable') and duration_scale != 1.0:
                        r['note'] = (r['note'] + ' ' if r['note'] else '') + \
                            (f'shortened to {cfg["scenario"]["duration_s"]:g} s: NOT the document\'s duration')
                        r['shortened'] = True
                    moved = (run.get('scenario') or {}).get('placed_short')
                    if moved:                       # the map's lane ended before an actor's distance (director)
                        r['placed_short'] = moved
                        r['note'] = (r['note'] + ' ' if r['note'] else '') + 'placed nearer than the case asks (' + \
                            ', '.join(f'{m["actor"]} at {m["placed_m"]:g} m, not {m["ahead_m"]:g} m' for m in moved) + \
                            '): the lane ended; NOT the document\'s geometry'
                    for k in ('pm1_avoidable', 'required_5252', 'speed', 'overlap'):
                        if k in c:
                            r[k] = c[k]
                    rows.append(r)
            except (RegPackError, ValueError) as e:
                rows.append(_row(c, result='refused', word='refused', tone='', why=str(e)))
        if progress:
            progress((n + 1) / len(chosen), f'{c["id"]}: {rows[-1]["word"]}')
    doc = {'schema': SCHEMA, 'pack': pid, 'title': p['title'], 'kind': p['kind'], 'name': name,
           'created': utc_now(), 'disclaimer': DISCLAIMER, 'about': p['about'], 'limits': p['limits'],
           'sources': {k: SOURCES[k] for k in p['sources']},
           'inputs': {'world': world, 'rig': rig_id, 'driver': driver, 'seed': seed, 'fps': fps,
                      'duration_scale': duration_scale, 'degraded_mode': bool(degraded_mode),
                      'backend': getattr(svc, 'backend', ''), 'cases_in_pack': len(p['cases']),
                      'cases_selected': len(chosen)},
           'cases': rows, 'unverified': unverified(p), 'choices': choices(p),
           'fidelity': F.combine(r.get('fidelity') for r in rows if r.get('run'))}
    doc.update(verdict_of(p, doc, len(chosen) == len(p['cases'])))
    return doc


def verdict_of(p: dict, doc: dict, whole: bool) -> dict:
    rows = doc['cases']
    counts = {}
    for r in rows:
        counts[r['result']] = counts.get(r['result'], 0) + 1
    out = {'counts': counts}
    if p['kind'] == 'scored':
        sc = aeb_score(rows) if p['id'] == 'ncap_aeb_c2c' else {'points': None, 'share': None, 'complete': False,
                                                              'note': 'no case of this pack runs'}
        out['score'] = sc
        ran = sum(1 for r in rows if r.get('factor') is not None)
        out['verdict'] = 'scored' if sc.get('complete') else 'partial' if ran else 'not run'
        out['verdict_note'] = sc['note']
        return out
    mand = [r for r in rows if r['mandatory']]
    failed = [r for r in mand if r['result'] == 'fail']
    open_ = [r for r in mand if r['result'] in ('refused', 'not judged')]
    short = [r for r in rows if r.get('shortened')]
    moved = [r for r in rows if r.get('placed_short')]
    passed = sum(r['result'] == 'pass' for r in mand)
    out['score'] = {'passed': passed, 'mandatory': len(mand), 'share': round(passed / len(mand), 4) if mand else None}
    if failed:
        v, note = 'failed', f'{len(failed)} mandatory case(s) failed: {", ".join(r["id"] for r in failed[:6])}'
    elif open_ or not whole or short or moved:
        v = 'incomplete'
        note = '; '.join(x for x in (
            f'{len(open_)} mandatory case(s) refused or not judged' if open_ else '',
            'only a selection of the pack\'s cases was run' if not whole else '',
            f'{len(short)} case(s) ran shortened' if short else '',
            f'{len(moved)} case(s) had an actor placed nearer than asked (the map\'s lane ended)' if moved else '')
            if x)
    else:
        v, note = 'passed', f'all {len(mand)} mandatory cases passed'
    out.update(verdict=v, verdict_note=note + ' — a simulation pre-check, not an approval')
    return out


def checklist_result(p: dict, workspace: str = '', name: str = '') -> dict:
    from .provenance import utc_now
    items = copy.deepcopy(p['items'])
    counts = {}
    for it in items:
        counts[it['status']] = counts.get(it['status'], 0) + 1
    by_cat = {}
    for it in items:
        c = by_cat.setdefault(it['category'], {'covered': 0, 'partial': 0, 'gap': 0})
        c[it['status']] += 1
    return {'schema': SCHEMA, 'pack': p['id'], 'title': p['title'], 'kind': 'checklist', 'name': name,
            'created': utc_now(), 'disclaimer': DISCLAIMER, 'about': p['about'], 'limits': p['limits'],
            'sources': {k: SOURCES[k] for k in p['sources']}, 'inputs': {}, 'cases': [], 'items': items,
            'unverified': unverified(p), 'choices': [], 'counts': counts, 'by_category': by_cat,
            'verdict': 'checklist', 'verdict_note': 'a coverage checklist: nothing was run and nothing passes',
            'score': {'covered': counts.get('covered', 0), 'partial': counts.get('partial', 0),
                      'gap': counts.get('gap', 0), 'items': len(items),
                      'share': round(counts.get('covered', 0) / len(items), 4)}}


def write(doc: dict, out_dir: str) -> str:
    from .provenance import write_json
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'regpack.json')
    write_json(p, doc)
    write_report(out_dir, os.path.join(out_dir, 'report.html'))
    return p


def load(path: str) -> dict:
    p = path if path.endswith('.json') else os.path.join(path, 'regpack.json')
    with open(p, encoding='utf-8') as fh:
        doc = json.load(fh)
    if doc.get('schema') != SCHEMA:
        raise RegPackError(f'{p} is not a regulation pack result')
    return doc


# ========================================================================================================= reports
def _val(v) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= 90 else s[:87] + '…'


def text_report(doc: dict) -> str:
    lines = [doc['title'], '!! ' + doc['disclaimer'], f'verdict: {doc["verdict"]} — {doc.get("verdict_note", "")}']
    if doc['kind'] == 'checklist':
        for it in doc['items']:
            lines.append(f'  [{it["status"]:7s}] {it["item"]}' + (f' — {", ".join(it["scenarios"])}'
                                                                 if it['scenarios'] else '') +
                         (f' ({it["note"]})' if it['note'] else ''))
    for r in doc['cases']:
        mark = ' *' if r.get('unverified') else ''
        why = r.get('why') or '; '.join(c['detail'] for c in r.get('checks') or [] if c.get('detail'))
        lines.append(f'  {r["id"]:28s} {r["word"]:26s} {r["clause"]}{mark}' + (f' — {why}' if why else '') +
                     (f' [{r["note"]}]' if r.get('note') else ''))
    sc = doc.get('score') or {}
    if doc['pack'] == 'ncap_aeb_c2c':
        for g, s in (sc.get('scenarios') or {}).items():
            lines.append(f'  {g}: ' + (f'{s["points"]:.3f} of {s["of"]:g} points' if s['points'] is not None else
                                      f'partial ({s["grid_points_run"]} grid points run): no score'))
    if doc['unverified']:
        lines.append('values NOT verified as numbers in the document\'s text (* marks the cases that use one):')
        for u in doc['unverified']:
            lines.append(f'  {u["name"]} = {u["value"]} {u["unit"]} ({SOURCES[u["source"]]["title"]}, {u["clause"]}): '
                         f'{u["why"]}')
    for c in doc.get('choices') or []:
        lines.append(f'Falcon\'s choice: {c["note"]}')
    for lim in doc.get('limits') or []:
        lines.append(f'limit: {lim}')
    return '\n'.join(lines)


def report(src: str):
    """The result as a core.reports Page (kind 'regpack'): the disclaimer first, the verdict or score, per case its
    clause and result, the unverified values, Falcon's choices, the sources and limits."""
    from . import reports as R
    d = src if os.path.isdir(src) else os.path.dirname(src)
    pg = R.Page('regpack', 'Regulation pack', d)
    doc = pg.json(os.path.join(d, 'regpack.json'))
    pg.title = f'{doc["title"]}' + (f' — {doc["name"]}' if doc.get('name') else '')
    pg.add(f'<p class="note"><strong>{R.esc(doc["disclaimer"])}</strong></p>')
    pg.h2('Verdict')
    sc = doc.get('score') or {}
    pairs = [('verdict', R.esc(doc['verdict'])), ('why', R.esc(doc.get('verdict_note', '')))]
    if doc['pack'] == 'ncap_aeb_c2c':
        for g, s in (sc.get('scenarios') or {}).items():
            pairs.append((g, R.esc(f'{s["points"]:.3f} of {s["of"]:g} points' if s['points'] is not None else
                                   f'partial: {s["grid_points_run"]} grid points run, no score')))
    elif sc.get('mandatory') is not None:
        pairs.append(('mandatory cases passed', R.esc(f'{sc["passed"]} of {sc["mandatory"]}')))
    elif doc['kind'] == 'checklist':
        pairs.append(('items', R.esc(f'{sc["covered"]} covered, {sc["partial"]} partial, {sc["gap"]} gaps of '
                                     f'{sc["items"]}')))
    ins = doc.get('inputs') or {}
    if ins:
        pairs.append(('run with', R.esc(', '.join(f'{k} {v}' for k, v in ins.items()))))
    pg.add(R.kv(pairs))
    if doc['kind'] == 'checklist':
        pg.h2('Coverage checklist')
        pg.add(R.table(['category', 'item', 'status', 'Falcon scenarios / features', 'note', 'clause (JAMA Ver 3.0)'],
                       [[R.esc(i['category']), R.esc(i['item']), R.esc(i['status']),
                         R.esc(', '.join(i['scenarios']) or '—'), R.esc(i['note']), R.esc(i['clause'])]
                        for i in doc['items']]))
    else:
        pg.h2('Cases')
        pg.note('* a case that uses a value not verified as a number in the document\'s text (listed below).')
        cls = {'good': 'good', 'warning': 'warn', 'critical': 'crit'}
        pg.add(R.table(['case', 'clause', 'result', 'what was measured', 'note'], [
            [R.esc(r['title']) + (' *' if r.get('unverified') else ''),
             R.esc(f'{SOURCES[r["source"]]["title"].split(" — ")[0]} {r["clause"]}'),
             f'<span class="{cls.get(r.get("tone"), "muted")}">{R.esc(r["word"])}</span>',
             R.esc(r.get('why') or '; '.join(f'{c["name"]}: {c["detail"]}' for c in r.get('checks') or [])),
             R.esc(r.get('note', '') + ('' if r['mandatory'] else ' (not mandatory)'))] for r in doc['cases']]))
    pg.h2('Values not verified in the document\'s text')
    if doc['unverified']:
        pg.add(R.table(['value', 'as used', 'document, clause', 'why it is not verified'], [
            [R.esc(u['name']), R.esc(f'{u["value"]} {u["unit"]}'),
             R.esc(f'{SOURCES[u["source"]]["title"]}, {u["clause"]}'), R.esc(u['why'] + (f' — {u["note"]}' if
                                                                                     u['note'] else ''))]
            for u in doc['unverified']]))
    else:
        pg.note('None: every document value of this pack was read in the document\'s text.')
    if doc.get('choices'):
        pg.h2('Falcon\'s own choices (not any document\'s values)')
        pg.add('<ul>' + ''.join(f'<li>{R.esc(c["note"])}</li>' for c in doc['choices']) + '</ul>')
    pg.h2('Sources')
    pg.add(R.table(['document', 'version', 'date', 'read from'], [
        [R.esc(s['title']), R.esc(s['version']), R.esc(s['date']), R.esc(s['read_from'])]
        for s in doc['sources'].values()]))
    pg.h2('Limits')
    pg.add('<ul>' + ''.join(f'<li>{R.esc(x)}</li>' for x in [DISCLAIMER] + list(doc.get('limits') or [])) + '</ul>')
    pg.provenance(None)
    return pg


def write_report(src: str, out: str) -> dict:
    from .paths import atomic_write
    doc, meta = report(src).render()
    meta['path'] = os.path.abspath(out)
    atomic_write(out, doc)
    return meta


# ========================================================================================================== export
def export(pid: str, out_dir: str, driver: str = 'follower') -> dict:
    """The pack's runnable scenarios as Falcon scenario JSON and OpenSCENARIO files, and a falcon.suite/1 suite that
    names them (for `falcon stack run` / `falcon ci run` once the scenarios are added to a project)."""
    from . import openscenario as OX
    from . import scenarios as SC
    from .provenance import write_json
    p = pack(pid)
    os.makedirs(os.path.join(out_dir, 'scenarios'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'xosc'), exist_ok=True)
    entries, refused = [], []
    for c in p['cases']:
        if not runnable(c):
            refused.append({'case': c['id'], 'why': c.get('refuse') or 'no scenario'})
            continue
        scn = c['scenario']
        write_json(os.path.join(out_dir, 'scenarios', scn['id'] + '.json'), scn)
        with open(os.path.join(out_dir, 'xosc', scn['id'] + '.xosc'), 'w', encoding='utf-8') as fh:
            fh.write(OX.export_xosc(SC.concrete(scn, {})))
        entries.append({'scenario': scn['id'], 'seeds': [0], 'criteria': {'no_collision': True}})
    suite = {'schema': 'falcon.suite/1', 'name': f'regpack-{pid}', 'about': f'{p["title"]}. {DISCLAIMER} Pass/fail '
             'here is the suite\'s no-collision check only: `falcon regpack run` applies the pack\'s own criteria.',
             'stack': {'driver': driver}, 'world': '', 'rig': '', 'fps': 20.0, 'motion': 'kinematic',
             'traffic_vehicles': 0, 'criteria': {'no_collision': True}, 'entries': entries}
    write_json(os.path.join(out_dir, 'suite.json'), suite)
    return {'scenarios': len(entries), 'refused': refused, 'suite': os.path.join(out_dir, 'suite.json')}


def summary(pid: str) -> dict:
    p = pack(pid)
    return {'id': pid, 'title': p['title'], 'kind': p['kind'], 'about': p['about'],
            'cases': len(p['cases']) or len(p.get('items') or []), 'runnable': sum(runnable(c) for c in p['cases']),
            'unverified': len(unverified(p)), 'sources': [SOURCES[s]['title'] + (f' ({SOURCES[s]["version"]}, '
                                                          f'{SOURCES[s]["date"]})') for s in p['sources']]}


# ============================================================================================================= CLI
def main_regpack(argv) -> int:
    """`falcon regpack list|show|run|report|export …` (module doc): 0 ok (a failed pack is a result), 2 failed, 6
    refused."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon regpack', description='regulation test packs — a simulation pre-check, '
                                 'not a type-approval or a rating (docs/regulation-packs.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    ls = sub.add_parser('list', help='the packs')
    ls.add_argument('--json', action='store_true')
    sh = sub.add_parser('show', help='a pack: its cases with their clauses, the values and which are unverified')
    sh.add_argument('pack')
    sh.add_argument('--json', action='store_true')
    r = sub.add_parser('run', help='run a pack\'s cases as stack runs and score them')
    r.add_argument('project')
    r.add_argument('pack')
    r.add_argument('--name', required=True, help='the result\'s folder under <workspace>/regpacks/')
    r.add_argument('--case', action='append', default=[], help='a case id (repeatable; default: all)')
    r.add_argument('--group', action='append', default=[], help='a case group (repeatable)')
    r.add_argument('--driver', default='follower', choices=('follower', 'autopilot'),
                   help='the system under test (a ROS 2 stack: export the suite and use falcon stack)')
    r.add_argument('--degraded-mode', action='store_true', help='the follower with its health monitor')
    r.add_argument('--world', default='')
    r.add_argument('--rig', default='')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--fps', type=float, default=20.0)
    r.add_argument('--duration-scale', type=float, default=1.0,
                   help='shorten the long cases (the 5-minute lane keeping) by this factor; the result then says '
                        'the document\'s duration was not run and the verdict is incomplete')
    r.add_argument('--fake-carla', default='')
    rp = sub.add_parser('report', help='a result\'s report: falcon regpack report DIR [--out FILE.html]')
    rp.add_argument('result')
    rp.add_argument('--out', default='')
    rp.add_argument('--json', action='store_true')
    ex = sub.add_parser('export', help='a pack\'s scenarios as Falcon JSON and OpenSCENARIO, and a suite')
    ex.add_argument('pack')
    ex.add_argument('--out', required=True)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        if a.action == 'list':
            rows = [summary(p) for p in PACKS]
            if a.json:
                print(json.dumps({'disclaimer': DISCLAIMER, 'packs': rows}, indent=2))
            else:
                print(DISCLAIMER)
                for s in rows:
                    print(f'{s["id"]:13s} {s["title"]} — {s["cases"]} cases ({s["runnable"]} runnable), '
                          f'{s["unverified"]} unverified value(s)')
            return P.OK
        if a.action == 'show':
            p = pack(a.pack)
            if a.json:
                print(json.dumps(dict(p, disclaimer=DISCLAIMER, unverified=unverified(p),
                                      sources={k: SOURCES[k] for k in p['sources']}), indent=2, default=str))
                return P.OK
            print(p['title'])
            print('!! ' + DISCLAIMER)
            for s in p['sources']:
                print(f'source: {SOURCES[s]["title"]}, {SOURCES[s]["version"]}, {SOURCES[s]["date"]}')
            for c in p['cases']:
                print(f'  {c["id"]:28s} {c["clause"]:36s} ' + ('refused: ' + c['refuse'][:60] + '…' if c.get('refuse')
                                                              else 'mandatory' if c['mandatory'] else 'informative'))
            for it in p.get('items') or []:
                print(f'  [{it["status"]:7s}] {it["item"]}')
            print('values:')
            for name, v in values_of(p):
                if '.' in name:
                    continue
                tag = 'verified' if v['verified'] else 'Falcon\'s choice' if v['verified'] is None else \
                    'NOT VERIFIED: ' + BASIS_TEXT[v['basis']]
                print(f'  {name} = {_val(v["value"])} {v["unit"]} [{SOURCES[v["source"]]["title"].split(" — ")[0]}, '
                      f'{v["clause"]}] {tag}')
            return P.OK
        if a.action == 'export':
            out = export(a.pack, a.out)
            print(f'{out["scenarios"]} scenarios in {a.out}/scenarios and /xosc; suite {out["suite"]}')
            for x in out['refused']:
                print(f'refused: {x["case"]} — {x["why"]}')
            return P.OK
        if a.action == 'report':
            doc = load(a.result)
            d = a.result if os.path.isdir(a.result) else os.path.dirname(a.result)
            out = a.out or os.path.join(d, 'report.html')
            write_report(d, out)
            print(json.dumps(doc, indent=2) if a.json else text_report(doc))
            print(f'report: {out}')
            return P.OK
    except (RegPackError, OSError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED if isinstance(e, RegPackError) else P.FAILED
    return _run(a)


def _run(a) -> int:
    from . import protocol as P
    from .project import Project, ProjectError
    try:
        proj = Project.load(a.project)
        p = pack(a.pack)
        select(p, a.case, a.group)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    except RegPackError as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    out = result_dir(proj.workspace, a.name)
    em = P.Emitter()
    em.hello('regpack')
    em.log(DISCLAIMER, 'warning')
    svc = None
    try:
        if p['kind'] == 'checklist':
            doc = checklist_result(p, proj.workspace, a.name)
        else:
            from ..sim.service import study_service
            svc = study_service(None, proj.workspace, fake_carla=a.fake_carla)
            doc = run_pack(svc, proj, a.pack, out, a.world or proj.doc['worlds'][0]['id'],
                           a.rig or proj.doc['rigs'][0]['id'], a.case, a.group, a.driver, a.seed, a.fps,
                           a.duration_scale, a.degraded_mode,
                           lambda f, t: em.progress(f, t, force=True), a.name)
        path = write(doc, out)
    except (RegPackError, RuntimeError, OSError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if svc is not None:
            svc.shutdown()
    for line in text_report(doc).splitlines():
        em.log(line)
    em.result('ok', summary={'pack': a.pack, 'result': a.name, 'verdict': doc['verdict'],
                             'counts': doc.get('counts'), 'unverified': len(doc['unverified']), 'path': path})
    return P.OK
