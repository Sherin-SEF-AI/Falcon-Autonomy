"""Criticality measures against hand-computed cases: TTC, PET, DRAC, box distance, collisions, the summary."""
import json
import math

import numpy as np
import pytest

from falcon_autonomy.core import criticality as K


def traj(t, x, y, yaw, vx, vy, length, width):
    t = np.asarray(t, dtype=float)
    cols = [np.broadcast_to(np.asarray(c, dtype=float), t.shape) for c in (t, x, y, yaw, vx, vy, length, width)]
    return np.stack(cols, axis=1)


def test_head_on_boxes_give_the_exact_ttc():
    ego = [0, 0, 0, 0, 10, 0, 4.5, 2.0]
    actor = [0, 50, 0, 180, -5, 0, 4.0, 1.8]
    gap, closing = 50 - 4.5 / 2 - 4.0 / 2, 15.0                 # front to front 45.75 m at 15 m/s
    assert K.ttc(ego, actor) == pytest.approx(gap / closing, abs=1e-12)
    # the same encounter rotated by 30°: boxes are rotated rectangles, the answer does not change
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    ego_r = [0, 0, 0, 30, 10 * c, 10 * s, 4.5, 2.0]
    actor_r = [0, 50 * c, 50 * s, 210, -5 * c, -5 * s, 4.0, 1.8]
    assert K.ttc(ego_r, actor_r) == pytest.approx(gap / closing, abs=1e-9)
    assert K.ttc(ego, actor, horizon=3.0) == math.inf          # beyond the horizon
    assert K.ttc(ego, [0, 4, 0.5, 10, 0, 0, 4, 2]) == 0.0      # overlapping now
    # offset by more than the half widths: passes, never collides
    assert K.ttc(ego, [0, 50, 2.0, 180, -5, 0, 4.0, 1.8]) == math.inf
    # offset by less: box-based, so a partial overlap still collides at the same time
    assert K.ttc(ego, [0, 50, 1.8, 180, -5, 0, 4.0, 1.8]) == pytest.approx(gap / closing, abs=1e-12)


def test_parallel_non_closing_gives_inf_ttc():
    t = np.arange(0, 5, 0.05)
    ego = traj(t, 10 * t, 0, 0, 10, 0, 4.5, 2.0)
    beside = traj(t, 10 * t, 3.5, 0, 10, 0, 4.5, 2.0)          # next lane, same speed
    ahead = traj(t, 20 + 12 * t, 0, 0, 12, 0, 4.5, 2.0)        # same lane, pulling away
    for other in (beside, ahead):
        tt, v = K.ttc_series(ego, other)
        assert len(tt) == len(t) and np.all(np.isinf(v))
    assert K.ttc(ego[0], beside[0]) == math.inf


def test_ttc_series_aligns_an_actor_that_appears_late():
    t = np.arange(0, 4, 0.05)
    ego = traj(t, 10 * t, 0, 0, 10, 0, 4, 2)
    late = traj(t[40:], 100 - 5 * t[40:], 0, 180, -5, 0, 4, 2)
    tt, v = K.ttc_series(ego, late)
    assert np.allclose(tt, t[40:])
    gap = (100 - 5 * tt) - 10 * tt - 4
    assert np.allclose(v, gap / 15)


def test_rotated_box_distance_matches_brute_force_sampling_of_the_edges():
    rng = np.random.default_rng(7)

    def edge_points(P, n=200):
        s = np.linspace(0, 1, n)[:, None]
        return np.vstack([P[k] + s * (P[(k + 1) % 4] - P[k]) for k in range(4)])

    checked = 0
    while checked < 25:
        a = K.box_corners(*rng.uniform([-5, -5, -180, 1, 0.5], [5, 5, 180, 6, 3]))
        b = K.box_corners(*rng.uniform([-8, -8, -180, 1, 0.5], [8, 8, 180, 6, 3]))
        d = K.box_distance(a, b)
        pa, pb = edge_points(a), edge_points(b)
        brute = np.min(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1))
        if d == 0.0:
            continue                                              # overlapping pairs are checked below
        spacing = 6.0 / 199
        assert d <= brute + 1e-9 and brute - d <= spacing
        assert d == pytest.approx(K.box_distance(b, a), abs=1e-12)
        checked += 1
    # axis-aligned and rotated cases with a known answer
    assert K.box_distance(K.box_corners(0, 0, 0, 4, 2), K.box_corners(10, 0, 0, 4, 2)) == pytest.approx(6.0)
    # a diamond (45°) with its corner 1 m beside the edge of a box
    h = math.sqrt(2)                                              # half diagonal of a 2 × 2 square
    assert K.box_distance(K.box_corners(0, 0, 0, 4, 2), K.box_corners(0, 1 + 1 + h, 45, 2, 2)) == \
        pytest.approx(1.0)
    assert K.box_distance(K.box_corners(0, 0, 0, 4, 2), K.box_corners(1, 0.5, 30, 4, 2)) == 0.0


def test_a_braking_lead_gives_the_analytic_drac():
    t = np.arange(0, 4.0 + 1e-9, 0.05)
    ego = traj(t, 20 * t, 0, 0, 20, 0, 4, 2)
    lead = traj(t, 40 + 20 * t - 2 * t ** 2, 0, 0, 20 - 4 * t, 0, 4, 2)   # braking at 4 m/s²
    tt, drac = K.required_deceleration_series(ego, lead)
    gap = 36 - 2 * tt ** 2                                        # 40 m centre to centre, two half lengths of 2 m
    assert np.allclose(drac, (4 * tt) ** 2 / (2 * gap), rtol=1e-9, atol=1e-12)
    assert drac[-1] == pytest.approx(16 * 16 / (2 * 4))           # t = 4 s: closing 16 m/s, 4 m gap
    # the same at 60° heading
    c, s = math.cos(math.radians(60)), math.sin(math.radians(60))
    e, l_ = [0, 0, 0, 60, 20 * c, 20 * s, 4, 2], [0, 30 * c, 30 * s, 60, 14 * c, 14 * s, 4, 2]
    assert K.required_deceleration(e, l_) == pytest.approx(6 ** 2 / (2 * 26))
    # not in the path (next lane), not closing, behind: no braking needed
    assert K.required_deceleration([0, 0, 0, 0, 20, 0, 4, 2], [0, 30, 3.5, 0, 10, 0, 4, 2]) == 0.0
    assert K.required_deceleration([0, 0, 0, 0, 20, 0, 4, 2], [0, 30, 0, 0, 25, 0, 4, 2]) == 0.0
    assert K.required_deceleration([0, 0, 0, 0, 20, 0, 4, 2], [0, -30, 0, 0, 10, 0, 4, 2]) == 0.0
    assert K.required_deceleration([0, 0, 0, 0, 20, 0, 4, 2], [0, 3, 0, 0, 10, 0, 4, 2]) == math.inf


def _crossing(actor_y0, t_end=8.0, dt=0.01):
    t = np.arange(0, t_end + 1e-9, dt)
    ego = traj(t, -30 + 10 * t, 0, 0, 10, 0, 4.0, 2.0)            # along +x
    actor = traj(t, 0, actor_y0 + 5 * t, 90, 0, 5, 4.5, 1.8)      # along +y, at a right angle
    return ego, actor


def test_paths_crossing_at_a_right_angle_give_the_analytic_pet():
    # conflict area |x| ≤ 0.9, |y| ≤ 1; the ego leaves when its rear passes x = 0.9: (0.9 + 2 + 30) / 10 = 3.29 s;
    # the actor enters when its front reaches y = -1: (40 - 1 - 2.25) / 5 = 7.35 s
    ego, actor = _crossing(-40.0)
    d = K.pet_detail(ego, actor)
    assert d['first'] == 'ego'
    # sampled: the last occupied sample is at most one step before leaving, the first at most one after entering
    assert 4.06 - 1e-9 <= d['pet'] < 4.06 + 0.02
    assert d['t_first_leaves'] == pytest.approx(3.29, abs=0.0101)
    assert d['t_second_enters'] == pytest.approx(7.35, abs=0.0101)
    assert abs(d['x']) < 3 and abs(d['y']) < 3
    # sampled at 0.1 s, resolved at dt = 0.01 s by interpolation: the same answer
    coarse_e, coarse_a = _crossing(-40.0, dt=0.1)
    assert 4.06 - 1e-9 <= K.pet(coarse_e, coarse_a, dt=0.01) < 4.06 + 0.02
    assert 4.06 - 1e-9 <= K.pet(coarse_e, coarse_a) < 4.06 + 0.2
    # the actor first: it leaves at (10 + 1 + 2.25) / 5 = 2.65 s, the ego enters at (30 - 2 - 0.9) / 10 = 2.71 s;
    # starting 10 m further back the ego enters at 3.71 s
    ego2 = traj(np.arange(0, 8.0001, 0.01), -40 + 10 * np.arange(0, 8.0001, 0.01), 0, 0, 10, 0, 4.0, 2.0)
    _, actor2 = _crossing(-10.0)
    d2 = K.pet_detail(ego2, actor2)
    assert d2['first'] == 'actor' and 1.06 - 1e-9 <= d2['pet'] < 1.06 + 0.02
    # parallel lanes never share a footprint: no PET
    t = np.arange(0, 5, 0.05)
    assert K.pet(traj(t, 10 * t, 0, 0, 10, 0, 4, 2), traj(t, 10 * t, 3.5, 0, 10, 0, 4, 2)) is None
    # a follower on the leader's path: the time gap, (20 - 4) m / 10 m/s
    assert K.pet(traj(t, 10 * t, 0, 0, 10, 0, 4, 2), traj(t, 20 + 10 * t, 0, 0, 10, 0, 4, 2)) == \
        pytest.approx(1.6, abs=0.101)


def test_pet_handles_a_stopped_actor_and_simultaneous_occupation():
    t = np.arange(0, 6, 0.05)
    ego = traj(t, -30 + 10 * t, 0, 0, 10, 0, 4, 2)
    parked = traj(t, 10, 0, 0, 0, 0, 4, 2)                       # in the lane for the whole run: a collision
    d = K.pet_detail(ego, parked)
    assert d['pet'] == 0.0 and d['first'] is None
    gone = traj(t[:20], 10, 0, 0, 0, 0, 4, 2)                     # there for the first second only
    # it leaves at t = 0.95 s; the ego's front reaches its rear (x = 8) at (8 - 2 + 30) / 10 = 3.6 s
    assert 2.65 - 1e-9 <= K.pet_detail(ego, gone)['pet'] < 2.65 + 0.1


def test_min_distance_and_when():
    t = np.arange(0, 6, 0.05)
    ego = traj(t, 10 * t, 0, 0, 10, 0, 4, 2)
    other = traj(t, 60 - 10 * t, 3.0, 180, -10, 0, 4, 2)        # oncoming, next lane: 1 m between the boxes
    d, when = K.min_distance(ego, other)
    assert d == pytest.approx(1.0) and 2.6 <= when <= 3.4
    assert K.min_distance(ego, traj([100.0], 0, 0, 0, 0, 0, 4, 2)) == (math.inf, None)


def test_collisions_from_boxes_and_sensor_events():
    ego, actor = _crossing(-15.0, dt=0.05)                        # both centres at the crossing at t = 3 s
    col = K.collisions(ego, actor)
    assert col['times'] and col['first_t'] == col['times'][0]
    assert 2.0 < col['first_t'] < 4.0
    ev = [(3.1, 7, (100.0, 0.0, 0.0)), (3.2, 'static', 50.0)]
    c7 = K.collisions(ego, None, ev, actor_id=7)
    assert [e['t'] for e in c7['events']] == [3.1] and c7['events'][0]['impulse'] == 100.0
    assert len(K.collisions(ego, None, ev)['events']) == 2
    far = traj(ego[:, 0], 0, 50, 0, 0, 0, 4, 2)
    assert K.collisions(ego, far) == {'times': [], 'events': [], 'first_t': None}


def test_summarise_ranks_collisions_above_near_misses_and_is_json_ready():
    ego, crash = _crossing(-15.0, dt=0.05)
    _, near = _crossing(-22.7, dt=0.05)                           # enters 0.6 s after the ego left (3.29 s)
    _, late = _crossing(-35.0, dt=0.05)                           # enters 3.06 s after
    far = traj(ego[:, 0], 0, 80, 0, 0, 0, 4, 2)
    s = K.summarise(ego, {'near': near, 'late': late, 'far': far}, dt=0.05)
    json.dumps(s, allow_nan=False)
    a = s['actors']
    assert not s['overall']['collision'] and s['overall']['critical_actor'] == 'near'
    assert 0 <= a['far']['criticality'] < 0.01 < a['late']['criticality'] < a['near']['criticality'] < 0.9
    assert a['far']['min_ttc_s'] is None and a['far']['pet_s'] is None and a['far']['max_drac_mps2'] == 0.0
    assert a['near']['pet_first'] == 'ego' and s['overall']['min_pet_actor'] == 'near'
    assert s['criticality'] == pytest.approx(a['near']['criticality'])
    terms = a['near']['terms']
    q = 1.0
    for v in terms.values():
        q *= 1 - v
    assert a['near']['criticality'] == pytest.approx(0.9 * (1 - q))
    c = K.summarise(ego, {'crash': crash, 'far': far}, dt=0.05, events=[(3.0, 'crash', 80.0)])
    assert c['criticality'] == 1.0 and c['actors']['crash']['collision'] and c['overall']['collided_with'] == ['crash']
    assert c['actors']['crash']['collision_impulse'] == 80.0 and c['actors']['crash']['min_ttc_s'] == 0.0
    # a sensor event with something that is not a tracked actor (a pole) still makes the run a collision
    p = K.summarise(ego, {'far': far}, events=[(2.0, 'pole', 10.0)])
    assert p['criticality'] == 1.0 and p['other_collisions'][0]['other_id'] == 'pole'
    assert K.summarise(ego, {})['criticality'] == 0.0
