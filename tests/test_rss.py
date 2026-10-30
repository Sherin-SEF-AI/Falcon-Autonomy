"""Responsibility-Sensitive Safety (core.rss): the safe-distance formulas against hand computations and the paper's
own numbers, dangerous situations and the Danger Threshold, proper response and the failure labels on constructed
trajectories, kinematic feasibility of searched parameters, and the labels in a study (fake carla)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from falcon_autonomy.core import rss as R
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core import search as S

P = R.PARAMS
DT = 0.05


# ------------------------------------------------------------------------------------------------- the formulas
def test_paper_pedestrian_example_half_a_metre():
    """Section 3.8: a standing pedestrian, a delay of 500 ms and acceleration and braking of 2 m/s² — "her part of
    the safe longitudinal distance is 50cm": 0.5 s accelerating (0.25 m, reaching 1 m/s), then braking (0.25 m)."""
    assert R.safe_longitudinal_same(0.0, 0.0, 0.5, 2.0, 2.0, 1e9) == pytest.approx(0.5)


def test_paper_highway_braking_distance():
    """Section 3.7: braking at 10 m/s² from 30 m/s takes d = v²/2a = 45 m."""
    assert R.braking_distance(30.0, 10.0) == pytest.approx(45.0)


def test_lemma_2_same_direction_by_hand():
    # v_r = v_f = 20 m/s, ρ = 1 s, a_max,accel 3.5, a_min,brake 4, a_max,brake 8:
    # 20·1 + ½·3.5·1 + (20 + 3.5)²/(2·4) − 20²/(2·8) = 20 + 1.75 + 69.03125 − 25 = 65.78125 m
    assert R.safe_longitudinal_same(20, 20, 1.0, 3.5, 4.0, 8.0) == pytest.approx(65.78125)
    # a fast front car: the bracket is negative, d_min = 0
    assert R.safe_longitudinal_same(5, 40, 1.0, 3.5, 4.0, 8.0) == 0.0
    # a stopped front car: the rear car's whole stopping distance with its response
    assert R.safe_longitudinal_same(10, 0, 0.5, 2.0, 5.0, 8.0) == pytest.approx(10 * 0.5 + 0.25 + 11 ** 2 / 10)


def test_lemma_3_opposite_directions_by_hand():
    # v1 = 10, v2 = −15, ρ = 1, a_max,accel 3.5, a_min,brake 4, a_min,brake,correct 3: v1,ρ = 13.5, v2,ρ = 18.5
    # (10 + 13.5)/2 + 13.5²/6 + (15 + 18.5)/2 + 18.5²/8 = 11.75 + 30.375 + 16.75 + 42.78125 = 101.65625 m
    assert R.safe_longitudinal_opposite(10, -15, 1.0, 1.0, 3.5, 4.0, 3.0) == pytest.approx(101.65625)


def test_lemma_4_lateral_by_hand_and_the_signed_extension():
    # c1 (left) moving right at 0.5, c2 moving left at 0.3: μ 0.1, ρ 1, a_lat 0.2, lateral braking 0.8
    # v1,ρ = 0.7, v2,ρ = −0.5: 0.1 + [(1.2/2 + 0.49/1.6) − (−0.8/2 − 0.25/1.6)] = 0.1 + 0.90625 + 0.55625 = 1.5625
    paper = 0.1 + max((0.5 + 0.7) / 2 + 0.7 ** 2 / 1.6 - ((-0.3 - 0.5) / 2 - 0.5 ** 2 / 1.6), 0.0)
    assert paper == pytest.approx(1.5625)
    assert R.safe_lateral(0.5, -0.3, 0.1, 1.0, 0.2, 0.8) == pytest.approx(1.5625)
    # both still: each may drift towards the other for ρ, then brake: 0.1 + 2 × (0.1 + 0.2²/1.6) = 0.35
    assert R.safe_lateral(0.0, 0.0, 0.1, 1.0, 0.2, 0.8) == pytest.approx(0.35)
    # c1 still drifting away (left, −0.3 m/s; −0.1 after ρ) while c2 closes fast (−1.0): the paper's v1,ρ² counts
    # c1's braking from −0.1 as a move towards c2 (+0.00625 m); the signed v1,ρ|v1,ρ| as the move away it is
    # paper: 0.1 + (−0.2 + 0.00625) − (−1.1 − 0.9) = 1.90625; signed: 0.1 + (−0.2 − 0.00625) + 2.0 = 1.89375
    assert R.safe_lateral(-0.3, -1.0, 0.1, 1.0, 0.2, 0.8) == pytest.approx(1.89375)


# ------------------------------------------------------------------------------------------------- trajectories
def track(x0, y0, vx, n, yaw=0.0, L=4.6, W=2.0, ax=None, vy=None, t_brake=None, decel=0.0, x_list=None):
    """(t, x, y, yaw, vx, vy, L, W) rows at DT: constant velocity, optionally braking at decel from t_brake, or a
    lateral velocity profile vy(t)."""
    rows, x, y, v = [], x0, y0, vx
    for k in range(n):
        t = round(k * DT, 4)
        if t_brake is not None and t >= t_brake:
            v = max(v - decel * DT, 0.0) if v >= 0 else min(v + decel * DT, 0.0)
        lat = vy(t) if vy is not None else 0.0
        rows.append([t, x, y, yaw, v, lat, L, W])
        x += v * DT
        y += lat * DT
    return np.array(rows)


def closest(ego, actor):
    """The first contact (the longitudinal gap reaching 0 while laterally overlapping), else the smallest gap."""
    g = R.geometry(ego, actor, DT)
    gap = np.where(g['gap_lat'] < 0.0, g['gap_lon'], np.inf)
    hit = np.flatnonzero(gap <= 0.0)
    return float(g['t'][hit[0] if len(hit) else int(np.argmin(gap))])


def test_rear_end_with_an_ego_that_does_not_brake_is_avoidable():
    ego = track(0, 0, 15.0, 200)
    lead = track(30, 0, 15.0, 200, t_brake=2.0, decel=5.0)            # within a_max,brake: the ego must respond
    t = closest(ego, lead)
    assert t == pytest.approx(5.2, abs=0.06)                          # 25.4 m: 22.5 while it brakes, 2.9 after
    lab = R.label_failure(ego, lead, DT, t_event=t)
    assert lab['label'] == 'avoidable' and 'longitudinal' in lab['response'] and lab['relation'] == 'ego_behind'
    assert 'braked less than a_min,brake' in lab['reason']


def test_an_ego_braking_properly_behind_a_lead_braking_too_hard_is_not_to_blame():
    # 20 m apart at 15 m/s (unsafe from the start: the threshold is the first sample); the ego brakes at exactly
    # a_min,brake = 4 m/s² from ρ = 1 s on (compliant: 15 + 15²/8 = 43.1 m), the lead at 9 m/s² from 0.5 s (beyond
    # RSS's a_max,brake 8, within the feasible 9: 7.5 + 12.5 = 20 m): they meet — the lead's responsibility
    ego = track(0, 0, 15.0, 200, t_brake=1.0, decel=4.0)
    lead = track(24.6, 0, 15.0, 200, t_brake=0.5, decel=9.0)
    t = closest(ego, lead)
    lab = R.label_failure(ego, lead, DT, t_event=t)
    assert lab['ego_violations'] == [] and lab['label'] == 'other' and 'harder than a_max,brake' in lab['reason'], lab
    hard = track(24.6, 0, 15.0, 200, t_brake=0.5, decel=12.0)             # beyond the feasibility limit 9 m/s²
    lab = R.label_failure(ego, hard, DT, t_event=closest(ego, hard))
    assert lab['label'] == 'unavoidable' and 'feasibility' in lab['reason'], lab
    lazy = track(0, 0, 15.0, 200, t_brake=1.0, decel=2.0)                 # braking at 2 < a_min,brake: its own
    assert R.label_failure(lazy, lead, DT, t_event=closest(lazy, lead))['label'] == 'avoidable'


def test_a_cut_in_is_the_cutters_responsibility_or_unavoidable_inside_the_envelope():
    def lane_change(t0, dur=1.5):
        return lambda t: -3.5 / dur if t0 <= t < t0 + dur else 0.0     # from the left lane (y −3.5) to the ego's
    ego = track(0, 0, 15.0, 120)
    far = track(25, 3.5, 17.0, 120, vy=lambda t: -3.5 / 1.5 if 1.0 <= t < 2.5 else 0.0)
    lab = R.label_failure(ego, far, DT, t_event=2.4)
    assert lab['label'] == 'other' and lab['response'] == 'lateral', lab
    assert 'lateral' in lab['reason']
    near = track(8, 3.5, 15.0, 120, vy=lambda t: -3.5 / 1.5 if 1.0 <= t < 2.5 else 0.0)
    lab = R.label_failure(ego, near, DT, t_event=2.4)
    assert lab['label'] == 'unavoidable' and 'braking distance' in lab['reason'], lab


def test_walkers_entering_far_ahead_and_right_in_front():
    # at 8 m/s the ego's safe distance to a standing walker is 8 + 1.75 + 11.5²/8 = 26.3 m. A walker 60 m ahead
    # crosses slowly (0.5 m/s from 3.5 m right of the ego's line): its lateral distance turns unsafe at about 1.8 s,
    # when the ego is still 43 m off (safe); the longitudinal one at 26.3 m (t ≈ 3.9 s) — the later: the ego had to
    # brake, and did not
    ego = track(0, 0, 8.0, 200)
    far = track(60, 3.5, 0.0, 200, L=0.6, W=0.6, vy=lambda t: -0.5)
    lab = R.label_failure(ego, far, DT, t_event=(60 - 2.3 - 0.3) / 8.0, walker=True)
    assert lab['label'] == 'avoidable' and lab['response'] == 'longitudinal', lab
    assert lab['t_b'] == pytest.approx((57.4 - R.safe_longitudinal_same(8, 0, 1.0, 3.5, 4.0, 2.0)) / 8.0, abs=0.06)
    ego = track(0, 0, 12.0, 200)
    # steps out 6 m in front: inside the ego's braking distance (12²/16 = 9 m): unavoidable
    near = track(6 + 0.2 * 12, 2.0, 0.0, 60, L=0.6, W=0.6, vy=lambda t: -2.0 if t >= 0.2 else 0.0)
    lab = R.label_failure(ego, near, DT, t_event=0.9, walker=True)
    assert lab['label'] == 'unavoidable', lab


def test_a_parallel_car_is_not_dangerous_and_an_oncoming_one_needs_braking():
    ego = track(0, 0, 15.0, 100)
    beside = track(2, 3.5, 15.0, 100)
    assert R.label_failure(ego, beside, DT, t_event=2.0)['label'] == 'safe'
    wrong = track(120, 0.6, -8.0, 200, yaw=180.0, L=2.2, W=0.9)        # a scooter against the traffic in the lane
    lab = R.label_failure(ego, wrong, DT, t_event=4.9)
    assert lab['relation'] == 'oncoming' and lab['label'] == 'avoidable', lab
    braking = track(0, 0, 15.0, 300, t_brake=0.3, decel=5.0)         # stops; the scooter rides on into it
    lab = R.label_failure(braking, track(120, 0.6, -8.0, 300, yaw=180.0, L=2.2, W=0.9), DT, t_event=11.5)
    assert lab['label'] == 'other' and 'wrong-way' in lab['reason'], lab


def test_danger_threshold_is_the_later_of_the_two():
    ego = track(0, 0, 15.0, 120)
    cut = track(25, 3.5, 17.0, 120, vy=lambda t: -3.5 / 1.5 if 1.0 <= t < 2.5 else 0.0)
    g = R.geometry(ego, cut, DT)
    k = int(np.argmin(np.abs(g['t'] - 2.4)))
    thr = R.danger_threshold(g, k)
    assert thr['k_long'] < thr['k_lat'] == thr['k_b'] and thr['lateral'] and not thr['longitudinal']
    assert 1.0 <= thr['t_b'] <= 1.6                     # when its lateral move began to eat the safe distance


def test_feasibility_of_a_measured_actor():
    calm = track(0, 0, 10.0, 100, t_brake=1.0, decel=6.0)
    assert R.feasibility(calm, DT)['ok']
    wild = track(0, 0, 10.0, 100, t_brake=1.0, decel=12.0)
    f = R.feasibility(wild, DT)
    assert not f['ok'] and 'brake' in f['over_s'] and f['peak']['brake'] == pytest.approx(12.0, abs=0.5)


def test_feasible_space_narrows_what_sets_dynamics():
    cut = SC.builtin('cut_in')
    space, notes = R.feasible_space(cut, SC.search_space(cut))
    ct = next(p for p in space if p['name'] == 'cut_time')
    t_min = 2 * math.sqrt(3.5 / 4.0)                    # one lane at 4 m/s², accelerating then braking: 1.87 s
    assert ct['lo'] == pytest.approx(t_min) and ct['hi'] == 3.0 and any('cut_time' in n for n in notes)
    lead = SC.builtin('lead_brake')
    sp, _ = R.feasible_space(lead, SC.search_space(lead), {'a_brake_mps2': 7.0})
    assert next(p for p in sp if p['name'] == 'decel')['hi'] == 7.0
    same, notes = R.feasible_space(SC.builtin('stalled_vehicle'), SC.search_space(SC.builtin('stalled_vehicle')))
    assert same == SC.search_space(SC.builtin('stalled_vehicle')) and notes == []


def test_the_rss_objective_caps_what_is_not_avoidable():
    assert R.rss_objective(1.0, {'label': 'avoidable'}) == 1.0
    for lab in ('other', 'unavoidable', 'safe'):
        assert R.rss_objective(1.0, {'label': lab}) == R.NON_AVOIDABLE_CAP < 0.5
        assert R.rss_objective(0.2, {'label': lab}) == 0.2
    assert R.rss_objective(0.8, None) == 0.8
    hist = [{'params': {'a': 0.1}, 'value': R.rss_objective(1.0, {'label': 'other'}), 'i': 0},
            {'params': {'a': 0.2}, 'value': R.rss_objective(0.9, {'label': 'avoidable'}), 'i': 1}]
    assert [h['i'] for h in S.failures(hist, 0.5)] == [1]


# ------------------------------------------------------------------------------------------------- in a study
def test_a_sweep_labels_its_runs_and_counts_avoidable_failures(project, tmp_path, fake_carla):
    project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}

    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *a], capture_output=True, text=True,
                              env=env, timeout=900)
    r = cli('sweep', path, '--scenario', 'lead_brake', '--name', 'lb', '--grid', '2', '--fake-carla', fake_carla)
    assert r.returncode == 0, r.stdout + r.stderr
    ws = os.path.join(str(tmp_path), 'p.falcon', 'studies', 'lb')
    rows = [json.loads(x) for x in open(os.path.join(ws, 'results.jsonl'))]
    st = json.load(open(os.path.join(ws, 'study.json')))
    assert len(rows) == 16 and all(r['objective_key'] == 'criticality-rss' for r in rows)
    assert all(r['rss']['label'] in R.LABELS for r in rows)
    assert 'avoidable failures only' in st['objective']['about']
    for r in rows:
        if r['rss']['label'] == 'avoidable':
            assert r['objective'] == r['criticality']
        else:
            assert r['objective'] == min(r['criticality'], R.NON_AVOIDABLE_CAP)
    assert any(r['criticality'] >= 0.5 for r in rows)
    r = cli('sweep', path, '--scenario', 'lead_brake', '--name', 'lb', '--grid', '2', '--count', 'all',
            '--fake-carla', fake_carla)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [json.loads(x) for x in open(os.path.join(ws, 'results.jsonl'))]
    assert len(rows) == 16 and all(r['objective_key'] == 'criticality' and r['objective'] == r['criticality']
                                   for r in rows)                     # re-scored, not driven again
    from falcon_autonomy.core.reports import report_study
    html = report_study(ws).render()[0]
    assert '<th>RSS</th>' in html and any(lab in html for lab in R.LABELS)


# ------------------------------------------------------------------------------------------------- the GUI
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)


@pytest.mark.gui
def test_rare_cases_lists_counted_failures_first_then_the_others_with_their_label(window, qapp):
    page = window.go('rarecases')
    d = page.study_dir()
    os.makedirs(d, exist_ok=True)
    rows = [{'key': f'k{i}', 'values': {'speed': 30.0 + i, 'gap': 10.0, 'decel': 5.0, 'brake_at': 3.0}, 'run': f'/r/{i}', 'status': 'ok', 'criticality': c,
             'objective': R.rss_objective(c, {'label': lab}), 'rss': {'label': lab, 'reason': why}}
            for i, (c, lab, why) in enumerate([(1.0, 'other', 'the front road user braked harder than a_max,brake'),
                                               (0.9, 'avoidable', 'the ego braked less than a_min,brake'),
                                               (0.3, 'safe', 'not dangerous'),
                                               (1.0, 'unavoidable', 'inside its braking distance')])]
    with open(os.path.join(d, 'results.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(r) + '\n' for r in rows))
    page.load()
    items = [page.failures.topLevelItem(k) for k in range(page.failures.topLevelItemCount())]
    assert [it.text(4) for it in items] == ['avoidable', 'other (not counted)', 'unavoidable (not counted)']
    assert 'a_min,brake' in items[0].toolTip(4) and page.n_counted == 1
    assert '2 more not counted' in page.state.vals['State'].text()
    page.objective.setCurrentIndex(page.objective.findData('perception'))
    assert not page.count.isEnabled()
