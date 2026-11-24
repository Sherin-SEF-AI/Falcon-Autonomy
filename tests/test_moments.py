"""Moments (core.moments): planted events found in synthetic runs and none in a quiet one, the query table, the
incremental index, private runs, the exposure table, a hit made a scenario, the command line and the auto-index job.
No CARLA, no model, no network."""
import json
import math
import os

import numpy as np
import pytest

from falcon_autonomy.core import moments as M

DT = 0.05
NIGHT_RAIN = {'precipitation': 60.0, 'sun_altitude_angle': -20.0}
DAY = {'sun_altitude_angle': 45.0}


def lin(a, b, t0, t1):
    """A value going from a to b between t0 and t1 (held outside)."""
    return lambda t: a + (b - a) * min(max((t - t0) / (t1 - t0), 0.0), 1.0)


def actor(cls, x, y, yaw=0.0, size=(4.6, 1.84), cat=None, subclass=None, type_id=None):
    """x, y: functions of time (or numbers)."""
    cats = {'car': 'vehicle.car', 'motorcycle': 'vehicle.motorcycle', 'bicycle': 'vehicle.bicycle',
            'pedestrian': 'human.pedestrian.adult', 'bus': 'vehicle.bus.rigid', 'truck': 'vehicle.truck'}
    f = lambda v: v if callable(v) else (lambda t, v=v: v)      # noqa: E731
    return {'cls': cls, 'x': f(x), 'y': f(y), 'yaw': yaw, 'size': size, 'category': cat or cats.get(cls, ''),
            'subclass': subclass, 'type_id': type_id or ('walker.x' if cls == 'pedestrian' else 'vehicle.x')}


def write_run(ws, name, ego_x, actors=(), dur=8.0, weather=None, extra=None, settings=None, lights=None):
    """A run folder as the drive worker writes it: run.json, frames.jsonl, actors.jsonl (the ego's role is hero)."""
    d = os.path.join(ws, 'runs', name)
    os.makedirs(d)
    ex = ego_x if callable(ego_x) else (lambda t, v=ego_x: v * t)
    n = int(round(dur / DT)) + 1
    ts = [k * DT for k in range(n)]
    frames, rows = [], []
    for k, t in enumerate(ts):
        v = (ex(t + 1e-3) - ex(t - 1e-3)) / 2e-3
        f = {'step': k, 'sim_t': t, 'ego': {'x': ex(t), 'y': 0.0, 'yaw': 0.0}, 'speed_mps': v, 'files': {}, 'frame': k}
        if lights:
            f['traffic_light'] = lights(t, ex(t))
        frames.append(f)
        acts = [{'id': 1, 'role': 'hero', 'type_id': 'vehicle.lincoln.mkz_2020', 'category': 'vehicle.car',
                 'x': ex(t), 'y': 0.0, 'yaw': 0.0, 'vx': v, 'vy': 0.0, 'extent': [2.45, 0.92, 0.75],
                 'bbox_center': [0, 0, 0]}]
        for j, a in enumerate(actors):
            x, y = a['x'](t), a['y'](t)
            vx = (a['x'](t + 1e-3) - a['x'](t - 1e-3)) / 2e-3
            vy = (a['y'](t + 1e-3) - a['y'](t - 1e-3)) / 2e-3
            row = {'id': 10 + j, 'role': '', 'type_id': a['type_id'], 'category': a['category'], 'x': x, 'y': y,
                   'yaw': a['yaw'], 'vx': vx, 'vy': vy, 'extent': [a['size'][0] / 2, a['size'][1] / 2, 0.8],
                   'bbox_center': [0, 0, 0]}
            if a['subclass']:
                row['subclass'] = a['subclass']
            acts.append(row)
        rows.append({'step': k, 'frame': k, 'sim_t': t, 'actors': acts})
    with open(os.path.join(d, 'frames.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(f) + '\n' for f in frames))
    with open(os.path.join(d, 'actors.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(r) + '\n' for r in rows))
    st = dict({'weather': weather or DAY, 'fps': 20, 'project': {'world': 'w-test'}}, **(settings or {}))
    run = dict({'schema': 'falcon.run/1', 'status': 'ok', 'worker': 'drive', 'name': name,
                'started': '2026-01-01T00:00:00Z', 'provenance': {'settings': st}}, **(extra or {}))
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump(run, fh)
    return d


def moments_of(d):
    return M.extract(M.scene_from_run(d, xodr=''))


def types_of(d):
    return [m['type'] for m in moments_of(d)]


def one(d, typ):
    ms = moments_of(d)
    got = [m for m in ms if m['type'] == typ]
    assert len(got) == 1, (typ, [m['type'] for m in ms])
    return got[0]


# ------------------------------------------------------------------------------------------------ planted events
def test_a_quiet_run_has_no_moments(tmp_path):
    d = write_run(str(tmp_path), 'quiet', 10.0, [actor('car', lambda t: 40 + 10 * t, 0.0),
                                                 actor('car', lambda t: 20 + 10 * t, -3.5),
                                                 actor('pedestrian', 30.0, 8.0, size=(0.5, 0.5))])
    assert types_of(d) == []


def test_cut_in_and_cut_out_with_side_gap_and_class(tmp_path):
    ws = str(tmp_path)
    d = write_run(ws, 'cutin', 10.0, [actor('motorcycle', lambda t: 14 + 9 * t, lin(-3.5, 0.0, 2.0, 4.0),
                                            size=(2.2, 0.8))], weather=NIGHT_RAIN)
    m = one(d, 'cut_in')
    assert m['side'] == 'left' and m['actors'] == ['motorcycle'] and 2.0 < m['t0'] < 4.0
    assert 5.0 < m['gap_m'] < 10.0 and m['rel_speed_kmh'] == pytest.approx(3.6, abs=0.3)
    assert m['ego_speed_kmh'] == pytest.approx(36.0, abs=0.5) and m['speed_band'] == 'medium'
    assert types_of(d) == ['cut_in']
    d = write_run(ws, 'cutout', 10.0, [actor('car', lambda t: 20 + 10 * t, lin(0.0, 3.5, 2.0, 4.0))])
    m = one(d, 'cut_out')
    assert m['side'] == 'right' and m['actors'] == ['car'] and types_of(d) == ['cut_out']


def test_hard_braking_of_the_lead_and_of_the_ego(tmp_path):
    def lead(t):                                    # 10 m/s, braking at 5 m/s² from 2.0 to 3.5 s, then 2.5 m/s
        if t < 2.0:
            return 40 + 10 * t
        if t < 3.5:
            return 60 + 10 * (t - 2) - 2.5 * (t - 2) ** 2
        return 60 + 15 - 5.625 + 2.5 * (t - 3.5)

    def ego(t):                                     # 10 m/s, braking at 4 m/s² from 3.0 to 5.0 s, then 2 m/s
        if t < 3.0:
            return 10 * t
        if t < 5.0:
            return 30 + 10 * (t - 3) - 2.0 * (t - 3) ** 2
        return 30 + 20 - 8 + 2 * (t - 5)
    d = write_run(str(tmp_path), 'brake', ego, [actor('bus', lead, 0.0, size=(11.0, 2.5))])
    ts = types_of(d)
    lb = one(d, 'hard_brake_lead')
    assert lb['actors'] == ['bus'] and lb['decel_mps2'] == pytest.approx(5.0, abs=0.3) and 1.9 < lb['t0'] < 2.3
    eb = one(d, 'hard_brake_ego')
    assert eb['actors'] == ['bus'] and eb['decel_mps2'] == pytest.approx(4.0, abs=0.3) and 'collision' not in ts


def test_close_pass_and_overtaking_either_way(tmp_path):
    ws = str(tmp_path)
    d = write_run(ws, 'pass', 10.0, [actor('bicycle', lambda t: 20 + 3 * t, 1.9, size=(1.8, 0.6))])
    cp = one(d, 'close_pass')
    assert cp['side'] == 'right' and cp['actors'] == ['bicycle'] and cp['gap_m'] == pytest.approx(0.68, abs=0.05)
    ov = one(d, 'overtake')
    assert ov['numbers']['by'] == 'ego' and ov['side'] == 'right'
    d = write_run(ws, 'passed', 10.0, [actor('car', lambda t: -25 + 15 * t, -3.5)])
    ov = one(d, 'overtake')
    assert ov['numbers']['by'] == 'actor' and ov['side'] == 'left' and types_of(d) == ['overtake']


def test_crossing_pedestrian_and_stopped_cattle(tmp_path):
    ws = str(tmp_path)
    d = write_run(ws, 'cross', 8.0, [actor('pedestrian', 60.0, lambda t: -6 + 1.5 * (t - 0.5), yaw=90.0,
                                           size=(0.5, 0.5))])
    c = one(d, 'crossing')
    assert c['side'] == 'left' and c['actors'] == ['pedestrian']
    assert c['numbers']['walk_mps'] == pytest.approx(1.5, abs=0.05)
    assert 'jaywalking' not in types_of(d)           # no map: not claimed
    d = write_run(ws, 'cow', 5.0, [actor('cattle', 80.0, 0.0, size=(1.6, 0.45), subclass='cattle',
                                         type_id='static.prop.cow', cat='movable_object.pushable_pullable')])
    s = one(d, 'stopped_obstacle')
    assert s['actors'] == ['cattle'] and types_of(d) == ['stopped_obstacle']


def test_oncoming_low_ttc_collision_and_near_miss(tmp_path):
    ws = str(tmp_path)
    d = write_run(ws, 'headon', 10.0, [actor('car', lambda t: 120 - 10 * t, 0.0, yaw=180.0)], dur=7.0)
    ts = types_of(d)
    assert {'oncoming_in_lane', 'low_ttc', 'high_decel_needed', 'collision'} <= set(ts) and 'near_miss' not in ts
    col = one(d, 'collision')
    assert col['severity'] == 1.0 and col['rel_speed_kmh'] == pytest.approx(72.0, abs=1.0) and 5.5 < col['t0'] < 5.9
    assert one(d, 'low_ttc')['min_ttc_s'] < M.LOW_TTC_S
    d = write_run(ws, 'brush', 10.0, [actor('car', lambda t: 120 - 10 * t, -2.2, yaw=180.0)], dur=7.0)
    nm = one(d, 'near_miss')
    assert nm['side'] == 'left' and nm['gap_m'] == pytest.approx(0.36, abs=0.05) and 'collision' not in types_of(d)


def test_faults_mrm_and_a_red_light(tmp_path):
    ws = str(tmp_path)
    d = write_run(ws, 'fault', 8.0, extra={
        'faults_applied': {'cam_front': {'planned': [{'kind': 'blackout', 'onset_s': 2.0, 'until_s': 3.0}],
                                         'frames': 20}},
        'mrm': {'onset_s': 2.0, 'passed': True, 'measured': {'standstill_s': 6.0, 'max_decel_mps2': 2.5}}})
    f = one(d, 'sensor_fault')
    assert (f['t0'], f['t1']) == (2.0, 3.0) and f['numbers'] == {'fault': 'blackout', 'sensor': 'cam_front'}
    m = one(d, 'mrm')
    assert (m['t0'], m['t1']) == (2.0, 6.0) and m['numbers']['passed'] is True
    d = write_run(ws, 'red', 8.0, lights=lambda t, x: {'state': 'Red', 'distance_m': 30.0 - x})
    r = one(d, 'red_light_violation')
    assert r['t0'] == pytest.approx(30.0 / 8.0, abs=0.06)
    d = write_run(ws, 'green', 8.0, lights=lambda t, x: {'state': 'Green', 'distance_m': 30.0 - x})
    assert types_of(d) == []


def test_map_moments_on_a_scene_lane_departure_and_jaywalking():
    n = 161
    t = np.arange(n) * DT
    sc = M.Scene(t, 8.0 * t, np.zeros(n), np.zeros(n), np.full(n, 8.0))
    sc.sig['on_lane'] = np.where((t > 2.0) & (t < 3.0), 0.0, 1.0)
    sc.sig['in_junction'] = np.zeros(n)
    py = -6 + 1.5 * (t - 0.5)
    sc.add_actor('p', 'pedestrian', np.arange(n), np.full(n, 60.0), py, 90.0, 0.0, 1.5, 0.5, 0.5)
    ms = M.extract(sc)
    assert [m['type'] for m in ms] == ['lane_departure', 'crossing', 'jaywalking']
    assert ms[0]['t0'] == pytest.approx(2.05, abs=0.06) and ms[0]['road'] == 'road'
    assert 'heuristic' in ms[2]['numbers']['basis']
    sc.sig['ped_on_crossing'] = np.ones(n)                    # the map says it is on a crosswalk: not jaywalking
    assert [m['type'] for m in M.extract(sc)] == ['lane_departure', 'crossing']
    sc.sig['in_junction'] = np.ones(n)
    assert M.extract(sc)[1]['road'] == 'junction'
    assert M.extract(sc) == M.extract(sc)                     # deterministic


def test_thresholds_are_named_and_severity_orders():
    assert M.THRESHOLDS['LOW_TTC_S'] == 1.5 and M.THRESHOLDS['HARD_BRAKE_MPS2'] == 3.0
    assert 'VERSION' not in M.THRESHOLDS
    assert set(M.SEVERITY_BASE) == set(M.TYPES) == set(M.NAMES)
    sev = lambda **kw: M.severity(dict({'type': 'low_ttc'}, **kw))      # noqa: E731
    assert M.severity({'type': 'collision'}) == 1.0 and sev(min_ttc_s=0.3) > sev(min_ttc_s=1.4) >= 0.5
    assert sev(min_ttc_s=0.0) == 0.9


# ------------------------------------------------------------------------------------------------ the query table
QUERIES = [
    ('two-wheeler cuts in from the left at night in rain',
     {'types': ['cut_in'], 'actors': ['motorcycle', 'bicycle'], 'side': 'left', 'tod': 'night', 'weather': 'rain'}, []),
    ('auto cut in with TTC under 1.5 s', {'types': ['cut_in'], 'actors': ['autorickshaw'],
                                          'where': [('ttc', '<', 1.5)]}, []),
    ('cow standing on the road', {'types': ['stopped_obstacle'], 'actors': ['cattle']}, []),
    ('bike overtakes us above 40 km/h', {'types': ['overtake'], 'actors': ['motorcycle'],
                                         'where': [('speed', '>', 40.0)]}, []),
    ('pedestrian jaywalking at dusk', {'types': ['jaywalking'], 'actors': ['pedestrian'], 'tod': 'twilight'}, []),
    ('near miss with a bus at a junction', {'types': ['near_miss'], 'actors': ['bus'], 'road': 'junction'}, []),
    ('ego brakes hard, deceleration at least 5 m/s2', {'types': ['hard_brake_ego'], 'where': [('decel', '>=', 5.0)]},
     []),
    ('truck brakes suddenly', {'types': ['hard_brake_lead'], 'actors': ['truck']}, []),
    ('hard braking', {'types': ['hard_brake_ego', 'hard_brake_lead']}, []),
    ('oncoming scooty in our lane, gap below 20 m', {'types': ['oncoming_in_lane'], 'actors': ['motorcycle'],
                                                     'where': [('gap', '<', 20.0)]}, []),
    ('ran a red light', {'types': ['red_light_violation']}, []),
    ('camera fault then MRM', {'types': ['sensor_fault', 'mrm']}, []),
    ('collision with a child', {'types': ['collision'], 'actors': ['child']}, []),
    ('cut out to the right faster than 10 m/s', {'types': ['cut_out'], 'side': 'right',
                                                  'where': [('speed', '>', 36.0)]}, []),
    ('close pass by a cyclist without a collision', {'types': ['close_pass'], 'actors': ['bicycle'],
                                                     'not_types': ['collision']}, []),
    ('zebra unicorn cuts in quantumly', {'types': ['cut_in']}, ['zebra', 'unicorn', 'quantumly']),
    ('', {}, []),
]


@pytest.mark.parametrize('text,want,unknown', QUERIES, ids=[q[0][:30] or 'empty' for q in QUERIES])
def test_query_table(text, want, unknown):
    pq = M.parse_query(text)
    assert pq['filter'] == M.new_filter(**want), pq
    assert pq['not_understood'] == unknown
    assert pq == M.parse_query(text)
    assert M.check_filter(pq['filter']) == pq['filter']


def test_the_filter_is_shown_back_and_bad_values_refused():
    pq = M.parse_query('two wheeler cuts in from the left at night in rain, TTC under 1.5 s')
    assert pq['understood'] == ('type cut_in · actor motorcycle or bicycle · side left · light night · weather rain '
                                '· min TTC < 1.5 s')
    assert M.understood(M.new_filter()) == '(everything)'
    assert M.check_filter(M.new_filter(actors=['two_wheeler', 'cattle']))['actors'] == ['motorcycle', 'bicycle',
                                                                                         'cattle']
    for bad in ({'types': ['teleport']}, {'actors': ['dragon']}, {'side': 'up'}, {'where': ['ttc<<1']},
                {'where': ['mass<3']}):
        with pytest.raises(M.MomentsError):
            M.check_filter(M.new_filter(**bad))
    assert M.parse_where('ttc<=1.5') == ('ttc', '<=', 1.5)


# ------------------------------------------------------------------------------------------------ the index
@pytest.fixture()
def ws(tmp_path):
    w = str(tmp_path / 'ws')
    os.makedirs(w)
    write_run(w, 'cutin-night', 10.0, [actor('motorcycle', lambda t: 14 + 9 * t, lin(-3.5, 0.0, 2.0, 4.0),
                                             size=(2.2, 0.8))], weather=NIGHT_RAIN)
    write_run(w, 'cutin-day', 10.0, [actor('motorcycle', lambda t: 14 + 9 * t, lin(3.5, 0.0, 2.0, 4.0),
                                           size=(2.2, 0.8))])
    write_run(w, 'headon', 10.0, [actor('car', lambda t: 120 - 10 * t, 0.0, yaw=180.0)], dur=7.0)
    write_run(w, 'quiet', 10.0, [actor('car', lambda t: 40 + 10 * t, 0.0)])
    write_run(w, 'route', 10.0, [actor('car', lambda t: 20 + 10 * t, lin(-3.5, 0.0, 2.0, 4.0))],
              settings={'dashcam_route': {'id': 'r'}})
    return w


def test_index_is_incremental_and_never_touches_the_runs(ws):
    def snap():
        out = {}
        for root, _d, files in os.walk(os.path.join(ws, 'runs')):
            for f in files:
                p = os.path.join(root, f)
                out[p] = (os.path.getsize(p), os.stat(p).st_mtime_ns)
        return out
    before = snap()
    r = M.index(ws, xodr='')
    assert (r['sources'], r['indexed'], r['unchanged'], r['removed'], r['failed']) == (5, 5, 0, 0, [])
    assert r['private_runs'] == 1 and r['moments'] >= 6 and snap() == before
    assert os.path.isfile(M.db_path(ws))
    r2 = M.index(ws, xodr='')
    assert (r2['indexed'], r2['unchanged'], r2['moments']) == (0, 5, r['moments'])
    # a changed run is read again, a new one added, a deleted one removed
    q = os.path.join(ws, 'runs', 'quiet')
    with open(os.path.join(q, 'run.json')) as fh:
        doc = json.load(fh)
    doc['faults_applied'] = {'lidar': {'planned': [{'kind': 'dropout', 'onset_s': 1.0}]}}
    with open(os.path.join(q, 'run.json'), 'w') as fh:
        json.dump(doc, fh)
    write_run(ws, 'new', 5.0, [actor('cattle', 80.0, 0.0, size=(1.6, 0.45), subclass='cattle')])
    import shutil
    shutil.rmtree(os.path.join(ws, 'runs', 'headon'))
    r3 = M.index(ws, xodr='')
    assert (r3['sources'], r3['indexed'], r3['unchanged'], r3['removed']) == (5, 2, 3, 1)
    got = M.search(ws, M.new_filter(types=['sensor_fault', 'stopped_obstacle', 'collision']))
    assert sorted(m['type'] for m in got['hits']) == ['sensor_fault', 'stopped_obstacle']
    # one run only (the job after a run), and a folder that is not a recorded run
    os.makedirs(os.path.join(ws, 'runs', 'empty'))
    with open(os.path.join(ws, 'runs', 'empty', 'run.json'), 'w') as fh:
        json.dump({'status': 'failed'}, fh)
    r4 = M.index(ws, only=[os.path.join(ws, 'runs', 'empty')], xodr='')
    assert r4['indexed'] == 1 and 'frames.jsonl' in r4['failed'][0]['why']
    assert M.index(ws, xodr='')['unchanged'] == 6
    assert M.index(ws, force=True, xodr='')['indexed'] == 6


def test_search_ranks_full_matches_then_partial_then_severity(ws):
    M.index(ws, xodr='')
    pq = M.parse_query('two-wheeler cuts in from the left at night in rain')
    res = M.search(ws, pq['filter'])
    assert [(os.path.basename(m['rel']), m['match']) for m in res['hits']] == [
        ('cutin-night', 1.0), ('cutin-day', 0.25), ('route', 0.25)]
    assert res['hits'][1]['missed'] == ['side', 'tod', 'weather'] and res['full'] == 1 and res['total'] == 3
    top = res['hits'][0]
    assert top['tod'] == 'night' and top['weather'] == 'rain' and top['world'] == 'w-test'
    assert [m['rel'] for m in M.search(ws, pq['filter'], strict=True)['hits']] == [top['rel']]
    # numeric constraints are required; severity orders within a match level
    res = M.search(ws, M.new_filter(where=[('ttc', '<', 1.0)]))
    assert res['hits'] and all(m['min_ttc_s'] < 1.0 for m in res['hits'])
    assert [m['severity'] for m in res['hits']] == sorted((m['severity'] for m in res['hits']), reverse=True)
    assert res['hits'][0]['type'] == 'collision'
    m = M.get(ws, res['hits'][0]['id'])
    assert m['type'] == 'collision' and M.get(ws, 'nope-0') is None
    assert M.search(str(os.path.join(ws, 'nowhere')))['hits'] == []


def test_private_runs_are_flagged_and_only_counted_for_the_assistant(ws):
    M.index(ws, xodr='')
    f = M.new_filter(types=['cut_in'])
    local = M.search(ws, f)
    assert sorted(os.path.basename(m['rel']) for m in local['hits']) == ['cutin-day', 'cutin-night', 'route']
    assert [m['private'] for m in local['hits'] if m['rel'].endswith('route')] == [True]
    pub = M.search(ws, f, public_only=True)
    assert sorted(os.path.basename(m['rel']) for m in pub['hits']) == ['cutin-day', 'cutin-night']
    assert pub['private_left_out'] == 1 and local['private_left_out'] == 0
    s_all, s_pub = M.stats(ws), M.stats(ws, public_only=True)
    assert s_all['runs'] == 5 and s_pub['runs'] == 4 and s_pub['private_left_out'] == 1
    cut = lambda s: next(r for r in s['rows'] if r['type'] == 'cut_in')        # noqa: E731
    assert cut(s_all)['count'] == 3 and cut(s_pub)['count'] == 2


def test_a_near_miss_ranking_is_indexed_private_without_clip_detail(ws, tmp_path):
    ing = tmp_path / 'footage' / 'falcon-ingest'
    os.makedirs(ing)
    with open(ing / 'ranking.json', 'w') as fh:
        json.dump({'schema': 'falcon.nearmiss_ranking/1', 'events': [
            {'id': 'abcdef0123456789', 'kind': 'autorickshaw', 'window_s': [12.0, 18.0], 'min_ttc_s': 0.8,
             'min_distance_m': 1.2, 'max_drac_mps2': 4.0, 'score_median': 0.66, 'overlap': False,
             'clip_key': 'SECRET_CLIP_NAME'}]}, fh)
    r = M.index(ws, nearmiss=[str(tmp_path / 'footage')], xodr='')
    assert r['sources'] == 6 and r['private_runs'] == 2
    res = M.search(ws, M.parse_query('near miss with an auto')['filter'])
    m = res['hits'][0]
    assert m['private'] and m['source'] == 'nearmiss' and m['actors'] == ['autorickshaw'] and m['min_ttc_s'] == 0.8
    assert 'SECRET' not in json.dumps(res, default=str) and 'footage' not in m['rel']
    pub = M.search(ws, res['filter'], public_only=True)
    assert (pub['total'], pub['hits'], pub['private_left_out']) == (0, [], 1)
    assert M.stats(ws)['without_exposure'] == {'near_miss': 1}
    assert M.index(ws, xodr='')['unchanged'] == 6               # the ranking is remembered
    with pytest.raises(M.MomentsError):
        M.index(ws, nearmiss=[str(tmp_path / 'nothing')])
    with pytest.raises(M.MomentsError):
        M.make_scenario(ws, m, 'x')


def test_exposure_table(ws):
    M.index(ws, xodr='')
    s = M.stats(ws)
    hours = (8.0 * 4 + 7.0) / 3600.0
    assert s['hours'] == pytest.approx(hours, rel=1e-3) and s['km'] == pytest.approx(0.39, abs=0.01)
    row = next(r for r in s['rows'] if r['type'] == 'cut_in')
    assert row['count'] == 3 and row['per_hour'] == pytest.approx(3 / hours, rel=1e-3)
    assert row['per_hour_ci'][0] < row['per_hour'] < row['per_hour_ci'][1]
    assert [r['type'] for r in s['rows']] == list(M.TYPES)
    lo, hi = M.poisson_interval(10)
    assert lo == pytest.approx(4.80, abs=0.05) and hi == pytest.approx(18.39, abs=0.05)     # the exact interval's
    assert M.poisson_interval(0)[0] == 0.0
    by = M.stats(ws, by='tod')
    assert {r['tod'] for r in by['rows']} == {'day', 'night'} and all(r['count'] for r in by['rows'])
    assert M.stats(str(os.path.join(ws, 'none')))['rows'] == []


def test_a_hit_becomes_a_scenario_through_the_incident_path(ws):
    M.index(ws, xodr='')
    hit = M.search(ws, M.parse_query('motorcycle cuts in from the left at night')['filter'])['hits'][0]
    res = M.make_scenario(ws, hit, 'night-cut-in')
    d = os.path.join(ws, 'moments', 'scenarios', 'night-cut-in')
    assert res['dir'] == d and res['actors'] == 1 and not res['private'] and res['parameters'] >= 3
    assert sorted(os.listdir(d)) == ['moment.json', 'moment.xosc', 'scenario.json', 'variations.json']
    with open(os.path.join(d, 'variations.json')) as fh:
        vr = json.load(fh)
    assert vr['actors'][0]['kind'] == 'motorcycle' and vr['actors'][0]['place']['lane'] == 'adjacent'
    assert vr['source'] == {'format': 'falcon-run', 'kind': 'moment', 'run': hit['rel'], 'moment': hit['id']}
    assert any(a['do'] == 'lane_change' for e in vr['timeline'] for a in e['actions'])
    with open(os.path.join(d, 'moment.xosc')) as fh:
        assert '<OpenSCENARIO' in fh.read()
    from falcon_autonomy.assistant.privacy import private_reason
    assert private_reason(d) is None
    priv = next(m for m in M.search(ws, M.new_filter(types=['cut_in']))['hits'] if m['private'])
    res = M.make_scenario(ws, priv, 'from-route')
    assert res['private'] and private_reason(res['dir'])
    with pytest.raises(M.MomentsError):
        M.make_scenario(ws, hit, '../x')
    from falcon_autonomy.core.project import Edit
    edit, ds = M.drive_set_edit({}, ws, 'cut-ins', M.search(ws, M.new_filter(types=['cut_in']))['hits'])
    assert isinstance(edit, Edit) and len(ds['runs']) == 3 and ds['runs'][0].startswith('runs' + os.sep)


def test_an_imported_log_is_indexed(ws):
    from falcon_autonomy.core.logreplay.model import Log, Track
    t = np.arange(0, 8.01, 0.1)
    ego = Track('ego', 'car', t, 10 * t, 0 * t, 0 * t, 0 * t)
    mc = Track('m1', 'motorcycle', t, 14 + 9 * t, np.array([lin(-3.5, 0.0, 2.0, 4.0)(x) for x in t]), 0 * t, 0 * t)
    Log(ego, [mc], {'name': 'log1'}).save(os.path.join(ws, 'logs', 'log1'))
    r = M.index(ws, xodr='')
    assert r['sources'] == 6 and not r['failed']
    hits = M.search(ws, M.new_filter(types=['cut_in'], source='log'), strict=True)['hits']
    assert len(hits) == 1 and hits[0]['side'] == 'left' and hits[0]['actors'] == ['motorcycle']
    assert hits[0]['tod'] is None                               # a log records no light: not invented


# ------------------------------------------------------------------------------------------------ the command line
def test_cli(ws, capsys):
    assert M.main_moments(['index', ws, '--no-map']) == 0
    assert '5 source(s): 5 read' in capsys.readouterr().out
    assert M.main_moments(['search', ws, 'two-wheeler', 'cuts in from the left at night in rain, zebra']) == 0
    out = capsys.readouterr().out
    assert 'understood as: type cut_in · actor motorcycle or bicycle · side left · light night · weather rain' in out
    assert 'not understood: zebra' in out and '[partial: not side, tod, weather]' in out and '(private)' in out
    assert '3 moment(s), 1 matching every term' in out
    assert M.main_moments(['search', ws, '--type', 'cut_in', '--actor', 'two_wheeler', '--night', '--strict',
                           '--json']) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['total'] == 1 and doc['hits'][0]['tod'] == 'night'
    mid = doc['hits'][0]['id']
    assert M.main_moments(['search', ws, '--where', 'ttc<1', '--limit', '1']) == 0
    assert 'collision' in capsys.readouterr().out
    assert M.main_moments(['search', ws, '--actor', 'dragon']) == 2
    assert 'unknown actor class' in capsys.readouterr().out
    assert M.main_moments(['show', ws, mid]) == 0
    assert 'cut_in' in capsys.readouterr().out
    assert M.main_moments(['show', ws, mid, '--scenario', 'cli-scn']) == 0
    assert 'variations.json' in capsys.readouterr().out
    assert M.main_moments(['show', ws, 'zzz']) == 2
    capsys.readouterr()
    assert M.main_moments(['stats', ws]) == 0
    out = capsys.readouterr().out
    assert 'cut_in' in out and '/h' in out and 'not real traffic' in out
    assert M.main_moments(['stats', ws, '--by', 'weather', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['by'] == 'weather'


def test_the_auto_index_job_and_setting(ws):
    from falcon_autonomy.core import protocol as P
    from falcon_autonomy.core.settings import DEFAULTS, EXPLAIN
    assert DEFAULTS['moments_auto_index'] is False and 'moments_auto_index' in EXPLAIN

    class Q:
        def __init__(self):
            self.specs = []

        def submit(self, spec):
            self.specs.append(spec)
            return spec

    class Svc:
        workspace, queue = ws, Q()

    class Job:
        def __init__(self, run_dir, code=P.OK, kind='drive'):
            self.exit_code = code
            self.spec = type('S', (), {'run_dir': run_dir, 'kind': kind})()
    d = os.path.join(ws, 'runs', 'quiet')
    spec = M.after_job(Svc, Job(d))
    assert spec.kind == 'moments-index' and spec.argv[-5:] == ['moments', 'index', ws, '--run', d]
    assert not spec.needs_gpu and not spec.needs_carla
    assert M.after_job(Svc, Job(d, code=P.CRASHED)) is None and M.after_job(Svc, Job(d, kind='moments-index')) is None
    assert M.after_job(Svc, Job(os.path.join(ws, 'runs'))) is None
    assert math.isfinite(M.LOG_DT_S)
