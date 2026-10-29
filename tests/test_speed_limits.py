"""Traffic at each road's speed limit on an imported world, slowed for speed breakers (core.speedlimits,
workers.speedcontrol): the file's records read and looked up, the breaker approach, and the workers against the fake
carla (its one straight road: road 1, lanes ±1 ±2, s = x)."""
import json
import math
import types

import pytest

from conftest import worker_env  # noqa: F401  (the fake carla's environment for the worker subprocess)
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import speedlimits as SL


def _xodr(sections, road_type=None, road_id=1, unit=None):
    """A one-road OpenDRIVE file: sections [(s, {lane id: km/h or [(sOffset, km/h), …]})]."""
    def speeds(v):
        recs = v if isinstance(v, list) else [(0.0, v)]
        u = f' unit="{unit}"' if unit else ''
        return ''.join(f'<speed sOffset="{o}" max="{(k if unit == "km/h" else k / 3.6):.4f}"{u}/>' for o, k in recs)
    secs = []
    for s, lanes in sections:
        left = ''.join(f'<lane id="{i}" type="driving">{speeds(v)}</lane>' for i, v in sorted(lanes.items()) if i > 0)
        right = ''.join(f'<lane id="{i}" type="driving">{speeds(v)}</lane>' for i, v in sorted(lanes.items()) if i < 0)
        secs.append(f'<laneSection s="{s}"><left>{left}</left><center><lane id="0" type="none"/></center>'
                    f'<right>{right}</right></laneSection>')
    typ = (f'<type s="0" type="town"><speed max="{road_type}" unit="km/h"/></type>' if road_type else '')
    return (f'<OpenDRIVE><header/><road id="{road_id}" length="300" junction="-1">{typ}<planView/>'
            f'<lanes>{"".join(secs)}</lanes></road></OpenDRIVE>')


# ------------------------------------------------------------------------------------------------------- the file
def test_lane_records_by_section_and_offset_then_the_road_type():
    text = _xodr([(0.0, {-1: [(0.0, 50.0), (40.0, 30.0)], 1: 50.0}), (100.0, {-1: 80.0})], road_type=40.0)
    rs = SL.RoadSpeeds.from_xodr(text)
    kmh = lambda lane, s: (lambda v: (round(v[0] * 3.6, 2), v[1]) if v[0] else v)(rs.lookup(1, lane, s))
    assert kmh(-1, 10.0) == (50.0, 'lane')
    assert kmh(-1, 39.9) == (50.0, 'lane') and kmh(-1, 40.0) == (30.0, 'lane')     # sOffset within the section
    assert kmh(-1, 100.0) == (80.0, 'lane') and kmh(-1, 250.0) == (80.0, 'lane')   # the next section
    assert kmh(1, 120.0) == (40.0, 'road')        # lane 1 has no record in the second section: the road's type
    assert kmh(-1, -3.0) == (50.0, 'lane')        # s before the start (a waypoint at the very end) clamps to 0
    assert rs.lookup(99, -1, 0.0) == (None, None)
    assert SL.RoadSpeeds.from_xodr(_xodr([(0.0, {-1: 30.0})], unit='km/h')).lookup(1, -1, 0)[0] == \
        pytest.approx(30 / 3.6)
    assert rs.summary() == {'roads_with_limits': 1, 'limits_kmh': [30.0, 40.0, 50.0, 80.0]}


def test_no_limit_and_undefined_are_not_limits():
    text = ('<OpenDRIVE><road id="3" length="10"><type s="0" type="town"><speed max="no limit"/></type>'
            '<lanes><laneSection s="0"><right><lane id="-1" type="driving"><speed sOffset="0" max="undefined"/>'
            '</lane></right></laneSection></lanes></road></OpenDRIVE>')
    rs = SL.RoadSpeeds.from_xodr(text)
    assert len(rs) == 0 and rs.lookup(3, -1, 5.0) == (None, None)


def test_the_bengaluru_style_record_written_by_the_converter():
    """SUMO's netconvert (CARLA's Osm2Odr) writes <speed sOffset="0" max="13.89"/> in each lane, in m/s."""
    text = ('<OpenDRIVE><road id="7" length="50" junction="-1"><lanes><laneSection s="0"><left><lane id="1" '
            'type="driving"><speed sOffset="0" max="13.89" /></lane></left><right><lane id="-1" type="driving">'
            '<speed sOffset="0" max="8.33" /></lane></right></laneSection></lanes></road></OpenDRIVE>')
    rs = SL.RoadSpeeds.from_xodr(text)
    assert rs.lookup(7, 1, 3.0)[0] * 3.6 == pytest.approx(50.0, abs=0.01)
    assert rs.lookup(7, -1, 3.0)[0] * 3.6 == pytest.approx(30.0, abs=0.02)


# ------------------------------------------------------------------------------------------------------- breakers
def _breaker(x, y, kmh=25.0, hw=0.0, hdg=0.0):
    return {'x': x, 'y': y, 'hdg': hdg, 'crossing_kmh': kmh, 'half_width_left_m': hw, 'half_width_right_m': 0.0,
            'road': '1', 's': x, 'kind': 'hump'}


def test_breaker_approach_is_a_deceleration_profile_held_until_past():
    b = SL.Breakers([_breaker(100.0, 0.0), {'x': 0, 'y': 0, 'crossing_kmh': None}])   # a pothole: no crossing speed
    assert len(b) == 1
    v, T = 50 / 3.6, 0.25
    vc, a = 25 / 3.6, SL.APPROACH_DECEL_MPS2
    reach = SL.Breakers.approach_m(v, 25.0, T)
    assert reach == pytest.approx((v * v - vc * vc) / (2 * a) + v * T + SL.APPROACH_MARGIN_M)
    # the profile: sqrt(vc² + 2 a (d − v T − margin)), the crossing speed within that margin of the breaker
    for d in (5.0, 20.0, 40.0, reach):
        want = math.sqrt(vc * vc + 2 * a * max(0.0, d - v * T - SL.APPROACH_MARGIN_M)) * 3.6
        assert b.cap_kmh(100.0 - d, 0.0, 0.0, v, T) == pytest.approx(want)
    assert b.cap_kmh(100.0 - reach, 0.0, 0.0, v, T) == pytest.approx(50.0)      # where it starts to bite
    assert b.cap_kmh(100.0 - 5.0, 0.0, 0.0, v, T) == pytest.approx(25.0)
    assert b.cap_kmh(102.0, 0.0, 0.0, 7.0, T) == pytest.approx(25.0)            # on it: held until PAST_M beyond
    assert b.cap_kmh(100.0 + SL.PAST_M + 0.5, 0.0, 0.0, 7.0, T) is None
    assert b.cap_kmh(103.0, 0.0, 180.0, v, T) == pytest.approx(25.0)            # the other direction, heading to it
    assert b.cap_kmh(90.0, 0.0, 180.0, v, T) is None                            # driving away from it
    assert b.cap_kmh(95.0, 12.0, 0.0, v, T) is None                             # a road beside it, 12 m across
    assert b.cap_kmh(100.0, 20.0, -90.0, v, T) is None                          # crossing its road at a junction
    # CARLA's frame is OpenDRIVE's with y flipped: a breaker at OpenDRIVE y = +3.5 is at CARLA y = −3.5
    assert SL.Breakers([_breaker(50.0, 3.5)]).xy.tolist() == [[50.0, -3.5]]


def test_a_breaker_on_a_curve_is_found_along_the_lane_not_the_tangent():
    """On a 250 m radius a breaker 100 m ahead along the lane lies 20 m off the vehicle's heading: the straight-line
    test misses it, the lane's waypoints find it."""
    R, ahead = 250.0, 100.0
    th = ahead / R
    bx, by = R * math.sin(th), R - R * math.cos(th)            # CARLA frame: the vehicle at (0, 0) heading +x
    b = SL.Breakers([_breaker(bx, -by, hdg=-th)])               # (OpenDRIVE y is −CARLA y, hdg is −yaw)
    v, T = 80 / 3.6, 0.25
    assert b.cap_kmh(0.0, 0.0, 0.0, v, T) is None                                   # the tangent misses it
    path = [(d, R * math.sin(d / R), R - R * math.cos(d / R), math.degrees(d / R)) for d in range(5, 151, 5)]
    want = SL.allowed_mps(ahead, 25.0, T, v) * 3.6
    assert b.cap_kmh(0.0, 0.0, 0.0, v, T, path) == pytest.approx(want, abs=0.5)
    assert b.cap_kmh(0.0, 0.0, 0.0, v, T, path[:10]) is None                        # beyond the look-ahead


def test_features_json_next_to_the_world(tmp_path):
    (tmp_path / 'features.json').write_text(json.dumps({'placed': [_breaker(1.0, 2.0)]}))
    assert SL.features_path(str(tmp_path / 'world.xodr')) == str(tmp_path / 'features.json')
    assert len(SL.Breakers.from_features(str(tmp_path / 'features.json'))) == 1
    assert len(SL.Breakers.from_features(str(tmp_path / 'none.json'))) == 0


# ------------------------------------------------------------------------------------------------------- the control
@pytest.fixture
def fake(fake_module):
    return fake_module


def _world(carla, xodr):
    client = carla.Client('127.0.0.1', 2000)
    world = client.generate_opendrive_world(xodr)
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    return world, tm


def _car(carla, world, tm, x, y=0.0, yaw=0.0, role='traffic'):
    bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    bp.set_attribute('role_name', role)
    a = world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, 0.3), carla.Rotation(0.0, yaw, 0.0)))
    a.set_autopilot(True, tm.get_port())
    return a


def _speed(a):
    v = a.get_velocity()
    return math.hypot(v.x, v.y) * 3.6


def test_each_vehicle_at_its_lane_limit_with_its_difference_and_class_cap(fake, tmp_path):
    from falcon_autonomy.workers.speedcontrol import SpeedControl
    world, tm = _world(fake, _xodr([(0.0, {-1: 50.0, -2: 50.0, 1: 20.0, 2: 20.0}),
                                      (60.0, {-1: 80.0, -2: 80.0, 1: 20.0, 2: 20.0})]))
    w = types.SimpleNamespace(world=world, tm=tm)
    sc = SpeedControl(w, world.get_map().to_opendrive())
    a = _car(fake, world, tm, -40.0)                    # lane −1, heading +x: 50 km/h, then 80 from s = 60
    b = _car(fake, world, tm, 100.0, -3.5, 180.0)       # lane 1, heading −x: 20 km/h
    c = _car(fake, world, tm, -100.0, 3.5)              # lane −2: 10 % faster than the limit (diff −10)
    d = _car(fake, world, tm, -60.0, 3.5)               # lane −2, a stand-in: capped at 45 km/h
    sc.add(a)
    sc.add(b)
    sc.add(c, -10.0)
    sc.add(d, 0.0, 45.0, role='three_wheeler')
    world.tick()
    assert _speed(a) == pytest.approx(50.0, abs=0.1) and _speed(b) == pytest.approx(20.0, abs=0.1)
    assert _speed(c) == pytest.approx(55.0, abs=0.1) and _speed(d) == pytest.approx(45.0, abs=0.1)
    seen = []
    for k in range(int(14.0 / 0.05)):
        world.tick()
        sc.step((k + 1) * 0.05)
        if a.get_transform().location.x > 70.0:
            seen.append(_speed(a))
    assert seen and max(seen) == pytest.approx(80.0, abs=0.5)      # the second section's limit, reached
    assert _speed(d) == pytest.approx(45.0, abs=0.5)               # the class's top, below the 80 km/h limit
    rep = sc.report()
    assert rep['mode'] == 'map' and rep['samples_by_source']['lane'] > 0 and rep['samples_by_source']['carla'] == 0
    assert rep['vehicles'] == {'traffic': 3, 'three_wheeler': 1} and rep['limits_kmh'] == [20.0, 50.0, 80.0]


def test_vehicles_slow_before_a_lower_limit_not_after_it(fake):
    from falcon_autonomy.workers.speedcontrol import SpeedControl
    world, tm = _world(fake, _xodr([(0.0, {-1: 80.0, 1: 80.0}), (100.0, {-1: 30.0, 1: 30.0})]))
    w = types.SimpleNamespace(world=world, tm=tm)
    sc = SpeedControl(w, world.get_map().to_opendrive())
    a = _car(fake, world, tm, -140.0)
    sc.add(a)
    track = []
    for k in range(int(14.0 / 0.05)):
        world.tick()
        sc.step((k + 1) * 0.05)
        track.append((a.get_transform().location.x, _speed(a)))
    reach = SL.approach_m(80 / 3.6, 30.0, sc.EVERY_S)
    early = [v for x, v in track if x < 100.0 - reach - 15.0]
    at = [v for x, v in track if 99.0 <= x <= 103.0]
    assert early and min(early) == pytest.approx(80.0, abs=0.5)      # at its limit until the approach
    assert at and max(at) <= 30.0 + 2.0                               # at the lower limit where it begins
    assert sc.report()['updates_slowed_by'].get('limit ahead', 0) > 0


def test_a_road_without_records_keeps_carlas_own_limit(fake):
    from falcon_autonomy.workers.speedcontrol import SpeedControl
    world, tm = _world(fake, _xodr([(0.0, {-1: 50.0})], road_id=2))     # records for another road only
    w = types.SimpleNamespace(world=world, tm=tm)
    sc = SpeedControl(w, world.get_map().to_opendrive())
    a = _car(fake, world, tm, -100.0)
    sc.add(a, 10.0)
    world.tick()
    assert _speed(a) == pytest.approx(27.0, abs=0.1)        # the fake's 30 km/h get_speed_limit(), 10 % slower
    assert sc.report()['samples_by_source'] == {'lane': 0, 'road': 0, 'carla': 1}


def test_vehicles_slow_to_the_crossing_speed_before_a_breaker(fake, tmp_path):
    from falcon_autonomy.workers.speedcontrol import SpeedControl
    (tmp_path / 'features.json').write_text(json.dumps({'placed': [_breaker(50.0, 0.0, 25.0, hw=7.0)]}))
    world, tm = _world(fake, _xodr([(0.0, {-1: 60.0, -2: 60.0, 1: 60.0, 2: 60.0})]))
    w = types.SimpleNamespace(world=world, tm=tm)
    sc = SpeedControl(w, world.get_map().to_opendrive(), str(tmp_path / 'features.json'))
    a = _car(fake, world, tm, -120.0)
    sc.add(a)
    track = []
    for k in range(int(18.0 / 0.05)):
        world.tick()
        sc.step((k + 1) * 0.05)
        track.append((a.get_transform().location.x, _speed(a)))
    at = [v for x, v in track if 48.0 <= x <= 52.0]
    far = [v for x, v in track if x < -40.0]
    after = [v for x, v in track if x > 100.0]
    assert at and max(at) <= 25.0 + 1.0                       # crossed at the breaker's speed
    assert far and min(far) == pytest.approx(60.0, abs=0.5)   # at the limit until its approach
    assert after and max(after) == pytest.approx(60.0, abs=0.5)
    rep = sc.report()
    assert rep['breakers'] == 1 and rep['breaker_approaches'] == 1


def test_a_gone_vehicle_is_dropped(fake):
    from falcon_autonomy.workers.speedcontrol import SpeedControl
    world, tm = _world(fake, _xodr([(0.0, {-1: 50.0})]))
    w = types.SimpleNamespace(world=world, tm=tm)
    sc = SpeedControl(w, world.get_map().to_opendrive())
    a = _car(fake, world, tm, -100.0)
    sc.add(a)
    assert a.id in sc.rows

    def gone():
        raise RuntimeError('trying to operate on a destroyed actor')    # CARLA's, once the server removed it
    a.get_transform = gone
    sc.step(10.0)
    assert sc.rows == {}
    rep = sc.report()
    assert rep['vehicles'] == {'traffic': 1} and rep['removed_during_run'] == 1     # held once, gone since


# ------------------------------------------------------------------------------------------------------- the workers
def _drive(project, tmp_path, xodr_text, vehicles=6, **world_kw):
    import subprocess
    import sys
    from falcon_autonomy.core.drive import drive_config
    from falcon_autonomy.core.traffic import preset
    ws = tmp_path / 'test.falcon'
    ws.mkdir(exist_ok=True)
    (ws / 'w').mkdir(exist_ok=True)
    (ws / 'w' / 'world.xodr').write_text(xodr_text)
    prof = preset('tp-india-mixed')
    prof.update(vehicles=vehicles, walkers=0)
    project.doc['traffic_profiles'].append(prof)
    project.doc['worlds'].append(dict({'id': 'w-x', 'name': 'X', 'kind': 'osm', 'xodr': 'w/world.xodr',
                                       'osm': {'handedness': 'LHT'}, 'traffic_profile': 'tp-india-mixed',
                                       'weather': {'preset': 'ClearNoon'}}, **world_kw))
    cfg = drive_config(project, 'w-x', 'rig-default', str(tmp_path / 'run'), 2310, 8310, seconds=2.0, seed=3,
                       workspace=str(ws))
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK, r.stderr
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    frames = [json.loads(x) for x in (tmp_path / 'run' / 'frames.jsonl').read_text().splitlines()]
    return cfg, run, frames


def test_a_drive_on_an_imported_world_uses_the_files_limits(project, tmp_path):
    cfg, run, frames = _drive(project, tmp_path, _xodr([(0.0, {-1: 45.0, -2: 45.0, 1: 45.0, 2: 45.0})]))
    assert cfg['speed_limits'] == 'map'
    src = run['traffic']['speed_source']
    assert src['mode'] == 'map' and src['vehicles'].get('ego') == 1 and src['samples_by_source']['lane'] > 0
    assert src['samples_by_source']['carla'] == 0
    assert sum(src['vehicles'].values()) == 1 + sum(run['traffic']['placed'].values())
    # the traffic: the Indian profile's −10 % ± 25 % about the limit (a stand-in also capped at its class's top)
    acts = [json.loads(x) for x in (tmp_path / 'run' / 'actors.jsonl').read_text().splitlines()]
    v = {a['id']: math.hypot(a['vx'], a['vy']) * 3.6 for a in acts[0]['actors'] if a['role'] == 'falcon_traffic'}
    assert v and max(v.values()) <= 45.0 * 1.35 + 0.5 and len({round(x, 1) for x in v.values()}) > 1
    assert all(f['speed_mps'] * 3.6 <= 45.5 for f in frames)


def test_the_autopilot_ego_drives_at_the_files_limit(project, tmp_path):
    _, run, frames = _drive(project, tmp_path, _xodr([(0.0, {-1: 45.0, -2: 45.0, 1: 45.0, 2: 45.0})]), vehicles=0)
    assert run['traffic']['speed_source']['vehicles'] == {'ego': 1}
    assert all(f['speed_mps'] * 3.6 == pytest.approx(45.0, abs=0.05) for f in frames)


def test_carla_mode_leaves_the_traffic_manager_alone(project, tmp_path):
    cfg, run, frames = _drive(project, tmp_path, _xodr([(0.0, {-1: 45.0, -2: 45.0, 1: 45.0, 2: 45.0})]),
                              vehicles=0, speed_limits='carla')
    assert cfg['speed_limits'] == 'carla'
    assert run['traffic']['speed_source']['mode'] == 'carla'
    assert max(f['speed_mps'] * 3.6 for f in frames) < 40.0        # the fake's own autopilot speed (8 m/s)


def test_a_town_drive_has_no_speed_control_and_an_unchanged_config(project, tmp_path):
    from falcon_autonomy.core.drive import drive_config
    cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'r'), 2310, 8310, seconds=1.0)
    assert 'speed_limits' not in cfg          # a town's drives keep their cache keys
