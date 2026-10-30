"""Lateral filtering (workers.lateral): the chooser's arithmetic, the curvilinear frame, the profile setting, and the
filter against the fake carla — whose traffic manager, like CARLA 0.9.16's (measured, docs/indian-assets.md), keeps
looking along the lane's centre whatever a vehicle's lane offset."""
import math

import numpy as np
import pytest

from conftest import worker_env  # noqa: F401  (the fake carla's environment for the worker subprocess)
from falcon_autonomy.workers import lateral as LAT


def _others(*rows):
    """(s, d, hl, hw, v, kind) rows → choose()'s arrays."""
    keys = ('s', 'd', 'hl', 'hw', 'v')
    out = {k: np.array([r[i] for r in rows], float) for i, k in enumerate(keys)}
    out['clear'] = np.array([LAT.CLEAR_M[r[5]] for r in rows], float)
    out['sig'] = np.array([LAT.SIGMA_D_M[r[5]] for r in rows], float)
    return out


CANDS = np.arange(-1.2, 4.81, LAT.STRIP_M)        # a two-lane carriageway: the vehicle's lane and one to its right


# ------------------------------------------------------------------------------------------------- the chooser
def test_on_a_free_road_it_rides_at_its_home_offset():
    for home in (-0.68, 0.68):                   # the near side: left in left-hand traffic, right in right-hand
        r = LAT.choose(0.0, 0.45, 1.1, 8.0, CANDS, {}, home, home)
        assert abs(r['target'] - home) <= LAT.STRIP_M / 2 + 1e-9
        assert r['gap'] == LAT.LOOK_M and r['lead_v'] is None and LAT.cap_mps(r['gap'], r['lead_v']) is None


def test_it_filters_past_a_stopped_car_on_its_lane_centre():
    car = (25.0, 0.0, 2.3, 0.93, 0.0, 'car')
    r = LAT.choose(0.0, 0.45, 1.1, 8.0, CANDS, _others(car), 0.0, 0.0)
    # its strip clears the car's side and the car's clearance
    assert abs(r['target']) >= 0.93 + LAT.CLEAR_M['car'] + 0.45 - 1e-9
    assert r['gap'] == LAT.LOOK_M
    # where it is, the car is in its way: the centre strip's gap is the gap to the car
    c = LAT.choose(0.0, 0.45, 1.1, 8.0, [0.0], _others(car), 0.0, 0.0)
    assert c['gap'] == pytest.approx(25.0 - 2.3 - 1.1) and c['lead_v'] == 0.0
    v = LAT.cap_mps(c['gap'], c['lead_v'])
    assert v == pytest.approx(math.sqrt(2 * LAT.DECEL_MPS2 * (c['gap'] - LAT.STANDSTILL_M)))


def test_too_close_to_get_across_it_brakes_until_it_can():
    """At 8 m/s, 12 m behind a stopped car, LAT_RATE_MPS cannot take it clear of the car before it is level: every
    path meets the car; slowed to 2 m/s it can get across."""
    car = (12.0, 0.0, 2.3, 0.93, 0.0, 'car')
    fast = LAT.choose(0.0, 0.45, 1.1, 8.0, CANDS, _others(car), 0.0, 0.0)
    assert fast['gap'] < LAT.LOOK_M and LAT.cap_mps(fast['gap'], fast['lead_v']) < 8.0
    slow = LAT.choose(0.0, 0.45, 1.1, 2.0, CANDS, _others(car), 0.0, 0.0)
    assert slow['gap'] == LAT.LOOK_M and abs(slow['target']) >= 1.68 - 1e-9


def test_a_bus_repels_from_further_across_than_a_motorcycle():
    """The anisotropic field: the same edge gap costs more beside a bus than beside a motorcycle."""
    def cost(kind, hw):
        o = _others((4.0, 0.0, 1.0, hw, 6.0, kind))
        return [LAT.choose(0.0, 0.45, 1.1, 6.0, [c], o, c, c)['score'] for c in (hw + 0.45 + 0.6,)][0]
    assert cost('bus', 1.3) > cost('motorcycle', 1.3) + 0.1


def test_it_does_not_move_sideways_into_a_vehicle_alongside():
    beside = (0.5, 2.0, 2.3, 0.93, 8.0, 'car')            # a car level with it, to its right
    slow = (8.0, 0.0, 2.3, 0.93, 1.0, 'car')               # a slow car ahead on its lane
    r = LAT.choose(0.0, 0.45, 1.1, 8.0, CANDS, _others(beside, slow), 0.0, 0.0)
    assert r['target'] < 1.0                               # not through the car beside it
    assert not r['blocked']


def test_project_is_curvilinear_on_a_curve():
    R = 30.0                                               # a quarter circle of radius 30 m, turning right
    th = np.linspace(0.0, math.pi / 2, 40)
    path = {'x': R * np.sin(th), 'y': R - R * np.cos(th)}
    path['cum'] = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(path['x']), np.diff(path['y'])))])
    # a point 1 m outside the curve (to the left of travel) a third of the way round
    a = math.pi / 6
    px, py = (R + 1.0) * math.sin(a), R - (R + 1.0) * math.cos(a)
    s, d = LAT.project(np.array([px]), np.array([py]), path)
    assert s[0] == pytest.approx(R * a, abs=0.05) and d[0] == pytest.approx(-1.0, abs=0.02)


def test_profile_settings():
    assert LAT.settings_of({}) is None and LAT.settings_of({'lateral_filtering': {'enabled': False}}) is None
    s = LAT.settings_of({'lateral_filtering': {'enabled': True}})
    assert s['vehicles'] == ['motorcycle', 'bicycle'] and s['controller'] == 'tm'
    with pytest.raises(ValueError):
        LAT.settings_of({'lateral_filtering': {'enabled': True, 'vehicles': ['bus']}})
    with pytest.raises(ValueError):
        LAT.settings_of({'lateral_filtering': {'enabled': True, 'controller': 'sumo'}})
    from falcon_autonomy.core.project import Project, validate
    from falcon_autonomy.core.traffic import INDIA_MIXED
    # the Indian profile filters its two-wheelers (not its autorickshaw stand-ins) with the TM's lane offset
    assert LAT.settings_of(INDIA_MIXED) == {'enabled': True, 'vehicles': ['motorcycle', 'bicycle'], 'controller': 'tm'}
    p = Project.new('T')
    p.doc['traffic_profiles'].append(dict(INDIA_MIXED))
    assert validate(p.doc) == []
    p.doc['traffic_profiles'][-1]['lateral_filtering'] = {'enabled': True, 'vehicles': ['car']}
    assert validate(p.doc)


# ------------------------------------------------------------------------------------------------- the fake carla
class _W:
    def __init__(self, carla, client, world, tm, seed=3):
        self.carla, self.client, self.world, self.tm = carla, client, world, tm
        self.spawned, self.notes = [], []
        self.speed_control = self.class_limits = self.traffic_keeper = None
        self.cfg = {'seed': seed}

    def own(self, a):
        self.spawned.append(a)
        return a

    def note(self, t):
        self.notes.append(t)


def _scene(carla, controller, filtering=True, left=False):
    """A stopped car on lane −1's centre at x = 0, a motorcycle 30 m behind it on the autopilot at 30 km/h; 12 s."""
    client = carla.Client('127.0.0.1', 2000)
    world = client.get_world()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    tm = client.get_trafficmanager(8000)
    w = _W(carla, client, world, tm)
    lib = world.get_blueprint_library()
    car = world.try_spawn_actor(lib.find('vehicle.tesla.model3'), carla.Transform(carla.Location(0.0, 0.0, 0.3)))
    car.apply_control(carla.VehicleControl(hand_brake=True))
    m = world.try_spawn_actor(lib.find('vehicle.yamaha.yzf'), carla.Transform(carla.Location(-30.0, 0.0, 0.3)))
    m.set_autopilot(True, 8000)
    tm.set_desired_speed(m, 30.0)
    lf = None
    if filtering:
        lf = LAT.LateralFilter(w, world, {'controller': controller, 'vehicles': ['motorcycle']}, 3, left, {})
        assert lf.add(m, 'motorcycle') and not lf.add(car, 'car')
    track = []
    for i in range(240):
        world.tick()
        if lf is not None:
            lf.step((i + 1) * 0.05)
        L = m.get_transform().location
        track.append((round(L.x, 6), round(L.y, 6)))
    return w, lf, m, car, track


def test_without_filtering_the_two_wheeler_queues_behind_the_car(fake_module):
    _, _, m, _, track = _scene(fake_module, 'tm', filtering=False)
    assert track[-1][0] < -5.0 and all(abs(y) < 1e-6 for _, y in track)


@pytest.mark.parametrize('controller', ['tm', 'own'])
def test_with_filtering_it_passes_the_car_beside_it(fake_module, controller):
    carla = fake_module
    w, lf, m, car, track = _scene(carla, controller)
    assert track[-1][0] > 30.0                                         # past the car, going on
    beside = [y for x, y in track if -3.0 < x < 3.0]
    assert beside and min(abs(y) for y in beside) >= 0.93 + 0.45      # never through it
    rep = lf.report()
    assert rep['vehicles'] == 1 and rep['controller'] == controller and rep['updates'] > 100
    assert rep['parameters']['strip_m'] == LAT.STRIP_M and 'heuristic' in rep['parameters']['source']
    row = lf.log_rows()[m.id]
    assert set(row) >= {'offset_m', 'target_m', 'lane', 'mode', 'gap_m', 'cap_kmh'}
    if controller == 'tm':
        log = [e for e in carla.LOG if e[1:2] == (m.id,) and e[0] in ('tm_lane_offset', 'tm_ignore_vehicles')]
        assert any(e[0] == 'tm_lane_offset' and abs(e[2]) > 1.0 for e in log)
        ign = [e[2] for e in log if e[0] == 'tm_ignore_vehicles']
        assert ign[:1] == [100.0] and rep['ignore_on'] >= 1         # only while the car on its centre is in the way
        assert m.autopilot
    else:
        assert not m.autopilot and m.get_control().throttle >= 0.0


def test_left_hand_traffic_keeps_to_the_left(fake_module):
    """With nothing in the way the home offset is on the near side: the driver's left in left-hand traffic."""
    carla = fake_module
    for left, sign in ((True, -1.0), (False, 1.0)):
        carla.reset()
        w, lf, m, car, track = _scene(carla, 'own', left=left)
        car.destroy()
        for i in range(120):
            w.world.tick()
            lf.step(12.05 + i * 0.05)
        row = lf.log_rows()[m.id]
        assert row['offset_m'] * sign > 0.3, (left, row)       # a strip or more to the near side


def test_same_seed_same_motion(fake_module):
    carla = fake_module
    a = _scene(carla, 'own')[4]
    carla.reset()
    b = _scene(carla, 'own')[4]
    assert a == b


def test_a_replacement_and_a_moved_vehicle_filter_from_their_lane_centre(fake_module):
    carla = fake_module
    w, lf, m, car, _ = _scene(carla, 'tm')
    old = m.id
    bp = w.world.get_blueprint_library().find('vehicle.yamaha.yzf')
    new = w.world.try_spawn_actor(bp, carla.Transform(carla.Location(60.0, 3.5, 0.3)))
    assert lf.adopt(old, new) and lf.holds(new.id) and not lf.holds(old)
    assert lf.report()['adopted'] == 1 and lf.report()['vehicles'] == 1
    lf.rows[new.id]['cmd'] = 1.5
    lf.reset(new)
    assert lf.rows[new.id]['cmd'] == 0.0 and lf.rows[new.id]['path'] is None
    assert ('tm_lane_offset', new.id, 0.0) in carla.LOG


# ------------------------------------------------------------------------------------------------- a drive
def test_a_drive_with_lateral_filtering_records_it(project, tmp_path):
    from test_traffic_keeper import _drive
    project.doc['traffic_profiles'][0].update(
        fleet={'motorcycle': 0.5, 'car': 0.5}, lateral_filtering={'enabled': True, 'vehicles': ['motorcycle']})
    cfg, run, frames, acts = _drive(project, tmp_path, 'lat', seconds=6.0)
    assert cfg['traffic']['lateral_filtering']['enabled'] is True
    lf = run['traffic']['lateral_filtering']
    placed = run['traffic']['placed']
    assert lf['vehicles'] >= placed.get('motorcycle', 0) >= 1 and lf['controller'] == 'tm'
    assert lf['parameters']['update_s'] == LAT.UPDATE_S and lf['step_ms']['calls'] > 0
    assert all('lateral' in f and f['lateral']['vehicles'] >= 1 for f in frames)
    held = [a for row in acts for a in row['actors'] if 'lateral' in a]
    assert held and all(a['category'] == 'vehicle.motorcycle' for a in held)
    assert {'offset_m', 'target_m', 'mode', 'ignoring'} <= set(held[-1]['lateral'])
    cars = [a for row in acts for a in row['actors'] if a['category'] == 'vehicle.car' and 'lateral' in a]
    assert not cars


def test_a_drive_without_it_is_unchanged(project, tmp_path):
    from test_traffic_keeper import _drive
    project.doc['traffic_profiles'][0].update(fleet={'motorcycle': 0.5, 'car': 0.5})
    _, run, frames, acts = _drive(project, tmp_path, 'nolat', seconds=3.0)
    assert 'lateral_filtering' not in run['traffic'] and not any('lateral' in f for f in frames)
    assert not any('lateral' in a for row in acts for a in row['actors'])


def test_the_same_seed_places_the_same_traffic_with_it_on_or_off(project, tmp_path):
    from test_traffic_keeper import _drive
    project.doc['traffic_profiles'][0].update(fleet={'motorcycle': 0.5, 'car': 0.5})
    _, _, _, off = _drive(project, tmp_path, 'off', seconds=1.0)
    project.doc['traffic_profiles'][0]['lateral_filtering'] = {'enabled': True}
    _, _, _, on = _drive(project, tmp_path, 'on', seconds=1.0)
    first = [sorted((a['type_id'], round(a['x'], 1), round(a['y'], 1)) for a in r[0]['actors']) for r in (off, on)]
    assert first[0] and first[0] == first[1]
