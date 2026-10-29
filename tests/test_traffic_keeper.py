"""Traffic kept in an imported world (workers.trafficguard.TrafficKeeper), sidewalks in the ground relabelling
(workers.camgt.LaneRaster), vehicle lights and the world's settings in every worker's config — against the fake carla
(its one straight road: x in −150…150 m, lanes ±1 ±2, sidewalks ±3; a road that ends at both ends, like the edge of
an imported area)."""
import json
import math

import pytest

from conftest import worker_env  # noqa: F401  (the fake carla's environment for the worker subprocess)
from falcon_autonomy.core import protocol as P

XODR = ('<OpenDRIVE><header/><road id="1" length="300" junction="-1"><planView/><lanes><laneSection s="0">'
        '<left><lane id="1" type="driving"/></left><center><lane id="0" type="none"/></center>'
        '<right><lane id="-1" type="driving"/></right></laneSection></lanes></road></OpenDRIVE>')


@pytest.fixture
def fake(fake_module):
    return fake_module


class _W:
    """The worker's side the keeper uses: carla, client, world, tm, own, the speed / class controls."""

    def __init__(self, carla, client, world, tm):
        self.carla, self.client, self.world, self.tm = carla, client, world, tm
        self.spawned = []
        self.speed_control = self.class_limits = None

    def own(self, a):
        self.spawned.append(a)
        return a


def _setup(carla, seed=0):
    from falcon_autonomy.workers.trafficguard import TrafficKeeper
    client = carla.Client('127.0.0.1', 2000)
    world = client.generate_opendrive_world(XODR)
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    w = _W(carla, client, world, tm)
    return w, TrafficKeeper(w, world, seed)


def _car(carla, w, x, y=0.0, yaw=0.0, role='falcon_traffic', kmh=36.0):
    bp = w.world.get_blueprint_library().find('vehicle.tesla.model3')
    bp.set_attribute('role_name', role)
    a = w.world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, 0.3), carla.Rotation(0.0, yaw, 0.0)))
    w.own(a)
    a.set_autopilot(True, w.tm.get_port())
    w.tm.set_desired_speed(a, kmh)
    return a


def _run(w, keeper, seconds, every=None):
    """Step the world and the keeper for seconds of simulated time (going on from the last call)."""
    out = []
    for _ in range(int(round(seconds / 0.05))):
        w.world.tick()
        w.t = getattr(w, 't', 0.0) + 0.05
        out += keeper.step(w.t)
        if every:
            every()
    return out


def test_a_vehicle_heading_for_the_road_end_out_of_view_is_moved_before_it(fake):
    w, keeper = _setup(fake)
    _car(fake, w, -120.0, role='hero', kmh=0.0)                 # the ego, far behind and still
    a = _car(fake, w, 90.0)                                      # 10 m/s towards the end at x = 150
    keeper.add(a, {'distance_to_leading_m': 2.0}, -5.0)
    xs = []
    evs = _run(w, keeper, 8.0, lambda: xs.append(a.get_transform().location.x))
    assert max(xs) < 150.0                                       # never past the end
    ev = next(e for e in evs if e['kind'] == 'dead_end')
    reach = min(max(keeper.LOOK_S * 10.0, keeper.LOOK_MIN_M), keeper.LOOK_MAX_M)
    assert 150.0 - reach - 2 * keeper.STEP_M < ev['from']['x'] < 150.0 - reach + 2 * keeper.STEP_M   # as it comes in reach
    to = (ev['to']['x'], ev['to']['y'])
    assert math.hypot(to[0] - ev['from']['x'], to[1] - ev['from']['y']) > keeper.AWAY_M
    assert any(abs(sp.location.x - to[0]) < 0.05 and abs(sp.location.y - to[1]) < 0.05 for sp in keeper.spawns)
    assert a.autopilot and a.is_alive                            # driving again from the spawn point
    log = [e for e in fake.LOG if e[0] == 'autopilot' and e[1] == a.id]
    assert log[-2:] == [('autopilot', a.id, False, 8000), ('autopilot', a.id, True, 8000)]
    rep = keeper.report()
    assert rep['dead_end'] >= 1 and rep['alive_at_end'] == rep['vehicles'] == 1 and rep['events'][0] == ev


def test_a_vehicle_the_ego_can_see_stops_and_moves_once_out_of_view(fake):
    w, keeper = _setup(fake)
    ego = _car(fake, w, 95.0, 3.5, role='hero', kmh=0.0)       # right behind it, looking its way
    a = _car(fake, w, 115.0)
    keeper.add(a)
    evs = _run(w, keeper, 6.0)
    assert evs == [] and not a.autopilot                         # held: off the autopilot, braking
    x = a.get_transform().location.x
    assert x < 150.0 and a.get_velocity().length() < 0.05      # stopped short of the end
    assert keeper.report()['max_held'] == 1
    ego.set_transform(fake.Transform(fake.Location(-140.0, 3.5, 0.3), fake.Rotation(0.0, 180.0, 0.0)))  # looks away
    evs = _run(w, keeper, 0.5)
    assert len(evs) == 1 and evs[0]['kind'] == 'dead_end' and evs[0]['held_s'] > 0
    assert a.autopilot and keeper.report()['dead_end_after_hold'] == 1


def test_a_removed_vehicle_is_replaced_and_a_lost_one_brought_back(fake):
    w, keeper = _setup(fake)
    _car(fake, w, -140.0, role='hero', kmh=0.0)
    a = _car(fake, w, 0.0)
    b = _car(fake, w, -30.0, 30.0)                               # beside the road network: lost
    keeper.add(a, {}, 4.0)
    keeper.add(b)
    a._batch = True                                              # the server removed it (a dead end): as a batch
    a.destroy()                                                  # DestroyActor by id, which the fake allows
    evs = _run(w, keeper, 3.0)
    rep = keeper.report()
    kinds = sorted(e['kind'] for e in evs)
    assert kinds == ['lost', 'replaced'] and rep['replaced'] == 1 and rep['lost'] == 1
    new = next(r['a'] for r in keeper.rows.values() if r['a'] is not b)
    assert new.type_id == a.type_id and new.is_alive and new.autopilot and new.attributes['role_name'] == \
        'falcon_traffic'
    assert rep['vehicles'] == 2 and rep['alive_at_end'] == 2 and new in w.spawned
    assert abs(b.get_transform().location.y) < 8.0              # back on the carriageway


def test_moves_repeat_under_the_seed(fake):
    def once(seed):
        fake.reset()
        w, keeper = _setup(fake, seed)
        _car(fake, w, -140.0, role='hero', kmh=0.0)
        for x in (60.0, 90.0, 120.0):
            keeper.add(_car(fake, w, x))
        return [(e['kind'], e['to']['x'], e['to']['y']) for e in _run(w, keeper, 12.0)]
    a, b = once(3), once(3)
    assert a and a == b


# ------------------------------------------------------------------------------------------------- sidewalks
def test_the_lane_raster_names_sidewalks(fake):
    from falcon_autonomy.workers.camgt import LaneRaster
    m = fake.Map('x')
    assert all(wp.lane_type == fake.LaneType.Driving for wp in m.generate_waypoints(5.0))     # as CARLA's
    r = LaneRaster(fake, m)
    got = r.lookup([(0.0, 0.0), (0.0, -7.0), (0.0, 7.0), (0.0, -10.5), (0.0, 20.0)])
    assert list(got) == [1, 1, 2, 2, 25]                        # road, road, sidewalk both sides, beyond: ground


# ------------------------------------------------------------------------------------------------- the workers
def _drive(project, tmp_path, name, seconds=16.0, **world_kw):
    import subprocess
    import sys
    from falcon_autonomy.core.drive import drive_config
    ws = tmp_path / 'test.falcon'
    (ws / 'w').mkdir(parents=True, exist_ok=True)
    (ws / 'w' / 'world.xodr').write_text(XODR)
    if project.item('worlds', 'w-x') is None:
        project.doc['worlds'].append(dict({'id': 'w-x', 'name': 'X', 'kind': 'osm', 'xodr': 'w/world.xodr',
                                           'osm': {'handedness': 'RHT'}, 'weather': {'preset': 'ClearNoon'}},
                                          **world_kw))
    cfg = drive_config(project, 'w-x', 'rig-default', str(tmp_path / name), 2310, 8310, seconds=seconds, seed=2,
                       workspace=str(ws))
    cfg['traffic'] = dict(cfg['traffic'], vehicles=6, walkers=0)
    cfg['rig']['sensors'] = [dict(s, width=64, height=48, labels=[]) for s in cfg['rig']['sensors']
                             if s['type'] == 'camera.pinhole'][:1]
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=300)
    assert r.returncode == P.OK, r.stderr[-3000:]
    run = json.loads((tmp_path / name / 'run.json').read_text())
    frames = [json.loads(x) for x in (tmp_path / name / 'frames.jsonl').read_text().splitlines()]
    acts = [json.loads(x) for x in (tmp_path / name / 'actors.jsonl').read_text().splitlines()]
    return cfg, run, frames, acts


def _traffic_x(acts):
    return [a['x'] for row in acts for a in row['actors'] if a['role'] == 'falcon_traffic']


def test_a_drive_keeps_its_traffic_on_the_road(project, tmp_path):
    cfg, run, frames, acts = _drive(project, tmp_path, 'kept')
    assert cfg['keep_traffic'] is True
    k = run['traffic']['keeper']
    placed = sum(run['traffic']['placed'].values())
    assert k['vehicles'] == placed and k['alive_at_end'] == placed
    assert k['dead_end'] + k['dead_end_after_hold'] >= 1
    moved = [e for f in frames for e in f.get('traffic_respawns', [])]
    assert len(moved) == k['dead_end'] + k['dead_end_after_hold'] + k['fell'] + k['lost'] + k['replaced']
    xs = _traffic_x(acts)
    assert xs and max(xs) <= 150.0 and min(xs) >= -150.0          # nobody past either end
    assert any('put back on the road' in n for n in run.get('notes', []))


def test_without_the_keeper_traffic_drives_off_the_end(project, tmp_path):
    cfg, run, frames, acts = _drive(project, tmp_path, 'off', keep_traffic=False)
    assert cfg['keep_traffic'] is False and 'keeper' not in run['traffic']
    assert max(_traffic_x(acts)) > 150.0 and not any('traffic_respawns' in f for f in frames)


def test_gym_and_hil_configs_load_the_world_as_a_drive_does(project, tmp_path):
    from falcon_autonomy.core.drive import drive_config
    from falcon_autonomy.core.gymenv import gym_config
    from falcon_autonomy.core.hil import hil_config
    ws = tmp_path / 'test.falcon'
    (ws / 'w' / 'city').mkdir(parents=True)
    (ws / 'w' / 'world.xodr').write_text(XODR)
    (ws / 'w' / 'city' / 'plan.json').write_text('{"items": []}')
    project.doc['worlds'].append({'id': 'w-x', 'name': 'X', 'kind': 'osm', 'xodr': 'w/world.xodr',
                                  'osm': {'handedness': 'RHT'}, 'weather': {'preset': 'ClearNoon'},
                                  'speed_limits': 'carla', 'keep_traffic': False,
                                  'city': {'enabled': True, 'plan': 'city/plan.json', 'plan_sha256': 'ab'}})
    d = drive_config(project, 'w-x', 'rig-default', str(tmp_path / 'd'), 2310, 8310, workspace=str(ws))
    g = gym_config(project, 'w-x', str(tmp_path / 'g'), 2310, 8310, workspace=str(ws))
    h = hil_config(project, 'w-x', 'rig-default', str(tmp_path / 'h'), 2310, 8310, workspace=str(ws))
    for c in (g, h):
        assert c['city'] == d['city'] and c['city']['plan'].endswith('plan.json')
        assert c['speed_limits'] == 'carla' and c['keep_traffic'] is False
    town = gym_config(project, 'world-town10', str(tmp_path / 't'), 2310, 8310)
    assert not {'city', 'speed_limits', 'keep_traffic'} & set(town)       # a town's config (and key) unchanged


def test_vehicle_lights_follow_the_weather(fake):
    from falcon_autonomy.workers import scene
    w, _ = _setup(fake)
    w.note = lambda *_: None
    cars = [_car(fake, w, x) for x in (-50.0, 0.0)]
    night = {'sun_altitude_angle': -20.0}
    got = scene.light_vehicles(w, {'weather': night})
    assert got['resolved'] == 'on' and got['vehicles'] == 2 and all(int(c.get_light_state()) != 0 for c in cars)
    extra = _car(fake, w, 50.0)                                    # one more (Live's cutting-in car)
    scene.light_vehicles(w, {'weather': night}, actors=[extra])
    assert int(extra.get_light_state()) != 0 and w.lights_applied['vehicles'] == 2
    day = scene.light_vehicles(w, {'weather': night}, {'sun_altitude_angle': 40.0})
    assert day['resolved'] == 'off' and all(int(c.get_light_state()) == 0 for c in cars + [extra])


def test_a_vehicle_falling_through_a_hole_in_the_road_is_caught(fake):
    """A gap in the generated mesh (a junction's connecting road not meeting its lanes): the map has a lane there,
    the vehicle sinks below it."""
    w, keeper = _setup(fake)
    _car(fake, w, -140.0, role='hero', kmh=0.0)
    a = _car(fake, w, 0.0)
    keeper.add(a)
    _run(w, keeper, 0.5)
    a.set_transform(fake.Transform(fake.Location(5.0, 0.0, -1.6), fake.Rotation()))
    evs = _run(w, keeper, 0.5)
    assert [e['kind'] for e in evs] == ['fell'] and evs[0]['on_lane'] is True
    assert abs(a.get_transform().location.z) < 1.0


from test_live import _frames, _wait, gui_socket  # noqa: E402,F401


def test_live_switches_vehicle_lights_with_its_weather(project, tmp_path, gui_socket):
    import subprocess
    import sys
    from falcon_autonomy.core.live import live_config
    path, got = gui_socket
    cfg = live_config(project, 'world-town10', 'rig-default', str(tmp_path / 'live'), 2310, 8310, path,
                      str(tmp_path / 'recs'), fps=20.0, mode='autopilot',
                      view={'kind': 'chase', 'width': 96, 'height': 64, 'fov': 90.0})
    cfg['weather'] = dict(cfg['weather'], sun_altitude_angle=-20.0)          # night
    cp = tmp_path / 'live.json'
    cp.write_text(json.dumps(cfg))
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'live', str(cp)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())

    def send(**c):
        p.stdin.write(json.dumps(c) + '\n')
        p.stdin.flush()
    try:
        assert _wait(lambda: len(_frames(got)) > 5), p.stderr.read() if p.poll() is not None else 'no frames'
        send(cmd='event', kind='weather', fields={'sun_altitude_angle': 40.0})     # day breaks
        n = len(_frames(got))
        assert _wait(lambda: len(_frames(got)) > n + 10, 20)
        send(cmd='stop')
        p.wait(60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == P.OK, p.stderr.read()[-3000:]
    run = json.loads((tmp_path / 'live' / 'run.json').read_text())
    L = run['lights']
    assert L['resolved'] == 'on' and L['vehicles'] >= 1 and L['vehicles_lit'] == L['vehicles']
    w = [e for e in run['events'] if e['kind'] == 'weather']
    assert w and w[0]['detail']['vehicle_lights'] == 'off'
