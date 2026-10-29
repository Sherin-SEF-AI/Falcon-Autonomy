"""The road-end guard (workers.scene.RoadEndGuard): an ego that leaves the road network where a road ends — at the
edge of an imported area — and falls is put back on the road; one merely beside the road, or on it, is left alone.
Against the fake carla's map (one straight road, x in −150…150 m)."""
import math
import types

import pytest


@pytest.fixture(autouse=True)
def _carla(fake_module):
    global carla
    carla = fake_module
    yield


class _Client:
    """Applies a synchronous ApplyTransform batch to the test's egos (by id)."""
    egos: dict = {}

    def apply_batch_sync(self, commands, do_tick=False):
        out = []
        for c in commands:
            self.egos[c.actor_id].set_transform(c.transform)
            out.append(types.SimpleNamespace(error=''))
        return out


class _W:
    def __init__(self):
        self.carla = carla
        self.client = _Client()


class _Ego:
    def __init__(self, x, y, z, vz=0.0):
        self.id = 7000 + len(_Client.egos)
        _Client.egos[self.id] = self
        self.t = carla.Transform(carla.Location(x, y, z), carla.Rotation())
        self.v = carla.Vector3D(0.0, 0.0, vz)
        self.moved_to = None

    def get_transform(self):
        return self.t

    def get_velocity(self):
        return self.v

    def set_target_velocity(self, v):
        self.v = v

    def set_target_angular_velocity(self, v):
        pass

    def set_transform(self, tr):
        self.moved_to = tr
        self.t = tr


def _guard(ego, n=10):
    from falcon_autonomy.workers.scene import RoadEndGuard
    spawns = [carla.Transform(carla.Location(-140.0 + 30.0 * k, -1.75, 0.5), carla.Rotation()) for k in range(n)]
    world = types.SimpleNamespace(get_map=lambda: carla.Map('Town10HD_Opt'))
    return RoadEndGuard(_W(), world, ego, spawns), spawns


def test_a_falling_ego_past_the_road_end_is_put_back():
    ego = _Ego(140.0, -1.75, 0.0)
    g, spawns = _guard(ego)
    assert g.step(1.0) is None and g.last_road_z == 0.0           # on the road: its height remembered
    ego.t = carla.Transform(carla.Location(160.0, -1.75, -0.4), carla.Rotation())
    assert g.step(1.5) is None                                       # past the end but not sinking yet
    ego.t = carla.Transform(carla.Location(162.0, -1.75, -1.6), carla.Rotation())
    ev = g.step(2.0)
    assert ev is not None and ev['kind'] == 'road_end' and ev['from']['x'] == 162.0
    assert ego.moved_to is not None
    to = ego.moved_to.location
    assert math.hypot(to.x - 162.0, to.y + 1.75) > g.AWAY_M        # far from the dead end
    assert to.x == min((s.location for s in spawns if math.hypot(s.location.x - 162.0, s.location.y + 1.75) > g.AWAY_M),
                       key=lambda l: abs(l.x - sum(s.location.x for s in spawns) / len(spawns))).x
    assert (ego.v.x, ego.v.y, ego.v.z) == (0.0, 0.0, 0.0) and g.events == [ev]


def test_falling_fast_triggers_even_without_a_known_road_height():
    ego = _Ego(170.0, -1.75, -0.5, vz=-5.0)
    g, _ = _guard(ego)
    assert g.step(0.1) is not None


def test_beside_the_road_and_not_sinking_is_left_alone():
    ego = _Ego(0.0, -1.75, 0.0)
    g, _ = _guard(ego)
    g.step(0.0)
    ego.t = carla.Transform(carla.Location(0.0, 12.0, 0.1), carla.Rotation())   # on the sidewalk, level
    assert g.step(0.5) is None and ego.moved_to is None


def test_a_destroyed_ego_is_reported_with_where_it_was_last_seen():
    ego = _Ego(0.0, -1.75, 0.0)
    ego.is_alive = True
    g, _ = _guard(ego)
    assert g.lost(0.0) is None
    g.step(0.5)
    ego.is_alive = False                         # the traffic manager removed it at a dead end of its graph
    why = g.lost(0.55)
    assert why.startswith('the ego was destroyed by the simulator at 0.6 s') and 'driving lane at (0, -2)' in why
    assert 'traffic manager' in why
    ego.is_alive = True
    ego.t = carla.Transform(carla.Location(170.0, -1.75, -3.0), carla.Rotation())   # past the end, falling
    g.step(1.0)
    ego.is_alive = False
    assert 'kill depth' in g.lost(1.05)


class _Look:
    """A road look-ahead (TrafficKeeper.dead_end_ahead) that reports a dead end ahead when told to."""
    SPAWN_CLEAR_M = 40.0

    def __init__(self, ahead=None, blocked_x=()):
        self.ahead, self.blocked_x, self.asked = ahead, set(blocked_x), 0

    def dead_end_ahead(self, a, wp, loc, speed):
        self.asked += 1
        return self.ahead

    def _walk(self, wp, d0, reach):                   # a dead end ahead of the spawn points at these x
        return 5.0 if round(wp.transform.location.x) in self.blocked_x else None


class _Tm:
    def __init__(self):
        self.on = []

    def get_port(self):
        return 8350


def test_a_dead_end_ahead_on_the_planned_path_puts_the_ego_back_before_it_is_removed():
    ego = _Ego(100.0, -1.75, 0.0)
    ego.autopilot = []
    ego.set_autopilot = lambda on, port: ego.autopilot.append((on, port))
    w = _W()
    w.tm = _Tm()
    w.road_look = _Look(ahead=None)
    from falcon_autonomy.workers.scene import RoadEndGuard
    spawns = [carla.Transform(carla.Location(-140.0 + 30.0 * k, -1.75, 0.5), carla.Rotation()) for k in range(10)]
    world = types.SimpleNamespace(get_map=lambda: carla.Map('Town10HD_Opt'))
    g = RoadEndGuard(w, world, ego, spawns)
    assert g.step(0.0) is None and ego.moved_to is None             # the road goes on
    w.road_look.ahead = 18.0
    assert g.step(0.1) is None and w.road_look.asked == 1           # looked at most every LOOK_EVERY_S
    w.road_look.blocked_x = {-20, 10}                                # the nearest spawn points face dead ends too
    ev = g.step(0.3)
    assert ev['cause'] == 'dead_end_ahead' and '18 m ahead' in ev['detail']
    to = ego.moved_to.location
    assert math.hypot(to.x - 100.0, to.y + 1.75) > g.AWAY_M and round(to.x) not in (-20, 10)
    assert ego.autopilot == [(True, 8350)] and g.events == [ev]


def test_a_town_has_no_look_ahead_and_is_only_guarded_against_falling():
    ego = _Ego(100.0, -1.75, 0.0)
    g, _ = _guard(ego)
    assert g.look is None and g.step(0.0) is None and g.step(1.0) is None
