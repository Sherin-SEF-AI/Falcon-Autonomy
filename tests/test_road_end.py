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


class _W:
    def __init__(self):
        self.carla = carla


class _Ego:
    def __init__(self, x, y, z, vz=0.0):
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
