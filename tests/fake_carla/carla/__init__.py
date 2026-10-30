"""A fake `carla` module (CARLA 0.9.16's Python API, the part the Vantage bridge uses) for tests without a simulator.

Put tests/fake_carla on sys.path (before any real carla) and `import carla` gets this. It records what the client
spawns and produces synthetic sensor data from a flat world (the ground at z = 0) with box-shaped vehicles, following
CARLA's documented and measured behaviour:

* Frames: Unreal's (x forward, y right, z up, left-handed); Rotation(pitch, yaw, roll) degrees with Transform::GetMatrix.
* World.tick() advances fixed_delta_seconds in synchronous mode and returns the frame; vehicles on autopilot drive at a
  constant speed and yaw rate (the ego turns gently so its rates are not zero).
* Sensors run on the first tick after they are spawned and listened to, then by sensor_tick with the remainder carried
  over (0 = every tick). Data go to the listener in the tick.
* sensor.lidar.ray_cast: channels evenly spaced from upper_fov (channel 0) to lower_fov; per tick
  round(pps·dt/channels) points per channel at azimuth fmod(angle + step·i, hfov) − hfov/2, the angle advancing by
  rotation_frequency·hfov·dt; hits on the ground and the boxes (the own vehicle's too) within range, XYZI float32 in the
  sensor's axes, channel by channel; intensity exp(−a·r); noise along the ray.
* sensor.other.radar: pps·dt random rays in the elliptical cone (radius uniform, as CARLA); depth, azimuth, altitude
  (azimuth positive right), velocity of the hit relative to the radar along the ray (the radar's velocity from its
  previous position, (0, 0, 0) before the first tick, as CARLA).
* sensor.camera.rgb: a pinhole (f = W / (2 tan(fov/2)), centre at the image centre) rendering sky, a ground checker
  and the boxes; BGRA. Weather: fog of fog_density ≥ FOG_RENDER_MIN blends each pixel towards a grey airlight by
  the transmission exp(−3.0 · range / V), V = 7000 m / fog_density (CARLA's measured visibility relation, a 5 %
  threshold; the sky is airlight); the presets' haze (fog_density 2–7) is not rendered. A sun below the horizon
  darkens the image to NIGHT_LIGHT of its daylight brightness. (Lidar and radar ignore weather, as CARLA's.)
* sensor.other.imu: accelerometer from three positions plus gravity in the sensor's axes, gyroscope the vehicle's rate
  (degrees→radians, Unreal's axes), compass. sensor.other.gnss: lat/lon from a flat-earth geo-reference at (0, 0).
* Ground truth, as measured on a 0.9.16 server: sensor.camera.semantic_segmentation (R = the tag),
  instance_segmentation (R = the tag, G + 256·B = the actor id; the map's parked cars, level meshes, an id of their
  own), depth (planar depth along the view's axis, 1000·(R + 256G + 65536B)/(2²⁴−1)), all from the same ray cast as the
  RGB camera at the same pose; sensor.lidar.ray_cast_semantic: the ray-cast lidar's rays without noise, x y z cos_inc
  (float32) object_idx object_tag (uint32), the actor id for actors and 0 for the ground and level meshes. Tags: the
  ground is road (1) on the carriageway and sidewalk (2) beyond, a vehicle by its base type (car 14, truck 15, the
  upper half of a bicycle its rider 13, the rest bicycle 19), walkers 12, props by kind.
* The map: a straight road along x, right-hand traffic like CARLA's towns (lanes −1, −2 heading +x at y = 0 and 3.5,
  lanes 1, 2 heading −x at y = −3.5 and −7, CARLA's left-handed axes: +y is the right of +x), Waypoint next/previous/
  get_left_lane/get_right_lane with CARLA's lane-id rules (sidewalk lanes −3 and 3 beyond the kerbs, reached by
  get_right_lane, as an OpenDRIVE road's outermost lanes; generate_waypoints gives driving lanes only, as CARLA's),
  three parked cars of the level (World.get_level_bbs(Car)).
* The traffic manager: an autopilot vehicle drives along its heading at its desired speed (set_desired_speed, else a
  random one; the hero turns gently unless given one), brakes for any actor ahead in its lane, and changes lanes when
  forced (force_lane_change: True to the right). Walkers move by WalkerControl; a vehicle off the autopilot brakes by
  its control (VehicleControl.brake, hand_brake) or keeps a constant velocity (enable_constant_velocity, its own axes).
  An autopilot vehicle's get_control() is the control that produces its motion under the fake's vehicle model.
* The traffic light (actor 50) controls the +x lanes with a stop line at x = STOP_LINE_X (get_stop_waypoints); its
  state is Green unless FAKE_CARLA_LIGHT_STATE names another (tests) or set_state() changed it. Map.get_waypoint with
  project_to_road=False returns None off the carriageway.
"""
from __future__ import annotations

import fnmatch
import math
import os

import numpy as np

__version__ = '0.9.16'
_GRAVITY = 9.81
EARTH_R = 6378137.0
LOG = []                       # (what, details) of every client call that changes the world; tests read it
STATE = {'world': None, 'server_version': '0.9.16'}


def reset():
    """A fresh simulator (tests call it before each run)."""
    LOG.clear()
    STATE['world'] = None


# ---------------------------------------------------------------------------------------------------- geometry
class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return type(self)(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return type(self)(self.x - o.x, self.y - o.y, self.z - o.z)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __repr__(self):
        return f'{type(self).__name__}(x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})'


class Vector2D:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)


class Location(Vector3D):
    def distance(self, o):
        return (self - o).length()


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)

    def matrix(self):
        p, y, r = math.radians(self.pitch), math.radians(self.yaw), math.radians(self.roll)
        cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
        return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                         [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                         [sp, -cp * sr, cp * cr]])

    def get_forward_vector(self):
        return Vector3D(*self.matrix()[:, 0])

    def __repr__(self):
        return f'Rotation(pitch={self.pitch:.6f}, yaw={self.yaw:.6f}, roll={self.roll:.6f})'


def _euler(M):
    sp = max(-1.0, min(1.0, float(M[2, 0])))
    pitch = math.degrees(math.asin(sp))
    yaw = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    roll = math.degrees(math.atan2(-M[2, 1], M[2, 2]))
    return pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location if location is not None else Location()
        self.rotation = rotation if rotation is not None else Rotation()

    def get_matrix(self):
        T = np.eye(4)
        T[:3, :3] = self.rotation.matrix()
        T[:3, 3] = (self.location.x, self.location.y, self.location.z)
        return T.tolist()

    def get_inverse_matrix(self):
        return np.linalg.inv(np.array(self.get_matrix())).tolist()

    def transform(self, loc):
        p = np.array(self.get_matrix()) @ np.array([loc.x, loc.y, loc.z, 1.0])
        return Location(*p[:3])

    def get_forward_vector(self):
        return self.rotation.get_forward_vector()

    def _m(self):
        return np.array(self.get_matrix())

    def __repr__(self):
        return f'Transform({self.location}, {self.rotation})'


def _T(m):
    """A Transform of a 4x4 (Unreal's axes)."""
    p, y, r = _euler(m[:3, :3])
    return Transform(Location(*m[:3, 3]), Rotation(p, y, r))


class BoundingBox:
    def __init__(self, location=None, extent=None):
        self.location = location if location is not None else Location()
        self.extent = extent if extent is not None else Vector3D(1, 1, 1)
        self.rotation = Rotation()


class Timestamp:
    def __init__(self, frame, elapsed):
        self.frame = frame
        self.elapsed_seconds = elapsed
        self.delta_seconds = 0.0
        self.platform_timestamp = 0.0


class VehicleLightState:
    NONE = 0
    Position = 1
    LowBeam = 2
    HighBeam = 4


class CityObjectLabel:
    NONE, Roads, Sidewalks, Buildings, Walls, Fences, Poles, TrafficLight, TrafficSigns, Vegetation = range(10)
    Terrain, Sky, Pedestrians, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle = range(10, 20)
    Static, Dynamic, Other, Water, RoadLines, Ground, Bridge, RailTrack, GuardRail = range(20, 29)
    Any = 255


class _Enum(int):
    """A Boost.Python-like enum value: an int with a name."""

    def __new__(cls, value, name):
        v = int.__new__(cls, value)
        v.name = name
        return v

    def __str__(self):
        return self.name


def _enum(cls, names):
    for i, n in enumerate(names):
        setattr(cls, n, _Enum(i, n))
    cls.names = {n: getattr(cls, n) for n in names}
    return cls


class VehicleDoor:
    pass


class LaneMarkingType:
    pass


class LaneMarkingColor:
    pass


_enum(VehicleDoor, ('FL', 'FR', 'RL', 'RR', 'All'))
_enum(LaneMarkingType, ('NONE', 'Other', 'Broken', 'Solid', 'SolidSolid', 'SolidBroken', 'BrokenSolid',
                        'BrokenBroken', 'BottsDots', 'Grass', 'Curb'))
_enum(LaneMarkingColor, ('Standard', 'Blue', 'Green', 'Red', 'White', 'Yellow', 'Other'))


class LaneMarking:
    def __init__(self, type_, color, width=0.15):
        self.type, self.color, self.width = type_, color, width
        self.lane_change = LaneChange.NONE if hasattr(LaneChange, 'NONE') else 0


class LaneType:
    NONE, Driving, Stop, Shoulder, Biking, Sidewalk = 'NONE', 'Driving', 'Stop', 'Shoulder', 'Biking', 'Sidewalk'


class LaneChange:
    NONE, Right, Left, Both = 'NONE', 'Right', 'Left', 'Both'


class WalkerControl:
    def __init__(self, direction=None, speed=0.0, jump=False):
        self.direction = direction if direction is not None else Vector3D(1.0, 0.0, 0.0)
        self.speed, self.jump = float(speed), bool(jump)


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False,
                 gear=0):
        self.throttle, self.steer, self.brake = float(throttle), float(steer), float(brake)
        self.hand_brake, self.reverse, self.manual_gear_shift, self.gear = bool(hand_brake), bool(reverse), \
            bool(manual_gear_shift), int(gear)


class LabelledPoint:
    def __init__(self, location, label):
        self.location, self.label = location, label


# ---------------------------------------------------------------------------------------------------- weather, settings
class WeatherParameters:
    FIELDS = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity', 'sun_azimuth_angle',
              'sun_altitude_angle', 'fog_density', 'fog_distance', 'fog_falloff', 'wetness', 'scattering_intensity',
              'mie_scattering_scale', 'rayleigh_scattering_scale', 'dust_storm')

    def __init__(self, cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=0.0,
                 sun_azimuth_angle=0.0, sun_altitude_angle=0.0, fog_density=0.0, fog_distance=0.0, fog_falloff=0.0,
                 wetness=0.0, scattering_intensity=0.0, mie_scattering_scale=0.0, rayleigh_scattering_scale=0.0331,
                 dust_storm=0.0):
        for k in self.FIELDS:
            setattr(self, k, float(locals()[k]))

    def __repr__(self):
        return 'WeatherParameters(' + ', '.join(f'{k}={getattr(self, k):.6f}' for k in self.FIELDS) + ')'


# CARLA 0.9.16's presets (the values of FIELDS in order, as carla.WeatherParameters of the 0.9.16 wheel has them)
_PRESETS = {
    'ClearNoon': (5, 0, 0, 10, -1, 45, 2, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'CloudyNoon': (60, 0, 0, 10, -1, 45, 3, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'WetNoon': (5, 0, 50, 10, -1, 45, 3, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'WetCloudyNoon': (60, 0, 50, 10, -1, 45, 3, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'SoftRainNoon': (20, 30, 50, 30, -1, 45, 3, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'MidRainyNoon': (60, 60, 60, 60, -1, 45, 3, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'HardRainNoon': (100, 100, 90, 100, -1, 45, 7, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'ClearSunset': (5, 0, 0, 10, -1, 15, 2, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'CloudySunset': (60, 0, 0, 10, -1, 15, 3, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'WetSunset': (5, 0, 50, 10, -1, 15, 2, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'WetCloudySunset': (60, 0, 50, 10, -1, 15, 2, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'SoftRainSunset': (20, 30, 50, 30, -1, 15, 2, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'MidRainSunset': (60, 60, 60, 60, -1, 15, 3, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'HardRainSunset': (100, 100, 90, 100, -1, 15, 7, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'ClearNight': (5, 0, 0, 10, -1, -90, 60, 75, 1, 0, 1, 0.03, 0.0331, 0),
    'CloudyNight': (60, 0, 0, 10, -1, -90, 60, 0.75, 0.1, 0, 1, 0.03, 0.0331, 0),
    'WetNight': (5, 0, 50, 10, -1, -90, 60, 75, 1, 60, 1, 0.03, 0.0331, 0),
    'WetCloudyNight': (60, 0, 50, 10, -1, -90, 60, 0.75, 0.1, 60, 1, 0.03, 0.0331, 0),
    'SoftRainNight': (60, 30, 50, 30, -1, -90, 60, 0.75, 0.1, 60, 1, 0.03, 0.0331, 0),
    'MidRainyNight': (80, 60, 60, 60, -1, -90, 60, 0.75, 0.1, 80, 1, 0.03, 0.0331, 0),
    'HardRainNight': (100, 100, 90, 100, -1, -90, 100, 0.75, 0.1, 100, 1, 0.03, 0.0331, 0),
    'DustStorm': (100, 0, 0, 100, -1, 45, 2, 0.75, 0.1, 0, 1, 0.03, 0.0331, 100),
    'Default': (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0.03, 0.0331, 0),
}
for _n, _v in _PRESETS.items():
    setattr(WeatherParameters, _n, WeatherParameters(*_v))


class WorldSettings:
    def __init__(self, synchronous_mode=False, no_rendering_mode=False, fixed_delta_seconds=None):
        self.synchronous_mode = synchronous_mode
        self.no_rendering_mode = no_rendering_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        self.substepping = True
        self.max_substep_delta_time = 0.01
        self.max_substeps = 10

    def _copy(self):
        s = WorldSettings(self.synchronous_mode, self.no_rendering_mode, self.fixed_delta_seconds)
        return s


# ---------------------------------------------------------------------------------------------------- blueprints
class ActorAttribute:
    """As the 0.9.16 wheel's: id, type, is_modifiable, recommended_values and as_*(); no `value` attribute (the real
    class has none — code must use as_str())."""

    def __init__(self, id, value):
        self.id, self._value = id, str(value)
        self.is_modifiable = True
        self.recommended_values = []
        self.type = 'String'

    def as_str(self):
        return self._value

    def as_float(self):
        return float(self._value)

    def as_int(self):
        return int(float(self._value))

    def as_bool(self):
        return self._value.lower() in ('true', '1')

    def __str__(self):
        return f'ActorAttribute(id={self.id},value={self._value})'


class ActorBlueprint:
    def __init__(self, id, attrs=None, tags=()):
        self.id = id
        self.tags = list(tags)
        self._a = {k: str(v) for k, v in (attrs or {}).items()}

    def has_attribute(self, k):
        return k in self._a

    def set_attribute(self, k, v):
        if k not in self._a and not k.startswith(('lens_', 'motion_', 'bloom', 'exposure', 'gamma', 'iso',
                                                  'shutter', 'fstop', 'chromatic', 'enable_')):
            raise IndexError(f'{self.id} has no attribute {k}')
        self._a[k] = str(v)

    def get_attribute(self, k):
        return ActorAttribute(k, self._a[k])

    def __iter__(self):
        return iter([ActorAttribute(k, v) for k, v in self._a.items()])

    def _f(self, k, d=0.0):
        try:
            return float(self._a.get(k, d))
        except ValueError:
            return d


# (base type, length, width, height, rear axle x) of the fake's vehicles, as CARLA 0.9.16's
VEHICLES = {'vehicle.audi.a2': ('car', 3.705, 1.789, 1.549, -1.256, 1.25),
            'vehicle.citroen.c3': ('car', 3.988, 1.851, 1.617, -1.410, 1.274),
            'vehicle.nissan.micra': ('car', 3.633, 1.845, 1.501, -1.317, 1.173),
            'vehicle.tesla.model3': ('car', 4.792, 2.163, 1.488, -1.386, 1.618),
            'vehicle.carlamotors.carlacola': ('truck', 5.204, 2.627, 2.467, -1.206, 1.459),
            'vehicle.ford.ambulance': ('van', 6.366, 2.351, 2.431, -1.661, 2.109)}
# found by name only (Blueprint.find), outside the library's listing, so the traffic the tests' seeds draw stays as it
# was: the car critscene stands a car target in with
NAMED_ONLY = {'vehicle.mini.cooper_s': ('car', 3.806, 1.970, 1.475, -1.367, 1.144),
              'vehicle.lincoln.mkz_2020': ('car', 4.901, 2.106, 1.510, -1.390, 1.450),   # Falcon's default rig car
              'vehicle.micro.microlino': ('car', 2.207, 1.481, 1.376, -0.800, 0.800),   # measured (inventory)
              'vehicle.mitsubishi.fusorosa': ('bus', 10.270, 3.930, 3.790, -3.300, 3.300)}
VEHICLES_ALL = dict(VEHICLES, **NAMED_ONLY)
# two-wheelers (base type, length, width, height): their rider is part of them, as in CARLA (tags 19/18 and 13)
TWO_WHEELERS = {'vehicle.diamondback.century': ('bicycle', 1.656, 0.582, 1.620),
                'vehicle.yamaha.yzf': ('motorcycle', 2.190, 0.900, 1.580)}
# found by name only, like NAMED_ONLY (two-wheelers): the cycle-rickshaw stand-in (core.assets), measured on 0.9.16
TWO_WHEELERS_NAMED = {'vehicle.bh.crossbike': ('bicycle', 1.509, 0.866, 1.612)}
# walkers: (age, half extents); the origin at the box centre, as CARLA's (measured: extent 0.19, 0.19, 0.93)
WALKERS = {'walker.pedestrian.0001': ('adult', (0.19, 0.19, 0.93)), 'walker.pedestrian.0002': ('adult', (0.2, 0.2, 0.9)),
           'walker.pedestrian.0009': ('child', (0.15, 0.15, 0.6)), 'walker.pedestrian.0010': ('child', (0.15, 0.15, 0.62))}
# props: (tag, half extents, box centre height); the origin on the ground (a cone measured: 0.44, 0.44, 0.57 at 0.57)
# (the street sign as measured on 0.9.16: a board 1.06 m along its x, 0.13 m thick, 2.16 m to its top)
PROPS = {'static.prop.trafficcone01': (21, (0.22, 0.22, 0.36), 0.36),
         'static.prop.streetbarrier': (20, (0.12, 1.0, 0.55), 0.55),
         'static.prop.streetsign': (8, (0.532, 0.064, 1.08), 1.08)}
# props Falcon's Indian proxies use (core.assets), as measured on 0.9.16: box (falcon_autonomy/resources/carla/
# inventory-0.9.16.json; the centre's x, y offsets left out) and the tag the instance camera gave them
PROPS.update({'static.prop.haybale': (21, (0.6239, 0.5967, 0.5975), 0.0039),
              'static.prop.foodcart': (21, (1.1423, 2.3192, 1.7866), 1.7834),
              'static.prop.maptable': (20, (0.7866, 0.3085, 0.5905), 0.5905),
              'static.prop.bench01': (20, (0.772, 0.3679, 0.4849), 0.4722)})
# static.prop.mesh: a cooked mesh by its path (tag, half extents, box centre height at scale 1). Measured on 0.9.16: the
# mesh scales about its origin with `scale`, but the reported bounding box keeps the unscaled centre (extents scaled);
# set_transform neither moves it nor keeps its scale; a path that does not exist spawns an empty actor (box ±inf)
MESHES = {'/Game/Carla/Static/Dynamic/Trash/SM_Box01.SM_Box01': (21, (0.3261, 0.3261, 0.3458), 0.3402),
          '/Game/Carla/Static/Static/SM_MapTable.SM_MapTable': (20, (0.7866, 0.3085, 0.5905), 0.5905),
          '/Game/Carla/Static/Static/SM_Bench01.SM_Bench01': (20, (0.772, 0.3679, 0.4849), 0.4722)}
# a traffic light of the map (an actor: its pole and head carry its id; the head is its light box and the level's
# TrafficLight box) and a sign (a level mesh: its own instance id; the plate is the level's TrafficSigns box):
# (cast id, instance id, tag, centre, half extents)
LIGHT_ID = 50
STOP_LINE_X = 8.0                    # the light's stop line across the +x lanes
FIXED = ((60001, LIGHT_ID, 7, (10.0, 6.0, 4.4), (0.2, 0.25, 0.6)),            # the head
         (60002, LIGHT_ID, 6, (10.0, 6.6, 2.5), (0.08, 0.08, 2.5)),            # its pole
         (60003, 41000, 8, (-10.0, 6.6, 2.3), (0.04, 0.4, 0.4)),               # the sign's plate
         (60004, 41001, 6, (-10.0, 6.7, 1.0), (0.03, 0.03, 1.0)))              # its post
LEVEL_CARS = [(-40.0, 11.5, 0.0), (0.0, 11.5, 0.0), (40.0, 11.5, 180.0)]   # the map's parked cars (x, y, yaw)
LEVEL_CAR_SIZE = (4.5, 1.9, 1.5)
LEVEL_ID0 = 40000                    # their instance ids (level meshes: not actor ids)
ROAD_Y = (-8.75, 5.25)               # the carriageway (road tag); beyond it the sidewalk
SENSOR_DEFAULTS = {
    'sensor.camera.rgb': {'image_size_x': 800, 'image_size_y': 600, 'fov': 90.0, 'sensor_tick': 0.0, 'gamma': 2.2,
                          'exposure_mode': 'histogram', 'exposure_compensation': 0.75, 'motion_blur_intensity': 0.45,
                          'lens_k': -1.0, 'lens_kcube': 0.0, 'lens_circle_multiplier': 0.0, 'bloom_intensity': 0.675,
                          'lens_flare_intensity': 0.1, 'role_name': 'front'},
    'sensor.lidar.ray_cast': {'channels': 32, 'range': 10.0, 'points_per_second': 56000, 'rotation_frequency': 10.0,
                              'upper_fov': 10.0, 'lower_fov': -30.0, 'horizontal_fov': 360.0,
                              'atmosphere_attenuation_rate': 0.004, 'dropoff_general_rate': 0.45,
                              'dropoff_intensity_limit': 0.8, 'dropoff_zero_intensity': 0.4, 'noise_stddev': 0.0,
                              'sensor_tick': 0.0, 'role_name': 'front'},
    'sensor.other.radar': {'horizontal_fov': 30.0, 'vertical_fov': 30.0, 'points_per_second': 1500, 'range': 100.0,
                           'sensor_tick': 0.0, 'role_name': 'front'},
    'sensor.other.gnss': {'noise_alt_bias': 0.0, 'noise_alt_stddev': 0.0, 'noise_lat_bias': 0.0,
                          'noise_lat_stddev': 0.0, 'noise_lon_bias': 0.0, 'noise_lon_stddev': 0.0, 'noise_seed': 0,
                          'sensor_tick': 0.0, 'role_name': 'front'},
    'sensor.other.imu': {'noise_accel_stddev_x': 0.0, 'noise_accel_stddev_y': 0.0, 'noise_accel_stddev_z': 0.0,
                         'noise_gyro_bias_x': 0.0, 'noise_gyro_bias_y': 0.0, 'noise_gyro_bias_z': 0.0,
                         'noise_gyro_stddev_x': 0.0, 'noise_gyro_stddev_y': 0.0, 'noise_gyro_stddev_z': 0.0,
                         'noise_seed': 0, 'sensor_tick': 0.0, 'role_name': 'front'},
    'sensor.lidar.ray_cast_semantic': {'channels': 32, 'range': 10.0, 'points_per_second': 56000,
                                       'rotation_frequency': 10.0, 'upper_fov': 10.0, 'lower_fov': -30.0,
                                       'horizontal_fov': 360.0, 'sensor_tick': 0.0, 'role_name': 'front'},
}
SENSOR_DEFAULTS['sensor.other.collision'] = {'role_name': 'collision'}
for _k in ('semantic_segmentation', 'instance_segmentation', 'depth', 'optical_flow'):
    SENSOR_DEFAULTS['sensor.camera.' + _k] = {'image_size_x': 800, 'image_size_y': 600, 'fov': 90.0, 'sensor_tick': 0.0,
                                             'lens_k': -1.0, 'lens_kcube': 0.0, 'lens_circle_multiplier': 0.0,
                                             'lens_circle_falloff': 5.0, 'role_name': 'front'}


class BlueprintLibrary:
    def __init__(self, bps=None):
        if bps is None:
            bps = [ActorBlueprint(k, {'base_type': v[0], 'number_of_wheels': 4, 'role_name': 'autopilot',
                                      'generation': 2}, ['vehicle', v[0]]) for k, v in VEHICLES.items()]
            bps += [ActorBlueprint(k, {'base_type': v[0], 'number_of_wheels': 4, 'role_name': 'autopilot',
                                      'generation': 2}, ['vehicle', v[0]]) for k, v in NAMED_ONLY.items()
                    if k in ('vehicle.micro.microlino', 'vehicle.mitsubishi.fusorosa')]
            bps += [ActorBlueprint(k, {'base_type': v[0], 'number_of_wheels': 2, 'role_name': 'autopilot',
                                       'generation': 2}, ['vehicle', v[0]]) for k, v in TWO_WHEELERS.items()]
            bps += [ActorBlueprint(k, {'age': v[0], 'role_name': 'pedestrian', 'is_invincible': 'true', 'speed': '1.4'},
                                   ['walker', 'pedestrian']) for k, v in WALKERS.items()]
            bps += [ActorBlueprint(k, {'role_name': 'prop', 'size': 'small'}, ['static', 'prop']) for k in PROPS]
            bps += [ActorBlueprint('static.prop.mesh', {'role_name': 'default', 'mesh_path': '', 'scale': '1.0',
                                                        'mass': '0.0'}, ['static', 'prop', 'mesh'])]
            bps += [ActorBlueprint('controller.ai.walker', {}, ['controller'])]
            bps += [ActorBlueprint(k, d, ['sensor']) for k, d in SENSOR_DEFAULTS.items()]
        self._bps = bps

    def filter(self, pattern):
        return BlueprintLibrary([b for b in self._bps if fnmatch.fnmatch(b.id, pattern)])

    def find(self, id):
        for b in self._bps:
            if b.id == id:
                return ActorBlueprint(b.id, dict(b._a), b.tags)
        if id in NAMED_ONLY:
            return ActorBlueprint(id, {'base_type': NAMED_ONLY[id][0], 'number_of_wheels': 4, 'role_name': 'autopilot',
                                       'generation': 2}, ['vehicle', NAMED_ONLY[id][0]])
        if id in TWO_WHEELERS_NAMED:
            return ActorBlueprint(id, {'base_type': TWO_WHEELERS_NAMED[id][0], 'number_of_wheels': 2,
                                       'role_name': 'autopilot', 'generation': 2}, ['vehicle', 'bicycle'])
        raise IndexError(f'blueprint {id!r} not found')

    def __iter__(self):
        return iter([ActorBlueprint(b.id, dict(b._a), b.tags) for b in self._bps])

    def __len__(self):
        return len(self._bps)


# ---------------------------------------------------------------------------------------------------- actors
class Actor:
    def __init__(self, world, aid, bp, transform, parent=None):
        self._w = world
        self.id = aid
        self.type_id = bp.id
        self.attributes = dict(bp._a)
        self._bp = bp
        self.parent = parent
        self._rel = transform._m() if parent is not None else None
        self._m_world = transform._m() if parent is None else None
        self.is_alive = True
        self.bounding_box = BoundingBox(Location(0, 0, 0), Vector3D(0.1, 0.1, 0.1))

    def world_matrix(self):
        if self.parent is None:
            return self._m_world
        return self.parent.world_matrix() @ self._rel

    def get_transform(self):
        return _T(self.world_matrix())

    def get_location(self):
        return self.get_transform().location

    def get_velocity(self):
        a = self.parent if self.parent is not None else self
        v = getattr(a, '_vel', np.zeros(3))
        return Vector3D(*v)

    def get_angular_velocity(self):
        a = self.parent if self.parent is not None else self
        return Vector3D(0.0, 0.0, math.degrees(getattr(a, '_yaw_rate', 0.0)))

    def get_world(self):
        return self._w

    def set_transform(self, tr):
        # asynchronous in LibCarla: FAKE_CARLA_LATE_SET_TRANSFORM=n delivers every n-th move a tick late (tests)
        w = self._w
        w._settle()
        if w._late_every and self.parent is None:
            w._late_n += 1
            if w._late_n % w._late_every == 0:
                w._late.append((self, tr))
                return
        self._apply_tf(tr)

    def _apply_tf(self, tr):
        if self.parent is None:
            self._m_world = tr._m()
        else:
            self._rel = tr._m()

    def set_location(self, loc):
        M = self.world_matrix().copy()
        M[:3, 3] = (loc.x, loc.y, loc.z)
        self.set_transform(_T(M))

    def set_simulate_physics(self, on=True):
        self.physics = bool(on)

    def destroy(self):
        if not self.is_alive:
            return False
        if getattr(self, '_tm_known', False) and not getattr(self, '_batch', False):
            # measured on 0.9.16: Actor.destroy() marks the client's actor object destroyed while the traffic manager
            # (a thread of this process) still holds it; its next update throws in that thread and libcarla ends the
            # process with std::terminate, even after set_autopilot(False). Vehicles that were on the autopilot must
            # be destroyed by id (client.apply_batch[_sync] with command.DestroyActor).
            import sys
            sys.stderr.write("terminate called after throwing an instance of 'std::runtime_error'\n  what():  trying "
                             "to operate on a destroyed actor; an actor's function was called, but the actor is "
                             "already destroyed.\n")
            sys.stderr.flush()
            os.abort()
        self.is_alive = False
        self._w._actors.pop(self.id, None)
        LOG.append(('destroy', self.id, self.type_id))
        return True


class _Wheel:
    def __init__(self, position, max_steer_angle=0.0):
        self.position = position
        self.max_steer_angle = max_steer_angle


class _Physics:
    def __init__(self, wheels, mass=1000.0, torque_curve=None, max_rpm=5000.0):
        self.wheels = wheels
        self.mass = mass
        self.max_rpm = max_rpm
        self.torque_curve = list(torque_curve) if torque_curve is not None else \
            [Vector2D(0.0, 400.0), Vector2D(max_rpm, 300.0)]


class Vehicle(Actor):
    def __init__(self, world, aid, bp, transform):
        super().__init__(world, aid, bp, transform)
        if bp.id in TWO_WHEELERS or bp.id in TWO_WHEELERS_NAMED:
            base, L, W, H = TWO_WHEELERS.get(bp.id) or TWO_WHEELERS_NAMED[bp.id]
            rear, front = -L / 2 + 0.3, L / 2 - 0.3
        else:
            base, L, W, H, rear, front = VEHICLES_ALL.get(bp.id, ('car', 4.5, 1.9, 1.5, -1.4, 1.4))
        self.bounding_box = BoundingBox(Location(0.0, 0.0, H / 2), Vector3D(L / 2, W / 2, H / 2))
        self._axles = (rear, front, W / 2 - 0.1)
        self._vel = np.zeros(3)
        self._yaw_rate = 0.0
        self._speed = 0.0
        self.autopilot = False
        self.light_state = 0
        self.control = VehicleControl()
        self.constant_velocity = None
        self.physics = True
        self.desired = None                 # the traffic manager's desired speed (m/s), when set
        self.lat_target = None              # a forced lane change: the target offset along the vehicle's right (m)
        self.lane_offset = None             # TrafficManager.vehicle_lane_offset (m, + right of its lane's centre)
        self.lane_y = None                  # … of the lane it had when the offset was first set (the TM's lane)
        self.ignore_vehicles = 0.0          # TrafficManager.ignore_vehicles_percentage

    def open_door(self, door):
        LOG.append(('open_door', self.id, str(door)))

    def close_door(self, door):
        LOG.append(('close_door', self.id, str(door)))

    def set_autopilot(self, on=True, port=8000):
        self.autopilot = bool(on)
        if on:
            self._tm_known = True       # the traffic manager (a thread of this client) now holds this actor
        rng = self._w._rng
        if on and self.desired is None:
            if self.attributes.get('role_name') == 'hero':
                self._speed, self._yaw_rate = 8.0, math.radians(3.0)
            else:
                self._speed, self._yaw_rate = float(rng.uniform(3.0, 10.0)), 0.0
        LOG.append(('autopilot', self.id, bool(on), port))

    def apply_control(self, c):
        self.control = c
        LOG.append(('control', self.id, c.throttle, c.brake, c.hand_brake))

    def get_control(self):
        return self.control

    def enable_constant_velocity(self, v):
        self.constant_velocity = np.array([v.x, v.y, v.z], float)

    def disable_constant_velocity(self):
        self.constant_velocity = None

    def set_target_velocity(self, v):
        self._vel = np.array([v.x, v.y, v.z], float)

    def set_target_angular_velocity(self, v):
        self._ang_vel = np.array([v.x, v.y, v.z], float)

    def get_physics_control(self):
        rear, front, half = self._axles
        M = self.world_matrix()
        pts = [(front, -half), (front, half), (rear, -half), (rear, half)]
        tc = getattr(self, '_torque', None)
        return _Physics([_Wheel(Vector3D(*(100.0 * (M @ np.array([x, y, 0.3, 1.0]))[:3])), 70.0 if x == front else 0.0)
                         for x, y in pts], torque_curve=tc)

    def apply_physics_control(self, pc):
        self._torque = list(pc.torque_curve)
        LOG.append(('physics_control', self.id, [(p.x, p.y) for p in pc.torque_curve]))

    def get_speed_limit(self):
        return 30.0                         # (the fake road's limit, km/h)

    def set_light_state(self, s):
        self.light_state = s

    def get_light_state(self):
        return self.light_state

    def _lead_gap(self):
        """The gap (m) to the nearest actor ahead in this vehicle's lane (within 1.6 m of its path), or None. Like
        CARLA 0.9.16's traffic manager (measured, docs/indian-assets.md): the path is the lane's centre whatever the
        vehicle's lane offset, and with ignore_vehicles_percentage 100 only walkers and props count."""
        M = self._m_world
        fwd, right = M[:3, 0], np.array([-M[1, 0], M[0, 0], 0.0])
        origin = M[:3, 3].copy()
        if self.lane_offset is not None:
            origin[1] = self.lane_y
        best = None
        for a in self._w._actors.values():
            if a is self or not isinstance(a, (Vehicle, Walker, Prop)) or not a.is_alive:
                continue
            if isinstance(a, Vehicle) and self.ignore_vehicles >= 100.0:
                continue
            q = a.world_matrix()[:3, 3] - origin
            lon, lat = float(q @ fwd), float(q @ right)
            ext = a.bounding_box.extent
            if lon > 0 and abs(lat) < 1.6 + 0.5 * ext.y and lon < 60.0:
                g = lon - self.bounding_box.extent.x - max(ext.x, ext.y)
                best = g if best is None else min(best, g)
        return best

    def _step(self, dt):
        M = self._m_world
        yaw = math.atan2(M[1, 0], M[0, 0])
        if self.constant_velocity is not None:             # its own axes, held whatever the controls
            v = M[:3, :3] @ self.constant_velocity
            self._vel = v
            self._m_world = M.copy()
            self._m_world[:3, 3] = M[:3, 3] + v * dt
            return
        if not self.autopilot:                         # its controls: a kinematic bicycle model
            c = self.control
            sp = float(np.hypot(*self._vel[:2]))
            if c.hand_brake or not self.physics:
                sp = 0.0
            else:
                sp = max(sp + (3.0 * c.throttle - 8.0 * c.brake) * dt, 0.0)
            rear, front, _ = self._axles
            yaw2 = yaw + sp * math.tan(c.steer * math.radians(70.0)) / max(front - rear, 0.5) * dt
            mid = (yaw + yaw2) / 2
            self._speed = sp
            self._vel = np.array([math.cos(yaw2), math.sin(yaw2), 0.0]) * sp
            M2 = np.eye(4)
            M2[:3, :3] = Rotation(0.0, math.degrees(yaw2), 0.0).matrix()
            M2[:3, 3] = M[:3, 3] + np.array([math.cos(mid), math.sin(mid), 0.0]) * sp * dt
            self._m_world = M2
            return
        sp0 = self._speed
        if self.desired is not None:                       # a scenario's vehicle: its lane, its speed, braking
            gap = self._lead_gap()
            target = self.desired if gap is None else min(self.desired, max(0.0, gap - 4.0))   # 1 s headway, 4 m
            self._speed += float(np.clip(target - self._speed, -8.0 * dt, 3.0 * dt))
            self._yaw_rate = 0.0
        # the control the traffic manager "applied": what produces this motion under the fake's own vehicle model
        # (throttle 3 m/s², brake 8 m/s², the bicycle model's steer), so get_control() is consistent with the motion
        rear, front, _ = self._axles
        dv = self._speed - sp0
        steer = math.atan(self._yaw_rate * max(front - rear, 0.5) / max(self._speed, 0.1)) / math.radians(70.0)
        self.control = VehicleControl(throttle=min(max(dv / dt / 3.0, 0.0), 1.0), steer=max(min(steer, 1.0), -1.0),
                                      brake=min(max(-dv / dt / 8.0, 0.0), 1.0))
        yaw2 = yaw + self._yaw_rate * dt
        mid = (yaw + yaw2) / 2
        d = np.array([math.cos(mid), math.sin(mid), 0.0]) * self._speed * dt
        lat_v = np.zeros(3)
        if self.lane_offset is not None and self.lat_target is None:     # the lane offset: 1 m/s sideways towards it
            right = np.array([-math.sin(yaw2), math.cos(yaw2), 0.0])
            off = float((M[1, 3] - self.lane_y) * right[1])
            lat_v = right * float(np.clip(self.lane_offset - off, -1.0 * dt, 1.0 * dt))
        if self.lat_target is not None:                    # a forced lane change: 1.75 m/s sideways
            step = float(np.clip(self.lat_target, -1.75 * dt, 1.75 * dt))
            self.lat_target -= step
            lat_v = np.array([-math.sin(yaw2), math.cos(yaw2), 0.0]) * step
            if abs(self.lat_target) < 1e-6:
                self.lat_target = None
        R = Rotation(0.0, math.degrees(yaw2), 0.0).matrix()
        M2 = np.eye(4)
        M2[:3, :3] = R
        M2[:3, 3] = M[:3, 3] + d + lat_v
        self._vel = np.array([math.cos(yaw2), math.sin(yaw2), 0.0]) * self._speed + lat_v / dt
        self._m_world = M2


class Walker(Actor):
    """A walker: its origin at its box centre (as CARLA's); it moves by WalkerControl, else stands."""

    def __init__(self, world, aid, bp, transform):
        super().__init__(world, aid, bp, transform)
        age, ext = WALKERS.get(bp.id, ('adult', (0.19, 0.19, 0.93)))
        self.bounding_box = BoundingBox(Location(0.0, 0.0, 0.0), Vector3D(*ext))
        self._vel = np.zeros(3)
        self.control = WalkerControl()
        self.physics = True

    def apply_control(self, c):
        self.control = c
        LOG.append(('walker_control', self.id, c.speed))

    def get_control(self):
        return self.control

    def _step(self, dt):
        c = self.control
        d = np.array([c.direction.x, c.direction.y, 0.0])
        n = np.linalg.norm(d)
        self._vel = d / n * c.speed if n > 1e-9 and c.speed > 0 else np.zeros(3)
        self._m_world = self._m_world.copy()
        self._m_world[:3, 3] += self._vel * dt


class WalkerAIController(Actor):
    """controller.ai.walker: attached to a walker, walks it to a target at its maximum speed once started."""

    def __init__(self, world, aid, bp, transform, parent):
        super().__init__(world, aid, bp, transform, parent)
        self.target, self.max_speed, self.running = None, 1.4, False

    def start(self):
        self.running = True
        LOG.append(('walker_ai_start', self.id))

    def stop(self):
        self.running = False
        LOG.append(('walker_ai_stop', self.id))

    def go_to_location(self, loc):
        self.target = np.array([loc.x, loc.y])

    def set_max_speed(self, v):
        self.max_speed = float(v)

    def _step(self, dt):
        w = self.parent
        if not self.running or self.target is None or not w.is_alive:
            return
        d = self.target - w.world_matrix()[:2, 3]
        n = float(np.linalg.norm(d))
        w.control = WalkerControl(Vector3D(d[0] / max(n, 1e-9), d[1] / max(n, 1e-9), 0.0),
                                  self.max_speed if n > 0.3 else 0.0)


class Prop(Actor):
    """A static prop: its origin on the ground under its box. As 0.9.16 (measured), its bounding_box is reported in the
    world's axes: a prop turned 90° gives its length and width swapped; _box is its own."""

    def __init__(self, world, aid, bp, transform):
        super().__init__(world, aid, bp, transform)
        self.scale = 1.0
        if bp.id == 'static.prop.mesh':
            self.mesh = bp._a.get('mesh_path', '')
            self.scale = float(bp._a.get('scale', 1.0) or 1.0)
            tag, ext, zc = MESHES.get(self.mesh, (0, (0.0, 0.0, 0.0), 0.0))
            self.empty = self.mesh not in MESHES
        else:
            tag, ext, zc = PROPS.get(bp.id, (20, (0.3, 0.3, 0.5), 0.5))
            self.empty = False
        self.tag, self._ext0, self._zc0 = tag, ext, zc
        self._set_box()
        self.physics = True

    def _set_box(self):
        k = self.scale                       # the drawn mesh: scaled about its origin
        self._box = BoundingBox(Location(0.0, 0.0, self._zc0 * k), Vector3D(*(e * k for e in self._ext0)))

    def _apply_tf(self, tr):
        if self.type_id == 'static.prop.mesh':   # measured: not moved, and its scale back to 1
            self.scale = 1.0
            self._set_box()
            return
        super()._apply_tf(tr)

    @property
    def bounding_box(self):
        if self.empty:
            return BoundingBox(Location(float('nan'), float('nan'), float('nan')),
                               Vector3D(-math.inf, -math.inf, -math.inf))
        R = self.world_matrix()[:3, :3]
        e = np.abs(R) @ np.array([self._box.extent.x, self._box.extent.y, self._box.extent.z])
        return BoundingBox(Location(0.0, 0.0, self._zc0), Vector3D(*e))   # the centre as CARLA reports it: unscaled

    @bounding_box.setter
    def bounding_box(self, b):
        pass


class TrafficLightState:
    pass


_enum(TrafficLightState, ['Red', 'Yellow', 'Green', 'Off', 'Unknown'])


class TrafficLight(Actor):
    """The map's traffic light: an actor of the world (not spawned, never destroyed by a client). It controls the two
    +x lanes (ids −1, −2), whose stop line is at x = STOP_LINE_X. Its state is constant: Green, or the environment's
    FAKE_CARLA_LIGHT_STATE (tests), or what set_state() set."""

    def __init__(self, world):
        bp = ActorBlueprint('traffic.traffic_light', {'role_name': ''}, ['traffic'])
        super().__init__(world, LIGHT_ID, bp, Transform(Location(10.0, 6.6, 0.0)))
        self.state = getattr(TrafficLightState, os.environ.get('FAKE_CARLA_LIGHT_STATE', 'Green'))

    def get_light_boxes(self):
        _, _, _, c, e = FIXED[0]
        return [BoundingBox(Location(*c), Vector3D(*e))]

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state
        LOG.append(('light_state', str(state)))

    def freeze(self, on):
        pass

    def get_stop_waypoints(self):
        return [Waypoint(-1, STOP_LINE_X), Waypoint(-2, STOP_LINE_X)]


class Sensor(Actor):
    def __init__(self, world, aid, bp, transform, parent):
        super().__init__(world, aid, bp, transform, parent)
        self._cb = None
        self._tick = bp._f('sensor_tick', 0.0)
        self._due = None
        self._prev = [None, None]       # positions (IMU: two before; radar: one before)
        self._rng = np.random.default_rng(aid)
        self._angle = 0.0

    def listen(self, cb):
        self._cb = cb
        LOG.append(('listen', self.id, self.type_id))

    def stop(self):
        self._cb = None

    @property
    def is_listening(self):
        return self._cb is not None

    def _ready(self, t, dt):
        if self._cb is None:
            return False
        if self._tick <= 0:
            return True
        if self._due is None:                         # the first tick after it is listened to
            self._due = t + self._tick
            return True
        if t >= self._due - 1e-9:
            self._due += self._tick                    # Unreal's tick interval carries the remainder over
            return True
        return False


class ServerSideSensor(Sensor):
    pass


# ---------------------------------------------------------------------------------------------------- sensor data
class SensorData:
    def __init__(self, frame, t, transform):
        self.frame, self.frame_number, self.timestamp, self.transform = frame, frame, t, transform


class CollisionEvent(SensorData):
    def __init__(self, frame, timestamp, transform, actor, other_actor, normal_impulse):
        super().__init__(frame, timestamp, transform)
        self.actor, self.other_actor, self.normal_impulse = actor, other_actor, normal_impulse


class Image(SensorData):
    def __init__(self, frame, t, transform, w, h, fov, raw):
        super().__init__(frame, t, transform)
        self.width, self.height, self.fov, self.raw_data = w, h, fov, raw


class LidarMeasurement(SensorData):
    def __init__(self, frame, t, transform, channels, counts, raw, angle):
        super().__init__(frame, t, transform)
        self.channels, self._counts, self.raw_data, self.horizontal_angle = channels, counts, raw, angle

    def get_point_count(self, ch):
        return int(self._counts[ch])

    def __len__(self):
        return int(sum(self._counts))


class SemanticLidarMeasurement(SensorData):
    def __init__(self, frame, t, transform, channels, counts, raw, angle):
        super().__init__(frame, t, transform)
        self.channels, self._counts, self.raw_data, self.horizontal_angle = channels, counts, raw, angle

    def get_point_count(self, ch):
        return int(self._counts[ch])

    def __len__(self):
        return int(sum(self._counts))


class RadarDetection:
    def __init__(self, depth, azimuth, altitude, velocity):
        self.depth, self.azimuth, self.altitude, self.velocity = depth, azimuth, altitude, velocity


class RadarMeasurement(SensorData):
    def __init__(self, frame, t, transform, dets):
        super().__init__(frame, t, transform)
        self._d = dets
        self.raw_data = np.array([[d.velocity, d.azimuth, d.altitude, d.depth] for d in dets], np.float32).tobytes()

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def get_detection_count(self):
        return len(self._d)


class GnssMeasurement(SensorData):
    def __init__(self, frame, t, transform, lat, lon, alt):
        super().__init__(frame, t, transform)
        self.latitude, self.longitude, self.altitude = lat, lon, alt


class IMUMeasurement(SensorData):
    def __init__(self, frame, t, transform, acc, gyro, compass):
        super().__init__(frame, t, transform)
        self.accelerometer, self.gyroscope, self.compass = acc, gyro, compass


# ---------------------------------------------------------------------------------------------------- the world
class _ActorSnapshot:
    def __init__(self, a):
        self.id = a.id
        self._t = a.get_transform()
        self._v = a.get_velocity()
        self._w = a.get_angular_velocity()

    def get_transform(self):
        return self._t

    def get_velocity(self):
        return self._v

    def get_angular_velocity(self):
        return self._w

    def get_acceleration(self):
        return Vector3D()


class WorldSnapshot:
    def __init__(self, frame, t, actors):
        self.frame = frame
        self.timestamp = Timestamp(frame, t)
        self._a = {a.id: _ActorSnapshot(a) for a in actors}

    def find(self, aid):
        return self._a.get(aid)

    def __iter__(self):
        return iter(self._a.values())

    def has_actor(self, aid):
        return aid in self._a


class ActorList(list):
    def filter(self, pattern):
        return ActorList([a for a in self if fnmatch.fnmatch(a.type_id, pattern)])

    def find(self, aid):
        return next((a for a in self if a.id == aid), None)


# the road: (lane id, y, yaw): right-hand traffic like CARLA's towns, CARLA's axes (+y is the right of +x)
LANES = ((-1, 0.0, 0.0), (-2, 3.5, 0.0), (1, -3.5, 180.0), (2, -7.0, 180.0))
SIDEWALKS = ((-3, 7.0, 0.0), (3, -10.5, 180.0))   # beyond the carriageway (ROAD_Y): lane type Sidewalk, as an
                                                 # OpenDRIVE road's outermost lanes; reached by get_right_lane only
ROAD_X = (-150.0, 150.0)


class Waypoint:
    """A lane centre point of the fake's straight road, with CARLA's lane-id rules (Map::GetRight moves away from the
    road's centre line, GetLeft towards it and, from lane ±1, across)."""

    def __init__(self, lane_id, x):
        self.lane_id = int(lane_id)
        self.road_id, self.section_id = 1, 0
        _, y, yaw = next(l for l in LANES + SIDEWALKS if l[0] == self.lane_id)
        self.s = float(x)
        self.transform = Transform(Location(float(x), y, 0.0), Rotation(0.0, yaw, 0.0))
        self.lane_width = 3.5
        self.lane_type = LaneType.Sidewalk if any(l[0] == self.lane_id for l in SIDEWALKS) else LaneType.Driving
        self.is_junction = False
        # the marking: broken between the two lanes of a direction, solid at the centre line and the kerb
        self.lane_change = LaneChange.Right if abs(self.lane_id) == 1 else LaneChange.Left
        inner = LaneMarking(LaneMarkingType.SolidSolid, LaneMarkingColor.Yellow)
        between = LaneMarking(LaneMarkingType.Broken, LaneMarkingColor.White)
        outer = LaneMarking(LaneMarkingType.Solid, LaneMarkingColor.White)
        self.left_lane_marking = inner if abs(self.lane_id) == 1 else between
        self.right_lane_marking = between if abs(self.lane_id) == 1 else outer
        self.id = hash((self.lane_id, round(float(x), 3)))

    def _dir(self):
        return 1.0 if abs(self.transform.rotation.yaw) < 90 else -1.0

    def next(self, d):
        x = self.transform.location.x + self._dir() * float(d)
        return [Waypoint(self.lane_id, x)] if ROAD_X[0] <= x <= ROAD_X[1] else []

    def previous(self, d):
        x = self.transform.location.x - self._dir() * float(d)
        return [Waypoint(self.lane_id, x)] if ROAD_X[0] <= x <= ROAD_X[1] else []

    def _lane(self, lid, sidewalks=False):
        lanes = LANES + SIDEWALKS if sidewalks else LANES
        return Waypoint(lid, self.transform.location.x) if any(l[0] == lid for l in lanes) else None

    def get_right_lane(self):
        """Away from the centre line, as CARLA's — across the lane types: the outermost driving lane's right lane
        is its sidewalk."""
        return self._lane(self.lane_id + (1 if self.lane_id > 0 else -1), sidewalks=True)

    def get_left_lane(self):
        lid = -self.lane_id if abs(self.lane_id) == 1 else self.lane_id + (-1 if self.lane_id > 0 else 1)
        return self._lane(lid)


class _Waypoint:
    def __init__(self, tr, width):
        self.transform, self.lane_width = tr, width


class Map:
    def __init__(self, name):
        self.name = f'Carla/Maps/{name}'
        self._lanes = [(y, yaw) for _, y, yaw in LANES]

    def to_opendrive(self):
        return getattr(self, '_xodr', f'<OpenDRIVE><!-- {self.name} --></OpenDRIVE>')

    def get_spawn_points(self):
        out = []
        for y, yaw in self._lanes:
            for x in np.arange(-60.0, 61.0, 12.0):
                out.append(Transform(Location(float(x), y, 0.3), Rotation(0.0, yaw, 0.0)))
        return out

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        """The nearest lane's waypoint; with project_to_road False, None off the carriageway (as CARLA: no driving lane
        there)."""
        if not project_to_road and not (ROAD_Y[0] <= location.y <= ROAD_Y[1] and
                                        ROAD_X[0] <= location.x <= ROAD_X[1]):
            return None
        lid = min(LANES, key=lambda l: abs(l[1] - location.y))[0]
        return Waypoint(lid, float(np.clip(location.x, *ROAD_X)))

    def get_topology(self):
        """(start, end) waypoint pairs of every lane segment: the fake's road is one segment per lane."""
        out = []
        for lid, _, yaw in LANES:
            a, b = (ROAD_X[0], ROAD_X[1]) if abs(yaw) < 90 else (ROAD_X[1], ROAD_X[0])
            out.append((Waypoint(lid, a), Waypoint(lid, b)))
        return out

    def generate_waypoints(self, distance):
        """Every driving lane's waypoints, distance apart (as CARLA's: full Waypoint objects)."""
        return [Waypoint(lid, float(x)) for lid, _, _ in LANES
                for x in np.arange(ROAD_X[0], ROAD_X[1], distance)]


class TrafficManager:
    def __init__(self, port):
        self.port = port
        self.sync = False

    def get_port(self):
        return self.port

    def set_synchronous_mode(self, on):
        self.sync = bool(on)
        LOG.append(('tm_sync', bool(on)))

    def set_random_device_seed(self, s):
        LOG.append(('tm_seed', int(s)))

    def set_desired_speed(self, actor, kmh):
        """(The fake's vehicle is at its desired speed at once, as if it had been cruising.)"""
        first = actor.desired is None
        actor.desired = float(kmh) / 3.6
        if first:
            actor._speed = actor.desired
        LOG.append(('tm_speed', actor.id, float(kmh)))

    def force_lane_change(self, actor, direction):
        """direction True: to the right lane (CARLA's)."""
        actor.lat_target = 3.5 if direction else -3.5
        LOG.append(('tm_lane_change', actor.id, bool(direction)))

    def auto_lane_change(self, actor, on):
        LOG.append(('tm_auto_lane_change', actor.id, bool(on)))

    def ignore_lights_percentage(self, actor, p):
        pass

    def ignore_signs_percentage(self, actor, p):
        pass

    def ignore_walkers_percentage(self, actor, p):
        pass

    def ignore_vehicles_percentage(self, actor, p):
        actor.ignore_vehicles = float(p)
        LOG.append(('tm_ignore_vehicles', actor.id, float(p)))

    def vehicle_lane_offset(self, actor, offset):
        """The offset from the centre of the lane the vehicle is on when it is first set (the fake's TM keeps that
        lane), + right; reached at 1 m/s sideways."""
        if actor.lane_offset is None:
            y = actor.get_transform().location.y
            actor.lane_y = min((l[1] for l in LANES), key=lambda ly: abs(ly - y))
        actor.lane_offset = float(offset)
        LOG.append(('tm_lane_offset', actor.id, round(float(offset), 3)))

    def distance_to_leading_vehicle(self, actor, d):
        pass

    def global_percentage_speed_difference(self, v):
        pass

    def vehicle_percentage_speed_difference(self, actor, p):
        """(Recorded only: the fake's autopilot speed is its own unless set_desired_speed gives one.)"""
        actor.pct_diff = float(p)
        LOG.append(('tm_speed_difference', actor.id, float(p)))

    def set_global_distance_to_leading_vehicle(self, v):
        pass

    def update_vehicle_lights(self, actor, on):
        pass

    def set_hybrid_physics_mode(self, on):
        pass


class World:
    def __init__(self, town='Town10HD_Opt'):
        self.id = 1
        self._map = Map(town)
        self._settings = WorldSettings()
        self._weather = WeatherParameters.ClearNoon
        self._actors = {}
        self._next = 100
        self._frame = 1000
        self._t = 4.5
        self._rng = np.random.default_rng(0)
        self.applied = []
        self._cast_cache = {}
        self._light = TrafficLight(self)
        self._late_every = int(os.environ.get('FAKE_CARLA_LATE_SET_TRANSFORM', '0') or 0)
        self._late, self._late_ready, self._late_n = [], [], 0

    def _settle(self):
        """Moves that missed a tick arrive now (before anything sent after them)."""
        ready, self._late_ready = self._late_ready, []
        for a, tr in ready:
            if a.is_alive:
                a._apply_tf(tr)

    # ---- queries
    def get_map(self):
        return self._map

    def get_settings(self):
        return self._settings._copy()

    def apply_settings(self, s):
        self._settings = s._copy()
        self.applied.append((s.synchronous_mode, s.fixed_delta_seconds))
        LOG.append(('settings', bool(s.synchronous_mode), s.fixed_delta_seconds))
        return self._frame

    def get_blueprint_library(self):
        return BlueprintLibrary()

    def get_weather(self):
        return self._weather

    def set_weather(self, w):
        self._weather = w
        LOG.append(('weather', repr(w)))

    def get_actors(self, ids=None):
        if getattr(self, '_unseen', False):          # a new client of a synchronous server before any step: no
            return ActorList([])                     # snapshot has reached it, so it knows no actors (0.9.16)
        acts = list(self._actors.values()) + [self._light]
        if ids is not None:
            acts = [a for a in acts if a.id in set(ids)]
        return ActorList(acts)

    def get_actor(self, aid):
        return self._actors.get(aid)

    def get_snapshot(self):
        return WorldSnapshot(self._frame, self._t, list(self._actors.values()))

    def get_level_bbs(self, label=None):
        """The map's parked cars (level meshes, not actors) for CityObjectLabel.Car, the traffic light's head for
        TrafficLight, the sign's plate for TrafficSigns: world boxes."""
        if label in (CityObjectLabel.TrafficLight, CityObjectLabel.TrafficSigns):
            _, _, _, c, e = FIXED[0 if label == CityObjectLabel.TrafficLight else 2]
            return [BoundingBox(Location(*c), Vector3D(*e))]
        if label != CityObjectLabel.Car:
            return []
        L, W, H = LEVEL_CAR_SIZE
        out = []
        for x, y, yaw in LEVEL_CARS:
            b = BoundingBox(Location(x, y, H / 2), Vector3D(L / 2, W / 2, H / 2))
            b.rotation = Rotation(0.0, yaw, 0.0)
            out.append(b)
        return out

    def reset_all_traffic_lights(self):
        LOG.append(('reset_traffic_lights',))

    def set_pedestrians_seed(self, s):
        self._nav = np.random.default_rng([int(s), 77])
        LOG.append(('pedestrians_seed', int(s)))

    def get_random_location_from_navigation(self):
        """A point of the sidewalks (beyond the carriageway on either side), from the pedestrians' seed."""
        rng = getattr(self, '_nav', None) or self._rng
        side = ROAD_Y[1] + 1.5 if rng.random() < 0.5 else ROAD_Y[0] - 1.5
        return Location(float(rng.uniform(-50.0, 50.0)), side, 0.0)

    def ground_projection(self, location, search_distance=10.0):
        if 0.0 <= location.z <= search_distance:
            return LabelledPoint(Location(location.x, location.y, 0.0), _ground_tag(location.y))
        return None

    # ---- spawning
    def try_spawn_actor(self, bp, transform, attach_to=None):
        try:
            return self.spawn_actor(bp, transform, attach_to)
        except RuntimeError:
            return None

    def spawn_actor(self, bp, transform, attach_to=None):
        if bp.id.startswith(('vehicle.', 'walker.', 'static.prop.')):
            near = 2.0 if bp.id.startswith('vehicle.') else 0.6
            for a in self._actors.values():
                if isinstance(a, (Vehicle, Walker, Prop)) and a.get_location().distance(transform.location) < near:
                    raise RuntimeError('Spawn failed because of collision at spawn position')
            cls = Vehicle if bp.id.startswith('vehicle.') else Walker if bp.id.startswith('walker.') else Prop
            if cls is not Vehicle and bp.id not in (WALKERS if cls is Walker else PROPS) and \
                    bp.id != 'static.prop.mesh':
                raise RuntimeError(f'cannot spawn {bp.id}')
            a = cls(self, self._next, bp, transform)
            if cls is not Prop:                        # vehicles and walkers settle on the ground, as CARLA's physics
                a._m_world = a._m_world.copy()
                a._m_world[2, 3] = a.bounding_box.extent.z - a.bounding_box.location.z
        elif bp.id == 'controller.ai.walker':
            if not isinstance(attach_to, Walker):
                raise RuntimeError('a walker controller needs a walker')
            a = WalkerAIController(self, self._next, bp, transform, attach_to)
        elif bp.id.startswith('sensor.'):
            # attach_to None: a sensor standing in the world (as CARLA allows)
            a = ServerSideSensor(self, self._next, bp, transform, attach_to)
        else:
            raise RuntimeError(f'cannot spawn {bp.id}')
        self._next += 1
        self._actors[a.id] = a
        LOG.append(('spawn', a.id, bp.id, dict(bp._a), [transform.location.x, transform.location.y,
                                                        transform.location.z, transform.rotation.pitch,
                                                        transform.rotation.yaw, transform.rotation.roll],
                    attach_to.id if attach_to is not None else None))
        return a

    # ---- the step
    def tick(self, seconds=10.0):
        hang = os.environ.get('FAKE_CARLA_HANG_AT_TICK')   # a crashed server: the call never returns (tests)
        self._ticks = getattr(self, '_ticks', 0) + 1
        if hang and self._ticks >= int(hang):
            import time
            time.sleep(3600)
        dt = self._settings.fixed_delta_seconds or 0.05
        self._settle()
        self._unseen = False
        self._frame += 1
        self._t = round(self._t + dt, 12)
        for a in list(self._actors.values()):
            if isinstance(a, WalkerAIController):
                a._step(dt)
        for a in list(self._actors.values()):
            if isinstance(a, (Vehicle, Walker)):
                a._step(dt)
        self._cast_cache = {}
        boxes = self._boxes()
        for a in list(self._actors.values()):
            if isinstance(a, Sensor) and a._ready(self._t, dt):
                data = self._measure(a, dt, boxes)
                if data is not None and a._cb is not None:
                    a._cb(data)
        self._late_ready += self._late            # sent before this tick but processed after it
        self._late = []
        return self._frame

    def wait_for_tick(self, seconds=10.0):
        self._unseen = False
        return self.get_snapshot()

    def _boxes(self):
        """[(id, world 4x4 of the box centre, half extents)] of every vehicle, walker and prop (their actor ids) and
        the map's parked cars (LEVEL_ID0 + their index)."""
        out = []
        for a in self._actors.values():
            if isinstance(a, (Vehicle, Walker, Prop)):
                M = a.world_matrix().copy()
                bb = a._box if isinstance(a, Prop) else a.bounding_box
                M[:3, 3] = M[:3, 3] + M[:3, :3] @ np.array([bb.location.x, bb.location.y, bb.location.z])
                out.append((a.id, M, np.array([bb.extent.x, bb.extent.y, bb.extent.z])))
        L, W, H = LEVEL_CAR_SIZE
        for k, (x, y, yaw) in enumerate(LEVEL_CARS):
            M = np.eye(4)
            M[:3, :3] = Rotation(0.0, yaw, 0.0).matrix()
            M[:3, 3] = (x, y, H / 2)
            out.append((LEVEL_ID0 + k, M, np.array([L / 2, W / 2, H / 2])))
        for cid, _, _, c, e in FIXED:
            M = np.eye(4)
            M[:3, 3] = c
            out.append((cid, M, np.array(e, float)))
        return out

    def _labels(self, o, D, t, hid, boxes):
        """(tag, id, cos of the incidence) of each ray's hit: the ground road or sidewalk (id 0), an actor by its kind
        (its actor id; a two-wheeler's upper half its rider), a parked car of the map (car, its level id); sky 11."""
        n = len(D)
        tag = np.full(n, CityObjectLabel.Sky, np.uint32)
        ids = np.zeros(n, np.int64)
        cos = np.zeros(n, np.float32)
        hit = np.isfinite(t)
        g = hit & (hid < 0)
        P = o + D * np.where(hit, t, 0.0)[:, None]
        tag[g] = _ground_tag(P[g, 1])
        cos[g] = np.abs(D[g, 2])
        fixed = {f[0]: f for f in FIXED}
        for aid, Mb, ext in boxes:
            sel = hit & (hid == aid)
            if not sel.any():
                continue
            q = (P[sel] - Mb[:3, 3]) @ Mb[:3, :3]
            face = np.argmax(np.abs(q) / ext, axis=1)
            cos[sel] = np.abs(np.sum(D[sel] * Mb[:3, :3][:, face].T, axis=1))
            if aid in fixed:                           # the light (its actor id) and the sign (a level mesh's id)
                ids[sel] = fixed[aid][1]
                tag[sel] = fixed[aid][2]
                continue
            ids[sel] = aid
            a = self._actors.get(aid)
            if aid >= LEVEL_ID0 or isinstance(a, Vehicle):
                base = TWO_WHEELERS.get(a.type_id, (VEHICLES_ALL.get(a.type_id, ('car',))[0],))[0] if a is not None \
                    else 'car'
                tg = {'truck': CityObjectLabel.Truck, 'bicycle': CityObjectLabel.Bicycle,
                      'motorcycle': CityObjectLabel.Motorcycle}.get(base, CityObjectLabel.Car)
                tg = np.full(int(sel.sum()), tg, np.uint32)
                if base in ('bicycle', 'motorcycle'):
                    tg[q[:, 2] > 0.0] = CityObjectLabel.Rider
                tag[sel] = tg
            elif isinstance(a, Walker):
                tag[sel] = CityObjectLabel.Pedestrians
            elif isinstance(a, Prop):
                tag[sel] = a.tag
        return tag, ids, cos

    def cast_ray(self, start, end):
        o = np.array([start.x, start.y, start.z])
        e = np.array([end.x, end.y, end.z])
        L = float(np.linalg.norm(e - o))
        d = (e - o) / max(L, 1e-9)
        t, _ = _cast(o[None], d[None], self._boxes(), L)
        if not np.isfinite(t[0]):
            return []
        p = o + d * t[0]
        return [LabelledPoint(Location(*p), 14)]

    # ---- sensors
    def _footprint(self, a):
        """The actor's box footprint corners (4, 2) in the world."""
        M = a.world_matrix()
        e, c = a.bounding_box.extent, a.bounding_box.location
        pts = np.array([[c.x + sx * e.x, c.y + sy * e.y, 0.0, 1.0] for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        return (M @ pts.T).T[:, :2]

    def _overlap(self, A, B):
        for P in (A, B):
            for k in range(4):
                edge = P[(k + 1) % 4] - P[k]
                n = np.array([-edge[1], edge[0]])
                pa, pb = A @ n, B @ n
                if pa.max() < pb.min() or pb.max() < pa.min():
                    return False
        return True

    def _collision(self, s, f, t, tr):
        """sensor.other.collision: an event when its parent's box starts to overlap another vehicle's or walker's
        (one per step, as CARLA sends one per contact)."""
        par = s.parent
        if par is None:
            return None
        A = self._footprint(par)
        touching = set()
        for a in self._actors.values():
            if a is par or not isinstance(a, (Vehicle, Walker)):
                continue
            if self._overlap(A, self._footprint(a)):
                touching.add(a.id)
        new = sorted(touching - getattr(s, '_touching', set()))
        s._touching = touching
        if not new:
            return None
        other = self._actors[new[0]]
        return CollisionEvent(f, t, tr, par, other, Vector3D(1000.0, 0.0, 0.0))

    def _measure(self, s, dt, boxes):
        M = s.world_matrix()
        tr = _T(M)
        f, t = self._frame, self._t
        kind = s.type_id
        if kind == 'sensor.other.collision':
            return self._collision(s, f, t, tr)
        if kind == 'sensor.camera.optical_flow':
            return self._flow_camera(s, M, boxes, f, t, tr)
        if kind == 'sensor.lidar.ray_cast':
            return self._lidar(s, M, dt, boxes, f, t, tr)
        if kind == 'sensor.lidar.ray_cast_semantic':
            return self._lidar(s, M, dt, boxes, f, t, tr, semantic=True)
        if kind in ('sensor.camera.semantic_segmentation', 'sensor.camera.instance_segmentation',
                    'sensor.camera.depth'):
            return self._gt_camera(s, M, boxes, f, t, tr)
        if kind == 'sensor.other.radar':
            return self._radar(s, M, dt, boxes, f, t, tr)
        if kind == 'sensor.camera.rgb':
            return self._camera(s, M, boxes, f, t, tr)
        if kind == 'sensor.other.imu':
            p = M[:3, 3].copy()
            y0, y1 = s._prev
            if y0 is None or y1 is None:
                acc = np.array([1e5, -4e4, -950.0]) if y1 is not None else np.zeros(3)
            else:
                acc = (p - 2 * y1 + y0) / (dt * dt)
            s._prev = [y1, p]
            acc = acc + np.array([0.0, 0.0, _GRAVITY])
            a_s = M[:3, :3].T @ acc
            v = s.parent
            gz = getattr(v, '_yaw_rate', 0.0)
            yaw = math.degrees(math.atan2(M[1, 0], M[0, 0]))
            return IMUMeasurement(f, t, tr, Vector3D(*a_s), Vector3D(0.0, 0.0, gz), math.radians((90.0 + yaw) % 360.0))
        if kind == 'sensor.other.gnss':
            x, y, z = M[:3, 3]
            rng = s._rng
            lat = math.degrees(-y / EARTH_R) + rng.normal(0, s._bp._f('noise_lat_stddev')) if s._bp._f(
                'noise_lat_stddev') > 0 else math.degrees(-y / EARTH_R)
            lon = math.degrees(x / EARTH_R) + (rng.normal(0, s._bp._f('noise_lon_stddev'))
                                               if s._bp._f('noise_lon_stddev') > 0 else 0.0)
            return GnssMeasurement(f, t, tr, lat, lon, float(z))
        return None

    def _lidar(self, s, M, dt, boxes, f, t, tr, semantic=False):
        bp = s._bp
        ch = max(int(bp._f('channels', 32)), 1)
        up, lo = bp._f('upper_fov', 10.0), bp._f('lower_fov', -30.0)
        hfov = bp._f('horizontal_fov', 360.0)
        pts_one = int(math.floor(bp._f('points_per_second', 56000) * dt / ch + 0.5))
        rot = bp._f('rotation_frequency', 10.0)
        rng_max = bp._f('range', 10.0)
        tick_ang = rot * hfov * dt
        step = tick_ang / max(pts_one, 1)
        elev = np.array([up - i * ((up - lo) / (ch - 1) if ch > 1 else 0.0) for i in range(ch)])
        az = np.fmod(s._angle + step * np.arange(pts_one), hfov) - hfov / 2
        s._angle = math.fmod(s._angle + tick_ang, hfov)
        E, A = np.meshgrid(np.radians(elev), np.radians(az), indexing='ij')
        d_loc = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        d_w = d_loc @ M[:3, :3].T
        o = M[:3, 3]
        tt, hid = _cast(np.repeat(o[None], len(d_w), 0), d_w, boxes, rng_max)
        hit = np.isfinite(tt)
        if semantic:                                   # the same rays, exact, with each hit's labels
            tag, ids, cos = self._labels(o, d_w, tt, hid, boxes)
            ids = np.where(ids >= LEVEL_ID0, 0, ids)   # level meshes: object_idx 0, as CARLA (the light: its id)
            P = (d_loc * np.where(hit, tt, 0.0)[:, None]).reshape(ch, pts_one, 3)
            hit2 = hit.reshape(ch, pts_one)
            rows = []
            for i in range(ch):
                h_ = hit2[i]
                r = np.zeros(int(h_.sum()), [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'), ('idx', '<u4'),
                                             ('tag', '<u4')])
                r['x'], r['y'], r['z'] = P[i][h_, 0], P[i][h_, 1], P[i][h_, 2]
                sl = slice(i * pts_one, (i + 1) * pts_one)
                r['cos'], r['idx'], r['tag'] = cos[sl][h_], ids[sl][h_], tag[sl][h_]
                rows.append(r)
            raw = np.concatenate(rows) if rows else np.zeros(0, rows[0].dtype)
            return SemanticLidarMeasurement(f, t, tr, ch, hit2.sum(1), raw.tobytes(), math.radians(s._angle))
        noise = bp._f('noise_stddev', 0.0)
        if noise > 0:
            tt = tt + s._rng.normal(0.0, noise, len(tt))
        tt = np.where(hit, tt, 0.0)
        P = d_loc * tt[:, None]
        inten = np.exp(-bp._f('atmosphere_attenuation_rate', 0.004) * tt)
        P, inten = P.reshape(ch, pts_one, 3), inten.reshape(ch, pts_one)
        hit = hit.reshape(ch, pts_one)
        counts = hit.sum(1)
        raw = np.concatenate([np.c_[P[i][hit[i]], inten[i][hit[i]]] for i in range(ch)]).astype(np.float32)
        return LidarMeasurement(f, t, tr, ch, counts, raw.tobytes(), math.radians(s._angle))

    def _radar(self, s, M, dt, boxes, f, t, tr):
        bp = s._bp
        n = int(bp._f('points_per_second', 1500) * dt)
        R = bp._f('range', 100.0)
        mx = math.tan(math.radians(bp._f('horizontal_fov', 30.0) / 2)) * R
        my = math.tan(math.radians(bp._f('vertical_fov', 30.0) / 2)) * R
        rad = s._rng.uniform(0, 1, n)
        ang = s._rng.uniform(0, 2 * math.pi, n)
        end = np.stack([np.full(n, R), mx * rad * np.cos(ang), my * rad * np.sin(ang)], -1)
        d_loc = end / np.linalg.norm(end, axis=1, keepdims=True)
        d_w = d_loc @ M[:3, :3].T
        o = M[:3, 3]
        prev = s._prev[1] if s._prev[1] is not None else np.zeros(3)
        v_r = (o - prev) / dt
        s._prev = [None, o.copy()]
        tt, hid = _cast(np.repeat(o[None], n, 0), d_w, boxes, R)
        vel = {a.id: getattr(a, '_vel', np.zeros(3)) for a in self._actors.values() if isinstance(a, Vehicle)}
        dets = []
        for k in np.nonzero(np.isfinite(tt))[0]:
            vt = vel.get(int(hid[k]), np.zeros(3)) if hid[k] >= 0 else np.zeros(3)
            az = math.atan2(d_loc[k, 1], d_loc[k, 0])
            alt = math.atan2(d_loc[k, 2], math.hypot(d_loc[k, 0], d_loc[k, 1]))
            dets.append(RadarDetection(float(tt[k]), az, alt, float((vt - v_r) @ d_w[k])))
        return RadarMeasurement(f, t, tr, dets)

    def _camera(self, s, M, boxes, f, t, tr):
        bp = s._bp
        W, H = int(bp._f('image_size_x', 800)), int(bp._f('image_size_y', 600))
        fov = bp._f('fov', 90.0)
        fp = W / (2.0 * math.tan(math.radians(fov) / 2))
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        # pixel centre (u, v) -> the ray in the camera's axes: x forward, y right (image right), z up (image up)
        d_loc = np.stack([np.full_like(u, fp), u - (W / 2 - 0.5), -(v - (H / 2 - 0.5))], -1).reshape(-1, 3)
        d_loc /= np.linalg.norm(d_loc, axis=1, keepdims=True)
        img = render(M[:3, 3], d_loc @ M[:3, :3].T, boxes, self._weather).reshape(H, W, 3)
        bgra = np.concatenate([img[..., ::-1], np.full((H, W, 1), 255, np.uint8)], -1)
        return Image(f, t, tr, W, H, fov, bgra.tobytes())

    def _flow_camera(self, s, M, boxes, f, t, tr):
        """sensor.camera.optical_flow: each pixel's motion from the previous frame, as CARLA 0.9.16 encodes it (two
        float32 per pixel, normalised: x = du / (W / 2), y = −dv / (H / 2); measured). The fake moves the world points
        a pixel sees now into the camera's previous pose — static-world flow: a moving actor's own motion is left out.
        Zero on the camera's first frame."""
        bp = s._bp
        W, H = int(bp._f('image_size_x', 800)), int(bp._f('image_size_y', 600))
        fov = bp._f('fov', 90.0)
        fp = W / (2.0 * math.tan(math.radians(fov) / 2))
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        d_loc = np.stack([np.full_like(u, fp), u - (W / 2 - 0.5), -(v - (H / 2 - 0.5))], -1).reshape(-1, 3)
        d_loc /= np.linalg.norm(d_loc, axis=1, keepdims=True)
        D = d_loc @ M[:3, :3].T
        o = M[:3, 3]
        out = np.zeros((H, W, 2), np.float32)
        prev = getattr(s, '_prev_M', None)
        s._prev_M = M.copy()
        if prev is not None:
            tt, _ = _cast(np.repeat(o[None], len(D), 0), D, boxes, 1000.0)
            ok = np.isfinite(tt)
            P = o + D * np.where(ok, tt, 0.0)[:, None]
            q = (P - prev[:3, 3]) @ prev[:3, :3]                 # in the previous camera's axes
            front = ok & (q[:, 0] > 1e-3)
            up = np.where(front, fp * q[:, 1] / np.where(front, q[:, 0], 1.0) + (W / 2 - 0.5), u.ravel())
            vp = np.where(front, -fp * q[:, 2] / np.where(front, q[:, 0], 1.0) + (H / 2 - 0.5), v.ravel())
            du, dv = u.ravel() - up, v.ravel() - vp
            out[..., 0] = (du / (W / 2.0)).reshape(H, W)
            out[..., 1] = (-dv / (H / 2.0)).reshape(H, W)
        return Image(f, t, tr, W, H, fov, out.tobytes())

    def _gt_camera(self, s, M, boxes, f, t, tr):
        """A semantic, instance or depth image: one ray cast per view and step, shared by the views' cameras at the
        same pose (as the RGB camera's, pixel for pixel)."""
        bp = s._bp
        W, H = int(bp._f('image_size_x', 800)), int(bp._f('image_size_y', 600))
        fov = bp._f('fov', 90.0)
        key = (f, tuple(np.round(M, 9).ravel()), W, H, fov)
        hit = self._cast_cache.get(key)
        if hit is None:
            fp = W / (2.0 * math.tan(math.radians(fov) / 2))
            u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
            d_loc = np.stack([np.full_like(u, fp), u - (W / 2 - 0.5), -(v - (H / 2 - 0.5))], -1).reshape(-1, 3)
            d_loc /= np.linalg.norm(d_loc, axis=1, keepdims=True)
            D = d_loc @ M[:3, :3].T
            o = M[:3, 3]
            tt, hid = _cast(np.repeat(o[None], len(D), 0), D, boxes, 1000.0)
            tag, ids, _ = self._labels(o, D, tt, hid, boxes)
            depth = np.where(np.isfinite(tt), tt * d_loc[:, 0], 1000.0)
            hit = self._cast_cache[key] = (tag.reshape(H, W), ids.reshape(H, W), depth.reshape(H, W))
        tag, ids, depth = hit
        kind = s.type_id.rsplit('.', 1)[1]
        out = np.zeros((H, W, 4), np.uint8)
        out[..., 3] = 255
        if kind == 'semantic_segmentation':
            out[..., 2] = tag
        elif kind == 'instance_segmentation':
            out[..., 2] = tag
            out[..., 1] = ids & 255
            out[..., 0] = (ids >> 8) & 255
        else:
            n = np.clip(np.rint(depth / 1000.0 * 16777215.0), 0, 16777215).astype(np.int64)
            out[..., 2], out[..., 1], out[..., 0] = n & 255, (n >> 8) & 255, (n >> 16) & 255
        return Image(f, t, tr, W, H, fov, out.tobytes())


def _ground_tag(y):
    y = np.asarray(y, float)
    return np.where((y >= ROAD_Y[0]) & (y <= ROAD_Y[1]), CityObjectLabel.Roads, CityObjectLabel.Sidewalks).astype(np.uint32)


def _cast(O, D, boxes, far):
    """Nearest hit distance along each ray (inf for none within far) and the box hit (actor id, −1 ground)."""
    n = len(D)
    best = np.full(n, np.inf)
    hid = np.full(n, -1, np.int64)
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(D[:, 2] < -1e-9, -O[:, 2] / D[:, 2], np.inf)
    tg = np.where(tg > 1e-6, tg, np.inf)
    best = np.minimum(best, tg)
    for aid, Mb, ext in boxes:
        Rb, cb = Mb[:3, :3], Mb[:3, 3]
        o = (O - cb) @ Rb
        d = D @ Rb
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (-ext - o) / d
            t2 = (ext - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        inside = np.all(np.abs(o) < ext, axis=1)
        t = np.where(inside, np.where(tmax > 1e-6, tmax, np.inf), np.where((tmax >= tmin) & (tmin > 1e-6), tmin, np.inf))
        better = t < best
        best = np.where(better, t, best)
        hid = np.where(better, aid, hid)
    best = np.where(best <= far, best, np.inf)
    return best, hid


FOG_RENDER_MIN = 10.0
NIGHT_LIGHT = 0.15
AIRLIGHT = np.array([190.0, 192.0, 196.0])


def render(o, D, boxes, weather=None) -> np.ndarray:
    """RGB uint8 of world rays (K, 3) from o: sky by elevation, a 2 m ground checker, boxes shaded by their id and the
    face they hit; with a weather, its fog and night (module docstring)."""
    t, hid = _cast(np.repeat(np.asarray(o, float)[None], len(D), 0), D, boxes, 500.0)
    out = _shade(o, D, boxes, t, hid)
    fog = float(getattr(weather, 'fog_density', 0.0) or 0.0) if weather is not None else 0.0
    if fog >= FOG_RENDER_MIN:
        tr = np.exp(-3.0 * np.where(np.isfinite(t), t, 1e9) / (7000.0 / fog))[:, None]
        out = np.clip(out * tr + AIRLIGHT * (1.0 - tr), 0, 255).astype(np.uint8)
    if weather is not None and float(getattr(weather, 'sun_altitude_angle', 45.0)) < 0.0:
        out = (out * NIGHT_LIGHT).astype(np.uint8)
    return out


def _shade(o, D, boxes, t, hid) -> np.ndarray:
    out = np.zeros((len(D), 3), np.uint8)
    sky = ~np.isfinite(t)
    out[sky] = (np.c_[110 + 80 * np.clip(D[sky, 2], 0, 1), 160 + 60 * np.clip(D[sky, 2], 0, 1),
                      np.full(sky.sum(), 230)]).astype(np.uint8)
    g = np.isfinite(t) & (hid < 0)
    P = o + D[g] * t[g, None]
    chk = ((np.floor(P[:, 0] / 2.0) + np.floor(P[:, 1] / 2.0)) % 2 == 0)
    out[g] = np.where(chk[:, None], np.array([150, 150, 145]), np.array([70, 72, 75])).astype(np.uint8)
    b = np.isfinite(t) & (hid >= 0)
    if b.any():
        P = o + D[b] * t[b, None]
        shade = np.zeros(b.sum())
        for aid, Mb, ext in boxes:
            sel = hid[b] == aid
            if not sel.any():
                continue
            q = (P[sel] - Mb[:3, 3]) @ Mb[:3, :3] / ext
            face = np.argmax(np.abs(q), axis=1)
            shade[sel] = 0.55 + 0.15 * face
        base = np.array([[200, 40, 40], [40, 160, 60], [40, 70, 200], [200, 170, 30]])[hid[b] % 4]
        out[b] = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------------------------------- the client
class OpendriveGenerationParameters:
    def __init__(self, vertex_distance=2.0, max_road_length=50.0, wall_height=1.0, additional_width=0.6,
                 smooth_junctions=True, enable_mesh_visibility=True, enable_pedestrian_navigation=True):
        self.vertex_distance, self.max_road_length, self.wall_height = vertex_distance, max_road_length, wall_height
        self.additional_width, self.smooth_junctions = additional_width, smooth_junctions
        self.enable_mesh_visibility = enable_mesh_visibility
        self.enable_pedestrian_navigation = enable_pedestrian_navigation


class _Response:
    def __init__(self, actor_id=0, error=''):
        self.actor_id, self.error = actor_id, error

    def has_error(self):
        return bool(self.error)


class command:                       # carla.command (a submodule in the real wheel)
    class DestroyActor:
        def __init__(self, actor):
            self.actor_id = actor if isinstance(actor, int) else actor.id

    class SetAutopilot:
        def __init__(self, actor, enabled, tm_port=8000):
            self.actor_id = actor if isinstance(actor, int) else actor.id
            self.enabled, self.tm_port = bool(enabled), tm_port

    class ApplyTransform:
        def __init__(self, actor, transform):
            self.actor_id = actor if isinstance(actor, int) else actor.id
            self.transform = transform

    class SpawnActor:                  # the blueprint as it is now (the real command keeps a description of it)
        def __init__(self, blueprint, transform, parent=None):
            self.blueprint = ActorBlueprint(blueprint.id, dict(blueprint._a), blueprint.tags)
            self.transform, self.parent, self.actor_id = transform, parent, 0


class Client:
    def __init__(self, host='127.0.0.1', port=2000, worker_threads=0):
        self.host, self.port = host, port
        self._timeout = 5.0
        LOG.append(('connect', host, port))
        w = STATE['world']
        if w is not None and w._settings.synchronous_mode:
            w._unseen = True

    def set_timeout(self, s):
        self._timeout = s

    def get_world(self):
        if STATE['world'] is None:
            STATE['world'] = World()
        return STATE['world']

    def generate_opendrive_world(self, opendrive, parameters=None, reset_settings=True):
        """An OpenDRIVE world. The fake keeps its own straight road (it does not build meshes from the file) but the
        map remembers the file, as Map.to_opendrive() returns it on CARLA."""
        STATE['world'] = World('OpenDriveMap')
        STATE['world']._map._xodr = opendrive
        LOG.append(('generate_opendrive_world', len(opendrive)))
        return STATE['world']

    def reload_world(self, reset_settings=True):
        name = STATE['world']._map.name.rpartition('/')[2] if STATE['world'] is not None else 'Town10HD_Opt'
        return self.load_world(name, reset_settings)

    def load_world(self, name, reset_settings=True):
        STATE['world'] = World(name)
        LOG.append(('load_world', name))
        return STATE['world']

    def get_available_maps(self):
        return [f'/Game/Carla/Maps/{t}' for t in ('Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD',
                                                   'Town10HD_Opt')]

    def get_trafficmanager(self, port=8000):
        tm = TrafficManager(port)
        STATE['tm'] = tm
        return tm

    def get_server_version(self):
        if os.environ.get('FAKE_CARLA_NO_SERVER'):           # nothing listens (tests): CARLA's client times out
            raise RuntimeError(f'time-out of {int(self._timeout * 1000)}ms while waiting for the simulator, make sure '
                               f'the simulator is ready and connected to {self.host}:{self.port}')
        return STATE['server_version']

    def apply_batch_sync(self, commands, do_tick=False):
        w = self.get_world()
        w._settle()
        out = []
        for c in commands:
            if isinstance(c, command.SpawnActor):
                try:
                    out.append(_Response(w.spawn_actor(c.blueprint, c.transform, c.parent).id))
                except RuntimeError as e:
                    out.append(_Response(0, str(e)))
                continue
            a = w._actors.get(c.actor_id)
            if a is None:
                out.append(_Response(c.actor_id, f'actor {c.actor_id} not found'))
                continue
            if isinstance(c, command.DestroyActor):
                a._batch = True
                a.destroy()
            elif isinstance(c, command.SetAutopilot):
                a.set_autopilot(c.enabled, c.tm_port)
            elif isinstance(c, command.ApplyTransform):
                a._apply_tf(c.transform)                 # a synchronous batch: applied before the call returns
            out.append(_Response(c.actor_id))
        if do_tick:
            w.tick()
        return out

    def apply_batch(self, commands):
        self.apply_batch_sync(commands)

    def get_client_version(self):
        return __version__


# ---------------------------------------------------------------------------------------------------- OSM conversion
class Osm2OdrSettings:
    def __init__(self):
        self.proj_string = '+proj=tmerc'
        self.center_map = True
        self.use_offsets = False
        self.offset_x = self.offset_y = 0.0
        self.default_lane_width = 3.35
        self.elevation_layer_height = 0.0
        self.generate_traffic_lights = True
        self.all_junctions_with_traffic_lights = False
        self._way_types = None

    def set_osm_way_types(self, types):
        self._way_types = list(types)

    def set_traffic_light_excluded_way_types(self, types):
        pass


class Osm2Odr:
    """A stand-in for SUMO's netconvert: every converted highway way becomes one road per travel direction along its
    nodes (line geometries), right lanes only, the lane count from its OSM tags (default 1), projected with the
    settings' tmerc centre (an equirectangular approximation: metres, not PROJ-exact), no junctions. Like the real
    converter it writes no rule (right-hand) and the street's name."""

    @staticmethod
    def convert(osm_text, settings):
        import re
        import xml.etree.ElementTree as ET
        from xml.sax.saxutils import quoteattr
        m0 = re.search(r'lat_0=([-\d.]+)', settings.proj_string)
        m1 = re.search(r'lon_0=([-\d.]+)', settings.proj_string)
        lat0, lon0 = (float(m0.group(1)), float(m1.group(1))) if m0 and m1 else (0.0, 0.0)
        root = ET.fromstring(osm_text)
        nodes = {n.get('id'): (float(n.get('lat')), float(n.get('lon'))) for n in root.iter('node')}
        kx, ky = 111319.49 * math.cos(math.radians(lat0)), 110574.0
        types = settings._way_types or ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
                                        'residential', 'motorway_link', 'trunk_link', 'primary_link',
                                        'secondary_link', 'tertiary_link']
        roads, rid = [], 1
        for w in root.iter('way'):
            tags = {t.get('k'): t.get('v') for t in w.iter('tag')}
            if tags.get('highway') not in types:
                continue
            pts = [nodes[nd.get('ref')] for nd in w.iter('nd') if nd.get('ref') in nodes]
            xy = [((lon - lon0) * kx, (lat - lat0) * ky) for lat, lon in pts]
            if len(xy) < 2:
                continue
            ow = tags.get('oneway') in ('yes', '1', 'true') or tags.get('junction') == 'roundabout'
            rev = tags.get('oneway') == '-1'
            total = int(tags['lanes']) if tags.get('lanes', '').isdigit() else None
            fwd = int(tags.get('lanes:forward', 0) or 0) or (total if ow or rev else (total // 2 if total else 1))
            bwd = int(tags.get('lanes:backward', 0) or 0) or (total // 2 if total and not (ow or rev) else 1)
            dirs = [(xy[::-1] if rev else xy, max(fwd, 1))] + ([] if ow or rev else [(xy[::-1], max(bwd, 1))])
            for line, n in dirs:
                geo, s = [], 0.0
                for (x0, y0), (x1, y1) in zip(line, line[1:]):
                    L = math.hypot(x1 - x0, y1 - y0)
                    if L < 1e-3:
                        continue
                    geo.append(f'<geometry s="{s:.4f}" x="{x0:.4f}" y="{y0:.4f}" hdg="{math.atan2(y1 - y0, x1 - x0):.6f}" '
                               f'length="{L:.4f}"><line/></geometry>')
                    s += L
                if not geo:
                    continue
                past = os.environ.get('FAKE_CARLA_OBJECT_PAST_END_M')   # as SUMO does now and then (tests)
                objs = (f'<objects><object id=":x{rid}_c0" type="crosswalk" s="{s + float(past):.8f}" t="0" '
                        f'hdg="0"/></objects>') if past and rid == 1 else ''
                lanes = ''.join(f'<lane id="-{k}" type="driving" level="true"><link/><width sOffset="0" '
                                f'a="{settings.default_lane_width}" b="0" c="0" d="0"/></lane>' for k in range(1, n + 1))
                roads.append(f'<road name={quoteattr(tags.get("name", ""))} length="{s:.4f}" id="{rid}" '
                             f'junction="-1"><link/><planView>{"".join(geo)}</planView><lanes><laneSection s="0">'
                             f'<center><lane id="0" type="none" level="true"><link/></lane></center>'
                             f'<right>{lanes}</right></laneSection></lanes>{objs}</road>')
                rid += 1
        return ('<?xml version="1.0" encoding="UTF-8"?>\n<OpenDRIVE><header revMajor="1" revMinor="4" name="" '
                f'version="1.00"><geoReference><![CDATA[{settings.proj_string}]]></geoReference></header>'
                + ''.join(roads) + '</OpenDRIVE>')
