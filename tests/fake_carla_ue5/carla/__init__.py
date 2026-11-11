"""A fake `carla` module of CARLA's UE5 line (0.10.x) for tests without a simulator: the 0.9.16 fake (tests/fake_carla)
run in this module's namespace, then changed where Falcon's backend adapter (falcon_autonomy.workers.compat) has to
cope. Put tests/fake_carla_ue5 on sys.path (before any real carla) and `import carla` gets this.

What differs from the 0.9.16 fake — chosen to exercise the adapter, not a statement of what a real 0.10 server does
(none of it was checked against one):

* version 0.10.0 (client and server);
* blueprints renamed: vehicle.mini.cooper, vehicle.ford.crown, vehicle.nissan.patrol, vehicle.dodge.charger,
  vehicle.lincoln.mkz instead of the 0.9.16 fake's audi.a2, citroen.c3, nissan.micra, tesla.model3, lincoln.mkz_2020;
  no mini.cooper_s, micro.microlino or bh.crossbike (an alias, then a blueprint of the same base type, stand in);
* carla.CityObjectLabel: the same 29 names with their values shuffled (SHUFFLE, a fixed permutation); every tag the
  fake's sensors write follows it;
* no carla.Osm2Odr / Osm2OdrSettings, no TrafficManager.set_osm_mode, no Sensor.enable_for_ros;
* WeatherParameters without dust_storm;
* camera blueprints without the lens_* attributes, and set_attribute refusing a lens_* attribute it does not have;
* two maps, Town10HD_Opt and Mine_01; load_world refuses any other.

root/ holds a stand-in CARLA folder of this line (root/CarlaUnreal/Binaries/Linux/CarlaUnreal-Linux-Shipping, a stub
that exits 1, and VERSION) for the folder detection; it never runs a server.
"""
import os as _os
import random as _random

_BASE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'fake_carla',
                      'carla', '__init__.py')
_DOC = __doc__
with open(_BASE, encoding='utf-8') as _fh:
    exec(compile(_fh.read(), _BASE, 'exec'), globals())          # noqa: S102 (the 0.9.16 fake, in this namespace)
__doc__ = _DOC

__version__ = '0.10.0'
STATE['server_version'] = '0.10.0'                                  # noqa: F821 (defined by the 0.9.16 fake)
MAPS = ('Town10HD_Opt', 'Mine_01')

# ---------------------------------------------------------------------------------------------------- semantic tags
_NAMES = ('NONE', 'Roads', 'Sidewalks', 'Buildings', 'Walls', 'Fences', 'Poles', 'TrafficLight', 'TrafficSigns',
          'Vegetation', 'Terrain', 'Sky', 'Pedestrians', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle',
          'Bicycle', 'Static', 'Dynamic', 'Other', 'Water', 'RoadLines', 'Ground', 'Bridge', 'RailTrack', 'GuardRail')
_values = list(range(len(_NAMES)))
_random.Random(1005).shuffle(_values)
SHUFFLE = dict(zip(range(len(_NAMES)), _values))                   # 0.9.16 value -> this fake's value


class CityObjectLabel:
    Any = 255


for _n, _v in zip(_NAMES, _values):
    setattr(CityObjectLabel, _n, _v)

FIXED = tuple((c, i, SHUFFLE[t], ctr, ext) for c, i, t, ctr, ext in FIXED)   # noqa: F821
_prop_init = Prop.__init__                                                    # noqa: F821


def _prop_init_ue5(self, *a, **k):
    _prop_init(self, *a, **k)
    self.tag = SHUFFLE.get(int(self.tag), self.tag)


Prop.__init__ = _prop_init_ue5                                                # noqa: F821

# ---------------------------------------------------------------------------------------------------- blueprints
RENAMED = {'vehicle.audi.a2': 'vehicle.mini.cooper', 'vehicle.citroen.c3': 'vehicle.ford.crown',
           'vehicle.nissan.micra': 'vehicle.nissan.patrol', 'vehicle.tesla.model3': 'vehicle.dodge.charger',
           'vehicle.lincoln.mkz_2020': 'vehicle.lincoln.mkz'}
REMOVED = ('vehicle.mini.cooper_s', 'vehicle.micro.microlino', 'vehicle.bh.crossbike')
VEHICLES = {RENAMED.get(k, k): v for k, v in VEHICLES.items()}                              # noqa: F821
NAMED_ONLY = {RENAMED.get(k, k): v for k, v in NAMED_ONLY.items() if k not in REMOVED}      # noqa: F821
VEHICLES_ALL = dict(VEHICLES, **NAMED_ONLY)
TWO_WHEELERS_NAMED = {}
for _k in [k for k in SENSOR_DEFAULTS if k.startswith('sensor.camera.')]:                  # noqa: F821
    SENSOR_DEFAULTS[_k] = {a: v for a, v in SENSOR_DEFAULTS[_k].items() if not a.startswith('lens_')}  # noqa: F821
_bp_set = ActorBlueprint.set_attribute                                                       # noqa: F821


def _set_attribute_ue5(self, k, v):
    if k.startswith('lens_') and k not in self._a:
        raise IndexError(f'{self.id} has no attribute {k}')
    return _bp_set(self, k, v)


ActorBlueprint.set_attribute = _set_attribute_ue5                                            # noqa: F821
_lib_init = BlueprintLibrary.__init__                                                        # noqa: F821


def _lib_init_ue5(self, bps=None):
    _lib_init(self, bps)
    if bps is None:                     # the named-only vehicles of this line are listed too
        have = {b.id for b in self._bps}
        self._bps += [ActorBlueprint(k, {'base_type': v[0], 'number_of_wheels': 4, 'role_name': 'autopilot',  # noqa
                                         'generation': 3}, ['vehicle', v[0]]) for k, v in NAMED_ONLY.items()
                      if k not in have]


BlueprintLibrary.__init__ = _lib_init_ue5                                                    # noqa: F821

# ---------------------------------------------------------------------------------------------------- APIs it lacks
del Osm2Odr, Osm2OdrSettings                                                                 # noqa: F821
del TrafficManager.set_osm_mode                                                              # noqa: F821
del Sensor.enable_for_ros                                                                    # noqa: F821

# ---------------------------------------------------------------------------------------------------- weather
_W0916 = WeatherParameters                                                                   # noqa: F821


class WeatherParameters:
    FIELDS = tuple(f for f in _W0916.FIELDS if f != 'dust_storm')

    def __init__(self, **kw):
        base = _W0916()
        for k in self.FIELDS:
            setattr(self, k, float(kw.get(k, getattr(base, k))))

    def __repr__(self):
        return 'WeatherParameters(' + ', '.join(f'{k}={getattr(self, k):.6f}' for k in self.FIELDS) + ')'


for _n, _v in _PRESETS.items():                                                              # noqa: F821
    setattr(WeatherParameters, _n, WeatherParameters(**dict(zip(_W0916.FIELDS, _v))))

# ---------------------------------------------------------------------------------------------------- maps
_load = Client.load_world                                                                    # noqa: F821


def _load_world_ue5(self, name, reset_settings=True):
    if str(name).rpartition('/')[2] not in MAPS:
        raise RuntimeError(f'map not found: {name}')
    return _load(self, name, reset_settings)


def _maps_ue5(self):
    return [f'/Game/Carla/Maps/{t}' for t in MAPS]


Client.load_world = _load_world_ue5                                                          # noqa: F821
Client.get_available_maps = _maps_ue5                                                        # noqa: F821
