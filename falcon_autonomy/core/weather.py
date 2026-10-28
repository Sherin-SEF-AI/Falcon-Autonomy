"""Weather: CARLA 0.9.16's presets and parameters, as plain data (standard library only).

A world's weather is {'preset': name} optionally overridden field by field (the project schema's "weather"). The
values of the presets are CARLA 0.9.16's (LibCarla/source/carla/rpc/WeatherParameters.cpp); the worker applies them
through carla.WeatherParameters, so these tables only drive the GUI's sliders and the fog-visibility figure.

Limits of CARLA's weather, stated wherever weather is chosen:
* CARLA's lidar and radar ignore weather (no loss of range or spurious returns in rain or fog).
* Rain on a camera is water on the lens and wet surfaces; it does not reduce lidar range.
* Fog: visibility ≈ 7000 m / fog_density (measured with CARLA 0.9.16; a 5 % contrast threshold, a human observer's —
  for a detector it is an upper bound).
"""
from __future__ import annotations

FIELDS = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity', 'sun_azimuth_angle',
          'sun_altitude_angle', 'fog_density', 'fog_distance', 'fog_falloff', 'wetness', 'scattering_intensity',
          'mie_scattering_scale', 'rayleigh_scattering_scale', 'dust_storm')

# name: the values of FIELDS in order, generated from carla.WeatherParameters of the 0.9.16 wheel
_P = {
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
}
PRESETS = tuple(_P)
DEFAULT_PRESET = 'ClearNoon'

# slider ranges for the GUI: field -> (lo, hi, unit, label)
SLIDERS = {
    'sun_altitude_angle': (-90.0, 90.0, '°', 'Sun altitude'),
    'sun_azimuth_angle': (0.0, 360.0, '°', 'Sun azimuth'),
    'cloudiness': (0.0, 100.0, '%', 'Cloud'),
    'precipitation': (0.0, 100.0, '%', 'Rain'),
    'wetness': (0.0, 100.0, '%', 'Wetness'),
    'precipitation_deposits': (0.0, 100.0, '%', 'Puddles'),
    'fog_density': (0.0, 100.0, '%', 'Fog density'),
    'wind_intensity': (0.0, 100.0, '%', 'Wind'),
}


def preset_values(name: str) -> dict:
    if name not in _P:
        raise KeyError(f'Unknown CARLA weather preset {name!r}')
    return dict(zip(FIELDS, (float(x) for x in _P[name])))


def resolve(w: dict | None) -> dict:
    """A world's weather as all of CARLA's fields: the preset's values with the overrides on top."""
    w = dict(w or {})
    d = preset_values(w.pop('preset', DEFAULT_PRESET) or DEFAULT_PRESET)
    for k, v in w.items():
        if k in FIELDS:
            d[k] = float(v)
    return d


def fog_visibility_m(fog_density: float) -> float | None:
    """Meteorological visibility for CARLA's fog density (measured relation, 5 % contrast; None without fog)."""
    if fog_density <= 0:
        return None
    return 7000.0 / fog_density


def describe(w: dict | None) -> str:
    d = resolve(w)
    sun = d['sun_altitude_angle']
    tod = 'night' if sun < -6 else 'twilight' if sun < 6 else 'low sun' if sun < 20 else 'day'
    bits = [tod]
    if d['precipitation'] > 0:
        bits.append(f'rain {d["precipitation"]:.0f} %')
    if d['cloudiness'] >= 50:
        bits.append('cloudy')
    vis = fog_visibility_m(d['fog_density'])
    if vis is not None and vis < 1000:
        bits.append(f'fog (visibility ≈ {vis:.0f} m)')
    return ', '.join(bits)
