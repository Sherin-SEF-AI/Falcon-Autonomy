"""Vehicle lights at night, and what the map's own lights do.

CARLA spawns every vehicle with its lights off, at night too, and nothing switches them on unless asked: a night drive
without this has dark cars with no headlights (measured on CARLA 0.9.16, docs/translation.md "At night"). A drive's
config may say `vehicle_lights`:

    'auto' (the default)  on when the drive's sun is below the horizon (weather sun_altitude_angle < 0), else left
                          as spawned (off)
    'on'                  position lights and low beams on every vehicle (ego and traffic), whatever the sun
    'tm'                  the traffic manager switches each vehicle's lights by the weather and time of day
                          (TrafficManager.update_vehicle_lights) — CARLA's own rule
    'off'                 left as spawned

The street and building lights belong to the map: CARLA's towns switch theirs on by themselves when the sun is down;
an imported OpenDRIVE world has none (it has no buildings or street furniture). report() counts them and how many are
on, and how many vehicles' lights actually read back as on, for run.json."""
from __future__ import annotations

MODES = ('auto', 'on', 'tm', 'off')


def mode_of(cfg: dict) -> str:
    """The configured mode, 'auto' resolved by the drive's sun altitude."""
    mode = cfg.get('vehicle_lights', 'auto') or 'auto'
    if mode not in MODES:
        raise ValueError(f'vehicle_lights is one of {", ".join(MODES)}, not {mode!r}')
    if mode == 'auto':
        sun = float((cfg.get('weather') or {}).get('sun_altitude_angle', 45.0))
        return 'on' if sun < 0.0 else 'off'
    return mode


def apply(w, cfg: dict, actors: list) -> dict:
    """Set the vehicles' lights by the config's mode; returns {'mode', 'resolved', 'vehicles'}."""
    from .common import Failed
    carla = w.carla
    try:
        mode = mode_of(cfg)
    except ValueError as e:
        raise Failed(str(e)) from None
    n = 0
    if mode == 'on':
        st = carla.VehicleLightState
        bits = int(st.Position) | int(st.LowBeam)
        try:
            state = carla.VehicleLightState(bits)          # as CARLA's manual_control.py builds it
        except TypeError:
            state = bits
        for a in actors:
            if a is not None and hasattr(a, 'set_light_state'):
                a.set_light_state(state)
                n += 1
    elif mode == 'tm':
        for a in actors:
            if a is not None:
                w.tm.update_vehicle_lights(a, True)
                n += 1
    return {'mode': cfg.get('vehicle_lights', 'auto') or 'auto', 'resolved': mode, 'vehicles': n}


def report(world, actors: list, applied: dict) -> dict:
    """What the lights are after the warm-up: vehicles whose light state reads back as not NONE, and the map's
    street and building lights (LightManager; absent on a server without it)."""
    out = dict(applied)
    lit = 0
    for a in actors:
        try:
            if a is not None and int(a.get_light_state()) != 0:
                lit += 1
        except (AttributeError, RuntimeError, TypeError):
            pass
    out['vehicles_lit'] = lit
    try:
        lights = list(world.get_lightmanager().get_all_lights())
        out['map_lights'] = len(lights)
        out['map_lights_on'] = sum(1 for L in lights if L.is_on)
    except (AttributeError, RuntimeError):
        out['map_lights'] = out['map_lights_on'] = None
    return out
