"""Traffic profiles: named, editable traffic (the project's traffic_profiles), and the Indian mixed-traffic preset.

A profile's fleet gives weights per CARLA base type (car, truck, van, bus, motorcycle, bicycle) or per blueprint
pattern ('blueprint:vehicle.micro.*'). Its behaviour sets the traffic manager per vehicle; speed_spread_pct draws each
vehicle's speed difference uniformly in ±spread (seeded). Walkers walk on CARLA's navigation mesh under its walker AI;
walker_crossing_factor is the share of them that cross roads anywhere, not only at crossings
(World.set_pedestrians_cross_factor).

Handedness: a profile may assert 'left' or 'right' traffic; a run is refused when the map's lanes say otherwise
(check_handedness) — Falcon never silently simulates right-hand traffic for a left-hand place.

Limits, stated wherever the Indian profile is chosen:
* CARLA 0.9.16 has no autorickshaw. The profile substitutes vehicle.micro.microlino (a small three-door bubble car,
  the nearest size) and labels it "autorickshaw (substitute: Microlino)"; docs/worlds.md says how to add a real
  autorickshaw blueprint (an Unreal asset cooked into the CARLA build).
* The fleet shares are a heuristic from registration statistics (two-wheelers are about two thirds of vehicles
  registered in Bengaluru), not a measured on-road count.
* CARLA's traffic manager keeps vehicles in lanes: lane-splitting by two-wheelers is not reproduced by the profile
  (the scenario studio's "two-wheeler filtering" scenario scripts it explicitly).
"""
from __future__ import annotations

import copy

AUTORICKSHAW_SUBSTITUTE = 'vehicle.micro.microlino'

INDIA_MIXED = {
    'id': 'tp-india-mixed', 'name': 'Indian mixed traffic (Bengaluru)', 'handedness': 'left',
    'vehicles': 60, 'walkers': 40,
    'fleet': {'motorcycle': 0.45, 'bicycle': 0.04, 'car': 0.26, 'van': 0.03, 'bus': 0.06, 'truck': 0.04,
              f'blueprint:{AUTORICKSHAW_SUBSTITUTE}': 0.12},
    'fleet_labels': {f'blueprint:{AUTORICKSHAW_SUBSTITUTE}': 'autorickshaw (substitute: Microlino)'},
    'fleet_source': 'heuristic: two-wheelers ≈ two thirds of vehicles registered in Bengaluru; not an on-road count',
    'behaviour': {'distance_to_leading_m': 1.0, 'speed_spread_pct': 25.0, 'speed_difference_pct': -10.0,
                  'ignore_lights_pct': 15.0, 'ignore_signs_pct': 25.0, 'ignore_walkers_pct': 0.0,
                  'auto_lane_change': True, 'random_left_lanechange_pct': 10.0, 'random_right_lanechange_pct': 10.0,
                  'walker_crossing_factor': 0.35},
}

PRESETS = {'tp-default': None, 'tp-india-mixed': INDIA_MIXED}


def preset(ident: str) -> dict:
    if ident == 'tp-default':
        from .project import default_traffic_profile
        return default_traffic_profile()
    return copy.deepcopy(PRESETS[ident])


def fleet_label(key: str, prof: dict) -> str:
    return (prof.get('fleet_labels') or {}).get(key, key.split(':', 1)[-1])


def check_handedness(profile: dict, map_rule: str) -> tuple[bool, str]:
    """(ok, why): a profile asserting left (right) traffic needs a map whose lanes are left- (right-) hand."""
    want = (profile or {}).get('handedness', 'map')
    if want == 'map':
        return True, f'the map\'s own ({map_rule})'
    have = {'LHT': 'left', 'RHT': 'right'}.get(map_rule, map_rule)
    if want == have:
        return True, f'{want}-hand traffic, as the map\'s lanes are'
    return False, (f'the traffic profile "{profile.get("name")}" is {want}-hand traffic but the map\'s lanes are '
                   f'{have}-hand ({map_rule}). CARLA\'s towns are right-hand; import the place with left-hand '
                   'traffic (Worlds ▸ import, "Left-hand traffic") or choose another profile.')
