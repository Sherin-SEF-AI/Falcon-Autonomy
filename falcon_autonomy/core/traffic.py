"""Traffic profiles: named, editable traffic (the project's traffic_profiles), and the Indian mixed-traffic preset.

A profile's fleet gives weights per CARLA base type (car, truck, van, bus, motorcycle, bicycle) or per blueprint
pattern ('blueprint:vehicle.micro.*'). Its behaviour sets the traffic manager per vehicle; speed_spread_pct draws each
vehicle's speed difference uniformly in ±spread (seeded). Walkers walk on CARLA's navigation mesh under its walker AI;
walker_crossing_factor is the share of them that cross roads anywhere, not only at crossings
(World.set_pedestrians_cross_factor).

Handedness: a profile may assert 'left' or 'right' traffic; a run is refused when the map's lanes say otherwise
(check_handedness) — Falcon never silently simulates right-hand traffic for a left-hand place.

Limits, stated wherever the Indian profile is chosen:
* CARLA 0.9.16 has no autorickshaw. The profile's stand-in is vehicle.micro.microlino, the vehicle nearest in measured
  size to a Bajaj RE (core.assets: 2.21 × 1.48 × 1.38 m against 2.64 × 1.30 × 1.70 m), labelled "autorickshaw
  (stand-in)" and given the class's limits ('fleet_classes'): the traffic manager's desired speed capped at the
  three-wheeler's top speed, re-read against each road's limit (workers.scene.ClassLimits), and its engine torque
  scaled to a three-wheeler's acceleration. docs/indian-assets.md says what was measured and how to add a real
  autorickshaw (a source build of CARLA).
* The fleet shares are a heuristic from registration statistics (two-wheelers are about two thirds of vehicles
  registered in Bengaluru), not a measured on-road count.
* CARLA's traffic manager keeps vehicles on their lane centres. A profile's 'lateral_filtering' (lateral_settings)
  gives its two-wheelers (and, when named, the autorickshaw stand-ins) a lateral position of their own —
  workers.lateral; docs/indian-assets.md has the measurements and its limits. The scenario studio's "two-wheeler
  filtering" scenario scripts one rider on the lane line explicitly.
"""
from __future__ import annotations

import copy

AUTORICKSHAW_SUBSTITUTE = 'vehicle.micro.microlino'

INDIA_MIXED = {
    'id': 'tp-india-mixed', 'name': 'Indian mixed traffic (Bengaluru)', 'handedness': 'left',
    'vehicles': 60, 'walkers': 40,
    'fleet': {'motorcycle': 0.45, 'bicycle': 0.04, 'car': 0.26, 'van': 0.03, 'bus': 0.06, 'truck': 0.04,
              f'blueprint:{AUTORICKSHAW_SUBSTITUTE}': 0.12},
    'fleet_labels': {f'blueprint:{AUTORICKSHAW_SUBSTITUTE}': 'autorickshaw (stand-in)'},
    'fleet_classes': {f'blueprint:{AUTORICKSHAW_SUBSTITUTE}': 'autorickshaw'},
    'fleet_source': 'heuristic: two-wheelers ≈ two thirds of vehicles registered in Bengaluru; not an on-road count',
    'behaviour': {'distance_to_leading_m': 1.0, 'speed_spread_pct': 25.0, 'speed_difference_pct': -10.0,
                  'ignore_lights_pct': 15.0, 'ignore_signs_pct': 25.0, 'ignore_walkers_pct': 0.0,
                  'auto_lane_change': True, 'random_left_lanechange_pct': 10.0, 'random_right_lanechange_pct': 10.0,
                  'walker_crossing_factor': 0.35},
    # on for two-wheelers: measured on central Bengaluru (docs/indian-assets.md, 2026-10-30) with no more contacts or
    # put-backs than without at the profile's 60 vehicles; autorickshaws are not in it (their contacts rose by half)
    'lateral_filtering': {'enabled': True, 'vehicles': ['motorcycle', 'bicycle'], 'controller': 'tm'},
}

PRESETS = {'tp-default': None, 'tp-india-mixed': INDIA_MIXED}

# lateral filtering (workers.lateral): a profile's 'lateral_filtering'; absent or enabled false: off
LATERAL_KEY = 'lateral_filtering'
LATERAL_KINDS = ('motorcycle', 'bicycle', 'autorickshaw')
LATERAL_CONTROLLERS = ('tm', 'own')
LATERAL_DEFAULTS = {'enabled': False, 'vehicles': ['motorcycle', 'bicycle'], 'controller': 'tm'}


def lateral_settings(prof: dict | None) -> dict | None:
    """The profile's lateral filtering filled with LATERAL_DEFAULTS, or None when it is off; ValueError when it
    names an unknown vehicle kind or controller."""
    s = (prof or {}).get(LATERAL_KEY)
    if not s or not s.get('enabled'):
        return None
    out = dict(LATERAL_DEFAULTS, **s)
    bad = [k for k in out['vehicles'] if k not in LATERAL_KINDS]
    if bad or not out['vehicles']:
        raise ValueError(f'lateral_filtering.vehicles: {bad or "none"} (choose from {", ".join(LATERAL_KINDS)})')
    if out['controller'] not in LATERAL_CONTROLLERS:
        raise ValueError(f'lateral_filtering.controller {out["controller"]!r} (choose from '
                         f'{", ".join(LATERAL_CONTROLLERS)})')
    return out


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
