"""The token observations of the driving environment, shared by both sides (numpy only; importable from CARLA's Python
and the app's): what the environment worker (workers/gym.py) sends when a config asks for tokens, and what the
transformer policy (core.tokenpolicy) and the introspection tools (core.introspect, core.biasaudit) read.

Frames and units as core.drivefeatures: CARLA's (x forward, y right, z up, left-handed), metres, seconds, yaw in
degrees positive turning right. Object positions and velocities are given in the EGO's frame (forward, right).

A config's "tokens" block (core.gymenv.gym_config(tokens=True | {...})): max_objects (default MAX_OBJECTS), range_m
(RANGE_M), history (HISTORY_STEPS). With it the observation has three more arrays:

    objects      (max_objects, N_OBJ_FEATURES) float32   the nearest road users within range_m of the ego's centre,
                                                           nearest first (box centres), one row each (OBJ_FEATURES);
                                                           rows past the last are zero (present 0)
    object_ids   (max_objects,) int64                      their CARLA actor ids (−1 for an empty row)
    ego_history  (history,) float32                        the ego's speed at the last `history` steps, oldest
                                                           first, the current step last; before the episode's first
                                                           step the first speed repeats

An object row (OBJ_FEATURES): present; its class one-hot over OBJ_CLASSES; its box centre ahead (dx) and to the
right (dy) of the ego's centre; its heading relative to the ego's as (cos, sin); its velocity in the ego's frame
(vx forward, vy right; absolute, not relative to the ego); its box length and width; its brake-light bit (1: the
vehicle's Brake light is on — CARLA's VehicleLightState.Brake as the vehicle reports it; always 0 for pedestrians).

Classes (object_class): Falcon's subclass first (a stand-in's label: an "autorickshaw (stand-in)" is an
autorickshaw, a "cycle-rickshaw" a bicycle, a proxy obstacle "other"), then the nuScenes category of
core.groundtruth.category_of: human.* pedestrian, vehicle.bicycle bicycle, vehicle.motorcycle two_wheeler,
vehicle.truck truck, vehicle.bus.* bus, any other vehicle.* car, anything else other.

The route token and the ego token are read from the state vector (core.drivefeatures) by name: ROUTE_FEATURES and
EGO_FEATURES. Neither holds the lead or vulnerable-road-user summaries (lead_*, vru_*): a token policy must find the
other road users in its object tokens or its image — so what it relies on can be measured (core.biasaudit). The
previous action (prev_*) is left out as end-to-end cloning leaves it out (the copycat shortcut, core.imitation).
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA = 'falcon.tokenfeatures/1'
MAX_OBJECTS = 16
RANGE_M = 60.0
HISTORY_STEPS = 10
BRAKE_BIT = 8                      # carla.VehicleLightState.Brake

OBJ_CLASSES = ('car', 'truck', 'bus', 'two_wheeler', 'bicycle', 'autorickshaw', 'pedestrian', 'other')
OBJ_FEATURES = (('present', '', '1: a road user in this row'),) + tuple(
    (f'class_{c}', '', f'1: a {c}') for c in OBJ_CLASSES) + (
    ('dx_m', 'm', 'its box centre ahead of the ego\'s centre (ego frame)'),
    ('dy_m', 'm', 'its box centre to the ego\'s right (ego frame)'),
    ('cos_rel_yaw', '', 'cos(its yaw − the ego\'s)'),
    ('sin_rel_yaw', '', 'sin(its yaw − the ego\'s)'),
    ('vx_mps', 'm/s', 'its velocity, forward in the ego\'s frame'),
    ('vy_mps', 'm/s', 'its velocity, right in the ego\'s frame'),
    ('length_m', 'm', 'its box length'),
    ('width_m', 'm', 'its box width'),
    ('brake_light', '', '1: its brake light is on'),
)
OBJ_FEATURE_NAMES = tuple(f[0] for f in OBJ_FEATURES)
N_OBJ_FEATURES = len(OBJ_FEATURES)
OI = {n: k for k, n in enumerate(OBJ_FEATURE_NAMES)}

ROUTE_FEATURES = ('lateral_offset_m', 'heading_error_rad', 'lane_width_m', 'curvature_10m', 'curvature_25m',
                  'curvature_50m', 'route_remaining_m', 'tl_known', 'tl_red', 'tl_yellow', 'tl_green', 'tl_distance_m')
EGO_FEATURES = ('speed_mps', 'target_speed_mps', 'accel_mps2')
DEFAULTS = {'max_objects': MAX_OBJECTS, 'range_m': RANGE_M, 'history': HISTORY_STEPS}


def spec(d) -> dict | None:
    """A config's tokens block filled with DEFAULTS (True: the defaults; None or False: no tokens); ValueError when a
    value is out of range or unknown."""
    if d is None or d is False:
        return None
    d = {} if d is True else dict(d)
    unknown = sorted(set(d) - set(DEFAULTS))
    if unknown:
        raise ValueError(f'unknown tokens keys {unknown} (keys: {sorted(DEFAULTS)})')
    out = dict(DEFAULTS, **d)
    out['max_objects'], out['history'] = int(out['max_objects']), int(out['history'])
    out['range_m'] = float(out['range_m'])
    if not 1 <= out['max_objects'] <= 64 or not 1 <= out['history'] <= 200 or not 1.0 <= out['range_m'] <= 200.0:
        raise ValueError('tokens: max_objects 1–64, history 1–200 steps, range_m 1–200 m')
    return out


def object_class(row: dict) -> str:
    """The OBJ_CLASSES class of an actors.jsonl-style row (module doc)."""
    sub = str(row.get('subclass') or '').lower()
    if 'autorickshaw' in sub:
        return 'autorickshaw'
    if 'cycle' in sub and 'rickshaw' in sub:
        return 'bicycle'
    if sub:
        return 'other'
    cat = str(row.get('category') or '')
    if cat.startswith('human.'):
        return 'pedestrian'
    if cat == 'vehicle.bicycle':
        return 'bicycle'
    if cat == 'vehicle.motorcycle':
        return 'two_wheeler'
    if cat == 'vehicle.truck':
        return 'truck'
    if cat.startswith('vehicle.bus'):
        return 'bus'
    if cat.startswith('vehicle.'):
        return 'car'
    return 'other'


def object_tokens(ego: dict, rows, spec_: dict, ego_id=None, brake=None) -> tuple:
    """(objects (max_objects, N_OBJ_FEATURES) float32, object_ids (max_objects,) int64) of the scene (module doc).
    ego: {'x', 'y', 'yaw'}; rows: actors.jsonl-style rows {'id', 'x', 'y', 'yaw', 'vx', 'vy', 'extent',
    'category', 'subclass'?, 'brake_light'?}; brake: {actor id: bool} overriding the rows' brake_light."""
    n, rng = int(spec_['max_objects']), float(spec_['range_m'])
    out = np.zeros((n, N_OBJ_FEATURES), np.float64)
    ids = np.full(n, -1, np.int64)
    r = math.radians(float(ego['yaw']))
    c, s = math.cos(r), math.sin(r)
    cands = []
    for row in rows or ():
        if ego_id is not None and row.get('id') == ego_id:
            continue
        if ego_id is None and row.get('role') == 'hero':
            continue
        cls = object_class(row)
        if cls == 'other' and not row.get('subclass') and not str(row.get('category', '')).startswith(
                ('vehicle.', 'human.', 'movable_object')):
            continue
        dxw, dyw = float(row['x']) - float(ego['x']), float(row['y']) - float(ego['y'])
        d = math.hypot(dxw, dyw)
        if d > rng:
            continue
        cands.append((d, int(row.get('id', -1)), row, cls, dxw, dyw))
    cands.sort(key=lambda t: (t[0], t[1]))
    for k, (_d, aid, row, cls, dxw, dyw) in enumerate(cands[:n]):
        f = out[k]
        f[OI['present']] = 1.0
        f[OI[f'class_{cls}']] = 1.0
        f[OI['dx_m']], f[OI['dy_m']] = dxw * c + dyw * s, -dxw * s + dyw * c
        rel = math.radians(float(row.get('yaw', 0.0)) - float(ego['yaw']))
        f[OI['cos_rel_yaw']], f[OI['sin_rel_yaw']] = math.cos(rel), math.sin(rel)
        vx, vy = float(row.get('vx', 0.0)), float(row.get('vy', 0.0))
        f[OI['vx_mps']], f[OI['vy_mps']] = vx * c + vy * s, -vx * s + vy * c
        ext = row.get('extent') or (2.25, 0.95, 0.75)
        f[OI['length_m']], f[OI['width_m']] = 2.0 * float(ext[0]), 2.0 * float(ext[1])
        on = (brake or {}).get(aid, row.get('brake_light', False))
        f[OI['brake_light']] = float(bool(on)) if cls != 'pedestrian' else 0.0
        ids[k] = aid
    return out.astype(np.float32), ids


class SpeedHistory:
    """The ego's last `n` speeds, oldest first (module doc): reset() at an episode's start, push() at every step."""

    def __init__(self, n: int):
        self.n = int(n)
        self.v: list = []

    def reset(self):
        self.v = []

    def push(self, speed: float) -> np.ndarray:
        if not self.v:
            self.v = [float(speed)] * self.n
        else:
            self.v = (self.v + [float(speed)])[-self.n:]
        return self.array()

    def array(self) -> np.ndarray:
        return np.asarray(self.v or [0.0] * self.n, np.float32)


def brake_light_on(state) -> bool:
    """Whether a vehicle light state (carla.VehicleLightState or an int) has the Brake bit."""
    try:
        return bool(int(state) & BRAKE_BIT)
    except (TypeError, ValueError):
        return False


def with_brake(state, on: bool) -> int:
    """A light state's bits with the Brake bit set or cleared (the other lights kept)."""
    try:
        bits = int(state)
    except (TypeError, ValueError):
        bits = 0
    return (bits | BRAKE_BIT) if on else (bits & ~BRAKE_BIT)
