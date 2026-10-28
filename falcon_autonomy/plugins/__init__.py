"""Plug-ins: small, documented interfaces for models and AV stacks (docs/plugins.md).

A plug-in is a Python class named by "module:Class" (importable from the worker's Python). This build defines the
driving-policy interface; detectors, segmenters, BEV models, trackers and AV-stack connections follow with their
milestones.

Policy (runs inside a live or drive worker, in CARLA's Python — no Qt, no carla needed by the policy itself):

    class MyPolicy:
        name = 'my policy'
        def __init__(self, **params): ...
        def reset(self, seed: int): ...                 # a new episode
        def act(self, obs: dict) -> dict: ...           # every simulation step

    obs  = {'t': sim seconds, 'dt': step seconds,
            'ego': {'x', 'y', 'z', 'yaw' (°, CARLA's frame), 'speed_mps', 'length', 'width'},
            'route': [[x, y, speed_limit_mps], ...]    # lane centre ahead every 2 m (the worker follows the lane,
                                                        # straight-most through junctions)
            'lead': {'distance_m', 'speed_mps'} | None, # the nearest vehicle ahead in the route's corridor
            'images': {sensor_id: HxWx3 uint8 RGB}}     # only when the policy asks for them (needs_images)
    act -> {'throttle': 0..1, 'steer': -1..1, 'brake': 0..1}

CARLA's frame is left-handed: x forward, y right, yaw clockwise seen from above; a positive steer turns right.
"""
import importlib


def load(spec: str, **params):
    """An instance of the class "module:Class"."""
    mod, _, cls = spec.partition(':')
    if not cls:
        raise ValueError(f'a plug-in is named "module:Class", not {spec!r}')
    return getattr(importlib.import_module(mod), cls)(**params)


BUILTIN_POLICIES = {'lane_follower': 'falcon_autonomy.plugins.lane_follower:LaneFollower'}
