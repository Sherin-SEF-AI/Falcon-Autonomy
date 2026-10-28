# Plug-ins

A plug-in is a Python class named `module:Class`, importable from the Python it runs in. Interfaces are added with
their milestones; this version defines the **driving policy**.

## Driving policy

Runs inside a live (or drive) worker, in CARLA's Python. It needs neither Qt nor carla itself.

```python
class MyPolicy:
    name = 'my policy'
    needs_images = False                    # True: obs['images'] carries the rig's camera images

    def __init__(self, **params): ...
    def reset(self, seed: int): ...         # a new episode
    def act(self, obs: dict) -> dict:       # every simulation step
        return {'throttle': 0.3, 'steer': 0.0, 'brake': 0.0}
```

`obs`:

| key | meaning |
|---|---|
| `t`, `dt` | simulation time and step, seconds |
| `ego` | `x`, `y`, `z`, `yaw` (degrees), `speed_mps`, `length`, `width` — CARLA's frame: x forward, y right, z up (left-handed), yaw clockwise seen from above |
| `route` | the lane centre ahead every 2 m, `[[x, y, speed_limit_mps or null], …]`; straight-most through junctions |
| `lead` | `{distance_m, speed_mps}` of the nearest vehicle within 1.8 m of the route, or null |

`act` returns throttle 0–1, steer −1–1 (positive: right), brake 0–1.

The built-in `falcon_autonomy.plugins.lane_follower:LaneFollower` (pure-pursuit steering, a speed controller, a time
gap behind the lead vehicle) is a baseline: it ignores traffic lights, pedestrians off its corridor and cross traffic.
