"""A built-in driving policy: pure-pursuit steering along the route's lane centre, a speed controller towards the
lane's limit (capped), and a time-gap rule behind the vehicle ahead. Deterministic; no learning.

It is a baseline to drive with and to test the policy interface, not a safe driver: it ignores traffic lights,
pedestrians off its corridor and cross traffic (stated in the Live page)."""
from __future__ import annotations

import math


class LaneFollower:
    name = 'lane follower (pure pursuit)'
    needs_images = False

    def __init__(self, cruise_kmh: float = 30.0, lookahead_min_m: float = 4.0, lookahead_s: float = 0.8,
                 time_gap_s: float = 1.8, min_gap_m: float = 5.0, wheelbase_m: float = 2.9,
                 max_steer_deg: float = 70.0):
        self.cruise = cruise_kmh / 3.6
        self.la_min, self.la_s = lookahead_min_m, lookahead_s
        self.gap_s, self.min_gap = time_gap_s, min_gap_m
        self.L, self.max_steer = wheelbase_m, math.radians(max_steer_deg)
        self.i_err = 0.0

    def reset(self, seed: int = 0):
        self.i_err = 0.0

    def act(self, obs: dict) -> dict:
        ego, route = obs['ego'], obs.get('route') or []
        v = float(ego['speed_mps'])
        if not route:
            return {'throttle': 0.0, 'steer': 0.0, 'brake': 1.0}
        yaw = math.radians(ego['yaw'])
        la = max(self.la_min, self.la_s * v)
        target = route[-1]
        for p in route:
            if math.hypot(p[0] - ego['x'], p[1] - ego['y']) >= la:
                target = p
                break
        dx, dy = target[0] - ego['x'], target[1] - ego['y']
        # the target in the vehicle's frame (CARLA: x forward, y right)
        fx = math.cos(yaw) * dx + math.sin(yaw) * dy
        fy = -math.sin(yaw) * dx + math.cos(yaw) * dy
        ld2 = max(fx * fx + fy * fy, 1e-6)
        curvature = 2.0 * fy / ld2
        steer = math.atan(self.L * curvature) / self.max_steer
        limit = min(self.cruise, float(target[2]) if len(target) > 2 and target[2] else self.cruise)
        lead = obs.get('lead')
        if lead is not None:
            safe = self.min_gap + self.gap_s * v
            if lead['distance_m'] < safe:
                limit = min(limit, max(lead['speed_mps'] * lead['distance_m'] / safe, 0.0))
        err = limit - v
        self.i_err = max(min(self.i_err + err * obs.get('dt', 0.05), 5.0), -5.0)
        u = 0.35 * err + 0.05 * self.i_err
        if u >= 0:
            return {'throttle': min(u, 0.75), 'steer': max(min(steer, 1.0), -1.0), 'brake': 0.0}
        return {'throttle': 0.0, 'steer': max(min(steer, 1.0), -1.0), 'brake': min(-u, 1.0)}
