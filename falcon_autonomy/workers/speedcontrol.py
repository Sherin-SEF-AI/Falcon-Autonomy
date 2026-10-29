"""Traffic that drives at each road's speed limit on an imported (OpenDRIVE) world, and slows for speed breakers.

CARLA's traffic manager drives at Vehicle.get_speed_limit() — 30 km/h everywhere on a standalone OpenDRIVE world,
which has no speed signs (core.speedlimits). SpeedControl gives every autopilot vehicle it holds (the traffic, an
autopilot ego, the stand-ins of core.assets) a desired speed with TrafficManager.set_desired_speed, re-read every
EVERY_S of simulated time as the vehicle moves:

    the limit of the lane it is on (the file's lane <speed>, else the road's <type><speed>, else CARLA's own
    get_speed_limit()), less the vehicle's percentage difference (the traffic profile's, with its spread),
    capped at its class's top speed (a stand-in), and lowered for what lies ahead along its lane — a road with a
    lower limit, a breaker (core.speedlimits.Breakers) — by a heuristic deceleration profile that reaches the slower
    speed there.

set_desired_speed replaces the traffic manager's limit-relative target (vehicle_percentage_speed_difference), so the
difference is applied here. A call is made only when the target moves by more than RESEND_KMH.

Made by scene.load_map for an OpenDRIVE world (cfg['speed_limits'] != 'carla'); run.json traffic.speed_source says
what was used: the file's records, how many vehicles and samples took their limit from which source, the breaker
slow-downs.
"""
from __future__ import annotations

import math

from ..core import speedlimits as SL

RESEND_KMH = 0.5
MIN_AHEAD_KMH = 10.0                 # the look-ahead reaches as far as slowing to this would need …
MAX_AHEAD_M = 150.0                  # … and no further


class SpeedControl:
    EVERY_S = 0.25

    def __init__(self, w, xodr_text: str, features_path: str | None = None):
        self.w = w
        self.table = SL.RoadSpeeds.from_xodr(xodr_text)
        self.breakers = SL.Breakers.from_features(features_path) if features_path else SL.Breakers([])
        self.rows = {}                   # actor id → [actor, diff %, top km/h, last sent km/h, role]
        self.last = -1e9
        self.map = None
        self.samples = {'lane': 0, 'road': 0, 'carla': 0}
        self.slowed = {}                 # actor id → number of breaker approaches (rising edges)
        self.in_zone = set()
        self.slowed_for = {}             # 'breaker' | 'limit ahead' → updates in which it lowered the target
        self.roles = {}                  # role → vehicles held (ever)
        self.removed = 0                 # of them, gone during the run (a dead end, the run's own removal)

    def add(self, a, diff_pct: float = 0.0, top_kmh: float | None = None, role: str = 'traffic'):
        self.rows[a.id] = [a, float(diff_pct), top_kmh, None, role]
        self.roles[role] = self.roles.get(role, 0) + 1
        self.update(a.id)

    def _map(self):
        if self.map is None:
            self.map = self.w.world.get_map()
        return self.map

    def path(self, wp, reach_m: float) -> list:
        """Waypoints ahead along the vehicle's lane every PATH_STEP_M up to reach_m: [(distance, waypoint)]. Where
        the lane branches (a junction) it follows the branch that keeps the road, else the straightest — a guess:
        the traffic manager picks its own route, and a slower stretch on another branch is caught by the next
        updates."""
        out, d, w = [], 0.0, wp
        step = SL.PATH_STEP_M
        while d + step <= reach_m:
            nxt = w.next(step)
            if not nxt:
                break
            if len(nxt) > 1:
                yaw = w.transform.rotation.yaw
                nxt = sorted(nxt, key=lambda n: (n.road_id != w.road_id,
                                                 abs((n.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0)))
            w = nxt[0]
            d += step
            out.append((d, w))
        return out

    def limit_kmh(self, a, wp) -> tuple[float | None, str]:
        lim_mps, src = (self.table.lookup(wp.road_id, wp.lane_id, wp.s) if wp is not None else (None, None))
        if lim_mps is not None:
            return lim_mps * 3.6, src
        try:
            return float(a.get_speed_limit()), 'carla'
        except RuntimeError:
            return None, 'carla'

    def target_kmh(self, a, row) -> tuple[float, str, str | None] | None:
        """(desired km/h, limit source, what slowed it: 'breaker' | 'limit ahead' | None) for a vehicle now; None
        when it is gone.

        The desired speed is the lane's limit less the vehicle's difference (capped at its top), lowered where
        something slower lies ahead along its lane within its approach distance — a breaker's crossing speed, or a
        road with a lower limit — to the speed from which APPROACH_DECEL_MPS2 reaches it there (SL.allowed_mps):
        drivers slow before the sign, not after it."""
        from ..core import assets as AS
        _, diff, top, _, _ = row
        try:
            tf = a.get_transform()
            vel = a.get_velocity()
        except RuntimeError:
            return None
        wp = self._map().get_waypoint(tf.location)
        lim_kmh, src = self.limit_kmh(a, wp)
        if lim_kmh is None:
            return None
        target = AS.desired_speed_kmh(lim_kmh, diff, top)
        v = math.hypot(vel.x, vel.y)
        reach = min(SL.approach_m(max(v, target / 3.6), MIN_AHEAD_KMH, self.EVERY_S), MAX_AHEAD_M)
        ahead = self.path(wp, reach) if wp is not None else []
        why, best = None, target
        for d, w in ahead:
            lk, _ = self.limit_kmh(a, w)
            if lk is None:
                continue
            tw = AS.desired_speed_kmh(lk, diff, top)
            if tw < best:
                al = SL.allowed_mps(d, tw, self.EVERY_S, v) * 3.6
                if al < best - 1e-6:
                    best, why = al, 'limit ahead'
        pts = [(d, w.transform.location.x, w.transform.location.y, w.transform.rotation.yaw) for d, w in ahead]
        cap = self.breakers.cap_kmh(tf.location.x, tf.location.y, tf.rotation.yaw, v, self.EVERY_S, pts)
        if cap is not None and cap < best - 1e-6:
            best, why = cap, 'breaker'
        return best, src, why

    def update(self, aid):
        row = self.rows.get(aid)
        if row is None:
            return
        a = row[0]
        got = self.target_kmh(a, row)
        if got is None:
            self.drop(aid)               # removed by the server (a dead end) or by the run
            return
        target, src, why = got
        self.samples[src] += 1
        if why is not None:
            self.slowed_for[why] = self.slowed_for.get(why, 0) + 1
        slowed = why == 'breaker'
        if slowed and aid not in self.in_zone:
            self.slowed[aid] = self.slowed.get(aid, 0) + 1
        (self.in_zone.add if slowed else self.in_zone.discard)(aid)
        tm = self.w.tm
        if tm is None:
            return
        if row[3] is None or abs(target - row[3]) > RESEND_KMH:
            try:
                tm.set_desired_speed(a, target)
            except RuntimeError:
                self.drop(aid)
                return
            row[3] = target

    def drop(self, aid):
        if self.rows.pop(aid, None) is not None:
            self.removed += 1

    def step(self, sim_t: float):
        if not self.rows or sim_t - self.last < self.EVERY_S:
            return
        self.last = sim_t
        for aid in list(self.rows):
            self.update(aid)

    def report(self) -> dict:
        return {'mode': 'map', 'method': 'TrafficManager.set_desired_speed from the OpenDRIVE file\'s lane / road '
                'speed records, re-read every %.2f s of simulated time (workers.speedcontrol)' % self.EVERY_S,
                **self.table.summary(), 'vehicles': dict(self.roles), 'removed_during_run': self.removed,
                'samples_by_source': dict(self.samples),
                'breakers': len(self.breakers), 'breaker_approaches': int(sum(self.slowed.values())),
                'updates_slowed_by': dict(self.slowed_for),
                'look_ahead': f'waypoints every {SL.PATH_STEP_M} m along the lane, as far as slowing to '
                              f'{MIN_AHEAD_KMH} km/h at {SL.APPROACH_DECEL_MPS2} m/s² needs (at most {MAX_AHEAD_M} m); '
                              'a lower limit or a breaker ahead lowers the target to the speed that reaches it there',
                'breaker_rule': f'a speed profile decelerating at {SL.APPROACH_DECEL_MPS2} m/s² to the breaker\'s '
                                f'crossing speed at the breaker (less one update\'s travel and {SL.APPROACH_MARGIN_M} m), '
                                f'held until {SL.PAST_M} m past it (heuristic)'}
