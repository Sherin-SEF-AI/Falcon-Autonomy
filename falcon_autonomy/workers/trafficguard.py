"""Traffic that stays in an imported (OpenDRIVE) world.

An imported area is cut from a larger map, and its traffic is lost (measured on CARLA 0.9.16, a scratch copy of the
re-imported central Bengaluru, 60 vehicles, keep_traffic false: 14–17 left after 150 s). Two things take it, both
measured there:

* the traffic manager's OSM mode — on by default in 0.9.16 — **destroys a vehicle as soon as its planned path meets a
  dead end of its road graph** (54 vehicles, 30 s: 41 left with it on, 54 with it off). Its graph has far more dead
  ends than the map's 14 lane ends without a successor: 41 of 64 vehicles removed in 180 s were last seen on a lane
  more than 40 m from any of those. With the mode off, the vehicles drive on past the end of their path instead,
  off the road mesh, and fall (150 s: 39 of 54 fell);
* holes in the generated road mesh, where a junction's connecting road does not meet its lanes: a vehicle sinks
  through the road where the map still has a lane.

TrafficKeeper keeps the count. Every EVERY_S of simulated time, for each traffic vehicle it holds:

    * gone (removed by the server or the traffic manager): a vehicle of the same blueprint is spawned in its place,
      set up as the traffic profile set the first (its behaviour, speed difference, class and road speed);
    * falling — FALL_M below its lane (or its last road height off the lanes), or faster than FALL_MPS — or off the
      road network (no driving lane within LOST_M) for LOST_S: moved at once (it is leaving anyway);
    * a dead end of the map on its way — the traffic manager's planned path (TrafficManager.get_all_actions) and the
      lane straight on from there, as far as LOOK_S of travel (LOOK_MIN_M … LOOK_MAX_M); a lane end with no successor
      and no driving lane PROBE_M beyond it — moved, unless the ego might see it go (in_view: within VIEW_NEAR_M, or
      within VIEW_FAR_M and VIEW_HALF_DEG of the ego's heading — a heuristic for the rig's cameras): then it comes off
      the autopilot and stops with full brakes, like a car at a closed road, and is moved once out of view.

A move or a replacement takes a spawn point in a seeded order (the run's seed) among those AWAY_M from where the
vehicle was, out of the ego's view, with no vehicle within CLEAR_M and no dead end within SPAWN_CLEAR_M ahead; a move
is one synchronous ApplyTransform (common.place_now, before the next tick) with the autopilot restarted there, so the
traffic manager plans afresh. Every move is an event (frames.jsonl "traffic_respawns", run.json traffic.keeper).
Measured over 180 s (60 vehicles, the Indian mixed profile): 58–60 alive throughout and 60 at the end; 64 replaced,
31 caught falling; none of the 95 moved or replaced vehicles vanished or appeared within 120 m and 45° of the ego's
heading. Tried and left out: the OSM mode off with a move where the planned path runs out — get_all_actions gives
the next actions, not the path's end (3882 moves in 180 s, nearly all needless).

Made by scene.load_map for an OpenDRIVE world unless cfg['keep_traffic'] is false. CARLA's towns are closed and
never trigger it."""
from __future__ import annotations

import math

from .common import ROLE_HERO, place_now, role


class TrafficKeeper:
    EVERY_S = 0.25
    # the traffic manager removes a vehicle well before a dead end: measured on central Bengaluru (CARLA 0.9.16), the
    # ego removed 26–30 m before one at city speed and 48 m before one at 16.5 m/s; its planned path is not readable
    # (get_all_actions: the next action only, 0.5–5 m ahead), so the look-ahead is longer than those
    LOOK_S = 3.0
    LOOK_MIN_M = 40.0
    LOOK_MAX_M = 75.0
    STEP_M = 2.0
    PROBE_M = 4.0
    FALL_M = 1.0
    FALL_MPS = 3.0
    LOST_S = 2.0
    LOST_M = 10.0
    AWAY_M = 60.0
    CLEAR_M = 10.0
    SPAWN_CLEAR_M = 40.0
    VIEW_NEAR_M = 50.0
    VIEW_FAR_M = 120.0
    VIEW_HALF_DEG = 60.0
    MAX_EVENTS = 200

    def __init__(self, w, world, seed: int = 0):
        import numpy as np
        self.w, self.world = w, world
        self.cmap = world.get_map()
        spawns = list(self.cmap.get_spawn_points())
        rng = np.random.default_rng([int(seed), 0x7AF1C])
        self.spawns = [spawns[int(i)] for i in rng.permutation(len(spawns))]
        self.cursor = 0
        self.rows: dict = {}             # actor id → row (dict)
        self.ego = None
        self.last = -1e9
        self.counts = {'dead_end': 0, 'dead_end_after_hold': 0, 'fell': 0, 'lost': 0, 'replaced': 0,
                       'no_free_spawn': 0}
        self.events: list = []
        self.held_now = 0
        self.max_held = 0
        self.vehicles = 0
        self._ends: dict = {}            # lane end key → dead end?

    # ------------------------------------------------------------------ holding vehicles
    def add(self, a, beh: dict | None = None, diff_pct: float | None = None, kind: str = '', cls: str | None = None,
            label: str | None = None):
        """A traffic vehicle (on the autopilot), with how the traffic profile set it up (re-applied after a move):
        its behaviour, speed difference and, for a stand-in (core.assets), its class and label."""
        self.rows[a.id] = {'a': a, 'beh': dict(beh or {}), 'diff': diff_pct, 'kind': kind, 'cls': cls, 'label': label,
                           'type_id': a.type_id, 'last_z': None, 'off_since': None, 'held_since': None, 'seen': None}
        self.vehicles += 1

    def _ego(self):
        if self.ego is not None:
            try:
                if self.ego.is_alive:
                    return self.ego
            except RuntimeError:
                pass
            self.ego = None
        e = getattr(self.w, 'ego', None)
        if e is None:
            try:
                e = next((x for x in self.world.get_actors().filter('vehicle.*') if role(x) == ROLE_HERO), None)
            except RuntimeError:
                e = None
        self.ego = e
        return e

    def in_view(self, x: float, y: float, ego_tf=None) -> bool:
        if ego_tf is None:
            return False
        dx, dy = x - ego_tf.location.x, y - ego_tf.location.y
        d = math.hypot(dx, dy)
        if d < self.VIEW_NEAR_M:
            return True
        if d >= self.VIEW_FAR_M:
            return False
        rel = (math.degrees(math.atan2(dy, dx)) - ego_tf.rotation.yaw + 180.0) % 360.0 - 180.0
        return abs(rel) < self.VIEW_HALF_DEG

    # ------------------------------------------------------------------ the road ahead
    def _is_dead_end(self, wp) -> bool:
        """wp's lane ends within STEP_M with no successor (wp.next(STEP_M) is empty): a dead end when no driving lane
        lies STEP_M + PROBE_M beyond wp straight on (else the lane only ends — a lane drop, a road going on)."""
        key = (wp.road_id, wp.section_id, wp.lane_id, round(wp.s, 0))
        if key in self._ends:
            return self._ends[key]
        carla = self.w.carla
        t = wp.transform
        yaw = math.radians(t.rotation.yaw)
        d = self.STEP_M + self.PROBE_M
        p = carla.Location(t.location.x + math.cos(yaw) * d, t.location.y + math.sin(yaw) * d, t.location.z)
        dead = self.cmap.get_waypoint(p, project_to_road=False, lane_type=carla.LaneType.Driving) is None
        self._ends[key] = dead
        return dead

    def _walk(self, wp, d0: float, reach: float):
        """From waypoint wp (d0 m ahead of the vehicle), along the lane straight on: the distance of a dead end within
        reach, else None."""
        d, cur = d0, wp
        while d <= reach:
            nxt = cur.next(self.STEP_M)
            if not nxt:
                return d if self._is_dead_end(cur) else None
            if len(nxt) > 1:
                yaw = cur.transform.rotation.yaw
                nxt = sorted(nxt, key=lambda n: (n.road_id != cur.road_id,
                                                 abs((n.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0)))
            cur = nxt[0]
            d += self.STEP_M
        return None

    def dead_end_ahead(self, a, wp, loc, speed: float):
        """The distance (m) to a dead end of the map on the vehicle's way within its look-ahead, else None: to the
        traffic manager's next action (TrafficManager.get_all_actions — its last action's waypoint, a few metres
        ahead) and from there the lane straight on; without it, along the lane from where the vehicle is."""
        reach = min(max(speed * self.LOOK_S, self.LOOK_MIN_M), self.LOOK_MAX_M)
        tm = getattr(self.w, 'tm', None)
        start, d0 = wp, 0.0
        if tm is not None and hasattr(tm, 'get_all_actions'):
            try:
                acts = tm.get_all_actions(a)
            except RuntimeError:
                acts = None
            if acts:
                last = acts[-1][1]
                d0 = last.transform.location.distance(loc)
                if d0 > reach:
                    return None
                start = last
        return self._walk(start, d0, reach)

    # ------------------------------------------------------------------ moving a vehicle
    def _free_spawn(self, frm, ego_tf, others, strict_view=True):
        n = len(self.spawns)
        for k in range(n):
            i = (self.cursor + k) % n
            sp = self.spawns[i]
            L = sp.location
            if frm is not None and math.hypot(L.x - frm.x, L.y - frm.y) <= self.AWAY_M:
                continue
            if strict_view and self.in_view(L.x, L.y, ego_tf):
                continue
            if ego_tf is not None and math.hypot(L.x - ego_tf.location.x, L.y - ego_tf.location.y) < self.CLEAR_M:
                continue
            if any(math.hypot(L.x - o[0], L.y - o[1]) < self.CLEAR_M for o in others):
                continue
            wp = self.cmap.get_waypoint(L, project_to_road=True, lane_type=self.w.carla.LaneType.Driving)
            if wp is None or self._walk(wp, 0.0, self.SPAWN_CLEAR_M) is not None:
                continue
            self.cursor = (i + 1) % n
            return sp
        return None

    def _others(self, skip_id):
        out = []
        try:
            for v in self.world.get_actors().filter('vehicle.*'):
                if v.id != skip_id:
                    L = v.get_location()
                    out.append((L.x, L.y))
        except RuntimeError:
            pass
        return out

    def _restart(self, a, row):
        """The autopilot again, with the profile's settings (the traffic manager plans from the new place)."""
        from .scene import apply_behaviour
        tm = self.w.tm
        a.set_autopilot(True, tm.get_port())
        apply_behaviour(tm, a, row['beh'])
        if row['diff'] is not None and hasattr(tm, 'vehicle_percentage_speed_difference'):
            tm.vehicle_percentage_speed_difference(a, float(row['diff']))
        sc = getattr(self.w, 'speed_control', None)
        if sc is not None and a.id in sc.rows:
            sc.rows[a.id][3] = None          # the desired speed is sent again
            sc.update(a.id)
        cl = getattr(self.w, 'class_limits', None)
        if cl is not None and a.id in cl.rows and sc is None:
            cl.update(a.id, force=True)
        lat = getattr(self.w, 'lateral', None)
        if lat is not None:                  # a vehicle that filters laterally: from its lane centre again
            lat.reset(a)

    def _move(self, a, row, loc, why: str, sim_t: float, ego_tf, strict_view=True):
        carla = self.w.carla
        sp = self._free_spawn(loc, ego_tf, self._others(a.id), strict_view)
        if sp is None:
            self.counts['no_free_spawn'] += 1
            return None
        try:
            if row['held_since'] is None:
                a.set_autopilot(False, self.w.tm.get_port())
            a.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            a.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            place_now(self.w.client, carla, [(a, sp)])
            a.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
            self._restart(a, row)
        except RuntimeError:
            return None                      # removed meanwhile: replaced on the next update
        held = None if row['held_since'] is None else round(sim_t - row['held_since'], 2)
        if row['held_since'] is not None:
            self.held_now -= 1
        row.update(last_z=sp.location.z, off_since=None, held_since=None)
        self.counts[why] += 1
        ev = {'sim_t': round(float(sim_t), 3), 'kind': why.replace('_after_hold', ''), 'actor': a.id,
              'from': {'x': round(loc.x, 1), 'y': round(loc.y, 1), 'z': round(loc.z, 1)},
              'to': {'x': round(sp.location.x, 1), 'y': round(sp.location.y, 1)}}
        if held is not None:
            ev['held_s'] = held
        if len(self.events) < self.MAX_EVENTS:
            self.events.append(ev)
        return ev

    def _hold(self, a, row, sim_t):
        carla = self.w.carla
        try:
            a.set_autopilot(False, self.w.tm.get_port())
            a.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except RuntimeError:
            return
        row['held_since'] = sim_t
        self.held_now += 1
        self.max_held = max(self.max_held, self.held_now)

    def _replace(self, aid, row, sim_t, ego_tf):
        """A vehicle the server removed: one of the same blueprint at a free spawn point, set up as the first."""
        from .common import ROLE_TRAFFIC
        world = self.world
        self.rows.pop(aid, None)
        if row['held_since'] is not None:
            self.held_now -= 1
        sc = getattr(self.w, 'speed_control', None)
        if sc is not None:
            sc.drop(aid)
        lib = world.get_blueprint_library()
        try:
            bp = lib.find(row['type_id'])
        except (IndexError, RuntimeError):
            return None
        bp.set_attribute('role_name', ROLE_TRAFFIC)
        for _ in range(3):
            sp = self._free_spawn(None, ego_tf, self._others(None))
            if sp is None:
                break
            a = world.try_spawn_actor(bp, sp)
            if a is None:
                continue
            self.w.own(a)
            self.add(a, row['beh'], row['diff'], row['kind'], row['cls'], row['label'])
            self.vehicles -= 1               # the same vehicle of the profile, not one more
            lat = getattr(self.w, 'lateral', None)
            if lat is not None:              # it filtered laterally: its replacement does too
                lat.adopt(aid, a)
            a.set_autopilot(True, self.w.tm.get_port())
            if row['cls']:                   # a stand-in: its class's label and limits again
                from .scene import class_limits
                class_limits(self.w).add(a, row['cls'], float(row['diff'] or 0.0), row['label'])
            elif sc is not None:
                sc.add(a, float(row['diff'] or 0.0), role='traffic')
            self._restart(a, self.rows[a.id])
            try:
                lit = next((r['a'].get_light_state() for r in self.rows.values() if r['a'] is not a), None)
                if lit is not None and int(lit) != 0:
                    a.set_light_state(lit)
            except (RuntimeError, TypeError, AttributeError):
                pass
            self.counts['replaced'] += 1
            ev = {'sim_t': round(float(sim_t), 3), 'kind': 'replaced', 'actor': a.id, 'was': aid,
                  'to': {'x': round(sp.location.x, 1), 'y': round(sp.location.y, 1)}}
            if row.get('seen'):
                ev['last_seen'] = dict(zip('xyz', row['seen']))
            if len(self.events) < self.MAX_EVENTS:
                self.events.append(ev)
            return ev
        self.counts['no_free_spawn'] += 1
        return None

    # ------------------------------------------------------------------ each update
    def step(self, sim_t: float) -> list:
        """Look at every vehicle held (every EVERY_S of simulated time); the moves made, as events."""
        if not self.rows or sim_t - self.last < self.EVERY_S:
            return []
        self.last = sim_t
        carla = self.w.carla
        ego = self._ego()
        try:
            ego_tf = ego.get_transform() if ego is not None else None
        except RuntimeError:
            ego_tf = None
        out = []
        for aid in list(self.rows):
            row = self.rows.get(aid)
            if row is None:
                continue
            a = row['a']
            try:
                alive = a.is_alive
                tf = a.get_transform() if alive else None
                vel = a.get_velocity() if alive else None
            except RuntimeError:
                alive = False
            if not alive:
                ev = self._replace(aid, row, sim_t, ego_tf)
                if ev is not None:
                    out.append(ev)
                continue
            loc = tf.location
            row['seen'] = (round(loc.x, 1), round(loc.y, 1), round(loc.z, 1))
            wp = self.cmap.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
            if wp is not None and (loc.z < wp.transform.location.z - self.FALL_M or vel.z < -self.FALL_MPS):
                # falling where the map has a lane: a hole in the generated road mesh (a gap where a junction's
                # connecting road does not meet its lanes)
                ev = self._move(a, row, loc, 'fell', sim_t, ego_tf, strict_view=False)
                if ev is not None:
                    ev['on_lane'] = True
                    out.append(ev)
                continue
            if wp is None:
                sinking = (row['last_z'] is not None and loc.z < row['last_z'] - self.FALL_M) or vel.z < -self.FALL_MPS
                if row['off_since'] is None:
                    row['off_since'] = sim_t
                lost = False
                if not sinking and sim_t - row['off_since'] >= self.LOST_S:
                    near = self.cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    lost = near is None or near.transform.location.distance(loc) > self.LOST_M
                if sinking or lost:
                    ev = self._move(a, row, loc, 'fell' if sinking else 'lost', sim_t, ego_tf, strict_view=False)
                    if ev is not None:
                        out.append(ev)
                continue
            row['last_z'], row['off_since'] = loc.z, None
            if row['held_since'] is not None:
                if not self.in_view(loc.x, loc.y, ego_tf):
                    ev = self._move(a, row, loc, 'dead_end_after_hold', sim_t, ego_tf)
                    if ev is not None:
                        out.append(ev)
                continue
            if self.dead_end_ahead(a, wp, loc, math.hypot(vel.x, vel.y)) is None:
                continue
            if self.in_view(loc.x, loc.y, ego_tf):
                self._hold(a, row, sim_t)
                continue
            ev = self._move(a, row, loc, 'dead_end', sim_t, ego_tf)
            if ev is not None:
                out.append(ev)
        return out

    def report(self) -> dict:
        return {'method': 'workers.trafficguard.TrafficKeeper: a dead end on a vehicle\'s way within '
                          f'{self.LOOK_S:g} s of travel ({self.LOOK_MIN_M:g}–{self.LOOK_MAX_M:g} m; the traffic '
                          'manager\'s planned path, then the lane straight on) moves it to a free spawn point out of '
                          'the ego\'s view (a vehicle the ego might see stops first and moves when out of view); a '
                          'falling or lost vehicle moves at once; a removed one is replaced',
                'view_rule': f'within {self.VIEW_NEAR_M:g} m of the ego, or within {self.VIEW_FAR_M:g} m and '
                             f'{self.VIEW_HALF_DEG:g}° of its heading (heuristic)',
                'vehicles': self.vehicles, 'alive_at_end': self.alive(), **self.counts, 'max_held': self.max_held,
                'events': self.events[:self.MAX_EVENTS], 'events_truncated': sum(
                    v for k, v in self.counts.items() if k != 'no_free_spawn') > len(self.events)}

    def alive(self) -> int:
        n = 0
        for r in self.rows.values():
            try:
                n += bool(r['a'].is_alive)
            except RuntimeError:
                pass
        return n
