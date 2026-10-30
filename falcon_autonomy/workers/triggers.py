"""The trigger-rule runner (core.triggers) inside a worker: reads the signals of the frame just ticked, asks the
engine which rules fire, carries out their actions before the next tick (runs in CARLA's Python).

    runner = RuleRunner.from_config(w, world, cfg, ego, director, faults=…, physics=…)   # None when no rules
    each step:  runner.before_tick(t)          # an attached director steps the actors rules added
                frame = world.tick()
                runner.after_tick(frame, t, rec)   # signals → firings → actions; rec["triggers"] when any fired

Signals are computed lazily, only those the rules read, once per tick: the ego's pose and velocity from the
scenario director where it drives the ego kinematically (CARLA reports a physics-off actor still), else from the
world snapshot of the frame (never get_transform(), which can be a tick late); the lane ahead by Waypoint.next in 2 m
steps (straight-most at forks) up to core.triggers.LOOKAHEAD_M, junctions by Waypoint.is_junction, lights by their
stop waypoints (TrafficLight.get_stop_waypoints, read once); actors' boxes from the snapshot with their extents read
once per actor; the weather is the one in force (the config's, then each weather action's).

Spawns go through the scenario director (workers.director): the run's own when it runs a scenario, else one
attached on the first spawn (Director.attach) that drives only those actors. Its random draws come from a generator
seeded by (the run's seed, crc32 of the rule id), never the run's own, so a rule changes nothing it does not do.
"""
from __future__ import annotations

import math
import zlib

from ..core import triggers as T
from .common import Failed

STEP_M = 2.0


class RuleRunner:
    def __init__(self, w, world, cfg, ego, director=None, faults=None, physics=None, on_event=None):
        import numpy as np
        self.np = np
        self.w, self.world, self.cfg, self.ego = w, world, cfg, ego
        self.carla = w.carla
        self.map = world.get_map()
        doc = cfg['trigger_rules']
        self.rules = doc['rules']
        self.engine = T.Engine(self.rules)
        self.director = director
        self.own_director = False
        self.faults, self.physics = faults, physics
        self.on_event = on_event                 # Live: firings shown as its events (log_event)
        self.seed = int(cfg.get('seed', 0))
        self.rngs = {r['id']: np.random.default_rng([self.seed, zlib.crc32(r['id'].encode())]) for r in self.rules}
        from ..core import weather as W
        self.weather = dict(W.resolve(None), **(cfg.get('weather') or {}))
        self.firings = []
        self.spawned = {}                        # actor id in the run -> CARLA id
        self.n_spawn = {}
        self.cache = {}
        self.prev_speed = None
        self.prev_t = None
        self.extent = {}                         # CARLA id -> (type, ex, ey) of the actors seen
        self.stop_lines = None
        w.emit.log(f'trigger rules: {len(self.rules)} ({", ".join(r["id"] for r in self.rules)}), signals '
                   f'{", ".join(self.engine.signals) or "none"}')

    @classmethod
    def from_config(cls, w, world, cfg, ego, director=None, **kw):
        if not (cfg.get('trigger_rules') or {}).get('rules'):
            return None
        try:
            return cls(w, world, cfg, ego, director, **kw)
        except T.RuleError as e:
            raise Failed(f'trigger rules: {e}') from None

    # ------------------------------------------------------------------ stepping
    def before_tick(self, t):
        if self.own_director:
            self.director.step(t)

    def after_tick(self, frame, t, rec=None) -> list:
        """Evaluate every rule on the state of `frame` (t: the run's simulation time); carry out the firings.
        Returns the firings (also written into rec['triggers'] when rec is given)."""
        if self.own_director:
            self.director.sync(frame)
            self.director.record(t)
        self.cache = {'_t': t, '_frame': frame}
        fired = self.engine.step(t, self.signal)
        out = []
        for rule, actions, seen in fired:
            done = [self.act(rule, a, t) for a in actions]
            f = {'rule': rule['id'], 't': round(float(t), 4), 'signals': {k: T.jsonable(v) for k, v in seen.items()},
                 'actions': done}
            if rec is not None:
                f['step'] = rec.get('step')
            out.append(f)
            self.firings.append(f)
            self.w.emit.log(f'trigger {rule["id"]} at {t:.2f} s: ' + '; '.join(
                d['did'] if 'did' in d else f'{d["action"]} — not done: {d["error"]}' for d in done))
            if self.on_event is not None:
                self.on_event('trigger', {'rule': rule['id'], 'actions': done})
        if out and rec is not None:
            rec['triggers'] = [{k: v for k, v in f.items() if k != 'step'} for f in out]
        self.prev_t = t
        self.prev_speed = self.signal_speed_mps()
        return out

    # ------------------------------------------------------------------ signals
    def snap(self):
        s = self.cache.get('_snap')
        if s is None:
            s = self.cache['_snap'] = self.world.get_snapshot()
        return s

    def _tf_vel(self, actor_id):
        d = self.director
        if d is not None and actor_id in getattr(d, 'kin', {}):
            k = d.kin[actor_id]
            r = math.radians(k[3])
            return (k[0], k[1], k[3]), (k[4] * math.cos(r), k[4] * math.sin(r))
        a = self.snap().find(actor_id)
        if a is None:
            return None, None
        t, v = a.get_transform(), a.get_velocity()
        return (t.location.x, t.location.y, t.rotation.yaw), (v.x, v.y)

    def ego_pose(self):
        if '_ego' not in self.cache:
            self.cache['_ego'] = self._tf_vel(self.ego.id)
        return self.cache['_ego']

    def signal_speed_mps(self):
        p, v = self.ego_pose()
        return None if v is None else math.hypot(v[0], v[1])

    def signal(self, sig):
        if sig in self.cache:
            return self.cache[sig]
        v = self._compute(sig)
        self.cache[sig] = v
        return v

    def _compute(self, sig):
        w = self.weather
        if sig == 'time':
            return self.cache['_t']
        if sig == 'ego.speed':
            s = self.signal_speed_mps()
            return None if s is None else s * 3.6
        if sig == 'ego.accel':
            s = self.signal_speed_mps()
            if s is None or self.prev_speed is None or self.prev_t is None or self.cache['_t'] <= self.prev_t:
                return 0.0
            return (s - self.prev_speed) / (self.cache['_t'] - self.prev_t)
        if sig in ('ego.junction_m', 'ego.light_m', 'ego.light'):
            self.lane_ahead()
            return self.cache[sig]
        if sig == 'visibility':
            from ..core.weather import fog_visibility_m
            v = fog_visibility_m(float(w.get('fog_density', 0.0)))
            return math.inf if v is None else v
        if sig in ('fog', 'rain', 'wetness', 'puddles', 'cloud', 'sun'):
            key = {'fog': 'fog_density', 'rain': 'precipitation', 'wetness': 'wetness',
                   'puddles': 'precipitation_deposits', 'cloud': 'cloudiness', 'sun': 'sun_altitude_angle'}[sig]
            return float(w.get(key, 0.0))
        if sig == 'tod':
            from ..core.runindex import tod_of
            return tod_of(float(w.get('sun_altitude_angle', 0.0)))
        if sig in ('actor.nearest_m', 'vehicle.nearest_m', 'walker.nearest_m', 'actor.ttc', 'actor.count'):
            self.actors_near()
            return self.cache[sig]
        return None

    def lane_ahead(self):
        """ego.junction_m, ego.light_m, ego.light: one walk along the ego's lane."""
        self.cache.update({'ego.junction_m': math.inf, 'ego.light_m': math.inf, 'ego.light': 'none'})
        p, _ = self.ego_pose()
        if p is None:
            self.cache.update({'ego.junction_m': None, 'ego.light_m': None, 'ego.light': None})
            return
        wp = self.map.get_waypoint(self.carla.Location(x=p[0], y=p[1], z=0.0))
        if wp is None:
            return
        if self.stop_lines is None:
            self.stop_lines = []
            for tl in self.world.get_actors().filter('traffic.traffic_light'):
                try:
                    for s in tl.get_stop_waypoints():
                        t = s.transform
                        self.stop_lines.append((tl, t.location.x, t.location.y, t.rotation.yaw,
                                                s.lane_width / 2.0))
                except (RuntimeError, AttributeError):
                    continue
        # where the ego stands on its lane: its front's offset from the waypoint along the lane
        yaw = math.radians(wp.transform.rotation.yaw)
        fwd = (math.cos(yaw), math.sin(yaw))
        off = (p[0] - wp.transform.location.x) * fwd[0] + (p[1] - wp.transform.location.y) * fwd[1] + \
            self.ego.bounding_box.extent.x
        cur, s = wp, -off
        junction = light = None
        for _ in range(int(T.LOOKAHEAD_M / STEP_M) + 1):
            if junction is None and cur.is_junction and s > 0:
                junction = s
            if light is None:
                cx, cy, cyaw = cur.transform.location.x, cur.transform.location.y, cur.transform.rotation.yaw
                for tl, x, y, lyaw, half in self.stop_lines:
                    if abs((lyaw - cyaw + 180.0) % 360.0 - 180.0) > 30.0:
                        continue
                    r = math.radians(cyaw)
                    along = (x - cx) * math.cos(r) + (y - cy) * math.sin(r)
                    across = -(x - cx) * math.sin(r) + (y - cy) * math.cos(r)
                    if abs(along) <= STEP_M / 2.0 + 1e-6 and abs(across) <= half and s + along >= 0:
                        light = (s + along, tl)
                        break
            if junction is not None and light is not None:
                break
            nxt = cur.next(STEP_M)
            if not nxt:
                break
            cur = min(nxt, key=lambda n: abs((n.transform.rotation.yaw - cur.transform.rotation.yaw + 180.0)
                                             % 360.0 - 180.0))
            s += STEP_M
        if junction is not None:
            self.cache['ego.junction_m'] = max(junction, 0.0)
        if light is not None:
            self.cache['ego.light_m'] = max(light[0], 0.0)
            try:
                st = light[1].get_state()
                name = str(getattr(st, 'name', st)).rsplit('.', 1)[-1].lower()
                self.cache['ego.light'] = name if name in ('red', 'yellow', 'green', 'off') else 'unknown'
            except RuntimeError:
                self.cache['ego.light'] = None

    def actors_near(self):
        """Nearest box distances, the smallest TTC and the count, over vehicles, walkers and scenario props."""
        from ..core import criticality as C
        np = self.np
        self.cache.update({'actor.nearest_m': math.inf, 'vehicle.nearest_m': math.inf,
                           'walker.nearest_m': math.inf, 'actor.ttc': math.inf, 'actor.count': 0})
        p, v = self.ego_pose()
        if p is None:
            for k in ('actor.nearest_m', 'vehicle.nearest_m', 'walker.nearest_m', 'actor.ttc', 'actor.count'):
                self.cache[k] = None
            return
        snap = self.snap()
        ids = [a.id for a in snap if a.id != self.ego.id]
        new = [i for i in ids if i not in self.extent]
        if new:
            for a in self.world.get_actors(new):
                tid = str(a.type_id)
                kind = 'vehicle' if tid.startswith('vehicle.') else 'walker' if tid.startswith('walker.pedestrian') \
                    else 'prop' if tid.startswith('static.prop') and a.id in (getattr(self.w, 'proxies', None) or {}) \
                    else None
                if kind is None:
                    self.extent[a.id] = None
                    continue
                px = (getattr(self.w, 'proxies', None) or {}).get(a.id)
                if px is not None:
                    ex, ey = px['extent'][0], px['extent'][1]
                    if round(abs(px['yaw_offset'])) % 180 == 90:
                        ex, ey = ey, ex
                else:
                    ex, ey = a.bounding_box.extent.x, a.bounding_box.extent.y
                self.extent[a.id] = (kind, float(ex), float(ey))
            for i in new:
                self.extent.setdefault(i, None)
        eb = self.ego.bounding_box.extent
        ego_state = np.array([0.0, p[0], p[1], p[2], v[0], v[1], 2 * eb.x, 2 * eb.y])
        ego_box = C.box_corners(p[0], p[1], p[2], 2 * eb.x, 2 * eb.y)
        count = 0
        for i in ids:
            e = self.extent.get(i)
            if e is None:
                continue
            q, u = self._tf_vel(i)
            if q is None:
                continue
            d0 = math.hypot(q[0] - p[0], q[1] - p[1])
            if d0 <= T.COUNT_RANGE_M:
                count += 1
            if d0 > T.ACTOR_RANGE_M:
                continue
            kind, ex, ey = e
            d = C.box_distance(ego_box, C.box_corners(q[0], q[1], q[2], 2 * ex, 2 * ey))
            ttc = C.ttc(ego_state, np.array([0.0, q[0], q[1], q[2], u[0], u[1], 2 * ex, 2 * ey]), T.TTC_HORIZON_S)
            self.cache['actor.nearest_m'] = min(self.cache['actor.nearest_m'], d)
            if kind in ('vehicle', 'walker'):
                self.cache[f'{kind}.nearest_m'] = min(self.cache[f'{kind}.nearest_m'], d)
            self.cache['actor.ttc'] = min(self.cache['actor.ttc'], ttc)
        self.cache['actor.count'] = count

    # ------------------------------------------------------------------ actions
    def act(self, rule, a, t) -> dict:
        """Carry out one action: {'action': its text, 'did': what was done} or {'action', 'error': why not}."""
        text = T.describe_action(a)
        try:
            if a['do'] == 'spawn':
                return dict(self.spawn(rule, a, t), action=text)
            if a['do'] == 'weather':
                return dict(self.set_weather(a), action=text)
            if a['do'] == 'fault':
                return dict(self.start_fault(a, t), action=text)
            return {'action': text, 'did': 'marker' + (f': {a["text"]}' if a['text'] else '')}
        except Failed as e:
            return {'action': text, 'error': str(e)}
        except RuntimeError as e:                    # the server refused (an actor destroyed meanwhile …)
            return {'action': text, 'error': f'CARLA: {e}'}

    def spawn(self, rule, a, t) -> dict:
        d = self.director
        if d is None:
            from .director import Director
            d = self.director = Director.attach(self.w, self.world, self.cfg, self.world.get_blueprint_library(),
                                                self.rngs[rule['id']], self.ego)
            self.own_director = True
        if not hasattr(d, 'add_actor'):
            raise Failed('a log replay cannot add actors')
        n = self.n_spawn[rule['id']] = self.n_spawn.get(rule['id'], 0) + 1
        aid = f'{rule["id"]}-{a["what"]}-{n}'
        doc, extra = self.actor_doc(a, aid)
        cid = d.add_actor(doc, t, self.rngs[rule['id']])
        for act in extra.get('now', []):
            d.act(dict(act, actor=aid), t)
        if extra.get('timeline'):
            d.add_timeline(extra['timeline'])
        self.spawned[aid] = cid
        return {'did': f'spawned {aid} (CARLA id {cid})', 'actor': aid, 'carla_id': cid}

    def actor_doc(self, a, aid):
        """The scenario actor document (core.scenarios) of a spawn, and what follows it: actions now, a timeline
        entry."""
        what, args = a['what'], a['args']
        spec = T.SPAWNS.get(what)

        def arg(k):
            return float(args.get(k, spec['args'][k][0]))
        ego_kmh = (self.signal_speed_mps() or 0.0) * 3.6
        if what in ('jaywalker', 'child'):
            return ({'id': aid, 'kind': spec['kind'], 'place': {'lane': 'sidewalk', 'ahead_m': arg('distance')},
                     'heading': 'across', 'speed_kmh': 0, 'start': 'stopped'},
                    {'now': [{'do': 'walk', 'speed_mps': arg('walk')}]})
        if what in ('cut_in', 'autorickshaw_cut_in'):
            return ({'id': aid, 'kind': spec['kind'], 'place': {'lane': 'adjacent', 'ahead_m': -arg('behind')},
                     'heading': 'lane', 'speed_kmh': ego_kmh + arg('faster'), 'start': 'moving'},
                    {'timeline': {'trigger': {'type': 'ahead', 'actor': aid, 'gt_m': arg('gap')},
                                  'actions': [{'actor': aid, 'do': 'lane_change', 'to': 'ego',
                                               'duration_s': arg('duration')}]}})
        if what == 'cattle':
            return ({'id': aid, 'kind': 'cattle', 'place': {'lane': 'ego', 'ahead_m': arg('distance'),
                                                            'lateral_m': arg('lateral')},
                     'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}, {})
        if what == 'handcart':
            return ({'id': aid, 'kind': 'handcart', 'place': {'lane': 'ego', 'ahead_m': arg('distance'),
                                                              'lateral_m': arg('lateral')},
                     'heading': 'lane', 'speed_kmh': arg('speed'), 'start': 'moving'}, {})
        if what == 'stalled':
            return ({'id': aid, 'kind': args.get('kind', 'car'), 'place': {'lane': 'ego', 'ahead_m': arg('distance')},
                     'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}, {})
        if what == 'wrong_way':
            return ({'id': aid, 'kind': 'motorcycle', 'place': {'lane': 'ego', 'ahead_m': arg('distance'),
                                                                'lateral_m': arg('lateral')},
                     'heading': 'against', 'speed_kmh': arg('speed'), 'start': 'moving'}, {})
        g = T.GENERIC_ARGS
        start = args.get('start', 'moving' if float(args.get('speed', 0.0)) > 0 else 'stopped')
        heading = args.get('heading', 'lane')
        doc = {'id': aid, 'kind': what, 'place': {'lane': args.get('lane', 'ego'),
                                                  'ahead_m': float(args.get('ahead', g['ahead'][0])),
                                                  'lateral_m': float(args.get('lateral', g['lateral'][0]))},
               'heading': heading, 'speed_kmh': float(args.get('speed', 0.0)), 'start': start}
        extra = {}
        if what in ('pedestrian', 'child') and float(args.get('speed', 0.0)) > 0:
            extra = {'now': [{'do': 'walk', 'speed_mps': float(args['speed']) / 3.6}]}
        return doc, extra

    def set_weather(self, a) -> dict:
        from ..core import weather as W
        from . import scene
        if a.get('preset'):
            self.weather = W.resolve({'preset': a['preset']})
        self.weather.update(a['fields'])
        self.world.set_weather(scene.weather_params(self.carla, self.weather))
        if self.physics is not None:
            self.physics.set_weather(dict(self.weather))
        try:
            lit = scene.light_vehicles(self.w, self.cfg, self.weather)       # night falls: headlights on (and off)
        except (RuntimeError, KeyError):
            lit = {'resolved': None}
        changed = {'preset': a['preset']} if a.get('preset') else {}
        changed.update(a['fields'])
        return {'did': 'weather ' + ', '.join(f'{k} {v}' for k, v in changed.items()),
                'weather': {k: round(float(v), 3) for k, v in sorted(self.weather.items())},
                'vehicle_lights': lit.get('resolved')}

    def start_fault(self, a, t) -> dict:
        sid, f = a['sensor'], dict(a['fault'])
        if self.faults is None:
            if self.on_event is not None:            # Live: its sensor failures (the view, the recording)
                return self.live_fault(sid, f, t, a.get('for_s'))
            raise Failed('this run does not apply sensor faults')
        sensors = {s['id']: s for s in self.cfg['rig']['sensors']}
        if sid not in sensors:
            raise Failed(f'the rig has no sensor {sid!r}')
        from ..core import faults as F
        if not F.applies_to(f['kind'], sensors[sid]['type']):
            raise Failed(f'a {f["kind"]} fault cannot act on {sid} (a {sensors[sid]["type"]})')
        if f['kind'] in F.STREAM_FAULTS and f['kind'] != 'dropout':
            raise Failed(f'a {f["kind"]} fault acts on the data a stack or a device receives (stack tests, hardware '
                         'in the loop), not on a recorded drive\'s files')
        # from the next frame on: this frame is already recorded
        onset = t + 1.0 / float(self.cfg.get('fps', 20.0))
        got = self.faults.start(sid, f, onset, a.get('for_s'))
        return {'did': f'{f["kind"]} on {sid} from {onset:.3f} s' +
                (f' to {got["until_s"]:.3f} s' if 'until_s' in got else ''), 'fault': got, 'sensor': sid}

    LIVE_MODES = {'drop': 'none', 'freeze': 'freeze', 'dirt': 'black'}

    def live_fault(self, sid, f, t, for_s) -> dict:
        mode = self.LIVE_MODES.get(f['kind'])
        if mode is None:
            raise Failed(f'Live shows sensor failures as dropped, frozen or black frames: a {f["kind"]} fault is '
                         'applied in recorded drives only')
        self.on_event('fault_request', {'sensor': sid, 'mode': mode, 'seconds': float(for_s or 3600.0)})
        return {'did': f'{f["kind"]} on {sid} shown as {mode}', 'sensor': sid, 'mode': mode}

    # ------------------------------------------------------------------ the record
    def report(self) -> dict:
        out = {'schema': T.RULES_SCHEMA, 'context': self.cfg['trigger_rules'].get('context'), 'rules': self.rules,
               'signals': self.engine.signals, 'firings': self.firings,
               'fired': {r['id']: self.engine.counts.get(r['id'], 0) for r in self.rules},
               'spawned': self.spawned}
        if self.own_director and self.director is not None:
            out['criticality'] = self.criticality()
        return out

    def criticality(self):
        """core.criticality of the ego against the actors the rules added (an attached director's trajectories)."""
        from ..core import criticality as C
        np = self.np
        d = self.director
        ego = np.asarray(d.traj['ego'], float)
        actors = {k: np.asarray(v, float) for k, v in d.traj.items() if k != 'ego' and v}
        if not len(ego) or not actors:
            return None
        return C.summarise(ego, actors, 1.0 / float(self.cfg.get('fps', 20.0)), d.events)

    def trajectories(self):
        """The attached director's trajectories (trajectories.json of a drive without a scenario), or None."""
        if not self.own_director or self.director is None:
            return None
        return self.director.traj
