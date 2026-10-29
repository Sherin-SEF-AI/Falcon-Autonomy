"""adversary: the driving environment with learned traffic agents — the gym worker (workers/gym.py) with several
controlled road users that try to make a fixed ego fail, realistically (the definitions are core.advfeatures'; the
client and the multi-agent PPO are core.adversary, in the app's Python; docs/adversarial-traffic.md is the guide).

    python -m falcon_autonomy.workers adversary CONFIG.json

CONFIG.json: a gym config (falcon.gym_config/1, core.adversary.adversary_config writes it) with an "adversary"
block: ego ("follower": the director's lane follower | "none": constant speed, no braking | "autopilot": CARLA's
traffic manager | "agent": the client sends the ego's action — a Falcon policy folder acting in the app's Python),
action (lateral | discrete | continuous: the adversaries' actions), roles (the default roster, core.advfeatures.ROLES),
reward (weights over DEFAULT_ADV_REWARD), realism (limits over REALISM_LIMITS), ruleset (core.rules: a built-in id, a
file or null for the world's handedness), log_samples (the adversaries' and the traffic manager's accelerations and
jerks in each episode's line), log_adv_trace (each adversary's rule trace in it).

Commands on stdin, as the gym worker's (reset, step, close), with:
    {"cmd": "reset", "id": n, "seed": s, "options": {"roles": [...], "scenario": KEY | doc, "values": {...},
                                                     "ego_speed_kmh": v, "max_episode_s": t, "motion": m}}
        roles: the adversaries of this episode (learned agents); scenario instead: a scripted scenario of
        core.scenarios (its actors driven by the director — the baselines), judged by the same failure and realism
        measures
    {"cmd": "step", "id": n, "action": EGO ACTION | null, "adversaries": {id: action, ...}}
Replies as the gym worker's, and the observation payload also carries "adversaries": (N, N_ADV_FEATURES) float32,
one row per learned adversary in info["adversaries"]'s order; info["rewards"] {id: reward}, info["terms"] {id:
{term: value}}, info["failures"] (the step's new ego failures, each with its fault attribution).

An episode. The roles become a concrete scenario (core.advfeatures.setup_scenario) the director places like any
other: the adversaries are the scenario's actors, marked external (the director does not drive them). Each step the
director drives the ego (follower / none / autopilot) and any scripted actor, each learned adversary's action goes
through its actuator (lateral, discrete) or straight to the kinematic bicycle model (continuous), one tick, then the
facts: the ego's observation and infractions exactly as the gym worker's; for every tracked road user (the learned
adversaries, or a scripted scenario's vehicles) its pose, CARLA's lane under its centre, its dynamics, its contacts,
its time to collision and box gap with the ego, its rule-trace row; at an ego collision the fault attribution from both
histories; the adversaries' reward terms. The episode terminates as the ego's does (drivefeatures' terminate_on), or
when an adversary goes more than ADV_OFFROAD_END_M off the driving lanes ('adversary_off_road'); it is truncated at
the time limit (default 15 s) or when a learned adversary reaches the end of its lane path ('adversary_path_end').
At its end each tracked road user's rule trace is checked with core.rules.

Determinism: kinematic motion, no background traffic and an ego that is not CARLA's autopilot make an episode a
function of the seed, the options and the actions (the adversaries', and the ego's when the client drives it) —
which is what makes every failure replayable (core.adversary.replay). The autopilot ego runs on CARLA's physics and
its episodes are not repeatable.

Outputs: episodes.jsonl (the gym worker's line, plus "adversary": the roster, roles, failures with fault, realism
summaries, rule counts; "adversary_actions" when log_actions; "adversary_samples" and "tm_samples" when
log_samples; "adversary_traces" when log_adv_trace), run.json.
"""
from __future__ import annotations

import collections
import math
import time

from ..core import advfeatures as AF
from ..core import drivefeatures as DF
from ..core.provenance import read_json
from ..core import protocol as P
from .gym import BadCommand, Gym
from .common import Failed

PATH_M = 600.0                             # a learned adversary's lane path (the director's are 200–300 m)
PATH_END_M = 3.0                           # an adversary this close to its lane path's end truncates the episode
HIST_S = 2.6                               # seconds of history kept for fault attribution (> CUT_IN_S + a step)
HEAVY = ('adversary_actions', 'adversary_samples', 'adversary_traces', 'tm_samples')
EGO_DRIVERS = ('follower', 'none', 'autopilot', 'agent')


class HintedPath:
    """A director Path whose projection is tracked: the nearest point within [s − 15 m, s + 40 m] of the last one (a
    lane path through a town can loop back near its start; a global nearest point would jump along it)."""

    BACK_M, AHEAD_M = 15.0, 40.0

    def __init__(self, path):
        self.p = path
        self.pts, self.yaw, self.w, self.z, self.s = path.pts, path.yaw, path.w, path.z, path.s
        self.hint = 0.0

    def at(self, s, lateral=0.0):
        return self.p.at(s, lateral)

    def width(self, s):
        return self.p.width(s)

    def project(self, x, y) -> float:
        import numpy as np
        lo = int(np.searchsorted(self.s, self.hint - self.BACK_M))
        hi = max(int(np.searchsorted(self.s, self.hint + self.AHEAD_M)) + 1, lo + 1)
        d = np.hypot(self.pts[lo:hi, 0] - x, self.pts[lo:hi, 1] - y)
        self.hint = float(self.s[lo + int(np.argmin(d))])
        return self.hint


class Adversary(Gym):
    name = 'adversary'

    def __init__(self, cfg: dict, argv: list, cfg_path: str = ''):
        super().__init__(cfg, argv, cfg_path)
        self.acfg = dict(cfg.get('adversary') or {})
        self.ego_driver = self.acfg.get('ego', 'follower')
        self.adv_action = self.acfg.get('action', 'lateral')
        self.weights = dict(AF.DEFAULT_ADV_REWARD, **(self.acfg.get('reward') or {}))
        self.limits = dict(AF.REALISM_LIMITS, **(self.acfg.get('realism') or {}))
        self.ruleset = None

    def body(self) -> dict:
        if self.ego_driver not in EGO_DRIVERS:
            raise Failed(f'adversary.ego must be one of {EGO_DRIVERS}, not {self.ego_driver!r}')
        if self.adv_action not in AF.ACTION_KINDS:
            raise Failed(f'adversary.action must be one of {AF.ACTION_KINDS}, not {self.adv_action!r}')
        from ..core import rules as RU
        try:
            self.ruleset = RU.load_ruleset(self.acfg.get('ruleset'), self.cfg.get('handedness'))
        except RU.RulesError as e:
            raise Failed(str(e)) from None
        out = super().body()
        out['adversary'] = {'ego': self.ego_driver, 'action': self.adv_action,
                            'ruleset': {k: self.ruleset.get(k) for k in ('id', 'name', 'handedness', 'sha256')}}
        return out

    def ready_doc(self) -> dict:
        doc = super().ready_doc()
        act = {'kind': self.adv_action, 'keys': ['lateral', 'accel'] if self.adv_action != 'continuous'
               else ['steer', 'accel']}
        if self.adv_action == 'discrete':
            act.update(n=AF.n_discrete(), table=[list(m) for m in AF.MANOEUVRES])
        else:
            act.update(low=[-1.0, -1.0], high=[1.0, 1.0])
        doc['adversary'] = {'features': [list(f) for f in AF.ADV_FEATURES], 'advfeatures': AF.SCHEMA,
                            'action': act, 'ego': self.ego_driver, 'roles': sorted(AF.ROLES),
                            'reward_weights': self.weights, 'realism': self.limits,
                            'ruleset': {k: self.ruleset.get(k) for k in ('id', 'name', 'handedness')}}
        return doc

    # ------------------------------------------------------------------ episodes
    def episode_options(self, opts: dict) -> dict:
        keys = ('roles', 'scenario', 'values', 'motion', 'max_episode_s', 'ego_speed_kmh', 'route_m')
        unknown = sorted(set(opts) - set(keys))
        if unknown:
            raise BadCommand(f'unknown reset options {unknown} (allowed: {list(keys)})')
        base = {'roles': list(self.acfg.get('roles') or []), 'motion': self.cfg.get('motion', 'kinematic'),
                'max_episode_s': self.cfg.get('max_episode_s') or 15.0}
        out = dict(base, **opts)
        if out.get('motion', 'kinematic') not in ('kinematic', 'physics'):
            raise BadCommand(f'motion must be kinematic or physics, not {out.get("motion")!r}')
        if out.get('scenario') in ('', 'free'):
            out['scenario'] = None
        return out

    def resolve_scenario(self, o: dict, seed: int):
        from ..core import scenarios as SC
        driver = self.ego_driver
        scn = o.get('scenario')
        if scn is not None:
            if isinstance(scn, str):
                if scn not in SC.LIBRARY:
                    raise BadCommand(f'no built-in scenario {scn!r} (built-ins: {sorted(SC.LIBRARY)})')
                scn = SC.builtin(scn)
            if not isinstance(scn, dict):
                raise BadCommand('scenario must be a built-in key or a scenario document')
            problems = SC.validate(scn)
            if problems:
                raise BadCommand(f'scenario {scn.get("id")!r}: {"; ".join(problems)}')
            conc = SC.sample(scn, seed, o.get('values') or {})
            conc['mode'] = 'scripted'
            if o.get('max_episode_s'):
                conc['duration_s'] = float(o['max_episode_s'])
        else:
            try:
                conc = AF.setup_scenario(o.get('roles') or [], seed, o.get('ego_speed_kmh'),
                                         float(o.get('max_episode_s') or 20.0), driver)
            except ValueError as e:
                raise BadCommand(str(e)) from None
            conc['mode'] = 'learned'
        conc['ego'] = dict(conc.get('ego') or {}, driver=driver, route='straight')
        conc['motion'] = o.get('motion', 'kinematic')
        return conc

    # ------------------------------------------------------------------ the tracked road users
    def tracked_pose(self, a) -> dict:
        d = self.ep['director']
        tr = d.tf(a)
        k = d.kin.get(a.id)
        if k is not None:
            sp = float(k[4])
        else:
            v = d.vel(a)
            sp = math.hypot(v.x, v.y)
        bb = a.bounding_box.extent
        return {'x': tr.location.x, 'y': tr.location.y, 'z': tr.location.z, 'yaw': tr.rotation.yaw, 'speed': sp,
                'speed_mps': sp, 'length': 2.0 * bb.x, 'width': 2.0 * bb.y}

    def reset_extras(self, info: dict, arrays: list):
        ep = self.ep
        d = ep['director']
        conc = ep['scenario']
        mode = conc.get('mode', 'learned')
        roles = {a['id']: a.get('role') for a in conc.get('actors', [])}
        adv = collections.OrderedDict()
        for aid, st in d.actors.items():
            if st['walker'] or st['prop']:
                continue
            role = roles.get(aid) if mode == 'learned' else AF.SCRIPTED_ROLES.get((conc.get('id'), aid))
            spec = AF.ROLES.get(role) or {}
            w = st['path'].width(0.0)
            adv[aid] = {'actor': st['actor'], 'st': st, 'kind': st['kind'], 'role': role, 'learned': mode == 'learned',
                        'allow': set(spec.get('allow') or ()), 'budget_ww_s': float(spec.get('budget_ww_s', 0.0)),
                        'act': AF.Actuator(self.dt, st['lateral'], AF.LAT_LIMIT_LANES * w),
                        'dyn': AF.Dynamics(self.dt, self.limits), 'prev': [0.0, 0.0], 'best_s': 0.0,
                        'min_ttc': math.inf, 'min_gap': math.inf, 'actions': [], 'trace': [],
                        'hist': collections.deque(maxlen=int(HIST_S * self.fps) + 2), 'contact_last': {},
                        'off_open': False, 'n': {'fault_collisions': 0, 'shared_collisions': 0, 'contacts': 0,
                                                 'ego_caused_collisions': 0, 'off_road_s': 0.0, 'wrong_way_s': 0.0,
                                                 'wrong_way_over_s': 0.0, 'distance_m': 0.0, 'max_offroad_m': 0.0},
                        'terms': {t: 0.0 for t in AF.ADV_REWARD_TERMS}, 'ret': 0.0, 'accel': 0.0}
        d.external = {aid for aid, a in adv.items() if a['learned']}
        from .director import lane_path
        for aid in d.external:                     # a learned adversary's lane path: PATH_M long where the map allows
            st = d.actors[aid]
            p0 = st['path']
            wp = self.map.get_waypoint(self.carla.Location(x=float(p0.pts[0][0]), y=float(p0.pts[0][1]), z=p0.z))
            if wp is not None and len(p0.pts) > 1:
                back = abs(float(DF.wrap_deg(p0.yaw[0] - wp.transform.rotation.yaw))) > 90.0
                longer = lane_path(wp, PATH_M, backwards=back)
                if longer.s[-1] > p0.s[-1]:
                    p0 = longer
            st['path'] = HintedPath(p0)
        ep['adv'] = adv
        ep['ego_hist'] = collections.deque(maxlen=int(HIST_S * self.fps) + 2)
        ep['ego_dyn'] = AF.Dynamics(self.dt, self.limits)
        ep['failures'] = []
        ep['ego_accel'] = 0.0
        ep['tm_dyn'] = {}
        ep['tm_samples'] = {'a_long': [], 'a_lat': [], 'jerk': []}
        ep['adv_end'] = None
        ep['path_end'] = None
        ego_p = self.tracked_pose(ep['ego'])
        ego_p['accel'] = 0.0
        self.push_ego_hist(ego_p)
        for aid, a in adv.items():
            p = self.tracked_pose(a['actor'])
            a['dyn'].update(p['speed'], p['yaw'])
            a['pose'] = p
            a['lane'] = self.adv_lane(p)
            self.push_hist(a, p)
        obs = self.adv_observations(ego_p, 0.0)
        info = dict(info, mode=mode, adversaries=[aid for aid, a in adv.items() if a['learned']],
                    tracked=list(adv), roles={aid: a['role'] for aid, a in adv.items()},
                    kinds={aid: a['kind'] for aid, a in adv.items()}, ego_driver=self.ego_driver)
        return info, arrays + [('adversaries', obs)]

    def adv_lane(self, p) -> dict:
        """Lane facts of a road user: on a driving lane, its heading, the lateral offset, width, same-direction lanes
        beside it, junction, wrong way, off-road distance; and the rule-trace row's lane part."""
        row = self.lane_row(p)
        out = {'row': row, 'on_lane': bool(row[0]), 'junction': bool(row[1]) if row[0] else False,
               'lane_yaw': float(row[2]) if row[0] else math.nan, 'lat': float(row[3]) if row[0] else 0.0,
               'width': float(row[4]) if row[0] else 3.5, 'lanes_left': row[5] or 0, 'lanes_right': row[6] or 0,
               'offroad_m': 0.0}
        out['wrong_way'] = bool(out['on_lane'] and not out['junction'] and
                                abs(float(DF.wrap_deg(out['lane_yaw'] - p['yaw']))) > DF.WRONG_WAY_DEG)
        if not out['on_lane']:
            carla = self.carla
            loc = carla.Location(x=p['x'], y=p['y'], z=p.get('z', 0.0))
            near = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if near is not None:
                t = near.transform.location
                out['offroad_m'] = max(math.hypot(t.x - p['x'], t.y - p['y']) - near.lane_width / 2.0, 0.0)
        return out

    def push_hist(self, a, p):
        ln = a['lane']
        a['hist'].append((p['x'], p['y'], p['yaw'], p['speed'], ln['lane_yaw'], ln['wrong_way'], ln['on_lane']))

    def push_ego_hist(self, p, lf=None, lane_yaw=math.nan):
        lf = lf or {'off_road': False, 'wrong_way': False}
        self.ep['ego_hist'].append((p['x'], p['y'], p['yaw'], p['speed'], lane_yaw, bool(lf['wrong_way']),
                                    not lf['off_road']))
        self.ep['ego_pose'] = p

    @staticmethod
    def hist_doc(hist, length, width) -> dict:
        cols = list(zip(*hist))
        return {'x': list(cols[0]), 'y': list(cols[1]), 'yaw': list(cols[2]), 'speed': list(cols[3]),
                'lane_yaw': [float(v) for v in cols[4]], 'wrong_way': list(cols[5]), 'on_lane': list(cols[6]),
                'length': float(length), 'width': float(width)}

    def route_rel(self, p, ego_p) -> dict:
        route = self.ep['route']
        se, le, _ = route.project(ego_p['x'], ego_p['y'], self.ep.get('ego_s'), window=AF.REL_CAP_M + 60.0)
        sa, la, _ = route.project(p['x'], p['y'], se, window=AF.REL_CAP_M + 60.0)
        return {'s_rel': sa - se, 'lat': la, 'ego_lat': le}

    def adv_observations(self, ego_p, t):
        """(N_learned, N_ADV_FEATURES) float32: each learned adversary's observation (core.advfeatures.features)."""
        from ..core import criticality as C
        np = self.np
        ep = self.ep
        rows = []
        es = AF.state_row(t, ego_p)
        ego_box = C.box_corners(ego_p['x'], ego_p['y'], ego_p['yaw'], ego_p['length'], ego_p['width'])
        poses = {aid: a['pose'] for aid, a in ep['adv'].items()}
        for aid, a in ep['adv'].items():
            p = a['pose']
            ttc = C.ttc(es, AF.state_row(t, p), AF.TTC_CAP_S)
            gap = C.box_distance(ego_box, C.box_corners(p['x'], p['y'], p['yaw'], p['length'], p['width']))
            a['ttc'], a['gap'] = ttc, gap
            if not a['learned']:
                continue
            other, best = None, math.inf
            for bid, q in poses.items():
                if bid != aid:
                    dd = math.hypot(q['x'] - p['x'], q['y'] - p['y'])
                    if dd < best:
                        best, other = dd, (q['x'] - p['x'], q['y'] - p['y'])
            rows.append(AF.features(dict(p, accel=a['accel']), ego_p, a['lane'], self.route_rel(p, ego_p), other,
                                    t / max(ep['max_episode_s'], 1e-9), a['prev'], ttc, gap,
                                    a['budget_ww_s'] > 0, a['kind']))
        return np.asarray(rows, np.float32).reshape(-1, AF.N_ADV_FEATURES)

    # ------------------------------------------------------------------ a step
    def drive_adversary(self, a, action):
        """Apply one learned adversary's action for the coming tick (core.advfeatures: actuator and pure pursuit, or
        the kinematic bicycle model)."""
        from .director import follow
        d = self.ep['director']
        actor, st = a['actor'], a['st']
        try:
            c0, c1, applied = AF.command(action, a['kind'], self.adv_action)
        except ValueError as e:
            raise BadCommand(f'adversary {a["role"]}: {e}') from None
        a['prev'] = applied
        a['actions'].append([round(float(v), 6) for v in applied] if self.adv_action != 'discrete' else
                            int(self.np.asarray(action).reshape(-1)[0]))
        v = a['pose']['speed']
        if st['path'].project(a['pose']['x'], a['pose']['y']) >= st['path'].s[-1] - PATH_END_M:
            self.ep['path_end'] = self.ep.get('path_end') or a['role'] or 'adversary'
        if self.adv_action == 'continuous':
            amax = AF.KIND_TOP_ACCEL[AF.kind_of(a['kind'])]
            u = {'throttle': max(c1, 0.0) * amax / 3.0, 'steer': c0, 'brake': max(-c1, 0.0)}
        else:
            lat, acc = a['act'].update(c0, c1)
            st['lateral'] = lat
            top = AF.KIND_TOP_KMH[AF.kind_of(a['kind'])] / 3.6
            if st.get('top') is not None:
                top = min(top, st['top'])
            v_next = min(max(v + acc * self.dt, 0.0), top)
            if v_next <= 0.0 and acc < 0:
                a['act'].a = 0.0
            steer = follow(actor, st['path'], lat, max(v_next, 0.5), self.dt, st, d.tf(actor), d.vel(actor))['steer']
            u = AF.accel_to_control(v, v_next, self.dt, steer)
        if d.kinematic:
            d.kin_step(actor, u)
        else:
            actor.apply_control(self.carla.VehicleControl(throttle=float(u['throttle']), steer=float(u['steer']),
                                                          brake=float(u['brake'])))

    def step(self, cmd):
        ep = self.ep
        if ep is None:
            raise BadCommand('no episode: reset first')
        if ep['done']:
            raise BadCommand('the episode has ended: reset first')
        np = self.np
        d, ego = ep['director'], ep['ego']
        u_ego, applied = None, None
        if self.ego_driver == 'agent':
            try:
                u_ego, applied = DF.to_control(cmd.get('action'), self.action_kind)
            except ValueError as e:
                raise BadCommand(f'ego: {e}') from None
        acts = cmd.get('adversaries') or {}
        if not isinstance(acts, dict):
            raise BadCommand('adversaries must be an object {id: action}')
        learned = [aid for aid, a in ep['adv'].items() if a['learned']]
        missing = [aid for aid in learned if aid not in acts]
        extra = sorted(set(acts) - set(learned))
        if missing or extra:
            raise BadCommand(f'adversary actions: missing {missing}, unknown {extra}')
        for aid in learned:                      # validated before anything moves
            try:
                AF.command(acts[aid], ep['adv'][aid]['kind'], self.adv_action)
            except ValueError as e:
                raise BadCommand(f'adversary {aid}: {e}') from None
        t0 = time.monotonic()
        world, carla = self.world, self.carla
        d.step(ep['k'] * self.dt)
        for aid in learned:
            self.drive_adversary(ep['adv'][aid], acts[aid])
        if u_ego is not None:
            if d.kinematic:
                d.kin_step(ego, u_ego)
            else:
                ego.apply_control(carla.VehicleControl(throttle=u_ego['throttle'], steer=u_ego['steer'],
                                                       brake=u_ego['brake']))
        if applied is None:
            c = d.ego_control or {}
            applied = [float(c.get('steer', 0.0)), float(c.get('throttle', 0.0)) - float(c.get('brake', 0.0))]
        frame = world.tick()
        self.alive()
        d.sync(frame)
        ep['k'] += 1
        k = ep['k']
        sim_t = k * self.dt
        d.record(sim_t)
        steer_rate = (applied[0] - ep['prev_action'][0]) / self.dt
        ep['prev_action'] = applied
        state, facts, arrays, rows = self.observe(frame)
        pose = self.ego_pose()
        lf = self.lane_facts(pose)
        front = self.front()
        travel = facts['speed'] * self.dt
        ep['travel'] += travel
        ds = max(min(facts['s'] - ep['s_prev'], 5.0), -5.0)
        ep['s_prev'] = facts['s']
        ep['s_max'] = max(ep['s_max'], facts['s'])
        ep['ego_s'] = facts['s']
        inside = {sid: True for sid, cx, cy, yaw, ex, ey in self.stops if DF.in_box(pose['x'], pose['y'], cx, cy,
                                                                                    yaw, ex, ey)}
        mf = dict(lf, contacts=self.contacts(pose, rows), red_light=self.red_lights(ep['front_prev'], front),
                  stop_inside=inside, lateral=facts['lateral'], speed=facts['speed'], s=facts['s'],
                  route_length=facts['route_length'], travel_m=travel)
        ep['front_prev'] = front
        new, reason = ep['monitor'].update(k, sim_t, pose, mf)
        q = dict(lf, ds=ds, speed=facts['speed'], target=ep['target'], lateral=facts['lateral'],
                 lane_width=facts['lane_width'], heading_err=facts['heading_err'], jerk=facts['jerk'],
                 steer_rate=steer_rate)
        eterms = DF.reward_terms(q, new, self.cfg.get('reward'), self.dt)
        ep['ret'] += float(sum(eterms[t] for t in DF.REWARD_TERMS))
        for t_ in DF.REWARD_TERMS:
            ep['terms'][t_] += eterms[t_]
        if self.ego_driver == 'agent' and self.cfg.get('log_actions', True):
            ep['actions'].append(applied if self.action_kind == 'continuous' else
                                 int(np.asarray(cmd.get('action')).reshape(-1)[0]))
        if self.cfg.get('log_trace', True):
            ep['trace'].append(self.trace_row(sim_t, pose, state, lf))
        # ---- the ego's and the tracked road users' state
        ego_p = dict(self.tracked_pose(ego), accel=facts['accel'])
        ego_lane = self.lane_row(ego_p)
        self.push_ego_hist(ego_p, lf, float(ego_lane[2]) if ego_lane[0] else math.nan)
        ep['ego_dyn'].update(ego_p['speed'], ego_p['yaw'])
        per = {}
        for aid, a in ep['adv'].items():               # every pose first: the trace rows read the others'
            a['pose'] = self.tracked_pose(a['actor'])
            a['lane'] = self.adv_lane(a['pose'])
        for aid, a in ep['adv'].items():
            p, ln = a['pose'], a['lane']
            a['n']['distance_m'] += p['speed'] * self.dt
            al, _alat, _j, over = a['dyn'].update(p['speed'], p['yaw'])
            a['accel'] = al
            self.push_hist(a, p)
            a['trace'].append(self.adv_trace_row(sim_t, aid, p, ln))
            q_ = {'off_road': not ln['on_lane'], 'off_road_onset': (not ln['on_lane']) and not a['off_open'],
                  'dyn_over': len(over)}
            a['off_open'] = not ln['on_lane']
            if not ln['on_lane']:
                a['n']['off_road_s'] += self.dt
                a['n']['max_offroad_m'] = max(a['n']['max_offroad_m'], ln['offroad_m'])
                if ln['offroad_m'] > AF.ADV_OFFROAD_END_M and ep['adv_end'] is None:
                    ep['adv_end'] = aid
            if ln['wrong_way']:
                a['n']['wrong_way_s'] += self.dt
                if a['n']['wrong_way_s'] > a['budget_ww_s'] + 1e-9:
                    a['n']['wrong_way_over_s'] += self.dt
                    q_['wrong_way_over'] = True
            per[aid] = q_
        obs = self.adv_observations(ego_p, sim_t)
        for aid, a in ep['adv'].items():
            s_now = AF.ttc_score(a['ttc'])
            per[aid]['ttc_gain'] = max(s_now - a['best_s'], 0.0)
            a['best_s'] = max(a['best_s'], s_now)
            a['min_ttc'] = min(a['min_ttc'], a['ttc'])
            a['min_gap'] = min(a['min_gap'], a['gap'])
        # ---- contacts between tracked road users and others (not the ego): realism
        self.adv_contacts(rows, sim_t, per)
        # ---- the ego's failures: collisions with fault, failure infractions
        fails = self.ego_failures(new, k, sim_t, ego_p, per)
        # ---- terminations
        if reason is None and ep['adv_end'] is not None:
            reason = 'adversary_off_road'
        terminated = reason is not None
        truncated = not terminated and (k >= ep['max_steps'] or bool(ep.get('path_end')))
        if terminated or truncated:
            self.rule_penalties(per)
        terms = {aid: AF.adversary_terms(per[aid], self.weights, self.dt) for aid in ep['adv']}
        learned_terms = {aid: terms[aid] for aid in learned}
        rewards = AF.mix_team(learned_terms, self.weights)
        for aid, a in ep['adv'].items():
            for t_ in AF.ADV_REWARD_TERMS:
                a['terms'][t_] += terms[aid][t_]
            a['ret'] += rewards.get(aid, 0.0)
        self.tm_step(rows)
        info = {'episode': ep['n'], 'step': k, 'sim_t': round(sim_t, 4), 'infractions': new,
                'progress_m': round(facts['s'], 4), 'ego': {kk: round(float(v), 4) for kk, v in pose.items()},
                'action_applied': applied, 'rewards': rewards,
                'terms': {aid: learned_terms[aid] for aid in learned}, 'failures': fails,
                'adversaries': learned, 'off_road': lf['off_road'], 'wrong_way': lf['wrong_way'],
                'route_completion': round(min(max(facts['s'] / max(facts['route_length'], 1e-9), 0.0), 1.0), 4),
                'termination': reason if terminated else (('time_limit' if k >= ep['max_steps'] else
                                                             'adversary_path_end') if truncated else None),
                'min_ttc': {aid: (round(a['min_ttc'], 4) if math.isfinite(a['min_ttc']) else None)
                            for aid, a in ep['adv'].items()}}
        if getattr(self, 'watch', None) is not None:
            info['odd'] = self.watch.step(sim_t, ego, facts['speed'])
        self.n_steps += 1
        self.step_wall += time.monotonic() - t0
        if terminated or truncated:
            info['episode_summary'] = self.end_episode(info['termination'], terminated, truncated)
        reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
        self.send({'t': 'obs', 'id': cmd.get('id'), 'kind': 'step', 'reward': reward, 'terminated': terminated,
                   'truncated': truncated, 'info': info}, [('state', state)] + arrays + [('adversaries', obs)])

    def adv_trace_row(self, sim_t, aid, p, ln) -> list:
        """A rule-trace row (core.rules.TRACE_COLUMNS) of a tracked road user: its pose and lane, its lead (the
        nearest road user ahead in its corridor: |lateral| < half widths + 0.3 m, heading within 60°, up to 80 m),
        no vulnerable road user or light (not tracked for adversaries), CARLA's speed limit for it."""
        r = lambda v, nd=3: None if v is None or not math.isfinite(float(v)) else round(float(v), nd)   # noqa: E731
        ep = self.ep
        lead = None
        others = [ep['ego_pose']] + [b['pose'] for bid, b in ep['adv'].items() if bid != aid]
        for o in others:
            dx, dy = AF.to_frame(o['x'] - p['x'], o['y'] - p['y'], p['yaw'])
            rel = math.radians(o['yaw'] - p['yaw'])
            if dx <= 0 or abs(float(DF.wrap_deg(o['yaw'] - p['yaw']))) > 60.0:
                continue
            across = abs(o['length'] / 2 * math.sin(rel)) + abs(o['width'] / 2 * math.cos(rel))
            along = abs(o['length'] / 2 * math.cos(rel)) + abs(o['width'] / 2 * math.sin(rel))
            if abs(dy) - across >= p['width'] / 2 + 0.3:
                continue
            gap = max(dx - p['length'] / 2 - along, 0.0)
            if gap <= 80.0 and (lead is None or gap < lead):
                lead = gap
        return [r(sim_t, 4), r(p['x']), r(p['y']), r(p['yaw']), r(p['speed'])] + ln['row'] + \
            [r(lead), None, None, None, self.speed_limit(ep['adv'][aid]['actor'])]

    def adv_contacts(self, rows, sim_t, per):
        """Box contacts of each tracked road user with any road user but the ego (rearmed after out of contact for
        drivefeatures.COLLISION_REARM_S): counted against its realism."""
        from ..core import criticality as C
        ep = self.ep
        ego_id = ep['ego'].id
        mine = {a['actor'].id: aid for aid, a in ep['adv'].items()}
        for aid, a in ep['adv'].items():
            p = a['pose']
            A = C.box_corners(p['x'], p['y'], p['yaw'], p['length'], p['width'])
            n_new = 0
            for r in rows:
                oid = r['id']
                if oid in (ego_id, a['actor'].id) or math.hypot(r['x'] - p['x'], r['y'] - p['y']) > 15.0:
                    continue
                if oid in mine:
                    q = ep['adv'][mine[oid]]['pose']
                    B = C.box_corners(q['x'], q['y'], q['yaw'], q['length'], q['width'])
                else:
                    B = C.box_corners(r['x'], r['y'], r['yaw'], 2.0 * r['extent'][0], 2.0 * r['extent'][1])
                if C.box_distance(A, B) > 0.0:
                    continue
                last = a['contact_last'].get(oid)
                if last is None or sim_t - last > DF.COLLISION_REARM_S + 1e-9:
                    n_new += 1
                a['contact_last'][oid] = sim_t
            a['n']['contacts'] += n_new
            per[aid]['adv_contact'] = n_new

    def ego_failures(self, new, k, sim_t, ego_p, per) -> list:
        """The step's new ego failures: a collision with a tracked road user (with its fault attribution) or with
        anything else, and the failure infractions (core.advfeatures.EGO_FAILURE_INFRACTIONS), credited."""
        ep = self.ep
        by_id = {a['actor'].id: aid for aid, a in ep['adv'].items()}
        out = []
        ego = ep['ego']
        for e in new:
            kind = e['kind']
            rec = None
            if kind == 'collision_vehicle' and e['actors'] and e['actors'][0] in by_id:
                aid = by_id[e['actors'][0]]
                a = ep['adv'][aid]
                bb = ego.bounding_box.extent
                f = AF.attribute_fault(self.hist_doc(ep['ego_hist'], 2 * bb.x, 2 * bb.y),
                                       self.hist_doc(a['hist'], a['pose']['length'], a['pose']['width']), self.dt)
                fault = {'a': 'ego', 'b': aid, 'shared': 'shared'}[f['at_fault']]
                if fault == 'ego':
                    per[aid]['ego_collision'] = per[aid].get('ego_collision', 0) + 1
                    a['n']['ego_caused_collisions'] += 1
                elif fault == 'shared':
                    per[aid]['shared_fault'] = per[aid].get('shared_fault', 0) + 1
                    a['n']['shared_collisions'] += 1
                else:
                    per[aid]['adv_fault'] = per[aid].get('adv_fault', 0) + 1
                    a['n']['fault_collisions'] += 1
                rec = {'kind': 'collision', 'actor': aid, 'fault': fault, 'rule': f['rule'], 'detail': f['detail'],
                       'failure': fault == 'ego'}
            elif kind in DF.COLLISION_KINDS:
                rec = {'kind': kind, 'actor': None, 'fault': None, 'rule': None, 'detail': {},
                       'failure': kind in AF.EGO_FAILURE_INFRACTIONS}
            elif kind in AF.EGO_FAILURE_INFRACTIONS:
                rec = {'kind': kind, 'actor': None, 'fault': 'ego', 'rule': None, 'detail': {}, 'failure': True}
                near = [aid for aid, a in ep['adv'].items() if a['gap'] <= AF.CREDIT_RADIUS_M]
                for aid in near:
                    per[aid]['ego_infractions'] = per[aid].get('ego_infractions', 0) + 1
                if near:
                    rec['actor'] = min(near, key=lambda x: ep['adv'][x]['gap'])
                rec['credited'] = near
            if rec is None:
                continue
            rec.update(step=k, sim_t=round(sim_t, 4), ego={kk: round(float(v), 4) for kk, v in e['ego'].items()},
                       infraction=len(ep['monitor'].events) - len(new) + new.index(e))
            if rec['actor'] is not None:
                a = ep['adv'][rec['actor']]
                dx, dy = AF.to_frame(a['pose']['x'] - ego_p['x'], a['pose']['y'] - ego_p['y'], ego_p['yaw'])
                rec.update(role=a['role'], kind_of_actor=a['kind'], sector=AF.sector(dx, dy),
                           gap_m=round(a['gap'], 3))
            ep['failures'].append(rec)
            out.append(rec)
        return out

    def rule_penalties(self, per):
        """At the episode's end: core.rules on each tracked road user's trace; violations of rules its role does not
        allow beyond RULE_BUDGET are penalised."""
        from ..core import rules as RU
        from ..core.rules import TRACE_COLUMNS
        ep = self.ep
        for aid, a in ep['adv'].items():
            a['rules'] = None
            if len(a['trace']) < 2:
                continue
            e = {'trace': {'columns': list(TRACE_COLUMNS), 'rows': a['trace'],
                           'ego': {'length': round(a['pose']['length'], 3), 'width': round(a['pose']['width'], 3)},
                           'handedness': self.cfg.get('handedness')}}
            try:
                res = RU.check(RU.trace_from_episode(e), self.ruleset)
            except (RU.RulesError, ValueError, ZeroDivisionError) as ex:
                self.note(f'rules on {aid}: {ex}')
                continue
            counts = {r['id']: r['count'] for r in res['rules'] if r['state'] != 'not evaluable'}
            charged = sum(c for rid, c in counts.items() if rid not in a['allow'])
            a['rules'] = {'counts': counts, 'charged': charged, 'over': max(charged - AF.RULE_BUDGET, 0),
                          'distance_km': res['distance_km'],
                          'not_evaluable': res['totals']['not_evaluable']}
            per[aid]['rules_over'] = a['rules']['over']

    def tm_step(self, rows):
        """Background vehicles (CARLA's traffic manager): their accelerations and jerks, the realism reference."""
        ep = self.ep
        if not self.background or not self.acfg.get('log_samples'):
            return
        mine = {a['actor'].id for a in ep['adv'].values()} | {ep['ego'].id}
        for r in rows:
            if r['id'] in mine or not str(r.get('type_id', '')).startswith('vehicle.'):
                continue
            if r.get('role', '').startswith('falcon_scenario'):
                continue
            dyn = ep['tm_dyn'].get(r['id'])
            if dyn is None:
                dyn = ep['tm_dyn'][r['id']] = AF.Dynamics(self.dt, self.limits)
            n0 = len(dyn.samples['a_long'])
            dyn.update(math.hypot(r['vx'], r['vy']), r['yaw'])
            for key in ('a_long', 'a_lat', 'jerk'):
                ep['tm_samples'][key] += dyn.samples[key][n0:]

    # ------------------------------------------------------------------ the episode's line
    def summary_extras(self, summ: dict) -> dict:
        ep = self.ep
        if 'adv' not in ep:
            return summ
        adv = {}
        for aid, a in ep['adv'].items():
            n = dict(a['n'])
            n['dynamics_over_s'] = round(sum(a['dyn'].over_s.values()), 4)
            n['dynamics_over_by_limit_s'] = {kk: round(v, 4) for kk, v in a['dyn'].over_s.items()}
            n['rules_over'] = (a.get('rules') or {}).get('over', 0)
            n = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in n.items()}
            n['realistic'] = AF.realism_ok(n)
            adv[aid] = {'role': a['role'], 'kind': a['kind'], 'learned': a['learned'], 'realism': n,
                        'rules': a.get('rules'), 'return': round(a['ret'], 6),
                        'reward_terms': {t: round(v, 6) for t, v in a['terms'].items()},
                        'min_ttc_s': round(a['min_ttc'], 4) if math.isfinite(a['min_ttc']) else None,
                        'min_gap_m': round(a['min_gap'], 4) if math.isfinite(a['min_gap']) else None,
                        'actor_id': int(a['actor'].id), 'blueprint': a['actor'].type_id}
        realistic = all(v['realism']['realistic'] for v in adv.values())
        fails = [dict(f, realistic=realistic) for f in ep['failures']]
        summ['adversary'] = {'mode': ep['scenario'].get('mode', 'learned'), 'roles': ep['scenario'].get('roles'),
                             'ego_driver': self.ego_driver, 'action': self.adv_action, 'actors': adv,
                             'order': [aid for aid, a in ep['adv'].items() if a['learned']],
                             'failures': fails, 'failed': any(f['failure'] for f in fails),
                             'realistic': realistic,
                             'realistic_failure': realistic and any(f['failure'] for f in fails),
                             'ego_dynamics_over_s': round(sum(ep['ego_dyn'].over_s.values()), 4),
                             'scenario': ep['scenario'] if ep['scenario'].get('mode') == 'learned' else None}
        if self.cfg.get('log_actions', True):
            summ['adversary_actions'] = {aid: a['actions'] for aid, a in ep['adv'].items() if a['learned']}
        if self.acfg.get('log_samples'):
            summ['adversary_samples'] = {aid: a['dyn'].samples for aid, a in ep['adv'].items()}
            summ['tm_samples'] = ep['tm_samples']
        if self.acfg.get('log_adv_trace'):
            from ..core.rules import TRACE_COLUMNS
            summ['adversary_traces'] = {aid: {'columns': list(TRACE_COLUMNS), 'rows': a['trace'],
                                              'ego': {'length': round(a['pose']['length'], 3),
                                                      'width': round(a['pose']['width'], 3)},
                                              'handedness': self.cfg.get('handedness')}
                                        for aid, a in ep['adv'].items()}
        return summ

    def end_episode(self, reason, terminated=False, truncated=False) -> dict:
        res = super().end_episode(reason, terminated, truncated)
        return {k: v for k, v in res.items() if k not in HEAVY}


def main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='falcon worker adversary')
    ap.add_argument('config', help='adversary config JSON (falcon_autonomy.core.adversary.adversary_config)')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return Adversary(cfg, argv, a.config).execute()
