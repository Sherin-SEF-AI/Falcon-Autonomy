"""The driving environment's definitions, shared by both sides (numpy only; importable from CARLA's Python and the
app's): the route, the observation vector, the actions, the reward terms and the infractions. The environment worker
(workers/gym.py, in CARLA's Python) and behaviour cloning from recorded drives (core.imitation, in the app's Python)
build observations with the SAME code below, so a policy trained on one sees exactly the other's features.

Frames and units: CARLA's throughout — x forward, y right, z up (LEFT-handed), metres, seconds; yaw in degrees,
positive turning right (clockwise seen from above). A heading (cos yaw, sin yaw) has its right at (−sin yaw, cos yaw).

The route
---------
A Route is the lane-centre polyline the ego is to follow (the environment: the director's lane path from the start,
straight-most through junctions; behaviour cloning: lanes.json's centre points matched to the lanes the drive went
through), with the lane width at each point. Route.project(x, y) gives the arc length s of the nearest point on the
polyline (segments, not vertices) and the signed lateral offset (+ right). Headings along the route are the segments'
own directions; curvature(s) = Δheading / Δs over ±CURV_HALF_M (1/m, + turning right). Projection is windowed around
the previous step's s (±window m), so a route that comes back near itself is not jumped along.

The observation vector (FEATURES, float32, in this order; raw units, no normalisation — policies normalise)
----------------------------------------------------------------------------------------------------------
Sentinels: a quantity that does not exist is given its cap and a 0 presence flag, never NaN.

    speed_mps            the ego's speed
    target_speed_mps     the speed the episode asks for (env: the scenario's ego speed, or target_speed_kmh; recorded
                         drives: the scenario's ego speed, else the option, else the drive's median moving speed —
                         heuristic, the autopilot's own target is not recorded)
    accel_mps2           (speed − previous step's speed) / dt; 0 at the first step
    lateral_offset_m     signed distance from the route's centre line (+ right)
    heading_error_rad    ego yaw − route heading at the projection, wrapped to (−π, π] (+ turned right)
    lane_width_m         the route's lane width at the projection
    curvature_10m/25m/50m  route curvature at s + 10, 25, 50 m (1/m, + right)
    lead_present         1 when a vehicle (any vehicle.* category but bicycles) is ahead in the corridor
    lead_gap_m           bumper to bumper along the route (the route distance less both boxes' half extents along it,
                         ≥ 0); LEAD_RANGE_M when none. Corridor: |lateral| − the actor's half extent across the route
                         < ego half width + LEAD_MARGIN_M
    lead_closing_mps     ego speed along the route − the lead's velocity along the route (+ closing); 0 when none
    lead_ttc_s           gap / closing when closing > 0.1 m/s, else TTC_CAP_S (capped at TTC_CAP_S)
    vru_present          1 when a pedestrian (human.*) or cyclist (vehicle.bicycle) is in the path ahead or entering it:
                         within VRU_RANGE_M ahead and |lateral| < ego half width + VRU_MARGIN_M + its half extent
                         across,
                         now or — at constant lateral velocity — when the ego reaches it (≤ VRU_HORIZON_S), or crossing
    vru_ahead_m          its distance ahead (box to box along the route, ≥ 0); VRU_RANGE_M when none
    vru_lateral_m        its lateral offset from the route (+ right); 0 when none
    vru_lat_speed_mps    its velocity across the route (+ moving right); 0 when none
    vru_ttc_s            vru_ahead_m / the ego's speed along the route, capped at TTC_CAP_S
    tl_known             1 when traffic-light states are known (the environment), 0 when not (recorded drives do not
                         record them); the tl_* fields below are then 0 and TL_RANGE_M
    tl_red, tl_yellow, tl_green   the state of the first light whose stop line is on the route ahead (one-hot)
    tl_distance_m        from the ego's front to that stop line along the route; TL_RANGE_M when none within range
    route_progress       s / route length, 0–1
    route_remaining_m    route length − s, capped at ROUTE_REMAINING_CAP_M
    prev_steer, prev_accel   the action applied at the previous step (0, 0 after a reset); recorded drives: the
                         control in effect during the tick that produced this step

Optional observations: "bev", the BEV raster of core.worldmodel.bev_raster (the same function `falcon derive
worldmodel` uses) at BEV_DEFAULT unless configured, as 8 bit planes (8, H, W) uint8 0/1 in worldmodel.BEV_CHANNELS
order; "camera", a low-resolution RGB image (H, W, 3) uint8.

Actions
-------
Continuous: [steer, accel] ∈ [−1, 1]²; steer is CARLA's (−1 full left … +1 full right); accel ≥ 0 is throttle = accel,
brake 0; accel < 0 is brake = −accel, throttle 0; hand brake and reverse are never used. Out-of-range values are
clipped; a non-finite value is refused. Discrete: an index into DISCRETE_ACTIONS (steer × accel grid below).

Rewards (each term logged separately; the reward is their sum; weights in DEFAULT_REWARD are heuristic, not tuned)
-------
    progress        + w · Δs                     route metres gained this step (negative going back)
    speed           − w · k · |v − v*| / max(v*, 1) · dt, k = OVERSPEED_FACTOR above v*, else 1 (finishing a route
                    early escapes per-step costs: with k = 1, PPO on the fake learnt to drive at twice the target)
    lane_offset     − w · min((lateral / (lane width / 2))², 4) · dt
    heading         − w · min(|heading error|, π/2) · dt
    jerk            − w · min(|jerk| / JERK_REF, 10) · dt      jerk = Δaccel / dt (longitudinal, from the speed)
    steer_rate      − w · min(|Δsteer| / dt / STEER_RATE_REF, 10) · dt
    collision       − w per new vehicle or static collision; − w_ped per new pedestrian collision
    off_road        − w_onset per off-road onset − w · dt while off the road (the one-off part: without it, ending an
                    episode early by leaving the road escapes the per-step penalties — measured with PPO on the fake)
    wrong_way       − w · dt while in a lane of the opposite direction
    outside_lanes   − w · dt while on a driving lane that is not one of the route's (and not wrong way)
    red_light, stop_sign, route_deviation, agent_blocked    − w per infraction

Infractions (INFRACTIONS; each recorded once at its onset with step, sim_t, the actor ids involved and the ego pose;
the condition kinds also with the step and distance at which they ended)
-----------
See INFRACTIONS below for the exact definitions and InfractionMonitor for the bookkeeping. Termination happens on the
kinds listed in a config's terminate_on (default DEFAULT_TERMINATE_ON) and when the route is complete; truncation at
the time limit.

Limits: lateral and heading are relative to the route's lane centre, not the lane the ego is in after a lane change;
the corridor tests use the actors' boxes at their centres' projections (a long vehicle on a sharp bend is approximate);
traffic-light states are known only in the environment.
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA = 'falcon.drivefeatures/1'

LEAD_RANGE_M = 60.0
LEAD_MARGIN_M = 0.3
VRU_RANGE_M = 40.0
VRU_MARGIN_M = 0.5
VRU_HORIZON_S = 3.0
TL_RANGE_M = 60.0
TTC_CAP_S = 10.0
CURV_AT_M = (10.0, 25.0, 50.0)
CURV_HALF_M = 5.0
ROUTE_REMAINING_CAP_M = 500.0
JERK_REF = 2.0                 # m/s³: heuristic comfort scale (longitudinal jerk of ~2 m/s³ is commonly called mild)
STEER_RATE_REF = 1.0           # steer units per second: heuristic scale
STOP_SPEED_MPS = 0.1           # a stop at a stop sign: the speed falls below this inside its area (heuristic)
OVERSPEED_FACTOR = 4.0         # the speed term's weight above the target, relative to below it (heuristic)

FEATURES = (
    ('speed_mps', 'm/s', 'the ego\'s speed'),
    ('target_speed_mps', 'm/s', 'the speed the episode asks for'),
    ('accel_mps2', 'm/s²', '(speed − previous speed) / dt'),
    ('lateral_offset_m', 'm', 'signed offset from the route centre (+ right)'),
    ('heading_error_rad', 'rad', 'ego yaw − route heading (+ right)'),
    ('lane_width_m', 'm', 'route lane width'),
    ('curvature_10m', '1/m', 'route curvature 10 m ahead (+ right)'),
    ('curvature_25m', '1/m', 'route curvature 25 m ahead (+ right)'),
    ('curvature_50m', '1/m', 'route curvature 50 m ahead (+ right)'),
    ('lead_present', '', '1: a vehicle ahead in the corridor'),
    ('lead_gap_m', 'm', 'bumper-to-bumper gap along the route (cap when none)'),
    ('lead_closing_mps', 'm/s', 'closing speed along the route (+ closing)'),
    ('lead_ttc_s', 's', 'gap / closing (cap when not closing)'),
    ('vru_present', '', '1: a pedestrian or cyclist in or entering the path ahead'),
    ('vru_ahead_m', 'm', 'its distance ahead along the route (cap when none)'),
    ('vru_lateral_m', 'm', 'its lateral offset from the route (+ right)'),
    ('vru_lat_speed_mps', 'm/s', 'its velocity across the route (+ right)'),
    ('vru_ttc_s', 's', 'its distance / ego speed along the route (cap)'),
    ('tl_known', '', '1: traffic-light states are known'),
    ('tl_red', '', 'the next light on the route is red'),
    ('tl_yellow', '', 'the next light on the route is yellow'),
    ('tl_green', '', 'the next light on the route is green'),
    ('tl_distance_m', 'm', 'ego front to its stop line along the route (cap)'),
    ('route_progress', '', 's / route length'),
    ('route_remaining_m', 'm', 'route length − s (capped)'),
    ('prev_steer', '', 'previous action\'s steer'),
    ('prev_accel', '', 'previous action\'s accel (throttle − brake)'),
)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
N_FEATURES = len(FEATURES)
FI = {name: k for k, name in enumerate(FEATURE_NAMES)}

# ---------------------------------------------------------------------------------------------------- actions
ACTION_KEYS = ('steer', 'accel')
DISCRETE_STEER = (-0.6, -0.2, 0.0, 0.2, 0.6)
DISCRETE_ACCEL = (-1.0, -0.3, 0.0, 0.3, 0.7)
DISCRETE_ACTIONS = tuple((s, a) for s in DISCRETE_STEER for a in DISCRETE_ACCEL)    # index = 5 · i_steer + i_accel

# ---------------------------------------------------------------------------------------------------- rewards
REWARD_TERMS = ('progress', 'speed', 'lane_offset', 'heading', 'jerk', 'steer_rate', 'collision', 'off_road',
                'wrong_way', 'outside_lanes', 'red_light', 'stop_sign', 'route_deviation', 'agent_blocked')
DEFAULT_REWARD = {                 # heuristic weights (not tuned): per metre, per second or per infraction
    'progress': 0.1, 'speed': 0.2, 'lane_offset': 0.3, 'heading': 0.2, 'jerk': 0.02, 'steer_rate': 0.02,
    'collision': 50.0, 'collision_pedestrian': 100.0, 'off_road': 5.0, 'off_road_onset': 20.0, 'wrong_way': 2.0,
    'outside_lanes': 0.5,
    'red_light': 20.0, 'stop_sign': 10.0, 'route_deviation': 20.0, 'agent_blocked': 10.0,
}

# ---------------------------------------------------------------------------------------------------- infractions
INFRACTIONS = {
    'collision_vehicle': 'the ego\'s box overlaps a vehicle\'s (any vehicle.*, two-wheelers included), or the '
                         'collision sensor reports one; once per actor until out of contact for COLLISION_REARM_S',
    'collision_pedestrian': 'as collision_vehicle with a walker (walker.*)',
    'collision_static': 'the collision sensor reports a hit with anything that is not a vehicle or walker (map '
                        'geometry, props, poles); kinematic motion has no physics, so it is detected only when CARLA '
                        'reports it',
    'red_light': 'the ego\'s front centre crosses a stop line of a traffic light that is red, driving in the stop '
                 'line\'s lane direction (within 90°); yellow is not an infraction',
    'stop_sign': 'the ego\'s centre enters and leaves a stop sign\'s trigger volume without its speed falling below '
                 'STOP_SPEED_MPS inside it',
    'off_road': 'the ego\'s centre is on no driving lane (CARLA\'s map, project_to_road=False); ends when it is on '
                'one again; offroad_m is the distance beyond the nearest driving lane\'s edge',
    'wrong_way': 'the ego\'s centre is on a driving lane (outside junctions) whose direction differs from the ego\'s '
                 'heading by more than WRONG_WAY_DEG',
    'outside_route_lanes': 'the ego\'s centre is on a driving lane (outside junctions) that is none of the route\'s '
                           'lanes (road id, lane id) and not wrong way, e.g. a lane change',
    'route_deviation': '|lateral offset from the route| exceeds route_deviation_m',
    'agent_blocked': 'the speed stayed below BLOCKED_SPEED_MPS for blocked_s seconds',
}
DURATION_KINDS = ('off_road', 'wrong_way', 'outside_route_lanes')
COLLISION_KINDS = ('collision_vehicle', 'collision_pedestrian', 'collision_static')
WRONG_WAY_DEG = 135.0
BLOCKED_SPEED_MPS = 0.1
COLLISION_REARM_S = 1.0
DEFAULT_THRESHOLDS = {'offroad_terminate_m': 1.0, 'route_deviation_m': 8.0, 'blocked_s': 20.0, 'route_end_m': 2.0}
DEFAULT_TERMINATE_ON = ('collision', 'off_road', 'route_deviation', 'agent_blocked')
FAILURE_KINDS = ('collision_vehicle', 'collision_pedestrian', 'collision_static', 'red_light', 'stop_sign', 'off_road',
                 'route_deviation', 'agent_blocked', 'wrong_way')


def wrap_rad(a):
    return (np.asarray(a, float) + math.pi) % (2.0 * math.pi) - math.pi


def wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


# ============================================================================================ the route
class Route:
    """A lane-centre polyline (module doc): pts (n, 2) in CARLA's world, widths (n,) metres; lane_keys: optional
    {(road id, lane id)} of the lanes it runs on (for outside_route_lanes)."""

    def __init__(self, pts, widths=None, lane_keys=None):
        P = np.asarray(pts, float).reshape(-1, 2)
        W = np.full(len(P), 3.5) if widths is None else np.asarray(widths, float).reshape(-1)
        keep = [0]
        for k in range(1, len(P)):                   # drop repeated points (zero-length segments)
            if np.hypot(*(P[k] - P[keep[-1]])) > 1e-6:
                keep.append(k)
        P, W = P[keep], W[keep]
        if len(P) < 2:
            raise ValueError('a route needs at least two distinct points')
        self.pts, self.w = P, W
        seg = np.diff(P, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.dir = seg / self.seg_len[:, None]
        self.s = np.r_[0.0, np.cumsum(self.seg_len)]
        self.length = float(self.s[-1])
        self.heading = np.degrees(np.unwrap(np.arctan2(self.dir[:, 1], self.dir[:, 0])))    # per segment, unwrapped
        self.lane_keys = set(map(tuple, lane_keys)) if lane_keys else set()

    def _seg(self, s):
        return int(np.clip(np.searchsorted(self.s, s, side='right') - 1, 0, len(self.seg_len) - 1))

    def at(self, s):
        """(x, y, heading_deg) at arc length s (clipped to the route)."""
        s = float(np.clip(s, 0.0, self.length))
        k = self._seg(s)
        p = self.pts[k] + self.dir[k] * (s - self.s[k])
        return float(p[0]), float(p[1]), float(wrap_deg(self.heading[k]))

    def heading_at(self, s) -> float:
        return float(wrap_deg(self.heading[self._seg(float(np.clip(s, 0.0, self.length)))]))

    def width_at(self, s) -> float:
        s = float(np.clip(s, 0.0, self.length))
        return float(self.w[int(np.clip(np.searchsorted(self.s, s), 0, len(self.w) - 1))])

    def curvature(self, s, half=CURV_HALF_M) -> float:
        """Δheading / Δs over [s − half, s + half] (clipped to the route; 0 when that span is shorter than 1 m)."""
        a, b = max(float(s) - half, 0.0), min(float(s) + half, self.length)
        if b - a < 1.0:
            return 0.0
        dh = float(self.heading[self._seg(b)] - self.heading[self._seg(a)])     # unwrapped: no ±360 jumps
        return math.radians(dh) / (b - a)

    def project(self, x, y, hint=None, window=40.0):
        """(s, lateral, heading_deg): the nearest point on the polyline's segments (within ±window m of hint)."""
        if hint is None:
            lo, hi = 0, len(self.seg_len)
        else:
            lo = max(int(np.searchsorted(self.s, float(hint) - window)) - 1, 0)
            hi = min(int(np.searchsorted(self.s, float(hint) + window)) + 1, len(self.seg_len))
            if hi <= lo:
                lo, hi = 0, len(self.seg_len)
        A, D, L = self.pts[lo:hi], self.dir[lo:hi], self.seg_len[lo:hi]
        q = np.array([x, y], float)
        t = np.clip(np.sum((q - A) * D, axis=1), 0.0, L)
        Q = A + D * t[:, None]
        d = np.hypot(*(q - Q).T)
        k = int(np.argmin(d))
        dk = D[k]
        lateral = float((q - Q[k]) @ np.array([-dk[1], dk[0]]))
        return float(self.s[lo + k] + t[k]), lateral, float(wrap_deg(self.heading[lo + k]))

    def truncated(self, length):
        """The first `length` m of the route."""
        if length >= self.length:
            return self
        k = int(np.searchsorted(self.s, length))
        x, y, _ = self.at(length)
        pts = np.r_[self.pts[:k], [[x, y]]]
        return Route(pts, np.r_[self.w[:k], [self.w[min(k, len(self.w) - 1)]]], self.lane_keys)

    def doc(self, every_m=5.0) -> dict:
        """A light description for logs: length and points every every_m metres."""
        ss = np.arange(0.0, self.length + 1e-9, every_m)
        return {'length_m': round(self.length, 3),
                'points': [[round(v, 3) for v in self.at(s)[:2]] for s in ss]}


def route_from_lanes(lanes: dict, path_xy, path_yaw_deg, step_m=1.0, max_yaw_diff=45.0) -> Route:
    """The lane-centre route of a recorded drive (behaviour cloning): the ego's driven path sampled every step_m
    metres, each sample snapped to the nearest lanes.json centre point whose lane direction is within max_yaw_diff of
    the ego's heading there. A lane change becomes a sideways step of the route (a limit, stated in the module doc)."""
    cols = lanes['columns']
    ix = {k: cols.index(k) for k in ('x', 'y', 'yaw', 'width', 'road', 'lane')}
    L = np.asarray([[r[ix['x']], r[ix['y']], r[ix['yaw']], r[ix['width']], r[ix['road']], r[ix['lane']]]
                    for r in lanes['points']], float)
    if not len(L):
        raise ValueError('lanes.json has no lane points')
    P = np.asarray(path_xy, float).reshape(-1, 2)
    Y = np.asarray(path_yaw_deg, float).reshape(-1)
    seg = np.hypot(*np.diff(P, axis=0).T) if len(P) > 1 else np.zeros(0)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < step_m:
        raise ValueError('the drive moved less than one route step')
    ss = np.arange(0.0, s[-1] + 1e-9, step_m)
    xs, ys = np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])
    yaws = np.degrees(np.unwrap(np.radians(Y)))
    ya = np.interp(ss, s, yaws)
    pts, ws, keys = [], [], set()
    lo, hi = L[:, :2].min(0) - 50.0, L[:, :2].max(0) + 50.0
    for x, y, yaw in zip(xs, ys, ya):
        if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
            continue
        near = np.flatnonzero((np.abs(L[:, 0] - x) < 15.0) & (np.abs(L[:, 1] - y) < 15.0))
        if not len(near):
            continue
        ok = near[np.abs(wrap_deg(L[near, 2] - yaw)) < max_yaw_diff]
        if not len(ok):
            continue
        k = ok[int(np.argmin(np.hypot(L[ok, 0] - x, L[ok, 1] - y)))]
        pts.append(L[k, :2])
        ws.append(L[k, 3])
        keys.add((int(L[k, 4]), int(L[k, 5])))
    if len(pts) < 2:
        raise ValueError('the drive\'s path matches no lanes.json lane')
    # lane points are 2 m apart: consecutive samples often snap to the same point; keep the order, drop repeats
    return Route(pts, ws, keys)


# ============================================================================================ the observation
class Featuriser:
    """Scene state → the observation vector (module doc). Stateful within an episode (the previous speed and
    acceleration, the projection hint): reset() at the start of each.

    ego: {'x', 'y', 'yaw' (deg), 'speed_mps', 'length', 'width'}; actors: actors.jsonl-style rows {'id', 'x', 'y',
    'yaw', 'vx', 'vy', 'extent': [ex, ey, ez], 'category' (nuScenes, core.groundtruth.category_of)}; lights: None
    (unknown) or [(stop line arc length on the route, state 'Red' | 'Yellow' | 'Green' | 'Off' | 'Unknown'), …]."""

    def __init__(self, route: Route, target_speed_mps: float, dt: float):
        self.route = route
        self.target = float(target_speed_mps)
        self.dt = float(dt)
        self.reset()

    def reset(self):
        self.prev_speed = None
        self.prev_accel = 0.0
        self.hint = None

    def __call__(self, ego: dict, actors=(), lights=None, prev_action=(0.0, 0.0), ego_id=None):
        """(vector (N_FEATURES,) float32, facts dict — the quantities behind it, for rewards and logs)."""
        R = self.route
        v = float(ego['speed_mps'])
        accel = 0.0 if self.prev_speed is None else (v - self.prev_speed) / self.dt
        jerk = 0.0 if self.prev_speed is None else (accel - self.prev_accel) / self.dt
        self.prev_speed, self.prev_accel = v, accel
        s, lat, hr = R.project(ego['x'], ego['y'], self.hint)
        self.hint = s
        herr = float(wrap_rad(math.radians(float(ego['yaw']) - hr)))
        width = R.width_at(s)
        half_len, half_w = float(ego.get('length', 4.5)) / 2.0, float(ego.get('width', 2.0)) / 2.0
        v_along = v * math.cos(herr)
        lead, vru = self._actors(s, v_along, half_len, half_w, actors, ego_id)
        light = self._light(s, half_len, lights)
        f = np.zeros(N_FEATURES, np.float64)
        f[FI['speed_mps']] = v
        f[FI['target_speed_mps']] = self.target
        f[FI['accel_mps2']] = accel
        f[FI['lateral_offset_m']] = lat
        f[FI['heading_error_rad']] = herr
        f[FI['lane_width_m']] = width
        for d, name in zip(CURV_AT_M, ('curvature_10m', 'curvature_25m', 'curvature_50m')):
            f[FI[name]] = R.curvature(s + d)
        f[FI['lead_gap_m']], f[FI['lead_ttc_s']] = LEAD_RANGE_M, TTC_CAP_S
        if lead is not None:
            f[FI['lead_present']] = 1.0
            f[FI['lead_gap_m']] = lead['gap_m']
            f[FI['lead_closing_mps']] = lead['closing_mps']
            f[FI['lead_ttc_s']] = lead['ttc_s']
        f[FI['vru_ahead_m']], f[FI['vru_ttc_s']] = VRU_RANGE_M, TTC_CAP_S
        if vru is not None:
            f[FI['vru_present']] = 1.0
            f[FI['vru_ahead_m']] = vru['ahead_m']
            f[FI['vru_lateral_m']] = vru['lateral_m']
            f[FI['vru_lat_speed_mps']] = vru['lat_speed_mps']
            f[FI['vru_ttc_s']] = vru['ttc_s']
        f[FI['tl_distance_m']] = TL_RANGE_M
        if lights is not None:
            f[FI['tl_known']] = 1.0
            if light is not None:
                st = light['state']
                f[FI['tl_red']] = float(st == 'Red')
                f[FI['tl_yellow']] = float(st == 'Yellow')
                f[FI['tl_green']] = float(st == 'Green')
                f[FI['tl_distance_m']] = light['distance_m']
        f[FI['route_progress']] = min(max(s / max(R.length, 1e-9), 0.0), 1.0)
        f[FI['route_remaining_m']] = min(max(R.length - s, 0.0), ROUTE_REMAINING_CAP_M)
        f[FI['prev_steer']], f[FI['prev_accel']] = float(prev_action[0]), float(prev_action[1])
        facts = {'s': s, 'lateral': lat, 'heading_err': herr, 'lane_width': width, 'speed': v, 'accel': accel,
                 'jerk': jerk, 'v_along': v_along, 'lead': lead, 'vru': vru, 'light': light,
                 'remaining': R.length - s, 'route_length': R.length}
        return f.astype(np.float32), facts

    def _actors(self, s, v_along, half_len, half_w, actors, ego_id):
        R = self.route
        lead = vru = None
        ex, ey, _ = R.at(s)
        for a in actors or ():
            if (ego_id is not None and a.get('id') == ego_id) or (ego_id is None and a.get('role') == 'hero'):
                continue
            cat = str(a.get('category') or '')
            is_vru = cat.startswith('human.') or cat == 'vehicle.bicycle'
            if not is_vru and not cat.startswith('vehicle.'):
                continue
            ax, ay = float(a['x']), float(a['y'])
            if math.hypot(ax - ex, ay - ey) > LEAD_RANGE_M + 30.0:
                continue
            sa, la, ha = R.project(ax, ay, s, window=LEAD_RANGE_M + 10.0)
            if sa < s - half_len - 10.0 or sa > s + LEAD_RANGE_M + 10.0:
                continue
            rel = math.radians(float(a.get('yaw', 0.0)) - ha)
            ext = a.get('extent') or (2.25, 0.95, 0.75)
            along = abs(ext[0] * math.cos(rel)) + abs(ext[1] * math.sin(rel))
            across = abs(ext[0] * math.sin(rel)) + abs(ext[1] * math.cos(rel))
            hr = math.radians(ha)
            vx, vy = float(a.get('vx', 0.0)), float(a.get('vy', 0.0))
            a_along = vx * math.cos(hr) + vy * math.sin(hr)
            a_lat = -vx * math.sin(hr) + vy * math.cos(hr)
            if not is_vru:
                if sa <= s or abs(la) - across >= half_w + LEAD_MARGIN_M:
                    continue
                gap = max(sa - s - half_len - along, 0.0)
                if gap >= LEAD_RANGE_M or (lead is not None and gap >= lead['gap_m']):
                    continue
                closing = v_along - a_along
                ttc = 0.0 if gap <= 0.0 else (min(gap / closing, TTC_CAP_S) if closing > 0.1 else TTC_CAP_S)
                lead = {'id': a.get('id'), 'gap_m': gap, 'closing_mps': closing, 'ttc_s': ttc}
            else:
                if sa < s - half_len:
                    continue
                ahead = max(sa - s - half_len - along, 0.0)
                if ahead >= VRU_RANGE_M or (vru is not None and ahead >= vru['ahead_m']):
                    continue
                corridor = half_w + VRU_MARGIN_M + across
                t_reach = min(ahead / max(v_along, 1.0), VRU_HORIZON_S)
                pred = la + a_lat * t_reach
                if not (abs(la) < corridor or abs(pred) < corridor or (la * pred) < 0.0):
                    continue
                ttc = min(ahead / v_along, TTC_CAP_S) if v_along > 0.1 else TTC_CAP_S
                vru = {'id': a.get('id'), 'ahead_m': ahead, 'lateral_m': la, 'lat_speed_mps': a_lat, 'ttc_s': ttc}
        return lead, vru

    def _light(self, s, half_len, lights):
        if not lights:
            return None
        best = None
        for stop_s, state in lights:
            d = float(stop_s) - s - half_len
            if d < -1.0 or d > TL_RANGE_M:
                continue
            if best is None or d < best['distance_m']:
                best = {'distance_m': max(d, 0.0), 'state': str(state), 'stop_s': float(stop_s)}
        return best


# ============================================================================================ BEV
def bev_bits(ego_pose: dict, actors, spec, lane_map=None, ego_id=None) -> np.ndarray:
    """The BEV raster's bit planes as one (H, W) uint8 (bit k = worldmodel.BEV_CHANNELS[k]): core.worldmodel's own
    bev_raster, so recorded-drive BEV maps and the environment's agree."""
    from . import worldmodel as WM
    return WM.bev_raster(ego_pose, actors, spec, lane_map, ego_id)['bits']


def unpack_bev(bits) -> np.ndarray:
    """(…, H, W) packed bits → (…, 8, H, W) uint8 0/1, channel k = bit k."""
    b = np.asarray(bits, np.uint8)
    return ((b[..., None, :, :] >> np.arange(8, dtype=np.uint8)[:, None, None]) & 1).astype(np.uint8)


def pack_bev(planes) -> np.ndarray:
    """(…, 8, H, W) 0/1 → (…, H, W) uint8 packed bits: the inverse of unpack_bev."""
    p = np.asarray(planes, np.uint8)
    return (p * (1 << np.arange(8, dtype=np.uint16))[:, None, None]).sum(axis=-3).astype(np.uint8)


BEV_DEFAULT = {'res': 0.5, 'forward': 32.0, 'back': 8.0, 'left': 16.0, 'right': 16.0}


def bev_spec(d: dict | None = None):
    from .worldmodel import BevSpec
    d = dict(BEV_DEFAULT, **(d or {}))
    return BevSpec(d['res'], d['forward'], d['back'], d['left'], d['right'])


# ============================================================================================ actions
def to_control(action, kind: str = 'continuous'):
    """(control {throttle, steer, brake, hand_brake, reverse}, applied [steer, accel]) of an action (module doc).
    Raises ValueError on a malformed action."""
    if kind == 'discrete':
        try:
            k = int(np.asarray(action).reshape(-1)[0]) if np.ndim(action) else int(action)
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f'a discrete action is an index, not {action!r}') from e
        if not 0 <= k < len(DISCRETE_ACTIONS):
            raise ValueError(f'discrete action {k} is not in 0…{len(DISCRETE_ACTIONS) - 1}')
        steer, accel = DISCRETE_ACTIONS[k]
    else:
        a = np.asarray(action, float).reshape(-1)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError(f'a continuous action is two finite numbers [steer, accel], not {action!r}')
        steer, accel = float(np.clip(a[0], -1.0, 1.0)), float(np.clip(a[1], -1.0, 1.0))
    return ({'throttle': max(accel, 0.0), 'steer': steer, 'brake': max(-accel, 0.0), 'hand_brake': False,
             'reverse': False}, [steer, accel])


def control_to_action(c: dict):
    """[steer, accel] of a recorded control {throttle, steer, brake}: accel = throttle − brake (the inverse of
    to_control where only one pedal is pressed; both pressed map to their difference)."""
    return [float(c.get('steer', 0.0)), float(c.get('throttle', 0.0)) - float(c.get('brake', 0.0))]


# ============================================================================================ rewards
def reward_terms(q: dict, events, weights: dict | None = None, dt: float = 0.05) -> dict:
    """The reward's terms for one step (module doc); their sum is the reward. q: ds, speed, target, lateral,
    lane_width, heading_err, jerk, steer_rate, off_road, wrong_way, outside_lanes; events: the step's new infraction
    records (dicts with 'kind')."""
    w = dict(DEFAULT_REWARD, **(weights or {}))
    tgt = max(float(q.get('target', 0.0)), 1.0)
    half = max(float(q.get('lane_width', 3.5)) / 2.0, 0.5)
    kinds = [e['kind'] for e in events or ()]
    t = {
        'progress': w['progress'] * float(q.get('ds', 0.0)),
        'speed': -w['speed'] * abs(float(q.get('speed', 0.0)) - float(q.get('target', 0.0))) / tgt * dt *
        (OVERSPEED_FACTOR if float(q.get('speed', 0.0)) > float(q.get('target', 0.0)) else 1.0),
        'lane_offset': -w['lane_offset'] * min((float(q.get('lateral', 0.0)) / half) ** 2, 4.0) * dt,
        'heading': -w['heading'] * min(abs(float(q.get('heading_err', 0.0))), math.pi / 2.0) * dt,
        'jerk': -w['jerk'] * min(abs(float(q.get('jerk', 0.0))) / JERK_REF, 10.0) * dt,
        'steer_rate': -w['steer_rate'] * min(abs(float(q.get('steer_rate', 0.0))) / STEER_RATE_REF, 10.0) * dt,
        'collision': -(w['collision'] * sum(k in ('collision_vehicle', 'collision_static') for k in kinds)
                       + w['collision_pedestrian'] * kinds.count('collision_pedestrian')),
        'off_road': -(w['off_road'] * dt if q.get('off_road') else 0.0) - w['off_road_onset'] * kinds.count('off_road'),
        'wrong_way': -w['wrong_way'] * dt if q.get('wrong_way') else 0.0,
        'outside_lanes': -w['outside_lanes'] * dt if q.get('outside_lanes') and not q.get('wrong_way') else 0.0,
        'red_light': -w['red_light'] * kinds.count('red_light'),
        'stop_sign': -w['stop_sign'] * kinds.count('stop_sign'),
        'route_deviation': -w['route_deviation'] * kinds.count('route_deviation'),
        'agent_blocked': -w['agent_blocked'] * kinds.count('agent_blocked'),
    }
    return {k: float(t[k]) for k in REWARD_TERMS}


# ============================================================================================ geometry for infractions
def crossed_line(p0, p1, centre, yaw_deg, half_len) -> bool:
    """Whether the segment p0→p1 crosses the line through centre perpendicular to heading yaw_deg, within half_len of
    centre, going in the heading's direction (from behind the line to in front of or on it)."""
    c = np.asarray(centre, float)
    r = math.radians(float(yaw_deg))
    fwd = np.array([math.cos(r), math.sin(r)])
    side = np.array([-math.sin(r), math.cos(r)])
    a0 = float((np.asarray(p0, float) - c) @ fwd)
    a1 = float((np.asarray(p1, float) - c) @ fwd)
    if not (a0 < 0.0 <= a1):
        return False
    f = -a0 / (a1 - a0)
    p = np.asarray(p0, float) + (np.asarray(p1, float) - np.asarray(p0, float)) * f
    return abs(float((p - c) @ side)) <= float(half_len)


def in_box(px, py, cx, cy, yaw_deg, ex, ey) -> bool:
    """Whether (px, py) lies in the oriented rectangle centred at (cx, cy), heading yaw_deg, half extents ex, ey."""
    r = math.radians(float(yaw_deg))
    dx, dy = float(px) - float(cx), float(py) - float(cy)
    lon = dx * math.cos(r) + dy * math.sin(r)
    lat = -dx * math.sin(r) + dy * math.cos(r)
    return abs(lon) <= float(ex) and abs(lat) <= float(ey)


# ============================================================================================ infractions
class InfractionMonitor:
    """The episode's infractions (INFRACTIONS) and its termination, from per-step facts (pure Python, no CARLA).

    update(step, sim_t, ego, facts) per step after the tick: ego = {'x', 'y', 'z', 'yaw', 'speed_mps'}; facts:
        'contacts': [(other id, 'vehicle' | 'pedestrian' | 'static', source 'box' | 'sensor')] touching this step,
        'red_light': [traffic light id crossed on red this step], 'stop_inside': {stop id: True} volumes the centre
        is in, 'off_road' bool, 'offroad_m', 'wrong_way' bool, 'outside_lanes' bool, 'lateral', 'speed', 's',
        'route_length', 'travel_m' (distance driven this step).
    Returns (new event records, termination reason or None). A record: {'kind', 'step', 'sim_t', 'actors': [ids],
    'ego': pose, 'detail': {…}}; the condition kinds (DURATION_KINDS) gain 'end_step', 'end_sim_t' and 'distance_m'
    when the condition ends (or at finish()). Termination reasons: 'collision', 'off_road', 'route_deviation',
    'agent_blocked', 'red_light', 'stop_sign', 'wrong_way' (when in terminate_on) and 'route_complete'."""

    def __init__(self, dt: float, thresholds: dict | None = None, terminate_on=DEFAULT_TERMINATE_ON):
        self.dt = float(dt)
        self.th = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
        self.terminate_on = set(terminate_on)
        self.events: list = []
        self.open: dict = {}           # duration kind -> its open record
        self.contact_last: dict = {}   # other id -> sim_t last in contact
        self.stops: dict = {}          # stop id -> min speed inside
        self.slow_since = None
        self.deviated = False

    @staticmethod
    def _rec(kind, step, sim_t, ego, actors=(), **detail):
        return {'kind': kind, 'step': int(step), 'sim_t': round(float(sim_t), 4), 'actors': [a for a in actors],
                'ego': {k: round(float(ego[k]), 4) for k in ('x', 'y', 'z', 'yaw', 'speed_mps') if k in ego},
                'detail': detail}

    def update(self, step, sim_t, ego, facts):
        new, reason = [], None
        # collisions: once per actor until out of contact for COLLISION_REARM_S
        for oid, kind, source in facts.get('contacts') or ():
            last = self.contact_last.get(oid)
            if last is None or sim_t - last > COLLISION_REARM_S + 1e-9:
                k = {'vehicle': 'collision_vehicle', 'pedestrian': 'collision_pedestrian'}.get(kind,
                                                                                              'collision_static')
                new.append(self._rec(k, step, sim_t, ego, [oid], source=source))
                if 'collision' in self.terminate_on:
                    reason = reason or 'collision'
            self.contact_last[oid] = sim_t
        for tl in facts.get('red_light') or ():
            new.append(self._rec('red_light', step, sim_t, ego, [tl]))
            if 'red_light' in self.terminate_on:
                reason = reason or 'red_light'
        inside = facts.get('stop_inside') or {}
        for sid in list(self.stops):
            if sid not in inside:
                vmin = self.stops.pop(sid)
                if vmin > STOP_SPEED_MPS:
                    new.append(self._rec('stop_sign', step, sim_t, ego, [sid], min_speed_mps=round(vmin, 3)))
                    if 'stop_sign' in self.terminate_on:
                        reason = reason or 'stop_sign'
        for sid in inside:
            self.stops[sid] = min(self.stops.get(sid, math.inf), float(facts.get('speed', 0.0)))
        # conditions with a duration
        for kind, on in (('off_road', facts.get('off_road')), ('wrong_way', facts.get('wrong_way')),
                         ('outside_route_lanes', facts.get('outside_lanes') and not facts.get('wrong_way'))):
            rec = self.open.get(kind)
            if on and rec is None:
                rec = self._rec(kind, step, sim_t, ego)
                rec['distance_m'] = 0.0
                self.open[kind] = rec
                new.append(rec)
                if kind == 'wrong_way' and 'wrong_way' in self.terminate_on:
                    reason = reason or 'wrong_way'
            elif on:
                rec['distance_m'] = round(rec['distance_m'] + float(facts.get('travel_m', 0.0)), 4)
            elif rec is not None:
                self._close(kind, step, sim_t)
        if facts.get('off_road'):
            off = self.open.get('off_road')
            off['detail']['max_offroad_m'] = round(max(off['detail'].get('max_offroad_m', 0.0),
                                                       float(facts.get('offroad_m', 0.0))), 4)
            if 'off_road' in self.terminate_on and float(facts.get('offroad_m', 0.0)) > self.th['offroad_terminate_m']:
                reason = reason or 'off_road'
        if not self.deviated and abs(float(facts.get('lateral', 0.0))) > self.th['route_deviation_m']:
            self.deviated = True
            new.append(self._rec('route_deviation', step, sim_t, ego, lateral_m=round(float(facts['lateral']), 3)))
            if 'route_deviation' in self.terminate_on:
                reason = reason or 'route_deviation'
        if float(facts.get('speed', 0.0)) < BLOCKED_SPEED_MPS:
            self.slow_since = sim_t if self.slow_since is None else self.slow_since
            if sim_t - self.slow_since >= self.th['blocked_s'] - 1e-9:
                new.append(self._rec('agent_blocked', step, sim_t, ego, blocked_s=round(sim_t - self.slow_since, 3)))
                self.slow_since = None
                if 'agent_blocked' in self.terminate_on:
                    reason = reason or 'agent_blocked'
        else:
            self.slow_since = None
        if reason is None and float(facts.get('s', 0.0)) >= float(facts.get('route_length', math.inf)) - \
                self.th['route_end_m']:
            reason = 'route_complete'
        self.events.extend(new)
        return new, reason

    def _close(self, kind, step, sim_t):
        rec = self.open.pop(kind)
        rec['end_step'], rec['end_sim_t'] = int(step), round(float(sim_t), 4)

    def finish(self, step, sim_t):
        """Close the open conditions at the episode's end (their end is the last step)."""
        for kind in list(self.open):
            self._close(kind, step, sim_t)

    def counts(self) -> dict:
        out = {k: 0 for k in INFRACTIONS}
        for e in self.events:
            out[e['kind']] += 1
        return out


def episode_success(summary: dict, fail_on=FAILURE_KINDS, min_progress_m: float = 0.0) -> bool:
    """An episode succeeded when none of fail_on happened, it was not ended by a failure, and it progressed at least
    min_progress_m along its route (the curriculum's default criterion; core.rl)."""
    counts = summary.get('infraction_counts') or {}
    if any(counts.get(k, 0) for k in fail_on):
        return False
    if summary.get('termination') not in (None, 'route_complete', 'time_limit'):
        return False
    return float(summary.get('progress_m', 0.0)) >= float(min_progress_m)
