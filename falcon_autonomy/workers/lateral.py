"""Lateral filtering: background two-wheelers that use the road's width, as Indian traffic does.

CARLA's traffic manager (TM) keeps every vehicle on its lane's centre line. In Bengaluru a two-wheeler rides the lane
line, squeezes between a bus and an autorickshaw, keeps to the kerb side and passes a queue. A traffic profile's
'lateral_filtering' ({'enabled': true, 'vehicles': ['motorcycle', 'bicycle'], 'controller': 'tm'}) gives the
background vehicles of those kinds (CARLA base type motorcycle / bicycle; 'autorickshaw' the stand-in class of
core.assets) a lateral position of their own, chosen every UPDATE_S of simulated time by a gap-seeking chooser:

    the same-direction drivable width around the vehicle — its lane and, beside it, one same-direction driving lane
    (or paved shoulder) each side; never an oncoming lane — is cut into STRIP_M strips (candidate offsets from the
    centre of the vehicle's reference lane, + towards the driver's right, CARLA's right vector (−sin yaw, cos yaw));
    every neighbour within NEAR_M is put in the lane's curvilinear frame (along s, across d: a polyline of the lane
    ahead, so curves do not bend the strips); each candidate is scored by
      * clearance: the speed the strip allows against the speed the vehicle wants (its road's limit less its own
        difference) — the first object whose across-span (widened by the lateral clearance its class is given,
        CLEAR_M) overlaps the vehicle's path to the strip and along it allows its own speed plus the gap (less
        STANDSTILL_M) covered in T_LOOK_S; the path sideways counts while the vehicle gets across, at LAT_RATE_MPS
        against its closing speed;
      * an anisotropic repulsion field from neighbours: exp(−(Δs/σs)²)·exp(−(edge gap/σd)²) with σd by the
        neighbour's class (SIGMA_D_M: a bus or truck repels from further across than a motorcycle) and σs longer
        ahead than behind;
      * the route: distance from a home offset a little to the near (kerb) side — left in left-hand traffic, right
        in right-hand (the Indian rule of the road: keep to the left; core.traffic handedness) — and a penalty on
        changing the choice (hysteresis);
      * moving sideways into a vehicle alongside is not allowed;
    the lowest score wins and the vehicle's commanded offset moves towards it at LAT_RATE_MPS. Approaching and inside
    a junction (JUNCTION_M) the target is the lane centre: the TM negotiates junctions from its lane.

Actuation ('controller'):

* 'tm' (the default): the vehicle stays on the TM's autopilot — routing, traffic lights, signs, walkers, speed — and
  the offset goes to TrafficManager.vehicle_lane_offset. Measured on CARLA 0.9.16 (Town10HD, 2026-10-30, probe in
  docs/indian-assets.md): the TM tracks an offset to ±2.4 m within about 2 s, but its collision detection ignores the
  offset both ways — an offset motorcycle still stops behind a car on its lane's centre it would clear by 0.3 m, and
  drives into a car on the neighbouring lane it overlaps by 0.4 m (10 collision events). So while the vehicle is off
  its lane centre (IGNORE_FROM_M) the chooser's own clearance along its path caps its desired speed — the speed from
  which DECEL_MPS2 stops STANDSTILL_M behind the object (or matches its speed) — and when the TM would stop it for
  an object on the lane's centre that its own path clears (filtering past it), the TM's vehicle detection is
  switched off for it (ignore_vehicles_percentage 100) until that is no longer so. In junctions the TM's own
  detection and speed take over again. The TM's own lane changes are off for these vehicles (their lateral moves are
  the chooser's; a TM lane change under an offset shifts the reference lane by a lane width).
* 'own': the vehicle comes off the autopilot and is driven here: its route along the lane graph (a seeded choice at
  each branch, avoiding dead ends of an imported world), pure-pursuit steering to the route shifted by the offset,
  a PI speed controller to min(the road's limit — workers.speedcontrol, with breakers —, the clearance cap, a red or
  yellow light it is at, a yield to vehicles crossing its path in a junction). Kept for comparison: docs/indian-
  assets.md has the measurements of both.

Every constant below marked "heuristic" was chosen, not measured on Indian roads, and checked only in so far as the
measurements in docs/indian-assets.md say. Deterministic with the run's seed: vehicles are taken in the order they
were added, the 'own' controller's branch choices come from a generator seeded with (seed, the vehicle's index).

Made by scene.spawn_traffic when the profile enables it (w.lateral); stepped by scene.speeds_step after each tick;
TrafficKeeper hands it the vehicles it replaces and moves; run.json traffic.lateral_filtering, actors.jsonl
'lateral' per held vehicle and frames.jsonl 'lateral' say what it did (docs/dataset-format.md)."""
from __future__ import annotations

import math
import time

from ..core.traffic import LATERAL_DEFAULTS as DEFAULTS  # noqa: F401  (the profile's setting)
from ..core.traffic import lateral_settings as settings_of  # noqa: F401

UPDATE_S = 0.1                 # the chooser's period (simulated time)
STRIP_M = 0.4                  # heuristic: candidate offsets this far apart
EDGE_M = 0.25                  # heuristic: kept between the vehicle's side and a kerb or the oncoming lanes' line
NEAR_M = 40.0                  # neighbours considered
LOOK_M = 30.0                  # clearance looked for ahead (also the polyline's reach)
BEHIND_M = 8.0                 # the polyline starts this far behind
PATH_STEP_M = 2.5
T_LOOK_S = 4.0                 # heuristic: the strip's speed: the leader's plus the gap closed in this time
JUNCTION_M = 5.0               # heuristic: back to the lane centre this far before a junction
LAT_RATE_MPS = 1.0             # heuristic: sideways speed of the commanded offset (3.4 m in 3.4 s)
IGNORE_FROM_M = 0.3            # off its lane centre by more than this (or aiming so): mode 'filter', speed capped
TM_MAX_OFFSET_M = 2.4          # 'tm': the largest offset sent (measured: the TM tracks ±2.4 m, module doc)
REBUILD_M = 1.5                # 'tm': this far from where the TM would put it on its path: the path is rebuilt
DECEL_MPS2 = 3.5               # heuristic: the clearance cap's braking
STANDSTILL_M = 1.5             # heuristic: gap kept to a stopped object
HOME_SHARE = 0.2               # heuristic: the home offset, a share of the lane width to the near (kerb) side
# heuristic, by neighbour class: the lateral clearance (m) a rider keeps when passing (a hard limit of the clearance
# test) and the across-σ (m) of the repulsion field (a soft preference) — both larger for heavy vehicles
CLEAR_M = {'bus': 0.5, 'truck': 0.5, 'van': 0.35, 'car': 0.3, 'autorickshaw': 0.25, 'motorcycle': 0.2,
           'bicycle': 0.2, 'walker': 0.6, 'other': 0.3}
SIGMA_D_M = {'bus': 1.0, 'truck': 1.0, 'van': 0.7, 'car': 0.6, 'autorickshaw': 0.5, 'motorcycle': 0.35,
             'bicycle': 0.35, 'walker': 0.8, 'other': 0.6}
SIGMA_S_AHEAD_M, SIGMA_S_BEHIND_M = 8.0, 3.0      # heuristic: along-σ of the field
W_CLEAR, W_FIELD, W_ROUTE, W_SWITCH = 4.0, 1.5, 0.6, 0.4    # heuristic weights of the score
BLOCKED = 100.0                # the score of a candidate reached only through a vehicle alongside
RESEND_KMH = 0.5
CLOSING_MIN_MPS = 0.5          # the slowest closing speed the sweep assumes
# 'own' controller (heuristic gains)
LD_MIN_M, LD_MAX_M, LD_GAIN_S = 4.0, 14.0, 0.7    # pure-pursuit look-ahead: clip(LD_GAIN_S·v, LD_MIN_M, LD_MAX_M)
KP, KI, KB = 0.35, 0.08, 0.5                      # PI throttle, proportional brake (per m/s of error)
YIELD_S = (0.5, 1.0, 1.5, 2.0, 2.5)               # junction yield: predicted times checked


def parameters() -> dict:
    """The chooser's constants as run.json records them (all heuristics but the measured TM facts)."""
    return {'update_s': UPDATE_S, 'strip_m': STRIP_M, 'edge_m': EDGE_M, 'near_m': NEAR_M, 'look_m': LOOK_M,
            't_look_s': T_LOOK_S, 'junction_m': JUNCTION_M, 'lat_rate_mps': LAT_RATE_MPS,
            'ignore_from_m': IGNORE_FROM_M, 'decel_mps2': DECEL_MPS2, 'standstill_m': STANDSTILL_M,
            'home_share': HOME_SHARE, 'clear_m': dict(CLEAR_M), 'sigma_d_m': dict(SIGMA_D_M),
            'sigma_s_m': [SIGMA_S_AHEAD_M, SIGMA_S_BEHIND_M], 'tm_max_offset_m': TM_MAX_OFFSET_M,
            'weights': {'clear': W_CLEAR, 'field': W_FIELD, 'route': W_ROUTE, 'switch': W_SWITCH},
            'source': 'heuristic (chosen, not measured on Indian roads); docs/indian-assets.md has what was measured'}


def choose(d_self: float, hw: float, hl: float, v: float, cands, others: dict, home: float, prev: float,
           lane_w: float = 3.5, v_want: float | None = None) -> dict:
    """The best lateral offset for a vehicle: pure arithmetic on its lane's curvilinear frame (numpy arrays).

    d_self: the vehicle's offset now (m, + right of its reference lane's centre); hw, hl: its half width and length;
    v: its speed (m/s); cands: candidate offsets; others: arrays 's' (along, m, relative to the vehicle), 'd' (across,
    m, like d_self), 'hl', 'hw' (half extents along / across the lane), 'v' (speed along the lane, m/s), 'clear' (the
    lateral clearance kept from it, m), 'sig' (its field's across-σ, m); home: the preferred offset; prev: the
    previous choice. Returns {'target', 'gap' (m to the first object
    in the path to the target, LOOK_M when none), 'lead_v' (its speed along), 'v_strip' (the speed the strip
    allows), 'score', 'blocked'}. v_want: the speed it wants (m/s; default: its speed, at least 5 m/s)."""
    import numpy as np
    C = np.asarray(cands, float)
    s, d = np.asarray(others.get('s', []), float), np.asarray(others.get('d', []), float)
    ohl, ohw = np.asarray(others.get('hl', []), float), np.asarray(others.get('hw', []), float)
    ov, buf = np.asarray(others.get('v', []), float), np.asarray(others.get('clear', []), float)
    sig = np.asarray(others.get('sig', []), float)
    K, M = len(C), len(s)
    gap = np.full(K, LOOK_M)
    lead_v = np.full(K, np.nan)
    blocked = np.zeros(K, bool)
    field = np.zeros(K)
    if M:
        # the vehicle's across-span while it is level with each object: it moves sideways at LAT_RATE_MPS towards
        # the candidate as it closes on the object (closing speed v − v_o, at least CLOSING_MIN_MPS: a neighbour at
        # its own speed stays level with it the whole way across)
        L = (ohl + hl)[None, :]
        dist = np.abs(C - d_self)[:, None]
        sgn = np.sign(C - d_self)[:, None]
        rate = LAT_RATE_MPS / np.maximum(v - ov, CLOSING_MIN_MPS)[None, :]     # metres across per metre closed
        d0 = d_self + sgn * np.minimum(dist, rate * np.maximum(s[None, :] - L, 0.0))
        d1 = d_self + sgn * np.minimum(dist, rate * np.maximum(s[None, :] + L, 0.0))
        lo_c, hi_c = np.minimum(d0, d1) - hw, np.maximum(d0, d1) + hw
        o_lo, o_hi = (d - ohw - buf)[None, :], (d + ohw + buf)[None, :]
        over = (o_lo < hi_c) & (o_hi > lo_c)
        ahead = s[None, :] > -0.5 * hl
        g = np.where(over & ahead, np.maximum(s - ohl - hl, 0.0)[None, :], np.inf)
        k = np.argmin(g, axis=1)
        gmin = g[np.arange(K), k]
        has = np.isfinite(gmin) & (gmin < LOOK_M)
        gap = np.where(has, gmin, LOOK_M)
        lead_v = np.where(has, ov[k], np.nan)
        # blocked: moving sideways into a vehicle level with it, or into the strip of one closing from behind
        # before it is across (staying where it is never is)
        moving = (dist[:, 0] > STRIP_M / 2)[:, None]
        level = (np.abs(s) < (ohl + hl + 0.5))[None, :]
        t_across = dist / LAT_RATE_MPS + 1.0
        in_strip = (o_lo < C[:, None] + hw) & (o_hi > C[:, None] - hw)
        closing = np.maximum(ov - v, 0.0)
        rear = (s < 0)[None, :] & in_strip & ((-s - L) < closing[None, :] * t_across)
        blocked = ((over & level) | rear).any(axis=1) & moving[:, 0]
        sig_s = np.where(s > 0, SIGMA_S_AHEAD_M, SIGMA_S_BEHIND_M)
        edge = np.maximum(np.abs(C[:, None] - d[None, :]) - hw - ohw[None, :], 0.0)
        field = (np.exp(-(s / sig_s) ** 2)[None, :] * np.exp(-(edge / np.maximum(sig, 0.1)[None, :]) ** 2)).sum(1)
    vw = max(float(v_want) if v_want is not None else v, 5.0)
    v_strip = np.where(np.isnan(lead_v), vw, np.minimum(
        vw, np.maximum(np.nan_to_num(lead_v), 0.0) + np.maximum(gap - STANDSTILL_M, 0.0) / T_LOOK_S))
    score = (W_CLEAR * (vw - v_strip) / vw + W_FIELD * field + W_ROUTE * np.abs(C - home) / lane_w +
             W_SWITCH * np.abs(C - prev) / lane_w + BLOCKED * blocked)
    i = int(np.argmin(score))
    return {'target': float(C[i]), 'gap': float(gap[i]), 'lead_v': None if np.isnan(lead_v[i]) else float(lead_v[i]),
            'v_strip': float(v_strip[i]), 'score': float(score[i]), 'blocked': bool(blocked.all())}


def cap_mps(gap: float, lead_v: float | None) -> float | None:
    """The clearance cap: the speed from which DECEL_MPS2 brings the vehicle to its leader's speed STANDSTILL_M
    behind it (None: nothing in reach)."""
    if gap >= LOOK_M:
        return None
    vl = max(lead_v or 0.0, 0.0)
    return math.sqrt(vl * vl + 2.0 * DECEL_MPS2 * max(gap - STANDSTILL_M, 0.0))


def project(px, py, path) -> tuple:
    """Points (px, py arrays) in a polyline's curvilinear frame: (s along from its first point, d across, + right).
    path: arrays x, y (the polyline) and cum (arc length at each point)."""
    import numpy as np
    X, Y, cum = path['x'], path['y'], path['cum']
    ax, ay = X[:-1], Y[:-1]
    ux, uy = X[1:] - ax, Y[1:] - ay
    L2 = np.maximum(ux * ux + uy * uy, 1e-9)
    t = np.clip(((px[:, None] - ax[None, :]) * ux[None, :] + (py[:, None] - ay[None, :]) * uy[None, :]) / L2, 0, 1)
    qx, qy = ax[None, :] + t * ux[None, :], ay[None, :] + t * uy[None, :]
    dist2 = (px[:, None] - qx) ** 2 + (py[:, None] - qy) ** 2
    j = np.argmin(dist2, axis=1)
    r = np.arange(len(px))
    L = np.sqrt(L2[j])
    s = cum[j] + t[r, j] * L
    # beyond either end: along the end segment's line (t unclipped there)
    t0 = ((px - ax[0]) * ux[0] + (py - ay[0]) * uy[0]) / L2[0]
    s = np.where((j == 0) & (t0 < 0), t0 * np.sqrt(L2[0]), s)
    tn = ((px - ax[-1]) * ux[-1] + (py - ay[-1]) * uy[-1]) / L2[-1]
    s = np.where((j == len(ax) - 1) & (tn > 1), cum[-2] + tn * np.sqrt(L2[-1]), s)
    # across: + right of the segment's direction (CARLA's left-handed axes: right = (−uy, ux) / |u|)
    d = ((px - ax[j]) * -uy[j] + (py - ay[j]) * ux[j]) / L
    return s, d


class LateralFilter:
    """The vehicles of a run that filter laterally, and the chooser and actuation that move them (module doc)."""

    def __init__(self, w, world, settings: dict, seed: int = 0, left_hand: bool = False,
                 behaviour: dict | None = None):
        self.w, self.world = w, world
        self.cmap = world.get_map()
        self.settings = dict(settings)
        self.controller = settings['controller']
        self.kinds = set(settings['vehicles'])
        self.seed = int(seed)
        self.near_sign = -1.0 if left_hand else 1.0      # the near (kerb) side: the driver's left in LHT
        self.rows: dict = {}                             # actor id → row
        self.order = 0
        self.last = self.last_drive = -1e9
        self.ignore_lights_pct = float((behaviour or {}).get('ignore_lights_pct', 0.0))
        self.static: dict = {}                           # actor id → (half length, half width, clear, σd, class)
        self.ids: list = []                              # vehicles and walkers in the world (refreshed)
        self.ids_at = -1e9
        self.bounds_cache: dict = {}
        self.counts = {'vehicles': 0, 'adopted': 0, 'updates': 0, 'junction_updates': 0, 'capped_updates': 0,
                       'ignore_on': 0, 'blocked_updates': 0, 'off_road_updates': 0, 'lights_stopped': 0,
                       'yields': 0, 'dead_end_branches_avoided': 0, 'path_builds': 0}
        self.by_kind: dict = {}
        self.ms: list = []                               # wall ms of each step() that worked
        self.hist = {'abs_offset': [0] * 12, 'between_lanes': 0, 'samples': 0}
        tm = getattr(w, 'tm', None)
        self.ok = True
        if self.controller == 'tm' and (tm is None or not hasattr(tm, 'vehicle_lane_offset')):
            self.ok = False
            w.note('lateral filtering needs TrafficManager.vehicle_lane_offset (CARLA 0.9.14 and later): off')

    # ------------------------------------------------------------------ the vehicles
    def wants(self, a, kind: str = '', cls: str | None = None) -> bool:
        """Whether a traffic vehicle of this fleet kind / stand-in class filters (by its CARLA base type)."""
        if not self.ok:
            return False
        if cls:
            return cls in self.kinds
        base = str((getattr(a, 'attributes', None) or {}).get('base_type', kind)).lower()
        return base in self.kinds or kind in self.kinds

    def add(self, a, kind: str = '', cls: str | None = None, diff_pct: float | None = None):
        import numpy as np
        if not self.wants(a, kind, cls):
            return False
        n = self.order
        self.order += 1
        bb = a.bounding_box.extent
        label = cls or str((getattr(a, 'attributes', None) or {}).get('base_type', kind)).lower() or 'other'
        row = {'a': a, 'n': n, 'kind': label, 'hl': float(bb.x), 'hw': float(bb.y), 'cmd': 0.0, 'target': 0.0,
               'd': 0.0, 'mode': 'lane', 'ignoring': False, 'cap': None, 'gap': LOOK_M, 'path': None,
               'lane': None, 'diff': float(diff_pct or 0.0), 'sent_offset': None, 'sent_kmh': None,
               'rng': np.random.default_rng([self.seed, 0x1A7E, n]), 'manual': False, 'i_err': 0.0,
               'light': None, 'ignore_light': False, 'geom': None}
        self.rows[a.id] = row
        self.counts['vehicles'] += 1
        self.by_kind[label] = self.by_kind.get(label, 0) + 1
        self._no_lane_changes(row)
        return True

    def _no_lane_changes(self, row):
        """'tm': the TM's own lane changes off for a vehicle that filters — its lateral moves are the chooser's, and
        the offset is relative to the TM's lane, so a TM lane change would shift it by a lane width (measured on
        central Bengaluru, 2026-10-30: with them on, the reference lane flipped under offset vehicles)."""
        tm = self.w.tm
        if self.controller != 'tm' or tm is None:
            return
        a = row['a']
        tm.auto_lane_change(a, False)
        for f in ('random_left_lanechange_percentage', 'random_right_lanechange_percentage'):
            if hasattr(tm, f):
                getattr(tm, f)(a, 0.0)

    def holds(self, aid) -> bool:
        return aid in self.rows

    def adopt(self, old_id, a):
        """A replacement (TrafficKeeper) for a vehicle that filtered: it filters too, from its lane centre."""
        row = self.rows.pop(old_id, None)
        if row is None:
            return False
        self.counts['vehicles'] -= 1
        self.by_kind[row['kind']] -= 1
        ok = self.add(a, row['kind'] if row['kind'] in ('motorcycle', 'bicycle') else '',
                      row['kind'] if row['kind'] == 'autorickshaw' else None, row['diff'])
        if ok:
            self.counts['adopted'] += 1
        return ok

    def reset(self, a):
        """The vehicle was moved (TrafficKeeper, on the autopilot again there): its offset, path and state start
        afresh."""
        row = self.rows.get(a.id)
        if row is None:
            return
        row.update(cmd=0.0, target=0.0, d=0.0, mode='lane', ignoring=False, cap=None, path=None, lane=None,
                   sent_offset=None, sent_kmh=None, manual=False, i_err=0.0, light=None)
        if self.controller == 'tm':
            self._offset(row, 0.0, force=True)
            self._ignore(row, False, force=True)
            self._no_lane_changes(row)            # (the keeper's restart applied the profile's behaviour again)

    def drop(self, aid):
        self.rows.pop(aid, None)

    # ------------------------------------------------------------------ the world around
    def _class_of(self, a) -> str:
        tid = str(a.type_id)
        if tid.startswith('walker.'):
            return 'walker'
        sub = (getattr(self.w, 'subclass', None) or {}).get(a.id)
        if sub and 'autorickshaw' in str(sub):
            return 'autorickshaw'
        base = str((getattr(a, 'attributes', None) or {}).get('base_type', '')).lower()
        return base if base in CLEAR_M else ('car' if tid.startswith('vehicle.') else 'other')

    def _refresh_ids(self, sim_t):
        if sim_t - self.ids_at < 1.0 and self.ids:
            return
        self.ids_at = sim_t
        ids = []
        for a in self.world.get_actors():
            tid = str(a.type_id)
            if not tid.startswith(('vehicle.', 'walker.')):
                continue
            if a.id not in self.static:
                try:
                    e = a.bounding_box.extent
                except RuntimeError:
                    continue
                cls = self._class_of(a)
                self.static[a.id] = (float(e.x), float(e.y), CLEAR_M.get(cls, CLEAR_M['other']),
                                     SIGMA_D_M.get(cls, SIGMA_D_M['other']), cls)
            ids.append(a.id)
        self.ids = sorted(ids)

    def _scene(self):
        """Arrays of every vehicle and walker now (the snapshot of the frame just ticked)."""
        import numpy as np
        snap = self.world.get_snapshot()
        rows = []
        for aid in self.ids:
            sa = snap.find(aid)
            if sa is None:
                continue
            t, v = sa.get_transform(), sa.get_velocity()
            hl, hw, clear, sig, _ = self.static[aid]
            rows.append((aid, t.location.x, t.location.y, t.rotation.yaw, v.x, v.y, hl, hw, clear, sig))
        if not rows:
            return None, snap
        A = np.array(rows, float)
        return {'id': A[:, 0].astype(np.int64), 'x': A[:, 1], 'y': A[:, 2], 'yaw': np.radians(A[:, 3]), 'vx': A[:, 4],
                'vy': A[:, 5], 'hl': A[:, 6], 'hw': A[:, 7], 'clear': A[:, 8], 'sig': A[:, 9]}, snap

    def _bounds(self, wp) -> tuple:
        """(left, right) edges of the same-direction drivable width around wp's lane, as offsets from its centre
        (left negative): its lane, and one same-direction driving lane or shoulder beside it each side."""
        key = (wp.road_id, wp.section_id, wp.lane_id, int(wp.s // 10.0))
        got = self.bounds_cache.get(key)
        if got is not None:
            return got
        carla = self.w.carla
        ok_types = {carla.LaneType.Driving, getattr(carla.LaneType, 'Shoulder', carla.LaneType.Driving)}
        half = float(wp.lane_width) / 2.0
        edges = []
        for get, sign in (('get_left_lane', -1.0), ('get_right_lane', 1.0)):
            e = half
            try:
                nb = getattr(wp, get)()
            except RuntimeError:
                nb = None
            if nb is not None and nb.lane_type in ok_types and (nb.lane_id > 0) == (wp.lane_id > 0) and \
                    nb.lane_id != wp.lane_id:
                e += float(nb.lane_width)
                edge_kind = 'lane'
            else:
                edge_kind = 'kerb'
            edges.append((sign * e, edge_kind))
        got = (edges[0], edges[1])
        self.bounds_cache[key] = got
        return got

    def _next(self, row, wp):
        """The waypoint after wp along the vehicle's path: 'tm' guesses the TM's (the branch keeping the road, else
        the straightest); 'own' chooses (seeded), avoiding branches that dead-end within the keeper's look-ahead."""
        nxt = wp.next(PATH_STEP_M)
        if len(nxt) <= 1:
            return nxt[0] if nxt else None
        yaw = wp.transform.rotation.yaw
        nxt = sorted(nxt, key=lambda n: (n.road_id != wp.road_id,
                                         abs((n.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0), n.lane_id))
        if self.controller != 'own':
            return nxt[0]
        keeper = getattr(self.w, 'road_look', None) or getattr(self.w, 'traffic_keeper', None)
        if keeper is not None:
            ok = [n for n in nxt if keeper._walk(n, 0.0, keeper.SPAWN_CLEAR_M) is None]
            if ok and len(ok) < len(nxt):
                self.counts['dead_end_branches_avoided'] += 1
            nxt = ok or nxt
        return nxt[int(row['rng'].integers(len(nxt)))]

    def _arrays(self, p):
        import numpy as np
        X = np.array([q.transform.location.x for q in p['wps']])
        Y = np.array([q.transform.location.y for q in p['wps']])
        p.update(x=X, y=Y, cum=np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]))
        return p

    def _build(self, row, wp):
        """A new polyline for the vehicle from wp: BEHIND_M behind it, LOOK_M ahead (None: no lane ahead)."""
        back = wp.previous(BEHIND_M)
        pts = [back[0] if back else wp, wp] if back else [wp]
        cur = wp
        for _ in range(int(LOOK_M / PATH_STEP_M) + 1):
            cur = self._next(row, cur)
            if cur is None:
                break
            pts.append(cur)
        if len(pts) < 2:
            return None
        row['path'] = self._arrays({'wps': pts})
        return row['path']

    def _advance(self, row, p, s_self: float) -> float:
        """Drop the polyline's points more than BEHIND_M behind the vehicle, extend it to LOOK_M ahead; the vehicle's
        new arc length."""
        pts = p['wps']
        while len(pts) > 2 and p['cum'][1] < s_self - BEHIND_M:
            s_self -= p['cum'][1]
            pts.pop(0)
            self._arrays(p)
        while p['cum'][-1] - s_self < LOOK_M and len(pts) < 60:
            nxt = self._next(row, pts[-1])
            if nxt is None:
                break
            pts.append(nxt)
            self._arrays(p)
        return s_self

    # ------------------------------------------------------------------ each step
    def step(self, sim_t: float):
        """After each tick: every UPDATE_S the chooser for every vehicle held; the 'own' controller every tick."""
        if not self.rows or not self.ok:
            return
        due = sim_t - self.last >= UPDATE_S - 1e-9
        if not due and self.controller != 'own':
            return
        t0 = time.perf_counter()
        dt = min(max(sim_t - self.last_drive, 1e-3), 1.0) if self.last_drive > -1e8 else UPDATE_S
        self.last_drive = sim_t
        if due:
            self.last = sim_t
            self._refresh_ids(sim_t)
        scene, snap = self._scene()
        if scene is None:
            return
        keeper = getattr(self.w, 'traffic_keeper', None)
        index = {int(i): k for k, i in enumerate(scene['id'])}
        for aid in sorted(self.rows, key=lambda x: self.rows[x]['n']):
            row = self.rows[aid]
            k = index.get(aid)
            if k is None:
                continue                                  # gone: the keeper replaces it (adopt)
            krow = keeper.rows.get(aid) if keeper is not None else None
            if krow is not None and krow.get('held_since') is not None:
                row['mode'] = 'held'                      # stopped by the keeper before a dead end
                continue
            try:
                if due:
                    self._update(row, scene, k)
                if self.controller == 'own':
                    self._drive(row, scene, k, dt)
            except RuntimeError:
                continue                                  # destroyed meanwhile
        self.ms.append((time.perf_counter() - t0) * 1000.0)

    def _update(self, row, sc, k):
        import numpy as np
        carla = self.w.carla
        a = row['a']
        x, y, yaw = sc['x'][k], sc['y'][k], sc['yaw'][k]
        v = float(math.hypot(sc['vx'][k], sc['vy'][k]))
        if self.controller == 'own' and not row['manual']:
            a.set_autopilot(False, self.w.tm.get_port() if self.w.tm is not None else 8000)
            row['manual'] = True
            sc_ = getattr(self.w, 'speed_control', None)
            if sc_ is not None:
                sc_.manual.add(a.id)
        # the reference lane: for 'tm' the lane the TM drives (where the vehicle would be without its offset)
        bx = x + math.sin(yaw) * row['cmd']
        by = y - math.cos(yaw) * row['cmd']
        wp = self.cmap.get_waypoint(carla.Location(float(bx), float(by), 0.0), project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            self.counts['off_road_updates'] += 1
            return
        px, py = np.array([x]), np.array([y])
        p = row['path']
        if p is not None:
            s_arr, d_arr = project(px, py, p)
            # off it: the vehicle is not where the TM would put it on this path (it took another branch at a
            # junction), or beyond its end
            off = abs(d_arr[0] - row['cmd']) > REBUILD_M if self.controller == 'tm' else abs(d_arr[0]) > 6.0
            if off or s_arr[0] > p['cum'][-1]:
                p = None
        if p is None:
            self.counts['path_builds'] += 1
            p = self._build(row, wp)
            if p is None:
                return
        s_self = self._advance(row, p, float(project(px, py, p)[0][0]))
        s_arr, d_arr = project(px, py, p)
        s_self, d_self = float(s_arr[0]), float(d_arr[0])
        j = int(np.clip(np.searchsorted(p['cum'], s_self) - 1, 0, len(p['wps']) - 2))
        here = p['wps'][j] if abs(p['cum'][j] - s_self) <= abs(p['cum'][j + 1] - s_self) else p['wps'][j + 1]
        row['d'], row['lane'] = d_self, [int(here.road_id), int(here.lane_id)]
        (le, lk), (re, rk) = self._bounds(here)
        ahead = next((q for q, c in zip(p['wps'], p['cum']) if c - s_self >= 15.0), None)
        if ahead is not None and not ahead.is_junction:   # a lane that ends ahead narrows the width now
            (le2, lk2), (re2, rk2) = self._bounds(ahead)
            if le2 > le:
                le, lk = le2, lk2
            if re2 < re:
                re, rk = re2, rk2
        hw, hl = row['hw'], row['hl']
        lo = le + hw + (EDGE_M if lk == 'kerb' else 0.0)
        hi = re - hw - (EDGE_M if rk == 'kerb' else 0.0)
        if self.controller == 'tm':                      # within the reach measured for the TM's controller
            lo, hi = max(lo, -TM_MAX_OFFSET_M), min(hi, TM_MAX_OFFSET_M)
        junction = any(q.is_junction for q, c in zip(p['wps'], p['cum']) if -2.0 <= c - s_self <= JUNCTION_M)
        lane_w = float(here.lane_width)
        # the neighbours, in the polyline's frame, with their extents and speeds along it
        m = (sc['id'] != sc['id'][k]) & ((sc['x'] - x) ** 2 + (sc['y'] - y) ** 2 < NEAR_M ** 2)
        idx = np.nonzero(m)[0]
        if len(idx):
            so, do = project(sc['x'][idx], sc['y'][idx], p)
            jj = np.clip(np.searchsorted(p['cum'], so) - 1, 0, len(p['x']) - 2)
            pyaw = np.arctan2(np.diff(p['y'])[jj], np.diff(p['x'])[jj])
            dyaw = sc['yaw'][idx] - pyaw
            c_, s_ = np.abs(np.cos(dyaw)), np.abs(np.sin(dyaw))
            others = {'s': so - s_self, 'd': do, 'hl': sc['hl'][idx] * c_ + sc['hw'][idx] * s_,
                      'hw': sc['hl'][idx] * s_ + sc['hw'][idx] * c_,
                      'v': sc['vx'][idx] * np.cos(pyaw) + sc['vy'][idx] * np.sin(pyaw), 'clear': sc['clear'][idx],
                      'sig': sc['sig'][idx]}
            keep = np.abs(others['d']) < 10.0
            others = {kk: vv[keep] for kk, vv in others.items()}
        else:
            others = {}
        if junction or hi < lo:
            target = min(max(0.0, lo), hi) if lo <= hi else 0.0
            res = choose(d_self, hw, hl, v, [target], others, target, row['target'], lane_w)
            row['mode'] = 'junction'
            self.counts['junction_updates'] += 1
        else:
            grid = np.arange(math.ceil(lo / STRIP_M), math.floor(hi / STRIP_M) + 1) * STRIP_M
            cands = np.unique(np.concatenate([grid, [lo, hi, min(max(0.0, lo), hi)]]))
            home = min(max(self.near_sign * HOME_SHARE * lane_w, lo), hi)
            res = choose(d_self, hw, hl, v, cands, others, home, row['target'], lane_w, self._want(row))
            target = res['target']
            row['mode'] = 'filter' if abs(target) > IGNORE_FROM_M or abs(d_self) > IGNORE_FROM_M else 'lane'
            if res['blocked']:
                self.counts['blocked_updates'] += 1
            self._sample(row, lane_w, lk, rk, d_self)
        row['target'] = target
        step = LAT_RATE_MPS * UPDATE_S
        row['cmd'] = float(row['cmd'] + max(-step, min(step, target - row['cmd'])))
        row['gap'] = res['gap']
        row['cap'] = cap_mps(res['gap'], res['lead_v'])
        self.counts['updates'] += 1
        if self.controller == 'tm':
            self._offset(row, row['cmd'])
            # the TM looks along its lane's centre whatever the offset (measured): off the centre, the clearance cap
            # covers what it does not see; its vehicle detection is switched off only when it would stop the vehicle
            # for an object on the centre the vehicle's own path clears (a filter-through)
            ignoring = False
            if row['mode'] == 'filter':
                centre = choose(0.0, hw, hl, v, [0.0], others, 0.0, 0.0, lane_w)
                ignoring = centre['gap'] < LOOK_M and centre['gap'] + 1.0 < res['gap']
            self._ignore(row, ignoring)
            self._speed(row, row['cap'] if row['mode'] == 'filter' else None)

    def _want(self, row) -> float:
        """The speed the vehicle wants (m/s): SpeedControl's last target (an imported world's road limit, breakers),
        else the road's limit less its difference."""
        a = row['a']
        sc = getattr(self.w, 'speed_control', None)
        kmh = sc.base.get(a.id) if sc is not None else None
        if kmh is None:
            try:
                kmh = float(a.get_speed_limit()) * (1.0 - row['diff'] / 100.0)
            except RuntimeError:
                kmh = 30.0
        return kmh / 3.6

    def _sample(self, row, lane_w, lk, rk, d_self):
        """The measured offset's histogram (0.25 m bins) and the share of samples straddling a line between two
        same-direction lanes."""
        b = min(int(abs(d_self) / 0.25), len(self.hist['abs_offset']) - 1)
        self.hist['abs_offset'][b] += 1
        self.hist['samples'] += 1
        half = lane_w / 2.0
        if (d_self + row['hw'] > half and rk == 'lane') or (d_self - row['hw'] < -half and lk == 'lane'):
            self.hist['between_lanes'] += 1

    # ------------------------------------------------------------------ 'tm' actuation
    def _offset(self, row, off, force=False):
        tm = self.w.tm
        if tm is None:
            return
        if force or row['sent_offset'] is None or abs(off - row['sent_offset']) > 0.02:
            tm.vehicle_lane_offset(row['a'], float(off))
            row['sent_offset'] = float(off)

    def _ignore(self, row, on: bool, force=False):
        tm = self.w.tm
        if tm is None or (row['ignoring'] == on and not force):
            return
        tm.ignore_vehicles_percentage(row['a'], 100.0 if on else 0.0)
        if on and not row['ignoring']:
            self.counts['ignore_on'] += 1
        row['ignoring'] = on

    def _speed(self, row, cap):
        """The clearance cap (m/s, None: none) on the TM's desired speed, through whoever sets it: SpeedControl (an
        imported world's road limits), ClassLimits (a stand-in in a town), else here from the road's limit."""
        a = row['a']
        kmh = None if cap is None else cap * 3.6
        if kmh is not None:
            self.counts['capped_updates'] += 1
        sc = getattr(self.w, 'speed_control', None)
        if sc is not None and a.id in sc.rows:
            sc.apply_cap(a.id, kmh)
            return
        cl = getattr(self.w, 'class_limits', None)
        if cl is not None and a.id in cl.rows:
            cl.apply_cap(a.id, kmh)
            return
        tm = self.w.tm
        if kmh is None and row['sent_kmh'] is None:
            return                                        # never capped: the TM's own speed
        try:
            lim = float(a.get_speed_limit())
        except RuntimeError:
            return
        base = lim * (1.0 - row['diff'] / 100.0)
        want = base if kmh is None else min(base, kmh)
        if row['sent_kmh'] is None or abs(want - row['sent_kmh']) > RESEND_KMH:
            tm.set_desired_speed(a, max(0.0, want))
            row['sent_kmh'] = want

    # ------------------------------------------------------------------ 'own' actuation
    def _geom(self, row):
        if row['geom'] is None:
            wb, steer = 1.2 * row['hl'], 70.0
            try:
                pc = row['a'].get_physics_control()
                ws = list(pc.wheels)
                if len(ws) >= 2:
                    f, r = ws[0].position, ws[-1].position
                    wb = max(math.dist((f.x, f.y), (r.x, r.y)) / 100.0, 0.5)
                    steer = max(float(ws[0].max_steer_angle), 10.0)
            except (RuntimeError, AttributeError):
                pass
            row['geom'] = (wb, math.radians(steer))
        return row['geom']

    def _drive(self, row, sc, k, dt):
        """The 'own' controller's control for this tick: pure pursuit to the path shifted by the offset, PI speed."""
        import numpy as np
        carla = self.w.carla
        a, p = row['a'], row['path']
        if p is None or not row['manual']:
            return
        x, y, yaw = sc['x'][k], sc['y'][k], sc['yaw'][k]
        v = float(math.hypot(sc['vx'][k], sc['vy'][k]))
        s_arr, _ = project(np.array([x]), np.array([y]), p)
        s_self = float(s_arr[0])
        wb, max_steer = self._geom(row)
        ld = min(max(LD_GAIN_S * v, LD_MIN_M), LD_MAX_M)
        st = s_self + ld
        j = int(np.clip(np.searchsorted(p['cum'], st) - 1, 0, len(p['x']) - 2))
        seg = max(p['cum'][j + 1] - p['cum'][j], 1e-6)
        t = min(max((st - p['cum'][j]) / seg, 0.0), 1.0)
        ux, uy = (p['x'][j + 1] - p['x'][j]) / seg, (p['y'][j + 1] - p['y'][j]) / seg
        tx = p['x'][j] + t * (p['x'][j + 1] - p['x'][j]) - uy * row['cmd']
        ty = p['y'][j] + t * (p['y'][j + 1] - p['y'][j]) + ux * row['cmd']
        dx, dy = tx - x, ty - y
        lx = dx * math.cos(yaw) + dy * math.sin(yaw)
        ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
        L2 = max(lx * lx + ly * ly, 1.0)
        delta = math.atan(wb * 2.0 * ly / L2)
        steer = max(-1.0, min(1.0, delta / max_steer))
        target = self._target_speed(row, a, sc, k, p, s_self, v)
        err = target - v
        if target < 0.3 and v < 0.5:
            ctl = carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0)
            row['i_err'] = 0.0
        elif err >= 0:
            row['i_err'] = min(row['i_err'] + err * dt, 10.0)
            ctl = carla.VehicleControl(throttle=float(min(max(KP * err + KI * row['i_err'], 0.0), 1.0)), steer=steer,
                                       brake=0.0)
        else:
            row['i_err'] = max(row['i_err'] + err * dt, 0.0)
            ctl = carla.VehicleControl(throttle=0.0, steer=steer,
                                       brake=float(min(max(-KB * err, 0.0), 1.0)) if err < -0.3 else 0.0)
        a.apply_control(ctl)

    def _target_speed(self, row, a, sc, k, p, s_self, v) -> float:
        """min(the road's limit (SpeedControl's target, else the limit less the vehicle's difference), the clearance
        cap, 0 at a red or yellow light (unless it ignores it, the profile's ignore_lights_pct), the junction yield)."""
        sc_ = getattr(self.w, 'speed_control', None)
        base = sc_.base.get(a.id) if sc_ is not None else None
        if base is None:
            try:
                base = float(a.get_speed_limit()) * (1.0 - row['diff'] / 100.0)
            except RuntimeError:
                base = 30.0
        cl = getattr(self.w, 'class_limits', None)
        if cl is not None and a.id in cl.rows and cl.rows[a.id][3]:
            base = min(base, float(cl.rows[a.id][3]))
        target = base / 3.6
        if row['cap'] is not None:
            target = min(target, row['cap'])
        at = getattr(a, 'is_at_traffic_light', None)
        if at is not None and at():
            light = a.get_traffic_light()
            lid = getattr(light, 'id', None)
            if lid != row['light']:
                row['light'] = lid
                row['ignore_light'] = float(row['rng'].uniform(0.0, 100.0)) < self.ignore_lights_pct
            state = str(a.get_traffic_light_state())
            if not row['ignore_light'] and state.rsplit('.', 1)[-1] in ('Red', 'Yellow'):
                if target > 0:
                    self.counts['lights_stopped'] += 1
                target = 0.0
        else:
            row['light'] = None
        if row['mode'] == 'junction' and target > 0:
            target = min(target, self._yield(row, sc, k, p, s_self, v, target))
        return target

    def _yield(self, row, sc, k, p, s_self, v, target) -> float:
        """Near or in a junction: a vehicle predicted (constant velocity) within reach of the vehicle's own predicted
        place on its path in the next YIELD_S, that is not behind it on its lane, makes it slow (to a stop when the
        meeting is within 1 s); the vehicle added first goes first between two of the filter's own."""
        import numpy as np
        x, y = sc['x'][k], sc['y'][k]
        m = (sc['id'] != sc['id'][k]) & ((sc['x'] - x) ** 2 + (sc['y'] - y) ** 2 < 30.0 ** 2)
        idx = np.nonzero(m)[0]
        if not len(idx):
            return target
        vv = max(v, target * 0.5)
        r_self = row['hl'] + 0.5
        for t in YIELD_S:
            st = s_self + vv * t
            j = int(np.clip(np.searchsorted(p['cum'], st) - 1, 0, len(p['x']) - 2))
            seg = max(p['cum'][j + 1] - p['cum'][j], 1e-6)
            u = min(max((st - p['cum'][j]) / seg, 0.0), 1.0)
            px = p['x'][j] + u * (p['x'][j + 1] - p['x'][j])
            py = p['y'][j] + u * (p['y'][j + 1] - p['y'][j])
            ox = sc['x'][idx] + sc['vx'][idx] * t
            oy = sc['y'][idx] + sc['vy'][idx] * t
            hit = np.hypot(ox - px, oy - py) < r_self + np.maximum(sc['hl'][idx], sc['hw'][idx])
            if not hit.any():
                continue
            for i in idx[hit]:
                # behind it, going its way: not a crossing
                rel = (sc['x'][i] - x) * math.cos(sc['yaw'][k]) + (sc['y'][i] - y) * math.sin(sc['yaw'][k])
                same = math.cos(sc['yaw'][i] - sc['yaw'][k]) > 0.7
                if same and rel < 0:
                    continue
                other = self.rows.get(int(sc['id'][i]))
                if other is not None and other['n'] > row['n']:
                    continue                              # the filter's own vehicle added later yields instead
                self.counts['yields'] += 1
                return 0.0 if t <= 1.0 else min(target, v * 0.5)
        return target

    # ------------------------------------------------------------------ output
    def log_rows(self) -> dict:
        """actors.jsonl 'lateral' per held vehicle: offset_m (measured, from its reference lane's centre, + right),
        target_m (commanded), lane [road, lane], mode, gap_m, cap_kmh, ignoring (the TM's vehicle detection off)."""
        out = {}
        for aid, r in self.rows.items():
            out[aid] = {'offset_m': round(r['d'], 3), 'target_m': round(r['cmd'], 3), 'lane': r['lane'],
                        'mode': r['mode'], 'gap_m': round(r['gap'], 2),
                        'cap_kmh': None if r['cap'] is None else round(r['cap'] * 3.6, 1)}
            if self.controller == 'tm':
                out[aid]['ignoring'] = r['ignoring']
        return out

    def frame_summary(self) -> dict:
        """frames.jsonl 'lateral': the vehicles held, off their lane centre (mode filter), at a junction, held."""
        modes = {}
        for r in self.rows.values():
            modes[r['mode']] = modes.get(r['mode'], 0) + 1
        return {'vehicles': len(self.rows), **{m: modes.get(m, 0) for m in ('filter', 'lane', 'junction', 'held')}}

    def report(self) -> dict:
        import numpy as np
        ms = np.array(self.ms) if self.ms else np.zeros(1)
        n = max(self.hist['samples'], 1)
        return {'enabled': True, 'controller': self.controller, 'vehicles_kinds': sorted(self.kinds),
                'handedness': 'left' if self.near_sign < 0 else 'right', 'by_kind': dict(self.by_kind),
                **self.counts, 'parameters': parameters(),
                'offset_histogram': {'bin_m': 0.25, 'counts': list(self.hist['abs_offset']),
                                     'samples': self.hist['samples']},
                'between_lanes_share': round(self.hist['between_lanes'] / n, 4),
                'step_ms': {'mean': round(float(ms.mean()), 3), 'p95': round(float(np.percentile(ms, 95)), 3),
                            'calls': len(self.ms), 'what': 'wall ms of each step() that worked: every update_s '
                                                          '(\'tm\'), every tick (\'own\')'},
                'method': ('workers.lateral: a gap-seeking chooser over the same-direction width every '
                           f'{UPDATE_S:g} s; ' + ('TrafficManager.vehicle_lane_offset, the TM\'s vehicle detection '
                                                  'off while off the lane centre and the chooser\'s clearance capping '
                                                  'its desired speed' if self.controller == 'tm' else
                                                  'own pure-pursuit and PI control off the autopilot'))}
