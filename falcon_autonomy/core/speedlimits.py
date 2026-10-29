"""Speed limits and speed breakers of an imported (OpenDRIVE) world, for traffic that obeys them.

CARLA 0.9.16's traffic manager drives every autopilot vehicle at Vehicle.get_speed_limit(): the limit of the last
speed sign it passed, else 30 km/h. A standalone OpenDRIVE world has no speed signs — the file's <speed> records are
not read by the traffic manager (measured: a road written at 20 km/h was driven at 28.9 km/h; type-274 speed-limit
signals become landmarks but spawn no sign, docs/worlds.md) — so every vehicle drove 30 km/h on every road. This
module reads the limits from the file itself; workers.speedcontrol gives each vehicle its road's limit with
TrafficManager.set_desired_speed as it moves.

RoadSpeeds: road id → its lane sections → each lane's <speed sOffset max unit> records (OpenDRIVE 1.4 §3.5.6; the
unit m/s when not given), and the road's own <type s><speed max unit/></type> records as the fallback.
CARLA's Waypoint gives road_id, lane_id and s (along the road from its start), which is all a lookup needs.

Breakers: the speed breakers an import placed (features.json, core.roadfeatures: OpenDRIVE frame x, y, hdg, each
with its crossing speed). A vehicle with a breaker ahead in its path slows to that crossing speed before it — a
heuristic the traffic manager lacks (it does not see road objects at all).
"""
from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

UNIT_TO_MPS = {'m/s': 1.0, 'km/h': 1.0 / 3.6, 'mph': 0.44704}

# The approach to a breaker (heuristic): a speed profile that decelerates at APPROACH_DECEL_MPS2 to the breaker's
# crossing speed at the breaker (sqrt(vc² + 2·a·d), less one update's travel and a margin), the crossing speed held
# until the vehicle's middle is PAST_M beyond it. 2 m/s² is a comfortable deceleration (well under the 3.4 m/s² AASHTO's stopping-sight
# distance assumes); the crossing speeds are core.roadfeatures' (IRC:99-1988: 25 km/h for its recommended hump).
APPROACH_DECEL_MPS2 = 2.0
APPROACH_MARGIN_M = 5.0
PAST_M = 3.0
NEAR_M = 10.0                       # within this far, a breaker is found along the vehicle's own heading
PATH_STEP_M = 5.0                   # the spacing of the look-ahead's waypoints (workers.speedcontrol)
ALONG_COS = 0.7                     # within about 45° of the breaker's road direction (either way)
LATERAL_MIN_M = 4.0                 # a breaker spans the carriageway: within its half width (+ a lane's half) of it


def _mps(el) -> float | None:
    try:
        v = float(el.get('max'))
    except (TypeError, ValueError):
        return None                 # "no limit" / "undefined" (OpenDRIVE allows both): not a limit
    f = UNIT_TO_MPS.get((el.get('unit') or 'm/s').strip())
    return v * f if f is not None and v > 0 else None


class RoadSpeeds:
    """The speed records of an OpenDRIVE file, looked up by (road id, lane id, s)."""

    def __init__(self, roads: dict):
        # road id → {'sections': [(s0, {lane id: [(sOffset, m/s), …]})], 'types': [(s, m/s), …]}
        self.roads = roads

    @classmethod
    def from_xodr(cls, text: str) -> 'RoadSpeeds':
        root = ET.fromstring(text)
        roads = {}
        for road in root.iter('road'):
            types = []
            for t in road.findall('type'):
                sp = t.find('speed')
                v = _mps(sp) if sp is not None else None
                if v is not None:
                    types.append((float(t.get('s', 0.0)), v))
            sections = []
            lanes_el = road.find('lanes')
            for sec in (lanes_el.findall('laneSection') if lanes_el is not None else []):
                lanes = {}
                for lane in sec.iter('lane'):
                    recs = [(float(sp.get('sOffset', 0.0)), v) for sp in lane.findall('speed')
                            if (v := _mps(sp)) is not None]
                    if recs:
                        lanes[int(lane.get('id'))] = sorted(recs)
                sections.append((float(sec.get('s', 0.0)), lanes))
            sections.sort(key=lambda x: x[0])
            if types or any(l for _, l in sections):
                roads[int(road.get('id'))] = {'sections': sections, 'types': sorted(types)}
        return cls(roads)

    def __len__(self):
        return len(self.roads)

    def lookup(self, road_id: int, lane_id: int, s: float) -> tuple[float | None, str | None]:
        """(limit in m/s, 'lane' | 'road') at the position, or (None, None) when the file gives none there."""
        r = self.roads.get(int(road_id))
        if r is None:
            return None, None
        s = max(float(s), 0.0)
        sec = None
        for s0, lanes in r['sections']:
            if s0 <= s + 1e-6:
                sec = (s0, lanes)
            else:
                break
        if sec is not None:
            recs = sec[1].get(int(lane_id))
            if recs:
                v = recs[0][1]
                for off, mps in recs:
                    if off <= s - sec[0] + 1e-6:
                        v = mps
                return v, 'lane'
        v = None
        for s0, mps in r['types']:
            if s0 <= s + 1e-6:
                v = mps
        return (v, 'road') if v is not None else (None, None)

    def summary(self) -> dict:
        kmh = sorted({round(v * 3.6, 1) for r in self.roads.values()
                      for _, lanes in r['sections'] for recs in lanes.values() for _, v in recs}
                     | {round(v * 3.6, 1) for r in self.roads.values() for _, v in r['types']})
        return {'roads_with_limits': len(self.roads), 'limits_kmh': kmh}


class Breakers:
    """The breakers of an import (features.json 'placed') in CARLA's frame (x, −y of OpenDRIVE's), with the speed
    each is crossed at; cap_kmh() is the speed a vehicle should drive at given the breakers ahead of it."""

    def __init__(self, placed: list):
        rows = [p for p in placed or [] if p.get('crossing_kmh')]
        self.rows = rows
        self.xy = np.array([[float(p['x']), -float(p['y'])] for p in rows], float).reshape(-1, 2)
        self.kmh = np.array([float(p['crossing_kmh']) for p in rows], float)
        self.lat = np.array([max(float(p.get('half_width_left_m') or 0.0), float(p.get('half_width_right_m') or 0.0))
                             + 1.75 for p in rows], float)
        self.lat = np.maximum(self.lat, LATERAL_MIN_M)
        # the road's direction at each breaker (OpenDRIVE hdg; CARLA's yaw is its negative): a breaker counts only for
        # a vehicle driving along that road, either way — not for one crossing it at a junction
        h = np.array([-float(p['hdg']) if p.get('hdg') is not None else np.nan for p in rows], float)
        self.dir = np.stack([np.cos(h), np.sin(h)], axis=1) if len(rows) else np.zeros((0, 2))

    @classmethod
    def from_features(cls, path: str) -> 'Breakers':
        if not path or not os.path.isfile(path):
            return cls([])
        with open(path, encoding='utf-8') as fh:
            return cls((json.load(fh) or {}).get('placed') or [])

    def __len__(self):
        return len(self.rows)

    allowed_mps = staticmethod(lambda dist_m, cross_kmh, period_s, v_mps: allowed_mps(dist_m, cross_kmh, period_s,
                                                                                        v_mps))
    approach_m = staticmethod(lambda v_mps, cross_kmh, period_s: approach_m(v_mps, cross_kmh, period_s))

    def cap_kmh(self, x: float, y: float, yaw_deg: float, v_mps: float, period_s: float,
                path: list | None = None) -> float | None:
        """The speed (km/h) the breakers ahead allow a vehicle at (x, y) heading yaw_deg (CARLA's frame) at speed
        v_mps: the lowest allowed_mps over the breakers in its path, the crossing speed itself on a breaker and until
        PAST_M beyond it; None when no breaker is in its path.

        path: points ahead along the vehicle's lane, [(distance along the lane, x, y, yaw), …] (Map waypoints'
        next()): a breaker within its lateral reach of a point is that far ahead — on a curve a straight line from
        the vehicle misses it (on a 250 m radius a point 100 m ahead lies 20 m off the tangent). Without a path, or
        within NEAR_M, the straight line is used."""
        if not len(self.rows):
            return None
        f = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
        rel = self.xy - np.array([x, y])
        along = rel @ f
        lat = np.abs(rel[:, 0] * f[1] - rel[:, 1] * f[0])
        par = np.abs(self.dir @ f)
        straight = (lat <= self.lat) & (along >= -PAST_M) & ((par >= ALONG_COS) | np.isnan(par))
        if path:
            straight &= along <= NEAR_M
        dist = np.where(straight, np.maximum(along, 0.0), np.inf)
        for d, px, py, pyaw in path or []:
            g = np.array([math.cos(math.radians(pyaw)), math.sin(math.radians(pyaw))])
            r = self.xy - np.array([px, py])
            a = r @ g
            pg = np.abs(self.dir @ g)
            hit = (np.abs(r[:, 0] * g[1] - r[:, 1] * g[0]) <= self.lat) & (np.abs(a) <= PATH_STEP_M) & \
                ((pg >= ALONG_COS) | np.isnan(pg))
            dist = np.where(hit, np.minimum(dist, np.maximum(d + a, 0.0)), dist)
        ok = np.isfinite(dist)
        if not ok.any():
            return None
        return float(min(allowed_mps(float(dd), float(k), period_s, v_mps) * 3.6
                         for dd, k in zip(dist[ok], self.kmh[ok])))


def allowed_mps(dist_m: float, target_kmh: float, period_s: float, v_mps: float) -> float:
    """The speed a vehicle dist_m before a slower stretch (a breaker's crossing speed, a lower limit) may drive at:
    the speed from which a steady APPROACH_DECEL_MPS2 brings it to target_kmh there, less the travel of one update
    (the target is re-read only every period_s) and a margin: sqrt(vt² + 2·a·max(0, d − v·T − margin))."""
    vt = target_kmh / 3.6
    d = max(0.0, dist_m - v_mps * period_s - APPROACH_MARGIN_M)
    return math.sqrt(vt * vt + 2.0 * APPROACH_DECEL_MPS2 * d)


def approach_m(v_mps: float, target_kmh: float, period_s: float) -> float:
    """How far before a slower stretch a vehicle at v starts slowing (where allowed_mps falls below v)."""
    vt = target_kmh / 3.6
    return max(0.0, (v_mps * v_mps - vt * vt) / (2.0 * APPROACH_DECEL_MPS2)) + v_mps * period_s + APPROACH_MARGIN_M


def features_path(xodr_path: str) -> str:
    """features.json next to a world's .xodr (written by the import with its road features)."""
    return os.path.join(os.path.dirname(os.path.abspath(xodr_path)), 'features.json')
