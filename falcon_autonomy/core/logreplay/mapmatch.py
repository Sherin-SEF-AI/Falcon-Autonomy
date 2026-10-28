"""Map matching: does a log's ego trajectory run on the driving lanes of a world? (numpy)

The measure. The ego's logged positions, taken every MATCH_STEP_M of path (so a long stop does not weigh more than a
metre of driving), are compared with the world's driving-lane centre points (every 2 m or finer, each with its
heading and lane width). For each position, among the lane points within SEARCH_M, the lane is the nearest one whose
direction is within HEADING_TOL_DEG of the ego's heading ("in its direction"); its lateral distance is measured along
that lane point's normal. Reported: the share of positions on a driving lane in their direction (lateral distance
≤ half the lane's width), on a driving lane of either direction, the wrong way only, and the median, 95th percentile
and largest lateral distance to the nearest lane centre in the ego's direction.

The verdict (thresholds heuristic, stated in every result): 'match' when at least 90 % of the positions are on a lane
in their direction and the 95th percentile of the lateral distance is at most 2 m; 'partial' when at least half are
on some driving lane; else 'none'. A replay whose road should match ('matched' or 'geo' placement) is refused when the
verdict is 'none', unless the caller allows it; a log that does not match runs only as "no matching road".

Where the lanes come from: the server's own map (the replay worker: Map.generate_waypoints, CARLA's lanes — the
authoritative check); offline, a Falcon run's lanes.json of the same world, or an OpenDRIVE file (lanes_from_xodr:
each driving lane's centre from the reference line, laneOffset and the lane widths, every 1 m; elevation and lateral
profiles are not modelled, superelevation ignored — planar).

Limits: a lane is matched by position and heading only — not by the lane graph (a log that jumps between lanes still
matches lane by lane); inside junctions the overlapping turning lanes make "in its direction" lenient; the
nearest-point search over 2 m spacing overstates a lateral distance by at most about 1 m along a curve's chord.
"""
from __future__ import annotations

import math

import numpy as np

MATCH_STEP_M = 1.0
SEARCH_M = 8.0
HEADING_TOL_DEG = 45.0
THRESHOLDS = {'match_share_in_direction': 0.9, 'match_p95_lateral_m': 2.0, 'partial_share_on_lane': 0.5}
RULE = ('heuristic: match if ≥ 90 % of positions are on a driving lane in their direction and the 95th percentile of '
        'the lateral distance ≤ 2 m; partial if ≥ 50 % are on some driving lane; else none')


def lanes_from_lanes_json(doc: dict) -> np.ndarray:
    """(N, 4) x, y, yaw (degrees), width of a lanes.json (CARLA's frame)."""
    cols = doc['columns']
    ix, iy, iyaw, iw = (cols.index(k) for k in ('x', 'y', 'yaw', 'width'))
    pts = doc.get('points') or []
    return np.array([[p[ix], p[iy], p[iyaw], p[iw]] for p in pts], float).reshape(-1, 4)


def _poly(entries, s):
    """A cubic a + b·u + c·u² + d·u³ from the last entry whose start ≤ s; 0 when none."""
    best = None
    for e in entries:
        if e[0] <= s + 1e-9:
            best = e
    if best is None:
        return 0.0
    u = s - best[0]
    return best[1] + best[2] * u + best[3] * u * u + best[4] * u ** 3


def lanes_from_xodr(text, step: float = 1.0) -> np.ndarray:
    """(N, 4) x, y, yaw (degrees), width of every driving lane's centre of an OpenDRIVE document, every `step` m, in
    CARLA's frame (OpenDRIVE's y negated, its heading negated; a lane runs along +s when it is a right lane of a
    right-hand road or a left lane of a left-hand one). The header's <offset> is not applied (CARLA builds its map
    from the planView coordinates as they are — not verified on a file with an offset)."""
    from .. import xodr as X
    od = X.parse(text)
    rows = []
    for road in od.roads:
        if not road.sections:
            continue
        for g in road.geometries:
            if g.length <= 0:
                continue
            n = max(1, math.ceil(g.length / step))
            ds = np.linspace(0.0, g.length, n + 1)
            xs, ys, hs = X.eval_geometry(g, ds)
            for x, y, h, s in zip(xs, ys, hs, g.s + ds):
                sec = road.sections[0]
                for cand in road.sections:
                    if cand.s <= s + 1e-9:
                        sec = cand
                ls = s - sec.s
                base = _poly(road.lane_offsets, s)
                nx, ny = -math.sin(h), math.cos(h)            # OpenDRIVE t: to the left of the reference line
                for side, lanes, sign in (('right', sec.right, -1.0), ('left', sorted(sec.left, key=lambda q: q.id),
                                                                       1.0)):
                    t = base
                    for ln in lanes:
                        w = _poly(ln.widths, ls)
                        c = t + sign * w / 2.0
                        t += sign * w
                        if ln.type != 'driving' or w <= 0.05:
                            continue
                        along = (side == 'right') != road.lht
                        hd = h if along else h + math.pi
                        rows.append((x + nx * c, -(y + ny * c), -math.degrees(hd), w))
    a = np.array(rows, float).reshape(-1, 4)
    if len(a):
        a[:, 2] = (a[:, 2] + 180.0) % 360.0 - 180.0
    return a


class _Grid:
    def __init__(self, pts: np.ndarray, cell: float):
        self.cell = cell
        self.pts = pts
        keys = np.floor(pts[:, :2] / cell).astype(np.int64)
        self.buckets = {}
        for i, k in enumerate(map(tuple, keys)):
            self.buckets.setdefault(k, []).append(i)

    def near(self, x, y, r):
        c = self.cell
        k0, k1 = int(math.floor((x - r) / c)), int(math.floor((x + r) / c))
        j0, j1 = int(math.floor((y - r) / c)), int(math.floor((y + r) / c))
        out = []
        for i in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                out.extend(self.buckets.get((i, j), ()))
        return np.asarray(out, int)


def resample_path(x, y, yaw, step: float = MATCH_STEP_M) -> np.ndarray:
    """Positions every `step` m of path (the first always): (K, 3) x, y, yaw."""
    x, y, yaw = (np.asarray(v, float) for v in (x, y, yaw))
    keep, acc = [0], 0.0
    for i in range(1, len(x)):
        acc += math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        if acc >= step:
            keep.append(i)
            acc = 0.0
    return np.column_stack([x[keep], y[keep], yaw[keep]])


def match(ego_x, ego_y, ego_yaw, lanes: np.ndarray, source: str = '') -> dict:
    """The match of an ego trajectory (CARLA's frame) with lane centre points (lanes: (N, 4) x, y, yaw, width)."""
    P = resample_path(ego_x, ego_y, ego_yaw)
    out = {'rule': RULE, 'thresholds': THRESHOLDS, 'lanes_from': source, 'lane_points': int(len(lanes)),
           'positions': int(len(P)), 'step_m': MATCH_STEP_M}
    if not len(lanes):
        out.update(verdict='none', why='the world has no driving lanes')
        return out
    grid = _Grid(lanes, SEARCH_M)
    lat_dir, on_dir, on_any, wrong = [], 0, 0, 0
    for x, y, yaw in P:
        idx = grid.near(x, y, SEARCH_M)
        if len(idx):
            L = lanes[idx]
            d = np.hypot(L[:, 0] - x, L[:, 1] - y)
            close = d <= SEARCH_M
            L, d = L[close], d[close]
        else:
            L, d = lanes[:0], np.zeros(0)
        if not len(L):
            lat_dir.append(math.inf)
            continue
        a = np.radians(L[:, 2])
        lateral = np.abs((x - L[:, 0]) * -np.sin(a) + (y - L[:, 1]) * np.cos(a))
        along = np.abs((x - L[:, 0]) * np.cos(a) + (y - L[:, 1]) * np.sin(a))
        hd = np.abs((yaw - L[:, 2] + 180.0) % 360.0 - 180.0)
        inside = (lateral <= L[:, 3] / 2.0) & (along <= 1.5)
        ok_dir = hd <= HEADING_TOL_DEG
        if (inside & ok_dir).any():
            on_dir += 1
        if inside.any():
            on_any += 1
            if not (inside & ok_dir).any():
                wrong += 1
        if ok_dir.any():
            k = np.flatnonzero(ok_dir)[int(np.argmin(d[ok_dir]))]
            lat_dir.append(float(lateral[k]))
        else:
            lat_dir.append(math.inf)
    n = len(P)
    lat = np.asarray(lat_dir, float)
    fin = lat[np.isfinite(lat)]
    share_dir, share_any = on_dir / n, on_any / n
    p95 = float(np.percentile(np.where(np.isfinite(lat), lat, 1e9), 95))
    p95 = p95 if p95 <= SEARCH_M else None              # beyond the search radius: no lane in reach
    out.update(share_on_lane_in_direction=round(share_dir, 4), share_on_lane=round(share_any, 4),
               share_wrong_way=round(wrong / n, 4),
               share_no_lane_within_m=round(float(np.mean(~np.isfinite(lat))), 4), search_m=SEARCH_M,
               median_lateral_m=round(float(np.median(fin)), 3) if len(fin) else None,
               p95_lateral_m=None if p95 is None else round(p95, 3),
               max_lateral_m=round(float(fin.max()), 3) if len(fin) else None)
    if share_dir >= THRESHOLDS['match_share_in_direction'] and p95 is not None and \
            p95 <= THRESHOLDS['match_p95_lateral_m']:
        out['verdict'] = 'match'
    elif share_any >= THRESHOLDS['partial_share_on_lane']:
        out['verdict'] = 'partial'
    else:
        out['verdict'] = 'none'
    out['why'] = (f'{100 * share_dir:.1f} % of {n} positions on a driving lane in their direction, '
                  f'{100 * share_any:.1f} % on any; lateral distance median '
                  f'{out["median_lateral_m"] if out["median_lateral_m"] is not None else "—"} m, 95th percentile '
                  f'{out["p95_lateral_m"] if out["p95_lateral_m"] is not None else f"> {SEARCH_M:g}"} m')
    return out
