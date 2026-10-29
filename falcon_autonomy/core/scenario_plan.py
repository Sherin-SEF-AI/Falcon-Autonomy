"""A scenario's plan: where its actors would go over time on a straight, schematic road, and when its timeline
would fire — for the scenario editor's preview and its time axis (standard library only).

This is a plan, not a simulation. It reads one concrete scenario (core.scenarios.sample/concrete) and applies the
director's own rules (workers.director) to point actors on a straight road:

* lanes are 3.5 m wide (LANE_W, assumed: a map's lanes differ); offsets follow the director's placement — the
  adjacent lane on the overtaking side, the opposite lane beyond it when the scenario also uses an adjacent lane,
  the kerb row against the lane's edge (the actor's half width + 0.2 m), the sidewalk 2 m past the lane's edge;
* a vehicle's set speed ramps to an action's target at its acceleration or deceleration (the director's speed
  controller); a lane change moves the lateral offset linearly over its duration; a walker stands until "walk",
  then walks across the road from its side (or along it) at its speed;
* the ego holds its set speed on a straight route: the lane follower's braking for what is ahead and the traffic
  manager (driver "autopilot") are NOT planned — a distance or "ahead" trigger that depends on the ego slowing
  fires earlier here than in a run;
* triggers as the director checks them: time ≥ t_s; the ego within lt_m of the actor (centre to centre);
  the actor's rear more than gt_m ahead of the ego's front (lengths from SIZES, typical values).

Coordinates: x along the ego's direction from the ego's start, y to the kerb side (right in right-hand traffic).
"""
from __future__ import annotations

import math

LANE_W = 3.5
EGO_SIZE = (4.6, 2.0)
SIZES = {'bus': (10.3, 3.9), 'truck': (6.0, 2.6), 'van': (5.5, 2.0), 'motorcycle': (2.2, 0.9),
         'bicycle': (1.7, 0.6), 'pedestrian': (0.6, 0.6), 'child': (0.5, 0.5), 'autorickshaw': (2.6, 1.3),
         'cycle_rickshaw': (2.2, 1.0), 'cattle': (1.6, 0.5), 'handcart': (2.4, 1.2), 'vendor': (2.3, 1.5)}
WALKERS = ('pedestrian', 'child')
NOTE = ('A plan, not a simulation: a straight schematic road with 3.5 m lanes. The ego holds its set speed — the '
        'lane follower\'s braking and the autopilot are not planned — and the actors follow the director\'s rules '
        'without physics. A run in CARLA is what counts.')


def size_of(kind):
    return SIZES.get(kind, (4.6, 2.0))


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def lane_offset(a: dict, uses_adjacent: bool) -> float:
    """An actor's lateral position (m, + towards the kerb) from the ego lane's centre, as the director places it."""
    place = a.get('place') or {}
    lane, lat = place.get('lane', 'ego'), _f(place.get('lateral_m'))
    half = LANE_W / 2.0
    if lane == 'adjacent':
        return -LANE_W + lat
    if lane == 'opposite':
        return -(2 if uses_adjacent else 1) * LANE_W + lat
    if lane == 'line':
        return -half + lat
    if lane == 'kerb':
        return half + size_of(a.get('kind'))[1] / 2.0 + 0.2 - lat
    if lane == 'sidewalk':
        return half + 2.0 + lat
    if lane == 'far_sidewalk':
        return -(half + LANE_W + 2.0) + lat
    return lat


def plan(conc: dict, dt: float = 0.05, duration_s: float | None = None) -> dict:
    """The plan of a concrete scenario: {'t': [...], 'tracks': {id: [(x, y)]} ('ego' included), 'fired': [(event
    index, t)], 'actions': [(event index, action index, t, t_end or None)], 'sizes': {id: (L, W)}, 'note'}."""
    actors = conc.get('actors') or []
    uses_adj = any((a.get('place') or {}).get('lane') == 'adjacent' for a in actors) or \
        'adjacent_same' in (conc.get('needs') or [])
    T = _f(duration_s if duration_s is not None else conc.get('duration_s'), 12.0)
    n = max(int(round(T / dt)), 1)
    ego_v = _f((conc.get('ego') or {}).get('speed_kmh'), 30.0) / 3.6
    st = {}
    for a in actors:
        place = a.get('place') or {}
        lane, heading = place.get('lane', 'ego'), a.get('heading', 'lane')
        walker = a.get('kind') in WALKERS
        moving = a.get('start', 'moving') == 'moving'
        v = 0.0 if walker or not moving else _f(a.get('speed_kmh')) / 3.6
        if heading == 'across':
            d = (0.0, -1.0)                       # from its side across the road (towards the far side)
        elif lane == 'opposite' or heading == 'against':
            d = (-1.0, 0.0)
        else:
            d = (1.0, 0.0)
        y0 = lane_offset(a, uses_adj)
        st[a['id']] = {'x': _f(place.get('ahead_m')), 'y': y0, 'd': d, 'v': v, 'target': v, 'accel': 3.0,
                       'walker': walker, 'walking': False, 'walk_v': 0.0, 'lat_from': y0, 'lat_to': y0,
                       'lat_t0': 0.0, 'lat_dur': 0.0, 'kind': a.get('kind')}
    ex = 0.0
    out = {'t': [], 'tracks': {'ego': []}, 'fired': [], 'actions': [], 'sizes': {'ego': EGO_SIZE}, 'note': NOTE,
           'unknown': []}
    for aid, s in st.items():
        out['tracks'][aid] = []
        out['sizes'][aid] = size_of(s['kind'])
    fired = set()
    timeline = conc.get('timeline') or []
    for k in range(n + 1):
        t = k * dt
        # the timeline, as the director checks it before the step
        for i, ev in enumerate(timeline):
            if i in fired:
                continue
            tr = ev.get('trigger') or {}
            go = False
            if tr.get('type') == 'time':
                go = t >= _f(tr.get('t_s'), math.inf)
            elif tr.get('type') in ('distance', 'ahead') and tr.get('actor') in st:
                s = st[tr['actor']]
                if tr['type'] == 'distance':
                    go = math.hypot(s['x'] - ex, s['y']) < _f(tr.get('lt_m'), -1.0)
                else:
                    gap = (s['x'] - ex) - size_of(s['kind'])[0] / 2.0 - EGO_SIZE[0] / 2.0
                    go = gap > _f(tr.get('gt_m'), math.inf)
            if not go:
                continue
            fired.add(i)
            out['fired'].append((i, round(t, 3)))
            for j, act in enumerate(ev.get('actions') or []):
                s = st.get(act.get('actor'))
                end = None
                if s is None:
                    out['unknown'].append((i, j))
                    out['actions'].append((i, j, round(t, 3), None))
                    continue
                do = act.get('do')
                if do == 'speed':
                    s['target'], s['accel'] = _f(act.get('kmh')) / 3.6, max(_f(act.get('accel'), 3.0), 1e-3)
                    end = t + abs(s['target'] - s['v']) / s['accel']
                elif do in ('brake', 'stop'):
                    s['target'] = 0.0
                    s['accel'] = max(_f(act.get('decel'), 6.0), 1e-3) if do == 'brake' else 6.0
                    end = t + s['v'] / s['accel']
                elif do == 'go':
                    s['target'] = max(s['target'], 5.0)
                elif do == 'walk':
                    s['walking'], s['walk_v'] = True, _f(act.get('speed_mps'), 1.4)
                elif do == 'lane_change':
                    to = act.get('to', 'ego')
                    target = {'ego': 0.0, 'kerb': LANE_W, 'adjacent': -LANE_W}.get(to, 0.0)
                    s.update(lat_from=s['y'], lat_to=target, lat_t0=t, lat_dur=max(_f(act.get('duration_s'), 2.0),
                                                                                    1e-3))
                    end = t + s['lat_dur']
                out['actions'].append((i, j, round(t, 3), None if end is None else round(min(end, T), 3)))
        out['t'].append(round(t, 4))
        out['tracks']['ego'].append((round(ex, 3), 0.0))
        for aid, s in st.items():
            out['tracks'][aid].append((round(s['x'], 3), round(s['y'], 3)))
        # then everyone moves for one step
        ex += ego_v * dt
        for s in st.values():
            if s['walker']:
                if s['walking']:
                    s['x'] += s['d'][0] * s['walk_v'] * dt
                    s['y'] += s['d'][1] * s['walk_v'] * dt
                continue
            if s['v'] < s['target']:
                s['v'] = min(s['target'], s['v'] + s['accel'] * dt)
            elif s['v'] > s['target']:
                s['v'] = max(s['target'], s['v'] - s['accel'] * dt)
            s['x'] += s['d'][0] * s['v'] * dt
            s['y'] += s['d'][1] * s['v'] * dt
            if s['lat_dur'] > 0:
                f = min(max((t + dt - s['lat_t0']) / s['lat_dur'], 0.0), 1.0)
                s['y'] = s['lat_from'] + (s['lat_to'] - s['lat_from']) * f
    return out
