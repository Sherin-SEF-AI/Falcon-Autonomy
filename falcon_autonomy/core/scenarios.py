"""Scenarios: actors placed around the ego, a timeline of triggers and actions, parameters as distributions
(standard library + numpy; the runner is falcon_autonomy.workers.director).

A scenario document (the project's scenarios, and the built-in LIBRARY):

    {"id", "name", "about", "duration_s": expr, "needs": ["adjacent_same", "opposite", "junction_turn", …],
     "params": {name: {"dist": "fixed" | "uniform" | "normal" | "choice", "value" | "lo","hi" | "mean","sd" |
                       "choices", "unit", "about"}},
     "ego": {"speed_kmh": expr, "driver": "follower" | "none" | "autopilot", "route": "straight" | "turn_across"},
     "weather": {CARLA weather: a preset and fields} (optional: replaces the world's for this scenario),
     "actors": [{"id", "kind": car | truck | bus | van | motorcycle | bicycle | pedestrian | child
                         | autorickshaw | cycle_rickshaw (stand-ins) | cattle | handcart | vendor (proxies),
                 "place": {"lane": "ego" | "adjacent" | "line" | "opposite" | "kerb" | "sidewalk" | "far_sidewalk",
                           "ahead_m": expr, "lateral_m": expr},
                 "heading": "lane" | "against" | "across",
                 "speed_kmh": expr, "start": "moving" | "stopped"}],
     "timeline": [{"trigger": {"type": "time", "t_s": expr} | {"type": "distance", "actor": id, "lt_m": expr}
                              | {"type": "ahead", "actor": id, "gt_m": expr},
                   "actions": [{"actor": id, "do": "speed", "kmh": expr, "accel": expr}
                               | {"actor": id, "do": "brake", "decel": expr}
                               | {"actor": id, "do": "lane_change", "to": "ego" | "adjacent" | "kerb",
                                  "duration_s": expr}
                               | {"actor": id, "do": "walk", "speed_mps": expr}
                               | {"actor": id, "do": "open_door", "door": "FL" | "FR" | "RL" | "RR"}
                               | {"actor": id, "do": "go"} | {"actor": id, "do": "stop"}]}],
     "pass": {"no_collision": true, "min_ttc_s": expr}}

The Indian kinds (core.assets, docs/indian-assets.md): autorickshaw and cycle_rickshaw are stand-in vehicles driven
within their class's top speed and acceleration; cattle, handcart and vendor are proxy props of the real object's
size (a moving one is moved kinematically). Their labels keep CARLA's class and carry Falcon's subclass.

An expression is a number or "$name" (a parameter). sample(scenario, seed) draws every parameter from its
distribution with a seeded generator and substitutes it: the same seed gives the same concrete scenario. Sweeps and
the rare-case search set parameters explicitly (concrete(scenario, values)).

Lanes are relative to the ego and the traffic's handedness: "adjacent" is the next lane in the ego's direction (the
overtaking side), "opposite" the nearest lane of the other direction, "kerb" the edge of the road on the ego's
near side (left in left-hand traffic, right in right-hand), "sidewalk" beyond the kerb, "far_sidewalk" beyond the
opposite lanes, "line" the lane line between the ego's lane and the adjacent one (filtering two-wheelers). A
"distance" trigger fires when the ego comes within lt_m of the actor; an "ahead" trigger when the actor's rear is more
than gt_m ahead of the ego's front. A scenario that needs a road feature the map lacks around the chosen start is placed elsewhere or
refused (the runner says which).
"""
from __future__ import annotations

import copy
import random

SCENARIO_SCHEMA = 'falcon.scenario/1'
DISTS = ('fixed', 'uniform', 'normal', 'choice')


def _p(dist, unit='', about='', **kw):
    return dict({'dist': dist, 'unit': unit, 'about': about}, **kw)


def U(lo, hi, unit='', about=''):
    return _p('uniform', unit, about, lo=lo, hi=hi)


LIBRARY = {
    'lead_brake': {
        'name': 'Lead vehicle brakes', 'about': 'A car ahead in the ego\'s lane brakes hard.',
        'needs': [], 'duration_s': 12.0,
        'params': {'speed': U(20, 60, 'km/h', 'speed of both before the brake'),
                   'gap': U(8, 40, 'm', 'gap at the start'), 'decel': U(3, 9, 'm/s²', 'the lead\'s braking'),
                   'brake_at': U(2, 5, 's', 'when it brakes')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'lead', 'kind': 'car', 'place': {'lane': 'ego', 'ahead_m': '$gap'}, 'heading': 'lane',
                    'speed_kmh': '$speed', 'start': 'moving'}],
        'timeline': [{'trigger': {'type': 'time', 't_s': '$brake_at'},
                      'actions': [{'actor': 'lead', 'do': 'brake', 'decel': '$decel'}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    'cut_in': {
        'name': 'Cut-in', 'about': 'A faster car in the next lane cuts in just ahead of the ego.',
        'needs': ['adjacent_same'], 'duration_s': 12.0,
        'params': {'speed': U(30, 60, 'km/h'), 'cutter_speed': U(35, 80, 'km/h'),
                   'start_offset': U(-15, 5, 'm', 'the cutter\'s start along the road from the ego'),
                   'cut_gap': U(2, 15, 'm', 'its rear ahead of the ego\'s front when it cuts'),
                   'cut_time': U(1.0, 3.0, 's', 'lane-change duration')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'cutter', 'kind': 'car', 'place': {'lane': 'adjacent', 'ahead_m': '$start_offset'},
                    'heading': 'lane', 'speed_kmh': '$cutter_speed', 'start': 'moving'}],
        'timeline': [{'trigger': {'type': 'ahead', 'actor': 'cutter', 'gt_m': '$cut_gap'},
                      'actions': [{'actor': 'cutter', 'do': 'lane_change', 'to': 'ego', 'duration_s': '$cut_time'}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    'pedestrian_crossing': {
        'name': 'Pedestrian crossing', 'about': 'A pedestrian walks across the road ahead.',
        'needs': [], 'duration_s': 14.0,
        'params': {'speed': U(20, 50, 'km/h'), 'distance': U(20, 60, 'm', 'where the pedestrian waits'),
                   'walk': U(0.8, 2.0, 'm/s'), 'trigger': U(15, 45, 'm', 'the ego\'s distance when it starts')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'ped', 'kind': 'pedestrian', 'place': {'lane': 'sidewalk', 'ahead_m': '$distance'},
                    'heading': 'across', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [{'trigger': {'type': 'distance', 'actor': 'ped', 'lt_m': '$trigger'},
                      'actions': [{'actor': 'ped', 'do': 'walk', 'speed_mps': '$walk'}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    'cyclist_crossing': {
        'name': 'Cyclist crossing', 'about': 'A cyclist rides across the road ahead.',
        'needs': [], 'duration_s': 14.0,
        'params': {'speed': U(20, 50, 'km/h'), 'distance': U(25, 60, 'm'), 'ride': U(2.5, 6.0, 'm/s'),
                   'trigger': U(20, 45, 'm')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'cyclist', 'kind': 'bicycle', 'place': {'lane': 'sidewalk', 'ahead_m': '$distance'},
                    'heading': 'across', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [{'trigger': {'type': 'distance', 'actor': 'cyclist', 'lt_m': '$trigger'},
                      'actions': [{'actor': 'cyclist', 'do': 'speed', 'kmh': '$ride * 3.6', 'accel': 2.0}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    'door_opening': {
        'name': 'Parked car opens a door', 'about': 'A car parked at the kerb opens its door into the lane.',
        'needs': [], 'duration_s': 12.0,
        'params': {'speed': U(20, 50, 'km/h'), 'distance': U(25, 60, 'm'), 'trigger': U(8, 30, 'm'),
                   'offset': U(0.0, 0.6, 'm', 'how far the parked car stands into the lane')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'parked', 'kind': 'car', 'place': {'lane': 'kerb', 'ahead_m': '$distance',
                                                              'lateral_m': '$offset'},
                    'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [{'trigger': {'type': 'distance', 'actor': 'parked', 'lt_m': '$trigger'},
                      'actions': [{'actor': 'parked', 'do': 'open_door', 'door': 'offside'}]}],
        'pass': {'no_collision': True}},
    'stalled_vehicle': {
        'name': 'Stalled vehicle', 'about': 'A vehicle stands still in the ego\'s lane.',
        'needs': [], 'duration_s': 14.0,
        'params': {'speed': U(30, 70, 'km/h'), 'distance': U(40, 120, 'm'),
                   'kind': _p('choice', '', 'what stalled', choices=['car', 'truck', 'van'])},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'stalled', 'kind': '$kind', 'place': {'lane': 'ego', 'ahead_m': '$distance'},
                    'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [],
        'pass': {'no_collision': True, 'min_ttc_s': 1.5}},
    'unprotected_turn': {
        'name': 'Unprotected turn across traffic', 'about': 'The ego turns across the oncoming lanes at a junction '
                                                             'while a car comes straight on.',
        'needs': ['junction_turn'], 'duration_s': 16.0,
        'params': {'speed': U(15, 30, 'km/h'), 'oncoming_speed': U(25, 60, 'km/h'),
                   'oncoming_gap': U(25, 80, 'm', 'the oncoming car\'s distance past the junction entry')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'turn_across'},
        'actors': [{'id': 'oncoming', 'kind': 'car', 'place': {'lane': 'opposite', 'ahead_m': '$oncoming_gap'},
                    'heading': 'against', 'speed_kmh': '$oncoming_speed', 'start': 'moving'}],
        'timeline': [],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    # ---- Indian roads
    'two_wheeler_filtering': {
        'name': 'Two-wheeler filtering between lanes', 'about': 'A motorcycle overtakes along the line between the '
                                                                'ego\'s lane and the next, close past the ego.',
        'needs': ['adjacent_same'], 'duration_s': 12.0,
        'params': {'speed': U(15, 40, 'km/h'), 'bike_speed': U(30, 60, 'km/h'),
                   'start_behind': U(10, 30, 'm'), 'line_offset': U(-0.4, 0.4, 'm', 'off the lane line')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'bike', 'kind': 'motorcycle', 'place': {'lane': 'line', 'ahead_m': '-$start_behind',
                                                                   'lateral_m': '$line_offset'},
                    'heading': 'lane', 'speed_kmh': '$bike_speed', 'start': 'moving'}],
        'timeline': [],
        'pass': {'no_collision': True, 'min_ttc_s': 0.5}},
    'jaywalker_behind_bus': {
        'name': 'Jaywalker from behind a bus', 'about': 'A pedestrian steps out from behind a bus stopped at the '
                                                        'kerb, hidden until the last moment.',
        'needs': [], 'duration_s': 14.0,
        'params': {'speed': U(20, 45, 'km/h'), 'distance': U(30, 70, 'm'), 'walk': U(1.0, 2.5, 'm/s'),
                   'trigger': U(10, 35, 'm')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'bus', 'kind': 'bus', 'place': {'lane': 'kerb', 'ahead_m': '$distance'},
                    'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'},
                   {'id': 'ped', 'kind': 'pedestrian', 'place': {'lane': 'kerb', 'ahead_m': '$distance - 7',
                                                                  'lateral_m': -1.5},
                    'heading': 'across', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [{'trigger': {'type': 'distance', 'actor': 'ped', 'lt_m': '$trigger'},
                      'actions': [{'actor': 'ped', 'do': 'walk', 'speed_mps': '$walk'}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 0.8}},
    'wrong_way_scooter': {
        'name': 'Wrong-way scooter', 'about': 'A scooter rides against the traffic in the ego\'s lane, near the kerb.',
        'needs': [], 'duration_s': 12.0,
        'params': {'speed': U(20, 45, 'km/h'), 'scooter_speed': U(15, 35, 'km/h'), 'distance': U(40, 100, 'm'),
                   'lateral': U(0.5, 1.4, 'm', 'from the lane centre towards the kerb')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'scooter', 'kind': 'motorcycle', 'place': {'lane': 'ego', 'ahead_m': '$distance',
                                                                      'lateral_m': '$lateral'},
                    'heading': 'against', 'speed_kmh': '$scooter_speed', 'start': 'moving'}],
        'timeline': [],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    'bus_pulling_out': {
        'name': 'Bus pulling out', 'about': 'A bus stopped at the kerb pulls out into the ego\'s lane.',
        'needs': [], 'duration_s': 14.0,
        'params': {'speed': U(20, 50, 'km/h'), 'distance': U(25, 70, 'm'), 'trigger': U(15, 45, 'm'),
                   'pull_time': U(2.0, 5.0, 's'), 'bus_speed': U(10, 25, 'km/h')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'bus', 'kind': 'bus', 'place': {'lane': 'kerb', 'ahead_m': '$distance'},
                    'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [{'trigger': {'type': 'distance', 'actor': 'bus', 'lt_m': '$trigger'},
                      'actions': [{'actor': 'bus', 'do': 'speed', 'kmh': '$bus_speed', 'accel': 1.0},
                                  {'actor': 'bus', 'do': 'lane_change', 'to': 'ego', 'duration_s': '$pull_time'}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    'stray_cattle': {
        'name': 'Stray cattle in the lane', 'about': 'A cow stands in the ego\'s lane ahead (a proxy obstacle of a '
                                                     'non-descript cow\'s size: docs/indian-assets.md).',
        'needs': [], 'duration_s': 14.0,
        'params': {'speed': U(20, 45, 'km/h'), 'distance': U(30, 90, 'm'),
                   'lateral': U(-0.8, 0.8, 'm', 'off the lane centre (positive: towards the kerb)')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'cow', 'kind': 'cattle', 'place': {'lane': 'ego', 'ahead_m': '$distance',
                                                              'lateral_m': '$lateral'},
                    'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [],
        'pass': {'no_collision': True, 'min_ttc_s': 1.5}},
    'handcart_kerb_lane': {
        'name': 'Handcart in the kerb lane', 'about': 'A handcart pushed at walking pace near the kerb in the ego\'s '
                                                      'lane (the left lane in left-hand traffic), its pusher walking '
                                                      'behind it (a proxy: docs/indian-assets.md).',
        'needs': [], 'duration_s': 16.0,
        'params': {'speed': U(20, 40, 'km/h'), 'distance': U(30, 80, 'm'),
                   'cart_speed': U(3, 6, 'km/h', 'walking pace (heuristic)'),
                   'lateral': U(0.3, 0.9, 'm', 'from the lane centre towards the kerb')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'cart', 'kind': 'handcart', 'place': {'lane': 'ego', 'ahead_m': '$distance',
                                                                 'lateral_m': '$lateral'},
                    'heading': 'lane', 'speed_kmh': '$cart_speed', 'start': 'moving'},
                   {'id': 'pusher', 'kind': 'pedestrian', 'place': {'lane': 'ego', 'ahead_m': '$distance - 1.3',
                                                                     'lateral_m': '$lateral'},
                    'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}],
        'timeline': [{'trigger': {'type': 'time', 't_s': 0.0},
                      'actions': [{'actor': 'pusher', 'do': 'walk', 'speed_mps': '$cart_speed / 3.6'}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    'autorickshaw_cut_in': {
        'name': 'Autorickshaw cuts in', 'about': 'An autorickshaw (a stand-in with a three-wheeler\'s top speed and '
                                                 'acceleration) in the next lane cuts in just ahead of the ego.',
        'needs': ['adjacent_same'], 'duration_s': 12.0,
        'params': {'speed': U(20, 40, 'km/h'), 'faster': U(5, 20, 'km/h', 'the autorickshaw\'s speed over the '
                                                                            'ego\'s (capped at its top speed)'),
                   'start_offset': U(0, 8, 'm', 'its start along the road ahead of the ego'),
                   'cut_gap': U(1, 8, 'm'), 'cut_time': U(0.8, 2.5, 's', 'lane-change duration')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'auto', 'kind': 'autorickshaw', 'place': {'lane': 'adjacent', 'ahead_m': '$start_offset'},
                    'heading': 'lane', 'speed_kmh': '$speed + $faster', 'start': 'moving'}],
        'timeline': [{'trigger': {'type': 'ahead', 'actor': 'auto', 'gt_m': '$cut_gap'},
                      'actions': [{'actor': 'auto', 'do': 'lane_change', 'to': 'ego', 'duration_s': '$cut_time'}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
    'monsoon_waterlogged': {
        'name': 'Monsoon: waterlogged road', 'about': 'The lead vehicle brakes on a waterlogged road in heavy rain: '
                                                      'CARLA\'s puddles (precipitation_deposits 100) and wetness. '
                                                      'Measured: puddles are drawn on CARLA\'s towns, not on an '
                                                      'imported (OpenDRIVE) world; they are global, not placed; '
                                                      'they do not change tyre grip (not modelled by CARLA here).',
        'needs': [], 'duration_s': 12.0,
        'weather': {'preset': 'HardRainNoon', 'precipitation_deposits': 100.0, 'wetness': 100.0},
        'params': {'speed': U(20, 45, 'km/h'), 'gap': U(10, 35, 'm'), 'decel': U(3, 7, 'm/s²'),
                   'brake_at': U(2, 5, 's')},
        'ego': {'speed_kmh': '$speed', 'driver': 'follower', 'route': 'straight'},
        'actors': [{'id': 'lead', 'kind': 'car', 'place': {'lane': 'ego', 'ahead_m': '$gap'}, 'heading': 'lane',
                    'speed_kmh': '$speed', 'start': 'moving'}],
        'timeline': [{'trigger': {'type': 'time', 't_s': '$brake_at'},
                      'actions': [{'actor': 'lead', 'do': 'brake', 'decel': '$decel'}]}],
        'pass': {'no_collision': True, 'min_ttc_s': 1.0}},
}
INDIAN = ('two_wheeler_filtering', 'jaywalker_behind_bus', 'wrong_way_scooter', 'bus_pulling_out', 'stray_cattle',
          'handcart_kerb_lane', 'autorickshaw_cut_in', 'monsoon_waterlogged')
KINDS = ('car', 'truck', 'bus', 'van', 'motorcycle', 'bicycle', 'pedestrian', 'child', 'autorickshaw',
         'cycle_rickshaw', 'cattle', 'handcart', 'vendor')


def builtin(key: str) -> dict:
    d = copy.deepcopy(LIBRARY[key])
    d.update(id=key, schema=SCENARIO_SCHEMA, builtin=True)
    return d


def sample_params(scn: dict, seed: int) -> dict:
    """One value per parameter, drawn with a generator seeded by (seed, the parameter's name): adding a parameter
    does not change the others' draws."""
    out = {}
    for name in sorted(scn.get('params') or {}):
        p = scn['params'][name]
        rng = random.Random(f'{seed}:{name}')
        d = p.get('dist', 'fixed')
        if d == 'fixed':
            out[name] = p['value']
        elif d == 'uniform':
            out[name] = rng.uniform(float(p['lo']), float(p['hi']))
        elif d == 'normal':
            v = rng.gauss(float(p['mean']), float(p['sd']))
            lo, hi = p.get('lo'), p.get('hi')
            out[name] = min(max(v, lo if lo is not None else v), hi if hi is not None else v)
        elif d == 'choice':
            out[name] = rng.choice(list(p['choices']))
        else:
            raise ValueError(f'parameter {name}: unknown distribution {d!r}')
    return out


def evaluate(expr, values: dict):
    """A number, a '$name', or a small arithmetic expression of parameters ('$ride * 3.6', '-$start_behind',
    '$distance - 7'). Only parameters, numbers and + − × / are allowed."""
    if isinstance(expr, (int, float)):
        return expr
    if not isinstance(expr, str):
        return expr
    s = expr.strip()
    if '$' not in s:
        return expr                                   # a plain string value (a kind, a lane name)
    if s.startswith('$') and s[1:].isidentifier():
        return values[s[1:]]
    import ast
    import operator as op
    import re
    src = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', lambda m: f'__p_{m.group(1)}', s)
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg,
           ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name) and n.id.startswith('__p_'):
            return values[n.id[4:]]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError(f'not allowed in a scenario expression: {expr!r}')
    try:
        return ev(ast.parse(src, mode='eval'))
    except SyntaxError:
        return expr                                   # a plain string value (a kind, a lane name)


def _subst(obj, values):
    if isinstance(obj, dict):
        return {k: _subst(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, values) for v in obj]
    if isinstance(obj, str) and '$' in obj:
        return evaluate(obj, values)
    return obj


def concrete(scn: dict, values: dict) -> dict:
    """The scenario with every expression evaluated for these parameter values (and the values kept)."""
    body = {k: v for k, v in scn.items() if k != 'params'}
    out = _subst(body, values)
    out['values'] = dict(values)
    out['params'] = copy.deepcopy(scn.get('params') or {})
    return out


def sample(scn: dict, seed: int, overrides: dict | None = None) -> dict:
    values = sample_params(scn, seed)
    values.update(overrides or {})
    out = concrete(scn, values)
    out['seed'] = int(seed)
    return out


def search_space(scn: dict) -> list:
    """The scenario's parameters as a search space (core.search): continuous ones by their range, choices as
    choices; fixed ones left out."""
    out = []
    for name, p in sorted((scn.get('params') or {}).items()):
        if p.get('dist') == 'uniform':
            out.append({'name': name, 'lo': float(p['lo']), 'hi': float(p['hi'])})
        elif p.get('dist') == 'normal':
            out.append({'name': name, 'lo': float(p.get('lo', p['mean'] - 3 * p['sd'])),
                        'hi': float(p.get('hi', p['mean'] + 3 * p['sd']))})
        elif p.get('dist') == 'choice':
            out.append({'name': name, 'choices': list(p['choices'])})
    return out


def validate(scn: dict) -> list:
    out = []
    ids = [a['id'] for a in scn.get('actors', [])]
    for d in sorted({i for i in ids if ids.count(i) > 1}):
        out.append(f'actor id {d!r} used twice')
    for name, p in (scn.get('params') or {}).items():
        if p.get('dist') not in DISTS:
            out.append(f'parameter {name}: unknown distribution {p.get("dist")!r}')
        if p.get('dist') == 'uniform' and float(p['lo']) > float(p['hi']):
            out.append(f'parameter {name}: lo > hi')
    for k, ev_ in enumerate(scn.get('timeline', [])):
        for a in ev_.get('actions', []):
            if a.get('actor') not in ids:
                out.append(f'timeline {k}: unknown actor {a.get("actor")!r}')
        t = ev_.get('trigger', {})
        if t.get('type') in ('distance', 'ahead') and t.get('actor') not in ids:
            out.append(f'timeline {k}: trigger on unknown actor {t.get("actor")!r}')
    try:
        sample(scn, 0)
    except (KeyError, ValueError) as e:
        out.append(f'cannot be sampled: {e}')
    return out
