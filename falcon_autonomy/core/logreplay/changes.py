"""What-if changes to a log before it is replayed (numpy). A change list (schema falcon.replay_changes/1) is a JSON
list of objects, applied in order to a copy of the log; every draw comes from a generator seeded by the replay's seed
and the change's position in the list, so the same log, changes and seed give the same replay.

    {"op": "time_shift",  "actor": ID, "dt_s": 1.5, "outside": "absent" | "hold"}
        The actor's whole timeline later by dt_s (earlier if negative). Outside its samples it is absent, as in the
        log (default), or held at its first / last pose ("hold": it waits where it was first seen).
    {"op": "speed_scale", "actor": ID, "factor": 0.7, "from_s": 2.0}
        Re-timed along its own path: from from_s (default its first sample) it covers the path at factor × its
        speed — sample times after from_s become from_s + (t − from_s) / factor. factor 0 stops it at from_s, held
        there. The path itself is unchanged.
    {"op": "remove", "actor": ID}
    {"op": "add_pedestrian", "id": "ped_1", "at_ego_s": 8.0 | "at": {"x": .., "y": ..}, "lateral_m": 0.0,
     "from": "right" | "left", "start_m": 6.0, "walk_m": 12.0, "speed_mps": 1.4 | [lo, hi], "t_s": 8.0,
     "size": [0.6, 0.6, 1.75]}
        A pedestrian crossing the ego's logged path at right angles: at the point the ego was at log time at_ego_s
        (or the point "at", in the log's frame), shifted lateral_m to the path's right. It starts start_m to the
        given side of the crossing point, walks walk_m at speed_mps (a range is drawn, seeded) and reaches the
        crossing point at t_s (default at_ego_s: when the ego was there — a conflict). Before it starts walking it
        stands at its start; after, at its end. Kinematic: a straight line at constant speed.
    {"op": "weather", "preset": "WetCloudyNoon", "params": {"fog_density": 40}}
        The replay's weather: a preset of core.weather and/or CARLA weather fields.

"actor" may be "ego" for time_shift and speed_scale (they then change the logged ego the replay follows).
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from .model import LogError, Track, wrap_deg

CHANGES_SCHEMA = 'falcon.replay_changes/1'
OPS = ('time_shift', 'speed_scale', 'remove', 'add_pedestrian', 'weather')


def rng_for(seed: int, k: int, op: str):
    return np.random.default_rng([int(seed) & 0xffffffff, int(k), zlib.crc32(op.encode())])


def validate(changes) -> list:
    """The change list checked (a list of dicts with known ops and their fields); raises LogError."""
    if isinstance(changes, dict):
        changes = changes.get('changes', [])
    if not isinstance(changes, list):
        raise LogError('changes are a JSON list of objects')
    out = []
    for k, c in enumerate(changes):
        if not isinstance(c, dict) or c.get('op') not in OPS:
            raise LogError(f'change {k}: "op" must be one of {", ".join(OPS)}')
        op = c['op']
        need = {'time_shift': ('actor', 'dt_s'), 'speed_scale': ('actor', 'factor'), 'remove': ('actor',),
                'add_pedestrian': (), 'weather': ()}[op]
        miss = [f for f in need if f not in c]
        if miss:
            raise LogError(f'change {k} ({op}): missing {", ".join(miss)}')
        if op == 'speed_scale' and float(c['factor']) < 0:
            raise LogError(f'change {k}: a speed factor is ≥ 0')
        if op == 'add_pedestrian' and 'at_ego_s' not in c and 'at' not in c:
            raise LogError(f'change {k}: add_pedestrian needs "at_ego_s" or "at"')
        if op == 'weather' and not (c.get('preset') or c.get('params')):
            raise LogError(f'change {k}: weather needs "preset" or "params"')
        if op == 'time_shift' and c.get('outside', 'absent') not in ('absent', 'hold'):
            raise LogError(f'change {k}: "outside" is "absent" or "hold"')
        out.append(dict(c))
    return out


def _draw(v, rng):
    if isinstance(v, (list, tuple)):
        lo, hi = float(v[0]), float(v[1])
        return float(rng.uniform(lo, hi)) if hi > lo else lo
    return float(v)


def apply(log, changes, seed: int = 0):
    """(the changed log copy, weather or None, [what each change did]) — the log is not modified."""
    changes = validate(changes)
    out = log.copy()
    weather = None
    done = []
    for k, c in enumerate(changes):
        op = c['op']
        rng = rng_for(seed, k, op)
        if op == 'weather':
            weather = {'preset': c.get('preset'), 'params': dict(c.get('params') or {})}
            done.append({'op': op, 'preset': c.get('preset'), 'params': c.get('params') or {}})
            continue
        if op == 'add_pedestrian':
            tr, info = pedestrian(out, c, rng, k)
            if any(a.id == tr.id for a in out.actors):
                raise LogError(f'change {k}: an actor {tr.id!r} exists already')
            out.actors.append(tr)
            done.append(dict(info, op=op))
            continue
        tr = out.actor(c['actor'])
        if op == 'remove':
            if tr is out.ego:
                raise LogError('the ego cannot be removed')
            out.actors = [a for a in out.actors if a is not tr]
            done.append({'op': op, 'actor': tr.id})
        elif op == 'time_shift':
            tr.t = tr.t + float(c['dt_s'])
            if c.get('outside') == 'hold':
                tr.hold = (True, True)
            done.append({'op': op, 'actor': tr.id, 'dt_s': float(c['dt_s']), 'outside': c.get('outside', 'absent')})
        elif op == 'speed_scale':
            f = float(c['factor'])
            t_from = float(c.get('from_s', tr.t[0]))
            if f == 0.0:
                ok, x, y, z, yaw = tr.sample([t_from])
                if not ok[0]:
                    raise LogError(f'change {k}: actor {tr.id} is not observed at {t_from} s')
                keep = tr.t < t_from - 1e-9
                for name, v in (('x', x), ('y', y), ('z', z), ('yaw', yaw)):
                    setattr(tr, name, np.r_[getattr(tr, name)[keep], v])
                for name in ('pitch', 'roll'):
                    a = getattr(tr, name)
                    if a is not None:
                        setattr(tr, name, np.r_[a[keep], np.interp(t_from, tr.t, a)])
                tr.t = np.r_[tr.t[keep], t_from]
                tr.hold = (tr.hold[0], True)
            else:
                after = tr.t > t_from
                tr.t = np.where(after, t_from + (tr.t - t_from) / f, tr.t)
                tr.max_gap_s = tr.max_gap_s / f if f < 1 else tr.max_gap_s
            done.append({'op': op, 'actor': tr.id, 'factor': f, 'from_s': t_from})
    return out, weather, done


def pedestrian(log, c: dict, rng, k: int):
    """The added pedestrian's track (see the module docstring) and what was drawn."""
    ego = log.ego
    if 'at_ego_s' in c:
        T = float(c['at_ego_s'])
        ok, x, y, _, yaw = ego.sample([T])
        if not ok[0]:
            raise LogError(f'change {k}: the ego is not in the log at {T} s (it spans {ego.span[0]:.2f}–'
                           f'{ego.span[1]:.2f} s)')
        px, py, heading = float(x[0]), float(y[0]), float(yaw[0])
        t_cross = float(c.get('t_s', T))
    else:
        px, py = float(c['at']['x']), float(c['at']['y'])
        d = np.hypot(ego.x - px, ego.y - py)
        i = int(np.argmin(d))
        heading = float(ego.yaw[i])
        t_cross = float(c.get('t_s', ego.t[i]))
    a = math.radians(heading)
    right = np.array([-math.sin(a), math.cos(a)])           # CARLA: the right of heading (cos, sin) is (−sin, cos)
    cx, cy = np.array([px, py]) + right * float(c.get('lateral_m', 0.0))
    side = 1.0 if c.get('from', 'right') == 'right' else -1.0
    start_m = float(c.get('start_m', 6.0))
    walk_m = float(c.get('walk_m', 2.0 * start_m))
    speed = _draw(c.get('speed_mps', 1.4), rng)
    if speed <= 0 or walk_m <= 0:
        raise LogError(f'change {k}: a pedestrian walks at a positive speed over a positive distance')
    p0 = np.array([cx, cy]) + side * right * start_m
    direction = -side * right
    walk_yaw = math.degrees(math.atan2(direction[1], direction[0]))
    t_start = t_cross - start_m / speed
    t_end = t_start + walk_m / speed
    n = max(int(math.ceil((t_end - t_start) / 0.05)), 1)
    ts = np.linspace(t_start, t_end, n + 1)
    s = (ts - t_start) * speed
    xs, ys = p0[0] + direction[0] * s, p0[1] + direction[1] * s
    zi = int(np.argmin(np.abs(ego.t - t_cross)))
    size = tuple(c.get('size') or (0.6, 0.6, 1.75))
    z = float(ego.z[zi]) + size[2] / 2.0
    ident = str(c.get('id') or f'added_pedestrian_{k}')
    tr = Track(ident, 'pedestrian', ts, xs, ys, np.full(len(ts), z), np.full(len(ts), wrap_deg(walk_yaw)), size,
               'added pedestrian', 'change' if c.get('size') else 'typical', max_gap_s=1.0, hold=(True, True))
    return tr, {'id': ident, 'crossing_point': [round(float(cx), 4), round(float(cy), 4)],
                'start': [round(float(p0[0]), 4), round(float(p0[1]), 4)], 'speed_mps': round(speed, 6),
                't_start_s': round(t_start, 6), 't_cross_s': round(t_cross, 6), 't_end_s': round(t_end, 6),
                'from': c.get('from', 'right'), 'walk_m': walk_m}
