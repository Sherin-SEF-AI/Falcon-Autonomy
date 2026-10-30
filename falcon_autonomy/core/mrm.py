"""Minimal-risk-manoeuvre (MRM) verification of a stack run with a sensor fault (numpy): after the fault's onset,
did the stack bring the vehicle to a minimal risk condition — a controlled stop in its lane (or on the shoulder),
held, touching nothing?

    verify(frames, onset_s, params, collisions) -> {'passed', 'checks': [...], 'measured': {...}}

frames: the run's frames.jsonl rows (sim_t, speed_mps, and from the stack worker lane_offset_m, lane_width_m,
ego_width_m); onset_s: the fault's onset (simulation seconds); collisions: the times of the ego's contacts
(core.criticality). Every check, its threshold heuristic (PARAMS: a controlled stop's values, chosen for Falcon's test
stack, not a regulation's — ISO 22737 / UN R157 set their own for the systems they cover):

    reaction     the vehicle starts slowing (deceleration ≥ react_decel_mps2 held for react_hold_s) within
                 react_within_s of the onset
    standstill   speed ≤ stopped_mps within react_within_s + v_onset / min_mean_decel_mps2 of the onset, and held to
                 the end of the run (no restart)
    deceleration the deceleration never exceeds max_decel_mps2 (a controlled stop, not emergency braking)
    jerk         |jerk| ≤ max_jerk_mps3, from the speed smoothed over jerk_window_s (the differences of a 20 Hz speed
                 amplify its quantisation), except for jerk_budget_s in all
    lane         |lateral offset from the lane centre| ≤ (lane width − vehicle width) / 2 + lane_tol_m throughout
                 (in its lane), or, at the stop, on the kerb side beyond the lane's edge by at most shoulder_m (on the
                 shoulder); not judged when the run has no lane offsets (no scenario route)
    no collision no contact of the ego after the onset

A run whose fault never acts (onset after the end) or that ends before the standstill window closes is judged on what
it recorded; the window is in measured.
"""
from __future__ import annotations


import numpy as np

SCHEMA = 'falcon.mrm/1'
PARAMS = {'react_within_s': 1.5, 'react_decel_mps2': 0.5, 'react_hold_s': 0.3, 'min_mean_decel_mps2': 1.5,
          'stopped_mps': 0.2, 'max_decel_mps2': 4.0, 'max_jerk_mps3': 10.0, 'jerk_window_s': 0.25,
          'jerk_budget_s': 0.2, 'lane_tol_m': 0.2, 'shoulder_m': 2.5}


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    pad = np.r_[np.full(k // 2, x[0]), x, np.full(k - 1 - k // 2, x[-1])]
    return np.convolve(pad, np.ones(k) / k, mode='valid')


def verify(frames: list, onset_s: float, params: dict | None = None, collisions=()) -> dict:
    p = dict(PARAMS, **(params or {}))
    t = np.array([float(f['sim_t']) for f in frames])
    v = np.array([float(f.get('speed_mps') or 0.0) for f in frames])
    out = {'schema': SCHEMA, 'onset_s': round(float(onset_s), 3), 'params': p, 'checks': []}
    if len(t) < 3:
        out.update(passed=False, checks=[{'name': 'recorded', 'ok': False, 'detail': 'fewer than 3 steps'}])
        return out
    dt = float(np.median(np.diff(t)))
    after = t >= onset_s - 1e-9
    if not after.any():
        out.update(passed=False, checks=[{'name': 'fault acted', 'ok': False,
                                          'detail': f'the run ended at {t[-1]:.2f} s, before the onset'}])
        return out
    k0 = int(np.argmax(after))
    v0 = float(v[k0])
    k = max(int(round(p['jerk_window_s'] / dt)), 1)
    vs = _smooth(v, k)
    a = np.gradient(vs, dt)
    j = np.gradient(_smooth(a, k), dt)
    checks = []
    # reaction: deceleration ≥ react_decel held for react_hold_s, starting within react_within_s
    hold = max(int(round(p['react_hold_s'] / dt)), 1)
    decel = (-a >= p['react_decel_mps2']) | (v <= p['stopped_mps'])
    t_react = None
    for i in range(k0, len(t) - hold + 1):
        if decel[i:i + hold].all():
            t_react = float(t[i] - onset_s)
            break
    checks.append({'name': f'reacts within {p["react_within_s"]:g} s', 'ok': t_react is not None and
                   t_react <= p['react_within_s'] + 1e-9,
                   'detail': f'slowing from {t_react:.2f} s after the onset' if t_react is not None
                   else 'never slowed'})
    window = p['react_within_s'] + v0 / p['min_mean_decel_mps2']
    stopped = v <= p['stopped_mps']
    t_stop = None
    for i in range(k0, len(t)):
        if stopped[i:].all():
            t_stop = float(t[i] - onset_s)
            break
    first_stop = next((float(t[i] - onset_s) for i in range(k0, len(t)) if stopped[i]), None)
    restarted = first_stop is not None and t_stop is None
    ended_early = t[-1] - onset_s < window
    checks.append({'name': f'standstill within {window:.1f} s, held', 'ok': t_stop is not None and
                   t_stop <= window + 1e-9,
                   'detail': (f'stopped {t_stop:.2f} s after the onset (from {v0:.1f} m/s)' if t_stop is not None
                              else 'stopped, then moved again' if restarted
                              else f'still at {v[-1]:.1f} m/s when the run ended' +
                              (f' ({t[-1] - onset_s:.1f} s after the onset: the run is shorter than the window)'
                               if ended_early else ''))})
    a_after = a[k0:]
    max_dec = float(max(-a_after.min(), 0.0))
    checks.append({'name': f'deceleration ≤ {p["max_decel_mps2"]:g} m/s²', 'ok': max_dec <= p['max_decel_mps2'] + 1e-9,
                   'detail': f'peak {max_dec:.2f} m/s²'})
    j_after = np.abs(j[k0:])
    over = float(np.count_nonzero(j_after > p['max_jerk_mps3'])) * dt
    checks.append({'name': f'|jerk| ≤ {p["max_jerk_mps3"]:g} m/s³', 'ok': over <= p['jerk_budget_s'] + 1e-9,
                   'detail': f'peak {float(j_after.max()):.1f} m/s³, {over:.2f} s above (budget '
                             f'{p["jerk_budget_s"]:g} s)'})
    offs = [f.get('lane_offset_m') for f in frames[k0:]]
    if all(o is not None for o in offs) and offs:
        off = np.array(offs, float)
        lw = float(frames[k0].get('lane_width_m') or 3.5)
        ew = float(frames[k0].get('ego_width_m') or 2.0)
        limit = (lw - ew) / 2.0 + p['lane_tol_m']
        side = float(frames[k0].get('kerb_sign') or 1.0)       # + : the kerb on the lane's right
        in_lane = bool(np.all(np.abs(off) <= limit))
        shoulder = (not in_lane) and side * off[-1] > limit and side * off[-1] <= limit + p['shoulder_m'] and \
            bool(np.all(side * off >= -limit))
        checks.append({'name': 'stays in its lane or moves to the shoulder', 'ok': in_lane or shoulder,
                       'detail': f'lateral offset {float(np.abs(off).max()):.2f} m at most (in-lane limit '
                                 f'{limit:.2f} m)' + (', ended on the shoulder' if shoulder else '')})
    else:
        checks.append({'name': 'stays in its lane or moves to the shoulder', 'ok': True,
                       'detail': 'not judged: the run has no lane offsets'})
    hits = sorted(float(c) for c in collisions or () if c is not None and float(c) >= onset_s - 1e-9)
    checks.append({'name': 'no collision after the fault', 'ok': not hits,
                   'detail': f'contact at {hits[0]:.2f} s' if hits else 'none'})
    out['checks'] = checks
    out['passed'] = all(c['ok'] for c in checks)
    out['measured'] = {'v_onset_mps': round(v0, 3), 'reaction_s': t_react, 'standstill_s': t_stop,
                       'window_s': round(window, 3), 'max_decel_mps2': round(max_dec, 3),
                       'max_jerk_mps3': round(float(j_after.max()), 3)}
    return out


def first_onset(faults: list) -> float | None:
    """The earliest onset of a stack run's timed faults (the moment the MRM is judged from)."""
    ts = [float(f['onset_s']) for f in faults or [] if f.get('onset_s') is not None]
    return min(ts) if ts else None


def validate_faults(faults: list, rig: dict) -> list:
    """Problems with a stack run's timed faults [{sensor, kind, onset_s, for_s?, …}] against its rig."""
    from . import faults as F
    out = []
    sensors = {s['id']: s['type'] for s in (rig or {}).get('sensors') or []}
    for k, f in enumerate(faults or []):
        if f.get('sensor') not in sensors:
            out.append(f'fault {k}: no sensor {f.get("sensor")!r} in the rig')
            continue
        if f.get('kind') not in F.TIMED:
            out.append(f'fault {k}: {f.get("kind")!r} cannot start mid-run ({", ".join(F.TIMED)})')
            continue
        if not isinstance(f.get('onset_s'), (int, float)) or f['onset_s'] < 0:
            out.append(f'fault {k}: onset_s (seconds) is needed')
            continue
        if not F.applies_to(f['kind'], sensors[f['sensor']]):
            out.append(f'fault {k}: a {f["kind"]} fault cannot act on {f["sensor"]} (a {sensors[f["sensor"]]})')
        body = {kk: vv for kk, vv in f.items() if kk not in ('sensor', 'for_s')}
        if f.get('for_s'):
            body['until_s'] = float(f['onset_s']) + float(f['for_s'])
        out += [f'fault {k}: {p}' for p in F.validate([body])]
    return out


def as_fault_lists(faults: list) -> dict:
    """{sensor: [core.faults entries with onset_s / until_s]} of a stack run's timed faults."""
    out = {}
    for f in faults or []:
        body = {k: v for k, v in f.items() if k not in ('sensor', 'for_s')}
        if f.get('for_s'):
            body['until_s'] = float(f['onset_s']) + float(f['for_s'])
        out.setdefault(f['sensor'], []).append(body)
    return out
