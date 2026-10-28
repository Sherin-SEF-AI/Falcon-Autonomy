"""Runtime ODD monitoring: flag when a run's conditions or its sensor data leave the operational design domain the
safety case covers (numpy at import; OpenCV and torch only inside the data monitor's functions).

The ODD. The safety case's ODD (core.safetycase, schema falcon.odd/1 with "dimensions") is what is monitored. The
Loop's ODD (core.loop, the same schema string but worlds × weather entries) is mapped onto it by from_loop_odd():
place = its worlds (and their CARLA maps); each weather field = the hull, over its weather entries, of the entry's
value (its preset's, or fixed) and its range — an over-approximation (a combination of fields that no single entry
allows is still called inside) that is stated in the mapped ODD's "about".

Condition monitors (ConditionMonitor), exact from the simulator's state at every step:
* weather and light — every range dimension named like a CARLA weather field (precipitation, fog_density, wetness,
  cloudiness, precipitation_deposits, wind_intensity, …) and sun_altitude_deg (CARLA's sun_altitude_angle), from
  World.get_weather() at runtime (a live weather event is seen) or the run's config offline;
* speed — ego_speed_kmh, the ego's ACTUAL speed (the safety case's coverage uses the scenario's set speed instead);
* road type — at runtime from the map at the ego: junction (Waypoint.is_junction), multi_lane (a driving lane beside
  it in the same direction), two_way (a lane beside it in the other direction), straight or curve (the lane's heading
  changes by less, or more, than 10° over the next 30 m); offline from frames' "odd" records when the worker wrote
  them, else the safety case's heuristic from the scenario (safetycase.run_coordinates) and said so;
* place, handedness, scenario, actors, traffic_vehicles — from the run's config.
A dimension the state cannot give is listed as "not monitored", never assumed inside.

Data monitors (out of distribution), per frame of a camera:
* appearance — the 11 image statistics of core.realgap (CIELAB, saturation, clipping, noise, sharpness, spectrum
  slope) standardised by the reference's mean and spread; the Mahalanobis distance under a Ledoit–Wolf shrunk
  covariance (Ledoit & Wolf 2004);
* features (optional, only when the pinned ResNet-50 extractor of core.realgap is downloaded) — the mean cosine
  distance to the k = 5 nearest reference frames.
A frame is out of distribution when a score exceeds its threshold.

Calibration (calibrate(); `falcon odd calibrate`). Frames of one drive are not independent, so the unit is the
DRIVE: leave-one-drive-out cross-fitting. Each in-ODD drive is scored by a reference fitted on the other drives; the
threshold is the conformal quantile of these out-of-fold scores for the target false-alarm rate a (the
⌈(n + 1)(1 − a)⌉-th smallest of n), so drive-to-drive variation is part of it; the final reference is fitted on all
drives. The false-alarm rate is MEASURED nested: the whole procedure is repeated without each drive in turn and that
drive's frames are counted above the threshold so obtained — the drive takes no part in its own threshold or
reference. The detection rate is measured on each out-of-ODD set given. Both carry Wilson 95 % intervals over frames
(optimistic: frames within a drive are correlated; the number of drives is reported beside them). At least three
in-ODD drives are needed; with fewer, each drive is cut into three contiguous blocks used as the units, and the
report says the rate is then optimistic.

Outputs. A run's monitor (monitor_run, `falcon odd monitor`): odd-monitor.json in the run's folder — the timeline as
segments of constant state (t0, t1, inside, the dimensions outside and why), time and share outside per reason, the
data monitor's flags — and in the workers, run.json "odd_monitor" and per frame "odd" when the run's config carries
an ODD (workers.oddwatch). The safety-case report sums the time outside per scenario (suite_outside), the Loop flags
its evaluation cells outside the project's ODD (loop_cells).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

TIMELINE_SCHEMA = 'falcon.odd_monitor/1'
REFERENCE_SCHEMA = 'falcon.odd_reference/1'
MONITOR_FILE = 'odd-monitor.json'
STRAIGHT_DEG = 10.0
STRAIGHT_AHEAD_M = 30.0
KNN_K = 5
WEATHER_DIMS = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity', 'sun_azimuth_angle',
                'fog_density', 'fog_distance', 'fog_falloff', 'wetness', 'scattering_intensity',
                'mie_scattering_scale', 'rayleigh_scattering_scale', 'dust_storm')
ALIASES = {'sun_altitude_deg': 'sun_altitude_angle'}       # ODD dimension -> CARLA weather field


class OddError(Exception):
    pass


# ================================================================================================ the ODD
def is_loop_odd(odd: dict) -> bool:
    return isinstance(odd, dict) and 'weathers' in odd and 'dimensions' not in odd


def from_loop_odd(loop_odd: dict, project=None) -> dict:
    """The Loop's ODD as a safety-case ODD (dimensions): place, and each weather field's hull over the entries."""
    from . import weather as W
    lo, hi = {}, {}
    for we in loop_odd.get('weathers') or []:
        we = {'name': we, 'preset': we} if isinstance(we, str) else we
        base = W.resolve({k: v for k, v in we.items() if k in W.FIELDS or k == 'preset'})
        for k, v in base.items():
            r = (we.get('ranges') or {}).get(k)
            a, b = (float(r[0]), float(r[1])) if r else (float(v), float(v))
            lo[k], hi[k] = min(lo.get(k, a), a), max(hi.get(k, b), b)
    dims = {}
    places = list(loop_odd.get('worlds') or [])
    if project is not None:
        for wid in loop_odd.get('worlds') or []:
            w = project.item('worlds', wid) or {}
            if w.get('town'):
                places.append(w['town'])
    if places:
        dims['place'] = {'kind': 'categorical', 'values': sorted(set(places))}
    for k in sorted(lo):
        if k in ('sun_azimuth_angle', 'scattering_intensity', 'mie_scattering_scale', 'rayleigh_scattering_scale',
                 'fog_distance', 'fog_falloff'):
            continue                                       # not conditions an ODD names
        name = 'sun_altitude_deg' if k == 'sun_altitude_angle' else k
        dims[name] = {'kind': 'range', 'lo': lo[k], 'hi': hi[k], 'bins': [lo[k], hi[k]],
                      'unit': '°' if name == 'sun_altitude_deg' else '%'}
    return {'schema': 'falcon.odd/1', 'name': f'{loop_odd.get("name") or loop_odd.get("id") or "Loop ODD"} (mapped)',
            'about': 'mapped from the Loop\'s ODD (core.oddmonitor.from_loop_odd): each weather field\'s range is '
                     'the hull over its weather entries — combinations no single entry allows are still inside',
            'dimensions': dims, 'pairs': []}


PROJECT_ODD = 'odd.json'                    # a workspace's monitored ODD (the safety case's format, or the Loop's)
PROJECT_REFERENCE = 'odd-reference.json'    # its calibrated data monitor (falcon odd calibrate --out)
CFG_KEYS = ('odd', 'odd_reference', 'odd_every', 'odd_workspace')


def attach(cfg: dict, workspace: str, project=None) -> dict:
    """Add the workspace's ODD (WORKSPACE/odd.json, else the project's Loop ODD) and its reference to a run's config,
    so the worker monitors it; nothing when there is none. Monitoring only observes: these keys are left out of a
    drive's cache key (core.datasets.drive_key)."""
    odd = None
    p = os.path.join(workspace, PROJECT_ODD) if workspace else ''
    if p and os.path.isfile(p):
        with open(p, encoding='utf-8') as fh:
            odd = json.load(fh)
    elif project is not None and ((project.doc.get('loop') or {}).get('odd')):
        odd = project.doc['loop']['odd']
    if not odd:
        return cfg
    cfg['odd'] = odd
    cfg['odd_workspace'] = workspace
    ref = os.path.join(workspace, PROJECT_REFERENCE) if workspace else ''
    if ref and os.path.isfile(ref):
        cfg['odd_reference'] = ref
    return cfg


def monitor_odd(odd: dict, project=None) -> dict:
    """The ODD to monitor (a safety-case ODD), from either format; ValueError when it is not valid."""
    from .safetycase import validate_odd
    if is_loop_odd(odd):
        odd = from_loop_odd(odd, project)
    bad = validate_odd(odd)
    if bad:
        raise OddError('the ODD: ' + '; '.join(bad))
    return odd


def load_odd(path: str, project=None) -> dict:
    with open(path, encoding='utf-8') as fh:
        return monitor_odd(json.load(fh), project)


# ================================================================================================ conditions
def state_from(weather: dict | None = None, speed_mps: float | None = None, place=None, road_type=None,
               handedness: str | None = None, traffic_vehicles=None, actors=None, scenario=None) -> dict:
    """A run's state in ODD terms (None: not known)."""
    st = {}
    for k, v in (weather or {}).items():
        if isinstance(v, (int, float)):
            st[k] = float(v)
    if 'sun_altitude_angle' in st:
        st['sun_altitude_deg'] = st['sun_altitude_angle']
    if speed_mps is not None:
        st['ego_speed_kmh'] = 3.6 * float(speed_mps)
    if place:
        st['place'] = [p for p in (place if isinstance(place, (list, tuple)) else [place]) if p]
    if road_type is not None:
        st['road_type'] = list(road_type)
    if handedness:
        st['handedness'] = [handedness]
    if traffic_vehicles is not None:
        st['traffic_vehicles'] = float(traffic_vehicles)
    if actors is not None:
        st['actors'] = sorted(set(actors))
    if scenario:
        st['scenario'] = [scenario]
    return st


def _fmt(v) -> str:
    return f'{v:.4g}' if isinstance(v, float) else str(v)


class ConditionMonitor:
    """Checks a state against the ODD's dimensions: (inside, [reasons], [dimensions not monitored])."""

    def __init__(self, odd: dict):
        self.odd = odd
        self.dims = odd['dimensions']

    def check(self, st: dict) -> tuple:
        reasons, unmon = [], []
        for name, d in self.dims.items():
            v = st.get(name, st.get(ALIASES.get(name, name)))
            if v is None or (isinstance(v, list) and not v and d['kind'] == 'categorical'):
                unmon.append(name)
                continue
            if d['kind'] == 'range':
                lo, hi = float(d['lo']), float(d['hi'])
                u = f' {d["unit"]}' if d.get('unit') else ''
                if v < lo:
                    reasons.append({'dim': name, 'text': f'{name} {_fmt(v)}{u} below the ODD\'s {lo:g}–{hi:g}{u}'})
                elif v > hi:
                    reasons.append({'dim': name, 'text': f'{name} {_fmt(v)}{u} above the ODD\'s {lo:g}–{hi:g}{u}'})
                continue
            vals = [str(x) for x in d['values']]
            got = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
            if d['kind'] == 'categorical':
                if not any(x in vals for x in got):
                    reasons.append({'dim': name, 'text': f'{name} {", ".join(got)} not in the ODD '
                                                         f'({", ".join(vals)})'})
            else:
                extra = [x for x in got if x not in vals]
                if extra:
                    reasons.append({'dim': name, 'text': f'{name}: {", ".join(extra)} not in the ODD'})
        return not reasons, reasons, unmon


class Timeline:
    """Per-step states compressed into segments of the same (inside, dimensions outside, data flag)."""

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.segments: list = []
        self.by_dim: dict = {}
        self.steps = 0
        self.outside_steps = 0
        self.data_scored = 0
        self.data_flagged = 0

    def add(self, t: float, inside: bool, reasons: list, data_flag: bool | None = None, data_score=None,
            scored: bool | None = None):
        """One step. data_flag: the data monitor's flag in effect (a flag holds until the next scored frame);
        scored: whether this step's frame was scored (default: a flag was given)."""
        self.steps += 1
        dims = sorted({r['dim'] for r in reasons})
        if data_flag is not None if scored is None else scored:
            self.data_scored += 1
            self.data_flagged += bool(data_flag)
        out = (not inside) or bool(data_flag)
        if out:
            self.outside_steps += 1
        for d in dims:
            self.by_dim[d] = self.by_dim.get(d, 0) + 1
        if data_flag:
            self.by_dim['data (out of distribution)'] = self.by_dim.get('data (out of distribution)', 0) + 1
        key = (inside, tuple(dims), bool(data_flag))
        s = self.segments[-1] if self.segments else None
        if s is not None and s['_key'] == key:
            s['t1'] = round(t + self.dt, 6)
            s['steps'] += 1
            if data_score is not None:
                s['max_score'] = max(s.get('max_score') or 0.0, float(data_score))
            return
        seg = {'_key': key, 't0': round(t, 6), 't1': round(t + self.dt, 6), 'steps': 1, 'inside': not out,
               'conditions_inside': inside, 'outside': dims, 'reasons': [r['text'] for r in reasons],
               'data_ood': bool(data_flag)}
        if data_score is not None:
            seg['max_score'] = float(data_score)
        self.segments.append(seg)

    def summary(self) -> dict:
        segs = [{k: v for k, v in s.items() if k != '_key'} for s in self.segments]
        return {'steps': self.steps, 'dt': self.dt, 'time_s': round(self.steps * self.dt, 4),
                'time_outside_s': round(self.outside_steps * self.dt, 4),
                'share_outside': (self.outside_steps / self.steps) if self.steps else None,
                'outside_by_dimension_s': {k: round(v * self.dt, 4) for k, v in sorted(self.by_dim.items())},
                'data': {'frames_scored': self.data_scored, 'frames_flagged': self.data_flagged},
                'segments': segs}


def road_features(cmap, location) -> list | None:
    """Road features at a location from a CARLA map (see the module docstring); None off the road."""
    try:
        wp = cmap.get_waypoint(location)
    except RuntimeError:
        return None
    if wp is None:
        return None
    out = []
    if getattr(wp, 'is_junction', False):
        out.append('junction')
    side = []
    for f in ('get_left_lane', 'get_right_lane'):
        try:
            n = getattr(wp, f)()
        except RuntimeError:
            n = None
        if n is not None and str(getattr(n, 'lane_type', 'Driving')).endswith('Driving'):
            side.append(n.lane_id * wp.lane_id > 0)
    if any(side):
        out.append('multi_lane')
    if any(not s for s in side):
        out.append('two_way')
    try:
        ahead = wp.next(STRAIGHT_AHEAD_M)
    except RuntimeError:
        ahead = []
    if ahead:
        dy = abs((ahead[0].transform.rotation.yaw - wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
        out.append('straight' if dy < STRAIGHT_DEG else 'curve')
    return out


# ================================================================================================ data: statistics
def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n <= 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def stats_vector(rgb: np.ndarray) -> np.ndarray:
    """The appearance statistics (core.realgap.IMAGE_STATS order) of one RGB uint8 image."""
    from .realgap import IMAGE_STATS, image_stats
    s = image_stats(rgb)
    return np.array([s[k] for k in IMAGE_STATS], float)


def ledoit_wolf(X: np.ndarray) -> tuple:
    """(shrunk covariance, shrinkage) of standardised rows X (n × p): Ledoit & Wolf (2004), towards μ·I."""
    n, p = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    d2 = float(np.sum((S - mu * np.eye(p)) ** 2))
    b2 = float(sum(np.sum((np.outer(x, x) - S) ** 2) for x in Xc)) / (n * n)
    b2 = min(b2, d2)
    lam = b2 / d2 if d2 > 0 else 1.0
    return lam * mu * np.eye(p) + (1 - lam) * S, lam


def fit_appearance(X: np.ndarray) -> dict:
    X = np.asarray(X, float)
    ok = np.all(np.isfinite(X), axis=1)
    X = X[ok]
    if len(X) < 3:
        raise OddError(f'{len(X)} reference frames: at least 3 are needed')
    mean = X.mean(0)
    std = X.std(0)
    std = np.where(std > 1e-9, std, 1.0)
    Z = (X - mean) / std
    C, lam = ledoit_wolf(Z)
    return {'mean': mean.tolist(), 'std': std.tolist(), 'precision': np.linalg.inv(C).tolist(),
            'shrinkage': float(lam), 'n': int(len(X))}


def appearance_score(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    Z = (X - np.asarray(model['mean'])) / np.asarray(model['std'])
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e3, neginf=-1e3)
    P = np.asarray(model['precision'])
    return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', Z, P, Z), 0.0))


def knn_score(ref: np.ndarray, F: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Mean cosine distance of each row of F to its k nearest rows of ref."""
    A = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    B = np.atleast_2d(F) / np.maximum(np.linalg.norm(np.atleast_2d(F), axis=1, keepdims=True), 1e-12)
    D = 1.0 - B @ A.T
    k = min(k, A.shape[0])
    return np.sort(D, axis=1)[:, :k].mean(1)


def conformal_threshold(scores, far: float) -> float:
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if not n:
        raise OddError('no calibration frames')
    k = math.ceil((n + 1) * (1 - far))
    return float(s[min(k, n) - 1]) if k <= n else float(s[-1] * 1.0000001 + 1e-12)


# ================================================================================================ data: frames
def run_frames(run_dir: str, camera: str | None = None, every: int = 1, limit: int | None = None) -> list:
    """[(step, sim_t, image path)] of one camera of a recorded run (falcon-raw frames.jsonl; a pinhole camera by
    default; a frozen repeat of an earlier frame is skipped). Needs no run.json, so an unfinished recording reads."""
    recs = _frames_of(run_dir)
    sensors = {}
    for f in ('dataset.json', 'drive.json', 'stack.json'):
        p = os.path.join(run_dir, f)
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as fh:
                d = json.load(fh)
            for s in d.get('sensors') or (d.get('rig') or {}).get('sensors') or []:
                sensors.setdefault(s['id'], s.get('type'))
    cams = sorted({k[:-4] for r in recs for k in (r.get('files') or {}) if k.endswith('/rgb')})
    if not cams:
        return []
    pin = [c for c in cams if sensors.get(c) == 'camera.pinhole'] or cams
    cid = camera or pin[0]
    if cid not in cams:
        raise OddError(f'{run_dir}: no camera {cid!r} (has {", ".join(cams)})')
    out = []
    for rec in recs:
        rel = (rec.get('files') or {}).get(f'{cid}/rgb')
        if rel is None or rec['step'] % max(every, 1):
            continue
        try:
            if int(os.path.basename(rel).split('.')[0]) != int(rec['step']):
                continue
        except ValueError:
            pass
        out.append((rec['step'], float(rec.get('sim_t', 0.0)), os.path.join(run_dir, rel)))
        if limit and len(out) >= limit:
            break
    return out


def frame_stats(paths: list, progress=None) -> np.ndarray:
    from .realgap import load_rgb
    out = []
    for i, p in enumerate(paths):
        out.append(stats_vector(load_rgb(p)))
        if progress:
            progress((i + 1) / len(paths))
    return np.asarray(out, float).reshape(len(paths), -1)


def frame_features(paths: list, workspace: str, progress=None) -> np.ndarray:
    from .realgap import Features, load_rgb
    net = Features(workspace)
    out = []
    for i in range(0, len(paths), 32):
        out.append(net([load_rgb(p) for p in paths[i:i + 32]]))
        if progress:
            progress(min(i + 32, len(paths)) / len(paths))
    return np.concatenate(out) if out else np.zeros((0, 2048))


def features_available(workspace: str) -> bool:
    from .realgap import DEFAULT_FEATURES, FEATURES, features_path
    p = features_path(workspace, DEFAULT_FEATURES) if workspace else ''
    return bool(p) and os.path.isfile(p) and os.path.getsize(p) > FEATURES[DEFAULT_FEATURES]['mb'] * 0.9e6


def _groups_of(in_sets: dict) -> tuple:
    """({group: rows}, unit, note): drives as groups when there are three or more, else each drive cut into three
    contiguous blocks (frames of one drive are correlated: then the measured rate is optimistic)."""
    names = sorted(in_sets)
    if len(names) >= 3:
        return {k: np.asarray(in_sets[k], float) for k in names}, 'drive', ''
    out = {}
    for k in names:
        X = np.asarray(in_sets[k], float)
        for j, idx in enumerate(np.array_split(np.arange(len(X)), 3)):
            if len(idx):
                out[f'{k}#{j}'] = X[idx]
    return out, 'block', ('fewer than three in-ODD drives: each was cut into three contiguous blocks used as the '
                          'units — frames of one drive are correlated, so the measured false-alarm rate is optimistic')


def _oof(groups: dict, names: list, fit, score) -> np.ndarray:
    """Out-of-fold scores: each group's frames scored by a model fitted on the other groups of names."""
    return np.concatenate([score(fit([o for o in names if o != g]), g) for g in names])


def calibrate(in_sets: dict, out_sets: dict | None = None, far: float = 0.01, seed: int = 0,
              features: dict | None = None, about: str = '') -> dict:
    """A reference and its threshold(s) from in-ODD frames, and the measured rates (module docstring).

    in_sets: {drive: (n × 11) appearance statistics}; out_sets: {name: (m × 11)} frames outside the ODD;
    features (optional): {'in': {drive: (n × d)}, 'out': {name: (m × d)}} ResNet-50 features of the same frames.

    Leave-one-drive-out cross-fitting: the threshold is the conformal (1 − far) quantile of OUT-OF-FOLD scores (each
    drive scored by a reference fitted without it), so drive-to-drive variation is in it; the final reference is
    fitted on every drive. The false-alarm rate is measured by repeating the whole procedure without each drive in
    turn and counting that drive's frames above the threshold so obtained (nested: the drive takes no part in its
    own threshold or reference)."""
    from .realgap import IMAGE_STATS
    groups, unit, note = _groups_of(in_sets)
    names = sorted(groups)
    if len(names) < 3 or min(len(v) for v in groups.values()) < 2:
        raise OddError('too few in-ODD frames: at least three drives (or one drive of six frames) are needed')
    mons = {'appearance': (lambda ns: fit_appearance(np.concatenate([groups[n] for n in ns])),
                           lambda m, g: appearance_score(m, groups[g]),
                           lambda m, X: appearance_score(m, X))}
    if features:
        fin = {}
        if unit == 'drive':
            fin = {k: np.asarray(features['in'][k], float) for k in names}
        else:
            for k in sorted(in_sets):
                F = np.asarray(features['in'][k], float)
                for j, idx in enumerate(np.array_split(np.arange(len(F)), 3)):
                    if len(idx):
                        fin[f'{k}#{j}'] = F[idx]
        mons['features'] = (lambda ns: np.concatenate([fin[n] for n in ns]),
                            lambda m, g: knn_score(m, fin[g]),
                            lambda m, X: knn_score(m, X))
    monitors, per_unit, flags_all = {}, {}, {}
    for key, (fit, score_g, score_x) in mons.items():
        thr = conformal_threshold(_oof(groups, names, fit, score_g), far)
        model = fit(names)
        monitors[key] = ({'kind': 'mahalanobis', 'model': model, 'threshold': thr, 'stats': list(IMAGE_STATS)}
                         if key == 'appearance' else
                         {'kind': 'knn_cosine', 'k': KNN_K, 'reference': np.asarray(model, np.float32).tolist(),
                          'threshold': thr, 'extractor': 'resnet50-imagenet1k-v2'})
        for t in names:                                   # nested: t takes no part in its own threshold or reference
            rest = [n for n in names if n != t]
            thr_t = conformal_threshold(_oof(groups, rest, fit, score_g), far)
            f = score_g(fit(rest), t) > thr_t
            flags_all[t] = f if t not in flags_all else (flags_all[t] | f)
            per_unit.setdefault(t, {})[key] = int(f.sum())
    n_test = sum(len(groups[t]) for t in names)
    k_fa = int(sum(int(f.sum()) for f in flags_all.values()))
    report = {'target_false_alarm_rate': far, 'unit': unit, 'units': len(names), 'frames': n_test,
              'test': {'frames': n_test, 'false_alarms': k_fa, 'false_alarm_rate': k_fa / n_test,
                       'ci95': wilson(k_fa, n_test),
                       'per_unit': {t: {'frames': int(len(groups[t])), 'false_alarms': int(flags_all[t].sum()),
                                        'per_monitor': per_unit[t]} for t in names}},
              'detection': {}, 'note': note,
              'interval': 'Wilson 95 % over frames (frames of a drive are correlated: the interval is optimistic)'}
    for name, Xo in (out_sets or {}).items():
        Xo = np.asarray(Xo, float)
        if not len(Xo):
            continue
        so = appearance_score(monitors['appearance']['model'], Xo)
        fl = so > monitors['appearance']['threshold']
        per = {'appearance': int(fl.sum())}
        if 'features' in monitors and name in (features.get('out') or {}):
            fo = knn_score(np.asarray(monitors['features']['reference'], float),
                           np.asarray(features['out'][name], float))
            per['features'] = int((fo > monitors['features']['threshold']).sum())
            fl = fl | (fo > monitors['features']['threshold'])
        k = int(fl.sum())
        report['detection'][name] = {'frames': int(len(Xo)), 'detected': k, 'rate': k / len(Xo),
                                     'ci95': wilson(k, len(Xo)), 'per_monitor': per,
                                     'median_appearance_score': float(np.median(so))}
    from .provenance import block
    return {'schema': REFERENCE_SCHEMA, 'about': about, 'monitors': monitors, 'calibration': report,
            'method': 'leave-one-drive-out cross-fitting: the threshold is the conformal quantile of out-of-fold '
                      'scores for the target false-alarm rate; the false-alarm rate is measured nested (each drive '
                      'excluded from its own threshold and reference)',
            'provenance': block({}, seed=seed)}


class DataMonitor:
    """A reference (calibrate()) applied to frames: (flag, scores)."""

    def __init__(self, ref: dict, workspace: str = ''):
        if ref.get('schema') != REFERENCE_SCHEMA:
            raise OddError(f'not an ODD reference (schema {ref.get("schema")!r})')
        self.ref = ref
        self.app = ref['monitors']['appearance']
        self.feat = ref['monitors'].get('features')
        self.net = None
        if self.feat is not None:
            if workspace and features_available(workspace):
                from .realgap import Features
                self.net = Features(workspace)
                self.feat_ref = np.asarray(self.feat['reference'], float)
            else:
                self.feat = None                       # not downloaded here: the appearance monitor alone
        self.unavailable = 'features' in ref['monitors'] and self.feat is None

    def score(self, rgb: np.ndarray) -> dict:
        a = float(appearance_score(self.app['model'], stats_vector(rgb)[None])[0])
        out = {'appearance': a, 'flag': a > self.app['threshold']}
        if self.feat is not None:
            f = float(knn_score(self.feat_ref, self.net([rgb]))[0])
            out['features'] = f
            out['flag'] = out['flag'] or f > self.feat['threshold']
        out['score'] = a / self.app['threshold'] if self.app['threshold'] > 0 else a
        return out


def load_reference(path: str) -> dict:
    p = os.path.join(path, 'reference.json') if os.path.isdir(path) else path
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


# ================================================================================================ a recorded run
def _config_of(run_dir: str) -> dict:
    for f in ('stack.json', 'drive.json'):
        p = os.path.join(run_dir, f)
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as fh:
                return json.load(fh)
    p = os.path.join(run_dir, 'run.json')
    if os.path.isfile(p):
        with open(p, encoding='utf-8') as fh:
            return ((json.load(fh).get('provenance') or {}).get('settings')) or {}
    raise OddError(f'{run_dir}: no run config (stack.json, drive.json or run.json)')


def _frames_of(run_dir: str) -> list:
    p = os.path.join(run_dir, 'frames.jsonl')
    if not os.path.isfile(p):
        raise OddError(f'{run_dir}: no frames.jsonl')
    with open(p, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def static_state(cfg: dict) -> tuple:
    """(state from the config, road types from the scenario heuristic, notes)."""
    from . import weather as W
    from .safetycase import run_coordinates
    c = run_coordinates(cfg)
    try:
        w = W.resolve(cfg.get('weather') or cfg.get('weather_source'))
    except (KeyError, ValueError, TypeError):
        w = {}
    st = state_from(w, None, c['place'], None, c['handedness'][0] if c['handedness'] else None,
                    c['traffic_vehicles'], c['actors'] if cfg.get('scenario') else None,
                    (c['scenario'] or [None])[0])
    return st, c['road_type']


def monitor_run(run_dir: str, odd: dict, reference: dict | None = None, every: int = 5, camera: str | None = None,
                workspace: str = '', write: bool = True, data: bool = True) -> dict:
    """The monitor over a recorded run (module docstring); writes odd-monitor.json into the run folder."""
    cfg = _config_of(run_dir)
    frames = _frames_of(run_dir)
    if not frames:
        raise OddError(f'{run_dir}: no frames')
    fps = float(cfg.get('fps', 20.0))
    st0, road_cfg = static_state(cfg)
    notes = []
    has_road = any('odd' in r and 'road' in (r.get('odd') or {}) for r in frames)
    if not has_road and 'road_type' in odd['dimensions']:
        notes.append('road type from the scenario\'s requirements at the start (heuristic, safetycase.'
                     'run_coordinates), not observed along the route')
    mon = ConditionMonitor(odd)
    dm = None
    imgs = {}
    if data and reference is not None:
        dm = DataMonitor(reference, workspace)
        if dm.unavailable:
            notes.append('the reference has a feature monitor but the ResNet-50 extractor is not downloaded here: '
                         'appearance statistics only')
        try:
            imgs = {s: p for s, _, p in run_frames(run_dir, camera, every)}
        except OddError as e:
            notes.append(f'no camera frames for the data monitor: {e}')
    tl = Timeline(1.0 / fps)
    unmon = set()
    last = None
    from .realgap import load_rgb
    for rec in frames:
        st = dict(st0)
        if rec.get('speed_mps') is not None:
            st['ego_speed_kmh'] = 3.6 * float(rec['speed_mps'])
        o = rec.get('odd') or {}
        st['road_type'] = o.get('road') if o.get('road') is not None else road_cfg
        if o.get('weather'):
            st.update(state_from(o['weather']))
        inside, reasons, um = mon.check(st)
        unmon.update(um)
        score = None
        scored = dm is not None and rec['step'] in imgs
        if scored:
            r = dm.score(load_rgb(imgs[rec['step']]))
            last, score = bool(r['flag']), r['score']
        tl.add(float(rec.get('sim_t', rec['step'] / fps)), inside, reasons, last, score, scored)
    doc = {'schema': TIMELINE_SCHEMA, 'run': os.path.abspath(run_dir), 'odd': odd.get('name'),
           'monitored': sorted(set(odd['dimensions']) - unmon), 'not_monitored': sorted(unmon),
           'data_monitor': None if dm is None else {
               'appearance_threshold': dm.app['threshold'], 'features': dm.feat is not None, 'every': every,
               'rule': 'a frame\'s flag holds until the next scored frame',
               'target_false_alarm_rate': (reference.get('calibration') or {}).get('target_false_alarm_rate')},
           'notes': notes, **tl.summary()}
    if write:
        from .provenance import write_json
        write_json(os.path.join(run_dir, MONITOR_FILE), doc)
    return doc


# ================================================================================================ suites, the Loop
def suite_outside(rows: list, odd: dict) -> dict:
    """Per scenario of a suite: the time its runs spent outside the ODD (condition monitors, from each run's
    frames), and the dimensions why."""
    per, total, outside, runs, unread = {}, 0.0, 0.0, 0, 0
    for r in rows:
        if r.get('status') != 'ok' or not r.get('run'):
            continue
        try:
            doc = monitor_run(r['run'], odd, None, write=False, data=False)
        except (OddError, OSError, ValueError, KeyError):
            unread += 1
            continue
        runs += 1
        g = per.setdefault(r.get('scenario') or '?', {'name': r.get('scenario_name') or r.get('scenario'),
                                                      'runs': 0, 'runs_outside': 0, 'time_s': 0.0,
                                                      'time_outside_s': 0.0, 'by_dimension_s': {}})
        g['runs'] += 1
        g['runs_outside'] += doc['time_outside_s'] > 0
        g['time_s'] += doc['time_s']
        g['time_outside_s'] += doc['time_outside_s']
        for k, v in doc['outside_by_dimension_s'].items():
            g['by_dimension_s'][k] = round(g['by_dimension_s'].get(k, 0.0) + v, 4)
        total += doc['time_s']
        outside += doc['time_outside_s']
    for g in per.values():
        g['share'] = g['time_outside_s'] / g['time_s'] if g['time_s'] else None
        g['time_s'], g['time_outside_s'] = round(g['time_s'], 4), round(g['time_outside_s'], 4)
    return {'runs': runs, 'unread': unread, 'time_s': round(total, 4), 'time_outside_s': round(outside, 4),
            'share': outside / total if total else None, 'scenarios': per,
            'method': 'condition monitors over each run\'s recorded frames (core.oddmonitor): weather and light from '
                      'the run\'s config, the ego\'s actual speed per frame, place, handedness, scenario, actors and '
                      'road type (from the worker\'s per-frame record when it monitored, else the scenario '
                      'heuristic)'}


def loop_cells(loop_odd: dict, odd: dict, project=None) -> list:
    """Each Loop evaluation cell (world × weather entry) against a safety-case ODD: [{label, inside, partly,
    reasons}] — a ranged weather field is checked at both ends (partly: one end inside, one outside)."""
    from . import loop as L
    from . import weather as W
    mon = ConditionMonitor(monitor_odd(odd, project))
    out = []
    for cell in L.odd_cells(loop_odd):
        we = cell['weather']
        base = W.resolve({k: v for k, v in we.items() if k in W.FIELDS or k == 'preset'})
        place = [cell['world']]
        if project is not None:
            w = project.item('worlds', cell['world']) or {}
            if w.get('town'):
                place.append(w['town'])
        ends = []
        for pick in (0, 1):
            wf = dict(base)
            for k, r in (we.get('ranges') or {}).items():
                wf[k] = float(r[pick])
            ends.append(mon.check(state_from(wf, place=place)))
        reasons = {r['text']: r for e in ends for r in e[1]}
        n_in = sum(1 for e in ends if e[0])
        out.append({'label': cell['label'], 'world': cell['world'], 'weather': we.get('name'),
                    'inside': n_in == 2, 'partly': n_in == 1, 'reasons': [r['text'] for r in reasons.values()]})
    return out


# ================================================================================================ the command
def main_odd(argv) -> int:
    """falcon odd monitor RUN... --odd FILE [--reference REF] | falcon odd calibrate --in RUN... --out FILE ..."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog='falcon odd', description='runtime ODD monitoring (docs/odd-monitoring.md)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('monitor', help='the ODD monitor over recorded runs: odd-monitor.json in each')
    m.add_argument('runs', nargs='+')
    m.add_argument('--odd', required=True, help='a safety-case ODD (falcon.odd/1 with dimensions) or the Loop\'s')
    m.add_argument('--reference', default='', help='a reference from `falcon odd calibrate` (file, or its folder)')
    m.add_argument('--every', type=int, default=5, help='score every n-th camera frame (default 5)')
    m.add_argument('--camera', default=None)
    m.add_argument('--workspace', default='', help='where the ResNet-50 weights are (a feature monitor)')
    m.add_argument('--no-write', action='store_true', help='print only; do not write odd-monitor.json')
    c = sub.add_parser('calibrate', help='a reference of in-ODD drives, its threshold and the measured rates')
    c.add_argument('--in', dest='inside', nargs='+', required=True, help='in-ODD runs (three or more drives)')
    c.add_argument('--out-of', nargs='*', default=[], metavar='NAME=RUN[,RUN]',
                   help='out-of-ODD runs to measure the detection rate on')
    c.add_argument('--far', type=float, default=0.01, help='the target false-alarm rate (default 0.01)')
    c.add_argument('--every', type=int, default=5)
    c.add_argument('--camera', default=None)
    c.add_argument('--features', default='', metavar='WORKSPACE',
                   help='also a ResNet-50 feature monitor (the weights must be downloaded in this workspace)')
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--out', required=True, help='the reference file to write (reference.json)')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'monitor':
            odd = load_odd(a.odd)
            ref = load_reference(a.reference) if a.reference else None
            rc = 0
            for r in a.runs:
                doc = monitor_run(r, odd, ref, a.every, a.camera, a.workspace, write=not a.no_write)
                share = doc['share_outside']
                print(f'{r}: {doc["time_outside_s"]:.2f} s of {doc["time_s"]:.2f} s outside the ODD '
                      f'({"—" if share is None else f"{share:.0%}"})'
                      + (f'; data flagged {doc["data"]["frames_flagged"]} of {doc["data"]["frames_scored"]} frames'
                         if doc['data']['frames_scored'] else ''))
                for k, v in doc['outside_by_dimension_s'].items():
                    print(f'    {k}: {v:.2f} s')
                if doc['not_monitored']:
                    print(f'    not monitored: {", ".join(doc["not_monitored"])}')
                for n in doc['notes']:
                    print(f'    note: {n}')
            return rc
        ins = {}
        for r in a.inside:
            name = os.path.basename(os.path.normpath(r))
            while name in ins:
                name += "'"
            fr = run_frames(r, a.camera, a.every)
            if not fr:
                raise OddError(f'{r}: no camera frames')
            ins[name] = [p for _, _, p in fr]
        outs = {}
        for spec in a.out_of:
            name, _, runs = spec.partition('=')
            if not runs:
                raise OddError(f'--out-of {spec!r}: NAME=RUN[,RUN]')
            outs[name] = [p for r in runs.split(',') for _, _, p in run_frames(r, a.camera, a.every)]
        X_in = {k: frame_stats(v) for k, v in ins.items()}
        X_out = {k: frame_stats(v) for k, v in outs.items()}
        feats = None
        if a.features:
            if not features_available(a.features):
                raise OddError(f'the ResNet-50 weights are not downloaded in {a.features} (falcon gap fetches them)')
            feats = {'in': {k: frame_features(v, a.features) for k, v in ins.items()},
                     'out': {k: frame_features(v, a.features) for k, v in outs.items()}}
        ref = calibrate(X_in, X_out, a.far, a.seed, feats, about=f'in-ODD: {", ".join(a.inside)}')
        ref['inputs'] = {'in': list(a.inside), 'out_of': list(a.out_of), 'every': a.every, 'camera': a.camera}
        from .provenance import write_json
        out = os.path.join(a.out, 'reference.json') if os.path.isdir(a.out) else a.out
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        write_json(out, ref)
        c = ref['calibration']
        t = c['test']
        print(f'reference: {out}')
        print(f'thresholds: ' + ', '.join(f'{k} {v["threshold"]:.4g}' for k, v in ref['monitors'].items()))
        print(f'false alarms (measured, nested by {c["unit"]}): {t["false_alarms"]} of {t["frames"]} frames = '
              f'{t["false_alarm_rate"]:.3f} (95 % {t["ci95"][0]:.3f}–{t["ci95"][1]:.3f}); target {a.far:g}')
        for k, d in c['detection'].items():
            print(f'detected {k}: {d["detected"]} of {d["frames"]} = {d["rate"]:.3f} '
                  f'(95 % {d["ci95"][0]:.3f}–{d["ci95"][1]:.3f})')
        if c['note']:
            print(f'note: {c["note"]}')
        return 0
    except (OddError, OSError, ValueError) as e:
        print(f'falcon odd: {e}', file=sys.stderr)
        return 2
