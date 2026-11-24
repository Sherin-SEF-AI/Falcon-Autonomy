"""Moments: every recorded run indexed by what happened in it, searched in plain words (numpy and sqlite3 from the
standard library; no model, no network). docs/moments.md is the manual.

    extract(scene) -> [moment]                 the events of one drive, from its tracks (deterministic)
    scene_from_run(run_dir) / scene_from_log(log.json)
    index(workspace, …) -> counts              the workspace's store, only new or changed runs read again
    parse_query(text) -> filter                plain words -> a structured filter, by a fixed vocabulary
    search(workspace, filter) -> hits          ranked by how much of the filter matched, then by severity
    stats(workspace) -> exposure table         how often each moment type occurs per hour of driving
    make_scenario(workspace, moment, name)     a hit as a Falcon scenario and OpenSCENARIO (core.incident's path)
    falcon moments index|search|show|stats

What is read. A run folder's frames.jsonl (the ego's pose and speed per recorded step), actors.jsonl (every other
actor's pose, velocity, box, category and Falcon subclass) or, without it, trajectories.json (a scenario's own actors),
lanes.json and the map's OpenDRIVE file when the run names one (through core.rules.trace_from_run and its signals:
on_lane, in_junction, on_breaker, tl_red, tl_distance_m, ped_on_crossing), and run.json (faults applied, a verified
minimal-risk manoeuvre, the weather and sun). An imported log (logs/<name>/log.json, core.logreplay) is sampled at
LOG_DT_S. A near-miss ranking (core.nearmiss, from `falcon dashcam ingest`) is taken as it is: one moment per mined
event, private. Nothing is written into a run folder.

Geometry. Everything is in the ego's frame at each sample: lon (the actor's centre ahead of the ego's, m), lat (+ to
the ego's right; CARLA is left-handed), the box-to-box gaps along and across the ego's heading
(core.criticality.axis_gaps), the box distance, the time to collision at constant velocity and the deceleration
required to avoid a crash (core.criticality: exact for oriented boxes). "In the ego's path" is a straight corridor:
|lat| < half the two widths + IN_PATH_MARGIN_M. On a curve a straight corridor mislabels far actors; ranges are
therefore short (CUT_GAP_MAX_M, LEAD_RANGE_M) and the limit is stated in the manual.

Moment types (TYPES) and what each needs — every threshold is a named constant below with where it comes from:

    cut_in, cut_out            a same-direction vehicle entering / leaving the corridor ahead from / to the side
    hard_brake_ego             the ego's deceleration at or over HARD_BRAKE_MPS2 (the actor ahead named when there is one)
    hard_brake_lead            the same for a vehicle ahead in the corridor
    close_pass                 an actor alongside (the boxes overlap along the heading) closer than CLOSE_PASS_M
    overtake                   the ego passing a moving same-direction vehicle, or being passed (numbers.by)
    crossing, jaywalking       a pedestrian or animal moving across the corridor ahead; jaywalking when it is a
                               pedestrian off every crosswalk the map knows (else, a heuristic: not at a junction)
    oncoming_in_lane           a moving vehicle heading against the ego inside its corridor
    stopped_obstacle           anything standing in the corridor ahead while the ego moves
    red_light_violation        the ego's front past the stop line of a red light (only runs that record the light)
    low_ttc, high_decel_needed TTC under LOW_TTC_S; required deceleration at or over DRAC_HIGH_MPS2
    near_miss, collision       no contact but TTC under NEAR_MISS_TTC_S or a fast pass within NEAR_MISS_DIST_M; overlap
    lane_departure             the ego's centre off every driving lane of the map (needs lanes)
    sensor_fault, mrm          run.json's faults_applied; its verified minimal-risk manoeuvre

Stop-sign violations are not a type: core.rules has no stop-sign signal for a recorded run.

The store: <workspace>/moments/moments.sqlite (SQLite, standard library). SQLite and not Parquet because the index is
incremental — one run's rows are replaced in one transaction when its files change — which a Parquet file cannot do
without being rewritten, and because pyarrow is optional in Falcon while sqlite3 is always there. A run is keyed the
way core.runindex keys its cache: its path and the sizes and modification times of the files read, plus VERSION (a
change of the extractor re-reads everything). The store can be deleted at any time; the runs are the truth.

Queries. parse_query() uses core.scenario_check's phrase set (actor kinds with the Indian English road words: auto,
two wheeler, bike, scooty, cow …; manoeuvres; side; light; weather; junction and speed breaker) plus this module's
phrases for the moment types it does not have (TYPE_PHRASES) and numeric constraints ("TTC under 1.5 s", "above 40
km/h"). It is rule-based: a word outside the vocabulary is not understood, and is listed (never dropped silently);
the filter is always shown back ("understood as: …"). Types and numeric constraints are required of a hit; the other
terms (actor, side, light, weather, world, road, source) are scored — a hit matching all of them comes first, then
partial matches with what they miss — and ties go to the more severe moment. Severity is a 0–1 ranking heuristic
(SEVERITY_NOTE), not a risk estimate.

Privacy. A run derived from the user's dashcam footage (assistant.privacy.private_reason, a route replay, a near-miss
ranking) is indexed on this machine like any other but flagged private: the command line and the window show it to
the user; the assistant's tools get counts only (public_only below). A scenario made from a private moment carries
core.incident's dashcam source, so it stays private.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3

import numpy as np

SCHEMA = 'falcon.moments/1'
VERSION = 1                          # bumped when a detector or a threshold changes: every run is read again
FOLDER = 'moments'
DB_FILE = 'moments.sqlite'
FILES = ('run.json', 'frames.jsonl', 'actors.jsonl', 'trajectories.json', 'lanes.json')

TYPES = ('cut_in', 'cut_out', 'hard_brake_ego', 'hard_brake_lead', 'close_pass', 'overtake', 'crossing', 'jaywalking',
         'oncoming_in_lane', 'stopped_obstacle', 'red_light_violation', 'low_ttc', 'high_decel_needed', 'near_miss',
         'collision', 'lane_departure', 'sensor_fault', 'mrm')
NAMES = {'cut_in': 'cut-in', 'cut_out': 'cut-out', 'hard_brake_ego': 'hard braking (ego)',
         'hard_brake_lead': 'hard braking (lead)', 'close_pass': 'close pass', 'overtake': 'overtaking',
         'crossing': 'crossing', 'jaywalking': 'jaywalking', 'oncoming_in_lane': 'oncoming in lane',
         'stopped_obstacle': 'stopped obstacle', 'red_light_violation': 'red-light violation', 'low_ttc': 'low TTC',
         'high_decel_needed': 'high required deceleration', 'near_miss': 'near-miss', 'collision': 'collision',
         'lane_departure': 'lane departure', 'sensor_fault': 'sensor fault active', 'mrm': 'MRM triggered'}

# actor classes: core.scenarios.KINDS (the Indian kinds among them) and 'obstacle' for a prop without a subclass
VEHICLES = ('car', 'truck', 'bus', 'van', 'motorcycle', 'bicycle', 'autorickshaw', 'cycle_rickshaw')
WALKERS = ('pedestrian', 'child')
ANIMALS = ('cattle',)
STATIC = ('handcart', 'vendor', 'obstacle')
CLASSES = VEHICLES + WALKERS + ANIMALS + STATIC
GROUPS = {'two_wheeler': ('motorcycle', 'bicycle'), 'three_wheeler': ('autorickshaw', 'cycle_rickshaw'),
          'vru': ('pedestrian', 'child', 'bicycle', 'motorcycle', 'cycle_rickshaw'), 'animal': ANIMALS,
          'vehicle': VEHICLES, 'walker': WALKERS}

# ---- thresholds (each: value, unit, where it comes from). Heuristic = Falcon's own choice, not from a standard.
IN_PATH_MARGIN_M = 0.3         # corridor: half widths + this (core.rules' lead corridor uses the same 0.3 m)
SAME_DIR_DEG = 45.0            # a vehicle "going the ego's way": heading within this (heuristic; rules uses 30–60)
OPPOSITE_DEG = 135.0           # heading against the ego: core.scenario_check THRESHOLDS wrong_way_deg
MOVING_MPS = 1.0               # "moving" for vehicles (heuristic: above creep and tracking noise)
STOPPED_MPS = 0.3              # "standing": core.scenario_check THRESHOLDS stopped_mps
STOPPED_S = 1.0                # … for at least this long: core.scenario_check THRESHOLDS stopped_s
EGO_MOVING_MPS = 2.0           # the ego "driving" (7 km/h; heuristic: excludes queue creep)
LEAD_RANGE_M = 60.0            # how far ahead a lead, an oncoming vehicle or an obstacle counts: core.drivefeatures
CUT_GAP_MAX_M = 25.0           # a cut-in enters the corridor within this gap: core.scenario_check cut_gap_max_m
CUT_OUT_RANGE_M = 40.0         # a cut-out's lead was within this gap (heuristic)
CUT_OUTSIDE_M = 0.5            # "beside the corridor": this far outside it (heuristic hysteresis against jitter)
CUT_LOOKBACK_S = 3.0           # the lateral move happens within this time (heuristic: a lane change takes 2–6 s)
CUT_HOLD_S = 0.5               # it then stays in the corridor (cut-in) / was a lead (cut-out) at least this long
CUT_TTC_WINDOW_S = 3.0         # a cut-in's min TTC is taken over this time after the entry
HARD_BRAKE_MPS2 = 3.0          # hard braking: core.scenario_check THRESHOLDS brake_decel_mps2
BRAKE_HOLD_S = 0.25            # … held this long: core.scenario_check THRESHOLDS brake_window_s
CLOSE_PASS_M = 1.0             # lateral clearance of a close pass (heuristic; not a legal passing distance)
CLOSE_PASS_REL_MPS = 1.0       # … with at least this speed difference (one is passing the other)
OVERTAKE_LAT_M = 5.25          # an overtake happens within one and a half 3.5 m lanes beside the ego (heuristic)
OVERTAKE_MAX_S = 15.0          # from behind to ahead (or the reverse) within this time (heuristic)
CROSS_RANGE_M = 40.0           # a crossing counts within this gap ahead: core.drivefeatures VRU_RANGE_M
CROSS_LAT_MPS = 0.3            # … moving across at this speed or more: core.scenario_check THRESHOLDS moving_mps
ONCOMING_S = 0.5               # an oncoming vehicle stays in the corridor at least this long (heuristic)
LOW_TTC_S = 1.5                # low TTC: core.criticality TTC_SCALE_S; 1.5 s is the usual serious-conflict threshold
#                                in the traffic-conflict literature (recalled, not checked against the document)
DRAC_HIGH_MPS2 = 3.35          # high required deceleration: the DRAC conflict threshold usually cited from Archer
#                                (2005) (recalled, not checked against the document)
NEAR_MISS_TTC_S = 1.0          # near-miss: TTC under this without contact (heuristic, stricter than LOW_TTC_S)
NEAR_MISS_DIST_M = 0.5         # … or the boxes within this distance at NEAR_MISS_REL_MPS or more (heuristic)
NEAR_MISS_REL_MPS = 2.0
LANE_DEPART_S = 0.25           # the ego's centre off every driving lane at least this long (heuristic)
RED_MIN_MPS = 0.5              # past a red light's stop line faster than this (heuristic: not a stop on the line)
JOIN_S = 0.5                   # two detections of one kind for one actor closer than this are one moment
TTC_HORIZON_S = 10.0           # core.rules TTC_HORIZON_S
LOG_DT_S = 0.1                 # an imported log is sampled at 10 Hz
SPEED_BANDS = ((1.0, 'stopped'), (30.0, 'low'), (60.0, 'medium'))       # km/h upper bounds; above: 'high' (heuristic)
THRESHOLDS = {k: v for k, v in dict(globals()).items()
              if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool) and k != 'VERSION'}

SEVERITY_BASE = {'collision': 1.0, 'near_miss': 0.7, 'red_light_violation': 0.6, 'low_ttc': 0.5,
                 'high_decel_needed': 0.5, 'oncoming_in_lane': 0.5, 'lane_departure': 0.4, 'mrm': 0.4,
                 'hard_brake_ego': 0.3, 'hard_brake_lead': 0.3, 'close_pass': 0.3, 'crossing': 0.3, 'jaywalking': 0.35,
                 'cut_in': 0.3, 'sensor_fault': 0.3, 'stopped_obstacle': 0.2, 'cut_out': 0.1, 'overtake': 0.1}
SEVERITY_TTC_S = 3.0
SEVERITY_DECEL = 8.0
SEVERITY_NOTE = ('severity is a ranking heuristic in 0–1: 1 for a collision; otherwise the largest of the type\'s base '
                 f'value, 0.9·(1 − min TTC / {SEVERITY_TTC_S:g} s) and 0.9·(deceleration / {SEVERITY_DECEL:g} m/s²), '
                 'capped at 0.9. It orders hits; it is not a probability of harm')


class MomentsError(ValueError):
    pass


# ================================================================================================ a drive's tracks
class Scene:
    """One drive as arrays: sample times (s from the first), the ego's pose and speed, the other actors (each on its
    own subset of the samples), the per-sample signals a map gives (NaN where unknown) and ready-made moments
    (faults, an MRM)."""

    def __init__(self, t, x, y, yaw, speed, ego_size=(4.9, 1.84)):
        self.t = np.asarray(t, float)
        self.x, self.y, self.yaw = np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float)
        self.speed = np.nan_to_num(np.asarray(speed, float))
        self.ego_size = (float(ego_size[0]), float(ego_size[1]))
        self.actors: dict = {}
        self.sig: dict = {}
        self.given: list = []
        self.notes: list = []

    def add_actor(self, ident, cls: str, idx, x, y, yaw, vx, vy, length, width):
        idx = np.asarray(idx, int)
        if len(idx) < 2:
            return
        f = lambda v: np.broadcast_to(np.asarray(v, float), idx.shape).astype(float)       # noqa: E731
        self.actors[str(ident)] = {'cls': cls if cls in CLASSES else 'obstacle', 'idx': idx, 'x': f(x), 'y': f(y),
                                   'yaw': f(yaw), 'vx': f(vx), 'vy': f(vy), 'l': f(length), 'w': f(width)}

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t))) if len(self.t) > 1 else 0.05

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def distance_m(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y)))) if len(self.x) > 1 else 0.0


def class_of(row: dict) -> str:
    """An actors.jsonl row's Falcon class: its subclass (a stand-in's or proxy's) when it has one, else from its
    nuScenes category (core.groundtruth.category_of) and blueprint id."""
    sub = str(row.get('subclass') or row.get('kind') or '')
    if sub in CLASSES:
        return sub
    c, tid = str(row.get('category') or ''), str(row.get('type_id') or '')
    if c.startswith('human.pedestrian.child'):
        return 'child'
    if c.startswith(('human', 'walker', 'pedestrian')) or tid.startswith('walker.'):
        return 'pedestrian'
    if c.startswith('vehicle.bicycle'):
        return 'bicycle'
    if c.startswith('vehicle.motorcycle'):
        return 'motorcycle'
    if c.startswith('vehicle.bus'):
        return 'bus'
    if c.startswith('vehicle.truck'):
        return 'truck'
    if c.startswith('vehicle.') or tid.startswith('vehicle.'):
        return 'car'
    return 'obstacle'


def _read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _actor_classes(run_dir: str, cfg: dict) -> dict:
    """{actor id (as str): class} — actors.jsonl's rows (the first row of each actor: its static fields), and the
    scenario's own kinds for trajectories.json's names."""
    out = {}
    for a in ((cfg.get('scenario') or {}).get('actors') or ()) if isinstance(cfg.get('scenario'), dict) else ():
        if isinstance(a, dict) and a.get('id') is not None:
            out[str(a['id'])] = a.get('kind') if a.get('kind') in CLASSES else 'car'
    ap = os.path.join(run_dir, 'actors.jsonl')
    if os.path.isfile(ap):
        with open(ap, encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for a in row.get('actors') or ():
                    k = str(a.get('id'))
                    if k not in out or 'subclass' in a:
                        out.setdefault(k, class_of(a))
    return out


def _given_moments(run: dict, span_s: float) -> list:
    """Moments a run states itself: faults applied (run.json faults_applied) and a verified minimal-risk manoeuvre."""
    out = []

    def faults(v, sensor=''):
        if isinstance(v, dict):
            if any(k in v for k in ('kind', 'type', 'fault')) and not isinstance(v.get('fault'), (dict, list)):
                yield sensor, v
            else:
                for k, x in v.items():
                    yield from faults(x, sensor if k in ('planned', 'started', 'fault') else str(k))
        elif isinstance(v, list):
            for x in v:
                yield from faults(x, sensor)
    seen = set()
    for sensor, f in faults(run.get('faults_applied')):
        t0 = next((float(f[k]) for k in ('onset_s', 'start_s', 't_s', 'at_s') if isinstance(f.get(k), (int, float))),
                  0.0)
        dur = next((float(f[k]) for k in ('duration_s', 'for_s') if isinstance(f.get(k), (int, float))), None)
        t1 = min(span_s, t0 + dur) if dur is not None else span_s
        what = str(f.get('kind') or f.get('type') or f.get('fault') or 'fault')
        key = (sensor, what, round(t0, 2))
        if key in seen or t0 > span_s:
            continue
        seen.add(key)
        out.append({'type': 'sensor_fault', 't0': t0, 't1': max(t1, t0), 'actors': [], 'actor_ids': [],
                    'numbers': {'fault': what[:40], 'sensor': sensor[:40]}})
    m = run.get('mrm')
    if isinstance(m, dict) and m.get('onset_s') is not None:
        meas = m.get('measured') or {}
        t0 = float(m['onset_s'])
        end = meas.get('standstill_s') if isinstance(meas.get('standstill_s'), (int, float)) else None
        t1 = float(end) if end is not None else min(span_s, t0 + float(meas.get('window_s') or 0.0))
        out.append({'type': 'mrm', 't0': t0, 't1': max(t1, t0), 'actors': [], 'actor_ids': [],
                    'decel_mps2': meas.get('max_decel_mps2'),
                    'numbers': {'passed': bool(m.get('passed')), 'reaction_s': meas.get('reaction_s')}})
    return out


def scene_from_run(run_dir: str, xodr: str | None = None) -> Scene:
    """A recorded run's scene (module doc). xodr as core.rules.trace_from_run takes it: None finds the map's
    OpenDRIVE file, '' uses none. Raises MomentsError when the run has no recorded ego track."""
    from . import rules as R
    try:
        tr = R.trace_from_run(run_dir, xodr=xodr)
    except (R.RulesError, OSError, ValueError, KeyError) as e:
        raise MomentsError(str(e)) from None
    sc = Scene(tr.t, tr.x, tr.y, tr.yaw, tr.speed, tr.ego_size)
    sc.notes = list(tr.notes)
    cfg = R._run_config(run_dir)
    classes = _actor_classes(run_dir, cfg) if tr.actors is not None else {}
    acc: dict = {}
    for i, acts in enumerate(tr.actors or ()):
        for a in acts:
            acc.setdefault(str(a['id']), []).append((i, a['x'], a['y'], a['yaw'], a['vx'], a['vy'], a['l'], a['w'],
                                                     a.get('kind')))
    for aid, rows in acc.items():
        A = np.array([r[:8] for r in rows], float)
        cls = classes.get(aid) or ('pedestrian' if rows[0][8] == 'pedestrian' else 'car')
        sc.add_actor(aid, cls, A[:, 0].astype(int), A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5], A[:, 6], A[:, 7])
    if tr.map.has_lanes or tr.given:
        try:
            sig = R.signals(tr)
        except Exception as e:                    # noqa: BLE001 (a map that cannot be read: the map moments are left out)
            sig = {}
            sc.notes.append(f'map signals not computed ({type(e).__name__}): lane, junction and light moments left out')
        for k in ('on_lane', 'in_junction', 'on_breaker', 'tl_red', 'tl_distance_m', 'ped_on_crossing'):
            v = sig.get(k)
            if v is not None and len(v) == len(sc.t) and not np.all(np.isnan(v)):
                sc.sig[k] = np.asarray(v, float)
    run = _read_json(os.path.join(run_dir, 'run.json'), {}) or {}
    sc.given = _given_moments(run, sc.duration_s)
    return sc


def scene_from_log(path: str) -> Scene:
    """An imported log's scene (core.logreplay.model.Log): every track sampled at LOG_DT_S on the ego's span,
    velocities from central differences of the positions."""
    from .logreplay.model import Log, LogError
    try:
        log = Log.load(path)
    except (LogError, OSError, ValueError, KeyError) as e:
        raise MomentsError(f'{path}: {e}') from None
    t0, t1 = log.ego.span() if callable(getattr(log.ego, 'span', None)) else log.ego.span
    if t1 - t0 < 2 * LOG_DT_S:
        raise MomentsError(f'{path}: the ego\'s track is shorter than {2 * LOG_DT_S:g} s')
    tq = np.arange(t0, t1 + 1e-9, LOG_DT_S)
    ok, x, y, _z, yaw = log.ego.sample(tq)
    tq, x, y, yaw = tq[ok], x[ok], y[ok], yaw[ok]
    if len(tq) < 3:
        raise MomentsError(f'{path}: fewer than three ego samples')
    size = tuple(getattr(log.ego, 'size', None) or (4.9, 1.84, 1.5))
    sc = Scene(tq - tq[0], x, y, yaw, np.hypot(np.gradient(x, tq), np.gradient(y, tq)), size[:2])
    for a in log.actors:
        ok, ax, ay, _az, ayaw = a.sample(tq)
        idx = np.flatnonzero(ok)
        if len(idx) < 3:
            continue
        ta = tq[idx]
        s = tuple(getattr(a, 'size', None) or (4.6, 2.0, 1.5))
        sc.add_actor(getattr(a, 'id', None) or getattr(a, 'ident', len(sc.actors)), str(getattr(a, 'cls', 'car')), idx,
                     ax[idx], ay[idx], ayaw[idx], np.gradient(ax[idx], ta), np.gradient(ay[idx], ta), s[0], s[1])
    return sc


# ================================================================================================ extraction
def _spans(mask, t, min_s: float = 0.0, join_s: float = 0.0, max_gap: float | None = None) -> list:
    """Maximal runs of True as (first index, last index) pairs. Runs separated by less than join_s are joined; a
    run shorter than min_s is dropped; a gap in t longer than max_gap (the actor was absent) ends a run."""
    mask = np.asarray(mask, bool)
    n = len(mask)
    if not n or not mask.any():
        return []
    brk = np.zeros(n, bool)
    if max_gap is not None and n > 1:
        brk[1:] = np.diff(t) > max_gap
    out, i = [], 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1] and not brk[j + 1]:
            j += 1
        out.append([i, j])
        i = j + 1
    joined = [out[0]]
    for a, b in out[1:]:
        p = joined[-1]
        if join_s > 0 and t[a] - t[p[1]] <= join_s and not brk[p[1] + 1:a + 1].any():
            p[1] = b
        else:
            joined.append([a, b])
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    return [(a, b) for a, b in joined if t[b] - t[a] + dt >= min_s - 1e-9]


def _smooth(v, k: int):
    if k <= 1 or len(v) < k:
        return np.asarray(v, float)
    pad = k // 2
    vp = np.pad(np.asarray(v, float), (pad, k - 1 - pad), mode='edge')
    return np.convolve(vp, np.ones(k) / k, mode='valid')


def _fin(v, nd=3):
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


def _side(lat: float) -> str:
    return 'right' if lat > 0 else 'left'          # CARLA's frame: +lat is the ego's right


def severity(m: dict) -> float:
    """SEVERITY_NOTE."""
    if m['type'] == 'collision':
        return 1.0
    s = SEVERITY_BASE.get(m['type'], 0.1)
    if m.get('min_ttc_s') is not None:
        s = max(s, 0.9 * (1.0 - min(float(m['min_ttc_s']), SEVERITY_TTC_S) / SEVERITY_TTC_S))
    if m.get('decel_mps2') is not None:
        s = max(s, 0.9 * min(float(m['decel_mps2']), SEVERITY_DECEL) / SEVERITY_DECEL)
    return round(min(s, 0.9), 3)


def speed_band(kmh) -> str | None:
    if kmh is None:
        return None
    for hi, name in SPEED_BANDS:
        if kmh < hi:
            return name
    return 'high'


def extract(sc: Scene) -> list:
    """The moments of a scene (module doc), ordered by start time. Deterministic: the same tracks give the same
    list. Each: {'type', 't0', 't1', 'actors' [class], 'actor_ids', 'side', 'min_ttc_s', 'gap_m', 'rel_speed_kmh',
    'ego_speed_kmh', 'decel_mps2', 'numbers', 'road', 'speed_band', 'severity'}."""
    from . import criticality as CR
    n = len(sc.t)
    out = []
    if n < 3:
        return out
    t, dt = sc.t, sc.dt
    gap_max = 2.5 * dt
    le, we = sc.ego_size
    yaw = np.radians(sc.yaw)
    E = np.column_stack([t, sc.x, sc.y, sc.yaw, sc.speed * np.cos(yaw), sc.speed * np.sin(yaw), np.full(n, le),
                         np.full(n, we)])
    k_brake = max(1, int(round(BRAKE_HOLD_S / dt)))
    ego_acc = _smooth(np.gradient(sc.speed, t), k_brake)
    lead_gap = {}                                   # actor id -> its gap ahead in the corridor per sample (inf: none)

    def add(typ, i0, i1, ts, cls=None, aid=None, **kw):
        m = {'type': typ, 't0': round(float(ts[i0]), 3), 't1': round(float(ts[i1]), 3),
             'actors': [cls] if cls else [], 'actor_ids': [aid] if aid is not None else [], 'side': kw.pop('side', None),
             'min_ttc_s': _fin(kw.pop('min_ttc', None)), 'gap_m': _fin(kw.pop('gap', None), 2),
             'rel_speed_kmh': _fin(kw.pop('rel_kmh', None), 1), 'ego_speed_kmh': _fin(kw.pop('ego_kmh', None), 1),
             'decel_mps2': _fin(kw.pop('decel', None), 2), 'numbers': {k: v for k, v in kw.items() if v is not None}}
        out.append(m)
        return m

    def tmin(v, a, b):
        w = v[a:b + 1]
        w = w[np.isfinite(w)]
        return float(w.min()) if len(w) else None

    for aid, a in sc.actors.items():
        idx, cls = a['idx'], a['cls']
        ta = t[idx]
        Ei = E[idx]
        A = np.column_stack([ta, a['x'], a['y'], a['yaw'], a['vx'], a['vy'], a['l'], a['w']])
        c, s = np.cos(yaw[idx]), np.sin(yaw[idx])
        dx, dy = a['x'] - sc.x[idx], a['y'] - sc.y[idx]
        lon, lat = dx * c + dy * s, -dx * s + dy * c
        half_l, half_w = (le + a['l']) / 2.0, (we + a['w']) / 2.0
        gap = lon - half_l
        relh = (a['yaw'] - sc.yaw[idx] + 180.0) % 360.0 - 180.0
        va = np.hypot(a['vx'], a['vy'])
        va_lon, va_lat = a['vx'] * c + a['vy'] * s, -a['vx'] * s + a['vy'] * c
        ve = sc.speed[idx]
        closing = ve - va_lon
        rel_total = np.hypot(Ei[:, 4] - a['vx'], Ei[:, 5] - a['vy'])
        ttc = CR.ttc_rows(Ei, A, TTC_HORIZON_S)
        drac = CR.drac_rows(Ei, A)
        g_lon, g_lat, _ahead = CR.axis_gaps(Ei, A)
        dist = CR._distance(CR._corners_of(Ei), CR._corners_of(A))
        in_corr = np.abs(lat) < half_w + IN_PATH_MARGIN_M
        ahead = gap > 0
        outside = np.abs(lat) >= half_w + IN_PATH_MARGIN_M + CUT_OUTSIDE_M
        same = np.abs(relh) < SAME_DIR_DEG
        vehicle = cls in VEHICLES
        k_look = max(1, int(round(CUT_LOOKBACK_S / dt)))
        present = np.concatenate([[True], np.diff(ta) <= gap_max])       # False where the actor was absent just before
        lead = in_corr & ahead & (gap <= LEAD_RANGE_M)
        g = np.full(n, np.inf)
        g[idx[lead]] = gap[lead]
        lead_gap[aid] = (g, cls)
        common = dict(cls=cls, aid=aid)

        # ---- collision, near-miss, low TTC, high required deceleration
        hit = dist <= 0.0
        for i0, i1 in _spans(hit, ta, join_s=JOIN_S, max_gap=gap_max):
            add('collision', i0, i1, ta, side=_side(lat[i0]) if abs(lat[i0]) > half_w[i0] else None, gap=0.0,
                rel_kmh=rel_total[max(i0 - 1, 0)] * 3.6, ego_kmh=ve[max(i0 - 1, 0)] * 3.6, min_ttc=0.0, **common)
        free = ~hit
        if not hit.any():
            nm = free & ((ttc < NEAR_MISS_TTC_S) | ((dist < NEAR_MISS_DIST_M) & (rel_total >= NEAR_MISS_REL_MPS)))
            for i0, i1 in _spans(nm, ta, join_s=JOIN_S, max_gap=gap_max):
                k = i0 + int(np.argmin(dist[i0:i1 + 1]))
                add('near_miss', i0, i1, ta, side=_side(lat[k]) if abs(lat[k]) > half_w[k] else None,
                    min_ttc=tmin(ttc, i0, i1), gap=dist[k], rel_kmh=rel_total[k] * 3.6, ego_kmh=ve[i0] * 3.6,
                    **common)
        for i0, i1 in _spans(free & (ttc < LOW_TTC_S), ta, join_s=JOIN_S, max_gap=gap_max):
            k = i0 + int(np.argmin(ttc[i0:i1 + 1]))
            add('low_ttc', i0, i1, ta, min_ttc=ttc[k], gap=dist[k], rel_kmh=rel_total[k] * 3.6, ego_kmh=ve[i0] * 3.6,
                **common)
        hd = free & np.isfinite(drac) & (drac >= DRAC_HIGH_MPS2)
        for i0, i1 in _spans(hd, ta, join_s=JOIN_S, max_gap=gap_max):
            k = i0 + int(np.argmax(drac[i0:i1 + 1]))
            add('high_decel_needed', i0, i1, ta, decel=drac[k], min_ttc=tmin(ttc, i0, i1), gap=dist[k],
                rel_kmh=closing[k] * 3.6, ego_kmh=ve[i0] * 3.6, **common)

        # ---- close pass (anything alongside)
        cp = free & (g_lon <= 0.0) & (g_lat > 0.0) & (g_lat < CLOSE_PASS_M) & \
            (np.abs(closing) >= CLOSE_PASS_REL_MPS) & (np.maximum(ve, va) >= EGO_MOVING_MPS)
        for i0, i1 in _spans(cp, ta, join_s=JOIN_S, max_gap=gap_max):
            k = i0 + int(np.argmin(g_lat[i0:i1 + 1]))
            add('close_pass', i0, i1, ta, side=_side(lat[k]), gap=g_lat[k], rel_kmh=abs(closing[k]) * 3.6,
                ego_kmh=ve[k] * 3.6, lateral_clearance_m=_fin(g_lat[k], 2), **common)

        # ---- standing in the corridor ahead (any class)
        st = lead & (va < STOPPED_MPS)
        for i0, i1 in _spans(st, ta, min_s=STOPPED_S, join_s=JOIN_S, max_gap=gap_max):
            if ve[i0:i1 + 1].max() < EGO_MOVING_MPS:
                continue
            add('stopped_obstacle', i0, i1, ta, gap=gap[i0:i1 + 1].min(), min_ttc=tmin(ttc, i0, i1),
                rel_kmh=closing[i0] * 3.6, ego_kmh=ve[i0] * 3.6, **common)

        # ---- a pedestrian or animal crossing the corridor ahead
        if cls in WALKERS or cls in ANIMALS:
            cr = in_corr & ahead & (gap <= CROSS_RANGE_M) & (np.abs(va_lat) >= CROSS_LAT_MPS)
            for i0, i1 in _spans(cr, ta, join_s=JOIN_S, max_gap=gap_max):
                side = 'left' if np.median(va_lat[i0:i1 + 1]) > 0 else 'right'       # where it came from
                kw = dict(side=side, gap=gap[i0:i1 + 1].min(), min_ttc=tmin(ttc, i0, i1), ego_kmh=ve[i0] * 3.6,
                          rel_kmh=closing[i0] * 3.6, walk_mps=_fin(np.median(np.abs(va_lat[i0:i1 + 1])), 2))
                add('crossing', i0, i1, ta, **kw, **common)
                if cls in WALKERS:
                    w = slice(int(idx[i0]), int(idx[i1]) + 1)
                    poc, junc = sc.sig.get('ped_on_crossing'), sc.sig.get('in_junction')
                    basis = None
                    if poc is not None and np.isfinite(poc[w]).any():
                        basis = 'off every crosswalk of the map' if np.nanmax(poc[w]) < 0.5 else None
                    elif junc is not None and np.isfinite(junc[w]).any() and np.nanmax(junc[w]) < 0.5:
                        basis = 'not at a junction (the map\'s crosswalks are not known: a heuristic)'
                    if basis:
                        add('jaywalking', i0, i1, ta, basis=basis, **kw, **common)

        if not vehicle:
            continue
        # ---- oncoming inside the corridor
        on = lead & (np.abs(relh) >= OPPOSITE_DEG) & (va >= MOVING_MPS)
        for i0, i1 in _spans(on, ta, min_s=ONCOMING_S, join_s=JOIN_S, max_gap=gap_max):
            add('oncoming_in_lane', i0, i1, ta, gap=gap[i0:i1 + 1].min(), min_ttc=tmin(ttc, i0, i1),
                rel_kmh=closing[i0] * 3.6, ego_kmh=ve[i0] * 3.6, **common)

        # ---- the lead braking hard
        if len(ta) >= 3:
            acc = _smooth(np.gradient(va, ta), k_brake)
            hb = lead & same & (acc <= -HARD_BRAKE_MPS2)
            for i0, i1 in _spans(hb, ta, min_s=BRAKE_HOLD_S, join_s=JOIN_S, max_gap=gap_max):
                add('hard_brake_lead', i0, i1, ta, decel=-acc[i0:i1 + 1].min(), gap=gap[i0:i1 + 1].min(),
                    min_ttc=tmin(ttc, i0, i1 + k_look), rel_kmh=closing[i1] * 3.6, ego_kmh=ve[i0] * 3.6, **common)

        # ---- cut-in: from beside the corridor into it, ahead and close
        cin = in_corr & ahead & same & (gap <= CUT_GAP_MAX_M)
        k_hold = max(1, int(round(CUT_HOLD_S / dt)))
        k = 1
        while k < len(ta):
            if cin[k] and not cin[k - 1] and present[k]:
                lo = max(0, k - k_look)
                was = np.flatnonzero(outside[lo:k] & (lon[lo:k] > -half_l[lo:k]) & same[lo:k])
                end = k
                while end + 1 < len(ta) and cin[end + 1] and present[end + 1]:
                    end += 1
                if len(was) and present[lo + was[-1] + 1:k + 1].all() and end - k + 1 >= k_hold:
                    j = lo + int(was[-1])
                    e2 = min(end, k + max(1, int(round(CUT_TTC_WINDOW_S / dt))))
                    add('cut_in', j, e2, ta, side=_side(lat[j]), gap=gap[k], rel_kmh=closing[k] * 3.6,
                        ego_kmh=ve[k] * 3.6, min_ttc=tmin(ttc, k, e2), **common)
                k = end + 1
                continue
            k += 1

        # ---- cut-out: a lead leaving the corridor to the side
        ld = in_corr & ahead & same & (gap <= CUT_OUT_RANGE_M)
        k = 1
        while k < len(ta):
            if ld[k - 1] and not in_corr[k] and ahead[k] and present[k]:
                b = k - 1
                while b > 0 and ld[b - 1] and present[b]:
                    b -= 1
                hi = min(len(ta), k + k_look + 1)
                went = np.flatnonzero(outside[k:hi] & ahead[k:hi])
                if (k - b) >= k_hold and len(went) and present[k:k + went[0] + 1].all():
                    j = k + int(went[0])
                    add('cut_out', k - 1, j, ta, side=_side(lat[j]), gap=gap[k - 1], ego_kmh=ve[k - 1] * 3.6,
                        rel_kmh=closing[k - 1] * 3.6, **common)
                    k = j + 1
                    continue
            k += 1

        # ---- overtaking: from ahead to behind (the ego passes) or from behind to ahead (it passes the ego)
        zone = np.where(lon > half_l, 1, np.where(lon < -half_l, -1, 0))
        last, li = 0, 0
        for k in range(len(ta)):
            if not present[k]:
                last = 0
            z = int(zone[k])
            if z == 0:
                continue
            if last and z != last:
                w = slice(li, k + 1)
                if ta[k] - ta[li] <= OVERTAKE_MAX_S and np.abs(lat[w]).max() <= OVERTAKE_LAT_M and same[w].all() and \
                        min(va[w].min(), ve[w].min()) >= MOVING_MPS and not hit[w].any():
                    mid = li + int(np.argmin(np.abs(lon[w])))
                    add('overtake', li, k, ta, side=_side(lat[mid]), gap=g_lat[w].min(),
                        rel_kmh=abs(closing[mid]) * 3.6, ego_kmh=ve[mid] * 3.6, by='ego' if last == 1 else 'actor',
                        **common)
            last, li = z, k

    # ---- the ego braking hard (the nearest actor ahead in its corridor named, when there is one)
    for i0, i1 in _spans(ego_acc <= -HARD_BRAKE_MPS2, t, min_s=BRAKE_HOLD_S, join_s=JOIN_S):
        best = None
        for aid, (g, cls) in lead_gap.items():
            v = float(g[i0:i1 + 1].min())
            if math.isfinite(v) and (best is None or v < best[0]):
                best = (v, cls, aid)
        add('hard_brake_ego', i0, i1, t, cls=best[1] if best else None, aid=best[2] if best else None,
            decel=-ego_acc[i0:i1 + 1].min(), gap=best[0] if best else None, ego_kmh=sc.speed[i0] * 3.6)

    # ---- map moments
    on_lane = sc.sig.get('on_lane')
    if on_lane is not None:
        off = np.nan_to_num(on_lane, nan=1.0) < 0.5
        for i0, i1 in _spans(off & (sc.speed > RED_MIN_MPS), t, min_s=LANE_DEPART_S, join_s=JOIN_S):
            add('lane_departure', i0, i1, t, ego_kmh=sc.speed[i0] * 3.6)
    red, tld = sc.sig.get('tl_red'), sc.sig.get('tl_distance_m')
    if red is not None and tld is not None:
        past = (np.nan_to_num(red) > 0.5) & (np.nan_to_num(tld, nan=np.inf) <= 0.0) & (sc.speed > RED_MIN_MPS)
        for i0, i1 in _spans(past, t, join_s=JOIN_S):
            add('red_light_violation', i0, i1, t, ego_kmh=sc.speed[i0] * 3.6)

    for g in sc.given:
        m = dict({'side': None, 'min_ttc_s': None, 'gap_m': None, 'rel_speed_kmh': None, 'ego_speed_kmh': None,
                  'decel_mps2': None, 'numbers': {}}, **g)
        m['t0'], m['t1'] = round(float(m['t0']), 3), round(float(m['t1']), 3)
        out.append(m)

    junc, brk = sc.sig.get('in_junction'), sc.sig.get('on_breaker')
    for m in out:
        i0 = int(np.searchsorted(t, m['t0'] - 1e-6))
        i1 = max(i0, int(np.searchsorted(t, m['t1'] + 1e-6)) - 1)
        i0, i1 = min(i0, n - 1), min(i1, n - 1)
        if m['ego_speed_kmh'] is None:
            m['ego_speed_kmh'] = _fin(sc.speed[i0] * 3.6, 1)
        m['speed_band'] = speed_band(m['ego_speed_kmh'])
        road = None
        w = slice(i0, i1 + 1)
        if brk is not None and np.isfinite(brk[w]).any() and np.nanmax(brk[w]) > 0.5:
            road = 'speed_breaker'
        elif junc is not None and np.isfinite(junc[w]).any():
            road = 'junction' if np.nanmax(junc[w]) > 0.5 else 'road'
        m['road'] = road
        m['severity'] = severity(m)
    out.sort(key=lambda m: (m['t0'], TYPES.index(m['type']), ','.join(map(str, m['actor_ids']))))
    return out


# ================================================================================================ the store
def db_path(workspace: str) -> str:
    return os.path.join(workspace, FOLDER, DB_FILE)


_DDL = (
    'CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)',
    'CREATE TABLE IF NOT EXISTS sources (kind TEXT, path TEXT, PRIMARY KEY (kind, path))',
    'CREATE TABLE IF NOT EXISTS runs (path TEXT PRIMARY KEY, rel TEXT, source TEXT, kind TEXT, name TEXT, key TEXT, '
    'private INTEGER, duration_s REAL, distance_m REAL, world TEXT, tod TEXT, weather TEXT, rain REAL, fog REAL, '
    'started TEXT, tracks INTEGER, note TEXT)',
    'CREATE TABLE IF NOT EXISTS moments (id TEXT PRIMARY KEY, run TEXT, type TEXT, t0 REAL, t1 REAL, actors TEXT, '
    'actor_ids TEXT, side TEXT, min_ttc_s REAL, gap_m REAL, rel_speed_kmh REAL, ego_speed_kmh REAL, decel_mps2 REAL, '
    'severity REAL, road TEXT, speed_band TEXT, numbers TEXT)',
    'CREATE INDEX IF NOT EXISTS moments_type ON moments (type)',
    'CREATE INDEX IF NOT EXISTS moments_run ON moments (run)',
)


def _open(workspace: str, create: bool = True):
    p = db_path(workspace)
    if not os.path.isfile(p):
        if not create:
            return None
        os.makedirs(os.path.dirname(p), exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    for s in _DDL:
        con.execute(s)
    con.execute('INSERT OR REPLACE INTO meta VALUES (?, ?)', ('schema', SCHEMA))
    return con


def _private(path: str, entry: dict | None = None) -> bool:
    if entry and entry.get('dashcam_route'):
        return True
    try:
        from ..assistant.privacy import private_reason
        return bool(private_reason(path))
    except Exception:              # noqa: BLE001 (the check failing must not make a private run public)
        return True


def _key(paths) -> str:
    from .runindex import _stat
    return json.dumps([VERSION, _stat(paths)], separators=(',', ':'))


def moment_id(run_path: str, k: int) -> str:
    return f'{hashlib.sha1(os.path.abspath(run_path).encode()).hexdigest()[:8]}-{k}'


def _store(con, run: dict, moments: list):
    con.execute('DELETE FROM moments WHERE run = ?', (run['path'],))
    con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                tuple(run.get(k) for k in ('path', 'rel', 'source', 'kind', 'name', 'key', 'private', 'duration_s',
                                           'distance_m', 'world', 'tod', 'weather', 'rain', 'fog', 'started',
                                           'tracks', 'note')))
    for k, m in enumerate(moments):
        con.execute('INSERT INTO moments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (moment_id(run['path'], k), run['path'], m['type'], m['t0'], m['t1'], ','.join(m['actors']),
                     ','.join(str(x) for x in m['actor_ids']), m.get('side'), m.get('min_ttc_s'), m.get('gap_m'),
                     m.get('rel_speed_kmh'), m.get('ego_speed_kmh'), m.get('decel_mps2'), m.get('severity'),
                     m.get('road'), m.get('speed_band'), json.dumps(m.get('numbers') or {}, default=str)))


def log_files(workspace: str) -> list:
    d = os.path.join(workspace, 'logs')
    return [os.path.join(d, n, 'log.json') for n in (sorted(os.listdir(d)) if os.path.isdir(d) else ())
            if os.path.isfile(os.path.join(d, n, 'log.json'))]


def _ranking_file(path: str) -> str | None:
    """A near-miss ranking: the file itself, an ingest folder, or the footage folder holding falcon-ingest/."""
    for p in (path, os.path.join(path, 'ranking.json'), os.path.join(path, 'falcon-ingest', 'ranking.json')):
        if os.path.isfile(p) and p.endswith('.json'):
            return p
    return None


def _nearmiss_moments(doc: dict) -> list:
    """core.nearmiss events as moments: the class, the window in the clip (seconds), the measures. Nothing of the
    clip's name, date or position is taken (the event id is a hash)."""
    out = []
    for e in (doc or {}).get('events') or ():
        w = e.get('window_s') or [0.0, 0.0]
        cls = e.get('kind') if e.get('kind') in CLASSES else 'obstacle'
        m = {'type': 'collision' if e.get('overlap') else 'near_miss', 't0': float(w[0]), 't1': float(w[-1]),
             'actors': [cls], 'actor_ids': [str(e.get('id') or '')[:12]], 'side': None,
             'min_ttc_s': _fin(e.get('min_ttc_s')), 'gap_m': _fin(e.get('min_distance_m'), 2), 'rel_speed_kmh': None,
             'ego_speed_kmh': None, 'decel_mps2': _fin(e.get('max_drac_mps2'), 2), 'road': None, 'speed_band': None,
             'numbers': {'event': str(e.get('id') or '')[:12], 'criticality': _fin(e.get('score_median')),
                         'pet_s': _fin(e.get('pet_s')), 'ego_peak_decel_mps2': _fin(e.get('ego_peak_decel_mps2'))}}
        m['severity'] = severity(m) if not e.get('score_median') else round(
            max(severity(m), min(float(e['score_median']), 1.0)), 3)
        out.append(m)
    return out


def index(workspace: str, runs=(), nearmiss=(), force: bool = False, only=None, progress=None,
          xodr: str | None = None) -> dict:
    """Bring the workspace's store up to date (module doc): every run folder (core.runindex.run_dirs, plus `runs`),
    every imported log and every near-miss ranking registered (`nearmiss`: folders or ranking.json files, remembered
    in the store). only: index just these run folders (the job after a run). Unchanged sources are not read again;
    sources that are gone are removed. Returns {'sources', 'indexed', 'unchanged', 'removed', 'failed' [{path, why}],
    'moments', 'private_runs'}."""
    from . import runindex as RI
    if not workspace or not os.path.isdir(workspace):
        raise MomentsError(f'{workspace!r} is not a workspace folder')
    con = _open(workspace)
    res = {'sources': 0, 'indexed': 0, 'unchanged': 0, 'removed': 0, 'failed': []}
    try:
        for p in nearmiss:
            rf = _ranking_file(os.path.abspath(os.path.expanduser(p)))
            if rf is None:
                raise MomentsError(f'{p}: no near-miss ranking there (ranking.json; falcon dashcam ingest writes it)')
            con.execute('INSERT OR IGNORE INTO sources VALUES (?, ?)', ('nearmiss', rf))
        have = {r['path']: r['key'] for r in con.execute('SELECT path, key FROM runs')}
        todo = []                                                   # (source, path, key files)
        if only:
            for d in only:
                d = os.path.abspath(d)
                todo.append(('run', d, [os.path.join(d, f) for f in FILES]))
        else:
            for d in RI.run_dirs(workspace, [os.path.abspath(r) for r in runs]):
                todo.append(('run', d, [os.path.join(d, f) for f in FILES]))
            for lp in log_files(workspace):
                todo.append(('log', lp, [lp]))
            for r in con.execute('SELECT path FROM sources WHERE kind = ?', ('nearmiss',)).fetchall():
                if os.path.isfile(r['path']):
                    todo.append(('nearmiss', r['path'], [r['path']]))
                else:
                    con.execute('DELETE FROM sources WHERE kind = ? AND path = ?', ('nearmiss', r['path']))
        seen = set()
        for k, (source, path, files) in enumerate(todo):
            seen.add(path)
            res['sources'] += 1
            key = _key(files)
            if not force and have.get(path) == key:
                res['unchanged'] += 1
                continue
            if progress:
                progress(k, len(todo), path)
            row = {'path': path, 'rel': RI.key_of(path, workspace), 'source': source, 'key': key, 'kind': source,
                   'name': os.path.basename(os.path.dirname(path) if source != 'run' else path), 'private': 0,
                   'tracks': 0, 'note': ''}
            moments = []
            try:
                if source == 'run':
                    e = RI.summarise(path, workspace) or {}
                    row.update(kind=e.get('kind') or 'run', name=e.get('name') or row['name'],
                               world=e.get('world') or e.get('map'), tod=e.get('tod'), weather=e.get('weather_kind'),
                               rain=e.get('rain'), fog=e.get('fog'), started=e.get('started'),
                               private=int(_private(path, e)))
                    sc = scene_from_run(path, xodr)
                else:
                    row['private'] = int(_private(path)) if source == 'log' else 1
                    sc = scene_from_log(path) if source == 'log' else None
                if sc is not None:
                    moments = extract(sc)
                    row.update(duration_s=round(sc.duration_s, 3), distance_m=round(sc.distance_m, 2),
                               tracks=len(sc.actors), note='; '.join(sc.notes)[:500])
                else:
                    doc = _read_json(path, {}) or {}
                    moments = _nearmiss_moments(doc)
                    row.update(name='near-miss ranking', rel='(near-miss ranking)',
                               note='mined from dashcam footage (core.nearmiss): the events only, no driving time')
            except MomentsError as e:
                row['note'] = f'not indexed: {e}'[:500]
                res['failed'].append({'path': row['rel'], 'why': str(e)[:300]})
            with con:
                _store(con, row, moments)
            res['indexed'] += 1
        if not only:
            for path in set(have) - seen:
                with con:
                    con.execute('DELETE FROM moments WHERE run = ?', (path,))
                    con.execute('DELETE FROM runs WHERE path = ?', (path,))
                res['removed'] += 1
        con.commit()
        res['moments'] = con.execute('SELECT COUNT(*) FROM moments').fetchone()[0]
        res['private_runs'] = con.execute('SELECT COUNT(*) FROM runs WHERE private = 1').fetchone()[0]
    finally:
        con.close()
    return res


def index_job_spec(workspace: str, run_dir: str):
    """The cheap CPU job that indexes one finished run (`falcon moments index WORKSPACE --run RUN`)."""
    import sys
    from .jobs import JobSpec
    argv = [sys.executable, '-m', 'falcon_autonomy', 'moments', 'index', workspace, '--run', run_dir]
    return JobSpec('moments-index', f'Moments of {os.path.basename(os.path.normpath(run_dir))}', argv, memory_gb=0.5,
                   meta={'moments_of': run_dir})


def after_job(service, job):
    """For CarlaService._on_job (the setting moments_auto_index, off by default): queue the index of a run that
    finished well and recorded frames. None when there is nothing to index."""
    from . import protocol as P
    s = job.spec
    if job.exit_code != P.OK or not s.run_dir or s.kind == 'moments-index' or not service.workspace:
        return None
    if not os.path.isfile(os.path.join(s.run_dir, 'frames.jsonl')):
        return None
    return service.queue.submit(index_job_spec(service.workspace, s.run_dir))


def _rows(workspace: str, public_only: bool = False, types=None) -> list:
    con = _open(workspace, create=False)
    if con is None:
        return []
    try:
        q = ('SELECT m.*, r.rel, r.source, r.kind AS run_kind, r.name AS run_name, r.private, r.world, r.tod, '
             'r.weather, r.rain, r.fog, r.started FROM moments m JOIN runs r ON r.path = m.run')
        cond, args = [], []
        if public_only:
            cond.append('r.private = 0')
        if types:
            cond.append(f'm.type IN ({",".join("?" * len(types))})')
            args += list(types)
        if cond:
            q += ' WHERE ' + ' AND '.join(cond)
        out = []
        for r in con.execute(q, args):
            d = dict(r)
            d['actors'] = [x for x in (d['actors'] or '').split(',') if x]
            d['actor_ids'] = [x for x in (d['actor_ids'] or '').split(',') if x]
            d['numbers'] = json.loads(d['numbers'] or '{}')
            d['private'] = bool(d['private'])
            out.append(d)
        return out
    finally:
        con.close()


def get(workspace: str, mid: str) -> dict | None:
    return next((m for m in _rows(workspace) if m['id'] == mid), None)


# ================================================================================================ queries
FIELDS = {'ttc': ('min_ttc_s', 's', 'min TTC'), 'gap': ('gap_m', 'm', 'gap'),
          'speed': ('ego_speed_kmh', 'km/h', 'ego speed'), 'rel_speed': ('rel_speed_kmh', 'km/h', 'relative speed'),
          'decel': ('decel_mps2', 'm/s²', 'deceleration'), 'severity': ('severity', '', 'severity'),
          'duration': ('duration_s', 's', 'duration'), 'fog': ('fog', '', 'fog'), 'rain': ('rain', '', 'rain')}
SOFT = ('actors', 'side', 'tod', 'weather', 'world', 'road', 'source')
TODS = ('day', 'low sun', 'twilight', 'night')
WEATHERS = ('clear', 'cloudy', 'wet', 'rain')
ROADS = ('road', 'junction', 'speed_breaker')
SOURCES = ('run', 'log', 'nearmiss')


def _rx(p):
    return re.compile(rf'\b(?:{p})\b', re.I)


_S = r'[\s-]+'
TYPE_PHRASES = [
    ('cut_out', _rx(rf'cut(?:s|ting)?{_S}out|cut[\s-]?out|(?:pulls?|moves?|swerves?){_S}out{_S}of{_S}(?:the{_S}|our{_S}|my{_S})?(?:lane|way)')),
    ('close_pass', _rx(rf'close{_S}pass(?:es|ing)?|pass(?:es|ing|ed)?{_S}(?:too{_S}|very{_S})?close(?:ly)?|'
                       rf'(?:squeez(?:es|ing|e)|brush(?:es|ing)?|scrap(?:es|ing|e)){_S}(?:past|by)')),
    ('oncoming_in_lane', _rx(rf'on[\s-]?coming|head[\s-]?on')),
    ('red_light_violation', _rx(rf'(?:r[ua]n(?:s|ning)?|jump(?:s|ed|ing)?|cross(?:es|ed|ing)?|skip(?:s|ped|ping)?)'
                                rf'{_S}(?:a{_S}|the{_S})?red(?:{_S}(?:lights?|signals?))?|red{_S}(?:light|signal)'
                                rf'(?:{_S}(?:violations?|jump(?:ing)?))?|signal{_S}jump(?:ing)?')),
    ('low_ttc', _rx(rf'low{_S}(?:ttc|time{_S}to{_S}collision)')),
    ('high_decel_needed', _rx(rf'(?:high|large){_S}(?:required{_S})?(?:drac|decel(?:eration)?(?:{_S}(?:needed|required))?)|'
                              rf'drac|required{_S}decel(?:eration)?')),
    ('near_miss', _rx(rf'near[\s-]?miss(?:es)?|close{_S}calls?|narrow(?:ly)?{_S}(?:miss(?:es|ed)?|escapes?)')),
    ('lane_departure', _rx(rf'lane{_S}departures?|(?:leav(?:es|ing|e)|left|departs?|drift(?:s|ing)?{_S}out{_S}of|'
                           rf'r[ua]ns?{_S}off|goes{_S}off){_S}(?:the{_S}|its{_S}|our{_S})?(?:lane|road)|off[\s-]?road')),
    ('sensor_fault', _rx(rf'(?:sensor|camera|lidar|radar|gnss|gps|imu){_S}(?:faults?|failures?|drop[\s-]?outs?|'
                         rf'blackouts?)|faults?(?:{_S}(?:active|injected|injection))?|calibration{_S}drift')),
    ('mrm', _rx(rf'mrm|minimal{_S}risk(?:{_S}(?:manoeuvres?|maneuvers?|condition))?|safe{_S}stop|'
                rf'emergency{_S}pull[\s-]?over')),
    ('jaywalking', _rx(r'jaywalk(?:s|ing|ers?|ed)?')),
    ('stopped_obstacle', _rx(rf'stand(?:s|ing)|sit(?:s|ting)|lying|block(?:s|ing|ed)|obstacles?|obstruction')),
    ('collision', _rx(r'collid(?:es|ed|e|ing)|collisions?|crash(?:es|ed|ing)?|accidents?|rear[\s-]?end(?:s|ed|ing)?')),
    ('hard_brake_ego', _rx(rf'(?:ego|we|i|our{_S}car)(?:{_S}vehicle)?{_S}(?:had{_S}to{_S}|have{_S}to{_S})?'
                           rf'(?:brak(?:es|ed|e)|slam(?:s|med)?)(?:{_S}hard)?|ego{_S}(?:hard{_S})?braking')),
    ('hard_brake_lead', _rx(rf'lead(?:{_S}(?:vehicle|car))?{_S}(?:hard{_S})?brak(?:es|ed|e|ing)|'
                            rf'(?:vehicle|car|bus|truck|auto){_S}(?:ahead|in{_S}front){_S}brak(?:es|ed|e|ing)')),
]
GROUP_PHRASES = [
    ('two_wheeler', _rx(r'(?:two|2)[\s-]*wheelers?')),
    ('three_wheeler', _rx(r'(?:three|3)[\s-]*wheelers?')),
    ('animal', _rx(r'animals?|stray(?:s)?|dogs?|goats?')),
    ('vru', _rx(rf'vrus?|vulnerable{_S}road{_S}users?')),
]
SOURCE_PHRASES = [('log', _rx(rf'(?:imported{_S})?logs?|real{_S}(?:drives?|logs?|data)')),
                  ('nearmiss', _rx(rf'dash[\s-]?cam|near[\s-]?miss{_S}clips?|footage|clips?')),
                  ('run', _rx(r'simulat(?:ion|ed)(?:\s+runs?)?|sim(?:\s+runs?)?'))]
_NUM = re.compile(
    r'(?:(?P<field>ttc|time[\s-]+to[\s-]+collision|gap|distance|headway|ego[\s-]+speed|relative[\s-]+speed|'
    r'closing[\s-]+speed|rel[\s-]+speed|speed|decel(?:eration)?|braking|severity|duration|fog|rain)\s*(?:is|of|was)?\s*)?'
    r'(?P<op>under|below|less[\s-]+than|slower[\s-]+than|shorter[\s-]+than|<=|<|within|at[\s-]+most|up[\s-]+to|over|'
    r'above|more[\s-]+than|greater[\s-]+than|faster[\s-]+than|longer[\s-]+than|>=|>|at[\s-]+least|=)\s*'
    r'(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>km/?h|kmph|kph|m/s(?:\^?2|²)|m/s|seconds?|secs?|s|metres?|meters?|m)?(?![\w/])',
    re.I)
_SIDE = re.compile(r'\b(?:from|on|to|towards?|into)[\s-]+(?:the[\s-]+|your[\s-]+|our[\s-]+|its[\s-]+)?(left|right)\b|'
                   r'\b(left|right)[\s-]+(?:hand[\s-]+)?side\b', re.I)
_OPS = {'under': '<', 'below': '<', 'less than': '<', 'slower than': '<', 'shorter than': '<', '<': '<', '<=': '<=',
        'within': '<=', 'at most': '<=', 'up to': '<=', 'over': '>', 'above': '>', 'more than': '>',
        'greater than': '>', 'faster than': '>', 'longer than': '>', '>': '>', '>=': '>=', 'at least': '>=', '=': '='}
_FIELD_WORDS = {'ttc': 'ttc', 'time to collision': 'ttc', 'gap': 'gap', 'distance': 'gap', 'headway': 'gap',
                'ego speed': 'speed', 'speed': 'speed', 'relative speed': 'rel_speed', 'closing speed': 'rel_speed',
                'rel speed': 'rel_speed', 'decel': 'decel', 'deceleration': 'decel', 'braking': 'decel',
                'severity': 'severity', 'duration': 'duration', 'fog': 'fog', 'rain': 'rain'}
# words that carry no constraint in a query (beyond core.scenario_check.STOP)
FILLER = set('''find show list search get give all any every moment moments event events run runs drive drives case
cases scene scenes where when which that with and the from into during while time times happen happens happened
involving involves near around hard heavy light sudden suddenly front behind someone something anything vehicle
vehicles car cars traffic road lane ego kmh kmph kph seconds metres meters under below over above than less more
least most within'''.split())
_CK_TYPE = {'cut_in': ('cut_in',), 'brake': ('hard_brake_ego', 'hard_brake_lead'), 'cross': ('crossing',),
            'stopped': ('stopped_obstacle',), 'wrong_way': ('oncoming_in_lane',), 'overtake': ('overtake',)}
_CK_TOD = {'day': 'day', 'low_sun': 'low sun', 'twilight': 'twilight', 'night': 'night'}


def new_filter(**kw) -> dict:
    f = {'types': [], 'not_types': [], 'actors': [], 'side': None, 'tod': None, 'weather': None, 'world': None,
         'road': None, 'source': None, 'where': []}
    for k, v in kw.items():
        if k not in f:
            raise MomentsError(f'unknown filter field {k!r}')
        f[k] = v
    return f


def expand_actors(names) -> list:
    """Classes and groups (two_wheeler, vru, animal …) as classes; an unknown name raises."""
    out = []
    for a in names or ():
        a = str(a).strip().lower().replace('-', '_').replace(' ', '_')
        got = GROUPS.get(a) or ((a,) if a in CLASSES else None)
        if got is None:
            raise MomentsError(f'unknown actor class {a!r} (one of {", ".join(CLASSES + tuple(GROUPS))})')
        out += [c for c in got if c not in out]
    return out


def parse_where(term: str) -> tuple:
    """'ttc<1.5' -> ('ttc', '<', 1.5); the fields are FIELDS."""
    m = re.fullmatch(r'\s*([a-z_]+)\s*(<=|>=|<|>|=)\s*(-?\d+(?:\.\d+)?)\s*', str(term))
    if not m or m.group(1) not in FIELDS:
        raise MomentsError(f'{term!r}: write a constraint as FIELD<NUMBER, with FIELD one of {", ".join(FIELDS)}')
    return (m.group(1), m.group(2), float(m.group(3)))


def check_filter(f: dict) -> dict:
    """The filter normalised (classes expanded, names checked): raises MomentsError on an unknown value."""
    g = new_filter()
    for k in ('types', 'not_types'):
        for t in f.get(k) or ():
            t = str(t).strip().lower().replace('-', '_')
            if t not in TYPES:
                raise MomentsError(f'unknown moment type {t!r} (one of {", ".join(TYPES)})')
            if t not in g[k]:
                g[k].append(t)
    g['actors'] = expand_actors(f.get('actors'))
    for k, allowed in (('side', ('left', 'right')), ('tod', TODS), ('weather', WEATHERS), ('road', ROADS),
                       ('source', SOURCES)):
        v = f.get(k)
        if v:
            v = str(v).strip().lower().replace('_', ' ') if k == 'tod' else str(v).strip().lower()
            if v not in allowed:
                raise MomentsError(f'{k}: {v!r} is not one of {", ".join(allowed)}')
            g[k] = v
    g['world'] = str(f['world']).strip() if f.get('world') else None
    for w in f.get('where') or ():
        w = parse_where(w) if isinstance(w, str) else tuple(w)
        if len(w) != 3 or w[0] not in FIELDS or w[1] not in ('<', '<=', '>', '>=', '='):
            raise MomentsError(f'bad constraint {w!r}')
        g['where'].append((w[0], w[1], float(w[2])))
    return g


def understood(f: dict) -> str:
    """The filter in words, shown back to the user."""
    parts = []
    if f['types']:
        parts.append('type ' + ' or '.join(f['types']))
    if f['not_types']:
        parts.append('not ' + ', '.join(f['not_types']))
    if f['actors']:
        parts.append('actor ' + ' or '.join(f['actors']))
    for k, name in (('side', 'side'), ('tod', 'light'), ('weather', 'weather'), ('world', 'world'), ('road', 'road'),
                    ('source', 'source')):
        if f[k]:
            parts.append(f'{name} {f[k]}')
    for fld, op, v in f['where']:
        _c, unit, name = FIELDS[fld]
        parts.append(f'{name} {op} {v:g}{(" " + unit) if unit else ""}')
    return ' · '.join(parts) if parts else '(everything)'


def parse_query(text: str) -> dict:
    """Plain words -> {'filter', 'understood', 'not_understood' [words], 'recognised' [{'phrase', 'as'}]} (module
    doc). Deterministic and local: regular expressions over a fixed vocabulary, nothing else."""
    from . import scenario_check as CK
    text = str(text or '')
    f = new_filter()
    spans, recog = [], []

    def note(a, b, what):
        spans.append((a, b))
        recog.append({'phrase': text[a:b], 'as': what, 'span': [a, b]})

    def over(a, b, where=None):
        return any(a < y and x < b for x, y in (spans if where is None else where))

    # numbers first: "under 1.5 s" must not leave "under" for the words
    for m in _NUM.finditer(text):
        fw = re.sub(r'[\s-]+', ' ', (m.group('field') or '').lower())
        unit = (m.group('unit') or '').lower()
        val = float(m.group('num'))
        fld = _FIELD_WORDS.get(fw)
        if fld is None:
            fld = ('speed' if unit in ('km/h', 'kmh', 'kmph', 'kph', 'm/s') else
                   'decel' if unit.startswith('m/s') else
                   'ttc' if unit in ('s', 'sec', 'secs', 'second', 'seconds') else
                   'gap' if unit in ('m', 'metre', 'metres', 'meter', 'meters') else None)
        if fld is None:
            continue                                # a bare number: left to the words not understood
        if fld in ('speed', 'rel_speed') and unit == 'm/s':
            val *= 3.6
        op = _OPS[re.sub(r'[\s-]+', ' ', m.group('op').lower())]
        f['where'].append((fld, op, round(val, 4)))
        note(m.start(), m.end(), f'{FIELDS[fld][2]} {op} {val:g}')
    own = []
    m = CK.PHRASES['no_collision'].search(text)
    if m:                                           # "without a collision": before the collision phrases see it
        own.append((m.start(), m.end()))
        note(m.start(), m.end(), 'not collision')
        f['not_types'].append('collision')
    for typ, rx in TYPE_PHRASES:
        for m in rx.finditer(text):
            if over(m.start(), m.end()):
                continue
            own.append((m.start(), m.end()))
            note(m.start(), m.end(), f'type {typ}')
            if typ not in f['types']:
                f['types'].append(typ)
    groups = []
    for g, rx in GROUP_PHRASES:
        for m in rx.finditer(text):
            if over(m.start(), m.end(), own):
                continue
            groups.append((m.start(), m.end()))
            note(m.start(), m.end(), f'actor {g}')
            f['actors'] += [c for c in GROUPS[g] if c not in f['actors']]
    for src, rx in SOURCE_PHRASES:
        m = rx.search(text)
        if m and not over(m.start(), m.end()) and f['source'] is None:
            f['source'] = src
            note(m.start(), m.end(), f'source {src}')
    ck = CK.extract(text)
    ego_brake = 'hard_brake_ego' in f['types']
    lead_brake = 'hard_brake_lead' in f['types']
    for c in ck['claims']:
        a, b = c['span']
        if c['type'] == 'actor':
            if over(a, b, groups) or over(a, b, own):
                continue                              # "two wheeler" is the group; "jaywalker" the type
            if c['kind'] in CLASSES and c['kind'] not in f['actors']:
                f['actors'].append(c['kind'])
            note(a, b, f'actor {c["kind"]}')
        elif c['type'] == 'manoeuvre':
            if over(a, b, own):
                continue
            ts = _CK_TYPE.get(c['what'], ())
            if c['what'] == 'brake':
                ts = () if (ego_brake or lead_brake) else (('hard_brake_lead',) if c.get('kind') else ts)
            for t in ts:
                if t not in f['types']:
                    f['types'].append(t)
            if c.get('side') and f['side'] is None:
                f['side'] = c['side']
            note(a, b, f'type {" or ".join(ts) or c["what"]}')
        elif c['type'] == 'location' and c.get('feature') in ('junction', 'speed_breaker'):
            if not over(a, b, own):
                f['road'] = c['feature']
                note(a, b, f'road {c["feature"]}')
        elif c['type'] == 'light':
            f['tod'] = _CK_TOD[c['class']]
            note(a, b, f'light {f["tod"]}')
        elif c['type'] == 'weather':
            if c['class'] in WEATHERS:
                f['weather'] = c['class']
                note(a, b, f'weather {c["class"]}')
            elif c['class'] == 'fog':
                f['where'].append(('fog', '>', 0.0))
                note(a, b, 'fog > 0')
        elif c['type'] == 'outcome':
            if over(a, b, own) or 'collision' in f['not_types']:
                continue
            if c.get('collision'):
                if 'collision' not in f['types']:
                    f['types'].append('collision')
                note(a, b, 'type collision')
            else:
                f['not_types'].append('collision')
                f['types'] = [t for t in f['types'] if t != 'collision']
                note(a, b, 'not collision')
    if f['side'] is None:
        m = _SIDE.search(text)
        if m:
            f['side'] = (m.group(1) or m.group(2)).lower()
            note(m.start(), m.end(), f'side {f["side"]}')
    for r in ck['recognised']:                      # every span the phrase set used (sides, repeated words)
        a, b = r['span']
        if r['as'].startswith(('side', 'distance')):
            spans.append((a, b))
    un = []
    for m in re.finditer(r"[A-Za-z][A-Za-z'/-]*|\d+(?:\.\d+)?", text):
        w = m.group(0).lower()
        if over(m.start(), m.end()) or w in CK.STOP or w in FILLER or (len(w) < 3 and not w[0].isdigit()):
            continue
        if w not in un:
            un.append(w)
    recog.sort(key=lambda r: r['span'])
    return {'filter': f, 'understood': understood(f), 'not_understood': un,
            'recognised': [{'phrase': r['phrase'], 'as': r['as']} for r in recog]}


def _cmp(v, op, x) -> bool:
    if v is None:
        return False
    v = float(v)
    return {'<': v < x, '<=': v <= x, '>': v > x, '>=': v >= x, '=': abs(v - x) < 1e-9}[op]


def match(m: dict, f: dict) -> tuple:
    """(share of the scored terms matched in 0–1, or None when a required term fails; the scored terms missed)."""
    if f['types'] and m['type'] not in f['types']:
        return None, []
    if m['type'] in f['not_types']:
        return None, []
    for fld, op, x in f['where']:
        col = FIELDS[fld][0]
        v = (m['t1'] - m['t0']) if fld == 'duration' else m.get(col)
        if not _cmp(v, op, x):
            return None, []
    total, missed = 0, []
    for k in SOFT:
        want = f.get(k)
        if not want:
            continue
        total += 1
        if k == 'actors':
            ok = any(a in want for a in m['actors'])
        elif k == 'world':
            ok = str(want).lower() in str(m.get('world') or '').lower()
        else:
            ok = m.get(k) == want
        if not ok:
            missed.append(k)
    return (1.0 if not total else (total - len(missed)) / total), missed


def search(workspace: str, flt: dict | None = None, limit: int = 50, strict: bool = False,
           public_only: bool = False) -> dict:
    """Hits for a filter (module doc): {'hits' [moment + 'match', 'missed'], 'total', 'full', 'private_left_out'}.
    strict: only hits matching every term. public_only: private runs' moments are counted, not returned."""
    f = check_filter(flt or new_filter())
    hits, hidden = [], 0
    for m in _rows(workspace, types=f['types'] or None):
        share, missed = match(m, f)
        if share is None or (strict and missed):
            continue
        if public_only and m['private']:
            hidden += share == 1.0
            continue
        m['match'], m['missed'] = round(share, 3), missed
        hits.append(m)
    hits.sort(key=lambda m: (-m['match'], -(m['severity'] or 0.0), m['rel'], m['t0'], m['id']))
    return {'filter': f, 'understood': understood(f), 'total': len(hits),
            'full': sum(1 for m in hits if m['match'] == 1.0), 'hits': hits[:int(limit)] if limit else hits,
            'private_left_out': int(hidden)}


# ================================================================================================ exposure
def poisson_interval(n: int, z: float = 1.96) -> tuple:
    """An approximate 95 % interval for a Poisson count (Byar's approximation; recalled, not checked against the
    document — within a few per cent of the exact interval for n ≥ 1; for n = 0 the upper bound is −ln 0.025)."""
    lo = 0.0 if n <= 0 else n * (1.0 - 1.0 / (9.0 * n) - z / (3.0 * math.sqrt(n))) ** 3
    hi = 3.689 if n <= 0 else (n + 1) * (1.0 - 1.0 / (9.0 * (n + 1)) + z / (3.0 * math.sqrt(n + 1))) ** 3
    return max(lo, 0.0), hi


def stats(workspace: str, public_only: bool = False, by: str | None = None) -> dict:
    """The exposure table: per moment type the count, the rate per hour of indexed driving and per 100 km, each with
    an approximate 95 % Poisson interval. The driving time is the recorded time of the runs and logs that have
    tracks (near-miss rankings add events but no time: their events are counted apart, in 'without_exposure').
    by: 'tod', 'weather', 'world' or 'source' splits the table."""
    con = _open(workspace, create=False)
    out = {'schema': SCHEMA, 'hours': 0.0, 'km': 0.0, 'runs': 0, 'rows': [], 'without_exposure': {},
           'private_left_out': 0, 'by': by, 'note': 'rates are per hour of the driving indexed here (mostly '
           'simulated scenarios chosen to be eventful): they describe this workspace, not real traffic'}
    if con is None:
        return out
    if by not in (None, 'tod', 'weather', 'world', 'source'):
        raise MomentsError(f'--by {by!r}: one of tod, weather, world, source')
    try:
        runs = [dict(r) for r in con.execute('SELECT * FROM runs')]
    finally:
        con.close()
    if public_only:
        out['private_left_out'] = sum(1 for r in runs if r['private'])
        runs = [r for r in runs if not r['private']]
    exposure, key_of = {}, {}
    for r in runs:
        g = str(r.get(by) or 'not recorded') if by else ''
        key_of[r['path']] = (g, r['source'] != 'nearmiss' and bool(r.get('duration_s')))
        if key_of[r['path']][1]:
            e = exposure.setdefault(g, [0.0, 0.0, 0])
            e[0] += float(r['duration_s'] or 0.0) / 3600.0
            e[1] += float(r['distance_m'] or 0.0) / 1000.0
            e[2] += 1
    counts = {}
    for m in _rows(workspace, public_only=public_only):
        g, has = key_of.get(m['run'], ('', False))
        if not has:
            out['without_exposure'][m['type']] = out['without_exposure'].get(m['type'], 0) + 1
            continue
        counts[(g, m['type'])] = counts.get((g, m['type']), 0) + 1
    out['hours'] = round(sum(e[0] for e in exposure.values()), 6)
    out['km'] = round(sum(e[1] for e in exposure.values()), 4)
    out['runs'] = sum(e[2] for e in exposure.values())
    for g in sorted(exposure):
        h, km, _n = exposure[g]
        for typ in TYPES:
            n = counts.get((g, typ), 0)
            if not n and by:
                continue
            lo, hi = poisson_interval(n)
            row = {'type': typ, 'count': n, 'per_hour': round(n / h, 4) if h > 0 else None,
                   'per_hour_ci': [round(lo / h, 4), round(hi / h, 4)] if h > 0 else None,
                   'per_100km': round(100.0 * n / km, 4) if km > 0 else None, 'hours': round(h, 6)}
            if by:
                row[by] = g
            out['rows'].append(row)
    return out


# ================================================================================================ a hit as a scenario
def make_scenario(workspace: str, moment: dict, name: str, world: str = '', pad_s: float = 2.0) -> dict:
    """The moment as a scenario through core.incident's path: its window (± pad_s) of the run's tracks as an incident
    document, core.incident.fit's parametric placement, the replay and variations scenarios and OpenSCENARIO, written
    to <workspace>/moments/scenarios/<name>/ (moment.json, scenario.json, variations.json, moment.xosc). A private
    run's scenario keeps core.incident's dashcam source and marker, so it stays private. Returns {'dir', 'files',
    'actors', 'private', 'parameters'}."""
    from . import incident as IN
    from . import scenarios as SC
    from .openscenario import export_xosc
    from .provenance import utc_now, write_json
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name or ''):
        raise MomentsError(f'a scenario name is letters, digits, _ . - (not {name!r})')
    if moment.get('source') == 'nearmiss':
        raise MomentsError('a near-miss clip\'s event becomes a scenario with `falcon dashcam nearmiss scenario` (it '
                           'needs the reconstruction, which the index does not hold)')
    path = moment['run']
    sc = scene_from_log(path) if moment.get('source') == 'log' else scene_from_run(path)
    lo, hi = max(0.0, moment['t0'] - pad_s), min(sc.duration_s, moment['t1'] + pad_s)
    keep = np.flatnonzero((sc.t >= lo - 1e-9) & (sc.t <= hi + 1e-9))
    if len(keep) < 3:
        raise MomentsError('the moment\'s window holds fewer than three samples')
    tt = sc.t[keep] - sc.t[keep[0]]
    cfg = {}
    if moment.get('source') != 'log':
        from .rules import _run_config
        cfg = _run_config(path)
    lht = str(cfg.get('handedness') or (cfg.get('scenario') or {}).get('handedness') or 'RHT').upper() == 'LHT'
    lanes = _read_json(os.path.join(path, 'lanes.json'), {}) if os.path.isdir(path) else {}
    widths = [p[4] for p in (lanes or {}).get('points') or () if len(p) > 4 and p[4]]
    actors, order = [], list(moment.get('actor_ids') or [])
    for aid, a in sc.actors.items():                 # the moment's actors first, then whoever else was near
        if aid not in order:
            order.append(aid)
    pos = {int(k): j for j, k in enumerate(keep)}
    for aid in order:
        a = sc.actors.get(str(aid))
        if a is None or a['cls'] not in SC.KINDS or len(actors) >= 8:
            continue
        sel = [j for j, i in enumerate(a['idx']) if int(i) in pos]
        if len(sel) < 2:
            continue
        j = np.array([pos[int(a['idx'][s])] for s in sel])
        near = np.hypot(a['x'][sel] - sc.x[keep][j], a['y'][sel] - sc.y[keep][j]).min()
        if str(aid) not in map(str, moment.get('actor_ids') or ()) and near > IN.MAX_RANGE_M:
            continue
        actors.append({'id': f'a{len(actors) + 1}', 'kind': a['cls'],
                       'track': {'t': tt[j].round(3).tolist(), 'x': a['x'][sel].round(3).tolist(),
                                 'y': a['y'][sel].round(3).tolist(), 'yaw': np.round(a['yaw'][sel], 2).tolist()},
                       'speed_mps': np.hypot(a['vx'][sel], a['vy'][sel]).round(3).tolist(), 'boxes': len(sel)})
    route = np.c_[sc.x[keep], sc.y[keep]]
    kr = [0]
    for i in range(1, len(route)):
        if np.hypot(*(route[i] - route[kr[-1]])) >= 0.5:
            kr.append(i)
    private = bool(moment.get('private'))
    inc = {'schema': IN.SCHEMA, 'source': dict(IN.SOURCE) if private else {'format': 'falcon-run', 'kind': 'moment'},
           'world': world or moment.get('world') or 'unknown', 'handedness': 'LHT' if lht else 'RHT',
           'kerb_sign': -1.0 if lht else 1.0, 'lane_width_m': round(float(np.median(widths)), 3) if widths else 3.5,
           'window_s': [round(lo, 3), round(hi, 3)], 'duration_s': round(hi - lo, 3),
           'ego': {'track': {'t': tt.round(3).tolist(), 'x': sc.x[keep].round(3).tolist(),
                             'y': sc.y[keep].round(3).tolist(), 'yaw': np.round(sc.yaw[keep], 2).tolist()},
                   'speed_mps': sc.speed[keep].round(3).tolist(), 'offset_m': [None] * len(keep)},
           'route_xy': route[kr].round(3).tolist() if len(kr) >= 2 else route.round(3).tolist(),
           'match': {'matched_share': 1.0, 'lane_changes': 0, 'median_offset_m': None},
           'actors': actors, 'detections': {'frames': 0, 'boxes_used': 0, 'tracks': len(actors)},
           'moment': {k: moment.get(k) for k in ('id', 'type', 't0', 't1', 'actors', 'side', 'min_ttc_s', 'gap_m')},
           'method': 'core.moments.make_scenario: the run\'s recorded tracks over the moment\'s window, placed and '
                     'parametrised by core.incident.fit'}
    try:
        inc['fit'] = IN.fit(inc)
        rp, vr = IN.scenarios(inc, name)
    except (IN.IncidentError, ValueError, KeyError, IndexError) as e:
        raise MomentsError(f'the moment could not be made a scenario: {e}') from None
    out_dir = os.path.join(workspace, FOLDER, 'scenarios', name)
    os.makedirs(out_dir, exist_ok=True)
    if not private:
        what = f'{NAMES.get(moment["type"], moment["type"])} at {moment["t0"]:g} s of {moment.get("rel")}'
        for d, how in ((rp, 'replayed: the ego and every road user on their recorded tracks'),
                       (vr, 'varied: each extracted value the centre of a uniform range')):
            d['source'] = {'format': 'falcon-run', 'kind': 'moment', 'run': moment.get('rel'), 'moment': moment['id']}
            d['about'] = f'A recorded moment ({what}) {how} (core.moments, core.incident).'
        for d in (rp, vr):
            errs = SC.validate(d)
            if errs:
                raise MomentsError('the scenario does not validate: ' + '; '.join(errs))
    write_json(os.path.join(out_dir, 'moment.json'), inc)
    write_json(os.path.join(out_dir, 'scenario.json'), rp)
    write_json(os.path.join(out_dir, 'variations.json'), vr)
    with open(os.path.join(out_dir, 'moment.xosc'), 'w', encoding='utf-8') as fh:
        fh.write(export_xosc(SC.concrete(rp, {})))
    if private:                                   # core.incident's marker: assistant.privacy reads it
        write_json(os.path.join(out_dir, 'run.json'), {'worker': 'dashcam-incident', 'status': 'ok', 'name': name,
                                                       'created': utc_now(), 'source': dict(IN.SOURCE)})
    return {'dir': out_dir, 'files': ['moment.json', 'scenario.json', 'variations.json', 'moment.xosc'],
            'actors': len(actors), 'private': private, 'parameters': len(vr['params'])}


def drive_set_edit(doc: dict, workspace: str, name: str, hits: list):
    """(the project Edit adding a drive set of the hits' runs, the set): core.runindex.drive_set_edit."""
    from .runindex import drive_set_edit as dse
    runs = []
    for m in hits:
        if m.get('source') == 'run' and m['run'] not in runs:
            runs.append(m['run'])
    return dse(doc, workspace, name, 'runs', runs)


# ================================================================================================ the command line
def describe(m: dict) -> str:
    who = '+'.join(m['actors']) or '—'
    ttc = f'{m["min_ttc_s"]:.2f} s' if m.get('min_ttc_s') is not None else '—'
    return (f'{m["id"]:12} {m["type"]:19} {who:14} {(m.get("side") or "—"):5} TTC {ttc:7} sev {m.get("severity") or 0:.2f}'
            f'  {m["t0"]:7.2f}–{m["t1"]:<7.2f} {"(private) " if m.get("private") else ""}{m.get("rel")}')


def _ws(arg: str) -> str:
    from .paths import PROJECT_SUFFIX, workspace_for
    return workspace_for(arg) if arg.endswith(PROJECT_SUFFIX) else arg


def filter_from_args(a) -> dict:
    f = new_filter(types=list(a.type or ()), actors=list(a.actor or ()), side=a.side, tod=a.tod, weather=a.weather,
                   world=a.world, road=a.road, source=a.source, where=[parse_where(w) for w in a.where or ()])
    for tod in ('night', 'day', 'twilight'):
        if getattr(a, tod, False):
            f['tod'] = tod
    for w in ('rain', 'wet', 'clear', 'cloudy'):
        if getattr(a, w, False):
            f['weather'] = w
    return f


def merge_filters(a: dict, b: dict) -> dict:
    """b's set fields over a's (the structured flags over the words)."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in a.items()}
    for k, v in b.items():
        if isinstance(v, list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        elif v:
            out[k] = v
    return out


def main_moments(argv) -> int:
    """falcon moments index|search|show|stats (docs/moments.md)."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon moments', description='What happened in every recorded run, searched in '
                                                                    'plain words (docs/moments.md).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    ws_help = 'a project file (*.falcon.json) or its workspace folder'
    ix = sub.add_parser('index', help='read new or changed runs, imported logs and near-miss rankings into the store')
    ix.add_argument('workspace', help=ws_help)
    ix.add_argument('--run', action='append', default=[], help='index only this run folder (repeatable)')
    ix.add_argument('--nearmiss', action='append', default=[], metavar='DIR',
                    help='a near-miss ranking to include from now on (an ingested footage folder, its falcon-ingest '
                         'folder or ranking.json); its moments are private')
    ix.add_argument('--force', action='store_true', help='read every source again')
    ix.add_argument('--no-map', action='store_true', help='do not look for the map\'s OpenDRIVE file (lanes.json only)')
    ix.add_argument('--json', action='store_true')
    se = sub.add_parser('search', help='moments matching words ("two-wheeler cuts in from the left at night in rain") '
                                       'and/or flags')
    se.add_argument('workspace', help=ws_help)
    se.add_argument('query', nargs='*', help='plain words (a fixed vocabulary; what is not understood is listed)')
    se.add_argument('--type', action='append', choices=TYPES, metavar='TYPE', help=', '.join(TYPES))
    se.add_argument('--actor', action='append', metavar='CLASS',
                    help='a class or group: ' + ', '.join(CLASSES + tuple(GROUPS)))
    se.add_argument('--side', choices=('left', 'right'))
    se.add_argument('--tod', choices=[t.replace(' ', '_') for t in TODS])
    se.add_argument('--weather', choices=WEATHERS)
    se.add_argument('--world')
    se.add_argument('--road', choices=ROADS)
    se.add_argument('--source', choices=SOURCES)
    se.add_argument('--where', action='append', metavar='FIELD<N',
                    help='a numeric constraint, e.g. ttc<1.5 or speed>40 (fields: ' + ', '.join(FIELDS) + ')')
    for flag in ('night', 'day', 'twilight', 'rain', 'wet', 'clear', 'cloudy'):
        se.add_argument('--' + flag, action='store_true')
    se.add_argument('--strict', action='store_true', help='only hits matching every term (no partial matches)')
    se.add_argument('--limit', type=int, default=20)
    se.add_argument('--json', action='store_true')
    sh = sub.add_parser('show', help='one moment in full; --scenario NAME makes it a scenario')
    sh.add_argument('workspace', help=ws_help)
    sh.add_argument('id')
    sh.add_argument('--scenario', metavar='NAME', help='write it as a scenario and OpenSCENARIO '
                                                       '(moments/scenarios/NAME/)')
    sh.add_argument('--world', default='', help='the project world id the scenario is placed in (default: the run\'s)')
    sh.add_argument('--pad', type=float, default=2.0, help='seconds kept before and after the moment')
    sh.add_argument('--json', action='store_true')
    stp = sub.add_parser('stats', help='how often each moment type occurs per hour of indexed driving')
    stp.add_argument('workspace', help=ws_help)
    stp.add_argument('--by', choices=('tod', 'weather', 'world', 'source'))
    stp.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    ws = _ws(a.workspace)
    try:
        if a.cmd == 'index':
            res = index(ws, nearmiss=a.nearmiss, force=a.force, only=a.run or None, xodr='' if a.no_map else None)
            if a.json:
                print(json.dumps(res, indent=1))
            else:
                print(f'{res["sources"]} source(s): {res["indexed"]} read, {res["unchanged"]} unchanged, '
                      f'{res["removed"]} removed; {res["moments"]} moment(s) in the store '
                      f'({res["private_runs"]} private source(s))')
                for f in res['failed']:
                    print(f'  not indexed: {f["path"]}: {f["why"]}')
            return 0
        if a.cmd == 'search':
            pq = parse_query(' '.join(a.query))
            f = merge_filters(pq['filter'], check_filter(filter_from_args(a)))
            res = search(ws, f, a.limit, a.strict)
            if a.json:
                print(json.dumps(dict(res, not_understood=pq['not_understood']), indent=1, default=str))
                return 0
            print(f'understood as: {res["understood"]}')
            if pq['not_understood']:
                print(f'not understood: {", ".join(pq["not_understood"])}')
            for m in res['hits']:
                extra = '' if m['match'] == 1.0 else f'   [partial: not {", ".join(m["missed"])}]'
                print(describe(m) + extra)
            print(f'{res["total"]} moment(s), {res["full"]} matching every term'
                  + ('' if os.path.isfile(db_path(ws)) else ' — no index yet: falcon moments index'))
            return 0
        if a.cmd == 'show':
            m = get(ws, a.id)
            if m is None:
                print(f'falcon moments show: no moment {a.id!r} (falcon moments search lists them)')
                return 2
            if a.scenario:
                res = make_scenario(ws, m, a.scenario, a.world, a.pad)
                print(json.dumps(res, indent=1) if a.json else
                      f'{res["dir"]}: {", ".join(res["files"])} ({res["actors"]} actor(s), {res["parameters"]} '
                      f'search parameter(s){", private" if res["private"] else ""})')
                return 0
            if a.json:
                print(json.dumps(m, indent=1, default=str))
                return 0
            print(describe(m))
            for k in ('gap_m', 'rel_speed_kmh', 'ego_speed_kmh', 'decel_mps2', 'road', 'speed_band', 'tod', 'weather',
                      'world', 'source'):
                if m.get(k) is not None:
                    print(f'  {k}: {m[k]}')
            for k, v in m['numbers'].items():
                print(f'  {k}: {v}')
            print(f'  run: {m["run"]}\n  open: the Data page ▸ Browse shows the run; the Moments tab opens it at '
                  f'{m["t0"]:g} s')
            return 0
        res = stats(ws, by=a.by)
        if a.json:
            print(json.dumps(res, indent=1))
            return 0
        print(f'{res["hours"]:.4f} h and {res["km"]:.2f} km of driving in {res["runs"]} indexed source(s)')
        for r in res['rows']:
            if r['count'] or not a.by:
                ci = r['per_hour_ci']
                print(f'{(r.get(a.by) + "  ") if a.by else ""}{r["type"]:20} {r["count"]:5d}  '
                      + (f'{r["per_hour"]:9.2f} /h  [{ci[0]:.2f}, {ci[1]:.2f}]' if ci else '        — /h'))
        for typ, n in sorted(res['without_exposure'].items()):
            print(f'{typ:20} {n:5d}  from near-miss rankings (no driving time: not in the rates)')
        print(res['note'])
        return 0
    except MomentsError as e:
        print(f'falcon moments {a.cmd}: {e}')
        return 2
