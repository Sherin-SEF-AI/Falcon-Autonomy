"""Automatic root-cause attribution (numpy at import; the harness, the model, the service and the environment inside
the functions). docs/root-cause.md is the manual; `falcon why …` the command.

Three attributions, each written as rootcause.json (SCHEMA, with Falcon's provenance block) and run.json in its
folder, and reported by core.reports' "rootcause" kind:

1. counterfactual(case) — ONE failure of a scenario run (a perception miss, a failing case of a search, a critical
   run) attributed to factors by re-runs that change one factor at a time while everything else is held. Scenario
   runs in kinematic motion with no background traffic repeat bit for bit on the same server (docs/scenarios.md,
   measured), so a difference between a re-run and the case is the factor's doing; a control re-run of the case
   itself, unchanged, checks that (it must fail again, and its trajectories are compared with the case's). Factors
   (FACTORS) and how each is removed:
     occlusion   the scenario actors that hide the target in the failure frames (their boxes over its box, nearer)
                 are removed from the scenario (with their timeline actions; refused when a trigger depends on
                 them). An occluder that is part of the map or not labelled cannot be moved: not applicable.
     range       the target actor starts half its range at the failure moment nearer along its lane, so it is
                 about half as far then (relative triggers fire on distances, so their times shift with it).
     lighting    a sun below LOW_SUN_DEG is raised to DAY_SUN_DEG.
     fog         fog_density ≥ FOG_MIN to 0.
     rain        precipitation, puddles (precipitation_deposits) and wetness ≥ RAIN_MIN to 0.
     appearance  the target's blueprint swapped within its class (workers.director "variant": the next ones of its
                 kind's pool, adults with adults), SWAPS re-runs; it "fixes" the failure when any swap does.
     sensor_faults  the rig's injected sensor faults (core.faults) removed.
   A factor is PRESENT when the case has it (heuristic thresholds above, stated per row). Per present factor:
     necessary (but-for): removing it alone fixes the failure;
     sufficient (here): with every other present factor removed, keeping it alone still fails;
   plus an "all removed" run. Runs: control + one per present factor (two for appearance) + all removed + one per
   factor kept alone (when two or more are present). Every run is a drive under the CARLA lock, cache-keyed in
   <out>/runs/, and scored with the case's own failure test; a model scores re-runs as the case itself
   (harness.collect meta_run), so a seeded test model draws the same numbers for the same frame.
   Failure tests (case['failure']): "perception" (core.milsearch's failure of the target actor, or the run's chosen
   metric, ≥ threshold), "missed" (the target object not detected at the given step), "criticality" (≥ threshold),
   "both" (min of the two) and "scenario" (its pass criteria failed). WORDING says what it is: in this scenario, with
   this model, a single-case attribution, not a general cause.

2. static(evaluation) — CORRELATES of misses over a whole evaluation (its predictions.jsonl, no re-run): every scored
   ground-truth object is missed or found (core.milsearch.frame_hits); per attribute (range, occlusion, truncation,
   box height, class, the frame's weather and scenario values) the miss rate per quantile bin and Spearman's rank
   correlation with the miss, each with a 95 % percentile bootstrap interval over whole drives (over frames, said
   so, with one drive). Correlational: attributes move together (range and box height; fog and range in a fogged
   drive), so a correlation is not a cause.

3. episode(e) — a DRIVING failure (a Leaderboard infraction, core.leaderboard) attributed to the policy's input
   features: the episode is driven again by the policy in the closed loop (kinematic, no traffic: it repeats), once
   unchanged (the control: the same infraction must recur) and once per FEATURE_GROUPS group with that group of the
   observation replaced by its neutral value (the presence groups — lead, vulnerable road user, traffic light — by
   their "nothing there" sentinels of core.drivefeatures; the others held at their value at the episode's start).
   A group whose removal prevents the infraction is one the failing decision depended on. Policies read ground
   truth features, so weather and lighting cannot change them (said so); sufficiency is not tested for features (a
   policy blind to all but one input is not a driving policy).

Limits. A counterfactual changes what it names and anything that comes with it (a removed bus takes its shadow; a
nearer pedestrian meets the ego at another time); a fix by one swap of appearance is about those blueprints; a
re-run on a real server repeats the motion bit for bit but its images only to within a few levels (exposure, motion
blur), which can flip a borderline detection — the control says whether the failure repeated. Traffic-manager drives
(not scenarios) are not repeatable and are refused (the static attribution of their evaluation is offered).
"""
from __future__ import annotations

import copy
import json
import math
import os
import time

import numpy as np

from . import provenance as PV

SCHEMA = 'falcon.rootcause/1'
FAILURES = ('perception', 'missed', 'criticality', 'both', 'scenario')
FACTORS = ('occlusion', 'range', 'lighting', 'fog', 'rain', 'appearance', 'sensor_faults')
FACTOR_TITLES = {'occlusion': 'Occlusion', 'range': 'Range', 'lighting': 'Lighting (sun altitude)', 'fog': 'Fog',
                 'rain': 'Rain and wet road', 'appearance': 'Appearance (blueprint within class)',
                 'sensor_faults': 'Injected sensor faults'}
LOW_SUN_DEG = 15.0                 # heuristic: a sun below this is "low" (dusk, dawn, night)
DAY_SUN_DEG = 45.0
FOG_MIN = 5.0                      # heuristic: fog_density from which fog is a factor (presets carry 0–3)
RAIN_MIN = 5.0
NEAR_M = 10.0                      # heuristic: a target nearer than this is "already near": range is not a factor
OCCLUDED_MIN = 0.1                 # heuristic: a target hidden less than this share is not occluded
SWAPS = 2
WORDING = ('In this scenario, with this model: a single-case attribution, not a general cause. Each re-run changes '
           'one factor of this case and holds everything else; a factor whose removal fixes the failure is a '
           'but-for cause here, and may not be one in another case.')
STATIC_WORDING = ('Correlational: these attributes go with misses in this evaluation\'s frames. Attributes move '
                  'together (range and box height; fog and range in a fogged drive), so a correlation is not a cause; '
                  'the counterfactual attribution of a single case tests one.')
EPISODE_WORDING = ('In this episode, with this policy: a single-case attribution, not a general cause. Each re-run '
                   'hides one group of the policy\'s inputs and keeps everything else; the policy reads ground-truth '
                   'features, so weather and lighting cannot change what it sees.')


class RootCauseError(Exception):
    pass


class Refused(RootCauseError):
    """The case cannot be attributed by re-runs (not repeatable, not failing, …): exit code 6 from the command."""


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


# ================================================================================================ the case
def load_case(run_dir: str) -> tuple:
    """(run.json, drive config) of a finished scenario run; Refused when it cannot be re-run exactly."""
    run_dir = os.path.abspath(run_dir)
    try:
        run = PV.read_json(os.path.join(run_dir, 'run.json'))
        cfg = PV.read_json(os.path.join(run_dir, 'drive.json'))
    except (OSError, ValueError) as e:
        raise RootCauseError(f'{run_dir} is not a finished drive with its config: {e}') from None
    if run.get('status') != 'ok':
        raise RootCauseError(f'{run_dir}: the run did not finish ok')
    why = repeatable(cfg)
    if why:
        raise Refused(f'{os.path.basename(run_dir)} cannot be re-run exactly: {why}. The static attribution of its '
                      'evaluation (falcon why evaluation) is correlational but needs no re-run')
    if not run.get('scenario'):
        raise RootCauseError(f'{run_dir}: no scenario record in run.json')
    return run, cfg


def repeatable(cfg: dict) -> str | None:
    """None when a drive config repeats bit for bit on the same server (a kinematic scenario, no traffic, the ego not
    on CARLA's autopilot), else why not."""
    scn = cfg.get('scenario')
    if cfg.get('replay'):
        return 'a log replay is not attributed by re-runs in this release'
    if not scn:
        return 'it is not a scenario run: its traffic manager drives through CARLA\'s physics, which does not repeat'
    if str(scn.get('motion', cfg.get('motion', 'kinematic'))) != 'kinematic':
        return 'physics motion does not repeat in CARLA 0.9.16 (measured); only kinematic scenarios do'
    t = cfg.get('traffic') or {}
    if int(t.get('vehicles', 0) or 0) or int(t.get('walkers', 0) or 0):
        return 'background traffic makes a run non-repeatable'
    if (scn.get('ego') or {}).get('driver') == 'autopilot':
        return 'an ego on CARLA\'s autopilot keeps physics'
    return None


def _camera(cfg: dict, cam: str | None = None) -> dict | None:
    cams = [s for s in cfg['rig']['sensors'] if s.get('type') == 'camera.pinhole']
    return next((s for s in cams if s['id'] == cam), None) if cam else (cams[0] if cams else None)


def _labelled(cfg: dict, cam: str | None = None) -> bool:
    c = _camera(cfg, cam)
    return bool(c and 'boxes2d' in (c.get('labels') or []))


def _objects_by_step(run_dir: str, cam: str) -> dict:
    """{step: [objects]} of a run's camera labels (objects/<cam>/<step>.json)."""
    out = {}
    d = os.path.join(run_dir, 'objects', cam)
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith('.json'):
            out[int(f[:-5])] = PV.read_json(os.path.join(d, f)).get('objects', [])
    return out


def _fps(cfg: dict) -> float:
    return float(cfg.get('fps') or 20.0)


# ================================================================================================ failure tests
def score_case(case: dict, run_dir: str, scores: dict | None = None) -> dict:
    """{'value', 'fails', 'detail'} of a run under the case's failure test (module doc). scores: a perception score
    of this run already computed (core.milsearch.perception_score)."""
    run = PV.read_json(os.path.join(run_dir, 'run.json'))
    if run.get('status') != 'ok':
        return {'value': None, 'fails': None, 'detail': f'the run failed: {run.get("error")}'}
    scn = run.get('scenario') or {}
    thr = float(case.get('threshold', 0.5))
    crit = float((scn.get('criticality') or {}).get('criticality', 0.0))
    kind = case['failure']
    if kind == 'criticality':
        return {'value': _r(crit, 6), 'fails': crit >= thr, 'detail': f'criticality {crit:.3f}'}
    if kind == 'scenario':
        bad = [c['name'] for c in scn.get('checks') or () if not c.get('ok')]
        return {'value': 1.0 if bad else 0.0, 'fails': bool(bad),
                'detail': ('failed: ' + ', '.join(bad)) if bad else 'every pass criterion met'}
    if kind == 'missed':
        return _missed(case, run_dir, run, scores)
    sc = scores
    actor = case.get('actor')
    if actor:
        row = next((a for a in sc['actors'] if a['actor'] == actor), None)
        if row is None:
            return {'value': 0.0, 'fails': False, 'detail': f'no scenario actor {actor!r} in this run'}
        if not row.get('scored'):
            v, detail = 0.0, f'{actor}: {row.get("note") or "not scored"}'
        else:
            v = float(row['failure'])
            detail = (f'{actor}: window recall {row["window_recall"] if row["window_recall"] is not None else "—"}, '
                      f'lost lead {row["lost_lead"]:.2f}')
    else:
        v, detail = float(sc['value']), f'{sc["metric"]} {sc["value"]:.3f}'
    if kind == 'both':
        detail += f'; criticality {crit:.3f}'
        v = min(v, crit)
    return {'value': _r(v, 6), 'fails': v >= thr, 'detail': detail}


def _track_of(case: dict, run: dict) -> object:
    """The target's track id in this run: a scenario actor's CARLA id (it can differ between runs), else as given."""
    if case.get('actor'):
        return ((run.get('scenario') or {}).get('actor_ids') or {}).get(case['actor'])
    return case.get('track_id')


def _missed(case, run_dir, run, scores) -> dict:
    from . import milsearch as MS
    step = int(case['step'])
    tid = _track_of(case, run)
    fr = next((f for f in scores['frames'] if f['step'] == step), None) if scores else None
    if fr is None:
        return {'value': 0.0, 'fails': False, 'detail': f'no scored frame at step {step}'}
    hits = MS.frame_hits(fr, float(case.get('iou', 0.5)), float(case.get('min_score', 0.5)))
    for g, h in zip(fr['gt'], hits):
        if g.get('track_id') == tid:
            if g.get('ignore'):
                return {'value': 0.0, 'fails': False, 'detail': 'hidden or too small to be scored there'}
            return {'value': 0.0 if h else 1.0, 'fails': not h, 'detail': 'detected' if h else 'missed'}
    return {'value': 0.0, 'fails': False, 'detail': f'the object is not in view at step {step}'}


# ================================================================================================ the factors
def failure_frames(case: dict, run: dict, cfg: dict) -> list:
    """The steps the failure is about: the given step (missed); the target's critical window (perception, both);
    the critical actor's last window_s before its critical moment (criticality, scenario)."""
    from . import milsearch as MS
    fps = _fps(cfg)
    if case['failure'] == 'missed':
        return [int(case['step'])]
    mom = MS.critical_moments(run.get('scenario'))
    actor = case.get('actor') or ((run.get('scenario') or {}).get('criticality') or {}).get('overall', {}).get(
        'critical_actor')
    if actor not in mom:
        return []
    tc, w = mom[actor]['t'], float(case.get('window_s', MS.DEFAULTS['window_s']))
    return [s for s in range(int(round(_scenario_steps(cfg)))) if tc - w - 1e-9 <= (s + 1) / fps <= tc + 1e-9]


def _scenario_steps(cfg):
    return float(cfg['scenario'].get('duration_s', 12.0)) * _fps(cfg)


def _target_objects(case, run, cfg, run_dir, steps) -> list:
    cam = _camera(cfg, case.get('camera'))
    if cam is None or not _labelled(cfg, cam['id']):
        return []
    tid = _track_of(case, run)
    objs = _objects_by_step(run_dir, cam['id'])
    out = []
    for s in steps:
        for o in objs.get(s, ()):
            if o.get('track_id') == tid:
                out.append((s, o, objs[s]))
    return out


def _overlap(a, b) -> float:
    """Intersection area of two xyxy boxes over a's area."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    return max(w, 0.0) * max(h, 0.0) / area if w > 0 and h > 0 else 0.0


def plan_factors(case: dict, run: dict, cfg: dict, run_dir: str) -> list:
    """Per factor of FACTORS: {'factor', 'title', 'present', 'in_case' (what the case has), 'applicable', 'why_not',
    'intervention' (in words), 'transforms' ([(name, cfg -> cfg)] — one per re-run: two for appearance),
    'removal' (the transform the "all removed" run applies), 'caveat'}."""
    scn = cfg['scenario']
    ids = (run.get('scenario') or {}).get('actor_ids') or {}
    actor = case.get('actor')
    steps = failure_frames(case, run, cfg)
    tobjs = _target_objects(case, run, cfg, run_dir, steps)
    w = cfg.get('weather') or {}
    rows = []

    def row(f, present, in_case, transforms=(), why_not=None, intervention='', caveat=None):
        rows.append({'factor': f, 'title': FACTOR_TITLES[f], 'present': bool(present), 'in_case': in_case,
                     'applicable': bool(present and transforms), 'why_not': why_not, 'intervention': intervention,
                     'transforms': list(transforms), 'caveat': caveat})

    # occlusion
    if not tobjs:
        row('occlusion', False, 'not known: the target has no labels in the failure frames'
            if not _labelled(cfg, case.get('camera')) else 'the target is not in view in the failure frames',
            why_not='no labelled view of the target to tell occlusion')
    else:
        occ = max(float(o.get('occluded') or 0.0) for _, o, _ in tobjs)
        if occ < OCCLUDED_MIN:
            row('occlusion', False, f'hidden at most {occ:.0%} of its box in the failure frames (below '
                                    f'{OCCLUDED_MIN:.0%}: not occluded)')
        else:
            by_track = {int(v): k for k, v in ids.items()}
            seen = {}
            for _, o, frame in tobjs:
                box = o.get('bbox_full') or o.get('bbox')
                for q in frame:
                    if q is o or q.get('track_id') == o.get('track_id') or not q.get('bbox'):
                        continue
                    if float(q.get('range_m') or 1e9) < float(o.get('range_m') or 0.0) and \
                            _overlap(box, q['bbox']) > 0.05:
                        seen[q.get('track_id')] = seen.get(q.get('track_id'), 0) + 1
            scen = sorted({by_track[t] for t in seen if isinstance(t, int) and t in by_track} - {actor})
            other = sorted(str(t) for t in seen if not (isinstance(t, int) and t in by_track))
            desc = f'hidden up to {occ:.0%} of its box'
            if scen:
                trig = [a for a in scen for ev in scn.get('timeline', ())
                        if (ev.get('trigger') or {}).get('actor') == a]
                if trig:
                    row('occlusion', True, desc + f' by {", ".join(scen)}',
                        why_not=f'a timeline trigger depends on {", ".join(sorted(set(trig)))}: removing it would '
                                'change the scenario\'s timing, not only the view')
                else:
                    row('occlusion', True, desc + f' by {", ".join(scen)}' + (f' (and {", ".join(other)})'
                                                                             if other else ''),
                        [('remove ' + ', '.join(scen), lambda c, scen=scen: _remove_actors(c, scen))],
                        intervention=f'remove {", ".join(scen)} from the scenario',
                        caveat='a removed actor takes its shadow and reflections with it' +
                               (f'; {", ".join(other)} cannot be moved (part of the map, or not a scenario actor)'
                                if other else ''))
            else:
                row('occlusion', True, desc + (f' by {", ".join(other)}' if other else
                                               ' by something unlabelled (the ego\'s own body, or the map)'),
                    why_not='the occluder is not a scenario actor: it cannot be moved')
    # range
    a_doc = next((a for a in scn.get('actors', []) if a['id'] == actor), None) if actor else None
    if a_doc is None:
        row('range', False, 'no scenario actor is the target', why_not='only a scenario actor can be moved')
    elif not tobjs:
        row('range', False, 'not known (no labelled view of the target)', why_not='its range is not known')
    else:
        rng = float(np.median([float(o.get('range_m') or 0.0) for _, o, _ in tobjs]))
        ahead = float((a_doc.get('place') or {}).get('ahead_m', 0.0))
        if rng < NEAR_M:
            row('range', False, f'{rng:.1f} m in the failure frames (already within {NEAR_M:g} m)')
        elif ahead <= 0:
            row('range', True, f'{rng:.1f} m', why_not='it starts beside or behind the ego: no nearer start along '
                                                       'its lane')
        else:
            shift = min(rng / 2.0, ahead - 1.0)
            row('range', True, f'{rng:.1f} m in the failure frames',
                [(f'{shift:.1f} m nearer', lambda c, s=shift: _move_nearer(c, actor, s))],
                intervention=f'start {actor} {shift:.1f} m nearer along its lane (about {rng - shift:.0f} m then)',
                caveat='triggers on distance to it fire at other times; a nearer object can also meet the ego '
                       'differently')
    # lighting, fog, rain
    sun = float(w.get('sun_altitude_angle', 90.0))
    if sun < LOW_SUN_DEG:
        row('lighting', True, f'sun at {sun:.0f}°', [(f'sun at {DAY_SUN_DEG:.0f}°',
                                                      lambda c: _weather(c, sun_altitude_angle=DAY_SUN_DEG))],
            intervention=f'raise the sun from {sun:.0f}° to {DAY_SUN_DEG:.0f}°')
    else:
        row('lighting', False, f'sun at {sun:.0f}° (not below {LOW_SUN_DEG:.0f}°)')
    fog = float(w.get('fog_density', 0.0))
    if fog >= FOG_MIN:
        row('fog', True, f'fog density {fog:.0f} (visibility ≈ {7000 / fog:.0f} m)',
            [('no fog', lambda c: _weather(c, fog_density=0.0))], intervention=f'fog density {fog:.0f} → 0')
    else:
        row('fog', False, f'fog density {fog:.0f} (below {FOG_MIN:g})')
    wet = {k: float(w.get(k, 0.0)) for k in ('precipitation', 'precipitation_deposits', 'wetness')}
    if max(wet.values()) >= RAIN_MIN:
        row('rain', True, ', '.join(f'{k} {v:.0f}' for k, v in wet.items()),
            [('dry', lambda c: _weather(c, precipitation=0.0, precipitation_deposits=0.0, wetness=0.0))],
            intervention='precipitation, puddles and wetness → 0')
    else:
        row('rain', False, 'dry (precipitation, puddles and wetness below ' f'{RAIN_MIN:g})')
    # appearance
    if a_doc is None:
        row('appearance', False, 'no scenario actor is the target', why_not='only a scenario actor can be swapped')
    else:
        bp = ((run.get('scenario') or {}).get('blueprints') or {}).get(actor) or a_doc['kind']
        base = int(a_doc.get('variant', 0) or 0)
        row('appearance', True, f'{bp} ({a_doc["kind"]})',
            [(f'variant {base + k}', lambda c, k=base + k: _variant(c, actor, k)) for k in range(1, SWAPS + 1)],
            intervention=f'swap {actor}\'s blueprint for the next {SWAPS} of its kind',
            caveat='a fix by a swap is about those blueprints; another one of the class may fail again')
    # sensor faults
    faulty = [s['id'] for s in cfg['rig']['sensors'] if s.get('faults')]
    if faulty:
        row('sensor_faults', True, 'faults on ' + ', '.join(faulty), [('no faults', _no_faults)],
            intervention='remove every injected sensor fault')
    else:
        row('sensor_faults', False, 'no sensor faults were injected')
    for r in rows:
        r['removal'] = r['transforms'][0][1] if r['applicable'] else None
    return rows


def _remove_actors(cfg, ids):
    scn = cfg['scenario']
    scn['actors'] = [a for a in scn['actors'] if a['id'] not in ids]
    tl = []
    for ev in scn.get('timeline', []):
        acts = [x for x in ev.get('actions', []) if x.get('actor') not in ids]
        if acts:
            tl.append(dict(ev, actions=acts))
    scn['timeline'] = tl
    return cfg


def _move_nearer(cfg, actor, shift):
    for a in cfg['scenario']['actors']:
        if a['id'] == actor:
            place = a.setdefault('place', {})
            place['ahead_m'] = round(float(place.get('ahead_m', 0.0)) - float(shift), 4)
    return cfg


def _weather(cfg, **fields):
    from . import weather as W
    spec = dict(cfg.get('weather_source') or {})
    spec.update(fields)
    cfg['weather_source'] = spec
    cfg['weather'] = dict(W.resolve(spec), **{k: float(v) for k, v in fields.items()})
    return cfg


def _variant(cfg, actor, k):
    for a in cfg['scenario']['actors']:
        if a['id'] == actor:
            a['variant'] = int(k)
    return cfg


def _no_faults(cfg):
    for s in cfg['rig']['sensors']:
        s.pop('faults', None)
    return cfg


# ================================================================================================ running
def _cfg_for(base: dict, transforms: list, out_root: str, label: str, svc) -> dict:
    from . import datasets as D
    c = copy.deepcopy(base)
    for t in transforms:
        c = t(c)
    c.pop('cache_key', None)
    c['port'], c['tm_port'] = svc.port, svc.tm_port      # this service's server (ports are not in the key)
    c['out'] = '/pending'
    key = D.drive_key(c)
    c['out'] = os.path.join(out_root, 'runs', key[:16])
    c['name'] = f'why: {label}'
    c['cache_key'] = key
    return c


def _run_all(svc, cfgs: list, cancel=None, progress=None) -> None:
    """Drive every config not already recorded (cache key), through the service's queue; wait for them."""
    from .jobs import FINAL
    jobs, seen = [], set()
    for c in cfgs:
        if c['cache_key'] in seen:                     # the same re-run planned twice (one factor: "all" is it)
            continue
        seen.add(c['cache_key'])
        try:
            if PV.read_json(os.path.join(c['out'], 'run.json')).get('status') == 'ok' and \
                    PV.read_json(os.path.join(c['out'], 'drive.json')).get('cache_key') == c['cache_key']:
                continue
        except (OSError, ValueError):
            pass
        jobs.append(svc.submit_drive(c, title=c['name']))
    total = max(len(jobs), 1)
    while any(j.state not in FINAL for j in jobs):
        if cancel is not None and cancel():
            for j in jobs:
                svc.queue.cancel(j)
        svc.queue.poll()
        if progress:
            progress(sum(j.state in FINAL for j in jobs) / total)
        time.sleep(0.2)
    if cancel is not None and cancel():
        raise RootCauseError('cancelled')


def _trajectories(run_dir):
    try:
        with open(os.path.join(run_dir, 'trajectories.json'), encoding='utf-8') as fh:
            return json.load(fh).get('actors')
    except (OSError, ValueError):
        return None


def _repeat(base_dir, ctrl_dir) -> dict:
    a, b = _trajectories(base_dir), _trajectories(ctrl_dir)
    if a is None or b is None:
        return {'bit_for_bit': None, 'repeat': 'no trajectories to compare'}
    if a == b:
        return {'bit_for_bit': True, 'repeat': 'every trajectory identical'}
    worst = 0.0
    for k in set(a) & set(b):
        for p, q in zip(a[k], b[k]):
            worst = max(worst, math.hypot(p[1] - q[1], p[2] - q[2]))
    return {'bit_for_bit': False, 'repeat': f'trajectories differ, by up to {worst:.3f} m'}


def counterfactual(case: dict, svc, out_dir: str, workspace: str = '', emit=None, cancel=None, progress=None,
                   sufficiency: bool = True) -> dict:
    """Attribute one failing scenario run (module doc, 1). case: {'run', 'failure', 'threshold', 'model' (a
    registration, for perception/missed/both), 'actor' (a scenario actor id) or 'track_id', 'step' (missed),
    'window_s', 'lead_s', 'iou', 'min_score', 'metric', 'camera'}. Writes rootcause.json and run.json in out_dir."""
    from . import milsearch as MS
    t0 = time.monotonic()
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rec = PV.new_run('why', ['case', case.get('run')], None, {k: v for k, v in case.items() if k != 'model'})
    PV.write_json(os.path.join(out_dir, 'run.json'), rec)
    try:
        doc = _counterfactual(case, svc, out_dir, workspace, emit, cancel, progress, sufficiency, MS)
        rec.update(status='ok', exit_code=0)
    except Exception as e:                       # noqa: BLE001 (recorded in run.json, then re-raised)
        rec.update(status='refused' if isinstance(e, Refused) else 'failed', error=str(e),
                   exit_code=6 if isinstance(e, Refused) else 2, finished=PV.utc_now())
        PV.write_json(os.path.join(out_dir, 'run.json'), rec)
        raise
    rec.update(finished=PV.utc_now(), timings={'wall_s': round(time.monotonic() - t0, 3)},
               outputs={'rootcause.json': PV.sha256_file(os.path.join(out_dir, 'rootcause.json'))},
               summary={'status': doc['status'], 'runs': doc['n_runs']})
    PV.write_json(os.path.join(out_dir, 'run.json'), rec)
    return doc


def _counterfactual(case, svc, out_dir, workspace, emit, cancel, progress, sufficiency, MS) -> dict:
    case = dict(case)
    if case.get('failure') not in FAILURES:
        raise RootCauseError(f'failure must be one of {", ".join(FAILURES)}')
    run_dir = os.path.abspath(case['run'])
    run, cfg = load_case(run_dir)
    needs_model = case['failure'] in ('perception', 'missed', 'both')
    if needs_model:
        if not _labelled(cfg, case.get('camera')):
            raise Refused('the run\'s camera did not record boxes2d labels: the model cannot be scored on it')
        why = MS.evaluable(case.get('model'), workspace)
        if why:
            raise Refused(f'the model cannot be run: {why}')
    if case['failure'] == 'missed' and case.get('step') is None:
        raise RootCauseError('a missed detection needs the frame (step)')
    spec = {'objective': 'perception', 'model': case.get('model'), 'metric': case.get('metric') or 'perception',
            'window_s': case.get('window_s'), 'lead_s': case.get('lead_s'), 'iou': case.get('iou'),
            'min_score': case.get('min_score'), 'camera': case.get('camera'),
            'target_classes': case.get('target_classes'), 'range_m': case.get('range_m')}
    spec = MS.normalise_spec(spec) if needs_model else None
    # the target: a scenario actor (its CARLA id differs between runs), else a track id
    ids = (run.get('scenario') or {}).get('actor_ids') or {}
    if case.get('track_id') is not None and not case.get('actor'):
        by_track = {int(v): k for k, v in ids.items()}
        tid = case['track_id']
        tid = int(tid) if isinstance(tid, (int, float)) or str(tid).isdigit() else tid
        case['track_id'] = tid
        if isinstance(tid, int) and tid in by_track:
            case['actor'] = by_track[tid]
    if case.get('actor') and case['actor'] not in ids:
        raise RootCauseError(f'no scenario actor {case["actor"]!r} in this run ({", ".join(sorted(ids)) or "none"})')
    meta_run = os.path.basename(run_dir)

    def perception(dirs):
        return MS.score_runs(dirs, spec, workspace, emit, cancel, meta_run=meta_run,
                             keep_frames=case['failure'] == 'missed')

    def score(dirs):
        if not needs_model:
            return {d: score_case(case, d) for d in dirs}
        sc = perception(dirs)
        return {d: score_case(case, d, sc[d]) for d in dirs}

    if case.get('actor') is None and case['failure'] != 'missed':
        crit_actor = (((run.get('scenario') or {}).get('criticality') or {}).get('overall') or {}).get(
            'critical_actor')
        case['actor'] = (perception([run_dir])[run_dir].get('worst_actor') if needs_model else None) or crit_actor
    base = score([run_dir])[run_dir]
    if not base['fails']:
        raise Refused(f'the case does not fail under this test ({case["failure"]}: {base["detail"]}, value '
                      f'{base["value"]} below {case.get("threshold", 0.5)})')
    factors = plan_factors(case, run, cfg, run_dir)
    present = [f for f in factors if f['applicable']]
    root = out_dir
    plan = [('control', 'the case again, unchanged', [], None)]
    for f in present:
        for name, t in f['transforms']:
            plan.append((f['factor'], f'{f["factor"]}: {name}', [t], 'removed'))
    if present:
        plan.append(('all', 'every present factor removed', [f['removal'] for f in present], 'all'))
    if sufficiency and len(present) >= 2:
        for f in present:
            others = [g['removal'] for g in present if g is not f]
            plan.append((f['factor'], f'{f["factor"]} alone (the others removed)', others, 'alone'))
    cfgs = [_cfg_for(cfg, ts, root, label, svc) for _, label, ts, _ in plan]
    if emit is not None:
        emit.log(f'{len(cfgs)} re-runs: the control, {len(present)} factor(s) removed one at a time'
                 + (', all removed and each kept alone' if sufficiency and len(present) >= 2 else ''))
    _run_all(svc, cfgs, cancel, (lambda f: progress(0.8 * f, 're-runs')) if progress else None)
    def status(c):
        try:
            return PV.read_json(os.path.join(c['out'], 'run.json'))
        except (OSError, ValueError):
            return {'status': 'failed', 'error': 'no run.json'}
    failed = [c for c in cfgs if status(c).get('status') != 'ok']
    ok_dirs = list(dict.fromkeys(c['out'] for c in cfgs if c not in failed))
    if progress:
        progress(0.85, 'scoring the re-runs')
    results = score(ok_dirs) if ok_dirs else {}
    for c in failed:
        results[c['out']] = {'value': None, 'fails': None, 'detail': f'the re-run failed: {status(c).get("error")}'}
    runs = []
    for (factor, label, _, role), c in zip(plan, cfgs):
        r = results[c['out']]
        runs.append({'factor': factor, 'role': role or 'control', 'label': label,
                     'run': os.path.relpath(c['out'], out_dir), 'key': c['cache_key'], **r})
    control = runs[0]
    control.update(_repeat(run_dir, cfgs[0]['out']))
    doc = {'schema': SCHEMA, 'kind': 'counterfactual', 'wording': WORDING,
           'case': {'run': run_dir, 'scenario': (run.get('scenario') or {}).get('id'),
                    'values': (run.get('scenario') or {}).get('values'), 'failure': case['failure'],
                    'threshold': float(case.get('threshold', 0.5)), 'actor': case.get('actor'),
                    'track_id': case.get('track_id'), 'step': case.get('step'),
                    'model': {k: (case.get('model') or {}).get(k) for k in ('id', 'name', 'runtime', 'entry', 'path')}
                    if needs_model else None,
                    'model_sha256': MS.model_fingerprint(case['model'], workspace) if needs_model else None,
                    'settings': {k: spec[k] for k in ('metric', 'window_s', 'lead_s', 'iou', 'min_score')}
                    if spec else None},
           'base': base, 'control': control, 'runs': runs, 'n_runs': len(runs)}
    if control['fails'] is not True:
        doc['status'] = 'not_reproduced'
        doc['summary'] = (f'the control re-run of the case did not fail again ({control["detail"]}; '
                          f'{control["repeat"]}): a failure that does not repeat cannot be attributed by re-runs, '
                          'so nothing is')
        doc['factors'] = [_public(f) for f in factors]
    else:
        table = []
        for f in factors:
            pf = _public(f)
            mine = [r for r in runs if r['factor'] == f['factor']]
            rem = [r for r in mine if r['role'] == 'removed']
            alone = next((r for r in mine if r['role'] == 'alone'), None)
            if f['applicable'] and rem:
                fixes = [r['fails'] is False for r in rem]
                pf['removing_fixes'] = any(fixes) if all(r['fails'] is not None for r in rem) else None
                pf['fixed_in'] = f'{sum(fixes)} of {len(rem)}'
                pf['values_after'] = [r['value'] for r in rem]
                if alone is not None:
                    pf['alone_fails'] = alone['fails']
                    pf['value_alone'] = alone['value']
                elif len(present) == 1:
                    pf['alone_fails'] = True             # the only factor present: the case itself is "it alone"
                    pf['value_alone'] = base['value']
                pf['verdict'] = verdict(pf.get('removing_fixes'), pf.get('alone_fails'))
            table.append(pf)
        allr = next((r for r in runs if r['role'] == 'all'), None)
        doc['factors'] = table
        doc['all_removed'] = allr
        nec = [t['factor'] for t in table if t.get('removing_fixes')]
        doc['status'] = 'attributed' if present else 'no_factor'
        if not present:
            doc['summary'] = 'none of the factors Falcon can change is present in this case'
        elif nec:
            doc['summary'] = (f'removing {_and(nec)} alone fixes the failure in this case' +
                              ('' if allr is None or allr['fails'] is False else
                               '; with every present factor removed it still fails'))
        elif allr is not None and allr['fails'] is False:
            doc['summary'] = ('no single factor\'s removal fixes it, but removing them all does: the failure is '
                              'overdetermined (more than one factor produces it here)')
        else:
            doc['summary'] = ('it still fails with every factor Falcon can change removed: its cause is something '
                              'else (the scenario itself, the model on this content)')
    doc['provenance'] = PV.block({'case/run.json': os.path.join(run_dir, 'run.json'),
                                  'case/drive.json': os.path.join(run_dir, 'drive.json')}
                                 | ({'model': _model_path(case['model'], workspace)} if needs_model and
                                    _model_path(case['model'], workspace) else {}),
                                 cfg.get('seed'), {'case': {k: v for k, v in case.items() if k != 'model'},
                                                   'sufficiency': sufficiency,
                                                   'thresholds': {'low_sun_deg': LOW_SUN_DEG, 'fog_min': FOG_MIN,
                                                                  'rain_min': RAIN_MIN, 'near_m': NEAR_M,
                                                                  'occluded_min': OCCLUDED_MIN, 'swaps': SWAPS}})
    PV.write_json(os.path.join(out_dir, 'rootcause.json'), _clean(doc))
    if progress:
        progress(1.0, 'done')
    return doc


def _model_path(m, ws):
    if not m or m.get('runtime') == 'python' or not m.get('path'):
        return None
    from .models import resolve_path
    return resolve_path(m, ws)


def _public(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ('transforms', 'removal')}


def _and(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' or ' + xs[-1]


def verdict(fixes, alone) -> str:
    if fixes is None:
        return 'not decided (a re-run failed)'
    if fixes and alone:
        return 'necessary and sufficient here: removing it fixes the failure, and it alone produces it'
    if fixes:
        return 'necessary here: removing it alone fixes the failure; it does not produce it on its own'
    if alone:
        return ('sufficient here: it alone produces the failure, but removing it alone does not fix it (another '
                'factor also produces it)')
    if alone is None:
        return 'removing it alone does not fix the failure'
    return 'neither: removing it does not fix the failure, and alone it does not produce it'


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ================================================================================================ static
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), as Spearman's correlation needs."""
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    ranks = np.empty(len(x), float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry) / d if d > 0 else None


def object_records(frames: list, iou: float = 0.5, min_score: float = 0.5) -> list:
    """Every scored (not ignored) ground-truth object of an evaluation's frames: {'drive', 'missed', 'class',
    'range_m', 'occluded', 'truncated', 'height_px', <the frame's numeric conditions>}."""
    from . import milsearch as MS
    out = []
    for f in frames:
        hits = MS.frame_hits(f, iou, min_score)
        cond = {k: float(v) for k, v in (f.get('conditions') or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for g, h in zip(f.get('gt') or (), hits):
            if int(g.get('ignore', 0)):
                continue
            rec = {'drive': f['drive'], 'missed': 0.0 if h else 1.0, 'class': g['class'],
                   'range_m': g.get('range_m'), 'occluded': g.get('occluded'), 'truncated': g.get('truncated'),
                   'height_px': float(g['bbox'][3])}
            rec.update(cond)
            out.append(rec)
    return out


def _bins(v: np.ndarray, k: int) -> np.ndarray:
    u = np.unique(v)
    if len(u) <= k:
        mids = (u[:-1] + u[1:]) / 2.0
        return np.concatenate([[u[0] - 1e-9], mids, [u[-1] + 1e-9]])
    e = np.unique(np.quantile(v, np.linspace(0, 1, k + 1)))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def static(eval_dir: str, out_dir: str | None = None, n_boot: int = 1000, seed: int = 0, iou: float = 0.5,
           min_score: float = 0.5, bins: int = 4) -> dict:
    """Correlates of misses over an evaluation (module doc, 2): rootcause.json (and run.json) in out_dir."""
    from . import harness as H
    t0 = time.monotonic()
    ev_path = os.path.join(eval_dir, 'evaluation.json')
    pr_path = os.path.join(eval_dir, 'predictions.jsonl')
    try:
        ev = PV.read_json(ev_path)
        frames = H.read_predictions(pr_path)
    except (OSError, ValueError) as e:
        raise RootCauseError(f'{eval_dir} is not an evaluation folder: {e}') from None
    if (ev.get('model') or {}).get('kind') != 'detector':
        raise RootCauseError('the static attribution reads a detector\'s evaluation')
    recs = object_records(frames, iou, min_score)
    if not recs:
        raise RootCauseError('the evaluation has no scored object')
    drives = sorted({r['drive'] for r in recs})
    unit = 'drive' if len(drives) > 1 else 'frame'
    if unit == 'frame':                           # one drive: resample frames (said so: too narrow)
        recs = []
        for f in frames:
            for rr in object_records([f], iou, min_score):
                rr['drive'] = f['image_id']
                recs.append(rr)
        drives = sorted({r['drive'] for r in recs})
    idx = {d: np.asarray([i for i, r in enumerate(recs) if r['drive'] == d]) for d in drives}
    miss = np.asarray([r['missed'] for r in recs])
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([idx[drives[j]] for j in rng.integers(len(drives), size=len(drives))])
             for _ in range(n_boot)]

    def ci(fn):
        vals = np.asarray([v for v in (fn(b) for b in boots) if v is not None and math.isfinite(v)])
        if len(vals) < max(10, n_boot // 10):
            return None, None
        return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
    numeric = ['range_m', 'occluded', 'truncated', 'height_px'] + sorted(
        {k for r in recs for k in r if k not in ('drive', 'missed', 'class', 'range_m', 'occluded', 'truncated',
                                                 'height_px')})
    attrs = []
    for name in numeric:
        have = np.asarray([r.get(name) is not None for r in recs])
        if have.sum() < 3:
            continue
        x = np.asarray([float(r[name]) if r.get(name) is not None else np.nan for r in recs])
        sel = np.flatnonzero(have)
        if np.all(x[sel] == x[sel][0]):
            attrs.append({'attribute': name, 'kind': 'numeric', 'note': f'no variation (always {x[sel][0]:g})',
                          'n': int(len(sel))})
            continue
        rho = spearman(x[sel], miss[sel])

        def rho_b(b, name=name):
            b = b[have[b]]
            return spearman(x[b], miss[b])
        lo, hi = ci(rho_b)
        edges = _bins(x[sel], bins)
        rows = []
        for a, b2 in zip(edges, edges[1:]):
            m = have & (x > a) & (x <= b2)
            n = int(m.sum())
            if not n:
                continue

            def rate(b, m=m):
                mm = m[b]
                return float(miss[b][mm].mean()) if mm.any() else None
            blo, bhi = ci(rate)
            rows.append({'lo': _r(a + 1e-9, 4), 'hi': _r(b2 - 1e-9, 4), 'n': n, 'misses': int(miss[m].sum()),
                         'miss_rate': _r(float(miss[m].mean()), 4), 'ci_low': _r(blo), 'ci_high': _r(bhi)})
        attrs.append({'attribute': name, 'kind': 'numeric', 'n': int(len(sel)), 'spearman': _r(rho),
                      'ci_low': _r(lo), 'ci_high': _r(hi), 'bins': rows,
                      'clear': lo is not None and (lo > 0 or hi < 0)})
    classes = sorted({r['class'] for r in recs})
    crow = []
    cls = np.asarray([r['class'] for r in recs])
    for c in classes:
        m = cls == c

        def rate(b, m=m):
            mm = m[b]
            return float(miss[b][mm].mean()) if mm.any() else None
        lo, hi = ci(rate)
        crow.append({'value': c, 'n': int(m.sum()), 'misses': int(miss[m].sum()),
                     'miss_rate': _r(float(miss[m].mean())), 'ci_low': _r(lo), 'ci_high': _r(hi)})
    attrs.append({'attribute': 'class', 'kind': 'categorical', 'n': len(recs), 'categories': crow})
    order = sorted((a for a in attrs if a.get('spearman') is not None),
                   key=lambda a: (not a['clear'], -abs(a['spearman'])))
    doc = {'schema': SCHEMA, 'kind': 'static', 'wording': STATIC_WORDING,
           'evaluation': os.path.abspath(eval_dir), 'model': ev.get('model'),
           'synthetic_model': ev.get('synthetic_model'), 'n_objects': len(recs), 'n_missed': int(miss.sum()),
           'miss_rate': _r(float(miss.mean())), 'unit': unit, 'n_units': len(drives), 'n_boot': n_boot,
           'seed': seed, 'matching': {'iou': iou, 'min_score': min_score},
           'note': None if unit == 'drive' and len(drives) >= 5 else
           ('one drive: the intervals are over frames, which are not independent — too narrow' if unit == 'frame'
            else f'only {len(drives)} drives: an interval over so few units is not reliable'),
           'ranking': [a['attribute'] for a in order], 'attributes': attrs,
           'summary': _static_summary(order),
           'provenance': PV.block({'evaluation.json': ev_path, 'predictions.jsonl': pr_path}, seed,
                                  {'n_boot': n_boot, 'iou': iou, 'min_score': min_score, 'bins': bins})}
    doc = _clean(doc)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        PV.write_json(os.path.join(out_dir, 'rootcause.json'), doc)
        rec = PV.new_run('why', ['evaluation', eval_dir], seed, {'n_boot': n_boot, 'iou': iou,
                                                                 'min_score': min_score, 'bins': bins},
                         {'evaluation.json': ev_path, 'predictions.jsonl': pr_path})
        rec.update(status='ok', exit_code=0, finished=PV.utc_now(), timings={'wall_s': round(time.monotonic() - t0, 3)},
                   outputs={'rootcause.json': PV.sha256_file(os.path.join(out_dir, 'rootcause.json'))})
        PV.write_json(os.path.join(out_dir, 'run.json'), rec)
    return doc


def _static_summary(order: list) -> str:
    clear = [a for a in order if a['clear']]
    if not clear:
        return 'no attribute is clearly associated with the misses (every interval of ρ includes 0)'
    return 'misses go with ' + '; '.join(
        f'{"higher" if a["spearman"] > 0 else "lower"} {a["attribute"]} (Spearman ρ {a["spearman"]:+.2f}, 95 % '
        f'[{a["ci_low"]:+.2f}, {a["ci_high"]:+.2f}])' for a in clear[:4])


# ================================================================================================ episodes
FEATURE_GROUPS = {
    'ego_motion': ('speed_mps', 'accel_mps2', 'prev_steer', 'prev_accel'),
    'target_speed': ('target_speed_mps',),
    'lane': ('lateral_offset_m', 'heading_error_rad', 'lane_width_m'),
    'curvature': ('curvature_10m', 'curvature_25m', 'curvature_50m'),
    'lead': ('lead_present', 'lead_gap_m', 'lead_closing_mps', 'lead_ttc_s'),
    'vru': ('vru_present', 'vru_ahead_m', 'vru_lateral_m', 'vru_lat_speed_mps', 'vru_ttc_s'),
    'traffic_light': ('tl_known', 'tl_red', 'tl_yellow', 'tl_green', 'tl_distance_m'),
    'route': ('route_progress', 'route_remaining_m'),
}
GROUP_TITLES = {'ego_motion': 'Its own motion (speed, acceleration, last action)', 'target_speed': 'Target speed',
                'lane': 'Lane position and heading', 'curvature': 'Road curvature ahead',
                'lead': 'The lead vehicle', 'vru': 'Pedestrians and cyclists in the path',
                'traffic_light': 'Traffic lights', 'route': 'Route progress'}


INFRACTION_WORDS = {'collision_vehicle': 'collision with a vehicle', 'collision_pedestrian': 'collision with a '
                    'pedestrian', 'collision_static': 'collision with a static object', 'red_light': 'red light run',
                    'stop_sign': 'stop sign run', 'off_road': 'drive off the road', 'wrong_way': 'wrong-way drive',
                    'route_deviation': 'route deviation', 'agent_blocked': 'block'}


def neutral_values() -> dict:
    """The "nothing there" value of each presence group's features (core.drivefeatures' sentinels)."""
    from . import drivefeatures as DF
    return {'lead_present': 0.0, 'lead_gap_m': DF.LEAD_RANGE_M, 'lead_closing_mps': 0.0,
            'lead_ttc_s': DF.TTC_CAP_S, 'vru_present': 0.0, 'vru_ahead_m': DF.VRU_RANGE_M, 'vru_lateral_m': 0.0,
            'vru_lat_speed_mps': 0.0, 'vru_ttc_s': DF.TTC_CAP_S, 'tl_known': 0.0, 'tl_red': 0.0, 'tl_yellow': 0.0,
            'tl_green': 0.0, 'tl_distance_m': DF.TL_RANGE_M}


def _drive_episode(e, env, act, names, mask=None, frozen=None, tol_steps=20):
    """One closed-loop episode with the episode's seed and options; mask: feature indices replaced by `frozen`
    (array) or their neutral value. Returns {'infractions', 'actions', 'steps', 'first_obs'}."""
    obs, info = env.reset(seed=int(e['seed']), options=e.get('options') or None)
    first = np.asarray(obs['state'], np.float32).copy()
    actions, infs = [], []
    term = trunc = False
    steps = 0
    limit = max(len(e.get('actions') or ()) + tol_steps * 4, 1)
    while not (term or trunc) and steps < limit:
        o = dict(obs)
        if mask is not None:
            st = np.asarray(o['state'], np.float32).copy()
            st[mask] = frozen[mask]
            o['state'] = st
        a = act(o, names)
        actions.append(np.asarray(a).tolist())
        obs, _r, term, trunc, info = env.step(a)
        steps += 1
        infs.extend(info.get('infractions_new') or ())
    summ = info.get('episode_summary') or {}
    if summ.get('infractions') is not None:
        infs = summ['infractions']
    return {'infractions': infs, 'actions': actions, 'steps': steps, 'first_obs': first}


def episode(e: dict, env, policy, infraction: int | None = None, groups=None, tol_s: float = 1.0) -> dict:
    """Attribute a Leaderboard episode's infraction to the policy's input groups (module doc, 3). env: an
    environment like the episode's (core.gymenv.FalconDriveEnv); policy: a core.policy.Policy or a callable
    obs -> action. Returns the document (not written: episode_command writes it)."""
    from . import drivefeatures as DF
    from . import leaderboard as LB
    why = LB.replayable(e)
    if why:
        raise Refused(f'the episode cannot be driven again exactly: {why}')
    infs = e.get('infractions') or []
    fails = [i for i in infs if i.get('kind') in DF.FAILURE_KINDS] or infs
    if not fails:
        raise Refused('the episode has no infraction to attribute')
    target = infs[infraction] if infraction is not None else fails[0]
    names = list(env.feature_names)
    act = policy.act if hasattr(policy, 'act') else (lambda o, n: policy(o))
    fps = float(getattr(env, 'ready', {}).get('fps') or 20.0)
    tol = int(round(tol_s * fps))

    def recurs(res):
        same = [i for i in res['infractions'] if i.get('kind') == target['kind']]
        near = [i for i in same if abs(int(i.get('step', 0)) - int(target['step'])) <= tol]
        return bool(same), near

    ctrl = _drive_episode(e, env, act, names)
    c_any, c_near = recurs(ctrl)
    logged = e.get('actions') or []
    same_actions = len(ctrl['actions']) >= len(logged) and all(
        np.allclose(np.asarray(a, float), np.asarray(b, float)) for a, b in zip(ctrl['actions'], logged))
    neutral = neutral_values()
    rows = []
    groups = groups or list(FEATURE_GROUPS)
    if c_any:
        for g in groups:
            idx = [names.index(f) for f in FEATURE_GROUPS[g] if f in names]
            if not idx:
                rows.append({'group': g, 'title': GROUP_TITLES[g], 'applicable': False,
                             'why_not': 'the environment does not give these features'})
                continue
            frozen = ctrl['first_obs'].copy()
            how = 'held at their value at the episode\'s start'
            if all(names[i] in neutral for i in idx):
                for i in idx:
                    frozen[i] = neutral[names[i]]
                how = 'replaced by "nothing there" (' + ', '.join(f'{names[i]} = {neutral[names[i]]:g}'
                                                                   for i in idx) + ')'
            res = _drive_episode(e, env, act, names, np.asarray(idx), frozen)
            anyk, near = recurs(res)
            rows.append({'group': g, 'title': GROUP_TITLES[g], 'applicable': True, 'features': [names[i] for i in idx],
                         'intervention': how, 'infraction_recurs': anyk, 'removing_fixes': not anyk,
                         'infractions': [{'kind': i.get('kind'), 'step': i.get('step')} for i in res['infractions']
                                         if i.get('kind') in DF.FAILURE_KINDS][:10], 'steps': res['steps']})
    nec = [r['group'] for r in rows if r.get('removing_fixes')]
    doc = {'schema': SCHEMA, 'kind': 'episode', 'wording': EPISODE_WORDING,
           'episode': {'episode': e.get('episode'), 'seed': e.get('seed'), 'scenario': (e.get('scenario') or {})
                       .get('id') if isinstance(e.get('scenario'), dict) else e.get('scenario'),
                       'infraction': target, 'motion': e.get('motion', 'kinematic')},
           'policy': getattr(policy, 'path', None) or getattr(policy, '__name__', None) or type(policy).__name__,
           'control': {'infraction_recurs': c_any, 'at_same_step': bool(c_near), 'same_actions': same_actions,
                       'steps': ctrl['steps']},
           'groups': rows, 'n_runs': 1 + len([r for r in rows if r.get('applicable')]),
           'tolerance_s': tol_s, 'sufficiency': 'not tested for input features (module doc)'}
    if not c_any:
        doc['status'] = 'not_reproduced'
        doc['summary'] = ('the control re-run did not meet the infraction again: a failure that does not repeat '
                          'cannot be attributed by re-runs')
    else:
        doc['status'] = 'attributed'
        kind = INFRACTION_WORDS.get(target['kind'], target['kind'].replace('_', ' '))
        doc['summary'] = (f'hiding {_and(GROUP_TITLES[g].lower() for g in nec)} from the policy prevents the {kind}'
                          if nec else
                          'no single input group\'s removal prevents the infraction')
    return _clean(doc)


# ================================================================================================ the command
def main_why(argv) -> int:
    """`falcon why case|evaluation|episode …` (docs/root-cause.md): protocol lines; 0 ok, 2 failed, 6 refused."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon why', description='automatic root-cause attribution')
    sub = ap.add_subparsers(dest='what', required=True)
    c = sub.add_parser('case', help='one failing scenario run, by counterfactual re-runs')
    c.add_argument('project')
    c.add_argument('--run', required=True, help='the failing run\'s folder (a scenario drive)')
    c.add_argument('--failure', default='', choices=('',) + FAILURES,
                   help='the failure test (default: perception with --model, else criticality)')
    c.add_argument('--model', default='')
    c.add_argument('--actor', default='', help='the scenario actor the failure is about')
    c.add_argument('--track', default='', help='or its track id in the run\'s labels')
    c.add_argument('--step', type=int, default=None, help='the frame of a missed detection')
    c.add_argument('--threshold', type=float, default=0.5)
    c.add_argument('--metric', default='perception')
    c.add_argument('--window', type=float, default=3.0)
    c.add_argument('--lead', type=float, default=3.0)
    c.add_argument('--camera', default=None)
    c.add_argument('--no-sufficiency', action='store_true', help='skip the "each factor alone" re-runs')
    c.add_argument('--out', '-o', required=True)
    c.add_argument('--fake-carla', default='')
    v = sub.add_parser('evaluation', help='correlates of misses over a whole evaluation (no re-run)')
    v.add_argument('evaluation')
    v.add_argument('--out', '-o', required=True)
    v.add_argument('--boot', type=int, default=1000)
    v.add_argument('--seed', type=int, default=0)
    v.add_argument('--bins', type=int, default=4)
    e = sub.add_parser('episode', help='a Leaderboard infraction, by the policy\'s input features')
    e.add_argument('project')
    e.add_argument('--episodes', required=True, help='an evaluation\'s episodes.jsonl')
    e.add_argument('--episode', type=int, required=True, help='the line number (0-based)')
    e.add_argument('--infraction', type=int, default=None)
    e.add_argument('--policy', required=True, help='a policy folder, or module:callable (obs -> action)')
    e.add_argument('--world', default='')
    e.add_argument('--rig', default='')
    e.add_argument('--out', '-o', required=True)
    e.add_argument('--fake-carla', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return P.OK if ex.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('why')
    try:
        if a.what == 'evaluation':
            doc = static(a.evaluation, a.out, a.boot, a.seed, bins=a.bins)
            em.result('ok', path=os.path.join(os.path.abspath(a.out), 'rootcause.json'),
                      summary={'objects': doc['n_objects'], 'missed': doc['n_missed'], 'top': doc['ranking'][:3]})
            return P.OK
        from .project import Project, ProjectError
        try:
            proj = Project.load(a.project)
        except (OSError, ProjectError) as ex:
            raise RootCauseError(str(ex)) from None
        if a.what == 'case':
            return _case_command(a, proj, em)
        return _episode_command(a, proj, em)
    except Refused as ex:
        print(f'failed: {ex}', flush=True)
        return P.REFUSED
    except (RootCauseError, ValueError, OSError, KeyError) as ex:
        print(f'failed: {ex}', flush=True)
        return P.FAILED


def _case_command(a, proj, em) -> int:
    from . import protocol as P
    from .gymenv import make_service
    m = proj.item('models', a.model) if a.model else None
    if a.model and m is None:
        raise RootCauseError(f'no model {a.model!r} in the project')
    failure = a.failure or ('missed' if a.step is not None and m else 'perception' if m else 'criticality')
    if failure in ('perception', 'missed', 'both') and m is None:
        raise RootCauseError(f'the {failure} test needs --model')
    track = a.track
    if track and track.lstrip('-').isdigit():
        track = int(track)
    case = {'run': a.run, 'failure': failure, 'threshold': a.threshold, 'model': m, 'actor': a.actor or None,
            'track_id': track if track != '' else None, 'step': a.step, 'metric': a.metric, 'window_s': a.window,
            'lead_s': a.lead, 'camera': a.camera}
    stop = {'flag': False}
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    svc = make_service(fake_carla=a.fake_carla, workspace=proj.workspace)
    try:
        doc = counterfactual(case, svc, a.out, proj.workspace, em, lambda: stop['flag'],
                             lambda f, t='': em.progress(f, t), not a.no_sufficiency)
    finally:
        svc.shutdown()
    em.result('ok', path=os.path.join(os.path.abspath(a.out), 'rootcause.json'),
              summary={'status': doc['status'], 'runs': doc['n_runs'], 'summary': doc.get('summary')})
    return P.OK


def load_policy_arg(spec: str):
    """A policy folder (core.policy.load_policy) or "module:callable" (a function obs -> action, or a class)."""
    import importlib
    if os.path.isdir(spec):
        from .policy import load_policy
        return load_policy(spec)
    mod, _, name = spec.partition(':')
    if not name:
        raise RootCauseError(f'{spec!r} is neither a policy folder nor module:callable')
    obj = getattr(importlib.import_module(mod), name)
    return obj() if isinstance(obj, type) else obj


def _episode_command(a, proj, em) -> int:
    from . import protocol as P
    from .gymenv import FalconDriveEnv, gym_config, make_service
    with open(a.episodes, encoding='utf-8') as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    if not 0 <= a.episode < len(rows):
        raise RootCauseError(f'{a.episodes} has {len(rows)} episodes')
    e = rows[a.episode]
    pol = load_policy_arg(a.policy)
    svc = make_service(fake_carla=a.fake_carla, workspace=proj.workspace)
    t0 = time.monotonic()
    env = None
    try:
        kind = 'discrete' if e.get('actions') and not isinstance(e['actions'][0], list) else 'continuous'
        kw = {k: v for k, v in (e.get('env_config') or {}).items() if v is not None}
        scn = e.get('scenario')
        scn = scn.get('id') if isinstance(scn, dict) else scn
        cfg = gym_config(proj, a.world or proj.doc['worlds'][0]['id'], os.path.join(a.out, 'env'), svc.port,
                         svc.tm_port, rig_id=a.rig or None, seed=int(e.get('seed', 0)), scenario=scn,
                         motion=e.get('motion', 'kinematic'), action=kind, log_actions=True,
                         workspace=proj.workspace, **kw)
        env = FalconDriveEnv(cfg, svc)
        em.progress(0.05, 'driving the episode again')
        doc = episode(e, env, pol, a.infraction)
    finally:
        if env is not None:
            env.close()
        svc.shutdown()
    doc['provenance'] = PV.block({'episodes.jsonl': a.episodes} | (
        {'policy': os.path.join(a.policy, 'policy.json')} if os.path.isdir(a.policy) else {}),
        e.get('seed'), {'episode': a.episode, 'infraction': a.infraction, 'policy': a.policy})
    os.makedirs(a.out, exist_ok=True)
    PV.write_json(os.path.join(a.out, 'rootcause.json'), doc)
    rec = PV.new_run('why', ['episode', a.episodes, str(a.episode)], e.get('seed'),
                     {'episode': a.episode, 'infraction': a.infraction, 'policy': a.policy},
                     {'episodes.jsonl': a.episodes})
    rec.update(status='ok', exit_code=0, finished=PV.utc_now(), timings={'wall_s': round(time.monotonic() - t0, 3)},
               outputs={'rootcause.json': PV.sha256_file(os.path.join(a.out, 'rootcause.json'))})
    PV.write_json(os.path.join(a.out, 'run.json'), rec)
    em.result('ok', path=os.path.join(os.path.abspath(a.out), 'rootcause.json'),
              summary={'status': doc['status'], 'runs': doc['n_runs'], 'summary': doc['summary']})
    return P.OK


# ================================================================================================ the report
def report(src: str):
    """The "rootcause" report (core.reports): a rootcause.json's folder (or the file) as a Page."""
    from . import reports as R
    path = src if src.endswith('.json') else os.path.join(src, 'rootcause.json')
    pg = R.Page('rootcause', 'Root cause', os.path.dirname(os.path.abspath(path)))
    d = pg.json(path)
    if d.get('schema') != SCHEMA:
        raise R.ReportError(f'{path} is not a root-cause attribution')
    esc, num = R.esc, R.num
    kind = d['kind']
    if kind == 'counterfactual':
        c = d['case']
        pg.title = f'Why did it fail? {c.get("scenario")}, {c.get("actor") or c.get("track_id") or "the run"}'
        pg.add(R.kv([('case', f'<span class="mono">{esc(c["run"])}</span>'), ('scenario', esc(c.get('scenario'))),
                     ('values', f'<span class="mono">{esc(json.dumps(c.get("values")))}</span>'),
                     ('failure test', esc(f'{c["failure"]} ≥ {c["threshold"]}' if c['failure'] not in
                                          ('missed', 'scenario') else c['failure'])),
                     ('target', esc(c.get('actor') or c.get('track_id'))), ('frame', esc(c.get('step'))),
                     ('model', esc((c.get('model') or {}).get('name') or '—')),
                     ('the case', esc(f'{d["base"]["detail"]} (value {d["base"]["value"]})')),
                     ('control re-run', esc(f'{"fails again" if d["control"]["fails"] else "does not fail"} '
                                            f'({d["control"]["detail"]}; {d["control"].get("repeat")})')),
                     ('re-runs', esc(d['n_runs']))]))
        pg.note(d['wording'])
        pg.add(f'<p><b>{esc(d.get("summary"))}</b></p>')
        pg.h2('Factors')
        rows = []
        for f in d['factors']:
            yn = {True: 'yes', False: 'no', None: '—'}
            rows.append([esc(f['title']), esc(f['in_case']),
                         esc(f['intervention'] if f.get('applicable') else (f.get('why_not') or 'not present')),
                         esc(yn[f.get('removing_fixes')]) + (f' ({esc(f["fixed_in"])})' if f.get('fixed_in') and
                                                             f['factor'] == 'appearance' else ''),
                         esc(yn[f.get('alone_fails')]), esc(f.get('verdict') or ('' if f['present'] else
                                                                                  'not present in this case')),
                         esc(f.get('caveat') or '')])
        pg.add(R.table(['factor', 'in this case', 'removed by', 'removing it alone fixes it (necessary)',
                        'it alone still fails (sufficient)', 'verdict', 'caveat'], rows))
        pg.h2('Re-runs')
        pg.add(R.table(['run', 'role', 'value', 'fails', 'detail', 'folder'], [
            [esc(r['label']), esc(r['role']), num(r.get('value')), esc({True: 'yes', False: 'no'}.get(r['fails'], '—')),
             esc(r.get('detail')), f'<span class="mono">{esc(r["run"])}</span>'] for r in d['runs']], numeric=(2,)))
    elif kind == 'static':
        m = d.get('model') or {}
        pg.title = f'Correlates of misses: {m.get("name")}'
        pg.add(R.kv([('evaluation', f'<span class="mono">{esc(d["evaluation"])}</span>'),
                     ('objects', esc(f'{d["n_objects"]} scored, {d["n_missed"]} missed ({d["miss_rate"]:.1%})')),
                     ('matching', esc(f'IoU ≥ {d["matching"]["iou"]}, score ≥ {d["matching"]["min_score"]}')),
                     ('intervals', esc(f'95 % percentile bootstrap over {d["n_units"]} {d["unit"]}s, '
                                       f'{d["n_boot"]} resamples, seed {d["seed"]}'))]))
        if d.get('synthetic_model'):
            pg.note('SYNTHETIC: the model reads the ground truth (a test model with planted weaknesses).')
        pg.note(d['wording'])
        pg.note(d.get('note'))
        pg.add(f'<p><b>{esc(d["summary"])}</b></p>')
        pg.h2('Spearman correlation with a miss')
        num_attrs = [a for a in d['attributes'] if a.get('spearman') is not None]
        pg.add(R.legend('estimate', None), R.interval_chart(
            [{'label': a['attribute'], 'before': {'value': a['spearman'], 'ci_low': a['ci_low'],
                                                 'ci_high': a['ci_high']}} for a in num_attrs], -1.0, 1.0,
            'Spearman correlation', names=('estimate', None)),
            R.table(['attribute', 'ρ', '95 % interval', 'objects', 'clear (interval excludes 0)'], [
                [esc(a['attribute']), num(a['spearman'], 2, True), f'[{num(a["ci_low"], 2, True)}, '
                 f'{num(a["ci_high"], 2, True)}]', esc(a['n']), esc('yes' if a['clear'] else 'no')]
                for a in num_attrs], numeric=(1, 2, 3)))
        for a in d['attributes']:
            if a.get('bins'):
                pg.add(f'<h3>Miss rate by {esc(a["attribute"])}</h3>', R.table(
                    ['bin', 'objects', 'missed', 'miss rate', '95 % interval'],
                    [[esc(f'{b["lo"]:g} – {b["hi"]:g}'), esc(b['n']), esc(b['misses']), num(b['miss_rate']),
                      f'[{num(b["ci_low"])}, {num(b["ci_high"])}]'] for b in a['bins']], numeric=(1, 2, 3, 4)))
            elif a.get('categories'):
                pg.add(f'<h3>Miss rate by {esc(a["attribute"])}</h3>', R.table(
                    ['value', 'objects', 'missed', 'miss rate', '95 % interval'],
                    [[esc(c['value']), esc(c['n']), esc(c['misses']), num(c['miss_rate']),
                      f'[{num(c["ci_low"])}, {num(c["ci_high"])}]'] for c in a['categories']], numeric=(1, 2, 3, 4)))
            elif a.get('note'):
                pg.add(f'<p class="muted">{esc(a["attribute"])}: {esc(a["note"])}</p>')
    else:
        e = d['episode']
        pg.title = f'Why did the policy fail? episode {e.get("episode")} ({(e.get("infraction") or {}).get("kind")})'
        pg.add(R.kv([('policy', f'<span class="mono">{esc(d.get("policy"))}</span>'),
                     ('episode', esc(f'{e.get("episode")}, seed {e.get("seed")}, {e.get("scenario") or "free"}')),
                     ('infraction', esc(json.dumps(e.get('infraction')))),
                     ('control re-run', esc(f'infraction {"again" if d["control"]["infraction_recurs"] else "not met"}'
                                            f', same actions: {d["control"]["same_actions"]}')),
                     ('re-runs', esc(d['n_runs']))]))
        pg.note(d['wording'])
        pg.add(f'<p><b>{esc(d["summary"])}</b></p>')
        pg.add(R.table(['input group', 'features', 'hidden as', 'removing it prevents the infraction'], [
            [esc(g['title']), f'<span class="mono">{esc(", ".join(g.get("features") or []))}</span>',
             esc(g.get('intervention') or g.get('why_not')),
             esc({True: 'yes', False: 'no'}.get(g.get('removing_fixes'), '—'))] for g in d['groups']]))
        pg.note(f'Sufficiency: {d.get("sufficiency")}.')
    pg.provenance(d.get('provenance'), 'the attribution')
    return pg
