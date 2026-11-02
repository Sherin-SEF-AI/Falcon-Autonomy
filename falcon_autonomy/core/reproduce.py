"""Comparing a re-run with the original (`falcon reproduce RUN`): which outputs agree, within which tolerance.

A drive is re-run from its own drive.json (the same seed, settings, map and rig) into a new folder, then compared:

* frames.jsonl: the same steps and frames; the ego's pose and speed within pose_tol (metres, degrees, m/s);
* semantic (and instance) labels: the share of pixels that are identical, at least label_agree;
* lidar sweeps and radar detections: the largest difference of any value, at most pose_tol;
* object files: the same objects (track ids, classes, tight boxes);
* RGB images: the mean absolute difference in 8-bit levels at most rgb_tol (JPEG and Unreal's temporal effects make
  bit-exact images unlikely even when the scene is identical).

Bit-for-bit reproduction needs the same server build, the same map and synchronous mode from a fresh load; the
report says what differed rather than a single yes/no.

An experiment is re-run from its own definition and seeds (`falcon experiment ablation rerun`) and compared with
compare_experiments: the same planned runs and seeds, each run's metric, each contrast's difference and interval, the
verdicts — and every pin that changed between the two (Falcon's version, the CARLA build, the hashes of the worlds,
rigs, traffic profiles and model weights).
"""
from __future__ import annotations

import json
import math
import os

DEFAULT_TOL = {'pose_tol': 1e-3, 'rgb_tol': 2.0, 'label_agree': 0.999}


def _frames(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _objects_key(doc) -> list:
    objs = doc.get('objects', doc) if isinstance(doc, dict) else doc
    return sorted((str(o.get('track_id')), o.get('class'), tuple(o.get('bbox') or ())) for o in objs)


def compare_drives(a_dir: str, b_dir: str, tol: dict | None = None, max_images: int = 50) -> dict:
    import numpy as np
    import cv2
    t = dict(DEFAULT_TOL, **(tol or {}))
    fa, fb = _frames(os.path.join(a_dir, 'frames.jsonl')), _frames(os.path.join(b_dir, 'frames.jsonl'))
    rep = {'tolerance': t, 'steps': [len(fa), len(fb)], 'checks': []}

    def check(name, ok, detail):
        rep['checks'].append({'name': name, 'ok': bool(ok), 'detail': detail})
    check('step count', len(fa) == len(fb), f'{len(fa)} vs {len(fb)}')
    n = min(len(fa), len(fb))
    worst = 0.0
    for ra, rb in zip(fa[:n], fb[:n]):
        for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
            worst = max(worst, abs(ra['ego'][k] - rb['ego'][k]))
        worst = max(worst, abs(ra.get('speed_mps', 0) - rb.get('speed_mps', 0)))
    check('ego trajectory', worst <= t['pose_tol'], f'largest difference {worst:.6g} (tolerance {t["pose_tol"]:g})')
    stride = max(n // max(max_images, 1), 1)
    rgb_worst, label_worst = 0.0, 1.0
    rgb_n = label_n = obj_n = obj_diff = arr_n = 0
    arr_worst = 0.0
    for i in range(0, n, stride):
        for key, rel in fa[i]['files'].items():
            rel_b = fb[i]['files'].get(key)
            if rel_b is None:
                check(f'file {key} step {i}', False, 'missing in the re-run')
                continue
            if rel.endswith('.json'):
                with open(os.path.join(a_dir, rel), encoding='utf-8') as fa_, \
                        open(os.path.join(b_dir, rel_b), encoding='utf-8') as fb_:
                    oa, ob = json.load(fa_), json.load(fb_)
                obj_n += 1
                if _objects_key(oa) != _objects_key(ob):
                    obj_diff += 1
                continue
            if rel.endswith(('.bin', '.npy')):
                load = np.load if rel.endswith('.npy') else (lambda p: np.fromfile(p, np.float32))
                xa, xb = load(os.path.join(a_dir, rel)), load(os.path.join(b_dir, rel_b))
                arr_n += 1
                arr_worst = max(arr_worst, float(np.max(np.abs(xa - xb))) if xa.shape == xb.shape and xa.size
                                else (0.0 if xa.shape == xb.shape else float('inf')))
                continue
            ia = cv2.imread(os.path.join(a_dir, rel), cv2.IMREAD_UNCHANGED)
            ib = cv2.imread(os.path.join(b_dir, rel_b), cv2.IMREAD_UNCHANGED)
            if ia is None or ib is None or ia.shape != ib.shape:
                check(f'file {key} step {i}', False, 'unreadable or a different size')
                continue
            if key.endswith('/rgb'):
                rgb_worst = max(rgb_worst, float(np.mean(np.abs(ia.astype(np.int16) - ib.astype(np.int16)))))
                rgb_n += 1
            else:
                same = ia == ib
                agree = float(np.mean(same.all(axis=-1) if same.ndim == 3 else same))
                label_worst = min(label_worst, agree)
                label_n += 1
    if rgb_n:
        check('RGB images', rgb_worst <= t['rgb_tol'], f'{rgb_n} compared, worst mean difference {rgb_worst:.3f} '
                                                       f'levels (tolerance {t["rgb_tol"]:g})')
    if label_n:
        check('label images', label_worst >= t['label_agree'], f'{label_n} compared, worst agreement '
                                                               f'{100 * label_worst:.3f} % (needed '
                                                               f'{100 * t["label_agree"]:.1f} %)')
    if arr_n:
        check('point clouds and radar', arr_worst <= t['pose_tol'], f'{arr_n} compared, largest difference '
                                                                   f'{arr_worst:.6g} (tolerance {t["pose_tol"]:g})')
    if obj_n:
        check('objects', obj_diff == 0, f'{obj_n} object files compared, {obj_diff} differ (ids, classes, tight '
                                        'boxes)')
    rep['ok'] = all(c['ok'] for c in rep['checks'])
    return rep


EXPERIMENT_TOL = {'metric_tol': 1e-9, 'effect_tol': 1e-9}


def compare_experiments(a_dir: str, b_dir: str, tol: dict | None = None) -> dict:
    """An experiment's re-run against the original (core.ablation.rerun, `falcon experiment ablation rerun`): the
    same planned runs, each run's metric within metric_tol, each contrast's difference and interval within
    effect_tol, the same verdicts; and every pin that changed (core.provenance.diff_pins: Falcon's version, the CARLA
    build, the world, rig, traffic and weights hashes, the seeds) — at the experiment's level and per run. The
    default tolerances ask for the same numbers: a deterministic executor (the fake simulator, the simulated
    executor) reproduces them exactly; on a real server the drives' images can differ slightly (see compare_drives),
    so give the tolerance the comparison needs."""
    from . import provenance as PV
    from .experiments import Experiment
    t = dict(EXPERIMENT_TOL, **(tol or {}))
    ea, eb = Experiment(a_dir), Experiment(b_dir)
    rep = {'tolerance': t, 'original': ea.id, 'rerun': eb.id, 'checks': [], 'pins_changed': [], 'differing': []}

    def check(name, ok, detail):
        rep['checks'].append({'name': name, 'ok': bool(ok), 'detail': detail})

    def rd(d, f):
        p = os.path.join(d, f)
        return PV.read_json(p) if os.path.isfile(p) else {}
    pa, pb = rd(a_dir, 'plan.json'), rd(b_dir, 'plan.json')
    names_a, names_b = [r['name'] for r in pa.get('runs') or ()], [r['name'] for r in pb.get('runs') or ()]
    seeds_ok = [(r['name'], r['seed']) for r in pa.get('runs') or ()] == [(r['name'], r['seed'])
                                                                          for r in pb.get('runs') or ()]
    check('planned runs', names_a == names_b and seeds_ok, f'{len(names_a)} vs {len(names_b)} runs; seeds '
                                                           f'{"the same" if seeds_ok else "differ"}')
    rep['pins_changed'] = [d for d in PV.diff_pins(pa.get('pins') or {}, pb.get('pins') or {})
                           if d['key'] != 'spec_sha256']
    metric = ((ea.doc.get('definition') or {}).get('spec') or {}).get('metric')
    worst, n, missing, run_pins = 0.0, 0, [], {}
    for name in names_a:
        ra, rb = ea.run(name), eb.run(name)
        if ra.status != 'ok' or rb.status != 'ok':
            missing.append(name)
            continue
        va, vb = ra.latest().get(metric), rb.latest().get(metric)
        if va is None or vb is None:
            if (va is None) != (vb is None):
                missing.append(name)
            continue
        dv = abs(float(va) - float(vb))
        if dv > t['metric_tol'] and len(rep['differing']) < 50:          # the first 50 that differ
            rep['differing'].append({'run': name, 'original': va, 'rerun': vb, 'diff': dv})
        worst = max(worst, dv)
        n += 1
        for d in PV.diff_pins(ra.info().get('pins') or {}, rb.info().get('pins') or {}):
            if d['key'] != 'spec_sha256':
                run_pins.setdefault(d['key'], dict(d, run=name, runs=0))['runs'] += 1
    check('runs finished in both', not missing, f'{len(missing)} not finished in one of them' if missing else
          f'{n} compared')
    check(f'each run\'s {metric}', worst <= t['metric_tol'], f'largest difference {worst:.6g} (tolerance '
                                                             f'{t["metric_tol"]:g})')
    # the runs' own pins too (the CARLA build a drive read, the drive's key, the weights): the first run that differs
    rep['pins_changed'] += [dict(v, key=f'runs.{k}') for k, v in sorted(run_pins.items())]
    ra_, rb_ = rd(a_dir, 'results.json'), rd(b_dir, 'results.json')
    if ra_ and rb_:
        ka = {c['id']: c for c in ra_.get('contrasts') or ()}
        worst_e, verd = 0.0, []
        for c in rb_.get('contrasts') or ():
            o = ka.get(c['id'])
            if o is None:
                verd.append(c['id'])
                continue
            for k in ('diff', 'ci_low', 'ci_high'):
                if (o.get(k) is None) != (c.get(k) is None):
                    worst_e = math.inf
                elif o.get(k) is not None:
                    worst_e = max(worst_e, abs(o[k] - c[k]))
            if o.get('verdict') != c.get('verdict'):
                verd.append(c['id'])
        check('effects and intervals', worst_e <= t['effect_tol'], f'largest difference {worst_e:.6g} (tolerance '
                                                                   f'{t["effect_tol"]:g})')
        check('verdicts', not verd and ra_.get('verdict') == rb_.get('verdict'),
              f'hypothesis {ra_.get("verdict")} vs {rb_.get("verdict")}'
              + (f'; contrasts differing: {", ".join(verd)}' if verd else ''))
    else:
        check('analysis', False, 'results.json missing in ' + ('the original' if not ra_ else 'the re-run'))
    rep['ok'] = all(c['ok'] for c in rep['checks'])
    return rep
