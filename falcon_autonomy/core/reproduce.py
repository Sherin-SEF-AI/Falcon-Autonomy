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
"""
from __future__ import annotations

import json
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
