"""Dataset health: what a set of recorded drives holds and lacks (docs/runs.md, "Dataset health").

For a set of runs, measured from their files:

* **Class balance** — objects per class in the camera frames' labels (objects/<camera>/<step>.json), with the share
  of each class and of frames with no labelled object.
* **Condition coverage** — frames per world × time of day × weather kind (core.runindex's buckets: day, low sun,
  twilight, night from the sun's altitude; clear, cloudy, wet, rain from the weather parameters), the empty cells of
  that grid for the project's worlds, and each scenario parameter's values over its range in ten bins.
* **Near-duplicate frames** — a 64-bit difference hash (dHash: 9 × 8 grey thumbnail, one bit per horizontal
  gradient sign) per sampled RGB frame; a frame within DUP_BITS bits of an earlier kept frame of the set is a near
  duplicate (a greedy pass in recording order). It flags a stopped car's run of identical frames, not similar scenes.
* **Label sanity** — boxes that are degenerate (no area), outside the image, or tiny (under TINY_H px high or
  TINY_AREA px²), and frames with no labelled object.
* **Suggestions** — the empty or thinnest coverage cells as "record more of" items, each with the CARLA preset that
  gives it (Record ▸ Weather).

Frames are sampled evenly (at most max_frames over the set); the numbers say how many were read. Nothing is
extrapolated to the frames not read.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import runindex as RI

SCHEMA = 'falcon.data_health/1'
DUP_BITS = 4
TINY_H, TINY_AREA = 8.0, 64.0
VALUE_BINS = 10
GRID_TODS = ('day', 'low sun', 'night')          # the times of day a CARLA preset gives (twilight has none)
PRESET_FOR = {('day', 'clear'): 'ClearNoon', ('day', 'cloudy'): 'CloudyNoon', ('day', 'wet'): 'WetNoon',
              ('day', 'rain'): 'SoftRainNoon', ('low sun', 'clear'): 'ClearSunset',
              ('low sun', 'cloudy'): 'CloudySunset', ('low sun', 'wet'): 'WetSunset',
              ('low sun', 'rain'): 'SoftRainSunset', ('night', 'clear'): 'ClearNight',
              ('night', 'cloudy'): 'CloudyNight', ('night', 'wet'): 'WetNight', ('night', 'rain'): 'SoftRainNight'}


def dhash(grey: np.ndarray) -> int:
    """64-bit difference hash of a greyscale image (any size)."""
    import cv2
    small = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


_POP = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def hamming(h: int, arr: np.ndarray) -> np.ndarray:
    """Bit differences between one hash and an array of uint64 hashes."""
    x = np.bitwise_xor(arr, np.uint64(h)).view(np.uint8).reshape(-1, 8)
    return _POP[x].sum(axis=1)


def _frames(run_dir: str) -> list:
    p = os.path.join(run_dir, 'frames.jsonl')
    out = []
    try:
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _sizes(run_dir: str) -> dict:
    try:
        with open(os.path.join(run_dir, 'dataset.json'), encoding='utf-8') as fh:
            return {s['id']: (s.get('width'), s.get('height')) for s in json.load(fh).get('sensors') or ()}
    except (OSError, ValueError, KeyError):
        return {}


def analyse(run_dirs: list, workspace: str = '', project_doc: dict | None = None, max_frames: int = 3000,
            dup_bits: int = DUP_BITS, progress=None) -> dict:
    """The health report of these runs (see the module doc). progress(fraction, text) is called as frames are read."""
    import cv2
    entries = {e['path']: e for e in RI.build(workspace, extra=run_dirs)} if workspace else \
        {os.path.abspath(d): RI.summarise(d) for d in run_dirs}
    runs = []
    for d in run_dirs:
        e = entries.get(os.path.abspath(d))
        if e is not None:
            runs.append((os.path.abspath(d), e))
    per_run = [(d, e, [f for f in _frames(d) if f.get('files')]) for d, e in runs]
    total = sum(len(fr) for _, _, fr in per_run)
    step = max(1, int(np.ceil(total / max(max_frames, 1))))
    classes: dict = {}
    n_obj = n_lab_frames = n_empty = 0
    tiny = degenerate = outside = 0
    tiny_ex, empty_ex = [], []
    hashes, dup_ex = [], []
    n_hashed = n_dup = 0
    grid: dict = {}
    values: dict = {}
    read = k = 0
    for d, e, frames in per_run:
        cell = (e.get('world') or e.get('map') or '?', e.get('tod') or '?', e.get('weather_kind') or '?')
        grid[cell] = grid.get(cell, 0) + len(frames)
        for name, v in (e.get('values') or {}).items():
            if isinstance(v, (int, float)):
                values.setdefault((e.get('scenario') or '?', name), []).append(float(v))
        sizes = _sizes(d)
        for i, f in enumerate(frames):
            k += 1
            if (k - 1) % step:
                continue
            read += 1
            if progress is not None and read % 50 == 0:
                progress(min(read / max(total / step, 1), 1.0), f'{read} frames read')
            rgb_key = next((x for x in sorted(f['files']) if x.endswith('/rgb')), None)
            for key, rel in sorted(f['files'].items()):
                if not key.endswith('/objects'):
                    continue
                cam = key.split('/')[0]
                try:
                    with open(os.path.join(d, rel), encoding='utf-8') as fh:
                        objs = json.load(fh).get('objects') or []
                except (OSError, ValueError):
                    continue
                n_lab_frames += 1
                if not objs:
                    n_empty += 1
                    if len(empty_ex) < 5:
                        empty_ex.append({'run': d, 'step': f.get('step'), 'camera': cam})
                W, H = sizes.get(cam, (None, None))
                for o in objs:
                    n_obj += 1
                    c = o.get('class') or '?'
                    classes[c] = classes.get(c, 0) + 1
                    b = o.get('bbox') or [0, 0, 0, 0]
                    w_, h_ = float(b[2]) - float(b[0]), float(b[3]) - float(b[1])
                    if w_ <= 0 or h_ <= 0:
                        degenerate += 1
                    elif W and H and (b[2] <= 0 or b[3] <= 0 or b[0] >= W or b[1] >= H):
                        outside += 1
                    elif h_ < TINY_H or w_ * h_ < TINY_AREA:
                        tiny += 1
                        if len(tiny_ex) < 5:
                            tiny_ex.append({'run': d, 'step': f.get('step'), 'camera': cam, 'class': c,
                                            'bbox': [round(float(x), 1) for x in b]})
            if rgb_key is not None:
                g = cv2.imread(os.path.join(d, f['files'][rgb_key]), cv2.IMREAD_REDUCED_GRAYSCALE_4)
                if g is None:
                    continue
                h = dhash(g)
                n_hashed += 1
                if hashes:
                    dist = hamming(h, np.array([x for x, _ in hashes], np.uint64))
                    j = int(np.argmin(dist))
                    if dist[j] <= dup_bits:
                        n_dup += 1
                        if len(dup_ex) < 6:
                            dup_ex.append({'run': d, 'step': f.get('step'), 'like': hashes[j][1],
                                           'bits': int(dist[j])})
                        continue
                hashes.append((h, {'run': d, 'step': f.get('step')}))
    cov = coverage(grid, project_doc)
    hist = []
    for (scn, name), vs in sorted(values.items()):
        lo, hi = min(vs), max(vs)
        pr = _param_range(project_doc, scn, name)
        if pr is not None:
            lo, hi = pr
        counts = np.histogram(vs, bins=VALUE_BINS, range=(lo, hi if hi > lo else lo + 1e-9))[0].tolist()
        hist.append({'scenario': scn, 'param': name, 'lo': lo, 'hi': hi, 'counts': counts, 'runs': len(vs),
                     'empty_bins': sum(1 for c in counts if c == 0), 'range_from': 'the scenario' if pr else
                     'the values seen'})
    return {
        'schema': SCHEMA, 'runs': [d for d, _ in runs], 'frames': total, 'frames_read': read, 'every': step,
        'classes': dict(sorted(classes.items(), key=lambda kv: -kv[1])), 'objects': n_obj,
        'labelled_frames': n_lab_frames, 'empty_frames': n_empty, 'empty_examples': empty_ex,
        'labels': {'tiny': tiny, 'degenerate': degenerate, 'outside': outside, 'tiny_examples': tiny_ex,
                   'tiny_rule': f'height < {TINY_H:g} px or area < {TINY_AREA:g} px²'},
        'duplicates': {'hashed': n_hashed, 'near_duplicates': n_dup,
                       'share': (n_dup / n_hashed) if n_hashed else None, 'bits': dup_bits, 'examples': dup_ex},
        'coverage': {k: cov[k] for k in ('cells', 'empty', 'of')},
        'values': hist, 'suggestions': cov['suggestions'],
        'method': 'docs/runs.md, "Dataset health"; frames sampled evenly, every '
                  f'{step}{"st" if step == 1 else "th"} frame',
    }


def coverage(grid: dict, project_doc: dict | None = None) -> dict:
    """The condition-coverage grid of {(world, tod, weather kind): frames} over the project's worlds (or the ones
    seen): {'cells': every cell with its frames and the preset that gives it (then any cell outside the grid, preset
    None), 'empty': how many grid cells have no frame, 'of': the grid's size, 'suggestions': the empty cells (worlds
    with frames first, then by time of day and weather) and the three thinnest, at most 8, as "record more of" items}.
    analyse() fills it from the frames it reads; the assistant's suggestions from the run index (no image read)."""
    worlds = [w['id'] for w in (project_doc or {}).get('worlds') or ()] or sorted({c[0] for c in grid})
    cells = []
    for w in worlds:
        for t in GRID_TODS:
            for wk in RI.WEATHER_KINDS:
                cells.append({'world': w, 'tod': t, 'weather': wk, 'frames': grid.get((w, t, wk), 0),
                              'preset': PRESET_FOR[(t, wk)]})
    extra = [{'world': w, 'tod': t, 'weather': wk, 'frames': n, 'preset': None}
             for (w, t, wk), n in grid.items() if w in worlds and (t not in GRID_TODS or wk not in RI.WEATHER_KINDS)]
    empty = [c for c in cells if c['frames'] == 0]
    names = {w['id']: w['name'] for w in (project_doc or {}).get('worlds') or ()}
    thin = sorted((c for c in cells if c['frames'] > 0), key=lambda c: c['frames'])
    have_worlds = {c['world'] for c in cells if c['frames'] > 0}
    sugg = []
    for c in sorted(empty, key=lambda c: (c['world'] not in have_worlds, GRID_TODS.index(c['tod']),
                                          RI.WEATHER_KINDS.index(c['weather']))) + thin[:3]:
        if len(sugg) >= 8:
            break
        sugg.append({'world': c['world'], 'world_name': names.get(c['world'], c['world']), 'tod': c['tod'],
                     'weather': c['weather'], 'preset': c['preset'], 'frames': c['frames'],
                     'text': f'{names.get(c["world"], c["world"])}: {c["tod"]}, {c["weather"]} — '
                             + ('no frames' if c['frames'] == 0 else f'only {c["frames"]} frames')})
    return {'cells': cells + extra, 'empty': len(empty), 'of': len(cells), 'suggestions': sugg}


def _param_range(doc, scn_id, name):
    from . import scenarios as SC
    scn = next((s for s in (doc or {}).get('scenarios') or () if s.get('id') == scn_id), None)
    if scn is None and scn_id in SC.LIBRARY:
        scn = SC.builtin(scn_id)
    p = ((scn or {}).get('params') or {}).get(name) or {}
    if 'lo' in p and 'hi' in p:
        return float(p['lo']), float(p['hi'])
    return None
