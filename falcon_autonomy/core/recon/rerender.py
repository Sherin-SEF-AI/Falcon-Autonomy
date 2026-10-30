"""Re-render a recorded drive from a changed ego path or camera: a falcon-raw-like run of the reconstruction's images
(torch, numpy, OpenCV).

    rerender('…/recon/drive', '…/runs/drive-shift-1m', lateral_m=1.0)

For every frame of the source (or every n-th; or only the held-out ones), the ego is moved in its own frame —
lateral_m to the right, forward_m, up_m, yaw_deg (CARLA's sense: positive turns right) — and/or the camera's mount is
replaced (mount: a CARLA vehicle-frame pose, as a rig's) and/or its intrinsics (fov and size: a centred pinhole). The
reconstruction is rendered there. The run folder holds:

* rgb/<camera>/<step>.jpg — the rendered images;
* range/<camera>/<step>.png — uint16, round(range × 256) along each pixel's ray, 0 = nothing (falcon-raw's encoding)
  — **the reconstruction's depth** (core.recon.scene), not a simulator's ground truth;
* semantic/<camera>/<step>.png — only when the reconstruction voted classes (a Falcon source with semantic labels):
  each pixel's class from the Gaussians' votes, **an estimate**; 0 where the alpha is below ½;
* frames.jsonl (the shifted ego in CARLA's convention of the source's world: a Falcon run's own CARLA world; KITTI's
  and nuScenes' worlds mirrored as log replay does), dataset.json, drive.json, run.json; lanes.json copied from a
  Falcon source (the map's lanes do not move with the ego).

Not available, and said in dataset.json, run.json and docs/reconstruction.md: instance, objects (2D/3D boxes), optical
flow, lidar, radar, actors.jsonl. The reconstruction is of the **static scene**: moving actors were masked out of
training, so none is drawn (where a mask missed one, a blurred ghost of it may be). A detector therefore cannot be
scored on these runs (the harness needs objects); a segmenter can be, against the voted semantic classes, and any
model can be run on them for its predictions. A run with objects is a hybrid world (core.recon.hybrid): CARLA drives
and renders the actors at the camera's poses and they are composited over the reconstruction, CARLA's labels kept.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import numpy as np

SCHEMA = 'falcon.dataset/1'
NOT_AVAILABLE = ('instance', 'objects', 'flow', 'lidar', 'radar', 'actors.jsonl')


class _Quiet:
    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass


def rerender(recon, out_dir: str, lateral_m=0.0, forward_m=0.0, up_m=0.0, yaw_deg=0.0, mount: dict | None = None,
             fov: float | None = None, size=None, every: int = 1, split: str = 'all', emitter=None, cancel=None,
             argv=None, device: str = 'auto') -> dict:
    """Write the run (module doc); returns its summary. recon: a core.recon.scene.Recon or its folder."""
    import cv2
    from .. import provenance as PV
    from ..paths import ensure_dir
    from .scene import Recon, carla_pose_of
    em = emitter or _Quiet()
    rec = recon if isinstance(recon, Recon) else Recon.load(recon, device)
    frames = rec.frames if split == 'all' else rec.split(split)
    frames = frames[::max(1, int(every))]
    if not frames:
        raise ValueError(f'no frames to render (split {split!r})')
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise ValueError(f'{out_dir} exists and is not empty')
    out_dir = ensure_dir(os.path.abspath(out_dir))
    src = rec.views['source']
    cid = src.get('camera') or 'camera'
    cam_id = cid if src['kind'] == 'falcon' else f'recon-{cid}'.replace('/', '-')
    shift = {'lateral_m': float(lateral_m), 'forward_m': float(forward_m), 'up_m': float(up_m),
             'yaw_deg': float(yaw_deg), 'mount': mount, 'fov': fov, 'size': list(size) if size else None}
    run = PV.new_run('recon.rerender', argv or [], None, shift, {'recon/run.json': os.path.join(rec.path, 'run.json'),
                                                                 'recon/point_cloud.ply':
                                                                     os.path.join(rec.path, 'point_cloud.ply')})
    votes = rec.gs.votes is not None
    for d in ('rgb', 'range') + (('semantic',) if votes else ()):
        os.makedirs(os.path.join(out_dir, d, cam_id), exist_ok=True)
    rows, egos = [], []
    t0 = time.perf_counter()
    cam0 = None
    for n, f in enumerate(frames):
        if cancel and cancel():
            raise KeyboardInterrupt
        cam = rec.camera(f, lateral_m, forward_m, up_m, yaw_deg, mount, fov, size)
        cam0 = cam0 or cam
        out = rec.render(cam, semantic=votes)
        files = {}
        rel = f'rgb/{cam_id}/{n:06d}.jpg'
        cv2.imwrite(os.path.join(out_dir, rel), cv2.cvtColor(out['image'], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        files[f'{cam_id}/rgb'] = rel
        rel = f'range/{cam_id}/{n:06d}.png'
        r16 = np.where(out['range'] < 256.0, np.round(out['range'] * 256.0), 0).astype(np.uint16)
        cv2.imwrite(os.path.join(out_dir, rel), r16)
        files[f'{cam_id}/range'] = rel
        if votes:
            rel = f'semantic/{cam_id}/{n:06d}.png'
            cv2.imwrite(os.path.join(out_dir, rel), out['semantic'])
            files[f'{cam_id}/semantic'] = rel
        ego = carla_pose_of(cam['ego'])
        egos.append((float(f['t']), ego))
        rows.append({'step': n, 'frame': int(f['index']), 'sim_t': round(float(f['t']), 6), 'files': files,
                     'ego': ego, 'recorded': True, 'control': None, 'source_split': f['split']})
        em.progress((n + 1) / len(frames), f'frame {n + 1} of {len(frames)}')
    t_render = time.perf_counter() - t0
    for k, row in enumerate(rows):          # speed from the shifted positions (central differences)
        a, b = max(0, k - 1), min(len(rows) - 1, k + 1)
        dt = egos[b][0] - egos[a][0]
        dx = math.hypot(egos[b][1]['x'] - egos[a][1]['x'], egos[b][1]['y'] - egos[a][1]['y'])
        row['speed_mps'] = round(dx / dt, 4) if dt > 0 else 0.0
    with open(os.path.join(out_dir, 'frames.jsonl'), 'w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(',', ':')) + '\n')
    w, h = cam0['size']
    K = cam0['K']
    fov_h = math.degrees(2 * math.atan(w / (2 * K[0])))
    centred = abs(K[2] - w / 2) < 1e-6 and abs(K[3] - h / 2) < 1e-6 and abs(K[0] - K[1]) < 1e-9
    sensor = {'id': cam_id, 'name': cam_id, 'type': 'camera.pinhole', 'width': int(w), 'height': int(h),
              'fov': round(fov_h, 6), 'pose': carla_pose_of(cam0['mount'], optical=True),
              'labels': ['depth'] + (['semantic'] if votes else [])}
    src_ds, src_drive = {}, {}
    if src['kind'] == 'falcon':
        for name, store in (('dataset.json', src_ds), ('drive.json', src_drive)):
            p = os.path.join(src['path'], name)
            if os.path.isfile(p):
                store.update(PV.read_json(p))
        if os.path.isfile(os.path.join(src['path'], 'lanes.json')):
            shutil.copyfile(os.path.join(src['path'], 'lanes.json'), os.path.join(out_dir, 'lanes.json'))
    times = [r['sim_t'] for r in rows]
    fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 10.0
    what = {'from': 'a 3D Gaussian Splatting reconstruction of the static scene (core.recon)',
            'reconstruction': rec.path, 'shift': shift,
            'labels': {'rgb': 'rendered', 'range': 'the reconstruction\'s depth (not ground truth)',
                       'semantic': 'voted from the source\'s semantic labels (an estimate)' if votes else
                       'not available (the source had no semantic labels)'},
            'not_available': list(NOT_AVAILABLE) + ([] if votes else ['semantic']),
            'static_scene': 'moving actors were masked out of training and are not drawn',
            'world': rec.views.get('world'), 'carla_exact': bool(rec.views.get('carla_exact'))}
    labels = {'range': {'encoding': 'uint16 PNG, round(range_m × 256) along each pixel\'s ray, 0 = nothing; the '
                                    'reconstruction\'s depth, not ground truth'}}
    if votes:
        labels['semantic'] = dict((src_ds.get('labels') or {}).get('semantic') or
                                  {'encoding': 'uint8 PNG, a class per pixel (CARLA 0.9.16 semantic tags)'},
                                  note='voted from the source\'s labels by the Gaussians (an estimate)')
    ds = {'schema': SCHEMA, 'layout': 'falcon-raw', 'name': f'{os.path.basename(rec.path)} re-rendered',
          'frames': len(rows), 'fps': round(fps, 6),
          'sensors': [dict(sensor, K=[[K[0], 0.0, K[2]], [0.0, K[1], K[3]], [0.0, 0.0, 1.0]], views=1,
                           lens={'model': 'rectilinear', 'f_px': K[0], 'k': [0.0] * 4, 'cx': K[2], 'cy': K[3],
                                 'aspect': K[1] / K[0], 'max_theta_deg': 89.0, 'width': int(w), 'height': int(h),
                                 'f_heuristic': False, 'centred': centred},
                           streams=sorted(['rgb', 'range'] + (['semantic'] if votes else [])))],
          'labels': labels, 'coordinates': 'CARLA\'s frames: x forward, y right, z up (left-handed); metres, degrees',
          'reconstruction': what, 'town': src_ds.get('town') or rec.views.get('town'),
          'weather': src_ds.get('weather') or src_drive.get('weather')}
    drive = dict({k: v for k, v in src_drive.items() if k in ('town', 'xodr', 'opendrive', 'handedness', 'weather',
                                                              'weather_source', 'fps', 'seed')},
                 schema='falcon.drive/1', name=ds['name'], rig={'id': 'recon', 'name': 'reconstruction camera',
                                                                'vehicle': (src_drive.get('rig') or {})
                                                                .get('vehicle'), 'sensors': [sensor]},
                 reconstruction=what, out=out_dir)
    drive.setdefault('fps', round(fps, 6))
    PV.write_json(os.path.join(out_dir, 'dataset.json'), ds)
    PV.write_json(os.path.join(out_dir, 'drive.json'), drive)
    run.update(status='ok', exit_code=0, finished=PV.utc_now(), name=ds['name'], reconstruction=what,
               timings={'render_s': round(t_render, 2), 'frames_per_s': round(len(rows) / max(t_render, 1e-9), 3)},
               notes=[f'labels not available: {", ".join(what["not_available"])}',
                      'range (and semantic) are the reconstruction\'s estimates, not ground truth',
                      'the static scene only: moving actors are not drawn'])
    run['outputs'] = PV.hash_tree(out_dir, skip=('run.json',))
    PV.write_json(os.path.join(out_dir, 'run.json'), run)
    return {'run': out_dir, 'frames': len(rows), 'camera': cam_id, 'labels': sorted(ds['sensors'][0]['streams']),
            'not_available': what['not_available'], 'render_s': round(t_render, 2)}
