"""End-to-end driving: a camera (+ BEV) to control policy learnt by behaviour cloning from recorded drives (numpy at
import; OpenCV and torch only inside functions). The result is a policy folder (core.policy, net kind "e2e") that the
environment runs with its camera observation — `falcon policy rollout` and the Leaderboard open the environment with
the policy's own camera — so it is evaluated offline (action error by drive) and online (Leaderboard-style scores)
like every other policy.

Pairs. The same (step, next control) pairs as behaviour cloning (core.imitation.drive_samples: steps whose control
and next control were recorded; the action is the next step's control as [steer, throttle − brake]), each with:
* the camera's rgb at that step (frames.jsonl files[<camera>/rgb]; default the run's first camera), read as RGB and
  resized to width × height (default 128 × 72, the environment's default camera) with area interpolation when the
  drive recorded another size;
* a few state features (default speed_mps and target_speed_mps: the "measurement" and the "command", as conditional
  imitation learning gives them — Codevilla et al. 2018 — without which a camera cannot know the speed it is asked
  for); the prev_* features are left out on purpose (the copycat shortcut, core.imitation);
* optionally the BEV (core.drivefeatures' raster at the environment's spec), for a camera + BEV policy.
The camera's pose and field of view are the drive's rig's (drive.json); drives with different camera poses or
fields of view are refused together, since the environment can open one camera.

Network: core.policy.E2ENet — PilotNet's convolutions (Bojarski et al. 2016, "End to End Learning for Self-Driving
Cars") over the image, an MLP head with the state features (and the BEV's convolutions), tanh outputs. Training: MSE
on both components, Adam, minibatches in a seeded order, torch.manual_seed(seed); no augmentation; the split is BY
DRIVE (val_fraction of the drives, at least one); the epoch with the lowest validation loss is kept (as behaviour
cloning does: the validation drives choose the epoch, so their error is slightly optimistic — `falcon worldmodel
e2e-offline` on drives not used at all is the clean number). Metrics on both splits: mean absolute and RMS error per
component, for the policy and the constant baseline (the training split's mean action), and per validation drive.

Limits: cloning copies the recording driver, faults included, and errors compound in closed loop (covariate shift);
a camera policy trained on one town's look meets another look elsewhere; the fake carla's images (tests) are flat
renderings, not CARLA's.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import drivefeatures as DF

E2E_DEFAULTS = {
    'epochs': 30, 'batch': 64, 'lr': 3e-4, 'weight_decay': 1e-5, 'hidden': [128, 64], 'activation': 'relu',
    'conv': [24, 36, 48, 64, 64], 'bev_conv': [16, 32, 32], 'val_fraction': 0.25, 'seed': 0,
    'state_features': ['speed_mps', 'target_speed_mps'], 'bev': False, 'bev_spec': None, 'camera': '',
    'width': 128, 'height': 72, 'every': 1, 'device': 'auto', 'target_speed_kmh': None, 'name': '',
}


def _camera_of(run_dir: str, cam: str) -> tuple:
    """(camera id, its rig sensor dict or None) of a drive."""
    cfg = {}
    for name in ('drive.json', 'live.json'):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as fh:
                cfg = json.load(fh)
            break
    sensors = [s for s in (cfg.get('rig') or {}).get('sensors', []) if str(s.get('type', '')).startswith('camera')]
    if not cam:
        with open(os.path.join(run_dir, 'frames.jsonl'), encoding='utf-8') as fh:
            f0 = json.loads(fh.readline())
        cams = sorted(k.split('/')[0] for k in f0.get('files', {}) if k.endswith('/rgb'))
        if not cams:
            raise ValueError(f'{run_dir}: no camera rgb recorded')
        cam = cams[0]
    return cam, next((s for s in sensors if s.get('id') == cam), None)


def read_rgb(path: str, width: int, height: int) -> np.ndarray:
    """(height, width, 3) uint8 RGB of an image file, area-resized when its size differs."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'{path}: not an image')
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img[:, :, ::-1])


def drive_samples(run_dir: str, options: dict | None = None) -> dict:
    """{'X', 'Y', 'steps', 'bev', 'img' (n, H, W, 3) uint8, 'camera', 'sensor', 'report'} of one drive."""
    from . import imitation as IM
    o = dict(E2E_DEFAULTS, **(options or {}))
    d = IM.drive_samples(run_dir, {'bev': o['bev'], 'bev_spec': o['bev_spec'],
                                   'target_speed_kmh': o['target_speed_kmh']})
    cam, sensor = _camera_of(run_dir, o['camera'])
    by_step = {}
    with open(os.path.join(run_dir, 'frames.jsonl'), encoding='utf-8') as fh:
        for line in fh:
            f = json.loads(line)
            rel = f.get('files', {}).get(f'{cam}/rgb')
            if rel:
                by_step[f['step']] = rel
    keep = [i for i, st in enumerate(d['steps']) if int(st) in by_step][::max(int(o['every']), 1)]
    if not keep:
        raise ValueError(f'{run_dir}: no step with both a control pair and a {cam} image')
    img = np.stack([read_rgb(os.path.join(run_dir, by_step[int(d['steps'][i])]), int(o['width']), int(o['height']))
                    for i in keep])
    rep = dict(d['report'], samples=len(keep), camera=cam)
    return {'X': d['X'][keep], 'Y': d['Y'][keep], 'steps': d['steps'][keep],
            'bev': d['bev'][keep] if d['bev'] is not None else None, 'img': img, 'camera': cam, 'sensor': sensor,
            'report': rep}


def e2e_spec(o: dict) -> dict:
    return {'kind': 'e2e', 'hidden': [int(h) for h in o['hidden']], 'activation': o['activation'], 'output': 'tanh',
            'conv': [int(c) for c in o['conv']], 'bev': bool(o['bev']), 'bev_conv': [int(c) for c in o['bev_conv']]}


def train_e2e(runs, out_dir: str, options: dict | None = None, log=None, progress=None, cancel=None) -> dict:
    """Train an end-to-end policy (module doc) into out_dir; returns its metrics."""
    import torch
    from .imitation import action_errors
    from .policy import build_net, feature_index, normalise, save_policy
    from .provenance import block, new_run, utc_now
    from .wmtrain import _device, _mix
    o = dict(E2E_DEFAULTS, **(options or {}))
    log = log or (lambda s: None)
    t0 = time.monotonic()
    runs = [os.path.abspath(r) for r in runs]
    data, notes = [], []
    for r in runs:
        try:
            d = drive_samples(r, o)
        except (ValueError, OSError) as e:
            notes.append(str(e))
            log(f'skipped: {e}')
            continue
        data.append(d)
        log(f'{os.path.basename(r)}: {d["report"]["samples"]} pairs')
    if len(data) < 2:
        raise ValueError('end-to-end cloning splits by drive and needs at least two drives with recorded control and '
                         f'camera images ({len(data)} usable{": " + "; ".join(notes) if notes else ""})')
    sens = [(json.dumps((d['sensor'] or {}).get('pose'), sort_keys=True), (d['sensor'] or {}).get('fov'))
            for d in data]
    if len(set(sens)) > 1:
        raise ValueError('the drives\' cameras differ in pose or field of view; the environment can open one camera: '
                         'train on drives recorded with the same rig')
    sensor = data[0]['sensor'] or {}
    camera = {'width': int(o['width']), 'height': int(o['height']), 'fov': float(sensor.get('fov', 90.0)),
              'pose': dict({'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                           **(sensor.get('pose') or {}))}
    if not data[0]['sensor']:
        notes.append('the drives name no rig camera: the environment\'s default camera pose and 90° are assumed')
    rng = np.random.default_rng(int(o['seed']))
    order = rng.permutation(len(data))
    n_val = min(max(int(round(o['val_fraction'] * len(data))), 1), len(data) - 1)
    val_ids, train_ids = sorted(order[:n_val].tolist()), sorted(order[n_val:].tolist())
    feats = list(o['state_features'])
    idx = feature_index(feats)

    def stack(ids, key):
        parts = [data[i][key] for i in ids]
        return None if parts[0] is None else np.concatenate(parts)
    Xtr, Ytr, Itr = stack(train_ids, 'X')[:, idx], stack(train_ids, 'Y'), stack(train_ids, 'img')
    Xva, Yva, Iva = stack(val_ids, 'X')[:, idx], stack(val_ids, 'Y'), stack(val_ids, 'img')
    Btr = stack(train_ids, 'bev') if o['bev'] else None
    Bva = stack(val_ids, 'bev') if o['bev'] else None
    mean = Xtr.astype(np.float64).mean(0) if len(feats) else np.zeros(0)
    std = Xtr.astype(np.float64).std(0) if len(feats) else np.zeros(0)
    norm = {'mean': mean.tolist(), 'std': np.where(std < 1e-6, 1.0, std).tolist(), 'clip': 10.0}
    dev = _device(o['device'])
    torch.manual_seed(_mix(o['seed'], -1))
    spec = e2e_spec(o)
    net = build_net(spec, len(feats), 2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=float(o['lr']), weight_decay=float(o['weight_decay']))

    def tens(X, I, B, sel):
        x = torch.as_tensor(normalise(X[sel], norm), device=dev)
        im = torch.as_tensor(I[sel], device=dev).permute(0, 3, 1, 2).float() / 255.0
        b = torch.as_tensor(DF.unpack_bev(B[sel]).astype(np.float32), device=dev) if B is not None else None
        return x, b, im

    def predict(X, I, B):
        net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(X), 256):
                sel = np.arange(k, min(k + 256, len(X)))
                x, b, im = tens(X, I, B, sel)
                out.append(net(x, b, im).cpu().numpy())
        net.train()
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)
    best, best_epoch, history = math.inf, 0, []
    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
    bs = int(o['batch'])
    yt = torch.as_tensor(Ytr, device=dev)
    for ep in range(int(o['epochs'])):
        if cancel and cancel():
            from .training import Cancelled
            raise Cancelled()
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for k in range(0, len(perm), bs):
            sel = perm[k:k + bs]
            x, b, im = tens(Xtr, Itr, Btr, sel)
            loss = torch.mean((net(x, b, im) - yt[torch.as_tensor(sel, device=dev)]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(sel)
        v = float(np.mean((predict(Xva, Iva, Bva) - Yva) ** 2))
        history.append({'epoch': ep + 1, 'train_mse': tot / max(len(perm), 1), 'val_mse': v})
        if progress:
            progress((ep + 1) / int(o['epochs']), f'epoch {ep + 1}: validation MSE {v:.4f}')
        if v < best:
            best, best_state, best_epoch = v, {k: t.detach().clone() for k, t in net.state_dict().items()}, ep + 1
    net.load_state_dict(best_state)
    base = Ytr.mean(0)
    pv, pt = predict(Xva, Iva, Bva), predict(Xtr, Itr, Btr)
    metrics = {
        'val': action_errors(pv, Yva), 'train': action_errors(pt, Ytr),
        'baseline_val': action_errors(np.broadcast_to(base, Yva.shape), Yva),
        'baseline_train': action_errors(np.broadcast_to(base, Ytr.shape), Ytr),
        'baseline_action': base.tolist(), 'best_epoch': best_epoch, 'history': history,
        'per_drive_val': [dict(data[i]['report'], **action_errors(
            predict(data[i]['X'][:, idx], data[i]['img'], data[i]['bev']), data[i]['Y'])) for i in val_ids],
        'drives': {'train': [data[i]['report']['run'] for i in train_ids],
                   'val': [data[i]['report']['run'] for i in val_ids]},
        'samples': {'train': int(len(Xtr)), 'val': int(len(Xva))}, 'skipped': notes, 'device': str(dev),
        'source': 'measured', 'wall_s': round(time.monotonic() - t0, 2)}
    metrics['val_mae_vs_baseline'] = metrics['val']['mae'] / max(metrics['baseline_val']['mae'], 1e-12)
    ins = {f'run{i}': os.path.join(d['report']['run'], 'frames.jsonl') for i, d in enumerate(data)}
    settings = {k: v for k, v in o.items() if k != 'name'}
    run = new_run('e2e', runs, int(o['seed']), settings, ins)
    run.update(status='ok', exit_code=0, finished=utc_now(), name=o['name'] or 'end-to-end (camera)',
               summary={'val_mae': metrics['val']['mae'], 'baseline_val_mae': metrics['baseline_val']['mae'],
                        'drives': len(data), 'samples': metrics['samples']},
               performance={'wall_s': metrics['wall_s'], 'device': str(dev), 'source': 'measured'}, notes=notes)
    doc = {'name': o['name'] or 'end-to-end (camera' + (' + BEV)' if o['bev'] else ')'), 'trained_by': 'e2e-bc',
           'features': feats, 'normaliser': norm,
           'bev': dict(DF.BEV_DEFAULT, **(o['bev_spec'] or {})) if o['bev'] else None, 'camera': camera,
           'action': {'kind': 'continuous', 'keys': list(DF.ACTION_KEYS)}, 'net': spec,
           'provenance': block(ins, seed=int(o['seed']), settings=settings),
           'training': {'drives': len(data), 'samples': metrics['samples'], 'seed': int(o['seed']),
                        'camera_id': data[0]['camera']}}
    save_policy(out_dir, net.cpu(), doc, metrics, run)
    return metrics


def offline_eval(policy_dir: str, runs, out_dir: str = '', device: str = 'cpu') -> dict:
    """An end-to-end policy's action error on drives (the same pairs as training), per drive and overall, against
    the drives' own mean action."""
    from .imitation import action_errors
    from .policy import load_policy
    from .provenance import block, write_json
    pol = load_policy(policy_dir, device)
    if not pol.uses_camera:
        raise ValueError(f'{policy_dir}: not an end-to-end policy (falcon policy offline evaluates the others)')
    cam = pol.doc['camera']
    rows, P_, T_ = [], [], []
    for r in runs:
        d = drive_samples(r, {'bev': pol.uses_bev, 'bev_spec': pol.doc.get('bev'), 'width': cam['width'],
                              'height': cam['height'], 'camera': pol.doc.get('training', {}).get('camera_id', '')})
        bev = DF.unpack_bev(d['bev']) if d['bev'] is not None else None
        p = np.concatenate([pol.act_batch(d['X'][k:k + 256], None if bev is None else bev[k:k + 256],
                                          cameras=d['img'][k:k + 256]) for k in range(0, len(d['X']), 256)])
        rows.append(dict(d['report'], **action_errors(p, d['Y'])))
        P_.append(p)
        T_.append(d['Y'])
    P_, T_ = np.concatenate(P_), np.concatenate(T_)
    doc = {'policy': pol.path, 'drives': rows, 'overall': action_errors(P_, T_),
           'baseline_mean_action': action_errors(np.broadcast_to(T_.mean(0), T_.shape), T_), 'source': 'measured'}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        doc['provenance'] = block(dict({'policy': os.path.join(pol.path, 'policy.json')},
                                       **{f'run{i}': os.path.join(r, 'frames.jsonl') for i, r in enumerate(runs)}))
        write_json(os.path.join(out_dir, 'offline.json'), doc)
    return doc
