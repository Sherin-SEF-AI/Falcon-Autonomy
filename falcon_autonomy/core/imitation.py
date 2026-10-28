"""Behaviour cloning from recorded drives (numpy at import; torch only inside functions): (observation, action) pairs
built with the environment's own featuriser (core.drivefeatures), a small torch MLP (or a CNN over the BEV plus the
state), a train/validation split by drive, seeded; the result is a policy folder (core.policy) that the environment
can run (falcon policy rollout).

Pairs. A drive (falcon-raw: frames.jsonl with `control`, actors.jsonl, lanes.json; docs/dataset-format.md) is replayed
step by step through core.drivefeatures.Featuriser exactly as the environment worker does:
* the ego: frames.jsonl's pose and speed, its box from actors.jsonl's hero row;
* the actors: actors.jsonl at that step (the same rows the worker builds: workers/scene.py ActorLog);
* the route: drivefeatures.route_from_lanes — lanes.json's centre points matched to the lanes the ego drove through
  (the environment's route is the lane path it is asked to follow; here it is the lanes it was seen to follow);
* traffic lights: unknown (drives do not record their states), so the tl_* features are left out of a cloned policy
  by default (TL_FEATURES; it would otherwise meet in the environment inputs it never saw);
* the target speed: the scenario's ego speed for scenario drives, else options target_speed_kmh, else — heuristic —
  the drive's median speed over the steps it moved (> 0.5 m/s); which one is in the dataset report per drive;
* prev_steer/prev_accel: the control recorded at that step (in effect during the tick that produced it), and the
  action — the target — is the next step's recorded control as [steer, throttle − brake]: the control applied from
  this step's state (frames.jsonl `control`: the control in effect during the tick that produced its step).
Steps without a recorded control (or whose next step has none) are skipped; a drive without any is refused (drives
recorded before control was recorded). Optional: the BEV at each step (core.worldmodel.bev_raster with
drivefeatures' default spec unless given), for a CNN.

Training: inputs normalised with the training split's mean and standard deviation (a feature constant in training
gets std 1: it cannot be scaled from data), MSE on both action components, Adam, minibatches in a seeded order,
torch.manual_seed(seed); the weights of the epoch with the lowest validation loss are kept. The split is by drive
(val_fraction of the drives, at least one; a split needs two drives). Metrics on both splits: mean absolute and RMS
error per component, for the policy and for the constant baseline (the training split's mean action) — a policy
that does not beat that baseline has learnt nothing from the state.

Limits: cloning reproduces the recorded driver's behaviour (the autopilot's or the lane follower's) including its
faults; small errors compound in closed loop (covariate shift) — offline action error does not predict closed-loop
driving: evaluate through the environment (falcon policy rollout). prev_* inputs carry much of the next action
(controls are smooth): a known shortcut ("copycat") that flatters the offline error; exclude them to see.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import drivefeatures as DF
from .policy import TL_FEATURES

BC_DEFAULTS = {
    'epochs': 40, 'batch': 256, 'lr': 1e-3, 'weight_decay': 1e-5, 'hidden': [128, 128], 'activation': 'relu',
    'val_fraction': 0.25, 'seed': 0, 'exclude_features': list(TL_FEATURES), 'bev': False, 'bev_spec': None,
    'conv': [16, 32, 32], 'device': 'cpu', 'target_speed_kmh': None, 'name': '',
}


def _jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def drive_samples(run_dir: str, options: dict | None = None) -> dict:
    """{'X' (n, N_FEATURES) float32, 'Y' (n, 2) float32, 'steps' (n,), 'bev' (n, H, W) packed bits or None, 'report'}
    of one drive (module doc). Raises ValueError when the drive cannot give pairs."""
    o = dict(BC_DEFAULTS, **(options or {}))
    from . import worldmodel as WM
    fp = os.path.join(run_dir, 'frames.jsonl')
    if not os.path.isfile(fp):
        raise ValueError(f'{run_dir}: no frames.jsonl')
    data = WM.load_run(run_dir)
    frames = data['frames']
    if not any(isinstance(f.get('control'), dict) for f in frames):
        raise ValueError(f'{run_dir}: frames.jsonl records no control (drives recorded before Falcon 0.6 do not; '
                         're-record it)')
    if data['actors'] is None:
        raise ValueError(f'{run_dir}: no actors.jsonl (the other road users are needed for the lead and pedestrian '
                         'features)')
    if not data.get('lanes'):
        raise ValueError(f'{run_dir}: no lanes.json (the route is built from the map\'s lanes; record the drive with a '
                         'labelled camera)')
    ego_id = data['ego_id']
    dt = 1.0 / data['fps']
    ego_row = next((a for rows in data['actors'].values() for a in rows if a['id'] == ego_id), None)
    length, width = (2.0 * ego_row['extent'][0], 2.0 * ego_row['extent'][1]) if ego_row else (4.5, 2.0)
    cfg = {}
    for name in ('drive.json', 'live.json'):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as fh:
                cfg = json.load(fh)
            break
    speeds = np.array([next((math.hypot(a['vx'], a['vy']) for a in data['actors'].get(f['step'], [])
                             if a['id'] == ego_id), f.get('speed_mps', 0.0)) for f in frames], float)
    scn = cfg.get('scenario') or {}
    if scn.get('ego', {}).get('speed_kmh') is not None:
        target, src = float(scn['ego']['speed_kmh']) / 3.6, 'the scenario\'s ego speed'
    elif o.get('target_speed_kmh'):
        target, src = float(o['target_speed_kmh']) / 3.6, 'the option target_speed_kmh'
    else:
        moving = speeds[speeds > 0.5]
        target, src = (float(np.median(moving)) if len(moving) else 0.0), 'heuristic: the drive\'s median moving speed'
    xy = np.array([[f['ego']['x'], f['ego']['y']] for f in frames], float)
    yaw = np.array([f['ego']['yaw'] for f in frames], float)
    route = DF.route_from_lanes(data['lanes'], xy, yaw)
    feat = DF.Featuriser(route, target, dt)
    spec = lane_map = None
    if o.get('bev'):
        spec = DF.bev_spec(o.get('bev_spec'))
        lane_map = WM.lane_map_for(data['lanes'], [f['ego'] for f in frames], spec)
    X, Y, steps, bevs = [], [], [], []
    for t, f in enumerate(frames):
        c = f.get('control')
        prev = DF.control_to_action(c) if isinstance(c, dict) else [0.0, 0.0]
        rows = data['actors'].get(f['step'], [])
        me = next((a for a in rows if a['id'] == ego_id), None)
        # the speed: actors.jsonl's ego velocity (kinematic scenario drives before 0.6 recorded speed_mps 0: CARLA
        # reports a physics-off actor still, actors.jsonl carries the director's velocity), else frames.jsonl's
        speed = math.sqrt(me['vx'] ** 2 + me['vy'] ** 2 + me.get('vz', 0.0) ** 2) if me is not None else \
            f.get('speed_mps', 0.0)
        e = {'x': f['ego']['x'], 'y': f['ego']['y'], 'yaw': f['ego']['yaw'], 'speed_mps': speed,
             'length': length, 'width': width}
        x, _facts = feat(e, rows, None, prev, ego_id)          # every step: the featuriser is stateful
        nxt = frames[t + 1].get('control') if t + 1 < len(frames) else None
        if not isinstance(c, dict) or not isinstance(nxt, dict) or frames[t + 1]['step'] != f['step'] + 1:
            continue
        X.append(x)
        Y.append(DF.control_to_action(nxt))
        steps.append(f['step'])
        if spec is not None:
            bevs.append(DF.bev_bits(f['ego'], rows, spec, lane_map, ego_id))
    if not X:
        raise ValueError(f'{run_dir}: no step with a recorded control followed by another')
    return {'X': np.asarray(X, np.float32), 'Y': np.clip(np.asarray(Y, np.float32), -1.0, 1.0),
            'steps': np.asarray(steps, np.int64), 'bev': np.asarray(bevs, np.uint8) if bevs else None,
            'report': {'run': run_dir, 'samples': len(X), 'steps': len(frames), 'target_speed_mps': round(target, 4),
                       'target_speed_source': src, 'route_length_m': round(route.length, 2), 'fps': data['fps']}}


def action_errors(pred, true) -> dict:
    """Mean absolute and RMS error per component (steer, accel) and overall."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    out = {}
    for k, name in enumerate(DF.ACTION_KEYS):
        out[f'{name}_mae'] = float(np.mean(np.abs(d[:, k])))
        out[f'{name}_rmse'] = float(np.sqrt(np.mean(d[:, k] ** 2)))
    out['mae'] = float(np.mean(np.abs(d)))
    out['mse'] = float(np.mean(d ** 2))
    out['n'] = int(len(d))
    return out


def _torch_setup(seed, device):
    import torch
    torch.manual_seed(int(seed))
    if device != 'cpu' and not torch.cuda.is_available():
        raise ValueError(f'device {device!r} is not available')
    return torch


def train_bc(runs, out_dir: str, options: dict | None = None, log=None) -> dict:
    """Train a behaviour-cloning policy on drives (module doc) and write its policy folder to out_dir; returns its
    metrics."""
    from .policy import feature_index, net_spec, save_policy, select_features
    from .provenance import block, new_run, utc_now
    o = dict(BC_DEFAULTS, **(options or {}))
    log = log or (lambda s: None)
    t0 = time.monotonic()
    runs = [os.path.abspath(r) for r in runs]
    data, notes = [], []
    for r in runs:
        try:
            d = drive_samples(r, o)
        except ValueError as e:
            notes.append(str(e))
            log(f'skipped: {e}')
            continue
        data.append(d)
        log(f'{os.path.basename(r)}: {d["report"]["samples"]} pairs')
    if len(data) < 2:
        raise ValueError('behaviour cloning splits by drive and needs at least two drives with recorded control '
                         f'({len(data)} usable{": " + "; ".join(notes) if notes else ""})')
    rng = np.random.default_rng(int(o['seed']))
    order = rng.permutation(len(data))
    n_val = min(max(int(round(o['val_fraction'] * len(data))), 1), len(data) - 1)
    val_ids, train_ids = sorted(order[:n_val].tolist()), sorted(order[n_val:].tolist())
    feats = select_features(o['exclude_features'])
    idx = feature_index(feats)

    def stack(ids, key):
        parts = [data[i][key] for i in ids]
        return None if parts[0] is None else np.concatenate(parts)
    Xtr, Ytr, Xva, Yva = stack(train_ids, 'X')[:, idx], stack(train_ids, 'Y'), stack(val_ids, 'X')[:, idx], \
        stack(val_ids, 'Y')
    Btr = stack(train_ids, 'bev') if o['bev'] else None
    Bva = stack(val_ids, 'bev') if o['bev'] else None
    mean = Xtr.astype(np.float64).mean(0)
    std = Xtr.astype(np.float64).std(0)
    std = np.where(std < 1e-6, 1.0, std)
    norm = {'mean': mean.tolist(), 'std': std.tolist(), 'clip': 10.0}
    from .policy import build_net, normalise
    torch = _torch_setup(o['seed'], o['device'])
    spec = net_spec('cnn' if o['bev'] else 'mlp', o['hidden'], o['activation'], 'tanh', o['conv'])
    net = build_net(spec, len(feats), 2).to(o['device'])
    opt = torch.optim.Adam(net.parameters(), lr=float(o['lr']), weight_decay=float(o['weight_decay']))
    xt = torch.as_tensor(normalise(Xtr, norm), device=o['device'])
    yt = torch.as_tensor(Ytr, device=o['device'])
    xv = torch.as_tensor(normalise(Xva, norm), device=o['device'])

    def bev_t(B, sel=None):
        if B is None:
            return None
        b = B if sel is None else B[sel]
        return torch.as_tensor(DF.unpack_bev(b).astype(np.float32), device=o['device'])

    def predict(xs, B):
        net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(xs), 2048):
                out.append(net(xs[k:k + 2048], bev_t(B, slice(k, k + 2048))).cpu().numpy())
        net.train()
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)
    best, best_epoch, history = math.inf, 0, []
    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
    bs = int(o['batch'])
    for ep in range(int(o['epochs'])):
        perm = rng.permutation(len(xt))
        tot = 0.0
        for k in range(0, len(perm), bs):
            sel = perm[k:k + bs]
            st = torch.as_tensor(sel, device=o['device'])
            loss = torch.mean((net(xt[st], bev_t(Btr, sel)) - yt[st]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(sel)
        v = float(np.mean((predict(xv, Bva) - Yva) ** 2))
        history.append({'epoch': ep + 1, 'train_mse': tot / max(len(perm), 1), 'val_mse': v})
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
            best_epoch = ep + 1
    net.load_state_dict(best_state)
    base = Ytr.mean(0)
    pv, pt = predict(xv, Bva), predict(xt, Btr)
    metrics = {
        'val': action_errors(pv, Yva), 'train': action_errors(pt, Ytr),
        'baseline_val': action_errors(np.broadcast_to(base, Yva.shape), Yva),
        'baseline_train': action_errors(np.broadcast_to(base, Ytr.shape), Ytr),
        'baseline_action': base.tolist(), 'best_epoch': best_epoch, 'history': history,
        'per_drive_val': [dict(data[i]['report'], **action_errors(
            predict(torch.as_tensor(normalise(data[i]['X'][:, idx], norm), device=o['device']), data[i]['bev']),
            data[i]['Y'])) for i in val_ids],
        'drives': {'train': [data[i]['report']['run'] for i in train_ids],
                   'val': [data[i]['report']['run'] for i in val_ids]},
        'samples': {'train': int(len(Xtr)), 'val': int(len(Xva))}, 'skipped': notes,
        'source': 'measured', 'wall_s': round(time.monotonic() - t0, 2)}
    metrics['val_mae_vs_baseline'] = metrics['val']['mae'] / max(metrics['baseline_val']['mae'], 1e-12)
    ins = {f'run{i}': os.path.join(d['report']['run'], 'frames.jsonl') for i, d in enumerate(data)}
    settings = {k: v for k, v in o.items() if k != 'name'}
    prov = block(ins, seed=int(o['seed']), settings=settings)
    run = new_run('bc', runs, int(o['seed']), settings, ins)
    run.update(status='ok', exit_code=0, finished=utc_now(), name=o['name'] or 'behaviour cloning',
               summary={'val_mae': metrics['val']['mae'], 'baseline_val_mae': metrics['baseline_val']['mae'],
                        'drives': len(data), 'samples': metrics['samples']},
               performance={'wall_s': metrics['wall_s'], 'source': 'measured'}, notes=notes)
    doc = {'name': o['name'] or 'behaviour cloning', 'trained_by': 'bc', 'features': feats, 'normaliser': norm,
           'bev': dict(DF.BEV_DEFAULT, **(o['bev_spec'] or {})) if o['bev'] else None,
           'action': {'kind': 'continuous', 'keys': list(DF.ACTION_KEYS)}, 'net': spec, 'provenance': prov,
           'training': {'drives': len(data), 'samples': metrics['samples'], 'seed': int(o['seed'])}}
    save_policy(out_dir, net.cpu(), doc, metrics, run)
    return metrics


def offline_eval(policy_dir: str, runs, out_dir: str = '') -> dict:
    """The action error of a policy on drives (the same pairs as training) against the constant baseline of zero
    action and of the drives' own mean action."""
    from .policy import load_policy
    from .provenance import block, write_json
    pol = load_policy(policy_dir)
    rows, P_, T_ = [], [], []
    for r in runs:
        d = drive_samples(r, {'bev': pol.uses_bev, 'bev_spec': pol.doc.get('bev'),
                              'exclude_features': []})
        bev = DF.unpack_bev(d['bev']) if d['bev'] is not None else None
        p = pol.act_batch(d['X'], bev)
        rows.append(dict(d['report'], **action_errors(p, d['Y'])))
        P_.append(p)
        T_.append(d['Y'])
    P_, T_ = np.concatenate(P_), np.concatenate(T_)
    doc = {'policy': pol.path, 'drives': rows, 'overall': action_errors(P_, T_),
           'baseline_mean_action': action_errors(np.broadcast_to(T_.mean(0), T_.shape), T_),
           'source': 'measured'}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        doc['provenance'] = block(dict({'policy': os.path.join(pol.path, 'policy.json')},
                                       **{f'run{i}': os.path.join(r, 'frames.jsonl') for i, r in enumerate(runs)}))
        write_json(os.path.join(out_dir, 'offline.json'), doc)
    return doc


def main_bc(argv) -> int:
    import argparse
    from . import protocol as P
    from .policy import _kv
    ap = argparse.ArgumentParser(prog='falcon policy bc', description='behaviour cloning from recorded drives')
    ap.add_argument('--run', action='append', required=True, help='a recorded drive (repeat for more)')
    ap.add_argument('--out', '-o', required=True, help='the policy folder to write')
    ap.add_argument('--epochs', type=int, default=BC_DEFAULTS['epochs'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-fraction', type=float, default=BC_DEFAULTS['val_fraction'])
    ap.add_argument('--bev', action='store_true', help='a CNN over the BEV as well as the state')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE', help='any of BC_DEFAULTS')
    try:
        a = ap.parse_args(argv)
        opts = dict(_kv(a.option), epochs=a.epochs, seed=a.seed, val_fraction=a.val_fraction, bev=a.bev,
                    device=a.device)
    except SystemExit as e:
        return 0 if e.code == 0 else P.FAILED
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    if a.device != 'cpu':
        from .jobs import FileLock
        lock = FileLock('compute-gpu0.lock')
        if not lock.acquire():
            print(f'refused: the GPU is busy (its lock is held by process {lock.holder() or "?"})', flush=True)
            return P.REFUSED
    try:
        m = train_bc(a.run, a.out, opts, log=lambda s: print(s, flush=True))
    except (OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    print(f'validation action MAE {m["val"]["mae"]:.4f} (constant baseline {m["baseline_val"]["mae"]:.4f}); '
          f'policy in {a.out}', flush=True)
    return P.OK


def main_offline(argv) -> int:
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon policy offline', description='a policy\'s action error on drives')
    ap.add_argument('--policy', required=True)
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--out', '-o', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else P.FAILED
    try:
        doc = offline_eval(a.policy, a.run, a.out)
    except (OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    print(json.dumps({'policy': doc['overall'], 'baseline_mean_action': doc['baseline_mean_action']}, indent=2),
          flush=True)
    return P.OK
