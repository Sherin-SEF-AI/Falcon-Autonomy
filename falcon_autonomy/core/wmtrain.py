"""The learned world model: training on Falcon's derived data, an honest open-loop evaluation against simple baselines,
policy learning in imagination, and the `falcon worldmodel` command (numpy at import; torch only inside functions:
the GUI imports this module). The networks are core.wmnet's; the data core.wmdata's; docs/world-model-training.md is
the guide with the measured numbers.

    train(settings, out_dir, emitter, cancel) -> model.json        a DreamerV3-style RSSM (core.wmnet)
    evaluate(model_dir, sources, out_dir, ...) -> eval.json         BEV IoU per channel and reward error vs horizon
    imagine(model_dir, out_dir, settings, ...) -> policy folder     Dreamer's actor-critic in the latent space

Training. Sequences (core.wmdata: recorded drives and environment episodes) are split BY UNIT — a whole drive or a
whole experience folder — into training and held-out (val_fraction of the units, at least one when there are two;
or the units given as `val`). The state vector's features (all of core.drivefeatures' FEATURE_NAMES less
exclude_features, by default the traffic-light ones, unknown in recorded drives) are normalised with the training
units' mean and standard deviation (std 1 for a constant feature) and clipped to ±10, as policies normalise. Each
iteration draws `batch` windows of `seq_len` steps (a sequence by its length, a start uniformly; shorter sequences
padded and masked) from a generator seeded by (seed, iteration), and torch is re-seeded from the same pair, so a
resumed run continues the same random stream. The loss is core.wmnet's (BEV binary cross-entropy summed over the
planes and pixels, state squared error, symlog reward squared error — not counted at a window's first step, which
has no transition — continue cross-entropy, β_dyn·dynamics KL + β_rep·representation KL with free nats), averaged
over the valid steps; Adam with gradient clipping. A checkpoint every ckpt_every iterations and on a cancel; the same
command resumes it; a folder made with other settings or data is refused. The final weights are the LAST
iteration's: the held-out units are never used to choose anything (their loss is logged as a curve only), so the
evaluation on them is a held-out evaluation.

Output (a model folder): model.json (falcon.wmnet/1: the network's config, the features and their normaliser, the
BEV spec, the units of each split, the cost, the provenance), weights.pt (SHA-256 in model.json), metrics.jsonl (the
training curve: losses every log_every iterations, the held-out loss every eval_every), data.json (every sequence:
what was derived and how), run.json, checkpoints/last.pt (resume; removed at the end unless keep_checkpoint).

Evaluation (open loop, on held-out units unless sources are given). Each sequence is filtered through the posterior
from its first step (the latent's mode, so the evaluation is deterministic). From every start t0 = context, context +
stride, … the model imagines forward with the sequence's own recorded actions for k = 1 … horizon steps (the
prior's mode at each step) and decodes the BEV (a plane is predicted where its probability exceeds 0.5), the reward
and the continue probability. Against the truth at t0 + k:
* BEV IoU per channel (worldmodel.BEV_CHANNELS, plus "actors": the union of vehicle, pedestrian, cyclist and other),
  as the pooled intersection over the pooled union of every (start, k) pair at that horizon — the BEV segmentation
  literature's dataset IoU; a channel empty in both the truth and the prediction at every pair is None;
* reward: the mean absolute error of the predicted reward;
* continue: on environment episodes only (recorded drives have no terminal step), the Brier score of the continue
  probability and the terminal steps found.
Baselines, on the same pairs:
* copy-last-frame: the BEV at t0 for every k, the reward at t0 for every k;
* constant velocity (recorded drives only: it needs actors.jsonl): every actor whose centre is within the BEV window
  (2 m margin) at t0, and the ego, moved for k·dt at its recorded velocity with its yaw kept; the actors redrawn with
  core.worldmodel.bev_raster in the extrapolated ego frame; the map planes (drivable, centerline, lane boundary)
  are the BEV at t0 shifted by the ego's extrapolated motion (a pixel that comes from outside the window is empty) —
  so the baseline uses only the frame at t0 and the actors' velocities at t0, never the map beyond the window;
* the training units' mean reward.
eval.json holds every curve (k = 1 … horizon), the counts of pairs, per-sequence numbers, a verdict per reported
horizon (1, 5, 10, 20) and channel — whether the model beats every baseline — and the provenance; samples.npz holds
a few starts' truth, prediction and baselines for the viewer (Training ▸ World model).

Imagination (Dreamer's actor-critic, core.wmnet module doc). Start states are posterior states of training windows;
the actor (core.policy's MLP over the decoded state features, a tanh-Gaussian with a state-independent standard
deviation) rolls the frozen model's prior for `horizon` steps; rewards from the reward head, continues from the
continue head; λ-returns (γ, λ) bootstrapped with the critic; the actor's loss is the negative λ-return normalised by
max(1, the 5th–95th percentile range of the returns, an exponential moving average: DreamerV3's return
normalisation) less an entropy bonus, differentiated through the model's dynamics (Dreamer, Hafner et al. 2020); the
critic regresses the symlog λ-returns, regularised towards a slow (exponential moving average) copy of itself. The
deterministic actor (tanh of the mean) is written as a policy folder (core.policy: trained_by "dreamer", net kind
mlp) that the environment, rollouts, the Leaderboard and the replay run like any other.

Limits: the model has seen only what its data covered (recorded drives carry the recording driver's behaviour only);
a policy learnt in imagination exploits the model's errors wherever the data did not constrain it, and its value in
the real environment is measured there, not here; a recorded drive's reward lacks the map terms (core.wmdata), so
the reward head learns a mixture when drives and environment episodes are mixed; on the CUDA device results are not
bit for bit (cuDNN's kernels), on the CPU they are with the same thread count.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

from . import drivefeatures as DF
from . import wmdata as D

MODEL_SCHEMA = 'falcon.wmnet/1'
EVAL_SCHEMA = 'falcon.wmeval/1'
MODEL_JSON = 'model.json'
ACTOR_CHANNELS = ('vehicle', 'pedestrian', 'cyclist', 'other')
MAP_CHANNELS = ('drivable', 'centerline', 'lane_boundary')
REPORT_HORIZONS = (1, 5, 10, 20)
DEFAULTS = {
    'train': [], 'val': [], 'iterations': 3000, 'batch': 16, 'seq_len': 32, 'lr': 3e-4, 'adam_eps': 1e-8,
    'grad_clip': 100.0, 'deter': 256, 'stoch': 16, 'classes': 16, 'hidden': 256, 'depth': 32, 'unimix': 0.01,
    'free_nats': 1.0, 'beta_dyn': 0.5, 'beta_rep': 0.1, 'scale_bev': 1.0, 'scale_state': 1.0, 'scale_reward': 1.0,
    'scale_cont': 1.0, 'val_fraction': 0.25, 'seed': 0, 'device': 'auto', 'bev_spec': None,
    'exclude_features': ['tl_known', 'tl_red', 'tl_yellow', 'tl_green', 'tl_distance_m'], 'target_speed_kmh': None,
    'log_every': 50, 'eval_every': 250, 'ckpt_every': 500, 'val_batches': 4, 'gpu_memory_fraction': 0.6,
    'keep_checkpoint': False, 'id': '',
}
VOLATILE = ('log_every', 'eval_every', 'ckpt_every', 'val_batches', 'gpu_memory_fraction', 'keep_checkpoint', 'id')
IMAGINE_DEFAULTS = {
    'iterations': 1500, 'batch': 16, 'seq_len': 16, 'horizon': 15, 'gamma': 0.99, 'lambda': 0.95, 'actor_lr': 1e-4,
    'critic_lr': 3e-4, 'entropy': 3e-4, 'hidden': [64, 64], 'activation': 'tanh', 'init_std': 0.5,
    'slow_critic': 0.02, 'grad_clip': 100.0, 'seed': 0, 'device': 'auto', 'features': None, 'log_every': 50,
    'name': '',
}


class WMError(Exception):
    pass


class _Quiet:
    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass

    def metric(self, *a, **k):
        pass


def _mix(*xs) -> int:
    h = hashlib.sha256(json.dumps([int(x) for x in xs]).encode()).digest()
    return int.from_bytes(h[:4], 'little') & 0x7FFFFFFF


def normalise_settings(s: dict) -> dict:
    o = dict(DEFAULTS, **{k: v for k, v in (s or {}).items() if v is not None})
    bad = sorted(set(o) - set(DEFAULTS))
    if bad:
        raise WMError(f'unknown settings {bad}')
    for k in ('iterations', 'batch', 'seq_len', 'deter', 'stoch', 'classes', 'hidden', 'depth'):
        if int(o[k]) < 1:
            raise WMError(f'{k} must be at least 1')
        o[k] = int(o[k])
    o['train'] = [os.path.abspath(p) for p in o['train']]
    o['val'] = [os.path.abspath(p) for p in o['val']]
    o['exclude_features'] = sorted(set(o['exclude_features']))
    bad = sorted(set(o['exclude_features']) - set(DF.FEATURE_NAMES))
    if bad:
        raise WMError(f'unknown features {bad}')
    return o


def _device(name: str):
    from .training import TrainingError, _device as dv
    try:
        return dv({'device': name})
    except TrainingError as e:
        raise WMError(str(e)) from None


def _torch_flags(dev):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')     # cuBLAS's deterministic workspace
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# ============================================================================================ data in arrays
class Data:
    """Sequences in the model's units: packed BEV, normalised features, actions, rewards, terminals."""

    def __init__(self, seqs: list, feat_idx, norm: dict):
        from .policy import normalise
        self.seqs = []
        for s in seqs:
            self.seqs.append({'bev': s['bev'], 'state': normalise(s['state'][:, feat_idx], norm),
                              'action': s['action'].astype(np.float32), 'reward': s['reward'].astype(np.float32),
                              'terminal': s['terminal'].astype(bool), 'name': s['name'], 'kind': s['kind']})
        self.lengths = np.array([len(s['state']) for s in self.seqs], np.int64)

    def windows(self, rng, batch: int, L: int) -> dict:
        """`batch` windows of L steps (module doc): numpy arrays, BEV still packed."""
        w = np.maximum(self.lengths - 1, 1).astype(float)
        idx = rng.choice(len(self.seqs), size=batch, p=w / w.sum())
        H, W = self.seqs[0]['bev'].shape[1:]
        F = self.seqs[0]['state'].shape[1]
        out = {'bev': np.zeros((batch, L, H, W), np.uint8), 'state': np.zeros((batch, L, F), np.float32),
               'action': np.zeros((batch, L, 2), np.float32), 'reward': np.zeros((batch, L), np.float32),
               'terminal': np.zeros((batch, L), bool), 'mask': np.zeros((batch, L), np.float32),
               'first': np.zeros((batch, L), np.float32)}
        for b, i in enumerate(idx):
            s = self.seqs[i]
            n = len(s['state'])
            st = int(rng.integers(0, max(n - L, 0) + 1))
            m = min(L, n - st)
            for k in ('bev', 'state', 'action', 'reward', 'terminal'):
                out[k][b, :m] = s[k][st:st + m]
            out['mask'][b, :m] = 1.0
            out['first'][b, 0] = 1.0
        return out


def to_tensors(batch: dict, dev) -> dict:
    import torch
    out = {k: torch.as_tensor(v, device=dev) for k, v in batch.items() if k != 'bev'}
    out['terminal'] = out['terminal'].float()
    out['bev'] = torch.as_tensor(DF.unpack_bev(batch['bev']), device=dev).float()
    return out


def wm_loss(model, b: dict, s: dict) -> tuple:
    """(total loss, {component: float}) of a batch (module doc)."""
    import torch
    import torch.nn.functional as F
    from . import wmnet as N
    kl = N.classes()['kl']
    h, z, lp, lq = model.observe(b['bev'], b['state'], b['action'], b['first'])
    out = model.heads(model.feat(h, z))
    m = b['mask']
    rm = m * (1.0 - b['first'])
    bce = F.binary_cross_entropy_with_logits(out['bev'], b['bev'], reduction='none').sum((-3, -2, -1))
    st = ((out['state'] - b['state']) ** 2).sum(-1)
    rw = (out['reward'] - N.symlog(b['reward'])) ** 2
    ct = F.binary_cross_entropy_with_logits(out['cont'], 1.0 - b['terminal'], reduction='none')
    dyn = kl(lq.detach(), lp).clamp(min=float(s['free_nats']))
    rep = kl(lq, lp.detach()).clamp(min=float(s['free_nats']))
    n, nr = m.sum().clamp(min=1.0), rm.sum().clamp(min=1.0)
    parts = {'bev': (bce * m).sum() / n, 'state': (st * m).sum() / n, 'reward': (rw * rm).sum() / nr,
             'cont': (ct * m).sum() / n, 'dyn': (dyn * m).sum() / n, 'rep': (rep * m).sum() / n}
    total = s['scale_bev'] * parts['bev'] + s['scale_state'] * parts['state'] + s['scale_reward'] * parts['reward'] \
        + s['scale_cont'] * parts['cont'] + s['beta_dyn'] * parts['dyn'] + s['beta_rep'] * parts['rep']
    return total, {k: float(v.detach()) for k, v in parts.items()}


def net_config(s: dict, spec, n_state: int) -> dict:
    return {'channels': 8, 'bev_hw': [spec.h, spec.w], 'n_state': int(n_state), 'n_action': 2, 'deter': s['deter'],
            'stoch': s['stoch'], 'classes': s['classes'], 'hidden': s['hidden'], 'depth': s['depth'],
            'unimix': float(s['unimix'])}


def _split(seqs, s, notes):
    units = sorted({q['unit'] for q in seqs})
    if s['val']:
        val = [u for u in units if u in set(s['val'])]
        train = [u for u in units if u not in set(s['val'])]
    else:
        rng = np.random.default_rng(int(s['seed']))
        order = rng.permutation(len(units))
        n_val = min(max(int(round(s['val_fraction'] * len(units))), 1), len(units) - 1) if len(units) >= 2 else 0
        val = sorted(units[i] for i in order[:n_val])
        train = sorted(units[i] for i in order[n_val:])
        if not val:
            notes.append('one unit only: nothing is held out (give more drives, or --val)')
    if not train:
        raise WMError('no training unit left after the split')
    return train, val


# ============================================================================================ training
def train(settings: dict, out_dir: str, emitter=None, cancel=None, argv=None) -> dict:
    """Train a world model (module doc) into out_dir; returns model.json's document."""
    import torch
    from . import wmnet as N
    from .policy import feature_index
    from .provenance import block, new_run, read_json, sha256_file, sha256_json, utc_now, write_json
    from .worldmodel import BEV_CHANNELS
    em = emitter or _Quiet()
    s = normalise_settings(settings)
    if not s['train']:
        raise WMError('at least one training source (a drive or an experience folder) is needed')
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(out_dir, MODEL_JSON)):
        doc = read_json(os.path.join(out_dir, MODEL_JSON))
        em.log(f'{out_dir} is a finished world model: not trained again')
        return doc
    t0 = time.monotonic()
    spec = DF.bev_spec(s['bev_spec'])
    notes = []
    sources = list(dict.fromkeys(s['train'] + s['val']))
    seqs, skipped = D.load_sources(sources, {'bev_spec': s['bev_spec'], 'target_speed_kmh': s['target_speed_kmh']},
                                   em.log)
    notes += skipped
    if not seqs:
        raise WMError('no usable sequence: ' + '; '.join(skipped))
    D.check_bev_shape(seqs, spec)
    train_u, val_u = _split(seqs, s, notes)
    feats = [n for n in DF.FEATURE_NAMES if n not in set(s['exclude_features'])]
    idx = feature_index(feats)
    tr_seqs = [q for q in seqs if q['unit'] in set(train_u)]
    va_seqs = [q for q in seqs if q['unit'] in set(val_u)]
    X = np.concatenate([q['state'][:, idx] for q in tr_seqs]).astype(np.float64)
    std = X.std(0)
    norm = {'mean': X.mean(0).tolist(), 'std': np.where(std < 1e-6, 1.0, std).tolist(), 'clip': 10.0}
    reward_mean = float(np.mean(np.concatenate([q['reward'][1:] for q in tr_seqs])))
    data_doc = {'sequences': [dict(q['report'], name=q['name'], kind=q['kind'], unit=q['unit'],
                                   split='train' if q['unit'] in set(train_u) else 'val') for q in seqs],
                'skipped': skipped}
    data_sha = sha256_json([[q['name'], int(len(q['state'])), float(q['state'].sum()), int(q['bev'].sum())]
                            for q in seqs])
    key = sha256_json([{k: v for k, v in s.items() if k not in VOLATILE}, data_sha])
    trd, vad = Data(tr_seqs, idx, norm), (Data(va_seqs, idx, norm) if va_seqs else None)
    dev = _device(s['device'])
    _torch_flags(dev)
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(s['gpu_memory_fraction']), dev)
        torch.cuda.reset_peak_memory_stats(dev)
    torch.manual_seed(_mix(s['seed'], -1))
    cfg = net_config(s, spec, len(feats))
    model = N.classes()['WorldModel'](cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(s['lr']), eps=float(s['adam_eps']))
    ck = os.path.join(out_dir, 'checkpoints', 'last.pt')
    mpath = os.path.join(out_dir, 'metrics.jsonl')
    start = 0
    if os.path.isfile(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        if st.get('key') != key:
            raise WMError(f'{out_dir} holds a training with other settings or data: choose another folder')
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        start = int(st['it'])
        em.log(f'resuming at iteration {start}')
        rows = [json.loads(x) for x in open(mpath, encoding='utf-8')] if os.path.isfile(mpath) else []
        with open(mpath, 'w', encoding='utf-8') as fh:
            for r in rows:
                if r.get('it', 0) <= start:
                    fh.write(json.dumps(r) + '\n')
    elif os.path.isfile(mpath):
        os.remove(mpath)
    run = new_run('worldmodel', list(argv or ['worldmodel', 'train']), int(s['seed']),
                  {k: v for k, v in s.items() if k not in ('train', 'val')},
                  {f'source{i}': (os.path.join(p, D.EXPERIENCE) if D.is_experience(p) else
                                  os.path.join(p, 'frames.jsonl')) for i, p in enumerate(sources)})
    write_json(os.path.join(out_dir, 'run.json'), run)
    write_json(os.path.join(out_dir, 'data.json'), data_doc)

    def checkpoint(it):
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'it': it, 'key': key}, ck + '.part')
        os.replace(ck + '.part', ck)

    def val_loss():
        if vad is None:
            return None
        model.eval()
        tot, parts = 0.0, {}
        with torch.no_grad():
            for j in range(int(s['val_batches'])):
                b = to_tensors(vad.windows(np.random.default_rng([int(s['seed']), 999_999, j]), s['batch'],
                                           s['seq_len']), dev)
                loss, p = wm_loss(model, b, s)
                tot += float(loss) / s['val_batches']
                for k, v in p.items():
                    parts[k] = parts.get(k, 0.0) + v / s['val_batches']
        model.train()
        return dict(parts, loss=tot)

    model.train()
    it = start
    t_train = time.monotonic()
    wall_seen = 0.0
    status = 'ok'
    try:
        with open(mpath, 'a', encoding='utf-8') as mf:
            acc, n_acc, t_log = {}, 0, time.monotonic()
            while it < s['iterations']:
                if cancel and cancel():
                    checkpoint(it)
                    status = 'cancelled'
                    break
                rng = np.random.default_rng([int(s['seed']), it])
                torch.manual_seed(_mix(s['seed'], it))
                b = to_tensors(trd.windows(rng, s['batch'], s['seq_len']), dev)
                loss, parts = wm_loss(model, b, s)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(s['grad_clip']))
                opt.step()
                it += 1
                parts['loss'] = float(loss.detach())
                parts['grad_norm'] = float(gn)
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + v
                n_acc += 1
                if it % s['log_every'] == 0 or it == s['iterations']:
                    dt_ = time.monotonic() - t_log
                    row = {'it': it, **{k: round(v / n_acc, 5) for k, v in acc.items()},
                           'steps_per_s': round(n_acc * s['batch'] * s['seq_len'] / max(dt_, 1e-9), 1)}
                    if it % s['eval_every'] == 0 or it == s['iterations']:
                        v = val_loss()
                        if v is not None:
                            row['val'] = {k: round(x, 5) for k, x in v.items()}
                    mf.write(json.dumps(row) + '\n')
                    mf.flush()
                    em.metric('wm_loss', row['loss'], it=it)
                    em.progress(it / s['iterations'], f'iteration {it}: loss {row["loss"]:.2f} (bev {row["bev"]:.1f})')
                    acc, n_acc, t_log = {}, 0, time.monotonic()
                if it % s['ckpt_every'] == 0 and it < s['iterations']:
                    checkpoint(it)
    except BaseException:
        checkpoint(it)
        raise
    wall_seen = time.monotonic() - t_train
    if status == 'cancelled':
        run.update(status='cancelled', exit_code=3, finished=utc_now(), notes=notes + [f'cancelled at {it}'])
        write_json(os.path.join(out_dir, 'run.json'), run)
        from .training import Cancelled
        raise Cancelled()
    wp = os.path.join(out_dir, 'weights.pt')
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wp)
    peak = round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1) if dev.type == 'cuda' else None
    cost = {'train_s': round(wall_seen, 1), 'wall_s': round(time.monotonic() - t0, 1), 'iterations': it,
            'device': str(dev), 'gpu_name': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None,
            'peak_gpu_mib': peak, 'parameters': int(sum(p.numel() for p in model.parameters())),
            'train_steps': int(sum(len(q['state']) for q in tr_seqs)),
            'val_steps': int(sum(len(q['state']) for q in va_seqs)), 'source': 'measured',
            'threads': torch.get_num_threads()}
    doc = {'schema': MODEL_SCHEMA, 'name': s['id'] or os.path.basename(os.path.normpath(out_dir)),
           'method': 'RSSM world model in the form of DreamerV3 (Hafner et al. 2023, arXiv:2301.04104); '
                     'core.wmnet module doc lists the differences',
           'config': cfg, 'features': feats, 'normaliser': norm, 'bev': dict(DF.BEV_DEFAULT, **(s['bev_spec'] or {})),
           'channels': list(BEV_CHANNELS),
           'action_keys': list(DF.ACTION_KEYS), 'reward_mean': reward_mean,
           'data': {'train': train_u, 'val': val_u, 'sequences': len(seqs)},
           'settings': {k: v for k, v in s.items() if k not in ('train', 'val')}, 'weights': 'weights.pt',
           'weights_sha256': sha256_file(wp), 'cost': cost, 'notes': notes,
           'provenance': block({f'source{i}': (os.path.join(p, D.EXPERIENCE) if D.is_experience(p) else
                                               os.path.join(p, 'frames.jsonl')) for i, p in enumerate(sources)},
                               int(s['seed']), {k: v for k, v in s.items() if k not in ('train', 'val')})}
    write_json(os.path.join(out_dir, MODEL_JSON), doc)
    run.update(status='ok', exit_code=0, finished=utc_now(), notes=notes, summary={
        'train_steps': cost['train_steps'], 'val_steps': cost['val_steps'], 'iterations': it},
        performance=cost)
    write_json(os.path.join(out_dir, 'run.json'), run)
    if not s['keep_checkpoint'] and os.path.isfile(ck):
        os.remove(ck)
    return doc


def load_model(model_dir: str, device='cpu'):
    """(WorldModel on device in eval mode, model.json)."""
    import torch
    from . import wmnet as N
    from .provenance import read_json, sha256_file
    doc = read_json(os.path.join(model_dir, MODEL_JSON))
    if doc.get('schema') != MODEL_SCHEMA:
        raise WMError(f'{model_dir}: not a world model folder')
    wp = os.path.join(model_dir, doc['weights'])
    if sha256_file(wp) != doc['weights_sha256']:
        raise WMError(f'{wp}: its SHA-256 is not the one model.json records')
    model = N.classes()['WorldModel'](doc['config'])
    model.load_state_dict(torch.load(wp, map_location='cpu', weights_only=True))
    return model.to(device).eval(), doc


def model_data(doc: dict, sources=None, log=None) -> tuple:
    """(Data, raw sequences, notes) of the given sources (default: the model's held-out units) in its units."""
    from .policy import feature_index
    srcs = list(sources) if sources else list(doc['data']['val'])
    if not srcs:
        raise WMError('the model held nothing out: give the drives to evaluate on')
    seqs, notes = D.load_sources(srcs, {'bev_spec': doc['bev']}, log)
    if not seqs:
        raise WMError('no usable sequence: ' + '; '.join(notes))
    D.check_bev_shape(seqs, DF.bev_spec(doc['bev']))
    return Data(seqs, feature_index(doc['features']), doc['normaliser']), seqs, notes


# ============================================================================================ baselines
class ConstantVelocity:
    """The constant-velocity baseline of one recorded drive (module doc)."""

    def __init__(self, run_dir: str, spec):
        from . import worldmodel as WM
        self.WM = WM
        self.data = WM.load_run(run_dir)
        self.spec = spec
        self.dt = 1.0 / self.data['fps']
        self.ego_id = self.data['ego_id']
        self.map_bits = WM.BEV_BIT['drivable'] | WM.BEV_BIT['centerline'] | WM.BEV_BIT['lane_boundary']
        self.C = spec.centres()

    def predict(self, step: int, bev0: np.ndarray, ks) -> list:
        """BEV bits (H, W) at step + k for each k of ks, from the state at `step` and its BEV bev0."""
        sp = self.spec
        f = next(f for f in self.data['frames'] if f['step'] == step)
        rows = self.data['actors'].get(step, [])
        ego = dict(f['ego'])
        er = next((a for a in rows if a['id'] == self.ego_id), None)
        evx, evy = (er['vx'], er['vy']) if er is not None else (0.0, 0.0)
        psi = math.radians(ego['yaw'])
        cy, sy = math.cos(psi), math.sin(psi)
        vis = []
        for a in rows:
            if a['id'] == self.ego_id:
                vis.append(a)
                continue
            dx, dy = a['x'] - ego['x'], a['y'] - ego['y']
            xe, ye = cy * dx + sy * dy, -sy * dx + cy * dy
            if -sp.back - 2.0 <= xe <= sp.forward + 2.0 and -sp.left - 2.0 <= ye <= sp.right + 2.0:
                vis.append(a)
        out = []
        for k in ks:
            tau = k * self.dt
            mx, my = evx * tau, evy * tau
            dxe, dye = cy * mx + sy * my, -sy * mx + cy * my
            rc = sp.pixel(self.C + np.array([dxe, dye]))
            r, c = np.floor(rc[..., 0] + 0.5).astype(int), np.floor(rc[..., 1] + 0.5).astype(int)
            ok = (r >= 0) & (r < sp.h) & (c >= 0) & (c < sp.w)
            mp = np.zeros((sp.h, sp.w), np.uint8)
            mp[ok] = bev0[r[ok], c[ok]] & self.map_bits
            moved = [dict(a, x=a['x'] + a.get('vx', 0.0) * tau, y=a['y'] + a.get('vy', 0.0) * tau) for a in vis]
            pose = dict(ego, x=ego['x'] + mx, y=ego['y'] + my)
            bits = self.WM.bev_raster(pose, moved, sp, None, self.ego_id)['bits']
            out.append(mp | bits)
        return out


def _channel_planes(bits: np.ndarray) -> np.ndarray:
    """(…, H, W) packed → (…, 9, H, W) bool: the 8 channels and 'actors'."""
    u = DF.unpack_bev(bits).astype(bool)
    act = u[..., 3:7, :, :].any(-3, keepdims=True)
    return np.concatenate([u, act], -3)


class _IoU:
    def __init__(self, K: int, n_ch: int = 9):
        self.I = np.zeros((K, n_ch), np.int64)
        self.U = np.zeros((K, n_ch), np.int64)

    def add(self, k: int, pred: np.ndarray, true: np.ndarray):
        """pred, true: (N, 9, H, W) bool at horizon k (1-based)."""
        self.I[k - 1] += (pred & true).sum((0, 2, 3))
        self.U[k - 1] += (pred | true).sum((0, 2, 3))

    def values(self) -> np.ndarray:
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self.U > 0, self.I / np.maximum(self.U, 1), np.nan)


def _nan_none(a):
    return [None if not np.isfinite(x) else round(float(x), 5) for x in a]


# ============================================================================================ evaluation
def evaluate(model_dir: str, sources=None, out_dir: str = '', horizon: int = 20, stride: int = 5, context: int = 5,
             n_samples: int = 12, device: str = 'cpu', emitter=None, cancel=None) -> dict:
    """The open-loop evaluation (module doc); writes eval.json and samples.npz in out_dir (default
    MODEL/eval)."""
    import torch
    from . import wmnet as N
    from .provenance import block, write_json
    from .worldmodel import BEV_CHANNELS
    em = emitter or _Quiet()
    t0_wall = time.monotonic()
    dev = _device(device)
    model, doc = load_model(model_dir, dev)
    data, seqs, notes = model_data(doc, sources, em.log)
    spec = DF.bev_spec(doc['bev'])
    K = int(horizon)
    names = list(BEV_CHANNELS) + ['actors']
    iou = {'model': _IoU(K), 'copy_last': _IoU(K), 'constant_velocity': _IoU(K)}
    recon = _IoU(1)
    rerr = {k: [[] for _ in range(K)] for k in ('model', 'copy_last', 'train_mean')}
    cont_sq, cont_n, cont_terms = [[] for _ in range(K)], 0, 0
    pairs = np.zeros(K, np.int64)
    per_seq = []
    cand = []                                   # (seq index, t0) for the samples
    for si, (q, raw) in enumerate(zip(data.seqs, seqs)):
        if cancel and cancel():
            break
        T = len(q['state'])
        starts = list(range(int(context), T - 1, int(stride)))
        if not starts:
            notes.append(f'{q["name"]}: too short for a start ({T} steps)')
            continue
        cv = ConstantVelocity(raw['source'], spec) if raw['kind'] == 'drive' else None
        with torch.no_grad():
            b = to_tensors({'bev': q['bev'][None], 'state': q['state'][None], 'action': q['action'][None],
                            'reward': q['reward'][None], 'terminal': q['terminal'][None],
                            'mask': np.ones((1, T), np.float32), 'first': np.eye(1, T, dtype=np.float32)}, dev)
            h, z, _lp, _lq = model.observe(b['bev'], b['state'], b['action'], b['first'], mode=True)
            rec = torch.sigmoid(model.dec(model.feat(h, z)))[0].cpu().numpy() > 0.5
            recon.add(1, _channel_planes(DF.pack_bev(rec.astype(np.uint8))), _channel_planes(q['bev']))
            S = torch.as_tensor(starts, device=dev)
            hh, zz = h[0, S], z[0, S]
            preds, rws, cps = [], [], []
            for k in range(1, K + 1):
                ai = np.minimum(np.asarray(starts) + k, T - 1)
                a = torch.as_tensor(q['action'][ai], device=dev)
                hh, zz, _ = model.img_step(hh, zz, a, mode=True)
                f = model.feat(hh, zz)
                preds.append((torch.sigmoid(model.dec(f)) > 0.5).cpu().numpy().astype(np.uint8))
                rws.append(model.reward(f).cpu().numpy())
                cps.append(torch.sigmoid(model.cont_head(f)[..., 0]).cpu().numpy())
        seq_iou = _IoU(K)
        seq_base = _IoU(K)
        st = np.asarray(starts)
        cvp = {}
        if cv is not None:
            for t in starts:
                ks = [k for k in range(1, K + 1) if t + k < T]
                cvp[t] = dict(zip(ks, cv.predict(raw['first_step'] + t, q['bev'][t], ks)))
        for k in range(1, K + 1):
            ok = st + k < T
            if not ok.any():
                continue
            tt = st[ok] + k
            true = _channel_planes(q['bev'][tt])
            pm = _channel_planes(DF.pack_bev(preds[k - 1][ok]))
            pc = _channel_planes(q['bev'][st[ok]])
            iou['model'].add(k, pm, true)
            iou['copy_last'].add(k, pc, true)
            seq_iou.add(k, pm, true)
            seq_base.add(k, pc, true)
            if cv is not None:
                cvb = np.stack([cvp[int(t)][k] for t in st[ok]])
                iou['constant_velocity'].add(k, _channel_planes(cvb), true)
            pairs[k - 1] += int(ok.sum())
            r_true = q['reward'][tt]
            rerr['model'][k - 1] += np.abs(rws[k - 1][ok] - r_true).tolist()
            rerr['copy_last'][k - 1] += np.abs(q['reward'][st[ok]] - r_true).tolist()
            rerr['train_mean'][k - 1] += np.abs(doc['reward_mean'] - r_true).tolist()
            if raw['kind'] == 'env':
                cont_sq[k - 1] += ((cps[k - 1][ok] - (1.0 - q['terminal'][tt].astype(float))) ** 2).tolist()
        if raw['kind'] == 'env':
            cont_n += 1
            cont_terms += int(q['terminal'].sum())
        sv, bv = seq_iou.values(), seq_base.values()
        per_seq.append({'name': q['name'], 'kind': raw['kind'], 'steps': T, 'starts': len(starts),
                        'action_source': raw['report'].get('action_source'),
                        'iou_actors_model': _nan_none(sv[[r - 1 for r in REPORT_HORIZONS if r <= K], 8]),
                        'iou_actors_copy_last': _nan_none(bv[[r - 1 for r in REPORT_HORIZONS if r <= K], 8])})
        full = [(si, t, j, preds, cv) for j, t in enumerate(starts) if t + K < T]      # the viewer: whole horizons
        cand += full[::max(1, len(full) // 3)][:4]
        em.progress((si + 1) / len(data.seqs), f'{q["name"]}: {len(starts)} starts')
    curves = {m: iou[m].values() for m in iou}
    has_cv = bool(iou['constant_velocity'].U.sum())
    if not has_cv:
        curves.pop('constant_velocity')
    report = [r for r in REPORT_HORIZONS if r <= K]
    verdict = {}
    for r in report:
        row = {}
        for ci, ch in enumerate(names):
            mv = curves['model'][r - 1, ci]
            bs = [curves[b][r - 1, ci] for b in curves if b != 'model']
            if not np.isfinite(mv) or not all(np.isfinite(bs)):
                row[ch] = None
            else:
                row[ch] = bool(mv > max(bs))
        verdict[str(r)] = row
    reward = {m: [round(float(np.mean(v)), 6) if v else None for v in rerr[m]] for m in rerr}
    out = {'schema': EVAL_SCHEMA, 'model': os.path.abspath(model_dir), 'horizon': K, 'horizons': list(range(1, K + 1)),
           'report_horizons': report, 'stride': int(stride), 'context': int(context), 'dt_s': 1.0 / 20.0,
           'channels': names, 'threshold': 0.5,
           'iou': {m: {ch: _nan_none(curves[m][:, ci]) for ci, ch in enumerate(names)} for m in curves},
           'reconstruction_iou': {ch: _nan_none(recon.values()[0])[ci] for ci, ch in enumerate(names)},
           'reward_mae': reward, 'pairs': pairs.tolist(),
           'continue': ({'episodes': cont_n, 'terminal_steps': cont_terms,
                         'brier': [round(float(np.mean(v)), 6) if v else None for v in cont_sq]} if cont_n else
                        {'episodes': 0, 'note': 'no environment episode among the sources: recorded drives have no '
                                                'terminal step, so episode ends are not measured'}),
           'verdict': verdict, 'per_sequence': per_seq, 'notes': notes,
           'baselines': {'copy_last': 'the BEV (and reward) at the start, for every horizon',
                         'constant_velocity': ('recorded drives only: actors within the BEV window at the start and '
                                               'the ego moved at their recorded velocities; map planes shifted by '
                                               'the ego\'s motion') if has_cv else 'not run: no recorded drive',
                         'train_mean': 'the training units\' mean reward'},
           'sources': [s['name'] for s in seqs], 'cost': {'wall_s': round(time.monotonic() - t0_wall, 1),
                                                          'device': str(dev), 'source': 'measured'}}
    fps = None
    for s in seqs:
        fps = s['report'].get('fps') or fps
    if fps:
        out['dt_s'] = 1.0 / float(fps)
    out_dir = out_dir or os.path.join(model_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    _write_samples(os.path.join(out_dir, 'samples.npz'), cand, data, seqs, K, n_samples)
    out['provenance'] = block({'model': os.path.join(model_dir, MODEL_JSON)}, None,
                              {'sources': out['sources'], 'horizon': K, 'stride': stride, 'context': context})
    write_json(os.path.join(out_dir, 'eval.json'), out)
    return out


def _write_samples(path, cand, data, seqs, K, n):
    """A few starts for the viewer: truth (N, K + 1, H, W) packed from t0, the model's and the baselines'
    predictions (N, K + 1, H, W) (index 0 is the frame at t0 itself), valid (N, K + 1)."""
    if not cand:
        return
    pick = [cand[i] for i in np.linspace(0, len(cand) - 1, min(n, len(cand))).round().astype(int)]
    H, W = data.seqs[0]['bev'].shape[1:]
    N_ = len(pick)
    truth = np.zeros((N_, K + 1, H, W), np.uint8)
    model = np.zeros_like(truth)
    copy = np.zeros_like(truth)
    cvel = np.zeros_like(truth)
    valid = np.zeros((N_, K + 1), bool)
    has_cv = np.zeros(N_, bool)
    names = []
    for j, (si, t0, row, preds, cv) in enumerate(pick):
        q = data.seqs[si]
        T = len(q['state'])
        names.append(f'{q["name"]} step {t0}')
        for k in range(K + 1):
            if t0 + k >= T:
                break
            truth[j, k] = q['bev'][t0 + k]
            copy[j, k] = q['bev'][t0]
            model[j, k] = q['bev'][t0] if k == 0 else DF.pack_bev(preds[k - 1][row])
            valid[j, k] = True
        if cv is not None:
            has_cv[j] = True
            cvel[j, 0] = q['bev'][t0]
            ks = [k for k in range(1, K + 1) if valid[j, k]]
            for k, bits in zip(ks, cv.predict(seqs[si]['first_step'] + t0, q['bev'][t0], ks)):
                cvel[j, k] = bits
    np.savez_compressed(path, truth=truth, model=model, copy_last=copy, constant_velocity=cvel, valid=valid,
                        has_cv=has_cv, names=np.asarray(names))


# ============================================================================================ imagination
def imagine(model_dir: str, out_dir: str, settings: dict | None = None, sources=None, emitter=None, cancel=None,
            argv=None) -> dict:
    """Learn a policy in the world model's imagination (module doc) and write it as a policy folder to out_dir;
    returns its metrics."""
    import torch
    from . import wmnet as N
    from .policy import feature_index, save_policy
    from .provenance import block, new_run, utc_now
    em = emitter or _Quiet()
    o = dict(IMAGINE_DEFAULTS, **{k: v for k, v in (settings or {}).items() if v is not None})
    bad = sorted(set(o) - set(IMAGINE_DEFAULTS))
    if bad:
        raise WMError(f'unknown settings {bad}')
    t0 = time.monotonic()
    dev = _device(o['device'])
    _torch_flags(dev)
    model, doc = load_model(model_dir, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    data, seqs, notes = model_data(doc, sources or doc['data']['train'], em.log)
    feats = list(o['features'] or doc['features'])
    pidx = feature_index(feats, doc['features'])
    torch.manual_seed(_mix(o['seed'], -1))
    C = N.classes()
    actor = C['Actor'](len(feats), o['hidden'], o['activation'], float(o['init_std'])).to(dev)
    critic = C['mlp'](model.n_feat, 256, 1).to(dev)
    slow = C['mlp'](model.n_feat, 256, 1).to(dev)
    slow.load_state_dict(critic.state_dict())
    for p in slow.parameters():
        p.requires_grad_(False)
    aopt = torch.optim.Adam(actor.parameters(), lr=float(o['actor_lr']), eps=1e-5)
    copt = torch.optim.Adam(critic.parameters(), lr=float(o['critic_lr']), eps=1e-5)
    pidx_t = torch.as_tensor(pidx, device=dev)
    scale_ema = None
    hist = []
    acc, n_acc = {}, 0
    H, gamma, lam = int(o['horizon']), float(o['gamma']), float(o['lambda'])
    for it in range(int(o['iterations'])):
        if cancel and cancel():
            from .training import Cancelled
            raise Cancelled()
        rng = np.random.default_rng([int(o['seed']), it])
        torch.manual_seed(_mix(o['seed'], it))
        b = to_tensors(data.windows(rng, o['batch'], o['seq_len']), dev)
        with torch.no_grad():
            h, z, _, _ = model.observe(b['bev'], b['state'], b['action'], b['first'])
            keep = b['mask'].reshape(-1) > 0
            h, z = h.reshape(-1, h.shape[-1])[keep], z.reshape(-1, z.shape[-1])[keep]
        feats_t, ents = [model.feat(h, z)], []
        for _k in range(H):
            x = model.state_head(feats_t[-1])[:, pidx_t]
            a, ent = actor(x)
            h, z, _ = model.img_step(h, z, a)
            feats_t.append(model.feat(h, z))
            ents.append(ent)
        Fs = torch.stack(feats_t)                                # (H + 1, N, n_feat)
        rew = model.reward(Fs[1:])                               # (H, N)
        cont = torch.sigmoid(model.cont_head(Fs[1:])[..., 0])
        val = N.symexp(critic(Fs)[..., 0])                       # (H + 1, N)
        R = [val[-1]]
        for k in reversed(range(H)):
            R.append(rew[k] + gamma * cont[k] * ((1.0 - lam) * val[k + 1] + lam * R[-1]))
        R = torch.stack(list(reversed(R[1:])))                   # (H, N): R_0 … R_{H−1}
        with torch.no_grad():
            w = torch.cumprod(torch.cat([torch.ones_like(cont[:1]), gamma * cont[:-1]]), 0)
            lo, hi = torch.quantile(R.detach().flatten().float(), torch.tensor([0.05, 0.95], device=dev))
            rng_ = float(hi - lo)
            scale_ema = rng_ if scale_ema is None else 0.99 * scale_ema + 0.01 * rng_
            scale = max(1.0, scale_ema)
        ent = torch.stack(ents)
        actor_loss = -(w * R).mean() / scale - float(o['entropy']) * (w * ent).mean()
        aopt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), float(o['grad_clip']))
        aopt.step()
        tgt = N.symlog(R.detach())
        Fd = Fs[:-1].detach()
        pred = critic(Fd)[..., 0]
        critic_loss = (w * ((pred - tgt) ** 2 + (pred - slow(Fd)[..., 0]) ** 2)).mean()
        copt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), float(o['grad_clip']))
        copt.step()
        with torch.no_grad():
            for ps, pc in zip(slow.parameters(), critic.parameters()):
                ps.mul_(1.0 - float(o['slow_critic'])).add_(float(o['slow_critic']) * pc)
        parts = {'actor_loss': float(actor_loss.detach()), 'critic_loss': float(critic_loss.detach()),
                 'imagined_return': float(R[0].detach().mean()), 'imagined_reward': float(rew.detach().mean()),
                 'continue': float(cont.detach().mean()), 'return_scale': scale,
                 'action_std': float(torch.exp(actor.log_std.detach()).mean())}
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        if (it + 1) % int(o['log_every']) == 0 or it + 1 == int(o['iterations']):
            row = {'it': it + 1, **{k: round(v / n_acc, 5) for k, v in acc.items()}}
            hist.append(row)
            em.metric('imagined_return', row['imagined_return'], it=it + 1)
            em.progress((it + 1) / int(o['iterations']), f'iteration {it + 1}: imagined return '
                                                         f'{row["imagined_return"]:.3f}')
            acc, n_acc = {}, 0
    norm = {'mean': [doc['normaliser']['mean'][i] for i in pidx], 'std': [doc['normaliser']['std'][i] for i in pidx],
            'clip': doc['normaliser']['clip']}
    metrics = {'history': hist, 'final': hist[-1] if hist else None, 'world_model': os.path.abspath(model_dir),
               'data_steps': int(sum(len(q['state']) for q in data.seqs)), 'source': 'measured (in imagination; '
               'the real environment\'s numbers come from falcon policy rollout or leaderboard)',
               'wall_s': round(time.monotonic() - t0, 1), 'device': str(dev), 'notes': notes}
    ins = {'world_model': os.path.join(model_dir, MODEL_JSON)}
    settings_doc = dict(o, features=feats)
    run = new_run('worldmodel-imagine', list(argv or ['worldmodel', 'imagine']), int(o['seed']), settings_doc, ins)
    run.update(status='ok', exit_code=0, finished=utc_now(), name=o['name'] or 'Dreamer-style imagination',
               summary={'imagined_return': metrics['final']['imagined_return'] if hist else None},
               performance={'wall_s': metrics['wall_s'], 'device': str(dev), 'source': 'measured'})
    pol = {'name': o['name'] or 'imagination (world model)', 'trained_by': 'dreamer', 'features': feats,
           'normaliser': norm, 'bev': None, 'action': {'kind': 'continuous', 'keys': list(DF.ACTION_KEYS)},
           'net': actor.spec, 'provenance': block(ins, int(o['seed']), settings_doc),
           'training': {'world_model': os.path.abspath(model_dir), 'iterations': int(o['iterations']),
                        'horizon': H, 'data_steps': metrics['data_steps'], 'seed': int(o['seed'])}}
    actor.net.cpu()
    save_policy(out_dir, actor.net, pol, metrics, run)
    return metrics


# ============================================================================================ the command
def job_spec(argv: list, title: str, memory_gb: float = 4.0, needs_gpu: bool = True):
    """A `falcon worldmodel …` job: needs_gpu for training, evaluation and imagination (the JobQueue never runs it
    while a CARLA job holds the GPU; the process knows the queue holds the GPU lock)."""
    import sys
    from .jobs import JobSpec
    env = dict(os.environ, FALCON_GPU_LOCK_HELD='1') if needs_gpu else None
    return JobSpec('worldmodel', title, [sys.executable, '-m', 'falcon_autonomy', 'worldmodel'] + list(argv),
                   env=env, needs_gpu=needs_gpu, memory_gb=memory_gb, resumable=argv[:1] == ['train'])


def table(ev: dict) -> str:
    """eval.json as text: IoU per channel at the reported horizons, model against each baseline; reward error."""
    lines = []
    ms = list(ev['iou'])
    hdr = f'{"channel":15s}' + ''.join(f'{"k=" + str(r):>8s}' * 1 for r in ev['report_horizons'])
    for m in ms:
        lines.append(f'IoU, {m}')
        lines.append(hdr)
        for ch in ev['channels']:
            vals = ev['iou'][m][ch]
            lines.append(f'{ch:15s}' + ''.join(f'{("—" if vals[r - 1] is None else f"{vals[r - 1]:.3f}"):>8s}'
                                               for r in ev['report_horizons']))
    lines.append('reward mean absolute error')
    for m, v in ev['reward_mae'].items():
        lines.append(f'{m:15s}' + ''.join(f'{("—" if v[r - 1] is None else f"{v[r - 1]:.4f}"):>8s}'
                                          for r in ev['report_horizons']))
    lines.append('pairs per horizon: ' + ', '.join(f'k={r}: {ev["pairs"][r - 1]}' for r in ev['report_horizons']))
    return '\n'.join(lines)


def main_worldmodel(argv) -> int:
    """`falcon worldmodel train|eval|imagine|collect|e2e|e2e-offline|show …` (docs/world-model-training.md):
    protocol lines on stdout; 0 ok, 2 failed, 3 cancelled."""
    import argparse
    from . import protocol as P
    from . import provenance as PV
    from . import training as T
    from .policy import _kv
    ap = argparse.ArgumentParser(prog='falcon worldmodel', description='learned world model, policy learning in '
                                                                       'imagination, end-to-end driving')
    sub = ap.add_subparsers(dest='action', required=True)
    tr = sub.add_parser('train', help='train a world model on drives and experience folders (resumable)')
    tr.add_argument('--out', '-o', default='')
    tr.add_argument('--workspace', default='')
    tr.add_argument('--id', default='')
    tr.add_argument('--config', help='a JSON file of settings (the flags override it)')
    tr.add_argument('--run', action='append', default=None, help='a recorded drive or an experience folder')
    tr.add_argument('--val', action='append', default=None, help='a held-out drive or experience folder')
    for k, t in (('iterations', int), ('batch', int), ('seq-len', int), ('seed', int), ('lr', float),
                 ('val-fraction', float)):
        tr.add_argument(f'--{k}', type=t)
    tr.add_argument('--device')
    tr.add_argument('--option', action='append', default=[], metavar='KEY=VALUE')
    ev = sub.add_parser('eval', help='open-loop prediction quality against baselines')
    ev.add_argument('--model', required=True)
    ev.add_argument('--run', action='append', default=None, help='drives or experience (default: held-out units)')
    ev.add_argument('--out', '-o', default='')
    ev.add_argument('--horizon', type=int, default=20)
    ev.add_argument('--stride', type=int, default=5)
    ev.add_argument('--context', type=int, default=5)
    ev.add_argument('--device', default='auto')
    im = sub.add_parser('imagine', help='learn a policy in the world model\'s imagination')
    im.add_argument('--model', required=True)
    im.add_argument('--out', '-o', required=True, help='the policy folder')
    im.add_argument('--run', action='append', default=None, help='start states from these (default: training units)')
    for k, t in (('iterations', int), ('horizon', int), ('seed', int)):
        im.add_argument(f'--{k}', type=t)
    im.add_argument('--device')
    im.add_argument('--option', action='append', default=[], metavar='KEY=VALUE')
    co = sub.add_parser('collect', help='environment episodes for the world model (an experience folder)')
    from .policy import env_args
    env_args(co)
    co.add_argument('--behaviour', default='scripted', help='scripted, random or a policy folder')
    co.add_argument('--episodes', type=int, default=10)
    co.add_argument('--scenario', default='', help='a built-in or project scenario; empty: free driving')
    co.add_argument('--sigma', type=float, default=D.OU['sigma'])
    co.add_argument('--theta', type=float, default=D.OU['theta'])
    co.add_argument('--start-speed', type=float, default=None, help='free driving: km/h at the start')
    e2 = sub.add_parser('e2e', help='end-to-end driving: a camera(+BEV)-to-control policy by behaviour cloning')
    e2.add_argument('--run', action='append', required=True)
    e2.add_argument('--out', '-o', required=True)
    e2.add_argument('--epochs', type=int, default=None)
    e2.add_argument('--seed', type=int, default=0)
    e2.add_argument('--bev', action='store_true')
    e2.add_argument('--device', default='auto')
    e2.add_argument('--option', action='append', default=[], metavar='KEY=VALUE')
    eo = sub.add_parser('e2e-offline', help='an end-to-end policy\'s action error on drives')
    eo.add_argument('--policy', required=True)
    eo.add_argument('--run', action='append', required=True)
    eo.add_argument('--out', '-o', default='')
    eo.add_argument('--device', default='cpu')
    sh = sub.add_parser('show', help='a world model\'s or an evaluation\'s numbers')
    sh.add_argument('path')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('worldmodel')
    if a.action == 'show':
        try:
            p = a.path
            if os.path.isfile(os.path.join(p, 'eval', 'eval.json')):
                p = os.path.join(p, 'eval')
            if os.path.isfile(os.path.join(p, 'eval.json')):
                print(table(PV.read_json(os.path.join(p, 'eval.json'))), flush=True)
            else:
                d = PV.read_json(os.path.join(p, MODEL_JSON))
                print(json.dumps({k: d[k] for k in ('name', 'method', 'config', 'data', 'cost')}, indent=2),
                      flush=True)
        except (OSError, ValueError, KeyError) as e:
            print(f'failed: {e}', flush=True)
            return P.FAILED
        em.result('ok')
        return P.OK
    flag = T.CancelFlag().install()
    locks = []

    def gpu(device):
        if _device(device or 'auto').type == 'cuda':
            locks.extend(T.gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1'))
    try:
        if a.action == 'train':
            s = PV.read_json(a.config) if a.config else {}
            s.update(_kv(a.option))
            s.update({k: v for k, v in {'iterations': a.iterations, 'batch': a.batch, 'seq_len': a.seq_len,
                                        'seed': a.seed, 'lr': a.lr, 'val_fraction': a.val_fraction,
                                        'device': a.device, 'id': a.id or None}.items() if v is not None})
            if a.run:
                s['train'] = list(a.run)
            if a.val:
                s['val'] = list(a.val)
            s = normalise_settings(s)
            out = a.out or (os.path.join(a.workspace, 'worldmodels', a.id or time.strftime('%Y%m%d-%H%M%S-rssm'))
                            if a.workspace else '')
            if not out:
                raise WMError('--out DIR or --workspace W is needed')
            gpu(s['device'])
            doc = train(s, out, em, flag, ['worldmodel'] + list(argv))
            print(f'world model {out}: {doc["cost"]["iterations"]} iterations in {doc["cost"]["train_s"]} s on '
                  f'{doc["cost"]["device"]}; {doc["cost"]["train_steps"]} training steps, '
                  f'{doc["cost"]["val_steps"]} held out', flush=True)
            em.result('ok', model=os.path.join(out, MODEL_JSON))
            return P.OK
        if a.action == 'eval':
            gpu(a.device)
            doc = evaluate(a.model, a.run, a.out, a.horizon, a.stride, a.context, device=a.device, emitter=em,
                           cancel=flag)
            print(table(doc), flush=True)
            em.result('ok', eval=os.path.join(a.out or os.path.join(a.model, 'eval'), 'eval.json'))
            return P.OK
        if a.action == 'imagine':
            o = _kv(a.option)
            o.update({k: v for k, v in {'iterations': a.iterations, 'horizon': a.horizon, 'seed': a.seed,
                                        'device': a.device}.items() if v is not None})
            gpu(o.get('device', 'auto'))
            m = imagine(a.model, a.out, o, a.run, em, flag, ['worldmodel'] + list(argv))
            print(f'policy {a.out}: imagined return {m["final"]["imagined_return"]:.3f} after '
                  f'{m["final"]["it"]} iterations ({m["wall_s"]} s); evaluate it in the environment with falcon '
                  'policy rollout or leaderboard', flush=True)
            em.result('ok', policy=a.out)
            return P.OK
        if a.action in ('e2e', 'e2e-offline'):
            from . import e2e as E
            if a.action == 'e2e':
                o = dict(_kv(a.option), seed=a.seed, bev=a.bev, device=a.device)
                if a.epochs is not None:
                    o['epochs'] = a.epochs
                gpu(a.device)
                m = E.train_e2e(a.run, a.out, o, log=em.log, progress=em.progress, cancel=flag)
                print(f'validation action MAE {m["val"]["mae"]:.4f} (constant baseline '
                      f'{m["baseline_val"]["mae"]:.4f}); policy in {a.out}', flush=True)
                em.result('ok', policy=a.out)
                return P.OK
            doc = E.offline_eval(a.policy, a.run, a.out, a.device)
            print(json.dumps({'policy': doc['overall'], 'baseline_mean_action': doc['baseline_mean_action']},
                             indent=2), flush=True)
            em.result('ok')
            return P.OK
        # collect
        from .policy import open_env
        opts = {}
        if a.start_speed is not None:
            opts['start_speed_kmh'] = float(a.start_speed)
        a.bev = True
        svc = env = None
        try:
            svc, env = open_env(a, os.path.join(a.out, 'env'), a.scenario or None, 'continuous', log_actions=False)
            doc = D.collect(env, a.behaviour, a.episodes, a.out, a.seed, opts or None, a.theta, a.sigma,
                            progress=lambda k, r: em.progress(k / a.episodes, f'episode {k}: {r["steps"]} steps, '
                                                                             f'{r["termination"]}'), cancel=flag)
        finally:
            if env is not None:
                env.close()
            if svc is not None and not a.keep_server:
                svc.shutdown()
        print(f'experience {a.out}: {len(doc["episodes"])} episodes, {doc["steps"]} steps', flush=True)
        em.result('ok', experience=os.path.join(a.out, D.EXPERIENCE))
        return P.OK
    except T.Cancelled:
        em.result('cancelled')
        return P.CANCELLED
    except (WMError, T.TrainingError, ValueError, OSError, KeyError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        return P.FAILED
    finally:
        for lk in locks:
            lk.release()
