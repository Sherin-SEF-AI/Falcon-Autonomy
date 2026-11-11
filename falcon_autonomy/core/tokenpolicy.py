"""A token transformer driving policy (numpy at import; torch and torchvision inside functions): the camera's image
patches, the road users as object tokens, a route token and an ego token, encoded by a small transformer and read out
by a plan query — built so that what it attends to and what it relies on can be measured (core.introspect,
core.biasaudit). Trained by behaviour cloning in the environment (core.gymenv.FalconDriveEnv with token
observations, core.tokenfeatures) and saved as an ordinary policy folder (core.policy, net kind "token"), so
`falcon policy rollout`, the Leaderboard and the adversaries' "agent" ego run it like any other policy.

Tokens (TokenNet; d_model wide, each with a learned type embedding — patch, object, route, ego):
* patches: the camera's RGB (ImageNet-normalised) through a ResNet-18 trunk (He et al. 2016) cut after layer
  `trunk_layers` (default 3: stride 16, so a 128 × 72 image gives 8 × 5 = 40 patches of 256 channels), each cell
  projected to d_model and given learned row and column embeddings. The trunk starts from random weights and is
  trained end to end by the cloning; torchvision's ImageNet weights are used only when spec trunk_weights names an
  entry of core.training.WEIGHTS that is already on disk (pinned SHA-256; nothing is ever downloaded here — ResNet-18
  is not pinned in this build, so the trunk is trained from scratch).
* objects: core.tokenfeatures' object rows (class, position and heading in the ego's frame, velocity, box, brake-light
  bit), scaled by OBJ_SCALE and projected; empty rows are masked out of attention.
* route: core.tokenfeatures.ROUTE_FEATURES of the state vector (normalised by the training data's statistics).
* ego: core.tokenfeatures.EGO_FEATURES and the speed history (normalised by the speed's statistics).
Encoder: `layers` pre-norm transformer blocks (default 4) of `heads`-head self-attention (default 8) and a GELU
feed-forward; a learned plan query cross-attends (the same head count) to the encoded tokens, then a feed-forward and
two tanh heads: steer and accel (core.drivefeatures' continuous action). Attention is written out (Attention: q, k, v
and o projections, the weights returned) so that forward hooks read every head's weights, a pre-hook on o zeroes a
head, and a key mask hides any token — the hook points core.introspect uses, as it does on Qwen2-VL.

Behaviour cloning (train_bc; `falcon policy bc --arch token PROJECT --out DIR`): a privileged expert (StateExpert:
lane keeping on the route's offset, heading and curvature, IDM on the lead's gap and closing speed, a stop for a red
light on the route and for a vulnerable road user in the path — it reads the state vector's lead_*, vru_* and tl_*
features, which the student does not get) drives episodes of the scenarios asked for (seeds seed, seed + 1, …); every
step's observation and the expert's action is a sample. Optional DAgger rounds (Ross, Gordon & Bagnell 2011): the
student drives (with probability 1 − β the student's action is applied, β halving each round), the expert labels.
Training: MSE on both components, AdamW, seeded minibatches; validation split BY EPISODE; the epoch with the lowest
validation loss is kept. Metrics as core.e2e's (MAE per component against the constant baseline).

Limits: the expert is a rule set, not a human; images from the fake carla are flat renderings; the patch grid is
coarse (16 px cells), and a patch's position is its cell's centre (its receptive field is wider).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import drivefeatures as DF
from . import tokenfeatures as TF

TOKEN_DEFAULTS = {
    'd_model': 128, 'heads': 8, 'layers': 4, 'ff': 256, 'trunk_layers': 3, 'trunk_weights': '', 'image': True,
    'width': 128, 'height': 72, 'fov': 90.0, 'tokens': dict(TF.DEFAULTS),
}
BC_DEFAULTS = {
    'episodes': 12, 'scenarios': ['lead_brake', 'cut_in', 'autorickshaw_cut_in', 'free'], 'dagger': 0,
    'beta': 0.5, 'epochs': 15, 'batch': 64, 'lr': 3e-4, 'weight_decay': 1e-4, 'val_fraction': 0.25, 'seed': 0,
    'device': 'cpu', 'max_episode_s': 10.0, 'every': 1, 'name': '',
}
OBJ_SCALE = np.asarray([1.0] + [1.0] * len(TF.OBJ_CLASSES) + [30.0, 10.0, 1.0, 1.0, 10.0, 10.0, 5.0, 2.0, 1.0],
                       np.float32)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TYPES = ('patch', 'object', 'route', 'ego')


def token_spec(o: dict | None = None) -> dict:
    """The net spec of a token policy (policy.json "net"; kind "token")."""
    s = dict(TOKEN_DEFAULTS, **(o or {}))
    if s['d_model'] % s['heads']:
        raise ValueError('d_model must be a multiple of heads')
    if not 1 <= int(s['trunk_layers']) <= 4:
        raise ValueError('trunk_layers is 1–4 (ResNet-18 layer1 … layer4)')
    return {'kind': 'token', 'd_model': int(s['d_model']), 'heads': int(s['heads']), 'layers': int(s['layers']),
            'ff': int(s['ff']), 'trunk_layers': int(s['trunk_layers']), 'trunk_weights': str(s['trunk_weights'] or ''),
            'image': bool(s['image']), 'output': 'tanh', 'tokens': TF.spec(s['tokens'])}


def state_features() -> list:
    """The state features a token policy reads (policy.json "features"): route then ego."""
    return list(TF.ROUTE_FEATURES) + list(TF.EGO_FEATURES)


_CLASSES: dict = {}


def _classes() -> dict:
    if _CLASSES:
        return _CLASSES
    import torch
    from torch import nn

    class Attention(nn.Module):
        """Multi-head attention written out (module doc): forward(xq, xkv, key_mask) → (out, weights (B, h, Tq, Tk));
        key_mask (B, Tk) True hides a key."""

        def __init__(self, d, h):
            super().__init__()
            self.h, self.dk = h, d // h
            self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

        def forward(self, xq, xkv, key_mask=None):
            B, Tq, D = xq.shape
            Tk = xkv.shape[1]
            q = self.q(xq).view(B, Tq, self.h, self.dk).transpose(1, 2)
            k = self.k(xkv).view(B, Tk, self.h, self.dk).transpose(1, 2)
            v = self.v(xkv).view(B, Tk, self.h, self.dk).transpose(1, 2)
            s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
            if key_mask is not None:
                s = s.masked_fill(key_mask[:, None, None, :], float('-inf'))
            w = torch.softmax(s, dim=-1)
            ctx = (w @ v).transpose(1, 2).reshape(B, Tq, D)
            return self.o(ctx), w

    class Block(nn.Module):
        def __init__(self, d, h, ff):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn = Attention(d, h)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

        def forward(self, x, key_mask=None):
            y = self.ln1(x)
            a, _w = self.attn(y, y, key_mask)
            x = x + a
            return x + self.ff(self.ln2(x))

    class TokenNet(nn.Module):
        """The token policy (module doc). forward(inputs) with inputs: camera (B, 3, H, W) float in [0, 1] (when the
        spec has image), objects (B, M, F), obj_present (B, M) bool, route (B, R), ego (B, E), token_mask (B, T) bool
        (optional: True hides a token from every query). Returns (B, 2)."""

        def __init__(self, spec, n_in=0, n_out=2):
            super().__init__()
            import torchvision
            d, h = spec['d_model'], spec['heads']
            self.spec = spec
            self.image = spec['image']
            if self.image:
                r = torchvision.models.resnet18(weights=None)
                parts = [r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4]
                self.trunk = nn.Sequential(*parts[:4 + spec['trunk_layers']])
                ch = (64, 128, 256, 512)[spec['trunk_layers'] - 1]
                self.patch_in = nn.Conv2d(ch, d, 1)
                self.row = nn.Embedding(64, d)
                self.col = nn.Embedding(64, d)
            self.obj_in = nn.Linear(TF.N_OBJ_FEATURES, d)
            self.route_in = nn.Linear(len(TF.ROUTE_FEATURES), d)
            self.ego_in = nn.Linear(len(TF.EGO_FEATURES) + spec['tokens']['history'], d)
            self.type_emb = nn.Embedding(len(TYPES), d)
            self.blocks = nn.ModuleList([Block(d, h, spec['ff']) for _ in range(spec['layers'])])
            self.ln = nn.LayerNorm(d)
            self.query = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.query, std=0.02)
            self.plan = Attention(d, h)
            self.plan_ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, spec['ff']), nn.GELU(), nn.Linear(spec['ff'], d))
            self.head = nn.Linear(d, n_out)
            self.register_buffer('obj_scale', torch.as_tensor(OBJ_SCALE))
            self.register_buffer('img_mean', torch.as_tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('img_std', torch.as_tensor(IMAGENET_STD).view(1, 3, 1, 1))
            self.grid = (0, 0)

        def patches(self, camera):
            f = self.patch_in(self.trunk((camera - self.img_mean) / self.img_std))
            B, D, gh, gw = f.shape
            self.grid = (gh, gw)
            pos = self.row(torch.arange(gh, device=f.device))[:, None, :] + \
                self.col(torch.arange(gw, device=f.device))[None, :, :]
            return f.permute(0, 2, 3, 1).reshape(B, gh * gw, D) + pos.reshape(1, gh * gw, D)

        def tokens(self, inputs):
            """(tokens (B, T, d), key mask (B, T) bool) before the encoder."""
            objs = inputs['objects']
            B, M, _F = objs.shape
            parts, masks = [], []
            if self.image:
                p = self.patches(inputs['camera'])
                parts.append(p + self.type_emb.weight[0])
                masks.append(torch.zeros(B, p.shape[1], dtype=torch.bool, device=p.device))
            parts.append(self.obj_in(objs / self.obj_scale) + self.type_emb.weight[1])
            masks.append(~inputs['obj_present'])
            parts.append((self.route_in(inputs['route']) + self.type_emb.weight[2])[:, None])
            parts.append((self.ego_in(inputs['ego']) + self.type_emb.weight[3])[:, None])
            masks += [torch.zeros(B, 1, dtype=torch.bool, device=objs.device)] * 2
            x, m = torch.cat(parts, 1), torch.cat(masks, 1)
            if inputs.get('token_mask') is not None:
                m = m | inputs['token_mask']
            m[:, -1] = m[:, -1] & ~m.all(1)                  # never every key hidden: the ego token stays
            return x, m

        def forward(self, inputs):
            x, m = self.tokens(inputs)
            for blk in self.blocks:
                x = blk(x, m)
            x = self.ln(x)
            q = self.query.expand(x.shape[0], -1, -1)
            a, _w = self.plan(q, x, m)
            z = q + a
            z = z + self.plan_ff(z)
            return torch.tanh(self.head(z[:, 0]))

        def layout(self, M: int) -> dict:
            """Token index ranges after the last forward: {'patch': (start, stop), 'object': (…), 'route': i,
            'ego': i, 'grid': (gh, gw)}."""
            P = self.grid[0] * self.grid[1] if self.image else 0
            return {'patch': (0, P), 'object': (P, P + M), 'route': P + M, 'ego': P + M + 1, 'grid': self.grid,
                    'n': P + M + 2}

    _CLASSES.update(Attention=Attention, Block=Block, TokenNet=TokenNet)
    return _CLASSES


def build(spec: dict, n_in: int = 0, n_out: int = 2, workspace: str = ''):
    """A TokenNet (core.policy.build_net's 'token'), its trunk from pinned weights when the spec names them."""
    net = _classes()['TokenNet'](spec, n_in, n_out)
    if spec.get('trunk_weights') and spec.get('image'):
        load_trunk(net, spec['trunk_weights'], workspace)
    return net


def load_trunk(net, name: str, workspace: str = ''):
    """Torchvision ResNet-18 weights into the trunk, only from a pinned file already on disk (module doc)."""
    import torch
    from . import training as T
    if name not in T.WEIGHTS:
        raise ValueError(f'{name} is not pinned in core.training.WEIGHTS: the trunk trains from random weights '
                         '(nothing is downloaded here)')
    if T.WEIGHTS[name].get('arch') != 'resnet18':
        raise ValueError(f'{name} is not ResNet-18 weights')
    dirs = [os.path.join(workspace, 'models', 'pretrained')] if workspace else []
    dirs.append(os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints'))
    path, notes = T.find_weights(name, dirs)
    if path is None:
        raise ValueError(f'{name} is not on disk ({"; ".join(notes) or "fetch it first, after its licence"})')
    sd = torch.load(path, map_location='cpu', weights_only=True)
    keys = ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')
    mods = dict(zip(keys[:2 + net.spec['trunk_layers']], [0, 1] + [4 + k for k in range(net.spec['trunk_layers'])]))
    for prefix, idx in mods.items():
        sub = {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + '.')}
        net.trunk[idx].load_state_dict(sub)


# ============================================================================================ observations → inputs
def inputs_np(obs: dict, norm: dict, features: list, names=DF.FEATURE_NAMES, image: bool = True) -> dict:
    """One observation (the environment's dict with tokens) as numpy inputs (no batch axis): camera (3, H, W) float32
    in [0, 1], objects, obj_present, route, ego, object_ids. ValueError when the token arrays are missing."""
    if 'objects' not in obs:
        raise ValueError('this policy reads object tokens: open the environment with tokens (falcon policy rollout '
                         'does it for a token policy)')
    from .policy import feature_index
    st = np.asarray(obs['state'], np.float64)[feature_index(features, names)]
    mean, std = np.asarray(norm['mean']), np.asarray(norm['std'])
    x = np.clip((st - mean) / std, -norm['clip'], norm['clip']).astype(np.float32)
    nr = len(TF.ROUTE_FEATURES)
    k_v = features.index('speed_mps')
    hist = (np.asarray(obs['ego_history'], np.float64) - mean[k_v]) / std[k_v]
    out = {'objects': np.asarray(obs['objects'], np.float32),
           'obj_present': np.asarray(obs['objects'], np.float32)[:, TF.OI['present']] > 0.5,
           'route': x[:nr],
           'ego': np.concatenate([x[nr:], np.clip(hist, -norm['clip'], norm['clip'])]).astype(np.float32),
           'object_ids': np.asarray(obs['object_ids'], np.int64)}
    if image:
        if obs.get('camera') is None:
            raise ValueError('this policy reads the camera: open the environment with its camera')
        out['camera'] = np.ascontiguousarray(np.asarray(obs['camera'], np.uint8).transpose(2, 0, 1)).astype(
            np.float32) / 255.0
    return out


def to_torch(batch: list, device='cpu') -> dict:
    """A list of inputs_np dicts as one batch of tensors (object_ids stay numpy)."""
    import torch
    out = {}
    for k in batch[0]:
        if k == 'object_ids':
            out[k] = np.stack([b[k] for b in batch])
            continue
        arr = np.stack([b[k] for b in batch])
        out[k] = torch.as_tensor(arr, device=device)
    if 'token_mask' in batch[0]:
        out['token_mask'] = out['token_mask'].bool()
    return out


def policy_act(pol, obs: dict, names=DF.FEATURE_NAMES):
    """A token policy's action for one observation (core.policy.Policy.act)."""
    import torch
    inp = to_torch([inputs_np(obs, pol.norm, pol.features, names, pol.doc['net'].get('image', True))], pol.device)
    with torch.no_grad():
        y = pol.net(inp).cpu().numpy()[0]
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ============================================================================================ the expert
class StateExpert:
    """A privileged rule-based driver on the state vector (module doc): act(obs) → [steer, accel]."""

    K_LAT, K_HEAD, WHEELBASE, STEER_MAX_DEG = 0.25, 1.2, 2.9, 70.0
    A_MAX, B_COMF, S0, T_HEAD = 2.0, 3.0, 3.0, 1.4

    def __init__(self, names=DF.FEATURE_NAMES):
        self.i = {n: k for k, n in enumerate(names)}

    def accel(self, s) -> float:
        g = lambda n: float(s[self.i[n]])            # noqa: E731
        v, v0 = g('speed_mps'), max(g('target_speed_mps'), 0.5)

        def idm(gap, dv):
            s_star = self.S0 + max(v * self.T_HEAD + v * dv / (2 * math.sqrt(self.A_MAX * self.B_COMF)), 0.0)
            return self.A_MAX * (1 - (v / v0) ** 4 - (s_star / max(gap, 0.1)) ** 2)
        a = self.A_MAX * (1 - (v / v0) ** 4)
        if g('lead_present') > 0.5:
            a = min(a, idm(g('lead_gap_m'), g('lead_closing_mps')))
        if g('vru_present') > 0.5:
            a = min(a, idm(g('vru_ahead_m'), v))
        if g('tl_known') > 0.5 and (g('tl_red') > 0.5 or g('tl_yellow') > 0.5) and g('tl_distance_m') < DF.TL_RANGE_M:
            a = min(a, idm(g('tl_distance_m'), v))
        return a

    def act(self, obs, names=None):
        s = np.asarray(obs['state'], np.float64)
        g = lambda n: float(s[self.i[n]])            # noqa: E731
        ff = math.degrees(math.atan(self.WHEELBASE * g('curvature_10m'))) / self.STEER_MAX_DEG
        steer = ff - self.K_LAT * g('lateral_offset_m') - self.K_HEAD * g('heading_error_rad')
        a = self.accel(s)
        acc = a / 3.0 if a >= 0 else a / 8.0         # the kinematic model's full throttle and full brake
        return np.clip(np.asarray([steer, acc], np.float32), -1.0, 1.0)


# ============================================================================================ cloning
def _episode_options(o: dict, k: int) -> dict:
    scn = o['scenarios'][k % len(o['scenarios'])]
    return {'scenario': None if scn in ('', 'free') else scn, 'max_episode_s': float(o['max_episode_s'])}


def collect(env, expert, episodes: int, o: dict, seed: int, student=None, beta: float = 1.0, log=None) -> list:
    """[(episode, observation, expert action)] from `episodes` episodes: the expert's action applied with
    probability beta, else the student's (DAgger); every step labelled by the expert."""
    rng = np.random.default_rng([int(seed), 4049])
    out = []
    for k in range(int(episodes)):
        obs, _info = env.reset(seed=int(seed) + k, options=_episode_options(o, k))
        done, n = False, 0
        while not done:
            a_exp = expert.act(obs)
            if n % max(int(o['every']), 1) == 0:
                out.append((k, {kk: v for kk, v in obs.items() if kk != 'depth'}, a_exp))
            a = a_exp if student is None or rng.random() < beta else student(obs)
            obs, _r, term, trunc, _i = env.step(a)
            done = term or trunc
            n += 1
        if log:
            log(f'episode {k}: {n} steps')
    return out


def train_bc(env, out_dir: str, options: dict | None = None, spec: dict | None = None, log=None, progress=None,
             cancel=None) -> dict:
    """Clone the expert into a token policy (module doc) in out_dir; returns its metrics. env: a FalconDriveEnv opened
    with tokens (and the camera when the spec reads the image)."""
    import torch
    from .imitation import action_errors
    from .policy import save_policy
    from .provenance import block, new_run, utc_now
    o = dict(BC_DEFAULTS, **(options or {}))
    sp = token_spec(spec)
    log = log or (lambda m: None)
    t0 = time.monotonic()
    feats = state_features()
    names = env.feature_names
    expert = StateExpert(names)
    data = collect(env, expert, o['episodes'], o, o['seed'], log=log)
    rounds = []
    torch.manual_seed(int(o['seed']))
    dev = torch.device(o['device'])

    def fit(samples, epochs):
        eps = sorted({e for e, _, _ in samples})
        rng = np.random.default_rng(int(o['seed']))
        order = list(rng.permutation(eps))
        n_val = min(max(int(round(o['val_fraction'] * len(eps))), 1), len(eps) - 1) if len(eps) > 1 else 0
        val_eps = set(order[:n_val])
        fi = features_index(feats, names)
        X = np.stack([np.asarray(ob['state'], np.float64)[fi] for _, ob, _ in samples])
        tr = [i for i, (e, _, _) in enumerate(samples) if e not in val_eps]
        va = [i for i, (e, _, _) in enumerate(samples) if e in val_eps] or tr
        mean, std = X[tr].mean(0), X[tr].std(0)
        norm = {'mean': mean.tolist(), 'std': np.where(std < 1e-6, 1.0, std).tolist(), 'clip': 10.0}
        inp = [inputs_np(ob, norm, feats, names, sp['image']) for _, ob, _ in samples]
        Y = np.stack([a for _, _, a in samples]).astype(np.float32)
        net = build(sp, len(feats), 2).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=float(o['lr']), weight_decay=float(o['weight_decay']))

        def predict(ids):
            net.eval()
            outs = []
            with torch.no_grad():
                for k in range(0, len(ids), 128):
                    outs.append(net(to_torch([inp[i] for i in ids[k:k + 128]], dev)).cpu().numpy())
            net.train()
            return np.concatenate(outs) if outs else np.zeros((0, 2), np.float32)
        best, best_state, best_ep, hist = math.inf, None, 0, []
        for ep in range(int(epochs)):
            if cancel and cancel():
                from .training import Cancelled
                raise Cancelled()
            perm = rng.permutation(tr)
            tot = 0.0
            for k in range(0, len(perm), int(o['batch'])):
                sel = perm[k:k + int(o['batch'])]
                y = torch.as_tensor(Y[sel], device=dev)
                loss = torch.mean((net(to_torch([inp[i] for i in sel], dev)) - y) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(sel)
            v = float(np.mean((predict(va) - Y[va]) ** 2))
            hist.append({'epoch': ep + 1, 'train_mse': tot / max(len(tr), 1), 'val_mse': v})
            if progress:
                progress((ep + 1) / int(epochs), f'epoch {ep + 1}: validation MSE {v:.4f}')
            if v < best:
                best, best_ep = v, ep + 1
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        net.load_state_dict(best_state)
        base = Y[tr].mean(0)
        m = {'val': action_errors(predict(va), Y[va]), 'train': action_errors(predict(tr), Y[tr]),
             'baseline_val': action_errors(np.broadcast_to(base, Y[va].shape), Y[va]), 'best_epoch': best_ep,
             'history': hist, 'samples': {'train': len(tr), 'val': len(va)},
             'episodes': {'train': len(eps) - n_val, 'val': n_val}}
        return net, norm, m
    net, norm, m = fit(data, o['epochs'])
    beta = float(o['beta'])
    for r in range(int(o['dagger'])):
        class Student:
            def __init__(self, net_, norm_):
                self.net, self.norm, self.features, self.device = net_, norm_, feats, dev
                self.doc = {'net': sp}

            def __call__(self, obs):
                return policy_act(self, obs, names)
        stu = Student(net.eval(), norm)
        more = collect(env, expert, o['episodes'], o, int(o['seed']) + 1000 * (r + 1), stu, beta, log)
        data += [(e + 1000 * (r + 1), ob, a) for e, ob, a in more]
        rounds.append({'round': r + 1, 'beta': beta, 'samples': len(more)})
        beta /= 2.0
        net, norm, m = fit(data, o['epochs'])
    m.update(dagger=rounds, device=str(dev), source='measured', wall_s=round(time.monotonic() - t0, 2),
             expert='core.tokenpolicy.StateExpert (privileged: lead, vulnerable road user and light features)')
    m['val_mae_vs_baseline'] = m['val']['mae'] / max(m['baseline_val']['mae'], 1e-12)
    cam = {'width': int(env.cfg['camera']['width']), 'height': int(env.cfg['camera']['height']),
           'fov': float(env.cfg['camera'].get('fov', 90.0)),
           'pose': dict({'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                        **(env.cfg['camera'].get('pose') or {}))} if sp['image'] else None
    settings = {k: v for k, v in o.items() if k != 'name'}
    run = new_run('token-bc', [env.run_dir], int(o['seed']), settings, {'environment': os.path.join(env.run_dir,
                                                                                                    'gym.json')})
    run.update(status='ok', exit_code=0, finished=utc_now(), name=o['name'] or 'token policy (cloned)',
               summary={'val_mae': m['val']['mae'], 'baseline_val_mae': m['baseline_val']['mae'],
                        'samples': m['samples']}, performance={'wall_s': m['wall_s'], 'source': 'measured'})
    doc = {'name': o['name'] or 'token policy (cloned)', 'trained_by': 'token-bc', 'features': feats,
           'normaliser': norm, 'bev': None, 'camera': cam, 'tokens': sp['tokens'],
           'action': {'kind': 'continuous', 'keys': list(DF.ACTION_KEYS)}, 'net': sp,
           'provenance': block({'environment': os.path.join(env.run_dir, 'gym.json')}, seed=int(o['seed']),
                               settings=settings),
           'training': {'episodes': int(o['episodes']), 'scenarios': list(o['scenarios']), 'dagger': int(o['dagger']),
                        'samples': m['samples'], 'seed': int(o['seed'])}}
    save_policy(out_dir, net.cpu(), doc, m, run)
    return m


def features_index(feats, names):
    from .policy import feature_index
    return feature_index(feats, names)


def main_bc_token(argv) -> int:
    """`falcon policy bc --arch token PROJECT --out DIR [options]` (module doc)."""
    import argparse
    from . import protocol as P
    from .policy import _kv, env_args, open_env
    ap = argparse.ArgumentParser(prog='falcon policy bc --arch token',
                                 description='behaviour cloning of the token transformer policy in the environment')
    env_args(ap)
    ap.add_argument('--arch', default='token', choices=('token',))
    ap.add_argument('--episodes', type=int, default=BC_DEFAULTS['episodes'])
    ap.add_argument('--scenario', action='append', default=None, help='a scenario to drive (repeat; "free" for free '
                                                                      'driving; default: lead_brake, cut_in, '
                                                                      'autorickshaw_cut_in, free)')
    ap.add_argument('--dagger', type=int, default=0, help='DAgger rounds after the first fit')
    ap.add_argument('--epochs', type=int, default=BC_DEFAULTS['epochs'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-image', action='store_true', help='object, route and ego tokens only (no camera)')
    ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE', help='any of BC_DEFAULTS')
    ap.add_argument('--net', action='append', default=[], metavar='KEY=VALUE', help='any of TOKEN_DEFAULTS')
    try:
        a = ap.parse_args(argv)
        opts = dict(_kv(a.option), episodes=a.episodes, dagger=a.dagger, epochs=a.epochs, device=a.device,
                    seed=a.seed)
        if a.scenario:
            opts['scenarios'] = a.scenario
        net = dict(_kv(a.net), image=not a.no_image)
        sp = token_spec(net)
    except SystemExit as e:
        return 0 if e.code == 0 else P.FAILED
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    svc = env = None
    try:
        cam = {'width': TOKEN_DEFAULTS['width'], 'height': TOKEN_DEFAULTS['height'], 'fov': TOKEN_DEFAULTS['fov']} \
            if sp['image'] else None
        svc, env = open_env(a, os.path.join(a.out, 'env'), None, 'continuous', camera=cam, tokens=sp['tokens'])
        m = train_bc(env, a.out, opts, net, log=lambda s: print(s, flush=True),
                     progress=lambda f, t: print(f'{f * 100:5.1f} % {t}', flush=True))
    except (OSError, ValueError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if env is not None:
            env.close()
        if svc is not None and not a.keep_server:
            svc.shutdown()
    print(json.dumps({'val_mae': m['val']['mae'], 'baseline_val_mae': m['baseline_val']['mae'],
                      'samples': m['samples'], 'policy': a.out}, indent=2), flush=True)
    return P.OK
