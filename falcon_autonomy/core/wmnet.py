"""The learned world model's networks and the imagination actor-critic (torch, imported only inside functions: the GUI
imports nothing here at start-up). core.wmtrain trains and evaluates them; docs/world-model-training.md is the guide.

The world model is a recurrent state-space model (RSSM) in the form of DreamerV3 (Hafner, Pasukonis, Ba and
Lillicrap 2023, "Mastering Diverse Domains through World Models", arXiv:2301.04104; the RSSM itself from PlaNet,
Hafner et al. 2019):

    deterministic state   h_t = GRU(h_{t−1}, MLP([z_{t−1}, a_{t−1}]))
    prior                 ẑ_t ~ p(ẑ_t | h_t)                    (what the model imagines without the observation)
    posterior             z_t ~ q(z_t | h_t, e_t),  e_t = enc(o_t)
    heads on [h_t, z_t]   BEV (8 Bernoulli planes per pixel), state vector (normalised, Gaussian, unit variance),
                          reward (symlog, squared error), continue (Bernoulli: 1 − terminal)

z is `stoch` categorical variables of `classes` classes each, sampled with straight-through gradients and mixed with
1 % uniform ("unimix"), as DreamerV3. The KL is split into the dynamics loss KL(sg(q) ‖ p) (weight β_dyn 0.5) and the
representation loss KL(q ‖ sg(p)) (β_rep 0.1), each clipped below at free_nats 1 nat (DreamerV3's free bits).
Differences from DreamerV3, all for this data's size and kind: the BEV is binary, so its likelihood is Bernoulli
(binary cross-entropy summed over pixels and planes) rather than a symlog squared error on pixel values; the reward
is regressed in symlog space with a squared error rather than DreamerV3's two-hot discrete regression; the GRU is
torch's GRUCell rather than DreamerV3's layer-normalised GRU; the convolutions use GroupNorm(1, C) (layer norm over
the feature map) and SiLU.

Encoder: four stride-2 convolutions (depth, 2·depth, 4·depth, 8·depth; kernel 4) over the BEV padded to a multiple of
16, flattened, beside a two-layer MLP over the state vector. Decoder: the mirror (transposed convolutions), cropped
back to the BEV's size.

Imagination (Dreamer's actor-critic in the latent space; Hafner et al. 2020, "Dream to Control", ICLR, for the
dynamics-backpropagation actor gradient used here with continuous actions; DreamerV3 for the return normalisation and
the slow critic): from posterior states of the data, the actor picks actions and the frozen world model's prior rolls
the latent forward for `horizon` steps; rewards and continues come from the heads; the critic regresses λ-returns;
the actor maximises them by backpropagating through the model's dynamics. One deliberate difference: the actor reads
the DECODED state vector (the state head's output, in the policy's normalised feature units), not the latent
[h, z]. That makes the trained actor an ordinary feed-forward policy over named features — core.policy's folder
format (net kind mlp, output tanh), which the environment, rollouts and the Leaderboard run without the world model
and without a recurrent state. The price: the actor sees only what the state head can decode, and a decoding error
in imagination is an input error in the real environment.
"""
from __future__ import annotations

import math

_C: dict = {}


def symlog(x):
    import torch
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    import torch
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def classes() -> dict:
    """The torch module classes (defined on first use)."""
    if _C:
        return _C
    import torch
    from torch import nn
    import torch.nn.functional as F

    def mlp(n_in, hidden, n_out, layers=2):
        mods, d = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, n_out))
        return nn.Sequential(*mods)

    class Encoder(nn.Module):
        def __init__(self, channels, hw, depth, n_state, hidden):
            super().__init__()
            self.hw = tuple(hw)
            self.pad = tuple((-s) % 16 for s in hw)
            h, w = hw[0] + self.pad[0], hw[1] + self.pad[1]
            mods, c = [], channels
            for k in range(4):
                mods += [nn.Conv2d(c, depth * 2 ** k, 4, 2, 1), nn.GroupNorm(1, depth * 2 ** k), nn.SiLU()]
                c = depth * 2 ** k
            self.conv = nn.Sequential(*mods, nn.Flatten())
            self.out_conv = c * (h // 16) * (w // 16)
            self.state = mlp(n_state, hidden, hidden, 1)
            self.out = self.out_conv + hidden

        def forward(self, bev, state):
            if any(self.pad):
                bev = F.pad(bev, (0, self.pad[1], 0, self.pad[0]))
            return torch.cat([self.conv(bev), F.silu(self.state(state))], -1)

    class Decoder(nn.Module):
        def __init__(self, channels, hw, depth, n_feat):
            super().__init__()
            self.hw = tuple(hw)
            pad = tuple((-s) % 16 for s in hw)
            self.h0, self.w0 = (hw[0] + pad[0]) // 16, (hw[1] + pad[1]) // 16
            self.c0 = depth * 8
            self.fc = nn.Linear(n_feat, self.c0 * self.h0 * self.w0)
            mods, c = [], self.c0
            for k in (2, 1, 0):
                mods += [nn.ConvTranspose2d(c, depth * 2 ** k, 4, 2, 1), nn.GroupNorm(1, depth * 2 ** k), nn.SiLU()]
                c = depth * 2 ** k
            mods.append(nn.ConvTranspose2d(c, channels, 4, 2, 1))
            self.deconv = nn.Sequential(*mods)

        def forward(self, feat):
            lead = feat.shape[:-1]
            x = self.fc(feat.reshape(-1, feat.shape[-1])).reshape(-1, self.c0, self.h0, self.w0)
            y = self.deconv(x)[:, :, :self.hw[0], :self.hw[1]]
            return y.reshape(*lead, *y.shape[1:])

    class WorldModel(nn.Module):
        """The RSSM and its heads (module doc). cfg: channels, bev_hw, n_state, n_action, deter, stoch, classes,
        hidden, depth, unimix."""

        def __init__(self, cfg):
            super().__init__()
            self.cfg = dict(cfg)
            c = self.cfg
            self.n_stoch = c['stoch'] * c['classes']
            self.n_feat = c['deter'] + self.n_stoch
            self.enc = Encoder(c['channels'], c['bev_hw'], c['depth'], c['n_state'], c['hidden'])
            self.img_in = nn.Sequential(nn.Linear(self.n_stoch + c['n_action'], c['hidden']),
                                        nn.LayerNorm(c['hidden']), nn.SiLU())
            self.gru = nn.GRUCell(c['hidden'], c['deter'])
            self.prior_net = mlp(c['deter'], c['hidden'], self.n_stoch, 1)
            self.post_net = mlp(c['deter'] + self.enc.out, c['hidden'], self.n_stoch, 1)
            self.dec = Decoder(c['channels'], c['bev_hw'], c['depth'], self.n_feat)
            self.state_head = mlp(self.n_feat, c['hidden'], c['n_state'])
            self.reward_head = mlp(self.n_feat, c['hidden'], 1)
            self.cont_head = mlp(self.n_feat, c['hidden'], 1)

        # ---------------------------------------------------------------- the latent
        def logits(self, raw):
            c = self.cfg
            raw = raw.reshape(*raw.shape[:-1], c['stoch'], c['classes'])
            probs = torch.softmax(raw, -1)
            probs = (1.0 - c['unimix']) * probs + c['unimix'] / c['classes']
            return torch.log(probs)

        def sample(self, logits, mode=False):
            """One-hot sample (straight-through gradients) or the mode, flattened."""
            probs = torch.softmax(logits, -1)
            if mode:
                idx = probs.argmax(-1)
            else:
                idx = torch.distributions.Categorical(probs=probs).sample()
            one = F.one_hot(idx, probs.shape[-1]).to(probs.dtype)
            one = one + probs - probs.detach()
            return one.reshape(*one.shape[:-2], -1)

        def initial(self, batch, device):
            return (torch.zeros(batch, self.cfg['deter'], device=device),
                    torch.zeros(batch, self.n_stoch, device=device))

        def img_step(self, h, z, a, mode=False):
            """One step of the prior: (h', z', prior logits)."""
            h = self.gru(self.img_in(torch.cat([z, a], -1)), h)
            lp = self.logits(self.prior_net(h))
            return h, self.sample(lp, mode), lp

        def obs_step(self, h, z, a, e, first, mode=False):
            """One filtering step: reset where `first`, then prior and posterior. (h, z_post, prior, post logits)."""
            keep = (1.0 - first.float()).unsqueeze(-1)
            h, z, a = h * keep, z * keep, a * keep
            h = self.gru(self.img_in(torch.cat([z, a], -1)), h)
            lp = self.logits(self.prior_net(h))
            lq = self.logits(self.post_net(torch.cat([h, e], -1)))
            return h, self.sample(lq, mode), lp, lq

        def observe(self, bev, state, action, first, mode=False, h=None, z=None):
            """Filter a batch of sequences: bev (B, L, C, H, W) float, state (B, L, F), action (B, L, A) — the action
            that led to each step — first (B, L). Returns h, z (B, L, ·), prior and posterior logits."""
            B, L = state.shape[:2]
            e = self.enc(bev.reshape(B * L, *bev.shape[2:]), state.reshape(B * L, -1)).reshape(B, L, -1)
            if h is None:
                h, z = self.initial(B, state.device)
            hs, zs, lps, lqs = [], [], [], []
            for t in range(L):
                h, z, lp, lq = self.obs_step(h, z, action[:, t], e[:, t], first[:, t], mode)
                hs.append(h)
                zs.append(z)
                lps.append(lp)
                lqs.append(lq)
            return torch.stack(hs, 1), torch.stack(zs, 1), torch.stack(lps, 1), torch.stack(lqs, 1)

        def feat(self, h, z):
            return torch.cat([h, z], -1)

        # ---------------------------------------------------------------- heads
        def heads(self, feat):
            return {'bev': self.dec(feat), 'state': self.state_head(feat), 'reward': self.reward_head(feat)[..., 0],
                    'cont': self.cont_head(feat)[..., 0]}

        def reward(self, feat):
            return symexp(self.reward_head(feat)[..., 0])

    def kl(lq, lp):
        """KL(q ‖ p) of the categoricals, summed over the variables: (…,)."""
        q = torch.softmax(lq, -1)
        return (q * (lq - lp)).sum(-1).sum(-1)

    class Actor(nn.Module):
        """The imagination actor: core.policy's MLP (its 'body' is the mean before tanh) and a state-independent log
        standard deviation; actions tanh(mean + std · ε)."""

        def __init__(self, n_in, hidden, activation, init_std=0.5):
            super().__init__()
            from .policy import build_net, net_spec
            self.spec = net_spec('mlp', hidden, activation, 'tanh')
            self.net = build_net(self.spec, n_in, 2)
            self.log_std = nn.Parameter(torch.full((2,), math.log(init_std)))

        def forward(self, x, deterministic=False):
            mu = self.net.body(x)
            if deterministic:
                return torch.tanh(mu), None
            std = torch.exp(self.log_std.clamp(-5.0, 1.0))
            u = mu + std * torch.randn_like(mu)
            ent = (0.5 * math.log(2 * math.pi * math.e) + torch.log(std)).sum().expand(mu.shape[:-1])
            return torch.tanh(u), ent

    _C.update(mlp=mlp, Encoder=Encoder, Decoder=Decoder, WorldModel=WorldModel, kl=kl, Actor=Actor)
    return _C
