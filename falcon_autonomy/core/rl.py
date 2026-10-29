"""Reinforcement learning in the environment: PPO in torch (no third-party RL library) and a scenario curriculum
(numpy at import; torch only inside functions).

PPO (Schulman et al. 2017, "Proximal Policy Optimization Algorithms", arXiv:1707.06347), as commonly implemented
(Huang et al. 2022, "The 37 Implementation Details of Proximal Policy Optimization", ICLR blog track):
* separate actor and critic MLPs (tanh, default 64-64, orthogonal initialisation: gain √2 hidden, 0.01 the actor's
  output, 1 the critic's); continuous actions: a Gaussian with the actor's output as its mean and a state-independent
  log standard deviation (init log_std_init), sampled unclipped and clipped to [−1, 1] only when sent to the
  environment (the log-probability is of the unclipped sample); discrete: a categorical over the 25 actions;
* rollouts of n_steps environment steps, then n_epochs passes of shuffled minibatches; generalised advantage estimation
  (Schulman et al. 2016, GAE(λ)); advantages normalised per minibatch; the clipped surrogate objective (clip ε);
  value loss 0.5·(R − V)²; entropy bonus ent_coef; gradient-norm clipping; Adam; optional early stop of an update when
  the approximate KL exceeds 1.5 × target_kl;
* time limits are not terminal: on truncation the last reward is bootstrapped with γ·V(final observation);
* observation normalisation: a running mean and variance of the state features (updated with every observation
  collected), clipped to ±obs_clip; reward scaling: rewards divided by the running standard deviation of the
  discounted return, clipped to ±reward_clip (no shift) — the logged returns are always the environment's own;
* seeded: torch.manual_seed and a numpy generator (episode seeds, curriculum draws) from `seed`;
* on the CPU by default: during RL the CARLA server owns the GPU (and Falcon never runs a GPU job beside a CARLA job);
  small MLPs gain nothing from a GPU at batch sizes this small.

Checkpoints (checkpoint_every updates, keep_checkpoints kept): checkpoints/ckpt-<update>.pt with the networks, the
optimiser, the normalisers, the RNG states, the counters and the curriculum's state; checkpoints/latest.json names
the newest. resume=True continues from it: the updates, episode numbers and RNG streams carry on; the episode that
was in progress at the checkpoint is not resumed (a new one starts), so a resumed run is not bit-identical to an
uninterrupted one. The policy folder (core.policy: the actor's mean / logits and the observation normaliser) is
exported to policy/ at every checkpoint and at the end.

Logs: episodes.jsonl — every episode (update, global step, curriculum stage, seed, options, return, length,
termination, route completion, progress, infraction counts, reward terms, success); updates.jsonl — every update
(policy, value and entropy losses, approximate KL, clip fraction, explained variance, the steps per second, the mean
return of the last 20 episodes, the stage); run.json — the run with provenance (the environment's config, the PPO
config, the curriculum, the seed).

Curriculum: ordered stages, each a built-in scenario (or free driving) with parameter ranges and reset options;
an episode's parameters are drawn uniformly from its stage's ranges (values the stage leaves open are drawn by the
environment from the scenario's own distributions with the episode seed). After each episode its success
(drivefeatures.episode_success with the stage's fail_on and min_progress_m) is recorded; when the stage has at
least `window` episodes and the success rate over its last `window` reaches `threshold`, the next stage starts. The
last stage is never left. DEFAULT_CURRICULUM's stages and thresholds are a heuristic ordering, not tuned; its first
stage starts the ego at its target speed: from standstill, zero-mean exploration noise never builds speed (braking is
8/3 as strong as throttle in the kinematic model) and PPO settles on standing still — measured on the fake.

Honest expectation: PPO from scratch needs on the order of 10⁵–10⁶ environment steps for lane keeping with speed
control on CARLA; a few minutes of training (10³–10⁴ steps) does not drive well. docs/policies.md has what was
measured here.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import drivefeatures as DF
from .policy import RunningMeanStd, feature_index, select_features

PPO_DEFAULTS = {
    'seed': 0, 'total_updates': 50, 'n_steps': 1024, 'n_epochs': 10, 'minibatch': 64, 'lr': 3e-4, 'gamma': 0.99,
    'gae_lambda': 0.95, 'clip': 0.2, 'vf_coef': 0.5, 'ent_coef': 0.0, 'max_grad_norm': 0.5, 'target_kl': None,
    'hidden': [64, 64], 'activation': 'tanh', 'log_std_init': -0.5, 'normalize_obs': True, 'normalize_reward': True,
    'obs_clip': 10.0, 'reward_clip': 10.0, 'checkpoint_every': 5, 'keep_checkpoints': 3, 'device': 'cpu',
    'exclude_features': [], 'cnn': False, 'conv': [16, 32, 32], 'log_actions': False, 'name': '',
}

DEFAULT_CURRICULUM = {
    'window': 20, 'threshold': 0.8,
    'stages': [
        {'name': 'keep the lane (free driving at 30 km/h)', 'scenario': None,
         'options': {'target_speed_kmh': 30.0, 'start_speed_kmh': 30.0, 'max_episode_s': 20.0, 'route_m': 150.0},
         'success': {'min_progress_m': 60.0}},
        {'name': 'gentle lead braking', 'scenario': 'lead_brake',
         'params': {'speed': [20.0, 35.0], 'gap': [25.0, 40.0], 'decel': [3.0, 5.0]},
         'success': {'min_progress_m': 20.0}},
        {'name': 'lead braking', 'scenario': 'lead_brake', 'success': {'min_progress_m': 20.0}},
        {'name': 'cut-in', 'scenario': 'cut_in', 'success': {'min_progress_m': 20.0}},
        {'name': 'pedestrian crossing', 'scenario': 'pedestrian_crossing', 'success': {'min_progress_m': 20.0}},
    ],
}


# ============================================================================================ the curriculum
class Curriculum:
    """Ordered stages with promotion on a success rate (module doc). doc: {'window', 'threshold', 'stages': [{'name',
    'scenario' (built-in key or None), 'params' ({name: [lo, hi] | {'choices': [...]} | value}), 'options' (other
    reset options), 'success' ({'fail_on': [...], 'min_progress_m': m})}]}."""

    def __init__(self, doc: dict):
        if not doc.get('stages'):
            raise ValueError('a curriculum needs at least one stage')
        self.doc = doc
        self.window = int(doc.get('window', 20))
        self.threshold = float(doc.get('threshold', 0.8))
        if self.window < 1 or not 0.0 < self.threshold <= 1.0:
            raise ValueError('window must be ≥ 1 and threshold in (0, 1]')
        self.stages = doc['stages']
        self.stage = 0
        self.results: list = [[] for _ in self.stages]
        self.promotions: list = []

    @classmethod
    def single(cls, scenario=None, options=None):
        """One stage, never promoted: training on one scenario (or free driving)."""
        return cls({'window': 1, 'threshold': 1.0,
                    'stages': [{'name': scenario or 'free driving', 'scenario': scenario, 'options': options or {}}]})

    @property
    def current(self) -> dict:
        return self.stages[self.stage]

    def options(self, rng) -> dict:
        """The reset options of the next episode of the current stage (parameters drawn with rng)."""
        st = self.current
        o = dict(st.get('options') or {})
        o['scenario'] = st.get('scenario')
        values = {}
        for name, r in sorted((st.get('params') or {}).items()):
            if isinstance(r, dict) and 'choices' in r:
                values[name] = r['choices'][int(rng.integers(len(r['choices'])))]
            elif isinstance(r, (list, tuple)) and len(r) == 2:
                values[name] = float(rng.uniform(float(r[0]), float(r[1])))
            else:
                values[name] = r
        if values:
            o['values'] = values
        return o

    def success(self, summary: dict) -> bool:
        crit = self.current.get('success') or {}
        return DF.episode_success(summary, crit.get('fail_on', DF.FAILURE_KINDS), crit.get('min_progress_m', 0.0))

    def record(self, success: bool, episode: int | None = None) -> bool:
        """Record the current stage's episode; True when this promoted to the next stage."""
        res = self.results[self.stage]
        res.append(bool(success))
        if self.stage + 1 < len(self.stages) and len(res) >= self.window and \
                float(np.mean(res[-self.window:])) >= self.threshold - 1e-12:
            self.promotions.append({'from': self.stage, 'to': self.stage + 1, 'episode': episode,
                                    'after_episodes': len(res), 'success_rate': float(np.mean(res[-self.window:]))})
            self.stage += 1
            return True
        return False

    def rate(self) -> float | None:
        res = self.results[self.stage][-self.window:]
        return float(np.mean(res)) if res else None

    def state(self) -> dict:
        return {'stage': self.stage, 'results': self.results, 'promotions': self.promotions}

    def load(self, d: dict):
        self.stage = int(d['stage'])
        self.results = [list(r) for r in d['results']]
        self.promotions = list(d['promotions'])
        return self


def load_curriculum(spec) -> Curriculum:
    """'default', a JSON file path, a dict, or a Curriculum."""
    if isinstance(spec, Curriculum):
        return spec
    if spec in (None, '', 'default'):
        return Curriculum(DEFAULT_CURRICULUM)
    if isinstance(spec, dict):
        return Curriculum(spec)
    with open(spec, encoding='utf-8') as fh:
        return Curriculum(json.load(fh))


# ============================================================================================ the agent
def _agent(n_in, n_act, discrete, cfg, bev_shape=None):
    """(actor, critic, log_std or None): orthogonally initialised (module doc)."""
    import torch
    from torch import nn
    from .policy import build_net, net_spec
    kind = 'cnn' if cfg['cnn'] else 'mlp'
    spec_a = net_spec(kind, cfg['hidden'], cfg['activation'], 'logits' if discrete else 'linear', cfg['conv'])
    spec_c = net_spec(kind, cfg['hidden'], cfg['activation'], 'linear', cfg['conv'])
    actor, critic = build_net(spec_a, n_in, n_act), build_net(spec_c, n_in, 1)
    for net, out_gain in ((actor, 0.01), (critic, 1.0)):
        lin = [m for m in net.body if isinstance(m, nn.Linear)]
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(lin[-1].weight, out_gain)
    log_std = None if discrete else nn.Parameter(torch.full((n_act,), float(cfg['log_std_init'])))
    return actor, critic, log_std, spec_a


def _dist(actor, log_std, x, b, discrete):
    import torch
    out = actor(x, b)
    if discrete:
        return torch.distributions.Categorical(logits=out)
    return torch.distributions.Normal(out, log_std.exp().expand_as(out))


def _logp_ent(dist, a, discrete):
    if discrete:
        return dist.log_prob(a), dist.entropy()
    return dist.log_prob(a).sum(-1), dist.entropy().sum(-1)


def gae(R, V, D, last_v: float, gamma: float, lam: float) -> np.ndarray:
    """Generalised advantage estimates (Schulman et al. 2016) of one stream of n steps: rewards R, values V, D 1
    where the step ended its episode, last_v the value after the last step (used when it did not)."""
    n = len(R)
    adv = np.zeros(n, np.float32)
    g = 0.0
    for t in reversed(range(n)):
        nv = last_v if t == n - 1 else V[t + 1]
        nonterm = 1.0 - D[t]
        delta = R[t] + gamma * nv * nonterm - V[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv


def ppo_update(actor, critic, log_std, opt, params, X, A, LP, adv, ret, cfg: dict, rng, discrete: bool, dev,
               B=None):
    """The PPO update over one batch (module doc): n_epochs passes of shuffled minibatches (the permutation drawn
    from rng), the clipped surrogate, the value loss, the entropy bonus, gradient-norm clipping, the optional early
    stop on the approximate KL. B: packed BEV planes for a CNN, or None. Returns (stats lists, early-stopped)."""
    import torch

    def tens(x):
        return torch.as_tensor(np.asarray(x), device=dev)
    n = len(X)
    Xt, At, LPt, ADVt, RETt = tens(X), tens(A), tens(LP), tens(adv), tens(ret)
    stats = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': [], 'clip_fraction': []}
    stop = False
    mb = int(cfg['minibatch'])
    for _epoch in range(int(cfg['n_epochs'])):
        perm = torch.as_tensor(rng.permutation(n), device=dev)
        for k in range(0, n, mb):
            sel = perm[k:k + mb]
            bb = None
            if B is not None:
                bb = tens(DF.unpack_bev(B[sel.cpu().numpy()]).astype(np.float32))
            d = _dist(actor, log_std, Xt[sel], bb, discrete)
            lp, ent = _logp_ent(d, At[sel], discrete)
            ratio = torch.exp(lp - LPt[sel])
            a_mb = ADVt[sel]
            a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8) if len(sel) > 1 else a_mb
            pg = -torch.min(ratio * a_mb, torch.clamp(ratio, 1 - cfg['clip'], 1 + cfg['clip']) * a_mb).mean()
            v = critic(Xt[sel], bb)[:, 0]
            vl = 0.5 * ((RETt[sel] - v) ** 2).mean()
            loss = pg + float(cfg['vf_coef']) * vl - float(cfg['ent_coef']) * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg['max_grad_norm']))
            opt.step()
            with torch.no_grad():
                lr_ = lp - LPt[sel]
                kl = float(((torch.exp(lr_) - 1) - lr_).mean())
                cf = float(((ratio - 1).abs() > cfg['clip']).float().mean())
            for key, val in (('policy_loss', pg.item()), ('value_loss', vl.item()),
                             ('entropy', ent.mean().item()), ('approx_kl', kl), ('clip_fraction', cf)):
                stats[key].append(val)
            if cfg['target_kl'] is not None and kl > 1.5 * float(cfg['target_kl']):
                stop = True
                break
        if stop:
            break
    return stats, stop


def _write_jsonl(path, row):
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, separators=(',', ':')) + '\n')


def train_ppo(env, out_dir: str, config: dict | None = None, curriculum=None, resume: bool = False,
              log=None) -> dict:
    """Train with PPO in `env` (a FalconDriveEnv or any env with its API and info['episode_summary'] at episode ends)
    and write everything to out_dir (module doc). curriculum: a Curriculum, a curriculum doc, 'default', a file, or
    None (free driving, one stage). Returns the run's summary."""
    import torch
    from .policy import save_policy
    from .provenance import block, new_run, sha256_json, utc_now, write_json
    cfg = dict(PPO_DEFAULTS, **(config or {}))
    log = log or (lambda s: None)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    cur = load_curriculum(curriculum) if curriculum is not None else Curriculum.single()
    discrete = env.action_kind == 'discrete'
    feats = select_features(cfg['exclude_features'])
    idx = feature_index(feats, env.feature_names)
    has_bev = 'bev' in env.observation_space.keys()
    if cfg['cnn'] and not has_bev:
        raise ValueError('cnn needs the environment\'s BEV observation (bev on)')
    n_act = env.action_space.n if discrete else int(env.action_space.shape[0])
    dev = cfg['device']
    torch.manual_seed(int(cfg['seed']))
    actor, critic, log_std, spec_a = _agent(len(feats), n_act, discrete, cfg)
    actor.to(dev)
    critic.to(dev)
    params = list(actor.parameters()) + list(critic.parameters()) + ([log_std] if log_std is not None else [])
    opt = torch.optim.Adam(params, lr=float(cfg['lr']), eps=1e-5)
    obs_rms = RunningMeanStd((len(feats),))
    ret_rms = RunningMeanStd(())
    rng = np.random.default_rng(int(cfg['seed']))
    update, global_step, episode = 0, 0, 0
    ck_dir = os.path.join(out_dir, 'checkpoints')
    latest = os.path.join(ck_dir, 'latest.json')
    if resume:
        if not os.path.isfile(latest):
            raise ValueError(f'nothing to resume: {latest} does not exist')
        with open(latest, encoding='utf-8') as fh:
            ck = torch.load(os.path.join(ck_dir, json.load(fh)['file']), map_location=dev, weights_only=False)
        actor.load_state_dict(ck['actor'])
        critic.load_state_dict(ck['critic'])
        if log_std is not None:
            with torch.no_grad():
                log_std.copy_(ck['log_std'])
        opt.load_state_dict(ck['optimizer'])
        obs_rms.load(ck['obs_rms'])
        ret_rms.load(ck['ret_rms'])
        rng.bit_generator.state = ck['rng']
        torch.set_rng_state(ck['torch_rng'])
        cur.load(ck['curriculum'])
        update, global_step, episode = ck['update'], ck['global_step'], ck['episode']
        log(f'resumed at update {update} (step {global_step}, episode {episode}, stage {cur.stage})')
    env_cfg = getattr(env, 'cfg', {})
    settings = {'ppo': cfg, 'curriculum': cur.doc, 'environment_sha256': sha256_json(env_cfg) if env_cfg else None}
    run = new_run('ppo', [out_dir], int(cfg['seed']), settings,
                  {'environment': os.path.join(env.run_dir, 'gym.json')} if hasattr(env, 'run_dir') else None)
    run['name'] = cfg['name'] or 'PPO'
    if resume:
        run['resumed_at_update'] = update
    write_json(os.path.join(out_dir, 'run.json'), run)
    ep_path, up_path = os.path.join(out_dir, 'episodes.jsonl'), os.path.join(out_dir, 'updates.jsonl')
    recent: list = []

    def norm_obs(state, update_stats):
        x = np.asarray(state, np.float64)[idx]
        if cfg['normalize_obs']:
            if update_stats:
                obs_rms.update(x[None])
            x = np.clip((x - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -cfg['obs_clip'], cfg['obs_clip'])
        return x.astype(np.float32)

    def tens(x):
        return torch.as_tensor(np.asarray(x), device=dev)

    def new_episode():
        nonlocal ep_seed, ep_opts
        ep_seed = int(rng.integers(0, 2 ** 31 - 1))
        ep_opts = cur.options(rng)
        o, _info = env.reset(seed=ep_seed, options=ep_opts)
        return o

    def export(final=False):
        doc = {'name': cfg['name'] or 'PPO', 'trained_by': 'ppo', 'features': feats,
               'normaliser': {'mean': obs_rms.mean.tolist() if cfg['normalize_obs'] else [0.0] * len(feats),
                              'std': np.sqrt(obs_rms.var + 1e-8).tolist() if cfg['normalize_obs'] else
                              [1.0] * len(feats), 'clip': float(cfg['obs_clip']) if cfg['normalize_obs'] else 1e9},
               'bev': dict(DF.BEV_DEFAULT, **(env_cfg.get('bev') or {})) if cfg['cnn'] else None,
               'action': ({'kind': 'discrete', 'n': n_act, 'table': [list(a) for a in DF.DISCRETE_ACTIONS]}
                          if discrete else {'kind': 'continuous', 'keys': list(DF.ACTION_KEYS)}),
               'net': spec_a, 'provenance': block({'environment': os.path.join(env.run_dir, 'gym.json')}
                                                  if hasattr(env, 'run_dir') else None, int(cfg['seed']), settings),
               'training': {'updates': update, 'global_step': global_step, 'episodes': episode,
                            'stage': cur.stage, 'final': final}}
        save_policy(os.path.join(out_dir, 'policy'), actor, doc, {'recent_episodes': recent[-20:],
                                                                 'curriculum': cur.state()})

    def checkpoint():
        name = f'ckpt-{update:06d}.pt'
        torch.save({'update': update, 'global_step': global_step, 'episode': episode, 'actor': actor.state_dict(),
                    'critic': critic.state_dict(), 'log_std': None if log_std is None else log_std.detach(),
                    'optimizer': opt.state_dict(), 'obs_rms': obs_rms.state(), 'ret_rms': ret_rms.state(),
                    'rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'curriculum': cur.state(),
                    'config': cfg}, os.path.join(ck_dir, name))
        write_json(latest, {'file': name, 'update': update, 'global_step': global_step})
        old = sorted(f for f in os.listdir(ck_dir) if f.startswith('ckpt-'))
        for f in old[:-int(cfg['keep_checkpoints'])]:
            os.remove(os.path.join(ck_dir, f))
        export()

    ep_seed, ep_opts = 0, {}
    obs = new_episode()
    ret_run, ep_return, ep_len = 0.0, 0.0, 0
    n = int(cfg['n_steps'])
    t_start = time.monotonic()
    env_wall = 0.0
    while update < int(cfg['total_updates']):
        X = np.zeros((n, len(feats)), np.float32)
        B = np.zeros((n,) + tuple(env.observation_space['bev'].shape[1:]), np.uint8) if cfg['cnn'] else None
        A = np.zeros((n,) if discrete else (n, n_act), np.int64 if discrete else np.float32)
        LP, V, R, D = (np.zeros(n, np.float32) for _ in range(4))
        t0 = time.monotonic()
        env_t = 0.0
        for t in range(n):
            x = norm_obs(obs['state'], True)
            X[t] = x
            if B is not None:
                B[t] = DF.pack_bev(obs['bev'])
            b = tens(obs['bev'][None].astype(np.float32)) if B is not None else None
            with torch.no_grad():
                d = _dist(actor, log_std, tens(x[None]), b, discrete)
                a = d.sample()
                lp, _ = _logp_ent(d, a, discrete)
                v = critic(tens(x[None]), b)[0, 0]
            A[t] = a[0].cpu().numpy()
            LP[t], V[t] = float(lp[0]), float(v)
            send = int(A[t]) if discrete else np.clip(A[t], -1.0, 1.0)
            te = time.monotonic()
            obs, r, term, trunc, info = env.step(send)
            env_t += time.monotonic() - te
            global_step += 1
            ep_return += r
            ep_len += 1
            if cfg['normalize_reward']:
                ret_run = ret_run * float(cfg['gamma']) + r
                ret_rms.update(np.array([ret_run]))
                r = float(np.clip(r / math.sqrt(ret_rms.var + 1e-8), -cfg['reward_clip'], cfg['reward_clip']))
            if trunc and not term:                 # a time limit is not terminal: bootstrap from the final state
                xf = norm_obs(obs['state'], False)
                bf = tens(obs['bev'][None].astype(np.float32)) if B is not None else None
                with torch.no_grad():
                    r += float(cfg['gamma']) * float(critic(tens(xf[None]), bf)[0, 0])
            R[t] = r
            D[t] = float(term or trunc)
            if term or trunc:
                summ = info.get('episode_summary') or {}
                ok = cur.success(summ)
                stage = cur.stage
                promoted = cur.record(ok, episode)
                row = {'episode': episode, 'update': update + 1, 'global_step': global_step, 'stage': stage,
                       'stage_name': cur.stages[stage].get('name'), 'seed': ep_seed, 'options': ep_opts,
                       'return': round(ep_return, 6), 'length': ep_len, 'termination': summ.get('termination'),
                       'route_completion': summ.get('route_completion'), 'progress_m': summ.get('progress_m'),
                       'infractions': {k: v for k, v in (summ.get('infraction_counts') or {}).items() if v},
                       'reward_terms': summ.get('reward_terms'), 'success': ok, 'env_episode': summ.get('episode')}
                _write_jsonl(ep_path, row)
                recent.append({k: row[k] for k in ('episode', 'return', 'length', 'termination', 'route_completion',
                                                   'success', 'stage')})
                recent[:] = recent[-100:]
                if promoted:
                    log(f'curriculum: stage {stage} → {cur.stage} ({cur.current.get("name")}) after episode '
                        f'{episode}')
                    _write_jsonl(ep_path, {'promotion': cur.promotions[-1]})
                episode += 1
                ret_run, ep_return, ep_len = 0.0, 0.0, 0
                obs = new_episode()
        env_wall += env_t
        # ---- advantages
        with torch.no_grad():
            xl = norm_obs(obs['state'], False)
            bl = tens(obs['bev'][None].astype(np.float32)) if B is not None else None
            last_v = float(critic(tens(xl[None]), bl)[0, 0])
        adv = gae(R, V, D, last_v, float(cfg['gamma']), float(cfg['gae_lambda']))
        ret = adv + V
        # ---- the update
        stats, stop = ppo_update(actor, critic, log_std, opt, params, X, A, LP, adv, ret, cfg, rng, discrete, dev, B)
        update += 1
        var_r = float(np.var(ret))
        ev = float(1.0 - np.var(ret - V) / var_r) if var_r > 1e-12 else None
        row = {'update': update, 'global_step': global_step, 'episodes': episode, 'stage': cur.stage,
               **{k: float(np.mean(v)) if v else None for k, v in stats.items()}, 'explained_variance': ev,
               'early_stop': stop, 'env_steps_per_s': round(n / max(env_t, 1e-9), 2),
               'update_wall_s': round(time.monotonic() - t0, 3),
               'mean_return_last20': float(np.mean([r_['return'] for r_ in recent[-20:]])) if recent else None,
               'success_rate_stage': cur.rate()}
        if log_std is not None:
            row['action_std'] = [round(float(s), 4) for s in log_std.detach().exp().cpu()]
        _write_jsonl(up_path, row)
        log(f'update {update}/{cfg["total_updates"]}: step {global_step}, episodes {episode}, mean return (last 20) '
            f'{row["mean_return_last20"] if row["mean_return_last20"] is None else round(row["mean_return_last20"], 2)}'
            f', stage {cur.stage}, {row["env_steps_per_s"]} env steps/s')
        if update % int(cfg['checkpoint_every']) == 0 or update == int(cfg['total_updates']):
            checkpoint()
    export(final=True)
    wall = time.monotonic() - t_start
    summary = {'updates': update, 'global_step': global_step, 'episodes': episode, 'stage': cur.stage,
               'promotions': cur.promotions,
               'mean_return_last20': float(np.mean([r_['return'] for r_ in recent[-20:]])) if recent else None,
               'success_rate_last20': float(np.mean([r_['success'] for r_ in recent[-20:]])) if recent else None}
    run.update(status='ok', exit_code=0, finished=utc_now(), summary=summary,
               performance={'wall_s': round(wall, 3), 'env_wall_s': round(env_wall, 3), 'source': 'measured'})
    write_json(os.path.join(out_dir, 'run.json'), run)
    return summary


# ============================================================================================ the CLI
def main_ppo(argv) -> int:
    import argparse
    from . import protocol as P
    from .gymenv import EnvError
    from .policy import _kv, env_args, open_env
    ap = argparse.ArgumentParser(prog='falcon policy ppo', description='PPO in the environment (docs/policies.md)')
    env_args(ap)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--scenario', default='', help='train on one scenario (a built-in or project id)')
    g.add_argument('--curriculum', default='', help='"default" or a curriculum JSON file')
    ap.add_argument('--updates', type=int, default=PPO_DEFAULTS['total_updates'])
    ap.add_argument('--steps', type=int, default=PPO_DEFAULTS['n_steps'], help='environment steps per update')
    ap.add_argument('--action', default='continuous', choices=('continuous', 'discrete'))
    ap.add_argument('--resume', action='store_true', help='continue from OUT/checkpoints/latest.json')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE', help='any of PPO_DEFAULTS')
    try:
        a = ap.parse_args(argv)
        cfg = dict(_kv(a.option), total_updates=a.updates, n_steps=a.steps, seed=a.seed, device=a.device,
                   cnn=bool(a.bev) and bool(_kv(a.option).get('cnn', True)))
    except SystemExit as e:
        return 0 if e.code == 0 else P.FAILED
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    try:
        cur = load_curriculum(a.curriculum) if a.curriculum else Curriculum.single(a.scenario or None)
    except (OSError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    svc = env = None
    try:
        svc, env = open_env(a, os.path.join(a.out, 'env'), None, a.action, bool(cfg.get('log_actions', False)))
        s = train_ppo(env, a.out, cfg, cur, a.resume, log=lambda m: print(m, flush=True))
    except (EnvError, OSError, ValueError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if env is not None:
            env.close()
        if svc is not None and not a.keep_server:
            svc.shutdown()
    print(json.dumps(s, indent=2), flush=True)
    return P.OK
