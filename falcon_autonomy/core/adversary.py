"""Adversarial multi-agent traffic: learned road users that try to make a fixed ego fail, realistically — the
multi-agent environment's client, the multi-agent PPO that trains them, their evaluation against scripted scenarios
searched with the same budget, the realism metrics, the replay of every failure, and the CLI (numpy at import; torch
only inside functions, so the GUI may import this module). The definitions (roles, observations, actions, fault,
realism, rewards) are core.advfeatures'; the environment worker is workers/adversary.py; docs/adversarial-traffic.md
is the guide.

    falcon adversary train PROJECT --out DIR [--ego follower|none|autopilot|POLICY] [--action lateral|discrete|
                                              continuous] [--curriculum default|FILE] [--updates N --steps N]
                                              [--resume] [--seed S] [--option KEY=VALUE]...
    falcon adversary eval PROJECT --policy DIR --out DIR [--ego SPEC]... [--episodes N] [--baselines LIST]
                                              [--tm-reference N]
    falcon adversary replay PROJECT --episodes EPISODES.jsonl --episode N [--failure K] --out DIR

The environment (AdversaryEnv). The parallel multi-agent API of PettingZoo (Terry et al. 2021, "PettingZoo: Gym for
multi-agent reinforcement learning", NeurIPS): reset(seed, options) → (observations {agent: vector}, info);
step({agent: action}) → (observations, rewards, terminations, truncations, infos), each a dict over the agents
(env.agents: this episode's learned adversaries, in order). Every agent's episode ends with the episode (the ego's
end, an adversary off the road, or the time limit), so terminations and truncations are equal across agents. The ego
is not an agent: it is the system under test, fixed — "follower" (the director's lane follower), "none" (constant
speed), "autopilot" (CARLA's traffic manager; its episodes do not repeat), or a Falcon policy folder (core.policy),
which this client runs on the ego's own observation (env.ego_obs, the gym environment's) at every step. The
single-agent environment (core.gymenv.FalconDriveEnv) is unchanged; this one subclasses it and talks to its own worker.

Training: independent PPO with parameter sharing (IPPO, de Witt et al. 2020, "Is Independent Learning All You Need in
the StarCraft Multi-Agent Challenge?", arXiv:2011.09533; parameter sharing as Gupta, Egorov & Kochenderfer 2017,
"Cooperative Multi-agent Control Using Deep Reinforcement Learning", AAMAS workshops). One actor and one critic serve
every adversary; each sees its own observation (its kind one-hot is in it, so the shared network can behave per
kind), and each adversary's experience is its own stream: GAE per stream, one PPO update over all of them (core.rl's
gae and ppo_update, the same clipped surrogate, normalisers and seeding as the ego's PPO). The critic is decentralised
(its own observation); MAPPO's centralised critic (Yu et al. 2022, "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games", NeurIPS) is not implemented. Parameter sharing is what lets the curriculum grow from one adversary
to several without new networks: the policy learnt with one keeps acting for each of three.

The curriculum (AdvCurriculum): stages of rosters (lists of roles); an episode draws one roster of its stage. The
next stage starts when the share of realistic failures over the last `window` episodes reaches `threshold`, or after
`max_episodes` episodes in the stage (adversaries that find no failure must still meet the harder rosters).
DEFAULT_ADV_CURRICULUM (heuristic, not tuned): one adversary → two → three.

Evaluation (evaluate). For each ego, with the same episode budget per method:
* learned: the trained adversaries, acting deterministically (the Gaussian mean, the argmax), over the policy's
  last-stage rosters, episode seeds seed, seed + 1, …;
* random: the same rosters with uniformly random actions (a seeded generator) — learning against no learning;
* idle: the same rosters holding their lane and speed — the placement alone;
* scripted_random and scripted_adaptive: the built-in scenarios whose actors play the same roles (cut_in, lead_brake,
  two_wheeler_filtering, wrong_way_scooter, autorickshaw_cut_in; the budget split evenly), their parameters searched
  by core.search — a Halton sequence with a seeded shift ("random") and the cross-entropy method for rare events
  ("adaptive"; population 4 after 4–8 uniform starts, so that it adapts within a small budget) — on the objective 1
  for a realistic ego failure, else 0.9 · exp(−min TTC / 1.5 s).
Every episode is judged alike by the worker: the ego's failures (a collision the ego caused, by fault attribution, or a
failure infraction) and the realism of every other road user. The failure rate is the share of episodes with a
realistic ego failure, with Wilson's 95 % interval (core.stack.wilson); the searched baselines' episodes are not
independent draws, so their interval is descriptive. Distinct failures: distinct descriptors (core.advfeatures).

Realism metrics: per method, the adversaries' traffic-rule violations per km (core.rules; all, and those their roles
do not allow), the collisions they caused and their contacts per episode, their wrong-way and off-road time per km,
and their dynamics — |longitudinal acceleration|, |lateral acceleration|, |jerk|: percentiles, the share beyond
REALISM_LIMITS, and against CARLA's traffic-manager traffic (an episode set with background vehicles and no
adversaries, physics) the two-sample Kolmogorov–Smirnov statistic and the Wasserstein-1 distance.

Every failure is replayable (replay): the evaluation's episodes.jsonl holds each episode with its seed, options, the
adversaries' actions (and the ego's when a policy drove it) and the environment's settings; replay re-runs it with a
640 × 360 chase camera (a camera changes nothing in the motion), keeps the frames from 3 s before the failure to 1 s
after, and checks that the same failure happens at the same step. Episodes with physics motion, background traffic or
the autopilot ego are refused (they do not repeat in CARLA 0.9.16). Failures are also written as cases
(cases.jsonl), which the Rare cases page lists beside the searched ones.

Related work: Wachi 2019 ("Failure-Scenario Maker for Rule-Based Agent using Multi-agent Adversarial Reinforcement
Learning and its Application to Autonomous Driving", IJCAI) trains non-player agents with multi-agent RL (MADDPG) to
make a rule-based player fail, with a personal reward beside the adversarial one so that the failures stay natural;
Ding et al. 2023 ("A Survey on Safety-Critical Driving Scenario Generation — A Methodological Perspective", IEEE
T-ITS 24(7)) survey adversarial generation and name fidelity and diversity among its open problems; Hanselmann et al.
2022 ("KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients", ECCV)
optimise adversarial agents' actions through a differentiable kinematic bicycle model in CARLA and keep the ones that
collide with the ego. Implemented here: Wachi's setting (several learned adversaries, a rule-based or learned player),
with realism enforced by fault attribution, traffic rules, dynamics bounds and lane keeping instead of a personal
goal, and a count-based diversity bonus; KING's kinematic bicycle model as the motion (here integrated by the
director, not differentiated). Not implemented: KING's gradient-based optimisation, learned behaviour priors from
data (e.g. STRIVE, Rempe et al. 2022), MAPPO.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import advfeatures as AF
from .gymenv import EnvError, FalconDriveEnv

ADV_POLICY_SCHEMA = 'falcon.adversary_policy/1'
EVAL_SCHEMA = 'falcon.adversary_eval/1'
CASES_SCHEMA = 'falcon.adversary_case/1'
ADV_PPO_DEFAULTS = {
    'seed': 0, 'total_updates': 50, 'n_steps': 512, 'n_epochs': 8, 'minibatch': 128, 'lr': 3e-4, 'gamma': 0.99,
    'gae_lambda': 0.95, 'clip': 0.2, 'vf_coef': 0.5, 'ent_coef': 0.005, 'max_grad_norm': 0.5, 'target_kl': None,
    'hidden': [64, 64], 'activation': 'tanh', 'log_std_init': -0.5, 'normalize_obs': True, 'normalize_reward': True,
    'obs_clip': 10.0, 'reward_clip': 10.0, 'checkpoint_every': 5, 'keep_checkpoints': 3, 'device': 'cpu',
    'cnn': False, 'conv': [16, 32, 32], 'name': '',
}
DEFAULT_ADV_CURRICULUM = {
    'window': 20, 'threshold': 0.3, 'max_episodes': 200,
    'stages': [
        {'name': 'one adversary', 'rosters': [['cut_in_car'], ['lead_car'], ['filtering_bike']]},
        {'name': 'two adversaries', 'rosters': [['cut_in_car', 'filtering_bike'], ['lead_car', 'autorickshaw_cut_in']]},
        {'name': 'three adversaries', 'rosters': [['cut_in_car', 'filtering_bike', 'wrong_way_scooter'],
                                                  ['lead_car', 'autorickshaw_cut_in', 'filtering_bike']]},
    ],
}
SCRIPTED_FAMILIES = ('cut_in', 'lead_brake', 'two_wheeler_filtering', 'wrong_way_scooter', 'autorickshaw_cut_in')
METHODS = ('learned', 'random', 'idle', 'scripted_random', 'scripted_adaptive')
EGO_BUILTIN = ('follower', 'none', 'autopilot')
RESET_TRIES = 5


# ============================================================================================ the config
def adversary_config(project, world_id: str, out_dir: str, port: int, tm_port: int, *, ego: str = 'follower',
                     action: str = 'lateral', roles=None, reward: dict | None = None, realism: dict | None = None,
                     ruleset=None, max_episode_s: float = 15.0, log_samples: bool = False,
                     log_adv_trace: bool = False, log_actions: bool = True, log_trace: bool = False,
                     ego_action: str = 'continuous', bev=None, camera=None, **kw) -> dict:
    """The adversary worker's config: core.gymenv.gym_config (the world, the rig's vehicle as the ego, fps, seed,
    motion, traffic, timeouts) plus the "adversary" block (workers/adversary.py). ego: 'follower', 'none',
    'autopilot' or 'agent' (a policy acting in this process: AdversaryEnv(ego_policy=…)); ego_action: that policy's
    action kind; bev: the BEV spec a CNN ego policy reads."""
    from .gymenv import gym_config
    if action not in AF.ACTION_KINDS:
        raise ValueError(f'action must be one of {AF.ACTION_KINDS}, not {action!r}')
    if ego not in EGO_BUILTIN + ('agent',):
        raise ValueError(f'ego must be one of {EGO_BUILTIN + ("agent",)}, not {ego!r}')
    bad = [r for r in roles or () if r not in AF.ROLES]
    if bad:
        raise ValueError(f'unknown roles {bad} (roles: {sorted(AF.ROLES)})')
    cfg = gym_config(project, world_id, out_dir, port, tm_port, action=ego_action, bev=bev, camera=camera,
                     max_episode_s=max_episode_s, log_actions=log_actions, name=kw.pop('name', '') or
                     'adversarial traffic', **kw)
    cfg['log_trace'] = bool(log_trace)
    cfg['adversary'] = {'ego': ego, 'action': action, 'roles': list(roles or []), 'reward': dict(reward or {}),
                        'realism': dict(realism or {}), 'ruleset': ruleset, 'log_samples': bool(log_samples),
                        'log_adv_trace': bool(log_adv_trace)}
    return cfg


def ego_spec(spec: str) -> tuple:
    """(worker ego driver, policy folder or None) of an ego given as 'follower' | 'none' | 'autopilot' | a policy
    folder (core.policy)."""
    if spec in EGO_BUILTIN:
        return spec, None
    if os.path.isfile(os.path.join(spec, 'policy.json')):
        return 'agent', os.path.abspath(spec)
    raise ValueError(f'the ego must be follower, none, autopilot or a policy folder, not {spec!r}')


# ============================================================================================ the environment
class AdversaryEnv(FalconDriveEnv):
    """The multi-agent environment (module doc). config: adversary_config(); ego_policy: a core.policy.Policy (or a
    callable obs → action) when the config's ego is 'agent'."""

    worker = 'adversary'

    def __init__(self, config: dict, service=None, *, ego_policy=None, **kw):
        if (config.get('adversary') or {}).get('ego') == 'agent' and ego_policy is None:
            raise ValueError('an "agent" ego needs ego_policy (a policy folder loaded with core.policy)')
        self.ego_policy = ego_policy
        self.agents: list = []
        self.ego_obs: dict = {}
        self.last_info: dict = {}
        kw.setdefault('title', 'Adversarial traffic')
        super().__init__(config, service, **kw)
        a = self.ready['adversary']
        self.adv_feature_names = tuple(f[0] for f in a['features'])
        if self.adv_feature_names != AF.ADV_FEATURE_NAMES:
            raise EnvError('the worker\'s adversary features are not this build\'s (core.advfeatures)')
        self.adv_action_kind = a['action']['kind']
        from .gymenv import make_box, make_discrete
        self.adv_action_space = make_discrete(a['action']['n']) if self.adv_action_kind == 'discrete' else \
            make_box(np.asarray(a['action']['low'], np.float32), np.asarray(a['action']['high'], np.float32), (2,),
                     np.float32)
        self.adv_observation_space = make_box(-np.inf, np.inf, (AF.N_ADV_FEATURES,), np.float32)

    def _split(self, obs: dict, info: dict) -> dict:
        arr = obs.pop('adversaries', np.zeros((0, AF.N_ADV_FEATURES), np.float32))
        self.agents = list(info.get('adversaries') or [])
        self.ego_obs = obs
        if len(self.agents) != len(arr):
            raise EnvError(f'{len(arr)} adversary observations for {len(self.agents)} adversaries')
        return {aid: arr[k] for k, aid in enumerate(self.agents)}

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.last_info = info
        return self._split(obs, info), info

    def step(self, actions):
        """actions: {agent: action} for every agent of the episode. Returns (observations, rewards, terminations,
        truncations, infos), dicts over the agents; the episode's shared info is env.last_info (its summary in
        'episode_summary' at the end)."""
        if self._needs_reset:
            raise EnvError('step() needs reset() first: the episode ended or never started')
        cmd = {'cmd': 'step', 'adversaries': {}}
        for aid in self.agents:
            if aid not in actions:
                raise EnvError(f'no action for adversary {aid}')
            a = actions[aid]
            cmd['adversaries'][aid] = int(np.asarray(a).reshape(-1)[0]) if self.adv_action_kind == 'discrete' else \
                [float(x) for x in np.asarray(a, np.float64).reshape(-1)]
        cmd['action'] = None
        if self.ego_policy is not None:
            act = self.ego_policy.act if hasattr(self.ego_policy, 'act') else self.ego_policy
            a = act(self.ego_obs, self.feature_names) if hasattr(self.ego_policy, 'act') else act(self.ego_obs)
            cmd['action'] = int(a) if self.action_kind == 'discrete' else [float(x) for x in np.asarray(a).reshape(-1)]
        return self._step_cmd(cmd)

    def step_replay(self, adv_actions: dict, ego_action=None):
        """A step with given actions for the adversaries and the ego (a replay of logged actions)."""
        cmd = {'cmd': 'step', 'adversaries': adv_actions, 'action': ego_action}
        return self._step_cmd(cmd)

    def _step_cmd(self, cmd):
        h, payload = self._request(cmd)
        term, trunc = bool(h['terminated']), bool(h['truncated'])
        self._needs_reset = term or trunc
        info = h.get('info') or {}
        self.last_info = info
        agents = list(self.agents)
        obs = self._split(self._obs(h, payload), info)
        rewards = {aid: float((info.get('rewards') or {}).get(aid, 0.0)) for aid in agents}
        infos = {aid: {'terms': (info.get('terms') or {}).get(aid)} for aid in agents}
        return obs, rewards, {a: term for a in agents}, {a: trunc for a in agents}, infos


def open_adversary_env(project, world, out_dir, svc, ego: str, *, action='lateral', rig=None, seed=0, fps=20.0,
                       motion='kinematic', traffic=0, log_samples=False, log_actions=True, camera=None,
                       max_episode_s=15.0, reward=None, realism=None, ruleset=None, timeout=20.0):
    """(env, ego policy or None) for an ego spec (ego_spec)."""
    from .policy import load_policy
    driver, pdir = ego_spec(ego)
    pol = load_policy(pdir) if pdir else None
    if camera is not None or not getattr(pol, 'uses_camera', False):
        camera = {'camera': camera}
    else:
        camera = {'camera': pol.doc.get('camera')}       # an end-to-end ego reads its camera (core.policy)
    cfg = adversary_config(project, world, out_dir, svc.port, svc.tm_port, ego=driver, action=action, rig_id=rig,
                           seed=seed, fps=fps, motion=motion, traffic_vehicles=traffic, log_samples=log_samples,
                           log_actions=log_actions, max_episode_s=max_episode_s, reward=reward,
                           realism=realism, ruleset=ruleset, workspace=project.workspace, timeout=timeout,
                           ego_action=pol.action['kind'] if pol else 'continuous',
                           bev=(pol.doc.get('bev') if pol is not None and pol.uses_bev else None), **camera)
    return AdversaryEnv(cfg, svc, ego_policy=pol), pol


# ============================================================================================ the curriculum
class AdvCurriculum:
    """Stages of rosters (module doc): {'window', 'threshold', 'max_episodes', 'stages': [{'name', 'rosters':
    [[role, …], …], 'options': {reset options}}]}."""

    def __init__(self, doc: dict):
        from .rl import Curriculum
        if not doc.get('stages'):
            raise ValueError('a curriculum needs at least one stage')
        for st in doc['stages']:
            if not st.get('rosters'):
                raise ValueError(f'stage {st.get("name")!r} has no rosters')
            for ro in st['rosters']:
                bad = [r for r in ro if r not in AF.ROLES]
                if bad:
                    raise ValueError(f'stage {st.get("name")!r}: unknown roles {bad}')
        self.doc = doc
        self.max_episodes = int(doc.get('max_episodes') or 0)
        self.c = Curriculum({'window': doc.get('window', 20), 'threshold': doc.get('threshold', 0.3),
                             'stages': [{'name': s.get('name')} for s in doc['stages']]})

    @classmethod
    def single(cls, roster):
        return cls({'window': 1, 'threshold': 1.0, 'stages': [{'name': ', '.join(roster), 'rosters': [roster]}]})

    @property
    def stage(self) -> int:
        return self.c.stage

    @property
    def current(self) -> dict:
        return self.doc['stages'][self.c.stage]

    def options(self, rng) -> dict:
        st = self.current
        o = dict(st.get('options') or {})
        o['roles'] = list(st['rosters'][int(rng.integers(len(st['rosters'])))])
        return o

    def record(self, success: bool, episode=None) -> bool:
        if self.c.record(success, episode):
            self.c.promotions[-1]['by'] = 'success rate'
            return True
        n = len(self.c.results[self.c.stage])
        if self.max_episodes and n >= self.max_episodes and self.c.stage + 1 < len(self.doc['stages']):
            self.c.promotions.append({'from': self.c.stage, 'to': self.c.stage + 1, 'episode': episode,
                                      'after_episodes': n, 'success_rate': self.c.rate(), 'by': 'max_episodes'})
            self.c.stage += 1
            return True
        return False

    def rate(self):
        return self.c.rate()

    @property
    def promotions(self):
        return self.c.promotions

    def state(self) -> dict:
        return self.c.state()

    def load(self, d: dict):
        self.c.load(d)
        return self


def load_adv_curriculum(spec) -> AdvCurriculum:
    if isinstance(spec, AdvCurriculum):
        return spec
    if spec in (None, '', 'default'):
        return AdvCurriculum(DEFAULT_ADV_CURRICULUM)
    if isinstance(spec, dict):
        return AdvCurriculum(spec)
    with open(spec, encoding='utf-8') as fh:
        return AdvCurriculum(json.load(fh))


# ============================================================================================ the policy folder
def save_adversary_policy(out_dir: str, net, doc: dict, metrics: dict | None = None) -> str:
    """An adversary policy folder: policy.json (falcon.adversary_policy/1: the features by name, the normaliser, the
    action, the network, the training and its provenance), weights.pt with its SHA-256, metrics.json."""
    import torch
    from .provenance import sha256_file, write_json
    os.makedirs(out_dir, exist_ok=True)
    wp = os.path.join(out_dir, 'weights.pt')
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, wp)
    write_json(os.path.join(out_dir, 'policy.json'), dict(doc, schema=ADV_POLICY_SCHEMA, weights='weights.pt',
                                                         weights_sha256=sha256_file(wp), advfeatures=AF.SCHEMA))
    if metrics is not None:
        write_json(os.path.join(out_dir, 'metrics.json'), metrics)
    return out_dir


class AdversaryPolicy:
    """A loaded adversary policy folder; act(observation vector) → action, deterministic (the mean clipped to
    [−1, 1], the argmax)."""

    def __init__(self, path: str):
        import torch
        from .policy import build_net, feature_index
        from .provenance import read_json, sha256_file
        self.path = os.path.abspath(path)
        self.doc = read_json(os.path.join(self.path, 'policy.json'))
        if self.doc.get('schema') != ADV_POLICY_SCHEMA:
            raise ValueError(f'{path}: not an adversary policy folder (schema {self.doc.get("schema")!r})')
        wp = os.path.join(self.path, self.doc.get('weights', 'weights.pt'))
        if self.doc.get('weights_sha256') and sha256_file(wp) != self.doc['weights_sha256']:
            raise ValueError(f'{wp}: its SHA-256 is not the one policy.json records')
        self.idx = feature_index(self.doc['features'], AF.ADV_FEATURE_NAMES)
        self.norm = self.doc['normaliser']
        self.action = self.doc['action']
        n_out = int(self.action['n']) if self.action['kind'] == 'discrete' else 2
        self.net = build_net(self.doc['net'], len(self.idx), n_out)
        self.net.load_state_dict(torch.load(wp, map_location='cpu', weights_only=True))
        self.net.eval()

    def act_batch(self, X) -> np.ndarray:
        import torch
        from .policy import normalise
        x = normalise(np.asarray(X, np.float64).reshape(-1, AF.N_ADV_FEATURES)[:, self.idx], self.norm)
        with torch.no_grad():
            y = self.net(torch.as_tensor(x)).numpy()
        return y.argmax(1) if self.action['kind'] == 'discrete' else np.clip(y, -1.0, 1.0).astype(np.float32)

    def act(self, x):
        y = self.act_batch(np.asarray(x)[None])
        return int(y[0]) if self.action['kind'] == 'discrete' else y[0]

    def rosters(self) -> list:
        """The rosters of the last stage it trained on."""
        cur = (self.doc.get('training') or {}).get('curriculum') or DEFAULT_ADV_CURRICULUM
        st = cur['stages'][min(int((self.doc.get('training') or {}).get('stage', len(cur['stages']) - 1)),
                               len(cur['stages']) - 1)]
        return [list(r) for r in st['rosters']]


def load_adversary_policy(path: str) -> AdversaryPolicy:
    return AdversaryPolicy(path)


# ============================================================================================ training
def _jsonl(path, row):
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, separators=(',', ':')) + '\n')


def train_adversaries(env, out_dir: str, config: dict | None = None, curriculum=None, resume: bool = False,
                      log=None) -> dict:
    """Train the adversaries with parameter-shared independent PPO (module doc) in an AdversaryEnv and write
    everything to out_dir: episodes.jsonl, updates.jsonl, checkpoints/ (resume=True continues from latest.json),
    policy/ (an adversary policy folder, at every checkpoint and at the end), run.json. Returns the summary."""
    import torch
    from . import rl
    from .policy import RunningMeanStd
    from .provenance import block, new_run, sha256_json, utc_now, write_json
    cfg = dict(ADV_PPO_DEFAULTS, **(config or {}))
    log = log or (lambda s: None)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    cur = load_adv_curriculum(curriculum)
    discrete = env.adv_action_kind == 'discrete'
    n_in = AF.N_ADV_FEATURES
    n_act = int(env.adv_action_space.n) if discrete else 2
    dev = cfg['device']
    torch.manual_seed(int(cfg['seed']))
    actor, critic, log_std, spec_a = rl._agent(n_in, n_act, discrete, dict(cfg, cnn=False))
    actor.to(dev)
    critic.to(dev)
    params = list(actor.parameters()) + list(critic.parameters()) + ([log_std] if log_std is not None else [])
    opt = torch.optim.Adam(params, lr=float(cfg['lr']), eps=1e-5)
    obs_rms, ret_rms = RunningMeanStd((n_in,)), RunningMeanStd(())
    rng = np.random.default_rng(int(cfg['seed']))
    update, global_step, episode = 0, 0, 0
    novelty: dict = {}
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
        novelty = dict(ck['novelty'])
        update, global_step, episode = ck['update'], ck['global_step'], ck['episode']
        log(f'resumed at update {update} (step {global_step}, episode {episode}, stage {cur.stage})')
    env_cfg = getattr(env, 'cfg', {})
    settings = {'ppo': cfg, 'curriculum': cur.doc, 'algorithm': 'IPPO with parameter sharing',
                'environment_sha256': sha256_json(env_cfg) if env_cfg else None,
                'ego': (env_cfg.get('adversary') or {}).get('ego'),
                'ego_policy': getattr(env.ego_policy, 'path', None)}
    ins = {'environment': os.path.join(env.run_dir, 'gym.json')}
    run = new_run('adversary_train', [out_dir], int(cfg['seed']), settings, ins)
    run['name'] = cfg['name'] or 'adversarial traffic'
    if resume:
        run['resumed_at_update'] = update
    write_json(os.path.join(out_dir, 'run.json'), run)
    ep_path, up_path = os.path.join(out_dir, 'episodes.jsonl'), os.path.join(out_dir, 'updates.jsonl')
    recent: list = []
    wdiv = float(dict(AF.DEFAULT_ADV_REWARD, **((env_cfg.get('adversary') or {}).get('reward') or {}))['diversity'])

    def norm(x, upd):
        x = np.asarray(x, np.float64)
        if cfg['normalize_obs']:
            if upd:
                obs_rms.update(x[None])
            x = np.clip((x - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -cfg['obs_clip'], cfg['obs_clip'])
        return x.astype(np.float32)

    def tens(x):
        return torch.as_tensor(np.asarray(x), device=dev)

    def value(x):
        with torch.no_grad():
            return float(critic(tens(x[None]), None)[0, 0])

    state = {'seed': 0, 'opts': {}}

    def new_episode():
        for _try in range(RESET_TRIES):          # a seed whose roster the map cannot place: logged, another drawn
            state['seed'] = int(rng.integers(0, 2 ** 31 - 1))
            state['opts'] = cur.options(rng)
            try:
                o, _ = env.reset(seed=state['seed'], options=state['opts'])
                return o
            except EnvError as e:
                if 'no start on this map' in str(e) or _try == RESET_TRIES - 1:
                    raise
                _jsonl(ep_path, {'skipped': {'seed': state['seed'], 'options': state['opts'], 'error': str(e)}})
        raise AssertionError('unreachable')

    def export(final=False):
        doc = {'name': cfg['name'] or 'adversarial traffic', 'trained_by': 'ippo',
               'features': list(AF.ADV_FEATURE_NAMES),
               'normaliser': {'mean': obs_rms.mean.tolist() if cfg['normalize_obs'] else [0.0] * n_in,
                              'std': np.sqrt(obs_rms.var + 1e-8).tolist() if cfg['normalize_obs'] else [1.0] * n_in,
                              'clip': float(cfg['obs_clip']) if cfg['normalize_obs'] else 1e9},
               'action': ({'kind': 'discrete', 'n': n_act, 'table': [list(m) for m in AF.MANOEUVRES]}
                          if discrete else {'kind': env.adv_action_kind, 'keys': ['lateral', 'accel']
                                            if env.adv_action_kind == 'lateral' else ['steer', 'accel']}),
               'net': spec_a, 'provenance': block(ins, int(cfg['seed']), settings),
               'training': {'updates': update, 'global_step': global_step, 'episodes': episode, 'stage': cur.stage,
                            'final': final, 'curriculum': cur.doc, 'ego': settings['ego'],
                            'ego_policy': settings['ego_policy'], 'reward': (env_cfg.get('adversary') or {}).get(
                                'reward'), 'distinct_failures': len(novelty)}}
        save_adversary_policy(os.path.join(out_dir, 'policy'), actor, doc,
                              {'recent_episodes': recent[-20:], 'curriculum': cur.state(), 'novelty': novelty})

    def checkpoint():
        name = f'ckpt-{update:06d}.pt'
        torch.save({'update': update, 'global_step': global_step, 'episode': episode, 'actor': actor.state_dict(),
                    'critic': critic.state_dict(), 'log_std': None if log_std is None else log_std.detach(),
                    'optimizer': opt.state_dict(), 'obs_rms': obs_rms.state(), 'ret_rms': ret_rms.state(),
                    'rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'curriculum': cur.state(),
                    'novelty': novelty, 'config': cfg}, os.path.join(ck_dir, name))
        write_json(latest, {'file': name, 'update': update, 'global_step': global_step})
        for f in sorted(f for f in os.listdir(ck_dir) if f.startswith('ckpt-'))[:-int(cfg['keep_checkpoints'])]:
            os.remove(os.path.join(ck_dir, f))
        export()

    obs = new_episode()
    ret_run: dict = {}
    ep_ret: dict = {}
    ep_len = 0
    t_start = time.monotonic()
    env_wall = 0.0
    n = int(cfg['n_steps'])
    gamma = float(cfg['gamma'])
    while update < int(cfg['total_updates']):
        streams: dict = {}                       # slot → {'X', 'A', 'LP', 'V', 'R', 'D'}; slot = (episode, agent)
        order: list = []
        t0 = time.monotonic()
        env_t = 0.0
        for _t in range(n):
            ids = list(env.agents)
            acts = {}
            for aid in ids:
                key = (episode, aid)
                if key not in streams:
                    streams[key] = {'X': [], 'A': [], 'LP': [], 'V': [], 'R': [], 'D': []}
                    order.append(key)
                x = norm(obs[aid], True)
                with torch.no_grad():
                    d = rl._dist(actor, log_std, tens(x[None]), None, discrete)
                    a = d.sample()
                    lp, _ = rl._logp_ent(d, a, discrete)
                    v = critic(tens(x[None]), None)[0, 0]
                s = streams[key]
                s['X'].append(x)
                s['A'].append(int(a[0]) if discrete else a[0].cpu().numpy())
                s['LP'].append(float(lp[0]))
                s['V'].append(float(v))
                acts[aid] = int(a[0]) if discrete else np.clip(a[0].cpu().numpy(), -1.0, 1.0)
            te = time.monotonic()
            obs, rews, terms, truncs, _infos = env.step(acts)
            env_t += time.monotonic() - te
            global_step += 1
            ep_len += 1
            done = bool(ids) and (terms[ids[0]] or truncs[ids[0]]) or (not ids and env._needs_reset)
            info = env.last_info
            summ = info.get('episode_summary') if done else None
            bonus = {}
            if summ:
                for f in (summ.get('adversary') or {}).get('failures') or ():
                    if f.get('failure') and f.get('realistic'):
                        dsc = AF.descriptor(f)
                        novelty[dsc] = novelty.get(dsc, 0) + 1
                        to = [f['actor']] if f.get('actor') in ids else ids
                        for aid in to:
                            bonus[aid] = bonus.get(aid, 0.0) + wdiv / math.sqrt(novelty[dsc]) / max(len(to), 1)
            for aid in ids:
                r = rews[aid] + bonus.get(aid, 0.0)
                ep_ret[aid] = ep_ret.get(aid, 0.0) + r
                if cfg['normalize_reward']:
                    ret_run[aid] = ret_run.get(aid, 0.0) * gamma + r
                    ret_rms.update(np.array([ret_run[aid]]))
                    r = float(np.clip(r / math.sqrt(ret_rms.var + 1e-8), -cfg['reward_clip'], cfg['reward_clip']))
                if truncs[aid] and not terms[aid]:
                    r += gamma * value(norm(obs[aid], False)) if aid in obs else 0.0
                s = streams[(episode, aid)]
                s['R'].append(r)
                s['D'].append(float(done))
            if done:
                summ = summ or {}
                adv = summ.get('adversary') or {}
                ok = bool(adv.get('realistic_failure'))
                stage = cur.stage
                promoted = cur.record(ok, episode)
                row = {'episode': episode, 'update': update + 1, 'global_step': global_step, 'stage': stage,
                       'stage_name': cur.doc['stages'][stage].get('name'), 'seed': state['seed'],
                       'options': state['opts'], 'length': ep_len, 'termination': summ.get('termination'),
                       'failed': bool(adv.get('failed')), 'realistic': bool(adv.get('realistic')),
                       'realistic_failure': ok, 'returns': {k: round(v, 6) for k, v in ep_ret.items()},
                       'mean_return': float(np.mean(list(ep_ret.values()))) if ep_ret else 0.0,
                       'failures': [{k: f.get(k) for k in ('kind', 'actor', 'fault', 'rule', 'step', 'role')}
                                    for f in adv.get('failures') or ()],
                       'realism': {aid: v.get('realism') for aid, v in (adv.get('actors') or {}).items()},
                       'diversity_bonus': bonus, 'env_episode': summ.get('episode')}
                _jsonl(ep_path, row)
                recent.append({k: row[k] for k in ('episode', 'mean_return', 'length', 'termination', 'failed',
                                                   'realistic_failure', 'stage')})
                recent[:] = recent[-100:]
                if promoted:
                    log(f'curriculum: stage {stage} → {cur.stage} ({cur.current.get("name")}) after episode {episode}')
                    _jsonl(ep_path, {'promotion': cur.promotions[-1]})
                episode += 1
                ret_run, ep_ret, ep_len = {}, {}, 0
                obs = new_episode()
        env_wall += env_t
        # ---- advantages per stream, then one update over all of them
        X, A, LP, ADV, RET, V = [], [], [], [], [], []
        for key in order:
            s = streams[key]
            if not s['R']:
                continue
            last_v = 0.0
            if not s['D'][-1] and key[0] == episode and key[1] in obs:
                last_v = value(norm(obs[key[1]], False))
            Vs = np.asarray(s['V'], np.float32)
            adv_ = rl.gae(np.asarray(s['R'], np.float32), Vs, np.asarray(s['D'], np.float32), last_v, gamma,
                          float(cfg['gae_lambda']))
            X += s['X'][:len(s['R'])]
            A += s['A'][:len(s['R'])]
            LP += s['LP'][:len(s['R'])]
            V.append(Vs[:len(s['R'])])
            ADV.append(adv_)
            RET.append(adv_ + Vs[:len(s['R'])])
        if not X:
            raise ValueError('the rollout had no adversary steps: every roster of the stage is empty')
        Xa = np.asarray(X, np.float32)
        Aa = np.asarray(A, np.int64) if discrete else np.asarray(A, np.float32)
        ADVa, RETa, Va = np.concatenate(ADV), np.concatenate(RET), np.concatenate(V)
        stats, stop = rl.ppo_update(actor, critic, log_std, opt, params, Xa, Aa, np.asarray(LP, np.float32), ADVa,
                                    RETa, cfg, rng, discrete, dev)
        update += 1
        var_r = float(np.var(RETa))
        row = {'update': update, 'global_step': global_step, 'episodes': episode, 'stage': cur.stage,
               'transitions': int(len(Xa)), **{k: float(np.mean(v)) if v else None for k, v in stats.items()},
               'explained_variance': float(1.0 - np.var(RETa - Va) / var_r) if var_r > 1e-12 else None,
               'early_stop': stop, 'env_steps_per_s': round(n / max(env_t, 1e-9), 2),
               'update_wall_s': round(time.monotonic() - t0, 3),
               'mean_return_last20': float(np.mean([r_['mean_return'] for r_ in recent[-20:]])) if recent else None,
               'realistic_failure_rate_last20': float(np.mean([r_['realistic_failure'] for r_ in recent[-20:]]))
               if recent else None, 'distinct_failures': len(novelty), 'stage_success_rate': cur.rate()}
        if log_std is not None:
            row['action_std'] = [round(float(s), 4) for s in log_std.detach().exp().cpu()]
        _jsonl(up_path, row)
        log(f'update {update}/{cfg["total_updates"]}: step {global_step}, episodes {episode}, stage {cur.stage}, '
            f'realistic failures (last 20) {row["realistic_failure_rate_last20"]}, distinct {len(novelty)}, '
            f'{row["env_steps_per_s"]} env steps/s')
        if update % int(cfg['checkpoint_every']) == 0 or update == int(cfg['total_updates']):
            checkpoint()
    export(final=True)
    summary = {'updates': update, 'global_step': global_step, 'episodes': episode, 'stage': cur.stage,
               'promotions': cur.promotions, 'distinct_failures': len(novelty),
               'realistic_failure_rate_last20': float(np.mean([r_['realistic_failure'] for r_ in recent[-20:]]))
               if recent else None,
               'mean_return_last20': float(np.mean([r_['mean_return'] for r_ in recent[-20:]])) if recent else None}
    run.update(status='ok', exit_code=0, finished=utc_now(), summary=summary,
               performance={'wall_s': round(time.monotonic() - t_start, 3), 'env_wall_s': round(env_wall, 3),
                            'source': 'measured'})
    write_json(os.path.join(out_dir, 'run.json'), run)
    return summary


# ============================================================================================ episodes
def idle_action(kind: str):
    """Hold the lane and the speed: lateral [0, 0], discrete the (0, hold) manoeuvre, continuous [0, 0] (coasting
    without steering: not lane keeping)."""
    return AF.MANOEUVRES.index((0.0, 'hold')) if kind == 'discrete' else np.zeros(2, np.float32)


def run_episode(env, chooser, seed: int, options: dict) -> dict:
    """One episode: chooser(agent, observation) → action. Returns the worker's full episode line (with the actions)
    from its episodes.jsonl."""
    obs, info = env.reset(seed=int(seed), options=options)
    while True:
        acts = {aid: chooser(aid, obs[aid]) for aid in env.agents}
        obs, _r, terms, truncs, _i = env.step(acts)
        if env._needs_reset:
            break
    summ = env.last_info.get('episode_summary') or {}
    return full_line(env, summ)


def full_line(env, summ: dict) -> dict:
    """The episode's full line (actions, samples) from the worker's episodes.jsonl, matched by episode and seed."""
    p = env.episodes_path
    try:
        with open(p, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - (1 << 24)))
            lines = fh.read().decode('utf-8', 'replace').splitlines()
        for line in reversed(lines):
            if not line.strip().startswith('{'):
                continue
            row = json.loads(line)
            if row.get('episode') == summ.get('episode') and row.get('seed') == summ.get('seed'):
                return row
    except (OSError, ValueError):
        pass
    return dict(summ)


def failure_score(e: dict) -> float:
    """The scripted searches' objective: 1 for a realistic ego failure, else 0.9 · exp(−min TTC / 1.5 s) over the
    tracked road users (core.criticality's TTC scale)."""
    adv = e.get('adversary') or {}
    if adv.get('realistic_failure'):
        return 1.0
    ttcs = [v.get('min_ttc_s') for v in (adv.get('actors') or {}).values() if v.get('min_ttc_s') is not None]
    return 0.9 * (AF.ttc_score(min(ttcs)) if ttcs else 0.0)


# ============================================================================================ metrics
def ks_statistic(a, b) -> float | None:
    """The two-sample Kolmogorov–Smirnov statistic sup |F_a − F_b|; None when a sample is empty."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return None
    v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, 'right') / len(a) - np.searchsorted(b, v, 'right') / len(b))))


def wasserstein1(a, b) -> float | None:
    """The Wasserstein-1 distance between two empirical distributions: ∫ |F_a − F_b| dx; None when one is empty."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return None
    v = np.sort(np.concatenate([a, b]))
    d = np.diff(v)
    fa = np.searchsorted(a, v[:-1], 'right') / len(a)
    fb = np.searchsorted(b, v[:-1], 'right') / len(b)
    return float(np.sum(np.abs(fa - fb) * d))


def distribution(x) -> dict:
    x = np.abs(np.asarray(x, float))
    if not len(x):
        return {'n': 0}
    q = np.percentile(x, [50, 90, 95, 99])
    return {'n': int(len(x)), 'p50': float(q[0]), 'p90': float(q[1]), 'p95': float(q[2]), 'p99': float(q[3]),
            'max': float(x.max()), 'mean': float(x.mean())}


def realism_metrics(episodes: list, tm: dict | None = None, limits: dict | None = None) -> dict:
    """The realism of the tracked road users over a set of episodes (module doc)."""
    lim = dict(AF.REALISM_LIMITS, **(limits or {}))
    km = 0.0
    n_act = 0
    rules_all: dict = {}
    rules_charged = 0
    km_rules = 0.0
    totals = {'fault_collisions': 0, 'shared_collisions': 0, 'contacts': 0, 'off_road_s': 0.0, 'wrong_way_s': 0.0,
              'wrong_way_over_s': 0.0, 'dynamics_over_s': 0.0, 'unrealistic_actors': 0}
    samples = {'a_long': [], 'a_lat': [], 'jerk': []}
    for e in episodes:
        adv = e.get('adversary') or {}
        for aid, a in (adv.get('actors') or {}).items():
            r = a.get('realism') or {}
            n_act += 1
            km += float(r.get('distance_m', 0.0)) / 1000.0
            for k in totals:
                if k == 'unrealistic_actors':
                    totals[k] += 0 if r.get('realistic', True) else 1
                else:
                    totals[k] += r.get(k, 0)
            ru = a.get('rules')
            if ru:
                km_rules += float(ru.get('distance_km') or 0.0)
                rules_charged += int(ru.get('charged', 0))
                for rid, c in (ru.get('counts') or {}).items():
                    rules_all[rid] = rules_all.get(rid, 0) + int(c)
        for aid, s in (e.get('adversary_samples') or {}).items():
            for k in samples:
                samples[k] += list(s.get(k) or ())
    n_ep = len(episodes)
    out = {'episodes': n_ep, 'adversaries': n_act, 'distance_km': km,
           'fault_collisions_per_episode': totals['fault_collisions'] / n_ep if n_ep else None,
           'shared_collisions_per_episode': totals['shared_collisions'] / n_ep if n_ep else None,
           'contacts_per_episode': totals['contacts'] / n_ep if n_ep else None,
           'off_road_s_per_km': totals['off_road_s'] / km if km > 0 else None,
           'wrong_way_s_per_km': totals['wrong_way_s'] / km if km > 0 else None,
           'wrong_way_over_budget_s_per_km': totals['wrong_way_over_s'] / km if km > 0 else None,
           'dynamics_over_s_per_km': totals['dynamics_over_s'] / km if km > 0 else None,
           'unrealistic_share': totals['unrealistic_actors'] / n_act if n_act else None,
           'rule_violations_per_km': {k: v / km_rules for k, v in sorted(rules_all.items())} if km_rules > 0 else {},
           'rule_violations_total_per_km': sum(rules_all.values()) / km_rules if km_rules > 0 else None,
           'charged_rule_violations_per_km': rules_charged / km_rules if km_rules > 0 else None,
           'dynamics': {k: distribution(v) for k, v in samples.items()},
           'beyond_limits_share': {k: (float(np.mean(np.abs(samples[k]) > lim[lk])) if samples[k] else None)
                                   for k, lk in (('a_long', 'a_long_mps2'), ('a_lat', 'a_lat_mps2'),
                                                 ('jerk', 'jerk_mps3'))}}
    if tm is not None:
        out['vs_traffic_manager'] = {k: {'ks': ks_statistic(np.abs(samples[k]), np.abs(tm.get(k) or [])),
                                         'wasserstein1': wasserstein1(np.abs(samples[k]), np.abs(tm.get(k) or []))}
                                     for k in samples}
    return out


def summarise_method(episodes: list, tm=None, limits=None) -> dict:
    from .stack import wilson
    n = len(episodes)
    fails = [e for e in episodes if (e.get('adversary') or {}).get('realistic_failure')]
    raw = [e for e in episodes if (e.get('adversary') or {}).get('failed')]
    kinds: dict = {}
    desc = set()
    ego_caused = 0
    for e in fails:
        for f in (e.get('adversary') or {}).get('failures') or ():
            if f.get('failure'):
                kinds[f['kind']] = kinds.get(f['kind'], 0) + 1
                desc.add(AF.descriptor(f))
                ego_caused += f['kind'] == 'collision'
    lo, hi = wilson(len(fails), n)
    lo2, hi2 = wilson(len(raw), n)
    ttc = [min([v['min_ttc_s'] for v in (e.get('adversary') or {}).get('actors', {}).values()
                if v.get('min_ttc_s') is not None] or [math.inf]) for e in episodes]
    fin = [t for t in ttc if math.isfinite(t)]
    return {'episodes': n, 'realistic_failures': len(fails), 'failure_rate': len(fails) / n if n else None,
            'failure_rate_ci95': [lo, hi] if n else None, 'any_failures': len(raw),
            'any_failure_rate': len(raw) / n if n else None, 'any_failure_rate_ci95': [lo2, hi2] if n else None,
            'distinct_failures': len(desc), 'failure_kinds': kinds, 'ego_caused_collisions': ego_caused,
            'median_min_ttc_s': float(np.median(fin)) if fin else None,
            'realism': realism_metrics(episodes, tm, limits)}


# ============================================================================================ evaluation
def evaluate(project, svc, world: str, policy_dir: str, egos: list, out_dir: str, episodes: int = 20,
             methods=METHODS, seed: int = 0, tm_reference: int = 0, rig=None, fps: float = 20.0,
             families=SCRIPTED_FAMILIES, rosters=None, progress=None, timeout: float = 20.0) -> dict:
    """Evaluate the learned adversaries against each ego beside the baselines (module doc); writes
    adversary-eval.json, episodes.jsonl (every episode, replayable), failures.jsonl and cases.jsonl in out_dir."""
    from . import scenarios as SC
    from . import search as S
    from .provenance import block, write_json
    os.makedirs(out_dir, exist_ok=True)
    pol = load_adversary_policy(policy_dir) if 'learned' in methods or policy_dir else None
    kind = pol.action['kind'] if pol is not None else 'lateral'
    rosters = rosters or (pol.rosters() if pol is not None else DEFAULT_ADV_CURRICULUM['stages'][-1]['rosters'])
    progress = progress or (lambda f, t: None)
    ep_path = os.path.join(out_dir, 'episodes.jsonl')
    open(ep_path, 'w').close()
    n_line = [0]
    failed_runs = []
    tm = None
    if tm_reference:
        env = None
        try:
            env, _ = open_adversary_env(project, world, os.path.join(out_dir, 'env', 'tm'), svc, 'follower',
                                        action=kind, rig=rig, seed=seed, fps=fps, traffic=int(tm_reference),
                                        log_samples=True, log_actions=False, timeout=timeout)
            tm = {'a_long': [], 'a_lat': [], 'jerk': []}
            for k in range(max(1, episodes // 10)):
                e = run_episode(env, lambda aid, o: idle_action(kind), seed + 500000 + k, {'roles': []})
                for key in tm:
                    tm[key] += list((e.get('tm_samples') or {}).get(key) or ())
        except Exception as ex:          # recorded, not hidden
            failed_runs.append({'what': 'traffic-manager reference', 'error': f'{type(ex).__name__}: {ex}'})
            tm = None
        finally:
            if env is not None:
                env.close()
    results = {}
    all_fail = []
    total = len(egos) * len(methods) * episodes
    done = [0]

    def record(e, ego, method, extra):
        e = dict(e, eval={'ego': ego, 'method': method, 'action': kind, **extra})
        with open(ep_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(e, separators=(',', ':')) + '\n')
        line = n_line[0]
        n_line[0] += 1
        done[0] += 1
        adv = e.get('adversary') or {}
        for j, f in enumerate(adv.get('failures') or ()):
            if f.get('failure'):
                all_fail.append({'ego': ego, 'method': method, 'line': line, 'failure': j, 'seed': e.get('seed'),
                                 'options': e.get('options'), 'kind': f['kind'], 'fault': f.get('fault'),
                                 'actor': f.get('actor'), 'role': f.get('role'), 'step': f.get('step'),
                                 'sim_t': f.get('sim_t'), 'realistic': bool(f.get('realistic')),
                                 'descriptor': AF.descriptor(f), 'replayable': replayable(e) is None})
        progress(done[0] / max(total, 1), f'{ego} · {method}: {done[0]}/{total}')
        return e
    for ego in egos:
        env = None
        res = results[ego] = {}
        try:
            env, _ego_pol = open_adversary_env(project, world, os.path.join(out_dir, 'env', _slug(ego)), svc, ego,
                                               action=kind, rig=rig, seed=seed, fps=fps, log_samples=True,
                                               timeout=timeout)
            env_info = {'fps': fps, 'max_episode_s': env.cfg.get('max_episode_s'), 'motion': env.cfg.get('motion'),
                        'traffic_vehicles': 0, 'world': world, 'rig': rig}
            for method in methods:
                eps = []
                if method in ('learned', 'random', 'idle'):
                    rr = np.random.default_rng([int(seed), METHODS.index(method)])
                    for k in range(episodes):
                        roster = rosters[k % len(rosters)]
                        if method == 'learned':
                            def ch(aid, o):
                                return pol.act(o)
                        elif method == 'random':
                            def ch(aid, o):
                                return int(rr.integers(AF.n_discrete())) if kind == 'discrete' else \
                                    rr.uniform(-1.0, 1.0, 2).astype(np.float32)
                        else:
                            def ch(aid, o):
                                return idle_action(kind)
                        try:
                            e = run_episode(env, ch, seed + k, {'roles': roster})    # the same seeds for each
                        except EnvError as ex:
                            failed_runs.append({'ego': ego, 'method': method, 'episode': k, 'error': str(ex)})
                            continue
                        eps.append(record(e, ego, method, env_info))
                else:
                    fams = [f for f in families if f in SC.LIBRARY]
                    share = [episodes // len(fams) + (1 if i < episodes % len(fams) else 0) for i in range(len(fams))]
                    for i, fam in enumerate(fams):
                        if not share[i]:
                            continue
                        space = SC.search_space(SC.builtin(fam))
                        if method == 'scripted_random':
                            srch = S.RandomSearch(space, seed=seed + 101 * (i + 1), sequence='halton')
                        else:
                            srch = S.make('adaptive', space, seed=seed + 211 * (i + 1), threshold=1.0,
                                          population=4, n_init=max(4, min(8, share[i] // 2)))
                        for k in range(share[i]):
                            params = srch.ask(1)[0]
                            try:
                                e = run_episode(env, lambda aid, o: idle_action(kind), seed + 10000 * (i + 1) + k,
                                                {'scenario': fam, 'values': params})
                            except EnvError as ex:
                                srch.tell([params], [None])
                                failed_runs.append({'ego': ego, 'method': method, 'family': fam, 'error': str(ex)})
                                if 'no start on this map' in str(ex):
                                    break
                                continue
                            srch.tell([params], [failure_score(e)])
                            eps.append(record(e, ego, method, dict(env_info, family=fam)))
                res[method] = summarise_method(eps, tm)
        except Exception as ex:           # an ego the town cannot host, a lost server: recorded, not hidden
            failed_runs.append({'ego': ego, 'error': f'{type(ex).__name__}: {ex}'})
        finally:
            if env is not None:
                env.close()
    with open(os.path.join(out_dir, 'failures.jsonl'), 'w', encoding='utf-8') as fh:
        for f in all_fail:
            fh.write(json.dumps(f, separators=(',', ':')) + '\n')
    write_cases(out_dir, all_fail)
    doc = {'schema': EVAL_SCHEMA, 'egos': list(egos), 'methods': list(methods), 'episodes_per_method': episodes,
           'rosters': rosters, 'families': list(families), 'seed': seed, 'policy': pol.path if pol else None,
           'action': kind, 'results': results, 'traffic_manager_reference': (
               {k: distribution(v) for k, v in tm.items()} if tm is not None else None),
           'failed': failed_runs, 'failures': len(all_fail),
           'realistic_failures': sum(1 for f in all_fail if f['realistic']),
           'method_note': ('failure rate: share of episodes with a realistic ego failure, Wilson 95 %; the scripted '
                           'searches\' episodes are adaptive, not independent draws: their interval is descriptive'),
           'limits': dict(AF.REALISM_LIMITS)}
    ins = {'policy': os.path.join(pol.path, 'policy.json')} if pol else {}
    doc['provenance'] = block(ins, seed=seed, settings={'egos': list(egos), 'methods': list(methods),
                                                         'episodes': episodes, 'tm_reference': tm_reference,
                                                         'world': world, 'fps': fps})
    write_json(os.path.join(out_dir, 'adversary-eval.json'), doc)
    return doc


def _slug(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '-' for c in os.path.basename(s.rstrip('/')) or s)[:40]


def write_cases(out_dir: str, failures: list):
    """cases.jsonl: each realistic failure as a case for the Rare cases page (falcon.adversary_case/1)."""
    with open(os.path.join(out_dir, 'cases.jsonl'), 'w', encoding='utf-8') as fh:
        for f in failures:
            if not f.get('realistic'):
                continue
            fh.write(json.dumps({'schema': CASES_SCHEMA, 'source': 'adversary',
                                 'episodes': os.path.join(os.path.abspath(out_dir), 'episodes.jsonl'), **f},
                                separators=(',', ':')) + '\n')


def list_cases(workspace: str) -> list:
    """Every adversarial case under the workspace's adversary/ folder (each evaluation's cases.jsonl), newest
    evaluation first."""
    root = os.path.join(workspace, 'adversary')
    out = []
    if not os.path.isdir(root):
        return out
    dirs = []
    for dp, dn, fn in os.walk(root):
        if 'cases.jsonl' in fn and 'adversary-eval.json' in fn:
            dirs.append(dp)
        if dp.count(os.sep) - root.count(os.sep) >= 2:
            dn[:] = []
    for d in sorted(dirs, key=lambda p: -os.path.getmtime(os.path.join(p, 'cases.jsonl'))):
        with open(os.path.join(d, 'cases.jsonl'), encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    out.append(dict(json.loads(line), eval_dir=d))
    return out


# ============================================================================================ replay
def replayable(e: dict) -> str | None:
    """None when the episode replays exactly, else why not."""
    if e.get('motion', 'kinematic') != 'kinematic':
        return 'physics motion is not repeatable in CARLA 0.9.16 (measured); only kinematic episodes replay exactly'
    ev = e.get('eval') or {}
    if ev.get('traffic_vehicles'):
        return 'background traffic makes an episode non-repeatable'
    if (e.get('adversary') or {}).get('ego_driver') == 'autopilot':
        return 'the autopilot ego runs on CARLA\'s physics and traffic manager: not repeatable'
    if 'adversary_actions' not in e and (e.get('adversary') or {}).get('order'):
        return 'the episode was logged without its actions (log_actions off)'
    return None


def replay(e: dict, env, failure: int | None = None, before_s: float = 3.0, after_s: float = 1.0,
           out_dir: str = '') -> dict:
    """Re-run an episode of an evaluation (module doc) in env (an AdversaryEnv with the episode's ego driver and
    action kind, a chase camera): {'frames', 'steps', 'match'} — match: the same failure at the same step."""
    import cv2
    why = replayable(e)
    if why:
        raise ValueError(why)
    fails = (e.get('adversary') or {}).get('failures') or []
    if failure is not None and not 0 <= failure < len(fails):
        raise ValueError(f'the episode has {len(fails)} failures')
    fps = float(env.ready.get('fps') or 20.0)
    target = fails[failure] if failure is not None else None
    lo = max(0, int(target['step'] - before_s * fps)) if target else 0
    hi = int(target['step'] + after_s * fps) if target else 10 ** 9
    os.makedirs(out_dir or '.', exist_ok=True)
    acts = e.get('adversary_actions') or {}
    ego_acts = e.get('actions') or []
    env.reset(seed=int(e['seed']), options=e.get('options') or None)
    if sorted(env.agents) != sorted(acts):
        raise ValueError(f'the replay placed adversaries {env.agents}, the episode had {sorted(acts)}')
    n = max([len(v) for v in acts.values()] + [len(ego_acts), 0])
    frames, met, step = [], False, 0
    for k in range(n):
        adv_k = {aid: (a[k] if not isinstance(a[k], list) else [float(x) for x in a[k]]) for aid, a in acts.items()}
        ego_k = ego_acts[k] if ego_acts else None
        obs, _r, _t, _u, _i = env.step_replay(adv_k, ego_k)
        step = k + 1
        info = env.last_info
        if target and not met:
            met = any(f.get('kind') == target['kind'] and f.get('step') == target['step'] and
                      f.get('actor') == target.get('actor') for f in info.get('failures') or ())
        cam = env.ego_obs.get('camera')
        if lo <= step <= hi and cam is not None:
            p = os.path.join(out_dir, f'{step:06d}.jpg')
            cv2.imwrite(p, cv2.cvtColor(np.asarray(cam), cv2.COLOR_RGB2BGR))
            frames.append(p)
        if env._needs_reset or step > hi:
            break
    doc = {'episode': e.get('episode'), 'seed': e.get('seed'), 'failure': target, 'steps': [lo, min(hi, step)],
           'frames': frames, 'match': met if target else None, 'fps': fps}
    with open(os.path.join(out_dir, 'replay.json'), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    return doc


# ============================================================================================ the CLI
def main_adversary(argv) -> int:
    """`falcon adversary train|eval|replay …` (module doc). 0 ok, 2 failed or a usage error."""
    import argparse
    from . import protocol as P
    from .policy import _kv, env_args
    usage = ('usage: falcon adversary train PROJECT --out DIR [--ego SPEC] [--action KIND] [--curriculum default|FILE]'
             ' [--updates N --steps N] [--resume]\n'
             '       falcon adversary eval PROJECT --policy DIR --out DIR [--ego SPEC]... [--episodes N] '
             '[--baselines LIST] [--tm-reference N]\n'
             '       falcon adversary replay PROJECT --episodes EPISODES.jsonl --episode N [--failure K] --out DIR')
    if not argv or argv[0] not in ('train', 'eval', 'replay'):
        print(usage, flush=True)
        return P.FAILED if argv and argv[0] not in ('-h', '--help') else 0
    what = argv[0]
    ap = argparse.ArgumentParser(prog=f'falcon adversary {what}')
    env_args(ap)
    ap.add_argument('--timeout', type=float, default=20.0, help='CARLA\'s client time-out per call (s)')
    if what == 'train':
        ap.add_argument('--ego', default='follower', help='follower, none, autopilot or a policy folder')
        ap.add_argument('--action', default='lateral', choices=AF.ACTION_KINDS)
        ap.add_argument('--curriculum', default='default', help='"default" or a curriculum JSON file')
        ap.add_argument('--roster', default='', help='train on one roster: comma-separated roles (no curriculum)')
        ap.add_argument('--updates', type=int, default=ADV_PPO_DEFAULTS['total_updates'])
        ap.add_argument('--steps', type=int, default=ADV_PPO_DEFAULTS['n_steps'])
        ap.add_argument('--resume', action='store_true')
        ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE', help='any of ADV_PPO_DEFAULTS')
        ap.add_argument('--reward', action='append', default=[], metavar='TERM=W', help='adversary reward weights')
    elif what == 'eval':
        ap.add_argument('--policy', required=True, help='an adversary policy folder (falcon adversary train)')
        ap.add_argument('--ego', action='append', default=None, help='follower, none, autopilot or a policy folder')
        ap.add_argument('--episodes', type=int, default=20, help='episodes per method and ego (the budget)')
        ap.add_argument('--baselines', default=','.join(METHODS[1:]), help=f'any of {",".join(METHODS[1:])}')
        ap.add_argument('--tm-reference', type=int, default=0, help='background vehicles for the traffic-manager '
                                                                    'dynamics reference (0: none)')
        ap.add_argument('--roster', action='append', default=None, help='comma-separated roles (default: the '
                                                                        'policy\'s last stage)')
    else:
        ap.add_argument('--episodes', required=True, help='an evaluation\'s episodes.jsonl')
        ap.add_argument('--episode', type=int, required=True, help='the line number (0-based)')
        ap.add_argument('--failure', type=int, default=None)
        ap.add_argument('--before', type=float, default=3.0)
        ap.add_argument('--after', type=float, default=1.0)
    try:
        a = ap.parse_args(argv[1:])
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    from .gymenv import make_service
    from .project import Project
    svc = env = None
    try:
        proj = Project.load(a.project)
        svc = make_service(fake_carla=a.fake_carla, workspace=proj.workspace)
        world = a.world or proj.doc['worlds'][0]['id']
        if what == 'train':
            cfg = dict(_kv(a.option), total_updates=a.updates, n_steps=a.steps, seed=a.seed)
            cur = AdvCurriculum.single([r for r in a.roster.split(',') if r]) if a.roster else \
                load_adv_curriculum(a.curriculum)
            env, _ = open_adversary_env(proj, world, os.path.join(a.out, 'env'), svc, a.ego, action=a.action,
                                        rig=a.rig or None, seed=a.seed, fps=a.fps, motion=a.motion,
                                        traffic=a.traffic, log_actions=False, reward=_kv(a.reward),
                                        timeout=a.timeout)
            s = train_adversaries(env, a.out, cfg, cur, a.resume, log=lambda m: print(m, flush=True))
            print(json.dumps(s, indent=2), flush=True)
        elif what == 'eval':
            methods = ['learned'] + [m for m in a.baselines.split(',') if m]
            bad = [m for m in methods if m not in METHODS]
            if bad:
                raise ValueError(f'unknown baselines {bad} (any of {METHODS[1:]})')
            rosters = [[r for r in ro.split(',') if r] for ro in a.roster] if a.roster else None
            doc = evaluate(proj, svc, world, a.policy, a.ego or ['follower'], a.out, a.episodes, methods, a.seed,
                           a.tm_reference, a.rig or None, a.fps, rosters=rosters, timeout=a.timeout,
                           progress=lambda f, t: print(f'{f * 100:5.1f} % {t}', flush=True))
            short = {ego: {m: {k: r.get(k) for k in ('episodes', 'realistic_failures', 'failure_rate',
                                                    'failure_rate_ci95', 'distinct_failures')}
                           for m, r in res.items()} for ego, res in doc['results'].items()}
            print(json.dumps({'results': short, 'failed': doc['failed']}, indent=2), flush=True)
        else:
            with open(a.episodes, encoding='utf-8') as fh:
                rows = [json.loads(x) for x in fh if x.strip()]
            if not 0 <= a.episode < len(rows):
                raise ValueError(f'{a.episodes} has {len(rows)} episodes')
            e = rows[a.episode]
            why = replayable(e)
            if why:
                raise ValueError(why)
            ev = e.get('eval') or {}
            adv = e.get('adversary') or {}
            from .leaderboard import CHASE_CAMERA
            ego = ev.get('ego') or adv.get('ego_driver') or 'follower'
            env, _ = open_adversary_env(proj, ev.get('world') or world, os.path.join(a.out, 'env'), svc, ego,
                                        action=adv.get('action') or ev.get('action') or 'lateral',
                                        rig=ev.get('rig') or a.rig or None, seed=a.seed,
                                        fps=float(ev.get('fps') or a.fps), camera=CHASE_CAMERA,
                                        max_episode_s=float(ev.get('max_episode_s') or 15.0), timeout=a.timeout)
            if env.ego_policy is not None:
                env.ego_policy = None                # the logged ego actions are replayed, not recomputed
            doc = replay(e, env, a.failure, a.before, a.after, a.out)
            print(json.dumps({'frames': len(doc['frames']), 'match': doc['match'], 'steps': doc['steps']}),
                  flush=True)
        return P.OK
    except (EnvError, OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if env is not None:
            env.close()
        if svc is not None and not a.keep_server:
            svc.shutdown()
