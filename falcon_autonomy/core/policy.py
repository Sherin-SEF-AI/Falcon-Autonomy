"""Driving policies: the folder format shared by behaviour cloning (core.imitation) and PPO (core.rl), their
networks, the one loader, and evaluation through the environment (numpy at import; torch only inside functions).

A policy folder:

    policy.json   {"schema": "falcon.policy/1", "name", "trained_by": "bc" | "ppo",
                   "features": [names of core.drivefeatures.FEATURES the network reads, in its input order],
                   "normaliser": {"mean": [...], "std": [...], "clip": c}   x_in = clip((x − mean) / std, −c, c),
                   "bev": BEV spec | null      (a CNN also reads the "bev" observation, 8 × H × W in 0/1),
                   "action": {"kind": "continuous", "keys": ["steer", "accel"]} | {"kind": "discrete", "n": 25},
                   "net": {"kind": "mlp" | "cnn", "hidden": [...], "activation": "relu" | "tanh",
                           "output": "tanh" | "linear" | "logits", "conv": [...]},
                   "weights": "weights.pt", "weights_sha256", "drivefeatures": "falcon.drivefeatures/1",
                   "provenance": {...}}
    weights.pt    the network's state_dict (torch.save; loaded with weights_only=True)
    metrics.json  what its trainer measured
    run.json      the training run (falcon.run/1 with provenance)

Acting (Policy.act): the observation's "state" vector is reduced to policy.json's features BY NAME (so a policy keeps
working when FEATURES grows), normalised, and run through the network: output "tanh" (behaviour cloning) is the
action; "linear" (PPO's Gaussian mean) is clipped to [−1, 1]; "logits" (discrete) gives the argmax index. Acting is
deterministic (the mean, the argmax): rollouts of the same policy on the same seeds repeat.

Evaluation through the environment (rollout): episodes with seeds seed, seed + 1, …, one action per step, the
environment's own episode summaries (core.gymenv / workers/gym.py) aggregated: return, route completion, progress,
success (core.drivefeatures.episode_success), collisions per episode, infractions per km, terminations. rollout.json
holds every episode (with its seed and options, so any can be replayed) and the aggregate, with provenance.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import drivefeatures as DF

POLICY_SCHEMA = 'falcon.policy/1'
ROLLOUT_SCHEMA = 'falcon.rollout/1'
TL_FEATURES = ('tl_known', 'tl_red', 'tl_yellow', 'tl_green', 'tl_distance_m')


def net_spec(kind='mlp', hidden=(128, 128), activation='relu', output='tanh', conv=(16, 32, 32)) -> dict:
    if kind not in ('mlp', 'cnn'):
        raise ValueError(f'net kind must be mlp or cnn, not {kind!r}')
    return {'kind': kind, 'hidden': [int(h) for h in hidden], 'activation': activation, 'output': output,
            'conv': [int(c) for c in conv]}


_CLASSES: dict = {}


def _classes() -> dict:
    """The torch module classes (defined on first use: torch is imported only here)."""
    if _CLASSES:
        return _CLASSES
    import torch
    from torch import nn

    def act(name):
        return {'relu': nn.ReLU, 'tanh': nn.Tanh}[name]()

    def mlp(n_in, hidden, n_out, activation):
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), act(activation)]
            d = h
        layers.append(nn.Linear(d, n_out))
        return nn.Sequential(*layers)

    class MLPNet(nn.Module):
        """state (B, F) → (B, n_out)."""

        def __init__(self, spec, n_in, n_out):
            super().__init__()
            self.body = mlp(n_in, spec['hidden'], n_out, spec['activation'])
            self.out = spec['output']

        def forward(self, state, bev=None):
            y = self.body(state)
            return torch.tanh(y) if self.out == 'tanh' else y

    class CNNNet(nn.Module):
        """bev (B, 8, H, W) through three strided convolutions and a 4 × 4 average pool, concatenated with the
        state, then the MLP head."""

        def __init__(self, spec, n_in, n_out):
            super().__init__()
            c1, c2, c3 = spec['conv']
            self.conv = nn.Sequential(nn.Conv2d(8, c1, 5, stride=2, padding=2), act(spec['activation']),
                                      nn.Conv2d(c1, c2, 3, stride=2, padding=1), act(spec['activation']),
                                      nn.Conv2d(c2, c3, 3, stride=2, padding=1), act(spec['activation']),
                                      nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
            self.body = mlp(n_in + 16 * c3, spec['hidden'], n_out, spec['activation'])
            self.out = spec['output']

        def forward(self, state, bev=None):
            y = self.body(torch.cat([state, self.conv(bev)], dim=1))
            return torch.tanh(y) if self.out == 'tanh' else y

    _CLASSES.update(MLPNet=MLPNet, CNNNet=CNNNet, mlp=mlp)
    return _CLASSES


def build_net(spec: dict, n_in: int, n_out: int):
    c = _classes()
    return (c['CNNNet'] if spec['kind'] == 'cnn' else c['MLPNet'])(spec, n_in, n_out)


def feature_index(names, available=DF.FEATURE_NAMES) -> np.ndarray:
    """Indexes of `names` in the `available` feature order; ValueError naming any missing."""
    pos = {n: k for k, n in enumerate(available)}
    missing = [n for n in names if n not in pos]
    if missing:
        raise ValueError(f'the observation has no features {missing}')
    return np.asarray([pos[n] for n in names], np.int64)


def select_features(exclude=()) -> list:
    bad = sorted(set(exclude) - set(DF.FEATURE_NAMES))
    if bad:
        raise ValueError(f'unknown features {bad}')
    return [n for n in DF.FEATURE_NAMES if n not in set(exclude)]


class RunningMeanStd:
    """Running mean and variance (Chan et al.'s parallel update), as observation and return normalisers use."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = float(epsilon)

    def update(self, x):
        x = np.asarray(x, np.float64).reshape((-1,) + self.mean.shape)
        bm, bv, n = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + n
        self.mean = self.mean + d * n / tot
        self.var = (self.var * self.count + bv * n + d * d * self.count * n / tot) / tot
        self.count = tot

    def state(self) -> dict:
        return {'mean': self.mean.tolist(), 'var': self.var.tolist(), 'count': self.count}

    def load(self, d: dict):
        self.mean, self.var = np.asarray(d['mean'], np.float64), np.asarray(d['var'], np.float64)
        self.count = float(d['count'])
        return self


def normalise(x, norm: dict) -> np.ndarray:
    return np.clip((np.asarray(x, np.float64) - np.asarray(norm['mean'])) / np.asarray(norm['std']),
                   -norm['clip'], norm['clip']).astype(np.float32)


# ============================================================================================ saving and loading
def save_policy(out_dir: str, net, doc: dict, metrics: dict | None = None, run: dict | None = None) -> str:
    """Write a policy folder (module doc): doc is policy.json without weights/hash; returns its path."""
    import torch
    from .provenance import sha256_file, write_json
    os.makedirs(out_dir, exist_ok=True)
    wp = os.path.join(out_dir, 'weights.pt')
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, wp)
    d = dict(doc, schema=POLICY_SCHEMA, weights='weights.pt', weights_sha256=sha256_file(wp),
             drivefeatures=DF.SCHEMA)
    write_json(os.path.join(out_dir, 'policy.json'), d)
    if metrics is not None:
        write_json(os.path.join(out_dir, 'metrics.json'), metrics)
    if run is not None:
        write_json(os.path.join(out_dir, 'run.json'), run)
    return out_dir


class Policy:
    """A loaded policy folder: act(obs) → action (module doc)."""

    def __init__(self, path: str, device: str = 'cpu'):
        import torch
        from .provenance import read_json, sha256_file
        self.path = os.path.abspath(path)
        self.doc = read_json(os.path.join(self.path, 'policy.json'))
        if self.doc.get('schema') != POLICY_SCHEMA:
            raise ValueError(f'{path}: not a policy folder (schema {self.doc.get("schema")!r})')
        wp = os.path.join(self.path, self.doc.get('weights', 'weights.pt'))
        if self.doc.get('weights_sha256') and sha256_file(wp) != self.doc['weights_sha256']:
            raise ValueError(f'{wp}: its SHA-256 is not the one policy.json records')
        self.features = list(self.doc['features'])
        self.norm = self.doc['normaliser']
        self.action = self.doc['action']
        n_out = int(self.action['n']) if self.action['kind'] == 'discrete' else 2
        self.device = device
        self.net = build_net(self.doc['net'], len(self.features), n_out)
        self.net.load_state_dict(torch.load(wp, map_location='cpu', weights_only=True))
        self.net.to(device).eval()
        self.uses_bev = self.doc['net']['kind'] == 'cnn'
        self._idx = {}

    def index(self, names=DF.FEATURE_NAMES) -> np.ndarray:
        key = tuple(names)
        if key not in self._idx:
            self._idx[key] = feature_index(self.features, names)
        return self._idx[key]

    def act_batch(self, states, bevs=None, names=DF.FEATURE_NAMES) -> np.ndarray:
        """Actions for a batch: states (B, len(names)) float; bevs (B, 8, H, W) for a CNN. (B, 2) or (B,)."""
        import torch
        x = normalise(np.asarray(states)[:, self.index(names)], self.norm)
        with torch.no_grad():
            b = None
            if self.uses_bev:
                if bevs is None:
                    raise ValueError('this policy reads the BEV: run the environment with bev on')
                b = torch.as_tensor(np.asarray(bevs, np.float32), device=self.device)
            y = self.net(torch.as_tensor(x, device=self.device), b).cpu().numpy()
        if self.action['kind'] == 'discrete':
            return y.argmax(1)
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def act(self, obs: dict, names=DF.FEATURE_NAMES):
        """The action for one observation (the environment's dict)."""
        y = self.act_batch(np.asarray(obs['state'])[None], None if not self.uses_bev else obs['bev'][None], names)
        return int(y[0]) if self.action['kind'] == 'discrete' else y[0]


def load_policy(path: str, device: str = 'cpu') -> Policy:
    return Policy(path, device)


# ============================================================================================ evaluation in the env
def summarise_episodes(rows: list) -> dict:
    """The aggregate of episode summaries (module doc). Infractions per km: every recorded infraction over the
    distance driven; None when nothing was driven."""
    n = len(rows)
    if not n:
        return {'episodes': 0}
    ret = np.array([r['return'] for r in rows], float)
    comp = np.array([r['route_completion'] for r in rows], float)
    dist = float(sum(r.get('distance_m', 0.0) for r in rows))
    counts = {k: int(sum((r.get('infraction_counts') or {}).get(k, 0) for r in rows)) for k in DF.INFRACTIONS}
    terms = {}
    for r in rows:
        terms[r.get('termination')] = terms.get(r.get('termination'), 0) + 1
    coll = sum(counts[k] for k in DF.COLLISION_KINDS)
    return {'episodes': n, 'mean_return': float(ret.mean()), 'sd_return': float(ret.std(ddof=1)) if n > 1 else 0.0,
            'mean_route_completion': float(comp.mean()),
            'mean_progress_m': float(np.mean([r['progress_m'] for r in rows])),
            'success_rate': float(np.mean([bool(r.get('success')) for r in rows])),
            'collisions_per_episode': coll / n, 'distance_km': dist / 1000.0,
            'infractions_per_km': (sum(counts.values()) / (dist / 1000.0)) if dist > 0 else None,
            'infraction_counts': counts, 'terminations': terms,
            'mean_steps': float(np.mean([r['steps'] for r in rows]))}


def rollout(policy, env, episodes: int, seed: int = 0, options: dict | None = None, out_dir: str = '',
            progress=None) -> dict:
    """Run `episodes` episodes of a policy (a Policy, or any callable obs → action) in the env with seeds seed + k;
    returns (and with out_dir writes rollout.json) the episodes and their aggregate."""
    from .provenance import block, write_json
    act = policy.act if hasattr(policy, 'act') else policy
    names = env.feature_names
    rows = []
    for k in range(int(episodes)):
        obs, info = env.reset(seed=int(seed) + k, options=options)
        done = False
        summ = None
        while not done:
            a = act(obs, names) if hasattr(policy, 'act') else act(obs)
            obs, _r, term, trunc, info = env.step(a)
            done = term or trunc
            summ = info.get('episode_summary')
        rows.append(summ)
        if progress:
            progress(k + 1, summ)
    doc = {'schema': ROLLOUT_SCHEMA, 'episodes': rows, 'aggregate': summarise_episodes(rows),
           'seeds': [int(seed) + k for k in range(int(episodes))], 'options': options or {},
           'policy': getattr(policy, 'path', None),
           'environment': {'run_dir': env.run_dir, 'map': env.ready.get('map'), 'fps': env.ready.get('fps'),
                           'action': env.ready.get('action', {}).get('kind')}}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ins = {'environment': os.path.join(env.run_dir, 'gym.json')}
        if getattr(policy, 'path', None):
            ins['policy'] = os.path.join(policy.path, 'policy.json')
        doc['provenance'] = block(ins, seed=int(seed), settings={'episodes': int(episodes), 'options': options or {}})
        write_json(os.path.join(out_dir, 'rollout.json'), doc)
    return doc


# ============================================================================================ the CLI
def _kv(items) -> dict:
    out = {}
    for kv in items or ():
        k, eq, v = kv.partition('=')
        if not eq:
            raise ValueError(f'option {kv!r} is not KEY=VALUE')
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main_policy(argv) -> int:
    """`falcon policy bc|ppo|rollout|offline|leaderboard|replay …` (docs/policies.md). Exit codes as the CLI's: 0 ok, 2 failed or a usage
    error, a worker's code otherwise."""
    from . import protocol as P
    usage = ('usage: falcon policy bc --run RUN... --out DIR [options]\n'
             '       falcon policy offline --policy DIR --run RUN... [--out DIR]\n'
             '       falcon policy ppo PROJECT --out DIR [--scenario KEY | --curriculum default|FILE] [options]\n'
             '       falcon policy rollout PROJECT --policy DIR --out DIR [--episodes N] [options]\n'
             '       falcon policy leaderboard PROJECT --policy DIR --out DIR [--scenario K]... [--seeds-per N] '
             '[--free N]\n'
             '       falcon policy replay PROJECT --episodes EPISODES.jsonl --episode N [--infraction K] --out DIR')
    if not argv or argv[0] not in ('bc', 'ppo', 'rollout', 'offline', 'leaderboard', 'replay'):
        print(usage, flush=True)
        return P.FAILED if argv and argv[0] not in ('-h', '--help') else 0
    if argv[0] == 'bc':
        from .imitation import main_bc
        return main_bc(argv[1:])
    if argv[0] == 'offline':
        from .imitation import main_offline
        return main_offline(argv[1:])
    if argv[0] in ('leaderboard', 'replay'):
        from .leaderboard import main_leaderboard
        return main_leaderboard(argv)
    if argv[0] == 'ppo':
        from .rl import main_ppo
        return main_ppo(argv[1:])
    return main_rollout(argv[1:])


def env_args(ap):
    """The environment options `falcon policy ppo` and `rollout` share."""
    ap.add_argument('project')
    ap.add_argument('--out', '-o', required=True)
    ap.add_argument('--world', default='')
    ap.add_argument('--rig', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--motion', default='kinematic', choices=('kinematic', 'physics'))
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--bev', action='store_true', help='add the BEV observation (needed by a CNN policy)')
    ap.add_argument('--traffic', type=int, default=0, help='background vehicles (episodes are then not repeatable)')
    ap.add_argument('--keep-server', action='store_true')
    ap.add_argument('--fake-carla', default='')


def open_env(a, out_dir, scenario=None, action='continuous', log_actions=True, bev=None):
    """(service, env) for the CLI: Falcon's server and the environment worker, as every CARLA job. bev: a BEV spec
    dict (a CNN policy's own), else --bev's default."""
    from .gymenv import FalconDriveEnv, gym_config, make_service
    from .project import Project
    proj = Project.load(a.project)
    svc = make_service(fake_carla=a.fake_carla, workspace=proj.workspace)
    try:
        cfg = gym_config(proj, a.world or proj.doc['worlds'][0]['id'], out_dir, svc.port, svc.tm_port,
                         rig_id=a.rig or None, seed=a.seed, scenario=scenario, motion=a.motion, action=action,
                         bev=bev or bool(a.bev), fps=a.fps, traffic_vehicles=a.traffic, log_actions=log_actions,
                         workspace=proj.workspace)
        return svc, FalconDriveEnv(cfg, svc)
    except BaseException:
        if not a.keep_server:
            svc.shutdown()
        raise


def main_rollout(argv) -> int:
    import argparse
    from . import protocol as P
    from .gymenv import EnvError
    ap = argparse.ArgumentParser(prog='falcon policy rollout', description='evaluate a policy through the environment')
    env_args(ap)
    ap.add_argument('--policy', required=True, help='a policy folder (falcon policy bc or ppo)')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--scenario', default='', help='a built-in or project scenario; empty: free driving')
    ap.add_argument('--device', default='cpu')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else P.FAILED
    try:
        pol = load_policy(a.policy, a.device)
    except (OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    svc = env = None
    try:
        svc, env = open_env(a, os.path.join(a.out, 'env'), a.scenario or None, pol.action['kind'],
                            bev=pol.doc.get('bev') if pol.uses_bev else None)
        doc = rollout(pol, env, a.episodes, a.seed, out_dir=a.out,
                      progress=lambda k, s: print(f'episode {k}: return {s["return"]:.2f}, completion '
                                                  f'{s["route_completion"]:.2f}, {s["termination"]}', flush=True))
    except (EnvError, OSError, ValueError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if env is not None:
            env.close()
        if svc is not None and not a.keep_server:
            svc.shutdown()
    agg = doc['aggregate']
    print(json.dumps({k: agg[k] for k in ('episodes', 'mean_return', 'mean_route_completion', 'success_rate',
                                          'collisions_per_episode', 'infractions_per_km')}, indent=2), flush=True)
    return P.OK

