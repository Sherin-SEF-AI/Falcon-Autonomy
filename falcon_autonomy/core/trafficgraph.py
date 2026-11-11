"""Differentiable traffic (torch, the app's Python; core.levelk is its numpy twin that runs inside CARLA): the road
users as a graph, a differentiable kinematic bicycle rollout, the level-k quantal-response likelihood differentiable
in its parameters (maximum-likelihood calibration on logs), and gradient-based adversarial search over a scenario's
parameters against a cloned surrogate of the planner under test, each candidate validated by re-running it in CARLA.
docs/levelk.md is the guide.

The graph (proximity_graph): agents are nodes, an edge joins two agents closer than a radius (default
core.levelk.RANGE_M) — the pairs whose gaps enter a utility.

Calibration (calibrate, calibrate_logs). Decision samples (core.levelk.decisions_from_sim: from a simulation, or from
a CARLA run's recorded trajectories, every DT_PLAN) give for each agent its plan options' features (progress, speeds,
acceleration, lateral move, the gaps to the others over the horizon, collisions, red-light crossings, lane offset) and
the plan it was seen to follow. The likelihood of the observed plans is the quantal response of core.levelk.utility —
the SAME function, evaluated on torch tensors — so log P = log softmax(λ U(θ))[observed], differentiable in θ = (α,
λ, v_des, t_gap) through a smooth reparametrisation (α = sigmoid, the others softplus above their PARAM_BOUNDS floor).
The others' future motion is taken as OBSERVED (what they did over the horizon), not as the model's level-(k − 1)
prediction: the estimates are those of a quantal best response to the others' actual behaviour (an inverse-planning
assumption, stated in every result). Maximisation: Adam, then L-BFGS; standard errors from the inverse Hessian of the
negative log-likelihood in the unconstrained parameters, mapped by the delta method (Wald 95 % intervals; they assume
independent decisions, which consecutive decisions of one agent are not — they are optimistic).

Adversarial search (adversarial_search; KING, Hanselmann et al. 2022, ECCV, optimises adversarial agents through a
differentiable kinematic bicycle model). The planner under test is cloned into a small differentiable surrogate
(Surrogate: an MLP from the ego's speed, the gap to the vehicle ahead and the closing speed to its acceleration, fitted
on the ego's own recorded episodes — any ego, a black box included). A scenario family with continuous parameters
(here lead_brake: speed, gap, decel, brake_at; the lead's braking is a smooth step in time) is rolled out
differentiably with the surrogate driving the ego; the objective is a soft minimum over time of the bumper-to-bumper
gap (−log Σ exp(−k gap) / k) plus a penalty outside the parameters' bounds and the realism bound on the lead's
deceleration (core.advfeatures.BRAKE_MAX); projected Adam steps descend it. THE CARLA LOOP ITSELF IS NOT
DIFFERENTIABLE: gradients flow only through the surrogate and the kinematic model, so every candidate is validated by
re-running it in CARLA (validate_in_carla: the adversary environment with the scripted scenario and the real ego) and
the measured minimum gap, TTC and failures are reported beside the surrogate's prediction.

Limits: the surrogate sees only the longitudinal lead situation (speed, gap, closing speed); the search covers the
lead_brake family; likelihood decisions are not independent; calibration from CARLA logs uses each actor's driven
path as its reference path (lateral plans relative to it).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import levelk as LK

SCHEMA = 'falcon.trafficgraph/1'
FREE = ('alpha', 'lam', 'v_des', 't_gap')
LEAD_BRAKE_BOUNDS = {'speed': (20.0 / 3.6, 60.0 / 3.6), 'gap': (8.0, 40.0), 'decel': (3.0, 9.0),
                     'brake_at': (2.0, 5.0)}          # core.scenarios' lead_brake ranges (speed in m/s here)
SOFTMIN_K = 4.0


def _torch():
    import torch
    return torch


class _Torch:
    """core.levelk.utility's array operations on torch tensors."""

    def __init__(self):
        self.t = _torch()

    def exp(self, a):
        return self.t.exp(a)

    def abs(self, a):
        return self.t.abs(a)

    def minimum(self, a, b):
        return self.t.minimum(a, b)

    def maximum(self, a, b):
        return self.t.maximum(a, b)

    def mean(self, a, axis):
        return a.mean(dim=axis)

    def sum(self, a, axis):
        return a.sum(dim=axis)


# ============================================================================================ the graph
def proximity_graph(xy, radius: float = LK.RANGE_M):
    """Edges (2, E) long tensor of agents (N, 2) closer than radius (both directions, no self loops)."""
    torch = _torch()
    xy = torch.as_tensor(xy)
    d = torch.cdist(xy[None].double(), xy[None].double())[0]
    m = (d < radius) & ~torch.eye(len(xy), dtype=torch.bool)
    return torch.nonzero(m).T.contiguous()


# ============================================================================================ the bicycle
def bicycle_rollout(state0, controls, dt: float, wheelbase):
    """Kinematic bicycle (the director's kin_step, without the speed clamp so it stays differentiable): state (B, 4)
    = x, y, yaw (rad), v; controls (B, T, 2) = acceleration, steering angle (rad); wheelbase (B,) or a number.
    Returns (B, T, 4): the state after each step (midpoint heading, as director.kin_step)."""
    torch = _torch()
    x, y, yaw, v = state0.unbind(-1)
    out = []
    for k in range(controls.shape[1]):
        a, delta = controls[:, k, 0], controls[:, k, 1]
        v1 = v + a * dt
        yaw1 = yaw + v1 * torch.tan(delta) / wheelbase * dt
        mid = 0.5 * (yaw + yaw1)
        x = x + torch.cos(mid) * v1 * dt
        y = y + torch.sin(mid) * v1 * dt
        yaw, v = yaw1, v1
        out.append(torch.stack([x, y, yaw, v], -1))
    return torch.stack(out, 1)


# ============================================================================================ calibration
def theta_of(z):
    """θ (dict of 0-d tensors) of the unconstrained vector z (4,)."""
    torch = _torch()
    sp = torch.nn.functional.softplus
    lo = {k: LK.PARAM_BOUNDS[k][0] for k in FREE}
    return {'alpha': torch.sigmoid(z[0]), 'lam': lo['lam'] + sp(z[1]), 'v_des': lo['v_des'] + sp(z[2]),
            't_gap': lo['t_gap'] + sp(z[3])}


def z_of(theta: dict):
    def isp(y):                                    # inverse softplus
        return math.log(math.expm1(max(y, 1e-6)))
    a = min(max(float(theta['alpha']), 1e-4), 1 - 1e-4)
    lo = {k: LK.PARAM_BOUNDS[k][0] for k in FREE}
    return [math.log(a / (1 - a)), isp(float(theta['lam']) - lo['lam']), isp(float(theta['v_des']) - lo['v_des']),
            isp(float(theta['t_gap']) - lo['t_gap'])]


def _batches(samples: list, dtype):
    """Samples grouped by plan count, the gaps padded to the largest other count with 1e6 m (no barrier). A sample's
    'choice' is the observed plan's index; a sample may instead carry 'choices', the indices of the plans the
    observation cannot tell apart (core.levelk_nearmiss: the jerk limit or a standstill makes several plans move the
    same way) — a group with any such sample gets a (B, P) boolean mask instead of the (B,) indices."""
    torch = _torch()
    groups: dict = {}
    for smp in samples:
        groups.setdefault(len(smp['features']['ds']), []).append(smp)
    out = []
    for P, ss in groups.items():
        J = max(s['features']['gaps'].shape[2] for s in ss)
        T = ss[0]['features']['v'].shape[1]
        gaps = np.full((len(ss), P, T, max(J, 1)), 1e6)
        for k, s in enumerate(ss):
            g = s['features']['gaps']
            gaps[k, :, :, :g.shape[2]] = g
        F = {key: torch.as_tensor(np.stack([s['features'][key] for s in ss]), dtype=dtype)
             for key in ('ds', 'v', 'a', 'latmove', 'coll', 'red', 'lane')}
        F['gaps'] = torch.as_tensor(gaps, dtype=dtype)
        F['H'] = float(ss[0]['features']['H'])
        if any(s.get('choices') for s in ss):
            mask = torch.zeros((len(ss), P), dtype=torch.bool)
            for k, s in enumerate(ss):
                mask[k, list(s.get('choices') or [s['choice']])] = True
            out.append((F, mask))
        else:
            out.append((F, torch.as_tensor([s['choice'] for s in ss])))
    return out


def batched_utility(theta: dict, F: dict, weights: dict | None = None):
    """core.levelk.utility over a batch of samples (B, P, …): the same function on torch tensors."""
    return LK.utility(_Torch(), theta, F, weights)


def nll(z, batches, weights=None, fixed: dict | None = None, lapse: float = 0.0):
    """The negative log-likelihood of the observed plans (module doc). lapse ε > 0: a trembling hand — each decision
    is the quantal response with probability 1 − ε and any plan alike with ε (P = (1 − ε) softmax + ε / P), so one
    decision the model cannot explain (an observed plan inside its collision margin, from noisy tracks) does not
    decide the estimate; 0 (the default) is the plain quantal response."""
    torch = _torch()
    th = theta_of(z)
    th.update({k: torch.as_tensor(v, dtype=z.dtype) for k, v in (fixed or {}).items()})
    total = z.sum() * 0.0
    for F, choice in batches:
        U = batched_utility(th, F, weights)
        lp = torch.log_softmax(th['lam'] * U, dim=1)
        if lapse > 0.0:
            # logsumexp of the pair (torch.logaddexp's double backward is NaN far below the floor)
            lp = torch.logsumexp(torch.stack([lp + math.log1p(-lapse),
                                              torch.full_like(lp, math.log(lapse / lp.shape[1]))]), dim=0)
        if choice.dtype == torch.bool:              # a set of plans the observation cannot tell apart: P(set)
            # a finite fill: −inf there turns the Hessian's double backward into NaN
            total = total - torch.logsumexp(lp.masked_fill(~choice, -1e30), dim=1).sum()
        else:
            total = total - lp.gather(1, choice[:, None]).sum()
    return total


def calibrate(samples: list, init: dict | None = None, steps: int = 300, lr: float = 0.05, weights=None,
              fixed: dict | None = None, log=None, lapse: float = 0.0) -> dict:
    """Maximum-likelihood θ (module doc): {'estimates', 'se', 'ci95', 'loglik', 'n_decisions', 'converged'}."""
    torch = _torch()
    if not samples:
        raise ValueError('no decision samples to calibrate on')
    dtype = torch.float64
    batches = _batches(samples, dtype)
    th0 = dict({'alpha': 0.5, 'lam': 5.0, 'v_des': 8.0, 't_gap': 2.0}, **(init or {}))
    z = torch.tensor(z_of(th0), dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for k in range(int(steps)):
        opt.zero_grad()
        loss = nll(z, batches, weights, fixed, lapse)
        loss.backward()
        opt.step()
        if log and (k % 50 == 0 or k == steps - 1):
            log(f'step {k}: negative log-likelihood {loss.item():.4f}')
    lb = torch.optim.LBFGS([z], lr=0.5, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        lb.zero_grad()
        loss_ = nll(z, batches, weights, fixed, lapse)
        loss_.backward()
        return loss_
    lb.step(closure)
    zf = z.detach().clone()
    loss = float(nll(zf, batches, weights, fixed, lapse))
    grad = torch.autograd.functional.jacobian(lambda q: nll(q, batches, weights, fixed, lapse), zf)
    H = torch.autograd.functional.hessian(lambda q: nll(q, batches, weights, fixed, lapse), zf)
    th = {k: float(v) for k, v in theta_of(zf).items()}
    se, ci = {}, {}
    free = [k for k in FREE if k not in (fixed or {})]
    idx = [FREE.index(k) for k in free]
    try:
        cov = torch.linalg.inv(H[idx][:, idx])
        J = torch.autograd.functional.jacobian(lambda q: torch.stack([theta_of(q)[k] for k in FREE]), zf)
        Jf = J[idx][:, idx]
        cth = Jf @ cov @ Jf.T
        for n, k in enumerate(free):
            v = float(cth[n, n])
            se[k] = math.sqrt(v) if v >= 0 else None
            ci[k] = [th[k] - 1.96 * se[k], th[k] + 1.96 * se[k]] if se[k] is not None else None
    except RuntimeError:
        se = {k: None for k in free}
        ci = {k: None for k in free}
    for k in (fixed or {}):
        th[k] = float(fixed[k])
    return {'estimates': th, 'se': se, 'ci95': ci, 'loglik': -loss, 'n_decisions': len(samples),
            'gradient_norm': float(grad.norm()), 'converged': bool(float(grad.norm()) < 1e-2 * max(len(samples), 1)),
            'fixed': dict(fixed or {}), 'lapse': float(lapse),
            'assumption': 'quantal best response to the others\' observed motion (not their level-(k−1) prediction); '
                          'Wald intervals assume independent decisions (optimistic)'}


def sim_from_episode(e: dict) -> dict:
    """A core.levelk simulate()-shaped log from a CARLA episode line with 'trajectories' (workers/gym.py,
    workers/adversary.py with log_trace): each actor's driven polyline as its reference path, s along it, d = 0."""
    tr = e.get('trajectories') or {}
    dt = float(tr.get('dt') or 0.05)
    kinds = dict(tr.get('kinds') or {})
    traj, agents = {}, []
    for aid, rows in (tr.get('actors') or {}).items():
        if len(rows) < 3:
            continue
        R = np.asarray(rows, float)
        xy = R[:, 1:3]
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.05]
        pts = xy[keep]
        if len(pts) < 2:
            r = math.radians(R[0, 3])
            pts = np.asarray([xy[0], xy[0] + [math.cos(r), math.sin(r)]])
        path = LK.RefPath(pts)
        s = np.asarray([path.project(x, y, None)[0] for x, y in xy])
        v = np.hypot(R[:, 4], R[:, 5])
        a = np.r_[0.0, np.diff(v) / dt]
        typ = LK.agent_type(kinds.get(aid, 'car'))
        traj[aid] = {'t': R[:, 0].tolist(), 'x': xy[:, 0].tolist(), 'y': xy[:, 1].tolist(), 'yaw': R[:, 3].tolist(),
                     'v': v.tolist(), 'a': a.tolist(), 's': s.tolist(), 'd': [0.0] * len(s)}
        agents.append({'id': aid, 'type': typ, 'params': LK.params_of({}, typ), 'controlled': aid != 'ego',
                       'priority': aid == 'ego', 'path': path.to_doc()})
    n = min((len(v['t']) for v in traj.values()), default=0)
    for v in traj.values():
        for k in v:
            v[k] = v[k][:n]
    return {'schema': LK.SCHEMA, 'dt': dt, 'trajectories': traj, 'agents': agents}


def calibrate_logs(paths, out_dir: str, agents=None, steps: int = 300, log=None) -> dict:
    """`falcon levelk calibrate`: decision samples from levelk simulations (trajectories.json) and CARLA episode
    logs (episodes.jsonl lines with trajectories), then calibrate(); writes calibration.json."""
    from .provenance import block, write_json
    samples, used = [], []
    for p in paths:
        sims = []
        if p.endswith('.jsonl'):
            with open(p, encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        e = json.loads(line)
                        if e.get('trajectories'):
                            sims.append(sim_from_episode(e))
        else:
            with open(p, encoding='utf-8') as fh:
                sims.append(json.load(fh))
        for sim in sims:
            ids = agents or [a['id'] for a in sim['agents'] if a.get('controlled', True)]
            ids = [i for i in ids if i in sim['trajectories']]
            samples += LK.decisions_from_sim(sim, ids)
        used.append(os.path.abspath(p))
    if log:
        log(f'{len(samples)} decisions from {len(used)} logs')
    doc = calibrate(samples, steps=steps, log=log)
    doc.update(schema=SCHEMA + '#calibration', logs=used, agents=agents)
    os.makedirs(out_dir, exist_ok=True)
    doc['provenance'] = block({f'log{i}': p for i, p in enumerate(used)}, settings={'steps': int(steps)})
    write_json(os.path.join(out_dir, 'calibration.json'), doc)
    return doc


# ============================================================================================ the surrogate planner
SURROGATE_FEATURES = ('speed_mps', 'gap_m', 'closing_mps')
GAP_CAP_M = 60.0


def _classes():
    torch = _torch()

    class Surrogate(torch.nn.Module):
        """accel = f(speed, gap, closing) — a planner's longitudinal behaviour cloned (module doc)."""

        def __init__(self, hidden=32):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(3, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden),
                                           torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
            self.register_buffer('mean', torch.zeros(3, dtype=torch.float64))
            self.register_buffer('std', torch.ones(3, dtype=torch.float64))

        def forward(self, x):
            x = (x - self.mean) / self.std
            return 8.0 * torch.tanh(self.net(x)[..., 0] / 8.0)          # within ±8 m/s² (the director's brake)
    return Surrogate


def fit_surrogate(X, y, epochs: int = 400, seed: int = 0, lr: float = 0.01) -> tuple:
    """(surrogate, {'mae', 'baseline_mae', 'n'}) fitted on X (n, 3) SURROGATE_FEATURES → y (n,) accelerations."""
    torch = _torch()
    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(X, float), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y, float), dtype=torch.float64)
    if len(X) < 10:
        raise ValueError(f'{len(X)} samples are too few to clone the planner (at least 10)')
    S = _classes()().double()
    S.mean.copy_(X.mean(0))
    S.std.copy_(X.std(0).clamp_min(1e-3))
    opt = torch.optim.Adam(S.parameters(), lr=lr)
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = torch.mean((S(X) - y) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mae = float(torch.mean(torch.abs(S(X) - y)))
        base = float(torch.mean(torch.abs(y - y.mean())))
    return S, {'mae': mae, 'baseline_mae': base, 'n': int(len(X))}


def surrogate_data(episodes: list, ego='ego') -> tuple:
    """(X, y) for fit_surrogate from episode lines with trajectories: the ego's speed, the bumper-to-bumper gap to the
    nearest actor ahead within its lane (|lateral| < 1.8 m in its frame) and the closing speed, against its
    acceleration over the next step."""
    X, Y = [], []
    for e in episodes:
        tr = e.get('trajectories') or {}
        dt = float(tr.get('dt') or 0.05)
        acts = tr.get('actors') or {}
        if ego not in acts:
            continue
        E = np.asarray(acts[ego], float)
        others = [np.asarray(v, float) for k, v in acts.items() if k != ego and len(v) == len(E)]
        for k in range(len(E) - 1):
            t, x, y, yaw, vx, vy, L, _W = E[k]
            v = math.hypot(vx, vy)
            r = math.radians(yaw)
            best, close = GAP_CAP_M, 0.0
            for O in others:
                ox, oy = O[k, 1] - x, O[k, 2] - y
                fx, fy = ox * math.cos(r) + oy * math.sin(r), -ox * math.sin(r) + oy * math.cos(r)
                if fx > 0 and abs(fy) < 1.8:
                    g = fx - (L + O[k, 6]) / 2.0
                    if g < best:
                        best = max(g, 0.0)
                        close = v - (O[k, 4] * math.cos(r) + O[k, 5] * math.sin(r))
            v1 = math.hypot(E[k + 1, 4], E[k + 1, 5])
            X.append([v, best, close])
            Y.append((v1 - v) / dt)
    return np.asarray(X, float).reshape(-1, 3), np.asarray(Y, float)


# ============================================================================================ adversarial search
def soft_min(x, k: float = SOFTMIN_K):
    torch = _torch()
    return -torch.logsumexp(-k * x, dim=-1) / k


def lead_brake_rollout(p: dict, surrogate, seconds: float = 12.0, dt: float = 0.05, ego_len: float = 4.5,
                       lead_len: float = 4.5, tau: float = 0.1) -> dict:
    """A differentiable lead_brake episode (module doc): p {'speed' m/s, 'gap' m, 'decel' m/s², 'brake_at' s} as
    0-d tensors; the ego driven by the surrogate, both on a straight lane. Returns {'gap' (T,), 'ego_v', 'lead_v',
    't'} tensors; the lead's speed is kept ≥ 0 smoothly (softplus)."""
    torch = _torch()
    n = int(round(seconds / dt))
    sp = torch.nn.functional.softplus
    xe = torch.zeros((), dtype=torch.float64)
    xl = xe + p['gap'] + (ego_len + lead_len) / 2.0
    ve = p['speed'] * 1.0
    vl = p['speed'] * 1.0
    gaps, evs, lvs, ts = [], [], [], []
    for k in range(n):
        t = k * dt
        g = xl - xe - (ego_len + lead_len) / 2.0
        a_l = -p['decel'] * torch.sigmoid((t - p['brake_at']) / tau)
        a_e = surrogate(torch.stack([ve, g.clamp(max=GAP_CAP_M), ve - vl]))
        vl = sp(vl + a_l * dt, beta=20.0)
        ve = sp(ve + a_e * dt, beta=20.0)
        xl = xl + vl * dt
        xe = xe + ve * dt
        gaps.append(xl - xe - (ego_len + lead_len) / 2.0)
        evs.append(ve)
        lvs.append(vl)
        ts.append(t + dt)
    return {'gap': torch.stack(gaps), 'ego_v': torch.stack(evs), 'lead_v': torch.stack(lvs), 't': ts}


def adversarial_search(surrogate, init: dict, steps: int = 60, lr: float = 0.3, bounds: dict | None = None,
                       realism_decel: float | None = None, seconds: float = 12.0, dt: float = 0.05) -> dict:
    """Projected Adam on the lead_brake parameters to minimise the soft-min gap (module doc): {'start', 'best',
    'path': [{values, soft_min_gap, min_gap}], 'predicted_min_gap_m'}."""
    torch = _torch()
    from .advfeatures import BRAKE_MAX
    b = dict(LEAD_BRAKE_BOUNDS, **(bounds or {}))
    cap = float(realism_decel if realism_decel is not None else BRAKE_MAX)
    b['decel'] = (b['decel'][0], min(b['decel'][1], cap))
    keys = ('speed', 'gap', 'decel', 'brake_at')
    q = {k: torch.tensor(float(np.clip(init[k], *b[k])), dtype=torch.float64, requires_grad=True) for k in keys}
    opt = torch.optim.Adam(list(q.values()), lr=lr)
    for p in surrogate.parameters():
        p.requires_grad_(False)
    path = []
    best = None
    for k in range(int(steps) + 1):
        roll_ = lead_brake_rollout(q, surrogate, seconds, dt)
        obj = soft_min(roll_['gap'])
        mg = float(roll_['gap'].detach().min())
        row = {'values': {kk: float(v.detach()) for kk, v in q.items()}, 'soft_min_gap': float(obj.detach()),
               'min_gap': mg}
        path.append(row)
        if best is None or mg < best['min_gap']:
            best = row
        if k == steps:
            break
        opt.zero_grad()
        obj.backward()
        opt.step()
        with torch.no_grad():
            for kk in keys:
                q[kk].clamp_(*b[kk])
    return {'start': path[0], 'best': best, 'path': path, 'predicted_min_gap_m': best['min_gap'], 'bounds': b,
            'note': 'gradients through the surrogate and the kinematic model only: the CARLA loop is not '
                    'differentiable; validate_in_carla re-runs the candidate'}


def scenario_values(v: dict) -> dict:
    """lead_brake's scenario values (core.scenarios: speed in km/h) of searched values (m/s)."""
    return {'speed': float(v['speed']) * 3.6, 'gap': float(v['gap']), 'decel': float(v['decel']),
            'brake_at': float(v['brake_at'])}


def clone_ego(env, episodes: int = 6, seed: int = 0, rng_values=None) -> tuple:
    """Run lead_brake episodes with random values (core.scenarios' own draw per seed) in an adversary environment with
    its ego (env: core.adversary.AdversaryEnv, log_trace on) and fit the surrogate: (surrogate, fit report, lines)."""
    from .adversary import run_episode
    lines = [run_episode(env, lambda aid, o: None, seed + k, {'scenario': 'lead_brake'}) for k in range(episodes)]
    X, y = surrogate_data(lines)
    S, rep = fit_surrogate(X, y, seed=seed)
    return S, rep, lines


def validate_in_carla(env, values: dict, seed: int = 0) -> dict:
    """Re-run a candidate in CARLA (an adversary environment with the scripted lead_brake and the real ego): the
    measured minimum gap and TTC to the lead and the ego's failures."""
    from .adversary import run_episode
    e = run_episode(env, lambda aid, o: None, seed, {'scenario': 'lead_brake', 'values': scenario_values(values)})
    adv = e.get('adversary') or {}
    lead = (adv.get('actors') or {}).get('lead') or {}
    return {'values': scenario_values(values), 'min_gap_m': lead.get('min_gap_m'), 'min_ttc_s': lead.get('min_ttc_s'),
            'failed': bool(adv.get('failed')), 'failures': [f.get('kind') for f in adv.get('failures') or ()],
            'termination': e.get('termination')}


def attack(env, out_dir: str = '', clone_episodes: int = 6, steps: int = 60, seed: int = 0, log=None) -> dict:
    """Clone the ego, search adversarially from the scenario's centre, validate the start and the best in CARLA."""
    from .provenance import block, write_json
    t0 = time.monotonic()
    log = log or (lambda m: None)
    S, rep, _ = clone_ego(env, clone_episodes, seed)
    log(f'surrogate fitted on {rep["n"]} steps: MAE {rep["mae"]:.3f} m/s² '
        f'(constant baseline {rep["baseline_mae"]:.3f})')
    init = {k: 0.5 * (lo + hi) for k, (lo, hi) in LEAD_BRAKE_BOUNDS.items()}
    res = adversarial_search(S, init, steps)
    log(f'search: predicted min gap {res["start"]["min_gap"]:.2f} m → {res["best"]["min_gap"]:.2f} m')
    v0 = validate_in_carla(env, res['start']['values'], seed + 1000)
    v1 = validate_in_carla(env, res['best']['values'], seed + 1000)
    doc = {'schema': SCHEMA + '#attack', 'surrogate': rep, 'search': {k: res[k] for k in ('start', 'best', 'bounds',
                                                                                           'note')},
           'path': res['path'], 'validation': {'start': v0, 'best': v1},
           'prediction_error_m': (None if v1['min_gap_m'] is None else
                                  float(v1['min_gap_m']) - res['best']['min_gap']),
           'wall_s': round(time.monotonic() - t0, 2)}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        doc['provenance'] = block({'environment': os.path.join(env.run_dir, 'gym.json')}, seed=seed,
                                  settings={'clone_episodes': clone_episodes, 'steps': steps})
        write_json(os.path.join(out_dir, 'attack.json'), doc)
    return doc
