"""The release gate between two versions: run the old (A) and the new (B) stack or model on identical scenarios with
identical seeds and answer "better / worse / no difference within a margin / can't tell yet" with confidence that
stays valid however often one looks, stopping as soon as the answer is clear (standard library + numpy).
`falcon gate`; docs/release-gate.md is the manual.

THE DESIGN: PAIRS WITH COMMON RANDOM NUMBERS. Trial i is one scenario instance — scenario i mod m of the listed
scenarios, its parameters drawn from their declared distributions with the seed `seed · 1000003 + i`
(core.scenarios.sample_params) — run once under A and once under B with that same seed, world, rig, rate and motion.
Each run gives a binary outcome (FAILED: the scenario's pass criteria were not met, or with `failure: "value"` its
value reached the threshold) and a continuous one in [0, 1] (the VALUE: the run's criticality, core.criticality; for a
model the perception failure of core.milsearch on the same recorded run). What differs between the two runs of a
pair is the version alone, so the pair's difference carries the version's effect without the scenario's variance.

A version is: a baseline driver (`follower`, `autopilot`), the `stack` block of a falcon.suite/1 file (a ROS 2 stack
with its launch commands: `stack:SUITE.json` or the file's path), or a registered detector (`model:ID`; both versions
then score the SAME recorded run, driven once).

THE STATISTICS. All three bounds are time-uniform: they hold simultaneously at every number of trials, so stopping at
the first clear answer (optional stopping) does not inflate the error.

1. Discordant pairs (the binary outcome). Among the pairs where exactly one version failed, K of N are "B failed, A
   passed". With independent trials K ~ Binomial(N, θ), θ = P(B alone fails | discordant); the versions fail equally
   often exactly when θ = 1/2 (McNemar 1947). The test is a MIXTURE sequential probability ratio test: the likelihood
   ratio of θ against 1/2 averaged over a uniform prior on θ,

       E_N = ∫ θ^K (1−θ)^(N−K) dθ / (1/2)^N = 2^N / ((N + 1) · C(N, K)),

   a non-negative martingale under θ = 1/2 with E_0 = 1, so P(∃N: E_N ≥ 1/α) ≤ α (Ville 1939). The same statistic
   against any θ0 gives the confidence sequence {θ0 : B(K+1, N−K+1) / (θ0^K (1−θ0)^(N−K)) < 1/α} (Robbins 1970,
   Statistical methods related to the law of the iterated logarithm, Ann. Math. Statist. 41 — the method of
   mixtures). Why the mixture and not Wald's SPRT (Wald 1945): Wald's test needs one alternative θ1 chosen
   beforehand and is efficient only near it; the mixture needs none, is two-sided, and yields an interval. Nothing
   in it is a recalled constant: the formula above is exact.
2. The paired difference of the failure indicators, d_i = fail_B − fail_A ∈ {−1, 0, 1}, whose mean is the RISK
   DIFFERENCE Δ = P(B fails) − P(A fails); and
3. the paired difference of the values, x_i = value_B − value_A ∈ [−1, 1]:
   a predictable plug-in EMPIRICAL-BERNSTEIN confidence sequence for the mean of a bounded variable (Waudby-Smith &
   Ramdas 2024, Estimating means of bounded random variables by betting, JRSS B 86, their Theorem 2; the underlying
   exponential supermartingale is Howard, Ramdas, McAuliffe & Sekhon 2021, Time-uniform, nonparametric,
   nonasymptotic confidence sequences, Ann. Statist. 49). With X_i the difference mapped to [0, 1],
   μ̂_t = (1/2 + Σ_{i≤t} X_i)/(t + 1), σ̂²_t = (1/4 + Σ_{i≤t} (X_i − μ̂_i)²)/(t + 1), and a bet
   λ_t = min(√(2·log(2/α) / (σ̂²_{t−1} · t · log(1 + t))), c) that uses only the past,

       μ ∈ [ (Σ λ_i X_i ∓ (log(2/α) + Σ ψ(λ_i)(X_i − μ̂_{i−1})²)) / Σ λ_i ]  for every t,  ψ(λ) = −log(1−λ) − λ.

   Validity does not rest on memory: for ξ ≥ −1 and 0 ≤ λ < 1, exp(λξ − ψ(λ)ξ²) ≤ 1 + λξ (Fan, Grama & Liu 2015;
   checked numerically on a grid in the tests), so with ξ_i = X_i − μ̂_{i−1} the product of
   exp(λ_i(X_i − μ) − ψ(λ_i)ξ_i²) is a supermartingale for ANY predictable λ_i ∈ [0, 1) and μ̂_{i−1} ∈ [0, 1], and
   Ville's inequality gives each side at α/2. RECALLED, not checked against the paper: the tuning — the bet's
   formula, the starting values 1/2 and 1/4, and c = 1/2. They change the interval's width, never its coverage.
   The running intersection of the intervals is reported (also valid).

THE VERDICT, with primary outcome "failure" (default) and total error α (default 0.05) split α/2 to (1) and α/2 to
(2):
   worse        θ's interval lies above 1/2, or Δ's above 0          (B fails more)
   better       θ's interval lies below 1/2, or Δ's below 0
   equivalent   Δ's interval lies inside (−margin, +margin)          "no difference within the margin"
   non_inferior only when a non-inferiority margin is given: Δ's upper bound is below it ("not worse by more than")
   cant_tell    none of these when the trial budget is spent
checked in that order after every batch; the first that holds stops the gate. A difference that is shown AND inside
the margin is reported as better / worse with "within_margin" true. The value's interval (3) is computed at α on its
own and reported as a secondary result; with primary outcome "value" the verdict is read from (3) at α instead
(positive = B more critical) and (1), (2) are secondary. Whenever every interval covers its truth — probability at
least 1 − α — no verdict is wrong: the false "better / worse" rate under no difference, with a look after every trial,
is at most α (verified by simulation in tests/test_releasegate.py).

What a verdict needs, at α = 0.05 with the failure primary: "worse" at the earliest after 9 discordant pairs all one
way (2⁹/10 = 51 ≥ 40); "equivalent" with no discordant pair at all after about 17.5 / margin trials (350 for a margin
of 0.05, 88 for 0.2) — showing the absence of a difference is expensive, and the gate says "can't tell yet" before.

LIMITS. A verdict holds for the tested scenario distribution only: these scenarios, their declared parameter
distributions, this world, rig and motion — not for the road. Trials are taken as independent and identically
distributed draws of that distribution (the round-robin over scenarios makes them exchangeable within a round, not
i.i.d.; with one scenario they are i.i.d.). In kinematic motion a run repeats bit for bit, so a pair's difference is
the version's; in physics motion or with traffic it also holds run-to-run noise, which costs trials, not validity. A
pair with a side that did not finish is not a trial: it is listed under "skipped" with the reason and not run again
(which assumes that not finishing is unrelated to the outcome).

Outputs in <workspace>/gates/<name>/: gate.json (GATE_SCHEMA: the verdict, the bounds after every trial, the trials
used, the newly failing and newly passing scenarios each with its pair of runs), trials.jsonl (one line per finished
pair), gate.md (the same as a report section), runs/a|b/<key>/ (stack versions) or drives/ (model versions).
Re-running resumes: finished pairs are read back, and a gate that already reached a verdict runs nothing.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

GATE_SCHEMA = 'falcon.gate/1'
VERDICTS = ('better', 'worse', 'equivalent', 'non_inferior', 'cant_tell')
WORDS = {'better': 'better', 'worse': 'worse', 'equivalent': 'no difference within the margin',
         'non_inferior': 'not worse within the margin', 'cant_tell': 'can\'t tell yet'}
TONES = {'better': 'good', 'worse': 'critical', 'equivalent': 'good', 'non_inferior': 'good', 'cant_tell': 'warning'}
BET_CAP = 0.5                      # c: the largest bet (recalled tuning; any value below 1 keeps the coverage)
PRIMARY = ('failure', 'value')
FAILURE = ('criteria', 'value')


class GateError(Exception):
    pass


# ================================================================================================ the statistics
def psi_e(lam):
    """ψ(λ) = −log(1 − λ) − λ, 0 ≤ λ < 1."""
    lam = np.asarray(lam, dtype=float)
    return -np.log1p(-lam) - lam


def eb_sequence(x, alpha: float = 0.05, lo: float = -1.0, hi: float = 1.0, cap: float = BET_CAP) -> tuple:
    """The empirical-Bernstein confidence sequence (module doc, 3) for the mean of values in [lo, hi]: (lower,
    upper), arrays over the last axis of x — the running intersection after each observation, each side at α/2.
    Leading axes are independent sequences (the simulation in the tests)."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] == 0:
        return np.zeros(x.shape), np.zeros(x.shape)
    if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12):
        raise ValueError(f'values outside [{lo}, {hi}]')
    X = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    t = np.arange(1, X.shape[-1] + 1, dtype=float)
    mu = (0.5 + np.cumsum(X, axis=-1)) / (t + 1.0)                       # μ̂_t
    var = (0.25 + np.cumsum((X - mu) ** 2, axis=-1)) / (t + 1.0)         # σ̂²_t
    mu_prev = np.concatenate([np.full(X.shape[:-1] + (1,), 0.5), mu[..., :-1]], axis=-1)
    var_prev = np.concatenate([np.full(X.shape[:-1] + (1,), 0.25), var[..., :-1]], axis=-1)
    L = math.log(2.0 / alpha)
    lam = np.minimum(np.sqrt(2.0 * L / (var_prev * t * np.log1p(t))), cap)
    S = np.cumsum(lam, axis=-1)
    centre = np.cumsum(lam * X, axis=-1) / S
    width = (L + np.cumsum(psi_e(lam) * (X - mu_prev) ** 2, axis=-1)) / S
    lower = np.maximum.accumulate(np.clip(centre - width, 0.0, 1.0), axis=-1)
    upper = np.minimum.accumulate(np.clip(centre + width, 0.0, 1.0), axis=-1)
    return lo + (hi - lo) * lower, lo + (hi - lo) * upper


_LOGFACT = np.zeros(1)


def _logfact(n: int) -> np.ndarray:
    global _LOGFACT
    if len(_LOGFACT) <= n:
        m = max(n + 1, 2 * len(_LOGFACT))
        _LOGFACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, m, dtype=float)))])
    return _LOGFACT


def log_mixture(k, n) -> np.ndarray:
    """log ∫ θ^k (1−θ)^(n−k) dθ = log B(k + 1, n − k + 1) = −log(n + 1) − log C(n, k), for integer arrays."""
    k, n = np.asarray(k, dtype=int), np.asarray(n, dtype=int)
    lf = _logfact(int(n.max()) + 1 if n.size else 1)
    return -np.log(n + 1.0) - (lf[n] - lf[k] - lf[n - k])


def discordant_log_e(k, n) -> np.ndarray:
    """log E_N of the mixture test of θ = 1/2 (module doc, 1) after N = n discordant pairs with K = k."""
    return log_mixture(k, n) + np.asarray(n, dtype=float) * math.log(2.0)


def discordant_interval(k: int, n: int, alpha: float = 0.05) -> tuple:
    """The mixture confidence interval for θ at one (K, N): {θ0 : log B(K+1, N−K+1) − K log θ0 − (N−K) log(1−θ0)
    < log(1/α)}; (0, 1) with no discordant pair. Taken at every N it is a confidence sequence."""
    k, n = int(k), int(n)
    if n <= 0:
        return 0.0, 1.0
    need = float(log_mixture(k, n)) - math.log(1.0 / alpha)       # the interval: g(θ) > need

    def g(th):
        a = k * math.log(th) if k else 0.0
        b = (n - k) * math.log(1.0 - th) if n - k else 0.0
        return a + b
    mode = min(max(k / n, 1e-12), 1.0 - 1e-12)

    def edge(inside, outside):
        if g(outside) > need:
            return outside
        for _ in range(80):
            mid = 0.5 * (inside + outside)
            if g(mid) > need:
                inside = mid
            else:
                outside = mid
        return outside
    lo = 0.0 if k == 0 else edge(mode, 1e-300)
    hi = 1.0 if k == n else edge(mode, 1.0 - 1e-16)
    return float(lo), float(hi)


def analyse(fail_a, fail_b, value_a, value_b, alpha: float = 0.05, margin: float = 0.05,
            non_inferiority: float | None = None, primary: str = 'failure', min_trials: int = 1) -> dict:
    """The gate's analysis of n finished pairs (module doc): {'verdict', 'decided_at' (the trial at which the first
    verdict held, None for cant_tell), 'n', 'discordant': {b_only, a_only, n, theta, interval, log_e}, 'risk':
    {estimate, lower, upper}, 'value': {estimate, lower, upper}, 'path': the bounds after every trial, …}. The
    verdict is the first that held along the trials — looking after every trial — not only the last one's;
    'decided_by' says which bound gave it: 'discordant' (θ's interval), 'risk' or 'value'."""
    if primary not in PRIMARY:
        raise GateError(f'primary: one of {", ".join(PRIMARY)}')
    fa, fb = np.asarray(fail_a, dtype=bool), np.asarray(fail_b, dtype=bool)
    va, vb = np.asarray(value_a, dtype=float), np.asarray(value_b, dtype=float)
    n = len(fa)
    a_d = alpha / 2.0 if primary == 'failure' else alpha            # secondary analyses stand at α on their own
    a_r = alpha / 2.0 if primary == 'failure' else alpha
    d = fb.astype(float) - fa.astype(float)
    x = vb - va
    r_lo, r_hi = eb_sequence(d, a_r)
    v_lo, v_hi = eb_sequence(x, alpha)
    b_only, a_only = np.cumsum(fb & ~fa), np.cumsum(fa & ~fb)
    nd = b_only + a_only
    th_lo, th_hi = np.zeros(n), np.ones(n)
    cur = (0.0, 1.0)
    for i in range(n):                                               # the interval moves only at a discordant pair
        if i == 0 or nd[i] != nd[i - 1]:
            lo, hi = discordant_interval(int(b_only[i]), int(nd[i]), a_d)
            cur = (max(cur[0], lo), min(cur[1], hi))
        th_lo[i], th_hi[i] = cur
    lo_p, hi_p = (r_lo, r_hi) if primary == 'failure' else (v_lo, v_hi)
    verdict, at, by = 'cant_tell', None, None
    bound = 'risk' if primary == 'failure' else 'value'
    for i in range(max(int(min_trials), 1) - 1, n):
        worse = lo_p[i] > 0 or (primary == 'failure' and th_lo[i] > 0.5)
        better = hi_p[i] < 0 or (primary == 'failure' and th_hi[i] < 0.5)
        if worse:
            verdict, by = 'worse', bound if lo_p[i] > 0 else 'discordant'
        elif better:
            verdict, by = 'better', bound if hi_p[i] < 0 else 'discordant'
        elif -margin < lo_p[i] and hi_p[i] < margin:
            verdict, by = 'equivalent', bound
        elif non_inferiority is not None and hi_p[i] < non_inferiority:
            verdict, by = 'non_inferior', bound
        if verdict != 'cant_tell':
            at = i + 1
            break
    k = (at or n) - 1

    def last(arr, default):
        return float(arr[k]) if n else default
    out = {'verdict': verdict, 'word': WORDS[verdict], 'decided_at': at, 'decided_by': by, 'n': n, 'alpha': alpha,
           'primary': primary,
           'margin': margin, 'non_inferiority': non_inferiority,
           'within_margin': bool(n and -margin < lo_p[k] and hi_p[k] < margin),
           'failures': {'a': int(fa[:k + 1].sum()) if n else 0, 'b': int(fb[:k + 1].sum()) if n else 0},
           'discordant': {'b_only': int(b_only[k]) if n else 0, 'a_only': int(a_only[k]) if n else 0,
                          'n': int(nd[k]) if n else 0,
                          'theta': (float(b_only[k] / nd[k]) if n and nd[k] else None),
                          'interval': [last(th_lo, 0.0), last(th_hi, 1.0)], 'alpha': a_d,
                          'log_e': float(discordant_log_e(b_only[k], nd[k])) if n else 0.0,
                          'test': 'mixture SPRT of θ = 1/2, uniform prior (Robbins 1970)'},
           'risk': {'estimate': float(d[:k + 1].mean()) if n else None, 'lower': last(r_lo, -1.0),
                    'upper': last(r_hi, 1.0), 'alpha': a_r,
                    'about': 'P(B fails) − P(A fails); empirical-Bernstein confidence sequence'},
           'value': {'estimate': float(x[:k + 1].mean()) if n else None, 'lower': last(v_lo, -1.0),
                     'upper': last(v_hi, 1.0), 'alpha': alpha,
                     'about': 'mean of value(B) − value(A), higher = B more critical; empirical-Bernstein '
                              'confidence sequence'},
           'path': {'trial': list(range(1, n + 1)),
                    'risk_lower': [round(float(v), 5) for v in r_lo], 'risk_upper': [round(float(v), 5) for v in r_hi],
                    'value_lower': [round(float(v), 5) for v in v_lo],
                    'value_upper': [round(float(v), 5) for v in v_hi],
                    'theta_lower': [round(float(v), 5) for v in th_lo],
                    'theta_upper': [round(float(v), 5) for v in th_hi]}}
    return out


def simulate_verdicts(p_a: float, p_b: float, n: int, reps: int, alpha: float = 0.05, margin: float = 0.05,
                      seed: int = 0, rho: float = 0.0) -> dict:
    """Monte Carlo of the failure-primary gate with a look after every trial: `reps` gates of up to n pairs whose
    failures are Bernoulli(p_a), Bernoulli(p_b), sharing a common cause with probability rho (the pair then fails or
    passes together). Returns the share of each verdict at the first decision. Vectorised; used to check the
    type-I error (p_a = p_b) and to plan a budget (falcon gate plan)."""
    rng = np.random.default_rng(seed)
    u = rng.random((reps, n))
    same = rng.random((reps, n)) < rho
    ub = np.where(same, u, rng.random((reps, n)))
    fa, fb = u < p_a, ub < p_b
    d = fb.astype(float) - fa.astype(float)
    lo, hi = eb_sequence(d, alpha / 2.0)
    b_only, a_only = np.cumsum(fb & ~fa, axis=1), np.cumsum(fa & ~fb, axis=1)
    nd = b_only + a_only
    reject = discordant_log_e(b_only, nd) >= math.log(2.0 / alpha)
    worse = (lo > 0) | (reject & (2 * b_only > nd))
    better = (hi < 0) | (reject & (2 * b_only < nd))
    equiv = (-margin < lo) & (hi < margin)
    first = np.full(reps, n + 1)
    verdict = np.full(reps, 'cant_tell', dtype=object)
    for name, m in (('equivalent', equiv), ('better', better), ('worse', worse)):   # worse wins a tie at one trial
        hit = m.any(axis=1)
        at = np.where(hit, m.argmax(axis=1) + 1, n + 1)
        take = at <= first
        first = np.where(take, at, first)
        verdict = np.where(take & hit, name, verdict)
    out = {v: float(np.mean(verdict == v)) for v in ('better', 'worse', 'equivalent', 'cant_tell')}
    decided = first[first <= n]
    out.update(reps=reps, n=n, median_trials=float(np.median(decided)) if len(decided) else None)
    return out


# ================================================================================================ the versions
def parse_version(spec: str, base: str = '') -> dict:
    """A version from its command-line spelling (module doc): {'kind': 'stack', 'stack': {...}, 'label'} or
    {'kind': 'model', 'model': ID, 'label'}."""
    spec = str(spec or '').strip()
    if not spec:
        raise GateError('a version is follower, autopilot, stack:SUITE.json (or the file\'s path) or model:ID')
    if spec.startswith('model:'):
        if not spec[6:]:
            raise GateError('model: needs a registered model\'s id')
        return {'kind': 'model', 'model': spec[6:], 'label': spec}
    if spec in ('follower', 'autopilot'):
        return {'kind': 'stack', 'stack': {'driver': spec}, 'label': spec}
    path = spec[6:] if spec.startswith('stack:') else spec
    path = path if os.path.isabs(path) else os.path.join(base or os.getcwd(), path)
    if not os.path.isfile(path):
        raise GateError(f'version {spec!r}: not follower, autopilot or model:ID, and no suite file {path}')
    try:
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise GateError(f'version {spec!r}: {e}') from None
    st = doc.get('stack') if isinstance(doc, dict) else None
    if not isinstance(st, dict) or st.get('driver') not in ('ros2', 'follower', 'autopilot'):
        raise GateError(f'version {spec!r}: the file has no "stack" block with a driver (a falcon.suite/1 suite)')
    if st['driver'] == 'ros2' and not st.get('launch'):
        raise GateError(f'version {spec!r}: a ROS 2 stack needs its launch commands (stack.launch)')
    return {'kind': 'stack', 'stack': st, 'label': doc.get('name') or os.path.basename(path), 'suite': path}


def trial_plan(scenarios: list, seed: int, i: int) -> dict:
    """Trial i of the gate (module doc): {'i', 'scenario' (its document), 'seed', 'values'}."""
    from . import scenarios as SC
    scn = scenarios[i % len(scenarios)]
    s = int(seed) * 1000003 + i
    return {'i': i, 'scenario': scn, 'seed': s, 'values': SC.sample_params(scn, s)}


class StackSide:
    """One version that drives: each trial is a stack run (core.stack.stack_config, sim.stacksuite.run_one) in
    <gate>/runs/<side>/<key[:16]>/, read back when it is already there."""

    def __init__(self, svc, project, version: dict, gate_dir: str, side: str, world: str, rig: str, fps: float,
                 motion: str, failure: str, threshold: float):
        self.svc, self.project, self.v, self.dir, self.side = svc, project, version, gate_dir, side
        self.world, self.rig, self.fps, self.motion = world, rig, float(fps), motion
        self.failure, self.threshold = failure, float(threshold)

    def run(self, trials: list) -> list:
        from ..sim.stacksuite import run_one
        from . import stack as ST
        from .provenance import read_json
        st = self.v['stack']
        out = []
        for tr in trials:
            scn = tr['scenario']
            cfg = ST.stack_config(self.project, self.world, self.rig, '/pending', self.svc.port, self.svc.tm_port,
                                  st['driver'], scn, seed=tr['seed'], fps=self.fps, motion=self.motion,
                                  bridge=st.get('bridge'),
                                  render_budget_mpx=float(self.svc.settings.get('render_budget_mpx')),
                                  workspace=self.project.workspace,
                                  name=f'gate {self.side.upper()}: {scn["name"]} (seed {tr["seed"]})')
            key = ST.run_key(cfg)
            cfg['out'] = os.path.join(self.dir, 'runs', self.side, key[:16])
            cfg['cache_key'] = key
            run = None
            try:
                run = read_json(os.path.join(cfg['out'], 'run.json'))
            except (OSError, ValueError):
                run = None
            if not run or run.get('status') != 'ok':
                launch = (st.get('launch') or []) if st['driver'] == 'ros2' else []
                run = run_one(self.svc, cfg, launch, int(st.get('domain_id', 0)), cfg['name'])
                if run.get('_job_state') == 'cancelled':
                    raise KeyboardInterrupt
            row = ST.result_row({'key': key, 'entry': 0, 'scenario': scn.get('id'), 'scenario_name': scn.get('name'),
                                 'seed': tr['seed'], 'dir': cfg['out']}, run)
            if row.get('status') != 'ok' or row.get('criticality') is None:
                out.append({'ok': False, 'run': cfg['out'], 'error': row.get('error') or 'the run did not finish'})
                continue
            value = float(row['criticality'])
            failed = (value >= self.threshold) if self.failure == 'value' else (row.get('passed') is False)
            out.append({'ok': True, 'run': cfg['out'], 'failed': bool(failed), 'value': min(max(value, 0.0), 1.0),
                        'passed': row.get('passed'), 'collision': row.get('collision'),
                        'min_ttc_s': row.get('min_ttc_s')})
        return out


class ModelSide:
    """One detector version: the trial's scenario is recorded once (a sim.studio.Study whose drives both versions
    share: the cache key does not hold the model) and scored by this model (core.milsearch's perception failure);
    failed: the value reached the threshold. A model is told apart by its fingerprint (core.loop.model_fingerprint:
    its file's SHA-256; for a plug-in its registration without the params) — main_gate refuses two versions with the
    same one."""

    def __init__(self, svc, project, version: dict, gate_dir: str, name: str, scenarios: list, world: str,
                 rig: str, fps: float, seed: int, threshold: float):
        from ..sim.studio import Study
        m = project.item('models', version['model'])
        if m is None:
            raise GateError(f'no model {version["model"]!r} in the project')
        self.threshold = float(threshold)
        self.studies = {}
        for scn in scenarios:
            try:
                self.studies[scn['id']] = Study(svc, project, scn, f'gate-{name}-{scn["id"]}', world, rig, seed,
                                                objective={'objective': 'perception', 'model': m,
                                                           'feasibility': False}, fps=fps)
            except ValueError as e:
                raise GateError(str(e)) from None

    def run(self, trials: list) -> list:
        out = [None] * len(trials)
        for sid, st in self.studies.items():
            idx = [k for k, tr in enumerate(trials) if tr['scenario']['id'] == sid]
            if not idx:
                continue
            try:
                rows = st.evaluate([trials[k]['values'] for k in idx])
            except RuntimeError as e:
                raise GateError(str(e)) from None
            for k, r in zip(idx, rows):
                if not r or r.get('status') != 'ok':
                    out[k] = {'ok': False, 'run': (r or {}).get('run'), 'error': (r or {}).get('error') or
                              'the run did not finish'}
                else:
                    v = float(r['objective'])
                    out[k] = {'ok': True, 'run': r['run'], 'failed': v >= self.threshold,
                              'value': min(max(v, 0.0), 1.0)}
        return out


# ================================================================================================ the gate
def gate_dir(workspace: str, name: str) -> str:
    return os.path.join(workspace, 'gates', name)


def read_gate(path: str) -> dict | None:
    """gate.json of a gate folder (or the file itself); None when there is none."""
    p = os.path.join(path, 'gate.json') if os.path.isdir(path) else path
    try:
        with open(p, encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get('schema') == GATE_SCHEMA else None


def read_trials(d: str) -> list:
    """The finished pairs of a gate folder, by trial index (the last line per index wins)."""
    rows = {}
    try:
        with open(os.path.join(d, 'trials.jsonl'), encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows[int(r['i'])] = r
    except OSError:
        pass
    return [rows[k] for k in sorted(rows)]


def changes(trials: list, upto: int | None = None) -> tuple:
    """(newly failing, newly passing) among the first `upto` pairs: the scenarios B fails that A passed and the
    other way round, each {'i', 'scenario', 'seed', 'values', 'a': run, 'b': run, 'value_a', 'value_b'}."""
    new_fail, new_pass = [], []
    for t in trials[:upto]:
        a, b = t['a'], t['b']
        if a['failed'] == b['failed']:
            continue
        row = {'i': t['i'], 'scenario': t['scenario'], 'seed': t['seed'], 'values': t['values'], 'a': a['run'],
               'b': b['run'], 'value_a': a['value'], 'value_b': b['value']}
        (new_fail if b['failed'] else new_pass).append(row)
    return new_fail, new_pass


class Gate:
    """A gate over two sides (objects with run(trials) -> rows): batches of paired trials until a verdict or the
    budget (module doc). Pure given the sides: the tests drive it with planted ones."""

    def __init__(self, out_dir: str, side_a, side_b, scenarios: list, seed: int = 0, alpha: float = 0.05,
                 margin: float = 0.05, non_inferiority: float | None = None, primary: str = 'failure',
                 budget: int = 200, batch: int = 8, min_trials: int = 1, meta: dict | None = None):
        if not scenarios:
            raise GateError('a gate needs at least one scenario')
        if not 0.0 < alpha < 1.0:
            raise GateError('alpha must be in (0, 1)')
        if margin <= 0:
            raise GateError('the equivalence margin must be positive')
        if primary not in PRIMARY:
            raise GateError(f'primary: one of {", ".join(PRIMARY)}')
        self.dir, self.a, self.b, self.scenarios = out_dir, side_a, side_b, scenarios
        self.seed, self.alpha, self.margin, self.ni, self.primary = int(seed), float(alpha), float(margin), \
            non_inferiority, primary
        self.budget, self.batch, self.min_trials = int(budget), max(int(batch), 1), int(min_trials)
        self.meta = dict(meta or {})
        os.makedirs(self.dir, exist_ok=True)

    def design(self) -> dict:
        return {'scenarios': [s.get('id') for s in self.scenarios], 'seed': self.seed, 'alpha': self.alpha,
                'margin': self.margin, 'non_inferiority': self.ni, 'primary': self.primary}

    def result(self, trials: list, state: str, skipped: list | None = None) -> dict:
        from .provenance import utc_now
        an = analyse([t['a']['failed'] for t in trials], [t['b']['failed'] for t in trials],
                     [t['a']['value'] for t in trials], [t['b']['value'] for t in trials], self.alpha, self.margin,
                     self.ni, self.primary, self.min_trials)
        used = an['decided_at'] or len(trials)
        nf, np_ = changes(trials, used)
        doc = {'schema': GATE_SCHEMA, 'state': state, 'updated': utc_now(), 'design': self.design(),
               'budget': self.budget, 'batch': self.batch, 'trials_run': len(trials), 'trials_used': used,
               'verdict': an['verdict'], 'word': an['word'],
               'newly_failing': nf, 'newly_passing': np_, 'skipped': skipped or [],
               'scope': 'holds for the tested scenario distribution only: ' +
                        ', '.join(str(s.get('id')) for s in self.scenarios) + ' with their declared parameter '
                        'distributions, this world, rig and motion',
               **{k: an[k] for k in ('decided_at', 'decided_by', 'n', 'alpha', 'primary', 'margin', 'non_inferiority',
                                     'within_margin', 'failures', 'discordant', 'risk', 'value', 'path')}}
        doc.update(self.meta)
        return doc

    def write(self, doc: dict):
        from .provenance import write_json
        write_json(os.path.join(self.dir, 'gate.json'), doc)
        with open(os.path.join(self.dir, 'gate.md'), 'w', encoding='utf-8') as fh:
            fh.write(markdown(doc))

    def run(self, progress=None, log=None) -> dict:
        log = log or (lambda text: None)
        prev = read_gate(self.dir)
        trials = read_trials(self.dir)
        if prev and prev.get('design') != self.design() and trials:
            raise GateError(f'{self.dir} holds a gate with another design (scenarios, seed, alpha, margin or '
                            'primary outcome); give another --name, or remove it')
        skipped = list((prev or {}).get('skipped') or [])
        have = {t['i'] for t in trials} | {s['i'] for s in skipped}
        doc = self.result(trials, 'running', skipped)
        nxt = (max(have) + 1) if have else 0
        while doc['verdict'] == 'cant_tell' and len(trials) < self.budget:
            idx = list(range(nxt, nxt + min(self.batch, self.budget - len(trials))))
            nxt = idx[-1] + 1
            plan = [trial_plan(self.scenarios, self.seed, i) for i in idx]
            ra = self.a.run(plan)
            rb = self.b.run(plan)
            got = 0
            with open(os.path.join(self.dir, 'trials.jsonl'), 'a', encoding='utf-8') as fh:
                for tr, a, b in zip(plan, ra, rb):
                    if not (a and b and a.get('ok') and b.get('ok')):
                        skipped.append({'i': tr['i'], 'scenario': tr['scenario'].get('id'), 'seed': tr['seed'],
                                        'why': '; '.join(f'{s}: {r.get("error")}' for s, r in (('A', a or {}),
                                                                                                ('B', b or {}))
                                                         if not r.get('ok'))})
                        continue
                    row = {'i': tr['i'], 'scenario': tr['scenario'].get('id'), 'seed': tr['seed'],
                           'values': tr['values'], 'a': {k: v for k, v in a.items() if k != 'ok'},
                           'b': {k: v for k, v in b.items() if k != 'ok'}}
                    fh.write(json.dumps(row) + '\n')
                    trials.append(row)
                    got += 1
            doc = self.result(trials, 'running', skipped)
            self.write(doc)
            log(f'{len(trials)} pairs: {doc["failures"]["a"]} failed under A, {doc["failures"]["b"]} under B, '
                f'{doc["discordant"]["n"]} discordant; Δ in [{doc["risk"]["lower"]:+.3f}, '
                f'{doc["risk"]["upper"]:+.3f}]: {doc["word"]}')
            if progress:
                progress(len(trials), self.budget)
            if not got and len(skipped) >= 2 * self.batch and not trials:
                raise GateError('no pair finished in the first batches: ' + skipped[-1]['why'])
            if len(skipped) > max(self.budget, 4 * self.batch):
                raise GateError(f'{len(skipped)} pairs did not finish: ' + skipped[-1]['why'])
        doc = self.result(trials, 'finished', skipped)
        self.write(doc)
        return doc


def headline(d: dict) -> str:
    r, v, dc = d['risk'], d['value'], d['discordant']
    a, b = (d.get('versions') or {}).get('a', {}).get('label', 'A'), (d.get('versions') or {}).get('b', {}).get(
        'label', 'B')
    lines = [f'B ({b}) against A ({a}): {d["word"].upper()}' +
             (f' after {d["decided_at"]} paired trials' if d.get('decided_at') else
              f' at the budget of {d["budget"]} paired trials ({d["trials_run"]} run)') +
             (' (the difference is inside the margin)' if d['verdict'] in ('better', 'worse') and d['within_margin']
              else '') + (', shown by the discordant pairs' if d.get('decided_by') == 'discordant' else ''),
             f'failures: A {d["failures"]["a"]}, B {d["failures"]["b"]} of {d["trials_used"]}; discordant pairs '
             f'{dc["n"]} ({dc["b_only"]} B only, {dc["a_only"]} A only), θ in [{dc["interval"][0]:.3f}, '
             f'{dc["interval"][1]:.3f}]',
             f'risk difference P(B fails) − P(A fails) in [{r["lower"]:+.3f}, {r["upper"]:+.3f}] (margin '
             f'±{d["margin"]:g}); value difference in [{v["lower"]:+.3f}, {v["upper"]:+.3f}]; α = {d["alpha"]:g}, '
             f'valid at every look',
             f'newly failing under B: {len(d["newly_failing"])}; newly passing: {len(d["newly_passing"])}',
             d['scope']]
    return '\n'.join(lines)


def markdown(d: dict, max_rows: int = 40) -> str:
    """gate.md: the verdict, the bounds over the trials and the newly failing / passing scenarios with their run
    pairs (a report section; also what CI's summary takes)."""
    vs = d.get('versions') or {}
    out = [f'## Release gate: {d["word"]}', '',
           f'B `{(vs.get("b") or {}).get("label", "B")}` against A `{(vs.get("a") or {}).get("label", "A")}`, '
           f'{d["trials_used"]} paired trials used ({d["trials_run"]} run, budget {d["budget"]}); primary outcome: '
           f'{d["primary"]}; α = {d["alpha"]:g}; margin ±{d["margin"]:g}' +
           (f'; non-inferiority margin {d["non_inferiority"]:g}' if d.get('non_inferiority') is not None else '') +
           '.', '',
           '| quantity | estimate | lower | upper |', '|---|---|---|---|']
    dc, r, v = d['discordant'], d['risk'], d['value']

    def f(x):
        return '—' if x is None else f'{x:+.4f}'
    out.append(f'| risk difference P(B fails) − P(A fails) | {f(r["estimate"])} | {f(r["lower"])} | {f(r["upper"])} |')
    out.append(f'| value difference (B − A) | {f(v["estimate"])} | {f(v["lower"])} | {f(v["upper"])} |')
    out.append(f'| θ = P(B alone fails \\| discordant), {dc["n"]} discordant pairs | '
               + ('—' if dc['theta'] is None else f'{dc["theta"]:.4f}') +
               f' | {dc["interval"][0]:.4f} | {dc["interval"][1]:.4f} |')
    out += ['', 'The bounds are confidence sequences: they hold at every number of trials at once, so stopping '
            'early is allowed. ' + d['scope'] + '.', '']
    p = d['path']
    if p['trial']:
        out += ['### Bounds over the trials', '', '| trials | risk lower | risk upper | value lower | value upper |',
                '|---|---|---|---|---|']
        n = len(p['trial'])
        marks = sorted({min(n, k) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, n)})
        for k in marks:
            out.append(f'| {k} | {p["risk_lower"][k - 1]:+.3f} | {p["risk_upper"][k - 1]:+.3f} | '
                       f'{p["value_lower"][k - 1]:+.3f} | {p["value_upper"][k - 1]:+.3f} |')
        out.append('')
    for title, rows in (('Newly failing under B', d['newly_failing']), ('Newly passing under B', d['newly_passing'])):
        out += [f'### {title} ({len(rows)})', '']
        if rows:
            out += ['| trial | scenario | seed | value A → B | run A | run B |', '|---|---|---|---|---|---|']
            for x in rows[:max_rows]:
                out.append(f'| {x["i"]} | {x["scenario"]} | {x["seed"]} | {x["value_a"]:.3f} → {x["value_b"]:.3f} | '
                           f'`{x["a"]}` | `{x["b"]}` |')
            if len(rows) > max_rows:
                out.append(f'| … | {len(rows) - max_rows} more in gate.json | | | | |')
        else:
            out.append('None.')
        out.append('')
    if d.get('skipped'):
        out += [f'{len(d["skipped"])} pair(s) did not finish and are not trials (first: {d["skipped"][0]["why"]}).',
                '']
    return '\n'.join(out)


def ci_gate(spec: dict, base: str = '') -> dict:
    """CI mode's `gates.release` (core.ci): {'path': a gate folder or gate.json, 'fail_on': ['worse', …]} -> the
    fields of a CI gate: {'status': 'passed' | 'failed' | 'error', 'measured', 'limit', 'detail'}."""
    p = spec.get('path') or ''
    p = p if os.path.isabs(p) else os.path.join(base, p)
    fail_on = list(spec.get('fail_on') or ['worse'])
    d = read_gate(p)
    limit = 'not ' + ' / '.join(WORDS.get(v, v) for v in fail_on)
    if d is None:
        return {'status': 'error', 'measured': '', 'limit': limit, 'detail': f'no gate result at {p} (falcon gate '
                                                                             'run … before this CI run)'}
    if d.get('state') != 'finished':
        return {'status': 'error', 'measured': d.get('word', ''), 'limit': limit,
                'detail': 'the gate has not finished'}
    r = d['risk']
    return {'status': 'failed' if d['verdict'] in fail_on else 'passed', 'measured': d['word'], 'limit': limit,
            'detail': f'{d["trials_used"]} paired trials; risk difference in [{r["lower"]:+.3f}, {r["upper"]:+.3f}]; '
                      f'{len(d["newly_failing"])} newly failing'}


# ================================================================================================ the command
def gate_argv(project_path: str, name: str, a: str, b: str, scenarios: list, budget: int = 200, batch: int = 8,
              alpha: float = 0.05, margin: float = 0.05, non_inferiority: float | None = None,
              primary: str = 'failure', seed: int = 0, world: str = '', rig: str = '', fake_carla: str = '',
              on: str = 'local', failure: str = 'criteria', threshold: float = 0.5) -> list:
    """`falcon gate run …` as Reports ▸ Release gate queues it (the arguments after `python -m falcon_autonomy`)."""
    argv = ['gate', 'run', project_path, '--name', name, '--a', a, '--b', b]
    for s in scenarios:
        argv += ['--scenario', s]
    argv += ['--budget', str(int(budget)), '--batch', str(int(batch)), '--alpha', str(alpha), '--margin', str(margin),
             '--primary', primary, '--seed', str(int(seed)), '--failure', failure, '--threshold', str(threshold)]
    if non_inferiority is not None:
        argv += ['--non-inferiority', str(non_inferiority)]
    if world:
        argv += ['--world', world]
    if rig:
        argv += ['--rig', rig]
    if on and on != 'local':
        argv += ['--on', on]
    return argv + (['--fake-carla', fake_carla] if fake_carla else [])


def main_gate(argv) -> int:
    """falcon gate run PROJECT --name N --a VERSION --b VERSION --scenario KEY… […] | show PROJECT --name N [--json] |
    plan --p-a P --p-b P [--trials N] […]   (exit codes as the CLI's: 0 when the gate ran, whatever its verdict —
    it is in gate.json; with --gate, as a test runner: 1 when the verdict is one of --fail-on, default worse)."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon gate', description='The release gate between two versions: paired '
                                 'trials with common random numbers and anytime-valid confidence '
                                 '(docs/release-gate.md).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run the gate (resumable): batches of paired trials until a verdict or the budget')
    r.add_argument('project')
    r.add_argument('--name', required=True, help='the gate\'s folder under the workspace\'s gates/')
    r.add_argument('--a', required=True, help='the old version: follower, autopilot, stack:SUITE.json or model:ID')
    r.add_argument('--b', required=True, help='the new version (the same kind as --a)')
    r.add_argument('--scenario', action='append', required=True, help='a scenario of the tested distribution '
                                                                       '(repeat; taken in turn)')
    r.add_argument('--world', default='')
    r.add_argument('--rig', default='')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--budget', type=int, default=200, help='paired trials at most (each is two runs, or one run '
                                                           'scored twice for models)')
    r.add_argument('--batch', type=int, default=8, help='paired trials between two looks')
    r.add_argument('--alpha', type=float, default=0.05, help='the total error probability (default 0.05: 5 %%)')
    r.add_argument('--margin', type=float, default=0.05, help='the equivalence margin on the primary difference')
    r.add_argument('--non-inferiority', type=float, default=None, help='also stop when B is shown not worse by '
                                                                      'more than this')
    r.add_argument('--primary', default='failure', choices=PRIMARY, help='the outcome the verdict is read from')
    r.add_argument('--failure', default='criteria', choices=FAILURE, help='a stack run failed: its scenario\'s pass '
                   'criteria were not met (default), or its criticality reached --threshold')
    r.add_argument('--threshold', type=float, default=0.5)
    r.add_argument('--min-trials', type=int, default=1, help='no verdict before this many trials')
    r.add_argument('--fps', type=float, default=20.0)
    r.add_argument('--motion', default='kinematic', choices=('kinematic', 'physics'))
    r.add_argument('--gate', action='store_true', help='exit 1 when the verdict is one of --fail-on (for CI)')
    r.add_argument('--fail-on', default='worse', help='with --gate: the verdicts that fail, comma-separated '
                                                      '(worse, cant_tell; default worse)')
    r.add_argument('--fake-carla', default='')
    from ..sim.service import placement_arg
    placement_arg(r)
    s = sub.add_parser('show', help='a gate\'s result')
    s.add_argument('project')
    s.add_argument('--name', required=True)
    s.add_argument('--json', action='store_true')
    s.add_argument('--gate', action='store_true', help='exit 1 when the verdict is one of --fail-on')
    s.add_argument('--fail-on', default='worse')
    pl = sub.add_parser('plan', help='what a budget can show: the verdicts of simulated gates with given failure '
                                     'probabilities (no runs)')
    pl.add_argument('--p-a', type=float, required=True)
    pl.add_argument('--p-b', type=float, required=True)
    pl.add_argument('--trials', type=int, default=200)
    pl.add_argument('--reps', type=int, default=2000)
    pl.add_argument('--alpha', type=float, default=0.05)
    pl.add_argument('--margin', type=float, default=0.05)
    pl.add_argument('--common', type=float, default=0.0, help='the share of pairs that fail or pass together')
    a = ap.parse_args(argv)

    def code(d):
        bad = [v.strip() for v in a.fail_on.split(',') if v.strip()]
        return 1 if a.gate and d.get('state') == 'finished' and d['verdict'] in bad else P.OK

    if a.cmd == 'plan':
        print(json.dumps(simulate_verdicts(a.p_a, a.p_b, a.trials, a.reps, a.alpha, a.margin, rho=a.common),
                         indent=2))
        return P.OK
    from .project import Project, ProjectError
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    out = gate_dir(proj.workspace, a.name)
    if a.cmd == 'show':
        d = read_gate(out)
        if d is None:
            print(f'failed: no gate result in gates/{a.name} (falcon gate run …)', flush=True)
            return P.FAILED
        print(json.dumps(d, indent=2) if a.json else headline(d))
        return code(d)

    from . import scenarios as SC
    scns = []
    for key in a.scenario:
        scn = proj.item('scenarios', key) or (SC.builtin(key) if key in SC.LIBRARY else None)
        if scn is None:
            print(f'failed: no scenario {key!r}', flush=True)
            return P.FAILED
        scns.append(scn)
    base = os.path.dirname(os.path.abspath(proj.path)) if proj.path else ''
    try:
        va, vb = parse_version(a.a, base), parse_version(a.b, base)
        if va['kind'] != vb['kind']:
            raise GateError('both versions must be stacks (follower, autopilot, a suite file) or both models')
    except GateError as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    world = a.world or proj.doc['worlds'][0]['id']
    rig = a.rig or proj.doc['rigs'][0]['id']
    from ..cluster.bundle import placement_refusal
    from ..sim.service import study_service
    for scn in scns:
        why = placement_refusal(a.on, scn, proj.item('worlds', world))
        if why:
            print(f'failed: {why}', flush=True)
            return P.REFUSED
    try:
        svc = study_service(None, proj.workspace, fake_carla=a.fake_carla, placement=a.on)
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    em = P.Emitter()
    em.hello('gate')
    t0 = time.time()
    try:
        if va['kind'] == 'stack':
            sa = StackSide(svc, proj, va, out, 'a', world, rig, a.fps, a.motion, a.failure, a.threshold)
            sb = StackSide(svc, proj, vb, out, 'b', world, rig, a.fps, a.motion, a.failure, a.threshold)
        else:
            sa = ModelSide(svc, proj, va, out, a.name, scns, world, rig, a.fps, a.seed, a.threshold)
            sb = ModelSide(svc, proj, vb, out, a.name, scns, world, rig, a.fps, a.seed, a.threshold)
            if any(sa.studies[k].okey == sb.studies[k].okey for k in sa.studies):
                raise GateError(f'{va["label"]} and {vb["label"]} have the same fingerprint (the same model file, or '
                                'two plug-in registrations that differ only in their params): their scores cannot be '
                                'told apart, so there is nothing to compare')
        meta = {'name': a.name, 'versions': {'a': {k: va.get(k) for k in ('kind', 'label', 'model', 'suite')},
                                             'b': {k: vb.get(k) for k in ('kind', 'label', 'model', 'suite')}},
                'world': world, 'rig': rig, 'fps': a.fps, 'motion': a.motion, 'where': a.on,
                'failure': a.failure if va['kind'] == 'stack' else 'value', 'threshold': a.threshold}
        g = Gate(out, sa, sb, scns, a.seed, a.alpha, a.margin, a.non_inferiority, a.primary, a.budget, a.batch,
                 a.min_trials, meta)
        d = g.run(lambda n, total: em.progress(min(n / max(total, 1), 1.0), f'{n} of at most {total} pairs',
                                               force=True), em.log)
    except (GateError, ValueError, RuntimeError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.FAILED
    except KeyboardInterrupt:
        svc.shutdown()
        return P.CANCELLED
    svc.shutdown()
    for line in headline(d).splitlines():
        em.log(line)
    em.result('ok', summary={'verdict': d['verdict'], 'word': d['word'], 'trials': d['trials_used'],
                             'newly_failing': len(d['newly_failing']), 'seconds': round(time.time() - t0, 1)})
    return code(d)
