"""Failure rate with a confidence interval: adaptive importance sampling over a scenario's parameter space (standard
library + numpy). docs/failure-rate.md is the manual.

A rare-case search (core.search, core.milsearch) finds where a scenario fails; it cannot say how often it fails,
because it samples where failures are likely, not as the parameters occur. This module estimates

    P_fail = P_p(the run fails),   x ~ p (the NOMINAL distribution of the scenario's parameters),

the probability that one ENCOUNTER (one run of the scenario: its actors, its timeline, its world and weather, the
parameters drawn from p) fails, with a 95 % confidence interval — by importance sampling with a proposal adapted by the
cross-entropy method, so a failure probability of 1e-5 needs hundreds or a few thousand runs instead of millions.

The nominal distribution p (Nominal). Each searched parameter's own distribution as the scenario declares it
(core.scenarios: uniform on [lo, hi], normal (truncated to [lo, hi] when given), a choice uniform over its choices),
narrowed to the kinematically feasible part when the study narrows it (core.rss.feasible_space; the estimate is then
conditional on the feasible part, whose nominal mass is reported), and a searched weather field ("weather.<field>",
core.milsearch) uniform over its range. Parameters are independent. ASSUMPTION, the user's to check: these
distributions are how the parameters occur in the operating domain. The scenario's declared ranges are a test
designer's ranges, not measured frequencies; the estimate is exactly as good as they are.

Every parameter is written as a transform of one standard normal z_j (x_j = F_j⁻¹(Φ(z_j)); a choice takes the cell
⌊Φ(z_j)·K⌋), so p is the standard normal N(0, I) in z and the estimator works in z.

The proposal family (documented, fixed): a DEFENSIVE MIXTURE (Hesterberg 1995, Weighted average importance sampling and
defensive mixture distributions, Technometrics 37)

    q(z) = β·N(z; 0, I) + (1 − β)·N(z; μ, I),   β = `defensive` (default 0.1),

the nominal itself beside the MEAN-SHIFT family N(μ, I) (exponential tilting of the standard normal: the proposal
used for design-point importance sampling in structural reliability). The weights w = p/q =
1 / (β + (1 − β)·exp(μ·z − |μ|²/2)) are bounded by 1/β, so the estimator's variance is finite whatever μ is. A
full-covariance Gaussian was measured and is not offered: its fitted spread collapsed stage by stage (the elite of a
narrow proposal cannot show the target's spread), so it stalled below the threshold or gave intervals that missed the
truth (docs/failure-rate.md).

Adaptation: the multilevel cross-entropy method for rare events (Rubinstein 1997, Optimization of computer simulation
models with rare events, EJOR 99; Rubinstein & Kroese 2004, The Cross-Entropy Method, ch. 3; Gaussian proposals in
standard-normal space as in Geyer, Papaioannou & Straub 2019, Cross entropy-based importance sampling using Gaussian
densities revisited, Structural Safety 76). Stage 0 draws `stage_size` runs from p. After each stage the score s (the
search's objective, higher = worse) sets a level γ = min(t, the ⌈ρ·n⌉-th largest score) — when at least ⌈ρ·n⌉ runs
reach the failure threshold t, γ = t and the elite are the failures; else the elite are the ⌈ρ·n⌉ highest — and μ
is refitted to the elite by the likelihood-ratio-weighted maximum-likelihood update μ = Σ W z / Σ W (W = p/q of the
stage the run came from). Adaptation stops when γ = t, when it has not risen for two stages (stalled), or after
`max_stages`.

The estimate: `final_size` runs drawn from the final proposal (its Gaussian the last fit), independent given it.
With y_i = w_i·1{run i failed}:

    P̂ = mean(y),  SE = sd(y) / √n  (the sample standard deviation, n − 1),
    95 % interval: P̂·exp(∓1.96·SE/P̂)  (the delta method on log P̂: asymmetric, never negative),

the standard importance-sampling estimator and its variance (e.g. Owen 2013, Monte Carlo theory, methods and examples,
ch. 9). The adaptation stages' runs are not pooled into it (their proposals depended on earlier runs); they are
recorded. The failure counted is the JUDGE's: the search's own (the objective ≥ t; for the criticality objective by
default counting failures the ego could have avoided by RSS, core.rss.rss_objective), the scenario's pass criteria
(core.stack, as a suite judges), or a traffic-rule violation (core.rules: an STL formula with negative robustness,
core.stl). The proposal is always adapted on the objective's score; importance sampling stays unbiased for any judge,
because the defensive part keeps q > 0 wherever p > 0.

Trust. ESS_fail = (Σ_fail w)² / Σ_fail w² (Kish's effective sample size over the failing runs). REFUSED (no interval
given as an answer) when fewer than MIN_FAILURES failing runs or ESS_fail < MIN_ESS: with zero failures it gives
only the bound P_fail ≤ 3 / (β·n) at 95 % (the rule of three on the defensive part: P_q(fail) ≥ β·P_p(fail)); a
WARNING when ESS_fail < WARN_ESS, the relative standard error SE/P̂ > WARN_REL_SE, or one run carries more than
WARN_WEIGHT_SHARE of the failures' weight.

Exposure. The estimate is per encounter. "Failures per 10ⁿ km" (or per 10ⁿ hours) needs how many such encounters
happen per km (or hour) in the operating domain — a number Falcon cannot measure from simulation; the user gives it
with its source, it is taken as exact, and the rate is P̂ × encounters per km with the interval scaled alike.

Validation (validate(), `falcon failrate check`; docs/failure-rate.md has the measured table): on synthetic problems
with a known probability — a linear limit state in d standard normals, P = Φ(−c) — the interval's coverage over seeds
and the plain-Monte-Carlo runs that the same relative error would need, (1 − P) / (P·rel²).

Limits. One Gaussian component: a scenario with two separate failure regions is covered by the defensive part only
outside the one the Gaussian found (unbiased, but with heavy weights there — the warnings say so); a failure region
much narrower than the nominal in some direction is sampled with the nominal's spread there (correct, less
efficient); the delta-method
interval assumes the estimator is roughly normal, which needs enough effective failures (hence the refusal); a run
that fails to run (no run.json) is replaced by a fresh draw, which assumes such failures are unrelated to the
parameters; the encounter is the scenario's, with its fixed world, seed and traffic.
"""
from __future__ import annotations

import copy
import math
from statistics import NormalDist

import numpy as np

SCHEMA = 'falcon.failure_rate/1'
STATE_SCHEMA = 'falcon.failure_rate_state/1'
Z95 = 1.959963984540054
MIN_FAILURES = 10
MIN_ESS = 10.0
WARN_ESS = 30.0
WARN_REL_SE = 0.3
WARN_WEIGHT_SHARE = 0.2
JUDGES = ('objective', 'checks', 'rules')
DEFAULTS = {'stage_size': 100, 'final_size': 200, 'rho': 0.1, 'defensive': 0.1, 'max_stages': 6}
PROPOSAL = ('defensive mixture q(z) = β·N(z; 0, I) + (1 − β)·N(z; μ, I) in standard-normal coordinates (the '
            'mean-shift family), μ fitted by the multilevel cross-entropy method (Rubinstein 1997; Rubinstein & '
            'Kroese 2004)')
ESTIMATOR = ('importance sampling on the final stage only: P̂ = mean(w·1{fail}), w = p/q; SE = sd/√n; 95 % interval '
             'P̂·exp(∓1.96·SE/P̂) (delta method on log P̂)')
_N = NormalDist()


class FailRateError(Exception):
    pass


def phi_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _phi_vec(Z: np.ndarray) -> np.ndarray:
    return 0.5 * np.frompyfunc(math.erfc, 1, 1)(-np.asarray(Z, float) / math.sqrt(2.0)).astype(float)


# ============================================================================================ the nominal distribution
class Nominal:
    """The nominal distribution p of the parameters as transforms of independent standard normals (module doc).
    dims: [{'name', 'dist': 'uniform', 'lo', 'hi'} | {'name', 'dist': 'normal', 'mean', 'sd', ['lo', 'hi']}
           | {'name', 'dist': 'choice', 'choices'}]; 'mass' (optional): the share of the declared distribution kept
    (feasibility narrowing)."""

    def __init__(self, dims: list):
        self.dims = []
        for d in dims:
            d = dict(d)
            kind = d.get('dist')
            if kind == 'uniform':
                d['lo'], d['hi'] = float(d['lo']), float(d['hi'])
                if not d['hi'] > d['lo']:
                    raise FailRateError(f'{d["name"]}: hi must be above lo')
            elif kind == 'normal':
                d['mean'], d['sd'] = float(d['mean']), float(d['sd'])
                if not d['sd'] > 0:
                    raise FailRateError(f'{d["name"]}: sd must be positive')
                lo, hi = d.get('lo'), d.get('hi')
                d['lo'] = None if lo is None else float(lo)
                d['hi'] = None if hi is None else float(hi)
                a = 0.0 if d['lo'] is None else phi_cdf((d['lo'] - d['mean']) / d['sd'])
                b = 1.0 if d['hi'] is None else phi_cdf((d['hi'] - d['mean']) / d['sd'])
                if not b > a:
                    raise FailRateError(f'{d["name"]}: the truncation leaves no mass')
                d['_ab'] = (a, b)
            elif kind == 'choice':
                d['choices'] = list(d['choices'])
                if not d['choices']:
                    raise FailRateError(f'{d["name"]}: no choices')
            else:
                raise FailRateError(f'{d.get("name")}: distribution {kind!r} (uniform, normal or choice)')
            d.setdefault('mass', 1.0)
            self.dims.append(d)
        self.dim = len(self.dims)
        if self.dim == 0:
            raise FailRateError('nothing to sample: the scenario has no parameter with a distribution')
        self.names = [d['name'] for d in self.dims]

    def doc(self) -> list:
        return [{k: v for k, v in d.items() if not k.startswith('_')} for d in self.dims]

    @property
    def mass(self) -> float:
        """The nominal probability of the kept (feasible) part: the estimate is conditional on it."""
        return float(np.prod([d['mass'] for d in self.dims]))

    def to_value(self, d: dict, z: float):
        if d['dist'] == 'uniform':
            return d['lo'] + (d['hi'] - d['lo']) * phi_cdf(z)
        if d['dist'] == 'normal':
            a, b = d['_ab']
            if (a, b) == (0.0, 1.0):
                return d['mean'] + d['sd'] * z
            u = a + (b - a) * phi_cdf(z)
            u = min(max(u, 1e-300), 1.0 - 1e-16)
            return d['mean'] + d['sd'] * _N.inv_cdf(u)
        k = len(d['choices'])
        return d['choices'][min(int(phi_cdf(z) * k), k - 1)]

    def to_params(self, z) -> dict:
        return {d['name']: self.to_value(d, float(v)) for d, v in zip(self.dims, z, strict=True)}

    def to_matrix(self, Z: np.ndarray) -> np.ndarray:
        """Vectorised to_params for continuous dimensions (synthetic problems): rows of parameter values."""
        Z = np.atleast_2d(np.asarray(Z, float))
        X = np.empty_like(Z)
        for j, d in enumerate(self.dims):
            if d['dist'] == 'normal' and d['_ab'] == (0.0, 1.0):
                X[:, j] = d['mean'] + d['sd'] * Z[:, j]
            elif d['dist'] == 'uniform':
                X[:, j] = d['lo'] + (d['hi'] - d['lo']) * _phi_vec(Z[:, j])
            else:
                X[:, j] = [self.to_value(d, z) if d['dist'] != 'choice' else float('nan') for z in Z[:, j]]
        return X

    @classmethod
    def standard(cls, dim: int) -> 'Nominal':
        """d independent standard normals named z0, z1, …: the synthetic problems' nominal."""
        return cls([{'name': f'z{j}', 'dist': 'normal', 'mean': 0.0, 'sd': 1.0} for j in range(int(dim))])

    @classmethod
    def from_scenario(cls, scn: dict, space: list) -> 'Nominal':
        """The nominal of a study: for each entry of its search space (sim.studio.Study.space(): the scenario's
        parameters, perhaps narrowed to the feasible part, and "weather.<field>" dimensions) the scenario's declared
        distribution restricted to the entry's range; a weather dimension uniform over its range."""
        from . import scenarios as SC
        raw = {p['name']: p for p in SC.search_space(scn)}
        params = scn.get('params') or {}
        dims = []
        for p in space:
            name = p['name']
            decl = params.get(name)
            if 'choices' in p:
                dims.append({'name': name, 'dist': 'choice', 'choices': list(p['choices'])})
                continue
            lo, hi = float(p['lo']), float(p['hi'])
            if decl is None:                                    # an extra (weather) dimension
                dims.append({'name': name, 'dist': 'uniform', 'lo': lo, 'hi': hi, 'source': 'uniform over its range '
                             '(a searched weather field)'})
                continue
            r = raw.get(name) or {}
            narrowed = (r.get('lo'), r.get('hi')) != (lo, hi)
            if decl.get('dist') == 'normal':
                m, sd = float(decl['mean']), float(decl['sd'])
                dlo = decl.get('lo') if not narrowed else lo
                dhi = decl.get('hi') if not narrowed else hi
                full = (phi_cdf((float(decl['hi']) - m) / sd) if decl.get('hi') is not None else 1.0) - \
                       (phi_cdf((float(decl['lo']) - m) / sd) if decl.get('lo') is not None else 0.0)
                kept = (phi_cdf((float(dhi) - m) / sd) if dhi is not None else 1.0) - \
                       (phi_cdf((float(dlo) - m) / sd) if dlo is not None else 0.0)
                dims.append({'name': name, 'dist': 'normal', 'mean': m, 'sd': sd, 'lo': dlo, 'hi': dhi,
                             'mass': kept / full if full > 0 else 1.0, 'source': 'declared normal'
                             + (' (narrowed to the feasible part)' if narrowed else '')})
            else:
                dlo, dhi = float(decl.get('lo', lo)), float(decl.get('hi', hi))
                dims.append({'name': name, 'dist': 'uniform', 'lo': lo, 'hi': hi,
                             'mass': (hi - lo) / (dhi - dlo) if dhi > dlo else 1.0,
                             'source': 'declared uniform' + (' (narrowed to the feasible part)' if narrowed else '')})
        return cls(dims)


# =============================================================================================== densities in z
def log_std_normal(Z: np.ndarray) -> np.ndarray:
    Z = np.atleast_2d(Z)
    return -0.5 * np.sum(Z * Z, axis=1) - 0.5 * Z.shape[1] * math.log(2 * math.pi)


def log_gauss(Z: np.ndarray, mean, cov) -> np.ndarray:
    Z = np.atleast_2d(Z)
    L = np.linalg.cholesky(np.asarray(cov, float))
    D = np.linalg.solve(L, (Z - np.asarray(mean, float)).T)
    return -0.5 * np.sum(D * D, axis=0) - np.sum(np.log(np.diag(L))) - 0.5 * Z.shape[1] * math.log(2 * math.pi)


def weights(Z: np.ndarray, proposal: dict | None) -> np.ndarray:
    """w = p/q for rows of z: 1 under the nominal (proposal None); the defensive mixture's 1/(β + (1 − β)·N/φ)."""
    Z = np.atleast_2d(Z)
    if proposal is None:
        return np.ones(len(Z))
    beta = float(proposal['beta'])
    lr = log_gauss(Z, proposal['mean'], proposal['cov']) - log_std_normal(Z)
    if beta <= 0:
        return np.exp(-lr)
    return 1.0 / (beta + (1.0 - beta) * np.exp(np.minimum(lr, 700.0)))


FAMILIES = ('mean_shift', 'full')    # full: kept only to reproduce the measurement that ruled it out (module doc)


def ce_fit(Z: np.ndarray, W: np.ndarray, family: str = 'mean_shift', reg: float = 1e-6) -> tuple:
    """The likelihood-ratio-weighted maximum-likelihood Gaussian of the elite (module doc): (μ, Σ), Σ = I for the
    mean-shift family; 'full' fits Σ too (Σ W (z − μ)(z − μ)ᵀ / Σ W + reg·I, diagonal when the elite are fewer than
    the dimension + 2)."""
    Z, W = np.atleast_2d(Z), np.asarray(W, float)
    s = W.sum()
    if not s > 0:
        raise FailRateError('the elite carry no weight')
    mu = (W[:, None] * Z).sum(axis=0) / s
    if family == 'mean_shift':
        return mu, np.eye(Z.shape[1])
    D = Z - mu
    if len(Z) >= Z.shape[1] + 2:
        C = (W[:, None, None] * np.einsum('ni,nj->nij', D, D)).sum(axis=0) / s
    else:
        C = np.diag((W[:, None] * D * D).sum(axis=0) / s)
    C = 0.5 * (C + C.T) + reg * np.eye(Z.shape[1])
    return mu, C


# =============================================================================================== the statistics
def summary(w, fail, conf_z: float = Z95, defensive: float | None = None) -> dict:
    """The importance-sampling estimate of independent draws (w = p/q, fail ∈ {0, 1}) with its interval, effective
    sample sizes and trust (module doc: the estimate, trust)."""
    w = np.asarray(w, float)
    f = np.asarray(fail, bool)
    n = int(len(w))
    out = {'n': n, 'failures': int(f.sum()), 'p': None, 'se': None, 'rel_se': None, 'ci95': None,
           'ci95_wald': None, 'ess': None, 'ess_failures': None, 'max_weight_share': None, 'status': 'refused',
           'reasons': [], 'upper95': None}
    if n < 2:
        out['reasons'].append('fewer than two runs')
        return out
    y = w * f
    p = float(y.mean())
    se = float(y.std(ddof=1) / math.sqrt(n))
    out.update(p=p, se=se, ess=float(w.sum() ** 2 / np.sum(w * w)))
    if f.any():
        wf = w[f]
        out['ess_failures'] = float(wf.sum() ** 2 / np.sum(wf * wf))
        out['max_weight_share'] = float(wf.max() / wf.sum())
    if p > 0:
        out['rel_se'] = se / p
        k = math.exp(conf_z * se / p)
        out['ci95'] = [p / k, p * k]
        out['ci95_wald'] = [max(0.0, p - conf_z * se), p + conf_z * se]
    if out['failures'] == 0:
        out['reasons'].append(f'no run failed among {n}')
        if defensive and defensive > 0:
            out['upper95'] = min(1.0, 3.0 / (defensive * n))
            out['reasons'].append(f'only a bound: P ≤ {out["upper95"]:.3g} at 95 % (rule of three on the defensive '
                                  f'part, β = {defensive:g})')
        return out
    if out['failures'] < MIN_FAILURES:
        out['reasons'].append(f'only {out["failures"]} failing runs (at least {MIN_FAILURES} are needed for the '
                              'interval to mean much)')
    if out['ess_failures'] < MIN_ESS:
        out['reasons'].append(f'effective sample size of the failures {out["ess_failures"]:.1f} < {MIN_ESS:g}: a '
                              'few heavy weights carry the estimate')
    if out['reasons']:
        return out
    warn = []
    if out['ess_failures'] < WARN_ESS:
        warn.append(f'effective sample size of the failures {out["ess_failures"]:.1f} < {WARN_ESS:g}')
    if out['rel_se'] > WARN_REL_SE:
        warn.append(f'relative standard error {out["rel_se"]:.2f} > {WARN_REL_SE:g}')
    if out['max_weight_share'] > WARN_WEIGHT_SHARE:
        warn.append(f'one run carries {out["max_weight_share"]:.0%} of the failures\' weight')
    out['status'] = 'warning' if warn else 'ok'
    out['reasons'] = warn
    return out


def mc_runs_needed(p: float, rel_se: float) -> float | None:
    """Plain Monte Carlo runs for the same relative standard error: (1 − p) / (p·rel²)."""
    if not p or not rel_se:
        return None
    return (1.0 - p) / (p * rel_se * rel_se)


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    tot = 0.0
    for i in range(0, k + 1):
        tot += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(tot, 1.0)


def binom_upper(k: int, n: int, conf: float = 0.95) -> float:
    """The one-sided Clopper–Pearson upper bound on a proportion: the p with P(X ≤ k; n, p) = 1 − conf (1 for
    n = 0). For k = 0 it is 1 − (1 − conf)^(1/n) ≈ 3/n at 95 % (the rule of three)."""
    k, n = int(k), int(n)
    if n <= 0 or k >= n:
        return 1.0
    a = 1.0 - conf
    if k == 0:
        return 1.0 - a ** (1.0 / n)
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(k, n, mid) > a:
            lo = mid
        else:
            hi = mid
    return hi


def binom_lower(k: int, n: int, conf: float = 0.95) -> float:
    """The one-sided Clopper–Pearson lower bound (0 for k = 0)."""
    k, n = int(k), int(n)
    if k <= 0 or n <= 0:
        return 0.0
    return 1.0 - binom_upper(n - k, n, conf)


def runs_needed(target: float, conf: float = 0.95, failures: int = 0) -> int:
    """The fewest runs with this many failures whose Clopper–Pearson upper bound is at most target (zero failures:
    ⌈ln(1 − conf) / ln(1 − target)⌉, the rule of three's exact form)."""
    if not 0 < target < 1:
        raise ValueError('target must lie in (0, 1)')
    if failures <= 0:
        return int(math.ceil(math.log(1.0 - conf) / math.log1p(-target)))
    lo, hi = failures + 1, max(failures + 2, 2)
    while binom_upper(failures, hi, conf) > target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if binom_upper(failures, mid, conf) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# =============================================================================================== the sampler
class Sampler:
    """The adaptive importance sampler, ask/tell like core.search (module doc). ask(n) → [{'id', 'params'}] (asked
    and not told ones from an earlier session first); tell(ids, values, fails=None): a value None is a run that failed
    to run (replaced by a new draw); fails: the judge's verdicts (default value ≥ threshold). done once the final stage
    is complete; result() the estimate. state()/from_state(): a JSON-able snapshot to resume exactly."""

    def __init__(self, nominal: Nominal, threshold: float = 0.5, seed: int = 0, stage_size: int = 100,
                 final_size: int = 200, rho: float = 0.1, defensive: float = 0.1, max_stages: int = 6,
                 family: str = 'mean_shift'):
        self.nominal = nominal
        if family not in FAMILIES:
            raise FailRateError(f'proposal family {family!r}: one of {", ".join(FAMILIES)}')
        self.family = family
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.stage_size, self.final_size = int(stage_size), int(final_size)
        self.rho, self.defensive, self.max_stages = float(rho), float(defensive), int(max_stages)
        if self.stage_size < 4 or self.final_size < 2:
            raise FailRateError('a stage needs at least 4 runs and the final stage at least 2')
        if not 0 < self.rho < 1 or not 0 <= self.defensive < 1 or self.max_stages < 0:
            raise FailRateError('rho in (0, 1), defensive in [0, 1), max_stages ≥ 0')
        self.rng = np.random.default_rng(self.seed)
        self.stage = 0
        self.phase = 'adapt' if self.max_stages > 0 else 'final'
        self.proposal = None if self.phase == 'adapt' else self._final_proposal(None)
        self.samples: list = []          # {'id', 'stage', 'z', 'w', 'value', 'fail', 'lost', 'pending'}
        self.stages: list = []           # per finished adaptation stage: level, elite, proposal fitted after it
        self.stop_reason = ''
        self._issued: set = set()

    def options(self) -> dict:
        return {'threshold': self.threshold, 'seed': self.seed, 'stage_size': self.stage_size,
                'final_size': self.final_size, 'rho': self.rho, 'defensive': self.defensive,
                'max_stages': self.max_stages, 'family': self.family}

    # ---------------------------------------------------------------- state
    @property
    def done(self) -> bool:
        return self.phase == 'done'

    def _target(self) -> int:
        return self.final_size if self.phase == 'final' else self.stage_size

    def _current(self) -> list:
        return [s for s in self.samples if s['stage'] == self.stage]

    def remaining(self) -> int:
        """Runs still to ask in the current stage (pending ones count as asked)."""
        if self.done:
            return 0
        cur = self._current()
        return max(0, self._target() - sum(1 for s in cur if not s['lost']))

    def total_runs(self) -> int:
        return sum(1 for s in self.samples if not s['pending'])

    def _final_proposal(self, fit) -> dict:
        if fit is None:                      # no adaptation: the nominal itself (plain Monte Carlo)
            return {'beta': 1.0, 'mean': [0.0] * self.nominal.dim, 'cov': np.eye(self.nominal.dim).tolist()}
        mu, C = fit
        return {'beta': self.defensive, 'mean': list(map(float, mu)), 'cov': np.asarray(C).tolist()}

    def _draw(self) -> np.ndarray:
        d = self.nominal.dim
        q = self.proposal
        if q is None or self.rng.random() < float(q['beta']):
            return self.rng.standard_normal(d)
        L = np.linalg.cholesky(np.asarray(q['cov'], float))
        return np.asarray(q['mean'], float) + L @ self.rng.standard_normal(d)

    def ask(self, n: int = 1) -> list:
        out = []
        for s in self.samples:                     # asked before (an earlier session) and not told: those first
            if len(out) >= n:
                break
            if s['pending'] and s['id'] not in self._issued:
                self._issued.add(s['id'])
                out.append({'id': s['id'], 'params': self.nominal.to_params(s['z'])})
        k = min(int(n) - len(out), self.remaining())
        for _ in range(max(k, 0)):
            z = self._draw()
            if self.proposal is not None and float(self.proposal['beta']) >= 1.0:
                w = 1.0
            else:
                w = float(weights(z[None, :], self.proposal)[0])
            s = {'id': len(self.samples), 'stage': self.stage, 'z': [float(v) for v in z], 'w': w, 'value': None,
                 'fail': None, 'lost': False, 'pending': True}
            self.samples.append(s)
            self._issued.add(s['id'])
            out.append({'id': s['id'], 'params': self.nominal.to_params(z)})
        return out

    def tell(self, ids, values, fails=None):
        fails = [None] * len(ids) if fails is None else list(fails)
        by = {s['id']: s for s in self.samples}
        for i, v, f in zip(ids, values, fails, strict=True):
            s = by[int(i)]
            if not s['pending']:
                continue
            s['pending'] = False
            if v is None:
                s['lost'] = True
                continue
            s['value'] = float(v)
            s['fail'] = bool(s['value'] >= self.threshold) if f is None else bool(f)
        lost = sum(1 for s in self._current() if s['lost'])
        if lost > max(self._target(), 10):
            raise FailRateError(f'{lost} runs of stage {self.stage} failed to run: stopping')
        cur = [s for s in self._current() if not s['lost']]
        if len(cur) >= self._target() and not any(s['pending'] for s in cur):
            self._advance(cur)

    def _advance(self, cur: list):
        if self.phase == 'final':
            self.phase = 'done'
            return
        Z = np.asarray([s['z'] for s in cur])
        v = np.asarray([s['value'] for s in cur])
        W = np.asarray([s['w'] for s in cur])
        n_elite = max(2, int(math.ceil(self.rho * len(cur))))
        n_fail = int(np.sum(v >= self.threshold))
        if n_fail >= n_elite:
            level, idx = self.threshold, np.flatnonzero(v >= self.threshold)
        else:
            order = np.argsort(-v, kind='stable')
            idx = order[:n_elite]
            level = min(self.threshold, float(v[idx[-1]]))
            if level <= float(v.min()):                 # the score does not separate the runs: nothing to adapt on
                self.stages.append({'stage': self.stage, 'runs': len(cur), 'level': float(level), 'elite': 0,
                                    'failures': n_fail, 'mean': None, 'sd': None})
                self.stage += 1
                self.stop_reason = ('the score did not separate the runs (the elite tie with the rest): no adaptation '
                                    'from this stage')
                self.phase = 'final'
                q = self.proposal
                self.proposal = self._final_proposal(None if q is None else (q['mean'], q['cov']))
                return
        fit = ce_fit(Z[idx], W[idx], self.family)
        prev = [st['level'] for st in self.stages]
        self.stages.append({'stage': self.stage, 'runs': len(cur), 'level': float(level), 'elite': int(len(idx)),
                            'failures': n_fail, 'mean': list(map(float, fit[0])),
                            'sd': list(map(float, np.sqrt(np.diag(fit[1]))))})
        reached = level >= self.threshold
        stalled = len(prev) >= 2 and level <= prev[-1] and prev[-1] <= prev[-2]
        self.stage += 1
        self.proposal = {'beta': self.defensive, 'mean': list(map(float, fit[0])), 'cov': np.asarray(fit[1]).tolist()}
        if reached or stalled or self.stage >= self.max_stages:
            self.stop_reason = ('the failure threshold was reached' if reached else
                                'the level stopped rising (stalled)' if stalled else
                                f'the stage limit ({self.max_stages}) was reached')
            self.phase = 'final'
            self.proposal = self._final_proposal(fit)

    # ---------------------------------------------------------------- results
    def final_samples(self) -> list:
        if self.phase not in ('final', 'done'):
            return []
        return [s for s in self.samples if s['stage'] == self.stage and not s['lost'] and not s['pending']]

    def result(self) -> dict:
        fin = self.final_samples()
        st = summary([s['w'] for s in fin], [s['fail'] for s in fin],
                     defensive=float(self.proposal['beta']) if self.proposal else None)
        s0 = [s for s in self.samples if s['stage'] == 0 and not s['lost'] and not s['pending']]
        k0 = sum(1 for s in s0 if s['fail'])
        mc = {'runs': len(s0), 'failures': k0, 'p': (k0 / len(s0)) if s0 else None,
              'upper95': binom_upper(k0, len(s0)) if s0 else None, 'lower95': binom_lower(k0, len(s0)) if s0 else None,
              'about': 'stage 0 is a plain Monte Carlo sample of the nominal: its failures and one-sided '
                       'Clopper–Pearson 95 % bounds'}
        total = self.total_runs()
        need = mc_runs_needed(st['p'], st['rel_se']) if st['p'] else None
        return {'schema': SCHEMA, 'estimator': ESTIMATOR, 'proposal_family': PROPOSAL, 'options': self.options(),
                'nominal': self.nominal.doc(), 'nominal_mass': self.nominal.mass, 'stages': self.stages,
                'stop_reason': self.stop_reason, 'final_proposal': self.proposal, 'estimate': st,
                'runs': {'total': total, 'final': len(fin), 'lost': sum(1 for s in self.samples if s['lost'])},
                'plain_mc': mc, 'mc_runs_for_same_rel_se': need,
                'efficiency': (need / total) if need and total else None, 'done': self.done}

    def state(self) -> dict:
        return {'schema': STATE_SCHEMA, 'nominal': self.nominal.doc(), 'options': self.options(),
                'rng': self.rng.bit_generator.state, 'stage': self.stage, 'phase': self.phase,
                'proposal': copy.deepcopy(self.proposal), 'stages': copy.deepcopy(self.stages),
                'stop_reason': self.stop_reason, 'samples': copy.deepcopy(self.samples)}

    @classmethod
    def from_state(cls, d: dict) -> 'Sampler':
        if d.get('schema') != STATE_SCHEMA:
            raise FailRateError(f'not a failure-rate state ({d.get("schema")!r})')
        s = cls(Nominal(d['nominal']), **d['options'])
        s.rng.bit_generator.state = d['rng']
        s.stage, s.phase, s.proposal = int(d['stage']), d['phase'], d['proposal']
        s.stages, s.stop_reason, s.samples = d['stages'], d.get('stop_reason', ''), d['samples']
        return s


def run(sampler: Sampler, objective, batch: int = 0) -> dict:
    """Drive a sampler to the end with objective(list of params) -> list of values (or (value, fail) pairs);
    returns result()."""
    while not sampler.done:
        asked = sampler.ask(batch or max(sampler.remaining(), 1))
        if not asked:
            raise FailRateError('the sampler asked for nothing before it was done')
        out = objective([a['params'] for a in asked])
        vals = [o[0] if isinstance(o, tuple) else o for o in out]
        fails = [o[1] if isinstance(o, tuple) else None for o in out]
        sampler.tell([a['id'] for a in asked], vals, fails)
    return sampler.result()


# =============================================================================================== per ODD cell
def by_cell(samples: list, cell_of) -> dict:
    """The final stage's estimate restricted to ODD cells: P(fail | cell) = Σ w·1{fail}·1{cell} / Σ w·1{cell} (the
    ratio estimator), its delta-method standard error, the cell's effective sample size, and the same trust rules as
    the whole (a cell with too few effective failures is refused). samples: [{'w', 'fail', ...}]; cell_of(sample) ->
    a cell id or a list of them (a run can lie in several, e.g. two actor kinds)."""
    cells: dict = {}
    n = len(samples)
    for k, s in enumerate(samples):
        ids = cell_of(s)
        for c in ([ids] if isinstance(ids, str) else ids or ()):
            cells.setdefault(c, []).append(k)
    out = {}
    if not n:
        return out
    w = np.asarray([s['w'] for s in samples], float)
    f = np.asarray([bool(s['fail']) for s in samples], float)
    for c, idx in sorted(cells.items()):
        ind = np.zeros(n)
        ind[idx] = 1.0
        den = float(np.mean(w * ind))
        num = float(np.mean(w * ind * f))
        r = num / den if den > 0 else None
        row = {'runs': len(idx), 'failures': int(f[idx].sum()), 'p': r, 'ci95': None, 'se': None,
               'ess': float((w[idx].sum()) ** 2 / np.sum(w[idx] ** 2)), 'ess_failures': None, 'status': 'refused',
               'reasons': []}
        fi = [i for i in idx if f[i]]
        if fi:
            row['ess_failures'] = float(w[fi].sum() ** 2 / np.sum(w[fi] ** 2))
        if r is not None and n > 1:
            g = w * ind * (f - r)
            row['se'] = float(np.sqrt(np.sum(g * g) / (n * (n - 1))) / den)
        if row['failures'] == 0:
            row['reasons'].append(f'no failure among the cell\'s {len(idx)} runs')
        elif row['failures'] < MIN_FAILURES or (row['ess_failures'] or 0) < MIN_ESS:
            row['reasons'].append(f'{row["failures"]} failing runs, effective {row["ess_failures"]:.1f}: too few')
        elif r and row['se'] is not None:
            k = math.exp(Z95 * row['se'] / r)
            row['ci95'] = [r / k, r * k]
            row['status'] = 'warning' if row['ess_failures'] < WARN_ESS or row['se'] / r > WARN_REL_SE else 'ok'
        out[c] = row
    return out


# =============================================================================================== exposure
EXPOSURE_UNITS = ('encounter', 'km', 'hour')
PLURAL = {'encounter': 'encounters', 'km': 'km', 'hour': 'hours'}


def exposure_rate(est: dict, exposure: dict | None) -> dict:
    """The estimate per encounter, or — with exposure {'per': 'km' | 'hour', 'encounters': X, 'source': '…'} —
    failures per 10ⁿ km (or hours): P̂ × X, n chosen so the number is at least 1, the interval scaled alike. The
    encounter rate is the user's and taken as exact."""
    e = est or {}
    per = (exposure or {}).get('per') or 'encounter'
    if per not in EXPOSURE_UNITS:
        raise FailRateError(f'exposure per {per!r}: one of {", ".join(EXPOSURE_UNITS)}')
    x = 1.0 if per == 'encounter' else float((exposure or {}).get('encounters') or 0.0)
    if per != 'encounter' and not x > 0:
        raise FailRateError(f'give how many encounters happen per {per} (a positive number) and where it comes from')
    base = e.get('p') if e.get('status') != 'refused' else None
    ref = base if base else e.get('upper95')
    out = {'per': per, 'encounters_per_unit': x if per != 'encounter' else None,
           'source': (exposure or {}).get('source') or ('' if per == 'encounter' else 'not given'),
           'about': ('per encounter: one run of the scenario' if per == 'encounter' else
                     f'failures per 10ⁿ {per}s = P(fail per encounter) × {x:g} encounters per {per} (the user\'s '
                     'figure, taken as exact)')}
    if not ref:
        out.update(power=0, rate=None, ci95=None, upper95=None, text='no rate: the estimate was refused')
        return out
    r = ref * x
    power = max(0, int(math.ceil(-math.log10(r)))) if r < 1 else 0
    scale = 10.0 ** power
    unit = 'encounter' if per == 'encounter' else per
    out['power'] = power
    out['rate'] = base * x * scale if base else None
    out['ci95'] = [c * x * scale for c in e['ci95']] if base and e.get('ci95') else None
    out['upper95'] = e['upper95'] * x * scale if e.get('upper95') else None
    where = f'per 10^{power} {PLURAL[unit]}' if power else f'per {unit}'
    if out['rate'] is not None:
        out['text'] = (f'{out["rate"]:.3g} failures {where} (95 %: {out["ci95"][0]:.3g}–{out["ci95"][1]:.3g})'
                       if out['ci95'] else f'{out["rate"]:.3g} failures {where}')
    else:
        out['text'] = f'at most {out["upper95"]:.3g} failures {where} (95 % bound; no estimate)'
    return out


# =============================================================================================== validation
def tail_problem(p: float, dim: int = 2, direction=None) -> tuple:
    """A synthetic problem with a known failure probability: d standard normals, the score s(z) = a·z / |a| (a the
    direction, default all ones) and the threshold c = −Φ⁻¹(p), so P(s ≥ c) = Φ(−c) = p exactly. Returns
    (nominal, objective over params lists, threshold)."""
    a = np.ones(dim) if direction is None else np.asarray(direction, float)
    a = a / np.linalg.norm(a)
    c = -_N.inv_cdf(float(p))
    nom = Nominal.standard(dim)

    def objective(params_list):
        X = np.asarray([[q[n] for n in nom.names] for q in params_list], float)
        return list(map(float, X @ a))
    return nom, objective, c


def corner_problem(p: float, dim: int = 2) -> tuple:
    """A synthetic problem through the uniform transform: d parameters uniform on [0, 1], a failure when every one
    exceeds 1 − p^(1/d) (a corner of the box, so P = p exactly), the score min_j u_j − (1 − p^(1/d)) and the
    threshold 0. Returns (nominal, objective over params lists, threshold)."""
    edge = 1.0 - float(p) ** (1.0 / dim)
    nom = Nominal([{'name': f'u{j}', 'dist': 'uniform', 'lo': 0.0, 'hi': 1.0} for j in range(int(dim))])

    def objective(params_list):
        return [min(q[n] for n in nom.names) - edge for q in params_list]
    return nom, objective, 0.0


PROBLEMS = {'tail': tail_problem, 'corner': corner_problem}


def plain_mc(p_true: float, n: int, seed: int) -> dict:
    """Plain Monte Carlo of a known probability: the failures among n independent runs (binomial draw) and the
    estimate with its Clopper–Pearson bounds."""
    k = int(np.random.default_rng(seed).binomial(int(n), float(p_true)))
    return {'n': int(n), 'failures': k, 'p': k / n, 'upper95': binom_upper(k, n), 'lower95': binom_lower(k, n)}


def validate(probabilities=(1e-4, 1e-6), dims=(2, 5), seeds=range(50), stage_size: int = 500,
             final_size: int = 1000, rho: float = 0.1, defensive: float = 0.1, max_stages: int = 10,
             family: str = 'mean_shift', kinds=('tail',), progress=None) -> dict:
    """The estimator on synthetic problems with a known probability (kinds: tail — tail_problem, a linear limit state
    in standard normals; corner — corner_problem, uniform parameters) over seeds: per problem the share of seeds whose
    95 % interval covers the
    truth (and the Wald interval's), the refusals and warnings, the mean relative error and runs, and plain Monte
    Carlo with the same number of runs (the share of seeds with no failure at all, and the runs it would need for the
    estimator's median relative standard error)."""
    rows = []
    seeds = list(seeds)
    total = len(kinds) * len(probabilities) * len(dims) * len(seeds)
    k = 0
    for kind, p, d in ((kind, p, d) for kind in kinds for p in probabilities for d in dims):
        nom, f, c = PROBLEMS[kind](p, d)
        cov = covw = given = warned = refused = cov_ok = 0
        rel_err, runs, rse, ess = [], [], [], []
        mc_zero = 0
        for s in seeds:
            sm = Sampler(nom, threshold=c, seed=s, stage_size=stage_size, final_size=final_size, rho=rho,
                         defensive=defensive, max_stages=max_stages, family=family)
            res = run(sm, f)
            e = res['estimate']
            runs.append(res['runs']['total'])
            if e['status'] == 'refused':
                refused += 1
            else:
                given += 1
                warned += e['status'] == 'warning'
                hit = e['ci95'][0] <= p <= e['ci95'][1]
                cov += hit
                cov_ok += hit and e['status'] == 'ok'
                covw += e['ci95_wald'][0] <= p <= e['ci95_wald'][1]
                rel_err.append(abs(e['p'] - p) / p)
                rse.append(e['rel_se'])
                ess.append(e['ess_failures'])
            mc_zero += plain_mc(p, res['runs']['total'], 10 ** 6 + s)['failures'] == 0
            k += 1
            if progress:
                progress(k / total)
        med_rse = float(np.median(rse)) if rse else None
        rows.append({'problem': kind, 'p_true': p, 'dim': d, 'threshold': c, 'seeds': len(seeds), 'given': given,
                     'refused': refused, 'warned': warned,
                     'coverage': cov / given if given else None, 'coverage_wald': covw / given if given else None,
                     'coverage_ok': cov_ok / (given - warned) if given > warned else None,
                     'mean_abs_rel_error': float(np.mean(rel_err)) if rel_err else None,
                     'median_rel_se': med_rse, 'median_ess_failures': float(np.median(ess)) if ess else None,
                     'mean_runs': float(np.mean(runs)),
                     'plain_mc_zero_failure_share': mc_zero / len(seeds),
                     'plain_mc_runs_for_same_rel_se': mc_runs_needed(p, med_rse) if med_rse else None})
    return {'schema': 'falcon.failure_rate_validation/1', 'estimator': ESTIMATOR, 'proposal_family': PROPOSAL,
            'settings': {'stage_size': stage_size, 'final_size': final_size, 'rho': rho, 'defensive': defensive,
                         'max_stages': max_stages, 'family': family}, 'rows': rows}
