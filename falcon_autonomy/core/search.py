"""Rare-case search over scenario parameters (standard library + numpy): find where a system fails, in few runs.

A parameter space is a list of {'name', 'lo', 'hi'} (continuous) or {'name', 'choices': [...]} (discrete: a unit
coordinate u picks choices[min(⌊u·k⌋, k-1)], and a choice maps back to the centre of its cell), normalised to the
unit cube. An objective is f(params: dict) -> float, higher = more critical (in the app: a CARLA scenario run scored
by core.criticality, expensive). Every method is ask/tell: ask(n) proposes n parameter sets (a batch to run in
parallel), tell(params_list, values) reports them (a value None: the run failed, nothing is learnt), best(k),
history ([{'params', 'value', 'i'}]), and state() / from_state() — a JSON-able snapshot (history, pending asks, the
seeded generator's state, the method's internals) from which a resumed search continues exactly as the uninterrupted
one would. A search can also start from past evaluations (history=[{'params', 'value'}] or [(params, value)]).

Methods.
- RandomSearch: seeded uniform sampling, or a Halton sequence (Halton 1960) with a seeded random shift
  (Cranley & Patterson 1976) for low discrepancy. The baseline.
- CMAES: (μ/μ_w, λ) CMA-ES as in Hansen 2016, The CMA Evolution Strategy: A Tutorial (arXiv:1604.00772):
  weighted recombination, cumulative step-size adaptation, rank-one and rank-μ covariance updates, default
  λ = 4 + ⌊3 ln n⌋.
  The box is handled by reflecting samples into [0, 1]; the update uses the repaired (and choice-rounded) points,
  their steps clipped to Mahalanobis length √n + 2n/(n+2) as for injected solutions (Hansen 2011, Injecting
  External Solutions Into CMA-ES, arXiv:1110.4181). A run whose step size falls below 1e-4 restarts from a new
  random mean.
- BayesOpt: a Gaussian-process surrogate (Matérn 5/2 kernel with per-dimension lengthscales, constant mean, noise;
  Rasmussen & Williams 2006, GPML, eq. 4.17 and Algorithm 2.1, numpy Cholesky) with hyperparameters chosen by
  log marginal likelihood over a grid (an isotropic lengthscale × noise grid, then one coordinate sweep per
  dimension; the signal variance profiled out in closed form), and expected improvement (Jones, Schonlau & Welch
  1998, Efficient Global Optimization of Expensive Black-Box Functions, J. Global Optim. 13; Snoek, Larochelle &
  Adams 2012, Practical Bayesian Optimization of Machine Learning Algorithms) maximised over uniform random
  candidates plus local perturbations of the best points, refined once around the winner. Hyperparameters are
  refitted at every evaluation up to 30, then at every 5th. A batch is chosen by the kriging believer (Ginsbourger,
  Le Riche & Carraro 2010): each chosen or pending point is added at its predicted mean before the next is chosen.
  Starts with max(2n + 2, 6) uniform points.
- AdaptiveSampling: the cross-entropy method for rare events (Rubinstein 1997, Optimization of computer simulation
  models with rare events, EJOR 99; Rubinstein 1999, The cross-entropy method for combinatorial and continuous
  optimization; as used for AV testing by O'Kelly et al. 2018, Scalable end-to-end autonomous vehicle testing via
  rare-event simulation): after a uniform start, samples come from a Gaussian fitted to the elite — the failures
  (value ≥ threshold) once there are enough of them, until then the best evaluations — so the sampling concentrates
  on (and then maps) the failure region; a share of every batch stays uniform to find other failure modes. The
  sampling distribution is a function of the history alone (no smoothing state), so batches and resumes are exact.

Analysis: failures(history, threshold) ranks the failures; failure_map(...) bins the history on two parameters into
a grid of max / mean value and failure share (a heatmap of the failure region); compare(...) runs methods with equal
budgets over seeds and returns the curves of failures found and best value vs evaluations.

planted_objective(): a synthetic, known-weak model for testing the searchers (not a measurement of anything): a
detector-miss probability that is high only inside a planted ellipsoid (about 3 % of the unit cube's volume by
default), smooth around it, low elsewhere, with mild deterministic noise (seeded, a function of the parameters).

Limits: the GP is O(n³) in the number of evaluations (fine for the hundreds a CARLA search affords, not for 10⁴);
CMA-ES and the CEM fit one Gaussian, so they track one failure mode at a time (restarts and the uniform share find
others); discrete choices are searched through their rounded coordinate, with no notion of order beyond it.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

STATE_SCHEMA = 'falcon.search_state/1'


# --- the parameter space -------------------------------------------------------------------------------------------

def normalise_space(space) -> list:
    """A validated copy of a parameter space: [{'name', 'lo', 'hi'} | {'name', 'choices'}]."""
    out, seen = [], set()
    for p in space:
        name = str(p['name'])
        if name in seen:
            raise ValueError(f'parameter {name!r} appears twice')
        seen.add(name)
        if 'choices' in p:
            ch = list(p['choices'])
            if not ch:
                raise ValueError(f'parameter {name!r} has no choices')
            out.append({'name': name, 'choices': ch})
        else:
            lo, hi = float(p['lo']), float(p['hi'])
            if not hi > lo:
                raise ValueError(f'parameter {name!r}: hi must be above lo')
            out.append({'name': name, 'lo': lo, 'hi': hi})
    return out


def to_params(space, x) -> dict:
    """Unit-cube coordinates -> a parameter dict (clipped to the box; choices by their cell)."""
    out = {}
    for p, u in zip(space, x, strict=True):
        u = min(max(float(u), 0.0), 1.0)
        if 'choices' in p:
            k = len(p['choices'])
            out[p['name']] = p['choices'][min(int(u * k), k - 1)]
        else:
            out[p['name']] = p['lo'] + u * (p['hi'] - p['lo'])
    return out


def to_unit(space, params: dict) -> np.ndarray:
    """A parameter dict -> unit-cube coordinates (a choice -> the centre of its cell)."""
    x = np.empty(len(space))
    for d, p in enumerate(space):
        v = params[p['name']]
        if 'choices' in p:
            try:
                k = p['choices'].index(v)
            except ValueError:
                raise ValueError(f'{v!r} is not a choice of {p["name"]!r}') from None
            x[d] = (k + 0.5) / len(p['choices'])
        else:
            x[d] = (float(v) - p['lo']) / (p['hi'] - p['lo'])
    return np.clip(x, 0.0, 1.0)


def _reflect(x: np.ndarray) -> np.ndarray:
    """Fold coordinates into [0, 1] by reflection at the faces."""
    y = np.mod(x, 2.0)
    return np.where(y > 1.0, 2.0 - y, y)


# --- the common ask/tell machinery ---------------------------------------------------------------------------------

class Search:
    """Base of the searchers: ask/tell, history, best, state/from_state. Subclasses implement _propose(n) and,
    when they keep internals beyond the history, _observe / _extra_state / _load_extra."""

    method = 'base'

    def __init__(self, space, seed: int = 0, history=None):
        self.space = normalise_space(space)
        self.dim = len(self.space)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.history = []
        self._xs = []          # unit coordinates of the history, parallel to it
        self._pending = []     # unit coordinates asked and not yet told
        self._init()
        for h in history or ():
            p, v = (h['params'], h['value']) if isinstance(h, dict) else h
            self.tell([p], [v])

    default_batch = 1

    def _init(self):
        pass

    def _options(self) -> dict:
        return {}

    def _propose(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def _observe(self, X: np.ndarray, y: np.ndarray):
        pass

    def _extra_state(self) -> dict:
        return {}

    def _load_extra(self, d: dict):
        pass

    def ask(self, n: int = 1) -> list:
        """n parameter sets to evaluate next (pending until told)."""
        out = []
        for x in np.atleast_2d(self._propose(int(n)))[:n]:
            p = to_params(self.space, x)
            self._pending.append(to_unit(self.space, p).tolist())
            out.append(p)
        return out

    def tell(self, params_list, values):
        """Report evaluations (asked or not); a value None (the run failed) only clears the pending ask."""
        X, Y = [], []
        for p, v in zip(params_list, values, strict=True):
            x = to_unit(self.space, p)
            for k, q in enumerate(self._pending):
                if np.max(np.abs(np.asarray(q) - x)) < 1e-9:
                    del self._pending[k]
                    break
            if v is None:
                continue
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f'objective value {v} is not finite (tell None for a failed run)')
            self.history.append({'params': {s['name']: (p[s['name']] if 'choices' in s else float(p[s['name']]))
                                            for s in self.space}, 'value': v, 'i': len(self.history)})
            self._xs.append(x)
            X.append(x)
            Y.append(v)
        if X:
            self._observe(np.asarray(X), np.asarray(Y))

    def best(self, k: int = 1) -> list:
        """The k highest-valued evaluations (ties: earliest first)."""
        return sorted(self.history, key=lambda h: (-h['value'], h['i']))[:k]

    def X(self) -> np.ndarray:
        return np.asarray(self._xs).reshape(-1, self.dim)

    def y(self) -> np.ndarray:
        return np.asarray([h['value'] for h in self.history], dtype=float)

    def state(self) -> dict:
        """A JSON-able snapshot to resume from (from_state)."""
        return {'schema': STATE_SCHEMA, 'method': self.method, 'space': self.space, 'seed': self.seed,
                'options': self._options(), 'rng': self.rng.bit_generator.state,
                'history': [dict(h, params=dict(h['params'])) for h in self.history],
                'pending': [list(q) for q in self._pending], 'extra': self._extra_state()}

    @staticmethod
    def from_state(d: dict) -> 'Search':
        return from_state(d)


def from_state(d: dict) -> Search:
    """Rebuild a searcher from state(): it continues exactly where the snapshot was taken."""
    if d.get('schema') != STATE_SCHEMA:
        raise ValueError(f'not a search state ({d.get("schema")!r})')
    cls = METHODS[d['method']]
    s = cls(d['space'], seed=d['seed'], **d.get('options', {}))
    s.history = [dict(h, params=dict(h['params'])) for h in d['history']]
    s._xs = [to_unit(s.space, h['params']) for h in s.history]
    s._pending = [list(q) for q in d.get('pending', [])]
    s.rng.bit_generator.state = d['rng']
    s._load_extra(d.get('extra') or {})
    return s


# --- random search -------------------------------------------------------------------------------------------------

def _primes(n: int) -> list:
    out, k = [], 2
    while len(out) < n:
        if all(k % p for p in out if p * p <= k):
            out.append(k)
        k += 1
    return out


def _radical_inverse(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class RandomSearch(Search):
    """Seeded uniform sampling (sequence='uniform') or a randomly shifted Halton sequence (sequence='halton')."""

    method = 'random'

    def __init__(self, space, seed: int = 0, sequence: str = 'uniform', history=None):
        if sequence not in ('uniform', 'halton'):
            raise ValueError(f'unknown sequence {sequence!r}')
        self.sequence = sequence
        super().__init__(space, seed, history)

    def _init(self):
        self._k = 0
        self._shift = self.rng.random(self.dim).tolist() if self.sequence == 'halton' else []

    def _options(self):
        return {'sequence': self.sequence}

    def _propose(self, n):
        if self.sequence == 'uniform':
            return self.rng.random((n, self.dim))
        bases = _primes(self.dim)
        out = np.empty((n, self.dim))
        for r in range(n):
            self._k += 1
            out[r] = [_radical_inverse(self._k, b) for b in bases]
        return np.mod(out + np.asarray(self._shift), 1.0)

    def _extra_state(self):
        return {'k': self._k, 'shift': list(self._shift)}

    def _load_extra(self, d):
        self._k = int(d.get('k', 0))
        self._shift = list(d.get('shift', self._shift))


# --- CMA-ES --------------------------------------------------------------------------------------------------------

class CMAES(Search):
    """(μ/μ_w, λ) CMA-ES (Hansen 2016) maximising the objective in the unit cube; generations update once λ
    evaluations have been told (asked or injected), so any batching works."""

    method = 'cmaes'

    def __init__(self, space, seed: int = 0, sigma0: float = 0.3, popsize: int | None = None, history=None):
        self.sigma0 = float(sigma0)
        self.popsize = popsize
        super().__init__(space, seed, history)

    def _options(self):
        return {'sigma0': self.sigma0, 'popsize': self.popsize}

    def _init(self):
        n = self.dim
        self.lam = int(self.popsize or 4 + int(3 * math.log(n)))
        self.mu = self.lam // 2
        w = math.log((self.lam + 1) / 2) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        me = self.mueff
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.damps = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.cy = math.sqrt(n) + 2 * n / (n + 2)
        self.default_batch = self.lam
        self.restarts = 0
        self._restart()
        self._buffer = []

    def _restart(self):
        n = self.dim
        self.mean = self.rng.random(n)
        self.sigma = self.sigma0
        self.C = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.gen = 0

    def _eig(self):
        d2, B = np.linalg.eigh((self.C + self.C.T) / 2)
        return np.sqrt(np.maximum(d2, 1e-20)), B

    def _propose(self, n):
        D, B = self._eig()
        z = self.rng.standard_normal((n, self.dim))
        return _reflect(self.mean + self.sigma * (z * D) @ B.T)

    def _observe(self, X, y):
        self._buffer.extend([x.tolist(), float(v)] for x, v in zip(X, y))
        while len(self._buffer) >= self.lam:
            gen, self._buffer = self._buffer[:self.lam], self._buffer[self.lam:]
            self._update(gen)

    def _update(self, gen):
        n = self.dim
        gen = sorted(gen, key=lambda g: -g[1])[:self.mu]
        D, B = self._eig()
        inv_sqrt = (B / D) @ B.T
        Y = (np.asarray([g[0] for g in gen]) - self.mean) / self.sigma
        maha = np.linalg.norm(Y @ inv_sqrt.T, axis=1)
        Y *= np.minimum(1.0, self.cy / np.maximum(maha, 1e-300))[:, None]
        yw = self.weights @ Y
        self.mean = np.clip(self.mean + self.sigma * yw, 0.0, 1.0)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (inv_sqrt @ yw)
        ps_norm = float(np.linalg.norm(self.ps))
        hsig = ps_norm / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (n + 1)) * self.chin
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (Y * self.weights[:, None]).T @ Y
        self.C = ((1 - self.c1 - self.cmu + (1 - hsig) * self.c1 * self.cc * (2 - self.cc)) * self.C
                  + self.c1 * np.outer(self.pc, self.pc) + self.cmu * rank_mu)
        self.C = (self.C + self.C.T) / 2
        self.sigma = min(self.sigma * math.exp((self.cs / self.damps) * (ps_norm / self.chin - 1)), 1.0)
        self.gen += 1
        D, _ = self._eig()
        if self.sigma * D.max() < 1e-4 or D.max() > 1e7 * D.min():
            self.restarts += 1
            self._restart()

    def _extra_state(self):
        return {'mean': self.mean.tolist(), 'sigma': self.sigma, 'C': self.C.tolist(), 'pc': self.pc.tolist(),
                'ps': self.ps.tolist(), 'gen': self.gen, 'restarts': self.restarts, 'buffer': self._buffer}

    def _load_extra(self, d):
        self.mean, self.sigma = np.asarray(d['mean']), float(d['sigma'])
        self.C, self.pc, self.ps = np.asarray(d['C']), np.asarray(d['pc']), np.asarray(d['ps'])
        self.gen, self.restarts = int(d['gen']), int(d['restarts'])
        self._buffer = [[list(x), float(v)] for x, v in d['buffer']]


# --- Bayesian optimisation -----------------------------------------------------------------------------------------

_SQRT5 = math.sqrt(5.0)
_LS_GRID = (0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 2.0)
_NOISE_GRID = (1e-6, 1e-4, 1e-2, 1e-1)
_erf = np.vectorize(math.erf, otypes=[float])


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared differences per dimension, (len(A), len(B), d)."""
    return (A[:, None, :] - B[None, :, :]) ** 2


def _matern52(D2: np.ndarray, ls: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(D2 @ (1.0 / np.asarray(ls) ** 2), 0.0))
    return (1 + _SQRT5 * d + 5.0 / 3.0 * d * d) * np.exp(-_SQRT5 * d)


class _GP:
    """GP regression with a Matérn 5/2 kernel: K = sf2 · (R(ls) + eta·I) on standardised targets."""

    def __init__(self, X, y, ls, eta, D2=None):
        self.X, self.ls, self.eta = X, np.asarray(ls, dtype=float), float(eta)
        self.ym, self.ys = float(y.mean()), float(y.std()) or 1.0
        z = (y - self.ym) / self.ys
        n = len(X)
        # one Cholesky of R bordered by z: its last row is a = L⁻¹z (no triangular solve needed for the likelihood);
        # the corner just has to exceed zᵀR⁻¹z ≤ |z|² / (eta + 1e-10)
        M = np.empty((n + 1, n + 1))
        M[:n, :n] = _matern52(_sqdist(X, X) if D2 is None else D2, self.ls) + (eta + 1e-10) * np.eye(n)
        M[:n, n] = M[n, :n] = z
        M[n, n] = 2.0 * float(z @ z) / (eta + 1e-10) + 1.0
        Lb = np.linalg.cholesky(M)
        self.L, a = Lb[:n, :n], Lb[n, :n]
        self._a, self._alpha = a, None
        self.sf2 = max(float(a @ a) / n, 1e-12)            # the signal variance's maximum-likelihood value
        self.lml = -0.5 * n * math.log(self.sf2) - float(np.sum(np.log(np.diag(self.L)))) - 0.5 * n

    def predict(self, Xs):
        """Mean and standard deviation of the latent function, in standardised units."""
        Ks = _matern52(_sqdist(Xs, self.X), self.ls)
        if self._alpha is None:
            self._alpha = np.linalg.solve(self.L.T, self._a)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = self.sf2 * np.maximum(1.0 - np.sum(v * v, axis=0), 1e-12)
        return mu, np.sqrt(var)

    def standardise(self, y):
        return (np.asarray(y) - self.ym) / self.ys


def fit_gp(X: np.ndarray, y: np.ndarray) -> _GP:
    """Hyperparameters by log marginal likelihood: isotropic lengthscale × noise grid, then one sweep of each
    dimension's lengthscale over the grid."""
    best, D2 = None, _sqdist(X, X)
    for ls in _LS_GRID:
        for eta in _NOISE_GRID:
            try:
                g = _GP(X, y, np.full(X.shape[1], ls), eta, D2)
            except np.linalg.LinAlgError:
                continue
            if best is None or g.lml > best.lml:
                best = g
    for d in range(X.shape[1]):
        for ls in _LS_GRID:
            trial = best.ls.copy()
            trial[d] = ls
            try:
                g = _GP(X, y, trial, best.eta, D2)
            except np.linalg.LinAlgError:
                continue
            if g.lml > best.lml:
                best = g
    return best


def expected_improvement(mu, sd, best, xi=0.01):
    """EI for maximisation (Jones et al. 1998): (μ - best - ξ)Φ(z) + σφ(z), z = (μ - best - ξ)/σ."""
    imp = mu - best - xi
    z = imp / sd
    cdf = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return imp * cdf + sd * pdf


class BayesOpt(Search):
    """GP + expected improvement (Jones et al. 1998; Snoek et al. 2012); batches by the kriging believer."""

    method = 'bayesopt'

    def __init__(self, space, seed: int = 0, n_init: int | None = None, n_candidates: int = 2000,
                 xi: float = 0.01, history=None):
        self.n_init = n_init
        self.n_candidates = int(n_candidates)
        self.xi = float(xi)
        super().__init__(space, seed, history)

    def _options(self):
        return {'n_init': self.n_init, 'n_candidates': self.n_candidates, 'xi': self.xi}

    def _init(self):
        self._n_init = int(self.n_init or max(2 * self.dim + 2, 6))
        self._gp_cache = (None, None)

    @staticmethod
    def _hyper_n(n: int) -> int:
        """The hyperparameters at n evaluations are fitted on the first _hyper_n(n): every evaluation up to 30,
        then every 5th (a function of n alone, so a resumed search refits exactly as the uninterrupted one)."""
        return n if n <= 30 else n - (n - 30) % 5

    def _model(self) -> _GP:
        n = len(self.history)
        if self._gp_cache[0] != n:
            m = self._hyper_n(n)
            X, y = self.X(), self.y()
            h = fit_gp(X[:m], y[:m])
            self._gp_cache = (n, h if m == n else _GP(X, y, h.ls, h.eta))
        return self._gp_cache[1]

    def _propose(self, n):
        out = []
        for _ in range(n):
            if len(self.history) + len(self._pending) + len(out) < self._n_init or len(self.history) < 2:
                out.append(self.rng.random(self.dim))
                continue
            out.append(self._next(np.asarray(self._pending + [x.tolist() for x in out]).reshape(-1, self.dim)))
        return np.asarray(out)

    def _next(self, pending: np.ndarray) -> np.ndarray:
        gp = self._model()
        X, y = self.X(), self.y()
        if len(pending):
            # kriging believer: the pending points at their predicted mean, hyperparameters kept
            mu_p, _ = gp.predict(pending)
            yb = np.concatenate([y, gp.ym + gp.ys * mu_p])
            gp = _GP(np.vstack([X, pending]), yb, gp.ls, gp.eta)
        best = float(gp.standardise(y.max()))
        d = self.dim
        sd = np.clip(0.5 * gp.ls, 0.01, 0.2)
        top = X[np.argsort(-y, kind='stable')[:5]]
        local = (np.repeat(top, 100, axis=0) + self.rng.standard_normal((100 * len(top), d)) * sd)
        cand = np.vstack([self.rng.random((self.n_candidates, d)), np.clip(local, 0, 1)])
        mu, s = gp.predict(cand)
        ei = expected_improvement(mu, s, best, self.xi)
        x0 = cand[int(np.argmax(ei))]
        ref = np.clip(x0 + self.rng.standard_normal((200, d)) * sd * 0.25, 0, 1)
        mu, s = gp.predict(ref)
        ei_r = expected_improvement(mu, s, best, self.xi)
        return ref[int(np.argmax(ei_r))] if ei_r.max() > ei.max() else x0


# --- adaptive sampling (cross-entropy method) ----------------------------------------------------------------------

class AdaptiveSampling(Search):
    """Cross-entropy-method rare-event sampling (Rubinstein 1997/1999): a Gaussian fitted to the failures once
    there are min_elite of them (value ≥ threshold), else to the min_elite best; a share `explore` of each batch
    uniform; the covariance floored at min_sd² per dimension."""

    method = 'adaptive'

    def __init__(self, space, seed: int = 0, threshold: float = 0.5, population: int = 10,
                 n_init: int | None = None, min_elite: int = 5, explore: float = 0.1, min_sd: float = 0.02,
                 history=None):
        self.threshold = float(threshold)
        self.population = int(population)
        self.n_init = n_init
        self.min_elite = int(min_elite)
        self.explore = float(explore)
        self.min_sd = float(min_sd)
        super().__init__(space, seed, history)

    def _options(self):
        return {'threshold': self.threshold, 'population': self.population, 'n_init': self.n_init,
                'min_elite': self.min_elite, 'explore': self.explore, 'min_sd': self.min_sd}

    def _init(self):
        self.default_batch = self.population
        self._n_init = int(self.n_init or 2 * self.population)

    def distribution(self):
        """(mean, covariance) the next samples come from, or None while still sampling uniformly."""
        if len(self.history) < max(self._n_init, self.min_elite):
            return None
        X, y = self.X(), self.y()
        fail = y >= self.threshold
        elite = X[fail] if fail.sum() >= self.min_elite else X[np.argsort(-y, kind='stable')[:self.min_elite]]
        m = elite.mean(axis=0)
        C = np.cov(elite.T, bias=True).reshape(self.dim, self.dim) + self.min_sd ** 2 * np.eye(self.dim)
        return m, C

    def _propose(self, n):
        dist = self.distribution()
        out = np.empty((n, self.dim))
        for r in range(n):
            if dist is None or self.rng.random() < self.explore:
                out[r] = self.rng.random(self.dim)
            else:
                Lc = np.linalg.cholesky(dist[1])
                out[r] = _reflect(dist[0] + Lc @ self.rng.standard_normal(self.dim))
        return out


METHODS = {'random': RandomSearch, 'cmaes': CMAES, 'bayesopt': BayesOpt, 'adaptive': AdaptiveSampling}


def make(method: str, space, seed: int = 0, threshold: float | None = None, **opts) -> Search:
    """A searcher by name; the failure threshold goes to the methods that use it (adaptive)."""
    cls = METHODS[method]
    if cls is AdaptiveSampling and threshold is not None:
        opts.setdefault('threshold', threshold)
    return cls(space, seed=seed, **opts)


# --- analysis ------------------------------------------------------------------------------------------------------

def failures(history, threshold: float) -> list:
    """The evaluations with value ≥ threshold, most critical first (ties: earliest first)."""
    return sorted((h for h in history if h['value'] >= threshold), key=lambda h: (-h['value'], h['i']))


def failure_map(history, space, x_name: str, y_name: str, bins=10, threshold: float | None = None) -> dict:
    """The history binned on two parameters: {'x_name', 'y_name', 'x_edges', 'y_edges', 'x_labels', 'y_labels',
    'count', 'max', 'mean', 'failure_share'}; grids are rows of y bins × columns of x bins, None where empty (and
    failure_share None without a threshold). A choices parameter gets one bin per choice (edges are its indices)."""
    space = normalise_space(space)
    names = [p['name'] for p in space]
    dx, dy = names.index(x_name), names.index(y_name)
    bx, by = (bins, bins) if isinstance(bins, int) else bins

    def axis(p, b):
        if 'choices' in p:
            k = len(p['choices'])
            return k, [float(i) for i in range(k + 1)], list(p['choices'])
        return b, [p['lo'] + (p['hi'] - p['lo']) * i / b for i in range(b + 1)], None

    bx, xe, xl = axis(space[dx], bx)
    by, ye, yl = axis(space[dy], by)
    cnt = np.zeros((by, bx), int)
    mx = np.full((by, bx), -np.inf)
    sm = np.zeros((by, bx))
    nf = np.zeros((by, bx), int)
    for h in history:
        u = to_unit(space, h['params'])
        i, j = min(int(u[dy] * by), by - 1), min(int(u[dx] * bx), bx - 1)
        v = float(h['value'])
        cnt[i, j] += 1
        mx[i, j] = max(mx[i, j], v)
        sm[i, j] += v
        nf[i, j] += threshold is not None and v >= threshold

    def grid(a):
        return [[None if cnt[i, j] == 0 else float(a[i, j]) for j in range(bx)] for i in range(by)]

    return {'x_name': x_name, 'y_name': y_name, 'x_edges': xe, 'y_edges': ye, 'x_labels': xl, 'y_labels': yl,
            'count': cnt.tolist(), 'max': grid(mx), 'mean': grid(sm / np.maximum(cnt, 1)),
            'failure_share': grid(nf / np.maximum(cnt, 1)) if threshold is not None else None,
            'threshold': threshold}


def run(searcher: Search, objective, budget: int) -> list:
    """Evaluate up to budget parameter sets in the searcher's natural batches; the values in evaluation order."""
    vals = []
    while len(vals) < budget:
        ps = searcher.ask(min(searcher.default_batch, budget - len(vals)))
        ys = [float(objective(p)) for p in ps]
        searcher.tell(ps, ys)
        vals.extend(ys)
    return vals


def compare(objective, space, budget: int = 120, seeds=(0, 1, 2, 3, 4), threshold: float = 0.5,
            methods=None, n_failures: int = 5) -> dict:
    """Each method with the same budget on each seed: per method {'failures': mean failures found after 1..budget
    evaluations, 'best': mean best value after 1..budget, 'first_failure' and 'to_n_failures': per seed the
    evaluation (1-based) at which the first / the n_failures-th failure was found, or None, their means
    'first_failure_mean' and 'to_n_failures_mean' (a seed without one counts budget + 1: censored), 'found_share',
    'failures_at_budget' (mean), 'failures_per_seed'}. methods: names of METHODS (default all of them)."""
    out = {'budget': int(budget), 'seeds': list(seeds), 'threshold': threshold, 'n_failures': int(n_failures),
           'methods': {}}

    def at(f, k):
        return int(np.argmax(f >= k)) + 1 if f[-1] >= k else None

    def mean(v):
        return float(np.mean([budget + 1 if e is None else e for e in v]))

    for name in methods or list(METHODS):
        fc, bc = [], []
        for s in seeds:
            vals = np.asarray(run(make(name, space, s, threshold), objective, budget))
            fc.append(np.cumsum(vals >= threshold))
            bc.append(np.maximum.accumulate(vals))
        first, to_n = [at(f, 1) for f in fc], [at(f, n_failures) for f in fc]
        out['methods'][name] = {
            'failures': np.mean(fc, axis=0).tolist(), 'best': np.mean(bc, axis=0).tolist(),
            'first_failure': first, 'first_failure_mean': mean(first),
            'to_n_failures': to_n, 'to_n_failures_mean': mean(to_n),
            'found_share': float(np.mean([v is not None for v in first])),
            'failures_at_budget': float(np.mean([c[-1] for c in fc])),
            'failures_per_seed': [int(c[-1]) for c in fc]}
    return out


# --- a known-weak synthetic model ----------------------------------------------------------------------------------

PLANTED_SPACE = [{'name': 'fog_density', 'lo': 0.0, 'hi': 100.0},
                 {'name': 'target_distance_m', 'lo': 5.0, 'hi': 80.0},
                 {'name': 'sun_altitude_deg', 'lo': -10.0, 'hi': 90.0},
                 {'name': 'occlusion', 'lo': 0.0, 'hi': 1.0}]
# the planted weakness in unit coordinates: dense fog, mid range, low sun, partly occluded (synthetic)
PLANTED_REGION = {'center': [0.78, 0.32, 0.25, 0.60], 'radii': [0.20, 0.30, 0.25, 0.40]}


class PlantedObjective:
    """A synthetic detector-miss probability (not a model of any real detector): with d the ellipsoidal distance
    from the planted centre in unit coordinates (d = 1 on the region's boundary),
        p = lo + (hi - lo) / (1 + a·d²)², a = √((hi - lo)/(threshold - lo)) - 1,
    so p = threshold exactly on the boundary, p → hi at the centre, smooth everywhere and decaying to lo far away;
    plus noise N(0, noise²) that is a deterministic function of (seed, parameters), clipped to [0, 1]."""

    def __init__(self, region=None, space=None, seed: int = 0, noise: float = 0.01, threshold: float = 0.5,
                 lo: float = 0.02, hi: float = 0.98):
        self.space = normalise_space(space or PLANTED_SPACE)
        r = region or PLANTED_REGION
        self.center, self.radii = np.asarray(r['center'], float), np.asarray(r['radii'], float)
        if self.center.shape != (len(self.space),) or self.radii.shape != self.center.shape:
            raise ValueError('the region needs a centre and radii per dimension of the space')
        self.seed, self.noise, self.threshold, self.lo, self.hi = int(seed), float(noise), float(threshold), lo, hi
        self.a = math.sqrt((hi - lo) / (threshold - lo)) - 1.0
        n = len(self.space)
        # the volume share of the region (the ellipsoid, clipped to the cube by sampling only if it sticks out)
        ball = math.pi ** (n / 2) / math.gamma(n / 2 + 1) * float(np.prod(self.radii))
        inside_cube = np.all((self.center - self.radii >= 0) & (self.center + self.radii <= 1))
        if inside_cube:
            self.volume = ball
        else:
            u = np.random.default_rng(12345).random((200000, n))
            self.volume = float(np.mean(self.distance(u) <= 1.0))

    def distance(self, X) -> np.ndarray:
        return np.sqrt(np.sum(((np.atleast_2d(X) - self.center) / self.radii) ** 2, axis=-1))

    def clean(self, params: dict) -> float:
        d = float(self.distance(to_unit(self.space, params))[0])
        return self.lo + (self.hi - self.lo) / (1.0 + self.a * d * d) ** 2

    def inside(self, params: dict) -> bool:
        return bool(self.distance(to_unit(self.space, params))[0] <= 1.0)

    def __call__(self, params: dict) -> float:
        x = to_unit(self.space, params)
        h = hashlib.sha256(json.dumps([self.seed, np.round(x, 12).tolist()]).encode()).digest()
        eps = np.random.default_rng(int.from_bytes(h[:8], 'little')).standard_normal()
        return float(min(max(self.clean(params) + self.noise * eps, 0.0), 1.0))


def planted_objective(region=None, space=None, seed: int = 0, noise: float = 0.01,
                      threshold: float = 0.5) -> PlantedObjective:
    """The synthetic known-weak model (see PlantedObjective); its .space, .volume (share of the cube where the clean
    value exceeds the threshold), .inside(params) and .threshold describe the planted region."""
    return PlantedObjective(region, space, seed, noise, threshold)
