"""Uncertainty of a metric (numpy + standard library): bootstrap confidence intervals, paired comparisons of two models,
and the rows a results table shows.

bootstrap(metric_fn, items, ...): metric_fn maps a list of items (images, samples, sequences, or the per-image
records of coco_match / nuscenes_match, which make a resample cheap) to a number or a dict of numbers. The units
resampled with replacement are the items themselves (unit='image') or the groups of items sharing item[unit_key]
(unit='drive': a cluster bootstrap, Davison & Hinkley 1997, Bootstrap Methods and their Application, §3.8; frames of
one drive are correlated, and resampling them as if independent gives intervals that are too narrow). By default
the unit is 'drive' when every item carries a unit_key, else 'image'. Each of n resamples draws as many units as
there are; the metric of the resample is one draw of the estimate's distribution.
- The point estimate is metric_fn(items); sd = the standard deviation of the resampled values (the bootstrap
  standard error).
- CI 'percentile' (Efron & Tibshirani 1993, An Introduction to the Bootstrap, §13.3): the (1 − ci)/2 and (1 + ci)/2
  quantiles of the resampled values (numpy's linear quantiles). 'bca' (ibid. §14.3): bias-corrected and accelerated,
  z0 = Φ⁻¹(share of resampled values below the estimate, ties counted half), acceleration from the jackknife over
  units (leave one unit out), quantiles Φ(z0 + (z0 + z_q) / (1 − a (z0 + z_q))); it falls back to the percentile
  interval when z0 is infinite.
- Resamples where the metric is undefined (None or NaN, e.g. no GT drawn) are dropped; n_valid counts the others.
paired_bootstrap(metric_fn, items, model_a, model_b, ...): metric_fn(items, model) evaluates one model on items (a
model being whatever metric_fn understands: a key of the detections, a callable, ...). Both models are evaluated on the
same resamples, so the variation the items share cancels out of the difference B − A (a paired design): its point
estimate, CI, sd, and the share of resamples where B > A (for a metric where higher is better, the bootstrap's
probability that B is better; Koehn 2004, Statistical significance tests for machine translation evaluation, EMNLP,
as used for model comparison).
Seeded with numpy's default_rng(seed): the same inputs give the same numbers.

With method='expanded' (bootstrap and paired_bootstrap) the interval is the expanded percentile interval (Hesterberg
2015, What Teachers Should Know About the Bootstrap, The American Statistician 69(4)): the quantiles taken at
Φ(± √(n/(n−1)) · t_{n−1, (1+ci)/2}) instead of (1 ± ci)/2, n the number of units — it widens the percentile
interval by the factors a t interval has over a z interval, which removes most of the percentile interval's
under-coverage with few units. Each paired result also carries p_value: the two-sided p-value that inverts the
interval used (the largest 1 − ci at which the interval still holds 0), from the share of resamples on the far side
of 0 (ties counted half); with no resample beyond 0 it is computed at half a resample and is an upper bound.

summary_table(results): rows {'metric', 'value', 'ci_low', 'ci_high', 'n'} for a UI table from a bootstrap or
paired_bootstrap result, or from a dict of named numbers and results.

Planning and many contrasts (core.ablation): t_cdf / t_ppf (Student's t, central and noncentral), paired_power(n,
dz, alpha) — the power of the two-sided paired t-test with n pairs at the standardised effect dz = mean difference /
sd of the differences (Cohen 1988, Statistical Power Analysis for the Behavioral Sciences, §2.3.5: the noncentral t
with n − 1 degrees of freedom and noncentrality dz √n, computed exactly by numerical integration over the chi
distribution, not by the normal approximation) — and paired_sample_size(dz, alpha, power), the smallest n reaching it.
holm(p_values): Holm's step-down adjusted p-values (Holm 1979, A simple sequentially rejective multiple test
procedure, Scand. J. Statistics 6), which control the family-wise error rate at alpha for any dependence between the
contrasts and reject everything Bonferroni rejects.

Limits. The bootstrap assumes the units are exchangeable draws from what the results should generalise to (the same
towns, weathers and scenario mix); with few units (under about 20 drives) its intervals are too narrow, and the
percentile CI of a bounded metric near its bound (AP near 0 or 1) is biased (BCa corrects part of it).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def _units(items, unit, unit_key):
    if unit is None:
        unit = 'drive' if items and all(isinstance(it, Mapping) and it.get(unit_key) is not None
                                         for it in items) else 'image'
    if unit == 'image':
        return unit, [[i] for i in range(len(items))]
    if unit == 'drive':
        groups = {}
        for i, it in enumerate(items):
            if not isinstance(it, Mapping) or it.get(unit_key) is None:
                raise ValueError(f'unit is drive but item {i} has no {unit_key!r}')
            groups.setdefault(it[unit_key], []).append(i)
        return unit, list(groups.values())
    raise ValueError("unit is 'image', 'drive' or None")


def _scalars(v) -> dict:
    """A metric's value as {key: float}: '' for a bare number, else the dict's numeric top-level entries (None and
    non-finite numbers as NaN; nested dicts, lists, strings and booleans left out)."""
    def num(x):
        if x is None:
            return math.nan
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
            return None
        return float(x)
    if isinstance(v, Mapping):
        out = {}
        for k, x in v.items():
            f = num(x)
            if f is not None:
                out[k] = f
        return out
    f = num(v)
    if f is None:
        raise TypeError(f'metric_fn must return a number or a dict of numbers; got {type(v).__name__}')
    return {'': f}


def _resample(items, groups, rng):
    pick = rng.integers(0, len(groups), len(groups))
    return [items[i] for u in pick for i in groups[u]]


def _interval(vals: np.ndarray, est: float, ci: float, method: str, jack: np.ndarray | None):
    ok = vals[np.isfinite(vals)]
    if not ok.size or not math.isfinite(est):
        return None, None, None, 'percentile'
    lo_q, hi_q = (1 - ci) / 2, (1 + ci) / 2
    used = 'percentile'
    if method == 'bca' and jack is not None:
        share = (np.sum(ok < est) + 0.5 * np.sum(ok == est)) / ok.size
        jk = jack[np.isfinite(jack)]
        if 0 < share < 1 and jk.size > 1:
            z0 = _N.inv_cdf(share)
            d = jk.mean() - jk
            den = 6.0 * np.sum(d ** 2) ** 1.5
            a = float(np.sum(d ** 3) / den) if den > 0 else 0.0
            q = []
            for p in (lo_q, hi_q):
                z = _N.inv_cdf(p)
                q.append(_N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
            lo_q, hi_q = q
            used = 'bca'
    lo, hi = np.quantile(ok, [lo_q, hi_q])
    sd = float(np.std(ok, ddof=1)) if ok.size > 1 else 0.0
    return float(lo), float(hi), sd, used


def _jackknife(metric_fn, items, groups, keys):
    out = np.full((len(groups), len(keys)), np.nan)
    for g in range(len(groups)):
        drop = set(groups[g])
        v = _scalars(metric_fn([it for i, it in enumerate(items) if i not in drop]))
        out[g] = [v.get(k, math.nan) for k in keys]
    return out


def bootstrap(metric_fn, items, n: int = 1000, seed: int = 0, ci: float = 0.95, unit: str | None = None,
              unit_key: str = 'drive', method: str = 'percentile') -> dict:
    """Bootstrap CI of metric_fn(items). A number gives {'value', 'ci_low', 'ci_high', 'sd', 'n_valid', ...}; a dict
    gives {'metrics': {key: {'value', 'ci_low', 'ci_high', 'sd', 'n_valid'}}, ...}; both with 'ci', 'method', 'unit',
    'n_units', 'n_items', 'n_boot', 'seed'. method: 'percentile', 'bca' or 'expanded' (the module docstring)."""
    if method not in ('percentile', 'bca', 'expanded'):
        raise ValueError("method is 'percentile', 'bca' or 'expanded'")
    if not 0 < ci < 1:
        raise ValueError('ci must lie in (0, 1)')
    items = list(items)
    unit, groups = _units(items, unit, unit_key)
    if not groups:
        raise ValueError('no items')
    point = metric_fn(items)
    est = _scalars(point)
    keys = list(est)
    rng = np.random.default_rng(seed)
    vals = np.full((n, len(keys)), np.nan)
    for b in range(n):
        v = _scalars(metric_fn(_resample(items, groups, rng)))
        vals[b] = [v.get(k, math.nan) for k in keys]
    jack = _jackknife(metric_fn, items, groups, keys) if method == 'bca' else None
    per = {}
    for j, k in enumerate(keys):
        if method == 'expanded' and len(groups) > 1:
            lo, hi, sd = _expanded(vals[:, j], est[k], ci, len(groups))
            used = 'expanded'
        else:
            lo, hi, sd, used = _interval(vals[:, j], est[k], ci, method, None if jack is None else jack[:, j])
        per[k] = {'value': None if math.isnan(est[k]) else est[k], 'ci_low': lo, 'ci_high': hi, 'sd': sd,
                  'n_valid': int(np.isfinite(vals[:, j]).sum()), 'method': used}
    meta = {'ci': ci, 'method': method, 'unit': unit, 'n_units': len(groups), 'n_items': len(items), 'n_boot': n,
            'seed': seed}
    if not isinstance(point, Mapping):
        return {**per[''], **meta}
    return {'metrics': per, **meta}


def paired_bootstrap(metric_fn, items, model_a, model_b, n: int = 1000, seed: int = 0, ci: float = 0.95,
                     unit: str | None = None, unit_key: str = 'drive', method: str = 'percentile') -> dict:
    """Bootstrap of the difference metric_fn(items, model_b) − metric_fn(items, model_a) on shared resamples. A number
    gives {'a', 'b', 'diff', 'ci_low', 'ci_high', 'sd', 'p_b_gt_a', 'p_b_eq_a', 'p_value', 'n_valid', ...}; a dict
    gives {'metrics': {key: {...}}, ...}; both with 'ci', 'method', 'unit', 'n_units', 'n_items', 'n_boot', 'seed'.
    method: 'percentile' or 'expanded' (the module docstring)."""
    if not 0 < ci < 1:
        raise ValueError('ci must lie in (0, 1)')
    if method not in ('percentile', 'expanded'):
        raise ValueError("method is 'percentile' or 'expanded'")
    items = list(items)
    unit, groups = _units(items, unit, unit_key)
    if not groups:
        raise ValueError('no items')
    pa, pb = metric_fn(items, model_a), metric_fn(items, model_b)
    ea, eb = _scalars(pa), _scalars(pb)
    keys = [k for k in ea if k in eb]
    rng = np.random.default_rng(seed)
    va = np.full((n, len(keys)), np.nan)
    vb = np.full((n, len(keys)), np.nan)
    for b in range(n):
        s = _resample(items, groups, rng)
        xa, xb = _scalars(metric_fn(s, model_a)), _scalars(metric_fn(s, model_b))
        va[b] = [xa.get(k, math.nan) for k in keys]
        vb[b] = [xb.get(k, math.nan) for k in keys]
    per = {}
    n_u = len(groups)
    for j, k in enumerate(keys):
        d = vb[:, j] - va[:, j]
        ok = np.isfinite(d)
        est = eb[k] - ea[k]
        if method == 'expanded' and n_u > 1:
            lo, hi, sd = _expanded(d, est, ci, n_u)
        else:
            lo, hi, sd, _ = _interval(d, est, ci, 'percentile', None)
        p_gt = float(np.mean(d[ok] > 0)) if ok.any() else None
        p_eq = float(np.mean(d[ok] == 0)) if ok.any() else None
        per[k] = {'a': None if math.isnan(ea[k]) else ea[k], 'b': None if math.isnan(eb[k]) else eb[k],
                  'diff': None if math.isnan(est) else est, 'ci_low': lo, 'ci_high': hi, 'sd': sd,
                  'p_b_gt_a': p_gt, 'p_b_eq_a': p_eq,
                  'p_value': _p_value(p_gt, p_eq, int(ok.sum()), n_u, method) if p_gt is not None else None,
                  'n_valid': int(ok.sum())}
    meta = {'ci': ci, 'method': method, 'unit': unit, 'n_units': n_u, 'n_items': len(items), 'n_boot': n,
            'seed': seed}
    if not isinstance(pa, Mapping):
        return {**per[''], **meta}
    return {'metrics': per, **meta}


def _expanded(vals: np.ndarray, est: float, ci: float, n_units: int):
    """Hesterberg's expanded percentile interval of the resampled values (the module docstring)."""
    ok = vals[np.isfinite(vals)]
    if not ok.size or not math.isfinite(est):
        return None, None, None
    c = math.sqrt(n_units / (n_units - 1)) * t_ppf((1 + ci) / 2, n_units - 1)
    lo_q = _N.cdf(-c)
    lo, hi = np.quantile(ok, [lo_q, 1 - lo_q])
    return float(lo), float(hi), float(np.std(ok, ddof=1)) if ok.size > 1 else 0.0


def _p_value(p_gt: float, p_eq: float, n_valid: int, n_units: int, method: str) -> float:
    """The two-sided p-value inverting the interval: s, the share of resamples beyond 0 on the side away from the
    estimate (ties half; at least half a resample), is the tail the interval's end would sit at."""
    s = min(p_gt + 0.5 * p_eq, 1.0 - p_gt - 0.5 * p_eq)
    s = max(s, 0.5 / max(n_valid, 1))
    if method == 'expanded' and n_units > 1:
        c = math.sqrt(n_units / (n_units - 1))
        return float(min(1.0, 2.0 * (1.0 - t_cdf(_N.inv_cdf(1.0 - s) / c, n_units - 1))))
    return float(min(1.0, 2.0 * s))


# ================================================================================================ planning
def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the regularised incomplete beta function (modified Lentz; Press et al., Numerical
    Recipes, 3rd ed., §6.4)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def _phi(x: np.ndarray) -> np.ndarray:
    """The standard normal CDF, vectorised (erfc by Press et al.'s Chebyshev fit, fractional error < 1.2e-7)."""
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.5 * z)
    erfc = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, 1.0 - 0.5 * erfc, 0.5 * erfc)


def t_cdf(t: float, df: float, ncp: float = 0.0) -> float:
    """P(T ≤ t) for Student's t with df degrees of freedom, noncentral with ncp: the central one through the
    regularised incomplete beta function; the noncentral one as E[Φ(t·S − ncp)], S = √(χ²_df / df), integrated over
    S's density (midpoint rule on 6000 points over its mass)."""
    if df <= 0:
        raise ValueError('df must be positive')
    if ncp == 0.0:
        x = df / (df + t * t)
        tail = 0.5 * _betainc(df / 2.0, 0.5, x)
        return 1.0 - tail if t > 0 else tail
    sig = 1.0 / math.sqrt(2.0 * df)
    lo, hi = max(0.0, 1.0 - 12.0 * sig - 0.5), 1.0 + 12.0 * sig + 5.0
    k = 6000
    s = lo + (np.arange(k) + 0.5) * (hi - lo) / k
    v = df * s * s                                              # S = √(V/df): f_S(s) = f_χ²(df s²) · 2 df s
    logf = ((df / 2.0 - 1.0) * np.log(v) - v / 2.0 - (df / 2.0) * math.log(2.0) - math.lgamma(df / 2.0)
            + math.log(2.0 * df) + np.log(s))
    w = np.exp(logf) * (hi - lo) / k
    return float(np.sum(w * _phi(t * s - ncp)) / np.sum(w))


def t_ppf(q: float, df: float) -> float:
    """The q-quantile of the central t with df degrees of freedom (bisection on t_cdf to 1e-10)."""
    if not 0.0 < q < 1.0:
        raise ValueError('q must lie in (0, 1)')
    if q == 0.5:
        return 0.0
    if q < 0.5:
        return -t_ppf(1.0 - q, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2.0


def paired_power(n: int, dz: float, alpha: float = 0.05) -> float:
    """The power of the two-sided paired t-test at level alpha with n pairs and standardised effect dz."""
    if n < 2:
        return 0.0
    df = n - 1
    tc = t_ppf(1.0 - alpha / 2.0, df)
    ncp = abs(dz) * math.sqrt(n)
    return float(1.0 - t_cdf(tc, df, ncp) + t_cdf(-tc, df, ncp))


def paired_sample_size(dz: float, alpha: float = 0.05, power: float = 0.8, n_max: int = 100000) -> dict:
    """The smallest number of pairs n whose two-sided paired t-test at level alpha reaches the power at effect dz:
    {'n', 'power' (exact at n), 'n_normal' (the normal approximation it started from), 'capped' (n_max reached)}."""
    if not math.isfinite(dz) or dz == 0:
        raise ValueError('the effect size must be a finite number other than 0')
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError('alpha and power must lie in (0, 1)')
    za, zb = _N.inv_cdf(1.0 - alpha / 2.0), _N.inv_cdf(power)
    n0 = max(2, math.ceil(((za + zb) / abs(dz)) ** 2 + za * za / 2.0))      # with Guenther's (1981) correction
    n = max(2, min(n0 - 3, n_max))
    while n < n_max and paired_power(n, dz, alpha) < power:
        n += 1
    while n > 2 and paired_power(n - 1, dz, alpha) >= power:
        n -= 1
    return {'n': n, 'power': paired_power(n, dz, alpha), 'n_normal': n0, 'capped': n >= n_max}


def holm(p_values) -> list:
    """Holm's step-down adjusted p-values, in the given order (None stays None)."""
    idx = [i for i, p in enumerate(p_values) if p is not None]
    m = len(idx)
    order = sorted(idx, key=lambda i: p_values[i])
    out = [None] * len(p_values)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * float(p_values[i])))
        out[i] = run
    return out


def _row(name, r, n):
    if isinstance(r, Mapping) and 'diff' in r:
        return {'metric': f'{name} (B − A)', 'value': r['diff'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
                'n': r.get('n_units', n)}
    if isinstance(r, Mapping) and 'value' in r:
        return {'metric': name, 'value': r['value'], 'ci_low': r.get('ci_low'), 'ci_high': r.get('ci_high'),
                'n': r.get('n_units', n)}
    f = _scalars({'x': r}).get('x')
    if f is None:
        return None
    return {'metric': name, 'value': None if math.isnan(f) else f, 'ci_low': None, 'ci_high': None, 'n': n}


def summary_table(results, n: int | None = None) -> list:
    """Rows {'metric', 'value', 'ci_low', 'ci_high', 'n'} (None where there is no CI) from a bootstrap() or
    paired_bootstrap() result, or a dict {name: number | such a result | a dict-form result}; n is the units the
    numbers rest on (a result's own n_units wins). Entries that are not numbers (lists, nested dicts) are skipped."""
    if isinstance(results, Mapping) and 'metrics' in results and 'n_units' in results:
        return [_row(k, r, results['n_units']) for k, r in results['metrics'].items()]
    if isinstance(results, Mapping) and ('value' in results or 'diff' in results) and 'n_units' in results:
        return [_row('metric', results, n)]
    rows = []
    for name, r in results.items():
        if isinstance(r, Mapping) and 'metrics' in r and 'n_units' in r:
            rows += [_row(f'{name}.{k}', v, r['n_units']) for k, v in r['metrics'].items()]
            continue
        if isinstance(r, Mapping) and not ('value' in r or 'diff' in r):
            continue
        row = _row(str(name), r, n)
        if row is not None:
            rows.append(row)
    return rows
