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

summary_table(results): rows {'metric', 'value', 'ci_low', 'ci_high', 'n'} for a UI table from a bootstrap or
paired_bootstrap result, or from a dict of named numbers and results.

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
    'n_units', 'n_items', 'n_boot', 'seed'."""
    if method not in ('percentile', 'bca'):
        raise ValueError("method is 'percentile' or 'bca'")
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
        lo, hi, sd, used = _interval(vals[:, j], est[k], ci, method, None if jack is None else jack[:, j])
        per[k] = {'value': None if math.isnan(est[k]) else est[k], 'ci_low': lo, 'ci_high': hi, 'sd': sd,
                  'n_valid': int(np.isfinite(vals[:, j]).sum()), 'method': used}
    meta = {'ci': ci, 'method': method, 'unit': unit, 'n_units': len(groups), 'n_items': len(items), 'n_boot': n,
            'seed': seed}
    if not isinstance(point, Mapping):
        return {**per[''], **meta}
    return {'metrics': per, **meta}


def paired_bootstrap(metric_fn, items, model_a, model_b, n: int = 1000, seed: int = 0, ci: float = 0.95,
                     unit: str | None = None, unit_key: str = 'drive') -> dict:
    """Bootstrap of the difference metric_fn(items, model_b) − metric_fn(items, model_a) on shared resamples. A number
    gives {'a', 'b', 'diff', 'ci_low', 'ci_high', 'sd', 'p_b_gt_a', 'p_b_eq_a', 'n_valid', ...}; a dict gives
    {'metrics': {key: {...}}, ...}; both with 'ci', 'unit', 'n_units', 'n_items', 'n_boot', 'seed'."""
    if not 0 < ci < 1:
        raise ValueError('ci must lie in (0, 1)')
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
    for j, k in enumerate(keys):
        d = vb[:, j] - va[:, j]
        ok = np.isfinite(d)
        est = eb[k] - ea[k]
        lo, hi, sd, _ = _interval(d, est, ci, 'percentile', None)
        per[k] = {'a': None if math.isnan(ea[k]) else ea[k], 'b': None if math.isnan(eb[k]) else eb[k],
                  'diff': None if math.isnan(est) else est, 'ci_low': lo, 'ci_high': hi, 'sd': sd,
                  'p_b_gt_a': float(np.mean(d[ok] > 0)) if ok.any() else None,
                  'p_b_eq_a': float(np.mean(d[ok] == 0)) if ok.any() else None, 'n_valid': int(ok.sum())}
    meta = {'ci': ci, 'unit': unit, 'n_units': len(groups), 'n_items': len(items), 'n_boot': n, 'seed': seed}
    if not isinstance(pa, Mapping):
        return {**per[''], **meta}
    return {'metrics': per, **meta}


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
