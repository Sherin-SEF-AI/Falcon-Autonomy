"""Latency in the loop: measured compute latency (spikes included) fed back into closed-loop simulation, and each
scenario's latency budget — the largest latency at which it still passes (docs/latency-loop.md is the manual; the
standard library at import, so the workers in CARLA's Python can load the delay line).

    falcon latency profile (--from-hil RUN | --from-deploy DIR | --from-file F | --synthetic SPEC) --out FILE
    falcon latency run PROJECT --scenario K (--profile FILE [--scale S] | --constant MS) …   one run, with and without
    falcon latency budget PROJECT --scenario K [--scenario K2 …] (--constant | --profile FILE) …
    falcon latency show FILE                          a profile or a latency-budget.json

PROFILES (falcon.latency_profile/1). A profile is an empirical distribution of latency in milliseconds: its sorted
samples (thinned to at most MAX_SAMPLES evenly spaced order statistics, the minimum and maximum kept), with where it
came from recorded in `source`:

    hil      a hardware-in-the-loop session's latency.jsonl (hil/report.py, core.hil.MEASURES): by default the
             `on_device` measure (camera received → control sent, on the device's own clock), or e2e, processing,
             round_trip — MEASURED, on the device the session ran on (a simulated device is said to be one)
    deploy   a deployment's engine profile (core.deploy / hil/engine_device.py profile.json: the timed runs'
             latency_ms) — MEASURED on the Orin, or on this machine's CPU when the device was simulated (said so)
    file     numbers from a JSON list, {"samples_ms": [...]} or a text file with one number per line — as given
    synthetic  constant:MS, or lognormal:median=MS,sigma=S[,spike_p=P,spike_ms=MS][,n=N] (seeded) — NOT measured

Sampling is by inverse CDF with linear interpolation between order statistics: latency = Q(u), u uniform from a
seeded generator. The same seed gives the same u at step k whatever the scale — common random numbers: a run at scale
2 sees exactly twice the latency of the run at scale 1 at every step.

THE DELAY LINE (DelayLine; workers/director.py holds one when a run's config has a `latency` block; default off). At
step k the stack / policy computes a control u_k from the observation at t_k. Its latency L_k = scale × (the constant,
or Q(u_k)) is quantised UP to whole simulation steps, d_k = ceil(L_k / dt): a result that arrives during a tick can
only act from the next tick on (L = 0 → d = 0: applied in the same step, exactly as without the delay line). The
control applied at step j is the most recently COMPUTED one that has arrived (k + d_k ≤ j; a result overtaken by a
newer one is dropped); until a newer one arrives the last applied control is HELD (zero-order hold), and before the
first arrives the vehicle coasts (throttle 0, steer 0, brake 0). mode "pipeline" (default): every observation is
processed (latencies overlap, as a pipelined stack). mode "serial": while a computation is running the observations
of the steps in between are skipped (one inference at a time), so the staleness reaches up to twice the latency.
Where it acts: the scenario director's own ego driver (scenario drives, the stack worker's follower baseline), the
gym environment's step, and the stack worker's ros2 driver (on top of its built-in one-step turnaround). Not wired:
workers/adversary.py, workers/live.py, workers/hil.py (HIL has the real latency) and recon/closedloop.py.
The run's run.json "scenario.latency" reports the delays that were applied.

BUDGET. For each scenario case (a scenario and one parameter set), the latency x is swept on a grid — constant
latency: whole steps (x = d·dt; finer values are indistinguishable after quantisation) up to --max-ms; a profile: its
scale from 0 in steps of --step up to --max-scale — with common random numbers (same seed, same parameters, same
u-sequence), and the largest x that still passes is found by bisection (core.regbisect.Bisector: the case at x = 0
must pass and at the maximum fail, else it is reported as "fails without latency" / "no failure up to the maximum").
Bisection ASSUMES failures are monotone in latency; --exhaustive runs every grid point instead and reports a case
that passes above a failure as non-monotone. The criterion: the scenario's own pass checks (run.json
scenario.passed), or its criticality below --threshold (core.criticality; 1 is a collision). The runs go through
sim.studio.Study on sim.service.study_service (cached by their config: a re-run resumes).

What a profile does NOT capture: the correlation of consecutive latencies (samples are drawn independently: real
spikes cluster), thermal throttling and power-mode changes unless the profile was measured under them, the latency's
dependence on the scene (more objects, slower), and anything outside the measured interval (sensor exposure, the
actuator's own lag). The simulation step quantises every latency up: at 20 Hz, 1 ms and 50 ms are the same delay.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

PROFILE_SCHEMA = 'falcon.latency_profile/1'
BUDGET_SCHEMA = 'falcon.latency_budget/1'
RUN_SCHEMA = 'falcon.latency_run/1'
MAX_SAMPLES = 1024
MODES = ('pipeline', 'serial')
HIL_MEASURES = ('on_device', 'e2e', 'processing', 'round_trip')
LAT_KEY = 'latency.x'
COAST = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0}


class LatencyError(ValueError):
    pass


# ================================================================================================ profiles
def quantile(sorted_ms: list, u: float) -> float:
    """The empirical quantile by linear interpolation between order statistics (u in [0, 1])."""
    n = len(sorted_ms)
    if n == 1:
        return float(sorted_ms[0])
    q = min(max(u, 0.0), 1.0) * (n - 1)
    i = min(int(q), n - 2)
    return float(sorted_ms[i] + (sorted_ms[i + 1] - sorted_ms[i]) * (q - i))


def stats(sorted_ms: list) -> dict:
    n = len(sorted_ms)
    mean = sum(sorted_ms) / n
    return {'n': n, 'min': float(sorted_ms[0]), 'median': quantile(sorted_ms, 0.5), 'p95': quantile(sorted_ms, 0.95),
            'p99': quantile(sorted_ms, 0.99), 'max': float(sorted_ms[-1]), 'mean': mean}


def make_profile(values, source: dict, name: str = '') -> dict:
    """A profile from latency values in ms (negative, NaN and infinite ones are refused)."""
    from .provenance import utc_now
    vals = []
    for v in values:
        v = float(v)
        if not math.isfinite(v) or v < 0:
            raise LatencyError(f'a latency of {v!r} ms: latencies are finite and not negative')
        vals.append(v)
    if not vals:
        raise LatencyError('no latency values')
    vals.sort()
    st = stats(vals)                                     # of every value given, before thinning
    if len(vals) > MAX_SAMPLES:
        m = MAX_SAMPLES
        vals = [vals[round(i * (len(vals) - 1) / (m - 1))] for i in range(m)]
    doc = {'schema': PROFILE_SCHEMA, 'name': name or source.get('kind', 'profile'), 'unit': 'ms', 'created': utc_now(),
           'source': source, 'measured': source.get('kind') in ('hil', 'deploy'), 'stats': st, 'samples_ms': vals,
           'not_captured': ['the correlation of consecutive latencies (drawn independently)',
                            'thermal throttling and power modes other than those it was measured under',
                            'the latency\'s dependence on the scene']}
    doc['sha256'] = profile_sha(doc)
    return doc


def profile_sha(doc: dict) -> str:
    from .provenance import sha256_json
    return sha256_json({'samples_ms': doc['samples_ms']})


def load_profile(path: str) -> dict:
    try:
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise LatencyError(f'cannot read the profile {path}: {e}') from None
    if not isinstance(doc, dict) or doc.get('schema') != PROFILE_SCHEMA or not doc.get('samples_ms'):
        raise LatencyError(f'{path} is not a latency profile ({PROFILE_SCHEMA}); make one with falcon latency profile')
    return doc


def profile_from_hil(run_dir: str, measure: str = 'on_device', name: str = '') -> dict:
    """From a HIL session's latency.jsonl (core.hil's stamps), one value per frame that has both stamps."""
    from . import hil as H
    if measure not in HIL_MEASURES:
        raise LatencyError(f'measure: one of {", ".join(HIL_MEASURES)}')
    recs = H.read_jsonl(os.path.join(run_dir, 'latency.jsonl'))
    if not recs:
        raise LatencyError(f'{run_dir} holds no HIL session (latency.jsonl)')
    try:
        with open(os.path.join(run_dir, 'hil.json'), encoding='utf-8') as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        cfg = {}
    vals = [v for v in H.stage_values(recs, {})[measure] if v is not None and v >= 0]
    if not vals:
        raise LatencyError(f'no frame of the session has the {measure} stamps')
    label = next(m[1] for m in H.MEASURES if m[0] == measure)
    mode = (cfg.get('device') or {}).get('mode') or 'unknown'
    src = {'kind': 'hil', 'measure': measure, 'definition': label, 'run': os.path.abspath(run_dir),
           'device': 'a simulated device on this machine' if mode == 'local' else 'the device of the session',
           'device_mode': mode, 'frames': len(recs), 'frames_used': len(vals)}
    return make_profile(vals, src, name or f'hil-{measure}')


def profile_from_deploy(out_dir: str, engine: str = '', name: str = '') -> dict:
    """From a deployment's on-device profiling (core.deploy): the timed runs of one engine (default: the first
    profiled one)."""
    edir = os.path.join(out_dir, 'engines')
    names = sorted(n for n in (os.listdir(edir) if os.path.isdir(edir) else [])
                   if os.path.isfile(os.path.join(edir, n, 'profile.json')))
    if engine and engine not in names:
        raise LatencyError(f'no profiled engine {engine!r} in {out_dir} ({", ".join(names) or "none profiled"})')
    if not names:
        raise LatencyError(f'{out_dir} holds no profiled engine (falcon deploy profile)')
    engine = engine or names[0]
    with open(os.path.join(edir, engine, 'profile.json'), encoding='utf-8') as fh:
        raw = json.load(fh)
    vals = raw.get('latency_ms') or []
    if not vals:
        raise LatencyError(f'the profile of {engine} has no latency samples')
    sim = bool(raw.get('simulated'))
    src = {'kind': 'deploy', 'engine': engine, 'folder': os.path.abspath(out_dir),
           'definition': raw.get('latency_definition') or 'the engine\'s inference latency',
           'device': 'this machine\'s CPU (a simulated device, not the Orin)' if sim else 'the device (Jetson)',
           'simulated_device': sim, 'power_mode': raw.get('power_mode'), 'runs': len(vals),
           'note': 'one model\'s inference only: pre- and post-processing and the rest of the stack are not in it'}
    return make_profile(vals, src, name or f'deploy-{engine}')


def profile_from_file(path: str, name: str = '') -> dict:
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as e:
        raise LatencyError(f'cannot read {path}: {e}') from None
    try:
        doc = json.loads(text)
        vals = doc.get('samples_ms') if isinstance(doc, dict) else doc
    except ValueError:
        try:
            vals = [float(x) for x in text.replace(',', ' ').split()]
        except ValueError:
            raise LatencyError(f'{path}: a JSON list, {{"samples_ms": [...]}} or numbers, one per line') from None
    if not isinstance(vals, list):
        raise LatencyError(f'{path}: a JSON list, {{"samples_ms": [...]}} or numbers, one per line')
    from .provenance import sha256_file
    return make_profile(vals, {'kind': 'file', 'path': os.path.abspath(path), 'file_sha256': sha256_file(path),
                               'device': 'as the file\'s author states (not known to Falcon)'},
                        name or os.path.splitext(os.path.basename(path))[0])


def profile_synthetic(spec: str, seed: int = 0, name: str = '') -> dict:
    """constant:MS | lognormal:median=MS,sigma=S[,spike_p=P,spike_ms=MS][,n=N]. A spike replaces a draw, with
    probability spike_p, by spike_ms."""
    kind, _, rest = spec.partition(':')
    try:
        if kind == 'constant':
            ms = float(rest)
            return make_profile([ms], {'kind': 'synthetic', 'spec': spec, 'device': 'none (synthetic)'},
                                name or f'constant-{ms:g}ms')
        if kind == 'lognormal':
            kv = dict(x.split('=', 1) for x in rest.split(',') if x)
            unknown = set(kv) - {'median', 'sigma', 'spike_p', 'spike_ms', 'n'}
            if unknown:
                raise LatencyError(f'lognormal: unknown field(s) {", ".join(sorted(unknown))}')
            med, sigma = float(kv['median']), float(kv.get('sigma', 0.25))
            p, spike, n = float(kv.get('spike_p', 0.0)), float(kv.get('spike_ms', 0.0)), int(kv.get('n', 2000))
            if not (med > 0 and sigma >= 0 and 0 <= p <= 1 and spike >= 0 and 1 <= n <= 1000000):
                raise LatencyError('lognormal: median > 0, sigma ≥ 0, 0 ≤ spike_p ≤ 1, spike_ms ≥ 0, n ≥ 1')
            rnd = random.Random(f'latency-profile:{seed}')
            vals = [spike if rnd.random() < p else med * math.exp(sigma * rnd.gauss(0.0, 1.0)) for _ in range(n)]
            return make_profile(vals, {'kind': 'synthetic', 'spec': spec, 'seed': int(seed),
                                       'device': 'none (synthetic)'}, name or 'lognormal')
    except (KeyError, ValueError) as e:
        if isinstance(e, LatencyError):
            raise
        raise LatencyError(f'--synthetic {spec!r}: constant:MS or lognormal:median=MS,sigma=S[,spike_p=P,'
                           f'spike_ms=MS][,n=N] ({e})') from None
    raise LatencyError(f'--synthetic {spec!r}: constant:MS or lognormal:median=MS,sigma=S[,spike_p=P,spike_ms=MS]')


def histogram(doc: dict, bins: int = 24) -> dict:
    """{'edges': bins + 1 values in ms, 'counts': per bin} over the profile's (thinned) samples."""
    v = doc['samples_ms']
    lo, hi = float(v[0]), float(v[-1])
    if hi <= lo:
        return {'edges': [lo, lo + max(lo * 0.01, 0.001)], 'counts': [len(v)]}
    w = (hi - lo) / bins
    counts = [0] * bins
    for x in v:
        counts[min(int((x - lo) / w), bins - 1)] += 1
    return {'edges': [lo + i * w for i in range(bins + 1)], 'counts': counts}


# ================================================================================================ the delay line
def latency_block(profile: dict | None = None, constant_ms: float | None = None, scale: float = 1.0,
                  mode: str = 'pipeline', seed: int = 0) -> dict:
    """A run config's `latency` block (self-contained: the profile's samples are in it, so the run's cache key
    follows the latency)."""
    if (profile is None) == (constant_ms is None):
        raise LatencyError('give a profile or a constant latency, one of them')
    if mode not in MODES:
        raise LatencyError(f'mode: one of {", ".join(MODES)}')
    if not (math.isfinite(float(scale)) and float(scale) >= 0):
        raise LatencyError('the latency scale is a number ≥ 0')
    b = {'scale': float(scale), 'mode': mode, 'seed': int(seed)}
    if constant_ms is not None:
        if not (math.isfinite(float(constant_ms)) and float(constant_ms) >= 0):
            raise LatencyError('a constant latency is a number of milliseconds ≥ 0')
        b['constant_ms'] = float(constant_ms)
    else:
        b['profile'] = {'name': profile.get('name'), 'sha256': profile.get('sha256') or profile_sha(profile),
                        'measured': bool(profile.get('measured')), 'samples_ms': list(profile['samples_ms'])}
    return b


class DelayLine:
    """The delay between the computed and the applied control (module doc). block: a run config's `latency` block;
    dt: the simulation step in seconds. step(u) takes the control just computed and returns the control to apply."""

    def __init__(self, block: dict, dt: float, seed: int = 0):
        self.dt_ms = float(dt) * 1000.0
        self.scale = float(block.get('scale', 1.0))
        self.mode = block.get('mode', 'pipeline')
        if self.mode not in MODES:
            raise LatencyError(f'latency mode: one of {", ".join(MODES)}')
        self.constant = block.get('constant_ms')
        self.samples = None if self.constant is not None else list((block.get('profile') or {}).get('samples_ms') or [])
        if self.constant is None and not self.samples:
            raise LatencyError('the latency block needs a profile (samples_ms) or constant_ms')
        self.rnd = random.Random(f'latency:{int(block.get("seed", seed))}')
        self.k = 0
        self.pending = []                      # (arrival step, computed step, control)
        self.applied = dict(COAST)
        self.applied_k = -1                    # the step whose control is being applied (-1: none yet, coasting)
        self.busy_until = 0                    # serial: the step at which the running computation ends
        self.hist = {}                         # delay in steps -> count
        self.sum_ms = self.max_ms = 0.0
        self.held = self.skipped = self.dropped = 0

    def draw_ms(self) -> float:
        u = self.rnd.random()                  # one draw per step, whatever the mode or scale (common random numbers)
        base = float(self.constant) if self.constant is not None else quantile(self.samples, u)
        return self.scale * base

    def step(self, u: dict) -> dict:
        k = self.k
        self.k += 1
        ms = self.draw_ms()
        d = max(int(math.ceil(ms / self.dt_ms - 1e-9)), 0)
        if self.mode == 'serial' and k < self.busy_until:
            self.skipped += 1                  # the stack is still computing: this observation is not processed
        else:
            self.pending.append((k + d, k, dict(u)))
            self.busy_until = k + d
            self.hist[d] = self.hist.get(d, 0) + 1
            self.sum_ms += ms
            self.max_ms = max(self.max_ms, ms)
        best = None
        for item in self.pending:
            if item[0] <= k and (best is None or item[1] > best[1]):
                best = item
        if best is not None and best[1] > self.applied_k:
            self.dropped += sum(1 for it in self.pending if it[1] < best[1])   # arrived or not: never applied now
            self.applied, self.applied_k = best[2], best[1]
            self.pending = [it for it in self.pending if it[1] > best[1]]
        else:
            self.held += 1
        return dict(self.applied)

    def report(self) -> dict:
        n = sum(self.hist.values())
        return {'mode': self.mode, 'scale': self.scale, 'constant_ms': self.constant, 'step_ms': self.dt_ms,
                'steps': self.k, 'computed': n, 'mean_ms': (self.sum_ms / n) if n else None, 'max_ms': self.max_ms,
                'delay_steps': {str(d): c for d, c in sorted(self.hist.items())},
                'mean_delay_steps': (sum(d * c for d, c in self.hist.items()) / n) if n else None,
                'held_steps': self.held, 'skipped_observations': self.skipped, 'overtaken': self.dropped,
                'hold': 'the last applied control is held; coasting before the first arrives'}


# ================================================================================================ the runs
def passes(row: dict, criterion: str = 'auto', threshold: float = 0.5):
    """True / False by the criterion, None when the run did not finish. criterion: 'passed' (the scenario's own
    checks), 'criticality' (its criticality < threshold), 'auto' (the checks when the scenario has them, else the
    criticality)."""
    if not row or row.get('status') != 'ok':
        return None
    sc = row.get('scenario') or {}
    crit = float((sc.get('criticality') or {}).get('criticality', 0.0))
    if criterion == 'criticality' or (criterion == 'auto' and sc.get('passed') is None):
        return crit < threshold
    return bool(sc.get('passed'))


def _study_class():
    from ..sim.studio import Study

    class LatencyStudy(Study):
        """A scenario study whose parameter sets may carry LAT_KEY: the latency x (ms for a constant latency, the
        profile's scale otherwise) of that run; x = 0 or absent runs without the delay line (the same run, and the
        same cache key, as any other study's)."""
        profile = None
        constant = False
        mode = 'pipeline'
        _x = None

        def config(self, values: dict, k: int) -> dict:
            v = dict(values)
            self._x = v.pop(LAT_KEY, None)
            try:
                return super().config(v, k)
            finally:
                self._x = None

        def base_config(self, values: dict) -> dict:
            cfg = super().base_config(values)
            if self._x:
                cfg['latency'] = (latency_block(constant_ms=float(self._x), mode=self.mode, seed=self.seed)
                                  if self.constant else
                                  latency_block(self.profile, scale=float(self._x), mode=self.mode, seed=self.seed))
            return cfg
    return LatencyStudy


def make_study(svc, project, scn: dict, name: str, world: str, rig: str, seed: int, fps: float | None,
               profile: dict | None, mode: str = 'pipeline'):
    st = _study_class()(svc, project, scn, name, world, rig, seed=seed, fps=fps, objective=None)
    st.profile, st.constant, st.mode = profile, profile is None, mode
    return st


def _brief(row: dict | None, criterion: str, threshold: float) -> dict:
    sc = (row or {}).get('scenario') or {}
    return {'status': (row or {}).get('status'), 'run': (row or {}).get('run'),
            'passed': passes(row, criterion, threshold), 'scenario_passed': sc.get('passed'),
            'criticality': (sc.get('criticality') or {}).get('criticality'),
            'latency': sc.get('latency'), 'error': (row or {}).get('error')}


def budget(svc, project, scenarios: list, name: str, world: str, rig: str, *, profile: dict | None = None,
           max_ms: float = 500.0, max_scale: float = 4.0, step: float = 0.25, samples: int = 1, seed: int = 0,
           fps: float | None = None, mode: str = 'pipeline', criterion: str = 'auto', threshold: float = 0.5,
           exhaustive: bool = False, progress=None) -> dict:
    """The latency budget of each scenario case (module doc). profile None: a constant latency swept in whole steps
    up to max_ms; else the profile's scale in `step`s up to max_scale. Returns the latency-budget document (the
    caller writes it)."""
    from . import regbisect as RB
    from . import scenarios as SC
    from .provenance import utc_now
    constant = profile is None
    dt_ms = 1000.0 / float(fps or 20.0)
    if constant:
        unit, gstep = 'ms', dt_ms
        n = int(math.ceil(float(max_ms) / dt_ms - 1e-9))
    else:
        unit, gstep = 'scale', float(step)
        if not gstep > 0:
            raise LatencyError('--step must be positive')
        n = int(math.ceil(float(max_scale) / gstep - 1e-9))
    if n < 1:
        raise LatencyError('the sweep has no latency above 0: raise --max-ms / --max-scale')
    if mode not in MODES:
        raise LatencyError(f'mode: one of {", ".join(MODES)}')

    def x_of(i: int) -> float:
        return round(i * gstep, 9)
    studies, cases = {}, []
    for scn in scenarios:
        st = make_study(svc, project, scn, f'{name}-{scn["id"]}', world, rig, seed, fps, profile, mode)
        studies[scn['id']] = st
        for j in range(max(int(samples), 1)):
            cases.append({'id': f'{scn["id"]}#{j}', 'scenario': scn['id'], 'name': scn.get('name', scn['id']),
                          'values': SC.sample_params(scn, seed * 1000003 + j), 'tested': {}})
    by = {c['id']: c for c in cases}
    done = [0]

    def evaluate(points: dict):
        """points: {case id: [grid indices]} → verdicts recorded in each case's `tested`."""
        for sid, st in studies.items():
            batch, keys = [], []
            for cid, idx in points.items():
                c = by[cid]
                if c['scenario'] != sid:
                    continue
                for i in idx:
                    if i in c['tested']:
                        continue
                    batch.append(dict(c['values'], **({LAT_KEY: x_of(i)} if i else {})))
                    keys.append((cid, i))
            if not batch:
                continue
            rows = st.evaluate(batch)
            for (cid, i), row in zip(keys, rows):
                by[cid]['tested'][i] = _brief(row, criterion, threshold)
                done[0] += 1
            if progress:
                progress(done[0])

    if exhaustive:
        evaluate({c['id']: list(range(n + 1)) for c in cases})
    else:
        evaluate({c['id']: [0, n] for c in cases})
        open_ = [c['id'] for c in cases if c['tested'][0]['passed'] is True and c['tested'][n]['passed'] is False]
        if open_ and n > 1:
            b = RB.Bisector(n, open_)               # index i ↔ grid point i + 1; the good ref is the grid's 0
            while True:
                i = b.next()
                if i is None:
                    break
                todo = [cid for cid in open_ if i in b.candidates(cid)]
                evaluate({cid: [i + 1] for cid in todo})
                b.record(i, {cid: (RB.SKIP if by[cid]['tested'][i + 1]['passed'] is None else
                                   RB.GOOD if by[cid]['tested'][i + 1]['passed'] else RB.BAD) for cid in todo})
    rows = []
    for c in cases:
        t = c['tested']
        base = t[0]['passed']
        fails = sorted(i for i, r in t.items() if r['passed'] is False)
        oks = sorted(i for i, r in t.items() if r['passed'] is True)
        row = {'case': c['id'], 'scenario': c['scenario'], 'name': c['name'], 'values': c['values'],
               'baseline_criticality': t[0]['criticality'], 'runs': len(t),
               'tested': [dict(r, x=x_of(i), index=i) for i, r in sorted(t.items())]}
        if base is None:
            row.update(status='not evaluated', budget=None, first_failure=None,
                       why=f'the run without latency did not finish ({t[0].get("error")})')
        elif base is False:
            row.update(status='fails without latency', budget=None, first_failure=0.0,
                       why='the scenario fails with no latency at all: its failure is not the computer being late')
        elif not fails:
            row.update(status='no failure up to the maximum', budget=x_of(n), first_failure=None,
                       why=f'passes at every tested latency up to {x_of(n):g} {unit}: the budget is at least that')
        else:
            first = fails[0]
            good = max((i for i in oks if i < first), default=0)
            gap = first - good > 1
            row.update(status='fails only when late' if not gap else 'fails only when late (interval open)',
                       budget=x_of(good), first_failure=x_of(first),
                       why=(f'passes up to {x_of(good):g} {unit}, fails from {x_of(first):g} {unit}'
                            + (' (untested or unfinished points between them)' if gap else '')))
            if any(i > first for i in oks):
                row['non_monotone'] = True
                row['why'] += '; NOT monotone: it passes again at a higher latency'
        if not constant and row.get('budget') is not None:
            row['budget_ms'] = {k: row['budget'] * profile['stats'][k] for k in ('median', 'p99', 'max')}
        rows.append(row)
    late = [r['case'] for r in rows if r['status'].startswith('fails only when late')]
    doc = {'schema': BUDGET_SCHEMA, 'name': name, 'created': utc_now(), 'unit': unit, 'grid_step': gstep,
           'grid_max': x_of(n), 'step_ms': dt_ms, 'mode': mode, 'seed': int(seed),
           'criterion': {'kind': criterion, 'threshold': threshold}, 'exhaustive': bool(exhaustive),
           'world': world, 'rig': rig, 'sweep': 'constant latency' if constant else 'the profile\'s scale',
           'profile': None if constant else {k: profile.get(k) for k in ('name', 'sha256', 'measured', 'source',
                                                                         'stats')},
           'cases': rows, 'fails_only_when_late': late, 'runs': sum(r['runs'] for r in rows),
           'assumes': 'common random numbers (one seed, one parameter set and one uniform sequence per case); '
                      + ('every grid point was run' if exhaustive else
                         'bisection: failures are assumed monotone in latency'),
           'quantisation': f'latencies are rounded up to whole steps of {dt_ms:g} ms'}
    return doc


def budget_dir(workspace: str, name: str) -> str:
    return os.path.join(workspace, 'latency', name)


def list_budgets(workspace: str) -> list:
    d = os.path.join(workspace, 'latency')
    out = []
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n, 'latency-budget.json')
            if os.path.isfile(p):
                out.append(p)
    return out


def list_profiles(workspace: str) -> list:
    d = os.path.join(workspace, 'latency', 'profiles')
    return [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.endswith('.json')] if os.path.isdir(d) else []


def with_profile(doc: dict, profile: dict | None) -> dict:
    """A constant-latency budget read against a profile: per case, the share of the profile's latencies that are
    over the case's budget (`over_budget_share`), so the cases a given computer would make fail can be read off."""
    if profile is None or doc.get('unit') != 'ms':
        return doc
    v = profile['samples_ms']
    for r in doc['cases']:
        if r.get('budget') is not None and r['status'].startswith('fails only when late'):
            r['over_budget_share'] = sum(1 for x in v if x > r['budget']) / len(v)
    doc['reference_profile'] = {k: profile.get(k) for k in ('name', 'sha256', 'measured', 'source', 'stats')}
    return doc


def budget_table(doc: dict) -> str:
    u = doc['unit']
    L = [f'latency budget {doc["name"]}: {doc["sweep"]} up to {doc["grid_max"]:g} {u} in steps of '
         f'{doc["grid_step"]:g} {u}; step {doc["step_ms"]:g} ms; criterion {doc["criterion"]["kind"]}'
         + (f' (criticality < {doc["criterion"]["threshold"]:g})' if doc['criterion']['kind'] != 'passed' else ''),
         f'{"case":28s} {"status":32s} {"budget":>10s} {"fails from":>11s} {"runs":>5s}']
    for r in doc['cases']:
        b = '—' if r['budget'] is None else (('≥ ' if r['first_failure'] is None else '') + f'{r["budget"]:g} {u}')
        f = '—' if r['first_failure'] is None else f'{r["first_failure"]:g} {u}'
        extra = ''
        if r.get('budget_ms'):
            extra = f'  (median {r["budget_ms"]["median"]:.1f} ms, p99 {r["budget_ms"]["p99"]:.1f} ms)'
        if r.get('over_budget_share') is not None:
            extra += f'  {100 * r["over_budget_share"]:.1f} % of the profile is over the budget'
        L.append(f'{r["case"]:28s} {r["status"]:32s} {b:>10s} {f:>11s} {r["runs"]:>5d}{extra}')
    late = doc['fails_only_when_late']
    L.append('fail only when the computer is late: ' + (', '.join(late) if late else 'none'))
    L.append(f'{doc["assumes"]}; {doc["quantisation"]}')
    return '\n'.join(L)


def profile_text(doc: dict) -> str:
    s, src = doc['stats'], doc['source']
    what = 'MEASURED' if doc.get('measured') else 'NOT measured'
    return (f'latency profile {doc["name"]} ({what}; source {src.get("kind")}: {src.get("device")}): '
            f'{s["n"]} values, median {s["median"]:.3f} ms, p95 {s["p95"]:.3f} ms, p99 {s["p99"]:.3f} ms, '
            f'max {s["max"]:.3f} ms')


# ================================================================================================ the command
def _scenario(proj, key: str) -> dict:
    from . import scenarios as SC
    scn = proj.item('scenarios', key) or (SC.builtin(key) if key in SC.LIBRARY else None)
    if scn is None and key.endswith('.json') and os.path.isfile(key):
        with open(key, encoding='utf-8') as fh:
            scn = json.load(fh)
    if scn is None:
        raise LatencyError(f'no scenario {key!r}')
    return scn


def main_latency(argv) -> int:
    """`falcon latency profile|run|budget|show …` (module doc): 0 ok, 2 failed, 6 refused."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon latency', description='latency in the loop (docs/latency-loop.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    pr = sub.add_parser('profile', help='make a latency profile from a measurement, a file or a synthetic '
                                        'distribution')
    pr.add_argument('--from-hil', default='', help='a HIL session\'s run folder (its latency.jsonl)')
    pr.add_argument('--measure', default='on_device', choices=HIL_MEASURES, help='which HIL measure (default '
                                                                                 'on_device)')
    pr.add_argument('--from-deploy', default='', help='a deployment folder profiled on the device')
    pr.add_argument('--engine', default='', help='the engine of the deployment (default: the first profiled)')
    pr.add_argument('--from-file', default='', help='latencies in ms: a JSON list or one number per line')
    pr.add_argument('--synthetic', default='', help='constant:MS or lognormal:median=MS,sigma=S[,spike_p=P,'
                                                    'spike_ms=MS][,n=N]')
    pr.add_argument('--seed', type=int, default=0)
    pr.add_argument('--name', default='')
    pr.add_argument('--out', default='', help='the profile file to write')
    pr.add_argument('--project', default='', help='or a project: <workspace>/latency/profiles/<name>.json')

    def common(p):
        p.add_argument('project')
        p.add_argument('--scenario', action='append', required=True, help='a built-in, a project scenario id or a '
                                                                          'scenario file (repeatable for budget)')
        p.add_argument('--profile', default='', help='a latency profile file')
        p.add_argument('--world', default='')
        p.add_argument('--rig', default='')
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--fps', type=float, default=0.0, help='the simulation step rate (default 20)')
        p.add_argument('--mode', default='pipeline', choices=MODES)
        p.add_argument('--criterion', default='auto', choices=('auto', 'passed', 'criticality'))
        p.add_argument('--threshold', type=float, default=0.5, help='criticality below this passes (default 0.5)')
        p.add_argument('--name', default='')
        p.add_argument('--fake-carla', default='')
        p.add_argument('--json', action='store_true')
    r = sub.add_parser('run', help='one scenario run with the latency in the loop, beside the same run without')
    common(r)
    r.add_argument('--constant', type=float, default=None, help='a constant latency in ms instead of a profile')
    r.add_argument('--scale', type=float, default=1.0, help='the profile\'s latencies multiplied by this')
    b = sub.add_parser('budget', help='the largest latency at which each scenario still passes')
    common(b)
    b.add_argument('--constant', action='store_true', help='sweep a constant latency in whole steps (with '
                                                           '--profile: read the budget against it)')
    b.add_argument('--max-ms', type=float, default=500.0, help='constant sweep: up to this latency (default 500)')
    b.add_argument('--max-scale', type=float, default=4.0, help='profile sweep: up to this scale (default 4)')
    b.add_argument('--step', type=float, default=0.25, help='profile sweep: the scale\'s grid step (default 0.25)')
    b.add_argument('--samples', type=int, default=1, help='parameter sets per scenario (seeded; default 1)')
    b.add_argument('--exhaustive', action='store_true', help='run every grid point instead of bisecting')
    s = sub.add_parser('show', help='print a profile or a latency-budget.json')
    s.add_argument('file')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    from .project import Project, ProjectError
    from .provenance import write_json
    try:
        if a.action == 'show':
            with open(a.file, encoding='utf-8') as fh:
                doc = json.load(fh)
            if doc.get('schema') == PROFILE_SCHEMA:
                print(profile_text(doc))
            elif doc.get('schema') == BUDGET_SCHEMA:
                print(budget_table(doc))
            else:
                raise LatencyError(f'{a.file} is neither a latency profile nor a latency budget')
            return P.OK
        if a.action == 'profile':
            given = [x for x in (a.from_hil, a.from_deploy, a.from_file, a.synthetic) if x]
            if len(given) != 1:
                raise LatencyError('give one of --from-hil, --from-deploy, --from-file, --synthetic')
            if a.from_hil:
                doc = profile_from_hil(a.from_hil, a.measure, a.name)
            elif a.from_deploy:
                doc = profile_from_deploy(a.from_deploy, a.engine, a.name)
            elif a.from_file:
                doc = profile_from_file(a.from_file, a.name)
            else:
                doc = profile_synthetic(a.synthetic, a.seed, a.name)
            out = a.out
            if not out and a.project:
                out = os.path.join(Project.load(a.project).workspace, 'latency', 'profiles', f'{doc["name"]}.json')
            if not out:
                raise LatencyError('give --out FILE (or --project P)')
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            write_json(out, doc)
            print(profile_text(doc))
            print(f'written to {out}')
            return P.OK
        proj = Project.load(a.project)
        scns = [_scenario(proj, k) for k in a.scenario]
        profile = load_profile(a.profile) if a.profile else None
        world = a.world or proj.doc['worlds'][0]['id']
        rig = a.rig or proj.doc['rigs'][0]['id']
        from ..sim.service import study_service
        if a.action == 'run':
            if (profile is None) == (a.constant is None):
                raise LatencyError('give --profile FILE or --constant MS, one of them')
            if len(scns) != 1:
                raise LatencyError('run takes one --scenario')
            svc = study_service(None, proj.workspace, fake_carla=a.fake_carla)
            try:
                from . import scenarios as SC
                scn = scns[0]
                st = make_study(svc, proj, scn, a.name or f'latency-run-{scn["id"]}', world, rig, a.seed,
                                a.fps or None, profile, a.mode)
                vals = SC.sample_params(scn, a.seed * 1000003)
                x = float(a.constant) if profile is None else float(a.scale)
                rows = st.evaluate([dict(vals), dict(vals, **({LAT_KEY: x} if x else {}))])
            finally:
                svc.shutdown()
            doc = {'schema': RUN_SCHEMA, 'scenario': scn['id'], 'values': vals, 'x': x,
                   'unit': 'ms' if profile is None else 'scale',
                   'without': _brief(rows[0], a.criterion, a.threshold),
                   'with': _brief(rows[1], a.criterion, a.threshold)}
            if a.json:
                print(json.dumps(doc, indent=1))
            else:
                for k in ('without', 'with'):
                    d = doc[k]
                    lat = d.get('latency') or {}
                    word = 'passes' if d['passed'] else 'FAILS' if d['passed'] is False else d['status']
                    print(f'{k:8s} latency: {word}, criticality {d["criticality"]}'
                          + (f', mean delay {lat.get("mean_delay_steps"):.2f} steps (max {lat.get("max_ms"):.1f} ms)'
                             if lat.get('mean_delay_steps') is not None else '') + f'  {d["run"]}')
            return P.OK if all(doc[k]['status'] == 'ok' for k in ('without', 'with')) else P.FAILED
        if a.action == 'budget':
            if not a.constant and profile is None:
                raise LatencyError('give --constant (a constant latency in whole steps) or --profile FILE (its scale)')
            name = a.name or ('latency-' + '-'.join(s['id'] for s in scns))[:60]
            em = P.Emitter()
            em.hello('latency')
            svc = study_service(None, proj.workspace, fake_carla=a.fake_carla)
            try:
                doc = budget(svc, proj, scns, name, world, rig, profile=None if a.constant else profile,
                             max_ms=a.max_ms, max_scale=a.max_scale, step=a.step, samples=a.samples, seed=a.seed,
                             fps=a.fps or None, mode=a.mode, criterion=a.criterion, threshold=a.threshold,
                             exhaustive=a.exhaustive,
                             progress=lambda n: em.progress(0.0, f'{n} runs', force=True))
            finally:
                svc.shutdown()
            if a.constant:
                doc = with_profile(doc, profile)
            out = budget_dir(proj.workspace, name)
            os.makedirs(out, exist_ok=True)
            write_json(os.path.join(out, 'latency-budget.json'), doc)
            print(json.dumps(doc, indent=1) if a.json else budget_table(doc))
            em.result('ok', summary={'cases': len(doc['cases']), 'runs': doc['runs'],
                                     'fails_only_when_late': len(doc['fails_only_when_late']),
                                     'out': os.path.join(out, 'latency-budget.json')})
            return P.OK
    except LatencyError as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return P.REFUSED if a.action in ('run', 'budget') else P.FAILED
    except (ProjectError, OSError, ValueError, RuntimeError) as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return P.FAILED
    return P.FAILED
