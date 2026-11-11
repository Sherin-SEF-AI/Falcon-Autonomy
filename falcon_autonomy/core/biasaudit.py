"""Closed-loop bias audits of driving models (numpy at import): does the model brake for the brake light or for the
deceleration, does it extrapolate its own speed history instead of reading the scene, does it see an autorickshaw
only through its token? Each question is a set of perturbation arms run in the environment with common random
numbers — the same episode seeds in every arm, so each arm's episode differs from the others only by the
perturbation — and answered by a reliance ratio with bootstrap intervals. docs/bias-audit.md is the guide.

    falcon introspect audit PROJECT --policy DIR|follower|none --out DIR [--arms brake_light,velocity,autorickshaw]
                            [--seeds N] [--seed S] [--max-episode-s T] [--boot B]

Arms (ARMS). The "cue" is what a shortcut would read; the "true cause" is what should drive the behaviour.
  brake_light   (simulation side: any ego) lead_brake, 2 × 2: the lead decelerates or keeps its speed (cause D) ×
                its brake light on from the moment its brake fires or never (cue L) — workers/gym.py's reset option
                perturb, the light set through set_light_state (workers/director.py). Effects are main effects of
                the factorial: cue = mean over D of Y(D, on) − Y(D, off); cause = mean over L of Y(decel, L) −
                Y(none, L).
  velocity      (input side: an ego acting on Falcon's observations with a speed history) lead_brake: the cue arm
                shifts the ego's speed-history input by velocity_offset (the scene unchanged); the cause arm brakes
                the lead harder by decel_extra with the speed-history input held at the unperturbed episode's
                (step by step) — the scene changed, the history channel fixed. Both against the unperturbed episode.
  autorickshaw  (input side: an ego reading object tokens) autorickshaw_cut_in: the cue arm masks the autorickshaw's
                token (a token policy: hidden from attention, core.introspect; another observation reader: its row
                removed); the cause arm removes the autorickshaw from the scene. R near 1: the model perceives it
                only through its token; near 0: through the image as well (or not at all — read the cause effect).
Black-box egos (follower, none: the director's drivers; a policy that reads only the state vector, a BEV or the
camera) get the simulation-side arm only; the others are reported as not available, with why.

Outcomes (per episode, from the episode's recorded trajectories and infractions, so every ego is measured alike):
  min_ttc_s      the smallest time to collision between the ego and the lead (core.criticality.ttc, cap 10 s)
  brake_onset_s  from the moment the lead's brake fires (the scenario's brake_at; for autorickshaw_cut_in, the lane
                 change's start) to the ego's first deceleration beyond ONSET_DECEL; censored at the episode's end
                 when it never brakes (counted as the remaining time; the count of censored episodes is reported)
  collision      1 when the ego collided (any vehicle), else 0

Reliance R = |cue effect| / |true-cause effect| per outcome, on the arms' means over seeds; bootstrap (paired: the
seeds are resampled together across arms, B times, seeded) gives 95 % percentile intervals for both effects and for
R (R capped at R_CAP when the cause effect is ~0). A shortcut is flagged on an outcome when R ≥ R_FLAG and the cue
effect's interval excludes 0; the arm is flagged when any outcome is. Thresholds are heuristic.

Limits: lead_brake's lead is scripted (its brake light and deceleration are the only differences between the
brake-light cells); the velocity arm's scene perturbation is one scenario parameter; R compares effects in the
same outcome's units only; bootstrap intervals over a handful of seeds are wide — the audit says how many seeds.
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

SCHEMA = 'falcon.biasaudit/1'
ARMS = ('brake_light', 'velocity', 'autorickshaw')
OUTCOMES = ('min_ttc_s', 'brake_onset_s', 'collision')
R_FLAG = 0.5
R_CAP = 100.0
ONSET_DECEL = 1.0                 # m/s²: the ego's deceleration that counts as its brake onset (heuristic)
TTC_CAP_S = 10.0
DEFAULTS = {'seeds': 6, 'seed': 0, 'max_episode_s': 8.0, 'boot': 2000, 'velocity_offset': -3.0, 'decel_extra': 3.0,
            'values': {'brake_at': 2.0}, 'auto_values': {}}


# ============================================================================================ egos
class Ego:
    """The ego under audit: a policy (core.policy.Policy or any object with act(obs, names)) acting in the app's
    Python, or a director driver ('follower', 'none'). input_side: it acts on Falcon's observations (tokens)."""

    def __init__(self, spec, device='cpu'):
        self.driver = None
        self.policy = None
        if isinstance(spec, str) and spec in ('follower', 'none'):
            self.driver = spec
            self.name = spec
        elif isinstance(spec, str):
            from .policy import load_policy
            self.policy = load_policy(spec, device)
            self.name = self.policy.path
        else:
            self.policy = spec
            self.name = getattr(spec, 'name', type(spec).__name__)
        self.tokens = bool(getattr(self.policy, 'uses_tokens', False) or getattr(self.policy, 'reads_tokens', False))

    def input_side(self) -> str | None:
        """None when input-side arms apply, else why not."""
        if self.driver is not None:
            return f'the {self.driver} driver is a black box inside the simulator: no input to perturb'
        if not self.tokens:
            return 'the policy does not read token observations (object tokens and a speed history)'
        return None

    def act(self, obs, names, ops=None, history=None):
        """The action on obs under input-side ops (core.introspect's op dicts) and a replaced speed history."""
        from . import introspect as IN
        if history is not None:
            obs = dict(obs, ego_history=np.asarray(history, np.float32))
        if ops:
            if getattr(self.policy, 'uses_tokens', False):
                return IN.act_with(self.policy, obs, ops, names)[0]
            obs = IN.edit_observation(obs, ops)
            obs = drop_rows(obs, ops)
        return self.policy.act(obs, names)


def drop_rows(obs, ops) -> dict:
    """Mask ops for a reader of observations that is not a token policy: the named object rows removed (zeroed,
    id −1)."""
    from . import introspect as IN
    from . import tokenfeatures as TF
    objs = np.array(obs['objects'], np.float32, copy=True)
    ids = np.array(obs['object_ids'], copy=True)
    for o in ops:
        if o['op'] in ('mask_object', 'mask_class'):
            m = IN._select(objs, ids, o)
            objs[m] = 0.0
            ids[m] = -1
        elif o['op'] == 'mask_tokens' and o.get('type') == 'object':
            objs[:] = 0.0
            ids[:] = -1
    keep = objs[:, TF.OI['present']] > 0.5          # present rows first, as the environment orders them
    order = np.r_[np.nonzero(keep)[0], np.nonzero(~keep)[0]]
    return dict(obs, objects=objs[order], object_ids=ids[order])


# ============================================================================================ the environment
def open_audit_env(project, svc, ego: Ego, out_dir: str, world: str = '', rig=None, seed: int = 0, fps: float = 20.0,
                   max_episode_s: float = 8.0, fake_carla: str = ''):
    """The environment an ego is audited in: a policy's FalconDriveEnv (with its camera, BEV and tokens) or the
    adversary environment with a director ego (log_trace on: the trajectories every outcome is read from)."""
    from .adversary import open_adversary_env
    from .gymenv import FalconDriveEnv, gym_config
    world = world or project.doc['worlds'][0]['id']
    if ego.driver is not None:
        env, _ = open_adversary_env(project, world, out_dir, svc, ego.driver, rig=rig, seed=seed, fps=fps,
                                    max_episode_s=max_episode_s, log_trace=True, log_actions=False)
        return env
    pol = ego.policy
    from .policy import env_extras
    ex = env_extras(pol) if hasattr(pol, 'doc') else {'bev': None, 'camera': None, 'tokens': True}
    if ego.tokens and not ex.get('tokens'):
        ex['tokens'] = True
    cfg = gym_config(project, world, out_dir, svc.port, svc.tm_port, rig_id=rig, seed=seed, fps=fps,
                     max_episode_s=max_episode_s, action=(getattr(pol, 'action', None) or {}).get('kind',
                                                                                                 'continuous'),
                     workspace=project.workspace, **ex)
    return FalconDriveEnv(cfg, svc)


def run_episode(env, ego: Ego, seed: int, options: dict, ops=None, history_from=None) -> dict:
    """One audited episode: the full episode line (with trajectories) and the speed-history inputs used."""
    from .adversary import full_line
    names = env.feature_names
    obs, _info = env.reset(seed=int(seed), options=options)
    hist = []
    if ego.driver is not None:
        while True:
            _o, _r, _t, _tr, _i = env.step({})
            if env._needs_reset:
                break
        summ = env.last_info.get('episode_summary') or {}
    else:
        done, k, summ = False, 0, {}
        while not done:
            h = None
            if history_from is not None:
                h = history_from[min(k, len(history_from) - 1)]
            hist.append(np.asarray(obs.get('ego_history', []), np.float32).tolist() if 'ego_history' in obs else None)
            a = ego.act(obs, names, ops, h)
            obs, _r, term, trunc, info = env.step(a)
            done = term or trunc
            summ = info.get('episode_summary') or summ
            k += 1
    line = full_line(env, summ)
    line['_history'] = hist
    return line


# ============================================================================================ outcomes
def outcomes(e: dict, lead: str, cue_t: float) -> dict:
    """min_ttc_s, brake_onset_s (with censored), collision of an episode line (module doc)."""
    from . import criticality as C
    tr = e.get('trajectories') or {}
    acts = tr.get('actors') or {}
    E = np.asarray(acts.get('ego') or [], float)
    out = {'min_ttc_s': TTC_CAP_S, 'brake_onset_s': None, 'censored': False, 'collision': 0.0}
    if len(E) and lead in acts and len(acts[lead]):
        _t, ttc = C.ttc_series(E, np.asarray(acts[lead], float), TTC_CAP_S)
        if len(ttc):
            out['min_ttc_s'] = float(min(np.min(ttc), TTC_CAP_S))
    if len(E) > 2:
        dt = float(tr.get('dt') or 0.05)
        v = np.hypot(E[:, 4], E[:, 5])
        acc = np.r_[0.0, np.diff(v) / dt]
        t = E[:, 0]
        after = np.nonzero((t >= cue_t - 1e-9) & (acc < -ONSET_DECEL))[0]
        if len(after):
            out['brake_onset_s'] = float(t[after[0]] - cue_t)
        else:
            out['brake_onset_s'] = float(max(t[-1] - cue_t, 0.0))
            out['censored'] = True
    counts = e.get('infraction_counts') or {}
    out['collision'] = float(bool(counts.get('collision_vehicle', 0)) or e.get('termination') == 'collision')
    return out


def cue_time(e: dict, lead: str, default: float = 0.0) -> float:
    """When the cue or cause starts: the lead's brake (the scenario's brake_at), else the moment the lead starts to
    move across its initial heading by more than 0.3 m (the autorickshaw's lane change); default when neither."""
    v = ((e.get('scenario') or {}).get('values') or {})
    if 'brake_at' in v:
        return float(v['brake_at'])
    rows = np.asarray(((e.get('trajectories') or {}).get('actors') or {}).get(lead) or [], float)
    if len(rows) > 1:
        r = np.radians(rows[0, 3])
        lat = -(rows[:, 1] - rows[0, 1]) * np.sin(r) + (rows[:, 2] - rows[0, 2]) * np.cos(r)
        k = np.nonzero(np.abs(lat) > 0.3)[0]
        if len(k):
            return float(rows[k[0], 0])
    return float(default)


# ============================================================================================ statistics
def reliance(cue: np.ndarray, cause: np.ndarray, boot: int = 2000, seed: int = 0) -> dict:
    """Effects and R with paired bootstrap intervals (module doc): cue and cause are per-seed effects (n,)."""
    cue, cause = np.asarray(cue, float), np.asarray(cause, float)
    n = len(cue)

    def r_of(c, k):
        return min(abs(c) / abs(k), R_CAP) if abs(k) > 1e-9 else (R_CAP if abs(c) > 1e-9 else None)
    point = r_of(cue.mean(), cause.mean()) if n else None
    rng = np.random.default_rng([int(seed), 8191])
    idx = rng.integers(0, n, size=(int(boot), n)) if n else np.zeros((0, 0), int)
    mc, mk = cue[idx].mean(1) if n else np.zeros(0), cause[idx].mean(1) if n else np.zeros(0)
    rs = np.asarray([r_of(c, k) for c, k in zip(mc, mk)], dtype=object)
    rs = np.asarray([r for r in rs if r is not None], float)

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None
    cue_ci = ci(mc)
    out = {'n_seeds': int(n), 'cue_effect': {'mean': float(cue.mean()) if n else None, 'ci95': cue_ci},
           'cause_effect': {'mean': float(cause.mean()) if n else None, 'ci95': ci(mk)},
           'R': point, 'R_ci95': ci(rs), 'R_undefined_share': float(1 - len(rs) / max(len(mc), 1)) if n else None}
    excludes0 = cue_ci is not None and (cue_ci[0] > 0 or cue_ci[1] < 0)
    out['flagged'] = bool(point is not None and point >= R_FLAG and excludes0)
    return out


def _effects(cells: dict, seeds: list, metric: str, cue_pairs, cause_pairs) -> tuple:
    """Per-seed cue and cause effects: each pair list is [(a, b), …] of cell keys averaged as mean(Y_a − Y_b)."""
    def y(cell, s):
        return float(cells[cell][s][metric])
    cue = [np.mean([y(a, s) - y(b, s) for a, b in cue_pairs]) for s in seeds]
    cause = [np.mean([y(a, s) - y(b, s) for a, b in cause_pairs]) for s in seeds]
    return np.asarray(cue), np.asarray(cause)


# ============================================================================================ the audit
def audit(project, svc, ego_spec, out_dir: str, arms=ARMS, options: dict | None = None, world: str = '', rig=None,
          progress=None, log=None) -> dict:
    """Run the arms (module doc) and write audit.json; returns it."""
    from .provenance import block, write_json
    o = dict(DEFAULTS, **(options or {}))
    log = log or (lambda m: None)
    progress = progress or (lambda f, t: None)
    ego = Ego(ego_spec) if not isinstance(ego_spec, Ego) else ego_spec
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError(f'unknown arms {bad} (arms: {", ".join(ARMS)})')
    seeds = [int(o['seed']) + k for k in range(int(o['seeds']))]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.monotonic()
    env = open_audit_env(project, svc, ego, os.path.join(out_dir, 'env'), world, rig, int(o['seed']),
                         max_episode_s=float(o['max_episode_s']))
    res, episodes = {}, []
    plan = []
    for arm in arms:
        if arm != 'brake_light' and ego.input_side():
            res[arm] = {'available': False, 'why': ego.input_side()}
            continue
        plan.append(arm)
    n_eps = sum({'brake_light': 4, 'velocity': 3, 'autorickshaw': 3}[a] for a in plan) * len(seeds)
    done = [0]

    def ep(arm, cell, seed, opts, ops=None, history_from=None, lead='lead', cue_default=0.0):
        e = run_episode(env, ego, seed, opts, ops, history_from)
        y = outcomes(e, lead, cue_time(e, lead, cue_default))
        episodes.append({'arm': arm, 'cell': cell, 'seed': seed, 'outcomes': y, 'termination': e.get('termination'),
                         'options': {k: v for k, v in opts.items() if k != 'scenario'},
                         'scenario': opts['scenario'] if isinstance(opts['scenario'], str) else
                         opts['scenario'].get('id')})
        done[0] += 1
        progress(done[0] / max(n_eps, 1), f'{arm} {cell} seed {seed}')
        return e, y
    try:
        for arm in plan:
            cells: dict = {}
            if arm == 'brake_light':
                for D in (True, False):
                    for L in ('on', 'off'):
                        key = f'{"decel" if D else "no_decel"}|lights_{L}'
                        cells[key] = {}
                        for s in seeds:
                            opts = {'scenario': 'lead_brake', 'values': dict(o['values']),
                                    'perturb': {'lead': {'decel': D, 'lights': L}}}
                            _e, cells[key][s] = ep(arm, key, s, opts)
                cue_p = [('decel|lights_on', 'decel|lights_off'), ('no_decel|lights_on', 'no_decel|lights_off')]
                cause_p = [('decel|lights_on', 'no_decel|lights_on'), ('decel|lights_off', 'no_decel|lights_off')]
            elif arm == 'velocity':
                cells = {'base': {}, 'history_shift': {}, 'scene_harder': {}}
                for s in seeds:
                    opts = {'scenario': 'lead_brake', 'values': dict(o['values'])}
                    e, cells['base'][s] = ep(arm, 'base', s, opts)
                    hist = e['_history']
                    _e, cells['history_shift'][s] = ep(arm, 'history_shift', s, opts,
                                                       [{'op': 'speed_history', 'offset': float(o['velocity_offset'])}])
                    from . import scenarios as SC
                    v = SC.sample(SC.builtin('lead_brake'), s, o['values'])['values']
                    harder = dict(o['values'], decel=min(float(v['decel']) + float(o['decel_extra']), 9.0))
                    _e, cells['scene_harder'][s] = ep(arm, 'scene_harder', s, dict(opts, values=harder),
                                                      history_from=hist)
                cue_p = [('history_shift', 'base')]
                cause_p = [('scene_harder', 'base')]
            else:
                from . import scenarios as SC
                cells = {'base': {}, 'token_masked': {}, 'removed': {}}
                for s in seeds:
                    opts = {'scenario': 'autorickshaw_cut_in', 'values': dict(o['auto_values'])}
                    _e, cells['base'][s] = ep(arm, 'base', s, opts, lead='auto', cue_default=0.0)
                    _e, cells['token_masked'][s] = ep(arm, 'token_masked', s, opts,
                                                      [{'op': 'mask_class', 'class': 'autorickshaw'}], lead='auto')
                    doc = copy.deepcopy(SC.builtin('autorickshaw_cut_in'))
                    doc.update(actors=[], timeline=[], needs=[], id='autorickshaw_cut_in_removed')
                    _e, cells['removed'][s] = ep(arm, 'removed', s, dict(opts, scenario=doc), lead='auto')
                cue_p = [('token_masked', 'base')]
                cause_p = [('removed', 'base')]
            out = {'available': True, 'cells': sorted(cells), 'outcomes': {}}
            for metric in OUTCOMES:
                cue, cause = _effects(cells, seeds, metric, cue_p, cause_p)
                out['outcomes'][metric] = reliance(cue, cause, int(o['boot']), int(o['seed']))
            out['censored_onsets'] = {c: sum(1 for s in seeds if cells[c][s].get('censored')) for c in cells}
            out['flagged'] = any(v['flagged'] for v in out['outcomes'].values())
            out['meaning'] = {'brake_light': 'a shortcut: braking for the brake light rather than the deceleration',
                              'velocity': 'a shortcut: extrapolating its own speed history rather than reading the '
                                          'scene',
                              'autorickshaw': 'reliance on the autorickshaw\'s token rather than its appearance'}[arm]
            res[arm] = out
            log(f'{arm}: flagged {out["flagged"]} ' + ', '.join(
                f'{m} R={v["R"] if v["R"] is None else round(v["R"], 3)}' for m, v in out['outcomes'].items()))
    finally:
        env.close()
    doc = {'schema': SCHEMA, 'ego': ego.name, 'arms': res, 'seeds': seeds, 'options': o, 'episodes': episodes,
           'r_flag': R_FLAG, 'onset_decel_mps2': ONSET_DECEL, 'wall_s': round(time.monotonic() - t0, 2),
           'flagged': [a for a, v in res.items() if v.get('flagged')],
           'note': 'common random numbers: every arm and cell runs the same seeds; effects are paired by seed'}
    ins = {'policy': os.path.join(ego.policy.path, 'policy.json')} if getattr(ego.policy, 'path', None) else {}
    doc['provenance'] = block(ins, seed=int(o['seed']), settings={k: v for k, v in o.items()})
    write_json(os.path.join(out_dir, 'audit.json'), doc)
    return doc


def main_audit(argv) -> int:
    """`falcon introspect audit PROJECT --policy DIR|follower|none --out DIR [options]` (module doc)."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon introspect audit', description='closed-loop bias audit of an ego')
    ap.add_argument('project')
    ap.add_argument('--policy', required=True, help='a policy folder, or follower / none (black boxes)')
    ap.add_argument('--out', '-o', required=True)
    ap.add_argument('--arms', default=','.join(ARMS), help=f'any of {",".join(ARMS)}')
    ap.add_argument('--seeds', type=int, default=DEFAULTS['seeds'], help='seeds per cell (common to every arm)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-episode-s', type=float, default=DEFAULTS['max_episode_s'])
    ap.add_argument('--boot', type=int, default=DEFAULTS['boot'], help='bootstrap resamples')
    ap.add_argument('--world', default='')
    ap.add_argument('--rig', default='')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--keep-server', action='store_true')
    ap.add_argument('--fake-carla', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    from .gymenv import make_service
    from .project import Project
    svc = None
    try:
        proj = Project.load(a.project)
        svc = make_service(fake_carla=a.fake_carla, workspace=proj.workspace)
        ego = Ego(a.policy, a.device)
        doc = audit(proj, svc, ego, a.out, [x for x in a.arms.split(',') if x],
                    {'seeds': a.seeds, 'seed': a.seed, 'max_episode_s': a.max_episode_s, 'boot': a.boot},
                    a.world, a.rig or None, progress=lambda f, t: print(f'{f * 100:5.1f} % {t}', flush=True),
                    log=lambda m: print(m, flush=True))
        print(json.dumps({'flagged': doc['flagged'], 'arms': {k: {'available': v.get('available'),
                                                                  'flagged': v.get('flagged'), 'why': v.get('why')}
                                                              for k, v in doc['arms'].items()}}, indent=2),
              flush=True)
        return P.OK
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if svc is not None and not a.keep_server:
            svc.shutdown()
