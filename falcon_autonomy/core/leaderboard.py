"""Driving evaluation in the style of the CARLA Leaderboard, and the replay of any infraction (numpy at import; the
environment and torch inside the functions).

A policy (core.policy: behaviour cloning or PPO, or any callable observation → action) drives a set of routes through
the environment (core.gymenv); each episode's summary (the gym worker's episodes.jsonl) is scored as the Leaderboard
scores a route:

    driving score DS = route completion RC × infraction penalty IP          (per route; the set's score is the mean)
    IP = Π_j p_j^(n_j)                                                       (n_j infractions of kind j)

Penalty coefficients p_j, as the CARLA Leaderboard 1.0 and 2.0 documents state them: collision with a pedestrian
0.50, with a vehicle 0.60, with a static object 0.65; running a red light 0.70; running a stop sign 0.80.

Route completion. The share of the route's length driven (the environment's progress along the route), less the
distance driven off the road, the wrong way or outside the route's lanes (the Leaderboard does not count route driven
off-road or outside the route's lanes towards completion). Route deviation and an agent blocked end the route (the
Leaderboard's shutdown events): they add no penalty, and their completion is what was driven until then; so does the
time limit.

Not scored (stated, not hidden): Leaderboard 2.0's failure to keep a minimum speed, scenario time-outs and failure
to yield to an emergency vehicle — Falcon's scenarios and environment have no such events. A Falcon route is not a
Leaderboard route (no official route files or towns' route sets): these numbers follow the Leaderboard's method and
are not comparable with its public results.

Intervals. Routes are independent episodes; each of DS, RC and IP gets a 95 % percentile bootstrap interval over
routes (core.metrics.bootstrap), and per kind the infractions per km driven.

Replay. Under kinematic motion with no background traffic an episode is a function of its seed, its options and its
actions (measured: identical repeats on a real server, core.gymenv). replay() resets the environment with the
episode's seed and options, adds a chase camera (a camera changes nothing in the motion), applies the logged actions
and keeps the camera's frames from `before_s` before the infraction to `after_s` after it — the one-click replay. An
episode with physics motion or traffic cannot be replayed exactly and is refused with the reason.
"""
from __future__ import annotations

import json
import os

import numpy as np

SCHEMA = 'falcon.leaderboard/1'
PENALTY = {'collision_pedestrian': 0.50, 'collision_vehicle': 0.60, 'collision_static': 0.65, 'red_light': 0.70,
           'stop_sign': 0.80}
NOT_COUNTED = ('off_road', 'wrong_way', 'outside_route_lanes')     # their distance is taken off route completion
SHUTDOWN = ('route_deviation', 'agent_blocked')
ENV_KEYS = ('values', 'route_m', 'max_episode_s', 'target_speed_kmh', 'start_speed_kmh', 'reward', 'thresholds',
            'terminate_on', 'fps')        # an episode's environment settings beyond its options: kept for replay
NOT_SCORED = ('Leaderboard 2.0: minimum-speed failures, scenario time-outs, failure to yield to an emergency '
              'vehicle — no such events in Falcon\'s scenarios')
CHASE_CAMERA = {'width': 640, 'height': 360, 'fov': 90.0, 'pose': {'x': -6.0, 'y': 0.0, 'z': 3.0, 'roll': 0.0,
                                                                   'pitch': -12.0, 'yaw': 0.0}}


class LeaderboardError(Exception):
    pass


def score_episode(e: dict) -> dict:
    """The Leaderboard numbers of one episode summary (module doc)."""
    length = max(float(e.get('route_length_m') or 0.0), 1e-9)
    counts = e.get('infraction_counts') or {}
    not_counted = sum(float(i.get('distance_m') or 0.0) for i in e.get('infractions') or ()
                      if i.get('kind') in NOT_COUNTED)
    rc_raw = float(e.get('route_completion') or 0.0)
    rc = max(0.0, min(1.0, rc_raw - not_counted / length))
    ip = 1.0
    for k, p in PENALTY.items():
        ip *= p ** int(counts.get(k, 0))
    shutdown = next((k for k in SHUTDOWN if counts.get(k)), None)
    return {'route_completion': rc, 'route_completion_raw': rc_raw, 'not_counted_m': round(not_counted, 3),
            'infraction_penalty': ip, 'driving_score': rc * ip, 'shutdown': shutdown,
            'termination': e.get('termination'), 'distance_m': float(e.get('distance_m') or 0.0),
            'infraction_counts': {k: int(v) for k, v in counts.items() if v}}


def score(episodes: list, n_boot: int = 1000, seed: int = 0, ruleset=None) -> dict:
    """The set's scores: per route, the means with bootstrap intervals over routes, infractions per km by kind; and,
    beside them, the traffic-rule violations (core.rules) of the routes whose episodes carry a trace — not part of the
    driving score (the Leaderboard has no such term)."""
    from .metrics import bootstrap
    if not episodes:
        raise LeaderboardError('no episodes to score')
    rows = [dict(score_episode(e), episode=e.get('episode'), seed=e.get('seed'),
                 scenario=(e.get('scenario') or {}).get('id') or 'free') for e in episodes]
    rules = rule_checks(episodes, ruleset, n_boot, seed)
    for r, rc in zip(rows, rules['per_route'] if rules else ()):
        r['rule_violations'] = rc['totals']['violations'] if rc else None
        r['rule_violations_per_km'] = rc['totals']['per_km'] if rc else None

    def means(rs):
        return {k: float(np.mean([r[k] for r in rs])) for k in ('driving_score', 'route_completion',
                                                                 'infraction_penalty')}
    boot = bootstrap(means, rows, n=n_boot, seed=seed, unit='image')
    km = sum(r['distance_m'] for r in rows) / 1000.0
    kinds = sorted({k for r in rows for k in r['infraction_counts']})
    per_km = {k: (sum(r['infraction_counts'].get(k, 0) for r in rows) / km if km > 0 else None) for k in kinds}
    by_scn = {}
    for r in rows:
        by_scn.setdefault(r['scenario'], []).append(r['driving_score'])
    return {'schema': SCHEMA, 'routes': len(rows), 'distance_km': km,
            'metrics': {k: {'value': v['value'], 'ci_low': v['ci_low'], 'ci_high': v['ci_high']}
                        for k, v in boot['metrics'].items()},
            'infractions_per_km': per_km, 'by_scenario': {k: float(np.mean(v)) for k, v in sorted(by_scn.items())},
            'per_route': rows, 'penalties': PENALTY, 'not_scored': NOT_SCORED,
            'rules': rules['summary'] if rules else None,
            'method': 'CARLA Leaderboard scoring (module doc); intervals: percentile bootstrap over routes',
            'n_boot': n_boot, 'seed': seed}


def rule_checks(episodes: list, ruleset=None, n_boot: int = 1000, seed: int = 0) -> dict | None:
    """core.rules on every episode with a trace: {'per_route': [result or None], 'summary': aggregate + the rule
    set}; None when no episode carries a trace (episodes recorded before Falcon 1.1)."""
    from . import rules as RU
    if not any(e.get('trace') for e in episodes):
        return None
    per = []
    for e in episodes:
        try:
            per.append(RU.check_episode(e, ruleset))
        except RU.RulesError:
            per.append(None)
    summ = RU.aggregate([r for r in per if r], n_boot, seed)
    summ['routes_checked'] = sum(1 for r in per if r)
    summ['note'] = ('beside the driving score, not in it; per km over the routes with a trace; the rule set\'s '
                    'heuristic thresholds and unrecorded facts are in docs/traffic-rules.md')
    return {'per_route': per, 'summary': summ}


def route_set(scenarios: list | None = None, seeds_per: int = 5, free: int = 5, seed: int = 0) -> list:
    """A route set: each scenario (built-in keys or project ids; default every built-in that runs in the
    environment's town) with seeds_per seeds, and `free` free-driving routes. Routes are (scenario or None, seed)."""
    from . import scenarios as SC
    keys = list(scenarios) if scenarios is not None else list(SC.LIBRARY)
    out = [{'scenario': k, 'seed': seed + 1000 * i + j} for i, k in enumerate(keys) for j in range(seeds_per)]
    out += [{'scenario': None, 'seed': seed + 100000 + j} for j in range(free)]
    return out


def evaluate(policy, env_factory, routes: list, out_dir: str, n_boot: int = 1000, progress=None) -> dict:
    """Drive every route (env_factory(scenario) -> env; one env per scenario, reused across its seeds) and score.
    Writes leaderboard.json and episodes.jsonl (the episodes with their actions, for replay) in out_dir."""
    from .provenance import block, write_json
    os.makedirs(out_dir, exist_ok=True)
    act = policy.act if hasattr(policy, 'act') else policy
    episodes, failed = [], []
    groups = {}
    for r in routes:
        groups.setdefault(r['scenario'], []).append(r)
    done = 0
    with open(os.path.join(out_dir, 'episodes.jsonl'), 'w', encoding='utf-8') as fh:
        for scn, rs in groups.items():
            env = None
            try:
                env = env_factory(scn)
                names = env.feature_names
                for r in rs:
                    obs, info = env.reset(seed=int(r['seed']))
                    summ = None
                    term = trunc = False
                    while not (term or trunc):
                        a = act(obs, names) if hasattr(policy, 'act') else act(obs)
                        obs, _r, term, trunc, info = env.step(a)
                        summ = info.get('episode_summary')
                    full = _with_actions(env, summ)
                    full['env_dir'] = env.run_dir
                    cfg = getattr(env, 'cfg', None) or {}
                    full['traffic_vehicles'] = int((cfg.get('traffic') or {}).get('vehicles', 0))
                    full['traffic_walkers'] = int((cfg.get('traffic') or {}).get('walkers', 0))
                    full['env_config'] = {k: cfg.get(k) for k in ENV_KEYS}
                    episodes.append(full)
                    fh.write(json.dumps(full, separators=(',', ':')) + '\n')
                    done += 1
                    if progress:
                        progress(done / len(routes), f'{scn or "free"} seed {r["seed"]}: DS '
                                                     f'{score_episode(summ)["driving_score"]:.2f}')
            except Exception as e:           # a scenario the town cannot host, a lost server: recorded, not hidden
                failed.append({'scenario': scn, 'error': f'{type(e).__name__}: {e}'})
            finally:
                if env is not None:
                    env.close()
    doc = score(episodes, n_boot)
    doc['failed'] = failed
    doc['policy'] = getattr(policy, 'path', None)
    ins = {'policy': os.path.join(policy.path, 'policy.json')} if getattr(policy, 'path', None) else {}
    doc['provenance'] = block(ins, settings={'routes': routes, 'n_boot': n_boot})
    write_json(os.path.join(out_dir, 'leaderboard.json'), doc)
    return doc


def _with_actions(env, summ: dict) -> dict:
    """The episode's full line from the worker's episodes.jsonl (with its actions), matched by episode number."""
    p = os.path.join(env.run_dir, 'episodes.jsonl')
    try:
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                row = json.loads(line)
                if row.get('episode') == summ.get('episode') and row.get('seed') == summ.get('seed'):
                    return row
    except (OSError, ValueError):
        pass
    return dict(summ)


def replayable(e: dict) -> str | None:
    """None when the episode can be replayed exactly, else why not."""
    if e.get('motion', 'kinematic') != 'kinematic':
        return 'physics motion is not repeatable in CARLA 0.9.16 (measured); only kinematic episodes replay exactly'
    if e.get('traffic_vehicles') or e.get('traffic_walkers'):
        return 'background traffic makes an episode non-repeatable'
    if 'actions' not in e:
        return 'the episode was logged without its actions (log_actions off)'
    return None


def replay(e: dict, env, infraction: int | None = None, before_s: float = 3.0, after_s: float = 1.0,
           out_dir: str = '') -> dict:
    """Re-run an episode with the environment (one with the chase camera) and keep its frames around an infraction
    (or the whole episode): {'frames': [paths], 'steps', 'match': whether the replay met the same infraction}."""
    import cv2
    why = replayable(e)
    if why:
        raise LeaderboardError(why)
    infs = e.get('infractions') or []
    if infraction is not None and not 0 <= infraction < len(infs):
        raise LeaderboardError(f'the episode has {len(infs)} infractions')
    have = getattr(env, 'cfg', None) or {}
    for k in ('values', 'route_m', 'max_episode_s', 'target_speed_kmh', 'start_speed_kmh'):
        want = (e.get('env_config') or {}).get(k)       # the replay's environment must be the episode's
        if 'env_config' in e and (want or None) != (have.get(k) or None):
            raise LeaderboardError(f'the environment\'s {k} ({have.get(k)!r}) is not the episode\'s ({want!r})')
    fps = float(getattr(env, 'ready', {}).get('fps') or 20.0)
    target = infs[infraction] if infraction is not None else None
    lo = max(0, int(target['step'] - before_s * fps)) if target else 0
    hi = int(target['step'] + after_s * fps) if target else 10 ** 9
    os.makedirs(out_dir, exist_ok=True)
    obs, info = env.reset(seed=int(e['seed']), options=e.get('options') or None)
    frames, met = [], False
    for k, a in enumerate(e['actions']):
        obs, _r, term, trunc, info = env.step(np.asarray(a, np.float32) if isinstance(a, list) else a)
        step = k + 1
        if target and not met:
            met = any(i.get('kind') == target['kind'] and i.get('step') == target['step']
                      for i in info.get('infractions_new') or info.get('infractions') or ())
        if lo <= step <= hi and isinstance(obs, dict) and 'camera' in obs:
            p = os.path.join(out_dir, f'{step:06d}.jpg')
            cv2.imwrite(p, cv2.cvtColor(np.asarray(obs['camera']), cv2.COLOR_RGB2BGR))
            frames.append(p)
        if term or trunc or step > hi:
            break
    summ = info.get('episode_summary')
    if target and not met and summ:
        met = any(i.get('kind') == target['kind'] and i.get('step') == target['step']
                  for i in summ.get('infractions') or ())
    doc = {'episode': e.get('episode'), 'seed': e.get('seed'), 'infraction': target, 'steps': [lo, min(hi, step)],
           'frames': frames, 'match': met if target else None, 'fps': fps}
    with open(os.path.join(out_dir, 'replay.json'), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    return doc


# ================================================================================================ the command
def main_leaderboard(argv) -> int:
    """`falcon policy leaderboard PROJECT --policy DIR --out DIR [--scenario K]... [--seeds-per N] [--free N]` and
    `falcon policy replay EPISODES.jsonl --episode N [--infraction K] --out DIR PROJECT`: 0 ok, 2 failed."""
    import argparse
    from . import protocol as P
    from .gymenv import EnvError
    from .policy import env_args, load_policy
    replay_mode = argv[:1] == ['replay']
    ap = argparse.ArgumentParser(prog=f'falcon policy {"replay" if replay_mode else "leaderboard"}')
    env_args(ap)
    if replay_mode:
        argv = argv[1:]
        ap.add_argument('--episodes', required=True, help='an evaluation\'s episodes.jsonl')
        ap.add_argument('--episode', type=int, required=True, help='the line number (0-based)')
        ap.add_argument('--infraction', type=int, default=None)
        ap.add_argument('--before', type=float, default=3.0)
        ap.add_argument('--after', type=float, default=1.0)
    else:
        argv = argv[1:] if argv[:1] == ['leaderboard'] else argv
        ap.add_argument('--policy', required=True)
        ap.add_argument('--scenario', action='append', default=None)
        ap.add_argument('--seeds-per', type=int, default=5)
        ap.add_argument('--free', type=int, default=5)
        ap.add_argument('--boot', type=int, default=1000)
        ap.add_argument('--device', default='cpu')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('leaderboard')
    from .gymenv import FalconDriveEnv, gym_config, make_service
    from .project import Project
    svc = None
    try:
        proj = Project.load(a.project)
        svc = make_service(fake_carla=a.fake_carla, workspace=proj.workspace)
        world = a.world or proj.doc['worlds'][0]['id']

        def make_env(scn, out, action='continuous', bev=None, camera=None, motion=None, env_config=None):
            kw = {k: v for k, v in (env_config or {}).items() if v is not None}
            kw.setdefault('fps', a.fps)
            cfg = gym_config(proj, world, out, svc.port, svc.tm_port, rig_id=a.rig or None, seed=a.seed,
                             scenario=scn, motion=motion or a.motion, action=action, bev=bev, camera=camera,
                             traffic_vehicles=a.traffic, log_actions=True, workspace=proj.workspace, **kw)
            return FalconDriveEnv(cfg, svc)
        if replay_mode:
            with open(a.episodes, encoding='utf-8') as fh:
                rows = [json.loads(x) for x in fh if x.strip()]
            if not 0 <= a.episode < len(rows):
                raise LeaderboardError(f'{a.episodes} has {len(rows)} episodes')
            e = rows[a.episode]
            why = replayable(e)
            if why:
                raise LeaderboardError(why)
            kind = 'discrete' if e['actions'] and not isinstance(e['actions'][0], list) else 'continuous'
            env = make_env((e.get('scenario') or {}).get('id'), os.path.join(a.out, 'env'), kind,
                           camera=CHASE_CAMERA, motion=e.get('motion', 'kinematic'), env_config=e.get('env_config'))
            try:
                doc = replay(e, env, a.infraction, a.before, a.after, a.out)
            finally:
                env.close()
            em.result('ok', summary={'frames': len(doc['frames']), 'match': doc['match']})
        else:
            pol = load_policy(a.policy, a.device)
            routes = route_set(a.scenario, a.seeds_per, a.free, a.seed)
            bev = pol.doc.get('bev') if pol.uses_bev else (True if a.bev else None)
            doc = evaluate(pol, lambda scn: make_env(scn, os.path.join(a.out, 'env', scn or 'free'),
                                                     pol.action['kind'], bev),
                           routes, a.out, a.boot, lambda f, t: em.progress(f, t))
            m = doc['metrics']
            em.result('ok', summary={'routes': doc['routes'], 'failed': len(doc['failed']),
                                     **{k: round(v['value'], 4) for k, v in m.items()}})
        return P.OK
    except (LeaderboardError, EnvError, OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if svc is not None and not a.keep_server:
            svc.shutdown()
