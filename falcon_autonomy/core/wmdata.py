"""World-model training data: sequences of (BEV, state vector, previous action, reward, terminal) built from recorded
drives (falcon-raw) and from episodes collected in the learning environment (numpy only; the environment and torch
inside the functions that need them). core.wmtrain trains the learned world model on them.

    drive_sequence(run_dir, options) -> sequence          one recorded drive, from what it recorded
    collect(env, behaviour, episodes, out_dir, ...) -> doc  environment episodes (falcon worldmodel collect)
    load_sources(paths, options) -> [sequence]             drives and experience folders, in order

A sequence (dict), one row per step t, the DreamerV3 convention (Hafner et al. 2023, "Mastering Diverse Domains
through World Models", arXiv:2301.04104): the observation at t, the action that LED to it, the reward received on
arriving at it, and whether it is terminal:

    bev       (T, H, W) uint8   the BEV raster's bit planes packed (core.drivefeatures.bev_bits: core.worldmodel's
                                own raster, the function `falcon derive worldmodel` and the environment use), at the
                                environment's BEV spec unless options bev_spec says otherwise
    state     (T, F) float32    core.drivefeatures' observation vector (all of FEATURE_NAMES, raw units)
    action    (T, 2) float32    [steer, accel] applied from t − 1 to t (at t = 0: the recorded control or zeros;
                                a model resets its state there and ignores it)
    reward    (T,) float32      the reward of that transition (0 at t = 0), and terms (T, len(REWARD_TERMS))
    terminal  (T,) bool         t is a terminal state of the task (the environment's `terminated`)
    name, kind ('drive' | 'env'), source (the path), report (what was derived and how)

From a recorded drive (drive_sequence). The drive is replayed through core.drivefeatures.Featuriser exactly as
behaviour cloning does (core.imitation.drive_samples: the ego from frames.jsonl and its actors.jsonl row, the route
from lanes.json matched to the lanes the drive went through, traffic lights unknown, the target speed the scenario's
ego speed, else options target_speed_kmh, else — heuristic — the drive's median moving speed). Then:
* the action: the control recorded at step t (frames.jsonl `control`, the control in effect during the tick that
  produced step t) as [steer, throttle − brake] (drives since 0.6). A drive that recorded no control gets DERIVED
  actions, labelled so (report action_source 'derived'): the inverse of the environment's kinematic bicycle model
  (workers/director.py kin_step: full throttle 3 m/s², full brake 8 m/s², steer × 70° wheel angle, wheelbase 1.2 × the
  ego's box half length): accel = Δv/dt ÷ 3 when ≥ 0 else ÷ 8, steer = atan(yaw rate · wheelbase / v) ÷ 70° (0 below
  DERIVE_MIN_SPEED). Exact for a kinematic drive (measured against recorded control: docs/world-model-training.md);
  for a physics drive (CARLA's vehicle dynamics, the autopilot's drives) it is an approximation.
* the reward: core.drivefeatures.reward_terms with the environment's default weights, from the replayed facts: the
  route progress Δs (clipped to ±5 m as the environment does), speed against the target, lateral offset, heading
  error, jerk, steer rate (from the actions), collisions (box overlaps with actors.jsonl's vehicles and walkers, as
  the environment's box test: core.criticality), off road (the ego's centre outside lanes.json's drivable raster —
  the environment asks CARLA's map; this is the map's driving lanes as recorded, a stated difference), route
  deviation and agent blocked (core.drivefeatures.InfractionMonitor). NOT computable from a recording and so always
  0: wrong_way and outside_lanes (they need CARLA's lane ids), red_light and stop_sign (light states are not
  recorded). The report lists them under reward_terms_unavailable.
* terminal: always False. A recorded drive is cut by time, not by a terminal state, and a collision does not end it;
  the continue head therefore learns episode ends only from environment episodes.
* gaps: a drive whose steps are not consecutive (a dropped step) is split there into several sequences.

From the environment (collect). A behaviour drives the environment (core.gymenv, with the BEV on) for N episodes
with seeds seed, seed + 1, …: the observation's state and BEV at each step, info['action_applied'], the
environment's own reward and terms, `terminated` as terminal (a time limit is not terminal). Behaviours: 'scripted' —
a proportional lane-keeping and speed controller on the state vector (heuristic gains, below) — or a policy folder
(core.policy), each with Ornstein–Uhlenbeck exploration noise (θ, σ in the options; Uhlenbeck & Ornstein 1930, as
DDPG uses it, Lillicrap et al. 2016) added to its action, or 'random' (the noise alone around [0, 0.3]). The
experience folder: experience.json (manifest, the behaviour, provenance over the environment's gym.json),
episodes/<n>.npz (one sequence each) and the environment's run folder (env/: episodes.jsonl, run.json).

Limits: a recorded drive's reward is the environment's reward less the map terms above, so it is not the same
function as an episode's; derived actions of a physics drive are approximate; recorded drives carry the recording
driver's behaviour only (the lane follower or the autopilot), so a model trained on them alone has never seen
off-policy actions (environment episodes with exploration noise are what covers them).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import drivefeatures as DF

SEQ_KEYS = ('bev', 'state', 'action', 'reward', 'terms', 'terminal')
EXPERIENCE_SCHEMA = 'falcon.wmexperience/1'
EXPERIENCE = 'experience.json'
KIN_ACCEL, KIN_BRAKE, KIN_STEER_DEG = 3.0, 8.0, 70.0      # workers/director.py's kinematic bicycle model
DERIVE_MIN_SPEED = 0.5                                       # m/s: below it a derived steer is not stated (0)
UNAVAILABLE_TERMS = ('wrong_way', 'outside_lanes', 'red_light', 'stop_sign')
SCRIPTED = {'k_lat': 0.15, 'k_head': 0.8, 'k_speed': 0.25, 'safe_ttc_s': 3.0, 'min_gap_m': 6.0}   # heuristic gains
OU = {'theta': 0.15, 'sigma': 0.2}
DEFAULTS = {'bev_spec': None, 'target_speed_kmh': None}


def _jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


# ============================================================================================ recorded drives
def derive_action(v0: float, v1: float, yaw0: float, yaw1: float, dt: float, wheelbase: float) -> list:
    """[steer, accel] that the environment's kinematic bicycle model would need to go from (v0, yaw0) to (v1, yaw1)
    in dt (module doc): speeds in m/s, yaws in degrees."""
    a = (float(v1) - float(v0)) / dt
    accel = a / KIN_ACCEL if a >= 0.0 else a / KIN_BRAKE
    steer = 0.0
    if v1 >= DERIVE_MIN_SPEED:
        w = math.radians(float(DF.wrap_deg(float(yaw1) - float(yaw0)))) / dt
        steer = math.atan(w * wheelbase / float(v1)) / math.radians(KIN_STEER_DEG)
    return [float(np.clip(steer, -1.0, 1.0)), float(np.clip(accel, -1.0, 1.0))]


def _target_speed(run_dir: str, speeds, options: dict) -> tuple:
    cfg = {}
    for name in ('drive.json', 'live.json'):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as fh:
                cfg = json.load(fh)
            break
    scn = cfg.get('scenario') or {}
    if scn.get('ego', {}).get('speed_kmh') is not None:
        return float(scn['ego']['speed_kmh']) / 3.6, 'the scenario\'s ego speed'
    if options.get('target_speed_kmh'):
        return float(options['target_speed_kmh']) / 3.6, 'the option target_speed_kmh'
    moving = np.asarray(speeds, float)
    moving = moving[moving > 0.5]
    return (float(np.median(moving)) if len(moving) else 0.0), 'heuristic: the drive\'s median moving speed'


def _contacts(pose: dict, length: float, width: float, rows, ego_id) -> list:
    from . import criticality as C
    A = C.box_corners(pose['x'], pose['y'], pose['yaw'], length, width)
    out = []
    for r in rows:
        if r['id'] == ego_id or math.hypot(r['x'] - pose['x'], r['y'] - pose['y']) > 15.0:
            continue
        cat = str(r.get('category') or '')
        if not (cat.startswith('vehicle.') or cat.startswith('human.')):
            continue
        B = C.box_corners(r['x'], r['y'], r['yaw'], 2.0 * r['extent'][0], 2.0 * r['extent'][1])
        if C.box_distance(A, B) <= 0.0:
            out.append((r['id'], 'pedestrian' if cat.startswith('human.') else 'vehicle', 'box'))
    return out


def drive_sequence(run_dir: str, options: dict | None = None) -> list:
    """The sequences of one recorded drive (module doc): usually one; a drive with dropped steps gives one per
    consecutive stretch. Raises ValueError when the drive lacks what is needed."""
    from . import worldmodel as WM
    o = dict(DEFAULTS, **(options or {}))
    if not os.path.isfile(os.path.join(run_dir, 'frames.jsonl')):
        raise ValueError(f'{run_dir}: no frames.jsonl')
    data = WM.load_run(run_dir)
    frames = data['frames']
    if data['actors'] is None:
        raise ValueError(f'{run_dir}: no actors.jsonl (the BEV\'s actors and the collision test need it)')
    if not data.get('lanes'):
        raise ValueError(f'{run_dir}: no lanes.json (the BEV\'s map and the route need it; record with a labelled '
                         'camera)')
    ego_id = data['ego_id']
    dt = 1.0 / data['fps']
    ego_row = next((a for rows in data['actors'].values() for a in rows if a['id'] == ego_id), None)
    length, width = (2.0 * ego_row['extent'][0], 2.0 * ego_row['extent'][1]) if ego_row else (4.5, 2.0)
    wheelbase = max(length * 0.6, 0.8)
    speeds = np.array([next((math.hypot(a['vx'], a['vy']) for a in data['actors'].get(f['step'], [])
                             if a['id'] == ego_id), f.get('speed_mps', 0.0)) for f in frames], float)
    target, tsrc = _target_speed(run_dir, speeds, o)
    recorded = any(isinstance(f.get('control'), dict) for f in frames)
    spec = DF.bev_spec(o.get('bev_spec'))
    lane_map = WM.lane_map_for(data['lanes'], [f['ego'] for f in frames], spec)
    # the stretches of consecutive steps
    cuts = [0] + [t + 1 for t in range(len(frames) - 1) if frames[t + 1]['step'] != frames[t]['step'] + 1] + \
        [len(frames)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        fr = frames[a:b]
        if len(fr) < 2:
            continue
        sp = speeds[a:b]
        xy = np.array([[f['ego']['x'], f['ego']['y']] for f in fr], float)
        yaw = np.array([f['ego']['yaw'] for f in fr], float)
        route = DF.route_from_lanes(data['lanes'], xy, yaw)
        feat = DF.Featuriser(route, target, dt)
        mon = DF.InfractionMonitor(dt, terminate_on=())
        T = len(fr)
        bev = np.zeros((T, spec.h, spec.w), np.uint8)
        state = np.zeros((T, DF.N_FEATURES), np.float32)
        act = np.zeros((T, 2), np.float32)
        rew = np.zeros(T, np.float32)
        terms = np.zeros((T, len(DF.REWARD_TERMS)), np.float32)
        n_derived = n_recorded = 0
        s_prev = None
        for t, f in enumerate(fr):
            rows = data['actors'].get(f['step'], [])
            c = f.get('control')
            if isinstance(c, dict):
                act[t] = DF.control_to_action(c)        # at t = 0 too: the featuriser's prev_* as behaviour cloning's
                n_recorded += t > 0
            elif t > 0:
                act[t] = derive_action(sp[t - 1], sp[t], yaw[t - 1], yaw[t], dt, wheelbase)
                n_derived += 1
            e = {'x': f['ego']['x'], 'y': f['ego']['y'], 'yaw': f['ego']['yaw'], 'speed_mps': float(sp[t]),
                 'length': length, 'width': width}
            x, facts = feat(e, rows, None, act[t].tolist(), ego_id)
            state[t] = x
            bev[t] = DF.bev_bits(f['ego'], rows, spec, lane_map, ego_id)
            pose = dict(f['ego'], speed_mps=float(sp[t]))
            on_road = bool(lane_map.sample(np.array([[pose['x'], pose['y']]]))[0] & WM.BEV_BIT['drivable']) \
                if lane_map is not None else True
            mf = {'contacts': _contacts(pose, length, width, rows, ego_id), 'off_road': not on_road,
                  'offroad_m': 0.0, 'wrong_way': False, 'outside_lanes': False, 'lateral': facts['lateral'],
                  'speed': facts['speed'], 's': facts['s'], 'route_length': facts['route_length'],
                  'travel_m': float(sp[t]) * dt}
            new, _ = mon.update(t, t * dt, pose, mf)
            if t > 0:
                ds = max(min(facts['s'] - s_prev, 5.0), -5.0)
                q = {'ds': ds, 'speed': facts['speed'], 'target': target, 'lateral': facts['lateral'],
                     'lane_width': facts['lane_width'], 'heading_err': facts['heading_err'], 'jerk': facts['jerk'],
                     'steer_rate': (act[t, 0] - act[t - 1, 0]) / dt if t > 1 else 0.0, 'off_road': not on_road,
                     'wrong_way': False, 'outside_lanes': False}
                tr = DF.reward_terms(q, new, None, dt)
                terms[t] = [tr[k] for k in DF.REWARD_TERMS]
                rew[t] = float(sum(tr.values()))
            s_prev = facts['s']
        name = os.path.basename(os.path.normpath(run_dir)) + (f'@{fr[0]["step"]}' if len(cuts) > 2 else '')
        src = 'recorded' if n_derived == 0 else ('derived' if n_recorded == 0 else 'recorded and derived')
        out.append({'bev': bev, 'state': state, 'action': act, 'reward': rew, 'terms': terms,
                    'terminal': np.zeros(T, bool), 'name': name, 'kind': 'drive', 'source': os.path.abspath(run_dir),
                    'first_step': int(fr[0]['step']),
                    'report': {'run': os.path.abspath(run_dir), 'steps': T, 'fps': data['fps'],
                               'action_source': src, 'actions_recorded': n_recorded, 'actions_derived': n_derived,
                               'control_recorded_in_run': recorded, 'target_speed_mps': round(target, 4),
                               'target_speed_source': tsrc, 'route_length_m': round(route.length, 2),
                               'reward_terms_unavailable': list(UNAVAILABLE_TERMS),
                               'infractions': mon.counts()}})
    if not out:
        raise ValueError(f'{run_dir}: no two consecutive steps')
    return out


# ============================================================================================ environment episodes
class OUNoise:
    """Ornstein–Uhlenbeck noise per action component, Euler-discretised per step: x += θ(μ − x) + σ·N(0, 1)."""

    def __init__(self, rng, theta=OU['theta'], sigma=OU['sigma'], dim=2):
        self.rng, self.theta, self.sigma = rng, float(theta), float(sigma)
        self.x = np.zeros(dim)

    def reset(self):
        self.x[:] = 0.0

    def __call__(self):
        self.x = self.x + self.theta * (0.0 - self.x) + self.sigma * self.rng.standard_normal(self.x.shape)
        return self.x.copy()


def scripted_action(state, names=DF.FEATURE_NAMES, gains: dict | None = None) -> np.ndarray:
    """A heuristic lane-keeping and speed controller on the state vector (module doc): steer against the lateral
    offset and the heading error; accel towards the target speed, braking hard when a lead or a vulnerable road user
    is closer than its time-to-collision or gap threshold. Not a good driver: a behaviour to collect from."""
    g = dict(SCRIPTED, **(gains or {}))
    i = {n: k for k, n in enumerate(names)}
    s = np.asarray(state, float)
    steer = -g['k_lat'] * s[i['lateral_offset_m']] - g['k_head'] * s[i['heading_error_rad']]
    accel = g['k_speed'] * (s[i['target_speed_mps']] - s[i['speed_mps']])
    if s[i['lead_present']] > 0.5 and (s[i['lead_ttc_s']] < g['safe_ttc_s'] or s[i['lead_gap_m']] < g['min_gap_m']):
        accel = -1.0
    if s[i['vru_present']] > 0.5 and s[i['vru_ttc_s']] < g['safe_ttc_s']:
        accel = -1.0
    return np.clip(np.array([steer, accel], float), -1.0, 1.0)


def make_behaviour(kind: str, seed: int, theta=OU['theta'], sigma=OU['sigma'], device='cpu'):
    """(act(obs, names) → action, reset(), description) of a behaviour (module doc)."""
    rng = np.random.default_rng(int(seed))
    noise = OUNoise(rng, theta, sigma)
    if kind == 'random':
        def act(obs, names):
            return np.clip(np.array([0.0, 0.3]) + noise(), -1.0, 1.0)
        desc = {'kind': 'random', 'centre': [0.0, 0.3]}
    elif kind == 'scripted':
        def act(obs, names):
            return np.clip(scripted_action(obs['state'], names) + noise(), -1.0, 1.0)
        desc = {'kind': 'scripted', 'gains': dict(SCRIPTED), 'heuristic': True}
    else:
        from .policy import load_policy
        pol = load_policy(kind, device)
        if pol.action['kind'] != 'continuous':
            raise ValueError('a behaviour policy must have continuous actions (noise is added to them)')

        def act(obs, names):
            return np.clip(np.asarray(pol.act(obs, names), float) + noise(), -1.0, 1.0)
        desc = {'kind': 'policy', 'policy': pol.path}
    desc.update(noise={'process': 'Ornstein-Uhlenbeck', 'theta': float(theta), 'sigma': float(sigma)})
    return act, noise.reset, desc


def collect(env, behaviour: str, episodes: int, out_dir: str, seed: int = 0, options: dict | None = None,
            theta=OU['theta'], sigma=OU['sigma'], device='cpu', progress=None, cancel=None) -> dict:
    """Run a behaviour in the environment (it must have the BEV on) for `episodes` episodes and write the experience
    folder (module doc); returns experience.json's document."""
    from .provenance import block, write_json
    if 'bev' not in env.observation_space.keys():
        raise ValueError('collecting world-model data needs the environment\'s BEV observation (bev on)')
    act, reset_noise, desc = make_behaviour(behaviour, seed, theta, sigma, device)
    names = env.feature_names
    os.makedirs(os.path.join(out_dir, 'episodes'), exist_ok=True)
    rows = []
    for k in range(int(episodes)):
        if cancel and cancel():
            break
        reset_noise()
        obs, info = env.reset(seed=int(seed) + k, options=options)
        S, B, A, R, TE, TR = [obs['state']], [DF.pack_bev(obs['bev'])], [np.zeros(2)], [0.0], [False], \
            [np.zeros(len(DF.REWARD_TERMS))]
        done = False
        summ = None
        while not done:
            a = act(obs, names)
            obs, r, term, trunc, info = env.step(a)
            S.append(obs['state'])
            B.append(DF.pack_bev(obs['bev']))
            A.append(np.asarray(info.get('action_applied', a), float))
            R.append(float(r))
            TE.append(bool(term))
            TR.append(np.array([info['reward_terms'][t] for t in DF.REWARD_TERMS], float))
            done = term or trunc
            summ = info.get('episode_summary')
        state = np.asarray(S, np.float32)
        if list(names) != list(DF.FEATURE_NAMES):               # in FEATURE_NAMES order, by name
            pos = {n: j for j, n in enumerate(names)}
            full = np.zeros((len(S), DF.N_FEATURES), np.float32)
            for j, n in enumerate(DF.FEATURE_NAMES):
                if n in pos:
                    full[:, j] = state[:, pos[n]]
            state = full
        np.savez_compressed(os.path.join(out_dir, 'episodes', f'{k:04d}.npz'), bev=np.asarray(B, np.uint8),
                            state=state, action=np.asarray(A, np.float32), reward=np.asarray(R, np.float32),
                            terms=np.asarray(TR, np.float32), terminal=np.asarray(TE, bool))
        row = {'file': f'episodes/{k:04d}.npz', 'seed': int(seed) + k, 'steps': len(S),
               'return': float(sum(R)), 'termination': (summ or {}).get('termination'),
               'route_completion': (summ or {}).get('route_completion')}
        rows.append(row)
        if progress:
            progress(k + 1, row)
    doc = {'schema': EXPERIENCE_SCHEMA, 'episodes': rows, 'behaviour': desc, 'options': options or {},
           'features': list(DF.FEATURE_NAMES), 'bev': env.ready.get('observation', {}).get('bev'),
           'environment': {'run_dir': env.run_dir, 'map': env.ready.get('map'), 'fps': env.ready.get('fps')},
           'steps': int(sum(r['steps'] for r in rows)),
           'provenance': block({'environment': os.path.join(env.run_dir, 'gym.json')}, seed=int(seed),
                               settings={'behaviour': desc, 'episodes': int(episodes), 'options': options or {}})}
    write_json(os.path.join(out_dir, EXPERIENCE), doc)
    return doc


def experience_sequences(path: str) -> list:
    """The sequences of an experience folder (collect)."""
    from .provenance import read_json
    doc = read_json(os.path.join(path, EXPERIENCE))
    if doc.get('schema') != EXPERIENCE_SCHEMA:
        raise ValueError(f'{path}: not an experience folder')
    if list(doc.get('features') or ()) != list(DF.FEATURE_NAMES):
        raise ValueError(f'{path}: recorded with other features than this build\'s')
    out = []
    for r in doc['episodes']:
        with np.load(os.path.join(path, r['file'])) as z:
            seq = {k: z[k] for k in SEQ_KEYS}
        seq.update(name=f'{os.path.basename(os.path.normpath(path))}/{os.path.basename(r["file"])[:-4]}', kind='env',
                   source=os.path.abspath(path), first_step=0,
                   report={'experience': os.path.abspath(path), 'episode': r['file'], 'steps': r['steps'],
                           'seed': r['seed'], 'termination': r['termination'], 'action_source': 'applied',
                           'behaviour': doc['behaviour'].get('kind')})
        out.append(seq)
    return out


def is_experience(path: str) -> bool:
    return os.path.isfile(os.path.join(path, EXPERIENCE))


def load_sources(paths, options: dict | None = None, log=None) -> tuple:
    """(sequences, notes) of drives and experience folders; a source that cannot give sequences is skipped with a
    note (the caller decides whether that is fatal). Each sequence's 'unit' is its source path: the split is by unit
    (a whole drive or a whole experience folder)."""
    log = log or (lambda s: None)
    seqs, notes = [], []
    for p in paths:
        p = os.path.abspath(p)
        try:
            got = experience_sequences(p) if is_experience(p) else drive_sequence(p, options)
        except (OSError, ValueError, KeyError) as e:
            notes.append(f'{p}: {e}')
            log(f'skipped: {e}')
            continue
        for s in got:
            s['unit'] = p
        seqs.extend(got)
        log(f'{os.path.basename(p)}: {sum(len(s["state"]) for s in got)} steps in {len(got)} sequence(s)')
    return seqs, notes


def check_bev_shape(seqs, spec) -> None:
    for s in seqs:
        if s['bev'].shape[1:] != (spec.h, spec.w):
            raise ValueError(f'{s["name"]}: its BEV is {s["bev"].shape[1:]}, not the model\'s {(spec.h, spec.w)} '
                             '(collect with the same BEV spec)')
