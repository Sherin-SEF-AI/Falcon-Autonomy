# Driving policies: the environment, behaviour cloning and PPO

Falcon's training lab for driving: a learning **environment** on CARLA with the Gymnasium API, **behaviour cloning**
from recorded drives, **PPO** with a scenario **curriculum**, and evaluation of any policy through the environment.
The definitions live in code, with the exact wording in the module docstrings: `core/drivefeatures.py` (observations,
actions, rewards, infractions), `workers/gym.py` (the environment worker), `core/gymenv.py` (the client),
`core/imitation.py`, `core/rl.py` and `core/policy.py`.

## How it runs

The simulation and the learning run in two Pythons, as everywhere in Falcon ([architecture](architecture.md)):

- **The environment worker** (`python -m falcon_autonomy.workers gym CONFIG`) runs in CARLA's Python. It loads the
  world, steps it in synchronous mode at a fixed step, places each episode (a scenario through the scenario director,
  or free driving), applies the agent's actions, and computes observations, rewards and infractions.
- **The client** (`FalconDriveEnv`, in the app's Python with torch) starts the worker through `CarlaService`, exactly
  like every CARLA job: it takes the machine-wide CARLA lock, waits for memory and starts Falcon's own server when
  needed. Commands (`reset`, `step`, `close`) go to the worker's stdin as JSON lines; replies come back on a Unix
  socket the client listens on, in the live link's framing (a JSON header and a binary payload). The observation
  arrays travel as raw bytes (the BEV as packed bits).

```python
from falcon_autonomy.core.gymenv import FalconDriveEnv, gym_config, make_service
svc = make_service(workspace=project.workspace)
cfg = gym_config(project, 'world-town10', out_dir, svc.port, svc.tm_port, scenario='lead_brake', bev=True)
with FalconDriveEnv(cfg, svc) as env:
    obs, info = env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step([0.0, 0.3])
svc.shutdown()                       # stops Falcon's server if this process started it
```

From the command line:

```sh
falcon policy bc --run RUN --run RUN2 ... --out POLICY [--epochs N --seed S --bev]
falcon policy offline --policy POLICY --run RUN ...                     # action error on drives
falcon policy ppo PROJECT --out DIR [--scenario KEY | --curriculum default|FILE] [--updates N --steps N] [--resume]
falcon policy rollout PROJECT --policy POLICY --out DIR [--scenario KEY] [--episodes N] [--seed S]
```

`ppo` and `rollout` also take `--world`, `--rig`, `--motion kinematic|physics`, `--fps`, `--bev`, `--traffic N` and
`--option KEY=VALUE` (any PPO setting).

## The Gymnasium API, and what of it is honoured

- `FalconDriveEnv` subclasses `gymnasium.Env` and uses `gymnasium.spaces` when gymnasium is importable. It is not
  installed on the reference machine, and Falcon does not need it: a minimal shim stands in (`Box`, `Discrete`, `Dict`
  with `shape`, `dtype`, `low`, `high`, `n`, `sample()` from a seeded generator, `contains()`, `seed()`).
- `reset(*, seed=None, options=None) -> (obs, info)`. With a seed, that episode's seed is exactly `seed`, and the env's
  generator is reseeded; resets without a seed draw their episode seeds from it, so a sequence of resets is
  reproducible from the first seed. **Unlike Gymnasium**, an env that was never seeded draws from its config's seed, not
  from entropy.
- `step(action) -> (obs, reward, terminated, truncated, info)`. `terminated` is a terminal state of the task;
  `truncated` is the time limit. Calling `step()` after either, without `reset()`, raises (Gymnasium leaves it
  undefined). A continuous action outside the box is clipped; a non-finite value or an index out of range is refused.
- `observation_space` is a `Dict`: `state` always, `bev` and `camera` when configured. `action_space` is
  `Box(-1, 1, (2,), float32)` or `Discrete(25)`.
- `close()` ends the worker, which tidies the server; it is idempotent, and the env is a context manager.
- `render()` returns `None` (`render_mode` is `None`). Ask for the `camera` observation instead. `metadata` is
  `{'render_modes': []}`, `spec` is `None`.
- **No vector API.** One CARLA server can host one synchronous master, and Falcon runs one CARLA job per GPU. A second
  environment would need a second server, and two do not fit this machine's memory. PPO therefore collects from one
  environment, sequentially.
- **Not verified:** gymnasium's `env_checker`, because gymnasium is not installed here.

## Episodes

`reset` options override the config's default episode:

| option | meaning |
|---|---|
| `scenario` | a built-in key (`lead_brake`, `cut_in`, … — [scenarios](scenarios.md)), a scenario document, or `None` for free driving |
| `values` | scenario parameters; the parameters not given are drawn from the scenario's distributions with the episode seed |
| `motion` | `kinematic` (default; repeatable) or `physics` (CARLA's vehicle physics; not bit-for-bit repeatable) |
| `route_m` | the route's length (free driving: 200 m; a scenario: the director's 400 m path) |
| `max_episode_s` | the time limit (free driving: 60 s; a scenario: its `duration_s`) |
| `target_speed_kmh` | free driving: the speed asked for (default 30). A scenario asks for its ego speed |
| `start_speed_kmh` | free driving: the speed at the start (default 0). A scenario starts at its ego speed |

**A scenario episode** is placed by the director, as in a scenario drive. The agent drives the ego along the
director's lane path; the director drives the other actors and fires the timeline.

**A free-driving episode** starts at the first spawn point, in seeded order, that is outside a junction and whose lane
runs at least min(`route_m`, 30 m). The route follows the lane from there, taking the straight-most branch through
junctions.

**Every reset** removes the previous episode's actors. It draws everything from the episode seed, resets the traffic
lights' cycle (`World.reset_all_traffic_lights`) and, in kinematic motion, places the ego and every scenario vehicle
exactly at its planned start pose, so CARLA's physics settling never enters the episode. Background traffic is off by
default. When asked for (`--traffic`), it is spawned once and persists across episodes, and episodes are then not
repeatable.

## Observations

The `state` vector (float32, raw units, in this order). Its definitions are shared with behaviour cloning, so a cloned
policy sees exactly the environment's features. A quantity that does not exist takes its cap and a 0 presence flag,
never NaN.

| feature | unit | definition |
|---|---|---|
| `speed_mps` | m/s | the ego's speed |
| `target_speed_mps` | m/s | the speed the episode asks for |
| `accel_mps2` | m/s² | (speed − the previous step's) / dt |
| `lateral_offset_m` | m | signed distance from the route's centre line (+ right) |
| `heading_error_rad` | rad | ego yaw − the route's heading (+ turned right) |
| `lane_width_m` | m | the route lane's width |
| `curvature_10m`, `_25m`, `_50m` | 1/m | the route's curvature 10, 25 and 50 m ahead (Δheading/Δs over ±5 m; + right) |
| `lead_present`, `lead_gap_m`, `lead_closing_mps`, `lead_ttc_s` | | the nearest vehicle ahead in the ego's corridor: gap bumper to bumper along the route (60 m when none), closing speed along the route, gap/closing (10 s cap) |
| `vru_present`, `vru_ahead_m`, `vru_lateral_m`, `vru_lat_speed_mps`, `vru_ttc_s` | | the nearest pedestrian or cyclist in the path, or entering it within 3 s at constant lateral speed, up to 40 m ahead |
| `tl_known`, `tl_red`, `tl_yellow`, `tl_green`, `tl_distance_m` | | the first traffic light whose stop line is on the route ahead (from the ego's front, 60 m when none). `tl_known` is 0 where states are unknown (recorded drives) |
| `route_progress`, `route_remaining_m` | | s / route length; the metres left (capped at 500) |
| `prev_steer`, `prev_accel` | | the previous step's action |

`bev` (optional): core.worldmodel's own BEV raster, the same function `falcon derive worldmodel` uses. It is
8 bit planes (8 × H × W, 0/1): drivable, centerline, lane boundary, vehicle, pedestrian, cyclist, other, ego. It is
ego-centred with the heading up, and by default 0.5 m pixels, 32 m ahead, 8 m behind and 16 m to each side
(80 × 64). The worker writes `lanes.json` for it.

`camera` (optional): an RGB image, 128 × 72 at 90° by default, 1.5 m ahead of the ego's origin and 1.6 m up.

## Actions

**Continuous** `[steer, accel]` ∈ [−1, 1]². `steer` is CARLA's (+ right). For `accel` ≥ 0, throttle = accel and
brake is 0; for `accel` < 0, brake = −accel and throttle is 0. The hand brake and reverse are not used.

**Discrete**: index = 5 · i_steer + i_accel, with steer ∈ {−0.6, −0.2, 0, 0.2, 0.6} and accel ∈
{−1, −0.3, 0, 0.3, 0.7}.

In kinematic motion, the ego follows the director's kinematic bicycle model: full throttle accelerates at 3 m/s² and
full brake decelerates at 8 m/s², and the steer maps to a wheel angle of up to 70°.

## Rewards

The reward is the sum of these terms. Each term is in `info['reward_terms']` at every step and summed per episode
in the episode summary, so they can be audited. The weights are **heuristic, not tuned**; a config's `reward`
overrides them.

| term | per step | default weight |
|---|---|---|
| `progress` | + w · Δs (route metres gained; negative going back) | 0.1 /m |
| `speed` | − w · k · \|v − v\*\| / max(v\*, 1) · dt; k = 4 above the target speed, else 1 | 0.2 /s |
| `lane_offset` | − w · min((lateral / half lane width)², 4) · dt | 0.3 /s |
| `heading` | − w · min(\|heading error\|, π/2) · dt | 0.2 /s |
| `jerk` | − w · min(\|jerk\| / 2 m/s³, 10) · dt (longitudinal, from the speed) | 0.02 /s |
| `steer_rate` | − w · min(\|Δsteer\| / dt / 1 s⁻¹, 10) · dt | 0.02 /s |
| `collision` | − w per new collision (vehicle or static); − w_ped per pedestrian | 50; 100 |
| `off_road` | − w_onset per off-road onset − w · dt while off the road | 20; 5 /s |
| `wrong_way` | − w · dt while in a lane of the opposite direction | 2 /s |
| `outside_lanes` | − w · dt while on a lane that is not the route's (and not wrong way) | 0.5 /s |
| `red_light`, `stop_sign`, `route_deviation`, `agent_blocked` | − w per infraction | 20, 10, 20, 10 |

Two of these terms were shaped by what PPO found on the fake:

- **The one-off off-road penalty.** With only a per-second penalty, ending an episode early by leaving the road
  escaped the per-step costs, and PPO learnt to do it.
- **The overspeed factor k.** With overspeed costed like underspeed, finishing the route early at twice the target
  speed paid, and PPO learnt that too.

## Termination and truncation

An episode **terminates** on a new collision, on being off the road by more than 1 m beyond the nearest driving
lane's edge, on a route deviation (more than 8 m from the route), on being blocked (below 0.1 m/s for 20 s), and when
the route is complete (within 2 m of its end). These are the config's `terminate_on` (default `collision`,
`off_road`, `route_deviation`, `agent_blocked`) and `thresholds`. `red_light`, `stop_sign` and `wrong_way` can be added
to `terminate_on`.

It is **truncated** at the time limit. Both flags are in `info['termination']`: the reason, or `time_limit`.

## Infractions

Each infraction is recorded once, at its onset, with the step, the simulation time, the CARLA actor ids involved and
the ego's pose:

```json
{"kind": "collision_vehicle", "step": 83, "sim_t": 4.15, "actors": [101],
 "ego": {"x": 83.58, "y": 3.5, "z": 0.0, "yaw": 0.0, "speed_mps": 23.4}, "detail": {"source": "box"}}
```

The condition kinds (`off_road`, `wrong_way`, `outside_route_lanes`) also get `end_step`, `end_sim_t` and
`distance_m`, the distance driven in that condition, when it ends or the episode does.

| kind | definition |
|---|---|
| `collision_vehicle` | the ego's box overlaps a vehicle's (two-wheelers included), or the collision sensor reports one. Counted once per actor until it has been out of contact for 1 s |
| `collision_pedestrian` | the same with a walker |
| `collision_static` | the collision sensor reports a hit with anything that is not a vehicle or a walker. In kinematic motion the ego has no physics, so this is detected only where CARLA reports it (not verified on a server) |
| `red_light` | the ego's front centre crosses a traffic light's stop line (`TrafficLight.get_stop_waypoints`) while the light is red, driving in the lane's direction. Yellow is not an infraction |
| `stop_sign` | the ego's centre enters and leaves a stop sign's trigger volume without its speed falling below 0.1 m/s inside it (a simplification of the Leaderboard's test; unit-tested only, not on a map with stop signs) |
| `off_road` | the ego's centre is on no driving lane (CARLA's map, `project_to_road=False`). `detail.max_offroad_m` is the farthest beyond the nearest lane's edge |
| `wrong_way` | the ego's centre is on a driving lane, outside junctions, whose direction differs from its heading by more than 135° |
| `outside_route_lanes` | the ego's centre is on a driving lane, outside junctions, that is not one of the route's lanes (road id, lane id) and not wrong way: a lane change, for instance |
| `route_deviation` | more than 8 m from the route (once per episode) |
| `agent_blocked` | below 0.1 m/s for 20 s |

Every episode, finished or interrupted, is a line of `episodes.jsonl` in the environment's run folder. The line holds
the seed, the options, the concrete scenario (with its parameter values), the start pose, the route, the return and
its terms, the termination, the infractions and their counts, the route completion, and — unless `log_actions` is off
— every action. Under kinematic motion, `reset(seed, options)` followed by those actions replays the episode
exactly. That is the basis for replaying an infraction.

## Determinism

With kinematic motion, no background traffic and the same seed, an episode is a function of the seed, the options and
the actions. Measured on the fake: two `cut_in` episodes with the same seed and the same 60 actions give identical
observations (state and BEV) and rewards (`tests/test_gym.py`). On a real server: see *Measured* below. CARLA's
physics (`motion: physics`, or traffic-manager traffic) is not bit-for-bit repeatable ([scenarios](scenarios.md)).

## Behaviour cloning

`falcon policy bc` learns (observation, action) pairs from recorded drives (`core/imitation.py`). Drives record
the ego's control from 0.6 ([dataset format](dataset-format.md#frames)). It replays each drive through the
environment's featuriser, with these inputs:

- **The ego and the actors** come from frames.jsonl and actors.jsonl.
- **The route** is built from lanes.json: the lane centre points matched to the lanes the drive went through. A drive
  without lanes.json (no labelled camera) is refused.
- **The target speed** is the scenario's ego speed for a scenario drive. Otherwise it is `target_speed_kmh` if given,
  or else the drive's median moving speed (a heuristic; the report says which one was used).
- **Traffic lights** are unknown, so the `tl_*` features are left out of a cloned policy by default.
- **The action** for step *i* is step *i + 1*'s recorded control, as [steer, throttle − brake].

Training:

- **The split is by drive.** At least two drives are needed; `val_fraction` (0.25 by default) of them, at least one,
  are held out.
- **The network** is an MLP (128-128, ReLU, tanh outputs), or with `--bev` a CNN over the BEV beside the state.
- **The loss and optimiser** are MSE with Adam. The minibatch order is seeded, and the weights of the epoch with the
  lowest validation loss are kept.

The metrics are the mean absolute and RMS error per action component, on both splits, for the policy and for the
constant baseline (the training split's mean action).

Its limits:

- **Cloning copies its teacher's faults.** The autopilot's or the lane follower's mistakes, such as the lane
  follower's stop-and-creep chatter behind a stopped car, are learnt too.
- **Errors compound in closed loop** (covariate shift), so a low offline error does not mean the policy drives well.
  Evaluate it with `falcon policy rollout`.
- **The `prev_*` inputs flatter the offline error.** They carry much of the next action, a known shortcut; exclude them
  (`--option exclude_features='[...]'`) to see the difference.

## PPO

`falcon policy ppo` (`core/rl.py`) is PPO written in torch: no third-party RL library is needed or installed. It
follows Schulman et al. 2017 and the usual implementation details (Huang et al. 2022).

**The networks.** The actor and the critic are separate tanh MLPs (64-64, orthogonal initialisation). A continuous
policy is a Gaussian with a state-independent log standard deviation; a discrete one is a categorical over the 25
actions.

**The update.**

- Advantages are GAE(λ = 0.95), with γ = 0.99, and are normalised per minibatch.
- The objective is the clipped surrogate (ε = 0.2), plus a value loss (coefficient 0.5) and an entropy bonus
  (coefficient 0 by default).
- Gradients are clipped at a norm of 0.5. Adam runs at 3·10⁻⁴.
- Each update collects 1024 steps and makes 10 passes over minibatches of 64.
- A time limit is bootstrapped with γ·V(final state), not treated as terminal.

**Normalisation.** Observations are normalised with running statistics. Rewards are scaled by the running standard
deviation of the discounted return; the logged returns are always the environment's own.

**Seeding and device.** Runs are seeded: the same seed gives the same updates (`tests/test_rl.py`). PPO runs on
**the CPU** by default: during RL the CARLA server owns the GPU, Falcon never runs a GPU job beside a CARLA job, and
these networks are too small to gain from a GPU.

**Logs and checkpoints.**

- `episodes.jsonl` has a line per episode: stage, seed, options, return, length, termination, completion, infraction
  counts, reward terms and success.
- `updates.jsonl` has a line per update: the losses, approximate KL, clip fraction, explained variance, environment
  steps per second, the mean return of the last 20 episodes, and the action standard deviation.
- Checkpoints go to `checkpoints/ckpt-<update>.pt`, the last three kept, with `latest.json` naming the newest.
  `--resume` continues from it, with the updates, episode numbers, normalisers and RNG streams carrying on. The
  episode in progress is not resumed: a new one starts.
- `run.json` holds the provenance.

**The policy.** It is exported to `DIR/policy/` at every checkpoint and at the end.

### The curriculum

A curriculum is a list of ordered stages. Each stage has:

- a scenario: a built-in, or `null` for free driving;
- parameter ranges, `[lo, hi]` or `{"choices": [...]}`;
- other reset options;
- a success criterion: none of the failure infractions, an ending that is not a failure, and at least
  `min_progress_m` of progress.

After each episode its success is recorded. When the stage has at least `window` episodes and the success rate over
the last `window` reaches `threshold`, the next stage starts. The last stage is never left. `--curriculum default` is a
heuristic ordering, not tuned, at `window` 20 and `threshold` 0.8:

1. Keep the lane in free driving at 30 km/h.
2. Gentle lead braking.
3. Lead braking over its full ranges.
4. Cut-in.
5. Pedestrian crossing.

Its first stage starts at the target speed. From a standstill, zero-mean exploration noise never builds speed, because
braking is 8/3 as strong as throttle, and PPO on the fake settled on standing still (measured below). Promotions are
logged in `episodes.jsonl`.

## Policy folders and evaluation

Behaviour cloning and PPO write the same folder, read by one loader (`core/policy.py`):

- `policy.json` holds the features it reads, **by name**, the input normaliser, the action kind, the network's
  architecture and the provenance;
- `weights.pt` holds the network, with its SHA-256 checked on loading;
- `metrics.json` and `run.json`.

Acting is deterministic: the tanh output for behaviour cloning, the clipped Gaussian mean for continuous PPO, the
argmax for discrete PPO.

`falcon policy rollout` runs a policy for N episodes with seeds `seed`, `seed + 1`, …. It writes `rollout.json`: every
episode summary (each replayable) and the aggregate — mean return, route completion, success rate, collisions per
episode, infractions per km and terminations — with provenance.

## Leaderboard-style evaluation

`falcon policy leaderboard PROJECT --policy DIR --out DIR` (Training ▸ Leaderboard) drives a policy over a route set:
each chosen scenario (default: every built-in) with `--seeds-per` seeds, plus `--free` free-driving routes. Each
route is scored as the CARLA Leaderboard scores one (`core/leaderboard.py`):

- **Route completion** is the share of the route driven, less the distance driven off the road, the wrong way or
  outside the route's lanes. The Leaderboard does not count those towards completion either.
- **Infraction penalty** is the product of the coefficients of the infractions, as the Leaderboard 1.0 and 2.0
  documents state them: a pedestrian collision 0.50, a vehicle 0.60, a static object 0.65, a red light 0.70, a stop
  sign 0.80.
- **Driving score** is completion × penalty, and the set's score is the mean over routes. Each of the three gets a
  95 % bootstrap interval over routes; infractions per km are given by kind.
- **Shutdowns:** route deviation and an agent blocked end a route, as the Leaderboard's shutdowns do. They add no
  penalty; completion is what was driven until then.

**Not scored**, and said in every result: Leaderboard 2.0's minimum-speed failures, scenario time-outs and failure to
yield to an emergency vehicle, because Falcon has no such events. **Falcon's routes are not the Leaderboard's**, so
these numbers compare policies with each other, not with its public results.

`leaderboard.json` holds the scores, per route and in aggregate, with provenance. `episodes.jsonl` holds every episode
with its actions and the environment settings it ran under.

### Replaying an infraction

`falcon policy replay PROJECT --episodes EPISODES.jsonl --episode N --infraction K --out DIR` (Training ▸ Leaderboard ▸
*Replay the selected infraction*) rebuilds the episode's environment from the episode's own settings and adds a
640 × 360 chase camera, which does not change the motion. It then resets with the episode's seed and options, applies
its logged actions, and keeps the frames from 3 s before the infraction to 1 s after it.

The replay checks that it meets the same infraction at the same step (`match` in `replay.json`). An episode with
physics motion or background traffic is refused, with the reason: neither repeats in CARLA 0.9.16.

## Measured

All on the reference machine, 2026-10-28, often loaded by other jobs; wall times vary with that load.

### A real CARLA 0.9.16 server

Town10HD_Opt; Falcon's own server, capped at 8 GB.

**The environment.**

- It was ready in 5.1 s with the BEV on (lanes.json and the lane raster included). It found 15 traffic lights and 3
  stop signs.
- A reset took 0.1 s.
- It ran at 57–101 steps/s with the BEV and about 200 steps/s on the state alone (164–207 in PPO's collection).

**Determinism.** Three `cut_in` episodes with seed 3 and the same 159 actions were identical: every state vector, BEV,
reward and ego pose. An earlier run found the BEV's ego channel differing by 8 pixels at 1–3 of 197 steps. CARLA's
snapshot of a pose set with `set_transform` varies between runs by enough to move a pixel edge, so the worker now draws
kinematic actors at the director's exact poses.

**Recorded control.** A 4 s autopilot drive with 5 vehicles recorded `control` at every step: throttle up to 0.85,
brake up to 0.7, steer within −0.005…0.014.

**PPO for a few minutes does not drive.** Six updates took 6,144 steps and 41 s, 31 s of it in the environment, in
free driving from 30 km/h (the default curriculum's first stage).

- The 57 episodes lasted 5.4 s on average.
- 36 ended off the road, 12 in a route deviation and 9 at the time limit.
- The mean return of the last 20 was −34, and none succeeded.

On the fake's straight road, the same stage needed about 50,000 steps before every episode finished its route
(below). Town10's curves need more; that was not measured.

**A start under memory pressure failed.** One server start, with 6.5 GB free while another job ran, kept the worker's
setup waiting more than 40 s. The watchdog ended it with `server_lost`. On a loaded machine raise `gym_config`'s
`timeout` (20 s by default; the watchdog allows twice it).

### The fake carla

The fake is the tests' stand-in (straight road, simple traffic). It is not CARLA.

**The environment** ran at about 3,000 steps/s on the state alone, and about 100 steps/s with the BEV and a
128 × 72 camera (the fake renders in numpy).

**Behaviour cloning.** The teacher was the lane follower, on three 12 s `lead_brake` drives: two for training, one
for validation, 30 epochs, 1.3 s. On validation:

| | policy | constant baseline |
|---|---|---|
| action mean absolute error | 0.221 | 0.247 |
| accel mean absolute error | 0.42 | 0.49 |

On training it was 0.129 against 0.157. Steer is constantly 0 on the straight road. The gain is modest: the teacher's
stop-and-creep chatter behind a stopped car cannot be predicted from the state.

**PPO, free driving.** Each run was 150 updates and 153,600 steps, with the default curriculum's first stage: 30 km/h,
a 150 m route, 20 s.

| variant | result after 153,600 steps |
|---|---|
| from standstill | settled on **standing still**: return −4.0, the speed term only |
| from 30 km/h, overspeed costed like underspeed | finished its route by about 75,000 steps, but at **about twice the target speed**: finishing early escapes the per-step costs. Return +9.1 |
| from 30 km/h, overspeed costed four times (the default) | first finished the route after 15,300 steps, and finished it in every episode from about 53,000 steps. Final return +7.9, the last 20 all successful, but at about 46 km/h. It has not converged: driving the same route at 30 km/h would score about +13. Wall time was 131–657 s with the machine's load |

These runs are why the defaults are what they are: the one-off off-road penalty, the overspeed factor, and a first
stage that starts at speed. They also show what to expect. Even on a straight road with no traffic, PPO needs tens of
thousands of steps to keep a lane, and its reward shaping is exploited wherever a loophole is left.

## Limits

- **Kinematic motion is not vehicle dynamics.** The ego and the scenario vehicles follow a kinematic bicycle model:
  no tyres, suspension or load transfer, and no collision dynamics, since collisions are measured on the boxes. A
  policy trained there meets different dynamics under `physics` and on a real car.
- **The route is a lane path.** It takes the straight-most branch through junctions, and turns only in the
  `unprotected_turn` scenario. There is no global route planning and no lane changes along the route.
- **Static collisions** in kinematic motion are only what CARLA's collision sensor reports.
- **Stop signs** follow a simplified test. `outside_route_lanes` and `wrong_way` are not judged inside junctions.
- **The fake carla in the tests is not CARLA.** Its road is straight and its traffic simple. The numbers measured on
  it show that the code works, not how well a policy drives.
