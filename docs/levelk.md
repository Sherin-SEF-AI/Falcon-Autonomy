# Level-k traffic: game-theoretic road users, calibrated and searched

CARLA's traffic manager keeps its vehicles on their lane centres and gives way by rule. Bengaluru's traffic does not:
a two-wheeler creeps into a stream that leaves it no gap until a driver yields, an autorickshaw squeezes past a bus,
a rider filters along the lane line. Level-k road users reproduce such behaviour without scripting it: each one
chooses its acceleration and lateral position by best-responding to what it predicts the others will do, a level
below its own. The same model drives scenario actors, adversaries against the ego and background traffic; a
differentiable twin calibrates its parameters on logs and searches adversarial scenarios through a gradient.

Code: `core/levelk.py` (numpy only — it runs per tick inside CARLA's Python, where there is no torch),
`core/trafficgraph.py` (torch, the app's Python: calibration and the differentiable search),
`workers/levelk_traffic.py` (background traffic), the level-k drivers in `workers/director.py` and
`workers/adversary.py`, `core/recon/levelk_actors.py` (road users in the neural closed loop), `core/levelk_nearmiss.py`
(calibration on your dashcam near-misses). In the GUI: Worlds ▸ traffic profile ▸ *Behaviour model* (CARLA traffic
manager or level-k, k, λ and α per kind, *Load calibrated parameters…*: [worlds](worlds.md#indian-mixed-traffic)) and
Replay ▸ reconstruction ▸ *Level-k road users* beside *Drive a policy here*. Each module's docstring is the
reference; this page is the guide.

```sh
falcon levelk simulate [--scene crossing|merge|filtering|pedestrian|platoon] [--level ID=K]... [--param ID.KEY=V]...
                       [--seconds S] [--seed N] [--mode sample|best] [--lht] --out DIR
falcon levelk check [--scene NAME]... [--seeds N] [--seconds S] [--out DIR]
falcon levelk calibrate --log FILE... [--agent ID]... [--steps N] --out DIR
falcon levelk calibrate --from-nearmiss INGEST_DIR [--kind KIND]... [--depth-draws N] [--seed N] [--out DIR]
falcon recon closedloop RECON --actors levelk [--levelk-params JSON|FILE] [--levelk-mode sample|best] ...
```

## The model

**Agents.** Five types — car, two-wheeler, autorickshaw, bus, pedestrian — each with a box (a CARLA actor's own
measured box when it has one), a top speed, an acceleration limit and its lateral targets: every vehicle may keep
its lane or change a lane; a two-wheeler may also ride on the lane line beside a car (lane sharing, filtering) and an
autorickshaw squeeze 0.9 m to either side. An agent moves along a reference path (its lane centre) at arc length *s*
with a lateral offset *d*, inside the carriageway's same-direction lanes (`RefPath.bounds`, read from CARLA's map).

**Plans.** A plan is a constant acceleration from {−4, −3, −2, −1, 0, 1, 2} m/s² and a lateral target, held for the
horizon (3 s in steps of 0.25 s). Rollouts start from the agent's current acceleration and reach the plan's at the
jerk limit (8 m/s³, `core.advfeatures.JERK_ACT`), as the plan is then executed: agents re-plan every 0.25 s and
execute every tick through that limit, so level-k motion stays within `core.advfeatures.REALISM_LIMITS` by
construction.

**Utility** (`utility()`, one function evaluated on numpy arrays here and on torch tensors in
`core.trafficgraph`):

    U = W_prog (1 + α) · progress + W_speed · speed − W_comf · comfort − W_safe (1 − α) · barrier
        − W_coll · collisions − W_light · compliance · red − W_lane · lane (+ W_target · target)

progress (metres gained, at most `v_des · horizon`, normalised), speed (the squared relative difference from
`v_des`), comfort (acceleration and lateral move), the TTC/gap barrier (Σ over the others of exp(−gap / (2 m +
v · t_gap / 2)), its scale growing with speed and the agent's gap acceptance), collisions (a plan within 0.3 m +
0.15 m/s · t of another box — the margin grows with the prediction's error — or that would hit a leader braking at
6 m/s²: the room for a hard stop of Responsibility-Sensitive Safety's longitudinal rule, Shalev-Shwartz et al. 2017,
its formula not evaluated), a red stop line crossed, and the final offset from a lane centre (a two-wheeler: or a
lane line). Assertiveness α moves weight from safety to progress; collisions cost the same for every α.

**Choice.** Quantal response (McKelvey & Palfrey 1995): P(plan) ∝ exp(λ U). Simulation samples it with a seeded
generator (mode `sample`) or takes the most likely plan (`best`).

**Levels** (Stahl & Wilson 1995; Costa-Gomes, Crawford & Broseta 2001; in driving Oyler et al. 2016 and Li et al.
2017, IEEE TCST 26(5)).
* Level 0 is non-strategic: IDM car following (Treiber, Hennecke & Helbing 2000) on the nearest road user in its
  corridor, braking at most 4 m/s², and gap acceptance at a conflict with a road user that has the right of way — it
  goes only when the other arrives later than `t_gap`, else it stops before the conflict; a two-wheeler behind a
  slower leader filters to the lane line when the line is clear.
* Level k ≥ 1 best-responds (quantally) to the others predicted at level k − 1. A level-0 other is predicted
  **reactively**: its IDM responds, step by step within the horizon, to each candidate plan — which is what lets a
  level-1 agent expect a level-0 driver to yield to it. A higher-level other commits to its own most likely plan.
  Uncontrolled road users — the ego under test, CARLA's own traffic — are predicted at level 0 by everyone.

**Parameters** per agent: level k, α ∈ [0, 1], λ, `v_des`, `t_gap` and light compliance (`PARAM_DEFAULTS`, bounds
in `PARAM_BOUNDS`). A traffic profile names them per kind (`core.project.levelk_params`): `{"motorcycle": {"k": 2,
"lambda": 4.0, "alpha": 0.7}, "*": {...}}`. Adversaries default to `ADVERSARY` (level 1, α 0.8, λ 20, t_gap 1 s) and
target the ego: closeness to it is a gain (`TARGET_WEIGHT`) while collisions stay as costly. In CARLA the desired
speed defaults to the type's within the road's speed limit (`Vehicle.get_speed_limit`).

## Where it drives

| Use | How | Code |
|---|---|---|
| Background traffic | a traffic profile with `behaviour: {"model": "levelk", "levelk": {…}}` — its vehicles go to `workers/levelk_traffic.py` instead of the traffic manager, in every worker that spawns traffic (drives, the learning environment, Live, stacks, HIL) | `workers/scene.py` hooks, `LevelKTraffic` |
| A scenario actor | `"driver": "levelk", "levelk": {"k": 1, "alpha": 0.5, "target": "ego", "priority": false}` on an actor; its timeline's motion actions are ignored (logged); its desired speed is the scenario's | `Director.levelk_step`, `drive_levelk` |
| Adversaries | `falcon adversary eval … --baselines …,levelk`, or the reset option `controller: "levelk"`: the roster's adversaries are driven in the worker, tracked and judged by the same fault attribution, rules and dynamics as learned ones | `workers/adversary.py levelk_actions` |
| The neural closed loop | `falcon recon closedloop RECON --actors levelk`, or *Level-k road users* beside *Drive a policy here* (Replay ▸ reconstruction): the reconstruction's tracked road users react to the policy's ego ([below](#in-the-neural-closed-loop)) | `core/recon/levelk_actors.py` on `Controller.step` (below) |

Level-k vehicles move kinematically (CARLA's physics off; the director's kinematic bicycle model, the same in
either motion), along their lane paths with their lateral offsets, placed for the next tick in one synchronous batch.
Their brake lights follow the executed deceleration (`set_light_state`), so a camera and the token observations see
them brake. The episode's `run.json` `traffic.levelk` reports the vehicles by type, decisions, steps, paths renewed
and the parameters.

### The per-tick interface (for NeuralSim)

```python
from falcon_autonomy.core import levelk as LK
ctl = LK.Controller(seed=0, dt=0.05, lht=False)          # params per id or '*' optional: Controller({'*': {...}})
actions = ctl.step(t, agents)
# agents: [{'id', 'type' (car | two_wheeler | autorickshaw | bus | pedestrian, or a scenario kind),
#           'path': LK.RefPath(points (N, 2), bounds=(lo, hi) | None), 's', 'd', 'v',
#           'controlled': bool, 'priority': bool, 'd_ref': its lane centre's offset, 'params': {k, lambda, alpha,
#           v_des, t_gap, compliance}, 'length', 'width', 'stops': [(s_stop, 'Red' | 'Green' | …)],
#           'target': an id | None, 'weights': {…} | None}, …]
# actions: {id: {'accel': m/s² to apply now (jerk-limited), 'lateral': target offset (m), 'plan_accel', 'level',
#                'brake_light': bool}}  — for every controlled agent
```

`LK.execute(agent, plan_accel, lateral, dt)` advances a state exactly as the controller assumes (the simulator may
instead integrate its own vehicle model and pass the observed state back next tick). Everything is numpy: the call
costs a few milliseconds per controlled agent and decision on this machine's CPU.

## Calibration (`core/trafficgraph.py`)

Decision samples come from a simulation (`trajectories.json`) or a CARLA run's recorded trajectories (an
`episodes.jsonl` with trajectories; each actor's driven polyline is its reference path): every 0.25 s an agent's
plan options with their features, and the plan it was seen to follow (the nearest option to its observed motion).
The likelihood of the observed plans is the quantal response of the same `utility()` on torch tensors, so
log P = log softmax(λ U(θ))[observed] is differentiable in θ = (α, λ, v_des, t_gap) through a smooth
reparametrisation (α a sigmoid, the others softplus above their bounds). Maximisation: Adam, then L-BFGS; standard
errors from the inverse Hessian, mapped by the delta method (Wald 95 % intervals).

Two assumptions are stated in every result: the others' future motion is taken as **observed** (what they did over
the horizon — a quantal best response to the others' actual behaviour, an inverse-planning assumption), not as the
model's own level-(k − 1) prediction; and consecutive decisions of one agent are treated as independent, which they
are not, so the intervals are optimistic.

## In the neural closed loop

`core.recon.levelk_actors.LevelKActors` is an `ActorController` of the neural simulator
([reconstruction: closed loop](reconstruction.md#moving-actors-a-neural-lidar-fisheyes-and-closed-loop-4d)): the
dynamic reconstruction's tracked road users, instead of replaying their logged motion, decide their accelerations and
lateral offsets every step against the ego the policy drives and against each other.

- **Frames.** The reconstruction's world is right-handed; level-k, like the simulator's ego state `[x, y, yaw°, v]`,
  works in CARLA's convention of that world (y mirrored, yaw clockwise). Every logged pose is taken into CARLA's
  convention once (x, −y, −yaw), level-k runs there with the ego's state as it is, and each pose is mirrored back for
  rendering: the box centre (x, −y, z) and the rotation Rz(yaw) times the logged tilt (roll and pitch, Rz(−ψ_logged)
  R_logged), z and the tilt from the logged annotation nearest along the track's path — the reconstruction has no road
  surface. At the episode's start every actor is drawn exactly where its logged box is (tested to 10⁻⁶).
- **Who is driven.** Every dynamic track of a road-user kind (car — a bus or truck by its class —, autorickshaw,
  two-wheeler, pedestrian); static tracks and rigid objects replay their logged poses and are obstacles the others
  see. The ego is an uncontrolled agent on the logged ego path, predicted at level 0 by everyone with the episode's
  target speed. An actor joins at the first instant its logged track is shown at or after the episode's start, from
  its logged pose and speed there, and is driven to the episode's end.
- **Paths and parameters.** An actor's reference path is its own logged path (extrapolated straight past either end;
  a track that barely moved: a straight line along its logged heading), lateral targets offsets from it, unbounded —
  there is no map of the carriageway. Parameters per kind from `--levelk-params` (a `behaviour.levelk` block inline,
  a traffic profile, or a calibration file); the desired speed, unless set, is the track's logged median speed while
  it moved. The right of way (it matters to level-0 gap acceptance only): the ego, and tracks travelling along the
  ego's road (heading within 30° of it, either way, within 8 m of the logged ego path where they join).
- **Appearance.** An actor keeps its logged appearance: its Gaussians are drawn at the level-k box, and its wheels and
  a pedestrian's limbs move as they did at that time in the log (they follow time, not the new motion). An actor driven
  off its logged path meets lighting and backgrounds it was never seen against, and its far side may never have been
  seen.
- **Records.** Decisions at a step use every agent's state at the previous tick, the ego's included (simultaneous
  moves); `--levelk-mode sample` (the default) draws the quantal response from a generator seeded by the episode's
  seed, so the same seed gives the same motion. Each episode's folder gains `levelk.json` (the agents, their
  trajectories in CARLA's convention, `core.levelk.realism` of the level-k agents' motion and `core.levelk.inching`
  for each two-wheeler and autorickshaw); `run.json` says `actors.model: levelk`, and parameters from a dashcam
  calibration mark it `source: dashcam` (the episode is then private). The episode stays a normal run folder: the
  traffic rules accept it.

### Measured: inching against the policy's ego (`tests/test_recon_levelk.py`)

Tiny synthetic reconstructions (a street of Gaussians, a forward camera, written directly — no training): the ego's
logged drive at 9 m/s and a two-wheeler that waited on a side road crossing the street 30 m ahead (level 1, α 0.6;
in the first, also a car well past the crossing, level 0); the ego driven by IDM policies on the observation's lead
features, NeuralSim at 10 Hz, five seeds each, 1–6 s an episode on the CPU.

| Ego 22 m (2.4 s) from the crossing | Rider | When the rider is across the ego's lane, the ego is |
|---|---|---|
| gives way (slows to a crawl for the side road) | creeps out (approach 1.4–2.0 m/s) and crosses in front | 5.5–6.9 m short of the crossing, crawling at 0.7–0.9 m/s |
| keeps 9 m/s (brakes only for what enters its path) | waits, crosses behind | 14–22 m past it |

| Ego 40 m (4.4 s) from the crossing | Rider enters its path | Ego's time to reach it then | Ego |
|---|---|---|---|
| keeps 9 m/s, brakes for what enters its path | at 2.2–2.5 s, creeping (1.8–2.1 m/s) | 2.0–2.25 s — under its own 3 s gap acceptance | brakes 3.7–4.5 m/s² |
| keeps 9 m/s, brakes for nothing | at the same instant | 1.5–2.0 s | passes behind it at 9 m/s |

Every seed gave the same outcome (the tables give the ranges); no contact and no level-k motion outside the realism
limits in any episode. The second table is the point of level-k against a real policy: the rider inches into a gap
shorter than its own gap acceptance because it predicts that the ego, a level-0 driver to it, will yield — and its
choice cannot depend on whether this ego actually will, only on the ego's state. Against an ego that gives way early it
goes; against one already too close and fast to stop for it, it waits.

## Calibrated on your near-misses (private)

`falcon levelk calibrate --from-nearmiss INGEST_DIR [--kind two-wheeler]…` (`core/levelk_nearmiss.py`) calibrates α, λ,
the desired speed and the gap acceptance per kind on the road users of your own dashcam near-misses
([near-misses](nearmiss.md)): every mined window's smoothed (N, 8) tracks and the ego from the GPS, from
`falcon-ingest/clips/*/nearmiss.json` (all mined windows, not only the top 20).

- **Decisions.** Every road user of the asked kinds (`car`, `bus` — trucks too —, `two_wheeler` — motorcycles and
  bicycles —, `autorickshaw` — D's heuristic class included —, `pedestrian`; cattle and unknown kinds are left out),
  every 0.25 s of its track, the first and last 0.5 s of a track left out (the smoother's ends). Its reference path is
  a quadratic fitted to its path over the window (its lane's direction, not its swerves), and the likelihood is that of
  its **acceleration given the lateral course it took**: every acceleration option is rolled out along the reference
  path with the lateral offset it actually had at each step's time — lateral choices cannot be identified from a track
  without a map. The others (the ego and every road user seen then) over the 3 s horizon as observed, constant
  velocity past the end of their track; a leader's hard stop as the solver adds it.
- **The observed plan is a set.** The options whose execution through the jerk limit reproduces the observed change
  of speed over the next 0.25 s to within 0.1 m/s of the best: the jerk limit makes −2 and −4 m/s² from rest identical
  over a re-plan interval, a standstill makes every braking plan identical, and a smoothed track blurs accelerations.
  The likelihood is the probability of that set (`core.trafficgraph`'s `choices`), with a 2 % lapse — any plan alike —
  so one decision the model cannot explain (an observed plan inside its collision margin, as monocular boxes produce)
  does not decide the estimate (`calibrate(..., lapse=)`; without it every range × 0.9 drove λ to its floor).
- **Intervals.** Wald 95 % intervals — optimistic: decisions of one road user treated as independent, the others'
  motion taken as observed (inverse planning). **The depth scale's uncertainty is propagated**: the near-miss
  bootstrap's factor (every range × U[0.9, 1.1], one per window) redrawn `--depth-draws` times (20), the tracks rebuilt
  — exactly, by `core.nearmiss.actor_traj` on the clip's perception rows, or approximately without them (offsets from
  the ego scaled) — and the calibration repeated from the estimate; the spread is added in quadrature (the combined
  interval, within the parameter's bounds). `--depth-draws 0` states it instead. Level k is not estimated (k 1).
- **Output.** `falcon-ingest/levelk/calibration.json` (or `--out`): per kind the estimates, standard errors, the Wald,
  depth-scale and combined intervals, decisions, road users, the assumptions; and a traffic profile's
  `behaviour: {"model": "levelk", "levelk": {kind: {k, lambda, alpha, v_des, t_gap}}, "levelk_source": {"format":
  "dashcam"}}` — Worlds ▸ traffic profile ▸ *Load calibrated parameters…* applies it as one undoable edit, and
  `falcon recon closedloop --levelk-params FILE` reads it.
- **Private.** Derived from your footage: written inside the ingest's private folder by default, marked (`run.json`
  worker `dashcam-levelk-calibrate`, source dashcam) wherever it is written; a profile it is loaded into is marked
  (`behaviour.levelk_source`), so `assistant.privacy` keeps the calibration, the profile's level-k numbers
  (`project_get` withholds them) and every drive and run with that profile from the assistant's model, and cluster
  job and cloud study bundles refuse them (tested).

### Measured: planted parameters (`tests/test_levelk_nearmiss.py`)

Twelve 8 s level-k episodes at 20 Hz — a two-wheeler with planted α 0.6, λ 6, v_des 9 m/s, t_gap 2 s behind a slower
car, your car ahead in the next lane, everyone keeping their lanes — written as an ingest's near-miss windows with
level-k's own (N, 8) tracks; 336 decisions, 20 depth draws (approximate scaling), 19 s on the CPU:

| | true | estimate | Wald 95 % | depth-scale sd | combined 95 % |
|---|---|---|---|---|---|
| α | 0.6 | 0.640 | [0.45, 0.83] | 0.119 | [0.34, 0.94] |
| λ | 6 | 6.18 | [3.6, 8.7] | 1.48 | [2.3, 10.0] |
| v_des | 9 m/s | 9.09 | [8.26, 9.91] | 1.13 | [6.7, 11.4] |
| t_gap | 2 s | 2.34 | [0.12, 4.56] | 0.61 | [0.2, 4.9] |

The depth scale dominates the desired speed's uncertainty (every speed scales with it). The same episodes **through
core.nearmiss's smoother** (perception rows of the boxes' near faces at 5 cm range noise, then `actor_traj`, as an
ingest makes them): α 0.48 (Wald [0.23, 0.73]), λ 9.2 (Wald [6.7, 11.8], combined [5.1, 13.4]), v_des 7.65 m/s (Wald
[7.27, 8.04], combined [6.6, 8.7]: the true 9 m/s outside both) and t_gap on its lower bound (0.2 s: not identified, no
interval) — the smoother blurs the step changes of acceleration that level-k plans make, and the calibration reads a
blurred acceleration as a gentler plan. Real tracks are noisier: treat calibrated values as an indication, compare kinds
and places with each other rather than reading them absolutely, and check them in a simulation before relying on them.
Not measured on real footage (the user's clips are never read during the build).

## Gradient adversaries, validated in CARLA

KING (Hanselmann et al. 2022) optimises adversarial agents through a differentiable kinematic bicycle model. Here:
the planner under test is **cloned** into a small differentiable surrogate (an MLP from the ego's speed, the gap to
the vehicle ahead and the closing speed to its acceleration, fitted on the ego's own recorded episodes — so any ego,
a black box included); the lead_brake family (speed, gap, deceleration, brake time; the lead's braking a smooth step)
is rolled out differentiably with the surrogate driving the ego; projected Adam steps lower a soft minimum of the
gap over time, within the scenario's ranges and the realism bound on the lead's deceleration (6 m/s²,
`core.advfeatures.BRAKE_MAX`).

**The CARLA loop itself is not differentiable.** Gradients flow only through the surrogate and the kinematic model,
so every candidate is re-run in CARLA (`validate_in_carla`: the adversary environment with the scripted scenario
and the real ego) and the measured minimum gap, TTC and failures are reported beside the surrogate's prediction
(`attack()`; `prediction_error_m`). `bicycle_rollout` is the differentiable vehicle model (checked with
`torch.autograd.gradcheck`).

## Measured

### Inching emerges (numpy, `tests/test_levelk.py`)

The crossing scene: a stream of four level-0 cars with the right of way at 9 m/s, 22 m apart (under 2.5 s: no gap
a level-0 rider with `t_gap` 3 s accepts), and a two-wheeler stopped on a side road that crosses the stream 30 m ahead
of the first car. Ten seconds, mode `best`:

| Rider | Enters the stream | Car it enters in front of: time to reach it | Rider's approach speed | That car's braking |
|---|---|---|---|---|
| level 0 (any α) | never | — | — | — |
| level 1, α 0.3 | never | — | — | — |
| level 1, α 0.5 / 0.6 / 0.8 | at 3.05 s | car1, 1.29 s | 1.76 m/s (enters at 4.0 m/s) | 4.0 m/s² (to 6.7 m/s) |
| level 2, α 0.5 / 0.6 / 0.8 | at 3.05 s | car1, 1.29 s | 1.76 m/s | 4.0 m/s² |

No collision in any run; every agent within the realism limits. The rider creeps forward from a standstill into a
gap of 1.3 s — less than half its own gap acceptance — because it predicts that the level-0 driver will yield, and
the driver does, at the realism bound. At level 0 it waits for a gap that never comes; a timid level-1 rider (α 0.3)
waits too: assertiveness is what makes it inch. Sampled quantal responses (λ 8) inch as well, the same for the
same seed.

### Calibration recovers planted parameters

300 synthetic decisions of a car (random speeds, 0–2 others at random positions and speeds, choices sampled from the
quantal response of θ*): α 0.629 ± 0.040 (true 0.6), λ 6.20 ± 0.54 (6.0), v_des 9.33 ± 0.15 m/s (9.0), t_gap 3.7 s
with a Wald interval [1.6, 5.8] (2.5: weakly identified, but covered); 7 s on the CPU.

### The gradient search

On a surrogate cloned from an IDM planner (MAE 0.12 m/s² against a 3.2 m/s² constant baseline), 40 projected Adam
steps from the lead_brake ranges' centre lowered the predicted minimum gap from 2.66 m to 2.45 m within the bounds.
The CARLA validation of real egos is an acceptance step (real server).

### In CARLA (the fake)

Level-k adversaries (a cut-in car and a filtering two-wheeler, follower ego, six seeds, 15 s): all twelve adversary
episodes realistic (no fault collisions, no contacts, no rule budget exceeded, dynamics within limits); the
two-wheeler closed on the ego in two of six (minimum TTC 6.2 and 8.3 s), and no episode ended in an ego failure —
these level-k adversaries press without breaking the realism rules, and on the fake's straight road the lane follower
was not provoked into a failure. Getting there took four fixes, each measured on these episodes: real boxes
(CARLA's Tesla is wider than the type's 1.9 m: two of six episodes ended in a rear-end the bike caused), plans
rolled out through the jerk limit as they are executed, the margin that grows with the prediction's time and room
for the leader's hard stop (three of six had collisions before them), and a desired speed within the road's limit
(else the speed-limit rule was charged). A real-server measurement (Town10HD, central Bengaluru) is an acceptance step.

## Limits

Agents follow fixed reference paths (lane changes are lateral offsets, not route choices); corridors use each
agent's local heading (approximate on sharp bends); conflicts are where two paths cross at ≥ 30°, with the right of
way a flag (no junction rules: background level-k traffic has no junction priority); plans are constant over the
horizon; only level-0 others are predicted reactively; pedestrians walk their paths. Calibration assumes observed
others and independent decisions; the gradient search covers the lead_brake family and a longitudinal surrogate.
The weights (`WEIGHTS`) are heuristic, not fitted. In the neural closed loop: actors keep their logged appearance
(wheels and limbs follow time), their reference paths are their logged paths with no map of the carriageway (lateral
targets unbounded), and they stay to the episode's end once joined. Calibrated on near-misses: monocular tracks
through a smoother (biased, measured above), the acceleration conditioned on the lateral course taken (lateral choices
not identified), level k not estimated, and near-miss windows over-represent hard interactions — the estimates
describe road users near your car in the mined moments, not traffic at large.
