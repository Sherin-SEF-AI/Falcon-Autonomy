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
`workers/adversary.py`. Each module's docstring is the reference; this page is the guide.

```sh
falcon levelk simulate [--scene crossing|merge|filtering|pedestrian|platoon] [--level ID=K]... [--param ID.KEY=V]...
                       [--seconds S] [--seed N] [--mode sample|best] [--lht] --out DIR
falcon levelk check [--scene NAME]... [--seeds N] [--seconds S] [--out DIR]
falcon levelk calibrate --log FILE... [--agent ID]... [--steps N] --out DIR
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
| A neural simulator (4.0 wave 2) | `Controller.step(t, agents) → actions` per tick (below) | `core/levelk.py` |

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
The weights (`WEIGHTS`) are heuristic, not fitted.
