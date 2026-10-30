# Adversarial traffic: learned road users that make the ego fail, realistically

Scripted scenarios ([scenarios](scenarios.md)) fire triggers someone wrote; a rare-case search
([rare cases](rare-cases-mil.md)) tunes their parameters. Here the other road users **learn**: several adversaries,
trained with multi-agent PPO, try to provoke the ego into a failure — a cut-in that leaves it no room, a lead car that
brakes when it is close, a two-wheeler filtering between the lanes, a scooter riding the wrong way near the kerb. The
ego under test stays fixed: a Falcon driving policy, the lane follower, or CARLA's autopilot.

Learned adversaries find "failures" that mean nothing if they simply ram the ego, leave the road or ride the wrong way
all the time. So the definitions here are mostly about **realism**: who caused a collision, what dynamics and lane
discipline an adversary must keep, which traffic rules its role may bend, and how the failures it finds are counted.

Code: `core/advfeatures.py` (the shared definitions: roles, observations, actions, fault attribution, realism,
rewards), `workers/adversary.py` (the environment worker), `core/adversary.py` (the client, the multi-agent PPO, the
evaluation, the replay, the CLI), `ui/adversary_panel.py` (the Rare cases page's section). Each module's docstring is
the reference; this page is the guide.

```sh
falcon adversary train PROJECT --out DIR [--ego follower|none|autopilot|POLICY] [--action lateral|discrete|continuous]
                       [--curriculum default|FILE | --roster ROLE,ROLE] [--updates N --steps N] [--resume]
                       [--seed S] [--option KEY=VALUE] [--reward TERM=W]
falcon adversary eval PROJECT --policy DIR --out DIR [--ego SPEC]... [--episodes N]
                       [--baselines random,idle,scripted_random,scripted_adaptive] [--tm-reference N] [--roster ...]
falcon adversary replay PROJECT --episodes EPISODES.jsonl --episode N [--failure K] --out DIR
```

All three also take the environment options of `falcon policy` (`--world`, `--rig`, `--fps`, `--motion`, `--seed`,
`--keep-server`, `--fake-carla`) and `--timeout`.

## The multi-agent environment

It is the learning environment of [policies](policies.md) with controlled traffic: the gym worker is subclassed
(`workers/adversary.py`), so placement, kinematic motion, the ego's observation, its infractions and termination are
the gym environment's, unchanged. The single-agent API (`FalconDriveEnv`) is unchanged too; `AdversaryEnv` subclasses
it and talks to its own worker.

**The API** is PettingZoo's parallel one (Terry et al. 2021):

```python
from falcon_autonomy.core.adversary import AdversaryEnv, adversary_config
cfg = adversary_config(project, 'world-town10', out_dir, svc.port, svc.tm_port, ego='follower',
                       roles=['cut_in_car', 'filtering_bike'])
with AdversaryEnv(cfg, svc) as env:
    obs, info = env.reset(seed=3)                       # {agent: observation vector}
    obs, rewards, terminations, truncations, infos = env.step({a: [0.0, 0.0] for a in env.agents})
```

`env.agents` are this episode's learned adversaries in order. Every agent's episode ends with the episode, so
terminations and truncations are equal across agents. The episode's shared information is `env.last_info`, and at the
end its summary is `env.last_info['episode_summary']`.

**The ego is not an agent.** It is the system under test, fixed for a run:

| ego | driven by | repeatable |
|---|---|---|
| `follower` | the director's lane follower: pure pursuit, a 1.8 s time gap behind what is in its corridor | yes |
| `none` | the director at constant speed, never braking (a blind ego: the sanity baseline) | yes |
| `autopilot` | CARLA's traffic manager, on CARLA's physics | **no** |
| a policy folder | a Falcon policy (behaviour cloning, PPO, world-model or end-to-end) run by the client on the ego's own observation (`env.ego_obs`), its action sent with the adversaries' | yes |

**An episode.** The roster (a list of roles) becomes a concrete scenario, which the director places like any other.
The adversaries are its actors, marked external, so the director does not drive them. Each step:

1. The director drives the ego and any scripted actor.
2. Each adversary's action goes through its actuator (below).
3. One tick.
4. The facts: the ego's observation and infractions, as the gym worker computes them. For every tracked road user,
   its pose, CARLA's lane under its centre, its dynamics, its contacts, its time to collision and box gap with the
   ego, and its traffic-rule trace row. At an ego collision, the fault attribution.
5. The rewards.

The episode **terminates** as the ego's does (a collision, off the road, a route deviation, blocked, the route
complete), or when an adversary goes more than 2 m beyond the driving lanes. It is **truncated** at the time limit
(15 s by default), or when an adversary reaches the end of its lane path.

**Scripted episodes** (`reset(options={'scenario': 'cut_in', 'values': {...}})`) run a built-in scenario's actors
through the director's timeline. They are judged by exactly the same failure and realism measures, which is how the
baselines below compare like with like.

**Determinism.** With kinematic motion, no background traffic and an ego that is not the autopilot, an episode is a
function of its seed, its options and the actions (the adversaries', and the ego's when a policy drives it). Measured
below, on the fake and on a real server. That is what makes every failure replayable.

### Roles

| role | kind | start | speed | its role allows |
|---|---|---|---|---|
| `cut_in_car` | car | the next lane, 15 m behind to 5 m ahead of the ego | ego + 0–10 km/h | following closely |
| `lead_car` | car | the ego's lane, 15–40 m ahead | ego ± 5 km/h | — |
| `filtering_bike` | motorcycle | on the lane line, 10–30 m behind | ego + 5–20 km/h | riding the line, keeping to either side, overtaking on either side, following closely |
| `wrong_way_scooter` | motorcycle | the ego's lane against the traffic, 40–100 m ahead, 0.5–1.4 m towards the kerb | 15–35 km/h | riding the wrong way (unlimited), keeping to either side, straddling |
| `autorickshaw_cut_in` | autorickshaw (stand-in) | the next lane, 0–8 m ahead | ego + 3–12 km/h | following closely, straddling |

The ranges are heuristic, near the built-in Indian scenarios. The ego starts at 15–25 km/h, and no adversary starts
faster than 30 km/h: on Town10HD, `Vehicle.get_speed_limit()` reported 30 km/h for every adversary (measured below).
Going faster is the adversary's own choice, and the speed-limit rule charges it. Starts that would overlap another road
user are redrawn (`advfeatures._overlaps`).

The "allows" column says where Bengaluru-style behaviour is realistic. A filtering two-wheeler on the lane line, or a
scooter against the traffic near the kerb, is everyday traffic, not an unrealistic adversary.

### Observations

`ADV_FEATURES` is one float32 vector per adversary, 31 numbers, in its own frame:

- **Its motion:** speed and longitudinal acceleration.
- **Its lane, from CARLA's map:** lateral offset from the lane's centre, heading error, lane width, on a driving lane,
  wrong way, and same-direction lanes on each side.
- **The ego relative to it:** position, heading, speed, relative velocity, the ego's acceleration, the box-to-box gap,
  and the time to collision of the two boxes at constant velocity (`core.criticality.ttc`, capped at 10 s).
- **Both on the ego's route:** its arc length ahead of the ego's, its lateral offset, and the ego's own offset.
- **Context:** the nearest other road user, the share of the episode elapsed, its previous action, whether its role
  may ride the wrong way, and its kind (car, motorcycle, autorickshaw).

An absent quantity takes its cap and a 0 presence flag, never NaN.

### Actions and the actuator

| kind | action | how it moves |
|---|---|---|
| `lateral` (default) | `[lateral, accel]` ∈ [−1, 1]²: a lateral velocity command (× 1.5 m/s) and an acceleration command (× its kind's top acceleration forward, × 6 m/s² braking) | through the actuator, then followed along its lane path by the director's pure pursuit |
| `discrete` | one of 9 manoeuvres: lateral −/0/+ 1 m/s × brake −4 m/s² / hold / accelerate +2 m/s² | the same |
| `continuous` | `[steer, accel]` with the ego's own semantics | straight to the kinematic bicycle model; the throttle is scaled to its kind's top acceleration |

The actuator bounds the rates:

- The acceleration moves towards its command at no more than 8 m/s³.
- The lateral velocity changes at no more than 2 m/s².
- The lateral offset from its own lane path stays within ±1.6 lane widths, and the lateral velocity slows in time to
  stop there.

So the `lateral` and `discrete` kinds are smooth by construction: jerk and lateral acceleration are bounded. The
`continuous` kind is not, and the realism penalties have to keep it in check. The motion is kinematic, the director's
bicycle model, as in KING (Hanselmann et al. 2022). Top accelerations are 3 m/s² for a car or a motorcycle and
2 m/s² for the autorickshaw (core.assets). Top speeds are 80 km/h for a car and 70 km/h for a motorcycle (heuristic urban caps), and 65 km/h for the
autorickshaw (core.assets).

## Fault attribution: who hit whom

At the onset of a box contact between two vehicles A and B (the ego and an adversary, or two road users), the first
rule that applies decides. It uses their last 2.6 s of states: pose, speed, the heading of the lane each is on, and the
wrong-way and on-lane flags.

| rule | condition | at fault |
|---|---|---|
| F1 wrong way / off road | exactly one of them is driving the wrong way, or is off the driving lanes, at the contact | that one |
| F2 lateral encroachment | each one's mean lateral velocity in its own lane's frame (speed · sin(yaw − lane yaw)) towards the other over the last 0.5 s; the larger is ≥ 0.3 m/s and exceeds the other's by ≥ 0.3 m/s | the one moving across: a cut-in, a swerve, a side-swipe |
| F3 rear-end | one's front zone meets the other's rear zone. Zones come from the closest points of the two boxes the step before the contact: the side of the box, normalised by its half extents, that the point lies beyond most | the striker; **unless** the struck vehicle was laterally clear of the striker's path at some sample within the last 2 s — then it cut in at an unsafe distance, and it is at fault |
| F4 | none of the above | shared |

F3 follows the longitudinal rule of Responsibility-Sensitive Safety (Shalev-Shwartz, Shammah & Shashua 2017): the
follower keeps a safe distance, and a vehicle cutting in must leave one. RSS's safe-distance formula itself is not
evaluated here (the thresholds are heuristic); the rare-case search's labels evaluate it in full (`core/rss.py`,
[scenarios](scenarios.md#avoidable-failures-responsibility-sensitive-safety)).

A **collision caused by the ego** is one where the ego is at fault. It is unit-tested on constructed histories: a
rear-end, an unsafe cut-in, a swerve, a wrong-way head-on and a side-by-side contact (`tests/test_adversary.py`).

## Realism

An adversary is **realistic** in an episode when all of these hold:

1. **It caused no collision** with the ego (F1–F3), and touched no other road user.
2. **It stayed on the lanes:** its centre was always on a driving lane.
3. **It kept to its direction:** wrong-way driving no longer than its role's budget (0 s, except the wrong-way scooter).
4. **Its dynamics stayed bounded:** |longitudinal acceleration| ≤ 4 m/s², |lateral acceleration| (speed × yaw rate)
   ≤ 4 m/s² and |jerk| ≤ 8 m/s³, except for 0.5 s per episode. These bounds are heuristic.
5. **It kept the traffic rules,** checked by `core.rules` on its own trace: the Indian set on a left-hand-traffic world,
   the generic set on CARLA's right-hand towns, or a configured set. It may break the rules its role does not allow at
   most once per episode (a heuristic budget). Rules its role allows are counted but not charged.

Its rule trace is the gym trace's format: pose, CARLA's lane and markings under it, its lead (the nearest road user
ahead in its corridor) and CARLA's speed limit. A gym trace has no passing events, so the overtaking rules are not
evaluable for adversaries. They are listed as such.

A **realistic failure** is an ego failure in an episode where every adversary was realistic. Only realistic failures
count towards failure rates, train the curriculum, earn the diversity bonus and become cases. Every failure is
recorded, with `realistic` false when it is not.

**Ego failures** are:

- a collision the ego caused;
- going off the road, the wrong way, or into a route deviation;
- a collision with a static object or a pedestrian.

They are not:

- `agent_blocked`: an adversary standing in front of the ego for 20 s is not a driving failure.
- `outside_route_lanes`: an evasive lane change is fine.
- `red_light` and `stop_sign`: an adversary cannot plausibly make the ego run a light. The lane follower ignores
  lights; in the first real-server run it ran a red light in an adversarial episode, and crediting that would reward
  the adversaries for the ego's own standing fault.

## Rewards

Per adversary, per step, each term logged (`info['terms']`, summed in the episode summary). The weights are
**heuristic, not tuned**; `--reward TERM=W` overrides them.

| term | per step | default weight |
|---|---|---|
| `ego_collision` | + w per new collision the ego caused with it | 10 |
| `ego_infraction` | + w per new ego failure infraction while it is within 25 m of the ego | 5 |
| `ttc` | + w · (s(TTC) − its best s so far)⁺, with s = exp(−TTC / 1.5 s). It pays only new records, so over an episode it sums to w · exp(−min TTC / 1.5 s) | 3 |
| `adv_fault` | − w per collision it caused; − w/2 per shared one | 10 |
| `adv_contact` | − w per new contact with another road user | 10 |
| `off_road` | − w_onset per onset, − w · dt while off the lanes | 5; 2 /s |
| `wrong_way` | − w · dt beyond its role's budget | 1 /s |
| `dynamics` | − w · dt per realism bound exceeded | 1 /s |
| `rules` | − w per rule violation beyond the budget, at the episode's end | 2 |
| `diversity` | + w / √N per realistic failure, N being the failures with the same descriptor so far in training; added by the trainer | 2 |

**Credit is split between the adversary and the team.** The ego-failure terms (`ego_collision`, `ego_infraction`,
`ttc`) are mixed: (1 − s) × its own + s × the team's mean, with s = 0.5 (`team_share`). One adversary can herd the
ego into another's trap, and both are credited. The penalties stay individual.

**The diversity descriptor** is (role, failure kind, the adversary's sector relative to the ego, the ego's speed in
10 km/h bins). This is count-based novelty: Strehl & Littman 2008; Bellemare et al. 2016.

## Training: parameter-shared independent PPO

`train_adversaries` (CLI `falcon adversary train`) is **independent PPO with parameter sharing**:

- One actor and one critic serve every adversary. Each adversary sees its own observation, and its kind is in it.
- Each adversary's experience is its own stream: GAE(λ) per stream, then one PPO update over all of them.
- `core.rl.gae` and `core.rl.ppo_update` are the ego's PPO, factored out for this. Its defaults, normalisers,
  clipping and seeding are reused.
- The critic is decentralised. MAPPO's centralised critic is **not implemented**.

For the method, see de Witt et al. 2020 for IPPO, Gupta, Egorov & Kochenderfer 2017 for parameter sharing, and Yu et
al. 2022 for MAPPO. Parameter sharing is what lets the curriculum grow the roster without new networks.

| setting | default |
|---|---|
| steps per update | 512 environment steps, which gives 512 × N adversary transitions |
| optimisation | 8 epochs, minibatch 128, Adam 3·10⁻⁴ |
| PPO | clip 0.2, γ 0.99, λ 0.95, entropy coefficient 0.005 |
| network | MLP 64-64 tanh |

Training runs on **the CPU**: CARLA owns the GPU during RL, and these networks are small. It is seeded: the same seed
gives the same updates (tested). It writes:

- `episodes.jsonl`, with every episode's roster, failures, fault, realism and returns;
- `updates.jsonl`;
- `checkpoints/`: the last three are kept, and `--resume` continues from `latest.json`, with the novelty counts and
  the curriculum carrying on;
- `policy/`: a `falcon.adversary_policy/1` folder, with the weights' SHA-256;
- `run.json`, with the provenance.

**The curriculum** has three stages: one adversary → two → three. The default rosters are:

| stage | rosters |
|---|---|
| 1 | `cut_in_car`; `lead_car`; `filtering_bike` |
| 2 | `cut_in_car` + `filtering_bike`; `lead_car` + `autorickshaw_cut_in` |
| 3 | `cut_in_car` + `filtering_bike` + `wrong_way_scooter`; `lead_car` + `autorickshaw_cut_in` + `filtering_bike` |

The next stage starts when realistic failures reach 30 % of the last 20 episodes, or after 200 episodes in the stage.
Without that cap, adversaries facing an ego they cannot beat would never meet the larger rosters. The thresholds are
heuristic.

## Evaluation

`falcon adversary eval` runs each ego with **the same episode budget per method**:

| method | what |
|---|---|
| `learned` | the trained adversaries, acting deterministically (the Gaussian mean, the argmax) |
| `random` | the same rosters, uniformly random actions (seeded) |
| `idle` | the same rosters holding their lane and speed: the placement alone |
| `scripted_random` | the built-in scenarios whose actors play the same roles (`cut_in`, `lead_brake`, `two_wheeler_filtering`, `wrong_way_scooter`, `autorickshaw_cut_in`; the budget split evenly), parameters from a Halton sequence (core.search) |
| `scripted_adaptive` | the same, parameters from the cross-entropy method for rare events (core.search `adaptive`), on the objective 1 for a realistic ego failure, else 0.9 · exp(−min TTC / 1.5 s) |

The first three methods use the same seeds, so the same placements where their rosters agree.

**The failure rate** is the share of episodes with a realistic ego failure, with Wilson's 95 % interval. The scripted
searches' episodes are adaptive, not independent draws, so their interval is descriptive. Also reported:

- the share with any ego failure, realistic or not;
- the distinct failure descriptors;
- the failure kinds;
- the median minimum TTC.

**Realism metrics**, per method:

- the adversaries' rule violations per km, by rule and in total, and those their roles do not allow;
- the collisions they caused, the shared ones and their contacts, per episode;
- their wrong-way and off-road time per km;
- the share of adversaries that were unrealistic;
- their dynamics: percentiles of |longitudinal acceleration|, |lateral acceleration| and |jerk|, and the share beyond
  the bounds.

**The traffic-manager reference** (`--tm-reference N`) is an episode set with N background vehicles driven by CARLA's
traffic manager on its physics, with no adversaries. Their dynamics are measured the same way. Each method's
dynamics are compared with them by the two-sample Kolmogorov–Smirnov statistic and the Wasserstein-1 distance.

The output is `adversary-eval.json` (with provenance), plus:

- `episodes.jsonl`: every episode with its actions and its environment settings;
- `failures.jsonl`: every failure, with its episode line;
- `cases.jsonl`: the realistic failures as cases.

## Replay and the Rare cases page

`falcon adversary replay` re-runs an evaluation episode from its own line:

1. It rebuilds the environment with the same ego, action kind, fps and time limit, plus a 640 × 360 chase camera. A
   camera changes nothing in the motion.
2. It resets with the episode's seed and options, and applies its logged adversary actions (and the ego's, when a
   policy drove).
3. It keeps the frames from 3 s before the failure to 1 s after.
4. It checks that the same failure (kind, step and adversary) happens. `match` in `replay.json` records the result.

Episodes with physics motion, background traffic or the autopilot ego are refused, with the reason: they do not repeat
in CARLA 0.9.16.

The **Rare cases** page has a section *Adversarial traffic (learned)*:

- choose the ego, the action kind and the number of updates, then *Train adversaries* (a job);
- choose the trained adversaries, then *Evaluate*;
- the realistic failures of every evaluation under the workspace's `adversary/` folder are listed as cases;
- *Replay the selected case* runs the replay as a job and reports whether it met the same failure.

## Related work, and what is implemented

- **Wachi 2019**, "Failure-Scenario Maker for Rule-Based Agent using Multi-agent Adversarial Reinforcement Learning
  and its Application to Autonomous Driving" (IJCAI). Non-player agents are trained with multi-agent RL (MADDPG) to
  make a rule-based player fail. A personal reward beside the adversarial one keeps the failures natural; pure
  adversarial training produced agents that deliberately collide with the player.
- **Ding et al. 2023**, "A Survey on Safety-Critical Driving Scenario Generation — A Methodological Perspective" (IEEE
  T-ITS 24(7)). A taxonomy of data-driven, adversarial and knowledge-based generation, with fidelity, efficiency,
  diversity, transferability and controllability as the open challenges.
- **Hanselmann et al. 2022**, "KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics
  Gradients" (ECCV). Adversarial agents' actions are optimised by gradients through a differentiable kinematic bicycle
  model in CARLA.
- **Also used:** Shalev-Shwartz et al. 2017 (RSS, for F3), de Witt et al. 2020 (IPPO), Gupta et al. 2017 (parameter
  sharing), Yu et al. 2022 (MAPPO, not implemented), Terry et al. 2021 (PettingZoo's API), Rubinstein 1997 and
  O'Kelly et al. 2018 (the cross-entropy method, core.search).

**Implemented:**

- Wachi's setting: several learned adversaries, a fixed player that may be rule-based or learned.
- Realism enforced by fault attribution, traffic rules, dynamics bounds, lane keeping and role budgets, in place of a
  personal goal.
- A count-based diversity bonus.
- KING's kinematic bicycle model as the motion, integrated by the director, not differentiated.

**Not implemented:**

- KING's gradient-based search;
- behaviour priors learned from real traffic data (such as STRIVE, Rempe et al. 2022), so the adversaries' style is
  shaped only by the realism constraints;
- MAPPO;
- adversarial pedestrians.

## Measured

All on the reference machine, 2026-10-29, while other jobs loaded it; wall times vary with that load. The raw
outputs (each evaluation's `adversary-eval.json`) were kept in the session's scratch folder, not in the repository.

### The fake carla

The fake carla is the tests' stand-in: a straight 300 m road with two lanes each way, right-hand traffic, and a
30 km/h speed limit. It is not CARLA.

**The environment** ran at 550 steps/s with three adversaries and their observations, on a lightly loaded machine.

**Training.** These are the adversaries evaluated below; `lateral` actions, default settings and curriculum, seed 0:

| trained against | updates | steps | episodes | wall | last stage | realistic failures (last 20 episodes) |
|---|---|---|---|---|---|---|
| `none` (constant speed) | 40 | 20,480 | 106 | 43 s | 3: promoted twice by success rate | 45 % |
| `follower` | 150 | 76,800 | 294 | 146 s | 2: promoted after 200 episodes | 0 % |
| a PPO ego (below) | 100 | 51,200 | 186 | 83 s | 1: never promoted | 0 % |

The PPO ego was trained with `falcon policy ppo`'s default curriculum for 120 updates (122,880 steps). It reached
the curriculum's last stage, pedestrian crossing, passing cut-in at 80 %.

**Evaluation.** 60 episodes per method and ego, the three-adversary rosters of the last stage, seed 1000. Failure
rate means realistic failures over episodes, with Wilson's 95 % interval.

Adversaries trained against `none`:

| ego | learned | random | idle | scripted, Halton | scripted, CEM |
|---|---|---|---|---|---|
| `none` | **30/60 = 0.50 [0.38, 0.62]** | 5/60 = 0.08 [0.04, 0.18] | 1/60 = 0.02 [0.00, 0.09] | 3/60 = 0.05 [0.02, 0.14] | 1/60 = 0.02 [0.00, 0.09] |
| `follower` | 0/60 = 0.00 [0.00, 0.06] | 1/60 = 0.02 [0.00, 0.09] | 0/60 | 0/60 | 0/60 |
| the PPO ego | 0/60 = 0.00 [0.00, 0.06] | 3/60 = 0.05 [0.02, 0.14] | 2/60 = 0.03 [0.01, 0.11] | 1/60 = 0.02 [0.00, 0.09] | 3/60 = 0.05 [0.02, 0.14] |

Adversaries trained against the `follower`, and against the PPO ego, compared with the same baselines (the baselines
do not depend on the policy):

| ego | trained against `follower` | trained against the PPO ego |
|---|---|---|
| `follower` | 0/60 = 0.00 [0.00, 0.06] | — |
| `none` | 0/60 = 0.00 [0.00, 0.06] | 0/60 = 0.00 [0.00, 0.06] |
| the PPO ego | 4/60 = 0.07 [0.03, 0.16] | 3/60 = 0.05 [0.02, 0.14] |

What this shows:

- **Against the blind ego, the learned adversaries found a realistic failure in half the episodes,** six times the
  random adversaries and ten to twenty-five times the searched scripted scenarios with the same budget. All 30 were
  the same mechanism: the lead car slowed to a stop at about 3 m/s² (within the bounds) and the ego, which never
  brakes, ran into it. That gave 2 distinct descriptors, the ego's speed bin being the only difference. The 30 were
  exactly the episodes whose roster had a lead car; the other roster (a cut-in car, a filtering bike, a wrong-way
  scooter) found none.
- **Against the lane follower, no method found a realistic failure,** except one random episode. The follower's 1.8 s
  time gap absorbs every braking within the 4 m/s² bound, and a cut-in close enough to beat it is the cutter's fault
  (F3).
- **Against the PPO ego, the learned adversaries were no better than random ones** (7 % and 5 %, intervals
  overlapping). The adversaries trained against the PPO ego itself never left the first stage. Adversaries trained
  against one ego did not transfer to the others: the ones trained against `none` found nothing against the PPO ego.
- **The scripted scenarios find failures mostly by unrealistic means.** Against `none`, 12 of 60 episodes in each
  scripted method had an ego failure, but only 1–3 were realistic. In the rest, the lead braked at up to 9 m/s², or
  the wrong-way scooter rode into the ego. 63–73 % of scripted actors were unrealistic, against 16 % of the learned
  adversaries trained against `none`.

**Realism of the learned adversaries** (trained against `none`, evaluated against `none`; 3.5 km driven):

| measure | value |
|---|---|
| collisions they caused, per episode | 0.32: all 19 were the wrong-way scooter's head-ons (F1) |
| contacts, per episode | 0.07 |
| unrealistic adversaries | 16 % |
| rule violations their roles do not allow, per km | 3.5 (random: 17.2; idle: 0.4) |
| all rule violations, per km | 31 (following distance, lane discipline and wrong way, which their roles allow) |
| wrong way beyond the budget | 0.03 s/km |
| \|longitudinal acceleration\| | p95 2.5 m/s², max 3.6 |
| \|lateral acceleration\| | p95 0.38 m/s², max 1.2 |
| \|jerk\| | p95 8.0 m/s³ (the actuator's limit), max 48 |

The jerk maxima come from the speed reaching 0 or the kind's top speed in mid-ramp, which the speed-derived jerk sees
as a step. On the fake, the traffic manager drives at constant speed (17,940 samples, all 0), so the comparison with
it is degenerate there.

**Tests:** `tests/test_adversary.py`, 19 tests (one of them the GUI section), about 2 minutes on the loaded machine.

### A real CARLA 0.9.16 server

Town10HD_Opt on Falcon's own server: port 2310, an 8 GB cap, started for these runs and stopped after them. The
machine was loaded by the full test suite running at the same time.

**The environment.**

- It was ready in 9.4–10.4 s. A reset took 0.04–0.27 s.
- It ran at 68–125 steps/s with three adversaries, and at 80–173 environment steps/s during training (median 131).
- `Vehicle.get_speed_limit()` reported 30 km/h for every adversary, at every step.

**Determinism.** Twice per run, a three-adversary episode was run with seed 5 and the same random actions. Over its
70 steps, every adversary observation, every reward and the failures (with their fault) were identical, in both runs.

**Training against `none`:** 30 updates (15,360 steps, 65 episodes) in 140 s. The curriculum promoted twice by success
rate (after episodes 43 and 63), and the last 20 episodes had 30 % realistic failures. A first, rehearsal run against
the follower credited the follower's own red-light runs as failures. That is why `red_light` and `stop_sign` are no
longer credited. Its numbers are not reported.

**Evaluation:** 20 episodes per method and ego, the three-adversary rosters, seed 2000.

| ego | adversaries | learned | random | idle | scripted, Halton | scripted, CEM |
|---|---|---|---|---|---|---|
| `none` | trained on Town10 | **8/20 = 0.40 [0.22, 0.61]** | 2/20 = 0.10 [0.03, 0.30] | 0/20 = 0.00 [0.00, 0.16] | 1/20 = 0.05 [0.01, 0.24] | 1/20 = 0.05 [0.01, 0.24] |
| `none` | trained on the fake | **8/20 = 0.40 [0.22, 0.61]** | (the same baselines) | | | |
| `follower` | either | 0/20 = 0.00 [0.00, 0.16] | 0/20 | 0/20 | 0/20 | 0/20 |

- **The picture on the fake carries over to Town10.** Against the blind ego, the learned adversaries found realistic
  failures about four times as often as random ones, and eight times as often as the searched scripted scenarios. The
  intervals of learned and random overlap only slightly at 20 episodes.
- **The adversaries trained on the fake's straight road transferred** to Town10's curved streets at the same rate.
- **Against the lane follower, no method found a realistic failure.**
- **Every learned failure was the lead car stopping in the lane** (descriptor `lead_car|collision|front|10` or `…|20`).
- **The scripted scenarios** had 4 ego failures in 20 episodes each, of which 1 was realistic.

**Realism**, adversaries trained on Town10 against `none`:

| method | unrealistic adversaries | rule violations not allowed, per km | caused collisions per episode | contacts per episode |
|---|---|---|---|---|
| learned | 30 % (the fake-trained ones: 17 %) | 15.6 | 0.45 | 0.10 |
| random | 48 % | 17.4 | 0.35 | 0.30 |
| idle | 32 % | 1.6 | 0.35 | 0.10 |
| scripted | 75 % | 7.2–7.5 | 0.10 | 0.00 |

**Dynamics against CARLA's traffic manager.** The reference was 20 background vehicles on CARLA's physics, 11,960
samples, from the speeds of the world snapshots:

| quantity | p50 | p95 | p99 | max |
|---|---|---|---|---|
| \|longitudinal acceleration\| | 0.31 | 4.97 | 10.8 | 27.6 |
| \|lateral acceleration\| | 0.01 | 2.12 | 5.48 | 7.03 |
| \|jerk\| | 5.2 | 74 | 162 | 531 |

The learned adversaries had |longitudinal acceleration| p95 2.0 m/s² and |lateral acceleration| p95 1.6 m/s². Their
two-sample KS statistic against the traffic manager was 0.27 for the longitudinal acceleration (idle: 0.72), 0.31 for
the lateral acceleration and 0.45 for the jerk.

**CARLA's own traffic exceeds Falcon's dynamics bounds.** 7 % of its steps were above 4 m/s², and 46 % were above
the 8 m/s³ jerk bound. Jerk taken twice from snapshot speeds at 20 Hz amplifies their noise, so that comparison is
dominated by the physics' noise, not by driving style. The scripted actors' lateral acceleration (p95 8.7 m/s²) comes
from the director's lane changes and its pure pursuit on Town10's curves.

**Replay.** A learned failure (the lead car stopping, seed 2005, step 87) was replayed with the chase camera. It met
the same failure at the same step, and kept 61 frames from 3 s before it. The rehearsal run's replay matched too.

## Limits

- **Realism is a set of constraints, not a model of human driving.** An adversary that meets every constraint can
  still behave in ways no driver would: the bounds are permissive enough for aggression, and its style is not learned
  from data.
- **Fault attribution is geometric and heuristic.** It has no right of way at junctions, no indicators, and no RSS
  safe-distance formula. A contact in a junction falls to F2 or F4.
- **Adversaries carry no collision sensor.** Their contacts with map geometry are not detected; off-lane detection
  stands in for it.
- **The overtaking rules are not evaluable on gym traces,** for adversaries as for the ego.
- **The motion is kinematic:** no tyre or collision dynamics, as in the gym environment ([policies](policies.md)).
  The autopilot ego and `--motion physics` are not repeatable.
- **The fake carla's straight 300 m road is not CARLA.** Its numbers show the method works, not how a town behaves.
  Its traffic manager drives at constant speed, so its dynamics reference is degenerate.
- **The scripted baselines search each scenario's own parameters.** Their failures depend on those scenarios' ranges,
  and their intervals are descriptive.
