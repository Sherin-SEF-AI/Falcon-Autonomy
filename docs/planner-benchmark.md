# Planner benchmark

Falcon's [leaderboard](training.md) scores a policy as the CARLA Leaderboard does: route completion times infraction
penalties. The planner benchmark scores **any planner Falcon can drive** with the closed-loop terms nuPlan and NAVSIM
use — a multiplicative safety gate times a weighted average of progress, time to collision and comfort — over a fixed
set of routes, with bootstrap intervals and paired comparisons (`core/planbench.py`, `sim/benchrun.py`,
Scenarios ▸ *Planner benchmark*, `falcon bench`).

## Routes

A route is a world, a scenario and a pinned seed. The seed draws the scenario's parameters and the start (the
director's seeded spawn order), so every planner meets the same situation. Scenario actors move **kinematically** and
there is **no background traffic**: CARLA 0.9.16's vehicle physics is not bit-for-bit repeatable and the traffic
manager drives through it, while a kinematic scenario repeats exactly ([scenarios](scenarios.md#bit-for-bit-repetition--measured)).
An autopilot ego keeps CARLA's physics, so its runs are not exactly repeatable.

| set | routes |
|---|---|
| `falcon-pdm-v1` | 11 on central Bengaluru (an imported OpenStreetMap world: lead brake, autorickshaw cut-in (two seeds), two-wheeler filtering, jaywalker behind a bus, wrong-way scooter, bus pulling out, stray cattle, handcart in the kerb lane, stalled vehicle, pedestrian crossing), 9 on Town10HD_Opt (the generic built-ins, two seeds for lead brake and cut-in), 3 on Town04 |
| `falcon-pdm-mini` | 4 Town10HD_Opt routes for a quick check |

`falcon bench routes [--set S]` lists them. A route whose world the project lacks is skipped, saying which: the
Bengaluru routes need an imported world whose name or id says Bengaluru (Worlds ▸ Import a place), Town04 a world of
that town (Worlds ▸ Add a town).

## Planners

| planner | how it drives |
|---|---|
| `follower` | Falcon's lane follower (the scenario director's): it brakes for what is in its corridor. It sees the scenario's actors directly — a privileged planner, the **reference** for progress, as NAVSIM's PDM-Closed is |
| `none` | constant set speed, never braking (a lower bound) |
| `autopilot` | CARLA's traffic manager (physics) |
| `policy:DIR` | a trained policy ([policies](policies.md): behaviour cloning, PPO, a CNN or end-to-end policy) in the environment, one episode per route with the route's seed |
| `stack:SUITE.json` | the stack of a [stack suite](stack-testing.md) — its driver, bridge and launch commands; a ROS 2 stack through the bridge — one stack run per route |

Drive planners record with the rig reduced to one 64 × 36 camera without labels (they read the director's state, not
the sensors); a stack keeps the whole rig, since it perceives. Every run writes the map's driving lanes (`lanes.json`)
for the drivable-area and direction terms. A policy's episode carries every box over the episode
(`episodes.jsonl` `trajectories`, from Falcon 2.5).

## The terms

Each term is a pure function of what the run recorded — the ego's and every scenario actor's box (t, x, y, yaw,
vx, vy, length, width) in CARLA's frame, the collision sensor's events, the map's lanes — and is unit-tested on
synthetic trajectories (`tests/test_planbench.py`).

| term | Falcon's definition | as in |
|---|---|---|
| NC, no at-fault collisions | 1; 0.5 when the only at-fault collisions are with static objects; 0 otherwise. Each actor counts once (its first contact). At fault: the other stopped (< 0.05 m/s) while the ego moves; a contact at the ego's front; a lateral contact while the ego's body crosses a lane line or leaves the lanes. Not at fault: the ego stopped; a contact at its rear | nuPlan `no_ego_at_fault_collisions` and its collision classes (stopped ego, stopped track, active front, rear, lateral); NAVSIM's NC |
| DAC, drivable-area compliance | 0 when a corner of the ego's box is more than 0.3 m outside every driving lane at any sample, else 1 | nuPlan `drivable_area_compliance` (maximum violation 0.3 m); NAVSIM's DAC |
| DDC, driving-direction compliance | the distance driven where the ego's centre is on lanes of the other direction only, summed over any 1 s window: ≤ 2 m → 1, ≤ 6 m → 0.5, more → 0 | nuPlan `driving_direction_compliance` (2 m and 6 m over 1 s) |
| TTC, time to collision within bound | 0 when, at a sample with the ego moving (> 0.005 m/s), the time to collision with an actor not behind the ego and not already in contact is below 0.95 s (horizon 1 s); else 1 | nuPlan `time_to_collision_within_bound` (0.95 s, 1 s horizon); NAVSIM's TTC |
| EP, ego progress | the ego's progress along the route (the reference run's path; a sample more than 4 m from it does not count) over the reference's, clipped to [0, 1]; 1 when the reference progressed less than 5 m | nuPlan `ego_progress_along_expert_route` (against the expert log); NAVSIM's EP (against PDM-Closed) |
| C, comfort | 1 when longitudinal acceleration ∈ [−4.05, 2.40] m/s², \|lateral acceleration\| ≤ 4.89 m/s², \|yaw rate\| ≤ 0.95 rad/s, \|yaw acceleration\| ≤ 1.93 rad/s², \|longitudinal jerk\| ≤ 4.13 m/s³ and \|jerk\| ≤ 8.37 m/s³ at every sample; else 0 | nuPlan `ego_is_comfortable`; NAVSIM's C |

The aggregate:

    PDMS            = NC × DAC × DDC × (5·TTC + 5·EP + 2·C) / 12        Falcon's score
    PDMS_navsim_v1  = NC × DAC × (5·TTC + 5·EP + 2·C) / 12              NAVSIM v1's PDM score, reported beside it

Sources: Caesar et al., *nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles*, 2021
(arXiv:2106.11810), and the nuplan-devkit's closed-loop metrics and their documented thresholds; Dauner et al.,
*NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking*, NeurIPS 2024 Datasets and Benchmarks
(arXiv:2406.15349) for the PDM score's form — the multipliers NC and DAC, the weighted average of TTC (5), EP (5) and
C (2); Dauner et al., *Parting with Misconceptions about Learning-based Vehicle Motion Planning*, CoRL 2023
(arXiv:2306.07962) for PDM-Closed. The thresholds above are as those documents give them; they were not re-checked
against the devkit's source on this machine.

### Where Falcon's version differs

- **Closed loop in CARLA, not log replay.** nuPlan replays logged scenes with reactive or non-reactive agents; NAVSIM
  scores a 4 s plan non-reactively against a log. Here the planner drives the whole scenario (12–16 s) in CARLA and the
  other road users follow the scenario's script — they react to triggers, not to the planner.
- **Driving direction is a multiplier.** NAVSIM v1 leaves DDC out of its PDM score; nuPlan's closed-loop score has
  it as a multiplier (as does NAVSIM v2's extended score). Falcon multiplies by it and reports NAVSIM v1's form beside.
- **The weighted average has no speed-limit term.** nuPlan's closed-loop score also weights speed-limit compliance (4)
  and multiplies by "making progress"; Falcon follows NAVSIM's three terms. Speed limits are checked by the
  [traffic rules](traffic-rules.md).
- **The reference** is Falcon's lane follower on the same route (privileged, like PDM-Closed), not an expert log; when
  it did not drive a route the best progress any planner made there stands in, and the results say so.
- **At fault is approximated** from the contact point (the corners of either box inside the other) in the ego's frame:
  front beyond half the ego's half-length, rear behind it, lateral between. nuPlan classifies from the contact
  geometry and lane occupancy in a similar way but not identically. Without a map a lateral contact counts as at fault.
- **Drivable area is the driving lanes** (CARLA's `Driving` lanes, junctions included). nuPlan's drivable area also
  includes car parks and some shoulders; a parked car's lane on an imported world may be counted as off the area.
- **Static objects** are Falcon's proxies that do not drive (a handcart, a vendor's stall) and sensor contacts with
  something that is not a scenario actor; cattle count as road users (nuPlan has no animals).
- **TTC is box-exact** (the moving separating-axis test of [criticality](methods.md#criticality)) rather than sampled
  every 0.1 s over the horizon; an actor whose centre is behind the ego's is not counted.
- **Comfort** derivatives come from a local quadratic fit over ±0.35 s (a Savitzky–Golay filter of 15 samples at
  20 Hz) of the recorded velocity — the director's own, exact for kinematic motion. nuPlan filters its own way. The
  lane follower in kinematic motion brakes in steps, so its comfort often fails on jerk (measured on the fake: 4.7 and
  6.1 m/s³ on two routes): that is the controller, not the score.
- **No background traffic**: the routes test a planner against the scenario's actors only.

## Results

A planner's score on a set is the mean over its routes of every term and of PDMS, each with a **95 % percentile
bootstrap interval over routes** (`core.metrics.bootstrap`; routes are the independent units). Two planners are
compared on the routes both drove with a **paired bootstrap** of B − A (`core.metrics.paired_bootstrap`); the
comparison says the share of resamples where B was better. With few routes the intervals are too narrow
([methods](methods.md)). A term a run cannot evaluate (no map: DAC and DDC) is left out of that run's product or
average and the run is marked partial.

The benchmark's folder is `benchmarks/<name>/` in the workspace: `benchmark.json` (the set, the planners),
`runs/<planner>/<route>/` (or a policy's `episodes.jsonl`), `index.jsonl` (each finished route), `scores.json` (every
term of every run, with the collisions, the worst comfort values, the closest TTC) and `results.json` (per planner,
per route and aggregate; the comparisons). Re-running resumes: a route already driven is not driven again, and a new
planner joins the comparison.

    falcon bench routes [--set falcon-pdm-v1|falcon-pdm-mini] [--json]
    falcon bench run PROJECT --name N --planner P [--planner P]… [--set S] [--route ID]… [--rig R] [--no-reference]
    falcon bench score RUN… [--reference RUN] [--json]          any scenario drive or stack run, scored alone
    falcon bench results PROJECT --name N | BENCH_DIR [--boot N] [--reference P] [--json]

The assistant queues it with `research_bench_run` (planners, route set, name; a new planner joins an existing
benchmark's comparison) and reads it with `research_bench_results` ([assistant tools](assistant-tools.md)).

## Measured, and what needs a real server

On the fake CARLA (`tests/test_planbench.py`): the lane follower and the constant-speed ego on two Town10HD_Opt
routes — the follower NC 1, TTC 1, DAC 1 and comfort 0 (jerk), PDMS 0.83; the constant-speed ego hits the braking
lead and the stalled van (NC 0, TTC 0), PDMS 0. Not yet measured on a real CARLA 0.9.16 server: the full set's
numbers, a policy's and a ROS 2 stack's, the time a set takes, and whether every Bengaluru route finds a start on
the imported world (a scenario the map cannot host is recorded as a failed run with the director's reason).
