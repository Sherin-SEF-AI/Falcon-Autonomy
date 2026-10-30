# Scenarios and rare cases

## A scenario
Actors placed around the ego (by lane and distance), a **timeline** of triggers and actions, and **parameters as
distributions** — `core/scenarios.py` documents the format. A seed draws every parameter (each from a generator seeded
by the seed and its own name, so adding a parameter changes no other draw); sweeps and searches set them explicitly.

Lanes are relative to the traffic's handedness: *adjacent* is the next lane in the ego's direction on the overtaking
side, *opposite* the nearest lane of the other direction, *kerb* the near edge of the road (left in left-hand traffic),
*sidewalk* beyond it, *line* the lane line a filtering two-wheeler rides along.

**Built-ins**: lead vehicle brakes, cut-in, pedestrian crossing, cyclist crossing, a parked car opens its door, stalled
vehicle, unprotected turn across traffic — and for Indian roads: two-wheeler filtering between lanes, a jaywalker from
behind a bus, a wrong-way scooter, a bus pulling out, stray cattle in the lane, a handcart in the kerb lane, an
autorickshaw cutting in, and the monsoon variant (a lead vehicle braking on a waterlogged road, with the scenario's own
weather). Copy one into the project to edit it.

**Indian kinds** of actor: `autorickshaw` and `cycle_rickshaw` are stand-in vehicles held to their class's top speed
and acceleration; `cattle`, `handcart` and `vendor` are proxy props of the real object's size, a moving one moved
kinematically. They keep CARLA's labels and carry Falcon's `subclass` ([Indian road assets](indian-assets.md)). A
scenario's optional `weather` replaces the world's for its runs.

## Editing the timeline
Scenarios ▸ Timeline edits a project scenario's timeline (a built-in is copied first). One row per event on a time
axis:

- a **time trigger** is a diamond at its time. A numeric one can be dragged, snapped to the chosen step (0.05, 0.1,
  0.5 or 1 s, or off); one bound to a parameter (`$brake_at`) sits at the seed's sampled value, the parameter's range
  as a band, and is changed through the parameter (or typed over in the form);
- a **distance** or **ahead** trigger has no time of its own: a hollow circle marks where the plan (below) has it
  fire, or the row says *not reached in the plan*;
- each **action** is a bar from its firing: a lane change for its duration (a numeric one dragged by its right end), a
  brake or speed change for as long as the plan's ramp takes.

The wheel zooms around the pointer (shift + wheel pans; *Fit* shows it all); a click on the axis moves the playhead.
The form below edits the selected event: the trigger's kind, actor and value (a number or an expression), and the
actions — actor, kind (changing it fills that kind's fields with their defaults: `ACTIONS` in `core/scenarios.py`)
and fields as `name=value, …`. **Add an event** (at the playhead), **Duplicate**, **Remove**, **Add an action**;
*Edit as JSON…* for anything else. Each change is one project edit — undo and redo apply — and nothing is written
until something changes, so a scenario looked at but not edited saves byte for byte as it was. Validation runs on
every change: a trigger without its kind's fields (`TRIGGERS`) is not applied, and says why; an unknown actor or
action is applied and flagged in red on the axis and under the form, as the runner would refuse it.

**The plan** (`core/scenario_plan.py`; shown above the tab in place of the sketch, with *Play the plan*) applies the
director's rules to the seed's sample on a straight schematic road with 3.5 m lanes: set speeds ramp at the actions'
rates, lane changes move linearly over their duration, walkers stand until *walk* and then cross from their side,
triggers are checked as the director checks them (time; the ego's centre within `lt_m` of the actor's; the actor's
rear more than `gt_m` ahead of the ego's front, with typical lengths). **It is a plan, not a simulation**: the ego
holds its set speed — the lane follower's braking and the autopilot are not planned, so a distance trigger that
depends on the ego slowing fires earlier here than in a run — there is no physics, and a map's lanes and curves are
not those of the schematic. A run in CARLA is what counts.

## Trigger rules
A **trigger rule** reacts to what happens in a run, at every tick: `when <condition> -> <action>[; <action>]`.

    when ego.speed > 40 and visibility < 30 -> spawn jaywalker
    when actor.ttc < 2 or ego.light:red and ego.light_m < 25 -> marker "close call"; weather fog_density=40
    when time >= 8 -> fault cam-front dirt cover=0.3 for=4

Rules belong to the project (`trigger_rules`, [project format](project-format.md#trigger-rule)) and are edited on
Scenarios ▸ **Trigger rules** (one row per rule, each change one undoable edit, problems listed under the table with
the column and the reason) or with `falcon triggers add|remove|list|check|signals`. Each says where it applies —
`drives` (Data ▸ Record, `falcon drive`, which takes `--triggers all|none|ID,…`), `scenarios` (a scenario run, its
sweeps and rare-case searches, their replays) and `live` (Live sessions) — and can be switched off. A run refuses to
start while a rule it would use does not parse or names a sensor its rig lacks; the config carries the rules
(`drive.json` `trigger_rules`), so `falcon reproduce` re-runs them.

**Conditions** reuse the [run query language](runs.md)'s comparisons — `key>n`, `key<n`, `key>=n`, `key<=n`,
`key=n` on numbers (a unit after the number is checked: `40 km/h`, `30m`, `2 s`), `key:value` on text, `!=` —
joined by `and`, `or`, `not` and parentheses (`and` binds tighter than `or`). The signals, all read from the world
snapshot of the frame just ticked (the scenario director's own state for the actors it moves kinematically):

| signal | unit | what |
|---|---|---|
| `time` (`t`) | s | simulation time since the run's first recorded frame |
| `ego.speed`, `ego.accel` | km/h, m/s² | the ego's speed; its longitudinal acceleration over the last step (braking negative) |
| `ego.junction_m` | m | from the ego's front along its lane to the next junction (`Waypoint.is_junction`, 2 m steps, straight-most at forks, look-ahead 150 m; none: inf) |
| `ego.light_m`, `ego.light` | m, text | from its front along its lane to the stop line of the next traffic light (`TrafficLight.get_stop_waypoints`); that light's state `red`, `yellow`, `green`, `off`, `unknown` or `none` |
| `visibility` | m | the fog's meteorological visibility, 7000 m / `fog_density` (measured relation, [sensor physics](sensor-physics.md)); no fog: inf |
| `fog`, `rain`, `wetness`, `puddles`, `cloud` | % | CARLA's `fog_density`, `precipitation`, `wetness`, `precipitation_deposits`, `cloudiness` in force |
| `sun`, `tod` | °, text | the sun's altitude; `day`, `low sun`, `twilight` or `night` (the run index's names) |
| `actor.nearest_m`, `vehicle.nearest_m`, `walker.nearest_m` | m | box to box ([criticality](methods.md#criticality)) to the nearest vehicle, walker or scenario prop within 80 m |
| `actor.ttc` | s | the smallest time to collision with any of them (boxes at constant velocity, horizon 10 s; none: inf) |
| `actor.count` | | vehicles, walkers and props within 50 m |
| `fired.<rule id>` | | how many times that rule has fired so far |

A signal that cannot be read makes its comparison false.

**Actions.** `spawn` adds an actor of a kind the scenario director already runs — `jaywalker` (`distance=25`,
`walk=1.4`), `child`, `cut_in` (`behind=8`, `faster=15` km/h over the ego, `gap=5`, `duration=1.5`),
`autorickshaw_cut_in`, `cattle` (`distance`, `lateral`), `handcart`, `stalled` (`kind=car|truck|van|bus|autorickshaw`),
`wrong_way`, or any scenario kind with `lane= ahead= lateral= heading= speed= start=`; arguments are range-checked.
`weather [Preset] field=value …` changes the weather (and with it the weather signals, the sensor physics and, at
night, the vehicle lights). `fault <sensor> <kind> key=value … [for=s]` starts a sensor fault from the next frame — a
timed one of known size ([faults](index.md#rigs): `drop` (default rate 1: the sensor dies), `freeze`,
`noise`, `dirt`, `rain`, `glare`, `rolling_shutter`, `compression`; mount, clock and motion-blur faults are set at
spawn and refused). `marker [text]` only marks the run.

**Repeat.** `once` (the default) fires the first tick the condition holds. `every` fires each tick the condition
becomes true, skipped within `cooldown_s` of the last firing. `cooldown` fires every tick it holds, at most once per
`cooldown_s` (> 0).

**Determinism.** Rules read only simulation state and time; their random choices (a walker's blueprint) draw from a
generator seeded by the run's seed and the rule's id, never the run's own. An actor a rule spawns is driven by the
scenario director — kinematic in scenario runs, so **a scenario run with rules still repeats bit for bit** (tested
against the fake; measured on the real server below); a drive without a scenario attaches a director that drives
only those actors (the ego stays on the traffic manager, whose physics is not bit-for-bit, so such drives repeat as
closely as drives do — `falcon reproduce` measures it).

**Records.** A firing is written on the frame whose state made the condition true (`frames.jsonl` `triggers`); its
actions act from the next tick. `run.json` `trigger_rules` lists the rules, every firing (step, time, the signals the
condition read, what each action did — or why it could not, e.g. *the map has no adjacent lane here*; the run goes
on), the counts and the actors spawned; a drive whose rules spawned actors also writes `trajectories.json` and their
criticality. The run index tags such runs `events:trigger`. In Live, firings are events (shown and recorded), a rule's
`dirt`/`drop`/`freeze` fault is shown as Live's black, dropped or frozen view, other faults are refused there.

Limits: the lane look-ahead follows the straight-most branch at a fork (not the route a traffic-manager ego will
take); `ego.accel` is a one-step difference (noisy under physics); a rule spawn placed where something already stands
is refused (*could not be placed*), not moved.

## Running one
A scenario runs inside a drive (so a scenario run records the rig's data with all its labels): the director picks a
start that has what the scenario needs (a straight run, an adjacent lane, an oncoming lane, a junction turn), places
the actors, fires the timeline, and records every actor's trajectory and the ego's collisions. A map that cannot host
a scenario anywhere is refused, saying what was missing. The ego is driven by the built-in lane follower (it brakes
for what is in its corridor, holds behind a stopped vehicle, and does not evade oncoming traffic — a baseline), at a
constant speed without braking, or by CARLA's autopilot.

### Bit-for-bit repetition — measured
On CARLA 0.9.16 (Town10HD_Opt, the reference machine), driven through CARLA's **vehicle physics**, two runs of the
same scenario with the same seed drift apart — the ego by 3 cm with no feedback controller and by up to 0.4 m with
its lane follower within 12 s — whatever the ego's blueprint, role name or sensors, with the world freshly reloaded
and every state read from the snapshot of the frame just ticked. The scripted actors driven by the same physics
repeated exactly in some runs, the ego never.

Scenario runs therefore default to **kinematic motion**: Falcon integrates every scenario vehicle (the ego included)
with a kinematic bicycle model driven by the same controllers, and walkers along their direction, and places them with
`set_transform` with physics off. Measured: **all eleven built-ins repeat bit for bit** with the same seed on the same
server (every trajectory identical, run twice each). Its limits: the wheels do not turn, and there are no tyre,
suspension or collision dynamics (collisions are measured on the boxes). `"motion": "physics"` in a scenario drives
them through CARLA's physics instead — realistic dynamics, not bit-for-bit. An ego on CARLA's autopilot always keeps
physics (the traffic manager drives through it).

## Criticality
Per actor and overall: time to collision, post-encroachment time, minimum distance, required deceleration, collisions,
and a criticality scalar for searches ([methods](methods.md#criticality)). A scenario's pass criteria (no collision,
a minimum time to collision) are checked against them in every run.

## Sweeps and rare-case searches
A **study** (`studies/<name>/` in the workspace) runs a scenario over many parameter sets unattended: a sweep over a
grid or seeded random samples, or a search — Bayesian optimisation, CMA-ES, adaptive sampling (cross-entropy) or
random — maximising the criticality. Every evaluation is a drive whose folder is named by its cache key; a study
resumes after a cancel and never runs the same parameters twice; a search starts from the sweeps of its study.
**Rare cases** shows how the search went, the failing cases ranked (each replayable as a drive and exportable as
OpenSCENARIO) and a map of the failure region over any two parameters. The objective can instead be a registered
model's **perception failure** on the run's labelled frames, or both, with weather fields searched beside the
scenario's parameters ([model-in-the-loop](rare-cases-mil.md)); *Why did this fail?* attributes a failing case to its
factors by re-runs ([root cause](root-cause.md)). Failures are labelled by Responsibility-Sensitive Safety and, by
default, only those the ego could have avoided are counted ([below](#avoidable-failures-responsibility-sensitive-safety)).

## Avoidable failures: Responsibility-Sensitive Safety
A rare-case search used to count any failure — including the lead car that brakes beyond any tyre's grip, or the
pedestrian who steps out a metre in front of the bumper. Those find nothing about the planner. Falcon now labels
every study run by **Responsibility-Sensitive Safety** (Shalev-Shwartz, Shammah & Shashua 2017, *On a Formal Model of
Safe and Scalable Self-driving Cars*, arXiv:1708.06374 v6; `core/rss.py`), and the criticality objective counts
**avoidable** failures by default.

The formulas (lemma numbers of that version; unit-tested against hand computations and the paper's own numbers — its
pedestrian example, ρ 0.5 s and 2 m/s² giving 50 cm, and its highway braking distance, 45 m from 30 m/s at 10 m/s²):

- same direction (Lemma 2): `d_min = [v_r ρ + ½ a_max,accel ρ² + (v_r + ρ a_max,accel)²/(2 a_min,brake) − v_f²/(2 a_max,brake)]+`
- opposite directions (Lemma 3): `d_min = (v1 + v1,ρ)/2 ρ + v1,ρ²/(2 a_min,brake,correct) + (|v2| + v2,ρ)/2 ρ + v2,ρ²/(2 a_min,brake)`
- lateral (Lemma 4): `d_min = μ + [(v1 + v1,ρ)/2 ρ + v1,ρ|v1,ρ|/(2 a_lat,brake) − ((v2 + v2,ρ)/2 ρ + v2,ρ|v2,ρ|/(2 a_lat,brake))]+`
  — the paper writes v1,ρ² and −v2,ρ², its case of both cars moving towards each other; the signed form, equal there,
  keeps a car still drifting away after ρ from being counted as approaching. Each road user has its own ρ and limits.

A time is **dangerous** when both distances are unsafe (Definitions 3, 7, 9); its Danger Threshold is the later of the
two unsafe runs' starts, and it says which **proper response** applies (Definition 10): longitudinal — the rear car
brakes at least a_min,brake after ρ, the front car at most a_max,brake, against the direction both brake — or lateral —
lateral braking to a μ-lateral velocity of 0 — or both. Whoever did not comply is responsible (Section 3.10).

| parameter | default | source |
|---|---|---|
| ρ ego / other | 1 s / 2 s | ad-rss-lib parameter discussion (conservative starting values; heuristic) |
| a_max,accel · a_min,brake · a_max,brake · a_min,brake,correct | 3.5 · 4 · 8 · 3 m/s² | same (4 and 8: a German driving-school rule of thumb) |
| lateral a_max,accel · a_min,brake · μ | 0.2 · 0.8 m/s² · 0.1 m | same |
| walkers: ρ · acceleration · braking | 0.5 s · 2 · 2 m/s² | the paper's example (Section 3.8) |
| compliance tolerance | 0.5 m/s², 0.1 m/s lateral | heuristic (differences of recorded velocities) |

**Labels** (the run's critical event: its first contact with the critical actor, else its minimum TTC, else its
minimum distance): **avoidable** — the ego did not give its proper response between the Danger Threshold and the event
(a real planner blind spot); **other** — the ego complied, the other did not (a cut-in inside the safe distance, a lead
braking harder than a_max,brake, a wrong-way rider who did not brake); **unavoidable** — the danger began with the other
already inside the ego's braking distance at a_max,brake (it appeared there: a pedestrian stepping out in front, a
cut-in a few metres ahead), or the other exceeded its kinematic feasibility limits, or both complied by these checks;
**safe** — the event was never dangerous in RSS's sense (a near miss by the criticality scalar only). With RSS
counting on (`--count avoidable`, the default; *Count* on the Rare cases page) a run whose label is not avoidable has
its objective capped at **0.45** (heuristic, below the default threshold 0.5; continuous, so the search keeps its
gradient) — keep the threshold above 0.45, or count all (`--count all`). Its criticality is kept. The ranked list
shows every failure with its label and why, the ones not counted after the counted; the failure-region map shows the
counted share; the study report lists both. Switching the count re-scores a study's runs; nothing is driven again.

**Kinematic feasibility.** A search also narrows the parameters that set an adversary's dynamics directly to what is
feasible (`core.rss.FEASIBILITY`, heuristic, labelled): a brake's deceleration ≤ 9 m/s² (≈ 0.9 g, a dry road's tyre
limit), a speed action's acceleration ≤ 4 m/s², a lane change's duration ≥ 2·√(lane / a_lat) = 1.87 s for a 3.5 m lane
at 4 m/s² lateral (accelerating then braking sideways) — so the built-in cut-in's 1.0–3.0 s becomes 1.87–3.0 s and the
autorickshaw's 0.8–2.5 s becomes 1.87–2.5 s; study.json lists what was narrowed (`--no-feasibility` searches the full
ranges). After each run the other's measured dynamics (longitudinal and lateral acceleration, jerk ≤ 8 m/s³, each
tolerated for 0.5 s — Falcon's adversarial-traffic realism bounds) decide the "exceeded its limits" label.

Limits: the road axis is the ego's heading (its mean over the run on the director's straight starts, else over the
previous 2 s), and the ego is taken to drive in its lane's direction; junction geometry and priority (Definitions
14–17), occlusion (Section 3.9) and evasive manoeuvres (Definitions 11–13) are not evaluated — an unprotected turn is
judged on a straight axis. Accelerations are differences of the recorded velocities. The parameters are not a
regulation: the paper asks for them to be set by one.

## OpenSCENARIO
CARLA's ScenarioRunner is not part of a CARLA 0.9.16 install and is not on the reference machine (it was not
verified with 0.9.16 here). Falcon runs scenarios with its own director and imports and exports a documented subset of
**OpenSCENARIO 1.x** (`core/openscenario.py`): relative lane teleports, speeds, simulation-time and relative-distance
triggers, speed and lane-change actions, custom commands for what 1.x has no action for (a door opening), a time stop
trigger. Every built-in round-trips. Anything outside the subset is refused on import, saying what.
