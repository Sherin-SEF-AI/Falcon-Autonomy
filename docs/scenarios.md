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
behind a bus, a wrong-way scooter, a bus pulling out. Copy one into the project to edit it.

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
factors by re-runs ([root cause](root-cause.md)).

## OpenSCENARIO
CARLA's ScenarioRunner is not part of a CARLA 0.9.16 install and is not on the reference machine (it was not
verified with 0.9.16 here). Falcon runs scenarios with its own director and imports and exports a documented subset of
**OpenSCENARIO 1.x** (`core/openscenario.py`): relative lane teleports, speeds, simulation-time and relative-distance
triggers, speed and lane-change actions, custom commands for what 1.x has no action for (a door opening), a time stop
trigger. Every built-in round-trips. Anything outside the subset is refused on import, saying what.
