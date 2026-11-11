# A scenario from a description

Describe what should happen — *"an autorickshaw cuts in from the left at a speed breaker at dusk"* — and Falcon
writes it as a scenario, runs it, and checks that **the run did what the words said**: the description is split into
checkable claims (an autorickshaw is there; it cuts in from the left; a speed breaker is within 30 m; the light is
twilight), and each claim is verified against the run's recorded ground truth — met, not met or not checkable, with
the measured value. Code: `falcon_autonomy/core/scenario_text.py` (writing), `core/scenario_check.py` (the claims and
the check; their docstrings are the reference), `schemas/scenario.schema.json` (the format), `sim/scenariotext.py`
(`falcon scenario`), `ui/scenario_text_panel.py` (Scenarios ▸ Describe a scenario).

## Writing it

**With the assistant on** (Settings ▸ Assistant: your Groq key and consent, [assistant](assistant.md)), the main
model (`openai/gpt-oss-120b` by default) writes the scenario as one JSON object. What is sent is **text only**: the
description, the scenario schema, a paragraph of the format's rules (lanes relative to the world's traffic
handedness, the triggers and actions, the sun altitudes and weather fields for light and weather, the
`speed_breaker` need), one scenario as an example — Falcon's own draft from the description's phrases when it can
make one, else the library's autorickshaw cut-in — and, in a repair round, the validation errors. Never a run, an
image, lidar, audio or anything from the dashcam footage; a description that names a private location (your
footage) is refused before anything is sent.

**Validation, then repair.** The reply is parsed (a code fence or prose around the object is tolerated) and checked:

1. against `schemas/scenario.schema.json` (JSON Schema 2020-12: the fields, the actor kinds, the lanes, the triggers,
   the actions, the weather fields and their ranges, expressions of `$parameters`);
2. by `core.scenarios.validate` (actor ids, actions on known actors, trigger fields, that it can be sampled) and
   Falcon's own checks (a `$kind` parameter is a choice of known kinds; the ego's speed within 0–130 km/h);
3. against **the description's claims, as far as the scenario itself shows them** (mismatches): an actor of each
   named kind, a `lane_change` to `ego` for a cut-in and a start on the named side (the adjacent lane is on the
   overtaking side — the left in right-hand traffic, the right in left-hand traffic; the kerb the other side), a
   `brake` action for braking, heading `across` for a crossing, `start: stopped` for standing still, heading
   `against` for the wrong way, a start behind the ego for an overtake, `speed_breaker` in `needs`, a sun altitude in
   the light's class, the weather's class, the ego's speed.

An invalid reply (1–2) is sent back with its errors; so is a valid one with mismatches (3) while rounds remain. Two
repair rounds by default (`--rounds`, at most 4): at most three requests. The result is the last valid scenario —
with its remaining mismatches listed — or none. Every attempt's errors are kept in the record
(`workspace/scenario-text/<id>.json`, `falcon.scenario_text/1`, with the number of requests and characters sent).

**Without a model** (offline, or *Falcon's draft only*) the claims are mapped onto a library scenario: the manoeuvre
picks the base (a cut-in: the autorickshaw cut-in for an autorickshaw, else the cut-in; braking: the lead brake; a
crossing: the pedestrian or cyclist crossing; standing still: stray cattle for cattle, else the stalled vehicle; the
wrong way: the wrong-way scooter; overtaking: two-wheeler filtering), the named kind replaces its actor's, a side that
is not the overtaking side moves the actor to the kerb (a crossing to the far sidewalk), the light and weather set
the coverage grid's representative weather ([ODD coverage](odd-coverage.md): sun 0° for twilight, −30° at night, 12°
for a low sun, 45° by day; rain, fog, wet road, cloud and dust fields), a speed breaker adds the need, the ego's speed
fixes its parameter. A description without a manoeuvre has no draft.

The scenario is kept with its `source`: `{"format": "text", "description", "claims", "mode", "rounds", "model"}`, and
an id `txt-<hash>` of the description and the scenario.

**At a speed breaker.** A scenario's `needs` may hold `speed_breaker`: the director starts the ego where a speed
breaker of the map lies 10–60 m ahead along its lane, within 4.5 m of it (`workers/director.py`). The breakers are the
road features of an imported world's OpenDRIVE file ([Indian road assets](indian-assets.md#road-features-in-opendrive-real-geometry);
Worlds ▸ Road features): a CARLA town has none, and a run asked of one is refused before it starts. The need places
the start, not the event: whether the cut-in happened at the breaker is what the check measures.

## The claims

Falcon's phrase set (deterministic, offline; `PHRASES` in `core/scenario_check.py`):

| claim | phrases | example |
|---|---|---|
| actor | autorickshaw, auto, tuk-tuk, three-wheeler, rickshaw · cycle rickshaw · scooter, motorbike, bike, two-wheeler · cyclist, bicycle · car, taxi, sedan, SUV · truck, lorry · bus · van, tempo · pedestrian, person, man, woman, jaywalker · child, kid · cow, cattle, bull, buffalo · handcart, pushcart, cart · vendor, hawker | *an autorickshaw* |
| manoeuvre | cuts in, cut-in, swerves in, merges in front, pulls out · brakes, brakes hard, slams on the brakes, stops suddenly · crosses, walks across, steps out, jaywalks · stalled, broken down, stopped, parked, stationary, stands in · wrong way, against the traffic · overtakes, filters, weaves — with *from the left / right* | *cuts in from the left* |
| location | speed breaker, speed bump, hump, bump, speed table · pothole, dip · junction, intersection, crossroads, roundabout, traffic light — with *within N m* | *at a speed breaker* |
| light | dusk, dawn, twilight, sunset, sunrise, evening · night, dark · low sun, glare · noon, midday, daytime, daylight, afternoon | *at dusk* |
| weather | rain, monsoon, downpour, drizzle · fog, mist, haze, smog · dust storm, dusty · wet road, waterlogged, puddles · cloudy, overcast · sunny, clear sky | *in the rain* |
| ego speed | *we drive at 40 km/h*, *the ego at 30 kmph*, *driving at 50 km/h* | |
| outcome | *without a collision*, *avoids it*, *stops in time* · *collides*, *a collision*, *hits* | |

A manoeuvre takes the kind named nearest before it in its sentence (*a parked bus*: the one right after). Words of
four letters or more that no phrase covers are listed as **not recognised** — the description's parts Falcon cannot
check. With the assistant on, the model is also asked to turn the description into claims of the same types (only
the text is sent; each claim is validated against the claim schema and those of a kind the phrases already gave are
dropped).

## Checking a run

Each claim against what the run recorded (`trajectories.json` — the director's record of every scenario actor and
the ego, the rows the criticality uses —, run.json, drive.json, the map's OpenDRIVE file). Positions are in the ego's
frame: forward, and lateral positive to the right (CARLA's frame); *left* is a negative lateral.

| claim | met when (thresholds: `THRESHOLDS`, stated in every report) | measured |
|---|---|---|
| actor | a scenario actor of that kind spawned (its CARLA id in run.json) and recorded (≥ 2 rows); a stand-in or proxy also recorded with that subclass | the actor, CARLA id, blueprint, subclass, seconds recorded |
| cut-in | first seen ≥ 2.0 m to one side; later within 1.0 m of the ego's centre line with its rear 0–25 m ahead of the ego's front; from the named side | the side and lateral offset it came from, when it entered, the gap then |
| braking | ahead within 1.75 m laterally, decelerating ≥ 3.0 m/s² (averaged over 0.25 s) | peak deceleration, when, the gap |
| crossing | moving across the ego's heading (45°–135°), entering the ego's path (within 1.0 m) from ≥ 2.0 m to one side, ahead | when it entered, how far ahead, from which side |
| standing still | ahead within 1.75 m laterally and 100 m, below 0.3 m/s for ≥ 1 s | how long, how far ahead |
| wrong way | ahead within 1.75 m laterally, moving, heading ≥ 135° from the ego's | the heading difference, how far ahead |
| overtaking | from behind the ego to ahead of it, within 3.5 m laterally as it passes; on the named side | the lateral offset and side at the pass |
| location | the ego within N m (30 by default) of a speed breaker or dip of the map's OpenDRIVE road features, or of a junction road (core.rules) — at the moment of the first manoeuvre found, otherwise at the closest approach | the distance, the feature, when |
| light | the sun's altitude of the run's weather in the class: night < −6°, twilight −6° to 6°, low sun 6° to 20°, day ≥ 20° | the altitude |
| weather | the run's weather class (core.oddcoverage); *wet* by wetness or puddles ≥ 40, *cloudy* by cloudiness ≥ 50 | the CARLA fields |
| ego speed | the ego's median speed over its first second within ±max(5 km/h, 15 %) | the speed |
| outcome | the director's collision events: none, or at least one | the count |

When a manoeuvre is found, the **grounded language labels** of [the language data](world-model.md#language-data-falconlanguage1)
(`core.language.build_state` on the recorded actors and lanes) are given beside the numbers, at the actor's first
sample and at the moment found — *"an autorickshaw ahead on your left"*, then *"an autorickshaw ahead"*.

**Not checkable**, and why: actors and manoeuvres on a run without a scenario (a free drive — background traffic is
not identified by kind), a breaker or dip on a map without its OpenDRIVE file (a CARLA town whose file Falcon cannot
find), a junction on a run without lanes.json or an OpenDRIVE file, a weather not recorded.

The report is `RUN/scenario-check.json` (`falcon.scenario_check/1`): every claim with its phrase, status, measured
values, the sentence shown, the thresholds, the recognised and unrecognised words, what each check read.

## The page

Scenarios ▸ **Describe a scenario**: the description; *Write the scenario* (the assistant's model when it is on —
said in the status line —, Falcon's draft otherwise or with *Falcon's draft only*); the status — valid or not, by
whom, the attempts, each mismatch and problem, the words not recognised, what was sent; the scenario's JSON (nothing
is added yet); *Add and run* (an undoable insert, then one run with the page's world, rig and seed); the claims table,
filled when that run ends — the claim, the words it came from, met / not met / not checkable, the measured value (the
full measurements as a tooltip); *Check the last run* checks the newest run of the scenario again.

## The commands

```sh
falcon scenario from-text PROJECT "an autorickshaw cuts in from the left at a speed breaker at dusk" \
    [--world W] [--rig R] [--seed N] [--rounds 2] [--offline] [--model M] [--apply] [--run] [--fake-carla DIR] [--json]
falcon scenario check RUN ["DESCRIPTION"] [--claims FILE] [--model] [--json] [--no-write]
```

`from-text` prints the claims, each attempt's errors, the scenario and the record's path; without `--offline` it needs
the assistant on (exit 2, nothing sent, otherwise). `--apply` adds the scenario to the project; `--run` also runs it
once and checks the run. `check` takes the run's own description when none is given (a scenario written from text
keeps it) and exits 0 when no claim is unmet, 1 when one is, 2 on an error.

The assistant (the **research** group): `research_scenario_check` gives a description's claims and Falcon's draft,
validates the agent's own scenario against the schema and the description (the repair rounds: at most five invalid
attempts per description in a session), and checks a run; `research_scenario_text` adds a valid scenario and runs it
once.

## Limits

- **The phrase set is small and literal.** Negations other than the outcome's (*no cut-in*), counts (*two
  buses*), relative positions (*behind the bus*), speeds of actors and times are not claims; what it does not
  recognise is listed, not guessed. The model's claims, when asked for, are only the same types.
- **Thresholds are Falcon's heuristics** (2.0 m to a side, 1.0 m in the lane, 25 m for a cut-in, 3 m/s² for hard
  braking, 30 m for *at*), not standards; they are in every report and can be passed to `check_run`.
- The ego-frame lateral offset assumes a roughly straight road between the ego and the actor; on a curve an actor in
  the next lane can read as nearer or farther.
- Dusk and dawn are both twilight: the sun's azimuth is not read. Light and weather are the run's settings, not what
  the camera saw (CARLA's limits apply: [worlds](worlds.md)).
- Distances to a breaker are to its point on the road's reference line, so they include up to a lane or two of
  lateral offset; junctions need the run's lanes or the map's OpenDRIVE file.
- `speed_breaker` places the start 10–60 m before a breaker; the timing of the event is the scenario's. A junction
  can be asked for only by a turn across traffic (`junction_turn`); a dip by nothing.
- Tested against the fake CARLA (a straight road with an IRC hump written into its OpenDRIVE file) and a scripted
  model and fake Groq server; not yet on a real CARLA server, nor measured against the real model over a set of
  descriptions.
