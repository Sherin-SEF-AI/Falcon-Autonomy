# Runs: search, drive sets, comparison and dataset health

Data ▸ Browse, Compare and Health (and `falcon runs search`) work on the **run index**: one summary per recorded run of the
workspace, read from what the run itself wrote. Nothing in it is estimated: a fact a run did not record is shown as
"not recorded", never as a zero.

## The run index

Every folder with a `run.json` under the workspace's `runs/` and `studies/*/runs/` (and any other run the catalogue
knows) is indexed by `falcon_autonomy.core.runindex`:

| field | from |
|---|---|
| name, status, worker, started | `run.json` |
| kind | `route` (a dashcam route), `search` (a study's run), `scenario`, else the worker (`drive`, `replay`, `live` …) |
| world, rig, map | the config in `run.json` provenance: the project's world and rig ids, the town or the imported world's folder |
| weather | the preset (or `custom`), and its **kind**: `rain` (any rain), `wet` (wetness or puddles, no rain), `cloudy` (≥ 50 % cloud), else `clear` — heuristic thresholds |
| time of day | from the sun's altitude, as the Worlds page names it: night below −6°, twilight below 6°, low sun below 20°, else day |
| scenario, values | the scenario's id and its parameter values |
| duration, frames, fps, seed | `run.json` performance and config; the frame count from `frames.jsonl` when performance has none |
| speed (mean, max), distance | `frames.jsonl`: the ego's recorded speed, and its path length (a teleport's jump is not counted) |
| collisions, minimum TTC, passed | the scenario's criticality in `run.json` (core.criticality); **none for a plain drive**, which has no collision check |
| teleports | `frames.jsonl` `"teleport"` (the road-end guard: [worlds](worlds.md)), seconds from the first frame |
| time outside the ODD | the ODD monitor's summary, when the run was monitored ([ODD monitoring](odd-monitoring.md)) |
| rule violations | the newest saved traffic-rule check of the run (Data ▸ Browse ▸ *Save for a report*), when there is one |
| events | `collision`, `teleport`, `odd_exit`, `failed_check`, `fault`, `rule_violation`, and the status when not `ok` |

The summaries are cached in `<workspace>/runindex.json`, keyed by each run's path and the sizes and modification times
of `run.json`, `frames.jsonl` and `odd-monitor.json`: a run is re-read only when one of them changed. The cache can be
deleted at any time.

## Searching

The Browse search box and `falcon runs search PROJECT QUERY…` take the same language. Terms are ANDed:

| term | matches |
|---|---|
| `world:w-blr`, `map:Town10`, `rig:…`, `scenario:lead`, `kind:search`, `status:ok`, `worker:drive`, `folder:studies` | the field contains the value (case-insensitive) |
| `weather:rain` | the preset's name **or** the weather kind (`clear`, `cloudy`, `wet`, `rain`) |
| `tod:night` | the time of day (`day`, `low sun`, `twilight`, `night`) |
| `events:collision`, `tag:golden` | one of the run's events, one of its tags |
| `dur>20`, `frames>=100`, `speed>50`, `mean_speed<10`, `dist>500`, `ttc<1.5`, `collisions>0`, `seed=3`, `rain>=30`, `fog>10`, `sun<0` | a number (`>`, `<`, `>=`, `<=`, `=`); a run without the number never matches |
| `after:2026-10-28`, `before:2026-11-01T12:00` | the start time (UTC), as a date or a date-time |
| `-term` | the term negated: `-kind:search`, `-tag:bad` |
| anything else, `"quoted words"` | free text in the name, folder, world, map, weather, scenario, kind, status and tags |

A malformed term (`dur>abc`, `world>3`, `after:yesterday`, an unclosed quote) is refused with the reason. Column
headers sort; dates and durations sort as values.

```
falcon runs search ~/FalconWorkspace/Bengaluru.falcon.json kind:search speed>40 --sort duration
falcon runs search ~/FalconWorkspace/Bengaluru.falcon.json events:collision --json
```

## Tags

Tags are yours, not the run's: they live in the project (`run_tags`, keyed by the run's folder relative to the
workspace), so a recorded run is never modified, and tagging is undoable like every project edit. Browse ▸ *Tags…*
(or right-click) sets the selected runs' tags; `tag:` searches them.

## Drive sets

A drive set (`drive_sets` in the project) is either a **list** of runs (Browse ▸ *Save as a set…*, or *Save checked…*
under any list of drives) or a **saved query** (Browse ▸ *Save search…*), which is evaluated again whenever the set is
used — a set of "every rainy night drive" grows as such drives are recorded. A listed run that is no longer on disk
drops out.

Every list of drives — Models ▸ Evaluate, Training (detectors, the sim + real experiment, active learning's pool and
test, behaviour cloning), the VLM evaluation, world-model training and deployment's evaluation and calibration drives
— has a **Drive set** picker: choosing a set checks its drives in that list (and says how many of the set the list
does not offer, such as a failed run or a replay in a list of finished drives). The Loop and Rare cases pages make
their own drives, so they have no picker.

## Comparing runs

Data ▸ Compare (or Browse ▸ *Compare* with two to four runs selected) shows the runs side by side:

- **One clock.** The runs are aligned by simulation time from each run's first frame, not by step: at time *t* each
  shows its frame nearest *t* (a run at 10 Hz next to one at 20 Hz shows the same instant), and a run that has ended
  says when. The scrubber's arrows move by the smallest step of the runs.
- **Events** of each run on a shared timeline: collisions (red), rule violations (amber), teleports (orange), time
  outside the ODD (blue), scenario triggers and the minimum TTC (grey); hover for the details, click to go there.
- **What each recorded**, with the difference from the first run (A): duration, step, frames, distance, mean and
  maximum speed, collisions and minimum TTC (scenario runs only), teleports, time outside the ODD (monitored runs
  only), and from a traffic-rule check of each run made in the background (core.rules, the rule set by the run's
  traffic side): the rule violations and the ego's **lane margin** — how far its body is inside its lane — minimum and
  median (not recorded without a lane map). "never closing" for the minimum TTC means the scenario's actors were
  never on a collision course.

## Dataset health

Data ▸ Health measures a set of drives — all finished drives, Browse's selection, or a drive set
(`falcon_autonomy.core.datahealth`):

| measure | method |
|---|---|
| class balance | objects per class in the camera frames' box labels (`objects/<camera>/<step>.json`; the rig needs the `boxes2d` layer), with each class's share |
| empty frames | labelled camera frames with no object |
| box sanity | boxes without area, wholly outside the image, or **tiny** (under 8 px high or 64 px²) — thresholds stated, not learned |
| near duplicates | a 64-bit difference hash (dHash: a 9 × 8 grey thumbnail, one bit per horizontal gradient sign) per sampled RGB frame; a frame within **4 bits** of an earlier kept frame of the set (in recording order) is a near duplicate. It flags a stopped car's identical frames, not merely similar scenes |
| condition coverage | frames per world × time of day (day, low sun, night) × weather kind (clear, cloudy, wet, rain): 12 cells per world of the project, and how many are empty. Twilight and other combinations are listed when seen |
| scenario values | each parameter's recorded values in ten bins over the scenario's own range (the values seen, for a scenario not in the project or the library) |
| suggestions | the empty cells (worlds that already have data first), then the thinnest, each with the CARLA preset that gives it: Clear, Cloudy, Wet or SoftRain + Noon, Sunset or Night. *Record this…* opens Data ▸ Record with that world and that weather for the drive alone |

At most *Frames read* frames (3000 by default) are read, evenly spaced over the set; the report says how many and
nothing is extrapolated to the frames not read.

**Measured** (tests, synthetic runs): the class counts, tiny, degenerate and outside boxes, empty frames and the
coverage cells match what was written exactly. On three 12 s search runs recorded against the fake CARLA (720
frames) the analysis took 0.6 s and found 98 % near duplicates — the fake server renders the same image every step,
which is what the measure is for.

## Limits

- The index reads what the workers write; runs from before a field existed show it as not recorded.
- Collisions are recorded for scenario runs only (their director's collision check); a plain drive's collisions are
  not detected (CARLA's collision sensor is not on the rig).
- Rule violations in the index come from saved rule checks only; Compare checks each run itself, which takes seconds
  per minute of driving.
- The weather kind and the time of day are bucketed by fixed thresholds; fog is a number (`fog>…`), not a kind,
  because CARLA's night presets carry a fog density of 60.
- Near duplicates are found against every kept frame of the set (quadratic in the kept frames: 3000 frames take
  under a second), and only by appearance.
