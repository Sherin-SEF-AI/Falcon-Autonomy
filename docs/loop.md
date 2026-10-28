# The loop

The Loop page (the home page) and `falcon loop` close the loop: **pick a model and a target operational design
domain (ODD) → evaluate → find the weak cells → search for rare cases there → generate targeted data → retrain →
evaluate again → the before and after, with intervals.** Each iteration is a versioned, reproducible record, and each
step's cost is shown before it runs. Code: `falcon_autonomy/core/loop.py` (its docstring is the reference);
the page is `falcon_autonomy/ui/pages/loop.py`.

The loop reimplements no tool. It runs the data engine (dataset plans, cache-keyed drives), the test harness
(`core.harness`: evaluation, drive-level intervals, paired comparison), the rare-case search (`sim.studio`,
`core.search`) and the training lab (`falcon train` as a job, as the Training page runs it).

## The ODD

The project's `loop.odd`, a JSON object (Loop ▸ Edit the ODD…, or `falcon loop PROJECT set --odd FILE`):

| field | meaning |
|---|---|
| `id`, `name` | |
| `rig` | the rig recorded; its first pinhole camera is evaluated and must record the `boxes2d` label |
| `worlds` | world ids (towns or imported places) |
| `traffic_profile` | a traffic profile id (empty: each world's own) |
| `weathers` | `[{"name", "preset", <CARLA weather fields>, "ranges": {field: [lo, hi]}}]` — each drive draws a ranged field uniformly, seeded by the drive's seed and the field's name |
| `classes` | the Falcon classes of interest (only those the model can name are scored) |
| `range_bins` | edges in metres from 0; the last bin is open (default 0, 20, 40, 60, 80) |
| `evaluation` | `drives_per_cell` (3), `drive_s` (20), `fps` (20), `record_every` (5), `seed` (1), `every` (1), `n_boot` (1000) |
| `weak_cells` | `min_objects` (30), `max_cells` (3), `fallback` (`none` or `lowest`) |
| `search` | `method` (bayes, cmaes, adaptive, random), `budget` (16 runs per cell), `batch` (4), `threshold` (0.5), `fps` (20), `traffic_vehicles` (0), `seed` (0) |
| `generate` | `objects_per_cell` (300), `drive_s` (30), `fps` (20), `record_every` (5), `max_drives` (20), `seed` (2), `reruns_per_failure` (1), `max_failures` (4) |
| `retrain` | `epochs`: `null` keeps the model's own |

A **cell** (a condition) is one world × one weather entry.

**Bengaluru, Indian mixed traffic, monsoon dusk** — *Use the Bengaluru monsoon-dusk example* (or
`falcon loop PROJECT set --odd example`) writes it, and says what the project still lacks: an imported Bengaluru
world (Worlds ▸ Import a place), the Indian mixed-traffic profile, a rig whose camera records `boxes2d`.

```json
{"id": "blr-monsoon-dusk", "name": "Bengaluru, Indian mixed traffic, monsoon dusk",
 "rig": "rig-default", "worlds": ["world-bengaluru"], "traffic_profile": "tp-india-mixed",
 "weathers": [
   {"name": "monsoon dusk", "preset": "MidRainSunset", "wetness": 80,
    "ranges": {"sun_altitude_angle": [-2, 6], "precipitation": [50, 90], "precipitation_deposits": [40, 80]}},
   {"name": "monsoon dusk, downpour", "preset": "HardRainSunset", "sun_altitude_angle": 2, "fog_density": 10,
    "wetness": 100}],
 "classes": ["pedestrian", "rider", "motorcycle", "bicycle", "car", "bus", "truck"],
 "range_bins": [0, 20, 40, 60, 80]}
```

Dusk is the sun's altitude between −2° and 6°; "monsoon" is CARLA's rain, wetness and puddles. CARLA's weather limits
apply ([worlds](worlds.md)): rain is water on the lens and wet surfaces, not loss of range; lidar and radar ignore
weather; the Indian profile's autorickshaw is a substitute.

## The evaluation set

`drives_per_cell` drives in every cell at fixed seeds (900 000 000 + `seed` × 10 000 + i), recorded through the data
engine once per ODD — its key hashes the worlds, weathers, traffic profile and rig as the project defines them and
the evaluation's drive settings — into `datasets/loop-eval-<key>/drives/` (cache-keyed) and listed with every drive's
run.json hash in `loop/evalsets/loop-eval-<key>.json`. Every iteration on the same ODD uses the same set, so their
numbers compare.

**It is never trained on.** The retrain step refuses (the step fails, saying which) any training, validation or test
source that is an evaluation drive, a folder holding one, a dataset version listing one, or a drive with an evaluation
drive's cache key; the generate step refuses a drive whose cache key is an evaluation drive's. Tested.

## The steps

| # | step | what it does | outputs |
|---|---|---|---|
| 1 | Pick | freezes the model (its registration and the SHA-256 of its file) and the ODD (normalised, hashed) | `pick.json` |
| 2 | Evaluate | records the evaluation set if needed; `core.harness.evaluate` on it, intervals over drives | `before/evaluation.json`, `predictions.jsonl` |
| 3 | Weak cells | AP per class × range bin × condition with intervals; the rule below | `weak_cells.json` |
| 4 | Search | a rare-case search per chosen cell | `search.json`, the studies |
| 5 | Generate | a targeted dataset and the failing cases re-run with labels | `generate.json`, the drives |
| 6 | Retrain | `falcon train` from the model's own settings on its sources + the new drives | `training/loop-NNN-…/` |
| 7 | Evaluate again | the retrained model on the same evaluation set | `after/evaluation.json` |
| 8 | Before and after | `core.harness.compare` and the same paired bootstrap per weak cell | `compare/comparison.json`, `cells.json` |

An iteration that follows a finished one whose step 7 evaluated this very model file on the same set reuses that
evaluation as its step 2.

### The weak-cell rule (heuristic)

From the evaluation's predictions: COCO AP of each class of interest in each range bin (COCO's area-range rule
applied to range, as the harness's breakdowns), for each condition and for all conditions together, each with a 95 %
percentile bootstrap interval over drives (`core.metrics.bootstrap`; over frames with one drive). A cell is **weak**
when it holds at least `min_objects` scored objects and the **upper end of its interval is below the overall AP** of the
evaluation (all classes). Weak cells are ranked by that upper end, lowest first — the surest shortfall first — ties by
more objects; the first `max_cells` of distinct class × range bin are chosen (a class and bin weak in one condition and
in all conditions together is chosen once).

It is a heuristic: a class's AP is compared with the mean over classes, so a class that is hard everywhere is always
weak; and with few drives the interval is not reliable (the evaluation says so below five drives). A model uniformly
weak (AP 0 everywhere, as a barely trained one) or uniformly strong has no weak cell: then nothing is targeted, and
the record says so — unless `weak_cells.fallback` is `lowest`, which targets the cells with the lowest upper ends
instead, each marked as a fallback.

### The search

Each chosen cell's class is mapped to built-in scenarios ([scenarios](scenarios.md)), in this order:

| class | scenarios |
|---|---|
| pedestrian | jaywalker from behind a bus, pedestrian crossing |
| rider | two-wheeler filtering, cyclist crossing |
| motorcycle | two-wheeler filtering, wrong-way scooter |
| bicycle | cyclist crossing |
| car | cut-in, lead vehicle brakes, parked car opens a door |
| truck | stalled vehicle (with `kind` held at truck) |
| bus | bus pulling out, jaywalker from behind a bus |

The first whose runs succeed in the cell's world is searched (a map that cannot host a scenario makes its runs fail,
and the next is tried), in the cell's weather (for "all conditions": the condition where that class and bin scored
lowest), with the ODD's rig **without label cameras** (the criticality needs none; it renders a third of the pixels).
A class without a scenario is served by the targeted dataset alone.

**The objective is the scenario's criticality** (`core.criticality`; 1 for a collision), **not the model's
detections**: the search finds where driving is hard in the weak cell's conditions, not where the model fails. The
range bin is not mapped onto scenario parameters.

### Generation

- **A targeted dataset spec** (`loop-NNN-targeted`, added to the project's dataset specs): `objects_per_cell` objects of
  each chosen cell's class, in its conditions; how many drives that takes comes from the class's yield **in its range
  bin**, measured on the evaluation set (objects per recorded frame), through `core.datasets.plan` (capped at
  `max_drives`). Drives are cache-keyed, so a re-run resumes.
- **The failing cases re-run with labels**: each search's failing cases (criticality ≥ `threshold`; with none, the most
  critical cases, and the record says so), up to `max_failures`, re-run with the labelled rig — the first re-run with
  the search's own seed (the same case, now labelled: kinematic scenarios repeat bit for bit), further ones
  (`reruns_per_failure` > 1) with other seeds, which start the scenario elsewhere on the map.

### Retraining

Only a detector Falcon trained can be retrained: its registration names its training folder, whose run.json holds its
settings. The step continues from them — the same architecture, initialisation, epochs, seed and sources — and adds the
new drives to the training sources; validation and test sources stay. A plug-in (the synthetic detector) or an imported
model **stops the step**, saying why: Falcon has neither its training code nor its data. The retrained model is exported
to ONNX, checked against torch, and registered in the project (`Name + loop N`).

### Before and after

The two evaluations are of the same frames. Each number is shown with its 95 % bootstrap interval over drives, and the
difference with a **paired** bootstrap over drives (both evaluations resampled together; `core.harness.compare`): AP and
AP50 overall, and the AP of each chosen weak cell. *Improvement* or *regression* only when the whole interval of the
difference is on one side of zero; otherwise "no detectable change at 95 %", which is not "no change".

## Costs

Before a step runs its estimate is computed, shown (the page's *Next* line, `falcon loop PROJECT status`, the job's
log) and recorded in the step; its wall time is measured after. Every estimate says where it comes from:

| what | source |
|---|---|
| drives (evaluation set, searches, generation) | `core.budget.Throughput`: this machine's measured throughput (*measured*), or before the smoke drive the reference machine's (*typical*); far outside the measured loads, *extrapolated from measured* |
| an evaluation | an earlier loop evaluation of the same model file on this machine (wall time per frame, *measured*); before any, **not estimated** (the step measures it) |
| training | images × epochs ÷ the images/s measured by the model's own training or another of the same architecture on the same device (run.json timings), plus its validation time; peak GPU memory the same way. Without one, on CUDA, the reference machine's numbers from [training](training.md) (*typical*); on the CPU, **not estimated** |
| weak cells, before and after | the last iteration's measured time for the step; before any, **not estimated** (a bootstrap over predictions; no simulation) |

Runs the tools have already made are not counted: a search's finished runs, the targeted drives already recorded, an
evaluation set already recorded. Nothing is made up: a step with nothing measured to go on says *not estimated* and
why.

## Records and resuming

`workspace/loop/NNN/iteration.json` (`falcon.loop_iteration/1`):

| field | meaning |
|---|---|
| `n`, `status` | the number; `running`, `complete`, `stopped` (a step refused, `stopped` says why), `failed`, `cancelled`, `abandoned` |
| `model`, `model_sha256`, `model_why`, `origin_model` | the model (its registration), its file's hash, why it was chosen, the project's model id |
| `odd`, `odd_sha256`, `odd_key`, `evalset` | the normalised ODD, its hash, the evaluation set's key and manifest |
| `steps.<step>` | `state`, `runs`, `started`, `finished`, `estimate` (seconds, label, source, parts; memory for training), `measured` (wall time; frames, latency, images/s as the step has them), `outputs` (workspace-relative paths → SHA-256), `upstream` (hash of every earlier step's outputs), `decisions`, `result`, `error` |
| `summary` | the before and after rows |
| `provenance` | Falcon's provenance block: the project and model files hashed, the ODD |

A step is done when its state is done, its outputs exist with their recorded hashes, and its `upstream` matches the
earlier steps' current outputs. Running again skips done steps; a step whose outputs changed runs again, and so does
everything after it — from the tools' caches (drives by cache key, studies from their results, trainings from their
checkpoints), so nothing finished is repeated. A cancelled step resumes inside its tool.

The project's `loop.iterations` holds one summary per iteration (`n`, `path`, `status`, `model`, `model_after`, `odd`,
`odd_sha256`, `stopped`, `AP_before`, `AP_after`, `AP_diff`, times); the record is the truth. The page applies these as
undoable edits when a loop job ends; the CLI writes them unless `--no-save`.

## The command

```sh
falcon loop PROJECT                       # status: the iteration, each step's state, the next step and its cost
falcon loop PROJECT set --model ID --odd FILE|example
falcon loop PROJECT next                  # run the next step
falcon loop PROJECT run [--until STEP]    # run to a step (1–8 or its name; default 8)
falcon loop PROJECT new                   # leave an unfinished iteration and start a new one (step 1)
falcon loop PROJECT show [N] [--json]     # an iteration's record
```

`next`, `run` and `new` speak the [protocol](protocol.md) (progress, ETA, a cancel on stdin or SIGTERM) and exit 0
(done), 2 (failed), 3 (cancelled) or 6 (stopped: refused — a model Falcon cannot retrain, an unusable ODD, or a
project whose model or ODD changed under an unfinished iteration). A finished iteration's next run starts a new one,
with the model it retrained when the project's model and ODD are unchanged.

## The page

*Model and ODD* (the model each new iteration starts with, the ODD summary and its problems); *Iteration* (the eight
steps with their state, estimate and measured time and decisions; the next step's cost; Run the next step, Run to step
N, Cancel, New iteration, Report…); *Weak cells* (the lowest cells with their intervals, which are weak, which were
chosen); *Before and after* (a chart of each row's before — blue circle — and after — orange square — with its 95 %
interval as a line, and the table of differences with their paired intervals and verdicts); *History* (every
iteration; double-click one to show it). The **Status** tab is what the first page always showed: the server, the
machine, the throughput and the latest runs.

## Limits

- 2D box detectors only (the harness's and the trainer's limit), one camera.
- The weak-cell rule is a heuristic (above); its intervals need five drives or more per condition to be read as such.
- The search's objective is the scenario's criticality, not the model's misses; a class without a matched scenario
  gets no search.
- Retraining needs a model Falcon trained; it repeats the model's whole training on the old and new data (no
  fine-tuning from the previous weights), so the before and after differ by the data, at the cost of a full training.
- One loop job at a time from the page; the steps share the machine-wide CARLA and GPU locks with every other job.
- Verified against the test suite's fake CARLA (tiny drives, the synthetic detector, a tiny CPU-trained detector);
  not yet run end to end on a real CARLA server with a GPU-trained detector.
