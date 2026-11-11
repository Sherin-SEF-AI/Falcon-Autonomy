# Generating data

[Active learning](active-learning.md) chooses which recorded frames go to training. A **generation round** goes one
step further: it asks the simulator for new data where the detector is weak, trains on it and checks, on a held-out set
the new data never touches, whether those cells got better. Training ▸ **Generation**; `falcon generate`. Code:
`falcon_autonomy/core/genround.py` (its docstring is the reference), `ui/pages/generation_tab.py`.

Nothing here is new machinery. The held-out set, its leak check, the per-cell AP table and the paired comparison are
the [loop](loop.md)'s; the drives are the data engine's (cache-keyed, resumable); the scenarios come from the
[coverage grid](odd-coverage.md)'s fill; the retrain is the [trainer](training.md) with the model's own settings; the
experiment is in the [registry](experiments.md#the-registry).

## Cells

A **cell** is a class × a condition. A condition is a weather class and a light class of the coverage grid
(`clear/day`, `rain/night`, `fog/twilight`, …: weather clear, cloudy, wet, rain, fog, dust; light day, low_sun,
twilight, night). Every drive of a condition — held-out or generated — uses that condition's representative CARLA weather
(`core.oddcoverage.CELL_WEATHER` and `CELL_SUN`, chosen so a run lands in its own class). A cell's AP is COCO AP of that
class on the held-out drives of that condition, with a 95 % percentile bootstrap interval over drives.

## The acquisition rule

For a cell c of class k in condition w (heuristic, stated in every record):

```
EV(c) = π(w) · gap(c) · s(c) · f(c)

π(w)   = exposure(w) / Σ exposure                 the condition's share of the domain (equal unless you weight it)
gap(c) = max(0, AP_ref − AP(c))                  on the held-out set; AP_ref the overall AP ("overall"), or 1
                                                 ("perfect": for a model weak everywhere, which has no gap to its mean)
s(c)   = sqrt(n0 / (n0 + n(c)))                  n(c) objects of k in condition w already in training (the model's
                                                 own drives and every round's data so far); n0 = 100
f(c)   = 1 + (failures + 1) / (runs + 2)         the coverage grid's scenario runs with an actor of class k in w:
                                                 high-failure cells count more, untested ones get 0.5
```

`gap` is the shortfall the data could close; `s` is the diminishing return of a learning curve (AP gained per object
falling as the square root of what is already there); `f` brings in the scenario runs' failures — the coverage grid's
under-sampled or failing cells. Cells with at least `min_objects` held-out objects and EV > 0 are ranked by EV (ties:
more held-out objects), and the first `max_cells` are chosen. **It is a ranking, not a prediction of the gain.** Real
sets in the training sources have no condition here and are not counted in n(c) (the rank says so).

**The budget.** A round records at most `budget.drives` drives and `budget.frames` frames; the drive count is
min(drives, frames ÷ a free drive's frames). It is shared among the chosen cells in proportion to EV (largest
remainders, one drive each first). Within a cell, `scenario_share` of its drives are **scenario runs** — a library or
project scenario with an actor of the class, copied for the cell by the coverage grid's `cell_scenario` (the
condition's weather and light, its ego speed narrowed to a band), with `traffic_vehicles` background vehicles — and
the rest **free drives** with traffic in the condition. A drive that would pass the frame budget is skipped and the
record says so. Labels come from the ground truth of every recorded frame (the rig's `boxes2d`): no annotation.

## A round

| # | step | what it does | outputs (`rounds/rNNN/`) |
|---|---|---|---|
| — | fix | before round 1 only: the held-out set recorded (`drives_per_condition` drives per condition at fixed seeds, through the loop's evaluation-set code) and hashed into `state.json`; the starting model evaluated on it | `baseline/` |
| 1 | rank | the rule above on the previous evaluation (the baseline, or the last round's) | `rank.json` |
| 2 | generate | the chosen cells' drives and scenario runs through the job queue | `generate.json` |
| 3 | version | a dataset version of every drive generated so far (`datasets/gen-<id>/`, all to train) | `version.json` |
| 4 | retrain | `falcon train` with the starting model's own settings on its sources plus that version | `retrain-config.json`, `training/gen-<id>-rNNN/` |
| 5 | evaluate | the retrained model on the held-out set | `after/` |
| 6 | compare | per cell before and after, each with its interval; the difference by a paired bootstrap over drives | `cells.json` |

**The held-out set is never trained on.** It is fixed before the first round, and every round checks its manifest
against the hash taken then (a re-recorded drive or an edited manifest is refused). A generated drive whose cache key
or folder is a held-out drive's is refused; so is any training source that is, holds or lists one — checked on the
starting model's own sources when the set is fixed and on every retrain. Held out by drive and seed, not by place:
the generated and the held-out drives share the world.

**Retraining** repeats the starting model's whole training (architecture, initialisation, epochs, seed) on its own
sources plus all the generated data so far — no warm start, so the rounds differ by their data only (the loop's choice;
[training](training.md) has no fine-tuning from earlier weights). Only a detector Falcon trained can be used: a plug-in
or an imported model is refused, saying why.

**Before and after** are evaluations of the same held-out frames: each cell's AP with its 95 % interval over drives, the
difference with a paired bootstrap over drives, *improvement* only when that interval lies wholly above zero. The
chosen cells decide; the other cells with enough objects are compared too, so a regression elsewhere shows.

**Stop.** The chain stops after a round in which **no chosen cell improved**, when **no cell has a positive expected
value**, or after `rounds` rounds; `state.json` says which. A stopped or complete chain is not run again (register a
new id). Each step is resumable: a step whose outputs exist with their recorded hashes and whose inputs have not changed
is skipped; drives are cache-keyed and trainings resume from their checkpoints.

## The definition

```json
{"id": "gen-1", "model": "my-detector", "world": "world-blr", "rig": "rig-default", "traffic_profile": "",
 "conditions": ["clear/day", "rain/day", "fog/day", "clear/night"], "exposure": {},
 "classes": [],
 "held_out": {"drives_per_condition": 3, "drive_s": 20, "fps": 20, "record_every": 5, "seed": 1, "n_boot": 1000},
 "budget": {"drives": 8, "frames": 400},
 "generate": {"drive_s": 20, "fps": 20, "record_every": 5, "scenario_share": 0.5, "traffic_vehicles": 10, "seed": 2},
 "acquisition": {"max_cells": 3, "min_objects": 30, "n0": 100, "reference": "overall"},
 "rounds": 3, "epochs": null}
```

`classes` empty: the model's. `exposure` weights the conditions (say a monsoon city's share of rain). The definition is
fixed once registered: the same id with another definition is refused.

## The command and the page

```sh
falcon generate plan PROJECT --id ID [--spec FILE] [--model M] [--world W] [--rig R] [--conditions c1,c2] \
    [--rounds N] [--budget-drives N] [--budget-frames N] [--max-cells K] [--json]
falcon generate new PROJECT --id ID [the same options]        # register it
falcon generate run PROJECT --id ID [--rounds N] [--fake-carla DIR]   # fix the held-out set, run or resume rounds
falcon generate show PROJECT --id ID [--json]                 # every round's cells, intervals and verdicts
falcon generate list PROJECT
```

`run` speaks the [protocol](protocol.md): 0 done (also when the stop rule ended the chain: the result says so), 2
failed, 3 cancelled, 6 refused (a chain already stopped or complete, no cell to rank, a model Falcon cannot retrain).

Training ▸ **Generation**: the detector (Falcon-trained ones), world, rig, conditions, the budget per round, the most
rounds and cells; *Plan*, *Register*, *Run or resume* (a job). On the right, each round's chosen cells with their EV,
the AP before and after with intervals, the paired difference as a bar on a common axis and the verdict; the stop
reason below. The assistant: `research_generate` (registers from a spec — the plan is its approval card — and queues
the run), `research_registry` with the id for the results.

## Files

`experiments/<id>/`: `experiment.json` (the definition and its hash), `plan.json`, `state.json` (the held-out set's
key, manifest and hash, the baseline, every round's status, why it stopped), `baseline/`, `rounds/rNNN/`,
`runs/round-NNN/` (each round as a tracked run: AP before, after and the difference, cells improved, drives, frames),
`results.json` (`falcon.generation_results/1`). The registry lists it with the rule as its hypothesis and the outcome
as its verdict.

## Measured

Only on the test suite's fake CARLA (`tests/test_genround.py`): two conditions × two held-out drives of 1 s, a budget of
three drives and 120 frames, a scratch Faster R-CNN (MobileNetV3-Small, 128 × 64, one epoch on the CPU). One round: a
60-frame scenario run (`lead_brake` copied for its cell) and two 5-frame free drives; rank 0.6 s, generate 16 s,
retrain 9 s, evaluate 3 s, compare 4 s; the whole test, the held-out set and the baseline included, 47 s. The model
detects nothing before or after (AP 0), so no cell improved and the stop rule ended the chain after round 1, as it
should. **Not yet run on a real CARLA server with a GPU-trained detector**: whether a round raises a weak cell's AP is
not measured.

## Limits

- 2D box detectors only, one camera (the harness's and the trainer's limits).
- The rule is a heuristic: the learning-curve exponent (½) and n0 are assumptions, the failure term mixes scenario
  outcomes (driving failures) into a perception ranking, and `gap` uses the point estimate. Its ranking is recorded with
  every term so it can be checked.
- A condition is a representative weather, not a distribution: every drive of `rain/day` has the same rain. Variety
  comes from the seeds (traffic, starts), not the weather.
- Held out by drive and seed, not by place: one world. A finding holds for that world.
- With fewer than five held-out drives per condition the intervals over drives are not reliable (the results warn).
- Each round retrains from the start: the cost grows with the data.
- Indian stand-ins (autorickshaw, cycle-rickshaw) are not mapped to a class here: their labelled class is the stand-in
  blueprint's, so their cells come only through free drives with the Indian traffic profile.
