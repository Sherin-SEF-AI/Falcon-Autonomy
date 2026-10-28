# Model-in-the-loop rare-case search

A rare-case search ([scenarios](scenarios.md#sweeps-and-rare-case-searches)) looks over a scenario's parameters for
where something fails. By default that something is driving: the objective is the run's
[criticality](methods.md#criticality). With a registered detector it can be **perception**: each evaluation runs the
scenario with the labelled rig, runs the model on the recorded frames and scores how badly the model saw the objects
that matter. Code: `falcon_autonomy/core/milsearch.py` (its docstring is the reference), `sim/studio.py` (the study),
`core/search.py` (the searchers).

## Objectives

| objective | the value searched | a failure |
|---|---|---|
| `criticality` (default) | the run's criticality, 1 for a collision | criticality ≥ threshold |
| `perception` | the model's **perception failure** (below), or one of its parts | perception failure ≥ threshold |
| `both` | min(perception failure, criticality) | both ≥ threshold: the model failed **and** driving was critical |

`both` is the minimum, the fuzzy "and": a case is a combined failure exactly when both reach the threshold (a product
would put two values of 0.7 under a threshold of 0.5).

## The perception failure

**The objects that matter**: the scenario's actors (the director reports each one's CARLA id, which the labels' track
ids are), and — when target classes are given — every object of those classes whose range lies in the range bin
`[lo, hi)` (default 0–50 m). Only what the harness scores counts: an object the model cannot name, with fewer than 20
visible pixels or more than 90 % of its box hidden is not *present* in that frame ([models](models.md)).

**Matching**, per frame, as the harness's failure gallery: per class, detections scoring at least `min_score` (0.5)
in score order, each taking the unmatched box of its class with the highest IoU ≥ `iou` (0.5), present objects before
ignored ones. A detection of the wrong class is a miss. A frame's time is (step + 1) / fps — the time of the
trajectories the criticality is computed on.

For each scenario actor present before its critical moment:

| measure | definition |
|---|---|
| critical moment t_c | its collision time; else the time of its minimum time to collision; else of its minimum distance |
| window recall | detected ÷ present frames in [t_c − `window_s`, t_c] (`window_s` 3 s); window miss = 1 − window recall |
| detection delay | first detected − first visible, both at or before t_c (the worst-case time to first detection; none when never detected) |
| lost lead | 1 − (t_c − first detected + Δ) ÷ min(`lead_s`, t_c − first visible + Δ), clipped to [0, 1]; Δ the frame period, `lead_s` 3 s; 1 when never detected — the share of the warning time the model could have given (up to `lead_s`) that it did not |
| failure of the actor | max(window miss, lost lead) |

Over the run: `window_miss` and `lost_lead` are their worst over the actors; `miss_rate` = 1 − detected ÷ present
object-frames over every object that matters; the **perception failure** (heuristic) is the worst actor's failure —
or the miss rate when no actor was present before its critical moment, or 0 when nothing that matters was present
(the record says "nothing the model could be asked to see"). `--metric` picks which of `perception`, `window_miss`,
`lost_lead` and `miss_rate` is searched; all are recorded, per actor too. All lie in [0, 1], higher is worse.

It is a heuristic: recall over a few frames is coarse (3 s at 20 fps is 60 frames, at 10 fps 30), and the critical
moment comes from the ego's driving, so an actor that never nears the ego is judged at its closest approach.

## The GPU

The CARLA worker records; the model runs afterwards, in the app's Python, on each batch's frames. By then the queue
has released the CARLA lock (the batch's drives have ended), and an ONNX or TorchScript model **takes the CARLA and
GPU locks** while it runs (`core.training.gpu_locks`), so a CARLA job and a GPU job never run at full load together;
the server stays up, idle. A Python plug-in — the synthetic detector — needs no GPU and takes no lock. The model is
loaded once per scoring pass and released after it. Tested: an ONNX-registered model's scoring holds both locks and
releases them.

## Searching the weather

A scenario's parameters do not include the weather, and the synthetic detector's weakness is in fog. `--weather FIELD`
(or `FIELD:LO:HI`; on the page, *Also search*: fog, sun altitude, rain) adds `weather.<field>` to the search space
(defaults: fog density 0–100, sun altitude −10–90°, precipitation, puddles, wetness and cloudiness 0–100). A value
overrides that field of the run's weather; the scenario's own parameters are unchanged.

## Studies, resuming and re-scoring

A study records its objective in `study.json` (the model's file hash, or for a plug-in its registration's hash, and
the settings) and its space. Every scored run keeps its full score in `perception/<run>-<objective key>.json`; the
results line holds a summary, the criticality and the objective. A study whose objective or model changed re-scores
its finished runs from their frames — nothing is driven again (tested: perception → both → criticality on the same
three runs). A search saves its searcher's state after every batch (`searcher-<method>.json`) and resumes from it
exactly; when it cannot (another objective), it rebuilds the searcher from the evaluations.

## Searchers: probability of failure

`bayes-pof` is Bayesian optimisation choosing the point most likely to reach the threshold, P(f ≥ t) = Φ((μ − t)/σ),
instead of the largest expected improvement — an active-learning-style rule that maps the failure region rather than
refining its peak. It was added because it measurably helps (the planted objective of `core.search`, 120
evaluations):

| acquisition | 5th failure (mean, seeds 0–4 / 100–109) | failures found at 120 | share of the region within 0.1 of a found failure, at 40 / 80 / 120 (seeds 0–9) |
|---|---|---|---|
| expected improvement (`bayes`) | 22.0 / 22.3 | 51.0 / 50.3 | 0.13 / 0.17 / 0.18 |
| probability of failure (`bayes-pof`) | 25.2 / 22.5 | 99.2 / 101.8 | 0.12 / 0.33 / 0.49 |
| straddle, 1.96σ − \|μ − t\| (Bryan et al. 2005) | 27.2 / 30.9 | 36.2 / 33.3 | 0.09 / 0.17 / 0.27 |
| expected improvement over the threshold | 28.0 / 21.5 | 96.0 / 103.3 | 0.09 / 0.11 / 0.11 |

As fast to the first failures as expected improvement, twice the failures and nearly three times the coverage of the
region by 120 evaluations; the other two did not help and are not offered. Seeds 100–109's coverage repeated it
(0.17 against 0.47 at 120).

## Measured: the synthetic detector's planted weakness

The synthetic detector misses small, occluded and fogged objects by a documented formula
([models](models.md)). The search should find that weakness in fewer evaluations than random sampling.

**Against the test suite's fake CARLA** (2026-10-29): `lead_brake` with `weather.fog_density` (five dimensions), a
160 × 90 camera recording `boxes2d` at 10 Hz, the synthetic detector with `h50` 6 px and `h_scale` 1.5 (its documented
24 / 6 px scaled to the small image), perception failure ≥ 0.8 as a failure; 30 evaluations in batches of 2, scenario
seed 11, search seeds 0–9. A failure is rare: 7.0 % of the 300 uniform evaluations (the random runs).

| method | first failure (mean; 31 = none in 30) | 3rd failure | failures at 30 | seeds with a failure |
|---|---|---|---|---|
| random | 15.9 | 27.7 | 2.1 | 8 of 10 |
| Bayesian optimisation (`bayes`) | 12.0 | 20.7 | 6.2 | 10 of 10 |
| probability of failure (`bayes-pof`) | 11.0 | 18.1 | 7.5 | 10 of 10 |

Read honestly: both surrogate searches find three times as many failures as random and reach the third sooner; the
first failure comes only modestly sooner, because Bayesian optimisation starts with 12 uniform points (2 × 5 + 2) — the
same 12 the random search draws with the same seed — so on seeds where random found one early, all three did. Each
run took about 4 s on the fake. The fake renders no fog: the planted fog weakness acts through the detector's formula
only.

**On a real CARLA 0.9.16 server** (2026-10-29; Falcon's own server, 8 GB cap, Town10HD_Opt, RTX 5080, the machine
shared with other work): `lead_brake` with `weather.fog_density`, a 640 × 360 camera recording `boxes2d` at 20 Hz, the
synthetic detector with `h50` 12 px and `h_scale` 3 (its documented 24 / 6 px at half the resolution). The threshold
was fixed **before** the comparison, from 12 separate uniform runs (search seed 100): the lowest of 0.5, 0.6, … with
at most 20 % of them failing — 0.5 (their values 0.02–0.44: none failed). Then 24 evaluations per method in batches of
4, scenario seed 11, **one** search seed (0):

| method | first failure | 3rd failure | failures at 24 | failures in evaluations 13–24 | best value |
|---|---|---|---|---|---|
| random | 6 | — | 1 | 0 | 0.59 |
| Bayesian optimisation (`bayes`) | 6 | 17 | 9 | 8 | 0.62 |
| probability of failure (`bayes-pof`) | 6 | 16 | 6 | 5 | 0.64 |

Failures are rare on the real server: 1 of the 36 uniform runs (2.8 %). The first 12 evaluations are the same
uniform points for the three methods (the shared seed and Bayesian optimisation's initial design), so all three met
the one early failure at the 6th; after them the surrogate searches went to the planted region — a large gap (37–40 m,
the parameter's top), hard braking, 40–50 km/h and dense fog (62–100) — and found 8 and 5 more where random found
none. One seed and 24 evaluations: an illustration on the real renderer, not a statistical comparison (the fake's ten
seeds are that). A run took 24–40 s (the map is reloaded before every scenario run for bit-for-bit repetition).

## The loop

The ODD's `search.objective` (`criticality` by default, `perception` or `both`; `search.metric` as `--metric`) makes
step 4 model-in-the-loop: the search runs with the ODD's **labelled** rig (its label cameras render too, where the
criticality's search drops them; the step's estimate counts them), scores the iteration's model on the scenario's actors
and on the cell's class within the cell's range bin, and searches the cell weather's ranged fields (its `ranges`) as
dimensions instead of drawing them. The criticality stays the default; a model that cannot be run (a segmenter, a
missing file, a runtime not installed) falls back to the criticality and the step's decisions say why. The failing
cases re-run in step 5 are then the perception failures.

## The page and the command

**Rare cases**: *Objective* (criticality, perception failure of a model, both), *Model* (the project's detectors),
*Also search* (fog, sun altitude, rain), the method (now with *Bayesian optimisation, probability of failure*), the
simulation rate. Each objective, model, searched weather and rate gets its own study (`mil-<scenario>-<model>-<objective>
[-fog…][-10hz]`), so a study's values compare among themselves. The failing cases show the objective, the criticality
and the perception failure; *Why did this fail?* attributes the selected case ([root cause](root-cause.md)).

```sh
falcon search PROJECT --scenario lead_brake --name fog-mil --objective perception --model syn \
    --weather fog_density [--metric perception|window_miss|lost_lead|miss_rate] [--target-class car] \
    [--range 0,50] [--window 3] [--lead 3] [--iou 0.5] [--min-score 0.5] [--every 1] [--fps 20] \
    [--method bayes|bayes-pof|cmaes|adaptive|random] [--search-seed N]
falcon sweep PROJECT --scenario S --name N --samples K --objective both --model ID --weather fog_density:20:100
```

`--objective perception` without `--model` is refused (exit 6); so is a rig whose camera does not record `boxes2d`
and a model that cannot be run.

## Limits

- 2D detectors and one camera (the harness's limits).
- The perception failure is a heuristic; its parts are recorded so another reading can be made from the same runs.
- A recorded run renders the label cameras: a perception search renders more than a criticality search, whose rig
  drops them (the cost estimate counts them).
- The synthetic detector is not a model; its numbers here measure the search, not a detector.
