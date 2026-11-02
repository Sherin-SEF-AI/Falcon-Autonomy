# Coverage of the operating domain

Which parts of the operating domain have been tested, how many runs each has, and which are untested or have too few
runs to say anything — as a grid of the ODD: **weather × road type × actor kind × ego speed × light**. The
[failure-rate estimates](failure-rate.md) appear in their cells, and the empty or under-sampled cells can be filled:
Falcon writes a scenario for each and queues a rare-case search (or a sweep) of it. Code:
`falcon_autonomy/core/oddcoverage.py` (its docstring is the reference), `sim/estimate.py` (`falcon coverage`),
`ui/failrate_panel.py` (Rare cases ▸ Operating-domain coverage).

## The cells

Each finished scenario run is placed by its own config (`drive.json`, a suite's `stack.json`), read with the safety
case's `run_coordinates` ([stack testing](stack-testing.md)):

| dimension | labels | from |
|---|---|---|
| weather | clear, cloudy, wet road, rain, fog, dust storm | the CARLA weather the run used, in this order (heuristic thresholds): rain when precipitation ≥ 10; fog when fog_density ≥ 20 and the fog starts within 50 m (CARLA's ClearNight has density 60 starting at 75 m: not fog); dust when dust_storm ≥ 50; wet when wetness or puddles ≥ 40; cloudy when cloudiness ≥ 50; else clear |
| road | straight, multi-lane, two-way, junction | the most specific road feature the scenario required at its start (the safety case's heuristic: what the director required, not the whole route) |
| actor | the scenario kinds (car … vendor) | each kind of the scenario's actors — a run with a bus and a pedestrian is a run in both kinds' cells |
| speed | 0–20, 20–40, 40–60, 60–80, 80–130 km/h | the ego's set speed |
| light | day, low sun, twilight, night | the sun's altitude as `core.weather.describe` reads it: night below −6°, twilight to 6°, low sun to 20° |

Free drives (no scenario) are not tests of an encounter and are not counted. The domain is every combination of these
labels (6 × 4 × 13 × 5 × 4 = 6,240 cells) — or, when the workspace has an ODD (`odd.json`, or the Loop's;
[ODD monitoring](odd-monitoring.md)), its actor kinds, road types and speed bins, and the weather and light classes
whose representative conditions lie inside its weather and sun ranges. Runs outside the ODD are counted apart.

**What counts**: rare-case studies (`studies/*`: sweeps, searches, failure-rate estimates — a failure is the study's
objective ≥ its threshold, as the safety case's triggering conditions read it), stack suites (`suites/*`: a failed
pass criterion, as `core.stack` judges) and single scenario runs (`runs/*`: the scenario's pass criteria). Each cell
says how many runs came from each source.

## The rule

A cell has **enough** runs when the one-sided Clopper–Pearson upper bound on its failure rate, at the confidence
(95 %), is at most the target (5 % per encounter by default — the safety case's reporting threshold, not a validated
acceptance criterion; both are yours to set). Without a failure that is n ≥ ln(1 − c)/ln(1 − target) runs: 59 for
5 % at 95 % (the rule of three's exact form, ≈ 3/target), 299 for 1 %. With k failures, the n whose bound reaches the
target.

| status | when |
|---|---|
| untested | no run |
| too few | the upper bound is above the target — the runs it still needs, if none of them fail, are given |
| enough | the upper bound is at most the target |
| above the target | even the lower bound is above the target: the evidence says the cell fails more often than the target |

**The bound treats the runs as independent draws of the cell's encounters.** A search's runs are not: they are
concentrated where failures are likely, so its failures over-state the rate (pessimistic) and its passing runs are not
a random sample. A sweep's random runs are the draws the rule assumes. The sources are shown so the reading can be
made.

## The grid

Rare cases ▸ **Operating-domain coverage**: two dimensions at a time (rows and columns; the others summed over), each
entry with its distinct runs and failures, coloured by the rule on them (amber: too few, green: enough, red: above the
target), and how many of its cells are untested; a chosen entry's cells listed with their runs, failures, status,
upper bound, the runs still needed and the failure-rate estimates there. Tiles: the domain's cells, and how many have
enough, too few or no runs. *Target failure rate* and *Confidence* change the rule.

## Filling the gaps

*Fill the gaps* (and `falcon coverage fill`) takes the cells that need runs — too-few cells first, the largest
shortfall first; then untested cells, those one dimension away from a tested cell first (coverage grows outward from
what has been run) — and for each:

1. finds a scenario of the library or the project that can produce it: its actors include the cell's kind, its road
   requirement is the cell's road type, its ego speed can lie in the cell's band;
2. copies it as a project scenario (`cov-<base>-<hash>`, an undoable edit) with the cell's weather and light (a
   representative CARLA weather for each class, chosen so a run of it lands in its own cell — tested) and its
   ego-speed parameter narrowed to the band;
3. queues a rare-case search of it (`falcon search … --method bayes-pof`: it maps where the cell fails) — or, with
   *A sweep of random runs*, `falcon sweep` (the random draws the rule assumes) — with the runs the cell still needs
   (at least 8, at most 200) or the budget given.

Cells no scenario can produce (on the library: no scenario puts a cattle proxy at a junction, none needs a two-way road
outside a junction turn) are counted apart; add a project scenario for them.

## The command

```sh
falcon coverage show PROJECT [--rows weather|road|actor|speed|light] [--cols …] [--target 0.05] [--confidence 0.95] \
    [--cells] [--json]
falcon coverage fill PROJECT [--max 10] [--kind search|sweep] [--budget N] [--apply] [--run] [--world W] [--rig R] \
    [--seed N] [--threshold 0.5] [--target …] [--confidence …] [--json]
```

`fill` prints the plan; `--apply` adds the scenarios to the project (saved); `--run` then runs each study in turn. The
assistant: `analysis_outputs` with kind `coverage` (the grid's summary and the next cells to fill; with a cell id,
`weather/road/actor/speed/light`, that cell) and `jobs_search` with `scenario: "cell:<cell>"` (the cell's scenario,
added when the job is queued; with `method: "estimate"`, its failure rate).

## Limits

- The classes are heuristics on the recorded settings: the weather and light thresholds above, the road type from what
  the scenario required at its start. A run's place in the grid is its configuration, not what the run encountered
  along the way (the [runtime ODD monitor](odd-monitoring.md) measures that).
- A cell counts the runs placed in it; it says nothing about how the parameters within the cell were spread.
- The rule's bound assumes independent draws; searches' runs are not.
- The domain is a product of the labels: some cells may be impossible or meaningless (an autorickshaw at 80–130 km/h),
  and the counts include them unless an ODD narrows them.
