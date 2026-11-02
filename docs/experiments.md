# Experiments, ablations and outputs for papers

An **ablation** starts from a hypothesis you state — *the fine-tuned detector raises night recall* — and the factors it
depends on — model × weather × light, each with its levels. Falcon expands it into a grid of runs with seeds, works
out how many replicates the comparison needs (a power analysis), runs them through the same jobs the pages use,
analyses them with paired bootstrap intervals and a correction for several comparisons, and says whether the
hypothesis is supported. Every experiment is in a **registry**, every run is **pinned** (Falcon's version, the CARLA
build, the world, rig and weights hashes, the seeds), and the whole experiment re-runs with one command and is checked
against the original. Any comparison can then be written out **for a paper**: LaTeX and CSV tables and PDF, PNG and SVG
figures with their intervals, byte-identical for the same input; every dataset export carries a **data card**.

Code: `falcon_autonomy/core/ablation.py` (the experiment), `core/metrics/uncertainty.py` (the statistics),
`core/provenance.py` (pins), `core/reproduce.py` (the re-run check), `core/paper.py` (tables, figures, data cards),
`ui/pages/ablations_tab.py` (Training ▸ Ablations).

## Stating the hypothesis

A spec is a JSON object (`falcon experiment ablation example` prints one):

```json
{"id": "night-recall",
 "hypothesis": {"text": "the fine-tuned detector raises night recall", "direction": "increase"},
 "metric": "AR100",
 "factors": [{"name": "model", "kind": "model", "levels": ["base", "finetuned"]},
             {"name": "weather", "kind": "weather", "levels": ["ClearNoon", "HardRainNoon"]},
             {"name": "light", "kind": "weather",
              "levels": {"day": {"sun_altitude_angle": 45}, "night": {"sun_altitude_angle": -90}}}],
 "contrasts": [{"factor": "model", "a": "base", "b": "finetuned", "where": {"light": "night"}, "by": ["weather"]}],
 "design": {"effect": 0.05, "sd": 0.06, "alpha": 0.05, "power": 0.8},
 "run": {"executor": "detection", "seconds": 10, "fps": 10}}
```

| field | meaning |
|---|---|
| `hypothesis` | `text`; `direction` of the metric, B against A: `increase`, `decrease` or `change`; `rule` over the contrasts: `all` (default) or `any` |
| `metric` | what one run measures: for the detection executor a number of the evaluation's table — `AP`, `AP50`, `AP75`, `AR100`, `AP.<class>`, `AP.range_m[…)` ([models](models.md)) |
| `factors` | `name`, `kind` and `levels` (a list, or `{level: value}`). Kinds: `model` (registered model ids — every model is evaluated on the same drive), `weather` (a CARLA preset, or weather fields over the world's weather; several weather factors apply in order), `world`, `rig`, `traffic` (project ids), `value` (any levels, for the simulated executor and executors given from Python) |
| `contrasts` | B against A of one factor, at the levels `where` fixes, split into one contrast per level of the `by` factors; an `interaction` contrast is that difference's difference between two levels of another factor. Default: the first two levels of the first factor that has two |
| `design` | the effect to detect: `effect` (in the metric's units) and `sd` (the spread of the paired differences, e.g. from a pilot), or `dz` directly; `alpha`, `power`; or `replicates` given outright; `min_replicates` (2), `max_replicates` (50); the first `seed` |
| `analysis` | `n_boot` (at least, default 2000), `boot_seed`, `ci` (0.95), `correction` (`holm`, `bonferroni`, `none`), `interval` (`expanded`, `percentile`), `cluster` (`unit`, or `replicate` to resample whole replicates) |
| `run` | `executor` (`detection` or `simulated`); `world`, `rig`, `traffic` when no factor sets them (default the project's first); `seconds` and `fps` per drive; `every` (evaluate every n-th frame), `camera`; `model` when no factor is a model; `simulate` (below) |

## The plan and the sample size

A contrast's **unit** is everything it does not name: the levels of the other factors and the replicate. Each unit
gives one paired value — B's run minus A's (an interaction pairs four runs). With the example above, each of the two
contrasts (one per weather) has one unit per replicate.

The comparison is planned as a **two-sided paired t-test** of those differences. The standardised effect is
dz = effect / sd = 0.05 / 0.06 = 0.83. With m contrasts each is planned at alpha / m (Bonferroni — Holm's first step
is the same, its later steps are kinder), here 0.025. The number of pairs is the smallest n whose **exact** power —
the noncentral t with n − 1 degrees of freedom and noncentrality dz √n (Cohen 1988, §2.3.5), computed by numerical
integration, not by the normal approximation — reaches the target: 17 pairs, power 82.1 %. The replicates per cell
are those pairs over the units one replicate gives (rounded up, the largest over the contrasts), within
`[min_replicates, max_replicates]`; a cap is said, with the power it leaves. So the example is 8 cells × 17
replicates = 136 runs: 68 drives (one per weather × light × replicate) and 136 evaluations.

Every cell of a replicate shares its seed, so cells that differ only in one factor drive the same traffic; the model
levels share the drive itself.

## Running it

- **detection**: every drive cell (the factors other than the model) is a recorded drive — `core.drive.drive_config`
  with the cell's world, rig, traffic and weather, submitted to the job queue (under the machine-wide CARLA lock, one
  at a time, as the Record page's) into `experiments/<id>/drives/` — then each model level is evaluated on it
  (`core.harness.evaluate`, as the Models page; a non-plug-in model takes the GPU lock first). Each run is a tracked
  run of the experiment (`experiments/<id>/runs/<name>/`: params, tags, the metric, its evaluation, run.json with its
  pins). On the fake simulator, 4 drives of 1 s and 8 evaluations took 6 s on the reference machine under load
  (measured 2026-11-02).
- **simulated**: each run's value is drawn from a stated model — `base`, plus every effect whose `when` matches the
  cell (`{"when": {"model": "ft", "light": "night"}, "add": 0.05}`), plus a replicate effect N(0, `unit_sd`) shared by
  every cell of the replicate, plus noise N(0, `noise_sd`) per run; seeded. It is a simulation of the design — to
  check a plan and the analysis before spending drives — and its results say SIMULATED.
- A Python caller (a tool, a test, a research loop) can pass any executor to `run()`.

A finished run is never repeated: run it again and it resumes. The first failure stops the experiment (its run says
why); running it again resumes from there.

## The analysis

For each contrast, the units' A and B values are compared by `core.metrics.paired_bootstrap` — both resampled
together, so what a unit shares cancels — with the **expanded percentile interval** (Hesterberg 2015): the quantiles
taken at Φ(± √(n/(n−1)) · t₍ₙ₋₁₎) instead of the nominal ones, which removes most of the plain percentile interval's
under-coverage with few units. Each contrast reports:

- A's and B's means, the difference B − A with its 95 % interval, and its **simultaneous** interval at
  1 − 0.05 / m (Bonferroni);
- the standardised effect dz with a bootstrap interval;
- a two-sided p-value that inverts the interval, and the **Holm-adjusted** p-value over the m contrasts (Holm 1979:
  the family-wise error stays at alpha whatever the dependence between contrasts);
- the verdict: **supported** when the adjusted p is below alpha in the stated direction, **contradicted** when it is
  below alpha the other way, otherwise **no detectable effect** — never "no effect" — with the power the experiment
  had for the planned effect at the spread it observed.

The hypothesis is supported when every contrast supports it (rule `all`) or at least one does and none contradicts it
(`any`); contradicted when any contrast contradicts it; otherwise not supported. Each cell's mean over its replicates
is reported with its interval too.

Resamples: at least `n_boot`, and as many more as the simultaneous interval's tail needs to rest on about 20 resamples
(at most 50,000). A bootstrap p-value cannot fall below the one half a resample gives; when every resample went one
way and that floor is what stopped a verdict, the result says so.

### How the statistics were checked

`tests/test_ablation.py`, measured 2026-11-02:

- **Sample sizes** match Cohen's tables and G*Power for the paired t-test at alpha 0.05, power 0.8: dz 0.5 → 34 pairs
  (power 0.808), 0.8 → 15 (0.821), 0.2 → 199 (0.802). The t quantiles match Student's table (2.2281 at 10 df).
- **The exact power** against 20,000 simulated t-tests: 0.8086 simulated against 0.8078 computed (n 34, dz 0.5),
  0.6162 against 0.6162 (10, 0.8), 0.6584 against 0.6550 (6, 1.2).
- **Coverage** of a known effect by the expanded paired interval (400 simulated data sets each): 0.943 with 8 units,
  0.945 with 12, 0.953 with 20 — the plain percentile interval: 0.880, 0.890, 0.923.
- **Power of the bootstrap test** at the planned size: 0.818 rejections against 0.821 planned (15 pairs, dz 0.8), 0.803
  against 0.808 (34, 0.5); under no effect 0.07 (alpha 0.05; 400 data sets, standard error 0.011).

## Pins and re-runs

Each experiment's `plan.json` holds its pins (`core.provenance.pins`): Falcon's version, the hash of each world (its
definition, and its OpenDRIVE file when imported), rig, traffic profile and model's weights (the file's SHA-256, or a
plug-in's registration with its parameters), the seeds. Each run adds its own: the CARLA build its drive's worker read
from the server, the drive's key (its config without where it was written) and the evaluated model's weights.

`falcon experiment ablation rerun ID --project P` re-runs the experiment from its own definition and seeds into
`experiments/<id>-rerun<k>` and compares it with the original (`core.reproduce.compare_experiments`): the same planned
runs and seeds, each run's metric, each contrast's difference and intervals, the verdicts — and lists every pin that
changed (`pins_changed`). The default tolerance asks for the same numbers: the fake simulator and the simulated
executor reproduce them exactly; on a real server images can differ slightly ([methods](methods.md#reproduction)), so
give the tolerance the comparison needs (`tol` in the API).

## The registry

Every experiment is a folder `experiments/<id>/` in the workspace — the files are the truth — and a row of the
catalogue (kind `experiment`, re-indexed by `falcon runs PROJECT --rebuild`). An experiment's definition never changes
under it: registering another definition under a used id is refused. The registry lists every experiment of every
kind (ablations, mixed sim/real, active learning, translation) with its status (planned, running, failed, cancelled,
complete, analysed) and, for an ablation, its hypothesis, factors, metric and verdict; a re-run names its original.

```
experiments/<id>/
  experiment.json   the definition (the normalised spec) and its SHA-256
  plan.json         the cells, contrasts, replicates and seeds, the runs, the power analysis, the pins
  runs/<name>/      a tracked run: params.json, tags.json, metrics.jsonl, run.json (status, pins), evaluation/
  drives/<drive>/   the detection executor's recorded drives
  results.json      the analysis (falcon.ablation_results/1)
  rerun.json        in a re-run: the check against the original
```

## The page and the assistant

**Training ▸ Ablations**: the hypothesis, direction, metric, the effect to detect and its spread, alpha, power, the
executor and the drive length; the factors and contrasts as JSON (*Example* fills one from the project's models).
*Plan* shows the grid and the sample size without writing anything; *Register* writes it. The **Registry** lists
every experiment; select one to *Run or resume*, *Analyse*, *Re-run and check* (jobs, except Analyse) or *Export for a
paper*. The **Result** shows each contrast with its interval drawn on a shared axis, its adjusted p and its verdict.

The assistant's **research** tools: `research_ablation` with `action` create (from a spec; the approval card shows
the plan), run and rerun (jobs), analyse, export; `research_registry` lists the registry and, with an id, reads an
experiment's plan or results; `research_paper` writes the outputs for a paper of an experiment, an evaluation or a
comparison, or an export's data card. `analysis_outputs` with kind `experiment` lists the same rows.

## The command

```sh
falcon experiment ablation example > night.json            # edit it
falcon experiment ablation plan night.json --project p.falcon.json
falcon experiment ablation new night.json --project p.falcon.json
falcon experiment ablation run night-recall --project p.falcon.json      # resumable; then analysed
falcon experiment ablation show night-recall --project p.falcon.json
falcon experiment ablation rerun night-recall --project p.falcon.json    # exit 2 when the re-run differs
falcon experiment ablation list --project p.falcon.json
falcon paper experiment WORKSPACE/experiments/night-recall --out paper/night-recall
```

## The core API

For tools and a research loop (`core.ablation`); every function raises `AblationError` with what is wrong:

| call | does |
|---|---|
| `example_spec() -> dict` | the example above |
| `normalise(spec, project=None) -> dict` | the spec with its defaults, checked (against the project when given); idempotent |
| `plan(spec) -> dict` | the grid, contrasts, replicates, seeds, runs, counts and `power` (`power_plan`) of a normalised spec; `describe_plan(plan)` as text |
| `create(workspace, spec, project=None) -> {'id', 'path', 'plan'}` | register it (plan and pins written, catalogue indexed) |
| `run(workspace, id, project=None, service=None, executor=None, emitter=None, cancel=None) -> results` | run or resume, then analyse; raises `Cancelled` |
| `analyse(workspace, id) -> results` | results.json (recomputed only when a run's value changed) |
| `table(results) -> str` | the results as text |
| `rerun(workspace, id, project=None, service=None, executor=None, emitter=None, cancel=None, tol=None) -> dict` | the re-run and its check |
| `registry(workspace) -> [row]`, `summary(exp_dir) -> row`, `status(exp_dir)`, `load(workspace, id)` | the registry |

## Outputs for papers

`falcon paper KIND SOURCE --out DIR` (Reports ▸ *Outputs for a paper*, Data ▸ Compare ▸ *Export for a paper*, the
Ablations tab's *Export for a paper*):

| kind | source | tables | figures |
|---|---|---|---|
| `experiment` | an ablation | `contrasts` (n, A, B, Δ with its 95 % and simultaneous intervals, dz with its interval, adjusted p, verdict), `cells` (each cell's mean with its interval) | `contrasts` (each Δ: thick 95 %, thin simultaneous, a dashed line at 0), `cells` (the means by the contrast's factor) |
| `experiment` | a mixed sim/real experiment | `arms` (AP with its interval, seed spread, Δ from real only with its interval, verdict) | `arms` |
| `evaluation` | an evaluation folder | `scores` (every number with its bootstrap interval) | headline and per-class numbers |
| `comparison` | a comparison folder (`falcon evaluate --compare`) | `comparison` (A, B, B − A with its paired interval, P(B > A)) | each difference with a line at 0 |
| `runs` | two to four run folders | `runs` (what each recorded, as Data ▸ Compare) | none: single runs carry no interval |
| `datacard` | an export folder | the data card (below) | — |

Each table is a `.tex` file (a `table` environment with booktabs rules; `\usepackage{booktabs}`; its header comment
names the source and the hash of its inputs) and a `.csv` (an interval as three columns, value, `_ci_low`,
`_ci_high`, at six decimals). Figures are dot-and-whisker plots in `.pdf` (vector, Helvetica, no embedded font), `.svg`
(each mark with its value as a tooltip) and `.png` (3 px per point). Up to four series differ by colour and by shape
(blue circles, orange squares, green diamonds, purple triangles), named in a legend. `paper.json` lists the inputs and
every file with its SHA-256, and each table's and figure's caption.

**Deterministic**: the same source gives byte-identical files — fixed decimals, no time or machine recorded, a PDF
without a creation date — so the outputs can live in a paper's repository and be diffed. Labels from the dashcam
footage are never written: a run from it is "run k".

## Data cards

Every export (`falcon export`, Data ▸ Export) writes `DATACARD.md` and `datacard.json` beside `falcon_export.json`;
`falcon paper datacard EXPORT` writes them again. After Datasheets for Datasets (Gebru et al. 2021), the sections that
apply to a rendered set:

- **Composition**: runs, splits, frames written per split, the export's counts, objects per class (counted in the
  runs' object files, every camera and step — possibly more than the frames the export kept);
- **Collection process**: the CARLA build each run's worker read, Falcon's versions, the workers, the rigs and their
  sensors, synchronous recording;
- **Labels**: the simulator's ground truth, not human annotation; the format and how to load it;
- **Conditions covered**: runs and frames per world × weather × time of day;
- **Known gaps**: times of day and rain absent, classes with fewer than 100 objects, what the export left out and
  what is heuristic, few runs, the sim-to-real gap;
- **Licence**: CARLA's terms as CARLA states them (its code MIT, its own assets CC-BY), that Falcon claims nothing over
  the data, the publisher's choice to state; translated runs inherit their real data's terms;
- **Provenance**: the export manifest's SHA-256 and every run's run.json and config hashes.

A run from the user's own dashcam footage (the assistant's privacy rule: a drive along a dashcam route, frames from
clips, a dashcam set) is listed as "run k" with its world withheld, and the card then carries no date.

## Limits

- The sample size is the paired t-test's; the analysis is a bootstrap. They agree for moderate numbers of units (the
  validation above); with fewer than about 10 units per contrast the bootstrap has few distinct resamples and the
  result says so.
- Units that share a seed but differ in another factor's level are treated as independent unless `cluster` is
  `replicate`; the plan counts units as pairs either way, so with `replicate` it is optimistic.
- Contrasts are differences of means of one metric per run; a per-frame metric pooled over a run (AP) is computed per
  run first, so every run weighs the same.
- The detection executor runs model, weather, world, rig and traffic factors; other factors (scenario parameters,
  sensor settings) need an executor from Python.
- Run comparisons have tables only. PNG text uses OpenCV's Hershey font; the PDF and SVG are the ones for print.
