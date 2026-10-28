# Reports

The Reports page and `falcon report KIND SRC --out FILE` make a **self-contained HTML report** of a result: one file
with inline CSS and SVG — no script, font, image or style fetched from anywhere — that follows the reader's light or
dark setting, and carries the provenance of every number. Code: `falcon_autonomy/core/reports.py`.

A report computes nothing new: it shows what the source files say, as Falcon wrote them, and lists the SHA-256 of every
file it read, beside the source's own provenance block (the app version, the seed, the settings, the inputs' hashes).

## Kinds

| kind | source | what it shows |
|---|---|---|
| `iteration` | a loop iteration's folder (`workspace/loop/NNN`) or its `iteration.json` | the model and ODD with their hashes; the before and after with intervals and paired differences, and the verdicts; every step with its cost estimated before and measured after, its decisions and errors; the weak cells (the rule, the lowest cells with intervals); the rare-case searches (scenario, conditions, runs, failures, the objective stated); the generated data; the retraining or why it stopped; every step's outputs with their hashes |
| `evaluation` | an evaluation folder (`evaluation.json`) | the model, drives, frames, method (ignore rule, bootstrap unit); AP, AP50, AP75, AR100 with intervals; AP by class and by range with intervals; latency; the worst frames; the synthetic-model and few-drives warnings |
| `dataset` | a dataset spec's folder (`workspace/datasets/SPEC`, with `plan.json`) or a version file (`versions/vNNN.json`) | objects against the targets per class, per range bin and per cell; a version's splits and hash |
| `study` | a scenario study's folder (`workspace/studies/NAME`) | the scenario, world, weather and seed; each search's curve (best criticality so far) and failing cases; the runs ranked |
| `leaderboard` | a Leaderboard-style evaluation's folder (`leaderboard.json`) | driving score, route completion and infraction penalty with intervals over routes, by scenario, infractions per km, routes that could not run; what is not scored and why the numbers are not the Leaderboard's |
| `rootcause` | a root-cause attribution's folder (`rootcause.json`; `workspace/rootcause/…`, `falcon why`) | a failing case's factors — in the case, how each was removed, whether removing it alone fixes the failure and whether it alone still fails, the verdicts — with every re-run, the control and the honest wording; or an evaluation's correlates of misses (Spearman ρ with intervals, miss rate per bin); or a policy failure by its input groups ([root cause](root-cause.md)) |

## Charts

Dots with their 95 % intervals as lines (before and after: blue circles and orange squares; one series: blue), bars with
the target as a tick (coverage), a line of the best value so far (searches). The colours are checked for
colour-vision deficiency (the blue and orange pair: ΔE 25–27 under protanopia, light and dark), and the two series also
differ by shape and are named in a legend. Each mark has a tooltip with its value, and every chart has a table of the
same numbers.

## The page

*New report*: choose the kind and a source from the workspace (loop iterations, evaluations — the Models page's and
the loop's before and after —, dataset specs and versions, studies, Leaderboard results) and *Make the report*: a job
(`falcon report`) writes it into `workspace/reports/` with a time-stamped name. *Reports* lists them (newest first)
with their title, kind, source and the number of files read; double-click (or *Open in the browser*) opens one;
*Export to a folder…* copies the selected reports, which need nothing else. The Loop page's *Report…* makes the
current iteration's report.

## The command

```sh
falcon report iteration WORKSPACE/loop/001 --out loop-001.html
falcon report evaluation WORKSPACE/evaluations/20261028-101500-yolo --out eval.html
falcon report dataset WORKSPACE/datasets/spec-1/versions/v001.json --out data.html
falcon report study WORKSPACE/studies/cut-in-search --out study.html
falcon report leaderboard WORKSPACE/leaderboard/20261028-120000-bc --out lb.html
```

Exit codes: 0 written; 2 failed (a missing source, or one that is not of that kind — nothing is written).

## Format

The report embeds its meta as JSON in `<script type="application/json" id="falcon-report">` (data, not a script):
`schema` (`falcon.report/1`), `kind`, `title`, `source`, `created`, `app_version`, `inputs` ({path: SHA-256} of every
file read). `core.reports.read_meta(path)` reads it back; the page lists a folder's reports by it.

## Limits

- No images: the frames of the worst-case gallery are listed, not embedded (they would make the file large).
- A report is a snapshot: it is not updated when its source changes (make it again; the hashes say which files it read).
- Charts are static (tooltips only): no zoom or filtering.
