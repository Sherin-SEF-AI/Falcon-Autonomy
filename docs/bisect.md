# Regression bisect

"Which commit broke right turns at night?" Given a driving stack's git repository and two of its refs — one where the
stack was good, a later one where it is not — `falcon bisect run` runs the stack's [CI config](ci.md) at both, finds
the scenarios that got worse, then bisects the history between them through CI mode to the **first commit that broke
each one**. Code: `falcon_autonomy/core/regbisect.py` (its docstring is the reference); the pure bisection is
`Bisector`, tested with a fake evaluator.

```sh
falcon bisect run REPO --config falcon-ci.yml --good v1.4 --bad main [--out DIR] [--repeats N] [--alpha 0.05]
                  [--scenario ID]... [--build "CMD"] [--max-steps N] [--keep-checkouts] [--fake-carla DIR]
falcon bisect show OUT [--json]
```

Triage ▸ **Regression bisect** queues the same command as a job into `WORKSPACE/bisect/<time>-<good>-<bad>/` and lists
the answers; the assistant's `research_bisect` does too (with an approval card in Ask mode).

## What it does

1. **The endpoints.** The CI config is read *from each commit* (so the suite, the stack's launch files and its image
   build are that commit's) and run by CI mode with no baseline and no store. Per scenario, the finished runs give *k*
   passed of *n* at each end.
2. **Which scenarios regressed** — the stated rule:
   - **deterministic mode** (kinematic motion, no traffic, a stack not declared non-deterministic: a run repeats bit
     for bit, [CI mode](ci.md#determinism-and-flaky-runs)) — a scenario regressed when at least one of its runs (the
     same seed and parameter values: CI mode's pairing key) passed at the good end and fails at the bad end. One run
     per commit; `--repeats` above 1 is refused with a note (it would repeat each outcome).
   - **otherwise** — every run is repeated `--repeats` times (default 5), and **Fisher's exact test, one-sided**, on
     the 2 × 2 table (passed, failed) × (good, bad): the scenario regressed when the probability of a pass count at
     the bad end this low or lower, given both ends' totals, is below `--alpha` (0.05) and its pass rate fell. With *n*
     runs per end the smallest p reachable is 1 / C(2n, n): 0.5 for 1, 0.167 for 2, 0.05 for 3, 0.014 for 4 — fewer
     than four runs per end and scenario can never show a regression at 0.05, and the output says so for that
     scenario.
3. **The bisection.** The commits after the good ref up to the bad ref along the **first-parent chain** (`git rev-list
   --first-parent --ancestry-path`: a merge is one step, as `git bisect --first-parent`). Every regressed scenario
   keeps its own interval (last good, first bad); each step tests the commit nearest the middle of the widest
   interval, and that commit's runs answer for every scenario whose interval holds it — three scenarios broken at
   different commits cost fewer runs than three bisects. A commit is
   - **skipped** for every scenario, like `git bisect skip`, when its checkout or the `--build` command fails (exit 125
     or any other failure; the command runs in the checkout with `FALCON_BISECT_COMMIT` set), or when CI mode refuses
     its config or finishes no run of that scenario;
   - **bad** for a scenario — deterministic: when any run that regressed between the ends fails there; otherwise: when
     its pass rate is below the midpoint of the two ends' (exactly at the midpoint it is undecided and counts as
     skipped);
   - **good** otherwise.
4. **The answer**, per regressed scenario: *found* (the first bad commit with its author, date and subject, and the
   last good one), *ambiguous* (only skipped commits are left between the last good and the first bad — the answer is
   the list of them, git's "the first bad commit could be any of"), or *stopped* (`--max-steps` reached, with the
   commits still possible).

Each commit is checked out with `git worktree add --detach` into `OUT/checkouts/<sha>` (removed afterwards, and the
repository's worktree list pruned, unless `--keep-checkouts`); the repository's own checkout is never touched.
`{repo}` in the config's `subject.stack.command` becomes the checkout's path, so a command can start the stack from
the commit under test (`[python3, "{repo}/scripts/start_stack.py", "{port}"]`). A commit's records are kept in
`OUT/commits/<sha>/records.json` with its CI outputs in `run-<k>/`: running the same bisect again reads them back.

## Outputs

| file | contents |
|---|---|
| `bisect.json` | `falcon.bisect/1`: the repository, config, the good and bad commits, every commit between, the mode and the rule, per scenario both ends' counts with the test and its verdict, per regressed scenario its answer, every step (commit, verdicts, counts), the skipped commits with why, notes, provenance |
| `summary.md` | the same as a table and one line per scenario |
| `commits/<sha>/` | each commit's CI outputs (`run-<k>/`: [CI mode's outputs](ci.md#outputs)) and its records |

Exit codes: `0` done (a first bad commit found, ambiguous, or no regression to bisect), `2` failed (not a git
repository, the good ref not an ancestor of the bad one, an endpoint that cannot be evaluated, a config outside the
repository), `3` cancelled.

## Measured (tests)

`tests/test_bisect.py`, on the fake carla, CPU only: a throw-away repository whose stand-in ROS 2 stack (tests/data/
ci_stack.py, playing Falcon's bridge node) stops for a stalled car until the commit "raise the cruise speed", with a
commit that does not build (`--build "test ! -f BROKEN"`) before it — the stalled-vehicle scenario regressed (1 of 1
→ 0 of 1, deterministic), the unbuildable commit was skipped, the first bad commit found in 3 tested commits (about
5 s); a second run read every commit back. The pure bisection finds the first bad of up to 257 commits in at most
⌈log₂ n⌉ evaluations, three scenarios broken at commits 11, 39 and 40 of 64 in fewer evaluations than three bisects,
and returns the candidates when skips surround the change. A noisy stack (a right turn passing 9 of 10 before the
change, 2 of 10 after; five repeats) over a real 12-commit history: Fisher's p below 0.05, first bad commit found.

## Limits

- **One change per scenario** between the ends is assumed. A scenario that breaks, is fixed and breaks again, or a
  flaky run in a mode taken as deterministic, can mislead it; when a test contradicts the interval it is noted.
- **Noise.** The midpoint rule can misclassify a commit when its runs are few; more repeats, fewer mistakes. Each
  tested commit's pass counts are in `bisect.json`.
- **Cost.** Every tested commit is a full CI run of the suite (all scenarios, not only the regressed ones), times the
  repeats: about log₂(commits) runs per regressed scenario, plus the two ends.
- **First-parent only.** A commit on a merged branch is not tested on its own; the merge that brought it is.
- **Simulation evidence** within the suite's scenarios and seeds, as [CI mode](ci.md#limits) says.
