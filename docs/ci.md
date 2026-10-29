# CI mode

`falcon ci run CONFIG` runs a driving stack's scenario suite headless — or a Falcon policy's routes, or a perception
model's evaluation — judges it with **gates**, and writes what CI systems read: JUnit XML, a Markdown summary, the
HTML report and a results JSON with the commit under test. Run on every commit, it guards a stack's codebase the way
unit tests guard a library: a change that makes the stack collide where it used to stop fails the build, with the
evidence attached.

It is the same code as the rest of Falcon: suites are `falcon.suite/1` suites run by the suite runner
([stack testing](stack-testing.md)), a policy is scored as the Leaderboard-style evaluation does it
([policies](policies.md)), a model through the test harness ([models](models.md)), traffic rules by the temporal-logic
checker ([traffic rules](traffic-rules.md)). CI mode adds the config, the gates, the comparison with a baseline, the
outputs and the GitHub templates (`core/ci.py`, `core/ci_github.py`).

## Quick start

```sh
falcon ci init path/to/my-stack --github     # falcon-ci.yml, falcon-suite.json, .github/workflows/falcon.yml
falcon ci validate path/to/my-stack/falcon-ci.yml
falcon ci run path/to/my-stack/falcon-ci.yml --fake-carla FALCON/tests/fake_carla --no-baseline   # a smoke run
falcon ci run path/to/my-stack/falcon-ci.yml                                       # on Falcon's own CARLA server
```

`--fake-carla` runs against the fake carla package in Falcon's tests: no server, no GPU, seconds per run. It checks
that the stack starts, the bridge connects, controls come back in time and the gates work; its numbers say nothing
about how the stack drives. Without it, Falcon starts its own CARLA 0.9.16 server (as every Falcon command does),
runs the suite and stops the server.

## Commands

| command | what it does |
|---|---|
| `falcon ci run CONFIG [--out DIR] [--mode falcon\|fake] [--fake-carla DIR] [--baseline DIR \| --no-baseline] [--force] [--keep-server] [--quiet]` | runs the config; the options override the config's `out`, `carla` and `baseline` |
| `falcon ci validate CONFIG` | checks a config against its schema and what the schema cannot say (files, combinations) |
| `falcon ci schema` | prints the `falcon.ci/1` JSON Schema (for an editor's validation) |
| `falcon ci init [TARGET] [--kind stack\|policy\|model] [--github] [--config NAME] [--falcon-ref REF] [--force]` | writes a commented config (and, for a stack, a suite) into a repository; `--github` adds `.github/workflows/falcon.yml`. Files that exist are kept unless `--force` |
| `falcon ci summary RESULTS` | prints a run's Markdown summary |
| `falcon ci lint WORKFLOW` | structural checks of a GitHub Actions workflow (and actionlint's, when it is installed) |
| `falcon ci post RESULTS [--repo O/N] [--pr N] [--sha SHA] [--as comment\|check\|status] [--via gh\|api] [--post]` | posts the summary to GitHub — a dry run that sends nothing unless `--post` ([posting](#posting-to-github)) |

## Exit codes

| code | `falcon ci run` |
|---|---|
| `0` | every gate passed |
| `1` | a gate failed: the suite ran and the evidence says no |
| `2` | an error, so no verdict: a config that does not validate (the problems on stderr, nothing written), no server, a run that did not finish, a gate that could not be judged (a latency budget with nothing measured, a `demonstrate` rule with no interval), a crash (the traceback in `ci-results.json`) |
| `3` | cancelled: SIGINT or SIGTERM (what a CI system sends when a job is cancelled); the outputs so far are written |

This is a test runner's convention (pytest's, JUnit runners'), so a CI system treats a failed gate as a failed build
and anything else as broken infrastructure. It differs from the worker codes of the other commands
([CLI](cli.md#exit-codes)), where `1` means a crash; the other `falcon ci` sub-commands keep those (`0` ok, `2`
failed or a usage error). Outputs are written for codes 1–3 whenever the output folder was known.

## The config (`falcon.ci/1`)

YAML (needs PyYAML) or JSON. Relative paths are relative to the config's folder; `${VAR}` and `~` are expanded, and
a path whose variable is not set counts as not configured (so a baseline store set only on the self-hosted runner
is simply absent on a laptop). The full schema: `falcon ci schema`.

```yaml
schema: falcon.ci/1
name: aeb                          # names the outputs, the JUnit suites and the PR comment
project: sim/project.falcon.json   # optional for a stack or a policy (a default project is made); needed for a model
out: falcon-ci-out                 # default: falcon-ci-out beside the config; --out overrides
seed: 0                            # the bootstrap's seed (every interval is reproducible)
subject:
  kind: stack                      # stack | policy | model
  repo: .                          # the git repository whose commit is under test
  suite: falcon-suite.json         # a falcon.suite/1 file, or the suite itself inline
  stack:                           # overrides the suite's stack
    driver: ros2                   # ros2 | follower | autopilot (the baselines of stack testing)
    image: ${STACK_IMAGE}          # your stack's Docker image (variables expanded)
    image_args: [ros2, launch, my_stack, sim.launch.py]
    image_env: {RMW_IMPLEMENTATION: rmw_fastrtps_cpp}
    command: [[./scripts/start_stack.sh, "{port}"]]   # and/or commands that start it
    bridge_node: image             # image | native | none (your command starts the bridge node itself)
    bridge: {control: autoware, deadline_s: 0.5}      # the bridge section (docs/stack-testing.md)
    test_stack: false              # true: Falcon's own test follower instead (needs the bridge image)
    domain_id: 0
    deterministic: true            # false: your stack is not repeatable given its inputs (allows retries)
  motion: kinematic                # overrides the suite's: kinematic | physics
  traffic_vehicles: 0
  # world, rig, fps, criteria, rules: override the suite's too
carla:
  mode: falcon                     # falcon: Falcon's own server (CarlaService); fake: the fake carla
  fake_carla: ''                   # mode fake: the package (default: tests/fake_carla beside Falcon's sources)
  keep_server: false
baseline:
  kind: store                      # none | dir | store | model
  dir: ''                          # kind dir: a previous run's output, or a suite / Leaderboard / evaluation folder
  store: ${FALCON_CI_STORE}        # kind store: the store's folder …
  branch: main                     # … and the branch whose stored results are the baseline
  model: ''                        # kind model (a model subject): another registered model, on the same drives, now
store:                             # keep this run's results as its branch's baseline
  dir: ${FALCON_CI_STORE}
  branches: [main]
  only_if_passed: true
gates:
  pass_rate: {min: 0.5, rule: demonstrate, per_scenario: false}
  metrics:
    - {metric: collision_rate, max: 0.1, rule: refute}
  regressions: {metrics: [pass_rate, collision_rate, criticality, min_ttc_s], allow: [], n_boot: 1000,
                on_missing_baseline: skip}
  rule_violations: {max_per_km: 5.0, rule: estimate, rules: {india.keep_left: 2.0}}
  latency: {p95_ms: 200, p99_ms: 400}
  max_missed_deadlines: 0
flaky: {retries: 0, count_as: pass}
outputs: {html: true, github_step_summary: auto}
```

### What is tested (`subject`)

- **A stack** (`kind: stack`): a suite — scenarios × seeds with pass/fail criteria ([suites](stack-testing.md#suites))
  — run into `OUT/suite` by the suite runner. A ROS 2 stack is started for every run by launch commands Falcon builds
  from `stack`: the bridge node (in the `falcon-ros2-bridge:humble` image, natively with `ROS_DOMAIN_ID` set, or not
  at all), then your stack — `docker run --rm --name falcon-<run>-stack --network host --ipc host -e
  ROS_DOMAIN_ID=<d> IMAGE ARGS…` for an image (the container is removed with the run), and each `command` as given.
  `{port}`, `{domain}`, `{name}` and `{run}` are filled in. A suite's own `launch` is used when `stack` gives none.
  The baselines `follower` and `autopilot` need nothing started. The output folder is always fresh, so a suite never
  resumes runs made by another commit.
- **A policy** (`kind: policy`): a Falcon policy folder driven over Leaderboard-style routes (`routes`: scenarios
  (default `lead_brake`), seeds per scenario, free routes, seed) into `OUT/leaderboard`. A route **passes** when its
  route completion reaches `route_pass.min_route_completion` (default 0.9) with at most
  `route_pass.max_penalised_infractions` (default 0) of the infractions the Leaderboard penalises. `max_episode_s`,
  `world`, `rig`, `fps`, `motion`, `traffic_vehicles`, `device` as for `falcon policy leaderboard`.
- **A model** (`kind: model`): a registered model scored by the test harness into `OUT/evaluation`, on pinned drives
  (`drives`: run folders, globs allowed) or on drives recorded now (`record: {seeds, seconds, fps, world, rig,
  traffic_vehicles}` into `OUT/drives/drive-<world>-seed<N>`: stable names, so frames keep their ids). `camera`,
  `every`, `n_boot` as for `falcon evaluate`. A model has no pass/fail runs: its drives appear as informational
  testcases, and `metrics` gates judge its numbers.

### Gates

Each gate becomes a JUnit testcase and a row of the summary, with what it measured, its limit and the rule that
judged it. A gate **fails** when the evidence says no, is an **error** when it cannot be judged, and is **skipped**
only where documented (a regression gate with no baseline, unless `on_missing_baseline: error`).

| gate | measures | for |
|---|---|---|
| `pass_rate` | passed / finished runs (routes), with Wilson's 95 % score interval; `per_scenario: true` also judges each scenario | stack, policy |
| `metrics` | any number of the run's summary, with its interval: `pass_rate`, `collision_rate` (Wilson), `criticality` and `min_ttc_s` (mean, bootstrap over runs), `missed_deadlines`, `rule_violations_per_km` (ratio of sums, bootstrap); a policy's `driving_score`, `route_completion`, `infraction_penalty`; a model's every metric of its evaluation (`AP`, `AP50`, `AP.car`, `mIoU` …, bootstrap over drives) | all |
| `regressions` | none of the listed metrics regressed against the baseline (below) | all |
| `rule_violations` | traffic-rule violations per km over the runs (core.rules; the suite's `rules`, by default by handedness), and per rule for a stack | stack, policy |
| `latency` | percentile budgets (`p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`, `max_ms`, `mean_ms`): a ROS 2 stack's control wait through the bridge (every step of every run, from `frames.jsonl`), a policy's decision time (`act()`), a model's inference per frame (`p50`, `p90`, `p99`, `mean`) — this machine's, under its load | all |
| `max_missed_deadlines` | missed control deadlines summed over the runs | stack |

**How an interval judges a limit** (`rule`):

- `demonstrate` — passes only when the whole 95 % interval is on the good side of the limit (for a minimum: the
  lower bound reaches it). It demands evidence, and so runs: with every run passing, Wilson's lower bound reaches 0.5
  after 4 runs, 0.8 after 16, 0.9 after 35, 0.95 after 76 and 0.99 after 382. Six of six passing give [0.61, 1.00].
  A failing gate says how many runs it would need. The default for `pass_rate`.
- `refute` — fails only when the whole interval is on the bad side (a shortfall shown at 95 %). It rarely fails by
  chance, and with few runs it catches only gross failures (0 of 6 passing: [0.00, 0.39]).
- `estimate` — compares the point estimate. The default for `metrics` and `rule_violations`.

A `demonstrate` or `refute` rule on a number without an interval (one run) is an error, not a pass.

### Regressions against a baseline

The **baseline** is a previous run's results: a `falcon ci run` output folder (`dir`; also a CI artifact you
downloaded), a bare suite, Leaderboard or evaluation folder, the **store**'s folder for a branch (`store` +
`branch`: what earlier runs on `main` kept), or — for a model — another registered model evaluated on the same
drives in this run (`model`).

- **Stacks and policies:** runs are paired by their scenario, seed and parameter values (the same scenario instance
  before and after), and each metric's change now − baseline gets a **paired bootstrap** interval with the pair as the
  unit (`core.metrics.paired_bootstrap`, `n_boot` resamples, seeded). A **regression** is a change whose interval lies
  wholly on the metric's bad side (lower pass rate, higher collision rate, …); an **improvement** wholly on its good
  side; anything else is "no detectable change at 95 %", not "no change". The comparison also lists the runs that
  passed before and fail now. Runs only in one of the two are listed and not compared; with no run in common the
  gate counts as having no baseline. Fewer than 5 pairs: the interval is flagged as not reliable.
- **Models:** `falcon evaluate --compare`'s paired bootstrap over drives on the frames both evaluations scored —
  refused when a shared frame's ground truth differs (drives recorded again on a real server are not the same drives:
  pin them). By default the headline metrics (`AP`, `AP50`, `AP75`, `AR100`, `mIoU`) are gated (`metrics: all` for
  every breakdown — more metrics, more chance findings).

`allow` lists metrics whose regression is accepted (said in the gate's detail).

**The store.** With `store` set, a run whose branch (from the CI system, else git) is in `branches` — and that
passed, unless `only_if_passed: false` — copies `ci-results.json`, `ci-runs.jsonl`, `summary.md`, the resolved config
and a model's `evaluation/` to `STORE/<branch>/`, replacing the previous ones atomically. The next run on any branch
compares against it with `baseline: {kind: store, branch: main}`. A store is a folder: on a self-hosted runner, a
path on its disk (`FALCON_CI_STORE`); for GitHub-hosted runners, an artifact you download into place.

### Determinism and flaky runs

Kinematic motion with no background traffic repeats bit for bit (measured: [stack testing](stack-testing.md#what-a-run-records));
physics motion, CARLA's traffic manager and its autopilot do not. The config therefore defaults to `motion:
kinematic` and `traffic_vehicles: 0`, every output says whether the run was deterministic and why not, and:

- **retries are refused in a deterministic mode** (`flaky.retries` with kinematic, no traffic: exit 2) — a retry
  would repeat the failure exactly;
- in a non-deterministic mode (physics, traffic, `autopilot`, or `stack.deterministic: false` for a stack that is not
  repeatable given its inputs — a ROS 2 stack's own threads and timers can make it so), a failed or unfinished run is
  run again up to `retries` times (at most 5) into `OUT/retries/`. A run that fails and then passes is **flaky**:
  every attempt is recorded (`attempts` in `ci-runs.jsonl`, the JUnit testcase's output, the summary), and
  `count_as: pass` (default) counts it as passed with the passing attempt's numbers, `count_as: fail` keeps the
  failure. Counting flaky runs as passes raises the pass rate above what a single attempt shows; the count is in the
  summary.

A model subject's evaluation is deterministic; drives recorded in the run are not repeatable on a real server, which
is why regressions across commits need pinned drives.

## Outputs

In the output folder:

| file | contents |
|---|---|
| `ci-results.json` | `falcon.ci_results/1`: the verdict and exit code, the subject, the commit, CARLA's mode, determinism, every run, the summary with intervals (overall and per scenario), the gates, the comparison and its baseline, the latency, the rule aggregate, the store, errors and notes, provenance |
| `ci-runs.jsonl` | one `falcon.ci_run/1` line per run: its pairing key, scenario, seed, values, status, pass, failed checks, metrics, folder, attempts — what a later run pairs against |
| `junit.xml` | two `<testsuite>`s: the runs (one `<testcase>` per scenario run: `<failure>` with the failed checks, `<error>` when it did not finish; a model's drives informational) and the gates (`<failure>`, `<error>`, `<skipped>`), with the commit, branch and CARLA mode as properties |
| `summary.md` | the verdict, the gates table, pass rates per scenario, the comparison, failing and flaky runs, latency, provenance — appended to `$GITHUB_STEP_SUMMARY` when that is set (`outputs.github_step_summary: auto`), and the body of a PR comment |
| `report.html` | the HTML report (`falcon report ci OUT`, [reports](reports.md)): gates, pass rates with Wilson whiskers, the comparison, runs, latency, limits, provenance |
| `run.json`, `ci-config.json` | `falcon.run/1` (worker `ci`, the outputs' hashes) and the resolved config |
| `suite/`, `leaderboard/`, `evaluation/`, `drives/`, `retries/` | the runs themselves, as the other commands write them (every stack run replayable with `falcon stack replay`) |

**The commit under test** is read from the subject's `repo` with git (HEAD's SHA, branch, whether tracked files have
uncommitted changes, the origin remote with any credential removed) and from the CI system's variables: GitHub
Actions (`GITHUB_SHA`, `GITHUB_REF`, `GITHUB_HEAD_REF`/`GITHUB_REF_NAME`, the run's URL), GitLab CI (`CI_COMMIT_SHA`,
`CI_COMMIT_REF_NAME`, `CI_MERGE_REQUEST_IID`, `CI_JOB_URL`) and Jenkins (`GIT_COMMIT`, `BRANCH_NAME`, `CHANGE_ID`,
`BUILD_URL`). Git's HEAD wins for the SHA (it is what was checked out); the CI system's branch wins (CI checkouts are
detached). The provenance block hashes the config, the suite, the policy and the project.

The Stack testing page lists the CI runs in a project's workspace (`falcon ci run … --out WORKSPACE/ci/NAME`) with
their verdicts and failed gates, opens their reports, and writes a config or the workflow into a folder you choose.
It never posts.

## GitHub Actions

`falcon ci init TARGET --github` writes `.github/workflows/falcon.yml` into the repository you name (and the config
and suite if they are not there). It has two jobs:

- **`smoke`** on a GitHub-hosted runner (`ubuntu-24.04`): checks out your code and Falcon Autonomy
  (`Sherin-SEF-AI/Falcon-Autonomy` at `FALCON_REF`, default the tag of this release — the tag must exist on GitHub,
  or set a branch), installs Falcon with pip (plus the GL libraries OpenCV needs, PyYAML, and the CPU build of torch
  for a policy or onnxruntime for a model), builds your stack's image and Falcon's bridge image for a ROS 2 stack,
  and runs the config against the fake carla with `--no-baseline`. Minutes, no GPU: it catches a stack that no longer
  starts, connects or answers in time.
- **`carla`** on your **self-hosted** runner, after `smoke`: `falcon doctor`, the images, then `falcon ci run` on
  Falcon's own CARLA server, comparing with the store (`FALCON_CI_STORE`). It never runs for a pull request from a
  fork: a self-hosted runner would execute a stranger's code with your GPU and network.

Both upload the output folder as an artifact (`if: always()`, so a failed gate still leaves its evidence). The
template was checked here with `falcon ci lint` and validated against SchemaStore's GitHub workflow schema
(check-jsonschema 0.39.0, `--builtin-schema vendor.github-workflows`, all four variants valid); actionlint was not
installed on this machine, so `falcon ci lint` runs it only where it is.

### The self-hosted runner

What the `carla` job needs on the machine that runs it — the same as Falcon on a desktop, unattended:

1. **Linux x86_64 with an NVIDIA GPU** that runs CARLA 0.9.16 (with NVIDIA's Vulkan driver), at least 16 GB of RAM,
   `systemd-run` (Falcon starts its server and workers only in memory-capped user scopes) — what `falcon doctor`
   checks.
2. **CARLA 0.9.16** unpacked, with its Python venv beside it holding the `carla==0.9.16` wheel, numpy and OpenCV
   (as the first-run wizard sets up; [packaging](packaging.md)).
3. **Falcon Autonomy** installed for the runner's user (the bundle on `PATH` as `falcon`, or `pip install` from a
   checkout) and its settings for that user (`$XDG_CONFIG_HOME/FalconAutonomy/settings.json`: `carla_root`,
   `client_python`, the server memory cap; set them once with the app or by editing the file). `falcon doctor` must
   pass as that user. Falcon starts its server on its own ports (2310 by default) and never touches a server it did
   not start.
4. **A systemd user session that survives logout**, since the runner is a service: `sudo loginctl enable-linger
   RUNNER_USER` (the memory-capped scopes are user units).
5. **Docker** (the runner's user in the `docker` group) for a ROS 2 stack: the bridge image (`falcon stack image`
   builds `falcon-ros2-bridge:humble` from `ros:humble-ros-base`, cached after the first build) and your stack's image.
6. **The runner**, registered for the repository with the labels the workflow names — `self-hosted, linux, x64, gpu,
   carla`: `./config.sh --url https://github.com/OWNER/REPO --token TOKEN --labels gpu,carla` (the first three
   labels are added by default), then `sudo ./svc.sh install RUNNER_USER && sudo ./svc.sh start`. One runner per
   GPU: Falcon's machine-wide CARLA lock queues a second job's server rather than starting two.
7. **The store** (optional): a folder the runner's user can write, set as the repository variable
   `FALCON_CI_STORE` (Settings ▸ Secrets and variables ▸ Actions ▸ Variables), or in the runner's `.env`.

### Posting to GitHub

Posting is a **public** action: a comment, check or status is visible to everyone who can see the repository.

- `falcon ci post RESULTS` without `--post` is a **dry run**: it prints the requests it would send (method, path,
  the start of the body) and sends nothing.
- With `--post` it sends, through the `gh` CLI (`--via gh`: `gh api`, authenticated by `gh auth login` or
  `GH_TOKEN`) or the REST API (`--via api`: `GITHUB_TOKEN` or `GH_TOKEN`, `--api-url` or `GITHUB_API_URL` for
  GitHub Enterprise). `FALCON_CI_NO_POST=1` refuses every post (Falcon's tests set it; nothing in them reaches GitHub).
- `--as comment` (default) posts the summary on the pull request (`--pr`, else from `GITHUB_REF` or the event), and
  edits its own earlier comment (found by a hidden `<!-- falcon-ci:NAME -->` marker in the first 1000 comments)
  rather than adding another. It needs the `pull-requests: write` (or `issues: write`) permission.
- `--as check` creates a check run on the commit (`--sha`, else the PR's head, else the result's commit) with the
  summary as its output; GitHub lets only GitHub Apps create check runs — `GITHUB_TOKEN` inside Actions with `checks:
  write`, not a personal token.
- `--as status` sets a commit status (`falcon-ci/NAME`: success, failure or error, the failed gates as its
  description, the run's URL); a personal token with `repo:status` works.

The workflow's `carla` job has a comment step that runs only when the repository variable `FALCON_CI_POST` is
`true` — off until the repository's owner turns it on — with `pull-requests: write` for that job only.

## GitLab CI, Jenkins and others

Any CI system that runs a command and reads JUnit works: run `falcon ci run`, keep the folder, give the system
`junit.xml`, show `summary.md`.

```yaml
# .gitlab-ci.yml
falcon:
  tags: [gpu, carla]                       # a runner set up as above
  script:
    - falcon ci run falcon-ci.yml --out falcon-ci-out --force
  artifacts:
    when: always
    paths: [falcon-ci-out/]
    reports:
      junit: falcon-ci-out/junit.xml       # the merge request's test report
```

```groovy
// Jenkinsfile
pipeline {
  agent { label 'gpu && carla' }
  stages {
    stage('Falcon CI') {
      steps { sh 'falcon ci run falcon-ci.yml --out falcon-ci-out --force' }
    }
  }
  post {
    always {
      junit 'falcon-ci-out/junit.xml'
      archiveArtifacts artifacts: 'falcon-ci-out/**', allowEmptyArchive: true
    }
  }
}
```

The exit code fails the job on a failed gate (`1`) or an error (`2`). `summary.md` is Markdown for a merge request
note (GitLab's API or `glab mr note`), which Falcon does not post itself. The commit is read from `CI_COMMIT_SHA`,
`GIT_COMMIT` or the checkout.

## Tests

`tests/test_ci.py`, against the fake carla, CPU only: the config's schema and its semantic checks; Wilson's interval
rules and the runs needed to demonstrate a rate; the commit from a git repository (a credential removed from the
remote) and from GitHub's and GitLab's variables; the paired comparison on known runs; gates that cannot be judged; the
output folder never reused silently; a follower suite end to end (every output, JUnit counts, `$GITHUB_STEP_SUMMARY`,
the report, a failing gate's exit 1, a re-run against it with no change, exit 2 for an occupied folder); a ROS 2
stack started from a command — `tests/data/ci_stack.py` plays the bridge node with the node's own logic — whose
"commit" that no longer stops regresses in pass and collision rate against the stopping one, and the latency gate on
its control wait; retries refused in a deterministic mode and recorded as flaky otherwise; cancelling with SIGTERM
(exit 3, outputs written, the stack's process stopped); a policy over two routes with the store on `main` and a
feature branch compared against it; a model against the oracle as baseline model (a regression) and on pinned drives
against the first run (none), and frames whose ground truth differs refused; `ci init --github` and the workflow's
structure and checks; broken workflows caught; posting as a dry run, refused by `FALCON_CI_NO_POST`, and through a
recording fake transport (comment created, earlier comment edited, check run, commit status, the requests the gh and
HTTP transports build); the Stack testing page's CI section.

## Limits

- **Simulation evidence within the scenarios, seeds and parameter values run**: not a certification claim, and with
  the safety case's own limits ([stack testing](stack-testing.md#the-safety-case-report)).
- **Multiple gates, multiple chances.** Each interval is 95 %: with many metrics and gates, one can fail by chance
  when nothing changed. `refute`, paired intervals and gating few metrics keep that down; the summary shows every
  number so a reviewer can judge.
- **Small suites decide little.** A handful of runs can only catch gross changes (see the runs needed above); the
  bootstrap over fewer than 5 pairs is flagged as unreliable.
- **Latency is the runner's**, under its load at the time: a budget on a shared machine is noisy, and a model's
  latency is not the target device's (the Jetson in the loop measures that: [hardware](hardware.md)).
- **The fake carla** tests the integration — the stack starting, the bridge, the topics, the controls, the gates —
  not driving; its scenarios run but its sensors are synthetic. The `smoke` job's numbers are not evidence.
- **Not run against GitHub from here.** The workflow was linted and schema-validated, the posting code exercised only
  through fake transports; nothing was posted. A real run needs the runner above, the Falcon tag on GitHub, and your
  stack's image build in place of the template's `docker build`.
- **One CARLA server at a time per machine**; runs within a suite are sequential.
