# The release gate between two versions

Before a new version of a driving stack or a perception model replaces the old one, the question is: **is it better,
worse, or no different — and do we know yet?** The release gate runs the old version (A) and the new version (B) on
identical scenarios with identical seeds, looks at the results after every batch, and stops the moment the answer is
clear. Its confidence is *anytime-valid*: it holds however often the gate looks, so stopping early costs nothing in
error. Code: `falcon_autonomy/core/releasegate.py` (its docstring is the reference); `ui/gate_tab.py` (the section).

Its neighbours answer other questions: [CI mode](ci.md) judges one version against fixed limits (and against a
baseline with a fixed number of runs); [bisect](bisect.md) finds the commit that broke a scenario; the
[failure rate](failure-rate.md) estimates how often one version fails. The gate compares two versions with as few
runs as the answer allows.

## Where it shows

| where | what |
|---|---|
| Reports ▸ *Release gate* | the old and the new version, the scenarios, the trial budget, α, the margin, the primary outcome, *Runs on*; *Run the gate* queues it as a job (resumable); the verdict as a word with its colour, the paired trials used, the confidence band of the primary difference over the trials (zero solid, ±margin dashed, the trial where the verdict held marked — and said when the discordant pairs decided it before the band cleared zero), the scenarios newly failing and newly passing under B with their pair of runs (*Browse run A / B*, *Compare the pair*) |
| `falcon gate run PROJECT --name N --a VERSION --b VERSION --scenario KEY… [--budget 200] [--batch 8] [--alpha 0.05] [--margin 0.05] [--non-inferiority M] [--primary failure\|value] [--failure criteria\|value] [--threshold 0.5] [--min-trials N] [--seed N] [--world W] [--rig R] [--fps F] [--motion kinematic\|physics] [--on local\|cluster\|cluster:POOL] [--gate [--fail-on worse,cant_tell]] [--fake-carla DIR]` | the gate in `<workspace>/gates/<name>/`; running it again resumes it; exit 0 when it ran whatever the verdict, and with `--gate` exit 1 when the verdict is one of `--fail-on` (default `worse`) |
| `falcon gate show PROJECT --name N [--json] [--gate [--fail-on …]]` | the result's headline, or `gate.json` |
| `falcon gate plan --p-a P --p-b P [--trials N] [--reps R] [--alpha A] [--margin M] [--common RHO]` | what a budget can show: the share of each verdict over simulated gates with those failure probabilities (no runs) |
| [CI mode](ci.md) `gates: {release: {path: DIR, fail_on: [worse]}}` | a CI gate that fails the build when a release gate made earlier in the job says *worse* (an error when there is no finished gate at `path`, relative to the config) |
| [the safety case](gsn.md) | evidence kind `gate` (`gates/<name>`): claim `{measure: verdict, equals: equivalent}` or a `field` such as `risk.upper < 0.05`; bound by hash like every result |
| the assistant | `formal_gate_run` (act), `formal_gate_show` (read) ([assistant tools](assistant-tools.md)) |

Since 4.2 the section also shows each version's **fidelity** as a chip (A, B: `core.fidelity`, from its trials'
runs), a critical callout when the gate refused a verdict or a version's trials mix levels (below), the scope as one
line ("the verdict holds for the tested scenario distribution only") with the whole sentence behind *Details*, and on
each newly failing row the pair's fidelity, short mono run ids (a click copies the whole path), and **Compare in
Replay** — both runs of the pair on the Replay page (its `compare_runs`, when this build's Replay page has it; else
the button is off and its tooltip says why; *Compare the pair* still shows them on the Data page). A row selected
makes its B run the window's active run; its context menu has *Send to ▸*. The intro prose is three badges with the
text behind *Details*.

## Fidelity: no verdict across simulators

Every trial records both sides' level (`a.fidelity`, `b.fidelity` in `trials.jsonl`; a run's `run.json`
provenance, `unknown` for a run written before 4.2). `gate.json` keeps each version's (`fidelity_by_version`,
`core.fidelity.combine` over its trials) and the gate's (`fidelity`).

| case | what the gate does |
|---|---|
| both versions at one level (both CARLA, both the fake …) | the verdict as usual |
| the versions at different levels that are not like for like (`core.fidelity.comparable`: a fake or kinematic side against a real one, CARLA against CARLA UE5) | **no verdict**: `verdict: refused`, word "no verdict: different fidelity", `refused` says why, and the gate stops after the batch that showed it instead of spending its budget. `falcon gate show` prints NO VERDICT; with `--gate` it exits 1 whatever `--fail-on` says; CI's release gate is an error; the safety case's claim on the verdict is not met |
| a version whose own trials mix levels (`mixed`) | the verdict is kept and flagged: `fidelity_note` ("Mixed fidelity: version A's trials mix …"), shown in the critical colour beside a *Mixed fidelity* chip — the pairs pool runs of different fidelity; re-run on one simulator to confirm it |
| one side does not record its level (`unknown`) | compared, with the reason in `fidelity_note` |

Tested with planted versions (`tests/test_releasegate.py`, `tests/test_ui42_studies.py`) and on the fake CARLA
(both sides `fake`).

## Versions

| spelling | the version |
|---|---|
| `follower`, `autopilot` | a baseline driver: Falcon's lane follower, CARLA's traffic manager |
| `stack:SUITE.json` (or the file's path) | the `stack` block of a `falcon.suite/1` file ([stack testing](stack-testing.md)): a ROS 2 stack with its bridge settings and launch commands, started for every run as a suite starts it |
| `model:ID` | a registered detector: both versions score the **same recorded run** (driven once, with the rig's labels) by the perception failure of the [model-in-the-loop search](rare-cases-mil.md) |

Both versions must be of the same kind. Two commits of one stack are two suite files (or two checkouts each with its
own); the gate does not check anything out — [bisect](bisect.md) does.

## The design: pairs with common random numbers

Trial *i* is one scenario instance: scenario *i* mod *m* of the scenarios given, its parameters drawn from their
declared distributions with the seed `seed · 1000003 + i`. It is run once under A and once under B with that seed,
world, rig, rate and motion, so the two runs of a pair differ in the version only (tested: the two runs' configs hold
the same seed and parameter values). Each run gives

- a binary outcome, **failed**: the scenario's pass criteria were not met (`--failure criteria`, the default: no
  collision, the minimum time to collision), or its value reached `--threshold` (`--failure value`; always so for
  models);
- a continuous outcome in [0, 1], the **value**: the run's [criticality](methods.md#criticality), or for a model its
  perception failure. Higher is worse.

Pairing removes the scenario's own variance from the comparison: what is analysed is the pair's difference.

## The statistics

Three bounds, each **time-uniform** — it holds at every number of trials simultaneously, with total error α — so the
gate may look after every batch and stop at the first clear answer.

**1. The discordant pairs.** Among the pairs where exactly one version failed, K of N are "B failed, A passed". With
independent trials K is Binomial(N, θ), and the versions fail equally often exactly when θ = ½ (McNemar 1947). The
test is a *mixture* sequential probability ratio test: the likelihood ratio of θ against ½, averaged over a uniform
prior on θ,

    E_N = ∫ θ^K (1−θ)^(N−K) dθ / (½)^N = 2^N / ((N + 1) · C(N, K)).

Under θ = ½ it is a non-negative martingale starting at 1, so the probability that it *ever* reaches 1/α is at most α
(Ville's inequality). Inverting it for every θ₀ gives a confidence sequence for θ (Robbins 1970, the method of
mixtures). **Why this and not Wald's SPRT:** Wald's test needs the alternative θ₁ chosen beforehand and is efficient
only near it; the mixture needs no guess of the effect, is two-sided, and gives an interval. The formula is exact;
nothing in it is a remembered constant.

**2. The risk difference** Δ = P(B fails) − P(A fails), the mean of d = fail_B − fail_A ∈ {−1, 0, 1}, and
**3. the value difference**, the mean of x = value_B − value_A ∈ [−1, 1]: a predictable plug-in
**empirical-Bernstein confidence sequence** for the mean of a bounded variable (Waudby-Smith & Ramdas 2024,
*Estimating means of bounded random variables by betting*, JRSS B 86, Theorem 2; the supermartingale is Howard,
Ramdas, McAuliffe & Sekhon 2021, Ann. Statist. 49). With X the difference mapped to [0, 1], running estimates
μ̂_t = (½ + ΣX)/(t + 1) and σ̂²_t = (¼ + Σ(X_i − μ̂_i)²)/(t + 1), and a bet that uses only the past,
λ_t = min(√(2 log(2/α) / (σ̂²_{t−1} · t · log(1 + t))), c):

    μ ∈ (Σ λ_i X_i ∓ (log(2/α) + Σ ψ(λ_i)(X_i − μ̂_{i−1})²)) / Σ λ_i   at every t,   ψ(λ) = −log(1 − λ) − λ.

Its validity is proved in the module's docstring from one inequality — exp(λξ − ψ(λ)ξ²) ≤ 1 + λξ for ξ ≥ −1,
0 ≤ λ < 1 (Fan, Grama & Liu 2015) — which the tests check on a grid; it holds for *any* bet that uses only the past.
**Recalled, not checked against the paper:** the tuning — the bet's formula, the starting values ½ and ¼, and the cap
c = ½. They change the interval's width, never its coverage. The running intersection of the intervals is reported.

## The verdict

With the primary outcome `failure` (default), α is split: α/2 to (1), α/2 to (2).

| verdict | when (checked in this order after every batch; the first that holds stops the gate) |
|---|---|
| **worse** | θ's interval lies above ½, or Δ's above 0 |
| **better** | θ's interval lies below ½, or Δ's below 0 |
| **equivalent** — "no difference within the margin" | Δ's interval lies inside (−margin, +margin) |
| **non_inferior** — "not worse within the margin" | only with `--non-inferiority M`: Δ's upper bound is below M |
| **cant_tell** — "can't tell yet" | none of these when the budget is spent |

A difference that is shown *and* inside the margin is still reported as better or worse, with `within_margin: true`.
The value's interval (3) is computed at α on its own and reported as a secondary result — it is not part of the
verdict's error budget. With `--primary value` the verdict is read from (3) at α instead (positive: B is more
critical), and (1) and (2) are secondary.

Whenever every interval covers its truth — probability at least 1 − α — no verdict is wrong. `gate.json` says which
bound decided (`decided_by`: `discordant`, `risk` or `value`); the discordant-pair test usually decides *worse* and
*better* long before the risk difference's band clears zero, and the section says so on the band.

### Measured: the error under optional stopping

`simulate_verdicts` (the gate's own rule, vectorised; `falcon gate plan`), no difference between the versions, a look
after **every** trial for 400 trials, 20,000 simulated gates each, α = 0.05, margin 0.02 (2026-11-24):

| failure probability of both | pairs failing together | said *better* or *worse* |
|---|---|---|
| 0.02 | independent | 0.7 % (0.40 % better, 0.32 % worse) |
| 0.10 | independent | 1.4 % (0.73 %, 0.66 %) |
| 0.30 | 60 % share a cause | 1.3 % (0.72 %, 0.60 %) |
| 0.50 | independent | 1.5 % (0.77 %, 0.75 %) |

All below α: the mixture test and the empirical-Bernstein bound are conservative (they pay for being valid at every
look). The tests repeat this with 2,000 gates per row and require the rate to stay within α plus three Monte-Carlo
standard errors.

### What a verdict costs

At α = 0.05 with the failure primary, the earliest possible verdicts:

- *worse* or *better*: 9 discordant pairs all one way (2⁹ / 10 = 51 ≥ 40). With mixed discordant pairs, more.
- *equivalent*: with **no** discordant pair at all, about 17.5 / margin trials — 88 for a margin of 0.2, 175 for
  0.1, 350 for 0.05. Showing the absence of a difference is expensive, and until then the gate says "can't tell yet".

Simulated (2,000 gates each, margin 0.05; "common": the share of pairs that fail or pass together, which common random
numbers produce):

| P(A fails) → P(B fails) | budget | common | *worse* | *better* | can't tell | median trials to the verdict |
|---|---|---|---|---|---|---|
| 0.05 → 0.30 | 400 | none | 100 % | 0 | 0 | 42 |
| 0.05 → 0.30 | 400 | half | 100 % | 0 | 0 | 37 |
| 0.10 → 0.20 | 1000 | none | 99.9 % | 0 | 0.1 % | 234 |
| 0.10 → 0.20 | 1000 | half | 100 % | 0 | 0 | 154 |
| 0.10 → 0.15 | 3000 | none | 99.2 % | 0 | 0.6 % (0.2 % *equivalent*: the difference is exactly the margin) | 863 |
| 0.10 → 0.15 | 3000 | half | 99.8 % | 0 | 0 (0.3 % *equivalent*) | 492 |
| 0.02 → 0.10 | 1000 | none | 100 % | 0 | 0 | 154 |
| 0.02 → 0.10 | 1000 | half | 100 % | 0 | 0 | 127 |
| 0.30 → 0.10 | 400 | none | 0 | 100 % | 0 | 72 |
| 0.30 → 0.10 | 400 | half | 0 | 100 % | 0 | 56 |

Pairs that share their fate (what common random numbers buy) shorten the gate: fewer concordant failures dilute the
discordant ones. With no true difference and half the pairs sharing their fate (both fail with probability 0.1), the
gate said *equivalent* in 99.3 % of 300 simulated gates after a median of 245 trials for a margin of 0.1, and in
98.7 % after 725 for 0.05 (the rest said better or worse: within α) — against 176 and 351 trials when no pair is
discordant at all.

## Outputs

`<workspace>/gates/<name>/`:

| file | holds |
|---|---|
| `gate.json` | `falcon.gate/1`: the verdict and its word, `decided_at` and `decided_by`, the trials run and used, the design (scenarios, seed, α, margin, primary), the versions, failures under A and B, the discordant pairs with θ's interval, the risk and value differences with their bounds, `path` (every bound after every trial — the band), `newly_failing` and `newly_passing` (trial, scenario, seed, parameter values, the value under A and B, **the two run folders**), `skipped`, `scope`; since 4.2 `fidelity`, `fidelity_by_version`, `refused` and `fidelity_note` |
| `trials.jsonl` | one line per finished pair (each side with its `fidelity` since 4.2) |
| `gate.md` | the same as a report section: the verdict, the bounds table over the trials, the newly failing and newly passing scenarios with their run pairs |
| `runs/a/<key>/`, `runs/b/<key>/` | the stack runs of each side (`stack.json`, `run.json`, `frames.jsonl`, `trajectories.json`): browsable, replayable (`falcon stack replay`) and comparable on the Data page |
| `studies/gate-<name>-<scenario>/` | for models: the recorded drives both versions scored |

**Resuming.** Finished pairs are read back; a gate that reached a verdict runs nothing more; a gate that could not
tell continues when the budget is raised. Another design (scenarios, seed, α, margin, primary outcome) in the same
folder is refused rather than mixed in.

**Where the runs go.** `--on cluster` sends the runs to the cluster's nodes as a sweep's go
([cluster](cluster.md)); the gate itself runs here. A scenario or world made from dashcam footage never leaves this
machine: the gate is refused before anything is queued.

## Limits

- **A verdict holds for the tested scenario distribution only**: these scenarios, their declared parameter
  distributions, this world, rig, rate and motion. It says nothing about the road, and "better here" can hide "worse
  elsewhere"; choose the scenarios as the release's claim requires. `gate.json` carries this sentence (`scope`).
- Trials are analysed as independent, identically distributed draws of that distribution. With several scenarios the
  round-robin makes them exchangeable within a round rather than i.i.d.; with one scenario they are i.i.d.
- In kinematic motion a run repeats bit for bit, so a pair's difference is the version's alone. In physics motion,
  with traffic, or with a ROS 2 stack that is not deterministic, the difference also holds run-to-run noise: it costs
  trials, not validity.
- A pair with a side that did not finish is not a trial: it is listed under `skipped` and not run again, which
  assumes that not finishing is unrelated to the outcome. Many skipped pairs stop the gate with an error.
- The margin is on the primary difference in its own units (a failure-probability difference, or criticality); the
  secondary interval has its own α and is not covered by the verdict's.
- The empirical-Bernstein band is wide for the first tens of trials (it cannot be narrower than about
  17.5 / t at α = 0.05); the discordant-pair test is what makes early *worse* and *better* verdicts possible.
- A stack version is what its suite file launches; the gate records the file's path and name, not a commit. Models
  are compared on one recorded run per trial: the run's own driving is the same for both. Two models are told apart
  by their fingerprint (the model file's SHA-256; for a Python plug-in its registration without the `params`): two
  versions with the same fingerprint are refused rather than compared with themselves.
- Only fake-CARLA runs have been made by the tests (the lane follower against CARLA's autopilot on the fake); a gate
  on a real server with real stacks has not been run here.
