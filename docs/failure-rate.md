# Failure rate with a confidence interval

A rare-case search ([scenarios](scenarios.md#sweeps-and-rare-case-searches), [model in the loop](rare-cases-mil.md))
finds *where* a scenario fails; it cannot say *how often*, because it samples where failures are likely, not as the
parameters occur. The failure-rate estimate answers the second question: the probability that one **encounter** — one
run of the scenario, its parameters drawn as the scenario declares them — fails, with a 95 % confidence interval, by
adaptive importance sampling. A probability of 1e-5 then needs hundreds or a few thousand runs instead of millions.
Code: `falcon_autonomy/core/failrate.py` (its docstring is the reference), `sim/estimate.py` (the study and the
commands), `ui/failrate_panel.py` (Rare cases ▸ Failure rate).

## What is estimated

P_fail = P(the run fails) with the parameters x drawn from the **nominal distribution** p:

- each searched parameter's own distribution as the scenario declares it — uniform on its range, a normal (truncated
  to its range when one is given), a choice uniform over its choices;
- narrowed to the kinematically feasible part when the study narrows it ([core.rss](scenarios.md)): the estimate is
  then conditional on that part, whose nominal mass is reported (`nominal_mass`);
- a searched weather field (`--weather fog_density`) uniform over its range;
- the parameters independent.

**The assumption is yours to check**: these distributions stand for how the parameters occur in the operating domain.
A scenario's ranges are a test designer's ranges, not measured frequencies, and the estimate is exactly as good as they
are. Everything else of the encounter is the study's: the world, the rig, the traffic, the scenario seed (the world is
reloaded before each run, so a run is a deterministic function of its parameters).

**A failure** is judged by one of the existing judges, chosen with `--judge`:

| judge | a failure | code |
|---|---|---|
| `objective` (default) | the study's objective ≥ the threshold — the search's own failure; for the criticality objective it counts by default only failures the ego could have avoided under RSS | `core.milsearch.objective_value`, `core.rss.rss_objective` |
| `checks` | the scenario's pass criteria failed (no collision, minimum time to collision), as a stack suite judges a run | `run.json`, `core.stack` |
| `rules` | a traffic rule was violated: a signal-temporal-logic formula with negative robustness, the rule set for the run's handedness | `core.rules.check_run`, `core.stl` |
| `stl` | the study's own requirement was falsified: `--stl FORMULA` or `--rule SET:ID` has robustness ρ < 0 on the run (a run whose ρ is unknown is not judged and is replaced); one estimate per requirement (`rate-…-stl-<key>`) | `core.stlobjective` ([falsification](falsification.md)) |

The sampling is always adapted on the objective's score; importance sampling stays unbiased for any judge (the
proposal's defensive part covers every parameter set the nominal can produce), it is only less efficient when the
judge and the score disagree.

## The method

**Standard-normal coordinates.** Every parameter is a transform of one standard normal, x_j = F_j⁻¹(Φ(z_j)) (a choice
takes the cell ⌊Φ(z_j)·K⌋), so the nominal is N(0, I) in z.

**The proposal family** (fixed and documented): a defensive mixture (Hesterberg 1995)

    q(z) = β·N(z; 0, I) + (1 − β)·N(z; μ, I),   β = 0.1 (--defensive)

— the nominal itself beside the mean-shift family N(μ, I). Its weights w = p/q = 1 / (β + (1 − β)·exp(μ·z − |μ|²/2))
are at most 1/β, so the estimator's variance is finite whatever μ is.

**Adaptation**: the multilevel cross-entropy method for rare events (Rubinstein 1997; Rubinstein & Kroese 2004).
Stage 0 draws `--stage-size` runs from the nominal (it is also a plain Monte Carlo sample, reported as such). After
each stage the score sets a level γ = min(threshold, the ⌈ρ·n⌉-th largest score) (ρ = `--rho`, 0.1); μ is refitted to
the runs at or above it by the likelihood-ratio-weighted maximum-likelihood update μ = Σ W z / Σ W. Adaptation stops
when the level reaches the failure threshold, when it has not risen for two stages, when the score does not separate
the runs, or after `--max-stages`.

**The estimate**: `--final` runs from the final proposal, independent given it. With y_i = w_i·1{run i failed}:

    P̂ = mean(y),   SE = sd(y)/√n,   95 % interval P̂·exp(∓1.96·SE/P̂)

— the standard importance-sampling estimator (Owen 2013, ch. 9) and a delta-method interval on log P̂ (asymmetric,
never negative). The adaptation stages' runs are recorded but not pooled into it (their proposals depended on earlier
runs).

**Trust.** ESS_fail = (Σ_fail w)²/Σ_fail w² (Kish's effective sample size over the failing runs).

| | when |
|---|---|
| **refused** | fewer than 10 failing final runs, or ESS_fail < 10. With no failure at all, only a bound: P ≤ 3/(β·n) at 95 % (the rule of three on the defensive part, which draws from the nominal with probability β) |
| **warning** | ESS_fail < 30, the relative standard error SE/P̂ > 0.3, or one run carrying more than 20 % of the failures' weight |
| ok | otherwise |

A refused estimate's numbers are kept in the file for diagnosis and shown nowhere as an answer.

**Exposure.** The estimate is per encounter. *Failures per 10ⁿ km* (or per 10ⁿ hours) needs how many such encounters
happen per km (or hour) in the operating domain — a figure simulation cannot measure. You give it with its source
(`--exposure-per km --encounters X --exposure-source "…"`); it is taken as exact, the rate is P̂ × X with the interval
scaled alike, and n is chosen so the number is at least 1.

## Validation

`falcon failrate check` runs the estimator on synthetic problems whose failure probability is known exactly, over
seeds, and compares plain Monte Carlo with the same number of runs:

- **tail**: d standard normals, a failure when a·z/|a| ≥ c with c = −Φ⁻¹(P) — a linear limit state;
- **corner**: d parameters uniform on [0, 1], a failure when every one exceeds 1 − P^(1/d) — a corner of the box,
  through the uniform transform, the score min_j u_j − (1 − P^(1/d)).

Measured 2026-11-02, 200 seeds per row, `falcon failrate check --seeds 200 --stage-size 500 --final 1000` (ρ 0.1,
β 0.1, at most 10 stages). *covered*: the share of given estimates whose 95 % interval contains the truth; *ok-cov*: the
same among those without a warning; *MC 0-fail*: the share of seeds where plain Monte Carlo with the same runs saw no
failure; *MC runs*: what plain Monte Carlo needs for the estimator's median relative standard error, (1 − P)/(P·rel²).

| problem | P | d | given | warned | covered | ok-cov | mean \|rel. error\| | runs | MC 0-fail | MC runs, same precision |
|---|---|---|---|---|---|---|---|---|---|---|
| tail | 1e-4 | 2 | 200/200 | 0 | 0.920 | 0.920 | 0.058 | 2,500 | 0.79 | 2.1 × 10⁶ |
| tail | 1e-4 | 5 | 200/200 | 0 | 0.935 | 0.935 | 0.059 | 2,500 | 0.79 | 2.0 × 10⁶ |
| tail | 1e-6 | 2 | 200/200 | 0 | 0.935 | 0.935 | 0.066 | 2,995 | 0.99 | 1.6 × 10⁸ |
| tail | 1e-6 | 5 | 200/200 | 0 | 0.945 | 0.945 | 0.069 | 2,988 | 0.99 | 1.5 × 10⁸ |
| corner | 1e-4 | 2 | 200/200 | 0 | 0.925 | 0.925 | 0.100 | 2,580 | 0.79 | 7.3 × 10⁵ |
| corner | 1e-4 | 5 | 172/200 | 165 | 0.884 | 0.143 (1 of 7) | 0.215 | 3,698 | 0.73 | 1.8 × 10⁵ |
| corner | 1e-6 | 2 | 200/200 | 0 | 0.940 | 0.940 | 0.121 | 3,120 | 0.99 | 4.8 × 10⁷ |
| corner | 1e-6 | 5 | 87/200 | 87 | 0.621 | — | 0.338 | 5,472 | 0.98 | 1.3 × 10⁷ |

With a smaller budget, `--stage-size 100 --final 200` (about 500–960 runs, a CARLA-sized study):

| problem | P | d | given | warned | covered | ok-cov | mean \|rel. error\| | runs | MC 0-fail |
|---|---|---|---|---|---|---|---|---|---|
| tail | 1e-4 | 2 | 200/200 | 45 | 0.945 | 0.948 | 0.135 | 500 | 0.95 |
| tail | 1e-4 | 5 | 193/200 | 142 | 0.938 | 0.941 | 0.165 | 500 | 0.95 |
| tail | 1e-6 | 2 | 198/200 | 157 | 0.929 | 0.951 | 0.140 | 582 | 1.00 |
| tail | 1e-6 | 5 | 174/200 | 166 | 0.908 | 0.875 | 0.175 | 582 | 1.00 |
| corner | 1e-4 | 2 | 186/200 | 186 | 0.914 | — | 0.218 | 538 | 0.95 |
| corner | 1e-4 | 5 | 9/200 | 9 | 0.333 | — | 0.504 | 726 | 0.94 |
| corner | 1e-6 | 2 | 108/200 | 108 | 0.852 | — | 0.306 | 638 | 1.00 |
| corner | 1e-6 | 5 | 1/200 | 1 | 0 | — | 0.737 | 962 | 1.00 |

Read honestly:

- On the linear tail, in 2 and 5 dimensions, at 1e-4 and 1e-6, the interval covered the truth in 92–95 % of the seeds
  (binomial standard error of a 200-seed share: 1.5 points): about the nominal rate, a little under it — the delta
  method's interval assumes a roughly normal estimator. Over the 200 seeds of the 1e-4, 2-D row the estimates averaged
  1.005 × the truth and the misses fell 8 above and 8 below: unbiased, the interval slightly short.
- Plain Monte Carlo with the same 2,500–3,000 runs saw **no failure at all** in 79 % (1e-4) and 99 % (1e-6) of the
  seeds, and would need 2 × 10⁶ and 1.6 × 10⁸ runs for the same precision — about 820 and 54,000 times as many.
- The corner of a 5-dimensional box is where the mean-shift family is weak: most estimates were refused or warned, and
  the warned ones covered 92 % (1e-4) and 62 % (1e-6). **Take a warning seriously**: it is the estimator saying its
  effective sample is small. In 7 of the 1e-4 row's 172 given estimates no rule fired and 6 of those 7 missed — the
  trust rules do not catch everything. The Pareto-k̂ diagnostic of the weights' tail (Vehtari et al., Pareto smoothed
  importance sampling) was tried on seeds 0–99 of that row and did not separate them (the 4 estimates without a warning
  there, 3 of them missing, all had k̂ ≤ 0.5), so it is not used.
- At the small budget, most estimates carry a warning; on the tail the warned ones still covered 91–94 %.

**The full-covariance Gaussian, measured and not offered**: `falcon failrate check --family full --problems tail
--seeds 40` (the same budget, the covariance refitted as well): its spread collapsed stage by stage — the elite of a
narrow proposal cannot show the target's spread — so 21, 30, 38 and 40 of the 40 seeds stalled below the threshold
(refused), and the given estimates covered the truth in 2 of 19, 0 of 10 and 1 of 2.

## The page and the command

Rare cases ▸ **Failure rate** uses the search's scenario, objective, world, rig, threshold, weather searched, rate and
seed, and adds *Runs per stage*, *Final runs*, *Stages at most*, *A failure is* (the judge), and the *Rate* (per
encounter, km or hour, with the encounters per unit and their source). It queues `falcon failrate run` as a job and
shows the probability per encounter with its interval, the rate per exposure unit, the effective failures, the runs,
the reasons for a refusal or a warning, and what plain Monte Carlo would have needed. Each scenario, objective (and
model), weather searched, rate and judge has its own study, `studies/rate-<scenario>…`.

```sh
falcon failrate run PROJECT --scenario lead_brake [--name N] [--stage-size 100] [--final 200] [--max-stages 6] \
    [--rho 0.1] [--defensive 0.1] [--batch 4] [--threshold 0.5] [--judge objective|checks|rules|stl] \
    [--exposure-per km|hour --encounters X --exposure-source TEXT] [--world W] [--rig R] [--seed N] \
    [--search-seed N] [--objective criticality|perception|both --model ID] [--weather FIELD[:LO:HI]] [--fps F]
falcon failrate show STUDY_DIR | PROJECT --name N [--json] [--exposure-per km --encounters X]
falcon failrate check [--p 1e-4,1e-6] [--dims 2,5] [--problems tail,corner] [--seeds 50] [--stage-size 500] \
    [--final 1000] [--family mean_shift|full] [--json]
```

An estimate is resumable: the sampler's state is saved after every batch (`failrate-state.json`, its random
generator included), and the runs are the study's, cached by their drive's key — run the same command again and it
continues where it stopped, driving nothing twice. Options that change the estimate (the judge, the threshold, the
sizes, the objective) start a fresh sampler.

The assistant: `jobs_search` with `method: "estimate"` (its budget the runs at most: a fifth per adaptation stage, three
stages, the rest for the final stage) and `analysis_outputs` with kind `failure_rate`.

## Outputs

`studies/<name>/failure-rate.json` (`falcon.failure_rate/1`): the estimate (P̂, SE, both intervals, ESS, the trust
status and reasons, the zero-failure bound), the nominal and its feasible mass, the proposal family and the final
proposal, the stages (level, elite, failures, μ), why adaptation stopped, the runs (total, final, lost), stage 0 as
plain Monte Carlo with Clopper–Pearson bounds, the plain-Monte-Carlo runs for the same precision, the rate per exposure
unit, the estimate per [ODD cell](odd-coverage.md) (the ratio estimator Σ w·1{fail}·1{cell} / Σ w·1{cell} over the
final runs, with its delta-method error and the same trust rules), the final runs with their weights, the judge, and
provenance. `results.jsonl` holds every run as a search's does.

**As evidence**: the safety-case report (`falcon stack report … --study studies/rate-…`) lists each estimate in a
*Failure rates* section and among the residual risks — a refused one as insufficient evidence; a given one by its
interval against the report's failure-rate threshold. The [coverage grid](odd-coverage.md) shows the per-cell
estimates in their cells.

## Limits

- The nominal distributions are the scenario's declared ones; they are not measured frequencies of the operating
  domain, and the estimate means nothing beyond them. A rate per km or hour is only as good as the encounter rate
  given.
- One Gaussian component: a scenario with two separate failure regions is covered outside the one found by the
  defensive part only — unbiased, but with heavy weights (the warnings say so). A failure region much narrower than
  the nominal in some direction (a corner in many dimensions) is sampled with the nominal's spread there: measured
  above, most such estimates are refused or warned, and warned ones under-cover.
- The interval is a large-sample one; it under-covers by a few points even when all is well (92–95 % measured).
- A run that fails to run is replaced by a new draw, which assumes such failures do not depend on the parameters; the
  lost runs are counted.
- Simulation's own limits bound every estimate (docs/carla-lessons.md, the safety case's *Limits*).
