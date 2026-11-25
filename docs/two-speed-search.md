# Two-speed failure search

A [rare-case search](scenarios.md#sweeps-and-rare-case-searches) runs every evaluation in CARLA: seconds to minutes
each. The two-speed search puts a **fast simulator** in front: it rates thousands of parameter sets of the scenario in
seconds, and CARLA runs only the candidates it rates most critical — plus a share of random controls that measure how
far the two simulators agree. The budget is counted in CARLA runs. The cheap value is a **screen, never a result**:
every failure reported is a CARLA run. Code: `falcon_autonomy/core/fastsim.py` (its docstring is the reference),
`ui/prescreen_tab.py` (the section).

## Where it shows

| where | what |
|---|---|
| Rare cases ▸ *Two-speed search* | the scenario, the CARLA runs and how they split (per round, the share of random controls — the split is written out), the cheap candidates, the rank correlation below which the screen is not trusted, *Runs on*; the world, rig, seed, threshold and counting are the page's. *Search at two speeds* queues the job (resumable). Then: the rank correlation with the trust word in its colour, the false-negative bound, the scatter of the cheap against the CARLA criticality of every confirmed run (filled: selected; ring: random control; the fitted relationship as a line), and the confirmed failures. The study is the page's own search of that scenario, so the page's curve, failing cases (replay, export, *Why did this fail?*) and failure map show its runs |
| `falcon prescreen run PROJECT --scenario KEY [--name N] [--budget 40] [--batch 8] [--threshold 0.5] [--screen 10000] [--refine 2000] [--controls 0.2] [--min-spearman 0.5] [--min-pairs 12] [--fallback bayes-pof\|…] [--retrust] [--driver follower\|idm] [--device cpu\|cuda] [--count avoidable\|all] [--no-feasibility] [--seed N] [--search-seed N] [--world W] [--rig R] [--fps F] [--on local\|cluster\|cluster:POOL] [--fake-carla DIR]` | the study `studies/search-<scenario>` (or `--name`); refused, with the reason, for a scenario the fast simulator cannot run |
| `falcon prescreen show PROJECT (--scenario KEY \| --name N) [--json]` | the result's headline, or `prescreen.json` |
| `falcon prescreen screen --scenario KEY [--n 10000] [--top 10] [--json]` | the cheap screen alone — no CARLA, no project: the most critical candidates by the cheap model |
| `falcon prescreen scenarios` · `falcon prescreen bench [--scenario KEY] [--n N] [--device cpu\|cuda]` | which built-ins are covered and why not the rest; the throughput on this machine |
| the assistant | `research_prescreen_run` (act), `research_prescreen_show` (read) ([assistant tools](assistant-tools.md)) |

Since 4.2 the section also shows (each tested in `tests/test_ui42_studies.py`):

- **Fidelity chips** (`core.fidelity`): *Confirmed runs* — the level the CARLA runs' `run.json` record
  (`Fake simulator` on the fake, `Mixed fidelity` in the critical colour when the study pooled simulators) — and
  *Screen* — always `Kinematic`. Each confirmed failure's row has its run's level; run ids are short, mono, and a
  click copies the whole path.
- **Rank correlation as the pairs arrived**: ρ over the first *k* pairs for every *k*, one axis (−1 … +1), the trust
  threshold the shown study was judged with as a labelled dashed line (*trust at ρ ≥ 0.50*; before any study, the
  *Trust at ρ ≥* box's value). The box sets the next search's threshold: when it differs from the study's, it shows as a
  faint dotted *next search: ρ ≥ …* line, so a verdict is never shown beside a threshold it was not judged by. A dotted
  mark at the 12 pairs trust is judged from.
- **Why correlation is low**, shown when ρ is under the study's threshold or the screen is not trusted: key-value rows
  (ρ, too few pairs, ties in the cheap value, the worst pair, the parameter the disagreement grows with, a tiny
  camera), the three pairs whose ranks disagree most (each with its parameters, cheap and CARLA values and run —
  double-click browses it, right-click *Send to ▸*), and the full sentences behind *Details*.
- The intro is three badges (*Screen kinematic, straight road* · *Confirms CARLA* · *Controls random share*), the
  caveat "the cheap value is a screen, never a result" as one line, the rest behind *Details*. A failure row selected
  becomes the window's active run; its context menu has *Send to ▸*.

## The fast simulator

A scenario ([scenarios](scenarios.md)) is run for a whole batch of parameter sets at once, on **one straight road** in
road coordinates, mirroring what the scenario director does in its *kinematic* motion (the default motion of scenario
runs):

- **The ego** is driven by the same controller: the lane follower's time-gap rule and speed controller
  (`plugins/lane_follower.py`) on the nearest actor within 1.8 m of its route, through the director's kinematic speed
  law (3 m/s² at full throttle, 8 m/s² at full brake). The constants are read from those classes, not copied, and
  the director's 1 m path points and 2 m route points are kept. On a straight road the ego does not steer.
- **Scenario vehicles** follow their path as the director drives them: the set speed ramping at the action's rate,
  the director's speed controller and pure pursuit through the kinematic bicycle, a lane change moving the lateral
  offset linearly; walkers walk along their direction; proxies (cattle, handcart, vendor) move along their path;
  stand-ins keep their class's top speed and acceleration ([Indian assets](indian-assets.md)).
- **The timeline** fires as the director fires it (time, distance and ahead triggers; speed, brake, stop, go, walk,
  lane change). A door opening is accepted and has no kinematic effect — nor has it in kinematic CARLA runs, which
  measure collisions on the boxes.
- **The objective is the same formula**: [criticality](methods.md#criticality) computed by `core/criticality.py`'s
  own functions on the simulated trajectories (time to collision, post-encroachment time, required deceleration,
  box distance, overlap). One approximation: a post-encroachment time above 6 s is taken as absent, which changes the
  criticality by less than 0.002.

Tested: for `lead_brake` and `cut_in` the batched rollout equals, step for step to 1e-6, a scenario stepped with the
real `LaneFollower.act` and the director's `follow`; and for six scenarios the batched criticality equals
`criticality.summarise` on the same trajectories.

`--driver idm` replaces the lane follower by the intelligent driver model (Treiber, Hennecke & Helbing 2000;
`core/levelk.py`'s, clipped to the same acceleration limits): a stated driver model for a stack that cannot run
outside CARLA. It is a *different driver* from the one in CARLA; how well it still ranks the scenarios is exactly what
the fidelity report measures.

### What it covers

`falcon prescreen scenarios`: 14 of the 15 built-in scenarios; `unprotected_turn` is refused (the ego turns across a
junction: the fast simulator has one straight road). A project scenario is refused, by name with the reason, when it
needs a junction, the opposite lanes or a speed breaker; places an actor in the opposite lanes; replays a recorded
track or is placed on a map (a dashcam event); has a level-k actor; or an ego driven by CARLA's autopilot.

### What it does not have

The map (one straight lane 3.5 m wide with one beside it); the vehicles' real boxes (the director's nominal sizes; the
ego 4.6 × 2.0 m); CARLA's physics (it mirrors kinematic motion only); the weather; background traffic; and the
**RSS label** — the search's default objective counts only failures the ego could have avoided
([rare cases](rare-cases-mil.md)), which the fast simulator cannot label, so a run it rates critical can be capped in
CARLA's objective. `--count all` removes that difference; with the default the confirmed pairs show what it costs.

### Measured: agreement with the director

Against the test suite's fake CARLA, which runs the real scenario director on a straight town road (2026-11-24): 16
uniform parameter sets per scenario, criticality counting all failures, the cheap value against the run's.

| scenario | Spearman ρ | mean \|difference\| | largest | same side of 0.5 |
|---|---|---|---|---|
| lead_brake | 0.98 | 0.008 | 0.022 | 16 of 16 |
| cut_in | 0.92 | 0.077 | 0.496 | 14 of 16 |
| pedestrian_crossing | 1.00 | 0.009 | 0.027 | 16 of 16 |
| cyclist_crossing | 0.98 | 0.008 | 0.100 | 16 of 16 |
| door_opening | 1.00 | 0.047 | 0.052 | 16 of 16 |
| stalled_vehicle | 0.99 | 0.004 | 0.016 | 16 of 16 |
| two_wheeler_filtering | 0.99 | 0.038 | 0.173 | 16 of 16 |
| jaywalker_behind_bus | 0.99 | 0.011 | 0.036 | 16 of 16 |
| wrong_way_scooter | not defined (all 16 collide in both) | 0.000 | 0.000 | 16 of 16 |
| bus_pulling_out | 1.00 | 0.001 | 0.012 | 16 of 16 |
| stray_cattle | 0.96 | 0.004 | 0.014 | 16 of 16 |
| handcart_kerb_lane | 0.98 | 0.004 | 0.020 | 16 of 16 |
| autorickshaw_cut_in | 0.99 | 0.014 | 0.023 | 16 of 16 |
| monsoon_waterlogged | 0.99 | 0.008 | 0.017 | 16 of 16 |

Read honestly: this is the fast simulator against the same director code on the fake's straight, flat road with the
fake's vehicle boxes — its best case. `cut_in` shows the kind of disagreement to expect: two of 16 runs landed on the
other side of 0.5 (a lane change that just clears the ego in one model and not in the other). **No real CARLA
server was run for this feature**: on a real map (curved lanes, real boxes, the server's own lane widths) the
agreement will be lower, and the two-speed loop measures it for every study rather than assume it.

### Measured: throughput

`falcon prescreen bench --n 2048`, this machine (AMD Ryzen 9 9900X, one process; RTX 5080), 2026-11-24. An
evaluation is one full scenario (240–320 steps at 20 Hz) with its criticality.

| scenario | rollouts/s (CPU) | evaluations/s (CPU, rollout + criticality) |
|---|---|---|
| lead_brake | 18,200 | 740 |
| cut_in | 23,500 | 880 |
| pedestrian_crossing | 16,300 | 1,060 |
| cyclist_crossing | 19,200 | 1,080 |
| door_opening | 27,000 | 1,330 |
| stalled_vehicle | 16,700 | 710 |
| two_wheeler_filtering | 25,600 | 1,100 |
| jaywalker_behind_bus | 14,800 | 520 |
| wrong_way_scooter | 22,900 | 1,110 |
| bus_pulling_out | 15,200 | 800 |
| stray_cattle | 28,900 | 640 |
| handcart_kerb_lane | 5,500 | 180 |
| autorickshaw_cut_in | 18,800 | 910 |
| monsoon_waterlogged | 25,900 | 760 |

So **about 200 to 1,300 evaluations per second on one CPU core**, against roughly one run per second on the fake
CARLA with a minimal rig (about 4 s with a labelled camera: [rare cases](rare-cases-mil.md)); a real server's
scenario run takes longer still and was not timed here. The criticality — exact box geometry from
`core/criticality.py`, on the CPU — takes about 95 % of the time; the rollout is cheap.

**The GPU does not help, as built.** `--device cuda` moves the *rollout* to torch; the criticality stays on the CPU.
Measured rollouts per second: `lead_brake` 22,800 (CPU) against 7,900 (CUDA) at a batch of 2,048, and 9,200 against
13,500 at 32,768; `cut_in` 22,200 against 7,500, and 8,700 against 15,600. Since the rollout is 5 % of an
evaluation, evaluations per second are the same on either device. A GPU criticality would need the box geometry
rewritten for torch; it was not done, to keep one implementation of the formula.

## The loop

1. **Screen.** `--screen` Latin-hypercube points (default 10,000) plus `--refine` points of the cross-entropy
   method on the cheap objective (2,000); every point's cheap value is kept as the pool.
2. **Rounds of CARLA runs** (`--batch`, default 8). Of each round, the share `--controls` (default 20 %, at least
   one run of any round of two or more) are **random controls**: uniform over the search space, independent of the
   screen. The rest are the pool's most promising points, no two closer than 0.04·√d in unit coordinates.
3. **Fit low → high** from every confirmed pair after each round: Spearman's rank correlation, an isotonic
   regression of the CARLA value on the cheap one (pool adjacent violators), and a Gaussian process on the residual
   over the parameters (`core/search.py`'s). The pool is re-ranked by the probability that the corrected value reaches
   the threshold. (A monotone correction alone cannot change the order; the residual model can.)
4. **Trust.** With at least `--min-pairs` (12) pairs, a rank correlation below `--min-spearman` (0.5) means the
   screen is **not trusted** for this scenario: the rest of the budget goes to a plain search (`--fallback`, default
   `bayes-pof`) that is told every confirmed run. The verdict stays for the study; `--retrust` judges it again.
   A rank correlation from n pairs is itself uncertain by about 1/√(n − 3) — 0.33 at 12 pairs, 0.45 at 8 — which is
   why trust is not judged earlier (see the measurement below).

A search resumes where it stopped: finished runs are read back (their `fidelity` record says what they were), and
runs another search of the same study made are used as pairs (never as controls). The budget counts all of the
study's finished runs.

## The fidelity gap

`prescreen.json` → `gap` (called `fidelity` before 4.2; a file written then is still read, and its rows'
`fidelity` record too), shown on the section and by `falcon prescreen show`:

| field | what |
|---|---|
| `spearman` | Spearman's ρ of the cheap and the CARLA value over **all** confirmed pairs — what trust is judged on |
| `spearman_controls` | the same over the random controls alone: unbiased, but few points (and not defined when they all tie) |
| `residual_rmse`, `isotonic` | the fitted monotone relationship and its in-sample residual |
| `selected`, `control_failures` | how many of the selected points, and of the controls, failed in CARLA |
| `false_negative` | of the random controls the screen **predicted below the threshold when they were drawn** (the prediction is recorded before the run), how many failed in CARLA: `failed` of `dropped_controls`, `rate`, and `upper95`, the one-sided Clopper–Pearson 95 % bound |

The false-negative risk estimates P(fails in CARLA | the screen would have dropped it) for points uniform over the
search space. It is honest only as far as its sample goes: with *n* dropped controls and none failing, the bound is
1 − 0.05^(1/n) ≈ 3/n — **31 % from 8 controls, 9.5 % from 30**. A 40-run search with 20 % controls cannot show a
risk below about a third; raise `--controls` or the budget when that number matters.

Two things to know about the rank correlation: the selected points stretch the range (most confirmed runs are from
the top of the screen), and where the CARLA landscape is flat outside a small failure region the ranks there are
noise — so ρ over all pairs can read low for a cheap model that still finds failures. A false "not trusted" costs
little: the plain search starts from every confirmed run.

### Why correlation is low

`core.fastsim.diagnose(pairs, min_spearman, min_pairs, rig)` — a pure function of the confirmed pairs, tested on
planted pairs in `tests/test_fastsim_diagnose.py` — reads them for the usual causes, each a measured fact:

| finding | how it is measured |
|---|---|
| too few pairs | fewer than `min_pairs` (12, the trust rule): ρ over so few moves a lot with each run |
| ties / saturation | the largest group of pairs sharing one cheap value (to 6 decimals); named when it is ≥ 25 % of the pairs and at least 3 (heuristic) — at 0 the screen sees no conflict, at 1 a collision; tied values cannot be ordered |
| the worst pairs | the pairs with the largest \|cheap rank − CARLA rank\| (average ranks), the three largest listed with their parameters and runs |
| a parameter | Spearman of \|rank difference\| against each numeric scenario parameter (from 5 pairs); the strongest is named when \|ρ\| ≥ 0.5 (heuristic): the screen misreads that parameter's high (or low) values |
| a tiny camera | a camera of the study's rig under 640 × 360 pixels (heuristic): a stack that sees through it can miss what the kinematic screen — perfect knowledge — assumes it sees |
| an inverted order | ρ < 0: the screen orders the runs against CARLA |

They are leads, not proof: a Spearman over a dozen pairs is itself noisy, and a parameter "named" may be one that
merely co-varies with the true cause.

## Measured: against plain search

**A planted landscape** (`core/search.py`'s planted objective as the stand-in for CARLA: 4 dimensions, 0.4 % of the
space fails; the cheap model: the region displaced and wider, a monotone distortion and noise). 24 "CARLA" runs in
rounds of 6, seeds 0–9 (the acceptance test, `tests/test_fastsim.py`):

| method | seeds with a failure | failures found (per seed) |
|---|---|---|
| two-speed | 10 of 10 | 13–20 |
| Bayesian optimisation, probability of failure | 3 of 10 | 0–6 |
| random | 3 of 10 | 0–1 |

The screen was judged *trusted* in all ten seeds (ρ 0.72–0.92). With a cheap model distorted further (the region
displaced by up to half its radius) two-speed still found failures in 10 of 10 seeds (12–20 each), but the
correlation read between 0.0 and 0.7 and the screen was judged *not trusted* in 8 of them — the effect described
above; the plain search that took over started from the failures already confirmed. With a **decorrelated** cheap model
(its critical region elsewhere) ρ was −0.13 to −0.87 after the second round, the screen was judged not trusted in
every seed, and the plain search took the remaining 12 runs (tested).

**The fake CARLA** (the real scenario director on the fake's road; all failures counted; 16 runs in rounds of 4
with one control each; scenario seed 11; 4,000 screened plus 500 refined; 2026-11-24). Two settings, by how rare a
failure is under the cheap model over the feasible parameters:

| scenario, failure at | share failing | search seeds | method | seeds with a failure | failures found (per seed) |
|---|---|---|---|---|---|
| `cut_in`, criticality ≥ 0.7 | 0.4 % | 0–5 | two-speed | 6 of 6 | 7–12 |
| | | | Bayesian optimisation, probability of failure | 3 of 6 | 0–2 |
| | | | random | 0 of 6 | 0 |
| `lead_brake`, criticality ≥ 0.75 | 2.7 % | 0–7 | two-speed | 8 of 8 | 12 |
| | | | Bayesian optimisation, probability of failure | 8 of 8 | 2–7 |
| | | | random | 2 of 8 | 0–3 |

Read honestly. Where failures are rare (`cut_in`) the two-speed search found them in every seed and plain search in
half; where they are less rare (`lead_brake`) Bayesian optimisation also found one in every seed, and the gain is in
how many: every one of the 12 selected runs failed, in every seed. The 4 random controls of each search never failed,
so the false-negative bound of each search was "0 of 4 dropped controls failed, at most 53 %" — four controls say
almost nothing, as the section above warns. The screen was judged trusted in all 14 searches (ρ 0.62–0.97). An
earlier run of the `cut_in` rows judged trust from 8 pairs (the default then): two of the six seeds were judged
*not trusted* with ρ below 0.5 at 8 pairs — their final ρ was 0.72 and 0.76 — and fell back to plain search (they
still found 5 and 7 failures). That is why the default is now 12 pairs. The fake has the director's kinematics and
no more: this is the best case for the fast simulator, not a result about a real CARLA server.

## Outputs

An ordinary study, `<workspace>/studies/<name>/`:

| file | holds |
|---|---|
| `results.jsonl` | one row per finished CARLA run, as any search's (values, run folder, criticality, RSS label, objective, `fidelity`: the run's level), plus `screen`: `{low, role: top \| control \| fallback, round, predicted, p_fail, driver}` (before 4.2 this record was `fidelity`) |
| `search-prescreen.json` | the search result in the searches' format (`method: prescreen`, history, failures, threshold, space) plus `gap` (the fidelity gap), `fallback`, `fidelity` (core.fidelity.combine over the confirmed runs) and `screen_fidelity: kinematic` — what the study report, the safety case's failure evidence, the ODD coverage and the failure clusters read |
| `prescreen.json` | `falcon.prescreen/1`: the pairs (cheap, CARLA, role, round, the prediction made before the run, the run, its fidelity), the fit, the trust word, the fallback, the failures (each with its run's fidelity), the throughput measured in this run, `fidelity` and `screen_fidelity` |
| `runs/<key>/` | the CARLA runs: replayable, exportable, browsable like any study's |

`--on cluster` sends the CARLA runs to the cluster's nodes ([cluster](cluster.md)); the screen and the study run
here. A scenario or world made from dashcam footage never leaves this machine: refused before anything is queued.

## Limits

- The fast simulator is a kinematic model with the director's lane follower (or IDM) on a straight road. It does
  not run a driving stack, a learned policy or a perception model: for those it is a *driver model*, and only the
  confirmed pairs say whether it ranks their failures. The perception objectives are refused.
- It was compared with the fake CARLA only (above). No real server ran for this feature.
- Trust is one number over few pairs; it can be wrong both ways (see *The fidelity gap*). The false-negative bound is
  wide unless many controls are run.
- The selection assumes the cheap value and the residual vary smoothly over the parameters; a failure mode the cheap
  model has no notion of (a perception miss, a physics effect, a map feature) is found only by the random controls,
  at the rate random search finds it.
- Choice parameters (a vehicle kind) are screened as groups of the batch. The weather is not searched by this
  command (the fast simulator has none); a search with weather dimensions is `falcon search`.
