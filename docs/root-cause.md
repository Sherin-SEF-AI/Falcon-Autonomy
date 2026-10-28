# Root cause: why did it fail?

Falcon attributes a failure to its factors in three ways, each written as `rootcause.json` (with Falcon's provenance
block) and `run.json` in its folder (`workspace/rootcause/<time>-<what>/` from the pages), each reportable
(`falcon report rootcause FOLDER`). Code: `falcon_autonomy/core/rootcause.py` (its docstring is the reference).

| attribution | of | how | what it can say |
|---|---|---|---|
| **counterfactual** | one failing scenario run: a missed object, a failing case of a search, a critical run | re-runs that change one factor at a time, everything else held | whether removing a factor fixes the failure (necessary) and whether it alone still fails (sufficient) — in this case |
| **static** | a whole evaluation | the misses in its predictions related to the objects' and frames' attributes | correlations, with intervals — not causes |
| **episode** | a Leaderboard infraction | the episode driven again with one group of the policy's inputs hidden at a time | which inputs the failing decision depended on — in this episode |

## Counterfactual: one case, one factor at a time

Scenario runs in kinematic motion with no background traffic repeat **bit for bit** on the same server
([scenarios](scenarios.md#bit-for-bit-repetition--measured)), so a difference between a re-run and the case is the
changed factor's doing. Falcon re-runs:

1. **the control** — the case again, unchanged: it must fail again (and its trajectories are compared with the
   case's). If it does not, nothing is attributed: the record says *not reproduced* — re-runs of a failure that does
   not repeat would compare noise.
2. **each present factor removed**, alone;
3. **every present factor removed** together;
4. **each factor kept alone**, the others removed (with two or more factors present; `--no-sufficiency` skips them).

| factor | present when (heuristic thresholds) | removed by |
|---|---|---|
| occlusion | the target is hidden ≥ 10 % of its box in the failure frames | removing the scenario actors that hide it (their boxes over its box, nearer) from the scenario, with their timeline actions. An occluder that is part of the map, not labelled (the ego's own body) or that a timeline trigger depends on is named and not moved |
| range | the target is ≥ 10 m away in the failure frames | starting it half that range nearer along its lane (about half as far at the failure moment) |
| lighting | the sun is below 15° | raising it to 45° |
| fog | fog density ≥ 5 | fog density 0 |
| rain | precipitation, puddles or wetness ≥ 5 | all three 0 |
| appearance | the target is a scenario actor | its blueprint swapped for the next two of its kind (`variant` 1, 2: adults with adults, children with children); fixed when either swap fixes it |
| sensor faults | the rig carries injected faults | removing every fault |

The **failure frames** are the given frame (a missed object), the target's critical window (a perception failure:
[model-in-the-loop](rare-cases-mil.md)), or the critical actor's last 3 s before its critical moment (a critical run).

**Failure tests** (`--failure`): `perception` — the target actor's perception failure (or the run's chosen metric)
≥ threshold; `missed` — the target object not detected at the given step (IoU 0.5, score 0.5); `criticality` ≥
threshold; `both`; `scenario` — its pass criteria failed. A model scores every re-run **as the case itself** (the same
frame identity: a seeded test model draws the same numbers for the same frame; a real model sees only the images).
The target is a scenario actor (its CARLA id differs between runs, so it is followed by its scenario id) or a track id;
by default the worst-perceived actor (perception) or the critical actor.

The table, per factor: *in this case* (what the case has), *removed by*, *removing it alone fixes it* (necessary,
but-for), *it alone still fails* (sufficient here), and the verdict — necessary and sufficient; necessary only;
sufficient only (another factor also produces it: overdetermined); neither. The summary says which removals fix it,
whether removing them all does, or that it still fails with every factor Falcon can change removed (its cause is
something else: the scenario itself, the model on this content).

The wording, on every attribution: **in this scenario, with this model: a single-case attribution, not a general
cause.**

### Measured against the fake CARLA (tests)

A test model blind in fog (it sees every object unless the fog density is 50 or more) on `lead_brake` at fog 80: the
control fails again with identical trajectories; **fog: necessary and sufficient**; range: neither (the lead moved
nearer is still unseen, and nearer without fog is seen); appearance: fixed in 0 of 2 swaps; lighting, rain, faults:
not present; all removed: fixed. Nine re-runs (control, fog, range, two swaps, all removed, three alone), about 33 s on
the fake; run again, every re-run is read back from its cache key. A model that misses a frame only the first time it
sees it gives a control that does not fail: *not reproduced*, nothing attributed.

## Static: correlates of misses over an evaluation

From an evaluation's `predictions.jsonl` (no re-run): every scored ground-truth object is missed or found (the same
matching), and for each attribute — range, occlusion, truncation, box height, class, and the frame's numeric
conditions (fog density, sun altitude, rain, …, scenario values) — the miss rate per quantile bin and **Spearman's rank
correlation** with the miss, each with a 95 % percentile bootstrap interval over **whole drives** (over frames with
one drive, and said to be too narrow; with fewer than five drives, said to be unreliable). Attributes whose interval
of ρ excludes 0 are *clear* and ranked first. Constant attributes say *no variation*.

**Correlational**: attributes move together — range and box height, fog and range in a fogged drive — so a
correlation is not a cause; the counterfactual of a single case tests one. Tested on hand-made predictions with a
planted cause (cars missed beyond 40 m): range is clear (ρ's interval above 0.5), the sun altitude (varying, unrelated)
is not, fog (constant) has no variation.

## Episode: a driving failure by the policy's inputs

For a Leaderboard infraction (`episodes.jsonl` of `falcon policy leaderboard`): the episode is driven again by the
policy in the closed loop — kinematic, no traffic, so it repeats ([training](training.md)) — once unchanged (the
control: the same infraction must recur; its actions are compared with the logged ones), and once per input group with
that group of the observation hidden:

| group | features | hidden as |
|---|---|---|
| its own motion | speed, acceleration, last action | held at the episode's start |
| target speed | target_speed_mps | held |
| lane | lateral offset, heading error, lane width | held |
| curvature | curvature at 10, 25, 50 m | held |
| the lead vehicle | lead_* | "nothing there" (present 0, gap 60 m, closing 0, TTC 10 s) |
| pedestrians and cyclists | vru_* | "nothing there" (present 0, ahead 40 m, TTC 10 s) |
| traffic lights | tl_* | "unknown" (known 0, distance 60 m) |
| route progress | route_progress, route_remaining_m | held |

A group whose hiding prevents the infraction is one the failing decision depended on. Policies read ground-truth
features, so weather and lighting cannot change what they see (said so); sufficiency is not tested for inputs (a
policy blind to all but one input is not a driving policy); a CNN policy's BEV input is not hidden in this release.
Tested in a small closed loop with a planted bug (a policy that tailgates a lead car within 40 m): hiding the lead
prevents the collision, hiding any other group does not.

## Pages and command

- **Rare cases** ▸ a failing case ▸ *Why did this fail?*: the case's counterfactual attribution with the search's own
  objective, model and threshold, as a job; the table opens when it ends (*Report…* makes its HTML report).
- **Models** ▸ *Failures* ▸ a frame ▸ *Why did this fail?*: its first missed object, at that frame (`missed`), when the
  drive is a repeatable scenario run; for any other drive (a traffic drive: CARLA's traffic manager drives through
  physics, which does not repeat) it says why and shows the evaluation's correlates instead. *Correlates of misses*:
  the static attribution of the evaluation.

```sh
falcon why case PROJECT --run RUN --out DIR [--failure perception|missed|criticality|both|scenario] [--model ID]
                [--actor ID | --track N] [--step S] [--threshold 0.5] [--metric …] [--window 3] [--lead 3]
                [--no-sufficiency] [--fake-carla DIR]
falcon why evaluation EVAL_DIR --out DIR [--boot 1000] [--seed 0] [--bins 4]
falcon why episode PROJECT --episodes EPISODES.jsonl --episode N [--infraction K] --policy DIR|module:callable
                --out DIR [--world W] [--rig R]
```

Exit codes: 0 done; 2 failed; 6 refused — a run that cannot be re-run exactly (not a scenario, physics motion,
traffic, a log replay), a case that does not fail its own test, a model that cannot be run.

## Limits

- A counterfactual changes what it names **and what comes with it**: a removed bus takes its shadow; a nearer
  pedestrian meets the ego at another time (triggers fire on distances); a swapped blueprint is a few blueprints, not
  the class.
- On a real server the motion repeats bit for bit but the images only to within a few levels (exposure, motion blur),
  which can flip a borderline detection: the control says whether the failure repeated.
- Factors Falcon cannot change are not tested: the map, the road surface, the model's training.
- Log replays and traffic drives are not attributed by re-runs in this release (the static attribution covers their
  evaluations).
