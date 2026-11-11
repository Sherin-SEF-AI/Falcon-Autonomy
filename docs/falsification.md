# Falsification of STL requirements

A **requirement** is a signal-temporal-logic formula over what a run recorded — "the time to collision never drops
below 1.5 s", "the post-encroachment time stays above 1 s", or one of the [traffic rules](traffic-rules.md). A
**falsifier** searches a scenario's parameters for a run that *violates* it. Instead of a pass/fail verdict per run it
uses the requirement's **robustness** ρ — how far the run was from violating it (ρ > 0) or how badly it violated it
(ρ < 0) — so a searcher can climb towards the violation instead of guessing. Code: `falcon_autonomy/core/stlobjective.py`
(the objective, the commands), `core/rules.py` (`robustness()`, the signals), `core/stl.py` (the semantics, the soft
minimum, the deciding time), `core/search.py` (the searchers, `cem` among them); their docstrings are the reference.

## Where it shows

| where | what |
|---|---|
| Rare cases ▸ *Falsify* | the requirement typed with its parse error (and column) as you type, or a rule of a set, or a saved one; scenario, method, budget, scale, smoothing, near-miss margin; *Falsify* queues the study as a job; the best ρ, the falsified runs (ρ < 0), the near misses (0 ≤ ρ < margin), the runs whose ρ was unknown, and the chosen run's ρ(t) with the time that decides it |
| `falcon falsify run PROJECT --scenario S (--stl TEXT \| --rule SET:ID \| --name N) [--method cmaes\|cem\|bayes\|bayes-pof\|adaptive\|random] [--budget N] [--batch N] [--scale S] [--smooth β] [--margin M] [--world W] [--rig R] [--seed N] [--weather FIELD] [--fps F] [--fake-carla DIR]` | a resumable study `studies/falsify-<scenario>-<key>`; its `falsify.json` at the end |
| `falcon falsify show STUDY` · `falcon falsify check` | the result's headline (or `--json`); the planted-landscape comparison below (no CARLA) |
| `falcon stl compile TEXT \| --rule SET:ID [--save NAME --project P]` | the normal form, signals, horizon and key; `--save` writes it to `safety/formulas.json` |
| `falcon stl robustness RUN… (--stl TEXT \| --rule SET:ID \| --name N --project P) [--series] [--json]` | ρ of recorded runs, the deciding time, the share evaluated, the signals not recorded |
| `falcon search … --objective robustness --stl TEXT` | the same objective in an ordinary rare-case search (and `falcon sweep`) |
| `falcon failrate run … --judge stl --stl TEXT` | a failure rate whose failure is the requirement's violation ([failure rate](failure-rate.md)) |
| the loop's ODD `search: {objective: robustness, stl: …}` | the loop's search step falsifies the requirement in each weak cell |
| the assistant | `formal_stl_check`, `formal_stl_compile`, `formal_stl_robustness`, `formal_falsify_stl` ([assistant tools](assistant-tools.md)) |
| [the safety case](gsn.md) | a falsification study as evidence: `min_robustness ≥ 0` (not falsified), `falsified_runs ≤ k` |

## The robustness and the search's value

ρ is core.stl's quantitative semantics (Fainekos & Pappas 2009; Donzé & Maler 2010) on the run's signals: a formula's
ρ at the run's first sample (so write `G …` for "at every sample"), or for rules the lowest ρ over the known samples,
as `falcon rules check` reports them. ρ is in the predicate's units (seconds for a TTC, metres for a gap); its sign is
the verdict (ρ = 0 counts as satisfied). The search maximises

    value = σ(−ρ / scale) = 1 / (1 + exp(ρ / scale)),   clamped to the exact verdict's side of 0.5,

so **ρ < 0 ⇔ value > 0.5** and ρ ≥ 0 ⇔ value < 0.5 — exactly, even for ρ = 0 or a ρ so small that σ rounds to 0.5.
The failure threshold 0.5 that the safety case, the ODD coverage, the failure rate and the Rare cases page read then
means "falsified", and higher = worse as for every objective. `scale` (default 1) only shapes the landscape a
model-based searcher fits (Bayesian optimisation); the rank-based ones (CMA-ES, the cross-entropy method) see the same
order whatever it is.

**Smoothing** (`--smooth β`, optional): the searcher sees core.stl's `evaluate_smooth` — every minimum (`and`, `G`,
`H`) replaced by the soft minimum −1/β·log Σ exp(−β·x), every maximum by the soft maximum — which removes the kinks
of min and max. It is an **approximation** (a soft minimum lies below the minimum by up to log(k)/β over k terms;
`until`/`since` stay exact; unknown samples are left out rather than treated three-valued), so the value is still
clamped to the exact ρ's side of 0.5 and **the exact ρ is what every row, verdict and report states**.

**Unknown ρ.** A run whose formula reads signals it did not record (traffic-light states in drives, actors in gym
traces) has an unknown ρ: it is not a failure, the searcher is told nothing about it, it counts towards the budget,
and a batch in which every run is unknown stops the study with the signals named — rather than spending the budget.

Every scored run keeps ρ, the value, the verdict, the deciding time (core.stl's `witness`: down through `not`,
the side of `and`/`or` that sets the result and the sample of `G`/`F` windows that does, to a predicate's sample) and
the share of the run evaluated in `results.jsonl` (`stl`), and its ρ(t) series in `stl/<run>-<key>.json`. ρ in JSON is
a number, `"inf"`/`"-inf"` (a requirement nothing in the run can violate; a collision under a DRAC bound) or `null`
(unknown).

**The study key** hashes the formula's normal form (or the rule set's SHA-256 and the rule ids), the scale, β and the
signals' version: another requirement re-scores a study's recorded runs (nothing is driven again) and never reuses
another's scores.

## The signals added for requirements

Beside the rule signals ([traffic rules](traffic-rules.md#the-signals)), six surrogate-safety signals computed by
core.criticality per sample over the other recorded actors (vehicles and pedestrians; unknown when none are recorded):
`ttc_lead_s`, `ttc_min_s`, `drac_mps2`, `pet_s`, `lat_clear_m`, `lon_gap_m` — their definitions are in the traffic-rules
signal table. Examples:

```
G (ttc_min_s >= 1.5)                                   no time to collision below 1.5 s
G (drac_mps2 <= 3.35)                                  never need more than Archer's 3.35 m/s² to avoid a crash
G (pet_s >= 1.0 or speed_mps < 1)                      no post-encroachment below 1 s while moving
G (lat_clear_m >= 0.5) and G (lon_gap_m >= 2)          clearances
G (lead_gap_m < 40 -> F[0,2] (ttc_lead_s >= 3))        a short TTC to the lead is recovered within 2 s
```

## The searchers on a planted landscape

`falcon falsify check` (and `tests/test_stlobjective.py`) runs the searchers on a synthetic robustness landscape —
not a model of any stack: three parameters in the unit cube, ρ = d − 1 with d the ellipsoidal distance from a planted
centre (ρ < 0 inside: 0.115 % of the cube) plus deterministic noise of 0.02 — once on the robustness value and once on
the binary verdict (1 falsified, 0 not), which is all a search without robustness has. 100 evaluations, seeds 0–9,
measured on the reference machine on 2026-11-12:

| method | on the robustness: seeds that reached ρ < 0 | median evaluations to it | on the binary verdict |
|---|---|---|---|
| CMA-ES | **10 of 10** | 64.5 | 0 of 10 |
| cross-entropy method (`cem`) | **9 of 10** | 49 | 1 of 10 |
| Bayesian optimisation (`bayes`) | **10 of 10** | 21 | 2 of 10 |
| random (baseline) | 1 of 10 | — | 1 of 10 |

Random sampling finds a region of 0.115 % in 100 evaluations with probability 1 − (1 − 0.00115)¹⁰⁰ ≈ 11 %; on the
binary verdict the searchers have no gradient to follow and do no better. The cross-entropy method's defaults (a
generation of max(20, 5n), elite 10 %, smoothing 0.7, σ0 = 0.5) were chosen on this landscape: σ0 = 0.3 found it in 4
of 10 seeds, generations of 10 or 14 in 6–8.

## Outputs

In `studies/<study>/`: `study.json` (the objective with the requirement), `results.jsonl` (each run with its `stl`
record), `stl/<run>-<key>.json` (ρ(t), the smoothed series when asked), `searcher-<method>.json` (the searcher's state:
a stopped study resumes exactly), `search-<method>.json`, and `falsify.json` (`falcon.falsify/1`): the requirement, the
best ρ, the falsified runs and the near misses with their parameters and runs, the runs whose ρ was unknown and the
signals they missed, provenance.

## Limits

- Robustness mixes units across predicates of one formula (the usual STL caveat): `G (ttc_min_s >= 1.5 and
  lat_clear_m >= 0.5)` compares seconds with metres. Prefer one unit per requirement, or a scale per study.
- A run is falsified only at the recorded rate; an event between samples is seen through the samples around it.
- `pet_s` is non-causal (it reads the whole run), `ttc_*` extrapolate at constant velocity and yaw, boxes are the
  recorded extents (core.criticality's limits).
- A falsifier finds violations; not finding one within a budget is not a proof that none exists — that is what the
  [failure rate](failure-rate.md) (`--judge stl`) bounds, under its stated assumptions.
- The planted landscape checks the searchers, not any stack.
