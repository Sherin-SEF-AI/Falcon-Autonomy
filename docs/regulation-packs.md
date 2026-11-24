# Regulation test packs

> **These packs are an engineering pre-check in simulation. They are not a type-approval, not an official Euro NCAP
> rating and not a conformity assessment, and Falcon Autonomy is not a qualified tool.** The vehicles, sensors and
> brakes are simulated and the scenarios are Falcon's rendering of the published procedures. The same sentence is
> printed in every result, report and assistant answer.

A **pack** is a ready-made scenario suite scored after a published test procedure: its test cases, each with the
clause it comes from, the Falcon scenario it runs and its pass criteria, and the rule that turns the cases into a
verdict or score. A pack is *data* — `falcon regpack show PACK` prints it — and every number that comes from a document
carries its document, clause and **whether it was verified**. Code: `falcon_autonomy/core/regpacks.py`.

## Where it shows

| where | what |
|---|---|
| Reports ▸ *Regulation packs* | the notice; the packs with their cases, how many run and how many values are unverified; the case table with each case's clause, its result as a word with its colour, a mark on every case that uses an unverified value, and what was measured; *Run the pack* (a job) and *Export the report* into the reports folder |
| `falcon regpack list` · `falcon regpack show PACK [--json]` | the packs; one pack's cases, values, sources and which values are unverified |
| `falcon regpack run PROJECT PACK --name N [--case ID…] [--group G…] [--driver follower\|autopilot] [--degraded-mode] [--world W] [--rig R] [--seed N] [--fps F] [--duration-scale X] [--fake-carla DIR]` | run the cases as stack runs and score them: `regpacks/<name>/regpack.json` and `report.html` |
| `falcon regpack report DIR [--out FILE.html]` | the result as text and as a self-contained report |
| `falcon regpack export PACK --out DIR` | the scenarios as Falcon JSON and OpenSCENARIO 1.2, and a `falcon.suite/1` suite naming them |
| `falcon report regpack DIR --out FILE` | the same report through the [reports](reports.md) command |
| the assistant | `formal_regpack_run` (act), `formal_regpack_show` (read) |
| [the safety case](gsn.md) | evidence kind `regpack`: the `verdict` (equals `passed`) or a `field` such as `score.share >= 1` |

## Verified, unverified, and Falcon's own

Every value record is `{value, unit, source, clause, verified, basis, note}`:

- **verified** (`basis: text`) — read as a number in the document's text.
- **not verified** (`basis: figure`) — read from a chart or figure of the document by measurement, not printed there
  as a number. Flagged in `show`, in every result (`unverified`), in the report's own section and on each case that
  uses it.
- **not verified** (`basis: recalled`) — recalled, not checked against the document.
- **Falcon's choice** (`source: falcon`) — where the document leaves a parameter to the test laboratory or says
  nothing. Never presented as the document's; each carries its reason, and a report lists them separately.

The unverified values today:

| pack | value | why |
|---|---|---|
| `r157` | `pm_dy0` = 1.6 m, the initial lateral gap of the performance model's cut-in | an example inside Figure 6 (lane 3.5 m, vehicle 1.9 m), not normative text |
| `r157` | `pm_vy_max` = 3.0 m/s | the upper end of the data sheets' lateral-velocity axis, read from the figures |
| `ncap_aeb_c2c` | the impact-speed colour bands of every test speed except 50 km/h | the assessment protocol prints them only as a bar chart; they were measured from it. The 50 km/h row is a text table and agrees |
| `ncap_aeb_c2c` | which side a negative overlap is on | recalled; the grid is run mirror-symmetrically, so the score does not depend on it |
| `iso34502` | that ISO 34502 Annex B holds the same 24-combination matrix as the JAMA framework | recalled; the annex body is paywalled and was not read |

## Sources

Read in November 2026. UNECE's and Euro NCAP's own servers refused automated access, so the documents were read from
archived copies of the official files, as noted.

| document | version, date | read from |
|---|---|---|
| UN Regulation No. 157 — Uniform provisions concerning the approval of vehicles with regard to Automated Lane Keeping Systems | E/ECE/TRANS/505/Rev.3/Add.156, original series, 4 March 2021 | an Internet Archive copy of the UNECE PDF (R157e.pdf) |
| UN Regulation No. 157, 01 series of amendments (consolidated) | Add.156/Amend.4, 3 March 2023 | a mirror of the UNECE PDF, cross-checked against ECE/TRANS/WP.29/2022/59/Rev.1 |
| Euro NCAP Test Protocol — AEB Car-to-Car systems | v4.3, December 2023 | an Internet Archive copy of the Euro NCAP PDF |
| Euro NCAP Assessment Protocol — Safety Assist — Collision Avoidance | v10.4, December 2023 | an Internet Archive copy of the Euro NCAP PDF |
| Euro NCAP Test Protocol — Lane Support Systems | v4.3, December 2023 | an Internet Archive copy of the Euro NCAP PDF |
| ISO 34502:2022 Road vehicles — Test scenarios for automated driving systems — Scenario based safety evaluation framework | first edition, November 2022 | ISO's catalogue page and public sample (cover, contents, introduction) only |
| JAMA Automated Driving Safety Evaluation Framework | Ver 3.0, December 2022 | an Internet Archive copy of the JAMA PDF |

Not read: amendments 1–3 to R157's original series and supplements after the 01 series; Euro NCAP's 2026 protocols
(Crash Avoidance — Frontal Collisions, Lane Departure Collisions) and assessment protocol v10.4.1; the body of
ISO 34502; JAMA Ver 4.0 (March 2026).

## `r157` — UN Regulation No. 157 (ALKS), up to 60 km/h

The provisions the original series and the 01 series share for systems limited to 60 km/h.

| value | from |
|---|---|
| maximum operational speed 60 km/h (130 km/h in the 01 series, with an MRM lane change) | 5.2.3.1 |
| minimum following distance d = v · t_front; t_front 1.0 s at 7.2 km/h, 1.1 at 10, 1.2 at 20, 1.3 at 30, 1.4 at 40, 1.5 at 50, 1.6 at 60, linear in between; never below 2 m | 5.2.3.3 |
| cut-in to be avoided when the lateral movement was visible for ≥ 0.72 s and TTC_LaneIntrusion > v_rel / (2 · 6 m/s²) + 0.35 s | 5.2.5.2 |
| lane intrusion: the front wheel 0.3 m beyond the outside edge of the lane marking | 5.2.5.2 |
| crossing pedestrian: lateral speed up to 5 km/h | 5.2.5.3 |
| a deceleration demand above 5.0 m/s² is an emergency manoeuvre | 5.3.1.1 |
| minimum risk manoeuvre: deceleration demand not greater than 4.0 m/s² | 5.5.1 (5.5.2 in the 01 series) |
| MRM earliest 10 s after the transition demand; immediately on a severe failure | 5.4.4.1, 5.4.4.1.1 |
| lane keeping test: at least 5 minutes | Annex 5, 4.1.2(a) |
| lead vehicle deceleration of at least 6 m/s² to standstill | Annex 5, 4.3.2(f) |
| performance model 1: wander zone 0.375 m, perception 0.4 s, delay 0.75 s, 0.6 s to 0.774 G; data sheets at a 2.0 s headway | Annex 4 Appendix 3 (01 series: Annex 3) |

**Cases** (22): lane keeping behind a lead (4.1); a stationary car at 20, 40 and 60 km/h and a pedestrian crossing at
5 km/h (4.2); following a lead that slows to half and recovers, judged against the 5.2.3.3 distance, and a lead braking
at 6 m/s² (4.3); six cut-ins (4.4) and three cut-outs (4.5); the minimum risk manoeuvre after a lidar dropout (5.5,
through `core.mrm` with the 4.0 m/s² limit).

**The regulation's driver model decides which cut-ins must be avoided.** `pm1_cut_in_avoidable()` implements
performance model 1 — the "competent and careful human driver" — from its parameter table: the other vehicle moves
sideways from t = 0; the risk is perceived when its centre has left the 0.375 m wander zone; after 0.4 s of perception
and a 0.75 s delay the driver brakes, the deceleration ramping over 0.6 s to 0.774 G. A cut-in this driver avoids —
or that 5.2.5.2 itself requires — is **mandatory**; one neither covers is still run and reported as **informative**:
a collision there is not a failure. The same model classifies the cut-outs and the lead braking. It reproduces the
regulation's own statements about its data sheets (a lead braking at 1.0 G from a 2.0 s headway is avoidable at
every speed up to 60 km/h); the data sheets were not reproduced point by point.

**Annex 5 gives no numeric test parameters** — "the specific test parameters for each test shall be selected by the
Technical Service" — so the speeds, gaps and lateral velocities of the cases are Falcon's selection inside the
regulation's ranges, and are labelled as Falcon's choices.

**Verdict**: `passed` only when every mandatory case of the whole pack ran at its full duration and passed; `failed`
when a mandatory case failed; otherwise `incomplete`. *In this build the verdict is never better than incomplete*: the
scenario runner's ego route is 400 m of one lane, so the 5-minute lane keeping test cannot run at its duration — it is
refused by name, or run shortened with `--duration-scale` and marked "NOT the document's duration".

Not in the pack: the transition demand and driver monitoring, HMI, the field-of-view tests, powered-two-wheeler
targets, curves, lane changes (01 series), performance model 2 (the 01 series' fuzzy model).

## `ncap_aeb_c2c` — Euro NCAP AEB Car-to-Car 2023

| value | from |
|---|---|
| CCRs: stationary target; AEB 10–50 km/h in 5 km/h steps (55–80 km/h are FCW tests) | test protocol 8.2.2.1 |
| CCRm: target at 20 km/h; 30–80 km/h in 5 km/h steps | 8.2.2.2 |
| CCRb: both at 50 km/h; headways 12 and 40 m; target decelerations 2 and 6 m/s² | 8.2.2.3 |
| overlaps −50, −75, 100, 75, 50 %; per speed (−50 + −75 + 2 × 100 + 75 + 50) / 6 | assessment protocol 3.3.2 |
| colour factors: green 1.0, yellow 0.75, orange 0.5, brown 0.25, red 0 | 3.3.2 |
| bands at 50 km/h (CCRs, CCRb): green < 5, yellow < 15, orange < 30, brown < 40 km/h impact speed | 3.3.2 (text table) |
| points per speed: CCRs 1, 2, 2, 2, 2, 2, 1, 1, 1 (10…50 km/h, total 14); CCRm 1 × 7 then 2 × 4 (30…80 km/h, total 15); CCRb 4 × 1 | 3.3.2 |
| full avoidance needed up to and including 20 km/h | 3.3 |
| each scenario 1 point (AEB); test start T0 at TTC 4 s | 3.3.2; test protocol 4.1.1 |

**Cases** (110): 45 CCRs and 55 CCRm grid points, 4 CCRb, and the 6 FCW rows — refused by name, because FCW tests need
the protocol's braking-robot driver (it reacts 1.2 s after the warning with −4 m/s²) and Falcon simulates no warned
driver. Each grid point is scored by the colour band of its impact speed (CCRm: the relative impact speed), read from
the run's trajectories at the step before the first contact.

**Score**: per scenario the weighted share of its points; the pack total out of **3** — the AEB points of CCRs, CCRm
and CCRb, of the 9 that AEB Car-to-Car carries (FCW, turn-across-path, crossing, head-on and HMI are not assessed). A
scenario whose grid was not run completely has **no score**: the verdict is `partial`, never an extrapolation.

Differences from the protocol that matter: the target is Falcon's stand-in car, not the Global Vehicle Target (the
overlap offset uses both widths); there is no manufacturer prediction, hence no correction factor and no 2 km/h
verification tolerance; the test tolerances (speed +1 km/h, lateral 0 ± 0.05 m) are not checked on the runs.

## `ncap_lss` — Euro NCAP Lane Support Systems 2023 (data only)

The grid and its limits: 72 km/h; lateral velocities 0.2–0.6 m/s in 0.1 m/s steps; LKA on dashed and solid lines with
a limit of −0.3 m distance to the lane edge; ELK road edge −0.1 m, solid line −0.3 m, oncoming and overtaking vehicle
"no impact"; lane 3.5–3.7 m; LKA 0.5, ELK 2.0, HMI 0.5 points (test protocol 6.1.3.1, 7.2–7.4; assessment protocol
4.3).

**No case runs.** The test steers the vehicle onto a lane-departure path — a yaw angle per lateral velocity (0.57° at
0.2 m/s … 1.72° at 0.6 m/s) — and Falcon's scenario language has no action that imposes a path or a steering
disturbance on the ego. Every case is refused by name with that reason; the pack is the data and the scoring rule,
ready for when the runner can.

## `iso34502` — a coverage checklist, not a test

ISO 34502:2022 is paywalled. From ISO itself come only the title, the scope ("limited access highways") and the annex
titles — B *Traffic-related*, C *Perception-related*, D *Vehicle control related critical scenarios*. **The breakdown
inside each category is the public JAMA framework's (Ver 3.0), labelled as that**, not ISO's text:

- traffic: road geometry (straight, merging, branching) × ego behaviour (going straight, lane change) × the other
  vehicle's motion (accelerating, decelerating, cut-in, cut-out) = 24 motorway combinations;
- perception: factors of the vehicle and sensor, of the surrounding environment, of the perception target; blind
  spots; connectivity;
- vehicle control: body inputs (road shape, wind) and tyre inputs (road surface, tyre condition).

Each item is mapped to the Falcon scenarios or features that touch it and marked **covered**, **partial** or **gap**,
with the reason. Today: 3 of the 24 traffic combinations are covered (a decelerating lead, a cut-in, a cut-out, all on
a straight road with the ego going straight); the others are gaps because the scenario ego does not change lanes and
no scenario is placed on a merging or branching section. Nothing is run and nothing passes: "covered" means a scenario
of that kind exists, not that its parameter space was explored.

## How a case is run and scored

Each runnable case is one stack run (`core/stack.py`, the worker `workers/stack.py`) through
`sim.service.study_service`, with the chosen driver as the system under test — Falcon's built-in follower, CARLA's
autopilot; for a ROS 2 stack, `falcon regpack export` writes the scenarios and a suite for `falcon stack`. A case the
world or backend cannot run (no adjacent lane for a cut-in, a route too short) is **refused by name** with the reason.
Scoring reads the run's own results:

| signal | from |
|---|---|
| collision, its time, with whom | the run's criticality summary ([criticality](scenarios.md)) |
| the ego's and the relative speed at the first contact | `trajectories.json`, the step before the boxes overlap |
| the bumper gap to the lead against d_min(v) | `trajectories.json`; judged only while the lead has been steady for 2 s (Falcon's reading of "restore") |
| how far the body went beyond its lane | the recorded lateral offset from the lane centre, the lane's and the vehicle's width |
| the minimum risk manoeuvre | `core/mrm.py`'s verdict with the deceleration limit set to the regulation's 4.0 m/s² (its reaction, standstill, jerk and lane limits are Falcon's heuristics) |

## Limits

- A simulation pre-check: see the notice at the top. A pass says the stack handled Falcon's rendering of the case.
- The scenarios run on straight lanes of at most 400 m; no curves, no long tests.
- Runs were made on the fake CARLA only (it tests the integration, not a stack): no pack has been run on a real CARLA
  server yet.
- One seed per case; no repetition statistics.
- `core/ci.py` is unchanged: a pack reaches CI as the exported `falcon.suite/1` suite (its scenarios added to the
  project), whose pass/fail is then the suite's no-collision check, not the pack's own criteria.
