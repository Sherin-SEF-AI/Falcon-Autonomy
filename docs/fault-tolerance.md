# Sensor fault and calibration-drift tolerance

[Sensor faults](stack-testing.md) inject a fault of a chosen size and check the reaction. A **tolerance campaign**
asks the other question: *how large can each fault get before perception or the stack fails?* For every (sensor,
fault) it bisects the severity at which a criterion flips from pass to fail and writes a **tolerance table** — sensor ×
fault → the largest tolerated severity with its bracket [last pass, first fail] — the most fragile sensor, and what the
tolerances imply for recalibration and time synchronisation. Code: `falcon_autonomy/core/faulttol.py` (models, search,
campaigns, the command), `core/faults.py` (the fault transforms it reuses, and the two stream faults added for it);
their docstrings are the reference.

## Where it shows

| where | what |
|---|---|
| Stack testing ▸ *Fault tolerance* | a campaign's table: the tolerated severity in mono numerals with its unit, the bracket, the requirement you set for the selected row's fault (kept in the settings), a meter of how much of the tolerance that requirement uses, and the status as a word with its colour (*met*, *not met*, *no requirement*, *not measured* with the reason). Selecting a row draws its response curve — one axis, the threshold dashed, circles pass and squares fail, hover names a point. Below: the most fragile sensor and the derived calibration statements, one compact row per sensor (how they are derived behind *Details*). The campaign wears its fidelity chip — closed loop: its runs' level; offline: *Recorded · of <level> runs*. The failed or missing checks are flags with next steps (below). *Where* picks the world and rig of the project; *Run a closed-loop campaign* queues one there; a recorded run — sent with *Send to ▸ Fault tolerance* from any run row, or chosen — is analysed offline with *Analyse the recorded run* (a job; nothing starts without the click) |
| `falcon faulttol models [--json]` | the graded fault models: unit, largest severity tried, sensors, what is modelled, and which are offline only |
| `falcon faulttol run PROJECT --name N --runs RUN… [--model ID] [--sensor ID…] [--fault F…] [--require FAULT=SEVERITY…] [--ap-drop X] [--ap-metric AP\|AP50\|AP75] [--recall-drop X] [--retention-drop X] [--lidar-tol M] [--max-frames N] [--grid N] [--steps N] [--seed N]` | an **offline** campaign on recorded runs: no simulator |
| `falcon faulttol run PROJECT --name N --closed-loop --scenario S [--criterion degraded\|criticality\|fail\|stl] [--threshold X] [--stl TEXT] [--seeds 0,1] [--driver follower\|autopilot] [--no-degraded-mode] [--onset S] [--world W] [--rig R] [--fps F] [--fake-carla DIR]` | a **closed-loop** campaign: stack runs through the simulator, the fault starting at `--onset` |
| `falcon faulttol show DIR [--json]` | the table, the most fragile sensor, the derived statements |
| the assistant | `research_faulttol_run` (act), `research_faulttol_show` (read) ([assistant tools](assistant-tools.md)) |
| [the safety case](gsn.md) | evidence kind `faulttol`: a `field` claim such as `tolerance.<sensor>.<fault>.tolerated >= 0.5` |

Output: `<workspace>/faulttol/<name>/faulttol.json` (schema `falcon.faulttol/1`: every cell with its curve) and
`tolerance.csv`; a closed-loop campaign keeps its runs in `runs/<key>/` (the same run is never repeated).

## The fault models

Each has **one severity in a physical unit**, a stated model, and is deterministic given the seed.

| fault | unit (largest tried) | sensors | the model |
|---|---|---|---|
| `cam_yaw`, `cam_pitch`, `cam_roll` | degrees (5, 5, 10) | pinhole camera | the camera is turned on its mount while the stack keeps the nominal calibration. Offline the recorded image is re-projected by the pure-rotation homography K Rᵀ K⁻¹ — exact for a rotation about the optical centre; what the turned camera would newly see at the image's edge is black |
| `lidar_yaw` | degrees (5) | lidar | the lidar is turned about its vertical axis; its points are delivered in the turned sensor's frame (an exact rotation of the cloud) |
| `lidar_sector` | degrees of azimuth (180) | lidar | no returns in a sector of this width centred straight ahead |
| `clock_offset` | ms (500) | camera, lidar | data captured at t is stamped t + offset, so the stack pairs it with the world of a later moment. Offline a camera gets the nearest recorded frame — the severity is resolved to **one frame period** (50 ms at 20 Hz; the cell's `resolution` says so) — and a lidar's points move by the ego's speed × the offset |
| `dirty_lens` | share of the image (0.5) | camera | opaque dirt blotches (`core.faults.dirt`); one seeded mask for the whole run, growing with the severity |
| `frozen` | seconds (2) | camera | one freeze of this duration at a seeded time; the frames inside repeat the last one before it |
| `dropped` | rate (0.9) | camera, lidar | each frame (a lidar: each 1° azimuth packet) is lost with this probability; the last camera frame is held |
| `noise` | σ in 8-bit levels (50) | camera | Gaussian noise on every channel |
| `exposure` | EV (3) | camera | over-exposure: gain 2^EV in linear light (gamma 2.2 assumed), clipped. Under-exposure is not a model here |
| `gnss_bias` | metres (20) | GNSS | every fix offset north by this distance |

`core/faults.py` already had mount rotation and translation (at spawn), the clock offset of a recorded drive, drop,
freeze, noise, dirt, rain, glare, occlusion, sector loss, packet drop, multipath and dropout. Added for the campaigns:
`extrinsic_drift` (a lidar's points rotated, as a timed stream fault) and `bias` (a constant GNSS offset). The graded
models above are applied **at the data level** — on recorded images, point clouds and stamps offline, and on the data a
stack receives in closed loop — so they work on the fake CARLA and on recorded runs alike.

**Common random numbers.** Every severity of a cell uses the same seed, the same frames (or the same scenario seeds)
and *nested* draws: a frame dropped at rate 0.2 is also dropped at 0.3; the dirt mask at 10 % lies inside the one at
20 %; the noise field is the same field scaled. The response then differs between severities by the fault alone.

## The criteria

The response is a number; a cell **fails** at a severity when its response exceeds the threshold.

Offline, on recorded runs:

- **`ap_drop`** (with `--model`, a registered detector): the model runs on the faulted images and is scored against the
  labels the *nominal* calibration and stamps give (COCO AP50 by default). The response is the AP without the fault
  minus the AP with it, in points; the default threshold is 5. A detection therefore counts only where the stack's
  calibration says the object is — a detector is not "wrong" about a turned camera's image, the stack's use of it is.
  A plug-in that reads the labels instead of the pixels (the synthetic detector) is flagged `synthetic_model`: image
  faults cannot move it.
- **`recall_drop`** (no model): the share of labelled objects whose box in the delivered image no longer overlaps its
  nominal box with IoU ≥ 0.5 — what a perfect detector would lose — in points. Defined for the geometric and timing
  faults (`cam_*`, `clock_offset`, `frozen`, `dropped`); `dirty_lens`, `noise` and `exposure` change the image, not
  where things are, and are **refused by name** without a model.
- **`retention_drop`** (lidar): the share of actor-tagged points (the recorded per-point actor ids) that are lost or
  land further than `--lidar-tol` (0.5 m) from where the nominal calibration puts them. The clock offset's version is
  a static-scene approximation: the actors' own motion is ignored.
- **`position_error`** (GNSS): the bias against the in-lane margin (lane width − vehicle width) / 2 = 0.8 m for a
  3.5 m lane and a 1.9 m vehicle. Analytic — the bias *is* the error — and marked so; it is there so the table and the
  calibration statement have the row.

Closed loop, through `sim.service.study_service` (stack runs, the same as a [suite](stack-testing.md)'s):

- **`degraded`**: the stack's health monitor declared a failure (the follower's degraded mode) — the size at which the
  test stack stops tolerating the fault and starts its minimal-risk manoeuvre.
- **`criticality`**: the run's criticality, the worst over the seeds, reaches the threshold (0.5).
- **`fail`**: a run failed its scenario's pass checks.
- **`stl`**: a signal-temporal-logic requirement is violated ([falsification](falsification.md)); the response is −ρ.

Closed-loop faults are the ones the stack worker applies to the data a stack receives: `lidar_yaw`, `lidar_sector`,
`dropped` (lidar packets), `gnss_bias`, `dirty_lens`, `noise`. The others are refused by name. **The built-in follower
drives from ground truth**: a fault reaches its behaviour only through its degraded mode, so with it the meaningful
criterion is `degraded`. A ROS 2 stack receives the faulted data itself.

## The search

Per cell: the response at severity 0 — it must pass, or the cell is `fails_nominal` and there is no tolerance to
measure — and on a coarse grid (5 points) up to the model's largest severity. No failure on the grid:
`tolerates_max`, the true tolerance is *at least* that. Otherwise the verdicts must be **monotone** (no pass above a
fail). If they are, the bracket between the last passing and the first failing grid point is bisected (6 steps, or
down to the model's resolution): `bracketed`. If they are not, the cell is `not_monotone` and reports the **lowest
failing severity the grid found**, unbisected — a bisection would be meaningless there, and the cell says so.

The tolerated severity is the bracket's lower end: the largest severity *seen to pass*, not an interpolation.

## Flags and their next steps (4.2)

`core.faulttol.flags(doc, requirements)` lists what a campaign leaves to act on, most severe first: a requirement
**not met**; a cell that **fails without the fault** (the scenario or the criterion fails, not the sensor); a
requirement met with less than 10 % to spare (**no safety factor** is applied anywhere); a cell **not measured** (a
missing check, with its reason); a **non-monotone** response; and the measured faults with **no requirement** (no
verdict at all). The tab shows the first four as callouts, the rest behind *Details*. Each has *Jump to config* (the
row and its requirement field, or the campaign's scenario) and *Attach evidence* (the campaign as a Solution of the
[safety case](gsn.md), claiming `tolerance.<sensor>.<fault>.tolerated >= <requirement>` when one is set); a
requirement flag also has *Run sensitivity check*: `core.faulttol.sensitivity(cell, requirements, rel=0.10)` judges
met / not met at the requirement ±10 % (your number: no document gives it a range) and at both ends of the
tolerance's bracket [last pass, first fail], and says whether the verdict flips. Nothing is run.

## Fidelity (4.2)

A closed-loop campaign records `fidelity` = core.fidelity.combine over its stack runs. An offline campaign is an
analysis of recorded runs: `fidelity` is `recorded` and `source_fidelity` the combined level of the runs it read. A
[safety case](gsn.md) Solution bound to an offline campaign of fake runs verifies amber at best.

## What is derived, not measured

`calibration` in the result restates each tolerance as the accuracy it implies — "cam-front: extrinsic yaw must hold
within ±0.23°; its stamps must be synchronised within 50 ms". Every statement is labelled *derived*: it follows from
this stack, these runs or scenarios, this criterion and these fault models. It is not a general requirement, and no
safety factor is applied — a real calibration budget needs margin on top.

**The most fragile sensor** is the one with the least room. With requirements (`--require cam_yaw=0.5`, or set in the
tab) it is the lowest *tolerated ÷ required*; below 1 the requirement is not met. Without any, it is the lowest
*tolerated ÷ largest severity tried* — a rough ordering only, because the faults' units differ; the result says which
basis it used.

## The fake simulator and a large rig (4.2)

The fake `carla` module the tests and demos use ray-casts every camera pixel in numpy. Measured on the development
machine (`render()` of a scene with six boxes, three frames each): 64 × 36 0.002 s, 320 × 180 0.04 s, 640 × 360 0.20 s,
1280 × 720 **1.19 s per frame**; a label image is a second frame of the same size. A project's default rig (1280 × 720
with semantic labels) therefore spends minutes per run on the fake, and a campaign of dozens of runs looks stalled —
the cost is the test double's, not Falcon's and not a real CARLA's. The section's *Where* row (world and rig of the
project, the active ones preselected) warns when the fake is in use and the chosen rig renders more than about
0.1 s per step there (`ui/robustkit.py`: about 1.3 s per megapixel-frame), and suggests a small rig (320 × 180, no
labels) for runs on the fake.

## Limits

- One fault at a time. Combinations (a misaligned camera in the rain) are not searched.
- Tolerances of *this* stack on *these* runs or scenarios; another scene mix moves them. Offline campaigns score at most
  `--max-frames` (60) frames per sensor.
- The misalignment homography is exact only for rotation; a mount *translation* is not a graded model here (it needs
  depth to re-project).
- The clock offset offline cannot resolve below one frame period, and for a lidar ignores the other actors' motion.
- The lidar criterion is geometric (point retention), not a lidar detector's accuracy: no lidar model registry exists.
- Simulated sensors: real dirt, noise and exposure behave differently in detail.
- Closed loop was tested on the fake CARLA only (the health monitor's thresholds are found: a 45° sector, a GNSS jump
  of about 4 m); no campaign has been run on a real CARLA server or against a real ROS 2 stack.
