# Methods

How every figure Falcon shows is computed, and what it rests on. Figures are labelled **measured** (on this machine,
by a run whose folder is named), **typical** (the reference machine's measurement, before this machine has its own)
or **heuristic** (a stated rule of thumb).

## Throughput and cost

A run's cost is modelled as seconds per simulation step = *overhead* + *rendered Mpx per step* / *throughput*.

- **Rendered Mpx per step** — the sum over cameras of width × height × (1 + label cameras). Label cameras are
  CARLA's semantic segmentation, instance segmentation and depth cameras, one per kind (2D boxes come from the
  instance image and add none). A fisheye renders five cube faces of N × N pixels, N = 2·⌈f·tan(45° + 0.75°) + 1.5⌉
  where f is the lens's focal length at its centre in px/rad (from the lens model; without one, heuristic: an
  equidistant lens whose circle fits the shorter image side). Lidar, radar, GNSS and IMU are ray-cast or computed,
  not rendered, and count 0.
- **Throughput** — every successful drive on this machine adds a point: (rendered Mpx per step, wall seconds per
  recorded step; setup, map loading and warm-up excluded). With points at two or more clearly different rendering
  loads (> 20 % apart), overhead and throughput are the least-squares line through them (overhead ≥ 0). With one load
  only, overhead and rendering cannot be told apart, so all of the time is put on rendering. The last 30 points are
  kept. Until a drive has run, the reference machine's figure is used, labelled **typical**: six half-size fisheyes
  with ground truth (77 Mpx per step), 602 labelled frames in 114 s on an RTX 5080 with a 12 GB-capped 0.9.16 server
  (≈ 68 Mpx/s).
- **Estimate** — setup (20 s, heuristic) + steps × seconds per step, labelled **measured**, **extrapolated from
  measured** (a load more than twice the largest measured, or under half the smallest) or **typical**.

Measured on the reference machine (RTX 5080 16 GB, CARLA 0.9.16 capped at 8 GB): one 1280 × 720 camera with
semantic labels (1.84 Mpx per step) ran at 54–60 steps per second (2.7–3.0× real time at a 0.05 s step), ≈ 100–110
Mpx/s, bounded by JPEG and PNG encoding rather than rendering (the run's `encoder_waits` counts the steps that waited
for the encoder).

## Render budget

A 0.9.16 server capped at 12 GB crashed (signal 11) at 306 Mpx per step — six 1920 × 1536 fisheyes with instance and
depth ground truth — and ran at 77. Runs over the budget (200 Mpx by default; Settings) are refused before they
start, with exit code 6.

## Fog visibility

CARLA's fog density maps to a meteorological visibility of ≈ 7000 m / fog_density (measured on 0.9.16, for objects
0.2–2.5 m above the road, at a 5 % contrast threshold). The 5 % threshold is a human observer's; for a detector it is
an upper bound on what it can see.

## Labels

Semantic labels are CARLA's semantic segmentation camera at the colour camera's exact pose, size, field of view and
tick (matched by frame number): the image's R channel is the 0.9.16 semantic tag (`carla.CityObjectLabel`; the
numbers are checked against the installed wheel by the test suite), stored unchanged as an 8-bit PNG. Colours are
CARLA's CityScapes palette. The legend's shares are pixel counts over the frame.

## Reproduction

`falcon reproduce RUN` re-runs a drive from its own `drive.json` (same seed, settings, map, rig) into a new folder and
compares:

- **step count** — equal;
- **ego trajectory** — every step's ego position, rotation and speed within `pose_tol` (default 0.001 m, ° and m/s);
- **label images** — the share of identical pixels, at least 99.9 %;
- **RGB images** — the mean absolute difference in 8-bit levels at most `rgb_tol` (default 2): JPEG and Unreal's
  temporal effects (exposure adaptation, motion blur) can change images slightly even when the scene is identical.

On the fake simulator the comparison is exact (bit for bit). On a real server, exact reproduction needs the same
server build, the same map loaded fresh and synchronous mode throughout; the report states which checks agreed
rather than a single yes or no.

## Lidar labels and their proof

Each rig lidar is spawned twice on the same step with the same attributes: CARLA's ray-cast lidar (the recorded
points, with its range noise) and its semantic lidar (no noise; the CARLA actor id and semantic tag of every point).
A recorded point takes the semantic point at its index **only when** both slices have the same points per channel
and every direction agrees within 0.01°; otherwise the slice is matched per channel by azimuth and counted as a
fallback. Every run reports the number of slices matched each way, the worst direction difference and the RMS range
difference (which should equal the noise set on the lidar).

**Points in vehicle boxes.** Of the points inside a vehicle's 3D box (the ego excluded), the share labelled a vehicle
class (car, truck, bus, train, motorcycle, bicycle, rider). A return lies on the surface it hit, so the box is grown
5 cm outward on its sides and top, and its bottom is raised 10 cm so ground returns under the vehicle are not counted.

**Range against a camera.** For the first camera with a depth label camera and the first lidar, every fifth step:
each lidar point is moved into the camera's frame (through the two mounts), projected, and its range compared with
the camera's range along the same ray (planar depth × |ray|; inverse depth interpolated bilinearly only when the four
pixels around are one surface, within 2 %). The two sensors stand apart: a point the camera sees a surface more than
0.5 m nearer in front of is hidden from it (parallax), counted apart and not compared.

Measured on the reference machine (CARLA 0.9.16, Town10HD_Opt, 40 vehicles, 20 s at 20 Hz; a 32-channel lidar at
600 000 points/s, 1.5 cm noise; a 1280 × 720 camera with depth):

| check | result | acceptance |
|---|---|---|
| slices matched by order | 400 of 400 (11.2 M points), worst direction 3.5·10⁻⁶°, range RMS 1.500 cm (noise set 1.5 cm) | — |
| points in vehicle boxes labelled vehicle | 100 % of 262 282 | ≥ 99 % |
| lidar range − camera range | median 0.9 mm, median absolute 1.15 cm, 90th percentile 3.2 cm (338 311 points; 24 608 hidden from the camera) | median ≤ 2 cm |

## Criticality

Computed from the recorded trajectories (every 0.05 s: position, heading, velocity, box) — `core/criticality.py`.

- **Time to collision (TTC)** — the first time the two oriented boxes overlap if both keep their velocity (exact, by
  the separating-axis test on moving boxes; Hayward 1972). Its minimum over the run.
- **Post-encroachment time (PET)** — the time between one road user leaving the area where the two paths' footprints
  overlap and the other entering it (Allen, Shin & Cooper 1978); computed pointwise, it can only overestimate, by less
  than two samples.
- **Minimum distance** — between the boxes (Ericson 2005).
- **Required deceleration (DRAC)** — v_rel² / (2 · gap) for a road user ahead in the ego's path (Almqvist et al. 1991;
  Archer 2005).
- **Collisions** — steps where the boxes overlap, and the ego's collision-sensor events.
- **The criticality scalar** (heuristic, what searches maximise): 1 for a collision, else
  0.9 · (1 − (1 − s_ttc)(1 − s_pet)(1 − s_drac)(1 − s_dist)) with s_ttc = exp(−TTC / 1.5 s) (1.5 s: van der Horst
  1990's critical TTC), s_pet = exp(−PET / 1 s), s_drac = 1 − exp(−DRAC / 3.35 m/s²) (Archer 2005's threshold),
  s_dist = exp(−d / 2 m); the maximum over the actors.

## Rare-case search

Random (uniform or Halton), CMA-ES (Hansen 2016), Bayesian optimisation (a Gaussian process with a Matérn-5/2 kernel,
hyperparameters by marginal likelihood, expected improvement; Jones et al. 1998, Snoek et al. 2012) and adaptive
sampling (the cross-entropy method; Rubinstein 1999, O'Kelly et al. 2018). Tested on a known-weak synthetic model — a
detector-miss probability over fog, distance, sun altitude and occlusion with a planted failure region of 2.96 % of the
volume — with 120 evaluations each:

| method | evaluations to the 5th failure (mean, seeds 0–4) | failures found at 120 |
|---|---|---|
| random | 98.8 | 6.2 |
| CMA-ES | 48.6 | 49.8 |
| Bayesian optimisation | 22.0 | 51.0 |
| adaptive sampling | 43.0 | 64.8 |

Over 20 other seeds (100–119): random 115.3 (most seeds never reached five failures) and 3.4; CMA-ES 39.9 and 62.6;
Bayesian optimisation 22.2 and 50.6; adaptive sampling 33.5 and 73.8. `core.search.compare` returns both curves.

Bayesian optimisation by the **probability of failure** (`bayes-pof`: P(f ≥ t) = Φ((μ − t)/σ) instead of expected
improvement) reaches the fifth failure as soon (25.2 on seeds 0–4, 22.5 on 100–109) and finds about twice the
failures at 120 (99.2, 101.8) with nearly three times the coverage of the planted region; straddle and expected
improvement over the threshold were measured and not adopted ([model-in-the-loop](rare-cases-mil.md#searchers-probability-of-failure)).

## Perception failure and root cause

The model-in-the-loop objective — window recall, lost lead, miss rate and the perception failure (heuristic) — is
defined in [model-in-the-loop](rare-cases-mil.md#the-perception-failure); counterfactual and correlational
attribution in [root cause](root-cause.md).
