# Rig variants compared by perception

Vantage plans a camera rig geometrically: coverage, blind ground, where the stitched bird's-eye view changes camera.
Geometry alone cannot say whether a change matters to a detector. Tilting the Tigor EV's mirror cameras from 45° to
40° changed almost nothing geometrically (blind area 0.01 m², coverage 99.97 %), yet the weakest stitch seam moved
from ML|QL to QR|MR. Falcon answers the other half: it records each variant on the same scenarios with the same
seeds, runs a detector on every camera, and says whether recall changed — overall, round the car and near each seam.

Rigs ▸ Import / export has three entries; `falcon rigs` does the same headless.

| step | Rigs page | command |
|---|---|---|
| look at a Vantage file | Import variants from a Vantage project… (the dialog lists them) | `falcon rigs vantage FILE [--json]` |
| make rigs of its variants | the same dialog: tick variants, choose the car and the image scale | `falcon rigs import PROJECT FILE [--variant V]… [--vehicle auto\|BP] [--scale S]` |
| record and compare | Compare rigs by perception… (queued as a job) | `falcon rigs compare PROJECT --rig A --rig B … --model M [--scenario S]… [--seeds N] [--world W] [--name N]` |
| score again, other settings | — | `falcon rigs report STUDY [--pair A,B]… [--radius M] [--band DEG] [--threshold T] [--boot N]` |
| a note for Vantage | Rig studies… ▸ Vantage design-log note… | `falcon rigs note STUDY --out FILE.json [--pair A,B]` |

`falcon rigs compare` also takes `--vantage FILE [--variant V]…` to use a file's variants without adding them to the
project, `--fps` (default 10), `--seconds` (each run's length) and `--every N`. The assistant's `research_rig_compare`
queues the same command; `analysis_outputs` kind `rigstudy` reads a study's numbers.

## Reading a Vantage project

Falcon never imports Vantage: it reads the `.vantage` file (a zip of `project.json` and `mesh.npz`) itself, read
only, and keeps its path and SHA-256 in each rig (`rig.vantage`). Nothing is written back.

**Read:** each variant's cameras (tag, pose, module, lens, a unit calibration, hidden or not), its lidars, radars,
GNSS antennas and IMUs (pose and model), the seams set in Vantage (`variant.seams['uidA|uidB']`, azimuth in
degrees), the modules (resolution, pixel size, frame rate), the lenses (focal length, projection — rectilinear,
equidistant, equisolid, stereographic, orthographic or Kannala–Brandt with k1–k4 — datasheet field of view,
image-circle limit, principal-point offset, aspect), the sensor models (channels, beam elevations, fields of view,
range, rate, radar zones), the body (mesh extents without mirrors, wheel centres, datasheet dimensions) and the
ground region of the analysis. **Ignored:** targets, photos, obstacles, brackets, housings, the ECU, cable routing,
GMSL ports and sync lines, streams, analysis settings and the log — none of it changes what a simulated camera sees.

**Lenses** resolve as Vantage resolves them (a unit calibration replaces the library lens, scaled to the module's
resolution); a camera becomes a Falcon fisheye with that exact model, rendered as CARLA pinhole views remapped
through it ([hardware](hardware.md), `core/lens.py`); a centred rectilinear lens becomes a pinhole camera. Images are
scaled by the import's image scale (default 0.5, Vantage's own simulation default).

**Frames.** Vantage's poses (ISO 8855: y left, pitch positive down, origin at the rear axle on the ground) go into
CARLA's vehicle frame by `sim/frames.py` (y mirrored; yaw and pitch change sign by conjugation of the rotation), and
the rear axle is placed at the CARLA car's rear axle — how Vantage's own CARLA bridge mounts a rig.

**The car.** CARLA renders its own car, not your body. All variants of one import go on one CARLA car: the nearest
by length, width, height and wheelbase (Vantage's rule and its table, measured on a 0.9.16 server), or one you
choose. The Tigor sample's nearest is `vehicle.audi.a2`, 3.1 % off on average. A camera that sat on your body may sit
inside or off CARLA's: the import says so when the car is more than 8 % off.

**Seams.** The stitched view gives the ground round the body to its cameras by azimuth about the body centre; a seam
hands the sector after it (counter-clockwise) to its second camera. A seam you set in Vantage is used exactly. Vantage
does not store the seams it computes itself, so Falcon estimates those by Vantage's own rule (each camera's
density-weighted mean azimuth where it sees the ground best, then per neighbouring pair the ray with the most shared,
well-resolved ground) on a ground grid round the body's box, the box being the only occluder. **Measured** against
Vantage's own analysis of its sample Tigor project (5 variants, 26 seams, 2026-11-02): median difference 1.1°, mean
1.8°, 23 within 5°, largest 6.5° (the front-heavy variant's FL|ML). Each seam says whether it is Vantage's or
estimated. The parser reads the user's own Tigor project (4 variants of 6, 4, 4 and 6 cameras; no seams set in it,
so all are estimated).

## What a study records

Every rig records every scenario with every seed (default: two surround-view scenarios × 3 seeds), in **kinematic
motion** with **no background traffic**: the road users are at the same place at every step whichever rig looks, so
the variants pair object by object (a study checks this and reports objects present in only one run). The two
surround-view scenarios ship with Falcon (`core/rigcompare.py`): *children close round a stopped car* (a child crosses
just ahead of the bumper, another behind the car, a cyclist and a scooter pass along both sides about a metre off
the body — every seam is crossed) and *creeping past a child at the kerb* (the child goes from the front cameras
along the side to the rear). Any built-in or project scenario can be added. Runs, predictions and the report are in
`<workspace>/rigstudies/<name>/`; a study resumes (unchanged runs are not driven again, the model's predictions are
kept by the runs' hashes).

## The numbers

The detector's boxes are matched to each frame's ground truth per class at IoU 0.5 (score ≥ 0.5), as COCO matches;
objects with fewer than 20 visible pixels or more than 90 % hidden are ignored ([models](models.md)).

| number | what |
|---|---|
| precision | matched detections / detections, every camera's frames |
| camera_recall | matched objects / scored objects, every camera's frames |
| seen | road users within 8 m of the car's box that some camera scores |
| recall | road users within 8 m that some camera detects (a blind spot is a miss) |
| recall_seen | detected / seen |
| stitched_recall | detected by the camera the stitched view shows at the object's azimuth (the owner of its sector between the variant's seams) |
| `….front` … `….front-right` | by 45° sector round the body centre (Vantage's azimuth) |
| `….clearance[a,b)` | by the object's distance from the car's box: 0–1, 1–2, 2–4, 4–8 m |
| `….seam[A\|B ±N°]` | within ±10° of a seam of any rig compared — so two rigs are compared on the same objects near either rig's seams |
| AP | COCO AP, every camera's frames pooled, with the harness's drive-level interval |

**Intervals.** A unit is a scenario run when a study has five or more, else a 2 s block of one run (then the report
says the interval is too narrow). Each rig's numbers have a 95 % cluster-bootstrap interval over units.

**The paired comparison** of rig B against rig A counts only objects present in both. It reports B − A with its
paired bootstrap interval, and a verdict from a **cluster-robust paired t-test** over the units with **Holm's
adjustment** over all the pair's numbers: worse or better when the adjusted p is below 0.05, else *no detectable
change*, which is not *no change*. Why not the bootstrap's interval: with the handful of runs a study has, it claims a
difference too often — measured on null data (6 runs of 60 objects, both rigs detecting with 0.7): the percentile
bootstrap excluded zero 11 % of the time at a nominal 5 %, the cluster t-test 3.8 %, with 92 % power against a 0.15
drop. A pair has some forty numbers, so without the adjustment two would be "significant" by chance. Holm across
nested numbers (a sector and a seam band inside it) is conservative. The tests plant a known drop near one seam in
synthetic detections and check it is found there and nowhere else.

## Falcon renders no stitched image

`core/lens.py`'s seam-blend tables feather the cube faces **inside one fisheye** (the pinhole views CARLA renders for
it), not the cameras of a surround view. So *stitched recall* is detection in the camera whose image the stitched
view shows at that azimuth, not detection on Vantage's stitched bird's-eye image; a detector run on the stitched image
could behave differently (and Vantage's stitch also falls back to another camera where the owner cannot see, which
this does not model).

## A note for Vantage's design log

`falcon rigs note` (or *Vantage design-log note…* in a study's results) writes a file in the layout of Vantage's own
design-log export (`vantage.design_log/1`): one `note` entry per pair (at most 400 characters, Vantage's limit) with
the recall and stitched-view recall changes, their intervals and verdicts, and the seams where a verdict was reached,
`about` the cameras of those seams. Falcon never writes into a `.vantage` file and refuses an output path ending in
`.vantage`. This Vantage build has no importer for that file (checked 2026-11-02): paste the text in Vantage's design
log, or import it once Vantage reads one.

## Limits

- **Not yet run on a real CARLA server.** Everything above was tested against the fake carla and synthetic
  detections; the Tigor variants' actual recall numbers — and whether the 45° → 40° tilt matters near QR|MR — need a
  study on a CARLA 0.9.16 server with a real detector (the synthetic detector only tests the method: its detection
  probability falls with box height, so it does respond to a rig, but it is not a model). A Tigor 6-camera variant at
  image scale 0.5 renders about 77 Mpx per step (5 cube faces per camera, with the instance and depth cameras), under
  the 200 Mpx budget.
- CARLA's car, not your body: the ground the body hides near the car is CARLA's.
- Estimated seams are within a few degrees of Vantage's own, not equal; set a seam in Vantage to make it exact.
- Fisheye images are pinhole views remapped through the lens model: one instant per frame, no rolling shutter,
  motion blur, flare or the real lens's resolution fall-off beyond the model. Lidar beam elevations become CARLA's
  even spread; several radar zones become one cone.
- Road users only (vehicles and walkers within 8 m); the map's parked cars count in the per-camera numbers, not in the
  rig-level ones.
