# Near-misses in your dashcam footage

`falcon dashcam ingest DIR` watches a folder of your own dashcam clips for the moments another road user came
critically close: it detects and tracks the road users, ranges them, follows your car by its GPS (and, around each
candidate, by visual odometry), measures time to collision, post-encroachment time and the deceleration that would
have been needed, and ranks the events. The top 20 become small folders — the video stretch, the tracks, posed frames —
and any of them becomes a Falcon scenario with an OpenSCENARIO file and a parametric search around it, on this machine.
Code: `falcon_autonomy/core/dashcam_ingest.py` (the folder, the passes, the commands), `core/nearmiss.py` (ego motion,
candidates, odometry, trajectories, measures, export), `core/dashcam_detect.py`, `core/bytetrack.py`,
`core/dashcam_depth.py`, `core/perception_models.py`; the GUI is Replay ▸ **Near-misses** (`ui/dashcam_ingest.py`).
Their docstrings are the reference.

```sh
falcon dashcam nearmiss models                                   # the models: licence, state; the refused ones
falcon dashcam nearmiss fetch rtdetr-v2-r18 --accept-licence     # once each: licence shown, files checked and pinned
falcon dashcam nearmiss fetch da2-small --accept-licence
falcon dashcam ingest DIR [--detector rtdetr-v2-r18|rtdetr-v2-r50|file:PATH] [--depth da2-small|metric3d-small|none]
    [--rate 10] [--size 960x540] [--crop-bottom 0.06] [--camera-height 1.3] [--hfov 110 | --focal-px F]
    [--horizon-row V] [--top 20] [--device auto|cuda|cpu] [--no-vo] [--no-clips] [--redo]
falcon dashcam nearmiss list DIR [--top N]                       # the ranked events
falcon dashcam nearmiss scenario DIR EVENT --project P --world W [--name N]   # EVENT: its rank or id
falcon dashcam nearmiss variations PROJECT NAME [--budget N] [--method M]     # the local rare-case search
falcon dashcam nearmiss reconstruct DIR EVENT [--out OUT] [--pad 2] [--iterations N] [--width W] [--device D]
    [--intrinsics …] [--config FILE] [--no-train]                # the event in 4D, its road users moving
falcon recon closedloop OUT --policy P --out RUN                 # then drive a policy through it
```

## Models and licences

Checked on 2026-11-12 from each model card's metadata and the projects' own pages (no weights downloaded to read
them); each is pinned to a repository revision with every file's size and the publisher's checksum:

| model | use | licence | where it was read |
|---|---|---|---|
| `rtdetr-v2-r18` — RT-DETRv2-S, `PekingU/rtdetr_v2_r18vd` @ 5650961 | detector (default), 20 M parameters, 77 MiB | Apache-2.0 | [card](https://huggingface.co/PekingU/rtdetr_v2_r18vd), code [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) |
| `rtdetr-v2-r50` — `PekingU/rtdetr_v2_r50vd` @ 2824940 | detector, 43 M, 164 MiB | Apache-2.0 | [card](https://huggingface.co/PekingU/rtdetr_v2_r50vd) |
| `da2-small` — Depth Anything V2 Small, `depth-anything/Depth-Anything-V2-Small-hf` @ 5426e4f | relative depth (default), 25 M, 95 MiB | Apache-2.0 | [card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf); [README](https://github.com/DepthAnything/Depth-Anything-V2): "Depth-Anything-V2-Small model is under the Apache-2.0 license" |
| `metric3d-small` — Metric3D v2 ViT-S ONNX, `onnx-community/metric3d-vit-small` @ ee63b95 | metric depth, **opt-in**, 144 MiB | **not stated for the weights** | the conversion's card says only `cc0-1.0` in its metadata; the code ([YvanYin/Metric3D](https://github.com/YvanYin/Metric3D)) is BSD-2-Clause and asks commercial users to contact the authors |

Refused, whatever the command: **every Ultralytics YOLO** (AGPL-3.0 — [its LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE);
Falcon never imports `ultralytics`), **Depth Anything V2 Base and Large** (CC-BY-NC-4.0), and anything built on
**SMPL / SMPL-X** (non-commercial). `metric3d-small` is fetched and used only with `--accept-unstated-weights-terms`,
which is recorded in its `pins.json`. Downloads happen only through `nearmiss fetch NAME --accept-licence` (the licence,
sizes and checksums are printed first); every file is checked against the publisher's checksum, and Falcon's own
SHA-256 is pinned on that first download and checked on every load. Loading is offline (`local_files_only`,
`HF_HUB_OFFLINE`): nothing is ever downloaded by an ingest. Models live in `~/FalconWorkspace/models/perception/`
(`--models DIR`).

## What happens

Everything is written inside the folder, under `DIR/falcon-ingest/` — the clips are never modified:

```
falcon-ingest/
  ingest.json                      state, settings and their key, per clip what is done, counts (private marker)
  clips/<key>/perception.json      pass 1 (key: a hash of the clip's path — never its name, which holds its date)
  clips/<key>/nearmiss.json        pass 2
  ranking.json                     every event, best first
  events/<id>/                     the top 20: clip.mp4, event.json, detections.json, posed/posed.json,
                                   trajectories.json, run.json
```

**Pass 1 — perception**, per clip, streamed (no frame kept): frames at 10 a second and 960 × 540 from the dashcam
decoder (NVDEC when it works); in batches of 8 the detector (FP16 on CUDA) and the depth network; per frame the COCO
classes mapped (person → pedestrian, bicycle, car, motorcycle, bus, truck, cow → cattle; the rest dropped), a person
over a two-wheeler folded into it as its rider, **ByteTrack** (Falcon's own implementation of Zhang et al. 2022; the
reference code is MIT): a Kalman filter on (x, y, aspect, height), first association of tracked and lost tracks with
the boxes scoring ≥ 0.6 (IoU distance fused with the score, below 0.8, Hungarian assignment by scipy), second
association of the still-tracked ones with the boxes scoring 0.1–0.6 (IoU distance below 0.5), new tracks from boxes ≥
0.7, a lost track kept 30 frames at 30 frames/s (10 at 10). A clip with fewer than two GPS fixes, or a rear camera's
clip, is skipped.

**Ranges.** Each tracked box gets two ranges, fused by inverse variance:

- *flat ground*: its bottom centre cast onto the road under a camera at a known height (`--camera-height`, 1.3 m by
  default) and pitch (from the horizon row) — `core.incident.ground_point`. σ: 2 px in the bottom edge and 5 % for a
  road that is not flat.
- *the depth network*: Depth Anything V2 gives relative (affine-invariant) inverse depth d = α/Z + β. On a flat road
  1/Z = cos θ (v − v_h)/(f H) is linear in the row, so the road's d is a plane, fitted by RANSAC to the pixels below
  the horizon outside every box; with the horizon known α and β follow and every pixel has a range; a box's is the
  median over the lower half of its central third. σ from a bootstrap of the fit and 10 % for the network. Metric3D's
  metric depth is checked against the camera height on the same road fit and rescaled.

When the two disagree by more than 3 σ, the box is flagged and the geometric range kept (the network's for a box cut
by the image's bottom edge). The **horizon** is estimated per clip from the focus of expansion of the image flow
while the car drives straight above 4 m/s (median over frame pairs), or given (`--horizon-row`). Then each box goes
through the **autorickshaw heuristic**: COCO has no autorickshaw, so a car, truck or motorcycle box whose metric size
at its flat-ground range is 1.1–1.6 m wide, 1.4–2.0 m tall and taller than wide (1.05–1.6) becomes an autorickshaw,
marked `class_source: "heuristic"`; a track's kind is its boxes' majority.

**Pass 2 — mining**, per clip:

1. *Ego motion*: the GPS fixes in local metres, with the speed and course as velocity, through a constant-velocity
   Kalman filter and RTS smoother (σ 3 m, 0.5 m/s, process 2 m/s²) at every frame; acceleration and jerk by
   Savitzky–Golay derivatives over about 1 s.
2. *Candidates*: a track in the ego's corridor (|lateral| < 2.5 m) closing with a range-rate TTC under 4 s, any track
   nearer than 6 m, or the ego braking harder than 3 m/s² with a track in view; ±3 s windows, merged.
3. *Visual odometry on the candidate windows only*: Lucas–Kanade corners off the boxes (kept when tracked back to
   within 0.5 px), per frame pair the rotation of the road's homography (RANSAC; decomposed, the solution whose plane
   normal is the road's within 25° — exact for the road, with or without translation) or else of the essential matrix,
   its yaw integrated from the GPS heading (the 1 Hz course misses a swerve); positions from the GPS speed along that
   heading. Not used (and said why) when more than 30 % of the pairs fail or it departs more than 20°
   from the GPS's heading.
4. *Trajectories*: every track's points attached to the ego pose of their frame (a vehicle's centre half its typical
   length beyond its near face), smoothed by the same RTS smoother (σ along the ray the range's σ, across it 0.3 m +
   1 %; process 3 m/s², walkers 1 m/s²) → (N, 8) arrays (t, x, y, yaw, vx, vy, length, width) for the ego and each
   actor, sizes typical for the kind.
5. *Measures*: `core.criticality`'s TTC series (exact moving-box test), PET, DRAC series and minimum distance; the
   ego's peak deceleration and jerk; the criticality score (noisy-or of the four terms, heuristic scales —
   [scenarios](scenarios.md)). A **bootstrap over the depth scale** (every range × a factor drawn in [0.9, 1.1],
   50 draws) gives the score's median and 5–95 % interval; events are **ranked by that median**.

The top 20 are exported: the video from 5 s before to 5 s after the window by `ffmpeg -c copy` (no re-encoding; it
starts at the key frame at or before the cut), event.json (every measure, the window, the odometry's verdict, the
depth sources), a `falcon.dashcam_detections/1` file whose boxes carry their fused `range_m`, and `posed/posed.json` —
the form `core.incident` reads.

**A scenario from an event** (`nearmiss scenario`, or *Make scenario* in the GUI) runs the existing
[incident](dashcam.md#a-real-incident-as-a-scenario) path on the event: the ego snapped to the lanes of an imported
world where the footage drove, every actor on its track, a replay scenario, a parametric variations file and
`incident.xosc` (OpenSCENARIO 1.2), in the project's `incidents/<name>/`. *Queue variations* (`nearmiss variations`)
runs the rare-case search over the variations on this machine (sim.studio).

**The event in 4D** (`nearmiss reconstruct DIR EVENT`, or *Reconstruct with moving road users* in the GUI): the
event's window ± 2 s (`--pad`) of its clip goes through `falcon recon from-dashcam --dynamic --tracks EVENT_DIR`
([reconstruction](reconstruction.md#moving-road-users-from-a-dashcam)): the segment is posed by its own odometry, and
every road user the ingest tracked in it becomes a moving actor of the reconstruction instead of being masked — its
smoothed track carried frame by frame from the ingest's world into the reconstruction's, its size the kind's typical
one. The result (`~/FalconWorkspace/recon/nearmiss-<id>` by default; the GUI opens it in Replay ▸ Reconstruction) is a
dynamic reconstruction like any other: `falcon recon closedloop` (*Drive a policy here*) drives a policy through the
real incident, with the incident's road users replaying what they did. A GPU job unless `--device cpu`; on this machine
only.

**Resuming**: a clip already perceived with the same settings and the same clip (its fingerprint) is not processed
again, nor mined again; a cancel loses only the clip in progress. `--redo` starts over.

**On an 8 GB card** (the low_vram [profile](profiles.md#the-near-miss-perception)): the detector and the depth
network run in batches of 2, `rtdetr-v2-r50` becomes `rtdetr-v2-r18` (`--keep-detector` keeps it), Depth Anything V2
works at 364 px instead of 518 (about half the pixels), both models get 0.35 of the GPU
(`torch.cuda.set_per_process_memory_fraction`; onnxruntime's CUDA arena for Metric3D), and they run FP16 on CUDA unless
`inference_precision` is fp32. The profile and what it changed are recorded in `ingest.json` and every
`perception.json`. A swapped detector or the capped depth resolution changes the settings, so a folder perceived under
the other profile is perceived again; batch, precision and GPU share do not.

## Privacy

Your footage shows where and when you drove, and other people. Everything here stays on this machine: the folder,
every clip in it and everything under its `falcon-ingest/` are private to the assistant (`assistant.privacy` finds
`falcon-ingest/ingest.json` at any depth), the event folders and the scenarios made from them carry dashcam markers,
and dashcam-derived scenarios and studies are refused by cluster and cloud bundles. A reconstruction of an event is
private from its first file (its `dashcam.json`), with its `posed/tracks.json`; so is every drive through it — the
closed-loop run folders, re-rendered and hybrid runs carry the dashcam marker in their records (and a record that names
a private reconstruction is private while that exists) — and no bundle carries any of it. What a tool or a log may say
about an ingest (`dashcam_ingest.summary`) is counts and event ids only — no clip names, places, times, positions or
images; no image or track is sent to any model.

## Measured

On a synthetic clip (tests/test_dashcam_nearmiss.py: 7 s at 10 frames/s, 320 × 180, a textured road, a lead car
braking from 25 m to a 7 m gap, a truck and an autorickshaw-sized car in the next lanes, a ground-truth detector and an
affine-distorted true depth): every vehicle kept one track id; the lead car's fused ranges within 3 % (median); the
autorickshaw-sized car relabelled by the heuristic and the car not; the lead's minimum TTC within 0.5 s of the ground
truth's (1.5–3 s) and ranked first; the horizon from the focus of expansion within 6 px of the truth (about 2 px on
this clip). The odometry on a synthetic road yawing 1° a frame while moving 0.5 m: every pair within 0.3° of 1°
(tests/test_dashcam_perception.py); on the straight clip its integrated heading stayed within 5° of the GPS's over 7 s
at 320 px wide. **Not yet measured on real footage**: the models are downloaded only at acceptance (the lead's approved
download), and the user's clips are never read during the build.

An event in 4D (tests/test_recon_dashcam_dynamic.py: an 8 s synthetic clip at 480 × 270 of a walled, textured street,
the ego at 6 m/s weaving ±4°, a lead car closing at 1.5 m/s in its lane and a truck in the next one, ingested with a
ground-truth detector and flat-ground ranges, its top event reconstructed on the CPU with 30 training steps): both
road users became moving actors with their kinds' sizes; over 63 boxes each, the car's centre lay a median 0.73 m
and the truck's 0.45 m from where the clip drew them in the reconstruction's world (whose own GPS-aided poses sit a
median 0.54 m from the fixes), their headings within a median 1.5° and 1.9°; a policy drove through it in closed
loop with both replayed. The chain itself is exact: on a reconstruction world that is a rotated and shifted copy of the
ingest's, every box landed within 5 cm and 0.5° of where it should.

## Limits

- **Ranges hang on the horizon.** At forward distance X the flat-ground range moves by X/(f·H) per pixel of horizon
  error: about 6 % per pixel at 25 m for a 110° lens at 960 px (f ≈ 336 px) and 1.3 m. The focus-of-expansion
  estimate is good to a few pixels; a mount whose horizon row is measured once (`--horizon-row`) is better. The
  relative network's scale rests on the same horizon (its shift β does).
- Flat road, no roll, no lens distortion (a wide dashcam lens bends straight lines: ranges at the image's sides are
  worse); typical sizes, not measured ones; one camera, so an occluded road user is extrapolated by its track only.
- The ego's speed is the GPS's (1 Hz, smoothed), the GPS-to-video offset about ±0.5 s ([time base](dashcam.md#the-time-base));
  the odometry refines the heading only, over a window, and is noisy at low resolution.
- The ego's box is centred on the camera (a 4.6 m box: its front 2.3 m ahead of the windscreen).
- Events do not span clips; the stream-copied clip starts at a key frame (up to 0.5 s early on the verified camera).
- The autorickshaw class is a size heuristic, and COCO's classes miss handcarts and cycle-rickshaws (a two-wheeler or
  nothing); a real class needs a detector fine-tuned on labelled footage.
- The criticality score's scales are heuristic ([core.criticality](scenarios.md)); a box overlap from monocular noise
  scores as a collision.
- An event in 4D carries every limit above into its actors (monocular ranges, typical sizes, a flat road under the
  boxes) and the [reconstruction's](reconstruction.md#moving-road-users-from-a-dashcam) own; a road user seen in fewer
  than 3 frames of the segment is left out (counted in `dashcam.json`), and one hidden behind another is where its
  track extrapolates.
- The low_vram caps are chosen, not measured on an 8 GB card (the models are fetched only at acceptance).
