# Dataset format (falcon-raw)

Every recorded drive (and every live recording) is a folder in this layout. Exports to COCO, YOLO, KITTI, nuScenes,
Waymo-style, BDD lanes, ROS 2 MCAP and semantic occupancy (Occ3D-nuScenes, OpenOcc v2, SurroundOcc: [below](#semantic-occupancy-exports-occ3d-openocc-surroundocc)) are made from it (Data ▸ Export, `falcon export`).

```
run/
  drive.json          the input: world, weather, rig, traffic, seed, step, …   (live: live.json)
  run.json            what ran: versions, seed, hashes, timings, measured performance, lidar checks, faults applied;
                      traffic: the vehicles placed, and on an imported world speed_source — the traffic's speeds
                      from the file's limits ("map") or CARLA's own ("carla"), see worlds.md, and keeper — the
                      traffic moved away from roads that end (counts, events); lights: vehicle lights set and read
                      back (drives, live, stack, gym, HIL); city: an imported
                      world's buildings, trees and lamps placed (the plan's SHA-256, spawned, failed, seconds);
                      trigger_rules: the rules, every firing, counts, actors spawned (and their criticality
                      in a drive without a scenario); faults_applied[sensor].started: faults started mid-run
  dataset.json        sensors (K, lens, views, planned pose, streams), label encodings and classes, provenance
  frames.jsonl        one line per simulation step (below)
  lanes.json          the map's driving lanes and lane graph (labelled drives)
  trajectories.json   scenario runs, and drives whose trigger rules spawned actors: every scripted actor's
                      (t, x, y, yaw, vx, vy, length, width) per step, for core.criticality
  actors.jsonl        every vehicle and walker at every step: pose, velocity, box, category (CARLA world frame);
                      actors a kinematic scenario moves carry the director's velocity (CARLA reports them still)
  rgb/<camera>/<n>.jpg
  semantic/<camera>/<n>.png      uint8 class per pixel
  instance/<camera>/<n>.png      uint16: 0 none, else an object's id in objects/
  range/<camera>/<n>.png         uint16 = round(range_m × 256) along the pixel's ray; 0: nothing within 256 m
  objects/<camera>/<n>.json      the objects seen (below)
  flow/<camera>/<n>.npy          optical flow (below)
  lidar/<lidar>/<sweep>.bin      float32 x y z intensity, the lidar's frame at the sweep's end
  lidar/<lidar>/<sweep>.labels.bin   uint8 tag per point, then uint32 actor id per point (tag 30: weather_noise)
  radar/<radar>/<n>.npy          float32 velocity, azimuth, altitude, depth (CARLA's order)
  radar/<radar>/<n>.flags.npy    uint8 per row: 1 = weather clutter (only frames the sensor physics touched)
  odd-monitor.json               the ODD monitor's timeline (runs with an ODD, or `falcon odd monitor`)
  gnss/<gnss>.jsonl  imu/<imu>.jsonl
```

## Frames
`frames.jsonl`: `step`, `frame` (CARLA's), `sim_t` (s from the first recorded step), `ego` (pose in CARLA's world),
`speed_mps`, `control`, `recorded` (false on steps `record_every` skips), `teleport` (on the step the road-end guard put the ego back on the road: `cause` (`dead_end_ahead` or `fell`), from, to, detail — the pose jumps there; see [worlds](worlds.md#roads-that-end-the-edge-of-an-imported-area)), `traffic_respawns` (on a step the traffic keeper moved traffic away from a road that ends: per vehicle kind — `dead_end`, `fell`, `lost`, `replaced` — actor, from, to and, after a stop in view, `held_s`; those actors' poses jump there), `files` (`"<sensor>/<kind>": path`), and
when used `stamps` (per sensor, with its clock-offset fault), `faults` (per sensor: `drop` or `freeze`), `lidar_pose`
(per lidar: the 3 × 4 world matrix of the sweep's end, row-major), `physics` (per sensor: the weather effects that
touched the frame — [sensor physics](sensor-physics.md)) and `odd` (`in`, `why`, `road`, … — [ODD
monitoring](odd-monitoring.md)), and `triggers` (on the step whose state made a trigger rule's condition true: per firing `rule`, `t`, `signals` — the values its condition read, inf as `"inf"` — and `actions`, each with `did` or `error`; the actions act from the next step — [trigger rules](scenarios.md#trigger-rules)).

Lidar labels: the tag is CARLA's semantic tag of the surface hit, except **30 `weather_noise`**: a spurious return of
fog or rain added by Falcon's sensor physics (actor id 0; nuScenes-lidarseg `noise`). Real returns always keep CARLA's
tag and actor id.

`control` is the ego's control **in effect during the tick that produced this step** — `throttle` 0–1, `steer` −1…1
(positive right), `brake` 0–1, `hand_brake`, `reverse` — so the action taken from step *i*'s state is step *i + 1*'s
`control`. On the autopilot it is CARLA's `Vehicle.get_control()` after the tick (what the traffic manager applied);
when the scenario director drives the ego it is the director's own command (in kinematic motion no `VehicleControl`
reaches CARLA). Live recordings keep their own `control` (the manual, policy or `null` for autopilot steps). Drives
recorded before 0.6 have no `control`. In kinematic scenario drives `speed_mps` is the director's speed (CARLA reports
a physics-off actor as still; drives before 0.6 recorded 0 there — `actors.jsonl` had the right velocity).

## Coordinates
CARLA's frames throughout: x forward, y right, z up — left-handed — metres and degrees. Sensor poses are in the
vehicle frame (CARLA's vehicle origin). `dataset.json` states the **planned** calibration; a mount fault moves the
sensor CARLA spawns, not the calibration (as on a real faulty rig).

## Labels
- **Classes**: CARLA 0.9.16's semantic tags (0–28, `carla.CityObjectLabel`) and 29 `ego` (the vehicle carrying the
  rig); each with its nuScenes-lidarseg category (`dataset.json` → `labels.semantic.classes`).
- **Semantic** comes from CARLA's instance camera (its red channel is the tag — measured equal to the semantic
  camera's), so semantic, instance, range and objects are one consistent read of the same pixels.
- **Range** is along each pixel's own ray: the depth camera's planar depth × |ray|, inverse depth interpolated only
  within one surface.
- **Objects** (`objects/…json`): `id` (in the instance image), `track_id` (the CARLA actor id — the same in every frame;
  map objects `level:<k>`), `actor_id`, `class`, `category` (nuScenes), `bbox` (tight, from its pixels, pixel
  edges), `bbox_full` (its 3D box through the lens, unclipped), `truncated` (share of the box outside the image),
  `occluded` (share of the box's pixels hidden by something nearer), `range_m`, `pixels`, `box3d` (vehicle frame:
  centre, size l w h, yaw), `rle` (COCO run-length mask).
- **Stand-ins and proxies**: an object row (and an `actors.jsonl` row) may carry `subclass` — see below.
- **Imported worlds**: the ground's labels and range come from the map ([worlds](worlds.md#labels-in-an-imported-world-a-carla-limit-worked-around));
  `objects.json` → `ground_relabelled` counts the pixels.
- **Lidar**: per point the semantic lidar's tag and actor id, matched point for point and proven every step
  ([methods](methods.md#lidar-labels-and-their-proof)).
- **Lanes** (`lanes.json`): driving-lane waypoints every 2 m with width and left/right lane markings (CARLA's
  LaneMarkingType and colour), and the lane graph (CARLA's topology).
- **Optical flow** (pinhole cameras with the `flow` label): `flow/<camera>/<n>.npy`, float16 H × W × 2, each
  pixel's motion in pixels from the previous frame, u right and v down — CARLA's optical-flow camera decoded as
  measured on 0.9.16 (pixels = x × W/2, −y × H/2; median error 0.4 px against the flow predicted from depth and the
  camera's motion). Fisheye flow is not recorded.

## Indian stand-ins and proxies
CARLA 0.9.16 has no autorickshaw, cycle-rickshaw, cattle or handcart; Falcon drives a CARLA vehicle of the nearest
measured size as a **stand-in** and places a prop of the real object's size as a **proxy**
([Indian road assets](indian-assets.md)). Their pixels and points keep **CARLA's class** — the mesh they are: an
autorickshaw stand-in is `class` car, `category` vehicle.car; the cattle proxy `static`, the handcart proxy `dynamic`
(both `movable_object.pushable_pullable`) — and their rows add **`subclass`**: `autorickshaw (stand-in)`,
`cycle-rickshaw (stand-in)`, `cattle (proxy obstacle)`, `handcart (proxy)`, `street vendor (proxy)`. run.json's
`subclasses` maps actor ids to them; a proxy's `box3d` and `actors.jsonl` box are its true size (CARLA reports a scaled
mesh's box centre unscaled; Falcon writes the real one).

For training: the **appearance is CARLA's mesh**, not the Indian object's. Training a detector on `subclass` would teach
it what a Microlino or a map table looks like, not an autorickshaw or a cow — use the subclass to **filter or weight**
(e.g. leave stand-ins out of an appearance-based class, or count them as their CARLA class), and use stand-ins and
proxies for what their size, place and motion are real for: tracking and prediction of a three-wheeler's speeds,
occupancy and free space, planning around an obstacle of a cow's footprint, lidar returns of an object that size.
Real imagery (or real meshes, the source-build path) is needed to learn their look.

## Semantic occupancy exports (Occ3D, OpenOcc, SurroundOcc)
`falcon export RUN --format occ3d [--option variant=occ3d|openocc|surroundocc] [--option every=N] [--option
aggregate=K] [--option camera_ray_step=S] [--option lidar=ID]` (or Data ▸ Browse ▸ Export) writes one occupancy sample
per lidar sweep of a run with a labelled lidar (`core/occexport.py`). The layouts, as checked against the public
loaders and metrics on 2026-10-30:

| variant | file | arrays | grid | classes |
|---|---|---|---|---|
| `occ3d` — Occ3D-nuScenes ([Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D); the CVPR 2023 challenge's `LoadOccGTFromFile`, `Metric_mIoU`, `evaluate_miou`) | `gts/<scene>/<token>/labels.npz` | `semantics`, `mask_lidar`, `mask_camera`: uint8 (200, 200, 16), masks 0/1 (read with `.astype(bool)`) | 0.4 m over [−40, −40, −1, 40, 40, 5.4] m | 0–16 the nuScenes-lidarseg 16-class mapping (0 `others`), 17 `free` |
| `openocc` — OpenOcc v2 ([OpenDriveLab/OccNet](https://github.com/OpenDriveLab/OccNet), its loader and `ray_metrics.py`) | `openocc_v2/<scene>/<token>/labels.npz` | `semantics` uint8 (200, 200, 16); `flow` float32 (200, 200, 16, 2), m/s; the two masks too | as Occ3D | car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier, driveable_surface, other_flat, sidewalk, terrain, manmade, vegetation, 16 `free` |
| `surroundocc` — SurroundOcc ([weiyithu/SurroundOcc](https://github.com/weiyithu/SurroundOcc), `LoadOccupancy`) | `gts/<scene>__<token>.npy` | float32 (N, 4) `[ix, iy, iz, label]` of the occupied voxels | 0.5 m over [−50, −50, −5, 50, 50, 3] m (200 × 200 × 16) | 1–16 barrier … vegetation, 0 ignored |

`annotations.json` follows Occ3D's: `train_split`, `val_split` (and `test_split`), `scene_infos[scene][token]` with
`timestamp` (µs), `camera_sensor[CAM_*]` (`img_path` into `samples/`, OpenCV `intrinsic`, camera → ego `extrinsic` and
`ego_pose` as translation + quaternion w, x, y, z), `ego_pose`, `gt_path`, `prev`, `next`. A scene is a run; a token
is md5(run, step). Axes are nuScenes' (x forward, y **left**, z up): Falcon mirrors y. The origin is Falcon's vehicle
origin (CARLA's: the box centre at ground height), not nuScenes' rear axle.

How a voxel gets its class: the labelled lidar points vote (majority; with `aggregate=K` the static points of K sweeps
either side join, moved into the sample's ego frame — Occ3D's multi-frame densification for the static scene), then
every actor's 3D box fills its voxels by its nuScenes category (two-wheelers whole, rider included); the ego is left
out. `mask_lidar`: the voxels each return's ray (Amanatides & Woo traversal from its sweep's sensor origin) passes and
ends in. `mask_camera`: from each pinhole camera, rays through the pixel grid (one every ⌊camera_ray_step · 0.4 m /
56.6 m · f⌋ pixels — the far corner's voxel angle) up to and including the first occupied voxel (Occ3D Section 3.3.2).

| Falcon (CARLA tag or actor category) | Occ3D / devkit coarse class |
|---|---|
| road, road line | driveable_surface |
| sidewalk | sidewalk |
| terrain | terrain |
| ground, water, rail track | other_flat |
| building, wall, fence, pole, traffic light, traffic sign, static, bridge, guard rail | manmade |
| vegetation | vegetation |
| car (and the autorickshaw / cycle-rickshaw stand-ins), truck, bus, motorcycle, bicycle (with rider), pedestrian | car, truck, bus, motorcycle, bicycle, pedestrian |
| ambulance, police car, train, CARLA's dynamic and other, the cattle / handcart / vendor proxies (`movable_object.pushable_pullable`) | the devkit's ignore class: Occ3D `others`, SurroundOcc 0; **OpenOcc has no such class: written as free**, counted in the manifest |
| unlabelled, sky, weather noise (lidar) | no vote (their rays still count for `mask_lidar`) |

The mapping is the nuScenes devkit's own (`LidarsegClassMapper.get_fine2coarse`), compared with it by the tests.
**Never produced**: construction_vehicle, trailer, traffic_cone, barrier — CARLA has no such actor (and no tarpaulin,
debris or bicycle rack); the manifest lists them under `left_out`.

**Flow** (OpenOcc only): the actor's velocity in the world, expressed in the sample's ego frame (x forward, y left),
in every voxel of its box; 0 for the static scene. Its shape and unit (m/s, the challenge's AVE) were verified; its
frame is not stated in OpenOcc's code — Falcon's choice follows nuScenes box velocities. **Not written**:
OpenOccupancy / nuScenes-Occupancy (its layout was not verified), OpenOcc's mmdet3d `.pkl` infos (annotations.json
carries the same facts), Occ3D-Waymo.

Read back (2026-10-30, the 24-thread reference machine under other agents' load) from two recorded 15 s drives of the demo
project (32-channel lidar, 20 Hz), 300
samples each, independently of the exporter — numpy only, as the Occ3D loaders read it, and a second visibility test
by a fixed-step (2 cm) march from the camera to a 3 × 3 × 3 lattice of points in each sampled voxel: every sample had
the shapes, dtypes and value ranges above.

| drive | aggregate | export | occupied | observed (lidar / camera) | visible but every sight line blocked | hidden but a sight line clear |
|---|---|---|---|---|---|---|
| Town10 | 0 | 190 s | 1.08 % | 25.9 % / 17.9 % | 887 of 60 000 (1.5 %) | 791 of 60 000 (1.3 %) |
| Town10 | 5 | 698 s | 1.94 % | 30.7 % / 16.7 % | 1 259 (2.1 %) | 3 223 (5.4 %) |
| Central Bengaluru (imported) | 0 | 128 s | 0.43 % | 8.6 % / 16.4 % | 523 (0.9 %) | 28 (0.05 %) |
| Central Bengaluru | 5 | 418 s | 0.46 % | 8.9 % / 16.4 % | 558 (0.9 %) | 56 (0.1 %) |

The disagreements are the two methods' differences at voxel edges and between camera rays (a voxel seen through a
sliver only; one the ray spacing skipped), most where aggregation makes the scene denser. Denser rays trade one for
the other (Town10, aggregate 5, every 10th sweep, 30 samples, 6 000 voxels each way):

| `camera_ray_step` | export | visible but every sight line blocked | hidden but a sight line clear |
|---|---|---|---|
| 1 (default) | 70 s | 135 (2.3 %) | 316 (5.3 %) |
| 0.5 | 128 s | 233 (3.9 %) | 158 (2.6 %) |
| 0.25 | 394 s | 333 (5.6 %) | 57 (1.0 %) |

— more rays reach voxels through gaps narrower than the check's 27-point lattice sees. Aggregating 5 sweeps either side more than doubled the driveable surface voxels on Town10
(463 636 → 1 101 570) at 3.7 × the time; on the Bengaluru drive it added little (0.43 → 0.46 % occupied). The
Bengaluru drive's classes were manmade 735 k, sidewalk 58 k, driveable
surface 35 k, motorcycles 4 k, bicycles 108 voxels (no car near its lidar in that drive).

Limits: one sweep — or K static ones — is far sparser than Occ3D's whole-sequence aggregation and mesh-based
densification; free space is known only along rays that hit something; a moving actor's points lie where it was in its
slice of the sweep, its box where it is at the step; a far voxel narrower than the ray spacing can fall between camera
rays (`camera_ray_step` < 1 casts more).
