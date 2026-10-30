# Dataset format (falcon-raw)

Every recorded drive (and every live recording) is a folder in this layout. Exports to COCO, YOLO, KITTI, nuScenes,
Waymo-style, BDD lanes and ROS 2 MCAP are made from it (Data ▸ Export, `falcon export`).

```
run/
  drive.json          the input: world, weather, rig, traffic, seed, step, …   (live: live.json)
  run.json            what ran: versions, seed, hashes, timings, measured performance, lidar checks, faults applied;
                      traffic: the vehicles placed, and on an imported world speed_source — the traffic's speeds
                      from the file's limits ("map") or CARLA's own ("carla"), see worlds.md, and keeper — the
                      traffic moved away from roads that end (counts, events), and lateral_filtering — the
                      two-wheelers that filter laterally (controller, parameters, counts, offset histogram,
                      ms per update; indian-assets.md); lights: vehicle lights set and read
                      back (drives, live, stack, gym, HIL); city: an imported
                      world's buildings, trees and lamps placed (the plan's SHA-256, spawned, failed, seconds);
                      derived: products made from the run afterwards into its folder (audio: path, schema, the
                      parameters and their SHA-256, created)
  dataset.json        sensors (K, lens, views, planned pose, streams), label encodings and classes, provenance
  frames.jsonl        one line per simulation step (below)
  lanes.json          the map's driving lanes and lane graph (labelled drives)
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
  audio/                         SYNTHETIC array audio derived after the drive (`falcon derive audio`, below):
                                 audio.wav, ground_truth.jsonl, events.jsonl, audio.json, doa.json
  gnss/<gnss>.jsonl  imu/<imu>.jsonl
```

## Frames
`frames.jsonl`: `step`, `frame` (CARLA's), `sim_t` (s from the first recorded step), `ego` (pose in CARLA's world),
`speed_mps`, `control`, `recorded` (false on steps `record_every` skips), `teleport` (on the step the road-end guard put the ego back on the road: `cause` (`dead_end_ahead` or `fell`), from, to, detail — the pose jumps there; see [worlds](worlds.md#roads-that-end-the-edge-of-an-imported-area)), `traffic_respawns` (on a step the traffic keeper moved traffic away from a road that ends: per vehicle kind — `dead_end`, `fell`, `lost`, `replaced` — actor, from, to and, after a stop in view, `held_s`; those actors' poses jump there), `files` (`"<sensor>/<kind>": path`), and
when used `stamps` (per sensor, with its clock-offset fault), `faults` (per sensor: `drop` or `freeze`), `lidar_pose`
(per lidar: the 3 × 4 world matrix of the sweep's end, row-major), `physics` (per sensor: the weather effects that
touched the frame — [sensor physics](sensor-physics.md)), `odd` (`in`, `why`, `road`, … — [ODD
monitoring](odd-monitoring.md)) and `lateral` (when the traffic profile turns lateral filtering on: `vehicles` it
holds, how many are in each mode — `filter` (off their lane centre), `lane`, `junction` (back on the centre near a
junction), `held` (stopped by the traffic keeper) — [Indian road assets](indian-assets.md#lateral-filtering)).

## Lateral filtering in actors.jsonl
With lateral filtering on, the row of every vehicle it holds carries **`lateral`**: `offset_m` (its measured lateral
offset from the centre of its reference lane, metres, **+ to the driver's right**, CARLA's right vector (−sin yaw,
cos yaw) — in left-hand traffic the kerb side is negative), `target_m` (the offset commanded that step), `lane`
([road id, lane id] of the reference lane: the lane the traffic manager drives for the `tm` controller, the vehicle's
own route for `own`), `mode` (as in frames.jsonl), `gap_m` (the clearance ahead along its path, 30 = nothing within
the look-ahead), `cap_kmh` (the clearance's speed cap, `null` when none) and, for the `tm` controller, `ignoring`
(true while the traffic manager's vehicle detection is off for it: it is filtering past an object on its lane's
centre). The offset is measured on the reference lane's polyline, not on the nearest lane: a vehicle riding the line
into the next lane has an offset beyond half the lane's width. Junction steps are in mode `junction`.

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

## Synthetic audio (derived)
`falcon derive audio RUN` (or Data ▸ Browse ▸ *Derive audio*, or `falcon drive … --audio`) writes `RUN/audio/`
(schema `falcon.audio/1`; the model, its sources and its validation are in [audio](audio.md)). **Every sound is
synthetic** — CARLA has no audio; nothing here was recorded.

- `audio.wav`: WAVE_FORMAT_EXTENSIBLE, one channel per microphone of the rig's `audio_array` sensor (channel k = mic k),
  48 kHz; float32 samples are **sound pressure in pascals** (dB SPL = 20·lg(rms / 20 µPa)), or 24-bit PCM with
  `full_scale_pa` (default 200 Pa) as full scale. Sample 0 is `sim_t` 0, the first recorded step.
- `ground_truth.jsonl`: one line per simulation step — `step`, `sim_t`, `sources`, loudest first: every source received
  at `audible_db` (default 30 dB SPL) or more — `id` (the actor id of actors.jsonl), `type_id`, `class` (car, van,
  bus, truck, two_wheeler, autorickshaw), `kind` (engine, siren, horn), `az_deg` and `el_deg` in the **array's frame**
  (azimuth 0° ahead, positive to the right, CARLA's sense; elevation up) and `distance_m` of the source's retarded
  position (where the sound arriving now was emitted), `spl_db` (its received mean-square pressure over the step, mean
  of the microphones, direct + ground path), `spl_dba` (an A-weighted estimate), `doppler` (dτ/dt at the array's
  centre: received / emitted frequency), `speed_mps`.
- `events.jsonl`: horn events (`actor`, `class`, `rule` — the heuristic rule that fired, or `explicit` — `start_s`,
  `end_s`) and siren spans (`actor`, `pattern`: wail, yelp or wail_yelp), in time order.
- `audio.json`: the array (pose in the vehicle frame, mic positions in the array and vehicle frames), every parameter
  (temperature, humidity, ground reflection, occlusion, horn profile, seed, …), the models and their sources, the
  limits, the cost and the provenance (SHA-256 of run.json, frames.jsonl, actors.jsonl, drive.json). When the folder is
  inside the run, run.json gets `derived.audio` with the same parameters.
- `doa.json`: SRP-PHAT direction-of-arrival estimates on the array output against the ground truth (median and 90th
  percentile azimuth error; per frame samples).

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

