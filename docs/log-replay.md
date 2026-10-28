# Log replay

Real logs replayed in CARLA: **import** a drive (nuScenes, Waymo Open Dataset v2, a ROS 2 bag, KITTI raw, or a Falcon
run), **rebuild** its scene — on its own road when the map matches, otherwise flagged as having no matching road —
**re-simulate** it with changes (an actor's timing or speed, an added pedestrian, the weather, an actor removed), and
**compare** how the ego, a stack's policy or a perception model behaves between the original and the changed replay.

The code is `core/logreplay/` (the formats, the log, changes, placement, map matching, the comparison) and
`workers/replay.py` (the replay in CARLA's Python). The page is *Replay*.

```sh
falcon replay import nuscenes:~/data/nuscenes --scene scene-0061 --out ~/demo.falcon/logs/scene-0061
falcon replay import ros2:~/bags/drive_01 --town Town10HD_Opt --out ~/demo.falcon/logs/drive_01
falcon replay match ~/demo.falcon/logs/drive_01 ~/demo.falcon.json --world world-town10
falcon replay run ~/demo.falcon.json --log ~/demo.falcon/logs/drive_01 --out RUN_A --driver follower
falcon replay run ~/demo.falcon.json --log ~/demo.falcon/logs/drive_01 --out RUN_B --driver follower \
    --changes changes.json --seed 3
falcon replay compare --original RUN_A --changed RUN_B --out CMP [--synthetic-detector | --project P --model ID]
```

`run` also takes `--world`, `--rig`, `--fps` (20), `--t-start` / `--t-end` (a window of the log), `--anchor-spawn K`
(where a log without a matching road goes), `--ego-offset M`, `--allow-mismatch`, `--driver plugin --policy
MODULE:CLASS`. Exit codes are the CLI's ([cli](cli.md#exit-codes)).

## The log

`log.json` (schema `falcon.log/1`) holds trajectories only — never a dataset's images or points:

| field | meaning |
|---|---|
| `ego`, `actors[]` | tracks: `id`, `class` (car, truck, bus, motorcycle, bicycle, pedestrian), `source_class`, `size` (length, width, height in m) and `size_source` (`source` or `typical` — a heuristic size, listed in `heuristic`), `max_gap_s`, `hold`, and samples `t` (s from the ego's first sample), `x`, `y`, `z` (m), `yaw` (degrees), `pitch`, `roll`; optionally `blueprint` (the CARLA blueprint of a Falcon run's actor) |
| `map` | what the coordinates refer to: `carla` (a town name, or an OpenDRIVE file and its SHA-256), `nuscenes`, `waymo`, `geo` (a geographic origin `lat0`, `lon0`) or `unknown` |
| `source` | the format, the files read with their SHA-256, the scene or segment, `t0_s` (the source time of t = 0) |
| `ego_origin` | the point of the ego the source's poses give (nuScenes: the rear axle; Falcon: CARLA's vehicle origin) |
| `notes`, `left_out`, `heuristic` | what was read, what was not and why, every value that is not the source's |

An actor's position is the **centre of its box**; the ego's is the source's ego origin. A track is observed between
consecutive samples at most `max_gap_s` apart and absent outside its samples (unless `hold`).

## Formats: what is read

| format | read | not read |
|---|---|---|
| **nuScenes** (`v1.0-*` JSON tables, no devkit) | the scene's samples and log (its map); the ego from the `ego_pose` of every sample_data of one channel (LIDAR_TOP, about 20 Hz with its sweeps); actors from `sample_annotation` per instance (box centre, size, rotation; category → class), interpolated between key frames as the devkit's `get_boxes` does (centre linearly, yaw by slerp), across at most one missed key frame (1.05 s, heuristic) | images, lidar, radar, attributes, visibility; static objects, cones, barriers, debris, bicycle racks and animals (counted in `left_out`) |
| **Waymo v2** (Parquet, `<root>/<component>/<segment>.parquet`, pyarrow) | `vehicle_pose` (`[VehiclePoseComponent].world_from_vehicle.transform`: the ego); `lidar_box` (centre, size, heading in the vehicle frame of each frame, type: vehicle, pedestrian, cyclist; signs left out); `stats` (location, time of day, weather) | `camera_box` (2D image boxes have no 3D pose: counted); images, lidar returns, maps. **v1 TFRecords are not supported**: they need the dataset's .proto definitions, which are not on this machine, and decoding them without them could not be verified |
| **ROS 2 bags** (rosbag2 folders, `.mcap`, `.db3`) | the ego from `/tf` + `/tf_static` (the chain from `map`, else `odom`, else the root, to `base_link`, `base_footprint`, `ego_vehicle` or `hero` — or the frames given), else `nav_msgs/Odometry`, else `geometry_msgs/PoseStamped` / `PoseWithCovarianceStamped`, else `sensor_msgs/NavSatFix` (geo-located; heading from the direction of motion — heuristic); actors from `vision_msgs/Detection3DArray` and `autoware_perception_msgs/TrackedObjects` (or `autoware_auto_perception_msgs`), each box taken to the world frame through TF at its stamp | detections without an id (they cannot be followed: counted); classes naming none of Falcon's (counted; `--unknown-by-size` classifies them by length, heuristic); every other topic |
| **KITTI raw** (`…_sync/oxts/`) | the ego from OXTS (lat, lon, alt, roll, pitch, yaw), geo-located at its first fix | the tracklets (a separate download), images, velodyne |
| **Falcon runs** (falcon-raw folders) | the ego from `frames.jsonl`, every actor from `actors.jsonl` with its blueprint; the map from `drive.json` | — |

**ROS 2 messages are decoded by Falcon's own CDR decoder** (`core/logreplay/cdr.py`), schema-driven: it reads the
ros2msg definitions the bag stores (an MCAP's schema records; a `.db3`'s `message_definitions` table from rosbag2
Iron on). A `.db3` from before Iron stores none: the standard types above and `vision_msgs` 4.x (Humble) are built in;
a topic of any other type is refused, saying so. Only plain CDR (XCDR1, either byte order) is read; parameter lists
and XCDR2 are refused, as is `wstring`. The `mcap` package reads the container (chunks, zstd, lz4).

**Verified**: the decoder field by field against `mcap_ros2`'s decoder on bags written by `mcap_ros2`'s own encoder
(TF, Odometry, PoseStamped, Detection3DArray, autoware TrackedObjects); Falcon's MCAP export round-trips exactly (next
section); `.db3` files with and without stored definitions; nuScenes tables against the nuscenes-devkit — every sweep's
interpolated box equal to `NuScenes.get_boxes` within 10⁻⁶ m and 10⁻⁶ ° — and the devkit's geographic reference
reproduced exactly; KITTI raw on the three real drives on this machine (below). Waymo: synthetic files in the
documented v2 layout only — **not verified on real Waymo files** (none here; Falcon downloads no datasets). The
autoware definitions used in the tests are written from the package's published messages, not taken from an
installed package.

### Falcon's own bags round-trip

`falcon export RUN --format mcap --option objects=true` adds every actor's box, every step, as a
`vision_msgs/Detection3DArray` on `/falcon/objects` (frame `map`, id the CARLA actor id, the ego's own box as id
`ego`, class_id its nuScenes category). Imported back (the town is read from the export's `falcon_export.json`), the
trajectories are the run's own: on the fake, an 8 s scenario drive's ego and actor poses equal `frames.jsonl` and
`actors.jsonl` to within 10⁻⁹ m and degree, and times to 10⁻⁹ s (stamps are kept in integer nanoseconds from a whole
second, not as float seconds since 1970). Without the option, the bag carries the ego only.

## Frames

Every source is right-handed with z up — nuScenes' global frame (x east, y north), Waymo's world and vehicle frames,
ROS's `map` / `odom` (REP 103/105), KITTI's OXTS (east-north-up) — and CARLA is left-handed (x forward, y right,
z up). A pose (R, t) becomes CARLA's by the reflection S = diag(1, −1, 1): t → S·t, R → S·R·S, then CARLA's Euler
angles of that matrix (Transform::GetMatrix's convention). For a heading alone: yaw → −yaw (turning left 30° is CARLA
yaw −30°); nose up stays positive pitch; left side up stays positive roll. The tests check this on known poses and,
for random poses with roll and pitch, that a body point lands at S·(R·p + t) both ways. On the KITTI drives the
direction of motion of the imported positions agrees with the imported yaw to a median 0.09–1.2°.

## Where a log goes: placement and map matching

| placement | when | coordinates |
|---|---|---|
| `matched` | a log recorded in this very world: the same CARLA town (Town10HD and Town10HD_Opt share their roads) or the same OpenDRIVE file (SHA-256) | as they are |
| `geo` | a geo-located log (nuScenes with the devkit's reference corner, a NavSatFix-only bag, KITTI raw) and a world with a geographic origin (an OpenStreetMap import, or an OpenDRIVE geoReference with `+proj=tmerc +lat_0 +lon_0`) whose area holds at least 99 % of the ego's positions | through latitude and longitude into the world's projection |
| `anchored` | anything else — **no matching road** | moved rigidly so the ego's first pose sits on a spawn point of the map (`--anchor-spawn`) |

**The measure.** The ego's positions, every 1 m of path (a stop counts once), against the world's driving-lane
centres: for each, the nearest lane within 8 m whose direction is within 45° of the ego's heading, and the lateral
distance across it. Reported: the share of positions on a driving lane in their direction (lateral distance ≤ half
the lane's width), on a lane of either direction, the wrong way only, with no lane within 8 m, and the median, 95th
percentile and largest lateral distance. **The verdict** (heuristic thresholds, stated in every result): `match` when
≥ 90 % are on a lane in their direction and the 95th percentile is ≤ 2 m; `partial` when ≥ 50 % are on some lane;
else `none`.

**Where the lanes come from.** The replay worker measures on the server's own map (`Map.generate_waypoints` every 2 m)
— the check that decides — and records it in `replay.json`. A `matched` or `geo` placement measured `none` is
**refused** with its numbers (`--allow-mismatch` replays it anyway, flagged). An `anchored` replay is always
`road_matches: false`, said in run.json's notes, `replay.json` and the run's scenario block. Offline (`falcon replay
match`, the page's *Check the map*), the lanes come from the world's OpenDRIVE file — each driving lane's centre from
the reference line, laneOffset and lane widths, every metre (planar; elevation and superelevation ignored; the
header's `<offset>` is not applied, as CARLA builds from the planView coordinates — not verified on a file with an
offset) — or from the lanes.json of a labelled run of the same town; a town without such a run is measured only on
the server.

**Geography's limits.** The nuScenes reference corners are the devkit's own approximate ones. Falcon keeps the global
frame's metres (a transverse Mercator about the corner); the devkit's own conversion (a great circle on a sphere of
the equatorial radius, which `nuscenes_latlon` reproduces exactly) stretches east-west distances by about 0.15 % at
Boston, so the two disagree by about 1.5 m a kilometre from the corner. A matching OpenStreetMap world must be
imported by the user; none is bundled. Waymo publishes no geographic reference: a Waymo log always replays
`anchored`.

## The replay

`make_plan` (core) resamples the changed log to the replay's fixed step (20 steps/s by default): step i shows log time
t_start + i / fps; each actor is interpolated as its track says and its velocity is the central difference of its
own samples. The plan travels inside the run's `drive.json` (a drive config with a `replay` block), so the run folder
is a complete input: `falcon reproduce` re-runs a replay through the drive worker.

The worker (`workers/replay.py`, inside the drive's loop — the same rig, labels, lidar, frames.jsonl, actors.jsonl,
lanes.json, dataset.json, run.json as any drive, so every downstream tool works on replays):

- **Actors** are spawned once — the log's own blueprint when it names one the server has (a Falcon run), else one of
  its class chosen with the seed — parked 3 km away while unobserved (and left out of actors.jsonl then), and
  otherwise placed with `set_transform`, physics off: **kinematic, so repeatable bit for bit**. The actor's origin is
  put at centre − R·(its blueprint's box centre), so the replayed box is centred where the logged one was. **CARLA
  cannot scale a mesh**: the replayed box is the blueprint's size; `replay.json` lists each actor's logged and
  replayed size.
- **Heights**: a `matched` log keeps its own; otherwise each box stands on the map's ground under it (the nearest
  driving lane's height).
- **The ego** (`--driver`): `log` — placed at its logged pose every step (frames.jsonl `control` is null: a log's
  control is unknown); `follower` — the built-in lane follower drives it along the logged path, asked for the logged
  ego's speed at the same moment of the log (a logged stop is a stop, and it goes again when the log did) but never
  more than lets it stop at the path's end at 3 m/s², braking for any replayed actor within 1.8 m of its path ahead
  or that will be within 2 s at its present velocity (a pedestrian about to cross; heuristic), integrated with the
  director's kinematic bicycle model (3 m/s² full throttle, 8 m/s² full brake, 70° wheel angle); `plugin` — a policy
  plug-in (`module:Class`, the same observation). `replay.json` gives a re-driven ego's deviation from the logged path.
  The ego vehicle is the rig's; its CARLA origin goes where the source's ego origin was (`--ego-offset` moves it
  forward: a nuScenes ego origin is the rear axle, about half a wheelbase behind a CARLA vehicle's origin).
- **Records**: `trajectories.json` (each step's box centre, heading, velocity and footprint of the ego and every
  observed actor), run.json's `scenario` block (core.criticality and the no-collision check; collision-sensor events
  named by the log's ids) and `replay.json` (placement, anchor transform, the measured match, the changes, the actors'
  blueprints and sizes, the ego driver).

Background traffic is off: a replay holds the log's actors only.

## Changes

A JSON list (schema `falcon.replay_changes/1`), applied in order to a copy of the log; each draw from a generator
seeded by the replay's seed and the change's position, so **the same log, changes and seed give the same replay**
(the tests run each replay twice and compare trajectories.json, actors.jsonl and every image and label exactly).

```json
[{"op": "time_shift",  "actor": "101", "dt_s": 1.5, "outside": "absent"},
 {"op": "speed_scale", "actor": "101", "factor": 0.7, "from_s": 2.0},
 {"op": "remove",      "actor": "102"},
 {"op": "add_pedestrian", "id": "ped_1", "at_ego_s": 8.0, "from": "right", "start_m": 6.0, "walk_m": 12.0,
  "speed_mps": [1.0, 1.8], "t_s": 8.0, "lateral_m": 0.0},
 {"op": "weather", "preset": "WetCloudyNoon", "params": {"fog_density": 40}}]
```

- `time_shift`: the actor's timeline later by `dt_s` (earlier if negative); outside its samples absent, as in the log,
  or `"hold"` at its first / last pose.
- `speed_scale`: re-timed along its own path — after `from_s` (default its first sample) sample times become
  from_s + (t − from_s) / factor; the path is unchanged. `factor` 0 stops it at `from_s`, held there (an instant stop:
  kinematic, not physical).
- `add_pedestrian`: crosses the ego's **logged** path at right angles at the point the ego was at `at_ego_s` (or at
  `"at": {"x", "y"}`), shifted `lateral_m` to the right; starts `start_m` to the given side, walks `walk_m` at
  `speed_mps` (a `[lo, hi]` range is drawn, seeded), reaching the crossing point at `t_s` (default `at_ego_s`: when
  the logged ego was there — a conflict for a logged ego). Before and after it stands. Size 0.6 × 0.6 × 1.75 m
  (typical) unless given.
- `weather`: a CARLA preset of core.weather and/or fields over the world's weather.
- `actor` may be `ego` for `time_shift` and `speed_scale` (they change the logged ego the replay follows).

## Comparing

`falcon replay compare` (the page's *Compare*) pairs the original run A with each changed run B **by step**: they must
replay the same log over the same window at the same step (else refused).

**Driving behaviour**, per run, from trajectories.json with core.criticality ([methods](methods.md#criticality)):
the minimum time to collision and with whom, the minimum box-to-box distance, the largest required deceleration, the
post-encroachment time, collisions (boxes overlapping, or the collision sensor), the criticality scalar (heuristic);
and the ego's path driven, mean and minimum speed, time stopped (below 0.1 m/s) and, re-driven, its deviation from
the logged path. Per pair: every difference B − A, and the ego's largest position and speed difference step by step,
and when.

**Intervals.** A kinematic replay is deterministic, so one pair is one sample: no interval, and the result says so.
With several changed runs — the same change drawn with several seeds (a pedestrian's speed from a range), or several
changes — each mean difference gets a 95 % bootstrap interval over the pairs (seeded), flagged as not reliable below
five pairs.

**A model** (`--project P --model ID`, or `--synthetic-detector`, which is not a model): core.harness runs it on every
n-th frame of one camera of A and of each B against each run's own ground truth; frames are paired by step and
core.harness's paired bootstrap gives each number's difference B − A — over pairs of runs when there are several, else
over frames (not independent: too narrow, and the result says so). `predictions-*.jsonl` keep every frame.

`comparison.json` (schema `falcon.replay_comparison/1`) holds all of it, with provenance over each run's run.json.

## Measured

**KITTI raw** (real data, 2011_09_26 drives 0005, 0015, 0027): the speed of the imported positions against the OXTS
unit's own forward speed, median absolute difference 0.12, 0.03 and 0.04 m/s (mean speeds 4.5, 11.9 and 17.9 m/s);
the direction of motion against the imported yaw, median 1.2°, 0.09° and 0.36°.

**On the fake carla** (the tests' stand-in: a straight road; it shows the code works, not how CARLA renders), an 8 s
kinematic `lead_brake` drive (lead braking at 6 m/s² from 36 km/h, 15 m ahead):

- MCAP export with objects 0.07 s, import 0.03 s; the replay with the logged ego in the same town reproduced the
  drive's ego poses exactly and the lead's poses exactly with the run's own blueprint (through the MCAP, which carries
  no blueprint, a different car of the class was chosen: its box centred where the logged one was, its origin 3 cm
  lower for its different box); road match 100 % in direction, lateral 0.0 m.
- Replays ran at 65–80 steps/s with a 96 × 54 camera with labels.
- A pedestrian crossing the logged path at 5 s (1.5 m/s, from 6 m to the right): the **logged ego** hits it (collision
  0 → 1, minimum TTC 2.77 → 0 s); **re-driven by the lane follower** it slows to 0.19 m/s for it, no collision,
  minimum TTC 2.44 → 2.20 s (now with the pedestrian), minimum distance 3.90 m to it, 7.8 m less path driven and up
  to 9.2 m behind the original replay at 5.7 s.
- The lead at half speed: minimum TTC 2.44 → 2.55 s, 6.9 m less path; the lead 1 s late (held at its first pose):
  minimum TTC 2.44 → 1.16 s, minimum distance 6.89 → 5.71 m.

**On a real CARLA 0.9.16 server** (Falcon's own, Town10HD_Opt, capped at 8 GB, 2026-10-28): a 10 s drive on the
traffic manager with 15 vehicles and a 640 × 360 camera with labels as the source.

- MCAP export with objects → import: all 15 actors and the ego, positions equal to the run's own to 0.0 m; the town
  read from the export's manifest.
- The replay with the logged ego and the run's own blueprints rebuilt it: ego poses equal to 10⁻⁴ m (frames.jsonl's
  rounding), every actor within 2·10⁻⁴ m; road match 100 % of positions on a lane in their direction, lateral
  distance median 0.11 m, 95th percentile 0.34 m (the traffic manager does not drive exactly on the centre line).
- **Repeatable**: the same replay twice — trajectories.json identical, frames.jsonl's ego identical, actors.jsonl
  identical but for CARLA's actor ids (a running server numbers new actors anew).
- Replays ran at 26–27 steps/s (the camera's encoding the bottleneck).
- A pedestrian crossing the logged path at 5 s: the logged ego hits it (collision 0 → 1); the lane follower, re-driven,
  stops for it (3.15 s stopped, no collision, minimum TTC 2.28 s with it, 6.2 m less path), and keeps to the logged
  path within 0.52 m. The synthetic detector on the two logged-ego replays (100 paired frames, 46 and 66 scored
  objects — the pedestrian added): AP 0.605 → 0.581, difference −0.024 [−0.106, 0.085] over frames (one pair: too
  narrow, and said so).
- A log from elsewhere (an unknown map) replayed on spawn point 0: `anchored`, `road_matches: false`, the note in
  run.json — though the anchor happened to fit a lane (100 % in direction): the flag says what the placement is, the
  measure what the road there is. A log claiming Town10HD but 400 m off its roads: refused, "0.0 % of 67 positions on
  a driving lane …".
- `lanes_from_xodr` on Town10HD_Opt's own OpenDRIVE (Map.to_opendrive) against CARLA's waypoints every 2 m (3066
  positions): 100 % on a lane in their direction, lateral distance median 0.0 m, 95th percentile 0.006 m, largest
  0.40 m.
- Through the MCAP (which carries no blueprint) the actors' blueprints are chosen by class with the seed, not by size:
  4.6–5.6 m logged cars were replayed as 3.6–4.8 m cars (each listed in replay.json).

## Limits

- **Map matching is geometric**: position and heading against lane centres, not the lane graph; lenient inside
  junctions; a real place matches only an imported OpenStreetMap world of it (no CARLA town is a real place).
- **No matching road** means the log is replayed rigidly on an anchor of the map: its actors may cross buildings or
  kerbs, and the run says so everywhere; a flat empty world is not generated.
- **Boxes**: CARLA cannot scale its meshes; a replayed actor has its blueprint's size (listed against the log's), and
  without a blueprint in the log the blueprint is chosen by class with the seed, not by size. Static objects (cones,
  barriers) are not replayed.
- **Kinematic motion**: actors move exactly as logged (or changed), with no dynamics; the added pedestrian walks a
  straight line at constant speed; an actor stopped by `speed_scale` 0 stops instantly.
- **A re-driven ego** follows the logged path only (no lane changes or route of its own) and inherits its stops (a
  logged stop at a red light that the replay does not have is reproduced as a limit of 0 there). Trained policy
  folders (core.policy, torch) are not loaded in the worker: the lane follower and policy plug-ins are.
- **Sensors** are CARLA's, rendered anew: a log's own camera, lidar and radar data are not replayed.
- **Waymo v1** (TFRecord) is not read; **Waymo v2** is not verified on real files; **KITTI raw** tracklets are not read.
