# Dataset format (falcon-raw)

Every recorded drive (and every live recording) is a folder in this layout. Exports to COCO, YOLO, KITTI, nuScenes,
Waymo-style, BDD lanes and ROS 2 MCAP are made from it (Data ▸ Export, `falcon export`).

```
run/
  drive.json          the input: world, weather, rig, traffic, seed, step, …   (live: live.json)
  run.json            what ran: versions, seed, hashes, timings, measured performance, lidar checks, faults applied;
                      traffic: the vehicles placed, and on an imported world speed_source — the traffic's speeds
                      from the file's limits ("map") or CARLA's own ("carla"), see worlds.md, and keeper — the
                      traffic moved away from roads that end (counts, events); lights: vehicle lights set and read
                      back (drives, live, stack, gym, HIL); city: an imported
                      world's buildings, trees and lamps placed (the plan's SHA-256, spawned, failed, seconds)
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
  cpm.jsonl                      the roadside units' Collective Perception Messages, one line per message (below)
  gnss/<gnss>.jsonl  imu/<imu>.jsonl
```

## Frames
`frames.jsonl`: `step`, `frame` (CARLA's), `sim_t` (s from the first recorded step), `ego` (pose in CARLA's world),
`speed_mps`, `control`, `recorded` (false on steps `record_every` skips), `teleport` (on the step the road-end guard put the ego back on the road: `cause` (`dead_end_ahead` or `fell`), from, to, detail — the pose jumps there; see [worlds](worlds.md#roads-that-end-the-edge-of-an-imported-area)), `traffic_respawns` (on a step the traffic keeper moved traffic away from a road that ends: per vehicle kind — `dead_end`, `fell`, `lost`, `replaced` — actor, from, to and, after a stop in view, `held_s`; those actors' poses jump there), `files` (`"<sensor>/<kind>": path`), and
when used `stamps` (per sensor, with its clock-offset fault), `faults` (per sensor: `drop` or `freeze`), `lidar_pose`
(per lidar: the 3 × 4 world matrix of the sweep's end, row-major), `physics` (per sensor: the weather effects that
touched the frame — [sensor physics](sensor-physics.md)) and `odd` (`in`, `why`, `road`, … — [ODD
monitoring](odd-monitoring.md)).

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

## Roadside units
A drive in a world with roadside units ([V2X](v2x.md)) records their sensors in the same layout and with the same
labels as the ego's, under the id **`<rsu id>.<sensor id>`** (`rgb/rsu-036d8e6f.cam/000042.jpg`,
`lidar/rsu-036d8e6f.lidar/000021.bin`, `objects/rsu-036d8e6f.cam/…`; frames.jsonl `files` and `lidar_pose` likewise).
Their poses and labels are in the **station frame** — CARLA's vehicle frame put at the pole's foot and turned by the
RSU's heading — where an ego's are in its vehicle frame: a roadside camera's `objects` `box3d` (`"frame": "vehicle"`
names that convention) is relative to the pole's foot. No pixel or point of a roadside sensor is class 29 `ego`: the
ego vehicle is an ordinary car to it, with its track id. `dataset.json`: each roadside sensor's entry has `station`
(the RSU id; `null` for the ego's), `infrastructure.stations` the poles (world pose, mast height, ETSI station id,
reference position in latitude/longitude, CPM settings, each sensor's station-frame pose and CPM sensor id), and `ego`
the ego's actor id. `run.json` → `infrastructure.stations`: CPMs sent, objects reported, message sizes; the roadside
lidars' proofs are in `lidar_checks` with the ego's.

`cpm.jsonl`, per message: `step`, `frame`, `sim_t`, `station`, `station_id`, `cpm` (the CollectivePerceptionMessage as
a dict with the ASN.1's own names: CHOICEs a one-key dict, ENUMERATEDs their names), `uper` (its UPER encoding,
base64; `falcon v2x decode RUN` checks every line decodes to its `cpm`), `bytes`, and `objects` — per perceived object
Falcon's extras: `objectId`, `actor_id`, `type_id`, `category`, `distance_m` (from the pole), `evidence` (CPM sensor id
→ instance pixels or lidar points that reached it).

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

