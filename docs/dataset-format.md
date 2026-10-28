# Dataset format (falcon-raw)

Every recorded drive (and every live recording) is a folder in this layout. Exports to COCO, YOLO, KITTI, nuScenes,
Waymo-style, BDD lanes and ROS 2 MCAP are made from it (Data ▸ Export, `falcon export`).

```
run/
  drive.json          the input: world, weather, rig, traffic, seed, step, …   (live: live.json)
  run.json            what ran: versions, seed, hashes, timings, measured performance, lidar checks, faults applied
  dataset.json        sensors (K, lens, views, planned pose, streams), label encodings and classes, provenance
  frames.jsonl        one line per simulation step (below)
  lanes.json          the map's driving lanes and lane graph (labelled drives)
  rgb/<camera>/<n>.jpg
  semantic/<camera>/<n>.png      uint8 class per pixel
  instance/<camera>/<n>.png      uint16: 0 none, else an object's id in objects/
  range/<camera>/<n>.png         uint16 = round(range_m × 256) along the pixel's ray; 0: nothing within 256 m
  objects/<camera>/<n>.json      the objects seen (below)
  lidar/<lidar>/<sweep>.bin      float32 x y z intensity, the lidar's frame at the sweep's end
  lidar/<lidar>/<sweep>.labels.bin   uint8 tag per point, then uint32 actor id per point
  radar/<radar>/<n>.npy          float32 velocity, azimuth, altitude, depth (CARLA's order)
  gnss/<gnss>.jsonl  imu/<imu>.jsonl
```

## Frames
`frames.jsonl`: `step`, `frame` (CARLA's), `sim_t` (s from the first recorded step), `ego` (pose in CARLA's world),
`speed_mps`, `recorded` (false on steps `record_every` skips), `files` (`"<sensor>/<kind>": path`), and when used
`stamps` (per sensor, with its clock-offset fault), `faults` (per sensor: `drop` or `freeze`), `lidar_pose` (per
lidar: the 3 × 4 world matrix of the sweep's end, row-major).

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
- **Imported worlds**: the ground's labels and range come from the map ([worlds](worlds.md#labels-in-an-imported-world-a-carla-limit-worked-around));
  `objects.json` → `ground_relabelled` counts the pixels.
- **Lidar**: per point the semantic lidar's tag and actor id, matched point for point and proven every step
  ([methods](methods.md#lidar-labels-and-their-proof)).
- **Lanes** (`lanes.json`): driving-lane waypoints every 2 m with width and left/right lane markings (CARLA's
  LaneMarkingType and colour), and the lane graph (CARLA's topology).
- **Optical flow** is not recorded in this version.
