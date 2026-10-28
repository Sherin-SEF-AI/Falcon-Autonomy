# Changelog

## 0.3.0 — milestone 3: rigs and data engine

- Rig editor: pinhole and fisheye cameras, lidar, radar, GNSS, IMU; poses by numbers or by dragging in a 3D view
  (OpenGL, software fallback); cost per sensor against the render budget; import/export as Falcon JSON, ROS URDF/TF
  and nuScenes calibration.
- Fisheyes rendered as the cube faces their lens needs and remapped (Vantage's lens model and seam feathering); the
  budget costs them from the same plan (it used to ignore a fisheye's lens block).
- Camera ground truth from CARLA's instance and depth cameras: semantic, instance and range images; objects with track
  ids, tight and full boxes, truncation, occlusion, range, 3D boxes and masks. Lane lines and the lane graph.
- Lidar labels proven point for point every step; radar, GNSS and IMU. Measured: 100 % of 262 282 points in vehicle
  boxes labelled vehicle; lidar against camera range 1.15 cm median absolute difference.
- Imported worlds: CARLA's label cameras do not draw a generated world's road mesh (measured); its ground is labelled
  from the map (camera pixels with range, lidar points).
- Sensor faults of known size, measured back by their tests; outputs keep the planned calibration.
- The data engine: dataset specs as targets, plans with cost, cache-keyed resumable drives, coverage, versions with
  leak-free splits; the Datasets tab; browser layers and objects.
- Exporters (COCO, YOLO, KITTI, nuScenes with lidarseg and panoptic, Waymo-style JSON, BDD lanes, ROS 2 MCAP) checked
  with pycocotools, nuscenes-devkit and the MCAP ROS 2 decoder.
- Not yet: optical flow (milestone 5).

## 0.2.0 — milestone 2: worlds and live

- Live sessions: chase, free and sensor views streamed over a local socket (latest frame only, drops counted), ground-
  truth boxes, a top-down map; manual driving by keyboard or Linux joystick/wheel (input loss brakes), CARLA's
  autopilot, or a policy plug-in (built-in pure-pursuit lane follower); live events (pedestrian, cut-in, weather,
  sensor failure); recording as labelled runs. Measured: 1280 × 720 at 20.0 frames/s, real-time factor 1.0.
- OpenStreetMap import from .osm and .osm.pbf with an area picker, a centred projection, the road-graph preview, a lane
  check against OSM, left-hand conversion (CARLA 0.9.16 verified to honour rule="LHT"), and a fix for converter
  crosswalks lying past their road's end (which crash CARLA's world generation).
- Traffic profiles, the Indian mixed-traffic profile (autorickshaw substitute labelled), a handedness guard and a
  measured handedness verification; pedestrians on the navigation mesh.
- Acceptance, central Bengaluru (1 km²): drivable with traffic 12–17 s after the import starts; 22 of 23 sampled ways
  match OSM's lanes; traffic measured keeping left in 100 % of 1044 samples.
- Known limit, reported per run: CARLA labels a generated world's road and sidewalk meshes as sky.

## 0.1.0 — milestone 1: foundation (unreleased)

- App shell: navigator, Jobs and Log docks, undo/redo through project edits, command palette (Ctrl+K), dark and light
  themes, status bar (server, GPU, memory).
- Project format version 1 with JSON Schema, migration from the pre-release version 0, validation on load and save.
- Job queue with a machine-wide CARLA lock, a GPU lock and a memory guard; QProcess launcher (GUI) and subprocess
  launcher (CLI) over the same specs and protocol.
- CARLA server manager: headless, memory-capped systemd scope, NVIDIA Vulkan ICD pinned, own ports, never a
  user's server or CARLA's default ports unless chosen.
- Workers (probe, tidy, drive) with signals blocked before numpy, a watchdog, run.json and exit codes.
- Drive: synchronous fixed step, seeded traffic, cameras with semantic label cameras at the same pose and tick.
- Data page (record, browse with label overlay and legend), Worlds page (towns, weather), Home, Settings, Help.
- First-run wizard with a 30 s smoke drive that records the machine's throughput.
- CLI: doctor, project, probe, smoke, drive, tidy, reproduce, runs, schema.
- Found and fixed on a real 0.9.16 server: `ActorAttribute` has no `.value` (use `as_str()`); destroying the ego
  before its sensors, ticking after destroying traffic-manager vehicles, or `Actor.destroy()` on any vehicle that was
  on the autopilot ends the client process with `std::terminate` — vehicles are now destroyed by id in a batch.
