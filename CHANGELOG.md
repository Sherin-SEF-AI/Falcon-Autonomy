# Changelog

## 1.0.0 — milestones 7 and 8: stack, replay, hardware, the Loop

- **The Loop** becomes the home page: model + ODD → evaluate → weak cells (class × range × condition, a heuristic
  rule) → rare-case search there (scenario criticality as the objective, stated) → targeted data → retrain → evaluate
  again on the same held-out set (never trained on: enforced) → before and after with paired intervals. Costs are
  estimated before each step from measured throughput and measured after; iterations are versioned and resumable.
  The former Home is the Loop page's Status tab.
- **Reports**: self-contained HTML (inline CSS/SVG, light and dark) with provenance for loop iterations, evaluations,
  datasets, rare-case studies and Leaderboard results; `falcon report`.
- **Stack testing** (`falcon stack`): measured on Falcon's own server — CARLA 0.9.16's native ROS 2 publishes only
  sensors enabled for ROS (no odometry or vehicle status, a wrong camera_info focal length, control accepted, no
  waiting for a stack), and the official carla-ros-bridge refuses 0.9.16. Falcon's bridge waits each step for the
  stack's control (Autoware Control, Ackermann, CARLA's message) or records a missed deadline: 31–45 steps/s with a
  1280 × 720 camera and a 32-channel lidar. Suites with pass/fail and Wilson intervals; a SOTIF-style safety-case
  report (not a certification claim). Autoware: topic mapping and a compose example, not run here.
- **Log replay** (`falcon replay`): nuScenes, Waymo v2 Parquet, ROS 2 bags (MCAP, .db3), KITTI raw and Falcon runs,
  converted to CARLA's frame, map-matched with numbers (a mismatch is refused), replayed kinematically, changed and
  compared. Measured: a drive's MCAP round trip rebuilt within 0.2 mm on a real server; repeat replays identical.
- **Hardware in the loop** (`falcon hil`): H.264/RTP (NVENC or x264) or raw camera and TCP/UDP or ROS 2 lidar to a
  device, control back, emulated sensor clocks, an NTP-style offset estimate with a bound, 12 latency stages and a
  waterfall. Measured with a local simulated device on a real server: step start to control at the simulation
  22–28 ms median, no frames lost. The Jetson AGX Orin acceptance is one command, pending the device's address.
- **Packaging**: a Linux x86_64 PyInstaller bundle (4.3 GB compressed, torch and CUDA included), an app icon and a
  desktop entry; Ctrl+C in the terminal quits the app cleanly. Windows is not supported (documented).
- Fixed: a crashed CARLA server could not be restarted in the same process (a reused systemd unit name).

## 0.6.0 — milestone 6: the training lab

- Detector fine-tuning (`falcon train`, Training ▸ Detectors): Faster R-CNN MobileNetV3, FCOS and RetinaNet v2 from
  torchvision (BSD-3-Clause), on Falcon drives, dataset versions and real sets (KITTI, COCO, BDD100K, nuScenes).
  Seven pinned weight files, each SHA-256 verified, download only on request. Seeded per iteration: a resumed run
  ends bit for bit like an uninterrupted one on the CPU; on CUDA, FCOS and RetinaNet repeated bit for bit and Faster
  R-CNN does not (RoIAlign), as run.json says. Every ONNX export is checked against torch through core.models.
  Measured on an RTX 5080 at 640 × 384: 97–139 images/s, 1.1–2.8 GB peak.
- The mixed sim/real experiment (`falcon experiment mixed`): real only against real + X % synthetic at a fixed
  budget, K seeds, the same real test set, paired bootstrap against real only plus the seed-to-seed spread.
  Local experiment tracking with an MLflow 2.x file-store export (MLflow itself is not needed).
- A CARLA environment with the Gymnasium API (gymnasium's classes when installed): 27 documented features, an
  optional BEV and camera, continuous or discrete actions, 14 reward terms that sum to the reward, and infractions
  with step, actors and pose. Measured on a real server: 57–101 steps/s with the BEV, about 200 without, and
  identical repeats of kinematic episodes.
- Behaviour cloning from recorded drives (drives now record the ego's control), and PPO with a scenario curriculum,
  sharing one policy format; `falcon policy`.
- Leaderboard-style evaluation (`falcon policy leaderboard`, Training ▸ Leaderboard): route completion × infraction
  penalty with the Leaderboard's coefficients and intervals over routes. One-click replay of any infraction re-runs
  the episode exactly with a chase camera and checks that it meets the same infraction.
- The Training page; the Models page refreshes when shown.
- Fixed: ONNX/TorchScript segmenters could not be evaluated (class maps by name, decoded twice); a kinematic ego's
  recorded speed was 0.
- Honest results: PPO trained for minutes does not drive well (measured and documented), and behaviour cloning on a
  few drives beats a constant baseline only modestly.

## 0.5.0 — milestone 5: models and the sim-to-real gap

- Optical flow as a camera label (CARLA's optical-flow camera decoded as measured on 0.9.16: median error 0.43 / 0.23
  px against the flow predicted from depth and the camera's motion).
- Evaluation metrics (`core.metrics`): COCO AP/AR with attribute breakdowns, nuScenes mAP/NDS, mIoU, CLEAR-MOT and
  HOTA, TuSimple and CULane, calibration; bootstrap and paired bootstrap over whole drives. Equal to pycocotools
  (maximum difference 0.0 over 54 values) and nuscenes-devkit (1.1e-16 over 183).
- The model test harness and the Models page: registration (AGPL refused; weights on request, SHA-256 checked),
  evaluation with intervals, breakdowns, a condition heat map, a failure gallery, paired regression tests;
  `falcon evaluate`, `falcon model fetch`. A synthetic detector (ground truth with planted weaknesses) to test it.
  An evaluation over fewer than five drives says its interval is not reliable.
- The sim-to-real gap: KITTI, nuScenes, BDD100K, COCO and folders read in place; appearance and object statistics;
  Fréchet and kernel distances in a pinned ResNet-50 feature space next to their real-vs-real floor; appearance
  matching with its effect measured; `falcon gap`. Measured, CARLA against KITTI raw: FD 184.8 against a floor of
  23.1; matching brings brightness, contrast, colour and noise most of the way and does not reduce FD (195.3) — the
  gap is content and rendering.
- World-model data (occupancy with lidar visibility, BEV maps, scene flow, future trajectories, ego actions, clips)
  and grounded language data (captions and QA whose every phrase names its CARLA ids; a paraphrase hook that only
  uses a client you pass in); `falcon derive`. On a real server: every product of an 8 s Town10 drive in 26 s.
- Fixed: actors.jsonl logged kinematic scenario actors as standing still (CARLA's snapshot with physics off); the
  language data's lane relations across road ids and inside junctions (found on a real lead_brake run).
- Not yet: mixed sim/real training experiments run in the training lab (milestone 6); learned style transfer is not
  planned.

## 0.4.0 — milestone 4: scenarios and rare cases

- The scenario model (actors, a timeline of triggers and actions, parameters as distributions, seeded sampling) and
  eleven built-ins, four for Indian roads; the director runs them inside drives (they record labelled data).
- Measured: CARLA 0.9.16's vehicle physics does not repeat bit for bit across runs; scenarios default to kinematic
  motion, and all eleven built-ins then repeat bit for bit on a real server.
- Criticality: time to collision, post-encroachment time, minimum distance, required deceleration, collisions, a
  documented scalar; pass criteria per scenario.
- Sweeps and rare-case searches (Bayesian optimisation, CMA-ES, adaptive sampling) as resumable studies; on a planted
  failure region they find five failures in 22–49 evaluations against random's 99.
- OpenSCENARIO 1.x subset import and export (ScenarioRunner is not part of the install).
- The Scenarios and Rare cases pages; falcon sweep and falcon search.
- Fixed: an intermittent segfault in the GUI (a background-task result object freed from a worker thread); Falcon
  processes share Falcon's own server (a registry) instead of refusing a busy port; the lane follower now measures
  gaps bumper to bumper and holds behind a stopped vehicle.

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
