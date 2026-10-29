# Changelog

## 1.4.0 — milestone 12: learned worlds

- **World-model training** (`falcon worldmodel`, Training ▸ World model): a DreamerV3-style RSSM on BEV, state, action
  and reward sequences from drives and episodes, scored per horizon (1–20 steps) against copy-last-frame and
  constant-velocity baselines. Measured honestly: it does **not** beat the baselines on the BEV (actors IoU 0.40 / 0.22
  at 1 / 20 steps against constant velocity's 0.98 / 0.74), but it predicts reward better at every horizon. Policies
  learnt in its imagination reach the highest Leaderboard score on the fake (0.59) by exploiting the model (speeding,
  5–9 collisions in 20 routes); an end-to-end camera policy (PilotNet) drove best (14/20 successes, 1 collision) while
  being no better offline than the mean action. Too little real data to say more (stated).
- **Adversarial traffic** (`falcon adversary`, Rare cases): learned adversaries (parameter-shared independent PPO,
  1→3 agents) that try to make a fixed ego fail, counted only when realistic — no collision they cause (geometric
  fault attribution), on lanes, bounded acceleration and jerk, within their role's rule budget. Against a blind ego:
  0.50 [0.38, 0.62] realistic failures on the fake and 0.40 [0.22, 0.61] on real Town10, against 0.05–0.10 for random
  and scripted baselines; against the lane follower no method found a realistic failure; against a PPO ego the
  learned adversaries were no better than random. Every failure replays exactly.
- **Vision-language explanations** (`falcon vlm`, Models ▸ Language): VLM answers to grounded driving questions scored
  against the simulator's truth — exact answers, numbers within stated tolerance bands, grounding to track ids,
  hallucination (CHAIR-style), explanation faithfulness by rules against the brake facts — with intervals over drives.
  Validated with an oracle (1.000 on every type) and a seeded noisy baseline (0.720, every answer scored as its
  corruption predicts). Runners for local Hugging Face models (offline, pinned commit, publisher checksums, approved
  download only), plug-ins; LoRA fine-tuning in plain torch. Seven Apache-2.0 candidates listed with verified licences;
  non-commercial and remote-code models refused by default.
- The bundle keeps transformers, tokenizers and huggingface_hub (the vision-language runner needs them).

## 1.3.0 — milestone 11: real to sim

- **Sim-to-real translation** (`falcon translate`, Training ▸ Sim-to-real translation): Falcon's own CUT (Park et al.
  2020) with a geometry-preserving generator (no layer moves pixels; a residual on the input; the semantic label as
  input), trained here on drives against real images — no model download. Translated runs keep every label; a drift
  score per frame (structure, edges, object boxes; heuristic threshold calibrated on 1–2 px shifts) refuses frames
  that moved. Measured on the RTX 5080 (12 000 iterations, 19 min, 1.9 GB): FD to KITTI 184.8 → 181.0 (held-out
  drive: 248.4 → 231.3), still 8–12× the real-vs-real floor; brightness and colour move most; the flat imported
  Bengaluru world is refused (the translator invents structure). The CUT default (λ_NCE 1) repaints buildings and is
  refused by the check. The downstream effect is not measured on KITTI (no labelled set on disk). Data terms (KITTI:
  CC BY-NC-SA 3.0) travel with every translator and translated run.
- **Neural reconstruction** (`falcon recon`, Replay page): 3D Gaussian Splatting (Kerbl et al. 2023) with a
  differentiable tile rasteriser in pure PyTorch (no CUDA toolkit or gsplat build on this machine; gradients match
  finite differences to 1e-8): 9–11 steps/s at 640 × 360, 7000 steps in 11–13 min, 2 GB. Measured on a real Town10
  drive: held-out 28.97 dB PSNR / 0.895 SSIM; 27.0 / 25.4 / 23.3 dB at 0.5 / 1 / 2 m off the path; against genuine CARLA
  renders 3.5 m to the side 16.8 dB. KITTI 0005: 20.4 dB held-out. Re-rendered drives carry RGB, range and voted
  semantic labels (no boxes). Static scenes only; CUDA runs bit for bit.

## 1.2.0 — milestone 10: realism

- **Sensor weather physics** (per world, off by default, recorded per run and frame): lidar extinction in fog (Kim et
  al. 2001 at 905 nm) and rain (Carbonneau), speckle-based dropout, fog backscatter and rain-drop returns labelled with
  a new class 30 `weather_noise` (real returns keep their labels); radar rain attenuation (ITU-R P.838-3 at 77 GHz),
  Swerling I detection and rain clutter (an upper bound); camera lens droplets, a wet-lens veil, glare and bloom (CIE
  146), night headlights (heuristic). On 20 real Town10 sweeps the kept share per range bin matches the model at fog
  10–100 and rain 25–50 mm/h. Not calibrated against a real sensor in real weather (stated).
- **Runtime ODD monitoring**: exact condition monitors from the simulator state and an image-statistics monitor
  (Mahalanobis, Ledoit–Wolf), calibrated by leave-one-drive-out cross-fitting to a target false-alarm rate. Measured on
  real Town10 frames: 0/260 false alarms; 78 % of frames flagged at 350 m visibility, 100 % at 140 m and 100 % of
  Bengaluru frames (the fog synthesised from range labels). Live, gym, stack, HIL, the Loop and the safety case show
  it; a workspace's `odd.json` is monitored on every run (outside the drive cache key); `falcon odd`.
- **Indian road assets** within the packaged CARLA 0.9.16 (measured: 214 blueprints, no animal, three-wheeler or
  handcart; `static.prop.mesh` spawns any cooked mesh): autorickshaw (Microlino, by measured size against a Bajaj RE;
  65 km/h and 2.0 m/s² applied through the traffic manager and torque) and cycle-rickshaw stand-ins with a Falcon
  `subclass` label; speed breakers as OpenDRIVE elevation profiles (IRC:99-1988 hump within 1.1 mm RMS at 0.25 m mesh
  spacing; the vehicle rises 9 cm at 15 km/h), placed from OSM tags; potholes refused (CARLA ignores lateral shape);
  waterlogging global only and not drawn on imported worlds (measured); cattle and handcart proxies; four new Indian
  scenarios; the source-build path for real meshes documented, not done. `falcon assets`.
- The Worlds page switches sensor weather physics per world.

## 1.1.0 — milestone 9: a sharper loop

- **Model-in-the-loop rare-case search** (`falcon search --objective perception|both --model M`): each evaluation runs
  the model on the scenario's frames and scores its failure on the actors that matter — recall over the 3 s before the
  critical moment and how late the first detection came (heuristic, defined and tested). A new acquisition rule,
  probability of failure, kept only because it measurably helped on the planted objective (99–102 failures in 120
  evaluations against 51). On the fake: the third failure at 18.1 evaluations against random's 27.7 (10 seeds); on a
  real server (one seed, an illustration) 9 failures in 24 against random's 1. Criticality stays the default; the Loop
  can use either.
- **Why did this fail?** (`falcon why`): counterfactual re-runs of a kinematic case, one factor at a time (occlusion,
  range, lighting, fog, rain, appearance, sensor faults), with necessary/sufficient per factor — "in this scenario, with
  this model: a single-case attribution, not a general cause"; correlates of an evaluation's misses with intervals
  (labelled correlational); a Leaderboard infraction by the policy's input features. From the Models gallery and the
  Rare cases list; a report kind.
- **Active learning** (`falcon experiment active`, Training ▸ Active learning): entropy, margin, detection
  uncertainty, k-centre diversity and a hybrid against random selection, same budget per round, paired intervals and
  seed spread, the area under the AP-vs-frames curve and the frames needed to match random. The Loop's data step can
  select by acquisition. Measured on real drives: no finding yet (too few test drives, no COCO weights on disk), and the
  result says so.
- **Traffic rules in temporal logic** (`falcon rules`): a small formula language (parsed, never evaluated as Python)
  with quantitative semantics and three-valued handling of unknown signals; an Indian rule set (left-hand traffic) and
  a right-hand set, editable; lanes, junctions, speed limits and crossings from lanes.json and the map's OpenDRIVE
  (Town10 lanes within a median 0.4 mm of CARLA's waypoints). Violations beside Leaderboard scores, in stack suites,
  the safety case, reports and the run browser. Not checkable, as not recorded: indicators, horn, right of way at
  unsignalled junctions.

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
